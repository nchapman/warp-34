use super::*;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ClefEncodedQuestion {
    pub question_id: String,
    pub question_type: usize,
    pub question_span: (usize, usize),
    pub option_spans: Vec<(usize, usize)>,
    pub option_ids: Vec<String>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ClefEncodedRecord {
    pub input_ids: Vec<i32>,
    pub questions: Vec<ClefEncodedQuestion>,
}

#[derive(Clone, Copy, Debug)]
pub struct ClefMediaTokens {
    pub vision_start_id: i32,
    pub image_pad_id: i32,
    pub vision_end_id: i32,
}

/// Clef `render(value)` (`joint_schema_model.py:37-45`):
/// returns strings verbatim and serializes non-strings as compact UTF-8 JSON
/// (`separators=(",", ":")`, `sort_keys=True`, `ensure_ascii=False`).
pub fn clef_render_value(v: &Value) -> Result<String, DecisionError> {
    if let Some(s) = v.as_str() {
        return Ok(s.to_owned());
    }
    let mut out = String::new();
    clef_dump_compact_sorted(v, &mut out)?;
    Ok(out)
}

pub(super) fn clef_dump_compact_sorted(v: &Value, out: &mut String) -> Result<(), DecisionError> {
    match v {
        Value::Array(arr) => {
            out.push('[');
            for (i, item) in arr.iter().enumerate() {
                if i > 0 {
                    out.push(',');
                }
                clef_dump_compact_sorted(item, out)?;
            }
            out.push(']');
        }
        Value::Object(obj) => {
            let mut entries: Vec<(&String, &Value)> = obj.iter().collect();
            entries.sort_by(|a, b| a.0.cmp(b.0));
            out.push('{');
            for (i, (k, val)) in entries.into_iter().enumerate() {
                if i > 0 {
                    out.push(',');
                }
                out.push_str(
                    &serde_json::to_string(k)
                        .map_err(|e| DecisionError::bad_request(e.to_string()))?,
                );
                out.push(':');
                clef_dump_compact_sorted(val, out)?;
            }
            out.push('}');
        }
        Value::Number(n) => {
            out.push_str(&format_python_json_number(n));
        }
        _ => {
            out.push_str(
                &serde_json::to_string(v).map_err(|e| DecisionError::bad_request(e.to_string()))?,
            );
        }
    }
    Ok(())
}

/// Formats a `serde_json::Number` identically to Python `json.dumps(...)`:
/// integers render verbatim; floats use shortest round-trip decimal (`ryu`) in fixed-point
/// when `f == 0.0` or `-4 <= exp < 16`, and `Xe+YY` / `Xe-YY` (explicit sign, `>= 2` exponent
/// digits) outside `1e-4..1e16`.
pub(crate) fn format_python_json_number(n: &serde_json::Number) -> String {
    if n.is_i64() || n.is_u64() {
        return n.to_string();
    }
    let raw = n.to_string();
    let (neg, unsigned) = match raw.strip_prefix('-') {
        Some(rest) => (true, rest),
        None => (false, raw.as_str()),
    };
    let sign_prefix = if neg { "-" } else { "" };
    let (mant, exp_part) = match unsigned.split_once(['e', 'E']) {
        Some((m, e)) => (m, e.parse::<i32>().unwrap_or(0)),
        None => (unsigned, 0),
    };
    let (int_str, frac_str) = match mant.split_once('.') {
        Some((i, f)) => (i, f),
        None => (mant, ""),
    };
    let mut digits = String::with_capacity(int_str.len() + frac_str.len());
    digits.push_str(int_str);
    digits.push_str(frac_str);
    let trimmed_leading = digits.trim_start_matches('0');
    if trimmed_leading.is_empty() {
        return format!("{sign_prefix}0.0");
    }
    let leading_zeros = digits.len() - trimmed_leading.len();
    let decpt = (int_str.len() as i32) - (leading_zeros as i32) + exp_part;
    let sig = trimmed_leading.trim_end_matches('0');
    let exp = decpt - 1;
    if (-4..16).contains(&exp) {
        if decpt <= 0 {
            format!("{sign_prefix}0.{}{sig}", "0".repeat((-decpt) as usize))
        } else if (decpt as usize) >= sig.len() {
            format!(
                "{sign_prefix}{sig}{}.0",
                "0".repeat((decpt as usize) - sig.len())
            )
        } else {
            let split = decpt as usize;
            format!("{sign_prefix}{}.{}", &sig[..split], &sig[split..])
        }
    } else {
        let mantissa = if sig.len() == 1 {
            sig.to_owned()
        } else {
            format!("{}.{}", &sig[..1], &sig[1..])
        };
        let esign = if exp >= 0 { '+' } else { '-' };
        format!("{sign_prefix}{mantissa}e{esign}{:02}", exp.unsigned_abs())
    }
}

pub(super) fn collect_clef_added_control_tokens(tokenizer: &Tokenizer) -> Vec<String> {
    let mut toks: Vec<String> = tokenizer
        .get_added_tokens_decoder()
        .into_values()
        .map(|t| t.content)
        .filter(|s| !s.is_empty() && !s.contains("<|") && !s.contains("|>"))
        .collect();
    toks.sort_by(|a, b| b.len().cmp(&a.len()).then_with(|| a.cmp(b)));
    toks.dedup();
    toks
}

pub(super) fn sanitize_clef_untrusted(text: &str, added_tokens: &[String]) -> String {
    let mut clean = kev_sanitize_special_tokens(text);
    for tok in added_tokens {
        if clean.contains(tok.as_str()) {
            let neutralized = if tok.contains('<') || tok.contains('>') {
                tok.replace('<', "<\u{00a6}").replace('>', "\u{00a6}>")
            } else {
                let mut ch = tok.chars();
                match ch.next() {
                    Some(first) => format!("{first}\u{200b}{}", ch.as_str()),
                    None => String::new(),
                }
            };
            if neutralized != *tok {
                clean = clean.replace(tok.as_str(), &neutralized);
            }
        }
    }
    clean
}

pub(super) fn slice_utf8_prefix(s: &str, max_bytes: usize) -> &str {
    if s.len() <= max_bytes {
        return s;
    }
    let mut end = max_bytes;
    while end > 0 && !s.is_char_boundary(end) {
        end -= 1;
    }
    &s[..end]
}

pub(super) fn encode_plain(tokenizer: &Tokenizer, text: &str) -> Result<Vec<i32>, DecisionError> {
    tokenizer
        .encode(text, false)
        .map(|enc| enc.get_ids().iter().map(|&id| id as i32).collect())
        .map_err(|e| DecisionError::internal(format!("clef tokenize: {e}")))
}

pub(super) fn encode_untrusted(
    tokenizer: &Tokenizer,
    text: &str,
    added_tokens: &[String],
) -> Result<Vec<i32>, DecisionError> {
    let clean = sanitize_clef_untrusted(text, added_tokens);
    encode_plain(tokenizer, &clean)
}

/// Port of `joint_schema_model.py` `encode_record`, preserving exact token IDs,
/// `question_span`, and `option_spans` across text, JSON, and multimodal image records.
pub fn clef_encode_record(
    tokenizer: &Tokenizer,
    body: &Value,
    max_length: usize,
    media: Option<(ClefMediaTokens, &[usize])>,
) -> Result<ClefEncodedRecord, DecisionError> {
    let obj = body
        .as_object()
        .ok_or_else(|| DecisionError::bad_request("request body must be an object"))?;
    let state_val = obj
        .get("state")
        .ok_or_else(|| DecisionError::bad_request("state is required"))?;
    let q_obj = obj
        .get("questions")
        .and_then(Value::as_object)
        .filter(|m| !m.is_empty())
        .ok_or_else(|| DecisionError::bad_request("questions must be a non-empty object"))?;

    let added_control_tokens = collect_clef_added_control_tokens(tokenizer);
    let mut schema_ids = encode_plain(tokenizer, "\n\nSCHEMA FIELDS:\n")?;
    let mut questions = Vec::with_capacity(q_obj.len());

    for (q_idx, (question_id, question_val)) in q_obj.iter().enumerate() {
        let qm = question_val
            .as_object()
            .ok_or_else(|| DecisionError::bad_request("question must be an object"))?;
        let q_type_str = qm
            .get("type")
            .and_then(Value::as_str)
            .ok_or_else(|| DecisionError::bad_request("question missing string type"))?;
        let q_type_id = match q_type_str {
            "noul" => 0usize,
            "choice" => 1usize,
            "score" => 2usize,
            other => {
                return Err(DecisionError::bad_request(format!(
                    "question '{question_id}' has unknown type: {other}"
                )));
            }
        };

        let safe_qid = sanitize_clef_untrusted(question_id, &added_control_tokens);
        let field_header = format!(
            "\nFIELD {}\nID: {safe_qid}\nTYPE: {q_type_str}\nINSTRUCTION: ",
            q_idx + 1
        );
        schema_ids.extend(encode_plain(tokenizer, &field_header)?);

        let question_start = schema_ids.len();
        let instr_rendered = match qm.get("instructions") {
            Some(v) if !v.is_null() && v.as_str() != Some("") => clef_render_value(v)?,
            _ => question_id.clone(),
        };
        let instr_ids = encode_untrusted(tokenizer, &instr_rendered, &added_control_tokens)?;
        if instr_ids.is_empty() {
            return Err(DecisionError::bad_request(format!(
                "question '{question_id}': instructions produced zero tokens"
            )));
        }
        schema_ids.extend(instr_ids);
        let question_end = schema_ids.len();
        schema_ids.extend(encode_plain(tokenizer, "\nALLOWED OPTIONS:\n")?);

        let options: Vec<(String, Option<Value>)> = match q_type_str {
            "noul" => {
                let mut t_desc = Value::String(CLEF_NOUL_TRUE_DEFAULT.to_owned());
                let mut f_desc = Value::String(CLEF_NOUL_FALSE_DEFAULT.to_owned());
                if let Some(cm) = qm.get("criteria").and_then(Value::as_object) {
                    if let Some(tv) = cm.get("true").filter(|v| !v.is_null()) {
                        t_desc = tv.clone();
                    }
                    if let Some(fv) = cm.get("false").filter(|v| !v.is_null()) {
                        f_desc = fv.clone();
                    }
                }
                vec![
                    ("true".to_owned(), Some(t_desc)),
                    ("false".to_owned(), Some(f_desc)),
                ]
            }
            "choice" => {
                let cm = qm
                    .get("criteria")
                    .and_then(Value::as_object)
                    .ok_or_else(|| {
                        DecisionError::bad_request(format!(
                            "question '{question_id}': choice requires a criteria object"
                        ))
                    })?;
                let mut pairs: Vec<(String, Option<Value>)> = cm
                    .iter()
                    .map(|(k, v)| (k.clone(), (!v.is_null()).then(|| v.clone())))
                    .collect();
                pairs.sort_by(|a, b| a.0.cmp(&b.0));
                pairs
            }
            "score" => {
                let ca = qm
                    .get("criteria")
                    .and_then(Value::as_array)
                    .ok_or_else(|| {
                        DecisionError::bad_request(format!(
                            "question '{question_id}': score requires a criteria array"
                        ))
                    })?;
                ca.iter()
                    .enumerate()
                    .map(|(idx, v)| (idx.to_string(), (!v.is_null()).then(|| v.clone())))
                    .collect()
            }
            _ => unreachable!(),
        };

        let mut option_spans = Vec::with_capacity(options.len());
        let mut option_ids = Vec::with_capacity(options.len());
        for (opt_idx, (opt_id, desc_opt)) in options.into_iter().enumerate() {
            schema_ids.extend(encode_plain(
                tokenizer,
                &format!("OPTION {}: ", opt_idx + 1),
            )?);
            let option_start = schema_ids.len();
            let mut sem = Map::new();
            sem.insert("option_id".to_owned(), Value::String(opt_id.clone()));
            if let Some(d) = desc_opt {
                sem.insert("description".to_owned(), d);
            }
            let sem_str = clef_render_value(&Value::Object(sem))?;
            let opt_toks = encode_untrusted(tokenizer, &sem_str, &added_control_tokens)?;
            if opt_toks.is_empty() {
                return Err(DecisionError::bad_request(format!(
                    "question '{question_id}': option '{opt_id}' produced zero tokens"
                )));
            }
            schema_ids.extend(opt_toks);
            let option_end = schema_ids.len();
            option_spans.push((option_start, option_end));
            option_ids.push(opt_id);
            schema_ids.extend(encode_plain(tokenizer, "\n")?);
        }
        schema_ids.extend(encode_plain(tokenizer, "END FIELD\n")?);
        questions.push(ClefEncodedQuestion {
            question_id: question_id.clone(),
            question_type: q_type_id,
            question_span: (question_start, question_end),
            option_spans,
            option_ids,
        });
    }

    let mut prefix_ids = encode_plain(
        tokenizer,
        &format!("<|im_start|>system\n{CLEF_SYSTEM_PROMPT}<|im_end|>\n<|im_start|>user\nSTATE:\n"),
    )?;
    if let Some((mtoks, counts)) = media
        && !counts.is_empty()
    {
        for &n_merged in counts {
            prefix_ids.push(mtoks.vision_start_id);
            prefix_ids.resize(prefix_ids.len() + n_merged, mtoks.image_pad_id);
            prefix_ids.push(mtoks.vision_end_id);
        }
        prefix_ids.extend(encode_plain(tokenizer, "\n")?);
    }
    let suffix_ids = encode_plain(
        tokenizer,
        "\n<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\nJOINT SCHEMA DECISIONS:",
    )?;

    let fixed_length = prefix_ids.len() + schema_ids.len() + suffix_ids.len();
    if fixed_length >= max_length {
        return Err(DecisionError::bad_request(format!(
            "schema requires {fixed_length} tokens before state; maximum is {max_length}"
        )));
    }
    let state_budget = max_length - fixed_length;
    let state_rendered = clef_render_value(state_val)?;
    // Bound UTF-8 input length before BPE tokenization (Qwen3.5 max single-token UTF-8 span is
    // 256 bytes, so `state_budget * 256 + 256` is lossless for the kept `state_budget` prefix).
    let max_state_bytes = state_budget.saturating_mul(256).saturating_add(256);
    let state_slice = slice_utf8_prefix(&state_rendered, max_state_bytes);
    let mut state_ids = encode_untrusted(tokenizer, state_slice, &added_control_tokens)?;
    if state_ids.is_empty() {
        return Err(DecisionError::bad_request("state produced zero tokens"));
    }
    state_ids.truncate(state_budget);

    let schema_offset = prefix_ids.len() + state_ids.len();
    for q in &mut questions {
        q.question_span.0 += schema_offset;
        q.question_span.1 += schema_offset;
        for sp in &mut q.option_spans {
            sp.0 += schema_offset;
            sp.1 += schema_offset;
        }
    }

    let mut input_ids = Vec::with_capacity(schema_offset + schema_ids.len() + suffix_ids.len());
    input_ids.extend(prefix_ids);
    input_ids.extend(state_ids);
    input_ids.extend(schema_ids);
    input_ids.extend(suffix_ids);

    Ok(ClefEncodedRecord {
        input_ids,
        questions,
    })
}

/// Maps logits produced in `question.option_ids` order (which sorts `choice` keys alphabetically)
/// back onto `field.keys` order (caller's `criteria` key order).
pub fn reorder_clef_question_logits(
    question: &ClefEncodedQuestion,
    field: &DecisionField,
    raw_logits: &[f64],
) -> Result<Vec<f64>, DecisionError> {
    if raw_logits.len() != question.option_ids.len() || raw_logits.len() != field.keys.len() {
        return Err(DecisionError::internal(format!(
            "clef: option count mismatch for '{}'",
            field.name
        )));
    }
    if question.option_ids == field.keys {
        return Ok(raw_logits.to_vec());
    }
    let mut out = Vec::with_capacity(field.keys.len());
    for key in &field.keys {
        let idx = question
            .option_ids
            .iter()
            .position(|oid| oid == key)
            .ok_or_else(|| {
                DecisionError::internal(format!(
                    "clef: option '{key}' missing from encoded question '{}'",
                    field.name
                ))
            })?;
        out.push(raw_logits[idx]);
    }
    Ok(out)
}
