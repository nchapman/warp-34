use super::*;

pub fn candidate_token_ids(
    tokenizer: &Tokenizer,
    boundary: &str,
    labels: &[u8],
    model_label: &str,
) -> Result<Vec<i32>, DecisionError> {
    let tail_enc = tokenizer
        .encode(boundary, false)
        .map_err(|e| DecisionError::internal(format!("{model_label}: encode boundary: {e}")))?;
    let tail_ids = tail_enc.get_ids();
    let special_ids: std::collections::HashSet<u32> = tokenizer
        .get_added_tokens_decoder()
        .into_iter()
        .filter_map(|(id, tok)| tok.special.then_some(id))
        .collect();

    let mut out = Vec::with_capacity(labels.len());
    for &b in labels {
        let ch = b as char;
        let combined_str = format!("{boundary}{ch}");
        let comb_enc = tokenizer
            .encode(combined_str.as_str(), false)
            .map_err(|e| DecisionError::internal(format!("{model_label}: encode label: {e}")))?;
        let comb_ids = comb_enc.get_ids();
        if comb_ids.len() != tail_ids.len() + 1
            || comb_ids[..tail_ids.len()] != *tail_ids
            || special_ids.contains(&comb_ids[tail_ids.len()])
        {
            return Err(DecisionError::internal(format!(
                "{model_label}: choice code {ch} is not one ordinary token at the answer boundary"
            )));
        }
        out.push(comb_ids[tail_ids.len()] as i32);
    }
    let mut sorted = out.clone();
    sorted.sort_unstable();
    if sorted.windows(2).any(|w| w[0] == w[1]) {
        return Err(DecisionError::internal(format!(
            "{model_label}: choice codes must have distinct token ids"
        )));
    }
    Ok(out)
}

pub fn entropy_confidence(probs: &[f64]) -> f64 {
    if probs.len() < 2 {
        return 1.0;
    }
    let mut entropy = 0.0;
    for &x in probs {
        if x > 0.0 {
            entropy -= x * x.ln();
        }
    }
    (1.0 - entropy / (probs.len() as f64).ln()).clamp(0.0, 1.0)
}

pub fn margin_choice_confidence(probs: &[f64]) -> f64 {
    let k = probs.len();
    if k <= 1 {
        return 1.0;
    }
    let mx = probs.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let inv_k = 1.0 / (k as f64);
    ((mx - inv_k) / (1.0 - inv_k)).clamp(0.0, 1.0)
}

pub fn mode_distance_score_confidence(probs: &[f64]) -> f64 {
    let l = probs.len();
    if l <= 1 {
        return 1.0;
    }
    let mut mode = 0usize;
    for i in 1..l {
        if probs[i] > probs[mode] {
            mode = i;
        }
    }
    let mut s = 0.0;
    for (i, &p) in probs.iter().enumerate() {
        s += p * ((i as f64) - (mode as f64)).abs();
    }
    (1.0 - s / ((l - 1) as f64)).clamp(0.0, 1.0)
}

pub fn softmax_temperature(logits: &[f64], temperature: f64) -> Result<Vec<f64>, DecisionError> {
    if logits.is_empty() {
        return Err(DecisionError::internal("decision: empty candidate logits"));
    }
    if !(temperature > 0.0 && temperature.is_finite()) {
        return Err(DecisionError::internal(
            "decision: temperature must be positive and finite",
        ));
    }
    let mut peak = logits[0];
    for &v in logits {
        if !v.is_finite() {
            return Err(DecisionError::internal(
                "decision: non-finite candidate logit",
            ));
        }
        if v > peak {
            peak = v;
        }
    }
    let mut probs = Vec::with_capacity(logits.len());
    let mut total = 0.0;
    for &v in logits {
        let e = ((v - peak) / temperature).exp();
        probs.push(e);
        total += e;
    }
    for p in &mut probs {
        *p /= total;
    }
    Ok(probs)
}

/// Averages forward `[0..N]` and reverse `[N-1..=0]` probability distributions (`Xor`/`Lev`).
pub fn calibrate_forward_reverse_logits(
    fwd_logits: &[f64],
    rev_logits: &[f64],
    temperature: f64,
) -> Result<Vec<f64>, DecisionError> {
    if fwd_logits.len() != rev_logits.len() {
        return Err(DecisionError::internal(
            "xor: forward/reverse logit count mismatch",
        ));
    }
    let fwd = softmax_temperature(fwd_logits, temperature)?;
    let rev = softmax_temperature(rev_logits, temperature)?;
    let n = fwd.len();
    Ok((0..n).map(|i| 0.5 * (fwd[i] + rev[n - 1 - i])).collect())
}

pub fn answer_from_probabilities(
    field: &DecisionField,
    probs: &[f64],
    conf_kind: ConfidenceKind,
) -> Result<Value, DecisionError> {
    if probs.len() != field.keys.len() || probs.is_empty() {
        return Err(DecisionError::internal(
            "decision: probability count differs from field choices",
        ));
    }
    if field.field_type == "noul" {
        let true_idx = field
            .keys
            .iter()
            .position(|k| k == "true")
            .unwrap_or(probs.len() - 1);
        let mut out = Map::new();
        out.insert("type".into(), Value::String("noul".into()));
        out.insert("noul".into(), json!(probs[true_idx]));
        return Ok(Value::Object(out));
    }

    let mut dist = Map::new();
    for (k, &p) in field.keys.iter().zip(probs) {
        dist.insert(k.clone(), json!(p));
    }

    if field.field_type == "score" {
        let mut score = 0.0;
        let mut legend = Map::new();
        for (i, &p) in probs.iter().enumerate() {
            score += (i as f64) * p;
            legend.insert(
                field.keys[i].clone(),
                Value::String(field.value_descriptions[i].clone()),
            );
        }
        let conf = match conf_kind {
            ConfidenceKind::NormalizedEntropy => entropy_confidence(probs),
            ConfidenceKind::MarginAndModeDistance => mode_distance_score_confidence(probs),
            ConfidenceKind::MaxProbability => probs.iter().copied().fold(0.0, f64::max),
        };
        let mut out = Map::new();
        out.insert("type".into(), Value::String("score".into()));
        out.insert("score".into(), json!(score));
        out.insert("legend".into(), Value::Object(legend));
        out.insert("probabilities".into(), Value::Object(dist));
        out.insert("confidence".into(), json!(conf));
        return Ok(Value::Object(out));
    }

    let mut best = 0usize;
    for i in 1..probs.len() {
        if probs[i] > probs[best] {
            best = i;
        }
    }
    let conf = match conf_kind {
        ConfidenceKind::NormalizedEntropy => entropy_confidence(probs),
        ConfidenceKind::MarginAndModeDistance => margin_choice_confidence(probs),
        ConfidenceKind::MaxProbability => probs[best],
    };
    let mut out = Map::new();
    out.insert("type".into(), Value::String("choice".into()));
    out.insert("choice".into(), Value::String(field.keys[best].clone()));
    out.insert("probabilities".into(), Value::Object(dist));
    out.insert("confidence".into(), json!(conf));
    Ok(Value::Object(out))
}

pub fn answer_from_logits(
    field: &DecisionField,
    logits: &[f64],
    temperature: f64,
    conf_kind: ConfidenceKind,
) -> Result<Value, DecisionError> {
    let probs = softmax_temperature(logits, temperature)?;
    answer_from_probabilities(field, &probs, conf_kind)
}

/// Normalizes an ergonomic `/v1/decisions` request (supporting `context` or `state`, and an
/// array of `decisions` or an object `questions`) into canonical `/v1/systemone` request JSON.
pub fn normalize_decisions_adapter_request(body: &Value) -> Result<Value, DecisionError> {
    let Some(obj) = body.as_object() else {
        return Err(DecisionError::bad_request("request body must be an object"));
    };
    if obj.contains_key("questions") && obj.get("questions").is_some_and(Value::is_object) {
        let mut canonical = Map::new();
        if let Some(m) = obj.get("model") {
            canonical.insert("model".into(), m.clone());
        }
        if let Some(st) = obj.get("state").or_else(|| obj.get("context")) {
            canonical.insert("state".into(), st.clone());
        }
        if let Some(imgs) = obj.get("images") {
            canonical.insert("images".into(), imgs.clone());
        }
        if let Some(vids) = obj.get("videos") {
            canonical.insert("videos".into(), vids.clone());
        }
        canonical.insert("questions".into(), obj["questions"].clone());
        return Ok(Value::Object(canonical));
    }

    let Some(state) = obj.get("state").or_else(|| obj.get("context")) else {
        return Err(DecisionError::bad_request("state (or context) is required"));
    };
    let Some(items) = obj
        .get("decisions")
        .or_else(|| obj.get("questions"))
        .and_then(Value::as_array)
        .filter(|a| !a.is_empty())
    else {
        return Err(DecisionError::bad_request(
            "decisions (or questions) must be a non-empty array or object",
        ));
    };

    let mut questions = Map::new();
    for (idx, item) in items.iter().enumerate() {
        let Some(im) = item.as_object() else {
            return Err(DecisionError::bad_request(format!(
                "decisions[{idx}] must be an object"
            )));
        };
        let default_id = format!("decision_{idx}");
        let id = im
            .get("id")
            .or_else(|| im.get("name"))
            .and_then(Value::as_str)
            .unwrap_or(&default_id);
        if questions.contains_key(id) {
            return Err(DecisionError::bad_request(format!(
                "duplicate decision id '{id}'"
            )));
        }
        let qtype = im.get("type").and_then(Value::as_str).unwrap_or("choice");
        let Some(instr) = im.get("instructions").or_else(|| im.get("question")) else {
            return Err(DecisionError::bad_request(format!(
                "decision '{id}': instructions (or question) is required"
            )));
        };
        let mut q = Map::new();
        q.insert("type".into(), Value::String(qtype.to_owned()));
        q.insert("instructions".into(), instr.clone());
        if let Some(crit) = im.get("criteria").or_else(|| im.get("options")) {
            if qtype == "choice"
                && let Some(arr) = crit.as_array()
            {
                let mut cm = Map::new();
                for opt in arr {
                    let (k, desc) = if let Some(s) = opt.as_str() {
                        (s.to_owned(), s.to_owned())
                    } else if let Some(om) = opt.as_object()
                        && let Some(k) = om
                            .get("key")
                            .or_else(|| om.get("id"))
                            .and_then(Value::as_str)
                    {
                        let desc = om.get("description").and_then(Value::as_str).unwrap_or(k);
                        (k.to_owned(), desc.to_owned())
                    } else {
                        return Err(DecisionError::bad_request(format!(
                            "decision '{id}': invalid choice option entry"
                        )));
                    };
                    if cm.contains_key(&k) {
                        return Err(DecisionError::bad_request(format!(
                            "decision '{id}': duplicate choice option key '{k}'"
                        )));
                    }
                    cm.insert(k, Value::String(desc));
                }
                q.insert("criteria".into(), Value::Object(cm));
            } else {
                q.insert("criteria".into(), crit.clone());
            }
        }
        questions.insert(id.to_owned(), Value::Object(q));
    }

    let mut canonical = Map::new();
    if let Some(m) = obj.get("model") {
        canonical.insert("model".into(), m.clone());
    }
    canonical.insert("state".into(), state.clone());
    if let Some(imgs) = obj.get("images") {
        canonical.insert("images".into(), imgs.clone());
    }
    if let Some(vids) = obj.get("videos") {
        canonical.insert("videos".into(), vids.clone());
    }
    canonical.insert("questions".into(), Value::Object(questions));
    Ok(Value::Object(canonical))
}

pub fn format_decisions_adapter_response(systemone_resp: &Value) -> Value {
    let answers = systemone_resp
        .get("answers")
        .and_then(Value::as_object)
        .cloned()
        .unwrap_or_default();
    let decisions: Vec<Value> = answers
        .iter()
        .map(|(id, ans)| {
            let mut m = Map::new();
            m.insert("id".into(), Value::String(id.clone()));
            if let Some(am) = ans.as_object() {
                for (k, v) in am {
                    m.insert(k.clone(), v.clone());
                }
            }
            Value::Object(m)
        })
        .collect();
    json!({
        "object": "decision.list",
        "model": systemone_resp.get("model").cloned().unwrap_or(Value::Null),
        "decisions": decisions,
        "answers": Value::Object(answers),
        "usage": systemone_resp.get("usage").cloned().unwrap_or(Value::Null),
    })
}
