//! Phase 3.3: `SamplerV1` (penalties, temperature, `top_k`, `top_p`, `min_p`, seeded RNG,
//! and structured-output `TokenBitmaskConstraint` with speculative rollback) and
//! `Qwen3Parser` (`vllm/parser/qwen3.py` parity for `<think>...</think>` reasoning and
//! Qwen3-XML / Hermes JSON `<tool_call>` extraction).

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SamplingParams {
    pub temperature: f32,
    pub top_k: i32,
    pub top_p: f32,
    pub min_p: f32,
    pub repetition_penalty: f32,
    pub frequency_penalty: f32,
    pub presence_penalty: f32,
    pub seed: u64,
    pub max_tokens: usize,
    pub stop_token_ids: Vec<i32>,
    pub stop_strings: Vec<String>,
    pub enable_thinking: bool,
}

impl Default for SamplingParams {
    fn default() -> Self {
        Self {
            temperature: 0.0,
            top_k: -1,
            top_p: 1.0,
            min_p: 0.0,
            repetition_penalty: 1.0,
            frequency_penalty: 0.0,
            presence_penalty: 0.0,
            seed: 42,
            max_tokens: 256,
            stop_token_ids: vec![151643, 151645, 248044, 248046],
            stop_strings: Vec::new(),
            enable_thinking: true,
        }
    }
}

pub use crate::constraints::TokenBitmaskConstraint;

/// Transactional `SamplerV1` implementing `vLLM`'s 9-step sampling pipeline with
/// deterministic per-(seed, position) RNG and speculative rollback.
#[derive(Clone, Debug)]
pub struct SamplerV1 {
    pub params: SamplingParams,
    pub prompt_counts: BTreeMap<i32, u32>,
    pub output_counts: BTreeMap<i32, u32>,
    pub output_history: Vec<i32>,
    pub constraint: Option<TokenBitmaskConstraint>,
}

impl SamplerV1 {
    pub fn new(
        params: SamplingParams,
        prompt_tokens: &[i32],
        constraint: Option<TokenBitmaskConstraint>,
    ) -> Self {
        let mut prompt_counts = BTreeMap::new();
        for &t in prompt_tokens {
            *prompt_counts.entry(t).or_insert(0) += 1;
        }
        Self {
            params,
            prompt_counts,
            output_counts: BTreeMap::new(),
            output_history: Vec::new(),
            constraint,
        }
    }

    /// Deterministic SplitMix64 uniform float in `[0.0, 1.0)` keyed by `(seed, step_pos)`.
    /// Stateless: callers may domain-separate RNG lanes using distinct derived seeds.
    /// `step_pos` is the absolute output position, independent of proposer consumption.
    pub fn uniform_f32(seed: u64, step_pos: usize) -> f32 {
        let mut z = seed
            .wrapping_add((step_pos as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15))
            .wrapping_add(0x9E37_79B9_7F4A_7C15);
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^= z >> 31;
        ((z >> 40) as u32 as f32) * (1.0 / ((1u32 << 24) as f32))
    }

    /// Returns normalized dense vocabulary probabilities without advancing RNG, history,
    /// or grammar state. Greedy policy yields a point mass (ties use lowest token ID).
    /// Masked logits may be nonfinite; allowed logits and processed scores must be finite.
    pub fn processed_distribution(&self, logits: &[f32]) -> Result<Vec<f32>, String> {
        let probs = self.processed_candidates(logits)?;
        let sum: f32 = probs.iter().map(|p| p.1).sum();
        let mut dense = vec![0.0; logits.len()];
        for (tid, p) in probs {
            dense[tid as usize] = p / sum;
        }
        Ok(dense)
    }

    // Keep sorted weights (including the pre-CDF nucleus sum) for legacy RNG ordering.
    fn processed_candidates(&self, logits: &[f32]) -> Result<Vec<(i32, f32)>, String> {
        let p = &self.params;
        if !p.temperature.is_finite()
            || p.temperature < 0.0
            || p.top_k < -1
            || !p.top_p.is_finite()
            || p.top_p <= 0.0
            || p.top_p > 1.0
            || !p.min_p.is_finite()
            || !(0.0..=1.0).contains(&p.min_p)
            || !p.repetition_penalty.is_finite()
            || p.repetition_penalty <= 0.0
            || !p.frequency_penalty.is_finite()
            || !p.presence_penalty.is_finite()
        {
            return Err("invalid sampling parameters".into());
        }
        if logits.is_empty() || logits.len() > i32::MAX as usize {
            return Err("empty or oversized vocabulary".into());
        }
        for (tid, &logit) in logits.iter().enumerate() {
            if self
                .constraint
                .as_ref()
                .is_none_or(|c| c.is_token_allowed(tid as i32))
                && !logit.is_finite()
            {
                return Err(format!("nonfinite allowed logit at token {tid}"));
            }
        }
        let mut candidates: Vec<(i32, f32)> = logits
            .iter()
            .enumerate()
            .filter_map(|(idx, &val)| {
                let tid = idx as i32;
                if let Some(ref c) = self.constraint
                    && !c.is_token_allowed(tid)
                {
                    return None;
                }
                let mut l = val;
                let in_prompt = self.prompt_counts.get(&tid).copied().unwrap_or(0) > 0;
                let out_cnt = self.output_counts.get(&tid).copied().unwrap_or(0);
                if (in_prompt || out_cnt > 0) && (self.params.repetition_penalty - 1.0).abs() > 1e-6
                {
                    if l > 0.0 {
                        l /= self.params.repetition_penalty;
                    } else {
                        l *= self.params.repetition_penalty;
                    }
                }
                if out_cnt > 0 {
                    l -= self.params.frequency_penalty * (out_cnt as f32);
                    l -= self.params.presence_penalty;
                }
                Some((tid, l))
            })
            .collect();

        if candidates.is_empty() {
            return Err("grammar has no support in the vocabulary".into());
        }
        if candidates.iter().any(|c| !c.1.is_finite()) {
            return Err("nonfinite penalized logit".into());
        }

        // Greedy fast-path when temperature <= 0 or top_k == 1
        if self.params.temperature <= 1e-6 || self.params.top_k == 1 {
            candidates.sort_by(|a, b| {
                b.1.partial_cmp(&a.1)
                    .unwrap_or(std::cmp::Ordering::Equal)
                    .then_with(|| a.0.cmp(&b.0))
            });
            return Ok(vec![(candidates[0].0, 1.0)]);
        }

        let inv_temp = 1.0 / self.params.temperature;
        for c in &mut candidates {
            c.1 *= inv_temp;
            if !c.1.is_finite() {
                return Err("nonfinite temperature-scaled logit".into());
            }
        }
        candidates.sort_by(|a, b| {
            b.1.partial_cmp(&a.1)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| a.0.cmp(&b.0))
        });

        if self.params.top_k > 0 && (self.params.top_k as usize) < candidates.len() {
            candidates.truncate(self.params.top_k as usize);
        }

        let max_logit = candidates[0].1;
        let mut probs: Vec<(i32, f32)> = candidates
            .iter()
            .map(|&(tid, l)| (tid, (l - max_logit).exp()))
            .collect();
        let sum_p: f32 = probs.iter().map(|x| x.1).sum();
        if sum_p > 0.0 {
            for p in &mut probs {
                p.1 /= sum_p;
            }
        }

        // Apply `min_p` relative to top-1 probability
        if self.params.min_p > 0.0 && !probs.is_empty() {
            let threshold = probs[0].1 * self.params.min_p;
            let keep = probs.iter().take_while(|p| p.1 >= threshold).count().max(1);
            probs.truncate(keep);
            let renorm: f32 = probs.iter().map(|x| x.1).sum();
            if renorm > 0.0 {
                for p in &mut probs {
                    p.1 /= renorm;
                }
            }
        }

        // Apply `top_p` nucleus cutoff
        if self.params.top_p < 1.0 && probs.len() > 1 {
            let mut cum = 0.0f32;
            let mut keep = 0usize;
            for p in &probs {
                cum += p.1;
                keep += 1;
                if cum >= self.params.top_p {
                    break;
                }
            }
            probs.truncate(keep.max(1));
        }

        Ok(probs)
    }

    /// Samples and commits a token using the same policy as `processed_distribution`.
    /// Empty input retains the legacy non-committing return value of zero.
    /// Panics on other invalid inputs before changing transactional state; use
    /// `processed_distribution` for fallible validation.
    pub fn sample_from_logits(&mut self, logits: &[f32]) -> i32 {
        if logits.is_empty() {
            return 0;
        }
        self.try_sample_from_logits(logits)
            .unwrap_or_else(|err| panic!("invalid sampling input: {err}"))
    }

    pub fn try_sample_from_logits(&mut self, logits: &[f32]) -> Result<i32, String> {
        let probs = self.processed_candidates(logits)?;
        let norm_sum: f32 = probs.iter().map(|x| x.1).sum();
        let u = Self::uniform_f32(self.params.seed, self.output_history.len()) * norm_sum;
        let mut acc = 0.0f32;
        let mut chosen = probs.last().unwrap().0;
        for &(tid, p) in &probs {
            acc += p;
            if u <= acc {
                chosen = tid;
                break;
            }
        }
        self.try_commit_token(chosen)?;
        Ok(chosen)
    }

    pub fn commit_token(&mut self, token_id: i32) {
        self.try_commit_token(token_id)
            .expect("invalid sampler token commit");
    }

    pub fn try_commit_token(&mut self, token_id: i32) -> Result<(), String> {
        if let Some(ref mut c) = self.constraint {
            c.advance(token_id)?;
        }
        self.output_history.push(token_id);
        *self.output_counts.entry(token_id).or_insert(0) += 1;
        Ok(())
    }

    /// Rolls back `num_rejected` speculative tokens from `output_history`, `output_counts`,
    /// and `constraint` (`docs/ROADMAP.md` §4 Phase 3.3).
    pub fn rollback_tokens(&mut self, num_rejected: usize) {
        self.try_rollback_tokens(num_rejected)
            .expect("invalid sampler rollback");
    }

    pub fn try_rollback_tokens(&mut self, num_rejected: usize) -> Result<(), String> {
        if num_rejected > self.output_history.len() {
            return Err("sampler rollback exceeds committed tokens".into());
        }
        if let Some(ref mut c) = self.constraint {
            c.rollback(num_rejected)?;
        }
        for _ in 0..num_rejected {
            if let Some(tid) = self.output_history.pop()
                && let Some(cnt) = self.output_counts.get_mut(&tid)
            {
                if *cnt <= 1 {
                    self.output_counts.remove(&tid);
                } else {
                    *cnt -= 1;
                }
            }
        }
        Ok(())
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ParsedFunctionCall {
    pub name: String,
    /// JSON-serialized arguments object string (OpenAI `function.arguments` contract).
    pub arguments: String,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ParsedToolCall {
    pub id: String,
    #[serde(rename = "type")]
    pub call_type: String,
    pub function: ParsedFunctionCall,
}

#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct ParsedAssistantTurn {
    pub reasoning_content: Option<String>,
    pub content: Option<String>,
    pub tool_calls: Vec<ParsedToolCall>,
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct StreamParseDelta {
    pub reasoning_delta: Option<String>,
    pub content_delta: Option<String>,
    pub completed_tool_calls: Vec<ParsedToolCall>,
}

/// Unified Qwen 3.5 / 3.6 / 3.8 reasoning and tool parser (`vllm/parser/qwen3.py` parity).
///
/// Supports:
/// - `enable_thinking: bool` (`<think>...</think>` extraction into `reasoning_content`)
/// - Implicit `</think>` transition when `<tool_call>` appears directly inside reasoning
/// - Duplicate `</think>` absorption in content state
/// - Both **Qwen3-XML** (`<tool_call><function=name><parameter=k>v</parameter></function></tool_call>`)
///   and **Hermes JSON** (`<tool_call>{"name": "...", "arguments": {...}}</tool_call>`) tool calls
#[derive(Clone, Debug)]
pub struct Qwen3Parser {
    pub enable_thinking: bool,
    in_reasoning: bool,
    in_tool_call: bool,
    buffer: String,
    reasoning_accum: String,
    content_accum: String,
    tool_calls: Vec<ParsedToolCall>,
    next_tool_idx: usize,
}

impl Qwen3Parser {
    pub const THINK_START: &'static str = "<think>";
    pub const THINK_END: &'static str = "</think>";
    pub const TOOL_START: &'static str = "<tool_call>";
    pub const TOOL_END: &'static str = "</tool_call>";

    pub fn new(enable_thinking: bool) -> Self {
        Self {
            enable_thinking,
            in_reasoning: enable_thinking,
            in_tool_call: false,
            buffer: String::new(),
            reasoning_accum: String::new(),
            content_accum: String::new(),
            tool_calls: Vec::new(),
            next_tool_idx: 0,
        }
    }

    fn trim_wrapping_newlines(s: &str) -> &str {
        let s = s.strip_prefix('\n').unwrap_or(s);
        s.strip_suffix('\n').unwrap_or(s)
    }

    /// Locate a structural delimiter without treating JSON string contents as
    /// markup. Incomplete JSON is held until another streaming chunk arrives.
    fn find_tool_delimiter(text: &str, delimiter: &str) -> Option<usize> {
        fn skip_json(text: &str, start: usize) -> Result<usize, ()> {
            let rest = &text[start..];
            let trimmed = rest.trim_start();
            let offset = start + rest.len() - trimmed.len();
            if !trimmed.starts_with(['"', '{', '[']) {
                return Ok(start);
            }
            let mut values =
                serde_json::Deserializer::from_str(trimmed).into_iter::<serde_json::Value>();
            match values.next() {
                Some(Ok(_)) => Ok(offset + values.byte_offset()),
                Some(Err(error)) if error.is_eof() => Err(()),
                _ => Ok(start), // Native XML also allows unquoted text values.
            }
        }
        let mut cursor = skip_json(text, 0).ok()?;
        while cursor < text.len() {
            let next = cursor + text[cursor..].find('<')?;
            let rest = &text[next..];
            if rest.starts_with(delimiter) {
                return Some(next);
            }
            if rest.starts_with("<parameter=") {
                let value_start = next + rest.find('>')? + 1;
                cursor = skip_json(text, value_start).ok()?;
            } else {
                cursor = next + 1;
            }
        }
        None
    }

    /// Parses either Qwen3-XML (`<function=name><parameter=k>v</parameter></function>`)
    /// or Hermes JSON (`{"name": "...", "arguments": ...}`) inside a `<tool_call>...</tool_call>` body.
    pub fn parse_tool_body(raw_body: &str, call_idx: usize) -> Option<ParsedToolCall> {
        let body = raw_body.trim();
        if body.is_empty() {
            return None;
        }

        // 1. Check Qwen3-XML format: `<function=func_name>...`
        if let Some(after_prefix) = body.strip_prefix("<function=") {
            let close_angle = after_prefix.find('>')?;
            let fn_name = after_prefix[..close_angle].trim().to_string();
            let rest = &after_prefix[close_angle + 1..];
            let fn_body = match Self::find_tool_delimiter(rest, "</function>") {
                Some(end_idx) => &rest[..end_idx],
                None => rest,
            };

            let mut params = serde_json::Map::new();
            let mut search = fn_body;
            while let Some(p_start) = Self::find_tool_delimiter(search, "<parameter=") {
                let after_p = &search[p_start + "<parameter=".len()..];
                let Some(p_angle) = after_p.find('>') else {
                    break;
                };
                let param_name = after_p[..p_angle].trim().to_string();
                let val_start = &after_p[p_angle + 1..];
                let (raw_val, next_slice) = if let Some(p_end) =
                    Self::find_tool_delimiter(val_start, "</parameter>")
                {
                    (
                        &val_start[..p_end],
                        &val_start[p_end + "</parameter>".len()..],
                    )
                } else if let Some(next_p) = Self::find_tool_delimiter(val_start, "<parameter=") {
                    (&val_start[..next_p], &val_start[next_p..])
                } else {
                    (val_start, "")
                };
                let trimmed = Self::trim_wrapping_newlines(raw_val);
                // Preserve structured JSON numbers/booleans/arrays/objects when valid JSON,
                // or store string value (`_qwen3_arg_converter`).
                let json_val = serde_json::from_str::<serde_json::Value>(trimmed)
                    .unwrap_or_else(|_| serde_json::Value::String(trimmed.to_string()));
                params.insert(param_name, json_val);
                search = next_slice;
            }

            let args_json =
                serde_json::to_string(&serde_json::Value::Object(params)).unwrap_or_default();
            return Some(ParsedToolCall {
                id: format!("call_w34_{call_idx}"),
                call_type: "function".to_string(),
                function: ParsedFunctionCall {
                    name: fn_name,
                    arguments: args_json,
                },
            });
        }

        // 2. Check Hermes JSON format: `{"name": "func_name", "arguments": {...}}`
        if let Ok(val) = serde_json::from_str::<serde_json::Value>(body)
            && let Some(name) = val.get("name").and_then(|v| v.as_str())
        {
            let args_str = match val.get("arguments") {
                Some(serde_json::Value::String(s)) => s.clone(),
                Some(other) => serde_json::to_string(other).unwrap_or_else(|_| "{}".to_string()),
                None => "{}".to_string(),
            };
            return Some(ParsedToolCall {
                id: format!("call_w34_{call_idx}"),
                call_type: "function".to_string(),
                function: ParsedFunctionCall {
                    name: name.to_string(),
                    arguments: args_str,
                },
            });
        }

        None
    }

    fn longest_tag_prefix_suffix(s: &str, tags: &[&str]) -> usize {
        let mut max_keep = 0usize;
        for &tag in tags {
            let max_check = (tag.len() - 1).min(s.len());
            for k in (1..=max_check).rev() {
                if s.ends_with(&tag[..k]) {
                    max_keep = max_keep.max(k);
                    break;
                }
            }
        }
        max_keep
    }

    /// Feeds a streaming text chunk and returns incremental `reasoning_delta`, `content_delta`,
    /// and any newly completed `tool_calls`.
    pub fn push_str(&mut self, chunk: &str) -> StreamParseDelta {
        self.buffer.push_str(chunk);
        let mut delta = StreamParseDelta::default();
        let mut r_out = String::new();
        let mut c_out = String::new();

        loop {
            if self.in_tool_call {
                if let Some(end_pos) = Self::find_tool_delimiter(&self.buffer, Self::TOOL_END) {
                    let body = self.buffer[..end_pos].to_string();
                    self.buffer.drain(..end_pos + Self::TOOL_END.len());
                    self.in_tool_call = false;
                    if let Some(tc) = Self::parse_tool_body(&body, self.next_tool_idx) {
                        self.next_tool_idx += 1;
                        self.tool_calls.push(tc.clone());
                        delta.completed_tool_calls.push(tc);
                    }
                    continue;
                }
                break;
            } else if self.in_reasoning {
                // Absorb leading `<think>` if emitted explicitly
                if let Some(rest) = self.buffer.strip_prefix(Self::THINK_START) {
                    self.buffer = rest.to_string();
                    continue;
                }
                let think_end = self.buffer.find(Self::THINK_END);
                let tool_start = self.buffer.find(Self::TOOL_START);
                match (think_end, tool_start) {
                    (Some(te), Some(ts)) if ts < te => {
                        // Implicit `</think>` on `<tool_call>` directly from reasoning
                        let text = &self.buffer[..ts];
                        r_out.push_str(text);
                        self.reasoning_accum.push_str(text);
                        self.buffer.drain(..ts + Self::TOOL_START.len());
                        self.in_reasoning = false;
                        self.in_tool_call = true;
                        continue;
                    }
                    (Some(te), _) => {
                        let text = &self.buffer[..te];
                        r_out.push_str(text);
                        self.reasoning_accum.push_str(text);
                        self.buffer.drain(..te + Self::THINK_END.len());
                        self.in_reasoning = false;
                        continue;
                    }
                    (None, Some(ts)) => {
                        let text = &self.buffer[..ts];
                        r_out.push_str(text);
                        self.reasoning_accum.push_str(text);
                        self.buffer.drain(..ts + Self::TOOL_START.len());
                        self.in_reasoning = false;
                        self.in_tool_call = true;
                        continue;
                    }
                    (None, None) => {
                        let keep = Self::longest_tag_prefix_suffix(
                            &self.buffer,
                            &[Self::THINK_START, Self::THINK_END, Self::TOOL_START],
                        );
                        let emit_len = self.buffer.len() - keep;
                        if emit_len > 0 {
                            let text = &self.buffer[..emit_len];
                            r_out.push_str(text);
                            self.reasoning_accum.push_str(text);
                            self.buffer.drain(..emit_len);
                        }
                        break;
                    }
                }
            } else {
                // In CONTENT state: check for `<think>` (if thinking wasn't initially active),
                // absorb duplicate `</think>`, or transition to `<tool_call>`.
                let think_start = self.buffer.find(Self::THINK_START);
                let dup_think_end = self.buffer.find(Self::THINK_END);
                let tool_start = self.buffer.find(Self::TOOL_START);

                let earliest = [
                    think_start.map(|p| (p, 0u8)),
                    dup_think_end.map(|p| (p, 1u8)),
                    tool_start.map(|p| (p, 2u8)),
                ]
                .into_iter()
                .flatten()
                .min_by_key(|&(pos, _)| pos);

                if let Some((pos, kind)) = earliest {
                    let text = &self.buffer[..pos];
                    c_out.push_str(text);
                    self.content_accum.push_str(text);
                    match kind {
                        0 => {
                            self.buffer.drain(..pos + Self::THINK_START.len());
                            self.in_reasoning = true;
                        }
                        1 => {
                            // Absorb duplicate `</think>` silently (`vllm/parser/qwen3.py:130-134`)
                            self.buffer.drain(..pos + Self::THINK_END.len());
                        }
                        _ => {
                            self.buffer.drain(..pos + Self::TOOL_START.len());
                            self.in_tool_call = true;
                        }
                    }
                    continue;
                }

                let keep = Self::longest_tag_prefix_suffix(
                    &self.buffer,
                    &[Self::THINK_START, Self::THINK_END, Self::TOOL_START],
                );
                let emit_len = self.buffer.len() - keep;
                if emit_len > 0 {
                    let text = &self.buffer[..emit_len];
                    c_out.push_str(text);
                    self.content_accum.push_str(text);
                    self.buffer.drain(..emit_len);
                }
                break;
            }
        }

        if !r_out.is_empty() {
            delta.reasoning_delta = Some(r_out);
        }
        if !c_out.is_empty() {
            delta.content_delta = Some(c_out);
        }
        delta
    }

    /// Flushes any trailing buffered text at end-of-sequence and returns the final turn.
    pub fn finish(&mut self) -> (StreamParseDelta, ParsedAssistantTurn) {
        let mut delta = StreamParseDelta::default();
        if !self.buffer.is_empty() {
            let rem = std::mem::take(&mut self.buffer);
            if self.in_tool_call {
                // A truncated call is never executable. In particular, a
                // delimiter inside an unfinished JSON string is not a close tag.
                let fallback = format!("{}{rem}", Self::TOOL_START);
                self.content_accum.push_str(&fallback);
                delta.content_delta = Some(fallback);
                self.in_tool_call = false;
            } else if self.in_reasoning {
                self.reasoning_accum.push_str(&rem);
                delta.reasoning_delta = Some(rem);
            } else {
                self.content_accum.push_str(&rem);
                delta.content_delta = Some(rem);
            }
        }

        let trimmed_content = self.content_accum.trim_matches('\n');
        let turn = ParsedAssistantTurn {
            reasoning_content: if self.reasoning_accum.is_empty() {
                None
            } else {
                Some(self.reasoning_accum.clone())
            },
            content: if trimmed_content.is_empty() {
                None
            } else {
                Some(trimmed_content.to_string())
            },
            tool_calls: self.tool_calls.clone(),
        };
        (delta, turn)
    }

    /// One-shot helper for non-streaming completions.
    pub fn parse_complete(text: &str, enable_thinking: bool) -> ParsedAssistantTurn {
        let mut p = Self::new(enable_thinking);
        let _ = p.push_str(text);
        p.finish().1
    }
}
