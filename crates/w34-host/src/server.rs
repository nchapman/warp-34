//! OpenAI request validation and Axum HTTP/SSE serving.
pub use crate::chatml::{ChatMessage, format_chatml_prompt, format_chatml_prompt_with_options};
#[cfg(feature = "cuda")]
#[path = "server_http.rs"]
mod http;
#[cfg(feature = "cuda")]
pub use http::*;

/// Endpoint-specific wire validation; chat retains its boolean logprobs contract.
pub fn completion_sampling_params(
    req: &serde_json::Value,
    max_tokens: usize,
) -> Result<crate::parser::SamplingParams, String> {
    if !req.is_object() {
        return Err("completion request must be a JSON object".into());
    }
    let mut normalized = req.clone();
    let count = match req.get("logprobs") {
        None | Some(serde_json::Value::Null) => None,
        Some(value) => Some(
            value
                .as_u64()
                .filter(|&n| n <= w34_kernels::ffi::W34_MAX_TOP_LOGPROBS as u64)
                .ok_or_else(|| {
                    format!(
                        "logprobs must be an integer in 0..={}",
                        w34_kernels::ffi::W34_MAX_TOP_LOGPROBS
                    )
                })?,
        ),
    };
    // top_logprobs is a chat-only option, not an override for the integer count.
    if req.get("top_logprobs").is_some_and(|v| !v.is_null()) {
        return Err("top_logprobs is only supported for chat completions".into());
    }
    normalized["logprobs"] = serde_json::json!(count.is_some());
    normalized["top_logprobs"] = serde_json::json!(count.unwrap_or(0));
    native_sampling_params(&normalized, max_tokens)
}

/// Convert token records to the legacy parallel-array wire shape. Offsets are
/// Unicode character positions in generated text (the prompt is not echoed).
pub fn completion_logprobs(logs: &[serde_json::Value], offset: &mut usize) -> serde_json::Value {
    use serde_json::{Value, json};
    let mut tokens = Vec::new();
    let mut probabilities = Vec::new();
    let mut alternatives = Vec::new();
    let mut offsets = Vec::new();
    for log in logs {
        let token = log["token"].as_str().unwrap_or_default();
        // Serving records carry positions from the response's incremental decoder.
        // Single-token decoding can contain replacement characters for byte fragments.
        if let Some(position) = log["_text_offset"].as_u64() {
            offsets.push(position as usize);
            *offset = position as usize;
        } else {
            // Compatibility for callers supplying already-decoded token records.
            offsets.push(*offset);
            *offset += token.chars().count();
        }
        tokens.push(token);
        probabilities.push(log["logprob"].clone());
        let mut top = serde_json::Map::new();
        if let Some(entries) = log["top_logprobs"].as_array() {
            for entry in entries {
                if let Some(token) = entry["token"].as_str() {
                    top.insert(token.to_owned(), entry["logprob"].clone());
                }
            }
        }
        alternatives.push(Value::Object(top));
    }
    json!({"tokens":tokens,"token_logprobs":probabilities,"top_logprobs":alternatives,"text_offset":offsets})
}

pub fn native_sampling_params(
    req: &serde_json::Value,
    max_tokens: usize,
) -> Result<crate::parser::SamplingParams, String> {
    use crate::parser::SamplingParams;
    if max_tokens == 0 || max_tokens > 4096 {
        return Err("max_tokens must be in 1..=4096".into());
    }
    if req
        .get("stop_token_ids")
        .and_then(serde_json::Value::as_array)
        .is_some_and(|ids| ids.len() > 256)
        || req
            .get("logit_bias")
            .and_then(serde_json::Value::as_object)
            .is_some_and(|bias| bias.len() > 256)
    {
        return Err("stop_token_ids and logit_bias are limited to 256 entries each".into());
    }
    let defaults = SamplingParams::default();
    let number = |name: &str, default: f32| -> Result<f32, String> {
        req.get(name).map_or(Ok(default), |v| {
            v.as_f64()
                .map(|x| x as f32)
                .ok_or_else(|| format!("invalid {name}"))
        })
    };
    let boolean = |name: &str| -> Result<bool, String> {
        req.get(name).map_or(Ok(false), |v| {
            v.as_bool().ok_or_else(|| format!("invalid {name}"))
        })
    };
    let unsigned = |name: &str, default: u64| -> Result<u64, String> {
        req.get(name).map_or(Ok(default), |v| {
            v.as_u64().ok_or_else(|| format!("invalid {name}"))
        })
    };
    let stop_strings = match req.get("stop") {
        None | Some(serde_json::Value::Null) => Vec::new(),
        Some(serde_json::Value::String(s)) => vec![s.clone()],
        Some(serde_json::Value::Array(a)) if a.len() <= 4 => a
            .iter()
            .map(|s| {
                s.as_str()
                    .map(str::to_owned)
                    .ok_or("invalid stop string".to_string())
            })
            .collect::<Result<Vec<_>, _>>()?,
        _ => return Err("stop must be a string or up to four strings".into()),
    };
    if stop_strings.iter().any(|s| s.is_empty() || s.len() > 4096) {
        return Err("stop strings must contain 1..4096 bytes".into());
    }
    let stop_token_ids = match req.get("stop_token_ids") {
        None | Some(serde_json::Value::Null) => Vec::new(),
        Some(v) => v
            .as_array()
            .ok_or("invalid stop_token_ids")?
            .iter()
            .map(|v| {
                v.as_u64()
                    .and_then(|i| i32::try_from(i).ok())
                    .ok_or("invalid stop token".to_string())
            })
            .collect::<Result<Vec<_>, _>>()?,
    };
    let mut logit_bias = std::collections::BTreeMap::new();
    if let Some(v) = req.get("logit_bias").filter(|v| !v.is_null()) {
        for (key, value) in v.as_object().ok_or("invalid logit_bias")? {
            logit_bias.insert(
                key.parse::<i32>().map_err(|_| "invalid logit_bias token")?,
                value.as_f64().ok_or("invalid logit_bias value")? as f32,
            );
        }
    }
    let params = SamplingParams {
        temperature: number("temperature", defaults.temperature)?,
        top_k: req.get("top_k").map_or(Ok(defaults.top_k), |v| {
            v.as_i64()
                .and_then(|x| i32::try_from(x).ok())
                .ok_or("invalid top_k")
        })?,
        top_p: number("top_p", defaults.top_p)?,
        min_p: number("min_p", defaults.min_p)?,
        repetition_penalty: number("repetition_penalty", defaults.repetition_penalty)?,
        frequency_penalty: number("frequency_penalty", defaults.frequency_penalty)?,
        presence_penalty: number("presence_penalty", defaults.presence_penalty)?,
        seed: unsigned("seed", defaults.seed)?,
        max_tokens,
        min_tokens: usize::try_from(unsigned("min_tokens", 0)?)
            .map_err(|_| "invalid min_tokens")?,
        ignore_eos: boolean("ignore_eos")?,
        logprobs: boolean("logprobs")?,
        top_logprobs: usize::try_from(unsigned("top_logprobs", 0)?)
            .map_err(|_| "invalid top_logprobs")?,
        logit_bias,
        stop_strings,
        stop_token_ids,
        ..defaults
    };
    params.validate()?;
    if params.min_tokens > max_tokens {
        return Err("min_tokens exceeds max_tokens".into());
    }
    if params.top_logprobs > 0 && !params.logprobs {
        return Err("top_logprobs requires logprobs=true".into());
    }
    Ok(params)
}

pub fn parse_openai_messages_multimodal(
    messages: &serde_json::Value,
) -> Result<(Vec<ChatMessage>, Vec<crate::vision::PreprocessedImage>), String> {
    parse_openai_messages_with_image_policy(messages, false)
}

pub fn parse_openai_messages_with_image_policy(
    messages: &serde_json::Value,
    allow_remote_images: bool,
) -> Result<(Vec<ChatMessage>, Vec<crate::vision::PreprocessedImage>), String> {
    let arr = messages.as_array().ok_or("'messages' must be an array")?;
    if arr.is_empty() {
        return Err("'messages' must not be empty".into());
    }
    let image_count = arr
        .iter()
        .filter_map(|m| m.get("content").and_then(serde_json::Value::as_array))
        .flatten()
        .filter(|p| p.get("type").and_then(serde_json::Value::as_str) == Some("image_url"))
        .count();
    if image_count > crate::vision::MAX_IMAGES_PER_REQUEST {
        return Err(format!(
            "at most {} images are allowed per request",
            crate::vision::MAX_IMAGES_PER_REQUEST
        ));
    }
    let deadline = std::time::Instant::now() + crate::remote_images::REMOTE_IMAGE_TIMEOUT;
    let mut out = Vec::with_capacity(arr.len());
    let mut images = Vec::new();
    for (index, item) in arr.iter().enumerate() {
        let role = item
            .get("role")
            .and_then(|v| v.as_str())
            .ok_or("message missing string 'role'")?;
        if !matches!(role, "system" | "user" | "assistant" | "tool") {
            return Err(format!("unsupported message role '{role}'"));
        }
        if role == "system" && index != 0 {
            return Err("system message must be at the beginning".into());
        }
        let tool_calls = crate::chatml::parse_history_tools(item.get("tool_calls"))?;
        if role != "assistant" && !tool_calls.is_empty() {
            return Err("tool_calls are only valid on assistant messages".into());
        }
        let reasoning_content = item
            .get("reasoning_content")
            .and_then(|v| v.as_str())
            .map(str::to_owned);
        let content = match item.get("content") {
            None | Some(serde_json::Value::Null) => None,
            Some(serde_json::Value::String(s)) => Some(s.clone()),
            Some(serde_json::Value::Array(parts)) => {
                let mut text = String::new();
                for part in parts {
                    match part.get("type").and_then(|v| v.as_str()).unwrap_or("text") {
                        "text" => {
                            if let Some(s) = part.get("text").and_then(|v| v.as_str()) {
                                text.push_str(s);
                            }
                        }
                        "image_url" => {
                            if role == "system" {
                                return Err("system message cannot contain images".into());
                            }
                            let url = part
                                .get("image_url")
                                .and_then(|v| {
                                    v.as_str().or_else(|| v.get("url").and_then(|v| v.as_str()))
                                })
                                .ok_or("image_url part missing 'url'")?;
                            let (rgb, h, w) = crate::vision::parse_image_url_with_deadline(
                                url,
                                allow_remote_images,
                                deadline,
                            )
                            .map_err(|e| format!("invalid image_url: {e}"))?;
                            crate::vision::validate_image_aspect_ratio(h, w)
                                .map_err(|e| format!("invalid image_url: {e}"))?;
                            let prep = crate::vision::preprocess_rgb_image(
                                &rgb,
                                h,
                                w,
                                16,
                                2,
                                2,
                                32 * 32,
                                1024 * 1024,
                            );
                            text.push_str("<|vision_start|>");
                            for _ in 0..prep.num_merged_tokens(2) {
                                text.push_str("<|image_pad|>");
                            }
                            text.push_str("<|vision_end|>");
                            images.push(prep);
                        }
                        other => {
                            return Err(format!("unsupported message content part type '{other}'"));
                        }
                    }
                }
                Some(text)
            }
            _ => return Err("message 'content' must be string, array, or null".into()),
        };
        out.push(ChatMessage {
            role: role.into(),
            content,
            reasoning_content,
            tool_calls,
        });
    }
    if !out.iter().any(|m| {
        let c = m.content.as_deref().unwrap_or("").trim();
        m.role == "user" && !(c.starts_with("<tool_response>") && c.ends_with("</tool_response>"))
    }) {
        return Err("no user query found in messages".into());
    }
    Ok((out, images))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_completion_sampling_params_validation() {
        let valid_req = serde_json::json!({
            "temperature": 0.7,
            "max_tokens": 100,
            "logprobs": 5
        });
        let params = completion_sampling_params(&valid_req, 100).unwrap();
        assert_eq!(params.temperature, 0.7);
        assert!(params.logprobs);
        assert_eq!(params.top_logprobs, 5);

        // Invalid: non-object
        assert!(completion_sampling_params(&serde_json::json!("not an object"), 100).is_err());

        // Invalid: logprobs exceeds W34_MAX_TOP_LOGPROBS
        let oversized = serde_json::json!({
            "logprobs": w34_kernels::ffi::W34_MAX_TOP_LOGPROBS + 1
        });
        assert!(completion_sampling_params(&oversized, 100).is_err());

        // Invalid: top_logprobs specified in legacy completion endpoint
        let chat_opt = serde_json::json!({
            "logprobs": 2,
            "top_logprobs": 2
        });
        assert!(completion_sampling_params(&chat_opt, 100).is_err());
    }

    #[test]
    fn test_completion_logprobs_wire_format() {
        let logs = vec![
            serde_json::json!({
                "token": "Hello",
                "logprob": -0.5,
                "top_logprobs": [{"token": "Hello", "logprob": -0.5}, {"token": "Hi", "logprob": -1.2}]
            }),
            serde_json::json!({
                "token": " world",
                "logprob": -0.1,
                "top_logprobs": [{"token": " world", "logprob": -0.1}]
            }),
        ];

        let mut offset = 0;
        let result = completion_logprobs(&logs, &mut offset);

        assert_eq!(result["tokens"], serde_json::json!(["Hello", " world"]));
        assert_eq!(result["token_logprobs"], serde_json::json!([-0.5, -0.1]));
        assert_eq!(result["text_offset"], serde_json::json!([0, 5]));
        assert_eq!(offset, 11); // "Hello world" has length 11
    }

    #[test]
    fn test_native_sampling_params_bounds() {
        // max_tokens = 0 must fail
        assert!(native_sampling_params(&serde_json::json!({}), 0).is_err());
        // max_tokens > 4096 must fail
        assert!(native_sampling_params(&serde_json::json!({}), 5000).is_err());

        // Valid request
        let req = serde_json::json!({
            "temperature": 0.8,
            "top_p": 0.95,
            "seed": 42
        });
        let params = native_sampling_params(&req, 128).unwrap();
        assert_eq!(params.temperature, 0.8);
        assert_eq!(params.top_p, 0.95);
        assert_eq!(params.seed, 42);
    }
}
