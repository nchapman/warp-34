//! OpenAI request validation and Axum HTTP/SSE serving.
pub use crate::chatml::{ChatMessage, format_chatml_prompt, format_chatml_prompt_with_options};
#[cfg(feature = "cuda")]
#[path = "server_http.rs"]
mod http;
#[cfg(feature = "cuda")]
pub use http::*;

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
