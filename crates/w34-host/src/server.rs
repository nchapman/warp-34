//! Phase 3.3: OpenAI-compatible HTTP/1.1 + SSE Server (`w34 serve`) and ChatML formatter.
//!
//! Endpoints:
//! - `POST /v1/chat/completions` (non-streaming JSON & streaming SSE with `delta.reasoning_content`,
//!   `delta.content`, `delta.tool_calls`, and `usage.prompt_tokens_details.cached_tokens`)
//! - `POST /v1/completions` (non-streaming JSON & streaming SSE)
//! - `GET /v1/models`
//! - `POST /tokenize` & `POST /detokenize`
//! - `GET /health`
//! - `GET /metrics` (Prometheus exposition format)

#[cfg(feature = "cuda")]
use std::io::{BufRead, BufReader, Read, Write};
#[cfg(feature = "cuda")]
use std::net::{TcpListener, TcpStream};
use std::sync::atomic::AtomicU64;
#[cfg(feature = "cuda")]
use std::sync::atomic::{AtomicBool, Ordering};
#[cfg(feature = "cuda")]
use std::sync::{Arc, Mutex};

pub use crate::chatml::{ChatMessage, format_chatml_prompt, format_chatml_prompt_with_options};
#[cfg(feature = "cuda")]
use serde_json::json;

#[cfg(feature = "cuda")]
use crate::model::Qwen3_5Engine;
#[cfg(feature = "cuda")]
use crate::parser::Qwen3Parser;

/// Parse the sampling subset supported by native speculative requests. Validate
/// before GPU dispatch; do not silently accept malformed numeric parameters.
pub fn native_sampling_params(
    req: &serde_json::Value,
    max_tokens: usize,
) -> Result<crate::parser::SamplingParams, String> {
    use crate::parser::{SamplerV1, SamplingParams};
    let number = |name: &str, default: f32| -> Result<f32, String> {
        match req.get(name) {
            None => Ok(default),
            Some(v) => v
                .as_f64()
                .map(|x| x as f32)
                .ok_or_else(|| format!("invalid {name}")),
        }
    };
    let defaults = SamplingParams::default();
    let top_k = match req.get("top_k") {
        None => defaults.top_k,
        Some(v) => v
            .as_i64()
            .and_then(|x| i32::try_from(x).ok())
            .ok_or("invalid top_k")?,
    };
    let seed = match req.get("seed") {
        None => defaults.seed,
        Some(v) => v.as_u64().ok_or("invalid seed")?,
    };
    if req
        .get("stop")
        .is_some_and(|v| !v.is_null() && v.as_array().is_none_or(|a| !a.is_empty()))
    {
        return Err("native speculative sampling does not yet support stop strings".into());
    }
    let params = SamplingParams {
        temperature: number("temperature", defaults.temperature)?,
        top_k,
        top_p: number("top_p", defaults.top_p)?,
        min_p: number("min_p", defaults.min_p)?,
        repetition_penalty: number("repetition_penalty", defaults.repetition_penalty)?,
        frequency_penalty: number("frequency_penalty", defaults.frequency_penalty)?,
        presence_penalty: number("presence_penalty", defaults.presence_penalty)?,
        seed,
        max_tokens,
        ..defaults
    };
    SamplerV1::new(params.clone(), &[], None).processed_distribution(&[0.0])?;
    Ok(params)
}

/// Parses OpenAI `messages` array supporting both string `content` and multimodal
/// array `content` parts (`{"type": "text", "text": "..."}` and
/// `{"type": "image_url", "image_url": {"url": "..."}}`).
/// Each `image_url` part is preprocessed into a [`crate::vision::PreprocessedImage`] and
/// expanded inline into `<|vision_start|><|image_pad|>...<|vision_end|>`.
pub fn parse_openai_messages_multimodal(
    messages_val: &serde_json::Value,
) -> Result<(Vec<ChatMessage>, Vec<crate::vision::PreprocessedImage>), String> {
    parse_openai_messages_with_image_policy(messages_val, false)
}

pub fn parse_openai_messages_with_image_policy(
    messages_val: &serde_json::Value,
    allow_remote_images: bool,
) -> Result<(Vec<ChatMessage>, Vec<crate::vision::PreprocessedImage>), String> {
    let arr = messages_val
        .as_array()
        .ok_or_else(|| "'messages' must be an array".to_string())?;
    if arr.is_empty() {
        return Err("'messages' must not be empty".to_string());
    }
    let image_count = arr
        .iter()
        .filter_map(|message| message.get("content").and_then(serde_json::Value::as_array))
        .flatten()
        .filter(|part| part.get("type").and_then(serde_json::Value::as_str) == Some("image_url"))
        .count();
    if image_count > crate::vision::MAX_IMAGES_PER_REQUEST {
        return Err(format!(
            "at most {} images are allowed per request",
            crate::vision::MAX_IMAGES_PER_REQUEST
        ));
    }
    let image_deadline = std::time::Instant::now() + crate::remote_images::REMOTE_IMAGE_TIMEOUT;
    let mut out_msgs = Vec::with_capacity(arr.len());
    let mut out_images = Vec::new();
    for (index, item) in arr.iter().enumerate() {
        let role = item
            .get("role")
            .and_then(|v| v.as_str())
            .ok_or_else(|| "message missing string 'role'".to_string())?
            .to_string();
        if !matches!(role.as_str(), "system" | "user" | "assistant" | "tool") {
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
            .map(ToOwned::to_owned);
        let content_opt = match item.get("content") {
            None | Some(serde_json::Value::Null) => None,
            Some(serde_json::Value::String(s)) => Some(s.clone()),
            Some(serde_json::Value::Array(parts)) => {
                let mut combined = String::new();
                for part in parts {
                    let ptype = part.get("type").and_then(|v| v.as_str()).unwrap_or("text");
                    match ptype {
                        "text" => {
                            if let Some(t) = part.get("text").and_then(|v| v.as_str()) {
                                combined.push_str(t);
                            }
                        }
                        "image_url" => {
                            if role == "system" {
                                return Err("system message cannot contain images".into());
                            }
                            let url = part
                                .get("image_url")
                                .and_then(|iu| {
                                    iu.as_str()
                                        .or_else(|| iu.get("url").and_then(|u| u.as_str()))
                                })
                                .ok_or_else(|| "image_url part missing 'url'".to_string())?;
                            let (rgb, h, w) = crate::vision::parse_image_url_with_deadline(
                                url,
                                allow_remote_images,
                                image_deadline,
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
                            let num_vis = prep.num_merged_tokens(2);
                            combined.push_str("<|vision_start|>");
                            for _ in 0..num_vis {
                                combined.push_str("<|image_pad|>");
                            }
                            combined.push_str("<|vision_end|>");
                            out_images.push(prep);
                        }
                        other => {
                            return Err(format!("unsupported message content part type '{other}'"));
                        }
                    }
                }
                Some(combined)
            }
            Some(_) => {
                return Err("message 'content' must be string, array, or null".to_string());
            }
        };
        out_msgs.push(ChatMessage {
            role,
            content: content_opt,
            reasoning_content,
            tool_calls,
        });
    }
    if !out_msgs.iter().any(|m| {
        let content = m.content.as_deref().unwrap_or("").trim();
        m.role == "user"
            && !(content.starts_with("<tool_response>") && content.ends_with("</tool_response>"))
    }) {
        return Err("no user query found in messages".into());
    }
    Ok((out_msgs, out_images))
}

pub struct ServerMetrics {
    pub requests_total: AtomicU64,
    pub tokens_generated_total: AtomicU64,
    pub prompt_tokens_total: AtomicU64,
    pub cached_prompt_tokens_total: AtomicU64,
    pub last_ttft_us: AtomicU64,
    pub last_itl_us: AtomicU64,
}

impl Default for ServerMetrics {
    fn default() -> Self {
        Self {
            requests_total: AtomicU64::new(0),
            tokens_generated_total: AtomicU64::new(0),
            prompt_tokens_total: AtomicU64::new(0),
            cached_prompt_tokens_total: AtomicU64::new(0),
            last_ttft_us: AtomicU64::new(0),
            last_itl_us: AtomicU64::new(0),
        }
    }
}

#[cfg(feature = "cuda")]
pub struct OpenAiServerState {
    pub model_name: String,
    pub engine: Mutex<Qwen3_5Engine>,
    pub metrics: ServerMetrics,
    pub shutdown: AtomicBool,
    pub allow_remote_images: bool,
    speculative_enabled: bool,
    constraint_factory: Option<crate::constraints::ConstraintFactory>,
}

#[cfg(feature = "cuda")]
impl OpenAiServerState {
    /// Parses and compiles request constraints outside `self.engine` lock so
    /// client schemas/grammars never stall GPU inference or other endpoints.
    fn request_constraint(
        &self,
        request: &serde_json::Value,
        thinking: bool,
        multimodal: bool,
    ) -> Result<Option<crate::parser::TokenBitmaskConstraint>, String> {
        let Some(spec) = crate::request_constraints::RequestConstraint::parse(request)? else {
            return Ok(None);
        };
        if multimodal || !self.speculative_enabled {
            if !spec.is_mandatory() {
                return Ok(None);
            }
            return Err(
                "structured outputs currently require text-only native MTP or DFlash2 generation"
                    .into(),
            );
        }
        let Some(factory) = self.constraint_factory.as_ref() else {
            if !spec.is_mandatory() {
                return Ok(None);
            }
            return Err("tokenizer unavailable for structured outputs".into());
        };
        match spec.compile(factory, thinking) {
            Ok(constraint) => Ok(Some(constraint)),
            Err(_) if !spec.is_mandatory() => Ok(None),
            Err(err) => Err(err),
        }
    }

    pub fn new(model_name: String, engine: Qwen3_5Engine) -> Arc<Self> {
        Self::with_image_policy(model_name, engine, false)
    }

    pub fn with_image_policy(
        model_name: String,
        engine: Qwen3_5Engine,
        allow_remote_images: bool,
    ) -> Arc<Self> {
        let speculative_enabled = engine.speculative.is_some();
        let constraint_factory = engine.tokenizer.as_ref().and_then(|tokenizer| {
            let eos = engine
                .cfg
                .eos_token_ids
                .iter()
                .map(|&id| id as u32)
                .collect::<Vec<_>>();
            crate::constraints::ConstraintFactory::new(
                tokenizer.clone(),
                engine.cfg.vocab_size,
                &eos,
            )
            .ok()
        });
        Arc::new(Self {
            allow_remote_images,
            speculative_enabled,
            constraint_factory,
            model_name,
            engine: Mutex::new(engine),
            metrics: ServerMetrics::default(),
            shutdown: AtomicBool::new(false),
        })
    }
}

#[cfg(feature = "cuda")]
fn write_json_response(
    stream: &mut TcpStream,
    status_code: u16,
    status_text: &str,
    body: &serde_json::Value,
) -> std::io::Result<()> {
    let payload = serde_json::to_vec(body).unwrap_or_default();
    write!(
        stream,
        "HTTP/1.1 {status_code} {status_text}\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
        payload.len()
    )?;
    stream.write_all(&payload)?;
    stream.flush()
}

#[cfg(feature = "cuda")]
fn write_text_response(
    stream: &mut TcpStream,
    status_code: u16,
    status_text: &str,
    content_type: &str,
    body: &str,
) -> std::io::Result<()> {
    let payload = body.as_bytes();
    write!(
        stream,
        "HTTP/1.1 {status_code} {status_text}\r\nContent-Type: {content_type}\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
        payload.len()
    )?;
    stream.write_all(payload)?;
    stream.flush()
}

#[cfg(feature = "cuda")]
fn read_http_request(stream: &mut TcpStream) -> std::io::Result<(String, String, Vec<u8>)> {
    const MAX_HEADER_LINES: usize = 64;
    const MAX_HEADER_LINE_BYTES: usize = 16 * 1024;
    const MAX_BODY_BYTES: usize = 8 * 1024 * 1024;

    let _ = stream.set_read_timeout(Some(std::time::Duration::from_secs(30)));
    let _ = stream.set_write_timeout(Some(std::time::Duration::from_secs(30)));

    let mut reader = BufReader::new(stream.try_clone()?);
    let mut request_line = String::new();
    if reader
        .by_ref()
        .take(MAX_HEADER_LINE_BYTES as u64)
        .read_line(&mut request_line)?
        == 0
    {
        return Ok((String::new(), String::new(), Vec::new()));
    }
    let parts: Vec<&str> = request_line.split_whitespace().collect();
    let method = parts.first().copied().unwrap_or("").to_string();
    let path = parts.get(1).copied().unwrap_or("").to_string();

    let mut content_length = 0usize;
    let mut header_count = 0usize;
    loop {
        if header_count >= MAX_HEADER_LINES {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                "too many HTTP headers",
            ));
        }
        let mut line = String::new();
        let n = reader
            .by_ref()
            .take(MAX_HEADER_LINE_BYTES as u64)
            .read_line(&mut line)?;
        if n == 0 || line == "\r\n" || line == "\n" {
            break;
        }
        header_count += 1;
        if let Some((k, v)) = line.split_once(':')
            && k.trim().eq_ignore_ascii_case("content-length")
            && let Ok(len) = v.trim().parse::<usize>()
        {
            if len > MAX_BODY_BYTES {
                return Err(std::io::Error::new(
                    std::io::ErrorKind::InvalidData,
                    "HTTP Content-Length exceeds 8 MiB limit",
                ));
            }
            content_length = len;
        }
    }

    let mut body = vec![0u8; content_length];
    if content_length > 0 {
        reader.read_exact(&mut body)?;
    }
    Ok((method, path, body))
}

#[cfg(feature = "cuda")]
pub fn handle_http_connection(mut stream: TcpStream, state: &OpenAiServerState) {
    let Ok((method, path, body)) = read_http_request(&mut stream) else {
        return;
    };
    if method.is_empty() {
        return;
    }

    match (method.as_str(), path.as_str()) {
        ("GET", "/health") => {
            let _ = write_json_response(&mut stream, 200, "OK", &json!({ "status": "ok" }));
        }
        ("GET", "/v1/models") => {
            let resp = json!({
                "object": "list",
                "data": [{
                    "id": state.model_name,
                    "object": "model",
                    "created": 1730000000,
                    "owned_by": "warp-34"
                }]
            });
            let _ = write_json_response(&mut stream, 200, "OK", &resp);
        }
        ("GET", "/metrics") => {
            let (evict_pages, prot_pages, evict_mamba, prot_mamba, hit_rate) = {
                let eng = state.engine.lock().unwrap();
                (
                    eng.radix.full_evictable_pages(),
                    eng.radix.full_protected_pages(),
                    eng.radix.mamba_evictable_slots(),
                    eng.radix.mamba_protected_slots(),
                    eng.radix.hit_rate(),
                )
            };
            let reqs = state.metrics.requests_total.load(Ordering::Relaxed);
            let gen_toks = state.metrics.tokens_generated_total.load(Ordering::Relaxed);
            let prompt_toks = state.metrics.prompt_tokens_total.load(Ordering::Relaxed);
            let cached_toks = state
                .metrics
                .cached_prompt_tokens_total
                .load(Ordering::Relaxed);
            let ttft_ms = (state.metrics.last_ttft_us.load(Ordering::Relaxed) as f64) / 1000.0;
            let itl_ms = (state.metrics.last_itl_us.load(Ordering::Relaxed) as f64) / 1000.0;
            let prom = format!(
                "# TYPE w34_requests_total counter\n\
                 w34_requests_total {reqs}\n\
                 # TYPE w34_tokens_generated_total counter\n\
                 w34_tokens_generated_total {gen_toks}\n\
                 # TYPE w34_prompt_tokens_total counter\n\
                 w34_prompt_tokens_total {prompt_toks}\n\
                 # TYPE w34_cached_prompt_tokens_total counter\n\
                 w34_cached_prompt_tokens_total {cached_toks}\n\
                 # TYPE w34_radix_kv_hit_ratio gauge\n\
                 w34_radix_kv_hit_ratio {hit_rate:.6}\n\
                 # TYPE w34_radix_evictable_pages gauge\n\
                 w34_radix_evictable_pages {evict_pages}\n\
                 # TYPE w34_radix_protected_pages gauge\n\
                 w34_radix_protected_pages {prot_pages}\n\
                 # TYPE w34_radix_evictable_mamba_slots gauge\n\
                 w34_radix_evictable_mamba_slots {evict_mamba}\n\
                 # TYPE w34_radix_protected_mamba_slots gauge\n\
                 w34_radix_protected_mamba_slots {prot_mamba}\n\
                 # TYPE w34_ttft_ms gauge\n\
                 w34_ttft_ms {ttft_ms:.3}\n\
                 # TYPE w34_itl_ms gauge\n\
                 w34_itl_ms {itl_ms:.3}\n"
            );
            let _ = write_text_response(&mut stream, 200, "OK", "text/plain; version=0.0.4", &prom);
        }
        ("POST", "/tokenize") => {
            let Ok(req_json) = serde_json::from_slice::<serde_json::Value>(&body) else {
                let _ = write_json_response(
                    &mut stream,
                    400,
                    "Bad Request",
                    &json!({"error": "invalid json"}),
                );
                return;
            };
            let Some(prompt) = req_json.get("prompt").and_then(|v| v.as_str()) else {
                let _ = write_json_response(
                    &mut stream,
                    400,
                    "Bad Request",
                    &json!({"error": "missing or non-string 'prompt' field"}),
                );
                return;
            };
            let tokens = {
                let eng = state.engine.lock().unwrap();
                match eng.encode(prompt) {
                    Ok(t) => t,
                    Err(e) => {
                        let _ = write_json_response(
                            &mut stream,
                            500,
                            "Internal Server Error",
                            &json!({"error": e.to_string()}),
                        );
                        return;
                    }
                }
            };
            let _ = write_json_response(
                &mut stream,
                200,
                "OK",
                &json!({
                    "tokens": tokens,
                    "count": tokens.len(),
                }),
            );
        }
        ("POST", "/detokenize") => {
            let Ok(req_json) = serde_json::from_slice::<serde_json::Value>(&body) else {
                let _ = write_json_response(
                    &mut stream,
                    400,
                    "Bad Request",
                    &json!({"error": "invalid json"}),
                );
                return;
            };
            let Some(arr) = req_json.get("tokens").and_then(|v| v.as_array()) else {
                let _ = write_json_response(
                    &mut stream,
                    400,
                    "Bad Request",
                    &json!({"error": "missing or non-array 'tokens' field"}),
                );
                return;
            };
            let mut ids = Vec::with_capacity(arr.len());
            for x in arr {
                let Some(n) = x.as_i64().and_then(|v| i32::try_from(v).ok()) else {
                    let _ = write_json_response(
                        &mut stream,
                        400,
                        "Bad Request",
                        &json!({"error": "invalid token id in 'tokens'"}),
                    );
                    return;
                };
                ids.push(n);
            }
            let text = {
                let eng = state.engine.lock().unwrap();
                match eng.decode(&ids) {
                    Ok(t) => t,
                    Err(e) => {
                        let _ = write_json_response(
                            &mut stream,
                            500,
                            "Internal Server Error",
                            &json!({"error": e.to_string()}),
                        );
                        return;
                    }
                }
            };
            let _ = write_json_response(&mut stream, 200, "OK", &json!({ "prompt": text }));
        }
        ("POST", "/v1/completions") => {
            let Ok(req_json) = serde_json::from_slice::<serde_json::Value>(&body) else {
                let _ = write_json_response(
                    &mut stream,
                    400,
                    "Bad Request",
                    &json!({"error": "invalid json"}),
                );
                return;
            };
            let Some(prompt) = req_json.get("prompt").and_then(|v| v.as_str()) else {
                let _ = write_json_response(
                    &mut stream,
                    400,
                    "Bad Request",
                    &json!({"error": "missing or non-string 'prompt' field"}),
                );
                return;
            };
            let max_tokens = req_json
                .get("max_tokens")
                .and_then(|v| v.as_u64())
                .unwrap_or(64)
                .clamp(1, 4096) as usize;
            let is_stream = req_json
                .get("stream")
                .and_then(|v| v.as_bool())
                .unwrap_or(false);

            let constraint = match state.request_constraint(&req_json, false, false) {
                Ok(value) => value,
                Err(error) => {
                    let _ = write_json_response(
                        &mut stream,
                        400,
                        "Bad Request",
                        &json!({"error": error}),
                    );
                    return;
                }
            };
            let (metrics, text, hit_eos) = {
                let mut eng = state.engine.lock().unwrap();
                let prompt_ids = match eng.encode(prompt) {
                    Ok(ids) => ids,
                    Err(e) => {
                        let _ = write_json_response(
                            &mut stream,
                            500,
                            "Internal Server Error",
                            &json!({"error": e.to_string()}),
                        );
                        return;
                    }
                };
                let generation = if eng.speculative.is_some() {
                    let params = match native_sampling_params(&req_json, max_tokens) {
                        Ok(p) => p,
                        Err(e) => {
                            let _ = write_json_response(
                                &mut stream,
                                400,
                                "Bad Request",
                                &json!({"error":e}),
                            );
                            return;
                        }
                    };
                    eng.generate_sampled_with_radix_cache(&prompt_ids, params, constraint, 64)
                } else {
                    eng.generate_with_radix_cache(&prompt_ids, max_tokens, 64, true)
                };
                let m = match generation {
                    Ok(m) => m,
                    Err(e) => {
                        let _ = write_json_response(
                            &mut stream,
                            500,
                            "Internal Server Error",
                            &json!({"error": e.to_string()}),
                        );
                        return;
                    }
                };
                let last_is_eos = m
                    .generated_token_ids
                    .last()
                    .is_some_and(|&id| eng.is_eos_token(id));
                let t = eng.decode(&m.generated_token_ids).unwrap_or_default();
                (m, t, last_is_eos)
            };
            let finish_reason = if !hit_eos && metrics.gen_len >= max_tokens {
                "length"
            } else {
                "stop"
            };

            let req_num = state.metrics.requests_total.fetch_add(1, Ordering::Relaxed) + 1;
            state
                .metrics
                .tokens_generated_total
                .fetch_add(metrics.gen_len as u64, Ordering::Relaxed);
            state
                .metrics
                .prompt_tokens_total
                .fetch_add(metrics.prompt_len as u64, Ordering::Relaxed);
            state
                .metrics
                .cached_prompt_tokens_total
                .fetch_add(metrics.cached_prefix_tokens as u64, Ordering::Relaxed);
            state
                .metrics
                .last_ttft_us
                .store((metrics.ttft_ms * 1000.0) as u64, Ordering::Relaxed);
            state
                .metrics
                .last_itl_us
                .store((metrics.itl_ms * 1000.0) as u64, Ordering::Relaxed);

            if is_stream {
                let _ = write!(
                    stream,
                    "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nCache-Control: no-cache\r\nConnection: close\r\n\r\n"
                );
                let chunk = json!({
                    "id": format!("cmpl-w34-{req_num}"),
                    "object": "text_completion",
                    "model": state.model_name,
                    "choices": [{
                        "index": 0,
                        "text": text,
                        "finish_reason": finish_reason
                    }]
                });
                let _ = writeln!(stream, "data: {}\n", serde_json::to_string(&chunk).unwrap());
                let _ = writeln!(stream, "data: [DONE]\n");
                let _ = stream.flush();
            } else {
                let resp = json!({
                    "id": format!("cmpl-w34-{req_num}"),
                    "object": "text_completion",
                    "model": state.model_name,
                    "choices": [{
                        "index": 0,
                        "text": text,
                        "finish_reason": finish_reason
                    }],
                    "usage": {
                        "prompt_tokens": metrics.prompt_len,
                        "completion_tokens": metrics.gen_len,
                        "total_tokens": metrics.prompt_len + metrics.gen_len,
                        "prompt_tokens_details": {
                            "cached_tokens": metrics.cached_prefix_tokens
                        }
                    }
                });
                let _ = write_json_response(&mut stream, 200, "OK", &resp);
            }
        }
        ("POST", "/v1/chat/completions") => {
            let Ok(req_json) = serde_json::from_slice::<serde_json::Value>(&body) else {
                let _ = write_json_response(
                    &mut stream,
                    400,
                    "Bad Request",
                    &json!({"error": "invalid json"}),
                );
                return;
            };
            let (messages, mm_images) = match req_json
                .get("messages")
                .ok_or_else(|| "missing 'messages' field".to_string())
                .and_then(|messages| {
                    parse_openai_messages_with_image_policy(messages, state.allow_remote_images)
                }) {
                Ok(v) => v,
                Err(msg) => {
                    let _ = write_json_response(
                        &mut stream,
                        400,
                        "Bad Request",
                        &json!({"error": msg}),
                    );
                    return;
                }
            };
            let enable_thinking = req_json
                .get("chat_template_kwargs")
                .and_then(|v| v.get("enable_thinking"))
                .and_then(|v| v.as_bool())
                .unwrap_or(true);
            let max_tokens = req_json
                .get("max_tokens")
                .or_else(|| req_json.get("max_completion_tokens"))
                .and_then(|v| v.as_u64())
                .unwrap_or(64)
                .clamp(1, 4096) as usize;
            let is_stream = req_json
                .get("stream")
                .and_then(|v| v.as_bool())
                .unwrap_or(false);

            let preserve_thinking = req_json
                .get("chat_template_kwargs")
                .and_then(|v| v.get("preserve_thinking"))
                .and_then(|v| v.as_bool())
                .unwrap_or(false);
            let prompt = format_chatml_prompt_with_options(
                &messages,
                req_json.get("tools"),
                enable_thinking,
                preserve_thinking,
            );
            let constraint =
                match state.request_constraint(&req_json, enable_thinking, !mm_images.is_empty()) {
                    Ok(value) => value,
                    Err(error) => {
                        let _ = write_json_response(
                            &mut stream,
                            400,
                            "Bad Request",
                            &json!({"error": error}),
                        );
                        return;
                    }
                };
            let (metrics, raw_text, hit_eos) = {
                let mut eng = state.engine.lock().unwrap();
                let prompt_ids = match eng.encode(&prompt) {
                    Ok(ids) => ids,
                    Err(e) => {
                        let _ = write_json_response(
                            &mut stream,
                            500,
                            "Internal Server Error",
                            &json!({"error": e.to_string()}),
                        );
                        return;
                    }
                };
                let generation = if !mm_images.is_empty() {
                    eng.generate_multimodal_with_radix_cache(
                        &prompt_ids,
                        &mm_images,
                        max_tokens,
                        64,
                        true,
                    )
                } else if eng.speculative.is_some() {
                    let params = match native_sampling_params(&req_json, max_tokens) {
                        Ok(p) => p,
                        Err(e) => {
                            let _ = write_json_response(
                                &mut stream,
                                400,
                                "Bad Request",
                                &json!({"error":e}),
                            );
                            return;
                        }
                    };
                    eng.generate_sampled_with_radix_cache(&prompt_ids, params, constraint, 64)
                } else {
                    eng.generate_with_radix_cache(&prompt_ids, max_tokens, 64, true)
                };
                let m = match generation {
                    Ok(m) => m,
                    Err(e) => {
                        let _ = write_json_response(
                            &mut stream,
                            500,
                            "Internal Server Error",
                            &json!({"error": e.to_string()}),
                        );
                        return;
                    }
                };
                let last_is_eos = m
                    .generated_token_ids
                    .last()
                    .is_some_and(|&id| eng.is_eos_token(id));
                let t = eng.decode(&m.generated_token_ids).unwrap_or_default();
                (m, t, last_is_eos)
            };

            let req_num = state.metrics.requests_total.fetch_add(1, Ordering::Relaxed) + 1;
            state
                .metrics
                .tokens_generated_total
                .fetch_add(metrics.gen_len as u64, Ordering::Relaxed);
            state
                .metrics
                .prompt_tokens_total
                .fetch_add(metrics.prompt_len as u64, Ordering::Relaxed);
            state
                .metrics
                .cached_prompt_tokens_total
                .fetch_add(metrics.cached_prefix_tokens as u64, Ordering::Relaxed);
            state
                .metrics
                .last_ttft_us
                .store((metrics.ttft_ms * 1000.0) as u64, Ordering::Relaxed);
            state
                .metrics
                .last_itl_us
                .store((metrics.itl_ms * 1000.0) as u64, Ordering::Relaxed);

            let parsed = Qwen3Parser::parse_complete(&raw_text, enable_thinking);
            let finish_reason = if !parsed.tool_calls.is_empty() {
                "tool_calls"
            } else if !hit_eos && metrics.gen_len >= max_tokens {
                "length"
            } else {
                "stop"
            };

            if is_stream {
                let _ = write!(
                    stream,
                    "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nCache-Control: no-cache\r\nConnection: close\r\n\r\n"
                );
                if let Some(ref r_text) = parsed.reasoning_content {
                    let chunk = json!({
                        "id": format!("chatcmpl-w34-{req_num}"),
                        "object": "chat.completion.chunk",
                        "model": state.model_name,
                        "choices": [{
                            "index": 0,
                            "delta": {
                                "role": "assistant",
                                "reasoning_content": r_text
                            },
                            "finish_reason": serde_json::Value::Null
                        }]
                    });
                    let _ = writeln!(stream, "data: {}\n", serde_json::to_string(&chunk).unwrap());
                }
                if let Some(ref c_text) = parsed.content {
                    let chunk = json!({
                        "id": format!("chatcmpl-w34-{req_num}"),
                        "object": "chat.completion.chunk",
                        "model": state.model_name,
                        "choices": [{
                            "index": 0,
                            "delta": {
                                "content": c_text
                            },
                            "finish_reason": serde_json::Value::Null
                        }]
                    });
                    let _ = writeln!(stream, "data: {}\n", serde_json::to_string(&chunk).unwrap());
                }
                if !parsed.tool_calls.is_empty() {
                    let chunk = json!({
                        "id": format!("chatcmpl-w34-{req_num}"),
                        "object": "chat.completion.chunk",
                        "model": state.model_name,
                        "choices": [{
                            "index": 0,
                            "delta": {
                                "tool_calls": parsed.tool_calls
                            },
                            "finish_reason": serde_json::Value::Null
                        }]
                    });
                    let _ = writeln!(stream, "data: {}\n", serde_json::to_string(&chunk).unwrap());
                }
                let done_chunk = json!({
                    "id": format!("chatcmpl-w34-{req_num}"),
                    "object": "chat.completion.chunk",
                    "model": state.model_name,
                    "choices": [{
                        "index": 0,
                        "delta": {},
                        "finish_reason": finish_reason
                    }],
                    "usage": {
                        "prompt_tokens": metrics.prompt_len,
                        "completion_tokens": metrics.gen_len,
                        "total_tokens": metrics.prompt_len + metrics.gen_len,
                        "prompt_tokens_details": {
                            "cached_tokens": metrics.cached_prefix_tokens
                        }
                    }
                });
                let _ = writeln!(
                    stream,
                    "data: {}\n",
                    serde_json::to_string(&done_chunk).unwrap()
                );
                let _ = writeln!(stream, "data: [DONE]\n");
                let _ = stream.flush();
            } else {
                let resp = json!({
                    "id": format!("chatcmpl-w34-{req_num}"),
                    "object": "chat.completion",
                    "model": state.model_name,
                    "choices": [{
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "reasoning_content": parsed.reasoning_content,
                            "content": parsed.content,
                            "tool_calls": parsed.tool_calls,
                        },
                        "finish_reason": finish_reason
                    }],
                    "usage": {
                        "prompt_tokens": metrics.prompt_len,
                        "completion_tokens": metrics.gen_len,
                        "total_tokens": metrics.prompt_len + metrics.gen_len,
                        "prompt_tokens_details": {
                            "cached_tokens": metrics.cached_prefix_tokens
                        }
                    }
                });
                let _ = write_json_response(&mut stream, 200, "OK", &resp);
            }
        }
        _ => {
            let _ = write_json_response(
                &mut stream,
                404,
                "Not Found",
                &json!({"error": "endpoint not found"}),
            );
        }
    }
}

#[cfg(feature = "cuda")]
pub fn run_blocking_server(addr: &str, state: Arc<OpenAiServerState>) -> std::io::Result<()> {
    let listener = TcpListener::bind(addr)?;
    serve_connections(listener, &state.shutdown, |stream| {
        handle_http_connection(stream, &state)
    })
}

#[cfg(all(test, feature = "cuda"))]
#[path = "server_connection_tests.rs"]
mod connection_tests;

// This bounds network/image work independently of the serialized GPU engine.
// The Phase 6 scheduler worker will replace GPU serialization, not this ingress pool.
#[cfg(feature = "cuda")]
fn serve_connections(
    listener: TcpListener,
    shutdown: &AtomicBool,
    handler: impl Fn(TcpStream) + Sync,
) -> std::io::Result<()> {
    serve_incoming(listener.incoming(), shutdown, handler)
}

#[cfg(feature = "cuda")]
fn serve_incoming(
    incoming: impl IntoIterator<Item = std::io::Result<TcpStream>>,
    shutdown: &AtomicBool,
    handler: impl Fn(TcpStream) + Sync,
) -> std::io::Result<()> {
    let (sender, receiver) = std::sync::mpsc::sync_channel::<TcpStream>(16);
    let receiver = Mutex::new(receiver);
    std::thread::scope(|scope| {
        for _ in 0..8 {
            let receiver = &receiver;
            let handler = &handler;
            scope.spawn(move || {
                loop {
                    let next = receiver.lock().unwrap().recv();
                    let Ok(stream) = next else { break };
                    if shutdown.load(Ordering::Relaxed) {
                        break;
                    }
                    handler(stream);
                }
            });
        }
        for incoming in incoming {
            if shutdown.load(Ordering::Relaxed) {
                break;
            }
            // Aborted/interrupted accepts must not shut down a healthy model server.
            let Ok(stream) = incoming else { continue };
            // Saturation closes the connection instead of blocking accept or
            // spawning unbounded threads. Queued connections consume no image memory.
            let _ = sender.try_send(stream);
        }
        drop(sender);
        Ok(())
    })
}
