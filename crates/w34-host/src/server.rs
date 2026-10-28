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

use serde::{Deserialize, Serialize};
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

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ChatMessage {
    pub role: String,
    #[serde(default)]
    pub content: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reasoning_content: Option<String>,
}

/// Formats OpenAI chat `messages` (and optional `tools` schema array) into a Qwen 3.5 / 3.8
/// ChatML prompt string. Preserves prefix stability across turns for `MambaRadixCache` hits.
pub fn format_chatml_prompt(
    messages: &[ChatMessage],
    tools: Option<&serde_json::Value>,
    enable_thinking: bool,
) -> String {
    let mut out = String::new();
    let has_system = messages.first().is_some_and(|m| m.role == "system");
    if !has_system
        && let Some(tools_arr) = tools.and_then(|v| v.as_array())
        && !tools_arr.is_empty()
    {
        out.push_str("<|im_start|>system\nYou are a helpful assistant.\n\n# Tools\n\nYou may call one or more functions to assist with the user query.\n<tools>\n");
        for t in tools_arr {
            if let Ok(s) = serde_json::to_string(t) {
                out.push_str(&s);
                out.push('\n');
            }
        }
        out.push_str("</tools>\n\nFor each function call, return a <tool_call><function=NAME><parameter=KEY>VALUE</parameter></function></tool_call> block.<|im_end|>\n");
    }

    for (idx, msg) in messages.iter().enumerate() {
        out.push_str("<|im_start|>");
        out.push_str(&msg.role);
        out.push('\n');
        if let Some(ref c) = msg.content {
            out.push_str(c);
        }
        if idx == 0
            && msg.role == "system"
            && let Some(tools_arr) = tools.and_then(|v| v.as_array())
            && !tools_arr.is_empty()
        {
            out.push_str("\n\n# Tools\n<tools>\n");
            for t in tools_arr {
                if let Ok(s) = serde_json::to_string(t) {
                    out.push_str(&s);
                    out.push('\n');
                }
            }
            out.push_str("</tools>");
        }
        out.push_str("<|im_end|>\n");
    }

    out.push_str("<|im_start|>assistant\n");
    if enable_thinking {
        out.push_str("<think>\n");
    } else {
        out.push_str("<think>\n\n</think>\n\n");
    }
    out
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
}

#[cfg(feature = "cuda")]
impl OpenAiServerState {
    pub fn new(model_name: String, engine: Qwen3_5Engine) -> Arc<Self> {
        Arc::new(Self {
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
                    eng.generate_sampled_with_radix_cache(&prompt_ids, params, None, 64)
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
            let messages: Vec<ChatMessage> = match req_json
                .get("messages")
                .ok_or_else(|| "missing 'messages' field".to_string())
                .and_then(|v| {
                    serde_json::from_value::<Vec<ChatMessage>>(v.clone())
                        .map_err(|e| format!("invalid 'messages': {e}"))
                }) {
                Ok(msgs) if !msgs.is_empty() => msgs,
                Ok(_) => {
                    let _ = write_json_response(
                        &mut stream,
                        400,
                        "Bad Request",
                        &json!({"error": "'messages' must not be empty"}),
                    );
                    return;
                }
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

            let prompt = format_chatml_prompt(&messages, req_json.get("tools"), enable_thinking);
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
                    eng.generate_sampled_with_radix_cache(&prompt_ids, params, None, 64)
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
    for incoming in listener.incoming() {
        if state.shutdown.load(Ordering::Relaxed) {
            break;
        }
        if let Ok(stream) = incoming {
            handle_http_connection(stream, &state);
        }
    }
    Ok(())
}
