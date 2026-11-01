//! Axum owns transport; only EngineWorker's OS thread owns CUDA objects.
use crate::{
    engine_worker::{EngineWorker, GenerationEvent, GenerationStream},
    model::Qwen3_5Engine,
    parser::{Qwen3Parser, SamplingParams, StepLogprobs, StreamParseDelta},
    scheduler::SchedulerConfig,
};
use axum::{
    Json, Router,
    body::to_bytes,
    extract::{Path, Request, State},
    http::StatusCode,
    response::{
        IntoResponse, Response, Sse,
        sse::{Event, KeepAlive},
    },
    routing::{get, post},
};
use serde_json::{Value, json};
use std::{
    convert::Infallible,
    sync::{
        Arc,
        atomic::{AtomicU64, Ordering},
    },
    time::{Duration, Instant},
};
use tokio::sync::{OwnedSemaphorePermit, Semaphore};
use tokio_stream::{Stream, StreamExt};

const BODY_LIMIT: usize = 32 * 1024 * 1024;
const BODY_TIMEOUT: Duration = Duration::from_secs(10);

#[derive(Default)]
pub struct ServerMetrics {
    pub requests_total: AtomicU64,
    pub tokens_generated_total: AtomicU64,
    pub prompt_tokens_total: AtomicU64,
    pub cached_prompt_tokens_total: AtomicU64,
    pub last_ttft_us: AtomicU64,
    pub last_itl_us: AtomicU64,
}

pub struct OpenAiServerState {
    pub model_name: String,
    pub worker: EngineWorker,
    pub metrics: ServerMetrics,
    pub allow_remote_images: bool,
    pub response_store: crate::responses_api::ResponseStore,
    constraint_factory: Option<crate::constraints::ConstraintFactory>,
    decoded_vocab: Vec<std::sync::OnceLock<Box<str>>>,
    preprocessing: Arc<Semaphore>,
    admission: Arc<Semaphore>,
}
impl OpenAiServerState {
    pub fn new(model_name: String, engine: Qwen3_5Engine) -> Arc<Self> {
        Self::with_image_policy(model_name, engine, false)
    }
    pub fn with_image_policy(
        model_name: String,
        engine: Qwen3_5Engine,
        allow_remote_images: bool,
    ) -> Arc<Self> {
        Self::with_worker(
            model_name,
            EngineWorker::start(engine, SchedulerConfig::default()),
            allow_remote_images,
        )
    }
    pub fn from_loader(
        model_name: String,
        allow_remote_images: bool,
        loader: impl FnOnce() -> Result<Qwen3_5Engine, String> + Send + 'static,
    ) -> Result<Arc<Self>, String> {
        Ok(Self::with_worker(
            model_name,
            EngineWorker::from_loader(loader, SchedulerConfig::default())?,
            allow_remote_images,
        ))
    }
    fn with_worker(
        model_name: String,
        worker: EngineWorker,
        allow_remote_images: bool,
    ) -> Arc<Self> {
        let eos: Vec<_> = worker.eos_token_ids.iter().map(|&id| id as u32).collect();
        let constraint_factory = worker.tokenizer.as_ref().and_then(|t| {
            crate::constraints::ConstraintFactory::new(t.clone(), worker.vocab_size, &eos).ok()
        });
        let decoded_vocab = if worker.tokenizer.is_some() {
            (0..worker.vocab_size)
                .map(|_| std::sync::OnceLock::new())
                .collect()
        } else {
            Vec::new()
        };
        Arc::new(Self {
            model_name,
            worker,
            allow_remote_images,
            response_store: crate::responses_api::ResponseStore::default(),
            constraint_factory,
            decoded_vocab,
            metrics: ServerMetrics::default(),
            preprocessing: Arc::new(Semaphore::new(32)),
            admission: Arc::new(Semaphore::new(32)),
        })
    }
}

type ApiError = (StatusCode, Json<Value>);
fn error(status: StatusCode, message: impl ToString) -> ApiError {
    (
        status,
        Json(json!({"error":{"message":message.to_string(),"type":"invalid_request_error"}})),
    )
}
fn bad(message: impl ToString) -> ApiError {
    error(StatusCode::BAD_REQUEST, message)
}
fn busy() -> ApiError {
    error(
        StatusCode::SERVICE_UNAVAILABLE,
        "server at capacity or shutting down",
    )
}

pub fn router(state: Arc<OpenAiServerState>) -> Router {
    Router::new()
        .route("/health", get(|| async { Json(json!({"status":"ok"})) }))
        .route("/v1/models", get(models))
        .route("/metrics", get(metrics))
        .route("/tokenize", post(tokenize))
        .route("/detokenize", post(detokenize))
        .route("/v1/chat/completions", post(chat))
        .route("/v1/chat/completions/input_tokens", post(chat_input_tokens))
        .route("/chat/completions/input_tokens", post(chat_input_tokens))
        .route("/v1/completions", post(completion))
        .route("/v1/responses", post(responses_create))
        .route("/v1/responses/input_tokens", post(responses_input_tokens))
        .route("/responses/input_tokens", post(responses_input_tokens))
        .route(
            "/v1/responses/{response_id}",
            get(responses_get).delete(responses_delete),
        )
        .route("/v1/responses/{response_id}/cancel", post(responses_cancel))
        .route("/v1/messages", post(anthropic_messages))
        .route("/v1/messages/count_tokens", post(anthropic_count_tokens))
        .route("/v1/systemone", post(systemone_decide))
        .route("/systemone", post(systemone_decide))
        .route("/v1/decisions", post(decisions_adapter))
        .route("/decisions", post(decisions_adapter))
        .fallback(|| async { error(StatusCode::NOT_FOUND, "not found") })
        .with_state(state)
}
async fn models(State(s): State<Arc<OpenAiServerState>>) -> Json<Value> {
    Json(
        json!({"object":"list","data":[{"id":s.model_name,"object":"model","owned_by":"warp-34"}]}),
    )
}
async fn metrics(State(s): State<Arc<OpenAiServerState>>) -> impl IntoResponse {
    let m = &s.metrics;
    let load = |n: &AtomicU64| n.load(Ordering::Relaxed);
    let prompts = load(&m.prompt_tokens_total);
    let cached = load(&m.cached_prompt_tokens_total);
    (
        [(
            axum::http::header::CONTENT_TYPE,
            "text/plain; version=0.0.4",
        )],
        format!(
            "w34_requests_total {}\nw34_tokens_generated_total {}\nw34_prompt_tokens_total {}\nw34_cached_prompt_tokens_total {}\nw34_radix_cache_hit_rate {}\nw34_last_ttft_ms {}\nw34_last_itl_ms {}\nw34_active_requests {}\nw34_cancelled_requests_total {}\nw34_peak_active_requests {}\nw34_decode_bucket_mask {}\n",
            load(&m.requests_total),
            load(&m.tokens_generated_total),
            prompts,
            cached,
            cached as f64 / prompts.max(1) as f64,
            load(&m.last_ttft_us) as f64 / 1000.0,
            load(&m.last_itl_us) as f64 / 1000.0,
            load(&s.worker.stats.active),
            load(&s.worker.stats.cancelled),
            load(&s.worker.stats.peak_active),
            load(&s.worker.stats.decode_bucket_mask)
        ),
    )
}
async fn read_json(state: &OpenAiServerState, req: Request) -> Result<Value, ApiError> {
    read_json_with_len(state, req).await.map(|(v, _)| v)
}
async fn read_json_with_len(
    state: &OpenAiServerState,
    req: Request,
) -> Result<(Value, usize), ApiError> {
    let bytes = tokio::time::timeout(BODY_TIMEOUT, to_bytes(req.into_body(), BODY_LIMIT))
        .await
        .map_err(|_| {
            error(
                StatusCode::REQUEST_TIMEOUT,
                "request body deadline exceeded",
            )
        })?
        .map_err(|e| error(StatusCode::PAYLOAD_TOO_LARGE, e))?;
    let len = bytes.len();
    if len <= 16 * 1024 {
        return serde_json::from_slice(&bytes)
            .map(|v| (v, len))
            .map_err(bad);
    }
    cpu(state, move || {
        serde_json::from_slice(&bytes)
            .map(|v| (v, len))
            .map_err(|e| e.to_string())
    })
    .await
}
async fn cpu<T: Send + 'static>(
    state: &OpenAiServerState,
    f: impl FnOnce() -> Result<T, String> + Send + 'static,
) -> Result<T, ApiError> {
    let permit = state
        .preprocessing
        .clone()
        .acquire_owned()
        .await
        .map_err(|_| busy())?;
    tokio::task::spawn_blocking(move || {
        let _permit = permit;
        f()
    })
    .await
    .map_err(|e| error(StatusCode::INTERNAL_SERVER_ERROR, e))?
    .map_err(bad)
}
async fn tokenize(
    State(s): State<Arc<OpenAiServerState>>,
    req: Request,
) -> Result<Json<Value>, ApiError> {
    let _admission = s
        .admission
        .clone()
        .try_acquire_owned()
        .map_err(|_| busy())?;
    let body = read_json(&s, req).await?;
    if s.worker.tokenizer.is_none() {
        return Err(bad("tokenizer unavailable"));
    }
    let s_tok = s.clone();
    cpu(&s, move || {
        let tokenizer = s_tok
            .worker
            .tokenizer
            .as_ref()
            .ok_or("tokenizer unavailable")?;
        let text = body
            .get("prompt")
            .or_else(|| body.get("text"))
            .and_then(Value::as_str)
            .ok_or("missing prompt")?;
        let encoded = tokenizer.encode(text, false).map_err(|e| e.to_string())?;
        Ok(Json(
            json!({"tokens":encoded.get_ids(),"count":encoded.len()}),
        ))
    })
    .await
}
async fn detokenize(
    State(s): State<Arc<OpenAiServerState>>,
    req: Request,
) -> Result<Json<Value>, ApiError> {
    let _admission = s
        .admission
        .clone()
        .try_acquire_owned()
        .map_err(|_| busy())?;
    let body = read_json(&s, req).await?;
    if s.worker.tokenizer.is_none() {
        return Err(bad("tokenizer unavailable"));
    }
    let s_tok = s.clone();
    cpu(&s, move || {
        let tokenizer = s_tok
            .worker
            .tokenizer
            .as_ref()
            .ok_or("tokenizer unavailable")?;
        let ids: Vec<u32> =
            serde_json::from_value(body.get("tokens").cloned().ok_or("missing tokens")?)
                .map_err(|e| e.to_string())?;
        if ids.iter().any(|&id| tokenizer.id_to_token(id).is_none()) {
            return Err("invalid token id".into());
        }
        let text = tokenizer.decode(&ids, false).map_err(|e| e.to_string())?;
        Ok(Json(json!({"text":text})))
    })
    .await
}
async fn chat(State(s): State<Arc<OpenAiServerState>>, req: Request) -> Result<Response, ApiError> {
    generate(s, req, true).await
}
async fn completion(
    State(s): State<Arc<OpenAiServerState>>,
    req: Request,
) -> Result<Response, ApiError> {
    generate(s, req, false).await
}

struct Prepared {
    prompt: Vec<i32>,
    images: Vec<crate::vision::PreprocessedImage>,
    params: SamplingParams,
    constraint: Option<crate::parser::TokenBitmaskConstraint>,
    stream: bool,
    include_usage: bool,
    thinking: bool,
}
fn prepare(s: &OpenAiServerState, request: Value, chat: bool) -> Result<Prepared, String> {
    if !request.is_object() {
        return Err("request must be an object".into());
    }
    if request.get("n").is_some_and(|n| n.as_u64() != Some(1)) {
        return Err("only n=1 is supported".into());
    }
    let max_tokens = request
        .get("max_completion_tokens")
        .or_else(|| request.get("max_tokens"))
        .map_or(Ok(256), |n| {
            n.as_u64()
                .and_then(|n| usize::try_from(n).ok())
                .ok_or("invalid max_tokens")
        })?;
    if max_tokens == 0 {
        return Err("max_tokens must be positive".into());
    }
    let mut params = if chat {
        super::native_sampling_params(&request, max_tokens)?
    } else {
        super::completion_sampling_params(&request, max_tokens)?
    };
    if params
        .stop_token_ids
        .iter()
        .chain(params.logit_bias.keys())
        .any(|&id| id < 0 || id as usize >= s.worker.vocab_size)
    {
        return Err("stop_token_ids/logit_bias token outside model vocabulary".into());
    }
    let thinking = chat
        && request
            .pointer("/chat_template_kwargs/enable_thinking")
            .and_then(Value::as_bool)
            .unwrap_or(true);
    params.enable_thinking = thinking;
    let (prompt, images) = if chat {
        let (messages, images) = super::parse_openai_messages_with_image_policy(
            request.get("messages").ok_or("missing messages")?,
            s.allow_remote_images,
        )?;
        let preserve = request
            .pointer("/chat_template_kwargs/preserve_reasoning")
            .and_then(Value::as_bool)
            .or_else(|| {
                request
                    .pointer("/chat_template_kwargs/preserve_thinking")
                    .and_then(Value::as_bool)
            })
            .unwrap_or(s.worker.default_preserve_thinking);
        (
            super::format_chatml_prompt_with_options(
                &messages,
                request.get("tools").filter(|v| !v.is_null()),
                thinking,
                preserve,
            ),
            images,
        )
    } else {
        (
            request
                .get("prompt")
                .and_then(Value::as_str)
                .ok_or("missing string prompt")?
                .to_owned(),
            Vec::new(),
        )
    };
    let tokenizer = s.worker.tokenizer.as_ref().ok_or("tokenizer unavailable")?;
    let prompt: Vec<i32> = tokenizer
        .encode(prompt, false)
        .map_err(|e| e.to_string())?
        .get_ids()
        .iter()
        .map(|&id| id as i32)
        .collect();
    if prompt.is_empty() {
        return Err("prompt must be nonempty".into());
    }
    let constraint = match crate::request_constraints::RequestConstraint::parse(&request)? {
        Some(spec) => {
            let factory = s
                .constraint_factory
                .as_ref()
                .ok_or("constraint tokenizer unavailable")?;
            match spec.compile(factory, thinking) {
                Ok(c) => Some(c),
                Err(_) if !spec.is_mandatory() => None,
                Err(e) => return Err(e),
            }
        }
        None => None,
    };
    let stream = request
        .get("stream")
        .map_or(Ok(false), |v| v.as_bool().ok_or("invalid stream"))?;
    let include_usage = request
        .pointer("/stream_options/include_usage")
        .map_or(Ok(false), |v| v.as_bool().ok_or("invalid include_usage"))?;
    Ok(Prepared {
        prompt,
        images,
        params,
        constraint,
        stream,
        include_usage,
        thinking,
    })
}
async fn generate(
    s: Arc<OpenAiServerState>,
    request: Request,
    chat: bool,
) -> Result<Response, ApiError> {
    // Held through body consumption and the complete output stream, not just response creation.
    let admission = s
        .admission
        .clone()
        .try_acquire_owned()
        .map_err(|_| busy())?;
    let start = Instant::now();
    let (json, body_len) = read_json_with_len(&s, request).await?;
    let prep_state = s.clone();
    let inline_fast = !chat
        && body_len <= 16 * 1024
        && json.get("prompt").is_some_and(Value::is_string)
        && json
            .get("prompt")
            .and_then(Value::as_str)
            .is_some_and(|p| p.len() <= 4096)
        && matches!(
            crate::request_constraints::RequestConstraint::parse(&json),
            Ok(None)
        );
    let prepared = if inline_fast {
        prepare(&prep_state, json, chat).map_err(bad)?
    } else {
        cpu(&s, move || prepare(&prep_state, json, chat)).await?
    };
    let prompt_len = prepared.prompt.len();
    let generation = s
        .worker
        .submit_with_images(
            prepared.prompt,
            prepared.images,
            prepared.params.clone(),
            prepared.constraint,
        )
        .map_err(|e| error(StatusCode::SERVICE_UNAVAILABLE, e))?;
    let stream = prepared.stream;
    let include_usage = prepared.include_usage;
    let request_id = s.metrics.requests_total.fetch_add(1, Ordering::Relaxed) + 1;
    s.metrics
        .prompt_tokens_total
        .fetch_add(prompt_len as u64, Ordering::Relaxed);
    let output = output_stream(
        s,
        generation,
        prepared.params,
        prepared.thinking,
        chat,
        prompt_len,
        request_id,
        start,
        admission,
    );
    if stream {
        let events = async_stream::stream! {
            let mut output = std::pin::pin!(output);
            while let Some(item) = output.next().await {
                match item {
                    Output::Delta(value) => yield Ok::<_,Infallible>(Event::default().data(value.to_string())),
                    Output::Complete(value) => {
                        let mut final_chunk = value.clone();
                        final_chunk["object"] = json!(if chat { "chat.completion.chunk" } else { "text_completion" });
                        if chat { final_chunk["choices"][0].as_object_mut().unwrap().remove("message"); final_chunk["choices"][0]["delta"] = json!({}); }
                        else { final_chunk["choices"][0]["text"] = json!(""); }
                        final_chunk.as_object_mut().unwrap().remove("usage");
                        final_chunk["choices"][0]["logprobs"] = Value::Null;
                        yield Ok(Event::default().data(final_chunk.to_string()));
                        if include_usage { final_chunk["choices"] = json!([]); final_chunk["usage"] = value["usage"].clone(); yield Ok(Event::default().data(final_chunk.to_string())); }
                    }
                    Output::Error(message) => {
                        yield Ok(Event::default().data(
                            json!({"error":{"message":message,"type":"server_error"}}).to_string(),
                        ));
                    }
                }
            }
            yield Ok(Event::default().data("[DONE]"));
        };
        Ok(Sse::new(events)
            .keep_alive(KeepAlive::new().interval(Duration::from_secs(15)))
            .into_response())
    } else {
        let mut output = std::pin::pin!(output);
        while let Some(item) = output.next().await {
            match item {
                Output::Complete(value) => return Ok(Json(value).into_response()),
                Output::Error(message) => {
                    return Err(error(StatusCode::INTERNAL_SERVER_ERROR, message));
                }
                Output::Delta(_) => {}
            }
        }
        Err(error(
            StatusCode::INTERNAL_SERVER_ERROR,
            "engine stream ended without completion",
        ))
    }
}

enum Output {
    Delta(Value),
    Complete(Value),
    Error(String),
}

/// Hold only the suffix that might begin a stop string. Never expose part of a delimiter.
#[derive(Default)]
struct StopBuffer {
    pending: String,
}
impl StopBuffer {
    fn push(&mut self, chunk: &str, stops: &[String]) -> (String, bool) {
        let (text, matched) = self.push_matched(chunk, stops);
        (text, matched.is_some())
    }

    fn push_matched(&mut self, chunk: &str, stops: &[String]) -> (String, Option<String>) {
        self.pending.push_str(chunk);
        if let Some((pos, matched)) = stops
            .iter()
            .filter_map(|s| self.pending.find(s).map(|p| (p, s)))
            .min_by_key(|(p, _)| *p)
        {
            let matched = matched.clone();
            let text = self.pending[..pos].to_owned();
            self.pending.clear();
            return (text, Some(matched));
        }
        let hold = stops
            .iter()
            .flat_map(|s| {
                s.char_indices()
                    .map(|(n, _)| n)
                    .filter(|&n| n > 0)
                    .map(move |n| &s[..n])
            })
            .filter(|prefix| self.pending.ends_with(prefix))
            .map(str::len)
            .max()
            .unwrap_or(0);
        let split = self.pending.len() - hold;
        let rest = self.pending.split_off(split);
        (std::mem::replace(&mut self.pending, rest), None)
    }
}
fn delta_value(d: StreamParseDelta, tool_index: &mut usize) -> Value {
    let mut value = json!({});
    if let Some(text) = d.reasoning_delta {
        value["reasoning_content"] = json!(text);
    }
    if let Some(text) = d.content_delta {
        value["content"] = json!(text);
    }
    if !d.completed_tool_calls.is_empty() {
        value["tool_calls"] = json!(
            d.completed_tool_calls
                .into_iter()
                .map(|call| {
                    let mut v = serde_json::to_value(call).unwrap();
                    v["index"] = json!(*tool_index);
                    *tool_index += 1;
                    v
                })
                .collect::<Vec<_>>()
        );
    }
    value
}
fn logprob_value(
    lp: &StepLogprobs,
    tokenizer: &tokenizers::Tokenizer,
    decoded_vocab: &[std::sync::OnceLock<Box<str>>],
) -> Value {
    let token_str = |id: i32| -> String {
        if id >= 0
            && let Some(cell) = decoded_vocab.get(id as usize)
        {
            return cell
                .get_or_init(|| {
                    tokenizer
                        .decode(&[id as u32], false)
                        .unwrap_or_default()
                        .into_boxed_str()
                })
                .to_string();
        }
        tokenizer.decode(&[id as u32], false).unwrap_or_default()
    };
    let text = token_str(lp.token_id);
    json!({"token":text,"logprob":lp.logprob,"bytes":text.as_bytes(),"top_logprobs":lp.top_logprobs.iter().map(|p| { let text = token_str(p.token_id); json!({"token":text,"logprob":p.logprob,"bytes":text.as_bytes()}) }).collect::<Vec<_>>()})
}
#[allow(clippy::too_many_arguments)]
fn output_stream(
    s: Arc<OpenAiServerState>,
    mut generation: GenerationStream,
    params: SamplingParams,
    thinking: bool,
    chat: bool,
    prompt_len: usize,
    request_id: u64,
    start: Instant,
    admission: OwnedSemaphorePermit,
) -> impl Stream<Item = Output> + Send {
    async_stream::stream! {
        let _admission = admission;
        let tokenizer = s.worker.tokenizer.as_ref().unwrap();
        let mut decoder = tokenizer.decode_stream(false);
        let mut parser = Qwen3Parser::new(thinking);
        let mut stops = StopBuffer::default();
        let mut text = String::new();
        let mut count = 0;
        let mut cached = 0;
        let mut done = false;
        let mut stopped = false;
        let mut logs = Vec::new();
        let mut legacy_offset = 0;
        let mut decoded_offset = 0;
        let mut tool_index = 0;
        let mut last_token = Instant::now();
        let id = format!("{}-{request_id}",if chat { "chatcmpl" } else { "cmpl" });
        let created = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap_or_default().as_secs();
        let chunk = |delta: Value, lp: Value| {
            let choice = if chat { json!({"index":0,"delta":delta,"finish_reason":null,"logprobs":lp}) }
                else { json!({"index":0,"text":delta["content"].as_str().unwrap_or(""),"finish_reason":null,"logprobs":lp}) };
            json!({"id":id,"object":if chat {"chat.completion.chunk"} else {"text_completion"},"created":created,"model":s.model_name,"choices":[choice]})
        };
        if chat { yield Output::Delta(chunk(json!({"role":"assistant"}),Value::Null)); }
        while let Some(event) = generation.events.recv().await {
            match event {
                GenerationEvent::Token { id, logprobs } => {
                    count += 1;
                    s.metrics.tokens_generated_total.fetch_add(1,Ordering::Relaxed);
                    if count == 1 { s.metrics.last_ttft_us.store(start.elapsed().as_micros() as u64,Ordering::Relaxed); }
                    else { s.metrics.last_itl_us.store(last_token.elapsed().as_micros() as u64,Ordering::Relaxed); }
                    last_token = Instant::now();
                    let mut lp = logprobs.as_ref().map(|lp| logprob_value(lp, tokenizer, &s.decoded_vocab));
                    if chat && let Some(lp) = &lp { logs.push(lp.clone()); }
                    if count >= params.min_tokens && params.is_stop_or_eos(id,&s.worker.eos_token_ids) {
                        // The hidden terminal token still counts toward usage. Record its
                        // position without decoding it or exposing stop/EOS text. Emit a
                        // logprob-only legacy delta so SSE and collected arrays agree.
                        if !chat && let Some(mut lp) = lp {
                            lp["_text_offset"] = json!(decoded_offset);
                            logs.push(lp.clone());
                            yield Output::Delta(chunk(json!({"content":""}),super::completion_logprobs(&[lp], &mut legacy_offset)));
                        }
                        stopped = true;
                        continue;
                    }
                    let decoded = match decoder.step(id as u32) {
                        Ok(Some(text)) => text, Ok(None) if chat => continue, Ok(None) => String::new(),
                        Err(e) => { yield Output::Error(e.to_string()); return; }
                    };
                    if !chat {
                        if let Some(lp) = &mut lp {
                            lp["_text_offset"] = json!(decoded_offset);
                            logs.push(lp.clone());
                        }
                        // Advance even without logprobs; decoder None contributes zero.
                        // Stop buffering may delay emission, but does not change positions.
                        decoded_offset += decoded.chars().count();
                    }
                    let active_stops = if count >= params.min_tokens { params.stop_strings.as_slice() } else { &[] };
                    let (visible, matched) = stops.push(&decoded, active_stops);
                    text.push_str(&visible);
                    let delta = if chat { delta_value(parser.push_str(&visible),&mut tool_index) } else { json!({"content":visible}) };
                    if delta.as_object().is_some_and(|o| !o.is_empty()) || lp.is_some() {
                        yield Output::Delta(chunk(delta,lp.map_or(Value::Null,|v| if chat { json!({"content":[v]}) } else { super::completion_logprobs(&[v], &mut legacy_offset) })));
                    }
                    if matched { stopped = true; done = true; break; }
                }
                GenerationEvent::Finished { cached_tokens, .. } => { cached = cached_tokens; done = true; break; }
                GenerationEvent::Error(message) => { yield Output::Error(message); return; }
            }
        }
        cached = cached.max(generation.cached_tokens.load(Ordering::Acquire) as usize);
        drop(generation); // cancellation also covers stop strings and a dropped HTTP response body
        if !done { yield Output::Error("engine stream closed before completion (overload, shutdown, or worker failure)".into()); return; }
        let tail = std::mem::take(&mut stops.pending);
        text.push_str(&tail);
        if !tail.is_empty() {
            let delta = if chat { delta_value(parser.push_str(&tail),&mut tool_index) } else { json!({"content":tail}) };
            yield Output::Delta(chunk(delta,Value::Null));
        }
        let (tail,turn) = parser.finish();
        let delta = delta_value(tail,&mut tool_index);
        if chat && delta.as_object().is_some_and(|o| !o.is_empty()) { yield Output::Delta(chunk(delta,Value::Null)); }
        let finish = if chat && !turn.tool_calls.is_empty() { "tool_calls" } else if !stopped && count >= params.max_tokens { "length" } else { "stop" };
        s.metrics.cached_prompt_tokens_total.fetch_add(cached as u64,Ordering::Relaxed);
        let lp = if !params.logprobs { Value::Null } else if chat { json!({"content":logs}) } else { super::completion_logprobs(&logs, &mut 0) };
        let choice = if chat { json!({"index":0,"message":{"role":"assistant","content":turn.content,"reasoning_content":turn.reasoning_content,"tool_calls":turn.tool_calls},"finish_reason":finish,"logprobs":lp}) }
            else { json!({"index":0,"text":text,"finish_reason":finish,"logprobs":lp}) };
        yield Output::Complete(json!({"id":id,"object":if chat {"chat.completion"} else {"text_completion"},"created":created,"model":s.model_name,"choices":[choice],"usage":{"prompt_tokens":prompt_len,"completion_tokens":count,"total_tokens":prompt_len+count,"prompt_tokens_details":{"cached_tokens":cached}}}));
    }
}

fn responses_api_err(err: crate::responses_api::ResponsesApiError) -> ApiError {
    let status = match err.kind {
        crate::responses_api::ResponsesErrorKind::InvalidRequest => StatusCode::BAD_REQUEST,
        crate::responses_api::ResponsesErrorKind::NotFound => StatusCode::NOT_FOUND,
    };
    (status, Json(err.to_openai_json()))
}

fn anthropic_api_err(status: StatusCode, err: crate::anthropic_api::AnthropicApiError) -> ApiError {
    (status, Json(err.to_json()))
}

fn to_anthropic_err((status, Json(body)): ApiError) -> ApiError {
    let msg = body
        .pointer("/error/message")
        .and_then(Value::as_str)
        .unwrap_or("request failed")
        .to_owned();
    let err_type = match status {
        StatusCode::UNAUTHORIZED | StatusCode::FORBIDDEN => {
            crate::anthropic_api::AnthropicErrorType::Authentication
        }
        StatusCode::NOT_FOUND => crate::anthropic_api::AnthropicErrorType::NotFound,
        StatusCode::TOO_MANY_REQUESTS | StatusCode::SERVICE_UNAVAILABLE => {
            crate::anthropic_api::AnthropicErrorType::Overloaded
        }
        s if s.is_client_error() => crate::anthropic_api::AnthropicErrorType::InvalidRequest,
        _ => crate::anthropic_api::AnthropicErrorType::Api,
    };
    (
        status,
        Json(
            crate::anthropic_api::AnthropicApiError {
                error_type: err_type,
                message: msg,
            }
            .to_json(),
        ),
    )
}

struct ActiveResponseGuard {
    state: Arc<OpenAiServerState>,
    id: String,
}
impl Drop for ActiveResponseGuard {
    fn drop(&mut self) {
        self.state.response_store.unregister_active(&self.id);
    }
}

async fn chat_input_tokens(
    State(s): State<Arc<OpenAiServerState>>,
    req: Request,
) -> Result<Json<Value>, ApiError> {
    let _admission = s
        .admission
        .clone()
        .try_acquire_owned()
        .map_err(|_| busy())?;
    let mut body = read_json(&s, req).await?;
    if let Some(obj) = body.as_object_mut() {
        obj.remove("max_tokens");
        obj.remove("max_completion_tokens");
    }
    let prep_state = s.clone();
    let count = cpu(&s, move || {
        prepare(&prep_state, body, true).map(|p| p.prompt.len())
    })
    .await?;
    Ok(Json(json!({
        "object": "response.input_tokens",
        "input_tokens": count,
    })))
}

fn responses_caller_scope(
    s: &OpenAiServerState,
    headers: &axum::http::HeaderMap,
) -> Option<String> {
    let raw = headers
        .get(axum::http::header::AUTHORIZATION)
        .or_else(|| headers.get("x-api-key"))
        .map(|v| v.as_bytes());
    s.response_store.scope_for_credential(raw)
}

async fn responses_input_tokens(
    State(s): State<Arc<OpenAiServerState>>,
    req: Request,
) -> Result<Json<Value>, ApiError> {
    let _admission = s
        .admission
        .clone()
        .try_acquire_owned()
        .map_err(|_| busy())?;
    let caller_scope = responses_caller_scope(&s, req.headers());
    let mut body = read_json(&s, req).await?;
    if let Some(obj) = body.as_object_mut() {
        obj.remove("max_output_tokens");
        obj.remove("max_tokens");
    }
    let (chat_req, _) = crate::responses_api::translate_responses_to_chat(
        &body,
        &s.model_name,
        &s.response_store,
        caller_scope.as_deref(),
        0,
    )
    .map_err(responses_api_err)?;
    let prep_state = s.clone();
    let count = cpu(&s, move || {
        prepare(&prep_state, chat_req, true).map(|p| p.prompt.len())
    })
    .await?;
    Ok(Json(json!({
        "object": "response.input_tokens",
        "input_tokens": count,
    })))
}

async fn anthropic_count_tokens(
    State(s): State<Arc<OpenAiServerState>>,
    req: Request,
) -> Result<Json<Value>, ApiError> {
    let _admission = s
        .admission
        .clone()
        .try_acquire_owned()
        .map_err(|_| to_anthropic_err(busy()))?;
    let body = read_json(&s, req).await.map_err(to_anthropic_err)?;
    let (mut chat_req, _) =
        crate::anthropic_api::translate_anthropic_to_chat(&body, &s.model_name, 0, false)
            .map_err(|e| anthropic_api_err(StatusCode::BAD_REQUEST, e))?;
    if let Some(obj) = chat_req.as_object_mut() {
        obj.insert("max_tokens".into(), json!(256));
    }
    let prep_state = s.clone();
    let count = cpu(&s, move || {
        prepare(&prep_state, chat_req, true).map(|p| p.prompt.len())
    })
    .await
    .map_err(to_anthropic_err)?;
    Ok(Json(json!({
        "input_tokens": count,
    })))
}

async fn responses_get(
    State(s): State<Arc<OpenAiServerState>>,
    headers: axum::http::HeaderMap,
    Path(response_id): Path<String>,
) -> Result<Json<Value>, ApiError> {
    let not_found = || {
        responses_api_err(crate::responses_api::ResponsesApiError::not_found(format!(
            "response '{response_id}' not found"
        )))
    };
    let caller_scope = responses_caller_scope(&s, &headers).ok_or_else(not_found)?;
    s.response_store
        .get_response(&response_id, &caller_scope)
        .map(Json)
        .ok_or_else(not_found)
}

async fn responses_delete(
    State(s): State<Arc<OpenAiServerState>>,
    headers: axum::http::HeaderMap,
    Path(response_id): Path<String>,
) -> Result<Json<Value>, ApiError> {
    let not_found = || {
        responses_api_err(crate::responses_api::ResponsesApiError::not_found(format!(
            "response '{response_id}' not found"
        )))
    };
    let caller_scope = responses_caller_scope(&s, &headers).ok_or_else(not_found)?;
    if s.response_store.delete(&response_id, &caller_scope) {
        Ok(Json(json!({
            "id": response_id,
            "object": "response.deleted",
            "deleted": true,
        })))
    } else {
        Err(not_found())
    }
}

async fn responses_cancel(
    State(s): State<Arc<OpenAiServerState>>,
    headers: axum::http::HeaderMap,
    Path(response_id): Path<String>,
) -> Result<Json<Value>, ApiError> {
    let not_found = || {
        responses_api_err(crate::responses_api::ResponsesApiError::not_found(format!(
            "response '{response_id}' not found"
        )))
    };
    let caller_scope = responses_caller_scope(&s, &headers).ok_or_else(not_found)?;
    s.response_store
        .cancel(&response_id, &caller_scope)
        .map(Json)
        .ok_or_else(not_found)
}

fn count_reasoning_tokens(
    tokenizer: &tokenizers::Tokenizer,
    turn: &crate::parser::ParsedAssistantTurn,
    output_tokens: usize,
) -> usize {
    match turn.reasoning_content.as_deref().filter(|s| !s.is_empty()) {
        None => 0,
        Some(text) => tokenizer
            .encode(text, false)
            .map(|e| e.len().min(output_tokens))
            .unwrap_or(0),
    }
}

async fn responses_create(
    State(s): State<Arc<OpenAiServerState>>,
    request: Request,
) -> Result<Response, ApiError> {
    let admission = s
        .admission
        .clone()
        .try_acquire_owned()
        .map_err(|_| busy())?;
    let start = Instant::now();
    let caller_scope = responses_caller_scope(&s, request.headers());
    let body = read_json(&s, request).await?;
    let request_id = s.metrics.requests_total.fetch_add(1, Ordering::Relaxed) + 1;
    let (chat_req, meta) = crate::responses_api::translate_responses_to_chat(
        &body,
        &s.model_name,
        &s.response_store,
        caller_scope.as_deref(),
        request_id,
    )
    .map_err(responses_api_err)?;
    let prep_state = s.clone();
    let prepared = cpu(&s, move || prepare(&prep_state, chat_req, true)).await?;
    let prompt_len = prepared.prompt.len();
    let generation = s
        .worker
        .submit_with_images(
            prepared.prompt,
            prepared.images,
            prepared.params.clone(),
            prepared.constraint,
        )
        .map_err(|e| error(StatusCode::SERVICE_UNAVAILABLE, e))?;
    if let Some(ref scope) = caller_scope {
        s.response_store.register_active(
            meta.response_id.clone(),
            scope.clone(),
            generation.cancel_handle(),
        );
    }
    let active_guard = ActiveResponseGuard {
        state: s.clone(),
        id: meta.response_id.clone(),
    };
    s.metrics
        .prompt_tokens_total
        .fetch_add(prompt_len as u64, Ordering::Relaxed);

    if meta.stream {
        let params = prepared.params;
        let thinking = prepared.thinking;
        let events = async_stream::stream! {
            let _admission = admission;
            let _active_guard = active_guard;
            let tokenizer = s.worker.tokenizer.as_ref().unwrap();
            let mut decoder = tokenizer.decode_stream(false);
            let mut parser = Qwen3Parser::new(thinking);
            let mut stops = StopBuffer::default();
            let mut sse = crate::responses_api::ResponsesSseFormatter::new(meta.clone());
            let mut generation = generation;
            let mut count = 0usize;
            let mut cached = 0usize;
            let mut done = false;
            let mut stopped = false;
            let mut last_token = Instant::now();

            for (ev_name, payload) in sse.start() {
                yield Ok::<_, Infallible>(Event::default().event(ev_name).data(payload.to_string()));
            }

            while let Some(event) = generation.events.recv().await {
                match event {
                    GenerationEvent::Token { id, .. } => {
                        count += 1;
                        s.metrics.tokens_generated_total.fetch_add(1, Ordering::Relaxed);
                        if count == 1 {
                            s.metrics.last_ttft_us.store(start.elapsed().as_micros() as u64, Ordering::Relaxed);
                        } else {
                            s.metrics.last_itl_us.store(last_token.elapsed().as_micros() as u64, Ordering::Relaxed);
                        }
                        last_token = Instant::now();
                        if count >= params.min_tokens && params.is_stop_or_eos(id, &s.worker.eos_token_ids) {
                            stopped = true;
                            continue;
                        }
                        let decoded = match decoder.step(id as u32) {
                            Ok(Some(t)) => t,
                            Ok(None) => continue,
                            Err(e) => {
                                for (ev_name, payload) in sse.fail(&e.to_string()) {
                                    yield Ok(Event::default().event(ev_name).data(payload.to_string()));
                                }
                                return;
                            }
                        };
                        let active_stops = if count >= params.min_tokens { params.stop_strings.as_slice() } else { &[] };
                        let (visible, matched) = stops.push(&decoded, active_stops);
                        for (ev_name, payload) in sse.push_delta(parser.push_str(&visible)) {
                            yield Ok(Event::default().event(ev_name).data(payload.to_string()));
                        }
                        if matched {
                            stopped = true;
                            done = true;
                            break;
                        }
                    }
                    GenerationEvent::Finished { cached_tokens, .. } => {
                        cached = cached_tokens;
                        done = true;
                        break;
                    }
                    GenerationEvent::Error(message) => {
                        for (ev_name, payload) in sse.fail(&message) {
                            yield Ok(Event::default().event(ev_name).data(payload.to_string()));
                        }
                        return;
                    }
                }
            }
            cached = cached.max(generation.cached_tokens.load(Ordering::Acquire) as usize);
            drop(generation);
            if !done {
                for (ev_name, payload) in sse.fail("engine stream closed before completion") {
                    yield Ok(Event::default().event(ev_name).data(payload.to_string()));
                }
                return;
            }
            let tail = std::mem::take(&mut stops.pending);
            if !tail.is_empty() {
                for (ev_name, payload) in sse.push_delta(parser.push_str(&tail)) {
                    yield Ok(Event::default().event(ev_name).data(payload.to_string()));
                }
            }
            let (tail_delta, turn) = parser.finish();
            for (ev_name, payload) in sse.push_delta(tail_delta) {
                yield Ok(Event::default().event(ev_name).data(payload.to_string()));
            }
            let finish = if !turn.tool_calls.is_empty() {
                "tool_calls"
            } else if !stopped && count >= params.max_tokens {
                "length"
            } else {
                "stop"
            };
            s.metrics.cached_prompt_tokens_total.fetch_add(cached as u64, Ordering::Relaxed);
            let reasoning_tokens = count_reasoning_tokens(tokenizer, &turn, count);
            let (end_frames, final_resp, history) =
                sse.finish(&turn, finish, prompt_len, cached, count, reasoning_tokens);
            if meta.store
                && let Some(ref scope) = caller_scope
            {
                s.response_store.insert(crate::responses_api::StoredResponse {
                    id: meta.response_id.clone(),
                    owner_scope: scope.clone(),
                    response: final_resp,
                    history_messages: history,
                });
            }
            for (ev_name, payload) in end_frames {
                yield Ok(Event::default().event(ev_name).data(payload.to_string()));
            }
        };
        Ok(Sse::new(events)
            .keep_alive(KeepAlive::new().interval(Duration::from_secs(15)))
            .into_response())
    } else {
        let _admission = admission;
        let _active_guard = active_guard;
        let tokenizer = s.worker.tokenizer.as_ref().unwrap();
        let mut decoder = tokenizer.decode_stream(false);
        let mut parser = Qwen3Parser::new(prepared.thinking);
        let mut stops = StopBuffer::default();
        let mut generation = generation;
        let mut count = 0usize;
        let mut cached = 0usize;
        let mut done = false;
        let mut stopped = false;
        let mut last_token = Instant::now();

        while let Some(event) = generation.events.recv().await {
            match event {
                GenerationEvent::Token { id, .. } => {
                    count += 1;
                    s.metrics
                        .tokens_generated_total
                        .fetch_add(1, Ordering::Relaxed);
                    if count == 1 {
                        s.metrics
                            .last_ttft_us
                            .store(start.elapsed().as_micros() as u64, Ordering::Relaxed);
                    } else {
                        s.metrics
                            .last_itl_us
                            .store(last_token.elapsed().as_micros() as u64, Ordering::Relaxed);
                    }
                    last_token = Instant::now();
                    if count >= prepared.params.min_tokens
                        && prepared.params.is_stop_or_eos(id, &s.worker.eos_token_ids)
                    {
                        stopped = true;
                        continue;
                    }
                    let decoded = match decoder.step(id as u32) {
                        Ok(Some(t)) => t,
                        Ok(None) => continue,
                        Err(e) => return Err(error(StatusCode::INTERNAL_SERVER_ERROR, e)),
                    };
                    let active_stops = if count >= prepared.params.min_tokens {
                        prepared.params.stop_strings.as_slice()
                    } else {
                        &[]
                    };
                    let (visible, matched) = stops.push(&decoded, active_stops);
                    let _ = parser.push_str(&visible);
                    if matched {
                        stopped = true;
                        done = true;
                        break;
                    }
                }
                GenerationEvent::Finished { cached_tokens, .. } => {
                    cached = cached_tokens;
                    done = true;
                    break;
                }
                GenerationEvent::Error(message) => {
                    return Err(error(StatusCode::INTERNAL_SERVER_ERROR, message));
                }
            }
        }
        cached = cached.max(generation.cached_tokens.load(Ordering::Acquire) as usize);
        drop(generation);
        if !done {
            return Err(error(
                StatusCode::INTERNAL_SERVER_ERROR,
                "engine stream closed before completion",
            ));
        }
        let tail = std::mem::take(&mut stops.pending);
        if !tail.is_empty() {
            let _ = parser.push_str(&tail);
        }
        let (_, turn) = parser.finish();
        let finish = if !turn.tool_calls.is_empty() {
            "tool_calls"
        } else if !stopped && count >= prepared.params.max_tokens {
            "length"
        } else {
            "stop"
        };
        s.metrics
            .cached_prompt_tokens_total
            .fetch_add(cached as u64, Ordering::Relaxed);
        let reasoning_tokens = count_reasoning_tokens(tokenizer, &turn, count);
        let (resp, history) = crate::responses_api::build_response_object(
            &meta,
            &turn,
            finish,
            prompt_len,
            cached,
            count,
            reasoning_tokens,
        );
        if meta.store
            && let Some(scope) = caller_scope
        {
            s.response_store
                .insert(crate::responses_api::StoredResponse {
                    id: meta.response_id.clone(),
                    owner_scope: scope,
                    response: resp.clone(),
                    history_messages: history,
                });
        }
        Ok(Json(resp).into_response())
    }
}

async fn anthropic_messages(
    State(s): State<Arc<OpenAiServerState>>,
    request: Request,
) -> Result<Response, ApiError> {
    let admission = s
        .admission
        .clone()
        .try_acquire_owned()
        .map_err(|_| to_anthropic_err(busy()))?;
    let start = Instant::now();
    let body = read_json(&s, request).await.map_err(to_anthropic_err)?;
    let request_id = s.metrics.requests_total.fetch_add(1, Ordering::Relaxed) + 1;
    let (chat_req, meta) =
        crate::anthropic_api::translate_anthropic_to_chat(&body, &s.model_name, request_id, true)
            .map_err(|e| anthropic_api_err(StatusCode::BAD_REQUEST, e))?;
    let prep_state = s.clone();
    let prepared = cpu(&s, move || prepare(&prep_state, chat_req, true))
        .await
        .map_err(to_anthropic_err)?;
    let prompt_len = prepared.prompt.len();
    let generation = s
        .worker
        .submit_with_images(
            prepared.prompt,
            prepared.images,
            prepared.params.clone(),
            prepared.constraint,
        )
        .map_err(|e| {
            anthropic_api_err(
                StatusCode::SERVICE_UNAVAILABLE,
                crate::anthropic_api::AnthropicApiError::overloaded(e),
            )
        })?;
    s.metrics
        .prompt_tokens_total
        .fetch_add(prompt_len as u64, Ordering::Relaxed);

    if meta.stream {
        let params = prepared.params;
        let thinking = prepared.thinking;
        let events = async_stream::stream! {
            let _admission = admission;
            let tokenizer = s.worker.tokenizer.as_ref().unwrap();
            let mut decoder = tokenizer.decode_stream(false);
            let mut parser = Qwen3Parser::new(thinking);
            let mut stops = StopBuffer::default();
            let mut sse = crate::anthropic_api::AnthropicSseFormatter::new(meta, prompt_len);
            let mut generation = generation;
            let mut count = 0usize;
            let mut cached = 0usize;
            let mut done = false;
            let mut stopped = false;
            let mut matched_stop: Option<String> = None;
            let mut last_token = Instant::now();

            while let Some(event) = generation.events.recv().await {
                match event {
                    GenerationEvent::Token { id, .. } => {
                        count += 1;
                        s.metrics.tokens_generated_total.fetch_add(1, Ordering::Relaxed);
                        if count == 1 {
                            s.metrics.last_ttft_us.store(start.elapsed().as_micros() as u64, Ordering::Relaxed);
                        } else {
                            s.metrics.last_itl_us.store(last_token.elapsed().as_micros() as u64, Ordering::Relaxed);
                        }
                        last_token = Instant::now();
                        cached = cached.max(generation.cached_tokens.load(Ordering::Acquire) as usize);
                        for (ev_name, payload) in sse.ensure_started(cached) {
                            yield Ok::<_, Infallible>(Event::default().event(ev_name).data(payload.to_string()));
                        }
                        if count >= params.min_tokens && params.is_stop_or_eos(id, &s.worker.eos_token_ids) {
                            stopped = true;
                            continue;
                        }
                        let decoded = match decoder.step(id as u32) {
                            Ok(Some(t)) => t,
                            Ok(None) => continue,
                            Err(e) => {
                                let (ev_name, payload) = crate::anthropic_api::AnthropicSseFormatter::error_event(&e.to_string());
                                yield Ok(Event::default().event(ev_name).data(payload.to_string()));
                                return;
                            }
                        };
                        let active_stops = if count >= params.min_tokens { params.stop_strings.as_slice() } else { &[] };
                        let (visible, m_stop) = stops.push_matched(&decoded, active_stops);
                        for (ev_name, payload) in sse.push_delta(parser.push_str(&visible), cached) {
                            yield Ok(Event::default().event(ev_name).data(payload.to_string()));
                        }
                        if let Some(seq) = m_stop {
                            matched_stop = Some(seq);
                            stopped = true;
                            done = true;
                            break;
                        }
                    }
                    GenerationEvent::Finished { cached_tokens, .. } => {
                        cached = cached.max(cached_tokens);
                        done = true;
                        break;
                    }
                    GenerationEvent::Error(message) => {
                        let (ev_name, payload) = crate::anthropic_api::AnthropicSseFormatter::error_event(&message);
                        yield Ok(Event::default().event(ev_name).data(payload.to_string()));
                        return;
                    }
                }
            }
            cached = cached.max(generation.cached_tokens.load(Ordering::Acquire) as usize);
            drop(generation);
            if !done {
                let (ev_name, payload) = crate::anthropic_api::AnthropicSseFormatter::error_event(
                    "engine stream closed before completion",
                );
                yield Ok(Event::default().event(ev_name).data(payload.to_string()));
                return;
            }
            let tail = std::mem::take(&mut stops.pending);
            if !tail.is_empty() {
                for (ev_name, payload) in sse.push_delta(parser.push_str(&tail), cached) {
                    yield Ok(Event::default().event(ev_name).data(payload.to_string()));
                }
            }
            let (tail_delta, turn) = parser.finish();
            for (ev_name, payload) in sse.push_delta(tail_delta, cached) {
                yield Ok(Event::default().event(ev_name).data(payload.to_string()));
            }
            let finish = if !turn.tool_calls.is_empty() {
                "tool_calls"
            } else if !stopped && count >= params.max_tokens {
                "length"
            } else {
                "stop"
            };
            s.metrics.cached_prompt_tokens_total.fetch_add(cached as u64, Ordering::Relaxed);
            for (ev_name, payload) in sse.finish(&turn, finish, matched_stop.as_deref(), cached, count) {
                yield Ok(Event::default().event(ev_name).data(payload.to_string()));
            }
        };
        Ok(Sse::new(events)
            .keep_alive(KeepAlive::new().interval(Duration::from_secs(15)))
            .into_response())
    } else {
        let _admission = admission;
        let tokenizer = s.worker.tokenizer.as_ref().unwrap();
        let mut decoder = tokenizer.decode_stream(false);
        let mut parser = Qwen3Parser::new(prepared.thinking);
        let mut stops = StopBuffer::default();
        let mut generation = generation;
        let mut count = 0usize;
        let mut cached = 0usize;
        let mut done = false;
        let mut stopped = false;
        let mut matched_stop: Option<String> = None;
        let mut last_token = Instant::now();

        while let Some(event) = generation.events.recv().await {
            match event {
                GenerationEvent::Token { id, .. } => {
                    count += 1;
                    s.metrics
                        .tokens_generated_total
                        .fetch_add(1, Ordering::Relaxed);
                    if count == 1 {
                        s.metrics
                            .last_ttft_us
                            .store(start.elapsed().as_micros() as u64, Ordering::Relaxed);
                    } else {
                        s.metrics
                            .last_itl_us
                            .store(last_token.elapsed().as_micros() as u64, Ordering::Relaxed);
                    }
                    last_token = Instant::now();
                    if count >= prepared.params.min_tokens
                        && prepared.params.is_stop_or_eos(id, &s.worker.eos_token_ids)
                    {
                        stopped = true;
                        continue;
                    }
                    let decoded = match decoder.step(id as u32) {
                        Ok(Some(t)) => t,
                        Ok(None) => continue,
                        Err(e) => {
                            return Err(anthropic_api_err(
                                StatusCode::INTERNAL_SERVER_ERROR,
                                crate::anthropic_api::AnthropicApiError::api_error(e.to_string()),
                            ));
                        }
                    };
                    let active_stops = if count >= prepared.params.min_tokens {
                        prepared.params.stop_strings.as_slice()
                    } else {
                        &[]
                    };
                    let (visible, m_stop) = stops.push_matched(&decoded, active_stops);
                    let _ = parser.push_str(&visible);
                    if let Some(seq) = m_stop {
                        matched_stop = Some(seq);
                        stopped = true;
                        done = true;
                        break;
                    }
                }
                GenerationEvent::Finished { cached_tokens, .. } => {
                    cached = cached_tokens;
                    done = true;
                    break;
                }
                GenerationEvent::Error(message) => {
                    return Err(anthropic_api_err(
                        StatusCode::INTERNAL_SERVER_ERROR,
                        crate::anthropic_api::AnthropicApiError::api_error(message),
                    ));
                }
            }
        }
        cached = cached.max(generation.cached_tokens.load(Ordering::Acquire) as usize);
        drop(generation);
        if !done {
            return Err(anthropic_api_err(
                StatusCode::INTERNAL_SERVER_ERROR,
                crate::anthropic_api::AnthropicApiError::api_error(
                    "engine stream closed before completion",
                ),
            ));
        }
        let tail = std::mem::take(&mut stops.pending);
        if !tail.is_empty() {
            let _ = parser.push_str(&tail);
        }
        let (_, turn) = parser.finish();
        let finish = if !turn.tool_calls.is_empty() {
            "tool_calls"
        } else if !stopped && count >= prepared.params.max_tokens {
            "length"
        } else {
            "stop"
        };
        s.metrics
            .cached_prompt_tokens_total
            .fetch_add(cached as u64, Ordering::Relaxed);
        let msg = crate::anthropic_api::build_anthropic_message(
            &meta,
            &turn,
            finish,
            matched_stop.as_deref(),
            prompt_len,
            cached,
            count,
        );
        Ok(Json(msg).into_response())
    }
}

fn decision_err(err: crate::decision::DecisionError) -> ApiError {
    if err.is_bad_request {
        bad(err.message)
    } else {
        error(StatusCode::INTERNAL_SERVER_ERROR, err.message)
    }
}

fn shared_mamba_branch_hint<'a>(
    first: &[i32],
    others: impl IntoIterator<Item = &'a [i32]>,
    cap: Option<usize>,
) -> Option<usize> {
    let mut lcp = first.len();
    if let Some(c) = cap {
        lcp = lcp.min(c);
    }
    let mut has_other = false;
    for other in others {
        has_other = true;
        let common = first[..lcp]
            .iter()
            .zip(other.iter())
            .take_while(|(a, b)| a == b)
            .count();
        lcp = lcp.min(common);
    }
    if !has_other {
        return None;
    }
    let aligned = (lcp / 64) * 64;
    (aligned >= 64 && aligned < first.len()).then_some(aligned)
}

async fn run_one_candidate_query(
    s: Arc<OpenAiServerState>,
    prompt: Vec<i32>,
    images: Vec<crate::vision::PreprocessedImage>,
    cands: Vec<i32>,
    branch_hint: Option<usize>,
) -> Result<(Vec<f64>, usize), ApiError> {
    let (mut stream, rx) = s
        .worker
        .submit_decision_candidate_logits_with_images(prompt, images, cands, branch_hint)
        .map_err(|e| error(StatusCode::SERVICE_UNAVAILABLE, e))?;
    let tap = rx
        .await
        .map_err(|_| error(StatusCode::INTERNAL_SERVER_ERROR, "decision worker dropped"))?
        .map_err(|e| error(StatusCode::INTERNAL_SERVER_ERROR, e))?;
    let mut cached_done = 0usize;
    while let Some(ev) = stream.events.recv().await {
        match ev {
            GenerationEvent::Finished { cached_tokens, .. } => {
                cached_done = cached_tokens;
                break;
            }
            GenerationEvent::Error(msg) => {
                return Err(error(StatusCode::INTERNAL_SERVER_ERROR, msg));
            }
            GenerationEvent::Token { .. } => {}
        }
    }
    match tap {
        crate::engine_worker::DecisionTapOutput::CandidateLogits {
            logits,
            cached_tokens,
        } => Ok((logits, cached_tokens.max(cached_done))),
        crate::engine_worker::DecisionTapOutput::HiddenRows { .. } => Err(error(
            StatusCode::INTERNAL_SERVER_ERROR,
            "unexpected decision tap variant",
        )),
    }
}

async fn run_one_hidden_query(
    s: Arc<OpenAiServerState>,
    prompt: Vec<i32>,
    images: Vec<crate::vision::PreprocessedImage>,
    positions: Vec<usize>,
    prefix_cap: Option<usize>,
    branch_hint: Option<usize>,
) -> Result<(Vec<Vec<f32>>, usize), ApiError> {
    let (mut stream, rx) = s
        .worker
        .submit_decision_hidden_positions_with_images(
            prompt,
            images,
            positions,
            prefix_cap,
            branch_hint,
        )
        .map_err(|e| error(StatusCode::SERVICE_UNAVAILABLE, e))?;
    let tap = rx
        .await
        .map_err(|_| error(StatusCode::INTERNAL_SERVER_ERROR, "decision worker dropped"))?
        .map_err(|e| error(StatusCode::INTERNAL_SERVER_ERROR, e))?;
    let mut cached_done = 0usize;
    while let Some(ev) = stream.events.recv().await {
        match ev {
            GenerationEvent::Finished { cached_tokens, .. } => {
                cached_done = cached_tokens;
                break;
            }
            GenerationEvent::Error(msg) => {
                return Err(error(StatusCode::INTERNAL_SERVER_ERROR, msg));
            }
            GenerationEvent::Token { .. } => {}
        }
    }
    match tap {
        crate::engine_worker::DecisionTapOutput::HiddenRows {
            rows,
            cached_tokens,
        } => Ok((rows, cached_tokens.max(cached_done))),
        crate::engine_worker::DecisionTapOutput::CandidateLogits { .. } => Err(error(
            StatusCode::INTERNAL_SERVER_ERROR,
            "unexpected decision tap variant",
        )),
    }
}

async fn run_one_full_hidden_query(
    s: Arc<OpenAiServerState>,
    prompt: Vec<i32>,
    images: Vec<crate::vision::PreprocessedImage>,
) -> Result<(Vec<Vec<f32>>, usize), ApiError> {
    let (mut stream, rx) = s
        .worker
        .submit_decision_full_hidden_with_images(prompt, images, None)
        .map_err(|e| error(StatusCode::SERVICE_UNAVAILABLE, e))?;
    let tap = rx
        .await
        .map_err(|_| error(StatusCode::INTERNAL_SERVER_ERROR, "decision worker dropped"))?
        .map_err(|e| error(StatusCode::INTERNAL_SERVER_ERROR, e))?;
    let mut cached_done = 0usize;
    while let Some(ev) = stream.events.recv().await {
        match ev {
            GenerationEvent::Finished { cached_tokens, .. } => {
                cached_done = cached_tokens;
                break;
            }
            GenerationEvent::Error(msg) => {
                return Err(error(StatusCode::INTERNAL_SERVER_ERROR, msg));
            }
            GenerationEvent::Token { .. } => {}
        }
    }
    match tap {
        crate::engine_worker::DecisionTapOutput::HiddenRows {
            rows,
            cached_tokens,
        } => Ok((rows, cached_tokens.max(cached_done))),
        crate::engine_worker::DecisionTapOutput::CandidateLogits { .. } => Err(error(
            StatusCode::INTERNAL_SERVER_ERROR,
            "unexpected decision tap variant",
        )),
    }
}

async fn run_candidate_queries_radix_shared(
    s: &Arc<OpenAiServerState>,
    queries: Vec<(Vec<i32>, Vec<i32>)>,
    images: &[crate::vision::PreprocessedImage],
) -> Result<Vec<(Vec<f64>, usize)>, ApiError> {
    if queries.is_empty() {
        return Ok(Vec::new());
    }
    let mut results = Vec::with_capacity(queries.len());
    let mut iter = queries.into_iter();
    // Run Query 0 first to donate the shared state prefix into MambaRadixCache.
    let (p0, c0) = iter.next().unwrap();
    let rest: Vec<_> = iter.collect();
    let branch0 = shared_mamba_branch_hint(&p0, rest.iter().map(|(p, _)| p.as_slice()), None);
    results.push(run_one_candidate_query(s.clone(), p0, images.to_vec(), c0, branch0).await?);

    for wave in rest.chunks(8) {
        let mut handles = Vec::with_capacity(wave.len());
        for (p, c) in wave.iter().cloned() {
            let st = s.clone();
            let imgs = images.to_vec();
            handles.push(tokio::spawn(async move {
                run_one_candidate_query(st, p, imgs, c, None).await
            }));
        }
        for h in handles {
            let out = h
                .await
                .map_err(|e| error(StatusCode::INTERNAL_SERVER_ERROR, e))??;
            results.push(out);
        }
    }
    Ok(results)
}

async fn execute_systemone_json(
    s: &Arc<OpenAiServerState>,
    body: Value,
) -> Result<Value, ApiError> {
    let prep_state = s.clone();
    let (req_model, batch, images) = cpu(s, move || {
        Ok::<_, String>((|| {
            let Some(tokenizer) = prep_state.worker.tokenizer.as_ref() else {
                return Err(bad("loaded model has no tokenizer"));
            };
            let vctx = crate::decision::DecisionVisionContext {
                vision_config: prep_state.worker.vision_config.clone(),
                has_vision_weights: prep_state.worker.has_vision_weights,
                allow_remote_images: prep_state.allow_remote_images,
            };
            let images = crate::decision::parse_decision_images(
                &body,
                &prep_state.worker.decision_profile,
                &vctx,
            )
            .map_err(decision_err)?;
            let (req_model, batch) = crate::decision::prepare_decision_batch_with_images(
                &prep_state.worker.decision_profile,
                tokenizer,
                prep_state.worker.kv_token_capacity,
                &body,
                prep_state.worker.vision_config.as_ref(),
                &images,
            )
            .map_err(decision_err)?;
            Ok((req_model, batch, images))
        })())
    })
    .await??;

    s.metrics.requests_total.fetch_add(1, Ordering::Relaxed);
    let mut answers = serde_json::Map::new();
    let mut input_tokens = 0usize;
    let mut cached_tokens = 0usize;

    match batch {
        crate::decision::PreparedDecisionBatch::Clef {
            head,
            fields,
            record,
        } => {
            input_tokens = record.input_ids.len();
            let (hidden_rows, cached) =
                run_one_full_hidden_query(s.clone(), record.input_ids.clone(), images).await?;
            cached_tokens += cached;
            let (questions, all_logits) = cpu(s, move || {
                Ok::<_, String>(
                    head.forward(&hidden_rows, &record.input_ids, &record.questions)
                        .map(|logits| (record.questions, logits)),
                )
            })
            .await?
            .map_err(|e| error(StatusCode::INTERNAL_SERVER_ERROR, e))?;
            for (field, (q_enc, raw_logits)) in
                fields.iter().zip(questions.iter().zip(all_logits.iter()))
            {
                let ordered = crate::clef::reorder_clef_question_logits(q_enc, field, raw_logits)
                    .map_err(decision_err)?;
                let ans = crate::decision::answer_from_logits(
                    field,
                    &ordered,
                    1.0,
                    crate::decision::ConfidenceKind::MaxProbability,
                )
                .map_err(decision_err)?;
                answers.insert(field.name.clone(), ans);
            }
        }
        crate::decision::PreparedDecisionBatch::CandidateLogits {
            temperature,
            conf_kind,
            fields,
            prompts,
            candidates,
            ..
        } => {
            for p in &prompts {
                input_tokens += p.len();
            }
            let pairs: Vec<_> = prompts.into_iter().zip(candidates).collect();
            let outs = run_candidate_queries_radix_shared(s, pairs, &images).await?;
            for (field, (logits, cached)) in fields.iter().zip(outs) {
                cached_tokens += cached;
                let ans =
                    crate::decision::answer_from_logits(field, &logits, temperature, conf_kind)
                        .map_err(decision_err)?;
                answers.insert(field.name.clone(), ans);
            }
        }
        crate::decision::PreparedDecisionBatch::Kev {
            head,
            temperature,
            fields,
            encoded,
        } => {
            let mut queries = Vec::with_capacity(encoded.len());
            for q_enc in &encoded {
                input_tokens += q_enc.token_ids.len();
                let mut positions = Vec::with_capacity(1 + q_enc.option_indices.len());
                positions.push(q_enc.decide_index);
                positions.extend_from_slice(&q_enc.option_indices);
                let min_pos = positions.iter().copied().min().unwrap_or(0);
                queries.push((q_enc.token_ids.clone(), positions, Some(min_pos)));
            }
            let mut outs = Vec::with_capacity(queries.len());
            let mut iter = queries.into_iter();
            if let Some((p0, pos0, cap0)) = iter.next() {
                let rest: Vec<_> = iter.collect();
                let branch0 =
                    shared_mamba_branch_hint(&p0, rest.iter().map(|(p, _, _)| p.as_slice()), cap0);
                outs.push(
                    run_one_hidden_query(s.clone(), p0, images.clone(), pos0, cap0, branch0)
                        .await?,
                );
                for wave in rest.chunks(8) {
                    let mut handles = Vec::with_capacity(wave.len());
                    for (p, pos, cap) in wave.iter().cloned() {
                        let st = s.clone();
                        let imgs = images.clone();
                        handles.push(tokio::spawn(async move {
                            run_one_hidden_query(st, p, imgs, pos, cap, None).await
                        }));
                    }
                    for h in handles {
                        let out = h
                            .await
                            .map_err(|e| error(StatusCode::INTERNAL_SERVER_ERROR, e))??;
                        outs.push(out);
                    }
                }
            }
            for (field, (rows, cached)) in fields.iter().zip(outs) {
                cached_tokens += cached;
                let n_opts = field.keys.len();
                let h_decide = &rows[0];
                let h_opts: Vec<f32> = rows[1..].iter().flatten().copied().collect();
                let raw_logits = head
                    .forward(h_decide, &h_opts, n_opts)
                    .map_err(|e| error(StatusCode::INTERNAL_SERVER_ERROR, e))?;
                let logits_f64: Vec<f64> = raw_logits.into_iter().map(|x| x as f64).collect();
                let ans = crate::decision::answer_from_logits(
                    field,
                    &logits_f64,
                    temperature,
                    crate::decision::ConfidenceKind::MarginAndModeDistance,
                )
                .map_err(decision_err)?;
                answers.insert(field.name.clone(), ans);
            }
        }
        crate::decision::PreparedDecisionBatch::Xor {
            temperature,
            lev_noul_9_scale,
            fields,
            variants,
        } => {
            let mut pairs = Vec::with_capacity(variants.len() * 2);
            for (fwd, rev, cands) in variants {
                input_tokens += fwd.len() + rev.len();
                pairs.push((fwd, cands.clone()));
                pairs.push((rev, cands));
            }
            let outs = run_candidate_queries_radix_shared(s, pairs, &images).await?;
            for (i, field) in fields.iter().enumerate() {
                let (fwd_logits, c_fwd) = &outs[2 * i];
                let (rev_logits, c_rev) = &outs[2 * i + 1];
                cached_tokens += *c_fwd + *c_rev;
                let probs = crate::decision::calibrate_forward_reverse_logits(
                    fwd_logits,
                    rev_logits,
                    temperature,
                )
                .map_err(decision_err)?;
                let ans = if lev_noul_9_scale && field.field_type == "noul" {
                    let expected: f64 = probs
                        .iter()
                        .enumerate()
                        .map(|(idx, &p)| p * (idx as f64) / 8.0)
                        .sum();
                    json!({
                        "type": "noul",
                        "noul": expected,
                    })
                } else {
                    crate::decision::answer_from_probabilities(
                        field,
                        &probs,
                        crate::decision::ConfidenceKind::MarginAndModeDistance,
                    )
                    .map_err(decision_err)?
                };
                answers.insert(field.name.clone(), ans);
            }
        }
    }

    s.metrics
        .prompt_tokens_total
        .fetch_add(input_tokens as u64, Ordering::Relaxed);
    s.metrics
        .cached_prompt_tokens_total
        .fetch_add(cached_tokens as u64, Ordering::Relaxed);

    Ok(json!({
        "model": req_model.unwrap_or_else(|| s.model_name.clone()),
        "answers": Value::Object(answers),
        "usage": {
            "input_tokens": input_tokens,
            "output_tokens": 0,
            "cached_tokens": cached_tokens,
        }
    }))
}

async fn systemone_decide(
    State(s): State<Arc<OpenAiServerState>>,
    request: Request,
) -> Result<Json<Value>, ApiError> {
    let _admission = s
        .admission
        .clone()
        .try_acquire_owned()
        .map_err(|_| busy())?;
    let body = read_json(&s, request).await?;
    let resp = execute_systemone_json(&s, body).await?;
    Ok(Json(resp))
}

async fn decisions_adapter(
    State(s): State<Arc<OpenAiServerState>>,
    request: Request,
) -> Result<Json<Value>, ApiError> {
    let _admission = s
        .admission
        .clone()
        .try_acquire_owned()
        .map_err(|_| busy())?;
    let body = read_json(&s, request).await?;
    let canonical =
        crate::decision::normalize_decisions_adapter_request(&body).map_err(decision_err)?;
    let resp = execute_systemone_json(&s, canonical).await?;
    Ok(Json(crate::decision::format_decisions_adapter_response(
        &resp,
    )))
}

pub async fn serve(
    listener: tokio::net::TcpListener,
    state: Arc<OpenAiServerState>,
    shutdown: impl std::future::Future<Output = ()> + Send + 'static,
) -> std::io::Result<()> {
    use axum::serve::ListenerExt;
    use std::future::IntoFuture;
    let (stop_tx, stop_rx) = tokio::sync::oneshot::channel();
    let listener = listener.tap_io(|tcp_stream| {
        let _ = tcp_stream.set_nodelay(true);
    });
    let mut serving = Box::pin(
        axum::serve(listener, router(state.clone()))
            .with_graceful_shutdown(async {
                let _ = stop_rx.await;
            })
            .into_future(),
    );
    let result = tokio::select! {
        result = &mut serving => result,
        _ = shutdown => {
            state.admission.close();
            state.preprocessing.close();
            // Existing generations can complete while HTTP drains. shutdown() below
            // cancels remaining work only once draining finishes or hits its deadline.
            let _ = stop_tx.send(());
            tokio::time::timeout(Duration::from_secs(30), &mut serving).await
                .map_err(|_| std::io::Error::other("HTTP graceful shutdown deadline exceeded"))
                .and_then(|result| result)
        }
    };
    state
        .worker
        .shutdown()
        .await
        .map_err(std::io::Error::other)?;
    result
}
pub fn run_blocking_server(addr: &str, state: Arc<OpenAiServerState>) -> std::io::Result<()> {
    let runtime = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()?;
    runtime.block_on(async {
        let listener = match tokio::net::TcpListener::bind(addr).await {
            Ok(listener) => listener,
            Err(error) => {
                state
                    .worker
                    .shutdown()
                    .await
                    .map_err(std::io::Error::other)?;
                return Err(error);
            }
        };
        eprintln!("[w34 serve] ready on http://{}", listener.local_addr()?);
        serve(listener, state, async {
            #[cfg(unix)]
            {
                let mut terminate =
                    tokio::signal::unix::signal(tokio::signal::unix::SignalKind::terminate())
                        .expect("SIGTERM handler");
                tokio::select! { _ = tokio::signal::ctrl_c() => {}, _ = terminate.recv() => {} }
            }
            #[cfg(not(unix))]
            {
                let _ = tokio::signal::ctrl_c().await;
            }
        })
        .await
    })
}

#[cfg(test)]
mod tests {
    use super::StopBuffer;
    #[test]
    fn stop_holdback_handles_every_utf8_split_and_overlaps() {
        let input = "hello 🛑END trailing";
        for (split, _) in input.char_indices() {
            let mut b = StopBuffer::default();
            let stops = vec!["🛑END".to_string(), "ENDING".to_string()];
            let (mut visible, done) = b.push(&input[..split], &stops);
            if !done {
                let (tail, done) = b.push(&input[split..], &stops);
                visible.push_str(&tail);
                assert!(done);
            }
            assert_eq!(visible, "hello ");
        }
    }
}
