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
    extract::{Request, State},
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
    constraint_factory: Option<crate::constraints::ConstraintFactory>,
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
        Arc::new(Self {
            model_name,
            worker,
            allow_remote_images,
            constraint_factory,
            metrics: ServerMetrics::default(),
            preprocessing: Arc::new(Semaphore::new(8)),
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
        .route("/v1/completions", post(completion))
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
    let bytes = tokio::time::timeout(BODY_TIMEOUT, to_bytes(req.into_body(), BODY_LIMIT))
        .await
        .map_err(|_| {
            error(
                StatusCode::REQUEST_TIMEOUT,
                "request body deadline exceeded",
            )
        })?
        .map_err(|e| error(StatusCode::PAYLOAD_TOO_LARGE, e))?;
    cpu(state, move || {
        serde_json::from_slice(&bytes).map_err(|e| e.to_string())
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
    let tokenizer = s
        .worker
        .tokenizer
        .clone()
        .ok_or_else(|| bad("tokenizer unavailable"))?;
    cpu(&s, move || {
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
    let tokenizer = s
        .worker
        .tokenizer
        .clone()
        .ok_or_else(|| bad("tokenizer unavailable"))?;
    cpu(&s, move || {
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
            .pointer("/chat_template_kwargs/preserve_thinking")
            .and_then(Value::as_bool)
            .unwrap_or(false);
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
    let json = read_json(&s, request).await?;
    let prep_state = s.clone();
    let prepared = cpu(&s, move || prepare(&prep_state, json, chat)).await?;
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
                    Output::Error(message) => { yield Ok(Event::default().data(json!({"error":{"message":message,"type":"server_error"}}).to_string())); }
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
        self.pending.push_str(chunk);
        if let Some(pos) = stops.iter().filter_map(|s| self.pending.find(s)).min() {
            let text = self.pending[..pos].to_owned();
            self.pending.clear();
            return (text, true);
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
        (std::mem::replace(&mut self.pending, rest), false)
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
fn logprob_value(lp: &StepLogprobs, tokenizer: &tokenizers::Tokenizer) -> Value {
    let token = |id: i32| tokenizer.decode(&[id as u32], false).unwrap_or_default();
    let text = token(lp.token_id);
    json!({"token":text,"logprob":lp.logprob,"bytes":text.as_bytes(),"top_logprobs":lp.top_logprobs.iter().map(|p| { let text = token(p.token_id); json!({"token":text,"logprob":p.logprob,"bytes":text.as_bytes()}) }).collect::<Vec<_>>()})
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
        let tokenizer = s.worker.tokenizer.clone().unwrap();
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
                    let mut lp = logprobs.as_ref().map(|lp| logprob_value(lp,&tokenizer));
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

pub async fn serve(
    listener: tokio::net::TcpListener,
    state: Arc<OpenAiServerState>,
    shutdown: impl std::future::Future<Output = ()> + Send + 'static,
) -> std::io::Result<()> {
    use std::future::IntoFuture;
    let (stop_tx, stop_rx) = tokio::sync::oneshot::channel();
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
