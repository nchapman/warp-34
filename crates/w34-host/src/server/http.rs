//! Axum owns transport; only EngineWorker's OS thread owns CUDA objects.
use crate::{
    engine_worker::{EngineWorker, GenerationEvent, GenerationStream, WorkerError},
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
    time::Duration,
};
use tokio::sync::{OwnedSemaphorePermit, Semaphore};
use tokio_stream::{Stream, StreamExt};

const BODY_LIMIT: usize = 32 * 1024 * 1024;
const BODY_TIMEOUT: Duration = Duration::from_secs(10);

pub struct OpenAiServerState {
    pub model_name: String,
    api_key: Option<crate::api_auth::ApiKey>,
    pub worker: EngineWorker,
    pub metrics: Arc<crate::telemetry::Telemetry>,
    next_request_id: AtomicU64,
    pub allow_remote_images: bool,
    pub response_store: crate::responses_api::ResponseStore,
    constraint_factory: Option<crate::constraints::ConstraintFactory>,
    decoded_vocab: Vec<std::sync::OnceLock<Box<str>>>,
    pub(super) preprocessing: Arc<Semaphore>,
    pub(super) admission: Arc<Semaphore>,
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
        Self::from_loader_with_options(
            model_name,
            allow_remote_images,
            super::ServeOptions::default(),
            loader,
        )
    }
    pub fn from_loader_with_options(
        model_name: String,
        allow_remote_images: bool,
        options: super::ServeOptions,
        loader: impl FnOnce() -> Result<Qwen3_5Engine, String> + Send + 'static,
    ) -> Result<Arc<Self>, String> {
        options.validate()?;
        let config = SchedulerConfig {
            max_num_seqs: options.max_num_seqs,
            ..SchedulerConfig::default()
        };
        let mut worker = EngineWorker::from_loader(loader, config)?;
        worker.set_max_model_len(options.max_model_len)?;
        let mut state = Self::with_worker(model_name, worker, allow_remote_images);
        // This newly created state has not been shared with any request handler.
        Arc::get_mut(&mut state)
            .expect("unshared server state")
            .api_key = options.api_key;
        Ok(state)
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
        let metrics = worker.stats.telemetry.clone();
        metrics.set_model(model_name.clone());
        Arc::new(Self {
            model_name,
            api_key: None,
            worker,
            allow_remote_images,
            response_store: crate::responses_api::ResponseStore::default(),
            constraint_factory,
            decoded_vocab,
            metrics,
            next_request_id: AtomicU64::new(0),
            preprocessing: Arc::new(Semaphore::new(32)),
            admission: Arc::new(Semaphore::new(32)),
        })
    }
}

pub(super) type ApiError = (StatusCode, Json<Value>);
pub(super) fn error(status: StatusCode, message: impl ToString) -> ApiError {
    let err_type = if status == StatusCode::TOO_MANY_REQUESTS {
        "rate_limit_error"
    } else if status.is_server_error() {
        "server_error"
    } else {
        "invalid_request_error"
    };
    (
        status,
        Json(json!({"error":{"message":message.to_string(),"type":err_type}})),
    )
}
pub(super) fn worker_api_err(err: WorkerError) -> ApiError {
    (
        err.http_status(),
        Json(json!({"error":{"message":err.to_string(),"type":err.openai_error_type()}})),
    )
}
pub(super) fn bad(message: impl ToString) -> ApiError {
    error(StatusCode::BAD_REQUEST, message)
}
fn busy() -> ApiError {
    error(
        StatusCode::SERVICE_UNAVAILABLE,
        "server at capacity or shutting down",
    )
}

pub(super) fn admission_error(err: tokio::sync::TryAcquireError) -> ApiError {
    match err {
        tokio::sync::TryAcquireError::NoPermits => {
            worker_api_err(WorkerError::overload("server admission capacity exceeded"))
        }
        tokio::sync::TryAcquireError::Closed => {
            worker_api_err(WorkerError::worker_stopped("server shutting down"))
        }
    }
}

async fn health(State(s): State<Arc<OpenAiServerState>>) -> Response {
    if s.worker.is_panicked() {
        let detail = s
            .worker
            .panic_reason()
            .unwrap_or_else(|| "engine worker panicked".into());
        return (
            StatusCode::SERVICE_UNAVAILABLE,
            Json(json!({"status":"error","error":format!("engine worker terminated after panic: {detail}")})),
        )
            .into_response();
    }
    Json(json!({"status":"ok"})).into_response()
}

pub fn router(state: Arc<OpenAiServerState>) -> Router {
    Router::new()
        .route("/v1/models", get(models))
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
        // Protect every inference, tokenization, model, and stored-response route,
        // including unversioned aliases, before reading bodies or fetching media.
        .route_layer(axum::middleware::from_fn_with_state(
            state.api_key.clone(),
            crate::api_auth::authenticate,
        ))
        .route("/health", get(health))
        .route("/metrics", get(metrics))
        .route("/api/metrics", get(metrics_json))
        .merge(crate::dashboard::router())
        .layer(axum::middleware::from_fn_with_state(
            state.clone(),
            count_inference_responses,
        ))
        .fallback(|| async { error(StatusCode::NOT_FOUND, "not found") })
        .with_state(state)
}
async fn models(State(s): State<Arc<OpenAiServerState>>) -> Json<Value> {
    Json(
        json!({"object":"list","data":[{"id":s.model_name,"object":"model","owned_by":"warp-34"}]}),
    )
}
async fn metrics(State(s): State<Arc<OpenAiServerState>>) -> impl IntoResponse {
    (
        [(
            axum::http::header::CONTENT_TYPE,
            "text/plain; version=0.0.4",
        )],
        s.metrics.snapshot().prometheus(),
    )
}
async fn metrics_json(State(s): State<Arc<OpenAiServerState>>) -> impl IntoResponse {
    (
        [(axum::http::header::CACHE_CONTROL, "no-store")],
        Json(s.metrics.snapshot()),
    )
}
fn is_inference_endpoint(method: &axum::http::Method, path: &str) -> bool {
    *method == axum::http::Method::POST
        && matches!(
            path,
            "/v1/chat/completions"
                | "/v1/completions"
                | "/v1/responses"
                | "/v1/messages"
                | "/v1/systemone"
                | "/systemone"
                | "/v1/decisions"
                | "/decisions"
        )
}
async fn count_inference_responses(
    State(s): State<Arc<OpenAiServerState>>,
    request: Request,
    next: axum::middleware::Next,
) -> Response {
    let count = is_inference_endpoint(request.method(), request.uri().path());
    let response = next.run(request).await;
    if count {
        s.metrics.http_response(response.status().as_u16());
    }
    response
}
pub(super) async fn read_json(state: &OpenAiServerState, req: Request) -> Result<Value, ApiError> {
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
pub(super) async fn cpu<T: Send + 'static>(
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
        .map_err(admission_error)?;
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
        .map_err(admission_error)?;
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
    tools: Option<Value>,
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
    params.eos_token_ids.clone_from(&s.worker.eos_token_ids);
    if params
        .stop_token_ids
        .iter()
        .chain(params.logit_bias.keys())
        .any(|&id| id < 0 || id as usize >= s.worker.vocab_size)
    {
        return Err("stop_token_ids/logit_bias token outside model vocabulary".into());
    }
    let thinking = if chat {
        super::resolve_chat_enable_thinking(&request)?
    } else {
        false
    };
    params.enable_thinking = thinking;
    let tools_opt = request.get("tools").filter(|v| !v.is_null()).cloned();
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
        let tool_format = crate::request_constraints::parse_request_tool_format(&request)?;
        (
            crate::chatml::format_chatml_prompt_with_tool_format(
                &messages,
                tools_opt.as_ref(),
                thinking,
                preserve,
                tool_format,
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
        tools: tools_opt,
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
        .map_err(admission_error)?;
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
        .map_err(worker_api_err)?;
    let stream = prepared.stream;
    let include_usage = prepared.include_usage;
    let request_id = s.next_request_id.fetch_add(1, Ordering::Relaxed) + 1;
    let output = output_stream(
        s,
        generation,
        prepared.params,
        prepared.thinking,
        prepared.tools,
        chat,
        prompt_len,
        request_id,
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
                    Output::Error(err) => {
                        yield Ok(Event::default().data(
                            json!({"error":{"message":err.to_string(),"type":err.openai_error_type()}}).to_string(),
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
                Output::Error(err) => {
                    return Err(worker_api_err(err));
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
    Error(WorkerError),
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
    tools: Option<Value>,
    chat: bool,
    prompt_len: usize,
    request_id: u64,
    admission: OwnedSemaphorePermit,
) -> impl Stream<Item = Output> + Send {
    async_stream::stream! {
        let _admission = admission;
        let tokenizer = s.worker.tokenizer.as_ref().unwrap();
        let mut decoder = tokenizer.decode_stream(false);
        let mut parser = Qwen3Parser::new(thinking).with_tools(tools.as_ref());
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
                        Err(e) => { yield Output::Error(WorkerError::internal(e.to_string())); return; }
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
                GenerationEvent::Error(err) => { yield Output::Error(err); return; }
            }
        }
        cached = cached.max(generation.cached_tokens.load(Ordering::Acquire) as usize);
        drop(generation); // cancellation also covers stop strings and a dropped HTTP response body
        if !done { yield Output::Error(WorkerError::internal("engine stream closed before completion (overload, shutdown, or worker failure)")); return; }
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
        .map_err(admission_error)?;
    let mut body = read_json(&s, req).await?;
    if let Some(obj) = body.as_object_mut() {
        obj.remove("max_tokens");
        obj.remove("max_completion_tokens");
        obj.remove("min_tokens");
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
        .map_err(admission_error)?;
    let caller_scope = responses_caller_scope(&s, req.headers());
    let mut body = read_json(&s, req).await?;
    if let Some(obj) = body.as_object_mut() {
        obj.remove("max_output_tokens");
        obj.remove("max_tokens");
        obj.remove("min_tokens");
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
        .map_err(|err| to_anthropic_err(admission_error(err)))?;
    let body = read_json(&s, req).await.map_err(to_anthropic_err)?;
    let (mut chat_req, _) =
        crate::anthropic_api::translate_anthropic_to_chat(&body, &s.model_name, 0, false)
            .map_err(|e| anthropic_api_err(StatusCode::BAD_REQUEST, e))?;
    if let Some(obj) = chat_req.as_object_mut() {
        obj.insert("max_tokens".into(), json!(256));
        obj.remove("min_tokens");
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
        .map_err(admission_error)?;
    let caller_scope = responses_caller_scope(&s, request.headers());
    let body = read_json(&s, request).await?;
    let request_id = s.next_request_id.fetch_add(1, Ordering::Relaxed) + 1;
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
        .map_err(worker_api_err)?;
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

    if meta.stream {
        let params = prepared.params;
        let thinking = prepared.thinking;
        let tools = prepared.tools;
        let events = async_stream::stream! {
            let _admission = admission;
            let _active_guard = active_guard;
            let tokenizer = s.worker.tokenizer.as_ref().unwrap();
            let mut decoder = tokenizer.decode_stream(false);
            let mut parser = Qwen3Parser::new(thinking).with_tools(tools.as_ref());
            let mut stops = StopBuffer::default();
            let mut sse = crate::responses_api::ResponsesSseFormatter::new(meta.clone());
            let mut generation = generation;
            let mut count = 0usize;
            let mut cached = 0usize;
            let mut done = false;
            let mut stopped = false;

            for (ev_name, payload) in sse.start() {
                yield Ok::<_, Infallible>(Event::default().event(ev_name).data(payload.to_string()));
            }

            while let Some(event) = generation.events.recv().await {
                match event {
                    GenerationEvent::Token { id, .. } => {
                        count += 1;
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
                    GenerationEvent::Error(err) => {
                        let msg = err.to_string();
                        for (ev_name, payload) in sse.fail(&msg) {
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
        let mut parser = Qwen3Parser::new(prepared.thinking).with_tools(prepared.tools.as_ref());
        let mut stops = StopBuffer::default();
        let mut generation = generation;
        let mut count = 0usize;
        let mut cached = 0usize;
        let mut done = false;
        let mut stopped = false;

        while let Some(event) = generation.events.recv().await {
            match event {
                GenerationEvent::Token { id, .. } => {
                    count += 1;
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
                GenerationEvent::Error(err) => {
                    return Err(worker_api_err(err));
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
        .map_err(|err| to_anthropic_err(admission_error(err)))?;
    let body = read_json(&s, request).await.map_err(to_anthropic_err)?;
    let request_id = s.next_request_id.fetch_add(1, Ordering::Relaxed) + 1;
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
        .map_err(|e| to_anthropic_err(worker_api_err(e)))?;

    if meta.stream {
        let params = prepared.params;
        let thinking = prepared.thinking;
        let tools = prepared.tools;
        let events = async_stream::stream! {
            let _admission = admission;
            let tokenizer = s.worker.tokenizer.as_ref().unwrap();
            let mut decoder = tokenizer.decode_stream(false);
            let mut parser = Qwen3Parser::new(thinking).with_tools(tools.as_ref());
            let mut stops = StopBuffer::default();
            let mut sse = crate::anthropic_api::AnthropicSseFormatter::new(meta, prompt_len);
            let mut generation = generation;
            let mut count = 0usize;
            let mut cached = 0usize;
            let mut done = false;
            let mut stopped = false;
            let mut matched_stop: Option<String> = None;

            while let Some(event) = generation.events.recv().await {
                match event {
                    GenerationEvent::Token { id, .. } => {
                        count += 1;
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
                    GenerationEvent::Error(err) => {
                        let msg = err.to_string();
                        let (ev_name, payload) = crate::anthropic_api::AnthropicSseFormatter::error_event(&msg);
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
        let mut parser = Qwen3Parser::new(prepared.thinking).with_tools(prepared.tools.as_ref());
        let mut stops = StopBuffer::default();
        let mut generation = generation;
        let mut count = 0usize;
        let mut cached = 0usize;
        let mut done = false;
        let mut stopped = false;
        let mut matched_stop: Option<String> = None;

        while let Some(event) = generation.events.recv().await {
            match event {
                GenerationEvent::Token { id, .. } => {
                    count += 1;
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
                GenerationEvent::Error(err) => {
                    return Err(to_anthropic_err(worker_api_err(err)));
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

use super::decision_http::{decisions_adapter, systemone_decide};

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
            state.metrics.set_status("stopping");
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
    use super::{StopBuffer, admission_error, to_anthropic_err, worker_api_err};
    use crate::engine_worker::WorkerError;
    use axum::http::StatusCode;

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

    #[test]
    fn telemetry_http_filter_excludes_scrapes_status_static_and_token_counts() {
        use axum::http::Method;
        for path in [
            "/v1/chat/completions",
            "/v1/completions",
            "/v1/responses",
            "/v1/messages",
            "/systemone",
            "/v1/systemone",
            "/decisions",
            "/v1/decisions",
        ] {
            assert!(super::is_inference_endpoint(&Method::POST, path));
            assert!(!super::is_inference_endpoint(&Method::GET, path));
        }
        for path in [
            "/",
            "/metrics",
            "/api/metrics",
            "/health",
            "/assets/app.js",
            "/tokenize",
            "/v1/messages/count_tokens",
            "/v1/responses/input_tokens",
            "/v1/responses/123/cancel",
        ] {
            assert!(!super::is_inference_endpoint(&Method::POST, path));
            assert!(!super::is_inference_endpoint(&Method::GET, path));
        }
    }

    #[test]
    fn admission_distinguishes_capacity_from_shutdown() {
        let (status, body) = admission_error(tokio::sync::TryAcquireError::NoPermits);
        assert_eq!(status, StatusCode::TOO_MANY_REQUESTS);
        assert_eq!(body.0["error"]["type"], "rate_limit_error");
        let (status, body) = admission_error(tokio::sync::TryAcquireError::Closed);
        assert_eq!(status, StatusCode::SERVICE_UNAVAILABLE);
        assert_eq!(body.0["error"]["type"], "server_error");
    }

    #[test]
    fn worker_api_err_maps_openai_and_anthropic_status_codes_and_types() {
        let (st, body) =
            worker_api_err(WorkerError::overload("engine admission capacity exceeded"));
        assert_eq!(st, StatusCode::TOO_MANY_REQUESTS);
        assert_eq!(body.0["error"]["type"], "rate_limit_error");

        let (ast, abody) = to_anthropic_err((st, body));
        assert_eq!(ast, StatusCode::TOO_MANY_REQUESTS);
        assert_eq!(abody.0["error"]["type"], "overloaded_error");

        let (st, body) = worker_api_err(WorkerError::invalid_request(
            "invalid tokens or request exceeds KV capacity",
        ));
        assert_eq!(st, StatusCode::BAD_REQUEST);
        assert_eq!(body.0["error"]["type"], "invalid_request_error");

        let (ast, abody) = to_anthropic_err((st, body));
        assert_eq!(ast, StatusCode::BAD_REQUEST);
        assert_eq!(abody.0["error"]["type"], "invalid_request_error");

        let (st, body) = worker_api_err(WorkerError::worker_panicked("boom"));
        assert_eq!(st, StatusCode::INTERNAL_SERVER_ERROR);
        assert_eq!(body.0["error"]["type"], "server_error");

        let (ast, abody) = to_anthropic_err((st, body));
        assert_eq!(ast, StatusCode::INTERNAL_SERVER_ERROR);
        assert_eq!(abody.0["error"]["type"], "api_error");
    }
}
