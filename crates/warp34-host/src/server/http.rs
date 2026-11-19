//! Axum owns transport; only EngineWorker's OS thread owns CUDA objects.
use super::request_log::TurnLog;
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
    webui: bool,
    pub default_reasoning_effort: Option<String>,
    pub default_reasoning_budget: Option<usize>,
    pub default_reasoning_budget_message: Option<String>,
    pub reasoning_effort_budget: bool,
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
        let unshared = Arc::get_mut(&mut state).expect("unshared server state");
        unshared.api_key = options.api_key;
        unshared.webui = options.webui;
        unshared.default_reasoning_effort = options.reasoning_effort;
        unshared.default_reasoning_budget = options.reasoning_budget;
        unshared.default_reasoning_budget_message = options.reasoning_budget_message;
        unshared.reasoning_effort_budget = options.reasoning_effort_budget;
        Ok(state)
    }
    fn with_worker(
        model_name: String,
        worker: EngineWorker,
        allow_remote_images: bool,
    ) -> Arc<Self> {
        let constraint_factory = worker.tokenizer.as_ref().and_then(|t| {
            let eos: Vec<u32> = worker
                .eos_token_ids
                .iter()
                .filter_map(|&id| u32::try_from(id).ok())
                .filter(|&id| t.id_to_token(id).is_some())
                .collect();
            match crate::constraints::ConstraintFactory::new(t, worker.vocab_size, &eos) {
                Ok(factory) => Some(factory),
                Err(err) => {
                    eprintln!("warning: structured output constraints disabled: {err}");
                    None
                }
            }
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
            webui: true,
            default_reasoning_effort: None,
            default_reasoning_budget: None,
            default_reasoning_budget_message: None,
            reasoning_effort_budget: true,
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

    pub fn resolve_request_lora(
        &self,
        req_model: Option<&str>,
        explicit_lora: Option<&str>,
    ) -> Result<(i32, String), String> {
        if let Some(lora_name) = explicit_lora.filter(|s| !s.is_empty()) {
            let id = self
                .worker
                .lora_registry
                .resolve_name(lora_name)
                .ok_or_else(|| format!("unknown LoRA adapter '{lora_name}'"))?;
            return Ok((id, lora_name.to_owned()));
        }
        if let Some(m) = req_model.filter(|s| !s.is_empty())
            && let Some(id) = self.worker.lora_registry.resolve_name(m)
        {
            return Ok((id, m.to_owned()));
        }
        Ok((crate::lora::BASE_MODEL_LORA_ID, self.model_name.clone()))
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
    let mut obj = serde_json::Map::from_iter([
        ("message".to_owned(), json!(err.to_string())),
        ("type".to_owned(), json!(err.openai_error_type())),
    ]);
    if let Some(code) = err.openai_error_code() {
        obj.insert("code".to_owned(), json!(code));
        obj.insert("param".to_owned(), Value::Null);
    }
    (err.http_status(), Json(json!({"error": obj})))
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
    health_response(
        s.worker.is_panicked(),
        s.admission.is_closed() || s.worker.is_stopping(),
    )
}

fn health_response(panicked: bool, stopping: bool) -> Response {
    // Readiness is public. The worker logs diagnostics; never publish them here.
    if panicked {
        return (
            StatusCode::SERVICE_UNAVAILABLE,
            Json(json!({"status":"error"})),
        )
            .into_response();
    }
    if stopping {
        return (
            StatusCode::SERVICE_UNAVAILABLE,
            Json(json!({"status":"stopping"})),
        )
            .into_response();
    }
    Json(json!({"status":"ok"})).into_response()
}

pub fn router(state: Arc<OpenAiServerState>) -> Router {
    let app = Router::new()
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
        .route("/api/metrics", get(metrics_json));
    // UI visibility is independent of API authentication and telemetry exposure.
    let app = if state.webui {
        app.merge(crate::dashboard::router())
    } else {
        app
    };
    app.layer(axum::middleware::from_fn_with_state(
        state.clone(),
        count_inference_responses,
    ))
    .fallback(|| async { error(StatusCode::NOT_FOUND, "not found") })
    .with_state(state)
}
async fn models(State(s): State<Arc<OpenAiServerState>>) -> Json<Value> {
    let max_model_len = s.worker.max_model_len();
    let mut data = Vec::with_capacity(1 + s.worker.lora_registry.len());
    data.push(json!({
        "id": s.model_name,
        "object": "model",
        "owned_by": "warp34",
        "max_model_len": max_model_len,
    }));
    for entry in s.worker.lora_registry.entries() {
        data.push(json!({
            "id": entry.name,
            "object": "model",
            "owned_by": "warp34",
            "parent": s.model_name,
            "max_model_len": max_model_len,
        }));
    }
    Json(json!({"object":"list","data":data}))
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
    let endpoint = is_inference_endpoint(request.method(), request.uri().path())
        .then(|| request.uri().path().to_owned());
    let response = next.run(request).await;
    if let Some(endpoint) = endpoint {
        let status = response.status();
        s.metrics.http_response(status.as_u16());
        if status.is_client_error() || status.is_server_error() {
            super::request_log::http_error(&endpoint, status.as_u16());
        }
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
    cpu_with(state, f, bad).await
}
async fn cpu_with<T: Send + 'static, E: Send + 'static>(
    state: &OpenAiServerState,
    f: impl FnOnce() -> Result<T, E> + Send + 'static,
    map_err: impl FnOnce(E) -> ApiError,
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
    .map_err(map_err)
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

#[derive(Clone, Copy, PartialEq, Eq)]
enum PrepareMode {
    Generate,
    CountTokens,
}

enum PrepareError {
    BadRequest(String),
    ContextLengthExceeded {
        prompt_tokens: usize,
        max_model_len: usize,
        param: &'static str,
    },
}

impl From<String> for PrepareError {
    fn from(msg: String) -> Self {
        Self::BadRequest(msg)
    }
}

impl From<&str> for PrepareError {
    fn from(msg: &str) -> Self {
        Self::BadRequest(msg.to_owned())
    }
}

fn prepare_openai_err(err: PrepareError) -> ApiError {
    match err {
        PrepareError::BadRequest(msg) => bad(msg),
        PrepareError::ContextLengthExceeded {
            prompt_tokens,
            max_model_len,
            param,
        } => (
            StatusCode::BAD_REQUEST,
            Json(json!({
                "error": {
                    "message": format!(
                        "This model's maximum context length is {max_model_len} tokens (serving context limit). However, your prompt contains {prompt_tokens} input tokens, leaving no room for generation. Please reduce the length of the {param}."
                    ),
                    "type": "invalid_request_error",
                    "param": param,
                    "code": "context_length_exceeded",
                }
            })),
        ),
    }
}

fn prepare_responses_err(err: PrepareError) -> ApiError {
    match err {
        PrepareError::BadRequest(msg) => {
            responses_api_err(crate::responses_api::ResponsesApiError::bad_request(msg))
        }
        PrepareError::ContextLengthExceeded {
            prompt_tokens,
            max_model_len,
            ..
        } => responses_api_err(
            crate::responses_api::ResponsesApiError::context_length_exceeded(
                prompt_tokens,
                max_model_len,
            ),
        ),
    }
}

fn prepare_anthropic_err(err: PrepareError) -> ApiError {
    match err {
        PrepareError::BadRequest(msg) => anthropic_api_err(
            StatusCode::BAD_REQUEST,
            crate::anthropic_api::AnthropicApiError::bad_request(msg),
        ),
        PrepareError::ContextLengthExceeded {
            prompt_tokens,
            max_model_len,
            ..
        } => anthropic_api_err(
            StatusCode::BAD_REQUEST,
            crate::anthropic_api::AnthropicApiError::bad_request(format!(
                "prompt is too long: {prompt_tokens} tokens > {} maximum",
                max_model_len.saturating_sub(1)
            )),
        ),
    }
}

struct Prepared {
    explicit_max_tokens: Option<usize>,
    parser: Qwen3Parser,
    prompt: Vec<i32>,
    images: Vec<crate::vision::PreprocessedImage>,
    params: SamplingParams,
    soft_output_budget: bool,
    constraint: Option<crate::parser::TokenBitmaskConstraint>,
    lora_id: i32,
    served_model: String,
    stream: bool,
    include_usage: bool,
    messages_count: Option<usize>,
    tools_count: Option<usize>,
    enable_thinking: Option<bool>,
    reasoning_effort: Option<String>,
}
impl Prepared {
    fn turn_log(&self, id: String, api: &'static str, context: usize) -> TurnLog {
        TurnLog::new(
            id,
            api,
            self.served_model.clone(),
            self.stream,
            self.prompt.len(),
            self.explicit_max_tokens,
            self.params.max_tokens,
            context,
        )
        .with_turn_metadata(
            self.messages_count,
            self.tools_count,
            self.enable_thinking,
            self.reasoning_effort.clone(),
            self.params.thinking_budget.as_ref().map(|b| b.budget),
            &self.params,
        )
    }
}

fn build_thinking_budget_config(
    s: &OpenAiServerState,
    request: &Value,
    tokenizer: &tokenizers::Tokenizer,
    per_request_effort: Option<&str>,
    resolved_effort: Option<&str>,
    max_tokens: usize,
    has_tools: bool,
) -> Result<Option<crate::parser::ThinkingBudgetConfig>, PrepareError> {
    let (budget_kind, budget_msg) = super::resolve_chat_reasoning_budget(
        request,
        s.default_reasoning_budget,
        s.default_reasoning_budget_message.as_deref(),
    )?;
    let (effective_budget, is_effort_derived) = match budget_kind {
        super::ResolvedReasoningBudget::PerRequestExplicit(n) => (Some(n), false),
        super::ResolvedReasoningBudget::Unlimited => (None, false),
        super::ResolvedReasoningBudget::ServerDefault(_)
            if per_request_effort.is_some() && s.reasoning_effort_budget =>
        {
            (
                super::compute_effort_reasoning_budget(resolved_effort, max_tokens),
                true,
            )
        }
        super::ResolvedReasoningBudget::ServerDefault(n) => (Some(n), false),
        super::ResolvedReasoningBudget::Default if s.reasoning_effort_budget => (
            super::compute_effort_reasoning_budget(resolved_effort, max_tokens),
            true,
        ),
        super::ResolvedReasoningBudget::Default => (None, false),
    };
    let Some(mut budget) = effective_budget else {
        return Ok(None);
    };
    let enc = |text: &str| -> Result<Vec<i32>, String> {
        Ok(tokenizer
            .encode(text, false)
            .map_err(|e| e.to_string())?
            .get_ids()
            .iter()
            .map(|&id| id as i32)
            .collect())
    };
    let tag_ids = |tag: &str| -> Result<Vec<i32>, String> {
        if let Some(id) = tokenizer.token_to_id(tag) {
            Ok(vec![id as i32])
        } else {
            enc(tag)
        }
    };
    let start_token_ids = tag_ids("<think>")?;
    let end_token_ids = tag_ids("</think>")?;
    let explicit_custom_msg = budget_msg.is_some();
    let mut forced_token_ids = Vec::new();
    // Skip the default transition line when budget == 0 (immediate close) unless
    // the caller explicitly configured a custom message.
    let effective_msg = match budget_msg.as_deref() {
        Some(m) => m,
        None if budget == 0 => "",
        None => super::DEFAULT_REASONING_BUDGET_MESSAGE,
    };
    if !effective_msg.is_empty() {
        forced_token_ids.extend(enc(effective_msg)?);
    }
    // Stop forcing at `</think>` (do not append `\n`) so XGrammar structured
    // outputs and tool grammars control byte 0 of the answer cleanly.
    forced_token_ids.extend_from_slice(&end_token_ids);
    if start_token_ids.is_empty() || end_token_ids.is_empty() || forced_token_ids.is_empty() {
        return Err(
            "tokenizer cannot encode <think>/</think> tags required for thinking budget".into(),
        );
    }
    // Reserve headroom for forced closing tokens + at least 1 answer token so
    // small `max_tokens` requests never truncate mid-transition before `</think>`.
    if budget
        .saturating_add(forced_token_ids.len())
        .saturating_add(1)
        > max_tokens
    {
        if is_effort_derived {
            if max_tokens > forced_token_ids.len() + 1 {
                budget = max_tokens - forced_token_ids.len() - 1;
            } else {
                forced_token_ids.clone_from(&end_token_ids);
                budget = max_tokens.saturating_sub(forced_token_ids.len() + 1);
            }
        } else if !explicit_custom_msg {
            forced_token_ids.clone_from(&end_token_ids);
        }
    }
    let mut extra_end_token_ids = Vec::new();
    if has_tools
        && let Ok(tool_ids) = tag_ids("<tool_call>")
        && !tool_ids.is_empty()
    {
        extra_end_token_ids.push(tool_ids);
    }
    Ok(Some(crate::parser::ThinkingBudgetConfig {
        budget,
        start_token_ids,
        end_token_ids,
        forced_token_ids,
        extra_end_token_ids,
    }))
}

fn prepare(
    s: &OpenAiServerState,
    request: Value,
    chat: bool,
    mode: PrepareMode,
) -> Result<Prepared, PrepareError> {
    if !request.is_object() {
        return Err("request must be an object".into());
    }
    if request.get("n").is_some_and(|n| n.as_u64() != Some(1)) {
        return Err("only n=1 is supported".into());
    }
    let req_model = request.get("model").and_then(Value::as_str);
    let explicit_lora = request
        .get("lora")
        .or_else(|| request.get("adapter"))
        .and_then(Value::as_str);
    let (lora_id, served_model) = s.resolve_request_lora(req_model, explicit_lora)?;
    let explicit_max_tokens = match request
        .get("max_completion_tokens")
        .or_else(|| request.get("max_tokens"))
        .filter(|v| !v.is_null())
    {
        None => None,
        Some(n) => {
            let n = n
                .as_u64()
                .and_then(|n| usize::try_from(n).ok())
                .ok_or("invalid max_tokens")?;
            if n == 0 {
                return Err("max_tokens must be positive".into());
            }
            Some(n)
        }
    };
    let initial_max_tokens = explicit_max_tokens.unwrap_or(super::MAX_OUTPUT_TOKENS);
    let mut params = if chat {
        super::native_sampling_params_with_defaults(
            &request,
            initial_max_tokens,
            &s.worker.default_sampling_params,
        )?
    } else {
        super::completion_sampling_params_with_defaults(
            &request,
            initial_max_tokens,
            &s.worker.default_sampling_params,
        )?
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
    let per_request_effort = if chat {
        super::request_explicit_reasoning_effort(&request)?
    } else {
        None
    };
    let (thinking, resolved_effort) = if chat {
        super::resolve_chat_reasoning_config(&request, s.default_reasoning_effort.as_deref())?
    } else {
        (false, None)
    };
    let (prompt, images, messages_count, tools_count, enable_thinking) = if chat {
        let (messages, images) = super::parse_openai_messages_with_vision_config(
            request.get("messages").ok_or("missing messages")?,
            s.allow_remote_images,
            &s.worker.vision_preprocessor,
        )?;
        let mut template_kwargs = match request.get("chat_template_kwargs").filter(|v| !v.is_null())
        {
            None => serde_json::Map::new(),
            Some(v) => v
                .as_object()
                .cloned()
                .ok_or("chat_template_kwargs must be an object")?,
        };
        template_kwargs.insert("enable_thinking".into(), Value::Bool(thinking));
        if let Some(ref effort) = resolved_effort {
            template_kwargs.insert("reasoning_effort".into(), Value::String(effort.clone()));
        } else {
            template_kwargs.remove("reasoning_effort");
        }
        let active_tools = request
            .get("tools")
            .filter(|v| !v.is_null() && request["tool_choice"] != "none");
        let msg_cnt = messages.len();
        let tool_cnt = active_tools.and_then(Value::as_array).map_or(0, Vec::len);
        let rendered =
            s.worker
                .chat_template
                .render(&messages, active_tools, true, &template_kwargs)?;
        let effective_thinking = thinking && !rendered.trim_end().ends_with("</think>");
        (
            rendered,
            images,
            Some(msg_cnt),
            Some(tool_cnt),
            Some(effective_thinking),
        )
    } else {
        (
            request
                .get("prompt")
                .and_then(Value::as_str)
                .ok_or("missing string prompt")?
                .to_owned(),
            Vec::new(),
            None,
            None,
            None,
        )
    };
    let thinking = enable_thinking.unwrap_or(false);
    params.enable_thinking = thinking;
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
    let mut soft_output_budget = false;
    if mode == PrepareMode::Generate {
        let max_len = s.worker.max_model_len();
        if prompt.len() >= max_len {
            return Err(PrepareError::ContextLengthExceeded {
                prompt_tokens: prompt.len(),
                max_model_len: max_len,
                param: if chat { "messages" } else { "prompt" },
            });
        }
        let remaining = max_len - prompt.len();
        const SOFT_CEILING_THRESHOLD_TOKENS: usize = 2048;
        let (resolved_max, soft) = match explicit_max_tokens {
            Some(n) => (
                n.min(remaining),
                n > remaining || (chat && !params.ignore_eos && n > SOFT_CEILING_THRESHOLD_TOKENS),
            ),
            None if chat => (remaining, true),
            None => (remaining.min(super::DEFAULT_COMPLETION_MAX_TOKENS), false),
        };
        params.max_tokens = resolved_max;
        soft_output_budget = soft;
        if params.min_tokens > params.max_tokens {
            return Err("min_tokens exceeds max_tokens".into());
        }
        if chat && thinking {
            params.thinking_budget = build_thinking_budget_config(
                s,
                &request,
                tokenizer,
                per_request_effort.as_deref(),
                resolved_effort.as_deref(),
                params.max_tokens,
                tools_count.unwrap_or(0) > 0,
            )?;
        }
    }
    let constraint = if mode == PrepareMode::Generate {
        match crate::request_constraints::RequestConstraint::parse(&request)? {
            Some(spec) => {
                let factory = s
                    .constraint_factory
                    .as_ref()
                    .ok_or("constraint tokenizer unavailable")?;
                let request_factory = factory.with_stop_tokens(&params.stop_token_ids)?;
                Some(spec.compile(&request_factory, thinking)?)
            }
            None => None,
        }
    } else {
        None
    };
    let stream = request
        .get("stream")
        .map_or(Ok(false), |v| v.as_bool().ok_or("invalid stream"))?;
    let include_usage = request
        .pointer("/stream_options/include_usage")
        .map_or(Ok(false), |v| v.as_bool().ok_or("invalid include_usage"))?;
    Ok(Prepared {
        explicit_max_tokens,
        parser: Qwen3Parser::with_tools(
            thinking,
            &crate::request_constraints::parser_tools(&request)?,
        )?,
        prompt,
        images,
        params,
        soft_output_budget,
        constraint,
        lora_id,
        served_model,
        stream,
        include_usage,
        messages_count,
        tools_count,
        enable_thinking,
        reasoning_effort: resolved_effort,
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
        prepare(&prep_state, json, chat, PrepareMode::Generate).map_err(prepare_openai_err)?
    } else {
        cpu_with(
            &s,
            move || prepare(&prep_state, json, chat, PrepareMode::Generate),
            prepare_openai_err,
        )
        .await?
    };
    let request_id = s.next_request_id.fetch_add(1, Ordering::Relaxed) + 1;
    let id = crate::responses_api::generate_unguessable_prefixed_id(
        if chat { "chatcmpl-" } else { "cmpl-" },
        request_id,
    );
    let mut log = prepared.turn_log(
        id.clone(),
        if chat { "chat" } else { "completions" },
        s.worker.max_model_len(),
    );
    let prompt_len = prepared.prompt.len();
    let generation = s
        .worker
        .submit_with_images_lora_and_budget_mode(
            prepared.prompt,
            prepared.images,
            prepared.params.clone(),
            prepared.constraint,
            prepared.lora_id,
            prepared.soft_output_budget,
        )
        .inspect_err(|_| log.finish("error", false))
        .map_err(worker_api_err)?;
    log.track_cache(
        generation.cached_tokens.clone(),
        generation.cache_lookup.clone(),
    );
    let stream = prepared.stream;
    let include_usage = prepared.include_usage;
    let output = output_stream(
        s,
        generation,
        prepared.params,
        prepared.parser,
        prepared.served_model,
        chat,
        prompt_len,
        id,
        log,
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
    if !d.tool_call_deltas.is_empty() {
        value["tool_calls"] = json!(
            d.tool_call_deltas
                .into_iter()
                .map(|tc| {
                    *tool_index = (*tool_index).max(tc.index + 1);
                    let mut fn_obj = serde_json::Map::new();
                    if let Some(name) = tc.name {
                        fn_obj.insert("name".into(), json!(name));
                    }
                    if let Some(args) = tc.arguments_delta {
                        fn_obj.insert("arguments".into(), json!(args));
                    } else if fn_obj.contains_key("name") {
                        fn_obj.insert("arguments".into(), json!(""));
                    }
                    let mut item = serde_json::Map::new();
                    item.insert("index".into(), json!(tc.index));
                    if let Some(id) = tc.id {
                        item.insert("id".into(), json!(id));
                        item.insert("type".into(), json!("function"));
                    }
                    item.insert("function".into(), Value::Object(fn_obj));
                    Value::Object(item)
                })
                .collect::<Vec<_>>()
        );
    } else if !d.completed_tool_calls.is_empty() {
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
    parser: Qwen3Parser,
    served_model: String,
    chat: bool,
    prompt_len: usize,
    id: String,
    mut log: TurnLog,
    admission: OwnedSemaphorePermit,
) -> impl Stream<Item = Output> + Send {
    async_stream::stream! {
        let _admission = admission;
        let tokenizer = s.worker.tokenizer.as_ref().unwrap();
        let mut decoder = tokenizer.decode_stream(false);
        let mut parser = parser;
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
        let created = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap_or_default().as_secs();
        let chunk = |delta: Value, lp: Value| {
            let choice = if chat { json!({"index":0,"delta":delta,"finish_reason":null,"logprobs":lp}) }
                else { json!({"index":0,"text":delta["content"].as_str().unwrap_or(""),"finish_reason":null,"logprobs":lp}) };
            json!({"id":id,"object":if chat {"chat.completion.chunk"} else {"text_completion"},"created":created,"model":&served_model,"choices":[choice]})
        };
        if chat { yield Output::Delta(chunk(json!({"role":"assistant"}),Value::Null)); }
        while let Some(event) = generation.events.recv().await {
            match event {
                GenerationEvent::Token { id, logprobs } => {
                    log.token();
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
                        Ok(Some(text)) => text,
                        Ok(None) if chat => {
                            if parser.in_reasoning() {
                                log.reasoning_token();
                            }
                            continue;
                        }
                        Ok(None) => String::new(),
                        Err(e) => { log.finish("error", false); yield Output::Error(WorkerError::internal(e.to_string())); return; }
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
                    let was_reasoning = chat && parser.in_reasoning();
                    let delta = if chat {
                        let raw_delta = parser.push_str(&visible);
                        if was_reasoning && (parser.in_reasoning() || raw_delta.reasoning_delta.is_some()) {
                            log.reasoning_token();
                        }
                        log.set_tool_calls(parser.emitted_tool_calls_count());
                        delta_value(raw_delta, &mut tool_index)
                    } else {
                        json!({"content":visible})
                    };
                    if delta.as_object().is_some_and(|o| !o.is_empty()) || lp.is_some() {
                        yield Output::Delta(chunk(delta,lp.map_or(Value::Null,|v| if chat { json!({"content":[v]}) } else { super::completion_logprobs(&[v], &mut legacy_offset) })));
                    }
                    if matched { stopped = true; done = true; break; }
                }
                GenerationEvent::Finished { cached_tokens, thinking_budget_exhausted, .. } => {
                    if thinking_budget_exhausted { log.mark_thinking_budget_exhausted(); }
                    cached = cached_tokens; done = true; break;
                }
                GenerationEvent::Error(err) => { log.finish("error", false); yield Output::Error(err); return; }
            }
        }
        cached = cached.max(generation.cached_tokens.load(Ordering::Acquire) as usize);
        let hit_limit = generation.hit_limit.load(Ordering::Acquire);
        drop(generation); // cancellation also covers stop strings and a dropped HTTP response body
        if !done { log.finish("error", false); yield Output::Error(WorkerError::internal("engine stream closed before completion (client backpressure, overload, shutdown, or worker failure)")); return; }
        let tail = std::mem::take(&mut stops.pending);
        text.push_str(&tail);
        if !tail.is_empty() {
            let delta = if chat { delta_value(parser.push_str(&tail),&mut tool_index) } else { json!({"content":tail}) };
            yield Output::Delta(chunk(delta,Value::Null));
        }
        let (tail,turn) = parser.finish();
        let delta = delta_value(tail,&mut tool_index);
        if chat && delta.as_object().is_some_and(|o| !o.is_empty()) { yield Output::Delta(chunk(delta,Value::Null)); }
        if chat {
            let reasoning_override = turn
                .reasoning_content
                .as_deref()
                .is_none_or(str::is_empty)
                .then_some(0);
            log.set_turn_outcome(reasoning_override, turn.tool_calls.len());
        }
        let finish = if chat && !turn.tool_calls.is_empty() { "tool_calls" } else if !stopped && (count >= params.max_tokens || hit_limit) { "length" } else { "stop" };
        let lp = if !params.logprobs { Value::Null } else if chat { json!({"content":logs}) } else { super::completion_logprobs(&logs, &mut 0) };
        let choice = if chat { json!({"index":0,"message":{"role":"assistant","content":turn.content,"reasoning_content":turn.reasoning_content,"tool_calls":turn.tool_calls},"finish_reason":finish,"logprobs":lp}) }
            else { json!({"index":0,"text":text,"finish_reason":finish,"logprobs":lp}) };
        log.finish(finish, hit_limit);
        yield Output::Complete(json!({"id":id,"object":if chat {"chat.completion"} else {"text_completion"},"created":created,"model":served_model,"choices":[choice],"usage":{"prompt_tokens":prompt_len,"completion_tokens":count,"total_tokens":prompt_len+count,"prompt_tokens_details":{"cached_tokens":cached}}}));
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

fn worker_anthropic_err(err: WorkerError) -> ApiError {
    if let WorkerError::ContextLengthExceeded {
        prompt_tokens,
        output_tokens,
        limit,
    } = err
    {
        let max_input = if output_tokens == 0 {
            limit
        } else {
            limit.saturating_sub(1)
        };
        return anthropic_api_err(
            StatusCode::BAD_REQUEST,
            crate::anthropic_api::AnthropicApiError::bad_request(format!(
                "prompt is too long: {prompt_tokens} tokens > {max_input} maximum"
            )),
        );
    }
    to_anthropic_err(worker_api_err(err))
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
    let count = cpu_with(
        &s,
        move || prepare(&prep_state, body, true, PrepareMode::CountTokens).map(|p| p.prompt.len()),
        prepare_openai_err,
    )
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
    let count = cpu_with(
        &s,
        move || {
            prepare(&prep_state, chat_req, true, PrepareMode::CountTokens).map(|p| p.prompt.len())
        },
        prepare_responses_err,
    )
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
    let count = cpu_with(
        &s,
        move || {
            prepare(&prep_state, chat_req, true, PrepareMode::CountTokens).map(|p| p.prompt.len())
        },
        prepare_anthropic_err,
    )
    .await?;
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
            .map_or(0, |e| e.len().min(output_tokens)),
    }
}

fn finalize_response(
    store: &crate::responses_api::ResponseStore,
    meta: &crate::responses_api::ResponsesPreparedMeta,
    scope: Option<&str>,
    response: Value,
    history: Vec<Value>,
) -> Value {
    let Some(scope) = scope else { return response };
    store.finalize(
        crate::responses_api::StoredResponse {
            id: meta.response_id.clone(),
            owner_scope: scope.to_owned(),
            response,
            history_messages: history,
        },
        meta.store,
    )
}

fn responses_stream_failure(
    store: &crate::responses_api::ResponseStore,
    meta: &crate::responses_api::ResponsesPreparedMeta,
    scope: Option<&str>,
    sse: &mut crate::responses_api::ResponsesSseFormatter,
    message: &str,
) -> (String, Value) {
    let response = finalize_response(
        store,
        meta,
        scope,
        sse.failure_response(message),
        Vec::new(),
    );
    sse.terminal_response(response)
}

fn responses_sync_failure(
    store: &crate::responses_api::ResponseStore,
    meta: &crate::responses_api::ResponsesPreparedMeta,
    scope: Option<&str>,
    failure: ApiError,
) -> Result<Response, ApiError> {
    let response = finalize_response(store, meta, scope, json!({"status": "failed"}), Vec::new());
    if response["status"] == "cancelled" {
        Ok(Json(response).into_response())
    } else {
        Err(failure)
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
    let (chat_req, mut meta) = crate::responses_api::translate_responses_to_chat(
        &body,
        &s.model_name,
        &s.response_store,
        caller_scope.as_deref(),
        request_id,
    )
    .map_err(responses_api_err)?;
    let prep_state = s.clone();
    let prepared = cpu_with(
        &s,
        move || prepare(&prep_state, chat_req, true, PrepareMode::Generate),
        prepare_responses_err,
    )
    .await?;
    meta.model.clone_from(&prepared.served_model);
    meta.max_output_tokens = prepared.params.max_tokens;
    meta.temperature = f64::from(prepared.params.temperature);
    meta.top_p = f64::from(prepared.params.top_p);
    let mut log = prepared.turn_log(
        meta.response_id.clone(),
        "responses",
        s.worker.max_model_len(),
    );
    let prompt_len = prepared.prompt.len();
    let generation = s
        .worker
        .submit_with_images_lora_and_budget_mode(
            prepared.prompt,
            prepared.images,
            prepared.params.clone(),
            prepared.constraint,
            prepared.lora_id,
            prepared.soft_output_budget,
        )
        .inspect_err(|_| log.finish("error", false))
        .map_err(worker_api_err)?;
    log.track_cache(
        generation.cached_tokens.clone(),
        generation.cache_lookup.clone(),
    );
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
        let parser = prepared.parser;
        let events = async_stream::stream! {
            let _admission = admission;
            let _active_guard = active_guard;
            let tokenizer = s.worker.tokenizer.as_ref().unwrap();
            let mut decoder = tokenizer.decode_stream(false);
            let mut parser = parser;
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
                        log.token();
                        count += 1;
                        if count >= params.min_tokens && params.is_stop_or_eos(id, &s.worker.eos_token_ids) {
                            stopped = true;
                            continue;
                        }
                        let decoded = match decoder.step(id as u32) {
                            Ok(Some(t)) => t,
                            Ok(None) => {
                                if parser.in_reasoning() {
                                    log.reasoning_token();
                                }
                                continue;
                            }
                            Err(e) => {
                                log.finish("error", false);
                                let (ev_name, payload) = responses_stream_failure(&s.response_store, &meta, caller_scope.as_deref(), &mut sse, &e.to_string());
                                yield Ok(Event::default().event(ev_name).data(payload.to_string()));
                                return;
                            }
                        };
                        let active_stops = if count >= params.min_tokens { params.stop_strings.as_slice() } else { &[] };
                        let (visible, matched) = stops.push(&decoded, active_stops);
                        let was_reasoning = parser.in_reasoning();
                        let raw_delta = parser.push_str(&visible);
                        let emitted_reasoning = raw_delta.reasoning_delta.is_some();
                        for (ev_name, payload) in sse.push_delta(raw_delta) {
                            yield Ok(Event::default().event(ev_name).data(payload.to_string()));
                        }
                        if was_reasoning && (parser.in_reasoning() || emitted_reasoning) {
                            log.reasoning_token();
                        }
                        log.set_tool_calls(parser.emitted_tool_calls_count());
                        if matched {
                            stopped = true;
                            done = true;
                            break;
                        }
                    }
                    GenerationEvent::Finished {
                        cached_tokens,
                        thinking_budget_exhausted,
                        ..
                    } => {
                        if thinking_budget_exhausted {
                            log.mark_thinking_budget_exhausted();
                        }
                        cached = cached_tokens;
                        done = true;
                        break;
                    }
                    GenerationEvent::Error(err) => {
                        log.finish("error", false);
                        let msg = err.to_string();
                        let (ev_name, payload) = responses_stream_failure(&s.response_store, &meta, caller_scope.as_deref(), &mut sse, &msg);
                        yield Ok(Event::default().event(ev_name).data(payload.to_string()));
                        return;
                    }
                }
            }
            cached = cached.max(generation.cached_tokens.load(Ordering::Acquire) as usize);
            let hit_limit = generation.hit_limit.load(Ordering::Acquire);
            drop(generation);
            if !done {
                log.finish("error", false);
                let (ev_name, payload) = responses_stream_failure(&s.response_store, &meta, caller_scope.as_deref(), &mut sse, "engine stream closed before completion");
                yield Ok(Event::default().event(ev_name).data(payload.to_string()));
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
            } else if !stopped && (count >= params.max_tokens || hit_limit) {
                "length"
            } else {
                "stop"
            };
            let reasoning_tokens = count_reasoning_tokens(tokenizer, &turn, count);
            log.set_turn_outcome(Some(reasoning_tokens as u64), turn.tool_calls.len());
            let (end_frames, final_resp, history) =
                sse.finish_output(&turn, finish, prompt_len, cached, count, reasoning_tokens);
            let final_resp = finalize_response(&s.response_store, &meta, caller_scope.as_deref(), final_resp, history);
            for (ev_name, payload) in end_frames {
                yield Ok(Event::default().event(ev_name).data(payload.to_string()));
            }
            let (ev_name, payload) = sse.terminal_response(final_resp);
            log.finish(finish, hit_limit);
            yield Ok(Event::default().event(ev_name).data(payload.to_string()));
        };
        Ok(Sse::new(events)
            .keep_alive(KeepAlive::new().interval(Duration::from_secs(15)))
            .into_response())
    } else {
        let _admission = admission;
        let _active_guard = active_guard;
        let tokenizer = s.worker.tokenizer.as_ref().unwrap();
        let mut decoder = tokenizer.decode_stream(false);
        let mut parser = prepared.parser;
        let mut stops = StopBuffer::default();
        let mut generation = generation;
        let mut count = 0usize;
        let mut cached = 0usize;
        let mut done = false;
        let mut stopped = false;

        while let Some(event) = generation.events.recv().await {
            match event {
                GenerationEvent::Token { id, .. } => {
                    log.token();
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
                            log.finish("error", false);
                            return responses_sync_failure(
                                &s.response_store,
                                &meta,
                                caller_scope.as_deref(),
                                error(StatusCode::INTERNAL_SERVER_ERROR, e),
                            );
                        }
                    };
                    let active_stops = if count >= prepared.params.min_tokens {
                        prepared.params.stop_strings.as_slice()
                    } else {
                        &[]
                    };
                    let (visible, matched) = stops.push(&decoded, active_stops);
                    let was_reasoning = parser.in_reasoning();
                    let _ = parser.push_str(&visible);
                    if was_reasoning && parser.in_reasoning() {
                        log.reasoning_token();
                    }
                    log.set_tool_calls(parser.emitted_tool_calls_count());
                    if matched {
                        stopped = true;
                        done = true;
                        break;
                    }
                }
                GenerationEvent::Finished {
                    cached_tokens,
                    thinking_budget_exhausted,
                    ..
                } => {
                    if thinking_budget_exhausted {
                        log.mark_thinking_budget_exhausted();
                    }
                    cached = cached_tokens;
                    done = true;
                    break;
                }
                GenerationEvent::Error(err) => {
                    log.finish("error", false);
                    return responses_sync_failure(
                        &s.response_store,
                        &meta,
                        caller_scope.as_deref(),
                        worker_api_err(err),
                    );
                }
            }
        }
        cached = cached.max(generation.cached_tokens.load(Ordering::Acquire) as usize);
        let hit_limit = generation.hit_limit.load(Ordering::Acquire);
        drop(generation);
        if !done {
            log.finish("error", false);
            return responses_sync_failure(
                &s.response_store,
                &meta,
                caller_scope.as_deref(),
                error(
                    StatusCode::INTERNAL_SERVER_ERROR,
                    "engine stream closed before completion",
                ),
            );
        }
        let tail = std::mem::take(&mut stops.pending);
        if !tail.is_empty() {
            let _ = parser.push_str(&tail);
        }
        let (_, turn) = parser.finish();
        let finish = if !turn.tool_calls.is_empty() {
            "tool_calls"
        } else if !stopped && (count >= prepared.params.max_tokens || hit_limit) {
            "length"
        } else {
            "stop"
        };
        let reasoning_tokens = count_reasoning_tokens(tokenizer, &turn, count);
        log.set_turn_outcome(Some(reasoning_tokens as u64), turn.tool_calls.len());
        let (resp, history) = crate::responses_api::build_response_object(
            &meta,
            &turn,
            finish,
            prompt_len,
            cached,
            count,
            reasoning_tokens,
        );
        let resp = finalize_response(
            &s.response_store,
            &meta,
            caller_scope.as_deref(),
            resp,
            history,
        );
        log.finish(finish, hit_limit);
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
    let (chat_req, mut meta) =
        crate::anthropic_api::translate_anthropic_to_chat(&body, &s.model_name, request_id, true)
            .map_err(|e| anthropic_api_err(StatusCode::BAD_REQUEST, e))?;
    let prep_state = s.clone();
    let prepared = cpu_with(
        &s,
        move || prepare(&prep_state, chat_req, true, PrepareMode::Generate),
        prepare_anthropic_err,
    )
    .await?;
    meta.model.clone_from(&prepared.served_model);
    let mut log = prepared.turn_log(
        meta.message_id.clone(),
        "anthropic",
        s.worker.max_model_len(),
    );
    let prompt_len = prepared.prompt.len();
    let generation = s
        .worker
        .submit_with_images_lora_and_budget_mode(
            prepared.prompt,
            prepared.images,
            prepared.params.clone(),
            prepared.constraint,
            prepared.lora_id,
            prepared.soft_output_budget,
        )
        .inspect_err(|_| log.finish("error", false))
        .map_err(worker_anthropic_err)?;
    log.track_cache(
        generation.cached_tokens.clone(),
        generation.cache_lookup.clone(),
    );

    if meta.stream {
        let params = prepared.params;
        let parser = prepared.parser;
        let events = async_stream::stream! {
            let _admission = admission;
            let tokenizer = s.worker.tokenizer.as_ref().unwrap();
            let mut decoder = tokenizer.decode_stream(false);
            let mut parser = parser;
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
                        log.token();
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
                            Ok(None) => {
                                if parser.in_reasoning() {
                                    log.reasoning_token();
                                }
                                continue;
                            }
                            Err(e) => {
                                log.finish("error", false);
                                let (ev_name, payload) = crate::anthropic_api::AnthropicSseFormatter::error_event(&e.to_string());
                                yield Ok(Event::default().event(ev_name).data(payload.to_string()));
                                return;
                            }
                        };
                        let active_stops = if count >= params.min_tokens { params.stop_strings.as_slice() } else { &[] };
                        let (visible, m_stop) = stops.push_matched(&decoded, active_stops);
                        let was_reasoning = parser.in_reasoning();
                        let raw_delta = parser.push_str(&visible);
                        let emitted_reasoning = raw_delta.reasoning_delta.is_some();
                        for (ev_name, payload) in sse.push_delta(raw_delta, cached) {
                            yield Ok(Event::default().event(ev_name).data(payload.to_string()));
                        }
                        if was_reasoning && (parser.in_reasoning() || emitted_reasoning) {
                            log.reasoning_token();
                        }
                        log.set_tool_calls(parser.emitted_tool_calls_count());
                        if let Some(seq) = m_stop {
                            matched_stop = Some(seq);
                            stopped = true;
                            done = true;
                            break;
                        }
                    }
                    GenerationEvent::Finished {
                        cached_tokens,
                        thinking_budget_exhausted,
                        ..
                    } => {
                        if thinking_budget_exhausted {
                            log.mark_thinking_budget_exhausted();
                        }
                        cached = cached.max(cached_tokens);
                        done = true;
                        break;
                    }
                    GenerationEvent::Error(err) => {
                        log.finish("error", false);
                        let msg = err.to_string();
                        let (ev_name, payload) = crate::anthropic_api::AnthropicSseFormatter::error_event(&msg);
                        yield Ok(Event::default().event(ev_name).data(payload.to_string()));
                        return;
                    }
                }
            }
            cached = cached.max(generation.cached_tokens.load(Ordering::Acquire) as usize);
            let hit_limit = generation.hit_limit.load(Ordering::Acquire);
            drop(generation);
            if !done {
                log.finish("error", false);
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
            let reasoning_override = turn
                .reasoning_content
                .as_deref()
                .is_none_or(str::is_empty)
                .then_some(0);
            log.set_turn_outcome(reasoning_override, turn.tool_calls.len());
            let finish = if !turn.tool_calls.is_empty() {
                "tool_calls"
            } else if !stopped && (count >= params.max_tokens || hit_limit) {
                "length"
            } else {
                "stop"
            };
            for (ev_name, payload) in sse.finish(&turn, finish, matched_stop.as_deref(), cached, count) {
                if ev_name == "message_delta" {
                    let (reason, _) = crate::anthropic_api::resolve_stop_reason(&turn, finish, matched_stop.as_deref());
                    log.finish(reason, hit_limit);
                }
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
        let mut parser = prepared.parser;
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
                    log.token();
                    count += 1;
                    if count >= prepared.params.min_tokens
                        && prepared.params.is_stop_or_eos(id, &s.worker.eos_token_ids)
                    {
                        stopped = true;
                        continue;
                    }
                    let decoded = match decoder.step(id as u32) {
                        Ok(Some(t)) => t,
                        Ok(None) => {
                            if parser.in_reasoning() {
                                log.reasoning_token();
                            }
                            continue;
                        }
                        Err(e) => {
                            log.finish("error", false);
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
                    let was_reasoning = parser.in_reasoning();
                    let raw_delta = parser.push_str(&visible);
                    if was_reasoning
                        && (parser.in_reasoning() || raw_delta.reasoning_delta.is_some())
                    {
                        log.reasoning_token();
                    }
                    log.set_tool_calls(parser.emitted_tool_calls_count());
                    if let Some(seq) = m_stop {
                        matched_stop = Some(seq);
                        stopped = true;
                        done = true;
                        break;
                    }
                }
                GenerationEvent::Finished {
                    cached_tokens,
                    thinking_budget_exhausted,
                    ..
                } => {
                    if thinking_budget_exhausted {
                        log.mark_thinking_budget_exhausted();
                    }
                    cached = cached_tokens;
                    done = true;
                    break;
                }
                GenerationEvent::Error(err) => {
                    log.finish("error", false);
                    return Err(worker_anthropic_err(err));
                }
            }
        }
        cached = cached.max(generation.cached_tokens.load(Ordering::Acquire) as usize);
        let hit_limit = generation.hit_limit.load(Ordering::Acquire);
        drop(generation);
        if !done {
            log.finish("error", false);
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
        let reasoning_override = turn
            .reasoning_content
            .as_deref()
            .is_none_or(str::is_empty)
            .then_some(0);
        log.set_turn_outcome(reasoning_override, turn.tool_calls.len());
        let finish = if !turn.tool_calls.is_empty() {
            "tool_calls"
        } else if !stopped && (count >= prepared.params.max_tokens || hit_limit) {
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
        let (reason, _) =
            crate::anthropic_api::resolve_stop_reason(&turn, finish, matched_stop.as_deref());
        log.finish(reason, hit_limit);
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
/// Format only deployment state, never credentials or user-supplied hostnames.
pub(crate) fn serve_exposure_notice(
    addr: std::net::SocketAddr,
    auth_enabled: bool,
    webui: bool,
) -> String {
    let auth = if auth_enabled { "enabled" } else { "disabled" };
    let ui = if webui {
        "webui enabled and unauthenticated"
    } else {
        "webui disabled (API and metrics remain available)"
    };
    let warning = if addr.ip().is_loopback() {
        ""
    } else {
        "WARNING: non-loopback listener exposes public endpoints; "
    };
    format!(
        "[warp34 serve] {warning}ready on http://{addr}; API auth {auth} (all API operations including aliases); metrics (/metrics and /api/metrics) and minimal health are unauthenticated; {ui}"
    )
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
        eprintln!(
            "{}",
            serve_exposure_notice(listener.local_addr()?, state.api_key.is_some(), state.webui)
        );
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
    use super::{Qwen3Parser, StopBuffer, admission_error, json, to_anthropic_err, worker_api_err};
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

    #[tokio::test]
    async fn responses_failure_paths_distinguish_explicit_cancel_from_transport_failure() {
        use crate::responses_api::{
            ResponseStore, ResponsesSseFormatter, translate_responses_to_chat,
        };
        use serde_json::{Value, json};
        use std::sync::{Arc, atomic::AtomicBool};
        for streaming in [false, true] {
            for explicit_cancel in [false, true] {
                let store = ResponseStore::default();
                let (_, meta) = translate_responses_to_chat(
                    &json!({"input": "hi", "store": true}),
                    "q",
                    &store,
                    Some("owner"),
                    1,
                )
                .unwrap();
                // The worker flag is also set on backpressure/disconnect. It must
                // not itself turn a transport error into explicit cancellation.
                store.register_active(
                    meta.response_id.clone(),
                    "owner".into(),
                    Arc::new(AtomicBool::new(true)),
                );
                let cancelled =
                    explicit_cancel.then(|| store.cancel(&meta.response_id, "owner").unwrap());
                if streaming {
                    let mut formatter = ResponsesSseFormatter::new(meta.clone());
                    let (event, payload) = super::responses_stream_failure(
                        &store,
                        &meta,
                        Some("owner"),
                        &mut formatter,
                        "channel closed",
                    );
                    if let Some(ref cancelled) = cancelled {
                        assert_eq!(event, "response.cancelled");
                        assert_eq!(&payload["response"], cancelled);
                    } else {
                        assert_eq!(event, "response.failed");
                        assert_eq!(payload["response"]["error"]["message"], "channel closed");
                    }
                } else {
                    let result = super::responses_sync_failure(
                        &store,
                        &meta,
                        Some("owner"),
                        super::error(StatusCode::INTERNAL_SERVER_ERROR, "channel closed"),
                    );
                    if let Some(ref cancelled) = cancelled {
                        let response = result.unwrap();
                        assert_eq!(response.status(), StatusCode::OK);
                        let bytes = axum::body::to_bytes(response.into_body(), 4096)
                            .await
                            .unwrap();
                        assert_eq!(&serde_json::from_slice::<Value>(&bytes).unwrap(), cancelled);
                    } else {
                        assert_eq!(result.unwrap_err().0, StatusCode::INTERNAL_SERVER_ERROR);
                    }
                }
                assert_eq!(store.get_response(&meta.response_id, "owner"), cancelled);
            }
        }
    }

    #[tokio::test]
    async fn readiness_is_unavailable_without_exposing_panic_diagnostics() {
        use axum::body::to_bytes;
        assert_eq!(
            super::health_response(false, false).status(),
            StatusCode::OK
        );
        let stopping = super::health_response(false, true);
        assert_eq!(stopping.status(), StatusCode::SERVICE_UNAVAILABLE);
        let body: serde_json::Value =
            serde_json::from_slice(&to_bytes(stopping.into_body(), 4096).await.unwrap()).unwrap();
        assert_eq!(body["status"], "stopping");
        let failed = super::health_response(true, true);
        assert_eq!(failed.status(), StatusCode::SERVICE_UNAVAILABLE);
        let body: serde_json::Value =
            serde_json::from_slice(&to_bytes(failed.into_body(), 4096).await.unwrap()).unwrap();
        assert_eq!(body, serde_json::json!({"status": "error"}));
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

    #[test]
    fn chat_completions_delta_value_streams_incremental_tool_call_chunks() {
        let tools = vec![json!({"function":{"name":"write","parameters":{
            "type":"object",
            "properties":{
                "path":{"type":"string"},
                "content":{"type":"string"}
            }
        }}})];
        let mut parser = Qwen3Parser::with_tools(false, &tools).unwrap();
        let mut tool_index = 0usize;
        let mut chunks = Vec::new();
        for piece in [
            "<tool_call>\n<function=write>\n<parameter=path>\nsrc/",
            "scene.js\n</parameter>\n<parameter=content>\nhello",
            " world\n</parameter>\n</function>\n</tool_call>",
        ] {
            let v = super::delta_value(parser.push_str(piece), &mut tool_index);
            if v != json!({}) {
                chunks.push(v);
            }
        }
        let (final_d, turn) = parser.finish();
        let vf = super::delta_value(final_d, &mut tool_index);
        if vf != json!({}) {
            chunks.push(vf);
        }
        assert_eq!(tool_index, 1);
        assert_eq!(chunks.len(), 3);
        assert_eq!(chunks[0]["tool_calls"][0]["index"], 0);
        assert_eq!(chunks[0]["tool_calls"][0]["id"], "call_warp34_0");
        assert_eq!(chunks[0]["tool_calls"][0]["type"], "function");
        assert_eq!(chunks[0]["tool_calls"][0]["function"]["name"], "write");
        for later in &chunks[1..] {
            assert_eq!(later["tool_calls"][0]["index"], 0);
            assert!(later["tool_calls"][0].get("id").is_none());
            assert!(later["tool_calls"][0].get("type").is_none());
            assert!(later["tool_calls"][0]["function"].get("name").is_none());
        }
        let joined_args: String = chunks
            .iter()
            .filter_map(|c| c["tool_calls"][0]["function"]["arguments"].as_str())
            .collect();
        assert_eq!(joined_args, turn.tool_calls[0].function.arguments);
        assert_eq!(
            joined_args,
            r#"{"path":"src/scene.js","content":"hello world"}"#
        );
    }
}
