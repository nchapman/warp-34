//! Bounded engine ingress and incremental scheduled generation. Network writers never run
//! on the GPU worker; a full output channel cancels that request rather than blocking peers.
use std::{
    collections::HashMap,
    sync::{
        Arc, Mutex, OnceLock,
        atomic::{AtomicBool, AtomicU64, Ordering},
    },
    thread::JoinHandle,
};
use tokio::sync::mpsc::{self, Receiver, Sender};

pub use crate::server::WorkerError;
use crate::{
    model::{EngineBatchSlice, PipelinedDecodeOutput, Qwen3_5Engine},
    parser::{SamplerV1, SamplingParams, StepLogprobs, TokenBitmaskConstraint},
    scheduler::{ScheduledReqSlice, SchedulerConfig, SchedulerV1},
    telemetry::{Config as TelemetryConfig, Lifecycle, Outcome, Pool, Pools, Telemetry},
};

#[derive(Clone, Debug)]
pub enum GenerationEvent {
    Token {
        id: i32,
        logprobs: Option<StepLogprobs>,
    },
    Finished {
        cached_tokens: usize,
        generated_tokens: usize,
    },
    Error(WorkerError),
}

/// Bounded token events followed by exactly one out-of-band terminal event.
/// Publishing the terminal result never waits for a slow consumer to free queue space.
/// The request owns the sole sender, so channel closure orders terminal delivery after
/// all buffered tokens (and wakes a consumer already waiting in `recv`).
pub struct GenerationEvents {
    receiver: Receiver<GenerationEvent>,
    terminal: Arc<OnceLock<GenerationEvent>>,
    terminal_delivered: bool,
}

impl GenerationEvents {
    pub async fn recv(&mut self) -> Option<GenerationEvent> {
        if let Some(event) = self.receiver.recv().await {
            return Some(event);
        }
        if self.terminal_delivered {
            return None;
        }
        self.terminal_delivered = true;
        self.terminal.get().cloned()
    }
}

fn generation_channel(capacity: usize) -> (Sender<GenerationEvent>, GenerationEvents) {
    let (sender, receiver) = mpsc::channel(capacity);
    (
        sender,
        GenerationEvents {
            receiver,
            terminal: Arc::new(OnceLock::new()),
            terminal_delivered: false,
        },
    )
}

pub struct GenerationStream {
    pub events: GenerationEvents,
    pub cached_tokens: Arc<AtomicU64>,
    cancelled: Arc<AtomicBool>,
}
impl GenerationStream {
    pub fn cancel_handle(&self) -> Arc<AtomicBool> {
        self.cancelled.clone()
    }
}
impl Drop for GenerationStream {
    fn drop(&mut self) {
        self.cancelled.store(true, Ordering::Release);
    }
}

#[derive(Debug)]
pub enum DecisionTapOutput {
    CandidateLogits {
        logits: Vec<f64>,
        cached_tokens: usize,
    },
    HiddenRows {
        rows: Vec<Vec<f32>>,
        cached_tokens: usize,
    },
    ClefLogits {
        logits: Vec<Vec<f64>>,
        cached_tokens: usize,
    },
}

enum DecisionReadoutTap {
    CandidateLogits {
        token_ids: Vec<i32>,
        tx: tokio::sync::oneshot::Sender<Result<DecisionTapOutput, WorkerError>>,
    },
    HiddenPositions {
        positions: Vec<usize>,
        captured: Vec<Option<Vec<f32>>>,
        tx: tokio::sync::oneshot::Sender<Result<DecisionTapOutput, WorkerError>>,
    },
    ClefJoint {
        record: Arc<crate::clef::ClefEncodedRecord>,
        gpu_state: Option<crate::clef::ClefDeviceRequestState>,
        tx: tokio::sync::oneshot::Sender<Result<DecisionTapOutput, WorkerError>>,
    },
}

impl DecisionReadoutTap {
    fn fail(self, error: WorkerError) {
        match self {
            Self::CandidateLogits { tx, .. }
            | Self::HiddenPositions { tx, .. }
            | Self::ClefJoint { tx, .. } => {
                let _ = tx.send(Err(error));
            }
        }
    }

    fn recycle_into(self, engine: &mut Qwen3_5Engine) {
        if let Self::ClefJoint {
            gpu_state: Some(st),
            ..
        } = self
        {
            engine.clef_recycle_request(st);
        }
    }

    fn fail_and_recycle(self, engine: &mut Qwen3_5Engine, error: WorkerError) {
        match self {
            Self::ClefJoint { gpu_state, tx, .. } => {
                if let Some(st) = gpu_state {
                    engine.clef_recycle_request(st);
                }
                let _ = tx.send(Err(error));
            }
            other => other.fail(error),
        }
    }
}

struct Request {
    lifecycle: Option<Lifecycle>,
    prompt: Vec<i32>,
    images: Vec<crate::vision::PreprocessedImage>,
    vision: Option<crate::scheduled_vision::ScheduledVision>,
    reserved_pages: usize,
    cached_tokens: Arc<AtomicU64>,
    sampler: SamplerV1,
    output: Sender<GenerationEvent>,
    terminal: Arc<OnceLock<GenerationEvent>>,
    cancelled: Arc<AtomicBool>,
    max_prefix_lookup_len: Option<usize>,
    mamba_branching_seqlen: Option<usize>,
    decision_tap: Option<DecisionReadoutTap>,
}

impl Request {
    fn computed(&self, tokens: usize) {
        if let Some(life) = &self.lifecycle {
            life.computed(tokens);
        }
    }

    fn set_terminal(&self, event: GenerationEvent) {
        // First terminal result wins; in particular, unwinding must not replace an
        // already published result while dropping request-owned resources.
        if self.terminal.set(event.clone()).is_ok()
            && let Some(life) = &self.lifecycle
        {
            let outcome = match event {
                GenerationEvent::Finished { .. } if self.cancelled.load(Ordering::Acquire) => {
                    Outcome::Cancelled
                }
                GenerationEvent::Finished { .. } => Outcome::Completed,
                _ => Outcome::Failed,
            };
            life.finish(outcome);
        }
    }
}

impl Drop for Request {
    fn drop(&mut self) {
        if !std::thread::panicking()
            && self.cancelled.load(Ordering::Acquire)
            && let Some(life) = &self.lifecycle
        {
            life.finish(Outcome::Cancelled);
        }
        if std::thread::panicking() {
            // Admission and batched MTP temporarily own requests outside the registry.
            // The panic payload is only available later at the supervisor boundary.
            let error = WorkerError::worker_panicked(
                "engine worker panicked while processing a locally owned request (unwinding)",
            );
            self.set_terminal(GenerationEvent::Error(error.clone()));
            if let Some(tap) = self.decision_tap.take() {
                tap.fail(error);
            }
        }
    }
}

#[derive(Default)]
pub struct WorkerStats {
    pub telemetry: Arc<Telemetry>,
    pub active: AtomicU64,
    pub peak_active: AtomicU64,
    pub decode_bucket_mask: AtomicU64,
    pub completed: AtomicU64,
    pub cancelled: AtomicU64,
    pub accepted_drafts: AtomicU64,
    pub verify_rounds: AtomicU64,
    pub cached_tokens: AtomicU64,
}

pub struct EngineWorker {
    sender: Mutex<Option<Sender<Request>>>,
    thread: Mutex<Option<JoinHandle<()>>>,
    shutdown: Arc<AtomicBool>,
    panicked: Arc<AtomicBool>,
    panic_reason: Arc<Mutex<Option<String>>>,
    pub stats: Arc<WorkerStats>,
    pub tokenizer: Option<tokenizers::Tokenizer>,
    pub eos_token_ids: Vec<i32>,
    pub vocab_size: usize,
    pub default_preserve_thinking: bool,
    pub decision_profile: crate::decision::DecisionModelProfile,
    pub vision_config: Option<crate::vision::VisionConfig>,
    pub has_vision_weights: bool,
    pub kv_token_capacity: usize,
}
impl EngineWorker {
    pub fn start(engine: Qwen3_5Engine, config: SchedulerConfig) -> Self {
        Self::from_loader(move || Ok(engine), config).expect("infallible engine transfer")
    }

    /// Returns whether the background `w34-cuda` worker thread terminated due to a panic.
    #[inline]
    pub fn is_panicked(&self) -> bool {
        self.panicked.load(Ordering::Acquire)
    }

    /// Returns the captured panic message if the worker thread panicked.
    pub fn panic_reason(&self) -> Option<String> {
        self.panic_reason
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .clone()
    }

    fn panic_worker_error(&self) -> WorkerError {
        WorkerError::worker_panicked(
            self.panic_reason()
                .unwrap_or_else(|| "unknown panic payload".into()),
        )
    }

    /// Called before starting the async runtime. Initialization, inference, and destruction
    /// all take place on this named OS thread, never on a Tokio worker or blocking pool.
    pub fn from_loader(
        loader: impl FnOnce() -> Result<Qwen3_5Engine, String> + Send + 'static,
        config: SchedulerConfig,
    ) -> Result<Self, String> {
        let (sender, receiver) = mpsc::channel(32);
        let (ready_tx, ready_rx) = std::sync::mpsc::sync_channel(1);
        let shutdown = Arc::new(AtomicBool::new(false));
        let stop = shutdown.clone();
        let panicked = Arc::new(AtomicBool::new(false));
        let worker_panicked = panicked.clone();
        let panic_reason = Arc::new(Mutex::new(None));
        let worker_panic_reason = panic_reason.clone();
        let stats = Arc::new(WorkerStats::default());
        let worker_stats = stats.clone();
        let thread = std::thread::Builder::new()
            .name("w34-cuda".into())
            .spawn(move || match loader() {
                Ok(engine) => {
                    worker_stats.telemetry.initialize(
                        TelemetryConfig {
                            tp_size: engine.tp_size,
                            kv_format: kv_format_name(&engine),
                            speculation: engine
                                .speculative
                                .as_ref()
                                .map_or("none", |p| if p.is_mtp() { "mtp" } else { "dflash2" })
                                .into(),
                        },
                        engine
                            .model_dir
                            .file_name()
                            .unwrap_or_default()
                            .to_string_lossy()
                            .into_owned(),
                    );
                    publish_pools(&engine, &engine.radix, &worker_stats.telemetry);
                    let mut clamped_cfg = config;
                    for rank in &engine.ranks {
                        match rank.ordinary_prefill_token_capacity(clamped_cfg.max_num_seqs) {
                            Ok(cap) => clamped_cfg.clamp_for_token_capacity(cap),
                            Err(e) => {
                                let _ = ready_tx.send(Err(e.to_string()));
                                return;
                            }
                        }
                    }
                    let default_preserve_thinking =
                        crate::chatml::detect_default_preserve_thinking(&engine.model_dir);
                    let kv_token_capacity = engine.ranks[0].kv_pool.num_pages * 16;
                    let info = (
                        engine.tokenizer.clone(),
                        engine.cfg.eos_token_ids.clone(),
                        engine.cfg.vocab_size,
                        default_preserve_thinking,
                        engine.decision_profile.clone(),
                        engine.cfg.vision_config.clone(),
                        engine.has_vision_weights,
                        kv_token_capacity,
                    );
                    if ready_tx.send(Ok(info)).is_ok() {
                        run(
                            engine,
                            clamped_cfg,
                            receiver,
                            stop,
                            worker_stats,
                            worker_panicked,
                            worker_panic_reason,
                        );
                    }
                }
                Err(e) => {
                    let _ = ready_tx.send(Err(e));
                }
            })
            .map_err(|e| e.to_string())?;
        let (
            tokenizer,
            eos_token_ids,
            vocab_size,
            default_preserve_thinking,
            decision_profile,
            vision_config,
            has_vision_weights,
            kv_token_capacity,
        ) = match ready_rx.recv() {
            Ok(Ok(info)) => info,
            Ok(Err(error)) => {
                let _ = thread.join();
                return Err(error);
            }
            Err(error) => {
                let _ = thread.join();
                return Err(error.to_string());
            }
        };
        Ok(Self {
            sender: Mutex::new(Some(sender)),
            thread: Mutex::new(Some(thread)),
            shutdown,
            panicked,
            panic_reason,
            stats,
            tokenizer,
            eos_token_ids,
            vocab_size,
            default_preserve_thinking,
            decision_profile,
            vision_config,
            has_vision_weights,
            kv_token_capacity,
        })
    }

    pub fn request_shutdown(&self) {
        self.stats.telemetry.set_status("stopping");
        self.shutdown.store(true, Ordering::Release);
        self.sender.lock().unwrap_or_else(|e| e.into_inner()).take(); // wake blocking_recv when idle
    }

    pub async fn shutdown(&self) -> Result<(), String> {
        self.request_shutdown();
        let thread = self.thread.lock().unwrap_or_else(|e| e.into_inner()).take();
        if let Some(thread) = thread {
            // Only joining is offloaded; no CUDA state ever enters spawn_blocking.
            tokio::task::spawn_blocking(move || thread.join())
                .await
                .map_err(|e| e.to_string())?
                .map_err(|_| "CUDA worker panicked".to_string())?;
        }
        if self.is_panicked() {
            let reason = self
                .panic_reason()
                .unwrap_or_else(|| "unknown panic payload".into());
            return Err(format!("CUDA worker panicked: {reason}"));
        }
        Ok(())
    }

    fn enqueue_request(&self, mut req: Request) -> Result<(), WorkerError> {
        if self.is_panicked() {
            return Err(self.panic_worker_error());
        }
        let guard = self.sender.lock().unwrap_or_else(|e| e.into_inner());
        let Some(sender) = guard.as_ref() else {
            return if self.is_panicked() {
                Err(self.panic_worker_error())
            } else {
                Err(WorkerError::worker_stopped("engine worker stopped"))
            };
        };
        let permit = sender.try_reserve().map_err(|err| match err {
            mpsc::error::TrySendError::Full(_) => {
                WorkerError::overload("engine ingress queue full")
            }
            mpsc::error::TrySendError::Closed(_) => {
                if self.is_panicked() {
                    self.panic_worker_error()
                } else {
                    WorkerError::worker_stopped("engine worker stopped")
                }
            }
        })?;
        // Reserve first: rejected ingress is HTTP admission, not a submitted engine request.
        // Count before publishing to avoid a worker dequeue racing the waiting increment.
        req.lifecycle = Some(self.stats.telemetry.request(req.prompt.len()));
        permit.send(req);
        Ok(())
    }

    pub fn submit(
        &self,
        prompt: Vec<i32>,
        params: SamplingParams,
        constraint: Option<TokenBitmaskConstraint>,
    ) -> Result<GenerationStream, WorkerError> {
        self.submit_with_images(prompt, Vec::new(), params, constraint)
    }

    pub fn submit_with_images(
        &self,
        prompt: Vec<i32>,
        images: Vec<crate::vision::PreprocessedImage>,
        params: SamplingParams,
        constraint: Option<TokenBitmaskConstraint>,
    ) -> Result<GenerationStream, WorkerError> {
        params.validate().map_err(WorkerError::invalid_request)?;
        if prompt.is_empty() || params.max_tokens == 0 {
            return Err(WorkerError::invalid_request(
                "prompt and output budget must be nonempty",
            ));
        }
        let (output, events) = generation_channel(64);
        let cached_tokens = Arc::new(AtomicU64::new(0));
        let cancelled = Arc::new(AtomicBool::new(false));
        let sampler = SamplerV1::new(params, &prompt, constraint);
        self.enqueue_request(Request {
            lifecycle: None,
            prompt,
            images,
            vision: None,
            reserved_pages: 0,
            cached_tokens: cached_tokens.clone(),
            sampler,
            output,
            terminal: events.terminal.clone(),
            cancelled: cancelled.clone(),
            max_prefix_lookup_len: None,
            mamba_branching_seqlen: None,
            decision_tap: None,
        })?;
        Ok(GenerationStream {
            events,
            cancelled,
            cached_tokens,
        })
    }

    pub fn submit_decision_candidate_logits(
        &self,
        prompt: Vec<i32>,
        token_ids: Vec<i32>,
        mamba_branching_seqlen: Option<usize>,
    ) -> Result<
        (
            GenerationStream,
            tokio::sync::oneshot::Receiver<Result<DecisionTapOutput, WorkerError>>,
        ),
        WorkerError,
    > {
        self.submit_decision_candidate_logits_with_images(
            prompt,
            Vec::new(),
            token_ids,
            mamba_branching_seqlen,
        )
    }

    pub fn submit_decision_candidate_logits_with_images(
        &self,
        prompt: Vec<i32>,
        images: Vec<crate::vision::PreprocessedImage>,
        token_ids: Vec<i32>,
        mamba_branching_seqlen: Option<usize>,
    ) -> Result<
        (
            GenerationStream,
            tokio::sync::oneshot::Receiver<Result<DecisionTapOutput, WorkerError>>,
        ),
        WorkerError,
    > {
        let (tx, rx) = tokio::sync::oneshot::channel();
        let stream = self.submit_decision_internal(
            prompt,
            images,
            None,
            mamba_branching_seqlen,
            DecisionReadoutTap::CandidateLogits { token_ids, tx },
        )?;
        Ok((stream, rx))
    }

    pub fn submit_decision_hidden_positions(
        &self,
        prompt: Vec<i32>,
        positions: Vec<usize>,
        max_prefix_lookup_len: Option<usize>,
        mamba_branching_seqlen: Option<usize>,
    ) -> Result<
        (
            GenerationStream,
            tokio::sync::oneshot::Receiver<Result<DecisionTapOutput, WorkerError>>,
        ),
        WorkerError,
    > {
        self.submit_decision_hidden_positions_with_images(
            prompt,
            Vec::new(),
            positions,
            max_prefix_lookup_len,
            mamba_branching_seqlen,
        )
    }

    pub fn submit_decision_hidden_positions_with_images(
        &self,
        prompt: Vec<i32>,
        images: Vec<crate::vision::PreprocessedImage>,
        positions: Vec<usize>,
        max_prefix_lookup_len: Option<usize>,
        mamba_branching_seqlen: Option<usize>,
    ) -> Result<
        (
            GenerationStream,
            tokio::sync::oneshot::Receiver<Result<DecisionTapOutput, WorkerError>>,
        ),
        WorkerError,
    > {
        let captured = vec![None; positions.len()];
        let (tx, rx) = tokio::sync::oneshot::channel();
        let stream = self.submit_decision_internal(
            prompt,
            images,
            max_prefix_lookup_len,
            mamba_branching_seqlen,
            DecisionReadoutTap::HiddenPositions {
                positions,
                captured,
                tx,
            },
        )?;
        Ok((stream, rx))
    }

    pub fn submit_decision_clef_with_images(
        &self,
        record: Arc<crate::clef::ClefEncodedRecord>,
        images: Vec<crate::vision::PreprocessedImage>,
    ) -> Result<
        (
            GenerationStream,
            tokio::sync::oneshot::Receiver<Result<DecisionTapOutput, WorkerError>>,
        ),
        WorkerError,
    > {
        let prompt = record.input_ids.clone();
        let (tx, rx) = tokio::sync::oneshot::channel();
        // Clef JointSchemaHead cross-attends over every token [0..total_tokens], so disable
        // radix prefix skipping (`max_prefix_lookup_len: Some(0)`).
        let stream = self.submit_decision_internal(
            prompt,
            images,
            Some(0),
            None,
            DecisionReadoutTap::ClefJoint {
                record,
                gpu_state: None,
                tx,
            },
        )?;
        Ok((stream, rx))
    }

    fn submit_decision_internal(
        &self,
        prompt: Vec<i32>,
        images: Vec<crate::vision::PreprocessedImage>,
        max_prefix_lookup_len: Option<usize>,
        mamba_branching_seqlen: Option<usize>,
        tap: DecisionReadoutTap,
    ) -> Result<GenerationStream, WorkerError> {
        if prompt.is_empty() {
            return Err(WorkerError::invalid_request(
                "decision prompt must be nonempty",
            ));
        }
        let (output, events) = generation_channel(8);
        let cached_tokens = Arc::new(AtomicU64::new(0));
        let cancelled = Arc::new(AtomicBool::new(false));
        let params = SamplingParams {
            temperature: 0.0,
            top_k: 1,
            max_tokens: 0,
            ..SamplingParams::default()
        };
        let sampler = SamplerV1::new(params, &prompt, None);
        self.enqueue_request(Request {
            lifecycle: None,
            prompt,
            images,
            vision: None,
            reserved_pages: 0,
            cached_tokens: cached_tokens.clone(),
            sampler,
            output,
            terminal: events.terminal.clone(),
            cancelled: cancelled.clone(),
            max_prefix_lookup_len,
            mamba_branching_seqlen,
            decision_tap: Some(tap),
        })?;
        Ok(GenerationStream {
            events,
            cancelled,
            cached_tokens,
        })
    }
}
impl Drop for EngineWorker {
    fn drop(&mut self) {
        self.request_shutdown();
        // Normal server shutdown awaits shutdown(); Drop must not block an async executor.
    }
}

fn kv_format_name(engine: &Qwen3_5Engine) -> String {
    use w34_kernels::ffi::{
        W34_KV_FORMAT_BF16, W34_KV_FORMAT_FP16, W34_KV_FORMAT_Q8_0, W34_KV_FORMAT_TQ4,
    };
    let layers = &engine.ranks[0].kv_pool.layers;
    let all = |format| {
        layers
            .iter()
            .all(|l| l.layout.k_format == format && l.layout.v_format == format)
    };
    if layers.is_empty() {
        "unknown"
    } else if all(W34_KV_FORMAT_FP16) {
        "fp16"
    } else if all(W34_KV_FORMAT_BF16) {
        "bf16"
    } else if all(W34_KV_FORMAT_Q8_0) {
        "q8_0"
    } else if all(W34_KV_FORMAT_TQ4) {
        "tq4_all"
    } else if layers.iter().all(|l| {
        [W34_KV_FORMAT_Q8_0, W34_KV_FORMAT_TQ4].contains(&l.layout.k_format)
            && l.layout.k_format == l.layout.v_format
    }) {
        "tq4"
    } else {
        "mixed"
    }
    .into()
}

fn publish_pools(
    engine: &Qwen3_5Engine,
    radix: &crate::radix::MambaRadixCache,
    telemetry: &Telemetry,
) {
    // Logical allocation is replicated across ranks; summing TP would double count.
    let rank = &engine.ranks[0];
    telemetry.publish_pools(Pools {
        kv: Pool::known(
            rank.kv_pool.num_pages,
            rank.kv_pool.free_count(),
            radix.full_evictable_pages(),
        ),
        gdn: Pool::known(
            rank.state_pool.total_slots,
            rank.state_pool.free_count(),
            radix.mamba_evictable_slots(),
        ),
    });
}

fn release(engine: &mut Qwen3_5Engine, donation: Option<crate::radix::RadixDonationOutcome>) {
    if let Some(d) = donation {
        engine.free_kv_pages(&d.freed_kv_pages);
        engine.free_state_slots(&d.freed_mamba_slots);
    }
}

fn enqueue(
    engine: &Qwen3_5Engine,
    sched: &mut SchedulerV1,
    requests: &mut HashMap<u64, Request>,
    mut request: Request,
) {
    if requests.len() >= 32 {
        let err = WorkerError::overload("engine admission capacity exceeded");
        if let Some(tap) = request.decision_tap.take() {
            tap.fail(err.clone());
        }
        request.set_terminal(GenerationEvent::Error(err));
        return;
    }
    let vocab = engine.cfg.vocab_size;
    let p = &mut request.sampler.params;
    let valid = request
        .prompt
        .iter()
        .chain(p.stop_token_ids.iter())
        .chain(p.eos_token_ids.iter())
        .all(|&id| id >= 0 && (id as usize) < vocab)
        && p.logit_bias
            .keys()
            .all(|&id| id >= 0 && (id as usize) < vocab)
        && request
            .prompt
            .len()
            .checked_add(p.max_tokens)
            .is_some_and(|n| n.div_ceil(16) <= engine.ranks[0].kv_pool.num_pages);
    if !valid {
        let err = WorkerError::invalid_request("invalid tokens or request exceeds KV capacity");
        if let Some(tap) = request.decision_tap.take() {
            tap.fail(err.clone());
        }
        request.set_terminal(GenerationEvent::Error(err));
        return;
    }
    request.reserved_pages = (request.prompt.len() + p.max_tokens).div_ceil(16);
    p.eos_token_ids = engine.cfg.eos_token_ids.clone();
    if !request.images.is_empty() {
        match crate::scheduled_vision::ScheduledVision::new(
            engine,
            &request.prompt,
            std::mem::take(&mut request.images),
        ) {
            Ok(vision) => request.vision = Some(vision),
            Err(error) => {
                let err = WorkerError::invalid_request(error);
                if let Some(tap) = request.decision_tap.take() {
                    tap.fail(err.clone());
                }
                request.set_terminal(GenerationEvent::Error(err));
                return;
            }
        }
    }
    let key = request
        .vision
        .as_ref()
        .map_or_else(|| request.prompt.clone(), |v| v.radix_key.clone());
    let id = sched.submit_request_with_prefix_options(
        request.prompt.clone(),
        key,
        p.max_tokens,
        p.min_tokens,
        p.ignore_eos,
        p.stop_token_ids.clone(),
        p.eos_token_ids.clone(),
        request.vision.is_some(),
        request.max_prefix_lookup_len,
        request.mamba_branching_seqlen,
    );
    requests.insert(id, request);
}

fn panic_payload_message(payload: &(dyn std::any::Any + Send)) -> String {
    if let Some(msg) = payload.downcast_ref::<&'static str>() {
        (*msg).to_string()
    } else if let Some(msg) = payload.downcast_ref::<String>() {
        msg.clone()
    } else {
        "unknown panic payload".to_string()
    }
}

fn notify_request_error(mut r: Request, err: WorkerError) {
    r.set_terminal(GenerationEvent::Error(err.clone()));
    if let Some(tap) = r.decision_tap.take() {
        tap.fail(err);
    }
}

/// On any exit, closes ingress and terminates active and queued requests. A panic
/// first publishes dead health and its actual diagnostic; normal return reports
/// [`WorkerError::WorkerStopped`]. CUDA engine ownership must stay outside `body`
/// so engine teardown cannot delay publication of poisoned health.
fn run_supervised_loop(
    receiver: &mut Receiver<Request>,
    requests: &mut HashMap<u64, Request>,
    stats: &WorkerStats,
    panicked: &AtomicBool,
    panic_reason: &Mutex<Option<String>>,
    body: impl FnOnce(&mut Receiver<Request>, &mut HashMap<u64, Request>),
) {
    let outcome = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        body(receiver, requests);
    }));
    let err = if let Err(payload) = outcome {
        let reason = panic_payload_message(&*payload);
        eprintln!("[w34-cuda] FATAL: engine worker thread panicked: {reason}");
        *panic_reason.lock().unwrap_or_else(|e| e.into_inner()) = Some(reason.clone());
        panicked.store(true, Ordering::Release);
        stats.telemetry.set_status("unhealthy");
        WorkerError::worker_panicked(reason)
    } else {
        WorkerError::worker_stopped("engine worker stopped")
    };
    stats.telemetry.set_status("stopping");
    stats.active.store(0, Ordering::Relaxed);
    receiver.close();
    for (_, r) in requests.drain() {
        notify_request_error(r, err.clone());
    }
    while let Ok(r) = receiver.try_recv() {
        notify_request_error(r, err.clone());
    }
}

fn run(
    mut engine: Qwen3_5Engine,
    config: SchedulerConfig,
    mut receiver: Receiver<Request>,
    shutdown: Arc<AtomicBool>,
    stats: Arc<WorkerStats>,
    panicked: Arc<AtomicBool>,
    panic_reason: Arc<Mutex<Option<String>>>,
) {
    let mut requests = HashMap::new();
    run_supervised_loop(
        &mut receiver,
        &mut requests,
        &stats,
        &panicked,
        &panic_reason,
        |receiver, requests| {
            // Borrow only: CUDA destruction (which may synchronize) must happen after
            // the supervisor publishes poisoned health and notifies retained callers.
            run_loop(&mut engine, config, receiver, requests, &shutdown, &stats);
        },
    );
}

type MtpBatchSlots = [Option<(u64, Vec<i32>)>; crate::mtp::MAX_MTP_BATCH_SLOTS];

/// Tick-to-tick speculative bookkeeping only. Engine ownership remains in `run`,
/// outside the supervisor's caught closure.
#[derive(Default)]
struct SpeculativeLoopState {
    speculative_id: Option<u64>,
    pending_ingest: Vec<i32>,
    mtp_batch_slots: MtpBatchSlots,
}

fn run_loop(
    engine: &mut Qwen3_5Engine,
    config: SchedulerConfig,
    receiver: &mut Receiver<Request>,
    requests: &mut HashMap<u64, Request>,
    shutdown: &AtomicBool,
    stats: &WorkerStats,
) {
    let mut sched = SchedulerV1::new(config);
    let mut speculation = SpeculativeLoopState::default();
    let mut prev_active_peak = 1usize;
    loop {
        if shutdown.load(Ordering::Acquire) {
            break;
        }
        let first = if requests.is_empty() {
            match receiver.blocking_recv() {
                Some(r) => Some(r),
                None => break,
            }
        } else {
            None
        };
        let was_idle = first.is_some();
        // The engine retains the canonical radix between ticks (including diagnostics).
        std::mem::swap(&mut engine.radix, &mut sched.radix);
        if let Some(r) = first {
            enqueue(engine, &mut sched, requests, r);
        }
        // Bound queue draining even if producers keep replenishing it.
        for _ in 0..32 {
            match receiver.try_recv() {
                Ok(r) => enqueue(engine, &mut sched, requests, r),
                Err(_) => break,
            }
        }
        if was_idle && requests.len() < config.max_num_seqs && !shutdown.load(Ordering::Acquire) {
            coalesce_idle_burst(
                engine,
                &mut sched,
                requests,
                receiver,
                shutdown,
                prev_active_peak,
            );
        }
        if !requests.is_empty() {
            prev_active_peak = requests.len();
        }
        cancel_requests(engine, &mut sched, requests, &mut speculation, stats);
        yield_single_speculation(engine, &mut sched, requests, &mut speculation);
        extend_or_yield_mtp_batch(
            engine,
            &mut sched,
            requests,
            &mut speculation.mtp_batch_slots,
        );
        admit_requests(engine, &mut sched, requests, &mut speculation);
        reserve_request_capacity(engine, &mut sched, requests, &mut speculation, stats);
        publish_pools(engine, &sched.radix, &stats.telemetry);

        let result = if let Some(id) = speculation.speculative_id {
            let Some(result) =
                step_single_speculation(engine, &mut sched, requests, &mut speculation, id, stats)
            else {
                // A constraint error fails only this request and skips tick completion.
                std::mem::swap(&mut engine.radix, &mut sched.radix);
                continue;
            };
            result
        } else if speculation.mtp_batch_slots.iter().any(|s| s.is_some()) {
            step_mtp_batch(
                engine,
                &mut sched,
                requests,
                &mut speculation.mtp_batch_slots,
                stats,
            )
        } else {
            step_plain(engine, &mut sched, requests, receiver, shutdown)
        };
        if let Err(error) = result {
            abort_active_batch(engine, &mut sched, requests, &mut speculation, error);
        }
        complete_tick(engine, &mut sched, requests, stats);
        publish_pools(engine, &sched.radix, &stats.telemetry);
        std::mem::swap(&mut engine.radix, &mut sched.radix);
    }
    std::mem::swap(&mut engine.radix, &mut sched.radix);
    let ids: Vec<_> = requests.keys().copied().collect();
    for id in ids {
        release(engine, sched.cancel_request(id));
    }
    if let Some(p) = &mut engine.speculative {
        p.abort_scheduled();
    }
    std::mem::swap(&mut engine.radix, &mut sched.radix);
    publish_pools(engine, &engine.radix, &stats.telemetry);
}

// When waking from idle, briefly coalesce concurrent HTTP burst peers that are
// finishing socket connect + tokenization (including Python client GIL bursts and
// wave-to-wave batch turnarounds) so simultaneous N=8/16 arrivals batch onto tick 1
// instead of staggering request 0 into a lone prefill pass.
fn coalesce_idle_burst(
    engine: &Qwen3_5Engine,
    sched: &mut SchedulerV1,
    requests: &mut HashMap<u64, Request>,
    receiver: &mut Receiver<Request>,
    shutdown: &AtomicBool,
    prev_active_peak: usize,
) {
    let burst_start = std::time::Instant::now();
    let mut last_arrival = burst_start;
    let spec_active = engine.speculative.is_some();
    while requests.len() < sched.cfg.max_num_seqs && !shutdown.load(Ordering::Acquire) {
        if prev_active_peak >= 2 && requests.len() >= prev_active_peak {
            break;
        }
        let max_window = if requests.len() >= 2 || prev_active_peak >= 2 {
            std::time::Duration::from_millis(8)
        } else {
            std::time::Duration::from_micros(3000)
        };
        if burst_start.elapsed() >= max_window {
            break;
        }
        match receiver.try_recv() {
            Ok(r) => {
                enqueue(engine, sched, requests, r);
                last_arrival = std::time::Instant::now();
            }
            Err(mpsc::error::TryRecvError::Disconnected) => break,
            Err(mpsc::error::TryRecvError::Empty) => {
                let quiet_cap = if prev_active_peak >= 2 {
                    std::time::Duration::from_micros(3500)
                } else if requests.len() <= 1 {
                    if spec_active {
                        std::time::Duration::from_micros(2500)
                    } else {
                        std::time::Duration::from_micros(600)
                    }
                } else {
                    std::time::Duration::from_micros(1500)
                };
                if last_arrival.elapsed() >= quiet_cap {
                    break;
                }
                std::thread::sleep(std::time::Duration::from_micros(40));
            }
        }
    }
}

fn cancel_requests(
    engine: &mut Qwen3_5Engine,
    sched: &mut SchedulerV1,
    requests: &mut HashMap<u64, Request>,
    speculation: &mut SpeculativeLoopState,
    stats: &WorkerStats,
) {
    let SpeculativeLoopState {
        speculative_id,
        pending_ingest,
        mtp_batch_slots,
    } = speculation;
    let cancelled: Vec<_> = requests
        .iter()
        .filter_map(|(&id, r)| r.cancelled.load(Ordering::Acquire).then_some(id))
        .collect();
    for id in cancelled {
        stats.cancelled.fetch_add(1, Ordering::Relaxed);
        if *speculative_id == Some(id) {
            if let Some(p) = &mut engine.speculative {
                p.abort_scheduled();
            }
            *speculative_id = None;
            pending_ingest.clear();
        }
        for (slot_idx, slot_entry) in mtp_batch_slots.iter_mut().enumerate() {
            if slot_entry.as_ref().is_some_and(|(sid, _)| *sid == id) {
                *slot_entry = None;
                if let Some(p) = &mut engine.speculative {
                    let _ = p.reset_mtp_batch_slot(slot_idx);
                }
            }
        }
        if mtp_batch_slots.iter().all(|s| s.is_none())
            && let Some(p) = &mut engine.speculative
            && p.is_mtp_batch_mode()
        {
            let _ = p.set_mtp_batch_mode(false);
        }
        release(engine, sched.cancel_request(id));
        if let Some(mut r) = requests.remove(&id)
            && let Some(tap) = r.decision_tap.take()
        {
            tap.recycle_into(engine);
        }
    }
}

fn is_batched_mtp_eligible(r: &Request) -> bool {
    r.decision_tap.is_none()
        && r.sampler.params.max_tokens > 0
        && r.sampler.constraint.is_none()
        && (r.sampler.params.temperature <= 1e-6 || r.sampler.params.top_k == 1)
        && !r.sampler.params.logprobs
}

fn yield_single_speculation(
    engine: &mut Qwen3_5Engine,
    sched: &mut SchedulerV1,
    requests: &mut HashMap<u64, Request>,
    speculation: &mut SpeculativeLoopState,
) {
    let SpeculativeLoopState {
        speculative_id,
        pending_ingest,
        mtp_batch_slots,
    } = speculation;
    // Yield only at a committed tick boundary. Verification has resolved its
    // StateTx and released extra slots; the final emitted anchor is still
    // unconsumed, exactly as ordinary scheduler decode expects. Keep target
    // state, generated tokens, and sampler intact. pending_ingest contains
    // only deferred drafter features, not uncommitted target work.
    // Partial prefill stays speculative until its final chunk commits.
    if !sched.waiting.is_empty()
        && let Some(id) = *speculative_id
        && let Some(seq) = sched.active.iter_mut().find(|s| s.req_id == id)
        && seq.is_prefill_complete()
    {
        let active_need = (seq.prompt_tokens.len() + seq.max_new_tokens.max(1) + 4)
            .div_ceil(16)
            .max(requests[&id].reserved_pages);
        let waiting_need: usize = sched
            .waiting
            .iter()
            .map(|w| {
                (w.prompt_tokens.len() + w.max_new_tokens.max(1) + 4)
                    .div_ceil(16)
                    .max(requests[&w.req_id].reserved_pages)
            })
            .sum();
        let can_promote_to_mtp_batch = sched.waiting.len() < crate::mtp::MAX_MTP_BATCH_SLOTS
            && sched.waiting.len() < sched.cfg.max_num_seqs
            && active_need + waiting_need <= engine.ranks[0].kv_pool.num_pages
            && is_batched_mtp_eligible(&requests[&id])
            && engine.speculative.as_ref().is_some_and(|p| {
                p.is_mtp()
                    && p.fits_mtp_batch_slot_capacity(seq.prompt_tokens.len(), seq.max_new_tokens)
                    && sched.waiting.iter().all(|w| {
                        let r = &requests[&w.req_id];
                        !w.custom_prefill
                            && is_batched_mtp_eligible(r)
                            && p.fits_mtp_batch_slot_capacity(
                                w.prompt_tokens.len(),
                                w.max_new_tokens,
                            )
                    })
            });
        let promoted = if can_promote_to_mtp_batch {
            let mut proposer = engine.speculative.take().unwrap();
            let ok = proposer
                .promote_single_to_mtp_batch_slot0(engine, pending_ingest.len())
                .is_ok();
            engine.speculative = Some(proposer);
            ok
        } else {
            false
        };
        if promoted {
            let req = requests.get_mut(&id).unwrap();
            req.reserved_pages = req.reserved_pages.max(active_need);
            mtp_batch_slots[0] = Some((id, std::mem::take(pending_ingest)));
            *speculative_id = None;
        } else {
            engine.speculative.as_mut().unwrap().abort_scheduled();
            pending_ingest.clear();
            *speculative_id = None;
            seq.custom_prefill = false;
        }
    }
}

fn extend_or_yield_mtp_batch(
    engine: &mut Qwen3_5Engine,
    sched: &mut SchedulerV1,
    requests: &mut HashMap<u64, Request>,
    mtp_batch_slots: &mut MtpBatchSlots,
) {
    // Handle waiting arrivals while a B=2..4 batched MTP burst is active:
    // either admit eligible plain-greedy arrivals into free MTP slots, or yield
    // cleanly at a committed post-prefill tick boundary when C > 4 / non-greedy arrives.
    if !sched.waiting.is_empty() && mtp_batch_slots.iter().any(|s| s.is_some()) {
        let active_mtp = mtp_batch_slots.iter().filter(|s| s.is_some()).count();
        let active_reserved: usize = sched
            .active
            .iter()
            .map(|seq| requests[&seq.req_id].reserved_pages)
            .sum();
        let waiting_reserved: usize = sched
            .waiting
            .iter()
            .map(|w| {
                (w.prompt_tokens.len() + w.max_new_tokens.max(1) + 4)
                    .div_ceil(16)
                    .max(requests[&w.req_id].reserved_pages)
            })
            .sum();
        let can_extend_mtp = active_mtp + sched.waiting.len() <= crate::mtp::MAX_MTP_BATCH_SLOTS
            && active_mtp + sched.waiting.len() <= sched.cfg.max_num_seqs
            && active_reserved + waiting_reserved <= engine.ranks[0].kv_pool.num_pages
            && engine.speculative.as_ref().is_some_and(|p| {
                p.is_mtp()
                    && sched.waiting.iter().all(|w| {
                        let r = &requests[&w.req_id];
                        !w.custom_prefill
                            && is_batched_mtp_eligible(r)
                            && p.fits_mtp_batch_slot_capacity(
                                w.prompt_tokens.len(),
                                w.max_new_tokens,
                            )
                    })
            });
        if can_extend_mtp {
            for w in &sched.waiting {
                let need_pages = (w.prompt_tokens.len() + w.max_new_tokens.max(1) + 4).div_ceil(16);
                let req = requests.get_mut(&w.req_id).unwrap();
                req.reserved_pages = req.reserved_pages.max(need_pages);
            }
            let mut proposer = engine.speculative.take().unwrap();
            let saved_max = sched.cfg.max_num_seqs;
            while let Some(slot_idx) = mtp_batch_slots.iter().position(|s| s.is_none()) {
                if sched.waiting.is_empty() {
                    break;
                }
                let before = sched.active.len();
                sched.cfg.max_num_seqs = before + 1;
                sched.admit_waiting_with_filter(
                    |r| engine.alloc_three_state_slots_from(r),
                    |_, _, matched| matched == 0,
                );
                record_admissions(sched, requests);
                if sched.active.len() == before {
                    break;
                }
                let seq = sched.active.last_mut().unwrap();
                seq.custom_prefill = true;
                let _ = proposer.reset_mtp_batch_slot(slot_idx);
                mtp_batch_slots[slot_idx] = Some((seq.req_id, Vec::new()));
            }
            sched.cfg.max_num_seqs = saved_max;
            engine.speculative = Some(proposer);
        } else {
            let all_prefilled = mtp_batch_slots.iter().flatten().all(|(id, _)| {
                sched
                    .active
                    .iter()
                    .find(|s| s.req_id == *id)
                    .is_none_or(|s| s.is_prefill_complete())
            });
            if all_prefilled {
                if let Some(p) = &mut engine.speculative {
                    let _ = p.set_mtp_batch_mode(false);
                }
                for slot_entry in mtp_batch_slots {
                    if let Some((id, _)) = slot_entry.take()
                        && let Some(seq) = sched.active.iter_mut().find(|s| s.req_id == id)
                    {
                        seq.custom_prefill = false;
                    }
                }
            }
        }
    }
}

fn admit_requests(
    engine: &mut Qwen3_5Engine,
    sched: &mut SchedulerV1,
    requests: &mut HashMap<u64, Request>,
    speculation: &mut SpeculativeLoopState,
) {
    let SpeculativeLoopState {
        speculative_id,
        mtp_batch_slots,
        ..
    } = speculation;
    // A lone request uses the single-slot proposer (MTP or DFlash2).
    // A B=2..4 plain-greedy burst with MTP uses batched multi-slot MTP verification.
    if speculative_id.is_none() && mtp_batch_slots.iter().all(|s| s.is_none()) {
        if sched.active.is_empty()
            && sched.waiting.len() == 1
            && !sched.waiting[0].custom_prefill
            && requests[&sched.waiting[0].req_id].decision_tap.is_none()
            && requests[&sched.waiting[0].req_id].sampler.params.max_tokens > 0
            && engine.speculative.is_some()
        {
            let id = sched.waiting[0].req_id;
            let mut proposer = engine.speculative.take().unwrap();
            let _ = proposer.set_mtp_batch_mode(false);
            let p = &sched.waiting[0];
            if let Some(total) = proposer.fits_capacity(p.prompt_tokens.len(), p.max_new_tokens)
                && total.div_ceil(16) <= engine.ranks[0].kv_pool.num_pages
            {
                requests.get_mut(&id).unwrap().reserved_pages = total.div_ceil(16);
                let saved_max = sched.cfg.max_num_seqs;
                sched.cfg.max_num_seqs = 1;
                sched.admit_waiting_with_filter(
                    |r| engine.alloc_three_state_slots_from(r),
                    |_, prompt, matched| proposer.can_restore_prefix(&prompt[..matched]),
                );
                sched.cfg.max_num_seqs = saved_max;
                record_admissions(sched, requests);
                if let Some(seq) = sched.active.first_mut() {
                    match proposer
                        .restore_prefix(engine, &seq.prompt_tokens[..seq.cached_prefix_tokens])
                    {
                        Ok(()) => {
                            *speculative_id = Some(id);
                            seq.custom_prefill = true;
                        }
                        Err(e) => {
                            let r = requests.remove(&id).unwrap();
                            notify_request_error(r, WorkerError::Cuda(e));
                            release(engine, sched.cancel_request(id));
                            proposer.abort_scheduled();
                        }
                    }
                }
            }
            engine.speculative = Some(proposer);
        } else if sched.active.is_empty()
            && (2..=crate::mtp::MAX_MTP_BATCH_SLOTS).contains(&sched.waiting.len())
            && sched.waiting.len() <= sched.cfg.max_num_seqs
            && engine.speculative.as_ref().is_some_and(|p| p.is_mtp())
        {
            let mut proposer = engine.speculative.take().unwrap();
            let total_pages: usize = sched
                .waiting
                .iter()
                .map(|w| {
                    (w.prompt_tokens.len() + w.max_new_tokens.max(1) + 4)
                        .div_ceil(16)
                        .max(requests[&w.req_id].reserved_pages)
                })
                .sum();
            let eligible = total_pages <= engine.ranks[0].kv_pool.num_pages
                && sched.waiting.iter().all(|w| {
                    let r = &requests[&w.req_id];
                    !w.custom_prefill
                        && is_batched_mtp_eligible(r)
                        && proposer
                            .fits_mtp_batch_slot_capacity(w.prompt_tokens.len(), w.max_new_tokens)
                });
            if eligible && proposer.set_mtp_batch_mode(true).is_ok() {
                for w in &sched.waiting {
                    let need_pages =
                        (w.prompt_tokens.len() + w.max_new_tokens.max(1) + 4).div_ceil(16);
                    let req = requests.get_mut(&w.req_id).unwrap();
                    req.reserved_pages = req.reserved_pages.max(need_pages);
                }
                let saved_max = sched.cfg.max_num_seqs;
                let mut slot_idx = 0usize;
                while slot_idx < crate::mtp::MAX_MTP_BATCH_SLOTS && !sched.waiting.is_empty() {
                    let before = sched.active.len();
                    sched.cfg.max_num_seqs = before + 1;
                    sched.admit_waiting_with_filter(
                        |r| engine.alloc_three_state_slots_from(r),
                        |_, _, matched| matched == 0,
                    );
                    record_admissions(sched, requests);
                    if sched.active.len() == before {
                        break;
                    }
                    let seq = sched.active.last_mut().unwrap();
                    seq.custom_prefill = true;
                    let _ = proposer.reset_mtp_batch_slot(slot_idx);
                    mtp_batch_slots[slot_idx] = Some((seq.req_id, Vec::new()));
                    slot_idx += 1;
                }
                sched.cfg.max_num_seqs = saved_max;
                if mtp_batch_slots.iter().all(|s| s.is_none()) {
                    let _ = proposer.set_mtp_batch_mode(false);
                }
            }
            engine.speculative = Some(proposer);
        }
        if speculative_id.is_none() && mtp_batch_slots.iter().all(|s| s.is_none()) {
            let mut reserved: usize = sched
                .active
                .iter()
                .map(|seq| requests[&seq.req_id].reserved_pages)
                .sum();
            let limit = sched.cfg.max_num_seqs;
            while sched.active.len() < limit {
                let Some(waiting) = sched.waiting.front() else {
                    break;
                };
                let pages = requests[&waiting.req_id].reserved_pages;
                if reserved + pages > engine.ranks[0].kv_pool.num_pages {
                    break;
                }
                let before = sched.active.len();
                // Admit exactly one request so its complete budget is reserved before
                // considering the next. Existing requests can never lose capacity.
                sched.cfg.max_num_seqs = before + 1;
                sched.admit_waiting(|r| engine.alloc_three_state_slots_from(r));
                record_admissions(sched, requests);
                sched.cfg.max_num_seqs = limit;
                if sched.active.len() == before {
                    break;
                }
                reserved += pages;
            }
        }
    }
}

fn record_admissions(sched: &SchedulerV1, requests: &HashMap<u64, Request>) {
    for seq in &sched.active {
        if let Some(life) = &requests[&seq.req_id].lifecycle {
            life.scheduled(seq.cached_prefix_tokens);
        }
    }
}

fn reserve_request_capacity(
    engine: &mut Qwen3_5Engine,
    sched: &mut SchedulerV1,
    requests: &mut HashMap<u64, Request>,
    speculation: &mut SpeculativeLoopState,
    stats: &WorkerStats,
) {
    let SpeculativeLoopState {
        speculative_id,
        pending_ingest,
        mtp_batch_slots,
    } = speculation;
    if sched.active.is_empty() && !sched.waiting.is_empty() {
        // Admission failed with no active work that could release state slots.
        let ids: Vec<_> = sched.waiting.iter().map(|w| w.req_id).collect();
        for id in ids {
            sched.cancel_request(id);
            if let Some(r) = requests.remove(&id) {
                notify_request_error(r, WorkerError::overload("state pool exhausted"));
            }
        }
    }
    stats
        .peak_active
        .fetch_max(sched.active.len() as u64, Ordering::Relaxed);
    let mut failed = Vec::new();
    for seq in &mut sched.active {
        requests[&seq.req_id]
            .cached_tokens
            .store(seq.cached_prefix_tokens as u64, Ordering::Release);
        let needed = requests[&seq.req_id].reserved_pages;
        if !engine.reserve_kv_pages_from(&mut sched.radix, &mut seq.radix_state.kv_pages, needed) {
            failed.push((seq.req_id, WorkerError::overload("KV pool exhausted")));
        }
    }
    for (id, error) in failed {
        if let Some(r) = requests.remove(&id) {
            notify_request_error(r, error);
        }
        release(engine, sched.cancel_request(id));
        if *speculative_id == Some(id) {
            engine.speculative.as_mut().unwrap().abort_scheduled();
            *speculative_id = None;
            pending_ingest.clear();
        }
        for (slot_idx, slot_entry) in mtp_batch_slots.iter_mut().enumerate() {
            if slot_entry.as_ref().is_some_and(|(sid, _)| *sid == id) {
                *slot_entry = None;
                if let Some(p) = &mut engine.speculative {
                    let _ = p.reset_mtp_batch_slot(slot_idx);
                }
            }
        }
        if mtp_batch_slots.iter().all(|s| s.is_none())
            && let Some(p) = &mut engine.speculative
            && p.is_mtp_batch_mode()
        {
            let _ = p.set_mtp_batch_mode(false);
        }
    }
}

/// `None` means a request-local constraint failure: restore the radix and continue
/// the loop without batch error handling, tick stats, or finished-event publication.
fn step_single_speculation(
    engine: &mut Qwen3_5Engine,
    sched: &mut SchedulerV1,
    requests: &mut HashMap<u64, Request>,
    speculation: &mut SpeculativeLoopState,
    id: u64,
    stats: &WorkerStats,
) -> Option<Result<(), WorkerError>> {
    let SpeculativeLoopState {
        speculative_id,
        pending_ingest,
        ..
    } = speculation;
    let mut proposer = engine.speculative.take().unwrap();
    let seq = sched.active.iter_mut().find(|s| s.req_id == id).unwrap();
    let r = requests.get_mut(&id).unwrap();
    let gpu = !r
        .sampler
        .params
        .is_plain_greedy(r.sampler.constraint.is_some());
    if let Err(err) = r.sampler.materialize_pending_constraint() {
        engine.speculative = Some(proposer);
        fail_request(
            engine,
            sched,
            requests,
            id,
            WorkerError::invalid_request(err),
        );
        *speculative_id = None;
        pending_ingest.clear();
        return None;
    }
    let prefilled_before = seq.prefilled_tokens;
    let result = if !seq.is_prefill_complete() {
        proposer
            .step_scheduled_prefill_chunk(
                engine,
                &mut sched.radix,
                seq,
                &mut r.sampler,
                gpu,
                sched.cfg.chunk_prefill_size,
            )
            .map(|token| token.into_iter().collect::<Vec<_>>())
    } else {
        proposer
            .step_scheduled_verify_round(
                engine,
                &mut sched.radix,
                seq,
                pending_ingest,
                &mut r.sampler,
                gpu,
            )
            .map(|(tokens, lps)| {
                stats.telemetry.speculation(
                    1,
                    proposer.last_verified_draft_tokens as u64,
                    proposer.last_accepted.last().copied().unwrap_or(0) as u64,
                );
                stats.verify_rounds.fetch_add(1, Ordering::Relaxed);
                stats.accepted_drafts.fetch_add(
                    proposer.last_accepted.last().copied().unwrap_or(0) as u64,
                    Ordering::Relaxed,
                );
                tokens
                    .into_iter()
                    .enumerate()
                    .map(|(i, t)| (t, lps.get(i).cloned()))
                    .collect()
            })
    };
    r.computed(seq.prefilled_tokens - prefilled_before);
    let result = result.and_then(|tokens| {
        for (token, lp) in tokens {
            emit(r, token, lp);
        }
        if seq.finished {
            proposer.finish_scheduled(engine, pending_ingest)?;
        }
        Ok(())
    });
    engine.speculative = Some(proposer);
    if seq.finished {
        release(engine, sched.finish_request_early(id));
        *speculative_id = None;
    }
    Some(result.map_err(WorkerError::Cuda))
}

fn step_mtp_batch(
    engine: &mut Qwen3_5Engine,
    sched: &mut SchedulerV1,
    requests: &mut HashMap<u64, Request>,
    mtp_batch_slots: &mut MtpBatchSlots,
    stats: &WorkerStats,
) -> Result<(), WorkerError> {
    let mut proposer = engine.speculative.take().unwrap();
    let prefill_target = mtp_batch_slots
        .iter()
        .enumerate()
        .filter_map(|(slot_idx, entry)| entry.as_ref().map(|(id, _)| (slot_idx, *id)))
        .find(|(_, id)| {
            sched
                .active
                .iter()
                .any(|s| s.req_id == *id && !s.is_prefill_complete())
        });
    let res: std::result::Result<(), WorkerError> = if let Some((slot_idx, id)) = prefill_target {
        let seq = sched.active.iter_mut().find(|s| s.req_id == id).unwrap();
        let r = requests.get_mut(&id).unwrap();
        let gpu = !r
            .sampler
            .params
            .is_plain_greedy(r.sampler.constraint.is_some());
        let prefilled_before = seq.prefilled_tokens;
        let step = proposer
            .select_mtp_batch_slot(slot_idx)
            .and_then(|()| {
                proposer.step_scheduled_prefill_chunk(
                    engine,
                    &mut sched.radix,
                    seq,
                    &mut r.sampler,
                    gpu,
                    sched.cfg.chunk_prefill_size,
                )
            })
            .map_err(WorkerError::Cuda);
        r.computed(seq.prefilled_tokens - prefilled_before);
        match step {
            Ok(token) => {
                if let Some((tok, lp)) = token {
                    emit(r, tok, lp);
                }
                if seq.finished {
                    let _ = proposer.reset_mtp_batch_slot(slot_idx);
                    mtp_batch_slots[slot_idx] = None;
                    release(engine, sched.finish_request_early(id));
                    if mtp_batch_slots.iter().all(|s| s.is_none()) {
                        let _ = proposer.set_mtp_batch_mode(false);
                    }
                }
                Ok(())
            }
            Err(e) => Err(e),
        }
    } else {
        // All active MTP batch slots have completed prefill: execute one shared
        // batched target verification forward pass across all occupied slots.
        let slot_map: Vec<(usize, u64)> = mtp_batch_slots
            .iter()
            .enumerate()
            .filter_map(|(slot_idx, entry)| entry.as_ref().map(|(id, _)| (slot_idx, *id)))
            .collect();
        let mut pulled: Vec<(usize, u64, Vec<i32>, Request)> = Vec::with_capacity(slot_map.len());
        for &(slot_idx, id) in &slot_map {
            let (_, pending_vec) = mtp_batch_slots[slot_idx].take().unwrap();
            let req = requests.remove(&id).unwrap();
            pulled.push((slot_idx, id, pending_vec, req));
        }
        let round_res = {
            let mut seq_refs: Vec<&mut crate::scheduler::SeqEntry> = sched
                .active
                .iter_mut()
                .filter(|s| pulled.iter().any(|(_, id, _, _)| *id == s.req_id))
                .collect();
            // Align `pulled` order to `seq_refs` order cleanly.
            pulled.sort_by_key(|(_, id, _, _)| {
                seq_refs.iter().position(|s| s.req_id == *id).unwrap()
            });
            let mut items: Vec<(
                usize,
                &mut crate::scheduler::SeqEntry,
                &mut Vec<i32>,
                &mut crate::parser::SamplerV1,
            )> = seq_refs
                .iter_mut()
                .zip(pulled.iter_mut())
                .map(|(seq, (slot_idx, _, pending_vec, req))| {
                    (*slot_idx, &mut **seq, pending_vec, &mut req.sampler)
                })
                .collect();
            proposer.step_scheduled_batched_mtp_verify_round(engine, &mut sched.radix, &mut items)
        };
        match round_res {
            Ok(all_emitted) => {
                let n = all_emitted.len();
                stats.verify_rounds.fetch_add(n as u64, Ordering::Relaxed);
                let acc_sum: usize = proposer.last_accepted.iter().rev().take(n).copied().sum();
                stats.telemetry.speculation(
                    n as u64,
                    proposer.last_verified_draft_tokens as u64,
                    acc_sum as u64,
                );
                stats
                    .accepted_drafts
                    .fetch_add(acc_sum as u64, Ordering::Relaxed);
                let mut finished_ids = Vec::new();
                for ((slot_idx, id, pending_vec, req), emitted) in
                    pulled.into_iter().zip(all_emitted)
                {
                    for tok in emitted {
                        emit(&req, tok, None);
                    }
                    let is_done = sched
                        .active
                        .iter()
                        .find(|s| s.req_id == id)
                        .is_some_and(|s| s.finished);
                    requests.insert(id, req);
                    if is_done {
                        let _ = proposer.reset_mtp_batch_slot(slot_idx);
                        mtp_batch_slots[slot_idx] = None;
                        finished_ids.push(id);
                    } else {
                        mtp_batch_slots[slot_idx] = Some((id, pending_vec));
                    }
                }
                for id in finished_ids {
                    release(engine, sched.finish_request_early(id));
                }
                if mtp_batch_slots.iter().all(|s| s.is_none()) {
                    let _ = proposer.set_mtp_batch_mode(false);
                }
                Ok(())
            }
            Err(e) => {
                for (slot_idx, id, pending_vec, req) in pulled {
                    requests.insert(id, req);
                    mtp_batch_slots[slot_idx] = Some((id, pending_vec));
                }
                Err(WorkerError::Cuda(e))
            }
        }
    };
    engine.speculative = Some(proposer);
    res
}

fn abort_active_batch(
    engine: &mut Qwen3_5Engine,
    sched: &mut SchedulerV1,
    requests: &mut HashMap<u64, Request>,
    speculation: &mut SpeculativeLoopState,
    error: WorkerError,
) {
    let SpeculativeLoopState {
        speculative_id,
        pending_ingest,
        mtp_batch_slots,
    } = speculation;
    // GPU errors invalidate this batch; abort private state rather than publishing it.
    let ids: Vec<_> = sched.active.iter().map(|s| s.req_id).collect();
    for id in ids {
        if let Some(mut r) = requests.remove(&id) {
            if let Some(tap) = r.decision_tap.take() {
                tap.fail(error.clone());
            }
            r.set_terminal(GenerationEvent::Error(error.clone()));
        }
        release(engine, sched.cancel_request(id));
    }
    if let Some(p) = &mut engine.speculative {
        p.abort_scheduled();
        let _ = p.set_mtp_batch_mode(false);
    }
    *speculative_id = None;
    pending_ingest.clear();
    *mtp_batch_slots = std::array::from_fn(|_| None);
}

fn complete_tick(
    engine: &Qwen3_5Engine,
    sched: &mut SchedulerV1,
    requests: &mut HashMap<u64, Request>,
    stats: &WorkerStats,
) {
    for (bucket, _) in engine.captured_decode_graph_keys() {
        stats
            .decode_bucket_mask
            .fetch_or(1 << bucket.trailing_zeros(), Ordering::Relaxed);
    }
    stats
        .telemetry
        .publish_decode_bucket_mask(stats.decode_bucket_mask.load(Ordering::Relaxed));
    stats.active.store(
        sched.active.iter().filter(|s| !s.finished).count() as u64,
        Ordering::Relaxed,
    );
    for seq in sched.take_finished() {
        stats.completed.fetch_add(1, Ordering::Relaxed);
        stats
            .cached_tokens
            .fetch_add(seq.cached_prefix_tokens as u64, Ordering::Relaxed);
        if let Some(r) = requests.remove(&seq.req_id) {
            r.set_terminal(GenerationEvent::Finished {
                cached_tokens: seq.cached_prefix_tokens,
                generated_tokens: seq.generated_tokens.len(),
            });
        }
    }
}

fn emit(request: &Request, id: i32, logprobs: Option<StepLogprobs>) {
    // Engine output exists even if the bounded delivery channel rejects it.
    if let Some(life) = &request.lifecycle {
        life.token();
    }
    if request
        .output
        .try_send(GenerationEvent::Token { id, logprobs })
        .is_err()
    {
        request.cancelled.store(true, Ordering::Release);
    }
}

fn fail_request(
    engine: &mut Qwen3_5Engine,
    sched: &mut SchedulerV1,
    requests: &mut HashMap<u64, Request>,
    id: u64,
    error: WorkerError,
) {
    if let Some(mut r) = requests.remove(&id) {
        if let Some(tap) = r.decision_tap.take() {
            tap.fail_and_recycle(engine, error.clone());
        }
        r.set_terminal(GenerationEvent::Error(error));
    }
    release(engine, sched.cancel_request(id));
}

fn capture_decision_tap(
    engine: &mut Qwen3_5Engine,
    s: &crate::scheduler::ScheduledReqSlice,
    tok_off: usize,
    selected_row: Option<usize>,
    cached_toks: usize,
    req: &mut Request,
) {
    let slice_len = s.query_tokens.len();
    let mut tap_done = None;
    match &mut req.decision_tap {
        None => {}
        Some(DecisionReadoutTap::CandidateLogits { token_ids, .. }) => {
            if let Some(sel_row) = selected_row {
                let res = engine
                    .read_selected_row_candidate_logits(sel_row, token_ids)
                    .map(|logits| DecisionTapOutput::CandidateLogits {
                        logits,
                        cached_tokens: cached_toks,
                    })
                    .map_err(WorkerError::Cuda);
                if let Some(DecisionReadoutTap::CandidateLogits { tx, .. }) =
                    req.decision_tap.take()
                {
                    tap_done = Some((tx, res));
                }
            }
        }
        Some(DecisionReadoutTap::HiddenPositions {
            positions,
            captured,
            ..
        }) => {
            let chunk_start = s.ctx_len_before;
            let chunk_end = chunk_start + slice_len;
            let mut read_err = None;
            for (p_idx, &pos) in positions.iter().enumerate() {
                if (chunk_start..chunk_end).contains(&pos) {
                    match engine.read_norm_hidden_row_f32(tok_off + (pos - chunk_start)) {
                        Ok(row) => captured[p_idx] = Some(row),
                        Err(e) => {
                            read_err = Some(WorkerError::Cuda(e));
                            break;
                        }
                    }
                }
            }
            if (read_err.is_some() || s.is_final_prefill_chunk)
                && let Some(DecisionReadoutTap::HiddenPositions {
                    mut captured, tx, ..
                }) = req.decision_tap.take()
            {
                let res = if let Some(err) = read_err {
                    Err(err)
                } else if captured.iter().all(Option::is_some) {
                    Ok(DecisionTapOutput::HiddenRows {
                        rows: captured.drain(..).map(Option::unwrap).collect(),
                        cached_tokens: cached_toks,
                    })
                } else {
                    Err(WorkerError::internal(
                        "kev: missing hidden position after final prefill chunk",
                    ))
                };
                tap_done = Some((tx, res));
            }
        }
        Some(DecisionReadoutTap::ClefJoint {
            record, gpu_state, ..
        }) => {
            let expected_total = record.input_ids.len();
            let read_err = if gpu_state.is_none() {
                match check_full_hidden_chunk_offset(0, s.ctx_len_before)
                    .and_then(|()| engine.clef_begin_request(record))
                {
                    Ok(st) => {
                        *gpu_state = Some(st);
                        None
                    }
                    Err(e) => Some(WorkerError::internal(e)),
                }
            } else {
                None
            }
            .or_else(|| {
                let st = gpu_state.as_mut().unwrap();
                check_full_hidden_chunk_offset(st.next_offset(), s.ctx_len_before)
                    .and_then(|()| {
                        engine.clef_ingest_norm_hidden_range(
                            st,
                            tok_off,
                            s.ctx_len_before,
                            slice_len,
                        )
                    })
                    .err()
                    .map(WorkerError::internal)
            });
            if (read_err.is_some() || s.is_final_prefill_chunk)
                && let Some(DecisionReadoutTap::ClefJoint {
                    record,
                    gpu_state,
                    tx,
                }) = req.decision_tap.take()
            {
                let res = if let Some(err) = read_err {
                    if let Some(st) = gpu_state {
                        engine.clef_recycle_request(st);
                    }
                    Err(err)
                } else if let Some(st) = gpu_state {
                    check_full_hidden_final_count(st.next_offset(), expected_total)
                        .and_then(|()| engine.clef_finalize_request(st, &record))
                        .map(|logits| DecisionTapOutput::ClefLogits {
                            logits,
                            cached_tokens: cached_toks,
                        })
                        .map_err(WorkerError::internal)
                } else {
                    Err(WorkerError::internal("clef gpu request state missing"))
                };
                tap_done = Some((tx, res));
            }
        }
    }
    if let Some((tx, res)) = tap_done {
        if let Err(error) = &res {
            req.set_terminal(GenerationEvent::Error(error.clone()));
        }
        let _ = tx.send(res);
    }
}

fn record_computed_prefill(slices: &[ScheduledReqSlice], requests: &HashMap<u64, Request>) {
    for s in slices {
        if s.is_prefill_chunk {
            requests[&s.req_id].computed(s.query_tokens.len());
        }
    }
}

fn step_plain(
    engine: &mut Qwen3_5Engine,
    sched: &mut SchedulerV1,
    requests: &mut HashMap<u64, Request>,
    receiver: &Receiver<Request>,
    shutdown: &AtomicBool,
) -> Result<(), WorkerError> {
    let _batch_range = crate::profiling::Range::new(c"w34.serve.batch");
    // Vision prefill uses the same scheduler commit seam, but its 3-plane embedding
    // splice cannot share an ordinary text prefill batch. Decode can share batches.
    let vision_ids: Vec<_> = sched
        .active
        .iter()
        .filter(|seq| {
            !seq.finished && !seq.is_prefill_complete() && requests[&seq.req_id].vision.is_some()
        })
        .map(|seq| seq.req_id)
        .collect();
    for id in vision_ids {
        let seq = sched.active.iter().find(|seq| seq.req_id == id).unwrap();
        let cached_toks = seq.cached_prefix_tokens;
        let request = requests.get_mut(&id).unwrap();
        let (greedy, slice) = request.vision.as_mut().unwrap().step_prefill(
            engine,
            seq,
            sched.cfg.chunk_prefill_size,
        )?;
        request.computed(slice.query_tokens.len());
        //ScheduledVision::step_prefill executes forward_selected_rows on a 1-request slice
        // with selected_rows = [q_len - 1] (selected_row index 0 on every chunk).
        capture_decision_tap(
            engine,
            &slice,
            0,
            slice.is_final_prefill_chunk.then_some(0),
            cached_toks,
            request,
        );
        let token = if slice.is_final_prefill_chunk {
            if request.sampler.params.max_tokens > 0 {
                let sample = if request
                    .sampler
                    .params
                    .is_plain_greedy(request.sampler.constraint.is_some())
                {
                    request
                        .sampler
                        .try_commit_token(greedy)
                        .map(|()| (greedy, None))
                        .map_err(WorkerError::internal)
                } else {
                    engine
                        .sample_single_gpu_step(&mut request.sampler)
                        .map_err(WorkerError::Cuda)
                };
                let (token, lp) = match sample {
                    Ok(value) => value,
                    Err(error) => {
                        fail_request(engine, sched, requests, id, error);
                        continue;
                    }
                };
                emit(request, token, lp);
                Some(token)
            } else {
                Some(greedy)
            }
        } else {
            None
        };
        let mut replacement = engine.alloc_state_slot_from(&mut sched.radix);
        release(
            engine,
            sched.commit_slice(&slice, token, || replacement.take()),
        );
        if let Some(slot) = replacement {
            engine.free_state_slots(&[slot]);
        }
    }
    let slices = sched.schedule_step();
    if slices.is_empty() {
        return Ok(());
    }
    let checkpoints: Vec<Vec<_>> = slices
        .iter()
        .map(|s| {
            s.planned_checkpoints
                .iter()
                .map(|p| (p.trigger_offset_1based, p.dst_slot))
                .collect()
        })
        .collect();
    let batch: Vec<_> = slices
        .iter()
        .zip(&checkpoints)
        .map(|(s, cp)| {
            let seq = sched
                .active
                .iter()
                .find(|seq| seq.req_id == s.req_id)
                .unwrap();
            EngineBatchSlice {
                tokens: &s.query_tokens,
                ctx_len_before: s.ctx_len_before,
                block_table: &seq.radix_state.kv_pages,
                state_in_slot: s.state_in_slot,
                state_out_slot: s.state_out_slot,
                has_initial_state: s.has_initial_state,
                spec_slots: &[],
                checkpoints: cp,
                emit_all_query_tokens: false,
            }
        })
        .collect();
    let deltas: Vec<_> = slices
        .iter()
        .map(|s| {
            requests[&s.req_id]
                .vision
                .as_ref()
                .map_or(0, |v| v.position_delta)
        })
        .collect();
    let emit_mask: Vec<bool> = slices
        .iter()
        .map(|s| !s.is_prefill_chunk || s.is_final_prefill_chunk)
        .collect();
    let needs_gpu = slices.iter().enumerate().any(|(i, s)| {
        emit_mask[i] && {
            let r = &requests[&s.req_id];
            !r.sampler
                .params
                .is_plain_greedy(r.sampler.constraint.is_some())
        }
    });
    let all_pure_decode = slices.iter().all(|s| !s.is_prefill_chunk)
        && sched
            .active
            .iter()
            .all(|seq| seq.finished || seq.is_prefill_complete());
    let stateless_decode_ok = if needs_gpu {
        engine.cfg.vocab_size >= 1024
            && slices.iter().all(|s| {
                requests[&s.req_id]
                    .sampler
                    .is_fast_greedy_stateless(engine.cfg.eos_token_ids.len())
            })
    } else {
        slices
            .iter()
            .all(|s| requests[&s.req_id].sampler.params.stop_strings.is_empty())
    };
    if all_pure_decode && stateless_decode_ok && engine.can_pipeline_decode_slices(&batch) {
        return run_pipelined_decode_burst(
            engine, sched, requests, receiver, shutdown, slices, &deltas, needs_gpu,
        );
    }

    let greedy = engine.step_batch_slices_for_sampler_filtered(
        &batch,
        Some(&deltas),
        Some(&emit_mask),
        needs_gpu,
    )?;
    record_computed_prefill(&slices, requests);
    {
        let mut batch_tok_offset = 0usize;
        let mut selected_row_idx = 0usize;
        for (i, s) in slices.iter().enumerate() {
            let slice_len = s.query_tokens.len();
            let is_selected = emit_mask[i];
            let sel_row = selected_row_idx;
            if is_selected {
                selected_row_idx += 1;
            }
            let tok_off = batch_tok_offset;
            batch_tok_offset += slice_len;

            let cached_toks = sched
                .active
                .iter()
                .find(|seq| seq.req_id == s.req_id)
                .map_or(0, |seq| seq.cached_prefix_tokens);
            let Some(req) = requests.get_mut(&s.req_id) else {
                continue;
            };
            capture_decision_tap(
                engine,
                s,
                tok_off,
                is_selected.then_some(sel_row),
                cached_toks,
                req,
            );
        }
    }
    let mut pending_errors: Vec<Option<WorkerError>> = vec![None; slices.len()];
    let sampled = if needs_gpu {
        let _sampler_range = crate::profiling::Range::new(c"w34.serve.sampler");
        let mut masks = Vec::new();
        let mut sparse = Vec::new();
        let mut rows = Vec::new();
        for (i, s) in slices.iter().enumerate() {
            if !emit_mask[i] {
                continue;
            }
            let r = requests.get_mut(&s.req_id).unwrap();
            if let Err(err) = r.sampler.materialize_pending_constraint() {
                pending_errors[i] = Some(WorkerError::invalid_request(err));
                rows.push(w34_kernels::ffi::W34SamplerRowCfg {
                    temperature: 0.0,
                    top_k: 1,
                    top_p: 1.0,
                    min_p: 0.0,
                    repetition_penalty: 1.0,
                    frequency_penalty: 0.0,
                    presence_penalty: 0.0,
                    uniform_sample: 0.0,
                    recovery_uniform: 0.0,
                    draft_token_id: -1,
                    has_bitmask: 0,
                    row_bitmask_word_offset: 0,
                    sparse_offset: 0,
                    sparse_count: 0,
                    top_logprobs_k: -1,
                    use_token_id_cdf: 0,
                });
                continue;
            }
            let sampler = &r.sampler;
            let u1 = if sampler.params.temperature <= 1e-6 || sampler.params.top_k == 1 {
                0.0
            } else {
                SamplerV1::uniform_f32(sampler.params.seed, sampler.output_history.len())
            };
            rows.push(
                sampler
                    .append_gpu_row(
                        engine.cfg.vocab_size,
                        &engine.cfg.eos_token_ids,
                        u1,
                        0.0,
                        -1,
                        false,
                        &mut masks,
                        &mut sparse,
                    )
                    .map_err(WorkerError::internal)?,
            );
        }
        Some(engine.run_gpu_sampler_rows(&rows, &masks, &sparse)?)
    } else {
        None
    };
    let mut sample_idx = 0usize;
    for (i, s) in slices.iter().enumerate() {
        let token = if emit_mask[i] {
            let row_idx = sample_idx;
            sample_idx += 1;
            if let Some(error) = pending_errors[i].take() {
                fail_request(engine, sched, requests, s.req_id, error);
                continue;
            }
            let r = requests.get_mut(&s.req_id).unwrap();
            let (id, lp) = if let Some(rows) = &sampled {
                let out = &rows[row_idx];
                if out.status != 0 {
                    fail_request(
                        engine,
                        sched,
                        requests,
                        s.req_id,
                        WorkerError::sampler_status(out.status),
                    );
                    continue;
                }
                (
                    out.sampled_id,
                    r.sampler.params.logprobs.then(|| {
                        Qwen3_5Engine::extract_step_logprobs(
                            out.sampled_id,
                            out,
                            r.sampler.params.top_logprobs,
                        )
                    }),
                )
            } else {
                (greedy[i][0], None)
            };
            if r.sampler.params.max_tokens > 0 {
                if let Err(error) = r.sampler.try_commit_token_deferred(id) {
                    fail_request(
                        engine,
                        sched,
                        requests,
                        s.req_id,
                        WorkerError::internal(error),
                    );
                    continue;
                }
                emit(r, id, lp);
            }
            Some(id)
        } else {
            None
        };
        let mut replacement = if s.is_prefill_chunk {
            engine.alloc_state_slot_from(&mut sched.radix)
        } else {
            None
        };
        release(engine, sched.commit_slice(s, token, || replacement.take()));
        if let Some(slot) = replacement {
            engine.free_state_slots(&[slot]);
        }
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn run_pipelined_decode_burst(
    engine: &mut Qwen3_5Engine,
    sched: &mut SchedulerV1,
    requests: &mut HashMap<u64, Request>,
    receiver: &Receiver<Request>,
    shutdown: &AtomicBool,
    initial_slices: Vec<ScheduledReqSlice>,
    deltas: &[i32],
    needs_gpu: bool,
) -> Result<(), WorkerError> {
    let fg_storage = if needs_gpu {
        let mut masks = Vec::new();
        let mut sparse = Vec::new();
        let mut rows = Vec::new();
        for s in &initial_slices {
            let sampler = &requests[&s.req_id].sampler;
            rows.push(
                sampler
                    .append_gpu_row(
                        engine.cfg.vocab_size,
                        &engine.cfg.eos_token_ids,
                        0.0,
                        0.0,
                        -1,
                        false,
                        &mut masks,
                        &mut sparse,
                    )
                    .map_err(WorkerError::internal)?,
            );
        }
        debug_assert!(masks.is_empty());
        Some((rows, sparse))
    } else {
        None
    };
    let fg_ref = fg_storage
        .as_ref()
        .map(|(rows, sparse)| (rows.as_slice(), sparse.as_slice()));

    let mut cur_slot = 0usize;
    let mut slices_cur = initial_slices;
    let mut ticket_cur = {
        let checkpoints: Vec<Vec<_>> = slices_cur
            .iter()
            .map(|s| {
                s.planned_checkpoints
                    .iter()
                    .map(|p| (p.trigger_offset_1based, p.dst_slot))
                    .collect()
            })
            .collect();
        let batch_cur: Vec<_> = slices_cur
            .iter()
            .zip(&checkpoints)
            .map(|(s, cp)| {
                let seq = sched
                    .active
                    .iter()
                    .find(|seq| seq.req_id == s.req_id)
                    .unwrap();
                EngineBatchSlice {
                    tokens: &s.query_tokens,
                    ctx_len_before: s.ctx_len_before,
                    block_table: &seq.radix_state.kv_pages,
                    state_in_slot: s.state_in_slot,
                    state_out_slot: s.state_out_slot,
                    has_initial_state: s.has_initial_state,
                    spec_slots: &[],
                    checkpoints: cp,
                    emit_all_query_tokens: false,
                }
            })
            .collect();
        engine
            .enqueue_pipelined_decode_step(cur_slot, &batch_cur, Some(deltas), false, true, fg_ref)
            .map_err(|e| {
                engine.abort_pipelined_decode();
                WorkerError::Cuda(e)
            })?
    };

    loop {
        let max_seq_cur = slices_cur
            .iter()
            .map(|s| s.ctx_len_before + 1)
            .max()
            .unwrap_or(1);
        let max_seq_next = max_seq_cur + 1;
        let max_pages = engine.ranks[0].kv_pool.num_pages;
        let max_blocks = engine.ranks[0].batched_decode_io.max_blocks_per_req;
        let can_overlap_next = !shutdown.load(Ordering::Acquire)
            && receiver.is_empty()
            && sched.waiting.is_empty()
            && ((max_seq_next > 1024) == (max_seq_cur > 1024))
            && slices_cur.iter().all(|s| {
                let r = &requests[&s.req_id];
                let seq = sched
                    .active
                    .iter()
                    .find(|seq| seq.req_id == s.req_id)
                    .unwrap();
                let next_after = s.ctx_len_before + 2;
                !r.cancelled.load(Ordering::Acquire)
                    && r.sampler.params.stop_strings.is_empty()
                    && !seq.finished
                    && seq.generated_tokens.len() + 1 < seq.max_new_tokens
                    && !next_after.is_multiple_of(sched.cfg.mamba_chunk_size)
                    && seq.radix_state.kv_pages.len() <= max_blocks
                    && next_after.div_ceil(16) <= seq.radix_state.kv_pages.len()
                    && seq
                        .radix_state
                        .kv_pages
                        .iter()
                        .all(|&p| p >= 0 && (p as usize) < max_pages)
            });

        if !can_overlap_next {
            let out_cur = engine.wait_pipelined_decode_step(ticket_cur).map_err(|e| {
                engine.abort_pipelined_decode();
                WorkerError::Cuda(e)
            })?;
            commit_pipelined_step(engine, sched, requests, &slices_cur, &out_cur, None);
            return Ok(());
        }

        let next_slot = cur_slot ^ 1;
        let mut slices_next: Vec<ScheduledReqSlice> = slices_cur
            .iter()
            .map(|s| {
                let seq = sched
                    .active
                    .iter()
                    .find(|seq| seq.req_id == s.req_id)
                    .unwrap();
                ScheduledReqSlice {
                    req_id: s.req_id,
                    is_prefill_chunk: false,
                    is_final_prefill_chunk: false,
                    ctx_len_before: s.ctx_len_before + 1,
                    query_tokens: vec![0],
                    state_in_slot: seq.radix_state.active_state_slot,
                    state_out_slot: seq.radix_state.active_state_slot,
                    has_initial_state: true,
                    planned_checkpoints: Vec::new(),
                }
            })
            .collect();

        let ticket_next = {
            let batch_next: Vec<_> = slices_next
                .iter()
                .map(|s| {
                    let seq = sched
                        .active
                        .iter()
                        .find(|seq| seq.req_id == s.req_id)
                        .unwrap();
                    EngineBatchSlice {
                        tokens: &s.query_tokens,
                        ctx_len_before: s.ctx_len_before,
                        block_table: &seq.radix_state.kv_pages,
                        state_in_slot: s.state_in_slot,
                        state_out_slot: s.state_out_slot,
                        has_initial_state: s.has_initial_state,
                        spec_slots: &[],
                        checkpoints: &[],
                        emit_all_query_tokens: false,
                    }
                })
                .collect();
            engine
                .enqueue_pipelined_decode_step(
                    next_slot,
                    &batch_next,
                    Some(deltas),
                    true,
                    true,
                    fg_ref,
                )
                .map_err(|e| {
                    engine.abort_pipelined_decode();
                    WorkerError::Cuda(e)
                })?
        };

        let out_cur = engine.wait_pipelined_decode_step(ticket_cur).map_err(|e| {
            engine.abort_pipelined_decode();
            WorkerError::Cuda(e)
        })?;

        let mut deferred_releases = Vec::new();
        let mut deferred_failures = Vec::new();
        let mut emitted_ids = vec![None; slices_cur.len()];
        let mut any_stopped = false;

        for (i, s) in slices_cur.iter().enumerate() {
            let r = requests.get_mut(&s.req_id).unwrap();
            let parsed = match &out_cur {
                PipelinedDecodeOutput::FastGreedy(rows) => {
                    let out = &rows[i];
                    if out.status != 0 {
                        Err(WorkerError::sampler_status(out.status))
                    } else {
                        Ok((
                            out.sampled_id,
                            r.sampler.params.logprobs.then(|| {
                                Qwen3_5Engine::extract_step_logprobs(
                                    out.sampled_id,
                                    out,
                                    r.sampler.params.top_logprobs,
                                )
                            }),
                        ))
                    }
                }
                PipelinedDecodeOutput::PlainGreedy(ids) => Ok((ids[i], None)),
            };
            let (id, lp) = match parsed {
                Ok(v) => v,
                Err(err) => {
                    deferred_failures.push((s.req_id, err));
                    any_stopped = true;
                    continue;
                }
            };
            if let Err(err) = r.sampler.try_commit_token(id) {
                deferred_failures.push((s.req_id, WorkerError::internal(err)));
                any_stopped = true;
                continue;
            }
            emit(r, id, lp);
            emitted_ids[i] = Some(id);
            let outcome = sched.commit_slice(s, Some(id), || None);
            if outcome.is_some() || r.cancelled.load(Ordering::Acquire) {
                any_stopped = true;
            }
            deferred_releases.push(outcome);
        }

        if any_stopped {
            let out_next = engine
                .wait_pipelined_decode_step(ticket_next)
                .map_err(|e| {
                    engine.abort_pipelined_decode();
                    WorkerError::Cuda(e)
                })?;
            for outcome in deferred_releases {
                release(engine, outcome);
            }
            for (req_id, err) in deferred_failures {
                fail_request(engine, sched, requests, req_id, err);
            }
            let mut active_mask = vec![false; slices_next.len()];
            for (i, sn) in slices_next.iter_mut().enumerate() {
                if let Some(tok_prev) = emitted_ids[i] {
                    sn.query_tokens[0] = tok_prev;
                    let still_running = sched
                        .active
                        .iter()
                        .any(|seq| seq.req_id == sn.req_id && !seq.finished)
                        && requests
                            .get(&sn.req_id)
                            .is_some_and(|r| !r.cancelled.load(Ordering::Acquire));
                    active_mask[i] = still_running;
                }
            }
            commit_pipelined_step(
                engine,
                sched,
                requests,
                &slices_next,
                &out_next,
                Some(&active_mask),
            );
            return Ok(());
        }

        for outcome in deferred_releases {
            release(engine, outcome);
        }
        for (i, sn) in slices_next.iter_mut().enumerate() {
            sn.query_tokens[0] = emitted_ids[i].unwrap();
        }
        cur_slot = next_slot;
        ticket_cur = ticket_next;
        slices_cur = slices_next;
    }
}

fn commit_pipelined_step(
    engine: &mut Qwen3_5Engine,
    sched: &mut SchedulerV1,
    requests: &mut HashMap<u64, Request>,
    slices: &[ScheduledReqSlice],
    out: &PipelinedDecodeOutput,
    active_mask: Option<&[bool]>,
) {
    for (i, s) in slices.iter().enumerate() {
        if active_mask.is_some_and(|m| !m[i]) {
            continue;
        }
        let Some(r) = requests.get_mut(&s.req_id) else {
            continue;
        };
        let parsed = match out {
            PipelinedDecodeOutput::FastGreedy(rows) => {
                let row = &rows[i];
                if row.status != 0 {
                    Err(WorkerError::sampler_status(row.status))
                } else {
                    Ok((
                        row.sampled_id,
                        r.sampler.params.logprobs.then(|| {
                            Qwen3_5Engine::extract_step_logprobs(
                                row.sampled_id,
                                row,
                                r.sampler.params.top_logprobs,
                            )
                        }),
                    ))
                }
            }
            PipelinedDecodeOutput::PlainGreedy(ids) => Ok((ids[i], None)),
        };
        let (id, lp) = match parsed {
            Ok(v) => v,
            Err(err) => {
                fail_request(engine, sched, requests, s.req_id, err);
                continue;
            }
        };
        if let Err(err) = r.sampler.try_commit_token(id) {
            fail_request(
                engine,
                sched,
                requests,
                s.req_id,
                WorkerError::internal(err),
            );
            continue;
        }
        emit(r, id, lp);
        release(engine, sched.commit_slice(s, Some(id), || None));
    }
}

fn check_full_hidden_chunk_offset(
    captured_len: usize,
    ctx_len_before: usize,
) -> std::result::Result<(), String> {
    if ctx_len_before != captured_len {
        return Err(format!(
            "clef: non-contiguous prefill chunk (expected offset {captured_len}, got {ctx_len_before})"
        ));
    }
    Ok(())
}

fn check_full_hidden_final_count(
    captured_len: usize,
    expected_total: usize,
) -> std::result::Result<(), String> {
    if captured_len != expected_total {
        return Err(format!(
            "clef: captured {captured_len} hidden rows after final prefill chunk, expected {expected_total}"
        ));
    }
    Ok(())
}

#[cfg(test)]
mod full_hidden_tap_tests {
    use super::*;

    #[test]
    fn full_hidden_tap_accounting_detects_gaps_and_short_counts() {
        assert!(check_full_hidden_chunk_offset(0, 0).is_ok());
        assert!(check_full_hidden_chunk_offset(256, 256).is_ok());
        let gap_err = check_full_hidden_chunk_offset(128, 192).unwrap_err();
        assert!(
            gap_err.contains("expected offset 128, got 192"),
            "{gap_err}"
        );

        assert!(check_full_hidden_final_count(404, 404).is_ok());
        let short_err = check_full_hidden_final_count(380, 404).unwrap_err();
        assert!(
            short_err.contains("captured 380 hidden rows after final prefill chunk, expected 404"),
            "{short_err}"
        );
    }

    fn terminal_test_request() -> (
        Request,
        GenerationStream,
        tokio::sync::oneshot::Receiver<Result<DecisionTapOutput, WorkerError>>,
    ) {
        let (output, events) = generation_channel(2);
        let (tx, rx) = tokio::sync::oneshot::channel();
        let cached_tokens = Arc::new(AtomicU64::new(0));
        let cancelled = Arc::new(AtomicBool::new(false));
        let request = Request {
            lifecycle: Some(Arc::new(Telemetry::default()).request(1)),
            prompt: vec![1],
            images: Vec::new(),
            vision: None,
            reserved_pages: 0,
            cached_tokens: cached_tokens.clone(),
            sampler: SamplerV1::new(SamplingParams::default(), &[1], None),
            output,
            terminal: events.terminal.clone(),
            cancelled: cancelled.clone(),
            max_prefix_lookup_len: None,
            mamba_branching_seqlen: None,
            decision_tap: Some(DecisionReadoutTap::CandidateLogits {
                token_ids: vec![1],
                tx,
            }),
        };
        (
            request,
            GenerationStream {
                events,
                cached_tokens,
                cancelled,
            },
            rx,
        )
    }

    #[test]
    fn telemetry_counts_engine_output_even_when_delivery_is_full() {
        let (mut request, _stream, _tap) = terminal_test_request();
        let telemetry = Arc::new(Telemetry::default());
        request.lifecycle = Some(telemetry.request(1));
        request.lifecycle.as_ref().unwrap().scheduled(0);
        for id in 0..3 {
            emit(&request, id, None);
        }
        assert_eq!(telemetry.snapshot().tokens.generated, 3);
        assert!(request.cancelled.load(Ordering::Acquire));
        drop(request);
        let s = telemetry.snapshot();
        assert_eq!(s.requests.cancelled, 1);
        assert_eq!(s.requests.failed, 0);
        assert_eq!(s.requests.running, 0);
        assert_eq!(s.latency.ttft.count, 1);
    }

    #[tokio::test]
    async fn terminal_finished_survives_full_queue_and_is_delivered_once() {
        let (request, mut stream, _tap) = terminal_test_request();
        for id in 0..2 {
            emit(&request, id, None);
        }
        request.set_terminal(GenerationEvent::Finished {
            cached_tokens: 3,
            generated_tokens: 2,
        });
        // A later cleanup must not overwrite the already committed terminal result.
        notify_request_error(request, WorkerError::worker_stopped("late stop"));
        for id in 0..2 {
            assert!(matches!(stream.events.recv().await,
                Some(GenerationEvent::Token { id: actual, .. }) if actual == id));
        }
        assert!(matches!(
            stream.events.recv().await,
            Some(GenerationEvent::Finished {
                cached_tokens: 3,
                generated_tokens: 2,
            })
        ));
        assert!(stream.events.recv().await.is_none());
        assert!(stream.events.recv().await.is_none());
    }

    #[tokio::test]
    async fn terminal_delivery_wakes_a_waiting_receiver() {
        let (request, mut stream, _tap) = terminal_test_request();
        let receive = stream.events.recv();
        tokio::pin!(receive);
        // Poll recv while the channel is empty but still open, then terminate it.
        assert!(
            std::future::poll_fn(|cx| {
                assert!(receive.as_mut().poll(cx).is_pending());
                std::task::Poll::Ready(true)
            })
            .await
        );
        notify_request_error(request, WorkerError::worker_stopped("stop while waiting"));
        assert!(matches!(
            tokio::time::timeout(std::time::Duration::from_secs(1), receive).await,
            Ok(Some(GenerationEvent::Error(WorkerError::WorkerStopped(_))))
        ));
    }

    #[test]
    fn supervisor_publishes_poison_before_borrowed_state_teardown() {
        struct TeardownProbe<'a> {
            panicked: &'a AtomicBool,
            reason: &'a Mutex<Option<String>>,
            touched: bool,
        }
        impl Drop for TeardownProbe<'_> {
            fn drop(&mut self) {
                assert!(self.touched);
                assert!(!std::thread::panicking());
                assert!(self.panicked.load(Ordering::Acquire));
                assert_eq!(
                    self.reason.lock().unwrap().as_deref(),
                    Some("borrowed state panic")
                );
            }
        }
        let panicked = AtomicBool::new(false);
        let reason = Mutex::new(None);
        let mut engine = TeardownProbe {
            panicked: &panicked,
            reason: &reason,
            touched: false,
        };
        let (_sender, mut receiver) = mpsc::channel(2);
        let mut requests = HashMap::new();
        // Mirror run(): borrow state across the caught closure, never move it in.
        run_supervised_loop(
            &mut receiver,
            &mut requests,
            &WorkerStats::default(),
            &panicked,
            &reason,
            |_, _| {
                let borrowed = &mut engine;
                borrowed.touched = true;
                panic!("borrowed state panic");
            },
        );
        drop(engine);
    }

    #[tokio::test]
    async fn terminal_panic_survives_full_token_queue() {
        let (request, mut stream, tap) = terminal_test_request();
        for id in 0..2 {
            request
                .output
                .try_send(GenerationEvent::Token { id, logprobs: None })
                .unwrap();
        }
        let (_sender, mut receiver) = mpsc::channel(2);
        let mut requests = HashMap::from([(1, request)]);
        run_supervised_loop(
            &mut receiver,
            &mut requests,
            &WorkerStats::default(),
            &AtomicBool::new(false),
            &Mutex::new(None),
            |_, _| panic!("full queue panic"),
        );
        for id in 0..2 {
            assert!(
                matches!(stream.events.recv().await, Some(GenerationEvent::Token { id: actual, .. }) if actual == id)
            );
        }
        assert!(matches!(stream.events.recv().await,
            Some(GenerationEvent::Error(WorkerError::WorkerPanicked(msg))) if msg == "full queue panic"));
        assert!(
            matches!(tap.await, Ok(Err(WorkerError::WorkerPanicked(msg))) if msg == "full queue panic")
        );
        assert!(stream.events.recv().await.is_none());
    }

    #[tokio::test]
    async fn terminal_normal_stop_drains_active_and_queued_decisions() {
        let (active, mut active_stream, active_tap) = terminal_test_request();
        for id in 0..2 {
            emit(&active, id, None);
        }
        let (queued, mut queued_stream, queued_tap) = terminal_test_request();
        let (sender, mut receiver) = mpsc::channel(2);
        assert!(sender.try_send(queued).is_ok());
        let mut requests = HashMap::from([(1, active)]);
        let stats = WorkerStats::default();
        stats.active.store(1, Ordering::Relaxed);
        let panicked = AtomicBool::new(false);
        let reason = Mutex::new(None);
        run_supervised_loop(
            &mut receiver,
            &mut requests,
            &stats,
            &panicked,
            &reason,
            |_, _| {},
        );
        assert!(sender.is_closed(), "normal stop must close ingress");
        assert!(requests.is_empty());
        assert_eq!(stats.active.load(Ordering::Relaxed), 0);
        assert!(!panicked.load(Ordering::Acquire));
        assert!(reason.lock().unwrap().is_none());
        for id in 0..2 {
            assert!(matches!(active_stream.events.recv().await,
                Some(GenerationEvent::Token { id: actual, .. }) if actual == id));
        }
        for stream in [&mut active_stream, &mut queued_stream] {
            match stream.events.recv().await {
                Some(GenerationEvent::Error(err @ WorkerError::WorkerStopped(_))) => assert_eq!(
                    err.http_status(),
                    axum::http::StatusCode::SERVICE_UNAVAILABLE
                ),
                other => panic!("expected WorkerStopped, got {other:?}"),
            }
            assert!(stream.events.recv().await.is_none());
        }
        for tap in [active_tap, queued_tap] {
            assert!(matches!(tap.await, Ok(Err(WorkerError::WorkerStopped(_)))));
        }
    }

    async fn check_locally_owned_panic(pulled_batch: bool) {
        let (mut request, mut stream, tap) = terminal_test_request();
        let telemetry = Arc::new(Telemetry::default());
        request.lifecycle = Some(telemetry.request(1));
        let (sender, mut receiver) = mpsc::channel(2);
        let mut requests = HashMap::new();
        if pulled_batch {
            requests.insert(1, request);
        } else {
            assert!(sender.try_send(request).is_ok());
        }
        run_supervised_loop(
            &mut receiver,
            &mut requests,
            &WorkerStats::default(),
            &AtomicBool::new(false),
            &Mutex::new(None),
            |rx, reqs| {
                if pulled_batch {
                    // Exercise the heap-owned batch shape used by MTP verification.
                    #[allow(clippy::useless_vec)]
                    let _pulled = vec![(0, 1, Vec::<i32>::new(), reqs.remove(&1).unwrap())];
                    panic!("panic while MTP owns pulled requests");
                } else {
                    let _admitting = rx.try_recv().unwrap();
                    panic!("panic during admission before registry insert");
                }
            },
        );
        assert!(matches!(stream.events.recv().await,
            Some(GenerationEvent::Error(WorkerError::WorkerPanicked(msg))) if !msg.is_empty()));
        assert!(matches!(tap.await, Ok(Err(WorkerError::WorkerPanicked(msg))) if !msg.is_empty()));
        assert!(stream.events.recv().await.is_none());
        let s = telemetry.snapshot();
        assert_eq!(s.requests.failed, 1);
        assert_eq!(s.requests.waiting + s.requests.running, 0);
        assert_eq!(s.latency.e2e.count, 1);
    }

    #[tokio::test]
    async fn terminal_panic_during_admission() {
        check_locally_owned_panic(false).await;
    }

    #[tokio::test]
    async fn terminal_panic_while_mtp_requests_are_pulled() {
        check_locally_owned_panic(true).await;
    }

    #[test]
    fn worker_panic_containment_drains_in_flight_and_queued_and_poisons_future_submits() {
        let (sender, mut receiver) = mpsc::channel(32);
        let shutdown = Arc::new(AtomicBool::new(false));
        let panicked = Arc::new(AtomicBool::new(false));
        let panic_reason = Arc::new(Mutex::new(None));
        let stats = Arc::new(WorkerStats::default());
        stats.active.store(2, Ordering::Relaxed);

        let worker_panicked = panicked.clone();
        let worker_panic_reason = panic_reason.clone();
        let worker_stats = stats.clone();
        let (ingested_tx, ingested_rx) = std::sync::mpsc::sync_channel(1);
        let (unblock_tx, unblock_rx) = std::sync::mpsc::sync_channel(1);

        let thread = std::thread::Builder::new()
            .name("w34-cuda-test".into())
            .spawn(move || {
                let mut requests = HashMap::new();
                run_supervised_loop(
                    &mut receiver,
                    &mut requests,
                    &worker_stats,
                    &worker_panicked,
                    &worker_panic_reason,
                    |rx, reqs| {
                        let first = rx.blocking_recv().expect("first request");
                        reqs.insert(1, first);
                        ingested_tx.send(()).unwrap();
                        unblock_rx.recv().unwrap();
                        panic!("simulated scheduler invariant breach");
                    },
                );
            })
            .unwrap();

        let worker = EngineWorker {
            sender: Mutex::new(Some(sender)),
            thread: Mutex::new(Some(thread)),
            shutdown,
            panicked,
            panic_reason,
            stats,
            tokenizer: None,
            eos_token_ids: vec![2],
            vocab_size: 1024,
            default_preserve_thinking: false,
            decision_profile: crate::decision::DecisionModelProfile::Tev1 { temperature: 1.0 },
            vision_config: None,
            has_vision_weights: false,
            kv_token_capacity: 4096,
        };

        let mut active_stream = worker
            .submit(vec![10, 20], SamplingParams::default(), None)
            .expect("submit active request");
        ingested_rx.recv().unwrap();

        // Enqueue a second request with a decision tap sitting in `receiver` (not yet dequeued).
        let (mut queued_stream, queued_tap_rx) = worker
            .submit_decision_candidate_logits(vec![30, 40], vec![1, 2], None)
            .expect("submit queued decision request");

        unblock_tx.send(()).unwrap();

        let rt = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .unwrap();
        rt.block_on(async {
            match active_stream.events.recv().await {
                Some(GenerationEvent::Error(WorkerError::WorkerPanicked(msg))) => {
                    assert_eq!(msg, "simulated scheduler invariant breach");
                }
                other => panic!("expected WorkerPanicked on active stream, got {other:?}"),
            }

            match queued_stream.events.recv().await {
                Some(GenerationEvent::Error(WorkerError::WorkerPanicked(msg))) => {
                    assert_eq!(msg, "simulated scheduler invariant breach");
                }
                other => panic!("expected WorkerPanicked on queued stream, got {other:?}"),
            }

            match queued_tap_rx.await {
                Ok(Err(WorkerError::WorkerPanicked(msg))) => {
                    assert_eq!(msg, "simulated scheduler invariant breach");
                }
                other => panic!("expected WorkerPanicked on decision tap, got {other:?}"),
            }

            assert!(worker.is_panicked());
            assert_eq!(worker.stats.active.load(Ordering::Relaxed), 0);
            let metrics = worker.stats.telemetry.snapshot();
            assert_eq!(metrics.requests.submitted, 2);
            assert_eq!(metrics.requests.failed, 2);
            assert_eq!(metrics.requests.waiting + metrics.requests.running, 0);
            assert_eq!(metrics.latency.e2e.count, 2);
            assert_eq!(metrics.latency.ttft.count, 0);
            assert_eq!(metrics.status, "unhealthy");

            // Future submits must report WorkerPanicked (HTTP 500), never Overload/503.
            let post_panic = worker.submit(vec![1, 2], SamplingParams::default(), None);
            match post_panic {
                Err(err @ WorkerError::WorkerPanicked(_)) => {
                    assert_eq!(
                        err.http_status(),
                        axum::http::StatusCode::INTERNAL_SERVER_ERROR
                    );
                }
                Ok(_) => panic!("expected submit after panic to fail"),
                Err(other) => panic!("expected WorkerPanicked on post-panic submit, got {other:?}"),
            }

            let shutdown_err = worker
                .shutdown()
                .await
                .expect_err("shutdown must surface worker panic");
            assert!(
                shutdown_err.contains("simulated scheduler invariant breach"),
                "{shutdown_err}"
            );
        });
    }
}
