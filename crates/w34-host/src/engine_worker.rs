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

mod decode;
mod runner;
#[cfg(test)]
mod tests;

use self::runner::run;

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

fn validate_request_length(prompt: usize, output: usize, limit: usize) -> Result<(), WorkerError> {
    if prompt.checked_add(output).is_none_or(|total| total > limit) {
        return Err(WorkerError::invalid_request(format!(
            "prompt ({prompt}) + output budget ({output}) exceeds serving context limit ({limit} tokens)"
        )));
    }
    Ok(())
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
    pub hidden_size: usize,
    pub kv_token_capacity: usize,
    max_model_len: Option<usize>,
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
                    if engine.cfg.uses_tiered_moe()
                        && clamped_cfg.chunk_prefill_size == 2048
                        && clamped_cfg.max_num_batched_tokens == 2048
                    {
                        clamped_cfg.max_num_batched_tokens = 6144;
                        clamped_cfg.chunk_prefill_size = 6144;
                    }
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
                        engine.cfg.hidden_size,
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
            hidden_size,
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
            hidden_size,
            kv_token_capacity,
            max_model_len: None,
        })
    }

    /// Set a deployment admission ceiling before sharing the worker with callers.
    /// This does not resize KV pools or extend a model's supported context window.
    pub fn set_max_model_len(&mut self, limit: Option<usize>) -> Result<(), String> {
        if limit.is_some_and(|n| n == 0 || n > self.kv_token_capacity) {
            return Err(format!(
                "--max-model-len must be positive and no greater than the allocated KV capacity ({} tokens); increase --kv-pages if needed",
                self.kv_token_capacity
            ));
        }
        self.max_model_len = limit;
        Ok(())
    }

    pub fn max_model_len(&self) -> usize {
        self.max_model_len.unwrap_or(self.kv_token_capacity)
    }

    /// Shutdown is terminal for this worker; callers must stop admitting requests.
    pub fn is_stopping(&self) -> bool {
        self.shutdown.load(Ordering::Acquire)
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
        validate_request_length(
            req.prompt.len(),
            req.sampler.params.max_tokens,
            self.max_model_len(),
        )?;
        if !req.images.is_empty() && req.vision.is_none() {
            let images = std::mem::take(&mut req.images);
            match crate::scheduled_vision::ScheduledVision::from_vision_config(
                self.vision_config.as_ref(),
                self.hidden_size,
                &req.prompt,
                images,
            ) {
                Ok(vision) => req.vision = Some(vision),
                Err(error) => {
                    notify_request_error(req, WorkerError::invalid_request(error));
                    return Ok(());
                }
            }
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
            stop_token_ids: self.eos_token_ids.clone(),
            eos_token_ids: self.eos_token_ids.clone(),
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
    let prompt_tokens = std::mem::take(&mut request.prompt);
    let key = request
        .vision
        .as_ref()
        .map_or_else(|| prompt_tokens.clone(), |v| v.radix_key.clone());
    let id = sched.submit_request_with_prefix_options(
        prompt_tokens,
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
