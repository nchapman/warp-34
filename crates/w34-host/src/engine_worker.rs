//! Bounded engine ingress and incremental scheduled generation. Network writers never run
//! on the GPU worker; a full output channel cancels that request rather than blocking peers.
use std::{
    collections::HashMap,
    sync::{
        Arc, Mutex,
        atomic::{AtomicBool, AtomicU64, Ordering},
    },
    thread::JoinHandle,
};
use tokio::sync::mpsc::{self, Receiver, Sender};

use crate::{
    model::{EngineBatchSlice, Qwen3_5Engine},
    parser::{SamplerV1, SamplingParams, StepLogprobs, TokenBitmaskConstraint},
    scheduler::{SchedulerConfig, SchedulerV1},
};

#[derive(Debug)]
pub enum GenerationEvent {
    Token {
        id: i32,
        logprobs: Option<StepLogprobs>,
    },
    Finished {
        cached_tokens: usize,
        generated_tokens: usize,
    },
    Error(String),
}

pub struct GenerationStream {
    pub events: Receiver<GenerationEvent>,
    pub cached_tokens: Arc<AtomicU64>,
    cancelled: Arc<AtomicBool>,
}
impl Drop for GenerationStream {
    fn drop(&mut self) {
        self.cancelled.store(true, Ordering::Release);
    }
}

struct Request {
    prompt: Vec<i32>,
    images: Vec<crate::vision::PreprocessedImage>,
    vision: Option<crate::scheduled_vision::ScheduledVision>,
    reserved_pages: usize,
    cached_tokens: Arc<AtomicU64>,
    sampler: SamplerV1,
    output: Sender<GenerationEvent>,
    cancelled: Arc<AtomicBool>,
}

#[derive(Default)]
pub struct WorkerStats {
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
    pub stats: Arc<WorkerStats>,
    pub tokenizer: Option<tokenizers::Tokenizer>,
    pub eos_token_ids: Vec<i32>,
    pub vocab_size: usize,
}
impl EngineWorker {
    pub fn start(engine: Qwen3_5Engine, config: SchedulerConfig) -> Self {
        Self::from_loader(move || Ok(engine), config).expect("infallible engine transfer")
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
        let stats = Arc::new(WorkerStats::default());
        let worker_stats = stats.clone();
        let thread = std::thread::Builder::new()
            .name("w34-cuda".into())
            .spawn(move || match loader() {
                Ok(engine) => {
                    let info = (
                        engine.tokenizer.clone(),
                        engine.cfg.eos_token_ids.clone(),
                        engine.cfg.vocab_size,
                    );
                    if ready_tx.send(Ok(info)).is_ok() {
                        run(engine, config, receiver, stop, worker_stats);
                    }
                }
                Err(e) => {
                    let _ = ready_tx.send(Err(e));
                }
            })
            .map_err(|e| e.to_string())?;
        let (tokenizer, eos_token_ids, vocab_size) = match ready_rx.recv() {
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
            stats,
            tokenizer,
            eos_token_ids,
            vocab_size,
        })
    }

    pub fn request_shutdown(&self) {
        self.shutdown.store(true, Ordering::Release);
        self.sender.lock().unwrap().take(); // wake blocking_recv when idle
    }

    pub async fn shutdown(&self) -> Result<(), String> {
        self.request_shutdown();
        let thread = self.thread.lock().unwrap().take();
        if let Some(thread) = thread {
            // Only joining is offloaded; no CUDA state ever enters spawn_blocking.
            tokio::task::spawn_blocking(move || thread.join())
                .await
                .map_err(|e| e.to_string())?
                .map_err(|_| "CUDA worker panicked".to_string())?;
        }
        Ok(())
    }

    pub fn submit(
        &self,
        prompt: Vec<i32>,
        params: SamplingParams,
        constraint: Option<TokenBitmaskConstraint>,
    ) -> Result<GenerationStream, String> {
        self.submit_with_images(prompt, Vec::new(), params, constraint)
    }

    pub fn submit_with_images(
        &self,
        prompt: Vec<i32>,
        images: Vec<crate::vision::PreprocessedImage>,
        params: SamplingParams,
        constraint: Option<TokenBitmaskConstraint>,
    ) -> Result<GenerationStream, String> {
        params.validate()?;
        if prompt.is_empty() || params.max_tokens == 0 {
            return Err("prompt and output budget must be nonempty".into());
        }
        let (output, events) = mpsc::channel(64);
        let cached_tokens = Arc::new(AtomicU64::new(0));
        let cancelled = Arc::new(AtomicBool::new(false));
        let sampler = SamplerV1::new(params, &prompt, constraint);
        self.sender
            .lock()
            .unwrap()
            .as_ref()
            .ok_or("engine worker stopped")?
            .try_send(Request {
                prompt,
                images,
                vision: None,
                reserved_pages: 0,
                cached_tokens: cached_tokens.clone(),
                sampler,
                output,
                cancelled: cancelled.clone(),
            })
            .map_err(|_| "engine queue unavailable or full")?;
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
        let _ = request.output.try_send(GenerationEvent::Error(
            "engine admission capacity exceeded".into(),
        ));
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
        let _ = request.output.try_send(GenerationEvent::Error(
            "invalid tokens or request exceeds KV capacity".into(),
        ));
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
                let _ = request.output.try_send(GenerationEvent::Error(error));
                return;
            }
        }
    }
    let key = request
        .vision
        .as_ref()
        .map_or_else(|| request.prompt.clone(), |v| v.radix_key.clone());
    let id = sched.submit_request_configured(
        request.prompt.clone(),
        key,
        p.max_tokens,
        p.min_tokens,
        p.ignore_eos,
        p.stop_token_ids.clone(),
        p.eos_token_ids.clone(),
        request.vision.is_some(),
    );
    requests.insert(id, request);
}

fn run(
    mut engine: Qwen3_5Engine,
    config: SchedulerConfig,
    mut receiver: Receiver<Request>,
    shutdown: Arc<AtomicBool>,
    stats: Arc<WorkerStats>,
) {
    let mut sched = SchedulerV1::new(config);
    let mut requests = HashMap::new();
    let mut speculative_id = None;
    let mut pending_ingest = Vec::new();
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
        // The engine retains the canonical radix between ticks (including diagnostics).
        std::mem::swap(&mut engine.radix, &mut sched.radix);
        if let Some(r) = first {
            enqueue(&engine, &mut sched, &mut requests, r);
        }
        // Bound queue draining even if producers keep replenishing it.
        for _ in 0..32 {
            match receiver.try_recv() {
                Ok(r) => enqueue(&engine, &mut sched, &mut requests, r),
                Err(_) => break,
            }
        }
        let cancelled: Vec<_> = requests
            .iter()
            .filter_map(|(&id, r)| r.cancelled.load(Ordering::Acquire).then_some(id))
            .collect();
        for id in cancelled {
            stats.cancelled.fetch_add(1, Ordering::Relaxed);
            if speculative_id == Some(id) {
                if let Some(p) = &mut engine.speculative {
                    p.abort_scheduled();
                }
                speculative_id = None;
                pending_ingest.clear();
            }
            release(&mut engine, sched.cancel_request(id));
            requests.remove(&id);
        }

        // A lone request uses the trained proposer. Bursts use batched target decode.
        // Proposer context is currently single-request; never interleave target taps with it.
        if speculative_id.is_none() {
            if sched.active.is_empty()
                && sched.waiting.len() == 1
                && !sched.waiting[0].custom_prefill
                && engine.speculative.is_some()
            {
                let id = sched.waiting[0].req_id;
                let mut proposer = engine.speculative.take().unwrap();
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
                    if let Some(seq) = sched.active.first_mut() {
                        match proposer.restore_prefix(
                            &mut engine,
                            &seq.prompt_tokens[..seq.cached_prefix_tokens],
                        ) {
                            Ok(()) => {
                                speculative_id = Some(id);
                                seq.custom_prefill = true;
                            }
                            Err(e) => {
                                let r = requests.remove(&id).unwrap();
                                let _ = r.output.try_send(GenerationEvent::Error(e.to_string()));
                                release(&mut engine, sched.cancel_request(id));
                                proposer.abort_scheduled();
                            }
                        }
                    }
                }
                engine.speculative = Some(proposer);
            }
            if speculative_id.is_none() {
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
                    sched.cfg.max_num_seqs = limit;
                    if sched.active.len() == before {
                        break;
                    }
                    reserved += pages;
                }
            }
        }

        if sched.active.is_empty() && !sched.waiting.is_empty() {
            // Admission failed with no active work that could release state slots.
            let ids: Vec<_> = sched.waiting.iter().map(|w| w.req_id).collect();
            for id in ids {
                sched.cancel_request(id);
                if let Some(r) = requests.remove(&id) {
                    let _ = r
                        .output
                        .try_send(GenerationEvent::Error("state pool exhausted".into()));
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
            while seq.radix_state.kv_pages.len() < needed {
                match engine.alloc_kv_page_from(&mut sched.radix) {
                    Some(p) => seq.radix_state.kv_pages.push(p),
                    None => {
                        failed.push((seq.req_id, "KV pool exhausted".to_string()));
                        break;
                    }
                }
            }
        }
        for (id, error) in failed {
            if let Some(r) = requests.remove(&id) {
                let _ = r.output.try_send(GenerationEvent::Error(error));
            }
            release(&mut engine, sched.cancel_request(id));
            if speculative_id == Some(id) {
                engine.speculative.as_mut().unwrap().abort_scheduled();
                speculative_id = None;
                pending_ingest.clear();
            }
        }

        let result = if let Some(id) = speculative_id {
            let mut proposer = engine.speculative.take().unwrap();
            let seq = sched.active.iter_mut().find(|s| s.req_id == id).unwrap();
            let r = requests.get_mut(&id).unwrap();
            let gpu = !r
                .sampler
                .params
                .is_plain_greedy(r.sampler.constraint.is_some());
            let result = if !seq.is_prefill_complete() {
                proposer
                    .step_scheduled_prefill_chunk(
                        &mut engine,
                        &mut sched.radix,
                        seq,
                        &mut r.sampler,
                        gpu,
                        config.chunk_prefill_size,
                    )
                    .map(|token| token.into_iter().collect::<Vec<_>>())
            } else {
                proposer
                    .step_scheduled_verify_round(
                        &mut engine,
                        &mut sched.radix,
                        seq,
                        &mut pending_ingest,
                        &mut r.sampler,
                        gpu,
                    )
                    .map(|(tokens, lps)| {
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
            let result = result.and_then(|tokens| {
                for (token, lp) in tokens {
                    emit(r, token, lp);
                }
                if seq.finished {
                    proposer.finish_scheduled(&mut engine, &mut pending_ingest)?;
                }
                Ok(())
            });
            engine.speculative = Some(proposer);
            if seq.finished {
                release(&mut engine, sched.finish_request_early(id));
                speculative_id = None;
            }
            result.map_err(|e| e.to_string())
        } else {
            step_plain(&mut engine, &mut sched, &mut requests)
        };
        if let Err(error) = result {
            // GPU errors invalidate this batch; abort private state rather than publishing it.
            let ids: Vec<_> = sched.active.iter().map(|s| s.req_id).collect();
            for id in ids {
                if let Some(r) = requests.remove(&id) {
                    let _ = r.output.try_send(GenerationEvent::Error(error.clone()));
                }
                release(&mut engine, sched.cancel_request(id));
            }
            if let Some(p) = &mut engine.speculative {
                p.abort_scheduled();
            }
            speculative_id = None;
            pending_ingest.clear();
        }
        for (bucket, _) in engine.captured_decode_graph_keys() {
            stats
                .decode_bucket_mask
                .fetch_or(1 << bucket.trailing_zeros(), Ordering::Relaxed);
        }
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
                let _ = r.output.try_send(GenerationEvent::Finished {
                    cached_tokens: seq.cached_prefix_tokens,
                    generated_tokens: seq.generated_tokens.len(),
                });
            }
        }
        std::mem::swap(&mut engine.radix, &mut sched.radix);
    }
    std::mem::swap(&mut engine.radix, &mut sched.radix);
    let ids: Vec<_> = requests.keys().copied().collect();
    for id in ids {
        release(&mut engine, sched.cancel_request(id));
    }
    if let Some(p) = &mut engine.speculative {
        p.abort_scheduled();
    }
    std::mem::swap(&mut engine.radix, &mut sched.radix);
}

fn emit(request: &Request, id: i32, logprobs: Option<StepLogprobs>) {
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
    error: String,
) {
    if let Some(r) = requests.remove(&id) {
        let _ = r.output.try_send(GenerationEvent::Error(error));
    }
    release(engine, sched.cancel_request(id));
}

fn step_plain(
    engine: &mut Qwen3_5Engine,
    sched: &mut SchedulerV1,
    requests: &mut HashMap<u64, Request>,
) -> Result<(), String> {
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
        let request = requests.get_mut(&id).unwrap();
        let (greedy, slice) = request
            .vision
            .as_mut()
            .unwrap()
            .step_prefill(engine, seq, sched.cfg.chunk_prefill_size)
            .map_err(|e| e.to_string())?;
        let token = if slice.is_final_prefill_chunk {
            let sample = if request
                .sampler
                .params
                .is_plain_greedy(request.sampler.constraint.is_some())
            {
                request
                    .sampler
                    .try_commit_token(greedy)
                    .map(|()| (greedy, None))
            } else {
                engine
                    .sample_single_gpu_step(&mut request.sampler)
                    .map_err(|e| e.to_string())
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
    let greedy = engine
        .step_batch_slices_with_pos_deltas(&batch, Some(&deltas))
        .map_err(|e| e.to_string())?;
    let needs_gpu = slices.iter().any(|s| {
        (!s.is_prefill_chunk || s.is_final_prefill_chunk) && {
            let r = &requests[&s.req_id];
            !r.sampler
                .params
                .is_plain_greedy(r.sampler.constraint.is_some())
        }
    });
    let sampled = if needs_gpu {
        let mut masks = Vec::new();
        let mut sparse = Vec::new();
        let mut rows = Vec::new();
        for s in &slices {
            let sampler = &requests[&s.req_id].sampler;
            rows.push(sampler.append_gpu_row(
                engine.cfg.vocab_size,
                &engine.cfg.eos_token_ids,
                SamplerV1::uniform_f32(sampler.params.seed, sampler.output_history.len()),
                0.0,
                -1,
                false,
                &mut masks,
                &mut sparse,
            )?);
        }
        Some(
            engine
                .run_gpu_sampler_rows(&rows, &masks, &sparse)
                .map_err(|e| e.to_string())?,
        )
    } else {
        None
    };
    for (i, s) in slices.iter().enumerate() {
        let token = if !s.is_prefill_chunk || s.is_final_prefill_chunk {
            let r = requests.get_mut(&s.req_id).unwrap();
            let (id, lp) = if let Some(rows) = &sampled {
                let out = &rows[i];
                if out.status != 0 {
                    fail_request(
                        engine,
                        sched,
                        requests,
                        s.req_id,
                        format!("GPU sampler status {}", out.status),
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
            if let Err(error) = r.sampler.try_commit_token(id) {
                fail_request(engine, sched, requests, s.req_id, error);
                continue;
            }
            emit(r, id, lp);
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
