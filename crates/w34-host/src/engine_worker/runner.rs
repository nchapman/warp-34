use std::{
    collections::HashMap,
    sync::{
        Arc, Mutex,
        atomic::{AtomicBool, Ordering},
    },
};
use tokio::sync::mpsc::{self, Receiver};

use super::{
    GenerationEvent, Qwen3_5Engine, Request, SchedulerConfig, SchedulerV1, StepLogprobs,
    WorkerError, WorkerStats, decode::step_plain, enqueue, notify_request_error,
    panic_payload_message, publish_pools, release,
};

/// On any exit, closes ingress and terminates active and queued requests. A panic
/// first publishes dead health and its actual diagnostic; normal return reports
/// [`WorkerError::WorkerStopped`]. CUDA engine ownership must stay outside `body`
/// so engine teardown cannot delay publication of poisoned health.
pub(super) fn run_supervised_loop(
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

pub(super) fn run(
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

/// A failed host-assisted graph may have written invalid KV/recurrent state.
/// Escalate to the existing supervised worker boundary: it closes ingress,
/// publishes failed health and notifies active/queued requests before CUDA teardown.
/// Do not clear the error and keep serving from a potentially corrupted model.
pub(super) fn require_healthy_host_compute<'a>(
    failures: impl IntoIterator<Item = &'a crate::cuda::HostFailures>,
) {
    for failures in failures {
        if let Err(error) = failures.check() {
            panic!("host-assisted compute failed; restart engine: {error}");
        }
    }
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
        require_healthy_host_compute(engine.ranks.iter().map(|r| &r.stream.host_failures));
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

pub(super) fn emit(request: &Request, id: i32, logprobs: Option<StepLogprobs>) {
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

pub(super) fn fail_request(
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
