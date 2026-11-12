use std::{
    collections::HashMap,
    sync::atomic::{AtomicBool, Ordering},
};
use tokio::sync::mpsc::Receiver;

use super::{
    DecisionReadoutTap, DecisionTapOutput, EngineBatchSlice, GenerationEvent,
    PipelinedDecodeOutput, Qwen3_5Engine, Request, SamplerV1, ScheduledReqSlice, SchedulerV1,
    WorkerError, release,
    runner::{emit, fail_request},
};

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
            let mut target_slots = Vec::new();
            let mut batch_rows = Vec::new();
            for (p_idx, &pos) in positions.iter().enumerate() {
                if (chunk_start..chunk_end).contains(&pos) {
                    target_slots.push(p_idx);
                    batch_rows.push(tok_off + (pos - chunk_start));
                }
            }
            let read_err = if !batch_rows.is_empty() {
                match engine.read_norm_hidden_rows_f32(&batch_rows) {
                    Ok(rows) => {
                        for (p_idx, row) in target_slots.into_iter().zip(rows) {
                            captured[p_idx] = Some(row);
                        }
                        None
                    }
                    Err(e) => Some(WorkerError::Cuda(e)),
                }
            } else {
                None
            };
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
                        "code_readout: missing hidden position after final prefill chunk",
                    ))
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

fn alloc_prefill_checkpoint_slot(
    engine: &mut Qwen3_5Engine,
    radix: &mut crate::radix::MambaRadixCache,
    s: &ScheduledReqSlice,
) -> Option<i32> {
    if !s.is_prefill_chunk {
        return None;
    }
    if s.is_final_prefill_chunk || s.ctx_len_before < 2048 {
        engine.alloc_state_slot_from(radix)
    } else {
        engine.alloc_state_slot_preserving_leaves(radix)
    }
}

fn batch_is_noncausal_prefix_prefill(
    sched: &SchedulerV1,
    requests: &HashMap<u64, Request>,
    slices: &[ScheduledReqSlice],
) -> bool {
    let slice_is_bidir = |s: &ScheduledReqSlice| {
        let single_pos_readout = requests.get(&s.req_id).is_none_or(|r| {
            !matches!(
                &r.decision_tap,
                Some(DecisionReadoutTap::HiddenPositions { positions, .. }) if positions.len() > 1
            )
        });
        s.is_prefill_chunk
            && sched
                .active
                .iter()
                .find(|seq| seq.req_id == s.req_id)
                .is_some_and(|seq| {
                    seq.prefix_bidir_len.is_some_and(|bidir_end| {
                        s.ctx_len_before + s.query_tokens.len() <= bidir_end
                            || (seq.max_new_tokens <= 1 && single_pos_readout)
                    })
                })
    };
    let noncausal = !slices.is_empty() && slices.iter().all(slice_is_bidir);
    debug_assert!(
        slices.iter().all(|s| slice_is_bidir(s) == noncausal),
        "scheduler must not mix bidirectional prefix prefill slices with causal slices"
    );
    noncausal
}

pub(super) fn step_plain(
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
        engine.set_active_lora_id(seq.lora_id);
        engine.clear_slice_lora_ids();
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
    let slice_lora_ids: Vec<i32> = slices.iter().map(|s| s.lora_id).collect();
    engine.set_slice_lora_ids(&slice_lora_ids);
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
        .map(|s| {
            let needs_vocab_logits = requests.get(&s.req_id).is_some_and(|r| {
                r.sampler.params.max_tokens > 0
                    || matches!(
                        r.decision_tap,
                        Some(DecisionReadoutTap::CandidateLogits { .. })
                    )
            });
            (!s.is_prefill_chunk || s.is_final_prefill_chunk) && needs_vocab_logits
        })
        .collect();
    let noncausal_override = batch_is_noncausal_prefix_prefill(sched, requests, &slices);
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

    let greedy = engine.step_batch_slices_for_sampler_filtered_bidir(
        &batch,
        Some(&deltas),
        Some(&emit_mask),
        needs_gpu,
        noncausal_override,
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
        let mut replacement = alloc_prefill_checkpoint_slot(engine, &mut sched.radix, s);
        release(engine, sched.commit_slice(s, token, || replacement.take()));
        if let Some(slot) = replacement {
            engine.free_state_slots(&[slot]);
        }
        if requests
            .get(&s.req_id)
            .and_then(|r| r.sampler.constraint.as_ref())
            .is_some_and(|c| c.is_terminal_eos())
        {
            release(engine, sched.finish_request_early(s.req_id));
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
                    lora_id: s.lora_id,
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
            let out_next = wait_and_release_deferred(
                engine,
                |engine| engine.wait_pipelined_decode_step(ticket_next),
                Qwen3_5Engine::abort_pipelined_decode,
                deferred_releases,
                release,
            )
            .map_err(WorkerError::Cuda)?;
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

fn wait_and_release_deferred<C, T, E>(
    context: &mut C,
    wait: impl FnOnce(&mut C) -> Result<T, E>,
    abort: impl FnOnce(&mut C),
    releases: Vec<Option<crate::radix::RadixDonationOutcome>>,
    mut release: impl FnMut(&mut C, Option<crate::radix::RadixDonationOutcome>),
) -> Result<T, E> {
    let output = wait(context);
    if output.is_err() {
        // Abort synchronizes both streams before transferred scheduler resources
        // can be freed. Cancellation cannot recover these deferred records.
        abort(context);
    }
    for outcome in releases {
        release(context, outcome);
    }
    output
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

#[cfg(test)]
mod deferred_release_tests {
    use super::wait_and_release_deferred;
    use crate::radix::RadixDonationOutcome;

    fn check(fail: bool) {
        let donations = vec![
            Some(RadixDonationOutcome {
                freed_kv_pages: vec![11, 12],
                freed_mamba_slots: vec![21],
                ..Default::default()
            }),
            None,
            Some(RadixDonationOutcome {
                freed_kv_pages: vec![13],
                freed_mamba_slots: vec![22, 23],
                ..Default::default()
            }),
        ];
        let mut state = (Vec::new(), Vec::new());
        let original_error = Box::new("original wait error");
        let error_ptr = std::ptr::from_ref(original_error.as_ref());
        let result = wait_and_release_deferred(
            &mut state,
            |state| {
                state.0.push("wait");
                if fail { Err(original_error) } else { Ok(42) }
            },
            |state| state.0.push("abort"),
            donations.clone(),
            |state, outcome| {
                state.0.push("release");
                state.1.push(outcome);
            },
        );
        if fail {
            assert_eq!(std::ptr::from_ref(result.unwrap_err().as_ref()), error_ptr);
            assert_eq!(state.0, ["wait", "abort", "release", "release", "release"]);
        } else {
            assert_eq!(result, Ok(42));
            assert_eq!(state.0, ["wait", "release", "release", "release"]);
        }
        assert_eq!(
            state.1, donations,
            "deliver every resource record exactly once"
        );
    }

    #[test]
    fn failed_wait_aborts_before_delivering_deferred_resources() {
        check(true);
    }

    #[test]
    fn successful_wait_delivers_deferred_resources_without_abort() {
        check(false);
    }
}
