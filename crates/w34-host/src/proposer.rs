//! Native proposer orchestration and transactional greedy generation.
//! Warm target radix hits are used only when the matching committed proposer
//! features are available. A bounded, single-branch host memo rematerializes
//! draft KV without rerunning target layers; it is not a second target cache.
use crate::{
    cache::StateTx,
    cuda::{DeviceBuffer, Result, check_rc, f16_to_f32},
    dflash::{DFLASH2_MAX_BATCH_ROWS, DFlash2Rank},
    loader::resolve_hf_model_dir,
    model::{EngineBatchSlice, Qwen3_5Engine, RadixBenchMetrics, loader_err},
    mtp::{MAX_MTP_BATCH_SLOTS, Qwen3_5MtpRank},
    parser::{SamplerV1, SamplingParams, TokenBitmaskConstraint},
    radix::{PlannedCheckpoint, ReqRadixState},
    speculative::RejectionSampler,
};
use std::time::Instant;
use w34_kernels::ffi;

mod scheduled;

fn generation_stopped(sampler: &SamplerV1, anchor: i32, stop_eos: bool, model_eos: &[i32]) -> bool {
    // A closed grammar cannot be sampled again, even below min_tokens or when
    // ordinary EOS stopping is disabled.
    sampler
        .constraint
        .as_ref()
        .is_some_and(|c| c.is_terminal_eos())
        || (stop_eos
            && sampler.output_history.len() >= sampler.params.min_tokens
            && sampler.params.is_stop_or_eos(anchor, model_eos))
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SpeculativeKind {
    Mtp,
    DFlash2,
}
#[derive(Clone, Debug)]
pub struct SpeculativeConfig {
    pub kind: SpeculativeKind,
    pub draft_model: Option<String>,
    pub mtp_k: usize,
}
enum Ranks {
    Mtp(Vec<Qwen3_5MtpRank>),
    DFlash(Vec<DFlash2Rank>),
}
struct PrefixMemo {
    tokens: Vec<i32>,
    features: Vec<u8>,
}
struct MtpBatchSlot {
    tail: Vec<DeviceBuffer>,
    captured_verify_hidden: Vec<DeviceBuffer>,
    resident_tokens: Vec<i32>,
    processed: usize,
    current: Option<PrefixMemo>,
}
pub struct NativeProposer {
    ranks: Ranks,
    capacity: usize,
    total_capacity: usize,
    max_rows: usize,
    row_bytes: usize,
    k: usize,
    max_k: usize,
    tail: Vec<DeviceBuffer>,
    shifted: Vec<DeviceBuffer>,
    memos: Vec<PrefixMemo>,
    current: Option<PrefixMemo>,
    resident_tokens: Vec<i32>,
    processed: usize,
    mtp_batch_active_slot: usize,
    mtp_batch_slots: Vec<MtpBatchSlot>,
    pub last_accepted: Vec<usize>,
    /// Draft rows actually included in the last successful scheduled verify call
    /// (summed across requests for batched MTP), after budget/grammar/stop trimming.
    pub(crate) last_verified_draft_tokens: usize,
    pub last_restored_tokens: usize,
}
const MEMO_BUDGET: usize = 256 * 1024 * 1024;
const MAX_MEMO_ENTRIES: usize = 32;
const MTP_STAGE_ROWS: usize = 256;
const TIERED_MOE_MTP_MAX_TAP_ROWS: usize = 6144;
const TIERED_MOE_DFLASH2_MAX_TAP_ROWS: usize = 2048;
impl NativeProposer {
    pub fn set_mtp_k(&mut self, k: usize) -> Result<()> {
        if !matches!(self.ranks, Ranks::Mtp(_)) || !(1..=self.max_k).contains(&k) {
            return Err(loader_err(
                "set_mtp_k requires MTP proposer and k in 1..=max_k",
            ));
        }
        self.k = k;
        Ok(())
    }

    pub fn load(engine: &mut Qwen3_5Engine, config: &SpeculativeConfig) -> Result<Self> {
        for t in &mut engine.ranks {
            t.decode_graph = None;
            t.spec_verify_graph = None;
            t.batched_decode_graphs.clear();
        }
        if config.kind == SpeculativeKind::Mtp
            && (!matches!(config.mtp_k, 1..=3) || config.draft_model.is_some())
        {
            return Err(loader_err(
                "native MTP requires k=1..3 and uses the target checkpoint",
            ));
        }
        let capacity = engine.ranks[0].kv_pool.num_pages * 16;
        let max_rows = if engine.cfg.uses_tiered_moe() {
            match config.kind {
                SpeculativeKind::Mtp => capacity.min(TIERED_MOE_MTP_MAX_TAP_ROWS),
                SpeculativeKind::DFlash2 => capacity.min(TIERED_MOE_DFLASH2_MAX_TAP_ROWS),
            }
        } else {
            capacity.min(DFLASH2_MAX_BATCH_ROWS)
        };
        if max_rows < 16 {
            return Err(loader_err("insufficient speculative context capacity"));
        }
        let h = engine.cfg.hidden_size;
        let mtp_row_bytes = engine.cfg.hyper_hidden_size() * 2;
        let use_graph = engine.use_cuda_graph;
        let (ranks, taps, k, row_bytes) = match config.kind {
            SpeculativeKind::Mtp => {
                let model_dir = &engine.model_dir;
                let ranks = if engine.ranks.len() <= 1 {
                    engine
                        .ranks
                        .iter()
                        .map(|t| {
                            let mut r = Qwen3_5MtpRank::load(t, model_dir, capacity)?;
                            r.use_cuda_graph = use_graph && r.supports_cuda_graph();
                            Ok(r)
                        })
                        .collect::<Result<Vec<_>>>()?
                } else {
                    std::thread::scope(|s| -> Result<Vec<Qwen3_5MtpRank>> {
                        let handles: Vec<_> = engine
                            .ranks
                            .iter()
                            .map(|t| {
                                s.spawn(move || {
                                    let mut r = Qwen3_5MtpRank::load(t, model_dir, capacity)?;
                                    r.use_cuda_graph = use_graph && r.supports_cuda_graph();
                                    Ok(r)
                                })
                            })
                            .collect();
                        let mut out = Vec::with_capacity(handles.len());
                        for h in handles {
                            out.push(
                                h.join()
                                    .map_err(|_| loader_err("MTP rank loader thread panicked"))??,
                            );
                        }
                        Ok(out)
                    })?
                };
                (Ranks::Mtp(ranks), vec![], config.mtp_k, mtp_row_bytes)
            }
            SpeculativeKind::DFlash2 => {
                let name = config
                    .draft_model
                    .as_ref()
                    .ok_or_else(|| loader_err("DFlash2 requires --draft-model"))?;
                let dir = resolve_hf_model_dir(name)
                    .map_err(|_| loader_err("resolve DFlash2 checkpoint"))?;
                let dflash_max_rows = max_rows.min(DFLASH2_MAX_BATCH_ROWS);
                let ranks = if engine.ranks.len() <= 1 {
                    engine
                        .ranks
                        .iter()
                        .map(|t| {
                            let mut r = DFlash2Rank::load(t, &dir, capacity, dflash_max_rows)?;
                            r.use_cuda_graph = use_graph;
                            Ok(r)
                        })
                        .collect::<Result<Vec<_>>>()?
                } else {
                    let dir_ref = &dir;
                    std::thread::scope(|s| -> Result<Vec<DFlash2Rank>> {
                        let handles: Vec<_> = engine
                            .ranks
                            .iter()
                            .map(|t| {
                                s.spawn(move || {
                                    let mut r =
                                        DFlash2Rank::load(t, dir_ref, capacity, dflash_max_rows)?;
                                    r.use_cuda_graph = use_graph;
                                    Ok(r)
                                })
                            })
                            .collect();
                        let mut out = Vec::with_capacity(handles.len());
                        for h in handles {
                            out.push(h.join().map_err(|_| {
                                loader_err("DFlash2 rank loader thread panicked")
                            })??);
                        }
                        Ok(out)
                    })?
                };
                let taps = ranks[0].cfg.target_layers.clone();
                let k = ranks[0].cfg.block_size - 1;
                let row_bytes = h * taps.len() * 2;
                (Ranks::DFlash(ranks), taps, k, row_bytes)
            }
        };
        let tail = engine
            .ranks
            .iter()
            .map(|t| DeviceBuffer::alloc(t.device.device_id, mtp_row_bytes))
            .collect::<Result<Vec<_>>>()?;
        let shifted = engine
            .ranks
            .iter()
            .map(|t| {
                DeviceBuffer::alloc(t.device.device_id, max_rows.min(MTP_STAGE_ROWS) * row_bytes)
            })
            .collect::<Result<Vec<_>>>()?;
        let logits = engine
            .ranks
            .iter()
            .map(|t| DeviceBuffer::alloc(t.device.device_id, (k + 1) * t.local_vocab_size * 2))
            .collect::<Result<Vec<_>>>()?;
        for rank in 0..engine.ranks.len() {
            if let Err(e) = engine.ranks[rank].configure_speculative_taps(
                &taps,
                config.kind == SpeculativeKind::Mtp,
                max_rows,
            ) {
                for t in &mut engine.ranks {
                    let _ = t.configure_speculative_taps(&[], false, 0);
                }
                return Err(e);
            }
        }
        for (t, buffer) in engine.ranks.iter_mut().zip(logits) {
            t.speculative_logits = Some(buffer);
        }
        let mtp_batch_slots = if config.kind == SpeculativeKind::Mtp {
            (0..MAX_MTP_BATCH_SLOTS)
                .map(|_| {
                    let slot_tail = engine
                        .ranks
                        .iter()
                        .map(|t| DeviceBuffer::alloc(t.device.device_id, mtp_row_bytes))
                        .collect::<Result<Vec<_>>>()?;
                    let captured_verify_hidden = engine
                        .ranks
                        .iter()
                        .map(|t| DeviceBuffer::alloc(t.device.device_id, (k + 1) * mtp_row_bytes))
                        .collect::<Result<Vec<_>>>()?;
                    Ok(MtpBatchSlot {
                        tail: slot_tail,
                        captured_verify_hidden,
                        resident_tokens: vec![],
                        processed: 0,
                        current: None,
                    })
                })
                .collect::<Result<Vec<_>>>()?
        } else {
            vec![]
        };
        Ok(Self {
            ranks,
            capacity,
            total_capacity: capacity,
            max_rows,
            row_bytes,
            k,
            max_k: k,
            tail,
            shifted,
            memos: vec![],
            current: None,
            resident_tokens: vec![],
            processed: 0,
            mtp_batch_active_slot: 0,
            mtp_batch_slots,
            last_accepted: vec![],
            last_verified_draft_tokens: 0,
            last_restored_tokens: 0,
        })
    }
    fn reset(&mut self) -> Result<()> {
        self.resident_tokens.clear();
        self.processed = 0;
        self.last_accepted.clear();
        self.last_restored_tokens = 0;
        match &mut self.ranks {
            Ranks::Mtp(rs) => {
                for r in rs {
                    r.truncate(0)?;
                }
            }
            Ranks::DFlash(rs) => {
                for r in rs {
                    r.truncate(0)?;
                }
            }
        }
        Ok(())
    }
    fn commit_current_memo(&mut self) {
        let Some(memo) = self.current.take() else {
            return;
        };
        if memo.tokens.is_empty() {
            return;
        }
        if matches!(self.ranks, Ranks::DFlash(_)) {
            self.memos.clear();
            self.memos.push(memo);
            return;
        }
        self.memos
            .retain(|existing| !memo.tokens.starts_with(&existing.tokens));
        if let Some(pos) = self
            .memos
            .iter()
            .position(|existing| existing.tokens.starts_with(&memo.tokens))
        {
            let existing = self.memos.remove(pos);
            self.memos.push(existing);
            return;
        }
        let mut total_bytes: usize = self.memos.iter().map(|m| m.features.len()).sum();
        while !self.memos.is_empty()
            && (self.memos.len() >= MAX_MEMO_ENTRIES
                || total_bytes.saturating_add(memo.features.len()) > MEMO_BUDGET)
        {
            let removed = self.memos.remove(0);
            total_bytes = total_bytes.saturating_sub(removed.features.len());
        }
        self.memos.push(memo);
    }
    fn can_restore(&self, tokens: &[i32]) -> bool {
        if tokens.is_empty() {
            return true;
        }
        match &self.ranks {
            Ranks::DFlash(_) => {
                self.memos.iter().any(|m| m.tokens.starts_with(tokens))
                    && self.resident_tokens.starts_with(tokens)
            }
            Ranks::Mtp(_) => {
                let need_bytes = tokens.len().saturating_mul(self.row_bytes);
                self.memos
                    .iter()
                    .any(|m| m.tokens.starts_with(tokens) && m.features.len() >= need_bytes)
            }
        }
    }
    fn restore(&mut self, engine: &mut Qwen3_5Engine, tokens: &[i32]) -> Result<()> {
        if tokens.is_empty() {
            self.reset()?;
            self.current = Some(PrefixMemo {
                tokens: vec![],
                features: vec![],
            });
            return Ok(());
        }
        let need_mtp_bytes = matches!(self.ranks, Ranks::Mtp(_))
            .then(|| tokens.len().saturating_mul(self.row_bytes))
            .unwrap_or(0);
        // Fast path: if the drafter's resident GPU KV cache already starts with `tokens`,
        // truncate the drafter KV cache in-place and restore MTP's 1-row tail without
        // re-running drafter prefill over the cached prefix.
        if self.resident_tokens.starts_with(tokens)
            && let Some(idx) = self
                .memos
                .iter()
                .rposition(|m| m.tokens.starts_with(tokens) && m.features.len() >= need_mtp_bytes)
        {
            let memo = self.memos.remove(idx);
            let n = tokens.len();
            match &mut self.ranks {
                Ranks::DFlash(rs) => {
                    for r in rs {
                        r.truncate(n)?;
                    }
                }
                Ranks::Mtp(rs) => {
                    let tail_bytes = &memo.features[(n - 1) * self.row_bytes..n * self.row_bytes];
                    for ((r, t), tail) in rs.iter_mut().zip(&engine.ranks).zip(&self.tail) {
                        r.truncate(n - 1)?;
                        tail.copy_from_host(tail_bytes, &t.stream)?;
                    }
                }
            }
            self.resident_tokens.truncate(n);
            self.processed = n;
            self.last_accepted.clear();
            self.last_restored_tokens = n;
            let features = if memo.features.len() >= n * self.row_bytes {
                memo.features[..n * self.row_bytes].to_vec()
            } else {
                vec![]
            };
            self.memos.push(memo);
            self.current = Some(PrefixMemo {
                tokens: tokens.to_vec(),
                features,
            });
            return Ok(());
        }
        self.reset()?;
        self.current = None;
        let idx = self
            .memos
            .iter()
            .rposition(|m| m.tokens.starts_with(tokens) && m.features.len() >= need_mtp_bytes)
            .ok_or_else(|| loader_err("missing proposer prefix"))?;
        let memo = self.memos.remove(idx);
        let step = self.max_rows.min(MTP_STAGE_ROWS);
        let result = (|| {
            for start in (0..tokens.len()).step_by(step) {
                let end = (start + step).min(tokens.len());
                let bytes = &memo.features[start * self.row_bytes..end * self.row_bytes];
                let mut pointers = Vec::with_capacity(engine.tp_size);
                for t in &engine.ranks {
                    let stage = t
                        .last_final_hidden_f16
                        .as_ref()
                        .ok_or_else(|| loader_err("missing target feature capture"))?;
                    stage.copy_from_host(bytes, &t.stream)?;
                    pointers.push(stage.addr());
                }
                self.ingest(engine, &tokens[start..end], &pointers)?;
            }
            Ok(())
        })();
        let restored_features = if result.is_ok() {
            let prefix_bytes = tokens.len() * self.row_bytes;
            Some(memo.features[..prefix_bytes].to_vec())
        } else {
            None
        };
        self.memos.push(memo);
        result?;
        if let Some(features) = restored_features {
            self.current = Some(PrefixMemo {
                tokens: tokens.to_vec(),
                features,
            });
        }
        self.last_restored_tokens = tokens.len();
        Ok(())
    }
    fn captured(&self, engine: &Qwen3_5Engine) -> Result<Vec<u64>> {
        engine
            .ranks
            .iter()
            .map(|t| {
                match &self.ranks {
                    Ranks::Mtp(_) => t.last_final_hidden_f16.as_ref(),
                    Ranks::DFlash(_) => t.tap_aux_bf16.as_ref(),
                }
                .map(DeviceBuffer::addr)
                .ok_or_else(|| loader_err("missing target feature capture"))
            })
            .collect()
    }
    fn ingest(
        &mut self,
        engine: &mut Qwen3_5Engine,
        tokens: &[i32],
        features: &[u64],
    ) -> Result<()> {
        if tokens.is_empty() {
            return Ok(());
        }
        let stage_cap = self.max_rows.min(MTP_STAGE_ROWS);
        if tokens.len() > stage_cap {
            let mut offset = 0;
            while offset < tokens.len() {
                let take = (tokens.len() - offset).min(stage_cap);
                let sub_features: Vec<u64> = features
                    .iter()
                    .map(|&p| p + (offset * self.row_bytes) as u64)
                    .collect();
                self.ingest(engine, &tokens[offset..offset + take], &sub_features)?;
                offset += take;
            }
            return Ok(());
        }
        if features.len() != engine.tp_size
            || tokens.len() > self.max_rows
            || self.processed + tokens.len() > self.capacity
        {
            return Err(loader_err("invalid committed proposer span"));
        }
        // Host feature memo is only needed by MTP to restore its 1-row hidden tail or replay
        // non-resident prefixes; DFlash2 keeps its full draft KV cache resident on GPU and
        // restores warm radix hits in O(1) via `r.truncate(n)` without host D2H copies.
        let need_host_features = matches!(self.ranks, Ranks::Mtp(_));
        if let Some(memo) = &self.current
            && need_host_features
            && (memo.tokens.len() + tokens.len()) * self.row_bytes > MEMO_BUDGET
        {
            self.current = None;
        }
        if let Some(memo) = &mut self.current {
            if need_host_features {
                let t0 = &engine.ranks[0];
                t0.device.make_current()?;
                let len_bytes = tokens.len() * self.row_bytes;
                let old_len = memo.features.len();
                memo.features.resize(old_len + len_bytes, 0);
                check_rc(
                    // SAFETY: features[0] spans rows*row_bytes on rank 0's stream, and
                    // rank 0's stream is synchronized before ingest returns below.
                    unsafe {
                        ffi::w34_memcpy_d2h_async(
                            memo.features[old_len..].as_mut_ptr().cast(),
                            features[0] as *const _,
                            len_bytes,
                            t0.stream.handle,
                        )
                    },
                    "proposer memo copy",
                )?;
            }
            memo.tokens.extend_from_slice(tokens);
        }
        let ingest_res: Result<()> = match &mut self.ranks {
            Ranks::DFlash(rs) => {
                let n = tokens.len();
                std::thread::scope(|scope| {
                    let jobs: Vec<_> = rs
                        .iter_mut()
                        .zip(&engine.ranks)
                        .zip(features)
                        .map(|((r, t), &ptr)| scope.spawn(move || r.ingest(t, ptr, n)))
                        .collect();
                    for job in jobs {
                        job.join()
                            .map_err(|_| loader_err("DFlash ingest rank panic"))??;
                    }
                    Ok(())
                })
            }
            Ranks::Mtp(rs) => {
                let row_bytes = self.row_bytes;
                let before = self.processed;
                let comm = engine.tp2_comm.as_deref();
                std::thread::scope(|scope| {
                    let mut jobs = Vec::new();
                    for ((((r, t), tail), shifted), &ptr) in rs
                        .iter_mut()
                        .zip(&mut engine.ranks)
                        .zip(&self.tail)
                        .zip(&self.shifted)
                        .zip(features)
                    {
                        jobs.push(scope.spawn(move || -> Result<()> {
                            t.device.make_current()?;
                            r.truncate(before.saturating_sub(1))?;
                            let (ids, input) = if before == 0 {
                                (&tokens[1..], ptr)
                            } else {
                                check_rc(
                                    // SAFETY: both resident buffers hold at least one MTP hidden row.
                                    unsafe {
                                        ffi::w34_memcpy_d2d_async(
                                            shifted.addr() as *mut _,
                                            tail.addr() as *const _,
                                            row_bytes,
                                            t.stream.handle,
                                        )
                                    },
                                    "MTP preceding hidden",
                                )?;
                                if tokens.len() > 1 {
                                    check_rc(
                                        // SAFETY: rows<=max_rows and the final shifted row fits;
                                        // source is the owned live capture for this whole batch.
                                        unsafe {
                                            ffi::w34_memcpy_d2d_async(
                                                (shifted.addr() + row_bytes as u64) as *mut _,
                                                ptr as *const _,
                                                (tokens.len() - 1) * row_bytes,
                                                t.stream.handle,
                                            )
                                        },
                                        "MTP shifted hidden",
                                    )?;
                                }
                                (tokens, shifted.addr())
                            };
                            r.ingest_kv_only_no_sync(t, ids, input, comm)?;
                            check_rc(
                                // SAFETY: tokens is nonempty and ptr spans every captured row;
                                // tail owns exactly one MTP hidden row and survives synchronization.
                                unsafe {
                                    ffi::w34_memcpy_d2d_async(
                                        tail.addr() as *mut _,
                                        (ptr + ((tokens.len() - 1) * row_bytes) as u64) as *const _,
                                        row_bytes,
                                        t.stream.handle,
                                    )
                                },
                                "MTP tail hidden",
                            )?;
                            t.stream.synchronize()
                        }));
                    }
                    jobs.into_iter()
                        .map(|j| j.join().map_err(|_| loader_err("MTP ingest rank panic"))?)
                        .collect::<Result<Vec<_>>>()
                })
                .map(|_| ())
            }
        };
        if let Err(e) = ingest_res {
            let _ = engine.ranks[0].device.make_current();
            let _ = engine.ranks[0].stream.synchronize();
            return Err(e);
        }
        self.resident_tokens.extend_from_slice(tokens);
        self.processed += tokens.len();
        Ok(())
    }
    fn ingest_and_propose(
        &mut self,
        engine: &mut Qwen3_5Engine,
        tokens: &[i32],
        features: &[u64],
        anchor: i32,
        k: usize,
    ) -> Result<Vec<i32>> {
        if tokens.is_empty() {
            return self.propose(engine, anchor, k);
        }
        if k == 0 {
            self.ingest(engine, tokens, features)?;
            return Ok(vec![]);
        }
        if features.len() != engine.tp_size
            || tokens.len() > self.max_rows
            || self.processed + tokens.len() > self.capacity
        {
            return Err(loader_err("invalid committed proposer span"));
        }
        match &mut self.ranks {
            Ranks::DFlash(rs) => {
                if let Some(memo) = &mut self.current {
                    memo.tokens.extend_from_slice(tokens);
                }
                let n = tokens.len();
                let use_graph = engine.use_cuda_graph;
                for r in rs.iter_mut() {
                    r.use_cuda_graph = use_graph;
                }
                if rs.len() == 1 {
                    let res = rs[0]
                        .ingest_no_sync(&engine.ranks[0], features[0], n)
                        .and_then(|()| rs[0].forward_select_greedy(&engine.ranks[0], anchor));
                    let mut ids = match res {
                        Ok(ids) => ids,
                        Err(e) => {
                            let _ = engine.ranks[0].device.make_current();
                            let _ = engine.ranks[0].stream.synchronize();
                            return Err(e);
                        }
                    };
                    self.resident_tokens.extend_from_slice(tokens);
                    self.processed += tokens.len();
                    ids.truncate(k);
                    return Ok(ids);
                }
                let comm = engine.tp2_comm.as_deref();
                let dflash_graph =
                    use_graph && rs.iter().all(DFlash2Rank::can_capture_select_greedy);
                let outputs_res: Result<Vec<Vec<i32>>> = if dflash_graph {
                    use crate::coordinated::{ReplayPhase, replay_ranks};
                    let replay_res = replay_ranks(rs.len(), |phase, rank| {
                        let r = &mut rs[rank];
                        let t = &engine.ranks[rank];
                        let ptr = features[rank];
                        t.device.make_current()?;
                        match phase {
                            ReplayPhase::Upload => {
                                r.ingest_no_sync(t, ptr, n)?;
                                r.prepare_select_greedy_graph_inputs_async(t, anchor)
                            }
                            ReplayPhase::Capture => r.ensure_select_greedy_graph(t, comm),
                            ReplayPhase::Launch => r.replay_select_greedy_graph_async(t),
                        }
                    });
                    match replay_res {
                        Ok(()) => rs
                            .iter_mut()
                            .zip(&engine.ranks)
                            .map(|(r, t)| r.collect_select_greedy_host(t, anchor))
                            .collect::<Result<Vec<_>>>(),
                        Err(error) => {
                            if error.partial_launch {
                                eprintln!(
                                    "[w34] fatal: asymmetric DFlash TP graph launch failure: {}",
                                    error.source
                                );
                                std::process::abort();
                            }
                            Err(error.source)
                        }
                    }
                } else {
                    std::thread::scope(|scope| {
                        let jobs: Vec<_> = rs
                            .iter_mut()
                            .zip(&mut engine.ranks)
                            .zip(features)
                            .map(|((r, t), &ptr)| {
                                scope.spawn(move || -> Result<Vec<i32>> {
                                    r.ingest_no_sync(t, ptr, n)?;
                                    r.forward_select_greedy_tp(t, anchor, comm)
                                })
                            })
                            .collect();
                        jobs.into_iter()
                            .map(|j| {
                                j.join()
                                    .map_err(|_| loader_err("DFlash ingest+proposal rank panic"))?
                            })
                            .collect::<Result<Vec<_>>>()
                    })
                };
                let mut outputs = match outputs_res {
                    Ok(o) => o,
                    Err(e) => {
                        for t in &engine.ranks {
                            let _ = t.device.make_current();
                            let _ = t.stream.synchronize();
                        }
                        return Err(e);
                    }
                };
                if outputs.windows(2).any(|p| p[0] != p[1]) {
                    return Err(loader_err("DFlash TP token disagreement"));
                }
                self.resident_tokens.extend_from_slice(tokens);
                self.processed += tokens.len();
                let mut ids = outputs.swap_remove(0);
                ids.truncate(k);
                Ok(ids)
            }
            Ranks::Mtp(rs) => {
                if tokens.len() > self.max_rows.min(MTP_STAGE_ROWS) {
                    self.ingest(engine, tokens, features)?;
                    return self.propose(engine, anchor, k);
                }
                if let Some(memo) = &self.current
                    && (memo.tokens.len() + tokens.len()) * self.row_bytes > MEMO_BUDGET
                {
                    self.current = None;
                }
                let memo_prev_lens = self
                    .current
                    .as_ref()
                    .map(|m| (m.tokens.len(), m.features.len()));
                if let Some(memo) = &mut self.current {
                    let t0 = &engine.ranks[0];
                    t0.device.make_current()?;
                    let len_bytes = tokens.len() * self.row_bytes;
                    let old_len = memo.features.len();
                    memo.features.resize(old_len + len_bytes, 0);
                    check_rc(
                        // SAFETY: features[0] spans rows*row_bytes on rank 0's stream, and
                        // rank 0's stream is synchronized before ingest_and_propose returns below.
                        unsafe {
                            ffi::w34_memcpy_d2h_async(
                                memo.features[old_len..].as_mut_ptr().cast(),
                                features[0] as *const _,
                                len_bytes,
                                t0.stream.handle,
                            )
                        },
                        "proposer memo copy",
                    )?;
                    memo.tokens.extend_from_slice(tokens);
                }
                let use_graph = engine.use_cuda_graph;
                for r in rs.iter_mut() {
                    r.use_cuda_graph = use_graph && r.supports_cuda_graph();
                }
                let mtp_graph = use_graph && rs.iter().all(Qwen3_5MtpRank::supports_cuda_graph);
                let row_bytes = self.row_bytes;
                let before = self.processed;
                let comm = engine.tp2_comm.as_deref();
                let outputs_res: Result<Vec<Vec<i32>>> = if mtp_graph {
                    use crate::coordinated::{ReplayPhase, replay_ranks};
                    let replay_res = replay_ranks(rs.len(), |phase, rank| {
                        let r = &mut rs[rank];
                        let t = &engine.ranks[rank];
                        let tail = &self.tail[rank];
                        let shifted = &self.shifted[rank];
                        let ptr = features[rank];
                        t.device.make_current()?;
                        match phase {
                            ReplayPhase::Upload => {
                                r.truncate(before.saturating_sub(1))?;
                                let (ids, input) = if before == 0 {
                                    (&tokens[1..], ptr)
                                } else {
                                    check_rc(
                                        // SAFETY: both resident buffers hold at least one MTP hidden row.
                                        unsafe {
                                            ffi::w34_memcpy_d2d_async(
                                                shifted.addr() as *mut _,
                                                tail.addr() as *const _,
                                                row_bytes,
                                                t.stream.handle,
                                            )
                                        },
                                        "MTP preceding hidden",
                                    )?;
                                    if tokens.len() > 1 {
                                        check_rc(
                                            // SAFETY: rows<=max_rows and the final shifted row fits;
                                            // source is the owned live capture for this whole batch.
                                            unsafe {
                                                ffi::w34_memcpy_d2d_async(
                                                    (shifted.addr() + row_bytes as u64) as *mut _,
                                                    ptr as *const _,
                                                    (tokens.len() - 1) * row_bytes,
                                                    t.stream.handle,
                                                )
                                            },
                                            "MTP shifted hidden",
                                        )?;
                                    }
                                    (tokens, shifted.addr())
                                };
                                r.ingest_kv_only_no_sync(t, ids, input, comm)?;
                                check_rc(
                                    // SAFETY: tokens is nonempty and ptr spans every captured row;
                                    // tail owns exactly one MTP hidden row on this device.
                                    unsafe {
                                        ffi::w34_memcpy_d2d_async(
                                            tail.addr() as *mut _,
                                            (ptr + ((tokens.len() - 1) * row_bytes) as u64)
                                                as *const _,
                                            row_bytes,
                                            t.stream.handle,
                                        )
                                    },
                                    "MTP tail hidden",
                                )?;
                                r.prepare_propose_graph_inputs_async(t, anchor, tail.addr(), k)
                            }
                            ReplayPhase::Capture => r.ensure_propose_graph(t, k, comm),
                            ReplayPhase::Launch => r.replay_propose_graph_async(t, k),
                        }
                    });
                    match replay_res {
                        Ok(()) => rs
                            .iter_mut()
                            .zip(&engine.ranks)
                            .map(|(r, t)| r.collect_propose_host(t, k))
                            .collect::<Result<Vec<_>>>(),
                        Err(error) => {
                            if error.partial_launch {
                                eprintln!(
                                    "[w34] fatal: asymmetric MTP TP graph launch failure: {}",
                                    error.source
                                );
                                std::process::abort();
                            }
                            Err(error.source)
                        }
                    }
                } else {
                    std::thread::scope(|scope| {
                        let mut jobs = Vec::new();
                        for ((((r, t), tail), shifted), &ptr) in rs
                            .iter_mut()
                            .zip(&mut engine.ranks)
                            .zip(&self.tail)
                            .zip(&self.shifted)
                            .zip(features)
                        {
                            jobs.push(scope.spawn(move || -> Result<Vec<i32>> {
                                t.device.make_current()?;
                                r.truncate(before.saturating_sub(1))?;
                                let (ids, input) = if before == 0 {
                                    (&tokens[1..], ptr)
                                } else {
                                    check_rc(
                                        // SAFETY: both resident buffers hold at least one MTP hidden row.
                                        unsafe {
                                            ffi::w34_memcpy_d2d_async(
                                                shifted.addr() as *mut _,
                                                tail.addr() as *const _,
                                                row_bytes,
                                                t.stream.handle,
                                            )
                                        },
                                        "MTP preceding hidden",
                                    )?;
                                    if tokens.len() > 1 {
                                        check_rc(
                                            // SAFETY: rows<=max_rows and the final shifted row fits;
                                            // source is the owned live capture for this whole batch.
                                            unsafe {
                                                ffi::w34_memcpy_d2d_async(
                                                    (shifted.addr() + row_bytes as u64) as *mut _,
                                                    ptr as *const _,
                                                    (tokens.len() - 1) * row_bytes,
                                                    t.stream.handle,
                                                )
                                            },
                                            "MTP shifted hidden",
                                        )?;
                                    }
                                    (tokens, shifted.addr())
                                };
                                r.ingest_kv_only_no_sync(t, ids, input, comm)?;
                                check_rc(
                                    // SAFETY: tokens is nonempty and ptr spans every captured row;
                                    // tail owns exactly one MTP hidden row on this device.
                                    unsafe {
                                        ffi::w34_memcpy_d2d_async(
                                            tail.addr() as *mut _,
                                            (ptr + ((tokens.len() - 1) * row_bytes) as u64)
                                                as *const _,
                                            row_bytes,
                                            t.stream.handle,
                                        )
                                    },
                                    "MTP tail hidden",
                                )?;
                                r.propose(t, anchor, tail.addr(), k, comm)
                            }));
                        }
                        jobs.into_iter()
                            .map(|j| {
                                j.join()
                                    .map_err(|_| loader_err("MTP ingest+proposal rank panic"))?
                            })
                            .collect::<Result<Vec<_>>>()
                    })
                };
                let outputs = match outputs_res {
                    Ok(o) => o,
                    Err(e) => {
                        for (r, t) in rs.iter_mut().zip(&engine.ranks) {
                            let _ = t.device.make_current();
                            let _ = t.stream.synchronize();
                            let _ = r.truncate(before.saturating_sub(1));
                        }
                        if let Some((prev_tok_len, prev_feat_len)) = memo_prev_lens
                            && let Some(memo) = &mut self.current
                        {
                            memo.tokens.truncate(prev_tok_len);
                            memo.features.truncate(prev_feat_len);
                        }
                        return Err(e);
                    }
                };
                if outputs.windows(2).any(|p| p[0] != p[1]) {
                    return Err(loader_err("MTP TP token disagreement"));
                }
                self.resident_tokens.extend_from_slice(tokens);
                self.processed += tokens.len();
                Ok(outputs.into_iter().next().unwrap())
            }
        }
    }

    fn propose(&mut self, engine: &mut Qwen3_5Engine, anchor: i32, k: usize) -> Result<Vec<i32>> {
        if k == 0 {
            return Ok(vec![]);
        }
        let use_graph = engine.use_cuda_graph;
        let comm = engine.tp2_comm.as_deref();
        match &mut self.ranks {
            Ranks::Mtp(rs) => {
                for r in rs.iter_mut() {
                    r.use_cuda_graph = use_graph && r.supports_cuda_graph();
                }
                let mtp_graph = use_graph && rs.iter().all(|r| r.supports_cuda_graph());
                let outputs = if mtp_graph {
                    use crate::coordinated::{ReplayPhase, replay_ranks};
                    let replay_res = replay_ranks(rs.len(), |phase, rank| {
                        let r = &mut rs[rank];
                        let t = &engine.ranks[rank];
                        let tail = &self.tail[rank];
                        t.device.make_current()?;
                        match phase {
                            ReplayPhase::Upload => {
                                r.prepare_propose_graph_inputs_async(t, anchor, tail.addr(), k)
                            }
                            ReplayPhase::Capture => r.ensure_propose_graph(t, k, comm),
                            ReplayPhase::Launch => r.replay_propose_graph_async(t, k),
                        }
                    });
                    if let Err(error) = replay_res {
                        if error.partial_launch {
                            eprintln!(
                                "[w34] fatal: asymmetric MTP TP graph launch failure: {}",
                                error.source
                            );
                            std::process::abort();
                        }
                        return Err(error.source);
                    }
                    rs.iter_mut()
                        .zip(&engine.ranks)
                        .map(|(r, t)| r.collect_propose_host(t, k))
                        .collect::<Result<Vec<_>>>()?
                } else {
                    std::thread::scope(|scope| {
                        let jobs: Vec<_> = rs
                            .iter_mut()
                            .zip(&mut engine.ranks)
                            .zip(&self.tail)
                            .map(|((r, t), tail)| {
                                scope.spawn(move || r.propose(t, anchor, tail.addr(), k, comm))
                            })
                            .collect();
                        jobs.into_iter()
                            .map(|j| {
                                j.join()
                                    .map_err(|_| loader_err("MTP proposal rank panic"))?
                            })
                            .collect::<Result<Vec<_>>>()
                    })?
                };
                if outputs.windows(2).any(|p| p[0] != p[1]) {
                    return Err(loader_err("MTP TP token disagreement"));
                }
                Ok(outputs[0].clone())
            }
            Ranks::DFlash(rs) => {
                for r in rs.iter_mut() {
                    r.use_cuda_graph = use_graph;
                }
                if rs.len() == 1 {
                    let mut ids = rs[0].forward_select_greedy(&engine.ranks[0], anchor)?;
                    ids.truncate(k);
                    return Ok(ids);
                }
                let dflash_graph =
                    use_graph && rs.iter().all(DFlash2Rank::can_capture_select_greedy);
                let mut outputs = if dflash_graph {
                    use crate::coordinated::{ReplayPhase, replay_ranks};
                    let replay_res = replay_ranks(rs.len(), |phase, rank| {
                        let r = &mut rs[rank];
                        let t = &engine.ranks[rank];
                        t.device.make_current()?;
                        match phase {
                            ReplayPhase::Upload => {
                                r.prepare_select_greedy_graph_inputs_async(t, anchor)
                            }
                            ReplayPhase::Capture => r.ensure_select_greedy_graph(t, comm),
                            ReplayPhase::Launch => r.replay_select_greedy_graph_async(t),
                        }
                    });
                    if let Err(error) = replay_res {
                        if error.partial_launch {
                            eprintln!(
                                "[w34] fatal: asymmetric DFlash TP graph launch failure: {}",
                                error.source
                            );
                            std::process::abort();
                        }
                        return Err(error.source);
                    }
                    rs.iter_mut()
                        .zip(&engine.ranks)
                        .map(|(r, t)| r.collect_select_greedy_host(t, anchor))
                        .collect::<Result<Vec<_>>>()?
                } else {
                    std::thread::scope(|scope| {
                        let jobs: Vec<_> = rs
                            .iter_mut()
                            .zip(&mut engine.ranks)
                            .map(|(r, t)| {
                                scope.spawn(move || r.forward_select_greedy_tp(t, anchor, comm))
                            })
                            .collect();
                        jobs.into_iter()
                            .map(|j| {
                                j.join()
                                    .map_err(|_| loader_err("DFlash proposal rank panic"))?
                            })
                            .collect::<Result<Vec<_>>>()
                    })?
                };
                if outputs.windows(2).any(|p| p[0] != p[1]) {
                    return Err(loader_err("DFlash TP token disagreement"));
                }
                let mut ids = outputs.swap_remove(0);
                ids.truncate(k);
                Ok(ids)
            }
        }
    }
    /// Runs one diagnostic drafter ingest + proposal step on `prompt` and `anchor`,
    /// recording `"drafter_hidden"` and `draft_proposal_ids` into `dump`.
    pub fn dump_step(
        &mut self,
        engine: &mut Qwen3_5Engine,
        prompt: &[i32],
        anchor: i32,
        dump: &mut crate::model::ForwardDump,
    ) -> Result<()> {
        self.restore(engine, &[])?;
        let features = self.captured(engine)?;
        self.ingest(engine, prompt, &features)?;
        let drafts = self.propose(engine, anchor, self.k)?;
        dump.draft_proposal_ids = drafts;
        let t0 = &engine.ranks[0];
        let h = engine.cfg.hidden_size;
        t0.device.make_current()?;
        match &self.ranks {
            Ranks::Mtp(rs) => {
                let cols = self.row_bytes / 2;
                let mut bits = vec![0u16; cols];
                rs[0].hidden.copy_to_host(&mut bits, &t0.stream)?;
                let vals: Vec<f32> = bits.into_iter().map(f16_to_f32).collect();
                dump.tensors_f32
                    .insert("drafter_hidden".to_string(), (vec![1, cols], vals));
            }
            Ranks::DFlash(rs) => {
                let rows = rs[0].cfg.block_size;
                let mut bits = vec![0u16; rows * h];
                rs[0].hidden.copy_to_host(&mut bits, &t0.stream)?;
                let vals: Vec<f32> = bits
                    .into_iter()
                    .map(|b| f32::from_bits((b as u32) << 16))
                    .collect();
                dump.tensors_f32
                    .insert("drafter_hidden".to_string(), (vec![rows, h], vals));
            }
        }
        Ok(())
    }

    pub fn generate(
        &mut self,
        engine: &mut Qwen3_5Engine,
        prompt: &[i32],
        max_new: usize,
        chunk: usize,
        stop_eos: bool,
    ) -> Result<RadixBenchMetrics> {
        self.generate_impl(engine, prompt, max_new, chunk, stop_eos, None)
    }
    /// Exact stochastic target verification with deterministic greedy proposals
    /// (q is a point mass), including processed penalties and grammar masks.
    pub fn generate_sampled(
        &mut self,
        engine: &mut Qwen3_5Engine,
        prompt: &[i32],
        params: SamplingParams,
        constraint: Option<TokenBitmaskConstraint>,
        chunk: usize,
    ) -> Result<RadixBenchMetrics> {
        if !params.stop_strings.is_empty() {
            return Err(loader_err(
                "native sampling does not yet support stop strings",
            ));
        }
        if params
            .stop_token_ids
            .iter()
            .any(|&id| id < 0 || id as usize >= engine.cfg.vocab_size)
        {
            return Err(loader_err("invalid sampling stop token"));
        }
        params
            .validate()
            .map_err(|_| loader_err("invalid sampling parameters"))?;
        if let Some(ref c) = constraint
            && !(0..engine.cfg.vocab_size as i32).any(|tid| c.is_token_allowed(tid))
        {
            return Err(loader_err("grammar has no support in the vocabulary"));
        }
        let count = params.max_tokens;
        let sampler = SamplerV1::new(params, prompt, constraint);
        self.generate_impl(engine, prompt, count, chunk, true, Some(sampler))
    }
    fn generate_impl(
        &mut self,
        engine: &mut Qwen3_5Engine,
        prompt: &[i32],
        max_new: usize,
        chunk: usize,
        stop_eos: bool,
        configured_sampler: Option<SamplerV1>,
    ) -> Result<RadixBenchMetrics> {
        let total = prompt
            .len()
            .checked_add(max_new)
            .and_then(|n| n.checked_add(self.k + 1))
            .ok_or_else(|| loader_err("speculative length overflow"))?;
        if prompt.is_empty()
            || total > self.capacity
            || prompt
                .iter()
                .any(|&id| id < 0 || id as usize >= engine.cfg.vocab_size)
        {
            return Err(loader_err(
                "invalid speculative prompt or insufficient KV capacity",
            ));
        }
        let lookup = engine.radix.match_prefix(&prompt[..prompt.len() - 1]);
        let hit = if self.can_restore(&prompt[..lookup.matched_tokens]) {
            lookup
        } else {
            let mut cold = engine.radix.match_prefix(&[]);
            cold.mamba_branching_seqlen = lookup.mamba_branching_seqlen;
            cold
        };
        let cached = hit.matched_tokens;
        self.restore(engine, &prompt[..cached])?;
        let (active, pp) = engine
            .alloc_three_state_slots()
            .ok_or_else(|| loader_err("no speculative request state slots"))?;
        let mut req = ReqRadixState::new(16, 64, active, pp);
        req.restore_prefix_hit(&mut engine.radix, prompt, &hit);
        let mut tx = StateTx::new(active);
        let processed_sampling = configured_sampler.is_some();
        let mut sampler = configured_sampler.unwrap_or_else(|| {
            SamplerV1::new(
                SamplingParams {
                    stop_token_ids: engine.cfg.eos_token_ids.clone(),
                    eos_token_ids: engine.cfg.eos_token_ids.clone(),
                    ..SamplingParams::default()
                },
                prompt,
                None,
            )
        });
        let started = Instant::now();
        let result = (|| -> Result<(Vec<i32>, Vec<crate::parser::StepLogprobs>, f64, f64)> {
            engine.ensure_kv_pages_for_len(&mut req, total)?;
            let mut anchor = 0;
            let mut cursor = cached;
            while cursor < prompt.len() {
                let rem = prompt.len() - cursor;
                let mamba_chunk = req.mamba_chunk_size.max(1);
                let mut max_take = rem.min(chunk.max(1).min(self.max_rows));
                if let Some(branch_end) = hit.mamba_branching_seqlen {
                    let aligned_branch = (branch_end / mamba_chunk) * mamba_chunk;
                    if aligned_branch > cursor && aligned_branch < prompt.len() {
                        max_take = max_take.min(aligned_branch - cursor);
                    }
                }
                let take = if cursor + max_take < prompt.len() {
                    let next_aligned = ((cursor + max_take) / mamba_chunk) * mamba_chunk;
                    if next_aligned > cursor {
                        next_aligned - cursor
                    } else {
                        max_take
                    }
                } else {
                    max_take
                };
                let end = cursor + take;
                let tokens = &prompt[cursor..end];
                let planned = req.plan_checkpoints(tokens.len());
                let pairs: Vec<_> = planned
                    .iter()
                    .map(|p| (p.trigger_offset_1based, p.dst_slot))
                    .collect();
                let out = engine.step_batch_slices(&[EngineBatchSlice {
                    tokens,
                    ctx_len_before: cursor,
                    block_table: &req.kv_pages,
                    state_in_slot: req.current_state_in_slot(),
                    state_out_slot: req.active_state_slot,
                    has_initial_state: req.has_initial_state,
                    spec_slots: &[],
                    checkpoints: &pairs,
                    emit_all_query_tokens: false,
                }])?;
                anchor = out[0][0];
                let features = self.captured(engine)?;
                self.ingest(engine, tokens, &features)?;
                req.commit_tokens(tokens, &planned);
                cursor = end;
                if (cursor < prompt.len() || max_new > 1)
                    && req.last_track_seqlen.is_some()
                    && let Some(rep) = engine.alloc_state_slot()
                {
                    let don = req.donate_committed_checkpoint(&mut engine.radix, Some(rep), false);
                    engine.free_kv_pages(&don.freed_kv_pages);
                    engine.free_state_slots(&don.freed_mamba_slots);
                }
            }
            tx.committed_slot = req.active_state_slot;
            tx.commit_prefill(prompt.len());
            let ttft = started.elapsed().as_secs_f64() * 1000.;
            let decoding = Instant::now();
            let mut generated = Vec::with_capacity(max_new);
            let mut token_logprobs = Vec::new();
            if max_new > 0 {
                if processed_sampling {
                    let (tok, lp) = engine.sample_single_gpu_step(&mut sampler)?;
                    anchor = tok;
                    if let Some(lp) = lp {
                        token_logprobs.push(lp);
                    }
                } else {
                    sampler.commit_token(anchor);
                }
                generated.push(anchor);
            }
            let _decode_range = crate::profiling::Range::new(c"w34.decode");
            let mut pending_ingest: Vec<i32> = Vec::new();
            while generated.len() < max_new
                && !generation_stopped(&sampler, anchor, stop_eos, &engine.cfg.eos_token_ids)
            {
                let k = self.k.min(max_new - generated.len() - 1);
                let proposer_range = crate::profiling::Range::new(c"w34.proposer");
                let mut drafts = if pending_ingest.is_empty() {
                    self.propose(engine, anchor, k)?
                } else {
                    let features = self.captured(engine)?;
                    let out =
                        self.ingest_and_propose(engine, &pending_ingest, &features, anchor, k)?;
                    pending_ingest.clear();
                    out
                };
                drop(proposer_range);
                if stop_eos
                    && let Some(eos) = drafts.iter().enumerate().position(|(idx, id)| {
                        let pos_before = generated.len() + idx;
                        if pos_before < sampler.params.min_tokens {
                            sampler.is_token_soft_banned(*id, &engine.cfg.eos_token_ids)
                        } else {
                            sampler
                                .params
                                .is_stop_or_eos(*id, &engine.cfg.eos_token_ids)
                        }
                    })
                {
                    drafts.truncate(eos);
                }
                if processed_sampling {
                    let mut hypothetical = sampler.clone();
                    let mut valid = 0;
                    for &id in &drafts {
                        if hypothetical.materialize_pending_constraint().is_err()
                            || hypothetical
                                .constraint
                                .as_ref()
                                .is_some_and(|c| c.is_terminal_eos() || !c.is_token_allowed(id))
                        {
                            break;
                        }
                        hypothetical
                            .try_commit_token(id)
                            .map_err(|_| loader_err("draft grammar commit failed"))?;
                        if hypothetical
                            .constraint
                            .as_ref()
                            .is_some_and(|c| c.is_terminal_eos())
                        {
                            // A terminal EOS draft closes the grammar (`mask.fill(0)`); omit it from
                            // `drafts` so the bonus row itself samples/verifies that position with a
                            // non-empty bitmask rather than evaluating an empty post-EOS bonus row.
                            break;
                        }
                        valid += 1;
                    }
                    drafts.truncate(valid);
                }
                let k = drafts.len();
                let mut verify = vec![anchor];
                verify.extend_from_slice(&drafts);
                let mut extras = Vec::with_capacity(k);
                for _ in 0..k {
                    match engine.alloc_state_slot() {
                        Some(s) => extras.push(s),
                        None => {
                            engine.free_state_slots(&extras);
                            return Err(loader_err("no speculative verification state slots"));
                        }
                    }
                }
                let schedule = if k > 0 {
                    tx.begin_spec_verify(extras);
                    tx.verify_slot_schedule()
                } else {
                    vec![tx.committed_slot]
                };
                let before = req.token_ids.len();
                let planned: Vec<PlannedCheckpoint> = req.plan_checkpoints(verify.len());
                let pairs: Vec<_> = planned
                    .iter()
                    .map(|p| (p.trigger_offset_1based, p.dst_slot))
                    .collect();
                let out = engine.step_batch_slices(&[EngineBatchSlice {
                    tokens: &verify,
                    ctx_len_before: before,
                    block_table: &req.kv_pages,
                    state_in_slot: tx.committed_slot,
                    state_out_slot: *schedule.last().unwrap(),
                    has_initial_state: true,
                    spec_slots: &schedule,
                    checkpoints: &pairs,
                    emit_all_query_tokens: true,
                }])?;
                let (decision, round_lps) = if processed_sampling {
                    engine.verify_speculative_gpu_step(&sampler, &drafts)?
                } else {
                    (
                        RejectionSampler::greedy(&drafts, &out[0])
                            .map_err(|_| loader_err("greedy rejection geometry"))?,
                        Vec::new(),
                    )
                };
                let processed = decision.processed_rows();
                if k > 0 {
                    let freed = decision
                        .commit(&mut tx, &mut sampler, 0)
                        .map_err(|_| loader_err("speculative sampler commit"))?;
                    engine.free_state_slots(&freed);
                } else {
                    tx.kv_seq_len_before += 1;
                    sampler
                        .try_commit_token(decision.emitted_tokens()[0])
                        .map_err(|_| loader_err("sampler/grammar commit failed"))?;
                }
                req.active_state_slot = tx.committed_slot;
                req.commit_tokens(&verify[..processed], &planned);
                pending_ingest.extend_from_slice(&verify[..processed]);
                self.last_accepted.push(decision.accepted_drafts());
                generated.extend_from_slice(decision.emitted_tokens());
                token_logprobs.extend(round_lps);
                anchor = *generated.last().unwrap();
            }
            if !pending_ingest.is_empty() {
                let features = self.captured(engine)?;
                self.ingest(engine, &pending_ingest, &features)?;
            }
            debug_assert_eq!(sampler.output_history, generated);
            Ok((
                generated,
                token_logprobs,
                ttft,
                decoding.elapsed().as_secs_f64() * 1000.,
            ))
        })();
        let (generated, token_logprobs, ttft, decode) = match result {
            Ok(v) => v,
            Err(e) => {
                if !tx.spec_slots.is_empty() {
                    let freed = tx.resolve_spec_verify(1);
                    engine.free_state_slots(&freed);
                }
                req.active_state_slot = tx.committed_slot;
                let release = req.abort_and_release(&mut engine.radix);
                engine.free_kv_pages(&release.freed_kv_pages);
                engine.free_state_slots(&release.freed_mamba_slots);
                self.current = None;
                if matches!(self.ranks, Ranks::DFlash(_)) {
                    self.memos.clear();
                }
                return Err(e);
            }
        };
        let donation = req.donate_committed_checkpoint(&mut engine.radix, None, true);
        engine.free_kv_pages(&donation.freed_kv_pages);
        engine.free_state_slots(&donation.freed_mamba_slots);
        self.commit_current_memo();
        let steps = generated.len().saturating_sub(1);
        Ok(RadixBenchMetrics {
            prompt_len: prompt.len(),
            cached_prefix_tokens: cached,
            donated_seq_len: donation.donated_seq_len,
            gen_len: generated.len(),
            ttft_ms: ttft,
            prefill_tok_per_s: if ttft > 0. {
                (prompt.len() - cached) as f64 * 1000. / ttft
            } else {
                0.
            },
            decode_ms: decode,
            decode_tok_per_s: if decode > 0. {
                steps as f64 * 1000. / decode
            } else {
                0.
            },
            itl_ms: if steps > 0 { decode / steps as f64 } else { 0. },
            generated_token_ids: generated,
            token_logprobs,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sampling_stops_still_respect_min_tokens_and_stop_eos() {
        for constraint in [
            None,
            Some(TokenBitmaskConstraint::from_allowed_tokens(8, vec![4, 7])),
        ] {
            let mut sampler = SamplerV1::new(
                SamplingParams {
                    max_tokens: 4,
                    min_tokens: 2,
                    stop_token_ids: vec![4],
                    ..SamplingParams::default()
                },
                &[1],
                constraint,
            );
            sampler.try_commit_token(4).unwrap();
            assert!(!generation_stopped(&sampler, 4, true, &[7]));
            assert!(!generation_stopped(&sampler, 7, true, &[7]));
            sampler.try_commit_token(4).unwrap();
            for anchor in [4, 7] {
                assert!(generation_stopped(&sampler, anchor, true, &[7]));
                assert!(!generation_stopped(&sampler, anchor, false, &[7]));
            }
            assert!(!generation_stopped(&sampler, 3, true, &[7]));
        }
    }

    #[test]
    fn sampled_terminal_grammar_stops_independently() {
        for min_tokens in [0, 3] {
            for stop_eos in [false, true] {
                let constraint = TokenBitmaskConstraint::from_allowed_tokens(8, vec![4])
                    .with_terminal_eos_ids(vec![4]);
                let mut sampler = SamplerV1::new(
                    SamplingParams {
                        max_tokens: 4,
                        min_tokens,
                        ..SamplingParams::default()
                    },
                    &[1],
                    Some(constraint),
                );
                assert!(!generation_stopped(&sampler, 1, stop_eos, &[7]));
                sampler.try_commit_token(4).unwrap();
                assert!(sampler.constraint.as_ref().unwrap().is_terminal_eos());
                assert!(
                    generation_stopped(&sampler, 4, stop_eos, &[7]),
                    "terminal grammar must stop before another sample, min_tokens={min_tokens}, stop_eos={stop_eos}"
                );
            }
        }
    }
}
