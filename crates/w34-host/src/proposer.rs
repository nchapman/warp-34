//! Native proposer orchestration and transactional greedy generation.
//! Warm target radix hits are used only when the matching committed proposer
//! features are available. A bounded, single-branch host memo rematerializes
//! draft KV without rerunning target layers; it is not a second target cache.
use crate::{
    cache::StateTx,
    cuda::{DeviceBuffer, Result, check_rc, f16_to_f32},
    dflash::{DFlash2Rank, DraftCandidates},
    loader::resolve_hf_model_dir,
    model::{EngineBatchSlice, Qwen3_5Engine, RadixBenchMetrics, loader_err},
    mtp::{MAX_MTP_BATCH_SLOTS, Qwen3_5MtpRank},
    parser::{SamplerV1, SamplingParams, TokenBitmaskConstraint},
    radix::{PlannedCheckpoint, ReqRadixState},
    speculative::RejectionSampler,
};
use std::time::Instant;
use w34_kernels::ffi;

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
}
pub struct NativeProposer {
    ranks: Ranks,
    capacity: usize,
    total_capacity: usize,
    max_rows: usize,
    row_bytes: usize,
    k: usize,
    tail: Vec<DeviceBuffer>,
    shifted: Vec<DeviceBuffer>,
    memo: Option<PrefixMemo>,
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
const MEMO_BUDGET: usize = 64 * 1024 * 1024;
impl NativeProposer {
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
        let max_rows = capacity.min(256);
        if max_rows < 16 {
            return Err(loader_err("insufficient speculative context capacity"));
        }
        let h = engine.cfg.hidden_size;
        let mtp_row_bytes = engine.cfg.hyper_hidden_size() * 2;
        let use_graph = engine.use_cuda_graph;
        let (ranks, taps, k, row_bytes) = match config.kind {
            SpeculativeKind::Mtp => {
                let ranks = engine
                    .ranks
                    .iter()
                    .map(|t| {
                        let mut r = Qwen3_5MtpRank::load(t, &engine.model_dir, capacity)?;
                        r.use_cuda_graph = use_graph && r.supports_cuda_graph();
                        Ok(r)
                    })
                    .collect::<Result<Vec<_>>>()?;
                (Ranks::Mtp(ranks), vec![], config.mtp_k, mtp_row_bytes)
            }
            SpeculativeKind::DFlash2 => {
                let name = config
                    .draft_model
                    .as_ref()
                    .ok_or_else(|| loader_err("DFlash2 requires --draft-model"))?;
                let dir = resolve_hf_model_dir(name)
                    .map_err(|_| loader_err("resolve DFlash2 checkpoint"))?;
                let ranks = engine
                    .ranks
                    .iter()
                    .map(|t| {
                        let mut r = DFlash2Rank::load(t, &dir, capacity, max_rows)?;
                        r.use_cuda_graph = use_graph;
                        Ok(r)
                    })
                    .collect::<Result<Vec<_>>>()?;
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
            .map(|t| DeviceBuffer::alloc(t.device.device_id, max_rows * row_bytes))
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
            tail,
            shifted,
            memo: None,
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
        self.resident_tokens.clear();
        self.processed = 0;
        self.last_accepted.clear();
        self.last_restored_tokens = 0;
        Ok(())
    }
    fn can_restore(&self, tokens: &[i32]) -> bool {
        if tokens.is_empty() {
            return true;
        }
        let has_memo = self
            .memo
            .as_ref()
            .is_some_and(|m| m.tokens.starts_with(tokens));
        match &self.ranks {
            Ranks::DFlash(_) => has_memo && self.resident_tokens.starts_with(tokens),
            Ranks::Mtp(_) => has_memo,
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
        // Fast path: if the drafter's resident GPU KV cache already starts with `tokens`,
        // truncate the drafter KV cache in-place and restore MTP's 1-row tail without
        // re-running drafter prefill over the cached prefix.
        if self.resident_tokens.starts_with(tokens)
            && let Some(memo) = &self.memo
            && memo.tokens.starts_with(tokens)
        {
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
            self.current = Some(PrefixMemo {
                tokens: tokens.to_vec(),
                features,
            });
            return Ok(());
        }
        self.reset()?;
        self.current = Some(PrefixMemo {
            tokens: vec![],
            features: vec![],
        });
        let memo = self
            .memo
            .take()
            .ok_or_else(|| loader_err("missing proposer prefix"))?;
        let result = (|| {
            for start in (0..tokens.len()).step_by(self.max_rows) {
                let end = (start + self.max_rows).min(tokens.len());
                let bytes = &memo.features[start * self.row_bytes..end * self.row_bytes];
                let mut pointers = Vec::with_capacity(engine.tp_size);
                for (t, stage) in engine.ranks.iter().zip(&self.shifted) {
                    stage.copy_from_host(bytes, &t.stream)?;
                    pointers.push(stage.addr());
                }
                self.ingest(engine, &tokens[start..end], &pointers)?;
            }
            Ok(())
        })();
        self.memo = Some(memo);
        result?;
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
                            r.ingest_kv_only(t, ids, input, comm)?;
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
                let candidates = if use_graph && rs.iter().all(|r| r.has_forward_graph()) {
                    let enqueue_res: Result<()> = (|| {
                        for ((r, t), &ptr) in rs.iter_mut().zip(&engine.ranks).zip(features) {
                            r.ingest_no_sync(t, ptr, n)?;
                            r.launch_forward_graph_async(t, anchor)?;
                        }
                        Ok(())
                    })();
                    match enqueue_res {
                        Ok(()) => rs
                            .iter_mut()
                            .zip(&engine.ranks)
                            .map(|(r, t)| r.collect_forward_topk_host(t, anchor))
                            .collect::<Result<Vec<_>>>(),
                        Err(e) => Err(e),
                    }
                } else {
                    std::thread::scope(|scope| {
                        let jobs: Vec<_> = rs
                            .iter_mut()
                            .zip(&mut engine.ranks)
                            .zip(features)
                            .map(|((r, t), &ptr)| {
                                scope.spawn(move || -> Result<DraftCandidates> {
                                    r.ingest_no_sync(t, ptr, n)?;
                                    r.forward(t, anchor, comm)
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
                let candidates = match candidates {
                    Ok(c) => c,
                    Err(e) => {
                        for t in &engine.ranks {
                            let _ = t.device.make_current();
                            let _ = t.stream.synchronize();
                        }
                        return Err(e);
                    }
                };
                self.resident_tokens.extend_from_slice(tokens);
                self.processed += tokens.len();
                let merged = DraftCandidates::merge(&candidates)?;
                let mut ids = rs[0].select_greedy(&engine.ranks[0], anchor, &merged)?;
                ids.truncate(k);
                Ok(ids)
            }
            Ranks::Mtp(rs) => {
                if self.current.is_none()
                    && engine.use_cuda_graph
                    && rs.iter().all(Qwen3_5MtpRank::supports_cuda_graph)
                {
                    use crate::coordinated::{ReplayPhase, replay_ranks};
                    let row_bytes = self.row_bytes;
                    let before = self.processed;
                    let comm = engine.tp2_comm.as_deref();
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
                    if let Err(error) = replay_res {
                        if error.partial_launch {
                            eprintln!(
                                "[w34] fatal: asymmetric MTP TP graph launch failure: {}",
                                error.source
                            );
                            std::process::abort();
                        }
                        for (r, t) in rs.iter_mut().zip(&engine.ranks) {
                            let _ = t.device.make_current();
                            let _ = t.stream.synchronize();
                            let _ = r.truncate(before.saturating_sub(1));
                        }
                        return Err(error.source);
                    }
                    let outputs = match rs
                        .iter_mut()
                        .zip(&engine.ranks)
                        .map(|(r, t)| r.collect_propose_host(t, k))
                        .collect::<Result<Vec<_>>>()
                    {
                        Ok(o) => o,
                        Err(e) => {
                            for (r, t) in rs.iter_mut().zip(&engine.ranks) {
                                let _ = t.device.make_current();
                                let _ = t.stream.synchronize();
                                let _ = r.truncate(before.saturating_sub(1));
                            }
                            return Err(e);
                        }
                    };
                    if outputs.windows(2).any(|p| p[0] != p[1]) {
                        return Err(loader_err("MTP TP token disagreement"));
                    }
                    self.resident_tokens.extend_from_slice(tokens);
                    self.processed += tokens.len();
                    return Ok(outputs.into_iter().next().unwrap());
                }
                self.ingest(engine, tokens, features)?;
                self.propose(engine, anchor, k)
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
                let candidates = if use_graph && rs.iter().all(|r| r.has_forward_graph()) {
                    for (r, t) in rs.iter_mut().zip(&engine.ranks) {
                        r.launch_forward_graph_async(t, anchor)?;
                    }
                    rs.iter_mut()
                        .zip(&engine.ranks)
                        .map(|(r, t)| r.collect_forward_topk_host(t, anchor))
                        .collect::<Result<Vec<_>>>()?
                } else {
                    std::thread::scope(|scope| {
                        let jobs: Vec<_> = rs
                            .iter_mut()
                            .zip(&mut engine.ranks)
                            .map(|(r, t)| scope.spawn(move || r.forward(t, anchor, comm)))
                            .collect();
                        jobs.into_iter()
                            .map(|j| {
                                j.join()
                                    .map_err(|_| loader_err("DFlash proposal rank panic"))?
                            })
                            .collect::<Result<Vec<_>>>()
                    })?
                };
                let merged = DraftCandidates::merge(&candidates)?;
                let mut ids = rs[0].select_greedy(&engine.ranks[0], anchor, &merged)?;
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
    #[allow(clippy::too_many_arguments)]
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
            engine.radix.match_prefix(&[])
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
                let end = (cursor + chunk.max(1).min(self.max_rows)).min(prompt.len());
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
                && !(stop_eos
                    && generated.len() >= sampler.params.min_tokens
                    && sampler
                        .params
                        .is_stop_or_eos(anchor, &engine.cfg.eos_token_ids))
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
                        if hypothetical
                            .constraint
                            .as_ref()
                            .is_some_and(|c| !c.is_token_allowed(id))
                        {
                            break;
                        }
                        hypothetical
                            .try_commit_token(id)
                            .map_err(|_| loader_err("draft grammar commit failed"))?;
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
                    self.memo = None;
                }
                return Err(e);
            }
        };
        let donation = req.donate_committed_checkpoint(&mut engine.radix, None, true);
        engine.free_kv_pages(&donation.freed_kv_pages);
        engine.free_state_slots(&donation.freed_mamba_slots);
        self.memo = self.current.take();
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

    #[inline]
    pub(crate) fn fits_capacity(&self, prompt_len: usize, max_new: usize) -> Option<usize> {
        let total = prompt_len
            .checked_add(max_new)
            .and_then(|n| n.checked_add(self.k + 1))?;
        (total <= self.capacity).then_some(total)
    }

    #[inline]
    pub(crate) fn can_restore_prefix(&self, prefix: &[i32]) -> bool {
        self.can_restore(prefix)
    }

    pub(crate) fn restore_prefix(
        &mut self,
        engine: &mut Qwen3_5Engine,
        prefix: &[i32],
    ) -> Result<()> {
        self.restore(engine, prefix)
    }

    pub(crate) fn abort_scheduled(&mut self) {
        self.current = None;
        if matches!(self.ranks, Ranks::DFlash(_)) {
            self.memo = None;
        }
    }

    pub(crate) fn finish_scheduled(
        &mut self,
        engine: &mut Qwen3_5Engine,
        pending_ingest: &mut Vec<i32>,
    ) -> Result<()> {
        if !pending_ingest.is_empty() {
            let features = self.captured(engine)?;
            self.ingest(engine, pending_ingest, &features)?;
            pending_ingest.clear();
        }
        self.memo = self.current.take();
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn step_scheduled_prefill_chunk(
        &mut self,
        engine: &mut Qwen3_5Engine,
        radix: &mut crate::radix::MambaRadixCache,
        seq: &mut crate::scheduler::SeqEntry,
        sampler: &mut SamplerV1,
        use_gpu_sampler: bool,
        chunk: usize,
    ) -> Result<Option<(i32, Option<crate::parser::StepLogprobs>)>> {
        let start = seq.prefilled_tokens;
        let end = (start + chunk.max(1).min(self.max_rows)).min(seq.prompt_tokens.len());
        let tokens = &seq.prompt_tokens[start..end];
        let planned = seq.radix_state.plan_checkpoints(tokens.len());
        let pairs: Vec<_> = planned
            .iter()
            .map(|p| (p.trigger_offset_1based, p.dst_slot))
            .collect();
        let out = engine.step_batch_slices(&[EngineBatchSlice {
            tokens,
            ctx_len_before: start,
            block_table: &seq.radix_state.kv_pages,
            state_in_slot: seq.radix_state.current_state_in_slot(),
            state_out_slot: seq.radix_state.active_state_slot,
            has_initial_state: seq.radix_state.has_initial_state,
            spec_slots: &[],
            checkpoints: &pairs,
            emit_all_query_tokens: false,
        }])?;
        let mut anchor = out[0][0];
        let features = self.captured(engine)?;
        self.ingest(engine, tokens, &features)?;
        seq.radix_state.commit_tokens(tokens, &planned);
        seq.prefilled_tokens = end;

        if end < seq.prompt_tokens.len() {
            if seq.radix_state.last_track_seqlen.is_some()
                && let Some(rep) = engine.alloc_state_slot_from(radix)
            {
                let don = seq
                    .radix_state
                    .donate_committed_checkpoint(radix, Some(rep), false);
                engine.free_kv_pages(&don.freed_kv_pages);
                engine.free_state_slots(&don.freed_mamba_slots);
            }
            return Ok(None);
        }

        if seq.max_new_tokens == 0 {
            seq.finished = true;
            return Ok(None);
        }
        let lp = if use_gpu_sampler {
            let (tok, lp) = engine.sample_single_gpu_step(sampler)?;
            anchor = tok;
            lp
        } else {
            sampler.commit_token(anchor);
            None
        };
        seq.generated_tokens.push(anchor);
        if seq.generated_tokens.len() >= seq.max_new_tokens || seq.should_stop_on_token(anchor) {
            seq.finished = true;
        }
        Ok(Some((anchor, lp)))
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn step_scheduled_verify_round(
        &mut self,
        engine: &mut Qwen3_5Engine,
        radix: &mut crate::radix::MambaRadixCache,
        seq: &mut crate::scheduler::SeqEntry,
        pending_ingest: &mut Vec<i32>,
        sampler: &mut SamplerV1,
        use_gpu_sampler: bool,
    ) -> Result<(Vec<i32>, Vec<crate::parser::StepLogprobs>)> {
        let anchor = *seq
            .generated_tokens
            .last()
            .ok_or_else(|| loader_err("missing anchor in step_scheduled_verify_round"))?;
        let mut tx = StateTx::new(seq.radix_state.active_state_slot);
        tx.commit_prefill(seq.radix_state.token_ids.len());

        let k_budget = self.k.min(
            seq.max_new_tokens
                .saturating_sub(seq.generated_tokens.len() + 1),
        );
        let mut drafts = if pending_ingest.is_empty() {
            self.propose(engine, anchor, k_budget)?
        } else {
            let features = self.captured(engine)?;
            let out =
                self.ingest_and_propose(engine, pending_ingest, &features, anchor, k_budget)?;
            pending_ingest.clear();
            out
        };
        if let Some(eos) = drafts.iter().enumerate().position(|(idx, id)| {
            let pos_before = seq.generated_tokens.len() + idx;
            if pos_before < sampler.params.min_tokens {
                sampler.is_token_soft_banned(*id, &engine.cfg.eos_token_ids)
            } else {
                sampler
                    .params
                    .is_stop_or_eos(*id, &engine.cfg.eos_token_ids)
            }
        }) {
            drafts.truncate(eos);
        }
        if use_gpu_sampler {
            let mut hypothetical = sampler.clone();
            let mut valid = 0;
            for &id in &drafts {
                if hypothetical
                    .constraint
                    .as_ref()
                    .is_some_and(|c| !c.is_token_allowed(id))
                {
                    break;
                }
                hypothetical
                    .try_commit_token(id)
                    .map_err(|_| loader_err("draft grammar commit failed"))?;
                valid += 1;
            }
            drafts.truncate(valid);
        }
        let k = drafts.len();
        let mut verify = vec![anchor];
        verify.extend_from_slice(&drafts);
        let mut extras = Vec::with_capacity(k);
        for _ in 0..k {
            match engine.alloc_state_slot_from(radix) {
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
        let before = seq.radix_state.token_ids.len();
        let planned: Vec<PlannedCheckpoint> = seq.radix_state.plan_checkpoints(verify.len());
        let pairs: Vec<_> = planned
            .iter()
            .map(|p| (p.trigger_offset_1based, p.dst_slot))
            .collect();
        let step_res = engine.step_batch_slices(&[EngineBatchSlice {
            tokens: &verify,
            ctx_len_before: before,
            block_table: &seq.radix_state.kv_pages,
            state_in_slot: tx.committed_slot,
            state_out_slot: *schedule.last().unwrap(),
            has_initial_state: seq.radix_state.has_initial_state,
            spec_slots: &schedule,
            checkpoints: &pairs,
            emit_all_query_tokens: true,
        }]);
        let out = match step_res {
            Ok(v) => v,
            Err(e) => {
                if !tx.spec_slots.is_empty() {
                    let freed = tx.resolve_spec_verify(1);
                    engine.free_state_slots(&freed);
                }
                seq.radix_state.active_state_slot = tx.committed_slot;
                return Err(e);
            }
        };
        let (decision, round_lps) = if use_gpu_sampler {
            match engine.verify_speculative_gpu_step(sampler, &drafts) {
                Ok(v) => v,
                Err(e) => {
                    if !tx.spec_slots.is_empty() {
                        let freed = tx.resolve_spec_verify(1);
                        engine.free_state_slots(&freed);
                    }
                    seq.radix_state.active_state_slot = tx.committed_slot;
                    return Err(e);
                }
            }
        } else {
            (
                match RejectionSampler::greedy(&drafts, &out[0]) {
                    Ok(decision) => decision,
                    Err(_) => {
                        if !tx.spec_slots.is_empty() {
                            engine.free_state_slots(&tx.resolve_spec_verify(1));
                        }
                        seq.radix_state.active_state_slot = tx.committed_slot;
                        return Err(loader_err("greedy rejection geometry"));
                    }
                },
                Vec::new(),
            )
        };
        let processed = decision.processed_rows();
        if k > 0 {
            let freed = match decision.commit(&mut tx, sampler, 0) {
                Ok(freed) => freed,
                Err(_) => {
                    if !tx.spec_slots.is_empty() {
                        engine.free_state_slots(&tx.resolve_spec_verify(1));
                    }
                    seq.radix_state.active_state_slot = tx.committed_slot;
                    return Err(loader_err("speculative sampler commit"));
                }
            };
            engine.free_state_slots(&freed);
        } else {
            tx.kv_seq_len_before += 1;
            sampler
                .try_commit_token(decision.emitted_tokens()[0])
                .map_err(|_| loader_err("sampler/grammar commit failed"))?;
        }
        seq.radix_state.active_state_slot = tx.committed_slot;
        seq.radix_state
            .commit_tokens(&verify[..processed], &planned);
        pending_ingest.extend_from_slice(&verify[..processed]);
        self.last_accepted.push(decision.accepted_drafts());

        let emitted = decision.emitted_tokens().to_vec();
        let mut pushed = 0usize;
        for &tok in &emitted {
            seq.generated_tokens.push(tok);
            pushed += 1;
            if seq.generated_tokens.len() >= seq.max_new_tokens || seq.should_stop_on_token(tok) {
                seq.finished = true;
                break;
            }
        }
        let lp_len = round_lps.len().min(pushed);
        self.last_verified_draft_tokens = k;
        Ok((emitted[..pushed].to_vec(), round_lps[..lp_len].to_vec()))
    }

    pub(crate) fn is_mtp(&self) -> bool {
        matches!(self.ranks, Ranks::Mtp(_))
    }

    pub(crate) fn is_mtp_batch_mode(&self) -> bool {
        match &self.ranks {
            Ranks::Mtp(rs) => rs.first().is_some_and(|r| r.is_batch_mode()),
            Ranks::DFlash(_) => false,
        }
    }

    pub(crate) fn fits_mtp_batch_slot_capacity(&self, prompt_len: usize, max_new: usize) -> bool {
        let slot_cap = match &self.ranks {
            Ranks::Mtp(rs) => rs.first().map_or(0, |r| r.batch_slot_capacity()),
            Ranks::DFlash(_) => return false,
        };
        if slot_cap < 32 {
            return false;
        }
        let need = prompt_len
            .saturating_add(max_new.max(1))
            .saturating_add(self.k);
        need <= slot_cap
    }

    pub(crate) fn set_mtp_batch_mode(&mut self, enable: bool) -> Result<()> {
        let Ranks::Mtp(rs) = &mut self.ranks else {
            return if enable {
                Err(loader_err("batched proposer slots only supported for MTP"))
            } else {
                Ok(())
            };
        };
        if rs.first().is_some_and(|r| r.is_batch_mode() == enable) {
            return Ok(());
        }
        // If exiting batch mode while a non-zero slot was active, swap slot 0's tail back in.
        if !enable && !self.mtp_batch_slots.is_empty() && self.mtp_batch_active_slot != 0 {
            let prev = self.mtp_batch_active_slot;
            std::mem::swap(&mut self.tail, &mut self.mtp_batch_slots[prev].tail);
            std::mem::swap(&mut self.tail, &mut self.mtp_batch_slots[0].tail);
        }
        for r in rs.iter_mut() {
            r.set_batch_mode(enable)?;
        }
        for slot in &mut self.mtp_batch_slots {
            slot.resident_tokens.clear();
            slot.processed = 0;
        }
        self.resident_tokens.clear();
        self.processed = 0;
        self.current = None;
        if enable {
            self.memo = None;
            self.mtp_batch_active_slot = 0;
            self.capacity = rs[0].batch_slot_capacity();
        } else {
            self.mtp_batch_active_slot = 0;
            self.capacity = self.total_capacity;
        }
        Ok(())
    }

    /// Promotes an already-active single-slot MTP sequence (`batch_mode == false`) into
    /// batch slot 0 without discarding its resident MTP KV pages (`all_pages[0..slot_pages]`),
    /// `tail`, or `pending_ingest` target hidden rows.
    pub(crate) fn promote_single_to_mtp_batch_slot0(
        &mut self,
        engine: &Qwen3_5Engine,
        pending_ingest_rows: usize,
    ) -> Result<()> {
        let Ranks::Mtp(rs) = &mut self.ranks else {
            return Err(loader_err("batched proposer slots only supported for MTP"));
        };
        if self.mtp_batch_slots.is_empty() || pending_ingest_rows > self.k + 1 {
            return Err(loader_err("invalid single-to-batch MTP promotion state"));
        }
        for r in rs.iter_mut() {
            r.promote_single_to_batch_slot0()?;
        }
        for (idx, slot) in self.mtp_batch_slots.iter_mut().enumerate() {
            if idx != 0 {
                slot.resident_tokens.clear();
                slot.processed = 0;
            }
        }
        if pending_ingest_rows > 0 {
            let h_bytes = self.row_bytes;
            for (rank_idx, t) in engine.ranks.iter().enumerate() {
                let src = t
                    .last_final_hidden_f16
                    .as_ref()
                    .ok_or_else(|| loader_err("missing target final hidden capture"))?
                    .addr();
                let dst = self.mtp_batch_slots[0].captured_verify_hidden[rank_idx].addr();
                t.device.make_current()?;
                check_rc(
                    // SAFETY: `last_final_hidden_f16` holds at least `pending_ingest_rows` rows
                    // from slot 0's last verify round, and `captured_verify_hidden` holds `(k+1)` rows.
                    unsafe {
                        ffi::w34_memcpy_d2d_async(
                            dst as *mut _,
                            src as *const _,
                            pending_ingest_rows * h_bytes,
                            t.stream.handle,
                        )
                    },
                    "promote single MTP slot 0 hidden copy",
                )?;
                t.stream.synchronize()?;
            }
        }
        self.memo = None;
        self.current = None;
        self.mtp_batch_active_slot = 0;
        self.capacity = rs[0].batch_slot_capacity();
        Ok(())
    }

    pub(crate) fn select_mtp_batch_slot(&mut self, slot: usize) -> Result<()> {
        if !self.is_mtp_batch_mode() || slot >= self.mtp_batch_slots.len() {
            return Err(loader_err("invalid MTP batch slot selection"));
        }
        let Ranks::Mtp(rs) = &mut self.ranks else {
            return Err(loader_err("MTP batch slot selection on non-MTP"));
        };
        for r in rs.iter_mut() {
            r.select_batch_slot(slot)?;
        }
        if slot == self.mtp_batch_active_slot {
            return Ok(());
        }
        let prev = self.mtp_batch_active_slot;
        std::mem::swap(&mut self.tail, &mut self.mtp_batch_slots[prev].tail);
        std::mem::swap(
            &mut self.resident_tokens,
            &mut self.mtp_batch_slots[prev].resident_tokens,
        );
        self.mtp_batch_slots[prev].processed = self.processed;

        self.mtp_batch_active_slot = slot;
        std::mem::swap(&mut self.tail, &mut self.mtp_batch_slots[slot].tail);
        std::mem::swap(
            &mut self.resident_tokens,
            &mut self.mtp_batch_slots[slot].resident_tokens,
        );
        self.processed = self.mtp_batch_slots[slot].processed;
        self.current = None;
        Ok(())
    }

    pub(crate) fn reset_mtp_batch_slot(&mut self, slot: usize) -> Result<()> {
        self.select_mtp_batch_slot(slot)?;
        if let Ranks::Mtp(rs) = &mut self.ranks {
            for r in rs.iter_mut() {
                r.truncate(0)?;
            }
        }
        self.resident_tokens.clear();
        self.processed = 0;
        self.current = None;
        Ok(())
    }

    /// Execute one batched `B = 2..=4` greedy MTP propose + shared target verification round.
    /// Each active slot runs its fast single-layer MTP draft graph (`k = 1..3`), then all
    /// `B` sequences verify together in a single target `step_batch_slices` pass (`M = sum(1 + k_b)`).
    pub(crate) fn step_scheduled_batched_mtp_verify_round(
        &mut self,
        engine: &mut Qwen3_5Engine,
        radix: &mut crate::radix::MambaRadixCache,
        items: &mut [(
            usize,
            &mut crate::scheduler::SeqEntry,
            &mut Vec<i32>,
            &mut SamplerV1,
        )],
    ) -> Result<Vec<Vec<i32>>> {
        self.last_verified_draft_tokens = 0;
        if items.is_empty() {
            return Ok(vec![]);
        }
        let h_bytes = self.row_bytes;
        struct PreparedItem {
            slot: usize,
            drafts: Vec<i32>,
            verify: Vec<i32>,
            tx: StateTx,
            schedule: Vec<i32>,
            before: usize,
            planned: Vec<PlannedCheckpoint>,
            pairs: Vec<(i32, i32)>,
            row_offset: usize,
        }
        let mut prepared = Vec::with_capacity(items.len());
        let mut row_offset = 0usize;

        let batch_len = items.len();
        let k_cap = if batch_len >= 3 {
            self.k.min(1)
        } else {
            self.k
        };

        // Pass 1: run fallible draft proposal before allocating any `StateTx` GDN slots so
        // early `?` returns cannot leak extra state slots from earlier batch entries.
        let mut proposed = Vec::with_capacity(batch_len);
        for (slot, seq, pending_ingest, sampler) in items.iter_mut() {
            let slot = *slot;
            self.select_mtp_batch_slot(slot)?;
            let anchor = *seq
                .generated_tokens
                .last()
                .ok_or_else(|| loader_err("missing anchor in batched MTP verify"))?;
            let k_budget = k_cap.min(
                seq.max_new_tokens
                    .saturating_sub(seq.generated_tokens.len() + 1),
            );
            let mut drafts = if pending_ingest.is_empty() {
                self.propose(engine, anchor, k_budget)?
            } else {
                let ptrs: Vec<u64> = self.mtp_batch_slots[slot]
                    .captured_verify_hidden
                    .iter()
                    .map(|b| b.addr())
                    .collect();
                let out =
                    self.ingest_and_propose(engine, pending_ingest, &ptrs, anchor, k_budget)?;
                pending_ingest.clear();
                out
            };
            if let Some(eos) = drafts.iter().enumerate().position(|(idx, id)| {
                let pos_before = seq.generated_tokens.len() + idx;
                if pos_before < sampler.params.min_tokens {
                    sampler.is_token_soft_banned(*id, &engine.cfg.eos_token_ids)
                } else {
                    sampler.is_token_soft_banned(*id, &engine.cfg.eos_token_ids)
                        || sampler
                            .params
                            .is_stop_or_eos(*id, &engine.cfg.eos_token_ids)
                }
            }) {
                drafts.truncate(eos);
            }
            proposed.push((slot, anchor, drafts));
        }

        // Pass 2: infallibly allocate speculative GDN state slots (`StateTx`). Every fallible
        // operation after this point runs inside `verify_and_commit` so any error triggers
        // deterministic cleanup of all unresolved `prep.tx.spec_slots`.
        for ((_, seq, _, _), (slot, anchor, mut drafts)) in items.iter_mut().zip(proposed) {
            let mut extras = Vec::with_capacity(drafts.len());
            for _ in 0..drafts.len() {
                match engine.alloc_state_slot_from(radix) {
                    Some(s) => extras.push(s),
                    None => {
                        engine.free_state_slots(&extras);
                        extras.clear();
                        drafts.clear();
                        break;
                    }
                }
            }
            let mut tx = StateTx::new(seq.radix_state.active_state_slot);
            tx.commit_prefill(seq.radix_state.token_ids.len());
            let schedule = if !drafts.is_empty() {
                tx.begin_spec_verify(extras);
                tx.verify_slot_schedule()
            } else {
                vec![tx.committed_slot]
            };
            let mut verify = Vec::with_capacity(1 + drafts.len());
            verify.push(anchor);
            verify.extend_from_slice(&drafts);
            let before = seq.radix_state.token_ids.len();
            let planned = seq.radix_state.plan_checkpoints(verify.len());
            let pairs = planned
                .iter()
                .map(|p| (p.trigger_offset_1based, p.dst_slot))
                .collect();
            let this_offset = row_offset;
            row_offset += verify.len();
            prepared.push(PreparedItem {
                slot,
                drafts,
                verify,
                tx,
                schedule,
                before,
                planned,
                pairs,
                row_offset: this_offset,
            });
        }

        let res: Result<Vec<Vec<i32>>> = (|| {
            let outs = {
                let slices: Vec<EngineBatchSlice<'_>> = items
                    .iter()
                    .zip(&prepared)
                    .map(|((_, seq, _, _), prep)| EngineBatchSlice {
                        tokens: &prep.verify,
                        ctx_len_before: prep.before,
                        block_table: &seq.radix_state.kv_pages,
                        state_in_slot: prep.tx.committed_slot,
                        state_out_slot: *prep.schedule.last().unwrap(),
                        has_initial_state: seq.radix_state.has_initial_state,
                        spec_slots: &prep.schedule,
                        checkpoints: &prep.pairs,
                        emit_all_query_tokens: true,
                    })
                    .collect();
                engine.step_batch_slices(&slices)?
            };

            let any_gpu_sampler = items.iter().any(|(_, _, _, sampler)| {
                !sampler.params.is_plain_greedy(sampler.constraint.is_some())
            });
            let decisions = if any_gpu_sampler {
                let pairs: Vec<(&SamplerV1, &[i32])> = items
                    .iter()
                    .zip(&prepared)
                    .map(|((_, _, _, sampler), prep)| (&**sampler, prep.drafts.as_slice()))
                    .collect();
                engine.verify_batched_speculative_gpu_steps(&pairs)?
            } else {
                let mut ds = Vec::with_capacity(items.len());
                for (idx, prep) in prepared.iter().enumerate() {
                    let d = RejectionSampler::greedy(&prep.drafts, &outs[idx])
                        .map_err(|_| loader_err("batched MTP greedy rejection geometry"))?;
                    ds.push(d);
                }
                ds
            };

            // Copy each slot's `processed` final-normalized target hidden rows from the unified
            // `last_final_hidden_f16` buffer into that slot's dedicated per-rank staging buffer.
            for (rank_idx, t) in engine.ranks.iter().enumerate() {
                let final_h = t
                    .last_final_hidden_f16
                    .as_ref()
                    .ok_or_else(|| loader_err("missing target final hidden capture"))?
                    .addr();
                t.device.make_current()?;
                for (prep, decision) in prepared.iter().zip(&decisions) {
                    let processed = decision.processed_rows();
                    let src = final_h + (prep.row_offset * h_bytes) as u64;
                    let dst =
                        self.mtp_batch_slots[prep.slot].captured_verify_hidden[rank_idx].addr();
                    check_rc(
                        // SAFETY: `last_final_hidden_f16` holds all `row_offset <= max_rows` rows,
                        // and `captured_verify_hidden` holds `(k + 1) * h_bytes >= processed * h_bytes`.
                        unsafe {
                            ffi::w34_memcpy_d2d_async(
                                dst as *mut _,
                                src as *const _,
                                processed * h_bytes,
                                t.stream.handle,
                            )
                        },
                        "batched MTP verify hidden copy",
                    )?;
                }
            }
            for t in &engine.ranks {
                t.device.make_current()?;
                t.stream.synchronize()?;
            }

            let mut all_emitted = Vec::with_capacity(items.len());
            for (((_, seq, pending_ingest, sampler), prep), decision) in
                items.iter_mut().zip(&mut prepared).zip(decisions)
            {
                let processed = decision.processed_rows();
                if !prep.drafts.is_empty() {
                    let freed = decision
                        .commit(&mut prep.tx, sampler, 0)
                        .map_err(|_| loader_err("batched MTP sampler commit"))?;
                    engine.free_state_slots(&freed);
                } else {
                    prep.tx.kv_seq_len_before += 1;
                    sampler
                        .try_commit_token(decision.emitted_tokens()[0])
                        .map_err(|_| loader_err("batched MTP fallback commit"))?;
                }
                seq.radix_state.active_state_slot = prep.tx.committed_slot;
                seq.radix_state
                    .commit_tokens(&prep.verify[..processed], &prep.planned);
                pending_ingest.extend_from_slice(&prep.verify[..processed]);
                self.last_accepted.push(decision.accepted_drafts());

                let emitted = decision.emitted_tokens().to_vec();
                let mut pushed = 0usize;
                for &tok in &emitted {
                    seq.generated_tokens.push(tok);
                    pushed += 1;
                    if seq.generated_tokens.len() >= seq.max_new_tokens
                        || seq.should_stop_on_token(tok)
                    {
                        seq.finished = true;
                        break;
                    }
                }
                all_emitted.push(emitted[..pushed].to_vec());
            }
            Ok(all_emitted)
        })();

        if res.is_ok() {
            self.last_verified_draft_tokens = prepared.iter().map(|prep| prep.drafts.len()).sum();
        }
        if res.is_err() {
            for ((_, seq, _, _), prep) in items.iter_mut().zip(&mut prepared) {
                if !prep.tx.spec_slots.is_empty() {
                    let freed = prep.tx.resolve_spec_verify(1);
                    engine.free_state_slots(&freed);
                }
                seq.radix_state.active_state_slot = prep.tx.committed_slot;
            }
        }
        res
    }
}
