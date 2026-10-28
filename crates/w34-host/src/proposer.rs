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
    mtp::Qwen3_5MtpRank,
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
pub struct NativeProposer {
    ranks: Ranks,
    capacity: usize,
    max_rows: usize,
    row_bytes: usize,
    k: usize,
    tail: Vec<DeviceBuffer>,
    shifted: Vec<DeviceBuffer>,
    memo: Option<PrefixMemo>,
    current: Option<PrefixMemo>,
    resident_tokens: Vec<i32>,
    processed: usize,
    pub last_accepted: Vec<usize>,
    pub last_restored_tokens: usize,
}
const MEMO_BUDGET: usize = 64 * 1024 * 1024;
impl NativeProposer {
    pub fn load(engine: &mut Qwen3_5Engine, config: &SpeculativeConfig) -> Result<Self> {
        if engine.ranks.iter().any(|t| t.decode_graph.is_some()) {
            return Err(loader_err("enable speculation before graph capture"));
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
        let (ranks, taps, k, row_bytes) = match config.kind {
            SpeculativeKind::Mtp => {
                let ranks = engine
                    .ranks
                    .iter()
                    .map(|t| Qwen3_5MtpRank::load(t, &engine.model_dir, capacity))
                    .collect::<Result<Vec<_>>>()?;
                (Ranks::Mtp(ranks), vec![], config.mtp_k, h * 2)
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
                    .map(|t| DFlash2Rank::load(t, &dir, capacity, max_rows))
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
            .map(|t| DeviceBuffer::alloc(t.device.device_id, h * 2))
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
        Ok(Self {
            ranks,
            capacity,
            max_rows,
            row_bytes,
            k,
            tail,
            shifted,
            memo: None,
            current: None,
            resident_tokens: vec![],
            processed: 0,
            last_accepted: vec![],
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
        tokens.is_empty()
            || self
                .memo
                .as_ref()
                .is_some_and(|m| m.tokens.starts_with(tokens))
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
            self.current = Some(PrefixMemo {
                tokens: tokens.to_vec(),
                features: memo.features[..n * self.row_bytes].to_vec(),
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
        // Host memo holds committed features only (single copy from rank 0 since
        // target taps and final hidden states are replicated across TP ranks).
        if let Some(memo) = &self.current
            && (memo.tokens.len() + tokens.len()) * self.row_bytes > MEMO_BUDGET
        {
            self.current = None;
        }
        if let Some(memo) = &mut self.current {
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
                let h = engine.cfg.hidden_size;
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
                                    // SAFETY: both resident buffers hold at least H FP16 elements.
                                    unsafe {
                                        ffi::w34_memcpy_d2d_async(
                                            shifted.addr() as *mut _,
                                            tail.addr() as *const _,
                                            h * 2,
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
                                                (shifted.addr() + (h * 2) as u64) as *mut _,
                                                ptr as *const _,
                                                (tokens.len() - 1) * h * 2,
                                                t.stream.handle,
                                            )
                                        },
                                        "MTP shifted hidden",
                                    )?;
                                }
                                (tokens, shifted.addr())
                            };
                            r.ingest(t, ids, input, comm)?;
                            check_rc(
                                // SAFETY: tokens is nonempty and ptr spans every captured row;
                                // tail owns exactly one H-element row and survives synchronization.
                                unsafe {
                                    ffi::w34_memcpy_d2d_async(
                                        tail.addr() as *mut _,
                                        (ptr + ((tokens.len() - 1) * h * 2) as u64) as *const _,
                                        h * 2,
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
    fn propose(&mut self, engine: &mut Qwen3_5Engine, anchor: i32, k: usize) -> Result<Vec<i32>> {
        if k == 0 {
            return Ok(vec![]);
        }
        let comm = engine.tp2_comm.as_deref();
        match &mut self.ranks {
            Ranks::Mtp(rs) => {
                let outputs = std::thread::scope(|scope| {
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
                })?;
                if outputs.windows(2).any(|p| p[0] != p[1]) {
                    return Err(loader_err("MTP TP token disagreement"));
                }
                Ok(outputs[0].clone())
            }
            Ranks::DFlash(rs) => {
                let candidates = std::thread::scope(|scope| {
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
                })?;
                let merged = DraftCandidates::merge(&candidates)?;
                let mut ids = rs[0].select_greedy(&engine.ranks[0], anchor, &merged)?;
                ids.truncate(k);
                Ok(ids)
            }
        }
    }
    fn logits(engine: &Qwen3_5Engine, rows: usize) -> Result<Vec<Vec<f32>>> {
        let mut full = vec![vec![0f32; engine.cfg.vocab_size]; rows];
        for t in &engine.ranks {
            if t.speculative_logits_rows != rows {
                return Err(loader_err("selected logit rows mismatch"));
            }
            let mut bits = vec![0u16; rows * t.local_vocab_size];
            t.speculative_logits
                .as_ref()
                .ok_or_else(|| loader_err("missing selected logits"))?
                .copy_to_host(&mut bits, &t.stream)?;
            for (row, values) in bits.chunks_exact(t.local_vocab_size).enumerate() {
                for (col, &value) in values.iter().enumerate() {
                    let id = t.vocab_offset + col;
                    if id < engine.cfg.vocab_size {
                        full[row][id] = crate::cuda::f16_to_f32(value);
                    }
                }
            }
        }
        Ok(full)
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
                let mut bits = vec![0u16; h];
                rs[0].hidden.copy_to_host(&mut bits, &t0.stream)?;
                let vals: Vec<f32> = bits.into_iter().map(f16_to_f32).collect();
                dump.tensors_f32
                    .insert("drafter_hidden".to_string(), (vec![1, h], vals));
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
        let count = params.max_tokens;
        let sampler = SamplerV1::new(params, prompt, constraint);
        sampler
            .processed_distribution(&vec![0.0; engine.cfg.vocab_size])
            .map_err(|_| loader_err("invalid sampling parameters or grammar"))?;
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
        let mut sampler = configured_sampler
            .unwrap_or_else(|| SamplerV1::new(SamplingParams::default(), prompt, None));
        let started = Instant::now();
        let result = (|| -> Result<(Vec<i32>, f64, f64)> {
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
            if max_new > 0 {
                if processed_sampling {
                    let logits = Self::logits(engine, 1)?;
                    sampler
                        .processed_distribution(&logits[0])
                        .map_err(|_| loader_err("invalid initial target distribution"))?;
                    anchor = sampler.sample_from_logits(&logits[0]);
                } else {
                    sampler.commit_token(anchor);
                }
                generated.push(anchor);
            }
            while generated.len() < max_new
                && !(stop_eos
                    && (engine.cfg.eos_token_ids.contains(&anchor)
                        || sampler.params.stop_token_ids.contains(&anchor)))
            {
                let k = self.k.min(max_new - generated.len() - 1);
                let mut drafts = self.propose(engine, anchor, k)?;
                if stop_eos
                    && let Some(eos) = drafts.iter().position(|id| {
                        engine.cfg.eos_token_ids.contains(id)
                            || sampler.params.stop_token_ids.contains(id)
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
                        hypothetical.commit_token(id);
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
                let decision =
                    if processed_sampling {
                        let logits = Self::logits(engine, k + 1)?;
                        let mut hypothetical = sampler.clone();
                        let mut target = Vec::with_capacity(k + 1);
                        let mut proposal = Vec::with_capacity(k);
                        for (row, values) in logits.iter().enumerate() {
                            target.push(hypothetical.processed_distribution(values).map_err(
                                |_| loader_err("invalid processed target distribution"),
                            )?);
                            if row < k {
                                let mut q = vec![0f32; engine.cfg.vocab_size];
                                q[drafts[row] as usize] = 1.0;
                                proposal.push(q);
                                hypothetical.commit_token(drafts[row]);
                            }
                        }
                        let pos = sampler.output_history.len();
                        let draws: Vec<f64> = (0..k)
                            .map(|i| {
                                f64::from(SamplerV1::uniform_f32(
                                    sampler.params.seed ^ 0xa0761d6478bd642f,
                                    pos + i,
                                ))
                            })
                            .collect();
                        let recovery = f64::from(SamplerV1::uniform_f32(
                            sampler.params.seed ^ 0xe7037ed1a0b428db,
                            pos,
                        ));
                        RejectionSampler::stochastic(&drafts, &target, &proposal, &draws, recovery)
                            .map_err(|_| loader_err("stochastic rejection failed"))?
                    } else {
                        RejectionSampler::greedy(&drafts, &out[0])
                            .map_err(|_| loader_err("greedy rejection geometry"))?
                    };
                let processed = decision.processed_rows();
                if k > 0 {
                    let freed = decision
                        .commit(&mut tx, &mut sampler, 0)
                        .map_err(|_| loader_err("speculative sampler commit"))?;
                    engine.free_state_slots(&freed);
                } else {
                    tx.kv_seq_len_before += 1;
                    sampler.commit_token(decision.emitted_tokens()[0]);
                }
                req.active_state_slot = tx.committed_slot;
                req.commit_tokens(&verify[..processed], &planned);
                let features = self.captured(engine)?;
                self.ingest(engine, &verify[..processed], &features)?;
                self.last_accepted.push(decision.accepted_drafts());
                generated.extend_from_slice(decision.emitted_tokens());
                anchor = *generated.last().unwrap();
            }
            debug_assert_eq!(sampler.output_history, generated);
            Ok((generated, ttft, decoding.elapsed().as_secs_f64() * 1000.))
        })();
        let (generated, ttft, decode) = match result {
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
        })
    }
}
