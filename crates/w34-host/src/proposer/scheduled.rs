use w34_kernels::ffi;

use crate::{
    cache::StateTx,
    cuda::{Result, check_rc},
    model::{EngineBatchSlice, Qwen3_5Engine, loader_err},
    parser::SamplerV1,
    radix::PlannedCheckpoint,
    speculative::RejectionSampler,
};

use super::{NativeProposer, Ranks};

impl NativeProposer {
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

    pub(crate) fn can_restore_batch_slot(&self, _slot: usize, prefix: &[i32]) -> bool {
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
        self.commit_current_memo();
        let _ = self.reset();
        self.current = None;
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
        self.commit_current_memo();
        Ok(())
    }

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
        let rem = seq.prompt_tokens.len() - start;
        let mamba_chunk = seq.radix_state.mamba_chunk_size.max(1);
        let mut max_take = rem.min(chunk.max(1).min(self.max_rows));
        if let Some(branch_end) = seq.mamba_branching_seqlen {
            let aligned_branch = (branch_end / mamba_chunk) * mamba_chunk;
            if aligned_branch > start && aligned_branch < seq.prompt_tokens.len() {
                max_take = max_take.min(aligned_branch - start);
            }
        }
        let take = if start + max_take < seq.prompt_tokens.len() {
            let next_aligned = ((start + max_take) / mamba_chunk) * mamba_chunk;
            if next_aligned > start {
                next_aligned - start
            } else {
                max_take
            }
        } else {
            max_take
        };
        let end = start + take;
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
            if seq.radix_state.last_track_seqlen.is_some() {
                let rep = if start < 2048 {
                    engine.alloc_state_slot_from(radix)
                } else {
                    engine.alloc_state_slot_preserving_leaves(radix)
                };
                if let Some(rep) = rep {
                    let don = seq
                        .radix_state
                        .donate_committed_checkpoint(radix, Some(rep), false);
                    engine.free_kv_pages(&don.freed_kv_pages);
                    engine.free_state_slots(&don.freed_mamba_slots);
                }
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
        if seq.generated_tokens.len() >= seq.max_new_tokens
            || seq.should_stop_on_token(anchor)
            || sampler
                .constraint
                .as_ref()
                .is_some_and(|c| c.is_terminal_eos())
        {
            seq.finished = true;
        }
        if !seq.finished
            && seq.radix_state.last_track_seqlen.is_some()
            && let Some(rep) = engine.alloc_state_slot_from(radix)
        {
            let don = seq
                .radix_state
                .donate_committed_checkpoint(radix, Some(rep), false);
            engine.free_kv_pages(&don.freed_kv_pages);
            engine.free_state_slots(&don.freed_mamba_slots);
        }
        Ok(Some((anchor, lp)))
    }

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
                    break;
                }
                valid += 1;
            }
            drafts.truncate(valid);
        }
        let mut extras = Vec::with_capacity(drafts.len());
        for _ in 0..drafts.len() {
            let Some(s) = engine.alloc_state_slot_preserving_leaves(radix) else {
                break;
            };
            extras.push(s);
        }
        drafts.truncate(extras.len());
        let k = drafts.len();
        let mut verify = vec![anchor];
        verify.extend_from_slice(&drafts);
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
        if sampler
            .constraint
            .as_ref()
            .is_some_and(|c| c.is_terminal_eos())
        {
            seq.finished = true;
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
            Ranks::DFlash(rs) => rs.first().is_some_and(|r| r.is_batch_mode()),
        }
    }

    pub(crate) fn batch_slot_count(&self) -> usize {
        match &self.ranks {
            Ranks::Mtp(rs) => rs.first().map_or(0, |r| r.batch_slot_count()),
            Ranks::DFlash(rs) => rs.first().map_or(0, |r| r.batch_slot_count()),
        }
    }

    pub(crate) fn fits_mtp_batch_slot_capacity(&self, prompt_len: usize, max_new: usize) -> bool {
        let slot_cap = match &self.ranks {
            Ranks::Mtp(rs) => rs.first().map_or(0, |r| r.batch_slot_capacity()),
            Ranks::DFlash(rs) => rs.first().map_or(0, |r| r.batch_slot_capacity()),
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
        if self.is_mtp_batch_mode() == enable {
            return Ok(());
        }
        self.commit_current_memo();
        for slot in &mut self.mtp_batch_slots {
            if let Some(memo) = slot.current.take() {
                super::push_mtp_memo(&mut self.memos, memo);
            }
        }
        // If exiting batch mode while a non-zero slot was active, swap slot 0's tail back in.
        if !enable
            && matches!(self.ranks, Ranks::Mtp(_))
            && !self.mtp_batch_slots.is_empty()
            && self.mtp_batch_active_slot != 0
        {
            let prev = self.mtp_batch_active_slot;
            std::mem::swap(&mut self.tail, &mut self.mtp_batch_slots[prev].tail);
            std::mem::swap(&mut self.tail, &mut self.mtp_batch_slots[0].tail);
        }
        let slot_cap = match &mut self.ranks {
            Ranks::Mtp(rs) => {
                for r in rs.iter_mut() {
                    r.set_batch_mode(enable)?;
                }
                rs[0].batch_slot_capacity()
            }
            Ranks::DFlash(rs) => {
                for r in rs.iter_mut() {
                    r.set_batch_mode(enable)?;
                }
                rs[0].batch_slot_capacity()
            }
        };
        for slot in &mut self.mtp_batch_slots {
            slot.resident_tokens.clear();
            slot.processed = 0;
            slot.current = None;
            slot.last_used_tick = 0;
        }
        self.resident_tokens.clear();
        self.processed = 0;
        self.current = None;
        self.mtp_batch_active_slot = 0;
        self.active_slot_last_used_tick = 0;
        self.capacity = if enable {
            slot_cap
        } else {
            self.total_capacity
        };
        Ok(())
    }

    /// Promotes an already-active single-slot proposer sequence (`batch_mode == false`) into
    /// batch slot 0 without discarding its resident draft KV pages, `tail`, `current` memo,
    /// or `pending_ingest` target hidden rows.
    pub(crate) fn promote_single_to_mtp_batch_slot0(
        &mut self,
        engine: &Qwen3_5Engine,
        pending_ingest_rows: usize,
    ) -> Result<()> {
        if self.mtp_batch_slots.is_empty() || pending_ingest_rows > self.k + 1 {
            return Err(loader_err(
                "invalid single-to-batch proposer promotion state",
            ));
        }
        let slot_cap = match &mut self.ranks {
            Ranks::Mtp(rs) => {
                for r in rs.iter_mut() {
                    r.promote_single_to_batch_slot0()?;
                }
                rs[0].batch_slot_capacity()
            }
            Ranks::DFlash(rs) => {
                for r in rs.iter_mut() {
                    r.promote_single_to_batch_slot0()?;
                }
                rs[0].batch_slot_capacity()
            }
        };
        for (idx, slot) in self.mtp_batch_slots.iter_mut().enumerate() {
            if idx != 0 {
                slot.resident_tokens.clear();
                slot.processed = 0;
                slot.current = None;
                slot.last_used_tick = 0;
            }
        }
        self.batch_slot_tick_counter += 1;
        self.active_slot_last_used_tick = self.batch_slot_tick_counter;
        if pending_ingest_rows > 0 {
            let h_bytes = self.row_bytes;
            let src_ptrs = self.captured(engine)?;
            for (rank_idx, t) in engine.ranks.iter().enumerate() {
                let src = src_ptrs[rank_idx];
                let dst = self.mtp_batch_slots[0].captured_verify_hidden[rank_idx].addr();
                t.device.make_current()?;
                check_rc(
                    // SAFETY: `src` holds at least `pending_ingest_rows` rows from slot 0's
                    // last verify round, and `captured_verify_hidden` holds `(k+1)` rows.
                    unsafe {
                        ffi::w34_memcpy_d2d_async(
                            dst as *mut _,
                            src as *const _,
                            pending_ingest_rows * h_bytes,
                            t.stream.handle,
                        )
                    },
                    "promote single proposer slot 0 hidden copy",
                )?;
                t.stream.synchronize()?;
            }
        }
        self.mtp_batch_active_slot = 0;
        self.capacity = slot_cap;
        Ok(())
    }

    pub(crate) fn select_mtp_batch_slot(&mut self, slot: usize) -> Result<()> {
        let slot_count = self.batch_slot_count();
        if !self.is_mtp_batch_mode() || slot >= slot_count || slot >= self.mtp_batch_slots.len() {
            return Err(loader_err("invalid proposer batch slot selection"));
        }
        let is_mtp = matches!(self.ranks, Ranks::Mtp(_));
        match &mut self.ranks {
            Ranks::Mtp(rs) => {
                for r in rs.iter_mut() {
                    r.select_batch_slot(slot)?;
                }
            }
            Ranks::DFlash(rs) => {
                for r in rs.iter_mut() {
                    r.select_batch_slot(slot)?;
                }
            }
        }
        if slot == self.mtp_batch_active_slot {
            return Ok(());
        }
        let prev = self.mtp_batch_active_slot;
        if is_mtp {
            std::mem::swap(&mut self.tail, &mut self.mtp_batch_slots[prev].tail);
        }
        std::mem::swap(
            &mut self.resident_tokens,
            &mut self.mtp_batch_slots[prev].resident_tokens,
        );
        std::mem::swap(&mut self.current, &mut self.mtp_batch_slots[prev].current);
        self.mtp_batch_slots[prev].processed = self.processed;
        self.mtp_batch_slots[prev].last_used_tick = self.active_slot_last_used_tick;

        self.mtp_batch_active_slot = slot;
        if is_mtp {
            std::mem::swap(&mut self.tail, &mut self.mtp_batch_slots[slot].tail);
        }
        std::mem::swap(
            &mut self.resident_tokens,
            &mut self.mtp_batch_slots[slot].resident_tokens,
        );
        std::mem::swap(&mut self.current, &mut self.mtp_batch_slots[slot].current);
        self.processed = self.mtp_batch_slots[slot].processed;
        self.batch_slot_tick_counter += 1;
        self.active_slot_last_used_tick = self.batch_slot_tick_counter;
        Ok(())
    }

    fn batch_slot_resident_tokens(&self, slot: usize) -> &[i32] {
        if slot == self.mtp_batch_active_slot {
            &self.resident_tokens
        } else {
            &self.mtp_batch_slots[slot].resident_tokens
        }
    }

    fn batch_slot_last_used_tick(&self, slot: usize) -> u64 {
        if slot == self.mtp_batch_active_slot {
            self.active_slot_last_used_tick
        } else {
            self.mtp_batch_slots[slot].last_used_tick
        }
    }

    /// Selects a free batch slot (`active[slot].is_none()`), ranking candidates in 3 tiers:
    /// - Tier 2: true continuation of this slot's resident session (`lcp >= 16` and `res.len() - lcp <= 256`)
    /// - Tier 1: unused empty slot (`res.is_empty()`) so new sessions never overwrite another
    ///   session's warm conversation slot just because they share a system prompt
    /// - Tier 0: LRU fallback when all slots hold prior sessions (`u64::MAX - last_used_tick`).
    pub(crate) fn pick_free_mtp_batch_slot<T>(
        &self,
        active: &[Option<T>],
        prompt: &[i32],
    ) -> Option<usize> {
        let slot_count = self
            .batch_slot_count()
            .min(self.mtp_batch_slots.len())
            .min(active.len());
        let mut best: Option<(usize, (u8, usize, u64))> = None;
        for (slot, entry) in active.iter().enumerate().take(slot_count) {
            if entry.is_some() {
                continue;
            }
            let res = self.batch_slot_resident_tokens(slot);
            let tick = self.batch_slot_last_used_tick(slot);
            let rank = rank_batch_slot_candidate(res, prompt, tick);
            if best.is_none_or(|(_, best_rank)| rank > best_rank) {
                best = Some((slot, rank));
            }
        }
        best.map(|(slot, _)| slot)
    }

    pub(crate) fn restore_mtp_batch_slot(
        &mut self,
        engine: &mut Qwen3_5Engine,
        slot: usize,
        prefix: &[i32],
    ) -> Result<()> {
        self.select_mtp_batch_slot(slot)?;
        self.restore(engine, prefix)
    }

    /// Flushes any trailing target-verified tokens from `captured_verify_hidden[slot]` into
    /// the slot's draft KV cache + `PrefixMemo`, commits the memo for future multi-turn reuse,
    /// and leaves the slot's GPU draft KV cache warm for `pick_free_mtp_batch_slot`.
    pub(crate) fn finish_mtp_batch_slot(
        &mut self,
        engine: &mut Qwen3_5Engine,
        slot: usize,
        pending_ingest: &mut Vec<i32>,
    ) -> Result<()> {
        self.select_mtp_batch_slot(slot)?;
        if !pending_ingest.is_empty() {
            let ptrs: Vec<u64> = self.mtp_batch_slots[slot]
                .captured_verify_hidden
                .iter()
                .map(|b| b.addr())
                .collect();
            self.ingest(engine, pending_ingest, &ptrs)?;
            pending_ingest.clear();
        }
        self.commit_current_memo();
        Ok(())
    }

    pub(crate) fn reset_mtp_batch_slot(&mut self, slot: usize) -> Result<()> {
        self.select_mtp_batch_slot(slot)?;
        self.commit_current_memo();
        match &mut self.ranks {
            Ranks::Mtp(rs) => {
                for r in rs.iter_mut() {
                    r.truncate(0)?;
                }
            }
            Ranks::DFlash(rs) => {
                for r in rs.iter_mut() {
                    r.truncate(0)?;
                }
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
        let k_cap = match &self.ranks {
            Ranks::Mtp(_) => {
                if batch_len >= 5 {
                    self.k.min(1)
                } else if batch_len >= 3 {
                    if self.max_rows > 256 {
                        self.k.min(2)
                    } else {
                        self.k.min(1)
                    }
                } else {
                    self.k
                }
            }
            Ranks::DFlash(_) => {
                if batch_len >= 3 {
                    self.k.min(5)
                } else {
                    self.k
                }
            }
        };

        // Pre-allocate speculative GDN state slots round-robin across active items (preserving
        // waiting session leaf nodes in `radix`) so every slot gets a fair share of speculative
        // verify capacity when GDN state slots are contended on TP=1.
        let raw_budgets: Vec<usize> = items
            .iter()
            .map(|(_, seq, _, _)| {
                k_cap.min(
                    seq.max_new_tokens
                        .saturating_sub(seq.generated_tokens.len() + 1),
                )
            })
            .collect();
        let max_budget = raw_budgets.iter().copied().max().unwrap_or(0);
        let mut all_extras: Vec<Vec<i32>> =
            raw_budgets.iter().map(|&b| Vec::with_capacity(b)).collect();
        'outer: for round in 0..max_budget {
            for (idx, &budget) in raw_budgets.iter().enumerate() {
                if round < budget {
                    let Some(s) = engine.alloc_state_slot_preserving_leaves(radix) else {
                        break 'outer;
                    };
                    all_extras[idx].push(s);
                }
            }
        }

        // Pass 1: run proposer draft forwards for each active item using its reserved state slots.
        let mut proposed: Vec<(usize, i32, Vec<i32>, Vec<i32>)> = Vec::with_capacity(batch_len);
        let propose_res: Result<()> = (|| {
            for ((slot, seq, pending_ingest, sampler), slot_extras) in
                items.iter_mut().zip(all_extras.iter_mut())
            {
                let slot = *slot;
                self.select_mtp_batch_slot(slot)?;
                let anchor = *seq
                    .generated_tokens
                    .last()
                    .ok_or_else(|| loader_err("missing anchor in batched MTP verify"))?;
                let mut extras = std::mem::take(slot_extras);
                let k_budget = extras.len();
                let propose_step = if pending_ingest.is_empty() {
                    self.propose(engine, anchor, k_budget)
                } else {
                    let ptrs: Vec<u64> = self.mtp_batch_slots[slot]
                        .captured_verify_hidden
                        .iter()
                        .map(|b| b.addr())
                        .collect();
                    let res =
                        self.ingest_and_propose(engine, pending_ingest, &ptrs, anchor, k_budget);
                    if res.is_ok() {
                        pending_ingest.clear();
                    }
                    res
                };
                let mut drafts = match propose_step {
                    Ok(out) => out,
                    Err(e) => {
                        engine.free_state_slots(&extras);
                        return Err(e);
                    }
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
                if extras.len() > drafts.len() {
                    let excess = extras.split_off(drafts.len());
                    engine.free_state_slots(&excess);
                }
                proposed.push((slot, anchor, drafts, extras));
            }
            Ok(())
        })();
        if let Err(e) = propose_res {
            for extras in &all_extras {
                engine.free_state_slots(extras);
            }
            for (_, _, _, extras) in &proposed {
                engine.free_state_slots(extras);
            }
            return Err(e);
        }

        // Pass 2: infallibly assemble `StateTx` schedules. Every fallible operation after this
        // point runs inside `res` so any error triggers deterministic `StateTx` cleanup.
        for ((_, seq, _, _), (slot, anchor, drafts, extras)) in items.iter_mut().zip(proposed) {
            let mut tx = StateTx::new(seq.radix_state.active_state_slot);
            tx.commit_prefill(seq.radix_state.token_ids.len());
            let schedule = if !drafts.is_empty() {
                tx.begin_spec_verify(extras);
                tx.verify_slot_schedule()
            } else {
                engine.free_state_slots(&extras);
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

            // Copy each slot's `processed` target feature rows from the unified capture buffer
            // (`last_final_hidden_f16` for MTP or `tap_aux_bf16` for DFlash2) into that slot's
            // dedicated per-rank staging buffer.
            let captured_ptrs = self.captured(engine)?;
            for (rank_idx, t) in engine.ranks.iter().enumerate() {
                let final_h = captured_ptrs[rank_idx];
                t.device.make_current()?;
                for (prep, decision) in prepared.iter().zip(&decisions) {
                    let processed = decision.processed_rows();
                    let src = final_h + (prep.row_offset * h_bytes) as u64;
                    let dst =
                        self.mtp_batch_slots[prep.slot].captured_verify_hidden[rank_idx].addr();
                    check_rc(
                        // SAFETY: `captured_ptrs` holds all `row_offset <= max_rows` rows,
                        // and `captured_verify_hidden` holds `(k + 1) * h_bytes >= processed * h_bytes`.
                        unsafe {
                            ffi::w34_memcpy_d2d_async(
                                dst as *mut _,
                                src as *const _,
                                processed * h_bytes,
                                t.stream.handle,
                            )
                        },
                        "batched proposer verify hidden copy",
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

pub(crate) fn rank_batch_slot_candidate(
    res: &[i32],
    prompt: &[i32],
    last_used_tick: u64,
) -> (u8, usize, u64) {
    let lcp = res.iter().zip(prompt).take_while(|(a, b)| a == b).count();
    if lcp >= 16 && res.len().saturating_sub(lcp) <= 256 {
        (2, lcp, u64::MAX - (res.len() as u64))
    } else if res.is_empty() {
        (1, 0, u64::MAX)
    } else {
        (0, 0, u64::MAX - last_used_tick)
    }
}

#[cfg(test)]
mod tests {
    use super::rank_batch_slot_candidate;

    #[test]
    fn batch_slot_ranking_prefers_session_continuation_then_empty_then_shared_prefix() {
        let mut s0_turn1: Vec<i32> = (0..1800).collect();
        s0_turn1.extend(10_000..11_400); // 3200 tokens total
        let mut s4_turn1: Vec<i32> = (0..1800).collect();
        s4_turn1.extend(20_000..21_100); // shares 1800-token system prompt, diverges after
        let mut s0_turn2 = s0_turn1.clone();
        s0_turn2.extend(30_000..31_300); // continues s0_turn1

        // When Session 4 arrives on Turn 1, an empty slot (Tier 1) must beat Session 0's
        // diverged slot (Tier 0) even though Session 0 shares the 1800-token system prompt.
        let empty_rank = rank_batch_slot_candidate(&[], &s4_turn1, 0);
        let diverged_rank = rank_batch_slot_candidate(&s0_turn1, &s4_turn1, 10);
        assert!(empty_rank > diverged_rank);

        // When Session 0 arrives on Turn 2, its own slot (Tier 2) must beat both an empty
        // slot (Tier 1) and Session 4's diverged slot (Tier 0).
        let continuation_rank = rank_batch_slot_candidate(&s0_turn1, &s0_turn2, 10);
        let empty_for_s0 = rank_batch_slot_candidate(&[], &s0_turn2, 0);
        let s4_for_s0 = rank_batch_slot_candidate(&s4_turn1, &s0_turn2, 20);
        assert!(continuation_rank > empty_for_s0);
        assert!(empty_for_s0 > s4_for_s0);

        // When all slots are non-empty and none is a Tier 2 continuation, Tier 0 must evict
        // the oldest LRU slot (tick=5) rather than a newer short slot (tick=50), even if the
        // older slot has a longer resident token sequence from an earlier workload.
        let stale_long: Vec<i32> = (50_000..61_000).collect();
        let recent_short: Vec<i32> = (70_000..70_600).collect();
        let new_session: Vec<i32> = (80_000..81_500).collect();
        let stale_rank = rank_batch_slot_candidate(&stale_long, &new_session, 5);
        let recent_rank = rank_batch_slot_candidate(&recent_short, &new_session, 50);
        assert!(stale_rank > recent_rank);
    }
}
