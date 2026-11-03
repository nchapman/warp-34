//! Pool coordination, phased graph replay, pipelining, and batched stepping.

use super::*;

/// A Result error before launch is recoverable; a launched peer may already be
/// waiting in a collective. This handles Result failures, not unwinding panics.
#[cfg(feature = "cuda")]
fn finish_graph_replay(
    result: std::result::Result<(), crate::coordinated::ReplayError<CudaError>>,
) -> Result<()> {
    result.map_err(|error| {
        if error.partial_launch {
            // CUDA offers no safe cancellation of the peer's collective. Do not
            // unwind/free its buffers or retry; let the supervisor restart.
            eprintln!(
                "[w34] fatal: asymmetric TP graph launch failure: {}",
                error.source
            );
            std::process::abort();
        }
        error.source
    })
}

#[cfg(feature = "cuda")]
impl Qwen3_5Engine {
    pub fn alloc_kv_page(&mut self) -> Option<i32> {
        let mut radix = std::mem::replace(&mut self.radix, MambaRadixCache::new(16, 64));
        let out = self.alloc_kv_page_from(&mut radix);
        self.radix = radix;
        out
    }

    pub fn alloc_kv_page_from(&mut self, radix: &mut MambaRadixCache) -> Option<i32> {
        if let Some(p0) = self.ranks[0].kv_pool.alloc_page() {
            if self.tp_size == 2 {
                let p1 = self.ranks[1].kv_pool.alloc_page();
                debug_assert_eq!(Some(p0), p1);
            }
            return Some(p0);
        }
        let ev = radix.evict_full_pages(1);
        if !ev.freed_kv_pages.is_empty() || !ev.freed_mamba_slots.is_empty() {
            self.free_kv_pages(&ev.freed_kv_pages);
            self.free_state_slots(&ev.freed_mamba_slots);
            if let Some(p0) = self.ranks[0].kv_pool.alloc_page() {
                if self.tp_size == 2 {
                    let p1 = self.ranks[1].kv_pool.alloc_page();
                    debug_assert_eq!(Some(p0), p1);
                }
                return Some(p0);
            }
        }
        None
    }

    /// Evicts any deficit pages from `radix` in a single batch before allocating `target_pages`
    /// from the ascending `KvPool` bitmap, preserving physical page contiguity across evictions.
    pub fn reserve_kv_pages_from(
        &mut self,
        radix: &mut MambaRadixCache,
        dst: &mut Vec<i32>,
        target_pages: usize,
    ) -> bool {
        if dst.len() >= target_pages {
            return true;
        }
        let missing = target_pages - dst.len();
        let avail = self.ranks[0].kv_pool.free_count();
        if avail < missing {
            let ev = radix.evict_full_pages(missing - avail);
            if !ev.freed_kv_pages.is_empty() || !ev.freed_mamba_slots.is_empty() {
                self.free_kv_pages(&ev.freed_kv_pages);
                self.free_state_slots(&ev.freed_mamba_slots);
            }
        }
        if self.ranks[0].kv_pool.free_count() < missing {
            return false;
        }
        dst.reserve(missing);
        for _ in 0..missing {
            let p0 = self.ranks[0]
                .kv_pool
                .alloc_page()
                .expect("free_count verified before page allocation");
            if self.tp_size == 2 {
                let p1 = self.ranks[1].kv_pool.alloc_page();
                debug_assert_eq!(Some(p0), p1);
            }
            dst.push(p0);
        }
        true
    }

    pub fn free_kv_pages(&mut self, pages: &[i32]) {
        for r in &mut self.ranks {
            for &p in pages {
                if p >= 0 {
                    r.kv_pool.free_page(p);
                }
            }
        }
    }

    pub fn alloc_state_slot(&mut self) -> Option<i32> {
        let mut radix = std::mem::replace(&mut self.radix, MambaRadixCache::new(16, 64));
        let out = self.alloc_state_slot_from(&mut radix);
        self.radix = radix;
        out
    }

    pub fn alloc_state_slot_from(&mut self, radix: &mut MambaRadixCache) -> Option<i32> {
        if let Some(s0) = self.ranks[0].state_pool.alloc_slot() {
            if self.tp_size == 2 {
                let s1 = self.ranks[1].state_pool.alloc_slot();
                debug_assert_eq!(Some(s0), s1);
            }
            return Some(s0);
        }
        let ev = radix.evict_mamba(1);
        if !ev.freed_kv_pages.is_empty() || !ev.freed_mamba_slots.is_empty() {
            self.free_kv_pages(&ev.freed_kv_pages);
            self.free_state_slots(&ev.freed_mamba_slots);
            if let Some(s0) = self.ranks[0].state_pool.alloc_slot() {
                if self.tp_size == 2 {
                    let s1 = self.ranks[1].state_pool.alloc_slot();
                    debug_assert_eq!(Some(s0), s1);
                }
                return Some(s0);
            }
        }
        None
    }

    pub fn free_state_slots(&mut self, slots: &[i32]) {
        for r in &mut self.ranks {
            for &s in slots {
                if s >= 0 {
                    r.state_pool.free_slot(s);
                }
            }
        }
    }

    fn step_spec_verify_graph(&mut self, s: &EngineBatchSlice<'_>) -> Result<Vec<Vec<i32>>> {
        let q_len = s.tokens.len();
        let long_ctx = s.ctx_len_before + q_len > 1024;
        let host_meta = SpecVerifyGraphIo::pack_meta(s)?;
        use crate::coordinated::{ReplayPhase, replay_ranks};
        let comm = self.tp2_comm.as_ref().cloned();
        if let Some(comm) = &comm {
            let stage_bytes = q_len
                .checked_mul(self.cfg.hidden_size)
                .and_then(|v| v.checked_mul(2))
                .ok_or_else(|| loader_err("tp2 staging payload overflow"))?;
            if stage_bytes > comm.max_stage_bytes_per_epoch {
                return Err(loader_err(
                    "tp2 batch staging bytes exceed max_stage_bytes_per_epoch",
                ));
            }
        }
        // Capture records collectives without executing them. Both cached and
        // first-use graphs finish every rank's uploads/capture before any launch.
        let result = replay_ranks(self.ranks.len(), |phase, rank| {
            let r = &mut self.ranks[rank];
            r.device.make_current()?;
            match phase {
                ReplayPhase::Upload => {
                    if r.ple_runtime.is_some() {
                        let mut ckpt_dst = Vec::with_capacity(s.checkpoints.len());
                        let mut ckpt_off = Vec::with_capacity(s.checkpoints.len());
                        for &(trig_off, dst_slot) in s.checkpoints {
                            ckpt_off.push(trig_off);
                            ckpt_dst.push(dst_slot);
                        }
                        r.stage_ple_f8_for_batch(
                            s.tokens,
                            &[0, q_len as i32],
                            &[s.state_in_slot],
                            &[s.state_out_slot],
                            &[u8::from(s.has_initial_state)],
                            s.spec_slots,
                            s.spec_slots.len(),
                            &ckpt_dst,
                            &ckpt_off,
                            s.checkpoints.len(),
                        )?;
                    }
                    r.spec_verify_io
                        .meta_i32
                        .copy_from_host(&host_meta, &r.stream)?;
                    r.spec_verify_io
                        .block_tables
                        .copy_from_host(s.block_table, &r.stream)
                }
                ReplayPhase::Capture => {
                    r.ensure_spec_verify_graph(q_len, long_ctx, comm.as_deref())
                }
                ReplayPhase::Launch => {
                    r.spec_verify_graph.as_ref().unwrap().1.launch(&r.stream)?;
                    r.speculative_logits_rows = if r.has_spec_logits_capacity(q_len) {
                        q_len
                    } else {
                        0
                    };
                    r.last_logits_ptr = r.captured_spec_logits_ptr;
                    r.last_logits_rows = q_len;
                    Ok(())
                }
            }
        });
        finish_graph_replay(result)?;
        let mut out_ids = vec![0i32; q_len];
        self.ranks[0].device.make_current()?;
        self.ranks[0]
            .spec_verify_io
            .out_greedy_ids
            .copy_to_host(&mut out_ids, &self.ranks[0].stream)?;
        for r in &self.ranks[1..] {
            r.device.make_current()?;
            r.stream.synchronize()?;
        }
        Ok(vec![out_ids])
    }

    pub fn captured_decode_graph_keys(&self) -> Vec<(usize, bool)> {
        let mut keys: Vec<(usize, bool)> = self.ranks[0]
            .batched_decode_graphs
            .iter()
            .map(|(k, _, _)| (k.0, k.1))
            .collect();
        keys.sort_unstable();
        keys.dedup();
        keys
    }

    fn step_batched_decode_graph(
        &mut self,
        slices: &[EngineBatchSlice<'_>],
        rope_pos_deltas: Option<&[i32]>,
        bucket_b: usize,
        long_ctx: bool,
        skip_host_greedy: bool,
    ) -> Result<Vec<Vec<i32>>> {
        let n = slices.len();
        let (host_meta, host_masks) = BatchedDecodeGraphIo::pack_meta(slices, rope_pos_deltas)?;
        use crate::coordinated::{ReplayPhase, replay_ranks};
        let comm = self.tp2_comm.as_ref().cloned();
        let stage_bytes = bucket_b
            .checked_mul(self.cfg.hidden_size)
            .and_then(|v| v.checked_mul(2))
            .ok_or_else(|| loader_err("tp2 staging payload overflow"))?;
        if comm
            .as_ref()
            .is_some_and(|c| stage_bytes > c.max_stage_bytes_per_epoch)
        {
            return Err(loader_err(
                "tp2 batch staging bytes exceed max_stage_bytes_per_epoch",
            ));
        }
        let sample_mode = if skip_host_greedy {
            DecodeGraphSampleMode::LogitsOnly
        } else {
            DecodeGraphSampleMode::PlainGreedy
        };
        // Capture records nodes; it never executes collectives. Complete all uploads
        // and captures before launching either rank, including on first-use buckets.
        let result = replay_ranks(self.ranks.len(), |phase, rank| {
            let r = &mut self.ranks[rank];
            r.device.make_current()?;
            match phase {
                ReplayPhase::Upload => {
                    r.stage_ple_f8_for_decode_slices(slices)?;
                    r.batched_decode_io
                        .upload_inputs(&host_meta, &host_masks, slices, &r.stream)
                }
                ReplayPhase::Capture => {
                    r.ensure_batched_decode_graph(bucket_b, long_ctx, sample_mode, comm.as_deref())
                }
                ReplayPhase::Launch => {
                    let (logits_ptr, graph) = r
                        .find_batched_decode_graph(bucket_b, long_ctx, sample_mode)
                        .unwrap();
                    graph.launch(&r.stream)?;
                    r.speculative_logits_rows = if r.has_spec_logits_capacity(bucket_b) {
                        bucket_b
                    } else {
                        0
                    };
                    r.last_logits_ptr = logits_ptr;
                    r.last_logits_rows = bucket_b;
                    Ok(())
                }
            }
        });
        finish_graph_replay(result)?;
        if skip_host_greedy {
            return Ok(Vec::new());
        }
        let mut out_ids = vec![0i32; n];
        self.ranks[0].device.make_current()?;
        self.ranks[0]
            .batched_decode_io
            .out_greedy_ids
            .copy_to_host(&mut out_ids, &self.ranks[0].stream)?;
        self.ranks[0]
            .batched_decode_io
            .pinned_in_flight
            .store(false, std::sync::atomic::Ordering::Release);
        for r in &self.ranks[1..] {
            r.device.make_current()?;
            r.stream.synchronize()?;
            r.batched_decode_io
                .pinned_in_flight
                .store(false, std::sync::atomic::Ordering::Release);
        }
        Ok(out_ids.into_iter().map(|t| vec![t]).collect())
    }

    pub fn can_pipeline_decode_slices(&self, slices: &[EngineBatchSlice<'_>]) -> bool {
        if self.tp_size != 1 || !self.use_cuda_graph || slices.is_empty() {
            return false;
        }
        let Some(bucket_b) = decode_graph_bucket(slices.len()) else {
            return false;
        };
        let r0 = &self.ranks[0];
        if r0.current_pos_plane_stride != 0
            || r0.pending_visual_splice.is_some()
            || !r0.supports_cuda_graph_for_tokens(bucket_b)
        {
            return false;
        }
        let max_pages = r0.kv_pool.num_pages;
        let max_blocks = r0.batched_decode_io.max_blocks_per_req;
        slices.iter().all(|s| {
            let seq_after = s.ctx_len_before + 1;
            s.tokens.len() == 1
                && s.spec_slots.is_empty()
                && s.checkpoints.len() <= 1
                && s.block_table.len() <= max_blocks
                && seq_after.div_ceil(16) <= s.block_table.len()
                && s.block_table
                    .iter()
                    .all(|&p| p >= 0 && (p as usize) < max_pages)
        })
    }

    #[allow(clippy::too_many_arguments)]
    pub fn enqueue_pipelined_decode_step(
        &mut self,
        slot: usize,
        slices: &[EngineBatchSlice<'_>],
        rope_pos_deltas: Option<&[i32]>,
        tokens_already_on_device: bool,
        scatter_next_tokens_on_device: bool,
        fast_greedy_inputs: Option<(&[ffi::W34SamplerRowCfg], &[ffi::W34SamplerSparseEntry])>,
    ) -> Result<InFlightDecodeTicket> {
        if self.tp_size != 1 {
            return Err(loader_err("pipelined decode overlap requires tp_size == 1"));
        }
        if let Some(deltas) = rope_pos_deltas
            && deltas.len() != slices.len()
        {
            return Err(loader_err("rope_pos_deltas length mismatch"));
        }
        let bucket_b = decode_graph_bucket(slices.len())
            .ok_or_else(|| loader_err("invalid batch size for pipelined decode graph"))?;
        let max_seq_after = slices
            .iter()
            .map(|s| s.ctx_len_before + 1)
            .max()
            .unwrap_or(1);
        let long_ctx = max_seq_after > 1024;
        let sample_mode = if fast_greedy_inputs.is_some() {
            DecodeGraphSampleMode::FastGreedy
        } else {
            DecodeGraphSampleMode::PlainGreedy
        };

        let slot = slot & 1;
        let n = slices.len();
        let (host_meta, host_masks) = BatchedDecodeGraphIo::pack_meta(slices, rope_pos_deltas)?;
        let r0 = &mut self.ranks[0];
        r0.device.make_current()?;
        if tokens_already_on_device
            && r0
                .find_batched_decode_graph(bucket_b, long_ctx, sample_mode)
                .is_none()
        {
            return Err(loader_err(
                "pipelined lookahead step missing pre-captured decode graph",
            ));
        }
        r0.ensure_batched_decode_graph(bucket_b, long_ctx, sample_mode, None)?;
        r0.stage_ple_f8_for_decode_slices(slices)?;
        r0.batched_decode_io.upload_inputs_slot(
            slot,
            &host_meta,
            &host_masks,
            slices,
            tokens_already_on_device,
            true,
            &r0.stream,
        )?;
        if let Some((cfgs, sparse)) = fast_greedy_inputs {
            r0.batched_decode_io
                .upload_fast_greedy_slot(slot, cfgs, sparse, bucket_b, &r0.stream)?;
        }
        let (logits_ptr, graph) = r0
            .find_batched_decode_graph(bucket_b, long_ctx, sample_mode)
            .unwrap();
        graph.launch(&r0.stream)?;
        r0.speculative_logits_rows = if r0.has_spec_logits_capacity(bucket_b) {
            bucket_b
        } else {
            0
        };
        r0.last_logits_ptr = logits_ptr;
        r0.last_logits_rows = bucket_b;

        if scatter_next_tokens_on_device {
            let ex0 = r0.stream.exec(0, 0);
            let (row_outs_ptr, greedy_ids_ptr) = if fast_greedy_inputs.is_some() {
                (
                    r0.batched_decode_io.fast_greedy_outs.addr() as *const ffi::W34SamplerRowOut,
                    core::ptr::null(),
                )
            } else {
                (
                    core::ptr::null(),
                    r0.batched_decode_io.out_greedy_ids.addr() as *const i32,
                )
            };
            check_rc(
                // SAFETY: n is within the decode bucket. Exactly one source is
                // nonnull, selected to match the captured sampler mode; both source
                // and token-ID storage are rank-owned static buffers. This launch
                // follows graph output production on the same stream.
                unsafe {
                    ffi::w34_sampler_scatter_ids_to_tokens(
                        &ex0,
                        row_outs_ptr,
                        greedy_ids_ptr,
                        r0.batched_decode_io.token_ids_ptr() as *mut i32,
                        n as i32,
                    )
                },
                "w34_sampler_scatter_ids_to_tokens",
            )?;
        }

        // SAFETY: slot is 0/1, and new() allocates two fixed-size slots with
        // room for MAX_BATCHED_DECODE_REQS outputs at this aligned offset.
        // The pipeline caller must finish a slot's ticket before reusing it.
        let pout = unsafe {
            r0.batched_decode_io
                .pinned_staging
                .host_ptr
                .cast::<u8>()
                .add(
                    slot * BatchedDecodeGraphIo::PINNED_SLOT_BYTES
                        + BatchedDecodeGraphIo::PINNED_PIPELINE_OUT_OFFSET,
                )
        };
        let (dev_out_ptr, d2h_bytes) = if fast_greedy_inputs.is_some() {
            (
                r0.batched_decode_io.fast_greedy_outs.addr(),
                n * core::mem::size_of::<ffi::W34SamplerRowOut>(),
            )
        } else {
            (r0.batched_decode_io.out_greedy_ids.addr(), n * 4)
        };
        check_rc(
            // SAFETY: n <= MAX_BATCHED_DECODE_REQS bounds source and pinned output
            // extents for either ABI. The rank retains the pinned allocation;
            // the recorded completion event must be waited before host reads/reuse.
            unsafe {
                ffi::w34_memcpy_d2h_async(
                    pout.cast(),
                    dev_out_ptr as *const c_void,
                    d2h_bytes,
                    r0.stream.handle,
                )
            },
            "pipelined decode D2H output",
        )?;
        r0.batched_decode_io.pipeline_done_events[slot].record(&r0.stream)?;

        Ok(InFlightDecodeTicket {
            slot,
            num_reqs: n,
            fast_greedy: fast_greedy_inputs.is_some(),
        })
    }

    pub fn wait_pipelined_decode_step(
        &mut self,
        ticket: InFlightDecodeTicket,
    ) -> Result<PipelinedDecodeOutput> {
        let r0 = &mut self.ranks[0];
        r0.device.make_current()?;
        r0.batched_decode_io.pipeline_done_events[ticket.slot].synchronize()?;
        if ticket.slot == 0 {
            r0.batched_decode_io
                .pinned_in_flight
                .store(false, std::sync::atomic::Ordering::Release);
            r0.batched_decode_io
                .pinned_sampler_in_flight
                .store(false, std::sync::atomic::Ordering::Release);
        }
        // SAFETY: a ticket returned by enqueue identifies one of the two slots;
        // the event wait above completes its D2H. Ticket fields are public, so
        // provenance and not reusing the slot early remain caller obligations.
        let pout = unsafe {
            r0.batched_decode_io
                .pinned_staging
                .host_ptr
                .cast::<u8>()
                .add(
                    ticket.slot * BatchedDecodeGraphIo::PINNED_SLOT_BYTES
                        + BatchedDecodeGraphIo::PINNED_PIPELINE_OUT_OFFSET,
                )
        };
        if ticket.fast_greedy {
            let mut outs = vec![ffi::W34SamplerRowOut::default(); ticket.num_reqs];
            // SAFETY: for an unmodified completed ticket, this slot contains
            // num_reqs initialized RowOut records. The offset is 64-byte aligned;
            // outs is a distinct allocation with the same record count.
            unsafe {
                core::ptr::copy_nonoverlapping(
                    pout.cast::<ffi::W34SamplerRowOut>(),
                    outs.as_mut_ptr(),
                    ticket.num_reqs,
                );
            }
            Ok(PipelinedDecodeOutput::FastGreedy(outs))
        } else {
            let mut ids = vec![0i32; ticket.num_reqs];
            // SAFETY: an unmodified plain-greedy ticket denotes num_reqs i32
            // values whose D2H completed above. Slot offsets are i32-aligned and
            // ids is a separate allocation sized to exactly that count.
            unsafe {
                core::ptr::copy_nonoverlapping(
                    pout.cast::<i32>(),
                    ids.as_mut_ptr(),
                    ticket.num_reqs,
                );
            }
            Ok(PipelinedDecodeOutput::PlainGreedy(ids))
        }
    }

    pub fn abort_pipelined_decode(&mut self) {
        for r in &self.ranks {
            let _ = r.device.make_current();
            let _ = r.stream.synchronize();
            let _ = r.aux_stream.synchronize();
            r.batched_decode_io
                .pinned_in_flight
                .store(false, std::sync::atomic::Ordering::Release);
            r.batched_decode_io
                .pinned_sampler_in_flight
                .store(false, std::sync::atomic::Ordering::Release);
        }
    }

    /// Executes a multi-request / speculative / checkpoint-capturing batch step across `TP=1` or `TP=2`.
    /// Returns one `Vec<i32>` of greedy token IDs per slice in `slices` (either `[last_tok]` or all
    /// `q_len` query tokens when `slice.emit_all_query_tokens == true`).
    pub fn step_batch_slices(&mut self, slices: &[EngineBatchSlice<'_>]) -> Result<Vec<Vec<i32>>> {
        self.step_batch_slices_with_pos_deltas(slices, None)
    }

    pub fn step_batch_slices_with_pos_deltas(
        &mut self,
        slices: &[EngineBatchSlice<'_>],
        rope_pos_deltas: Option<&[i32]>,
    ) -> Result<Vec<Vec<i32>>> {
        self.step_batch_slices_for_sampler(slices, rope_pos_deltas, false)
    }

    /// Internal batch-step entrypoint that optionally skips host greedy argmax D2H (`Ok(Vec::new())`)
    /// when `skip_host_greedy == true` and the caller immediately samples device logits via `run_gpu_sampler_rows`.
    pub(crate) fn step_batch_slices_for_sampler(
        &mut self,
        slices: &[EngineBatchSlice<'_>],
        rope_pos_deltas: Option<&[i32]>,
        skip_host_greedy: bool,
    ) -> Result<Vec<Vec<i32>>> {
        self.step_batch_slices_for_sampler_filtered(slices, rope_pos_deltas, None, skip_host_greedy)
    }

    /// Batch-step entrypoint with an optional per-slice `emit_mask`. When `emit_mask[r] == false`
    /// (e.g. non-final chunked-prefill slices whose logits are discarded), skips selecting/projecting
    /// `lm_head` rows for slice `r`.
    pub(crate) fn step_batch_slices_for_sampler_filtered(
        &mut self,
        slices: &[EngineBatchSlice<'_>],
        rope_pos_deltas: Option<&[i32]>,
        emit_mask: Option<&[bool]>,
        skip_host_greedy: bool,
    ) -> Result<Vec<Vec<i32>>> {
        if slices.is_empty() {
            return Ok(Vec::new());
        }
        if let Some(deltas) = rope_pos_deltas
            && deltas.len() != slices.len()
        {
            return Err(loader_err("rope_pos_deltas length mismatch"));
        }
        if let Some(mask) = emit_mask
            && mask.len() != slices.len()
        {
            return Err(loader_err("emit_mask length mismatch"));
        }
        let all_emit = emit_mask.is_none_or(|m| m.iter().all(|&e| e));
        let max_pages = self.ranks[0].kv_pool.num_pages;
        if all_emit
            && let Some(bucket_b) = decode_graph_bucket(slices.len())
            && self.use_cuda_graph
            && self
                .ranks
                .iter()
                .all(|r| r.supports_cuda_graph_for_tokens(bucket_b))
            && slices.iter().all(|s| {
                s.tokens.len() == 1
                    && s.spec_slots.is_empty()
                    && s.checkpoints.len() <= 1
                    && s.block_table.len() <= self.ranks[0].batched_decode_io.max_blocks_per_req
            })
            && self
                .ranks
                .iter()
                .all(|r| r.current_pos_plane_stride == 0 && r.pending_visual_splice.is_none())
        {
            let mut max_seq_after = 0usize;
            for s in slices {
                let seq_after = s.ctx_len_before + 1;
                if seq_after.div_ceil(16) > s.block_table.len()
                    || s.block_table
                        .iter()
                        .any(|&p| p < 0 || (p as usize) >= max_pages)
                {
                    return Err(loader_err(
                        "KV cache page capacity exceeded in step_batch_slices",
                    ));
                }
                max_seq_after = max_seq_after.max(seq_after);
            }
            let long_ctx = max_seq_after > 1024;
            // Failed preparation/capture has not run a forward pass. Surface the error
            // without disabling unrelated graph families or replaying mutable state.
            return self.step_batched_decode_graph(
                slices,
                rope_pos_deltas,
                bucket_b,
                long_ctx,
                skip_host_greedy,
            );
        }
        if slices.len() == 1 && rope_pos_deltas.is_none() {
            let s = &slices[0];
            let q_len = s.tokens.len();
            if q_len == 0 {
                return Err(loader_err("empty slice in step_batch_slices"));
            }
            let max_pages = self.ranks[0].kv_pool.num_pages;
            let seq_after = s.ctx_len_before + q_len;
            if seq_after.div_ceil(16) > s.block_table.len()
                || s.block_table
                    .iter()
                    .any(|&p| p < 0 || (p as usize) >= max_pages)
            {
                return Err(loader_err(
                    "KV cache page capacity exceeded in step_batch_slices",
                ));
            }
            let can_use_spec_graph = self.use_cuda_graph
                && (2..=MAX_SPEC_VERIFY_TOKENS).contains(&q_len)
                && s.has_initial_state
                && s.emit_all_query_tokens
                && s.spec_slots.len() == q_len
                && s.checkpoints.len() <= MAX_SPEC_VERIFY_CHECKPOINTS
                && s.block_table.len() <= self.ranks[0].spec_verify_io.max_blocks_per_req
                && self
                    .ranks
                    .iter()
                    .all(|r| r.supports_cuda_graph_for_tokens(q_len))
                && self.ranks.iter().all(|r| {
                    let expected_m = r
                        .speculative_logits
                        .as_ref()
                        .map(|b| b.len_bytes / (r.local_vocab_size * 2));
                    r.current_pos_plane_stride == 0
                        && r.pending_visual_splice.is_none()
                        && expected_m.is_none_or(|cap| cap == q_len)
                        && r.spec_verify_graph
                            .as_ref()
                            .is_none_or(|((captured_m, _), _)| *captured_m == q_len)
                });
            if can_use_spec_graph {
                let long_ctx = seq_after > 1024;
                let needs_capture = self.ranks.iter().any(|r| {
                    r.spec_verify_graph
                        .as_ref()
                        .is_none_or(|(k, _)| *k != (q_len, long_ctx))
                });
                match self.step_spec_verify_graph(s) {
                    Ok(out) => return Ok(out),
                    Err(e) if needs_capture => {
                        for r in &mut self.ranks {
                            r.spec_verify_graph = None;
                        }
                        self.use_cuda_graph = false;
                        eprintln!(
                            "[w34] warning: spec_verify CUDA Graph capture failed ({e}); degrading to eager verification"
                        );
                    }
                    Err(e) => return Err(e),
                }
            }
        }
        let num_reqs = slices.len();
        let mut flat_tokens = Vec::new();
        let mut flat_positions = Vec::new();
        let mut flat_write_slots = Vec::new();
        let mut q_start_loc = Vec::with_capacity(num_reqs + 1);
        q_start_loc.push(0i32);
        let mut ctx_lens_before = Vec::with_capacity(num_reqs);
        let mut seq_lens_after = Vec::with_capacity(num_reqs);
        let mut state_in_indices = Vec::with_capacity(num_reqs);
        let mut state_out_indices = Vec::with_capacity(num_reqs);
        let mut has_initial_state = Vec::with_capacity(num_reqs);
        let mut selected_rows = Vec::new();
        let mut row_counts_per_req = Vec::with_capacity(num_reqs);

        let max_blocks_per_req = slices
            .iter()
            .map(|s| s.block_table.len())
            .max()
            .unwrap_or(1)
            .max(1);
        let max_verify_tokens = slices.iter().map(|s| s.spec_slots.len()).max().unwrap_or(0);
        let max_checkpoints_per_req = slices
            .iter()
            .map(|s| s.checkpoints.len())
            .max()
            .unwrap_or(0);
        let mut max_q_len = 0usize;
        let mut max_seq_len = 0usize;
        let max_pages = self.ranks[0].kv_pool.num_pages;

        let mut block_tables = vec![0i32; num_reqs * max_blocks_per_req];
        let mut spec_state_indices = if max_verify_tokens > 0 {
            vec![-1i32; num_reqs * max_verify_tokens]
        } else {
            Vec::new()
        };
        let mut checkpoint_dst_indices = if max_checkpoints_per_req > 0 {
            vec![-1i32; num_reqs * max_checkpoints_per_req]
        } else {
            Vec::new()
        };
        let mut checkpoint_token_offsets = if max_checkpoints_per_req > 0 {
            vec![-1i32; num_reqs * max_checkpoints_per_req]
        } else {
            Vec::new()
        };

        for (r, s) in slices.iter().enumerate() {
            let q_len = s.tokens.len();
            if q_len == 0 {
                return Err(loader_err("empty slice in step_batch_slices"));
            }
            let seq_after = s.ctx_len_before + q_len;
            if seq_after.div_ceil(16) > s.block_table.len()
                || s.block_table
                    .iter()
                    .any(|&p| p < 0 || (p as usize) >= max_pages)
            {
                return Err(loader_err(
                    "KV cache page capacity exceeded in step_batch_slices",
                ));
            }
            let base_tok = flat_tokens.len();
            flat_tokens.extend_from_slice(s.tokens);
            let delta = rope_pos_deltas.and_then(|d| d.get(r).copied()).unwrap_or(0);
            for step in 0..q_len {
                let pos = s.ctx_len_before + step;
                flat_positions.push((pos as i32) + delta);
                let page_id = s.block_table[pos / 16];
                let page_off = (pos % 16) as i32;
                flat_write_slots.push(page_id * 16 + page_off);
            }
            let emit_slice = emit_mask.is_none_or(|m| m[r]);
            if !emit_slice {
                row_counts_per_req.push(0);
            } else if s.emit_all_query_tokens {
                for step in 0..q_len {
                    selected_rows.push(base_tok + step);
                }
                row_counts_per_req.push(q_len);
            } else {
                selected_rows.push(base_tok + q_len - 1);
                row_counts_per_req.push(1);
            }
            q_start_loc.push(flat_tokens.len() as i32);
            ctx_lens_before.push(s.ctx_len_before as i32);
            seq_lens_after.push(seq_after as i32);
            state_in_indices.push(s.state_in_slot);
            state_out_indices.push(s.state_out_slot);
            has_initial_state.push(u8::from(s.has_initial_state));
            max_q_len = max_q_len.max(q_len);
            max_seq_len = max_seq_len.max(seq_after);

            block_tables[r * max_blocks_per_req..r * max_blocks_per_req + s.block_table.len()]
                .copy_from_slice(s.block_table);
            if max_verify_tokens > 0 {
                for (k, &slot) in s.spec_slots.iter().enumerate() {
                    spec_state_indices[r * max_verify_tokens + k] = slot;
                }
            }
            if max_checkpoints_per_req > 0 {
                for (e, &(trig_off, dst_slot)) in s.checkpoints.iter().enumerate() {
                    checkpoint_token_offsets[r * max_checkpoints_per_req + e] = trig_off;
                    checkpoint_dst_indices[r * max_checkpoints_per_req + e] = dst_slot;
                }
            }
        }

        let total_tokens = flat_tokens.len();
        let flat_ids = if self.tp_size == 1 {
            let r0 = &mut self.ranks[0];
            let (tok_ptr, pos_ptr, attn_batch, gdn_batch) = r0.upload_batch_meta(
                &flat_tokens,
                &flat_positions,
                &q_start_loc,
                &ctx_lens_before,
                &seq_lens_after,
                &block_tables,
                &flat_write_slots,
                &state_in_indices,
                &state_out_indices,
                &has_initial_state,
                &spec_state_indices,
                &checkpoint_dst_indices,
                &checkpoint_token_offsets,
                num_reqs,
                total_tokens,
                max_blocks_per_req,
                max_q_len,
                max_seq_len,
                max_verify_tokens,
                max_checkpoints_per_req,
            )?;
            r0.forward_selected_rows(
                tok_ptr,
                pos_ptr,
                total_tokens,
                &attn_batch,
                &gdn_batch,
                None,
                &selected_rows,
            )?
        } else {
            let comm = self.tp2_comm.as_ref().unwrap().clone();
            let [r0, r1] = &mut self.ranks[..] else {
                unreachable!()
            };
            let ft_ref = &flat_tokens;
            let fp_ref = &flat_positions;
            let qs_ref = &q_start_loc;
            let cb_ref = &ctx_lens_before;
            let sa_ref = &seq_lens_after;
            let bt_ref = &block_tables;
            let ws_ref = &flat_write_slots;
            let si_ref = &state_in_indices;
            let so_ref = &state_out_indices;
            let hi_ref = &has_initial_state;
            let sp_ref = &spec_state_indices;
            let cd_ref = &checkpoint_dst_indices;
            let co_ref = &checkpoint_token_offsets;
            let sr_ref = &selected_rows;

            let stage_bytes = total_tokens
                .checked_mul(self.cfg.hidden_size)
                .and_then(|v| v.checked_mul(2))
                .ok_or_else(|| loader_err("tp2 staging payload overflow"))?;
            if stage_bytes > comm.max_stage_bytes_per_epoch {
                return Err(loader_err(
                    "tp2 batch staging bytes exceed max_stage_bytes_per_epoch",
                ));
            }

            let (res0, res1) = std::thread::scope(|sc| {
                let c0 = comm.clone();
                let h0 = sc.spawn(move || -> Result<Vec<i32>> {
                    let (tok_ptr, pos_ptr, attn_batch, gdn_batch) = r0.upload_batch_meta(
                        ft_ref,
                        fp_ref,
                        qs_ref,
                        cb_ref,
                        sa_ref,
                        bt_ref,
                        ws_ref,
                        si_ref,
                        so_ref,
                        hi_ref,
                        sp_ref,
                        cd_ref,
                        co_ref,
                        num_reqs,
                        total_tokens,
                        max_blocks_per_req,
                        max_q_len,
                        max_seq_len,
                        max_verify_tokens,
                        max_checkpoints_per_req,
                    )?;
                    r0.forward_selected_rows(
                        tok_ptr,
                        pos_ptr,
                        total_tokens,
                        &attn_batch,
                        &gdn_batch,
                        Some(&c0),
                        sr_ref,
                    )
                });

                let c1 = comm.clone();
                let h1 = sc.spawn(move || -> Result<()> {
                    let (tok_ptr, pos_ptr, attn_batch, gdn_batch) = r1.upload_batch_meta(
                        ft_ref,
                        fp_ref,
                        qs_ref,
                        cb_ref,
                        sa_ref,
                        bt_ref,
                        ws_ref,
                        si_ref,
                        so_ref,
                        hi_ref,
                        sp_ref,
                        cd_ref,
                        co_ref,
                        num_reqs,
                        total_tokens,
                        max_blocks_per_req,
                        max_q_len,
                        max_seq_len,
                        max_verify_tokens,
                        max_checkpoints_per_req,
                    )?;
                    let _ = r1.forward_selected_rows(
                        tok_ptr,
                        pos_ptr,
                        total_tokens,
                        &attn_batch,
                        &gdn_batch,
                        Some(&c1),
                        sr_ref,
                    )?;
                    Ok(())
                });
                let j0 = h0
                    .join()
                    .map_err(|_| loader_err("tp2 rank 0 worker thread panicked"));
                let j1 = h1
                    .join()
                    .map_err(|_| loader_err("tp2 rank 1 worker thread panicked"));
                (j0, j1)
            });
            res1??;
            res0??
        };

        if skip_host_greedy {
            return Ok(Vec::new());
        }
        let mut out_per_req = Vec::with_capacity(num_reqs);
        let mut cursor = 0usize;
        for count in row_counts_per_req {
            out_per_req.push(flat_ids[cursor..cursor + count].to_vec());
            cursor += count;
        }
        Ok(out_per_req)
    }
}
