//! Rank-local CUDA graph capture and cached graph lookup.

use super::*;

#[cfg(feature = "cuda")]
impl Qwen3_5ModelRank {
    /// Captures the `M=1` decode forward pass into `self.decode_graph`, keyed by `long_ctx`
    /// (`seq_after > 1024`: `false` = single-pass GQA mma, `true` = parallel Split-KV).
    pub fn ensure_decode_graph(
        &mut self,
        long_ctx: bool,
        tp2: Option<&Tp2Collectives>,
    ) -> Result<()> {
        let long_ctx = !self.cfg.uses_qsa() && long_ctx;
        if let Some((captured_long, _)) = &self.decode_graph
            && *captured_long == long_ctx
        {
            return Ok(());
        }
        self.decode_graph = None;
        let max_seq_len = if long_ctx {
            (self.decode_io.max_blocks_per_req * 16).max(2048) as i32
        } else {
            1024
        };
        let attn_batch = W34AttentionBatch {
            query_start_loc: self.decode_io.q_start_loc.addr() as *const i32,
            context_lens_before: self.decode_io.ctx_lens_before.addr() as *const i32,
            seq_lens_after: self.decode_io.seq_lens_after.addr() as *const i32,
            block_tables: self.decode_io.block_tables.addr() as *const i32,
            write_slot_mapping: self.decode_io.write_slot_mapping.addr() as *const i32,
            valid_token_mask: core::ptr::null(),
            valid_req_mask: core::ptr::null(),
            num_reqs: 1,
            total_tokens: 1,
            max_blocks_per_req: self.decode_io.max_blocks_per_req as i32,
            max_q_len: 1,
            max_seq_len,
            mask_mode: W34_ATTN_MASK_CAUSAL,
            sliding_window: 0,
            sm_scale: (self.cfg.head_dim as f32).powf(-0.5),
            reserved: 0,
        };
        let gdn_batch = W34GdnBatch {
            query_start_loc: self.decode_io.q_start_loc.addr() as *const i32,
            state_in_indices: self.decode_io.state_in_indices.addr() as *const i32,
            state_out_indices: self.decode_io.state_out_indices.addr() as *const i32,
            spec_state_indices: core::ptr::null(),
            checkpoint_dst_indices: core::ptr::null(),
            checkpoint_token_offsets: core::ptr::null(),
            has_initial_state: self.decode_io.has_initial_state.addr() as *const u8,
            valid_req_mask: core::ptr::null(),
            num_reqs: 1,
            total_tokens: 1,
            max_verify_tokens: 0,
            max_checkpoints_per_req: 0,
        };
        let tok_ptr = self.decode_io.token_ids.addr();
        let pos_ptr = self.decode_io.positions.addr();
        let out_id_ptr = self.decode_io.out_greedy_id.addr();

        self.stream.synchronize()?;
        let stream_handle = self.stream.handle;
        let dev_id = self.device.device_id;
        // SAFETY: CUDA device selection takes an integer, not a borrowed pointer.
        // Select the rank's device before operating on its stream.
        check_rc(unsafe { ffi::w34_set_device(dev_id) }, "w34_set_device")?;
        check_rc(
            // SAFETY: stream_handle is this rank's live stream, synchronized above.
            unsafe { ffi::w34_graph_begin_capture(stream_handle) },
            "w34_graph_begin_capture",
        )?;
        self.is_capturing_graph = true;
        if let Some(rt) = &self.lora_runtime {
            rt.state
                .set_device_ids(self.decode_io.token_lora_ids.addr());
        }
        let res = self.forward_device_batches(
            tok_ptr,
            pos_ptr,
            1,
            &attn_batch,
            &gdn_batch,
            tp2,
            out_id_ptr,
            0,
            0,
            None,
        );
        self.is_capturing_graph = false;
        let mut graph = 0usize;
        let mut graph_exec = 0usize;
        // SAFETY: pair the successful begin with end even if the forward returned
        // Err. The stack out-parameters are live; own any returned handles before
        // propagating either Result error. A panic in forward is not handled here.
        let end_rc = unsafe {
            ffi::w34_graph_end_capture(stream_handle, &raw mut graph, &raw mut graph_exec)
        };
        let captured = CudaGraph {
            device_id: dev_id,
            graph,
            graph_exec,
        };
        res?;
        check_rc(end_rc, "w34_graph_end_capture")?;
        self.captured_decode_logits_ptr = self.last_logits_ptr;
        self.decode_graph = Some((long_ctx, captured));
        Ok(())
    }

    #[inline]
    pub fn has_spec_logits_capacity(&self, rows: usize) -> bool {
        self.speculative_logits
            .as_ref()
            .is_some_and(|b| rows <= b.len_bytes / (self.local_vocab_size * 2))
    }

    /// Captures a fixed-shape `M_verify = m_verify` (`2..=16`) single-request speculative
    /// verification forward pass into `self.spec_verify_graph`.
    pub fn ensure_spec_verify_graph(
        &mut self,
        m_verify: usize,
        long_ctx: bool,
        tp2: Option<&Tp2Collectives>,
    ) -> Result<()> {
        let long_ctx = !self.cfg.uses_qsa() && long_ctx;
        if let Some((captured_key, _)) = &self.spec_verify_graph
            && *captured_key == (m_verify, long_ctx)
        {
            return Ok(());
        }
        self.spec_verify_graph = None;
        let max_seq_len = if long_ctx {
            (self.spec_verify_io.max_blocks_per_req * 16).max(2048) as i32
        } else {
            1024
        };
        let attn_batch = W34AttentionBatch {
            query_start_loc: self.spec_verify_io.q_start_loc_ptr() as *const i32,
            context_lens_before: self.spec_verify_io.ctx_lens_before_ptr() as *const i32,
            seq_lens_after: self.spec_verify_io.seq_lens_after_ptr() as *const i32,
            block_tables: self.spec_verify_io.block_tables.addr() as *const i32,
            write_slot_mapping: self.spec_verify_io.write_slot_mapping_ptr() as *const i32,
            valid_token_mask: core::ptr::null(),
            valid_req_mask: core::ptr::null(),
            num_reqs: 1,
            total_tokens: m_verify as i32,
            max_blocks_per_req: self.spec_verify_io.max_blocks_per_req as i32,
            max_q_len: m_verify as i32,
            max_seq_len,
            mask_mode: W34_ATTN_MASK_CAUSAL,
            sliding_window: 0,
            sm_scale: (self.cfg.head_dim as f32).powf(-0.5),
            reserved: 0,
        };
        let gdn_batch = W34GdnBatch {
            query_start_loc: self.spec_verify_io.q_start_loc_ptr() as *const i32,
            state_in_indices: self.spec_verify_io.state_in_indices_ptr() as *const i32,
            state_out_indices: self.spec_verify_io.state_out_indices_ptr() as *const i32,
            spec_state_indices: self.spec_verify_io.spec_state_indices_ptr() as *const i32,
            checkpoint_dst_indices: self.spec_verify_io.checkpoint_dst_indices_ptr() as *const i32,
            checkpoint_token_offsets: self.spec_verify_io.checkpoint_token_offsets_ptr()
                as *const i32,
            has_initial_state: self.spec_verify_io.has_initial_state.addr() as *const u8,
            valid_req_mask: core::ptr::null(),
            num_reqs: 1,
            total_tokens: m_verify as i32,
            max_verify_tokens: m_verify as i32,
            max_checkpoints_per_req: MAX_SPEC_VERIFY_CHECKPOINTS as i32,
        };
        let tok_ptr = self.spec_verify_io.token_ids_ptr();
        let pos_ptr = self.spec_verify_io.positions_ptr();
        let out_ids_ptr = self.spec_verify_io.out_greedy_ids.addr();
        let selected_rows: Vec<usize> = (0..m_verify).collect();

        self.stream.synchronize()?;
        let stream_handle = self.stream.handle;
        let dev_id = self.device.device_id;
        // SAFETY: CUDA device selection takes an integer, not a borrowed pointer.
        // Select the rank's device before operating on its stream.
        check_rc(unsafe { ffi::w34_set_device(dev_id) }, "w34_set_device")?;
        check_rc(
            // SAFETY: stream_handle is this rank's live stream, synchronized above.
            unsafe { ffi::w34_graph_begin_capture(stream_handle) },
            "w34_graph_begin_capture (spec_verify)",
        )?;
        self.is_capturing_graph = true;
        if let Some(rt) = &self.lora_runtime {
            rt.state
                .set_device_ids(self.spec_verify_io.token_lora_ids_ptr());
        }
        let res = self.forward_selected_rows_device(
            tok_ptr,
            pos_ptr,
            m_verify,
            &attn_batch,
            &gdn_batch,
            tp2,
            &selected_rows,
            Some(out_ids_ptr),
            true,
        );
        self.is_capturing_graph = false;
        let mut graph = 0usize;
        let mut graph_exec = 0usize;
        // SAFETY: pair the successful begin with end even if the forward returned
        // Err. The stack out-parameters are live; own any returned handles before
        // propagating either Result error. A panic in forward is not handled here.
        let end_rc = unsafe {
            ffi::w34_graph_end_capture(stream_handle, &raw mut graph, &raw mut graph_exec)
        };
        let captured = CudaGraph {
            device_id: dev_id,
            graph,
            graph_exec,
        };
        res?;
        check_rc(end_rc, "w34_graph_end_capture (spec_verify)")?;
        self.captured_spec_logits_ptr = self.last_logits_ptr;
        self.spec_verify_graph = Some(((m_verify, long_ctx), captured));
        Ok(())
    }

    pub fn find_batched_decode_graph(
        &self,
        bucket_b: usize,
        long_ctx: bool,
        sample_mode: DecodeGraphSampleMode,
    ) -> Option<(u64, &CudaGraph)> {
        let long_ctx = !self.cfg.uses_qsa() && long_ctx;
        self.batched_decode_graphs
            .iter()
            .find(|(k, _, _)| *k == (bucket_b, long_ctx, sample_mode))
            .map(|(_, ptr, g)| (*ptr, g))
    }

    /// Captures a padded `B in {1, 2, 4, 8, 16}` multi-request 1-token decode CUDA Graph.
    pub fn ensure_batched_decode_graph(
        &mut self,
        bucket_b: usize,
        long_ctx: bool,
        sample_mode: DecodeGraphSampleMode,
        tp2: Option<&Tp2Collectives>,
    ) -> Result<()> {
        let long_ctx = !self.cfg.uses_qsa() && long_ctx;
        if self
            .find_batched_decode_graph(bucket_b, long_ctx, sample_mode)
            .is_some()
        {
            return Ok(());
        }
        let max_seq_len = if long_ctx {
            (self.batched_decode_io.max_blocks_per_req * 16).max(2048) as i32
        } else {
            1024
        };
        let attn_batch = W34AttentionBatch {
            query_start_loc: self.batched_decode_io.q_start_loc_ptr() as *const i32,
            context_lens_before: self.batched_decode_io.ctx_lens_before_ptr() as *const i32,
            seq_lens_after: self.batched_decode_io.seq_lens_after_ptr() as *const i32,
            block_tables: self.batched_decode_io.block_tables.addr() as *const i32,
            write_slot_mapping: self.batched_decode_io.write_slot_mapping_ptr() as *const i32,
            valid_token_mask: self.batched_decode_io.valid_token_mask_ptr() as *const u8,
            valid_req_mask: self.batched_decode_io.valid_req_mask_ptr() as *const u8,
            num_reqs: bucket_b as i32,
            total_tokens: bucket_b as i32,
            max_blocks_per_req: self.batched_decode_io.max_blocks_per_req as i32,
            max_q_len: 1,
            max_seq_len,
            mask_mode: W34_ATTN_MASK_CAUSAL,
            sliding_window: 0,
            sm_scale: (self.cfg.head_dim as f32).powf(-0.5),
            reserved: 0,
        };
        let gdn_batch = W34GdnBatch {
            query_start_loc: self.batched_decode_io.q_start_loc_ptr() as *const i32,
            state_in_indices: self.batched_decode_io.state_in_indices_ptr() as *const i32,
            state_out_indices: self.batched_decode_io.state_out_indices_ptr() as *const i32,
            spec_state_indices: core::ptr::null(),
            checkpoint_dst_indices: self.batched_decode_io.checkpoint_dst_indices_ptr()
                as *const i32,
            checkpoint_token_offsets: self.batched_decode_io.checkpoint_token_offsets_ptr()
                as *const i32,
            has_initial_state: self.batched_decode_io.has_initial_state_ptr() as *const u8,
            valid_req_mask: self.batched_decode_io.valid_req_mask_ptr() as *const u8,
            num_reqs: bucket_b as i32,
            total_tokens: bucket_b as i32,
            max_verify_tokens: 0,
            max_checkpoints_per_req: 1,
        };
        let tok_ptr = self.batched_decode_io.token_ids_ptr();
        let pos_ptr = self.batched_decode_io.positions_ptr();
        let out_ids_ptr = self.batched_decode_io.out_greedy_ids.addr();
        let selected_rows: Vec<usize> = (0..bucket_b).collect();

        self.stream.synchronize()?;
        let stream_handle = self.stream.handle;
        let dev_id = self.device.device_id;
        // SAFETY: CUDA device selection takes an integer, not a borrowed pointer.
        // Select the rank's device before operating on its stream.
        check_rc(unsafe { ffi::w34_set_device(dev_id) }, "w34_set_device")?;
        check_rc(
            // SAFETY: stream_handle is this rank's live stream, synchronized above.
            unsafe { ffi::w34_graph_begin_capture(stream_handle) },
            "w34_graph_begin_capture (batched_decode)",
        )?;
        self.is_capturing_graph = true;
        if let Some(rt) = &self.lora_runtime {
            rt.state
                .set_device_ids(self.batched_decode_io.token_lora_ids_ptr());
        }
        let emit_device_greedy = sample_mode == DecodeGraphSampleMode::PlainGreedy;
        let res = self
            .forward_selected_rows_device(
                tok_ptr,
                pos_ptr,
                bucket_b,
                &attn_batch,
                &gdn_batch,
                tp2,
                &selected_rows,
                emit_device_greedy.then_some(out_ids_ptr),
                emit_device_greedy,
            )
            .and_then(|v| {
                if sample_mode == DecodeGraphSampleMode::FastGreedy {
                    let (sampler_ws_ptr, sampler_ws_cap) =
                        self.sampler_workspace_around_last_logits(bucket_b)?;
                    let ex_sampler = self.stream.exec(sampler_ws_ptr, sampler_ws_cap);
                    check_rc(
                        // SAFETY: the engine uses FastGreedy only for TP=1 and a
                        // supported bucket <=16, so logits have the full vocabulary
                        // and the static IO fits the bucket. This method does not
                        // enforce those call-site restrictions itself. Workspace
                        // excludes logits; uploaded sparse offsets remain a caller
                        // obligation. Capture retains addresses, not &ex_sampler.
                        unsafe {
                            ffi::w34_sampler_step_fast_greedy_f16(
                                &ex_sampler,
                                self.last_logits_ptr as *const c_void,
                                self.batched_decode_io.fast_greedy_cfgs.addr()
                                    as *const ffi::W34SamplerRowCfg,
                                self.batched_decode_io.fast_greedy_sparse.addr()
                                    as *const ffi::W34SamplerSparseEntry,
                                self.batched_decode_io.fast_greedy_outs.addr()
                                    as *mut ffi::W34SamplerRowOut,
                                bucket_b as i32,
                                self.cfg.vocab_size as i32,
                            )
                        },
                        "w34_sampler_step_fast_greedy_f16 (graph capture)",
                    )?;
                }
                Ok(v)
            });
        self.is_capturing_graph = false;
        let mut graph = 0usize;
        let mut graph_exec = 0usize;
        // SAFETY: pair the successful begin with end even if the forward returned
        // Err. The stack out-parameters are live; own any returned handles before
        // propagating either Result error. A panic in forward is not handled here.
        let end_rc = unsafe {
            ffi::w34_graph_end_capture(stream_handle, &raw mut graph, &raw mut graph_exec)
        };
        let captured = CudaGraph {
            device_id: dev_id,
            graph,
            graph_exec,
        };
        res?;
        check_rc(end_rc, "w34_graph_end_capture (batched_decode)")?;
        let logits_ptr = self.last_logits_ptr;
        self.batched_decode_graphs
            .push(((bucket_b, long_ctx, sample_mode), logits_ptr, captured));
        Ok(())
    }

    pub fn sampler_workspace_around_last_logits(&self, num_rows: usize) -> Result<(u64, usize)> {
        if self.last_logits_ptr == 0 || self.last_logits_rows < num_rows {
            return Err(loader_err("missing device logits for GPU sampler"));
        }
        let arena_base = self.scratch_arena.allocation_base();
        let arena_cap = self.scratch_arena.capacity() as u64;
        let arena_end = arena_base + arena_cap;
        let reserved_rows = self.last_logits_rows.max(num_rows);
        let logits_bytes = (reserved_rows * self.local_vocab_size * 2) as u64;
        if self.last_logits_ptr >= arena_base && self.last_logits_ptr < arena_end {
            let before_end = self.last_logits_ptr & !255u64;
            let before_len = before_end.saturating_sub(arena_base);
            let after_start = (self.last_logits_ptr + logits_bytes + 255u64) & !255u64;
            let after_len = arena_end.saturating_sub(after_start);
            if after_len >= before_len {
                Ok((after_start, after_len as usize))
            } else {
                Ok((arena_base, before_len as usize))
            }
        } else {
            Ok((arena_base, arena_cap as usize))
        }
    }
}
