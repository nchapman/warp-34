//! Rank-local hybrid attention/GDN forward execution and metadata upload.

use super::*;

/// All layer phases share an overlay; outputs and GEMM workspace are reserved after it.
#[cfg(feature = "cuda")]
struct ForwardScratch {
    residual: u64,
    normed: u64,
    block: u64,
    mlp: u64,
    injection: u64,
    attention: [u64; 6],
    gdn: [u64; 5],
    ffn: [u64; 10],
    ple: [u64; 3],
    qsa: [u64; 7],
    overlay_end: u64,
}

#[cfg(feature = "cuda")]
impl Qwen3_5ModelRank {
    fn alloc_forward_scratch(&mut self, rows: usize) -> Result<ForwardScratch> {
        let h = self.cfg.hidden_size;
        let streams = if self.final_residual.is_hyper_connection() {
            self.cfg.hc_count.max(1)
        } else {
            1
        };
        let residual = self.scratch_arena.alloc_aligned(rows * streams * h * 2)?;
        let normed = self.scratch_arena.alloc_aligned(rows * h * 2)?;
        let block = self.scratch_arena.alloc_aligned(rows * h * 2)?;
        let mlp = self.scratch_arena.alloc_aligned(rows * h * 2)?;
        let injection = if self.final_residual.is_hyper_connection() {
            self.scratch_arena.alloc_aligned(rows * streams * 4)?
        } else {
            0
        };
        let (attention, gdn, ffn, ple, qsa, overlay_end) =
            alloc_layer_scratch(&mut self.scratch_arena, &self.cfg, self.tp_size, rows)?;
        Ok(ForwardScratch {
            residual,
            normed,
            block,
            mlp,
            injection,
            attention,
            gdn,
            ffn,
            ple,
            qsa,
            overlay_end,
        })
    }

    #[inline]
    fn layer_tap_target(
        &self,
        layer_idx: usize,
        num_tokens: usize,
        h_dim: usize,
    ) -> (u64, i32, i32) {
        if num_tokens <= self.max_tap_rows
            && let Some(slot) = self.tap_layer_ids.iter().position(|&id| id == layer_idx)
            && let Some(buf) = &self.tap_aux_bf16
        {
            (
                buf.addr(),
                (self.tap_layer_ids.len() * h_dim) as i32,
                (slot * h_dim) as i32,
            )
        } else {
            (0, 0, 0)
        }
    }

    fn record_final_forward_dump(
        &self,
        residual: u64,
        normed: u64,
        rows: usize,
        h: usize,
        dump: &mut ForwardDump,
    ) -> Result<()> {
        let streams = if self.final_residual.is_hyper_connection() {
            self.cfg.hc_count.max(1)
        } else {
            1
        };
        Self::record_dump_f16(
            &self.stream,
            residual,
            vec![rows, streams * h],
            &format!("layer_{}_hidden_after", self.layers.len() - 1),
            dump,
        )?;
        Self::record_dump_f16(&self.stream, normed, vec![rows, h], "final_norm", dump)?;
        if rows <= self.max_tap_rows {
            if let Some(buf) = &self.tap_aux_bf16
                && !self.tap_layer_ids.is_empty()
            {
                let cols = self.tap_layer_ids.len() * h;
                let mut host = vec![0u16; rows * cols];
                check_rc(
                    // SAFETY: tap capacity is bounded by max_tap_rows; host lives through synchronization.
                    unsafe {
                        ffi::w34_memcpy_d2h_async(
                            host.as_mut_ptr().cast(),
                            buf.addr() as *const c_void,
                            host.len() * 2,
                            self.stream.handle,
                        )
                    },
                    "record_dump_bf16 d2h",
                )?;
                self.stream.synchronize()?;
                dump.tensors_f32.insert(
                    "tap_aux_bf16".into(),
                    (
                        vec![rows, cols],
                        host.into_iter()
                            .map(|b| f32::from_bits((b as u32) << 16))
                            .collect(),
                    ),
                );
            }
            if let Some(buf) = &self.last_final_hidden_f16 {
                Self::record_dump_f16(
                    &self.stream,
                    buf.addr(),
                    vec![rows, streams * h],
                    "last_final_hidden_f16",
                    dump,
                )?;
            }
        }
        Ok(())
    }

    /// Enqueues the model and samples the last row. Device addresses and batch indices must
    /// remain valid through stream/graph completion; the caller owns their capacities.
    #[allow(clippy::too_many_arguments)]
    pub fn forward_device_batches(
        &mut self,
        token_ids_ptr: u64,
        positions_ptr: u64,
        num_tokens: usize,
        attn_batch: &W34AttentionBatch,
        gdn_batch: &W34GdnBatch,
        tp2: Option<&Tp2Collectives>,
        out_greedy_id_ptr: u64,
        out_top5_ids_ptr: u64,
        out_top5_logprobs_ptr: u64,
        mut dump_out: Option<&mut ForwardDump>,
    ) -> Result<i32> {
        self.device.make_current()?;
        self.scratch_arena.reset();
        let s = self.alloc_forward_scratch(num_tokens)?;
        let logits = self
            .scratch_arena
            .alloc_aligned(self.local_vocab_size * 2)?;
        self.last_logits_ptr = logits;
        self.last_logits_rows = 1;
        let (ws, cap) = self.scratch_arena.remaining_workspace();
        let ex = self.stream.exec(ws, cap);
        let final_out = self.final_norm_target(num_tokens, s.normed);
        self.run_layer_stack(
            &ex,
            token_ids_ptr,
            positions_ptr,
            num_tokens,
            attn_batch,
            gdn_batch,
            tp2,
            &s,
            final_out,
            dump_out.as_deref_mut(),
        )?;
        self.lm_head.forward(
            &ex,
            &self.cublas,
            self.marlin_ws.addr(),
            self.marlin_ws.len_bytes / 4,
            final_out + ((num_tokens - 1) * self.cfg.hidden_size * 2) as u64,
            logits,
            1,
        )?;
        if tp2.is_some() {
            self.sample_forward_rows(
                &ex,
                tp2,
                logits,
                1,
                out_greedy_id_ptr,
                [
                    self.decode_io.local_max_id.addr(),
                    self.decode_io.local_max_val.addr(),
                    self.decode_io.global_max_val.addr(),
                ],
            )?;
        }
        if tp2.is_none() || (out_top5_ids_ptr != 0 && out_top5_logprobs_ptr != 0) {
            check_rc(
                // SAFETY: logits is one vocab row; caller supplies five-element optional outputs.
                unsafe {
                    ffi::w34_sampler_greedy_top5_f16(
                        &ex,
                        logits as *const c_void,
                        if tp2.is_some() {
                            self.decode_io.local_max_id.addr()
                        } else {
                            out_greedy_id_ptr
                        } as *mut i32,
                        out_top5_ids_ptr as *mut i32,
                        out_top5_logprobs_ptr as *mut f32,
                        1,
                        self.local_vocab_size as i32,
                        self.vocab_offset as i32,
                    )
                },
                "w34_sampler_greedy_top5_f16 (top5)",
            )?;
        }
        if let Some(dump) = dump_out
            && self.rank == 0
        {
            Self::record_dump_f16(
                &self.stream,
                logits,
                vec![self.local_vocab_size],
                "logits_last",
                dump,
            )?;
        }
        Ok(0)
    }

    fn final_norm_target(&self, rows: usize, fallback: u64) -> u64 {
        if !self.final_residual.is_hyper_connection()
            && rows <= self.max_tap_rows
            && let Some(buf) = &self.last_final_hidden_f16
        {
            buf.addr()
        } else {
            fallback
        }
    }

    fn sample_forward_rows(
        &self,
        ex: &W34Exec,
        tp2: Option<&Tp2Collectives>,
        logits: u64,
        rows: usize,
        output: u64,
        [local_ids, local_vals, global_vals]: [u64; 3],
    ) -> Result<()> {
        check_rc(
            // SAFETY: caller reserves rows logits and ID/value outputs; null top IDs selects local argmax.
            unsafe {
                ffi::w34_sampler_greedy_top5_f16(
                    ex,
                    logits as *const c_void,
                    if tp2.is_some() { local_ids } else { output } as *mut i32,
                    core::ptr::null_mut(),
                    if tp2.is_some() {
                        local_vals as *mut f32
                    } else {
                        core::ptr::null_mut()
                    },
                    rows as i32,
                    self.local_vocab_size as i32,
                    self.vocab_offset as i32,
                )
            },
            "w34_sampler_greedy_top5_f16",
        )?;
        if let Some(comm) = tp2 {
            comm.all_gather_top1_f32_i32(
                ex,
                local_vals,
                local_ids,
                global_vals,
                output,
                self.rank as i32,
                rows as i32,
            )?;
        }
        Ok(())
    }

    /// Evaluates the stack and returns one greedy ID per selected row.
    #[allow(clippy::too_many_arguments)]
    pub fn forward_selected_rows(
        &mut self,
        token_ids_ptr: u64,
        positions_ptr: u64,
        num_tokens: usize,
        attn_batch: &W34AttentionBatch,
        gdn_batch: &W34GdnBatch,
        tp2: Option<&Tp2Collectives>,
        selected_rows: &[usize],
    ) -> Result<Vec<i32>> {
        self.forward_selected_rows_maybe_skip_greedy(
            token_ids_ptr,
            positions_ptr,
            num_tokens,
            attn_batch,
            gdn_batch,
            tp2,
            selected_rows,
            false,
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub(super) fn forward_selected_rows_maybe_skip_greedy(
        &mut self,
        token_ids_ptr: u64,
        positions_ptr: u64,
        num_tokens: usize,
        attn_batch: &W34AttentionBatch,
        gdn_batch: &W34GdnBatch,
        tp2: Option<&Tp2Collectives>,
        selected_rows: &[usize],
        skip_host_greedy: bool,
    ) -> Result<Vec<i32>> {
        if skip_host_greedy && !selected_rows.is_empty() {
            let (_, use_spec) = self.forward_selected_rows_device(
                token_ids_ptr,
                positions_ptr,
                num_tokens,
                attn_batch,
                gdn_batch,
                tp2,
                selected_rows,
                None,
                false,
            )?;
            self.speculative_logits_rows = if use_spec { selected_rows.len() } else { 0 };
            return Ok(Vec::new());
        }
        let (out, use_spec) = self.forward_selected_rows_device(
            token_ids_ptr,
            positions_ptr,
            num_tokens,
            attn_batch,
            gdn_batch,
            tp2,
            selected_rows,
            None,
            !selected_rows.is_empty(),
        )?;
        let mut ids = vec![0i32; selected_rows.len()];
        if !ids.is_empty() {
            check_rc(
                // SAFETY: device output and host vector both contain selected_rows.len() i32s.
                unsafe {
                    ffi::w34_memcpy_d2h_async(
                        ids.as_mut_ptr().cast(),
                        out as *const c_void,
                        ids.len() * 4,
                        self.stream.handle,
                    )
                },
                "copy selected greedy ids",
            )?;
        }
        self.stream.synchronize()?;
        self.speculative_logits_rows = if use_spec { selected_rows.len() } else { 0 };
        Ok(ids)
    }

    /// Device-only selected-row core; no synchronization unless tiered host dispatch requires it.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn forward_selected_rows_device(
        &mut self,
        token_ids_ptr: u64,
        positions_ptr: u64,
        num_tokens: usize,
        attn_batch: &W34AttentionBatch,
        gdn_batch: &W34GdnBatch,
        tp2: Option<&Tp2Collectives>,
        selected_rows: &[usize],
        out_greedy_ids_override: Option<u64>,
        emit_device_greedy: bool,
    ) -> Result<(u64, bool)> {
        if selected_rows.iter().any(|&r| r >= num_tokens) {
            return Err(loader_err("selected row outside token batch"));
        }
        self.speculative_logits_rows = 0;
        self.device.make_current()?;
        self.scratch_arena.reset();
        let h = self.cfg.hidden_size;
        let s = self.alloc_forward_scratch(num_tokens)?;
        let n = selected_rows.len();
        let use_spec = n > 0
            && self
                .speculative_logits
                .as_ref()
                .is_some_and(|b| n <= b.len_bytes / (self.local_vocab_size * 2));
        let logits = if n == 0 {
            0
        } else if use_spec {
            self.speculative_logits.as_ref().unwrap().addr()
        } else {
            self.scratch_arena
                .alloc_aligned(n * self.local_vocab_size * 2)?
        };
        let contiguous = n > 0
            && selected_rows
                .iter()
                .enumerate()
                .all(|(i, &r)| r == selected_rows[0] + i);
        let gathered = if n > 0 && !contiguous {
            self.scratch_arena.alloc_aligned(n * h * 2)?
        } else {
            0
        };
        let output = if emit_device_greedy && n > 0 {
            if let Some(out) = out_greedy_ids_override {
                out
            } else {
                self.scratch_arena.alloc_aligned(n * 4)?
            }
        } else {
            0
        };
        let (local_ids, local_vals, global_vals) = if emit_device_greedy && n > 0 && tp2.is_some() {
            (
                self.scratch_arena.alloc_aligned(n * 4)?,
                self.scratch_arena.alloc_aligned(n * 4)?,
                self.scratch_arena.alloc_aligned(n * 4)?,
            )
        } else {
            (0, 0, 0)
        };
        let (ws, cap) = self.scratch_arena.remaining_workspace();
        let ex = self.stream.exec(ws, cap);
        let final_out = self.final_norm_target(num_tokens, s.normed);
        self.run_layer_stack(
            &ex,
            token_ids_ptr,
            positions_ptr,
            num_tokens,
            attn_batch,
            gdn_batch,
            tp2,
            &s,
            final_out,
            None,
        )?;
        self.last_logits_ptr = logits;
        self.last_logits_rows = n;
        if n == 0 {
            return Ok((0, false));
        }
        let hidden = if contiguous {
            final_out + (selected_rows[0] * h * 2) as u64
        } else {
            for (i, &row) in selected_rows.iter().enumerate() {
                check_rc(
                    // SAFETY: rows were checked above and destination reserves n disjoint hidden rows.
                    unsafe {
                        ffi::w34_memcpy_d2d_async(
                            (gathered + (i * h * 2) as u64) as *mut c_void,
                            (final_out + (row * h * 2) as u64) as *const c_void,
                            h * 2,
                            self.stream.handle,
                        )
                    },
                    "gather selected hidden rows",
                )?;
            }
            gathered
        };
        self.lm_head.forward(
            &ex,
            &self.cublas,
            self.marlin_ws.addr(),
            self.marlin_ws.len_bytes / 4,
            hidden,
            logits,
            n,
        )?;
        if emit_device_greedy {
            self.sample_forward_rows(
                &ex,
                tp2,
                logits,
                n,
                output,
                [local_ids, local_vals, global_vals],
            )?;
        }
        Ok((output, use_spec))
    }

    /// The one model spine. Residual policy, PLE, QSA and tiered FFN are independent capabilities.
    #[allow(clippy::too_many_arguments, clippy::cognitive_complexity)]
    fn run_layer_stack(
        &mut self,
        ex: &W34Exec,
        token_ids: u64,
        positions: u64,
        rows: usize,
        attn: &W34AttentionBatch,
        gdn_batch: &W34GdnBatch,
        tp2: Option<&Tp2Collectives>,
        s: &ForwardScratch,
        final_out: u64,
        mut dump_out: Option<&mut ForwardDump>,
    ) -> Result<()> {
        let h = self.cfg.hidden_size;
        let hc = self.final_residual.is_hyper_connection();
        let streams = if hc { self.cfg.hc_count.max(1) } else { 1 };
        let lowrank = self.cfg.hc_lowrank;
        let eps = self.cfg.rms_norm_eps;
        let mw = self.marlin_ws.addr();
        let mi = self.marlin_ws.len_bytes / 4;
        let embed = if hc { s.normed } else { s.residual };
        check_rc(
            // SAFETY: embed reserves rows * h FP16 elements; caller supplies valid token IDs.
            unsafe {
                ffi::w34_embed_lookup_f16(
                    ex,
                    token_ids as *const i32,
                    self.embed_tokens_ptr as *const c_void,
                    embed as *mut c_void,
                    core::ptr::null(),
                    rows as i32,
                    h as i32,
                    self.cfg.vocab_size as i32,
                )
            },
            "w34_embed_lookup_f16",
        )?;
        if let Some((vis, indices, count)) = self.pending_visual_splice
            && count > 0
        {
            check_rc(
                // SAFETY: vision path keeps the embeddings and destination indices live through forward.
                unsafe {
                    ffi::w34_embedding_splice_f16(
                        ex,
                        embed as *mut c_void,
                        vis as *const c_void,
                        indices as *const i32,
                        count as i32,
                        h as i32,
                    )
                },
                "w34_embedding_splice_f16",
            )?;
        }
        if let Some(dump) = dump_out.as_deref_mut()
            && self.rank == 0
        {
            Self::record_dump_f16(&self.stream, embed, vec![rows, h], "embed", dump)?;
        }
        self.final_residual
            .init_streams(ex, embed, s.residual, rows, streams, h)?;
        let [pa, pb, pv] = s.ple;
        let [
            gu,
            act,
            router,
            topw,
            topids,
            interm,
            shgu,
            shact,
            shout,
            shgate,
        ] = s.ffn;
        let ple_staging = self
            .ple_runtime
            .as_ref()
            .map_or(0, |p| p.f8_staging_dev.addr());
        let ple_state = self
            .state_pool
            .ple_conv_state
            .as_ref()
            .map_or(0, |b| b.addr());
        for layer_idx in 0..self.layers.len() {
            let layer = &self.layers[layer_idx];
            if let Some(ple) = &layer.ple {
                ple.forward(
                    ex,
                    &self.cublas,
                    mw,
                    mi,
                    ple_staging,
                    s.residual,
                    s.normed,
                    pa,
                    pb,
                    pv,
                    ple_state,
                    gdn_batch,
                    rows,
                    streams,
                    h,
                    eps,
                )?;
            }
            // PLE mutates the residual: a single-stream layer must then redo the input norm,
            // rather than consume the norm fused by its predecessor before that mutation.
            let norm_layer_idx = if layer.ple.is_some() { 0 } else { layer_idx };
            let hc_ctx = super::HcStepContext {
                ex,
                cublas: &self.cublas,
                marlin_ws_ptr: mw,
                marlin_ws_ints: mi,
                residual_buf: s.residual,
                hc_residual_buf: s.residual,
                hc_inj_weights_buf: s.injection,
                ple_bufs: [pa, pb, pv],
                num_tokens: rows,
                hc_count: streams,
                hc_lowrank: lowrank,
                h_dim: h,
                eps,
            };
            let pre_attn_targets = layer.block.pre_attn_norm_targets(s.normed, ex, rows, h);
            layer
                .residual
                .pre_attn_with_targets(norm_layer_idx, &hc_ctx, pre_attn_targets)?;
            self.run_attention_block(
                ex,
                layer_idx,
                positions,
                rows,
                attn,
                gdn_batch,
                s,
                pre_attn_targets,
            )?;
            let pre_mlp_targets = layer.ffn.pre_mlp_norm_targets(s.normed, s.mlp, ex, rows, h);
            layer.residual.post_attn_pre_mlp_with_targets(
                &hc_ctx,
                tp2,
                self.rank,
                s.block,
                pre_mlp_targets,
                dump_out.is_some(),
            )?;
            if let Some(dump) = dump_out.as_deref_mut()
                && self.rank == 0
            {
                Self::record_dump_f16(
                    &self.stream,
                    s.block,
                    vec![rows, h],
                    &format!("layer_{layer_idx}_attn_or_gdn_out"),
                    dump,
                )?;
            }
            layer.ffn.forward_with_prerotated_input(
                ex,
                &self.cublas,
                mw,
                mi,
                s.normed,
                s.mlp,
                rows,
                h,
                self.cfg.intermediate_size / self.tp_size,
                gu,
                act,
                router,
                topw,
                topids,
                interm,
                shgu,
                shact,
                shout,
                shgate,
                pre_mlp_targets[1],
            )?;
            layer.ffn.execute_tiered(
                self.tiered_moe.as_mut(),
                &self.stream,
                self.is_capturing_graph,
                if pre_mlp_targets[1] != 0 {
                    pre_mlp_targets[1]
                } else {
                    s.normed
                },
                [topids, topw, shout, shgate, s.mlp],
                rows,
            )?;
            let [next_weight, next_out, next_had64] =
                if let Some(next) = self.layers.get(layer_idx + 1) {
                    let w = next.residual.single_stream_input_norm_ptr().unwrap_or(0);
                    let [no, nh] = if next.ple.is_some() {
                        [s.normed, 0]
                    } else {
                        next.block.pre_attn_norm_targets(s.normed, ex, rows, h)
                    };
                    [w, no, nh]
                } else {
                    [
                        self.final_residual.single_stream_norm_ptr().unwrap_or(0),
                        final_out,
                        0,
                    ]
                };
            let (dst, stride, offset) = self.layer_tap_target(layer_idx, rows, h);
            self.layers[layer_idx].residual.post_mlp_with_targets(
                &hc_ctx,
                tp2,
                self.rank,
                s.mlp,
                [next_weight, next_out, next_had64],
                super::TapExport {
                    dst,
                    stride,
                    offset,
                },
            )?;
            if layer_idx + 1 < self.layers.len()
                && let Some(dump) = dump_out.as_deref_mut()
                && self.rank == 0
            {
                Self::record_dump_f16(
                    &self.stream,
                    s.residual,
                    vec![rows, streams * h],
                    &format!("layer_{layer_idx}_hidden_after"),
                    dump,
                )?;
            }
        }
        let final_hc_ctx = super::HcStepContext {
            ex,
            cublas: &self.cublas,
            marlin_ws_ptr: mw,
            marlin_ws_ints: mi,
            residual_buf: s.residual,
            hc_residual_buf: s.residual,
            hc_inj_weights_buf: s.injection,
            ple_bufs: [pa, pb, pv],
            num_tokens: rows,
            hc_count: streams,
            hc_lowrank: lowrank,
            h_dim: h,
            eps,
        };
        self.final_residual.finish_stack(
            &final_hc_ctx,
            &self.stream,
            final_out,
            if rows <= self.max_tap_rows {
                self.last_final_hidden_f16.as_ref()
            } else {
                None
            },
        )?;
        self.last_norm_hidden_ptr = final_out;
        self.last_norm_hidden_rows = rows;
        if let Some(dump) = dump_out
            && self.rank == 0
        {
            self.record_final_forward_dump(s.residual, final_out, rows, h, dump)?;
        }
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    fn run_attention_block(
        &self,
        ex: &W34Exec,
        layer_idx: usize,
        positions: u64,
        rows: usize,
        attn: &W34AttentionBatch,
        batch: &W34GdnBatch,
        s: &ForwardScratch,
        [_pre_attn_out, pre_attn_had64]: [u64; 2],
    ) -> Result<()> {
        let mw = self.marlin_ws.addr();
        let mi = self.marlin_ws.len_bytes / 4;
        let eps = self.cfg.rms_norm_eps;
        match &self.layers[layer_idx].block {
            LayerBlockWeights::FullAttention(fa) => {
                let [qkv, q, k, gate, out, gated] = s.attention;
                let qh = self.cfg.num_attention_heads / self.tp_size;
                let kh = self.cfg.local_kv_heads(self.tp_size);
                let hd = self.cfg.head_dim;
                let qgd = qh * hd * 2;
                let kd = kh * hd;
                let stride = (qgd + 2 * kd) as i32;
                if pre_attn_had64 != 0 {
                    fa.qkv_proj.forward_prerotated(
                        ex,
                        &self.cublas,
                        mw,
                        mi,
                        pre_attn_had64,
                        qkv,
                        rows,
                    )?;
                } else {
                    fa.qkv_proj
                        .forward(ex, &self.cublas, mw, mi, s.normed, qkv, rows)?;
                }
                check_rc(
                    // SAFETY: packed QKV and separate Q/K/gate buffers match the local head geometry.
                    // Caller supplies position planes according to current_pos_plane_stride.
                    unsafe {
                        ffi::w34_fused_qk_rmsnorm_rope_gate_strided_f16(
                            ex,
                            qkv as *const c_void,
                            (qkv + (qgd * 2) as u64) as *const c_void,
                            fa.q_norm_ptr as *const c_void,
                            fa.k_norm_ptr as *const c_void,
                            positions as *const i32,
                            core::ptr::null(),
                            q as *mut c_void,
                            k as *mut c_void,
                            gate as *mut c_void,
                            rows as i32,
                            self.current_pos_plane_stride,
                            qh as i32,
                            kh as i32,
                            hd as i32,
                            self.cfg.rotary_dim as i32,
                            self.cfg.mrope_section[0] as i32,
                            self.cfg.mrope_section[1] as i32,
                            self.cfg.mrope_section[2] as i32,
                            stride,
                            stride,
                            self.cfg.rope_theta,
                            eps,
                        )
                    },
                    "w34_fused_qk_rmsnorm_rope_gate_strided_f16",
                )?;
                let layout = &self.kv_pool.layers[fa.kv_layer_ordinal].layout;
                check_rc(
                    // SAFETY: K is contiguous, V is the packed projection suffix; batch slots index the owned KV pool.
                    unsafe {
                        ffi::w34_kv_cache_write_strided(
                            ex,
                            layout,
                            attn,
                            k as *const c_void,
                            (qkv + ((qgd + kd) * 2) as u64) as *const c_void,
                            kd as i32,
                            stride,
                        )
                    },
                    "w34_kv_cache_write_strided",
                )?;
                let fuse_o_h64 = fa.o_proj.can_fuse_h64_from_producer();
                if let Some(indexer) = &fa.qsa_indexer {
                    fa.forward_qsa_attention(
                        indexer,
                        ex,
                        &self.cublas,
                        mw,
                        mi,
                        &self.cfg,
                        &self.kv_pool,
                        layout,
                        attn,
                        s.normed,
                        q,
                        gate,
                        gated,
                        s.qsa,
                        positions,
                        self.current_pos_plane_stride,
                        rows,
                        qh,
                        false,
                        fuse_o_h64,
                    )?;
                } else {
                    check_rc(
                        // SAFETY: Q/output are local-Q-sized rows; KV writes precede attention on this stream.
                        unsafe {
                            ffi::w34_paged_attention_f16(
                                ex,
                                layout,
                                attn,
                                q as *const c_void,
                                out as *mut c_void,
                                qh as i32,
                            )
                        },
                        "w34_paged_attention_f16",
                    )?;
                    let gate_fn = if fuse_o_h64 {
                        ffi::w34_apply_attn_output_gate_hadamard_f16
                    } else {
                        ffi::w34_apply_attn_output_gate_f16
                    };
                    check_rc(
                        // SAFETY: attention, gate and destination are distinct local-Q-sized overlay buffers.
                        unsafe {
                            gate_fn(
                                ex,
                                out as *const c_void,
                                gate as *const c_void,
                                gated as *mut c_void,
                                core::ptr::null(),
                                rows as i32,
                                (qh * hd) as i32,
                            )
                        },
                        "w34_apply_attn_output_gate_f16",
                    )?;
                }
                if fuse_o_h64 {
                    fa.o_proj
                        .forward_prerotated(ex, &self.cublas, mw, mi, gated, s.block, rows)?;
                } else {
                    fa.o_proj
                        .forward(ex, &self.cublas, mw, mi, gated, s.block, rows)?;
                }
            }
            LayerBlockWeights::LinearAttention(gdn) => {
                let [qkvz, ba, conv, raw, normed] = s.gdn;
                let kh = self.cfg.linear_num_key_heads / self.tp_size;
                let vh = self.cfg.linear_num_value_heads / self.tp_size;
                let kd = self.cfg.linear_key_head_dim;
                let vd = self.cfg.linear_value_head_dim;
                let cd = 2 * kh * kd + vh * vd;
                let stride = (cd + vh * vd) as i32;
                // cuBLAS uses the main stream and Hadamard uses ex.scratch_ptr, so only custom
                // non-Hadamard small-N/Marlin projections may fork onto aux_stream.
                let fork = gdn.in_proj_ba.can_fork_on_aux_stream();
                if fork {
                    self.aux_fork_event.record_in_current_ctx(&self.stream)?;
                    self.aux_stream.wait_event(&self.aux_fork_event)?;
                    let aux = self.aux_stream.exec(0, 0);
                    gdn.in_proj_ba.forward(
                        &aux,
                        &self.cublas,
                        self.aux_marlin_ws.addr(),
                        self.aux_marlin_ws.len_bytes / 4,
                        s.normed,
                        ba,
                        rows,
                    )?;
                    self.aux_join_event
                        .record_in_current_ctx(&self.aux_stream)?;
                }
                if pre_attn_had64 != 0 {
                    gdn.in_proj_qkvz.forward_prerotated(
                        ex,
                        &self.cublas,
                        mw,
                        mi,
                        pre_attn_had64,
                        qkvz,
                        rows,
                    )?;
                } else {
                    gdn.in_proj_qkvz
                        .forward(ex, &self.cublas, mw, mi, s.normed, qkvz, rows)?;
                }
                if !fork {
                    gdn.in_proj_ba
                        .forward(ex, &self.cublas, mw, mi, s.normed, ba, rows)?;
                }
                let state = &self.state_pool.layers[gdn.gdn_layer_ordinal];
                check_rc(
                    // SAFETY: QKV is the cd prefix of each row; state indices are supplied by the batch caller.
                    unsafe {
                        ffi::w34_gdn_causal_conv1d_strided_f16(
                            ex,
                            batch,
                            qkvz as *const c_void,
                            gdn.conv_weight_ptr as *const c_void,
                            core::ptr::null(),
                            state.conv_state.addr() as *mut c_void,
                            conv as *mut c_void,
                            cd as i32,
                            stride,
                            self.cfg.linear_conv_kernel_dim as i32,
                            self.state_pool.total_slots as i32,
                            1,
                        )
                    },
                    "w34_gdn_causal_conv1d_strided_f16",
                )?;
                if fork {
                    self.stream.wait_event(&self.aux_join_event)?;
                }
                // Never borrow the overlay tail while persistent HC/PLE/QSA data can be live.
                let tail = if self.cfg.uses_hyper_connection()
                    || self.cfg.uses_ple()
                    || self.cfg.uses_qsa()
                {
                    0
                } else {
                    s.overlay_end.saturating_sub(normed) as usize
                };
                let recurrent_ex = if tail > ex.scratch_size {
                    self.stream.exec(normed, tail)
                } else {
                    *ex
                };
                check_rc(
                    // SAFETY: BA writer is joined; conv/raw/state match local heads. Borrowed tail expires before normed is written.
                    unsafe {
                        ffi::w34_gdn_recurrent_strided_f16(
                            &recurrent_ex,
                            batch,
                            conv as *const c_void,
                            core::ptr::null(),
                            core::ptr::null(),
                            core::ptr::null(),
                            ba as *const c_void,
                            (ba + (vh * 2) as u64) as *const c_void,
                            gdn.a_log_ptr as *const f32,
                            gdn.dt_bias_ptr as *const f32,
                            state.ssm_state.addr() as *mut f32,
                            raw as *mut c_void,
                            kh as i32,
                            kd as i32,
                            vh as i32,
                            vd as i32,
                            (2 * vh) as i32,
                            self.state_pool.total_slots as i32,
                        )
                    },
                    "w34_gdn_recurrent_strided_f16",
                )?;
                const GDN_FUSED_GATED_NORM_HEAD_DIM: usize = 128;
                let fuse_out_h64 = vd == GDN_FUSED_GATED_NORM_HEAD_DIM
                    && stride % 4 == 0
                    && gdn.out_proj.can_fuse_h64_from_producer();
                let gated_norm = match (
                    self.cfg.output_gate_type.eq_ignore_ascii_case("sigmoid"),
                    fuse_out_h64,
                ) {
                    (true, true) => ffi::w34_rms_norm_gated_sigmoid_strided_hadamard_f16,
                    (true, false) => ffi::w34_rms_norm_gated_sigmoid_strided_f16,
                    (false, true) => ffi::w34_rms_norm_gated_silu_strided_hadamard_f16,
                    (false, false) => ffi::w34_rms_norm_gated_silu_strided_f16,
                };
                check_rc(
                    // SAFETY: Z is the suffix of QKVZ; raw/normed contain rows * vh * vd elements.
                    unsafe {
                        gated_norm(
                            ex,
                            raw as *const c_void,
                            (qkvz + (cd * 2) as u64) as *const c_void,
                            gdn.norm_weight_ptr as *const c_void,
                            normed as *mut c_void,
                            core::ptr::null(),
                            rows as i32,
                            vh as i32,
                            vd as i32,
                            stride,
                            eps,
                        )
                    },
                    "w34_rms_norm_gated_strided_f16",
                )?;
                if fuse_out_h64 {
                    gdn.out_proj.forward_prerotated(
                        ex,
                        &self.cublas,
                        mw,
                        mi,
                        normed,
                        s.block,
                        rows,
                    )?;
                } else {
                    gdn.out_proj
                        .forward(ex, &self.cublas, mw, mi, normed, s.block, rows)?;
                }
            }
        }
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    pub(super) fn upload_batch_meta(
        &mut self,
        flat_tokens: &[i32],
        flat_positions: &[i32],
        q_start_loc: &[i32],
        ctx_lens_before: &[i32],
        seq_lens_after: &[i32],
        block_tables: &[i32],
        flat_write_slots: &[i32],
        state_in_indices: &[i32],
        state_out_indices: &[i32],
        has_initial_state: &[u8],
        spec_state_indices: &[i32],
        checkpoint_dst_indices: &[i32],
        checkpoint_token_offsets: &[i32],
        num_reqs: usize,
        total_tokens: usize,
        max_blocks_per_req: usize,
        max_q_len: usize,
        max_seq_len: usize,
        max_verify_tokens: usize,
        max_checkpoints_per_req: usize,
    ) -> Result<(u64, u64, W34AttentionBatch, W34GdnBatch)> {
        self.device.make_current()?;
        if self.cfg.uses_ple() {
            self.stage_ple_f8_for_batch(
                flat_tokens,
                q_start_loc,
                state_in_indices,
                state_out_indices,
                has_initial_state,
                spec_state_indices,
                max_verify_tokens,
                checkpoint_dst_indices,
                checkpoint_token_offsets,
                max_checkpoints_per_req,
            )?;
        }
        self.batch_meta_arena.reset();
        let total_i32 = flat_tokens.len()
            + flat_positions.len()
            + q_start_loc.len()
            + ctx_lens_before.len()
            + seq_lens_after.len()
            + block_tables.len()
            + flat_write_slots.len()
            + state_in_indices.len()
            + state_out_indices.len()
            + spec_state_indices.len()
            + checkpoint_dst_indices.len()
            + checkpoint_token_offsets.len();
        let mut packed = Vec::with_capacity(total_i32);
        let push_slice = |v: &mut Vec<i32>, s: &[i32]| -> usize {
            let off = v.len();
            v.extend_from_slice(s);
            off
        };
        let off_tok = push_slice(&mut packed, flat_tokens);
        let off_pos = push_slice(&mut packed, flat_positions);
        let off_qs = push_slice(&mut packed, q_start_loc);
        let off_cb = push_slice(&mut packed, ctx_lens_before);
        let off_sa = push_slice(&mut packed, seq_lens_after);
        let off_bt = push_slice(&mut packed, block_tables);
        let off_ws = push_slice(&mut packed, flat_write_slots);
        let off_si = push_slice(&mut packed, state_in_indices);
        let off_so = push_slice(&mut packed, state_out_indices);
        let off_sp = push_slice(&mut packed, spec_state_indices);
        let off_cd = push_slice(&mut packed, checkpoint_dst_indices);
        let off_co = push_slice(&mut packed, checkpoint_token_offsets);
        let base = self.batch_meta_arena.upload_slice(&packed, &self.stream)?;
        let ptr_at = |off: usize| -> u64 { base + (off * 4) as u64 };
        let has_init_ptr = self
            .batch_meta_arena
            .upload_slice(has_initial_state, &self.stream)?;
        let attn_batch = W34AttentionBatch {
            query_start_loc: ptr_at(off_qs) as *const i32,
            context_lens_before: ptr_at(off_cb) as *const i32,
            seq_lens_after: ptr_at(off_sa) as *const i32,
            block_tables: ptr_at(off_bt) as *const i32,
            write_slot_mapping: ptr_at(off_ws) as *const i32,
            valid_token_mask: core::ptr::null(),
            valid_req_mask: core::ptr::null(),
            num_reqs: num_reqs as i32,
            total_tokens: total_tokens as i32,
            max_blocks_per_req: max_blocks_per_req as i32,
            max_q_len: max_q_len as i32,
            max_seq_len: max_seq_len as i32,
            mask_mode: W34_ATTN_MASK_CAUSAL,
            sliding_window: 0,
            sm_scale: (self.cfg.head_dim as f32).powf(-0.5),
            reserved: 0,
        };
        let gdn_batch = W34GdnBatch {
            query_start_loc: ptr_at(off_qs) as *const i32,
            state_in_indices: ptr_at(off_si) as *const i32,
            state_out_indices: ptr_at(off_so) as *const i32,
            spec_state_indices: if spec_state_indices.is_empty() {
                core::ptr::null()
            } else {
                ptr_at(off_sp) as *const i32
            },
            checkpoint_dst_indices: if checkpoint_dst_indices.is_empty() {
                core::ptr::null()
            } else {
                ptr_at(off_cd) as *const i32
            },
            checkpoint_token_offsets: if checkpoint_token_offsets.is_empty() {
                core::ptr::null()
            } else {
                ptr_at(off_co) as *const i32
            },
            has_initial_state: has_init_ptr as *const u8,
            valid_req_mask: core::ptr::null(),
            num_reqs: num_reqs as i32,
            total_tokens: total_tokens as i32,
            max_verify_tokens: max_verify_tokens as i32,
            max_checkpoints_per_req: max_checkpoints_per_req as i32,
        };
        Ok((ptr_at(off_tok), ptr_at(off_pos), attn_batch, gdn_batch))
    }

    fn record_dump_f16(
        stream: &CudaStream,
        dev_ptr: u64,
        shape: Vec<usize>,
        name: &str,
        dump: &mut ForwardDump,
    ) -> Result<()> {
        let count: usize = shape.iter().copied().product();
        let mut host = vec![0u16; count];
        check_rc(
            // SAFETY: caller supplies a live region matching shape; host remains live through synchronization.
            unsafe {
                ffi::w34_memcpy_d2h_async(
                    host.as_mut_ptr().cast(),
                    dev_ptr as *const c_void,
                    count * 2,
                    stream.handle,
                )
            },
            "record_dump_f16 d2h",
        )?;
        stream.synchronize()?;
        dump.tensors_f32.insert(
            name.to_string(),
            (shape, host.into_iter().map(f16_to_f32).collect()),
        );
        Ok(())
    }
}
