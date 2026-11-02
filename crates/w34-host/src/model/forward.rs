//! Rank-local hybrid attention/GDN forward execution and metadata upload.

use super::*;

#[cfg(feature = "cuda")]
impl Qwen3_5ModelRank {
    #[inline]
    fn layer_tap_target(
        &self,
        layer_idx: usize,
        num_tokens: usize,
        h_dim: usize,
    ) -> (u64, i32, i32) {
        if num_tokens <= self.max_tap_rows
            && let Some(tap_slot) = self.tap_layer_ids.iter().position(|&id| id == layer_idx)
            && let Some(tap_buf) = &self.tap_aux_bf16
        {
            let num_taps = self.tap_layer_ids.len();
            (
                tap_buf.addr(),
                (num_taps * h_dim) as i32,
                (tap_slot * h_dim) as i32,
            )
        } else {
            (0, 0, 0)
        }
    }

    fn record_final_forward_dump(
        &self,
        residual_buf: u64,
        final_norm_out_buf: u64,
        num_tokens: usize,
        h_dim: usize,
        dump: &mut ForwardDump,
    ) -> Result<()> {
        Self::record_dump_f16(
            &self.stream,
            residual_buf,
            vec![num_tokens, h_dim],
            &format!("layer_{}_hidden_after", self.layers.len() - 1),
            dump,
        )?;
        Self::record_dump_f16(
            &self.stream,
            final_norm_out_buf,
            vec![num_tokens, h_dim],
            "final_norm",
            dump,
        )?;
        if num_tokens <= self.max_tap_rows {
            if let Some(tap_buf) = &self.tap_aux_bf16
                && !self.tap_layer_ids.is_empty()
            {
                let tap_cols = self.tap_layer_ids.len() * h_dim;
                let count = num_tokens * tap_cols;
                let mut host_u16 = vec![0u16; count];
                check_rc(
                    // SAFETY: count is bounded by max_tap_rows and configured
                    // tap columns. The host vector has count u16 elements and
                    // is kept alive until the stream synchronization below.
                    unsafe {
                        ffi::w34_memcpy_d2h_async(
                            host_u16.as_mut_ptr() as *mut c_void,
                            tap_buf.addr() as *const c_void,
                            count * 2,
                            self.stream.handle,
                        )
                    },
                    "record_dump_bf16 d2h",
                )?;
                self.stream.synchronize()?;
                let vals_f32: Vec<f32> = host_u16
                    .into_iter()
                    .map(|b| f32::from_bits((b as u32) << 16))
                    .collect();
                dump.tensors_f32.insert(
                    "tap_aux_bf16".to_string(),
                    (vec![num_tokens, tap_cols], vals_f32),
                );
            }
            if let Some(final_h_buf) = &self.last_final_hidden_f16 {
                let final_cols = if self.cfg.is_qwen4_exp() {
                    self.cfg.hc_count.max(1) * h_dim
                } else {
                    h_dim
                };
                Self::record_dump_f16(
                    &self.stream,
                    final_h_buf.addr(),
                    vec![num_tokens, final_cols],
                    "last_final_hidden_f16",
                    dump,
                )?;
            }
        }
        Ok(())
    }

    /// Executes the full model spine for `num_tokens` tokens on this rank.
    ///
    /// Raw-address seam: the engine supplies live device token/position buffers,
    /// batch descriptors indexing its KV/state pools, and appropriately sized
    /// outputs on this rank. This API does not validate arbitrary addresses or
    /// device-side indices. They must remain live through stream/graph completion.
    #[allow(clippy::too_many_arguments, clippy::cognitive_complexity)]
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
        self.validate_tap_rows(num_tokens)?;
        self.device.make_current()?;
        self.scratch_arena.reset();

        let h_dim = self.cfg.hidden_size;
        let local_q_heads = self.cfg.num_attention_heads / self.tp_size;
        let local_kv_heads = self.cfg.num_key_value_heads / self.tp_size;
        let head_dim = self.cfg.head_dim;
        let local_k_heads = self.cfg.linear_num_key_heads / self.tp_size;
        let local_v_heads = self.cfg.linear_num_value_heads / self.tp_size;
        let head_k_dim = self.cfg.linear_key_head_dim;
        let head_v_dim = self.cfg.linear_value_head_dim;
        let conv_dim = 2 * local_k_heads * head_k_dim + local_v_heads * head_v_dim;
        let local_interm = self.cfg.intermediate_size / self.tp_size;
        let marlin_ws_ptr = self.marlin_ws.addr();
        let marlin_ws_ints = self.marlin_ws.len_bytes / 4;

        let hc_mult = self.cfg.hc_count.max(1);
        let residual_buf = self
            .scratch_arena
            .alloc_aligned(num_tokens * hc_mult * h_dim * 2)?;
        let normed_buf = self.scratch_arena.alloc_aligned(num_tokens * h_dim * 2)?;
        let block_out_buf = self.scratch_arena.alloc_aligned(num_tokens * h_dim * 2)?;
        let mlp_out_buf = self.scratch_arena.alloc_aligned(num_tokens * h_dim * 2)?;
        let hc_inj_weights_buf = if self.cfg.is_qwen4_exp() {
            self.scratch_arena.alloc_aligned(num_tokens * hc_mult * 4)?
        } else {
            0
        };

        let q_gate_dim = local_q_heads * head_dim * 2;
        let kv_dim = local_kv_heads * head_dim;
        let gdn_v_dim = local_v_heads * head_v_dim;

        let (
            [
                qkv_proj_buf,
                q_buf,
                k_buf,
                gate_buf,
                attn_out_buf,
                gated_attn_buf,
            ],
            [
                qkvz_buf,
                ba_buf,
                qkv_conv_out_buf,
                gdn_raw_out_buf,
                gdn_normed_buf,
            ],
            [
                gate_up_buf,
                act_buf,
                moe_router_logits_buf,
                moe_topk_weights_buf,
                moe_topk_ids_buf,
                moe_interm_buf,
                moe_sh_gu_buf,
                moe_sh_act_buf,
                moe_sh_out_buf,
                moe_sh_gate_buf,
            ],
            ple_hc_addrs,
            qsa_addrs,
            layer_overlay_end,
        ) = alloc_layer_scratch(&mut self.scratch_arena, &self.cfg, self.tp_size, num_tokens)?;
        // Persistent outputs are allocated after the ENTIRE overlay reservation;
        // remaining_workspace below cannot alias any phase, even the largest one.
        let logits_buf = self
            .scratch_arena
            .alloc_aligned(self.local_vocab_size * 2)?;
        self.last_logits_ptr = logits_buf;
        self.last_logits_rows = 1;
        let (ws_ptr, ws_cap) = self.scratch_arena.remaining_workspace();
        let ex = self.stream.exec(ws_ptr, ws_cap);

        let final_norm_out_buf = if !self.cfg.is_qwen4_exp()
            && num_tokens <= self.max_tap_rows
            && let Some(final_h_buf) = &self.last_final_hidden_f16
        {
            final_h_buf.addr()
        } else {
            normed_buf
        };

        if self.cfg.is_qwen4_exp() {
            self.run_qwen4_layer_stack(
                &ex,
                token_ids_ptr,
                positions_ptr,
                num_tokens,
                attn_batch,
                gdn_batch,
                tp2,
                residual_buf,
                normed_buf,
                block_out_buf,
                mlp_out_buf,
                hc_inj_weights_buf,
                final_norm_out_buf,
                [
                    qkv_proj_buf,
                    q_buf,
                    k_buf,
                    gate_buf,
                    attn_out_buf,
                    gated_attn_buf,
                ],
                [
                    qkvz_buf,
                    ba_buf,
                    qkv_conv_out_buf,
                    gdn_raw_out_buf,
                    gdn_normed_buf,
                ],
                [
                    gate_up_buf,
                    act_buf,
                    moe_router_logits_buf,
                    moe_topk_weights_buf,
                    moe_topk_ids_buf,
                    moe_interm_buf,
                    moe_sh_gu_buf,
                    moe_sh_act_buf,
                    moe_sh_out_buf,
                    moe_sh_gate_buf,
                ],
                ple_hc_addrs,
                qsa_addrs,
                layer_overlay_end,
                dump_out.as_deref_mut(),
            )?;
        } else {
            // 1. Embedding lookup into residual_buf (+ optional multimodal visual embedding splice)
            check_rc(
                // SAFETY: residual_buf reserves num_tokens * hidden_size FP16 elements;
                // embedding weights come from the rank arena. Token storage/IDs are
                // supplied by the raw-address caller, not validated by this wrapper.
                unsafe {
                    ffi::w34_embed_lookup_f16(
                        &ex,
                        token_ids_ptr as *const i32,
                        self.embed_tokens_ptr as *const c_void,
                        residual_buf as *mut c_void,
                        core::ptr::null(),
                        num_tokens as i32,
                        h_dim as i32,
                        self.cfg.vocab_size as i32,
                    )
                },
                "w34_embed_lookup_f16",
            )?;
            if let Some((vis_ptr, idx_ptr, num_vis)) = self.pending_visual_splice
                && num_vis > 0
            {
                check_rc(
                    // SAFETY: the vision path installs device embeddings/row indices
                    // in pending_visual_splice and keeps them live through forward.
                    // This raw tuple itself does not enforce index bounds or lifetime.
                    unsafe {
                        ffi::w34_embedding_splice_f16(
                            &ex,
                            residual_buf as *mut c_void,
                            vis_ptr as *const c_void,
                            idx_ptr as *const i32,
                            num_vis as i32,
                            h_dim as i32,
                        )
                    },
                    "w34_embedding_splice_f16",
                )?;
            }

            if let Some(dump) = dump_out.as_deref_mut()
                && self.rank == 0
            {
                Self::record_dump_f16(
                    &self.stream,
                    residual_buf,
                    vec![num_tokens, h_dim],
                    "embed",
                    dump,
                )?;
            }

            // 2. Layers
            for layer_idx in 0..self.layers.len() {
                let in_ln_ptr = self.layers[layer_idx].input_layernorm_ptr;
                let post_ln_ptr = self.layers[layer_idx].post_attention_layernorm_ptr;
                if layer_idx == 0 {
                    check_rc(
                        // SAFETY: residual/normed are separate num_tokens * h_dim
                        // regions; in_ln_ptr is the loaded layer's h_dim norm vector.
                        // All uses are ordered on this rank's main stream.
                        unsafe {
                            ffi::w34_gemma_rms_norm_f16(
                                &ex,
                                residual_buf as *const c_void,
                                in_ln_ptr as *const c_void,
                                normed_buf as *mut c_void,
                                core::ptr::null(),
                                num_tokens as i32,
                                h_dim as i32,
                                self.cfg.rms_norm_eps,
                            )
                        },
                        "w34_gemma_rms_norm_f16",
                    )?;
                }

                match &self.layers[layer_idx].block {
                    LayerBlockWeights::FullAttention(fa) => {
                        fa.qkv_proj.forward(
                            &ex,
                            &self.cublas,
                            marlin_ws_ptr,
                            marlin_ws_ints,
                            normed_buf,
                            qkv_proj_buf,
                            num_tokens,
                        )?;
                        let qkv_row_stride = (q_gate_dim + 2 * kv_dim) as i32;
                        let q_gate_ptr = qkv_proj_buf;
                        let k_raw_ptr = qkv_proj_buf + (q_gate_dim * 2) as u64;
                        let v_ptr = qkv_proj_buf + ((q_gate_dim + kv_dim) * 2) as u64;
                        check_rc(
                            // SAFETY: q/k/gate regions match the local head geometry;
                            // raw Q/K views use the packed projection's row stride.
                            // Position planes are supplied by the caller according to
                            // current_pos_plane_stride; their extent is not checked here.
                            unsafe {
                                ffi::w34_fused_qk_rmsnorm_rope_gate_strided_f16(
                                    &ex,
                                    q_gate_ptr as *const c_void,
                                    k_raw_ptr as *const c_void,
                                    fa.q_norm_ptr as *const c_void,
                                    fa.k_norm_ptr as *const c_void,
                                    positions_ptr as *const i32,
                                    core::ptr::null(),
                                    q_buf as *mut c_void,
                                    k_buf as *mut c_void,
                                    gate_buf as *mut c_void,
                                    num_tokens as i32,
                                    self.current_pos_plane_stride,
                                    local_q_heads as i32,
                                    local_kv_heads as i32,
                                    head_dim as i32,
                                    self.cfg.rotary_dim as i32,
                                    self.cfg.mrope_section[0] as i32,
                                    self.cfg.mrope_section[1] as i32,
                                    self.cfg.mrope_section[2] as i32,
                                    qkv_row_stride,
                                    qkv_row_stride,
                                    self.cfg.rope_theta,
                                    self.cfg.rms_norm_eps,
                                )
                            },
                            "w34_fused_qk_rmsnorm_rope_gate_strided_f16",
                        )?;
                        let kv_layout = &self.kv_pool.layers[fa.kv_layer_ordinal].layout;
                        check_rc(
                            // SAFETY: K is contiguous and V is a strided view of QKV;
                            // kv_layout describes the rank-owned layer allocation.
                            // The batch's device slot indices must address that pool;
                            // this wrapper does not read back and validate those indices.
                            unsafe {
                                ffi::w34_kv_cache_write_strided(
                                    &ex,
                                    kv_layout,
                                    attn_batch,
                                    k_buf as *const c_void,
                                    v_ptr as *const c_void,
                                    kv_dim as i32,
                                    qkv_row_stride,
                                )
                            },
                            "w34_kv_cache_write_strided",
                        )?;
                        check_rc(
                            // SAFETY: Q/output have local_q_heads * head_dim elements
                            // per token and the KV write precedes this on the same stream.
                            // Block-table contents and lengths are the batch caller's
                            // responsibility; the descriptor is consumed by the FFI call.
                            unsafe {
                                ffi::w34_paged_attention_f16(
                                    &ex,
                                    kv_layout,
                                    attn_batch,
                                    q_buf as *const c_void,
                                    attn_out_buf as *mut c_void,
                                    local_q_heads as i32,
                                )
                            },
                            "w34_paged_attention_f16",
                        )?;
                        check_rc(
                            // SAFETY: attention, gate, and destination are distinct
                            // local-Q-sized regions in the FA scratch phase, all used
                            // on the main stream before that phase's storage is reused.
                            unsafe {
                                ffi::w34_apply_attn_output_gate_f16(
                                    &ex,
                                    attn_out_buf as *const c_void,
                                    gate_buf as *const c_void,
                                    gated_attn_buf as *mut c_void,
                                    core::ptr::null(),
                                    num_tokens as i32,
                                    (local_q_heads * head_dim) as i32,
                                )
                            },
                            "w34_apply_attn_output_gate_f16",
                        )?;
                        fa.o_proj.forward(
                            &ex,
                            &self.cublas,
                            marlin_ws_ptr,
                            marlin_ws_ints,
                            gated_attn_buf,
                            block_out_buf,
                            num_tokens,
                        )?;
                    }
                    LayerBlockWeights::LinearAttention(gdn) => {
                        let can_fork_ba = !matches!(
                            gdn.in_proj_ba,
                            LinearProj::DenseF16 { size_n, size_k, .. }
                                if size_n > 256 || !size_k.is_multiple_of(8)
                        );
                        if can_fork_ba {
                            self.aux_fork_event.record_in_current_ctx(&self.stream)?;
                            self.aux_stream.wait_event(&self.aux_fork_event)?;
                            let ex_aux = self.aux_stream.exec(0, 0);
                            gdn.in_proj_ba.forward(
                                &ex_aux,
                                &self.cublas,
                                self.aux_marlin_ws.addr(),
                                self.aux_marlin_ws.len_bytes / 4,
                                normed_buf,
                                ba_buf,
                                num_tokens,
                            )?;
                            self.aux_join_event
                                .record_in_current_ctx(&self.aux_stream)?;
                        }
                        gdn.in_proj_qkvz.forward(
                            &ex,
                            &self.cublas,
                            marlin_ws_ptr,
                            marlin_ws_ints,
                            normed_buf,
                            qkvz_buf,
                            num_tokens,
                        )?;
                        if !can_fork_ba {
                            gdn.in_proj_ba.forward(
                                &ex,
                                &self.cublas,
                                marlin_ws_ptr,
                                marlin_ws_ints,
                                normed_buf,
                                ba_buf,
                                num_tokens,
                            )?;
                        }
                        let qkvz_row_stride = (conv_dim + gdn_v_dim) as i32;
                        let ba_row_stride = (2 * local_v_heads) as i32;
                        let qkv_ptr = qkvz_buf;
                        let z_ptr = qkvz_buf + (conv_dim * 2) as u64;
                        let b_ptr = ba_buf;
                        let a_ptr = ba_buf + (local_v_heads * 2) as u64;
                        let st_bufs = &self.state_pool.layers[gdn.gdn_layer_ordinal];
                        check_rc(
                            // SAFETY: QKV is the conv_dim prefix of each QKVZ row;
                            // conv output and state storage use the same loaded geometry.
                            // State/checkpoint indices in gdn_batch must refer to this
                            // state pool; device index values are not checked here.
                            unsafe {
                                ffi::w34_gdn_causal_conv1d_strided_f16(
                                    &ex,
                                    gdn_batch,
                                    qkv_ptr as *const c_void,
                                    gdn.conv_weight_ptr as *const c_void,
                                    core::ptr::null(),
                                    st_bufs.conv_state.addr() as *mut c_void,
                                    qkv_conv_out_buf as *mut c_void,
                                    conv_dim as i32,
                                    qkvz_row_stride,
                                    self.cfg.linear_conv_kernel_dim as i32,
                                    self.state_pool.total_slots as i32,
                                    1,
                                )
                            },
                            "w34_gdn_causal_conv1d_strided_f16",
                        )?;
                        if can_fork_ba {
                            self.stream.wait_event(&self.aux_join_event)?;
                        }
                        let gdn_tail_cap =
                            layer_overlay_end.saturating_sub(gdn_normed_buf) as usize;
                        let ex_gdn = if gdn_tail_cap > ex.scratch_size {
                            self.stream.exec(gdn_normed_buf, gdn_tail_cap)
                        } else {
                            ex
                        };
                        check_rc(
                            // SAFETY: conv output and FP32 SSM storage match local heads;
                            // the auxiliary BA writer was joined above before these reads.
                            // ex_gdn borrows the unused GDN tail only until this kernel
                            // completes; state indices still rely on the batch caller.
                            unsafe {
                                ffi::w34_gdn_recurrent_strided_f16(
                                    &ex_gdn,
                                    gdn_batch,
                                    qkv_conv_out_buf as *const c_void,
                                    core::ptr::null(),
                                    core::ptr::null(),
                                    core::ptr::null(),
                                    b_ptr as *const c_void,
                                    a_ptr as *const c_void,
                                    gdn.a_log_ptr as *const f32,
                                    gdn.dt_bias_ptr as *const f32,
                                    st_bufs.ssm_state.addr() as *mut f32,
                                    gdn_raw_out_buf as *mut c_void,
                                    local_k_heads as i32,
                                    head_k_dim as i32,
                                    local_v_heads as i32,
                                    head_v_dim as i32,
                                    ba_row_stride,
                                    self.state_pool.total_slots as i32,
                                )
                            },
                            "w34_gdn_recurrent_strided_f16",
                        )?;
                        check_rc(
                            // SAFETY: Z is the gdn_v_dim suffix of each QKVZ row;
                            // raw/normed outputs have num_tokens * gdn_v_dim elements.
                            // Main-stream order ends recurrent scratch use before writing
                            // gdn_normed_buf, which overlaps that temporary workspace.
                            unsafe {
                                ffi::w34_rms_norm_gated_silu_strided_f16(
                                    &ex,
                                    gdn_raw_out_buf as *const c_void,
                                    z_ptr as *const c_void,
                                    gdn.norm_weight_ptr as *const c_void,
                                    gdn_normed_buf as *mut c_void,
                                    core::ptr::null(),
                                    num_tokens as i32,
                                    local_v_heads as i32,
                                    head_v_dim as i32,
                                    qkvz_row_stride,
                                    self.cfg.rms_norm_eps,
                                )
                            },
                            "w34_rms_norm_gated_silu_strided_f16",
                        )?;
                        gdn.out_proj.forward(
                            &ex,
                            &self.cublas,
                            marlin_ws_ptr,
                            marlin_ws_ints,
                            gdn_normed_buf,
                            block_out_buf,
                            num_tokens,
                        )?;
                    }
                }

                if let Some(comm) = tp2 {
                    let ar_out = if dump_out.is_some() { block_out_buf } else { 0 };
                    comm.all_reduce_residual_gemma_rms_norm_f16(
                        &ex,
                        block_out_buf,
                        ar_out,
                        residual_buf,
                        post_ln_ptr,
                        normed_buf,
                        0,
                        self.rank as i32,
                        num_tokens as i32,
                        h_dim as i32,
                        self.cfg.rms_norm_eps,
                        0,
                        0,
                    )?;
                    if let Some(dump) = dump_out.as_deref_mut()
                        && self.rank == 0
                    {
                        Self::record_dump_f16(
                            &self.stream,
                            block_out_buf,
                            vec![num_tokens, h_dim],
                            &format!("layer_{layer_idx}_attn_or_gdn_out"),
                            dump,
                        )?;
                    }
                } else {
                    if let Some(dump) = dump_out.as_deref_mut()
                        && self.rank == 0
                    {
                        Self::record_dump_f16(
                            &self.stream,
                            block_out_buf,
                            vec![num_tokens, h_dim],
                            &format!("layer_{layer_idx}_attn_or_gdn_out"),
                            dump,
                        )?;
                    }
                    // Post-attention RMSNorm + MLP
                    check_rc(
                        // SAFETY: block_out, residual, and normed regions each contain
                        // num_tokens * h_dim FP16 elements; residual is updated in place
                        // by this kernel, with no concurrent main-stream reader.
                        unsafe {
                            ffi::w34_fused_add_gemma_rms_norm_f16(
                                &ex,
                                block_out_buf as *const c_void,
                                residual_buf as *mut c_void,
                                post_ln_ptr as *const c_void,
                                normed_buf as *mut c_void,
                                core::ptr::null(),
                                num_tokens as i32,
                                h_dim as i32,
                                self.cfg.rms_norm_eps,
                            )
                        },
                        "w34_fused_add_gemma_rms_norm_f16 (post_attn)",
                    )?;
                }

                self.layers[layer_idx].ffn.forward(
                    &ex,
                    &self.cublas,
                    marlin_ws_ptr,
                    marlin_ws_ints,
                    normed_buf,
                    mlp_out_buf,
                    num_tokens,
                    h_dim,
                    local_interm,
                    gate_up_buf,
                    act_buf,
                    moe_router_logits_buf,
                    moe_topk_weights_buf,
                    moe_topk_ids_buf,
                    moe_interm_buf,
                    moe_sh_gu_buf,
                    moe_sh_act_buf,
                    moe_sh_out_buf,
                    moe_sh_gate_buf,
                )?;
                let (next_norm_weight, next_norm_out) = if layer_idx + 1 < self.layers.len() {
                    (self.layers[layer_idx + 1].input_layernorm_ptr, normed_buf)
                } else {
                    (self.final_norm_ptr, final_norm_out_buf)
                };
                let (tap_dst, tap_stride, tap_offset) =
                    self.layer_tap_target(layer_idx, num_tokens, h_dim);
                if let Some(comm) = tp2 {
                    comm.all_reduce_residual_gemma_rms_norm_f16(
                        &ex,
                        mlp_out_buf,
                        0,
                        residual_buf,
                        next_norm_weight,
                        next_norm_out,
                        tap_dst,
                        self.rank as i32,
                        num_tokens as i32,
                        h_dim as i32,
                        self.cfg.rms_norm_eps,
                        tap_stride,
                        tap_offset,
                    )?;
                } else {
                    check_rc(
                        // SAFETY: residual/output rows match h_dim. tap_dst is either
                        // null (no tap) or a configured BF16 buffer with the checked
                        // max_tap_rows bound and the selected tap column offset.
                        unsafe {
                            ffi::w34_fused_add_gemma_rms_norm_tap_f16(
                                &ex,
                                mlp_out_buf as *const c_void,
                                residual_buf as *mut c_void,
                                next_norm_weight as *const c_void,
                                next_norm_out as *mut c_void,
                                tap_dst as *mut c_void,
                                core::ptr::null(),
                                num_tokens as i32,
                                h_dim as i32,
                                self.cfg.rms_norm_eps,
                                tap_stride,
                                tap_offset,
                            )
                        },
                        "w34_fused_add_gemma_rms_norm_tap_f16 (post_mlp)",
                    )?;
                }
                if layer_idx + 1 < self.layers.len()
                    && let Some(dump) = dump_out.as_deref_mut()
                    && self.rank == 0
                {
                    Self::record_dump_f16(
                        &self.stream,
                        residual_buf,
                        vec![num_tokens, h_dim],
                        &format!("layer_{layer_idx}_hidden_after"),
                        dump,
                    )?;
                }
            }
        }

        if let Some(dump) = dump_out.as_deref_mut()
            && self.rank == 0
        {
            self.record_final_forward_dump(
                residual_buf,
                final_norm_out_buf,
                num_tokens,
                h_dim,
                dump,
            )?;
        }

        // 4. LM Head on last token row `[1, h_dim]`
        let last_hidden_ptr = final_norm_out_buf + ((num_tokens - 1) * h_dim * 2) as u64;
        self.lm_head.forward(
            &ex,
            &self.cublas,
            marlin_ws_ptr,
            marlin_ws_ints,
            last_hidden_ptr,
            logits_buf,
            1,
        )?;

        // 5. Sampler
        if let Some(comm) = tp2 {
            // Fast 1-pass local argmax + raw max logit in FP32, then TP=2 top-1 gather
            check_rc(
                // SAFETY: logits_buf contains one local-vocab row. The rank's
                // decode IO owns the scalar local ID/value outputs used here;
                // null top-ID output selects the local-argmax kernel mode.
                unsafe {
                    ffi::w34_sampler_greedy_top5_f16(
                        &ex,
                        logits_buf as *const c_void,
                        self.decode_io.local_max_id.addr() as *mut i32,
                        core::ptr::null_mut(),
                        self.decode_io.local_max_val.addr() as *mut f32,
                        1,
                        self.local_vocab_size as i32,
                        self.vocab_offset as i32,
                    )
                },
                "w34_sampler_greedy_top5_f16 (tp2 local)",
            )?;
            comm.all_gather_top1_f32_i32(
                &ex,
                self.decode_io.local_max_val.addr(),
                self.decode_io.local_max_id.addr(),
                self.decode_io.global_max_val.addr(),
                out_greedy_id_ptr,
                self.rank as i32,
                1,
            )?;
            if out_top5_ids_ptr != 0 && out_top5_logprobs_ptr != 0 {
                check_rc(
                    // SAFETY: one local-vocab logits row is live. Nonzero top5
                    // addresses must each hold five i32/f32 values as supplied by
                    // the caller; checking nonzero alone does not establish capacity.
                    unsafe {
                        ffi::w34_sampler_greedy_top5_f16(
                            &ex,
                            logits_buf as *const c_void,
                            self.decode_io.local_max_id.addr() as *mut i32,
                            out_top5_ids_ptr as *mut i32,
                            out_top5_logprobs_ptr as *mut f32,
                            1,
                            self.local_vocab_size as i32,
                            self.vocab_offset as i32,
                        )
                    },
                    "w34_sampler_greedy_top5_f16 (tp2 top5)",
                )?;
            }
        } else {
            check_rc(
                // SAFETY: one local-vocab row is allocated above. The raw-address
                // caller provides a scalar greedy output and optional five-element
                // top5 outputs; null optional outputs are accepted by the kernel.
                unsafe {
                    ffi::w34_sampler_greedy_top5_f16(
                        &ex,
                        logits_buf as *const c_void,
                        out_greedy_id_ptr as *mut i32,
                        out_top5_ids_ptr as *mut i32,
                        out_top5_logprobs_ptr as *mut f32,
                        1,
                        self.local_vocab_size as i32,
                        self.vocab_offset as i32,
                    )
                },
                "w34_sampler_greedy_top5_f16",
            )?;
        }

        if let Some(dump) = dump_out
            && self.rank == 0
        {
            Self::record_dump_f16(
                &self.stream,
                logits_buf,
                vec![self.local_vocab_size],
                "logits_last",
                dump,
            )?;
        }

        Ok(0)
    }

    /// Evaluates the full layer stack on `num_tokens` and projects each row in `selected_rows`
    /// through `lm_head` + `w34_sampler_greedy_top5_f16`, returning one greedy token ID per
    /// selected row.
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
        if selected_rows.is_empty() {
            let _ = self.forward_selected_rows_device(
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
            self.stream.synchronize()?;
            self.speculative_logits_rows = 0;
            self.last_logits_ptr = 0;
            self.last_logits_rows = 0;
            return Ok(Vec::new());
        }
        let (out_greedy_ids_dev, use_spec_buf) = self.forward_selected_rows_device(
            token_ids_ptr,
            positions_ptr,
            num_tokens,
            attn_batch,
            gdn_batch,
            tp2,
            selected_rows,
            None,
            true,
        )?;
        let num_selected = selected_rows.len();
        let mut out_ids = vec![0i32; num_selected];
        check_rc(
            // SAFETY: the device core allocated num_selected greedy IDs; out_ids
            // has the same i32 extent and remains live until synchronization below.
            unsafe {
                ffi::w34_memcpy_d2h_async(
                    out_ids.as_mut_ptr().cast(),
                    out_greedy_ids_dev as *const c_void,
                    num_selected * 4,
                    self.stream.handle,
                )
            },
            "copy selected greedy ids",
        )?;
        self.stream.synchronize()?;
        self.speculative_logits_rows = if use_spec_buf { num_selected } else { 0 };
        Ok(out_ids)
    }

    /// Pure device-enqueue core of [`Self::forward_selected_rows`] (no D2H copy or host stream
    /// synchronize, making fixed-shape `M_verify` speculative verification CUDA-Graph-capturable).
    #[allow(clippy::too_many_arguments, clippy::cognitive_complexity)]
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
        if attn_batch.num_reqs <= 1 {
            self.validate_tap_rows(num_tokens)?;
        }
        if attn_batch.num_reqs <= 1
            && self
                .speculative_logits
                .as_ref()
                .is_some_and(|b| selected_rows.len() > b.len_bytes / (self.local_vocab_size * 2))
        {
            return Err(loader_err(
                "selected rows exceed speculative logits capacity",
            ));
        }
        self.speculative_logits_rows = 0;
        self.device.make_current()?;
        let h_dim = self.cfg.hidden_size;
        let head_dim = self.cfg.head_dim;
        let local_q_heads = self.cfg.num_attention_heads / self.tp_size;
        let local_kv_heads = self.cfg.num_key_value_heads / self.tp_size;
        let q_gate_dim = local_q_heads * head_dim * 2;
        let kv_dim = local_kv_heads * head_dim;
        let local_k_heads = self.cfg.linear_num_key_heads / self.tp_size;
        let local_v_heads = self.cfg.linear_num_value_heads / self.tp_size;
        let head_k_dim = self.cfg.linear_key_head_dim;
        let head_v_dim = self.cfg.linear_value_head_dim;
        let conv_dim = 2 * local_k_heads * head_k_dim + local_v_heads * head_v_dim;
        let gdn_v_dim = local_v_heads * head_v_dim;
        let local_interm = self.cfg.intermediate_size / self.tp_size;
        let marlin_ws_ptr = self.marlin_ws.addr();
        let marlin_ws_ints = self.marlin_ws.len_bytes / 4;

        self.scratch_arena.reset();
        let hc_mult = self.cfg.hc_count.max(1);
        let residual_buf = self
            .scratch_arena
            .alloc_aligned(num_tokens * hc_mult * h_dim * 2)?;
        let normed_buf = self.scratch_arena.alloc_aligned(num_tokens * h_dim * 2)?;
        let block_out_buf = self.scratch_arena.alloc_aligned(num_tokens * h_dim * 2)?;
        let mlp_out_buf = self.scratch_arena.alloc_aligned(num_tokens * h_dim * 2)?;
        let hc_inj_weights_buf = if self.cfg.is_qwen4_exp() {
            self.scratch_arena.alloc_aligned(num_tokens * hc_mult * 4)?
        } else {
            0
        };

        let (
            [
                qkv_proj_buf,
                q_buf,
                k_buf,
                gate_buf,
                attn_out_buf,
                gated_attn_buf,
            ],
            [
                qkvz_buf,
                ba_buf,
                qkv_conv_out_buf,
                gdn_raw_out_buf,
                gdn_normed_buf,
            ],
            [
                gate_up_buf,
                act_buf,
                moe_router_logits_buf,
                moe_topk_weights_buf,
                moe_topk_ids_buf,
                moe_interm_buf,
                moe_sh_gu_buf,
                moe_sh_act_buf,
                moe_sh_out_buf,
                moe_sh_gate_buf,
            ],
            ple_hc_addrs,
            qsa_addrs,
            layer_overlay_end,
        ) = alloc_layer_scratch(&mut self.scratch_arena, &self.cfg, self.tp_size, num_tokens)?;
        // Persistent outputs are allocated after the ENTIRE overlay reservation;
        // remaining_workspace below cannot alias any phase, even the largest one.
        let num_selected_pre = selected_rows.len();
        let use_spec_buf_pre = num_selected_pre > 0
            && self
                .speculative_logits
                .as_ref()
                .is_some_and(|b| num_selected_pre <= b.len_bytes / (self.local_vocab_size * 2));
        let pre_all_logits_buf = if num_selected_pre == 0 {
            0
        } else if use_spec_buf_pre {
            self.speculative_logits.as_ref().unwrap().addr()
        } else {
            self.scratch_arena
                .alloc_aligned(num_selected_pre * self.local_vocab_size * 2)?
        };
        let is_contiguous_pre = num_selected_pre > 0
            && selected_rows
                .iter()
                .enumerate()
                .all(|(i, &r)| r == selected_rows[0] + i);
        let pre_gathered_hidden = if num_selected_pre > 0 && !is_contiguous_pre {
            self.scratch_arena
                .alloc_aligned(num_selected_pre * h_dim * 2)?
        } else {
            0
        };
        let pre_out_greedy =
            if emit_device_greedy && num_selected_pre > 0 && out_greedy_ids_override.is_none() {
                self.scratch_arena.alloc_aligned(num_selected_pre * 4)?
            } else {
                0
            };
        let (pre_local_ids, pre_local_vals, pre_global_vals) =
            if emit_device_greedy && num_selected_pre > 0 && tp2.is_some() {
                (
                    self.scratch_arena.alloc_aligned(num_selected_pre * 4)?,
                    self.scratch_arena.alloc_aligned(num_selected_pre * 4)?,
                    self.scratch_arena.alloc_aligned(num_selected_pre * 4)?,
                )
            } else {
                (0, 0, 0)
            };
        let (ws_ptr, ws_cap) = self.scratch_arena.remaining_workspace();
        let ex = self.stream.exec(ws_ptr, ws_cap);

        let final_norm_out_buf = if !self.cfg.is_qwen4_exp()
            && num_tokens <= self.max_tap_rows
            && let Some(final_h_buf) = &self.last_final_hidden_f16
        {
            final_h_buf.addr()
        } else {
            normed_buf
        };

        if self.cfg.is_qwen4_exp() {
            self.run_qwen4_layer_stack(
                &ex,
                token_ids_ptr,
                positions_ptr,
                num_tokens,
                attn_batch,
                gdn_batch,
                tp2,
                residual_buf,
                normed_buf,
                block_out_buf,
                mlp_out_buf,
                hc_inj_weights_buf,
                final_norm_out_buf,
                [
                    qkv_proj_buf,
                    q_buf,
                    k_buf,
                    gate_buf,
                    attn_out_buf,
                    gated_attn_buf,
                ],
                [
                    qkvz_buf,
                    ba_buf,
                    qkv_conv_out_buf,
                    gdn_raw_out_buf,
                    gdn_normed_buf,
                ],
                [
                    gate_up_buf,
                    act_buf,
                    moe_router_logits_buf,
                    moe_topk_weights_buf,
                    moe_topk_ids_buf,
                    moe_interm_buf,
                    moe_sh_gu_buf,
                    moe_sh_act_buf,
                    moe_sh_out_buf,
                    moe_sh_gate_buf,
                ],
                ple_hc_addrs,
                qsa_addrs,
                layer_overlay_end,
                None,
            )?;
        } else {
            check_rc(
                // SAFETY: residual_buf reserves num_tokens * hidden_size FP16 elements;
                // embedding weights come from the rank arena. Token storage/IDs are
                // supplied by the raw-address caller, not validated by this wrapper.
                unsafe {
                    ffi::w34_embed_lookup_f16(
                        &ex,
                        token_ids_ptr as *const i32,
                        self.embed_tokens_ptr as *const c_void,
                        residual_buf as *mut c_void,
                        core::ptr::null(),
                        num_tokens as i32,
                        h_dim as i32,
                        self.cfg.vocab_size as i32,
                    )
                },
                "w34_embed_lookup_f16",
            )?;
            if let Some((vis_ptr, idx_ptr, num_vis)) = self.pending_visual_splice
                && num_vis > 0
            {
                check_rc(
                    // SAFETY: the vision path installs device embeddings/row indices
                    // in pending_visual_splice and keeps them live through forward.
                    // This raw tuple itself does not enforce index bounds or lifetime.
                    unsafe {
                        ffi::w34_embedding_splice_f16(
                            &ex,
                            residual_buf as *mut c_void,
                            vis_ptr as *const c_void,
                            idx_ptr as *const i32,
                            num_vis as i32,
                            h_dim as i32,
                        )
                    },
                    "w34_embedding_splice_f16",
                )?;
            }

            for layer_idx in 0..self.layers.len() {
                let in_ln_ptr = self.layers[layer_idx].input_layernorm_ptr;
                let post_ln_ptr = self.layers[layer_idx].post_attention_layernorm_ptr;
                if layer_idx == 0 {
                    check_rc(
                        // SAFETY: residual/normed are separate num_tokens * h_dim
                        // regions; in_ln_ptr is the loaded layer's h_dim norm vector.
                        // All uses are ordered on this rank's main stream.
                        unsafe {
                            ffi::w34_gemma_rms_norm_f16(
                                &ex,
                                residual_buf as *const c_void,
                                in_ln_ptr as *const c_void,
                                normed_buf as *mut c_void,
                                core::ptr::null(),
                                num_tokens as i32,
                                h_dim as i32,
                                self.cfg.rms_norm_eps,
                            )
                        },
                        "w34_gemma_rms_norm_f16",
                    )?;
                }

                match &self.layers[layer_idx].block {
                    LayerBlockWeights::FullAttention(fa) => {
                        fa.qkv_proj.forward(
                            &ex,
                            &self.cublas,
                            marlin_ws_ptr,
                            marlin_ws_ints,
                            normed_buf,
                            qkv_proj_buf,
                            num_tokens,
                        )?;
                        let qkv_row_stride = (q_gate_dim + 2 * kv_dim) as i32;
                        let q_gate_ptr = qkv_proj_buf;
                        let k_raw_ptr = qkv_proj_buf + (q_gate_dim * 2) as u64;
                        let v_ptr = qkv_proj_buf + ((q_gate_dim + kv_dim) * 2) as u64;
                        let rotary_dim = self.cfg.rotary_dim as i32;
                        check_rc(
                            // SAFETY: q/k/gate regions match the local head geometry;
                            // raw Q/K views use the packed projection's row stride.
                            // Position planes are supplied by the caller according to
                            // current_pos_plane_stride; their extent is not checked here.
                            unsafe {
                                ffi::w34_fused_qk_rmsnorm_rope_gate_strided_f16(
                                    &ex,
                                    q_gate_ptr as *const c_void,
                                    k_raw_ptr as *const c_void,
                                    fa.q_norm_ptr as *const c_void,
                                    fa.k_norm_ptr as *const c_void,
                                    positions_ptr as *const i32,
                                    core::ptr::null(),
                                    q_buf as *mut c_void,
                                    k_buf as *mut c_void,
                                    gate_buf as *mut c_void,
                                    num_tokens as i32,
                                    self.current_pos_plane_stride,
                                    local_q_heads as i32,
                                    local_kv_heads as i32,
                                    head_dim as i32,
                                    rotary_dim,
                                    self.cfg.mrope_section[0] as i32,
                                    self.cfg.mrope_section[1] as i32,
                                    self.cfg.mrope_section[2] as i32,
                                    qkv_row_stride,
                                    qkv_row_stride,
                                    self.cfg.rope_theta,
                                    self.cfg.rms_norm_eps,
                                )
                            },
                            "w34_fused_qk_rmsnorm_rope_gate_strided_f16",
                        )?;
                        let kv_layout = &self.kv_pool.layers[fa.kv_layer_ordinal].layout;
                        check_rc(
                            // SAFETY: K is contiguous and V is a strided view of QKV;
                            // kv_layout describes the rank-owned layer allocation.
                            // The batch's device slot indices must address that pool;
                            // this wrapper does not read back and validate those indices.
                            unsafe {
                                ffi::w34_kv_cache_write_strided(
                                    &ex,
                                    kv_layout,
                                    attn_batch,
                                    k_buf as *const c_void,
                                    v_ptr as *const c_void,
                                    kv_dim as i32,
                                    qkv_row_stride,
                                )
                            },
                            "w34_kv_cache_write_strided",
                        )?;
                        check_rc(
                            // SAFETY: Q/output have local_q_heads * head_dim elements
                            // per token and the KV write precedes this on the same stream.
                            // Block-table contents and lengths are the batch caller's
                            // responsibility; the descriptor is consumed by the FFI call.
                            unsafe {
                                ffi::w34_paged_attention_f16(
                                    &ex,
                                    kv_layout,
                                    attn_batch,
                                    q_buf as *const c_void,
                                    attn_out_buf as *mut c_void,
                                    local_q_heads as i32,
                                )
                            },
                            "w34_paged_attention_f16",
                        )?;
                        check_rc(
                            // SAFETY: attention, gate, and destination are distinct
                            // local-Q-sized regions in the FA scratch phase, all used
                            // on the main stream before that phase's storage is reused.
                            unsafe {
                                ffi::w34_apply_attn_output_gate_f16(
                                    &ex,
                                    attn_out_buf as *const c_void,
                                    gate_buf as *const c_void,
                                    gated_attn_buf as *mut c_void,
                                    core::ptr::null(),
                                    num_tokens as i32,
                                    (local_q_heads * head_dim) as i32,
                                )
                            },
                            "w34_apply_attn_output_gate_f16",
                        )?;
                        fa.o_proj.forward(
                            &ex,
                            &self.cublas,
                            marlin_ws_ptr,
                            marlin_ws_ints,
                            gated_attn_buf,
                            block_out_buf,
                            num_tokens,
                        )?;
                    }
                    LayerBlockWeights::LinearAttention(gdn) => {
                        let can_fork_ba = !matches!(
                            gdn.in_proj_ba,
                            LinearProj::DenseF16 { size_n, size_k, .. }
                                if size_n > 256 || !size_k.is_multiple_of(8)
                        );
                        if can_fork_ba {
                            self.aux_fork_event.record_in_current_ctx(&self.stream)?;
                            self.aux_stream.wait_event(&self.aux_fork_event)?;
                            let ex_aux = self.aux_stream.exec(0, 0);
                            gdn.in_proj_ba.forward(
                                &ex_aux,
                                &self.cublas,
                                self.aux_marlin_ws.addr(),
                                self.aux_marlin_ws.len_bytes / 4,
                                normed_buf,
                                ba_buf,
                                num_tokens,
                            )?;
                            self.aux_join_event
                                .record_in_current_ctx(&self.aux_stream)?;
                        }
                        gdn.in_proj_qkvz.forward(
                            &ex,
                            &self.cublas,
                            marlin_ws_ptr,
                            marlin_ws_ints,
                            normed_buf,
                            qkvz_buf,
                            num_tokens,
                        )?;
                        if !can_fork_ba {
                            gdn.in_proj_ba.forward(
                                &ex,
                                &self.cublas,
                                marlin_ws_ptr,
                                marlin_ws_ints,
                                normed_buf,
                                ba_buf,
                                num_tokens,
                            )?;
                        }
                        let qkvz_row_stride = (conv_dim + gdn_v_dim) as i32;
                        let ba_row_stride = (2 * local_v_heads) as i32;
                        let qkv_ptr = qkvz_buf;
                        let z_ptr = qkvz_buf + (conv_dim * 2) as u64;
                        let b_ptr = ba_buf;
                        let a_ptr = ba_buf + (local_v_heads * 2) as u64;
                        let st_bufs = &self.state_pool.layers[gdn.gdn_layer_ordinal];
                        check_rc(
                            // SAFETY: QKV is the conv_dim prefix of each QKVZ row;
                            // conv output and state storage use the same loaded geometry.
                            // State/checkpoint indices in gdn_batch must refer to this
                            // state pool; device index values are not checked here.
                            unsafe {
                                ffi::w34_gdn_causal_conv1d_strided_f16(
                                    &ex,
                                    gdn_batch,
                                    qkv_ptr as *const c_void,
                                    gdn.conv_weight_ptr as *const c_void,
                                    core::ptr::null(),
                                    st_bufs.conv_state.addr() as *mut c_void,
                                    qkv_conv_out_buf as *mut c_void,
                                    conv_dim as i32,
                                    qkvz_row_stride,
                                    self.cfg.linear_conv_kernel_dim as i32,
                                    self.state_pool.total_slots as i32,
                                    1,
                                )
                            },
                            "w34_gdn_causal_conv1d_strided_f16",
                        )?;
                        if can_fork_ba {
                            self.stream.wait_event(&self.aux_join_event)?;
                        }
                        let gdn_tail_cap =
                            layer_overlay_end.saturating_sub(gdn_normed_buf) as usize;
                        let ex_gdn = if gdn_tail_cap > ex.scratch_size {
                            self.stream.exec(gdn_normed_buf, gdn_tail_cap)
                        } else {
                            ex
                        };
                        check_rc(
                            // SAFETY: conv output and FP32 SSM storage match local heads;
                            // the auxiliary BA writer was joined above before these reads.
                            // ex_gdn borrows the unused GDN tail only until this kernel
                            // completes; state indices still rely on the batch caller.
                            unsafe {
                                ffi::w34_gdn_recurrent_strided_f16(
                                    &ex_gdn,
                                    gdn_batch,
                                    qkv_conv_out_buf as *const c_void,
                                    core::ptr::null(),
                                    core::ptr::null(),
                                    core::ptr::null(),
                                    b_ptr as *const c_void,
                                    a_ptr as *const c_void,
                                    gdn.a_log_ptr as *const f32,
                                    gdn.dt_bias_ptr as *const f32,
                                    st_bufs.ssm_state.addr() as *mut f32,
                                    gdn_raw_out_buf as *mut c_void,
                                    local_k_heads as i32,
                                    head_k_dim as i32,
                                    local_v_heads as i32,
                                    head_v_dim as i32,
                                    ba_row_stride,
                                    self.state_pool.total_slots as i32,
                                )
                            },
                            "w34_gdn_recurrent_strided_f16",
                        )?;
                        check_rc(
                            // SAFETY: Z is the gdn_v_dim suffix of each QKVZ row;
                            // raw/normed outputs have num_tokens * gdn_v_dim elements.
                            // Main-stream order ends recurrent scratch use before writing
                            // gdn_normed_buf, which overlaps that temporary workspace.
                            unsafe {
                                ffi::w34_rms_norm_gated_silu_strided_f16(
                                    &ex,
                                    gdn_raw_out_buf as *const c_void,
                                    z_ptr as *const c_void,
                                    gdn.norm_weight_ptr as *const c_void,
                                    gdn_normed_buf as *mut c_void,
                                    core::ptr::null(),
                                    num_tokens as i32,
                                    local_v_heads as i32,
                                    head_v_dim as i32,
                                    qkvz_row_stride,
                                    self.cfg.rms_norm_eps,
                                )
                            },
                            "w34_rms_norm_gated_silu_strided_f16",
                        )?;
                        gdn.out_proj.forward(
                            &ex,
                            &self.cublas,
                            marlin_ws_ptr,
                            marlin_ws_ints,
                            gdn_normed_buf,
                            block_out_buf,
                            num_tokens,
                        )?;
                    }
                }

                if let Some(comm) = tp2 {
                    comm.all_reduce_residual_gemma_rms_norm_f16(
                        &ex,
                        block_out_buf,
                        0,
                        residual_buf,
                        post_ln_ptr,
                        normed_buf,
                        0,
                        self.rank as i32,
                        num_tokens as i32,
                        h_dim as i32,
                        self.cfg.rms_norm_eps,
                        0,
                        0,
                    )?;
                } else {
                    check_rc(
                        // SAFETY: block_out, residual, and normed regions each contain
                        // num_tokens * h_dim FP16 elements; residual is updated in place
                        // by this kernel, with no concurrent main-stream reader.
                        unsafe {
                            ffi::w34_fused_add_gemma_rms_norm_f16(
                                &ex,
                                block_out_buf as *const c_void,
                                residual_buf as *mut c_void,
                                post_ln_ptr as *const c_void,
                                normed_buf as *mut c_void,
                                core::ptr::null(),
                                num_tokens as i32,
                                h_dim as i32,
                                self.cfg.rms_norm_eps,
                            )
                        },
                        "w34_fused_add_gemma_rms_norm_f16 (post)",
                    )?;
                }

                self.layers[layer_idx].ffn.forward(
                    &ex,
                    &self.cublas,
                    marlin_ws_ptr,
                    marlin_ws_ints,
                    normed_buf,
                    mlp_out_buf,
                    num_tokens,
                    h_dim,
                    local_interm,
                    gate_up_buf,
                    act_buf,
                    moe_router_logits_buf,
                    moe_topk_weights_buf,
                    moe_topk_ids_buf,
                    moe_interm_buf,
                    moe_sh_gu_buf,
                    moe_sh_act_buf,
                    moe_sh_out_buf,
                    moe_sh_gate_buf,
                )?;

                let (next_norm_weight, next_norm_out) = if layer_idx + 1 < self.layers.len() {
                    (self.layers[layer_idx + 1].input_layernorm_ptr, normed_buf)
                } else {
                    (self.final_norm_ptr, final_norm_out_buf)
                };
                let (tap_dst, tap_stride, tap_offset) =
                    self.layer_tap_target(layer_idx, num_tokens, h_dim);
                if let Some(comm) = tp2 {
                    comm.all_reduce_residual_gemma_rms_norm_f16(
                        &ex,
                        mlp_out_buf,
                        0,
                        residual_buf,
                        next_norm_weight,
                        next_norm_out,
                        tap_dst,
                        self.rank as i32,
                        num_tokens as i32,
                        h_dim as i32,
                        self.cfg.rms_norm_eps,
                        tap_stride,
                        tap_offset,
                    )?;
                } else {
                    check_rc(
                        // SAFETY: residual/output rows match h_dim. tap_dst is either
                        // null (no tap) or a configured BF16 buffer with the checked
                        // max_tap_rows bound and the selected tap column offset.
                        unsafe {
                            ffi::w34_fused_add_gemma_rms_norm_tap_f16(
                                &ex,
                                mlp_out_buf as *const c_void,
                                residual_buf as *mut c_void,
                                next_norm_weight as *const c_void,
                                next_norm_out as *mut c_void,
                                tap_dst as *mut c_void,
                                core::ptr::null(),
                                num_tokens as i32,
                                h_dim as i32,
                                self.cfg.rms_norm_eps,
                                tap_stride,
                                tap_offset,
                            )
                        },
                        "w34_fused_add_gemma_rms_norm_tap_f16 (selected)",
                    )?;
                }
            }
        }

        self.last_norm_hidden_ptr = final_norm_out_buf;
        self.last_norm_hidden_rows = num_tokens;
        if selected_rows.is_empty() {
            self.speculative_logits_rows = 0;
            self.last_logits_ptr = 0;
            self.last_logits_rows = 0;
            return Ok((0, false));
        }
        let num_selected = num_selected_pre;
        let use_spec_buf = use_spec_buf_pre;
        let all_logits_buf = pre_all_logits_buf;
        self.last_logits_ptr = all_logits_buf;
        self.last_logits_rows = num_selected;
        let selected_hidden_ptr = if is_contiguous_pre {
            final_norm_out_buf + (selected_rows[0] * h_dim * 2) as u64
        } else {
            let gathered_hidden = pre_gathered_hidden;
            for (selected_idx, &row_idx) in selected_rows.iter().enumerate() {
                check_rc(
                    // SAFETY: destination row is within the selected-row allocation
                    // and disjoint from final hidden storage. row_idx is supplied by
                    // the engine's row selection and must be < num_tokens; this
                    // raw-address core does not validate arbitrary selected rows.
                    unsafe {
                        ffi::w34_memcpy_d2d_async(
                            (gathered_hidden + (selected_idx * h_dim * 2) as u64) as *mut c_void,
                            (final_norm_out_buf + (row_idx * h_dim * 2) as u64) as *const c_void,
                            h_dim * 2,
                            self.stream.handle,
                        )
                    },
                    "gather selected hidden rows",
                )?;
            }
            gathered_hidden
        };
        self.lm_head.forward(
            &ex,
            &self.cublas,
            marlin_ws_ptr,
            marlin_ws_ints,
            selected_hidden_ptr,
            all_logits_buf,
            num_selected,
        )?;

        if !emit_device_greedy {
            return Ok((0, use_spec_buf));
        }

        let out_greedy_ids_dev = out_greedy_ids_override.unwrap_or(pre_out_greedy);
        if let Some(comm) = tp2 {
            let local_max_ids_dev = pre_local_ids;
            let local_max_vals_dev = pre_local_vals;
            let global_max_vals_dev = pre_global_vals;
            check_rc(
                // SAFETY: all_logits_buf has num_selected local-vocab rows;
                // local ID/value scratch was reserved for that many rows above.
                // Null top-ID output requests raw local maxima for TP gathering.
                unsafe {
                    ffi::w34_sampler_greedy_top5_f16(
                        &ex,
                        all_logits_buf as *const c_void,
                        local_max_ids_dev as *mut i32,
                        core::ptr::null_mut(),
                        local_max_vals_dev as *mut f32,
                        num_selected as i32,
                        self.local_vocab_size as i32,
                        self.vocab_offset as i32,
                    )
                },
                "w34_sampler_greedy_top5_f16 (tp2 local)",
            )?;
            comm.all_gather_top1_f32_i32(
                &ex,
                local_max_vals_dev,
                local_max_ids_dev,
                global_max_vals_dev,
                out_greedy_ids_dev,
                self.rank as i32,
                num_selected as i32,
            )?;
        } else {
            check_rc(
                // SAFETY: logits and the default ID output were sized for
                // num_selected rows. If overridden, the caller must provide that
                // many device i32 outputs; an override's capacity is not checked.
                unsafe {
                    ffi::w34_sampler_greedy_top5_f16(
                        &ex,
                        all_logits_buf as *const c_void,
                        out_greedy_ids_dev as *mut i32,
                        core::ptr::null_mut(),
                        core::ptr::null_mut(),
                        num_selected as i32,
                        self.local_vocab_size as i32,
                        self.vocab_offset as i32,
                    )
                },
                "w34_sampler_greedy_top5_f16",
            )?;
        }
        Ok((out_greedy_ids_dev, use_spec_buf))
    }

    #[allow(clippy::too_many_arguments, clippy::cognitive_complexity)]
    fn run_qwen4_layer_stack(
        &mut self,
        ex: &W34Exec,
        token_ids_ptr: u64,
        positions_ptr: u64,
        num_tokens: usize,
        attn_batch: &W34AttentionBatch,
        gdn_batch: &W34GdnBatch,
        tp2: Option<&Tp2Collectives>,
        hc_residual_buf: u64,
        normed_buf: u64,
        block_out_buf: u64,
        mlp_out_buf: u64,
        hc_inj_weights_buf: u64,
        final_norm_out_buf: u64,
        [
            qkv_proj_buf,
            q_buf,
            k_buf,
            gate_buf,
            attn_out_buf,
            gated_attn_buf,
        ]: [u64; 6],
        [
            qkvz_buf,
            ba_buf,
            qkv_conv_out_buf,
            gdn_raw_out_buf,
            gdn_normed_buf,
        ]: [u64; 5],
        [
            gate_up_buf,
            act_buf,
            moe_router_logits_buf,
            moe_topk_weights_buf,
            moe_topk_ids_buf,
            moe_interm_buf,
            moe_sh_gu_buf,
            moe_sh_act_buf,
            moe_sh_out_buf,
            moe_sh_gate_buf,
        ]: [u64; 10],
        [ple_buf_a, ple_buf_b, ple_val_buf]: [u64; 3],
        [
            qsa_qk_buf,
            qsa_q_buf,
            qsa_k_buf,
            qsa_sel_phys_buf,
            qsa_sel_log_buf,
            qsa_num_sel_buf,
            qsa_seq_pos_buf,
        ]: [u64; 7],
        layer_overlay_end: u64,
        mut dump_out: Option<&mut ForwardDump>,
    ) -> Result<()> {
        let h_dim = self.cfg.hidden_size;
        let hc_count = self.cfg.hc_count.max(1);
        let hc_lowrank = self.cfg.hc_lowrank;
        let head_dim = self.cfg.head_dim;
        let local_q_heads = self.cfg.num_attention_heads / self.tp_size;
        let local_kv_heads = self.cfg.num_key_value_heads / self.tp_size;
        let q_gate_dim = local_q_heads * head_dim * 2;
        let kv_dim = local_kv_heads * head_dim;
        let local_k_heads = self.cfg.linear_num_key_heads / self.tp_size;
        let local_v_heads = self.cfg.linear_num_value_heads / self.tp_size;
        let head_k_dim = self.cfg.linear_key_head_dim;
        let head_v_dim = self.cfg.linear_value_head_dim;
        let conv_dim = 2 * local_k_heads * head_k_dim + local_v_heads * head_v_dim;
        let gdn_v_dim = local_v_heads * head_v_dim;
        let local_interm = self.cfg.intermediate_size / self.tp_size;
        let marlin_ws_ptr = self.marlin_ws.addr();
        let marlin_ws_ints = self.marlin_ws.len_bytes / 4;

        // 1. Embedding lookup into normed_buf (+ optional visual splice) then replicate across 4 HC streams
        check_rc(
            // SAFETY: normed_buf reserves num_tokens * h_dim FP16 elements.
            unsafe {
                ffi::w34_embed_lookup_f16(
                    ex,
                    token_ids_ptr as *const i32,
                    self.embed_tokens_ptr as *const c_void,
                    normed_buf as *mut c_void,
                    core::ptr::null(),
                    num_tokens as i32,
                    h_dim as i32,
                    self.cfg.vocab_size as i32,
                )
            },
            "w34_embed_lookup_f16 (qwen4)",
        )?;
        if let Some((vis_ptr, idx_ptr, num_vis)) = self.pending_visual_splice
            && num_vis > 0
        {
            check_rc(
                // SAFETY: pending_visual_splice embeddings/indices are live on device.
                unsafe {
                    ffi::w34_embedding_splice_f16(
                        ex,
                        normed_buf as *mut c_void,
                        vis_ptr as *const c_void,
                        idx_ptr as *const i32,
                        num_vis as i32,
                        h_dim as i32,
                    )
                },
                "w34_embedding_splice_f16 (qwen4)",
            )?;
        }
        if let Some(dump) = dump_out.as_deref_mut()
            && self.rank == 0
        {
            Self::record_dump_f16(
                &self.stream,
                normed_buf,
                vec![num_tokens, h_dim],
                "embed",
                dump,
            )?;
        }
        crate::ple::gpu_ops::hc_init_streams_f16(
            ex,
            normed_buf,
            hc_residual_buf,
            0,
            num_tokens,
            hc_count,
            h_dim,
        )?;

        let ple_f8_staging_ptr = self.ple_f8_staging_dev.as_ref().map_or(0, |b| b.addr());
        let ple_conv_state_ptr = self
            .state_pool
            .ple_conv_state
            .as_ref()
            .map_or(0, |b| b.addr());

        for layer_idx in 0..self.layers.len() {
            // 2a. Optional PLE (before attn_hyper_connection.mix)
            if let Some(ple) = &self.layers[layer_idx].ple {
                ple.forward(
                    ex,
                    &self.cublas,
                    marlin_ws_ptr,
                    marlin_ws_ints,
                    ple_f8_staging_ptr,
                    hc_residual_buf,
                    normed_buf,
                    ple_buf_a,
                    ple_buf_b,
                    ple_val_buf,
                    ple_conv_state_ptr,
                    gdn_batch,
                    num_tokens,
                    hc_count,
                    h_dim,
                    self.cfg.rms_norm_eps,
                )?;
            }

            // 2b. Pre-Attention HyperConnection mix
            self.layers[layer_idx]
                .attn_hc
                .as_ref()
                .ok_or_else(|| loader_err("missing attn_hc on qwen4_exp layer"))?
                .forward_mix(
                    ex,
                    &self.cublas,
                    marlin_ws_ptr,
                    marlin_ws_ints,
                    hc_residual_buf,
                    normed_buf,
                    hc_inj_weights_buf,
                    ple_buf_a,
                    ple_buf_b,
                    ple_val_buf,
                    num_tokens,
                    hc_count,
                    hc_lowrank,
                    h_dim,
                    self.cfg.rms_norm_eps,
                )?;

            // 2c. Attention block (QSA FullAttention or GDN LinearAttention)
            match &self.layers[layer_idx].block {
                LayerBlockWeights::FullAttention(fa) => {
                    fa.qkv_proj.forward(
                        ex,
                        &self.cublas,
                        marlin_ws_ptr,
                        marlin_ws_ints,
                        normed_buf,
                        qkv_proj_buf,
                        num_tokens,
                    )?;
                    let qkv_row_stride = (q_gate_dim + 2 * kv_dim) as i32;
                    let q_gate_ptr = qkv_proj_buf;
                    let k_raw_ptr = qkv_proj_buf + (q_gate_dim * 2) as u64;
                    let v_ptr = qkv_proj_buf + ((q_gate_dim + kv_dim) * 2) as u64;
                    let rotary_dim = self.cfg.rotary_dim as i32;
                    check_rc(
                        // SAFETY: q/k/gate regions match local head geometry.
                        unsafe {
                            ffi::w34_fused_qk_rmsnorm_rope_gate_strided_f16(
                                ex,
                                q_gate_ptr as *const c_void,
                                k_raw_ptr as *const c_void,
                                fa.q_norm_ptr as *const c_void,
                                fa.k_norm_ptr as *const c_void,
                                positions_ptr as *const i32,
                                core::ptr::null(),
                                q_buf as *mut c_void,
                                k_buf as *mut c_void,
                                gate_buf as *mut c_void,
                                num_tokens as i32,
                                self.current_pos_plane_stride,
                                local_q_heads as i32,
                                local_kv_heads as i32,
                                head_dim as i32,
                                rotary_dim,
                                self.cfg.mrope_section[0] as i32,
                                self.cfg.mrope_section[1] as i32,
                                self.cfg.mrope_section[2] as i32,
                                qkv_row_stride,
                                qkv_row_stride,
                                self.cfg.rope_theta,
                                self.cfg.rms_norm_eps,
                            )
                        },
                        "w34_fused_qk_rmsnorm_rope_gate_strided_f16 (qwen4)",
                    )?;
                    let kv_bufs = &self.kv_pool.layers[fa.kv_layer_ordinal];
                    let kv_layout = &kv_bufs.layout;
                    check_rc(
                        // SAFETY: K and strided V write into kv_layout.
                        unsafe {
                            ffi::w34_kv_cache_write_strided(
                                ex,
                                kv_layout,
                                attn_batch,
                                k_buf as *const c_void,
                                v_ptr as *const c_void,
                                kv_dim as i32,
                                qkv_row_stride,
                            )
                        },
                        "w34_kv_cache_write_strided (qwen4)",
                    )?;
                    if let Some(qsa_idx) = &fa.qsa_indexer {
                        qsa_idx.index_qk_proj.forward(
                            ex,
                            &self.cublas,
                            marlin_ws_ptr,
                            marlin_ws_ints,
                            normed_buf,
                            qsa_qk_buf,
                            num_tokens,
                        )?;
                        crate::ple::gpu_ops::qsa_indexer_norm_rope_f16(
                            ex,
                            qsa_qk_buf,
                            qsa_idx.q_layernorm_ptr,
                            qsa_idx.k_layernorm_ptr,
                            positions_ptr,
                            qsa_q_buf,
                            qsa_k_buf,
                            0,
                            num_tokens,
                            self.cfg.indexer_n_heads,
                            self.cfg.indexer_kv_heads,
                            self.cfg.indexer_head_dim,
                            self.cfg.rotary_dim,
                            self.cfg.rope_theta,
                            self.cfg.rms_norm_eps,
                            self.cfg.mrope_section[0],
                            self.cfg.mrope_section[1],
                            self.current_pos_plane_stride as usize,
                        )?;
                        let raw_k_cache_ptr = kv_bufs
                            .qsa_raw_k_buf
                            .as_ref()
                            .ok_or_else(|| loader_err("missing qsa_raw_k_buf"))?
                            .addr();
                        let comp_k_cache_ptr = kv_bufs
                            .qsa_comp_k_buf
                            .as_ref()
                            .ok_or_else(|| loader_err("missing qsa_comp_k_buf"))?
                            .addr();
                        crate::ple::gpu_ops::qsa_indexer_cache_update_f16(
                            ex,
                            qsa_k_buf,
                            raw_k_cache_ptr,
                            comp_k_cache_ptr,
                            attn_batch.write_slot_mapping as usize as u64,
                            0,
                            num_tokens,
                            self.cfg.indexer_head_dim,
                        )?;
                        crate::ple::gpu_ops::qsa_select_blocks_f16(
                            ex,
                            qsa_q_buf,
                            comp_k_cache_ptr,
                            attn_batch,
                            qsa_sel_phys_buf,
                            qsa_sel_log_buf,
                            qsa_num_sel_buf,
                            qsa_seq_pos_buf,
                            self.cfg.indexer_n_heads,
                            self.cfg.indexer_head_dim,
                            16,
                            64,
                            4,
                            8,
                        )?;
                    }
                    if fa.qsa_indexer.is_some()
                        && kv_layout.k_format == ffi::W34_KV_FORMAT_FP16
                        && kv_layout.v_format == ffi::W34_KV_FORMAT_FP16
                    {
                        crate::ple::gpu_ops::qsa_sparse_paged_attn_f16(
                            ex,
                            q_buf,
                            gate_buf,
                            kv_layout,
                            qsa_sel_phys_buf,
                            qsa_sel_log_buf,
                            qsa_num_sel_buf,
                            qsa_seq_pos_buf,
                            0,
                            gated_attn_buf,
                            num_tokens,
                            local_q_heads,
                            64,
                            attn_batch.sm_scale,
                        )?;
                    } else {
                        check_rc(
                            // SAFETY: fallback to paged_attention + output gate when KV is TQ4/Q8_0.
                            unsafe {
                                ffi::w34_paged_attention_f16(
                                    ex,
                                    kv_layout,
                                    attn_batch,
                                    q_buf as *const c_void,
                                    attn_out_buf as *mut c_void,
                                    local_q_heads as i32,
                                )
                            },
                            "w34_paged_attention_f16 (qwen4)",
                        )?;
                        check_rc(
                            // SAFETY: apply sigmoid output gate on attn_out_buf.
                            unsafe {
                                ffi::w34_apply_attn_output_gate_f16(
                                    ex,
                                    attn_out_buf as *const c_void,
                                    gate_buf as *const c_void,
                                    gated_attn_buf as *mut c_void,
                                    core::ptr::null(),
                                    num_tokens as i32,
                                    (local_q_heads * head_dim) as i32,
                                )
                            },
                            "w34_apply_attn_output_gate_f16 (qwen4)",
                        )?;
                    }
                    fa.o_proj.forward(
                        ex,
                        &self.cublas,
                        marlin_ws_ptr,
                        marlin_ws_ints,
                        gated_attn_buf,
                        block_out_buf,
                        num_tokens,
                    )?;
                }
                LayerBlockWeights::LinearAttention(gdn) => {
                    let can_fork_ba = !matches!(
                        gdn.in_proj_ba,
                        LinearProj::DenseF16 { size_n, size_k, .. }
                            if size_n > 256 || !size_k.is_multiple_of(8)
                    );
                    if can_fork_ba {
                        self.aux_fork_event.record_in_current_ctx(&self.stream)?;
                        self.aux_stream.wait_event(&self.aux_fork_event)?;
                        let ex_aux = self.aux_stream.exec(0, 0);
                        gdn.in_proj_ba.forward(
                            &ex_aux,
                            &self.cublas,
                            self.aux_marlin_ws.addr(),
                            self.aux_marlin_ws.len_bytes / 4,
                            normed_buf,
                            ba_buf,
                            num_tokens,
                        )?;
                        self.aux_join_event
                            .record_in_current_ctx(&self.aux_stream)?;
                    }
                    gdn.in_proj_qkvz.forward(
                        ex,
                        &self.cublas,
                        marlin_ws_ptr,
                        marlin_ws_ints,
                        normed_buf,
                        qkvz_buf,
                        num_tokens,
                    )?;
                    if !can_fork_ba {
                        gdn.in_proj_ba.forward(
                            ex,
                            &self.cublas,
                            marlin_ws_ptr,
                            marlin_ws_ints,
                            normed_buf,
                            ba_buf,
                            num_tokens,
                        )?;
                    }
                    let qkvz_row_stride = (conv_dim + gdn_v_dim) as i32;
                    let ba_row_stride = (2 * local_v_heads) as i32;
                    let qkv_ptr = qkvz_buf;
                    let z_ptr = qkvz_buf + (conv_dim * 2) as u64;
                    let b_ptr = ba_buf;
                    let a_ptr = ba_buf + (local_v_heads * 2) as u64;
                    let st_bufs = &self.state_pool.layers[gdn.gdn_layer_ordinal];
                    check_rc(
                        // SAFETY: GDN causal conv1d on strided QKV prefix.
                        unsafe {
                            ffi::w34_gdn_causal_conv1d_strided_f16(
                                ex,
                                gdn_batch,
                                qkv_ptr as *const c_void,
                                gdn.conv_weight_ptr as *const c_void,
                                core::ptr::null(),
                                st_bufs.conv_state.addr() as *mut c_void,
                                qkv_conv_out_buf as *mut c_void,
                                conv_dim as i32,
                                qkvz_row_stride,
                                self.cfg.linear_conv_kernel_dim as i32,
                                self.state_pool.total_slots as i32,
                                1,
                            )
                        },
                        "w34_gdn_causal_conv1d_strided_f16 (qwen4)",
                    )?;
                    if can_fork_ba {
                        self.stream.wait_event(&self.aux_join_event)?;
                    }
                    let gdn_tail_cap = layer_overlay_end.saturating_sub(gdn_normed_buf) as usize;
                    let ex_gdn = if gdn_tail_cap > ex.scratch_size {
                        self.stream.exec(gdn_normed_buf, gdn_tail_cap)
                    } else {
                        *ex
                    };
                    check_rc(
                        // SAFETY: GDN recurrent step on strided conv output + BA.
                        unsafe {
                            ffi::w34_gdn_recurrent_strided_f16(
                                &ex_gdn,
                                gdn_batch,
                                qkv_conv_out_buf as *const c_void,
                                core::ptr::null(),
                                core::ptr::null(),
                                core::ptr::null(),
                                b_ptr as *const c_void,
                                a_ptr as *const c_void,
                                gdn.a_log_ptr as *const f32,
                                gdn.dt_bias_ptr as *const f32,
                                st_bufs.ssm_state.addr() as *mut f32,
                                gdn_raw_out_buf as *mut c_void,
                                local_k_heads as i32,
                                head_k_dim as i32,
                                local_v_heads as i32,
                                head_v_dim as i32,
                                ba_row_stride,
                                self.state_pool.total_slots as i32,
                            )
                        },
                        "w34_gdn_recurrent_strided_f16 (qwen4)",
                    )?;
                    if self.cfg.output_gate_type.eq_ignore_ascii_case("sigmoid") {
                        check_rc(
                            // SAFETY: gated RMSNorm + sigmoid on GDN output (`output_gate_type == "sigmoid"`).
                            unsafe {
                                ffi::w34_rms_norm_gated_sigmoid_strided_f16(
                                    ex,
                                    gdn_raw_out_buf as *const c_void,
                                    z_ptr as *const c_void,
                                    gdn.norm_weight_ptr as *const c_void,
                                    gdn_normed_buf as *mut c_void,
                                    core::ptr::null(),
                                    num_tokens as i32,
                                    local_v_heads as i32,
                                    head_v_dim as i32,
                                    qkvz_row_stride,
                                    self.cfg.rms_norm_eps,
                                )
                            },
                            "w34_rms_norm_gated_sigmoid_strided_f16 (qwen4)",
                        )?;
                    } else {
                        check_rc(
                            // SAFETY: gated RMSNorm + SiLU on GDN output.
                            unsafe {
                                ffi::w34_rms_norm_gated_silu_strided_f16(
                                    ex,
                                    gdn_raw_out_buf as *const c_void,
                                    z_ptr as *const c_void,
                                    gdn.norm_weight_ptr as *const c_void,
                                    gdn_normed_buf as *mut c_void,
                                    core::ptr::null(),
                                    num_tokens as i32,
                                    local_v_heads as i32,
                                    head_v_dim as i32,
                                    qkvz_row_stride,
                                    self.cfg.rms_norm_eps,
                                )
                            },
                            "w34_rms_norm_gated_silu_strided_f16 (qwen4)",
                        )?;
                    }
                    gdn.out_proj.forward(
                        ex,
                        &self.cublas,
                        marlin_ws_ptr,
                        marlin_ws_ints,
                        gdn_normed_buf,
                        block_out_buf,
                        num_tokens,
                    )?;
                }
            }

            if let Some(comm) = tp2 {
                comm.all_reduce_f16(
                    ex,
                    block_out_buf,
                    block_out_buf,
                    self.rank as i32,
                    (num_tokens * h_dim) as i32,
                )?;
            }

            // 2d. Post-Attention HyperConnection combine
            self.layers[layer_idx]
                .attn_hc
                .as_ref()
                .unwrap()
                .forward_combine(
                    ex,
                    block_out_buf,
                    hc_inj_weights_buf,
                    hc_residual_buf,
                    num_tokens,
                    hc_count,
                    h_dim,
                )?;

            // 2e. Pre-MLP HyperConnection mix
            self.layers[layer_idx]
                .mlp_hc
                .as_ref()
                .ok_or_else(|| loader_err("missing mlp_hc on qwen4_exp layer"))?
                .forward_mix(
                    ex,
                    &self.cublas,
                    marlin_ws_ptr,
                    marlin_ws_ints,
                    hc_residual_buf,
                    normed_buf,
                    hc_inj_weights_buf,
                    ple_buf_a,
                    ple_buf_b,
                    ple_val_buf,
                    num_tokens,
                    hc_count,
                    hc_lowrank,
                    h_dim,
                    self.cfg.rms_norm_eps,
                )?;

            // 2f. MLP block (Tiered 512-expert MoE + shared expert)
            self.layers[layer_idx].ffn.forward(
                ex,
                &self.cublas,
                marlin_ws_ptr,
                marlin_ws_ints,
                normed_buf,
                mlp_out_buf,
                num_tokens,
                h_dim,
                local_interm,
                gate_up_buf,
                act_buf,
                moe_router_logits_buf,
                moe_topk_weights_buf,
                moe_topk_ids_buf,
                moe_interm_buf,
                moe_sh_gu_buf,
                moe_sh_act_buf,
                moe_sh_out_buf,
                moe_sh_gate_buf,
            )?;
            let tiered_info = match &self.layers[layer_idx].ffn {
                FfnBlockWeights::TieredMoe(tmoe) => Some((
                    tmoe.top_k,
                    if tmoe.shared_down_proj.is_some() {
                        moe_sh_out_buf
                    } else {
                        0
                    },
                    if tmoe.shared_expert_gate.is_some() {
                        moe_sh_gate_buf
                    } else {
                        0
                    },
                )),
                _ => None,
            };
            if let Some((top_k, sh_out_ptr, sh_gate_ptr)) = tiered_info {
                let is_capturing = self.is_capturing_graph;
                let tiered_moe = self
                    .tiered_moe
                    .as_mut()
                    .ok_or_else(|| loader_err("missing tiered_moe on qwen4_exp rank"))?;
                let use_device_lru = is_capturing
                    || (num_tokens * top_k <= tiered_moe.layer_hot_k(layer_idx)
                        && tiered_moe.layer_has_mapped_store(layer_idx));
                if use_device_lru {
                    tiered_moe.execute_layer_device_lru(
                        &self.stream,
                        layer_idx,
                        normed_buf,
                        moe_topk_ids_buf,
                        moe_topk_weights_buf,
                        sh_out_ptr,
                        sh_gate_ptr,
                        mlp_out_buf,
                        num_tokens,
                        top_k,
                    )?;
                } else {
                    tiered_moe.execute_layer(
                        &self.stream,
                        layer_idx,
                        normed_buf,
                        moe_topk_ids_buf,
                        moe_topk_weights_buf,
                        sh_out_ptr,
                        sh_gate_ptr,
                        mlp_out_buf,
                        num_tokens,
                        top_k,
                    )?;
                }
            }
            if let Some(comm) = tp2 {
                comm.all_reduce_f16(
                    ex,
                    mlp_out_buf,
                    mlp_out_buf,
                    self.rank as i32,
                    (num_tokens * h_dim) as i32,
                )?;
            }

            // 2g. Post-MLP HyperConnection combine
            self.layers[layer_idx]
                .mlp_hc
                .as_ref()
                .unwrap()
                .forward_combine(
                    ex,
                    mlp_out_buf,
                    hc_inj_weights_buf,
                    hc_residual_buf,
                    num_tokens,
                    hc_count,
                    h_dim,
                )?;

            if let Some(dump) = dump_out.as_deref_mut()
                && self.rank == 0
            {
                Self::record_dump_f16(
                    &self.stream,
                    hc_residual_buf,
                    vec![num_tokens, hc_count * h_dim],
                    &format!("layer_{layer_idx}_hidden_after"),
                    dump,
                )?;
            }
        }

        // 3. Export pre-final-mixer 4-stream HC state `[num_tokens, hc_count * h_dim]` for Qwen4 MTP
        if num_tokens <= self.max_tap_rows
            && let Some(final_h_buf) = &self.last_final_hidden_f16
        {
            check_rc(
                // SAFETY: `hc_residual_buf` and `final_h_buf` both hold `num_tokens * hc_count * h_dim` FP16 elements.
                unsafe {
                    ffi::w34_memcpy_d2d_async(
                        final_h_buf.addr() as *mut c_void,
                        hc_residual_buf as *const c_void,
                        num_tokens * hc_count * h_dim * 2,
                        self.stream.handle,
                    )
                },
                "qwen4_exp export last_final_hidden_f16",
            )?;
        }

        // 4. Final HyperConnection mixer -> final_norm_out_buf [num_tokens, h_dim]
        self.final_hc_mixer
            .as_ref()
            .ok_or_else(|| loader_err("missing final_hc_mixer on qwen4_exp rank"))?
            .forward_mix(
                ex,
                &self.cublas,
                marlin_ws_ptr,
                marlin_ws_ints,
                hc_residual_buf,
                final_norm_out_buf,
                0,
                ple_buf_a,
                ple_buf_b,
                ple_val_buf,
                num_tokens,
                hc_count,
                hc_lowrank,
                h_dim,
                self.cfg.rms_norm_eps,
            )?;
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
        if self.cfg.is_qwen4_exp() {
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
        let mut host_u16 = vec![0u16; count];
        check_rc(
            // SAFETY: callers pass a live FP16 region matching shape; host_u16
            // reserves exactly count elements. The D2H is followed by stream
            // synchronization before reading or dropping the host destination.
            unsafe {
                ffi::w34_memcpy_d2h_async(
                    host_u16.as_mut_ptr() as *mut c_void,
                    dev_ptr as *const c_void,
                    count * 2,
                    stream.handle,
                )
            },
            "record_dump_f16 d2h",
        )?;
        stream.synchronize()?;
        let vals_f32: Vec<f32> = host_u16.into_iter().map(f16_to_f32).collect();
        dump.tensors_f32.insert(name.to_string(), (shape, vals_f32));
        Ok(())
    }
}
