use core::ffi::c_void;
use w34_kernels::ffi::{self, W34_ATTN_MASK_CAUSAL, W34AttentionBatch, W34Exec};

use crate::cuda::{CudaGraph, Result, Tp2Collectives, check_rc};
use crate::model::{
    FfnBlockWeights, HADAMARD_BLOCK_64, HcStepContext, LinearProj,
    PRODUCER_FUSED_H64_MAX_HIDDEN_SIZE, Qwen3_5ModelRank, TapExport, loader_err,
};

use super::{MtpInputFusion, Qwen3_5MtpRank};

impl Qwen3_5MtpRank {
    pub fn has_propose_graph(&self, k: usize) -> bool {
        let long_ctx = self.len + k > 1024;
        let idx = if self.batch_mode { self.active_slot } else { 0 };
        self.graph_io.propose_graphs[idx]
            .as_ref()
            .is_some_and(|(cap_k, cap_long, _)| *cap_k == k && *cap_long == long_ctx)
    }

    fn validate_propose_args(
        &self,
        target: &Qwen3_5ModelRank,
        anchor: i32,
        target_hidden: u64,
        k: usize,
        tp2_expected: Option<bool>,
    ) -> Result<()> {
        if target.embed_tokens_ptr != self.target_embedding {
            return Err(loader_err("MTP target rank mismatch"));
        }
        if let Some(has_tp2) = tp2_expected
            && (target.tp_size > 1) != has_tp2
        {
            return Err(loader_err("invalid MTP TP2 collective state"));
        }
        let span = self.hidden_size.checked_mul(2);
        if !(1..=3).contains(&k)
            || self.len != self.committed
            || self.len + k > self.capacity
            || anchor < 0
            || anchor as usize >= target.cfg.vocab_size
            || target_hidden == 0
            || !target_hidden.is_multiple_of(2)
            || span
                .and_then(|n| target_hidden.checked_add(n as u64))
                .is_none()
        {
            return Err(loader_err("invalid MTP proposal state or k"));
        }
        Ok(())
    }

    fn enqueue_propose_steps(
        &mut self,
        t: &Qwen3_5ModelRank,
        k: usize,
        long_ctx: bool,
        tp2: Option<&Tp2Collectives>,
    ) -> Result<()> {
        self.graph_io.propose_arena.reset();
        let h_dim = t.cfg.hidden_size;
        let head_dim = t.cfg.head_dim;
        let local_q_heads = t.cfg.num_attention_heads / t.tp_size;
        let local_kv_heads = t.cfg.local_kv_heads(t.tp_size);
        let local_interm = t.cfg.intermediate_size / t.tp_size;
        let q_gate_dim = local_q_heads * head_dim * 2;
        let kv_dim = local_kv_heads * head_dim;
        let arena = &mut self.graph_io.propose_arena;
        let residual_buf = arena.alloc_aligned(h_dim * 2)?;
        let normed_buf = arena.alloc_aligned(h_dim * 2)?;
        let embed = arena.alloc_aligned(h_dim * 2)?;
        let concat = arena.alloc_aligned((2 * h_dim) * 2)?;
        let qkv_proj_buf = arena.alloc_aligned((q_gate_dim + 2 * kv_dim) * 2)?;
        let q_buf = arena.alloc_aligned((local_q_heads * head_dim) * 2)?;
        let k_buf = arena.alloc_aligned(kv_dim * 2)?;
        let gate_buf = arena.alloc_aligned((local_q_heads * head_dim) * 2)?;
        let attn_out_buf = arena.alloc_aligned((local_q_heads * head_dim) * 2)?;
        let gated_attn_buf = arena.alloc_aligned((local_q_heads * head_dim) * 2)?;
        let block_out_buf = arena.alloc_aligned(h_dim * 2)?;
        let mlp_out_buf = arena.alloc_aligned(h_dim * 2)?;
        let (
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
        ) = match &self.ffn {
            FfnBlockWeights::Dense { .. } => (
                arena.alloc_aligned((2 * local_interm) * 2)?,
                arena.alloc_aligned(local_interm * 2)?,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
            ),
            FfnBlockWeights::Moe(moe) => {
                let sh_gu = if moe.local_shared_interm > 0 {
                    arena.alloc_aligned(2 * moe.local_shared_interm * 2)?
                } else {
                    0
                };
                let sh_act = if moe.local_shared_interm > 0 {
                    arena.alloc_aligned(moe.local_shared_interm * 2)?
                } else {
                    0
                };
                let sh_out = if moe.local_shared_interm > 0 {
                    arena.alloc_aligned(h_dim * 2)?
                } else {
                    0
                };
                let sh_gate = if moe.shared_expert_gate.is_some() {
                    arena.alloc_aligned(2)?
                } else {
                    0
                };
                let moe_interm =
                    if matches!(moe.experts, crate::model::MoeExpertStorage::Tiered { .. }) {
                        0
                    } else {
                        arena.alloc_aligned(moe.top_k * moe.local_moe_interm * 2)?
                    };
                (
                    0,
                    0,
                    arena.alloc_aligned(moe.num_experts * 2)?,
                    arena.alloc_aligned(moe.top_k * 4)?,
                    arena.alloc_aligned(moe.top_k * 4)?,
                    moe_interm,
                    sh_gu,
                    sh_act,
                    sh_out,
                    sh_gate,
                )
            }
        };
        let logits_buf = arena.alloc_aligned(t.local_vocab_size * 2)?;
        let local_id = arena.alloc_aligned(4)?;
        let local_val = arena.alloc_aligned(4)?;
        let global_val = arena.alloc_aligned(4)?;
        let (ws_ptr, ws_cap) = arena.remaining_workspace();
        let ex = t.stream.exec(ws_ptr, ws_cap);

        let meta_base = self.graph_io.meta_dev.addr();
        let token_ids_ptr = meta_base;
        let qs_ptr = meta_base + 10 * 4;
        let qkv_row_stride = (q_gate_dim + 2 * kv_dim) as i32;
        let q_gate_ptr = qkv_proj_buf;
        let k_raw_ptr = qkv_proj_buf + (q_gate_dim * 2) as u64;
        let v_ptr = qkv_proj_buf + ((q_gate_dim + kv_dim) * 2) as u64;
        let kv_layout = &self.kv.layers[self.fa.kv_layer_ordinal].layout;

        for step in 0..k {
            let positions_ptr = meta_base + ((1 + 3 * step) * 4) as u64;
            let slot_ptr = meta_base + ((2 + 3 * step) * 4) as u64;
            let after_ptr = meta_base + ((3 + 3 * step) * 4) as u64;
            let out_id = self.graph_io.out_ids_dev.addr() + (step * 4) as u64;
            let attn_batch = W34AttentionBatch {
                query_start_loc: qs_ptr as *const i32,
                context_lens_before: positions_ptr as *const i32,
                seq_lens_after: after_ptr as *const i32,
                block_tables: self.pages_dev as *const i32,
                write_slot_mapping: slot_ptr as *const i32,
                valid_token_mask: core::ptr::null(),
                valid_req_mask: core::ptr::null(),
                num_reqs: 1,
                total_tokens: 1,
                max_blocks_per_req: self.pages.len() as i32,
                max_q_len: 1,
                max_seq_len: if long_ctx {
                    self.capacity as i32
                } else {
                    self.capacity.min(1024) as i32
                },
                mask_mode: W34_ATTN_MASK_CAUSAL,
                sliding_window: 0,
                sm_scale: (head_dim as f32).powf(-0.5),
                reserved: 0,
            };
            check_rc(
                // SAFETY: token_ids_ptr contains valid token ID, embed_tokens_ptr is valid device memory of vocab_size * h_dim half elements, embed has capacity for h_dim half elements.
                unsafe {
                    ffi::w34_embed_lookup_f16(
                        &ex,
                        token_ids_ptr as *const i32,
                        t.embed_tokens_ptr as *const c_void,
                        embed as *mut c_void,
                        core::ptr::null(),
                        1,
                        h_dim as i32,
                        t.cfg.vocab_size as i32,
                    )
                },
                "MTP embed",
            )?;
            for (input, weight, output) in [
                (embed, self.norms[0], concat),
                (
                    self.hidden.addr(),
                    self.norms[1],
                    concat + (h_dim * 2) as u64,
                ),
            ] {
                check_rc(
                    // SAFETY: input and output are valid device pointers with capacity for h_dim half elements, weight is valid norm weight.
                    unsafe {
                        ffi::w34_gemma_rms_norm_f16(
                            &ex,
                            input as *const c_void,
                            weight as *const c_void,
                            output as *mut c_void,
                            core::ptr::null(),
                            1,
                            h_dim as i32,
                            t.cfg.rms_norm_eps,
                        )
                    },
                    "MTP pre FC norm",
                )?;
            }
            Self::project(self.concat_fc()?, &ex, t, concat, residual_buf, 1)?;
            let fa = &self.fa;
            let fuse_qkv_h64 = fa.qsa_indexer.is_none()
                && h_dim <= PRODUCER_FUSED_H64_MAX_HIDDEN_SIZE
                && h_dim.is_multiple_of(HADAMARD_BLOCK_64)
                && fa.qkv_proj.can_fuse_h64_from_producer();
            let [qkv_norm_out, qkv_norm_had64] = if fuse_qkv_h64 {
                [0, normed_buf]
            } else {
                [normed_buf, 0]
            };
            check_rc(
                // SAFETY: residual_buf and normed_buf are valid device buffers for h_dim half elements,
                // input_layernorm_ptr is valid norm weight on active stream.
                unsafe {
                    ffi::w34_gemma_rms_norm_hadamard_f16(
                        &ex,
                        residual_buf as *const c_void,
                        self.residual
                            .single_stream_input_norm_ptr()
                            .ok_or_else(|| {
                                loader_err("MTP graph requires single-stream residual")
                            })? as *const c_void,
                        qkv_norm_out as *mut c_void,
                        qkv_norm_had64 as *mut c_void,
                        core::ptr::null(),
                        1,
                        h_dim as i32,
                        t.cfg.rms_norm_eps,
                    )
                },
                "MTP input norm",
            )?;
            Self::project_maybe_prerotated(
                &fa.qkv_proj,
                fuse_qkv_h64,
                &ex,
                t,
                normed_buf,
                qkv_proj_buf,
                1,
            )?;
            check_rc(
                // SAFETY: q_gate_ptr, k_raw_ptr, and output buffers are valid strided device buffers for 1 token, positions_ptr contains valid position on t.stream.
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
                        1,
                        0,
                        local_q_heads as i32,
                        local_kv_heads as i32,
                        head_dim as i32,
                        t.cfg.rotary_dim as i32,
                        t.cfg.mrope_section[0] as i32,
                        t.cfg.mrope_section[1] as i32,
                        t.cfg.mrope_section[2] as i32,
                        qkv_row_stride,
                        qkv_row_stride,
                        t.cfg.rope_theta,
                        t.cfg.rms_norm_eps,
                    )
                },
                "w34_fused_qk_rmsnorm_rope_gate_strided_f16",
            )?;
            check_rc(
                // SAFETY: kv_layout describes valid paged memory, k_buf and v_ptr are valid device buffers with capacity for kv_dim elements.
                unsafe {
                    ffi::w34_kv_cache_write_strided(
                        &ex,
                        kv_layout,
                        &attn_batch,
                        k_buf as *const c_void,
                        v_ptr as *const c_void,
                        kv_dim as i32,
                        qkv_row_stride,
                    )
                },
                "w34_kv_cache_write_strided",
            )?;
            check_rc(
                // SAFETY: kv_layout and attn_batch describe valid paged attention batch, q_buf and attn_out_buf have capacity for local_q_heads * head_dim elements.
                unsafe {
                    ffi::w34_paged_attention_f16(
                        &ex,
                        kv_layout,
                        &attn_batch,
                        q_buf as *const c_void,
                        attn_out_buf as *mut c_void,
                        local_q_heads as i32,
                    )
                },
                "w34_paged_attention_f16",
            )?;
            let fuse_o_h64 = fa.o_proj.can_fuse_h64_from_producer();
            let gate_fn = if fuse_o_h64 {
                ffi::w34_apply_attn_output_gate_hadamard_f16
            } else {
                ffi::w34_apply_attn_output_gate_f16
            };
            check_rc(
                // SAFETY: attn_out_buf and gate_buf are valid device buffers of local_q_heads * head_dim half elements, gated_attn_buf has equal capacity.
                unsafe {
                    gate_fn(
                        &ex,
                        attn_out_buf as *const c_void,
                        gate_buf as *const c_void,
                        gated_attn_buf as *mut c_void,
                        core::ptr::null(),
                        1,
                        (local_q_heads * head_dim) as i32,
                    )
                },
                "w34_apply_attn_output_gate_f16",
            )?;
            Self::project_maybe_prerotated(
                &fa.o_proj,
                fuse_o_h64,
                &ex,
                t,
                gated_attn_buf,
                block_out_buf,
                1,
            )?;
            let post_ln_ptr = self
                .residual
                .single_stream_post_attn_norm_ptr()
                .ok_or_else(|| loader_err("MTP graph requires single-stream residual"))?;
            let [mlp_norm_out, mlp_norm_had64] =
                self.ffn
                    .pre_mlp_norm_targets(normed_buf, mlp_out_buf, &ex, 1, h_dim, true);
            if let Some(comm) = tp2 {
                comm.all_reduce_residual_gemma_rms_norm_hadamard_f16(
                    &ex,
                    block_out_buf,
                    0,
                    residual_buf,
                    post_ln_ptr,
                    mlp_norm_out,
                    mlp_norm_had64,
                    0,
                    t.rank as i32,
                    1,
                    h_dim as i32,
                    t.cfg.rms_norm_eps,
                    0,
                    0,
                )?;
            } else {
                check_rc(
                    // SAFETY: block_out_buf and residual_buf are valid device buffers for h_dim half elements, post_ln_ptr is valid norm weight, normed_buf has capacity for h_dim half elements.
                    unsafe {
                        ffi::w34_fused_add_gemma_rms_norm_tap_hadamard_f16(
                            &ex,
                            block_out_buf as *const c_void,
                            residual_buf as *mut c_void,
                            post_ln_ptr as *const c_void,
                            mlp_norm_out as *mut c_void,
                            mlp_norm_had64 as *mut c_void,
                            core::ptr::null_mut(),
                            core::ptr::null(),
                            1,
                            h_dim as i32,
                            t.cfg.rms_norm_eps,
                            0,
                            0,
                        )
                    },
                    "w34_fused_add_gemma_rms_norm_tap_hadamard_f16 (post_attn)",
                )?;
            }
            match &self.ffn {
                FfnBlockWeights::Dense {
                    gate_up_proj,
                    down_proj,
                } => {
                    Self::project_maybe_prerotated(
                        gate_up_proj,
                        mlp_norm_had64 != 0,
                        &ex,
                        t,
                        normed_buf,
                        gate_up_buf,
                        1,
                    )?;
                    let fuse_down_h64 = down_proj.can_fuse_h64_from_producer();
                    let silu_fn = if fuse_down_h64 {
                        ffi::w34_silu_and_mul_hadamard_f16
                    } else {
                        ffi::w34_silu_and_mul_f16
                    };
                    check_rc(
                        // SAFETY: gate_up_buf and act_buf are valid device buffers for 1 row of local_interm elements on active stream.
                        unsafe {
                            silu_fn(
                                &ex,
                                gate_up_buf as *const c_void,
                                act_buf as *mut c_void,
                                core::ptr::null(),
                                1,
                                local_interm as i32,
                            )
                        },
                        "w34_silu_and_mul_f16",
                    )?;
                    Self::project_maybe_prerotated(
                        down_proj,
                        fuse_down_h64,
                        &ex,
                        t,
                        act_buf,
                        mlp_out_buf,
                        1,
                    )?;
                }
                FfnBlockWeights::Moe(_) => {
                    self.ffn.forward_with_prerotated_input(
                        &ex,
                        &t.cublas,
                        t.marlin_ws.addr(),
                        t.marlin_ws.len_bytes / 4,
                        normed_buf,
                        mlp_out_buf,
                        1,
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
                        mlp_norm_had64,
                    )?;
                }
            }
            if let Some(comm) = tp2 {
                comm.all_reduce_residual_gemma_rms_norm_f16(
                    &ex,
                    mlp_out_buf,
                    0,
                    residual_buf,
                    self.final_residual
                        .single_stream_norm_ptr()
                        .ok_or_else(|| loader_err("MTP graph requires single-stream final norm"))?,
                    self.hidden.addr(),
                    0,
                    t.rank as i32,
                    1,
                    h_dim as i32,
                    t.cfg.rms_norm_eps,
                    0,
                    0,
                )?;
            } else {
                check_rc(
                    // SAFETY: mlp_out_buf and residual_buf are valid device buffers for h_dim half elements, final norm weight is valid, hidden buffer has capacity for h_dim half elements.
                    unsafe {
                        ffi::w34_fused_add_gemma_rms_norm_f16(
                            &ex,
                            mlp_out_buf as *const c_void,
                            residual_buf as *mut c_void,
                            self.final_residual
                                .single_stream_norm_ptr()
                                .ok_or_else(|| {
                                    loader_err("MTP graph requires single-stream final norm")
                                })? as *const c_void,
                            self.hidden.addr() as *mut c_void,
                            core::ptr::null(),
                            1,
                            h_dim as i32,
                            t.cfg.rms_norm_eps,
                        )
                    },
                    "MTP final norm",
                )?;
            }
            if step + 1 == k {
                check_rc(
                    // SAFETY: batch_hidden and hidden are valid device buffers with capacity for h_dim * 2 bytes on t.stream.
                    unsafe {
                        ffi::w34_memcpy_d2d_async(
                            self.batch_hidden.addr() as *mut c_void,
                            self.hidden.addr() as *const c_void,
                            h_dim * 2,
                            t.stream.handle,
                        )
                    },
                    "MTP batch hidden tail",
                )?;
            }
            Self::project(self.lm_head(t), &ex, t, self.hidden.addr(), logits_buf, 1)?;
            check_rc(
                // SAFETY: logits_buf contains local_vocab_size valid half logits, output pointers have capacity for argmax results.
                unsafe {
                    ffi::w34_sampler_greedy_top5_f16(
                        &ex,
                        logits_buf as *const c_void,
                        if tp2.is_some() { local_id } else { out_id } as *mut i32,
                        core::ptr::null_mut(),
                        if tp2.is_some() {
                            local_val as *mut f32
                        } else {
                            core::ptr::null_mut()
                        },
                        1,
                        t.local_vocab_size as i32,
                        t.vocab_offset as i32,
                    )
                },
                "MTP greedy",
            )?;
            if let Some(comm) = tp2 {
                comm.all_gather_top1_f32_i32(
                    &ex,
                    local_val,
                    local_id,
                    global_val,
                    out_id,
                    t.rank as i32,
                    1,
                )?;
            }
            if step + 1 < k {
                check_rc(
                    // SAFETY: token_ids_ptr and out_id are valid device pointers with capacity for 4 bytes on t.stream.
                    unsafe {
                        ffi::w34_memcpy_d2d_async(
                            token_ids_ptr as *mut c_void,
                            out_id as *const c_void,
                            4,
                            t.stream.handle,
                        )
                    },
                    "MTP recurrent draft token",
                )?;
            }
        }
        Ok(())
    }

    pub fn ensure_propose_graph(
        &mut self,
        target: &Qwen3_5ModelRank,
        k: usize,
        tp2: Option<&Tp2Collectives>,
    ) -> Result<()> {
        if self.has_propose_graph(k) {
            return Ok(());
        }
        let long_ctx = self.len + k > 1024;
        target.device.make_current()?;
        target.stream.synchronize()?;
        let graph = CudaGraph::capture(&target.stream, || {
            self.enqueue_propose_steps(target, k, long_ctx, tp2)
        })?;
        let idx = if self.batch_mode { self.active_slot } else { 0 };
        self.graph_io.propose_graphs[idx] = Some((k, long_ctx, graph));
        Ok(())
    }

    pub fn prepare_propose_graph_inputs_async(
        &mut self,
        target: &Qwen3_5ModelRank,
        anchor: i32,
        target_hidden: u64,
        k: usize,
    ) -> Result<()> {
        self.validate_propose_args(target, anchor, target_hidden, k, None)?;
        target.device.make_current()?;
        let mut meta = [0i32; 12];
        meta[0] = anchor;
        for s in 0..3 {
            let p = self.len + s.min(k - 1);
            meta[1 + 3 * s] = p as i32;
            meta[2 + 3 * s] = self.pages[p / 16] * 16 + (p % 16) as i32;
            meta[3 + 3 * s] = (p + 1) as i32;
        }
        meta[10] = 0;
        meta[11] = 1;
        self.graph_io
            .meta_dev
            .copy_from_host(&meta, &target.stream)?;
        if target_hidden != self.hidden.addr() {
            check_rc(
                // SAFETY: hidden and target_hidden are valid device buffers with capacity for hidden_size * 2 bytes on target.stream.
                unsafe {
                    ffi::w34_memcpy_d2d_async(
                        self.hidden.addr() as *mut c_void,
                        target_hidden as *const c_void,
                        self.hidden_size * 2,
                        target.stream.handle,
                    )
                },
                "MTP seed hidden",
            )?;
        }
        Ok(())
    }

    pub fn replay_propose_graph_async(&self, target: &Qwen3_5ModelRank, k: usize) -> Result<()> {
        let long_ctx = self.len + k > 1024;
        target.device.make_current()?;
        let idx = if self.batch_mode { self.active_slot } else { 0 };
        let (_, _, graph) = self.graph_io.propose_graphs[idx]
            .as_ref()
            .filter(|(cap_k, cap_long, _)| *cap_k == k && *cap_long == long_ctx)
            .ok_or_else(|| loader_err("MTP propose graph not captured"))?;
        graph.launch(&target.stream)
    }

    pub fn launch_propose_graph_async(
        &mut self,
        target: &Qwen3_5ModelRank,
        anchor: i32,
        target_hidden: u64,
        k: usize,
    ) -> Result<()> {
        self.prepare_propose_graph_inputs_async(target, anchor, target_hidden, k)?;
        self.replay_propose_graph_async(target, k)
    }

    pub fn collect_propose_host(
        &mut self,
        target: &Qwen3_5ModelRank,
        k: usize,
    ) -> Result<Vec<i32>> {
        target.device.make_current()?;
        let mut out = vec![0i32; k];
        self.graph_io
            .out_ids_dev
            .copy_to_host(&mut out, &target.stream)?;
        self.batch_hidden_rows = 1;
        self.len += k;
        Ok(out)
    }

    /// Anchor is `token[p+1]`, `target_hidden` is target final `hidden[p]`, and `len` must
    /// equal `p`. Returns `k` genuine greedy MTP drafts; cache grows by `k` provisional
    /// rows. The anchor itself is not returned. Truncate to the round's starting
    /// length and ingest accepted target pairs before the next proposal round.
    pub fn propose(
        &mut self,
        target: &Qwen3_5ModelRank,
        anchor: i32,
        target_hidden: u64,
        k: usize,
        tp2: Option<&Tp2Collectives>,
    ) -> Result<Vec<i32>> {
        if !(1..=3).contains(&k) || self.len != self.committed || self.len + k > self.capacity {
            return Err(loader_err("invalid MTP proposal state or k"));
        }
        if self.use_cuda_graph && self.supports_cuda_graph() {
            self.validate_propose_args(target, anchor, target_hidden, k, Some(tp2.is_some()))?;
            self.ensure_propose_graph(target, k, tp2)?;
            self.launch_propose_graph_async(target, anchor, target_hidden, k)?;
            return self.collect_propose_host(target, k);
        }
        let mut token = anchor;
        let mut hidden = target_hidden;
        let mut out = Vec::with_capacity(k);
        let step0_group = self.len / 4;
        for step in 0..k {
            if let Some(qsa_ws) = &mut self.qsa {
                qsa_ws.step0_page_idx = step0_group;
                qsa_ws.skip_topk = step > 0;
            }
            let res = self.forward(target, &[token], hidden, tp2);
            if let Some(qsa_ws) = &mut self.qsa {
                qsa_ws.skip_topk = false;
            }
            token = match res {
                Ok(token) => token,
                Err(error) => {
                    self.len = self.committed;
                    return Err(error);
                }
            };
            out.push(token);
            hidden = self.hidden.addr();
        }
        Ok(out)
    }

    // Shared LinearProj dispatches small-N dense projections via w34_small_n_gemm_f16.
    // MTP keeps cuBLAS GEMM for DenseF16 (matching vLLM donor M=5 prefill) while
    // dispatching MarlinW4A16 projections for quantized MTP checkpoints.
    fn project(
        proj: &LinearProj,
        ex: &W34Exec,
        t: &Qwen3_5ModelRank,
        input: u64,
        output: u64,
        rows: usize,
    ) -> Result<()> {
        match *proj {
            LinearProj::DenseF16 {
                weight_ptr,
                size_n,
                size_k,
            } => t.cublas.gemm_f16(
                input,
                weight_ptr,
                output,
                rows as i32,
                size_n as i32,
                size_k as i32,
            ),
            LinearProj::Hadamard { .. }
            | LinearProj::MarlinW4A16 { .. }
            | LinearProj::MarlinW4A16Zp { .. }
            | LinearProj::MarlinW8A16 { .. } => proj.forward(
                ex,
                &t.cublas,
                t.marlin_ws.addr(),
                t.marlin_ws.len_bytes / 4,
                input,
                output,
                rows,
            ),
        }
    }

    fn project_maybe_prerotated(
        proj: &LinearProj,
        prerotated: bool,
        ex: &W34Exec,
        t: &Qwen3_5ModelRank,
        input: u64,
        output: u64,
        rows: usize,
    ) -> Result<()> {
        if !prerotated {
            return Self::project(proj, ex, t, input, output, rows);
        }
        match *proj {
            LinearProj::DenseF16 { .. } => Self::project(proj, ex, t, input, output, rows),
            LinearProj::Hadamard { .. }
            | LinearProj::MarlinW4A16 { .. }
            | LinearProj::MarlinW4A16Zp { .. }
            | LinearProj::MarlinW8A16 { .. } => proj.forward_prerotated(
                ex,
                &t.cublas,
                t.marlin_ws.addr(),
                t.marlin_ws.len_bytes / 4,
                input,
                output,
                rows,
            ),
        }
    }

    fn forward(
        &mut self,
        target: &Qwen3_5ModelRank,
        tokens: &[i32],
        input_hidden: u64,
        tp2: Option<&Tp2Collectives>,
    ) -> Result<i32> {
        self.forward_inner(target, tokens, input_hidden, tp2, true, true, false)?
            .ok_or_else(|| loader_err("MTP forward did not sample token"))
    }

    #[allow(clippy::too_many_arguments, clippy::cognitive_complexity)]
    pub(super) fn forward_inner(
        &mut self,
        target: &Qwen3_5ModelRank,
        tokens: &[i32],
        input_hidden: u64,
        tp2: Option<&Tp2Collectives>,
        sample_last: bool,
        sync_stream: bool,
        kv_only: bool,
    ) -> Result<Option<i32>> {
        let t = target;
        if t.embed_tokens_ptr != self.target_embedding {
            return Err(loader_err("MTP target rank mismatch"));
        }
        let num_tokens = tokens.len();
        let span = num_tokens
            .checked_mul(self.hidden_size)
            .and_then(|n| n.checked_mul(2));
        if num_tokens == 0
            || num_tokens > self.max_batch_rows
            || num_tokens > self.capacity - self.len
            || tokens
                .iter()
                .any(|&id| id < 0 || id as usize >= t.cfg.vocab_size)
            || input_hidden == 0
            || !input_hidden.is_multiple_of(2)
            || span
                .and_then(|n| input_hidden.checked_add(n as u64))
                .is_none()
            || (t.tp_size > 1) != tp2.is_some()
        {
            return Err(loader_err("invalid MTP batch or input span"));
        }
        t.device.make_current()?;
        self.scratch.reset();
        let h_dim = t.cfg.hidden_size;
        let hc_count = self.hidden_size / h_dim;
        let hc_lowrank = t.cfg.hc_lowrank.max(1);
        let head_dim = t.cfg.head_dim;
        let local_q_heads = t.cfg.num_attention_heads / t.tp_size;
        let local_kv_heads = t.cfg.local_kv_heads(t.tp_size);
        let local_interm = t.cfg.intermediate_size / t.tp_size;
        let q_gate_dim = local_q_heads * head_dim * 2;
        let kv_dim = local_kv_heads * head_dim;

        let embed = self.scratch.alloc_aligned(num_tokens * h_dim * 2)?;
        let normed_buf = self.scratch.alloc_aligned(num_tokens * h_dim * 2)?;
        let fc_embed_buf = self.scratch.alloc_aligned(num_tokens * h_dim * 2)?;
        let concat = self.scratch.alloc_aligned(num_tokens * 2 * h_dim * 2)?;
        let hc_norm_buf = self
            .scratch
            .alloc_aligned(num_tokens * hc_count * h_dim * 2)?;
        let hc_residual_buf = self
            .scratch
            .alloc_aligned(num_tokens * hc_count * h_dim * 2)?;
        let hc_inj_weights_buf = self.scratch.alloc_aligned(num_tokens * hc_count * 4)?;
        let hc_scratch_a = self
            .scratch
            .alloc_aligned(num_tokens * hc_count * h_dim * 2)?;
        let hc_scratch_b = self
            .scratch
            .alloc_aligned(num_tokens * hc_count * h_dim * 2)?;
        let hc_scratch_c = self
            .scratch
            .alloc_aligned(num_tokens * (hc_lowrank + hc_count) * 2)?;
        let qkv_proj_buf = self
            .scratch
            .alloc_aligned(num_tokens * (q_gate_dim + 2 * kv_dim) * 2)?;
        let q_buf = self
            .scratch
            .alloc_aligned(num_tokens * (local_q_heads * head_dim) * 2)?;
        let k_buf = self.scratch.alloc_aligned(num_tokens * kv_dim * 2)?;
        let gate_buf = self
            .scratch
            .alloc_aligned(num_tokens * (local_q_heads * head_dim) * 2)?;
        let attn_out_buf = self
            .scratch
            .alloc_aligned(num_tokens * (local_q_heads * head_dim) * 2)?;
        let gated_attn_buf = self
            .scratch
            .alloc_aligned(num_tokens * (local_q_heads * head_dim) * 2)?;
        let block_out_buf = self.scratch.alloc_aligned(num_tokens * h_dim * 2)?;
        let mlp_out_buf = self.scratch.alloc_aligned(num_tokens * h_dim * 2)?;

        let qsa_qk_cols = (t.cfg.indexer_n_heads + t.cfg.indexer_kv_heads) * t.cfg.indexer_head_dim;
        let qsa_qk_buf = if qsa_qk_cols > 0 {
            self.scratch.alloc_aligned(num_tokens * qsa_qk_cols * 2)?
        } else {
            0
        };
        let qsa_q_buf = if qsa_qk_cols > 0 {
            self.scratch
                .alloc_aligned(num_tokens * t.cfg.indexer_n_heads * t.cfg.indexer_head_dim * 2)?
        } else {
            0
        };
        let qsa_k_buf = if qsa_qk_cols > 0 {
            self.scratch
                .alloc_aligned(num_tokens * t.cfg.indexer_kv_heads * t.cfg.indexer_head_dim * 2)?
        } else {
            0
        };

        let (
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
        ) = match &self.ffn {
            FfnBlockWeights::Dense { .. } => (
                self.scratch
                    .alloc_aligned(num_tokens * (2 * local_interm) * 2)?,
                self.scratch.alloc_aligned(num_tokens * local_interm * 2)?,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
            ),
            FfnBlockWeights::Moe(moe) => {
                let sh_gu = if moe.local_shared_interm > 0 {
                    self.scratch
                        .alloc_aligned(num_tokens * 2 * moe.local_shared_interm * 2)?
                } else {
                    0
                };
                let sh_act = if moe.local_shared_interm > 0 {
                    self.scratch
                        .alloc_aligned(num_tokens * moe.local_shared_interm * 2)?
                } else {
                    0
                };
                let sh_out = if moe.local_shared_interm > 0 {
                    self.scratch.alloc_aligned(num_tokens * h_dim * 2)?
                } else {
                    0
                };
                let sh_gate = if moe.shared_expert_gate.is_some() {
                    self.scratch.alloc_aligned(num_tokens * 2)?
                } else {
                    0
                };
                let moe_interm =
                    if matches!(moe.experts, crate::model::MoeExpertStorage::Tiered { .. }) {
                        0
                    } else {
                        self.scratch
                            .alloc_aligned(num_tokens * moe.top_k * moe.local_moe_interm * 2)?
                    };
                (
                    0,
                    0,
                    self.scratch
                        .alloc_aligned(num_tokens * moe.num_experts * 2)?,
                    self.scratch.alloc_aligned(num_tokens * moe.top_k * 4)?,
                    self.scratch.alloc_aligned(num_tokens * moe.top_k * 4)?,
                    moe_interm,
                    sh_gu,
                    sh_act,
                    sh_out,
                    sh_gate,
                )
            }
        };

        let mut meta = Vec::with_capacity(3 * num_tokens + 3);
        meta.extend_from_slice(tokens);
        meta.extend((self.len..self.len + num_tokens).map(|p| p as i32));
        meta.extend(
            (self.len..self.len + num_tokens).map(|p| self.pages[p / 16] * 16 + (p % 16) as i32),
        );
        meta.extend_from_slice(&[0i32, num_tokens as i32, (self.len + num_tokens) as i32]);
        let meta_base = self.scratch.upload_slice(&meta, &t.stream)?;
        let (logits_buf, out_id, local_id, local_val, global_val) = if sample_last {
            (
                self.scratch.alloc_aligned(t.local_vocab_size * 2)?,
                self.scratch.alloc_aligned(4)?,
                self.scratch.alloc_aligned(4)?,
                self.scratch.alloc_aligned(4)?,
                self.scratch.alloc_aligned(4)?,
            )
        } else {
            (0, 0, 0, 0, 0)
        };
        let (ws_ptr, ws_cap) = self.scratch.remaining_workspace();
        let ex = t.stream.exec(ws_ptr, ws_cap);
        let token_ids_ptr = meta_base;
        let positions_ptr = token_ids_ptr + (num_tokens * 4) as u64;
        let slot = positions_ptr + (num_tokens * 4) as u64;
        let qs = slot + (num_tokens * 4) as u64;
        let after = qs + 8;
        let attn_batch = &W34AttentionBatch {
            query_start_loc: qs as *const i32,
            context_lens_before: positions_ptr as *const i32,
            seq_lens_after: after as *const i32,
            block_tables: self.pages_dev as *const i32,
            write_slot_mapping: slot as *const i32,
            valid_token_mask: core::ptr::null(),
            valid_req_mask: core::ptr::null(),
            num_reqs: 1,
            total_tokens: num_tokens as i32,
            max_blocks_per_req: self.pages.len() as i32,
            max_q_len: num_tokens as i32,
            max_seq_len: (self.len + num_tokens) as i32,
            mask_mode: W34_ATTN_MASK_CAUSAL,
            sliding_window: 0,
            sm_scale: (head_dim as f32).powf(-0.5),
            reserved: 0,
        };

        // Predictor input fusion: concat-linear or dual-stream broadcast-add.
        check_rc(
            // SAFETY: token_ids_ptr contains num_tokens valid token IDs, embed_tokens_ptr is valid device memory.
            unsafe {
                ffi::w34_embed_lookup_f16(
                    &ex,
                    token_ids_ptr as *const i32,
                    t.embed_tokens_ptr as *const c_void,
                    embed as *mut c_void,
                    core::ptr::null(),
                    num_tokens as i32,
                    h_dim as i32,
                    t.cfg.vocab_size as i32,
                )
            },
            "MTP embed",
        )?;
        match &self.input_fusion {
            MtpInputFusion::ConcatLinear { fc } => {
                for (input, weight, output) in [
                    (embed, self.norms[0], normed_buf),
                    (input_hidden, self.norms[1], hc_norm_buf),
                ] {
                    check_rc(
                        // SAFETY: inputs and outputs contain num_tokens * h_dim FP16 elements.
                        unsafe {
                            ffi::w34_gemma_rms_norm_f16(
                                &ex,
                                input as *const c_void,
                                weight as *const c_void,
                                output as *mut c_void,
                                core::ptr::null(),
                                num_tokens as i32,
                                h_dim as i32,
                                t.cfg.rms_norm_eps,
                            )
                        },
                        "MTP pre FC norm",
                    )?;
                }
                check_rc(
                    // SAFETY: concat has room for both normalized num_tokens * h_dim inputs.
                    unsafe {
                        ffi::w34_concat_cols_2_f16(
                            &ex,
                            normed_buf as *const c_void,
                            hc_norm_buf as *const c_void,
                            concat as *mut c_void,
                            num_tokens as i32,
                            h_dim as i32,
                            h_dim as i32,
                        )
                    },
                    "MTP concat",
                )?;
                Self::project(fc, &ex, t, concat, hc_residual_buf, num_tokens)?;
            }
            MtpInputFusion::DualStreamBroadcast {
                fc_embedding,
                fc_hidden,
            } => {
                check_rc(
                    // SAFETY: embed and normed_buf hold num_tokens * h_dim FP16 elements.
                    unsafe {
                        ffi::w34_gemma_rms_norm_f16(
                            &ex,
                            embed as *const c_void,
                            self.norms[0] as *const c_void,
                            normed_buf as *mut c_void,
                            core::ptr::null(),
                            num_tokens as i32,
                            h_dim as i32,
                            t.cfg.rms_norm_eps,
                        )
                    },
                    "MTP pre_fc_norm_embedding",
                )?;
                Self::project(fc_embedding, &ex, t, normed_buf, fc_embed_buf, num_tokens)?;
                check_rc(
                    // SAFETY: input_hidden and hc_norm_buf hold num_tokens * hc_count * h_dim FP16 elements.
                    unsafe {
                        ffi::w34_gemma_rms_norm_f16(
                            &ex,
                            input_hidden as *const c_void,
                            self.norms[1] as *const c_void,
                            hc_norm_buf as *mut c_void,
                            core::ptr::null(),
                            num_tokens as i32,
                            (hc_count * h_dim) as i32,
                            t.cfg.rms_norm_eps,
                        )
                    },
                    "MTP pre_fc_norm_hidden",
                )?;
                Self::project(
                    fc_hidden,
                    &ex,
                    t,
                    hc_norm_buf,
                    hc_residual_buf,
                    num_tokens * hc_count,
                )?;
                check_rc(
                    // SAFETY: fc_embed_buf holds [num_tokens, h_dim] and hc_residual_buf holds [num_tokens * hc_count, h_dim].
                    unsafe {
                        ffi::w34_qwen4_mtp_broadcast_add_f16(
                            &ex,
                            fc_embed_buf as *const c_void,
                            hc_residual_buf as *const c_void,
                            hc_residual_buf as *mut c_void,
                            num_tokens as i32,
                            hc_count as i32,
                            h_dim as i32,
                        )
                    },
                    "w34_qwen4_mtp_broadcast_add_f16",
                )?;
            }
        }

        // Normalize / mix the residual before attention.
        let marlin_ws_ptr = t.marlin_ws.addr();
        let marlin_ws_ints = t.marlin_ws.len_bytes / 4;
        let hc_ctx = HcStepContext {
            ex: &ex,
            cublas: &t.cublas,
            marlin_ws_ptr,
            marlin_ws_ints,
            residual_buf: hc_residual_buf,
            hc_residual_buf,
            hc_inj_weights_buf,
            ple_bufs: [hc_scratch_a, hc_scratch_b, hc_scratch_c],
            num_tokens,
            hc_count,
            hc_lowrank,
            h_dim,
            eps: t.cfg.rms_norm_eps,
        };
        let fa = &self.fa;
        let is_single_stream = self.residual.is_single_stream();
        let can_use_scratch = ex.scratch_ptr != 0
            && ex.scratch_ptr.is_multiple_of(16)
            && ex.scratch_size >= num_tokens.saturating_mul(h_dim).saturating_mul(2);
        let fuse_qkv_h64 = h_dim <= PRODUCER_FUSED_H64_MAX_HIDDEN_SIZE
            && h_dim.is_multiple_of(HADAMARD_BLOCK_64)
            && fa.qkv_proj.can_fuse_h64_from_producer()
            && (fa.qsa_indexer.is_none() || can_use_scratch);
        let pre_attn_targets = if fuse_qkv_h64 {
            if fa.qsa_indexer.is_none() {
                [0, normed_buf]
            } else {
                [normed_buf, ex.scratch_ptr]
            }
        } else {
            [normed_buf, 0]
        };
        self.residual
            .pre_attn_with_targets(0, &hc_ctx, pre_attn_targets)?;

        // Full attention, with optional QSA indexing and block selection.
        Self::project_maybe_prerotated(
            &fa.qkv_proj,
            fuse_qkv_h64,
            &ex,
            t,
            if pre_attn_targets[1] != 0 {
                pre_attn_targets[1]
            } else {
                normed_buf
            },
            qkv_proj_buf,
            num_tokens,
        )?;
        let qkv_row_stride = (q_gate_dim + 2 * kv_dim) as i32;
        let q_gate_ptr = qkv_proj_buf;
        let k_raw_ptr = qkv_proj_buf + (q_gate_dim * 2) as u64;
        let v_ptr = qkv_proj_buf + ((q_gate_dim + kv_dim) * 2) as u64;
        check_rc(
            // SAFETY: q_gate_ptr, k_raw_ptr, and output buffers are valid strided device buffers.
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
                    0,
                    local_q_heads as i32,
                    local_kv_heads as i32,
                    head_dim as i32,
                    t.cfg.rotary_dim as i32,
                    t.cfg.mrope_section[0] as i32,
                    t.cfg.mrope_section[1] as i32,
                    t.cfg.mrope_section[2] as i32,
                    qkv_row_stride,
                    qkv_row_stride,
                    t.cfg.rope_theta,
                    t.cfg.rms_norm_eps,
                )
            },
            "w34_fused_qk_rmsnorm_rope_gate_strided_f16 (MTP)",
        )?;
        let kv_bufs = &self.kv.layers[fa.kv_layer_ordinal];
        let kv_layout = &kv_bufs.layout;
        check_rc(
            // SAFETY: kv_layout describes valid paged memory, k_buf and v_ptr are valid device buffers.
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
            "w34_kv_cache_write_strided (MTP)",
        )?;
        let reuse_step0_topk = self.qsa.as_ref().is_some_and(|helpers| {
            helpers.skip_topk && num_tokens == 1 && self.len / 4 == helpers.step0_page_idx
        });
        let select_k_blocks =
            (t.cfg.indexer_budget / t.cfg.indexer_compress_ratio.max(1)).clamp(1, 512) + 1;
        if let Some(qsa) = &fa.qsa_indexer {
            let qsa_ws = self
                .qsa
                .as_ref()
                .ok_or_else(|| loader_err("missing MTP QSA helpers"))?;
            qsa.index_qk_proj.forward(
                &ex,
                &t.cublas,
                marlin_ws_ptr,
                marlin_ws_ints,
                normed_buf,
                qsa_qk_buf,
                num_tokens,
            )?;
            crate::ple::gpu_ops::qsa_indexer_norm_rope_f16(
                &ex,
                qsa_qk_buf,
                qsa.q_layernorm_ptr,
                qsa.k_layernorm_ptr,
                positions_ptr,
                qsa_q_buf,
                qsa_k_buf,
                0,
                num_tokens,
                t.cfg.indexer_n_heads,
                t.cfg.indexer_kv_heads,
                t.cfg.indexer_head_dim,
                t.cfg.rotary_dim,
                t.cfg.rope_theta,
                t.cfg.rms_norm_eps,
                t.cfg.mrope_section[0],
                t.cfg.mrope_section[1],
                0,
            )?;
            let raw_k_cache_ptr = kv_bufs
                .qsa_raw_k_buf
                .as_ref()
                .ok_or_else(|| loader_err("missing MTP qsa_raw_k_buf"))?
                .addr();
            let comp_k_cache_ptr = kv_bufs
                .qsa_comp_k_buf
                .as_ref()
                .ok_or_else(|| loader_err("missing MTP qsa_comp_k_buf"))?
                .addr();
            crate::ple::gpu_ops::qsa_indexer_cache_update_f16(
                &ex,
                qsa_k_buf,
                raw_k_cache_ptr,
                comp_k_cache_ptr,
                qsa.k_layernorm_ptr,
                positions_ptr,
                slot,
                0,
                num_tokens,
                t.cfg.indexer_head_dim,
                t.cfg.rotary_dim,
                t.cfg.rope_theta,
                t.cfg.rms_norm_eps,
                t.cfg.mrope_section[0],
                t.cfg.mrope_section[1],
                0,
            )?;
            if !kv_only && !reuse_step0_topk {
                crate::ple::gpu_ops::qsa_select_blocks_f16(
                    &ex,
                    qsa_q_buf,
                    comp_k_cache_ptr,
                    attn_batch,
                    qsa_ws.qsa_sel_phys_buf.addr(),
                    qsa_ws.qsa_sel_log_buf.addr(),
                    qsa_ws.qsa_num_sel_buf.addr(),
                    qsa_ws.qsa_seq_pos_buf.addr(),
                    t.cfg.indexer_n_heads,
                    t.cfg.indexer_head_dim,
                    16,
                    select_k_blocks,
                    0,
                    0,
                )?;
            }
        }
        if kv_only {
            if sync_stream {
                t.stream.synchronize()?;
            }
            self.batch_hidden_rows = 0;
            self.len += num_tokens;
            return Ok(None);
        }
        let fuse_o_h64 = fa.o_proj.can_fuse_h64_from_producer();
        if fa.qsa_indexer.is_some() {
            let qsa_ws = self
                .qsa
                .as_ref()
                .ok_or_else(|| loader_err("missing MTP QSA helpers"))?;
            let seq_pos_ptr = if reuse_step0_topk {
                positions_ptr
            } else {
                qsa_ws.qsa_seq_pos_buf.addr()
            };
            let attn_fn = if fuse_o_h64 {
                crate::ple::gpu_ops::qsa_sparse_paged_attn_hadamard_f16
            } else {
                crate::ple::gpu_ops::qsa_sparse_paged_attn_f16
            };
            attn_fn(
                &ex,
                q_buf,
                gate_buf,
                kv_layout,
                qsa_ws.qsa_sel_phys_buf.addr(),
                qsa_ws.qsa_sel_log_buf.addr(),
                qsa_ws.qsa_num_sel_buf.addr(),
                seq_pos_ptr,
                0,
                gated_attn_buf,
                num_tokens,
                local_q_heads,
                select_k_blocks,
                attn_batch.sm_scale,
            )?;
        } else {
            check_rc(
                // SAFETY: `kv_layout`, `attn_batch`, `q_buf`, and `attn_out_buf` are valid
                // device allocations for `num_tokens`.
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
                "w34_paged_attention_f16 (MTP)",
            )?;
            let gate_fn = if fuse_o_h64 {
                ffi::w34_apply_attn_output_gate_hadamard_f16
            } else {
                ffi::w34_apply_attn_output_gate_f16
            };
            check_rc(
                // SAFETY: `attn_out_buf`, `gate_buf`, and `gated_attn_buf` are valid device
                // buffers for `num_tokens * local_q_heads * head_dim` FP16 elements.
                unsafe {
                    gate_fn(
                        &ex,
                        attn_out_buf as *const c_void,
                        gate_buf as *const c_void,
                        gated_attn_buf as *mut c_void,
                        core::ptr::null(),
                        num_tokens as i32,
                        (local_q_heads * head_dim) as i32,
                    )
                },
                "w34_apply_attn_output_gate_f16 (MTP)",
            )?;
        }
        Self::project_maybe_prerotated(
            &fa.o_proj,
            fuse_o_h64,
            &ex,
            t,
            gated_attn_buf,
            block_out_buf,
            num_tokens,
        )?;
        let pre_mlp_targets = self.ffn.pre_mlp_norm_targets(
            normed_buf,
            mlp_out_buf,
            &ex,
            num_tokens,
            h_dim,
            is_single_stream,
        );
        self.residual.post_attn_pre_mlp_with_targets(
            &hc_ctx,
            tp2,
            t.rank,
            block_out_buf,
            pre_mlp_targets,
            false,
        )?;

        // Keep the concat predictor's donor-compatible cuBLAS dense FFN path.
        if let FfnBlockWeights::Dense {
            gate_up_proj,
            down_proj,
        } = &self.ffn
            && matches!(self.input_fusion, MtpInputFusion::ConcatLinear { .. })
        {
            Self::project_maybe_prerotated(
                gate_up_proj,
                pre_mlp_targets[1] != 0,
                &ex,
                t,
                normed_buf,
                gate_up_buf,
                num_tokens,
            )?;
            let fuse_down_h64 = down_proj.can_fuse_h64_from_producer();
            let silu_fn = if fuse_down_h64 {
                ffi::w34_silu_and_mul_hadamard_f16
            } else {
                ffi::w34_silu_and_mul_f16
            };
            check_rc(
                // SAFETY: gate_up_buf and act_buf contain num_tokens rows of 2*local_interm and local_interm FP16 elements.
                unsafe {
                    silu_fn(
                        &ex,
                        gate_up_buf as *const c_void,
                        act_buf as *mut c_void,
                        core::ptr::null(),
                        num_tokens as i32,
                        local_interm as i32,
                    )
                },
                "w34_silu_and_mul_f16",
            )?;
            Self::project_maybe_prerotated(
                down_proj,
                fuse_down_h64,
                &ex,
                t,
                act_buf,
                mlp_out_buf,
                num_tokens,
            )?;
        } else {
            self.ffn.forward_with_prerotated_input(
                &ex,
                &t.cublas,
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
                pre_mlp_targets[1],
            )?;
        }
        let tiered_info = match &self.ffn {
            FfnBlockWeights::Moe(moe)
                if matches!(moe.experts, crate::model::MoeExpertStorage::Tiered { .. }) =>
            {
                Some((
                    moe.top_k,
                    if moe.shared_down_proj.is_some() {
                        moe_sh_out_buf
                    } else {
                        0
                    },
                    if moe.shared_expert_gate.is_some() {
                        moe_sh_gate_buf
                    } else {
                        0
                    },
                ))
            }
            _ => None,
        };
        if let Some((top_k, sh_out_ptr, sh_gate_ptr)) = tiered_info
            && let Some((store, cache, ws)) = &mut self.tiered_moe
        {
            let tiered_input_buf = if pre_mlp_targets[1] != 0 {
                pre_mlp_targets[1]
            } else {
                normed_buf
            };
            if cache.slots.capacity >= cache.expert_ids.len()
                || (num_tokens * top_k <= cache.slots.capacity && store.mapped_dev_ptr() != 0)
            {
                crate::moe_offload::TieredMoeEngine::execute_layer_device_lru(
                    store,
                    cache,
                    ws,
                    &t.stream,
                    tiered_input_buf,
                    moe_topk_ids_buf,
                    moe_topk_weights_buf,
                    sh_out_ptr,
                    sh_gate_ptr,
                    mlp_out_buf,
                    num_tokens,
                    top_k,
                )?;
            } else {
                let pairs = num_tokens * top_k;
                let mut h_ids = vec![0i32; pairs];
                check_rc(
                    // SAFETY: `h_ids` and `moe_topk_ids_buf` hold `pairs` i32 elements.
                    unsafe {
                        ffi::w34_memcpy_d2h_async(
                            h_ids.as_mut_ptr().cast(),
                            moe_topk_ids_buf as *const c_void,
                            pairs * 4,
                            t.stream.handle,
                        )
                    },
                    "MTP h_ids",
                )?;
                t.stream.synchronize()?;
                crate::moe_offload::TieredMoeEngine::execute_layer(
                    crate::moe_offload::MoeOffloadMode::Auto,
                    store,
                    cache,
                    ws,
                    &t.stream,
                    tiered_input_buf,
                    moe_topk_ids_buf,
                    moe_topk_weights_buf,
                    &h_ids,
                    &[],
                    sh_out_ptr,
                    sh_gate_ptr,
                    mlp_out_buf,
                    num_tokens,
                    top_k,
                )?;
            }
        }
        self.residual.post_mlp(
            &hc_ctx,
            tp2,
            t.rank,
            mlp_out_buf,
            [
                self.final_residual.single_stream_norm_ptr().unwrap_or(0),
                self.batch_hidden.addr(),
            ],
            TapExport::default(),
        )?;
        let multi_row_bytes = hc_count * h_dim * 2;
        if self.final_residual.is_hyper_connection() {
            check_rc(
                // SAFETY: batch_hidden and hc_residual_buf hold num_tokens * multi_row_bytes.
                unsafe {
                    ffi::w34_memcpy_d2d_async(
                        self.batch_hidden.addr() as *mut c_void,
                        hc_residual_buf as *const c_void,
                        num_tokens * multi_row_bytes,
                        t.stream.handle,
                    )
                },
                "MTP batch_hidden copy",
            )?;
        }
        let last_multi_ptr = self.batch_hidden.addr() + ((num_tokens - 1) * multi_row_bytes) as u64;
        check_rc(
            // SAFETY: self.hidden and last_multi_ptr hold multi_row_bytes.
            unsafe {
                ffi::w34_memcpy_d2d_async(
                    self.hidden.addr() as *mut c_void,
                    last_multi_ptr as *const c_void,
                    multi_row_bytes,
                    t.stream.handle,
                )
            },
            "MTP last multi_hidden copy",
        )?;

        // Collapse multi-stream residuals only for the LM head. Single-stream
        // post_mlp has already written the final-normalized recurrent hidden.
        let sampled_id = if sample_last {
            let mut final_hc_ctx = hc_ctx;
            final_hc_ctx.residual_buf = last_multi_ptr;
            final_hc_ctx.hc_residual_buf = last_multi_ptr;
            final_hc_ctx.num_tokens = 1;
            self.final_residual
                .finish_stack(&final_hc_ctx, &t.stream, normed_buf, None)?;
            let lm_input = if self.final_residual.is_hyper_connection() {
                normed_buf
            } else {
                last_multi_ptr
            };
            Self::project(self.lm_head(t), &ex, t, lm_input, logits_buf, 1)?;
            check_rc(
                // SAFETY: `logits_buf` holds `local_vocab_size` FP16 elements and `out_id`/
                // `local_id`/`local_val` are 4-byte device allocations.
                unsafe {
                    ffi::w34_sampler_greedy_top5_f16(
                        &ex,
                        logits_buf as *const c_void,
                        if tp2.is_some() { local_id } else { out_id } as *mut i32,
                        core::ptr::null_mut(),
                        if tp2.is_some() {
                            local_val as *mut f32
                        } else {
                            core::ptr::null_mut()
                        },
                        1,
                        t.local_vocab_size as i32,
                        t.vocab_offset as i32,
                    )
                },
                "MTP greedy",
            )?;
            if let Some(comm) = tp2 {
                comm.all_gather_top1_f32_i32(
                    &ex,
                    local_val,
                    local_id,
                    global_val,
                    out_id,
                    t.rank as i32,
                    1,
                )?;
            }
            let mut out = [0i32];
            check_rc(
                // SAFETY: `out` and `out_id` hold 4 bytes and the stream is synchronized next.
                unsafe {
                    ffi::w34_memcpy_d2h_async(
                        out.as_mut_ptr().cast(),
                        out_id as *const c_void,
                        4,
                        t.stream.handle,
                    )
                },
                "MTP greedy copy",
            )?;
            t.stream.synchronize()?;
            Some(out[0])
        } else {
            if sync_stream {
                t.stream.synchronize()?;
            }
            None
        };
        self.batch_hidden_rows = num_tokens;
        self.len += num_tokens;
        Ok(sampled_id)
    }
}
