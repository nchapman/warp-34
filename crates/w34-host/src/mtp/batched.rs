use core::ffi::c_void;
use w34_kernels::ffi::{self, W34_ATTN_MASK_CAUSAL, W34AttentionBatch};

use crate::cuda::{CudaGraph, Result, Tp2Collectives, check_rc};
use crate::model::{FfnBlockWeights, Qwen3_5ModelRank, loader_err};

use super::{BATCHED_MTP_META_I32_LEN, MAX_MTP_BATCH_SLOTS, Qwen3_5MtpRank};

const MAX_CACHED_BATCHED_PROPOSE_GRAPHS: usize = 16;

/// Per-slot shifted token span for batched multi-slot MTP `KV`-only ingest.
pub(crate) struct MtpIngestSlice<'a> {
    pub slot: usize,
    pub tokens: &'a [i32],
}

impl Qwen3_5MtpRank {
    fn validate_distinct_batch_slots(&self, slots: &[usize], extra_tokens: &[usize]) -> Result<()> {
        let slot_count = self.batch_slot_count();
        let slot_cap = self.batch_slot_capacity();
        if !self.batch_mode || !self.supports_cuda_graph() || slots.len() != extra_tokens.len() {
            return Err(loader_err("invalid batched MTP slot configuration"));
        }
        let mut seen = 0u16;
        for (&slot, &extra) in slots.iter().zip(extra_tokens) {
            if slot >= slot_count
                || slot >= 16
                || (seen & (1u16 << slot)) != 0
                || self.slot_len(slot) != self.slot_committed(slot)
                || self.slot_len(slot).saturating_add(extra) > slot_cap
            {
                return Err(loader_err("invalid or duplicate MTP batch slot state"));
            }
            seen |= 1u16 << slot;
        }
        Ok(())
    }

    /// Populates `mtp.layers.0` KV cache rows across multiple active batch slots in a single
    /// batched pass (`embed` -> `rms_norm` -> `concat_fc` -> `qkv_proj` -> `rope` -> `kv_write`)
    /// without synchronizing the target stream.
    pub(crate) fn ingest_kv_only_multi_slot_no_sync(
        &mut self,
        t: &Qwen3_5ModelRank,
        slices: &[MtpIngestSlice<'_>],
        packed_hidden: u64,
    ) -> Result<()> {
        let nonempty: Vec<&MtpIngestSlice<'_>> =
            slices.iter().filter(|s| !s.tokens.is_empty()).collect();
        if nonempty.is_empty() {
            return Ok(());
        }
        if t.embed_tokens_ptr != self.target_embedding {
            return Err(loader_err("MTP target rank mismatch"));
        }
        let slots: Vec<usize> = nonempty.iter().map(|s| s.slot).collect();
        let lens: Vec<usize> = nonempty.iter().map(|s| s.tokens.len()).collect();
        self.validate_distinct_batch_slots(&slots, &lens)?;
        let total_tokens: usize = lens.iter().sum();
        let span = total_tokens
            .checked_mul(self.hidden_size)
            .and_then(|n| n.checked_mul(2));
        if total_tokens > self.max_batch_rows
            || packed_hidden == 0
            || !packed_hidden.is_multiple_of(2)
            || span
                .and_then(|n| packed_hidden.checked_add(n as u64))
                .is_none()
            || nonempty.iter().any(|s| {
                s.tokens
                    .iter()
                    .any(|&id| id < 0 || id as usize >= t.cfg.vocab_size)
            })
        {
            return Err(loader_err("invalid multi-slot MTP ingest rows"));
        }

        t.device.make_current()?;
        self.scratch.reset();
        let h_dim = t.cfg.hidden_size;
        let head_dim = t.cfg.head_dim;
        let local_q_heads = t.cfg.num_attention_heads / t.tp_size;
        let local_kv_heads = t.cfg.local_kv_heads(t.tp_size);
        let q_gate_dim = local_q_heads * head_dim * 2;
        let kv_dim = local_kv_heads * head_dim;

        let embed = self.scratch.alloc_aligned(total_tokens * h_dim * 2)?;
        let norm_e = self.scratch.alloc_aligned(total_tokens * h_dim * 2)?;
        let norm_h = self.scratch.alloc_aligned(total_tokens * h_dim * 2)?;
        let concat = self.scratch.alloc_aligned(total_tokens * 2 * h_dim * 2)?;
        let residual_buf = self.scratch.alloc_aligned(total_tokens * h_dim * 2)?;
        let qkv_proj_buf = self
            .scratch
            .alloc_aligned(total_tokens * (q_gate_dim + 2 * kv_dim) * 2)?;
        let q_buf = self
            .scratch
            .alloc_aligned(total_tokens * (local_q_heads * head_dim) * 2)?;
        let k_buf = self.scratch.alloc_aligned(total_tokens * kv_dim * 2)?;
        let gate_buf = self
            .scratch
            .alloc_aligned(total_tokens * (local_q_heads * head_dim) * 2)?;

        let slot_count = self.batch_slot_count();
        let slot_pages = self.all_pages.len() / slot_count;
        let num_reqs = nonempty.len();
        let mut meta = Vec::with_capacity(3 * total_tokens + 3 * num_reqs + 1);
        for s in &nonempty {
            meta.extend_from_slice(s.tokens);
        }
        for s in &nonempty {
            let start = self.slot_len(s.slot);
            meta.extend((start..start + s.tokens.len()).map(|p| p as i32));
        }
        for s in &nonempty {
            let start = self.slot_len(s.slot);
            let page_base = s.slot * slot_pages;
            meta.extend(
                (start..start + s.tokens.len())
                    .map(|p| self.all_pages[page_base + p / 16] * 16 + (p % 16) as i32),
            );
        }
        let mut cursor = 0i32;
        meta.push(0);
        for s in &nonempty {
            cursor += s.tokens.len() as i32;
            meta.push(cursor);
        }
        for s in &nonempty {
            meta.push(self.slot_len(s.slot) as i32);
        }
        let mut max_after = 0usize;
        for s in &nonempty {
            let after = self.slot_len(s.slot) + s.tokens.len();
            max_after = max_after.max(after);
            meta.push(after as i32);
        }
        let meta_base = self.scratch.upload_slice(&meta, &t.stream)?;
        let tok_ptr = meta_base;
        let pos_ptr = tok_ptr + (total_tokens * 4) as u64;
        let ws_ptr = pos_ptr + (total_tokens * 4) as u64;
        let qs_ptr = ws_ptr + (total_tokens * 4) as u64;
        let cb_ptr = qs_ptr + ((num_reqs + 1) * 4) as u64;
        let sa_ptr = cb_ptr + (num_reqs * 4) as u64;

        let (rem_ptr, rem_cap) = self.scratch.remaining_workspace();
        let ex = t.stream.exec(rem_ptr, rem_cap);
        check_rc(
            // SAFETY: tok_ptr holds total_tokens valid token IDs; embed has total_tokens * h_dim half capacity.
            unsafe {
                ffi::w34_embed_lookup_f16(
                    &ex,
                    tok_ptr as *const i32,
                    t.embed_tokens_ptr as *const c_void,
                    embed as *mut c_void,
                    core::ptr::null(),
                    total_tokens as i32,
                    h_dim as i32,
                    t.cfg.vocab_size as i32,
                )
            },
            "MTP batched ingest embed",
        )?;
        for (input, weight, output) in [
            (embed, self.norms[0], norm_e),
            (packed_hidden, self.norms[1], norm_h),
        ] {
            check_rc(
                // SAFETY: input and output hold total_tokens * h_dim FP16 elements on t.stream.
                unsafe {
                    ffi::w34_gemma_rms_norm_f16(
                        &ex,
                        input as *const c_void,
                        weight as *const c_void,
                        output as *mut c_void,
                        core::ptr::null(),
                        total_tokens as i32,
                        h_dim as i32,
                        t.cfg.rms_norm_eps,
                    )
                },
                "MTP batched ingest pre FC norm",
            )?;
        }
        check_rc(
            // SAFETY: norm_e and norm_h hold total_tokens * h_dim elements; concat holds 2x that.
            unsafe {
                ffi::w34_concat_cols_2_f16(
                    &ex,
                    norm_e as *const c_void,
                    norm_h as *const c_void,
                    concat as *mut c_void,
                    total_tokens as i32,
                    h_dim as i32,
                    h_dim as i32,
                )
            },
            "MTP batched ingest concat",
        )?;
        Self::project(
            self.concat_fc()?,
            &ex,
            t,
            concat,
            residual_buf,
            total_tokens,
        )?;
        let fa = &self.fa;
        let [qkv_norm_out, qkv_norm_had64] =
            fa.pre_attn_norm_targets(norm_e, &ex, total_tokens, h_dim);
        let fuse_qkv_h64 = qkv_norm_had64 != 0;
        check_rc(
            // SAFETY: residual_buf and norm_e hold total_tokens * h_dim FP16 elements.
            unsafe {
                ffi::w34_gemma_rms_norm_hadamard_f16(
                    &ex,
                    residual_buf as *const c_void,
                    self.residual
                        .single_stream_input_norm_ptr()
                        .ok_or_else(|| loader_err("MTP requires single-stream residual"))?
                        as *const c_void,
                    qkv_norm_out as *mut c_void,
                    qkv_norm_had64 as *mut c_void,
                    core::ptr::null(),
                    total_tokens as i32,
                    h_dim as i32,
                    t.cfg.rms_norm_eps,
                )
            },
            "MTP batched ingest input norm",
        )?;
        Self::project_maybe_prerotated(
            &fa.qkv_proj,
            fuse_qkv_h64,
            &ex,
            t,
            norm_e,
            qkv_proj_buf,
            total_tokens,
        )?;
        let qkv_row_stride = (q_gate_dim + 2 * kv_dim) as i32;
        let q_gate_ptr = qkv_proj_buf;
        let k_raw_ptr = qkv_proj_buf + (q_gate_dim * 2) as u64;
        let v_ptr = qkv_proj_buf + ((q_gate_dim + kv_dim) * 2) as u64;
        check_rc(
            // SAFETY: qkv_proj_buf and output buffers span total_tokens strided rows on t.stream.
            unsafe {
                ffi::w34_fused_qk_rmsnorm_rope_gate_strided_f16(
                    &ex,
                    q_gate_ptr as *const c_void,
                    k_raw_ptr as *const c_void,
                    fa.q_norm_ptr as *const c_void,
                    fa.k_norm_ptr as *const c_void,
                    pos_ptr as *const i32,
                    core::ptr::null(),
                    q_buf as *mut c_void,
                    k_buf as *mut c_void,
                    gate_buf as *mut c_void,
                    total_tokens as i32,
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
            "MTP batched ingest rope",
        )?;
        let attn_batch = W34AttentionBatch {
            query_start_loc: qs_ptr as *const i32,
            context_lens_before: cb_ptr as *const i32,
            seq_lens_after: sa_ptr as *const i32,
            block_tables: core::ptr::null(),
            write_slot_mapping: ws_ptr as *const i32,
            valid_token_mask: core::ptr::null(),
            valid_req_mask: core::ptr::null(),
            num_reqs: num_reqs as i32,
            total_tokens: total_tokens as i32,
            max_blocks_per_req: slot_pages as i32,
            max_q_len: total_tokens as i32,
            max_seq_len: max_after as i32,
            mask_mode: W34_ATTN_MASK_CAUSAL,
            sliding_window: 0,
            sm_scale: (head_dim as f32).powf(-0.5),
            reserved: 0,
        };
        let kv_layout = &self.kv.layers[fa.kv_layer_ordinal].layout;
        check_rc(
            // SAFETY: kv_layout and ws_ptr describe valid disjoint slot pages for total_tokens rows.
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
            "MTP batched ingest kv write",
        )?;
        for s in &nonempty {
            let new_len = self.slot_len(s.slot) + s.tokens.len();
            self.set_slot_state(s.slot, new_len, new_len);
        }
        Ok(())
    }

    pub fn captured_batched_propose_graph_shapes(&self) -> Vec<(usize, usize, bool)> {
        let mut shapes: Vec<(usize, usize, bool)> = self
            .graph_io
            .batched
            .graphs
            .iter()
            .map(|(k, _)| *k)
            .collect();
        shapes.sort_unstable();
        shapes.dedup();
        shapes
    }

    fn enqueue_batched_propose_steps(
        &mut self,
        t: &Qwen3_5ModelRank,
        b_reqs: usize,
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
        let slot_pages = self.all_pages.len() / self.batch_slot_count();
        let slot_cap = slot_pages * 16;

        let arena = &mut self.graph_io.propose_arena;
        let residual_buf = arena.alloc_aligned(b_reqs * h_dim * 2)?;
        let normed_buf = arena.alloc_aligned(b_reqs * h_dim * 2)?;
        let norm_h_buf = arena.alloc_aligned(b_reqs * h_dim * 2)?;
        let embed = arena.alloc_aligned(b_reqs * h_dim * 2)?;
        let concat = arena.alloc_aligned(b_reqs * (2 * h_dim) * 2)?;
        let qkv_proj_buf = arena.alloc_aligned(b_reqs * (q_gate_dim + 2 * kv_dim) * 2)?;
        let q_buf = arena.alloc_aligned(b_reqs * (local_q_heads * head_dim) * 2)?;
        let k_buf = arena.alloc_aligned(b_reqs * kv_dim * 2)?;
        let gate_buf = arena.alloc_aligned(b_reqs * (local_q_heads * head_dim) * 2)?;
        let attn_out_buf = arena.alloc_aligned(b_reqs * (local_q_heads * head_dim) * 2)?;
        let gated_attn_buf = arena.alloc_aligned(b_reqs * (local_q_heads * head_dim) * 2)?;
        let block_out_buf = arena.alloc_aligned(b_reqs * h_dim * 2)?;
        let mlp_out_buf = arena.alloc_aligned(b_reqs * h_dim * 2)?;
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
        ) = Self::alloc_batched_ffn_scratch(arena, &self.ffn, b_reqs, h_dim, local_interm)?;
        let logits_buf = arena.alloc_aligned(b_reqs * t.local_vocab_size * 2)?;
        let local_id = arena.alloc_aligned(b_reqs * 4)?;
        let local_val = arena.alloc_aligned(b_reqs * 4)?;
        let global_val = arena.alloc_aligned(b_reqs * 4)?;
        let (ws_ptr, ws_cap) = arena.remaining_workspace();
        let ex = t.stream.exec(ws_ptr, ws_cap);

        let meta_base = self.graph_io.batched.meta_dev.addr();
        let token_ids_ptr = meta_base;
        let qs_ptr = meta_base + (10 * MAX_MTP_BATCH_SLOTS * 4) as u64;
        let hidden_ptr = self.graph_io.batched.hidden_dev.addr();
        let bt_ptr = self.graph_io.batched.block_tables_dev.addr();
        let qkv_row_stride = (q_gate_dim + 2 * kv_dim) as i32;
        let q_gate_ptr = qkv_proj_buf;
        let k_raw_ptr = qkv_proj_buf + (q_gate_dim * 2) as u64;
        let v_ptr = qkv_proj_buf + ((q_gate_dim + kv_dim) * 2) as u64;
        let kv_layout = &self.kv.layers[self.fa.kv_layer_ordinal].layout;

        for step in 0..k {
            let positions_ptr = meta_base + ((1 + 3 * step) * MAX_MTP_BATCH_SLOTS * 4) as u64;
            let slot_ptr = meta_base + ((2 + 3 * step) * MAX_MTP_BATCH_SLOTS * 4) as u64;
            let after_ptr = meta_base + ((3 + 3 * step) * MAX_MTP_BATCH_SLOTS * 4) as u64;
            let out_id =
                self.graph_io.batched.out_ids_dev.addr() + (step * MAX_MTP_BATCH_SLOTS * 4) as u64;
            let attn_batch = W34AttentionBatch {
                query_start_loc: qs_ptr as *const i32,
                context_lens_before: positions_ptr as *const i32,
                seq_lens_after: after_ptr as *const i32,
                block_tables: bt_ptr as *const i32,
                write_slot_mapping: slot_ptr as *const i32,
                valid_token_mask: core::ptr::null(),
                valid_req_mask: core::ptr::null(),
                num_reqs: b_reqs as i32,
                total_tokens: b_reqs as i32,
                max_blocks_per_req: slot_pages as i32,
                max_q_len: 1,
                max_seq_len: if long_ctx {
                    slot_cap as i32
                } else {
                    slot_cap.min(1024) as i32
                },
                mask_mode: W34_ATTN_MASK_CAUSAL,
                sliding_window: 0,
                sm_scale: (head_dim as f32).powf(-0.5),
                reserved: 0,
            };
            self.enqueue_batched_propose_one_step(
                t,
                &ex,
                b_reqs,
                step + 1 == k,
                step + 1 < k,
                tp2,
                token_ids_ptr,
                positions_ptr,
                hidden_ptr,
                out_id,
                &attn_batch,
                kv_layout,
                qkv_row_stride,
                q_gate_ptr,
                k_raw_ptr,
                v_ptr,
                residual_buf,
                normed_buf,
                norm_h_buf,
                embed,
                concat,
                qkv_proj_buf,
                q_buf,
                k_buf,
                gate_buf,
                attn_out_buf,
                gated_attn_buf,
                block_out_buf,
                mlp_out_buf,
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
                logits_buf,
                local_id,
                local_val,
                global_val,
            )?;
        }
        Ok(())
    }

    #[allow(clippy::type_complexity)]
    fn alloc_batched_ffn_scratch(
        arena: &mut crate::cuda::DeviceArena,
        ffn: &FfnBlockWeights,
        b_reqs: usize,
        h_dim: usize,
        local_interm: usize,
    ) -> Result<(u64, u64, u64, u64, u64, u64, u64, u64, u64, u64)> {
        match ffn {
            FfnBlockWeights::Dense { .. } => Ok((
                arena.alloc_aligned(b_reqs * (2 * local_interm) * 2)?,
                arena.alloc_aligned(b_reqs * local_interm * 2)?,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
            )),
            FfnBlockWeights::Moe(moe) => {
                let sh_gu = if moe.local_shared_interm > 0 {
                    arena.alloc_aligned(b_reqs * 2 * moe.local_shared_interm * 2)?
                } else {
                    0
                };
                let sh_act = if moe.local_shared_interm > 0 {
                    arena.alloc_aligned(b_reqs * moe.local_shared_interm * 2)?
                } else {
                    0
                };
                let sh_out = if moe.local_shared_interm > 0 {
                    arena.alloc_aligned(b_reqs * h_dim * 2)?
                } else {
                    0
                };
                let sh_gate = if moe.shared_expert_gate.is_some() {
                    arena.alloc_aligned(b_reqs * 2)?
                } else {
                    0
                };
                let moe_interm =
                    if matches!(moe.experts, crate::model::MoeExpertStorage::Tiered { .. }) {
                        0
                    } else {
                        arena.alloc_aligned(b_reqs * moe.top_k * moe.local_moe_interm * 2)?
                    };
                Ok((
                    0,
                    0,
                    arena.alloc_aligned(b_reqs * moe.num_experts * 2)?,
                    arena.alloc_aligned(b_reqs * moe.top_k * 4)?,
                    arena.alloc_aligned(b_reqs * moe.top_k * 4)?,
                    moe_interm,
                    sh_gu,
                    sh_act,
                    sh_out,
                    sh_gate,
                ))
            }
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn enqueue_batched_propose_one_step(
        &self,
        t: &Qwen3_5ModelRank,
        ex: &ffi::W34Exec,
        b_reqs: usize,
        is_last_step: bool,
        has_next_step: bool,
        tp2: Option<&Tp2Collectives>,
        token_ids_ptr: u64,
        positions_ptr: u64,
        hidden_ptr: u64,
        out_id: u64,
        attn_batch: &W34AttentionBatch,
        kv_layout: &ffi::W34KvLayerLayout,
        qkv_row_stride: i32,
        q_gate_ptr: u64,
        k_raw_ptr: u64,
        v_ptr: u64,
        residual_buf: u64,
        normed_buf: u64,
        norm_h_buf: u64,
        embed: u64,
        concat: u64,
        qkv_proj_buf: u64,
        q_buf: u64,
        k_buf: u64,
        gate_buf: u64,
        attn_out_buf: u64,
        gated_attn_buf: u64,
        block_out_buf: u64,
        mlp_out_buf: u64,
        gate_up_buf: u64,
        act_buf: u64,
        moe_router_logits_buf: u64,
        moe_topk_weights_buf: u64,
        moe_topk_ids_buf: u64,
        moe_interm_buf: u64,
        moe_sh_gu_buf: u64,
        moe_sh_act_buf: u64,
        moe_sh_out_buf: u64,
        moe_sh_gate_buf: u64,
        logits_buf: u64,
        local_id: u64,
        local_val: u64,
        global_val: u64,
    ) -> Result<()> {
        let h_dim = t.cfg.hidden_size;
        let head_dim = t.cfg.head_dim;
        let local_q_heads = t.cfg.num_attention_heads / t.tp_size;
        let local_kv_heads = t.cfg.local_kv_heads(t.tp_size);
        let local_interm = t.cfg.intermediate_size / t.tp_size;
        let kv_dim = local_kv_heads * head_dim;

        check_rc(
            // SAFETY: token_ids_ptr holds b_reqs valid IDs; embed has b_reqs * h_dim half elements.
            unsafe {
                ffi::w34_embed_lookup_f16(
                    ex,
                    token_ids_ptr as *const i32,
                    t.embed_tokens_ptr as *const c_void,
                    embed as *mut c_void,
                    core::ptr::null(),
                    b_reqs as i32,
                    h_dim as i32,
                    t.cfg.vocab_size as i32,
                )
            },
            "MTP batched embed",
        )?;
        for (input, weight, output) in [
            (embed, self.norms[0], normed_buf),
            (hidden_ptr, self.norms[1], norm_h_buf),
        ] {
            check_rc(
                // SAFETY: input and output hold b_reqs * h_dim FP16 elements.
                unsafe {
                    ffi::w34_gemma_rms_norm_f16(
                        ex,
                        input as *const c_void,
                        weight as *const c_void,
                        output as *mut c_void,
                        core::ptr::null(),
                        b_reqs as i32,
                        h_dim as i32,
                        t.cfg.rms_norm_eps,
                    )
                },
                "MTP batched pre FC norm",
            )?;
        }
        check_rc(
            // SAFETY: normed_buf and norm_h_buf hold b_reqs * h_dim elements; concat holds 2x that.
            unsafe {
                ffi::w34_concat_cols_2_f16(
                    ex,
                    normed_buf as *const c_void,
                    norm_h_buf as *const c_void,
                    concat as *mut c_void,
                    b_reqs as i32,
                    h_dim as i32,
                    h_dim as i32,
                )
            },
            "MTP batched concat",
        )?;
        Self::project(self.concat_fc()?, ex, t, concat, residual_buf, b_reqs)?;

        let fa = &self.fa;
        let [qkv_norm_out, qkv_norm_had64] =
            fa.pre_attn_norm_targets(normed_buf, ex, b_reqs, h_dim);
        let fuse_qkv_h64 = qkv_norm_had64 != 0;
        check_rc(
            // SAFETY: residual_buf and normed_buf hold b_reqs * h_dim half elements.
            unsafe {
                ffi::w34_gemma_rms_norm_hadamard_f16(
                    ex,
                    residual_buf as *const c_void,
                    self.residual
                        .single_stream_input_norm_ptr()
                        .ok_or_else(|| loader_err("MTP graph requires single-stream residual"))?
                        as *const c_void,
                    qkv_norm_out as *mut c_void,
                    qkv_norm_had64 as *mut c_void,
                    core::ptr::null(),
                    b_reqs as i32,
                    h_dim as i32,
                    t.cfg.rms_norm_eps,
                )
            },
            "MTP batched input norm",
        )?;
        Self::project_maybe_prerotated(
            &fa.qkv_proj,
            fuse_qkv_h64,
            ex,
            t,
            normed_buf,
            qkv_proj_buf,
            b_reqs,
        )?;
        check_rc(
            // SAFETY: qkv_proj_buf and output buffers hold b_reqs strided rows on t.stream.
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
                    b_reqs as i32,
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
            "MTP batched rope",
        )?;
        check_rc(
            // SAFETY: kv_layout and attn_batch describe valid paged memory for b_reqs tokens.
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
            "MTP batched kv write",
        )?;
        check_rc(
            // SAFETY: kv_layout and attn_batch describe valid b_reqs-request paged attention.
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
            "MTP batched paged attention",
        )?;
        let fuse_o_h64 = fa.o_proj.can_fuse_h64_from_producer();
        let gate_fn = if fuse_o_h64 {
            ffi::w34_apply_attn_output_gate_hadamard_f16
        } else {
            ffi::w34_apply_attn_output_gate_f16
        };
        check_rc(
            // SAFETY: attn_out_buf, gate_buf, and gated_attn_buf hold b_reqs * local_q_heads * head_dim halves.
            unsafe {
                gate_fn(
                    ex,
                    attn_out_buf as *const c_void,
                    gate_buf as *const c_void,
                    gated_attn_buf as *mut c_void,
                    core::ptr::null(),
                    b_reqs as i32,
                    (local_q_heads * head_dim) as i32,
                )
            },
            "MTP batched attn gate",
        )?;
        Self::project_maybe_prerotated(
            &fa.o_proj,
            fuse_o_h64,
            ex,
            t,
            gated_attn_buf,
            block_out_buf,
            b_reqs,
        )?;
        self.enqueue_batched_post_attn_and_ffn(
            t,
            ex,
            b_reqs,
            is_last_step,
            has_next_step,
            tp2,
            token_ids_ptr,
            hidden_ptr,
            out_id,
            h_dim,
            local_interm,
            residual_buf,
            normed_buf,
            block_out_buf,
            mlp_out_buf,
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
            logits_buf,
            local_id,
            local_val,
            global_val,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn enqueue_batched_post_attn_and_ffn(
        &self,
        t: &Qwen3_5ModelRank,
        ex: &ffi::W34Exec,
        b_reqs: usize,
        is_last_step: bool,
        has_next_step: bool,
        tp2: Option<&Tp2Collectives>,
        token_ids_ptr: u64,
        hidden_ptr: u64,
        out_id: u64,
        h_dim: usize,
        local_interm: usize,
        residual_buf: u64,
        normed_buf: u64,
        block_out_buf: u64,
        mlp_out_buf: u64,
        gate_up_buf: u64,
        act_buf: u64,
        moe_router_logits_buf: u64,
        moe_topk_weights_buf: u64,
        moe_topk_ids_buf: u64,
        moe_interm_buf: u64,
        moe_sh_gu_buf: u64,
        moe_sh_act_buf: u64,
        moe_sh_out_buf: u64,
        moe_sh_gate_buf: u64,
        logits_buf: u64,
        local_id: u64,
        local_val: u64,
        global_val: u64,
    ) -> Result<()> {
        let post_ln_ptr = self
            .residual
            .single_stream_post_attn_norm_ptr()
            .ok_or_else(|| loader_err("MTP graph requires single-stream residual"))?;
        let [mlp_norm_out, mlp_norm_had64] =
            self.ffn
                .pre_mlp_norm_targets(normed_buf, mlp_out_buf, ex, b_reqs, h_dim);
        if let Some(comm) = tp2 {
            comm.all_reduce_residual_gemma_rms_norm_hadamard_f16(
                ex,
                block_out_buf,
                0,
                residual_buf,
                post_ln_ptr,
                mlp_norm_out,
                mlp_norm_had64,
                0,
                t.rank as i32,
                b_reqs as i32,
                h_dim as i32,
                t.cfg.rms_norm_eps,
                0,
                0,
            )?;
        } else {
            check_rc(
                // SAFETY: block_out_buf, residual_buf, and mlp_norm_out hold b_reqs * h_dim halves.
                unsafe {
                    ffi::w34_fused_add_gemma_rms_norm_tap_hadamard_f16(
                        ex,
                        block_out_buf as *const c_void,
                        residual_buf as *mut c_void,
                        post_ln_ptr as *const c_void,
                        mlp_norm_out as *mut c_void,
                        mlp_norm_had64 as *mut c_void,
                        core::ptr::null_mut(),
                        core::ptr::null(),
                        b_reqs as i32,
                        h_dim as i32,
                        t.cfg.rms_norm_eps,
                        0,
                        0,
                    )
                },
                "MTP batched post_attn norm",
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
                    ex,
                    t,
                    normed_buf,
                    gate_up_buf,
                    b_reqs,
                )?;
                let fuse_down_h64 = down_proj.can_fuse_h64_from_producer();
                let silu_fn = if fuse_down_h64 {
                    ffi::w34_silu_and_mul_hadamard_f16
                } else {
                    ffi::w34_silu_and_mul_f16
                };
                check_rc(
                    // SAFETY: gate_up_buf and act_buf hold b_reqs rows of local_interm halves.
                    unsafe {
                        silu_fn(
                            ex,
                            gate_up_buf as *const c_void,
                            act_buf as *mut c_void,
                            core::ptr::null(),
                            b_reqs as i32,
                            local_interm as i32,
                        )
                    },
                    "MTP batched silu",
                )?;
                Self::project_maybe_prerotated(
                    down_proj,
                    fuse_down_h64,
                    ex,
                    t,
                    act_buf,
                    mlp_out_buf,
                    b_reqs,
                )?;
            }
            FfnBlockWeights::Moe(_) => {
                self.ffn.forward_with_prerotated_input(
                    ex,
                    &t.cublas,
                    t.marlin_ws.addr(),
                    t.marlin_ws.len_bytes / 4,
                    normed_buf,
                    mlp_out_buf,
                    b_reqs,
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
        let final_ln_ptr = self
            .final_residual
            .single_stream_norm_ptr()
            .ok_or_else(|| loader_err("MTP graph requires single-stream final norm"))?;
        if let Some(comm) = tp2 {
            comm.all_reduce_residual_gemma_rms_norm_f16(
                ex,
                mlp_out_buf,
                0,
                residual_buf,
                final_ln_ptr,
                hidden_ptr,
                0,
                t.rank as i32,
                b_reqs as i32,
                h_dim as i32,
                t.cfg.rms_norm_eps,
                0,
                0,
            )?;
        } else {
            check_rc(
                // SAFETY: mlp_out_buf, residual_buf, and hidden_ptr hold b_reqs * h_dim halves.
                unsafe {
                    ffi::w34_fused_add_gemma_rms_norm_f16(
                        ex,
                        mlp_out_buf as *const c_void,
                        residual_buf as *mut c_void,
                        final_ln_ptr as *const c_void,
                        hidden_ptr as *mut c_void,
                        core::ptr::null(),
                        b_reqs as i32,
                        h_dim as i32,
                        t.cfg.rms_norm_eps,
                    )
                },
                "MTP batched final norm",
            )?;
        }
        if is_last_step {
            check_rc(
                // SAFETY: batch_hidden and hidden_ptr hold >= b_reqs * h_dim * 2 bytes on t.stream.
                unsafe {
                    ffi::w34_memcpy_d2d_async(
                        self.batch_hidden.addr() as *mut c_void,
                        hidden_ptr as *const c_void,
                        b_reqs * h_dim * 2,
                        t.stream.handle,
                    )
                },
                "MTP batched hidden tail",
            )?;
        }
        Self::project(self.lm_head(t)?, ex, t, hidden_ptr, logits_buf, b_reqs)?;
        check_rc(
            // SAFETY: logits_buf holds b_reqs * local_vocab_size halves; output holds b_reqs i32/f32s.
            unsafe {
                ffi::w34_sampler_greedy_top5_f16(
                    ex,
                    logits_buf as *const c_void,
                    if tp2.is_some() { local_id } else { out_id } as *mut i32,
                    core::ptr::null_mut(),
                    if tp2.is_some() {
                        local_val as *mut f32
                    } else {
                        core::ptr::null_mut()
                    },
                    b_reqs as i32,
                    t.local_vocab_size as i32,
                    t.vocab_offset as i32,
                )
            },
            "MTP batched greedy",
        )?;
        if let Some(comm) = tp2 {
            comm.all_gather_top1_f32_i32(
                ex,
                local_val,
                local_id,
                global_val,
                out_id,
                t.rank as i32,
                b_reqs as i32,
            )?;
        }
        if has_next_step {
            check_rc(
                // SAFETY: token_ids_ptr and out_id hold >= b_reqs * 4 bytes on t.stream.
                unsafe {
                    ffi::w34_memcpy_d2d_async(
                        token_ids_ptr as *mut c_void,
                        out_id as *const c_void,
                        b_reqs * 4,
                        t.stream.handle,
                    )
                },
                "MTP batched recurrent draft token",
            )?;
        }
        Ok(())
    }

    pub(crate) fn ensure_batched_propose_graph(
        &mut self,
        target: &Qwen3_5ModelRank,
        b_reqs: usize,
        k: usize,
        long_ctx: bool,
        tp2: Option<&Tp2Collectives>,
    ) -> Result<()> {
        let key = (b_reqs, k, long_ctx);
        if let Some(pos) = self
            .graph_io
            .batched
            .graphs
            .iter()
            .position(|(cap_key, _)| *cap_key == key)
        {
            if pos + 1 != self.graph_io.batched.graphs.len() {
                let entry = self.graph_io.batched.graphs.remove(pos);
                self.graph_io.batched.graphs.push(entry);
            }
            return Ok(());
        }
        target.device.make_current()?;
        target.stream.synchronize()?;
        if self.graph_io.batched.graphs.len() >= MAX_CACHED_BATCHED_PROPOSE_GRAPHS {
            self.graph_io.batched.graphs.remove(0);
        }
        let graph = CudaGraph::capture(&target.stream, || {
            self.enqueue_batched_propose_steps(target, b_reqs, k, long_ctx, tp2)
        })?;
        self.graph_io.batched.graphs.push((key, graph));
        Ok(())
    }

    pub(crate) fn prepare_batched_propose_graph_inputs_async(
        &mut self,
        target: &Qwen3_5ModelRank,
        slots: &[usize],
        anchors: &[i32],
        target_hiddens: &[u64],
        k: usize,
    ) -> Result<bool> {
        let b_reqs = slots.len();
        if !(2..=MAX_MTP_BATCH_SLOTS).contains(&b_reqs)
            || !(1..=3).contains(&k)
            || anchors.len() != b_reqs
            || target_hiddens.len() != b_reqs
            || target.embed_tokens_ptr != self.target_embedding
        {
            return Err(loader_err("invalid batched MTP propose geometry"));
        }
        let extras = vec![k; b_reqs];
        self.validate_distinct_batch_slots(slots, &extras)?;
        let row_bytes = self.hidden_size * 2;
        if anchors
            .iter()
            .any(|&a| a < 0 || a as usize >= target.cfg.vocab_size)
            || target_hiddens.iter().any(|&h| {
                h == 0 || !h.is_multiple_of(2) || h.checked_add(row_bytes as u64).is_none()
            })
        {
            return Err(loader_err("invalid batched MTP anchor or hidden pointer"));
        }

        target.device.make_current()?;
        let slot_pages = self.all_pages.len() / self.batch_slot_count();
        if self.graph_io.batched.cached_slots.as_slice() != slots {
            let mut bt_host = Vec::with_capacity(b_reqs * slot_pages);
            for &slot in slots {
                bt_host
                    .extend_from_slice(&self.all_pages[slot * slot_pages..(slot + 1) * slot_pages]);
            }
            self.graph_io
                .batched
                .block_tables_dev
                .copy_from_host(&bt_host, &target.stream)?;
            self.graph_io.batched.cached_slots.clear();
            self.graph_io.batched.cached_slots.extend_from_slice(slots);
        }

        let mut meta = [0i32; BATCHED_MTP_META_I32_LEN];
        meta[..b_reqs].copy_from_slice(anchors);
        for step in 0..3 {
            let s_clamped = step.min(k - 1);
            let pos_base = (1 + 3 * step) * MAX_MTP_BATCH_SLOTS;
            let ws_base = (2 + 3 * step) * MAX_MTP_BATCH_SLOTS;
            let after_base = (3 + 3 * step) * MAX_MTP_BATCH_SLOTS;
            for (b, &slot) in slots.iter().enumerate() {
                let p = self.slot_len(slot) + s_clamped;
                meta[pos_base + b] = p as i32;
                meta[ws_base + b] =
                    self.all_pages[slot * slot_pages + p / 16] * 16 + (p % 16) as i32;
                meta[after_base + b] = (p + 1) as i32;
            }
        }
        let qs_base = 10 * MAX_MTP_BATCH_SLOTS;
        for i in 0..=MAX_MTP_BATCH_SLOTS {
            meta[qs_base + i] = i as i32;
        }
        self.graph_io
            .batched
            .meta_dev
            .copy_from_host(&meta, &target.stream)?;

        let dst_base = self.graph_io.batched.hidden_dev.addr();
        for (b, &src_ptr) in target_hiddens.iter().enumerate() {
            check_rc(
                // SAFETY: hidden_dev holds MAX_MTP_BATCH_SLOTS rows; src_ptr holds 1 row on target.stream.
                unsafe {
                    ffi::w34_memcpy_d2d_async(
                        (dst_base + (b * row_bytes) as u64) as *mut c_void,
                        src_ptr as *const c_void,
                        row_bytes,
                        target.stream.handle,
                    )
                },
                "MTP batched seed hidden",
            )?;
        }
        let long_ctx = slots.iter().any(|&slot| self.slot_len(slot) + k > 1024);
        Ok(long_ctx)
    }

    pub(crate) fn replay_batched_propose_graph_async(
        &self,
        target: &Qwen3_5ModelRank,
        b_reqs: usize,
        k: usize,
        long_ctx: bool,
    ) -> Result<()> {
        target.device.make_current()?;
        let key = (b_reqs, k, long_ctx);
        let (_, graph) = self
            .graph_io
            .batched
            .graphs
            .iter()
            .find(|(cap_key, _)| *cap_key == key)
            .ok_or_else(|| loader_err("batched MTP propose graph not captured"))?;
        graph.launch(&target.stream)
    }

    pub(crate) fn collect_batched_propose_host(
        &mut self,
        target: &Qwen3_5ModelRank,
        slots: &[usize],
        k_budgets: &[usize],
        k_max: usize,
    ) -> Result<Vec<Vec<i32>>> {
        target.device.make_current()?;
        let mut raw = [0i32; 3 * MAX_MTP_BATCH_SLOTS];
        self.graph_io
            .batched
            .out_ids_dev
            .copy_to_host(&mut raw[..k_max * MAX_MTP_BATCH_SLOTS], &target.stream)?;
        self.batch_hidden_rows = slots.len();
        let mut out = Vec::with_capacity(slots.len());
        for (b, (&slot, &kb)) in slots.iter().zip(k_budgets).enumerate() {
            let committed = self.slot_committed(slot);
            let new_len = self.slot_len(slot) + k_max;
            self.set_slot_state(slot, new_len, committed);
            let n = kb.min(k_max);
            out.push(
                (0..n)
                    .map(|step| raw[step * MAX_MTP_BATCH_SLOTS + b])
                    .collect(),
            );
        }
        Ok(out)
    }
}
