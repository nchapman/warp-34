use super::*;

#[cfg(feature = "cuda")]
pub struct QsaIndexerWeights {
    pub index_qk_proj: LinearProj,
    pub q_layernorm_ptr: u64,
    pub k_layernorm_ptr: u64,
}

#[cfg(feature = "cuda")]
pub struct FullAttentionLayerWeights {
    pub layer_id: usize,
    pub kv_layer_ordinal: usize,
    pub qkv_proj: LinearProj,
    pub o_proj: LinearProj,
    pub q_norm_ptr: u64,
    pub k_norm_ptr: u64,
    pub qsa_indexer: Option<QsaIndexerWeights>,
}

#[cfg(feature = "cuda")]
impl FullAttentionLayerWeights {
    /// Runs QSA pre-indexer norm+RoPE, updates raw/compressed indexer key caches,
    /// optionally selects top-k sparse blocks, and executes sparse paged GQA + sigmoid output gate.
    #[allow(clippy::too_many_arguments)]
    pub fn forward_qsa_attention(
        &self,
        qsa_idx: &QsaIndexerWeights,
        ex: &W34Exec,
        cublas: &CublasHandle,
        marlin_ws_ptr: u64,
        marlin_ws_ints: usize,
        cfg: &ModelConfig,
        kv_pool: &crate::cache::KvPool,
        kv_layout: &ffi::W34KvLayerLayout,
        attn_batch: &W34AttentionBatch,
        normed_buf: u64,
        q_buf: u64,
        gate_buf: u64,
        gated_attn_buf: u64,
        [
            qsa_qk_buf,
            qsa_q_buf,
            qsa_k_buf,
            qsa_sel_phys_buf,
            qsa_sel_log_buf,
            qsa_num_sel_buf,
            qsa_seq_pos_buf,
        ]: [u64; 7],
        positions_ptr: u64,
        pos_plane_stride: i32,
        num_tokens: usize,
        local_q_heads: usize,
        skip_topk: bool,
    ) -> Result<()> {
        let kv_bufs = &kv_pool.layers[self.kv_layer_ordinal];
        qsa_idx.index_qk_proj.forward(
            ex,
            cublas,
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
            cfg.indexer_n_heads,
            cfg.indexer_kv_heads,
            cfg.indexer_head_dim,
            cfg.rotary_dim,
            cfg.rope_theta,
            cfg.rms_norm_eps,
            cfg.mrope_section[0],
            cfg.mrope_section[1],
            pos_plane_stride as usize,
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
            qsa_idx.k_layernorm_ptr,
            positions_ptr,
            attn_batch.write_slot_mapping as usize as u64,
            0,
            num_tokens,
            cfg.indexer_head_dim,
            cfg.rotary_dim,
            cfg.rope_theta,
            cfg.rms_norm_eps,
            cfg.mrope_section[0],
            cfg.mrope_section[1],
            pos_plane_stride as usize,
        )?;
        let select_k_blocks =
            (cfg.indexer_budget / cfg.indexer_compress_ratio.max(1)).clamp(1, 512) + 1;
        if !skip_topk {
            crate::ple::gpu_ops::qsa_select_blocks_f16(
                ex,
                qsa_q_buf,
                comp_k_cache_ptr,
                attn_batch,
                qsa_sel_phys_buf,
                qsa_sel_log_buf,
                qsa_num_sel_buf,
                qsa_seq_pos_buf,
                cfg.indexer_n_heads,
                cfg.indexer_head_dim,
                16,
                select_k_blocks,
                0,
                0,
            )?;
        }
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
            select_k_blocks,
            attn_batch.sm_scale,
        )
    }
}

#[cfg(feature = "cuda")]
pub struct HcBlockWeights {
    pub hc_norm_ptr: u64,
    pub input_mix_weight_down: LinearProj,
    pub input_mix_weight_up: LinearProj,
    pub block_inject_weight: Option<LinearProj>,
}

#[cfg(feature = "cuda")]
impl HcBlockWeights {
    #[allow(clippy::too_many_arguments)]
    pub fn forward_mix(
        &self,
        ex: &W34Exec,
        cublas: &CublasHandle,
        marlin_ws_ptr: u64,
        marlin_ws_ints: usize,
        hc_residual_buf: u64,
        out_mixed_buf: u64,
        hc_inj_weights_buf: u64,
        ple_buf_a: u64,
        ple_buf_b: u64,
        ple_val_buf: u64,
        num_tokens: usize,
        hc_count: usize,
        hc_lowrank: usize,
        h_dim: usize,
        eps: f32,
    ) -> Result<()> {
        let low_dim = hc_lowrank;
        let hc_down_buf = ple_val_buf;
        let hc_inj_raw_buf = ple_val_buf + (num_tokens * low_dim * 2) as u64;
        crate::ple::gpu_ops::hc_rmsnorm_f16(
            ex,
            hc_residual_buf,
            self.hc_norm_ptr,
            ple_buf_a,
            0,
            num_tokens,
            hc_count,
            h_dim,
            eps,
        )?;
        // Fuse HC down (`N0 <= 512`, e.g. `hc_lowrank = 320`) + inject (`N1 <= 256`, e.g. `4`)
        // for decode and speculative verification (`num_tokens <= 16`), preserving 0-ULP parity
        // between `M = 1` decode and `M <= 16` speculative verify while keeping `m > 16` prefill
        // on cuBLAS Tensor Cores.
        let fused_dual_hc = match (
            &self.input_mix_weight_down,
            &self.block_inject_weight,
            num_tokens <= 16,
        ) {
            (
                LinearProj::DenseF16 {
                    weight_ptr: w0,
                    size_n: n0,
                    size_k: k0,
                },
                Some(LinearProj::DenseF16 {
                    weight_ptr: w1,
                    size_n: n1,
                    size_k: k1,
                }),
                true,
            ) if *n0 <= 512 && *n1 <= 256 && *k0 == *k1 && k0.is_multiple_of(8) => {
                Some((*w0, *n0, *w1, *n1, *k0))
            }
            _ => None,
        };
        let (inj_raw_ptr, inj_w_ptr) = if let Some((w0, n0, w1, n1, k)) = fused_dual_hc {
            check_rc(
                // SAFETY: both HC down and inject weights share the same K extent (`ple_buf_a`)
                // and write to disjoint `hc_down_buf` / `hc_inj_raw_buf` regions in `ple_val_buf`.
                unsafe {
                    ffi::w34_small_n_gemm_dual_f16(
                        ex,
                        ple_buf_a as *const c_void,
                        w0 as *const c_void,
                        hc_down_buf as *mut c_void,
                        n0 as i32,
                        w1 as *const c_void,
                        hc_inj_raw_buf as *mut c_void,
                        n1 as i32,
                        num_tokens as i32,
                        k as i32,
                    )
                },
                "w34_small_n_gemm_dual_f16 (hc_down + hc_inj)",
            )?;
            (hc_inj_raw_buf, hc_inj_weights_buf)
        } else {
            self.input_mix_weight_down.forward(
                ex,
                cublas,
                marlin_ws_ptr,
                marlin_ws_ints,
                ple_buf_a,
                hc_down_buf,
                num_tokens,
            )?;
            if let Some(inj_proj) = &self.block_inject_weight {
                inj_proj.forward(
                    ex,
                    cublas,
                    marlin_ws_ptr,
                    marlin_ws_ints,
                    ple_buf_a,
                    hc_inj_raw_buf,
                    num_tokens,
                )?;
                (hc_inj_raw_buf, hc_inj_weights_buf)
            } else {
                (0, 0)
            }
        };
        crate::ple::gpu_ops::hc_down_silu_inject_f16(
            ex,
            hc_down_buf,
            hc_down_buf,
            inj_raw_ptr,
            inj_w_ptr,
            0,
            num_tokens,
            low_dim,
            hc_count,
        )?;
        self.input_mix_weight_up.forward(
            ex,
            cublas,
            marlin_ws_ptr,
            marlin_ws_ints,
            hc_down_buf,
            ple_buf_b,
            num_tokens,
        )?;
        crate::ple::gpu_ops::hc_mix_streams_f16(
            ex,
            ple_buf_a,
            ple_buf_b,
            out_mixed_buf,
            0,
            num_tokens,
            hc_count,
            h_dim,
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub fn forward_combine(
        &self,
        ex: &W34Exec,
        block_out_buf: u64,
        hc_inj_weights_buf: u64,
        hc_residual_buf: u64,
        num_tokens: usize,
        hc_count: usize,
        h_dim: usize,
    ) -> Result<()> {
        crate::ple::gpu_ops::hc_combine_f16(
            ex,
            hc_residual_buf,
            block_out_buf,
            hc_inj_weights_buf,
            0,
            hc_residual_buf,
            0,
            num_tokens,
            hc_count,
            h_dim,
        )
    }
}

/// Per-layer residual stream topology and normalization weights.
#[cfg(feature = "cuda")]
// One value per layer; inline weights avoid extra indirection in the forward loop.
#[allow(clippy::large_enum_variant)]
pub enum LayerResidual {
    /// Single-stream `[num_tokens, h_dim]` residual with `GemmaRMSNorm` (Qwen 3.5 / 3.6).
    SingleStream {
        input_layernorm_ptr: u64,
        post_attention_layernorm_ptr: u64,
    },
    /// Multi-stream `[num_tokens, hc_count * h_dim]` HyperConnection residual (Qwen 3.8 / 4).
    HyperConnection {
        attn_hc: HcBlockWeights,
        mlp_hc: HcBlockWeights,
    },
}

#[cfg(feature = "cuda")]
impl LayerResidual {
    pub fn single_stream_input_norm_ptr(&self) -> Option<u64> {
        match self {
            Self::SingleStream {
                input_layernorm_ptr,
                ..
            } => Some(*input_layernorm_ptr),
            Self::HyperConnection { .. } => None,
        }
    }

    pub fn single_stream_post_attn_norm_ptr(&self) -> Option<u64> {
        match self {
            Self::SingleStream {
                post_attention_layernorm_ptr,
                ..
            } => Some(*post_attention_layernorm_ptr),
            Self::HyperConnection { .. } => None,
        }
    }

    /// Runs the pre-attention normalization / stream mixing step.
    /// On `SingleStream`, `layer_idx > 0` is a no-op because the preceding layer's
    /// `post_mlp` fused the residual addition with this layer's `input_layernorm`.
    #[allow(clippy::too_many_arguments)]
    pub fn pre_attn(
        &self,
        layer_idx: usize,
        ex: &W34Exec,
        cublas: &CublasHandle,
        marlin_ws_ptr: u64,
        marlin_ws_ints: usize,
        residual_buf: u64,
        hc_residual_buf: u64,
        normed_buf: u64,
        hc_inj_weights_buf: u64,
        ple_buf_a: u64,
        ple_buf_b: u64,
        ple_val_buf: u64,
        num_tokens: usize,
        hc_count: usize,
        hc_lowrank: usize,
        h_dim: usize,
        eps: f32,
    ) -> Result<()> {
        match self {
            Self::SingleStream {
                input_layernorm_ptr,
                ..
            } => {
                if layer_idx == 0 {
                    check_rc(
                        // SAFETY: `residual_buf` and `normed_buf` are disjoint `num_tokens * h_dim`
                        // FP16 buffers; `input_layernorm_ptr` holds `h_dim` FP16 elements.
                        unsafe {
                            ffi::w34_gemma_rms_norm_f16(
                                ex,
                                residual_buf as *const c_void,
                                *input_layernorm_ptr as *const c_void,
                                normed_buf as *mut c_void,
                                core::ptr::null(),
                                num_tokens as i32,
                                h_dim as i32,
                                eps,
                            )
                        },
                        "w34_gemma_rms_norm_f16",
                    )?;
                }
                Ok(())
            }
            Self::HyperConnection { attn_hc, .. } => attn_hc.forward_mix(
                ex,
                cublas,
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
                eps,
            ),
        }
    }

    /// Runs the post-attention residual update and pre-MLP normalization / stream mixing step.
    #[allow(clippy::too_many_arguments)]
    pub fn post_attn_pre_mlp(
        &self,
        ex: &W34Exec,
        cublas: &CublasHandle,
        marlin_ws_ptr: u64,
        marlin_ws_ints: usize,
        tp2: Option<&Tp2Collectives>,
        rank: usize,
        block_out_buf: u64,
        residual_buf: u64,
        hc_residual_buf: u64,
        normed_buf: u64,
        hc_inj_weights_buf: u64,
        ple_buf_a: u64,
        ple_buf_b: u64,
        ple_val_buf: u64,
        num_tokens: usize,
        hc_count: usize,
        hc_lowrank: usize,
        h_dim: usize,
        eps: f32,
        preserve_block_out_on_tp2: bool,
    ) -> Result<()> {
        match self {
            Self::SingleStream {
                post_attention_layernorm_ptr,
                ..
            } => {
                if let Some(comm) = tp2 {
                    let ar_out = if preserve_block_out_on_tp2 {
                        block_out_buf
                    } else {
                        0
                    };
                    comm.all_reduce_residual_gemma_rms_norm_f16(
                        ex,
                        block_out_buf,
                        ar_out,
                        residual_buf,
                        *post_attention_layernorm_ptr,
                        normed_buf,
                        0,
                        rank as i32,
                        num_tokens as i32,
                        h_dim as i32,
                        eps,
                        0,
                        0,
                    )
                } else {
                    check_rc(
                        // SAFETY: `block_out_buf`, `residual_buf`, and `normed_buf` each hold
                        // `num_tokens * h_dim` FP16 elements.
                        unsafe {
                            ffi::w34_fused_add_gemma_rms_norm_f16(
                                ex,
                                block_out_buf as *const c_void,
                                residual_buf as *mut c_void,
                                *post_attention_layernorm_ptr as *const c_void,
                                normed_buf as *mut c_void,
                                core::ptr::null(),
                                num_tokens as i32,
                                h_dim as i32,
                                eps,
                            )
                        },
                        "w34_fused_add_gemma_rms_norm_f16 (post_attn)",
                    )
                }
            }
            Self::HyperConnection { attn_hc, mlp_hc } => {
                if let Some(comm) = tp2 {
                    comm.all_reduce_f16(
                        ex,
                        block_out_buf,
                        block_out_buf,
                        rank as i32,
                        (num_tokens * h_dim) as i32,
                    )?;
                }
                attn_hc.forward_combine(
                    ex,
                    block_out_buf,
                    hc_inj_weights_buf,
                    hc_residual_buf,
                    num_tokens,
                    hc_count,
                    h_dim,
                )?;
                mlp_hc.forward_mix(
                    ex,
                    cublas,
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
                    eps,
                )
            }
        }
    }

    /// Runs the post-MLP residual update (fusing the next layer's input norm or final norm on `SingleStream`).
    #[allow(clippy::too_many_arguments)]
    pub fn post_mlp(
        &self,
        ex: &W34Exec,
        tp2: Option<&Tp2Collectives>,
        rank: usize,
        mlp_out_buf: u64,
        residual_buf: u64,
        hc_residual_buf: u64,
        hc_inj_weights_buf: u64,
        next_norm_weight: u64,
        next_norm_out: u64,
        tap_dst: u64,
        tap_stride: i32,
        tap_offset: i32,
        num_tokens: usize,
        hc_count: usize,
        h_dim: usize,
        eps: f32,
    ) -> Result<()> {
        match self {
            Self::SingleStream { .. } => {
                if let Some(comm) = tp2 {
                    comm.all_reduce_residual_gemma_rms_norm_f16(
                        ex,
                        mlp_out_buf,
                        0,
                        residual_buf,
                        next_norm_weight,
                        next_norm_out,
                        tap_dst,
                        rank as i32,
                        num_tokens as i32,
                        h_dim as i32,
                        eps,
                        tap_stride,
                        tap_offset,
                    )
                } else {
                    check_rc(
                        // SAFETY: `mlp_out_buf`, `residual_buf`, and `next_norm_out` hold
                        // `num_tokens * h_dim` FP16 elements; `tap_dst` is 0 or a validated BF16 tap buffer.
                        unsafe {
                            ffi::w34_fused_add_gemma_rms_norm_tap_f16(
                                ex,
                                mlp_out_buf as *const c_void,
                                residual_buf as *mut c_void,
                                next_norm_weight as *const c_void,
                                next_norm_out as *mut c_void,
                                tap_dst as *mut c_void,
                                core::ptr::null(),
                                num_tokens as i32,
                                h_dim as i32,
                                eps,
                                tap_stride,
                                tap_offset,
                            )
                        },
                        "w34_fused_add_gemma_rms_norm_tap_f16 (post_mlp)",
                    )
                }
            }
            Self::HyperConnection { mlp_hc, .. } => {
                if let Some(comm) = tp2 {
                    comm.all_reduce_f16(
                        ex,
                        mlp_out_buf,
                        mlp_out_buf,
                        rank as i32,
                        (num_tokens * h_dim) as i32,
                    )?;
                }
                mlp_hc.forward_combine(
                    ex,
                    mlp_out_buf,
                    hc_inj_weights_buf,
                    hc_residual_buf,
                    num_tokens,
                    hc_count,
                    h_dim,
                )
            }
        }
    }
}

/// Final pre-LM-head residual normalization / stream mixer.
#[cfg(feature = "cuda")]
pub enum FinalResidual {
    /// Single-stream `GemmaRMSNorm` (`[num_tokens, h_dim] -> [num_tokens, h_dim]`).
    SingleStream { norm_ptr: u64 },
    /// Multi-stream HyperConnection mixer (`[num_tokens, hc_count * h_dim] -> [num_tokens, h_dim]`).
    HyperConnection { mixer: HcBlockWeights },
}

#[cfg(feature = "cuda")]
impl FinalResidual {
    pub fn single_stream_norm_ptr(&self) -> Option<u64> {
        match self {
            Self::SingleStream { norm_ptr } => Some(*norm_ptr),
            Self::HyperConnection { .. } => None,
        }
    }

    pub fn is_hyper_connection(&self) -> bool {
        matches!(self, Self::HyperConnection { .. })
    }

    /// Initializes the multi-stream residual buffer from `embed_buf` (`[num_tokens, h_dim]`)
    /// when `HyperConnection` is active; no-op on `SingleStream`.
    pub fn init_streams(
        &self,
        ex: &W34Exec,
        embed_buf: u64,
        hc_residual_buf: u64,
        num_tokens: usize,
        hc_count: usize,
        h_dim: usize,
    ) -> Result<()> {
        match self {
            Self::SingleStream { .. } => Ok(()),
            Self::HyperConnection { .. } => crate::ple::gpu_ops::hc_init_streams_f16(
                ex,
                embed_buf,
                hc_residual_buf,
                0,
                num_tokens,
                hc_count,
                h_dim,
            ),
        }
    }

    /// Completes the final normalization / stream collapse after the layer stack.
    /// On `SingleStream`, the last layer's `post_mlp` already wrote `final_norm_out_buf`.
    /// On `HyperConnection`, exports `hc_residual_buf` (`[num_tokens, hc_count * h_dim]`) to
    /// `last_final_hidden_f16` (if enabled) and collapses `hc_residual_buf -> final_norm_out_buf`.
    #[allow(clippy::too_many_arguments)]
    pub fn finish_stack(
        &self,
        ex: &W34Exec,
        stream: &CudaStream,
        cublas: &CublasHandle,
        marlin_ws_ptr: u64,
        marlin_ws_ints: usize,
        hc_residual_buf: u64,
        final_norm_out_buf: u64,
        last_final_hidden_f16: Option<&DeviceBuffer>,
        ple_buf_a: u64,
        ple_buf_b: u64,
        ple_val_buf: u64,
        num_tokens: usize,
        hc_count: usize,
        hc_lowrank: usize,
        h_dim: usize,
        eps: f32,
    ) -> Result<()> {
        match self {
            Self::SingleStream { .. } => Ok(()),
            Self::HyperConnection { mixer } => {
                if let Some(final_h_buf) = last_final_hidden_f16
                    && num_tokens * hc_count * h_dim * 2 <= final_h_buf.len_bytes
                {
                    check_rc(
                        // SAFETY: `hc_residual_buf` and `final_h_buf` hold at least
                        // `num_tokens * hc_count * h_dim * 2` bytes.
                        unsafe {
                            ffi::w34_memcpy_d2d_async(
                                final_h_buf.addr() as *mut c_void,
                                hc_residual_buf as *const c_void,
                                num_tokens * hc_count * h_dim * 2,
                                stream.handle,
                            )
                        },
                        "export pre-mixer last_final_hidden_f16",
                    )?;
                }
                mixer.forward_mix(
                    ex,
                    cublas,
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
                    eps,
                )
            }
        }
    }
}

#[cfg(feature = "cuda")]
pub struct PleLayerWeights {
    pub key_proj: LinearProj,
    pub value_proj: LinearProj,
    pub norm_key_ptr: u64,
    pub norm_query_ptr: u64,
    pub norm_conv_ptr: u64,
    pub conv1d_weight_ptr: u64,
    pub weight_scale: f32,
}

#[cfg(feature = "cuda")]
impl PleLayerWeights {
    #[allow(clippy::too_many_arguments)]
    pub fn forward(
        &self,
        ex: &W34Exec,
        cublas: &CublasHandle,
        marlin_ws_ptr: u64,
        marlin_ws_ints: usize,
        ple_f8_staging_ptr: u64,
        hc_residual_buf: u64,
        normed_buf: u64,
        ple_buf_a: u64,
        ple_buf_b: u64,
        ple_val_buf: u64,
        ple_conv_state_ptr: u64,
        gdn_batch: &ffi::W34GdnBatch,
        num_tokens: usize,
        hc_count: usize,
        h_dim: usize,
        eps: f32,
    ) -> Result<()> {
        let total_dim = hc_count * h_dim;
        let ple_embed_dim = self.key_proj.size_k();
        crate::ple::gpu_ops::ple_dequant_f8_to_f16(
            ex,
            ple_f8_staging_ptr,
            ple_val_buf,
            0,
            num_tokens,
            ple_embed_dim,
            self.weight_scale,
        )?;
        self.key_proj.forward(
            ex,
            cublas,
            marlin_ws_ptr,
            marlin_ws_ints,
            ple_val_buf,
            ple_buf_a,
            num_tokens,
        )?;
        self.value_proj.forward(
            ex,
            cublas,
            marlin_ws_ptr,
            marlin_ws_ints,
            ple_val_buf,
            normed_buf,
            num_tokens,
        )?;
        crate::ple::gpu_ops::hc_rmsnorm_f16(
            ex,
            ple_buf_a,
            self.norm_key_ptr,
            ple_buf_a,
            0,
            num_tokens,
            hc_count,
            h_dim,
            eps,
        )?;
        crate::ple::gpu_ops::hc_rmsnorm_f16(
            ex,
            hc_residual_buf,
            self.norm_query_ptr,
            ple_buf_b,
            0,
            num_tokens,
            hc_count,
            h_dim,
            eps,
        )?;
        crate::ple::gpu_ops::ple_signed_sqrt_gate_f16(
            ex, ple_buf_b, ple_buf_a, normed_buf, ple_buf_a, 0, num_tokens, hc_count, h_dim, 1e-6,
        )?;
        check_rc(
            // SAFETY: hc_residual_buf and ple_buf_a each hold num_tokens * total_dim FP16 elements.
            unsafe {
                ffi::w34_add_inplace_f16(
                    ex,
                    hc_residual_buf as *mut c_void,
                    ple_buf_a as *const c_void,
                    (num_tokens * total_dim) as i64,
                )
            },
            "w34_add_inplace_f16 (ple gated_value)",
        )?;
        crate::ple::gpu_ops::hc_rmsnorm_f16(
            ex,
            ple_buf_a,
            self.norm_conv_ptr,
            ple_buf_b,
            0,
            num_tokens,
            hc_count,
            h_dim,
            eps,
        )?;
        crate::ple::gpu_ops::ple_dilated_conv1d_f16(
            ex,
            ple_buf_b,
            self.conv1d_weight_ptr,
            ple_conv_state_ptr,
            ple_buf_a,
            gdn_batch,
            total_dim,
        )?;
        check_rc(
            // SAFETY: hc_residual_buf and ple_buf_a each hold num_tokens * total_dim FP16 elements.
            unsafe {
                ffi::w34_add_inplace_f16(
                    ex,
                    hc_residual_buf as *mut c_void,
                    ple_buf_a as *const c_void,
                    (num_tokens * total_dim) as i64,
                )
            },
            "w34_add_inplace_f16 (ple conv_out)",
        )
    }
}

#[cfg(feature = "cuda")]
pub struct GdnLayerWeights {
    pub layer_id: usize,
    pub gdn_layer_ordinal: usize,
    pub in_proj_qkvz: LinearProj,
    pub in_proj_ba: LinearProj,
    pub out_proj: LinearProj,
    pub conv_weight_ptr: u64,
    pub a_log_ptr: u64,
    pub dt_bias_ptr: u64,
    pub norm_weight_ptr: u64,
}

#[cfg(feature = "cuda")]
pub enum LayerBlockWeights {
    FullAttention(FullAttentionLayerWeights),
    LinearAttention(GdnLayerWeights),
}
