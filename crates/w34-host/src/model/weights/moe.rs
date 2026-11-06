use super::*;

#[cfg(feature = "cuda")]
pub enum MoeExpertStorage {
    W4A16 {
        gate_up_packed_ptr: u64,
        gate_up_scales_ptr: u64,
        down_packed_ptr: u64,
        down_scales_ptr: u64,
        group_size: usize,
    },
    W4A16Zp {
        gate_up_packed_ptr: u64,
        gate_up_scales_ptr: u64,
        gate_up_qzeros_ptr: u64,
        down_packed_ptr: u64,
        down_scales_ptr: u64,
        down_qzeros_ptr: u64,
        group_size: usize,
    },
    DenseF16 {
        gate_up_ptr: u64,
        down_ptr: u64,
    },
    /// Host/VRAM tiered expert storage managed by `RankTieredMoe` (or MTP's tiered workspace).
    Tiered {
        layer_id: usize,
    },
}

#[cfg(feature = "cuda")]
pub struct MoeLayerWeights {
    pub router_gate: LinearProj,
    pub experts: MoeExpertStorage,
    pub num_experts: usize,
    pub top_k: usize,
    pub norm_topk_prob: bool,
    pub scoring_func: i32,
    pub local_moe_interm: usize,
    pub shared_gate_up_proj: Option<LinearProj>,
    pub shared_down_proj: Option<LinearProj>,
    pub shared_expert_gate: Option<LinearProj>,
    pub local_shared_interm: usize,
}

#[cfg(feature = "cuda")]
pub enum FfnBlockWeights {
    Dense {
        gate_up_proj: LinearProj,
        down_proj: LinearProj,
    },
    Moe(Box<MoeLayerWeights>),
}

#[cfg(feature = "cuda")]
impl FfnBlockWeights {
    #[inline]
    pub fn pre_mlp_norm_targets(
        &self,
        normed_buf: u64,
        h_dim: usize,
        is_single_stream: bool,
    ) -> [u64; 2] {
        if is_single_stream
            && h_dim <= 6144
            && h_dim.is_multiple_of(64)
            && matches!(self, Self::Dense { gate_up_proj, .. } if gate_up_proj.can_fuse_h64_from_producer())
        {
            [0, normed_buf]
        } else {
            [normed_buf, 0]
        }
    }

    /// Enqueue the loaded FFN using the model's phase-local scratch layout.
    /// Raw buffer extents, device affinity, and ordering are caller obligations;
    /// accepting u64 addresses here does not validate arbitrary allocations.
    #[allow(clippy::too_many_arguments)]
    pub fn forward(
        &self,
        ex: &W34Exec,
        cublas: &CublasHandle,
        marlin_ws_ptr: u64,
        marlin_ws_ints: usize,
        normed_buf: u64,
        mlp_out_buf: u64,
        num_tokens: usize,
        h_dim: usize,
        local_interm: usize,
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
    ) -> Result<()> {
        self.forward_with_prerotated_input(
            ex,
            cublas,
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
            false,
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub fn forward_with_prerotated_input(
        &self,
        ex: &W34Exec,
        cublas: &CublasHandle,
        marlin_ws_ptr: u64,
        marlin_ws_ints: usize,
        normed_buf: u64,
        mlp_out_buf: u64,
        num_tokens: usize,
        h_dim: usize,
        local_interm: usize,
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
        input_prerotated: bool,
    ) -> Result<()> {
        match self {
            Self::Dense {
                gate_up_proj,
                down_proj,
            } => {
                if input_prerotated {
                    gate_up_proj.forward_prerotated(
                        ex,
                        cublas,
                        marlin_ws_ptr,
                        marlin_ws_ints,
                        normed_buf,
                        gate_up_buf,
                        num_tokens,
                    )?;
                } else {
                    gate_up_proj.forward(
                        ex,
                        cublas,
                        marlin_ws_ptr,
                        marlin_ws_ints,
                        normed_buf,
                        gate_up_buf,
                        num_tokens,
                    )?;
                }
                let fuse_down_h64 = down_proj.can_fuse_h64_from_producer();
                let silu_fn = if fuse_down_h64 {
                    ffi::w34_silu_and_mul_hadamard_f16
                } else {
                    ffi::w34_silu_and_mul_f16
                };
                check_rc(
                    // SAFETY: the model's dense FFN phase reserves [M,2*interm]
                    // gate/up and [M,interm] activation regions. The projection
                    // precedes this launch on ex.stream; these addresses are supplied
                    // by the caller rather than capacity-checked by this method.
                    unsafe {
                        silu_fn(
                            ex,
                            gate_up_buf as *const c_void,
                            act_buf as *mut c_void,
                            core::ptr::null(),
                            num_tokens as i32,
                            local_interm as i32,
                        )
                    },
                    "w34_silu_and_mul_f16",
                )?;
                if fuse_down_h64 {
                    down_proj.forward_prerotated(
                        ex,
                        cublas,
                        marlin_ws_ptr,
                        marlin_ws_ints,
                        act_buf,
                        mlp_out_buf,
                        num_tokens,
                    )
                } else {
                    down_proj.forward(
                        ex,
                        cublas,
                        marlin_ws_ptr,
                        marlin_ws_ints,
                        act_buf,
                        mlp_out_buf,
                        num_tokens,
                    )
                }
            }
            Self::Moe(moe) => {
                let fused_dual_small_n = match (
                    &moe.router_gate,
                    &moe.shared_expert_gate,
                    moe.shared_gate_up_proj.is_some() && moe.shared_down_proj.is_some(),
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
                        // `n0 <= 256` runs across all `num_tokens`; `256 < n0 <= 512` (512-expert router)
                        // runs for decode and speculative verify (`num_tokens <= 16`) so `M = 1` and
                        // `M <= 16` verify are 0-ULP identical while `m > 16` prefill uses Tensor Cores.
                    ) if (*n0 <= 256 || (*n0 <= 512 && num_tokens <= 16))
                        && *n1 <= 256
                        && *k0 == *k1
                        && k0.is_multiple_of(8) =>
                    {
                        Some((*w0, *n0, *w1, *n1, *k0))
                    }
                    _ => None,
                };
                if let Some((w0, n0, w1, n1, k)) = fused_dual_small_n {
                    check_rc(
                        // SAFETY: the match checked both small-N geometries and
                        // equal K. Router/shared-gate weights come from upload;
                        // model scratch provides separate router and gate outputs.
                        unsafe {
                            ffi::w34_small_n_gemm_dual_f16(
                                ex,
                                normed_buf as *const c_void,
                                w0 as *const c_void,
                                moe_router_logits_buf as *mut c_void,
                                n0 as i32,
                                w1 as *const c_void,
                                moe_sh_gate_buf as *mut c_void,
                                n1 as i32,
                                num_tokens as i32,
                                k as i32,
                            )
                        },
                        "w34_small_n_gemm_dual_f16",
                    )?;
                } else {
                    moe.router_gate.forward(
                        ex,
                        cublas,
                        marlin_ws_ptr,
                        marlin_ws_ints,
                        normed_buf,
                        moe_router_logits_buf,
                        num_tokens,
                    )?;
                }
                check_rc(
                    // SAFETY: router projection precedes this on ex.stream.
                    // The model scratch layout reserves M*experts FP16 logits and
                    // M*top_k FP32 weights/i32 IDs for the loaded MoE geometry.
                    unsafe {
                        ffi::w34_moe_topk_router_f16(
                            ex,
                            moe_router_logits_buf as *const c_void,
                            moe_topk_weights_buf as *mut f32,
                            moe_topk_ids_buf as *mut i32,
                            num_tokens as i32,
                            moe.num_experts as i32,
                            moe.top_k as i32,
                            i32::from(moe.norm_topk_prob),
                            moe.scoring_func,
                        )
                    },
                    "w34_moe_topk_router_f16",
                )?;

                let (sh_out_ptr, sh_gate_ptr) = if let (Some(sh_gu), Some(sh_dn)) =
                    (&moe.shared_gate_up_proj, &moe.shared_down_proj)
                {
                    sh_gu.forward(
                        ex,
                        cublas,
                        marlin_ws_ptr,
                        marlin_ws_ints,
                        normed_buf,
                        moe_sh_gu_buf,
                        num_tokens,
                    )?;
                    check_rc(
                        // SAFETY: the shared projection just produced [M,2*shared]
                        // gate/up; model scratch reserves [M,shared] activation.
                        // Both regions are distinct within the FFN phase.
                        unsafe {
                            ffi::w34_silu_and_mul_f16(
                                ex,
                                moe_sh_gu_buf as *const c_void,
                                moe_sh_act_buf as *mut c_void,
                                core::ptr::null(),
                                num_tokens as i32,
                                moe.local_shared_interm as i32,
                            )
                        },
                        "w34_silu_and_mul_f16 (shared_expert)",
                    )?;
                    sh_dn.forward(
                        ex,
                        cublas,
                        marlin_ws_ptr,
                        marlin_ws_ints,
                        moe_sh_act_buf,
                        moe_sh_out_buf,
                        num_tokens,
                    )?;
                    let sg_ptr = if let Some(sh_gate) = &moe.shared_expert_gate {
                        if fused_dual_small_n.is_none() {
                            sh_gate.forward(
                                ex,
                                cublas,
                                marlin_ws_ptr,
                                marlin_ws_ints,
                                normed_buf,
                                moe_sh_gate_buf,
                                num_tokens,
                            )?;
                        }
                        moe_sh_gate_buf as *const c_void
                    } else {
                        core::ptr::null()
                    };
                    (moe_sh_out_buf as *const c_void, sg_ptr)
                } else {
                    (core::ptr::null(), core::ptr::null())
                };

                match &moe.experts {
                    MoeExpertStorage::W4A16 {
                        gate_up_packed_ptr,
                        gate_up_scales_ptr,
                        down_packed_ptr,
                        down_scales_ptr,
                        group_size,
                    } => check_rc(
                        // SAFETY: loader stores expert-major packed U4 weights and
                        // FP16 scales with the loaded group size. Top-k buffers were
                        // produced above; model scratch reserves routed intermediate
                        // and output rows. Shared inputs are null when absent.
                        unsafe {
                            ffi::w34_moe_w4a16_mlp_f16(
                                ex,
                                normed_buf as *const c_void,
                                *gate_up_packed_ptr as *const u32,
                                *gate_up_scales_ptr as *const c_void,
                                *down_packed_ptr as *const u32,
                                *down_scales_ptr as *const c_void,
                                moe_topk_ids_buf as *const i32,
                                moe_topk_weights_buf as *const f32,
                                sh_out_ptr,
                                sh_gate_ptr,
                                moe_interm_buf as *mut c_void,
                                mlp_out_buf as *mut c_void,
                                num_tokens as i32,
                                moe.top_k as i32,
                                moe.num_experts as i32,
                                moe.local_moe_interm as i32,
                                h_dim as i32,
                                *group_size as i32,
                            )
                        },
                        "w34_moe_w4a16_mlp_f16",
                    ),
                    MoeExpertStorage::W4A16Zp {
                        gate_up_packed_ptr,
                        gate_up_scales_ptr,
                        gate_up_qzeros_ptr,
                        down_packed_ptr,
                        down_scales_ptr,
                        down_qzeros_ptr,
                        group_size,
                    } => check_rc(
                        // SAFETY: expert-major U4 weights/scales/zero points use
                        // the same loaded geometry. Router outputs and shared inputs
                        // precede this on ex.stream; model scratch supplies separate
                        // routed intermediates/output, with null shared inputs allowed.
                        unsafe {
                            ffi::w34_moe_w4a16_mlp_zp_f16(
                                ex,
                                normed_buf as *const c_void,
                                *gate_up_packed_ptr as *const u32,
                                *gate_up_scales_ptr as *const c_void,
                                *gate_up_qzeros_ptr as *const u32,
                                *down_packed_ptr as *const u32,
                                *down_scales_ptr as *const c_void,
                                *down_qzeros_ptr as *const u32,
                                moe_topk_ids_buf as *const i32,
                                moe_topk_weights_buf as *const f32,
                                sh_out_ptr,
                                sh_gate_ptr,
                                moe_interm_buf as *mut c_void,
                                mlp_out_buf as *mut c_void,
                                num_tokens as i32,
                                moe.top_k as i32,
                                moe.num_experts as i32,
                                moe.local_moe_interm as i32,
                                h_dim as i32,
                                *group_size as i32,
                            )
                        },
                        "w34_moe_w4a16_mlp_zp_f16",
                    ),
                    MoeExpertStorage::DenseF16 {
                        gate_up_ptr,
                        down_ptr,
                    } => {
                        check_rc(
                            // SAFETY: dense expert weights are uploaded with this
                            // expert/intermediate/hidden geometry. Router IDs select
                            // experts; model scratch provides M*top_k*interm FP16
                            // intermediates distinct from the normalized input.
                            unsafe {
                                ffi::w34_moe_f16_gate_up_silu(
                                    ex,
                                    normed_buf as *const c_void,
                                    *gate_up_ptr as *const c_void,
                                    moe_topk_ids_buf as *const i32,
                                    moe_interm_buf as *mut c_void,
                                    num_tokens as i32,
                                    moe.top_k as i32,
                                    moe.num_experts as i32,
                                    moe.local_moe_interm as i32,
                                    h_dim as i32,
                                )
                            },
                            "w34_moe_f16_gate_up_silu",
                        )?;
                        check_rc(
                            // SAFETY: gate/up above produced routed intermediates;
                            // down weights use the matching expert geometry and
                            // output is M*h_dim FP16 in the model scratch layout.
                            // Same-stream ordering preserves router/shared inputs.
                            unsafe {
                                ffi::w34_moe_f16_down_accum(
                                    ex,
                                    moe_interm_buf as *const c_void,
                                    *down_ptr as *const c_void,
                                    moe_topk_ids_buf as *const i32,
                                    moe_topk_weights_buf as *const f32,
                                    sh_out_ptr,
                                    sh_gate_ptr,
                                    mlp_out_buf as *mut c_void,
                                    num_tokens as i32,
                                    moe.top_k as i32,
                                    moe.num_experts as i32,
                                    h_dim as i32,
                                    moe.local_moe_interm as i32,
                                )
                            },
                            "w34_moe_f16_down_accum",
                        )
                    }
                    MoeExpertStorage::Tiered { .. } => Ok(()),
                }
            }
        }
    }

    /// Executes the host/VRAM tiered MoE dispatch when `self` is a tiered `FfnBlockWeights::Moe`;
    /// no-op for `Dense` and VRAM-resident `Moe`.
    pub fn execute_tiered(
        &self,
        tiered_moe: Option<&mut crate::moe_offload::RankTieredMoe>,
        stream: &CudaStream,
        is_capturing_graph: bool,
        normed_buf: u64,
        [
            moe_topk_ids_buf,
            moe_topk_weights_buf,
            moe_sh_out_buf,
            moe_sh_gate_buf,
            mlp_out_buf,
        ]: [u64; 5],
        num_tokens: usize,
    ) -> Result<()> {
        let Self::Moe(moe) = self else {
            return Ok(());
        };
        let MoeExpertStorage::Tiered { layer_id } = moe.experts else {
            return Ok(());
        };
        let tmoe_engine =
            tiered_moe.ok_or_else(|| loader_err("missing RankTieredMoe for tiered MoE layer"))?;
        let sh_out_ptr = if moe.shared_down_proj.is_some() {
            moe_sh_out_buf
        } else {
            0
        };
        let sh_gate_ptr = if moe.shared_expert_gate.is_some() {
            moe_sh_gate_buf
        } else {
            0
        };
        let device_lru_safe = tmoe_engine.layer_is_fully_resident(layer_id)
            || (num_tokens * moe.top_k <= tmoe_engine.layer_hot_k(layer_id)
                && tmoe_engine.layer_has_mapped_store(layer_id));
        if is_capturing_graph && !device_lru_safe {
            return Err(loader_err(
                "tiered MoE graph exceeds mapped hot-cache capacity",
            ));
        }
        if device_lru_safe {
            tmoe_engine.execute_layer_device_lru(
                stream,
                layer_id,
                normed_buf,
                moe_topk_ids_buf,
                moe_topk_weights_buf,
                sh_out_ptr,
                sh_gate_ptr,
                mlp_out_buf,
                num_tokens,
                moe.top_k,
            )
        } else {
            tmoe_engine.execute_layer(
                stream,
                layer_id,
                normed_buf,
                moe_topk_ids_buf,
                moe_topk_weights_buf,
                sh_out_ptr,
                sh_gate_ptr,
                mlp_out_buf,
                num_tokens,
                moe.top_k,
            )
        }
    }
}

#[cfg(feature = "cuda")]
pub(crate) fn load_moe_layer_weights(
    archive: &SafetensorsArchive,
    cfg: &ModelConfig,
    tiered_layer_id: Option<usize>,
    pfx: &str,
    [rank, tp_size]: [usize; 2],
    weight_arena: &mut DeviceArena,
    (stream, repack_staging): (&CudaStream, &DeviceBuffer),
) -> Result<MoeLayerWeights> {
    let raw_router = RawLinearHost::load(archive, &format!("{pfx}.mlp.gate"), &cfg.quant)?;
    let router_gate = raw_router.upload(weight_arena, stream, repack_staging)?;

    let full_interm = cfg.moe_intermediate_size;
    if tiered_layer_id.is_none() && !full_interm.is_multiple_of(tp_size) {
        return Err(loader_err_msg(format!(
            "moe_intermediate_size {} is not divisible by TP={tp_size}",
            full_interm
        )));
    }
    // Tiered MoE partitions expert IDs across EP ranks, preserving full `moe_intermediate_size` per expert.
    let local_moe_interm = if tiered_layer_id.is_some() {
        full_interm
    } else {
        full_interm / tp_size
    };
    let num_exp = cfg.num_experts;
    let top_k = cfg.num_experts_per_tok.max(1);
    let scoring_func = i32::from(cfg.scoring_func.eq_ignore_ascii_case("sigmoid"));
    let h = cfg.hidden_size;

    let fused_gu_key = archive
        .resolve_name(&format!("{pfx}.mlp.experts.gate_up_proj"))
        .or_else(|| archive.resolve_name(&format!("{pfx}.mlp.experts.gate_up_proj.weight")));
    let fused_dn_key = archive
        .resolve_name(&format!("{pfx}.mlp.experts.down_proj"))
        .or_else(|| archive.resolve_name(&format!("{pfx}.mlp.experts.down_proj.weight")));

    let experts = if let Some(layer_id) = tiered_layer_id {
        MoeExpertStorage::Tiered { layer_id }
    } else if let (Some(gu_key), Some(dn_key)) = (fused_gu_key, fused_dn_key) {
        if archive
            .resolve_name(&format!("{pfx}.mlp.experts.gate_up_proj_scale_inv"))
            .is_some()
            || archive
                .resolve_name(&format!("{pfx}.mlp.experts.gate_up_proj.weight_scale"))
                .is_some()
            || archive
                .resolve_name(&format!("{pfx}.mlp.experts.gate_up_proj.weight_scale_inv"))
                .is_some()
        {
            return Err(loader_err_msg(format!(
                "unsupported FP8 fused MoE expert tensor '{gu_key}' (warp-34 sm_86/sm_89 spine requires W4A16 or BF16/FP16 MoE)"
            )));
        }
        let gu_info = archive
            .tensor_info(&gu_key)
            .map_err(|e| loader_err_msg(format!("tensor_info fused experts.gate_up_proj: {e}")))?;
        let dn_info = archive
            .tensor_info(&dn_key)
            .map_err(|e| loader_err_msg(format!("tensor_info fused experts.down_proj: {e}")))?;
        if gu_info.shape.as_slice() != [num_exp, 2 * full_interm, h]
            || dn_info.shape.as_slice() != [num_exp, h, full_interm]
        {
            return Err(loader_err_msg(format!(
                "unexpected fused 3D MoE expert shapes on '{pfx}': gate_up_proj={:?}, down_proj={:?}",
                gu_info.shape, dn_info.shape
            )));
        }
        let gu_all = archive
            .read_f16_bits(&gu_key)
            .map_err(|e| loader_err_msg(format!("read fused experts.gate_up_proj: {e}")))?;
        let dn_all = archive
            .read_f16_bits(&dn_key)
            .map_err(|e| loader_err_msg(format!("read fused experts.down_proj: {e}")))?;

        let expected_gu = num_exp
            .checked_mul(2)
            .and_then(|v| v.checked_mul(full_interm))
            .and_then(|v| v.checked_mul(h))
            .ok_or_else(|| loader_err("fused experts.gate_up_proj size overflow"))?;
        let expected_dn = num_exp
            .checked_mul(h)
            .and_then(|v| v.checked_mul(full_interm))
            .ok_or_else(|| loader_err("fused experts.down_proj size overflow"))?;
        if gu_all.len() != expected_gu || dn_all.len() != expected_dn {
            return Err(loader_err(
                "fused MoE expert tensor element count mismatch against shape",
            ));
        }
        let local_gu_cap = num_exp
            .checked_mul(2)
            .and_then(|v| v.checked_mul(local_moe_interm))
            .and_then(|v| v.checked_mul(h))
            .ok_or_else(|| loader_err("local fused gate_up capacity overflow"))?;
        let local_dn_cap = num_exp
            .checked_mul(h)
            .and_then(|v| v.checked_mul(local_moe_interm))
            .ok_or_else(|| loader_err("local fused down capacity overflow"))?;
        let mut all_gu_f16 = Vec::with_capacity(local_gu_cap);
        let mut all_dn_f16 = Vec::with_capacity(local_dn_cap);
        let col_start = rank * local_moe_interm;
        let col_end = (rank + 1) * local_moe_interm;

        for e in 0..num_exp {
            let exp_gu = &gu_all[e * 2 * full_interm * h..(e + 1) * 2 * full_interm * h];
            all_gu_f16.extend_from_slice(&exp_gu[col_start * h..col_end * h]);
            all_gu_f16.extend_from_slice(
                &exp_gu[(full_interm + col_start) * h..(full_interm + col_end) * h],
            );

            let exp_dn = &dn_all[e * h * full_interm..(e + 1) * h * full_interm];
            if tp_size == 1 {
                all_dn_f16.extend_from_slice(exp_dn);
            } else {
                for row in 0..h {
                    let row_base = row * full_interm;
                    all_dn_f16.extend_from_slice(&exp_dn[row_base + col_start..row_base + col_end]);
                }
            }
        }
        drop(gu_all);
        drop(dn_all);
        let gate_up_ptr = weight_arena.upload_slice(&all_gu_f16, stream)?;
        let down_ptr = weight_arena.upload_slice(&all_dn_f16, stream)?;
        MoeExpertStorage::DenseF16 {
            gate_up_ptr,
            down_ptr,
        }
    } else {
        if archive
            .resolve_name(&format!("{pfx}.mlp.experts.gate_up_proj_blocks"))
            .is_some()
            || archive
                .resolve_name(&format!("{pfx}.mlp.experts.gate_up_proj.weight_packed"))
                .is_some()
        {
            return Err(loader_err_msg(format!(
                "unsupported fused 3D quantized MoE expert layout on '{pfx}' (expected per-expert W4A16 or 3D BF16/FP16)"
            )));
        }
        let load_expert_parts = |e: usize| -> Result<(RawLinearHost, RawLinearHost)> {
            let epfx = format!("{pfx}.mlp.experts.{e}");
            let rg_full = RawLinearHost::load(archive, &format!("{epfx}.gate_proj"), &cfg.quant)?;
            if rg_full.dims() != (full_interm, h) {
                return Err(loader_err_msg(format!(
                    "MoE gate_proj dims {:?} at '{epfx}' do not match expected ({full_interm}, {h})",
                    rg_full.dims()
                )));
            }
            let ru_full = RawLinearHost::load(archive, &format!("{epfx}.up_proj"), &cfg.quant)?;
            if ru_full.dims() != (full_interm, h) {
                return Err(loader_err_msg(format!(
                    "MoE up_proj dims {:?} at '{epfx}' do not match expected ({full_interm}, {h})",
                    ru_full.dims()
                )));
            }
            let rd_full = RawLinearHost::load(archive, &format!("{epfx}.down_proj"), &cfg.quant)?;
            if rd_full.dims() != (h, full_interm) {
                return Err(loader_err_msg(format!(
                    "MoE down_proj dims {:?} at '{epfx}' do not match expected ({h}, {full_interm})",
                    rd_full.dims()
                )));
            }
            let rg = rg_full.slice_rows(rank * local_moe_interm, (rank + 1) * local_moe_interm);
            let ru = ru_full.slice_rows(rank * local_moe_interm, (rank + 1) * local_moe_interm);
            let gu = RawLinearHost::concat_rows(&[rg, ru])?;
            let rd = rd_full.slice_cols(rank * local_moe_interm, (rank + 1) * local_moe_interm)?;
            Ok((gu, rd))
        };

        // Probe expert 0 to determine W4A16 vs W4A16Zp vs DenseF16 storage
        let exp0_gate = RawLinearHost::load(
            archive,
            &format!("{pfx}.mlp.experts.0.gate_proj"),
            &cfg.quant,
        )?;
        if matches!(exp0_gate, RawLinearHost::U8B128NMajor { .. }) {
            return Err(loader_err_msg(format!(
                "unsupported 8-bit W8A16 routed MoE experts at '{pfx}' (sm_86/sm_89 MoE supports 4-bit W4A16 or FP16/BF16 experts)"
            )));
        }
        let num_workers = std::thread::available_parallelism()
            .map(|n| (n.get() / tp_size.max(1)).clamp(1, 8))
            .unwrap_or(4)
            .min(num_exp.max(1));
        const EXP_BATCH: usize = 32;
        let load_expert_batch =
            |batch_start: usize, batch_end: usize| -> Result<Vec<(RawLinearHost, RawLinearHost)>> {
                let batch_len = batch_end - batch_start;
                if num_workers <= 1 || batch_len <= 4 {
                    (batch_start..batch_end)
                        .map(&load_expert_parts)
                        .collect::<Result<Vec<_>>>()
                } else {
                    let chunk_size = batch_len.div_ceil(num_workers).max(1);
                    std::thread::scope(|s| -> Result<Vec<(RawLinearHost, RawLinearHost)>> {
                        let mut handles = Vec::with_capacity(num_workers);
                        for chunk_idx in 0..batch_len.div_ceil(chunk_size) {
                            let start = batch_start + chunk_idx * chunk_size;
                            let end = (start + chunk_size).min(batch_end);
                            let load_fn = &load_expert_parts;
                            handles.push(s.spawn(
                                move || -> Result<Vec<(RawLinearHost, RawLinearHost)>> {
                                    let mut out = Vec::with_capacity(end - start);
                                    for e in start..end {
                                        out.push(load_fn(e)?);
                                    }
                                    Ok(out)
                                },
                            ));
                        }
                        let mut all = Vec::with_capacity(batch_len);
                        for h in handles {
                            let chunk = h
                                .join()
                                .map_err(|_| loader_err("MoE expert loader thread panicked"))??;
                            all.extend(chunk);
                        }
                        Ok(all)
                    })
                }
            };

        match exp0_gate {
            RawLinearHost::U4B8NMajor { group_size, .. } => {
                if !h.is_multiple_of(8)
                    || !local_moe_interm.is_multiple_of(8)
                    || !h.is_multiple_of(group_size)
                    || !local_moe_interm.is_multiple_of(group_size)
                {
                    return Err(loader_err_msg(format!(
                        "W4A16 MoE requires hidden_size ({h}) and local_moe_interm ({local_moe_interm}) divisible by 8 and group_size ({group_size})"
                    )));
                }
                let gu_words_per_exp = (2 * local_moe_interm) * (h / 8);
                let gu_scales_per_exp = (2 * local_moe_interm) * (h / group_size);
                let dn_words_per_exp = h * (local_moe_interm / 8);
                let dn_scales_per_exp = h * (local_moe_interm / group_size);

                let mut all_gu_packed = Vec::with_capacity(num_exp * gu_words_per_exp);
                let mut all_gu_scales = Vec::with_capacity(num_exp * gu_scales_per_exp);
                let mut all_dn_packed = Vec::with_capacity(num_exp * dn_words_per_exp);
                let mut all_dn_scales = Vec::with_capacity(num_exp * dn_scales_per_exp);

                for batch_start in (0..num_exp).step_by(EXP_BATCH) {
                    let batch_end = (batch_start + EXP_BATCH).min(num_exp);
                    for (rel_e, (gu, rd)) in load_expert_batch(batch_start, batch_end)?
                        .into_iter()
                        .enumerate()
                    {
                        let e = batch_start + rel_e;
                        let epfx = format!("{pfx}.mlp.experts.{e}");
                        if let RawLinearHost::U4B8NMajor {
                            packed,
                            scales,
                            group_size: gs,
                            ..
                        } = gu
                            && gs == group_size
                        {
                            all_gu_packed.extend_from_slice(&packed);
                            all_gu_scales.extend_from_slice(&scales);
                        } else {
                            return Err(loader_err_msg(format!(
                                "inconsistent MoE expert quantization at '{epfx}' (expected symmetric W4A16 group_size={group_size})"
                            )));
                        }
                        if let RawLinearHost::U4B8NMajor {
                            packed,
                            scales,
                            group_size: gs,
                            ..
                        } = rd
                            && gs == group_size
                        {
                            all_dn_packed.extend_from_slice(&packed);
                            all_dn_scales.extend_from_slice(&scales);
                        } else {
                            return Err(loader_err_msg(format!(
                                "inconsistent MoE down_proj quantization at '{epfx}' (expected symmetric W4A16 group_size={group_size})"
                            )));
                        }
                    }
                }

                let gate_up_packed_ptr = weight_arena.upload_slice(&all_gu_packed, stream)?;
                let gate_up_scales_ptr = weight_arena.upload_slice(&all_gu_scales, stream)?;
                let down_packed_ptr = weight_arena.upload_slice(&all_dn_packed, stream)?;
                let down_scales_ptr = weight_arena.upload_slice(&all_dn_scales, stream)?;
                MoeExpertStorage::W4A16 {
                    gate_up_packed_ptr,
                    gate_up_scales_ptr,
                    down_packed_ptr,
                    down_scales_ptr,
                    group_size,
                }
            }
            RawLinearHost::U4ZpNMajor { group_size, .. } => {
                if !h.is_multiple_of(8)
                    || !local_moe_interm.is_multiple_of(8)
                    || !h.is_multiple_of(group_size)
                    || !local_moe_interm.is_multiple_of(group_size)
                {
                    return Err(loader_err_msg(format!(
                        "W4A16 asymmetric MoE requires hidden_size ({h}) and local_moe_interm ({local_moe_interm}) divisible by 8 and group_size ({group_size})"
                    )));
                }
                let gu_groups = h / group_size;
                let dn_groups = local_moe_interm / group_size;
                let gu_words_per_exp = (2 * local_moe_interm) * (h / 8);
                let gu_scales_per_exp = (2 * local_moe_interm) * gu_groups;
                let gu_qzeros_per_exp = ((2 * local_moe_interm) / 8) * gu_groups;
                let dn_words_per_exp = h * (local_moe_interm / 8);
                let dn_scales_per_exp = h * dn_groups;
                let dn_qzeros_per_exp = (h / 8) * dn_groups;

                let mut all_gu_packed = Vec::with_capacity(num_exp * gu_words_per_exp);
                let mut all_gu_scales = Vec::with_capacity(num_exp * gu_scales_per_exp);
                let mut all_gu_qzeros = Vec::with_capacity(num_exp * gu_qzeros_per_exp);
                let mut all_dn_packed = Vec::with_capacity(num_exp * dn_words_per_exp);
                let mut all_dn_scales = Vec::with_capacity(num_exp * dn_scales_per_exp);
                let mut all_dn_qzeros = Vec::with_capacity(num_exp * dn_qzeros_per_exp);

                for batch_start in (0..num_exp).step_by(EXP_BATCH) {
                    let batch_end = (batch_start + EXP_BATCH).min(num_exp);
                    for (rel_e, (gu, rd)) in load_expert_batch(batch_start, batch_end)?
                        .into_iter()
                        .enumerate()
                    {
                        let e = batch_start + rel_e;
                        let epfx = format!("{pfx}.mlp.experts.{e}");
                        if let RawLinearHost::U4ZpNMajor {
                            packed,
                            scales,
                            qzeros,
                            group_size: gs,
                            ..
                        } = gu
                            && gs == group_size
                        {
                            all_gu_packed.extend_from_slice(&packed);
                            all_gu_scales.extend_from_slice(&scales);
                            all_gu_qzeros.extend_from_slice(&qzeros);
                        } else {
                            return Err(loader_err_msg(format!(
                                "inconsistent MoE expert quantization at '{epfx}' (expected asymmetric W4A16 group_size={group_size})"
                            )));
                        }
                        if let RawLinearHost::U4ZpNMajor {
                            packed,
                            scales,
                            qzeros,
                            group_size: gs,
                            ..
                        } = rd
                            && gs == group_size
                        {
                            all_dn_packed.extend_from_slice(&packed);
                            all_dn_scales.extend_from_slice(&scales);
                            all_dn_qzeros.extend_from_slice(&qzeros);
                        } else {
                            return Err(loader_err_msg(format!(
                                "inconsistent MoE down_proj quantization at '{epfx}' (expected asymmetric W4A16 group_size={group_size})"
                            )));
                        }
                    }
                }

                let gate_up_packed_ptr = weight_arena.upload_slice(&all_gu_packed, stream)?;
                let gate_up_scales_ptr = weight_arena.upload_slice(&all_gu_scales, stream)?;
                let gate_up_qzeros_ptr = weight_arena.upload_slice(&all_gu_qzeros, stream)?;
                let down_packed_ptr = weight_arena.upload_slice(&all_dn_packed, stream)?;
                let down_scales_ptr = weight_arena.upload_slice(&all_dn_scales, stream)?;
                let down_qzeros_ptr = weight_arena.upload_slice(&all_dn_qzeros, stream)?;
                MoeExpertStorage::W4A16Zp {
                    gate_up_packed_ptr,
                    gate_up_scales_ptr,
                    gate_up_qzeros_ptr,
                    down_packed_ptr,
                    down_scales_ptr,
                    down_qzeros_ptr,
                    group_size,
                }
            }
            RawLinearHost::Hadamard { .. } => {
                return Err(loader_err("Hadamard routed MoE experts are not supported"));
            }
            RawLinearHost::U8B128NMajor { .. } => unreachable!(),
            RawLinearHost::DenseF16 { .. } => {
                let mut all_gu_f16 = Vec::with_capacity(num_exp * 2 * local_moe_interm * h);
                let mut all_dn_f16 = Vec::with_capacity(num_exp * h * local_moe_interm);
                for batch_start in (0..num_exp).step_by(EXP_BATCH) {
                    let batch_end = (batch_start + EXP_BATCH).min(num_exp);
                    for (rel_e, (gu, rd)) in load_expert_batch(batch_start, batch_end)?
                        .into_iter()
                        .enumerate()
                    {
                        let e = batch_start + rel_e;
                        let epfx = format!("{pfx}.mlp.experts.{e}");
                        if let RawLinearHost::DenseF16 { weight, .. } = gu {
                            all_gu_f16.extend_from_slice(&weight);
                        } else {
                            return Err(loader_err_msg(format!(
                                "inconsistent MoE expert quantization at '{epfx}' (expected DenseF16)"
                            )));
                        }
                        if let RawLinearHost::DenseF16 { weight, .. } = rd {
                            all_dn_f16.extend_from_slice(&weight);
                        } else {
                            return Err(loader_err_msg(format!(
                                "inconsistent MoE down_proj quantization at '{epfx}' (expected DenseF16)"
                            )));
                        }
                    }
                }
                let gate_up_ptr = weight_arena.upload_slice(&all_gu_f16, stream)?;
                let down_ptr = weight_arena.upload_slice(&all_dn_f16, stream)?;
                MoeExpertStorage::DenseF16 {
                    gate_up_ptr,
                    down_ptr,
                }
            }
        }
    };

    let full_shared_interm = cfg.shared_expert_intermediate_size;
    let local_shared_interm = full_shared_interm / tp_size;
    let (shared_gate_up_proj, shared_down_proj, shared_expert_gate) = if local_shared_interm > 0
        && archive
            .resolve_name(&format!("{pfx}.mlp.shared_expert.gate_proj.weight"))
            .or_else(|| archive.resolve_name(&format!("{pfx}.mlp.shared_expert.gate_proj.qweight")))
            .or_else(|| {
                archive.resolve_name(&format!("{pfx}.mlp.shared_expert.gate_proj.weight_packed"))
            })
            .is_some()
    {
        let s_gate_full = RawLinearHost::load(
            archive,
            &format!("{pfx}.mlp.shared_expert.gate_proj"),
            &cfg.quant,
        )?;
        if s_gate_full.dims() != (full_shared_interm, h) {
            return Err(loader_err_msg(format!(
                "shared_expert.gate_proj dims {:?} at '{pfx}' do not match expected ({full_shared_interm}, {h})",
                s_gate_full.dims()
            )));
        }
        let s_gate =
            s_gate_full.slice_rows(rank * local_shared_interm, (rank + 1) * local_shared_interm);
        let s_up_full = RawLinearHost::load(
            archive,
            &format!("{pfx}.mlp.shared_expert.up_proj"),
            &cfg.quant,
        )?;
        if s_up_full.dims() != (full_shared_interm, h) {
            return Err(loader_err_msg(format!(
                "shared_expert.up_proj dims {:?} at '{pfx}' do not match expected ({full_shared_interm}, {h})",
                s_up_full.dims()
            )));
        }
        let s_up =
            s_up_full.slice_rows(rank * local_shared_interm, (rank + 1) * local_shared_interm);
        let s_gu = RawLinearHost::concat_rows(&[s_gate, s_up])?.upload(
            weight_arena,
            stream,
            repack_staging,
        )?;
        let s_dn_full = RawLinearHost::load(
            archive,
            &format!("{pfx}.mlp.shared_expert.down_proj"),
            &cfg.quant,
        )?;
        if s_dn_full.dims() != (h, full_shared_interm) {
            return Err(loader_err_msg(format!(
                "shared_expert.down_proj dims {:?} at '{pfx}' do not match expected ({h}, {full_shared_interm})",
                s_dn_full.dims()
            )));
        }
        let s_dn = s_dn_full
            .slice_cols(rank * local_shared_interm, (rank + 1) * local_shared_interm)?
            .upload(weight_arena, stream, repack_staging)?;
        let s_eg = if archive
            .resolve_name(&format!("{pfx}.mlp.shared_expert_gate.weight"))
            .or_else(|| archive.resolve_name(&format!("{pfx}.mlp.shared_expert_gate.qweight")))
            .or_else(|| {
                archive.resolve_name(&format!("{pfx}.mlp.shared_expert_gate.weight_packed"))
            })
            .is_some()
        {
            let raw_eg = RawLinearHost::load(
                archive,
                &format!("{pfx}.mlp.shared_expert_gate"),
                &cfg.quant,
            )?;
            if raw_eg.dims() != (1, h) {
                return Err(loader_err_msg(format!(
                    "shared_expert_gate dims {:?} at '{pfx}' do not match expected (1, {h})",
                    raw_eg.dims()
                )));
            }
            Some(raw_eg.upload(weight_arena, stream, repack_staging)?)
        } else {
            None
        };
        (Some(s_gu), Some(s_dn), s_eg)
    } else {
        (None, None, None)
    };

    Ok(MoeLayerWeights {
        router_gate,
        experts,
        num_experts: num_exp,
        top_k,
        norm_topk_prob: cfg.norm_topk_prob,
        scoring_func,
        local_moe_interm,
        shared_gate_up_proj,
        shared_down_proj,
        shared_expert_gate,
        local_shared_interm,
    })
}

#[cfg(feature = "cuda")]
pub(crate) fn load_hc_block_weights(
    archive: &SafetensorsArchive,
    cfg: &ModelConfig,
    hc_pfx: &str,
    has_inject: bool,
    weight_arena: &mut DeviceArena,
    stream: &CudaStream,
    repack_staging: &DeviceBuffer,
) -> Result<HcBlockWeights> {
    let total_dim = cfg.hc_count * cfg.hidden_size;
    let low_dim = cfg.hc_lowrank;
    let hc_norm_key = archive
        .resolve_name(&format!("{hc_pfx}.hc_norm.weight"))
        .ok_or_else(|| loader_err_msg(format!("missing '{hc_pfx}.hc_norm.weight'")))?;
    let mut hc_norm_bits = archive
        .read_f16_bits(&hc_norm_key)
        .map_err(|e| loader_err_msg(format!("read '{hc_norm_key}': {e}")))?;
    if hc_norm_bits.len() == cfg.hidden_size && cfg.hc_count > 1 {
        hc_norm_bits = hc_norm_bits.repeat(cfg.hc_count);
    }
    if hc_norm_bits.len() != total_dim {
        return Err(loader_err_msg(format!(
            "'{hc_norm_key}' length {} != expected {total_dim} (or {})",
            hc_norm_bits.len(),
            cfg.hidden_size
        )));
    }
    let hc_norm_ptr = weight_arena.upload_slice(&hc_norm_bits, stream)?;

    let raw_down = RawLinearHost::load(
        archive,
        &format!("{hc_pfx}.input_mix_weight_down"),
        &cfg.quant,
    )?;
    if raw_down.dims() != (low_dim, total_dim) {
        return Err(loader_err_msg(format!(
            "'{hc_pfx}.input_mix_weight_down' dims {:?} != ({low_dim}, {total_dim})",
            raw_down.dims()
        )));
    }
    let input_mix_weight_down = raw_down.upload(weight_arena, stream, repack_staging)?;

    let raw_up = RawLinearHost::load(
        archive,
        &format!("{hc_pfx}.input_mix_weight_up"),
        &cfg.quant,
    )?;
    if raw_up.dims() != (total_dim, low_dim) {
        return Err(loader_err_msg(format!(
            "'{hc_pfx}.input_mix_weight_up' dims {:?} != ({total_dim}, {low_dim})",
            raw_up.dims()
        )));
    }
    let input_mix_weight_up = raw_up.upload(weight_arena, stream, repack_staging)?;

    let block_inject_weight = if has_inject {
        let raw_inj = RawLinearHost::load(
            archive,
            &format!("{hc_pfx}.block_inject_weight"),
            &cfg.quant,
        )?;
        if raw_inj.dims() != (cfg.hc_count, total_dim) {
            return Err(loader_err_msg(format!(
                "'{hc_pfx}.block_inject_weight' dims {:?} != ({}, {total_dim})",
                raw_inj.dims(),
                cfg.hc_count
            )));
        }
        Some(raw_inj.upload(weight_arena, stream, repack_staging)?)
    } else {
        None
    };

    Ok(HcBlockWeights {
        hc_norm_ptr,
        input_mix_weight_down,
        input_mix_weight_up,
        block_inject_weight,
    })
}

#[cfg(feature = "cuda")]
pub(crate) fn load_qsa_indexer_weights(
    archive: &SafetensorsArchive,
    cfg: &ModelConfig,
    pfx: &str,
    weight_arena: &mut DeviceArena,
    stream: &CudaStream,
    repack_staging: &DeviceBuffer,
) -> Result<QsaIndexerWeights> {
    let idx_pfx = format!("{pfx}.self_attn.indexer");
    let expected_n = (cfg.indexer_n_heads + cfg.indexer_kv_heads) * cfg.indexer_head_dim;
    let raw_qk = RawLinearHost::load(archive, &format!("{idx_pfx}.index_qk_proj"), &cfg.quant)?;
    if raw_qk.dims() != (expected_n, cfg.hidden_size) {
        return Err(loader_err_msg(format!(
            "'{idx_pfx}.index_qk_proj' dims {:?} != ({expected_n}, {})",
            raw_qk.dims(),
            cfg.hidden_size
        )));
    }
    let index_qk_proj = raw_qk.upload(weight_arena, stream, repack_staging)?;

    let q_ln_key = archive
        .resolve_name(&format!("{idx_pfx}.q_layernorm.weight"))
        .ok_or_else(|| loader_err_msg(format!("missing '{idx_pfx}.q_layernorm.weight'")))?;
    let q_ln_bits = archive
        .read_f16_bits(&q_ln_key)
        .map_err(|e| loader_err_msg(format!("read '{q_ln_key}': {e}")))?;
    if q_ln_bits.len() != cfg.indexer_head_dim {
        return Err(loader_err_msg(format!(
            "'{q_ln_key}' len {} != {}",
            q_ln_bits.len(),
            cfg.indexer_head_dim
        )));
    }
    let q_layernorm_ptr = weight_arena.upload_slice(&q_ln_bits, stream)?;

    let k_ln_key = archive
        .resolve_name(&format!("{idx_pfx}.k_layernorm.weight"))
        .ok_or_else(|| loader_err_msg(format!("missing '{idx_pfx}.k_layernorm.weight'")))?;
    let k_ln_bits = archive
        .read_f16_bits(&k_ln_key)
        .map_err(|e| loader_err_msg(format!("read '{k_ln_key}': {e}")))?;
    if k_ln_bits.len() != cfg.indexer_head_dim {
        return Err(loader_err_msg(format!(
            "'{k_ln_key}' len {} != {}",
            k_ln_bits.len(),
            cfg.indexer_head_dim
        )));
    }
    let k_layernorm_ptr = weight_arena.upload_slice(&k_ln_bits, stream)?;

    Ok(QsaIndexerWeights {
        index_qk_proj,
        q_layernorm_ptr,
        k_layernorm_ptr,
    })
}

#[cfg(feature = "cuda")]
pub(crate) fn load_ple_layer_weights(
    archive: &SafetensorsArchive,
    cfg: &ModelConfig,
    pfx: &str,
    weight_arena: &mut DeviceArena,
    stream: &CudaStream,
    repack_staging: &DeviceBuffer,
) -> Result<PleLayerWeights> {
    let ple_pfx = format!("{pfx}.ple");
    let total_dim = cfg.hc_count * cfg.hidden_size;
    let raw_key = RawLinearHost::load(archive, &format!("{ple_pfx}.key_proj"), &cfg.quant)?;
    if raw_key.dims() != (total_dim, cfg.ple_embed_dim) {
        return Err(loader_err_msg(format!(
            "'{ple_pfx}.key_proj' dims {:?} != ({total_dim}, {})",
            raw_key.dims(),
            cfg.ple_embed_dim
        )));
    }
    let key_proj = raw_key.upload(weight_arena, stream, repack_staging)?;

    let raw_val = RawLinearHost::load(archive, &format!("{ple_pfx}.value_proj"), &cfg.quant)?;
    if raw_val.dims() != (cfg.hidden_size, cfg.ple_embed_dim) {
        return Err(loader_err_msg(format!(
            "'{ple_pfx}.value_proj' dims {:?} != ({}, {})",
            raw_val.dims(),
            cfg.hidden_size,
            cfg.ple_embed_dim
        )));
    }
    let value_proj = raw_val.upload(weight_arena, stream, repack_staging)?;

    let load_norm = |suffix: &str, arena: &mut DeviceArena| -> Result<u64> {
        let key = archive
            .resolve_name(&format!("{ple_pfx}.{suffix}"))
            .ok_or_else(|| loader_err_msg(format!("missing '{ple_pfx}.{suffix}'")))?;
        let bits = archive
            .read_f16_bits(&key)
            .map_err(|e| loader_err_msg(format!("read '{key}': {e}")))?;
        if bits.len() != total_dim {
            return Err(loader_err_msg(format!(
                "'{key}' len {} != {total_dim}",
                bits.len()
            )));
        }
        arena.upload_slice(&bits, stream)
    };
    let norm_key_ptr = load_norm("norm_key.weight", weight_arena)?;
    let norm_query_ptr = load_norm("norm_query.weight", weight_arena)?;
    let norm_conv_ptr = load_norm("norm_conv.weight", weight_arena)?;

    let conv_key = archive
        .resolve_name(&format!("{ple_pfx}.conv1d.weight"))
        .ok_or_else(|| loader_err_msg(format!("missing '{ple_pfx}.conv1d.weight'")))?;
    let conv_bits = archive
        .read_f16_bits(&conv_key)
        .map_err(|e| loader_err_msg(format!("read '{conv_key}': {e}")))?;
    if conv_bits.len() != total_dim * cfg.ple_conv_kernel_size {
        return Err(loader_err_msg(format!(
            "'{conv_key}' len {} != {}",
            conv_bits.len(),
            total_dim * cfg.ple_conv_kernel_size
        )));
    }
    let conv1d_weight_ptr = weight_arena.upload_slice(&conv_bits, stream)?;

    let scale_key = archive
        .resolve_name(&format!(
            "{ple_pfx}.ple_embedding.ngram_embedding.weight_scale"
        ))
        .ok_or_else(|| {
            loader_err_msg(format!(
                "missing '{ple_pfx}.ple_embedding.ngram_embedding.weight_scale'"
            ))
        })?;
    let scale_vals = archive
        .read_f32(&scale_key)
        .map_err(|e| loader_err_msg(format!("read '{scale_key}': {e}")))?;
    let weight_scale = *scale_vals
        .first()
        .ok_or_else(|| loader_err_msg(format!("empty '{scale_key}'")))?;

    Ok(PleLayerWeights {
        key_proj,
        value_proj,
        norm_key_ptr,
        norm_query_ptr,
        norm_conv_ptr,
        conv1d_weight_ptr,
        weight_scale,
    })
}

#[cfg(feature = "cuda")]
pub struct DecoderLayerWeights {
    pub layer_id: usize,
    pub residual: LayerResidual,
    pub ple: Option<PleLayerWeights>,
    pub block: LayerBlockWeights,
    pub ffn: FfnBlockWeights,
}
