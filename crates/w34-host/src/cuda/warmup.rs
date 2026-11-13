use super::*;

#[cfg(feature = "cuda")]
#[path = "warmup_vision_moe.rs"]
mod warmup_vision_moe;

/// Pre-warms every kernel specialization on `device_id` and synchronizes the device.
///
/// Mandatory under `docs/contracts/execution-contract.md` and `docs/contracts/collectives.md`
/// before any `TP=2` collective barrier or `cudaStreamBeginCapture` to prevent CUDA 13
/// lazy-module-loading deadlocks.
///
/// # Safety Notes
/// Warmup launches use the local arena and dedicated fixtures below. Small shapes
/// reuse 64 KiB buffers; wider attention, GDN, sampler and MoE specializations allocate
/// separate storage. These owners remain live through the final `dev.synchronize()`.
/// Keep fixture dimensions and capacities in sync when adding a specialization.
#[cfg(feature = "cuda")]
fn warmup_norms(
    ex: &W34Exec,
    buf_a: u64,
    buf_b: u64,
    buf_c: u64,
    buf_d: u64,
    buf_ws: u64,
) -> Result<()> {
    for hs in [128i32, 3072, 5120, 6144, 10240] {
        for byte_off in [0u64, 4u64] {
            let out_had = if byte_off == 0 && hs <= 6144 {
                buf_d as *mut c_void
            } else {
                core::ptr::null_mut()
            };
            check_rc(
                // SAFETY: Warmup arena buffers are sized for this launch and remain live until `dev.synchronize()`.
                unsafe {
                    ffi::w34_gemma_rms_norm_hadamard_f16(
                        ex,
                        (buf_a + byte_off) as *const c_void,
                        (buf_b + byte_off) as *const c_void,
                        (buf_c + byte_off) as *mut c_void,
                        out_had,
                        core::ptr::null(),
                        1,
                        hs,
                        1e-6,
                    )
                },
                "warmup w34_gemma_rms_norm_hadamard_f16",
            )?;
            if hs <= 6144 {
                check_rc(
                    // SAFETY: Warmup arena buffers are sized for this launch and remain live until `dev.synchronize()`.
                    unsafe {
                        ffi::w34_fused_add_gemma_rms_norm_tap_hadamard_f16(
                            ex,
                            (buf_a + byte_off) as *const c_void,
                            (buf_b + byte_off) as *mut c_void,
                            (buf_c + byte_off) as *const c_void,
                            (buf_d + byte_off) as *mut c_void,
                            if byte_off == 0 {
                                buf_ws as *mut c_void
                            } else {
                                core::ptr::null_mut()
                            },
                            core::ptr::null_mut(),
                            core::ptr::null(),
                            1,
                            hs,
                            1e-6,
                            0,
                            0,
                        )
                    },
                    "warmup w34_fused_add_gemma_rms_norm_tap_hadamard_f16",
                )?;
            }
        }
    }
    Ok(())
}

#[cfg(feature = "cuda")]
pub fn prepare_execution_plan(device_id: i32) -> Result<()> {
    let dev = CudaDevice::set_current(device_id)?;
    let dev_props = dev.props()?;
    dev_props.validate_supported()?;
    let marlin_ws_ints = dev_props.marlin_workspace_ints() as i64;
    let stream = CudaStream::new(device_id)?;
    let cublas = CublasHandle::new(&stream)?;
    let mut arena = DeviceArena::new(device_id, 16 * 1024 * 1024)?;
    let ex = stream.exec(arena.buf.addr(), arena.capacity());

    macro_rules! warmup_ffi {
        ($call:expr, $label:expr) => {{
            // SAFETY: Warmup arena and fixture buffers are sized for this launch and remain live until `dev.synchronize()`.
            check_rc(unsafe { $call }, $label)
        }};
    }

    let buf_a = arena.alloc_aligned(64 * 1024)?;
    let buf_b = arena.alloc_aligned(64 * 1024)?;
    let buf_c = arena.alloc_aligned(64 * 1024)?;
    let buf_d = arena.alloc_aligned(64 * 1024)?;
    let buf_ws = arena.alloc_aligned((64 * 1024).max(dev_props.marlin_workspace_bytes()))?;

    // HGPTQ input transforms are independent of GEMM and must be resident before
    // any rank enters graph capture or a collective warmup barrier.
    for block_size in [16, 64] {
        for had_fn in [ffi::w34_block_hadamard_f16, ffi::w34_block_hadamard_bf16] {
            warmup_ffi!(
                had_fn(
                    &ex,
                    buf_a as *const c_void,
                    buf_b as *mut c_void,
                    2,
                    128,
                    192,
                    block_size,
                ),
                "warmup w34_block_hadamard"
            )?;
        }
        warmup_ffi!(
            ffi::w34_block_hadamard_inplace_f16(&ex, buf_b as *mut c_void, 2, 128, block_size),
            "warmup w34_block_hadamard_inplace_f16"
        )?;
    }

    warmup_ffi!(
        ffi::w34_linear_epilogue_f16(
            &ex,
            buf_a as *const c_void,
            buf_c as *const c_void,
            buf_b as *mut c_void,
            2,
            64,
            128,
        ),
        "warmup w34_linear_epilogue_f16"
    )?;

    warmup_ffi!(
        ffi::w34_midloop_damped_euler_f16(
            &ex,
            buf_c as *mut c_void,
            buf_a as *const c_void,
            buf_b as *const c_void,
            128,
            0.5,
        ),
        "warmup w34_midloop_damped_euler_f16"
    )?;

    // 1. Norms, RoPE, activations
    warmup_ffi!(
        ffi::w34_embed_lookup_f16(
            &ex,
            buf_a as *const i32,
            buf_b as *const c_void,
            buf_c as *mut c_void,
            core::ptr::null(),
            1,
            128,
            256,
        ),
        "warmup w34_embed_lookup_f16"
    )?;
    warmup_norms(&ex, buf_a, buf_b, buf_c, buf_d, buf_ws)?;
    warmup_ffi!(
        ffi::w34_rms_norm_gated_silu_f16(
            &ex,
            buf_a as *const c_void,
            buf_b as *const c_void,
            buf_c as *const c_void,
            buf_d as *mut c_void,
            core::ptr::null(),
            1,
            2,
            128,
            1e-6,
        ),
        "warmup w34_rms_norm_gated_silu_f16"
    )?;
    for gated_fn in [
        ffi::w34_rms_norm_gated_silu_strided_f16,
        ffi::w34_rms_norm_gated_sigmoid_strided_f16,
        ffi::w34_rms_norm_gated_silu_strided_hadamard_f16,
        ffi::w34_rms_norm_gated_sigmoid_strided_hadamard_f16,
    ] {
        warmup_ffi!(
            gated_fn(
                &ex,
                buf_a as *const c_void,
                buf_b as *const c_void,
                buf_c as *const c_void,
                buf_d as *mut c_void,
                core::ptr::null(),
                1,
                2,
                128,
                256,
                1e-6,
            ),
            "warmup w34_rms_norm_gated_strided_f16"
        )?;
    }
    warmup_ffi!(
        ffi::w34_fused_qk_rmsnorm_rope_gate_f16(
            &ex,
            buf_a as *const c_void,
            buf_b as *const c_void,
            buf_c as *const c_void,
            buf_c as *const c_void,
            buf_ws as *const i32,
            core::ptr::null(),
            buf_d as *mut c_void,
            buf_d as *mut c_void,
            buf_d as *mut c_void,
            1,
            1,
            2,
            1,
            256,
            64,
            11,
            11,
            10,
            10_000_000.0,
            1e-6,
        ),
        "warmup w34_fused_qk_rmsnorm_rope_gate_f16"
    )?;
    for d_warm in [128i32, 132] {
        warmup_ffi!(
            ffi::w34_silu_and_mul_f16(
                &ex,
                buf_a as *const c_void,
                buf_c as *mut c_void,
                core::ptr::null(),
                1,
                d_warm,
            ),
            "warmup w34_silu_and_mul_f16"
        )?;
    }
    warmup_ffi!(
        ffi::w34_silu_and_mul_hadamard_f16(
            &ex,
            buf_a as *const c_void,
            buf_c as *mut c_void,
            core::ptr::null(),
            1,
            128,
        ),
        "warmup w34_silu_and_mul_hadamard_f16"
    )?;
    type GateFn = unsafe extern "C" fn(
        *const ffi::W34Exec,
        *const c_void,
        *const c_void,
        *mut c_void,
        *const u8,
        i32,
        i32,
    ) -> i32;
    for (gate_fn, label) in [
        (
            ffi::w34_apply_attn_output_gate_f16 as GateFn,
            "warmup w34_apply_attn_output_gate_f16",
        ),
        (
            ffi::w34_apply_attn_output_gate_hadamard_f16 as GateFn,
            "warmup w34_apply_attn_output_gate_hadamard_f16",
        ),
    ] {
        warmup_ffi!(
            gate_fn(
                &ex,
                buf_a as *const c_void,
                buf_b as *const c_void,
                buf_c as *mut c_void,
                core::ptr::null(),
                1,
                256,
            ),
            label
        )?;
    }

    // 2. Marlin repack & GEMM + cuBLAS GEMM
    warmup_ffi!(
        ffi::w34_marlin_warmup_specializations(),
        "warmup w34_marlin_warmup_specializations"
    )?;
    warmup_ffi!(
        ffi::w34_marlin_repack_int2(&ex, buf_a as *const u32, buf_b as *mut u32, 128, 64),
        "warmup w34_marlin_repack_int2"
    )?;
    warmup_ffi!(
        ffi::w34_marlin_repack_int4(&ex, buf_a as *const u32, buf_b as *mut u32, 128, 64),
        "warmup w34_marlin_repack_int4"
    )?;
    warmup_ffi!(
        ffi::w34_marlin_repack_int8(&ex, buf_a as *const u32, buf_b as *mut u32, 128, 64),
        "warmup w34_marlin_repack_int8"
    )?;
    warmup_ffi!(
        ffi::w34_marlin_quantize_f16_u4b8(
            &ex,
            buf_a as *const c_void,
            buf_b as *mut u32,
            buf_c as *mut c_void,
            128,
            64,
            64,
        ),
        "warmup w34_marlin_quantize_f16_u4b8"
    )?;
    for m_warm in [1i64, 16, 40, 64] {
        type MarlinGemmFn = unsafe extern "C" fn(
            *const ffi::W34Exec,
            *const c_void,
            *const c_void,
            *const c_void,
            *mut c_void,
            *mut i32,
            i64,
            i64,
            i64,
            i64,
            i32,
        ) -> i32;
        for group_blocks in [8i32, 4, 2, -1] {
            for (gemm_fn, label) in [
                (
                    ffi::w34_marlin_gemm_f16u2 as MarlinGemmFn,
                    "warmup w34_marlin_gemm_f16u2",
                ),
                (
                    ffi::w34_marlin_gemm_f16u4 as MarlinGemmFn,
                    "warmup w34_marlin_gemm_f16u4",
                ),
                (
                    ffi::w34_marlin_gemm_bf16u4 as MarlinGemmFn,
                    "warmup w34_marlin_gemm_bf16u4",
                ),
                (
                    ffi::w34_marlin_gemm_f16u8 as MarlinGemmFn,
                    "warmup w34_marlin_gemm_f16u8",
                ),
            ] {
                warmup_ffi!(
                    gemm_fn(
                        &ex,
                        buf_a as *const c_void,
                        buf_b as *const c_void,
                        buf_c as *const c_void,
                        buf_d as *mut c_void,
                        buf_ws as *mut i32,
                        m_warm,
                        64,
                        128,
                        marlin_ws_ints,
                        group_blocks,
                    ),
                    label
                )?;
            }
            if group_blocks == -1 {
                continue;
            }
            warmup_ffi!(
                ffi::w34_marlin_gemm_f16u4_zp(
                    &ex,
                    buf_a as *const c_void,
                    buf_b as *const c_void,
                    buf_c as *const c_void,
                    buf_c as *const c_void,
                    buf_d as *mut c_void,
                    buf_ws as *mut i32,
                    m_warm,
                    64,
                    128,
                    marlin_ws_ints,
                    group_blocks,
                ),
                "warmup w34_marlin_gemm_f16u4_zp"
            )?;
        }
    }
    cublas.gemm_f16(buf_a, buf_b, buf_c, 1, 64, 128)?;
    // Mandatory lazy-module warmup before TP workers/capture.
    cublas.gemm_f16_bias(buf_a, buf_b, buf_d, buf_c, 1, 64, 128)?;
    cublas.gemm_bf16(buf_a, buf_b, buf_c, 1, 64, 128)?;
    for m_warm in [1i32, 4, 20] {
        warmup_ffi!(
            ffi::w34_small_n_gemm_f16(
                &ex,
                buf_a as *const c_void,
                buf_b as *const c_void,
                buf_c as *mut c_void,
                m_warm,
                64,
                128,
            ),
            "warmup w34_small_n_gemm_f16"
        )?;
        warmup_ffi!(
            ffi::w34_small_n_gemm_dual_f16(
                &ex,
                buf_a as *const c_void,
                buf_b as *const c_void,
                buf_c as *mut c_void,
                64,
                buf_b as *const c_void,
                buf_d as *mut c_void,
                1,
                m_warm,
                128,
            ),
            "warmup w34_small_n_gemm_dual_f16"
        )?;
    }
    for hs in [128i32, 3072, 5120, 6144] {
        // SAFETY: Pure query returning constant signal struct byte size.
        let sig_bytes = unsafe { ffi::w34_tp2_signal_bytes() };
        let sig_u32_len = (sig_bytes as usize).div_ceil(4);
        let mut fake_sig = vec![0u32; sig_u32_len.max(1216)];
        fake_sig[1] = 1; // pre-satisfy start[0][1] == epoch(1) for single-GPU warmup
        fake_sig[2] = 1;
        fake_sig[3] = 1;
        let sig_dev = arena.upload_slice(&fake_sig, &stream)?;
        warmup_ffi!(
            ffi::w34_tp2_all_reduce_residual_gemma_rms_norm_hadamard_f16(
                &ex,
                buf_a as *const c_void,
                core::ptr::null_mut(),
                buf_b as *mut c_void,
                buf_c as *const c_void,
                buf_d as *mut c_void,
                buf_d as *mut c_void,
                core::ptr::null_mut(),
                buf_ws as *mut c_void,
                buf_ws as *mut c_void,
                sig_dev as *mut c_void,
                sig_dev as *mut c_void,
                0,
                1,
                hs,
                1e-6,
                0,
                0,
            ),
            "warmup w34_tp2_all_reduce_residual_gemma_rms_norm_hadamard_f16"
        )?;
        let mut fake_sig4 = vec![0u32; sig_u32_len.max(1216)];
        fake_sig4[1] = 1;
        fake_sig4[2] = 1;
        fake_sig4[3] = 1;
        let sig4_dev = arena.upload_slice(&fake_sig4, &stream)?;
        warmup_ffi!(
            ffi::w34_tp4_all_reduce_residual_gemma_rms_norm_hadamard_f16(
                &ex,
                buf_a as *const c_void,
                core::ptr::null_mut(),
                buf_b as *mut c_void,
                buf_c as *const c_void,
                buf_d as *mut c_void,
                buf_d as *mut c_void,
                core::ptr::null_mut(),
                buf_ws as *mut c_void,
                buf_ws as *mut c_void,
                buf_ws as *mut c_void,
                buf_ws as *mut c_void,
                sig4_dev as *mut c_void,
                sig4_dev as *mut c_void,
                sig4_dev as *mut c_void,
                sig4_dev as *mut c_void,
                0,
                1,
                hs,
                1e-6,
                0,
                0,
            ),
            "warmup w34_tp4_all_reduce_residual_gemma_rms_norm_hadamard_f16"
        )?;
    }

    // 3. KV write & Paged Attention
    let q_start = arena.upload_slice(&[0i32, 1], &stream)?;
    let ctx_lens = arena.upload_slice(&[0i32], &stream)?;
    let seq_lens = arena.upload_slice(&[1i32], &stream)?;
    let blk_tbl = arena.upload_slice(&[0i32], &stream)?;
    let slot_map = arena.upload_slice(&[0i32], &stream)?;
    let layout = W34KvLayerLayout {
        k_base: buf_a,
        v_base: buf_b,
        k_page_stride_bytes: 16 * 256 * 2,
        k_head_stride_bytes: 16 * 256 * 2,
        k_token_stride_bytes: 256 * 2,
        v_page_stride_bytes: 16 * 256 * 2,
        v_head_stride_bytes: 16 * 256 * 2,
        v_token_stride_bytes: 256 * 2,
        namespace_id: ffi::W34_KV_NAMESPACE_TARGET_FULL_ATTN,
        layer_id: 0,
        local_kv_heads: 1,
        head_dim: 256,
        block_size: 16,
        k_format: ffi::W34_KV_FORMAT_FP16,
        v_format: ffi::W34_KV_FORMAT_FP16,
        k_transform: ffi::W34_KV_TRANSFORM_IDENTITY,
        v_transform: ffi::W34_KV_TRANSFORM_IDENTITY,
        reserved: 0,
    };
    let attn_batch = W34AttentionBatch {
        query_start_loc: q_start as *const i32,
        context_lens_before: ctx_lens as *const i32,
        seq_lens_after: seq_lens as *const i32,
        block_tables: blk_tbl as *const i32,
        write_slot_mapping: slot_map as *const i32,
        valid_token_mask: core::ptr::null(),
        valid_req_mask: core::ptr::null(),
        num_reqs: 1,
        total_tokens: 1,
        max_blocks_per_req: 1,
        max_q_len: 1,
        max_seq_len: 1,
        mask_mode: ffi::W34_ATTN_MASK_CAUSAL,
        sliding_window: 0,
        sm_scale: 0.0625,
        reserved: 0,
    };
    warmup_ffi!(
        ffi::w34_kv_cache_write(
            &ex,
            &layout,
            &attn_batch,
            buf_c as *const c_void,
            buf_c as *const c_void,
        ),
        "warmup w34_kv_cache_write"
    )?;
    warmup_ffi!(
        ffi::w34_paged_attention_f16(
            &ex,
            &layout,
            &attn_batch,
            buf_c as *const c_void,
            buf_d as *mut c_void,
            2,
        ),
        "warmup w34_paged_attention_f16"
    )?;
    // Exercise the eligible paged FA2 branch on this selected device before capture.
    // Dedicated zeroed inputs and LSE must not alias the generic arena workspace.
    let fa_q = arena.upload_slice(&vec![0u16; 17 * 2 * 256], &stream)?;
    let fa_k = arena.upload_slice(&vec![0u16; 32 * 256], &stream)?;
    let fa_v = arena.upload_slice(&vec![0u16; 32 * 256], &stream)?;
    let fa_o = arena.alloc_aligned(17 * 2 * 256 * 2)?;
    let fa_ws = arena.alloc_aligned(17 * 2 * 4)?;
    let fa_start = arena.upload_slice(&[0i32, 17], &stream)?;
    let fa_seq = arena.upload_slice(&[17i32], &stream)?;
    let fa_pages = arena.upload_slice(&[1i32, 0], &stream)?;
    let fa_layout = W34KvLayerLayout {
        k_base: fa_k,
        v_base: fa_v,
        ..layout
    };
    let fa_batch = W34AttentionBatch {
        query_start_loc: fa_start as *const i32,
        seq_lens_after: fa_seq as *const i32,
        block_tables: fa_pages as *const i32,
        total_tokens: 17,
        max_q_len: 17,
        max_seq_len: 17,
        max_blocks_per_req: 2,
        ..attn_batch
    };
    warmup_ffi!(
        ffi::w34_paged_attention_f16(
            &stream.exec(fa_ws, 17 * 2 * 4),
            &fa_layout,
            &fa_batch,
            fa_q as *const c_void,
            fa_o as *mut c_void,
            2,
        ),
        "warmup w34_paged_attention_f16 FA2"
    )?;

    // Warm 64-query-row TQ4 FlashAttention-2 prefill specialization before TP capture.
    let fa64_q = arena.upload_slice(&vec![0u16; 64 * 8 * 256], &stream)?;
    let fa64_k = arena.upload_slice(&vec![0u8; 4 * 4 * 16 * 132], &stream)?;
    let fa64_v = arena.upload_slice(&vec![0u8; 4 * 4 * 16 * 132], &stream)?;
    let fa64_o = arena.alloc_aligned(64 * 8 * 256 * 2)?;
    let fa64_start = arena.upload_slice(&[0i32, 64], &stream)?;
    let fa64_seq = arena.upload_slice(&[64i32], &stream)?;
    let fa64_pages = arena.upload_slice(&[0i32, 1, 2, 3], &stream)?;
    let fa64_layout = W34KvLayerLayout {
        k_base: fa64_k,
        v_base: fa64_v,
        k_token_stride_bytes: 132,
        v_token_stride_bytes: 132,
        k_head_stride_bytes: 16 * 132,
        v_head_stride_bytes: 16 * 132,
        k_page_stride_bytes: 4 * 16 * 132,
        v_page_stride_bytes: 4 * 16 * 132,
        local_kv_heads: 4,
        k_format: ffi::W34_KV_FORMAT_TQ4,
        v_format: ffi::W34_KV_FORMAT_TQ4,
        k_transform: ffi::W34_KV_TRANSFORM_RADEMACHER_WHT256,
        v_transform: ffi::W34_KV_TRANSFORM_RADEMACHER_WHT256,
        ..layout
    };
    let fa64_batch = W34AttentionBatch {
        query_start_loc: fa64_start as *const i32,
        seq_lens_after: fa64_seq as *const i32,
        block_tables: fa64_pages as *const i32,
        total_tokens: 64,
        max_q_len: 64,
        max_seq_len: 64,
        max_blocks_per_req: 4,
        ..attn_batch
    };
    for variant in [1, 2] {
        warmup_ffi!(
            ffi::w34_paged_attention_f16_force_kernel_for_test(
                &ex,
                &fa64_layout,
                &fa64_batch,
                fa64_q as *const c_void,
                fa64_o as *mut c_void,
                8,
                variant,
            ),
            "warmup w34_paged_attention_prefill_mma_f16"
        )?;
    }

    // Warm GQA mma.sync + Split-KV reduction specialization (head_dim = 256)
    let layout_256 = W34KvLayerLayout {
        head_dim: 256,
        k_token_stride_bytes: 512,
        v_token_stride_bytes: 512,
        k_head_stride_bytes: 16 * 512,
        v_head_stride_bytes: 16 * 512,
        k_page_stride_bytes: 16 * 512,
        v_page_stride_bytes: 16 * 512,
        ..layout
    };
    let blk_tbl_128 = arena.upload_slice(&[0i32; 128], &stream)?;
    let seq_lens_2048 = arena.upload_slice(&[2048i32], &stream)?;
    let q_start_2 = arena.upload_slice(&[0i32, 2], &stream)?;
    let slot_map_2 = arena.upload_slice(&[0i32, 1], &stream)?;
    let mut attn_batch_m2 = attn_batch;
    attn_batch_m2.query_start_loc = q_start_2 as *const i32;
    attn_batch_m2.write_slot_mapping = slot_map_2 as *const i32;
    attn_batch_m2.total_tokens = 2;
    attn_batch_m2.max_q_len = 2;
    attn_batch_m2.seq_lens_after = seq_lens_2048 as *const i32;
    attn_batch_m2.block_tables = blk_tbl_128 as *const i32;
    attn_batch_m2.max_blocks_per_req = 128;
    attn_batch_m2.max_seq_len = 2048;
    let mut attn_batch_split = attn_batch;
    attn_batch_split.block_tables = blk_tbl_128 as *const i32;
    attn_batch_split.seq_lens_after = seq_lens_2048 as *const i32;
    attn_batch_split.max_blocks_per_req = 128;
    attn_batch_split.max_seq_len = 2048;
    let split_ws = arena.alloc_aligned(512 * 1024)?;
    let ex_split = stream.exec(split_ws, 512 * 1024);
    for q_heads in [4i32, 6, 8] {
        warmup_ffi!(
            ffi::w34_paged_attention_f16(
                &ex,
                &layout_256,
                &attn_batch,
                buf_c as *const c_void,
                buf_d as *mut c_void,
                q_heads,
            ),
            "warmup w34_paged_attention_f16 gqa_vec_m1"
        )?;
        warmup_ffi!(
            ffi::w34_paged_attention_f16(
                &ex_split,
                &layout_256,
                &attn_batch_m2,
                buf_c as *const c_void,
                buf_d as *mut c_void,
                q_heads,
            ),
            "warmup w34_paged_attention_f16 gqa_mma_split_kv"
        )?;
        warmup_ffi!(
            ffi::w34_paged_attention_f16(
                &ex_split,
                &layout_256,
                &attn_batch_split,
                buf_c as *const c_void,
                buf_d as *mut c_void,
                q_heads,
            ),
            "warmup w34_paged_attention_f16 split_kv"
        )?;
    }
    let tq4_buf = arena.upload_slice(&vec![0u8; 16 * 132], &stream)?;
    let layout_256_tq4 = W34KvLayerLayout {
        k_base: tq4_buf,
        v_base: tq4_buf,
        k_token_stride_bytes: 132,
        v_token_stride_bytes: 132,
        k_head_stride_bytes: 16 * 132,
        v_head_stride_bytes: 16 * 132,
        k_page_stride_bytes: 16 * 132,
        v_page_stride_bytes: 16 * 132,
        k_format: ffi::W34_KV_FORMAT_TQ4,
        v_format: ffi::W34_KV_FORMAT_TQ4,
        k_transform: ffi::W34_KV_TRANSFORM_RADEMACHER_WHT256,
        v_transform: ffi::W34_KV_TRANSFORM_RADEMACHER_WHT256,
        ..layout_256
    };
    for q_heads in [4i32, 6, 8] {
        warmup_ffi!(
            ffi::w34_paged_attention_f16(
                &ex,
                &layout_256_tq4,
                &attn_batch,
                buf_c as *const c_void,
                buf_d as *mut c_void,
                q_heads,
            ),
            "warmup w34_paged_attention_f16 tq4 gqa_vec_m1"
        )?;
        warmup_ffi!(
            ffi::w34_paged_attention_f16(
                &ex_split,
                &layout_256_tq4,
                &attn_batch_m2,
                buf_c as *const c_void,
                buf_d as *mut c_void,
                q_heads,
            ),
            "warmup w34_paged_attention_f16 tq4 gqa_mma_split_kv"
        )?;
    }

    let bf16_layout = W34KvLayerLayout {
        namespace_id: ffi::W34_KV_NAMESPACE_DFLASH_DRAFT_CTX,
        k_format: ffi::W34_KV_FORMAT_BF16,
        v_format: ffi::W34_KV_FORMAT_BF16,
        ..layout
    };
    warmup_ffi!(
        ffi::w34_kv_cache_write_bf16(
            &ex,
            &bf16_layout,
            &attn_batch,
            buf_c as *const c_void,
            buf_c as *const c_void,
        ),
        "warmup w34_kv_cache_write_bf16"
    )?;
    warmup_ffi!(
        ffi::w34_paged_attention_bf16(
            &ex,
            &bf16_layout,
            &attn_batch,
            buf_c as *const c_void,
            buf_d as *mut c_void,
            2,
        ),
        "warmup w34_paged_attention_bf16 hd128"
    )?;
    let bf16_layout_256 = W34KvLayerLayout {
        namespace_id: ffi::W34_KV_NAMESPACE_DFLASH_DRAFT_CTX,
        k_format: ffi::W34_KV_FORMAT_BF16,
        v_format: ffi::W34_KV_FORMAT_BF16,
        ..layout_256
    };
    warmup_ffi!(
        ffi::w34_paged_attention_bf16(
            &ex,
            &bf16_layout_256,
            &attn_batch,
            buf_c as *const c_void,
            buf_d as *mut c_void,
            2,
        ),
        "warmup w34_paged_attention_bf16 hd256"
    )?;
    warmup_ffi!(
        ffi::w34_paged_attention_bf16(
            &ex,
            &bf16_layout,
            &attn_batch,
            (buf_c + 4) as *const c_void,
            buf_d as *mut c_void,
            2,
        ),
        "warmup w34_paged_attention_bf16 fallback"
    )?;

    // 4. GDN conv1d & recurrent
    let state_in = arena.upload_slice(&[0i32], &stream)?;
    let state_out = arena.upload_slice(&[0i32], &stream)?;
    let has_init = arena.upload_slice(&[0u8], &stream)?;
    let gdn_batch = W34GdnBatch {
        query_start_loc: q_start as *const i32,
        state_in_indices: state_in as *const i32,
        state_out_indices: state_out as *const i32,
        spec_state_indices: core::ptr::null(),
        checkpoint_dst_indices: core::ptr::null(),
        checkpoint_token_offsets: core::ptr::null(),
        has_initial_state: has_init as *const u8,
        valid_req_mask: core::ptr::null(),
        num_reqs: 1,
        total_tokens: 1,
        max_verify_tokens: 0,
        max_checkpoints_per_req: 0,
    };
    warmup_ffi!(
        ffi::w34_gdn_causal_conv1d_f16(
            &ex,
            &gdn_batch,
            buf_a as *const c_void,
            buf_b as *const c_void,
            core::ptr::null(),
            buf_c as *mut c_void,
            buf_d as *mut c_void,
            384,
            4,
            1,
            1,
        ),
        "warmup w34_gdn_causal_conv1d_f16"
    )?;
    warmup_ffi!(
        ffi::w34_gdn_causal_conv1d_strided_f16(
            &ex,
            &gdn_batch,
            buf_a as *const c_void,
            buf_b as *const c_void,
            core::ptr::null(),
            buf_c as *mut c_void,
            buf_d as *mut c_void,
            384,
            384,
            4,
            1,
            1,
        ),
        "warmup w34_gdn_causal_conv1d_strided_f16"
    )?;
    warmup_ffi!(
        ffi::w34_gdn_recurrent_f16(
            &ex,
            &gdn_batch,
            buf_d as *const c_void,
            core::ptr::null(),
            core::ptr::null(),
            core::ptr::null(),
            buf_a as *const c_void,
            buf_a as *const c_void,
            buf_b as *const f32,
            buf_b as *const f32,
            buf_ws as *mut f32,
            buf_c as *mut c_void,
            1,
            128,
            1,
            128,
            1,
        ),
        "warmup w34_gdn_recurrent_f16 decode"
    )?;
    let gdn_ws_wide = arena.alloc_aligned(4 * 1024 * 1024)?;
    warmup_ffi!(
        ffi::w34_gdn_recurrent_f16(
            &ex,
            &gdn_batch,
            buf_d as *const c_void,
            core::ptr::null(),
            core::ptr::null(),
            core::ptr::null(),
            buf_a as *const c_void,
            buf_a as *const c_void,
            buf_b as *const f32,
            buf_b as *const f32,
            gdn_ws_wide as *mut f32,
            buf_c as *mut c_void,
            1,
            128,
            96,
            32,
            1,
        ),
        "warmup w34_gdn_recurrent_f16 decode wide"
    )?;
    let q_start_prefill = arena.upload_slice(&[0i32, 3], &stream)?;
    let gdn_batch_prefill = W34GdnBatch {
        query_start_loc: q_start_prefill as *const i32,
        total_tokens: 3,
        ..gdn_batch
    };
    let sms = dev_props.multi_processor_count.max(1);
    for num_v_heads in [
        1i32,
        (sms / 5).clamp(1, 48),
        (sms / 4).clamp(1, 48),
        (sms / 3).clamp(1, 48),
        16,
        24,
        32,
    ] {
        warmup_ffi!(
            ffi::w34_gdn_recurrent_strided_f16(
                &ex,
                &gdn_batch_prefill,
                gdn_ws_wide as *const c_void,
                core::ptr::null(),
                core::ptr::null(),
                core::ptr::null(),
                buf_a as *const c_void,
                buf_a as *const c_void,
                buf_b as *const f32,
                buf_b as *const f32,
                gdn_ws_wide as *mut f32,
                buf_c as *mut c_void,
                1,
                128,
                num_v_heads,
                128,
                num_v_heads,
                1,
            ),
            "warmup w34_gdn_recurrent_strided_f16 prefill"
        )?;
    }
    let q_start_wy = arena.upload_slice(&[0i32, 256], &stream)?;
    let gdn_batch_wy = W34GdnBatch {
        query_start_loc: q_start_wy as *const i32,
        total_tokens: 256,
        ..gdn_batch
    };
    warmup_ffi!(
        ffi::w34_gdn_causal_conv1d_f16(
            &ex,
            &gdn_batch_wy,
            buf_a as *const c_void,
            buf_b as *const c_void,
            core::ptr::null(),
            buf_c as *mut c_void,
            buf_d as *mut c_void,
            8,
            4,
            1,
            1,
        ),
        "warmup w34_gdn_causal_conv1d_f16 paired prefill"
    )?;
    let ex_wy_split = stream.exec(gdn_ws_wide + 1024 * 1024, 1024 * 1024);
    for ex_wy in [&ex, &ex_wy_split] {
        warmup_ffi!(
            ffi::w34_gdn_recurrent_f16(
                ex_wy,
                &gdn_batch_wy,
                gdn_ws_wide as *const c_void,
                core::ptr::null(),
                core::ptr::null(),
                core::ptr::null(),
                gdn_ws_wide as *const c_void,
                gdn_ws_wide as *const c_void,
                buf_b as *const f32,
                buf_b as *const f32,
                buf_ws as *mut f32,
                (gdn_ws_wide + 512 * 1024) as *mut c_void,
                1,
                128,
                1,
                128,
                1,
            ),
            "warmup w34_gdn_recurrent_f16 wy prefill"
        )?;
    }
    for (hv_warm, dv_warm) in [(96i32, 32i32), (24, 128), (32, 128)] {
        warmup_ffi!(
            ffi::w34_gdn_recurrent_f16(
                &ex,
                &gdn_batch_prefill,
                buf_d as *const c_void,
                core::ptr::null(),
                core::ptr::null(),
                core::ptr::null(),
                buf_a as *const c_void,
                buf_a as *const c_void,
                buf_b as *const f32,
                buf_b as *const f32,
                gdn_ws_wide as *mut f32,
                buf_c as *mut c_void,
                1,
                128,
                hv_warm,
                dv_warm,
                1,
            ),
            "warmup w34_gdn_recurrent_f16 prefill wide"
        )?;
    }

    // 5. Sampler & MTP / DFlash2 kernels
    warmup_ffi!(ffi::w34_tp2_warmup(), "warmup op_collectives module")?;
    warmup_ffi!(
        ffi::w34_sampler_greedy_top5_f16(
            &ex,
            buf_a as *const c_void,
            buf_b as *mut i32,
            buf_c as *mut i32,
            buf_d as *mut f32,
            1,
            256,
            0,
        ),
        "warmup w34_sampler_greedy_top5_f16"
    )?;
    warmup_ffi!(
        ffi::w34_sampler_greedy_top5_f16(
            &ex,
            buf_a as *const c_void,
            buf_b as *mut i32,
            core::ptr::null_mut(),
            buf_d as *mut f32,
            1,
            4096,
            0,
        ),
        "warmup w34_sampler_greedy_top5_f16 split_top1"
    )?;
    let sampler_cfg = ffi::W34SamplerRowCfg {
        temperature: 1.0,
        top_k: 8,
        top_p: 0.9,
        min_p: 0.05,
        repetition_penalty: 1.1,
        frequency_penalty: 0.1,
        presence_penalty: 0.1,
        uniform_sample: 0.5,
        recovery_uniform: 0.5,
        draft_token_id: -1,
        has_bitmask: 0,
        row_bitmask_word_offset: 0,
        sparse_offset: 0,
        sparse_count: 1,
        top_logprobs_k: 5,
        use_token_id_cdf: 0,
    };
    let sampler_sparse = ffi::W34SamplerSparseEntry {
        token_id: 0,
        output_count: 1,
        logit_bias: 0.0,
        in_prompt: 1,
        banned: 0,
    };
    let sampler_cfg_dev = arena.upload_slice(&[sampler_cfg], &stream)?;
    let sampler_sparse_dev = arena.upload_slice(&[sampler_sparse], &stream)?;
    let sampler_out_dev = arena.alloc_aligned(core::mem::size_of::<ffi::W34SamplerRowOut>())?;
    let sampler_ws = arena.alloc_aligned(64 * 1024)?;
    let ex_sampler = stream.exec(sampler_ws, 64 * 1024);
    warmup_ffi!(
        ffi::w34_sampler_step_f16(
            &ex_sampler,
            buf_a as *const c_void,
            sampler_cfg_dev as *const ffi::W34SamplerRowCfg,
            core::ptr::null(),
            sampler_sparse_dev as *const ffi::W34SamplerSparseEntry,
            sampler_out_dev as *mut ffi::W34SamplerRowOut,
            1,
            256,
            1,
        ),
        "warmup w34_sampler_step_f16"
    )?;
    let greedy_sampler_cfg = ffi::W34SamplerRowCfg {
        temperature: 0.0,
        top_k: 1,
        sparse_count: 1,
        top_logprobs_k: 0,
        ..sampler_cfg
    };
    let greedy_sampler_cfg_dev = arena.upload_slice(&[greedy_sampler_cfg], &stream)?;
    warmup_ffi!(
        ffi::w34_sampler_step_fast_greedy_f16(
            &ex_sampler,
            buf_a as *const c_void,
            greedy_sampler_cfg_dev as *const ffi::W34SamplerRowCfg,
            sampler_sparse_dev as *const ffi::W34SamplerSparseEntry,
            sampler_out_dev as *mut ffi::W34SamplerRowOut,
            1,
            1024,
        ),
        "warmup w34_sampler_step_fast_greedy_f16"
    )?;
    let plain_greedy_cfg = ffi::W34SamplerRowCfg {
        sparse_count: 0,
        ..greedy_sampler_cfg
    };
    let plain_greedy_cfg_dev = arena.upload_slice(&[plain_greedy_cfg], &stream)?;
    let masked_greedy_cfg = ffi::W34SamplerRowCfg {
        has_bitmask: 1,
        row_bitmask_word_offset: 0,
        sparse_count: 0,
        top_logprobs_k: -1,
        ..greedy_sampler_cfg
    };
    let masked_greedy_cfg_dev = arena.upload_slice(&[masked_greedy_cfg], &stream)?;
    let masked_words_dev = arena.upload_slice(&[u32::MAX; 32], &stream)?;
    warmup_ffi!(
        ffi::w34_sampler_step_fast_greedy_masked_f16(
            &ex_sampler,
            buf_a as *const c_void,
            masked_greedy_cfg_dev as *const ffi::W34SamplerRowCfg,
            masked_words_dev as *const u32,
            core::ptr::null(),
            sampler_out_dev as *mut ffi::W34SamplerRowOut,
            1,
            1024,
        ),
        "warmup w34_sampler_step_fast_greedy_masked_f16"
    )?;
    warmup_ffi!(
        ffi::w34_sampler_shard_fast_greedy_f16(
            &ex_sampler,
            buf_a as *const c_void,
            masked_greedy_cfg_dev as *const ffi::W34SamplerRowCfg,
            masked_words_dev as *const u32,
            core::ptr::null(),
            buf_d as *mut f32,
            buf_b as *mut i32,
            1,
            1024,
            0,
        ),
        "warmup w34_sampler_shard_fast_greedy_f16"
    )?;
    warmup_ffi!(
        ffi::w34_sampler_step_f16(
            &ex_sampler,
            buf_a as *const c_void,
            plain_greedy_cfg_dev as *const ffi::W34SamplerRowCfg,
            core::ptr::null(),
            core::ptr::null(),
            sampler_out_dev as *mut ffi::W34SamplerRowOut,
            1,
            1024,
            0,
        ),
        "warmup w34_sampler_step_f16 fast_greedy"
    )?;
    warmup_ffi!(
        ffi::w34_sampler_scatter_ids_to_tokens(
            &ex_sampler,
            sampler_out_dev as *const ffi::W34SamplerRowOut,
            core::ptr::null(),
            buf_b as *mut i32,
            1,
        ),
        "warmup w34_sampler_scatter_ids_to_tokens"
    )?;
    warmup_ffi!(
        ffi::w34_concat_cols_2_f16(
            &ex,
            buf_a as *const c_void,
            buf_b as *const c_void,
            buf_c as *mut c_void,
            1,
            128,
            128,
        ),
        "warmup w34_concat_cols_2_f16"
    )?;
    warmup_ffi!(
        ffi::w34_tap_add_f16_to_bf16(
            &ex,
            buf_a as *const c_void,
            buf_b as *const c_void,
            buf_c as *mut c_void,
            1,
            128,
            640,
            0,
        ),
        "warmup w34_tap_add_f16_to_bf16"
    )?;
    // Selector IDs are indices, not the FP32 GDN state previously in buf_ws.
    let candidate_ids = arena.upload_slice(&[0i32; 16], &stream)?;
    warmup_ffi!(
        ffi::w34_dflash_qk_norm_rope_kv_write_bf16_f32_table(
            &ex,
            buf_a as *mut c_void,
            buf_d as *mut c_void,
            candidate_ids as *const i32,
            buf_b as *const c_void,
            buf_b as *const c_void,
            buf_c as *const c_void,
            &bf16_layout,
            slot_map as *const i32,
            1,
            1,
            1,
            256,
            1,
            1e-6,
        ),
        "warmup BF16 QK/RoPE + KV write"
    )?;
    warmup_ffi!(
        ffi::w34_dflash_select_greedy_bf16(
            &ex,
            candidate_ids as *const i32,
            buf_c as *const f32,
            buf_a as *const c_void,
            buf_a as *const c_void,
            buf_b as *const c_void,
            buf_d as *mut i32,
            0,
            1,
            16,
            256,
            1,
        ),
        "warmup dflash_select_greedy_bf16"
    )?;
    warmup_ffi!(
        ffi::w34_embed_lookup_f16_to_bf16(
            &ex,
            candidate_ids as *const i32,
            buf_a as *const c_void,
            buf_b as *mut c_void,
            1,
            128,
            1,
        ),
        "warmup embed_f16_to_bf16"
    )?;
    warmup_ffi!(
        ffi::w34_cast_bf16_to_f16(&ex, buf_a as *const c_void, buf_b as *mut c_void, 128),
        "warmup cast_bf16_to_f16"
    )?;
    warmup_ffi!(
        ffi::w34_dflash_edges_bf16(
            &ex,
            candidate_ids as *const i32,
            buf_c as *const f32,
            buf_a as *const c_void,
            buf_a as *const c_void,
            buf_b as *const c_void,
            buf_d as *mut f32,
            0,
            1,
            16,
            256,
            1,
        ),
        "warmup dflash_edges_bf16"
    )?;
    warmup_ffi!(
        ffi::w34_topk_rows_f16(
            &ex,
            buf_a as *const c_void,
            buf_b as *mut i32,
            buf_c as *mut f32,
            1,
            256,
            0,
            16,
        ),
        "warmup w34_topk_rows_f16"
    )?;
    warmup_ffi!(
        ffi::w34_topk_rows_split_f16(
            &ex,
            buf_a as *const c_void,
            buf_ws as *mut i32,
            (buf_ws + 4096) as *mut f32,
            buf_b as *mut i32,
            buf_c as *mut f32,
            1,
            2048,
            0,
            16,
            16,
        ),
        "warmup w34_topk_rows_split_f16"
    )?;
    let zero_i32 = arena.upload_slice(&[0i32; 16], &stream)?;
    warmup_ffi!(
        ffi::w34_dflash_select_greedy_bf16(
            &ex,
            zero_i32 as *const i32,
            buf_c as *const f32,
            buf_a as *const c_void,
            buf_b as *const c_void,
            buf_b as *const c_void,
            buf_d as *mut i32,
            0,
            1,
            16,
            256,
            1,
        ),
        "warmup w34_dflash_select_greedy_bf16"
    )?;
    warmup_ffi!(
        ffi::w34_dflash_select_greedy_dev_anchor_bf16(
            &ex,
            zero_i32 as *const i32,
            buf_c as *const f32,
            buf_a as *const c_void,
            buf_b as *const c_void,
            buf_b as *const c_void,
            buf_d as *mut i32,
            zero_i32 as *const i32,
            1,
            16,
            256,
            1,
        ),
        "warmup w34_dflash_select_greedy_dev_anchor_bf16"
    )?;

    warmup_ffi!(
        ffi::w34_rms_norm_bf16(
            &ex,
            buf_a as *const c_void,
            buf_b as *mut c_void,
            buf_c as *const c_void,
            buf_d as *mut c_void,
            1,
            128,
            1e-6,
        ),
        "warmup rms_norm_bf16"
    )?;
    for hs_bf16 in (256..=3072).step_by(256) {
        warmup_ffi!(
            ffi::w34_rms_norm_bf16(
                &ex,
                buf_a as *const c_void,
                buf_b as *mut c_void,
                buf_c as *const c_void,
                buf_d as *mut c_void,
                1,
                hs_bf16,
                1e-6,
            ),
            "warmup rms_norm_bf16 vec8"
        )?;
    }
    for (ne, tk) in [(256i32, 8i32), (512, 10), (128, 4), (384, 8), (768, 8)] {
        warmup_ffi!(
            ffi::w34_moe_topk_router_f16(
                &ex,
                buf_a as *const c_void,
                buf_b as *mut f32,
                buf_c as *mut i32,
                1,
                ne,
                tk,
                1,
                0,
            ),
            "warmup moe_topk_router_f16"
        )?;
    }
    warmup_ffi!(
        ffi::w34_silu_and_mul_bf16(&ex, buf_a as *const c_void, buf_b as *mut c_void, 1, 128),
        "warmup silu_and_mul_bf16"
    )?;
    warmup_ffi!(
        ffi::w34_dflash_conv_bf16(
            &ex,
            buf_a as *const c_void,
            buf_b as *const c_void,
            buf_c as *const c_void,
            buf_d as *mut c_void,
            1,
            128,
            8,
            64,
            3,
            0,
        ),
        "warmup dflash_conv_bf16"
    )?;
    warmup_ffi!(
        ffi::w34_dflash_qk_norm_rope_bf16(
            &ex,
            buf_a as *mut c_void,
            slot_map as *const i32,
            buf_b as *const c_void,
            buf_b as *const c_void,
            buf_c as *const c_void,
            1,
            2,
            1,
            128,
            1,
            1e-6,
        ),
        "warmup dflash_qk_norm_rope_bf16"
    )?;
    warmup_ffi!(
        ffi::w34_copy_cols_bf16(
            &ex,
            buf_a as *const c_void,
            buf_b as *mut c_void,
            1,
            256,
            128,
            128,
            0,
            128,
        ),
        "warmup copy_cols_bf16"
    )?;

    // Rejection kernels consume explicit probabilities/draws, not raw scratch.
    let reject_ids = arena.upload_slice(&[0i32; 2], &stream)?;
    let reject_probs = arena.upload_slice(&[1.0f32; 2], &stream)?;
    let reject_draws = arena.upload_slice(&[0.5f64], &stream)?;
    warmup_ffi!(
        ffi::w34_rejection_greedy(
            &ex,
            reject_ids as *const i32,
            reject_ids as *const i32,
            core::ptr::null(),
            buf_a as *mut i32,
            buf_b as *mut i32,
            buf_c as *mut i32,
            1,
            1,
        ),
        "warmup w34_rejection_greedy"
    )?;
    warmup_ffi!(
        ffi::w34_rejection_stochastic(
            &ex,
            reject_ids as *const i32,
            reject_probs as *const f32,
            reject_probs as *const f32,
            reject_draws as *const f64,
            reject_draws as *const f64,
            core::ptr::null(),
            buf_a as *mut i32,
            buf_b as *mut i32,
            buf_c as *mut i32,
            1,
            1,
            1,
        ),
        "warmup w34_rejection_stochastic"
    )?;
    warmup_ffi!(
        ffi::w34_lora_shrink_expand_f16(
            &ex,
            buf_a as *const c_void,
            buf_b as *const c_void,
            buf_c as *const c_void,
            reject_ids as *const i32,
            buf_ws as *mut f32,
            buf_d as *mut c_void,
            1,
            64,
            64,
            4,
            1,
            0,
            1,
        ),
        "warmup w34_lora_shrink_expand_f16 (fused)"
    )?;
    warmup_ffi!(
        ffi::w34_lora_shrink_expand_f16(
            &ex,
            buf_a as *const c_void,
            buf_b as *const c_void,
            buf_c as *const c_void,
            reject_ids as *const i32,
            buf_ws as *mut f32,
            buf_d as *mut c_void,
            1,
            64,
            64,
            32,
            1,
            0,
            1,
        ),
        "warmup w34_lora_shrink_expand_f16 (split)"
    )?;
    warmup_vision_moe::warmup_vision_moe_and_qwen4(
        &mut arena,
        &stream,
        &ex,
        buf_a,
        buf_b,
        buf_c,
        buf_d,
        buf_ws,
        zero_i32,
        reject_ids,
        reject_probs,
        &attn_batch,
        &layout_256,
        &gdn_batch,
    )?;

    dev.synchronize()
}
