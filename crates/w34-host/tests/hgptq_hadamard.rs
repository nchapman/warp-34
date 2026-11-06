#![cfg(feature = "cuda")]

use std::ffi::c_void;
use w34_host::cuda::{
    CudaDevice, CudaEvent, CudaGraph, CudaStream, DeviceBuffer, Tp2Collectives, f16_to_f32,
    f32_to_f16,
};
use w34_kernels::ffi;

static GPU_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

fn upload(values: &[u16], stream: &CudaStream) -> DeviceBuffer {
    let buffer = DeviceBuffer::alloc(0, std::mem::size_of_val(values)).unwrap();
    buffer.copy_from_host(values, stream).unwrap();
    buffer
}

// Independent dense Sylvester matrix, not the production butterfly algorithm.
fn reference(input: &[u16], k: usize, padded_k: usize, block: usize) -> Vec<u16> {
    let mut out = vec![0; input.len() / k * padded_k];
    for (row, x) in input.chunks_exact(k).enumerate() {
        for base in (0..k).step_by(block) {
            for i in 0..block {
                let sum: f64 = (0..block)
                    .map(|j| {
                        let sign = if (i & j).count_ones().is_multiple_of(2) {
                            1.0
                        } else {
                            -1.0
                        };
                        f64::from(f16_to_f32(x[base + j])) * sign
                    })
                    .sum();
                out[row * padded_k + base + i] = f32_to_f16((sum / (block as f64).sqrt()) as f32);
            }
        }
    }
    out
}

fn launch(
    stream: &CudaStream,
    src: &DeviceBuffer,
    dst: &DeviceBuffer,
    rows: usize,
    k: usize,
    padded_k: usize,
    block: usize,
) {
    assert!(src.len_bytes >= rows * k * 2);
    assert!(dst.len_bytes >= rows * padded_k * 2);
    // SAFETY: disjoint live device buffers have the asserted extents; test widths
    // are block-aligned. The stream and its device remain live through completion.
    let rc = unsafe {
        ffi::w34_block_hadamard_f16(
            &stream.exec(0, 0),
            src.addr() as *const c_void,
            dst.addr() as *mut c_void,
            rows as i32,
            k as i32,
            padded_k as i32,
            block as i32,
        )
    };
    assert_eq!(rc, 0);
}

#[test]
fn block_hadamard_matches_dense_sylvester_preserves_input_and_zeros_padding() {
    let _guard = GPU_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    let _device = CudaDevice::set_current(0).unwrap();
    let stream = CudaStream::new(0).unwrap();
    for (block, k, padded_k, rows) in [
        (16, 16, 16, 1),
        (64, 64, 64, 1),
        (16, 48, 80, 3),
        (64, 192, 320, 3),
        (16, 4304, 4352, 17),
        (64, 2048, 2048, 129),
    ] {
        let input: Vec<u16> = (0..rows * k)
            .map(|i| f32_to_f16(((i * 73 + i / k * 11) % 509) as f32 / 32.0 - 8.0))
            .collect();
        let src = upload(&input, &stream);
        // Canary tail catches writes beyond the logical padded output allocation.
        let dst = upload(&vec![0x7bff; rows * padded_k + 64], &stream);
        launch(&stream, &src, &dst, rows, k, padded_k, block);
        let mut got = vec![0; rows * padded_k + 64];
        dst.copy_to_host(&mut got, &stream).unwrap();
        assert_eq!(
            &got[..rows * padded_k],
            reference(&input, k, padded_k, block)
        );
        assert_eq!(&got[rows * padded_k..], &[0x7bff; 64]);
        let mut unchanged = vec![0; input.len()];
        src.copy_to_host(&mut unchanged, &stream).unwrap();
        assert_eq!(unchanged, input);
    }
}

#[test]
fn block_hadamard_basis_and_large_intermediates_match_reference() {
    let _guard = GPU_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    let _device = CudaDevice::set_current(0).unwrap();
    let stream = CudaStream::new(0).unwrap();
    for block in [16, 64] {
        let rows = block + 3;
        let mut input = vec![0; rows * block];
        for i in 0..block {
            input[i * block + i] = f32_to_f16(1.0);
        }
        // Unnormalized partial sums overflow FP16, but final output is finite.
        input[block * block..(block + 1) * block].fill(f32_to_f16(8000.0));
        input[(block + 1) * block..(block + 2) * block].fill(1); // min subnormal
        let src = upload(&input, &stream);
        let dst = upload(&vec![0; input.len()], &stream);
        launch(&stream, &src, &dst, rows, block, block, block);
        let mut got = vec![0; input.len()];
        dst.copy_to_host(&mut got, &stream).unwrap();
        assert_eq!(got, reference(&input, block, block, block));
    }
}

#[test]
fn block_hadamard_graph_replay_reads_updated_inputs() {
    let _guard = GPU_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    let _device = CudaDevice::set_current(0).unwrap();
    let stream = CudaStream::new(0).unwrap();
    for (block, k, padded_k) in [(16, 4304, 4352), (64, 2048, 2048)] {
        let src = upload(&vec![0; 3 * k], &stream);
        let dst = upload(&vec![0; 3 * padded_k], &stream);
        launch(&stream, &src, &dst, 3, k, padded_k, block);
        stream.synchronize().unwrap();
        let graph = CudaGraph::capture(&stream, || {
            launch(&stream, &src, &dst, 3, k, padded_k, block);
            Ok(())
        })
        .unwrap();
        for seed in 1..=3 {
            let input: Vec<u16> = (0..3 * k)
                .map(|i| f32_to_f16(((i * seed) % 113) as f32 / 16.0))
                .collect();
            src.copy_from_host(&input, &stream).unwrap();
            graph.launch(&stream).unwrap();
            let mut got = vec![0; 3 * padded_k];
            dst.copy_to_host(&mut got, &stream).unwrap();
            assert_eq!(got, reference(&input, k, padded_k, block));
        }
    }
}

#[test]
fn block_hadamard_rejects_invalid_geometry_and_aliasing() {
    let _guard = GPU_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    let _device = CudaDevice::set_current(0).unwrap();
    let stream = CudaStream::new(0).unwrap();
    let src = upload(&[0; 256], &stream);
    let dst = upload(&[0; 256], &stream);
    let ex = stream.exec(0, 0);
    for (rows, k, padded_k, block) in [
        (-1, 64, 64, 64),
        (1, 0, 64, 64),
        (1, 64, 32, 64),
        (1, 64, 64, 32),
        (1, 63, 64, 64),
        (1, 64, 65, 64),
    ] {
        assert_ne!(
            // SAFETY: invalid geometry must be rejected without touching valid buffers.
            unsafe {
                ffi::w34_block_hadamard_f16(
                    &ex,
                    src.addr() as *const c_void,
                    dst.addr() as *mut c_void,
                    rows,
                    k,
                    padded_k,
                    block,
                )
            },
            0
        );
    }
    for output in [src.addr(), src.addr() + 2, src.addr() - 2] {
        assert_ne!(
            // SAFETY: overlapping ranges must be rejected before dereferencing.
            unsafe {
                ffi::w34_block_hadamard_f16(
                    &ex,
                    src.addr() as *const c_void,
                    output as *mut c_void,
                    1,
                    64,
                    64,
                    64,
                )
            },
            0
        );
    }
    assert_eq!(
        // SAFETY: an empty batch permits null data pointers and never launches.
        unsafe {
            ffi::w34_block_hadamard_f16(&ex, std::ptr::null(), std::ptr::null_mut(), 0, 64, 64, 64)
        },
        0
    );
}

fn verify_gemma_norms_hadamard64(
    stream: &CudaStream,
    ex: &w34_kernels::ffi::W34Exec,
    hs: usize,
    rows: usize,
) {
    let x_host: Vec<u16> = (0..rows * hs)
        .map(|i| f32_to_f16(((i * 37 + 19) % 251) as f32 / 64.0 - 2.0))
        .collect();
    let res_host: Vec<u16> = (0..rows * hs)
        .map(|i| f32_to_f16(((i * 53 + 7) % 199) as f32 / 64.0 - 1.5))
        .collect();
    let w_host: Vec<u16> = (0..hs)
        .map(|i| f32_to_f16(((i * 17 + 3) % 97) as f32 / 128.0 - 0.25))
        .collect();
    let d_x = upload(&x_host, stream);
    let d_w = upload(&w_host, stream);
    let d_ref_norm = upload(&vec![0; rows * hs], stream);
    let d_ref_had = upload(&vec![0; rows * hs], stream);
    let d_fused_norm = upload(&vec![0; rows * hs], stream);
    let d_fused_had = upload(&vec![0; rows * hs], stream);
    let d_fused_had_only = upload(&vec![0; rows * hs], stream);

    // SAFETY: test device buffers are sized for [rows, hs].
    unsafe {
        assert_eq!(
            ffi::w34_gemma_rms_norm_f16(
                ex,
                d_x.addr() as *const c_void,
                d_w.addr() as *const c_void,
                d_ref_norm.addr() as *mut c_void,
                std::ptr::null(),
                rows as i32,
                hs as i32,
                1e-6,
            ),
            0
        );
        assert_eq!(
            ffi::w34_block_hadamard_f16(
                ex,
                d_ref_norm.addr() as *const c_void,
                d_ref_had.addr() as *mut c_void,
                rows as i32,
                hs as i32,
                hs as i32,
                64,
            ),
            0
        );
        assert_eq!(
            ffi::w34_gemma_rms_norm_hadamard_f16(
                ex,
                d_x.addr() as *const c_void,
                d_w.addr() as *const c_void,
                d_fused_norm.addr() as *mut c_void,
                d_fused_had.addr() as *mut c_void,
                std::ptr::null(),
                rows as i32,
                hs as i32,
                1e-6,
            ),
            0
        );
        assert_eq!(
            ffi::w34_gemma_rms_norm_hadamard_f16(
                ex,
                d_x.addr() as *const c_void,
                d_w.addr() as *const c_void,
                std::ptr::null_mut(),
                d_fused_had_only.addr() as *mut c_void,
                std::ptr::null(),
                rows as i32,
                hs as i32,
                1e-6,
            ),
            0
        );
    }
    let mut ref_norm = vec![0u16; rows * hs];
    let mut ref_had = vec![0u16; rows * hs];
    let mut got_norm = vec![0u16; rows * hs];
    let mut got_had = vec![0u16; rows * hs];
    let mut got_had_only = vec![0u16; rows * hs];
    d_ref_norm.copy_to_host(&mut ref_norm, stream).unwrap();
    d_ref_had.copy_to_host(&mut ref_had, stream).unwrap();
    d_fused_norm.copy_to_host(&mut got_norm, stream).unwrap();
    d_fused_had.copy_to_host(&mut got_had, stream).unwrap();
    d_fused_had_only
        .copy_to_host(&mut got_had_only, stream)
        .unwrap();
    assert_eq!(got_norm, ref_norm, "GemmaRMSNorm unrotated hs={hs}");
    assert_eq!(got_had, ref_had, "GemmaRMSNorm dual H64 hs={hs}");
    assert_eq!(got_had_only, ref_had, "GemmaRMSNorm had-only H64 hs={hs}");

    let d_res_ref = upload(&res_host, stream);
    let d_res_fused = upload(&res_host, stream);
    // SAFETY: test device buffers are sized for [rows, hs].
    unsafe {
        assert_eq!(
            ffi::w34_fused_add_gemma_rms_norm_f16(
                ex,
                d_x.addr() as *const c_void,
                d_res_ref.addr() as *mut c_void,
                d_w.addr() as *const c_void,
                d_ref_norm.addr() as *mut c_void,
                std::ptr::null(),
                rows as i32,
                hs as i32,
                1e-6,
            ),
            0
        );
        assert_eq!(
            ffi::w34_block_hadamard_f16(
                ex,
                d_ref_norm.addr() as *const c_void,
                d_ref_had.addr() as *mut c_void,
                rows as i32,
                hs as i32,
                hs as i32,
                64,
            ),
            0
        );
        assert_eq!(
            ffi::w34_fused_add_gemma_rms_norm_tap_hadamard_f16(
                ex,
                d_x.addr() as *const c_void,
                d_res_fused.addr() as *mut c_void,
                d_w.addr() as *const c_void,
                d_fused_norm.addr() as *mut c_void,
                d_fused_had.addr() as *mut c_void,
                std::ptr::null_mut(),
                std::ptr::null(),
                rows as i32,
                hs as i32,
                1e-6,
                0,
                0,
            ),
            0
        );
    }
    d_ref_norm.copy_to_host(&mut ref_norm, stream).unwrap();
    d_ref_had.copy_to_host(&mut ref_had, stream).unwrap();
    d_fused_norm.copy_to_host(&mut got_norm, stream).unwrap();
    d_fused_had.copy_to_host(&mut got_had, stream).unwrap();
    assert_eq!(got_norm, ref_norm, "FusedAddGemmaRMSNorm unrotated hs={hs}");
    assert_eq!(got_had, ref_had, "FusedAddGemmaRMSNorm H64 hs={hs}");
}

#[test]
fn producer_fused_hadamard64_matches_standalone_0_ulp() {
    let _guard = GPU_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    let _device = CudaDevice::set_current(0).unwrap();
    let stream = CudaStream::new(0).unwrap();
    let ex = stream.exec(0, 0);

    // 1. GemmaRMSNorm + FusedAddGemmaRMSNorm across all 4 hidden_size template regimes
    for hs in [128usize, 2048, 3072, 4096, 5120, 6144] {
        for rows in [1usize, 5] {
            verify_gemma_norms_hadamard64(&stream, &ex, hs, rows);
        }
    }

    // 2. GatedRMSNorm (SiLU and Sigmoid), SiluAndMul, and ApplyAttnOutputGate
    let rows = 4usize;
    let vh = 8usize;
    let vd = 128usize;
    let total_dim = vh * vd;
    let x_host: Vec<u16> = (0..rows * total_dim)
        .map(|i| f32_to_f16(((i * 41 + 11) % 211) as f32 / 64.0 - 1.5))
        .collect();
    let z_host: Vec<u16> = (0..rows * total_dim * 2)
        .map(|i| f32_to_f16(((i * 29 + 5) % 181) as f32 / 64.0 - 1.25))
        .collect();
    let w_host: Vec<u16> = (0..vd)
        .map(|i| f32_to_f16(((i * 13 + 7) % 89) as f32 / 64.0 + 0.5))
        .collect();
    let d_x = upload(&x_host, &stream);
    let d_z = upload(&z_host, &stream);
    let d_w = upload(&w_host, &stream);
    let d_ref = upload(&vec![0; rows * total_dim], &stream);
    let d_ref_had = upload(&vec![0; rows * total_dim], &stream);
    let d_fused = upload(&vec![0; rows * total_dim], &stream);

    for (ref_fn, fused_fn) in [
        (
            ffi::w34_rms_norm_gated_silu_strided_f16
                as unsafe extern "C" fn(
                    *const w34_kernels::ffi::W34Exec,
                    *const c_void,
                    *const c_void,
                    *const c_void,
                    *mut c_void,
                    *const u8,
                    i32,
                    i32,
                    i32,
                    i32,
                    f32,
                ) -> i32,
            ffi::w34_rms_norm_gated_silu_strided_hadamard_f16
                as unsafe extern "C" fn(
                    *const w34_kernels::ffi::W34Exec,
                    *const c_void,
                    *const c_void,
                    *const c_void,
                    *mut c_void,
                    *const u8,
                    i32,
                    i32,
                    i32,
                    i32,
                    f32,
                ) -> i32,
        ),
        (
            ffi::w34_rms_norm_gated_sigmoid_strided_f16,
            ffi::w34_rms_norm_gated_sigmoid_strided_hadamard_f16,
        ),
    ] {
        // SAFETY: buffers are sized for [rows, vh * vd] with z stride 2 * total_dim.
        unsafe {
            assert_eq!(
                ref_fn(
                    &ex,
                    d_x.addr() as *const c_void,
                    d_z.addr() as *const c_void,
                    d_w.addr() as *const c_void,
                    d_ref.addr() as *mut c_void,
                    std::ptr::null(),
                    rows as i32,
                    vh as i32,
                    vd as i32,
                    (total_dim * 2) as i32,
                    1e-6,
                ),
                0
            );
            assert_eq!(
                ffi::w34_block_hadamard_f16(
                    &ex,
                    d_ref.addr() as *const c_void,
                    d_ref_had.addr() as *mut c_void,
                    rows as i32,
                    total_dim as i32,
                    total_dim as i32,
                    64,
                ),
                0
            );
            assert_eq!(
                fused_fn(
                    &ex,
                    d_x.addr() as *const c_void,
                    d_z.addr() as *const c_void,
                    d_w.addr() as *const c_void,
                    d_fused.addr() as *mut c_void,
                    std::ptr::null(),
                    rows as i32,
                    vh as i32,
                    vd as i32,
                    (total_dim * 2) as i32,
                    1e-6,
                ),
                0
            );
        }
        let mut ref_had = vec![0u16; rows * total_dim];
        let mut got_had = vec![0u16; rows * total_dim];
        d_ref_had.copy_to_host(&mut ref_had, &stream).unwrap();
        d_fused.copy_to_host(&mut got_had, &stream).unwrap();
        assert_eq!(got_had, ref_had, "GatedRMSNorm H64 0-ULP mismatch");
    }

    // SiluAndMul
    // SAFETY: d_z has [rows, 2 * total_dim] elements; d_ref / d_fused have [rows, total_dim].
    unsafe {
        assert_eq!(
            ffi::w34_silu_and_mul_f16(
                &ex,
                d_z.addr() as *const c_void,
                d_ref.addr() as *mut c_void,
                std::ptr::null(),
                rows as i32,
                total_dim as i32,
            ),
            0
        );
        assert_eq!(
            ffi::w34_block_hadamard_f16(
                &ex,
                d_ref.addr() as *const c_void,
                d_ref_had.addr() as *mut c_void,
                rows as i32,
                total_dim as i32,
                total_dim as i32,
                64,
            ),
            0
        );
        assert_eq!(
            ffi::w34_silu_and_mul_hadamard_f16(
                &ex,
                d_z.addr() as *const c_void,
                d_fused.addr() as *mut c_void,
                std::ptr::null(),
                rows as i32,
                total_dim as i32,
            ),
            0
        );
    }
    let mut ref_had = vec![0u16; rows * total_dim];
    let mut got_had = vec![0u16; rows * total_dim];
    d_ref_had.copy_to_host(&mut ref_had, &stream).unwrap();
    d_fused.copy_to_host(&mut got_had, &stream).unwrap();
    assert_eq!(got_had, ref_had, "SiluAndMul H64 0-ULP mismatch");

    // ApplyAttnOutputGate
    // SAFETY: d_x, d_z, d_ref, d_fused have >= [rows, total_dim] elements.
    unsafe {
        assert_eq!(
            ffi::w34_apply_attn_output_gate_f16(
                &ex,
                d_x.addr() as *const c_void,
                d_z.addr() as *const c_void,
                d_ref.addr() as *mut c_void,
                std::ptr::null(),
                rows as i32,
                total_dim as i32,
            ),
            0
        );
        assert_eq!(
            ffi::w34_block_hadamard_f16(
                &ex,
                d_ref.addr() as *const c_void,
                d_ref_had.addr() as *mut c_void,
                rows as i32,
                total_dim as i32,
                total_dim as i32,
                64,
            ),
            0
        );
        assert_eq!(
            ffi::w34_apply_attn_output_gate_hadamard_f16(
                &ex,
                d_x.addr() as *const c_void,
                d_z.addr() as *const c_void,
                d_fused.addr() as *mut c_void,
                std::ptr::null(),
                rows as i32,
                total_dim as i32,
            ),
            0
        );
    }
    d_ref_had.copy_to_host(&mut ref_had, &stream).unwrap();
    d_fused.copy_to_host(&mut got_had, &stream).unwrap();
    assert_eq!(got_had, ref_had, "ApplyAttnOutputGate H64 0-ULP mismatch");
}

#[test]
fn tp2_all_reduce_residual_gemma_rms_norm_hadamard64_matches_standalone_0_ulp() {
    let _guard = GPU_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    if CudaDevice::count().unwrap_or(0) < 2 {
        return;
    }
    let comm = Tp2Collectives::new(12288 * 2).unwrap();
    let _d0 = CudaDevice::set_current(0).unwrap();
    let s0 = CudaStream::new(0).unwrap();
    let _d1 = CudaDevice::set_current(1).unwrap();
    let s1 = CudaStream::new(1).unwrap();

    for hidden in [128usize, 2560, 4096, 5120, 6144] {
        let in0_h: Vec<u16> = (0..hidden)
            .map(|i| f32_to_f16(((i * 31 + 11) % 197) as f32 / 64.0 - 1.5))
            .collect();
        let in1_h: Vec<u16> = (0..hidden)
            .map(|i| f32_to_f16(((i * 43 + 7) % 193) as f32 / 64.0 - 1.25))
            .collect();
        let res_h: Vec<u16> = (0..hidden)
            .map(|i| f32_to_f16(((i * 19 + 5) % 179) as f32 / 64.0 - 1.0))
            .collect();
        let w_h: Vec<u16> = (0..hidden)
            .map(|i| f32_to_f16(((i * 13 + 3) % 89) as f32 / 128.0 - 0.25))
            .collect();

        let _d0 = CudaDevice::set_current(0).unwrap();
        let d0_in = DeviceBuffer::alloc(0, hidden * 2).unwrap();
        let d0_res = DeviceBuffer::alloc(0, hidden * 2).unwrap();
        let d0_w = DeviceBuffer::alloc(0, hidden * 2).unwrap();
        let d0_out = DeviceBuffer::alloc(0, hidden * 2).unwrap();
        let d0_had = DeviceBuffer::alloc(0, hidden * 2).unwrap();
        let d0_ref_had = DeviceBuffer::alloc(0, hidden * 2).unwrap();
        d0_in.copy_from_host(&in0_h, &s0).unwrap();
        d0_res.copy_from_host(&res_h, &s0).unwrap();
        d0_w.copy_from_host(&w_h, &s0).unwrap();

        let _d1 = CudaDevice::set_current(1).unwrap();
        let d1_in = DeviceBuffer::alloc(1, hidden * 2).unwrap();
        let d1_res = DeviceBuffer::alloc(1, hidden * 2).unwrap();
        let d1_w = DeviceBuffer::alloc(1, hidden * 2).unwrap();
        let d1_out = DeviceBuffer::alloc(1, hidden * 2).unwrap();
        let d1_had = DeviceBuffer::alloc(1, hidden * 2).unwrap();
        d1_in.copy_from_host(&in1_h, &s1).unwrap();
        d1_res.copy_from_host(&res_h, &s1).unwrap();
        d1_w.copy_from_host(&w_h, &s1).unwrap();
        s0.synchronize().unwrap();
        s1.synchronize().unwrap();

        let (comm, s0_ref, s1_ref) = (&comm, &s0, &s1);
        std::thread::scope(|scope| {
            let (in0, res0, w0, out0, had0) = (
                d0_in.addr(),
                d0_res.addr(),
                d0_w.addr(),
                d0_out.addr(),
                d0_had.addr(),
            );
            let (in1, res1, w1, out1, had1) = (
                d1_in.addr(),
                d1_res.addr(),
                d1_w.addr(),
                d1_out.addr(),
                d1_had.addr(),
            );
            let t0 = scope.spawn(move || {
                let _dev = CudaDevice::set_current(0).unwrap();
                let ex0 = s0_ref.exec(0, 0);
                comm.all_reduce_residual_gemma_rms_norm_hadamard_f16(
                    &ex0,
                    in0,
                    0,
                    res0,
                    w0,
                    out0,
                    had0,
                    0,
                    0,
                    1,
                    hidden as i32,
                    1e-6,
                    0,
                    0,
                )
                .unwrap();
                s0_ref.synchronize().unwrap();
            });
            let t1 = scope.spawn(move || {
                let _dev = CudaDevice::set_current(1).unwrap();
                let ex1 = s1_ref.exec(0, 0);
                comm.all_reduce_residual_gemma_rms_norm_hadamard_f16(
                    &ex1,
                    in1,
                    0,
                    res1,
                    w1,
                    out1,
                    had1,
                    0,
                    1,
                    1,
                    hidden as i32,
                    1e-6,
                    0,
                    0,
                )
                .unwrap();
                s1_ref.synchronize().unwrap();
            });
            t0.join().unwrap();
            t1.join().unwrap();
        });

        let _d0 = CudaDevice::set_current(0).unwrap();
        launch(&s0, &d0_out, &d0_ref_had, 1, hidden, hidden, 64);
        let mut got0 = vec![0u16; hidden];
        let mut ref0 = vec![0u16; hidden];
        d0_had.copy_to_host(&mut got0, &s0).unwrap();
        d0_ref_had.copy_to_host(&mut ref0, &s0).unwrap();
        let _d1 = CudaDevice::set_current(1).unwrap();
        let mut got1 = vec![0u16; hidden];
        d1_had.copy_to_host(&mut got1, &s1).unwrap();
        assert_eq!(got0, ref0, "TP=2 rank 0 H64 0-ULP mismatch hidden={hidden}");
        assert_eq!(got1, ref0, "TP=2 rank 1 H64 0-ULP mismatch hidden={hidden}");
    }
}

#[test]
#[ignore = "manual GPU microbenchmark; run with --ignored --nocapture on an idle GPU"]
fn benchmark_block_hadamard() {
    let _guard = GPU_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    let _device = CudaDevice::set_current(0).unwrap();
    let stream = CudaStream::new(0).unwrap();
    for (block, k, padded_k) in [
        (64, 2048, 2048),
        (64, 4096, 4096),
        (64, 5120, 5120),
        (64, 6144, 6144),
        (64, 12288, 12288),
        (64, 17408, 17408),
        (16, 4304, 4352),
    ] {
        for rows in [1, 8, 16, 512, 4096] {
            let src = upload(&vec![f32_to_f16(0.25); rows * k], &stream);
            let dst = upload(&vec![0; rows * padded_k], &stream);
            launch(&stream, &src, &dst, rows, k, padded_k, block);
            stream.synchronize().unwrap();
            let graph = CudaGraph::capture(&stream, || {
                for _ in 0..100 {
                    launch(&stream, &src, &dst, rows, k, padded_k, block);
                }
                Ok(())
            })
            .unwrap();
            graph.launch(&stream).unwrap();
            let start = CudaEvent::new(0, true).unwrap();
            let end = CudaEvent::new(0, true).unwrap();
            start.record(&stream).unwrap();
            for _ in 0..10 {
                graph.launch(&stream).unwrap();
            }
            end.record(&stream).unwrap();
            stream.synchronize().unwrap();
            let us = start.elapsed_ms(&end).unwrap(); // 1000 kernels: ms total = us/kernel
            eprintln!("H{block} M={rows} K={k} padded_K={padded_k}: {us:.3} us/kernel");
        }
    }
}
