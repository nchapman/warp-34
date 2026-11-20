#![cfg(feature = "cuda")]

use std::ffi::c_void;
use warp34_host::{
    cuda::{CudaDevice, CudaStream, DeviceBuffer, f16_to_f32, f32_to_f16},
    loader::{marlin_permute_scales_from_n_major, marlin_permute_zero_points_from_n_major},
};
use warp34_kernels::ffi;

static GPU_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

/// Explicit opt-in: these cached artifacts total hundreds of GB. Missing models
/// fail rather than silently claiming checkpoint coverage.
#[test]
#[ignore = "requires the quantization qualification HF cache and two 24-GiB GPUs"]
fn cached_hf_quantized_and_unquantized_checkpoints() {
    use warp34_host::{cache::KvFormatPolicy, model::Qwen3_5Engine};
    let _guard = GPU_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    let cases = [
        ("Qwen/Qwen3.5-4B", 1),
        ("cyankiwi/Qwen3.5-9B-AWQ-4bit", 1),
        ("nchapman/Qwen3.8-27B-W4A16-HGPTQ", 2),
        ("cyankiwi/Qwen3.8-27B-AWQ-INT4", 2),
        ("TheUnderscore/Swift-Qwen3.8-27b-W4A16-AWQ", 2),
        ("Lorbus/Qwen3.6-27B-int4-AutoRound", 2),
        ("lued/Qwen3.8-27B-INT8-W8A16-DFlash2", 2),
        ("cyankiwi/Ornith-1.5-35B-A3B-AWQ-INT4", 2),
        ("biMEMO/Ornith-1.5-35B-A3B-int4-AutoRound-MTP", 2),
        ("nchapman/Qwen3.6-35B-A3B-W4A16-HGPTQ", 2),
    ];
    let mut failures = Vec::new();
    for (model, tp) in cases {
        eprintln!("Loading cached {model} TP={tp}");
        let mut engine = match Qwen3_5Engine::load(model, tp, KvFormatPolicy::Fp16, Some(32), false)
        {
            Ok(engine) => engine,
            Err(error) => {
                failures.push(format!("{model}: {error}"));
                continue;
            }
        };
        let prompt = engine.encode("The capital of France is").unwrap();
        let dump = engine.dump_prompt(&prompt).unwrap();
        let hidden = &dump.tensors_f32["final_norm"].1;
        assert!(
            hidden.iter().all(|v| v.is_finite()),
            "{model}: nonfinite final hidden state"
        );
        assert!(
            hidden.iter().any(|&v| v != 0.0),
            "{model}: zero final hidden state"
        );
        let generated = engine.generate_tokens(&prompt, 8, false).unwrap();
        assert_eq!(generated.generated_token_ids.len(), 8, "{model}");
        assert!(
            generated
                .generated_token_ids
                .iter()
                .all(|&t| t >= 0 && (t as usize) < engine.cfg.vocab_size)
        );
        eprintln!("{model} TP={tp}: {:?}", generated.generated_token_ids);
    }
    assert!(
        failures.is_empty(),
        "cached checkpoint load failures:\n{}",
        failures.join("\n")
    );
}

fn upload<T: Copy>(values: &[T], stream: &CudaStream) -> DeviceBuffer {
    let buffer = DeviceBuffer::alloc(0, std::mem::size_of_val(values)).unwrap();
    buffer.copy_from_host(values, stream).unwrap();
    buffer
}

fn assert_close(got: &[u16], reference: &[f32], label: &str) {
    assert_eq!(got.len(), reference.len());
    let mut error = 0.0f64;
    let mut energy = 0.0f64;
    for (i, (&bits, &expected)) in got.iter().zip(reference).enumerate() {
        let actual = f16_to_f32(bits);
        assert!(actual.is_finite(), "{label} nonfinite output at {i}");
        let delta = (actual - expected).abs();
        assert!(
            delta < 0.003 + expected.abs() * 0.004,
            "{label} at {i}: got {actual}, expected {expected}, delta {delta}"
        );
        error += f64::from(delta).powi(2);
        energy += f64::from(expected).powi(2);
    }
    let rel_l2 = (error / energy.max(1e-30)).sqrt();
    eprintln!("{label}: rel_l2={rel_l2:.6}");
    assert!(rel_l2 < 0.003, "{label}: rel_l2={rel_l2}");
}

// Independent scalar weight construction: references never call the production
// unpack/dequant helpers, so packing, scale permutation, and zero-point bugs fail.
struct Weights {
    packed: Vec<u32>,
    scales: Vec<u16>,
    zeros: Vec<u32>,
    dense: Vec<f32>,
}

fn weights(
    n: usize,
    k: usize,
    group: usize,
    bits: usize,
    asymmetric: bool,
    seed: usize,
) -> Weights {
    let groups = k / group;
    let per_word = 32 / bits;
    let mask = (1 << bits) - 1;
    let mut result = Weights {
        packed: vec![0; n * k / per_word],
        scales: vec![0; n * groups],
        zeros: vec![0; n / 8 * groups],
        dense: vec![0.0; n * k],
    };
    for row in 0..n {
        for g in 0..groups {
            let zero = if asymmetric {
                (row * 3 + g * 5 + seed) % 16
            } else {
                1 << (bits - 1)
            };
            let scale = (1 + (row + g + seed) % 3) as f32 / if bits == 4 { 128.0 } else { 2048.0 };
            result.scales[row * groups + g] = f32_to_f16(scale);
            if asymmetric {
                result.zeros[(row / 8) * groups + g] |= (zero as u32) << ((row % 8) * 4);
            }
            for col in g * group..(g + 1) * group {
                let q =
                    ((row * 73 + col * 31 + (row ^ col) * 13 + seed * 7) ^ ((col * 3) >> 2)) & mask;
                result.packed[row * (k / per_word) + col / per_word] |=
                    (q as u32) << ((col % per_word) * bits);
                result.dense[row * k + col] = (q as i32 - zero as i32) as f32 * scale;
            }
        }
    }
    result
}

#[test]
fn marlin_w4_symmetric_asymmetric_and_w8_match_scalar_reference() {
    let _guard = GPU_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    let _dev = CudaDevice::set_current(0).unwrap();
    let stream = CudaStream::new(0).unwrap();
    let ex = stream.exec(0, 0);
    let (n, k) = (256usize, 256usize);
    let workspace = DeviceBuffer::alloc(0, 65536 * 4).unwrap();
    for (bits, asymmetric) in [(4, false), (4, true), (8, false)] {
        let groups: &[usize] = if !asymmetric {
            &[32, 64, 128, 256]
        } else {
            &[32, 64, 128]
        };
        for &group in groups {
            let w = weights(n, k, group, bits, asymmetric, 1);
            let packed = upload(&w.packed, &stream);
            let repacked = DeviceBuffer::alloc(0, w.packed.len() * 4).unwrap();
            let perm_group = if group == k && group > 128 { 0 } else { group };
            let scales = upload(
                &marlin_permute_scales_from_n_major(&w.scales, k, n, perm_group),
                &stream,
            );
            let zeros = upload(
                &marlin_permute_zero_points_from_n_major(&w.zeros, k, n, group),
                &stream,
            );
            // SAFETY: packed and repacked buffers are allocated on the active GPU device with correct element counts.
            let rc = unsafe {
                if bits == 8 {
                    ffi::warp34_marlin_repack_int8(
                        &ex,
                        packed.addr() as *const u32,
                        repacked.addr() as *mut u32,
                        k as i64,
                        n as i64,
                    )
                } else {
                    ffi::warp34_marlin_repack_int4(
                        &ex,
                        packed.addr() as *const u32,
                        repacked.addr() as *mut u32,
                        k as i64,
                        n as i64,
                    )
                }
            };
            assert_eq!(rc, 0);
            for m in [1usize, 4, 8, 16, 130] {
                let label = format!("W{bits} asym={asymmetric} group={group} M={m}");
                let a: Vec<u16> = (0..m * k)
                    .map(|i| f32_to_f16(((i * 17 + i / k * 7) % 47) as f32 / 64.0 - 0.359375))
                    .collect();
                let mut reference = vec![0.0; m * n];
                for row in 0..m {
                    for col in 0..n {
                        reference[row * n + col] = (0..k)
                            .map(|j| f16_to_f32(a[row * k + j]) * w.dense[col * k + j])
                            .sum();
                    }
                }
                let input = upload(&a, &stream);
                let output = DeviceBuffer::alloc(0, m * n * 2).unwrap();
                workspace.zero_async(&stream).unwrap();
                let gb = if group == k { -1 } else { (group / 16) as i32 };
                // SAFETY: all tensor inputs, outputs, scales, and zero-point pointers are valid device allocations matching matrix dimensions.
                let rc = unsafe {
                    if asymmetric {
                        ffi::warp34_marlin_gemm_f16u4_zp(
                            &ex,
                            input.addr() as *const c_void,
                            repacked.addr() as *const c_void,
                            scales.addr() as *const c_void,
                            zeros.addr() as *const c_void,
                            output.addr() as *mut c_void,
                            workspace.addr() as *mut i32,
                            m as i64,
                            n as i64,
                            k as i64,
                            65536,
                            gb,
                        )
                    } else if bits == 8 {
                        ffi::warp34_marlin_gemm_f16u8(
                            &ex,
                            input.addr() as *const c_void,
                            repacked.addr() as *const c_void,
                            scales.addr() as *const c_void,
                            output.addr() as *mut c_void,
                            workspace.addr() as *mut i32,
                            m as i64,
                            n as i64,
                            k as i64,
                            65536,
                            gb,
                        )
                    } else {
                        ffi::warp34_marlin_gemm_f16u4(
                            &ex,
                            input.addr() as *const c_void,
                            repacked.addr() as *const c_void,
                            scales.addr() as *const c_void,
                            output.addr() as *mut c_void,
                            workspace.addr() as *mut i32,
                            m as i64,
                            n as i64,
                            k as i64,
                            65536,
                            gb,
                        )
                    }
                };
                assert_eq!(rc, 0, "{label}");
                let mut got = vec![0; m * n];
                output.copy_to_host(&mut got, &stream).unwrap();
                assert_close(&got, &reference, &label);
            }
        }
    }
}

// Wide N exercises persistent whole-tile CTAs; M=8 -> 9 switches variants.
#[test]
fn marlin_w4_persistent_tiles_match_scalar_and_identical_rows() {
    let _guard = GPU_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    let _dev = CudaDevice::set_current(0).unwrap();
    let stream = CudaStream::new(0).unwrap();
    let ex = stream.exec(0, 0);
    let k = 256usize;
    let workspace = DeviceBuffer::alloc(0, 65536 * 4).unwrap();
    let row: Vec<u16> = (0..k)
        .map(|i| f32_to_f16(((i * 17) % 47) as f32 / 64.0 - 0.359375))
        .collect();
    let rows = row.repeat(16);
    let input = upload(&rows, &stream);
    let row_f32: Vec<f32> = row.iter().map(|&v| f16_to_f32(v)).collect();
    for n in [6144usize, 10240, 12288, 24576] {
        let output = DeviceBuffer::alloc(0, 16 * n * 2).unwrap();
        for group in [64usize, 128] {
            let w = weights(n, k, group, 4, false, 1);
            let packed = upload(&w.packed, &stream);
            let repacked = DeviceBuffer::alloc(0, w.packed.len() * 4).unwrap();
            let scales = upload(
                &marlin_permute_scales_from_n_major(&w.scales, k, n, group),
                &stream,
            );
            // SAFETY: packed and repacked are device buffers sized for the K x N int4 weights.
            let rc = unsafe {
                ffi::warp34_marlin_repack_int4(
                    &ex,
                    packed.addr() as *const u32,
                    repacked.addr() as *mut u32,
                    k as i64,
                    n as i64,
                )
            };
            assert_eq!(rc, 0);
            let reference: Vec<f32> = w
                .dense
                .chunks_exact(k)
                .map(|weights| row_f32.iter().zip(weights).map(|(a, b)| a * b).sum())
                .collect();
            let mut baseline = Vec::new();
            for m in [1usize, 4, 8, 9, 16] {
                let label = format!("persistent W4 group={group} N={n} M={m}");
                workspace.zero_async(&stream).unwrap();
                // SAFETY: device tensors cover the requested dimensions (M <= 16),
                // scales cover all quant groups, and workspace holds 65536 i32 entries.
                let rc = unsafe {
                    ffi::warp34_marlin_gemm_f16u4(
                        &ex,
                        input.addr() as *const c_void,
                        repacked.addr() as *const c_void,
                        scales.addr() as *const c_void,
                        output.addr() as *mut c_void,
                        workspace.addr() as *mut i32,
                        m as i64,
                        n as i64,
                        k as i64,
                        65536,
                        (group / 16) as i32,
                    )
                };
                assert_eq!(rc, 0, "{label}");
                let mut got = vec![0u16; m * n];
                output.copy_to_host(&mut got, &stream).unwrap();
                if m == 1 {
                    assert_close(&got, &reference, &label);
                    baseline = got;
                } else {
                    for (r, values) in got.chunks_exact(n).enumerate() {
                        for (col, (&actual, &expected)) in values.iter().zip(&baseline).enumerate()
                        {
                            assert_eq!(actual, expected, "{label} row={r} col={col}");
                        }
                    }
                }
            }
        }
    }
}

#[test]
fn marlin_gemm_k_128_group_128_and_channelwise_match_reference() {
    let _guard = GPU_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    let _dev = CudaDevice::set_current(0).unwrap();
    let stream = CudaStream::new(0).unwrap();
    let ex = stream.exec(0, 0);
    let (n, k) = (128usize, 128usize);
    let workspace = DeviceBuffer::alloc(0, (n / 64) * 16 * 4).unwrap();
    for (bits, asymmetric, perm_gs, gb) in [
        (4usize, false, 128usize, 8i32),
        (4usize, true, 128usize, 8i32),
        (8usize, false, 128usize, 8i32),
        (8usize, false, 0usize, -1i32),
    ] {
        let w = weights(n, k, 128, bits, asymmetric, 3);
        let packed = upload(&w.packed, &stream);
        let repacked = DeviceBuffer::alloc(0, w.packed.len() * 4).unwrap();
        let scales = upload(
            &marlin_permute_scales_from_n_major(&w.scales, k, n, perm_gs),
            &stream,
        );
        let zeros = upload(
            &marlin_permute_zero_points_from_n_major(&w.zeros, k, n, 128),
            &stream,
        );
        // SAFETY: packed and repacked buffers are allocated on the active GPU device with correct element counts.
        let rc = unsafe {
            if bits == 8 {
                ffi::warp34_marlin_repack_int8(
                    &ex,
                    packed.addr() as *const u32,
                    repacked.addr() as *mut u32,
                    k as i64,
                    n as i64,
                )
            } else {
                ffi::warp34_marlin_repack_int4(
                    &ex,
                    packed.addr() as *const u32,
                    repacked.addr() as *mut u32,
                    k as i64,
                    n as i64,
                )
            }
        };
        assert_eq!(rc, 0);
        for m in [1usize, 8, 16, 65] {
            let label = format!("k=128 W{bits} asym={asymmetric} gb={gb} M={m}");
            let a: Vec<u16> = (0..m * k)
                .map(|i| f32_to_f16(((i * 19 + i / k * 5) % 43) as f32 / 64.0 - 0.328125))
                .collect();
            let mut reference = vec![0.0; m * n];
            for row in 0..m {
                for col in 0..n {
                    reference[row * n + col] = (0..k)
                        .map(|j| f16_to_f32(a[row * k + j]) * w.dense[col * k + j])
                        .sum();
                }
            }
            let input = upload(&a, &stream);
            let output = DeviceBuffer::alloc(0, m * n * 2).unwrap();
            workspace.zero_async(&stream).unwrap();
            // SAFETY: all tensor inputs, outputs, scales, and zero-point pointers are valid device allocations matching matrix dimensions.
            let rc = unsafe {
                if asymmetric {
                    ffi::warp34_marlin_gemm_f16u4_zp(
                        &ex,
                        input.addr() as *const c_void,
                        repacked.addr() as *const c_void,
                        scales.addr() as *const c_void,
                        zeros.addr() as *const c_void,
                        output.addr() as *mut c_void,
                        workspace.addr() as *mut i32,
                        m as i64,
                        n as i64,
                        k as i64,
                        65536,
                        gb,
                    )
                } else if bits == 8 {
                    ffi::warp34_marlin_gemm_f16u8(
                        &ex,
                        input.addr() as *const c_void,
                        repacked.addr() as *const c_void,
                        scales.addr() as *const c_void,
                        output.addr() as *mut c_void,
                        workspace.addr() as *mut i32,
                        m as i64,
                        n as i64,
                        k as i64,
                        65536,
                        gb,
                    )
                } else {
                    ffi::warp34_marlin_gemm_f16u4(
                        &ex,
                        input.addr() as *const c_void,
                        repacked.addr() as *const c_void,
                        scales.addr() as *const c_void,
                        output.addr() as *mut c_void,
                        workspace.addr() as *mut i32,
                        m as i64,
                        n as i64,
                        k as i64,
                        65536,
                        gb,
                    )
                }
            };
            assert_eq!(rc, 0, "{label}");
            let mut got = vec![0; m * n];
            output.copy_to_host(&mut got, &stream).unwrap();
            assert_close(&got, &reference, &label);
        }
    }
}

#[test]
fn asymmetric_moe_gemv_and_grouped_mma_match_scalar_reference() {
    let _guard = GPU_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    let _dev = CudaDevice::set_current(0).unwrap();
    let stream = CudaStream::new(0).unwrap();
    let (h, n, experts, topk) = (128usize, 128usize, 4usize, 2usize);
    let scratch = DeviceBuffer::alloc(0, 8 * 1024 * 1024).unwrap();
    for group in [32usize, 64, 128] {
        let gu: Vec<_> = (0..experts)
            .map(|e| weights(2 * n, h, group, 4, true, e + 1))
            .collect();
        let down: Vec<_> = (0..experts)
            .map(|e| weights(h, n, group, 4, true, e + 7))
            .collect();
        let gu_p = upload(
            &gu.iter()
                .flat_map(|w| w.packed.iter().copied())
                .collect::<Vec<_>>(),
            &stream,
        );
        let gu_s = upload(
            &gu.iter()
                .flat_map(|w| w.scales.iter().copied())
                .collect::<Vec<_>>(),
            &stream,
        );
        let gu_z = upload(
            &gu.iter()
                .flat_map(|w| w.zeros.iter().copied())
                .collect::<Vec<_>>(),
            &stream,
        );
        let dn_p = upload(
            &down
                .iter()
                .flat_map(|w| w.packed.iter().copied())
                .collect::<Vec<_>>(),
            &stream,
        );
        let dn_s = upload(
            &down
                .iter()
                .flat_map(|w| w.scales.iter().copied())
                .collect::<Vec<_>>(),
            &stream,
        );
        let dn_z = upload(
            &down
                .iter()
                .flat_map(|w| w.zeros.iter().copied())
                .collect::<Vec<_>>(),
            &stream,
        );
        for m in [1usize, 4, 8, 16, 130] {
            let x: Vec<u16> = (0..m * h)
                .map(|i| f32_to_f16(((i * 17 + i / h) % 31) as f32 / 128.0 - 0.1171875))
                .collect();
            let ids: Vec<i32> = (0..m * topk)
                .map(|i| ((i + i / topk) % experts) as i32)
                .collect();
            let tw: Vec<f32> = (0..m * topk)
                .map(|i| if i % topk == 0 { 0.75 } else { 0.25 })
                .collect();
            let mut reference = vec![0.0; m * h];
            for row in 0..m {
                for slot in 0..topk {
                    let e = ids[row * topk + slot] as usize;
                    let mut intermediate = vec![0.0; n];
                    for (j, value) in intermediate.iter_mut().enumerate() {
                        let gate: f32 = (0..h)
                            .map(|c| f16_to_f32(x[row * h + c]) * gu[e].dense[j * h + c])
                            .sum();
                        let up: f32 = (0..h)
                            .map(|c| f16_to_f32(x[row * h + c]) * gu[e].dense[(n + j) * h + c])
                            .sum();
                        *value = f16_to_f32(f32_to_f16(gate / (1.0 + (-gate).exp()) * up));
                    }
                    for c in 0..h {
                        let value: f32 = (0..n)
                            .map(|j| intermediate[j] * down[e].dense[c * n + j])
                            .sum();
                        reference[row * h + c] += value * tw[row * topk + slot];
                    }
                }
            }
            let input = upload(&x, &stream);
            let ids = upload(&ids, &stream);
            let tw = upload(&tw, &stream);
            let intermediate = DeviceBuffer::alloc(0, m * topk * n * 2).unwrap();
            let output = DeviceBuffer::alloc(0, m * h * 2).unwrap();
            // Zero scratch forces GEMV; bounded scratch also exercises chunked MMA.
            for scratch_bytes in [0usize, 32 * 1024, scratch.len_bytes] {
                let ex = stream.exec(scratch.addr(), scratch_bytes);
                let label = format!("MoE asymmetric group={group} M={m} scratch={scratch_bytes}");
                // SAFETY: all expert weights, routing buffers, and intermediate/output pointers are valid device buffers matching MoE geometry.
                let rc = unsafe {
                    ffi::warp34_moe_w4a16_mlp_zp_f16(
                        &ex,
                        input.addr() as *const c_void,
                        gu_p.addr() as *const u32,
                        gu_s.addr() as *const c_void,
                        gu_z.addr() as *const u32,
                        dn_p.addr() as *const u32,
                        dn_s.addr() as *const c_void,
                        dn_z.addr() as *const u32,
                        ids.addr() as *const i32,
                        tw.addr() as *const f32,
                        std::ptr::null(),
                        std::ptr::null(),
                        intermediate.addr() as *mut c_void,
                        output.addr() as *mut c_void,
                        m as i32,
                        topk as i32,
                        experts as i32,
                        n as i32,
                        h as i32,
                        group as i32,
                    )
                };
                assert_eq!(rc, 0, "{label}");
                let mut got = vec![0; m * h];
                output.copy_to_host(&mut got, &stream).unwrap();
                assert_close(&got, &reference, &label);
            }
        }
    }
}

#[test]
fn online_marlin_quantize_f16_u4b8_matches_cpu_reference_and_tracks_fp16_logits() {
    let _guard = GPU_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    let _dev = CudaDevice::set_current(0).unwrap();
    let stream = CudaStream::new(0).unwrap();
    let ex = stream.exec(0, 0);
    let (n, k) = (256usize, 256usize);
    let src_f16: Vec<u16> = (0..n * k)
        .map(|i| {
            let row = i / k;
            let col = i % k;
            match row {
                0 => 0u16,
                1 => 0x0001u16,
                2 => {
                    if col == 0 {
                        f32_to_f16(7.0 * 0.0625)
                    } else {
                        f32_to_f16(((col % 16) as f32 - 7.5) * 0.0625)
                    }
                }
                3 => {
                    if col == 0 {
                        0x3901u16
                    } else if col == 1 {
                        0xB901u16
                    } else {
                        f32_to_f16(((col % 15) as f32 - 7.0) * 0.08)
                    }
                }
                _ => {
                    let v = (((row * 37 + col * 13) % 101) as f32 - 50.0) * 0.015;
                    f32_to_f16(v)
                }
            }
        })
        .collect();
    let src_dev = upload(&src_f16, &stream);
    let workspace = DeviceBuffer::alloc(0, 65536 * 4).unwrap();

    for group_size in [32usize, 64, 128] {
        // 1. CPU reference quantization (`U4B8` N-major + scale permute + Marlin repack).
        let num_groups = k / group_size;
        let words_per_row = k / 8;
        let mut cpu_packed = vec![0u32; n * words_per_row];
        let mut cpu_scales_n_major = vec![0u16; n * num_groups];
        let mut dequant_ref = vec![0.0f32; n * k];
        for r in 0..n {
            for g in 0..num_groups {
                let slice = &src_f16[r * k + g * group_size..r * k + (g + 1) * group_size];
                let max_abs = slice
                    .iter()
                    .map(|&bits| f16_to_f32(bits).abs())
                    .fold(0.0f32, f32::max);
                let scale_h = f32_to_f16(max_abs / 7.0);
                let scale_f = f16_to_f32(scale_h);
                let inv_scale = if scale_f > 0.0 { 1.0 / scale_f } else { 0.0 };
                cpu_scales_n_major[r * num_groups + g] = scale_h;
                for (c, &bits) in slice.iter().enumerate() {
                    let k_col = g * group_size + c;
                    let w = f16_to_f32(bits);
                    let q = ((w * inv_scale).round_ties_even() as i32 + 8).clamp(0, 15) as u32;
                    cpu_packed[r * words_per_row + k_col / 8] |= q << ((k_col % 8) * 4);
                    dequant_ref[r * k + k_col] = (q as i32 - 8) as f32 * scale_f;
                }
            }
        }
        let cpu_perm_scales =
            marlin_permute_scales_from_n_major(&cpu_scales_n_major, k, n, group_size);
        let packed_dev = upload(&cpu_packed, &stream);
        let ref_repacked_dev = DeviceBuffer::alloc(0, cpu_packed.len() * 4).unwrap();
        // SAFETY: `packed_dev` and `ref_repacked_dev` have capacity for `n * (k / 8)` u32 elements.
        let rc = unsafe {
            ffi::warp34_marlin_repack_int4(
                &ex,
                packed_dev.addr() as *const u32,
                ref_repacked_dev.addr() as *mut u32,
                k as i64,
                n as i64,
            )
        };
        assert_eq!(rc, 0);

        // 2. Online GPU quantization (`warp34_marlin_quantize_f16_u4b8`).
        let gpu_repacked_dev = DeviceBuffer::alloc(0, cpu_packed.len() * 4).unwrap();
        let gpu_scales_dev = DeviceBuffer::alloc(0, cpu_perm_scales.len() * 2).unwrap();
        // SAFETY: `src_dev`, `gpu_repacked_dev`, and `gpu_scales_dev` match `(k, n, group_size)`.
        let rc = unsafe {
            ffi::warp34_marlin_quantize_f16_u4b8(
                &ex,
                src_dev.addr() as *const c_void,
                gpu_repacked_dev.addr() as *mut u32,
                gpu_scales_dev.addr() as *mut c_void,
                k as i64,
                n as i64,
                group_size as i32,
            )
        };
        assert_eq!(rc, 0, "group_size={group_size}");

        let mut ref_repacked = vec![0u32; cpu_packed.len()];
        let mut gpu_repacked = vec![0u32; cpu_packed.len()];
        let mut gpu_scales = vec![0u16; cpu_perm_scales.len()];
        ref_repacked_dev
            .copy_to_host(&mut ref_repacked, &stream)
            .unwrap();
        gpu_repacked_dev
            .copy_to_host(&mut gpu_repacked, &stream)
            .unwrap();
        gpu_scales_dev
            .copy_to_host(&mut gpu_scales, &stream)
            .unwrap();
        assert_eq!(
            gpu_scales, cpu_perm_scales,
            "scales mismatch at group_size={group_size}"
        );
        assert_eq!(
            gpu_repacked, ref_repacked,
            "repacked U4B8 mismatch at group_size={group_size}"
        );

        // 3. Verify `warp34_marlin_gemm_f16u4` matches the dequantized reference and tracks FP16 argmax.
        let m = 16usize;
        let x: Vec<u16> = (0..m * k)
            .map(|i| {
                let row = i / k;
                let col = i % k;
                let base = (((i * 23 + row * 7) % 67) as f32 - 33.0) * 0.01;
                let signal = f16_to_f32(src_f16[(row * 5 + 4) * k + col]);
                f32_to_f16(base + signal)
            })
            .collect();
        let x_dev = upload(&x, &stream);
        let out_dev = DeviceBuffer::alloc(0, m * n * 2).unwrap();
        workspace.zero_async(&stream).unwrap();
        // SAFETY: `x_dev`, `gpu_repacked_dev`, `gpu_scales_dev`, `out_dev`, and `workspace` match `(m, n, k)`.
        let rc = unsafe {
            ffi::warp34_marlin_gemm_f16u4(
                &ex,
                x_dev.addr() as *const c_void,
                gpu_repacked_dev.addr() as *const c_void,
                gpu_scales_dev.addr() as *const c_void,
                out_dev.addr() as *mut c_void,
                workspace.addr() as *mut i32,
                m as i64,
                n as i64,
                k as i64,
                65536,
                (group_size / 16) as i32,
            )
        };
        assert_eq!(rc, 0);
        let mut got = vec![0u16; m * n];
        out_dev.copy_to_host(&mut got, &stream).unwrap();
        let mut expected_dequant = vec![0.0f32; m * n];
        let mut expected_fp16 = vec![0.0f32; m * n];
        for row in 0..m {
            for col in 0..n {
                expected_dequant[row * n + col] = (0..k)
                    .map(|j| f16_to_f32(x[row * k + j]) * dequant_ref[col * k + j])
                    .sum();
                expected_fp16[row * n + col] = (0..k)
                    .map(|j| f16_to_f32(x[row * k + j]) * f16_to_f32(src_f16[col * k + j]))
                    .sum();
            }
            let got_argmax = (0..n)
                .max_by(|&a, &b| {
                    f16_to_f32(got[row * n + a]).total_cmp(&f16_to_f32(got[row * n + b]))
                })
                .unwrap();
            let fp16_argmax = (0..n)
                .max_by(|&a, &b| expected_fp16[row * n + a].total_cmp(&expected_fp16[row * n + b]))
                .unwrap();
            assert_eq!(
                got_argmax, fp16_argmax,
                "row {row} argmax mismatch vs FP16 at group_size={group_size}"
            );
        }
        assert_close(
            &got,
            &expected_dequant,
            &format!("online quantize gemm group_size={group_size}"),
        );
    }

    // 4. Multi-wave grid-stride coverage (`total_tiles = (512/16) * (2048/64) = 1024 > SMs * 8`)
    //    and boundary/alignment error validation.
    let (mw_n, mw_k, mw_gs) = (2048usize, 512usize, 32usize);
    let mw_src: Vec<u16> = (0..mw_n * mw_k)
        .map(|i| f32_to_f16((((i * 19 + i / mw_k * 11) % 97) as f32 - 48.0) * 0.02))
        .collect();
    let mw_src_dev = upload(&mw_src, &stream);
    let mw_qweight_dev = DeviceBuffer::alloc(0, (mw_n * (mw_k / 8)) * 4).unwrap();
    let mw_scales_dev = DeviceBuffer::alloc(0, (mw_n * (mw_k / mw_gs)) * 2).unwrap();
    // SAFETY: `mw_src_dev`, `mw_qweight_dev`, and `mw_scales_dev` match `(mw_k, mw_n, mw_gs)`.
    let rc = unsafe {
        ffi::warp34_marlin_quantize_f16_u4b8(
            &ex,
            mw_src_dev.addr() as *const c_void,
            mw_qweight_dev.addr() as *mut u32,
            mw_scales_dev.addr() as *mut c_void,
            mw_k as i64,
            mw_n as i64,
            mw_gs as i32,
        )
    };
    assert_eq!(rc, 0, "multi-wave quantize");

    for (src_ptr, bad_k, bad_n, bad_gs) in [
        (src_dev.addr() + 2, 256, 256, 64),
        (src_dev.addr(), 48, 256, 64),
        (src_dev.addr(), 256, 32, 64),
        (src_dev.addr(), 256, 256, 16),
    ] {
        // SAFETY: invalid parameters are rejected before launching any kernel.
        let rc = unsafe {
            ffi::warp34_marlin_quantize_f16_u4b8(
                &ex,
                src_ptr as *const c_void,
                mw_qweight_dev.addr() as *mut u32,
                mw_scales_dev.addr() as *mut c_void,
                bad_k,
                bad_n,
                bad_gs,
            )
        };
        assert_ne!(rc, 0, "expected error for ({bad_k}, {bad_n}, {bad_gs})");
    }
}

#[test]
fn hgptq_channelwise_w4_padded_k_matches_scalar() {
    let _guard = GPU_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    let _dev = CudaDevice::set_current(0).unwrap();
    let stream = CudaStream::new(0).unwrap();
    let ex = stream.exec(0, 0);
    for (k, logical_k, n) in [(256usize, 256usize, 256usize), (4352, 4304, 1152)] {
        let mut w = weights(n, k, k, 4, false, 3);
        for col in 0..n {
            for j in logical_k..k {
                let word = &mut w.packed[col * (k / 8) + j / 8];
                *word = (*word & !(15 << ((j % 8) * 4))) | (8 << ((j % 8) * 4));
                w.dense[col * k + j] = 0.0;
            }
        }
        let packed = upload(&w.packed, &stream);
        let repacked = DeviceBuffer::alloc(0, w.packed.len() * 4).unwrap();
        let scales = upload(
            &marlin_permute_scales_from_n_major(&w.scales, k, n, 0),
            &stream,
        );
        let workspace = DeviceBuffer::alloc(0, 65536 * 4).unwrap();
        assert_eq!(
            // SAFETY: device allocations match the packed matrix dimensions.
            unsafe {
                ffi::warp34_marlin_repack_int4(
                    &ex,
                    packed.addr() as *const u32,
                    repacked.addr() as *mut u32,
                    k as i64,
                    n as i64,
                )
            },
            0
        );
        for m in [1usize, 8, 16, 40, 64] {
            let a: Vec<u16> = (0..m * k)
                .map(|i| {
                    f32_to_f16(if i % k >= logical_k {
                        0.0
                    } else {
                        ((i * 17 + i / k * 7) % 47) as f32 / 64.0 - 0.359375
                    })
                })
                .collect();
            let input = upload(&a, &stream);
            let output = DeviceBuffer::alloc(0, m * n * 2).unwrap();
            workspace.zero_async(&stream).unwrap();
            assert_eq!(
                // SAFETY: all live buffers have the documented extents and scales are channelwise.
                unsafe {
                    ffi::warp34_marlin_gemm_f16u4(
                        &ex,
                        input.addr() as *const c_void,
                        repacked.addr() as *const c_void,
                        scales.addr() as *const c_void,
                        output.addr() as *mut c_void,
                        workspace.addr() as *mut i32,
                        m as i64,
                        n as i64,
                        k as i64,
                        65536,
                        -1,
                    )
                },
                0
            );
            let reference: Vec<f32> = (0..m * n)
                .map(|i| {
                    (0..logical_k)
                        .map(|j| f16_to_f32(a[i / n * k + j]) * w.dense[i % n * k + j])
                        .sum()
                })
                .collect();
            let mut got = vec![0u16; m * n];
            output.copy_to_host(&mut got, &stream).unwrap();
            assert_close(
                &got,
                &reference,
                &format!("channelwise W4 M={m} K={k} N={n}"),
            );
        }
    }
}

#[test]
fn linear_epilogue_crop_bias_in_place_and_validation() {
    let _guard = GPU_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    let _dev = CudaDevice::set_current(0).unwrap();
    let stream = CudaStream::new(0).unwrap();
    let ex = stream.exec(0, 0);
    for (rows, logical, physical) in [(3usize, 5usize, 8usize), (17, 4304, 4352), (2, 64, 64)] {
        let values: Vec<u16> = (0..rows * physical)
            .map(|i| f32_to_f16((i % 71) as f32 / 32.0 - 1.0))
            .collect();
        let bias: Vec<u16> = (0..logical)
            .map(|i| f32_to_f16((i % 13) as f32 / 17.0))
            .collect();
        let b = upload(&bias, &stream);
        for biased in [false, true] {
            let src = upload(&values, &stream);
            let dst = upload(&vec![0x7bffu16; rows * logical + 16], &stream);
            let bp = if biased {
                b.addr() as *const c_void
            } else {
                std::ptr::null()
            };
            assert_eq!(
                // SAFETY: buffers are disjoint and sized for the supplied geometry.
                unsafe {
                    ffi::warp34_linear_epilogue_f16(
                        &ex,
                        src.addr() as *const c_void,
                        bp,
                        dst.addr() as *mut c_void,
                        rows as i32,
                        logical as i32,
                        physical as i32,
                    )
                },
                0
            );
            stream.synchronize().unwrap();
            let graph = warp34_host::cuda::CudaGraph::capture(&stream, || {
                assert_eq!(
                    // SAFETY: same live disjoint buffers and geometry as the eager launch.
                    unsafe {
                        ffi::warp34_linear_epilogue_f16(
                            &ex,
                            src.addr() as *const c_void,
                            bp,
                            dst.addr() as *mut c_void,
                            rows as i32,
                            logical as i32,
                            physical as i32,
                        )
                    },
                    0
                );
                Ok(())
            })
            .unwrap();
            graph.launch(&stream).unwrap();
            graph.launch(&stream).unwrap();
            let expected: Vec<u16> = (0..rows * logical)
                .map(|i| {
                    let v = values[i / logical * physical + i % logical];
                    if biased {
                        f32_to_f16(f16_to_f32(v) + f16_to_f32(bias[i % logical]))
                    } else {
                        v
                    }
                })
                .collect();
            let mut got = vec![0u16; rows * logical + 16];
            dst.copy_to_host(&mut got, &stream).unwrap();
            assert_eq!(&got[..rows * logical], expected);
            assert_eq!(&got[rows * logical..], &[0x7bff; 16]);
            // SAFETY: equal widths allow aliasing; unequal widths must be rejected before launch.
            let rc = unsafe {
                ffi::warp34_linear_epilogue_f16(
                    &ex,
                    src.addr() as *const c_void,
                    bp,
                    src.addr() as *mut c_void,
                    rows as i32,
                    logical as i32,
                    physical as i32,
                )
            };
            if logical == physical {
                assert_eq!(rc, 0);
                let mut got = vec![0u16; rows * logical];
                src.copy_to_host(&mut got, &stream).unwrap();
                assert_eq!(got, expected);
            } else {
                assert_ne!(rc, 0);
            }
        }
    }
    // SAFETY: empty work permits null pointers; invalid inputs must not launch.
    unsafe {
        assert_eq!(
            ffi::warp34_linear_epilogue_f16(
                &ex,
                std::ptr::null(),
                std::ptr::null(),
                std::ptr::null_mut(),
                0,
                5,
                8
            ),
            0
        );
        for (rows, logical, physical) in [(1, 5, 8), (-1, 5, 8), (0, 9, 8), (0, 0, 8)] {
            assert_ne!(
                ffi::warp34_linear_epilogue_f16(
                    &ex,
                    std::ptr::null(),
                    std::ptr::null(),
                    std::ptr::null_mut(),
                    rows,
                    logical,
                    physical
                ),
                0
            );
        }
    }
}

fn round_half_away_i8(v: f32) -> i8 {
    let r = if v >= 0.0 {
        (v + 0.5).floor()
    } else {
        (v - 0.5).ceil()
    };
    r.clamp(-127.0, 127.0) as i8
}

fn host_per_token_quant_i8(a_f16: &[u16], m: usize, k: usize) -> (Vec<i8>, Vec<f32>) {
    let mut q = vec![0i8; m * k];
    let mut scales = vec![0.0f32; m];
    for row in 0..m {
        let xr = &a_f16[row * k..(row + 1) * k];
        let absmax = xr
            .iter()
            .map(|&bits| f16_to_f32(bits).abs())
            .fold(0.0f32, f32::max)
            .max(1e-10);
        let inv = 127.0f32 / absmax;
        scales[row] = absmax / 127.0f32;
        for col in 0..k {
            q[row * k + col] = round_half_away_i8(f16_to_f32(xr[col]) * inv);
        }
    }
    (q, scales)
}

#[test]
fn per_token_quant_i8_matches_ieee_round_half_away_and_zero_floor() {
    let _guard = GPU_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    let _dev = CudaDevice::set_current(0).unwrap();
    let stream = CudaStream::new(0).unwrap();
    let ex = stream.exec(0, 0);

    for (m, k) in [(5usize, 256usize), (3usize, 37usize)] {
        let mut x_f16 = vec![0u16; m * k];
        for row in 0..m {
            if row == 1 {
                // All-zero row exercises 1e-10f absmax floor.
                continue;
            }
            for col in 0..k {
                let raw = if row == 2 && col < 4 {
                    // Exact half-way tie values when absmax = 127.0 (`inv = 1.0`).
                    match col {
                        0 => 127.0,
                        1 => -127.0,
                        2 => 0.5,
                        _ => -0.5,
                    }
                } else {
                    let s = ((row * 31 + col * 17) % 113) as f32 - 56.0;
                    s * 0.03125
                };
                x_f16[row * k + col] = f32_to_f16(raw);
            }
        }
        let (exp_q, exp_s) = host_per_token_quant_i8(&x_f16, m, k);
        let d_x = upload(&x_f16, &stream);
        let d_q = DeviceBuffer::alloc(0, m * k).unwrap();
        let d_s = DeviceBuffer::alloc(0, m * 4).unwrap();
        // SAFETY: buffers match (m, k) extents.
        let rc = unsafe {
            ffi::warp34_per_token_quant_i8(
                &ex,
                d_x.addr() as *const c_void,
                d_q.addr() as *mut c_void,
                d_s.addr() as *mut f32,
                m as i64,
                k as i64,
            )
        };
        assert_eq!(rc, 0, "per_token_quant_i8 m={m} k={k}");
        let mut got_q = vec![0i8; m * k];
        let mut got_s = vec![0.0f32; m];
        d_q.copy_to_host(&mut got_q, &stream).unwrap();
        d_s.copy_to_host(&mut got_s, &stream).unwrap();
        stream.synchronize().unwrap();
        assert_eq!(got_q, exp_q, "quantized int8 mismatch for m={m} k={k}");
        for r in 0..m {
            let ds = (got_s[r] - exp_s[r]).abs();
            assert!(
                ds <= exp_s[r].abs() * 1e-6 + 1e-16,
                "scale mismatch at row {r}: got {}, expected {}",
                got_s[r],
                exp_s[r]
            );
        }
    }
}

#[test]
fn marlin_w4a8_shared_k16_matches_exact_s8_reference_and_mixed_sign_scales() {
    let _guard = GPU_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    let _dev = CudaDevice::set_current(0).unwrap();
    let stream = CudaStream::new(0).unwrap();
    let workspace = DeviceBuffer::alloc(0, 65536 * 4).unwrap();
    let scratch = DeviceBuffer::alloc(0, 8 * 1024 * 1024).unwrap();
    let ex_s8 = stream.exec(scratch.addr(), scratch.len_bytes);
    let ex_f16 = stream.exec(0, 0);

    // Cover `{64, 256, 256}`, `{64, 128, 128}`, and Split-K `{128, 64, 128}` across
    // g32, g64, g128, and channelwise (-1) with mixed positive & negative FP16 scales
    // and full nibble extremes `q = 0 (-8)` .. `q = 15 (+7)`.
    for (n, k, groups) in [
        (256usize, 256usize, &[32usize, 64, 128, 256][..]),
        (128usize, 256usize, &[32usize, 128][..]),
        (64usize, 1024usize, &[32usize, 64, 128, 1024][..]),
    ] {
        for &group in groups {
            let num_groups = k / group;
            let mut packed = vec![0u32; n * (k / 8)];
            let mut scales_nm = vec![0u16; n * num_groups];
            let mut w_q = vec![0i8; n * k];
            let mut w_s = vec![0.0f32; n * num_groups];
            for row in 0..n {
                for g in 0..num_groups {
                    let sign = if (row * 5 + g * 3) % 2 == 0 {
                        1.0f32
                    } else {
                        -1.0f32
                    };
                    let mag = (1 + (row * 7 + g * 11) % 5) as f32 / 128.0;
                    let s_bits = f32_to_f16(sign * mag);
                    scales_nm[row * num_groups + g] = s_bits;
                    w_s[row * num_groups + g] = f16_to_f32(s_bits);
                    for col in g * group..(g + 1) * group {
                        let q = ((row * 73 + col * 31 + (row ^ col) * 13) ^ (col >> 2)) & 0xF;
                        packed[row * (k / 8) + col / 8] |= (q as u32) << ((col % 8) * 4);
                        w_q[row * k + col] = (q as i32 - 8) as i8;
                    }
                }
            }
            let d_packed = upload(&packed, &stream);
            let d_repacked = DeviceBuffer::alloc(0, packed.len() * 4).unwrap();
            let perm_group = if group == k && group > 128 { 0 } else { group };
            let d_scales = upload(
                &marlin_permute_scales_from_n_major(&scales_nm, k, n, perm_group),
                &stream,
            );
            // SAFETY: packed and repacked buffers match `(k, n)`.
            let rc = unsafe {
                ffi::warp34_marlin_repack_int4(
                    &ex_f16,
                    d_packed.addr() as *const u32,
                    d_repacked.addr() as *mut u32,
                    k as i64,
                    n as i64,
                )
            };
            assert_eq!(rc, 0);

            let gb = if group == k && group > 128 {
                -1
            } else {
                (group / 16) as i32
            };
            for m in [16usize, 20, 48, 64, 100] {
                let label = format!("s8u4_shared N={n} K={k} group={group} M={m}");
                let a: Vec<u16> = (0..m * k)
                    .map(|i| f32_to_f16(((i * 19 + (i / k) * 11) % 83) as f32 / 64.0 - 0.64))
                    .collect();
                let (a_q, a_s) = host_per_token_quant_i8(&a, m, k);
                let mut exact_s8_ref = vec![0.0f32; m * n];
                let mut fp16_ref = vec![0.0f32; m * n];
                for r in 0..m {
                    for c in 0..n {
                        let mut acc_s8 = 0.0f32;
                        let mut acc_f16 = 0.0f32;
                        for g in 0..num_groups {
                            let mut grp_i32 = 0i32;
                            let ws = w_s[c * num_groups + g];
                            for j in g * group..(g + 1) * group {
                                grp_i32 += i32::from(a_q[r * k + j]) * i32::from(w_q[c * k + j]);
                                acc_f16 +=
                                    f16_to_f32(a[r * k + j]) * (f32::from(w_q[c * k + j]) * ws);
                            }
                            acc_s8 += (grp_i32 as f32) * ws;
                        }
                        exact_s8_ref[r * n + c] = acc_s8 * a_s[r];
                        fp16_ref[r * n + c] = acc_f16;
                    }
                }

                let d_a = upload(&a, &stream);
                let d_out = DeviceBuffer::alloc(0, m * n * 2).unwrap();
                workspace.zero_async(&stream).unwrap();
                // SAFETY: all device buffers and scratch workspace match `(m, n, k)`.
                let rc = unsafe {
                    ffi::warp34_marlin_gemm_s8u4_shared(
                        &ex_s8,
                        d_a.addr() as *const c_void,
                        d_repacked.addr() as *const c_void,
                        d_scales.addr() as *const c_void,
                        d_out.addr() as *mut c_void,
                        workspace.addr() as *mut i32,
                        m as i64,
                        n as i64,
                        k as i64,
                        65536,
                        gb,
                    )
                };
                assert_eq!(rc, 0, "{label}");
                let mut got = vec![0u16; m * n];
                d_out.copy_to_host(&mut got, &stream).unwrap();
                stream.synchronize().unwrap();
                assert_close(&got, &exact_s8_ref, &label);

                let mut err_vs_f16 = 0.0f64;
                let mut nrg_f16 = 0.0f64;
                for (&g_bits, &ref_v) in got.iter().zip(fp16_ref.iter()) {
                    let dv = f64::from(f16_to_f32(g_bits) - ref_v);
                    err_vs_f16 += dv * dv;
                    nrg_f16 += f64::from(ref_v) * f64::from(ref_v);
                }
                let rel_vs_f16 = (err_vs_f16 / nrg_f16.max(1e-30)).sqrt();
                assert!(
                    rel_vs_f16 < 0.012,
                    "{label}: W4A8 vs W4A16 rel_l2={rel_vs_f16} exceeded 1.2%"
                );
            }
        }
    }
}

#[test]
fn marlin_w4a8_prefill_flag_gates_m64_and_protects_overlapping_scratch() {
    let _guard = GPU_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    let dev = CudaDevice::set_current(0).unwrap();
    let props = dev.props().unwrap();
    let stream = CudaStream::new(0).unwrap();
    let (n, k, group) = (128usize, 256usize, 128usize);
    let w = weights(n, k, group, 4, false, 7);
    let d_packed = upload(&w.packed, &stream);
    let d_repacked = DeviceBuffer::alloc(0, w.packed.len() * 4).unwrap();
    let d_scales = upload(
        &marlin_permute_scales_from_n_major(&w.scales, k, n, group),
        &stream,
    );
    let ex0 = stream.exec(0, 0);
    assert_eq!(
        // SAFETY: valid device buffers.
        unsafe {
            ffi::warp34_marlin_repack_int4(
                &ex0,
                d_packed.addr() as *const u32,
                d_repacked.addr() as *mut u32,
                k as i64,
                n as i64,
            )
        },
        0
    );

    let arena = DeviceBuffer::alloc(0, 4 * 1024 * 1024).unwrap();
    let workspace = DeviceBuffer::alloc(0, 65536 * 4).unwrap();

    // 1. M = 64 with ALLOW_W4A8_PREFILL must remain 0-ULP identical to W4A16 (decode/verify/micro-chunk invariant).
    let m_small = 64usize;
    let a_small: Vec<u16> = (0..m_small * k)
        .map(|i| f32_to_f16(((i * 23 + 5) % 67) as f32 / 64.0 - 0.5))
        .collect();
    let d_a_small = upload(&a_small, &stream);
    let d_ref_small = DeviceBuffer::alloc(0, m_small * n * 2).unwrap();
    let d_got_small = DeviceBuffer::alloc(0, m_small * n * 2).unwrap();
    let mut ex_allow = stream.exec(arena.addr(), arena.len_bytes);
    ex_allow.flags = ffi::WARP34_EXEC_FLAG_ALLOW_W4A8_PREFILL;
    workspace.zero_async(&stream).unwrap();
    assert_eq!(
        // SAFETY: valid device buffers.
        unsafe {
            ffi::warp34_marlin_gemm_f16u4(
                &ex0,
                d_a_small.addr() as *const c_void,
                d_repacked.addr() as *const c_void,
                d_scales.addr() as *const c_void,
                d_ref_small.addr() as *mut c_void,
                workspace.addr() as *mut i32,
                m_small as i64,
                n as i64,
                k as i64,
                65536,
                8,
            )
        },
        0
    );
    workspace.zero_async(&stream).unwrap();
    assert_eq!(
        // SAFETY: valid device buffers.
        unsafe {
            ffi::warp34_marlin_gemm_f16u4(
                &ex_allow,
                d_a_small.addr() as *const c_void,
                d_repacked.addr() as *const c_void,
                d_scales.addr() as *const c_void,
                d_got_small.addr() as *mut c_void,
                workspace.addr() as *mut i32,
                m_small as i64,
                n as i64,
                k as i64,
                65536,
                8,
            )
        },
        0
    );
    let mut ref_small = vec![0u16; m_small * n];
    let mut got_small = vec![0u16; m_small * n];
    d_ref_small.copy_to_host(&mut ref_small, &stream).unwrap();
    d_got_small.copy_to_host(&mut got_small, &stream).unwrap();
    stream.synchronize().unwrap();
    assert_eq!(
        got_small, ref_small,
        "M=64 must stay 0-ULP W4A16 even when ALLOW_W4A8_PREFILL is set"
    );

    // 2. M = 164 (128 W4A8 + 36 W4A16 tail on sm_86/sm_89) with `A` placed AT `arena.addr()`
    //    (exercising in-scratch overlap protection) and `flags = 0` staying 0-ULP W4A16.
    let m_mixed = 164usize;
    let a_mixed: Vec<u16> = (0..m_mixed * k)
        .map(|i| f32_to_f16(((i * 29 + 3) % 71) as f32 / 64.0 - 0.55))
        .collect();
    arena.copy_from_host(&a_mixed, &stream).unwrap();
    let d_ref_mixed = DeviceBuffer::alloc(0, m_mixed * n * 2).unwrap();
    let d_noflag_mixed = DeviceBuffer::alloc(0, m_mixed * n * 2).unwrap();
    let d_flag_mixed = DeviceBuffer::alloc(0, m_mixed * n * 2).unwrap();
    let ex_noflag = stream.exec(arena.addr(), arena.len_bytes);
    workspace.zero_async(&stream).unwrap();
    assert_eq!(
        // SAFETY: valid device buffers.
        unsafe {
            ffi::warp34_marlin_gemm_f16u4(
                &ex0,
                arena.addr() as *const c_void,
                d_repacked.addr() as *const c_void,
                d_scales.addr() as *const c_void,
                d_ref_mixed.addr() as *mut c_void,
                workspace.addr() as *mut i32,
                m_mixed as i64,
                n as i64,
                k as i64,
                65536,
                8,
            )
        },
        0
    );
    workspace.zero_async(&stream).unwrap();
    assert_eq!(
        // SAFETY: valid device buffers.
        unsafe {
            ffi::warp34_marlin_gemm_f16u4(
                &ex_noflag,
                arena.addr() as *const c_void,
                d_repacked.addr() as *const c_void,
                d_scales.addr() as *const c_void,
                d_noflag_mixed.addr() as *mut c_void,
                workspace.addr() as *mut i32,
                m_mixed as i64,
                n as i64,
                k as i64,
                65536,
                8,
            )
        },
        0
    );
    workspace.zero_async(&stream).unwrap();
    assert_eq!(
        // SAFETY: valid device buffers; `arena.addr()` overlaps `ex_allow.scratch_ptr`.
        unsafe {
            ffi::warp34_marlin_gemm_f16u4(
                &ex_allow,
                arena.addr() as *const c_void,
                d_repacked.addr() as *const c_void,
                d_scales.addr() as *const c_void,
                d_flag_mixed.addr() as *mut c_void,
                workspace.addr() as *mut i32,
                m_mixed as i64,
                n as i64,
                k as i64,
                65536,
                8,
            )
        },
        0
    );
    let mut a_after = vec![0u16; m_mixed * k];
    let mut ref_mixed = vec![0u16; m_mixed * n];
    let mut noflag_mixed = vec![0u16; m_mixed * n];
    let mut flag_mixed = vec![0u16; m_mixed * n];
    arena.copy_to_host(&mut a_after, &stream).unwrap();
    d_ref_mixed.copy_to_host(&mut ref_mixed, &stream).unwrap();
    d_noflag_mixed
        .copy_to_host(&mut noflag_mixed, &stream)
        .unwrap();
    d_flag_mixed.copy_to_host(&mut flag_mixed, &stream).unwrap();
    stream.synchronize().unwrap();

    assert_eq!(
        a_after, a_mixed,
        "A inside scratch_ptr must never be clobbered"
    );
    assert_eq!(noflag_mixed, ref_mixed, "flags=0 must stay 0-ULP W4A16");
    // Tail rows 128..164 must remain 0-ULP W4A16 on all architectures.
    assert_eq!(&flag_mixed[128 * n..], &ref_mixed[128 * n..]);
    if props.compute_major == 8 && props.compute_minor >= 6 {
        let d_s8_128 = DeviceBuffer::alloc(0, 128 * n * 2).unwrap();
        workspace.zero_async(&stream).unwrap();
        assert_eq!(
            // SAFETY: valid device buffers.
            unsafe {
                ffi::warp34_marlin_gemm_s8u4_shared(
                    &ex_allow,
                    arena.addr() as *const c_void,
                    d_repacked.addr() as *const c_void,
                    d_scales.addr() as *const c_void,
                    d_s8_128.addr() as *mut c_void,
                    workspace.addr() as *mut i32,
                    128,
                    n as i64,
                    k as i64,
                    65536,
                    8,
                )
            },
            0
        );
        let mut s8_128 = vec![0u16; 128 * n];
        d_s8_128.copy_to_host(&mut s8_128, &stream).unwrap();
        stream.synchronize().unwrap();
        assert_eq!(
            &flag_mixed[..128 * n],
            &s8_128[..],
            "sm_86/sm_89 prefill bulk rows 0..128 must execute shared W4A8"
        );
    } else {
        assert_eq!(
            flag_mixed, ref_mixed,
            "sm_80 must auto-bypass W4A8 and stay 0-ULP W4A16"
        );
    }
}
