#![cfg(feature = "cuda")]

use std::ffi::c_void;
use w34_host::{
    cuda::{CudaDevice, CudaStream, DeviceBuffer, f16_to_f32, f32_to_f16},
    loader::{marlin_permute_scales_from_n_major, marlin_permute_zero_points_from_n_major},
};
use w34_kernels::ffi;

static GPU_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

/// Explicit opt-in: these cached artifacts total hundreds of GB. Missing models
/// fail rather than silently claiming checkpoint coverage.
#[test]
#[ignore = "requires the quantization qualification HF cache and two 24-GiB GPUs"]
fn cached_hf_quantized_and_unquantized_checkpoints() {
    use w34_host::{cache::KvFormatPolicy, model::Qwen3_5Engine};
    let _guard = GPU_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    let cases = [
        ("Qwen/Qwen3.5-4B", 1),
        ("cyankiwi/Qwen3.5-9B-AWQ-4bit", 1),
        ("cyankiwi/Qwen3.8-27B-AWQ-INT4", 2),
        ("TheUnderscore/Swift-Qwen3.8-27b-W4A16-AWQ", 2),
        ("Lorbus/Qwen3.6-27B-int4-AutoRound", 2),
        ("lued/Qwen3.8-27B-INT8-W8A16-DFlash2", 2),
        ("cyankiwi/Ornith-1.5-35B-A3B-AWQ-INT4", 2),
        (
            "llmfan46/Qwen3.6-35B-A3B-uncensored-heretic-Native-MTP-Preserved-GPTQ-Int4",
            2,
        ),
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
        let groups: &[usize] = if bits == 8 {
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
                    ffi::w34_marlin_repack_int8(
                        &ex,
                        packed.addr() as *const u32,
                        repacked.addr() as *mut u32,
                        k as i64,
                        n as i64,
                    )
                } else {
                    ffi::w34_marlin_repack_int4(
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
                        ffi::w34_marlin_gemm_f16u4_zp(
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
                        ffi::w34_marlin_gemm_f16u8(
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
                        ffi::w34_marlin_gemm_f16u4(
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
                ffi::w34_marlin_repack_int8(
                    &ex,
                    packed.addr() as *const u32,
                    repacked.addr() as *mut u32,
                    k as i64,
                    n as i64,
                )
            } else {
                ffi::w34_marlin_repack_int4(
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
                    ffi::w34_marlin_gemm_f16u4_zp(
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
                    ffi::w34_marlin_gemm_f16u8(
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
                    ffi::w34_marlin_gemm_f16u4(
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
                    ffi::w34_moe_w4a16_mlp_zp_f16(
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
