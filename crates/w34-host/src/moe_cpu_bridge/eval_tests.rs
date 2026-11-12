use super::*;

#[test]
fn permutation_and_sequential_sum() {
    let mut x = [1.0f32; 256];
    x[0] = 1e20;
    x[1] = -1e20;
    let mut xp = [0.0; 256];
    let mut xg = [0.0; 2];
    prepare_x_perm(&x, &mut xp, &mut xg).unwrap();
    assert_eq!(xg, [126.0, 128.0]);
    let mut xp_port = [0.0; 256];
    let mut xg_port = [0.0; 2];
    prepare_x_perm_portable(&x, &mut xp_port, &mut xg_port).unwrap();
    assert_eq!(xp, xp_port);
    assert_eq!(xg, xg_port);
    let mut xp_g64 = [0.0; 256];
    let mut xg_g64 = [0.0; 4];
    let mut xp_g64_port = [0.0; 256];
    let mut xg_g64_port = [0.0; 4];
    prepare_x_perm_grouped(&x, 64, &mut xp_g64, &mut xg_g64).unwrap();
    prepare_x_perm_grouped_portable(&x, 64, &mut xp_g64_port, &mut xg_g64_port).unwrap();
    assert_eq!(xp_g64, xp_g64_port);
    assert_eq!(xg_g64, xg_g64_port);
    let mut had_simd: Vec<f32> = (0..128)
        .map(|k| ((k * 37 % 103) as f32 - 51.0) / 16.0)
        .collect();
    let mut had_port = had_simd.clone();
    hadamard64_f32_inplace(&mut had_simd);
    hadamard64_f32_portable(&mut had_port);
    assert_eq!(
        had_simd.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
        had_port.iter().map(|v| v.to_bits()).collect::<Vec<_>>()
    );
    for g in 0..2 {
        for l in 0..16 {
            for b in 0..8 {
                assert_eq!(xp[g * 128 + b * 16 + l], x[g * 128 + l * 8 + b]);
            }
        }
    }
    assert_eq!(
        prepare_x_perm(&x[..127], &mut xp, &mut xg),
        Err(EvalError::InvalidShape)
    );
    assert_eq!(scratch_len(usize::MAX, 2), None);
}

#[test]
fn direct_fp16_preparation_matches_fp32_for_all_finite_half_values() {
    let input = (0..=u16::MAX)
        .filter(|v| v & 0x7c00 != 0x7c00)
        .collect::<Vec<_>>();
    let x = input
        .iter()
        .map(|&v| crate::cuda::f16_to_f32(v))
        .collect::<Vec<_>>();
    let mut xp = vec![0.; x.len()];
    let mut xg = vec![0.; x.len() / 128];
    prepare_x_perm(&x, &mut xp, &mut xg).unwrap();
    let mut hp = vec![0.; x.len()];
    let mut hg = vec![0.; x.len() / 128];
    prepare_x_perm_f16(&input, &mut hp, &mut hg).unwrap();
    let mut hp_port = vec![0.; x.len()];
    let mut hg_port = vec![0.; x.len() / 128];
    prepare_x_perm_f16_portable(&input, &mut hp_port, &mut hg_port).unwrap();
    assert_eq!(
        xp.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
        hp.iter().map(|v| v.to_bits()).collect::<Vec<_>>()
    );
    assert_eq!(
        hp.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
        hp_port.iter().map(|v| v.to_bits()).collect::<Vec<_>>()
    );
    assert_eq!(xg, hg);
    assert_eq!(hg, hg_port);
    let mut hp_g64 = vec![0.; x.len()];
    let mut hg_g64 = vec![0.; x.len() / 64];
    let mut hp_g64_port = vec![0.; x.len()];
    let mut hg_g64_port = vec![0.; x.len() / 64];
    prepare_x_perm_f16_grouped(&input, 64, &mut hp_g64, &mut hg_g64).unwrap();
    prepare_x_perm_f16_grouped_portable(&input, 64, &mut hp_g64_port, &mut hg_g64_port).unwrap();
    assert_eq!(
        hp_g64.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
        hp_g64_port.iter().map(|v| v.to_bits()).collect::<Vec<_>>()
    );
    assert_eq!(hg_g64, hg_g64_port);
    assert_eq!(
        prepare_x_perm_f16(&input[..127], &mut hp, &mut hg),
        Err(EvalError::InvalidShape)
    );
}

fn words(n: usize, seed: u32) -> Vec<u32> {
    let mut state = seed;
    (0..n)
        .map(|_| {
            state = state.wrapping_mul(1664525).wrapping_add(1013904223);
            state
        })
        .collect()
}

fn dot(w: &[u32], x: &[f32]) -> f32 {
    x.iter()
        .enumerate()
        .map(|(c, v)| (((w[c / 8] >> (4 * (c % 8))) & 15) as f32 - 8.0) * (1.0 / 128.0) * v)
        .sum()
}

#[test]
fn expert_scalar_and_tile_agreement() {
    let h = 256;
    let i = 256;
    let gu = words(2 * i * h / 8, 5);
    let dn = words(h * i / 8, 19);
    let gs = vec![0x2000; 2 * i * h / 128];
    let ds = vec![0x2000; h * i / 128];
    let d = ExpertLayout {
        hidden: h,
        intermediate: i,
        group_size: 128,
        bits: 4,
        down_hadamard: 0,
        gate_up_words: &gu,
        gate_up_scales: &gs,
        down_words: &dn,
        down_scales: &ds,
    };
    for nt in [1, 2, 3, 4, 5, 8, 9, 16, 17, 32, 33, 96] {
        let x: Vec<_> = (0..nt * h)
            .map(|j| ((j * 17 % 101) as f32 - 50.0) / 100.0)
            .collect();
        let wt: Vec<_> = (0..nt)
            .map(|t| if t % 2 == 0 { 0.7 } else { -0.2 })
            .collect();
        let mut xp = vec![0.0; nt * h];
        let mut xg = vec![0.0; nt * h / 128];
        for t in 0..nt {
            prepare_x_perm(
                &x[t * h..(t + 1) * h],
                &mut xp[t * h..(t + 1) * h],
                &mut xg[t * h / 128..(t + 1) * h / 128],
            )
            .unwrap();
        }
        let mut out = vec![0.0; nt * h];
        let mut scratch = vec![0.0; scratch_len(nt, 2).unwrap()];
        eval_expert_prefill(d, &xp, &xg, &wt, &mut out, &mut scratch).unwrap();
        let mut out_portable = vec![0.0; nt * h];
        expert_portable(d, 0..2, &xp, &xg, &wt, &mut out_portable, &mut scratch);
        assert_eq!(
            out.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
            out_portable.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
            "portable and SIMD prefill diverged at nt={nt}"
        );
        #[cfg(target_arch = "x86_64")]
        if supported() {
            let mut out_avx2 = vec![0.0; nt * h];
            // SAFETY: `supported()` verified AVX2 + FMA + F16C and buffer bounds above.
            unsafe {
                simd::expert_avx2(d, 0..2, [&xp, &xg], &wt, &mut out_avx2, &mut scratch);
            }
            assert_eq!(
                out.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                out_avx2.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                "AVX2 and AVX512 (tile=true) diverged at nt={nt}"
            );
            if nt <= 4
                && std::is_x86_feature_detected!("avx512f")
                && std::is_x86_feature_detected!("avx512vl")
            {
                let mut slice_avx512 = vec![0.0; nt * h];
                let mut slice_avx2 = vec![0.0; nt * h];
                // SAFETY: runtime AVX-512 + AVX2 guards and buffer bounds verified above.
                unsafe {
                    simd::expert_avx512(
                        d,
                        0..2,
                        [&xp, &xg],
                        &wt,
                        &mut slice_avx512,
                        &mut scratch,
                        false,
                    );
                    simd::expert_avx2(d, 0..2, [&xp, &xg], &wt, &mut slice_avx2, &mut scratch);
                }
                assert_eq!(
                    slice_avx512.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                    slice_avx2.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                    "AVX2 and AVX512 (tile=false) diverged at nt={nt}"
                );
            }
        }
        for t in 0..nt {
            let act: Vec<_> = (0..i)
                .map(|r| {
                    let g = dot(&gu[r * h / 8..], &x[t * h..(t + 1) * h]);
                    let u = dot(&gu[(i + r) * h / 8..], &x[t * h..(t + 1) * h]);
                    (g / (1.0 + (-g).exp())) * u
                })
                .collect();
            for r in 0..h {
                let expected = wt[t] * dot(&dn[r * i / 8..], &act);
                assert!((out[t * h + r] - expected).abs() < 2e-5, "t={t}, r={r}");
            }
        }
        if nt <= 4 {
            let mut full = vec![0.0; nt * h];
            eval_expert_slice(d, 0..2, &xp, &xg, &wt, &mut full, &mut scratch).unwrap();
            assert_eq!(full, out);
            let mut a = vec![0.0; nt * h];
            let mut b = a.clone();
            eval_expert_slice(d, 0..1, &xp, &xg, &wt, &mut a, &mut scratch).unwrap();
            eval_expert_slice(d, 1..2, &xp, &xg, &wt, &mut b, &mut scratch).unwrap();
            for j in 0..out.len() {
                assert!((a[j] + b[j] - out[j]).abs() < 2e-5);
            }
        }
    }
    let mut out = [0.0; 256];
    assert_eq!(
        eval_expert_slice(d, 0..2, &[], &[], &[1.0], &mut out, &mut []),
        Err(EvalError::InvalidShape)
    );
}

#[test]
fn expert_g64_hadamard64_scalar_and_tile_agreement() {
    let h = 256;
    let i = 256;
    let gu = words(2 * i * h / 8, 11);
    let dn = words(h * i / 8, 29);
    let gs: Vec<u16> = (0..(2 * i * h / 64))
        .map(|k| if k % 2 == 0 { 0x2000 } else { 0x2400 })
        .collect();
    let ds: Vec<u16> = (0..(h * i / 64))
        .map(|k| if k % 2 == 0 { 0x2000 } else { 0x1c00 })
        .collect();
    let dot_g64 = |w: &[u32], s: &[u16], x: &[f32]| -> f32 {
        x.iter()
            .enumerate()
            .map(|(c, v)| {
                let scale = crate::cuda::f16_to_f32(s[c / 64]);
                (((w[c / 8] >> (4 * (c % 8))) & 15) as f32 - 8.0) * scale * v
            })
            .sum()
    };
    for (group_size, down_hadamard) in [(64, 64), (64, 0), (128, 64)] {
        let d = ExpertLayout {
            hidden: h,
            intermediate: i,
            group_size,
            bits: 4,
            down_hadamard,
            gate_up_words: &gu,
            gate_up_scales: &gs,
            down_words: &dn,
            down_scales: &ds,
        };
        for nt in [1, 2, 3, 4, 5, 8, 9, 16, 17, 32, 33, 96] {
            let x_f16: Vec<u16> = (0..nt * h)
                .map(|j| crate::cuda::f32_to_f16(((j * 17 % 101) as f32 - 50.0) / 100.0))
                .collect();
            let x: Vec<f32> = x_f16.iter().map(|&b| crate::cuda::f16_to_f32(b)).collect();
            let wt: Vec<_> = (0..nt)
                .map(|t| if t % 2 == 0 { 0.7 } else { -0.2 })
                .collect();
            let mut xp = vec![0.0; nt * h];
            let mut xg = vec![0.0; nt * h / group_size];
            let mut xp_f16 = vec![0.0; nt * h];
            let mut xg_f16 = vec![0.0; nt * h / group_size];
            for t in 0..nt {
                prepare_x_perm_grouped(
                    &x[t * h..(t + 1) * h],
                    group_size,
                    &mut xp[t * h..(t + 1) * h],
                    &mut xg[t * h / group_size..(t + 1) * h / group_size],
                )
                .unwrap();
                prepare_x_perm_f16_grouped(
                    &x_f16[t * h..(t + 1) * h],
                    group_size,
                    &mut xp_f16[t * h..(t + 1) * h],
                    &mut xg_f16[t * h / group_size..(t + 1) * h / group_size],
                )
                .unwrap();
            }
            assert_eq!(xp, xp_f16);
            assert_eq!(xg, xg_f16);
            let mut out = vec![0.0; nt * h];
            let mut scratch = vec![0.0; scratch_len(nt, 2).unwrap()];
            eval_expert_prefill(d, &xp, &xg, &wt, &mut out, &mut scratch).unwrap();
            let mut out_portable = vec![0.0; nt * h];
            expert_portable(d, 0..2, &xp, &xg, &wt, &mut out_portable, &mut scratch);
            let ref_bits: Vec<u32> = out_portable.iter().map(|v| v.to_bits()).collect();
            assert_eq!(
                out.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                ref_bits,
                "gs={group_size} had={down_hadamard} portable and SIMD prefill diverged at nt={nt}"
            );
            #[cfg(target_arch = "x86_64")]
            if supported() {
                let mut out_avx2 = vec![0.0; nt * h];
                // SAFETY: `supported()` verified AVX2 + FMA + F16C and buffer bounds above.
                unsafe {
                    simd::expert_avx2(d, 0..2, [&xp, &xg], &wt, &mut out_avx2, &mut scratch);
                }
                assert_eq!(
                    out_avx2.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                    ref_bits,
                    "gs={group_size} had={down_hadamard} AVX2 diverged at nt={nt}"
                );
                if std::is_x86_feature_detected!("avx512f")
                    && std::is_x86_feature_detected!("avx512vl")
                {
                    for tile in [false, true] {
                        let mut out_512 = vec![0.0; nt * h];
                        // SAFETY: runtime AVX-512 guards and buffer bounds verified above.
                        unsafe {
                            simd::expert_avx512(
                                d,
                                0..2,
                                [&xp, &xg],
                                &wt,
                                &mut out_512,
                                &mut scratch,
                                tile,
                            );
                        }
                        assert_eq!(
                            out_512.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                            ref_bits,
                            "gs={group_size} had={down_hadamard} AVX512 tile={tile} diverged at nt={nt}"
                        );
                    }
                }
            }
            if group_size == 64 && down_hadamard == 64 && nt <= 32 {
                for t in 0..nt {
                    let mut act: Vec<_> = (0..i)
                        .map(|r| {
                            let g = dot_g64(
                                &gu[r * h / 8..],
                                &gs[r * h / 64..],
                                &x[t * h..(t + 1) * h],
                            );
                            let u = dot_g64(
                                &gu[(i + r) * h / 8..],
                                &gs[(i + r) * h / 64..],
                                &x[t * h..(t + 1) * h],
                            );
                            (g / (1.0 + (-g).exp())) * u
                        })
                        .collect();
                    hadamard64_f32_inplace(&mut act);
                    for r in 0..h {
                        let expected = wt[t] * dot_g64(&dn[r * i / 8..], &ds[r * i / 64..], &act);
                        assert!((out[t * h + r] - expected).abs() < 4e-5, "t={t}, r={r}");
                    }
                }
            }
            if nt <= 4 {
                let mut a = vec![0.0; nt * h];
                let mut b = a.clone();
                eval_expert_slice(d, 0..1, &xp, &xg, &wt, &mut a, &mut scratch).unwrap();
                eval_expert_slice(d, 1..2, &xp, &xg, &wt, &mut b, &mut scratch).unwrap();
                for j in 0..out.len() {
                    assert!((a[j] + b[j] - out[j]).abs() < 4e-5);
                }
            }
        }
    }
}

#[cfg(target_arch = "x86_64")]
#[test]
fn forced_simd_tier_throughput_and_parity() {
    assert!(supported(), "x86_64 host must support AVX2 + FMA + F16C");
    let script =
        std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../tools/check_cpu_simd.py");
    assert!(
        std::process::Command::new("python3")
            .arg(&script)
            .status()
            .expect("spawn check_cpu_simd.py")
            .success(),
        "tools/check_cpu_simd.py failed"
    );
    let (h, i, nt) = (2048, 512, 2);
    let gu: Vec<u32> = (0..2 * i * h / 8)
        .map(|k| (k as u32).wrapping_mul(0x9e37_79b9))
        .collect();
    let gs: Vec<u16> = (0..2 * i * h / 64)
        .map(|k| crate::cuda::f32_to_f16(0.01 + (k % 13) as f32 * 0.002))
        .collect();
    let dn: Vec<u32> = (0..h * i / 8)
        .map(|k| (k as u32).wrapping_mul(0x85eb_ca6b))
        .collect();
    let ds: Vec<u16> = (0..h * i / 64)
        .map(|k| crate::cuda::f32_to_f16(0.01 + (k % 11) as f32 * 0.002))
        .collect();
    let d = ExpertLayout {
        hidden: h,
        intermediate: i,
        group_size: 64,
        bits: 4,
        down_hadamard: 64,
        gate_up_words: &gu,
        gate_up_scales: &gs,
        down_words: &dn,
        down_scales: &ds,
    };
    let x: Vec<f32> = (0..nt * h)
        .map(|k| ((k % 23) as f32 - 11.0) * 0.1)
        .collect();
    let (mut xp, mut xg) = (vec![0.0; nt * h], vec![0.0; nt * h / 64]);
    for t in 0..nt {
        prepare_x_perm_grouped(
            &x[t * h..(t + 1) * h],
            64,
            &mut xp[t * h..(t + 1) * h],
            &mut xg[t * h / 64..(t + 1) * h / 64],
        )
        .unwrap();
    }
    let wt = [0.6f32, 0.4f32];
    let mut scratch = vec![0.0; scratch_len(nt, 1).unwrap()];
    let (mut out_port, mut out_avx2) = (vec![0.0; nt * h], vec![0.0; nt * h]);
    let (mut port_us, mut avx2_us) = (f64::INFINITY, f64::INFINITY);
    for _ in 0..5 {
        out_port.fill(0.0);
        let t0 = std::time::Instant::now();
        expert_portable(d, 0..1, &xp, &xg, &wt, &mut out_port, &mut scratch);
        port_us = port_us.min(t0.elapsed().as_secs_f64() * 1e6);
    }
    for _ in 0..20 {
        out_avx2.fill(0.0);
        let t1 = std::time::Instant::now();
        // SAFETY: `supported()` verified AVX2 + FMA + F16C and buffer bounds above.
        unsafe { simd::expert_avx2(d, 0..1, [&xp, &xg], &wt, &mut out_avx2, &mut scratch) };
        avx2_us = avx2_us.min(t1.elapsed().as_secs_f64() * 1e6);
    }
    let ref_bits: Vec<u32> = out_port.iter().map(|v| v.to_bits()).collect();
    assert_eq!(
        out_avx2.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
        ref_bits
    );
    #[cfg(not(debug_assertions))]
    assert!(
        avx2_us < port_us,
        "AVX2 {avx2_us:.1}us >= portable {port_us:.1}us"
    );
    if std::is_x86_feature_detected!("avx512f") && std::is_x86_feature_detected!("avx512vl") {
        let (mut out_512, mut avx512_us) = (vec![0.0; nt * h], f64::INFINITY);
        for _ in 0..20 {
            out_512.fill(0.0);
            let t2 = std::time::Instant::now();
            // SAFETY: runtime AVX-512 guards and buffer bounds verified above.
            unsafe {
                simd::expert_avx512(d, 0..1, [&xp, &xg], &wt, &mut out_512, &mut scratch, true);
            }
            avx512_us = avx512_us.min(t2.elapsed().as_secs_f64() * 1e6);
        }
        assert_eq!(
            out_512.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
            ref_bits
        );
        #[cfg(not(debug_assertions))]
        assert!(
            avx512_us < port_us,
            "AVX-512 {avx512_us:.1}us >= portable {port_us:.1}us"
        );
        eprintln!(
            "forced_simd_tier: port={port_us:.1}us avx2={avx2_us:.1}us avx512={avx512_us:.1}us"
        );
    }
}

#[test]
fn expert_u2b2_g64_hadamard64_scalar_and_tile_agreement() {
    let h = 256;
    let i = 256;
    let gu = words(2 * i * h / 16, 13);
    let dn = words(h * i / 16, 31);
    let gs: Vec<u16> = (0..(2 * i * h / 64))
        .map(|k| if k % 2 == 0 { 0x2000 } else { 0x2400 })
        .collect();
    let ds: Vec<u16> = (0..(h * i / 64))
        .map(|k| if k % 2 == 0 { 0x2000 } else { 0x1c00 })
        .collect();
    let dot_u2_g64 = |w: &[u32], s: &[u16], x: &[f32]| -> f32 {
        x.iter()
            .enumerate()
            .map(|(c, v)| {
                let scale = crate::cuda::f16_to_f32(s[c / 64]);
                (((w[c / 16] >> (2 * (c % 16))) & 3) as f32 - 2.0) * scale * v
            })
            .sum()
    };
    for (group_size, down_hadamard) in [(64, 64), (64, 0), (128, 64)] {
        let d = ExpertLayout {
            hidden: h,
            intermediate: i,
            group_size,
            bits: 2,
            down_hadamard,
            gate_up_words: &gu,
            gate_up_scales: &gs,
            down_words: &dn,
            down_scales: &ds,
        };
        for nt in [1, 2, 3, 4, 5, 8, 9, 16, 17, 32, 33, 96] {
            let x_f16: Vec<u16> = (0..nt * h)
                .map(|j| crate::cuda::f32_to_f16(((j * 17 % 101) as f32 - 50.0) / 100.0))
                .collect();
            let x: Vec<f32> = x_f16.iter().map(|&b| crate::cuda::f16_to_f32(b)).collect();
            let wt: Vec<_> = (0..nt)
                .map(|t| if t % 2 == 0 { 0.7 } else { -0.2 })
                .collect();
            let mut xp = vec![0.0; nt * h];
            let mut xg = vec![0.0; nt * h / group_size];
            for t in 0..nt {
                prepare_x_perm_grouped(
                    &x[t * h..(t + 1) * h],
                    group_size,
                    &mut xp[t * h..(t + 1) * h],
                    &mut xg[t * h / group_size..(t + 1) * h / group_size],
                )
                .unwrap();
            }
            let mut out = vec![0.0; nt * h];
            let mut scratch = vec![0.0; scratch_len(nt, 2).unwrap()];
            eval_expert_prefill(d, &xp, &xg, &wt, &mut out, &mut scratch).unwrap();
            let mut out_portable = vec![0.0; nt * h];
            expert_portable(d, 0..2, &xp, &xg, &wt, &mut out_portable, &mut scratch);
            let ref_bits: Vec<u32> = out_portable.iter().map(|v| v.to_bits()).collect();
            assert_eq!(
                out.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                ref_bits,
                "u2b2 gs={group_size} had={down_hadamard} portable and SIMD prefill diverged at nt={nt}"
            );
            #[cfg(target_arch = "x86_64")]
            if supported() {
                let mut out_avx2 = vec![0.0; nt * h];
                // SAFETY: `supported()` verified AVX2 + FMA + F16C and buffer bounds above.
                unsafe {
                    simd::expert_avx2(d, 0..2, [&xp, &xg], &wt, &mut out_avx2, &mut scratch);
                }
                assert_eq!(
                    out_avx2.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                    ref_bits,
                    "u2b2 gs={group_size} had={down_hadamard} AVX2 diverged at nt={nt}"
                );
                if std::is_x86_feature_detected!("avx512f")
                    && std::is_x86_feature_detected!("avx512vl")
                {
                    for tile in [false, true] {
                        let mut out_512 = vec![0.0; nt * h];
                        // SAFETY: runtime AVX-512 guards and buffer bounds verified above.
                        unsafe {
                            simd::expert_avx512(
                                d,
                                0..2,
                                [&xp, &xg],
                                &wt,
                                &mut out_512,
                                &mut scratch,
                                tile,
                            );
                        }
                        assert_eq!(
                            out_512.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                            ref_bits,
                            "u2b2 gs={group_size} had={down_hadamard} AVX512 tile={tile} diverged at nt={nt}"
                        );
                    }
                }
            }
            if group_size == 64 && down_hadamard == 64 && nt <= 32 {
                for t in 0..nt {
                    let mut act: Vec<_> = (0..i)
                        .map(|r| {
                            let g = dot_u2_g64(
                                &gu[r * h / 16..],
                                &gs[r * h / 64..],
                                &x[t * h..(t + 1) * h],
                            );
                            let u = dot_u2_g64(
                                &gu[(i + r) * h / 16..],
                                &gs[(i + r) * h / 64..],
                                &x[t * h..(t + 1) * h],
                            );
                            (g / (1.0 + (-g).exp())) * u
                        })
                        .collect();
                    hadamard64_f32_inplace(&mut act);
                    for r in 0..h {
                        let expected =
                            wt[t] * dot_u2_g64(&dn[r * i / 16..], &ds[r * i / 64..], &act);
                        assert!((out[t * h + r] - expected).abs() < 4e-5, "t={t}, r={r}");
                    }
                }
            }
            if nt <= 4 {
                let mut a = vec![0.0; nt * h];
                let mut b = a.clone();
                eval_expert_slice(d, 0..1, &xp, &xg, &wt, &mut a, &mut scratch).unwrap();
                eval_expert_slice(d, 1..2, &xp, &xg, &wt, &mut b, &mut scratch).unwrap();
                for j in 0..out.len() {
                    assert!((a[j] + b[j] - out[j]).abs() < 4e-5);
                }
            }
        }
    }
}
