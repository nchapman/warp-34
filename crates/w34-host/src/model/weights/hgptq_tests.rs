use super::*;

fn raw(n: usize, k: usize, group_size: usize) -> RawLinearHost {
    let mut packed = vec![0u32; n * k / 8];
    for row in 0..n {
        for col in 0..k {
            let q = ((row * 7 + col * 3 + col / 7) % 16) as u32;
            packed[row * (k / 8) + col / 8] |= q << ((col % 8) * 4);
        }
    }
    RawLinearHost::U4B8NMajor {
        packed,
        scales: (0..n * (k / group_size))
            .map(|i| f32_to_f16((i % 3 + 1) as f32 / 128.0))
            .collect(),
        n,
        k,
        group_size,
    }
}

#[test]
fn padding_preserves_weights_and_channel_scales() {
    for (n, k, group) in [(4304, 128, 64), (128, 4304, 4304)] {
        let w = raw(n, k, group);
        let original = w.to_f16_vec();
        let padded = w.pad_hadamard_weight().unwrap();
        let (pn, pk) = padded.dims();
        assert_eq!(pn % 64, 0);
        assert_eq!(pk % 128, 0);
        let dense = padded.to_f16_vec();
        for row in 0..pn {
            for col in 0..pk {
                assert_eq!(
                    dense[row * pk + col],
                    if row < n && col < k {
                        original[row * k + col]
                    } else {
                        0
                    }
                );
            }
        }
    }
}

#[test]
fn shards_and_fused_rows_preserve_transform_contract() {
    let w = RawLinearHost::Hadamard {
        inner: Box::new(raw(128, 256, 64)),
        block_size: 64,
    };
    assert!(w.slice_cols(16, 144).is_err());
    assert!(w.slice_cols(0, 0).is_err());
    let shard = w.slice_cols(64, 192).unwrap();
    assert!(matches!(
        shard,
        RawLinearHost::Hadamard { block_size: 64, .. }
    ));
    assert_eq!(shard.dims(), (128, 128));
    let fused = RawLinearHost::concat_rows(&[w.slice_rows(0, 64), w.slice_rows(64, 128)]).unwrap();
    assert!(matches!(
        fused,
        RawLinearHost::Hadamard { block_size: 64, .. }
    ));
    assert_eq!(fused.to_f16_vec(), w.to_f16_vec());
    assert!(RawLinearHost::concat_rows(&[w, raw(128, 256, 64)]).is_err());
    let channel = RawLinearHost::Hadamard {
        inner: Box::new(raw(128, 512, 512)),
        block_size: 16,
    };
    let shard = channel.slice_cols(256, 512).unwrap();
    let RawLinearHost::Hadamard { inner, .. } = shard else {
        panic!("lost transform");
    };
    assert!(matches!(
        *inner,
        RawLinearHost::U4B8NMajor {
            k: 256,
            group_size: 256,
            ..
        }
    ));
}

#[test]
fn load_with_transform_resolves_archive_names_and_rejects_mismatches() {
    let cfg_json = serde_json::json!({
        "model_type": "qwen3_5_text",
        "vocab_size": 256,
        "hidden_size": 128,
        "intermediate_size": 256,
        "num_hidden_layers": 1,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "head_dim": 64,
        "quantization_config": serde_json::from_str::<serde_json::Value>(
            include_str!("../../loader/config_rules_2b.json")
        ).unwrap()
    });
    let quant = ModelConfig::from_json(&cfg_json).unwrap().quant;
    let dir = std::env::temp_dir().join(format!(
        "w34-hgptq-load-{}-{}",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    ));
    std::fs::create_dir(&dir).unwrap();
    let write_st = |tensors: &[(&str, &str, &[usize], usize)]| -> SafetensorsArchive {
        let mut header = serde_json::Map::new();
        let mut data = Vec::new();
        for &(name, dtype, shape, elem_bytes) in tensors {
            let start = data.len();
            let elems: usize = shape.iter().product();
            data.resize(start + elems * elem_bytes, 0u8);
            header.insert(
                name.into(),
                serde_json::json!({
                    "dtype": dtype,
                    "shape": shape,
                    "data_offsets": [start, data.len()],
                }),
            );
        }
        let mut hdr = serde_json::to_vec(&header).unwrap();
        while !hdr.len().is_multiple_of(8) {
            hdr.push(b' ');
        }
        let mut bytes = (hdr.len() as u64).to_le_bytes().to_vec();
        bytes.extend(hdr);
        bytes.extend(data);
        std::fs::write(dir.join("model.safetensors"), bytes).unwrap();
        SafetensorsArchive::open(&dir).unwrap()
    };

    let archive = write_st(&[
        (
            "model.language_model.layers.0.mlp.down_proj.weight_packed",
            "I32",
            &[128, 16],
            4,
        ),
        (
            "model.language_model.layers.0.mlp.down_proj.weight_scale",
            "F16",
            &[128, 2],
            2,
        ),
        (
            "model.language_model.layers.0.linear_attn.in_proj_a.weight",
            "F16",
            &[16, 128],
            2,
        ),
        (
            "model.language_model.layers.0.mlp.up_proj.weight",
            "F16",
            &[128, 128],
            2,
        ),
        (
            "model.language_model.layers.0.mlp.gate_proj.weight_packed",
            "I32",
            &[128, 16],
            4,
        ),
        (
            "model.language_model.layers.0.mlp.gate_proj.weight_scale",
            "F16",
            &[128, 4],
            2,
        ),
    ]);
    let loaded = RawLinearHost::load(&archive, "model.layers.0.mlp.down_proj", &quant).unwrap();
    assert!(matches!(
        loaded,
        RawLinearHost::Hadamard { block_size: 64, .. }
    ));
    assert_eq!(loaded.dims(), (128, 128));

    let ignored =
        RawLinearHost::load(&archive, "model.layers.0.linear_attn.in_proj_a", &quant).unwrap();
    assert!(matches!(ignored, RawLinearHost::DenseF16 { .. }));

    // Transformed target stored as dense FP16 must fail closed.
    assert!(RawLinearHost::load(&archive, "model.layers.0.mlp.up_proj", &quant).is_err());
    // Transformed target whose scale group_size (32) disagrees with config (64) must fail closed.
    assert!(RawLinearHost::load(&archive, "model.layers.0.mlp.gate_proj", &quant).is_err());
    let _ = std::fs::remove_dir_all(&dir);
}

// Dense H from parity of bit intersections is independent of butterfly order.
#[cfg(feature = "cuda")]
fn rotate(input: &[u16], block: usize) -> Vec<u16> {
    input
        .chunks_exact(block)
        .flat_map(|chunk| {
            (0..block)
                .map(|i| {
                    let sum: f32 = chunk
                        .iter()
                        .enumerate()
                        .map(|(j, &v)| {
                            f16_to_f32(v)
                                * if (i & j).count_ones().is_multiple_of(2) {
                                    1.0
                                } else {
                                    -1.0
                                }
                        })
                        .sum();
                    f32_to_f16(sum / (block as f32).sqrt())
                })
                .collect::<Vec<_>>()
        })
        .collect()
}

#[cfg(feature = "cuda")]
#[test]
fn transformed_projection_matches_independent_reference_and_graph() {
    use crate::cuda::CudaGraph;
    let _device = CudaDevice::set_current(0).unwrap();
    let stream = CudaStream::new(0).unwrap();
    let cublas = CublasHandle::new(&stream).unwrap();
    let staging = DeviceBuffer::alloc(0, 16 * 1024 * 1024).unwrap();
    let locks = DeviceBuffer::alloc(0, 65536 * 4).unwrap();
    locks.copy_from_host(&vec![0i32; 65536], &stream).unwrap();
    for (n, k, group, block) in [
        (256, 256, 64, 64),
        (4304, 128, 64, 64),
        (128, 4304, 4304, 16),
    ] {
        let raw = raw(n, k, group);
        let weights = raw.to_f16_vec();
        let rotated = RawLinearHost::Hadamard {
            inner: Box::new(raw),
            block_size: block,
        };
        let mut arena = DeviceArena::new(0, 16 * 1024 * 1024).unwrap();
        let proj = rotated.upload(&mut arena, &stream, &staging).unwrap();
        assert_eq!((proj.size_n(), proj.size_k()), (n, k));
        let LinearProj::Hadamard { inner, .. } = &proj else {
            panic!("lost transform");
        };
        assert!(matches!(**inner, LinearProj::MarlinW4A16 { .. }));
        assert!(proj.byte_footprint() < inner.size_n() * inner.size_k());
        for m in [1, 3, 16] {
            let input: Vec<u16> = (0..m * k)
                .map(|i| f32_to_f16(((i * 13) % 71) as f32 / 32.0 - 1.0))
                .collect();
            let x = DeviceBuffer::alloc(0, input.len() * 2).unwrap();
            x.copy_from_host(&input, &stream).unwrap();
            let y = DeviceBuffer::alloc(0, m * n * 2).unwrap();
            let scratch =
                DeviceBuffer::alloc(0, proj.input_transform_workspace_bytes(m).unwrap()).unwrap();
            let ex = stream.exec(scratch.addr(), scratch.len_bytes);
            let run = || proj.forward(&ex, &cublas, locks.addr(), 65536, x.addr(), y.addr(), m);
            run().unwrap();
            let mut actual = vec![0u16; m * n];
            y.copy_to_host(&mut actual, &stream).unwrap();
            let hx = rotate(&input, block);
            let mut err = 0.0f64;
            let mut norm = 0.0f64;
            for row in 0..m {
                for col in 0..n {
                    let expected: f32 = (0..k)
                        .map(|i| f16_to_f32(hx[row * k + i]) * f16_to_f32(weights[col * k + i]))
                        .sum();
                    let got = f16_to_f32(actual[row * n + col]);
                    assert!(got.is_finite());
                    err += f64::from(got - expected).powi(2);
                    norm += f64::from(expected).powi(2);
                }
            }
            assert!(
                (err / norm.max(1e-30)).sqrt() < 0.004,
                "M={m} N={n} K={k} relL2={}",
                (err / norm).sqrt()
            );
            let graph = CudaGraph::capture(&stream, run).unwrap();
            graph.launch(&stream).unwrap();
            let mut replay = vec![0u16; m * n];
            y.copy_to_host(&mut replay, &stream).unwrap();
            assert_eq!(actual, replay);
            let mut unchanged = vec![0u16; input.len()];
            x.copy_to_host(&mut unchanged, &stream).unwrap();
            assert_eq!(unchanged, input);
            assert!(
                proj.forward(
                    &stream.exec(0, 0),
                    &cublas,
                    locks.addr(),
                    65536,
                    x.addr(),
                    y.addr(),
                    m
                )
                .is_err()
            );
        }
    }
}

#[cfg(feature = "cuda")]
#[test]
fn channelwise_hadamard_k4304_does_not_overflow_fp16_before_scale() {
    let _device = CudaDevice::set_current(0).unwrap();
    let stream = CudaStream::new(0).unwrap();
    let cublas = CublasHandle::new(&stream).unwrap();
    let staging = DeviceBuffer::alloc(0, 16 * 1024 * 1024).unwrap();
    let locks = DeviceBuffer::alloc(0, 65536 * 4).unwrap();
    locks.copy_from_host(&vec![0i32; 65536], &stream).unwrap();

    let (n, k, block) = (1152, 4304, 16);
    // Pack nibble 15 (q - 8 = +7) across all K=4304 elements with channelwise scale 0.05.
    // For constant input x = 1.0, H16(x)[0] = 4.0 and H16(x)[1..16] = 0.0, so the
    // unscaled accumulator on nibble-15 columns is (4304 / 16) * 4.0 * 7.0 * 10 = 75,320 > 65,504
    // when input = 10.0, while the true scaled output is 75,320 * 0.05 = 3,766.0 <= 65,504.
    let raw = RawLinearHost::U4B8NMajor {
        packed: vec![0xFFFF_FFFF; n * (k / 8)],
        scales: vec![f32_to_f16(0.05); n],
        n,
        k,
        group_size: k,
    };
    let weights = raw.to_f16_vec();
    let rotated = RawLinearHost::Hadamard {
        inner: Box::new(raw),
        block_size: block,
    };
    let mut arena = DeviceArena::new(0, 16 * 1024 * 1024).unwrap();
    let proj = rotated.upload(&mut arena, &stream, &staging).unwrap();
    for m in [1, 16] {
        let input = vec![f32_to_f16(10.0); m * k];
        let x = DeviceBuffer::alloc(0, input.len() * 2).unwrap();
        x.copy_from_host(&input, &stream).unwrap();
        let y = DeviceBuffer::alloc(0, m * n * 2).unwrap();
        let scratch =
            DeviceBuffer::alloc(0, proj.input_transform_workspace_bytes(m).unwrap()).unwrap();
        let ex = stream.exec(scratch.addr(), scratch.len_bytes);
        proj.forward(&ex, &cublas, locks.addr(), 65536, x.addr(), y.addr(), m)
            .unwrap();
        let mut actual = vec![0u16; m * n];
        y.copy_to_host(&mut actual, &stream).unwrap();
        let hx = rotate(&input, block);
        let expected_col0: f32 = (0..k)
            .map(|i| f16_to_f32(hx[i]) * f16_to_f32(weights[i]))
            .sum();
        assert!(expected_col0 > 3700.0 && expected_col0 < 3800.0);
        for &bits in &actual {
            let got = f16_to_f32(bits);
            assert!(
                got.is_finite() && (got - expected_col0).abs() / expected_col0 < 0.002,
                "m={m} got={got} expected={expected_col0}"
            );
        }
    }
}

#[cfg(feature = "cuda")]
#[test]
fn h64_moe_ffn_block_shares_input_rotation_and_matches_reference() {
    let _device = CudaDevice::set_current(0).unwrap();
    let stream = CudaStream::new(0).unwrap();
    let cublas = CublasHandle::new(&stream).unwrap();
    let staging = DeviceBuffer::alloc(0, 16 * 1024 * 1024).unwrap();
    let locks = DeviceBuffer::alloc(0, 65536 * 4).unwrap();
    locks.copy_from_host(&vec![0i32; 65536], &stream).unwrap();

    let m = 4usize;
    let h_dim = 128usize;
    let n_interm = 128usize;
    let sh_interm = 128usize;
    let num_experts = 4usize;
    let top_k = 2usize;
    let group_size = 64usize;

    let mut arena = DeviceArena::new(0, 32 * 1024 * 1024).unwrap();

    // Unrotated router_gate [num_experts, h_dim] and shared_expert_gate [1, h_dim].
    let router_w: Vec<u16> = (0..num_experts * h_dim)
        .map(|i| f32_to_f16(((i as f32 * 0.17).sin()) * 0.2))
        .collect();
    let sh_gate_w: Vec<u16> = (0..h_dim)
        .map(|i| f32_to_f16(((i as f32 * 0.23).cos()) * 0.2))
        .collect();
    let router_ptr = arena.upload_slice(&router_w, &stream).unwrap();
    let sh_gate_ptr = arena.upload_slice(&sh_gate_w, &stream).unwrap();

    // Shared expert gate_up [2*sh_interm, h_dim] and down [h_dim, sh_interm]:
    // build both an H_64-wrapped version and an unwrapped version for the step-by-step reference.
    let sh_gu_had = RawLinearHost::Hadamard {
        inner: Box::new(raw(2 * sh_interm, h_dim, group_size)),
        block_size: 64,
    }
    .upload(&mut arena, &stream, &staging)
    .unwrap();
    let sh_dn_had = RawLinearHost::Hadamard {
        inner: Box::new(raw(h_dim, sh_interm, group_size)),
        block_size: 64,
    }
    .upload(&mut arena, &stream, &staging)
    .unwrap();
    let sh_gu_unrot = raw(2 * sh_interm, h_dim, group_size)
        .upload(&mut arena, &stream, &staging)
        .unwrap();
    let sh_dn_unrot = raw(h_dim, sh_interm, group_size)
        .upload(&mut arena, &stream, &staging)
        .unwrap();

    // Routed experts: 4 experts of gate_up [2*n_interm, h_dim] and down [h_dim, n_interm].
    let mut gu_packed = Vec::new();
    let mut gu_scales = Vec::new();
    let mut dn_packed = Vec::new();
    let mut dn_scales = Vec::new();
    for _e in 0..num_experts {
        let RawLinearHost::U4B8NMajor {
            packed: gp,
            scales: gs,
            ..
        } = raw(2 * n_interm, h_dim, group_size)
        else {
            unreachable!()
        };
        let RawLinearHost::U4B8NMajor {
            packed: dp,
            scales: ds,
            ..
        } = raw(h_dim, n_interm, group_size)
        else {
            unreachable!()
        };
        gu_packed.extend_from_slice(&gp);
        gu_scales.extend_from_slice(&gs);
        dn_packed.extend_from_slice(&dp);
        dn_scales.extend_from_slice(&ds);
    }
    let gu_p_ptr = arena.upload_slice(&gu_packed, &stream).unwrap();
    let gu_s_ptr = arena.upload_slice(&gu_scales, &stream).unwrap();
    let dn_p_ptr = arena.upload_slice(&dn_packed, &stream).unwrap();
    let dn_s_ptr = arena.upload_slice(&dn_scales, &stream).unwrap();

    let ffn_h64 = FfnBlockWeights::Moe(Box::new(MoeLayerWeights {
        router_gate: LinearProj::DenseF16 {
            weight_ptr: router_ptr,
            size_n: num_experts,
            size_k: h_dim,
        },
        experts: MoeExpertStorage::W4A16 {
            gate_up_packed_ptr: gu_p_ptr,
            gate_up_scales_ptr: gu_s_ptr,
            down_packed_ptr: dn_p_ptr,
            down_scales_ptr: dn_s_ptr,
            group_size,
        },
        num_experts,
        top_k,
        norm_topk_prob: true,
        scoring_func: 0,
        local_moe_interm: n_interm,
        routed_input_hadamard: 64,
        routed_down_hadamard: 64,
        shared_gate_up_proj: Some(sh_gu_had),
        shared_down_proj: Some(sh_dn_had),
        shared_expert_gate: Some(LinearProj::DenseF16 {
            weight_ptr: sh_gate_ptr,
            size_n: 1,
            size_k: h_dim,
        }),
        local_shared_interm: sh_interm,
    }));

    let h_x: Vec<u16> = (0..m * h_dim)
        .map(|i| f32_to_f16(((i as f32 * 0.13).sin()) * 0.5))
        .collect();
    let h_x_rot = rotate(&h_x, 64);

    let d_normed = arena.upload_slice(&h_x, &stream).unwrap();
    let d_out = arena.alloc_aligned(m * h_dim * 2).unwrap();
    let d_router_logits = arena.alloc_aligned(m * num_experts * 2).unwrap();
    let d_topk_w = arena.alloc_aligned(m * top_k * 4).unwrap();
    let d_topk_i = arena.alloc_aligned(m * top_k * 4).unwrap();
    let d_interm = arena.alloc_aligned(m * top_k * n_interm * 2).unwrap();
    let d_sh_gu = arena.alloc_aligned(m * 2 * sh_interm * 2).unwrap();
    let d_sh_act = arena.alloc_aligned(m * sh_interm * 2).unwrap();
    let d_sh_out = arena.alloc_aligned(m * h_dim * 2).unwrap();
    let d_sh_gate = arena.alloc_aligned(m * 2).unwrap();
    let d_ws = arena.alloc_aligned(256 * 1024).unwrap();
    let ex = stream.exec(d_ws, 256 * 1024);

    ffn_h64
        .forward(
            &ex,
            &cublas,
            locks.addr(),
            65536,
            d_normed,
            d_out,
            m,
            h_dim,
            n_interm,
            0,
            0,
            d_router_logits,
            d_topk_w,
            d_topk_i,
            d_interm,
            d_sh_gu,
            d_sh_act,
            d_sh_out,
            d_sh_gate,
        )
        .unwrap();
    let mut actual = vec![0u16; m * h_dim];
    // SAFETY: `d_out` holds `m * h_dim` FP16 elements.
    unsafe {
        ffi::w34_memcpy_d2h_async(
            actual.as_mut_ptr().cast(),
            d_out as *const _,
            m * h_dim * 2,
            stream.handle,
        );
    }
    stream.synchronize().unwrap();

    // Verify `d_normed` was rotated in-place to `h_x_rot`, while router/shared_expert_gate
    // ran BEFORE the in-place rotation, by running the reference with pre-computed `d_topk_i`/`d_topk_w`/`d_sh_gate`
    // and feeding `h_x_rot` directly to `sh_gu_unrot` and `w34_moe_w4a16_mlp_ex_f16(..., 64)`.
    let mut normed_after = vec![0u16; m * h_dim];
    // SAFETY: `d_normed` holds `m * h_dim` FP16 elements.
    unsafe {
        ffi::w34_memcpy_d2h_async(
            normed_after.as_mut_ptr().cast(),
            d_normed as *const _,
            m * h_dim * 2,
            stream.handle,
        );
    }
    stream.synchronize().unwrap();
    assert_eq!(normed_after, h_x_rot);

    // Reference: compute `router_logits` and `sh_gate` from unrotated `h_x` explicitly,
    // and run `sh_gu_unrot` + `sh_dn_unrot` + `w34_moe_w4a16_mlp_ex_f16` on `h_x_rot`.
    let d_x_unrot = arena.upload_slice(&h_x, &stream).unwrap();
    let d_x_rot = arena.upload_slice(&h_x_rot, &stream).unwrap();
    let d_ref_out = arena.alloc_aligned(m * h_dim * 2).unwrap();
    LinearProj::DenseF16 {
        weight_ptr: router_ptr,
        size_n: num_experts,
        size_k: h_dim,
    }
    .forward(
        &ex,
        &cublas,
        locks.addr(),
        65536,
        d_x_unrot,
        d_router_logits,
        m,
    )
    .unwrap();
    LinearProj::DenseF16 {
        weight_ptr: sh_gate_ptr,
        size_n: 1,
        size_k: h_dim,
    }
    .forward(&ex, &cublas, locks.addr(), 65536, d_x_unrot, d_sh_gate, m)
    .unwrap();
    // SAFETY: `d_router_logits`, `d_topk_w`, `d_topk_i` are sized for `m`, `num_experts`, `top_k`.
    unsafe {
        assert_eq!(
            ffi::w34_moe_topk_router_f16(
                &ex,
                d_router_logits as *const _,
                d_topk_w as *mut f32,
                d_topk_i as *mut i32,
                m as i32,
                num_experts as i32,
                top_k as i32,
                1,
                0,
            ),
            0
        );
    }
    sh_gu_unrot
        .forward(&ex, &cublas, locks.addr(), 65536, d_x_rot, d_sh_gu, m)
        .unwrap();
    // SAFETY: `d_sh_gu` and `d_sh_act` are sized for `m * 2 * sh_interm` and `m * sh_interm`.
    unsafe {
        assert_eq!(
            ffi::w34_silu_and_mul_hadamard_f16(
                &ex,
                d_sh_gu as *const _,
                d_sh_act as *mut _,
                core::ptr::null(),
                m as i32,
                sh_interm as i32,
            ),
            0
        );
    }
    sh_dn_unrot
        .forward(&ex, &cublas, locks.addr(), 65536, d_sh_act, d_sh_out, m)
        .unwrap();
    // SAFETY: all pointers are valid allocations from `arena`.
    unsafe {
        assert_eq!(
            ffi::w34_moe_w4a16_mlp_ex_f16(
                &ex,
                d_x_rot as *const _,
                gu_p_ptr as *const u32,
                gu_s_ptr as *const _,
                dn_p_ptr as *const u32,
                dn_s_ptr as *const _,
                d_topk_i as *const i32,
                d_topk_w as *const f32,
                d_sh_out as *const _,
                d_sh_gate as *const _,
                d_interm as *mut _,
                d_ref_out as *mut _,
                m as i32,
                top_k as i32,
                num_experts as i32,
                n_interm as i32,
                h_dim as i32,
                group_size as i32,
                64,
            ),
            0
        );
    }
    let mut expected = vec![0u16; m * h_dim];
    // SAFETY: `d_ref_out` holds `m * h_dim` FP16 elements.
    unsafe {
        ffi::w34_memcpy_d2h_async(
            expected.as_mut_ptr().cast(),
            d_ref_out as *const _,
            m * h_dim * 2,
            stream.handle,
        );
    }
    stream.synchronize().unwrap();
    assert_eq!(actual, expected);

    // Also verify `pre_mlp_norm_targets` dual-output contract + guard fallbacks,
    // and verify `forward_with_prerotated_input(..., input_prerotated = true)` when
    // `d_normed` holds unrotated `h_x` and `d_out` starts with pre-rotated `h_x_rot`.
    assert_eq!(
        ffn_h64.pre_mlp_norm_targets(d_normed, d_out, &ex, m, h_dim),
        [d_normed, d_out]
    );
    assert_eq!(
        ffn_h64.pre_mlp_norm_targets(d_normed, 0, &ex, m, h_dim),
        [d_normed, 0]
    );
    assert_eq!(
        ffn_h64.pre_mlp_norm_targets(d_normed, d_normed, &ex, m, h_dim),
        [d_normed, 0]
    );
    assert_eq!(
        ffn_h64.pre_mlp_norm_targets(d_normed, d_out, &ex, m, h_dim + 16),
        [d_normed, 0]
    );
    assert_eq!(
        ffn_h64.pre_mlp_norm_targets(d_normed, d_out, &ex, m, 6208),
        [d_normed, 0]
    );

    let d_normed_prerot = arena.upload_slice(&h_x, &stream).unwrap();
    let d_out_prerot = arena.upload_slice(&h_x_rot, &stream).unwrap();
    ffn_h64
        .forward_with_prerotated_input(
            &ex,
            &cublas,
            locks.addr(),
            65536,
            d_normed_prerot,
            d_out_prerot,
            m,
            h_dim,
            n_interm,
            0,
            0,
            d_router_logits,
            d_topk_w,
            d_topk_i,
            d_interm,
            d_sh_gu,
            d_sh_act,
            d_sh_out,
            d_sh_gate,
            d_out_prerot,
        )
        .unwrap();
    let mut actual_prerot = vec![0u16; m * h_dim];
    let mut normed_prerot_after = vec![0u16; m * h_dim];
    // SAFETY: `d_out_prerot` and `d_normed_prerot` each hold `m * h_dim` FP16 elements.
    unsafe {
        ffi::w34_memcpy_d2h_async(
            actual_prerot.as_mut_ptr().cast(),
            d_out_prerot as *const _,
            m * h_dim * 2,
            stream.handle,
        );
        ffi::w34_memcpy_d2h_async(
            normed_prerot_after.as_mut_ptr().cast(),
            d_normed_prerot as *const _,
            m * h_dim * 2,
            stream.handle,
        );
    }
    stream.synchronize().unwrap();
    assert_eq!(normed_prerot_after, h_x);
    assert_eq!(actual_prerot, expected);
}

#[cfg(feature = "cuda")]
#[test]
fn marlin_w2a16_and_hadamard_w2a16_match_dilated_w4a16_zero_ulp() {
    let _device = CudaDevice::set_current(0).unwrap();
    let stream = CudaStream::new(0).unwrap();
    let cublas = CublasHandle::new(&stream).unwrap();
    let staging = DeviceBuffer::alloc(0, 16 * 1024 * 1024).unwrap();
    let locks = DeviceBuffer::alloc(0, 65536 * 4).unwrap();
    locks.copy_from_host(&vec![0i32; 65536], &stream).unwrap();

    for (n, k, group) in [
        (128, 128, 64),
        (256, 256, 64),
        (512, 1024, 64),
        (128, 512, 512),
    ] {
        let words_per_row = k / 16;
        let groups_per_row = k / group;
        let packed_u2: Vec<u32> = (0..n * words_per_row)
            .map(|i| {
                let mut w = 0u32;
                for c in 0..16 {
                    let crumb = ((i * 17 + c * 5 + 3) % 4) as u32;
                    w |= crumb << (c * 2);
                }
                w
            })
            .collect();
        let scales: Vec<u16> = (0..n * groups_per_row)
            .map(|i| f32_to_f16(((i % 19) as f32 + 1.0) / 32.0))
            .collect();
        let raw_u2 = RawLinearHost::U2B2NMajor {
            packed: packed_u2,
            scales,
            n,
            k,
            group_size: group,
        };
        let raw_u4 = raw_u2.slice_rows(0, n).into_u4b8_if_u2b2();
        assert_eq!(raw_u2.to_f16_vec(), raw_u4.to_f16_vec());

        let mut arena = DeviceArena::new(0, 16 * 1024 * 1024).unwrap();
        let proj_u2 = raw_u2.upload(&mut arena, &stream, &staging).unwrap();
        let proj_u4 = raw_u4.upload(&mut arena, &stream, &staging).unwrap();
        assert!(matches!(proj_u2, LinearProj::MarlinW2A16 { .. }));
        assert!(matches!(proj_u4, LinearProj::MarlinW4A16 { .. }));
        let scale_bytes = n * (if group == k { 1 } else { k / group }) * 2;
        assert_eq!(proj_u2.byte_footprint(), (n * k / 4) + scale_bytes);
        assert_eq!(proj_u4.byte_footprint(), (n * k / 2) + scale_bytes);

        for m in [1, 4, 8, 16, 32, 64] {
            let input: Vec<u16> = (0..m * k)
                .map(|i| f32_to_f16(((i * 11) % 53) as f32 / 16.0 - 1.5))
                .collect();
            let x = DeviceBuffer::alloc(0, input.len() * 2).unwrap();
            x.copy_from_host(&input, &stream).unwrap();
            let y_u2 = DeviceBuffer::alloc(0, m * n * 2).unwrap();
            let y_u4 = DeviceBuffer::alloc(0, m * n * 2).unwrap();
            let ex = stream.exec(0, 0);
            proj_u2
                .forward(&ex, &cublas, locks.addr(), 65536, x.addr(), y_u2.addr(), m)
                .unwrap();
            proj_u4
                .forward(&ex, &cublas, locks.addr(), 65536, x.addr(), y_u4.addr(), m)
                .unwrap();
            let mut out_u2 = vec![0u16; m * n];
            let mut out_u4 = vec![0u16; m * n];
            y_u2.copy_to_host(&mut out_u2, &stream).unwrap();
            y_u4.copy_to_host(&mut out_u4, &stream).unwrap();
            assert_eq!(
                out_u2, out_u4,
                "0-ULP mismatch for MarlinW2A16 vs dilated MarlinW4A16 at M={m}, N={n}, K={k}, group={group}"
            );
        }
    }

    // Also test unaligned Hadamard U2B2 padding + mixed U2B2+U4B8 row concatenation.
    let (n, k, group, block) = (70, 192, 64, 64);
    let words_per_row = k / 16;
    let groups_per_row = k / group;
    let raw_u2 = RawLinearHost::U2B2NMajor {
        packed: (0..n * words_per_row)
            .map(|i| 0x1b1b_1b1bu32.wrapping_add((i as u32).wrapping_mul(0x0101_0101)))
            .collect(),
        scales: (0..n * groups_per_row)
            .map(|i| f32_to_f16(((i % 13) as f32 + 1.0) / 16.0))
            .collect(),
        n,
        k,
        group_size: group,
    };
    let raw_u4 = raw_u2.slice_rows(0, n).into_u4b8_if_u2b2();
    let had_u2 = RawLinearHost::Hadamard {
        inner: Box::new(raw_u2.slice_rows(0, n)),
        block_size: block,
    };
    let had_u4 = RawLinearHost::Hadamard {
        inner: Box::new(raw_u4.slice_rows(0, n)),
        block_size: block,
    };
    let mut arena = DeviceArena::new(0, 16 * 1024 * 1024).unwrap();
    let proj_had_u2 = had_u2.upload(&mut arena, &stream, &staging).unwrap();
    let proj_had_u4 = had_u4.upload(&mut arena, &stream, &staging).unwrap();
    let LinearProj::Hadamard {
        inner: inner_u2, ..
    } = &proj_had_u2
    else {
        panic!("expected Hadamard");
    };
    assert!(matches!(**inner_u2, LinearProj::MarlinW2A16 { .. }));

    for m in [1, 5, 16] {
        let input: Vec<u16> = (0..m * k)
            .map(|i| f32_to_f16(((i * 7) % 37) as f32 / 16.0 - 1.0))
            .collect();
        let x = DeviceBuffer::alloc(0, input.len() * 2).unwrap();
        x.copy_from_host(&input, &stream).unwrap();
        let y_u2 = DeviceBuffer::alloc(0, m * n * 2).unwrap();
        let y_u4 = DeviceBuffer::alloc(0, m * n * 2).unwrap();
        let scratch =
            DeviceBuffer::alloc(0, proj_had_u2.input_transform_workspace_bytes(m).unwrap())
                .unwrap();
        let ex = stream.exec(scratch.addr(), scratch.len_bytes);
        proj_had_u2
            .forward(&ex, &cublas, locks.addr(), 65536, x.addr(), y_u2.addr(), m)
            .unwrap();
        proj_had_u4
            .forward(&ex, &cublas, locks.addr(), 65536, x.addr(), y_u4.addr(), m)
            .unwrap();
        let mut out_u2 = vec![0u16; m * n];
        let mut out_u4 = vec![0u16; m * n];
        y_u2.copy_to_host(&mut out_u2, &stream).unwrap();
        y_u4.copy_to_host(&mut out_u4, &stream).unwrap();
        assert_eq!(out_u2, out_u4, "Hadamard U2B2 vs U4B8 mismatch at M={m}");
    }

    // Mixed U2B2 + U4B8 row fusion preserves lossless U4B8 representation.
    let mixed =
        RawLinearHost::concat_rows(&[raw_u2.slice_rows(0, 32), raw_u4.slice_rows(32, 64)]).unwrap();
    assert!(matches!(
        mixed,
        RawLinearHost::U4B8NMajor { n: 64, k: 192, .. }
    ));
    assert_eq!(mixed.to_f16_vec(), raw_u2.slice_rows(0, 64).to_f16_vec());
}
