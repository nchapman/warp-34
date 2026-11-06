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
