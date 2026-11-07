use super::*;

#[test]
fn test_tensor_info_num_elems() {
    let t = TensorInfo {
        name: "test.weight".into(),
        shard_file: "model.safetensors".into(),
        offset: 0,
        len_bytes: 1024,
        dtype: "F16".into(),
        shape: vec![4, 8, 16],
    };
    assert_eq!(t.num_elems(), 4 * 8 * 16);
}

#[test]
fn test_transpose_gptq_qweight_k8_n_to_n_major() {
    // size_k = 16 (k8 = 2), size_n = 3
    // input is [k8, size_n] = [2, 3]
    // row 0: [1, 2, 3]
    // row 1: [4, 5, 6]
    let input = vec![1u32, 2, 3, 4, 5, 6];
    let out = transpose_gptq_qweight_k8_n_to_n_major(&input, 16, 3);
    // output should be [size_n, k8] = [3, 2]
    // col 0: [1, 4]
    // col 1: [2, 5]
    // col 2: [3, 6]
    assert_eq!(out, vec![1, 4, 2, 5, 3, 6]);
}

#[test]
fn test_model_config_from_json_parsing() {
    let json = serde_json::json!({
        "model_type": "qwen3_5",
        "hidden_size": 2560,
        "num_hidden_layers": 32,
        "num_attention_heads": 20,
        "num_key_value_heads": 4,
        "head_dim": 128,
        "vocab_size": 152064,
        "rms_norm_eps": 1e-6,
        "rope_theta": 1000000.0,
        "layer_types": ["linear_attention", "linear_attention", "linear_attention", "full_attention"]
    });

    let cfg = ModelConfig::from_json(&json).unwrap();
    assert_eq!(cfg.model_type, "qwen3_5");
    assert_eq!(cfg.hidden_size, 2560);
    assert_eq!(cfg.num_hidden_layers, 32);
    assert_eq!(cfg.num_attention_heads, 20);
    assert_eq!(cfg.num_key_value_heads, 4);
    assert_eq!(cfg.head_dim, 128);
    assert_eq!(cfg.vocab_size, 152064);
    assert_eq!(cfg.rms_norm_eps, 1e-6);
    assert_eq!(cfg.layer_types.len(), 4);
}

#[test]
fn test_architecture_capabilities_compose_independently() {
    let mut cfg = ModelConfig::from_json(&serde_json::json!({
        "model_type": "qwen3_5",
        "hidden_size": 2560,
        "num_hidden_layers": 4,
        "num_attention_heads": 20,
        "num_key_value_heads": 4,
        "head_dim": 128,
        "vocab_size": 152064
    }))
    .unwrap();
    assert!(!cfg.uses_hyper_connection());
    assert!(!cfg.uses_qsa());
    assert!(!cfg.uses_ple());
    assert!(!cfg.uses_tiered_moe());

    cfg.hc_count = 4;
    assert!(cfg.uses_hyper_connection());
    assert_eq!(cfg.hyper_hidden_size(), 10240);
    assert!(!cfg.uses_qsa());
    assert!(!cfg.uses_ple());
    cfg.hc_count = 1;
    cfg.indexer_n_heads = 4;
    cfg.indexer_head_dim = 128;
    assert!(cfg.uses_qsa());
    assert!(!cfg.uses_hyper_connection());
    cfg.ple_embed_dim = 2560;
    assert!(!cfg.uses_ple());
    cfg.ple_layer_ids = vec![2];
    assert!(cfg.uses_ple());
    cfg.num_experts = 512;
    assert!(!cfg.uses_tiered_moe());
    assert!(!cfg.is_qwen4_exp());
    // A non-qwen4_exp model whose symmetric U4B8 routed experts exceed 20 GiB
    // automatically selects tiered host/VRAM MoE storage.
    cfg.num_hidden_layers = 36;
    cfg.moe_intermediate_size = 640;
    cfg.quant = QuantScheme::Gptq {
        bits: 4,
        group_size: 32,
        sym: true,
        desc_act: false,
    };
    assert!(cfg.uses_tiered_moe());
    cfg.num_hidden_layers = 4;
    assert!(!cfg.uses_tiered_moe());
    cfg.model_type = "qwen4_exp".into();
    assert!(cfg.uses_tiered_moe());
    assert!(!cfg.uses_hyper_connection());
}

#[test]
fn test_model_config_layer_scale_rejected() {
    let json_bool = serde_json::json!({
        "model_type": "qwen3_5",
        "hidden_size": 2560,
        "num_hidden_layers": 32,
        "num_attention_heads": 20,
        "layer_scale": true
    });
    assert!(ModelConfig::from_json(&json_bool).is_err());

    let json_num = serde_json::json!({
        "model_type": "qwen3_5",
        "hidden_size": 2560,
        "num_hidden_layers": 32,
        "num_attention_heads": 20,
        "layer_scale": 1
    });
    assert!(ModelConfig::from_json(&json_num).is_err());
}

#[test]
fn test_read_row_f32_dtypes_and_bounds() {
    use crate::cuda::{f32_to_bf16, f32_to_f16};
    let tmp = std::env::temp_dir().join(format!("w34_loader_row_{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&tmp);
    std::fs::create_dir_all(&tmp).unwrap();
    let st_path = tmp.join("model.safetensors");

    let f32_vals: [f32; 6] = [1.0, -2.5, 3.25, -4.0, 5.5, 6.75];
    let f32_bytes: Vec<u8> = f32_vals.iter().flat_map(|v| v.to_le_bytes()).collect();
    let bf16_bytes: Vec<u8> = f32_vals
        .iter()
        .flat_map(|&v| f32_to_bf16(v).to_le_bytes())
        .collect();
    let f16_bytes: Vec<u8> = f32_vals
        .iter()
        .flat_map(|&v| f32_to_f16(v).to_le_bytes())
        .collect();
    let i32_bytes: Vec<u8> = [10i32, 20, 30, 40]
        .iter()
        .flat_map(|v| v.to_le_bytes())
        .collect();

    let tensors: &[(&str, &str, &[usize], &[u8])] = &[
        ("mat_f32", "F32", &[3, 2], &f32_bytes),
        ("mat_bf16", "BF16", &[3, 2], &bf16_bytes),
        ("mat_f16", "F16", &[3, 2], &f16_bytes),
        ("vec_1d", "F32", &[6], &f32_bytes),
        ("mat_i32", "I32", &[2, 2], &i32_bytes),
        ("mat_trunc", "F32", &[3, 2], &f32_bytes[..16]),
    ];
    let mut header = serde_json::Map::new();
    let mut off = 0usize;
    for (name, dtype, shape, bytes) in tensors {
        header.insert(
            (*name).to_owned(),
            serde_json::json!({
                "dtype": dtype,
                "shape": shape,
                "data_offsets": [off, off + bytes.len()],
            }),
        );
        off += bytes.len();
    }
    let mut hdr_bytes = serde_json::to_vec(&serde_json::Value::Object(header)).unwrap();
    while !hdr_bytes.len().is_multiple_of(8) {
        hdr_bytes.push(b' ');
    }
    let mut file_bytes = (hdr_bytes.len() as u64).to_le_bytes().to_vec();
    file_bytes.extend_from_slice(&hdr_bytes);
    for (_, _, _, bytes) in tensors {
        file_bytes.extend_from_slice(bytes);
    }
    std::fs::write(&st_path, file_bytes).unwrap();

    let archive = SafetensorsArchive::open(&st_path).unwrap();
    assert_eq!(archive.read_row_f32("mat_f32", 0).unwrap(), vec![1.0, -2.5]);
    assert_eq!(
        archive.read_row_f32("mat_f32", 1).unwrap(),
        vec![3.25, -4.0]
    );
    assert_eq!(archive.read_row_f32("mat_f32", 2).unwrap(), vec![5.5, 6.75]);
    assert_eq!(
        archive.read_row_f32("mat_bf16", 1).unwrap(),
        vec![3.25, -4.0]
    );
    assert_eq!(archive.read_row_f32("mat_f16", 2).unwrap(), vec![5.5, 6.75]);

    assert!(archive.read_row_f32("mat_f32", 3).is_err());
    assert!(archive.read_row_f32("vec_1d", 0).is_err());
    assert!(archive.read_row_f32("mat_i32", 0).is_err());
    assert!(archive.read_row_f32("mat_trunc", 2).is_err());
    let _ = std::fs::remove_dir_all(&tmp);
}

fn cfg_with_quant(qcfg: serde_json::Value) -> io::Result<QuantScheme> {
    let root = serde_json::json!({
        "model_type": "qwen3_5_text",
        "hidden_size": 1024,
        "num_hidden_layers": 2,
        "num_attention_heads": 8,
        "num_key_value_heads": 2,
        "head_dim": 128,
        "vocab_size": 1024,
        "layer_types": ["linear_attention", "full_attention"],
        "quantization_config": qcfg,
    });
    ModelConfig::from_json(&root).map(|c| c.quant)
}

#[test]
fn test_parse_quant_supported_schemes() {
    for gs in [32usize, 64, 128] {
        for sym in [true, false] {
            let q = cfg_with_quant(serde_json::json!({
                "quant_method": "compressed-tensors",
                "format": "pack-quantized",
                "config_groups": {
                    "group_0": {
                        "format": "pack-quantized",
                        "targets": ["Linear"],
                        "weights": {
                            "num_bits": 4,
                            "type": "int",
                            "strategy": "group",
                            "group_size": gs,
                            "symmetric": sym,
                            "actorder": "static"
                        }
                    }
                },
                "ignore": ["lm_head"]
            }))
            .unwrap();
            assert_eq!(
                q,
                QuantScheme::CompressedTensors {
                    num_bits: 4,
                    group_size: gs,
                    symmetric: sym,
                    ignore: vec!["lm_head".into()],
                    rules: None,
                }
            );
            assert_eq!(q.expects_zero_points(), !sym);
        }
    }

    // Compressed-tensors W8A16 group & channelwise + multi-group (Linear + Embedding)
    let q_w8_chan = cfg_with_quant(serde_json::json!({
            "quant_method": "compressed-tensors",
            "format": "pack-quantized",
            "config_groups": {
                "group_0": {
                    "targets": ["Embedding"],
                    "weights": { "num_bits": 8, "type": "int", "strategy": "channel", "symmetric": true }
                },
                "group_1": {
                    "targets": ["Linear"],
                    "weights": { "num_bits": 8, "type": "int", "strategy": "channel", "symmetric": true }
                }
            }
        }))
        .unwrap();
    assert_eq!(q_w8_chan.bits(), 8);
    assert!(q_w8_chan.is_channelwise());

    // GPTQ W4A16 & W8A16 with `-:` dynamic skip overrides
    for (bits, gs) in [(4usize, 32i64), (4, 64), (4, 128), (8, 128), (8, -1)] {
        let q = cfg_with_quant(serde_json::json!({
            "quant_method": "gptq",
            "bits": bits,
            "group_size": gs,
            "sym": true,
            "desc_act": false,
            "dynamic": { "-:.*in_proj_a.*": {} }
        }))
        .unwrap();
        assert_eq!(q.bits(), bits);
        assert_eq!(q.is_channelwise(), gs == -1);
    }

    // AWQ gemm W4A16 (zero_point = true & false)
    for zp in [true, false] {
        let q = cfg_with_quant(serde_json::json!({
            "quant_method": "awq",
            "bits": 4,
            "group_size": 64,
            "zero_point": zp,
            "version": "gemm"
        }))
        .unwrap();
        assert_eq!(
            q,
            QuantScheme::Awq {
                bits: 4,
                group_size: 64,
                zero_point: zp
            }
        );
        assert!(q.is_awq_packing());
        assert_eq!(q.expects_zero_points(), zp);
    }

    // AutoRound GPTQ & AWQ packing
    let ar_gptq = cfg_with_quant(serde_json::json!({
        "quant_method": "auto-round",
        "bits": 4,
        "group_size": 128,
        "sym": true,
        "data_type": "int",
        "packing_format": "auto_round:auto_gptq"
    }))
    .unwrap();
    assert!(!ar_gptq.is_awq_packing());

    let ar_awq = cfg_with_quant(serde_json::json!({
        "quant_method": "auto-round",
        "bits": 4,
        "group_size": 32,
        "sym": false,
        "data_type": "int",
        "packing_format": "auto_round:auto_awq"
    }))
    .unwrap();
    assert!(ar_awq.is_awq_packing());
    assert!(ar_awq.expects_zero_points());
}

#[test]
fn test_parse_quant_rejects_unsupported_schemes_with_clear_diagnostics() {
    let bad_cases: &[(&str, serde_json::Value, &str)] = &[
        (
            "fp8 method",
            serde_json::json!({ "quant_method": "fp8" }),
            "unsupported quantization method 'fp8'",
        ),
        (
            "mxfp4 method",
            serde_json::json!({ "quant_method": "mxfp4" }),
            "unsupported quantization method 'mxfp4'",
        ),
        (
            "bitsandbytes method",
            serde_json::json!({ "quant_method": "bitsandbytes", "load_in_4bit": true }),
            "unsupported quantization method 'bitsandbytes'",
        ),
        (
            "ct float-quantized",
            serde_json::json!({
                "quant_method": "compressed-tensors",
                "format": "float-quantized",
                "config_groups": { "group_0": { "weights": { "num_bits": 8, "type": "float" } } }
            }),
            "float-quantized",
        ),
        (
            "ct activation quant W8A8",
            serde_json::json!({
                "quant_method": "compressed-tensors",
                "format": "pack-quantized",
                "config_groups": {
                    "group_0": {
                        "input_activations": { "num_bits": 8, "type": "int" },
                        "weights": { "num_bits": 8, "type": "int", "strategy": "channel", "symmetric": true }
                    }
                }
            }),
            "activation quantization",
        ),
        (
            "ct asymmetric 8-bit",
            serde_json::json!({
                "quant_method": "compressed-tensors",
                "format": "pack-quantized",
                "config_groups": {
                    "group_0": {
                        "weights": { "num_bits": 8, "type": "int", "strategy": "group", "group_size": 128, "symmetric": false }
                    }
                }
            }),
            "asymmetric 8-bit",
        ),
        (
            "ct dynamic actorder",
            serde_json::json!({
                "quant_method": "compressed-tensors",
                "format": "pack-quantized",
                "config_groups": {
                    "group_0": {
                        "weights": { "num_bits": 4, "type": "int", "strategy": "group", "group_size": 128, "actorder": "weight" }
                    }
                }
            }),
            "actorder",
        ),
        (
            "gptq asymmetric",
            serde_json::json!({ "quant_method": "gptq", "bits": 4, "group_size": 128, "sym": false }),
            "sym=false",
        ),
        (
            "gptq desc_act",
            serde_json::json!({ "quant_method": "gptq", "bits": 4, "group_size": 128, "sym": true, "desc_act": true }),
            "desc_act=true",
        ),
        (
            "gptq mixed dynamic bits",
            serde_json::json!({
                "quant_method": "gptq",
                "bits": 4,
                "group_size": 128,
                "sym": true,
                "dynamic": { "+:.*mlp.*": { "bits": 8 } }
            }),
            "dynamic override",
        ),
        (
            "awq gemv",
            serde_json::json!({ "quant_method": "awq", "bits": 4, "group_size": 128, "version": "gemv" }),
            "AWQ version 'gemv'",
        ),
        (
            "unsupported 4-bit group_size=16",
            serde_json::json!({ "quant_method": "awq", "bits": 4, "group_size": 16, "version": "gemm" }),
            "group_size=16",
        ),
        (
            "autoround mxfp4",
            serde_json::json!({ "quant_method": "auto-round", "bits": 4, "data_type": "mx_fp4" }),
            "data_type='mx_fp4'",
        ),
    ];

    for (label, qcfg, needle) in bad_cases {
        let err = cfg_with_quant(qcfg.clone()).expect_err(label);
        assert!(
            err.to_string().contains(needle),
            "case '{label}' error '{}' did not contain '{needle}'",
            err
        );
    }
}

#[test]
fn test_standalone_quantize_config_fallback_and_g_idx_validation() {
    let tmp = std::env::temp_dir().join(format!("w34_qcfg_fallback_{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&tmp);
    std::fs::create_dir_all(&tmp).unwrap();
    std::fs::write(
        tmp.join("config.json"),
        serde_json::to_vec(&serde_json::json!({
            "model_type": "qwen3_5_text",
            "hidden_size": 1024,
            "num_hidden_layers": 2,
            "num_attention_heads": 8,
            "num_key_value_heads": 2,
            "head_dim": 128,
            "vocab_size": 1024,
            "layer_types": ["linear_attention", "full_attention"],
        }))
        .unwrap(),
    )
    .unwrap();
    std::fs::write(
        tmp.join("quantize_config.json"),
        serde_json::to_vec(&serde_json::json!({
            "bits": 4,
            "group_size": 64,
            "zero_point": true,
            "version": "gemm"
        }))
        .unwrap(),
    )
    .unwrap();
    let cfg = ModelConfig::from_dir(&tmp).unwrap();
    assert_eq!(
        cfg.quant,
        QuantScheme::Awq {
            bits: 4,
            group_size: 64,
            zero_point: true,
        }
    );
    let _ = std::fs::remove_dir_all(&tmp);

    // Static sequential & all-zero g_idx succeed; permuted g_idx fails fast.
    let seq_g: Vec<u32> = (0..128).map(|k| (k / 32) as u32).collect();
    validate_static_gptq_g_idx(&seq_g, 128, 32, "blk.g_idx").unwrap();
    validate_static_gptq_g_idx(&[0u32; 128], 128, 32, "blk.g_idx").unwrap();
    let mut perm_g = seq_g;
    perm_g.swap(0, 32);
    let err = validate_static_gptq_g_idx(&perm_g, 128, 32, "blk.g_idx").unwrap_err();
    assert!(err.to_string().contains("non-sequential g_idx"));
}

#[test]
fn test_awq_and_u8b128_unpack_and_dequant_roundtrip() {
    const AWQ_ORDER: [usize; 8] = [0, 2, 4, 6, 1, 3, 5, 7];
    let size_k = 64usize;
    let size_n = 16usize;
    let group_size = 32usize;
    let num_groups = size_k / group_size;
    let n8 = size_n / 8;

    let mut raw_q = vec![0u8; size_k * size_n];
    for k in 0..size_k {
        for n in 0..size_n {
            raw_q[k * size_n + n] = ((k * 7 + n * 3 + 5) & 0xF) as u8;
        }
    }
    let mut raw_zp = vec![0u8; num_groups * size_n];
    for g in 0..num_groups {
        for n in 0..size_n {
            raw_zp[g * size_n + n] = ((g * 5 + n * 11 + 1) & 0xF) as u8;
        }
    }
    let mut scales_n_kg = vec![0u16; size_n * num_groups];
    for n in 0..size_n {
        for g in 0..num_groups {
            scales_n_kg[n * num_groups + g] = f32_to_f16(0.0625 * ((n + g + 1) as f32));
        }
    }

    // Pack in AutoAWQ [K, N/8] and [K/g, N/8] interleaved order
    let mut awq_qw = vec![0u32; size_k * n8];
    for k in 0..size_k {
        for np in 0..n8 {
            let mut w = 0u32;
            for (slot, &c) in AWQ_ORDER.iter().enumerate() {
                w |= (raw_q[k * size_n + np * 8 + c] as u32) << (slot * 4);
            }
            awq_qw[k * n8 + np] = w;
        }
    }
    let mut awq_qz = vec![0u32; num_groups * n8];
    for g in 0..num_groups {
        for np in 0..n8 {
            let mut w = 0u32;
            for (slot, &c) in AWQ_ORDER.iter().enumerate() {
                w |= (raw_zp[g * size_n + np * 8 + c] as u32) << (slot * 4);
            }
            awq_qz[g * n8 + np] = w;
        }
    }

    let unpacked_w = unpack_awq_qweight_k_n8_to_n_major(&awq_qw, size_k, size_n);
    let unpacked_z = unpack_awq_qzeros_kg_n8_to_n_major(&awq_qz, size_k, size_n, group_size);
    let deq = dequant_u4_zp_n_major_to_f16(
        &unpacked_w,
        &scales_n_kg,
        &unpacked_z,
        size_n,
        size_k,
        group_size,
    );
    for n in 0..size_n {
        for k in 0..size_k {
            let g = k / group_size;
            let q = raw_q[k * size_n + n] as i32;
            let z = raw_zp[g * size_n + n] as i32;
            let s = f16_to_f32(scales_n_kg[n * num_groups + g]);
            let expected = f32_to_f16(((q - z) as f32) * s);
            assert_eq!(deq[n * size_k + k], expected, "mismatch at n={n}, k={k}");
        }
    }

    // U8B128 GPTQ [K/4, N] -> [N, K/4] -> FP16 roundtrip
    let k4 = size_k / 4;
    let mut gptq_u8 = vec![0u32; k4 * size_n];
    for kw in 0..k4 {
        for n in 0..size_n {
            let mut w = 0u32;
            for b in 0..4 {
                let k = kw * 4 + b;
                let u = ((k * 13 + n * 17) & 0xFF) as u32;
                w |= u << (b * 8);
            }
            gptq_u8[kw * size_n + n] = w;
        }
    }
    let n_maj_u8 = transpose_gptq_qweight_k4_n_to_n_major(&gptq_u8, size_k, size_n);
    let deq_u8 = dequant_u8b128_n_major_to_f16(&n_maj_u8, &scales_n_kg, size_n, size_k, group_size);
    for n in 0..size_n {
        for k in 0..size_k {
            let g = k / group_size;
            let q = (((k * 13 + n * 17) & 0xFF) as i32) - 128;
            let s = f16_to_f32(scales_n_kg[n * num_groups + g]);
            assert_eq!(deq_u8[n * size_k + k], f32_to_f16((q as f32) * s));
        }
    }
}

#[test]
fn standalone_dflash_window_policy_preserves_backbone_validation() {
    let tmp = std::env::temp_dir().join(format!("w34_drafter_config_{}", std::process::id()));
    std::fs::create_dir_all(&tmp).unwrap();
    let mut config = serde_json::json!({
        "architectures": ["DFlash2DraftModel"],
        "hidden_size": 2560,
        "intermediate_size": 6912,
        "num_attention_heads": 20,
        "num_key_value_heads": 4,
        "num_hidden_layers": 28,
        "vocab_size": 152064,
        "sliding_window": 2048
    });
    let write_config = |value: &serde_json::Value| {
        std::fs::write(tmp.join("config.json"), serde_json::to_vec(value).unwrap()).unwrap();
    };
    write_config(&config);
    assert!(
        ModelConfig::from_dir(&tmp)
            .unwrap_err()
            .to_string()
            .contains("sliding_window"),
        "a declared drafter architecture must not bypass target-model validation"
    );
    ModelConfig::from_drafter_json(&config).expect("explicit drafter parsing supports windows");
    assert!(
        ModelConfig::from_json(&config).is_err(),
        "target parse remains strict"
    );

    config["hidden_act"] = serde_json::json!("gelu");
    write_config(&config);
    assert!(
        ModelConfig::from_drafter_json(&config)
            .unwrap_err()
            .to_string()
            .contains("hidden_act")
    );
    config["hidden_act"] = serde_json::json!("silu");
    config["architectures"] = serde_json::json!(["Qwen3_5ForConditionalGeneration"]);
    write_config(&config);
    assert!(
        ModelConfig::from_dir(&tmp)
            .unwrap_err()
            .to_string()
            .contains("sliding_window")
    );
    std::fs::remove_dir_all(tmp).unwrap();
}

#[test]
fn test_model_config_rejects_unsupported_rope_scaling_and_biases() {
    let base_json = serde_json::json!({
        "hidden_size": 2560,
        "intermediate_size": 6912,
        "num_attention_heads": 20,
        "num_key_value_heads": 4,
        "num_hidden_layers": 28,
        "vocab_size": 152064
    });

    // Valid base parses cleanly
    ModelConfig::from_json(&base_json).unwrap();

    // rope_scaling: default is allowed
    let mut ok_rope = base_json.clone();
    ok_rope["rope_scaling"] = serde_json::json!({ "type": "default" });
    ModelConfig::from_json(&ok_rope).unwrap();

    // rope_scaling: yarn / dynamic fails fast
    let mut bad_rope = base_json.clone();
    bad_rope["rope_scaling"] = serde_json::json!({ "type": "yarn" });
    let err = ModelConfig::from_json(&bad_rope).unwrap_err();
    assert!(
        err.to_string()
            .contains("unsupported rope_scaling type='yarn'")
    );

    // attention_bias=true fails fast
    let mut bad_bias = base_json.clone();
    bad_bias["attention_bias"] = serde_json::json!(true);
    let err = ModelConfig::from_json(&bad_bias).unwrap_err();
    assert!(err.to_string().contains("unsupported attention_bias=true"));

    // linear_key_head_dim != 128 fails fast
    let mut bad_gdn = base_json.clone();
    bad_gdn["linear_key_head_dim"] = serde_json::json!(64);
    let err = ModelConfig::from_json(&bad_gdn).unwrap_err();
    assert!(
        err.to_string()
            .contains("unsupported linear_key_head_dim=64")
    );

    // hidden_act != silu fails fast
    let mut bad_act = base_json.clone();
    bad_act["hidden_act"] = serde_json::json!("gelu");
    let err = ModelConfig::from_json(&bad_act).unwrap_err();
    assert!(err.to_string().contains("unsupported hidden_act='gelu'"));

    // sliding_window > 0 fails fast on backbones
    let mut bad_swa = base_json.clone();
    bad_swa["sliding_window"] = serde_json::json!(4096);
    let err = ModelConfig::from_json(&bad_swa).unwrap_err();
    assert!(err.to_string().contains("unsupported sliding_window"));

    // DFlash2 drafter bodies legitimately use sliding-window attention:
    // the drafter parse path must accept them (regression: the blanket
    // backbone check broke `incoai/Qwen3.8-27B-DFlash2` loading).
    let mut drafter_swa = bad_swa.clone();
    drafter_swa["use_sliding_window"] = serde_json::json!(true);
    assert!(ModelConfig::from_drafter_json(&drafter_swa).is_ok());
}

#[test]
fn test_awq_key_aliases_and_auto_round_extra_config_validation() {
    let base = serde_json::json!({
        "hidden_size": 2560,
        "intermediate_size": 6912,
        "num_attention_heads": 20,
        "num_key_value_heads": 4,
        "num_hidden_layers": 28,
        "vocab_size": 152064
    });

    let mut awq_alias = base.clone();
    awq_alias["quantization_config"] = serde_json::json!({
        "quant_method": "awq",
        "w_bit": 4,
        "q_group_size": 64,
        "zero_point": false,
        "version": "gemm"
    });
    let cfg = ModelConfig::from_json(&awq_alias).unwrap();
    assert_eq!(
        cfg.quant,
        QuantScheme::Awq {
            bits: 4,
            group_size: 64,
            zero_point: false,
        }
    );

    let mut awq_bad_bits = base.clone();
    awq_bad_bits["quantization_config"] = serde_json::json!({
        "quant_method": "awq",
        "w_bit": 8,
        "q_group_size": 128,
        "zero_point": true
    });
    assert!(ModelConfig::from_json(&awq_bad_bits).is_err());

    let mut ar_backend = base.clone();
    ar_backend["quantization_config"] = serde_json::json!({
        "quant_method": "auto-round",
        "bits": 4,
        "group_size": 64,
        "sym": false,
        "backend": "awq:marlin",
        "extra_config": {
            "lm_head": { "bits": 16, "data_type": "fp", "group_size": 0 },
            "model.layers.0.mlp.gate": { "bits": 32, "data_type": "float" }
        }
    });
    let cfg = ModelConfig::from_json(&ar_backend).unwrap();
    assert_eq!(
        cfg.quant,
        QuantScheme::AutoRound {
            bits: 4,
            group_size: 64,
            sym: false,
            awq_packing: true,
        }
    );

    for bad_ec in [
        serde_json::json!({ "model.layers.0.self_attn.q_proj": { "bits": 4, "group_size": 16 } }),
        serde_json::json!({ "model.layers.0.self_attn.q_proj": { "bits": 4, "sym": false } }),
        serde_json::json!({ "model.layers.0.self_attn.q_proj": { "bits": 4, "data_type": "fp8" } }),
        serde_json::json!({ "model.layers.0.self_attn.q_proj": { "bits": 4, "act_bits": 8 } }),
    ] {
        let mut ar_bad = base.clone();
        ar_bad["quantization_config"] = serde_json::json!({
            "quant_method": "auto-round",
            "bits": 4,
            "group_size": 128,
            "sym": true,
            "packing_format": "auto_round:auto_gptq",
            "extra_config": bad_ec
        });
        assert!(
            ModelConfig::from_json(&ar_bad).is_err(),
            "expected error for {bad_ec}"
        );
    }
}

#[test]
fn test_marlin_permute_scales_k_equals_group_size_128_uses_grouped_transpose() {
    let size_k = 128usize;
    let size_n = 64usize;
    let scales: Vec<u16> = (0..size_n as u16).collect();
    let grouped = marlin_permute_scales_k_by_n(&scales, size_k, size_n, 128);
    let channelwise = marlin_permute_scales_k_by_n(&scales, size_k, size_n, 0);
    // For 8x8 transpose (`scale_perm`), index 1 is `0 + 8 * 1 = 8`.
    // For `scale_perm_single`, index 1 is `1`.
    assert_eq!(grouped[1], 8);
    assert_eq!(channelwise[1], 1);
}

#[test]
fn test_qwen4_exp_flash_next_config_and_validation() {
    let qfn_json: Json = serde_json::from_str(
            r#"{
            "architectures": ["Qwen4ExpForConditionalGeneration"],
            "model_type": "qwen4_exp",
            "image_token_id": 248056,
            "vision_start_token_id": 248053,
            "vision_end_token_id": 248054,
            "quantization_config": {
                "bits": 4,
                "group_size": 128,
                "packing_format": "auto_round:auto_gptq",
                "quant_method": "auto-round",
                "sym": true,
                "extra_config": {
                    "lm_head": { "bits": 16, "data_type": "fp" },
                    "model.language_model.layers.0.linear_attn.in_proj_qkv": { "bits": 16, "data_type": "fp" }
                }
            },
            "text_config": {
                "model_type": "qwen4_exp_text",
                "hidden_size": 2560,
                "num_hidden_layers": 48,
                "num_attention_heads": 24,
                "num_key_value_heads": 2,
                "head_dim": 256,
                "vocab_size": 248320,
                "rms_norm_eps": 1e-6,
                "hc_count": 4,
                "hc_lowrank": 320,
                "indexer_n_heads": 4,
                "indexer_kv_heads": 1,
                "indexer_head_dim": 128,
                "indexer_budget": 2048,
                "indexer_compress_ratio": 4,
                "linear_num_key_heads": 16,
                "linear_num_value_heads": 48,
                "linear_key_head_dim": 128,
                "linear_value_head_dim": 128,
                "linear_conv_kernel_dim": 4,
                "num_experts": 512,
                "num_experts_per_tok": 10,
                "moe_intermediate_size": 640,
                "shared_expert_intermediate_size": 640,
                "ple_layer_ids": [2],
                "ple_embed_dim": 2560,
                "ple_conv_kernel_size": 4,
                "ngram_size": 3,
                "heads_per_ngram": 8,
                "ngram_vocab_size_base": 20000000,
                "make_ngram_vocab_size_divisible_by": 128,
                "split_ngram_parts": 128,
                "ple_embedding_dtype": "float8_e4m3fn",
                "mtp_num_hidden_layers": 1,
                "mtp_use_dedicated_embeddings": false
            }
        }"#,
        )
        .unwrap();
    let cfg = ModelConfig::from_json(&qfn_json).unwrap();
    assert!(cfg.is_qwen4_exp());
    assert!(cfg.uses_hyper_connection());
    assert_eq!(cfg.hc_count, 4);
    assert_eq!(cfg.hc_lowrank, 320);
    assert_eq!(cfg.hyper_hidden_size(), 10240);
    assert!(cfg.uses_qsa());
    assert_eq!(cfg.indexer_n_heads, 4);
    assert_eq!(cfg.indexer_kv_heads, 1);
    assert_eq!(cfg.indexer_head_dim, 128);
    assert_eq!(cfg.indexer_budget, 2048);
    assert_eq!(cfg.indexer_compress_ratio, 4);
    assert_eq!(cfg.qsa_output_width(), 2051);
    assert_eq!(cfg.ple_layer_indices(), vec![1]);
    assert_eq!(cfg.num_ple_layers(), 1);
    assert_eq!(cfg.ple_dense_layer_index(0), None);
    assert_eq!(cfg.ple_dense_layer_index(1), Some(0));
    assert_eq!(cfg.ple_conv_state_len(), 9);
    assert_eq!(cfg.num_full_attention_layers(), 12);
    assert_eq!(cfg.num_linear_attention_layers(), 36);
    assert_eq!(cfg.num_experts, 512);
    assert_eq!(cfg.num_experts_per_tok, 10);
    assert!(!cfg.mtp_use_dedicated_embeddings);

    // Fail-fast on out-of-range ple_layer_ids or bad QSA indexer geometry
    let mut bad_ple = qfn_json.clone();
    bad_ple["text_config"]["ple_layer_ids"] = serde_json::json!([0]);
    assert!(ModelConfig::from_json(&bad_ple).is_err());

    let mut bad_qsa = qfn_json.clone();
    bad_qsa["text_config"]["indexer_compress_ratio"] = serde_json::json!(3);
    assert!(ModelConfig::from_json(&bad_qsa).is_err());

    verify_qwen4_exp_hgptq_config(&qfn_json);
}

fn verify_qwen4_exp_hgptq_config(qfn_json: &Json) {
    // Verify Qwen3.8-Flash-Next (qwen4_exp) W4A16-HGPTQ compressed-tensors config with
    // H_64 + g64 on attention/GDN and g32 on routed MoE experts resolves tiered MoE layout.
    let mut qfn_hgptq = qfn_json.clone();
    qfn_hgptq["quantization_config"] = serde_json::json!({
        "quant_method": "compressed-tensors",
        "format": "pack-quantized",
        "ignore": [
            "lm_head",
            "re:.*mlp\\.gate$",
            "re:.*mlp\\.shared_expert_gate$",
            "re:.*indexer\\..*",
            "re:.*hyper_connection.*"
        ],
        "config_groups": {
            "group_0": {
                "targets": ["re:.*self_attn\\..*", "re:.*linear_attn\\.in_proj_qkv$", "re:.*linear_attn\\.in_proj_z$", "re:.*linear_attn\\.out_proj$"],
                "weights": { "num_bits": 4, "type": "int", "symmetric": true, "strategy": "group", "group_size": 64, "dynamic": false }
            },
            "group_1": {
                "targets": ["re:.*mlp\\.experts\\..*", "re:.*mlp\\.shared_expert\\..*"],
                "weights": { "num_bits": 4, "type": "int", "symmetric": true, "strategy": "group", "group_size": 32, "dynamic": false }
            }
        },
        "transform_config": {
            "config_groups": {
                "h64": {
                    "type": "hadamard",
                    "head_dim": 64,
                    "randomize": false,
                    "apply": [
                        {
                            "location": "input",
                            "inverse": false,
                            "targets": ["re:.*self_attn\\..*", "re:.*linear_attn\\.in_proj_qkv$", "re:.*linear_attn\\.in_proj_z$", "re:.*linear_attn\\.out_proj$"],
                            "ignore": []
                        },
                        {
                            "location": "weight_input",
                            "inverse": true,
                            "targets": ["re:.*self_attn\\..*", "re:.*linear_attn\\.in_proj_qkv$", "re:.*linear_attn\\.in_proj_z$", "re:.*linear_attn\\.out_proj$"],
                            "ignore": []
                        }
                    ]
                }
            }
        }
    });
    let cfg_hgptq = ModelConfig::from_json(&qfn_hgptq).unwrap();
    assert_eq!(
        cfg_hgptq
            .routed_expert_quant_scheme()
            .symmetric_gptq_u4_group_size(),
        Some(32)
    );
    assert_eq!(cfg_hgptq.moe_expert_pack_layout().unwrap().group_size, 32);
    assert!(cfg_hgptq.uses_tiered_moe());
    assert_eq!(
        cfg_hgptq
            .quant
            .input_hadamard("model.language_model.layers.0.linear_attn.in_proj_qkv")
            .unwrap(),
        Some(64)
    );
    assert_eq!(
        cfg_hgptq
            .quant
            .input_hadamard("model.language_model.layers.0.mlp.experts.0.gate_proj")
            .unwrap(),
        None
    );
}

#[test]
fn test_safetensors_archive_rejects_truncated_and_out_of_bounds_headers() {
    use std::io::Write;
    let tmp = std::env::temp_dir().join(format!("w34_bad_st_{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&tmp);
    std::fs::create_dir_all(&tmp).unwrap();

    // 1. Truncated file (< 8 bytes)
    let p_short = tmp.join("model.safetensors");
    std::fs::write(&p_short, [1u8, 2, 3]).unwrap();
    assert!(SafetensorsArchive::open(&tmp).is_err());

    // 2. Header length exceeds file length
    let mut f = std::fs::File::create(&p_short).unwrap();
    f.write_all(&1024u64.to_le_bytes()).unwrap();
    f.write_all(b"{}").unwrap();
    drop(f);
    assert!(SafetensorsArchive::open(&tmp).is_err());

    // 3. Tensor data_offsets out of bounds of data section
    let hdr = br#"{"w":{"dtype":"F16","shape":[4,4],"data_offsets":[0,32]}}"#;
    let mut f = std::fs::File::create(&p_short).unwrap();
    f.write_all(&(hdr.len() as u64).to_le_bytes()).unwrap();
    f.write_all(hdr).unwrap();
    f.write_all(&[0u8; 16]).unwrap(); // only 16 bytes instead of 32
    drop(f);
    assert!(SafetensorsArchive::open(&tmp).is_err());

    let _ = std::fs::remove_dir_all(&tmp);
}
