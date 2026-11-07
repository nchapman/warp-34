//! Real safetensors boundary tests for loading and TP slicing, without a GPU.
use super::*;
use crate::cuda::f32_to_f16;
use serde_json::json;
use std::io::Write;
use std::sync::atomic::{AtomicUsize, Ordering};

struct Fixture(std::path::PathBuf);
impl Fixture {
    fn new(tensors: Vec<(&str, &str, Vec<usize>, Vec<u8>)>) -> Self {
        static NEXT: AtomicUsize = AtomicUsize::new(0);
        let nanos = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let dir = std::env::temp_dir().join(format!(
            "w34-quant-{}-{}-{}",
            std::process::id(),
            nanos,
            NEXT.fetch_add(1, Ordering::Relaxed)
        ));
        std::fs::DirBuilder::new().create(&dir).unwrap();
        let mut header = serde_json::Map::new();
        let mut data = Vec::new();
        for (name, dtype, shape, bytes) in tensors {
            let start = data.len();
            data.extend(bytes);
            header.insert(
                name.into(),
                json!({"dtype": dtype, "shape": shape, "data_offsets": [start, data.len()]}),
            );
        }
        let mut header = serde_json::to_vec(&header).unwrap();
        while !header.len().is_multiple_of(8) {
            header.push(b' ');
        }
        let mut file = (header.len() as u64).to_le_bytes().to_vec();
        file.extend(header);
        file.extend(data);
        let file_path = dir.join("model.safetensors");
        let mut out = std::fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&file_path)
            .unwrap();
        out.write_all(&file).unwrap();
        Self(dir)
    }
    fn archive(&self) -> SafetensorsArchive {
        SafetensorsArchive::open(&self.0).unwrap()
    }
}
impl Drop for Fixture {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}
fn words(values: &[u32]) -> Vec<u8> {
    values.iter().flat_map(|v| v.to_le_bytes()).collect()
}
fn halves(values: &[u16]) -> Vec<u8> {
    values.iter().flat_map(|v| v.to_le_bytes()).collect()
}
fn ct(bits: usize, group_size: usize, symmetric: bool) -> QuantScheme {
    QuantScheme::CompressedTensors {
        num_bits: bits,
        group_size,
        symmetric,
        ignore: vec![],
        rules: None,
    }
}

#[test]
fn compressed_tensors_loading_and_tp_slicing_preserve_weights() {
    let (n, k) = (16usize, 256usize);
    for bits in [4usize, 8] {
        for group in [32usize, 64, 128, 256] {
            for asym in [false, true] {
                if (bits == 8 && asym) || (bits == 4 && group == 256) {
                    continue;
                }
                let ng = k / group;
                let pack = 32 / bits;
                let mask = (1usize << bits) - 1;
                let mut packed = vec![0u32; n * k / pack];
                let mut zeros = vec![0u32; n / 8 * ng];
                let mut scales = vec![0u16; n * ng];
                let mut expected = vec![0u16; n * k];
                for row in 0..n {
                    for g in 0..ng {
                        let z = if asym {
                            (row + g * 3) % 16
                        } else {
                            1 << (bits - 1)
                        };
                        let scale = (row + g + 1) as f32 / 256.0;
                        scales[row * ng + g] = f32_to_f16(scale);
                        if asym {
                            zeros[(row / 8) * ng + g] |= (z as u32) << ((row % 8) * 4);
                        }
                        for col in g * group..(g + 1) * group {
                            let q = (row * 7 + col * 13) & mask;
                            packed[row * (k / pack) + col / pack] |=
                                (q as u32) << ((col % pack) * bits);
                            expected[row * k + col] =
                                f32_to_f16((q as i32 - z as i32) as f32 * scale);
                        }
                    }
                }
                let mut tensors = vec![
                    (
                        "proj.weight_packed",
                        "I32",
                        vec![n, k / pack],
                        words(&packed),
                    ),
                    ("proj.weight_scale", "F16", vec![n, ng], halves(&scales)),
                ];
                if asym {
                    tensors.push((
                        "proj.weight_zero_point",
                        "I32",
                        vec![n / 8, ng],
                        words(&zeros),
                    ));
                }
                let fixture = Fixture::new(tensors);
                let loaded =
                    RawLinearHost::load(&fixture.archive(), "proj", &ct(bits, group, !asym))
                        .unwrap();
                assert_eq!(loaded.dims(), (n, k));
                assert_eq!(loaded.to_f16_vec(), expected);
                for rank in 0..2 {
                    assert_eq!(
                        loaded
                            .slice_rows(rank * n / 2, (rank + 1) * n / 2)
                            .to_f16_vec(),
                        expected[rank * n / 2 * k..(rank + 1) * n / 2 * k]
                    );
                    let column_reference: Vec<_> = expected
                        .chunks_exact(k)
                        .flat_map(|row| row[rank * k / 2..(rank + 1) * k / 2].iter().copied())
                        .collect();
                    assert_eq!(
                        loaded
                            .slice_cols(rank * k / 2, (rank + 1) * k / 2)
                            .unwrap()
                            .to_f16_vec(),
                        column_reference
                    );
                }
            }
        }
    }
}

#[test]
fn raw_loading_reports_missing_scales_zero_points_and_bad_g_idx() {
    let packed = (
        "proj.weight_packed",
        "I32",
        vec![16, 32],
        words(&vec![0x99999999; 16 * 32]),
    );
    let fixture = Fixture::new(vec![packed.clone()]);
    let error = RawLinearHost::load(&fixture.archive(), "proj", &ct(4, 32, false))
        .err()
        .unwrap();
    assert!(error.to_string().contains("proj.weight_scale"), "{error}");
    let scales = (
        "proj.weight_scale",
        "F16",
        vec![16, 8],
        halves(&vec![f32_to_f16(0.125); 16 * 8]),
    );
    let fixture = Fixture::new(vec![packed.clone(), scales.clone()]);
    let error = RawLinearHost::load(&fixture.archive(), "proj", &ct(4, 32, false))
        .err()
        .unwrap();
    assert!(
        error.to_string().contains("proj.weight_zero_point"),
        "{error}"
    );
    let mut g_idx: Vec<u32> = (0..256).map(|i| i / 32).collect();
    g_idx.swap(0, 32);
    let fixture = Fixture::new(vec![
        packed,
        scales,
        ("proj.weight_g_idx", "I32", vec![256], words(&g_idx)),
    ]);
    let error = RawLinearHost::load(&fixture.archive(), "proj", &ct(4, 32, true))
        .err()
        .unwrap();
    assert!(
        error.to_string().contains("non-sequential g_idx"),
        "{error}"
    );
}

#[test]
fn unquantized_f16_bf16_f32_load_to_the_same_fp16_spine() {
    let values = [-2.0f32, -0.5, 0.0, 0.25, 1.0, 4.0];
    let expected: Vec<_> = values.iter().copied().map(f32_to_f16).collect();
    for (dtype, bytes) in [
        ("F16", halves(&expected)),
        (
            "BF16",
            halves(
                &values
                    .iter()
                    .map(|v| (v.to_bits() >> 16) as u16)
                    .collect::<Vec<_>>(),
            ),
        ),
        ("F32", values.iter().flat_map(|v| v.to_le_bytes()).collect()),
    ] {
        let fixture = Fixture::new(vec![("proj.weight", dtype, vec![2, 3], bytes)]);
        let loaded =
            RawLinearHost::load(&fixture.archive(), "proj", &QuantScheme::Unquantized).unwrap();
        assert_eq!(loaded.dims(), (2, 3));
        assert_eq!(loaded.to_f16_vec(), expected, "{dtype}");
    }
}

#[test]
fn compressed_tensors_weight_shape_validates_and_resolves_embed_tokens_bits() {
    let (n, k) = (8usize, 128usize);
    let packed_u8 = vec![0x82828282u32; n * (k / 4)];
    let scales = vec![f32_to_f16(0.25); n];
    let shape_i64: Vec<u8> = [n as i64, k as i64]
        .iter()
        .flat_map(|v| v.to_le_bytes())
        .collect();
    // 8-bit embed_tokens in a model whose linear scheme is 4-bit asymmetric
    let fixture = Fixture::new(vec![
        (
            "model.embed_tokens.weight_packed",
            "I32",
            vec![n, k / 4],
            words(&packed_u8),
        ),
        (
            "model.embed_tokens.weight_scale",
            "F16",
            vec![n, 1],
            halves(&scales),
        ),
        (
            "model.embed_tokens.weight_shape",
            "I64",
            vec![2],
            shape_i64.clone(),
        ),
    ]);
    let loaded =
        RawLinearHost::load(&fixture.archive(), "model.embed_tokens", &ct(4, 128, false)).unwrap();
    assert_eq!(loaded.dims(), (n, k));
    assert_eq!(loaded.to_f16_vec(), vec![f32_to_f16(0.5); n * k]);

    // Non-embedding layer with conflicting weight_shape (8-bit tensor vs 4-bit config) fails fast
    let bad_fixture = Fixture::new(vec![
        (
            "proj.weight_packed",
            "I32",
            vec![n, k / 4],
            words(&packed_u8),
        ),
        ("proj.weight_scale", "F16", vec![n, 1], halves(&scales)),
        ("proj.weight_shape", "I64", vec![2], shape_i64),
    ]);
    let err = RawLinearHost::load(&bad_fixture.archive(), "proj", &ct(4, 128, true))
        .err()
        .unwrap();
    assert!(err.to_string().contains("weight_shape"), "{err}");

    // Truncated weight_shape (e.g. 8 bytes declared for I64 [2]) returns an error instead of panicking
    let short_fixture = Fixture::new(vec![
        (
            "proj.weight_packed",
            "I32",
            vec![n, k / 4],
            words(&packed_u8),
        ),
        ("proj.weight_scale", "F16", vec![n, 1], halves(&scales)),
        ("proj.weight_shape", "I64", vec![2], vec![0u8; 8]),
    ]);
    let err = RawLinearHost::load(&short_fixture.archive(), "proj", &ct(8, 128, true))
        .err()
        .unwrap();
    assert!(err.to_string().contains("expected 16 bytes"), "{err}");

    // Ignored / unquantized 2D BF16 .weight takes precedence over an experimental 1D U8 .weight_packed sidecar
    let bf16_val = (1.5f32.to_bits() >> 16) as u16;
    let sidecar_fixture = Fixture::new(vec![
        (
            "mtp.layers.0.mlp.gate_proj.weight",
            "BF16",
            vec![n, k],
            halves(&vec![bf16_val; n * k]),
        ),
        (
            "mtp.layers.0.mlp.gate_proj.weight_packed",
            "U8",
            vec![n * k / 2],
            vec![0u8; n * k / 2],
        ),
        (
            "mtp.layers.0.mlp.gate_proj.weight_scale",
            "U8",
            vec![n, 2],
            vec![0u8; n * 2],
        ),
    ]);
    let loaded = RawLinearHost::load(
        &sidecar_fixture.archive(),
        "mtp.layers.0.mlp.gate_proj",
        &ct(4, 128, true),
    )
    .unwrap();
    assert_eq!(loaded.dims(), (n, k));
    assert_eq!(loaded.to_f16_vec(), vec![f32_to_f16(1.5); n * k]);
}

#[cfg(feature = "cuda")]
#[test]
fn synthetic_hybrid_engine_forward_generate_and_diagnostics() {
    let h = 128usize;
    let interm = 256usize;
    let vocab = 256usize;
    let f16_vec = |n: usize, val: f32| -> Vec<u8> { halves(&vec![f32_to_f16(val); n]) };
    let f32_vec = |n: usize, val: f32| -> Vec<u8> {
        std::iter::repeat_n(val, n)
            .flat_map(f32::to_le_bytes)
            .collect()
    };

    let fixture = Fixture::new(vec![
        (
            "model.embed_tokens.weight",
            "F16",
            vec![vocab, h],
            f16_vec(vocab * h, 0.02),
        ),
        ("model.norm.weight", "F16", vec![h], f16_vec(h, 0.0)),
        (
            "lm_head.weight",
            "F16",
            vec![vocab, h],
            f16_vec(vocab * h, 0.02),
        ),
        // Layer 0: LinearAttention (GDN)
        (
            "model.layers.0.input_layernorm.weight",
            "F16",
            vec![h],
            f16_vec(h, 0.0),
        ),
        (
            "model.layers.0.post_attention_layernorm.weight",
            "F16",
            vec![h],
            f16_vec(h, 0.0),
        ),
        (
            "model.layers.0.linear_attn.in_proj_qkv.weight",
            "F16",
            vec![3 * 256, h],
            f16_vec(3 * 256 * h, 0.01),
        ),
        (
            "model.layers.0.linear_attn.in_proj_z.weight",
            "F16",
            vec![256, h],
            f16_vec(256 * h, 0.01),
        ),
        (
            "model.layers.0.linear_attn.in_proj_b.weight",
            "F16",
            vec![2, h],
            f16_vec(2 * h, 0.01),
        ),
        (
            "model.layers.0.linear_attn.in_proj_a.weight",
            "F16",
            vec![2, h],
            f16_vec(2 * h, 0.01),
        ),
        (
            "model.layers.0.linear_attn.out_proj.weight",
            "F16",
            vec![h, 256],
            f16_vec(h * 256, 0.01),
        ),
        (
            "model.layers.0.linear_attn.conv1d.weight",
            "F16",
            vec![3 * 256, 1, 4],
            f16_vec(3 * 256 * 4, 0.05),
        ),
        (
            "model.layers.0.linear_attn.A_log",
            "F32",
            vec![2],
            f32_vec(2, 0.1),
        ),
        (
            "model.layers.0.linear_attn.dt_bias",
            "F32",
            vec![2],
            f32_vec(2, 0.1),
        ),
        (
            "model.layers.0.linear_attn.norm.weight",
            "F16",
            vec![128],
            f16_vec(128, 1.0),
        ),
        (
            "model.layers.0.mlp.gate_proj.weight",
            "F16",
            vec![interm, h],
            f16_vec(interm * h, 0.01),
        ),
        (
            "model.layers.0.mlp.up_proj.weight",
            "F16",
            vec![interm, h],
            f16_vec(interm * h, 0.01),
        ),
        (
            "model.layers.0.mlp.down_proj.weight",
            "F16",
            vec![h, interm],
            f16_vec(h * interm, 0.01),
        ),
        // Layer 1: FullAttention
        (
            "model.layers.1.input_layernorm.weight",
            "F16",
            vec![h],
            f16_vec(h, 0.0),
        ),
        (
            "model.layers.1.post_attention_layernorm.weight",
            "F16",
            vec![h],
            f16_vec(h, 0.0),
        ),
        (
            "model.layers.1.self_attn.q_proj.weight",
            "F16",
            vec![2 * 256, h],
            f16_vec(2 * 256 * h, 0.01),
        ),
        (
            "model.layers.1.self_attn.k_proj.weight",
            "F16",
            vec![128, h],
            f16_vec(128 * h, 0.01),
        ),
        (
            "model.layers.1.self_attn.v_proj.weight",
            "F16",
            vec![128, h],
            f16_vec(128 * h, 0.01),
        ),
        (
            "model.layers.1.self_attn.o_proj.weight",
            "F16",
            vec![h, 256],
            f16_vec(h * 256, 0.01),
        ),
        (
            "model.layers.1.self_attn.q_norm.weight",
            "F16",
            vec![128],
            f16_vec(128, 0.0),
        ),
        (
            "model.layers.1.self_attn.k_norm.weight",
            "F16",
            vec![128],
            f16_vec(128, 0.0),
        ),
        (
            "model.layers.1.mlp.gate_proj.weight",
            "F16",
            vec![interm, h],
            f16_vec(interm * h, 0.01),
        ),
        (
            "model.layers.1.mlp.up_proj.weight",
            "F16",
            vec![interm, h],
            f16_vec(interm * h, 0.01),
        ),
        (
            "model.layers.1.mlp.down_proj.weight",
            "F16",
            vec![h, interm],
            f16_vec(h * interm, 0.01),
        ),
        // Native 1-layer MTP head
        (
            "mtp.fc.weight",
            "F16",
            vec![h, 2 * h],
            f16_vec(h * 2 * h, 0.01),
        ),
        (
            "mtp.pre_fc_norm_embedding.weight",
            "F16",
            vec![h],
            f16_vec(h, 0.0),
        ),
        (
            "mtp.pre_fc_norm_hidden.weight",
            "F16",
            vec![h],
            f16_vec(h, 0.0),
        ),
        ("mtp.norm.weight", "F16", vec![h], f16_vec(h, 0.0)),
        (
            "mtp.layers.0.input_layernorm.weight",
            "F16",
            vec![h],
            f16_vec(h, 0.0),
        ),
        (
            "mtp.layers.0.post_attention_layernorm.weight",
            "F16",
            vec![h],
            f16_vec(h, 0.0),
        ),
        (
            "mtp.layers.0.self_attn.q_proj.weight",
            "F16",
            vec![2 * 256, h],
            f16_vec(2 * 256 * h, 0.01),
        ),
        (
            "mtp.layers.0.self_attn.k_proj.weight",
            "F16",
            vec![128, h],
            f16_vec(128 * h, 0.01),
        ),
        (
            "mtp.layers.0.self_attn.v_proj.weight",
            "F16",
            vec![128, h],
            f16_vec(128 * h, 0.01),
        ),
        (
            "mtp.layers.0.self_attn.o_proj.weight",
            "F16",
            vec![h, 256],
            f16_vec(h * 256, 0.01),
        ),
        (
            "mtp.layers.0.self_attn.q_norm.weight",
            "F16",
            vec![128],
            f16_vec(128, 0.0),
        ),
        (
            "mtp.layers.0.self_attn.k_norm.weight",
            "F16",
            vec![128],
            f16_vec(128, 0.0),
        ),
        (
            "mtp.layers.0.mlp.gate_proj.weight",
            "F16",
            vec![interm, h],
            f16_vec(interm * h, 0.01),
        ),
        (
            "mtp.layers.0.mlp.up_proj.weight",
            "F16",
            vec![interm, h],
            f16_vec(interm * h, 0.01),
        ),
        (
            "mtp.layers.0.mlp.down_proj.weight",
            "F16",
            vec![h, interm],
            f16_vec(h * interm, 0.01),
        ),
    ]);
    std::fs::write(
        fixture.0.join("config.json"),
        serde_json::to_vec(&json!({
            "architectures": ["Qwen3_5ForCausalLM"],
            "model_type": "qwen3_5_text",
            "vocab_size": vocab,
            "hidden_size": h,
            "intermediate_size": interm,
            "num_hidden_layers": 2,
            "mtp_num_hidden_layers": 1,
            "num_attention_heads": 2,
            "num_key_value_heads": 1,
            "head_dim": 128,
            "rms_norm_eps": 1e-6,
            "eos_token_id": [99],
            "tie_word_embeddings": false,
            "layer_types": ["linear_attention", "full_attention"],
            "linear_num_key_heads": 2,
            "linear_key_head_dim": 128,
            "linear_num_value_heads": 2,
            "linear_value_head_dim": 128,
            "linear_conv_kernel_dim": 4,
            "rope_parameters": {
                "rope_theta": 10000000.0,
                "partial_rotary_factor": 0.25,
                "mrope_section": [11, 11, 10]
            }
        }))
        .unwrap(),
    )
    .unwrap();

    // Write a synthetic ASCII BPE `tokenizer.json` so `encode_prompt`, `decode_tokens`,
    // `EngineWorker`, and `server::serve` HTTP routes run end-to-end without external files.
    {
        use tokenizers::AddedToken;
        use tokenizers::models::bpe::{BPE, Vocab};
        let mut vocab_map = Vocab::default();
        vocab_map.insert("[UNK]".to_owned(), 0);
        for (next_id, ch) in (1u32..).zip(
            ['\t', '\n', '\r']
                .into_iter()
                .chain((0x20u8..=0x7eu8).map(|b| b as char)),
        ) {
            vocab_map.insert(ch.to_string(), next_id);
        }
        let bpe = BPE::builder()
            .vocab_and_merges(vocab_map, Vec::new())
            .unk_token("[UNK]".to_owned())
            .build()
            .unwrap();
        let mut tok = tokenizers::Tokenizer::new(bpe);
        let added: Vec<AddedToken> = [
            "<|endoftext|>",
            "<|im_start|>",
            "<|im_end|>",
            "<|vision_start|>",
            "<|vision_end|>",
            "<|image_pad|>",
            "<think>",
            "</think>",
            "<tool_call>",
            "</tool_call>",
            "noul",
            "yul",
        ]
        .iter()
        .map(|&s| AddedToken::from(s, true))
        .collect();
        tok.add_special_tokens(&added);
        tok.save(fixture.0.join("tokenizer.json"), false).unwrap();
    }

    let mut engine = Qwen3_5Engine::load(
        fixture.0.to_str().unwrap(),
        1,
        KvFormatPolicy::Fp16,
        Some(64),
        true,
    )
    .unwrap();
    let encoded_hi = engine.encode("hi").unwrap();
    assert!(!encoded_hi.is_empty());
    let decoded_hi = engine.decode(&encoded_hi).unwrap();
    assert_eq!(decoded_hi, "h i");
    let dump = engine.dump_prompt(&[1, 2, 3, 4]).unwrap();
    assert!(dump.tensors_f32.contains_key("final_norm"));
    assert_eq!(dump.top5_ids.len(), 5);

    let generated = engine.generate_tokens(&[1, 2, 3, 4], 3, false).unwrap();
    assert_eq!(generated.generated_token_ids.len(), 3);

    // Exercise `generate_with_radix_cache` cold + warm prefix hit + checkpoint donation.
    let radix_cold = engine
        .generate_with_radix_cache(&[1, 2, 3, 4], 2, 64, false)
        .unwrap();
    assert_eq!(radix_cold.generated_token_ids.len(), 2);
    let radix_warm = engine
        .generate_with_radix_cache(&[1, 2, 3, 4, 5, 6], 2, 64, false)
        .unwrap();
    assert_eq!(radix_warm.generated_token_ids.len(), 2);

    // Exercise `step_batch_slices` with `emit_all_query_tokens: true`, `read_selected_row_candidate_logits`,
    // `read_norm_hidden_row_f32`, and `sample_single_gpu_step`.
    let out = engine
        .step_batch_slices(&[EngineBatchSlice {
            tokens: &[5, 6, 7],
            ctx_len_before: 0,
            block_table: &[0],
            state_in_slot: 0,
            state_out_slot: 0,
            has_initial_state: false,
            spec_slots: &[],
            checkpoints: &[],
            emit_all_query_tokens: true,
        }])
        .unwrap();
    assert_eq!(out[0].len(), 3);
    let cand_logits = engine
        .read_selected_row_candidate_logits(2, &[1, 2])
        .unwrap();
    assert_eq!(cand_logits.len(), 2);
    let hidden_row = engine.read_norm_hidden_row_f32(2).unwrap();
    assert_eq!(hidden_row.len(), h);

    let mut sampler = crate::parser::SamplerV1::new(
        crate::parser::SamplingParams {
            temperature: 0.0,
            logprobs: true,
            top_logprobs: 2,
            ..crate::parser::SamplingParams::default()
        },
        &[5, 6, 7],
        None,
    );
    let (tok_id, lp) = engine.sample_single_gpu_step(&mut sampler).unwrap();
    assert!((0..vocab as i32).contains(&tok_id));
    assert!(lp.is_some());

    // Exercise `generate_batch_scheduled` and `ScheduledVision` validation + `step_prefill`.
    let batch_sched = engine
        .generate_batch_scheduled(
            &[vec![1, 2, 3], vec![1, 2, 4]],
            2,
            crate::scheduler::SchedulerConfig::default(),
        )
        .unwrap();
    assert_eq!(batch_sched.len(), 2);
    assert_eq!(batch_sched[0].0.len(), 2);
    assert_eq!(batch_sched[1].0.len(), 2);

    assert!(crate::scheduled_vision::ScheduledVision::new(&engine, &[1, 2], Vec::new()).is_err());
    engine.cfg.vision_config = Some(serde_json::from_value(json!({})).unwrap());
    let mut sv =
        crate::scheduled_vision::ScheduledVision::new(&engine, &[1, 2, 3, 4], Vec::new()).unwrap();
    let mut sched =
        crate::scheduler::SchedulerV1::new(crate::scheduler::SchedulerConfig::default());
    let _req_id = sched.submit_request(vec![1, 2, 3, 4], 2, Vec::new());
    sched.admit_waiting(|_| Some((0, [1, 2])));
    sched.active[0].radix_state.kv_pages = vec![0];
    let seq = &sched.active[0];
    let (prefill_tok, slice) = sv.step_prefill(&mut engine, seq, 4).unwrap();
    assert!((0..vocab as i32).contains(&prefill_tok));
    assert!(slice.is_final_prefill_chunk);

    // Exercise native MTP speculative decoding (`enable_speculative`, `dump_prompt`, `generate_with_radix_cache`).
    engine
        .enable_speculative(&crate::proposer::SpeculativeConfig {
            kind: crate::proposer::SpeculativeKind::Mtp,
            draft_model: None,
            mtp_k: 2,
        })
        .unwrap();
    let mtp_dump = engine.dump_prompt(&[1, 2, 3, 4]).unwrap();
    assert!(mtp_dump.tensors_f32.contains_key("drafter_hidden"));
    assert_eq!(mtp_dump.draft_proposal_ids.len(), 2);
    let mtp_gen = engine
        .generate_with_radix_cache(&[1, 2, 3, 4], 4, 64, false)
        .unwrap();
    assert_eq!(mtp_gen.generated_token_ids.len(), 4);

    // Exercise `EngineWorker` + `OpenAiServerState` + `server::serve` HTTP routes in-process.
    engine
        .set_decision_profile(crate::decision::DecisionModelProfile::Xor {
            temperature: 1.0,
            lev_noul_9_scale: false,
        })
        .unwrap();
    let state = crate::server::OpenAiServerState::new("synthetic-model".to_string(), engine);
    tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .unwrap()
        .block_on(async {
            let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
            let addr = listener.local_addr().unwrap();
            let base = format!("http://{addr}");
            let (tx, rx) = tokio::sync::oneshot::channel::<()>();
            let srv_state = state.clone();
            let srv = tokio::spawn(async move {
                crate::server::serve(listener, srv_state, async {
                    let _ = rx.await;
                })
                .await
            });
            let client = reqwest::Client::new();

            for get_path in ["/health", "/v1/models", "/metrics", "/api/metrics"] {
                let resp = client.get(format!("{base}{get_path}")).send().await.unwrap();
                assert!(resp.status().is_success(), "GET {get_path} failed");
            }
            for (post_path, payload) in [
                ("/tokenize", json!({"prompt": "hi"})),
                ("/detokenize", json!({"tokens": [1, 2]})),
                (
                    "/v1/completions",
                    json!({"prompt": "hi", "max_tokens": 2, "stop_token_ids": []}),
                ),
                (
                    "/v1/completions",
                    json!({"prompt": "hi", "max_tokens": 2, "stop_token_ids": [], "stream": true, "stream_options": {"include_usage": true}}),
                ),
                (
                    "/v1/chat/completions",
                    json!({"messages": [{"role": "user", "content": "hi"}], "max_tokens": 2, "stop_token_ids": []}),
                ),
                (
                    "/v1/chat/completions",
                    json!({"messages": [{"role": "user", "content": "hi"}], "max_tokens": 2, "stop_token_ids": [], "stream": true, "stream_options": {"include_usage": true}}),
                ),
                (
                    "/v1/messages",
                    json!({"max_tokens": 2, "messages": [{"role": "user", "content": "hi"}]}),
                ),
                (
                    "/v1/messages",
                    json!({"max_tokens": 2, "stream": true, "messages": [{"role": "user", "content": "hi"}]}),
                ),
                (
                    "/v1/messages/count_tokens",
                    json!({"messages": [{"role": "user", "content": "hi"}]}),
                ),
                ("/v1/responses", json!({"input": "hi", "max_output_tokens": 2})),
                ("/v1/responses", json!({"input": "hi", "max_output_tokens": 2, "stream": true})),
                ("/v1/responses/input_tokens", json!({"input": "hi"})),
                (
                    "/v1/systemone",
                    json!({"state": "s", "questions": {"q": {"type": "noul", "instructions": "i"}}}),
                ),
                (
                    "/v1/decisions",
                    json!({"state": "s", "decisions": [{"id": "q", "type": "noul", "instructions": "i"}]}),
                ),
            ] {
                let resp = client
                    .post(format!("{base}{post_path}"))
                    .json(&payload)
                    .send()
                    .await
                    .unwrap();
                assert!(
                    resp.status().is_success(),
                    "POST {post_path} failed with {}",
                    resp.status()
                );
                let text = resp.text().await.unwrap();
                assert!(!text.is_empty(), "POST {post_path} returned empty body");
                if !payload
                    .get("stream")
                    .and_then(|v| v.as_bool())
                    .unwrap_or(false)
                {
                    let body_json: serde_json::Value = serde_json::from_str(&text)
                        .unwrap_or_else(|e| panic!("POST {post_path} invalid JSON '{text}': {e}"));
                    match post_path {
                        "/tokenize" => assert!(body_json["tokens"].is_array()),
                        "/detokenize" => assert!(body_json["text"].is_string()),
                        "/v1/completions" | "/v1/chat/completions" => {
                            assert!(body_json["choices"].as_array().is_some_and(|a| !a.is_empty()));
                        }
                        "/v1/messages" => assert!(body_json["content"].is_array()),
                        "/v1/messages/count_tokens" | "/v1/responses/input_tokens" => {
                            assert!(body_json["input_tokens"].as_u64().unwrap_or(0) > 0);
                        }
                        "/v1/responses" => assert!(body_json["output"].is_array()),
                        _ => {}
                    }
                }
            }

            let _ = tx.send(());
            srv.await.unwrap().unwrap();
        });
}
