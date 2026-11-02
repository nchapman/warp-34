#![cfg(feature = "cuda")]

use std::fs;
use std::path::{Path, PathBuf};
use w34_host::cache::KvFormatPolicy;
use w34_host::cuda::f32_to_bf16;
use w34_host::model::Qwen3_5Engine;

fn push_bf16_tensor(
    tensors: &mut Vec<(String, &'static str, Vec<usize>, Vec<u8>)>,
    name: impl Into<String>,
    shape: Vec<usize>,
    f32_fn: impl Fn(usize) -> f32,
) {
    let numel: usize = shape.iter().product();
    let mut bytes = Vec::with_capacity(numel * 2);
    for i in 0..numel {
        bytes.extend_from_slice(&f32_to_bf16(f32_fn(i)).to_le_bytes());
    }
    tensors.push((name.into(), "BF16", shape, bytes));
}

fn push_i64_tensor(
    tensors: &mut Vec<(String, &'static str, Vec<usize>, Vec<u8>)>,
    name: impl Into<String>,
    shape: Vec<usize>,
    vals: &[i64],
) {
    let mut bytes = Vec::with_capacity(vals.len() * 8);
    for &v in vals {
        bytes.extend_from_slice(&v.to_le_bytes());
    }
    tensors.push((name.into(), "I64", shape, bytes));
}

fn push_f8_tensor(
    tensors: &mut Vec<(String, &'static str, Vec<usize>, Vec<u8>)>,
    name: impl Into<String>,
    shape: Vec<usize>,
    u8_fn: impl Fn(usize) -> u8,
) {
    let numel: usize = shape.iter().product();
    let mut bytes = Vec::with_capacity(numel);
    for i in 0..numel {
        bytes.push(u8_fn(i));
    }
    tensors.push((name.into(), "F8_E4M3", shape, bytes));
}

fn push_autoround_expert_proj(
    tensors: &mut Vec<(String, &'static str, Vec<usize>, Vec<u8>)>,
    prefix: &str,
    out_features: usize,
    in_features: usize,
    group_size: usize,
    seed: usize,
) {
    let packed_rows = in_features / 8;
    let packed_cols = out_features;
    let qweight_numel = packed_rows * packed_cols;
    let mut qw_bytes = Vec::with_capacity(qweight_numel * 4);
    for i in 0..qweight_numel {
        // Nibbles around 8 (zero-point=8 in symmetric unsigned 4-bit) with small deterministic delta
        let d0 = 8u32.wrapping_add(((i + seed) % 3) as u32).wrapping_sub(1) & 0xF;
        let d1 = 8u32
            .wrapping_add(((i + seed + 1) % 3) as u32)
            .wrapping_sub(1)
            & 0xF;
        let word = d0
            | (d1 << 4)
            | (d0 << 8)
            | (d1 << 12)
            | (d0 << 16)
            | (d1 << 20)
            | (d0 << 24)
            | (d1 << 28);
        qw_bytes.extend_from_slice(&word.to_le_bytes());
    }
    tensors.push((
        format!("{prefix}.qweight"),
        "I32",
        vec![packed_rows, packed_cols],
        qw_bytes,
    ));

    let num_groups = in_features / group_size;
    push_bf16_tensor(
        tensors,
        format!("{prefix}.scales"),
        vec![num_groups, out_features],
        |i| 0.005 + ((i + seed) % 7) as f32 * 0.001,
    );
}

fn write_safetensors_file(
    path: &Path,
    tensors: &[(String, &'static str, Vec<usize>, Vec<u8>)],
) -> std::io::Result<()> {
    let mut header_entries = Vec::with_capacity(tensors.len());
    let mut offset = 0usize;
    for (name, dtype, shape, data) in tensors {
        let start = offset;
        let end = start + data.len();
        offset = end;
        let shape_str = shape
            .iter()
            .map(|d| d.to_string())
            .collect::<Vec<_>>()
            .join(",");
        header_entries.push(format!(
            "\"{name}\":{{\"dtype\":\"{dtype}\",\"shape\":[{shape_str}],\"data_offsets\":[{start},{end}]}}"
        ));
    }
    let mut header_json = format!("{{{}}}", header_entries.join(","));
    while !header_json.len().is_multiple_of(8) {
        header_json.push(' ');
    }
    let mut out = Vec::with_capacity(8 + header_json.len() + offset);
    out.extend_from_slice(&(header_json.len() as u64).to_le_bytes());
    out.extend_from_slice(header_json.as_bytes());
    for (_, _, _, data) in tensors {
        out.extend_from_slice(data);
    }
    fs::write(path, out)
}

fn build_synthetic_qwen4_exp_dir(tag: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!(
        "w34_qwen4_exp_synth_{}_{}",
        tag,
        std::process::id()
    ));
    let _ = fs::remove_dir_all(&dir);
    fs::create_dir_all(&dir).unwrap();

    // Compact geometry that satisfies all TP=1 and TP=2 divisibility & Marlin constraints:
    // hidden_size = 256, hc_count = 4 (hc_total = 1024), hc_lowrank = 32 (low_dim = 128)
    // head_dim = 256, num_attention_heads = 4, num_key_value_heads = 2
    // linear_num_key_heads = 4, linear_key_head_dim = 128, linear_num_value_heads = 4, linear_value_head_dim = 128
    // moe_intermediate_size = 128, shared_expert_intermediate_size = 128, num_experts = 8, num_experts_per_tok = 2
    // vocab_size = 256, ple_embed_dim = 256, ngram_heads = 16, head_dim_ple = 16
    let cfg_json = r#"{
      "architectures": ["Qwen4ExpForConditionalGeneration"],
      "model_type": "qwen4_exp",
      "text_config": {
        "model_type": "qwen4_exp_text",
        "vocab_size": 256,
        "hidden_size": 256,
        "intermediate_size": 128,
        "moe_intermediate_size": 128,
        "shared_expert_intermediate_size": 128,
        "num_hidden_layers": 2,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "head_dim": 256,
        "rms_norm_eps": 1e-6,
        "tie_word_embeddings": false,
        "layer_types": ["linear_attention", "full_attention"],
        "linear_num_key_heads": 4,
        "linear_key_head_dim": 128,
        "linear_num_value_heads": 4,
        "linear_value_head_dim": 128,
        "linear_conv_kernel_dim": 4,
        "num_experts": 8,
        "num_experts_per_tok": 2,
        "norm_topk_prob": true,
        "hc_count": 4,
        "hc_lowrank": 128,
        "ple_embed_dim": 256,
        "ple_layer_ids": [2],
        "ngram_size": 3,
        "ngram_heads": 16,
        "ngram_vocab_size_multiplier": 4,
        "split_ngram_parts": 4,
        "make_ngram_vocab_size_divisible_by": 4,
        "indexer_n_heads": 4,
        "indexer_kv_heads": 1,
        "indexer_head_dim": 128,
        "indexer_budget": 128,
        "indexer_compress_ratio": 4,
        "mtp_num_hidden_layers": 1,
        "rope_parameters": {
          "rope_theta": 10000000.0,
          "partial_rotary_factor": 0.25,
          "mrope_section": [11, 11, 10]
        }
      },
      "quantization_config": {
        "quant_method": "auto-round",
        "bits": 4,
        "group_size": 128,
        "sym": true
      }
    }"#;
    fs::write(dir.join("config.json"), cfg_json).unwrap();

    let h = 256usize;
    let hc_count = 4usize;
    let hc_total = hc_count * h; // 1024
    let hc_low = hc_count * 32; // 128
    let vocab = 256usize;
    let num_experts = 8usize;
    let moe_interm = 128usize;

    let mut tensors = Vec::new();
    push_bf16_tensor(
        &mut tensors,
        "model.language_model.embed_tokens.weight",
        vec![vocab, h],
        |i| ((i % 31) as f32 - 15.0) * 0.02,
    );
    push_bf16_tensor(&mut tensors, "lm_head.weight", vec![vocab, h], |i| {
        ((i % 29) as f32 - 14.0) * 0.02
    });
    // Final HC mixer
    push_bf16_tensor(
        &mut tensors,
        "model.language_model.hyper_connection_mixer.hc_norm.weight",
        vec![hc_total],
        |i| 0.05 * ((i % 13) as f32 - 6.0),
    );
    push_bf16_tensor(
        &mut tensors,
        "model.language_model.hyper_connection_mixer.input_mix_weight_down.weight",
        vec![hc_low, hc_total],
        |i| ((i % 19) as f32 - 9.0) * 0.01,
    );
    push_bf16_tensor(
        &mut tensors,
        "model.language_model.hyper_connection_mixer.input_mix_weight_up.weight",
        vec![hc_total, hc_low],
        |i| ((i % 17) as f32 - 8.0) * 0.01,
    );

    for layer_id in 0..2 {
        let pfx = format!("model.language_model.layers.{layer_id}");
        for hc_name in ["attn_hyper_connection", "mlp_hyper_connection"] {
            push_bf16_tensor(
                &mut tensors,
                format!("{pfx}.{hc_name}.hc_norm.weight"),
                vec![hc_total],
                |i| 0.05 * (((i + layer_id) % 11) as f32 - 5.0),
            );
            push_bf16_tensor(
                &mut tensors,
                format!("{pfx}.{hc_name}.input_mix_weight_down.weight"),
                vec![hc_low, hc_total],
                |i| (((i + layer_id) % 23) as f32 - 11.0) * 0.01,
            );
            push_bf16_tensor(
                &mut tensors,
                format!("{pfx}.{hc_name}.input_mix_weight_up.weight"),
                vec![hc_total, hc_low],
                |i| (((i + layer_id) % 17) as f32 - 8.0) * 0.01,
            );
            push_bf16_tensor(
                &mut tensors,
                format!("{pfx}.{hc_name}.block_inject_weight.weight"),
                vec![hc_count, hc_total],
                |i| (((i + layer_id) % 13) as f32 - 6.0) * 0.01,
            );
        }

        if layer_id == 0 {
            // GDN linear_attention: k_total = 4*128 = 512, v_total = 4*128 = 512, qkv = 1536
            push_bf16_tensor(
                &mut tensors,
                format!("{pfx}.linear_attn.in_proj_qkv.weight"),
                vec![1536, h],
                |i| ((i % 19) as f32 - 9.0) * 0.01,
            );
            push_bf16_tensor(
                &mut tensors,
                format!("{pfx}.linear_attn.in_proj_z.weight"),
                vec![512, h],
                |i| ((i % 17) as f32 - 8.0) * 0.01,
            );
            push_bf16_tensor(
                &mut tensors,
                format!("{pfx}.linear_attn.in_proj_b.weight"),
                vec![4, h],
                |i| ((i % 13) as f32 - 6.0) * 0.01,
            );
            push_bf16_tensor(
                &mut tensors,
                format!("{pfx}.linear_attn.in_proj_a.weight"),
                vec![4, h],
                |i| ((i % 11) as f32 - 5.0) * 0.01,
            );
            push_bf16_tensor(
                &mut tensors,
                format!("{pfx}.linear_attn.out_proj.weight"),
                vec![h, 512],
                |i| ((i % 23) as f32 - 11.0) * 0.01,
            );
            push_bf16_tensor(
                &mut tensors,
                format!("{pfx}.linear_attn.conv1d.weight"),
                vec![1536, 1, 4],
                |i| ((i % 7) as f32 - 3.0) * 0.02,
            );
            push_bf16_tensor(
                &mut tensors,
                format!("{pfx}.linear_attn.A_log"),
                vec![4],
                |i| 1.0 + i as f32 * 0.1,
            );
            push_bf16_tensor(
                &mut tensors,
                format!("{pfx}.linear_attn.dt_bias"),
                vec![4],
                |i| 0.1 * (i + 1) as f32,
            );
            push_bf16_tensor(
                &mut tensors,
                format!("{pfx}.linear_attn.norm.weight"),
                vec![128],
                |_| 1.0,
            );
        } else {
            // PLE on layer_id == 1 (ple_layer_ids: [2])
            push_bf16_tensor(
                &mut tensors,
                format!("{pfx}.ple.key_proj.weight"),
                vec![hc_total, h],
                |i| ((i % 19) as f32 - 9.0) * 0.01,
            );
            push_bf16_tensor(
                &mut tensors,
                format!("{pfx}.ple.value_proj.weight"),
                vec![h, h],
                |i| ((i % 17) as f32 - 8.0) * 0.01,
            );
            push_bf16_tensor(
                &mut tensors,
                format!("{pfx}.ple.norm_key.weight"),
                vec![hc_total],
                |_| 0.1,
            );
            push_bf16_tensor(
                &mut tensors,
                format!("{pfx}.ple.norm_query.weight"),
                vec![hc_total],
                |_| 0.1,
            );
            push_bf16_tensor(
                &mut tensors,
                format!("{pfx}.ple.norm_conv.weight"),
                vec![hc_total],
                |_| 0.1,
            );
            push_bf16_tensor(
                &mut tensors,
                format!("{pfx}.ple.conv1d.weight"),
                vec![hc_total, 1, 4],
                |i| ((i % 7) as f32 - 3.0) * 0.02,
            );
            push_bf16_tensor(
                &mut tensors,
                format!("{pfx}.ple.ple_embedding.ngram_embedding.weight_scale"),
                vec![1],
                |_| 0.02,
            );
            // Compute exact PleNgramHasher vocab sizes/offsets for vocab=256, mult=4 -> base=1024
            let root: serde_json::Value = serde_json::from_str(cfg_json).unwrap();
            let dummy_cfg = w34_host::loader::ModelConfig::from_json(&root).unwrap();
            let hasher = w34_host::ple::PleNgramHasher::from_config(&dummy_cfg, 1, 0);
            push_i64_tensor(
                &mut tensors,
                format!("{pfx}.ple.ple_embedding.layer_multipliers"),
                vec![3],
                &hasher.layer_multipliers,
            );
            push_i64_tensor(
                &mut tensors,
                format!("{pfx}.ple.ple_embedding.ngram_heads_vocab_sizes"),
                vec![16],
                &hasher.ngram_heads_vocab_sizes,
            );
            push_i64_tensor(
                &mut tensors,
                format!("{pfx}.ple.ple_embedding.ngram_heads_offsets"),
                vec![16],
                &hasher.ngram_heads_offsets,
            );
            for s in 0..4 {
                push_f8_tensor(
                    &mut tensors,
                    format!("{pfx}.ple.ple_embedding.ngram_embedding.shard_{s}.weight"),
                    vec![hasher.shard_rows, 16],
                    |i| ((i + s) % 64) as u8,
                );
            }

            // FullAttention + QSA indexer: q_gate = 4*256*2 = 2048, kv = 2*256 = 512, o = 4*256 = 1024
            push_bf16_tensor(
                &mut tensors,
                format!("{pfx}.self_attn.q_proj.weight"),
                vec![2048, h],
                |i| ((i % 23) as f32 - 11.0) * 0.01,
            );
            push_bf16_tensor(
                &mut tensors,
                format!("{pfx}.self_attn.k_proj.weight"),
                vec![512, h],
                |i| ((i % 19) as f32 - 9.0) * 0.01,
            );
            push_bf16_tensor(
                &mut tensors,
                format!("{pfx}.self_attn.v_proj.weight"),
                vec![512, h],
                |i| ((i % 17) as f32 - 8.0) * 0.01,
            );
            push_bf16_tensor(
                &mut tensors,
                format!("{pfx}.self_attn.o_proj.weight"),
                vec![h, 1024],
                |i| ((i % 13) as f32 - 6.0) * 0.01,
            );
            push_bf16_tensor(
                &mut tensors,
                format!("{pfx}.self_attn.q_norm.weight"),
                vec![256],
                |_| 0.1,
            );
            push_bf16_tensor(
                &mut tensors,
                format!("{pfx}.self_attn.k_norm.weight"),
                vec![256],
                |_| 0.1,
            );
            // QSA indexer: (4+1)*128 = 640
            push_bf16_tensor(
                &mut tensors,
                format!("{pfx}.self_attn.indexer.index_qk_proj.weight"),
                vec![640, h],
                |i| ((i % 19) as f32 - 9.0) * 0.01,
            );
            push_bf16_tensor(
                &mut tensors,
                format!("{pfx}.self_attn.indexer.q_layernorm.weight"),
                vec![128],
                |_| 0.1,
            );
            push_bf16_tensor(
                &mut tensors,
                format!("{pfx}.self_attn.indexer.k_layernorm.weight"),
                vec![128],
                |_| 0.1,
            );
        }

        // MoE router + shared expert + 8 AutoRound routed experts
        push_bf16_tensor(
            &mut tensors,
            format!("{pfx}.mlp.gate.weight"),
            vec![num_experts, h],
            |i| ((i + layer_id * 3) % 17) as f32 * 0.02 - 0.15,
        );
        push_bf16_tensor(
            &mut tensors,
            format!("{pfx}.mlp.shared_expert.gate_proj.weight"),
            vec![moe_interm, h],
            |i| ((i % 19) as f32 - 9.0) * 0.01,
        );
        push_bf16_tensor(
            &mut tensors,
            format!("{pfx}.mlp.shared_expert.up_proj.weight"),
            vec![moe_interm, h],
            |i| ((i % 17) as f32 - 8.0) * 0.01,
        );
        push_bf16_tensor(
            &mut tensors,
            format!("{pfx}.mlp.shared_expert.down_proj.weight"),
            vec![h, moe_interm],
            |i| ((i % 13) as f32 - 6.0) * 0.01,
        );
        push_bf16_tensor(
            &mut tensors,
            format!("{pfx}.mlp.shared_expert_gate.weight"),
            vec![1, h],
            |i| ((i % 7) as f32 - 3.0) * 0.01,
        );

        for e in 0..num_experts {
            let epfx = format!("{pfx}.mlp.experts.{e}");
            push_autoround_expert_proj(
                &mut tensors,
                &format!("{epfx}.gate_proj"),
                moe_interm,
                h,
                128,
                layer_id * 100 + e * 3,
            );
            push_autoround_expert_proj(
                &mut tensors,
                &format!("{epfx}.up_proj"),
                moe_interm,
                h,
                128,
                layer_id * 100 + e * 3 + 1,
            );
            push_autoround_expert_proj(
                &mut tensors,
                &format!("{epfx}.down_proj"),
                h,
                moe_interm,
                128,
                layer_id * 100 + e * 3 + 2,
            );
        }
    }

    // Native Qwen4 MTP weights (`mtp.*`): dual-stream `residual_linear_shared` + 1-layer HC + QSA + 8-expert MoE + final HC mixer
    push_bf16_tensor(
        &mut tensors,
        "mtp.pre_fc_norm_embedding.weight",
        vec![h],
        |i| 0.05 * ((i % 11) as f32 - 5.0),
    );
    push_bf16_tensor(
        &mut tensors,
        "mtp.pre_fc_norm_hidden.weight",
        vec![hc_total],
        |i| 0.05 * ((i % 13) as f32 - 6.0),
    );
    push_bf16_tensor(&mut tensors, "mtp.fc_embedding.weight", vec![h, h], |i| {
        ((i % 19) as f32 - 9.0) * 0.01
    });
    push_bf16_tensor(&mut tensors, "mtp.fc_hidden.weight", vec![h, h], |i| {
        ((i % 17) as f32 - 8.0) * 0.01
    });
    for sub in ["attn_hyper_connection", "mlp_hyper_connection"] {
        push_bf16_tensor(
            &mut tensors,
            format!("mtp.layers.0.{sub}.hc_norm.weight"),
            vec![hc_total],
            |i| 0.04 * ((i % 11) as f32 - 5.0),
        );
        push_bf16_tensor(
            &mut tensors,
            format!("mtp.layers.0.{sub}.input_mix_weight_down.weight"),
            vec![hc_low, hc_total],
            |i| ((i % 19) as f32 - 9.0) * 0.004,
        );
        push_bf16_tensor(
            &mut tensors,
            format!("mtp.layers.0.{sub}.input_mix_weight_up.weight"),
            vec![hc_total, hc_low],
            |i| ((i % 23) as f32 - 11.0) * 0.004,
        );
        push_bf16_tensor(
            &mut tensors,
            format!("mtp.layers.0.{sub}.block_inject_weight.weight"),
            vec![hc_count, hc_total],
            |i| ((i % 15) as f32 - 7.0) * 0.004,
        );
    }
    push_bf16_tensor(
        &mut tensors,
        "mtp.hyper_connection_mixer.hc_norm.weight",
        vec![hc_total],
        |i| 0.05 * ((i % 13) as f32 - 6.0),
    );
    push_bf16_tensor(
        &mut tensors,
        "mtp.hyper_connection_mixer.input_mix_weight_down.weight",
        vec![hc_low, hc_total],
        |i| ((i % 17) as f32 - 8.0) * 0.005,
    );
    push_bf16_tensor(
        &mut tensors,
        "mtp.hyper_connection_mixer.input_mix_weight_up.weight",
        vec![hc_total, hc_low],
        |i| ((i % 23) as f32 - 11.0) * 0.005,
    );
    push_bf16_tensor(
        &mut tensors,
        "mtp.layers.0.self_attn.q_proj.weight",
        vec![2048, h],
        |i| ((i % 19) as f32 - 9.0) * 0.01,
    );
    push_bf16_tensor(
        &mut tensors,
        "mtp.layers.0.self_attn.k_proj.weight",
        vec![512, h],
        |i| ((i % 17) as f32 - 8.0) * 0.01,
    );
    push_bf16_tensor(
        &mut tensors,
        "mtp.layers.0.self_attn.v_proj.weight",
        vec![512, h],
        |i| ((i % 23) as f32 - 11.0) * 0.01,
    );
    push_bf16_tensor(
        &mut tensors,
        "mtp.layers.0.self_attn.o_proj.weight",
        vec![h, 1024],
        |i| ((i % 13) as f32 - 6.0) * 0.01,
    );
    push_bf16_tensor(
        &mut tensors,
        "mtp.layers.0.self_attn.q_norm.weight",
        vec![256],
        |_| 0.1,
    );
    push_bf16_tensor(
        &mut tensors,
        "mtp.layers.0.self_attn.k_norm.weight",
        vec![256],
        |_| 0.1,
    );
    push_bf16_tensor(
        &mut tensors,
        "mtp.layers.0.self_attn.indexer.index_qk_proj.weight",
        vec![(4 + 1) * 128, h],
        |i| ((i % 21) as f32 - 10.0) * 0.01,
    );
    push_bf16_tensor(
        &mut tensors,
        "mtp.layers.0.self_attn.indexer.q_layernorm.weight",
        vec![128],
        |_| 0.1,
    );
    push_bf16_tensor(
        &mut tensors,
        "mtp.layers.0.self_attn.indexer.k_layernorm.weight",
        vec![128],
        |_| 0.1,
    );
    push_bf16_tensor(
        &mut tensors,
        "mtp.layers.0.mlp.gate.weight",
        vec![num_experts, h],
        |i| ((i + 9) % 17) as f32 * 0.02 - 0.15,
    );
    push_bf16_tensor(
        &mut tensors,
        "mtp.layers.0.mlp.shared_expert.gate_proj.weight",
        vec![moe_interm, h],
        |i| ((i % 19) as f32 - 9.0) * 0.01,
    );
    push_bf16_tensor(
        &mut tensors,
        "mtp.layers.0.mlp.shared_expert.up_proj.weight",
        vec![moe_interm, h],
        |i| ((i % 17) as f32 - 8.0) * 0.01,
    );
    push_bf16_tensor(
        &mut tensors,
        "mtp.layers.0.mlp.shared_expert.down_proj.weight",
        vec![h, moe_interm],
        |i| ((i % 13) as f32 - 6.0) * 0.01,
    );
    push_bf16_tensor(
        &mut tensors,
        "mtp.layers.0.mlp.shared_expert_gate.weight",
        vec![1, h],
        |i| ((i % 7) as f32 - 3.0) * 0.01,
    );
    for e in 0..num_experts {
        let epfx = format!("mtp.layers.0.mlp.experts.{e}");
        push_autoround_expert_proj(
            &mut tensors,
            &format!("{epfx}.gate_proj"),
            moe_interm,
            h,
            128,
            500 + e * 3,
        );
        push_autoround_expert_proj(
            &mut tensors,
            &format!("{epfx}.up_proj"),
            moe_interm,
            h,
            128,
            500 + e * 3 + 1,
        );
        push_autoround_expert_proj(
            &mut tensors,
            &format!("{epfx}.down_proj"),
            h,
            moe_interm,
            128,
            500 + e * 3 + 2,
        );
    }

    write_safetensors_file(&dir.join("model.safetensors"), &tensors).unwrap();
    dir
}

#[test]
fn test_qwen4_exp_synthetic_2layer_tp1_tp2_radix_and_cuda_graph() {
    let dir = build_synthetic_qwen4_exp_dir("tp1_tp2");
    let prompt = vec![10i32, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120];
    let extended_prompt: Vec<i32> = (0..68).map(|i| 10 + (i * 3) % 200).collect();

    // 1. TP=1 eager vs CUDA Graph + MambaRadixCache prefix reuse
    let mut engine_tp1_eager =
        Qwen3_5Engine::load_from_dir(&dir, 1, KvFormatPolicy::Fp16, Some(64), false)
            .expect("load synthetic qwen4_exp TP=1 eager");
    let out_tp1_eager = engine_tp1_eager
        .generate_tokens(&prompt, 6, false)
        .expect("generate_tokens TP=1 eager");
    assert_eq!(out_tp1_eager.generated_token_ids.len(), 6);
    for &tok in &out_tp1_eager.generated_token_ids {
        assert!((0..256).contains(&tok), "invalid token id {tok}");
    }

    // MambaRadixCache prefix reuse: first call populates radix tree (68 tokens -> 64-token boundary checkpoint)
    let radix_run1 = engine_tp1_eager
        .generate_with_radix_cache(&extended_prompt, 4, 64, false)
        .expect("radix run 1");
    assert_eq!(radix_run1.cached_prefix_tokens, 0);
    // Second call with same 64-token prefix should hit 64 cached prefix tokens
    // and match the first run and a fresh uncached generate_tokens run bit-for-bit!
    let radix_run2 = engine_tp1_eager
        .generate_with_radix_cache(&extended_prompt, 4, 64, false)
        .expect("radix run 2");
    assert_eq!(radix_run2.cached_prefix_tokens, 64);
    assert_eq!(
        radix_run2.generated_token_ids, radix_run1.generated_token_ids,
        "MambaRadixCache prefix reuse must preserve GDN + PLE + QSA state identically"
    );
    let fresh_extended = engine_tp1_eager
        .generate_tokens(&extended_prompt, 4, false)
        .expect("fresh extended");
    assert_eq!(
        radix_run2.generated_token_ids, fresh_extended.generated_token_ids,
        "MambaRadixCache prefix reuse must match uncached generate_tokens"
    );

    // CUDA Graph capture & replay on TP=1
    let mut engine_tp1_graph =
        Qwen3_5Engine::load_from_dir(&dir, 1, KvFormatPolicy::Fp16, Some(64), true)
            .expect("load synthetic qwen4_exp TP=1 graph");
    let out_tp1_graph = engine_tp1_graph
        .generate_tokens(&prompt, 6, false)
        .expect("generate_tokens TP=1 graph");
    assert_eq!(
        out_tp1_graph.generated_token_ids, out_tp1_eager.generated_token_ids,
        "CUDA Graph decode on qwen4_exp TP=1 must match eager decode token-for-token"
    );
    drop(engine_tp1_graph);

    // 2. Native Qwen4 MTP speculative decoding (`K=3`, `M_verify=4`, dual-stream HC handoff + QSA step-0 top-k reuse) on TP=1
    let spec_cfg = w34_host::proposer::SpeculativeConfig {
        kind: w34_host::proposer::SpeculativeKind::Mtp,
        draft_model: None,
        mtp_k: 3,
    };
    engine_tp1_eager
        .enable_speculative(&spec_cfg)
        .expect("enable Qwen4 MTP on TP=1");
    let out_tp1_mtp = engine_tp1_eager
        .generate_with_radix_cache(&prompt, 6, 64, false)
        .expect("generate_with_radix_cache Qwen4 MTP TP=1");
    assert_eq!(
        out_tp1_mtp.generated_token_ids, out_tp1_eager.generated_token_ids,
        "Qwen4 MTP speculative decoding on TP=1 must match target greedy tokens losslessly"
    );
    let out_tp1_mtp_long1 = engine_tp1_eager
        .generate_with_radix_cache(&extended_prompt, 4, 64, false)
        .expect("Qwen4 MTP radix run 1 TP=1");
    let out_tp1_mtp_long2 = engine_tp1_eager
        .generate_with_radix_cache(&extended_prompt, 4, 64, false)
        .expect("Qwen4 MTP radix run 2 TP=1");
    assert_eq!(out_tp1_mtp_long2.cached_prefix_tokens, 64);
    assert_eq!(
        out_tp1_mtp_long1.generated_token_ids, fresh_extended.generated_token_ids,
        "Qwen4 MTP cold radix run must match target greedy tokens"
    );
    assert_eq!(
        out_tp1_mtp_long2.generated_token_ids, fresh_extended.generated_token_ids,
        "Qwen4 MTP warm radix prefix hit must match target greedy tokens"
    );
    drop(engine_tp1_eager);

    // 3. TP=2 / EP=2 eager + CUDA Graph + Qwen4 MTP parity against TP=1 (when 2 GPUs are visible)
    if w34_host::cuda::CudaDevice::set_current(1).is_ok() {
        let mut engine_tp2 =
            Qwen3_5Engine::load_from_dir(&dir, 2, KvFormatPolicy::Fp16, Some(64), true)
                .expect("load synthetic qwen4_exp TP=2 / EP=2");
        let out_tp2 = engine_tp2
            .generate_tokens(&prompt, 6, false)
            .expect("generate_tokens TP=2 / EP=2");
        assert_eq!(
            out_tp2.generated_token_ids, out_tp1_eager.generated_token_ids,
            "TP=2 / EP=2 greedy tokens must match TP=1 on qwen4_exp"
        );
        drop(engine_tp2);

        let mut engine_tp2_mtp =
            Qwen3_5Engine::load_from_dir(&dir, 2, KvFormatPolicy::Fp16, Some(64), false)
                .expect("load synthetic qwen4_exp TP=2 / EP=2 for MTP");
        engine_tp2_mtp
            .enable_speculative(&spec_cfg)
            .expect("enable Qwen4 MTP on TP=2 / EP=2");
        let out_tp2_mtp = engine_tp2_mtp
            .generate_with_radix_cache(&prompt, 6, 64, false)
            .expect("generate_with_radix_cache Qwen4 MTP TP=2 / EP=2");
        assert_eq!(
            out_tp2_mtp.generated_token_ids, out_tp2.generated_token_ids,
            "Qwen4 MTP speculative decoding on TP=2 / EP=2 must match target greedy tokens losslessly"
        );
    }

    let _ = fs::remove_dir_all(&dir);
}

#[test]
#[ignore = "requires local albucino/Qwen3.8-Flash-Next-W4A16-FP8PLE checkpoint"]
fn test_qwen4_exp_real_checkpoint_tp1_and_tp2() {
    let snap = Path::new(
        "/home/nchapman/.cache/huggingface/hub/models--albucino--Qwen3.8-Flash-Next-W4A16-FP8PLE/snapshots/ef554143369a706525336f6b42a09094835dc077",
    );
    assert!(
        snap.exists(),
        "missing real checkpoint at {}",
        snap.display()
    );

    // 1. TP=1 load, eager + CUDA Graph generation, and MambaRadixCache prefix reuse
    let t_load_tp1 = std::time::Instant::now();
    let mut engine_tp1 =
        Qwen3_5Engine::load_from_dir(snap, 1, KvFormatPolicy::Fp16, Some(128), true)
            .expect("load real 48-layer Qwen3.8-Flash-Next TP=1");
    eprintln!(
        "[QF-3] Loaded 48-layer Qwen3.8-Flash-Next on TP=1 in {:.2}s (kv_pages={}, state_slots={})",
        t_load_tp1.elapsed().as_secs_f64(),
        engine_tp1.ranks[0].kv_pool.num_pages,
        engine_tp1.ranks[0].state_pool.total_slots,
    );

    let prompt_ids = engine_tp1
        .encode("The capital of France is")
        .unwrap_or_else(|_| vec![785, 6722, 315, 9625, 374]);
    let mut dump_tp1 = w34_host::model::ForwardDump::default();
    engine_tp1.reset_all_states().unwrap();
    engine_tp1
        .step_tokens(&prompt_ids, 0, &[0], 0, 0, false, Some(&mut dump_tp1))
        .expect("step_tokens dump TP=1");
    eprintln!(
        "[QF-3 TP=1 dump] greedy={} top5_ids={:?} top5_lps={:?}",
        dump_tp1.greedy_token_id, dump_tp1.top5_ids, dump_tp1.top5_logprobs
    );
    let metrics_tp1 = engine_tp1
        .generate_tokens(&prompt_ids, 16, true)
        .expect("generate_tokens on real Qwen3.8-Flash-Next TP=1");
    let decoded_tp1 = engine_tp1
        .decode(&metrics_tp1.generated_token_ids)
        .unwrap_or_default();
    eprintln!(
        "[QF-3 TP=1] prompt_len={} gen_len={} ttft={:.2}ms decode={:.2} tok/s text={:?} ids={:?}",
        metrics_tp1.prompt_len,
        metrics_tp1.gen_len,
        metrics_tp1.ttft_ms,
        metrics_tp1.decode_tok_per_s,
        decoded_tp1,
        metrics_tp1.generated_token_ids,
    );
    assert!(
        decoded_tp1.starts_with(" Paris"),
        "expected ' Paris...' on TP=1, got {decoded_tp1:?}"
    );

    // Verify MambaRadixCache prefix reuse on real 48-layer checkpoint (>64 tokens)
    let mut long_prompt = Vec::with_capacity(72);
    while long_prompt.len() < 70 {
        long_prompt.extend_from_slice(&prompt_ids);
    }
    let r1 = engine_tp1
        .generate_with_radix_cache(&long_prompt, 8, 64, false)
        .expect("real radix run 1 TP=1");
    assert_eq!(r1.cached_prefix_tokens, 0);
    let r2 = engine_tp1
        .generate_with_radix_cache(&long_prompt, 8, 64, false)
        .expect("real radix run 2 TP=1");
    assert_eq!(r2.cached_prefix_tokens, 64);
    assert_eq!(
        r1.generated_token_ids, r2.generated_token_ids,
        "real 48-layer MambaRadixCache prefix reuse must match uncached run"
    );
    eprintln!(
        "[QF-3 TP=1 Radix] cached_prefix_tokens={} ttft_cold={:.2}ms ttft_warm={:.2}ms ids={:?}",
        r2.cached_prefix_tokens, r1.ttft_ms, r2.ttft_ms, r2.generated_token_ids,
    );
    drop(engine_tp1);

    // 2. TP=2 / EP=2 load, generation, and MambaRadixCache prefix reuse on 2x RTX 3090 Ti
    if w34_host::cuda::CudaDevice::set_current(1).is_ok() {
        let t_load_tp2 = std::time::Instant::now();
        let mut engine_tp2 =
            Qwen3_5Engine::load_from_dir(snap, 2, KvFormatPolicy::Fp16, Some(256), true)
                .expect("load real 48-layer Qwen3.8-Flash-Next TP=2 / EP=2");
        eprintln!(
            "[QF-3] Loaded 48-layer Qwen3.8-Flash-Next on TP=2 / EP=2 in {:.2}s (kv_pages={}, state_slots={})",
            t_load_tp2.elapsed().as_secs_f64(),
            engine_tp2.ranks[0].kv_pool.num_pages,
            engine_tp2.ranks[0].state_pool.total_slots,
        );
        let mut dump_tp2 = w34_host::model::ForwardDump::default();
        engine_tp2.reset_all_states().unwrap();
        engine_tp2
            .step_tokens(&prompt_ids, 0, &[0], 0, 0, false, Some(&mut dump_tp2))
            .expect("step_tokens dump TP=2");
        eprintln!(
            "[QF-3 TP=2 dump] greedy={} top5_ids={:?} top5_lps={:?}",
            dump_tp2.greedy_token_id, dump_tp2.top5_ids, dump_tp2.top5_logprobs
        );
        for name in [
            "embed",
            "layer_0_hidden_after",
            "layer_1_hidden_after",
            "layer_2_hidden_after",
            "layer_3_hidden_after",
            "layer_4_hidden_after",
            "layer_47_hidden_after",
            "final_norm",
        ] {
            if let (Some(t1), Some(t2)) = (
                dump_tp1.tensors_f32.get(name),
                dump_tp2.tensors_f32.get(name),
            ) {
                let md =
                    t1.1.iter()
                        .zip(t2.1.iter())
                        .map(|(a, b)| (a - b).abs())
                        .fold(0.0f32, f32::max);
                eprintln!("[QF-3 diff] {name}: max_abs_diff={md:.6}");
            }
        }
        let metrics_tp2 = engine_tp2
            .generate_tokens(&prompt_ids, 16, true)
            .expect("generate_tokens on real Qwen3.8-Flash-Next TP=2 / EP=2");
        let decoded_tp2 = engine_tp2
            .decode(&metrics_tp2.generated_token_ids)
            .unwrap_or_default();
        eprintln!(
            "[QF-3 TP=2] prompt_len={} gen_len={} ttft={:.2}ms decode={:.2} tok/s text={:?} ids={:?}",
            metrics_tp2.prompt_len,
            metrics_tp2.gen_len,
            metrics_tp2.ttft_ms,
            metrics_tp2.decode_tok_per_s,
            decoded_tp2,
            metrics_tp2.generated_token_ids,
        );
        assert!(!metrics_tp2.generated_token_ids.is_empty());

        let r1_tp2 = engine_tp2
            .generate_with_radix_cache(&long_prompt, 8, 64, false)
            .expect("real radix run 1 TP=2");
        let r2_tp2 = engine_tp2
            .generate_with_radix_cache(&long_prompt, 8, 64, false)
            .expect("real radix run 2 TP=2");
        assert_eq!(r2_tp2.cached_prefix_tokens, 64);
        assert_eq!(r1_tp2.generated_token_ids, r2_tp2.generated_token_ids);
        assert_eq!(
            r2_tp2.generated_token_ids, r2.generated_token_ids,
            "TP=2 / EP=2 70-token radix generation must match TP=1 token-for-token"
        );
        eprintln!(
            "[QF-3 TP=2 Radix] cached_prefix_tokens={} ttft_cold={:.2}ms ttft_warm={:.2}ms ids={:?}",
            r2_tp2.cached_prefix_tokens, r1_tp2.ttft_ms, r2_tp2.ttft_ms, r2_tp2.generated_token_ids,
        );
    }
}
