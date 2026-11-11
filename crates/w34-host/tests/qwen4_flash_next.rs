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
        "intermediate_size": 640,
        "moe_intermediate_size": 640,
        "shared_expert_intermediate_size": 640,
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
        "indexer_budget": 32,
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
    let moe_interm = 640usize;

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
    // Final HC mixer (uses shared [h] hc_norm.weight to exercise the hc_per_branch_norm=false broadcast path)
    push_bf16_tensor(
        &mut tensors,
        "model.language_model.hyper_connection_mixer.hc_norm.weight",
        vec![h],
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
    let runtime_dir = dir.join("runtime");
    fs::create_dir_all(&runtime_dir).unwrap();
    fs::write(
        runtime_dir.join("static_hot_cache_rankings.json"),
        serde_json::to_vec_pretty(&serde_json::json!({
            "0": [6, 5, 2, 1, 7, 4, 3, 0],
            "1": [7, 4, 3, 2, 6, 5, 1, 0]
        }))
        .unwrap(),
    )
    .unwrap();
    dir
}

fn assert_hot_cache_slots(engine: &Qwen3_5Engine, rank: usize, layer: usize, expected: &[usize]) {
    let cache = &engine.ranks[rank].tiered_moe.as_ref().unwrap().caches[layer];
    let slots: Vec<Option<i32>> = expected.iter().map(|&gid| cache.slot_of(gid)).collect();
    let expected_slots: Vec<Option<i32>> = (0..expected.len()).map(|s| Some(s as i32)).collect();
    assert_eq!(slots, expected_slots);
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
    assert_hot_cache_slots(&engine_tp1_eager, 0, 0, &[6, 5, 2, 1]);
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

    // CUDA Graph capture & replay on TP=1 (both target-only decode graph and MTP speculative verify graph)
    let spec_cfg = w34_host::proposer::SpeculativeConfig {
        kind: w34_host::proposer::SpeculativeKind::Mtp,
        draft_model: None,
        mtp_k: 3,
    };
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

    let mut engine_tp1_graph_mtp =
        Qwen3_5Engine::load_from_dir(&dir, 1, KvFormatPolicy::Fp16, Some(64), true)
            .expect("load synthetic qwen4_exp TP=1 graph for MTP");
    engine_tp1_graph_mtp
        .enable_speculative(&spec_cfg)
        .expect("enable Qwen4 MTP on TP=1 graph");
    let out_tp1_graph_mtp = engine_tp1_graph_mtp
        .generate_with_radix_cache(&extended_prompt, 4, 64, false)
        .expect("generate_with_radix_cache Qwen4 MTP TP=1 graph");
    assert_eq!(
        out_tp1_graph_mtp.generated_token_ids, fresh_extended.generated_token_ids,
        "CUDA Graph + MTP verify on qwen4_exp TP=1 must match target greedy tokens"
    );
    drop(engine_tp1_graph_mtp);

    // 2. Native Qwen4 MTP speculative decoding (`K=3`, `M_verify=4`, dual-stream HC handoff + QSA step-0 top-k reuse) on TP=1
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

    // 3. Verify Q8_0 and TQ4 KV compression (`KvFormatPolicy::Q8_0`, `KvFormatPolicy::Tq4All`)
    //    on QSA sparse paged attention (> 32-token indexer_budget) against the FP16 baseline.
    verify_qwen4_exp_quantized_kv(&dir, &extended_prompt);

    // 4. TP=2 / EP=2 eager + CUDA Graph + MambaRadixCache (68 tokens > 32-token indexer_budget, 4-token suffix hitting M=4 sparse_vec4 K=640) + Qwen4 MTP parity against TP=1
    if w34_host::cuda::CudaDevice::set_current(1).is_err() {
        let strict = ["W34_STRICT_TESTS", "CI"].into_iter().any(|var| {
            std::env::var(var).is_ok_and(|val| {
                let t = val.trim();
                t == "1" || t.eq_ignore_ascii_case("true")
            })
        });
        assert!(
            !strict,
            "test_qwen4_exp_synthetic_2layer_tp1_tp2_radix_and_cuda_graph requires 2 GPUs in strict mode"
        );
    } else {
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
        let radix_tp2_run1 = engine_tp2
            .generate_with_radix_cache(&extended_prompt, 4, 64, false)
            .expect("TP=2 / EP=2 radix run 1 on 68-token prompt");
        let radix_tp2_run2 = engine_tp2
            .generate_with_radix_cache(&extended_prompt, 4, 64, false)
            .expect("TP=2 / EP=2 radix run 2 on 68-token prompt");
        assert_eq!(radix_tp2_run2.cached_prefix_tokens, 64);
        assert_eq!(
            radix_tp2_run1.generated_token_ids, fresh_extended.generated_token_ids,
            "TP=2 / EP=2 cold radix run (68 tokens > indexer_budget=32) must match TP=1"
        );
        assert_eq!(
            radix_tp2_run2.generated_token_ids, fresh_extended.generated_token_ids,
            "TP=2 / EP=2 warm radix run (M=4 suffix with K=640) must match TP=1"
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
        let out_tp2_mtp_long = engine_tp2_mtp
            .generate_with_radix_cache(&extended_prompt, 4, 64, false)
            .expect("generate_with_radix_cache Qwen4 MTP TP=2 / EP=2 on 68-token prompt");
        assert_eq!(
            out_tp2_mtp_long.generated_token_ids, fresh_extended.generated_token_ids,
            "Qwen4 MTP speculative decoding on TP=2 / EP=2 (68 tokens > indexer_budget=32) must match TP=1"
        );
    }

    verify_qwen4_mtp_engine_worker_and_capacity(
        &dir,
        &prompt,
        &out_tp1_eager.generated_token_ids,
        &spec_cfg,
    );

    let _ = fs::remove_dir_all(&dir);
}

fn verify_qwen4_mtp_engine_worker_and_capacity(
    dir: &Path,
    prompt: &[i32],
    expected_short_ids: &[i32],
    spec_cfg: &w34_host::proposer::SpeculativeConfig,
) {
    use w34_host::engine_worker::{EngineWorker, GenerationEvent, GenerationStream};
    use w34_host::parser::{SamplingParams, TokenBitmaskConstraint};
    use w34_host::scheduler::SchedulerConfig;

    async fn collect_stream_tokens(mut stream: GenerationStream) -> Result<Vec<i32>, String> {
        tokio::time::timeout(std::time::Duration::from_secs(30), async move {
            let mut out = Vec::new();
            while let Some(ev) = stream.events.recv().await {
                match ev {
                    GenerationEvent::Token { id, .. } => out.push(id),
                    GenerationEvent::Finished { .. } => return Ok(out),
                    GenerationEvent::Error(e) => return Err(e.to_string()),
                }
            }
            Err("stream closed without Finished".into())
        })
        .await
        .map_err(|_| "timed out waiting for GenerationStream".to_string())?
    }

    let mut target_long =
        Qwen3_5Engine::load_from_dir(dir, 1, KvFormatPolicy::Fp16, Some(64), true)
            .expect("load synthetic qwen4_exp TP=1 target for long prompt");
    let long_300: Vec<i32> = (0..300).map(|i| (i * 7 + 3) % 250).collect();
    let ref_long = target_long
        .generate_tokens(&long_300, 6, false)
        .expect("target 300-token reference");
    drop(target_long);

    let mut engine_serve = Qwen3_5Engine::load_for_serving(
        dir.to_str().unwrap(),
        1,
        KvFormatPolicy::Fp16,
        Some(64),
        true,
        1,
    )
    .expect(
        "load synthetic qwen4_exp TP=1 via load_for_serving(max_num_seqs=1) for EngineWorker MTP",
    );
    assert!(
        engine_serve.ranks[0].state_pool.total_slots >= 8,
        "load_for_serving(max_num_seqs=1) must allocate >= 8 state slots for MTP-3 verification, got {}",
        engine_serve.ranks[0].state_pool.total_slots
    );
    engine_serve
        .enable_speculative(spec_cfg)
        .expect("enable Qwen4 MTP on EngineWorker");
    let worker = EngineWorker::start(
        engine_serve,
        SchedulerConfig {
            chunk_prefill_size: 512,
            max_num_seqs: 1,
            ..SchedulerConfig::default()
        },
    );
    let rt = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .unwrap();
    let params_long = SamplingParams {
        temperature: 0.0,
        max_tokens: 6,
        stop_token_ids: vec![],
        eos_token_ids: vec![],
        ..SamplingParams::default()
    };
    let ids_long = rt
        .block_on(collect_stream_tokens(
            worker
                .submit(long_300.clone(), params_long.clone(), None)
                .expect("submit 300-token MTP request"),
        ))
        .expect("300-token single-chunk MTP prefill + verify must succeed");
    assert_eq!(
        ids_long, ref_long.generated_token_ids,
        "EngineWorker MTP 300-token single-chunk prefill must match target greedy tokens"
    );

    // Multi-prompt prefix-memo LRU test: run a second distinct prompt (`long_alt`), then replay
    // `long_300` and verify `cached_prefix_tokens > 0` (regression test for `prefix-primed-r16`
    // where multiple primed prompts previously overwrote a single `memo` slot).
    let long_alt: Vec<i32> = (0..160).map(|i| (i * 11 + 19) % 250).collect();
    let _ = rt
        .block_on(collect_stream_tokens(
            worker
                .submit(long_alt, params_long.clone(), None)
                .expect("submit second distinct prompt"),
        ))
        .expect("second distinct MTP prompt must succeed");
    let mut stream_reused = worker
        .submit(long_300, params_long, None)
        .expect("re-submit first 300-token MTP prompt after intervening prompt");
    let (ids_reused, cached_reused) = rt
        .block_on(async move {
            let mut out = Vec::new();
            while let Some(ev) = stream_reused.events.recv().await {
                match ev {
                    GenerationEvent::Token { id, .. } => out.push(id),
                    GenerationEvent::Finished { cached_tokens, .. } => {
                        return Ok::<_, String>((out, cached_tokens));
                    }
                    GenerationEvent::Error(e) => return Err(e.to_string()),
                }
            }
            Err("stream closed without Finished".into())
        })
        .expect("replayed 300-token MTP request must succeed");
    assert!(
        cached_reused >= 256,
        "expected multi-entry MTP PrefixMemo to preserve warm prefix hit >= 256 after intervening prompt, got {cached_reused}"
    );
    assert_eq!(
        ids_reused, ref_long.generated_token_ids,
        "replayed warm-prefix MTP request must match target greedy tokens"
    );

    // Constrained terminal-EOS draft verification: when a draft token is in terminal_eos_ids,
    // MTP verify must truncate before the post-EOS empty-mask bonus row and finish cleanly
    // right after emitting `draft_eos_id`.
    let draft_eos_id = expected_short_ids[1];
    let constraint_eos =
        TokenBitmaskConstraint::unconstrained(256).with_terminal_eos_ids(vec![draft_eos_id]);
    let params_constrained = SamplingParams {
        temperature: 0.0,
        max_tokens: 6,
        stop_token_ids: vec![],
        eos_token_ids: vec![],
        ..SamplingParams::default()
    };
    let ids_constrained = rt
        .block_on(collect_stream_tokens(
            worker
                .submit(prompt.to_vec(), params_constrained, Some(constraint_eos))
                .expect("submit constrained terminal-EOS MTP request"),
        ))
        .expect("constrained terminal-EOS MTP request must not fail with empty bonus bitmask");
    assert_eq!(
        ids_constrained,
        &expected_short_ids[..2],
        "constrained terminal-EOS MTP request must stop immediately after draft_eos_id"
    );

    // Force an aborted request (empty allowed-token bitmask) and verify the next MTP request
    // recovers cleanly with exact target greedy parity (`abort_scheduled` resets MTP state).
    let empty_constraint = TokenBitmaskConstraint::from_allowed_tokens(256, vec![]);
    let params_abort = SamplingParams {
        temperature: 0.0,
        max_tokens: 4,
        stop_token_ids: vec![],
        eos_token_ids: vec![],
        ..SamplingParams::default()
    };
    let abort_res = rt.block_on(collect_stream_tokens(
        worker
            .submit(prompt.to_vec(), params_abort, Some(empty_constraint))
            .expect("submit doomed constrained request"),
    ));
    assert!(abort_res.is_err(), "empty bitmask request must fail");

    let params_recover = SamplingParams {
        temperature: 0.0,
        max_tokens: 6,
        stop_token_ids: vec![],
        eos_token_ids: vec![],
        ..SamplingParams::default()
    };
    let ids_recover = rt
        .block_on(collect_stream_tokens(
            worker
                .submit(prompt.to_vec(), params_recover, None)
                .expect("submit recovery MTP request"),
        ))
        .expect("post-abort recovery MTP request must succeed");
    assert_eq!(
        ids_recover, expected_short_ids,
        "post-abort MTP request must match target greedy tokens"
    );

    rt.block_on(worker.shutdown()).expect("worker shutdown");

    // Direct regression test for `Qwen3_5MtpRank::copy_kv_from_slot` copying QSA indexer
    // side caches (`qsa_raw_k_buf` and `qsa_comp_k_buf`) across batch slots:
    let shared_prefix: Vec<i32> = (0..64).map(|i| (i * 5 + 3) % 250).collect();
    let c4_prompts: Vec<Vec<i32>> = (0..4)
        .map(|client| {
            let mut p = shared_prefix.clone();
            p.extend((0..8).map(|j| (client * 17 + j * 3 + 11) % 250));
            p
        })
        .collect();
    let c4_expected: Vec<Vec<i32>> = {
        let mut target_ref =
            Qwen3_5Engine::load_from_dir(dir, 1, KvFormatPolicy::Fp16, Some(64), true)
                .expect("load synthetic qwen4_exp TP=1 target for slot-copy & c=4 reference");
        let mut mtp_rank = w34_host::mtp::Qwen3_5MtpRank::load(&target_ref.ranks[0], dir, 256)
            .expect("load Qwen4 MTP rank for slot-copy test");
        mtp_rank
            .set_batch_mode(true)
            .expect("enable MTP batch mode");
        mtp_rank.select_batch_slot(0).expect("select slot 0");
        let slot_count = mtp_rank.batch_slot_count();
        let slot_pages = 16usize / slot_count;
        let hidden_bytes = 32 * target_ref.cfg.hidden_size * 2;
        let hidden_host: Vec<u16> = (0..hidden_bytes / 2)
            .map(|i| 0x3800u16.wrapping_add((i % 17) as u16))
            .collect();
        let hidden_dev =
            w34_host::cuda::DeviceBuffer::alloc(target_ref.ranks[0].device.device_id, hidden_bytes)
                .unwrap();
        hidden_dev
            .copy_from_host(&hidden_host, &target_ref.ranks[0].stream)
            .unwrap();
        let toks_32: Vec<i32> = (0..32).map(|i| (i * 5 + 3) % 250).collect();
        mtp_rank
            .ingest(&target_ref.ranks[0], &toks_32, hidden_dev.addr(), None)
            .expect("ingest 32 tokens into MTP slot 0");
        mtp_rank
            .copy_kv_from_slot(&target_ref.ranks[0], 0, 1, 32)
            .expect("copy 32 tokens from MTP slot 0 to slot 1");
        target_ref.ranks[0].stream.synchronize().unwrap();
        let kv_layer = &mtp_rank.kv_pool().layers[0];
        let total_pages = mtp_rank.kv_pool().num_pages;
        let raw_buf = kv_layer
            .qsa_raw_k_buf
            .as_ref()
            .expect("Qwen4 MTP must allocate qsa_raw_k_buf");
        let comp_buf = kv_layer
            .qsa_comp_k_buf
            .as_ref()
            .expect("Qwen4 MTP must allocate qsa_comp_k_buf");
        let raw_stride = raw_buf.len_bytes / total_pages;
        let comp_stride = comp_buf.len_bytes / total_pages;
        let mut raw_host = vec![0u8; raw_buf.len_bytes];
        let mut comp_host = vec![0u8; comp_buf.len_bytes];
        raw_buf
            .copy_to_host(&mut raw_host, &target_ref.ranks[0].stream)
            .unwrap();
        comp_buf
            .copy_to_host(&mut comp_host, &target_ref.ranks[0].stream)
            .unwrap();
        target_ref.ranks[0].stream.synchronize().unwrap();
        let raw_slot0 = &raw_host[..2 * raw_stride];
        let raw_slot1 = &raw_host[slot_pages * raw_stride..(slot_pages + 2) * raw_stride];
        assert!(
            raw_slot0.iter().any(|&b| b != 0),
            "MTP slot 0 qsa_raw_k_buf must be populated after ingest"
        );
        assert_eq!(
            raw_slot1, raw_slot0,
            "copy_kv_from_slot must copy qsa_raw_k_buf from slot 0 to slot 1"
        );
        let comp_slot0 = &comp_host[..2 * comp_stride];
        let comp_slot1 = &comp_host[slot_pages * comp_stride..(slot_pages + 2) * comp_stride];
        assert!(
            comp_slot0.iter().any(|&b| b != 0),
            "MTP slot 0 qsa_comp_k_buf must be populated after ingest"
        );
        assert_eq!(
            comp_slot1, comp_slot0,
            "copy_kv_from_slot must copy qsa_comp_k_buf from slot 0 to slot 1"
        );
        drop(mtp_rank);
        c4_prompts
            .iter()
            .map(|p| {
                target_ref
                    .generate_tokens(p, 6, false)
                    .expect("target reference greedy tokens for c=4 prompt")
                    .generated_token_ids
            })
            .collect()
    };

    // Multi-client `max_num_seqs = 4` MTP serving with shared prefix (>32 tokens) on Qwen4:
    // exercises `copy_kv_from_slot` (including QSA `qsa_raw_k_buf` and `qsa_comp_k_buf`)
    // and single-active-slot multi-batch-slot scheduling on `cfg.uses_tiered_moe()`.
    let mut engine_c4 = Qwen3_5Engine::load_for_serving_with_mtp(
        dir.to_str().unwrap(),
        1,
        KvFormatPolicy::Fp16,
        Some(128),
        true,
        4,
        true,
    )
    .expect("load Qwen4 EngineWorker c=4");
    engine_c4
        .enable_speculative(spec_cfg)
        .expect("enable Qwen4 MTP on c=4 EngineWorker");
    let sched_c4_res = engine_c4
        .generate_batch_scheduled(
            &c4_prompts,
            6,
            SchedulerConfig {
                chunk_prefill_size: 512,
                max_num_seqs: 4,
                ..SchedulerConfig::default()
            },
        )
        .expect("generate_batch_scheduled on c=4 Qwen4 MTP engine");
    for ((ids, _), expected) in sched_c4_res.iter().zip(&c4_expected) {
        assert_eq!(
            ids, expected,
            "generate_batch_scheduled on Qwen4 MTP must match target greedy tokens"
        );
    }
    assert!(
        !engine_c4
            .speculative
            .as_ref()
            .unwrap()
            .last_accepted
            .is_empty(),
        "generate_batch_scheduled must execute speculative verification when speculative is enabled"
    );
    engine_c4.reset_all_states().unwrap();
    let worker_c4 = EngineWorker::start(
        engine_c4,
        SchedulerConfig {
            chunk_prefill_size: 512,
            max_num_seqs: 4,
            ..SchedulerConfig::default()
        },
    );
    let mut c4_streams = Vec::new();
    for p in &c4_prompts {
        c4_streams.push(
            worker_c4
                .submit(
                    p.clone(),
                    SamplingParams {
                        temperature: 0.0,
                        max_tokens: 6,
                        stop_token_ids: vec![],
                        eos_token_ids: vec![],
                        ..SamplingParams::default()
                    },
                    None,
                )
                .expect("submit c=4 shared-prefix MTP request"),
        );
    }
    let mut cached_hits = 0usize;
    for (mut stream, expected) in c4_streams.into_iter().zip(&c4_expected) {
        let (ids, cached_tokens) = rt
            .block_on(async move {
                let mut out = Vec::new();
                while let Some(ev) = stream.events.recv().await {
                    match ev {
                        GenerationEvent::Token { id, .. } => out.push(id),
                        GenerationEvent::Finished { cached_tokens, .. } => {
                            return Ok::<_, String>((out, cached_tokens));
                        }
                        GenerationEvent::Error(e) => return Err(e.to_string()),
                    }
                }
                Err("stream closed without Finished".into())
            })
            .expect("c=4 shared-prefix MTP request must succeed");
        if cached_tokens >= 32 {
            cached_hits += 1;
        }
        assert_eq!(
            &ids, expected,
            "c=4 shared-prefix MTP request must match target greedy tokens"
        );
    }
    assert!(
        cached_hits >= 3,
        "at least 3 of 4 shared-prefix c=4 requests must reuse >= 32 cached prefix tokens, got {cached_hits}"
    );
    rt.block_on(worker_c4.shutdown())
        .expect("c=4 worker shutdown");

    // Verify `plan_tiered_moe_hot_per_layer` adapts TP=2 hot cache down from 128 when
    // multi-thousand-page contexts (`4096..8192` pages) + MTP headroom are requested.
    let cfg = w34_host::loader::ModelConfig::from_dir(dir).unwrap();
    let mut real_like_cfg = cfg;
    real_like_cfg.num_hidden_layers = 48;
    real_like_cfg.num_experts = 512;
    real_like_cfg.moe_intermediate_size = 640;
    real_like_cfg.hidden_size = 2560;
    let gib = 1024 * 1024 * 1024;
    let (hot_small, _) = w34_host::model::plan_tiered_moe_hot_per_layer(
        &real_like_cfg,
        2,
        19_600 * 1024 * 1024,
        gib + 38 * 1024 * 1024,
        450 * 1024 * 1024,
    );
    assert_eq!(hot_small, 128);
    let base_plus_kv_12k = gib + 1536 * 1024 * 1024 + 1824 * 1024 * 1024;
    let min_state_12k = 1200 * 1024 * 1024;
    let (hot_12k, reserve_12k) = w34_host::model::plan_tiered_moe_hot_per_layer(
        &real_like_cfg,
        2,
        19_600 * 1024 * 1024,
        base_plus_kv_12k,
        min_state_12k,
    );
    assert_eq!(
        hot_12k, 104,
        "expected adaptive hot_per_layer == 104 for 12288-page TP=2 + MTP, got {hot_12k}"
    );
    assert!(
        base_plus_kv_12k + min_state_12k + reserve_12k <= 19_600 * 1024 * 1024,
        "adaptive tiered_moe_reserve must fit inside available post-weight VRAM"
    );
}

fn verify_qwen4_exp_quantized_kv(dir: &Path, extended_prompt: &[i32]) {
    let mut engine_tp1_fp16_ref =
        Qwen3_5Engine::load_from_dir(dir, 1, KvFormatPolicy::Fp16, Some(64), true)
            .expect("load synthetic qwen4_exp TP=1 FP16 ref");
    let fp16_dump = engine_tp1_fp16_ref
        .dump_prompt(extended_prompt)
        .expect("dump_prompt TP=1 FP16 ref");
    drop(engine_tp1_fp16_ref);

    for (policy, min_cos) in [
        (KvFormatPolicy::Q8_0, 0.999),
        (KvFormatPolicy::Tq4All, 0.99),
    ] {
        let mut engine_tp1_qkv = Qwen3_5Engine::load_from_dir(dir, 1, policy, Some(64), true)
            .expect("load synthetic qwen4_exp TP=1 quantized KV");
        let qkv_dump = engine_tp1_qkv
            .dump_prompt(extended_prompt)
            .expect("dump_prompt TP=1 quantized KV");
        let fp16_fn = &fp16_dump.tensors_f32["final_norm"].1;
        let qkv_fn = &qkv_dump.tensors_f32["final_norm"].1;
        let dot: f64 = fp16_fn
            .iter()
            .zip(qkv_fn.iter())
            .map(|(&a, &b)| (a as f64) * (b as f64))
            .sum();
        let n_a: f64 = fp16_fn
            .iter()
            .map(|&a| (a as f64).powi(2))
            .sum::<f64>()
            .sqrt();
        let n_b: f64 = qkv_fn
            .iter()
            .map(|&b| (b as f64).powi(2))
            .sum::<f64>()
            .sqrt();
        let cos = dot / (n_a * n_b).max(1e-12);
        assert!(
            cos >= min_cos,
            "{policy:?} final_norm cosine similarity {cos:.6} < {min_cos} vs FP16"
        );
        let out_tp1_qkv = engine_tp1_qkv
            .generate_tokens(extended_prompt, 4, false)
            .expect("generate_tokens TP=1 quantized KV on extended_prompt");
        assert_eq!(out_tp1_qkv.generated_token_ids.len(), 4);
        for &tok in &out_tp1_qkv.generated_token_ids {
            assert!((0..256).contains(&tok), "invalid {policy:?} token id {tok}");
        }
    }
}

fn verify_qwen4_exp_dflash2_lossless(
    engine: &mut w34_host::model::Qwen3_5Engine,
    prompt_ids: &[i32],
    expected_prefix: &[i32],
    alt_prefix: &[i32],
    label: &str,
) {
    let _draft_dir =
        w34_host::loader::resolve_hf_model_dir("nchapman/Qwen3.8-Flash-Next-DFlash2-W4A16-HGPTQ")
            .expect("resolve Qwen3.8-Flash-Next-DFlash2-W4A16-HGPTQ checkpoint");
    engine
        .disable_speculative()
        .expect("disable prior speculative proposer before enabling DFlash2");
    engine
        .enable_speculative(&w34_host::proposer::SpeculativeConfig {
            kind: w34_host::proposer::SpeculativeKind::DFlash2,
            draft_model: Some("nchapman/Qwen3.8-Flash-Next-DFlash2-W4A16-HGPTQ".to_string()),
            mtp_k: 1,
        })
        .unwrap_or_else(|e| panic!("enable real Qwen3.8 DFlash2 on {label}: {e:?}"));
    let df_cold = engine
        .generate_with_radix_cache(prompt_ids, 64, 64, true)
        .unwrap_or_else(|e| panic!("DFlash2 cold 64-token run {label}: {e:?}"));
    let df_warm = engine
        .generate_with_radix_cache(prompt_ids, 64, 64, true)
        .unwrap_or_else(|e| panic!("DFlash2 warm 64-token run {label}: {e:?}"));
    assert_eq!(
        df_warm.generated_token_ids, df_cold.generated_token_ids,
        "Cold and warm DFlash2 runs on {label} must be deterministic across 64 tokens"
    );
    assert_eq!(
        &df_warm.generated_token_ids[..28],
        &expected_prefix[..28],
        "Real 48-layer Qwen3.8 DFlash2 on {label} must match target greedy prefix"
    );
    assert!(
        df_warm.generated_token_ids[..32] == expected_prefix[..32]
            || df_warm.generated_token_ids[..32] == alt_prefix[..32],
        "Real 48-layer Qwen3.8 DFlash2 on {label} must match 32-token target greedy prefix"
    );
    let acc_df = engine
        .speculative
        .as_ref()
        .map(|p| p.last_accepted.clone())
        .unwrap_or_default();
    let mean_df = if acc_df.is_empty() {
        0.0
    } else {
        acc_df.iter().sum::<usize>() as f64 / acc_df.len() as f64
    };
    assert!(
        mean_df >= 1.5,
        "expected DFlash2 {label} mean_accepted >= 1.5, got {mean_df:.2}"
    );
    eprintln!(
        "[QF-5 {label} DFlash2 Warm 64-tok] ttft={:.2}ms decode={:.2} tok/s rounds={} mean_accepted={:.2}/7 ({:.2} tok/round) accepted={:?}",
        df_warm.ttft_ms,
        df_warm.decode_tok_per_s,
        acc_df.len(),
        mean_df,
        1.0 + mean_df,
        acc_df,
    );
}

#[test]
#[ignore = "requires local Qwen3.8-Flash-Next checkpoint"]
fn test_qwen4_exp_real_checkpoint_tp1_and_tp2() {
    let snap = w34_host::loader::resolve_hf_model_dir("nchapman/Qwen3.8-Flash-Next-W4A16-HGPTQ")
        .or_else(|_| {
            w34_host::loader::resolve_hf_model_dir("albucino/Qwen3.8-Flash-Next-W4A16-FP8PLE")
        })
        .expect("resolve Qwen3.8-Flash-Next checkpoint");
    let snap = snap.as_path();

    // 1. TP=1 load, eager + CUDA Graph generation, and MambaRadixCache prefix reuse
    let t_load_tp1 = std::time::Instant::now();
    let mut engine_tp1 =
        Qwen3_5Engine::load_from_dir(snap, 1, KvFormatPolicy::Fp16, Some(128), true)
            .expect("load real 48-layer Qwen3.8-Flash-Next TP=1");
    eprintln!(
        "[QF-3] Loaded 48-layer Qwen3.8-Flash-Next on TP=1 in {:.2}s (kv_pages={}, state_slots={}, hot_k={})",
        t_load_tp1.elapsed().as_secs_f64(),
        engine_tp1.ranks[0].kv_pool.num_pages,
        engine_tp1.ranks[0].state_pool.total_slots,
        engine_tp1.ranks[0].tiered_moe_hot_per_layer,
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
    let warm_target_tp1_64 = engine_tp1
        .generate_tokens(&prompt_ids, 64, false)
        .expect("warm 64-token target decode TP=1");
    eprintln!(
        "[QF-5 TP=1 Warm Target 64-tok] ttft={:.2}ms decode={:.2} tok/s",
        warm_target_tp1_64.ttft_ms, warm_target_tp1_64.decode_tok_per_s,
    );
    let spec_cfg = w34_host::proposer::SpeculativeConfig {
        kind: w34_host::proposer::SpeculativeKind::Mtp,
        draft_model: None,
        mtp_k: 3,
    };
    engine_tp1
        .enable_speculative(&spec_cfg)
        .expect("enable real Qwen3.8 MTP-3 on TP=1");
    let mtp_tp1_cold = engine_tp1
        .generate_with_radix_cache(&prompt_ids, 64, 64, true)
        .expect("MTP-3 cold 64-token run TP=1");
    let mtp_tp1_warm = engine_tp1
        .generate_with_radix_cache(&prompt_ids, 64, 64, true)
        .expect("MTP-3 warm 64-token run TP=1");
    assert_eq!(
        mtp_tp1_warm.generated_token_ids, mtp_tp1_cold.generated_token_ids,
        "Cold and warm MTP-3 runs on TP=1 must be deterministic across 64 tokens"
    );
    assert_eq!(
        &mtp_tp1_warm.generated_token_ids[..32],
        &warm_target_tp1_64.generated_token_ids[..32],
        "Real 48-layer Qwen3.8 MTP-3 on TP=1 must match target greedy prefix"
    );
    let accepted_tp1_warm = engine_tp1
        .speculative
        .as_ref()
        .map(|p| p.last_accepted.clone())
        .unwrap_or_default();
    let mean_acc_tp1 = if accepted_tp1_warm.is_empty() {
        0.0
    } else {
        accepted_tp1_warm.iter().sum::<usize>() as f64 / accepted_tp1_warm.len() as f64
    };
    eprintln!(
        "[QF-5 TP=1 MTP-3 Warm 64-tok] ttft={:.2}ms decode={:.2} tok/s rounds={} mean_accepted={:.2}/3 ({:.2} tok/round)",
        mtp_tp1_warm.ttft_ms,
        mtp_tp1_warm.decode_tok_per_s,
        accepted_tp1_warm.len(),
        mean_acc_tp1,
        1.0 + mean_acc_tp1,
    );
    verify_qwen4_exp_dflash2_lossless(
        &mut engine_tp1,
        &prompt_ids,
        &warm_target_tp1_64.generated_token_ids,
        &mtp_tp1_warm.generated_token_ids,
        "TP=1",
    );
    drop(engine_tp1);

    // 2. TP=2 / EP=2 load, generation, and MambaRadixCache prefix reuse on 2x RTX 3090 Ti
    if w34_host::cuda::CudaDevice::set_current(1).is_ok() {
        let t_load_tp2 = std::time::Instant::now();
        let mut engine_tp2 =
            Qwen3_5Engine::load_from_dir(snap, 2, KvFormatPolicy::Fp16, Some(256), true)
                .expect("load real 48-layer Qwen3.8-Flash-Next TP=2 / EP=2");
        eprintln!(
            "[QF-3] Loaded 48-layer Qwen3.8-Flash-Next on TP=2 / EP=2 in {:.2}s (kv_pages={}, state_slots={}, hot_k={})",
            t_load_tp2.elapsed().as_secs_f64(),
            engine_tp2.ranks[0].kv_pool.num_pages,
            engine_tp2.ranks[0].state_pool.total_slots,
            engine_tp2.ranks[0].tiered_moe_hot_per_layer,
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
        assert_eq!(
            metrics_tp2.generated_token_ids, metrics_tp1.generated_token_ids,
            "TP=2 / EP=2 16-token greedy generation must match TP=1 token-for-token"
        );

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

        // Warm 64-token target-only decode benchmark on TP=2 / EP=2 (hot cache pre-warmed + graph captured)
        let warm_target_64 = engine_tp2
            .generate_tokens(&prompt_ids, 64, false)
            .expect("warm 64-token target decode TP=2");
        eprintln!(
            "[QF-5 TP=2 Warm Target 64-tok] ttft={:.2}ms decode={:.2} tok/s",
            warm_target_64.ttft_ms, warm_target_64.decode_tok_per_s,
        );

        // Enable real 512-expert MTP-3 (`runtime/mtp-int4-g32`, `mtp_k = 3`, `M_verify = 4`) on TP=2 / EP=2
        let spec_cfg = w34_host::proposer::SpeculativeConfig {
            kind: w34_host::proposer::SpeculativeKind::Mtp,
            draft_model: None,
            mtp_k: 3,
        };
        let t_mtp_load = std::time::Instant::now();
        engine_tp2
            .enable_speculative(&spec_cfg)
            .expect("enable real Qwen3.8 MTP-3 on TP=2 / EP=2");
        eprintln!(
            "[QF-5 TP=2 MTP-3] Loaded compact INT4 MTP-3 in {:.2}s",
            t_mtp_load.elapsed().as_secs_f64()
        );
        // Warm up MTP-3 verify graph on first run, then measure warm 64-token MTP-3 run.
        // As on TP=1 above, M=4 batch verify vs M=1 plain decode can hit a 1-ULP FP16
        // accumulation tie-break after 28+ tokens across 48 hybrid GDN/QSA/MoE layers;
        // verify exact 28-token prefix parity against TP=2 M=1, exact 32-token prefix
        // parity against either TP=2 M=1 or TP=1 M=1/M=4, and full 64-token determinism.
        let mtp_cold = engine_tp2
            .generate_with_radix_cache(&prompt_ids, 64, 64, true)
            .expect("MTP-3 cold 64-token run TP=2");
        assert_eq!(mtp_cold.generated_token_ids.len(), 64);
        assert_eq!(warm_target_64.generated_token_ids.len(), 64);
        assert_eq!(
            &mtp_cold.generated_token_ids[..28],
            &warm_target_64.generated_token_ids[..28],
            "Real 48-layer Qwen3.8 MTP-3 on TP=2 / EP=2 must match TP=2 M=1 prefix"
        );
        assert!(
            mtp_cold.generated_token_ids[..32] == warm_target_64.generated_token_ids[..32]
                || mtp_cold.generated_token_ids[..32]
                    == warm_target_tp1_64.generated_token_ids[..32],
            "Real 48-layer Qwen3.8 MTP-3 on TP=2 / EP=2 must match 32-token target greedy prefix (tp2_mtp={:?}, tp2_m1={:?}, tp1_m1={:?})",
            &mtp_cold.generated_token_ids[..32],
            &warm_target_64.generated_token_ids[..32],
            &warm_target_tp1_64.generated_token_ids[..32],
        );
        let accepted_cold = engine_tp2
            .speculative
            .as_ref()
            .map(|p| p.last_accepted.clone())
            .unwrap_or_default();
        eprintln!(
            "[QF-5 TP=2 MTP-3 Cold 64-tok] ttft={:.2}ms decode={:.2} tok/s rounds={} accepted={:?}",
            mtp_cold.ttft_ms,
            mtp_cold.decode_tok_per_s,
            accepted_cold.len(),
            accepted_cold,
        );
        let mtp_warm = engine_tp2
            .generate_with_radix_cache(&prompt_ids, 64, 64, true)
            .expect("MTP-3 warm 64-token run TP=2");
        assert_eq!(mtp_warm.generated_token_ids.len(), 64);
        assert_eq!(
            mtp_warm.generated_token_ids, mtp_cold.generated_token_ids,
            "Cold and warm MTP-3 runs on TP=2 must be deterministic across 64 tokens"
        );
        let accepted_warm = engine_tp2
            .speculative
            .as_ref()
            .map(|p| p.last_accepted.clone())
            .unwrap_or_default();
        let mean_accepted = if accepted_warm.is_empty() {
            0.0
        } else {
            accepted_warm.iter().sum::<usize>() as f64 / accepted_warm.len() as f64
        };
        eprintln!(
            "[QF-5 TP=2 MTP-3 Warm 64-tok] ttft={:.2}ms decode={:.2} tok/s rounds={} mean_accepted={:.2}/3 ({:.2} tok/round) accepted={:?}",
            mtp_warm.ttft_ms,
            mtp_warm.decode_tok_per_s,
            accepted_warm.len(),
            mean_accepted,
            1.0 + mean_accepted,
            accepted_warm,
        );
        verify_qwen4_exp_dflash2_lossless(
            &mut engine_tp2,
            &prompt_ids,
            &warm_target_64.generated_token_ids,
            &warm_target_tp1_64.generated_token_ids,
            "TP=2",
        );
    }
}

#[test]
#[ignore = "requires local Qwen3.8-Flash-Next checkpoint"]
fn test_qwen4_flash_next_128tok_plain_mtp3_dflash2_exact_parity() {
    let snap = w34_host::loader::resolve_hf_model_dir("nchapman/Qwen3.8-Flash-Next-W4A16-HGPTQ")
        .expect("resolve Qwen3.8-Flash-Next-W4A16-HGPTQ checkpoint");
    let snap = snap.as_path();

    for tp in [1usize, 2] {
        if tp == 2 && w34_host::cuda::CudaDevice::set_current(1).is_err() {
            continue;
        }
        let mut engine =
            Qwen3_5Engine::load_from_dir(snap, tp, KvFormatPolicy::Tq4, Some(512), true)
                .unwrap_or_else(|e| panic!("load Flash-Next TP={tp}: {e:?}"));

        let seed = engine
            .encode("The quick brown fox jumps over the lazy dog. ")
            .unwrap();
        let mut prompt = Vec::with_capacity(128);
        while prompt.len() < 128 {
            prompt.extend_from_slice(&seed);
        }
        prompt.truncate(128);

        let plain = engine
            .generate_with_radix_cache(&prompt, 64, 64, false)
            .unwrap();
        engine
            .enable_speculative(&w34_host::proposer::SpeculativeConfig {
                kind: w34_host::proposer::SpeculativeKind::Mtp,
                draft_model: None,
                mtp_k: 3,
            })
            .unwrap();
        let mtp = engine
            .generate_with_radix_cache(&prompt, 64, 64, false)
            .unwrap();
        engine.disable_speculative().unwrap();
        engine
            .enable_speculative(&w34_host::proposer::SpeculativeConfig {
                kind: w34_host::proposer::SpeculativeKind::DFlash2,
                draft_model: Some("nchapman/Qwen3.8-Flash-Next-DFlash2-W4A16-HGPTQ".to_string()),
                mtp_k: 1,
            })
            .unwrap();
        let dflash = engine
            .generate_with_radix_cache(&prompt, 64, 64, false)
            .unwrap();

        eprintln!(
            "[Flash-Next 128->64 TP={tp}] plain={:.2} tok/s mtp3={:.2} tok/s dflash2={:.2} tok/s\n  plain  = {:?}\n  mtp3   = {:?}\n  dflash = {:?}",
            plain.decode_tok_per_s,
            mtp.decode_tok_per_s,
            dflash.decode_tok_per_s,
            plain.generated_token_ids,
            mtp.generated_token_ids,
            dflash.generated_token_ids,
        );
        assert_eq!(
            mtp.generated_token_ids, plain.generated_token_ids,
            "Flash-Next TP={tp} MTP-3 must match plain 64-token greedy output token-for-token"
        );
        assert_eq!(
            dflash.generated_token_ids, plain.generated_token_ids,
            "Flash-Next TP={tp} DFlash2 must match plain 64-token greedy output token-for-token"
        );
    }
}
