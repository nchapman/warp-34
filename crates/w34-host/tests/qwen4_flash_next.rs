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
    //    on QSA sparse paged attention (> 32-token indexer_budget)
    for policy in [KvFormatPolicy::Q8_0, KvFormatPolicy::Tq4All] {
        let mut engine_tp1_qkv = Qwen3_5Engine::load_from_dir(&dir, 1, policy, Some(64), true)
            .expect("load synthetic qwen4_exp TP=1 quantized KV");
        let out_tp1_qkv = engine_tp1_qkv
            .generate_tokens(&extended_prompt, 4, false)
            .expect("generate_tokens TP=1 quantized KV on extended_prompt");
        assert_eq!(out_tp1_qkv.generated_token_ids.len(), 4);
        for &tok in &out_tp1_qkv.generated_token_ids {
            assert!((0..256).contains(&tok), "invalid {policy:?} token id {tok}");
        }
    }

    // 4. TP=2 / EP=2 eager + CUDA Graph + MambaRadixCache (68 tokens > 32-token indexer_budget, 4-token suffix hitting M=4 sparse_vec4 K=640) + Qwen4 MTP parity against TP=1
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
                .submit(long_300, params_long, None)
                .expect("submit 300-token MTP request"),
        ))
        .expect("300-token single-chunk MTP prefill + verify must succeed");
    assert_eq!(
        ids_long, ref_long.generated_token_ids,
        "EngineWorker MTP 300-token single-chunk prefill must match target greedy tokens"
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

    // Verify `plan_tiered_moe_hot_per_layer` adapts TP=2 hot cache down from 112 when
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
    assert_eq!(hot_small, 112);
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
        // Warm up MTP-3 verify graph on first run, then measure warm 64-token MTP-3 run
        let mtp_cold = engine_tp2
            .generate_with_radix_cache(&prompt_ids, 64, 64, true)
            .expect("MTP-3 cold 64-token run TP=2");
        assert_eq!(
            mtp_cold.generated_token_ids, warm_target_64.generated_token_ids,
            "Real 48-layer Qwen3.8 MTP-3 on TP=2 / EP=2 must match target greedy 64 tokens bit-for-bit"
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
        assert_eq!(
            mtp_warm.generated_token_ids, warm_target_64.generated_token_ids,
            "Warm MTP-3 64-token run must match target greedy tokens"
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
    }
}

/// Runs the exact reference benchmark prompts (`benchmark_serving.py` `short-chat` `128:256` & `128:512`,
/// the 5-task `512`-token chat benchmark suite, and `scripts/bench.py` `4096:256`) on the real
/// 48-layer `albucino/Qwen3.8-Flash-Next-W4A16-FP8PLE` checkpoint (`TP=2 / EP=2` + `hot88` + `MTP-3`).
#[test]
#[ignore = "requires local 116 GiB albucino/Qwen3.8-Flash-Next-W4A16-FP8PLE checkpoint and 2x RTX 3090 Ti"]
fn test_qwen4_exp_reference_bench_suite() {
    use w34_host::cache::KvFormatPolicy;
    use w34_host::cuda::CudaDevice;
    use w34_host::model::Qwen3_5Engine;

    if CudaDevice::count().unwrap_or(0) < 2 {
        return;
    }
    let dir = Path::new(
        "/home/nchapman/.cache/huggingface/hub/models--albucino--Qwen3.8-Flash-Next-W4A16-FP8PLE/snapshots/ef554143369a706525336f6b42a09094835dc077",
    );
    if !dir.join("model.safetensors.index.json").exists() {
        eprintln!(
            "Skipping test_qwen4_exp_reference_bench_suite: checkpoint not found at {}",
            dir.display()
        );
        return;
    }

    let mut engine = Qwen3_5Engine::load_from_dir(dir, 2, KvFormatPolicy::Fp16, Some(640), true)
        .expect("load real Qwen3.8-Flash-Next TP=2 / EP=2");
    let spec_cfg = w34_host::proposer::SpeculativeConfig {
        kind: w34_host::proposer::SpeculativeKind::Mtp,
        draft_model: None,
        mtp_k: 3,
    };
    engine
        .enable_speculative(&spec_cfg)
        .expect("enable real Qwen3.8 MTP-3 on TP=2 / EP=2");
    for r in &engine.ranks {
        let (free_b, total_b) = r.device.mem_info().unwrap_or((0, 0));
        eprintln!(
            "[REF-BENCH VRAM] rank={} free={:.2} GiB / {:.2} GiB (kv_pages={})",
            r.rank,
            free_b as f64 / (1024.0 * 1024.0 * 1024.0),
            total_b as f64 / (1024.0 * 1024.0 * 1024.0),
            r.kv_pool.num_pages,
        );
    }

    // Collect `runtime/vllm-overlay` corpus exactly like `scripts/bench.py` / `scripts/benchmark_serving.py`.
    let overlay_dir = dir.join("runtime/vllm-overlay");
    let mut overlay_files = Vec::new();
    fn visit_overlay(cur: &std::path::Path, out: &mut Vec<PathBuf>) {
        if let Ok(entries) = std::fs::read_dir(cur) {
            for e in entries.flatten() {
                let p = e.path();
                if p.is_dir() {
                    visit_overlay(&p, out);
                } else if p.is_file()
                    && p.extension()
                        .and_then(|s| s.to_str())
                        .is_some_and(|ext| matches!(ext, "py" | "cpp" | "cu"))
                {
                    out.push(p);
                }
            }
        }
    }
    visit_overlay(&overlay_dir, &mut overlay_files);
    overlay_files.sort();
    let mut corpus_text = String::new();
    for p in &overlay_files {
        let rel = p.strip_prefix(&overlay_dir).unwrap_or(p).display();
        let body = std::fs::read_to_string(p).unwrap_or_default();
        corpus_text.push_str(&format!("\n# ===== {rel} =====\n{body}"));
    }
    let corpus_ids = engine.encode(&corpus_text).expect("encode overlay corpus");
    if corpus_ids.is_empty() {
        eprintln!(
            "Skipping test_qwen4_exp_reference_bench_suite: runtime/vllm-overlay corpus is empty at {}",
            overlay_dir.display()
        );
        return;
    }

    // 1. `benchmark_serving.py` `short-chat` (`128` input tokens):
    //    Wrapper prefix/suffix from `<|im_start|>user\n{REPO_CHAT_PREFIX}{source}{REPO_CHAT_SUFFIX}<|im_end|>\n<|im_start|>assistant\n<think>\n`
    let repo_chat_prefix = "Write a detailed tutorial on multi-GPU inference: tensor parallelism, CPU offload, expert LRU caching, KV state, and speculative decoding. Use the source below where helpful. Include algorithms, numerical examples, tradeoffs, and tests. Continue with useful detail for the entire response budget.\n\n<source>\n";
    let repo_chat_suffix = "\n</source>";
    let empty_chat = format!(
        "<|im_start|>user\n{repo_chat_prefix}{repo_chat_suffix}<|im_end|>\n<|im_start|>assistant\n<think>\n"
    );
    let prefix_char_end = corpus_text.floor_char_boundary(corpus_text.len().min(8192));
    let full_chat = format!(
        "<|im_start|>user\n{repo_chat_prefix}{}{repo_chat_suffix}<|im_end|>\n<|im_start|>assistant\n<think>\n",
        &corpus_text[..prefix_char_end]
    );
    let empty_ids = engine.encode(&empty_chat).unwrap();
    let full_ids = engine.encode(&full_chat).unwrap();
    let mut pfx_len = 0;
    while pfx_len < empty_ids.len().min(full_ids.len()) && empty_ids[pfx_len] == full_ids[pfx_len] {
        pfx_len += 1;
    }
    let mut sfx_len = 0;
    let rem = empty_ids.len().min(full_ids.len()) - pfx_len;
    while sfx_len < rem
        && empty_ids[empty_ids.len() - 1 - sfx_len] == full_ids[full_ids.len() - 1 - sfx_len]
    {
        sfx_len += 1;
    }
    let interior = &full_ids[pfx_len..full_ids.len() - sfx_len];
    let src_budget = 128usize.saturating_sub(pfx_len + sfx_len);
    assert!(
        src_budget > 0 && interior.len() >= src_budget,
        "corpus excerpt too short for 128-token short-chat prompt"
    );
    let mut short_chat_128 = Vec::with_capacity(128);
    short_chat_128.extend_from_slice(&full_ids[..pfx_len]);
    short_chat_128.extend_from_slice(&interior[..src_budget]);
    short_chat_128.extend_from_slice(&full_ids[full_ids.len() - sfx_len..]);
    assert_eq!(short_chat_128.len(), 128);

    // 1a. Cold `short-chat 128:512` (first request after load, no prior warmup on this prompt)
    let m_cold_short = engine
        .generate_tokens(&short_chat_128, 512, false)
        .expect("cold short-chat 128:512");
    let acc_cold_short = engine
        .speculative
        .as_ref()
        .map(|p| p.last_accepted.clone())
        .unwrap_or_default();
    let mean_acc_cold = if acc_cold_short.is_empty() {
        0.0
    } else {
        acc_cold_short.iter().sum::<usize>() as f64 / acc_cold_short.len() as f64
    };
    eprintln!(
        "[REF-BENCH short-chat 128:512 COLD] ttft={:.2}ms prefill={:.1} tok/s decode={:.2} tok/s rounds={} mean_accepted={:.2}/3 ({:.2} tok/round)",
        m_cold_short.ttft_ms,
        m_cold_short.prefill_tok_per_s,
        m_cold_short.decode_tok_per_s,
        acc_cold_short.len(),
        mean_acc_cold,
        1.0 + mean_acc_cold,
    );

    // 1b. Warm `short-chat 128:256` & `128:512` (matching `benchmark_serving.py` 1 warmup + measured protocol)
    for &out_len in &[256usize, 512usize] {
        let m = engine
            .generate_tokens(&short_chat_128, out_len, false)
            .expect("short-chat run");
        let accepted = engine
            .speculative
            .as_ref()
            .map(|p| p.last_accepted.clone())
            .unwrap_or_default();
        let mean_acc = if accepted.is_empty() {
            0.0
        } else {
            accepted.iter().sum::<usize>() as f64 / accepted.len() as f64
        };
        eprintln!(
            "[REF-BENCH short-chat 128:{} WARM] ttft={:.2}ms prefill={:.1} tok/s decode={:.2} tok/s rounds={} mean_accepted={:.2}/3 ({:.2} tok/round)",
            out_len,
            m.ttft_ms,
            m.prefill_tok_per_s,
            m.decode_tok_per_s,
            accepted.len(),
            mean_acc,
            1.0 + mean_acc,
        );
    }

    // 2. 5-prompt diverse chat suite (`512` output tokens each, no prefix reuse between prompts)
    let chat_suite: &[(&str, &str)] = &[
        (
            "creative_writing",
            "Write a vivid science-fiction short story about an archivist on a tidally locked planet who discovers an ancient signal buried inside glacial ice.",
        ),
        (
            "code_generation",
            "Implement a complete in-memory B+ tree in Python supporting insert, point lookup, and range scan, with unit tests.",
        ),
        (
            "reasoning_math",
            "A snail climbs a 30-meter well. Each day it climbs 7 meters and each night it slips back 4 meters, except on every 4th day when rain causes it to slip 6 meters that night. Show step by step on which day the snail reaches the top.",
        ),
        (
            "factual_essay",
            "Compare the economic, institutional, and cultural consequences of the European printing press with the rise of the modern internet.",
        ),
        (
            "technical_explain",
            "Explain how Pin and Unpin work in async Rust, why self-referential futures require pinning, and how projections are safely implemented.",
        ),
    ];
    let mut suite_tok_s = Vec::new();
    for &(name, user_msg) in chat_suite {
        let formatted =
            format!("<|im_start|>user\n{user_msg}<|im_end|>\n<|im_start|>assistant\n<think>\n");
        let ids = engine.encode(&formatted).expect("encode chat prompt");
        let m = engine
            .generate_tokens(&ids, 512, false)
            .expect("chat suite 512 tokens");
        let accepted = engine
            .speculative
            .as_ref()
            .map(|p| p.last_accepted.clone())
            .unwrap_or_default();
        let mean_acc = if accepted.is_empty() {
            0.0
        } else {
            accepted.iter().sum::<usize>() as f64 / accepted.len() as f64
        };
        suite_tok_s.push(m.decode_tok_per_s);
        eprintln!(
            "[REF-BENCH chat-512 {name:<18}] prompt_len={:>3} ttft={:>6.2}ms decode={:>6.2} tok/s rounds={:>3} mean_accepted={:.2}/3 ({:.2} tok/round)",
            ids.len(),
            m.ttft_ms,
            m.decode_tok_per_s,
            accepted.len(),
            mean_acc,
            1.0 + mean_acc,
        );
    }
    let suite_mean = suite_tok_s.iter().sum::<f64>() / suite_tok_s.len() as f64;
    eprintln!(
        "[REF-BENCH chat-512 SUMMARY] mean_decode={:.2} tok/s min={:.2} tok/s max={:.2} tok/s",
        suite_mean,
        suite_tok_s.iter().cloned().fold(f64::INFINITY, f64::min),
        suite_tok_s
            .iter()
            .cloned()
            .fold(f64::NEG_INFINITY, f64::max),
    );

    // 3. `scripts/bench.py` shape `4096:256` (`offset = 5000`, `n = 4096`, `TASKS[0]`)
    let window_4k: Vec<i32> = (0..4096)
        .map(|i| corpus_ids[(5000 + i) % corpus_ids.len()])
        .collect();
    let window_4k_text = engine.decode(&window_4k).unwrap();
    let task0 = "Review this code for correctness bugs. For each bug, explain the failure scenario and propose a fix.";
    let prompt_4k_str = format!(
        "<|im_start|>user\n{window_4k_text}\n\n{task0}<|im_end|>\n<|im_start|>assistant\n<think>\n"
    );
    let prompt_4k_ids = engine.encode(&prompt_4k_str).unwrap();
    let m_4k = engine
        .generate_tokens(&prompt_4k_ids, 256, false)
        .expect("bench.py 4096:256");
    let accepted_4k = engine
        .speculative
        .as_ref()
        .map(|p| p.last_accepted.clone())
        .unwrap_or_default();
    let mean_acc_4k = if accepted_4k.is_empty() {
        0.0
    } else {
        accepted_4k.iter().sum::<usize>() as f64 / accepted_4k.len() as f64
    };
    eprintln!(
        "[REF-BENCH bench.py 4096:256] prompt_tokens={} completion_tokens={} ttft={:.2}s prefill={:.1} tok/s decode={:.2} tok/s rounds={} mean_accepted={:.2}/3 ({:.2} tok/round)",
        m_4k.prompt_len,
        m_4k.gen_len,
        m_4k.ttft_ms / 1000.0,
        m_4k.prefill_tok_per_s,
        m_4k.decode_tok_per_s,
        accepted_4k.len(),
        mean_acc_4k,
        1.0 + mean_acc_4k,
    );

    // 4. `scripts/bench.py` shape `8192:1024` (`offset = 5000 + 997 * 3 = 7991`, `n = 8192`, `TASKS[3]`)
    let window_8k: Vec<i32> = (0..8192)
        .map(|i| corpus_ids[(7991 + i) % corpus_ids.len()])
        .collect();
    let window_8k_text = engine.decode(&window_8k).unwrap();
    let task3 = "Write unit tests for the cache logic in this code, covering edge cases.";
    let prompt_8k_str = format!(
        "<|im_start|>user\n{window_8k_text}\n\n{task3}<|im_end|>\n<|im_start|>assistant\n<think>\n"
    );
    let prompt_8k_ids = engine.encode(&prompt_8k_str).unwrap();
    let m_8k = engine
        .generate_tokens(&prompt_8k_ids, 1024, false)
        .expect("bench.py 8192:1024");
    let accepted_8k = engine
        .speculative
        .as_ref()
        .map(|p| p.last_accepted.clone())
        .unwrap_or_default();
    let mean_acc_8k = if accepted_8k.is_empty() {
        0.0
    } else {
        accepted_8k.iter().sum::<usize>() as f64 / accepted_8k.len() as f64
    };
    eprintln!(
        "[REF-BENCH bench.py 8192:1024] prompt_tokens={} completion_tokens={} ttft={:.2}s prefill={:.1} tok/s decode={:.2} tok/s rounds={} mean_accepted={:.2}/3 ({:.2} tok/round)",
        m_8k.prompt_len,
        m_8k.gen_len,
        m_8k.ttft_ms / 1000.0,
        m_8k.prefill_tok_per_s,
        m_8k.decode_tok_per_s,
        accepted_8k.len(),
        mean_acc_8k,
        1.0 + mean_acc_8k,
    );
}
