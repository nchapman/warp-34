use super::*;

#[cfg(feature = "cuda")]
#[path = "gpu_tests.rs"]
mod gpu_tests;

fn sample_qwen4_exp_cfg() -> ModelConfig {
    let root: serde_json::Value = serde_json::from_str(
        r#"{"architectures":["Qwen4ExpForConditionalGeneration"],"model_type":"qwen4_exp","text_config":{
            "model_type":"qwen4_exp_text","hidden_size":2560,"num_hidden_layers":48,
            "num_attention_heads":24,"num_key_value_heads":2,"head_dim":256,"vocab_size":248320,"eos_token_id":248044,
            "linear_num_key_heads":16,"linear_num_value_heads":48,"linear_key_head_dim":128,"linear_value_head_dim":128,
            "linear_conv_kernel_dim":4,"num_experts":512,"num_experts_per_tok":10,"moe_intermediate_size":640,
            "shared_expert_intermediate_size":640,"hc_count":4,"hc_lowrank":80,"ple_embed_dim":2560,
            "ple_conv_kernel_size":4,"ple_layer_ids":[2],"ple_seed":20260202,"ngram_size":3,"heads_per_ngram":8,
            "ngram_vocab_size_base":20000000,"make_ngram_vocab_size_divisible_by":128,"split_ngram_parts":128,
            "indexer_head_dim":128,"indexer_n_heads":4,"indexer_kv_heads":1,"indexer_budget":1024,"indexer_compress_ratio":4
        }}"#,
    )
    .unwrap();
    ModelConfig::from_json(&root).unwrap()
}

#[test]
fn test_ple_hasher_matches_checkpoint_constants_and_shift_semantics() {
    let cfg = sample_qwen4_exp_cfg();
    let hasher = PleNgramHasher::from_config(&cfg, 1, 0);
    assert_eq!(
        hasher.layer_multipliers,
        vec![23_703_573_157_769, 20_109_073_645_365, 8_052_911_324_071]
    );
    assert_eq!(
        hasher.ngram_heads_vocab_sizes,
        vec![
            20_000_003, 20_000_023, 20_000_033, 20_000_047, 20_000_059, 20_000_063, 20_000_069,
            20_000_077, 20_000_081, 20_000_093, 20_000_107, 20_000_147, 20_000_153, 20_000_159,
            20_000_161, 20_000_171
        ]
    );
    assert_eq!(
        hasher.ngram_heads_offsets,
        vec![
            0,
            20_000_003,
            40_000_026,
            60_000_059,
            80_000_106,
            100_000_165,
            120_000_228,
            140_000_297,
            160_000_374,
            180_000_455,
            200_000_548,
            220_000_655,
            240_000_802,
            260_000_955,
            280_001_114,
            300_001_275
        ]
    );
    assert_eq!(hasher.padded_vocab_size, 320_001_536);
    assert_eq!(hasher.shard_rows, 2_500_012);

    // Verify chunked streaming matches single-shot prefill across EOS boundaries.
    let eos = hasher.eos_token_id;
    let seq = [101, 202, eos, 303, 404, 505, eos, eos, 606];
    let mut ctx_full = [[eos, eos]];
    let mut ids_full = vec![0i64; seq.len() * 16];
    hasher.compute_ngram_ids(&seq, &[0, seq.len() as i32], &mut ctx_full, &mut ids_full);

    let mut ctx_step = [[eos, eos]];
    let mut ids_step = vec![0i64; seq.len() * 16];
    for (i, &tok) in seq.iter().enumerate() {
        hasher.compute_ngram_ids(
            &[tok],
            &[0, 1],
            &mut ctx_step,
            &mut ids_step[i * 16..(i + 1) * 16],
        );
    }
    assert_eq!(ids_full, ids_step);
    assert_eq!(ctx_full, ctx_step);

    // Verify `compute_ngram_ids_for_slots` across radix checkpoint capture + warm restore
    // and speculative verification (`max_verify_tokens = 4`) + partial-acceptance rollback.
    let mut slot_contexts = [[eos, eos]; 8];
    let mut ids_slots = vec![0i64; seq.len() * 16];
    // Step A: prefill first 4 tokens (`seq[0..4]`) into slot 0, capturing checkpoint at token 3 (`offset = 3`) into slot 6.
    hasher.compute_ngram_ids_for_slots(
        &seq[..4],
        &[0, 4],
        &[0],
        &[0],
        &[0],
        &[],
        0,
        &[6],
        &[3],
        1,
        &mut slot_contexts,
        &mut ids_slots[..4 * 16],
    );
    assert_eq!(&ids_slots[..4 * 16], &ids_full[..4 * 16]);
    // Verify warm restore from checkpoint slot 6 (`seq[3..]`):
    let mut warm_contexts = slot_contexts;
    let mut ids_from_chk = vec![0i64; (seq.len() - 3) * 16];
    hasher.compute_ngram_ids_for_slots(
        &seq[3..],
        &[0, (seq.len() - 3) as i32],
        &[6],
        &[7],
        &[1],
        &[],
        0,
        &[],
        &[],
        0,
        &mut warm_contexts,
        &mut ids_from_chk,
    );
    assert_eq!(ids_from_chk, &ids_full[3 * 16..]);

    // Step B: speculative verify of 4 candidate tokens (`seq[4..6]` followed by 2 wrong drafts `[999, 998]`),
    // capturing per-token state in `spec_state_indices = [2, 3, 4, 5]`.
    let spec_tokens = [seq[4], seq[5], 999, 998];
    let mut spec_ids = vec![0i64; 4 * 16];
    let slot0_before_verify = slot_contexts[0];
    hasher.compute_ngram_ids_for_slots(
        &spec_tokens,
        &[0, 4],
        &[0],
        &[0],
        &[1],
        &[2, 3, 4, 5],
        4,
        &[],
        &[],
        0,
        &mut slot_contexts,
        &mut spec_ids,
    );
    // `state_out_indices` (`slot 0`) must not be clobbered during speculative verify:
    assert_eq!(slot_contexts[0], slot0_before_verify);
    assert_eq!(&spec_ids[..2 * 16], &ids_full[4 * 16..6 * 16]);
    ids_slots[4 * 16..6 * 16].copy_from_slice(&spec_ids[..2 * 16]);
    // Roll back to `accepted_len = 2` (`spec_state_indices[1] == slot 3`) and continue with `seq[6..]`:
    hasher.compute_ngram_ids_for_slots(
        &seq[6..],
        &[0, (seq.len() - 6) as i32],
        &[3],
        &[3],
        &[1],
        &[],
        0,
        &[],
        &[],
        0,
        &mut slot_contexts,
        &mut ids_slots[6 * 16..],
    );
    assert_eq!(ids_slots, ids_full);
    assert_eq!(slot_contexts[3], ctx_full[0]);
}

#[test]
fn test_qwen_flash_next_capacity_planner_tp1_and_tp2() {
    let cfg = sample_qwen4_exp_cfg();
    let vram_24g = 24 * 1024 * 1024 * 1024usize;
    let free_23_5g = 23_600 * 1024 * 1024usize;

    let plan_tp1 = QwenFlashNextCapacityPlan::plan(
        &cfg,
        1,
        vram_24g,
        free_23_5g,
        KvFormatPolicy::Tq4,
        16,
        Some(2048),
        Some(8),
        None,
        true,
    )
    .unwrap();
    assert!(plan_tp1.hot_experts_per_layer_per_rank >= 36);
    assert!(plan_tp1.headroom_per_rank_bytes >= 512 * 1024 * 1024);

    let plan_tp2 = QwenFlashNextCapacityPlan::plan(
        &cfg,
        2,
        vram_24g,
        free_23_5g,
        KvFormatPolicy::Tq4,
        16,
        Some(4096),
        Some(16),
        None,
        true,
    )
    .unwrap();
    assert!(plan_tp2.hot_experts_per_layer_per_rank >= 96);
    assert!(plan_tp2.headroom_per_rank_bytes >= 512 * 1024 * 1024);

    let plan_tp4 = QwenFlashNextCapacityPlan::plan(
        &cfg,
        4,
        vram_24g,
        free_23_5g,
        KvFormatPolicy::Tq4,
        16,
        Some(4096),
        Some(16),
        None,
        true,
    )
    .unwrap();
    assert_eq!(plan_tp4.hot_experts_per_layer_per_rank, 128);
    assert_eq!(plan_tp4.cold_experts_per_layer_per_rank, 0);
    assert_eq!(plan_tp4.host_expert_pinned_ram_bytes, 0);
    assert!(plan_tp4.headroom_per_rank_bytes >= 512 * 1024 * 1024);
    assert!(cfg.kv_heads_compatible_with_tp(4));
    assert_eq!(cfg.local_kv_heads(4), 1);
    assert_eq!(
        [0, 1, 2, 3].map(|r| cfg.kv_head_shard_index(r, 4)),
        [0, 0, 1, 1]
    );
}

#[test]
fn test_ple_mmap_table_against_real_snapshot_if_present() {
    // 1. Always-run synthetic 4-shard F8_E4M3 fixture test (including >= 64 IDs MADV_WILLNEED
    //    prefaulting and out-of-bounds / negative ID clamping).
    let mut synth_cfg = sample_qwen4_exp_cfg();
    synth_cfg.hidden_size = 32;
    synth_cfg.ple_embed_dim = 32;
    synth_cfg.heads_per_ngram = 1;
    synth_cfg.ngram_size = 2;
    synth_cfg.ngram_vocab_size_base = 64;
    synth_cfg.make_ngram_vocab_size_divisible_by = 16;
    synth_cfg.split_ngram_parts = 4;

    let tmp_dir = std::env::temp_dir().join(format!("w34_ple_mmap_test_{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&tmp_dir);
    std::fs::create_dir_all(&tmp_dir).unwrap();
    let synth_hasher = PleNgramHasher::from_config(&synth_cfg, 1, 0);
    let shard_rows = synth_hasher.shard_rows;
    let head_dim = synth_hasher.head_dim;
    let scale = 0.125f32;
    let lut = F8E4m3ScaleLut::new(scale);
    let mut header_entries = Vec::new();
    let mut data_blob = Vec::new();
    for s in 0..4usize {
        let w_name = format!(
            "model.language_model.layers.1.ple.ple_embedding.ngram_embedding.shard_{s}.weight"
        );
        let w_start = data_blob.len();
        for r in 0..shard_rows {
            for c in 0..head_dim {
                data_blob.push(((s * 31 + r * 7 + c * 3 + 1) % 120) as u8);
            }
        }
        let w_end = data_blob.len();
        header_entries.push(format!(
            "\"{w_name}\":{{\"dtype\":\"F8_E4M3\",\"shape\":[{shard_rows},{head_dim}],\"data_offsets\":[{w_start},{w_end}]}}"
        ));
        let sc_name =
            "model.language_model.layers.1.ple.ple_embedding.ngram_embedding.weight_scale";
        if s == 0 {
            let sc_start = data_blob.len();
            data_blob.extend_from_slice(&scale.to_le_bytes());
            let sc_end = data_blob.len();
            header_entries.push(format!(
                "\"{sc_name}\":{{\"dtype\":\"F32\",\"shape\":[1],\"data_offsets\":[{sc_start},{sc_end}]}}"
            ));
        }
    }
    let header_json = format!("{{{}}}", header_entries.join(","));
    let header_bytes = header_json.as_bytes();
    let mut file_bytes = Vec::with_capacity(8 + header_bytes.len() + data_blob.len());
    file_bytes.extend_from_slice(&(header_bytes.len() as u64).to_le_bytes());
    file_bytes.extend_from_slice(header_bytes);
    file_bytes.extend_from_slice(&data_blob);
    std::fs::write(tmp_dir.join("model.safetensors"), &file_bytes).unwrap();

    let synth_archive = SafetensorsArchive::open(&tmp_dir).unwrap();
    let synth_table = PleMmapTable::open(&synth_archive, &synth_cfg, 1).unwrap();
    assert_eq!(synth_table.shard_rows, shard_rows);
    assert_eq!(synth_table.head_dim, head_dim);
    // Test 66 IDs (>= 64 triggers MADV_WILLNEED prefaulting across all 4 shards) plus negative / OOB clamping:
    let test_ids: Vec<i64> = (0..66)
        .map(|i| match i {
            0 => -5,
            65 => 999_999,
            _ => ((i * 7) % (4 * shard_rows)) as i64,
        })
        .collect();
    let mut out_f8 = vec![0u8; test_ids.len() * head_dim];
    let mut out_f16 = vec![0u16; test_ids.len() * head_dim];
    let mut out_f32 = vec![0.0f32; test_ids.len() * head_dim];
    synth_table.gather_f8(&test_ids, &mut out_f8);
    synth_table.gather_dequant_f16(&test_ids, &mut out_f16);
    synth_table.gather_dequant_f32(&test_ids, &mut out_f32);
    for (idx, &raw_id) in test_ids.iter().enumerate() {
        let clamped = raw_id.max(0) as usize;
        let s = (clamped / shard_rows).min(3);
        let r = (clamped % shard_rows).min(shard_rows - 1);
        for c in 0..head_dim {
            let exp_byte = ((s * 31 + r * 7 + c * 3 + 1) % 120) as u8;
            assert_eq!(out_f8[idx * head_dim + c], exp_byte);
            assert_eq!(out_f16[idx * head_dim + c], lut.f16_lut[exp_byte as usize]);
            assert_eq!(out_f32[idx * head_dim + c], lut.f32_lut[exp_byte as usize]);
        }
    }
    let _ = std::fs::remove_dir_all(&tmp_dir);

    // 2. Real Qwen3.8-Flash-Next checkpoint verification when cached locally:
    let Ok(snap) = crate::loader::resolve_hf_model_dir("albucino/Qwen3.8-Flash-Next-W4A16-FP8PLE")
    else {
        return;
    };
    let cfg = ModelConfig::from_dir(&snap).unwrap();
    let archive = SafetensorsArchive::open(&snap).unwrap();
    let hasher = PleNgramHasher::from_archive(&archive, &cfg, 1, 0).unwrap();
    let table = PleMmapTable::open(&archive, &cfg, 1).unwrap();
    assert_eq!(table.shard_rows, 2_500_012);
    assert_eq!(table.head_dim, 160);
    assert!((table.lut.weight_scale - 0.00019931793).abs() < 1e-9);

    let eos = hasher.eos_token_id;
    let mut ctx = [[eos, eos]];
    let tokens = [151644, 872, 198, 3838];
    let mut ids = vec![0i64; tokens.len() * 16];
    hasher.compute_ngram_ids(&tokens, &[0, tokens.len() as i32], &mut ctx, &mut ids);
    let mut out_f32 = vec![0.0f32; tokens.len() * 2560];
    table.gather_dequant_f32(&ids, &mut out_f32);
    assert!(out_f32.iter().all(|v| v.is_finite()));
    assert!(out_f32.iter().any(|&v| v != 0.0));
}

#[test]
fn test_ple_mmap_table_lut4_gptq() {
    let mut synth_cfg = sample_qwen4_exp_cfg();
    synth_cfg.hidden_size = 64;
    synth_cfg.ple_embed_dim = 64;
    synth_cfg.heads_per_ngram = 1;
    synth_cfg.ngram_size = 2;
    synth_cfg.ngram_vocab_size_base = 64;
    synth_cfg.make_ngram_vocab_size_divisible_by = 16;
    synth_cfg.split_ngram_parts = 4;

    let tmp_dir = std::env::temp_dir().join(format!("w34_ple_lut4_test_{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&tmp_dir);
    std::fs::create_dir_all(&tmp_dir).unwrap();
    let synth_hasher = PleNgramHasher::from_config(&synth_cfg, 1, 0);
    let shard_rows = synth_hasher.shard_rows;
    let head_dim = synth_hasher.head_dim; // 64
    let group_size = 32usize;
    let num_groups = head_dim / group_size; // 2
    let packed_cols = head_dim / 2; // 32
    let lut_f32: [f32; PLE_LUT4_CODEBOOK_LEN] = [
        -1.0, -0.8128, -0.6546, -0.5151, -0.3918, -0.2808, -0.1791, -0.0834, 0.0105, 0.1048,
        0.2023, 0.306, 0.4208, 0.5524, 0.714, 0.9487,
    ];

    let mut header_entries = Vec::new();
    let mut data_blob = Vec::new();
    for s in 0..4usize {
        let pfx =
            format!("model.language_model.layers.1.ple.ple_embedding.ngram_embedding.shard_{s}");
        let lut_name = format!("{pfx}.weight_lut");
        let l_start = data_blob.len();
        for &v in &lut_f32 {
            data_blob.extend_from_slice(&v.to_le_bytes());
        }
        let l_end = data_blob.len();
        header_entries.push(format!(
            "\"{lut_name}\":{{\"dtype\":\"F32\",\"shape\":[{PLE_LUT4_CODEBOOK_LEN}],\"data_offsets\":[{l_start},{l_end}]}}"
        ));

        let p_name = format!("{pfx}.weight_packed");
        let p_start = data_blob.len();
        for r in 0..shard_rows {
            for j in 0..packed_cols {
                let lo = ((s * 7 + r * 3 + j * 5 + 1) & 0x0F) as u8;
                let hi = ((s * 11 + r * 5 + j * 9 + 2) & 0x0F) as u8;
                data_blob.push(lo | (hi << 4));
            }
        }
        let p_end = data_blob.len();
        header_entries.push(format!(
            "\"{p_name}\":{{\"dtype\":\"U8\",\"shape\":[{shard_rows},{packed_cols}],\"data_offsets\":[{p_start},{p_end}]}}"
        ));

        let sc_name = format!("{pfx}.weight_scale");
        let sc_start = data_blob.len();
        for r in 0..shard_rows {
            for g in 0..num_groups {
                let sc_bits = f32_to_f16(((s + r + g + 1) as f32) * 0.0625);
                data_blob.extend_from_slice(&sc_bits.to_le_bytes());
            }
        }
        let sc_end = data_blob.len();
        header_entries.push(format!(
            "\"{sc_name}\":{{\"dtype\":\"F16\",\"shape\":[{shard_rows},{num_groups}],\"data_offsets\":[{sc_start},{sc_end}]}}"
        ));
    }
    let header_json = format!("{{{}}}", header_entries.join(","));
    let header_bytes = header_json.as_bytes();
    let mut file_bytes = Vec::with_capacity(8 + header_bytes.len() + data_blob.len());
    file_bytes.extend_from_slice(&(header_bytes.len() as u64).to_le_bytes());
    file_bytes.extend_from_slice(header_bytes);
    file_bytes.extend_from_slice(&data_blob);
    std::fs::write(tmp_dir.join("model.safetensors"), &file_bytes).unwrap();

    let synth_archive = SafetensorsArchive::open(&tmp_dir).unwrap();
    let synth_table = PleMmapTable::open(&synth_archive, &synth_cfg, 1).unwrap();
    assert_eq!(synth_table.shard_rows, shard_rows);
    assert_eq!(synth_table.head_dim, head_dim);
    assert_eq!(
        synth_table.format,
        PleTableFormat::Lut4Gptq {
            group_size,
            lut_f32
        }
    );
    assert_eq!(
        synth_table.staged_bytes_per_head(),
        packed_cols + num_groups * 2
    );

    let test_ids: Vec<i64> = (0..66)
        .map(|i| match i {
            0 => -3,
            65 => 999_999,
            _ => ((i * 7) % (4 * shard_rows)) as i64,
        })
        .collect();
    let mut out_f16 = vec![0u16; test_ids.len() * head_dim];
    let mut out_f32 = vec![0.0f32; test_ids.len() * head_dim];
    synth_table.gather_dequant_f16(&test_ids, &mut out_f16);
    synth_table.gather_dequant_f32(&test_ids, &mut out_f32);
    for (idx, &raw_id) in test_ids.iter().enumerate() {
        let clamped = raw_id.max(0) as usize;
        let s = (clamped / shard_rows).min(3);
        let r = (clamped % shard_rows).min(shard_rows - 1);
        for c in 0..head_dim {
            let j = c >> 1;
            let nib = if (c & 1) == 0 {
                (s * 7 + r * 3 + j * 5 + 1) & 0x0F
            } else {
                (s * 11 + r * 5 + j * 9 + 2) & 0x0F
            };
            let g = c / group_size;
            let sc = crate::cuda::f16_to_f32(f32_to_f16(((s + r + g + 1) as f32) * 0.0625));
            let expected_f32 = lut_f32[nib] * sc;
            assert_eq!(out_f32[idx * head_dim + c], expected_f32);
            assert_eq!(out_f16[idx * head_dim + c], f32_to_f16(expected_f32));
        }
    }
    let _ = std::fs::remove_dir_all(&tmp_dir);

    if let Ok(snap) = crate::loader::resolve_hf_model_dir("nchapman/Qwen3.8-Flash-Next-W4A16-HGPTQ")
    {
        let cfg = ModelConfig::from_dir(&snap).unwrap();
        let archive = SafetensorsArchive::open(&snap).unwrap();
        let hasher = PleNgramHasher::from_archive(&archive, &cfg, 1, 0).unwrap();
        let table = PleMmapTable::open(&archive, &cfg, 1).unwrap();
        assert_eq!(table.shard_rows, 2_500_012);
        assert_eq!(table.head_dim, 160);
        assert!(matches!(
            table.format,
            PleTableFormat::Lut4Gptq { group_size: 32, .. }
        ));

        let eos = hasher.eos_token_id;
        let mut ctx = [[eos, eos]];
        let tokens = [151644, 872, 198, 3838];
        let mut ids = vec![0i64; tokens.len() * 16];
        hasher.compute_ngram_ids(&tokens, &[0, tokens.len() as i32], &mut ctx, &mut ids);
        let mut out_f32 = vec![0.0f32; tokens.len() * 2560];
        table.gather_dequant_f32(&ids, &mut out_f32);
        assert!(out_f32.iter().all(|v| v.is_finite()));
        assert!(out_f32.iter().any(|&v| v != 0.0));
    } else {
        eprintln!(
            "Skipping real-snapshot check in test_ple_mmap_table_lut4_gptq: nchapman/Qwen3.8-Flash-Next-W4A16-HGPTQ not cached"
        );
    }
}
