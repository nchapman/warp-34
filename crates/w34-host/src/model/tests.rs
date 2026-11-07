//! Model serving capacity and tokenizer regression tests.

use super::*;

#[cfg(test)]
mod serving_capacity_tests {
    use super::*;

    fn sample_qwen4_exp_config() -> ModelConfig {
        let qfn_json: serde_json::Value = serde_json::from_str(
            r#"{
            "architectures": ["Qwen4ExpForConditionalGeneration"],
            "model_type": "qwen4_exp",
            "quantization_config": {
                "bits": 4,
                "group_size": 128,
                "packing_format": "auto_round:auto_gptq",
                "quant_method": "auto-round",
                "sym": true
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
        ModelConfig::from_json(&qfn_json).unwrap()
    }

    #[test]
    fn normalize_hf_tokenizer_clears_embedded_truncation() {
        let mut tok = Tokenizer::new(tokenizers::models::wordlevel::WordLevel::default());
        tok.with_pre_tokenizer(Some(tokenizers::pre_tokenizers::whitespace::Whitespace));
        tok.add_tokens(&[
            tokenizers::AddedToken::from("hello", false),
            tokenizers::AddedToken::from("world", false),
        ]);
        tok.with_truncation(Some(tokenizers::TruncationParams {
            max_length: 2,
            ..Default::default()
        }))
        .unwrap();
        let prompt = "hello world hello world hello";
        assert_eq!(tok.encode(prompt, false).unwrap().get_ids().len(), 2);

        let tok = normalize_hf_tokenizer(tok);
        assert_eq!(tok.encode(prompt, false).unwrap().get_ids().len(), 5);
    }

    #[test]
    fn serving_budget_is_checked() {
        assert_eq!(serving_state_slots(1).unwrap(), 8);
        assert_eq!(serving_state_slots(2).unwrap(), 8);
        assert_eq!(serving_state_slots(16).unwrap(), 64);
        assert!(serving_state_slots(0).is_err());
        assert!(serving_state_slots(usize::MAX / 4 + 1).is_err());
    }

    #[test]
    fn serving_budget_respects_memory_clamp() {
        assert_eq!(clamp_state_slots(64, 1000, 400, 10), 60);
        assert_eq!(clamp_state_slots(64, 2000, 400, 10), 64);
        assert_eq!(clamp_state_slots(24, 2000, 400, 10), 24);
        assert_eq!(clamp_state_slots(10, 2000, 400, 10), 10);
        assert_eq!(clamp_state_slots(64, 400, 400, 10), 3);
        assert_eq!(clamp_state_slots(64, 410, 400, 10), 3);
        assert_eq!(clamp_state_slots(64, 0, 400, 0), 64);
    }

    #[test]
    fn plan_tiered_moe_hot_per_layer_adapts_and_clamps_bounds() {
        let cfg = sample_qwen4_exp_config();
        let gib = 1024 * 1024 * 1024;

        // Non-tiered model returns (0, 0)
        let mut dense_cfg = cfg.clone();
        dense_cfg.num_experts = 0;
        assert_eq!(
            plan_tiered_moe_hot_per_layer(&dense_cfg, 1, 20 * gib, gib, gib),
            (0, 0)
        );

        // TP=1 (<=1.5 GiB base+KV) and TP=2 with ample VRAM preserve default_hot (96 and 112),
        // while TP=1 with 4096-page KV reserve uses 88 so MTP-3 keeps k_hot = 192 in VRAM.
        let (hot_tp1_default, _) = plan_tiered_moe_hot_per_layer(
            &cfg,
            1,
            19_600 * 1024 * 1024,
            gib + 38 * 1024 * 1024,
            450 * 1024 * 1024,
        );
        assert_eq!(hot_tp1_default, 96);
        let (hot_tp1_4k_kv, _) = plan_tiered_moe_hot_per_layer(
            &cfg,
            1,
            19_600 * 1024 * 1024,
            2 * gib,
            450 * 1024 * 1024,
        );
        assert_eq!(hot_tp1_4k_kv, 88);

        let (hot_tp2_default, _) = plan_tiered_moe_hot_per_layer(
            &cfg,
            2,
            19_600 * 1024 * 1024,
            gib + 38 * 1024 * 1024,
            450 * 1024 * 1024,
        );
        assert_eq!(hot_tp2_default, 112);

        // TP=2 at 12288 pages + MTP reserve (~4.3 GiB base+KV+MTP) and 1.2 GiB state pool:
        // exact 8-aligned hot_per_layer = 104, fitting within 19_600 MiB.
        let base_plus_kv_12k = gib + 1536 * 1024 * 1024 + 1824 * 1024 * 1024;
        let min_state_12k = 1200 * 1024 * 1024;
        let (hot_tp2_12k, reserve_12k) = plan_tiered_moe_hot_per_layer(
            &cfg,
            2,
            19_600 * 1024 * 1024,
            base_plus_kv_12k,
            min_state_12k,
        );
        assert_eq!(hot_tp2_12k, 104);
        assert!(base_plus_kv_12k + min_state_12k + reserve_12k <= 19_600 * 1024 * 1024);

        // Insufficient VRAM falls back to min_hot (16 on TP=1, 32 on TP=2)
        let (hot_tp1_min, _) = plan_tiered_moe_hot_per_layer(&cfg, 1, gib, 2 * gib, gib);
        assert_eq!(hot_tp1_min, TIERED_MOE_TP1_MIN_HOT_EXPERTS);
        let (hot_tp2_min, _) = plan_tiered_moe_hot_per_layer(&cfg, 2, gib, 2 * gib, gib);
        assert_eq!(hot_tp2_min, TIERED_MOE_TP2_MIN_HOT_EXPERTS);
    }

    #[test]
    fn plan_mtp_tiered_moe_hot_experts_matches_target_post_load_reserve() {
        let cfg = sample_qwen4_exp_config();
        let exp_bytes = cfg
            .moe_expert_pack_layout()
            .expect("qwen4_exp expert layout")
            .stride;
        let post_target_extra_reserve =
            TIERED_MOE_STAGE_RESERVE_SLOTS * exp_bytes + TIERED_MOE_MTP_AND_GRAPH_RESERVE_BYTES;

        // Joint contract: the extra reserve set aside by `plan_tiered_moe_hot_per_layer`
        // (even after RankTieredMoe consumes ~275 MiB for staging + slot tables) leaves enough
        // free VRAM for `plan_mtp_tiered_moe_hot_experts` to pick 192 on TP=1 and 256 on TP=2.
        let post_target_free_vram = post_target_extra_reserve - 280 * 1024 * 1024;
        assert_eq!(
            plan_mtp_tiered_moe_hot_experts(1, cfg.num_experts, exp_bytes, post_target_free_vram),
            MTP_TP1_STREAMED_PREFILL_MAX_HOT
        );
        assert_eq!(
            plan_mtp_tiered_moe_hot_experts(
                2,
                cfg.num_experts / 2,
                exp_bytes,
                post_target_free_vram
            ),
            cfg.num_experts / 2
        );

        // Stepped degradation under tighter VRAM budgets
        assert_eq!(
            plan_mtp_tiered_moe_hot_experts(1, 512, exp_bytes, 600 * 1024 * 1024),
            MTP_MEDIUM_HOT_EXPERTS
        );
        assert_eq!(
            plan_mtp_tiered_moe_hot_experts(1, 512, exp_bytes, 300 * 1024 * 1024),
            MTP_COMPACT_HOT_EXPERTS
        );
        assert_eq!(
            plan_mtp_tiered_moe_hot_experts(1, 32, exp_bytes, 300 * 1024 * 1024),
            MTP_FALLBACK_HOT_EXPERTS
        );
    }

    #[test]
    fn plan_scratch_arena_bytes_aligns_with_scheduler_token_caps() {
        let tiered_cfg = sample_qwen4_exp_config();
        let mut dense_64l = tiered_cfg.clone();
        dense_64l.num_experts = 0;
        dense_64l.num_hidden_layers = EXTREME_LONGCTX_MIN_LAYERS;

        // 1. Extreme long-context TP=1 -> 336 MiB -> scheduler clamps to 1,024 tokens
        let extreme_bytes = plan_scratch_arena_bytes(
            &dense_64l,
            1,
            5 * 1024 * 1024 * 1024,
            Some(EXTREME_LONGCTX_MIN_PAGES),
        );
        assert_eq!(
            extreme_bytes,
            crate::scheduler::EXTREME_LONGCTX_SCRATCH_ARENA_BYTES
        );
        let mut sched = SchedulerConfig::default();
        sched.clamp_for_scratch_bytes(extreme_bytes);
        assert_eq!(
            sched.max_num_batched_tokens,
            crate::scheduler::MEDIUM_SCRATCH_MAX_BATCHED_TOKENS
        );

        // 2. Large model (>= 15 GiB weights) on TP=1 -> 384 MiB -> scheduler clamps to 1,024 tokens
        let large_tp1_bytes =
            plan_scratch_arena_bytes(&dense_64l, 1, LARGE_MODEL_WEIGHT_THRESHOLD_BYTES, Some(256));
        assert_eq!(
            large_tp1_bytes,
            crate::scheduler::LARGE_MODEL_TP1_SCRATCH_ARENA_BYTES
        );
        let mut sched = SchedulerConfig::default();
        sched.clamp_for_scratch_bytes(large_tp1_bytes);
        assert_eq!(
            sched.max_num_batched_tokens,
            crate::scheduler::MEDIUM_SCRATCH_MAX_BATCHED_TOKENS
        );

        // 3. Tiered MoE (None or <= 8,192 KV pages) -> 1,056 MiB -> fits 2,048-token layer scratch layout
        assert_eq!(
            plan_scratch_arena_bytes(&tiered_cfg, 1, 3 * 1024 * 1024 * 1024, None),
            crate::scheduler::TIERED_MOE_SCRATCH_ARENA_BYTES
        );
        let tiered_bytes = plan_scratch_arena_bytes(
            &tiered_cfg,
            1,
            3 * 1024 * 1024 * 1024,
            Some(crate::scheduler::TIERED_MOE_FULL_SCRATCH_MAX_KV_PAGES),
        );
        assert_eq!(
            tiered_bytes,
            crate::scheduler::TIERED_MOE_SCRATCH_ARENA_BYTES
        );
        let (tiered_2k_scratch_bytes, ..) = crate::model::layer_scratch_layout(
            &tiered_cfg,
            1,
            crate::scheduler::DEFAULT_MAX_NUM_BATCHED_TOKENS,
        )
        .expect("tiered 2048-token scratch layout");
        assert!(tiered_2k_scratch_bytes <= tiered_bytes);
        let mut sched = SchedulerConfig::default();
        sched.clamp_for_scratch_bytes(tiered_bytes);
        assert_eq!(
            sched.max_num_batched_tokens,
            crate::scheduler::DEFAULT_MAX_NUM_BATCHED_TOKENS
        );

        // 4. Standard model or TP=2 large model -> 512 MiB -> scheduler keeps full 2,048 tokens
        let default_bytes =
            plan_scratch_arena_bytes(&dense_64l, 2, LARGE_MODEL_WEIGHT_THRESHOLD_BYTES, Some(256));
        assert_eq!(default_bytes, crate::scheduler::DEFAULT_SCRATCH_ARENA_BYTES);
        let mut sched = SchedulerConfig::default();
        sched.clamp_for_scratch_bytes(default_bytes);
        assert_eq!(
            sched.max_num_batched_tokens,
            crate::scheduler::DEFAULT_MAX_NUM_BATCHED_TOKENS
        );
    }

    #[test]
    fn plan_mtp_draft_lm_head_bytes_respects_quantization_geometry_and_headroom() {
        use crate::mtp::{
            MTP_DRAFT_LM_HEAD_ARENA_SLACK_BYTES, MTP_DRAFT_LM_HEAD_GROUP_SIZE,
            MTP_DRAFT_LM_HEAD_POST_HEADROOM_BYTES, plan_mtp_draft_lm_head_bytes,
        };

        // 1. Already-quantized target `lm_head` -> never duplicated (`None`).
        let already_w4 = LinearProj::MarlinW4A16 {
            b_q_weight_ptr: 0x1000,
            b_scales_ptr: 0x2000,
            size_n: 124_160,
            size_k: 2_560,
            group_blocks: 8,
        };
        assert_eq!(
            plan_mtp_draft_lm_head_bytes(&already_w4, 4 * 1024 * 1024 * 1024),
            None
        );

        // 2. Unaligned `DenseF16` dimensions -> `None`.
        let unaligned_n = LinearProj::DenseF16 {
            weight_ptr: 0x1000,
            size_n: 124_160 + 32,
            size_k: 2_560,
        };
        assert_eq!(
            plan_mtp_draft_lm_head_bytes(&unaligned_n, 4 * 1024 * 1024 * 1024),
            None
        );

        // 3. `Qwen3.8-Flash-Next` TP=2 (`124,160 x 2,560` per rank):
        //    `qweight = 158,924,800 B` (151.56 MiB), `scales = 9,932,800 B` (9.47 MiB),
        //    total `168,857,600 B` (161.04 MiB vs 606.25 MiB DenseF16 = 3.76x smaller).
        let tp2_dense = LinearProj::DenseF16 {
            weight_ptr: 0x1000,
            size_n: 124_160,
            size_k: 2_560,
        };
        let expected_qw = (124_160 / 2) * 2_560;
        let expected_sc = 124_160 * (2_560 / MTP_DRAFT_LM_HEAD_GROUP_SIZE) * 2;
        let expected_arena = expected_qw + expected_sc + MTP_DRAFT_LM_HEAD_ARENA_SLACK_BYTES;
        assert_eq!(
            plan_mtp_draft_lm_head_bytes(
                &tp2_dense,
                expected_arena + MTP_DRAFT_LM_HEAD_POST_HEADROOM_BYTES - 1
            ),
            None,
            "must decline when post-MTP free VRAM cannot preserve headroom"
        );
        assert_eq!(
            plan_mtp_draft_lm_head_bytes(
                &tp2_dense,
                expected_arena + MTP_DRAFT_LM_HEAD_POST_HEADROOM_BYTES
            ),
            Some((expected_qw, expected_sc, expected_arena))
        );
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn estimate_weight_arena_bytes_excludes_tied_lm_head_and_lut4_ple_scales() {
        // 1. Always-run synthetic safetensors archive covering tied `lm_head.weight`
        //    and multi-element `ple_embedding.weight_scale` vs scalar `ple_embedding.weight_scale`.
        let tmp_dir =
            std::env::temp_dir().join(format!("w34_est_weight_arena_test_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&tmp_dir);
        std::fs::create_dir_all(&tmp_dir).unwrap();
        let lm_bytes = 64 * 1024 * 1024usize;
        let lut4_scale_bytes = 32 * 1024 * 1024usize;
        let header = serde_json::json!({
            "lm_head.weight": {
                "dtype": "BF16",
                "shape": [16384, 2048],
                "data_offsets": [0, lm_bytes]
            },
            "model.layers.5.ple.ple_embedding.shard_00.weight_scale": {
                "dtype": "F16",
                "shape": [3355443, 5],
                "data_offsets": [lm_bytes, lm_bytes + lut4_scale_bytes]
            },
            "model.layers.11.ple.ple_embedding.weight_scale": {
                "dtype": "F16",
                "shape": [1],
                "data_offsets": [lm_bytes + lut4_scale_bytes, lm_bytes + lut4_scale_bytes + 2]
            }
        })
        .to_string();
        let st_path = tmp_dir.join("model.safetensors");
        {
            use std::io::Write;
            let mut f = std::fs::File::create(&st_path).unwrap();
            f.write_all(&(header.len() as u64).to_le_bytes()).unwrap();
            f.write_all(header.as_bytes()).unwrap();
            f.set_len((8 + header.len() + lm_bytes + lut4_scale_bytes + 2) as u64)
                .unwrap();
        }
        let synth_archive = crate::loader::SafetensorsArchive::open(&tmp_dir).unwrap();
        let mut synth_cfg = sample_qwen4_exp_config();
        synth_cfg.vocab_size = 16384;
        synth_cfg.hidden_size = 2048;
        synth_cfg.tie_word_embeddings = false;
        let synth_untied =
            Qwen3_5ModelRank::estimate_weight_arena_bytes(&synth_cfg, &synth_archive, 1, Some(128));
        synth_cfg.tie_word_embeddings = true;
        let synth_tied =
            Qwen3_5ModelRank::estimate_weight_arena_bytes(&synth_cfg, &synth_archive, 1, Some(128));
        let _ = std::fs::remove_dir_all(&tmp_dir);
        assert_eq!(
            synth_untied - synth_tied,
            lm_bytes,
            "tied config must exclude redundant lm_head.weight (untied={synth_untied}, tied={synth_tied})"
        );
        // `synth_tied` only has `embed_tokens` (64 MiB) + 1-element FP8 PLE scale (256 B) + 96 MiB margin,
        // rounded up to 16 MiB alignment = 162 MiB (multi-element 32 MiB lut4-gptq `shard_00.weight_scale` is excluded).
        assert_eq!(synth_tied, 162 * 1024 * 1024);

        // 2. Real checkpoint check when `nchapman/Qwen3.8-Flash-Next-W4A16-HGPTQ` is cached locally.
        let Ok(dir) =
            crate::loader::resolve_hf_model_dir("nchapman/Qwen3.8-Flash-Next-W4A16-HGPTQ")
        else {
            eprintln!(
                "Skipping real-snapshot check in estimate_weight_arena_bytes_excludes_tied_lm_head_and_lut4_ple_scales: checkpoint not cached"
            );
            return;
        };
        let cfg = ModelConfig::from_dir(&dir).expect("parse config.json");
        let archive = crate::loader::SafetensorsArchive::open(&dir).expect("open archive");
        let tp1_bytes = Qwen3_5ModelRank::estimate_weight_arena_bytes(&cfg, &archive, 1, Some(128));
        let tp2_bytes = Qwen3_5ModelRank::estimate_weight_arena_bytes(&cfg, &archive, 2, Some(128));
        // Host-mmapped lut4-gptq ple_embedding.shard_*.weight_scale (~3.05 GiB across 128 shards) is excluded,
        // keeping TP=1 < 5.6 GiB (vs ~8.6 GiB without exclusion) and TP=2 < 4.3 GiB/rank (vs ~7.2 GiB).
        assert!(
            tp1_bytes < 5600 * 1024 * 1024,
            "unexpectedly large TP=1 weight_arena estimate: {} MiB",
            tp1_bytes / (1024 * 1024)
        );
        assert!(
            tp2_bytes < 4300 * 1024 * 1024,
            "unexpectedly large TP=2 weight_arena estimate: {} MiB",
            tp2_bytes / (1024 * 1024)
        );

        let mut tied_cfg = cfg.clone();
        tied_cfg.tie_word_embeddings = true;
        let tp1_tied =
            Qwen3_5ModelRank::estimate_weight_arena_bytes(&tied_cfg, &archive, 1, Some(128));
        assert!(
            tp1_bytes.saturating_sub(tp1_tied) >= 1200 * 1024 * 1024,
            "expected tied_cfg to save >= 1.2 GiB for redundant lm_head.weight on TP=1 (untied={} MiB, tied={} MiB)",
            tp1_bytes / (1024 * 1024),
            tp1_tied / (1024 * 1024),
        );
    }
}
