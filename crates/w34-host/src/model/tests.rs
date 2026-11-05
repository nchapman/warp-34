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
}
