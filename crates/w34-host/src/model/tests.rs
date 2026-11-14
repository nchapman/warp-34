//! Model serving capacity and tokenizer regression tests.

#[cfg(feature = "cuda")]
mod dflash_serving_preflight_tests {
    use super::*;

    #[test]
    fn dflash_missing_draft_fails_before_target_gpu_loading() {
        let target = tempfile::tempdir().unwrap();
        let config = crate::proposer::SpeculativeConfig {
            kind: crate::proposer::SpeculativeKind::DFlash2,
            draft_model: None,
            mtp_k: 1,
        };
        let error = Qwen3_5Engine::load_for_serving_with_speculative(
            target.path().to_str().unwrap(),
            1,
            KvFormatPolicy::Tq4,
            Some(16384),
            false,
            1,
            Some(&config),
        )
        .err()
        .expect("missing draft must fail before any target allocation");
        assert!(error.to_string().contains("requires --draft-model"));
    }

    #[test]
    fn dflash_reserve_changes_placement_and_live_pool_admission() {
        use super::super::rank::{clamp_state_slots, plan_host_embed_with_budget};
        let mut cfg = ModelConfig::from_drafter_json(&serde_json::json!({
            "hidden_size": 5120, "intermediate_size": 17408,
            "num_attention_heads": 40, "num_key_value_heads": 4,
            "head_dim": 128, "num_hidden_layers": 5, "vocab_size": 152064
        }))
        .unwrap();
        cfg.tie_word_embeddings = false;
        let gib = 1024 * 1024 * 1024;
        let weights = 12 * gib;
        let reserve = gib;
        let steady = 4 * gib;
        let free = weights + gib + steady;
        let placement = |extra| {
            plan_host_embed_with_budget(&cfg, weights, gib, gib / 8, free, 0, Some(steady + extra))
        };
        assert!(!placement(0).1);
        assert!(placement(reserve).1);
        let plan = |extra| {
            crate::cuda::CapacityPlanner::plan_with_headroom(
                free,
                free,
                weights,
                gib,
                gib,
                gib / 256,
                Some(256),
                2 * gib + extra,
            )
        };
        assert_eq!(plan(0).unwrap().max_kv_pages_per_layer, 256);
        assert!(plan(reserve).is_err());
        assert_eq!(
            clamp_state_slots(24, 24 * 1024 + reserve, reserve, 1024),
            24
        );
        assert!(clamp_state_slots(24, 24 * 1024 + reserve - 1, reserve, 1024) < 24);
    }
}

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
    fn legacy_pool_budget_charges_allocated_capacity_for_state() {
        // DeviceArena reserves capacity even when uploads leave most of it unused.
        let weights = super::super::rank::pool_weight_budget(false, 200, 600);
        let slots = clamp_state_slots(64, 1200, weights + 100 + 100 + 100, 50);
        assert_eq!(slots, 6);
        assert!(600 + 100 + 100 + 100 + slots * 50 <= 1200);
    }

    #[test]
    fn legacy_pool_budget_charges_allocated_capacity_for_kv() {
        let weights = super::super::rank::pool_weight_budget(false, 200, 600);
        let plan = crate::cuda::CapacityPlanner::plan_with_headroom(
            1200, 1200, weights, 100, 300, 50, None, 100,
        )
        .unwrap();
        assert_eq!(plan.max_kv_pages_per_layer, 2);
        assert!(
            crate::cuda::CapacityPlanner::plan_with_headroom(
                1200,
                1200,
                weights,
                100,
                300,
                50,
                Some(3),
                100,
            )
            .is_err()
        );
        // Live free VRAM already includes allocated capacity; keep its cancellation basis.
        assert_eq!(super::super::rank::pool_weight_budget(true, 200, 600), 200);
        assert_eq!(super::super::rank::pool_weight_budget(false, 600, 600), 600);
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

        // TP=1 (<=1.5 GiB base+KV) and TP=2 with ample VRAM preserve default_hot (104 and 128),
        // while TP=1 with 4096-page KV reserve uses 96 so MTP-3 keeps k_hot = 192 in VRAM.
        let (hot_tp1_default, _) = plan_tiered_moe_hot_per_layer(
            &cfg,
            1,
            19_600 * 1024 * 1024,
            gib + 38 * 1024 * 1024,
            450 * 1024 * 1024,
        );
        assert_eq!(hot_tp1_default, 104);
        let (hot_tp1_4k_kv, _) = plan_tiered_moe_hot_per_layer(
            &cfg,
            1,
            19_600 * 1024 * 1024,
            2 * gib,
            450 * 1024 * 1024,
        );
        assert_eq!(hot_tp1_4k_kv, 96);

        let (hot_tp2_default, _) = plan_tiered_moe_hot_per_layer(
            &cfg,
            2,
            20_800 * 1024 * 1024,
            gib + 38 * 1024 * 1024,
            450 * 1024 * 1024,
        );
        assert_eq!(hot_tp2_default, 128);

        // When DFlash2 reserve exceeds default MTP/graph reserve (e.g. 24 GiB sm_89 TP=2
        // Flash-Next serving at 4096 KV pages), `plan_tiered_moe_hot_per_layer_with_dflash`
        // steps `hot_per_layer` down so target GDN state slots are preserved.
        let (hot_without_dflash, _) = plan_tiered_moe_hot_per_layer_with_dflash(
            &cfg,
            2,
            13_200 * 1024 * 1024,
            1_800 * 1024 * 1024,
            600 * 1024 * 1024,
            0,
        );
        let (hot_with_dflash, _) = plan_tiered_moe_hot_per_layer_with_dflash(
            &cfg,
            2,
            13_200 * 1024 * 1024,
            1_800 * 1024 * 1024,
            600 * 1024 * 1024,
            2_400 * 1024 * 1024,
        );
        assert!(hot_with_dflash < hot_without_dflash);

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

        // Compact 8–12 GiB GPU regime (TP=1 ~3.24 GiB post-weight, TP=2 ~4.59 GiB post-weight):
        // selects `TIERED_MOE_TP1_MIN_HOT_EXPERTS` (16) hot experts/layer while reserving
        // `TIERED_MOE_COMPACT_STAGE_RESERVE_SLOTS * exp_bytes` double-buffered staging.
        let mib = 1024 * 1024;
        let exp_bytes = cfg
            .moe_expert_pack_layout()
            .expect("qwen4_exp expert layout")
            .stride;
        let (hot_3060ti_tp1, reserve_3060ti_tp1) =
            plan_tiered_moe_hot_per_layer(&cfg, 1, 3316 * mib, 584 * mib, 237 * mib);
        assert_eq!(hot_3060ti_tp1, TIERED_MOE_TP1_MIN_HOT_EXPERTS);
        assert_eq!(
            reserve_3060ti_tp1,
            (48 * TIERED_MOE_TP1_MIN_HOT_EXPERTS + TIERED_MOE_COMPACT_STAGE_RESERVE_SLOTS)
                * exp_bytes
        );
        assert!(584 * mib + 237 * mib + reserve_3060ti_tp1 <= 3316 * mib);
        let (hot_3060ti_tp2, reserve_3060ti_tp2) =
            plan_tiered_moe_hot_per_layer(&cfg, 2, 4702 * mib, 548 * mib, 252 * mib);
        assert_eq!(hot_3060ti_tp2, TIERED_MOE_TP1_MIN_HOT_EXPERTS);
        assert_eq!(
            reserve_3060ti_tp2,
            (48 * TIERED_MOE_TP1_MIN_HOT_EXPERTS + TIERED_MOE_COMPACT_STAGE_RESERVE_SLOTS)
                * exp_bytes
        );
        assert!(548 * mib + 252 * mib + reserve_3060ti_tp2 <= 4702 * mib);
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
                1,
                cfg.num_experts,
                exp_bytes,
                MTP_TP1_SERVING_MIN_FREE_VRAM_BYTES
            ),
            MTP_TP1_SERVING_HOT_EXPERTS
        );
        // Oversized expert_stride_bytes where 320 * stride + headroom exceeds 1,700 MiB must
        // fall through to a smaller tier rather than overcommitting VRAM.
        assert_eq!(
            plan_mtp_tiered_moe_hot_experts(
                1,
                cfg.num_experts,
                6 * 1024 * 1024,
                MTP_TP1_SERVING_MIN_FREE_VRAM_BYTES
            ),
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

        // 5. LoRA adapter with midloop pre-sizes scratch_arena to 960 MiB up front;
        //    non-midloop LoRA pre-sizes scratch_arena to 1,056 MiB.
        let midloop_lora = LoraCapacityReserve {
            device_bytes_per_rank: 1280 * 1024 * 1024,
            has_midloop: true,
            requires_atomic_prefix: true,
        };
        assert_eq!(
            plan_scratch_arena_bytes_with_lora(
                &dense_64l,
                2,
                LARGE_MODEL_WEIGHT_THRESHOLD_BYTES,
                None,
                midloop_lora,
            ),
            MIDLOOP_LORA_SCRATCH_ARENA_BYTES
        );
        let standard_lora = LoraCapacityReserve {
            device_bytes_per_rank: 512 * 1024 * 1024,
            has_midloop: false,
            requires_atomic_prefix: false,
        };
        assert_eq!(
            plan_scratch_arena_bytes_with_lora(
                &dense_64l,
                2,
                LARGE_MODEL_WEIGHT_THRESHOLD_BYTES,
                None,
                standard_lora,
            ),
            crate::scheduler::TIERED_MOE_SCRATCH_ARENA_BYTES
        );
    }

    #[test]
    fn plan_target_kv_pages_auto_sizes_serving_and_preserves_lora_and_headroom() {
        let mib = 1024 * 1024;
        let gib = 1024 * mib;
        let page_bytes = 2 * mib; // e.g. ~2 MiB/page

        // 1. Explicit `--kv-pages` is always honored verbatim.
        assert_eq!(
            plan_target_kv_pages(
                Some(1024),
                true,
                0,
                false,
                24 * gib,
                12 * gib,
                2 * gib,
                page_bytes
            ),
            1024
        );

        // 2. Non-serving CLI/test mode defaults to 256 pages (4,096 tokens).
        assert_eq!(
            plan_target_kv_pages(
                None,
                false,
                0,
                false,
                24 * gib,
                12 * gib,
                2 * gib,
                page_bytes
            ),
            DEFAULT_TARGET_KV_PAGES
        );

        // 3. Serving mode with ample VRAM (24 GiB total, 12 GiB fixed, 2.3 GiB headroom+LoRA)
        //    auto-sizes to DEFAULT_SERVING_TARGET_KV_PAGES (2,048 pages = 32,768 tokens).
        assert_eq!(
            plan_target_kv_pages(
                None,
                true,
                0,
                false,
                24 * gib,
                12 * gib,
                2300 * mib,
                page_bytes
            ),
            DEFAULT_SERVING_TARGET_KV_PAGES
        );

        // 4. Serving mode under tighter VRAM (e.g. TP=1 27B + LoRA where 1.8 GiB remains above
        //    fixed + required_headroom + 1.5 GiB extra margin) scales to 64-aligned 896 pages.
        let fixed = 17 * gib;
        let req_headroom = 3 * gib;
        let free = fixed + req_headroom + AUTO_SERVING_KV_EXTRA_HEADROOM_BYTES + 1800 * mib;
        assert_eq!(
            plan_target_kv_pages(None, true, 0, false, free, fixed, req_headroom, page_bytes),
            896
        );

        // 5. Very tight VRAM falls back to DEFAULT_TARGET_KV_PAGES (256).
        let tight_free = fixed + req_headroom + 512 * mib;
        assert_eq!(
            plan_target_kv_pages(
                None,
                true,
                0,
                false,
                tight_free,
                fixed,
                req_headroom,
                page_bytes
            ),
            DEFAULT_TARGET_KV_PAGES
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

    #[test]
    fn dense_mtp_placement_reserves_initialization_peak() {
        use super::super::rank::plan_host_embed_with_budget;
        let mut cfg = sample_qwen4_exp_config();
        cfg.num_experts = 0;
        cfg.indexer_n_heads = 0;
        cfg.tie_word_embeddings = false;
        let mib = 1024 * 1024;
        for tp in [1, 2, 4] {
            let low = crate::mtp::dense_serving_reserve(&cfg, tp, 229_376);
            let high = crate::mtp::dense_serving_reserve(&cfg, tp, 262_144);
            // Only FP16 KV and the two page tables grow once batch rows cap.
            assert_eq!(
                high - low,
                32_768 * cfg.local_kv_heads(tp) * cfg.head_dim * 4 + (32_768 / 16) * 8
            );
            for tokens in [229_376, 262_144] {
                let reserve = crate::mtp::dense_serving_reserve(&cfg, tp, tokens);
                assert!(reserve > tokens * cfg.local_kv_heads(tp) * cfg.head_dim * 4);
                let weights = 12 * 1024 * mib;
                let scratch = 64 * mib;
                let steady = 3 * 1024 * mib;
                let free = weights + scratch + steady;
                let plan = |free, extra| {
                    plan_host_embed_with_budget(
                        &cfg,
                        weights,
                        scratch,
                        64 * mib,
                        free,
                        0,
                        Some(steady + extra),
                    )
                };
                assert_eq!(plan(free, 0), (weights, false));
                assert!(
                    plan(free, reserve).1,
                    "MTP must influence embedding placement"
                );
                assert_eq!(plan(free + reserve, reserve), (weights, false));
                assert!(plan(free + reserve - 1, reserve).1);
            }
        }
    }

    #[test]
    fn mtp_dense_reserve_saturates_extreme_capacity() {
        let mut cfg = sample_qwen4_exp_config();
        cfg.num_experts = 0;
        cfg.indexer_n_heads = 0;
        assert_eq!(
            crate::mtp::dense_serving_reserve(&cfg, 1, usize::MAX),
            usize::MAX
        );
    }

    #[test]
    fn mtp_dense_reserve_leaves_moe_contract_unchanged() {
        let cfg = sample_qwen4_exp_config();
        assert_eq!(crate::mtp::dense_serving_reserve(&cfg, 1, 262_144), 0);
    }

    #[test]
    fn resident_moe_mtp_serving_reserve_matches_expert_precision_and_mode() {
        use super::super::rank::resident_moe_mtp_serving_reserve;
        let mib = 1024 * 1024;
        let mut cfg = sample_qwen4_exp_config();
        // Resident MoE (35B-A3B geometry: 40 layers, hidden=2048, moe_inter=512, 256 experts < 20 GiB).
        cfg.model_type = "qwen3_5_moe_text".to_string();
        cfg.architectures = vec!["Qwen3_5MoeForConditionalGeneration".to_string()];
        cfg.hidden_size = 2048;
        cfg.moe_intermediate_size = 512;
        cfg.num_hidden_layers = 40;
        cfg.num_experts = 256;
        cfg.hc_count = 1;
        cfg.indexer_n_heads = 0;
        assert_eq!(
            resident_moe_mtp_serving_reserve(&cfg, true, true, true),
            768 * mib
        );
        assert_eq!(
            resident_moe_mtp_serving_reserve(&cfg, true, true, false),
            1600 * mib
        );
        // Non-serving or non-MTP leaves single-request / non-MTP budgets untouched.
        assert_eq!(resident_moe_mtp_serving_reserve(&cfg, false, true, true), 0);
        assert_eq!(resident_moe_mtp_serving_reserve(&cfg, true, false, true), 0);
        // Tiered MoE and dense models have their own dedicated reserves.
        let tiered = sample_qwen4_exp_config();
        assert_eq!(
            resident_moe_mtp_serving_reserve(&tiered, true, true, true),
            0
        );
        cfg.num_experts = 0;
        assert_eq!(resident_moe_mtp_serving_reserve(&cfg, true, true, true), 0);
    }

    #[test]
    fn target_embedding_budget_boundaries() {
        use super::super::rank::{
            plan_host_embed_with_budget, target_base_headroom, target_pool_geometry,
        };
        let mib = 1024 * 1024;
        let mut cfg = sample_qwen4_exp_config();
        cfg.num_experts = 0;
        cfg.tie_word_embeddings = false;
        cfg.vocab_size = 248_320;
        cfg.hidden_size = 5120;
        cfg.num_hidden_layers = 64;
        cfg.indexer_n_heads = 0;
        cfg.ple_layer_ids.clear();
        cfg.num_key_value_heads = 4;
        cfg.head_dim = 256;
        cfg.layer_types = (0..64)
            .map(|i| {
                if i % 4 == 3 {
                    LayerKind::FullAttention
                } else {
                    LayerKind::LinearAttention
                }
            })
            .collect();
        let (page, state) = target_pool_geometry(&cfg, 1, KvFormatPolicy::Tq4);
        assert_eq!(page * 16_384, 5_603_590_144);
        assert_eq!(state, 153_944_064);
        let weights = 17_258 * mib;
        let host = 14_834 * mib; // full replicated embedding, arena rounded to 2 MiB
        let scratch = 336 * mib;
        let staging = 128 * mib;
        let steady = page * 16_384
            + 3 * state
            + target_base_headroom(&cfg, 1, 16_384, 24 * 1024 * mib)
            + 16 * mib;
        let plan = |free, reserve| {
            plan_host_embed_with_budget(
                &cfg,
                weights,
                scratch,
                staging,
                free,
                16 * mib,
                Some(reserve),
            )
        };
        assert_eq!(plan(weights + scratch + steady, steady), (weights, false));
        assert_eq!(plan(weights + scratch + steady - 1, steady), (host, true));
        assert_eq!(plan(host + scratch + steady, steady), (host, true));
        assert_eq!(plan(host + scratch + steady - 1, steady), (weights, false));
        // Three execution slots fit even when eight requested serving slots do not.
        assert_eq!(
            plan(host + scratch + steady, steady + 5 * state),
            (weights, false)
        );
        // Staging and steady pools fit separately at equality, but not together.
        let boot = host + scratch + staging + 16 * mib + MIN_BOOT_MARGIN_BYTES;
        assert!(boot < host + scratch + steady);
        assert!(host + scratch + steady + staging > host + scratch + steady);
        assert_eq!(plan(usize::MAX - 1, usize::MAX), (weights, false));
        cfg.tie_word_embeddings = true;
        assert_eq!(
            plan_host_embed_with_budget(&cfg, weights, scratch, staging, 1, 0, Some(usize::MAX)),
            (weights, false)
        );
        let tiered = sample_qwen4_exp_config();
        assert_eq!(
            plan_host_embed_with_budget(&tiered, weights, scratch, staging, 1, 0, Some(usize::MAX)),
            (weights, true)
        );
    }

    #[test]
    fn serving_budget_offloads_embed_when_concurrent_mtp_state_slots_exceed_gpu_embed_slack() {
        use super::super::rank::plan_host_embed_with_serving_budget;
        let mut cfg = sample_qwen4_exp_config();
        cfg.num_experts = 0;
        cfg.tie_word_embeddings = false;
        cfg.vocab_size = 248_320;
        cfg.hidden_size = 5120;
        cfg.num_hidden_layers = 64;
        cfg.indexer_n_heads = 0;
        cfg.ple_layer_ids.clear();
        let mib = 1024 * 1024;
        let weights = 17_700 * mib;
        let scratch = 640 * mib;
        let staging = 300 * mib;
        let floor_steady = 3_800 * mib;
        let desired_steady = 8_350 * mib;
        let free_24g = 24_100 * mib;
        let (_, offload_24g) = plan_host_embed_with_serving_budget(
            &cfg,
            weights,
            scratch,
            staging,
            free_24g,
            16 * mib,
            Some(floor_steady),
            Some(desired_steady),
        );
        assert!(
            offload_24g,
            "24 GB card must offload 2.37 GiB embed_tokens so 24 concurrent MTP state slots fit"
        );
        let (_, offload_32g) = plan_host_embed_with_serving_budget(
            &cfg,
            weights,
            scratch,
            staging,
            32_000 * mib,
            16 * mib,
            Some(floor_steady),
            Some(desired_steady),
        );
        assert!(
            !offload_32g,
            "32 GB card fits both GPU embed_tokens and all 24 state slots"
        );
        assert_eq!(crate::mtp::serving_state_slots(1, 3).unwrap(), 8);
        assert_eq!(crate::mtp::serving_state_slots(2, 3).unwrap(), 16);
        assert_eq!(crate::mtp::serving_state_slots(4, 3).unwrap(), 24);
        assert!(crate::mtp::serving_state_slots(0, 3).is_err());
    }

    #[test]
    fn target_pool_geometry_preserves_tp_qsa_and_ple() {
        use super::super::rank::target_pool_geometry;
        let mut cfg = sample_qwen4_exp_config();
        let (page, state) = target_pool_geometry(&cfg, 1, KvFormatPolicy::Tq4);
        let (tp_page, tp_state) = target_pool_geometry(&cfg, 2, KvFormatPolicy::Tq4);
        let full = cfg
            .layer_types
            .iter()
            .filter(|kind| matches!(kind, LayerKind::FullAttention))
            .count();
        let qsa = full
            * cfg.indexer_kv_heads
            * (16 + 16 / cfg.indexer_compress_ratio)
            * cfg.indexer_head_dim
            * 2;
        let ple = cfg.ple_layer_ids.len() * cfg.hc_count * cfg.hidden_size * 18;
        assert_eq!(tp_page, (page - qsa) / 2 + qsa);
        assert_eq!(tp_state, (state - ple) / 2 + ple);
        // KV heads replicate for TP=4 with two KV heads; QSA remains replicated.
        assert_eq!(
            target_pool_geometry(&cfg, 4, KvFormatPolicy::Tq4).0,
            tp_page
        );
        cfg.indexer_n_heads = 0;
        cfg.ple_layer_ids.clear();
        assert_eq!(
            target_pool_geometry(&cfg, 1, KvFormatPolicy::Tq4),
            (page - qsa, state - ple)
        );
    }

    #[test]
    fn plan_host_embed_offload_respects_slack_and_fit_bounds() {
        let mib = 1024 * 1024;
        let gib = 1024 * mib;

        // 1. Tiered-MoE untied (`Qwen3.8-Flash-Next`) always uses mapped host `embed_tokens`.
        let tiered_cfg = sample_qwen4_exp_config();
        assert_eq!(
            plan_host_embed_offload(&tiered_cfg, 3_200 * mib, 336 * mib, 128 * mib, 24 * gib),
            (3_200 * mib, true)
        );

        // 2. Tied embeddings (`2B` / `4B`) never offload `embed_tokens`.
        let mut tied_cfg = sample_qwen4_exp_config();
        tied_cfg.num_experts = 0;
        tied_cfg.tie_word_embeddings = true;
        assert_eq!(
            plan_host_embed_offload(&tied_cfg, 3_600 * mib, 512 * mib, 128 * mib, 7_500 * mib),
            (3_600 * mib, false)
        );

        // 3. Untied dense `9B` (`vocab = 248,320`, `hidden = 4,096` -> `embed = 1,940 MiB`,
        //    `est_weight_bytes = 7,104 MiB`, `scratch = 512 MiB`, `staging = 128 MiB`):
        //    - Keeps GPU `embed_tokens` on 16 GiB (`15,400 MiB` free) and 24 GiB (`23,500 MiB` free).
        //    - Offloads `embed_tokens` to `MappedHostBuffer` on 8 GiB (`7,670 MiB` free), reducing
        //      `est_weight_bytes` to `5,164 MiB`.
        let mut dense_9b = sample_qwen4_exp_config();
        dense_9b.num_experts = 0;
        dense_9b.tie_word_embeddings = false;
        dense_9b.vocab_size = 248_320;
        dense_9b.hidden_size = 4_096;
        let est_9b = 7_104 * mib;
        let scratch_9b = 512 * mib;
        let staging_9b = 128 * mib;

        assert_eq!(
            plan_host_embed_offload(&dense_9b, est_9b, scratch_9b, staging_9b, 23_500 * mib),
            (est_9b, false)
        );
        assert_eq!(
            plan_host_embed_offload(&dense_9b, est_9b, scratch_9b, staging_9b, 15_400 * mib),
            (est_9b, false)
        );
        let (adjusted_8gb, offload_8gb) =
            plan_host_embed_offload(&dense_9b, est_9b, scratch_9b, staging_9b, 7_670 * mib);
        assert!(offload_8gb, "9B TP=1 on 8 GiB must offload embed_tokens");
        assert_eq!(adjusted_8gb, 5_164 * mib);

        // 4. Untied dense `27B TP=1` on 16 GiB (`est_weight_bytes = 16,900 MiB`, `hidden = 5,120`
        //    -> `embed = 2,425 MiB`, `without_embed = 14,476 MiB`): even without `embed_tokens`,
        //    `boot_without_gpu_embed + MIN_HEADROOM_BYTES > 15,400 MiB`, so offload is declined
        //    and pre-flight fails fast with the full model requirement.
        let mut dense_27b = dense_9b;
        dense_27b.hidden_size = 5_120;
        assert_eq!(
            plan_host_embed_offload(&dense_27b, 16_900 * mib, 384 * mib, 256 * mib, 15_400 * mib),
            (16_900 * mib, false)
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
        // Tiered-MoE with untied embeddings hosts `embed_tokens` in `MappedHostBuffer`,
        // leaving only `lm_head.weight` (64 MiB) + 1-element FP8 PLE scale (256 B) + 96 MiB margin = 162 MiB.
        synth_cfg.tie_word_embeddings = false;
        let synth_tiered_untied =
            Qwen3_5ModelRank::estimate_weight_arena_bytes(&synth_cfg, &synth_archive, 1, Some(128));
        assert_eq!(synth_tiered_untied, 162 * 1024 * 1024);

        // With `embed_tokens` in `weight_arena` (`num_experts = 0`), untied includes both
        // `embed_tokens` (64 MiB) and `lm_head.weight` (64 MiB), while tied excludes `lm_head.weight`.
        synth_cfg.num_experts = 0;
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
        // rounded up to 2 MiB alignment = 162 MiB (multi-element 32 MiB lut4-gptq `shard_00.weight_scale` is excluded).
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

        // Both untied tiered-MoE (`embed_tokens` in `MappedHostBuffer`, `lm_head.weight` in `weight_arena`)
        // and tied (`embed_tokens` in `weight_arena`, redundant `lm_head.weight` excluded) avoid duplicating
        // two 1.21 GiB vocab tables in GPU `weight_arena` (~4,246 MiB vs ~5,482 MiB if both were in VRAM).
        assert!(
            tp1_bytes < 4400 * 1024 * 1024,
            "expected untied tiered-MoE host_embed_tokens to keep TP=1 weight_arena < 4400 MiB (got {} MiB)",
            tp1_bytes / (1024 * 1024)
        );
        let mut tied_cfg = cfg.clone();
        tied_cfg.tie_word_embeddings = true;
        let tp1_tied =
            Qwen3_5ModelRank::estimate_weight_arena_bytes(&tied_cfg, &archive, 1, Some(128));
        assert_eq!(
            tp1_bytes, tp1_tied,
            "untied (host_embed_tokens + GPU lm_head.weight) and tied (GPU embed_tokens) must use identical weight_arena bytes"
        );
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn pre_mlp_norm_targets_and_tiered_prerotated_scratch_reservation_preserve_0_ulp() {
        use crate::cuda::{CublasHandle, CudaStream, DeviceBuffer, f32_to_f16};
        use crate::model::weights::{
            FfnBlockWeights, LinearProj, MoeExpertStorage, MoeLayerWeights,
        };
        use w34_kernels::ffi;

        let Ok(stream) = CudaStream::new(0) else {
            return;
        };
        let cublas = CublasHandle::new(&stream).unwrap();
        let rows = 5usize;
        let h_dim = 256usize;
        let sh_interm = 320usize; // not divisible by 128 -> padded inner.size_k() = 384 (TP=2 geometry)
        let sh_interm_padded = 384usize;
        let num_experts = 16usize;
        let top_k = 4usize;

        let d_router_w = DeviceBuffer::alloc(0, num_experts * h_dim * 2).unwrap();
        let d_sh_gate_w = DeviceBuffer::alloc(0, h_dim * 2).unwrap();
        let d_sh_gu_q = DeviceBuffer::alloc(0, (2 * sh_interm) * (h_dim / 8) * 4).unwrap();
        let d_sh_gu_s = DeviceBuffer::alloc(0, (2 * sh_interm) * (h_dim / 64) * 2).unwrap();
        let d_sh_dn_q = DeviceBuffer::alloc(0, h_dim * (sh_interm_padded / 8) * 4).unwrap();
        let d_sh_dn_s = DeviceBuffer::alloc(0, h_dim * (sh_interm_padded / 64) * 2).unwrap();
        d_router_w
            .copy_from_host(&vec![0x3000u16; num_experts * h_dim], &stream)
            .unwrap();
        d_sh_gate_w
            .copy_from_host(&vec![0x3400u16; h_dim], &stream)
            .unwrap();
        d_sh_gu_q
            .copy_from_host(
                &vec![0x9876_5432u32; (2 * sh_interm) * (h_dim / 8)],
                &stream,
            )
            .unwrap();
        d_sh_gu_s
            .copy_from_host(&vec![0x3400u16; (2 * sh_interm) * (h_dim / 64)], &stream)
            .unwrap();
        d_sh_dn_q
            .copy_from_host(
                &vec![0x8888_8888u32; h_dim * (sh_interm_padded / 8)],
                &stream,
            )
            .unwrap();
        d_sh_dn_s
            .copy_from_host(&vec![0x3400u16; h_dim * (sh_interm_padded / 64)], &stream)
            .unwrap();

        let make_moe = |experts: MoeExpertStorage| {
            FfnBlockWeights::Moe(Box::new(MoeLayerWeights {
                router_gate: LinearProj::DenseF16 {
                    weight_ptr: d_router_w.addr(),
                    size_n: num_experts,
                    size_k: h_dim,
                },
                num_experts,
                top_k,
                norm_topk_prob: true,
                scoring_func: 0,
                local_moe_interm: 128,
                routed_input_hadamard: 64,
                routed_down_hadamard: 64,
                experts,
                local_shared_interm: sh_interm,
                shared_gate_up_proj: Some(LinearProj::Hadamard {
                    inner: Box::new(LinearProj::MarlinW4A16 {
                        b_q_weight_ptr: d_sh_gu_q.addr(),
                        b_scales_ptr: d_sh_gu_s.addr(),
                        size_n: 2 * sh_interm,
                        size_k: h_dim,
                        group_blocks: 4,
                    }),
                    size_n: 2 * sh_interm,
                    size_k: h_dim,
                    block_size: 64,
                }),
                shared_down_proj: Some(LinearProj::Hadamard {
                    inner: Box::new(LinearProj::MarlinW4A16 {
                        b_q_weight_ptr: d_sh_dn_q.addr(),
                        b_scales_ptr: d_sh_dn_s.addr(),
                        size_n: h_dim,
                        size_k: sh_interm_padded,
                        group_blocks: 4,
                    }),
                    size_n: h_dim,
                    size_k: sh_interm,
                    block_size: 64,
                }),
                shared_expert_gate: Some(LinearProj::DenseF16 {
                    weight_ptr: d_sh_gate_w.addr(),
                    size_n: 1,
                    size_k: h_dim,
                }),
            }))
        };

        let ffn_tiered = make_moe(MoeExpertStorage::Tiered { layer_id: 0 });
        let d_normed = DeviceBuffer::alloc(0, rows * h_dim * 2).unwrap();
        let d_mlp_out = DeviceBuffer::alloc(0, rows * h_dim * 2).unwrap();
        let d_scratch = DeviceBuffer::alloc(0, 64 * 1024).unwrap();
        let ex_small = stream.exec(d_scratch.addr(), rows * h_dim * 2);
        // Insufficient scratch for both prerotated input and padded shared_down_proj workspace -> fallback [normed, 0]
        assert_eq!(
            ffn_tiered.pre_mlp_norm_targets(
                d_normed.addr(),
                d_mlp_out.addr(),
                &ex_small,
                rows,
                h_dim,
            ),
            [d_normed.addr(), 0]
        );
        let ex_ok = stream.exec(d_scratch.addr(), d_scratch.len_bytes);
        assert_eq!(
            ffn_tiered.pre_mlp_norm_targets(d_normed.addr(), d_mlp_out.addr(), &ex_ok, rows, h_dim),
            [d_normed.addr(), d_scratch.addr()]
        );

        let h_in: Vec<u16> = (0..rows * h_dim)
            .map(|i| f32_to_f16(((i % 29) as f32 - 14.0) * 0.05))
            .collect();
        d_normed.copy_from_host(&h_in, &stream).unwrap();
        // SAFETY: `d_normed` and `d_scratch` are live disjoint device buffers of size `>= rows * h_dim * 2` bytes.
        let rc = unsafe {
            ffi::w34_block_hadamard_f16(
                &ex_ok,
                d_normed.addr() as *const std::ffi::c_void,
                d_scratch.addr() as *mut std::ffi::c_void,
                rows as i32,
                h_dim as i32,
                h_dim as i32,
                64,
            )
        };
        assert_eq!(rc, 0);
        let mut h_prerot_before = vec![0u16; rows * h_dim];
        d_scratch
            .copy_to_host(&mut h_prerot_before, &stream)
            .unwrap();

        let d_marlin_ws = DeviceBuffer::alloc(0, 4096).unwrap();
        let d_router = DeviceBuffer::alloc(0, rows * num_experts * 2).unwrap();
        let d_topw = DeviceBuffer::alloc(0, rows * top_k * 4).unwrap();
        let d_topids = DeviceBuffer::alloc(0, rows * top_k * 4).unwrap();
        let d_shgu = DeviceBuffer::alloc(0, rows * 2 * sh_interm * 2).unwrap();
        let d_shact = DeviceBuffer::alloc(0, rows * sh_interm * 2).unwrap();
        let d_shout = DeviceBuffer::alloc(0, rows * h_dim * 2).unwrap();
        let d_shgate = DeviceBuffer::alloc(0, rows * 2).unwrap();

        ffn_tiered
            .forward_with_prerotated_input(
                &ex_ok,
                &cublas,
                d_marlin_ws.addr(),
                d_marlin_ws.len_bytes / 4,
                d_normed.addr(),
                d_mlp_out.addr(),
                rows,
                h_dim,
                128,
                0,
                0,
                d_router.addr(),
                d_topw.addr(),
                d_topids.addr(),
                0,
                d_shgu.addr(),
                d_shact.addr(),
                d_shout.addr(),
                d_shgate.addr(),
                d_scratch.addr(),
            )
            .unwrap();

        // Verify the prerotated input stored at `d_scratch.addr()` was not overwritten by `shared_down_proj.forward`.
        let mut h_prerot_after = vec![0u16; rows * h_dim];
        d_scratch
            .copy_to_host(&mut h_prerot_after, &stream)
            .unwrap();
        assert_eq!(h_prerot_after, h_prerot_before);
    }
}
