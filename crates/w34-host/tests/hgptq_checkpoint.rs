//! Explicit opt-in qualification of the pinned HGPTQ checkpoint.
#![cfg(feature = "cuda")]

use std::path::PathBuf;
use w34_host::{
    cache::KvFormatPolicy,
    model::Qwen3_5Engine,
    proposer::{SpeculativeConfig, SpeculativeKind},
    vision,
};

fn qualify_model(model_dir: &std::path::Path, tp: usize) -> Vec<i32> {
    let mut engine = Qwen3_5Engine::load(
        model_dir.to_str().unwrap(),
        tp,
        KvFormatPolicy::Fp16,
        Some(64),
        false,
    )
    .unwrap();
    let seed = engine.encode("The capital of France is").unwrap();
    let eager = engine
        .generate_tokens(&seed, 32, false)
        .unwrap()
        .generated_token_ids;
    engine.use_cuda_graph = true;
    let graph = engine
        .generate_tokens(&seed, 32, false)
        .unwrap()
        .generated_token_ids;
    assert_eq!(eager, graph, "eager vs graph TP={tp}");
    let prompt: Vec<i32> = seed.iter().copied().cycle().take(130).collect();
    let plain = engine
        .generate_with_radix_cache(&prompt, 24, 64, false)
        .unwrap()
        .generated_token_ids;
    engine
        .enable_speculative(&SpeculativeConfig {
            kind: SpeculativeKind::Mtp,
            mtp_k: 3,
            draft_model: None,
        })
        .unwrap();
    for k in [1, 2, 3] {
        engine.speculative.as_mut().unwrap().set_mtp_k(k).unwrap();
        let cold = engine.generate_tokens(&prompt, 24, false).unwrap();
        assert_eq!(
            cold.generated_token_ids, plain,
            "lossless MTP k={k} TP={tp}"
        );
    }
    let cold = engine
        .generate_with_radix_cache(&prompt, 24, 64, false)
        .unwrap();
    let warm = engine
        .generate_with_radix_cache(&prompt, 24, 64, false)
        .unwrap();
    assert_eq!(cold.generated_token_ids, plain);
    assert_eq!(warm.generated_token_ids, plain);
    assert!(warm.cached_prefix_tokens >= 64);
    eager
}

#[test]
#[ignore = "requires pinned HGPTQ checkpoints and two free CUDA GPUs"]
fn pinned_hgptq_text_graph_mtp_and_tp2() {
    let cache = std::env::var_os("HF_HUB_CACHE")
        .map(PathBuf::from)
        .unwrap_or_else(|| {
            PathBuf::from(std::env::var_os("HOME").expect("HOME")).join(".cache/huggingface/hub")
        });
    for rel_dir in [
        "models--nchapman--Qwen3.5-2B-W4A16-HGPTQ/snapshots/0fbb9a38f631ea63678f4b51a491e7ecb26c681c",
        "models--nchapman--Qwen3.5-9B-W4A16-HGPTQ/snapshots/7b2e48b7e7e8ee38925f5337c66feed22d3b031b",
        "models--nchapman--Qwen3.8-27B-W4A16-HGPTQ/snapshots/ae51987e3339a2a70070702548571e10fe8bcbc2",
    ] {
        let dir = cache.join(rel_dir);
        assert!(
            dir.join("model_mtp.safetensors").exists(),
            "missing pinned HGPTQ checkpoint at {}",
            dir.display()
        );
        let tp1 = qualify_model(&dir, 1);
        let tp2 = qualify_model(&dir, 2);
        assert_eq!(
            tp1,
            tp2,
            "TP=1 vs TP=2 greedy continuation for {}",
            dir.display()
        );
    }
}

#[test]
#[ignore = "requires pinned HGPTQ checkpoints and two free CUDA GPUs"]
fn pinned_hgptq_vision_matches_vllm_and_tp2() {
    let cache = std::env::var_os("HF_HUB_CACHE")
        .map(PathBuf::from)
        .unwrap_or_else(|| {
            PathBuf::from(std::env::var_os("HOME").expect("HOME")).join(".cache/huggingface/hub")
        });
    let fixture = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../tools/oracle/fixtures/coco_sample.png");
    let (rgb, h, w) = vision::load_cli_image_with_policy(fixture.to_str().unwrap(), false).unwrap();
    let image = vision::preprocess_rgb_image(&rgb, h, w, 16, 2, 2, 1024, 1048576);
    let expanded = "<|vision_start|><|image_pad|><|vision_end|>Describe this image.".replace(
        "<|image_pad|>",
        &"<|image_pad|>".repeat(image.num_merged_tokens(2)),
    );
    let images = [image];

    let cases: [(&str, &str, &[i32]); 4] = [
        (
            "2B",
            "models--nchapman--Qwen3.5-2B-W4A16-HGPTQ/snapshots/0fbb9a38f631ea63678f4b51a491e7ecb26c681c",
            &[
                561, 2099, 4774, 1330, 18833, 19823, 383, 264, 17676, 25313, 13, 561, 18833, 513,
                2107, 5474, 1634, 18833, 11, 321, 781, 513, 2107, 20479, 13, 2532, 513, 1330, 8434,
                11241, 383, 279,
            ],
        ),
        (
            "9B",
            "models--nchapman--Qwen3.5-9B-W4A16-HGPTQ/snapshots/7b2e48b7e7e8ee38925f5337c66feed22d3b031b",
            &[8771, 18833, 513, 20479, 383, 264, 17676, 36781, 13, 248044],
        ),
        (
            "27B",
            "models--nchapman--Qwen3.8-27B-W4A16-HGPTQ/snapshots/ae51987e3339a2a70070702548571e10fe8bcbc2",
            &[
                8771, 18833, 513, 20479, 383, 264, 17676, 36781, 440, 1330, 8434, 11241, 13, 3648,
                7993, 369, 264, 74035, 11, 321, 279, 975, 369, 449, 6474, 13, 561, 74035, 369, 383,
                279, 2047,
            ],
        ),
        (
            "35B-A3B",
            "models--nchapman--Qwen3.6-35B-A3B-W4A16-HGPTQ/snapshots/4f5de21c886d79e17af5b8a4bfc625e226703fb6",
            &[
                8771, 18833, 513, 20479, 383, 264, 17676, 25313, 440, 1330, 8434, 11241, 13, 248044,
            ],
        ),
    ];

    for (label, rel_dir, expected_vllm_ids) in cases {
        let model_dir = cache.join(rel_dir);
        assert!(
            model_dir.join("config.json").exists(),
            "missing pinned HGPTQ checkpoint at {}",
            model_dir.display()
        );
        let run_tp = |tp: usize| -> Vec<i32> {
            let mut engine = Qwen3_5Engine::load(
                model_dir.to_str().unwrap(),
                tp,
                KvFormatPolicy::Fp16,
                Some(32),
                true,
            )
            .unwrap();
            let prompt = engine.encode(&expanded).unwrap();
            assert_eq!(prompt.len(), 306);
            engine
                .generate_multimodal_with_radix_cache(
                    &prompt,
                    &images,
                    expected_vllm_ids.len(),
                    64,
                    false,
                )
                .unwrap()
                .generated_token_ids
        };
        let tp1 = run_tp(1);
        let tp2 = run_tp(2);
        assert_eq!(
            tp1, expected_vllm_ids,
            "{label} TP=1 vision greedy vs vLLM 0.31.0"
        );
        assert_eq!(
            tp1, tp2,
            "{label} TP=1 vs TP=2 multimodal greedy continuation"
        );
    }
}

#[test]
#[ignore = "requires pinned HGPTQ 27B + DFlash2-W4A16-HGPTQ checkpoints and two free CUDA GPUs"]
fn pinned_hgptq_dflash2_lossless_and_tp2() {
    let cache = std::env::var_os("HF_HUB_CACHE")
        .map(PathBuf::from)
        .unwrap_or_else(|| {
            PathBuf::from(std::env::var_os("HOME").expect("HOME")).join(".cache/huggingface/hub")
        });
    let target_dir = cache
        .join("models--nchapman--Qwen3.8-27B-W4A16-HGPTQ/snapshots/ae51987e3339a2a70070702548571e10fe8bcbc2");
    let draft_dir = cache
        .join("models--nchapman--Qwen3.8-27B-DFlash2-W4A16-HGPTQ/snapshots/46b87ea125d24924affa8da9c3e981a937aab216");
    assert!(target_dir.join("config.json").exists());
    assert!(draft_dir.join("config.json").exists());

    for tp in [1usize, 2usize] {
        let mut engine = Qwen3_5Engine::load(
            target_dir.to_str().unwrap(),
            tp,
            KvFormatPolicy::Tq4,
            Some(64),
            true,
        )
        .unwrap();
        let seed = engine.encode("The capital of France is").unwrap();
        let prompt: Vec<i32> = seed.iter().copied().cycle().take(128).collect();
        let plain = engine
            .generate_tokens(&prompt, 32, false)
            .unwrap()
            .generated_token_ids;
        engine
            .enable_speculative(&SpeculativeConfig {
                kind: SpeculativeKind::DFlash2,
                mtp_k: 1,
                draft_model: Some(draft_dir.to_str().unwrap().to_string()),
            })
            .unwrap();
        let spec = engine.generate_tokens(&prompt, 32, false).unwrap();
        assert_eq!(
            spec.generated_token_ids, plain,
            "DFlash2-W4A16-HGPTQ lossless greedy parity TP={tp}"
        );
        let accepted: usize = engine
            .speculative
            .as_ref()
            .unwrap()
            .last_accepted
            .iter()
            .copied()
            .sum();
        assert!(
            accepted >= 16,
            "expected strong acceptance on TP={tp}, got {accepted}"
        );
    }
}

#[test]
#[ignore = "requires pinned HGPTQ 35B-A3B MoE checkpoint and two free CUDA GPUs"]
fn pinned_hgptq_moe_35b_text_vision_mtp_and_tp2() {
    let cache = std::env::var_os("HF_HUB_CACHE")
        .map(PathBuf::from)
        .unwrap_or_else(|| {
            PathBuf::from(std::env::var_os("HOME").expect("HOME")).join(".cache/huggingface/hub")
        });
    let dir = cache
        .join("models--nchapman--Qwen3.6-35B-A3B-W4A16-HGPTQ/snapshots/4f5de21c886d79e17af5b8a4bfc625e226703fb6");
    assert!(
        dir.join("model_mtp.safetensors").exists(),
        "missing pinned 35B-A3B HGPTQ checkpoint at {}",
        dir.display()
    );
    let tp1 = qualify_model(&dir, 1);
    let tp2 = qualify_model(&dir, 2);
    assert_eq!(tp1, tp2, "35B-A3B TP=1 vs TP=2 greedy continuation");

    let fixture = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../tools/oracle/fixtures/coco_sample.png");
    let (rgb, h, w) = vision::load_cli_image_with_policy(fixture.to_str().unwrap(), false).unwrap();
    let image = vision::preprocess_rgb_image(&rgb, h, w, 16, 2, 2, 1024, 1048576);
    let expanded = "<|vision_start|><|image_pad|><|vision_end|>Describe this image.".replace(
        "<|image_pad|>",
        &"<|image_pad|>".repeat(image.num_merged_tokens(2)),
    );
    let images = [image];
    let expected_vllm_ids = [
        8771, 18833, 513, 20479, 383, 264, 17676, 25313, 440, 1330, 8434, 11241, 13, 248044,
    ];
    for tp in [1usize, 2usize] {
        let mut engine = Qwen3_5Engine::load(
            dir.to_str().unwrap(),
            tp,
            KvFormatPolicy::Fp16,
            Some(32),
            true,
        )
        .unwrap();
        let prompt = engine.encode(&expanded).unwrap();
        assert_eq!(prompt.len(), 306);
        let got = engine
            .generate_multimodal_with_radix_cache(
                &prompt,
                &images,
                expected_vllm_ids.len(),
                64,
                false,
            )
            .unwrap()
            .generated_token_ids;
        assert_eq!(
            got, expected_vllm_ids,
            "35B-A3B TP={tp} vision greedy vs vLLM 0.31.0"
        );
    }
}
