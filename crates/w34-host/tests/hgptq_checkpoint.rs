//! Explicit opt-in qualification of the pinned HGPTQ checkpoint.
#![cfg(feature = "cuda")]

use std::path::PathBuf;
use w34_host::{
    cache::KvFormatPolicy,
    model::Qwen3_5Engine,
    proposer::{SpeculativeConfig, SpeculativeKind},
    vision,
};

fn checkpoint() -> PathBuf {
    let cache = std::env::var_os("HF_HUB_CACHE")
        .map(PathBuf::from)
        .unwrap_or_else(|| {
            PathBuf::from(std::env::var_os("HOME").expect("HOME")).join(".cache/huggingface/hub")
        });
    let path = cache.join("models--nchapman--Qwen3.5-2B-W4A16-HGPTQ-g64/snapshots/0fbb9a38f631ea63678f4b51a491e7ecb26c681c");
    assert!(
        path.join("model_mtp.safetensors").exists(),
        "download pinned HGPTQ checkpoint first: {}",
        path.display()
    );
    path
}

fn qualify(tp: usize) -> Vec<i32> {
    let mut engine = Qwen3_5Engine::load(
        checkpoint().to_str().unwrap(),
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
#[ignore = "requires pinned HGPTQ 2B checkpoint and two free CUDA GPUs"]
fn pinned_hgptq_text_graph_mtp_and_tp2() {
    let tp1 = qualify(1);
    let tp2 = qualify(2);
    assert_eq!(tp1, tp2, "TP=1 vs TP=2 greedy continuation");
}

#[test]
#[ignore = "requires pinned HGPTQ 2B checkpoint and two free CUDA GPUs"]
fn pinned_hgptq_vision_matches_vllm_and_tp2() {
    let fixture = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../tools/oracle/fixtures/coco_sample.png");
    let (rgb, h, w) = vision::load_cli_image_with_policy(fixture.to_str().unwrap(), false).unwrap();
    let image = vision::preprocess_rgb_image(&rgb, h, w, 16, 2, 2, 1024, 1048576);
    let expanded = "<|vision_start|><|image_pad|><|vision_end|>Describe this image.".replace(
        "<|image_pad|>",
        &"<|image_pad|>".repeat(image.num_merged_tokens(2)),
    );
    let images = [image];
    let expected_vllm_ids = vec![
        561, 2099, 4774, 1330, 18833, 19823, 383, 264, 17676, 25313, 13, 561, 18833, 513, 2107,
        5474, 1634, 18833, 11, 321, 781, 513, 2107, 20479, 13, 2532, 513, 1330, 8434, 11241, 383,
        279,
    ];

    let run_tp = |tp: usize| -> Vec<i32> {
        let mut engine = Qwen3_5Engine::load(
            checkpoint().to_str().unwrap(),
            tp,
            KvFormatPolicy::Fp16,
            Some(32),
            true,
        )
        .unwrap();
        let prompt = engine.encode(&expanded).unwrap();
        assert_eq!(prompt.len(), 306);
        engine
            .generate_multimodal_with_radix_cache(&prompt, &images, 32, 64, false)
            .unwrap()
            .generated_token_ids
    };

    let tp1 = run_tp(1);
    let tp2 = run_tp(2);
    assert_eq!(tp1, expected_vllm_ids, "TP=1 vision greedy vs vLLM 0.31.0");
    assert_eq!(tp1, tp2, "TP=1 vs TP=2 multimodal greedy continuation");
}
