#![cfg(feature = "cuda")]
mod common;

use std::{sync::atomic::Ordering, time::Duration};
use w34_host::{
    cache::KvFormatPolicy,
    engine_worker::{EngineWorker, GenerationEvent, GenerationStream},
    model::Qwen3_5Engine,
    parser::SamplingParams,
    proposer::{SpeculativeConfig, SpeculativeKind},
    scheduler::SchedulerConfig,
};

fn worker_with_policy(kind: Option<SpeculativeKind>, policy: KvFormatPolicy) -> EngineWorker {
    EngineWorker::from_loader(
        move || {
            let mut engine = Qwen3_5Engine::load_for_serving(
                "nchapman/Qwen3.5-4B-W4A16-GPTQ",
                1,
                policy,
                Some(256),
                true,
                16,
            )
            .map_err(|e| e.to_string())?;
            if let Some(kind) = kind {
                engine
                    .enable_speculative(&SpeculativeConfig {
                        kind,
                        mtp_k: 3,
                        draft_model: (kind == SpeculativeKind::DFlash2)
                            .then(|| "nchapman/Qwen3.5-4B-DFlash2".into()),
                    })
                    .map_err(|e| e.to_string())?;
            }
            Ok(engine)
        },
        SchedulerConfig::default(),
    )
    .unwrap()
}

fn worker(kind: SpeculativeKind) -> EngineWorker {
    worker_with_policy(Some(kind), KvFormatPolicy::Tq4)
}

fn params() -> SamplingParams {
    SamplingParams {
        max_tokens: 64,
        ignore_eos: true,
        temperature: 0.0,
        ..SamplingParams::default()
    }
}

async fn collect(stream: &mut GenerationStream, mut tokens: Vec<i32>) -> Result<Vec<i32>, String> {
    while let Some(event) = stream.events.recv().await {
        match event {
            GenerationEvent::Token { id, .. } => tokens.push(id),
            GenerationEvent::Finished {
                generated_tokens, ..
            } => {
                if generated_tokens != 64 || tokens.len() != 64 {
                    return Err(format!("counts: {generated_tokens}, {}", tokens.len()));
                }
                return Ok(tokens);
            }
            GenerationEvent::Error(e) => return Err(e.to_string()),
        }
    }
    Err("stream closed without Finished".into())
}

static GPU_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

// Run with flock /tmp/w34-gpu1.lock and CUDA_VISIBLE_DEVICES=1.
#[test]
fn native_speculation_yields_to_late_peer_losslessly() {
    if common::require_or_skip_cached_model(
        "nchapman/Qwen3.5-4B-W4A16-GPTQ",
        "native_speculation_yields_to_late_peer_losslessly",
    )
    .is_none()
        || common::require_or_skip_cached_model(
            "nchapman/Qwen3.5-4B-DFlash2",
            "native_speculation_yields_to_late_peer_losslessly",
        )
        .is_none()
    {
        return;
    }
    let _gpu_guard = GPU_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    let rt = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .unwrap();
    for kind in [SpeculativeKind::Mtp, SpeculativeKind::DFlash2] {
        // Loading, inference and destruction stay on the worker OS thread.
        let baseline_worker = worker(kind);
        let prompt: Vec<i32> = baseline_worker
            .tokenizer
            .as_ref()
            .unwrap()
            .encode("The capital of France is", false)
            .unwrap()
            .get_ids()
            .iter()
            .map(|&id| id as i32)
            .collect();
        let baseline = rt.block_on(async {
            let mut stream = baseline_worker
                .submit(prompt.clone(), params(), None)
                .unwrap();
            let result =
                tokio::time::timeout(Duration::from_secs(120), collect(&mut stream, vec![])).await;
            baseline_worker.shutdown().await.unwrap();
            assert!(baseline_worker.stats.verify_rounds.load(Ordering::Relaxed) > 0);
            result.unwrap().unwrap()
        });
        let worker = worker(kind);
        let result = rt.block_on(async {
            let result = tokio::time::timeout(Duration::from_secs(120), async {
                let mut first = worker.submit(prompt.clone(), params(), None)?;
                let mut prefix = Vec::new();
                while prefix.len() < 2 {
                    match first.events.recv().await {
                        Some(GenerationEvent::Token { id, .. }) => prefix.push(id),
                        other => return Err(format!("premature event: {other:?}")),
                    }
                }
                assert!(worker.stats.verify_rounds.load(Ordering::Relaxed) > 0);
                assert_eq!(worker.stats.completed.load(Ordering::Relaxed), 0);
                let mut peer = worker.submit(prompt.clone(), params(), None)?;
                // Drain both concurrently: output backpressure must not cancel either.
                let (first, peer) =
                    tokio::join!(collect(&mut first, prefix), collect(&mut peer, vec![]));
                let first = first?;
                peer?;
                Ok::<_, String>(first)
            })
            .await;
            worker.shutdown().await.unwrap();
            result
        });
        assert_eq!(
            result.unwrap().unwrap(),
            baseline,
            "{kind:?}: exact greedy output"
        );
        assert!(
            worker.stats.peak_active.load(Ordering::Relaxed) >= 2,
            "{kind:?}: late peer was held until speculative completion"
        );
        assert_eq!(worker.stats.completed.load(Ordering::Relaxed), 2);
        assert_eq!(worker.stats.cancelled.load(Ordering::Relaxed), 0);
    }
}

#[test]
fn batched_mtp_b4_matches_single_stream_losslessly() {
    if common::require_or_skip_cached_model(
        "nchapman/Qwen3.5-4B-W4A16-GPTQ",
        "batched_mtp_b4_matches_single_stream_losslessly",
    )
    .is_none()
    {
        return;
    }
    let _gpu_guard = GPU_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    let rt = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .unwrap();
    let texts = [
        "The capital of France is",
        "In computer architecture, cache coherence ensures",
        "Rust ownership rules guarantee memory safety by",
        "Binary search finds an element in a sorted array in",
    ];
    let baseline_worker = worker(SpeculativeKind::Mtp);
    let prompts: Vec<Vec<i32>> = texts
        .iter()
        .map(|s| {
            baseline_worker
                .tokenizer
                .as_ref()
                .unwrap()
                .encode(*s, false)
                .unwrap()
                .get_ids()
                .iter()
                .map(|&id| id as i32)
                .collect()
        })
        .collect();
    let baselines = rt.block_on(async {
        let mut outs = Vec::with_capacity(prompts.len());
        for p in &prompts {
            let mut st = baseline_worker.submit(p.clone(), params(), None).unwrap();
            let got = tokio::time::timeout(Duration::from_secs(120), collect(&mut st, vec![]))
                .await
                .unwrap()
                .unwrap();
            outs.push(got);
        }
        baseline_worker.shutdown().await.unwrap();
        outs
    });

    let batch_worker = worker(SpeculativeKind::Mtp);
    let got_all = rt.block_on(async {
        let mut s0 = batch_worker
            .submit(prompts[0].clone(), params(), None)
            .unwrap();
        let mut s1 = batch_worker
            .submit(prompts[1].clone(), params(), None)
            .unwrap();
        let mut s2 = batch_worker
            .submit(prompts[2].clone(), params(), None)
            .unwrap();
        let mut s3 = batch_worker
            .submit(prompts[3].clone(), params(), None)
            .unwrap();
        let res = tokio::time::timeout(Duration::from_secs(120), async {
            let (o0, o1, o2, o3) = tokio::join!(
                collect(&mut s0, vec![]),
                collect(&mut s1, vec![]),
                collect(&mut s2, vec![]),
                collect(&mut s3, vec![]),
            );
            Ok::<_, String>(vec![o0?, o1?, o2?, o3?])
        })
        .await
        .unwrap()
        .unwrap();
        batch_worker.shutdown().await.unwrap();
        res
    });

    assert_eq!(
        got_all, baselines,
        "batched B=4 MTP (TQ4) must match single-stream B=1 MTP greedy bit-for-bit across all 4 slots"
    );
    assert_eq!(batch_worker.stats.peak_active.load(Ordering::Relaxed), 4);
    assert!(
        batch_worker.stats.verify_rounds.load(Ordering::Relaxed) >= 40,
        "expected batched MTP verify rounds across all 4 slots, got {}",
        batch_worker.stats.verify_rounds.load(Ordering::Relaxed)
    );
    assert!(
        batch_worker.stats.accepted_drafts.load(Ordering::Relaxed) > 0,
        "expected accepted MTP drafts in batched B=4 verify"
    );

    // 2. Verify B=2 batched MTP (`M=8` CUDA Graph slot) matches single-stream B=1 bit-for-bit,
    //    then verify B=2 GPU-resident stochastic verification (`verify_batched_speculative_gpu_steps`).
    let b2_worker = worker(SpeculativeKind::Mtp);
    let (got_b2, gpu_stoch_a, gpu_stoch_b) = rt.block_on(async {
        let mut s0 = b2_worker
            .submit(prompts[0].clone(), params(), None)
            .unwrap();
        let mut s1 = b2_worker
            .submit(prompts[1].clone(), params(), None)
            .unwrap();
        let got_b2 = tokio::time::timeout(Duration::from_secs(120), async {
            let (o0, o1) = tokio::join!(collect(&mut s0, vec![]), collect(&mut s1, vec![]));
            Ok::<_, String>(vec![o0?, o1?])
        })
        .await
        .unwrap()
        .unwrap();

        // 2a. Plain greedy (`ignore_eos = false`) exercises the `any_gpu_sampler == false`
        //     (`RejectionSampler::greedy`) branch in `step_scheduled_batched_mtp_verify_round`,
        //     while `params()` (`ignore_eos = true`) above exercised `verify_batched_speculative_gpu_steps`.
        let mut plain_greedy = params();
        plain_greedy.ignore_eos = false;
        let rounds_before_plain = b2_worker.stats.verify_rounds.load(Ordering::Relaxed);
        let mut g0 = b2_worker
            .submit(prompts[2].clone(), plain_greedy.clone(), None)
            .unwrap();
        let mut g1 = b2_worker
            .submit(prompts[3].clone(), plain_greedy, None)
            .unwrap();
        let (gpu_stoch_a, gpu_stoch_b) = tokio::time::timeout(Duration::from_secs(120), async {
            let (o0, o1) = tokio::join!(collect(&mut g0, vec![]), collect(&mut g1, vec![]));
            Ok::<_, String>((o0?, o1?))
        })
        .await
        .unwrap()
        .unwrap();
        assert!(
            b2_worker.stats.verify_rounds.load(Ordering::Relaxed) > rounds_before_plain,
            "B=2 plain-greedy path must execute speculative verify rounds"
        );

        b2_worker.shutdown().await.unwrap();
        (got_b2, gpu_stoch_a, gpu_stoch_b)
    });
    assert_eq!(
        &got_b2[..],
        &baselines[..2],
        "B=2 batched MTP must match B=1"
    );
    assert_eq!(
        gpu_stoch_a, baselines[2],
        "B=2 plain-greedy slot 0 must match baseline[2]"
    );
    assert_eq!(
        gpu_stoch_b, baselines[3],
        "B=2 plain-greedy slot 1 must match baseline[3]"
    );
}

async fn collect_b5(worker: &EngineWorker, prompts: &[Vec<i32>; 5]) -> Vec<Vec<i32>> {
    // Submit the entire burst without awaiting; use the worker's existing coalescing.
    let [mut s0, mut s1, mut s2, mut s3, mut s4] = prompts
        .each_ref()
        .map(|prompt| worker.submit(prompt.clone(), params(), None).unwrap());
    let result = tokio::time::timeout(Duration::from_secs(120), async {
        let (o0, o1, o2, o3, o4) = tokio::join!(
            collect(&mut s0, vec![]),
            collect(&mut s1, vec![]),
            collect(&mut s2, vec![]),
            collect(&mut s3, vec![]),
            collect(&mut s4, vec![]),
        );
        Ok::<_, String>(vec![o0?, o1?, o2?, o3?, o4?])
    })
    .await
    .unwrap()
    .unwrap();
    worker.shutdown().await.unwrap();
    let outputs = result;
    assert_eq!(worker.stats.peak_active.load(Ordering::Relaxed), 5);
    assert_eq!(worker.stats.verify_rounds.load(Ordering::Relaxed), 0);
    assert_eq!(worker.stats.completed.load(Ordering::Relaxed), 5);
    assert_eq!(worker.stats.cancelled.load(Ordering::Relaxed), 0);
    assert_eq!(outputs[4], outputs[0], "B=5 duplicate slots must match");
    outputs
}

#[test]
fn batched_mtp_b5_fallback_matches_non_speculative_losslessly() {
    if common::require_or_skip_cached_model(
        "nchapman/Qwen3.5-4B-W4A16-GPTQ",
        "batched_mtp_b5_fallback_matches_non_speculative_losslessly",
    )
    .is_none()
    {
        return;
    }
    let _gpu_guard = GPU_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    let rt = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .unwrap();
    let fallback_worker = worker(SpeculativeKind::Mtp);
    let prompts = [
        "The capital of France is",
        "In computer architecture, cache coherence ensures",
        "Rust ownership rules guarantee memory safety by",
        "Binary search finds an element in a sorted array in",
        "The capital of France is",
    ]
    .map(|text| {
        fallback_worker
            .tokenizer
            .as_ref()
            .unwrap()
            .encode(text, false)
            .unwrap()
            .get_ids()
            .iter()
            .map(|&id| id as i32)
            .collect()
    });
    // B=5 exceeds MAX_MTP_BATCH_SLOTS=4 and must use ordinary batched decode.
    // Its oracle is fresh non-speculative B=5 with identical prompts/order, not B=1:
    // cuBLAS can choose different reduction algorithms at different M dimensions.
    // This avoids conflating cross-geometry numerical differences with fallback bugs;
    // it does not establish the cause of any particular B=5 versus B=1 mismatch.
    let fallback = rt.block_on(collect_b5(&fallback_worker, &prompts));
    let baseline_worker = worker_with_policy(None, KvFormatPolicy::Tq4);
    let baseline = rt.block_on(collect_b5(&baseline_worker, &prompts));
    assert_eq!(
        fallback, baseline,
        "B=5 MTP fallback must match non-speculative B=5 across all five complete outputs"
    );
}
