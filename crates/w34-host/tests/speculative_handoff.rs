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

async fn collect_with_cached(
    stream: &mut GenerationStream,
    expected_len: usize,
) -> Result<(Vec<i32>, usize), String> {
    let mut tokens = Vec::new();
    while let Some(event) = stream.events.recv().await {
        match event {
            GenerationEvent::Token { id, .. } => tokens.push(id),
            GenerationEvent::Finished {
                cached_tokens,
                generated_tokens,
            } => {
                if generated_tokens != expected_len || tokens.len() != expected_len {
                    return Err(format!(
                        "counts: finished={generated_tokens}, got={}, expected={expected_len}",
                        tokens.len()
                    ));
                }
                return Ok((tokens, cached_tokens));
            }
            GenerationEvent::Error(e) => return Err(e.to_string()),
        }
    }
    Err("stream closed without Finished".into())
}

#[test]
fn batched_mtp_max4_handles_8_burst_and_reuses_warm_prefixes() {
    if common::require_or_skip_cached_model(
        "nchapman/Qwen3.5-4B-W4A16-GPTQ",
        "batched_mtp_max4_handles_8_burst_and_reuses_warm_prefixes",
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

    let shared_prefix = "System architecture notes: requests carry bounded deadlines, workers pull from bounded queues, and cache invalidations execute strictly after transaction commit to preserve serializability across concurrent readers and writers. ".repeat(6);
    let suffixes = [
        "List two race conditions in cache invalidation:",
        "Explain how bounded queues prevent memory exhaustion:",
        "Describe safe cancellation semantics for streaming RPCs:",
        "Summarize retry backoff invariants under overload:",
        "Give pseudocode for atomic lease renewal:",
        "Identify lock-ordering hazards in connection pools:",
        "Explain read-your-writes consistency with replicas:",
        "Outline a deterministic test for timeout cleanup:",
    ];
    let followup_tail = " Next turn: provide a concise checklist for code review.";

    let mut short_params = params();
    short_params.max_tokens = 32;

    let baseline_worker = EngineWorker::from_loader(
        || {
            Qwen3_5Engine::load_for_serving_with_speculative(
                "nchapman/Qwen3.5-4B-W4A16-GPTQ",
                1,
                KvFormatPolicy::Tq4,
                Some(1024),
                true,
                1,
                Some(&SpeculativeConfig {
                    kind: SpeculativeKind::Mtp,
                    draft_model: None,
                    mtp_k: 3,
                }),
            )
            .map_err(|e| e.to_string())
        },
        SchedulerConfig {
            max_num_seqs: 1,
            ..SchedulerConfig::default()
        },
    )
    .unwrap();

    let tok = baseline_worker.tokenizer.as_ref().unwrap().clone();
    let encode = |text: &str| -> Vec<i32> {
        tok.encode(text, false)
            .unwrap()
            .get_ids()
            .iter()
            .map(|&id| id as i32)
            .collect()
    };
    let burst_prompts: Vec<Vec<i32>> = suffixes
        .iter()
        .map(|s| encode(&format!("{shared_prefix}{s}")))
        .collect();
    assert!(
        burst_prompts.iter().all(|p| p.len() >= 192),
        "shared prefix must cross >= 2 Mamba 64-token checkpoints"
    );
    let followup_suffix_ids = encode(followup_tail);

    let (burst_baselines, followup_prompts, followup_baselines) = rt.block_on(async {
        let mut burst_outs = Vec::with_capacity(burst_prompts.len());
        for p in &burst_prompts {
            let mut st = baseline_worker
                .submit(p.clone(), short_params.clone(), None)
                .unwrap();
            let (out, _) =
                tokio::time::timeout(Duration::from_secs(120), collect_with_cached(&mut st, 32))
                    .await
                    .unwrap()
                    .unwrap();
            burst_outs.push(out);
        }
        let mut follow_prompts = Vec::with_capacity(4);
        let mut follow_outs = Vec::with_capacity(4);
        for i in 0..4 {
            let mut fp = burst_prompts[i].clone();
            fp.extend_from_slice(&burst_outs[i]);
            fp.extend_from_slice(&followup_suffix_ids);
            let mut st = baseline_worker
                .submit(fp.clone(), short_params.clone(), None)
                .unwrap();
            let (out, _) =
                tokio::time::timeout(Duration::from_secs(120), collect_with_cached(&mut st, 32))
                    .await
                    .unwrap()
                    .unwrap();
            follow_prompts.push(fp);
            follow_outs.push(out);
        }
        baseline_worker.shutdown().await.unwrap();
        (burst_outs, follow_prompts, follow_outs)
    });

    let burst_worker = EngineWorker::from_loader(
        || {
            Qwen3_5Engine::load_for_serving_with_speculative(
                "nchapman/Qwen3.5-4B-W4A16-GPTQ",
                1,
                KvFormatPolicy::Tq4,
                Some(1024),
                true,
                4,
                Some(&SpeculativeConfig {
                    kind: SpeculativeKind::Mtp,
                    draft_model: None,
                    mtp_k: 3,
                }),
            )
            .map_err(|e| e.to_string())
        },
        SchedulerConfig {
            max_num_seqs: 4,
            ..SchedulerConfig::default()
        },
    )
    .unwrap();

    let (burst_got, rounds_after_burst, followup_got, rounds_after_followup) = rt.block_on(async {
        // Submit all 8 burst requests simultaneously while max_num_seqs == 4.
        let mut streams: Vec<_> = burst_prompts
            .iter()
            .map(|p| {
                burst_worker
                    .submit(p.clone(), short_params.clone(), None)
                    .unwrap()
            })
            .collect();
        let mut s_iter = streams.drain(..);
        let (mut s0, mut s1, mut s2, mut s3, mut s4, mut s5, mut s6, mut s7) = (
            s_iter.next().unwrap(),
            s_iter.next().unwrap(),
            s_iter.next().unwrap(),
            s_iter.next().unwrap(),
            s_iter.next().unwrap(),
            s_iter.next().unwrap(),
            s_iter.next().unwrap(),
            s_iter.next().unwrap(),
        );
        let burst_got = tokio::time::timeout(Duration::from_secs(120), async {
            let (r0, r1, r2, r3, r4, r5, r6, r7) = tokio::join!(
                collect_with_cached(&mut s0, 32),
                collect_with_cached(&mut s1, 32),
                collect_with_cached(&mut s2, 32),
                collect_with_cached(&mut s3, 32),
                collect_with_cached(&mut s4, 32),
                collect_with_cached(&mut s5, 32),
                collect_with_cached(&mut s6, 32),
                collect_with_cached(&mut s7, 32),
            );
            Ok::<_, String>(vec![r0?, r1?, r2?, r3?, r4?, r5?, r6?, r7?])
        })
        .await
        .unwrap()
        .unwrap();
        let rounds_after_burst = burst_worker.stats.verify_rounds.load(Ordering::Relaxed);

        // Submit a 4-request multi-turn follow-up wave extending the first 4 burst conversations.
        let mut f_streams: Vec<_> = followup_prompts
            .iter()
            .map(|p| {
                burst_worker
                    .submit(p.clone(), short_params.clone(), None)
                    .unwrap()
            })
            .collect();
        let mut f_iter = f_streams.drain(..);
        let (mut f0, mut f1, mut f2, mut f3) = (
            f_iter.next().unwrap(),
            f_iter.next().unwrap(),
            f_iter.next().unwrap(),
            f_iter.next().unwrap(),
        );
        let followup_got = tokio::time::timeout(Duration::from_secs(120), async {
            let (r0, r1, r2, r3) = tokio::join!(
                collect_with_cached(&mut f0, 32),
                collect_with_cached(&mut f1, 32),
                collect_with_cached(&mut f2, 32),
                collect_with_cached(&mut f3, 32),
            );
            Ok::<_, String>(vec![r0?, r1?, r2?, r3?])
        })
        .await
        .unwrap()
        .unwrap();
        let rounds_after_followup = burst_worker.stats.verify_rounds.load(Ordering::Relaxed);

        burst_worker.shutdown().await.unwrap();
        (
            burst_got,
            rounds_after_burst,
            followup_got,
            rounds_after_followup,
        )
    });

    assert_eq!(burst_worker.stats.peak_active.load(Ordering::Relaxed), 4);
    assert!(
        rounds_after_burst >= 110,
        "8-request burst under max_num_seqs=4 must stay in batched MTP instead of plain fallback, got {rounds_after_burst} rounds"
    );
    assert!(
        rounds_after_followup >= rounds_after_burst + 50,
        "4-request follow-up wave must execute batched MTP verify rounds, before={rounds_after_burst} after={rounds_after_followup}"
    );
    assert!(
        burst_worker.stats.accepted_drafts.load(Ordering::Relaxed) > 0,
        "expected accepted drafts across burst and follow-up waves"
    );
    for (i, ((got, _), expected)) in burst_got.iter().zip(&burst_baselines).enumerate() {
        assert_eq!(
            got, expected,
            "burst slot {i} must match B=1 greedy baseline"
        );
    }
    assert!(
        burst_got[4..].iter().all(|(_, cached)| *cached >= 128),
        "second 4-request wave of shared-prefix burst must reuse warm radix+MTP prefix, got {:?}",
        burst_got
            .iter()
            .map(|(_, cached)| *cached)
            .collect::<Vec<_>>()
    );
    for (i, ((got, cached), expected)) in followup_got.iter().zip(&followup_baselines).enumerate() {
        assert_eq!(
            got, expected,
            "follow-up turn {i} must match B=1 greedy baseline"
        );
        assert!(
            *cached >= 192,
            "multi-turn follow-up {i} in batched MTP must reuse prompt+generated prefix, got cached={cached}"
        );
    }
}
