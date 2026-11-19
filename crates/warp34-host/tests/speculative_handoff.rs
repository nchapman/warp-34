#![cfg(feature = "cuda")]
mod common;

use std::{sync::atomic::Ordering, time::Duration};
use warp34_host::{
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
            let mut engine =
                Qwen3_5Engine::load_for_serving("Qwen/Qwen3.5-4B", 1, policy, Some(256), true, 16)
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

// Run with flock /tmp/warp34-gpu1.lock and CUDA_VISIBLE_DEVICES=1.
#[test]
fn native_speculation_yields_to_late_peer_losslessly() {
    if common::require_or_skip_cached_model(
        "Qwen/Qwen3.5-4B",
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
        // Use an unambiguous factual prompt rather than "The capital of France is", which on
        // sm_89 TQ4 hits an exact 1-FP16-ULP (0.015625) tie at token 10 ("\n" vs "\n\n") when
        // DFlash2 yields from M=16 Tensor Core verify to M=2 cuBLAS batched decode.
        let baseline_worker = worker(kind);
        let prompt: Vec<i32> = baseline_worker
            .tokenizer
            .as_ref()
            .unwrap()
            .encode("In computer architecture, cache coherence ensures", false)
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
        "Qwen/Qwen3.5-4B",
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
        "Qwen/Qwen3.5-4B",
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
                ..
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
        "Qwen/Qwen3.5-4B",
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

    let lane_tags = [
        "[Lane Alpha] ",
        "[Lane Beta] ",
        "[Lane Gamma] ",
        "[Lane Delta] ",
    ];
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
                "Qwen/Qwen3.5-4B",
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
    // Pair `(i, i + 4)` shares a >= 128-token prefix while wave 1 (`0..4`) starts cold in both
    // `B=1` and `B=4`, so both workers traverse the exact same 1-hop `WHT128-Q8_0` checkpoints.
    let burst_prompts: Vec<Vec<i32>> = suffixes
        .iter()
        .enumerate()
        .map(|(i, s)| encode(&format!("{}{shared_prefix}{s}", lane_tags[i % 4])))
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
                "Qwen/Qwen3.5-4B",
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
        rounds_after_burst >= 80,
        "8-request burst under max_num_seqs=4 must stay in batched MTP instead of plain fallback, got {rounds_after_burst} rounds"
    );
    assert!(
        rounds_after_followup >= rounds_after_burst + 30,
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

#[test]
fn batched_mtp_stochastic_and_tp2_greedy_stay_speculative() {
    if common::require_or_skip_cached_model(
        "Qwen/Qwen3.5-4B",
        "batched_mtp_stochastic_and_tp2_greedy_stay_speculative",
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

    // 1. Stochastic B=4 batched MTP must stay in speculative verify rounds (not fall back to plain decode).
    let stoch_worker = EngineWorker::from_loader(
        || {
            let mut engine = Qwen3_5Engine::load_for_serving(
                "Qwen/Qwen3.5-4B",
                1,
                KvFormatPolicy::Tq4,
                Some(512),
                true,
                16,
            )
            .map_err(|e| e.to_string())?;
            engine
                .enable_speculative(&SpeculativeConfig {
                    kind: SpeculativeKind::Mtp,
                    mtp_k: 3,
                    draft_model: None,
                })
                .map_err(|e| e.to_string())?;
            Ok(engine)
        },
        SchedulerConfig {
            max_num_seqs: 4,
            ..SchedulerConfig::default()
        },
    )
    .unwrap();

    let prompts: Vec<Vec<i32>> = (0..4)
        .map(|i| {
            let mut p: Vec<i32> = (0..96).map(|j| 200 + ((j * 5) % 89)).collect();
            p.extend((0..16).map(|j| 600 + (i * 19 + j)));
            p
        })
        .collect();

    let (stoch_got, stoch_rounds, stoch_accepted) = rt.block_on(async {
        let mut streams: Vec<_> = prompts
            .iter()
            .enumerate()
            .map(|(i, p)| {
                let sp = SamplingParams {
                    max_tokens: 32,
                    ignore_eos: true,
                    temperature: 0.7,
                    top_p: 0.9,
                    seed: 1000 + i as u64,
                    ..SamplingParams::default()
                };
                stoch_worker.submit(p.clone(), sp, None).unwrap()
            })
            .collect();
        let mut it = streams.drain(..);
        let (mut s0, mut s1, mut s2, mut s3) = (
            it.next().unwrap(),
            it.next().unwrap(),
            it.next().unwrap(),
            it.next().unwrap(),
        );
        let out = tokio::time::timeout(Duration::from_secs(120), async {
            let (r0, r1, r2, r3) = tokio::join!(
                collect_with_cached(&mut s0, 32),
                collect_with_cached(&mut s1, 32),
                collect_with_cached(&mut s2, 32),
                collect_with_cached(&mut s3, 32),
            );
            Ok::<_, String>(vec![r0?, r1?, r2?, r3?])
        })
        .await
        .unwrap()
        .unwrap();
        let rounds = stoch_worker.stats.verify_rounds.load(Ordering::Relaxed);
        let acc = stoch_worker.stats.accepted_drafts.load(Ordering::Relaxed);
        stoch_worker.shutdown().await.unwrap();
        (out, rounds, acc)
    });
    assert!(
        stoch_rounds >= 20,
        "stochastic B=4 must execute batched MTP verify rounds, got {stoch_rounds}"
    );
    assert!(
        stoch_accepted > 0,
        "stochastic B=4 batched MTP must accept draft tokens"
    );
    assert!(stoch_got.iter().all(|(toks, _)| toks.len() == 32));

    // 2. TP=2 greedy B=4 batched MTP must match B=1 non-speculative greedy bit-for-bit.
    let tp2_plain = EngineWorker::from_loader(
        || {
            Qwen3_5Engine::load_for_serving(
                "Qwen/Qwen3.5-4B",
                2,
                KvFormatPolicy::Tq4,
                Some(512),
                true,
                16,
            )
            .map_err(|e| e.to_string())
        },
        SchedulerConfig {
            max_num_seqs: 1,
            ..SchedulerConfig::default()
        },
    )
    .unwrap();
    let greedy_short = SamplingParams {
        max_tokens: 32,
        ignore_eos: true,
        temperature: 0.0,
        ..SamplingParams::default()
    };
    let tp2_expected = rt.block_on(async {
        let mut out = Vec::with_capacity(prompts.len());
        for p in &prompts {
            let mut s = tp2_plain
                .submit(p.clone(), greedy_short.clone(), None)
                .unwrap();
            let (toks, _) =
                tokio::time::timeout(Duration::from_secs(90), collect_with_cached(&mut s, 32))
                    .await
                    .unwrap()
                    .unwrap();
            out.push(toks);
        }
        tp2_plain.shutdown().await.unwrap();
        out
    });

    let tp2_mtp = EngineWorker::from_loader(
        || {
            let mut engine = Qwen3_5Engine::load_for_serving(
                "Qwen/Qwen3.5-4B",
                2,
                KvFormatPolicy::Tq4,
                Some(512),
                true,
                16,
            )
            .map_err(|e| e.to_string())?;
            engine
                .enable_speculative(&SpeculativeConfig {
                    kind: SpeculativeKind::Mtp,
                    mtp_k: 3,
                    draft_model: None,
                })
                .map_err(|e| e.to_string())?;
            Ok(engine)
        },
        SchedulerConfig {
            max_num_seqs: 4,
            ..SchedulerConfig::default()
        },
    )
    .unwrap();
    let (tp2_got, tp2_rounds, tp2_accepted) = rt.block_on(async {
        let mut streams: Vec<_> = prompts
            .iter()
            .map(|p| {
                tp2_mtp
                    .submit(p.clone(), greedy_short.clone(), None)
                    .unwrap()
            })
            .collect();
        let mut it = streams.drain(..);
        let (mut s0, mut s1, mut s2, mut s3) = (
            it.next().unwrap(),
            it.next().unwrap(),
            it.next().unwrap(),
            it.next().unwrap(),
        );
        let out = tokio::time::timeout(Duration::from_secs(120), async {
            let (r0, r1, r2, r3) = tokio::join!(
                collect_with_cached(&mut s0, 32),
                collect_with_cached(&mut s1, 32),
                collect_with_cached(&mut s2, 32),
                collect_with_cached(&mut s3, 32),
            );
            Ok::<_, String>(vec![r0?, r1?, r2?, r3?])
        })
        .await
        .unwrap()
        .unwrap();
        let rounds = tp2_mtp.stats.verify_rounds.load(Ordering::Relaxed);
        let acc = tp2_mtp.stats.accepted_drafts.load(Ordering::Relaxed);
        tp2_mtp.shutdown().await.unwrap();
        (out, rounds, acc)
    });
    assert!(
        tp2_rounds >= 30,
        "TP=2 B=4 batched MTP must execute verify rounds, got {tp2_rounds}"
    );
    assert!(tp2_accepted > 0, "TP=2 B=4 batched MTP must accept drafts");
    for (i, ((got, _), exp)) in tp2_got.iter().zip(&tp2_expected).enumerate() {
        assert_eq!(got, exp, "TP=2 B=4 slot {i} must match greedy baseline");
    }
}

#[test]
fn batched_mtp_full_window_budgets_stay_speculative_and_preserve_radix_cache() {
    if common::require_or_skip_cached_model(
        "Qwen/Qwen3.5-4B",
        "batched_mtp_full_window_budgets_stay_speculative_and_preserve_radix_cache",
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

    // kv_pages = 512 => 8,192 total tokens, MTP batch slot_cap = (512 / 4) * 16 = 2,048 tokens.
    // Four concurrent requests with `max_tokens = 8192 - prompt_len` (exceeding `slot_cap` and
    // exceeding 100% of the KV pool in sum) must still admit together on tick 1 (`peak_active = 4`),
    // run batched MTP verify rounds (`verify_rounds > 0`), and leave an earlier warm session's
    // MambaRadixCache prefix intact via incremental KV slab allocation.
    let worker = EngineWorker::from_loader(
        || {
            let mut engine = Qwen3_5Engine::load_for_serving(
                "Qwen/Qwen3.5-4B",
                1,
                KvFormatPolicy::Tq4,
                Some(512),
                true,
                8,
            )
            .map_err(|e| e.to_string())?;
            engine
                .enable_speculative(&SpeculativeConfig {
                    kind: SpeculativeKind::Mtp,
                    mtp_k: 3,
                    draft_model: None,
                })
                .map_err(|e| e.to_string())?;
            Ok(engine)
        },
        SchedulerConfig {
            max_num_seqs: 4,
            ..SchedulerConfig::default()
        },
    )
    .unwrap();
    let max_len = worker.max_model_len();
    assert_eq!(max_len, 8192);

    let tok = worker.tokenizer.as_ref().unwrap().clone();
    let encode = |text: &str| -> Vec<i32> {
        tok.encode(text, false)
            .unwrap()
            .get_ids()
            .iter()
            .map(|&id| id as i32)
            .collect()
    };

    let warm_prefix_text = "Distributed consensus invariant log: every committed raft entry carries a monotonic term and index across quorum replicas, and follower snapshots advance state machines atomically at chunk boundaries. ".repeat(6);
    let warm_seed_prompt = encode(&format!(
        "{warm_prefix_text}Summarize leader election safety in one phrase:"
    ));
    let warm_probe_prompt = encode(&format!(
        "{warm_prefix_text}Summarize log matching safety in one phrase:"
    ));
    assert!(
        warm_seed_prompt.len() >= 192,
        "warm session prefix must span >= 3 Mamba 64-token checkpoints"
    );

    let burst_prompts: Vec<Vec<i32>> = [
        "The capital of France is",
        "In computer architecture, cache coherence ensures",
        "Rust ownership rules guarantee memory safety by",
        "Binary search finds an element in a sorted array in",
    ]
    .iter()
    .map(|s| encode(s))
    .collect();

    async fn collect_first_n_then_cancel(
        mut stream: GenerationStream,
        target: usize,
    ) -> Result<Vec<i32>, String> {
        let mut out = Vec::with_capacity(target);
        while let Some(ev) = stream.events.recv().await {
            match ev {
                GenerationEvent::Token { id, .. } => {
                    out.push(id);
                    if out.len() >= target {
                        drop(stream);
                        return Ok(out);
                    }
                }
                GenerationEvent::Finished { .. } => return Ok(out),
                GenerationEvent::Error(e) => return Err(e.to_string()),
            }
        }
        Err("stream ended before target tokens".into())
    }

    let (rounds_delta, probe_cached) = rt.block_on(async {
        // 1. Seed a warm 192+-token session into MambaRadixCache.
        let mut seed_stream = worker
            .submit(
                warm_seed_prompt,
                SamplingParams {
                    max_tokens: 16,
                    ignore_eos: true,
                    temperature: 0.0,
                    ..SamplingParams::default()
                },
                None,
            )
            .unwrap();
        let _ = tokio::time::timeout(
            Duration::from_secs(60),
            collect_with_cached(&mut seed_stream, 16),
        )
        .await
        .unwrap()
        .unwrap();

        // 2. Launch 4 concurrent full-window requests (`max_tokens = max_len - prompt.len()`,
        //    each >7,900 tokens > slot_cap=2,048, sum >31,000 tokens >> 8,192 KV pool).
        let rounds_before = worker.stats.verify_rounds.load(Ordering::Relaxed);
        let mut streams: Vec<_> = burst_prompts
            .iter()
            .map(|p| {
                let sp = SamplingParams {
                    max_tokens: max_len - p.len(),
                    ignore_eos: true,
                    temperature: 0.0,
                    ..SamplingParams::default()
                };
                worker
                    .submit_with_images_lora_and_budget_mode(
                        p.clone(),
                        Vec::new(),
                        sp,
                        None,
                        warp34_host::lora::BASE_MODEL_LORA_ID,
                        true,
                    )
                    .unwrap()
            })
            .collect();
        let mut it = streams.drain(..);
        let (s0, s1, s2, s3) = (
            it.next().unwrap(),
            it.next().unwrap(),
            it.next().unwrap(),
            it.next().unwrap(),
        );
        tokio::time::timeout(Duration::from_secs(120), async {
            let (r0, r1, r2, r3) = tokio::join!(
                collect_first_n_then_cancel(s0, 32),
                collect_first_n_then_cancel(s1, 32),
                collect_first_n_then_cancel(s2, 32),
                collect_first_n_then_cancel(s3, 32),
            );
            r0?;
            r1?;
            r2?;
            r3?;
            Ok::<_, String>(())
        })
        .await
        .unwrap()
        .unwrap();
        let rounds_after = worker.stats.verify_rounds.load(Ordering::Relaxed);

        // 3. Probe the earlier warm prefix: incremental KV slab allocation must have kept it
        //    resident in MambaRadixCache rather than evicting the pool for 4 full-window budgets.
        let mut probe_stream = worker
            .submit(
                warm_probe_prompt,
                SamplingParams {
                    max_tokens: 16,
                    ignore_eos: true,
                    temperature: 0.0,
                    ..SamplingParams::default()
                },
                None,
            )
            .unwrap();
        let (_, cached) = tokio::time::timeout(
            Duration::from_secs(60),
            collect_with_cached(&mut probe_stream, 16),
        )
        .await
        .unwrap()
        .unwrap();
        worker.shutdown().await.unwrap();
        (rounds_after - rounds_before, cached)
    });

    assert_eq!(
        worker.stats.peak_active.load(Ordering::Relaxed),
        4,
        "4 concurrent full-window soft-budget requests must admit together instead of serializing"
    );
    assert!(
        rounds_delta >= 20,
        "4 concurrent full-window requests must execute batched MTP verify rounds, got {rounds_delta}"
    );
    assert!(
        probe_cached >= 128,
        "warm MambaRadixCache prefix must survive concurrent full-window requests via incremental KV slabs, got cached={probe_cached}"
    );
}

#[test]
fn realistic_swe_trajectory_caches_across_plain_mtp_and_dflash2() {
    use serde_json::{Value, json};
    use warp34_host::{
        chatml::{ChatMessage, format_chatml_prompt_with_options, parse_history_tools},
        parser::Qwen3Parser,
    };

    if common::require_or_skip_cached_model(
        "Qwen/Qwen3.5-4B",
        "realistic_swe_trajectory_caches_across_plain_mtp_and_dflash2",
    )
    .is_none()
        || common::require_or_skip_cached_model(
            "nchapman/Qwen3.5-4B-DFlash2",
            "realistic_swe_trajectory_caches_across_plain_mtp_and_dflash2",
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

    // Load the first realistic trajectory from `benchmarks/perf/fixtures/open_swe_intact_16k.json`.
    let fixture_path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../benchmarks/perf/fixtures/open_swe_intact_16k.json");
    let fixture_json: Value =
        serde_json::from_str(&std::fs::read_to_string(&fixture_path).unwrap()).unwrap();
    let traj = &fixture_json["trajectories"][0];
    let system_prompt = traj["system"].as_str().unwrap();
    let turns = traj["turns"].as_array().unwrap();

    let run_trajectory = |kind: Option<SpeculativeKind>| -> (Vec<Vec<i32>>, Vec<usize>, usize) {
        let worker = EngineWorker::from_loader(
            move || {
                let mut engine = Qwen3_5Engine::load_for_serving(
                    "Qwen/Qwen3.5-4B",
                    1,
                    KvFormatPolicy::Tq4,
                    Some(512),
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
            SchedulerConfig {
                max_num_seqs: 1,
                ..SchedulerConfig::default()
            },
        )
        .unwrap();
        let tok = worker.tokenizer.as_ref().unwrap().clone();
        let encode = |text: &str| -> Vec<i32> {
            tok.encode(text, false)
                .unwrap()
                .get_ids()
                .iter()
                .map(|&id| id as i32)
                .collect()
        };
        let decode = |ids: &[i32]| -> String {
            let u32_ids: Vec<u32> = ids.iter().map(|&x| x as u32).collect();
            tok.decode(&u32_ids, false).unwrap()
        };

        let mut history = vec![ChatMessage {
            role: "system".into(),
            content: Some(system_prompt.into()),
            reasoning_content: None,
            tool_calls: vec![],
        }];
        let mut step_outputs = Vec::new();
        let mut step_cached = Vec::new();
        let mut prev_prompt_len = 0usize;

        for (step, turn) in turns.iter().take(4).enumerate() {
            if let Some(user_text) = turn["user"].as_str() {
                history.push(ChatMessage {
                    role: "user".into(),
                    content: Some(user_text.into()),
                    reasoning_content: None,
                    tool_calls: vec![],
                });
            }
            if let Some(tools) = turn["tools"].as_array()
                && !tools.is_empty()
            {
                let calls_val: Vec<Value> = tools
                    .iter()
                    .map(|t| {
                        json!({
                            "function": {
                                "name": t["name"],
                                "arguments": t["arguments"]
                            }
                        })
                    })
                    .collect();
                let calls = parse_history_tools(Some(&Value::Array(calls_val))).unwrap();
                history.push(ChatMessage {
                    role: "assistant".into(),
                    content: None,
                    reasoning_content: None,
                    tool_calls: calls,
                });
                for t in tools {
                    history.push(ChatMessage {
                        role: "tool".into(),
                        content: Some(t["result"].as_str().unwrap_or("").into()),
                        reasoning_content: None,
                        tool_calls: vec![],
                    });
                }
            }

            let prompt_str = format_chatml_prompt_with_options(&history, None, true, true);
            let prompt_ids = encode(&prompt_str);
            let (out_ids, cached) = rt.block_on(async {
                let mut stream = worker
                    .submit(
                        prompt_ids.clone(),
                        SamplingParams {
                            max_tokens: 24,
                            ignore_eos: true,
                            temperature: 0.0,
                            ..SamplingParams::default()
                        },
                        None,
                    )
                    .unwrap();
                tokio::time::timeout(
                    Duration::from_secs(60),
                    collect_with_cached(&mut stream, 24),
                )
                .await
                .unwrap()
                .unwrap()
            });

            if step > 0 {
                let expected_min_cached = (prev_prompt_len / 64) * 64;
                assert!(
                    cached >= expected_min_cached,
                    "mode={kind:?} step={step}: expected cached >= {expected_min_cached} (prev_prompt_len={prev_prompt_len}), got cached={cached}"
                );
            }
            prev_prompt_len = prompt_ids.len();
            step_cached.push(cached);

            let raw_text = decode(&out_ids);
            let parsed = Qwen3Parser::parse_complete(&raw_text, true);
            history.push(ChatMessage {
                role: "assistant".into(),
                content: parsed.content,
                reasoning_content: parsed.reasoning_content,
                tool_calls: vec![],
            });
            step_outputs.push(out_ids);
        }

        // Also verify mid-turn cancellation preserves committed prefix for the follow-up turn.
        let mut cancel_history = history.clone();
        cancel_history.push(ChatMessage {
            role: "user".into(),
            content: Some("Now run pytest on the modified module and verify.".into()),
            reasoning_content: None,
            tool_calls: vec![],
        });
        let cancel_prompt_ids = encode(&format_chatml_prompt_with_options(
            &cancel_history,
            None,
            true,
            true,
        ));
        let post_cancel_cached = rt.block_on(async {
            let mut stream = worker
                .submit(
                    cancel_prompt_ids.clone(),
                    SamplingParams {
                        max_tokens: 48,
                        ignore_eos: true,
                        temperature: 0.0,
                        ..SamplingParams::default()
                    },
                    None,
                )
                .unwrap();
            // Collect 4 tokens then drop `stream` to trigger client cancellation.
            let mut got = 0usize;
            while let Some(ev) = stream.events.recv().await {
                if matches!(ev, GenerationEvent::Token { .. }) {
                    got += 1;
                    if got >= 4 {
                        break;
                    }
                }
            }
            drop(stream);

            // Follow-up request with the same prompt + extra user suffix must hit the cache.
            let mut follow_ids = cancel_prompt_ids.clone();
            follow_ids.extend_from_slice(&encode("\n Confirm exit code 0."));
            let mut follow_stream = worker
                .submit(
                    follow_ids,
                    SamplingParams {
                        max_tokens: 8,
                        ignore_eos: true,
                        temperature: 0.0,
                        ..SamplingParams::default()
                    },
                    None,
                )
                .unwrap();
            let (_, cached) = tokio::time::timeout(
                Duration::from_secs(60),
                collect_with_cached(&mut follow_stream, 8),
            )
            .await
            .unwrap()
            .unwrap();
            worker.shutdown().await.unwrap();
            cached
        });

        let expected_cancel_cached = ((cancel_prompt_ids.len() - 1) / 64) * 64;
        assert!(
            post_cancel_cached >= expected_cancel_cached,
            "mode={kind:?}: post-cancel follow-up must preserve committed prefix >= {expected_cancel_cached}, got {post_cancel_cached}"
        );

        (step_outputs, step_cached, post_cancel_cached)
    };

    let (plain_outs, plain_cached, _) = run_trajectory(None);
    let (mtp_outs, mtp_cached, _) = run_trajectory(Some(SpeculativeKind::Mtp));
    let (dflash_outs, dflash_cached, _) = run_trajectory(Some(SpeculativeKind::DFlash2));

    assert_eq!(
        mtp_outs, plain_outs,
        "MTP trajectory tokens must match Plain bit-for-bit"
    );
    assert_eq!(
        dflash_outs, plain_outs,
        "DFlash2 trajectory tokens must match Plain bit-for-bit"
    );
    assert_eq!(
        mtp_cached, plain_cached,
        "MTP cached prefix depths must match Plain on every trajectory turn"
    );
    assert_eq!(
        dflash_cached, plain_cached,
        "DFlash2 cached prefix depths must match Plain on every trajectory turn"
    );
}
