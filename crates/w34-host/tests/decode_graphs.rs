//! Padded multi-request decode graph replay versus eager execution.
#![cfg(feature = "cuda")]
mod common;

use std::sync::Mutex;

use w34_host::{
    cache::KvFormatPolicy,
    model::{EngineBatchSlice, Qwen3_5Engine},
};

static GPU_LOCK: Mutex<()> = Mutex::new(());

#[test]
fn decode_buckets_replay_mutated_requests_match_eager() {
    if common::require_or_skip_cached_model(
        "Qwen/Qwen3.5-4B",
        "decode_buckets_replay_mutated_requests_match_eager",
    )
    .is_none()
    {
        return;
    }
    let _guard = GPU_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    for tp in [1, 2] {
        let mut engine =
            Qwen3_5Engine::load("Qwen/Qwen3.5-4B", tp, KvFormatPolicy::Tq4, Some(64), true)
                .unwrap();
        // Separate pages and state slots for each request, including padded rows.
        let pages: Vec<_> = (0..16).map(|_| engine.alloc_kv_page().unwrap()).collect();
        let slots: Vec<_> = (0..16)
            .map(|_| engine.alloc_state_slot().unwrap())
            .collect();
        for n in [1usize, 2, 3, 5, 9, 16, 7, 4, 2, 1] {
            for reverse in [false, true] {
                let ids: Vec<_> = (0..n).map(|i| 100 + (i * 13) as i32).collect();
                let order: Vec<_> = (0..n)
                    .map(|i| if reverse { n - i - 1 } else { i })
                    .collect();
                let tables: Vec<_> = order.iter().map(|&i| vec![pages[i]]).collect();
                let slices: Vec<_> = (0..n)
                    .map(|i| EngineBatchSlice {
                        tokens: &ids[i..i + 1],
                        ctx_len_before: 0,
                        block_table: &tables[i],
                        state_in_slot: slots[order[i]],
                        state_out_slot: slots[order[i]],
                        has_initial_state: false,
                        spec_slots: &[],
                        checkpoints: &[],
                        emit_all_query_tokens: false,
                    })
                    .collect();
                engine.use_cuda_graph = false;
                let eager = engine.step_batch_slices(&slices).unwrap();
                let eager_states: Vec<_> = engine
                    .ranks
                    .iter()
                    .map(|r| {
                        let mut bits = vec![0u32; r.state_pool.layers[0].ssm_state.len_bytes / 4];
                        r.state_pool.layers[0]
                            .ssm_state
                            .copy_to_host(&mut bits, &r.stream)
                            .unwrap();
                        bits
                    })
                    .collect();
                engine.use_cuda_graph = true;
                if tp == 2 && n == 1 {
                    // Inject a rank-1 host upload failure on cold capture and warm
                    // replay. Neither rank may launch or mutate live state.
                    let bytes = engine.ranks[1].batched_decode_io.meta_i32.len_bytes;
                    engine.ranks[1].batched_decode_io.meta_i32.len_bytes = 0;
                    let failure = engine.step_batch_slices(&slices);
                    engine.ranks[1].batched_decode_io.meta_i32.len_bytes = bytes;
                    assert!(failure.is_err(), "rank-1 upload failure was swallowed");
                    assert!(
                        engine.use_cuda_graph,
                        "unrelated graph families were disabled"
                    );
                    for (r, expected) in engine.ranks.iter().zip(&eager_states) {
                        let mut bits = vec![0u32; expected.len()];
                        r.state_pool.layers[0]
                            .ssm_state
                            .copy_to_host(&mut bits, &r.stream)
                            .unwrap();
                        assert_eq!(&bits, expected, "failed preparation mutated state");
                    }
                }
                let graph = engine.step_batch_slices(&slices).unwrap();
                assert!(engine.use_cuda_graph, "capture must not silently fall back");
                assert_eq!(graph, eager, "tp={tp}, n={n}, reverse={reverse}");
                for (r, expected) in engine.ranks.iter().zip(&eager_states) {
                    let mut bits = vec![0u32; expected.len()];
                    r.state_pool.layers[0]
                        .ssm_state
                        .copy_to_host(&mut bits, &r.stream)
                        .unwrap();
                    assert_eq!(&bits, expected, "state mutation tp={tp} n={n}");
                }
            }
        }
        assert_eq!(
            engine.captured_decode_graph_keys(),
            vec![(1, false), (2, false), (4, false), (8, false), (16, false)]
        );
    }
}

#[test]
fn pipelined_fast_greedy_burst_matches_single_and_handles_staggered_finish_and_chunk_boundary() {
    let _guard = GPU_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    use std::sync::atomic::Ordering;
    use w34_host::{
        engine_worker::GenerationEvent, parser::SamplingParams, server::OpenAiServerState,
    };

    let prompts: Vec<Vec<i32>> = vec![
        vec![785, 6722, 315, 9625, 374],
        vec![198, 2610, 525, 264, 10950],
        vec![3838, 374, 220, 17, 10, 17],
        vec![760, 1143, 279, 810, 419],
    ];
    let lengths = [12usize, 70, 25, 70];
    if common::require_or_skip_cached_model(
        "Qwen/Qwen3.5-4B",
        "pipelined_fast_greedy_burst_matches_single_and_handles_staggered_finish_and_chunk_boundary",
    )
    .is_none()
    {
        return;
    }

    let mut ref_engine =
        Qwen3_5Engine::load("Qwen/Qwen3.5-4B", 1, KvFormatPolicy::Fp16, Some(128), false).unwrap();
    let expected: Vec<Vec<i32>> = prompts
        .iter()
        .zip(lengths)
        .map(|(p, len)| {
            ref_engine
                .generate_tokens(p, len, false)
                .unwrap()
                .generated_token_ids
        })
        .collect();
    drop(ref_engine);

    let state = OpenAiServerState::from_loader("pipelined-fast-greedy".into(), false, || {
        Qwen3_5Engine::load("Qwen/Qwen3.5-4B", 1, KvFormatPolicy::Fp16, Some(128), true)
            .map_err(|e| e.to_string())
    })
    .unwrap();

    tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .unwrap()
        .block_on(async {
            let mut streams = Vec::new();
            for (p, len) in prompts.iter().zip(lengths) {
                streams.push(
                    state
                        .worker
                        .submit(
                            p.clone(),
                            SamplingParams {
                                max_tokens: len,
                                temperature: 0.0,
                                ignore_eos: true,
                                logprobs: true,
                                top_logprobs: 0,
                                ..SamplingParams::default()
                            },
                            None,
                        )
                        .unwrap(),
                );
            }

            let mut tasks = Vec::new();
            for mut s in streams {
                tasks.push(tokio::spawn(async move {
                    let mut ids = Vec::new();
                    while let Some(ev) = s.events.recv().await {
                        match ev {
                            GenerationEvent::Token { id, logprobs } => {
                                let lp = logprobs.expect("fast greedy must emit logprobs");
                                assert_eq!(lp.token_id, id);
                                assert!(lp.logprob.is_finite());
                                assert!(lp.top_logprobs.is_empty());
                                ids.push(id);
                            }
                            GenerationEvent::Finished {
                                generated_tokens, ..
                            } => {
                                assert_eq!(generated_tokens, ids.len());
                                break;
                            }
                            GenerationEvent::Error(e) => panic!("unexpected worker error: {e}"),
                        }
                    }
                    ids
                }));
            }

            for (i, t) in tasks.into_iter().enumerate() {
                let got = t.await.unwrap();
                assert_eq!(got, expected[i], "pipelined burst mismatch on req {i}");
            }
            assert_eq!(state.worker.stats.active.load(Ordering::Relaxed), 0);
            assert_eq!(state.worker.stats.completed.load(Ordering::Relaxed), 4);
            state.worker.shutdown().await.unwrap();
        });
}
