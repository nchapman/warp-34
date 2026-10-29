//! Padded multi-request decode graph replay versus eager execution.
#![cfg(feature = "cuda")]

use w34_host::{
    cache::KvFormatPolicy,
    model::{EngineBatchSlice, Qwen3_5Engine},
};

#[test]
fn decode_buckets_replay_mutated_requests_match_eager() {
    for tp in [1, 2] {
        let mut engine = Qwen3_5Engine::load(
            "nchapman/Qwen3.5-4B-W4A16-GPTQ",
            tp,
            KvFormatPolicy::Tq4,
            Some(64),
            true,
        )
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
