// The model module is CUDA-gated; exercise its pure layout without CUDA linkage.
#[path = "../src/model_scratch.rs"]
mod model_scratch;

use model_scratch::{Dimensions, layer_layout, required_bytes, token_capacity};

#[test]
fn model_bounds_and_maximality() {
    // Qwen3.5 dense 9B / 27B dimensions, plus a routed high-temporary case.
    for (hidden, heads, interm, value_heads) in [(4096, 16, 12288, 32), (5120, 24, 17408, 48)] {
        for tp in [1, 2] {
            let dense = Dimensions {
                q: heads * 256 / tp,
                kv: 4 * 256 / tp,
                v_heads: value_heads / tp,
                v: value_heads * 128 / tp,
                k: 16 * 128 / tp,
                interm: interm / tp,
                hidden,
                experts: 0,
                top_k: 0,
                moe: 0,
                shared: 0,
                hc_count: 1,
                uses_ple: false,
                indexer_head_dim: 0,
            };
            for d in [
                dense,
                Dimensions {
                    experts: 256,
                    top_k: 8,
                    moe: 32768 / tp,
                    shared: 4096 / tp,
                    ..dense
                },
            ] {
                for mib in [336, 384, 512] {
                    for rows in [1, 16, 128] {
                        let arena = mib * 1024 * 1024;
                        let vocab = 248320 / tp;
                        let layout = |t| layer_layout(d, t).map(|l| l.0);
                        let cap = token_capacity(arena, hidden, vocab, rows, layout).unwrap();
                        assert!(cap <= 2048 && cap % 64 == 0);
                        let bytes =
                            |t| required_bytes(t, hidden, vocab, rows, layout(t).unwrap()).unwrap();
                        assert!(bytes(cap) <= arena);
                        assert!(cap == 2048 || bytes(cap + 64) > arena);
                        // The selected-output reservation must reduce available space.
                        assert!(
                            bytes(cap)
                                >= 4 * cap * hidden * 2
                                    + layout(cap).unwrap()
                                    + rows.min(cap) * (vocab * 2 + hidden * 2 + 16)
                                    + 8 * 1024 * 1024
                        );
                    }
                }
            }
        }
    }
}

#[test]
fn insufficient_capacity_and_overflow_fail_closed() {
    assert_eq!(
        token_capacity(8 * 1024 * 1024, 4096, 248320, 16, |_| Some(0)),
        None
    );
    assert_eq!(
        token_capacity(usize::MAX, usize::MAX, 1, 1, |_| Some(0)),
        None
    );
    assert_eq!(token_capacity(usize::MAX, 1, 1, 1, |_| None), None);
}
