// The model module is CUDA-gated; exercise its pure layout without CUDA linkage.
#[path = "../src/model/scratch.rs"]
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

#[test]
fn qwen4_flash_next_hc_ple_qsa_layout_aligned_and_bounded() {
    for tp in [1, 2] {
        let d = Dimensions {
            q: 32 * 256 / tp,
            kv: 2 * 256 / tp,
            v_heads: 48 / tp,
            v: 48 * 128 / tp,
            k: 16 * 128 / tp,
            interm: 640 / tp,
            hidden: 2560,
            experts: 512,
            top_k: 10,
            moe: 640,
            shared: 640 / tp,
            hc_count: 4,
            uses_ple: true,
            indexer_head_dim: 128,
        };
        for tokens in [1, 4, 64, 512, 2048] {
            let (total, attn, gdn, moe, hc, qsa) =
                layer_layout(d, tokens).expect("qwen4_flash_next layer_layout");
            assert_eq!(total % 256, 0, "total layer scratch must be 256B-aligned");
            for off in attn
                .into_iter()
                .chain(gdn)
                .chain(moe.into_iter().flatten())
                .chain(hc)
                .chain(qsa)
            {
                assert_eq!(off % 256, 0, "scratch offset {off} must be 256B-aligned");
                assert!(off < total, "scratch offset {off} must lie within {total}");
            }
        }
        let arena = 512 * 1024 * 1024;
        let vocab = 248320 / tp;
        let cap = token_capacity(arena, d.hidden, vocab, 16, |t| {
            layer_layout(d, t).map(|l| l.0)
        })
        .expect("qwen4_flash_next token_capacity");
        assert!((64..=2048).contains(&cap) && cap % 64 == 0);
    }
}
