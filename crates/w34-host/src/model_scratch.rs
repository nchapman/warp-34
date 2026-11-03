//! Pure byte layout for the mutually exclusive FA, GDN and FFN phases.
//! No device allocation or arena rewind: reserve `bytes` once, then relocate offsets.

pub(super) const ALIGN: usize = 256;

/// Model dimensions needed by the shared device/capacity layout.
#[derive(Clone, Copy)]
pub(super) struct Dimensions {
    pub q: usize,
    pub kv: usize,
    pub v_heads: usize,
    pub v: usize,
    pub k: usize,
    pub interm: usize,
    pub experts: usize,
    pub top_k: usize,
    pub moe: usize,
    pub shared: usize,
    pub hidden: usize,
    pub hc_count: usize,
    pub uses_ple: bool,
    pub indexer_head_dim: usize,
}

/// High-water byte count and FA, GDN, FFN, PLE/HC, QSA offsets within the shared reservation.
pub(super) type LayerLayout = (
    usize,
    [usize; 6],
    [usize; 5],
    [Option<usize>; 10],
    [usize; 3],
    [usize; 7],
);

#[allow(clippy::cognitive_complexity)]
pub(super) fn layer_layout(d: Dimensions, tokens: usize) -> Option<LayerLayout> {
    let conv = d.k.checked_mul(2)?.checked_add(d.v)?;
    let mut fa = Phase::default();
    let mut gdn = Phase::default();
    let mut ffn = Phase::default();
    let mut ple_hc = Phase::default();
    let fa_offsets = [
        fa.alloc(&[
            tokens,
            d.q.checked_mul(2)?.checked_add(d.kv.checked_mul(2)?)?,
            2,
        ])?,
        fa.alloc(&[tokens, d.q, 2])?,
        fa.alloc(&[tokens, d.kv, 2])?,
        fa.alloc(&[tokens, d.q, 2])?,
        fa.alloc(&[tokens, d.q, 2])?,
        fa.alloc(&[tokens, d.q, 2])?,
    ];
    let qsa_offsets = if d.indexer_head_dim > 0 {
        [
            fa.alloc(&[tokens, d.indexer_head_dim.checked_mul(5)?, 2])?,
            fa.alloc(&[tokens, d.indexer_head_dim.checked_mul(4)?, 2])?,
            fa.alloc(&[tokens, d.indexer_head_dim, 2])?,
            fa.alloc(&[tokens, 513, 4])?,
            fa.alloc(&[tokens, 513, 4])?,
            fa.alloc(&[tokens, 4])?,
            fa.alloc(&[tokens, 4])?,
        ]
    } else {
        [0; 7]
    };
    let hc_mult = d.hc_count.max(1);
    let ple_hc_offsets = if d.hc_count > 1 || d.uses_ple {
        [
            ple_hc.alloc(&[tokens, d.hidden.checked_mul(hc_mult)?, 2])?,
            ple_hc.alloc(&[tokens, d.hidden.checked_mul(hc_mult)?, 2])?,
            ple_hc.alloc(&[tokens, d.hidden, 2])?,
        ]
    } else {
        [0; 3]
    };
    let gdn_offsets = [
        gdn.alloc(&[tokens, conv.checked_add(d.v)?, 2])?,
        gdn.alloc(&[tokens, 2, d.v_heads, 2])?,
        gdn.alloc(&[tokens, conv, 2])?,
        gdn.alloc(&[tokens, d.v, 2])?,
        gdn.alloc(&[tokens, d.v, 2])?,
    ];
    if tokens >= 256 && d.v == d.v_heads.saturating_mul(128) && d.k.is_multiple_of(128) {
        // Reserve split-WY workspace tail (including cached K_norm on 9B TP=1 and 27B TP=2;
        // on 27B TP=1 `v_heads == 48`, clamp to `ffn`/`gdn` envelope so 262K single-card budget
        // adds 0 bytes).
        let bytes_per_tok_vh = if d.v_heads >= 48 { 800 } else { 1056 };
        let wy_tail = gdn_offsets[4].checked_add(
            tokens
                .checked_mul(d.v_heads)?
                .checked_mul(bytes_per_tok_vh)?,
        )?;
        if wy_tail > gdn.bytes {
            gdn.bytes = wy_tail.checked_add(ALIGN - 1)? & !(ALIGN - 1);
        }
    }
    let mut offsets = [None; 10];
    if d.interm > 0 {
        offsets[0] = Some(ffn.alloc(&[tokens, d.interm, 4])?);
        offsets[1] = Some(ffn.alloc(&[tokens, d.interm, 2])?);
    }
    if d.experts > 0 {
        offsets[2] = Some(ffn.alloc(&[tokens, d.experts, 2])?);
        offsets[3] = Some(ffn.alloc(&[tokens, d.top_k.max(1), 4])?);
        offsets[4] = Some(ffn.alloc(&[tokens, d.top_k.max(1), 4])?);
        offsets[5] = Some(ffn.alloc(&[tokens, d.top_k.max(1), d.moe.max(8), 2])?);
        offsets[6] = Some(ffn.alloc(&[tokens, d.shared.max(8), 4])?);
        offsets[7] = Some(ffn.alloc(&[tokens, d.shared.max(8), 2])?);
        offsets[8] = Some(ffn.alloc(&[tokens, d.hidden, 2])?);
        offsets[9] = Some(ffn.alloc(&[tokens, 2])?);
    }
    Some((
        overlay_bytes(&[fa, gdn, ffn, ple_hc]),
        fa_offsets,
        gdn_offsets,
        offsets,
        ple_hc_offsets,
        qsa_offsets,
    ))
}

#[derive(Default, Debug)]
pub(super) struct Phase {
    pub bytes: usize,
}

impl Phase {
    pub fn alloc(&mut self, factors: &[usize]) -> Option<usize> {
        let size = factors.iter().try_fold(1usize, |n, &v| n.checked_mul(v))?;
        let offset = self.bytes.checked_add(ALIGN - 1)? & !(ALIGN - 1);
        // Round the end too, so the reservation's following workspace is aligned.
        let end = offset.checked_add(size)?.checked_add(ALIGN - 1)? & !(ALIGN - 1);
        self.bytes = end;
        Some(offset)
    }
}

pub(super) fn overlay_bytes(phases: &[Phase]) -> usize {
    phases.iter().map(|p| p.bytes).max().unwrap_or(0)
}

/// Full ordinary-forward reservation. Selected rows cannot exceed either the
/// sequence limit or the token count. Always budget noncontiguous gather and
/// all four greedy/top1 arrays, even when a fast path avoids them.
pub(super) fn required_bytes(
    tokens: usize,
    hidden: usize,
    vocab: usize,
    rows: usize,
    overlay: usize,
) -> Option<usize> {
    let mut p = Phase::default();
    for _ in 0..4 {
        p.alloc(&[tokens, hidden, 2])?;
    }
    p.alloc(&[overlay])?;
    let rows = rows.min(tokens).max(1);
    p.alloc(&[rows, vocab, 2])?;
    p.alloc(&[rows, hidden, 2])?;
    for _ in 0..4 {
        p.alloc(&[rows, 4])?;
    }
    p.alloc(&[8 * 1024 * 1024])?;
    // Extra alignment margin beyond the rounded individual allocations.
    p.bytes.checked_add(ALIGN)
}

pub(super) fn token_capacity(
    arena: usize,
    hidden: usize,
    vocab: usize,
    rows: usize,
    mut layout: impl FnMut(usize) -> Option<usize>,
) -> Option<usize> {
    (1..=32).rev().map(|n| n * 64).find(|&tokens| {
        layout(tokens)
            .and_then(|overlay| required_bytes(tokens, hidden, vocab, rows, overlay))
            .is_some_and(|bytes| bytes <= arena)
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn aligned_disjoint_within_phase_overlapping_between_phases() {
        let mut fa = Phase::default();
        let qkv = fa.alloc(&[3, 257, 2]).unwrap();
        let q = fa.alloc(&[3, 128, 2]).unwrap();
        assert_eq!(qkv, 0);
        assert_eq!(q % ALIGN, 0);
        assert!(q >= qkv + 3 * 257 * 2);
        let mut gdn = Phase::default();
        let qkvz = gdn.alloc(&[3, 513, 2]).unwrap();
        let ba = gdn.alloc(&[3, 16, 2]).unwrap();
        assert_eq!(qkvz, qkv);
        assert!(ba >= qkvz + 3 * 513 * 2); // ba is live concurrently on aux stream
        let mut ffn = Phase::default();
        assert_eq!(ffn.alloc(&[3, 2048, 4]), Some(0));
        let ends = [fa.bytes, gdn.bytes, ffn.bytes];
        let bytes = overlay_bytes(&[fa, gdn, ffn]);
        assert_eq!(bytes, *ends.iter().max().unwrap());
        assert!(ends.iter().all(|&end| end <= bytes));
        assert_eq!(bytes % ALIGN, 0);
        // Persistent buffers precede/follow this reservation, never any phase's end.
        let base = 4 * ALIGN;
        let persistent_after = base + bytes;
        assert!(ends.iter().all(|&end| base + end <= persistent_after));
    }

    #[test]
    fn single_stream_ple_has_disjoint_scratch() {
        let d = Dimensions {
            q: 128,
            kv: 128,
            v_heads: 1,
            v: 128,
            k: 128,
            interm: 256,
            experts: 0,
            top_k: 0,
            moe: 0,
            shared: 0,
            hidden: 128,
            hc_count: 1,
            uses_ple: true,
            indexer_head_dim: 0,
        };
        let (bytes, _, _, _, [a, b, value], _) = layer_layout(d, 2).unwrap();
        assert!(b >= a + 2 * 128 * 2);
        assert!(value >= b + 2 * 128 * 2);
        assert!(bytes >= value + 2 * 128 * 2);
    }

    #[test]
    fn boundaries_and_empty() {
        let mut p = Phase::default();
        assert_eq!(p.alloc(&[0]), Some(0));
        assert_eq!(p.alloc(&[256]), Some(0));
        assert_eq!(p.alloc(&[1]), Some(256));
        assert_eq!(p.bytes, 512);
        assert_eq!(overlay_bytes(&[]), 0);
    }

    #[test]
    fn overflow_is_rejected_without_mutation() {
        let mut p = Phase::default();
        assert_eq!(p.alloc(&[usize::MAX, 2]), None);
        assert_eq!(p.alloc(&[usize::MAX]), None);
        assert_eq!(p.bytes, 0);
        p.bytes = usize::MAX - 255;
        assert_eq!(p.alloc(&[256]), None);
        assert_eq!(p.bytes, usize::MAX - 255);
        p.bytes = usize::MAX;
        assert_eq!(p.alloc(&[0]), None);
    }
}
