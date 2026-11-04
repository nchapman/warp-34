//! Standalone U4B8 CPU arithmetic. No routing, reduction across experts, or ownership.
//! Packed rows contain eight consecutive unsigned nibbles per u32 (low nibble first),
//! with one IEEE binary16 scale per 128 columns. Gate rows precede up rows.
//! All outputs are token-major f32; slice outputs are partial down projections.

use std::ops::Range;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum EvalError {
    InvalidLayout,
    InvalidShape,
    UnsupportedCpu,
}

/// Borrowed storage for exactly one expert. Extra trailing storage is permitted.
#[derive(Clone, Copy)]
pub(super) struct ExpertLayout<'a> {
    pub hidden: usize,
    pub intermediate: usize,
    pub gate_up_words: &'a [u32],
    pub gate_up_scales: &'a [u16],
    pub down_words: &'a [u32],
    pub down_scales: &'a [u16],
}

pub(super) fn supported() -> bool {
    #[cfg(target_arch = "x86_64")]
    {
        std::is_x86_feature_detected!("avx2")
            && std::is_x86_feature_detected!("fma")
            && std::is_x86_feature_detected!("f16c")
            && std::is_x86_feature_detected!("avx512f")
            && std::is_x86_feature_detected!("avx512vl")
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        false
    }
}

/// Transpose each group from [16 words][8 nibbles] to [8][16]. The group
/// sum is deliberately sequential in the ORIGINAL input order, not a SIMD reduction.
/// This preparation is portable; evaluation requires `supported()`.
pub(super) fn prepare_x_perm(x: &[f32], xp: &mut [f32], xg: &mut [f32]) -> Result<(), EvalError> {
    if !x.len().is_multiple_of(128) || xp.len() < x.len() || xg.len() < x.len() / 128 {
        return Err(EvalError::InvalidShape);
    }
    for (g, src) in x.as_chunks::<128>().0.iter().enumerate() {
        let mut sum = 0.0f32;
        for l in 0..16 {
            for b in 0..8 {
                let v = src[l * 8 + b];
                xp[g * 128 + b * 16 + l] = v;
                sum += v;
            }
        }
        xg[g] = sum;
    }
    Ok(())
}

/// Prepare FP16 input directly, without allocating a full FP32 prompt copy.
/// Preserve the same sequential group-sum order as `prepare_x_perm`.
pub(super) fn prepare_x_perm_f16(
    x: &[u16],
    xp: &mut [f32],
    xg: &mut [f32],
) -> Result<(), EvalError> {
    if !x.len().is_multiple_of(128) || xp.len() < x.len() || xg.len() < x.len() / 128 {
        return Err(EvalError::InvalidShape);
    }
    for (g, src) in x.as_chunks::<128>().0.iter().enumerate() {
        let mut sum = 0.0f32;
        for l in 0..16 {
            for b in 0..8 {
                let v = crate::cuda::f16_to_f32(src[l * 8 + b]);
                xp[g * 128 + b * 16 + l] = v;
                sum += v;
            }
        }
        xg[g] = sum;
    }
    Ok(())
}

/// Required caller-owned scratch in f32 elements: act, permuted act, group sums.
pub(super) fn scratch_len(tokens: usize, slice_groups: usize) -> Option<usize> {
    tokens.checked_mul(slice_groups)?.checked_mul(257)
}

#[allow(clippy::too_many_arguments)]
fn validate(
    d: ExpertLayout<'_>,
    groups: &Range<usize>,
    xp: &[f32],
    xg: &[f32],
    wt: &[f32],
    out: &[f32],
    scratch: &[f32],
    max_tokens: usize,
) -> Result<(), EvalError> {
    let h = d.hidden;
    let i = d.intermediate;
    if h == 0 || i == 0 || !h.is_multiple_of(128) || !i.is_multiple_of(128) {
        return Err(EvalError::InvalidLayout);
    }
    let gu = i
        .checked_mul(2)
        .and_then(|v| v.checked_mul(h / 8))
        .ok_or(EvalError::InvalidLayout)?;
    let dn = h.checked_mul(i / 8).ok_or(EvalError::InvalidLayout)?;
    if d.gate_up_words.len() < gu
        || d.gate_up_scales.len() < gu / 16
        || d.down_words.len() < dn
        || d.down_scales.len() < dn / 16
    {
        return Err(EvalError::InvalidLayout);
    }
    let t = wt.len();
    if t == 0 || t > max_tokens || groups.start >= groups.end || groups.end > i / 128 {
        return Err(EvalError::InvalidShape);
    }
    let nh = t.checked_mul(h).ok_or(EvalError::InvalidShape)?;
    let ns = scratch_len(t, groups.end - groups.start).ok_or(EvalError::InvalidShape)?;
    if xp.len() < nh || xg.len() < nh / 128 || out.len() < nh || scratch.len() < ns {
        return Err(EvalError::InvalidShape);
    }
    if !supported() {
        return Err(EvalError::UnsupportedCpu);
    }
    Ok(())
}

/// Evaluate 1..=4 prepared tokens over a nonempty intermediate GROUP range.
/// `xp`/`xg` have strides hidden and hidden/128; `weights.len()` is the token count.
/// Writes (does not add) weights[t] * down(SiLU(gate) * up) into out[t*hidden..].
/// Scratch must contain at least `scratch_len(weights.len(), groups.len())` floats.
pub(super) fn eval_expert_slice(
    d: ExpertLayout<'_>,
    groups: Range<usize>,
    xp: &[f32],
    xg: &[f32],
    weights: &[f32],
    out: &mut [f32],
    scratch: &mut [f32],
) -> Result<(), EvalError> {
    validate(d, &groups, xp, xg, weights, out, scratch, 4)?;
    #[cfg(target_arch = "x86_64")]
    // SAFETY: runtime ISA guard and all sizes/ranges checked above.
    unsafe {
        match weights.len() {
            1 => simd::expert::<1>(d, groups, xp, xg, weights, out, scratch, false),
            2 => simd::expert::<2>(d, groups, xp, xg, weights, out, scratch, false),
            3 => simd::expert::<3>(d, groups, xp, xg, weights, out, scratch, false),
            4 => simd::expert::<4>(d, groups, xp, xg, weights, out, scratch, false),
            _ => unreachable!(),
        }
    }
    Ok(())
}

/// Full expert, 1..=96 prepared tokens. Same layout and output contract as slice.
/// Four-row tiles unpack each weight group once for all tokens (not token microbatches).
pub(super) fn eval_expert_prefill(
    d: ExpertLayout<'_>,
    xp: &[f32],
    xg: &[f32],
    weights: &[f32],
    out: &mut [f32],
    scratch: &mut [f32],
) -> Result<(), EvalError> {
    let groups = 0..d.intermediate / 128;
    validate(d, &groups, xp, xg, weights, out, scratch, 96)?;
    #[cfg(target_arch = "x86_64")]
    // SAFETY: runtime ISA guard and all sizes/ranges checked above.
    unsafe {
        match weights.len() {
            1 => simd::expert::<1>(d, groups, xp, xg, weights, out, scratch, true),
            2..=8 => simd::expert::<8>(d, groups, xp, xg, weights, out, scratch, true),
            9..=16 => simd::expert::<16>(d, groups, xp, xg, weights, out, scratch, true),
            17..=32 => simd::expert::<32>(d, groups, xp, xg, weights, out, scratch, true),
            _ => simd::expert::<96>(d, groups, xp, xg, weights, out, scratch, true),
        }
    }
    Ok(())
}

#[cfg(target_arch = "x86_64")]
#[allow(unsafe_op_in_unsafe_fn)]
mod simd {
    use super::*;
    use std::arch::x86_64::*;

    #[target_feature(enable = "avx2,fma,f16c,avx512f,avx512vl")]
    unsafe fn hsum(v: __m512) -> f32 {
        let s = _mm256_add_ps(
            _mm512_castps512_ps256(v),
            _mm256_castpd_ps(_mm512_extractf64x4_pd::<1>(_mm512_castps_pd(v))),
        );
        let s = _mm_add_ps(_mm256_castps256_ps128(s), _mm256_extractf128_ps::<1>(s));
        let s = _mm_hadd_ps(s, s);
        _mm_cvtss_f32(_mm_hadd_ps(s, s))
    }

    // R=1/2/4 is the row/two-row/four-row kernel. T>=8 uses the original
    // prefill group -> token -> nibble schedule, keeping only R live accumulators.
    // Small T (1..=4) uses nibble -> token, sharing decoded weights in registers.
    #[target_feature(enable = "avx2,fma,f16c,avx512f,avx512vl")]
    // Explicit row/token/nibble loops preserve the donor's SIMD evaluation order.
    #[allow(clippy::cognitive_complexity, clippy::needless_range_loop)]
    unsafe fn rows<const R: usize, const T: usize>(
        w: [&[u32]; R],
        s: [&[u16]; R],
        xp: &[f32],
        xg: &[f32],
        groups: usize,
        n: usize,
    ) -> [[f32; T]; R] {
        let z = _mm512_setzero_ps();
        let mut total = [[z; T]; R];
        let mut bias = [[0.0f32; T]; R];
        for g in 0..groups {
            let mut q = [_mm512_setzero_si512(); R];
            for r in 0..R {
                q[r] = _mm512_loadu_si512(w[r].as_ptr().add(g * 16).cast());
                if (g + 4) * 16 < w[r].len() {
                    _mm_prefetch::<_MM_HINT_T0>(w[r].as_ptr().add((g + 4) * 16).cast());
                }
            }
            let mut scale = [0.0; R];
            for r in 0..R {
                scale[r] = _mm_cvtss_f32(_mm_cvtph_ps(_mm_cvtsi32_si128(
                    *s[r].get_unchecked(g) as i32
                )));
            }
            if T >= 8 {
                let mut unpack = [[z; 8]; R];
                macro_rules! unpack {
                    ($b:literal) => {
                        for r in 0..R {
                            unpack[r][$b] = _mm512_cvtepi32_ps(_mm512_and_si512(
                                _mm512_srli_epi32::<{ 4 * $b }>(q[r]),
                                _mm512_set1_epi32(15),
                            ));
                        }
                    };
                }
                unpack!(0);
                unpack!(1);
                unpack!(2);
                unpack!(3);
                unpack!(4);
                unpack!(5);
                unpack!(6);
                unpack!(7);
                for t in 0..n {
                    let mut acc = [z; R];
                    for b in 0..8 {
                        let x =
                            _mm512_loadu_ps(xp.as_ptr().add(t * groups * 128 + g * 128 + b * 16));
                        for r in 0..R {
                            acc[r] = _mm512_fmadd_ps(unpack[r][b], x, acc[r]);
                        }
                    }
                    for r in 0..R {
                        total[r][t] =
                            _mm512_fmadd_ps(acc[r], _mm512_set1_ps(scale[r]), total[r][t]);
                        // Match the original host compiler's contracted scalar bias update.
                        bias[r][t] =
                            scale[r].mul_add(*xg.get_unchecked(t * groups + g), bias[r][t]);
                    }
                }
            } else {
                let mut acc = [[z; T]; R];
                macro_rules! nib {
                    ($b:literal) => {{
                        let mut ws = [z; R];
                        for r in 0..R {
                            ws[r] = _mm512_cvtepi32_ps(_mm512_and_si512(
                                _mm512_srli_epi32::<{ 4 * $b }>(q[r]),
                                _mm512_set1_epi32(15),
                            ));
                        }
                        for t in 0..T {
                            let x = _mm512_loadu_ps(
                                xp.as_ptr().add(t * groups * 128 + g * 128 + 16 * $b),
                            );
                            for r in 0..R {
                                acc[r][t] = _mm512_fmadd_ps(ws[r], x, acc[r][t]);
                            }
                        }
                    }};
                }
                nib!(0);
                nib!(1);
                nib!(2);
                nib!(3);
                nib!(4);
                nib!(5);
                nib!(6);
                nib!(7);
                for r in 0..R {
                    for t in 0..T {
                        total[r][t] =
                            _mm512_fmadd_ps(acc[r][t], _mm512_set1_ps(scale[r]), total[r][t]);
                        // Match the original host compiler's contracted scalar bias update.
                        bias[r][t] =
                            scale[r].mul_add(*xg.get_unchecked(t * groups + g), bias[r][t]);
                    }
                }
            }
        }
        let mut out = [[0.0; T]; R];
        for r in 0..R {
            for t in 0..n {
                out[r][t] = hsum(total[r][t]) - 8.0 * bias[r][t];
            }
        }
        out
    }

    #[target_feature(enable = "avx2,fma,f16c,avx512f,avx512vl")]
    #[allow(clippy::too_many_arguments)]
    pub(super) unsafe fn expert<const T: usize>(
        d: ExpertLayout<'_>,
        range: Range<usize>,
        xp: &[f32],
        xg: &[f32],
        wt: &[f32],
        out: &mut [f32],
        scratch: &mut [f32],
        tile: bool,
    ) {
        let h = d.hidden;
        let i = d.intermediate;
        let n = wt.len();
        let sg = range.end - range.start;
        let cols = sg * 128;
        let (act, rest) = scratch.split_at_mut(n * cols);
        let (ap, ag) = rest.split_at_mut(n * cols);
        let gw = |r: usize| &d.gate_up_words[r * (h / 8)..];
        let gs = |r: usize| &d.gate_up_scales[r * (h / 128)..];
        let mut c = 0;
        while c < cols {
            let r = range.start * 128 + c;
            if tile || T == 1 {
                let v = rows::<4, T>(
                    [gw(r), gw(r + 1), gw(i + r), gw(i + r + 1)],
                    [gs(r), gs(r + 1), gs(i + r), gs(i + r + 1)],
                    xp,
                    xg,
                    h / 128,
                    n,
                );
                for t in 0..n {
                    for j in 0..2 {
                        let g = v[j][t];
                        // f32::exp, NOT an f64 exp or approximate SIMD exponential.
                        act[t * cols + c + j] = (g / (1.0f32 + (-g).exp())) * v[j + 2][t];
                    }
                }
                c += 2;
            } else {
                let v = rows::<2, T>([gw(r), gw(i + r)], [gs(r), gs(i + r)], xp, xg, h / 128, n);
                for t in 0..n {
                    let g = v[0][t];
                    act[t * cols + c] = (g / (1.0f32 + (-g).exp())) * v[1][t];
                }
                c += 1;
            }
        }
        for t in 0..n {
            prepare_x_perm(
                &act[t * cols..(t + 1) * cols],
                &mut ap[t * cols..(t + 1) * cols],
                &mut ag[t * sg..(t + 1) * sg],
            )
            .unwrap();
        }
        let dw = |r: usize| &d.down_words[r * (i / 8) + range.start * 16..];
        let ds = |r: usize| &d.down_scales[r * (i / 128) + range.start..];
        let mut r = 0;
        if tile || T == 1 {
            while r + 4 <= h {
                let v = rows::<4, T>(
                    [dw(r), dw(r + 1), dw(r + 2), dw(r + 3)],
                    [ds(r), ds(r + 1), ds(r + 2), ds(r + 3)],
                    ap,
                    ag,
                    sg,
                    n,
                );
                for t in 0..n {
                    for j in 0..4 {
                        out[t * h + r + j] = wt[t] * v[j][t];
                    }
                }
                r += 4;
            }
        }
        while r + 2 <= h {
            let v = rows::<2, T>([dw(r), dw(r + 1)], [ds(r), ds(r + 1)], ap, ag, sg, n);
            for t in 0..n {
                for j in 0..2 {
                    out[t * h + r + j] = wt[t] * v[j][t];
                }
            }
            r += 2;
        }
        while r < h {
            let v = rows::<1, T>([dw(r)], [ds(r)], ap, ag, sg, n);
            for t in 0..n {
                out[t * h + r] = wt[t] * v[0][t];
            }
            r += 1;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn permutation_and_sequential_sum() {
        let mut x = [1.0f32; 256];
        x[0] = 1e20;
        x[1] = -1e20;
        let mut xp = [0.0; 256];
        let mut xg = [0.0; 2];
        prepare_x_perm(&x, &mut xp, &mut xg).unwrap();
        assert_eq!(xg, [126.0, 128.0]);
        for g in 0..2 {
            for l in 0..16 {
                for b in 0..8 {
                    assert_eq!(xp[g * 128 + b * 16 + l], x[g * 128 + l * 8 + b]);
                }
            }
        }
        assert_eq!(
            prepare_x_perm(&x[..127], &mut xp, &mut xg),
            Err(EvalError::InvalidShape)
        );
        assert_eq!(scratch_len(usize::MAX, 2), None);
    }

    #[test]
    fn direct_fp16_preparation_matches_fp32_for_all_finite_half_values() {
        let input = (0..=u16::MAX)
            .filter(|v| v & 0x7c00 != 0x7c00)
            .collect::<Vec<_>>();
        let x = input
            .iter()
            .map(|&v| crate::cuda::f16_to_f32(v))
            .collect::<Vec<_>>();
        let mut xp = vec![0.; x.len()];
        let mut xg = vec![0.; x.len() / 128];
        prepare_x_perm(&x, &mut xp, &mut xg).unwrap();
        let mut hp = vec![0.; x.len()];
        let mut hg = vec![0.; x.len() / 128];
        prepare_x_perm_f16(&input, &mut hp, &mut hg).unwrap();
        assert_eq!(
            xp.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
            hp.iter().map(|v| v.to_bits()).collect::<Vec<_>>()
        );
        assert_eq!(xg, hg);
        assert_eq!(
            prepare_x_perm_f16(&input[..127], &mut hp, &mut hg),
            Err(EvalError::InvalidShape)
        );
    }

    fn words(n: usize, seed: u32) -> Vec<u32> {
        let mut state = seed;
        (0..n)
            .map(|_| {
                state = state.wrapping_mul(1664525).wrapping_add(1013904223);
                state
            })
            .collect()
    }
    // Independent ordinary dequantized dot product; fixtures use scale=2^-7.
    fn dot(w: &[u32], x: &[f32]) -> f32 {
        x.iter()
            .enumerate()
            .map(|(c, v)| (((w[c / 8] >> (4 * (c % 8))) & 15) as f32 - 8.0) * (1.0 / 128.0) * v)
            .sum()
    }
    #[test]
    fn expert_scalar_and_tile_agreement() {
        if !supported() {
            eprintln!("AVX512 unavailable; skipping arithmetic test");
            return;
        }
        let h = 256;
        let i = 256;
        let gu = words(2 * i * h / 8, 5);
        let dn = words(h * i / 8, 19);
        let gs = vec![0x2000; 2 * i * h / 128];
        let ds = vec![0x2000; h * i / 128];
        let d = ExpertLayout {
            hidden: h,
            intermediate: i,
            gate_up_words: &gu,
            gate_up_scales: &gs,
            down_words: &dn,
            down_scales: &ds,
        };
        for nt in [1, 2, 3, 4, 5, 8, 9, 16, 17, 32, 33, 96] {
            let x: Vec<_> = (0..nt * h)
                .map(|j| ((j * 17 % 101) as f32 - 50.0) / 100.0)
                .collect();
            let wt: Vec<_> = (0..nt)
                .map(|t| if t % 2 == 0 { 0.7 } else { -0.2 })
                .collect();
            let mut xp = vec![0.0; nt * h];
            let mut xg = vec![0.0; nt * h / 128];
            for t in 0..nt {
                prepare_x_perm(
                    &x[t * h..(t + 1) * h],
                    &mut xp[t * h..(t + 1) * h],
                    &mut xg[t * h / 128..(t + 1) * h / 128],
                )
                .unwrap();
            }
            let mut out = vec![0.0; nt * h];
            let mut scratch = vec![0.0; scratch_len(nt, 2).unwrap()];
            eval_expert_prefill(d, &xp, &xg, &wt, &mut out, &mut scratch).unwrap();
            for t in 0..nt {
                let act: Vec<_> = (0..i)
                    .map(|r| {
                        let g = dot(&gu[r * h / 8..], &x[t * h..(t + 1) * h]);
                        let u = dot(&gu[(i + r) * h / 8..], &x[t * h..(t + 1) * h]);
                        (g / (1.0 + (-g).exp())) * u
                    })
                    .collect();
                for r in 0..h {
                    let expected = wt[t] * dot(&dn[r * i / 8..], &act);
                    assert!((out[t * h + r] - expected).abs() < 2e-5, "t={t}, r={r}");
                }
            }
            if nt <= 4 {
                let mut full = vec![0.0; nt * h];
                eval_expert_slice(d, 0..2, &xp, &xg, &wt, &mut full, &mut scratch).unwrap();
                assert_eq!(full, out);
                let mut a = vec![0.0; nt * h];
                let mut b = a.clone();
                eval_expert_slice(d, 0..1, &xp, &xg, &wt, &mut a, &mut scratch).unwrap();
                eval_expert_slice(d, 1..2, &xp, &xg, &wt, &mut b, &mut scratch).unwrap();
                for j in 0..out.len() {
                    assert!((a[j] + b[j] - out[j]).abs() < 2e-5);
                }
            }
        }
        let mut out = [0.0; 256];
        assert_eq!(
            eval_expert_slice(d, 0..2, &[], &[], &[1.0], &mut out, &mut []),
            Err(EvalError::InvalidShape)
        );
    }
}
