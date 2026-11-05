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

pub(crate) fn supported() -> bool {
    #[cfg(target_arch = "x86_64")]
    {
        std::is_x86_feature_detected!("avx2")
            && std::is_x86_feature_detected!("fma")
            && std::is_x86_feature_detected!("f16c")
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
    #[cfg(target_arch = "x86_64")]
    if supported() {
        // SAFETY: runtime ISA guard and slice bounds checked above.
        unsafe {
            simd::prepare_x_perm_f16c(x, xp, xg);
        }
        return Ok(());
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

/// Add `src` into `dst` elementwise (`dst[i] += src[i]`).
pub(super) fn add_assign_slice(dst: &mut [f32], src: &[f32]) {
    assert_eq!(dst.len(), src.len());
    #[cfg(target_arch = "x86_64")]
    if supported() {
        // SAFETY: runtime ISA guard and equal lengths asserted above.
        unsafe {
            simd::add_assign_avx2(dst, src);
        }
        return;
    }
    for (o, &v) in dst.iter_mut().zip(src) {
        *o += v;
    }
}

/// Convert `src` (`f32`) to IEEE-754 binary16 (`dst`, round-to-nearest-even).
pub(super) fn f32_to_f16_slice(src: &[f32], dst: &mut [u16]) {
    assert_eq!(src.len(), dst.len());
    #[cfg(target_arch = "x86_64")]
    if supported() {
        // SAFETY: runtime ISA guard and equal lengths asserted above.
        unsafe {
            simd::f32_to_f16_f16c(src, dst);
        }
        return;
    }
    for (o, &v) in dst.iter_mut().zip(src) {
        *o = crate::cuda::f32_to_f16(v);
    }
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
        simd::expert(d, groups, xp, xg, weights, out, scratch, weights.len() == 1);
    }
    Ok(())
}

/// Full expert, 1..=96 prepared tokens. Same layout and output contract as slice.
/// Four-row tiles reuse L1-resident weight groups across register-tiled token pairs.
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
        simd::expert(d, groups, xp, xg, weights, out, scratch, true);
    }
    Ok(())
}

#[cfg(target_arch = "x86_64")]
#[allow(unsafe_op_in_unsafe_fn)]
mod simd {
    use super::*;
    use std::arch::x86_64::*;

    #[target_feature(enable = "avx2,fma,f16c")]
    pub(super) unsafe fn prepare_x_perm_f16c(x: &[u16], xp: &mut [f32], xg: &mut [f32]) {
        let mut group = [0.0f32; 128];
        for (g, src) in x.as_chunks::<128>().0.iter().enumerate() {
            for k in (0..128).step_by(8) {
                let vh = _mm_loadu_si128(src.as_ptr().add(k).cast());
                _mm256_storeu_ps(group.as_mut_ptr().add(k), _mm256_cvtph_ps(vh));
            }
            let mut sum = 0.0f32;
            let dst = &mut xp[g * 128..(g + 1) * 128];
            for l in 0..16 {
                for b in 0..8 {
                    let v = group[l * 8 + b];
                    dst[b * 16 + l] = v;
                    sum += v;
                }
            }
            xg[g] = sum;
        }
    }

    #[target_feature(enable = "avx2")]
    pub(super) unsafe fn add_assign_avx2(dst: &mut [f32], src: &[f32]) {
        let n = dst.len();
        let mut i = 0;
        while i + 8 <= n {
            let a = _mm256_loadu_ps(dst.as_ptr().add(i));
            let b = _mm256_loadu_ps(src.as_ptr().add(i));
            _mm256_storeu_ps(dst.as_mut_ptr().add(i), _mm256_add_ps(a, b));
            i += 8;
        }
        while i < n {
            *dst.get_unchecked_mut(i) += *src.get_unchecked(i);
            i += 1;
        }
    }

    #[target_feature(enable = "avx2,f16c")]
    pub(super) unsafe fn f32_to_f16_f16c(src: &[f32], dst: &mut [u16]) {
        let n = src.len();
        let mut i = 0;
        while i + 8 <= n {
            let v = _mm256_loadu_ps(src.as_ptr().add(i));
            let vh = _mm256_cvtps_ph::<_MM_FROUND_TO_NEAREST_INT>(v);
            _mm_storeu_si128(dst.as_mut_ptr().add(i).cast(), vh);
            i += 8;
        }
        while i < n {
            *dst.get_unchecked_mut(i) = crate::cuda::f32_to_f16(*src.get_unchecked(i));
            i += 1;
        }
    }

    #[target_feature(enable = "avx2,fma,f16c")]
    unsafe fn hsum256_pair(lo: __m256, hi: __m256) -> f32 {
        let s = _mm256_add_ps(lo, hi);
        let s = _mm_add_ps(_mm256_castps256_ps128(s), _mm256_extractf128_ps::<1>(s));
        let s = _mm_hadd_ps(s, s);
        _mm_cvtss_f32(_mm_hadd_ps(s, s))
    }

    #[target_feature(enable = "avx2,fma,f16c,avx512f,avx512vl")]
    unsafe fn hsum(v: __m512) -> f32 {
        hsum256_pair(
            _mm512_castps512_ps256(v),
            _mm256_castpd_ps(_mm512_extractf64x4_pd::<1>(_mm512_castps_pd(v))),
        )
    }

    // AVX2 + FMA + F16C fallback for Zen 2 / Zen 3 / Alder Lake / Raptor Lake CPUs
    // without AVX-512. Splits each 16-lane group into (lo, hi) __m256 pairs with
    // identical FMA and reduction tree to `row_tile` (0 ULP bit-identical output).
    #[target_feature(enable = "avx2,fma,f16c")]
    #[allow(clippy::cognitive_complexity, clippy::needless_range_loop)]
    unsafe fn row_tile_avx2<const R: usize, const T: usize>(
        w: [&[u32]; R],
        s: [&[u16]; R],
        xp: &[f32],
        xg: &[f32],
        groups: usize,
    ) -> [[f32; R]; T] {
        let z = _mm256_setzero_ps();
        let mut total_lo = [[z; R]; T];
        let mut total_hi = [[z; R]; T];
        let mut bias = [[0.0f32; R]; T];
        let mask15 = _mm256_set1_epi32(15);
        for g in 0..groups {
            let mut q_lo = [_mm256_setzero_si256(); R];
            let mut q_hi = [_mm256_setzero_si256(); R];
            for r in 0..R {
                let ptr = w[r].as_ptr().add(g * 16);
                q_lo[r] = _mm256_loadu_si256(ptr.cast());
                q_hi[r] = _mm256_loadu_si256(ptr.add(8).cast());
                if (g + 4) * 16 < w[r].len() {
                    _mm_prefetch::<_MM_HINT_T0>(w[r].as_ptr().add((g + 4) * 16).cast());
                }
            }
            let mut scale = [0.0; R];
            let mut vscale = [z; R];
            for r in 0..R {
                scale[r] = _mm_cvtss_f32(_mm_cvtph_ps(_mm_cvtsi32_si128(
                    *s[r].get_unchecked(g) as i32
                )));
                vscale[r] = _mm256_set1_ps(scale[r]);
            }
            let mut acc_lo = [[z; R]; T];
            let mut acc_hi = [[z; R]; T];
            macro_rules! nib256 {
                ($b:literal) => {{
                    for r in 0..R {
                        let ws_lo = _mm256_cvtepi32_ps(_mm256_and_si256(
                            _mm256_srli_epi32::<{ 4 * $b }>(q_lo[r]),
                            mask15,
                        ));
                        let ws_hi = _mm256_cvtepi32_ps(_mm256_and_si256(
                            _mm256_srli_epi32::<{ 4 * $b }>(q_hi[r]),
                            mask15,
                        ));
                        for t in 0..T {
                            let x_ptr = xp.as_ptr().add(t * groups * 128 + g * 128 + 16 * $b);
                            let x_lo = _mm256_loadu_ps(x_ptr);
                            let x_hi = _mm256_loadu_ps(x_ptr.add(8));
                            acc_lo[t][r] = _mm256_fmadd_ps(ws_lo, x_lo, acc_lo[t][r]);
                            acc_hi[t][r] = _mm256_fmadd_ps(ws_hi, x_hi, acc_hi[t][r]);
                        }
                    }
                }};
            }
            nib256!(0);
            nib256!(1);
            nib256!(2);
            nib256!(3);
            nib256!(4);
            nib256!(5);
            nib256!(6);
            nib256!(7);
            for t in 0..T {
                let xgv = *xg.get_unchecked(t * groups + g);
                for r in 0..R {
                    total_lo[t][r] = _mm256_fmadd_ps(acc_lo[t][r], vscale[r], total_lo[t][r]);
                    total_hi[t][r] = _mm256_fmadd_ps(acc_hi[t][r], vscale[r], total_hi[t][r]);
                    bias[t][r] = scale[r].mul_add(xgv, bias[t][r]);
                }
            }
        }
        let mut out = [[0.0; R]; T];
        for t in 0..T {
            for r in 0..R {
                out[t][r] = hsum256_pair(total_lo[t][r], total_hi[t][r]) - 8.0 * bias[t][r];
            }
        }
        out
    }

    #[target_feature(enable = "avx2,fma,f16c")]
    unsafe fn rows_avx2<const R: usize>(
        w: [&[u32]; R],
        s: [&[u16]; R],
        xp: &[f32],
        xg: &[f32],
        groups: usize,
        n: usize,
        mut emit: impl FnMut(usize, [f32; R]),
    ) {
        let xp_stride = groups * 128;
        let mut t = 0;
        if R <= 2 {
            while t + 2 <= n {
                let v =
                    row_tile_avx2::<R, 2>(w, s, &xp[t * xp_stride..], &xg[t * groups..], groups);
                emit(t, v[0]);
                emit(t + 1, v[1]);
                t += 2;
            }
        }
        while t < n {
            let v = row_tile_avx2::<R, 1>(w, s, &xp[t * xp_stride..], &xg[t * groups..], groups);
            emit(t, v[0]);
            t += 1;
        }
    }

    // R=1/2/4 is the row/two-row/four-row kernel. T>=8 uses the original
    // prefill group -> token -> nibble schedule, keeping only R live accumulators.
    // Small T (1..=4) uses nibble -> token, sharing decoded weights in registers.
    // Explicit row/token/nibble loops preserve the donor's SIMD evaluation order while
    // keeping `R * T <= 8` (`<= 30` ZMM registers) 100% register-resident with zero spills.
    #[target_feature(enable = "avx2,fma,f16c,avx512f,avx512vl")]
    #[allow(clippy::cognitive_complexity, clippy::needless_range_loop)]
    unsafe fn row_tile<const R: usize, const T: usize>(
        w: [&[u32]; R],
        s: [&[u16]; R],
        xp: &[f32],
        xg: &[f32],
        groups: usize,
    ) -> [[f32; R]; T] {
        let z = _mm512_setzero_ps();
        let mut total = [[z; R]; T];
        let mut bias = [[0.0f32; R]; T];
        for g in 0..groups {
            let mut q = [_mm512_setzero_si512(); R];
            for r in 0..R {
                q[r] = _mm512_loadu_si512(w[r].as_ptr().add(g * 16).cast());
                if (g + 4) * 16 < w[r].len() {
                    _mm_prefetch::<_MM_HINT_T0>(w[r].as_ptr().add((g + 4) * 16).cast());
                }
            }
            let mut scale = [0.0; R];
            let mut vscale = [z; R];
            for r in 0..R {
                scale[r] = _mm_cvtss_f32(_mm_cvtph_ps(_mm_cvtsi32_si128(
                    *s[r].get_unchecked(g) as i32
                )));
                vscale[r] = _mm512_set1_ps(scale[r]);
            }
            let mut acc = [[z; R]; T];
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
                        let x =
                            _mm512_loadu_ps(xp.as_ptr().add(t * groups * 128 + g * 128 + 16 * $b));
                        for r in 0..R {
                            acc[t][r] = _mm512_fmadd_ps(ws[r], x, acc[t][r]);
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
            for t in 0..T {
                let xgv = *xg.get_unchecked(t * groups + g);
                for r in 0..R {
                    total[t][r] = _mm512_fmadd_ps(acc[t][r], vscale[r], total[t][r]);
                    // Match the original host compiler's contracted scalar bias update.
                    bias[t][r] = scale[r].mul_add(xgv, bias[t][r]);
                }
            }
        }
        let mut out = [[0.0; R]; T];
        for t in 0..T {
            for r in 0..R {
                out[t][r] = hsum(total[t][r]) - 8.0 * bias[t][r];
            }
        }
        out
    }

    #[target_feature(enable = "avx2,fma,f16c,avx512f,avx512vl")]
    unsafe fn rows<const R: usize>(
        w: [&[u32]; R],
        s: [&[u16]; R],
        xp: &[f32],
        xg: &[f32],
        groups: usize,
        n: usize,
        mut emit: impl FnMut(usize, [f32; R]),
    ) {
        let xp_stride = groups * 128;
        let mut t = 0;
        if R <= 2 {
            while t + 4 <= n {
                let v = row_tile::<R, 4>(w, s, &xp[t * xp_stride..], &xg[t * groups..], groups);
                for (dt, row) in v.into_iter().enumerate() {
                    emit(t + dt, row);
                }
                t += 4;
            }
        }
        while t + 2 <= n {
            let v = row_tile::<R, 2>(w, s, &xp[t * xp_stride..], &xg[t * groups..], groups);
            emit(t, v[0]);
            emit(t + 1, v[1]);
            t += 2;
        }
        if t < n {
            let v = row_tile::<R, 1>(w, s, &xp[t * xp_stride..], &xg[t * groups..], groups);
            emit(t, v[0]);
        }
    }

    #[allow(clippy::too_many_arguments)]
    pub(super) unsafe fn expert(
        d: ExpertLayout<'_>,
        range: Range<usize>,
        xp: &[f32],
        xg: &[f32],
        wt: &[f32],
        out: &mut [f32],
        scratch: &mut [f32],
        tile: bool,
    ) {
        if std::is_x86_feature_detected!("avx512f") && std::is_x86_feature_detected!("avx512vl") {
            expert_avx512(d, range, xp, xg, wt, out, scratch, tile);
        } else {
            expert_avx2(d, range, xp, xg, wt, out, scratch, tile);
        }
    }

    #[target_feature(enable = "avx2,fma,f16c")]
    #[allow(clippy::too_many_arguments)]
    pub(super) unsafe fn expert_avx2(
        d: ExpertLayout<'_>,
        range: Range<usize>,
        xp: &[f32],
        xg: &[f32],
        wt: &[f32],
        out: &mut [f32],
        scratch: &mut [f32],
        _tile: bool,
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
            rows_avx2::<2>(
                [gw(r), gw(i + r)],
                [gs(r), gs(i + r)],
                xp,
                xg,
                h / 128,
                n,
                |t, v| {
                    let g = v[0];
                    act[t * cols + c] = (g / (1.0f32 + (-g).exp())) * v[1];
                },
            );
            c += 1;
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
        while r + 2 <= h {
            rows_avx2::<2>(
                [dw(r), dw(r + 1)],
                [ds(r), ds(r + 1)],
                ap,
                ag,
                sg,
                n,
                |t, v| {
                    for j in 0..2 {
                        out[t * h + r + j] = wt[t] * v[j];
                    }
                },
            );
            r += 2;
        }
        while r < h {
            rows_avx2::<1>([dw(r)], [ds(r)], ap, ag, sg, n, |t, v| {
                out[t * h + r] = wt[t] * v[0];
            });
            r += 1;
        }
    }

    #[target_feature(enable = "avx2,fma,f16c,avx512f,avx512vl")]
    #[allow(clippy::too_many_arguments)]
    pub(super) unsafe fn expert_avx512(
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
            if tile {
                rows::<4>(
                    [gw(r), gw(r + 1), gw(i + r), gw(i + r + 1)],
                    [gs(r), gs(r + 1), gs(i + r), gs(i + r + 1)],
                    xp,
                    xg,
                    h / 128,
                    n,
                    |t, v| {
                        for j in 0..2 {
                            let g = v[j];
                            // f32::exp, NOT an f64 exp or approximate SIMD exponential.
                            act[t * cols + c + j] = (g / (1.0f32 + (-g).exp())) * v[j + 2];
                        }
                    },
                );
                c += 2;
            } else {
                rows::<2>(
                    [gw(r), gw(i + r)],
                    [gs(r), gs(i + r)],
                    xp,
                    xg,
                    h / 128,
                    n,
                    |t, v| {
                        let g = v[0];
                        act[t * cols + c] = (g / (1.0f32 + (-g).exp())) * v[1];
                    },
                );
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
        if tile {
            while r + 4 <= h {
                rows::<4>(
                    [dw(r), dw(r + 1), dw(r + 2), dw(r + 3)],
                    [ds(r), ds(r + 1), ds(r + 2), ds(r + 3)],
                    ap,
                    ag,
                    sg,
                    n,
                    |t, v| {
                        for j in 0..4 {
                            out[t * h + r + j] = wt[t] * v[j];
                        }
                    },
                );
                r += 4;
            }
        }
        while r + 2 <= h {
            rows::<2>(
                [dw(r), dw(r + 1)],
                [ds(r), ds(r + 1)],
                ap,
                ag,
                sg,
                n,
                |t, v| {
                    for j in 0..2 {
                        out[t * h + r + j] = wt[t] * v[j];
                    }
                },
            );
            r += 2;
        }
        while r < h {
            rows::<1>([dw(r)], [ds(r)], ap, ag, sg, n, |t, v| {
                out[t * h + r] = wt[t] * v[0];
            });
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
            eprintln!("AVX2+FMA+F16C unavailable; skipping arithmetic test");
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
            #[cfg(target_arch = "x86_64")]
            {
                let mut out_avx2 = vec![0.0; nt * h];
                // SAFETY: `supported()` verified AVX2 + FMA + F16C and buffer bounds above.
                unsafe {
                    simd::expert_avx2(d, 0..2, &xp, &xg, &wt, &mut out_avx2, &mut scratch, true);
                }
                assert_eq!(
                    out.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                    out_avx2.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                    "AVX2 and AVX512 (tile=true) diverged at nt={nt}"
                );
                if nt <= 4
                    && std::is_x86_feature_detected!("avx512f")
                    && std::is_x86_feature_detected!("avx512vl")
                {
                    let mut slice_avx512 = vec![0.0; nt * h];
                    let mut slice_avx2 = vec![0.0; nt * h];
                    // SAFETY: runtime AVX-512 + AVX2 guards and buffer bounds verified above.
                    unsafe {
                        simd::expert_avx512(
                            d,
                            0..2,
                            &xp,
                            &xg,
                            &wt,
                            &mut slice_avx512,
                            &mut scratch,
                            false,
                        );
                        simd::expert_avx2(
                            d,
                            0..2,
                            &xp,
                            &xg,
                            &wt,
                            &mut slice_avx2,
                            &mut scratch,
                            false,
                        );
                    }
                    assert_eq!(
                        slice_avx512.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                        slice_avx2.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                        "AVX2 and AVX512 (tile=false) diverged at nt={nt}"
                    );
                }
            }
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
