//! Standalone U4B8 CPU arithmetic. No routing, reduction across experts, or ownership.
//! Packed rows contain eight consecutive unsigned nibbles per u32 (low nibble first),
//! with one IEEE binary16 scale per 128 columns. Gate rows precede up rows.
//! All outputs are token-major f32; slice outputs are partial down projections.

use std::ops::Range;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum EvalError {
    InvalidLayout,
    InvalidShape,
}

/// Borrowed storage for exactly one expert. Extra trailing storage is permitted.
#[derive(Clone, Copy)]
pub(super) struct ExpertLayout<'a> {
    pub hidden: usize,
    pub intermediate: usize,
    pub group_size: usize,
    pub down_hadamard: usize,
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

fn prepare_x_perm_portable(x: &[f32], xp: &mut [f32], xg: &mut [f32]) -> Result<(), EvalError> {
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

/// Transpose each group from [16 words][8 nibbles] to [8][16]. The group
/// sum is deliberately sequential in the ORIGINAL input order, not a SIMD reduction.
/// This preparation is portable; evaluation requires `supported()`.
pub(super) fn prepare_x_perm(x: &[f32], xp: &mut [f32], xg: &mut [f32]) -> Result<(), EvalError> {
    if !x.len().is_multiple_of(128) || xp.len() < x.len() || xg.len() < x.len() / 128 {
        return Err(EvalError::InvalidShape);
    }
    #[cfg(target_arch = "x86_64")]
    if supported() {
        // SAFETY: runtime ISA guard and slice bounds checked above.
        unsafe {
            simd::prepare_x_perm_avx2(x, xp, xg);
        }
        return Ok(());
    }
    prepare_x_perm_portable(x, xp, xg)
}

fn prepare_x_perm_f16_portable(x: &[u16], xp: &mut [f32], xg: &mut [f32]) -> Result<(), EvalError> {
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
    prepare_x_perm_f16_portable(x, xp, xg)
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

fn prepare_x_perm_grouped_portable(
    x: &[f32],
    group_size: usize,
    xp: &mut [f32],
    xg: &mut [f32],
) -> Result<(), EvalError> {
    match group_size {
        128 => prepare_x_perm_portable(x, xp, xg),
        64 => {
            if !x.len().is_multiple_of(128) || xp.len() < x.len() || xg.len() < x.len() / 64 {
                return Err(EvalError::InvalidShape);
            }
            for (g, src) in x.as_chunks::<128>().0.iter().enumerate() {
                let mut sum_lo = 0.0f32;
                for l in 0..8 {
                    for b in 0..8 {
                        let v = src[l * 8 + b];
                        xp[g * 128 + b * 16 + l] = v;
                        sum_lo += v;
                    }
                }
                let mut sum_hi = 0.0f32;
                for l in 8..16 {
                    for b in 0..8 {
                        let v = src[l * 8 + b];
                        xp[g * 128 + b * 16 + l] = v;
                        sum_hi += v;
                    }
                }
                xg[2 * g] = sum_lo;
                xg[2 * g + 1] = sum_hi;
            }
            Ok(())
        }
        _ => Err(EvalError::InvalidLayout),
    }
}

/// Transpose each 128-column block from [16 words][8 nibbles] to [8][16] and compute
/// sequential group sums for `group_size` in `{64, 128}`.
pub(super) fn prepare_x_perm_grouped(
    x: &[f32],
    group_size: usize,
    xp: &mut [f32],
    xg: &mut [f32],
) -> Result<(), EvalError> {
    match group_size {
        128 => prepare_x_perm(x, xp, xg),
        64 => {
            if !x.len().is_multiple_of(128) || xp.len() < x.len() || xg.len() < x.len() / 64 {
                return Err(EvalError::InvalidShape);
            }
            #[cfg(target_arch = "x86_64")]
            if supported() {
                // SAFETY: runtime ISA guard and slice bounds checked above.
                unsafe {
                    simd::prepare_x_perm_avx2_g64(x, xp, xg);
                }
                return Ok(());
            }
            prepare_x_perm_grouped_portable(x, 64, xp, xg)
        }
        _ => Err(EvalError::InvalidLayout),
    }
}

fn prepare_x_perm_f16_grouped_portable(
    x: &[u16],
    group_size: usize,
    xp: &mut [f32],
    xg: &mut [f32],
) -> Result<(), EvalError> {
    match group_size {
        128 => prepare_x_perm_f16_portable(x, xp, xg),
        64 => {
            if !x.len().is_multiple_of(128) || xp.len() < x.len() || xg.len() < x.len() / 64 {
                return Err(EvalError::InvalidShape);
            }
            for (g, src) in x.as_chunks::<128>().0.iter().enumerate() {
                let mut sum_lo = 0.0f32;
                for l in 0..8 {
                    for b in 0..8 {
                        let v = crate::cuda::f16_to_f32(src[l * 8 + b]);
                        xp[g * 128 + b * 16 + l] = v;
                        sum_lo += v;
                    }
                }
                let mut sum_hi = 0.0f32;
                for l in 8..16 {
                    for b in 0..8 {
                        let v = crate::cuda::f16_to_f32(src[l * 8 + b]);
                        xp[g * 128 + b * 16 + l] = v;
                        sum_hi += v;
                    }
                }
                xg[2 * g] = sum_lo;
                xg[2 * g + 1] = sum_hi;
            }
            Ok(())
        }
        _ => Err(EvalError::InvalidLayout),
    }
}

/// Prepare FP16 input directly for `group_size` in `{64, 128}`.
pub(super) fn prepare_x_perm_f16_grouped(
    x: &[u16],
    group_size: usize,
    xp: &mut [f32],
    xg: &mut [f32],
) -> Result<(), EvalError> {
    match group_size {
        128 => prepare_x_perm_f16(x, xp, xg),
        64 => {
            if !x.len().is_multiple_of(128) || xp.len() < x.len() || xg.len() < x.len() / 64 {
                return Err(EvalError::InvalidShape);
            }
            #[cfg(target_arch = "x86_64")]
            if supported() {
                // SAFETY: runtime ISA guard and slice bounds checked above.
                unsafe {
                    simd::prepare_x_perm_f16c_g64(x, xp, xg);
                }
                return Ok(());
            }
            prepare_x_perm_f16_grouped_portable(x, 64, xp, xg)
        }
        _ => Err(EvalError::InvalidLayout),
    }
}

fn hadamard64_f32_portable(slice: &mut [f32]) {
    for blk in slice.as_chunks_mut::<64>().0 {
        let mut h = 1usize;
        while h < 64 {
            for i in (0..64).step_by(h * 2) {
                for j in i..(i + h) {
                    let x = blk[j];
                    let y = blk[j + h];
                    blk[j] = x + y;
                    blk[j + h] = x - y;
                }
            }
            h *= 2;
        }
        for v in blk.iter_mut() {
            *v *= 0.125f32;
        }
    }
}

fn hadamard64_f32_inplace(slice: &mut [f32]) {
    #[cfg(target_arch = "x86_64")]
    if supported() {
        // SAFETY: runtime ISA guard checked above.
        unsafe {
            simd::hadamard64_f32_avx2(slice);
        }
        return;
    }
    hadamard64_f32_portable(slice);
}

/// Required caller-owned scratch in f32 elements: act, permuted act, group sums.
pub(super) fn scratch_len(tokens: usize, slice_groups: usize) -> Option<usize> {
    tokens.checked_mul(slice_groups)?.checked_mul(258)
}

fn validate(
    d: ExpertLayout<'_>,
    groups: &Range<usize>,
    [xp, xg]: [&[f32]; 2],
    wt: &[f32],
    out: &[f32],
    scratch: &[f32],
    max_tokens: usize,
) -> Result<(), EvalError> {
    let h = d.hidden;
    let i = d.intermediate;
    let gs = d.group_size;
    if h == 0
        || i == 0
        || !h.is_multiple_of(128)
        || !i.is_multiple_of(128)
        || !matches!(gs, 64 | 128)
        || !matches!(d.down_hadamard, 0 | 64)
    {
        return Err(EvalError::InvalidLayout);
    }
    let words_per_group = gs / 8;
    let gu = i
        .checked_mul(2)
        .and_then(|v| v.checked_mul(h / 8))
        .ok_or(EvalError::InvalidLayout)?;
    let dn = h.checked_mul(i / 8).ok_or(EvalError::InvalidLayout)?;
    if d.gate_up_words.len() < gu
        || d.gate_up_scales.len() < gu / words_per_group
        || d.down_words.len() < dn
        || d.down_scales.len() < dn / words_per_group
    {
        return Err(EvalError::InvalidLayout);
    }
    let t = wt.len();
    if t == 0 || t > max_tokens || groups.start >= groups.end || groups.end > i / 128 {
        return Err(EvalError::InvalidShape);
    }
    let nh = t.checked_mul(h).ok_or(EvalError::InvalidShape)?;
    let ns = scratch_len(t, groups.end - groups.start).ok_or(EvalError::InvalidShape)?;
    if xp.len() < nh || xg.len() < nh / gs || out.len() < nh || scratch.len() < ns {
        return Err(EvalError::InvalidShape);
    }
    Ok(())
}

fn row_portable(
    w: &[u32],
    s: &[u16],
    xp_t: &[f32],
    xg_t: &[f32],
    groups: usize,
    group_size: usize,
) -> f32 {
    let mut total = [0.0f32; 16];
    let mut bias = 0.0f32;
    for g in 0..groups {
        let mut acc = [0.0f32; 16];
        let wg = &w[g * 16..(g + 1) * 16];
        let xpg = &xp_t[g * 128..(g + 1) * 128];
        for b in 0..8 {
            let xb = &xpg[b * 16..(b + 1) * 16];
            for l in 0..16 {
                let nib = ((wg[l] >> (4 * b)) & 15) as f32;
                acc[l] = nib.mul_add(xb[l], acc[l]);
            }
        }
        if group_size == 64 {
            let scale_lo = crate::cuda::f16_to_f32(s[2 * g]);
            let scale_hi = crate::cuda::f16_to_f32(s[2 * g + 1]);
            for l in 0..8 {
                total[l] = acc[l].mul_add(scale_lo, total[l]);
            }
            for l in 8..16 {
                total[l] = acc[l].mul_add(scale_hi, total[l]);
            }
            bias = scale_hi.mul_add(xg_t[2 * g + 1], scale_lo.mul_add(xg_t[2 * g], bias));
        } else {
            let scale = crate::cuda::f16_to_f32(s[g]);
            for l in 0..16 {
                total[l] = acc[l].mul_add(scale, total[l]);
            }
            bias = scale.mul_add(xg_t[g], bias);
        }
    }
    let mut s8 = [0.0f32; 8];
    for i in 0..8 {
        s8[i] = total[i] + total[i + 8];
    }
    let mut s4 = [0.0f32; 4];
    for i in 0..4 {
        s4[i] = s8[i] + s8[i + 4];
    }
    let s2_0 = s4[0] + s4[1];
    let s2_1 = s4[2] + s4[3];
    (s2_0 + s2_1) - 8.0 * bias
}

fn expert_portable(
    d: ExpertLayout<'_>,
    range: Range<usize>,
    xp: &[f32],
    xg: &[f32],
    wt: &[f32],
    out: &mut [f32],
    scratch: &mut [f32],
) {
    let h = d.hidden;
    let i = d.intermediate;
    let gs_len = d.group_size;
    let g_per_128 = 128 / gs_len;
    let n = wt.len();
    let g_in = h / 128;
    let sg = range.end - range.start;
    let cols = sg * 128;
    let (act, rest) = scratch.split_at_mut(n * cols);
    let (ap, ag) = rest.split_at_mut(n * cols);
    let gw = |r: usize| &d.gate_up_words[r * (h / 8)..];
    let gs = |r: usize| &d.gate_up_scales[r * (h / gs_len)..];
    for c in 0..cols {
        let r = range.start * 128 + c;
        let w_gate = gw(r);
        let s_gate = gs(r);
        let w_up = gw(i + r);
        let s_up = gs(i + r);
        for t in 0..n {
            let xp_t = &xp[t * h..(t + 1) * h];
            let xg_t = &xg[t * (h / gs_len)..(t + 1) * (h / gs_len)];
            let g = row_portable(w_gate, s_gate, xp_t, xg_t, g_in, gs_len);
            let u = row_portable(w_up, s_up, xp_t, xg_t, g_in, gs_len);
            act[t * cols + c] = (g / (1.0f32 + (-g).exp())) * u;
        }
    }
    if d.down_hadamard == 64 {
        hadamard64_f32_portable(act);
    }
    for t in 0..n {
        prepare_x_perm_grouped_portable(
            &act[t * cols..(t + 1) * cols],
            gs_len,
            &mut ap[t * cols..(t + 1) * cols],
            &mut ag[t * sg * g_per_128..(t + 1) * sg * g_per_128],
        )
        .unwrap();
    }
    let dw = |r: usize| &d.down_words[r * (i / 8) + range.start * 16..];
    let ds = |r: usize| &d.down_scales[r * (i / gs_len) + range.start * g_per_128..];
    for r in 0..h {
        let w_dn = dw(r);
        let s_dn = ds(r);
        for (t, &w_tok) in wt.iter().enumerate() {
            let ap_t = &ap[t * cols..(t + 1) * cols];
            let ag_t = &ag[t * sg * g_per_128..(t + 1) * sg * g_per_128];
            out[t * h + r] = w_tok * row_portable(w_dn, s_dn, ap_t, ag_t, sg, gs_len);
        }
    }
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
    validate(d, &groups, [xp, xg], weights, out, scratch, 4)?;
    #[cfg(target_arch = "x86_64")]
    if supported() {
        // SAFETY: runtime ISA guard and all sizes/ranges checked above.
        unsafe {
            simd::expert(
                d,
                groups,
                [xp, xg],
                weights,
                out,
                scratch,
                weights.len() == 1,
            );
        }
        return Ok(());
    }
    expert_portable(d, groups, xp, xg, weights, out, scratch);
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
    validate(d, &groups, [xp, xg], weights, out, scratch, 96)?;
    #[cfg(target_arch = "x86_64")]
    if supported() {
        // SAFETY: runtime ISA guard and all sizes/ranges checked above.
        unsafe {
            simd::expert(d, groups, [xp, xg], weights, out, scratch, true);
        }
        return Ok(());
    }
    expert_portable(d, groups, xp, xg, weights, out, scratch);
    Ok(())
}

#[cfg(target_arch = "x86_64")]
#[allow(unsafe_op_in_unsafe_fn)]
mod simd {
    use super::*;
    use std::arch::x86_64::*;

    #[target_feature(enable = "avx2")]
    #[inline]
    #[allow(clippy::too_many_arguments, clippy::identity_op, clippy::erasing_op)]
    unsafe fn transpose_8x8_avx2(
        r0: __m256,
        r1: __m256,
        r2: __m256,
        r3: __m256,
        r4: __m256,
        r5: __m256,
        r6: __m256,
        r7: __m256,
        dst_base: *mut f32,
    ) {
        let t0 = _mm256_unpacklo_ps(r0, r1);
        let t1 = _mm256_unpackhi_ps(r0, r1);
        let t2 = _mm256_unpacklo_ps(r2, r3);
        let t3 = _mm256_unpackhi_ps(r2, r3);
        let t4 = _mm256_unpacklo_ps(r4, r5);
        let t5 = _mm256_unpackhi_ps(r4, r5);
        let t6 = _mm256_unpacklo_ps(r6, r7);
        let t7 = _mm256_unpackhi_ps(r6, r7);

        let tt0 = _mm256_shuffle_ps::<0x44>(t0, t2);
        let tt1 = _mm256_shuffle_ps::<0xEE>(t0, t2);
        let tt2 = _mm256_shuffle_ps::<0x44>(t1, t3);
        let tt3 = _mm256_shuffle_ps::<0xEE>(t1, t3);
        let tt4 = _mm256_shuffle_ps::<0x44>(t4, t6);
        let tt5 = _mm256_shuffle_ps::<0xEE>(t4, t6);
        let tt6 = _mm256_shuffle_ps::<0x44>(t5, t7);
        let tt7 = _mm256_shuffle_ps::<0xEE>(t5, t7);

        _mm256_storeu_ps(
            dst_base.add(0 * 16),
            _mm256_permute2f128_ps::<0x20>(tt0, tt4),
        );
        _mm256_storeu_ps(
            dst_base.add(1 * 16),
            _mm256_permute2f128_ps::<0x20>(tt1, tt5),
        );
        _mm256_storeu_ps(
            dst_base.add(2 * 16),
            _mm256_permute2f128_ps::<0x20>(tt2, tt6),
        );
        _mm256_storeu_ps(
            dst_base.add(3 * 16),
            _mm256_permute2f128_ps::<0x20>(tt3, tt7),
        );
        _mm256_storeu_ps(
            dst_base.add(4 * 16),
            _mm256_permute2f128_ps::<0x31>(tt0, tt4),
        );
        _mm256_storeu_ps(
            dst_base.add(5 * 16),
            _mm256_permute2f128_ps::<0x31>(tt1, tt5),
        );
        _mm256_storeu_ps(
            dst_base.add(6 * 16),
            _mm256_permute2f128_ps::<0x31>(tt2, tt6),
        );
        _mm256_storeu_ps(
            dst_base.add(7 * 16),
            _mm256_permute2f128_ps::<0x31>(tt3, tt7),
        );
    }

    #[target_feature(enable = "avx2")]
    pub(super) unsafe fn prepare_x_perm_avx2(x: &[f32], xp: &mut [f32], xg: &mut [f32]) {
        for (g, src) in x.as_chunks::<128>().0.iter().enumerate() {
            let p = src.as_ptr();
            let d = xp.as_mut_ptr().add(g * 128);
            transpose_8x8_avx2(
                _mm256_loadu_ps(p),
                _mm256_loadu_ps(p.add(8)),
                _mm256_loadu_ps(p.add(16)),
                _mm256_loadu_ps(p.add(24)),
                _mm256_loadu_ps(p.add(32)),
                _mm256_loadu_ps(p.add(40)),
                _mm256_loadu_ps(p.add(48)),
                _mm256_loadu_ps(p.add(56)),
                d,
            );
            transpose_8x8_avx2(
                _mm256_loadu_ps(p.add(64)),
                _mm256_loadu_ps(p.add(72)),
                _mm256_loadu_ps(p.add(80)),
                _mm256_loadu_ps(p.add(88)),
                _mm256_loadu_ps(p.add(96)),
                _mm256_loadu_ps(p.add(104)),
                _mm256_loadu_ps(p.add(112)),
                _mm256_loadu_ps(p.add(120)),
                d.add(8),
            );
            let mut sum = 0.0f32;
            for &v in src {
                sum += v;
            }
            *xg.get_unchecked_mut(g) = sum;
        }
    }

    #[target_feature(enable = "avx2")]
    pub(super) unsafe fn prepare_x_perm_avx2_g64(x: &[f32], xp: &mut [f32], xg: &mut [f32]) {
        for (g, src) in x.as_chunks::<128>().0.iter().enumerate() {
            let p = src.as_ptr();
            let d = xp.as_mut_ptr().add(g * 128);
            transpose_8x8_avx2(
                _mm256_loadu_ps(p),
                _mm256_loadu_ps(p.add(8)),
                _mm256_loadu_ps(p.add(16)),
                _mm256_loadu_ps(p.add(24)),
                _mm256_loadu_ps(p.add(32)),
                _mm256_loadu_ps(p.add(40)),
                _mm256_loadu_ps(p.add(48)),
                _mm256_loadu_ps(p.add(56)),
                d,
            );
            transpose_8x8_avx2(
                _mm256_loadu_ps(p.add(64)),
                _mm256_loadu_ps(p.add(72)),
                _mm256_loadu_ps(p.add(80)),
                _mm256_loadu_ps(p.add(88)),
                _mm256_loadu_ps(p.add(96)),
                _mm256_loadu_ps(p.add(104)),
                _mm256_loadu_ps(p.add(112)),
                _mm256_loadu_ps(p.add(120)),
                d.add(8),
            );
            let mut sum_lo = 0.0f32;
            for &v in &src[..64] {
                sum_lo += v;
            }
            let mut sum_hi = 0.0f32;
            for &v in &src[64..] {
                sum_hi += v;
            }
            *xg.get_unchecked_mut(2 * g) = sum_lo;
            *xg.get_unchecked_mut(2 * g + 1) = sum_hi;
        }
    }

    #[target_feature(enable = "avx2,fma,f16c")]
    pub(super) unsafe fn prepare_x_perm_f16c(x: &[u16], xp: &mut [f32], xg: &mut [f32]) {
        let mut group = [0.0f32; 128];
        for (g, src) in x.as_chunks::<128>().0.iter().enumerate() {
            let p = src.as_ptr();
            let d = xp.as_mut_ptr().add(g * 128);
            let r0 = _mm256_cvtph_ps(_mm_loadu_si128(p.cast()));
            let r1 = _mm256_cvtph_ps(_mm_loadu_si128(p.add(8).cast()));
            let r2 = _mm256_cvtph_ps(_mm_loadu_si128(p.add(16).cast()));
            let r3 = _mm256_cvtph_ps(_mm_loadu_si128(p.add(24).cast()));
            let r4 = _mm256_cvtph_ps(_mm_loadu_si128(p.add(32).cast()));
            let r5 = _mm256_cvtph_ps(_mm_loadu_si128(p.add(40).cast()));
            let r6 = _mm256_cvtph_ps(_mm_loadu_si128(p.add(48).cast()));
            let r7 = _mm256_cvtph_ps(_mm_loadu_si128(p.add(56).cast()));
            _mm256_storeu_ps(group.as_mut_ptr(), r0);
            _mm256_storeu_ps(group.as_mut_ptr().add(8), r1);
            _mm256_storeu_ps(group.as_mut_ptr().add(16), r2);
            _mm256_storeu_ps(group.as_mut_ptr().add(24), r3);
            _mm256_storeu_ps(group.as_mut_ptr().add(32), r4);
            _mm256_storeu_ps(group.as_mut_ptr().add(40), r5);
            _mm256_storeu_ps(group.as_mut_ptr().add(48), r6);
            _mm256_storeu_ps(group.as_mut_ptr().add(56), r7);
            transpose_8x8_avx2(r0, r1, r2, r3, r4, r5, r6, r7, d);

            let r8 = _mm256_cvtph_ps(_mm_loadu_si128(p.add(64).cast()));
            let r9 = _mm256_cvtph_ps(_mm_loadu_si128(p.add(72).cast()));
            let r10 = _mm256_cvtph_ps(_mm_loadu_si128(p.add(80).cast()));
            let r11 = _mm256_cvtph_ps(_mm_loadu_si128(p.add(88).cast()));
            let r12 = _mm256_cvtph_ps(_mm_loadu_si128(p.add(96).cast()));
            let r13 = _mm256_cvtph_ps(_mm_loadu_si128(p.add(104).cast()));
            let r14 = _mm256_cvtph_ps(_mm_loadu_si128(p.add(112).cast()));
            let r15 = _mm256_cvtph_ps(_mm_loadu_si128(p.add(120).cast()));
            _mm256_storeu_ps(group.as_mut_ptr().add(64), r8);
            _mm256_storeu_ps(group.as_mut_ptr().add(72), r9);
            _mm256_storeu_ps(group.as_mut_ptr().add(80), r10);
            _mm256_storeu_ps(group.as_mut_ptr().add(88), r11);
            _mm256_storeu_ps(group.as_mut_ptr().add(96), r12);
            _mm256_storeu_ps(group.as_mut_ptr().add(104), r13);
            _mm256_storeu_ps(group.as_mut_ptr().add(112), r14);
            _mm256_storeu_ps(group.as_mut_ptr().add(120), r15);
            transpose_8x8_avx2(r8, r9, r10, r11, r12, r13, r14, r15, d.add(8));

            let mut sum = 0.0f32;
            for v in group {
                sum += v;
            }
            *xg.get_unchecked_mut(g) = sum;
        }
    }

    #[target_feature(enable = "avx2,fma,f16c")]
    pub(super) unsafe fn prepare_x_perm_f16c_g64(x: &[u16], xp: &mut [f32], xg: &mut [f32]) {
        let mut group = [0.0f32; 128];
        for (g, src) in x.as_chunks::<128>().0.iter().enumerate() {
            let p = src.as_ptr();
            let d = xp.as_mut_ptr().add(g * 128);
            let r0 = _mm256_cvtph_ps(_mm_loadu_si128(p.cast()));
            let r1 = _mm256_cvtph_ps(_mm_loadu_si128(p.add(8).cast()));
            let r2 = _mm256_cvtph_ps(_mm_loadu_si128(p.add(16).cast()));
            let r3 = _mm256_cvtph_ps(_mm_loadu_si128(p.add(24).cast()));
            let r4 = _mm256_cvtph_ps(_mm_loadu_si128(p.add(32).cast()));
            let r5 = _mm256_cvtph_ps(_mm_loadu_si128(p.add(40).cast()));
            let r6 = _mm256_cvtph_ps(_mm_loadu_si128(p.add(48).cast()));
            let r7 = _mm256_cvtph_ps(_mm_loadu_si128(p.add(56).cast()));
            _mm256_storeu_ps(group.as_mut_ptr(), r0);
            _mm256_storeu_ps(group.as_mut_ptr().add(8), r1);
            _mm256_storeu_ps(group.as_mut_ptr().add(16), r2);
            _mm256_storeu_ps(group.as_mut_ptr().add(24), r3);
            _mm256_storeu_ps(group.as_mut_ptr().add(32), r4);
            _mm256_storeu_ps(group.as_mut_ptr().add(40), r5);
            _mm256_storeu_ps(group.as_mut_ptr().add(48), r6);
            _mm256_storeu_ps(group.as_mut_ptr().add(56), r7);
            transpose_8x8_avx2(r0, r1, r2, r3, r4, r5, r6, r7, d);

            let r8 = _mm256_cvtph_ps(_mm_loadu_si128(p.add(64).cast()));
            let r9 = _mm256_cvtph_ps(_mm_loadu_si128(p.add(72).cast()));
            let r10 = _mm256_cvtph_ps(_mm_loadu_si128(p.add(80).cast()));
            let r11 = _mm256_cvtph_ps(_mm_loadu_si128(p.add(88).cast()));
            let r12 = _mm256_cvtph_ps(_mm_loadu_si128(p.add(96).cast()));
            let r13 = _mm256_cvtph_ps(_mm_loadu_si128(p.add(104).cast()));
            let r14 = _mm256_cvtph_ps(_mm_loadu_si128(p.add(112).cast()));
            let r15 = _mm256_cvtph_ps(_mm_loadu_si128(p.add(120).cast()));
            _mm256_storeu_ps(group.as_mut_ptr().add(64), r8);
            _mm256_storeu_ps(group.as_mut_ptr().add(72), r9);
            _mm256_storeu_ps(group.as_mut_ptr().add(80), r10);
            _mm256_storeu_ps(group.as_mut_ptr().add(88), r11);
            _mm256_storeu_ps(group.as_mut_ptr().add(96), r12);
            _mm256_storeu_ps(group.as_mut_ptr().add(104), r13);
            _mm256_storeu_ps(group.as_mut_ptr().add(112), r14);
            _mm256_storeu_ps(group.as_mut_ptr().add(120), r15);
            transpose_8x8_avx2(r8, r9, r10, r11, r12, r13, r14, r15, d.add(8));

            let mut sum_lo = 0.0f32;
            for &v in &group[..64] {
                sum_lo += v;
            }
            let mut sum_hi = 0.0f32;
            for &v in &group[64..] {
                sum_hi += v;
            }
            *xg.get_unchecked_mut(2 * g) = sum_lo;
            *xg.get_unchecked_mut(2 * g + 1) = sum_hi;
        }
    }

    #[target_feature(enable = "avx2,fma")]
    pub(super) unsafe fn hadamard64_f32_avx2(slice: &mut [f32]) {
        let sign_h1 = _mm256_setr_ps(1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0);
        let sign_h2 = _mm256_setr_ps(1.0, 1.0, -1.0, -1.0, 1.0, 1.0, -1.0, -1.0);
        let sign_h4 = _mm256_setr_ps(1.0, 1.0, 1.0, 1.0, -1.0, -1.0, -1.0, -1.0);
        let scale = _mm256_set1_ps(0.125f32);
        for blk in slice.as_chunks_mut::<64>().0 {
            let p = blk.as_mut_ptr();
            let mut v = [_mm256_setzero_ps(); 8];
            for (i, vi) in v.iter_mut().enumerate() {
                let x = _mm256_loadu_ps(p.add(i * 8));
                let x_swap1 = _mm256_permute_ps::<0b10_11_00_01>(x);
                let s1 = _mm256_fmadd_ps(x, sign_h1, x_swap1);
                let s1_swap2 = _mm256_permute_ps::<0b01_00_11_10>(s1);
                let s2 = _mm256_fmadd_ps(s1, sign_h2, s1_swap2);
                let s2_swap4 = _mm256_permute2f128_ps::<0x01>(s2, s2);
                *vi = _mm256_fmadd_ps(s2, sign_h4, s2_swap4);
            }
            for i in (0..8).step_by(2) {
                let a = v[i];
                let b = v[i + 1];
                v[i] = _mm256_add_ps(a, b);
                v[i + 1] = _mm256_sub_ps(a, b);
            }
            for i in [0, 1, 4, 5] {
                let a = v[i];
                let b = v[i + 2];
                v[i] = _mm256_add_ps(a, b);
                v[i + 2] = _mm256_sub_ps(a, b);
            }
            for i in 0..4 {
                let a = v[i];
                let b = v[i + 4];
                _mm256_storeu_ps(p.add(i * 8), _mm256_mul_ps(_mm256_add_ps(a, b), scale));
                _mm256_storeu_ps(
                    p.add((i + 4) * 8),
                    _mm256_mul_ps(_mm256_sub_ps(a, b), scale),
                );
            }
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
    // For `R * T <= 2`, `lo` and `hi` are interleaved across nibbles; for `R * T > 2`
    // (`T in {2, 4}` with `R = 2`), `acc_lo` is accumulated and folded into `total_lo`
    // before `acc_hi` so the working set stays within the 16 YMM architectural registers.
    #[target_feature(enable = "avx2,fma,f16c")]
    #[allow(clippy::cognitive_complexity, clippy::needless_range_loop)]
    unsafe fn row_tile_avx2<const R: usize, const T: usize, const G64: bool>(
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
        if R * T <= 2 {
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
                let mut scale_lo = [0.0; R];
                let mut scale_hi = [0.0; R];
                let mut vscale_lo = [z; R];
                let mut vscale_hi = [z; R];
                for r in 0..R {
                    if G64 {
                        scale_lo[r] = _mm_cvtss_f32(_mm_cvtph_ps(_mm_cvtsi32_si128(
                            *s[r].get_unchecked(2 * g) as i32,
                        )));
                        scale_hi[r] = _mm_cvtss_f32(_mm_cvtph_ps(_mm_cvtsi32_si128(
                            *s[r].get_unchecked(2 * g + 1) as i32,
                        )));
                    } else {
                        let sc = _mm_cvtss_f32(_mm_cvtph_ps(_mm_cvtsi32_si128(
                            *s[r].get_unchecked(g) as i32,
                        )));
                        scale_lo[r] = sc;
                        scale_hi[r] = sc;
                    }
                    vscale_lo[r] = _mm256_set1_ps(scale_lo[r]);
                    vscale_hi[r] = _mm256_set1_ps(scale_hi[r]);
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
                    for r in 0..R {
                        total_lo[t][r] =
                            _mm256_fmadd_ps(acc_lo[t][r], vscale_lo[r], total_lo[t][r]);
                        total_hi[t][r] =
                            _mm256_fmadd_ps(acc_hi[t][r], vscale_hi[r], total_hi[t][r]);
                        if G64 {
                            let xg_lo = *xg.get_unchecked(t * 2 * groups + 2 * g);
                            let xg_hi = *xg.get_unchecked(t * 2 * groups + 2 * g + 1);
                            bias[t][r] =
                                scale_hi[r].mul_add(xg_hi, scale_lo[r].mul_add(xg_lo, bias[t][r]));
                        } else {
                            let xgv = *xg.get_unchecked(t * groups + g);
                            bias[t][r] = scale_lo[r].mul_add(xgv, bias[t][r]);
                        }
                    }
                }
            }
        } else {
            for g in 0..groups {
                let mut q_lo = [_mm256_setzero_si256(); R];
                for r in 0..R {
                    let ptr = w[r].as_ptr().add(g * 16);
                    q_lo[r] = _mm256_loadu_si256(ptr.cast());
                    if (g + 4) * 16 < w[r].len() {
                        _mm_prefetch::<_MM_HINT_T0>(w[r].as_ptr().add((g + 4) * 16).cast());
                    }
                }
                let mut acc_lo = [[z; R]; T];
                macro_rules! nib_lo {
                    ($b:literal) => {{
                        let mut ws = [z; R];
                        for r in 0..R {
                            ws[r] = _mm256_cvtepi32_ps(_mm256_and_si256(
                                _mm256_srli_epi32::<{ 4 * $b }>(q_lo[r]),
                                mask15,
                            ));
                        }
                        for t in 0..T {
                            let x_lo = _mm256_loadu_ps(
                                xp.as_ptr().add(t * groups * 128 + g * 128 + 16 * $b),
                            );
                            for r in 0..R {
                                acc_lo[t][r] = _mm256_fmadd_ps(ws[r], x_lo, acc_lo[t][r]);
                            }
                        }
                    }};
                }
                nib_lo!(0);
                nib_lo!(1);
                nib_lo!(2);
                nib_lo!(3);
                nib_lo!(4);
                nib_lo!(5);
                nib_lo!(6);
                nib_lo!(7);
                let mut scale_lo = [0.0; R];
                for r in 0..R {
                    let idx = if G64 { 2 * g } else { g };
                    scale_lo[r] = _mm_cvtss_f32(_mm_cvtph_ps(_mm_cvtsi32_si128(
                        *s[r].get_unchecked(idx) as i32,
                    )));
                    let vscale_lo = _mm256_set1_ps(scale_lo[r]);
                    for t in 0..T {
                        total_lo[t][r] = _mm256_fmadd_ps(acc_lo[t][r], vscale_lo, total_lo[t][r]);
                    }
                }

                let mut q_hi = [_mm256_setzero_si256(); R];
                for r in 0..R {
                    let ptr = w[r].as_ptr().add(g * 16 + 8);
                    q_hi[r] = _mm256_loadu_si256(ptr.cast());
                }
                let mut acc_hi = [[z; R]; T];
                macro_rules! nib_hi {
                    ($b:literal) => {{
                        let mut ws = [z; R];
                        for r in 0..R {
                            ws[r] = _mm256_cvtepi32_ps(_mm256_and_si256(
                                _mm256_srli_epi32::<{ 4 * $b }>(q_hi[r]),
                                mask15,
                            ));
                        }
                        for t in 0..T {
                            let x_hi = _mm256_loadu_ps(
                                xp.as_ptr().add(t * groups * 128 + g * 128 + 16 * $b + 8),
                            );
                            for r in 0..R {
                                acc_hi[t][r] = _mm256_fmadd_ps(ws[r], x_hi, acc_hi[t][r]);
                            }
                        }
                    }};
                }
                nib_hi!(0);
                nib_hi!(1);
                nib_hi!(2);
                nib_hi!(3);
                nib_hi!(4);
                nib_hi!(5);
                nib_hi!(6);
                nib_hi!(7);
                for r in 0..R {
                    let scale_hi = if G64 {
                        _mm_cvtss_f32(_mm_cvtph_ps(_mm_cvtsi32_si128(
                            *s[r].get_unchecked(2 * g + 1) as i32,
                        )))
                    } else {
                        scale_lo[r]
                    };
                    let vscale_hi = _mm256_set1_ps(scale_hi);
                    for t in 0..T {
                        total_hi[t][r] = _mm256_fmadd_ps(acc_hi[t][r], vscale_hi, total_hi[t][r]);
                        if G64 {
                            let xg_lo = *xg.get_unchecked(t * 2 * groups + 2 * g);
                            let xg_hi = *xg.get_unchecked(t * 2 * groups + 2 * g + 1);
                            bias[t][r] =
                                scale_hi.mul_add(xg_hi, scale_lo[r].mul_add(xg_lo, bias[t][r]));
                        } else {
                            let xgv = *xg.get_unchecked(t * groups + g);
                            bias[t][r] = scale_lo[r].mul_add(xgv, bias[t][r]);
                        }
                    }
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

    #[allow(clippy::too_many_arguments)]
    #[target_feature(enable = "avx2,fma,f16c")]
    unsafe fn rows_avx2<const R: usize>(
        w: [&[u32]; R],
        s: [&[u16]; R],
        xp: &[f32],
        xg: &[f32],
        groups: usize,
        group_size: usize,
        n: usize,
        mut emit: impl FnMut(usize, [f32; R]),
    ) {
        let xp_stride = groups * 128;
        let xg_stride = groups * (128 / group_size);
        let mut t = 0;
        if R <= 2 {
            while t + 4 <= n {
                let v = if group_size == 64 {
                    row_tile_avx2::<R, 4, true>(
                        w,
                        s,
                        &xp[t * xp_stride..],
                        &xg[t * xg_stride..],
                        groups,
                    )
                } else {
                    row_tile_avx2::<R, 4, false>(
                        w,
                        s,
                        &xp[t * xp_stride..],
                        &xg[t * xg_stride..],
                        groups,
                    )
                };
                for (dt, row) in v.into_iter().enumerate() {
                    emit(t + dt, row);
                }
                t += 4;
            }
            while t + 2 <= n {
                let v = if group_size == 64 {
                    row_tile_avx2::<R, 2, true>(
                        w,
                        s,
                        &xp[t * xp_stride..],
                        &xg[t * xg_stride..],
                        groups,
                    )
                } else {
                    row_tile_avx2::<R, 2, false>(
                        w,
                        s,
                        &xp[t * xp_stride..],
                        &xg[t * xg_stride..],
                        groups,
                    )
                };
                emit(t, v[0]);
                emit(t + 1, v[1]);
                t += 2;
            }
        }
        while t < n {
            let v = if group_size == 64 {
                row_tile_avx2::<R, 1, true>(
                    w,
                    s,
                    &xp[t * xp_stride..],
                    &xg[t * xg_stride..],
                    groups,
                )
            } else {
                row_tile_avx2::<R, 1, false>(
                    w,
                    s,
                    &xp[t * xp_stride..],
                    &xg[t * xg_stride..],
                    groups,
                )
            };
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
    unsafe fn row_tile<const R: usize, const T: usize, const G64: bool>(
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
            let mut scale_lo = [0.0; R];
            let mut scale_hi = [0.0; R];
            let mut vscale = [z; R];
            for r in 0..R {
                if G64 {
                    scale_lo[r] = _mm_cvtss_f32(_mm_cvtph_ps(_mm_cvtsi32_si128(
                        *s[r].get_unchecked(2 * g) as i32,
                    )));
                    scale_hi[r] = _mm_cvtss_f32(_mm_cvtph_ps(_mm_cvtsi32_si128(
                        *s[r].get_unchecked(2 * g + 1) as i32,
                    )));
                    vscale[r] = _mm512_insertf32x8::<1>(
                        _mm512_castps256_ps512(_mm256_set1_ps(scale_lo[r])),
                        _mm256_set1_ps(scale_hi[r]),
                    );
                } else {
                    scale_lo[r] = _mm_cvtss_f32(_mm_cvtph_ps(_mm_cvtsi32_si128(
                        *s[r].get_unchecked(g) as i32,
                    )));
                    scale_hi[r] = scale_lo[r];
                    vscale[r] = _mm512_set1_ps(scale_lo[r]);
                }
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
                for r in 0..R {
                    total[t][r] = _mm512_fmadd_ps(acc[t][r], vscale[r], total[t][r]);
                    // Match the original host compiler's contracted scalar bias update.
                    if G64 {
                        let xg_lo = *xg.get_unchecked(t * 2 * groups + 2 * g);
                        let xg_hi = *xg.get_unchecked(t * 2 * groups + 2 * g + 1);
                        bias[t][r] =
                            scale_hi[r].mul_add(xg_hi, scale_lo[r].mul_add(xg_lo, bias[t][r]));
                    } else {
                        let xgv = *xg.get_unchecked(t * groups + g);
                        bias[t][r] = scale_lo[r].mul_add(xgv, bias[t][r]);
                    }
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

    #[allow(clippy::too_many_arguments)]
    #[target_feature(enable = "avx2,fma,f16c,avx512f,avx512vl")]
    unsafe fn rows<const R: usize>(
        w: [&[u32]; R],
        s: [&[u16]; R],
        xp: &[f32],
        xg: &[f32],
        groups: usize,
        group_size: usize,
        n: usize,
        mut emit: impl FnMut(usize, [f32; R]),
    ) {
        let xp_stride = groups * 128;
        let xg_stride = groups * (128 / group_size);
        let mut t = 0;
        if R <= 2 {
            while t + 4 <= n {
                let v = if group_size == 64 {
                    row_tile::<R, 4, true>(w, s, &xp[t * xp_stride..], &xg[t * xg_stride..], groups)
                } else {
                    row_tile::<R, 4, false>(
                        w,
                        s,
                        &xp[t * xp_stride..],
                        &xg[t * xg_stride..],
                        groups,
                    )
                };
                for (dt, row) in v.into_iter().enumerate() {
                    emit(t + dt, row);
                }
                t += 4;
            }
        }
        while t + 2 <= n {
            let v = if group_size == 64 {
                row_tile::<R, 2, true>(w, s, &xp[t * xp_stride..], &xg[t * xg_stride..], groups)
            } else {
                row_tile::<R, 2, false>(w, s, &xp[t * xp_stride..], &xg[t * xg_stride..], groups)
            };
            emit(t, v[0]);
            emit(t + 1, v[1]);
            t += 2;
        }
        if t < n {
            let v = if group_size == 64 {
                row_tile::<R, 1, true>(w, s, &xp[t * xp_stride..], &xg[t * xg_stride..], groups)
            } else {
                row_tile::<R, 1, false>(w, s, &xp[t * xp_stride..], &xg[t * xg_stride..], groups)
            };
            emit(t, v[0]);
        }
    }

    pub(super) unsafe fn expert(
        d: ExpertLayout<'_>,
        range: Range<usize>,
        [xp, xg]: [&[f32]; 2],
        wt: &[f32],
        out: &mut [f32],
        scratch: &mut [f32],
        tile: bool,
    ) {
        if std::is_x86_feature_detected!("avx512f") && std::is_x86_feature_detected!("avx512vl") {
            expert_avx512(d, range, [xp, xg], wt, out, scratch, tile);
        } else {
            expert_avx2(d, range, [xp, xg], wt, out, scratch);
        }
    }

    #[target_feature(enable = "avx2,fma,f16c")]
    pub(super) unsafe fn expert_avx2(
        d: ExpertLayout<'_>,
        range: Range<usize>,
        [xp, xg]: [&[f32]; 2],
        wt: &[f32],
        out: &mut [f32],
        scratch: &mut [f32],
    ) {
        let h = d.hidden;
        let i = d.intermediate;
        let gs_len = d.group_size;
        let g_per_128 = 128 / gs_len;
        let n = wt.len();
        let sg = range.end - range.start;
        let cols = sg * 128;
        let (act, rest) = scratch.split_at_mut(n * cols);
        let (ap, ag) = rest.split_at_mut(n * cols);
        let gw = |r: usize| &d.gate_up_words[r * (h / 8)..];
        let gs = |r: usize| &d.gate_up_scales[r * (h / gs_len)..];
        let mut c = 0;
        while c < cols {
            let r = range.start * 128 + c;
            rows_avx2::<2>(
                [gw(r), gw(i + r)],
                [gs(r), gs(i + r)],
                xp,
                xg,
                h / 128,
                gs_len,
                n,
                |t, v| {
                    let g = v[0];
                    act[t * cols + c] = (g / (1.0f32 + (-g).exp())) * v[1];
                },
            );
            c += 1;
        }
        if d.down_hadamard == 64 {
            hadamard64_f32_inplace(act);
        }
        for t in 0..n {
            prepare_x_perm_grouped(
                &act[t * cols..(t + 1) * cols],
                gs_len,
                &mut ap[t * cols..(t + 1) * cols],
                &mut ag[t * sg * g_per_128..(t + 1) * sg * g_per_128],
            )
            .unwrap();
        }
        let dw = |r: usize| &d.down_words[r * (i / 8) + range.start * 16..];
        let ds = |r: usize| &d.down_scales[r * (i / gs_len) + range.start * g_per_128..];
        let mut r = 0;
        while r + 2 <= h {
            rows_avx2::<2>(
                [dw(r), dw(r + 1)],
                [ds(r), ds(r + 1)],
                ap,
                ag,
                sg,
                gs_len,
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
            rows_avx2::<1>([dw(r)], [ds(r)], ap, ag, sg, gs_len, n, |t, v| {
                out[t * h + r] = wt[t] * v[0];
            });
            r += 1;
        }
    }

    #[target_feature(enable = "avx2,fma,f16c,avx512f,avx512vl")]
    pub(super) unsafe fn expert_avx512(
        d: ExpertLayout<'_>,
        range: Range<usize>,
        [xp, xg]: [&[f32]; 2],
        wt: &[f32],
        out: &mut [f32],
        scratch: &mut [f32],
        tile: bool,
    ) {
        let h = d.hidden;
        let i = d.intermediate;
        let gs_len = d.group_size;
        let g_per_128 = 128 / gs_len;
        let n = wt.len();
        let sg = range.end - range.start;
        let cols = sg * 128;
        let (act, rest) = scratch.split_at_mut(n * cols);
        let (ap, ag) = rest.split_at_mut(n * cols);
        let gw = |r: usize| &d.gate_up_words[r * (h / 8)..];
        let gs = |r: usize| &d.gate_up_scales[r * (h / gs_len)..];
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
                    gs_len,
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
                    gs_len,
                    n,
                    |t, v| {
                        let g = v[0];
                        act[t * cols + c] = (g / (1.0f32 + (-g).exp())) * v[1];
                    },
                );
                c += 1;
            }
        }
        if d.down_hadamard == 64 {
            hadamard64_f32_inplace(act);
        }
        for t in 0..n {
            prepare_x_perm_grouped(
                &act[t * cols..(t + 1) * cols],
                gs_len,
                &mut ap[t * cols..(t + 1) * cols],
                &mut ag[t * sg * g_per_128..(t + 1) * sg * g_per_128],
            )
            .unwrap();
        }
        let dw = |r: usize| &d.down_words[r * (i / 8) + range.start * 16..];
        let ds = |r: usize| &d.down_scales[r * (i / gs_len) + range.start * g_per_128..];
        let mut r = 0;
        if tile {
            while r + 4 <= h {
                rows::<4>(
                    [dw(r), dw(r + 1), dw(r + 2), dw(r + 3)],
                    [ds(r), ds(r + 1), ds(r + 2), ds(r + 3)],
                    ap,
                    ag,
                    sg,
                    gs_len,
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
                gs_len,
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
            rows::<1>([dw(r)], [ds(r)], ap, ag, sg, gs_len, n, |t, v| {
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
        let mut xp_port = [0.0; 256];
        let mut xg_port = [0.0; 2];
        prepare_x_perm_portable(&x, &mut xp_port, &mut xg_port).unwrap();
        assert_eq!(xp, xp_port);
        assert_eq!(xg, xg_port);
        let mut xp_g64 = [0.0; 256];
        let mut xg_g64 = [0.0; 4];
        let mut xp_g64_port = [0.0; 256];
        let mut xg_g64_port = [0.0; 4];
        prepare_x_perm_grouped(&x, 64, &mut xp_g64, &mut xg_g64).unwrap();
        prepare_x_perm_grouped_portable(&x, 64, &mut xp_g64_port, &mut xg_g64_port).unwrap();
        assert_eq!(xp_g64, xp_g64_port);
        assert_eq!(xg_g64, xg_g64_port);
        let mut had_simd: Vec<f32> = (0..128)
            .map(|k| ((k * 37 % 103) as f32 - 51.0) / 16.0)
            .collect();
        let mut had_port = had_simd.clone();
        hadamard64_f32_inplace(&mut had_simd);
        hadamard64_f32_portable(&mut had_port);
        assert_eq!(
            had_simd.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
            had_port.iter().map(|v| v.to_bits()).collect::<Vec<_>>()
        );
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
        let mut hp_port = vec![0.; x.len()];
        let mut hg_port = vec![0.; x.len() / 128];
        prepare_x_perm_f16_portable(&input, &mut hp_port, &mut hg_port).unwrap();
        assert_eq!(
            xp.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
            hp.iter().map(|v| v.to_bits()).collect::<Vec<_>>()
        );
        assert_eq!(
            hp.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
            hp_port.iter().map(|v| v.to_bits()).collect::<Vec<_>>()
        );
        assert_eq!(xg, hg);
        assert_eq!(hg, hg_port);
        let mut hp_g64 = vec![0.; x.len()];
        let mut hg_g64 = vec![0.; x.len() / 64];
        let mut hp_g64_port = vec![0.; x.len()];
        let mut hg_g64_port = vec![0.; x.len() / 64];
        prepare_x_perm_f16_grouped(&input, 64, &mut hp_g64, &mut hg_g64).unwrap();
        prepare_x_perm_f16_grouped_portable(&input, 64, &mut hp_g64_port, &mut hg_g64_port)
            .unwrap();
        assert_eq!(
            hp_g64.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
            hp_g64_port.iter().map(|v| v.to_bits()).collect::<Vec<_>>()
        );
        assert_eq!(hg_g64, hg_g64_port);
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
        let h = 256;
        let i = 256;
        let gu = words(2 * i * h / 8, 5);
        let dn = words(h * i / 8, 19);
        let gs = vec![0x2000; 2 * i * h / 128];
        let ds = vec![0x2000; h * i / 128];
        let d = ExpertLayout {
            hidden: h,
            intermediate: i,
            group_size: 128,
            down_hadamard: 0,
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
            let mut out_portable = vec![0.0; nt * h];
            expert_portable(d, 0..2, &xp, &xg, &wt, &mut out_portable, &mut scratch);
            assert_eq!(
                out.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                out_portable.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                "portable and SIMD prefill diverged at nt={nt}"
            );
            #[cfg(target_arch = "x86_64")]
            if supported() {
                let mut out_avx2 = vec![0.0; nt * h];
                // SAFETY: `supported()` verified AVX2 + FMA + F16C and buffer bounds above.
                unsafe {
                    simd::expert_avx2(d, 0..2, [&xp, &xg], &wt, &mut out_avx2, &mut scratch);
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
                            [&xp, &xg],
                            &wt,
                            &mut slice_avx512,
                            &mut scratch,
                            false,
                        );
                        simd::expert_avx2(d, 0..2, [&xp, &xg], &wt, &mut slice_avx2, &mut scratch);
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

    #[test]
    fn expert_g64_hadamard64_scalar_and_tile_agreement() {
        let h = 256;
        let i = 256;
        let gu = words(2 * i * h / 8, 11);
        let dn = words(h * i / 8, 29);
        let gs: Vec<u16> = (0..(2 * i * h / 64))
            .map(|k| if k % 2 == 0 { 0x2000 } else { 0x2400 })
            .collect();
        let ds: Vec<u16> = (0..(h * i / 64))
            .map(|k| if k % 2 == 0 { 0x2000 } else { 0x1c00 })
            .collect();
        let d = ExpertLayout {
            hidden: h,
            intermediate: i,
            group_size: 64,
            down_hadamard: 64,
            gate_up_words: &gu,
            gate_up_scales: &gs,
            down_words: &dn,
            down_scales: &ds,
        };
        let dot_g64 = |w: &[u32], s: &[u16], x: &[f32]| -> f32 {
            x.iter()
                .enumerate()
                .map(|(c, v)| {
                    let scale = crate::cuda::f16_to_f32(s[c / 64]);
                    (((w[c / 8] >> (4 * (c % 8))) & 15) as f32 - 8.0) * scale * v
                })
                .sum()
        };
        for nt in [1, 2, 3, 4, 8, 17, 32] {
            let x_f16: Vec<u16> = (0..nt * h)
                .map(|j| crate::cuda::f32_to_f16(((j * 17 % 101) as f32 - 50.0) / 100.0))
                .collect();
            let x: Vec<f32> = x_f16.iter().map(|&b| crate::cuda::f16_to_f32(b)).collect();
            let wt: Vec<_> = (0..nt)
                .map(|t| if t % 2 == 0 { 0.7 } else { -0.2 })
                .collect();
            let mut xp = vec![0.0; nt * h];
            let mut xg = vec![0.0; nt * h / 64];
            let mut xp_f16 = vec![0.0; nt * h];
            let mut xg_f16 = vec![0.0; nt * h / 64];
            for t in 0..nt {
                prepare_x_perm_grouped(
                    &x[t * h..(t + 1) * h],
                    64,
                    &mut xp[t * h..(t + 1) * h],
                    &mut xg[t * h / 64..(t + 1) * h / 64],
                )
                .unwrap();
                prepare_x_perm_f16_grouped(
                    &x_f16[t * h..(t + 1) * h],
                    64,
                    &mut xp_f16[t * h..(t + 1) * h],
                    &mut xg_f16[t * h / 64..(t + 1) * h / 64],
                )
                .unwrap();
            }
            assert_eq!(xp, xp_f16);
            assert_eq!(xg, xg_f16);
            let mut out = vec![0.0; nt * h];
            let mut scratch = vec![0.0; scratch_len(nt, 2).unwrap()];
            eval_expert_prefill(d, &xp, &xg, &wt, &mut out, &mut scratch).unwrap();
            let mut out_portable = vec![0.0; nt * h];
            expert_portable(d, 0..2, &xp, &xg, &wt, &mut out_portable, &mut scratch);
            assert_eq!(
                out.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                out_portable.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                "g64+had64 portable and SIMD prefill diverged at nt={nt}"
            );
            #[cfg(target_arch = "x86_64")]
            if supported() {
                let mut out_avx2 = vec![0.0; nt * h];
                // SAFETY: `supported()` verified AVX2 + FMA + F16C and buffer bounds above.
                unsafe {
                    simd::expert_avx2(d, 0..2, [&xp, &xg], &wt, &mut out_avx2, &mut scratch);
                }
                assert_eq!(
                    out.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                    out_avx2.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                    "g64+had64 AVX2 and AVX512 diverged at nt={nt}"
                );
            }
            for t in 0..nt {
                let mut act: Vec<_> = (0..i)
                    .map(|r| {
                        let g =
                            dot_g64(&gu[r * h / 8..], &gs[r * h / 64..], &x[t * h..(t + 1) * h]);
                        let u = dot_g64(
                            &gu[(i + r) * h / 8..],
                            &gs[(i + r) * h / 64..],
                            &x[t * h..(t + 1) * h],
                        );
                        (g / (1.0 + (-g).exp())) * u
                    })
                    .collect();
                hadamard64_f32_inplace(&mut act);
                for r in 0..h {
                    let expected = wt[t] * dot_g64(&dn[r * i / 8..], &ds[r * i / 64..], &act);
                    assert!((out[t * h + r] - expected).abs() < 4e-5, "t={t}, r={r}");
                }
            }
            if nt <= 4 {
                let mut a = vec![0.0; nt * h];
                let mut b = a.clone();
                eval_expert_slice(d, 0..1, &xp, &xg, &wt, &mut a, &mut scratch).unwrap();
                eval_expert_slice(d, 1..2, &xp, &xg, &wt, &mut b, &mut scratch).unwrap();
                for j in 0..out.len() {
                    assert!((a[j] + b[j] - out[j]).abs() < 4e-5);
                }
            }
        }
    }
}
