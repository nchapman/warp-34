/// Converts IEEE-754 binary16 bits (`u16`) to `f32`.
///
/// # Examples
///
/// ```
/// use w34_host::cuda::{f16_to_f32, f32_to_f16};
///
/// assert_eq!(f16_to_f32(f32_to_f16(1.0)), 1.0);
/// ```
#[inline]
pub fn f16_to_f32(bits: u16) -> f32 {
    let sign = ((bits & 0x8000) as u32) << 16;
    let exp = ((bits >> 10) & 0x1f) as u32;
    let mant = (bits & 0x03ff) as u32;
    let f_bits = if exp == 0 {
        if mant == 0 {
            sign
        } else {
            let mut m = mant;
            let mut e = 127u32 - 14u32;
            while (m & 0x0400) == 0 {
                m <<= 1;
                e -= 1;
            }
            m &= 0x03ff;
            sign | (e << 23) | (m << 13)
        }
    } else if exp == 31 {
        sign | 0x7f80_0000 | (mant << 13)
    } else {
        sign | ((exp + 112) << 23) | (mant << 13)
    };
    f32::from_bits(f_bits)
}

/// Converts `f32` to IEEE-754 binary16 bits (`u16`) with round-to-nearest-even.
#[inline]
pub fn f32_to_f16(val: f32) -> u16 {
    let bits = val.to_bits();
    let sign = ((bits >> 16) & 0x8000) as u16;
    let exp = ((bits >> 23) & 0xff) as i32;
    let mant = bits & 0x007f_ffff;

    if exp == 255 {
        if mant != 0 {
            return sign | 0x7e00;
        }
        return sign | 0x7c00;
    }

    let new_exp = exp - 127 + 15;
    if new_exp >= 31 {
        return sign | 0x7c00;
    }
    if new_exp <= 0 {
        if new_exp < -10 {
            return sign;
        }
        let m = mant | 0x0080_0000;
        let shift = 14 - new_exp;
        let out_m = (m >> shift) as u16;
        let round_bit = (m >> (shift - 1)) & 1;
        let sticky = m & ((1u32 << (shift - 1)) - 1);
        let round_up = round_bit == 1 && (sticky != 0 || (out_m & 1) != 0);
        return sign | (out_m + u16::from(round_up));
    }

    let out_e = (new_exp as u16) << 10;
    let out_m = (mant >> 13) as u16;
    let round_bit = (mant >> 12) & 1;
    let sticky = mant & 0x0fff;
    let round_up = round_bit == 1 && (sticky != 0 || (out_m & 1) != 0);
    sign | (out_e + out_m + u16::from(round_up))
}

/// Converts BF16 bits (`u16`) to `f32`.
#[inline]
pub fn bf16_to_f32(bits: u16) -> f32 {
    f32::from_bits((bits as u32) << 16)
}

/// Converts `f32` to BF16 bits (`u16`) with round-to-nearest-even.
#[inline]
pub fn f32_to_bf16(val: f32) -> u16 {
    let bits = val.to_bits();
    if (bits & 0x7f80_0000) == 0x7f80_0000 && (bits & 0x007f_ffff) != 0 {
        return ((bits >> 16) as u16) | 0x0040;
    }
    let lsb = (bits >> 16) & 1;
    let rounding_bias = 0x7fff + lsb;
    ((bits.wrapping_add(rounding_bias)) >> 16) as u16
}

/// Converts BF16 bits (`u16`) to IEEE-754 FP16 bits (`u16`).
#[inline]
pub fn bf16_to_f16(bits: u16) -> u16 {
    f32_to_f16(bf16_to_f32(bits))
}

#[cfg(target_arch = "x86_64")]
#[allow(unsafe_op_in_unsafe_fn)]
pub(crate) mod half_simd {
    use std::arch::x86_64::*;

    #[inline]
    pub fn supported() -> bool {
        std::is_x86_feature_detected!("avx2") && std::is_x86_feature_detected!("f16c")
    }

    #[target_feature(enable = "avx2,f16c")]
    pub unsafe fn f16_to_f32_f16c(src: &[u16], dst: &mut [f32]) {
        let n = src.len();
        let mut i = 0;
        while i + 8 <= n {
            let vh = _mm_loadu_si128(src.as_ptr().add(i).cast());
            _mm256_storeu_ps(dst.as_mut_ptr().add(i), _mm256_cvtph_ps(vh));
            i += 8;
        }
        while i < n {
            let b = *src.get_unchecked(i) as i32;
            *dst.get_unchecked_mut(i) = _mm_cvtss_f32(_mm_cvtph_ps(_mm_cvtsi32_si128(b)));
            i += 1;
        }
    }

    #[target_feature(enable = "avx2,f16c")]
    pub unsafe fn f32_to_f16_f16c(src: &[f32], dst: &mut [u16]) {
        let n = src.len();
        let mut i = 0;
        while i + 8 <= n {
            let v = _mm256_loadu_ps(src.as_ptr().add(i));
            let vh = _mm256_cvtps_ph::<_MM_FROUND_TO_NEAREST_INT>(v);
            _mm_storeu_si128(dst.as_mut_ptr().add(i).cast(), vh);
            i += 8;
        }
        while i < n {
            let v = _mm_set_ss(*src.get_unchecked(i));
            let vh = _mm_cvtps_ph::<_MM_FROUND_TO_NEAREST_INT>(v);
            *dst.get_unchecked_mut(i) = _mm_extract_epi16::<0>(vh) as u16;
            i += 1;
        }
    }

    #[target_feature(enable = "avx2,f16c")]
    pub unsafe fn bf16_to_f16_f16c_inplace(words: &mut [u16]) {
        let n = words.len();
        let mut i = 0;
        while i + 8 <= n {
            let ptr = words.as_mut_ptr().add(i);
            let vbf = _mm_loadu_si128(ptr.cast());
            let vf32 = _mm256_castsi256_ps(_mm256_slli_epi32::<16>(_mm256_cvtepu16_epi32(vbf)));
            let vf16 = _mm256_cvtps_ph::<_MM_FROUND_TO_NEAREST_INT>(vf32);
            _mm_storeu_si128(ptr.cast(), vf16);
            i += 8;
        }
        while i < n {
            let w = *words.get_unchecked(i) as i32;
            let vf32 = _mm_castsi128_ps(_mm_slli_epi32::<16>(_mm_cvtsi32_si128(w)));
            let vf16 = _mm_cvtps_ph::<_MM_FROUND_TO_NEAREST_INT>(vf32);
            *words.get_unchecked_mut(i) = _mm_extract_epi16::<0>(vf16) as u16;
            i += 1;
        }
    }
}

/// Converts a slice of IEEE-754 binary16 bits (`u16`) into `f32` (`dst`).
/// Uses `AVX2 + F16C` (`_mm256_cvtph_ps`) on `x86_64` when available.
pub fn f16_to_f32_slice(src: &[u16], dst: &mut [f32]) {
    assert_eq!(src.len(), dst.len());
    #[cfg(target_arch = "x86_64")]
    if half_simd::supported() {
        // SAFETY: runtime AVX2+F16C guard and equal slice lengths asserted above.
        unsafe {
            half_simd::f16_to_f32_f16c(src, dst);
        }
        return;
    }
    for (o, &b) in dst.iter_mut().zip(src) {
        *o = f16_to_f32(b);
    }
}

/// Converts a slice of `f32` values into IEEE-754 binary16 bits (`u16`, round-to-nearest-even).
/// Uses `AVX2 + F16C` (`_mm256_cvtps_ph`) on `x86_64` when available.
pub fn f32_to_f16_slice(src: &[f32], dst: &mut [u16]) {
    assert_eq!(src.len(), dst.len());
    #[cfg(target_arch = "x86_64")]
    if half_simd::supported() {
        // SAFETY: runtime AVX2+F16C guard and equal slice lengths asserted above.
        unsafe {
            half_simd::f32_to_f16_f16c(src, dst);
        }
        return;
    }
    for (o, &v) in dst.iter_mut().zip(src) {
        *o = f32_to_f16(v);
    }
}

/// Converts a slice of `BF16` bits (`u16`) in place into IEEE-754 `FP16` bits (`u16`).
/// Uses `AVX2 + F16C` on `x86_64` when available.
pub fn bf16_to_f16_slice_inplace(words: &mut [u16]) {
    #[cfg(target_arch = "x86_64")]
    if half_simd::supported() {
        // SAFETY: runtime AVX2+F16C guard verified above; `words` is a valid mutable slice.
        unsafe {
            half_simd::bf16_to_f16_f16c_inplace(words);
        }
        return;
    }
    for w in words {
        *w = bf16_to_f16(*w);
    }
}
