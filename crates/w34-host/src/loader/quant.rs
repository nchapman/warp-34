use super::*;

/// Permutes a `[K / group_size, N]` FP16 scale matrix into `vLLM` Marlin scale layout.
///
/// When `group_size > 0 && group_size <= 128` (`group_blocks > 0`), applies the 64-element
/// 8x8 tile transpose expected by `Marlin<..., group_blocks > 0>` (`s_sh_rd = 8 * warp_col + lane / 4`),
/// even when `size_k == group_size` (e.g. `size_k = 128, group_size = 128, group_blocks = 8`).
/// When `group_size == 0` or `group_size > 128` (`group_blocks == -1`, channelwise W8A16),
/// applies the 32-element post-MMA column permutation (`scale_perm_single`).
pub fn marlin_permute_scales_k_by_n(
    scales_k_by_n: &[u16],
    size_k: usize,
    size_n: usize,
    group_size: usize,
) -> Vec<u16> {
    let eff_gs = if group_size == 0 { size_k } else { group_size };
    let num_groups = size_k / eff_gs;
    assert_eq!(scales_k_by_n.len(), num_groups * size_n);
    let mut out = vec![0u16; scales_k_by_n.len()];
    let is_grouped_marlin = group_size > 0 && group_size <= 128;
    if is_grouped_marlin {
        let mut scale_perm = [0usize; 64];
        for i in 0..8 {
            for j in 0..8 {
                scale_perm[i * 8 + j] = i + 8 * j;
            }
        }
        for (chunk_idx, src_chunk) in scales_k_by_n.as_chunks::<64>().0.iter().enumerate() {
            let dst_base = chunk_idx * 64;
            for (i, &p) in scale_perm.iter().enumerate() {
                out[dst_base + i] = src_chunk[p];
            }
        }
    } else {
        let mut scale_perm_single = [0usize; 32];
        const BASE: [usize; 8] = [0, 1, 8, 9, 16, 17, 24, 25];
        for i in 0..4 {
            for (j, &b) in BASE.iter().enumerate() {
                scale_perm_single[i * 8 + j] = 2 * i + b;
            }
        }
        for (chunk_idx, src_chunk) in scales_k_by_n.as_chunks::<32>().0.iter().enumerate() {
            let dst_base = chunk_idx * 32;
            for (i, &p) in scale_perm_single.iter().enumerate() {
                out[dst_base + i] = src_chunk[p];
            }
        }
    }
    out
}

/// Transposes an N-major `[N, K / group_size]` scale tensor into `[K / group_size, N]`
/// and applies `marlin_permute_scales_k_by_n`.
pub fn marlin_permute_scales_from_n_major(
    scales_n_by_kg: &[u16],
    size_k: usize,
    size_n: usize,
    group_size: usize,
) -> Vec<u16> {
    let eff_gs = if group_size == 0 { size_k } else { group_size };
    let num_groups = size_k / eff_gs;
    assert_eq!(scales_n_by_kg.len(), size_n * num_groups);
    let mut k_by_n = vec![0u16; num_groups * size_n];
    for n in 0..size_n {
        for g in 0..num_groups {
            k_by_n[g * size_n + n] = scales_n_by_kg[n * num_groups + g];
        }
    }
    marlin_permute_scales_k_by_n(&k_by_n, size_k, size_n, group_size)
}

/// Dequantizes an N-major `[N, K/8]` `U4B8` weight + `[N, K/group_size]` `FP16` scale tensor
/// into an N-major `[N, K]` `FP16` matrix (used for skinny projections with `N < 64` or `N % 64 != 0`).
pub fn dequant_u4b8_n_major_to_f16(
    packed_n_by_k8: &[u32],
    scales_n_by_kg: &[u16],
    size_n: usize,
    size_k: usize,
    group_size: usize,
) -> Vec<u16> {
    let words_per_row = size_k / 8;
    let groups_per_row = size_k / group_size;
    assert_eq!(packed_n_by_k8.len(), size_n * words_per_row);
    assert_eq!(scales_n_by_kg.len(), size_n * groups_per_row);
    let mut out = vec![0u16; size_n * size_k];
    let dequant_rows = |row_start: usize, dst_rows: &mut [u16]| {
        for (rel_n, dst_row) in dst_rows.chunks_exact_mut(size_k).enumerate() {
            let n = row_start + rel_n;
            let row_words = &packed_n_by_k8[n * words_per_row..(n + 1) * words_per_row];
            let row_scales = &scales_n_by_kg[n * groups_per_row..(n + 1) * groups_per_row];
            for (w_idx, &word) in row_words.iter().enumerate() {
                for b in 0..8 {
                    let k = w_idx * 8 + b;
                    let nib = ((word >> (b * 4)) & 0xF) as i32 - 8;
                    let scale = f16_to_f32(row_scales[k / group_size]);
                    dst_row[k] = f32_to_f16((nib as f32) * scale);
                }
            }
        }
    };
    if size_n < 2048 {
        dequant_rows(0, &mut out);
    } else {
        let num_threads = std::thread::available_parallelism()
            .map(|n| n.get().clamp(1, 8))
            .unwrap_or(4);
        let rows_per_chunk = size_n.div_ceil(num_threads).max(1);
        std::thread::scope(|s| {
            for (chunk_idx, dst_chunk) in out.chunks_mut(rows_per_chunk * size_k).enumerate() {
                let row_start = chunk_idx * rows_per_chunk;
                s.spawn(move || dequant_rows(row_start, dst_chunk));
            }
        });
    }
    out
}

/// Transposes a K-major `[K/8, N]` 4-bit GPTQ `qweight` tensor into N-major `[N, K/8]` `u32` words.
pub fn transpose_gptq_qweight_k8_n_to_n_major(
    qweight_k8_by_n: &[u32],
    size_k: usize,
    size_n: usize,
) -> Vec<u32> {
    let k8 = size_k / 8;
    assert_eq!(qweight_k8_by_n.len(), k8 * size_n);
    let mut out = vec![0u32; size_n * k8];
    const TILE: usize = 32;
    let mut tile = [[0u32; TILE]; TILE];
    for kr_base in (0..k8).step_by(TILE) {
        let kr_len = (k8 - kr_base).min(TILE);
        for n_base in (0..size_n).step_by(TILE) {
            let n_len = (size_n - n_base).min(TILE);
            for (tk, tile_row) in tile.iter_mut().enumerate().take(kr_len) {
                let src = &qweight_k8_by_n[(kr_base + tk) * size_n + n_base..];
                for (tn, slot) in tile_row.iter_mut().enumerate().take(n_len) {
                    *slot = src[tn];
                }
            }
            for tn in 0..n_len {
                let dst = &mut out[(n_base + tn) * k8 + kr_base..];
                for (tk, &row) in tile.iter().enumerate().take(kr_len) {
                    dst[tk] = row[tn];
                }
            }
        }
    }
    out
}

/// Transposes a K-major `[K/4, N]` 8-bit GPTQ `qweight` tensor into N-major `[N, K/4]` `u32` words.
pub fn transpose_gptq_qweight_k4_n_to_n_major(
    qweight_k4_by_n: &[u32],
    size_k: usize,
    size_n: usize,
) -> Vec<u32> {
    let k4 = size_k / 4;
    assert_eq!(qweight_k4_by_n.len(), k4 * size_n);
    let mut out = vec![0u32; size_n * k4];
    const TILE: usize = 32;
    let mut tile = [[0u32; TILE]; TILE];
    for kr_base in (0..k4).step_by(TILE) {
        let kr_len = (k4 - kr_base).min(TILE);
        for n_base in (0..size_n).step_by(TILE) {
            let n_len = (size_n - n_base).min(TILE);
            for (tk, tile_row) in tile.iter_mut().enumerate().take(kr_len) {
                let src = &qweight_k4_by_n[(kr_base + tk) * size_n + n_base..];
                for (tn, slot) in tile_row.iter_mut().enumerate().take(n_len) {
                    *slot = src[tn];
                }
            }
            for tn in 0..n_len {
                let dst = &mut out[(n_base + tn) * k4 + kr_base..];
                for (tk, &row) in tile.iter().enumerate().take(kr_len) {
                    dst[tk] = row[tn];
                }
            }
        }
    }
    out
}

/// Unpacks an AutoAWQ `gemm` `[K, N/8]` `qweight` tensor (interleaved `[0, 2, 4, 6, 1, 3, 5, 7]`
/// along `N`) into N-major `[N, K/8]` `u32` words (8 consecutive K-nibbles per `u32`).
pub fn unpack_awq_qweight_k_n8_to_n_major(
    qweight_k_by_n8: &[u32],
    size_k: usize,
    size_n: usize,
) -> Vec<u32> {
    const UNDO_AWQ: [usize; 8] = [0, 4, 1, 5, 2, 6, 3, 7];
    let k8 = size_k / 8;
    let n8 = size_n / 8;
    assert_eq!(qweight_k_by_n8.len(), size_k * n8);
    let mut out = vec![0u32; size_n * k8];
    for k_word in 0..k8 {
        for b in 0..8 {
            let k = k_word * 8 + b;
            let src_row = &qweight_k_by_n8[k * n8..(k + 1) * n8];
            let shift_out = b * 4;
            for (n_pack, &awq_word) in src_row.iter().enumerate() {
                let n_base = n_pack * 8;
                for (c, &src_slot) in UNDO_AWQ.iter().enumerate() {
                    let nib = (awq_word >> (src_slot * 4)) & 0xF;
                    out[(n_base + c) * k8 + k_word] |= nib << shift_out;
                }
            }
        }
    }
    out
}

/// Unpacks an AutoAWQ `gemm` `[K / group_size, N / 8]` `qzeros` tensor (interleaved
/// `[0, 2, 4, 6, 1, 3, 5, 7]` along `N`) into N-major `[N / 8, K / group_size]` `u32` words
/// matching `compressed-tensors` `.weight_zero_point` layout (8 consecutive N-rows per `u32`).
pub fn unpack_awq_qzeros_kg_n8_to_n_major(
    qzeros_kg_by_n8: &[u32],
    size_k: usize,
    size_n: usize,
    group_size: usize,
) -> Vec<u32> {
    const UNDO_AWQ: [usize; 8] = [0, 4, 1, 5, 2, 6, 3, 7];
    let num_groups = size_k / group_size;
    let n8 = size_n / 8;
    assert_eq!(qzeros_kg_by_n8.len(), num_groups * n8);
    let mut out = vec![0u32; n8 * num_groups];
    for g in 0..num_groups {
        for n_pack in 0..n8 {
            let awq_word = qzeros_kg_by_n8[g * n8 + n_pack];
            let mut packed_col = 0u32;
            for (c, &src_slot) in UNDO_AWQ.iter().enumerate() {
                let nib = (awq_word >> (src_slot * 4)) & 0xF;
                packed_col |= nib << (c * 4);
            }
            out[n_pack * num_groups + g] = packed_col;
        }
    }
    out
}

/// Permutes an N-major `[N / 8, K / group_size]` packed 4-bit zero-point tensor into `vLLM`
/// Marlin `kU4` zero-point layout (`[K / group_size, N / 8]` `u32` words with `scale_perm` +
/// `[0, 2, 4, 6, 1, 3, 5, 7]` interleave).
pub fn marlin_permute_zero_points_from_n_major(
    qzeros_n8_by_kg: &[u32],
    size_k: usize,
    size_n: usize,
    group_size: usize,
) -> Vec<u32> {
    const INTERLEAVE: [usize; 8] = [0, 2, 4, 6, 1, 3, 5, 7];
    let num_groups = size_k / group_size;
    let n8 = size_n / 8;
    assert_eq!(size_n % 64, 0);
    assert_eq!(qzeros_n8_by_kg.len(), n8 * num_groups);

    let mut zp_g_by_n = vec![0u8; num_groups * size_n];
    for n_pack in 0..n8 {
        for g in 0..num_groups {
            let w = qzeros_n8_by_kg[n_pack * num_groups + g];
            let n_base = n_pack * 8;
            for b in 0..8 {
                zp_g_by_n[g * size_n + n_base + b] = ((w >> (b * 4)) & 0xF) as u8;
            }
        }
    }

    let mut permed = vec![0u8; zp_g_by_n.len()];
    let mut scale_perm = [0usize; 64];
    for i in 0..8 {
        for j in 0..8 {
            scale_perm[i * 8 + j] = i + 8 * j;
        }
    }
    for (chunk_idx, src_chunk) in zp_g_by_n.as_chunks::<64>().0.iter().enumerate() {
        let dst_base = chunk_idx * 64;
        for (i, &p) in scale_perm.iter().enumerate() {
            permed[dst_base + i] = src_chunk[p];
        }
    }

    let mut out = vec![0u32; num_groups * n8];
    for (word_idx, chunk8) in permed.as_chunks::<8>().0.iter().enumerate() {
        let mut packed = 0u32;
        for (i, &src_i) in INTERLEAVE.iter().enumerate() {
            packed |= ((chunk8[src_i] & 0xF) as u32) << (i * 4);
        }
        out[word_idx] = packed;
    }
    out
}

/// Dequantizes an N-major `[N, K/8]` asymmetric `U4` weight + `[N, K/group_size]` `FP16` scale
/// + `[N/8, K/group_size]` packed 4-bit zero-point tensor into an N-major `[N, K]` `FP16` matrix.
pub fn dequant_u4_zp_n_major_to_f16(
    packed_n_by_k8: &[u32],
    scales_n_by_kg: &[u16],
    qzeros_n8_by_kg: &[u32],
    size_n: usize,
    size_k: usize,
    group_size: usize,
) -> Vec<u16> {
    let words_per_row = size_k / 8;
    let groups_per_row = size_k / group_size;
    let n8 = size_n / 8;
    assert_eq!(size_n % 8, 0);
    assert_eq!(packed_n_by_k8.len(), size_n * words_per_row);
    assert_eq!(scales_n_by_kg.len(), size_n * groups_per_row);
    assert_eq!(qzeros_n8_by_kg.len(), n8 * groups_per_row);
    let mut out = vec![0u16; size_n * size_k];
    for n in 0..size_n {
        let n_pack = n >> 3;
        let n_shift = (n & 7) * 4;
        for w_idx in 0..words_per_row {
            let word = packed_n_by_k8[n * words_per_row + w_idx];
            for b in 0..8 {
                let k = w_idx * 8 + b;
                let g = k / group_size;
                let nib = ((word >> (b * 4)) & 0xF) as i32;
                let zp = ((qzeros_n8_by_kg[n_pack * groups_per_row + g] >> n_shift) & 0xF) as i32;
                let scale = f16_to_f32(scales_n_by_kg[n * groups_per_row + g]);
                out[n * size_k + k] = f32_to_f16(((nib - zp) as f32) * scale);
            }
        }
    }
    out
}

/// Dequantizes an N-major `[N, K/4]` symmetric `U8B128` weight + `[N, K/group_size]` `FP16` scale
/// tensor into an N-major `[N, K]` `FP16` matrix.
pub fn dequant_u8b128_n_major_to_f16(
    packed_n_by_k4: &[u32],
    scales_n_by_kg: &[u16],
    size_n: usize,
    size_k: usize,
    group_size: usize,
) -> Vec<u16> {
    let eff_gs = if group_size == 0 { size_k } else { group_size };
    let words_per_row = size_k / 4;
    let groups_per_row = size_k / eff_gs;
    assert_eq!(packed_n_by_k4.len(), size_n * words_per_row);
    assert_eq!(scales_n_by_kg.len(), size_n * groups_per_row);
    let mut out = vec![0u16; size_n * size_k];
    let dequant_rows = |row_start: usize, dst_rows: &mut [u16]| {
        for (rel_n, dst_row) in dst_rows.chunks_exact_mut(size_k).enumerate() {
            let n = row_start + rel_n;
            let row_words = &packed_n_by_k4[n * words_per_row..(n + 1) * words_per_row];
            let row_scales = &scales_n_by_kg[n * groups_per_row..(n + 1) * groups_per_row];
            for (w_idx, &word) in row_words.iter().enumerate() {
                for b in 0..4 {
                    let k = w_idx * 4 + b;
                    let q = ((word >> (b * 8)) & 0xFF) as i32 - 128;
                    let scale = f16_to_f32(row_scales[k / eff_gs]);
                    dst_row[k] = f32_to_f16((q as f32) * scale);
                }
            }
        }
    };
    if size_n < 2048 {
        dequant_rows(0, &mut out);
    } else {
        let num_threads = std::thread::available_parallelism()
            .map(|n| n.get().clamp(1, 8))
            .unwrap_or(4);
        let rows_per_chunk = size_n.div_ceil(num_threads).max(1);
        std::thread::scope(|s| {
            for (chunk_idx, dst_chunk) in out.chunks_mut(rows_per_chunk * size_k).enumerate() {
                let row_start = chunk_idx * rows_per_chunk;
                s.spawn(move || dequant_rows(row_start, dst_chunk));
            }
        });
    }
    out
}

/// Validates that an optional GPTQ `.g_idx` tensor represents static sequential group assignment
/// (`g_idx[k] == k / group_size`, or all `0` for channelwise) rather than an unapplied `desc_act`
/// channel permutation.
pub fn validate_static_gptq_g_idx(
    g_idx_words: &[u32],
    size_k: usize,
    group_size: usize,
    tensor_name: &str,
) -> io::Result<()> {
    if g_idx_words.len() != size_k {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!(
                "invalid g_idx length {} for '{tensor_name}' (expected K={size_k})",
                g_idx_words.len()
            ),
        ));
    }
    if g_idx_words.iter().all(|&x| x == 0) {
        return Ok(());
    }
    let eff_gs = if group_size == 0 { size_k } else { group_size };
    for (k, &raw_g) in g_idx_words.iter().enumerate() {
        let expected = (k / eff_gs) as u32;
        if raw_g != expected {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!(
                    "tensor '{tensor_name}' contains non-sequential g_idx[{k}]={} (expected {expected}); activation-order (desc_act) channel permutation is not supported",
                    raw_g as i32
                ),
            ));
        }
    }
    Ok(())
}
