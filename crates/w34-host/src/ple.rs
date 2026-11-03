//! Per-Layer Embedding (`PLE`) n-gram hasher, zero-copy `F8_E4M3` `mmap` table,
//! and `Qwen3.8-Flash-Next` (`qwen4_exp`) capacity planner for `TP=1` and `TP=2`.

use std::collections::HashMap;
use std::io;
use std::os::unix::io::AsRawFd;
use std::ptr::NonNull;

use crate::cache::KvFormatPolicy;
use crate::cuda::{bf16_to_f32, f32_to_bf16, f32_to_f16};
use crate::loader::{LayerKind, ModelConfig, SafetensorsArchive};

const PROT_READ: i32 = 0x1;
const MAP_PRIVATE: i32 = 0x02;
const MADV_RANDOM: i32 = 1;
const MADV_WILLNEED: i32 = 3;

unsafe extern "C" {
    fn mmap(
        addr: *mut core::ffi::c_void,
        length: usize,
        prot: i32,
        flags: i32,
        fd: i32,
        offset: i64,
    ) -> *mut core::ffi::c_void;
    fn munmap(addr: *mut core::ffi::c_void, length: usize) -> i32;
    fn madvise(addr: *mut core::ffi::c_void, length: usize, advice: i32) -> i32;
}

const SPLITMIX_GAMMA: u64 = 0x9E37_79B9_7F4A_7C15;
const SPLITMIX_M1: u64 = 0xBF58_476D_1CE4_E5B9;
const SPLITMIX_M2: u64 = 0x94D0_49BB_1331_11EB;
const PLE_LAYER_PRIME: u64 = 10_007;

/// Standard `splitmix64` bijection used by `Qwen3.8-Flash-Next` PLE n-gram hashing.
#[inline]
pub const fn splitmix64(mut value: u64) -> u64 {
    value = value.wrapping_add(SPLITMIX_GAMMA);
    value = (value ^ (value >> 30)).wrapping_mul(SPLITMIX_M1);
    value = (value ^ (value >> 27)).wrapping_mul(SPLITMIX_M2);
    value ^ (value >> 31)
}

#[inline]
const fn mod_pow_u64(mut base: u64, mut exp: u64, modulus: u64) -> u64 {
    if modulus == 1 {
        return 0;
    }
    let mut result = 1u64;
    base %= modulus;
    while exp > 0 {
        if (exp & 1) != 0 {
            result = ((result as u128 * base as u128) % (modulus as u128)) as u64;
        }
        exp >>= 1;
        base = ((base as u128 * base as u128) % (modulus as u128)) as u64;
    }
    result
}

/// Deterministic 64-bit Miller-Rabin primality test matching `Qwen4Exp` PLE prime table construction.
pub const fn is_prime_64(value: u64) -> bool {
    if value < 2 {
        return false;
    }
    if value.is_multiple_of(2) {
        return value == 2;
    }
    const SMALL_PRIMES: [u64; 11] = [3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37];
    let mut i = 0;
    while i < SMALL_PRIMES.len() {
        let p = SMALL_PRIMES[i];
        if value == p {
            return true;
        }
        if value.is_multiple_of(p) {
            return false;
        }
        i += 1;
    }
    let mut d = value - 1;
    let mut s = 0u32;
    while d.is_multiple_of(2) {
        d /= 2;
        s += 1;
    }
    const BASES: [u64; 7] = [2, 325, 9375, 28178, 450_775, 9_780_504, 1_795_265_022];
    let mut b_idx = 0;
    while b_idx < BASES.len() {
        let base = BASES[b_idx];
        if !base.is_multiple_of(value) {
            let mut x = mod_pow_u64(base, d, value);
            if x != 1 && x != value - 1 {
                let mut r = 0u32;
                let mut composite = true;
                while r + 1 < s {
                    x = ((x as u128 * x as u128) % (value as u128)) as u64;
                    if x == value - 1 {
                        composite = false;
                        break;
                    }
                    r += 1;
                }
                if composite {
                    return false;
                }
            }
        }
        b_idx += 1;
    }
    true
}

/// Returns the `count`-th (1-based) prime strictly greater than `start`.
pub fn nth_prime_after(start: u64, count: usize) -> u64 {
    assert!(count >= 1, "nth_prime_after requires count >= 1");
    let mut candidate = start.saturating_add(1).max(2);
    if candidate > 2 && candidate.is_multiple_of(2) {
        candidate += 1;
    }
    let mut found = 0usize;
    loop {
        if is_prime_64(candidate) {
            found += 1;
            if found == count {
                return candidate;
            }
        }
        candidate += if candidate == 2 { 1 } else { 2 };
    }
}

/// Converts an OCP `F8_E4M3` (`e4m3fn`) byte into IEEE-754 `f32`.
#[inline]
pub fn f8_e4m3_to_f32(byte: u8) -> f32 {
    let sign = if (byte & 0x80) != 0 { -1.0f32 } else { 1.0f32 };
    let exp = (byte >> 3) & 0x0f;
    let mant = byte & 0x07;
    if exp == 0x0f && mant == 0x07 {
        0.0f32
    } else if exp == 0 {
        sign * (mant as f32) * (1.0f32 / 512.0f32)
    } else {
        let scale = f32::from_bits(((exp as u32) + 127 - 7) << 23);
        sign * (1.0f32 + (mant as f32) * 0.125f32) * scale
    }
}

/// Precomputed 256-entry lookup table for `F8_E4M3 * weight_scale`.
#[derive(Clone, Debug)]
pub struct F8E4m3ScaleLut {
    pub weight_scale: f32,
    pub f32_lut: [f32; 256],
    pub f16_lut: [u16; 256],
    pub bf16_lut: [u16; 256],
}

impl F8E4m3ScaleLut {
    pub fn new(weight_scale: f32) -> Self {
        let mut f32_lut = [0.0f32; 256];
        let mut f16_lut = [0u16; 256];
        let mut bf16_lut = [0u16; 256];
        for b in 0..256usize {
            let val = f8_e4m3_to_f32(b as u8) * weight_scale;
            f32_lut[b] = val;
            f16_lut[b] = f32_to_f16(val);
            bf16_lut[b] = f32_to_bf16(val);
        }
        Self {
            weight_scale,
            f32_lut,
            f16_lut,
            bf16_lut,
        }
    }
}

/// Deterministic PLE bigram/trigram multi-head prime-modulo hasher.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PleNgramHasher {
    pub ple_layer_id: usize,
    pub ple_dense_layer_id: usize,
    pub ngram_size: usize,
    pub ngram_heads: usize,
    pub bigram_heads: usize,
    pub trigram_heads: usize,
    pub head_dim: usize,
    pub split_ngram_parts: usize,
    pub eos_token_id: i32,
    pub layer_multipliers: Vec<i64>,
    pub ngram_heads_vocab_sizes: Vec<i64>,
    pub ngram_heads_offsets: Vec<i64>,
    pub padded_vocab_size: usize,
    pub shard_rows: usize,
}

impl PleNgramHasher {
    pub fn from_config(cfg: &ModelConfig, ple_layer_id: usize, ple_dense_layer_id: usize) -> Self {
        let ngram_size = cfg.ngram_size.max(1);
        let ngram_heads = (cfg.heads_per_ngram * ngram_size.saturating_sub(1)).max(1);
        let bigram_heads = cfg.heads_per_ngram.min(ngram_heads);
        let trigram_heads = ngram_heads - bigram_heads;
        let head_dim = cfg.ple_embed_dim / ngram_heads;
        let split_ngram_parts = cfg.split_ngram_parts.max(1);
        let unigram_vocab_size = cfg.vocab_size.max(1) as u64;
        let max_multiplier = ((1u64 << 63) - 1) / unigram_vocab_size;
        let half_bound = (max_multiplier / 2).max(1);
        let base_seed = cfg
            .ple_seed
            .wrapping_add(PLE_LAYER_PRIME.wrapping_mul(ple_dense_layer_id as u64));

        let mut layer_multipliers = Vec::with_capacity(ngram_size);
        for index in 0..ngram_size {
            let value = base_seed.wrapping_add(SPLITMIX_GAMMA.wrapping_mul((index + 1) as u64));
            let m = 2 * (splitmix64(value) % half_bound) + 1;
            layer_multipliers.push(m as i64);
        }

        let base_vocab = cfg.ngram_vocab_size_base.saturating_sub(1) as u64;
        let mut ngram_heads_vocab_sizes = Vec::with_capacity(ngram_heads);
        let mut ngram_heads_offsets = Vec::with_capacity(ngram_heads);
        let mut offset = 0i64;
        for local_head in 0..ngram_heads {
            let global_head = ple_dense_layer_id * ngram_heads + local_head;
            let size = nth_prime_after(base_vocab, global_head + 1) as i64;
            ngram_heads_vocab_sizes.push(size);
            ngram_heads_offsets.push(offset);
            offset += size;
        }
        let divisor = cfg.make_ngram_vocab_size_divisible_by.max(1);
        let padded_vocab_size = (offset as usize).div_ceil(divisor) * divisor;
        let shard_rows = padded_vocab_size.div_ceil(split_ngram_parts);
        let eos_token_id = cfg.eos_token_ids.first().copied().unwrap_or(248_044);

        Self {
            ple_layer_id,
            ple_dense_layer_id,
            ngram_size,
            ngram_heads,
            bigram_heads,
            trigram_heads,
            head_dim,
            split_ngram_parts,
            eos_token_id,
            layer_multipliers,
            ngram_heads_vocab_sizes,
            ngram_heads_offsets,
            padded_vocab_size,
            shard_rows,
        }
    }

    /// Loads or validates the hasher tables against a `SafetensorsArchive`.
    pub fn from_archive(
        archive: &SafetensorsArchive,
        cfg: &ModelConfig,
        ple_layer_id: usize,
        ple_dense_layer_id: usize,
    ) -> io::Result<Self> {
        let mut hasher = Self::from_config(cfg, ple_layer_id, ple_dense_layer_id);
        let prefix = format!("model.language_model.layers.{ple_layer_id}.ple.ple_embedding");
        let mult_name = format!("{prefix}.layer_multipliers");
        if archive.resolve_name(&mult_name).is_some() {
            let mult = archive.read_i64(&mult_name)?;
            if mult != hasher.layer_multipliers {
                hasher.layer_multipliers = mult;
            }
        }
        let sizes_name = format!("{prefix}.ngram_heads_vocab_sizes");
        if archive.resolve_name(&sizes_name).is_some() {
            let sizes = archive.read_i64(&sizes_name)?;
            if sizes != hasher.ngram_heads_vocab_sizes {
                hasher.ngram_heads_vocab_sizes = sizes;
            }
        }
        let offsets_name = format!("{prefix}.ngram_heads_offsets");
        if archive.resolve_name(&offsets_name).is_some() {
            let offsets = archive.read_i64(&offsets_name)?;
            if offsets != hasher.ngram_heads_offsets {
                hasher.ngram_heads_offsets = offsets;
            }
        }
        Ok(hasher)
    }

    /// Computes `[num_tokens, ngram_heads]` global row IDs and updates each request's
    /// rolling 2-token `ngram_context` (`[t - 2, t - 1]`) in-place.
    pub fn compute_ngram_ids(
        &self,
        input_ids: &[i32],
        query_start_loc: &[i32],
        ngram_context: &mut [[i32; 2]],
        out_ids: &mut [i64],
    ) {
        let num_reqs = query_start_loc.len().saturating_sub(1);
        assert_eq!(ngram_context.len(), num_reqs);
        assert_eq!(out_ids.len(), input_ids.len() * self.ngram_heads);
        let m0 = self.layer_multipliers[0];
        let m1 = self.layer_multipliers.get(1).copied().unwrap_or(0);
        let m2 = self.layer_multipliers.get(2).copied().unwrap_or(0);
        let eos = self.eos_token_id;

        for req_idx in 0..num_reqs {
            let start = query_start_loc[req_idx].max(0) as usize;
            let end = (query_start_loc[req_idx + 1].max(0) as usize).min(input_ids.len());
            if start >= end {
                continue;
            }
            let mut w_prev2 = ngram_context[req_idx][0];
            let mut w_prev1 = ngram_context[req_idx][1];

            for (tok_pos, &raw_tok) in input_ids[start..end].iter().enumerate() {
                let tok = raw_tok.max(0);
                let s0 = tok as i64;
                let s1 = w_prev1.max(0) as i64;
                let s2 = if w_prev1 == eos {
                    eos as i64
                } else {
                    w_prev2.max(0) as i64
                };

                let bigram_hash = s0.wrapping_mul(m0) ^ s1.wrapping_mul(m1);
                let trigram_hash = bigram_hash ^ s2.wrapping_mul(m2);
                let row_out = &mut out_ids[(start + tok_pos) * self.ngram_heads
                    ..(start + tok_pos + 1) * self.ngram_heads];

                for (h, slot) in row_out.iter_mut().enumerate().take(self.bigram_heads) {
                    let mod_p = self.ngram_heads_vocab_sizes[h];
                    *slot = bigram_hash.rem_euclid(mod_p) + self.ngram_heads_offsets[h];
                }
                for (h, slot) in row_out
                    .iter_mut()
                    .enumerate()
                    .take(self.ngram_heads)
                    .skip(self.bigram_heads)
                {
                    let mod_p = self.ngram_heads_vocab_sizes[h];
                    *slot = trigram_hash.rem_euclid(mod_p) + self.ngram_heads_offsets[h];
                }

                w_prev2 = if tok == eos { eos } else { w_prev1 };
                w_prev1 = tok;
            }

            ngram_context[req_idx] = [w_prev2, w_prev1];
        }
    }

    /// Computes `[num_tokens, ngram_heads]` global row IDs while reading and updating per-slot
    /// `slot_contexts` (`StateBundlePool::ple_ngram_context`) across `state_in`, `state_out`,
    /// `spec_state_indices`, and `MambaRadixCache` `checkpoint_dst_indices`.
    #[allow(clippy::too_many_arguments)]
    pub fn compute_ngram_ids_for_slots(
        &self,
        input_ids: &[i32],
        query_start_loc: &[i32],
        state_in_indices: &[i32],
        state_out_indices: &[i32],
        has_initial_state: &[u8],
        spec_state_indices: &[i32],
        max_verify_tokens: usize,
        checkpoint_dst_indices: &[i32],
        checkpoint_token_offsets: &[i32],
        max_checkpoints_per_req: usize,
        slot_contexts: &mut [[i32; 2]],
        out_ids: &mut [i64],
    ) {
        let num_reqs = query_start_loc.len().saturating_sub(1);
        assert_eq!(out_ids.len(), input_ids.len() * self.ngram_heads);
        let m0 = self.layer_multipliers[0];
        let m1 = self.layer_multipliers.get(1).copied().unwrap_or(0);
        let m2 = self.layer_multipliers.get(2).copied().unwrap_or(0);
        let eos = self.eos_token_id;

        for req_idx in 0..num_reqs {
            let start = query_start_loc[req_idx].max(0) as usize;
            let end = (query_start_loc[req_idx + 1].max(0) as usize).min(input_ids.len());
            if start >= end {
                continue;
            }
            let in_slot = state_in_indices.get(req_idx).copied().unwrap_or(-1);
            let has_init = has_initial_state.get(req_idx).copied().unwrap_or(0) != 0;
            let [mut w_prev2, mut w_prev1] =
                if has_init && in_slot >= 0 && (in_slot as usize) < slot_contexts.len() {
                    slot_contexts[in_slot as usize]
                } else {
                    [eos, eos]
                };

            for (tok_pos, &raw_tok) in input_ids[start..end].iter().enumerate() {
                let tok = raw_tok.max(0);
                let s0 = tok as i64;
                let s1 = w_prev1.max(0) as i64;
                let s2 = if w_prev1 == eos {
                    eos as i64
                } else {
                    w_prev2.max(0) as i64
                };

                let bigram_hash = s0.wrapping_mul(m0) ^ s1.wrapping_mul(m1);
                let trigram_hash = bigram_hash ^ s2.wrapping_mul(m2);
                let row_out = &mut out_ids[(start + tok_pos) * self.ngram_heads
                    ..(start + tok_pos + 1) * self.ngram_heads];

                for (h, slot) in row_out.iter_mut().enumerate().take(self.bigram_heads) {
                    let mod_p = self.ngram_heads_vocab_sizes[h];
                    *slot = bigram_hash.rem_euclid(mod_p) + self.ngram_heads_offsets[h];
                }
                for (h, slot) in row_out
                    .iter_mut()
                    .enumerate()
                    .take(self.ngram_heads)
                    .skip(self.bigram_heads)
                {
                    let mod_p = self.ngram_heads_vocab_sizes[h];
                    *slot = trigram_hash.rem_euclid(mod_p) + self.ngram_heads_offsets[h];
                }

                w_prev2 = if tok == eos { eos } else { w_prev1 };
                w_prev1 = tok;

                if max_verify_tokens > 0 && tok_pos < max_verify_tokens {
                    let idx = req_idx * max_verify_tokens + tok_pos;
                    if let Some(&spec_slot) = spec_state_indices.get(idx)
                        && spec_slot >= 0
                        && (spec_slot as usize) < slot_contexts.len()
                    {
                        slot_contexts[spec_slot as usize] = [w_prev2, w_prev1];
                    }
                }
                if max_checkpoints_per_req > 0 {
                    for e in 0..max_checkpoints_per_req {
                        let idx = req_idx * max_checkpoints_per_req + e;
                        if checkpoint_token_offsets.get(idx).copied() == Some((tok_pos + 1) as i32)
                            && let Some(&chk_slot) = checkpoint_dst_indices.get(idx)
                            && chk_slot >= 0
                            && (chk_slot as usize) < slot_contexts.len()
                        {
                            slot_contexts[chk_slot as usize] = [w_prev2, w_prev1];
                        }
                    }
                }
            }

            if max_verify_tokens == 0 {
                let out_slot = state_out_indices.get(req_idx).copied().unwrap_or(in_slot);
                if out_slot >= 0 && (out_slot as usize) < slot_contexts.len() {
                    slot_contexts[out_slot as usize] = [w_prev2, w_prev1];
                }
            }
        }
    }
}

struct MmapRegion {
    ptr: NonNull<u8>,
    len: usize,
}

// SAFETY: Read-only `PROT_READ` memory mapping backed by immutable safetensors files.
unsafe impl Send for MmapRegion {}
// SAFETY: Read-only `PROT_READ` memory mapping backed by immutable safetensors files.
unsafe impl Sync for MmapRegion {}

impl Drop for MmapRegion {
    fn drop(&mut self) {
        if self.len > 0 {
            // SAFETY: `self.ptr` and `self.len` were returned by a successful `mmap`.
            unsafe {
                let _ = munmap(self.ptr.as_ptr().cast(), self.len);
            }
        }
    }
}

#[derive(Clone, Copy)]
struct ShardView {
    mmap_idx: usize,
    offset_bytes: usize,
    rows: usize,
}

/// Zero-copy read-only `mmap` table over the 128 `F8_E4M3` PLE n-gram embedding shards.
///
/// Consumes `0 B` of anonymous system RAM and `0 B` of GPU VRAM: the OS page cache serves hot
/// n-gram pages directly from NVMe with `MADV_RANDOM` + coalesced `MADV_WILLNEED` prefaulting.
pub struct PleMmapTable {
    mmaps: Vec<MmapRegion>,
    shards: Vec<ShardView>,
    pub shard_rows: usize,
    pub head_dim: usize,
    pub lut: F8E4m3ScaleLut,
}

impl PleMmapTable {
    pub fn open(
        archive: &SafetensorsArchive,
        cfg: &ModelConfig,
        ple_layer_id: usize,
    ) -> io::Result<Self> {
        let split_parts = cfg.split_ngram_parts.max(1);
        let ngram_heads = (cfg.heads_per_ngram * cfg.ngram_size.saturating_sub(1)).max(1);
        let head_dim = cfg.ple_embed_dim / ngram_heads;
        let prefix =
            format!("model.language_model.layers.{ple_layer_id}.ple.ple_embedding.ngram_embedding");
        let scale_name = format!("{prefix}.weight_scale");
        let weight_scale = if let Ok(info) = archive.tensor_info(&scale_name) {
            let raw = archive.read_bytes(&scale_name)?;
            match info.dtype.as_str() {
                "BF16" if raw.len() >= 2 => bf16_to_f32(u16::from_le_bytes([raw[0], raw[1]])),
                "F16" if raw.len() >= 2 => {
                    crate::cuda::f16_to_f32(u16::from_le_bytes([raw[0], raw[1]]))
                }
                "F32" if raw.len() >= 4 => f32::from_le_bytes([raw[0], raw[1], raw[2], raw[3]]),
                other => {
                    return Err(io::Error::new(
                        io::ErrorKind::InvalidData,
                        format!("unsupported PLE weight_scale dtype {other}"),
                    ));
                }
            }
        } else {
            1.0f32
        };

        let mut file_to_mmap: HashMap<String, usize> = HashMap::new();
        let mut mmaps: Vec<MmapRegion> = Vec::new();
        let mut shards: Vec<ShardView> = Vec::with_capacity(split_parts);
        let mut shard_rows = 0usize;

        for part in 0..split_parts {
            let tname = format!("{prefix}.shard_{part}.weight");
            let info = archive.tensor_info(&tname)?;
            if info.dtype != "F8_E4M3" {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    format!("expected F8_E4M3 for {tname}, got {}", info.dtype),
                ));
            }
            if info.shape.len() != 2 || info.shape[1] != head_dim {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    format!(
                        "unexpected PLE shard shape {:?} for {tname} (expected [rows, {head_dim}])",
                        info.shape
                    ),
                ));
            }
            let rows = info.shape[0];
            if part == 0 {
                shard_rows = rows;
            } else if rows != shard_rows {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    format!(
                        "inconsistent PLE shard rows: shard_0={shard_rows}, shard_{part}={rows}"
                    ),
                ));
            }
            let mmap_idx = if let Some(&idx) = file_to_mmap.get(&info.shard_file) {
                idx
            } else {
                let file = archive.shard_file(&info.shard_file).ok_or_else(|| {
                    io::Error::new(
                        io::ErrorKind::NotFound,
                        format!(
                            "missing safetensors file for shard_file={}",
                            info.shard_file
                        ),
                    )
                })?;
                let file_len = file.metadata()?.len() as usize;
                let fd = file.as_raw_fd();
                // SAFETY: Valid file descriptor and length; `PROT_READ` + `MAP_PRIVATE` is read-only.
                let raw_ptr = unsafe {
                    mmap(
                        core::ptr::null_mut(),
                        file_len,
                        PROT_READ,
                        MAP_PRIVATE,
                        fd,
                        0,
                    )
                };
                if raw_ptr as isize == -1 {
                    return Err(io::Error::last_os_error());
                }
                // SAFETY: `raw_ptr` is a valid non-null mapping of `file_len` bytes.
                unsafe {
                    let _ = madvise(raw_ptr, file_len, MADV_RANDOM);
                }
                let ptr = NonNull::new(raw_ptr.cast::<u8>())
                    .ok_or_else(|| io::Error::other("mmap returned null pointer"))?;
                let idx = mmaps.len();
                mmaps.push(MmapRegion { ptr, len: file_len });
                file_to_mmap.insert(info.shard_file.clone(), idx);
                idx
            };

            shards.push(ShardView {
                mmap_idx,
                offset_bytes: info.offset as usize,
                rows,
            });
        }

        Ok(Self {
            mmaps,
            shards,
            shard_rows,
            head_dim,
            lut: F8E4m3ScaleLut::new(weight_scale),
        })
    }

    /// Gathers raw `F8_E4M3` bytes (`[ids.len(), head_dim]`) with coalesced `MADV_WILLNEED`
    /// prefaulting for batches of `>= 64` row lookups.
    pub fn gather_f8(&self, ids: &[i64], out_f8: &mut [u8]) {
        assert_eq!(out_f8.len(), ids.len() * self.head_dim);
        if ids.is_empty() {
            return;
        }
        const PAGE_SIZE: usize = 4096;
        if ids.len() >= 64 {
            let mut page_keys: Vec<(usize, usize)> = Vec::with_capacity(ids.len());
            for &id in ids {
                let uid = id.max(0) as usize;
                let shard_idx = (uid / self.shard_rows).min(self.shards.len() - 1);
                let local_row = (uid % self.shard_rows).min(self.shards[shard_idx].rows - 1);
                let view = self.shards[shard_idx];
                let byte_off = view.offset_bytes + local_row * self.head_dim;
                page_keys.push((view.mmap_idx, byte_off / PAGE_SIZE));
            }
            page_keys.sort_unstable();
            page_keys.dedup();
            let mut idx = 0usize;
            while idx < page_keys.len() {
                let (mmap_idx, start_page) = page_keys[idx];
                let mut end_page = start_page + 1;
                idx += 1;
                while idx < page_keys.len()
                    && page_keys[idx].0 == mmap_idx
                    && page_keys[idx].1 == end_page
                {
                    end_page += 1;
                    idx += 1;
                }
                let region = &self.mmaps[mmap_idx];
                let adv_off = start_page * PAGE_SIZE;
                let adv_len =
                    ((end_page - start_page) * PAGE_SIZE).min(region.len.saturating_sub(adv_off));
                if adv_len > 0 {
                    // SAFETY: `adv_off..adv_off + adv_len` is page-aligned and within `region`.
                    unsafe {
                        let _ = madvise(
                            region.ptr.as_ptr().add(adv_off).cast(),
                            adv_len,
                            MADV_WILLNEED,
                        );
                    }
                }
            }
        }

        for (i, &id) in ids.iter().enumerate() {
            let uid = id.max(0) as usize;
            let shard_idx = (uid / self.shard_rows).min(self.shards.len() - 1);
            let local_row = (uid % self.shard_rows).min(self.shards[shard_idx].rows - 1);
            let view = self.shards[shard_idx];
            let region = &self.mmaps[view.mmap_idx];
            let byte_off = view.offset_bytes + local_row * self.head_dim;
            // SAFETY: Bounds checked at `PleMmapTable::open` (`offset + rows * head_dim <= region.len`).
            let src = unsafe {
                core::slice::from_raw_parts(region.ptr.as_ptr().add(byte_off), self.head_dim)
            };
            out_f8[i * self.head_dim..(i + 1) * self.head_dim].copy_from_slice(src);
        }
    }

    /// Gathers and dequantizes rows into `FP16` (`u16` bits) via the 256-entry L1 LUT.
    pub fn gather_dequant_f16(&self, ids: &[i64], out_f16: &mut [u16]) {
        assert_eq!(out_f16.len(), ids.len() * self.head_dim);
        let mut raw = vec![0u8; ids.len() * self.head_dim];
        self.gather_f8(ids, &mut raw);
        for (dst, &b) in out_f16.iter_mut().zip(raw.iter()) {
            *dst = self.lut.f16_lut[b as usize];
        }
    }

    /// Gathers and dequantizes rows into `f32` via the 256-entry L1 LUT.
    pub fn gather_dequant_f32(&self, ids: &[i64], out_f32: &mut [f32]) {
        assert_eq!(out_f32.len(), ids.len() * self.head_dim);
        let mut raw = vec![0u8; ids.len() * self.head_dim];
        self.gather_f8(ids, &mut raw);
        for (dst, &b) in out_f32.iter_mut().zip(raw.iter()) {
            *dst = self.lut.f32_lut[b as usize];
        }
    }
}

/// Complete per-rank and host RAM memory plan for `Qwen3.8-Flash-Next` (`qwen4_exp`)
/// on single-card (`TP=1`) and dual-card (`TP=2 / EP=2`) configurations.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct QwenFlashNextCapacityPlan {
    pub tp_size: usize,
    pub total_vram_per_rank_bytes: usize,
    pub free_vram_per_rank_bytes: usize,
    pub dense_weights_per_rank_bytes: usize,
    pub mtp_vram_per_rank_bytes: usize,
    pub expert_slot_bytes: usize,
    pub hot_experts_per_layer_per_rank: usize,
    pub hot_expert_vram_per_rank_bytes: usize,
    pub cold_experts_per_layer_per_rank: usize,
    pub host_expert_pinned_ram_bytes: usize,
    pub ple_mmap_virtual_bytes: usize,
    pub state_slots: usize,
    pub bytes_per_state_slot: usize,
    pub state_pool_per_rank_bytes: usize,
    pub bytes_per_kv_page_per_rank: usize,
    pub kv_pages_per_rank: usize,
    pub kv_pool_per_rank_bytes: usize,
    pub scratch_and_staging_per_rank_bytes: usize,
    pub headroom_per_rank_bytes: usize,
}

impl QwenFlashNextCapacityPlan {
    /// Plans memory across Tier 1 (VRAM), Tier 2 (Pinned RAM `HostExpertStore`),
    /// and Tier 3 (Zero-copy `PleMmapTable`) for `TP=1` or `TP=2 / EP=2`.
    #[allow(clippy::too_many_arguments)]
    pub fn plan(
        cfg: &ModelConfig,
        tp_size: usize,
        total_vram_per_rank_bytes: usize,
        free_vram_per_rank_bytes: usize,
        kv_policy: KvFormatPolicy,
        block_size: usize,
        requested_kv_pages: Option<usize>,
        requested_state_slots: Option<usize>,
        requested_hot_experts_per_layer: Option<usize>,
        include_mtp_in_vram: bool,
    ) -> Result<Self, String> {
        if tp_size != 1 && tp_size != 2 {
            return Err(format!("unsupported tp_size={tp_size} (expected 1 or 2)"));
        }
        let num_layers = cfg.num_hidden_layers;
        let num_full_attn = cfg
            .layer_types
            .iter()
            .filter(|&&t| t == LayerKind::FullAttention)
            .count();
        let num_gdn = num_layers.saturating_sub(num_full_attn);

        // Exact dense text footprint from checkpoint structure:
        // TP-sharded dense projections (embed, lm_head, gdn, qsa main, hc down/up/inj, shared_expert):
        //   8,828,608,512 B total on TP=1 -> 4,414,304,256 B/rank on TP=2.
        // Replicated small norms + gates + QSA indexer + PLE dense:
        //   1,066,788,888 B/rank.
        let tp_sharded_dense_total = 8_828_608_512usize;
        let tp_replicated_dense = 1_066_788_888usize;
        let dense_weights_per_rank_bytes =
            (tp_sharded_dense_total / tp_size).saturating_add(tp_replicated_dense);

        // Optional MTP dense spine (~680 MiB on TP=1, ~380 MiB/rank on TP=2).
        let mtp_vram_per_rank_bytes = if include_mtp_in_vram {
            if tp_size == 1 {
                680 * 1024 * 1024
            } else {
                380 * 1024 * 1024
            }
        } else {
            0
        };

        // Single W4A16 routed expert slot (gate + up + down qweight + scales, 4K aligned): 2,535,424 B.
        let expert_slot_bytes = 2_535_424usize;
        let local_total_experts_per_layer = cfg.num_experts / tp_size;

        // StateBundlePool per-slot footprint:
        // 36 GDN layers: conv_state (FP16) + ssm_state (FP32)
        // + 1 PLE dilated short-conv state: [10240, 9] FP16 = 184,320 B + 2x4B ngram_context.
        let local_k_heads = cfg.linear_num_key_heads / tp_size;
        let local_v_heads = cfg.linear_num_value_heads / tp_size;
        let conv_dim =
            2 * local_k_heads * cfg.linear_key_head_dim + local_v_heads * cfg.linear_value_head_dim;
        let gdn_conv_bytes = conv_dim * cfg.linear_conv_kernel_dim.saturating_sub(1) * 2;
        let gdn_ssm_bytes = local_v_heads * cfg.linear_value_head_dim * cfg.linear_key_head_dim * 4;
        let ple_conv_dilation = 3usize;
        let ple_conv_taps =
            ple_conv_dilation.saturating_mul(cfg.ple_conv_kernel_size.saturating_sub(1));
        let ple_key_dim = cfg.ple_embed_dim * cfg.hc_count.max(1);
        let ple_state_bytes = ple_key_dim * ple_conv_taps * 2 + 16;
        let bytes_per_state_slot = num_gdn * (gdn_conv_bytes + gdn_ssm_bytes) + ple_state_bytes;
        let state_slots = requested_state_slots.unwrap_or(if tp_size == 1 { 8 } else { 16 });
        let state_pool_per_rank_bytes = state_slots * bytes_per_state_slot;

        // KvPool per-page footprint across all 12 QSA layers:
        // Main KV (policy-dependent, local_kv_heads = 2/tp_size, head_dim = 256)
        // + QSA Indexer 4:1 compressed key cache (12 layers * 1 head * 128 dim * 2B / 4 = 768 B/tok)
        // + QSA Indexer raw key cache (12 layers * 1 head * 128 dim * 2B = 3,072 B/tok).
        let local_kv_heads = (cfg.num_key_value_heads / tp_size).max(1);
        let main_kv_bytes_per_tok =
            kv_policy.bytes_per_token(num_full_attn, local_kv_heads, cfg.head_dim);
        let indexer_kv_bytes_per_tok = if cfg.indexer_head_dim > 0 {
            let raw_b = num_full_attn * cfg.indexer_kv_heads.max(1) * cfg.indexer_head_dim * 2;
            let comp_b = raw_b / cfg.indexer_compress_ratio.max(1);
            raw_b + comp_b
        } else {
            0
        };
        let bytes_per_kv_page_per_rank =
            (main_kv_bytes_per_tok + indexer_kv_bytes_per_tok) * block_size.max(1);

        let default_kv_pages = if tp_size == 1 { 2048 } else { 4096 };
        let kv_pages_per_rank = requested_kv_pages.unwrap_or(default_kv_pages);
        let kv_pool_per_rank_bytes = kv_pages_per_rank * bytes_per_kv_page_per_rank;

        // Scratch + double-buffered PCIe staging + dynamic LRU expert buffer:
        let scratch_and_staging_per_rank_bytes = if tp_size == 1 {
            768 * 1024 * 1024
        } else {
            896 * 1024 * 1024
        };
        let min_headroom_bytes = 512 * 1024 * 1024;

        let fixed_without_hot = dense_weights_per_rank_bytes
            .saturating_add(mtp_vram_per_rank_bytes)
            .saturating_add(state_pool_per_rank_bytes)
            .saturating_add(kv_pool_per_rank_bytes)
            .saturating_add(scratch_and_staging_per_rank_bytes)
            .saturating_add(min_headroom_bytes);

        if fixed_without_hot > free_vram_per_rank_bytes {
            return Err(format!(
                "QwenFlashNextCapacityPlan: fixed per-rank footprint ({:.2} GiB) exceeds free VRAM ({:.2} GiB)",
                fixed_without_hot as f64 / 1024.0f64.powi(3),
                free_vram_per_rank_bytes as f64 / 1024.0f64.powi(3),
            ));
        }

        let avail_for_hot = free_vram_per_rank_bytes - fixed_without_hot;
        let bytes_per_hot_expert_across_layers = num_layers * expert_slot_bytes;
        let max_hot_per_layer =
            (avail_for_hot / bytes_per_hot_expert_across_layers).min(local_total_experts_per_layer);
        let default_hot = if tp_size == 1 {
            max_hot_per_layer.min(48)
        } else {
            max_hot_per_layer.min(104)
        };
        let hot_experts_per_layer_per_rank = match requested_hot_experts_per_layer {
            Some(req) => {
                if req > max_hot_per_layer {
                    return Err(format!(
                        "requested {req} hot experts/layer exceeds VRAM capacity {max_hot_per_layer}"
                    ));
                }
                req
            }
            None => default_hot,
        };

        let hot_expert_vram_per_rank_bytes =
            hot_experts_per_layer_per_rank * bytes_per_hot_expert_across_layers;
        let cold_experts_per_layer_per_rank =
            local_total_experts_per_layer.saturating_sub(hot_experts_per_layer_per_rank);
        let host_expert_pinned_ram_bytes =
            cold_experts_per_layer_per_rank * num_layers * expert_slot_bytes * tp_size;
        let headroom_per_rank_bytes = free_vram_per_rank_bytes
            .saturating_sub(dense_weights_per_rank_bytes)
            .saturating_sub(mtp_vram_per_rank_bytes)
            .saturating_sub(state_pool_per_rank_bytes)
            .saturating_sub(kv_pool_per_rank_bytes)
            .saturating_sub(scratch_and_staging_per_rank_bytes)
            .saturating_sub(hot_expert_vram_per_rank_bytes);

        let ple_mmap_virtual_bytes = 51_200_245_760usize;

        Ok(Self {
            tp_size,
            total_vram_per_rank_bytes,
            free_vram_per_rank_bytes,
            dense_weights_per_rank_bytes,
            mtp_vram_per_rank_bytes,
            expert_slot_bytes,
            hot_experts_per_layer_per_rank,
            hot_expert_vram_per_rank_bytes,
            cold_experts_per_layer_per_rank,
            host_expert_pinned_ram_bytes,
            ple_mmap_virtual_bytes,
            state_slots,
            bytes_per_state_slot,
            state_pool_per_rank_bytes,
            bytes_per_kv_page_per_rank,
            kv_pages_per_rank,
            kv_pool_per_rank_bytes,
            scratch_and_staging_per_rank_bytes,
            headroom_per_rank_bytes,
        })
    }
}

#[cfg(feature = "cuda")]
pub mod gpu_ops {
    use core::ffi::c_void;
    use w34_kernels::ffi::{self, W34AttentionBatch, W34Exec, W34GdnBatch, W34KvLayerLayout};

    use crate::cuda::{CudaError, Result};

    #[inline]
    fn check(rc: i32, op: &'static str) -> Result<()> {
        if rc == 0 {
            Ok(())
        } else {
            Err(CudaError::new(rc, op))
        }
    }

    #[allow(clippy::too_many_arguments)]
    pub fn hc_init_streams_f16(
        ex: &W34Exec,
        embed_in: u64,
        residual_out: u64,
        valid_token_mask: u64,
        num_tokens: usize,
        hc_count: usize,
        hidden_size: usize,
    ) -> Result<()> {
        // SAFETY: Device pointers are validated by the caller and sized for `num_tokens`.
        let rc = unsafe {
            ffi::w34_hc_init_streams_f16(
                ex,
                embed_in as usize as *const c_void,
                residual_out as usize as *mut c_void,
                valid_token_mask as usize as *const u8,
                num_tokens as i32,
                hc_count as i32,
                hidden_size as i32,
            )
        };
        check(rc, "w34_hc_init_streams_f16")
    }

    #[allow(clippy::too_many_arguments)]
    pub fn hc_rmsnorm_f16(
        ex: &W34Exec,
        residual_in: u64,
        norm_weight: u64,
        normed_out: u64,
        valid_token_mask: u64,
        num_tokens: usize,
        hc_count: usize,
        hidden_size: usize,
        eps: f32,
    ) -> Result<()> {
        // SAFETY: Device pointers are validated by the caller and sized for `num_tokens * hc_count * hidden_size`.
        let rc = unsafe {
            ffi::w34_hc_rmsnorm_f16(
                ex,
                residual_in as usize as *const c_void,
                norm_weight as usize as *const c_void,
                normed_out as usize as *mut c_void,
                valid_token_mask as usize as *const u8,
                num_tokens as i32,
                hc_count as i32,
                hidden_size as i32,
                eps,
            )
        };
        check(rc, "w34_hc_rmsnorm_f16")
    }

    #[allow(clippy::too_many_arguments)]
    pub fn hc_down_silu_inject_f16(
        ex: &W34Exec,
        down_in: u64,
        down_out: u64,
        inj_in: u64,
        inject_weights_out: u64,
        valid_token_mask: u64,
        num_tokens: usize,
        hc_rank_dim: usize,
        hc_count: usize,
    ) -> Result<()> {
        // SAFETY: Device pointers are validated by the caller.
        let rc = unsafe {
            ffi::w34_hc_down_silu_inject_f16(
                ex,
                down_in as usize as *const c_void,
                down_out as usize as *mut c_void,
                inj_in as usize as *const c_void,
                inject_weights_out as usize as *mut f32,
                valid_token_mask as usize as *const u8,
                num_tokens as i32,
                hc_rank_dim as i32,
                hc_count as i32,
            )
        };
        check(rc, "w34_hc_down_silu_inject_f16")
    }

    #[allow(clippy::too_many_arguments)]
    pub fn hc_mix_streams_f16(
        ex: &W34Exec,
        residual_in: u64,
        up_raw_in: u64,
        mixed_out: u64,
        valid_token_mask: u64,
        num_tokens: usize,
        hc_count: usize,
        hidden_size: usize,
    ) -> Result<()> {
        // SAFETY: Device pointers are validated by the caller.
        let rc = unsafe {
            ffi::w34_hc_mix_streams_f16(
                ex,
                residual_in as usize as *const c_void,
                up_raw_in as usize as *const c_void,
                mixed_out as usize as *mut c_void,
                valid_token_mask as usize as *const u8,
                num_tokens as i32,
                hc_count as i32,
                hidden_size as i32,
            )
        };
        check(rc, "w34_hc_mix_streams_f16")
    }

    #[allow(clippy::too_many_arguments)]
    pub fn hc_combine_f16(
        ex: &W34Exec,
        residual_in: u64,
        block_out: u64,
        inject_weights: u64,
        ple_delta: u64,
        residual_out: u64,
        valid_token_mask: u64,
        num_tokens: usize,
        hc_count: usize,
        hidden_size: usize,
    ) -> Result<()> {
        // SAFETY: Device pointers are validated by the caller.
        let rc = unsafe {
            ffi::w34_hc_combine_f16(
                ex,
                residual_in as usize as *const c_void,
                block_out as usize as *const c_void,
                inject_weights as usize as *const f32,
                ple_delta as usize as *const c_void,
                residual_out as usize as *mut c_void,
                valid_token_mask as usize as *const u8,
                num_tokens as i32,
                hc_count as i32,
                hidden_size as i32,
            )
        };
        check(rc, "w34_hc_combine_f16")
    }

    pub fn ple_dequant_f8_to_f16(
        ex: &W34Exec,
        f8_in: u64,
        f16_out: u64,
        valid_token_mask: u64,
        num_tokens: usize,
        embed_dim: usize,
        weight_scale: f32,
    ) -> Result<()> {
        // SAFETY: Device pointers are validated by the caller.
        let rc = unsafe {
            ffi::w34_ple_dequant_f8_to_f16(
                ex,
                f8_in as usize as *const u8,
                f16_out as usize as *mut c_void,
                valid_token_mask as usize as *const u8,
                num_tokens as i32,
                embed_dim as i32,
                weight_scale,
            )
        };
        check(rc, "w34_ple_dequant_f8_to_f16")
    }

    pub fn ple_dilated_conv1d_f16(
        ex: &W34Exec,
        key_in: u64,
        conv_weight: u64,
        conv_state_pool: u64,
        key_out: u64,
        batch: &W34GdnBatch,
        conv_dim: usize,
    ) -> Result<()> {
        // SAFETY: Device pointers and `batch` metadata are validated by the caller.
        let rc = unsafe {
            ffi::w34_ple_dilated_conv1d_f16(
                ex,
                key_in as usize as *const c_void,
                conv_weight as usize as *const c_void,
                conv_state_pool as usize as *mut c_void,
                key_out as usize as *mut c_void,
                batch,
                conv_dim as i32,
            )
        };
        check(rc, "w34_ple_dilated_conv1d_f16")
    }

    #[allow(clippy::too_many_arguments)]
    pub fn ple_signed_sqrt_gate_f16(
        ex: &W34Exec,
        norm_query: u64,
        norm_key: u64,
        value_in: u64,
        ple_delta_out: u64,
        valid_token_mask: u64,
        num_tokens: usize,
        hc_count: usize,
        embed_dim: usize,
        eps: f32,
    ) -> Result<()> {
        // SAFETY: Device pointers are validated by the caller.
        let rc = unsafe {
            ffi::w34_ple_signed_sqrt_gate_f16(
                ex,
                norm_query as usize as *const c_void,
                norm_key as usize as *const c_void,
                value_in as usize as *const c_void,
                ple_delta_out as usize as *mut c_void,
                valid_token_mask as usize as *const u8,
                num_tokens as i32,
                hc_count as i32,
                embed_dim as i32,
                eps,
            )
        };
        check(rc, "w34_ple_signed_sqrt_gate_f16")
    }

    #[allow(clippy::too_many_arguments)]
    pub fn qsa_indexer_norm_rope_f16(
        ex: &W34Exec,
        qk_in: u64,
        q_norm_weight: u64,
        k_norm_weight: u64,
        positions: u64,
        q_out: u64,
        k_out: u64,
        valid_token_mask: u64,
        num_tokens: usize,
        q_heads: usize,
        kv_heads: usize,
        head_dim: usize,
        rotary_dim: usize,
        rope_theta: f32,
        eps: f32,
        mrope_s1: usize,
        mrope_s2: usize,
        pos_plane_stride: usize,
    ) -> Result<()> {
        // SAFETY: Device pointers are validated by the caller.
        let rc = unsafe {
            ffi::w34_qsa_indexer_norm_rope_f16(
                ex,
                qk_in as usize as *const c_void,
                q_norm_weight as usize as *const c_void,
                k_norm_weight as usize as *const c_void,
                positions as usize as *const i32,
                q_out as usize as *mut c_void,
                k_out as usize as *mut c_void,
                valid_token_mask as usize as *const u8,
                num_tokens as i32,
                q_heads as i32,
                kv_heads as i32,
                head_dim as i32,
                rotary_dim as i32,
                rope_theta,
                eps,
                mrope_s1 as i32,
                mrope_s2 as i32,
                pos_plane_stride as i32,
            )
        };
        check(rc, "w34_qsa_indexer_norm_rope_f16")
    }

    #[allow(clippy::too_many_arguments)]
    pub fn qsa_indexer_cache_update_f16(
        ex: &W34Exec,
        k_in: u64,
        raw_k_cache: u64,
        comp_k_cache: u64,
        k_norm_weight: u64,
        positions: u64,
        write_slot_mapping: u64,
        valid_token_mask: u64,
        num_tokens: usize,
        head_dim: usize,
        rotary_dim: usize,
        rope_theta: f32,
        eps: f32,
        mrope_s1: usize,
        mrope_s2: usize,
        pos_plane_stride: usize,
    ) -> Result<()> {
        // SAFETY: Device pointers are validated by the caller.
        let rc = unsafe {
            ffi::w34_qsa_indexer_cache_update_f16(
                ex,
                k_in as usize as *const c_void,
                raw_k_cache as usize as *mut c_void,
                comp_k_cache as usize as *mut c_void,
                k_norm_weight as usize as *const c_void,
                positions as usize as *const i32,
                write_slot_mapping as usize as *const i32,
                valid_token_mask as usize as *const u8,
                num_tokens as i32,
                head_dim as i32,
                rotary_dim as i32,
                rope_theta,
                eps,
                mrope_s1 as i32,
                mrope_s2 as i32,
                pos_plane_stride as i32,
            )
        };
        check(rc, "w34_qsa_indexer_cache_update_f16")
    }

    #[allow(clippy::too_many_arguments)]
    pub fn qsa_select_blocks_f16(
        ex: &W34Exec,
        q_indexer: u64,
        comp_k_cache: u64,
        batch: &W34AttentionBatch,
        out_selected_phys_blocks: u64,
        out_selected_logical_blocks: u64,
        out_num_selected_blocks: u64,
        out_token_seq_pos: u64,
        q_heads: usize,
        head_dim: usize,
        block_size: usize,
        select_k_blocks: usize,
        sink_blocks: usize,
        recent_blocks: usize,
    ) -> Result<()> {
        // SAFETY: Device pointers and `batch` metadata are validated by the caller.
        let rc = unsafe {
            ffi::w34_qsa_select_blocks_f16(
                ex,
                q_indexer as usize as *const c_void,
                comp_k_cache as usize as *const c_void,
                batch,
                out_selected_phys_blocks as usize as *mut i32,
                out_selected_logical_blocks as usize as *mut i32,
                out_num_selected_blocks as usize as *mut i32,
                out_token_seq_pos as usize as *mut i32,
                q_heads as i32,
                head_dim as i32,
                block_size as i32,
                select_k_blocks as i32,
                sink_blocks as i32,
                recent_blocks as i32,
            )
        };
        check(rc, "w34_qsa_select_blocks_f16")
    }

    #[allow(clippy::too_many_arguments)]
    pub fn qsa_sparse_paged_attn_f16(
        ex: &W34Exec,
        q: u64,
        gate: u64,
        kv_layout: &W34KvLayerLayout,
        selected_phys_blocks: u64,
        selected_logical_blocks: u64,
        num_selected_blocks: u64,
        token_seq_pos: u64,
        valid_token_mask: u64,
        out: u64,
        num_tokens: usize,
        local_q_heads: usize,
        select_k_blocks: usize,
        sm_scale: f32,
    ) -> Result<()> {
        // SAFETY: Device pointers and `kv_layout` are validated by the caller.
        let rc = unsafe {
            ffi::w34_qsa_sparse_paged_attn_f16(
                ex,
                q as usize as *const c_void,
                gate as usize as *const c_void,
                kv_layout,
                selected_phys_blocks as usize as *const i32,
                selected_logical_blocks as usize as *const i32,
                num_selected_blocks as usize as *const i32,
                token_seq_pos as usize as *const i32,
                valid_token_mask as usize as *const u8,
                out as usize as *mut c_void,
                num_tokens as i32,
                local_q_heads as i32,
                select_k_blocks as i32,
                sm_scale,
            )
        };
        check(rc, "w34_qsa_sparse_paged_attn_f16")
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::Path;

    fn sample_qwen4_exp_cfg() -> ModelConfig {
        let root: serde_json::Value = serde_json::from_str(
            r#"{
              "architectures": ["Qwen4ExpForConditionalGeneration"],
              "model_type": "qwen4_exp",
              "text_config": {
                "model_type": "qwen4_exp_text",
                "hidden_size": 2560,
                "num_hidden_layers": 48,
                "num_attention_heads": 24,
                "num_key_value_heads": 2,
                "head_dim": 256,
                "vocab_size": 248320,
                "eos_token_id": 248044,
                "linear_num_key_heads": 16,
                "linear_num_value_heads": 48,
                "linear_key_head_dim": 128,
                "linear_value_head_dim": 128,
                "linear_conv_kernel_dim": 4,
                "num_experts": 512,
                "num_experts_per_tok": 10,
                "moe_intermediate_size": 640,
                "shared_expert_intermediate_size": 640,
                "hc_count": 4,
                "hc_lowrank": 80,
                "ple_embed_dim": 2560,
                "ple_conv_kernel_size": 4,
                "ple_layer_ids": [2],
                "ple_seed": 20260202,
                "ngram_size": 3,
                "heads_per_ngram": 8,
                "ngram_vocab_size_base": 20000000,
                "make_ngram_vocab_size_divisible_by": 128,
                "split_ngram_parts": 128,
                "indexer_head_dim": 128,
                "indexer_n_heads": 4,
                "indexer_kv_heads": 1,
                "indexer_budget": 1024,
                "indexer_compress_ratio": 4
              }
            }"#,
        )
        .unwrap();
        ModelConfig::from_json(&root).unwrap()
    }

    #[test]
    fn test_ple_hasher_matches_checkpoint_constants_and_shift_semantics() {
        let cfg = sample_qwen4_exp_cfg();
        let hasher = PleNgramHasher::from_config(&cfg, 1, 0);
        assert_eq!(
            hasher.layer_multipliers,
            vec![23_703_573_157_769, 20_109_073_645_365, 8_052_911_324_071]
        );
        assert_eq!(
            hasher.ngram_heads_vocab_sizes,
            vec![
                20_000_003, 20_000_023, 20_000_033, 20_000_047, 20_000_059, 20_000_063, 20_000_069,
                20_000_077, 20_000_081, 20_000_093, 20_000_107, 20_000_147, 20_000_153, 20_000_159,
                20_000_161, 20_000_171
            ]
        );
        assert_eq!(
            hasher.ngram_heads_offsets,
            vec![
                0,
                20_000_003,
                40_000_026,
                60_000_059,
                80_000_106,
                100_000_165,
                120_000_228,
                140_000_297,
                160_000_374,
                180_000_455,
                200_000_548,
                220_000_655,
                240_000_802,
                260_000_955,
                280_001_114,
                300_001_275
            ]
        );
        assert_eq!(hasher.padded_vocab_size, 320_001_536);
        assert_eq!(hasher.shard_rows, 2_500_012);

        // Verify chunked streaming matches single-shot prefill across EOS boundaries.
        let eos = hasher.eos_token_id;
        let seq = [101, 202, eos, 303, 404, 505, eos, eos, 606];
        let mut ctx_full = [[eos, eos]];
        let mut ids_full = vec![0i64; seq.len() * 16];
        hasher.compute_ngram_ids(&seq, &[0, seq.len() as i32], &mut ctx_full, &mut ids_full);

        let mut ctx_step = [[eos, eos]];
        let mut ids_step = vec![0i64; seq.len() * 16];
        for (i, &tok) in seq.iter().enumerate() {
            hasher.compute_ngram_ids(
                &[tok],
                &[0, 1],
                &mut ctx_step,
                &mut ids_step[i * 16..(i + 1) * 16],
            );
        }
        assert_eq!(ids_full, ids_step);
        assert_eq!(ctx_full, ctx_step);
    }

    #[test]
    fn test_qwen_flash_next_capacity_planner_tp1_and_tp2() {
        let cfg = sample_qwen4_exp_cfg();
        let vram_24g = 24 * 1024 * 1024 * 1024usize;
        let free_23_5g = 23_600 * 1024 * 1024usize;

        let plan_tp1 = QwenFlashNextCapacityPlan::plan(
            &cfg,
            1,
            vram_24g,
            free_23_5g,
            KvFormatPolicy::Tq4,
            16,
            Some(2048),
            Some(8),
            None,
            true,
        )
        .unwrap();
        assert!(plan_tp1.hot_experts_per_layer_per_rank >= 36);
        assert!(plan_tp1.headroom_per_rank_bytes >= 512 * 1024 * 1024);

        let plan_tp2 = QwenFlashNextCapacityPlan::plan(
            &cfg,
            2,
            vram_24g,
            free_23_5g,
            KvFormatPolicy::Tq4,
            16,
            Some(4096),
            Some(16),
            None,
            true,
        )
        .unwrap();
        assert!(plan_tp2.hot_experts_per_layer_per_rank >= 96);
        assert!(plan_tp2.headroom_per_rank_bytes >= 512 * 1024 * 1024);
    }

    #[test]
    fn test_ple_mmap_table_against_real_snapshot_if_present() {
        let snap = Path::new(
            "/home/nchapman/.cache/huggingface/hub/models--albucino--Qwen3.8-Flash-Next-W4A16-FP8PLE/snapshots/ef554143369a706525336f6b42a09094835dc077",
        );
        if !snap.exists() {
            return;
        }
        let cfg = ModelConfig::from_dir(snap).unwrap();
        let archive = SafetensorsArchive::open(snap).unwrap();
        let hasher = PleNgramHasher::from_archive(&archive, &cfg, 1, 0).unwrap();
        let table = PleMmapTable::open(&archive, &cfg, 1).unwrap();
        assert_eq!(table.shard_rows, 2_500_012);
        assert_eq!(table.head_dim, 160);
        assert!((table.lut.weight_scale - 0.00019931793).abs() < 1e-9);

        let eos = hasher.eos_token_id;
        let mut ctx = [[eos, eos]];
        let tokens = [151644, 872, 198, 3838];
        let mut ids = vec![0i64; tokens.len() * 16];
        hasher.compute_ngram_ids(&tokens, &[0, tokens.len() as i32], &mut ctx, &mut ids);
        let mut out_f32 = vec![0.0f32; tokens.len() * 2560];
        table.gather_dequant_f32(&ids, &mut out_f32);
        assert!(out_f32.iter().all(|v| v.is_finite()));
        assert!(out_f32.iter().any(|&v| v != 0.0));
    }

    #[cfg(feature = "cuda")]
    #[test]
    #[allow(clippy::cognitive_complexity)]
    fn test_gpu_hc_ple_qsa_kernels_parity() {
        use crate::cuda::{CudaDevice, CudaStream, DeviceBuffer, f16_to_f32};
        use core::ffi::c_void;
        use w34_kernels::ffi::{
            W34_ATTN_MASK_CAUSAL, W34_KV_FORMAT_FP16, W34_KV_FORMAT_Q8_0, W34_KV_FORMAT_TQ4,
            W34_KV_NAMESPACE_TARGET_FULL_ATTN, W34_KV_TRANSFORM_IDENTITY,
            W34_KV_TRANSFORM_RADEMACHER_WHT256, W34AttentionBatch, W34GdnBatch, W34KvLayerLayout,
            w34_kv_cache_write_strided,
        };

        fn dev_from_slice<T: Copy>(src: &[T], stream: &CudaStream) -> DeviceBuffer {
            let buf = DeviceBuffer::alloc(0, core::mem::size_of_val(src)).unwrap();
            buf.copy_from_host(src, stream).unwrap();
            stream.synchronize().unwrap();
            buf
        }

        fn dev_to_vec<T: Copy + Default>(
            buf: &DeviceBuffer,
            len: usize,
            stream: &CudaStream,
        ) -> Vec<T> {
            let mut out = vec![T::default(); len];
            buf.copy_to_host(&mut out, stream).unwrap();
            out
        }

        let _dev = CudaDevice::set_current(0).unwrap();
        let stream = CudaStream::new(0).unwrap();
        let ex = stream.exec(0, 0);

        // 1. Verify HyperConnection (GatedResidual) kernels:
        //    hc_init_streams_f16 -> hc_rmsnorm_f16 -> hc_down_silu_inject_f16 -> hc_mix_streams_f16 -> hc_combine_f16
        let m = 3usize;
        let hc_count = 4usize;
        let hidden = 2560usize;
        let rank_dim = 320usize;
        let eps = 1e-6f32;

        let embed_f32: Vec<f32> = (0..m * hidden)
            .map(|i| (((i * 17 + 5) % 97) as f32 - 48.0) * 0.02)
            .collect();
        let embed_f16: Vec<u16> = embed_f32.iter().map(|&v| f32_to_f16(v)).collect();
        let norm_w_f32: Vec<f32> = (0..hc_count * hidden)
            .map(|i| (((i * 13 + 3) % 53) as f32 - 26.0) * 0.01)
            .collect();
        let norm_w_f16: Vec<u16> = norm_w_f32.iter().map(|&v| f32_to_f16(v)).collect();
        let down_in_f32: Vec<f32> = (0..m * rank_dim)
            .map(|i| (((i * 23 + 7) % 89) as f32 - 44.0) * 0.1)
            .collect();
        let down_in_f16: Vec<u16> = down_in_f32.iter().map(|&v| f32_to_f16(v)).collect();
        let inj_in_f32: Vec<f32> = (0..m * hc_count)
            .map(|i| ((i as f32) - 5.5) * 0.8)
            .collect();
        let inj_in_f16: Vec<u16> = inj_in_f32.iter().map(|&v| f32_to_f16(v)).collect();
        let up_in_f32: Vec<f32> = (0..m * hc_count * hidden)
            .map(|i| (((i * 7 + 11) % 71) as f32 - 35.0) * 0.05)
            .collect();
        let up_in_f16: Vec<u16> = up_in_f32.iter().map(|&v| f32_to_f16(v)).collect();
        let blk_out_f32: Vec<f32> = (0..m * hidden)
            .map(|i| (((i * 19 + 1) % 67) as f32 - 33.0) * 0.03)
            .collect();
        let blk_out_f16: Vec<u16> = blk_out_f32.iter().map(|&v| f32_to_f16(v)).collect();
        let ple_delta_f32: Vec<f32> = (0..m * hc_count * hidden)
            .map(|i| (((i * 31 + 2) % 61) as f32 - 30.0) * 0.01)
            .collect();
        let ple_delta_f16: Vec<u16> = ple_delta_f32.iter().map(|&v| f32_to_f16(v)).collect();

        let d_embed = dev_from_slice(&embed_f16, &stream);
        let d_res = DeviceBuffer::alloc(0, m * hc_count * hidden * 2).unwrap();
        let d_norm_w = dev_from_slice(&norm_w_f16, &stream);
        let d_normed = DeviceBuffer::alloc(0, m * hc_count * hidden * 2).unwrap();
        let d_down_in = dev_from_slice(&down_in_f16, &stream);
        let d_down_out = DeviceBuffer::alloc(0, m * rank_dim * 2).unwrap();
        let d_inj_in = dev_from_slice(&inj_in_f16, &stream);
        let d_inj_w = DeviceBuffer::alloc(0, m * hc_count * 4).unwrap();
        let d_up_in = dev_from_slice(&up_in_f16, &stream);
        let d_mixed = DeviceBuffer::alloc(0, m * hidden * 2).unwrap();
        let d_blk_out = dev_from_slice(&blk_out_f16, &stream);
        let d_ple_delta = dev_from_slice(&ple_delta_f16, &stream);
        let d_res_out = DeviceBuffer::alloc(0, m * hc_count * hidden * 2).unwrap();

        gpu_ops::hc_init_streams_f16(&ex, d_embed.addr(), d_res.addr(), 0, m, hc_count, hidden)
            .unwrap();
        gpu_ops::hc_rmsnorm_f16(
            &ex,
            d_res.addr(),
            d_norm_w.addr(),
            d_normed.addr(),
            0,
            m,
            hc_count,
            hidden,
            eps,
        )
        .unwrap();
        gpu_ops::hc_down_silu_inject_f16(
            &ex,
            d_down_in.addr(),
            d_down_out.addr(),
            d_inj_in.addr(),
            d_inj_w.addr(),
            0,
            m,
            rank_dim,
            hc_count,
        )
        .unwrap();
        gpu_ops::hc_mix_streams_f16(
            &ex,
            d_res.addr(),
            d_up_in.addr(),
            d_mixed.addr(),
            0,
            m,
            hc_count,
            hidden,
        )
        .unwrap();
        gpu_ops::hc_combine_f16(
            &ex,
            d_res.addr(),
            d_blk_out.addr(),
            d_inj_w.addr(),
            d_ple_delta.addr(),
            d_res_out.addr(),
            0,
            m,
            hc_count,
            hidden,
        )
        .unwrap();

        let h_down_out: Vec<u16> = dev_to_vec(&d_down_out, m * rank_dim, &stream);
        let h_inj_w: Vec<f32> = dev_to_vec(&d_inj_w, m * hc_count, &stream);
        let h_mixed: Vec<u16> = dev_to_vec(&d_mixed, m * hidden, &stream);
        let h_res_out: Vec<u16> = dev_to_vec(&d_res_out, m * hc_count * hidden, &stream);

        for i in 0..m * rank_dim {
            let x = f16_to_f32(down_in_f16[i]) * 0.25;
            let ref_silu = x / (1.0 + (-x).exp());
            let got = f16_to_f32(h_down_out[i]);
            assert!((got - ref_silu).abs() < 2e-3, "down silu mismatch at {i}");
        }
        for i in 0..m * hc_count {
            let x = f16_to_f32(inj_in_f16[i]) * 0.25;
            let ref_inj = 2.0 / (1.0 + (-x).exp());
            assert!((h_inj_w[i] - ref_inj).abs() < 1e-5, "inj mismatch at {i}");
        }
        for tok in 0..m {
            for d in 0..hidden {
                let r = f16_to_f32(embed_f16[tok * hidden + d]);
                let mut acc = 0.0f32;
                for s in 0..hc_count {
                    let u = f16_to_f32(up_in_f16[(tok * hc_count + s) * hidden + d]);
                    acc += (0.25 / (1.0 + (-u).exp())) * r;
                }
                let got_mix = f16_to_f32(h_mixed[tok * hidden + d]);
                assert!((got_mix - acc).abs() < 2e-3);

                let y = f16_to_f32(blk_out_f16[tok * hidden + d]);
                for s in 0..hc_count {
                    let idx = (tok * hc_count + s) * hidden + d;
                    let p = f16_to_f32(ple_delta_f16[idx]);
                    let ref_comb = r + h_inj_w[tok * hc_count + s] * y + p;
                    let got_comb = f16_to_f32(h_res_out[idx]);
                    assert!((got_comb - ref_comb).abs() < 3e-3);
                }
            }
        }

        // 2. Verify PLE kernels:
        //    ple_dequant_f8_to_f16, ple_dilated_conv1d_f16 (prefill + continuation step), ple_signed_sqrt_gate_f16
        let f8_bytes: Vec<u8> = (0..m * hidden)
            .map(|i| ((i * 29 + 3) % 120) as u8)
            .collect();
        let scale = 0.00019931793f32;
        let lut = F8E4m3ScaleLut::new(scale);
        let d_f8 = dev_from_slice(&f8_bytes, &stream);
        let d_ple_emb = DeviceBuffer::alloc(0, m * hidden * 2).unwrap();
        gpu_ops::ple_dequant_f8_to_f16(&ex, d_f8.addr(), d_ple_emb.addr(), 0, m, hidden, scale)
            .unwrap();
        let h_ple_emb: Vec<u16> = dev_to_vec(&d_ple_emb, m * hidden, &stream);
        for (i, &b) in f8_bytes.iter().enumerate() {
            assert_eq!(h_ple_emb[i], lut.f16_lut[b as usize]);
        }

        // Verify PLE signed-sqrt sigmoid gate (including exact 0.0 sign preservation):
        let query_vals: Vec<f32> = (0..m * hc_count * hidden)
            .map(|i| {
                if i % 16 == 0 {
                    0.0
                } else {
                    (((i * 19 + 5) % 83) as f32 - 41.0) * 0.2
                }
            })
            .collect();
        let query_f16: Vec<u16> = query_vals.iter().map(|&v| f32_to_f16(v)).collect();
        let key_f16: Vec<u16> = (0..m * hc_count * hidden)
            .map(|i| f32_to_f16((((i * 7 + 3) % 59) as f32 - 29.0) * 0.1))
            .collect();
        let val_f16: Vec<u16> = (0..m * hidden)
            .map(|i| f32_to_f16((((i * 13 + 1) % 47) as f32 - 23.0) * 0.15))
            .collect();
        let d_qg = dev_from_slice(&query_f16, &stream);
        let d_kg = dev_from_slice(&key_f16, &stream);
        let d_vg = dev_from_slice(&val_f16, &stream);
        let d_gout = DeviceBuffer::alloc(0, m * hc_count * hidden * 2).unwrap();
        gpu_ops::ple_signed_sqrt_gate_f16(
            &ex,
            d_qg.addr(),
            d_kg.addr(),
            d_vg.addr(),
            d_gout.addr(),
            0,
            m,
            hc_count,
            hidden,
            1e-6,
        )
        .unwrap();
        let h_gout: Vec<u16> = dev_to_vec(&d_gout, m * hc_count * hidden, &stream);
        for tok in 0..m {
            for s in 0..hc_count {
                let mut dot = 0.0f32;
                for d in 0..hidden {
                    let idx = (tok * hc_count + s) * hidden + d;
                    dot += f16_to_f32(query_f16[idx]) * f16_to_f32(key_f16[idx]);
                }
                dot /= (hidden as f32).sqrt();
                let sgn = if dot > 0.0 {
                    1.0
                } else if dot < 0.0 {
                    -1.0
                } else {
                    0.0
                };
                let logit = sgn * dot.abs().max(1e-6).sqrt();
                let g = 1.0 / (1.0 + (-logit).exp());
                for d in 0..hidden {
                    let v = f16_to_f32(val_f16[tok * hidden + d]);
                    let idx = (tok * hc_count + s) * hidden + d;
                    let ref_out = g * v;
                    let got = f16_to_f32(h_gout[idx]);
                    assert!((got - ref_out).abs() < 3e-3, "gate mismatch at {idx}");
                }
            }
        }

        // Verify PLE stateful dilated conv1d (kernel=4, dilation=3, span=9):
        // Run 12 tokens in one prefill vs 10 tokens prefill + 2 tokens decode continuation!
        let conv_dim = 64usize;
        let seq_total = 12usize;
        let key_seq_f16: Vec<u16> = (0..seq_total * conv_dim)
            .map(|i| f32_to_f16((((i * 11 + 5) % 43) as f32 - 21.0) * 0.05))
            .collect();
        let conv_w_f16: Vec<u16> = (0..conv_dim * 4)
            .map(|i| f32_to_f16((((i * 7 + 1) % 19) as f32 - 9.0) * 0.1))
            .collect();
        let d_conv_w = dev_from_slice(&conv_w_f16, &stream);

        let d_key_full = dev_from_slice(&key_seq_f16, &stream);
        let d_out_full = DeviceBuffer::alloc(0, seq_total * conv_dim * 2).unwrap();
        let d_state_full = DeviceBuffer::alloc(0, 2 * conv_dim * 9 * 2).unwrap();
        let d_qsl_full = dev_from_slice(&[0i32, seq_total as i32], &stream);
        let d_slot0 = dev_from_slice(&[0i32], &stream);
        let d_has0 = dev_from_slice(&[0u8], &stream);
        let batch_full = W34GdnBatch {
            query_start_loc: d_qsl_full.addr() as usize as *const i32,
            state_in_indices: d_slot0.addr() as usize as *const i32,
            state_out_indices: d_slot0.addr() as usize as *const i32,
            spec_state_indices: core::ptr::null(),
            checkpoint_dst_indices: core::ptr::null(),
            checkpoint_token_offsets: core::ptr::null(),
            has_initial_state: d_has0.addr() as usize as *const u8,
            valid_req_mask: core::ptr::null(),
            num_reqs: 1,
            total_tokens: seq_total as i32,
            max_verify_tokens: 0,
            max_checkpoints_per_req: 0,
        };
        gpu_ops::ple_dilated_conv1d_f16(
            &ex,
            d_key_full.addr(),
            d_conv_w.addr(),
            d_state_full.addr(),
            d_out_full.addr(),
            &batch_full,
            conv_dim,
        )
        .unwrap();

        // Now run 10 tokens + 2 tokens continuation using state slot 1:
        let split = 10usize;
        let d_key_p1 = dev_from_slice(&key_seq_f16[..split * conv_dim], &stream);
        let d_out_p1 = DeviceBuffer::alloc(0, split * conv_dim * 2).unwrap();
        let d_qsl_p1 = dev_from_slice(&[0i32, split as i32], &stream);
        let d_slot1 = dev_from_slice(&[1i32], &stream);
        let batch_p1 = W34GdnBatch {
            query_start_loc: d_qsl_p1.addr() as usize as *const i32,
            state_in_indices: d_slot1.addr() as usize as *const i32,
            state_out_indices: d_slot1.addr() as usize as *const i32,
            total_tokens: split as i32,
            ..batch_full
        };
        gpu_ops::ple_dilated_conv1d_f16(
            &ex,
            d_key_p1.addr(),
            d_conv_w.addr(),
            d_state_full.addr(),
            d_out_p1.addr(),
            &batch_p1,
            conv_dim,
        )
        .unwrap();

        let rem = seq_total - split;
        let d_key_p2 = dev_from_slice(&key_seq_f16[split * conv_dim..], &stream);
        let d_out_p2 = DeviceBuffer::alloc(0, rem * conv_dim * 2).unwrap();
        let d_qsl_p2 = dev_from_slice(&[0i32, rem as i32], &stream);
        let d_has1 = dev_from_slice(&[1u8], &stream);
        let batch_p2 = W34GdnBatch {
            query_start_loc: d_qsl_p2.addr() as usize as *const i32,
            state_in_indices: d_slot1.addr() as usize as *const i32,
            state_out_indices: d_slot1.addr() as usize as *const i32,
            has_initial_state: d_has1.addr() as usize as *const u8,
            total_tokens: rem as i32,
            ..batch_full
        };
        gpu_ops::ple_dilated_conv1d_f16(
            &ex,
            d_key_p2.addr(),
            d_conv_w.addr(),
            d_state_full.addr(),
            d_out_p2.addr(),
            &batch_p2,
            conv_dim,
        )
        .unwrap();

        let h_out_full: Vec<u16> = dev_to_vec(&d_out_full, seq_total * conv_dim, &stream);
        let mut h_out_split: Vec<u16> = dev_to_vec(&d_out_p1, split * conv_dim, &stream);
        h_out_split.extend(dev_to_vec::<u16>(&d_out_p2, rem * conv_dim, &stream));
        assert_eq!(h_out_full, h_out_split);

        // 3. Verify QSA Indexer + Block Selector + Sparse Paged GQA kernels:
        let qsa_tokens = 2usize;
        let q_heads = 12usize;
        let kv_heads = 1usize;
        let head_dim = 256usize;
        let block_size = 16usize;
        let num_pages = 4usize;
        let select_k = 64usize;

        let d_k_cache = dev_from_slice(
            &vec![f32_to_f16(0.1); num_pages * kv_heads * block_size * head_dim],
            &stream,
        );
        let d_v_cache = dev_from_slice(
            &vec![f32_to_f16(0.25); num_pages * kv_heads * block_size * head_dim],
            &stream,
        );
        let kv_layout = W34KvLayerLayout {
            k_base: d_k_cache.addr(),
            v_base: d_v_cache.addr(),
            k_page_stride_bytes: (kv_heads * block_size * head_dim * 2) as u64,
            k_head_stride_bytes: (block_size * head_dim * 2) as u64,
            k_token_stride_bytes: (head_dim * 2) as u64,
            v_page_stride_bytes: (kv_heads * block_size * head_dim * 2) as u64,
            v_head_stride_bytes: (block_size * head_dim * 2) as u64,
            v_token_stride_bytes: (head_dim * 2) as u64,
            namespace_id: W34_KV_NAMESPACE_TARGET_FULL_ATTN,
            layer_id: 3,
            local_kv_heads: kv_heads as i32,
            head_dim: head_dim as i32,
            block_size: block_size as i32,
            k_format: W34_KV_FORMAT_FP16,
            v_format: W34_KV_FORMAT_FP16,
            k_transform: W34_KV_TRANSFORM_IDENTITY,
            v_transform: W34_KV_TRANSFORM_IDENTITY,
            reserved: 0,
        };

        let d_qk_idx_in = dev_from_slice(&vec![f32_to_f16(0.1); qsa_tokens * 5 * 128], &stream);
        let d_q_idx = dev_from_slice(&vec![f32_to_f16(0.1); qsa_tokens * 4 * 128], &stream);
        let d_k_idx = dev_from_slice(&vec![f32_to_f16(0.1); qsa_tokens * 128], &stream);
        let d_idx_norm_w = dev_from_slice(&vec![f32_to_f16(0.0); 128], &stream);
        let d_pos_ids = dev_from_slice(&[20i32, 21, 20, 21, 20, 21], &stream);
        gpu_ops::qsa_indexer_norm_rope_f16(
            &ex,
            d_qk_idx_in.addr(),
            d_idx_norm_w.addr(),
            d_idx_norm_w.addr(),
            d_pos_ids.addr(),
            d_q_idx.addr(),
            d_k_idx.addr(),
            0,
            qsa_tokens,
            4,
            1,
            128,
            64,
            10_000_000.0,
            1e-6,
            11,
            11,
            qsa_tokens,
        )
        .unwrap();

        let d_raw_k = dev_from_slice(&vec![f32_to_f16(0.1); num_pages * 16 * 128], &stream);
        let d_comp_k = dev_from_slice(&vec![f32_to_f16(0.1); num_pages * 4 * 128], &stream);
        let d_qsl_attn = dev_from_slice(&[0, qsa_tokens as i32], &stream);
        let d_ctx_before = dev_from_slice(&[20], &stream);
        let d_seq_after = dev_from_slice(&[22], &stream);
        let d_blk_tbl = dev_from_slice(&[2, 1, -1, -1], &stream);
        let d_slot_map = dev_from_slice(&[16 + 4, 16 + 5], &stream);
        gpu_ops::qsa_indexer_cache_update_f16(
            &ex,
            d_k_idx.addr(),
            d_raw_k.addr(),
            d_comp_k.addr(),
            d_idx_norm_w.addr(),
            d_pos_ids.addr(),
            d_slot_map.addr(),
            0,
            qsa_tokens,
            128,
            64,
            10_000_000.0,
            1e-6,
            11,
            11,
            qsa_tokens,
        )
        .unwrap();
        let attn_batch = W34AttentionBatch {
            query_start_loc: d_qsl_attn.addr() as usize as *const i32,
            context_lens_before: d_ctx_before.addr() as usize as *const i32,
            seq_lens_after: d_seq_after.addr() as usize as *const i32,
            block_tables: d_blk_tbl.addr() as usize as *const i32,
            write_slot_mapping: d_slot_map.addr() as usize as *const i32,
            valid_token_mask: core::ptr::null(),
            valid_req_mask: core::ptr::null(),
            num_reqs: 1,
            total_tokens: qsa_tokens as i32,
            max_blocks_per_req: 4,
            max_q_len: qsa_tokens as i32,
            max_seq_len: 22,
            mask_mode: W34_ATTN_MASK_CAUSAL,
            sliding_window: -1,
            sm_scale: (head_dim as f32).powf(-0.5),
            reserved: 0,
        };

        let d_sel_phys = DeviceBuffer::alloc(0, qsa_tokens * select_k * 4).unwrap();
        let d_sel_log = DeviceBuffer::alloc(0, qsa_tokens * select_k * 4).unwrap();
        let d_num_sel = DeviceBuffer::alloc(0, qsa_tokens * 4).unwrap();
        let d_seq_pos = DeviceBuffer::alloc(0, qsa_tokens * 4).unwrap();

        gpu_ops::qsa_select_blocks_f16(
            &ex,
            d_q_idx.addr(),
            d_comp_k.addr(),
            &attn_batch,
            d_sel_phys.addr(),
            d_sel_log.addr(),
            d_num_sel.addr(),
            d_seq_pos.addr(),
            4,
            128,
            block_size,
            select_k,
            8,
            16,
        )
        .unwrap();

        let h_num_sel: Vec<i32> = dev_to_vec(&d_num_sel, qsa_tokens, &stream);
        let h_seq_pos: Vec<i32> = dev_to_vec(&d_seq_pos, qsa_tokens, &stream);
        let h_sel_phys: Vec<i32> = dev_to_vec(&d_sel_phys, qsa_tokens * select_k, &stream);
        assert_eq!(h_num_sel, vec![2, 2]);
        assert_eq!(h_seq_pos, vec![20, 21]);
        assert_eq!(&h_sel_phys[0..2], &[2, 1]);

        let d_q_main = dev_from_slice(
            &vec![f32_to_f16(0.1); qsa_tokens * q_heads * head_dim],
            &stream,
        );
        let d_gate = dev_from_slice(
            &vec![f32_to_f16(0.0); qsa_tokens * q_heads * head_dim],
            &stream,
        );
        let d_attn_out = DeviceBuffer::alloc(0, qsa_tokens * q_heads * head_dim * 2).unwrap();

        gpu_ops::qsa_sparse_paged_attn_f16(
            &ex,
            d_q_main.addr(),
            d_gate.addr(),
            &kv_layout,
            d_sel_phys.addr(),
            d_sel_log.addr(),
            d_num_sel.addr(),
            d_seq_pos.addr(),
            0,
            d_attn_out.addr(),
            qsa_tokens,
            q_heads,
            select_k,
            attn_batch.sm_scale,
        )
        .unwrap();

        let h_attn_out: Vec<u16> =
            dev_to_vec(&d_attn_out, qsa_tokens * q_heads * head_dim, &stream);
        // Since V = 0.25 everywhere and sigmoid(0.0) = 0.5, gated attention output must equal 0.125!
        for &v in &h_attn_out {
            let vf = f16_to_f32(v);
            assert!((vf - 0.125).abs() < 1e-3, "expected 0.125, got {vf}");
        }

        // Verify hc_count = 2 scaling (1.0 / hc_count = 0.5, matching vLLM test_hc_ops.py)
        let d_hc2_down_in = dev_from_slice(&[f32_to_f16(2.0); 4], &stream);
        let d_hc2_down_out = DeviceBuffer::alloc(0, 4 * 2).unwrap();
        let d_hc2_inj_in = dev_from_slice(&[f32_to_f16(0.0); 2], &stream);
        let d_hc2_inj_out = DeviceBuffer::alloc(0, 2 * 4).unwrap();
        gpu_ops::hc_down_silu_inject_f16(
            &ex,
            d_hc2_down_in.addr(),
            d_hc2_down_out.addr(),
            d_hc2_inj_in.addr(),
            d_hc2_inj_out.addr(),
            0,
            1,
            4,
            2,
        )
        .unwrap();
        let h_hc2_down: Vec<u16> = dev_to_vec(&d_hc2_down_out, 4, &stream);
        // down = 2.0, hc_count = 2 -> x = 2.0 / 2 = 1.0 -> SiLU(1.0) = 1.0 / (1 + e^-1) = 0.7310586
        let expected_silu1 = 1.0f32 / (1.0f32 + (-1.0f32).exp());
        assert!(
            (f16_to_f32(h_hc2_down[0]) - expected_silu1).abs() < 2e-3,
            "expected SiLU(2.0 / 2) = {expected_silu1}, got {}",
            f16_to_f32(h_hc2_down[0])
        );

        // 4. Non-uniform QSA top-k block selection (4 blocks -> select_k = 2, recent_blocks = 1)
        //    + sparse paged GQA across FP16, Q8_0 (272 B/row), and TQ4 (132 B/row + Rademacher WHT256)
        //    against an exact CPU softmax + sigmoid-gate reference.
        let nu_pages = 4usize;
        let nu_seq_len = 50usize; // 4 blocks: blk 0 (0..16), blk 1 (16..32), blk 2 (32..48), blk 3 (48..50)
        let nu_select_k = 2usize; // must pick 1 highest-scoring past block + blk 3 (recent_blocks = 1)
        let nu_q_heads = 2usize;
        // Logical blocks [0, 1, 2, 3] -> physical pages [3, 1, 0, 2]
        let nu_blk_tbl = [3i32, 1, 0, 2];
        // Set compressed K so logical block 1 (phys page 1) has positive dot product with q_idx (+1.0),
        // while logical blocks 0 (phys 3) and 2 (phys 0) have negative dot products (-1.0).
        let mut h_nu_comp_k = vec![f32_to_f16(-0.5); nu_pages * 4 * 128];
        for sub in 0..4 {
            for d in 0..128 {
                h_nu_comp_k[(4 + sub) * 128 + d] = f32_to_f16(0.75);
            }
        }
        let d_nu_comp_k = dev_from_slice(&h_nu_comp_k, &stream);
        let d_nu_q_idx = dev_from_slice(&vec![f32_to_f16(0.5); 4 * 128], &stream);
        let d_nu_qsl = dev_from_slice(&[0i32, 1], &stream);
        let d_nu_ctx_before = dev_from_slice(&[(nu_seq_len - 1) as i32], &stream);
        let d_nu_seq_after = dev_from_slice(&[nu_seq_len as i32], &stream);
        let d_nu_blk_tbl = dev_from_slice(&nu_blk_tbl, &stream);
        let d_nu_slot_map = dev_from_slice(&[2i32 * 16 + 1], &stream);
        let nu_attn_batch = W34AttentionBatch {
            query_start_loc: d_nu_qsl.addr() as usize as *const i32,
            context_lens_before: d_nu_ctx_before.addr() as usize as *const i32,
            seq_lens_after: d_nu_seq_after.addr() as usize as *const i32,
            block_tables: d_nu_blk_tbl.addr() as usize as *const i32,
            write_slot_mapping: d_nu_slot_map.addr() as usize as *const i32,
            valid_token_mask: core::ptr::null(),
            valid_req_mask: core::ptr::null(),
            num_reqs: 1,
            total_tokens: 1,
            max_blocks_per_req: 4,
            max_q_len: 1,
            max_seq_len: nu_seq_len as i32,
            mask_mode: W34_ATTN_MASK_CAUSAL,
            sliding_window: -1,
            sm_scale: (head_dim as f32).powf(-0.5),
            reserved: 0,
        };
        let d_nu_sel_phys = DeviceBuffer::alloc(0, nu_select_k * 4).unwrap();
        let d_nu_sel_log = DeviceBuffer::alloc(0, nu_select_k * 4).unwrap();
        let d_nu_num_sel = DeviceBuffer::alloc(0, 4).unwrap();
        let d_nu_seq_pos = DeviceBuffer::alloc(0, 4).unwrap();
        gpu_ops::qsa_select_blocks_f16(
            &ex,
            d_nu_q_idx.addr(),
            d_nu_comp_k.addr(),
            &nu_attn_batch,
            d_nu_sel_phys.addr(),
            d_nu_sel_log.addr(),
            d_nu_num_sel.addr(),
            d_nu_seq_pos.addr(),
            4,
            128,
            block_size,
            nu_select_k,
            0, // sink_blocks = 0 (matching vLLM QSA)
            1, // recent_blocks = 1 (matching vLLM QSA)
        )
        .unwrap();
        let h_nu_num_sel: Vec<i32> = dev_to_vec(&d_nu_num_sel, 1, &stream);
        let h_nu_sel_log: Vec<i32> = dev_to_vec(&d_nu_sel_log, nu_select_k, &stream);
        let h_nu_sel_phys: Vec<i32> = dev_to_vec(&d_nu_sel_phys, nu_select_k, &stream);
        assert_eq!(h_nu_num_sel, vec![2]);
        // Must select logical block 1 (phys page 1, highest score) and logical block 3 (phys page 2, recent block)
        assert_eq!(h_nu_sel_log, vec![1, 3]);
        assert_eq!(h_nu_sel_phys, vec![1, 2]);

        // Populate all 50 tokens across the 4 pages with deterministic non-uniform K and V via w34_kv_cache_write_strided
        // for FP16, Q8_0 (272 B/row), and TQ4 (132 B/row + Rademacher WHT256).
        let h_nu_k: Vec<u16> = (0..nu_seq_len * head_dim)
            .map(|i| f32_to_f16((((i * 17 + 3) % 53) as f32 - 26.0) * 0.03))
            .collect();
        let h_nu_v: Vec<u16> = (0..nu_seq_len * head_dim)
            .map(|i| f32_to_f16((((i * 23 + 7) % 47) as f32 - 23.0) * 0.04))
            .collect();
        let h_nu_q: Vec<u16> = (0..nu_q_heads * head_dim)
            .map(|i| f32_to_f16((((i * 13 + 11) % 37) as f32 - 18.0) * 0.05))
            .collect();
        let h_nu_gate: Vec<u16> = (0..nu_q_heads * head_dim)
            .map(|i| f32_to_f16((((i * 19 + 5) % 29) as f32 - 14.0) * 0.08))
            .collect();
        let write_slots: Vec<i32> = (0..nu_seq_len)
            .map(|t| {
                let log_blk = t / block_size;
                let off = t % block_size;
                nu_blk_tbl[log_blk] * (block_size as i32) + (off as i32)
            })
            .collect();
        let d_nu_k_in = dev_from_slice(&h_nu_k, &stream);
        let d_nu_v_in = dev_from_slice(&h_nu_v, &stream);
        let d_nu_q = dev_from_slice(&h_nu_q, &stream);
        let d_nu_gate = dev_from_slice(&h_nu_gate, &stream);
        let d_write_slots = dev_from_slice(&write_slots, &stream);
        let d_write_qsl = dev_from_slice(&[0i32, nu_seq_len as i32], &stream);
        let d_write_ctx0 = dev_from_slice(&[0i32], &stream);
        let write_batch = W34AttentionBatch {
            query_start_loc: d_write_qsl.addr() as usize as *const i32,
            context_lens_before: d_write_ctx0.addr() as usize as *const i32,
            seq_lens_after: d_nu_seq_after.addr() as usize as *const i32,
            block_tables: d_nu_blk_tbl.addr() as usize as *const i32,
            write_slot_mapping: d_write_slots.addr() as usize as *const i32,
            max_q_len: nu_seq_len as i32,
            total_tokens: nu_seq_len as i32,
            ..nu_attn_batch
        };

        // Exact CPU reference over the selected logical blocks [1 (tokens 16..32), 3 (tokens 48..50)]:
        let selected_tokens: Vec<usize> = (16..32).chain(48..50).collect();
        let mut cpu_ref = vec![0.0f32; nu_q_heads * head_dim];
        for qh in 0..nu_q_heads {
            let mut logits = Vec::with_capacity(selected_tokens.len());
            for &t in &selected_tokens {
                let mut dot = 0.0f32;
                for d in 0..head_dim {
                    dot += f16_to_f32(h_nu_q[qh * head_dim + d])
                        * f16_to_f32(h_nu_k[t * head_dim + d]);
                }
                logits.push(dot * nu_attn_batch.sm_scale);
            }
            let max_l = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let exp_l: Vec<f32> = logits.iter().map(|&l| (l - max_l).exp()).collect();
            let sum_e: f32 = exp_l.iter().sum();
            for d in 0..head_dim {
                let mut acc = 0.0f32;
                for (idx, &t) in selected_tokens.iter().enumerate() {
                    acc += (exp_l[idx] / sum_e) * f16_to_f32(h_nu_v[t * head_dim + d]);
                }
                let g = f16_to_f32(h_nu_gate[qh * head_dim + d]);
                let sig = 1.0f32 / (1.0f32 + (-g).exp());
                cpu_ref[qh * head_dim + d] = acc * sig;
            }
        }

        for (fmt, xform, row_bytes, tol) in [
            (
                W34_KV_FORMAT_FP16,
                W34_KV_TRANSFORM_IDENTITY,
                head_dim * 2,
                1.5e-3f32,
            ),
            (
                W34_KV_FORMAT_Q8_0,
                W34_KV_TRANSFORM_IDENTITY,
                272usize,
                4.0e-3f32,
            ),
            (
                W34_KV_FORMAT_TQ4,
                W34_KV_TRANSFORM_RADEMACHER_WHT256,
                132usize,
                2.5e-2f32,
            ),
        ] {
            let d_kc = DeviceBuffer::alloc(0, nu_pages * block_size * row_bytes).unwrap();
            let d_vc = DeviceBuffer::alloc(0, nu_pages * block_size * row_bytes).unwrap();
            let layout = W34KvLayerLayout {
                k_base: d_kc.addr(),
                v_base: d_vc.addr(),
                k_page_stride_bytes: (block_size * row_bytes) as u64,
                k_head_stride_bytes: (block_size * row_bytes) as u64,
                k_token_stride_bytes: row_bytes as u64,
                v_page_stride_bytes: (block_size * row_bytes) as u64,
                v_head_stride_bytes: (block_size * row_bytes) as u64,
                v_token_stride_bytes: row_bytes as u64,
                namespace_id: W34_KV_NAMESPACE_TARGET_FULL_ATTN,
                layer_id: 3,
                local_kv_heads: 1,
                head_dim: head_dim as i32,
                block_size: block_size as i32,
                k_format: fmt,
                v_format: fmt,
                k_transform: xform,
                v_transform: xform,
                reserved: 0,
            };
            // SAFETY: `d_kc`, `d_vc`, `d_nu_k_in`, `d_nu_v_in`, and `write_batch` buffers are live
            // and sized for `nu_pages * block_size` rows of `row_bytes` bytes.
            unsafe {
                assert_eq!(
                    w34_kv_cache_write_strided(
                        &ex,
                        &layout,
                        &write_batch,
                        d_nu_k_in.addr() as *const c_void,
                        d_nu_v_in.addr() as *const c_void,
                        head_dim as i32,
                        head_dim as i32,
                    ),
                    0
                );
            }
            let d_nu_out = DeviceBuffer::alloc(0, nu_q_heads * head_dim * 2).unwrap();
            gpu_ops::qsa_sparse_paged_attn_f16(
                &ex,
                d_nu_q.addr(),
                d_nu_gate.addr(),
                &layout,
                d_nu_sel_phys.addr(),
                d_nu_sel_log.addr(),
                d_nu_num_sel.addr(),
                d_nu_seq_pos.addr(),
                0,
                d_nu_out.addr(),
                1,
                nu_q_heads,
                nu_select_k,
                nu_attn_batch.sm_scale,
            )
            .unwrap();
            let h_nu_out: Vec<u16> = dev_to_vec(&d_nu_out, nu_q_heads * head_dim, &stream);
            let max_err = h_nu_out
                .iter()
                .zip(cpu_ref.iter())
                .map(|(&got, &exp)| (f16_to_f32(got) - exp).abs())
                .fold(0.0f32, f32::max);
            assert!(
                max_err < tol,
                "QSA sparse paged attn format={fmt} max_err={max_err} exceeded tol={tol}"
            );
        }

        // 5. Verify QSA Indexer 4:1 raw-key pooling -> GemmaRMSNorm(k_norm_weight) -> mRoPE(first_pos)
        //    order on non-uniform 4-token inputs (ensuring pool-then-norm-rope is tested and distinct
        //    from norm-rope-then-pool), plus hc_count = 6 (> 4, up to MAX_HC = 8) on hc_mix, hc_combine,
        //    and ple_signed_sqrt_gate.
        let g_tokens = 4usize;
        let g_qk_in: Vec<u16> = (0..g_tokens * 5 * 128)
            .map(|i| {
                let tok = i / (5 * 128);
                let d = i % 128;
                let scale = match tok {
                    0 => 0.2f32,
                    1 => 1.5f32,
                    2 => -0.8f32,
                    _ => 2.1f32,
                };
                f32_to_f16((((d * 11 + tok * 29 + 3) % 41) as f32 - 20.0) * 0.05 * scale)
            })
            .collect();
        let g_k_norm_w: Vec<u16> = (0..128)
            .map(|d| f32_to_f16((((d * 7 + 5) % 23) as f32 - 11.0) * 0.04))
            .collect();
        let g_pos = [
            100i32, 101, 102, 103, 100, 101, 102, 103, 100, 101, 102, 103,
        ];
        let g_slots = [8i32, 9, 10, 11]; // sub_block = 2 (slots 8..12)
        let d_g_qk = dev_from_slice(&g_qk_in, &stream);
        let d_g_knw = dev_from_slice(&g_k_norm_w, &stream);
        let d_g_pos = dev_from_slice(&g_pos, &stream);
        let d_g_slots = dev_from_slice(&g_slots, &stream);
        let d_g_q_out = DeviceBuffer::alloc(0, g_tokens * 4 * 128 * 2).unwrap();
        let d_g_k_out = DeviceBuffer::alloc(0, g_tokens * 128 * 2).unwrap();
        let d_g_raw_k = DeviceBuffer::alloc(0, 16 * 128 * 2).unwrap();
        let d_g_comp_k = DeviceBuffer::alloc(0, 4 * 128 * 2).unwrap();
        gpu_ops::qsa_indexer_norm_rope_f16(
            &ex,
            d_g_qk.addr(),
            d_g_knw.addr(),
            d_g_knw.addr(),
            d_g_pos.addr(),
            d_g_q_out.addr(),
            d_g_k_out.addr(),
            0,
            g_tokens,
            4,
            1,
            128,
            64,
            10_000_000.0,
            1e-6,
            11,
            11,
            g_tokens,
        )
        .unwrap();
        gpu_ops::qsa_indexer_cache_update_f16(
            &ex,
            d_g_k_out.addr(),
            d_g_raw_k.addr(),
            d_g_comp_k.addr(),
            d_g_knw.addr(),
            d_g_pos.addr(),
            d_g_slots.addr(),
            0,
            g_tokens,
            128,
            64,
            10_000_000.0,
            1e-6,
            11,
            11,
            g_tokens,
        )
        .unwrap();
        let h_g_comp_k: Vec<u16> = dev_to_vec(&d_g_comp_k, 4 * 128, &stream);
        let got_sub2 = &h_g_comp_k[2 * 128..3 * 128];

        // CPU reference for pool(4 raw keys) -> GemmaRMSNorm -> mRoPE(first_pos = 100):
        let mut pooled_raw = [0.0f32; 128];
        for d in 0..128 {
            let mut s = 0.0f32;
            for tok in 0..4 {
                s += f16_to_f32(g_qk_in[(tok * 5 + 4) * 128 + d]);
            }
            pooled_raw[d] = f16_to_f32(f32_to_f16(0.25f32 * s));
        }
        let sum_sq: f32 = pooled_raw.iter().map(|&x| x * x).sum();
        let inv_rms = (sum_sq / 128.0f32 + 1e-6f32).powf(-0.5);
        let mut normed_pooled = [0.0f32; 128];
        for d in 0..128 {
            normed_pooled[d] = f16_to_f32(f32_to_f16(
                pooled_raw[d] * inv_rms * (1.0f32 + f16_to_f32(g_k_norm_w[d])),
            ));
        }
        let mut ref_comp_k = normed_pooled;
        for i in 0..32 {
            let inv_freq = 1.0f32 / 10_000_000.0f32.powf((2 * i) as f32 / 64.0f32);
            let angle = 100.0f32 * inv_freq; // first_pos = 100
            let sin_a = f16_to_f32(f32_to_f16(angle.sin()));
            let cos_a = f16_to_f32(f32_to_f16(angle.cos()));
            let x1 = normed_pooled[i];
            let x2 = normed_pooled[i + 32];
            ref_comp_k[i] = x1 * cos_a - x2 * sin_a;
            ref_comp_k[i + 32] = x2 * cos_a + x1 * sin_a;
        }
        for d in 0..128 {
            let diff = (f16_to_f32(got_sub2[d]) - ref_comp_k[d]).abs();
            assert!(
                diff < 2e-3,
                "comp_k pool->norm->rope mismatch at d={d}: got={} ref={}",
                f16_to_f32(got_sub2[d]),
                ref_comp_k[d]
            );
        }

        // Verify hc_count = 6 (up to MAX_HC = 8) on hc_mix_streams_f16, hc_combine_f16, and ple_signed_sqrt_gate_f16:
        let hc6 = 6usize;
        let h6 = 64usize;
        let d_res6 = dev_from_slice(&vec![f32_to_f16(1.0); hc6 * h6], &stream);
        let d_up6 = dev_from_slice(&vec![f32_to_f16(0.0); hc6 * h6], &stream);
        let d_mix6 = DeviceBuffer::alloc(0, h6 * 2).unwrap();
        gpu_ops::hc_mix_streams_f16(
            &ex,
            d_res6.addr(),
            d_up6.addr(),
            d_mix6.addr(),
            0,
            1,
            hc6,
            h6,
        )
        .unwrap();
        let h_mix6: Vec<u16> = dev_to_vec(&d_mix6, h6, &stream);
        // 6 streams of 1.0 with gate = sigmoid(0.0) / 6 = 0.5 / 6 -> sum = 6 * (0.5 / 6) * 1.0 = 0.5
        assert!((f16_to_f32(h_mix6[0]) - 0.5).abs() < 1e-3);

        let d_blk6 = dev_from_slice(&vec![f32_to_f16(2.0); h6], &stream);
        let d_inj6 = dev_from_slice(&vec![0.5f32; hc6], &stream);
        let d_comb6 = DeviceBuffer::alloc(0, hc6 * h6 * 2).unwrap();
        gpu_ops::hc_combine_f16(
            &ex,
            d_res6.addr(),
            d_blk6.addr(),
            d_inj6.addr(),
            0,
            d_comb6.addr(),
            0,
            1,
            hc6,
            h6,
        )
        .unwrap();
        let h_comb6: Vec<u16> = dev_to_vec(&d_comb6, hc6 * h6, &stream);
        assert!((f16_to_f32(h_comb6[5 * h6]) - 2.0).abs() < 1e-3);

        let d_ple_out6 = DeviceBuffer::alloc(0, hc6 * h6 * 2).unwrap();
        gpu_ops::ple_signed_sqrt_gate_f16(
            &ex,
            d_up6.addr(),
            d_up6.addr(),
            d_blk6.addr(),
            d_ple_out6.addr(),
            0,
            1,
            hc6,
            h6,
            1e-6,
        )
        .unwrap();
        let h_ple_out6: Vec<u16> = dev_to_vec(&d_ple_out6, hc6 * h6, &stream);
        // dot = 0 -> gate = sigmoid(0) = 0.5 -> 0.5 * 2.0 = 1.0 across all 6 streams
        assert!((f16_to_f32(h_ple_out6[5 * h6]) - 1.0).abs() < 1e-3);
    }
}
