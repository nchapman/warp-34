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
    scale_offset_bytes: usize,
    rows: usize,
}

/// Number of entries in a 4-bit `lut4-gptq` PLE codebook (`weight_lut`).
pub const PLE_LUT4_CODEBOOK_LEN: usize = 16;
/// Signed-sqrt clamp epsilon for PLE attention gating (`sgn(dot) * sqrt(max(|dot|, eps))`).
pub const PLE_SIGNED_SQRT_GATE_EPS: f32 = 1e-6;
/// Minimum batch of n-gram IDs before issuing coalesced `MADV_WILLNEED` prefaulting.
const PLE_MMAP_PREFAULT_MIN_IDS: usize = 64;
/// Standard OS virtual memory page size in bytes used for `madvise` alignment.
const PLE_OS_PAGE_BYTES: usize = 4096;

/// On-disk storage format of a PLE n-gram embedding table.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum PleTableFormat {
    /// `F8_E4M3` (`shard_{part}.weight` `U8 [rows, head_dim]` + scalar `weight_scale`).
    F8E4m3,
    /// 4-bit LUT-GPTQ (`shard_{part}.weight_packed` `U8 [rows, head_dim / 2]` +
    /// `shard_{part}.weight_scale` `F16 [rows, head_dim / group_size]` +
    /// `shard_{part}.weight_lut` `F32 [16]`).
    Lut4Gptq {
        group_size: usize,
        lut_f32: [f32; PLE_LUT4_CODEBOOK_LEN],
    },
}

/// Zero-copy read-only `mmap` table over the 128 PLE n-gram embedding shards
/// (`F8_E4M3` or 4-bit `lut4-gptq`).
///
/// Consumes `0 B` of anonymous system RAM and `0 B` of GPU VRAM: the OS page cache serves hot
/// n-gram pages directly from NVMe with `MADV_RANDOM` + coalesced `MADV_WILLNEED` prefaulting.
pub struct PleMmapTable {
    mmaps: Vec<MmapRegion>,
    shards: Vec<ShardView>,
    pub shard_rows: usize,
    pub head_dim: usize,
    pub lut: F8E4m3ScaleLut,
    pub format: PleTableFormat,
    gather_cache: std::sync::Mutex<(Vec<i64>, Vec<u8>)>,
}

impl PleMmapTable {
    fn ensure_mmap(
        archive: &SafetensorsArchive,
        shard_file: &str,
        file_to_mmap: &mut HashMap<String, usize>,
        mmaps: &mut Vec<MmapRegion>,
    ) -> io::Result<usize> {
        if let Some(&idx) = file_to_mmap.get(shard_file) {
            return Ok(idx);
        }
        let file = archive.shard_file(shard_file).ok_or_else(|| {
            io::Error::new(
                io::ErrorKind::NotFound,
                format!("missing safetensors file for shard_file={shard_file}"),
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
        file_to_mmap.insert(shard_file.to_owned(), idx);
        Ok(idx)
    }

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

        if archive
            .resolve_name(&format!("{prefix}.shard_0.weight_packed"))
            .is_some()
        {
            return Self::open_lut4_gptq(archive, &prefix, split_parts, head_dim);
        }

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
            let mmap_idx =
                Self::ensure_mmap(archive, &info.shard_file, &mut file_to_mmap, &mut mmaps)?;
            let offset_bytes = info.offset as usize;
            if offset_bytes.saturating_add(rows.saturating_mul(head_dim)) > mmaps[mmap_idx].len {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    format!("PLE shard {tname} exceeds mmap file bounds"),
                ));
            }
            shards.push(ShardView {
                mmap_idx,
                offset_bytes,
                scale_offset_bytes: 0,
                rows,
            });
        }

        Ok(Self {
            mmaps,
            shards,
            shard_rows,
            head_dim,
            lut: F8E4m3ScaleLut::new(weight_scale),
            format: PleTableFormat::F8E4m3,
            gather_cache: std::sync::Mutex::new((Vec::new(), Vec::new())),
        })
    }

    fn open_lut4_gptq(
        archive: &SafetensorsArchive,
        prefix: &str,
        split_parts: usize,
        head_dim: usize,
    ) -> io::Result<Self> {
        if !head_dim.is_multiple_of(2) || head_dim == 0 {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!("lut4-gptq PLE requires even non-zero head_dim, got {head_dim}"),
            ));
        }
        let packed_cols = head_dim / 2;
        let lut0_name = format!("{prefix}.shard_0.weight_lut");
        let lut0_vec = archive.read_f32(&lut0_name)?;
        if lut0_vec.len() != PLE_LUT4_CODEBOOK_LEN {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!(
                    "expected {PLE_LUT4_CODEBOOK_LEN}-entry weight_lut for {lut0_name}, got len={}",
                    lut0_vec.len()
                ),
            ));
        }
        let mut lut_f32 = [0.0f32; PLE_LUT4_CODEBOOK_LEN];
        lut_f32.copy_from_slice(&lut0_vec);

        let mut file_to_mmap: HashMap<String, usize> = HashMap::new();
        let mut mmaps: Vec<MmapRegion> = Vec::new();
        let mut shards: Vec<ShardView> = Vec::with_capacity(split_parts);
        let mut shard_rows = 0usize;
        let mut group_size = 0usize;

        for part in 0..split_parts {
            let s_pfx = format!("{prefix}.shard_{part}");
            let packed_name = format!("{s_pfx}.weight_packed");
            let scale_name = format!("{s_pfx}.weight_scale");
            let lut_name = format!("{s_pfx}.weight_lut");

            let lut_vec = archive.read_f32(&lut_name)?;
            if lut_vec != lut0_vec {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    format!("inconsistent PLE weight_lut in {lut_name} vs shard_0"),
                ));
            }

            let packed_info = archive.tensor_info(&packed_name)?;
            if packed_info.dtype != "U8" {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    format!("expected U8 for {packed_name}, got {}", packed_info.dtype),
                ));
            }
            if packed_info.shape.len() != 2 || packed_info.shape[1] != packed_cols {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    format!(
                        "unexpected PLE lut4 packed shape {:?} for {packed_name} (expected [rows, {packed_cols}])",
                        packed_info.shape
                    ),
                ));
            }
            let rows = packed_info.shape[0];
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

            let scale_info = archive.tensor_info(&scale_name)?;
            if scale_info.dtype != "F16" {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    format!("expected F16 for {scale_name}, got {}", scale_info.dtype),
                ));
            }
            if scale_info.shape.len() != 2
                || scale_info.shape[0] != rows
                || scale_info.shape[1] == 0
                || !head_dim.is_multiple_of(scale_info.shape[1])
            {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    format!(
                        "unexpected PLE lut4 scale shape {:?} for {scale_name} (rows={rows}, head_dim={head_dim})",
                        scale_info.shape
                    ),
                ));
            }
            let num_groups = scale_info.shape[1];
            let part_group_size = head_dim / num_groups;
            if part == 0 {
                group_size = part_group_size;
            } else if part_group_size != group_size {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    format!(
                        "inconsistent PLE lut4 group_size: shard_0={group_size}, shard_{part}={part_group_size}"
                    ),
                ));
            }

            let packed_mmap_idx = Self::ensure_mmap(
                archive,
                &packed_info.shard_file,
                &mut file_to_mmap,
                &mut mmaps,
            )?;
            let scale_mmap_idx = Self::ensure_mmap(
                archive,
                &scale_info.shard_file,
                &mut file_to_mmap,
                &mut mmaps,
            )?;
            if packed_mmap_idx != scale_mmap_idx {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    format!(
                        "PLE lut4 shard_{part} weight_packed and weight_scale must reside in the same shard file"
                    ),
                ));
            }
            let offset_bytes = packed_info.offset as usize;
            let scale_offset_bytes = scale_info.offset as usize;
            let scale_row_bytes = num_groups * 2;
            let region_len = mmaps[packed_mmap_idx].len;
            if offset_bytes.saturating_add(rows.saturating_mul(packed_cols)) > region_len
                || scale_offset_bytes.saturating_add(rows.saturating_mul(scale_row_bytes))
                    > region_len
            {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    format!("PLE lut4 shard_{part} exceeds mmap file bounds"),
                ));
            }

            shards.push(ShardView {
                mmap_idx: packed_mmap_idx,
                offset_bytes,
                scale_offset_bytes,
                rows,
            });
        }

        Ok(Self {
            mmaps,
            shards,
            shard_rows,
            head_dim,
            lut: F8E4m3ScaleLut::new(1.0),
            format: PleTableFormat::Lut4Gptq {
                group_size,
                lut_f32,
            },
            gather_cache: std::sync::Mutex::new((Vec::new(), Vec::new())),
        })
    }

    /// Number of staged bytes gathered per n-gram head lookup (`160 B` for `F8_E4M3`,
    /// `90 B` = `80 B` packed nibbles + `10 B` FP16 scales for `lut4-gptq` with `head_dim=160, g=32`).
    #[inline]
    pub fn staged_bytes_per_head(&self) -> usize {
        match self.format {
            PleTableFormat::F8E4m3 => self.head_dim,
            PleTableFormat::Lut4Gptq { group_size, .. } => {
                (self.head_dim / 2) + (self.head_dim / group_size) * 2
            }
        }
    }

    /// Gathers raw staged PLE bytes (`[ids.len(), staged_bytes_per_head()]`) with coalesced
    /// `MADV_WILLNEED` prefaulting for batches of `>= 64` row lookups and cross-rank caching
    /// when `TP > 1` ranks gather identical `ids`.
    pub fn gather_f8(&self, ids: &[i64], out_f8: &mut [u8]) {
        let head_stride = self.staged_bytes_per_head();
        assert_eq!(out_f8.len(), ids.len() * head_stride);
        if ids.is_empty() {
            return;
        }
        if let Ok(mut cache) = self.gather_cache.lock() {
            if cache.0.as_slice() == ids && cache.1.len() == out_f8.len() {
                out_f8.copy_from_slice(&cache.1);
                return;
            }
            self.gather_f8_uncached(ids, out_f8);
            cache.0.clear();
            cache.0.extend_from_slice(ids);
            cache.1.clear();
            cache.1.extend_from_slice(out_f8);
            return;
        }
        self.gather_f8_uncached(ids, out_f8);
    }

    fn gather_f8_uncached(&self, ids: &[i64], out_f8: &mut [u8]) {
        let head_stride = self.staged_bytes_per_head();
        if ids.len() >= PLE_MMAP_PREFAULT_MIN_IDS {
            let cap = match self.format {
                PleTableFormat::F8E4m3 => ids.len(),
                PleTableFormat::Lut4Gptq { .. } => ids.len() * 2,
            };
            let mut page_keys: Vec<(usize, usize)> = Vec::with_capacity(cap);
            for &id in ids {
                let uid = id.max(0) as usize;
                let shard_idx = (uid / self.shard_rows).min(self.shards.len() - 1);
                let local_row = (uid % self.shard_rows).min(self.shards[shard_idx].rows - 1);
                let view = self.shards[shard_idx];
                match self.format {
                    PleTableFormat::F8E4m3 => {
                        let byte_off = view.offset_bytes + local_row * self.head_dim;
                        page_keys.push((view.mmap_idx, byte_off / PLE_OS_PAGE_BYTES));
                    }
                    PleTableFormat::Lut4Gptq { group_size, .. } => {
                        let packed_bytes = self.head_dim / 2;
                        let scale_bytes = (self.head_dim / group_size) * 2;
                        let p_off = view.offset_bytes + local_row * packed_bytes;
                        let s_off = view.scale_offset_bytes + local_row * scale_bytes;
                        page_keys.push((view.mmap_idx, p_off / PLE_OS_PAGE_BYTES));
                        page_keys.push((view.mmap_idx, s_off / PLE_OS_PAGE_BYTES));
                    }
                }
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
                let adv_off = start_page * PLE_OS_PAGE_BYTES;
                let adv_len = ((end_page - start_page) * PLE_OS_PAGE_BYTES)
                    .min(region.len.saturating_sub(adv_off));
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

        match self.format {
            PleTableFormat::F8E4m3 => {
                for (i, &id) in ids.iter().enumerate() {
                    let uid = id.max(0) as usize;
                    let shard_idx = (uid / self.shard_rows).min(self.shards.len() - 1);
                    let local_row = (uid % self.shard_rows).min(self.shards[shard_idx].rows - 1);
                    let view = self.shards[shard_idx];
                    let region = &self.mmaps[view.mmap_idx];
                    let byte_off = view.offset_bytes + local_row * self.head_dim;
                    // SAFETY: Bounds checked at `PleMmapTable::open` (`offset + rows * head_dim <= region.len`).
                    let src = unsafe {
                        core::slice::from_raw_parts(
                            region.ptr.as_ptr().add(byte_off),
                            self.head_dim,
                        )
                    };
                    out_f8[i * self.head_dim..(i + 1) * self.head_dim].copy_from_slice(src);
                }
            }
            PleTableFormat::Lut4Gptq { group_size, .. } => {
                let packed_bytes = self.head_dim / 2;
                let scale_bytes = (self.head_dim / group_size) * 2;
                for (i, &id) in ids.iter().enumerate() {
                    let uid = id.max(0) as usize;
                    let shard_idx = (uid / self.shard_rows).min(self.shards.len() - 1);
                    let local_row = (uid % self.shard_rows).min(self.shards[shard_idx].rows - 1);
                    let view = self.shards[shard_idx];
                    let region = &self.mmaps[view.mmap_idx];
                    let p_off = view.offset_bytes + local_row * packed_bytes;
                    let s_off = view.scale_offset_bytes + local_row * scale_bytes;
                    // SAFETY: Bounds checked at `PleMmapTable::open_lut4_gptq`.
                    let src_packed = unsafe {
                        core::slice::from_raw_parts(region.ptr.as_ptr().add(p_off), packed_bytes)
                    };
                    // SAFETY: Bounds checked at `PleMmapTable::open_lut4_gptq`.
                    let src_scale = unsafe {
                        core::slice::from_raw_parts(region.ptr.as_ptr().add(s_off), scale_bytes)
                    };
                    let dst = &mut out_f8[i * head_stride..(i + 1) * head_stride];
                    dst[..packed_bytes].copy_from_slice(src_packed);
                    dst[packed_bytes..].copy_from_slice(src_scale);
                }
            }
        }
    }

    /// Gathers and dequantizes rows into `FP16` (`u16` bits).
    pub fn gather_dequant_f16(&self, ids: &[i64], out_f16: &mut [u16]) {
        assert_eq!(out_f16.len(), ids.len() * self.head_dim);
        let head_stride = self.staged_bytes_per_head();
        let mut staged = vec![0u8; ids.len() * head_stride];
        self.gather_f8(ids, &mut staged);
        match self.format {
            PleTableFormat::F8E4m3 => {
                for (dst, &b) in out_f16.iter_mut().zip(staged.iter()) {
                    *dst = self.lut.f16_lut[b as usize];
                }
            }
            PleTableFormat::Lut4Gptq {
                group_size,
                lut_f32,
            } => {
                let packed_bytes = self.head_dim / 2;
                for i in 0..ids.len() {
                    let head_raw = &staged[i * head_stride..(i + 1) * head_stride];
                    let dst_head = &mut out_f16[i * self.head_dim..(i + 1) * self.head_dim];
                    for (col, dst) in dst_head.iter_mut().enumerate() {
                        let byte = head_raw[col >> 1];
                        let nibble = if (col & 1) != 0 {
                            (byte >> 4) & 0x0F
                        } else {
                            byte & 0x0F
                        } as usize;
                        let sb_off = packed_bytes + (col / group_size) * 2;
                        let scale = crate::cuda::f16_to_f32(u16::from_le_bytes([
                            head_raw[sb_off],
                            head_raw[sb_off + 1],
                        ]));
                        *dst = crate::cuda::f32_to_f16(lut_f32[nibble] * scale);
                    }
                }
            }
        }
    }

    /// Gathers and dequantizes rows into `f32`.
    pub fn gather_dequant_f32(&self, ids: &[i64], out_f32: &mut [f32]) {
        assert_eq!(out_f32.len(), ids.len() * self.head_dim);
        let head_stride = self.staged_bytes_per_head();
        let mut staged = vec![0u8; ids.len() * head_stride];
        self.gather_f8(ids, &mut staged);
        match self.format {
            PleTableFormat::F8E4m3 => {
                for (dst, &b) in out_f32.iter_mut().zip(staged.iter()) {
                    *dst = self.lut.f32_lut[b as usize];
                }
            }
            PleTableFormat::Lut4Gptq {
                group_size,
                lut_f32,
            } => {
                let packed_bytes = self.head_dim / 2;
                for i in 0..ids.len() {
                    let head_raw = &staged[i * head_stride..(i + 1) * head_stride];
                    let dst_head = &mut out_f32[i * self.head_dim..(i + 1) * self.head_dim];
                    for (col, dst) in dst_head.iter_mut().enumerate() {
                        let byte = head_raw[col >> 1];
                        let nibble = if (col & 1) != 0 {
                            (byte >> 4) & 0x0F
                        } else {
                            byte & 0x0F
                        } as usize;
                        let sb_off = packed_bytes + (col / group_size) * 2;
                        let scale = crate::cuda::f16_to_f32(u16::from_le_bytes([
                            head_raw[sb_off],
                            head_raw[sb_off + 1],
                        ]));
                        *dst = lut_f32[nibble] * scale;
                    }
                }
            }
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
        if !matches!(tp_size, 1 | 2 | 4) {
            return Err(format!(
                "unsupported tp_size={tp_size} (expected 1, 2, or 4)"
            ));
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
        } else if tp_size == 2 {
            max_hot_per_layer.min(104)
        } else {
            max_hot_per_layer.min(local_total_experts_per_layer)
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
pub mod gpu_ops;

#[cfg(test)]
mod tests;
