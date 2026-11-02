//! Descriptor-driven `KvPool` (`docs/contracts/cache-layout.md`) and transactional
//! `StateBundlePool` / `StateTx` (`docs/contracts/state-transaction.md`).
#![allow(clippy::too_many_arguments)]

#[cfg(feature = "cuda")]
use crate::cuda::{DeviceBuffer, Result};
use w34_kernels::ffi::{
    W34_KV_FORMAT_BF16, W34_KV_FORMAT_FP16, W34_KV_FORMAT_Q8_0, W34_KV_FORMAT_TQ4,
    W34_KV_TRANSFORM_IDENTITY, W34_KV_TRANSFORM_RADEMACHER_WHT256,
};
#[cfg(feature = "cuda")]
use w34_kernels::ffi::{W34_KV_NAMESPACE_TARGET_FULL_ATTN, W34KvLayerLayout};

/// Policy governing KV cache storage precision and transforms across layers.
///
/// # Examples
///
/// ```
/// use w34_host::cache::KvFormatPolicy;
///
/// let policy = KvFormatPolicy::Tq4;
/// let bytes = policy.bytes_per_token(16, 8, 256);
/// assert!(bytes > 0);
/// ```
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum KvFormatPolicy {
    Fp16,
    Bf16,
    /// Plain 8-bit (`Q8_0` `272 B`/row for both `K` and `V`, `34,816 B/token`, `1.88x` compression on 27B).
    Q8_0,
    /// Boundary-anchored TurboQuant 4-bit (default): first 2 (`0, 1`) and last 2 (`N - 2, N - 1`)
    /// full-attention layers use WHT-256 rotated 8-bit `Q8_0`; interior layers use WHT-256 rotated
    /// 4-bit `TQ4` with per-128 least-squares scales (`21,376 B/token`, `3.07x` compression on 27B —
    /// fits full `262k` context on 1x 3090 Ti while matching `FP16` oracle accuracy).
    Tq4,
    /// Uniform all-layer TurboQuant 4-bit (`132 B`/row on every full-attention layer,
    /// `16,896 B/token`, `3.88x` compression on 27B).
    Tq4All,
}

impl KvFormatPolicy {
    pub fn resolve_layer(
        self,
        full_attn_layer_ordinal: usize,
        num_full_attn_layers: usize,
    ) -> (i32, i32, i32, i32) {
        match self {
            Self::Fp16 => (
                W34_KV_FORMAT_FP16,
                W34_KV_FORMAT_FP16,
                W34_KV_TRANSFORM_IDENTITY,
                W34_KV_TRANSFORM_IDENTITY,
            ),
            Self::Bf16 => (
                W34_KV_FORMAT_BF16,
                W34_KV_FORMAT_BF16,
                W34_KV_TRANSFORM_IDENTITY,
                W34_KV_TRANSFORM_IDENTITY,
            ),
            Self::Q8_0 => (
                W34_KV_FORMAT_Q8_0,
                W34_KV_FORMAT_Q8_0,
                W34_KV_TRANSFORM_IDENTITY,
                W34_KV_TRANSFORM_IDENTITY,
            ),
            Self::Tq4 => {
                let is_boundary = full_attn_layer_ordinal < 2
                    || (num_full_attn_layers >= 2
                        && full_attn_layer_ordinal >= num_full_attn_layers - 2);
                if is_boundary {
                    (
                        W34_KV_FORMAT_Q8_0,
                        W34_KV_FORMAT_Q8_0,
                        W34_KV_TRANSFORM_RADEMACHER_WHT256,
                        W34_KV_TRANSFORM_RADEMACHER_WHT256,
                    )
                } else {
                    (
                        W34_KV_FORMAT_TQ4,
                        W34_KV_FORMAT_TQ4,
                        W34_KV_TRANSFORM_RADEMACHER_WHT256,
                        W34_KV_TRANSFORM_RADEMACHER_WHT256,
                    )
                }
            }
            Self::Tq4All => (
                W34_KV_FORMAT_TQ4,
                W34_KV_FORMAT_TQ4,
                W34_KV_TRANSFORM_RADEMACHER_WHT256,
                W34_KV_TRANSFORM_RADEMACHER_WHT256,
            ),
        }
    }

    pub fn bytes_per_token(
        self,
        num_full_attn_layers: usize,
        kv_heads: usize,
        head_dim: usize,
    ) -> usize {
        (0..num_full_attn_layers)
            .map(|ord| {
                let (k_fmt, v_fmt, _, _) = self.resolve_layer(ord, num_full_attn_layers);
                kv_heads
                    * (kv_format_row_bytes(k_fmt, head_dim) + kv_format_row_bytes(v_fmt, head_dim))
            })
            .sum()
    }
}

/// Returns the row bytes for a single attention head in the specified format.
///
/// # Examples
///
/// ```
/// use w34_host::cache::kv_format_row_bytes;
/// use w34_kernels::ffi::W34_KV_FORMAT_FP16;
///
/// assert_eq!(kv_format_row_bytes(W34_KV_FORMAT_FP16, 256), 512);
/// ```
#[inline]
pub fn kv_format_row_bytes(format: i32, head_dim: usize) -> usize {
    match format {
        W34_KV_FORMAT_FP16 | W34_KV_FORMAT_BF16 => head_dim * 2,
        W34_KV_FORMAT_TQ4 => 66 * (head_dim / 128),
        W34_KV_FORMAT_Q8_0 => 68 * (head_dim / 64),
        _ => head_dim * 2,
    }
}

#[cfg(feature = "cuda")]
pub struct KvLayerBuffer {
    pub k_buf: DeviceBuffer,
    pub v_buf: DeviceBuffer,
    pub layout: W34KvLayerLayout,
}

/// Ascending physical-order free-page bitmap (`O(1)` amortized scan via `next_word`).
///
/// Always allocates the lowest free `page_id` so batch eviction from `MambaRadixCache`
/// coalesces freed leaf pages into monotonically increasing GDDR6X runs rather than
/// LIFO-reversed scattered leaf order.
#[cfg(any(feature = "cuda", test))]
#[derive(Clone, Debug)]
pub(crate) struct PageBitmap {
    num_pages: usize,
    words: Vec<u64>,
    free_count: usize,
    next_word: usize,
}

#[cfg(any(feature = "cuda", test))]
impl PageBitmap {
    pub(crate) fn new(num_pages: usize) -> Self {
        let mut bm = Self {
            num_pages,
            words: vec![0u64; num_pages.div_ceil(64)],
            free_count: 0,
            next_word: 0,
        };
        bm.reset();
        bm
    }

    pub(crate) fn free_count(&self) -> usize {
        self.free_count
    }

    pub(crate) fn reset(&mut self) {
        let full_words = self.num_pages / 64;
        let rem = self.num_pages % 64;
        for w in &mut self.words[..full_words] {
            *w = u64::MAX;
        }
        if rem > 0 {
            self.words[full_words] = (1u64 << rem) - 1;
        }
        self.free_count = self.num_pages;
        self.next_word = 0;
    }

    pub(crate) fn alloc(&mut self) -> Option<i32> {
        if self.free_count == 0 {
            return None;
        }
        let n = self.words.len();
        for pass in 0..2 {
            let range = if pass == 0 {
                self.next_word..n
            } else {
                0..self.next_word
            };
            for w in range {
                let word = self.words[w];
                if word != 0 {
                    let bit = word.trailing_zeros() as usize;
                    self.words[w] = word & !(1u64 << bit);
                    self.free_count -= 1;
                    self.next_word = w;
                    return Some((w * 64 + bit) as i32);
                }
            }
        }
        None
    }

    pub(crate) fn free(&mut self, page_id: i32) {
        if page_id < 0 {
            return;
        }
        let idx = page_id as usize;
        if idx >= self.num_pages {
            return;
        }
        let w = idx / 64;
        let mask = 1u64 << (idx % 64);
        if (self.words[w] & mask) == 0 {
            self.words[w] |= mask;
            self.free_count += 1;
            if w < self.next_word {
                self.next_word = w;
            }
        }
    }
}

#[cfg(feature = "cuda")]
pub struct KvPool {
    pub device_id: i32,
    pub num_pages: usize,
    pub block_size: usize,
    pub layers: Vec<KvLayerBuffer>,
    free_pages: PageBitmap,
}

#[cfg(feature = "cuda")]
impl KvPool {
    pub fn new(
        device_id: i32,
        namespace_id: i32,
        layer_ids: &[usize],
        local_kv_heads: usize,
        head_dim: usize,
        block_size: usize,
        num_pages: usize,
        policy: KvFormatPolicy,
    ) -> Result<Self> {
        let num_full = layer_ids.len();
        let mut layers = Vec::with_capacity(num_full);
        for (ord, &layer_id) in layer_ids.iter().enumerate() {
            let (k_format, v_format, k_transform, v_transform) =
                policy.resolve_layer(ord, num_full);
            let k_tok_bytes = kv_format_row_bytes(k_format, head_dim);
            let v_tok_bytes = kv_format_row_bytes(v_format, head_dim);
            let k_head_stride = block_size * k_tok_bytes;
            let v_head_stride = block_size * v_tok_bytes;
            let k_page_stride = local_kv_heads * k_head_stride;
            let v_page_stride = local_kv_heads * v_head_stride;

            let k_buf = DeviceBuffer::alloc(device_id, num_pages * k_page_stride)?;
            let v_buf = DeviceBuffer::alloc(device_id, num_pages * v_page_stride)?;
            let layout = W34KvLayerLayout {
                k_base: k_buf.addr(),
                v_base: v_buf.addr(),
                k_page_stride_bytes: k_page_stride as u64,
                k_head_stride_bytes: k_head_stride as u64,
                k_token_stride_bytes: k_tok_bytes as u64,
                v_page_stride_bytes: v_page_stride as u64,
                v_head_stride_bytes: v_head_stride as u64,
                v_token_stride_bytes: v_tok_bytes as u64,
                namespace_id,
                layer_id: layer_id as i32,
                local_kv_heads: local_kv_heads as i32,
                head_dim: head_dim as i32,
                block_size: block_size as i32,
                k_format,
                v_format,
                k_transform,
                v_transform,
                reserved: 0,
            };
            layers.push(KvLayerBuffer {
                k_buf,
                v_buf,
                layout,
            });
        }
        let free_pages = PageBitmap::new(num_pages);
        Ok(Self {
            device_id,
            num_pages,
            block_size,
            layers,
            free_pages,
        })
    }

    pub fn new_target_full_attn(
        device_id: i32,
        layer_ids: &[usize],
        local_kv_heads: usize,
        head_dim: usize,
        block_size: usize,
        num_pages: usize,
        policy: KvFormatPolicy,
    ) -> Result<Self> {
        Self::new(
            device_id,
            W34_KV_NAMESPACE_TARGET_FULL_ATTN,
            layer_ids,
            local_kv_heads,
            head_dim,
            block_size,
            num_pages,
            policy,
        )
    }

    pub fn free_count(&self) -> usize {
        self.free_pages.free_count()
    }

    pub fn alloc_page(&mut self) -> Option<i32> {
        self.free_pages.alloc()
    }

    pub fn free_page(&mut self, page_id: i32) {
        self.free_pages.free(page_id);
    }

    pub fn reset_all_pages(&mut self) {
        self.free_pages.reset();
    }

    /// Clamps logical page capacity downward so `TP=2` ranks operate in lockstep even when
    /// per-GPU free VRAM or rank-0 vision reservation produces slightly different initial budgets.
    pub fn truncate_pages(&mut self, max_pages: usize) {
        if max_pages < self.num_pages {
            self.num_pages = max_pages;
            self.free_pages = PageBitmap::new(max_pages);
        }
    }
}

/// Lifecycle states defined in `docs/contracts/state-transaction.md`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum StateTxPhase {
    Idle,
    Prefilling,
    Decoding,
    SpecDrafting,
    SpecVerifying,
    RolledBack,
}

/// Transactional GDN state handle with `extra_buffer` ping-pong slot indices.
#[derive(Clone, Debug)]
pub struct StateTx {
    pub phase: StateTxPhase,
    pub committed_slot: i32,
    pub spec_slots: Vec<i32>,
    pub has_initial_state: bool,
    pub kv_seq_len_before: usize,
}

impl StateTx {
    pub fn new(initial_slot: i32) -> Self {
        Self {
            phase: StateTxPhase::Idle,
            committed_slot: initial_slot,
            spec_slots: Vec::new(),
            has_initial_state: false,
            kv_seq_len_before: 0,
        }
    }

    pub fn begin_prefill(&mut self) {
        self.phase = StateTxPhase::Prefilling;
    }

    pub fn commit_prefill(&mut self, new_seq_len: usize) {
        self.has_initial_state = true;
        self.kv_seq_len_before = new_seq_len;
        self.phase = StateTxPhase::Decoding;
    }

    pub fn begin_spec_verify(&mut self, extra_slots: Vec<i32>) {
        assert!(!extra_slots.is_empty());
        self.spec_slots = extra_slots;
        self.phase = StateTxPhase::SpecVerifying;
    }

    /// Returns the `[1 + k]` slot indices (`[committed_slot, spec_slots[0], ...]`)
    /// consumed by `W34GdnBatch::spec_state_indices`.
    pub fn verify_slot_schedule(&self) -> Vec<i32> {
        let mut out = Vec::with_capacity(1 + self.spec_slots.len());
        out.push(self.committed_slot);
        out.extend_from_slice(&self.spec_slots);
        out
    }

    /// Commits or rolls back a speculative verification round of `1 + k` tokens where
    /// `accepted_len` in `1..=1 + k` tokens were accepted (`1` = bonus token only, all drafts rejected).
    /// Returns the freed slots to return to the CUDA `StateBundlePool`.
    pub fn resolve_spec_verify(&mut self, accepted_len: usize) -> Vec<i32> {
        assert!(accepted_len >= 1 && accepted_len <= 1 + self.spec_slots.len());
        let schedule = self.verify_slot_schedule();
        let winner_slot = schedule[accepted_len - 1];
        let mut freed = Vec::with_capacity(self.spec_slots.len());
        for &s in &schedule {
            if s != winner_slot {
                freed.push(s);
            }
        }
        self.committed_slot = winner_slot;
        self.spec_slots.clear();
        self.kv_seq_len_before += accepted_len;
        self.phase = if accepted_len == schedule.len() {
            StateTxPhase::Decoding
        } else {
            StateTxPhase::RolledBack
        };
        freed
    }
}

#[cfg(feature = "cuda")]
pub struct GdnLayerStateBuffers {
    pub layer_id: usize,
    /// `[total_slots, conv_dim, kernel_size - 1]` in `FP16`.
    pub conv_state: DeviceBuffer,
    /// `[total_slots, local_v_heads, head_v_dim, head_k_dim]` in `FP32`.
    pub ssm_state: DeviceBuffer,
}

#[cfg(feature = "cuda")]
pub struct StateBundlePool {
    pub device_id: i32,
    pub total_slots: usize,
    pub conv_dim: usize,
    pub conv_tap_count: usize,
    pub local_v_heads: usize,
    pub head_k_dim: usize,
    pub head_v_dim: usize,
    pub layers: Vec<GdnLayerStateBuffers>,
    free_slots: Vec<i32>,
}

#[cfg(feature = "cuda")]
impl StateBundlePool {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        device_id: i32,
        layer_ids: &[usize],
        total_slots: usize,
        conv_dim: usize,
        conv_kernel_size: usize,
        local_v_heads: usize,
        head_k_dim: usize,
        head_v_dim: usize,
    ) -> Result<Self> {
        let conv_tap_count = conv_kernel_size - 1;
        let conv_bytes_per_slot = conv_dim * conv_tap_count * 2; // FP16
        let ssm_bytes_per_slot = local_v_heads * head_v_dim * head_k_dim * 4; // FP32
        let mut layers = Vec::with_capacity(layer_ids.len());
        for &layer_id in layer_ids {
            let conv_state = DeviceBuffer::alloc(device_id, total_slots * conv_bytes_per_slot)?;
            let ssm_state = DeviceBuffer::alloc(device_id, total_slots * ssm_bytes_per_slot)?;
            layers.push(GdnLayerStateBuffers {
                layer_id,
                conv_state,
                ssm_state,
            });
        }
        let free_slots = (0..total_slots as i32).rev().collect();
        Ok(Self {
            device_id,
            total_slots,
            conv_dim,
            conv_tap_count,
            local_v_heads,
            head_k_dim,
            head_v_dim,
            layers,
            free_slots,
        })
    }

    pub fn free_count(&self) -> usize {
        self.free_slots.len()
    }

    pub fn alloc_slot(&mut self) -> Option<i32> {
        self.free_slots.pop()
    }

    pub fn free_slot(&mut self, slot: i32) {
        self.free_slots.push(slot);
    }

    pub fn free_slots(&mut self, slots: &[i32]) {
        for &s in slots {
            self.free_slots.push(s);
        }
    }

    pub fn reset_all_slots(&mut self) {
        self.free_slots = (0..self.total_slots as i32).rev().collect();
    }

    /// Clamps logical state-slot capacity downward so `TP=2` ranks operate in lockstep.
    pub fn truncate_slots(&mut self, max_slots: usize) {
        if max_slots < self.total_slots {
            self.total_slots = max_slots;
            self.reset_all_slots();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use w34_kernels::ffi::{
        W34_KV_FORMAT_BF16, W34_KV_FORMAT_FP16, W34_KV_FORMAT_Q8_0, W34_KV_FORMAT_TQ4,
        W34_KV_TRANSFORM_IDENTITY, W34_KV_TRANSFORM_RADEMACHER_WHT256,
    };

    #[test]
    fn test_kv_format_row_bytes_dimensions() {
        // Head dim 256 (standard target model)
        assert_eq!(kv_format_row_bytes(W34_KV_FORMAT_FP16, 256), 512);
        assert_eq!(kv_format_row_bytes(W34_KV_FORMAT_BF16, 256), 512);
        assert_eq!(kv_format_row_bytes(W34_KV_FORMAT_TQ4, 256), 132);
        assert_eq!(kv_format_row_bytes(W34_KV_FORMAT_Q8_0, 256), 272);

        // Head dim 128 (proposer draft context)
        assert_eq!(kv_format_row_bytes(W34_KV_FORMAT_FP16, 128), 256);
        assert_eq!(kv_format_row_bytes(W34_KV_FORMAT_BF16, 128), 256);
        assert_eq!(kv_format_row_bytes(W34_KV_FORMAT_TQ4, 128), 66);
        assert_eq!(kv_format_row_bytes(W34_KV_FORMAT_Q8_0, 128), 136);
    }

    #[test]
    fn test_kv_format_policy_resolve_layer_invariants() {
        let num_layers = 16;

        // Fp16 policy: all layers FP16 with Identity
        for layer in 0..num_layers {
            let (k_fmt, v_fmt, k_tf, v_tf) = KvFormatPolicy::Fp16.resolve_layer(layer, num_layers);
            assert_eq!(k_fmt, W34_KV_FORMAT_FP16);
            assert_eq!(v_fmt, W34_KV_FORMAT_FP16);
            assert_eq!(k_tf, W34_KV_TRANSFORM_IDENTITY);
            assert_eq!(v_tf, W34_KV_TRANSFORM_IDENTITY);
        }

        // Tq4All policy: all layers TQ4 with Rademacher-WHT256
        for layer in 0..num_layers {
            let (k_fmt, v_fmt, k_tf, v_tf) =
                KvFormatPolicy::Tq4All.resolve_layer(layer, num_layers);
            assert_eq!(k_fmt, W34_KV_FORMAT_TQ4);
            assert_eq!(v_fmt, W34_KV_FORMAT_TQ4);
            assert_eq!(k_tf, W34_KV_TRANSFORM_RADEMACHER_WHT256);
            assert_eq!(v_tf, W34_KV_TRANSFORM_RADEMACHER_WHT256);
        }

        // Tq4 boundary-anchored policy: first 2 and last 2 are Q8_0, interior are TQ4
        for layer in 0..num_layers {
            let (k_fmt, v_fmt, k_tf, v_tf) = KvFormatPolicy::Tq4.resolve_layer(layer, num_layers);
            assert_eq!(k_tf, W34_KV_TRANSFORM_RADEMACHER_WHT256);
            assert_eq!(v_tf, W34_KV_TRANSFORM_RADEMACHER_WHT256);

            let is_boundary = layer < 2 || layer >= num_layers - 2;
            if is_boundary {
                assert_eq!(k_fmt, W34_KV_FORMAT_Q8_0);
                assert_eq!(v_fmt, W34_KV_FORMAT_Q8_0);
            } else {
                assert_eq!(k_fmt, W34_KV_FORMAT_TQ4);
                assert_eq!(v_fmt, W34_KV_FORMAT_TQ4);
            }
        }
    }

    #[test]
    fn test_kv_format_policy_bytes_per_token_calculation() {
        let num_full_attn_layers = 16;
        let kv_heads = 8;
        let head_dim = 256;

        let fp16_bytes =
            KvFormatPolicy::Fp16.bytes_per_token(num_full_attn_layers, kv_heads, head_dim);
        assert_eq!(fp16_bytes, 16 * 8 * (512 + 512)); // 131,072 bytes/tok

        let tq4_all_bytes =
            KvFormatPolicy::Tq4All.bytes_per_token(num_full_attn_layers, kv_heads, head_dim);
        assert_eq!(tq4_all_bytes, 16 * 8 * (132 + 132)); // 33,792 bytes/tok (~3.88x)

        let tq4_boundary_bytes =
            KvFormatPolicy::Tq4.bytes_per_token(num_full_attn_layers, kv_heads, head_dim);
        // 4 boundary layers * 8 * (272 + 272) + 12 interior layers * 8 * (132 + 132)
        let expected_tq4 = 4 * 8 * 544 + 12 * 8 * 264;
        assert_eq!(tq4_boundary_bytes, expected_tq4); // 42,752 bytes/tok (~3.07x)
    }

    #[test]
    fn test_state_tx_lifecycle_and_rollback() {
        let mut tx = StateTx::new(42);
        assert_eq!(tx.phase, StateTxPhase::Idle);
        assert_eq!(tx.committed_slot, 42);
        assert!(!tx.has_initial_state);

        tx.begin_prefill();
        assert_eq!(tx.phase, StateTxPhase::Prefilling);

        tx.commit_prefill(64);
        assert_eq!(tx.phase, StateTxPhase::Decoding);
        assert!(tx.has_initial_state);
        assert_eq!(tx.kv_seq_len_before, 64);

        // Speculative verify round with 3 draft proposals (total 4 tokens: anchor + 3 drafts)
        tx.begin_spec_verify(vec![101, 102, 103]);
        assert_eq!(tx.phase, StateTxPhase::SpecVerifying);
        assert_eq!(tx.verify_slot_schedule(), vec![42, 101, 102, 103]);

        // Scenario A: Partial accept of 2 tokens (accepted_len = 2 -> winner is slot 101)
        let freed = tx.resolve_spec_verify(2);
        assert_eq!(tx.committed_slot, 101);
        assert_eq!(freed, vec![42, 102, 103]);
        assert_eq!(tx.kv_seq_len_before, 66);
        assert_eq!(tx.phase, StateTxPhase::RolledBack);

        // Scenario B: Full accept
        tx.begin_spec_verify(vec![201, 202]);
        let freed_full = tx.resolve_spec_verify(3); // accepted all 3 (winner is 202)
        assert_eq!(tx.committed_slot, 202);
        assert_eq!(freed_full, vec![101, 201]);
        assert_eq!(tx.kv_seq_len_before, 69);
        assert_eq!(tx.phase, StateTxPhase::Decoding);
    }

    #[test]
    fn test_page_bitmap_ascending_coalescing_and_idempotent_free() {
        let mut bm = PageBitmap::new(130);
        assert_eq!(bm.free_count(), 130);
        for expected in 0..130 {
            assert_eq!(bm.alloc(), Some(expected));
        }
        assert_eq!(bm.alloc(), None);
        assert_eq!(bm.free_count(), 0);

        // Scattered + reverse leaf frees must coalesce into strictly ascending allocations.
        for &p in &[129, 64, 65, 3, 2, 1, 0, 128, 63] {
            bm.free(p);
        }
        // Duplicate and out-of-range frees must be ignored safely.
        bm.free(64);
        bm.free(-1);
        bm.free(130);
        assert_eq!(bm.free_count(), 9);

        let reallocated: Vec<i32> = (0..9).map(|_| bm.alloc().unwrap()).collect();
        assert_eq!(reallocated, vec![0, 1, 2, 3, 63, 64, 65, 128, 129]);
        assert_eq!(bm.alloc(), None);

        bm.reset();
        assert_eq!(bm.free_count(), 130);
        assert_eq!(bm.alloc(), Some(0));
    }
}
