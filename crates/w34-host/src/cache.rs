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

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum KvFormatPolicy {
    Fp16,
    Bf16,
    Tq4,
    K8V4,
    /// Layers `0, 1` and `num_full_attn_layers - 2, num_full_attn_layers - 1` use `FP16`;
    /// interior layers use `TQ4`.
    FirstLast2,
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
            Self::Tq4 => (
                W34_KV_FORMAT_TQ4,
                W34_KV_FORMAT_TQ4,
                W34_KV_TRANSFORM_RADEMACHER_WHT256,
                W34_KV_TRANSFORM_RADEMACHER_WHT256,
            ),
            Self::K8V4 => (
                W34_KV_FORMAT_Q8_0,
                W34_KV_FORMAT_TQ4,
                W34_KV_TRANSFORM_RADEMACHER_WHT256,
                W34_KV_TRANSFORM_RADEMACHER_WHT256,
            ),
            Self::FirstLast2 => {
                let is_boundary = full_attn_layer_ordinal < 2
                    || (num_full_attn_layers >= 2
                        && full_attn_layer_ordinal >= num_full_attn_layers - 2);
                if is_boundary {
                    (
                        W34_KV_FORMAT_FP16,
                        W34_KV_FORMAT_FP16,
                        W34_KV_TRANSFORM_IDENTITY,
                        W34_KV_TRANSFORM_IDENTITY,
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
        }
    }
}

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

#[cfg(feature = "cuda")]
pub struct KvPool {
    pub device_id: i32,
    pub num_pages: usize,
    pub block_size: usize,
    pub layers: Vec<KvLayerBuffer>,
    free_pages: Vec<i32>,
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
        let free_pages = (0..num_pages as i32).rev().collect();
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

    pub fn alloc_page(&mut self) -> Option<i32> {
        self.free_pages.pop()
    }

    pub fn free_page(&mut self, page_id: i32) {
        self.free_pages.push(page_id);
    }

    pub fn reset_all_pages(&mut self) {
        self.free_pages = (0..self.num_pages as i32).rev().collect();
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
    /// Returns the freed slots to return to [`StateBundlePool`].
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
}
