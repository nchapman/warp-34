//! Tiered routed-expert storage and execution (pinned host RAM + VRAM hot cache + EP).

use std::io;
use std::ops::Range;

/// One expert's DMA layout. All four regions and successive packs are aligned.
/// We keep N-major U4B8 weights and unpermuted FP16 scales, matching the MoE kernel.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ExpertPackLayout {
    pub intermediate: usize,
    pub hidden: usize,
    pub group_size: usize,
    pub gate_up_packed: Range<usize>,
    pub gate_up_scales: Range<usize>,
    pub down_packed: Range<usize>,
    pub down_scales: Range<usize>,
    pub stride: usize,
}

impl ExpertPackLayout {
    pub fn new(intermediate: usize, hidden: usize, group_size: usize) -> io::Result<Self> {
        let invalid =
            || io::Error::new(io::ErrorKind::InvalidInput, "invalid expert pack geometry");
        if intermediate == 0
            || hidden == 0
            || group_size == 0
            || !intermediate.is_multiple_of(8)
            || !hidden.is_multiple_of(8)
            || !intermediate.is_multiple_of(group_size)
            || !hidden.is_multiple_of(group_size)
            || intermediate > i32::MAX as usize / 2
            || hidden > i32::MAX as usize
            || group_size > i32::MAX as usize
        {
            return Err(invalid());
        }
        let matrix = intermediate.checked_mul(hidden).ok_or_else(invalid)?;
        let sizes = [
            matrix,
            matrix
                .checked_div(group_size)
                .and_then(|v| v.checked_mul(4))
                .ok_or_else(invalid)?,
            matrix / 2,
            matrix
                .checked_div(group_size)
                .and_then(|v| v.checked_mul(2))
                .ok_or_else(invalid)?,
        ];
        let mut end = 0usize;
        let mut regions = Vec::with_capacity(4);
        for bytes in sizes {
            if bytes == 0 || !bytes.is_multiple_of(16) {
                return Err(invalid());
            }
            let next = end.checked_add(bytes).ok_or_else(invalid)?;
            regions.push(end..next);
            end = next;
        }
        let stride = end.checked_add(4095).ok_or_else(invalid)? & !4095;
        Ok(Self {
            intermediate,
            hidden,
            group_size,
            gate_up_packed: regions[0].clone(),
            gate_up_scales: regions[1].clone(),
            down_packed: regions[2].clone(),
            down_scales: regions[3].clone(),
            stride,
        })
    }

    /// Packs a single expert's N-major `U4B8` weights and `FP16` scales into a `stride`-byte slot.
    pub fn write_pack(
        &self,
        gate_up_packed: &[u32],
        gate_up_scales: &[u16],
        down_packed: &[u32],
        down_scales: &[u16],
        dst_pack: &mut [u8],
    ) -> io::Result<()> {
        if dst_pack.len() != self.stride
            || gate_up_packed.len() * 4 != self.gate_up_packed.len()
            || gate_up_scales.len() * 2 != self.gate_up_scales.len()
            || down_packed.len() * 4 != self.down_packed.len()
            || down_scales.len() * 2 != self.down_scales.len()
        {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "invalid expert slice lengths for write_pack",
            ));
        }
        dst_pack.fill(0);
        for (dst, &w) in dst_pack[self.gate_up_packed.clone()]
            .as_chunks_mut::<4>()
            .0
            .iter_mut()
            .zip(gate_up_packed)
        {
            dst.copy_from_slice(&w.to_le_bytes());
        }
        for (dst, &s) in dst_pack[self.gate_up_scales.clone()]
            .as_chunks_mut::<2>()
            .0
            .iter_mut()
            .zip(gate_up_scales)
        {
            dst.copy_from_slice(&s.to_le_bytes());
        }
        for (dst, &w) in dst_pack[self.down_packed.clone()]
            .as_chunks_mut::<4>()
            .0
            .iter_mut()
            .zip(down_packed)
        {
            dst.copy_from_slice(&w.to_le_bytes());
        }
        for (dst, &s) in dst_pack[self.down_scales.clone()]
            .as_chunks_mut::<2>()
            .0
            .iter_mut()
            .zip(down_scales)
        {
            dst.copy_from_slice(&s.to_le_bytes());
        }
        Ok(())
    }

    /// Evaluates one packed `U4B8` expert on CPU matching `op_moe.cu` (`marlin_silu_and_mul_f16`
    /// and FP16-rounded `U4B8` dequantization) and accumulates into `out_accum_f32: [hidden]`.
    pub fn eval_expert_u4b8_cpu(
        &self,
        pack: &[u8],
        x_f16: &[u16],
        topk_weight: f32,
        out_accum_f32: &mut [f32],
    ) {
        use crate::cuda::{f16_to_f32, f32_to_f16};
        assert_eq!(pack.len(), self.stride);
        assert_eq!(x_f16.len(), self.hidden);
        assert_eq!(out_accum_f32.len(), self.hidden);

        let gu_packed_bytes = &pack[self.gate_up_packed.clone()];
        let gu_scales_bytes = &pack[self.gate_up_scales.clone()];
        let dn_packed_bytes = &pack[self.down_packed.clone()];
        let dn_scales_bytes = &pack[self.down_scales.clone()];

        let words_per_k = self.hidden / 8;
        let groups_per_k = self.hidden / self.group_size;
        let words_per_group_k = self.group_size / 8;

        // 1. Gate + Up projection + donor vLLM SiluAndMul (FP16 intermediate)
        let mut interm_f16 = vec![0u16; self.intermediate];
        for (n, dst_interm) in interm_f16.iter_mut().enumerate() {
            let mut gate_acc = 0.0f32;
            let mut up_acc = 0.0f32;
            let gate_row_w = n * words_per_k;
            let up_row_w = (self.intermediate + n) * words_per_k;
            let gate_row_s = n * groups_per_k;
            let up_row_s = (self.intermediate + n) * groups_per_k;

            for g in 0..groups_per_k {
                let gs_off = (gate_row_s + g) * 2;
                let us_off = (up_row_s + g) * 2;
                let gs = f16_to_f32(u16::from_le_bytes([
                    gu_scales_bytes[gs_off],
                    gu_scales_bytes[gs_off + 1],
                ]));
                let us = f16_to_f32(u16::from_le_bytes([
                    gu_scales_bytes[us_off],
                    gu_scales_bytes[us_off + 1],
                ]));
                for w in 0..words_per_group_k {
                    let w_idx = g * words_per_group_k + w;
                    let gw_off = (gate_row_w + w_idx) * 4;
                    let uw_off = (up_row_w + w_idx) * 4;
                    let gw = u32::from_le_bytes([
                        gu_packed_bytes[gw_off],
                        gu_packed_bytes[gw_off + 1],
                        gu_packed_bytes[gw_off + 2],
                        gu_packed_bytes[gw_off + 3],
                    ]);
                    let uw = u32::from_le_bytes([
                        gu_packed_bytes[uw_off],
                        gu_packed_bytes[uw_off + 1],
                        gu_packed_bytes[uw_off + 2],
                        gu_packed_bytes[uw_off + 3],
                    ]);
                    let k_base = w_idx * 8;
                    for b in 0..8 {
                        let x_val = f16_to_f32(x_f16[k_base + b]);
                        let gnib = ((gw >> (b * 4)) & 0xF) as i32 - 8;
                        let unib = ((uw >> (b * 4)) & 0xF) as i32 - 8;
                        let gw_h = f16_to_f32(f32_to_f16((gnib as f32) * gs));
                        let uw_h = f16_to_f32(f32_to_f16((unib as f32) * us));
                        gate_acc += gw_h * x_val;
                        up_acc += uw_h * x_val;
                    }
                }
            }
            let gate_h = f16_to_f32(f32_to_f16(gate_acc));
            let up_h = f16_to_f32(f32_to_f16(up_acc));
            let silu_h = f16_to_f32(f32_to_f16(gate_h / (1.0f32 + (-gate_h).exp())));
            *dst_interm = f32_to_f16(silu_h * up_h);
        }

        // 2. Down projection + Top-K weight scaling
        let words_per_interm = self.intermediate / 8;
        let groups_per_interm = self.intermediate / self.group_size;
        let tw_h = f16_to_f32(f32_to_f16(topk_weight));
        for (h, dst_out) in out_accum_f32.iter_mut().enumerate() {
            let mut down_acc = 0.0f32;
            let row_w = h * words_per_interm;
            let row_s = h * groups_per_interm;
            for g in 0..groups_per_interm {
                let ds_off = (row_s + g) * 2;
                let ds = f16_to_f32(u16::from_le_bytes([
                    dn_scales_bytes[ds_off],
                    dn_scales_bytes[ds_off + 1],
                ]));
                for w in 0..words_per_group_k {
                    let w_idx = g * words_per_group_k + w;
                    let dw_off = (row_w + w_idx) * 4;
                    let dw = u32::from_le_bytes([
                        dn_packed_bytes[dw_off],
                        dn_packed_bytes[dw_off + 1],
                        dn_packed_bytes[dw_off + 2],
                        dn_packed_bytes[dw_off + 3],
                    ]);
                    let k_base = w_idx * 8;
                    for b in 0..8 {
                        let x_val = f16_to_f32(interm_f16[k_base + b]);
                        let dnib = ((dw >> (b * 4)) & 0xF) as i32 - 8;
                        let dw_h = f16_to_f32(f32_to_f16((dnib as f32) * ds));
                        down_acc += dw_h * x_val;
                    }
                }
            }
            let prod_h = f16_to_f32(f32_to_f16(down_acc)) * tw_h;
            *dst_out += f16_to_f32(f32_to_f16(prod_h));
        }
    }
}

/// Optional static expert frequency rankings (`"<layer_id>": [<global_expert_id>, ...]`).
///
/// Discovered automatically from `static_hot_cache_rankings.json` or
/// `runtime/static_hot_cache_rankings.json` inside a checkpoint directory when present.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct HotExpertRankings {
    by_layer: std::collections::HashMap<usize, Vec<usize>>,
}

impl HotExpertRankings {
    pub fn from_json(val: &serde_json::Value) -> Option<Self> {
        let obj = val.as_object()?;
        let mut by_layer = std::collections::HashMap::with_capacity(obj.len());
        for (k, v) in obj {
            let Ok(layer_id) = k.parse::<usize>() else {
                continue;
            };
            let Some(arr) = v.as_array() else {
                continue;
            };
            let ids: Vec<usize> = arr
                .iter()
                .filter_map(|x| x.as_u64().and_then(|n| usize::try_from(n).ok()))
                .collect();
            if !ids.is_empty() {
                by_layer.insert(layer_id, ids);
            }
        }
        (!by_layer.is_empty()).then_some(Self { by_layer })
    }

    /// Probes standard checkpoint locations (`static_hot_cache_rankings.json` and
    /// `runtime/static_hot_cache_rankings.json`) inside `model_dir`.
    pub fn from_model_dir(model_dir: &std::path::Path) -> Option<Self> {
        for rel in [
            "static_hot_cache_rankings.json",
            "runtime/static_hot_cache_rankings.json",
        ] {
            let path = model_dir.join(rel);
            if let Ok(bytes) = std::fs::read(&path)
                && let Ok(val) = serde_json::from_slice::<serde_json::Value>(&bytes)
                && let Some(rankings) = Self::from_json(&val)
            {
                return Some(rankings);
            }
        }
        None
    }

    /// Selects `k_hot` unique local expert IDs within `local_expert_ids` for `layer_id`
    /// in ranked frequency order, backfilling with any remaining local expert IDs if needed.
    pub fn select_for_layer(
        &self,
        layer_id: usize,
        local_expert_ids: Range<usize>,
        k_hot: usize,
    ) -> Option<Vec<usize>> {
        if k_hot == 0 || k_hot > local_expert_ids.len() {
            return None;
        }
        let ranked = self.by_layer.get(&layer_id)?;
        let mut chosen = Vec::with_capacity(k_hot);
        for &gid in ranked {
            if local_expert_ids.contains(&gid) && !chosen.contains(&gid) {
                chosen.push(gid);
                if chosen.len() == k_hot {
                    return Some(chosen);
                }
            }
        }
        for gid in local_expert_ids {
            if !chosen.contains(&gid) {
                chosen.push(gid);
                if chosen.len() == k_hot {
                    return Some(chosen);
                }
            }
        }
        None
    }
}

/// Maximum token rows (`6,144`) supported in a single tiered-MoE / `PLE` batch workspace.
pub const TIERED_MOE_MAX_BATCH_TOKENS: usize = 6144;

/// Maximum decode/verify batch size (`4` tokens) eligible for zero-copy GPU+CPU hybrid MoE execution.
pub const TIERED_MOE_CPU_BRIDGE_MAX_DECODE_TOKENS: usize = 4;

/// Maximum routed `top_k` (`16` experts/token) supported by the GPU+CPU hybrid MoE mailbox ABI.
pub const TIERED_MOE_CPU_BRIDGE_MAX_TOP_K: usize = 16;

/// Router weight threshold (`0.14`) below which cold experts are delegated to the CPU bridge
/// when dynamic LRU slots (`k_static < k_hot`) are available on the GPU.
pub const TIERED_MOE_CPU_BRIDGE_COLD_WEIGHT_THRESHOLD: f32 = 0.14;

/// Minimum GPU hot expert count (`16`) required before auto-enabling the `TP=1` CPU MoE bridge.
pub const TIERED_MOE_CPU_BRIDGE_MIN_HOT_EXPERTS: usize = 16;

/// Maximum model layer count (`60`) supported by the static `TP=1` CPU MoE bridge registration.
pub const TIERED_MOE_CPU_BRIDGE_MAX_LAYERS: usize = 60;

/// Quantization group size (`128`) required by the `U4B8` CPU MoE evaluator.
pub const TIERED_MOE_CPU_BRIDGE_GROUP_SIZE: usize = 128;

/// Execution policy for tiered 512-expert MoE layers.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MoeOffloadMode {
    /// Automatically selects `GpuLru` when unique active local experts fit in `K_hot`,
    /// or `StreamedPrefill` when active local experts exceed `K_hot`.
    Auto,
    /// Dynamic GPU LRU slot replacement via single-DMA H2D transfer + device unpack.
    /// Automatically falls back to `StreamedPrefill` if a batch activates `> K_hot` unique local experts.
    GpuLru,
    /// Evaluates hot-cache hits (plus up to `gpu_pcie_budget` cold experts via DMA) on the GPU
    /// while co-executing remaining cold experts in parallel on the CPU thread pool.
    CpuHybrid { gpu_pcie_budget: usize },
    /// Executes hot experts in Pass 0 and streams cold experts through double-buffered
    /// H2D + compute staging arenas (`stage_buf[0..2]`).
    StreamedPrefill,
}

#[cfg(feature = "cuda")]
mod engine;
#[cfg(feature = "cuda")]
mod host_store;

#[cfg(feature = "cuda")]
pub use engine::{RankTieredMoe, TieredMoeEngine};
#[cfg(feature = "cuda")]
pub use host_store::{DeviceExpertSlotTable, HostExpertStore, HotExpertCache, TieredMoeWorkspace};

#[cfg(test)]
mod tests;
