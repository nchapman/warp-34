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
mod host_store {
    use super::*;
    use crate::cuda::MappedHostBuffer;
    use crate::loader::{QuantScheme, SafetensorsArchive};
    use std::io::Result;

    fn require(valid: bool, message: impl Into<String>) -> Result<()> {
        if valid {
            Ok(())
        } else {
            Err(io::Error::new(io::ErrorKind::InvalidData, message.into()))
        }
    }

    /// Immutable pinned backing for one layer's locally owned, complete experts.
    /// EP partitions expert IDs, never the intermediate dimension. Retain this
    /// store until every stream reading its mapped address or DMA source completes.
    static TOTAL_PINNED_BYTES: std::sync::atomic::AtomicUsize =
        std::sync::atomic::AtomicUsize::new(0);

    enum HostPackStorage {
        Mapped(MappedHostBuffer),
        Heap(Vec<u8>),
    }

    impl Drop for HostPackStorage {
        fn drop(&mut self) {
            if let Self::Mapped(m) = self {
                TOTAL_PINNED_BYTES.fetch_sub(m.len_bytes, std::sync::atomic::Ordering::Relaxed);
            }
        }
    }

    impl HostPackStorage {
        fn alloc(bytes: usize, prefer_mapped: bool) -> Result<Self> {
            const MAX_PROCESS_PINNED_BYTES: usize = 64 * 1024 * 1024 * 1024;
            let cur = TOTAL_PINNED_BYTES.load(std::sync::atomic::Ordering::Relaxed);
            if prefer_mapped
                && cur.saturating_add(bytes) <= MAX_PROCESS_PINNED_BYTES
                && let Ok(m) = MappedHostBuffer::alloc(bytes)
            {
                TOTAL_PINNED_BYTES.fetch_add(bytes, std::sync::atomic::Ordering::Relaxed);
                return Ok(Self::Mapped(m));
            }
            Ok(Self::Heap(vec![0u8; bytes]))
        }

        fn as_mut_slice(&mut self) -> &mut [u8] {
            match self {
                // SAFETY: `m.host_ptr` points to a valid `m.len_bytes`-byte allocation
                // owned exclusively by `self` while `&mut self` is borrowed.
                Self::Mapped(m) => unsafe {
                    std::slice::from_raw_parts_mut(m.host_ptr.cast::<u8>(), m.len_bytes)
                },
                Self::Heap(v) => v.as_mut_slice(),
            }
        }

        fn as_slice(&self) -> &[u8] {
            match self {
                // SAFETY: `m.host_ptr` points to a valid `m.len_bytes`-byte allocation
                // alive for the lifetime of `&self`.
                Self::Mapped(m) => unsafe {
                    std::slice::from_raw_parts(m.host_ptr.cast::<u8>(), m.len_bytes)
                },
                Self::Heap(v) => v.as_slice(),
            }
        }

        fn len_bytes(&self) -> usize {
            match self {
                Self::Mapped(m) => m.len_bytes,
                Self::Heap(v) => v.len(),
            }
        }

        fn mapped_dev_ptr(&self) -> u64 {
            match self {
                Self::Mapped(m) => m.dev_ptr as u64,
                Self::Heap(_) => 0,
            }
        }
    }

    pub struct HostExpertStore {
        storage: HostPackStorage,
        layout: ExpertPackLayout,
        expert_ids: Range<usize>,
    }

    impl HostExpertStore {
        /// Load only this EP rank's experts, bounded by an explicit pinned-RAM budget.
        #[allow(clippy::too_many_arguments)]
        pub fn load(
            archive: &SafetensorsArchive,
            experts_prefix: &str,
            quant: &QuantScheme,
            layout: ExpertPackLayout,
            num_experts: usize,
            ep_rank: usize,
            ep_size: usize,
            max_pinned_bytes: usize,
        ) -> Result<Self> {
            require(
                matches!(ep_size, 1 | 2)
                    && ep_rank < ep_size
                    && num_experts > 0
                    && num_experts.is_multiple_of(ep_size),
                "invalid expert parallel partition",
            )?;
            require(
                quant.symmetric_gptq_u4_group_size() == Some(layout.group_size),
                "host expert store requires symmetric 4-bit GPTQ/AutoRound W4A16",
            )?;
            // Revalidate public geometry before using it to size or address unsafe storage.
            require(
                layout
                    == ExpertPackLayout::new(
                        layout.intermediate,
                        layout.hidden,
                        layout.group_size,
                    )?,
                "inconsistent expert pack layout",
            )?;
            let count = num_experts / ep_size;
            let bytes = layout
                .stride
                .checked_mul(count)
                .ok_or_else(|| io::Error::other("expert store size overflow"))?;
            require(
                bytes <= max_pinned_bytes,
                format!("expert store needs {bytes} pinned bytes, budget is {max_pinned_bytes}"),
            )?;
            let expert_ids = ep_rank * count..(ep_rank + 1) * count;
            let mut storage = HostPackStorage::alloc(bytes, true)?;
            let dst = storage.as_mut_slice();
            dst.fill(0);

            // Validate qzeros and g_idx on the first local expert (plus dimensions on all experts in parallel).
            if let Some(first_expert_id) = expert_ids.clone().next() {
                let prefix = format!("{experts_prefix}.{first_expert_id}");
                for projection in ["gate_proj", "up_proj", "down_proj"] {
                    let name = format!("{prefix}.{projection}.qzeros");
                    if let Some(name) = archive.resolve_name(&name) {
                        require(
                            archive.read_u32(&name)?.iter().all(|&v| v == 0x7777_7777),
                            format!("{name}: expected symmetric U4B8 zero points"),
                        )?;
                    }
                    let name = format!("{prefix}.{projection}.g_idx");
                    if let Some(name) = archive.resolve_name(&name) {
                        let k = if projection == "down_proj" {
                            layout.intermediate
                        } else {
                            layout.hidden
                        };
                        crate::loader::validate_static_gptq_g_idx(
                            &archive.read_u32(&name)?,
                            k,
                            layout.group_size,
                            &name,
                        )?;
                    }
                }
            }

            let num_threads = std::thread::available_parallelism()
                .map(|n| n.get())
                .unwrap_or(8)
                .clamp(1, 16)
                .min(count.max(1));
            let experts_per_chunk = count.div_ceil(num_threads).max(1);
            let stride = layout.stride;
            let base_expert = expert_ids.start;
            let layout_ref = &layout;
            let gu_half_packed = layout.gate_up_packed.len() / 2;
            let gu_half_scales = layout.gate_up_scales.len() / 2;

            std::thread::scope(|s| -> Result<()> {
                let mut handles = Vec::new();
                for (chunk_idx, dst_chunk) in dst.chunks_mut(experts_per_chunk * stride).enumerate()
                {
                    let local_start = chunk_idx * experts_per_chunk;
                    let chunk_count = dst_chunk.len() / stride;
                    handles.push(s.spawn(move || -> Result<()> {
                        let mut raw_qw_bytes = vec![0u8; gu_half_packed.max(layout_ref.down_packed.len())];
                        for i in 0..chunk_count {
                            let local_id = local_start + i;
                            let expert_id = base_expert + local_id;
                            let prefix = format!("{experts_prefix}.{expert_id}");
                            let pack = &mut dst_chunk[i * stride..(i + 1) * stride];
                            for (proj_name, n_exp, k_exp, w_range, s_range) in [
                                (
                                    "gate_proj",
                                    layout_ref.intermediate,
                                    layout_ref.hidden,
                                    layout_ref.gate_up_packed.start
                                        ..layout_ref.gate_up_packed.start + gu_half_packed,
                                    layout_ref.gate_up_scales.start
                                        ..layout_ref.gate_up_scales.start + gu_half_scales,
                                ),
                                (
                                    "up_proj",
                                    layout_ref.intermediate,
                                    layout_ref.hidden,
                                    layout_ref.gate_up_packed.start + gu_half_packed
                                        ..layout_ref.gate_up_packed.end,
                                    layout_ref.gate_up_scales.start + gu_half_scales
                                        ..layout_ref.gate_up_scales.end,
                                ),
                                (
                                    "down_proj",
                                    layout_ref.hidden,
                                    layout_ref.intermediate,
                                    layout_ref.down_packed.clone(),
                                    layout_ref.down_scales.clone(),
                                ),
                            ] {
                                let qw_key = archive
                                    .resolve_name(&format!("{prefix}.{proj_name}.qweight"))
                                    .ok_or_else(|| {
                                        io::Error::other(format!(
                                            "missing {prefix}.{proj_name}.qweight"
                                        ))
                                    })?;
                                let sc_key = archive
                                    .resolve_name(&format!("{prefix}.{proj_name}.scales"))
                                    .ok_or_else(|| {
                                        io::Error::other(format!(
                                            "missing {prefix}.{proj_name}.scales"
                                        ))
                                    })?;
                                let k8 = k_exp / 8;
                                let num_groups = k_exp / layout_ref.group_size;
                                let qw_info = archive.tensor_info(&qw_key)?;
                                let sc_info = archive.tensor_info(&sc_key)?;
                                require(
                                    qw_info.shape == [k8, n_exp]
                                        && sc_info.shape == [num_groups, n_exp],
                                    format!(
                                        "CUDA error -10 in model loader error: mismatched GPTQ qweight {:?} / scales {:?} (4-bit) on '{prefix}.{proj_name}'",
                                        qw_info.shape, sc_info.shape
                                    ),
                                )?;
                                let qw_buf = &mut raw_qw_bytes[..w_range.len()];
                                archive.read_exact_into(&qw_key, qw_buf)?;
                                let dst_w = &mut pack[w_range];
                                // Transpose `[k8, n_exp]` u32 words -> `[n_exp, k8]` u32 words directly into `dst_w`
                                for col in 0..k8 {
                                    let src_row = &qw_buf[col * n_exp * 4..(col + 1) * n_exp * 4];
                                    for row in 0..n_exp {
                                        let src_off = row * 4;
                                        let dst_off = (row * k8 + col) * 4;
                                        dst_w[dst_off..dst_off + 4]
                                            .copy_from_slice(&src_row[src_off..src_off + 4]);
                                    }
                                }
                                let sc_f16 = archive.read_f16_bits(&sc_key)?;
                                let dst_s = &mut pack[s_range];
                                for g in 0..num_groups {
                                    for row in 0..n_exp {
                                        let dst_off = (row * num_groups + g) * 2;
                                        dst_s[dst_off..dst_off + 2].copy_from_slice(
                                            &sc_f16[g * n_exp + row].to_le_bytes(),
                                        );
                                    }
                                }
                            }
                        }
                        Ok(())
                    }));
                }
                for h in handles {
                    h.join().map_err(|_| {
                        io::Error::other("HostExpertStore loader thread panicked")
                    })??;
                }
                Ok(())
            })?;
            Ok(Self {
                storage,
                layout,
                expert_ids,
            })
        }

        /// Constructs a `HostExpertStore` directly from pre-packed contiguous expert slots.
        pub fn from_raw_packs(
            layout: ExpertPackLayout,
            expert_ids: Range<usize>,
            raw_packs: &[u8],
            max_pinned_bytes: usize,
        ) -> Result<Self> {
            require(
                layout
                    == ExpertPackLayout::new(
                        layout.intermediate,
                        layout.hidden,
                        layout.group_size,
                    )?,
                "inconsistent expert pack layout",
            )?;
            let count = expert_ids.len();
            let bytes = layout
                .stride
                .checked_mul(count)
                .ok_or_else(|| io::Error::other("expert store size overflow"))?;
            require(
                count > 0 && raw_packs.len() == bytes,
                "invalid raw_packs length for HostExpertStore::from_raw_packs",
            )?;
            require(
                bytes <= max_pinned_bytes,
                format!("expert store needs {bytes} pinned bytes, budget is {max_pinned_bytes}"),
            )?;
            let mut storage = HostPackStorage::alloc(bytes, true)?;
            storage.as_mut_slice().copy_from_slice(raw_packs);
            Ok(Self {
                storage,
                layout,
                expert_ids,
            })
        }

        pub fn layout(&self) -> &ExpertPackLayout {
            &self.layout
        }
        pub fn expert_ids(&self) -> Range<usize> {
            self.expert_ids.clone()
        }
        pub fn pinned_bytes(&self) -> usize {
            self.storage.len_bytes()
        }
        pub fn mapped_dev_ptr(&self) -> u64 {
            self.storage.mapped_dev_ptr()
        }

        /// Read-only bytes, also suitable as a source for a synchronized H2D copy.
        pub fn expert_pack(&self, global_id: usize) -> Option<&[u8]> {
            if !self.expert_ids.contains(&global_id) {
                return None;
            }
            let offset = (global_id - self.expert_ids.start) * self.layout.stride;
            let slice = self.storage.as_slice();
            Some(&slice[offset..offset + self.layout.stride])
        }

        /// Evaluates a single locally owned expert on CPU and accumulates into `out_accum_f32`.
        pub fn eval_expert_u4b8_cpu(
            &self,
            global_id: usize,
            x_f16: &[u16],
            topk_weight: f32,
            out_accum_f32: &mut [f32],
        ) -> bool {
            let Some(pack) = self.expert_pack(global_id) else {
                return false;
            };
            self.layout
                .eval_expert_u4b8_cpu(pack, x_f16, topk_weight, out_accum_f32);
            true
        }

        /// Evaluates a batch of cold `(token_idx, global_expert_id, topk_weight)` tasks in parallel
        /// across CPU threads and returns `FP16` per-token partial sums `[num_tokens, hidden]`.
        pub fn eval_cold_tasks_cpu(
            &self,
            x_f16: &[u16],
            num_tokens: usize,
            tasks: &[(usize, usize, f32)],
        ) -> Vec<u16> {
            use crate::cuda::f32_to_f16;
            let h = self.layout.hidden;
            assert_eq!(x_f16.len(), num_tokens * h);
            if tasks.is_empty() || num_tokens == 0 {
                return vec![0u16; num_tokens * h];
            }

            let per_task_out: Vec<Vec<f32>> = if tasks.len() == 1 {
                let (tok, ge, weight) = tasks[0];
                let mut acc = vec![0.0f32; h];
                self.eval_expert_u4b8_cpu(ge, &x_f16[tok * h..(tok + 1) * h], weight, &mut acc);
                vec![acc]
            } else {
                let max_workers = std::thread::available_parallelism()
                    .map(|n| n.get().min(16))
                    .unwrap_or(4)
                    .min(tasks.len());
                let chunk_len = tasks.len().div_ceil(max_workers);
                let mut out_vecs = vec![Vec::new(); tasks.len()];
                std::thread::scope(|s| {
                    let mut handles = Vec::new();
                    for (chunk_idx, task_chunk) in tasks.chunks(chunk_len).enumerate() {
                        handles.push((
                            chunk_idx,
                            s.spawn(move || {
                                let mut chunk_res = Vec::with_capacity(task_chunk.len());
                                for &(tok, ge, weight) in task_chunk {
                                    let mut acc = vec![0.0f32; h];
                                    self.eval_expert_u4b8_cpu(
                                        ge,
                                        &x_f16[tok * h..(tok + 1) * h],
                                        weight,
                                        &mut acc,
                                    );
                                    chunk_res.push(acc);
                                }
                                chunk_res
                            }),
                        ));
                    }
                    for (chunk_idx, handle) in handles {
                        let chunk_res = handle.join().expect("CPU cold expert worker panicked");
                        let base = chunk_idx * chunk_len;
                        for (i, acc) in chunk_res.into_iter().enumerate() {
                            out_vecs[base + i] = acc;
                        }
                    }
                });
                out_vecs
            };

            let mut tok_accum = vec![0.0f32; num_tokens * h];
            for (&(tok, _, _), task_vec) in tasks.iter().zip(&per_task_out) {
                let dst = &mut tok_accum[tok * h..(tok + 1) * h];
                for (d, &v) in dst.iter_mut().zip(task_vec) {
                    *d += v;
                }
            }
            tok_accum.into_iter().map(f32_to_f16).collect()
        }
    }

    /// Device-resident table of `capacity` expert slots in VRAM (`U4B8` N-major layout).
    pub struct DeviceExpertSlotTable {
        pub device_id: i32,
        pub capacity: usize,
        pub layout: ExpertPackLayout,
        pub gate_up_packed: crate::cuda::DeviceBuffer,
        pub gate_up_scales: crate::cuda::DeviceBuffer,
        pub down_packed: crate::cuda::DeviceBuffer,
        pub down_scales: crate::cuda::DeviceBuffer,
    }

    impl DeviceExpertSlotTable {
        pub fn alloc(
            device_id: i32,
            capacity: usize,
            layout: ExpertPackLayout,
        ) -> crate::cuda::Result<Self> {
            use crate::cuda::{CudaError, DeviceBuffer};
            if capacity == 0 || capacity > 65535 {
                return Err(CudaError::new(-1, "invalid DeviceExpertSlotTable capacity"));
            }
            let gate_up_packed =
                DeviceBuffer::alloc(device_id, capacity * layout.gate_up_packed.len())?;
            let gate_up_scales =
                DeviceBuffer::alloc(device_id, capacity * layout.gate_up_scales.len())?;
            let down_packed = DeviceBuffer::alloc(device_id, capacity * layout.down_packed.len())?;
            let down_scales = DeviceBuffer::alloc(device_id, capacity * layout.down_scales.len())?;
            Ok(Self {
                device_id,
                capacity,
                layout,
                gate_up_packed,
                gate_up_scales,
                down_packed,
                down_scales,
            })
        }

        /// Unpacks `num_staged` contiguous expert packs on `stream` into `dst_slots_dev`.
        pub fn unpack_staged(
            &self,
            stream: &crate::cuda::CudaStream,
            staged_packs_addr: u64,
            dst_slots_addr: u64,
            num_staged: usize,
        ) -> crate::cuda::Result<()> {
            if num_staged == 0 {
                return Ok(());
            }
            let ex = stream.exec(0, 0);
            crate::cuda::check_rc(
                // SAFETY: device pointers and capacities are owned and validated by `self` and caller.
                unsafe {
                    w34_kernels::ffi::w34_moe_unpack_staged_experts(
                        &ex,
                        staged_packs_addr as *const core::ffi::c_void,
                        dst_slots_addr as *const i32,
                        self.gate_up_packed.addr() as *mut u32,
                        self.gate_up_scales.addr() as *mut core::ffi::c_void,
                        self.down_packed.addr() as *mut u32,
                        self.down_scales.addr() as *mut core::ffi::c_void,
                        num_staged as i32,
                        self.capacity as i32,
                        self.layout.stride,
                        self.layout.intermediate as i32,
                        self.layout.hidden as i32,
                        self.layout.group_size as i32,
                    )
                },
                "w34_moe_unpack_staged_experts",
            )
        }

        /// Launches `w34_moe_w4a16_mlp_f16` against this slot table.
        #[allow(clippy::too_many_arguments)]
        pub fn launch_moe_w4a16(
            &self,
            ex: &w34_kernels::ffi::W34Exec,
            x_addr: u64,
            remapped_topk_ids_addr: u64,
            topk_weights_addr: u64,
            shared_expert_out_addr: u64,
            shared_expert_gate_addr: u64,
            interm_buf_addr: u64,
            out_addr: u64,
            num_tokens: usize,
            top_k: usize,
        ) -> crate::cuda::Result<()> {
            crate::cuda::check_rc(
                // SAFETY: all device addresses are valid for `num_tokens` and `top_k`.
                unsafe {
                    w34_kernels::ffi::w34_moe_w4a16_mlp_f16(
                        ex,
                        x_addr as *const core::ffi::c_void,
                        self.gate_up_packed.addr() as *const u32,
                        self.gate_up_scales.addr() as *const core::ffi::c_void,
                        self.down_packed.addr() as *const u32,
                        self.down_scales.addr() as *const core::ffi::c_void,
                        remapped_topk_ids_addr as *const i32,
                        topk_weights_addr as *const f32,
                        shared_expert_out_addr as *const core::ffi::c_void,
                        shared_expert_gate_addr as *const core::ffi::c_void,
                        interm_buf_addr as *mut core::ffi::c_void,
                        out_addr as *mut core::ffi::c_void,
                        num_tokens as i32,
                        top_k as i32,
                        self.capacity as i32,
                        self.layout.intermediate as i32,
                        self.layout.hidden as i32,
                        self.layout.group_size as i32,
                    )
                },
                "w34_moe_w4a16_mlp_f16 (tiered)",
            )
        }
    }

    /// Per-layer VRAM hot expert cache (`K_hot` slots) with LRU eviction tracking and telemetry.
    pub struct HotExpertCache {
        pub slots: DeviceExpertSlotTable,
        pub expert_ids: Range<usize>,
        pub expert_to_slot: Vec<i32>,
        pub slot_to_expert: Vec<usize>,
        pub slot_to_expert_i32: Vec<i32>,
        pub slot_last_used: Vec<u64>,
        pub step_clock_buf: [u64; 1],
        pub d_expert_to_slot: crate::cuda::DeviceBuffer,
        pub d_slot_to_expert: crate::cuda::DeviceBuffer,
        pub d_slot_last_used: crate::cuda::DeviceBuffer,
        pub d_step_counter: crate::cuda::DeviceBuffer,
        pub device_lru_dirty: bool,
        pub step_clock: u64,
        pub hits: u64,
        pub misses: u64,
        pub evictions: u64,
        pub dma_experts_loaded: u64,
        pub cpu_cold_evals: u64,
    }

    impl HotExpertCache {
        pub fn new(
            device_id: i32,
            k_hot: usize,
            expert_ids: Range<usize>,
            layout: ExpertPackLayout,
        ) -> crate::cuda::Result<Self> {
            let local_count = expert_ids.len();
            if k_hot == 0 || k_hot > local_count {
                return Err(crate::cuda::CudaError::new(
                    -1,
                    "invalid HotExpertCache k_hot",
                ));
            }
            let slots = DeviceExpertSlotTable::alloc(device_id, k_hot, layout)?;
            let d_expert_to_slot = crate::cuda::DeviceBuffer::alloc(
                device_id,
                local_count * core::mem::size_of::<i32>(),
            )?;
            let d_slot_to_expert =
                crate::cuda::DeviceBuffer::alloc(device_id, k_hot * core::mem::size_of::<i32>())?;
            let d_slot_last_used =
                crate::cuda::DeviceBuffer::alloc(device_id, k_hot * core::mem::size_of::<u64>())?;
            let d_step_counter =
                crate::cuda::DeviceBuffer::alloc(device_id, core::mem::size_of::<u64>())?;
            Ok(Self {
                slots,
                expert_ids,
                expert_to_slot: vec![-1; local_count],
                slot_to_expert: vec![usize::MAX; k_hot],
                slot_to_expert_i32: vec![-1; k_hot],
                slot_last_used: vec![0; k_hot],
                step_clock_buf: [1],
                d_expert_to_slot,
                d_slot_to_expert,
                d_slot_last_used,
                d_step_counter,
                device_lru_dirty: false,
                step_clock: 1,
                hits: 0,
                misses: 0,
                evictions: 0,
                dma_experts_loaded: 0,
                cpu_cold_evals: 0,
            })
        }

        /// Populates the `K_hot` VRAM slots with `initial_experts` (or the first `K_hot` local experts).
        pub fn populate_initial(
            &mut self,
            store: &HostExpertStore,
            initial_experts: Option<&[usize]>,
            ws: &mut TieredMoeWorkspace,
            stream: &crate::cuda::CudaStream,
        ) -> crate::cuda::Result<()> {
            let k_hot = self.slots.capacity;
            let chosen: Vec<usize> = match initial_experts {
                Some(ids) => {
                    if ids.len() != k_hot || ids.iter().any(|id| !self.expert_ids.contains(id)) {
                        return Err(crate::cuda::CudaError::new(
                            -1,
                            "invalid initial_experts for HotExpertCache",
                        ));
                    }
                    ids.to_vec()
                }
                None => self.expert_ids.clone().take(k_hot).collect(),
            };
            self.expert_to_slot.fill(-1);
            for (slot, &ge) in chosen.iter().enumerate() {
                let le = ge - self.expert_ids.start;
                self.expert_to_slot[le] = slot as i32;
                self.slot_to_expert[slot] = ge;
                self.slot_last_used[slot] = self.step_clock;
            }
            self.step_clock += 1;

            // Upload in chunks of `ws.stage_capacity`.
            let chunk_cap = ws.stage_capacity;
            for chunk_start in (0..k_hot).step_by(chunk_cap) {
                let chunk_end = (chunk_start + chunk_cap).min(k_hot);
                let num_staged = chunk_end - chunk_start;
                ws.stage_experts_to_slots(
                    0,
                    store,
                    &chosen[chunk_start..chunk_end],
                    &(chunk_start as i32..chunk_end as i32).collect::<Vec<_>>(),
                    &self.slots,
                    stream,
                )?;
                let _ = num_staged;
            }
            self.d_expert_to_slot
                .copy_from_host(&self.expert_to_slot, stream)?;
            for (dst, &e) in self.slot_to_expert_i32.iter_mut().zip(&self.slot_to_expert) {
                *dst = if e == usize::MAX { -1 } else { e as i32 };
            }
            self.d_slot_to_expert
                .copy_from_host(&self.slot_to_expert_i32, stream)?;
            self.d_slot_last_used
                .copy_from_host(&self.slot_last_used, stream)?;
            self.step_clock_buf[0] = self.step_clock;
            self.d_step_counter
                .copy_from_host(&self.step_clock_buf, stream)?;
            self.device_lru_dirty = false;
            stream.synchronize()
        }

        /// Pulls updated LRU metadata from device after `execute_layer_device_lru` runs.
        pub fn sync_from_device_if_dirty(
            &mut self,
            stream: &crate::cuda::CudaStream,
        ) -> crate::cuda::Result<()> {
            if !self.device_lru_dirty {
                return Ok(());
            }
            let mut slot_i32 = vec![0i32; self.slots.capacity];
            let mut step_arr = [0u64; 1];
            self.d_expert_to_slot
                .copy_to_host(&mut self.expert_to_slot, stream)?;
            self.d_slot_to_expert.copy_to_host(&mut slot_i32, stream)?;
            self.d_slot_last_used
                .copy_to_host(&mut self.slot_last_used, stream)?;
            self.d_step_counter.copy_to_host(&mut step_arr, stream)?;
            stream.synchronize()?;
            for (dst, &src) in self.slot_to_expert.iter_mut().zip(&slot_i32) {
                *dst = if src < 0 { usize::MAX } else { src as usize };
            }
            self.step_clock = step_arr[0].max(1);
            self.device_lru_dirty = false;
            Ok(())
        }

        #[inline]
        pub fn slot_of(&self, global_id: usize) -> Option<i32> {
            if !self.expert_ids.contains(&global_id) {
                return None;
            }
            let slot = self.expert_to_slot[global_id - self.expert_ids.start];
            (slot >= 0).then_some(slot)
        }
    }

    /// Reusable double-buffered staging & scratch workspace for `TieredMoeEngine`.
    pub struct TieredMoeWorkspace {
        pub device_id: i32,
        pub layout: ExpertPackLayout,
        pub stage_capacity: usize,
        pub max_tokens: usize,
        pub top_k: usize,
        pub h_stage_packs: [MappedHostBuffer; 2],
        pub d_stage_packs: [crate::cuda::DeviceBuffer; 2],
        pub d_stage_slots: [crate::cuda::DeviceBuffer; 2],
        pub stage_tables: [DeviceExpertSlotTable; 2],
        pub h2d_stream: crate::cuda::CudaStream,
        pub h2d_done_events: [crate::cuda::CudaEvent; 2],
        pub compute_done_events: [crate::cuda::CudaEvent; 2],
        pub d_remapped_topk_ids: [crate::cuda::DeviceBuffer; 2],
        pub d_interm_buf: crate::cuda::DeviceBuffer,
        pub d_chunk_out: crate::cuda::DeviceBuffer,
        pub d_scratch: crate::cuda::DeviceBuffer,
        pub d_miss_local_experts: crate::cuda::DeviceBuffer,
        pub d_miss_victim_slots: crate::cuda::DeviceBuffer,
        pub d_num_misses: crate::cuda::DeviceBuffer,
    }

    impl TieredMoeWorkspace {
        pub fn new(
            device_id: i32,
            layout: ExpertPackLayout,
            stage_capacity: usize,
            max_tokens: usize,
            top_k: usize,
        ) -> crate::cuda::Result<Self> {
            use crate::cuda::{CudaEvent, CudaStream, DeviceBuffer};
            let stage_cap = stage_capacity.max(1);
            let stage_bytes = stage_cap * layout.stride;
            let pairs = (max_tokens * top_k).max(1);
            let h_stage_packs = [
                MappedHostBuffer::alloc(stage_bytes)?,
                MappedHostBuffer::alloc(stage_bytes)?,
            ];
            let d_stage_packs = [
                DeviceBuffer::alloc(device_id, stage_bytes)?,
                DeviceBuffer::alloc(device_id, stage_bytes)?,
            ];
            let d_stage_slots = [
                DeviceBuffer::alloc(device_id, stage_cap * core::mem::size_of::<i32>())?,
                DeviceBuffer::alloc(device_id, stage_cap * core::mem::size_of::<i32>())?,
            ];
            let stage_tables = [
                DeviceExpertSlotTable::alloc(device_id, stage_cap, layout.clone())?,
                DeviceExpertSlotTable::alloc(device_id, stage_cap, layout.clone())?,
            ];
            let h2d_stream = CudaStream::new(device_id)?;
            let h2d_done_events = [
                CudaEvent::new(device_id, false)?,
                CudaEvent::new(device_id, false)?,
            ];
            let compute_done_events = [
                CudaEvent::new(device_id, false)?,
                CudaEvent::new(device_id, false)?,
            ];
            // Record initial compute_done_events so the first wait_event is immediately satisfied.
            compute_done_events[0].record(&h2d_stream)?;
            compute_done_events[1].record(&h2d_stream)?;

            let d_remapped_topk_ids = [
                DeviceBuffer::alloc(device_id, pairs * core::mem::size_of::<i32>())?,
                DeviceBuffer::alloc(device_id, pairs * core::mem::size_of::<i32>())?,
            ];
            let d_interm_buf = DeviceBuffer::alloc(
                device_id,
                pairs * layout.intermediate * core::mem::size_of::<u16>(),
            )?;
            let d_chunk_out = DeviceBuffer::alloc(
                device_id,
                max_tokens.max(1) * layout.hidden * core::mem::size_of::<u16>(),
            )?;
            // Scratch for grouped-MMA token-expert sorting inside `w34_moe_w4a16_mlp_f16`.
            let d_scratch = DeviceBuffer::alloc(device_id, 16 * 1024 * 1024)?;
            let max_misses = pairs.clamp(16, 512);
            let d_miss_local_experts =
                DeviceBuffer::alloc(device_id, max_misses * core::mem::size_of::<i32>())?;
            let d_miss_victim_slots =
                DeviceBuffer::alloc(device_id, max_misses * core::mem::size_of::<i32>())?;
            let d_num_misses = DeviceBuffer::alloc(device_id, core::mem::size_of::<i32>())?;

            Ok(Self {
                device_id,
                layout,
                stage_capacity: stage_cap,
                max_tokens,
                top_k,
                h_stage_packs,
                d_stage_packs,
                d_stage_slots,
                stage_tables,
                h2d_stream,
                h2d_done_events,
                compute_done_events,
                d_remapped_topk_ids,
                d_interm_buf,
                d_chunk_out,
                d_scratch,
                d_miss_local_experts,
                d_miss_victim_slots,
                d_num_misses,
            })
        }

        /// Copies `global_expert_ids` into pinned `h_stage_packs[buf_idx]`, DMAs them to
        /// `d_stage_packs[buf_idx]`, and scatters them into `dst_table` at `dst_slots`.
        pub fn stage_experts_to_slots(
            &self,
            buf_idx: usize,
            store: &HostExpertStore,
            global_expert_ids: &[usize],
            dst_slots: &[i32],
            dst_table: &DeviceExpertSlotTable,
            stream: &crate::cuda::CudaStream,
        ) -> crate::cuda::Result<()> {
            let n = global_expert_ids.len();
            if n == 0 {
                return Ok(());
            }
            assert_eq!(n, dst_slots.len());
            assert!(n <= self.stage_capacity);
            self.compute_done_events[buf_idx].synchronize()?;
            let stride = self.layout.stride;
            let total_bytes = n * stride;
            // SAFETY: caller ensures previous DMA from `h_stage_packs[buf_idx]` has completed.
            let pinned_dst = unsafe {
                std::slice::from_raw_parts_mut(
                    self.h_stage_packs[buf_idx].host_ptr.cast::<u8>(),
                    total_bytes,
                )
            };
            for (i, &ge) in global_expert_ids.iter().enumerate() {
                let src = store
                    .expert_pack(ge)
                    .expect("expert not owned by local HostExpertStore");
                pinned_dst[i * stride..(i + 1) * stride].copy_from_slice(src);
            }
            self.d_stage_packs[buf_idx].copy_from_host(pinned_dst, stream)?;
            self.d_stage_slots[buf_idx].copy_from_host(dst_slots, stream)?;
            dst_table.unpack_staged(
                stream,
                self.d_stage_packs[buf_idx].addr(),
                self.d_stage_slots[buf_idx].addr(),
                n,
            )?;
            self.compute_done_events[buf_idx].record(stream)
        }
    }

    /// Executes a tiered 512-expert MoE layer across `TP=1` or `TP=2 / EP=2` in `GpuLru`,
    /// `CpuHybrid`, or double-buffered `StreamedPrefill` mode.
    pub struct TieredMoeEngine;

    impl TieredMoeEngine {
        /// Pure device-side CUDA-Graph-safe `GpuLru` execution for `num_tokens <= 16`.
        /// Uses `w34_moe_gpu_lru_update_and_pull` to update on-device LRU tables and pull any
        /// missed experts over zero-copy PCIe from `store.mapped_dev_ptr()` (when non-zero),
        /// followed by `w34_moe_w4a16_mlp_f16`.
        #[allow(clippy::too_many_arguments)]
        pub fn execute_layer_device_lru(
            store: &HostExpertStore,
            cache: &mut HotExpertCache,
            ws: &TieredMoeWorkspace,
            compute_stream: &crate::cuda::CudaStream,
            x_addr: u64,
            d_topk_ids_addr: u64,
            d_topk_weights_addr: u64,
            shared_expert_out_addr: u64,
            shared_expert_gate_addr: u64,
            out_addr: u64,
            num_tokens: usize,
            top_k: usize,
        ) -> crate::cuda::Result<()> {
            if num_tokens == 0 {
                return Ok(());
            }
            cache.device_lru_dirty = true;
            let total_pairs = num_tokens * top_k;
            let k_hot = cache.slots.capacity;
            let max_misses = total_pairs
                .min(k_hot)
                .min(ws.d_miss_local_experts.len_bytes / 4);
            let ex = compute_stream.exec(ws.d_scratch.addr(), ws.d_scratch.len_bytes);
            crate::cuda::check_rc(
                // SAFETY: device pointers in `cache` and `ws` are sized for `total_pairs`, `k_hot`, and `max_misses`.
                unsafe {
                    w34_kernels::ffi::w34_moe_gpu_lru_update_and_pull(
                        &ex,
                        d_topk_ids_addr as *const i32,
                        cache.d_expert_to_slot.addr() as *mut i32,
                        cache.d_slot_to_expert.addr() as *mut i32,
                        cache.d_slot_last_used.addr() as *mut u64,
                        cache.d_step_counter.addr() as *mut u64,
                        ws.d_remapped_topk_ids[0].addr() as *mut i32,
                        ws.d_miss_local_experts.addr() as *mut i32,
                        ws.d_miss_victim_slots.addr() as *mut i32,
                        ws.d_num_misses.addr() as *mut i32,
                        store.mapped_dev_ptr() as *const core::ffi::c_void,
                        cache.slots.gate_up_packed.addr() as *mut u32,
                        cache.slots.gate_up_scales.addr() as *mut core::ffi::c_void,
                        cache.slots.down_packed.addr() as *mut u32,
                        cache.slots.down_scales.addr() as *mut core::ffi::c_void,
                        total_pairs as i32,
                        cache.expert_ids.start as i32,
                        cache.expert_ids.len() as i32,
                        k_hot as i32,
                        max_misses as i32,
                        store.layout().stride,
                        store.layout().intermediate as i32,
                        store.layout().hidden as i32,
                        store.layout().group_size as i32,
                    )
                },
                "w34_moe_gpu_lru_update_and_pull",
            )?;
            cache.slots.launch_moe_w4a16(
                &ex,
                x_addr,
                ws.d_remapped_topk_ids[0].addr(),
                d_topk_weights_addr,
                shared_expert_out_addr,
                shared_expert_gate_addr,
                ws.d_interm_buf.addr(),
                out_addr,
                num_tokens,
                top_k,
            )
        }

        #[allow(clippy::too_many_arguments, clippy::cognitive_complexity)]
        pub fn execute_layer(
            mode: MoeOffloadMode,
            store: &HostExpertStore,
            cache: &mut HotExpertCache,
            ws: &mut TieredMoeWorkspace,
            compute_stream: &crate::cuda::CudaStream,
            x_addr: u64,
            d_topk_ids_addr: u64,
            d_topk_weights_addr: u64,
            h_topk_ids: &[i32],
            h_topk_weights: &[f32],
            shared_expert_out_addr: u64,
            shared_expert_gate_addr: u64,
            out_addr: u64,
            num_tokens: usize,
            top_k: usize,
        ) -> crate::cuda::Result<()> {
            if num_tokens == 0 {
                return Ok(());
            }
            cache.sync_from_device_if_dirty(compute_stream)?;
            assert_eq!(h_topk_ids.len(), num_tokens * top_k);
            assert_eq!(h_topk_weights.len(), num_tokens * top_k);

            // Identify unique active local experts and count cache hits/misses.
            let mut active_local: Vec<usize> = Vec::new();
            let mut hits_in_step: Vec<usize> = Vec::new();
            let mut misses_in_step: Vec<usize> = Vec::new();

            for &raw_id in h_topk_ids {
                if raw_id < 0 {
                    continue;
                }
                let ge = raw_id as usize;
                if !cache.expert_ids.contains(&ge) {
                    continue;
                }
                if !active_local.contains(&ge) {
                    active_local.push(ge);
                }
                if cache.slot_of(ge).is_some() {
                    cache.hits += 1;
                    if !hits_in_step.contains(&ge) {
                        hits_in_step.push(ge);
                    }
                } else {
                    cache.misses += 1;
                    if !misses_in_step.contains(&ge) {
                        misses_in_step.push(ge);
                    }
                }
            }

            // Touch LRU timestamps on all hit slots first so they cannot be evicted this step.
            let step = cache.step_clock;
            cache.step_clock += 1;
            for &ge in &hits_in_step {
                let slot = cache.slot_of(ge).unwrap() as usize;
                cache.slot_last_used[slot] = step;
            }

            let effective_mode = match mode {
                MoeOffloadMode::Auto => {
                    if active_local.len() <= cache.slots.capacity
                        && misses_in_step.len() <= ws.stage_capacity
                    {
                        MoeOffloadMode::GpuLru
                    } else {
                        MoeOffloadMode::StreamedPrefill
                    }
                }
                MoeOffloadMode::GpuLru => {
                    if active_local.len() <= cache.slots.capacity
                        && misses_in_step.len() <= ws.stage_capacity
                    {
                        MoeOffloadMode::GpuLru
                    } else {
                        MoeOffloadMode::StreamedPrefill
                    }
                }
                other => other,
            };

            let ex = compute_stream.exec(ws.d_scratch.addr(), ws.d_scratch.len_bytes);

            match effective_mode {
                MoeOffloadMode::GpuLru => {
                    if !misses_in_step.is_empty() {
                        let mut victim_slots = Vec::with_capacity(misses_in_step.len());
                        for &ge in &misses_in_step {
                            // Choose the slot with the smallest `slot_last_used` timestamp.
                            let (victim_slot, _) = cache
                                .slot_last_used
                                .iter()
                                .enumerate()
                                .min_by_key(|&(_, &ts)| ts)
                                .unwrap();
                            let old_ge = cache.slot_to_expert[victim_slot];
                            if old_ge < usize::MAX {
                                cache.expert_to_slot[old_ge - cache.expert_ids.start] = -1;
                                cache.evictions += 1;
                            }
                            cache.expert_to_slot[ge - cache.expert_ids.start] = victim_slot as i32;
                            cache.slot_to_expert[victim_slot] = ge;
                            cache.slot_last_used[victim_slot] = step;
                            victim_slots.push(victim_slot as i32);
                        }
                        cache.dma_experts_loaded += misses_in_step.len() as u64;
                        ws.stage_experts_to_slots(
                            0,
                            store,
                            &misses_in_step,
                            &victim_slots,
                            &cache.slots,
                            compute_stream,
                        )?;
                        ws.compute_done_events[0].record(compute_stream)?;
                        cache
                            .d_expert_to_slot
                            .copy_from_host(&cache.expert_to_slot, compute_stream)?;
                        for (dst, &e) in cache
                            .slot_to_expert_i32
                            .iter_mut()
                            .zip(&cache.slot_to_expert)
                        {
                            *dst = if e == usize::MAX { -1 } else { e as i32 };
                        }
                        cache
                            .d_slot_to_expert
                            .copy_from_host(&cache.slot_to_expert_i32, compute_stream)?;
                        cache
                            .d_slot_last_used
                            .copy_from_host(&cache.slot_last_used, compute_stream)?;
                        cache.step_clock_buf[0] = cache.step_clock;
                        cache
                            .d_step_counter
                            .copy_from_host(&cache.step_clock_buf, compute_stream)?;
                    }
                    // Remap global `d_topk_ids_addr` on GPU using `cache.d_expert_to_slot`.
                    crate::cuda::check_rc(
                        // SAFETY: pointers and counts match allocated device buffers.
                        unsafe {
                            w34_kernels::ffi::w34_moe_remap_topk_ids_i32(
                                &ex,
                                d_topk_ids_addr as *const i32,
                                cache.d_expert_to_slot.addr() as *const i32,
                                ws.d_remapped_topk_ids[0].addr() as *mut i32,
                                (num_tokens * top_k) as i32,
                                cache.expert_ids.start as i32,
                                cache.expert_ids.len() as i32,
                            )
                        },
                        "w34_moe_remap_topk_ids_i32",
                    )?;
                    cache.slots.launch_moe_w4a16(
                        &ex,
                        x_addr,
                        ws.d_remapped_topk_ids[0].addr(),
                        d_topk_weights_addr,
                        shared_expert_out_addr,
                        shared_expert_gate_addr,
                        ws.d_interm_buf.addr(),
                        out_addr,
                        num_tokens,
                        top_k,
                    )
                }

                MoeOffloadMode::CpuHybrid { gpu_pcie_budget } => {
                    // Optionally promote up to `gpu_pcie_budget` cold misses into GPU LRU slots.
                    let gpu_promote = gpu_pcie_budget
                        .min(misses_in_step.len())
                        .min(ws.stage_capacity)
                        .min(cache.slots.capacity.saturating_sub(hits_in_step.len()));
                    if gpu_promote > 0 {
                        let promoted = &misses_in_step[..gpu_promote];
                        let mut victim_slots = Vec::with_capacity(gpu_promote);
                        for &ge in promoted {
                            let (victim_slot, _) = cache
                                .slot_last_used
                                .iter()
                                .enumerate()
                                .min_by_key(|&(_, &ts)| ts)
                                .unwrap();
                            let old_ge = cache.slot_to_expert[victim_slot];
                            if old_ge < usize::MAX {
                                cache.expert_to_slot[old_ge - cache.expert_ids.start] = -1;
                                cache.evictions += 1;
                            }
                            cache.expert_to_slot[ge - cache.expert_ids.start] = victim_slot as i32;
                            cache.slot_to_expert[victim_slot] = ge;
                            cache.slot_last_used[victim_slot] = step;
                            victim_slots.push(victim_slot as i32);
                        }
                        cache.dma_experts_loaded += gpu_promote as u64;
                        ws.stage_experts_to_slots(
                            0,
                            store,
                            promoted,
                            &victim_slots,
                            &cache.slots,
                            compute_stream,
                        )?;
                        ws.compute_done_events[0].record(compute_stream)?;
                        cache
                            .d_expert_to_slot
                            .copy_from_host(&cache.expert_to_slot, compute_stream)?;
                        for (dst, &e) in cache
                            .slot_to_expert_i32
                            .iter_mut()
                            .zip(&cache.slot_to_expert)
                        {
                            *dst = if e == usize::MAX { -1 } else { e as i32 };
                        }
                        cache
                            .d_slot_to_expert
                            .copy_from_host(&cache.slot_to_expert_i32, compute_stream)?;
                        cache
                            .d_slot_last_used
                            .copy_from_host(&cache.slot_last_used, compute_stream)?;
                        cache.step_clock_buf[0] = cache.step_clock;
                        cache
                            .d_step_counter
                            .copy_from_host(&cache.step_clock_buf, compute_stream)?;
                    }

                    // Partition pairs into GPU hot/promoted slots vs CPU cold tasks.
                    let mut remapped_hot = vec![-1i32; num_tokens * top_k];
                    let mut cold_tasks = Vec::new();
                    for tok in 0..num_tokens {
                        for k in 0..top_k {
                            let idx = tok * top_k + k;
                            let raw_id = h_topk_ids[idx];
                            if raw_id < 0 {
                                continue;
                            }
                            let ge = raw_id as usize;
                            if !cache.expert_ids.contains(&ge) {
                                continue;
                            }
                            if let Some(slot) = cache.slot_of(ge) {
                                remapped_hot[idx] = slot;
                            } else {
                                cold_tasks.push((tok, ge, h_topk_weights[idx]));
                            }
                        }
                    }

                    ws.d_remapped_topk_ids[0].copy_from_host(&remapped_hot, compute_stream)?;

                    // Download `x` to host if there are cold CPU tasks, then launch GPU hot MoE
                    // concurrently while the CPU thread pool evaluates cold experts!
                    let mut h_x = vec![
                        0u16;
                        if cold_tasks.is_empty() {
                            0
                        } else {
                            num_tokens * ws.layout.hidden
                        }
                    ];
                    if !cold_tasks.is_empty() {
                        crate::cuda::check_rc(
                            // SAFETY: `h_x` and `x_addr` cover `num_tokens * hidden` FP16 elements.
                            unsafe {
                                w34_kernels::ffi::w34_memcpy_d2h_async(
                                    h_x.as_mut_ptr().cast::<core::ffi::c_void>(),
                                    x_addr as *const core::ffi::c_void,
                                    core::mem::size_of_val(h_x.as_slice()),
                                    compute_stream.handle,
                                )
                            },
                            "w34_memcpy_d2h_async (CpuHybrid x)",
                        )?;
                        compute_stream.synchronize()?;
                    }

                    cache.slots.launch_moe_w4a16(
                        &ex,
                        x_addr,
                        ws.d_remapped_topk_ids[0].addr(),
                        d_topk_weights_addr,
                        shared_expert_out_addr,
                        shared_expert_gate_addr,
                        ws.d_interm_buf.addr(),
                        out_addr,
                        num_tokens,
                        top_k,
                    )?;

                    if !cold_tasks.is_empty() {
                        cache.cpu_cold_evals += cold_tasks.len() as u64;
                        let h_cold_out = store.eval_cold_tasks_cpu(&h_x, num_tokens, &cold_tasks);
                        ws.d_chunk_out.copy_from_host(&h_cold_out, compute_stream)?;
                        crate::cuda::check_rc(
                            // SAFETY: `out_addr` and `ws.d_chunk_out` are disjoint 4-byte-aligned FP16 buffers.
                            unsafe {
                                w34_kernels::ffi::w34_add_inplace_f16(
                                    &ex,
                                    out_addr as *mut core::ffi::c_void,
                                    ws.d_chunk_out.addr() as *const core::ffi::c_void,
                                    (num_tokens * ws.layout.hidden) as i64,
                                )
                            },
                            "w34_add_inplace_f16 (CpuHybrid)",
                        )?;
                    }
                    Ok(())
                }

                MoeOffloadMode::StreamedPrefill => {
                    // Pass 0: Execute all hot-cache hits on `compute_stream` (plus shared expert).
                    crate::cuda::check_rc(
                        // SAFETY: pointers and counts match allocated device buffers.
                        unsafe {
                            w34_kernels::ffi::w34_moe_remap_topk_ids_i32(
                                &ex,
                                d_topk_ids_addr as *const i32,
                                cache.d_expert_to_slot.addr() as *const i32,
                                ws.d_remapped_topk_ids[0].addr() as *mut i32,
                                (num_tokens * top_k) as i32,
                                cache.expert_ids.start as i32,
                                cache.expert_ids.len() as i32,
                            )
                        },
                        "w34_moe_remap_topk_ids_i32 (Pass 0)",
                    )?;

                    // Before launching Pass 0 on `compute_stream`, kick off Chunk 0 H2D on `ws.h2d_stream`
                    // so PCIe DMA overlaps with Pass 0 GPU compute!
                    let chunks: Vec<&[usize]> = misses_in_step.chunks(ws.stage_capacity).collect();
                    if !chunks.is_empty() {
                        let c0 = chunks[0];
                        let dst_slots: Vec<i32> = (0..c0.len() as i32).collect();
                        ws.h2d_stream.wait_event(&ws.compute_done_events[0])?;
                        ws.stage_experts_to_slots(
                            0,
                            store,
                            c0,
                            &dst_slots,
                            &ws.stage_tables[0],
                            &ws.h2d_stream,
                        )?;
                        cache.dma_experts_loaded += c0.len() as u64;
                        ws.h2d_done_events[0].record(&ws.h2d_stream)?;
                    }

                    cache.slots.launch_moe_w4a16(
                        &ex,
                        x_addr,
                        ws.d_remapped_topk_ids[0].addr(),
                        d_topk_weights_addr,
                        shared_expert_out_addr,
                        shared_expert_gate_addr,
                        ws.d_interm_buf.addr(),
                        out_addr,
                        num_tokens,
                        top_k,
                    )?;

                    // Passes 1..=chunks.len(): Double-buffered H2D + Grouped-MMA streaming.
                    for (c_idx, chunk) in chunks.iter().enumerate() {
                        let buf_idx = c_idx & 1;
                        // Enqueue H2D for `c_idx + 1` on `ws.h2d_stream` into `1 - buf_idx`
                        // before launching compute for `c_idx` on `compute_stream`!
                        if c_idx + 1 < chunks.len() {
                            let next_buf = 1 - buf_idx;
                            let next_chunk = chunks[c_idx + 1];
                            let dst_slots: Vec<i32> = (0..next_chunk.len() as i32).collect();
                            ws.h2d_stream
                                .wait_event(&ws.compute_done_events[next_buf])?;
                            ws.stage_experts_to_slots(
                                next_buf,
                                store,
                                next_chunk,
                                &dst_slots,
                                &ws.stage_tables[next_buf],
                                &ws.h2d_stream,
                            )?;
                            cache.dma_experts_loaded += next_chunk.len() as u64;
                            ws.h2d_done_events[next_buf].record(&ws.h2d_stream)?;
                        }

                        // Build per-pair remapped IDs for `chunk`.
                        let mut chunk_remap = vec![-1i32; num_tokens * top_k];
                        for (pair_idx, &raw_id) in h_topk_ids.iter().enumerate() {
                            if raw_id < 0 {
                                continue;
                            }
                            let ge = raw_id as usize;
                            if let Some(pos) = chunk.iter().position(|&e| e == ge) {
                                chunk_remap[pair_idx] = pos as i32;
                            }
                        }

                        compute_stream.wait_event(&ws.h2d_done_events[buf_idx])?;
                        ws.d_remapped_topk_ids[buf_idx]
                            .copy_from_host(&chunk_remap, compute_stream)?;
                        ws.stage_tables[buf_idx].launch_moe_w4a16(
                            &ex,
                            x_addr,
                            ws.d_remapped_topk_ids[buf_idx].addr(),
                            d_topk_weights_addr,
                            0,
                            0,
                            ws.d_interm_buf.addr(),
                            ws.d_chunk_out.addr(),
                            num_tokens,
                            top_k,
                        )?;
                        crate::cuda::check_rc(
                            // SAFETY: `out_addr` and `ws.d_chunk_out` are disjoint 4-byte-aligned FP16 buffers.
                            unsafe {
                                w34_kernels::ffi::w34_add_inplace_f16(
                                    &ex,
                                    out_addr as *mut core::ffi::c_void,
                                    ws.d_chunk_out.addr() as *const core::ffi::c_void,
                                    (num_tokens * ws.layout.hidden) as i64,
                                )
                            },
                            "w34_add_inplace_f16 (StreamedPrefill)",
                        )?;
                        ws.compute_done_events[buf_idx].record(compute_stream)?;
                    }
                    Ok(())
                }

                MoeOffloadMode::Auto => unreachable!(),
            }
        }
    }

    /// Per-rank 48-layer tiered 512-expert MoE runtime (`stores`, `caches`, `ws`, and `mode`).
    pub struct RankTieredMoe {
        pub stores: Vec<HostExpertStore>,
        pub caches: Vec<HotExpertCache>,
        pub ws: TieredMoeWorkspace,
        pub mode: MoeOffloadMode,
    }

    impl RankTieredMoe {
        #[allow(clippy::too_many_arguments)]
        pub fn load_for_rank(
            archive: &SafetensorsArchive,
            cfg: &crate::loader::ModelConfig,
            device_id: i32,
            ep_rank: usize,
            ep_size: usize,
            k_hot: usize,
            stage_capacity: usize,
            stream: &crate::cuda::CudaStream,
        ) -> crate::cuda::Result<Self> {
            let group_size = cfg.quant.group_size_or(128);
            let layout =
                ExpertPackLayout::new(cfg.moe_intermediate_size, cfg.hidden_size, group_size)
                    .map_err(|e| {
                        crate::cuda::CudaError::with_detail(
                            -1,
                            "ExpertPackLayout::new",
                            e.to_string(),
                        )
                    })?;
            let mut ws = TieredMoeWorkspace::new(
                device_id,
                layout.clone(),
                stage_capacity,
                2048,
                cfg.num_experts_per_tok.max(1),
            )?;
            let mut stores = Vec::with_capacity(cfg.num_hidden_layers);
            let mut caches = Vec::with_capacity(cfg.num_hidden_layers);
            for layer_id in 0..cfg.num_hidden_layers {
                let prefix = format!("model.layers.{layer_id}.mlp.experts");
                let store = HostExpertStore::load(
                    archive,
                    &prefix,
                    &cfg.quant,
                    layout.clone(),
                    cfg.num_experts,
                    ep_rank,
                    ep_size,
                    usize::MAX,
                )
                .map_err(|e| {
                    crate::cuda::CudaError::with_detail(-1, "HostExpertStore::load", e.to_string())
                })?;
                let mut cache =
                    HotExpertCache::new(device_id, k_hot, store.expert_ids(), layout.clone())?;
                cache.populate_initial(&store, None, &mut ws, stream)?;
                stores.push(store);
                caches.push(cache);
            }
            Ok(Self {
                stores,
                caches,
                ws,
                mode: MoeOffloadMode::Auto,
            })
        }

        pub fn layer_has_mapped_store(&self, layer_idx: usize) -> bool {
            self.stores
                .get(layer_idx)
                .is_some_and(|s| s.mapped_dev_ptr() != 0)
        }

        pub fn layer_hot_k(&self, layer_idx: usize) -> usize {
            self.caches.get(layer_idx).map_or(0, |c| c.slots.capacity)
        }

        #[allow(clippy::too_many_arguments)]
        pub fn execute_layer_device_lru(
            &mut self,
            compute_stream: &crate::cuda::CudaStream,
            layer_idx: usize,
            x_addr: u64,
            d_topk_ids_addr: u64,
            d_topk_weights_addr: u64,
            shared_expert_out_addr: u64,
            shared_expert_gate_addr: u64,
            out_addr: u64,
            num_tokens: usize,
            top_k: usize,
        ) -> crate::cuda::Result<()> {
            TieredMoeEngine::execute_layer_device_lru(
                &self.stores[layer_idx],
                &mut self.caches[layer_idx],
                &self.ws,
                compute_stream,
                x_addr,
                d_topk_ids_addr,
                d_topk_weights_addr,
                shared_expert_out_addr,
                shared_expert_gate_addr,
                out_addr,
                num_tokens,
                top_k,
            )
        }

        #[allow(clippy::too_many_arguments)]
        pub fn execute_layer(
            &mut self,
            compute_stream: &crate::cuda::CudaStream,
            layer_idx: usize,
            x_addr: u64,
            d_topk_ids_addr: u64,
            d_topk_weights_addr: u64,
            shared_expert_out_addr: u64,
            shared_expert_gate_addr: u64,
            out_addr: u64,
            num_tokens: usize,
            top_k: usize,
        ) -> crate::cuda::Result<()> {
            if num_tokens == 0 {
                return Ok(());
            }
            let pairs = num_tokens * top_k;
            let mut h_topk_ids = vec![0i32; pairs];
            let mut h_topk_weights = vec![0.0f32; pairs];
            crate::cuda::check_rc(
                // SAFETY: `d_topk_ids_addr` and `d_topk_weights_addr` hold `pairs` elements.
                unsafe {
                    w34_kernels::ffi::w34_memcpy_d2h_async(
                        h_topk_ids.as_mut_ptr().cast::<core::ffi::c_void>(),
                        d_topk_ids_addr as *const core::ffi::c_void,
                        pairs * 4,
                        compute_stream.handle,
                    )
                },
                "w34_memcpy_d2h_async (h_topk_ids)",
            )?;
            crate::cuda::check_rc(
                // SAFETY: `d_topk_weights_addr` holds `pairs` f32 elements.
                unsafe {
                    w34_kernels::ffi::w34_memcpy_d2h_async(
                        h_topk_weights.as_mut_ptr().cast::<core::ffi::c_void>(),
                        d_topk_weights_addr as *const core::ffi::c_void,
                        pairs * 4,
                        compute_stream.handle,
                    )
                },
                "w34_memcpy_d2h_async (h_topk_weights)",
            )?;
            compute_stream.synchronize()?;
            TieredMoeEngine::execute_layer(
                self.mode,
                &self.stores[layer_idx],
                &mut self.caches[layer_idx],
                &mut self.ws,
                compute_stream,
                x_addr,
                d_topk_ids_addr,
                d_topk_weights_addr,
                &h_topk_ids,
                &h_topk_weights,
                shared_expert_out_addr,
                shared_expert_gate_addr,
                out_addr,
                num_tokens,
                top_k,
            )
        }
    }
}

#[cfg(feature = "cuda")]
pub use host_store::{
    DeviceExpertSlotTable, HostExpertStore, HotExpertCache, RankTieredMoe, TieredMoeEngine,
    TieredMoeWorkspace,
};

#[cfg(test)]
mod layout_tests {
    use super::*;

    #[test]
    fn canonical_pack_has_exact_regions_and_page_aligned_stride() {
        let layout = ExpertPackLayout::new(640, 2560, 128).unwrap();
        assert_eq!(layout.gate_up_packed, 0..1_638_400);
        assert_eq!(layout.gate_up_scales.len(), 51_200);
        assert_eq!(layout.down_packed.len(), 819_200);
        assert_eq!(layout.down_scales.len(), 25_600);
        assert_eq!(layout.down_scales.end, 2_534_400);
        assert_eq!(layout.stride, 2_535_424);
    }

    #[test]
    fn invalid_or_overflowing_geometry_is_rejected() {
        for (i, h, g) in [
            (0, 128, 128),
            (64, 128, 128),
            (640, 2561, 128),
            (640, 2560, 0),
            (usize::MAX, 2560, 128),
        ] {
            assert!(ExpertPackLayout::new(i, h, g).is_err());
        }
    }
}

#[cfg(all(test, feature = "cuda"))]
mod tests {
    use crate::cuda::{CudaDevice, CudaStream, DeviceBuffer};
    use w34_kernels::ffi;

    #[test]
    #[ignore = "requires the pinned Qwen3.8-Flash-Next checkpoint"]
    fn canonical_checkpoint_host_store_tp1_and_ep2_agree() {
        use super::{ExpertPackLayout, HostExpertStore};
        use crate::loader::{ModelConfig, SafetensorsArchive};
        let _device = CudaDevice::set_current(0).unwrap();
        let snapshot = std::path::Path::new(
            "/home/nchapman/.cache/huggingface/hub/models--albucino--Qwen3.8-Flash-Next-W4A16-FP8PLE/snapshots/ef554143369a706525336f6b42a09094835dc077",
        );
        let archive = SafetensorsArchive::open(snapshot).unwrap();
        let cfg = ModelConfig::from_dir(snapshot).unwrap();
        let layout = ExpertPackLayout::new(640, 2560, 128).unwrap();
        let prefix = "model.language_model.layers.0.mlp.experts";
        // Two real experts keep this qualification small; EP uses the same partition rule at 512.
        let single = HostExpertStore::load(
            &archive,
            prefix,
            &cfg.quant,
            layout.clone(),
            2,
            0,
            1,
            2 * layout.stride,
        )
        .unwrap();
        for rank in 0..2 {
            let shard = HostExpertStore::load(
                &archive,
                prefix,
                &cfg.quant,
                layout.clone(),
                2,
                rank,
                2,
                layout.stride,
            )
            .unwrap();
            assert_eq!(shard.expert_ids(), rank..rank + 1);
            assert_eq!(shard.pinned_bytes(), layout.stride);
            assert_eq!(single.expert_pack(rank), shard.expert_pack(rank));
            assert!(shard.expert_pack(1 - rank).is_none());
            let pack = shard.expert_pack(rank).unwrap();
            for (projection, region, n, k) in [
                (
                    "gate_proj",
                    layout.gate_up_packed.start..layout.gate_up_packed.start + 819200,
                    640,
                    2560,
                ),
                ("up_proj", 819200..layout.gate_up_packed.end, 640, 2560),
                ("down_proj", layout.down_packed.clone(), 2560, 640),
            ] {
                let original = archive
                    .read_u32(&format!("{prefix}.{rank}.{projection}.qweight"))
                    .unwrap();
                for row in [0, n / 2, n - 1] {
                    for col in [0, k / 16, k / 8 - 1] {
                        let offset = region.start + (row * (k / 8) + col) * 4;
                        let word = u32::from_le_bytes(pack[offset..offset + 4].try_into().unwrap());
                        assert_eq!(word, original[col * n + row]);
                    }
                }
            }
            assert!(pack[layout.down_scales.end..].iter().all(|&v| v == 0));
        }
        assert!(
            HostExpertStore::load(
                &archive,
                prefix,
                &cfg.quant,
                layout.clone(),
                2,
                0,
                1,
                layout.stride
            )
            .is_err()
        );
    }

    fn upload<T: Copy>(values: &[T], stream: &CudaStream) -> DeviceBuffer {
        let buffer = DeviceBuffer::alloc(0, std::mem::size_of_val(values)).unwrap();
        buffer.copy_from_host(values, stream).unwrap();
        stream.synchronize().unwrap();
        buffer
    }

    #[test]
    fn staged_experts_scatter_all_regions_and_remap_ep_ids() {
        let _device = CudaDevice::set_current(0).unwrap();
        let stream = CudaStream::new(0).unwrap();
        let ex = stream.exec(0, 0);
        let layout = super::ExpertPackLayout::new(128, 128, 128).unwrap();
        let source: Vec<u8> = (0..layout.stride * 2)
            .map(|i| ((i * 17 + i / layout.stride) % 251) as u8)
            .collect();
        let staged = upload(&source, &stream);
        let slots = upload(&[2i32, 0], &stream);
        let regions = [
            &layout.gate_up_packed,
            &layout.gate_up_scales,
            &layout.down_packed,
            &layout.down_scales,
        ];
        let outputs: Vec<_> = regions
            .iter()
            .map(|r| upload(&vec![0u8; r.len() * 3], &stream))
            .collect();
        // SAFETY: all buffers cover the specified geometry and destination slots are unique and in bounds.
        assert_eq!(
            // SAFETY: allocated buffers cover the geometry; destination slots are unique and bounded.
            unsafe {
                ffi::w34_moe_unpack_staged_experts(
                    &ex,
                    staged.addr() as *const _,
                    slots.addr() as *const _,
                    outputs[0].addr() as *mut _,
                    outputs[1].addr() as *mut _,
                    outputs[2].addr() as *mut _,
                    outputs[3].addr() as *mut _,
                    2,
                    3,
                    layout.stride,
                    128,
                    128,
                    128,
                )
            },
            0
        );
        stream.synchronize().unwrap();
        for (region, buffer) in regions.iter().zip(&outputs) {
            let mut actual = vec![0u8; region.len() * 3];
            buffer.copy_to_host(&mut actual, &stream).unwrap();
            assert_eq!(
                &actual[..region.len()],
                &source[layout.stride + region.start..layout.stride + region.end]
            );
            assert!(
                actual[region.len()..2 * region.len()]
                    .iter()
                    .all(|&v| v == 0)
            );
            assert_eq!(&actual[2 * region.len()..], &source[(*region).clone()]);
        }
        let ids = upload(&[-1i32, 255, 256, 257, 258, i32::MIN, i32::MAX], &stream);
        let map = upload(&[2i32, -1], &stream);
        let result = upload(&[99i32; 7], &stream);
        // SAFETY: input/output arrays have seven entries and map covers two local experts.
        assert_eq!(
            // SAFETY: input/output have seven entries and map has two local experts.
            unsafe {
                ffi::w34_moe_remap_topk_ids_i32(
                    &ex,
                    ids.addr() as *const _,
                    map.addr() as *const _,
                    result.addr() as *mut _,
                    7,
                    256,
                    2,
                )
            },
            0
        );
        let mut actual = [0i32; 7];
        result.copy_to_host(&mut actual, &stream).unwrap();
        assert_eq!(actual, [-1, -1, 2, -1, -1, -1, -1]);
    }

    #[test]
    fn staged_experts_reject_invalid_geometry() {
        let _device = CudaDevice::set_current(0).unwrap();
        let stream = CudaStream::new(0).unwrap();
        let ex = stream.exec(0, 0);
        let input = DeviceBuffer::alloc(0, 65536).unwrap();
        let output = DeviceBuffer::alloc(0, 65536).unwrap();
        let slots = DeviceBuffer::alloc(0, 4).unwrap();
        slots.copy_from_host(&[-1i32], &stream).unwrap();
        stream.synchronize().unwrap();
        for (intermediate, hidden, group, stride) in [
            (64, 128, 128, 16384),  // incomplete down-projection group
            (128, 128, 128, 25089), // misaligned second staged pack
            (128, 128, 128, 16),    // undersized pack
        ] {
            // SAFETY: buffers are live, aligned allocations; negative slot suppresses writes.
            let rc = unsafe {
                ffi::w34_moe_unpack_staged_experts(
                    &ex,
                    input.addr() as *const _,
                    slots.addr() as *const i32,
                    output.addr() as *mut u32,
                    (output.addr() + 32768) as *mut _,
                    (output.addr() + 40960) as *mut u32,
                    (output.addr() + 57344) as *mut _,
                    1,
                    1,
                    stride,
                    intermediate,
                    hidden,
                    group,
                )
            };
            assert_ne!(
                rc, 0,
                "accepted geometry {intermediate}/{hidden}/{group}/{stride}"
            );
        }
        stream.synchronize().unwrap();
    }

    #[test]
    fn tiered_moe_gpu_lru_cpu_hybrid_and_streamed_prefill_parity_tp1_and_ep2() {
        use super::{
            DeviceExpertSlotTable, ExpertPackLayout, HostExpertStore, HotExpertCache,
            MoeOffloadMode, TieredMoeEngine, TieredMoeWorkspace,
        };
        use crate::cuda::{f16_to_f32, f32_to_f16};

        let _device = CudaDevice::set_current(0).unwrap();
        let stream = CudaStream::new(0).unwrap();

        let num_experts = 16usize;
        let n_interm = 640usize;
        let k_hidden = 128usize;
        let group_size = 128usize;
        let top_k = 4usize;
        let num_tokens = 6usize;
        let layout = ExpertPackLayout::new(n_interm, k_hidden, group_size).unwrap();

        // Build deterministic synthetic U4B8 weights and scales for 16 experts.
        let gu_words_per_exp = (2 * n_interm) * (k_hidden / 8);
        let gu_scales_per_exp = (2 * n_interm) * (k_hidden / group_size);
        let dn_words_per_exp = k_hidden * (n_interm / 8);
        let dn_scales_per_exp = k_hidden * (n_interm / group_size);

        let mut raw_packs = vec![0u8; num_experts * layout.stride];
        for e in 0..num_experts {
            let gu_p: Vec<u32> = (0..gu_words_per_exp)
                .map(|i| {
                    let mut w = 0u32;
                    for b in 0..8 {
                        let nib = (((e * 13 + i * 7 + b * 3) % 15) as u32) & 0xF;
                        w |= nib << (b * 4);
                    }
                    w
                })
                .collect();
            let gu_s: Vec<u16> = (0..gu_scales_per_exp)
                .map(|i| f32_to_f16(0.015 + 0.002 * (((e * 5 + i) % 9) as f32)))
                .collect();
            let dn_p: Vec<u32> = (0..dn_words_per_exp)
                .map(|i| {
                    let mut w = 0u32;
                    for b in 0..8 {
                        let nib = (((e * 11 + i * 5 + b * 7) % 15) as u32) & 0xF;
                        w |= nib << (b * 4);
                    }
                    w
                })
                .collect();
            let dn_s: Vec<u16> = (0..dn_scales_per_exp)
                .map(|i| f32_to_f16(0.012 + 0.0015 * (((e * 3 + i) % 11) as f32)))
                .collect();
            layout
                .write_pack(
                    &gu_p,
                    &gu_s,
                    &dn_p,
                    &dn_s,
                    &mut raw_packs[e * layout.stride..(e + 1) * layout.stride],
                )
                .unwrap();
        }

        let h_x: Vec<u16> = (0..num_tokens * k_hidden)
            .map(|i| f32_to_f16(((i as f32 * 0.13).sin()) * 0.4))
            .collect();
        let h_shared_out: Vec<u16> = (0..num_tokens * k_hidden)
            .map(|i| f32_to_f16(((i as f32 * 0.07).cos()) * 0.25))
            .collect();
        // Split `shared_out` into two additive partial sums to verify TP=2 shared expert fusion!
        let h_shared_out_r0: Vec<u16> = h_shared_out
            .iter()
            .enumerate()
            .map(|(i, &v)| f32_to_f16(f16_to_f32(v) * if i % 2 == 0 { 0.6 } else { 0.35 }))
            .collect();
        let h_shared_out_r1: Vec<u16> = h_shared_out
            .iter()
            .zip(&h_shared_out_r0)
            .map(|(&full, &r0)| f32_to_f16(f16_to_f32(full) - f16_to_f32(r0)))
            .collect();
        let h_shared_gate: Vec<u16> = (0..num_tokens)
            .map(|t| f32_to_f16(-0.3 + 0.25 * (t as f32)))
            .collect();

        // Route 6 tokens x top_k=4 across 10 distinct experts (`0, 1, 4, 5, 8, 9, 10, 11, 12, 14`),
        // leaving `2, 3, 6, 7, 13, 15` inactive in this batch so LRU eviction of inactive slots is exercised.
        let h_topk_ids: Vec<i32> = vec![
            0, 4, 8, 11, // tok 0
            1, 4, 9, 14, // tok 1
            0, 5, 10, 12, // tok 2
            1, 5, 8, 12, // tok 3
            0, 4, 11, 14, // tok 4
            4, 5, 9, 10, // tok 5
        ];
        let h_topk_weights: Vec<f32> = (0..num_tokens * top_k)
            .map(|i| 0.1 + 0.05 * ((i % 4) as f32))
            .collect();

        let d_x = upload(&h_x, &stream);
        let d_shared_out = upload(&h_shared_out, &stream);
        let d_shared_out_r0 = upload(&h_shared_out_r0, &stream);
        let d_shared_out_r1 = upload(&h_shared_out_r1, &stream);
        let d_shared_gate = upload(&h_shared_gate, &stream);
        let d_topk_ids = upload(&h_topk_ids, &stream);
        let d_topk_weights = upload(&h_topk_weights, &stream);

        // 1. Compute full-resident GPU oracle output (`K_hot = 16` all resident).
        let full_store = HostExpertStore::from_raw_packs(
            layout.clone(),
            0..num_experts,
            &raw_packs,
            num_experts * layout.stride,
        )
        .unwrap();
        let mut ws_oracle =
            TieredMoeWorkspace::new(0, layout.clone(), 8, num_tokens, top_k).unwrap();
        let mut full_cache =
            HotExpertCache::new(0, num_experts, 0..num_experts, layout.clone()).unwrap();
        full_cache
            .populate_initial(&full_store, None, &mut ws_oracle, &stream)
            .unwrap();
        let d_oracle_out =
            DeviceBuffer::alloc(0, num_tokens * k_hidden * std::mem::size_of::<u16>()).unwrap();
        TieredMoeEngine::execute_layer(
            MoeOffloadMode::GpuLru,
            &full_store,
            &mut full_cache,
            &mut ws_oracle,
            &stream,
            d_x.addr(),
            d_topk_ids.addr(),
            d_topk_weights.addr(),
            &h_topk_ids,
            &h_topk_weights,
            d_shared_out.addr(),
            d_shared_gate.addr(),
            d_oracle_out.addr(),
            num_tokens,
            top_k,
        )
        .unwrap();
        let mut h_oracle = vec![0u16; num_tokens * k_hidden];
        d_oracle_out.copy_to_host(&mut h_oracle, &stream).unwrap();

        let assert_close_f16 = |actual: &[u16], label: &str| {
            let max_diff = actual
                .iter()
                .zip(&h_oracle)
                .map(|(&a, &b)| (f16_to_f32(a) - f16_to_f32(b)).abs())
                .fold(0.0f32, f32::max);
            assert!(
                max_diff <= 1.6e-2,
                "{label} max_abs_diff={max_diff} exceeds 1.6e-2 (<= 4 FP16 ULPs)"
            );
        };

        // 2. Verify TP=1 `CpuHybrid { gpu_pcie_budget: 0 }` and `CpuHybrid { gpu_pcie_budget: 2 }`
        //    with only `k_hot = 4` experts initially cached in VRAM!
        for budget in [0usize, 2usize] {
            let mut cache = HotExpertCache::new(0, 4, 0..num_experts, layout.clone()).unwrap();
            let mut ws = TieredMoeWorkspace::new(0, layout.clone(), 3, num_tokens, top_k).unwrap();
            cache
                .populate_initial(&full_store, Some(&[0, 1, 2, 3]), &mut ws, &stream)
                .unwrap();
            let d_out =
                DeviceBuffer::alloc(0, num_tokens * k_hidden * std::mem::size_of::<u16>()).unwrap();
            TieredMoeEngine::execute_layer(
                MoeOffloadMode::CpuHybrid {
                    gpu_pcie_budget: budget,
                },
                &full_store,
                &mut cache,
                &mut ws,
                &stream,
                d_x.addr(),
                d_topk_ids.addr(),
                d_topk_weights.addr(),
                &h_topk_ids,
                &h_topk_weights,
                d_shared_out.addr(),
                d_shared_gate.addr(),
                d_out.addr(),
                num_tokens,
                top_k,
            )
            .unwrap();
            let mut h_out = vec![0u16; num_tokens * k_hidden];
            d_out.copy_to_host(&mut h_out, &stream).unwrap();
            assert!(cache.cpu_cold_evals > 0);
            assert_eq!(cache.dma_experts_loaded, budget as u64);
            assert_close_f16(&h_out, &format!("TP=1 CpuHybrid(budget={budget})"));
        }

        // 3. Verify TP=1 `StreamedPrefill` with `k_hot = 4` and `stage_capacity = 3` (forces 4 cold chunks
        //    through the double-buffered `stage_tables[0..2]` pipeline!).
        {
            let mut cache = HotExpertCache::new(0, 4, 0..num_experts, layout.clone()).unwrap();
            let mut ws = TieredMoeWorkspace::new(0, layout.clone(), 3, num_tokens, top_k).unwrap();
            cache
                .populate_initial(&full_store, Some(&[0, 1, 2, 3]), &mut ws, &stream)
                .unwrap();
            let d_out =
                DeviceBuffer::alloc(0, num_tokens * k_hidden * std::mem::size_of::<u16>()).unwrap();
            TieredMoeEngine::execute_layer(
                MoeOffloadMode::StreamedPrefill,
                &full_store,
                &mut cache,
                &mut ws,
                &stream,
                d_x.addr(),
                d_topk_ids.addr(),
                d_topk_weights.addr(),
                &h_topk_ids,
                &h_topk_weights,
                d_shared_out.addr(),
                d_shared_gate.addr(),
                d_out.addr(),
                num_tokens,
                top_k,
            )
            .unwrap();
            let mut h_out = vec![0u16; num_tokens * k_hidden];
            d_out.copy_to_host(&mut h_out, &stream).unwrap();
            assert_eq!(cache.dma_experts_loaded, 8);
            assert_close_f16(&h_out, "TP=1 StreamedPrefill(stage_cap=3)");
        }

        // 4. Verify TP=1 `GpuLru` with `k_hot = 12` and `stage_capacity = 12` (8 hits + 2 misses -> 2 LRU evictions + 2 DMA loads).
        {
            let mut cache = HotExpertCache::new(0, 12, 0..num_experts, layout.clone()).unwrap();
            let mut ws = TieredMoeWorkspace::new(0, layout.clone(), 12, num_tokens, top_k).unwrap();
            cache
                .populate_initial(&full_store, None, &mut ws, &stream)
                .unwrap();
            let d_out =
                DeviceBuffer::alloc(0, num_tokens * k_hidden * std::mem::size_of::<u16>()).unwrap();
            TieredMoeEngine::execute_layer(
                MoeOffloadMode::GpuLru,
                &full_store,
                &mut cache,
                &mut ws,
                &stream,
                d_x.addr(),
                d_topk_ids.addr(),
                d_topk_weights.addr(),
                &h_topk_ids,
                &h_topk_weights,
                d_shared_out.addr(),
                d_shared_gate.addr(),
                d_out.addr(),
                num_tokens,
                top_k,
            )
            .unwrap();
            let mut h_out = vec![0u16; num_tokens * k_hidden];
            d_out.copy_to_host(&mut h_out, &stream).unwrap();
            assert_eq!(cache.dma_experts_loaded, 2);
            assert_eq!(cache.evictions, 2);
            assert_close_f16(&h_out, "TP=1 GpuLru(k_hot=12)");
        }

        // 4b. Verify pure device-side CUDA-Graph-safe `execute_layer_device_lru` with `k_hot = 12`
        //     (pulling the 2 missed experts over zero-copy PCIe from `full_store.mapped_dev_ptr()`).
        {
            let mut cache = HotExpertCache::new(0, 12, 0..num_experts, layout.clone()).unwrap();
            let mut ws = TieredMoeWorkspace::new(0, layout.clone(), 12, num_tokens, top_k).unwrap();
            cache
                .populate_initial(&full_store, None, &mut ws, &stream)
                .unwrap();
            let d_out =
                DeviceBuffer::alloc(0, num_tokens * k_hidden * std::mem::size_of::<u16>()).unwrap();
            TieredMoeEngine::execute_layer_device_lru(
                &full_store,
                &mut cache,
                &ws,
                &stream,
                d_x.addr(),
                d_topk_ids.addr(),
                d_topk_weights.addr(),
                d_shared_out.addr(),
                d_shared_gate.addr(),
                d_out.addr(),
                num_tokens,
                top_k,
            )
            .unwrap();
            let mut h_out = vec![0u16; num_tokens * k_hidden];
            d_out.copy_to_host(&mut h_out, &stream).unwrap();
            cache.sync_from_device_if_dirty(&stream).unwrap();
            assert!(cache.slot_of(12).is_some());
            assert!(cache.slot_of(14).is_some());
            assert_close_f16(&h_out, "TP=1 DeviceGpuLru(k_hot=12)");
        }

        // 5. Verify TP=2 / EP=2 (`Rank 0: 0..8`, `Rank 1: 8..16`) across `GpuLru`, `CpuHybrid`, and `StreamedPrefill`,
        //    including single-AllReduce fusion of EP=2 routed experts + TP=2 sharded shared_expert!
        let store_r0 = HostExpertStore::from_raw_packs(
            layout.clone(),
            0..8,
            &raw_packs[0..8 * layout.stride],
            8 * layout.stride,
        )
        .unwrap();
        let store_r1 = HostExpertStore::from_raw_packs(
            layout.clone(),
            8..16,
            &raw_packs[8 * layout.stride..16 * layout.stride],
            8 * layout.stride,
        )
        .unwrap();

        for mode in [
            MoeOffloadMode::GpuLru,
            MoeOffloadMode::CpuHybrid { gpu_pcie_budget: 1 },
            MoeOffloadMode::StreamedPrefill,
        ] {
            let k_hot_ep = if matches!(mode, MoeOffloadMode::GpuLru) {
                8
            } else {
                3
            };
            let mut cache_r0 = HotExpertCache::new(0, k_hot_ep, 0..8, layout.clone()).unwrap();
            let mut cache_r1 = HotExpertCache::new(0, k_hot_ep, 8..16, layout.clone()).unwrap();
            let mut ws_r0 =
                TieredMoeWorkspace::new(0, layout.clone(), 4, num_tokens, top_k).unwrap();
            let mut ws_r1 =
                TieredMoeWorkspace::new(0, layout.clone(), 4, num_tokens, top_k).unwrap();
            cache_r0
                .populate_initial(&store_r0, None, &mut ws_r0, &stream)
                .unwrap();
            cache_r1
                .populate_initial(&store_r1, None, &mut ws_r1, &stream)
                .unwrap();

            let d_out_r0 =
                DeviceBuffer::alloc(0, num_tokens * k_hidden * std::mem::size_of::<u16>()).unwrap();
            let d_out_r1 =
                DeviceBuffer::alloc(0, num_tokens * k_hidden * std::mem::size_of::<u16>()).unwrap();

            TieredMoeEngine::execute_layer(
                mode,
                &store_r0,
                &mut cache_r0,
                &mut ws_r0,
                &stream,
                d_x.addr(),
                d_topk_ids.addr(),
                d_topk_weights.addr(),
                &h_topk_ids,
                &h_topk_weights,
                d_shared_out_r0.addr(),
                d_shared_gate.addr(),
                d_out_r0.addr(),
                num_tokens,
                top_k,
            )
            .unwrap();
            TieredMoeEngine::execute_layer(
                mode,
                &store_r1,
                &mut cache_r1,
                &mut ws_r1,
                &stream,
                d_x.addr(),
                d_topk_ids.addr(),
                d_topk_weights.addr(),
                &h_topk_ids,
                &h_topk_weights,
                d_shared_out_r1.addr(),
                d_shared_gate.addr(),
                d_out_r1.addr(),
                num_tokens,
                top_k,
            )
            .unwrap();

            // Simulate the post-MoE TP=2 AllReduce (`out_r0 += out_r1`).
            let ex = stream.exec(0, 0);
            crate::cuda::check_rc(
                // SAFETY: `d_out_r0` and `d_out_r1` are disjoint 4-byte-aligned FP16 buffers.
                unsafe {
                    ffi::w34_add_inplace_f16(
                        &ex,
                        d_out_r0.addr() as *mut _,
                        d_out_r1.addr() as *const _,
                        (num_tokens * k_hidden) as i64,
                    )
                },
                "w34_add_inplace_f16 (EP=2 sum)",
            )
            .unwrap();

            let mut h_ep2 = vec![0u16; num_tokens * k_hidden];
            d_out_r0.copy_to_host(&mut h_ep2, &stream).unwrap();
            assert_close_f16(&h_ep2, &format!("TP=2/EP=2 {mode:?}"));
        }

        // 6. Verify `M = 4` (`num_tokens * top_k = 16 <= num_experts = 16`, `regime == 1` `moe_sparse_vec4_kernel`)
        //    with non-power-of-two `n_interm = 640` (`k_u4_per_row = 20`) against CPU reference.
        {
            let m4 = 4usize;
            let mut tasks = Vec::with_capacity(m4 * top_k);
            for tok in 0..m4 {
                for k in 0..top_k {
                    let idx = tok * top_k + k;
                    tasks.push((tok, h_topk_ids[idx] as usize, h_topk_weights[idx]));
                }
            }
            let h_cpu_m4 = full_store.eval_cold_tasks_cpu(&h_x[..m4 * k_hidden], m4, &tasks);
            let d_out_m4 =
                DeviceBuffer::alloc(0, m4 * k_hidden * std::mem::size_of::<u16>()).unwrap();
            TieredMoeEngine::execute_layer_device_lru(
                &full_store,
                &mut full_cache,
                &ws_oracle,
                &stream,
                d_x.addr(),
                d_topk_ids.addr(),
                d_topk_weights.addr(),
                0,
                0,
                d_out_m4.addr(),
                m4,
                top_k,
            )
            .unwrap();
            let mut h_gpu_m4 = vec![0u16; m4 * k_hidden];
            d_out_m4.copy_to_host(&mut h_gpu_m4, &stream).unwrap();
            let max_diff = h_gpu_m4
                .iter()
                .zip(&h_cpu_m4)
                .map(|(&a, &b)| (f16_to_f32(a) - f16_to_f32(b)).abs())
                .fold(0.0f32, f32::max);
            assert!(
                max_diff <= 1.6e-2,
                "M=4 sparse_vec4 (n_interm=640) max_abs_diff={max_diff} exceeds 1.6e-2"
            );
        }
        let _ = DeviceExpertSlotTable::alloc;
    }
}
