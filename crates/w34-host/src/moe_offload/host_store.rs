use super::*;
use crate::cuda::MappedHostBuffer;
use crate::loader::{QuantScheme, SafetensorsArchive};
use std::io::Result;

pub(super) fn require(valid: bool, message: impl Into<String>) -> Result<()> {
    if valid {
        Ok(())
    } else {
        Err(io::Error::new(io::ErrorKind::InvalidData, message.into()))
    }
}

/// Immutable pinned backing for one layer's locally owned, complete experts.
/// EP partitions expert IDs, never the intermediate dimension. Retain this
/// store until every stream reading its mapped address or DMA source completes.
static TOTAL_PINNED_BYTES: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);

pub(super) enum HostPackStorage {
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
        const MAX_PROCESS_PINNED_BYTES: usize = 80 * 1024 * 1024 * 1024;
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

#[derive(Clone)]
pub struct HostExpertStore {
    storage: std::sync::Arc<HostPackStorage>,
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
        Self::load_with_pin(
            archive,
            experts_prefix,
            quant,
            layout,
            num_experts,
            ep_rank,
            ep_size,
            max_pinned_bytes,
            true,
        )
    }

    /// Load only this EP rank's experts, optionally using mapped pinned RAM (`prefer_mapped`).
    #[allow(clippy::too_many_arguments)]
    pub fn load_with_pin(
        archive: &SafetensorsArchive,
        experts_prefix: &str,
        quant: &QuantScheme,
        layout: ExpertPackLayout,
        num_experts: usize,
        ep_rank: usize,
        ep_size: usize,
        max_pinned_bytes: usize,
        prefer_mapped: bool,
    ) -> Result<Self> {
        require(
            matches!(ep_size, 1 | 2 | 4)
                && ep_rank < ep_size
                && num_experts > 0
                && num_experts.is_multiple_of(ep_size),
            "invalid expert parallel partition",
        )?;
        // Revalidate public geometry before using it to size or address unsafe storage.
        require(
            layout
                == ExpertPackLayout::new(layout.intermediate, layout.hidden, layout.group_size)?
                    .with_down_hadamard(layout.down_hadamard_block_size)?,
            "inconsistent expert pack layout",
        )?;
        let count = num_experts / ep_size;
        let expert_ids = ep_rank * count..(ep_rank + 1) * count;
        let first_id = expert_ids.start;
        let (resolved_prefix, is_compressed_tensors) = if let Some(real_wp) = archive.resolve_name(
            &format!("{experts_prefix}.{first_id}.gate_proj.weight_packed"),
        ) {
            let pfx = real_wp
                .strip_suffix(&format!(".{first_id}.gate_proj.weight_packed"))
                .unwrap_or(experts_prefix)
                .to_string();
            (pfx, true)
        } else if let Some(real_qw) =
            archive.resolve_name(&format!("{experts_prefix}.{first_id}.gate_proj.qweight"))
        {
            let pfx = real_qw
                .strip_suffix(&format!(".{first_id}.gate_proj.qweight"))
                .unwrap_or(experts_prefix)
                .to_string();
            (pfx, false)
        } else {
            (experts_prefix.to_string(), false)
        };
        let resolved_prefix_ref = resolved_prefix.as_str();
        let first_gate_name = format!("{resolved_prefix_ref}.{first_id}.gate_proj");
        let first_up_name = format!("{resolved_prefix_ref}.{first_id}.up_proj");
        let first_down_name = format!("{resolved_prefix_ref}.{first_id}.down_proj");
        let gate_had = quant.input_hadamard(&first_gate_name)?.unwrap_or(0);
        let up_had = quant.input_hadamard(&first_up_name)?.unwrap_or(0);
        let down_had = quant.input_hadamard(&first_down_name)?.unwrap_or(0);
        require(
            gate_had == up_had
                && crate::loader::is_valid_hadamard_block_for_dim(gate_had, layout.hidden),
            "inconsistent or unsupported input Hadamard block size on routed MoE gate_proj/up_proj",
        )?;
        require(
            crate::loader::is_valid_hadamard_block_for_dim(down_had, layout.intermediate)
                && (layout.down_hadamard_block_size == 0
                    || layout.down_hadamard_block_size == down_had),
            "inconsistent or unsupported input Hadamard block size on routed MoE down_proj",
        )?;
        let mut check_name = String::with_capacity(resolved_prefix_ref.len() + 24);
        for expert_id in 0..num_experts {
            use std::fmt::Write as _;
            check_name.clear();
            let _ = write!(
                &mut check_name,
                "{resolved_prefix_ref}.{expert_id}.gate_proj"
            );
            let e_gate = quant.input_hadamard(&check_name)?.unwrap_or(0);
            check_name.clear();
            let _ = write!(&mut check_name, "{resolved_prefix_ref}.{expert_id}.up_proj");
            let e_up = quant.input_hadamard(&check_name)?.unwrap_or(0);
            check_name.clear();
            let _ = write!(
                &mut check_name,
                "{resolved_prefix_ref}.{expert_id}.down_proj"
            );
            let e_down = quant.input_hadamard(&check_name)?.unwrap_or(0);
            require(
                e_gate == gate_had && e_up == up_had && e_down == down_had,
                "routed MoE experts within a layer must share uniform input Hadamard transforms",
            )?;
        }
        let layout = layout.with_down_hadamard(down_had)?;
        let expert_quant = quant
            .for_linear(&first_gate_name)
            .ok()
            .filter(|q| *q != QuantScheme::Unquantized)
            .unwrap_or_else(|| quant.clone());
        require(
            expert_quant.symmetric_gptq_u4_group_size() == Some(layout.group_size),
            "host expert store requires symmetric 4-bit GPTQ/AutoRound/CompressedTensors W4A16",
        )?;
        let bytes = layout
            .stride
            .checked_mul(count)
            .ok_or_else(|| io::Error::other("expert store size overflow"))?;
        require(
            bytes <= max_pinned_bytes,
            format!("expert store needs {bytes} pinned bytes, budget is {max_pinned_bytes}"),
        )?;
        let mut storage = HostPackStorage::alloc(bytes, prefer_mapped)?;
        let dst = storage.as_mut_slice();

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
            .map(|n| (n.get() / ep_size.max(1)).clamp(1, 16))
            .unwrap_or(8)
            .min(count.max(1));
        let experts_per_chunk = count.div_ceil(num_threads).max(1);
        let stride = layout.stride;
        let base_expert = expert_ids.start;
        let layout_ref = &layout;
        let gu_half_packed = layout.gate_up_packed.len() / 2;
        let gu_half_scales = layout.gate_up_scales.len() / 2;

        std::thread::scope(|s| -> Result<()> {
            let mut handles = Vec::new();
            for (chunk_idx, dst_chunk) in dst.chunks_mut(experts_per_chunk * stride).enumerate() {
                let local_start = chunk_idx * experts_per_chunk;
                let chunk_count = dst_chunk.len() / stride;
                handles.push(s.spawn(move || -> Result<()> {
                    let mut raw_qw_bytes =
                        vec![0u8; gu_half_packed.max(layout_ref.down_packed.len())];
                    let mut raw_sc_words =
                        vec![0u16; (gu_half_scales / 2).max(layout_ref.down_scales.len() / 2)];
                    let mut key_buf = String::with_capacity(96);
                    for i in 0..chunk_count {
                        let local_id = local_start + i;
                        let expert_id = base_expert + local_id;
                        let pack = &mut dst_chunk[i * stride..(i + 1) * stride];
                        pack[layout_ref.down_scales.end..stride].fill(0);
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
                            use std::fmt::Write as _;
                            let k8 = k_exp / 8;
                            let num_groups = k_exp / layout_ref.group_size;
                            if is_compressed_tensors {
                                key_buf.clear();
                                let _ = write!(
                                    key_buf,
                                    "{resolved_prefix_ref}.{expert_id}.{proj_name}.weight_packed"
                                );
                                let wp_info = archive.tensor_info(&key_buf)?;
                                key_buf.clear();
                                let _ = write!(
                                    key_buf,
                                    "{resolved_prefix_ref}.{expert_id}.{proj_name}.weight_scale"
                                );
                                let ws_info = archive.tensor_info(&key_buf).map_err(|_| {
                                    io::Error::other(format!(
                                        "missing {experts_prefix}.{expert_id}.{proj_name}.weight_scale"
                                    ))
                                })?;
                                let is_u2 = k8.is_multiple_of(2) && wp_info.shape == [n_exp, k8 / 2];
                                require(
                                    (wp_info.shape == [n_exp, k8] || is_u2)
                                        && ws_info.shape == [n_exp, num_groups],
                                    format!(
                                        "mismatched compressed-tensors weight_packed {:?} / weight_scale {:?} (2/4-bit) on '{experts_prefix}.{expert_id}.{proj_name}'",
                                        wp_info.shape, ws_info.shape
                                    ),
                                )?;
                                if is_u2 {
                                    let u2_buf = &mut raw_qw_bytes[..w_range.len() / 2];
                                    archive.read_exact_info(wp_info, u2_buf)?;
                                    let dst_w = &mut pack[w_range];
                                    for (dst_pair, src_word) in dst_w
                                        .as_chunks_mut::<8>()
                                        .0
                                        .iter_mut()
                                        .zip(u2_buf.as_chunks::<4>().0.iter())
                                    {
                                        let w = u32::from_le_bytes(*src_word);
                                        for (half_idx, half) in
                                            [w & 0xFFFF, w >> 16].into_iter().enumerate()
                                        {
                                            let mut x = half;
                                            x = (x | (x << 8)) & 0x00FF_00FF;
                                            x = (x | (x << 4)) & 0x0F0F_0F0F;
                                            x = (x | (x << 2)) & 0x3333_3333;
                                            x = x.wrapping_add(0x6666_6666);
                                            dst_pair[half_idx * 4..(half_idx + 1) * 4]
                                                .copy_from_slice(&x.to_le_bytes());
                                        }
                                    }
                                } else {
                                    // `weight_packed` is already N-major `[n_exp, k8]`.
                                    archive.read_exact_info(wp_info, &mut pack[w_range])?;
                                }
                                // `weight_scale` is already N-major `[n_exp, num_groups]`.
                                let sc_words = &mut raw_sc_words[..n_exp * num_groups];
                                archive.read_f16_bits_info_into(ws_info, sc_words)?;
                                let dst_s = &mut pack[s_range];
                                for (dst, &s) in
                                    dst_s.as_chunks_mut::<2>().0.iter_mut().zip(&*sc_words)
                                {
                                    dst.copy_from_slice(&s.to_le_bytes());
                                }
                            } else {
                                key_buf.clear();
                                let _ = write!(
                                    key_buf,
                                    "{resolved_prefix_ref}.{expert_id}.{proj_name}.qweight"
                                );
                                let qw_info = archive.tensor_info(&key_buf).map_err(|_| {
                                    io::Error::other(format!(
                                        "missing {experts_prefix}.{expert_id}.{proj_name}.qweight"
                                    ))
                                })?;
                                key_buf.clear();
                                let _ = write!(
                                    key_buf,
                                    "{resolved_prefix_ref}.{expert_id}.{proj_name}.scales"
                                );
                                let sc_info = archive.tensor_info(&key_buf).map_err(|_| {
                                    io::Error::other(format!(
                                        "missing {experts_prefix}.{expert_id}.{proj_name}.scales"
                                    ))
                                })?;
                                require(
                                    qw_info.shape == [k8, n_exp]
                                        && sc_info.shape == [num_groups, n_exp],
                                    format!(
                                        "CUDA error -10 in model loader error: mismatched GPTQ qweight {:?} / scales {:?} (4-bit) on '{experts_prefix}.{expert_id}.{proj_name}'",
                                        qw_info.shape, sc_info.shape
                                    ),
                                )?;
                                let qw_buf = &mut raw_qw_bytes[..w_range.len()];
                                archive.read_exact_info(qw_info, qw_buf)?;
                                let dst_w = &mut pack[w_range];
                                // Cache-blocked transpose `[k8, n_exp]` u32 words -> `[n_exp, k8]` u32 words
                                // with sequential destination writes into pinned `dst_w`.
                                const TILE: usize = 32;
                                let mut tile = [[[0u8; 4]; TILE]; TILE];
                                for col_base in (0..k8).step_by(TILE) {
                                    let col_len = (k8 - col_base).min(TILE);
                                    for row_base in (0..n_exp).step_by(TILE) {
                                        let row_len = (n_exp - row_base).min(TILE);
                                        for c in 0..col_len {
                                            let src_row = &qw_buf
                                                [(col_base + c) * n_exp * 4 + row_base * 4..];
                                            for (r, slot) in
                                                src_row.as_chunks::<4>().0.iter().enumerate().take(row_len)
                                            {
                                                tile[r][c] = *slot;
                                            }
                                        }
                                        for (r, tile_row) in tile.iter().enumerate().take(row_len) {
                                            let dst_row_off =
                                                ((row_base + r) * k8 + col_base) * 4;
                                            let dst_slice = &mut dst_w
                                                [dst_row_off..dst_row_off + col_len * 4];
                                            for (dst_word, src_word) in dst_slice
                                                .as_chunks_mut::<4>()
                                                .0
                                                .iter_mut()
                                                .zip(&tile_row[..col_len])
                                            {
                                                *dst_word = *src_word;
                                            }
                                        }
                                    }
                                }
                                let sc_words = &mut raw_sc_words[..num_groups * n_exp];
                                archive.read_f16_bits_info_into(sc_info, sc_words)?;
                                let dst_s = &mut pack[s_range];
                                for row in 0..n_exp {
                                    let dst_row =
                                        &mut dst_s[row * num_groups * 2..(row + 1) * num_groups * 2];
                                    for (g, dst_elem) in
                                        dst_row.as_chunks_mut::<2>().0.iter_mut().enumerate()
                                    {
                                        dst_elem.copy_from_slice(
                                            &sc_words[g * n_exp + row].to_le_bytes(),
                                        );
                                    }
                                }
                            }
                        }
                    }
                    Ok(())
                }));
            }
            for h in handles {
                h.join()
                    .map_err(|_| io::Error::other("HostExpertStore loader thread panicked"))??;
            }
            Ok(())
        })?;
        archive.drop_page_cache();
        Ok(Self {
            storage: std::sync::Arc::new(storage),
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
                == ExpertPackLayout::new(layout.intermediate, layout.hidden, layout.group_size)?
                    .with_down_hadamard(layout.down_hadamard_block_size)?,
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
            storage: std::sync::Arc::new(storage),
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
    pub fn raw_packs_ptr(&self) -> *const u8 {
        self.storage.as_slice().as_ptr()
    }

    /// Releases the host backing buffer once all local experts are permanently resident in VRAM.
    pub fn release_host_storage(&mut self) {
        self.storage = std::sync::Arc::new(HostPackStorage::Heap(Vec::new()));
    }

    /// Read-only bytes, also suitable as a source for a synchronized H2D copy.
    pub fn expert_pack(&self, global_id: usize) -> Option<&[u8]> {
        if !self.expert_ids.contains(&global_id) {
            return None;
        }
        let offset = (global_id - self.expert_ids.start) * self.layout.stride;
        let slice = self.storage.as_slice();
        if offset.saturating_add(self.layout.stride) > slice.len() {
            return None;
        }
        Some(&slice[offset..offset + self.layout.stride])
    }

    /// Evaluates a single locally owned expert on CPU and accumulates into `out_accum_f32`.
    #[cfg(test)]
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
    #[cfg(test)]
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

    /// Pulls `num_staged` experts directly from zero-copy mapped host memory (`mapped_packs_addr`)
    /// at `src_local_experts_addr` into `dst_slots_addr` on `stream`.
    pub fn pull_mapped(
        &self,
        stream: &crate::cuda::CudaStream,
        mapped_packs_addr: u64,
        src_local_experts_addr: u64,
        dst_slots_addr: u64,
        num_staged: usize,
        local_num_experts: usize,
    ) -> crate::cuda::Result<()> {
        if num_staged == 0 {
            return Ok(());
        }
        let ex = stream.exec(0, 0);
        crate::cuda::check_rc(
            // SAFETY: device pointers and capacities are owned and validated by `self` and caller.
            unsafe {
                w34_kernels::ffi::w34_moe_pull_mapped_experts(
                    &ex,
                    mapped_packs_addr as *const core::ffi::c_void,
                    src_local_experts_addr as *const i32,
                    dst_slots_addr as *const i32,
                    self.gate_up_packed.addr() as *mut u32,
                    self.gate_up_scales.addr() as *mut core::ffi::c_void,
                    self.down_packed.addr() as *mut u32,
                    self.down_scales.addr() as *mut core::ffi::c_void,
                    num_staged as i32,
                    local_num_experts as i32,
                    self.capacity as i32,
                    self.layout.stride,
                    self.layout.intermediate as i32,
                    self.layout.hidden as i32,
                    self.layout.group_size as i32,
                )
            },
            "w34_moe_pull_mapped_experts",
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
                w34_kernels::ffi::w34_moe_w4a16_mlp_ex_f16(
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
                    self.layout.down_hadamard_block_size as i32,
                )
            },
            "w34_moe_w4a16_mlp_ex_f16 (tiered)",
        )
    }
}

/// Per-layer VRAM hot expert cache (`K_hot` slots) with LRU eviction tracking and telemetry.
pub struct HotExpertCache {
    pub slots: DeviceExpertSlotTable,
    pub expert_ids: Range<usize>,
    pub expert_to_slot: Vec<i32>,
    pub prefill_expert_to_slot: Vec<i32>,
    /// Global expert ID resident in each VRAM slot (`-1` if empty).
    pub slot_to_expert: Vec<i32>,
    pub slot_last_used: Vec<u64>,
    pub d_expert_to_slot: crate::cuda::DeviceBuffer,
    pub d_prefill_expert_to_slot: crate::cuda::DeviceBuffer,
    pub d_slot_to_expert: crate::cuda::DeviceBuffer,
    pub d_slot_last_used: crate::cuda::DeviceBuffer,
    pub d_step_counter: crate::cuda::DeviceBuffer,
    pub(crate) bridge_runtime: Option<std::sync::Arc<crate::moe_cpu_bridge::Runtime>>,
    pub(crate) bridge_mailbox: Option<std::sync::Arc<crate::moe_cpu_bridge::Mailbox>>,
    pub k_static: usize,
    pub initial_slot_to_expert: Vec<i32>,
    pub adapted_slots_dirty: bool,
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
        let d_expert_to_slot =
            crate::cuda::DeviceBuffer::alloc(device_id, local_count * core::mem::size_of::<i32>())?;
        let d_prefill_expert_to_slot =
            crate::cuda::DeviceBuffer::alloc(device_id, local_count * core::mem::size_of::<i32>())?;
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
            prefill_expert_to_slot: vec![-1; local_count],
            slot_to_expert: vec![-1; k_hot],
            slot_last_used: vec![0; k_hot],
            d_expert_to_slot,
            d_prefill_expert_to_slot,
            d_slot_to_expert,
            d_slot_last_used,
            d_step_counter,
            bridge_runtime: None,
            bridge_mailbox: None,
            k_static: k_hot,
            initial_slot_to_expert: Vec::new(),
            adapted_slots_dirty: false,
            device_lru_dirty: false,
            step_clock: 1,
            hits: 0,
            misses: 0,
            evictions: 0,
            dma_experts_loaded: 0,
            cpu_cold_evals: 0,
        })
    }

    /// Whether this cache has a registered decode/verify CPU mailbox (`TP=1` hybrid bridge).
    #[inline]
    pub fn has_cpu_bridge(&self) -> bool {
        self.bridge_mailbox.is_some()
    }

    /// Uploads the current host LRU metadata (`expert_to_slot`, `prefill_expert_to_slot`,
    /// `slot_to_expert`, `slot_last_used`, and `step_clock`) to device buffers on `stream`.
    pub fn upload_lru_metadata(
        &mut self,
        stream: &crate::cuda::CudaStream,
    ) -> crate::cuda::Result<()> {
        self.d_expert_to_slot
            .copy_from_host(&self.expert_to_slot, stream)?;
        self.sync_prefill_expert_to_slot(stream)?;
        self.d_slot_to_expert
            .copy_from_host(&self.slot_to_expert, stream)?;
        self.d_slot_last_used
            .copy_from_host(&self.slot_last_used, stream)?;
        self.d_step_counter
            .copy_from_host(std::slice::from_ref(&self.step_clock), stream)
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
        self.initial_slot_to_expert = chosen.iter().map(|&ge| ge as i32).collect();
        self.adapted_slots_dirty = false;
        self.expert_to_slot.fill(-1);
        for (slot, &ge) in chosen.iter().enumerate() {
            let le = ge - self.expert_ids.start;
            self.expert_to_slot[le] = slot as i32;
            self.slot_to_expert[slot] = ge as i32;
            // Preserve ranking order so the least-hot initial slot (`k_hot - 1`) is evicted
            // before the hottest initial slot (`0`).
            self.slot_last_used[slot] = (k_hot - slot) as u64;
        }
        self.step_clock = (k_hot as u64) + 1;

        // Upload in chunks of `ws.stage_capacity`.
        let chunk_cap = ws.stage_capacity;
        for chunk_start in (0..k_hot).step_by(chunk_cap) {
            let chunk_end = (chunk_start + chunk_cap).min(k_hot);
            ws.stage_experts_to_slots(
                0,
                store,
                &chosen[chunk_start..chunk_end],
                &(chunk_start as i32..chunk_end as i32).collect::<Vec<_>>(),
                &self.slots,
                stream,
            )?;
        }
        self.upload_lru_metadata(stream)?;
        self.device_lru_dirty = false;
        stream.synchronize()
    }

    #[inline]
    pub fn prefill_hot_limit(&self) -> usize {
        if self.slots.capacity >= self.expert_ids.len() {
            self.slots.capacity
        } else if self.has_cpu_bridge() {
            self.k_static
        } else if self.slots.capacity > TIERED_MOE_GPU_LRU_TAIL_SLOTS {
            self.slots.capacity - TIERED_MOE_GPU_LRU_TAIL_SLOTS
        } else {
            self.slots.capacity
        }
    }

    pub fn sync_prefill_expert_to_slot(
        &mut self,
        stream: &crate::cuda::CudaStream,
    ) -> crate::cuda::Result<()> {
        let prefill_cap = self.prefill_hot_limit();
        for (dst, &s) in self
            .prefill_expert_to_slot
            .iter_mut()
            .zip(&self.expert_to_slot)
        {
            *dst = if s >= 0 && (s as usize) < prefill_cap {
                s
            } else {
                -1
            };
        }
        self.d_prefill_expert_to_slot
            .copy_from_host(&self.prefill_expert_to_slot, stream)
    }

    #[inline]
    pub fn cpu_bridge_supported() -> bool {
        crate::moe_cpu_bridge::supported()
    }

    /// Whether `RankTieredMoe::load_for_rank` should enable the hybrid CPU+GPU MoE bridge
    /// by default (`avx2 + fma + f16c` or `avx512` on `x86_64`).
    #[inline]
    pub fn auto_cpu_bridge_preferred() -> bool {
        Self::cpu_bridge_supported()
    }

    /// Registers this layer with the owned Rust SIMD CPU+GPU hybrid cold-expert bridge and reserves the top `k_hot - k_static`
    /// slots (`k_static..k_hot`) as transient GPU PCIe staging slots while keeping
    /// slots `0..k_static` permanently resident in VRAM.
    pub fn register_cpu_bridge(
        &mut self,
        store: &HostExpertStore,
        k_static: usize,
        stream: &crate::cuda::CudaStream,
    ) -> crate::cuda::Result<()> {
        let k_hot = self.slots.capacity;
        let k_static = k_static.clamp(1, k_hot);
        if !Self::cpu_bridge_supported() {
            return Ok(());
        }
        if !store.layout.supports_cpu_bridge() {
            return Err(crate::cuda::CudaError::new(
                -1,
                "CPU MoE bridge requires hidden <= 4096 and % 128 == 0, intermediate <= 2048 and % 128 == 0, group size 64 or 128, and down Hadamard 0 or 64",
            ));
        }
        if self.bridge_runtime.is_none() {
            self.bridge_runtime = Some(crate::moe_cpu_bridge::Runtime::new(stream)?);
        }
        let runtime = self.bridge_runtime.as_ref().unwrap();
        self.bridge_mailbox = Some(runtime.register(store)?);
        self.k_static = k_static;
        self.sync_prefill_expert_to_slot(stream)?;
        stream.synchronize()?;
        Ok(())
    }

    /// Pulls updated LRU metadata from device before host-side `execute_layer` runs.
    pub fn sync_from_device_if_dirty(
        &mut self,
        stream: &crate::cuda::CudaStream,
    ) -> crate::cuda::Result<()> {
        if !self.device_lru_dirty {
            return Ok(());
        }
        let mut step_arr = [0u64; 1];
        self.d_expert_to_slot
            .copy_to_host(&mut self.expert_to_slot, stream)?;
        self.d_slot_to_expert
            .copy_to_host(&mut self.slot_to_expert, stream)?;
        self.d_slot_last_used
            .copy_to_host(&mut self.slot_last_used, stream)?;
        self.d_step_counter.copy_to_host(&mut step_arr, stream)?;
        stream.synchronize()?;
        self.step_clock = step_arr[0].saturating_add(1).max(1);
        let check_limit = if self.has_cpu_bridge() {
            self.k_static
        } else {
            self.prefill_hot_limit()
        }
        .min(self.initial_slot_to_expert.len());
        if (0..check_limit).any(|s| self.slot_to_expert[s] != self.initial_slot_to_expert[s]) {
            self.adapted_slots_dirty = true;
        }
        self.sync_prefill_expert_to_slot(stream)?;
        stream.synchronize()?;
        self.device_lru_dirty = false;
        Ok(())
    }

    /// Restores any static hot slots (`0..prefill_hot_limit()`) that were temporarily adapted for a
    /// long prompt (`num_tokens >= 1024`) or mutated by device-side LRU eviction back to
    /// `initial_slot_to_expert` when a subsequent normal/short prompt (`num_tokens < 1024`) arrives.
    pub fn restore_initial_if_adapted(
        &mut self,
        store: &HostExpertStore,
        ws: &mut TieredMoeWorkspace,
        stream: &crate::cuda::CudaStream,
    ) -> crate::cuda::Result<()> {
        if !self.adapted_slots_dirty {
            return Ok(());
        }
        let restore_limit = if self.k_static > 0 {
            self.k_static.min(self.initial_slot_to_expert.len())
        } else {
            self.slots.capacity.min(self.initial_slot_to_expert.len())
        };
        let local_start = self.expert_ids.start;
        let mut restore_ges = Vec::new();
        let mut restore_slots = Vec::new();
        for slot in 0..restore_limit {
            let init_ge_i32 = self.initial_slot_to_expert[slot];
            if self.slot_to_expert[slot] != init_ge_i32 {
                let old_ge_i32 = self.slot_to_expert[slot];
                if old_ge_i32 >= 0 {
                    let old_ge = old_ge_i32 as usize;
                    if self.expert_ids.contains(&old_ge) {
                        let old_le = old_ge - local_start;
                        if self.expert_to_slot[old_le] == slot as i32 {
                            self.expert_to_slot[old_le] = -1;
                        }
                    }
                }
                restore_ges.push(init_ge_i32 as usize);
                restore_slots.push(slot as i32);
            }
        }
        if !restore_ges.is_empty() {
            let k_hot = self.slots.capacity;
            for (&init_ge, &slot_i32) in restore_ges.iter().zip(&restore_slots) {
                let slot = slot_i32 as usize;
                let prev_slot = self.expert_to_slot[init_ge - local_start];
                if prev_slot >= 0 && (prev_slot as usize) != slot {
                    let ps = prev_slot as usize;
                    self.slot_to_expert[ps] = -1;
                    self.slot_last_used[ps] = 0;
                }
                self.expert_to_slot[init_ge - local_start] = slot_i32;
                self.slot_to_expert[slot] = init_ge as i32;
            }
            for slot in 0..k_hot {
                if self.slot_to_expert[slot] >= 0 {
                    self.slot_last_used[slot] = self.step_clock + (k_hot - slot) as u64;
                } else {
                    self.slot_last_used[slot] = 0;
                }
            }
            self.step_clock += (k_hot as u64) + 1;
            self.dma_experts_loaded += restore_ges.len() as u64;
            for (g_chunk, s_chunk) in restore_ges
                .chunks(ws.stage_capacity)
                .zip(restore_slots.chunks(ws.stage_capacity))
            {
                ws.stage_experts_to_slots(0, store, g_chunk, s_chunk, &self.slots, stream)?;
            }
            ws.compute_done_events[0].record(stream)?;
            self.upload_lru_metadata(stream)?;
            stream.synchronize()?;
        }
        self.adapted_slots_dirty = false;
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
    pub d_stage_src_local_experts: [crate::cuda::DeviceBuffer; 2],
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
    pub h_prefill_x: MappedHostBuffer,
    pub h_prefill_out: MappedHostBuffer,
    pub d2h_done_event: crate::cuda::CudaEvent,
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
        let d_stage_src_local_experts = [
            DeviceBuffer::alloc(device_id, stage_cap * core::mem::size_of::<i32>())?,
            DeviceBuffer::alloc(device_id, stage_cap * core::mem::size_of::<i32>())?,
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
        let identity_slots: Vec<i32> = (0..stage_cap as i32).collect();
        d_stage_slots[0].copy_from_host(&identity_slots, &h2d_stream)?;
        d_stage_slots[1].copy_from_host(&identity_slots, &h2d_stream)?;
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
        let max_misses = 512usize;
        let d_miss_local_experts =
            DeviceBuffer::alloc(device_id, max_misses * core::mem::size_of::<i32>())?;
        let d_miss_victim_slots =
            DeviceBuffer::alloc(device_id, max_misses * core::mem::size_of::<i32>())?;
        let d_num_misses = DeviceBuffer::alloc(device_id, 2 * core::mem::size_of::<i32>())?;
        let prefill_buf_bytes = max_tokens.max(1) * layout.hidden * core::mem::size_of::<u16>();
        let h_prefill_x = MappedHostBuffer::alloc(prefill_buf_bytes)?;
        let h_prefill_out = MappedHostBuffer::alloc(prefill_buf_bytes)?;
        let d2h_done_event = CudaEvent::new(device_id, false)?;
        h2d_stream.synchronize()?;

        Ok(Self {
            device_id,
            layout,
            stage_capacity: stage_cap,
            max_tokens,
            top_k,
            h_stage_packs,
            d_stage_packs,
            d_stage_src_local_experts,
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
            h_prefill_x,
            h_prefill_out,
            d2h_done_event,
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
        let stride = self.layout.stride;
        if store.mapped_dev_ptr() != 0 {
            let start = store.expert_ids().start;
            let local_ids: Vec<i32> = global_expert_ids
                .iter()
                .map(|&ge| (ge - start) as i32)
                .collect();
            self.d_stage_src_local_experts[buf_idx].copy_from_host(&local_ids, stream)?;
            self.d_miss_victim_slots.copy_from_host(dst_slots, stream)?;
            dst_table.pull_mapped(
                stream,
                store.mapped_dev_ptr(),
                self.d_stage_src_local_experts[buf_idx].addr(),
                self.d_miss_victim_slots.addr(),
                n,
                store.expert_ids().len(),
            )?;
            return self.compute_done_events[buf_idx].record(stream);
        }
        self.compute_done_events[buf_idx].synchronize()?;
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
        self.d_miss_victim_slots.copy_from_host(dst_slots, stream)?;
        dst_table.unpack_staged(
            stream,
            self.d_stage_packs[buf_idx].addr(),
            self.d_miss_victim_slots.addr(),
            n,
        )?;
        self.compute_done_events[buf_idx].record(stream)
    }

    /// Enqueues asynchronous H2D DMA of `global_expert_ids` into `d_stage_packs[buf_idx]`
    /// and `d_stage_slots[buf_idx]` on `h2d_stream` using the hardware PCIe Copy Engine
    /// (without touching `dst_table` or GPU SMs, so it overlaps 100% with `compute_stream`).
    pub fn stage_experts_h2d_dma(
        &self,
        buf_idx: usize,
        store: &HostExpertStore,
        global_expert_ids: &[usize],
        dst_slots: &[i32],
        h2d_stream: &crate::cuda::CudaStream,
    ) -> crate::cuda::Result<()> {
        let n = global_expert_ids.len();
        if n == 0 {
            return Ok(());
        }
        assert_eq!(n, dst_slots.len());
        assert!(n <= self.stage_capacity);
        let stride = self.layout.stride;
        if store.mapped_dev_ptr() != 0 {
            debug_assert!(
                dst_slots.iter().copied().eq(0..n as i32),
                "mapped stage_experts_h2d_dma requires contiguous identity dst_slots 0..n"
            );
            let mut i = 0;
            while i < n {
                let mut run_len = 1;
                while i + run_len < n
                    && global_expert_ids[i + run_len] == global_expert_ids[i] + run_len
                {
                    run_len += 1;
                }
                let src_ptr = store
                    .expert_pack(global_expert_ids[i])
                    .expect("expert not owned by local HostExpertStore")
                    .as_ptr();
                let dst_addr = self.d_stage_packs[buf_idx].addr() + ((i * stride) as u64);
                crate::cuda::check_rc(
                    // SAFETY: `store` is pinned host memory and `d_stage_packs[buf_idx]` fits `n * stride` bytes.
                    unsafe {
                        w34_kernels::ffi::w34_memcpy_h2d_async(
                            dst_addr as *mut core::ffi::c_void,
                            src_ptr.cast::<core::ffi::c_void>(),
                            run_len * stride,
                            h2d_stream.handle,
                        )
                    },
                    "w34_memcpy_h2d_async (stage_experts_h2d_dma)",
                )?;
                i += run_len;
            }
            return Ok(());
        }
        self.compute_done_events[buf_idx].synchronize()?;
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
        self.d_stage_packs[buf_idx].copy_from_host(pinned_dst, h2d_stream)?;
        self.d_stage_slots[buf_idx].copy_from_host(dst_slots, h2d_stream)
    }
}
