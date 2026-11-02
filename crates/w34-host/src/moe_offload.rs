//! Tiered routed-expert storage and execution for Qwen4Exp.

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
}

#[cfg(feature = "cuda")]
mod host_store {
    use super::*;
    use crate::cuda::MappedHostBuffer;
    use crate::loader::{QuantScheme, SafetensorsArchive};
    use crate::model::weights::RawLinearHost;
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
    pub struct HostExpertStore {
        storage: MappedHostBuffer,
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
                matches!(quant, QuantScheme::AutoRound { bits: 4, group_size, sym: true, awq_packing: false }
                if *group_size == layout.group_size),
                "host expert store requires symmetric GPTQ-packed AutoRound W4A16",
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
            let storage = MappedHostBuffer::alloc(bytes).map_err(io::Error::other)?;
            // SAFETY: newly allocated storage is exclusively owned and has no GPU readers.
            let dst =
                unsafe { std::slice::from_raw_parts_mut(storage.host_ptr.cast::<u8>(), bytes) };
            dst.fill(0);
            for (local_id, expert_id) in expert_ids.clone().enumerate() {
                let prefix = format!("{experts_prefix}.{expert_id}");
                // Symmetric GPTQ encodes zero=8 as 7. Reject incompatible tensors
                // rather than silently interpreting asymmetric weights as U4B8.
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
                let gate = RawLinearHost::load(archive, &format!("{prefix}.gate_proj"), quant)
                    .map_err(io::Error::other)?;
                let up = RawLinearHost::load(archive, &format!("{prefix}.up_proj"), quant)
                    .map_err(io::Error::other)?;
                let gu = RawLinearHost::concat_rows(&[gate, up]).map_err(io::Error::other)?;
                let down = RawLinearHost::load(archive, &format!("{prefix}.down_proj"), quant)
                    .map_err(io::Error::other)?;
                let pack = &mut dst[local_id * layout.stride..(local_id + 1) * layout.stride];
                for (raw, n_expected, k_expected, weights, scales_region) in [
                    (
                        gu,
                        2 * layout.intermediate,
                        layout.hidden,
                        &layout.gate_up_packed,
                        &layout.gate_up_scales,
                    ),
                    (
                        down,
                        layout.hidden,
                        layout.intermediate,
                        &layout.down_packed,
                        &layout.down_scales,
                    ),
                ] {
                    let RawLinearHost::U4B8NMajor {
                        packed,
                        scales,
                        n,
                        k,
                        group_size,
                    } = raw
                    else {
                        return Err(io::Error::other(format!(
                            "{prefix}: expected U4B8 expert weights"
                        )));
                    };
                    require(
                        (n, k, group_size) == (n_expected, k_expected, layout.group_size),
                        format!("{prefix}: unexpected expert dimensions"),
                    )?;
                    require(
                        packed.len() * 4 == weights.len()
                            && scales.len() * 2 == scales_region.len(),
                        format!("{prefix}: invalid packed length"),
                    )?;
                    for (dst, word) in pack[weights.clone()]
                        .as_chunks_mut::<4>()
                        .0
                        .iter_mut()
                        .zip(packed)
                    {
                        dst.copy_from_slice(&word.to_le_bytes());
                    }
                    for (dst, scale) in pack[scales_region.clone()]
                        .as_chunks_mut::<2>()
                        .0
                        .iter_mut()
                        .zip(scales)
                    {
                        dst.copy_from_slice(&scale.to_le_bytes());
                    }
                }
            }
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
            self.storage.len_bytes
        }

        /// Read-only bytes, also suitable as a source for a synchronized H2D copy.
        pub fn expert_pack(&self, global_id: usize) -> Option<&[u8]> {
            if !self.expert_ids.contains(&global_id) {
                return None;
            }
            let offset = (global_id - self.expert_ids.start) * self.layout.stride;
            // SAFETY: immutable initialized storage; checked ID selects a complete pack.
            Some(unsafe {
                std::slice::from_raw_parts(
                    self.storage.host_ptr.cast::<u8>().add(offset),
                    self.layout.stride,
                )
            })
        }
    }
}

#[cfg(feature = "cuda")]
pub use host_store::HostExpertStore;

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
}
