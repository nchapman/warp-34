//! Host quantization, weight loading, and device linear/FFN projections.

use super::*;
pub use crate::loader::{
    HADAMARD_BLOCK_16, HADAMARD_BLOCK_64, SUPPORTED_HADAMARD_BLOCKS,
    is_valid_hadamard_block_for_dim,
};

pub const FP16_ELEM_BYTES: usize = 2;
pub const HADAMARD_WORKSPACE_ALIGN_BYTES: usize = 16;
pub const PRODUCER_FUSED_H64_MAX_HIDDEN_SIZE: usize = 6144;
pub const MARLIN_TILE_N: usize = 64;
pub const MARLIN_TILE_K_W4: usize = 128;
pub const MARLIN_GROUP_BLOCK_K: usize = 16;
pub const MARLIN_CHANNELWISE_GROUP_BLOCKS: i32 = -1;
pub const U2B2_PACKED_ZERO_WORD: u32 = 0xaaaa_aaaa;
pub const U2_CRUMBS_PER_U32: usize = 16;
pub const U4B8_PACKED_ZERO_WORD: u32 = 0x8888_8888;
pub const U4_NIBBLES_PER_U32: usize = 8;
pub const U8_BYTES_PER_U32: usize = 4;
pub const DEVICE_SCRATCH_ALIGNMENT_BYTES: usize = 256;
pub const SMALL_N_GEMM_ALWAYS_MAX_N: usize = 256;
pub const SMALL_N_GEMM_ROUTER_512_MAX_N: usize = 512;
pub const SMALL_N_GEMM_DECODE_MAX_N: usize = 2048;
pub const SMALL_N_GEMM_DECODE_MAX_DIM: usize = 2560;
pub const SMALL_N_GEMM_PLE_MAX_N: usize = 12288;
pub const SMALL_N_GEMM_WIDE_N_MIN_K: usize = 2048;
pub const SMALL_N_GEMM_DECODE_MAX_M: usize = 16;
pub const SMALL_N_GEMM_K_ALIGN: usize = 8;

#[inline]
pub const fn align_up_hadamard_workspace(bytes: usize) -> usize {
    let mask = HADAMARD_WORKSPACE_ALIGN_BYTES - 1;
    bytes.saturating_add(mask) & !mask
}

#[inline]
pub const fn align_up_scratch(bytes: usize) -> Option<usize> {
    let mask = DEVICE_SCRATCH_ALIGNMENT_BYTES - 1;
    match bytes.checked_add(mask) {
        Some(v) => Some(v & !mask),
        None => None,
    }
}

#[cfg(feature = "cuda")]
pub enum LinearProj {
    /// Logical projection backed by padded Marlin weights in the Hadamard basis.
    Hadamard {
        inner: Box<LinearProj>,
        size_n: usize,
        size_k: usize,
        block_size: usize,
    },
    WithLora {
        inner: Box<LinearProj>,
        lora: Box<crate::lora::DeviceLoraStack>,
    },
    MarlinW2A16 {
        b_q_weight_ptr: u64,
        b_scales_ptr: u64,
        size_n: usize,
        size_k: usize,
        group_blocks: i32,
    },
    MarlinW4A16 {
        b_q_weight_ptr: u64,
        b_scales_ptr: u64,
        size_n: usize,
        size_k: usize,
        group_blocks: i32,
    },
    MarlinW4A16Zp {
        b_q_weight_ptr: u64,
        b_scales_ptr: u64,
        b_zeros_ptr: u64,
        size_n: usize,
        size_k: usize,
        group_blocks: i32,
    },
    MarlinW8A16 {
        b_q_weight_ptr: u64,
        b_scales_ptr: u64,
        size_n: usize,
        size_k: usize,
        group_blocks: i32,
    },
    DenseF16 {
        weight_ptr: u64,
        size_n: usize,
        size_k: usize,
    },
}

#[cfg(feature = "cuda")]
impl LinearProj {
    pub fn size_n(&self) -> usize {
        match self {
            Self::WithLora { inner, .. } => inner.size_n(),
            Self::Hadamard { size_n, .. }
            | Self::MarlinW2A16 { size_n, .. }
            | Self::MarlinW4A16 { size_n, .. }
            | Self::MarlinW4A16Zp { size_n, .. }
            | Self::MarlinW8A16 { size_n, .. }
            | Self::DenseF16 { size_n, .. } => *size_n,
        }
    }

    pub fn size_k(&self) -> usize {
        match self {
            Self::WithLora { inner, .. } => inner.size_k(),
            Self::Hadamard { size_k, .. }
            | Self::MarlinW2A16 { size_k, .. }
            | Self::MarlinW4A16 { size_k, .. }
            | Self::MarlinW4A16Zp { size_k, .. }
            | Self::MarlinW8A16 { size_k, .. }
            | Self::DenseF16 { size_k, .. } => *size_k,
        }
    }

    pub fn byte_footprint(&self) -> usize {
        match self {
            Self::Hadamard { inner, .. } | Self::WithLora { inner, .. } => inner.byte_footprint(),
            Self::MarlinW2A16 {
                size_n,
                size_k,
                group_blocks,
                ..
            } => {
                let groups = if *group_blocks < 0 {
                    1
                } else {
                    size_k / (*group_blocks as usize * MARLIN_GROUP_BLOCK_K)
                };
                (size_n * size_k / 4) + (size_n * groups * 2)
            }
            Self::MarlinW4A16 {
                size_n,
                size_k,
                group_blocks,
                ..
            } => {
                let groups = if *group_blocks < 0 {
                    1
                } else {
                    size_k / (*group_blocks as usize * MARLIN_GROUP_BLOCK_K)
                };
                (size_n * size_k / 2) + (size_n * groups * 2)
            }
            Self::MarlinW4A16Zp {
                size_n,
                size_k,
                group_blocks,
                ..
            } => {
                let g = (*group_blocks as usize) * MARLIN_GROUP_BLOCK_K;
                (size_n * size_k / 2) + (size_n * (size_k / g) * 2) + ((size_n / 2) * (size_k / g))
            }
            Self::MarlinW8A16 {
                size_n,
                size_k,
                group_blocks,
                ..
            } => {
                let num_groups = if *group_blocks < 0 {
                    1
                } else {
                    size_k / ((*group_blocks as usize) * MARLIN_GROUP_BLOCK_K)
                };
                (size_n * size_k) + (size_n * num_groups * 2)
            }
            Self::DenseF16 { size_n, size_k, .. } => size_n * size_k * 2,
        }
    }

    /// Enqueue a projection using caller-managed device addresses. The model
    /// loader supplies weights; forward supplies MxK input, MxN output, and
    /// nonaliasing workspace on the execution stream. This legacy raw-address
    /// API does not enforce those capacities, lifetimes, or device affinity.
    /// Whether this projection can execute on `aux_stream` without `ex.scratch_ptr` or `cuBLAS`.
    #[inline]
    pub fn can_fork_on_aux_stream(&self) -> bool {
        match *self {
            Self::Hadamard { .. } => false,
            Self::WithLora {
                ref inner,
                ref lora,
            } => lora.is_currently_noop() && inner.can_fork_on_aux_stream(),
            Self::DenseF16 { size_n, size_k, .. } => {
                size_n <= SMALL_N_GEMM_ALWAYS_MAX_N && size_k.is_multiple_of(SMALL_N_GEMM_K_ALIGN)
            }
            Self::MarlinW2A16 { .. }
            | Self::MarlinW4A16 { .. }
            | Self::MarlinW4A16Zp { .. }
            | Self::MarlinW8A16 { .. } => true,
        }
    }

    /// Returns the underlying projection with any outer [`LinearProj::WithLora`] wrapper peeled off.
    #[inline]
    pub fn without_lora(&self) -> &LinearProj {
        match self {
            Self::WithLora { inner, .. } => inner.without_lora(),
            other => other,
        }
    }

    /// Removes any attached [`LinearProj::WithLora`] wrapper in place, restoring the base projection.
    pub fn strip_lora(&mut self) {
        match self {
            Self::Hadamard { inner, .. } => inner.strip_lora(),
            Self::WithLora { inner, .. } => {
                inner.strip_lora();
                let dummy = Self::DenseF16 {
                    weight_ptr: 0,
                    size_n: 0,
                    size_k: 0,
                };
                let base = core::mem::replace(inner.as_mut(), dummy);
                *self = base;
            }
            _ => {}
        }
    }

    /// Wraps this projection (or its inner projection when `Self::Hadamard`) with a
    /// [`crate::lora::DeviceLoraStack`], uploading `a_f16` and `b_f16` (pre-rotated in the
    /// Hadamard basis when `Self::Hadamard`) into `lora_arena`.
    pub fn wrap_with_lora(
        &mut self,
        packed: &crate::lora::PackedProjLoraHost,
        lora_arena: &mut DeviceArena,
        stream: &CudaStream,
        state: &std::sync::Arc<crate::lora::RankLoraState>,
    ) -> Result<()> {
        self.strip_lora();
        let (target, had_blk): (&mut LinearProj, Option<usize>) = match self {
            Self::Hadamard {
                inner, block_size, ..
            } => (inner.as_mut(), Some(*block_size)),
            other => (other, None),
        };
        let (phys_n, phys_k) = (target.size_n(), target.size_k());
        let (a_f16, b_f16) = packed.to_device_f16(phys_n, phys_k, had_blk);
        let lora_a_ptr = lora_arena.upload_slice(&a_f16, stream)?;
        let lora_b_ptr = lora_arena.upload_slice(&b_f16, stream)?;
        let prev = core::mem::replace(
            target,
            Self::DenseF16 {
                weight_ptr: 0,
                size_n: 0,
                size_k: 0,
            },
        );
        *target = Self::WithLora {
            inner: Box::new(prev),
            lora: Box::new(crate::lora::DeviceLoraStack {
                lora_a_ptr,
                lora_b_ptr,
                size_n: phys_n,
                size_k: phys_k,
                rank_pad: packed.rank_pad,
                num_adapters: packed.num_adapters,
                adapter_mask: packed.adapter_mask,
                num_b_slices: if phys_n == packed.logical_n {
                    packed.num_b_slices
                } else {
                    0
                },
                b_slices4: packed.b_slices4,
                state: std::sync::Arc::clone(state),
            }),
        };
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    pub fn forward(
        &self,
        ex: &W34Exec,
        cublas: &CublasHandle,
        marlin_ws_ptr: u64,
        marlin_ws_ints: usize,
        x_ptr: u64,
        out_ptr: u64,
        m: usize,
    ) -> Result<()> {
        if let Self::WithLora { inner, lora } = self {
            inner.forward(ex, cublas, marlin_ws_ptr, marlin_ws_ints, x_ptr, out_ptr, m)?;
            return lora.forward_with_cublas(ex, cublas.handle, x_ptr, out_ptr, m);
        }
        if let Self::Hadamard {
            inner,
            size_n,
            size_k,
            block_size,
        } = self
        {
            return Self::forward_hadamard(
                inner,
                *size_n,
                *size_k,
                *block_size,
                ex,
                cublas,
                marlin_ws_ptr,
                marlin_ws_ints,
                x_ptr,
                out_ptr,
                m,
            );
        }
        match *self {
            Self::Hadamard { .. } | Self::WithLora { .. } => unreachable!("handled above"),
            Self::MarlinW2A16 {
                b_q_weight_ptr,
                b_scales_ptr,
                size_n,
                size_k,
                group_blocks,
            } => check_rc(
                // SAFETY: model upload repacks U2B2 weights and permutes scales
                // for this N/K/group geometry. X/output/workspace extents and
                // stream ordering are supplied by the raw-address caller.
                unsafe {
                    ffi::w34_marlin_gemm_f16u2(
                        ex,
                        x_ptr as *const c_void,
                        b_q_weight_ptr as *const c_void,
                        b_scales_ptr as *const c_void,
                        out_ptr as *mut c_void,
                        marlin_ws_ptr as *mut i32,
                        m as i64,
                        size_n as i64,
                        size_k as i64,
                        marlin_ws_ints as i64,
                        group_blocks,
                    )
                },
                "w34_marlin_gemm_f16u2",
            ),
            Self::MarlinW4A16 {
                b_q_weight_ptr,
                b_scales_ptr,
                size_n,
                size_k,
                group_blocks,
            } => check_rc(
                // SAFETY: model upload repacks U4B8 weights and permutes scales
                // for this N/K/group geometry. X/output/workspace extents and
                // stream ordering are supplied by the raw-address caller.
                unsafe {
                    ffi::w34_marlin_gemm_f16u4(
                        ex,
                        x_ptr as *const c_void,
                        b_q_weight_ptr as *const c_void,
                        b_scales_ptr as *const c_void,
                        out_ptr as *mut c_void,
                        marlin_ws_ptr as *mut i32,
                        m as i64,
                        size_n as i64,
                        size_k as i64,
                        marlin_ws_ints as i64,
                        group_blocks,
                    )
                },
                "w34_marlin_gemm_f16u4",
            ),
            Self::MarlinW4A16Zp {
                b_q_weight_ptr,
                b_scales_ptr,
                b_zeros_ptr,
                size_n,
                size_k,
                group_blocks,
            } => check_rc(
                // SAFETY: upload creates Marlin U4 tiles with correspondingly
                // permuted scales and zero points. Caller-managed X/output and
                // workspace must match M/N/K and remain live for this stream.
                unsafe {
                    ffi::w34_marlin_gemm_f16u4_zp(
                        ex,
                        x_ptr as *const c_void,
                        b_q_weight_ptr as *const c_void,
                        b_scales_ptr as *const c_void,
                        b_zeros_ptr as *const c_void,
                        out_ptr as *mut c_void,
                        marlin_ws_ptr as *mut i32,
                        m as i64,
                        size_n as i64,
                        size_k as i64,
                        marlin_ws_ints as i64,
                        group_blocks,
                    )
                },
                "w34_marlin_gemm_f16u4_zp",
            ),
            Self::MarlinW8A16 {
                b_q_weight_ptr,
                b_scales_ptr,
                size_n,
                size_k,
                group_blocks,
            } => check_rc(
                // SAFETY: upload creates U8B128 Marlin weights/scales for this
                // geometry (including channelwise groups). The raw-address caller
                // supplies live MxK/MxN buffers and separate Marlin workspace.
                unsafe {
                    ffi::w34_marlin_gemm_f16u8(
                        ex,
                        x_ptr as *const c_void,
                        b_q_weight_ptr as *const c_void,
                        b_scales_ptr as *const c_void,
                        out_ptr as *mut c_void,
                        marlin_ws_ptr as *mut i32,
                        m as i64,
                        size_n as i64,
                        size_k as i64,
                        marlin_ws_ints as i64,
                        group_blocks,
                    )
                },
                "w34_marlin_gemm_f16u8",
            ),
            Self::DenseF16 {
                weight_ptr,
                size_n,
                size_k,
            } => {
                // Small-N projections:
                // - `size_n <= 256` (GDN `in_proj_ba`, 256-expert `router_gate`, etc.) uses the
                //   batch-size-invariant warp-reduction kernel across all `m` for 0-ULP prefill/decode parity.
                // - `size_n <= 2048 && size_k <= 2048` (`hc_lowrank = 320`, 512-expert `router_gate`,
                //   and unquantized HGPTQ MoE `shared_expert` `gate_up`/`down` projections) uses the
                //   warp-reduction kernel for decode and speculative verify (`m <= 16`, guaranteeing
                //   0-ULP parity between `M = 1` decode and `M <= 16` MTP/DFlash2 verify) while keeping
                //   `m > 16` chunked prefill on cuBLAS Tensor Cores.
                if (size_n <= SMALL_N_GEMM_ALWAYS_MAX_N
                    || (size_k <= SMALL_N_GEMM_DECODE_MAX_DIM
                        && (size_n <= SMALL_N_GEMM_DECODE_MAX_N
                            || (size_n <= SMALL_N_GEMM_PLE_MAX_N
                                && size_k >= SMALL_N_GEMM_WIDE_N_MIN_K))
                        && m <= SMALL_N_GEMM_DECODE_MAX_M))
                    && size_k.is_multiple_of(SMALL_N_GEMM_K_ALIGN)
                {
                    check_rc(
                        // SAFETY: this branch enforces the small-N and K alignment
                        // specialization. The loaded weight is NxK FP16; input and
                        // output extents/lifetimes still rely on the caller.
                        unsafe {
                            ffi::w34_small_n_gemm_f16(
                                ex,
                                x_ptr as *const c_void,
                                weight_ptr as *const c_void,
                                out_ptr as *mut c_void,
                                m as i32,
                                size_n as i32,
                                size_k as i32,
                            )
                        },
                        "w34_small_n_gemm_f16",
                    )
                } else {
                    cublas.gemm_f16(
                        x_ptr,
                        weight_ptr,
                        out_ptr,
                        m as i32,
                        size_n as i32,
                        size_k as i32,
                    )
                }
            }
        }
    }
}

/// Raw host representation of a linear weight before sharding/uploading.
pub(crate) enum RawLinearHost {
    Hadamard {
        inner: Box<RawLinearHost>,
        block_size: usize,
    },
    U2B2NMajor {
        packed: Vec<u32>,
        scales: Vec<u16>,
        n: usize,
        k: usize,
        group_size: usize,
    },
    U4B8NMajor {
        packed: Vec<u32>,
        scales: Vec<u16>,
        n: usize,
        k: usize,
        group_size: usize,
    },
    U4ZpNMajor {
        packed: Vec<u32>,
        scales: Vec<u16>,
        /// N-major `[n / 8, k / group_size]` packed 4-bit zero-points (8 consecutive N-rows per `u32`).
        qzeros: Vec<u32>,
        n: usize,
        k: usize,
        group_size: usize,
    },
    U8B128NMajor {
        packed: Vec<u32>,
        scales: Vec<u16>,
        n: usize,
        k: usize,
        group_size: usize,
        channelwise: bool,
    },
    DenseF16 {
        weight: Vec<u16>,
        n: usize,
        k: usize,
    },
}

impl RawLinearHost {
    pub(crate) fn dims(&self) -> (usize, usize) {
        match self {
            Self::Hadamard { inner, .. } => inner.dims(),
            Self::U2B2NMajor { n, k, .. }
            | Self::U4B8NMajor { n, k, .. }
            | Self::U4ZpNMajor { n, k, .. }
            | Self::U8B128NMajor { n, k, .. }
            | Self::DenseF16 { n, k, .. } => (*n, *k),
        }
    }

    #[cfg(test)]
    pub(crate) fn into_u4b8_if_u2b2(self) -> Self {
        match self {
            Self::Hadamard { inner, block_size } => Self::Hadamard {
                inner: Box::new(inner.into_u4b8_if_u2b2()),
                block_size,
            },
            Self::U2B2NMajor {
                packed,
                scales,
                n,
                k,
                group_size,
            } => Self::U4B8NMajor {
                packed: repack_u2b2_to_u4b8(&packed),
                scales,
                n,
                k,
                group_size,
            },
            other => other,
        }
    }

    pub(crate) fn to_f16_vec(&self) -> Vec<u16> {
        match self {
            Self::Hadamard { inner, block_size } => hgptq::unrotate_weights(inner, *block_size),
            Self::DenseF16 { weight, .. } => weight.clone(),
            Self::U2B2NMajor {
                packed,
                scales,
                n,
                k,
                group_size,
            } => dequant_u2b2_n_major_to_f16(packed, scales, *n, *k, *group_size),
            Self::U4B8NMajor {
                packed,
                scales,
                n,
                k,
                group_size,
            } => dequant_u4b8_n_major_to_f16(packed, scales, *n, *k, *group_size),
            Self::U4ZpNMajor {
                packed,
                scales,
                qzeros,
                n,
                k,
                group_size,
            } => dequant_u4_zp_n_major_to_f16(packed, scales, qzeros, *n, *k, *group_size),
            Self::U8B128NMajor {
                packed,
                scales,
                n,
                k,
                group_size,
                ..
            } => dequant_u8b128_n_major_to_f16(packed, scales, *n, *k, *group_size),
        }
    }

    pub(crate) fn load(
        archive: &SafetensorsArchive,
        base_prefix: &str,
        quant: &QuantScheme,
    ) -> Result<Self> {
        Self::load_with_transform(archive, base_prefix, quant)
    }

    fn load_untransformed(
        archive: &SafetensorsArchive,
        base_prefix: &str,
        quant: &QuantScheme,
    ) -> Result<Self> {
        let has_unquantized_weight = archive
            .resolve_name(&format!("{base_prefix}.weight"))
            .and_then(|k| archive.tensor_info(&k).ok())
            .is_some_and(|info| {
                info.shape.len() == 2
                    && matches!(info.dtype.as_str(), "F16" | "BF16" | "F32")
                    && archive
                        .resolve_name(&format!("{base_prefix}.weight_scale_inv"))
                        .is_none()
                    && (archive
                        .resolve_name(&format!("{base_prefix}.weight_scale"))
                        .is_none()
                        || archive
                            .resolve_name(&format!("{base_prefix}.weight_packed"))
                            .is_some())
            });
        // 1. compressed-tensors pack-quantized (`*.weight_packed` + `*.weight_scale` [+ `*.weight_zero_point`])
        let packed_key = format!("{base_prefix}.weight_packed");
        if !has_unquantized_weight && let Some(real_packed) = archive.resolve_name(&packed_key) {
            let scale_key = archive
                .resolve_name(&format!("{base_prefix}.weight_scale"))
                .ok_or_else(|| {
                    loader_err_msg(format!(
                        "missing '{base_prefix}.weight_scale' for packed tensor '{real_packed}'"
                    ))
                })?;
            let p_info = archive
                .tensor_info(&real_packed)
                .map_err(|e| loader_err_msg(format!("tensor_info('{real_packed}') failed: {e}")))?;
            let s_info = archive
                .tensor_info(&scale_key)
                .map_err(|e| loader_err_msg(format!("tensor_info('{scale_key}') failed: {e}")))?;
            if p_info.shape.len() != 2 || s_info.shape.len() != 2 || s_info.shape[1] == 0 {
                return Err(loader_err_msg(format!(
                    "invalid compressed-tensors 2D shapes for '{base_prefix}': weight_packed={:?}, weight_scale={:?}",
                    p_info.shape, s_info.shape
                )));
            }
            if !matches!(p_info.dtype.as_str(), "I32" | "U32") {
                return Err(loader_err_msg(format!(
                    "unsupported weight_packed dtype '{}' on '{real_packed}' (expected I32/U32)",
                    p_info.dtype
                )));
            }
            if !matches!(s_info.dtype.as_str(), "F16" | "BF16" | "F32") {
                return Err(loader_err_msg(format!(
                    "unsupported weight_scale dtype '{}' on '{scale_key}' (expected F16/BF16/F32)",
                    s_info.dtype
                )));
            }
            let shape_override = archive
                .resolve_name(&format!("{base_prefix}.weight_shape"))
                .map(|sk| {
                    archive.read_weight_shape_2d(&sk).map_err(|e| {
                        loader_err_msg(format!("read_weight_shape_2d('{sk}') failed: {e}"))
                    })
                })
                .transpose()?;
            let bits = if let Some([shape_n, shape_k]) = shape_override {
                if shape_n != p_info.shape[0]
                    || p_info.shape[1] == 0
                    || !shape_k.is_multiple_of(p_info.shape[1])
                {
                    return Err(loader_err_msg(format!(
                        "mismatched compressed-tensors weight_shape [{shape_n}, {shape_k}] vs weight_packed {:?} on '{base_prefix}'",
                        p_info.shape
                    )));
                }
                let pf = shape_k / p_info.shape[1];
                let inferred = match pf {
                    16 => 2usize,
                    8 => 4usize,
                    4 => 8usize,
                    _ => {
                        return Err(loader_err_msg(format!(
                            "unsupported compressed-tensors pack_factor={pf} from weight_shape [{shape_n}, {shape_k}] on '{base_prefix}'",
                        )));
                    }
                };
                if quant.bits() != 0
                    && quant.bits() != inferred
                    && !base_prefix.ends_with("embed_tokens")
                {
                    return Err(loader_err_msg(format!(
                        "compressed-tensors weight_shape implies {inferred}-bit on '{base_prefix}', but config specifies {}-bit",
                        quant.bits()
                    )));
                }
                inferred
            } else {
                match quant.bits() {
                    8 => 8usize,
                    4 | 0 => 4usize,
                    2 => 2usize,
                    other => {
                        return Err(loader_err_msg(format!(
                            "unsupported compressed-tensors bit-width {other} on '{base_prefix}'"
                        )));
                    }
                }
            };
            let pack_factor = 32 / bits;
            let n = p_info.shape[0];
            let k = p_info.shape[1]
                .checked_mul(pack_factor)
                .ok_or_else(|| loader_err("overflow on packed K dimension"))?;
            let num_groups = s_info.shape[1];
            if n == 0
                || k == 0
                || num_groups == 0
                || s_info.shape[0] != n
                || !k.is_multiple_of(num_groups)
            {
                return Err(loader_err_msg(format!(
                    "mismatched compressed-tensors scale shape {:?} against [{n}, {k}] ({bits}-bit) on '{base_prefix}'",
                    s_info.shape
                )));
            }
            let group_size = k / num_groups;
            let channelwise = num_groups == 1 && (quant.is_channelwise() || k > 128);
            if let Some(g_idx_key) = archive
                .resolve_name(&format!("{base_prefix}.weight_g_idx"))
                .or_else(|| archive.resolve_name(&format!("{base_prefix}.g_idx")))
            {
                let g_idx = archive
                    .read_u32(&g_idx_key)
                    .map_err(|e| loader_err_msg(format!("read_u32('{g_idx_key}') failed: {e}")))?;
                validate_static_gptq_g_idx(
                    &g_idx,
                    k,
                    if channelwise { 0 } else { group_size },
                    &g_idx_key,
                )
                .map_err(|e| loader_err_msg(e.to_string()))?;
            }
            let packed = archive
                .read_u32(&real_packed)
                .map_err(|e| loader_err_msg(format!("read_u32('{real_packed}') failed: {e}")))?;
            let scales = archive
                .read_f16_bits(&scale_key)
                .map_err(|e| loader_err_msg(format!("read_f16_bits('{scale_key}') failed: {e}")))?;
            let expected_packed = n
                .checked_mul(k / pack_factor)
                .ok_or_else(|| loader_err("packed dimensions overflow"))?;
            if packed.len() != expected_packed {
                return Err(loader_err_msg(format!(
                    "packed length {} does not match expected {} on '{real_packed}'",
                    packed.len(),
                    expected_packed
                )));
            }
            let expected_scales = n
                .checked_mul(num_groups)
                .ok_or_else(|| loader_err("scale dimensions overflow"))?;
            if scales.len() != expected_scales {
                return Err(loader_err_msg(format!(
                    "scales length {} does not match expected {} on '{scale_key}'",
                    scales.len(),
                    expected_scales
                )));
            }

            if let Some(zp_key) = archive.resolve_name(&format!("{base_prefix}.weight_zero_point"))
            {
                if bits != 4 {
                    return Err(loader_err_msg(format!(
                        "unsupported {bits}-bit asymmetric weight_zero_point '{zp_key}' (only 4-bit zero-points are supported)"
                    )));
                }
                let zp_info = archive
                    .tensor_info(&zp_key)
                    .map_err(|e| loader_err_msg(format!("tensor_info('{zp_key}') failed: {e}")))?;
                if !matches!(zp_info.dtype.as_str(), "I32" | "U32") {
                    return Err(loader_err_msg(format!(
                        "unsupported weight_zero_point dtype '{}' on '{zp_key}' (expected I32/U32)",
                        zp_info.dtype
                    )));
                }
                if !n.is_multiple_of(8) || zp_info.shape.as_slice() != [n / 8, num_groups] {
                    return Err(loader_err_msg(format!(
                        "unexpected weight_zero_point shape {:?} on '{zp_key}' (expected [{}, {num_groups}])",
                        zp_info.shape,
                        n / 8
                    )));
                }
                let qzeros = archive
                    .read_u32(&zp_key)
                    .map_err(|e| loader_err_msg(format!("read_u32('{zp_key}') failed: {e}")))?;
                let expected_qzeros = (n / 8)
                    .checked_mul(num_groups)
                    .ok_or_else(|| loader_err("zero-point dimensions overflow"))?;
                if qzeros.len() != expected_qzeros {
                    return Err(loader_err_msg(format!(
                        "zero-point length {} does not match expected {} on '{zp_key}'",
                        qzeros.len(),
                        expected_qzeros
                    )));
                }
                return Ok(Self::U4ZpNMajor {
                    packed,
                    scales,
                    qzeros,
                    n,
                    k,
                    group_size,
                });
            }
            if quant.expects_zero_points() && !base_prefix.ends_with("embed_tokens") {
                return Err(loader_err_msg(format!(
                    "missing '{base_prefix}.weight_zero_point' required by asymmetric compressed-tensors config"
                )));
            }
            return if bits == 8 {
                Ok(Self::U8B128NMajor {
                    packed,
                    scales,
                    n,
                    k,
                    group_size,
                    channelwise,
                })
            } else if bits == 2 {
                Ok(Self::U2B2NMajor {
                    packed,
                    scales,
                    n,
                    k,
                    group_size,
                })
            } else {
                Ok(Self::U4B8NMajor {
                    packed,
                    scales,
                    n,
                    k,
                    group_size,
                })
            };
        }

        // 2. GPTQ / AutoRound / AutoAWQ (`*.qweight` + `*.scales` [+ `*.qzeros` / `*.g_idx`])
        let qweight_key = format!("{base_prefix}.qweight");
        if let Some(real_qw) = archive.resolve_name(&qweight_key) {
            let scale_key = archive
                .resolve_name(&format!("{base_prefix}.scales"))
                .ok_or_else(|| {
                    loader_err_msg(format!(
                        "missing '{base_prefix}.scales' for quantized tensor '{real_qw}'"
                    ))
                })?;
            let qw_info = archive
                .tensor_info(&real_qw)
                .map_err(|e| loader_err_msg(format!("tensor_info('{real_qw}') failed: {e}")))?;
            let s_info = archive
                .tensor_info(&scale_key)
                .map_err(|e| loader_err_msg(format!("tensor_info('{scale_key}') failed: {e}")))?;
            if qw_info.shape.len() != 2 || s_info.shape.len() != 2 || s_info.shape[0] == 0 {
                return Err(loader_err_msg(format!(
                    "invalid 2D qweight/scales shapes for '{base_prefix}': qweight={:?}, scales={:?}",
                    qw_info.shape, s_info.shape
                )));
            }
            if !matches!(qw_info.dtype.as_str(), "I32" | "U32") {
                return Err(loader_err_msg(format!(
                    "unsupported qweight dtype '{}' on '{real_qw}' (expected I32/U32)",
                    qw_info.dtype
                )));
            }

            if quant.is_awq_packing() {
                let k = qw_info.shape[0];
                let n8 = qw_info.shape[1];
                let n = n8
                    .checked_mul(8)
                    .ok_or_else(|| loader_err("overflow on AWQ N dimension"))?;
                let num_groups = s_info.shape[0];
                if n8 == 0
                    || k == 0
                    || num_groups == 0
                    || !k.is_multiple_of(8)
                    || s_info.shape[1] != n
                    || !k.is_multiple_of(num_groups)
                {
                    return Err(loader_err_msg(format!(
                        "mismatched AWQ qweight {:?} / scales {:?} on '{base_prefix}' (expected [K, N/8] and [K/g, N])",
                        qw_info.shape, s_info.shape
                    )));
                }
                let group_size = k / num_groups;
                let qw_k_n8 = archive
                    .read_u32(&real_qw)
                    .map_err(|e| loader_err_msg(format!("read_u32('{real_qw}') failed: {e}")))?;
                let expected_qw = k
                    .checked_mul(n8)
                    .ok_or_else(|| loader_err("AWQ qweight dimensions overflow"))?;
                if qw_k_n8.len() != expected_qw {
                    return Err(loader_err_msg(format!(
                        "AWQ qweight length {} does not match expected {} on '{real_qw}'",
                        qw_k_n8.len(),
                        expected_qw
                    )));
                }
                let packed = unpack_awq_qweight_k_n8_to_n_major(&qw_k_n8, k, n);
                let scales_kg_n = archive.read_f16_bits(&scale_key).map_err(|e| {
                    loader_err_msg(format!("read_f16_bits('{scale_key}') failed: {e}"))
                })?;
                let expected_sc = num_groups
                    .checked_mul(n)
                    .ok_or_else(|| loader_err("AWQ scale dimensions overflow"))?;
                if scales_kg_n.len() != expected_sc {
                    return Err(loader_err_msg(format!(
                        "AWQ scales length {} does not match expected {} on '{scale_key}'",
                        scales_kg_n.len(),
                        expected_sc
                    )));
                }
                let mut scales_n_kg = vec![0u16; n * num_groups];
                for g in 0..num_groups {
                    for c in 0..n {
                        scales_n_kg[c * num_groups + g] = scales_kg_n[g * n + c];
                    }
                }
                if quant.expects_zero_points() {
                    let qz_key = archive
                        .resolve_name(&format!("{base_prefix}.qzeros"))
                        .ok_or_else(|| {
                            loader_err_msg(format!(
                                "missing '{base_prefix}.qzeros' required by AWQ zero_point=true config"
                            ))
                        })?;
                    let qz_info = archive.tensor_info(&qz_key).map_err(|e| {
                        loader_err_msg(format!("tensor_info('{qz_key}') failed: {e}"))
                    })?;
                    if !matches!(qz_info.dtype.as_str(), "I32" | "U32") {
                        return Err(loader_err_msg(format!(
                            "unsupported qzeros dtype '{}' on '{qz_key}' (expected I32/U32)",
                            qz_info.dtype
                        )));
                    }
                    if qz_info.shape.as_slice() != [num_groups, n8] {
                        return Err(loader_err_msg(format!(
                            "unexpected AWQ qzeros shape {:?} on '{qz_key}' (expected [{num_groups}, {n8}])",
                            qz_info.shape
                        )));
                    }
                    let qz_kg_n8 = archive
                        .read_u32(&qz_key)
                        .map_err(|e| loader_err_msg(format!("read_u32('{qz_key}') failed: {e}")))?;
                    let expected_qz = num_groups
                        .checked_mul(n8)
                        .ok_or_else(|| loader_err("AWQ zero-point dimensions overflow"))?;
                    if qz_kg_n8.len() != expected_qz {
                        return Err(loader_err_msg(format!(
                            "AWQ qzeros length {} does not match expected {} on '{qz_key}'",
                            qz_kg_n8.len(),
                            expected_qz
                        )));
                    }
                    let qzeros = unpack_awq_qzeros_kg_n8_to_n_major(&qz_kg_n8, k, n, group_size);
                    return Ok(Self::U4ZpNMajor {
                        packed,
                        scales: scales_n_kg,
                        qzeros,
                        n,
                        k,
                        group_size,
                    });
                }
                return Ok(Self::U4B8NMajor {
                    packed,
                    scales: scales_n_kg,
                    n,
                    k,
                    group_size,
                });
            }

            let bits = if quant.bits() == 8 { 8usize } else { 4usize };
            let pack_factor = 32 / bits;
            let k = qw_info.shape[0]
                .checked_mul(pack_factor)
                .ok_or_else(|| loader_err("overflow on GPTQ K dimension"))?;
            let n = qw_info.shape[1];
            let num_groups = s_info.shape[0];
            if n == 0
                || k == 0
                || num_groups == 0
                || s_info.shape[1] != n
                || !k.is_multiple_of(num_groups)
            {
                return Err(loader_err_msg(format!(
                    "mismatched GPTQ qweight {:?} / scales {:?} ({bits}-bit) on '{base_prefix}'",
                    qw_info.shape, s_info.shape
                )));
            }
            let group_size = k.checked_div(num_groups).unwrap_or(quant.group_size_or(k));
            let channelwise = num_groups == 1 && (quant.is_channelwise() || k > 128);
            if let Some(g_idx_key) = archive.resolve_name(&format!("{base_prefix}.g_idx")) {
                let g_idx = archive
                    .read_u32(&g_idx_key)
                    .map_err(|e| loader_err_msg(format!("read_u32('{g_idx_key}') failed: {e}")))?;
                validate_static_gptq_g_idx(
                    &g_idx,
                    k,
                    if channelwise { 0 } else { group_size },
                    &g_idx_key,
                )
                .map_err(|e| loader_err_msg(e.to_string()))?;
            }
            let qw_raw = archive
                .read_u32(&real_qw)
                .map_err(|e| loader_err_msg(format!("read_u32('{real_qw}') failed: {e}")))?;
            let expected_qw = (k / pack_factor)
                .checked_mul(n)
                .ok_or_else(|| loader_err("GPTQ qweight dimensions overflow"))?;
            if qw_raw.len() != expected_qw {
                return Err(loader_err_msg(format!(
                    "GPTQ qweight length {} does not match expected {} on '{real_qw}'",
                    qw_raw.len(),
                    expected_qw
                )));
            }
            let packed = if bits == 8 {
                transpose_gptq_qweight_k4_n_to_n_major(&qw_raw, k, n)
            } else {
                transpose_gptq_qweight_k8_n_to_n_major(&qw_raw, k, n)
            };
            let scales_kg_n = archive
                .read_f16_bits(&scale_key)
                .map_err(|e| loader_err_msg(format!("read_f16_bits('{scale_key}') failed: {e}")))?;
            let expected_sc = num_groups
                .checked_mul(n)
                .ok_or_else(|| loader_err("GPTQ scale dimensions overflow"))?;
            if scales_kg_n.len() != expected_sc {
                return Err(loader_err_msg(format!(
                    "GPTQ scales length {} does not match expected {} on '{scale_key}'",
                    scales_kg_n.len(),
                    expected_sc
                )));
            }
            let mut scales_n_kg = vec![0u16; n * num_groups];
            const TILE: usize = 32;
            for g_base in (0..num_groups).step_by(TILE) {
                let g_end = (g_base + TILE).min(num_groups);
                for c_base in (0..n).step_by(TILE) {
                    let c_end = (c_base + TILE).min(n);
                    for g in g_base..g_end {
                        let src_row = &scales_kg_n[g * n..g * n + c_end];
                        for (c, &val) in src_row.iter().enumerate().skip(c_base) {
                            scales_n_kg[c * num_groups + g] = val;
                        }
                    }
                }
            }
            return if bits == 8 {
                Ok(Self::U8B128NMajor {
                    packed,
                    scales: scales_n_kg,
                    n,
                    k,
                    group_size,
                    channelwise,
                })
            } else {
                Ok(Self::U4B8NMajor {
                    packed,
                    scales: scales_n_kg,
                    n,
                    k,
                    group_size,
                })
            };
        }

        // 3. Unquantized FP16 / BF16 / FP32 (`*.weight`)
        let weight_key = archive
            .resolve_name(&format!("{base_prefix}.weight"))
            .ok_or_else(|| {
                if archive
                    .resolve_name(&format!("{base_prefix}_blocks"))
                    .is_some()
                    || archive
                        .resolve_name(&format!("{base_prefix}.weight_blocks"))
                        .is_some()
                {
                    loader_err_msg(format!(
                        "unsupported MXFP4/NVFP4 block-quantized tensor at '{base_prefix}' (sm_86/sm_89 supports W4A16/W8A16 integer Marlin or FP16/BF16)"
                    ))
                } else {
                    loader_err_msg(format!(
                        "missing tensor '{base_prefix}.weight' (nor quantized .weight_packed / .qweight)"
                    ))
                }
            })?;
        if (archive
            .resolve_name(&format!("{base_prefix}.weight_scale"))
            .is_some()
            && archive
                .resolve_name(&format!("{base_prefix}.weight_packed"))
                .is_none())
            || archive
                .resolve_name(&format!("{base_prefix}.weight_scale_inv"))
                .is_some()
        {
            return Err(loader_err_msg(format!(
                "unsupported FP8/float-quantized tensor '{weight_key}' (accompanied by weight_scale/weight_scale_inv; warp-34 sm_86/sm_89 spine requires W4A16/W8A16 or FP16/BF16)"
            )));
        }
        let w_info = archive
            .tensor_info(&weight_key)
            .map_err(|e| loader_err_msg(format!("tensor_info('{weight_key}') failed: {e}")))?;
        if w_info.shape.len() != 2 {
            return Err(loader_err_msg(format!(
                "expected 2D weight tensor for '{weight_key}', found shape {:?} (dtype={})",
                w_info.shape, w_info.dtype
            )));
        }
        let n = w_info.shape[0];
        let k = w_info.shape[1];
        let weight = archive
            .read_f16_bits(&weight_key)
            .map_err(|e| loader_err_msg(format!("read_f16_bits('{weight_key}') failed: {e}")))?;
        let expected_weight = n
            .checked_mul(k)
            .ok_or_else(|| loader_err("dense weight dimensions overflow"))?;
        if weight.len() != expected_weight {
            return Err(loader_err_msg(format!(
                "dense weight element count {} does not match expected {} on '{weight_key}'",
                weight.len(),
                expected_weight
            )));
        }
        Ok(Self::DenseF16 { weight, n, k })
    }

    pub(crate) fn slice_rows(&self, row_start: usize, row_end: usize) -> Self {
        let out_n = row_end - row_start;
        match self {
            Self::Hadamard { inner, block_size } => Self::Hadamard {
                inner: Box::new(inner.slice_rows(row_start, row_end)),
                block_size: *block_size,
            },
            Self::U2B2NMajor {
                packed,
                scales,
                k,
                group_size,
                ..
            } => {
                let words_per_row = k / 16;
                let groups_per_row = k / group_size;
                Self::U2B2NMajor {
                    packed: packed[row_start * words_per_row..row_end * words_per_row].to_vec(),
                    scales: scales[row_start * groups_per_row..row_end * groups_per_row].to_vec(),
                    n: out_n,
                    k: *k,
                    group_size: *group_size,
                }
            }
            Self::U4B8NMajor {
                packed,
                scales,
                k,
                group_size,
                ..
            } => {
                let words_per_row = k / 8;
                let groups_per_row = k / group_size;
                Self::U4B8NMajor {
                    packed: packed[row_start * words_per_row..row_end * words_per_row].to_vec(),
                    scales: scales[row_start * groups_per_row..row_end * groups_per_row].to_vec(),
                    n: out_n,
                    k: *k,
                    group_size: *group_size,
                }
            }
            Self::U4ZpNMajor {
                packed,
                scales,
                qzeros,
                k,
                group_size,
                ..
            } => {
                if row_start.is_multiple_of(8) && row_end.is_multiple_of(8) {
                    let words_per_row = k / 8;
                    let groups_per_row = k / group_size;
                    let zp_start = (row_start / 8) * groups_per_row;
                    let zp_end = (row_end / 8) * groups_per_row;
                    Self::U4ZpNMajor {
                        packed: packed[row_start * words_per_row..row_end * words_per_row].to_vec(),
                        scales: scales[row_start * groups_per_row..row_end * groups_per_row]
                            .to_vec(),
                        qzeros: qzeros[zp_start..zp_end].to_vec(),
                        n: out_n,
                        k: *k,
                        group_size: *group_size,
                    }
                } else {
                    let f16_all = self.to_f16_vec();
                    Self::DenseF16 {
                        weight: f16_all[row_start * k..row_end * k].to_vec(),
                        n: out_n,
                        k: *k,
                    }
                }
            }
            Self::U8B128NMajor {
                packed,
                scales,
                k,
                group_size,
                channelwise,
                ..
            } => {
                let words_per_row = k / 4;
                let groups_per_row = if *channelwise { 1 } else { k / group_size };
                Self::U8B128NMajor {
                    packed: packed[row_start * words_per_row..row_end * words_per_row].to_vec(),
                    scales: scales[row_start * groups_per_row..row_end * groups_per_row].to_vec(),
                    n: out_n,
                    k: *k,
                    group_size: *group_size,
                    channelwise: *channelwise,
                }
            }
            Self::DenseF16 { weight, k, .. } => Self::DenseF16 {
                weight: weight[row_start * k..row_end * k].to_vec(),
                n: out_n,
                k: *k,
            },
        }
    }

    pub(crate) fn slice_cols(&self, col_start: usize, col_end: usize) -> Result<Self> {
        if col_start >= col_end || col_end > self.dims().1 {
            return Err(loader_err("invalid linear K shard"));
        }
        if let Self::Hadamard { block_size, .. } = self
            && (!col_start.is_multiple_of(*block_size) || !col_end.is_multiple_of(*block_size))
        {
            return Err(loader_err("K shard splits a Hadamard block"));
        }
        Ok(self.slice_cols_inner(col_start, col_end))
    }

    fn slice_cols_inner(&self, col_start: usize, col_end: usize) -> Self {
        let out_k = col_end - col_start;
        match self {
            Self::Hadamard { inner, block_size } => Self::Hadamard {
                inner: Box::new(inner.slice_cols_inner(col_start, col_end)),
                block_size: *block_size,
            },
            Self::U2B2NMajor {
                packed,
                scales,
                n,
                k,
                group_size,
            } => {
                let channelwise = *group_size == *k && *group_size > 128;
                if !col_start.is_multiple_of(16)
                    || !out_k.is_multiple_of(16)
                    || (!channelwise
                        && (!col_start.is_multiple_of(*group_size)
                            || !out_k.is_multiple_of(*group_size)))
                {
                    let dense = self.to_f16_vec();
                    let col_major: Vec<u16> = dense
                        .chunks_exact(*k)
                        .flat_map(|row| row[col_start..col_end].iter().copied())
                        .collect();
                    return Self::DenseF16 {
                        weight: col_major,
                        n: *n,
                        k: out_k,
                    };
                }
                let in_words = k / 16;
                let out_words = out_k / 16;
                let w_start = col_start / 16;
                let in_groups = k / group_size;
                let out_groups = if channelwise { 1 } else { out_k / group_size };
                let g_start = if channelwise {
                    0
                } else {
                    col_start / group_size
                };
                let mut new_packed = vec![0u32; n * out_words];
                let mut new_scales = vec![0u16; n * out_groups];
                for r in 0..*n {
                    new_packed[r * out_words..(r + 1) * out_words].copy_from_slice(
                        &packed[r * in_words + w_start..r * in_words + w_start + out_words],
                    );
                    new_scales[r * out_groups..(r + 1) * out_groups].copy_from_slice(
                        &scales[r * in_groups + g_start..r * in_groups + g_start + out_groups],
                    );
                }
                Self::U2B2NMajor {
                    packed: new_packed,
                    scales: new_scales,
                    n: *n,
                    k: out_k,
                    group_size: if channelwise { out_k } else { *group_size },
                }
            }
            Self::U4B8NMajor {
                packed,
                scales,
                n,
                k,
                group_size,
            } => {
                let channelwise = *group_size == *k && *group_size > 128;
                if !col_start.is_multiple_of(8)
                    || !out_k.is_multiple_of(8)
                    || (!channelwise
                        && (!col_start.is_multiple_of(*group_size)
                            || !out_k.is_multiple_of(*group_size)))
                {
                    let dense = self.to_f16_vec();
                    let col_major: Vec<u16> = dense
                        .chunks_exact(*k)
                        .flat_map(|row| row[col_start..col_end].iter().copied())
                        .collect();
                    return Self::DenseF16 {
                        weight: col_major,
                        n: *n,
                        k: out_k,
                    };
                }
                let in_words = k / 8;
                let out_words = out_k / 8;
                let w_start = col_start / 8;
                let in_groups = k / group_size;
                let out_groups = if channelwise { 1 } else { out_k / group_size };
                let g_start = if channelwise {
                    0
                } else {
                    col_start / group_size
                };
                let mut new_packed = vec![0u32; n * out_words];
                let mut new_scales = vec![0u16; n * out_groups];
                for r in 0..*n {
                    new_packed[r * out_words..(r + 1) * out_words].copy_from_slice(
                        &packed[r * in_words + w_start..r * in_words + w_start + out_words],
                    );
                    new_scales[r * out_groups..(r + 1) * out_groups].copy_from_slice(
                        &scales[r * in_groups + g_start..r * in_groups + g_start + out_groups],
                    );
                }
                Self::U4B8NMajor {
                    packed: new_packed,
                    scales: new_scales,
                    n: *n,
                    k: out_k,
                    group_size: if channelwise { out_k } else { *group_size },
                }
            }
            Self::U4ZpNMajor {
                packed,
                scales,
                qzeros,
                n,
                k,
                group_size,
            } => {
                if !col_start.is_multiple_of(8)
                    || !out_k.is_multiple_of(8)
                    || !col_start.is_multiple_of(*group_size)
                    || !out_k.is_multiple_of(*group_size)
                {
                    let dense = self.to_f16_vec();
                    let col_major: Vec<u16> = dense
                        .chunks_exact(*k)
                        .flat_map(|row| row[col_start..col_end].iter().copied())
                        .collect();
                    return Self::DenseF16 {
                        weight: col_major,
                        n: *n,
                        k: out_k,
                    };
                }
                let in_words = k / 8;
                let out_words = out_k / 8;
                let w_start = col_start / 8;
                let in_groups = k / group_size;
                let out_groups = out_k / group_size;
                let g_start = col_start / group_size;
                let mut new_packed = vec![0u32; n * out_words];
                let mut new_scales = vec![0u16; n * out_groups];
                for r in 0..*n {
                    new_packed[r * out_words..(r + 1) * out_words].copy_from_slice(
                        &packed[r * in_words + w_start..r * in_words + w_start + out_words],
                    );
                    new_scales[r * out_groups..(r + 1) * out_groups].copy_from_slice(
                        &scales[r * in_groups + g_start..r * in_groups + g_start + out_groups],
                    );
                }
                let n8 = n / 8;
                let mut new_qzeros = vec![0u32; n8 * out_groups];
                for np in 0..n8 {
                    new_qzeros[np * out_groups..(np + 1) * out_groups].copy_from_slice(
                        &qzeros[np * in_groups + g_start..np * in_groups + g_start + out_groups],
                    );
                }
                Self::U4ZpNMajor {
                    packed: new_packed,
                    scales: new_scales,
                    qzeros: new_qzeros,
                    n: *n,
                    k: out_k,
                    group_size: *group_size,
                }
            }
            Self::U8B128NMajor {
                packed,
                scales,
                n,
                k,
                group_size,
                channelwise,
            } => {
                if !col_start.is_multiple_of(4)
                    || !out_k.is_multiple_of(4)
                    || (!*channelwise
                        && (!col_start.is_multiple_of(*group_size)
                            || !out_k.is_multiple_of(*group_size)))
                {
                    let dense = self.to_f16_vec();
                    let col_major: Vec<u16> = dense
                        .chunks_exact(*k)
                        .flat_map(|row| row[col_start..col_end].iter().copied())
                        .collect();
                    return Self::DenseF16 {
                        weight: col_major,
                        n: *n,
                        k: out_k,
                    };
                }
                let in_words = k / 4;
                let out_words = out_k / 4;
                let w_start = col_start / 4;
                let mut new_packed = vec![0u32; n * out_words];
                for r in 0..*n {
                    new_packed[r * out_words..(r + 1) * out_words].copy_from_slice(
                        &packed[r * in_words + w_start..r * in_words + w_start + out_words],
                    );
                }
                if *channelwise {
                    Self::U8B128NMajor {
                        packed: new_packed,
                        scales: scales.clone(),
                        n: *n,
                        k: out_k,
                        group_size: out_k,
                        channelwise: true,
                    }
                } else {
                    let in_groups = k / group_size;
                    let out_groups = out_k / group_size;
                    let g_start = col_start / group_size;
                    let mut new_scales = vec![0u16; n * out_groups];
                    for r in 0..*n {
                        new_scales[r * out_groups..(r + 1) * out_groups].copy_from_slice(
                            &scales[r * in_groups + g_start..r * in_groups + g_start + out_groups],
                        );
                    }
                    Self::U8B128NMajor {
                        packed: new_packed,
                        scales: new_scales,
                        n: *n,
                        k: out_k,
                        group_size: *group_size,
                        channelwise: false,
                    }
                }
            }
            Self::DenseF16 { weight, n, k } => {
                let mut new_weight = vec![0u16; n * out_k];
                for r in 0..*n {
                    new_weight[r * out_k..(r + 1) * out_k]
                        .copy_from_slice(&weight[r * k + col_start..r * k + col_end]);
                }
                Self::DenseF16 {
                    weight: new_weight,
                    n: *n,
                    k: out_k,
                }
            }
        }
    }

    pub(crate) fn concat_rows(parts: &[Self]) -> Result<Self> {
        Self::concat_row_refs(&parts.iter().collect::<Vec<_>>())
    }

    fn concat_row_refs(parts: &[&Self]) -> Result<Self> {
        if parts.iter().any(|p| matches!(p, Self::Hadamard { .. })) {
            return Self::concat_transformed_rows(parts);
        }
        if parts.is_empty() {
            return Err(loader_err_msg("concat_rows called with empty parts"));
        }
        let k = parts[0].dims().1;
        for (i, p) in parts.iter().enumerate() {
            if p.dims().1 != k {
                return Err(loader_err_msg(format!(
                    "concat_rows K mismatch: part {i} has K={} but part 0 has K={k}",
                    p.dims().1
                )));
            }
        }
        if let Self::U2B2NMajor {
            group_size: gs0, ..
        } = &parts[0]
            && parts
                .iter()
                .all(|p| matches!(p, Self::U2B2NMajor { group_size, .. } if group_size == gs0))
        {
            let mut packed = Vec::new();
            let mut scales = Vec::new();
            let mut total_n = 0;
            for p in parts {
                if let Self::U2B2NMajor {
                    packed: pk,
                    scales: sc,
                    n,
                    ..
                } = p
                {
                    packed.extend_from_slice(pk);
                    scales.extend_from_slice(sc);
                    total_n += *n;
                }
            }
            return Ok(Self::U2B2NMajor {
                packed,
                scales,
                n: total_n,
                k,
                group_size: *gs0,
            });
        }

        if let Self::U4B8NMajor {
            group_size: gs0, ..
        } = &parts[0]
            && parts
                .iter()
                .all(|p| matches!(p, Self::U4B8NMajor { group_size, .. } if group_size == gs0))
        {
            let mut packed = Vec::new();
            let mut scales = Vec::new();
            let mut total_n = 0;
            for p in parts {
                if let Self::U4B8NMajor {
                    packed: pk,
                    scales: sc,
                    n,
                    ..
                } = p
                {
                    packed.extend_from_slice(pk);
                    scales.extend_from_slice(sc);
                    total_n += *n;
                }
            }
            return Ok(Self::U4B8NMajor {
                packed,
                scales,
                n: total_n,
                k,
                group_size: *gs0,
            });
        }

        if let Self::U2B2NMajor {
            group_size: gs0, ..
        }
        | Self::U4B8NMajor {
            group_size: gs0, ..
        } = &parts[0]
            && parts.iter().all(|p| {
                matches!(
                    p,
                    Self::U2B2NMajor { group_size, .. } | Self::U4B8NMajor { group_size, .. }
                        if group_size == gs0
                )
            })
        {
            let mut packed = Vec::new();
            let mut scales = Vec::new();
            let mut total_n = 0;
            for p in parts {
                match p {
                    Self::U2B2NMajor {
                        packed: pk,
                        scales: sc,
                        n,
                        ..
                    } => {
                        packed.extend_from_slice(&repack_u2b2_to_u4b8(pk));
                        scales.extend_from_slice(sc);
                        total_n += *n;
                    }
                    Self::U4B8NMajor {
                        packed: pk,
                        scales: sc,
                        n,
                        ..
                    } => {
                        packed.extend_from_slice(pk);
                        scales.extend_from_slice(sc);
                        total_n += *n;
                    }
                    _ => unreachable!(),
                }
            }
            return Ok(Self::U4B8NMajor {
                packed,
                scales,
                n: total_n,
                k,
                group_size: *gs0,
            });
        }

        if let Self::U4ZpNMajor {
            group_size: gs0, ..
        } = &parts[0]
            && parts.iter().all(
                |p| matches!(p, Self::U4ZpNMajor { group_size, n, .. } if group_size == gs0 && n.is_multiple_of(8)),
            )
        {
            let mut packed = Vec::new();
            let mut scales = Vec::new();
            let mut qzeros = Vec::new();
            let mut total_n = 0;
            for p in parts {
                if let Self::U4ZpNMajor {
                    packed: pk,
                    scales: sc,
                    qzeros: qz,
                    n,
                    ..
                } = p
                {
                    packed.extend_from_slice(pk);
                    scales.extend_from_slice(sc);
                    qzeros.extend_from_slice(qz);
                    total_n += *n;
                }
            }
            return Ok(Self::U4ZpNMajor {
                packed,
                scales,
                qzeros,
                n: total_n,
                k,
                group_size: *gs0,
            });
        }

        if let Self::U8B128NMajor {
            group_size: gs0,
            channelwise: cw0,
            ..
        } = &parts[0]
            && parts.iter().all(|p| {
                matches!(
                    p,
                    Self::U8B128NMajor {
                        group_size,
                        channelwise,
                        ..
                    } if group_size == gs0 && channelwise == cw0
                )
            })
        {
            let mut packed = Vec::new();
            let mut scales = Vec::new();
            let mut total_n = 0;
            for p in parts {
                if let Self::U8B128NMajor {
                    packed: pk,
                    scales: sc,
                    n,
                    ..
                } = p
                {
                    packed.extend_from_slice(pk);
                    scales.extend_from_slice(sc);
                    total_n += *n;
                }
            }
            return Ok(Self::U8B128NMajor {
                packed,
                scales,
                n: total_n,
                k,
                group_size: *gs0,
                channelwise: *cw0,
            });
        }

        let mut weight = Vec::new();
        let mut total_n = 0;
        for p in parts {
            let dq = p.to_f16_vec();
            weight.extend_from_slice(&dq);
            total_n += p.dims().0;
        }
        Ok(Self::DenseF16 {
            weight,
            n: total_n,
            k,
        })
    }

    #[cfg(feature = "cuda")]
    fn repack_with_staging(
        packed: &[u32],
        n: usize,
        k: usize,
        bits: usize,
        arena: &mut DeviceArena,
        stream: &CudaStream,
        repack_staging: &DeviceBuffer,
    ) -> Result<u64> {
        let bytes = packed.len() * 4;
        let b_q_weight_ptr = arena.alloc_aligned(bytes)?;
        let ex = stream.exec(0, 0);
        if bytes > repack_staging.len_bytes {
            return Err(loader_err_msg(format!(
                "packed weight size {} exceeds repack staging buffer {}",
                bytes, repack_staging.len_bytes
            )));
        }
        repack_staging.copy_from_host(packed, stream)?;
        // SAFETY: staging capacity is checked and the destination reserves the
        // same packed byte count. Upload/repack are ordered on stream. The loader
        // supplies packed length and N/K consistent with the selected bit width.
        let rc = unsafe {
            if bits == 8 {
                ffi::w34_marlin_repack_int8(
                    &ex,
                    repack_staging.addr() as *const u32,
                    b_q_weight_ptr as *mut u32,
                    k as i64,
                    n as i64,
                )
            } else if bits == 2 {
                ffi::w34_marlin_repack_int2(
                    &ex,
                    repack_staging.addr() as *const u32,
                    b_q_weight_ptr as *mut u32,
                    k as i64,
                    n as i64,
                )
            } else {
                ffi::w34_marlin_repack_int4(
                    &ex,
                    repack_staging.addr() as *const u32,
                    b_q_weight_ptr as *mut u32,
                    k as i64,
                    n as i64,
                )
            }
        };
        check_rc(rc, "w34_marlin_repack")?;
        Ok(b_q_weight_ptr)
    }

    #[cfg(feature = "cuda")]
    pub(crate) fn upload(
        &self,
        arena: &mut DeviceArena,
        stream: &CudaStream,
        repack_staging: &DeviceBuffer,
    ) -> Result<LinearProj> {
        let marlin_geom_ok = |n: usize, k: usize| {
            (n.is_multiple_of(MARLIN_TILE_N) && k.is_multiple_of(MARLIN_TILE_K_W4))
                || (n.is_multiple_of(MARLIN_TILE_K_W4) && k.is_multiple_of(MARLIN_TILE_N))
        };
        match self {
            Self::Hadamard { inner, block_size } => {
                inner.upload_hadamard(*block_size, arena, stream, repack_staging)
            }
            Self::U2B2NMajor {
                packed,
                scales,
                n,
                k,
                group_size,
            } => {
                let channelwise = *group_size == *k && *group_size > MARLIN_TILE_K_W4;
                if marlin_geom_ok(*n, *k)
                    && k.is_multiple_of(64)
                    && (channelwise || matches!(*group_size, 32 | 64 | 128))
                {
                    let b_q_weight_ptr = Self::repack_with_staging(
                        packed,
                        *n,
                        *k,
                        2,
                        arena,
                        stream,
                        repack_staging,
                    )?;
                    let perm_scales = marlin_permute_scales_from_n_major(
                        scales,
                        *k,
                        *n,
                        if channelwise { 0 } else { *group_size },
                    );
                    let b_scales_ptr = arena.upload_slice(&perm_scales, stream)?;
                    Ok(LinearProj::MarlinW2A16 {
                        b_q_weight_ptr,
                        b_scales_ptr,
                        size_n: *n,
                        size_k: *k,
                        group_blocks: if channelwise {
                            MARLIN_CHANNELWISE_GROUP_BLOCKS
                        } else {
                            (*group_size / MARLIN_GROUP_BLOCK_K) as i32
                        },
                    })
                } else {
                    let f16_w = dequant_u2b2_n_major_to_f16(packed, scales, *n, *k, *group_size);
                    let weight_ptr = arena.upload_slice(&f16_w, stream)?;
                    Ok(LinearProj::DenseF16 {
                        weight_ptr,
                        size_n: *n,
                        size_k: *k,
                    })
                }
            }
            Self::U4B8NMajor {
                packed,
                scales,
                n,
                k,
                group_size,
            } => {
                let channelwise = *group_size == *k && *group_size > MARLIN_TILE_K_W4;
                if marlin_geom_ok(*n, *k) && (channelwise || matches!(*group_size, 32 | 64 | 128)) {
                    let b_q_weight_ptr = Self::repack_with_staging(
                        packed,
                        *n,
                        *k,
                        4,
                        arena,
                        stream,
                        repack_staging,
                    )?;
                    let perm_scales = marlin_permute_scales_from_n_major(
                        scales,
                        *k,
                        *n,
                        if channelwise { 0 } else { *group_size },
                    );
                    let b_scales_ptr = arena.upload_slice(&perm_scales, stream)?;
                    Ok(LinearProj::MarlinW4A16 {
                        b_q_weight_ptr,
                        b_scales_ptr,
                        size_n: *n,
                        size_k: *k,
                        group_blocks: if channelwise {
                            MARLIN_CHANNELWISE_GROUP_BLOCKS
                        } else {
                            (*group_size / MARLIN_GROUP_BLOCK_K) as i32
                        },
                    })
                } else {
                    let f16_w = dequant_u4b8_n_major_to_f16(packed, scales, *n, *k, *group_size);
                    let weight_ptr = arena.upload_slice(&f16_w, stream)?;
                    Ok(LinearProj::DenseF16 {
                        weight_ptr,
                        size_n: *n,
                        size_k: *k,
                    })
                }
            }
            Self::U4ZpNMajor {
                packed,
                scales,
                qzeros,
                n,
                k,
                group_size,
            } => {
                if marlin_geom_ok(*n, *k) && matches!(*group_size, 32 | 64 | 128) {
                    let b_q_weight_ptr = Self::repack_with_staging(
                        packed,
                        *n,
                        *k,
                        4,
                        arena,
                        stream,
                        repack_staging,
                    )?;
                    let perm_scales =
                        marlin_permute_scales_from_n_major(scales, *k, *n, *group_size);
                    let b_scales_ptr = arena.upload_slice(&perm_scales, stream)?;
                    let perm_zeros =
                        marlin_permute_zero_points_from_n_major(qzeros, *k, *n, *group_size);
                    let b_zeros_ptr = arena.upload_slice(&perm_zeros, stream)?;
                    Ok(LinearProj::MarlinW4A16Zp {
                        b_q_weight_ptr,
                        b_scales_ptr,
                        b_zeros_ptr,
                        size_n: *n,
                        size_k: *k,
                        group_blocks: (*group_size / 16) as i32,
                    })
                } else {
                    let f16_w =
                        dequant_u4_zp_n_major_to_f16(packed, scales, qzeros, *n, *k, *group_size);
                    let weight_ptr = arena.upload_slice(&f16_w, stream)?;
                    Ok(LinearProj::DenseF16 {
                        weight_ptr,
                        size_n: *n,
                        size_k: *k,
                    })
                }
            }
            Self::U8B128NMajor {
                packed,
                scales,
                n,
                k,
                group_size,
                channelwise,
            } => {
                let gs_ok = *channelwise || matches!(*group_size, 32 | 64 | 128);
                let eff_gs = if *channelwise { *k } else { *group_size };
                if marlin_geom_ok(*n, *k) && gs_ok {
                    let b_q_weight_ptr = Self::repack_with_staging(
                        packed,
                        *n,
                        *k,
                        8,
                        arena,
                        stream,
                        repack_staging,
                    )?;
                    let perm_gs = if *channelwise { 0 } else { *group_size };
                    let perm_scales = marlin_permute_scales_from_n_major(scales, *k, *n, perm_gs);
                    let b_scales_ptr = arena.upload_slice(&perm_scales, stream)?;
                    let group_blocks = if *channelwise {
                        -1
                    } else {
                        (*group_size / 16) as i32
                    };
                    Ok(LinearProj::MarlinW8A16 {
                        b_q_weight_ptr,
                        b_scales_ptr,
                        size_n: *n,
                        size_k: *k,
                        group_blocks,
                    })
                } else {
                    let f16_w = dequant_u8b128_n_major_to_f16(packed, scales, *n, *k, eff_gs);
                    let weight_ptr = arena.upload_slice(&f16_w, stream)?;
                    Ok(LinearProj::DenseF16 {
                        weight_ptr,
                        size_n: *n,
                        size_k: *k,
                    })
                }
            }
            Self::DenseF16 { weight, n, k } => {
                let weight_ptr = arena.upload_slice(weight, stream)?;
                Ok(LinearProj::DenseF16 {
                    weight_ptr,
                    size_n: *n,
                    size_k: *k,
                })
            }
        }
    }
}

impl RawLinearHost {
    pub(super) fn size_check(&self) -> usize {
        self.dims().0
    }
}

mod hgptq;

mod layers;
pub use layers::*;

mod moe;
pub use moe::*;
