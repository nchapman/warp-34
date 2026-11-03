//! Host quantization, weight loading, and device linear/FFN projections.

use super::*;

#[cfg(feature = "cuda")]
pub enum LinearProj {
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
            Self::MarlinW4A16 { size_n, .. }
            | Self::MarlinW4A16Zp { size_n, .. }
            | Self::MarlinW8A16 { size_n, .. }
            | Self::DenseF16 { size_n, .. } => *size_n,
        }
    }

    pub fn size_k(&self) -> usize {
        match self {
            Self::MarlinW4A16 { size_k, .. }
            | Self::MarlinW4A16Zp { size_k, .. }
            | Self::MarlinW8A16 { size_k, .. }
            | Self::DenseF16 { size_k, .. } => *size_k,
        }
    }

    pub fn byte_footprint(&self) -> usize {
        match self {
            Self::MarlinW4A16 {
                size_n,
                size_k,
                group_blocks,
                ..
            } => {
                let g = (*group_blocks as usize) * 16;
                (size_n * size_k / 2) + (size_n * (size_k / g) * 2)
            }
            Self::MarlinW4A16Zp {
                size_n,
                size_k,
                group_blocks,
                ..
            } => {
                let g = (*group_blocks as usize) * 16;
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
                    size_k / ((*group_blocks as usize) * 16)
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
        match *self {
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
                // Small-N projections (GDN in_proj_ba, size_n <= 256) use a single
                // batch-size-invariant warp-reduction kernel to avoid per-row cuBLAS
                // split-K launches and guarantee 0-ULP ssm_state parity across all M.
                if size_n <= 256 && size_k.is_multiple_of(8) {
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
            Self::U4B8NMajor { n, k, .. }
            | Self::U4ZpNMajor { n, k, .. }
            | Self::U8B128NMajor { n, k, .. }
            | Self::DenseF16 { n, k, .. } => (*n, *k),
        }
    }

    pub(crate) fn to_f16_vec(&self) -> Vec<u16> {
        match self {
            Self::DenseF16 { weight, .. } => weight.clone(),
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
            for g in 0..num_groups {
                for c in 0..n {
                    scales_n_kg[c * num_groups + g] = scales_kg_n[g * n + c];
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

    pub(crate) fn slice_cols(&self, col_start: usize, col_end: usize) -> Self {
        let out_k = col_end - col_start;
        match self {
            Self::U4B8NMajor {
                packed,
                scales,
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
                Self::U4B8NMajor {
                    packed: new_packed,
                    scales: new_scales,
                    n: *n,
                    k: out_k,
                    group_size: *group_size,
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
            (n.is_multiple_of(64) && k.is_multiple_of(128))
                || (n.is_multiple_of(128) && k.is_multiple_of(64))
        };
        match self {
            Self::U4B8NMajor {
                packed,
                scales,
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
                    Ok(LinearProj::MarlinW4A16 {
                        b_q_weight_ptr,
                        b_scales_ptr,
                        size_n: *n,
                        size_k: *k,
                        group_blocks: (*group_size / 16) as i32,
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

#[cfg(feature = "cuda")]
pub struct QsaIndexerWeights {
    pub index_qk_proj: LinearProj,
    pub q_layernorm_ptr: u64,
    pub k_layernorm_ptr: u64,
}

#[cfg(feature = "cuda")]
pub struct FullAttentionLayerWeights {
    pub layer_id: usize,
    pub kv_layer_ordinal: usize,
    pub qkv_proj: LinearProj,
    pub o_proj: LinearProj,
    pub q_norm_ptr: u64,
    pub k_norm_ptr: u64,
    pub qsa_indexer: Option<QsaIndexerWeights>,
}

#[cfg(feature = "cuda")]
impl FullAttentionLayerWeights {
    /// Runs QSA pre-indexer norm+RoPE, updates raw/compressed indexer key caches,
    /// optionally selects top-k sparse blocks, and executes sparse paged GQA + sigmoid output gate.
    #[allow(clippy::too_many_arguments)]
    pub fn forward_qsa_attention(
        &self,
        qsa_idx: &QsaIndexerWeights,
        ex: &W34Exec,
        cublas: &CublasHandle,
        marlin_ws_ptr: u64,
        marlin_ws_ints: usize,
        cfg: &ModelConfig,
        kv_pool: &crate::cache::KvPool,
        kv_layout: &ffi::W34KvLayerLayout,
        attn_batch: &W34AttentionBatch,
        normed_buf: u64,
        q_buf: u64,
        gate_buf: u64,
        gated_attn_buf: u64,
        [
            qsa_qk_buf,
            qsa_q_buf,
            qsa_k_buf,
            qsa_sel_phys_buf,
            qsa_sel_log_buf,
            qsa_num_sel_buf,
            qsa_seq_pos_buf,
        ]: [u64; 7],
        positions_ptr: u64,
        pos_plane_stride: i32,
        num_tokens: usize,
        local_q_heads: usize,
        skip_topk: bool,
    ) -> Result<()> {
        let kv_bufs = &kv_pool.layers[self.kv_layer_ordinal];
        qsa_idx.index_qk_proj.forward(
            ex,
            cublas,
            marlin_ws_ptr,
            marlin_ws_ints,
            normed_buf,
            qsa_qk_buf,
            num_tokens,
        )?;
        crate::ple::gpu_ops::qsa_indexer_norm_rope_f16(
            ex,
            qsa_qk_buf,
            qsa_idx.q_layernorm_ptr,
            qsa_idx.k_layernorm_ptr,
            positions_ptr,
            qsa_q_buf,
            qsa_k_buf,
            0,
            num_tokens,
            cfg.indexer_n_heads,
            cfg.indexer_kv_heads,
            cfg.indexer_head_dim,
            cfg.rotary_dim,
            cfg.rope_theta,
            cfg.rms_norm_eps,
            cfg.mrope_section[0],
            cfg.mrope_section[1],
            pos_plane_stride as usize,
        )?;
        let raw_k_cache_ptr = kv_bufs
            .qsa_raw_k_buf
            .as_ref()
            .ok_or_else(|| loader_err("missing qsa_raw_k_buf"))?
            .addr();
        let comp_k_cache_ptr = kv_bufs
            .qsa_comp_k_buf
            .as_ref()
            .ok_or_else(|| loader_err("missing qsa_comp_k_buf"))?
            .addr();
        crate::ple::gpu_ops::qsa_indexer_cache_update_f16(
            ex,
            qsa_k_buf,
            raw_k_cache_ptr,
            comp_k_cache_ptr,
            qsa_idx.k_layernorm_ptr,
            positions_ptr,
            attn_batch.write_slot_mapping as usize as u64,
            0,
            num_tokens,
            cfg.indexer_head_dim,
            cfg.rotary_dim,
            cfg.rope_theta,
            cfg.rms_norm_eps,
            cfg.mrope_section[0],
            cfg.mrope_section[1],
            pos_plane_stride as usize,
        )?;
        let select_k_blocks =
            (cfg.indexer_budget / cfg.indexer_compress_ratio.max(1)).clamp(1, 512) + 1;
        if !skip_topk {
            crate::ple::gpu_ops::qsa_select_blocks_f16(
                ex,
                qsa_q_buf,
                comp_k_cache_ptr,
                attn_batch,
                qsa_sel_phys_buf,
                qsa_sel_log_buf,
                qsa_num_sel_buf,
                qsa_seq_pos_buf,
                cfg.indexer_n_heads,
                cfg.indexer_head_dim,
                16,
                select_k_blocks,
                0,
                0,
            )?;
        }
        crate::ple::gpu_ops::qsa_sparse_paged_attn_f16(
            ex,
            q_buf,
            gate_buf,
            kv_layout,
            qsa_sel_phys_buf,
            qsa_sel_log_buf,
            qsa_num_sel_buf,
            qsa_seq_pos_buf,
            0,
            gated_attn_buf,
            num_tokens,
            local_q_heads,
            select_k_blocks,
            attn_batch.sm_scale,
        )
    }
}

#[cfg(feature = "cuda")]
pub struct HcBlockWeights {
    pub hc_norm_ptr: u64,
    pub input_mix_weight_down: LinearProj,
    pub input_mix_weight_up: LinearProj,
    pub block_inject_weight: Option<LinearProj>,
}

#[cfg(feature = "cuda")]
impl HcBlockWeights {
    #[allow(clippy::too_many_arguments)]
    pub fn forward_mix(
        &self,
        ex: &W34Exec,
        cublas: &CublasHandle,
        marlin_ws_ptr: u64,
        marlin_ws_ints: usize,
        hc_residual_buf: u64,
        out_mixed_buf: u64,
        hc_inj_weights_buf: u64,
        ple_buf_a: u64,
        ple_buf_b: u64,
        ple_val_buf: u64,
        num_tokens: usize,
        hc_count: usize,
        hc_lowrank: usize,
        h_dim: usize,
        eps: f32,
    ) -> Result<()> {
        let low_dim = hc_lowrank;
        let hc_down_buf = ple_val_buf;
        let hc_inj_raw_buf = ple_val_buf + (num_tokens * low_dim * 2) as u64;
        crate::ple::gpu_ops::hc_rmsnorm_f16(
            ex,
            hc_residual_buf,
            self.hc_norm_ptr,
            ple_buf_a,
            0,
            num_tokens,
            hc_count,
            h_dim,
            eps,
        )?;
        self.input_mix_weight_down.forward(
            ex,
            cublas,
            marlin_ws_ptr,
            marlin_ws_ints,
            ple_buf_a,
            hc_down_buf,
            num_tokens,
        )?;
        let (inj_raw_ptr, inj_w_ptr) = if let Some(inj_proj) = &self.block_inject_weight {
            inj_proj.forward(
                ex,
                cublas,
                marlin_ws_ptr,
                marlin_ws_ints,
                ple_buf_a,
                hc_inj_raw_buf,
                num_tokens,
            )?;
            (hc_inj_raw_buf, hc_inj_weights_buf)
        } else {
            (0, 0)
        };
        crate::ple::gpu_ops::hc_down_silu_inject_f16(
            ex,
            hc_down_buf,
            hc_down_buf,
            inj_raw_ptr,
            inj_w_ptr,
            0,
            num_tokens,
            low_dim,
            hc_count,
        )?;
        self.input_mix_weight_up.forward(
            ex,
            cublas,
            marlin_ws_ptr,
            marlin_ws_ints,
            hc_down_buf,
            ple_buf_b,
            num_tokens,
        )?;
        crate::ple::gpu_ops::hc_mix_streams_f16(
            ex,
            ple_buf_a,
            ple_buf_b,
            out_mixed_buf,
            0,
            num_tokens,
            hc_count,
            h_dim,
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub fn forward_combine(
        &self,
        ex: &W34Exec,
        block_out_buf: u64,
        hc_inj_weights_buf: u64,
        hc_residual_buf: u64,
        num_tokens: usize,
        hc_count: usize,
        h_dim: usize,
    ) -> Result<()> {
        crate::ple::gpu_ops::hc_combine_f16(
            ex,
            hc_residual_buf,
            block_out_buf,
            hc_inj_weights_buf,
            0,
            hc_residual_buf,
            0,
            num_tokens,
            hc_count,
            h_dim,
        )
    }
}

/// Per-layer residual stream topology and normalization weights.
#[cfg(feature = "cuda")]
// One value per layer; inline weights avoid extra indirection in the forward loop.
#[allow(clippy::large_enum_variant)]
pub enum LayerResidual {
    /// Single-stream `[num_tokens, h_dim]` residual with `GemmaRMSNorm` (Qwen 3.5 / 3.6).
    SingleStream {
        input_layernorm_ptr: u64,
        post_attention_layernorm_ptr: u64,
    },
    /// Multi-stream `[num_tokens, hc_count * h_dim]` HyperConnection residual (Qwen 3.8 / 4).
    HyperConnection {
        attn_hc: HcBlockWeights,
        mlp_hc: HcBlockWeights,
    },
}

#[cfg(feature = "cuda")]
impl LayerResidual {
    pub fn single_stream_input_norm_ptr(&self) -> Option<u64> {
        match self {
            Self::SingleStream {
                input_layernorm_ptr,
                ..
            } => Some(*input_layernorm_ptr),
            Self::HyperConnection { .. } => None,
        }
    }

    pub fn single_stream_post_attn_norm_ptr(&self) -> Option<u64> {
        match self {
            Self::SingleStream {
                post_attention_layernorm_ptr,
                ..
            } => Some(*post_attention_layernorm_ptr),
            Self::HyperConnection { .. } => None,
        }
    }

    /// Runs the pre-attention normalization / stream mixing step.
    /// On `SingleStream`, `layer_idx > 0` is a no-op because the preceding layer's
    /// `post_mlp` fused the residual addition with this layer's `input_layernorm`.
    #[allow(clippy::too_many_arguments)]
    pub fn pre_attn(
        &self,
        layer_idx: usize,
        ex: &W34Exec,
        cublas: &CublasHandle,
        marlin_ws_ptr: u64,
        marlin_ws_ints: usize,
        residual_buf: u64,
        hc_residual_buf: u64,
        normed_buf: u64,
        hc_inj_weights_buf: u64,
        ple_buf_a: u64,
        ple_buf_b: u64,
        ple_val_buf: u64,
        num_tokens: usize,
        hc_count: usize,
        hc_lowrank: usize,
        h_dim: usize,
        eps: f32,
    ) -> Result<()> {
        match self {
            Self::SingleStream {
                input_layernorm_ptr,
                ..
            } => {
                if layer_idx == 0 {
                    check_rc(
                        // SAFETY: `residual_buf` and `normed_buf` are disjoint `num_tokens * h_dim`
                        // FP16 buffers; `input_layernorm_ptr` holds `h_dim` FP16 elements.
                        unsafe {
                            ffi::w34_gemma_rms_norm_f16(
                                ex,
                                residual_buf as *const c_void,
                                *input_layernorm_ptr as *const c_void,
                                normed_buf as *mut c_void,
                                core::ptr::null(),
                                num_tokens as i32,
                                h_dim as i32,
                                eps,
                            )
                        },
                        "w34_gemma_rms_norm_f16",
                    )?;
                }
                Ok(())
            }
            Self::HyperConnection { attn_hc, .. } => attn_hc.forward_mix(
                ex,
                cublas,
                marlin_ws_ptr,
                marlin_ws_ints,
                hc_residual_buf,
                normed_buf,
                hc_inj_weights_buf,
                ple_buf_a,
                ple_buf_b,
                ple_val_buf,
                num_tokens,
                hc_count,
                hc_lowrank,
                h_dim,
                eps,
            ),
        }
    }

    /// Runs the post-attention residual update and pre-MLP normalization / stream mixing step.
    #[allow(clippy::too_many_arguments)]
    pub fn post_attn_pre_mlp(
        &self,
        ex: &W34Exec,
        cublas: &CublasHandle,
        marlin_ws_ptr: u64,
        marlin_ws_ints: usize,
        tp2: Option<&Tp2Collectives>,
        rank: usize,
        block_out_buf: u64,
        residual_buf: u64,
        hc_residual_buf: u64,
        normed_buf: u64,
        hc_inj_weights_buf: u64,
        ple_buf_a: u64,
        ple_buf_b: u64,
        ple_val_buf: u64,
        num_tokens: usize,
        hc_count: usize,
        hc_lowrank: usize,
        h_dim: usize,
        eps: f32,
        preserve_block_out_on_tp2: bool,
    ) -> Result<()> {
        match self {
            Self::SingleStream {
                post_attention_layernorm_ptr,
                ..
            } => {
                if let Some(comm) = tp2 {
                    let ar_out = if preserve_block_out_on_tp2 {
                        block_out_buf
                    } else {
                        0
                    };
                    comm.all_reduce_residual_gemma_rms_norm_f16(
                        ex,
                        block_out_buf,
                        ar_out,
                        residual_buf,
                        *post_attention_layernorm_ptr,
                        normed_buf,
                        0,
                        rank as i32,
                        num_tokens as i32,
                        h_dim as i32,
                        eps,
                        0,
                        0,
                    )
                } else {
                    check_rc(
                        // SAFETY: `block_out_buf`, `residual_buf`, and `normed_buf` each hold
                        // `num_tokens * h_dim` FP16 elements.
                        unsafe {
                            ffi::w34_fused_add_gemma_rms_norm_f16(
                                ex,
                                block_out_buf as *const c_void,
                                residual_buf as *mut c_void,
                                *post_attention_layernorm_ptr as *const c_void,
                                normed_buf as *mut c_void,
                                core::ptr::null(),
                                num_tokens as i32,
                                h_dim as i32,
                                eps,
                            )
                        },
                        "w34_fused_add_gemma_rms_norm_f16 (post_attn)",
                    )
                }
            }
            Self::HyperConnection { attn_hc, mlp_hc } => {
                if let Some(comm) = tp2 {
                    comm.all_reduce_f16(
                        ex,
                        block_out_buf,
                        block_out_buf,
                        rank as i32,
                        (num_tokens * h_dim) as i32,
                    )?;
                }
                attn_hc.forward_combine(
                    ex,
                    block_out_buf,
                    hc_inj_weights_buf,
                    hc_residual_buf,
                    num_tokens,
                    hc_count,
                    h_dim,
                )?;
                mlp_hc.forward_mix(
                    ex,
                    cublas,
                    marlin_ws_ptr,
                    marlin_ws_ints,
                    hc_residual_buf,
                    normed_buf,
                    hc_inj_weights_buf,
                    ple_buf_a,
                    ple_buf_b,
                    ple_val_buf,
                    num_tokens,
                    hc_count,
                    hc_lowrank,
                    h_dim,
                    eps,
                )
            }
        }
    }

    /// Runs the post-MLP residual update (fusing the next layer's input norm or final norm on `SingleStream`).
    #[allow(clippy::too_many_arguments)]
    pub fn post_mlp(
        &self,
        ex: &W34Exec,
        tp2: Option<&Tp2Collectives>,
        rank: usize,
        mlp_out_buf: u64,
        residual_buf: u64,
        hc_residual_buf: u64,
        hc_inj_weights_buf: u64,
        next_norm_weight: u64,
        next_norm_out: u64,
        tap_dst: u64,
        tap_stride: i32,
        tap_offset: i32,
        num_tokens: usize,
        hc_count: usize,
        h_dim: usize,
        eps: f32,
    ) -> Result<()> {
        match self {
            Self::SingleStream { .. } => {
                if let Some(comm) = tp2 {
                    comm.all_reduce_residual_gemma_rms_norm_f16(
                        ex,
                        mlp_out_buf,
                        0,
                        residual_buf,
                        next_norm_weight,
                        next_norm_out,
                        tap_dst,
                        rank as i32,
                        num_tokens as i32,
                        h_dim as i32,
                        eps,
                        tap_stride,
                        tap_offset,
                    )
                } else {
                    check_rc(
                        // SAFETY: `mlp_out_buf`, `residual_buf`, and `next_norm_out` hold
                        // `num_tokens * h_dim` FP16 elements; `tap_dst` is 0 or a validated BF16 tap buffer.
                        unsafe {
                            ffi::w34_fused_add_gemma_rms_norm_tap_f16(
                                ex,
                                mlp_out_buf as *const c_void,
                                residual_buf as *mut c_void,
                                next_norm_weight as *const c_void,
                                next_norm_out as *mut c_void,
                                tap_dst as *mut c_void,
                                core::ptr::null(),
                                num_tokens as i32,
                                h_dim as i32,
                                eps,
                                tap_stride,
                                tap_offset,
                            )
                        },
                        "w34_fused_add_gemma_rms_norm_tap_f16 (post_mlp)",
                    )
                }
            }
            Self::HyperConnection { mlp_hc, .. } => {
                if let Some(comm) = tp2 {
                    comm.all_reduce_f16(
                        ex,
                        mlp_out_buf,
                        mlp_out_buf,
                        rank as i32,
                        (num_tokens * h_dim) as i32,
                    )?;
                }
                mlp_hc.forward_combine(
                    ex,
                    mlp_out_buf,
                    hc_inj_weights_buf,
                    hc_residual_buf,
                    num_tokens,
                    hc_count,
                    h_dim,
                )
            }
        }
    }
}

/// Final pre-LM-head residual normalization / stream mixer.
#[cfg(feature = "cuda")]
pub enum FinalResidual {
    /// Single-stream `GemmaRMSNorm` (`[num_tokens, h_dim] -> [num_tokens, h_dim]`).
    SingleStream { norm_ptr: u64 },
    /// Multi-stream HyperConnection mixer (`[num_tokens, hc_count * h_dim] -> [num_tokens, h_dim]`).
    HyperConnection { mixer: HcBlockWeights },
}

#[cfg(feature = "cuda")]
impl FinalResidual {
    pub fn single_stream_norm_ptr(&self) -> Option<u64> {
        match self {
            Self::SingleStream { norm_ptr } => Some(*norm_ptr),
            Self::HyperConnection { .. } => None,
        }
    }

    pub fn is_hyper_connection(&self) -> bool {
        matches!(self, Self::HyperConnection { .. })
    }

    /// Initializes the multi-stream residual buffer from `embed_buf` (`[num_tokens, h_dim]`)
    /// when `HyperConnection` is active; no-op on `SingleStream`.
    pub fn init_streams(
        &self,
        ex: &W34Exec,
        embed_buf: u64,
        hc_residual_buf: u64,
        num_tokens: usize,
        hc_count: usize,
        h_dim: usize,
    ) -> Result<()> {
        match self {
            Self::SingleStream { .. } => Ok(()),
            Self::HyperConnection { .. } => crate::ple::gpu_ops::hc_init_streams_f16(
                ex,
                embed_buf,
                hc_residual_buf,
                0,
                num_tokens,
                hc_count,
                h_dim,
            ),
        }
    }

    /// Completes the final normalization / stream collapse after the layer stack.
    /// On `SingleStream`, the last layer's `post_mlp` already wrote `final_norm_out_buf`.
    /// On `HyperConnection`, exports `hc_residual_buf` (`[num_tokens, hc_count * h_dim]`) to
    /// `last_final_hidden_f16` (if enabled) and collapses `hc_residual_buf -> final_norm_out_buf`.
    #[allow(clippy::too_many_arguments)]
    pub fn finish_stack(
        &self,
        ex: &W34Exec,
        stream: &CudaStream,
        cublas: &CublasHandle,
        marlin_ws_ptr: u64,
        marlin_ws_ints: usize,
        hc_residual_buf: u64,
        final_norm_out_buf: u64,
        last_final_hidden_f16: Option<&DeviceBuffer>,
        ple_buf_a: u64,
        ple_buf_b: u64,
        ple_val_buf: u64,
        num_tokens: usize,
        hc_count: usize,
        hc_lowrank: usize,
        h_dim: usize,
        eps: f32,
    ) -> Result<()> {
        match self {
            Self::SingleStream { .. } => Ok(()),
            Self::HyperConnection { mixer } => {
                if let Some(final_h_buf) = last_final_hidden_f16
                    && num_tokens * hc_count * h_dim * 2 <= final_h_buf.len_bytes
                {
                    check_rc(
                        // SAFETY: `hc_residual_buf` and `final_h_buf` hold at least
                        // `num_tokens * hc_count * h_dim * 2` bytes.
                        unsafe {
                            ffi::w34_memcpy_d2d_async(
                                final_h_buf.addr() as *mut c_void,
                                hc_residual_buf as *const c_void,
                                num_tokens * hc_count * h_dim * 2,
                                stream.handle,
                            )
                        },
                        "export pre-mixer last_final_hidden_f16",
                    )?;
                }
                mixer.forward_mix(
                    ex,
                    cublas,
                    marlin_ws_ptr,
                    marlin_ws_ints,
                    hc_residual_buf,
                    final_norm_out_buf,
                    0,
                    ple_buf_a,
                    ple_buf_b,
                    ple_val_buf,
                    num_tokens,
                    hc_count,
                    hc_lowrank,
                    h_dim,
                    eps,
                )
            }
        }
    }
}

#[cfg(feature = "cuda")]
pub struct PleLayerWeights {
    pub key_proj: LinearProj,
    pub value_proj: LinearProj,
    pub norm_key_ptr: u64,
    pub norm_query_ptr: u64,
    pub norm_conv_ptr: u64,
    pub conv1d_weight_ptr: u64,
    pub weight_scale: f32,
}

#[cfg(feature = "cuda")]
impl PleLayerWeights {
    #[allow(clippy::too_many_arguments)]
    pub fn forward(
        &self,
        ex: &W34Exec,
        cublas: &CublasHandle,
        marlin_ws_ptr: u64,
        marlin_ws_ints: usize,
        ple_f8_staging_ptr: u64,
        hc_residual_buf: u64,
        normed_buf: u64,
        ple_buf_a: u64,
        ple_buf_b: u64,
        ple_val_buf: u64,
        ple_conv_state_ptr: u64,
        gdn_batch: &ffi::W34GdnBatch,
        num_tokens: usize,
        hc_count: usize,
        h_dim: usize,
        eps: f32,
    ) -> Result<()> {
        let total_dim = hc_count * h_dim;
        let ple_embed_dim = self.key_proj.size_k();
        crate::ple::gpu_ops::ple_dequant_f8_to_f16(
            ex,
            ple_f8_staging_ptr,
            ple_val_buf,
            0,
            num_tokens,
            ple_embed_dim,
            self.weight_scale,
        )?;
        self.key_proj.forward(
            ex,
            cublas,
            marlin_ws_ptr,
            marlin_ws_ints,
            ple_val_buf,
            ple_buf_a,
            num_tokens,
        )?;
        self.value_proj.forward(
            ex,
            cublas,
            marlin_ws_ptr,
            marlin_ws_ints,
            ple_val_buf,
            normed_buf,
            num_tokens,
        )?;
        crate::ple::gpu_ops::hc_rmsnorm_f16(
            ex,
            ple_buf_a,
            self.norm_key_ptr,
            ple_buf_a,
            0,
            num_tokens,
            hc_count,
            h_dim,
            eps,
        )?;
        crate::ple::gpu_ops::hc_rmsnorm_f16(
            ex,
            hc_residual_buf,
            self.norm_query_ptr,
            ple_buf_b,
            0,
            num_tokens,
            hc_count,
            h_dim,
            eps,
        )?;
        crate::ple::gpu_ops::ple_signed_sqrt_gate_f16(
            ex, ple_buf_b, ple_buf_a, normed_buf, ple_buf_a, 0, num_tokens, hc_count, h_dim, 1e-6,
        )?;
        check_rc(
            // SAFETY: hc_residual_buf and ple_buf_a each hold num_tokens * total_dim FP16 elements.
            unsafe {
                ffi::w34_add_inplace_f16(
                    ex,
                    hc_residual_buf as *mut c_void,
                    ple_buf_a as *const c_void,
                    (num_tokens * total_dim) as i64,
                )
            },
            "w34_add_inplace_f16 (ple gated_value)",
        )?;
        crate::ple::gpu_ops::hc_rmsnorm_f16(
            ex,
            ple_buf_a,
            self.norm_conv_ptr,
            ple_buf_b,
            0,
            num_tokens,
            hc_count,
            h_dim,
            eps,
        )?;
        crate::ple::gpu_ops::ple_dilated_conv1d_f16(
            ex,
            ple_buf_b,
            self.conv1d_weight_ptr,
            ple_conv_state_ptr,
            ple_buf_a,
            gdn_batch,
            total_dim,
        )?;
        check_rc(
            // SAFETY: hc_residual_buf and ple_buf_a each hold num_tokens * total_dim FP16 elements.
            unsafe {
                ffi::w34_add_inplace_f16(
                    ex,
                    hc_residual_buf as *mut c_void,
                    ple_buf_a as *const c_void,
                    (num_tokens * total_dim) as i64,
                )
            },
            "w34_add_inplace_f16 (ple conv_out)",
        )
    }
}

#[cfg(feature = "cuda")]
pub struct GdnLayerWeights {
    pub layer_id: usize,
    pub gdn_layer_ordinal: usize,
    pub in_proj_qkvz: LinearProj,
    pub in_proj_ba: LinearProj,
    pub out_proj: LinearProj,
    pub conv_weight_ptr: u64,
    pub a_log_ptr: u64,
    pub dt_bias_ptr: u64,
    pub norm_weight_ptr: u64,
}

#[cfg(feature = "cuda")]
pub enum LayerBlockWeights {
    FullAttention(FullAttentionLayerWeights),
    LinearAttention(GdnLayerWeights),
}

#[cfg(feature = "cuda")]
pub enum MoeExpertStorage {
    W4A16 {
        gate_up_packed_ptr: u64,
        gate_up_scales_ptr: u64,
        down_packed_ptr: u64,
        down_scales_ptr: u64,
        group_size: usize,
    },
    W4A16Zp {
        gate_up_packed_ptr: u64,
        gate_up_scales_ptr: u64,
        gate_up_qzeros_ptr: u64,
        down_packed_ptr: u64,
        down_scales_ptr: u64,
        down_qzeros_ptr: u64,
        group_size: usize,
    },
    DenseF16 {
        gate_up_ptr: u64,
        down_ptr: u64,
    },
    /// Host/VRAM tiered expert storage managed by `RankTieredMoe` (or MTP's tiered workspace).
    Tiered {
        layer_id: usize,
    },
}

#[cfg(feature = "cuda")]
pub struct MoeLayerWeights {
    pub router_gate: LinearProj,
    pub experts: MoeExpertStorage,
    pub num_experts: usize,
    pub top_k: usize,
    pub norm_topk_prob: bool,
    pub scoring_func: i32,
    pub local_moe_interm: usize,
    pub shared_gate_up_proj: Option<LinearProj>,
    pub shared_down_proj: Option<LinearProj>,
    pub shared_expert_gate: Option<LinearProj>,
    pub local_shared_interm: usize,
}

#[cfg(feature = "cuda")]
pub enum FfnBlockWeights {
    Dense {
        gate_up_proj: LinearProj,
        down_proj: LinearProj,
    },
    Moe(Box<MoeLayerWeights>),
}

#[cfg(feature = "cuda")]
impl FfnBlockWeights {
    /// Enqueue the loaded FFN using the model's phase-local scratch layout.
    /// Raw buffer extents, device affinity, and ordering are caller obligations;
    /// accepting u64 addresses here does not validate arbitrary allocations.
    #[allow(clippy::too_many_arguments)]
    pub fn forward(
        &self,
        ex: &W34Exec,
        cublas: &CublasHandle,
        marlin_ws_ptr: u64,
        marlin_ws_ints: usize,
        normed_buf: u64,
        mlp_out_buf: u64,
        num_tokens: usize,
        h_dim: usize,
        local_interm: usize,
        gate_up_buf: u64,
        act_buf: u64,
        moe_router_logits_buf: u64,
        moe_topk_weights_buf: u64,
        moe_topk_ids_buf: u64,
        moe_interm_buf: u64,
        moe_sh_gu_buf: u64,
        moe_sh_act_buf: u64,
        moe_sh_out_buf: u64,
        moe_sh_gate_buf: u64,
    ) -> Result<()> {
        match self {
            Self::Dense {
                gate_up_proj,
                down_proj,
            } => {
                gate_up_proj.forward(
                    ex,
                    cublas,
                    marlin_ws_ptr,
                    marlin_ws_ints,
                    normed_buf,
                    gate_up_buf,
                    num_tokens,
                )?;
                check_rc(
                    // SAFETY: the model's dense FFN phase reserves [M,2*interm]
                    // gate/up and [M,interm] activation regions. The projection
                    // precedes this launch on ex.stream; these addresses are supplied
                    // by the caller rather than capacity-checked by this method.
                    unsafe {
                        ffi::w34_silu_and_mul_f16(
                            ex,
                            gate_up_buf as *const c_void,
                            act_buf as *mut c_void,
                            core::ptr::null(),
                            num_tokens as i32,
                            local_interm as i32,
                        )
                    },
                    "w34_silu_and_mul_f16",
                )?;
                down_proj.forward(
                    ex,
                    cublas,
                    marlin_ws_ptr,
                    marlin_ws_ints,
                    act_buf,
                    mlp_out_buf,
                    num_tokens,
                )
            }
            Self::Moe(moe) => {
                let fused_dual_small_n = match (
                    &moe.router_gate,
                    &moe.shared_expert_gate,
                    moe.shared_gate_up_proj.is_some() && moe.shared_down_proj.is_some(),
                ) {
                    (
                        LinearProj::DenseF16 {
                            weight_ptr: w0,
                            size_n: n0,
                            size_k: k0,
                        },
                        Some(LinearProj::DenseF16 {
                            weight_ptr: w1,
                            size_n: n1,
                            size_k: k1,
                        }),
                        true,
                    ) if *n0 <= 256 && *n1 <= 256 && *k0 == *k1 && k0.is_multiple_of(8) => {
                        Some((*w0, *n0, *w1, *n1, *k0))
                    }
                    _ => None,
                };
                if let Some((w0, n0, w1, n1, k)) = fused_dual_small_n {
                    check_rc(
                        // SAFETY: the match checked both small-N geometries and
                        // equal K. Router/shared-gate weights come from upload;
                        // model scratch provides separate router and gate outputs.
                        unsafe {
                            ffi::w34_small_n_gemm_dual_f16(
                                ex,
                                normed_buf as *const c_void,
                                w0 as *const c_void,
                                moe_router_logits_buf as *mut c_void,
                                n0 as i32,
                                w1 as *const c_void,
                                moe_sh_gate_buf as *mut c_void,
                                n1 as i32,
                                num_tokens as i32,
                                k as i32,
                            )
                        },
                        "w34_small_n_gemm_dual_f16",
                    )?;
                } else {
                    moe.router_gate.forward(
                        ex,
                        cublas,
                        marlin_ws_ptr,
                        marlin_ws_ints,
                        normed_buf,
                        moe_router_logits_buf,
                        num_tokens,
                    )?;
                }
                check_rc(
                    // SAFETY: router projection precedes this on ex.stream.
                    // The model scratch layout reserves M*experts FP16 logits and
                    // M*top_k FP32 weights/i32 IDs for the loaded MoE geometry.
                    unsafe {
                        ffi::w34_moe_topk_router_f16(
                            ex,
                            moe_router_logits_buf as *const c_void,
                            moe_topk_weights_buf as *mut f32,
                            moe_topk_ids_buf as *mut i32,
                            num_tokens as i32,
                            moe.num_experts as i32,
                            moe.top_k as i32,
                            i32::from(moe.norm_topk_prob),
                            moe.scoring_func,
                        )
                    },
                    "w34_moe_topk_router_f16",
                )?;

                let (sh_out_ptr, sh_gate_ptr) = if let (Some(sh_gu), Some(sh_dn)) =
                    (&moe.shared_gate_up_proj, &moe.shared_down_proj)
                {
                    sh_gu.forward(
                        ex,
                        cublas,
                        marlin_ws_ptr,
                        marlin_ws_ints,
                        normed_buf,
                        moe_sh_gu_buf,
                        num_tokens,
                    )?;
                    check_rc(
                        // SAFETY: the shared projection just produced [M,2*shared]
                        // gate/up; model scratch reserves [M,shared] activation.
                        // Both regions are distinct within the FFN phase.
                        unsafe {
                            ffi::w34_silu_and_mul_f16(
                                ex,
                                moe_sh_gu_buf as *const c_void,
                                moe_sh_act_buf as *mut c_void,
                                core::ptr::null(),
                                num_tokens as i32,
                                moe.local_shared_interm as i32,
                            )
                        },
                        "w34_silu_and_mul_f16 (shared_expert)",
                    )?;
                    sh_dn.forward(
                        ex,
                        cublas,
                        marlin_ws_ptr,
                        marlin_ws_ints,
                        moe_sh_act_buf,
                        moe_sh_out_buf,
                        num_tokens,
                    )?;
                    let sg_ptr = if let Some(sh_gate) = &moe.shared_expert_gate {
                        if fused_dual_small_n.is_none() {
                            sh_gate.forward(
                                ex,
                                cublas,
                                marlin_ws_ptr,
                                marlin_ws_ints,
                                normed_buf,
                                moe_sh_gate_buf,
                                num_tokens,
                            )?;
                        }
                        moe_sh_gate_buf as *const c_void
                    } else {
                        core::ptr::null()
                    };
                    (moe_sh_out_buf as *const c_void, sg_ptr)
                } else {
                    (core::ptr::null(), core::ptr::null())
                };

                match &moe.experts {
                    MoeExpertStorage::W4A16 {
                        gate_up_packed_ptr,
                        gate_up_scales_ptr,
                        down_packed_ptr,
                        down_scales_ptr,
                        group_size,
                    } => check_rc(
                        // SAFETY: loader stores expert-major packed U4 weights and
                        // FP16 scales with the loaded group size. Top-k buffers were
                        // produced above; model scratch reserves routed intermediate
                        // and output rows. Shared inputs are null when absent.
                        unsafe {
                            ffi::w34_moe_w4a16_mlp_f16(
                                ex,
                                normed_buf as *const c_void,
                                *gate_up_packed_ptr as *const u32,
                                *gate_up_scales_ptr as *const c_void,
                                *down_packed_ptr as *const u32,
                                *down_scales_ptr as *const c_void,
                                moe_topk_ids_buf as *const i32,
                                moe_topk_weights_buf as *const f32,
                                sh_out_ptr,
                                sh_gate_ptr,
                                moe_interm_buf as *mut c_void,
                                mlp_out_buf as *mut c_void,
                                num_tokens as i32,
                                moe.top_k as i32,
                                moe.num_experts as i32,
                                moe.local_moe_interm as i32,
                                h_dim as i32,
                                *group_size as i32,
                            )
                        },
                        "w34_moe_w4a16_mlp_f16",
                    ),
                    MoeExpertStorage::W4A16Zp {
                        gate_up_packed_ptr,
                        gate_up_scales_ptr,
                        gate_up_qzeros_ptr,
                        down_packed_ptr,
                        down_scales_ptr,
                        down_qzeros_ptr,
                        group_size,
                    } => check_rc(
                        // SAFETY: expert-major U4 weights/scales/zero points use
                        // the same loaded geometry. Router outputs and shared inputs
                        // precede this on ex.stream; model scratch supplies separate
                        // routed intermediates/output, with null shared inputs allowed.
                        unsafe {
                            ffi::w34_moe_w4a16_mlp_zp_f16(
                                ex,
                                normed_buf as *const c_void,
                                *gate_up_packed_ptr as *const u32,
                                *gate_up_scales_ptr as *const c_void,
                                *gate_up_qzeros_ptr as *const u32,
                                *down_packed_ptr as *const u32,
                                *down_scales_ptr as *const c_void,
                                *down_qzeros_ptr as *const u32,
                                moe_topk_ids_buf as *const i32,
                                moe_topk_weights_buf as *const f32,
                                sh_out_ptr,
                                sh_gate_ptr,
                                moe_interm_buf as *mut c_void,
                                mlp_out_buf as *mut c_void,
                                num_tokens as i32,
                                moe.top_k as i32,
                                moe.num_experts as i32,
                                moe.local_moe_interm as i32,
                                h_dim as i32,
                                *group_size as i32,
                            )
                        },
                        "w34_moe_w4a16_mlp_zp_f16",
                    ),
                    MoeExpertStorage::DenseF16 {
                        gate_up_ptr,
                        down_ptr,
                    } => {
                        check_rc(
                            // SAFETY: dense expert weights are uploaded with this
                            // expert/intermediate/hidden geometry. Router IDs select
                            // experts; model scratch provides M*top_k*interm FP16
                            // intermediates distinct from the normalized input.
                            unsafe {
                                ffi::w34_moe_f16_gate_up_silu(
                                    ex,
                                    normed_buf as *const c_void,
                                    *gate_up_ptr as *const c_void,
                                    moe_topk_ids_buf as *const i32,
                                    moe_interm_buf as *mut c_void,
                                    num_tokens as i32,
                                    moe.top_k as i32,
                                    moe.num_experts as i32,
                                    moe.local_moe_interm as i32,
                                    h_dim as i32,
                                )
                            },
                            "w34_moe_f16_gate_up_silu",
                        )?;
                        check_rc(
                            // SAFETY: gate/up above produced routed intermediates;
                            // down weights use the matching expert geometry and
                            // output is M*h_dim FP16 in the model scratch layout.
                            // Same-stream ordering preserves router/shared inputs.
                            unsafe {
                                ffi::w34_moe_f16_down_accum(
                                    ex,
                                    moe_interm_buf as *const c_void,
                                    *down_ptr as *const c_void,
                                    moe_topk_ids_buf as *const i32,
                                    moe_topk_weights_buf as *const f32,
                                    sh_out_ptr,
                                    sh_gate_ptr,
                                    mlp_out_buf as *mut c_void,
                                    num_tokens as i32,
                                    moe.top_k as i32,
                                    moe.num_experts as i32,
                                    h_dim as i32,
                                    moe.local_moe_interm as i32,
                                )
                            },
                            "w34_moe_f16_down_accum",
                        )
                    }
                    MoeExpertStorage::Tiered { .. } => Ok(()),
                }
            }
        }
    }

    /// Executes the host/VRAM tiered MoE dispatch when `self` is a tiered `FfnBlockWeights::Moe`;
    /// no-op for `Dense` and VRAM-resident `Moe`.
    #[allow(clippy::too_many_arguments)]
    pub fn execute_tiered(
        &self,
        tiered_moe: Option<&mut crate::moe_offload::RankTieredMoe>,
        stream: &CudaStream,
        is_capturing_graph: bool,
        normed_buf: u64,
        moe_topk_ids_buf: u64,
        moe_topk_weights_buf: u64,
        moe_sh_out_buf: u64,
        moe_sh_gate_buf: u64,
        mlp_out_buf: u64,
        num_tokens: usize,
    ) -> Result<()> {
        let Self::Moe(moe) = self else {
            return Ok(());
        };
        let MoeExpertStorage::Tiered { layer_id } = moe.experts else {
            return Ok(());
        };
        let tmoe_engine =
            tiered_moe.ok_or_else(|| loader_err("missing RankTieredMoe for tiered MoE layer"))?;
        let sh_out_ptr = if moe.shared_down_proj.is_some() {
            moe_sh_out_buf
        } else {
            0
        };
        let sh_gate_ptr = if moe.shared_expert_gate.is_some() {
            moe_sh_gate_buf
        } else {
            0
        };
        let device_lru_safe = num_tokens * moe.top_k <= tmoe_engine.layer_hot_k(layer_id)
            && tmoe_engine.layer_has_mapped_store(layer_id);
        if is_capturing_graph && !device_lru_safe {
            return Err(loader_err(
                "tiered MoE graph exceeds mapped hot-cache capacity",
            ));
        }
        if device_lru_safe {
            tmoe_engine.execute_layer_device_lru(
                stream,
                layer_id,
                normed_buf,
                moe_topk_ids_buf,
                moe_topk_weights_buf,
                sh_out_ptr,
                sh_gate_ptr,
                mlp_out_buf,
                num_tokens,
                moe.top_k,
            )
        } else {
            tmoe_engine.execute_layer(
                stream,
                layer_id,
                normed_buf,
                moe_topk_ids_buf,
                moe_topk_weights_buf,
                sh_out_ptr,
                sh_gate_ptr,
                mlp_out_buf,
                num_tokens,
                moe.top_k,
            )
        }
    }
}

#[cfg(feature = "cuda")]
#[allow(clippy::too_many_arguments)]
pub(crate) fn load_moe_layer_weights(
    archive: &SafetensorsArchive,
    cfg: &ModelConfig,
    pfx: &str,
    rank: usize,
    tp_size: usize,
    weight_arena: &mut DeviceArena,
    stream: &CudaStream,
    repack_staging: &DeviceBuffer,
) -> Result<MoeLayerWeights> {
    load_moe_layer_weights_inner(
        archive,
        cfg,
        None,
        pfx,
        rank,
        tp_size,
        weight_arena,
        stream,
        repack_staging,
    )
}

#[cfg(feature = "cuda")]
#[allow(clippy::too_many_arguments)]
pub(crate) fn load_tiered_moe_layer_weights(
    archive: &SafetensorsArchive,
    cfg: &ModelConfig,
    layer_id: usize,
    pfx: &str,
    rank: usize,
    tp_size: usize,
    weight_arena: &mut DeviceArena,
    stream: &CudaStream,
    repack_staging: &DeviceBuffer,
) -> Result<MoeLayerWeights> {
    load_moe_layer_weights_inner(
        archive,
        cfg,
        Some(layer_id),
        pfx,
        rank,
        tp_size,
        weight_arena,
        stream,
        repack_staging,
    )
}

#[cfg(feature = "cuda")]
#[allow(clippy::too_many_arguments)]
fn load_moe_layer_weights_inner(
    archive: &SafetensorsArchive,
    cfg: &ModelConfig,
    tiered_layer_id: Option<usize>,
    pfx: &str,
    rank: usize,
    tp_size: usize,
    weight_arena: &mut DeviceArena,
    stream: &CudaStream,
    repack_staging: &DeviceBuffer,
) -> Result<MoeLayerWeights> {
    let raw_router = RawLinearHost::load(archive, &format!("{pfx}.mlp.gate"), &cfg.quant)?;
    let router_gate = raw_router.upload(weight_arena, stream, repack_staging)?;

    let full_interm = cfg.moe_intermediate_size;
    if tiered_layer_id.is_none() && !full_interm.is_multiple_of(tp_size) {
        return Err(loader_err_msg(format!(
            "moe_intermediate_size {} is not divisible by TP={tp_size}",
            full_interm
        )));
    }
    // Tiered MoE partitions expert IDs across EP ranks, preserving full `moe_intermediate_size` per expert.
    let local_moe_interm = if tiered_layer_id.is_some() {
        full_interm
    } else {
        full_interm / tp_size
    };
    let num_exp = cfg.num_experts;
    let top_k = cfg.num_experts_per_tok.max(1);
    let scoring_func = i32::from(cfg.scoring_func.eq_ignore_ascii_case("sigmoid"));
    let h = cfg.hidden_size;

    let fused_gu_key = archive
        .resolve_name(&format!("{pfx}.mlp.experts.gate_up_proj"))
        .or_else(|| archive.resolve_name(&format!("{pfx}.mlp.experts.gate_up_proj.weight")));
    let fused_dn_key = archive
        .resolve_name(&format!("{pfx}.mlp.experts.down_proj"))
        .or_else(|| archive.resolve_name(&format!("{pfx}.mlp.experts.down_proj.weight")));

    let experts = if let Some(layer_id) = tiered_layer_id {
        MoeExpertStorage::Tiered { layer_id }
    } else if let (Some(gu_key), Some(dn_key)) = (fused_gu_key, fused_dn_key) {
        if archive
            .resolve_name(&format!("{pfx}.mlp.experts.gate_up_proj_scale_inv"))
            .is_some()
            || archive
                .resolve_name(&format!("{pfx}.mlp.experts.gate_up_proj.weight_scale"))
                .is_some()
            || archive
                .resolve_name(&format!("{pfx}.mlp.experts.gate_up_proj.weight_scale_inv"))
                .is_some()
        {
            return Err(loader_err_msg(format!(
                "unsupported FP8 fused MoE expert tensor '{gu_key}' (warp-34 sm_86/sm_89 spine requires W4A16 or BF16/FP16 MoE)"
            )));
        }
        let gu_info = archive
            .tensor_info(&gu_key)
            .map_err(|e| loader_err_msg(format!("tensor_info fused experts.gate_up_proj: {e}")))?;
        let dn_info = archive
            .tensor_info(&dn_key)
            .map_err(|e| loader_err_msg(format!("tensor_info fused experts.down_proj: {e}")))?;
        if gu_info.shape.as_slice() != [num_exp, 2 * full_interm, h]
            || dn_info.shape.as_slice() != [num_exp, h, full_interm]
        {
            return Err(loader_err_msg(format!(
                "unexpected fused 3D MoE expert shapes on '{pfx}': gate_up_proj={:?}, down_proj={:?}",
                gu_info.shape, dn_info.shape
            )));
        }
        let gu_all = archive
            .read_f16_bits(&gu_key)
            .map_err(|e| loader_err_msg(format!("read fused experts.gate_up_proj: {e}")))?;
        let dn_all = archive
            .read_f16_bits(&dn_key)
            .map_err(|e| loader_err_msg(format!("read fused experts.down_proj: {e}")))?;

        let expected_gu = num_exp
            .checked_mul(2)
            .and_then(|v| v.checked_mul(full_interm))
            .and_then(|v| v.checked_mul(h))
            .ok_or_else(|| loader_err("fused experts.gate_up_proj size overflow"))?;
        let expected_dn = num_exp
            .checked_mul(h)
            .and_then(|v| v.checked_mul(full_interm))
            .ok_or_else(|| loader_err("fused experts.down_proj size overflow"))?;
        if gu_all.len() != expected_gu || dn_all.len() != expected_dn {
            return Err(loader_err(
                "fused MoE expert tensor element count mismatch against shape",
            ));
        }
        let local_gu_cap = num_exp
            .checked_mul(2)
            .and_then(|v| v.checked_mul(local_moe_interm))
            .and_then(|v| v.checked_mul(h))
            .ok_or_else(|| loader_err("local fused gate_up capacity overflow"))?;
        let local_dn_cap = num_exp
            .checked_mul(h)
            .and_then(|v| v.checked_mul(local_moe_interm))
            .ok_or_else(|| loader_err("local fused down capacity overflow"))?;
        let mut all_gu_f16 = Vec::with_capacity(local_gu_cap);
        let mut all_dn_f16 = Vec::with_capacity(local_dn_cap);
        let col_start = rank * local_moe_interm;
        let col_end = (rank + 1) * local_moe_interm;

        for e in 0..num_exp {
            let exp_gu = &gu_all[e * 2 * full_interm * h..(e + 1) * 2 * full_interm * h];
            all_gu_f16.extend_from_slice(&exp_gu[col_start * h..col_end * h]);
            all_gu_f16.extend_from_slice(
                &exp_gu[(full_interm + col_start) * h..(full_interm + col_end) * h],
            );

            let exp_dn = &dn_all[e * h * full_interm..(e + 1) * h * full_interm];
            if tp_size == 1 {
                all_dn_f16.extend_from_slice(exp_dn);
            } else {
                for row in 0..h {
                    let row_base = row * full_interm;
                    all_dn_f16.extend_from_slice(&exp_dn[row_base + col_start..row_base + col_end]);
                }
            }
        }
        drop(gu_all);
        drop(dn_all);
        let gate_up_ptr = weight_arena.upload_slice(&all_gu_f16, stream)?;
        let down_ptr = weight_arena.upload_slice(&all_dn_f16, stream)?;
        MoeExpertStorage::DenseF16 {
            gate_up_ptr,
            down_ptr,
        }
    } else {
        if archive
            .resolve_name(&format!("{pfx}.mlp.experts.gate_up_proj_blocks"))
            .is_some()
            || archive
                .resolve_name(&format!("{pfx}.mlp.experts.gate_up_proj.weight_packed"))
                .is_some()
        {
            return Err(loader_err_msg(format!(
                "unsupported fused 3D quantized MoE expert layout on '{pfx}' (expected per-expert W4A16 or 3D BF16/FP16)"
            )));
        }
        let load_expert_parts = |e: usize| -> Result<(RawLinearHost, RawLinearHost)> {
            let epfx = format!("{pfx}.mlp.experts.{e}");
            let rg_full = RawLinearHost::load(archive, &format!("{epfx}.gate_proj"), &cfg.quant)?;
            if rg_full.dims() != (full_interm, h) {
                return Err(loader_err_msg(format!(
                    "MoE gate_proj dims {:?} at '{epfx}' do not match expected ({full_interm}, {h})",
                    rg_full.dims()
                )));
            }
            let ru_full = RawLinearHost::load(archive, &format!("{epfx}.up_proj"), &cfg.quant)?;
            if ru_full.dims() != (full_interm, h) {
                return Err(loader_err_msg(format!(
                    "MoE up_proj dims {:?} at '{epfx}' do not match expected ({full_interm}, {h})",
                    ru_full.dims()
                )));
            }
            let rd_full = RawLinearHost::load(archive, &format!("{epfx}.down_proj"), &cfg.quant)?;
            if rd_full.dims() != (h, full_interm) {
                return Err(loader_err_msg(format!(
                    "MoE down_proj dims {:?} at '{epfx}' do not match expected ({h}, {full_interm})",
                    rd_full.dims()
                )));
            }
            let rg = rg_full.slice_rows(rank * local_moe_interm, (rank + 1) * local_moe_interm);
            let ru = ru_full.slice_rows(rank * local_moe_interm, (rank + 1) * local_moe_interm);
            let gu = RawLinearHost::concat_rows(&[rg, ru])?;
            let rd = rd_full.slice_cols(rank * local_moe_interm, (rank + 1) * local_moe_interm);
            Ok((gu, rd))
        };

        // Probe expert 0 to determine W4A16 vs W4A16Zp vs DenseF16 storage
        let exp0_gate = RawLinearHost::load(
            archive,
            &format!("{pfx}.mlp.experts.0.gate_proj"),
            &cfg.quant,
        )?;
        match exp0_gate {
            RawLinearHost::U4B8NMajor { group_size, .. } => {
                if !h.is_multiple_of(8)
                    || !local_moe_interm.is_multiple_of(8)
                    || !h.is_multiple_of(group_size)
                    || !local_moe_interm.is_multiple_of(group_size)
                {
                    return Err(loader_err_msg(format!(
                        "W4A16 MoE requires hidden_size ({h}) and local_moe_interm ({local_moe_interm}) divisible by 8 and group_size ({group_size})"
                    )));
                }
                let gu_words_per_exp = (2 * local_moe_interm) * (h / 8);
                let gu_scales_per_exp = (2 * local_moe_interm) * (h / group_size);
                let dn_words_per_exp = h * (local_moe_interm / 8);
                let dn_scales_per_exp = h * (local_moe_interm / group_size);

                let mut all_gu_packed = Vec::with_capacity(num_exp * gu_words_per_exp);
                let mut all_gu_scales = Vec::with_capacity(num_exp * gu_scales_per_exp);
                let mut all_dn_packed = Vec::with_capacity(num_exp * dn_words_per_exp);
                let mut all_dn_scales = Vec::with_capacity(num_exp * dn_scales_per_exp);

                for e in 0..num_exp {
                    let epfx = format!("{pfx}.mlp.experts.{e}");
                    let (gu, rd) = load_expert_parts(e)?;
                    if let RawLinearHost::U4B8NMajor {
                        packed,
                        scales,
                        group_size: gs,
                        ..
                    } = gu
                        && gs == group_size
                    {
                        all_gu_packed.extend_from_slice(&packed);
                        all_gu_scales.extend_from_slice(&scales);
                    } else {
                        return Err(loader_err_msg(format!(
                            "inconsistent MoE expert quantization at '{epfx}' (expected symmetric W4A16 group_size={group_size})"
                        )));
                    }
                    if let RawLinearHost::U4B8NMajor {
                        packed,
                        scales,
                        group_size: gs,
                        ..
                    } = rd
                        && gs == group_size
                    {
                        all_dn_packed.extend_from_slice(&packed);
                        all_dn_scales.extend_from_slice(&scales);
                    } else {
                        return Err(loader_err_msg(format!(
                            "inconsistent MoE down_proj quantization at '{epfx}' (expected symmetric W4A16 group_size={group_size})"
                        )));
                    }
                }

                let gate_up_packed_ptr = weight_arena.upload_slice(&all_gu_packed, stream)?;
                let gate_up_scales_ptr = weight_arena.upload_slice(&all_gu_scales, stream)?;
                let down_packed_ptr = weight_arena.upload_slice(&all_dn_packed, stream)?;
                let down_scales_ptr = weight_arena.upload_slice(&all_dn_scales, stream)?;
                MoeExpertStorage::W4A16 {
                    gate_up_packed_ptr,
                    gate_up_scales_ptr,
                    down_packed_ptr,
                    down_scales_ptr,
                    group_size,
                }
            }
            RawLinearHost::U4ZpNMajor { group_size, .. } => {
                if !h.is_multiple_of(8)
                    || !local_moe_interm.is_multiple_of(8)
                    || !h.is_multiple_of(group_size)
                    || !local_moe_interm.is_multiple_of(group_size)
                {
                    return Err(loader_err_msg(format!(
                        "W4A16 asymmetric MoE requires hidden_size ({h}) and local_moe_interm ({local_moe_interm}) divisible by 8 and group_size ({group_size})"
                    )));
                }
                let gu_groups = h / group_size;
                let dn_groups = local_moe_interm / group_size;
                let gu_words_per_exp = (2 * local_moe_interm) * (h / 8);
                let gu_scales_per_exp = (2 * local_moe_interm) * gu_groups;
                let gu_qzeros_per_exp = ((2 * local_moe_interm) / 8) * gu_groups;
                let dn_words_per_exp = h * (local_moe_interm / 8);
                let dn_scales_per_exp = h * dn_groups;
                let dn_qzeros_per_exp = (h / 8) * dn_groups;

                let mut all_gu_packed = Vec::with_capacity(num_exp * gu_words_per_exp);
                let mut all_gu_scales = Vec::with_capacity(num_exp * gu_scales_per_exp);
                let mut all_gu_qzeros = Vec::with_capacity(num_exp * gu_qzeros_per_exp);
                let mut all_dn_packed = Vec::with_capacity(num_exp * dn_words_per_exp);
                let mut all_dn_scales = Vec::with_capacity(num_exp * dn_scales_per_exp);
                let mut all_dn_qzeros = Vec::with_capacity(num_exp * dn_qzeros_per_exp);

                for e in 0..num_exp {
                    let epfx = format!("{pfx}.mlp.experts.{e}");
                    let (gu, rd) = load_expert_parts(e)?;
                    if let RawLinearHost::U4ZpNMajor {
                        packed,
                        scales,
                        qzeros,
                        group_size: gs,
                        ..
                    } = gu
                        && gs == group_size
                    {
                        all_gu_packed.extend_from_slice(&packed);
                        all_gu_scales.extend_from_slice(&scales);
                        all_gu_qzeros.extend_from_slice(&qzeros);
                    } else {
                        return Err(loader_err_msg(format!(
                            "inconsistent MoE expert quantization at '{epfx}' (expected asymmetric W4A16 group_size={group_size})"
                        )));
                    }
                    if let RawLinearHost::U4ZpNMajor {
                        packed,
                        scales,
                        qzeros,
                        group_size: gs,
                        ..
                    } = rd
                        && gs == group_size
                    {
                        all_dn_packed.extend_from_slice(&packed);
                        all_dn_scales.extend_from_slice(&scales);
                        all_dn_qzeros.extend_from_slice(&qzeros);
                    } else {
                        return Err(loader_err_msg(format!(
                            "inconsistent MoE down_proj quantization at '{epfx}' (expected asymmetric W4A16 group_size={group_size})"
                        )));
                    }
                }

                let gate_up_packed_ptr = weight_arena.upload_slice(&all_gu_packed, stream)?;
                let gate_up_scales_ptr = weight_arena.upload_slice(&all_gu_scales, stream)?;
                let gate_up_qzeros_ptr = weight_arena.upload_slice(&all_gu_qzeros, stream)?;
                let down_packed_ptr = weight_arena.upload_slice(&all_dn_packed, stream)?;
                let down_scales_ptr = weight_arena.upload_slice(&all_dn_scales, stream)?;
                let down_qzeros_ptr = weight_arena.upload_slice(&all_dn_qzeros, stream)?;
                MoeExpertStorage::W4A16Zp {
                    gate_up_packed_ptr,
                    gate_up_scales_ptr,
                    gate_up_qzeros_ptr,
                    down_packed_ptr,
                    down_scales_ptr,
                    down_qzeros_ptr,
                    group_size,
                }
            }
            RawLinearHost::U8B128NMajor { .. } => {
                return Err(loader_err_msg(format!(
                    "unsupported 8-bit W8A16 routed MoE experts at '{pfx}' (sm_86/sm_89 MoE supports 4-bit W4A16 or FP16/BF16 experts)"
                )));
            }
            RawLinearHost::DenseF16 { .. } => {
                let mut all_gu_f16 = Vec::with_capacity(num_exp * 2 * local_moe_interm * h);
                let mut all_dn_f16 = Vec::with_capacity(num_exp * h * local_moe_interm);
                for e in 0..num_exp {
                    let epfx = format!("{pfx}.mlp.experts.{e}");
                    let (gu, rd) = load_expert_parts(e)?;
                    if let RawLinearHost::DenseF16 { weight, .. } = gu {
                        all_gu_f16.extend_from_slice(&weight);
                    } else {
                        return Err(loader_err_msg(format!(
                            "inconsistent MoE expert quantization at '{epfx}' (expected DenseF16)"
                        )));
                    }
                    if let RawLinearHost::DenseF16 { weight, .. } = rd {
                        all_dn_f16.extend_from_slice(&weight);
                    } else {
                        return Err(loader_err_msg(format!(
                            "inconsistent MoE down_proj quantization at '{epfx}' (expected DenseF16)"
                        )));
                    }
                }
                let gate_up_ptr = weight_arena.upload_slice(&all_gu_f16, stream)?;
                let down_ptr = weight_arena.upload_slice(&all_dn_f16, stream)?;
                MoeExpertStorage::DenseF16 {
                    gate_up_ptr,
                    down_ptr,
                }
            }
        }
    };

    let full_shared_interm = cfg.shared_expert_intermediate_size;
    let local_shared_interm = full_shared_interm / tp_size;
    let (shared_gate_up_proj, shared_down_proj, shared_expert_gate) = if local_shared_interm > 0
        && archive
            .resolve_name(&format!("{pfx}.mlp.shared_expert.gate_proj.weight"))
            .or_else(|| archive.resolve_name(&format!("{pfx}.mlp.shared_expert.gate_proj.qweight")))
            .or_else(|| {
                archive.resolve_name(&format!("{pfx}.mlp.shared_expert.gate_proj.weight_packed"))
            })
            .is_some()
    {
        let s_gate_full = RawLinearHost::load(
            archive,
            &format!("{pfx}.mlp.shared_expert.gate_proj"),
            &cfg.quant,
        )?;
        if s_gate_full.dims() != (full_shared_interm, h) {
            return Err(loader_err_msg(format!(
                "shared_expert.gate_proj dims {:?} at '{pfx}' do not match expected ({full_shared_interm}, {h})",
                s_gate_full.dims()
            )));
        }
        let s_gate =
            s_gate_full.slice_rows(rank * local_shared_interm, (rank + 1) * local_shared_interm);
        let s_up_full = RawLinearHost::load(
            archive,
            &format!("{pfx}.mlp.shared_expert.up_proj"),
            &cfg.quant,
        )?;
        if s_up_full.dims() != (full_shared_interm, h) {
            return Err(loader_err_msg(format!(
                "shared_expert.up_proj dims {:?} at '{pfx}' do not match expected ({full_shared_interm}, {h})",
                s_up_full.dims()
            )));
        }
        let s_up =
            s_up_full.slice_rows(rank * local_shared_interm, (rank + 1) * local_shared_interm);
        let s_gu = RawLinearHost::concat_rows(&[s_gate, s_up])?.upload(
            weight_arena,
            stream,
            repack_staging,
        )?;
        let s_dn_full = RawLinearHost::load(
            archive,
            &format!("{pfx}.mlp.shared_expert.down_proj"),
            &cfg.quant,
        )?;
        if s_dn_full.dims() != (h, full_shared_interm) {
            return Err(loader_err_msg(format!(
                "shared_expert.down_proj dims {:?} at '{pfx}' do not match expected ({h}, {full_shared_interm})",
                s_dn_full.dims()
            )));
        }
        let s_dn = s_dn_full
            .slice_cols(rank * local_shared_interm, (rank + 1) * local_shared_interm)
            .upload(weight_arena, stream, repack_staging)?;
        let s_eg = if archive
            .resolve_name(&format!("{pfx}.mlp.shared_expert_gate.weight"))
            .or_else(|| archive.resolve_name(&format!("{pfx}.mlp.shared_expert_gate.qweight")))
            .or_else(|| {
                archive.resolve_name(&format!("{pfx}.mlp.shared_expert_gate.weight_packed"))
            })
            .is_some()
        {
            let raw_eg = RawLinearHost::load(
                archive,
                &format!("{pfx}.mlp.shared_expert_gate"),
                &cfg.quant,
            )?;
            if raw_eg.dims() != (1, h) {
                return Err(loader_err_msg(format!(
                    "shared_expert_gate dims {:?} at '{pfx}' do not match expected (1, {h})",
                    raw_eg.dims()
                )));
            }
            Some(raw_eg.upload(weight_arena, stream, repack_staging)?)
        } else {
            None
        };
        (Some(s_gu), Some(s_dn), s_eg)
    } else {
        (None, None, None)
    };

    Ok(MoeLayerWeights {
        router_gate,
        experts,
        num_experts: num_exp,
        top_k,
        norm_topk_prob: cfg.norm_topk_prob,
        scoring_func,
        local_moe_interm,
        shared_gate_up_proj,
        shared_down_proj,
        shared_expert_gate,
        local_shared_interm,
    })
}

#[cfg(feature = "cuda")]
pub(crate) fn load_hc_block_weights(
    archive: &SafetensorsArchive,
    cfg: &ModelConfig,
    hc_pfx: &str,
    has_inject: bool,
    weight_arena: &mut DeviceArena,
    stream: &CudaStream,
    repack_staging: &DeviceBuffer,
) -> Result<HcBlockWeights> {
    let total_dim = cfg.hc_count * cfg.hidden_size;
    let low_dim = cfg.hc_lowrank;
    let hc_norm_key = archive
        .resolve_name(&format!("{hc_pfx}.hc_norm.weight"))
        .ok_or_else(|| loader_err_msg(format!("missing '{hc_pfx}.hc_norm.weight'")))?;
    let mut hc_norm_bits = archive
        .read_f16_bits(&hc_norm_key)
        .map_err(|e| loader_err_msg(format!("read '{hc_norm_key}': {e}")))?;
    if hc_norm_bits.len() == cfg.hidden_size && cfg.hc_count > 1 {
        hc_norm_bits = hc_norm_bits.repeat(cfg.hc_count);
    }
    if hc_norm_bits.len() != total_dim {
        return Err(loader_err_msg(format!(
            "'{hc_norm_key}' length {} != expected {total_dim} (or {})",
            hc_norm_bits.len(),
            cfg.hidden_size
        )));
    }
    let hc_norm_ptr = weight_arena.upload_slice(&hc_norm_bits, stream)?;

    let raw_down = RawLinearHost::load(
        archive,
        &format!("{hc_pfx}.input_mix_weight_down"),
        &cfg.quant,
    )?;
    if raw_down.dims() != (low_dim, total_dim) {
        return Err(loader_err_msg(format!(
            "'{hc_pfx}.input_mix_weight_down' dims {:?} != ({low_dim}, {total_dim})",
            raw_down.dims()
        )));
    }
    let input_mix_weight_down = raw_down.upload(weight_arena, stream, repack_staging)?;

    let raw_up = RawLinearHost::load(
        archive,
        &format!("{hc_pfx}.input_mix_weight_up"),
        &cfg.quant,
    )?;
    if raw_up.dims() != (total_dim, low_dim) {
        return Err(loader_err_msg(format!(
            "'{hc_pfx}.input_mix_weight_up' dims {:?} != ({total_dim}, {low_dim})",
            raw_up.dims()
        )));
    }
    let input_mix_weight_up = raw_up.upload(weight_arena, stream, repack_staging)?;

    let block_inject_weight = if has_inject {
        let raw_inj = RawLinearHost::load(
            archive,
            &format!("{hc_pfx}.block_inject_weight"),
            &cfg.quant,
        )?;
        if raw_inj.dims() != (cfg.hc_count, total_dim) {
            return Err(loader_err_msg(format!(
                "'{hc_pfx}.block_inject_weight' dims {:?} != ({}, {total_dim})",
                raw_inj.dims(),
                cfg.hc_count
            )));
        }
        Some(raw_inj.upload(weight_arena, stream, repack_staging)?)
    } else {
        None
    };

    Ok(HcBlockWeights {
        hc_norm_ptr,
        input_mix_weight_down,
        input_mix_weight_up,
        block_inject_weight,
    })
}

#[cfg(feature = "cuda")]
pub(crate) fn load_qsa_indexer_weights(
    archive: &SafetensorsArchive,
    cfg: &ModelConfig,
    pfx: &str,
    weight_arena: &mut DeviceArena,
    stream: &CudaStream,
    repack_staging: &DeviceBuffer,
) -> Result<QsaIndexerWeights> {
    let idx_pfx = format!("{pfx}.self_attn.indexer");
    let expected_n = (cfg.indexer_n_heads + cfg.indexer_kv_heads) * cfg.indexer_head_dim;
    let raw_qk = RawLinearHost::load(archive, &format!("{idx_pfx}.index_qk_proj"), &cfg.quant)?;
    if raw_qk.dims() != (expected_n, cfg.hidden_size) {
        return Err(loader_err_msg(format!(
            "'{idx_pfx}.index_qk_proj' dims {:?} != ({expected_n}, {})",
            raw_qk.dims(),
            cfg.hidden_size
        )));
    }
    let index_qk_proj = raw_qk.upload(weight_arena, stream, repack_staging)?;

    let q_ln_key = archive
        .resolve_name(&format!("{idx_pfx}.q_layernorm.weight"))
        .ok_or_else(|| loader_err_msg(format!("missing '{idx_pfx}.q_layernorm.weight'")))?;
    let q_ln_bits = archive
        .read_f16_bits(&q_ln_key)
        .map_err(|e| loader_err_msg(format!("read '{q_ln_key}': {e}")))?;
    if q_ln_bits.len() != cfg.indexer_head_dim {
        return Err(loader_err_msg(format!(
            "'{q_ln_key}' len {} != {}",
            q_ln_bits.len(),
            cfg.indexer_head_dim
        )));
    }
    let q_layernorm_ptr = weight_arena.upload_slice(&q_ln_bits, stream)?;

    let k_ln_key = archive
        .resolve_name(&format!("{idx_pfx}.k_layernorm.weight"))
        .ok_or_else(|| loader_err_msg(format!("missing '{idx_pfx}.k_layernorm.weight'")))?;
    let k_ln_bits = archive
        .read_f16_bits(&k_ln_key)
        .map_err(|e| loader_err_msg(format!("read '{k_ln_key}': {e}")))?;
    if k_ln_bits.len() != cfg.indexer_head_dim {
        return Err(loader_err_msg(format!(
            "'{k_ln_key}' len {} != {}",
            k_ln_bits.len(),
            cfg.indexer_head_dim
        )));
    }
    let k_layernorm_ptr = weight_arena.upload_slice(&k_ln_bits, stream)?;

    Ok(QsaIndexerWeights {
        index_qk_proj,
        q_layernorm_ptr,
        k_layernorm_ptr,
    })
}

#[cfg(feature = "cuda")]
pub(crate) fn load_ple_layer_weights(
    archive: &SafetensorsArchive,
    cfg: &ModelConfig,
    pfx: &str,
    weight_arena: &mut DeviceArena,
    stream: &CudaStream,
    repack_staging: &DeviceBuffer,
) -> Result<PleLayerWeights> {
    let ple_pfx = format!("{pfx}.ple");
    let total_dim = cfg.hc_count * cfg.hidden_size;
    let raw_key = RawLinearHost::load(archive, &format!("{ple_pfx}.key_proj"), &cfg.quant)?;
    if raw_key.dims() != (total_dim, cfg.ple_embed_dim) {
        return Err(loader_err_msg(format!(
            "'{ple_pfx}.key_proj' dims {:?} != ({total_dim}, {})",
            raw_key.dims(),
            cfg.ple_embed_dim
        )));
    }
    let key_proj = raw_key.upload(weight_arena, stream, repack_staging)?;

    let raw_val = RawLinearHost::load(archive, &format!("{ple_pfx}.value_proj"), &cfg.quant)?;
    if raw_val.dims() != (cfg.hidden_size, cfg.ple_embed_dim) {
        return Err(loader_err_msg(format!(
            "'{ple_pfx}.value_proj' dims {:?} != ({}, {})",
            raw_val.dims(),
            cfg.hidden_size,
            cfg.ple_embed_dim
        )));
    }
    let value_proj = raw_val.upload(weight_arena, stream, repack_staging)?;

    let load_norm = |suffix: &str, arena: &mut DeviceArena| -> Result<u64> {
        let key = archive
            .resolve_name(&format!("{ple_pfx}.{suffix}"))
            .ok_or_else(|| loader_err_msg(format!("missing '{ple_pfx}.{suffix}'")))?;
        let bits = archive
            .read_f16_bits(&key)
            .map_err(|e| loader_err_msg(format!("read '{key}': {e}")))?;
        if bits.len() != total_dim {
            return Err(loader_err_msg(format!(
                "'{key}' len {} != {total_dim}",
                bits.len()
            )));
        }
        arena.upload_slice(&bits, stream)
    };
    let norm_key_ptr = load_norm("norm_key.weight", weight_arena)?;
    let norm_query_ptr = load_norm("norm_query.weight", weight_arena)?;
    let norm_conv_ptr = load_norm("norm_conv.weight", weight_arena)?;

    let conv_key = archive
        .resolve_name(&format!("{ple_pfx}.conv1d.weight"))
        .ok_or_else(|| loader_err_msg(format!("missing '{ple_pfx}.conv1d.weight'")))?;
    let conv_bits = archive
        .read_f16_bits(&conv_key)
        .map_err(|e| loader_err_msg(format!("read '{conv_key}': {e}")))?;
    if conv_bits.len() != total_dim * cfg.ple_conv_kernel_size {
        return Err(loader_err_msg(format!(
            "'{conv_key}' len {} != {}",
            conv_bits.len(),
            total_dim * cfg.ple_conv_kernel_size
        )));
    }
    let conv1d_weight_ptr = weight_arena.upload_slice(&conv_bits, stream)?;

    let scale_key = archive
        .resolve_name(&format!(
            "{ple_pfx}.ple_embedding.ngram_embedding.weight_scale"
        ))
        .ok_or_else(|| {
            loader_err_msg(format!(
                "missing '{ple_pfx}.ple_embedding.ngram_embedding.weight_scale'"
            ))
        })?;
    let scale_vals = archive
        .read_f32(&scale_key)
        .map_err(|e| loader_err_msg(format!("read '{scale_key}': {e}")))?;
    let weight_scale = *scale_vals
        .first()
        .ok_or_else(|| loader_err_msg(format!("empty '{scale_key}'")))?;

    Ok(PleLayerWeights {
        key_proj,
        value_proj,
        norm_key_ptr,
        norm_query_ptr,
        norm_conv_ptr,
        conv1d_weight_ptr,
        weight_scale,
    })
}

#[cfg(feature = "cuda")]
pub struct DecoderLayerWeights {
    pub layer_id: usize,
    pub residual: LayerResidual,
    pub ple: Option<PleLayerWeights>,
    pub block: LayerBlockWeights,
    pub ffn: FfnBlockWeights,
}

impl RawLinearHost {
    pub(super) fn size_check(&self) -> usize {
        self.dims().0
    }
}
