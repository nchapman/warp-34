//! Vision-local biased projection dispatch. Dense cuBLAS rounding is intentional.
use super::*;
use crate::model::{
    DEVICE_SCRATCH_ALIGNMENT_BYTES, MARLIN_TILE_K_W4, MARLIN_TILE_N, RawLinearHost,
};
use w34_kernels::ffi::W34Exec;

#[allow(clippy::too_many_arguments)]
pub(super) fn load_projection(
    archive: &SafetensorsArchive,
    prefix: &str,
    quant: &QuantScheme,
    n: usize,
    k: usize,
    arena: &mut DeviceArena,
    stream: &CudaStream,
    staging: &mut Option<DeviceBuffer>,
) -> Result<LinearProj> {
    let raw = RawLinearHost::load(archive, prefix, quant)?;
    if raw.dims() != (n, k) {
        return Err(CudaError::with_detail(
            -10,
            "vision projection dimensions",
            format!("{prefix}: expected ({n}, {k}), got {:?}", raw.dims()),
        ));
    }
    let bias_name = format!("{prefix}.bias");
    let key = archive
        .resolve_name(&bias_name)
        .ok_or_else(|| CudaError::with_detail(-10, "missing vision bias", &bias_name))?;
    let info = archive
        .tensor_info(&key)
        .map_err(|e| CudaError::with_detail(-10, "vision bias metadata", e.to_string()))?;
    if info.shape != [n] {
        return Err(CudaError::with_detail(
            -10,
            "vision bias dimensions",
            bias_name,
        ));
    }
    if let RawLinearHost::DenseF16 { weight, n, k } = &raw {
        return Ok(LinearProj::DenseF16 {
            weight_ptr: arena.upload_slice(weight, stream)?,
            size_n: *n,
            size_k: *k,
        });
    }
    let bytes = repack_bytes(&raw)?;
    if staging.as_ref().is_none_or(|buf| buf.len_bytes < bytes) {
        *staging = Some(DeviceBuffer::alloc(arena.device_id(), bytes)?);
    }
    raw.upload(arena, stream, staging.as_ref().unwrap())
}

// Hadamard upload pads its inner packed weight before repacking it.
fn repack_bytes(raw: &RawLinearHost) -> Result<usize> {
    let (n, k, divisor) = match raw {
        RawLinearHost::Hadamard { inner, .. } => return repack_bytes(inner),
        RawLinearHost::U4B8NMajor { n, k, .. } | RawLinearHost::U4ZpNMajor { n, k, .. } => {
            (*n, *k, 2)
        }
        RawLinearHost::U8B128NMajor { n, k, .. } => (*n, *k, 1),
        RawLinearHost::DenseF16 { .. } => return Ok(0),
    };
    n.checked_add(MARLIN_TILE_N - 1)
        .map(|v| v / MARLIN_TILE_N * MARLIN_TILE_N)
        .zip(
            k.checked_add(MARLIN_TILE_K_W4 - 1)
                .map(|v| v / MARLIN_TILE_K_W4 * MARLIN_TILE_K_W4),
        )
        .and_then(|(n, k)| n.checked_mul(k))
        .map(|bytes| bytes / divisor)
        .ok_or(CudaError::new(-1, "vision repack staging overflow"))
}

/// Two separately 256-byte-aligned FP16 matrices: transformed/padded input and
/// padded output. op_marlin uses SM-count locks supplied separately and no
/// execution scratch. Width rounding covers both N and K Marlin tile geometry.
pub(super) fn projection_scratch_bytes(rows: usize, max_width: usize) -> Result<usize> {
    max_width
        .checked_add(MARLIN_TILE_K_W4 - 1)
        .map(|v| v / MARLIN_TILE_K_W4 * MARLIN_TILE_K_W4)
        .and_then(|w| rows.checked_mul(w))
        .and_then(|v| v.checked_mul(4))
        .and_then(|v| v.checked_add(2 * DEVICE_SCRATCH_ALIGNMENT_BYTES))
        .ok_or(CudaError::new(-1, "vision projection scratch overflow"))
}

#[allow(clippy::too_many_arguments)]
pub(super) fn forward_bias(
    projection: &LinearProj,
    ex: &W34Exec,
    cublas: &CublasHandle,
    workspace: u64,
    workspace_ints: usize,
    input: u64,
    bias: u64,
    output: u64,
    rows: usize,
) -> Result<()> {
    if let LinearProj::DenseF16 {
        weight_ptr,
        size_n,
        size_k,
    } = projection
    {
        // Do NOT split this into GEMM then bias: cuBLAS's fused rounding differs.
        return cublas.gemm_f16_bias(
            input,
            *weight_ptr,
            bias,
            output,
            rows as i32,
            *size_n as i32,
            *size_k as i32,
        );
    }
    if let LinearProj::Hadamard {
        inner,
        size_n,
        size_k,
        block_size,
    } = projection
    {
        return LinearProj::forward_hadamard_with_bias(
            inner,
            *size_n,
            *size_k,
            *block_size,
            ex,
            cublas,
            workspace,
            workspace_ints,
            input,
            Some(bias),
            output,
            rows,
        );
    }
    projection.forward(ex, cublas, workspace, workspace_ints, input, output, rows)?;
    // The projection wrapper has already cropped any physical N padding.
    check_rc(
        // SAFETY: projection produced contiguous logical [rows, N] FP16 output;
        // loader validated the N-element bias. In-place bias addition is supported.
        unsafe {
            ffi::w34_vision_bias_add_f16(
                ex,
                output as *const c_void,
                bias as *const c_void,
                std::ptr::null(),
                output as *mut c_void,
                rows as i32,
                projection.size_n() as i32,
            )
        },
        "w34_vision_bias_add_f16(projection)",
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn repack_staging_tracks_packed_physical_dimensions() {
        let raw = RawLinearHost::U4B8NMajor {
            packed: vec![],
            scales: vec![],
            n: 65,
            k: 129,
            group_size: 64,
        };
        assert_eq!(repack_bytes(&raw).unwrap(), 128 * 256 / 2);
        let transformed = RawLinearHost::Hadamard {
            inner: Box::new(raw),
            block_size: 16,
        };
        assert_eq!(repack_bytes(&transformed).unwrap(), 128 * 256 / 2);
        let raw = RawLinearHost::U8B128NMajor {
            packed: vec![],
            scales: vec![],
            n: 64,
            k: 128,
            group_size: 64,
            channelwise: false,
        };
        assert_eq!(repack_bytes(&raw).unwrap(), 64 * 128);
        let dense = RawLinearHost::DenseF16 {
            weight: vec![],
            n: 64,
            k: 128,
        };
        assert_eq!(repack_bytes(&dense).unwrap(), 0);
    }

    #[test]
    fn scratch_covers_h16_4304_padding_and_output() {
        for rows in [1, 16, 257] {
            let bytes = projection_scratch_bytes(rows, 4304).unwrap();
            assert!(bytes >= rows * 4352 * 2 * 2 + 256);
            assert_eq!(bytes % 256, 0);
        }
        assert!(projection_scratch_bytes(usize::MAX, 4304).is_err());
        assert!(projection_scratch_bytes(1, usize::MAX).is_err());
    }
}

#[cfg(test)]
mod gpu_tests {
    use super::*;

    struct Fixture(std::path::PathBuf);
    impl Drop for Fixture {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.0);
        }
    }

    fn fixture(weight: Vec<usize>, bias: Option<Vec<usize>>) -> Fixture {
        let dir = std::env::temp_dir().join(format!(
            "w34-vision-linear-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        std::fs::create_dir(&dir).unwrap();
        let mut header = serde_json::Map::new();
        let mut data = Vec::new();
        for (name, shape) in
            std::iter::once(("proj.weight", weight)).chain(bias.map(|shape| ("proj.bias", shape)))
        {
            let start = data.len();
            data.resize(start + shape.iter().product::<usize>() * 2, 0u8);
            header.insert(
                name.into(),
                serde_json::json!({
                    "dtype": "F16", "shape": shape, "data_offsets": [start, data.len()]
                }),
            );
        }
        let mut header = serde_json::to_vec(&header).unwrap();
        while !header.len().is_multiple_of(8) {
            header.push(b' ');
        }
        let mut bytes = (header.len() as u64).to_le_bytes().to_vec();
        bytes.extend(header);
        bytes.extend(data);
        std::fs::write(dir.join("model.safetensors"), bytes).unwrap();
        Fixture(dir)
    }

    #[test]
    fn load_projection_validates_shapes_before_upload() {
        let _device = CudaDevice::set_current(0).unwrap();
        let stream = CudaStream::new(0).unwrap();
        let mut arena = DeviceArena::new(0, 4096).unwrap();
        let mut staging = None;
        for (weight, bias, error) in [
            (
                vec![2, 2],
                Some(vec![1]),
                Some("vision projection dimensions"),
            ),
            (vec![1, 2], None, Some("missing vision bias")),
            (vec![1, 2], Some(vec![1, 1]), Some("vision bias dimensions")),
            (vec![1, 2], Some(vec![2]), Some("vision bias dimensions")),
            (vec![1, 2], Some(vec![1]), None),
        ] {
            let fixture = fixture(weight, bias);
            let archive = SafetensorsArchive::open(&fixture.0).unwrap();
            let result = load_projection(
                &archive,
                "proj",
                &QuantScheme::Unquantized,
                1,
                2,
                &mut arena,
                &stream,
                &mut staging,
            );
            if let Some(error) = error {
                let err = result.err().expect("invalid projection accepted");
                assert!(err.to_string().contains(error), "{err}");
            } else {
                assert!(matches!(
                    result.unwrap(),
                    LinearProj::DenseF16 {
                        size_n: 1,
                        size_k: 2,
                        ..
                    }
                ));
            }
            assert!(staging.is_none(), "dense loading allocated repack staging");
        }
        stream.synchronize().unwrap();
    }

    #[test]
    fn dense_forward_bias_preserves_fused_rounding() {
        let _device = CudaDevice::set_current(0).unwrap();
        let stream = CudaStream::new(0).unwrap();
        let cublas = CublasHandle::new(&stream).unwrap();
        let mut arena = DeviceArena::new(0, 4096).unwrap();
        let input = arena
            .upload_slice(&[f32_to_f16(1.0), f32_to_f16(0.5)], &stream)
            .unwrap();
        let weight_ptr = arena
            .upload_slice(
                &[
                    f32_to_f16(1.0),
                    f32_to_f16(0.0009765625),
                    f32_to_f16(1.0),
                    f32_to_f16(0.0009765625),
                ],
                &stream,
            )
            .unwrap();
        let bias = arena
            .upload_slice(
                &[f32_to_f16(-0.00048828125), f32_to_f16(-0.00048828125)],
                &stream,
            )
            .unwrap();
        let output = DeviceBuffer::alloc(0, 4).unwrap();
        let projection = LinearProj::DenseF16 {
            weight_ptr,
            size_n: 2,
            size_k: 2,
        };
        forward_bias(
            &projection,
            &stream.exec(0, 0),
            &cublas,
            0,
            0,
            input,
            bias,
            output.addr(),
            1,
        )
        .unwrap();
        let mut got = [0u16; 2];
        output.copy_to_host(&mut got, &stream).unwrap();
        stream.synchronize().unwrap();
        // Fused: 1 + 2^-11 - 2^-11 = 1 across both sm_86 and sm_89 for even-aligned N=2.
        // Split rounds the dot to 1 first and yields 1 - 2^-11 (0x3bff).
        assert_eq!(got, [f32_to_f16(1.0), f32_to_f16(1.0)]);

        // Also exercise odd `size_n = 1`: `sm_86` (`compute_minor == 6`) preserves the fused
        // FP32 epilogue (`1.0`), whereas `sm_89` (`compute_minor == 9`) `cuBLASLt` falls back
        // on odd `ldc = 1` (`1.0 - 2^-11 = 0x3bff`).
        let proj_n1 = LinearProj::DenseF16 {
            weight_ptr,
            size_n: 1,
            size_k: 2,
        };
        forward_bias(
            &proj_n1,
            &stream.exec(0, 0),
            &cublas,
            0,
            0,
            input,
            bias,
            output.addr(),
            1,
        )
        .unwrap();
        let mut got_n1 = [0u16; 1];
        output.copy_to_host(&mut got_n1, &stream).unwrap();
        stream.synchronize().unwrap();
        let mut props = w34_kernels::ffi::W34DeviceProps::default();
        assert_eq!(
            // SAFETY: `props` is a valid initialized stack `W34DeviceProps`.
            unsafe { w34_kernels::ffi::w34_get_current_device_props(&raw mut props) },
            0
        );
        let expected_n1 = if props.compute_minor == 9 {
            f32_to_f16(1.0 - 0.00048828125)
        } else {
            f32_to_f16(1.0)
        };
        assert_eq!(got_n1[0], expected_n1);
    }

    #[test]
    fn quantized_biased_projection_matches_dense_exact_fixture() {
        let _device = CudaDevice::set_current(0).unwrap();
        let stream = CudaStream::new(0).unwrap();
        let cublas = CublasHandle::new(&stream).unwrap();
        let (n, k, rows) = (128, 128, 3);
        let mut arena = DeviceArena::new(0, 1024 * 1024).unwrap();
        let staging = DeviceBuffer::alloc(0, n * k).unwrap();
        let quant = RawLinearHost::U4B8NMajor {
            packed: vec![0x99999999; n * k / 8],
            scales: vec![f32_to_f16(1.0); n * k / 64],
            n,
            k,
            group_size: 64,
        }
        .upload(&mut arena, &stream, &staging)
        .unwrap();
        assert!(matches!(quant, LinearProj::MarlinW4A16 { .. }));
        let dense = RawLinearHost::DenseF16 {
            weight: vec![f32_to_f16(1.0); n * k],
            n,
            k,
        }
        .upload(&mut arena, &stream, &staging)
        .unwrap();
        let input = arena
            .upload_slice(&vec![f32_to_f16(0.25); rows * k], &stream)
            .unwrap();
        let bias_values: Vec<_> = (0..n).map(|i| f32_to_f16((i % 8) as f32 * 0.25)).collect();
        let bias = arena.upload_slice(&bias_values, &stream).unwrap();
        let output = DeviceBuffer::alloc(0, rows * n * 2).unwrap();
        let scratch =
            DeviceBuffer::alloc(0, projection_scratch_bytes(rows, n.max(k)).unwrap()).unwrap();
        let ws_ints = CudaDevice::props_for(0).unwrap().multi_processor_count as usize;
        let ws = DeviceBuffer::alloc(0, ws_ints * 4).unwrap();
        ws.zero_async(&stream).unwrap();
        let ex = stream.exec(scratch.addr(), scratch.len_bytes);
        for projection in [&dense, &quant] {
            forward_bias(
                projection,
                &ex,
                &cublas,
                ws.addr(),
                ws_ints,
                input,
                bias,
                output.addr(),
                rows,
            )
            .unwrap();
            let mut got = vec![0u16; rows * n];
            output.copy_to_host(&mut got, &stream).unwrap();
            stream.synchronize().unwrap();
            for (i, value) in got.iter().enumerate() {
                assert_eq!(*value, f32_to_f16(32.0 + f16_to_f32(bias_values[i % n])));
            }
        }
    }
}
