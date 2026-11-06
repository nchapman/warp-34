//! HGPTQ metadata stays attached to the weight through sharding and fusion.
//! Runtime rotation uses caller-owned scratch, never the shared source activation.

use super::*;
use crate::cuda::f32_to_f16;

#[cfg(test)]
#[path = "hgptq_tests.rs"]
mod tests;

impl RawLinearHost {
    pub(super) fn load_with_transform(
        archive: &SafetensorsArchive,
        prefix: &str,
        quant: &QuantScheme,
    ) -> Result<Self> {
        let name = ["weight_packed", "qweight", "weight"]
            .into_iter()
            .find_map(|suffix| {
                archive
                    .resolve_name(&format!("{prefix}.{suffix}"))
                    .map(|s| s.trim_end_matches(&format!(".{suffix}")).to_owned())
            })
            .unwrap_or_else(|| prefix.to_owned());
        let resolved = quant
            .for_linear(&name)
            .map_err(|e| loader_err_msg(e.to_string()))?;
        let block = quant
            .input_hadamard(&name)
            .map_err(|e| loader_err_msg(e.to_string()))?;
        let raw = Self::load_untransformed(archive, prefix, &resolved)?;
        let Some(block_size) = block else {
            return Ok(raw);
        };
        let Self::U4B8NMajor {
            n, k, group_size, ..
        } = &raw
        else {
            return Err(loader_err_msg(format!(
                "HGPTQ requires symmetric packed W4 weights: {name}"
            )));
        };
        if *n == 0
            || *k == 0
            || !k.is_multiple_of(block_size)
            || *n > i32::MAX as usize - 128
            || *k > i32::MAX as usize - 128
            || *group_size != resolved.group_size_or(*k)
        {
            return Err(loader_err_msg(format!(
                "HGPTQ geometry/group mismatch for {name}: N={n} K={k} group={group_size} block={block_size}"
            )));
        }
        Ok(Self::Hadamard {
            inner: Box::new(raw),
            block_size,
        })
    }

    pub(super) fn concat_transformed_rows(parts: &[&Self]) -> Result<Self> {
        let Some(Self::Hadamard { block_size, .. }) = parts.first().copied() else {
            return Err(loader_err(
                "cannot fuse transformed and untransformed projections",
            ));
        };
        let inner: Result<Vec<&Self>> = parts
            .iter()
            .map(|part| match part {
                Self::Hadamard {
                    inner,
                    block_size: b,
                } if b == block_size => Ok(inner.as_ref()),
                _ => Err(loader_err(
                    "cannot fuse projections with different input transforms",
                )),
            })
            .collect();
        Ok(Self::Hadamard {
            inner: Box::new(Self::concat_row_refs(&inner?)?),
            block_size: *block_size,
        })
    }

    #[cfg(feature = "cuda")]
    pub(super) fn upload_hadamard(
        &self,
        block_size: usize,
        arena: &mut DeviceArena,
        stream: &CudaStream,
        staging: &DeviceBuffer,
    ) -> Result<LinearProj> {
        let (n, k) = self.dims();
        let padded;
        let storage = if n.is_multiple_of(64) && k.is_multiple_of(128) {
            self
        } else {
            padded = self.pad_hadamard_weight()?;
            &padded
        };
        let inner = storage.upload(arena, stream, staging)?;
        if !matches!(inner, LinearProj::MarlinW4A16 { .. }) {
            return Err(loader_err("HGPTQ must remain native W4 after upload"));
        }
        Ok(LinearProj::Hadamard {
            inner: Box::new(inner),
            size_n: n,
            size_k: k,
            block_size,
        })
    }

    // Marlin accepts N multiples of 64 with K multiples of 128. Padding is in
    // the rotated basis: nibble 8 is U4B8 zero, not nibble 0 (-8).
    #[cfg(any(feature = "cuda", test))]
    pub(super) fn pad_hadamard_weight(&self) -> Result<Self> {
        let Self::U4B8NMajor {
            packed,
            scales,
            n,
            k,
            group_size,
        } = self
        else {
            return Err(loader_err("HGPTQ padding expects symmetric W4"));
        };
        let pn = n.div_ceil(64) * 64;
        let pk = k.div_ceil(128) * 128;
        let channelwise = *group_size == *k;
        if !channelwise && !matches!(*group_size, 32 | 64 | 128) {
            return Err(loader_err("unsupported HGPTQ group size"));
        }
        let groups = if channelwise { 1 } else { pk / group_size };
        let old_groups = k / group_size;
        let mut padded = vec![0x8888_8888; pn * (pk / 8)];
        let mut padded_scales = vec![f32_to_f16(1.0); pn * groups];
        for row in 0..*n {
            padded[row * (pk / 8)..row * (pk / 8) + k / 8]
                .copy_from_slice(&packed[row * (k / 8)..(row + 1) * (k / 8)]);
            padded_scales[row * groups..row * groups + old_groups]
                .copy_from_slice(&scales[row * old_groups..(row + 1) * old_groups]);
        }
        Ok(Self::U4B8NMajor {
            packed: padded,
            scales: padded_scales,
            n: pn,
            k: pk,
            group_size: if channelwise { pk } else { *group_size },
        })
    }
}

/// Materializing a logical weight must undo its offline basis change too.
/// This path is for host diagnostics; GPU upload never dequantizes HGPTQ.
pub(super) fn unrotate_weights(inner: &RawLinearHost, block: usize) -> Vec<u16> {
    let mut dense = inner.to_f16_vec();
    let mut values = vec![0.0f32; block];
    for chunk in dense.chunks_exact_mut(block) {
        for (v, &bits) in values.iter_mut().zip(chunk.iter()) {
            *v = f16_to_f32(bits);
        }
        let mut step = 1;
        while step < block {
            for base in (0..block).step_by(2 * step) {
                for i in 0..step {
                    let a = values[base + i];
                    let b = values[base + i + step];
                    values[base + i] = a + b;
                    values[base + i + step] = a - b;
                }
            }
            step *= 2;
        }
        for (bits, v) in chunk.iter_mut().zip(&values) {
            *bits = f32_to_f16(*v / (block as f32).sqrt());
        }
    }
    dense
}

#[cfg(feature = "cuda")]
impl LinearProj {
    /// Extra workspace beyond existing GEMM workspace, for this projection.
    pub fn input_transform_workspace_bytes(&self, rows: usize) -> Option<usize> {
        match self {
            Self::Hadamard { inner, size_n, .. } => {
                let input = rows.checked_mul(inner.size_k())?.checked_mul(2)?;
                let output = if *size_n != inner.size_n() {
                    rows.checked_mul(inner.size_n())?.checked_mul(2)?
                } else {
                    0
                };
                let input = input.checked_add(255)? & !255;
                input
                    .checked_add(output)?
                    .checked_add(255)
                    .map(|n| n & !255)
            }
            _ => Some(0),
        }
    }

    #[allow(clippy::too_many_arguments)]
    pub(super) fn forward_hadamard(
        inner: &Self,
        n: usize,
        k: usize,
        block: usize,
        ex: &W34Exec,
        cublas: &CublasHandle,
        locks: u64,
        lock_count: usize,
        input: u64,
        output: u64,
        rows: usize,
    ) -> Result<()> {
        Self::forward_hadamard_with_bias(
            inner, n, k, block, ex, cublas, locks, lock_count, input, None, output, rows,
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn forward_hadamard_with_bias(
        inner: &Self,
        n: usize,
        k: usize,
        block: usize,
        ex: &W34Exec,
        cublas: &CublasHandle,
        locks: u64,
        lock_count: usize,
        input: u64,
        bias: Option<u64>,
        output: u64,
        rows: usize,
    ) -> Result<()> {
        if rows == 0 {
            return Ok(());
        }
        let m = i32::try_from(rows).map_err(|_| loader_err("HGPTQ row count exceeds i32"))?;
        let input_bytes = rows
            .checked_mul(inner.size_k())
            .and_then(|v| v.checked_mul(2))
            .and_then(|v| v.checked_add(255))
            .map(|v| v & !255)
            .ok_or_else(|| loader_err("HGPTQ input scratch overflow"))?;
        let output_bytes = if n != inner.size_n() {
            rows.checked_mul(inner.size_n())
                .and_then(|v| v.checked_mul(2))
                .and_then(|v| v.checked_add(255))
                .map(|v| v & !255)
                .ok_or_else(|| loader_err("HGPTQ output scratch overflow"))?
        } else {
            0
        };
        let bytes = input_bytes
            .checked_add(output_bytes)
            .ok_or_else(|| loader_err("HGPTQ scratch overflow"))?;
        if ex.scratch_ptr == 0 || ex.scratch_size < bytes {
            return Err(loader_err_msg(format!(
                "HGPTQ scratch needs {bytes} bytes, available {}",
                ex.scratch_size
            )));
        }
        let tmp_out = if output_bytes == 0 {
            output
        } else {
            ex.scratch_ptr + input_bytes as u64
        };
        check_rc(
            // SAFETY: caller supplies disjoint live activation/output/workspace buffers;
            // capacity is checked above. Loader validates block-aligned logical K.
            unsafe {
                ffi::w34_block_hadamard_f16(
                    ex,
                    input as *const c_void,
                    ex.scratch_ptr as *mut c_void,
                    m,
                    k as i32,
                    inner.size_k() as i32,
                    block as i32,
                )
            },
            "HGPTQ block Hadamard",
        )?;
        let inner_ex = W34Exec {
            stream: ex.stream,
            scratch_ptr: ex.scratch_ptr + bytes as u64,
            scratch_size: ex.scratch_size - bytes,
        };
        inner.forward(
            &inner_ex,
            cublas,
            locks,
            lock_count,
            ex.scratch_ptr,
            tmp_out,
            rows,
        )?;
        if output_bytes != 0 || bias.is_some() {
            let bias_ptr = bias.map_or(std::ptr::null(), |b| b as *const c_void);
            check_rc(
                // SAFETY: `tmp_out` and `output` have checked extents (`tmp_out == output` only
                // when `n == inner.size_n()`), and `bias_ptr` is null or a live `n`-element FP16 vector.
                unsafe {
                    ffi::w34_linear_epilogue_f16(
                        ex,
                        tmp_out as *const c_void,
                        bias_ptr,
                        output as *mut c_void,
                        m,
                        n as i32,
                        inner.size_n() as i32,
                    )
                },
                "HGPTQ linear epilogue",
            )?;
        }
        Ok(())
    }
}
