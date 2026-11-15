//! `QwenImage21Transformer2DModel` (`transformer/` in `QwenImage21Pipeline`):
//! - 3-axis `[16, 56, 56]` complex RoPE (`theta = 10000`, `apply_rotary_emb_qwen(use_real=False)`)
//! - `QwenImage21TimestepProjEmbeddings` (`sinusoidal(256)` -> `Linear(256, D)` -> `SiLU` -> `Linear(D, D)`)
//! - Shared modulation (`SiLU(temb)` -> `Linear(D, 4*D)`) with `causal_condition` (`t=0` prefix row, `t` target row)
//! - `QwenImage21TextProjection` (`ZeroCenterRMSNorm(context_in_dim)` -> `Linear` -> `GELU(tanh)` -> `Linear`)
//! - 32 single-stream `QwenImage21TransformerBlock`s:
//!   - Parameterless `LayerNorm` + AdaLN `(1 + scale)` modulation
//!   - Per-head `RMSNorm(128)` + complex RoPE + FlashAttention-2 (`head_dim = 128`)
//!   - Step-0 `"extract"` block-causal prefill (`prefix_segments` + prefix KV cache extraction)
//!   - Steps `1..N-1` `"cached"` target-only decode against `[cached_prefix, target]` with CUDA Graph capture
//!   - `tanh(gate) * branch_out` residual add with `FP16` clamp `[-65504, 65504]`
//!   - `QwenImage21SwiGLUFeedForward` (`SiLU(gate_layer(x)) * proj(x)` -> `out`)
//! - `QwenImage21AdaLayerNormContinuous` (`norm_out`) + `proj_out`
//! - Tensor parallelism (`TP=1` and `TP=2` head/MLP sharding)

#[cfg(feature = "cuda")]
use crate::cuda::{
    CublasHandle, CudaDevice, CudaError, CudaGraph, CudaStream, DeviceArena, DeviceBuffer, Result,
    Tp2Collectives, check_rc, prepare_execution_plan,
};
use crate::cuda::{f16_to_f32, f32_to_f16};
#[cfg(feature = "cuda")]
use crate::loader::{QuantScheme, SafetensorsArchive};
#[cfg(feature = "cuda")]
use crate::model::{LinearProj, MARLIN_TILE_K_W4, MARLIN_TILE_N, RawLinearHost};
#[cfg(feature = "cuda")]
use crate::qwen_image::{
    IMG_TOKENS_PER_SLOT, QwenImage21TransformerConfig, build_token_metadata,
    compute_qwen_image_21_rope_cos_sin, prefix_segments,
};
#[cfg(feature = "cuda")]
use std::ffi::c_void;
#[cfg(feature = "cuda")]
use std::sync::{Arc, Mutex};
#[cfg(feature = "cuda")]
use warp34_kernels::ffi;

/// Packs `[C, H, W]` latents into `[H * W, C]` token-major order (`QwenImage21Pipeline._pack_latents`).
///
/// # Panics
///
/// Panics if `channels * height * width` overflows or does not equal `chw.len()`.
pub fn pack_latents_chw_to_hwc<T: Copy + Default>(
    chw: &[T],
    channels: usize,
    height: usize,
    width: usize,
) -> Vec<T> {
    let hw = height
        .checked_mul(width)
        .expect("pack_latents_chw_to_hwc height * width overflow");
    let total = channels
        .checked_mul(hw)
        .expect("pack_latents_chw_to_hwc channels * hw overflow");
    assert_eq!(
        chw.len(),
        total,
        "pack_latents_chw_to_hwc length mismatch: got {}, expected {total}",
        chw.len(),
    );
    let mut out = vec![T::default(); total];
    for c in 0..channels {
        let c_slice = &chw[c * hw..(c + 1) * hw];
        for idx in 0..hw {
            out[idx * channels + c] = c_slice[idx];
        }
    }
    out
}

/// Unpacks `[H * W, C]` token-major latents back to `[C, H, W]` (`QwenImage21Pipeline._unpack_latents`).
///
/// # Panics
///
/// Panics if `channels * height * width` overflows or does not equal `hwc.len()`.
pub fn unpack_latents_hwc_to_chw<T: Copy + Default>(
    hwc: &[T],
    channels: usize,
    height: usize,
    width: usize,
) -> Vec<T> {
    let hw = height
        .checked_mul(width)
        .expect("unpack_latents_hwc_to_chw height * width overflow");
    let total = hw
        .checked_mul(channels)
        .expect("unpack_latents_hwc_to_chw hw * channels overflow");
    assert_eq!(
        hwc.len(),
        total,
        "unpack_latents_hwc_to_chw length mismatch: got {}, expected {total}",
        hwc.len(),
    );
    let mut out = vec![T::default(); total];
    for idx in 0..hw {
        let row = &hwc[idx * channels..(idx + 1) * channels];
        for c in 0..channels {
            out[c * hw + idx] = row[c];
        }
    }
    out
}

/// Applies one `FlowMatchEulerDiscreteScheduler` step in `FP32` and rounds back to `FP16` (`u16` bits),
/// optionally combining conditional and unconditional velocity predictions with `true_cfg_scale`.
///
/// # Panics
///
/// Panics if `sample_f16.len() != cond_pred_f16.len()` or if `uncond_pred_f16` has a different length.
pub fn euler_cfg_step_f16(
    sample_f16: &mut [u16],
    cond_pred_f16: &[u16],
    uncond_pred_f16: Option<&[u16]>,
    true_cfg_scale: f32,
    sigma: f32,
    sigma_next: f32,
) {
    assert_eq!(sample_f16.len(), cond_pred_f16.len());
    let dt = sigma_next - sigma;
    if let Some(neg_f16) = uncond_pred_f16
        && true_cfg_scale > 1.0
    {
        assert_eq!(sample_f16.len(), neg_f16.len());
        for ((s, &pos_bits), &neg_bits) in sample_f16
            .iter_mut()
            .zip(cond_pred_f16.iter())
            .zip(neg_f16.iter())
        {
            // Match PyTorch: `noise_pred = neg + scale * (pos - neg)` in FP16 before `scheduler.step` upcasts `sample` to FP32.
            let pos = f16_to_f32(pos_bits);
            let neg = f16_to_f32(neg_bits);
            let diff_h = f16_to_f32(f32_to_f16(pos - neg));
            let scaled_h = f16_to_f32(f32_to_f16(true_cfg_scale * diff_h));
            let pred_h = f16_to_f32(f32_to_f16(neg + scaled_h));
            let next_val = f16_to_f32(*s) + dt * pred_h;
            *s = f32_to_f16(next_val);
        }
    } else {
        for (s, &pred_bits) in sample_f16.iter_mut().zip(cond_pred_f16.iter()) {
            let pred = f16_to_f32(pred_bits);
            let next_val = f16_to_f32(*s) + dt * pred;
            *s = f32_to_f16(next_val);
        }
    }
}

/// Output from Step-0 `"extract"` prefill of `QwenImage21TransformerEngine`.
#[cfg(feature = "cuda")]
#[derive(Clone, Debug)]
pub struct QwenImage21ExtractOutput {
    /// Full joint output `[total_seq_len, out_dim]` in `FP16` bits.
    pub full_output_f16: Vec<u16>,
    /// Target image slice `full_output_f16[prefix_len..]` (`[target_tokens, out_dim]`) in `FP16` bits.
    pub target_output_f16: Vec<u16>,
    /// Number of prefix (text + condition-image) tokens cached.
    pub prefix_len: usize,
    /// Number of target image tokens.
    pub target_tokens: usize,
}

/// Per-stream GPU KV cache for `QwenImage21TransformerEngine` (`kv_cache_mode = "extract" | "cached"`).
///
/// Each layer's `K` and `V` buffers are allocated with capacity `[prefix_len + target_tokens, heads_local, 128]`
/// so Step 0 writes `0..prefix_len` and Steps `1..N-1` write `prefix_len..prefix_len + target_tokens` in-place
/// with zero `torch.cat` copies and static addresses for CUDA Graph capture.
#[cfg(feature = "cuda")]
pub struct QwenImage21KvCacheGpu {
    pub device_id: i32,
    pub num_layers: usize,
    pub heads_local: usize,
    pub head_dim: usize,
    pub prefix_len: usize,
    pub target_tokens: usize,
    pub layer_k: Vec<DeviceBuffer>,
    pub layer_v: Vec<DeviceBuffer>,
    pub d_cos_target: Option<DeviceBuffer>,
    pub d_sin_target: Option<DeviceBuffer>,
    cached_graph: Option<(usize, usize, CudaGraph)>,
    graph_warmed_geom: Option<(usize, usize)>,
}

#[cfg(feature = "cuda")]
impl QwenImage21KvCacheGpu {
    pub fn new(device_id: i32, num_layers: usize, heads_local: usize, head_dim: usize) -> Self {
        Self {
            device_id,
            num_layers,
            heads_local,
            head_dim,
            prefix_len: 0,
            target_tokens: 0,
            layer_k: Vec::new(),
            layer_v: Vec::new(),
            d_cos_target: None,
            d_sin_target: None,
            cached_graph: None,
            graph_warmed_geom: None,
        }
    }

    fn ensure_geometry(
        &mut self,
        prefix_len: usize,
        target_tokens: usize,
        cos_target: &[f32],
        sin_target: &[f32],
        stream: &CudaStream,
    ) -> Result<()> {
        let total_kv_tokens = prefix_len + target_tokens;
        let layer_bytes = total_kv_tokens
            .saturating_mul(self.heads_local)
            .saturating_mul(self.head_dim)
            .saturating_mul(2)
            .max(16);
        let geom_changed = self.prefix_len != prefix_len || self.target_tokens != target_tokens;
        if self.layer_k.len() != self.num_layers
            || self.layer_k.iter().any(|b| b.len_bytes < layer_bytes)
        {
            self.layer_k.clear();
            self.layer_v.clear();
            for _ in 0..self.num_layers {
                self.layer_k
                    .push(DeviceBuffer::alloc(self.device_id, layer_bytes)?);
                self.layer_v
                    .push(DeviceBuffer::alloc(self.device_id, layer_bytes)?);
            }
            self.cached_graph = None;
            self.graph_warmed_geom = None;
        } else if geom_changed {
            self.cached_graph = None;
            self.graph_warmed_geom = None;
        }
        self.prefix_len = prefix_len;
        self.target_tokens = target_tokens;

        let rope_bytes = (target_tokens * (self.head_dim / 2) * 4).max(16);
        if self
            .d_cos_target
            .as_ref()
            .is_none_or(|b| b.len_bytes < rope_bytes)
        {
            self.d_cos_target = Some(DeviceBuffer::alloc(self.device_id, rope_bytes)?);
            self.d_sin_target = Some(DeviceBuffer::alloc(self.device_id, rope_bytes)?);
            self.cached_graph = None;
            self.graph_warmed_geom = None;
        }
        if !cos_target.is_empty() {
            self.d_cos_target
                .as_ref()
                .unwrap()
                .copy_from_host(cos_target, stream)?;
            self.d_sin_target
                .as_ref()
                .unwrap()
                .copy_from_host(sin_target, stream)?;
        }
        Ok(())
    }

    /// Reads the cached prefix `(K, V)` (`[prefix_len, heads_local, head_dim]` in `FP16` bits) for `layer_idx`.
    pub fn read_prefix_kv(
        &self,
        layer_idx: usize,
        stream: &CudaStream,
    ) -> Result<(Vec<u16>, Vec<u16>)> {
        if layer_idx >= self.layer_k.len() || layer_idx >= self.layer_v.len() {
            return Err(CudaError::new(
                -10,
                "layer_idx out of range in QwenImage21KvCacheGpu::read_prefix_kv",
            ));
        }
        let _dev = CudaDevice::set_current(self.device_id)?;
        let elems = self.prefix_len * self.heads_local * self.head_dim;
        let mut k_host = vec![0u16; elems];
        let mut v_host = vec![0u16; elems];
        if elems > 0 {
            check_rc(
                // SAFETY: `k_host` and `v_host` hold `elems * 2` bytes; `layer_k`/`layer_v` hold at least `elems * 2` bytes.
                unsafe {
                    let rc_k = ffi::warp34_memcpy_d2h_async(
                        k_host.as_mut_ptr() as *mut c_void,
                        self.layer_k[layer_idx].addr() as *const c_void,
                        elems * 2,
                        stream.handle,
                    );
                    if rc_k != 0 {
                        rc_k
                    } else {
                        ffi::warp34_memcpy_d2h_async(
                            v_host.as_mut_ptr() as *mut c_void,
                            self.layer_v[layer_idx].addr() as *const c_void,
                            elems * 2,
                            stream.handle,
                        )
                    }
                },
                "QwenImage21KvCacheGpu::read_prefix_kv D2H",
            )?;
            stream.synchronize()?;
        }
        Ok((k_host, v_host))
    }
}

#[cfg(feature = "cuda")]
struct QwenImage21BlockGpu {
    to_q: LinearProj,
    to_k: LinearProj,
    to_v: LinearProj,
    norm_q_w: u64,
    norm_k_w: u64,
    to_out: LinearProj,
    mlp_gate_up: LinearProj,
    mlp_out: LinearProj,
}

#[cfg(feature = "cuda")]
#[derive(Default)]
struct QwenImage21Workspace {
    marlin_ws: Option<DeviceBuffer>,
    exec_scratch: Option<DeviceBuffer>,
    d_img_in: Option<DeviceBuffer>,
    d_img_proj: Option<DeviceBuffer>,
    d_enc_in: Option<DeviceBuffer>,
    d_txt_norm: Option<DeviceBuffer>,
    d_txt_mid: Option<DeviceBuffer>,
    d_txt_proj: Option<DeviceBuffer>,
    d_row_map: Option<DeviceBuffer>,
    d_target_mask: Option<DeviceBuffer>,
    d_cos_full: Option<DeviceBuffer>,
    d_sin_full: Option<DeviceBuffer>,
    d_timesteps: Option<DeviceBuffer>,
    d_t_sin: Option<DeviceBuffer>,
    d_temb_mid: Option<DeviceBuffer>,
    d_temb: Option<DeviceBuffer>,
    d_temb_silu: Option<DeviceBuffer>,
    d_modulation: Option<DeviceBuffer>,
    d_norm_out_scale: Option<DeviceBuffer>,
    d_joint: Option<DeviceBuffer>,
    d_norm: Option<DeviceBuffer>,
    d_q: Option<DeviceBuffer>,
    d_k: Option<DeviceBuffer>,
    d_v: Option<DeviceBuffer>,
    d_attn_out: Option<DeviceBuffer>,
    d_branch_out: Option<DeviceBuffer>,
    d_gate_up: Option<DeviceBuffer>,
    d_mlp_interm: Option<DeviceBuffer>,
    d_out: Option<DeviceBuffer>,
}

#[cfg(feature = "cuda")]
impl QwenImage21Workspace {
    fn ensure_buf(
        slot: &mut Option<DeviceBuffer>,
        device_id: i32,
        min_bytes: usize,
    ) -> Result<u64> {
        let need = min_bytes.max(16);
        if slot.as_ref().is_none_or(|b| b.len_bytes < need) {
            *slot = Some(DeviceBuffer::alloc(device_id, need)?);
        }
        Ok(slot.as_ref().unwrap().addr())
    }
}

/// GPU execution engine for `QwenImage21Transformer2DModel`.
#[cfg(feature = "cuda")]
pub struct QwenImage21TransformerEngine {
    pub cfg: QwenImage21TransformerConfig,
    pub device_id: i32,
    pub tp_rank: usize,
    pub tp_size: usize,
    pub heads_local: usize,
    pub qkv_dim_local: usize,
    pub mlp_local: usize,
    pub arena: DeviceArena,
    time_linear_1: LinearProj,
    time_linear_2: LinearProj,
    txt_norm_w: u64,
    txt_in_layer: LinearProj,
    txt_out_layer: LinearProj,
    img_in: LinearProj,
    modulation_linear: LinearProj,
    blocks: Vec<QwenImage21BlockGpu>,
    norm_out_linear: LinearProj,
    proj_out: LinearProj,
    workspace: Mutex<QwenImage21Workspace>,
}

#[cfg(feature = "cuda")]
impl QwenImage21TransformerEngine {
    /// Loads `QwenImage21Transformer2DModel` weights onto `device_id` (`TP=1`).
    pub fn load(
        archive: &SafetensorsArchive,
        cfg: &QwenImage21TransformerConfig,
        quant: &QuantScheme,
        device_id: i32,
        stream: &CudaStream,
    ) -> Result<Self> {
        Self::load_with_prefix(archive, cfg, quant, "", device_id, 0, 1, stream)
    }

    /// Loads `QwenImage21Transformer2DModel` weights onto `device_id` with an optional `weight_prefix`
    /// and tensor-parallel shard `(tp_rank, tp_size)`.
    #[allow(clippy::too_many_arguments)]
    pub fn load_with_prefix(
        archive: &SafetensorsArchive,
        cfg: &QwenImage21TransformerConfig,
        quant: &QuantScheme,
        weight_prefix: &str,
        device_id: i32,
        tp_rank: usize,
        tp_size: usize,
        stream: &CudaStream,
    ) -> Result<Self> {
        if tp_size == 0 || tp_rank >= tp_size {
            return Err(CudaError::new(
                -10,
                "invalid TP rank/size for QwenImage21TransformerEngine",
            ));
        }
        if !cfg.num_attention_heads.is_multiple_of(tp_size)
            || !cfg.mlp_hidden_dim().is_multiple_of(tp_size)
        {
            return Err(CudaError::new(
                -10,
                "QwenImage21TransformerConfig heads/mlp not divisible by tp_size",
            ));
        }
        if cfg.attention_head_dim != 128 {
            return Err(CudaError::with_detail(
                -10,
                "QwenImage21TransformerEngine requires attention_head_dim == 128",
                cfg.attention_head_dim.to_string(),
            ));
        }

        let _dev = CudaDevice::set_current(device_id)?;
        let inner_dim = cfg.inner_dim();
        let head_dim = cfg.attention_head_dim;
        let heads_local = cfg.num_attention_heads / tp_size;
        let qkv_dim_local = heads_local * head_dim;
        let mlp_full = cfg.mlp_hidden_dim();
        let mlp_local = mlp_full / tp_size;
        let in_dim = cfg.in_channels * cfg.patch_size * cfg.patch_size;
        let out_dim = cfg.out_channels * cfg.patch_size * cfg.patch_size;
        let ctx_dim = cfg.context_in_dim;

        let block_linear_elems = 4usize
            .saturating_mul(qkv_dim_local)
            .saturating_mul(inner_dim)
            .saturating_add(3usize.saturating_mul(mlp_local).saturating_mul(inner_dim));
        let per_block_bytes = if *quant == QuantScheme::Unquantized {
            block_linear_elems
                .saturating_add(2usize.saturating_mul(head_dim))
                .saturating_mul(2)
        } else {
            let bytes_per_elem_num = if quant.bits() == 8 { 5 } else { 3 };
            (block_linear_elems.saturating_mul(bytes_per_elem_num)) / 4
                + 2usize.saturating_mul(head_dim).saturating_mul(2)
                + 64 * 1024
        };
        let top_level_elems = 256usize
            .saturating_mul(inner_dim)
            .saturating_add(7usize.saturating_mul(inner_dim).saturating_mul(inner_dim))
            .saturating_add(ctx_dim)
            .saturating_add(inner_dim.saturating_mul(ctx_dim))
            .saturating_add(inner_dim.saturating_mul(in_dim))
            .saturating_add(out_dim.saturating_mul(inner_dim));
        let top_level_bytes = top_level_elems.saturating_mul(2);
        let est_bytes = cfg
            .num_layers
            .saturating_mul(per_block_bytes)
            .saturating_add(top_level_bytes)
            .saturating_add(64 * 1024 * 1024);
        let mut arena = DeviceArena::new(device_id, est_bytes)?;

        let upload_f16 =
            |arena: &mut DeviceArena, suffix: &str, expected_len: usize| -> Result<u64> {
                let name = format!("{weight_prefix}{suffix}");
                let key = archive.resolve_name(&name).ok_or_else(|| {
                    CudaError::with_detail(-10, "missing QwenImage21 tensor", &name)
                })?;
                let bits = archive.read_f16_bits(&key).map_err(|e| {
                    CudaError::with_detail(-10, "read QwenImage21 tensor", e.to_string())
                })?;
                if bits.len() != expected_len {
                    return Err(CudaError::with_detail(
                        -10,
                        "QwenImage21 tensor length mismatch",
                        format!("{name}: expected {expected_len}, got {}", bits.len()),
                    ));
                }
                arena.upload_slice(&bits, stream)
            };

        let mut repack_staging: Option<DeviceBuffer> = None;
        let upload_raw = |arena: &mut DeviceArena,
                          staging: &mut Option<DeviceBuffer>,
                          raw: &RawLinearHost|
         -> Result<LinearProj> {
            if let RawLinearHost::DenseF16 { weight, n, k } = raw {
                return Ok(LinearProj::DenseF16 {
                    weight_ptr: arena.upload_slice(weight, stream)?,
                    size_n: *n,
                    size_k: *k,
                });
            }
            let (n, k) = raw.dims();
            let need = n
                .div_ceil(MARLIN_TILE_N)
                .saturating_mul(MARLIN_TILE_N)
                .saturating_mul(
                    k.div_ceil(MARLIN_TILE_K_W4)
                        .saturating_mul(MARLIN_TILE_K_W4),
                );
            if need > 0 && staging.as_ref().is_none_or(|b| b.len_bytes < need) {
                *staging = Some(DeviceBuffer::alloc(device_id, need)?);
            }
            raw.upload(arena, stream, staging.as_ref().unwrap())
        };

        let load_raw =
            |suffix: &str, expected_n: usize, expected_k: usize| -> Result<RawLinearHost> {
                let prefix = format!("{weight_prefix}{suffix}");
                let raw = RawLinearHost::load(archive, &prefix, quant)?;
                if raw.dims() != (expected_n, expected_k) {
                    return Err(CudaError::with_detail(
                        -10,
                        "QwenImage21 linear shape mismatch",
                        format!(
                            "{prefix}: expected ({expected_n}, {expected_k}), got {:?}",
                            raw.dims()
                        ),
                    ));
                }
                Ok(raw)
            };

        let time_linear_1 = upload_raw(
            &mut arena,
            &mut repack_staging,
            &load_raw("time_text_embed.timestep_embedder.linear_1", inner_dim, 256)?,
        )?;
        let time_linear_2 = upload_raw(
            &mut arena,
            &mut repack_staging,
            &load_raw(
                "time_text_embed.timestep_embedder.linear_2",
                inner_dim,
                inner_dim,
            )?,
        )?;
        let txt_norm_w = upload_f16(&mut arena, "txt_in.text_norm.weight", ctx_dim)?;
        let txt_in_layer = upload_raw(
            &mut arena,
            &mut repack_staging,
            &load_raw("txt_in.in_layer", inner_dim, ctx_dim)?,
        )?;
        let txt_out_layer = upload_raw(
            &mut arena,
            &mut repack_staging,
            &load_raw("txt_in.out_layer", inner_dim, inner_dim)?,
        )?;
        let img_in = upload_raw(
            &mut arena,
            &mut repack_staging,
            &load_raw("img_in", inner_dim, in_dim)?,
        )?;
        let modulation_linear = upload_raw(
            &mut arena,
            &mut repack_staging,
            &load_raw("modulation.1", 4 * inner_dim, inner_dim)?,
        )?;

        let qkv_start = tp_rank * qkv_dim_local;
        let qkv_end = qkv_start + qkv_dim_local;
        let mlp_start = tp_rank * mlp_local;
        let mlp_end = mlp_start + mlp_local;

        let mut blocks = Vec::with_capacity(cfg.num_layers);
        for i in 0..cfg.num_layers {
            let bp = format!("transformer_blocks.{i}");
            let q_raw = load_raw(&format!("{bp}.attn.to_q"), inner_dim, inner_dim)?;
            let k_raw = load_raw(&format!("{bp}.attn.to_k"), inner_dim, inner_dim)?;
            let v_raw = load_raw(&format!("{bp}.attn.to_v"), inner_dim, inner_dim)?;
            let out_raw = load_raw(&format!("{bp}.attn.to_out.0"), inner_dim, inner_dim)?;

            let to_q = upload_raw(
                &mut arena,
                &mut repack_staging,
                &if tp_size > 1 {
                    q_raw.slice_rows(qkv_start, qkv_end)
                } else {
                    q_raw
                },
            )?;
            let to_k = upload_raw(
                &mut arena,
                &mut repack_staging,
                &if tp_size > 1 {
                    k_raw.slice_rows(qkv_start, qkv_end)
                } else {
                    k_raw
                },
            )?;
            let to_v = upload_raw(
                &mut arena,
                &mut repack_staging,
                &if tp_size > 1 {
                    v_raw.slice_rows(qkv_start, qkv_end)
                } else {
                    v_raw
                },
            )?;
            let norm_q_w = upload_f16(&mut arena, &format!("{bp}.attn.norm_q.weight"), head_dim)?;
            let norm_k_w = upload_f16(&mut arena, &format!("{bp}.attn.norm_k.weight"), head_dim)?;
            let to_out = upload_raw(
                &mut arena,
                &mut repack_staging,
                &if tp_size > 1 {
                    out_raw.slice_cols(qkv_start, qkv_end)?
                } else {
                    out_raw
                },
            )?;

            let gate_raw = load_raw(&format!("{bp}.img_mlp.gate_layer"), mlp_full, inner_dim)?;
            let proj_raw = load_raw(&format!("{bp}.img_mlp.proj"), mlp_full, inner_dim)?;
            let mlp_out_raw = load_raw(&format!("{bp}.img_mlp.out"), inner_dim, mlp_full)?;

            let gate_shard = if tp_size > 1 {
                gate_raw.slice_rows(mlp_start, mlp_end)
            } else {
                gate_raw
            };
            let proj_shard = if tp_size > 1 {
                proj_raw.slice_rows(mlp_start, mlp_end)
            } else {
                proj_raw
            };
            let gate_up_raw = RawLinearHost::concat_rows(&[gate_shard, proj_shard])?;
            let mlp_gate_up = upload_raw(&mut arena, &mut repack_staging, &gate_up_raw)?;
            let mlp_out = upload_raw(
                &mut arena,
                &mut repack_staging,
                &if tp_size > 1 {
                    mlp_out_raw.slice_cols(mlp_start, mlp_end)?
                } else {
                    mlp_out_raw
                },
            )?;

            blocks.push(QwenImage21BlockGpu {
                to_q,
                to_k,
                to_v,
                norm_q_w,
                norm_k_w,
                to_out,
                mlp_gate_up,
                mlp_out,
            });
        }

        let norm_out_linear = upload_raw(
            &mut arena,
            &mut repack_staging,
            &load_raw("norm_out.linear", inner_dim, inner_dim)?,
        )?;
        let proj_out = upload_raw(
            &mut arena,
            &mut repack_staging,
            &load_raw("proj_out", out_dim, inner_dim)?,
        )?;

        stream.synchronize()?;
        Ok(Self {
            cfg: cfg.clone(),
            device_id,
            tp_rank,
            tp_size,
            heads_local,
            qkv_dim_local,
            mlp_local,
            arena,
            time_linear_1,
            time_linear_2,
            txt_norm_w,
            txt_in_layer,
            txt_out_layer,
            img_in,
            modulation_linear,
            blocks,
            norm_out_linear,
            proj_out,
            workspace: Mutex::new(QwenImage21Workspace::default()),
        })
    }

    /// Creates a new empty `QwenImage21KvCacheGpu` matching this engine's layer and head configuration.
    pub fn new_kv_cache(&self) -> QwenImage21KvCacheGpu {
        QwenImage21KvCacheGpu::new(
            self.device_id,
            self.cfg.num_layers,
            self.heads_local,
            self.cfg.attention_head_dim,
        )
    }

    /// Runs Step-0 `"extract"` (or uncached full forward) over the joint text + condition-image + target-image sequence.
    #[allow(clippy::too_many_arguments, clippy::cognitive_complexity)]
    pub fn forward_extract(
        &self,
        cond_latents_f16: &[u16],
        target_latents_f16: &[u16],
        encoder_hidden_f16: &[u16],
        vlm_img_mask: &[bool],
        img_shapes: &[(usize, usize, usize)],
        timestep: f32,
        kv_cache: Option<&mut QwenImage21KvCacheGpu>,
        cublas: &CublasHandle,
        stream: &CudaStream,
    ) -> Result<QwenImage21ExtractOutput> {
        self.forward_extract_with_allreduce::<fn(u64, usize, &CudaStream) -> Result<()>>(
            cond_latents_f16,
            target_latents_f16,
            encoder_hidden_f16,
            vlm_img_mask,
            img_shapes,
            timestep,
            kv_cache,
            cublas,
            stream,
            None,
        )
    }

    /// Runs Step-0 `"extract"` with an optional tensor-parallel allreduce callback `allreduce_fn(d_buf, num_f16_elems, stream)`.
    #[allow(clippy::too_many_arguments, clippy::cognitive_complexity)]
    pub fn forward_extract_with_allreduce<F>(
        &self,
        cond_latents_f16: &[u16],
        target_latents_f16: &[u16],
        encoder_hidden_f16: &[u16],
        vlm_img_mask: &[bool],
        img_shapes: &[(usize, usize, usize)],
        timestep: f32,
        mut kv_cache: Option<&mut QwenImage21KvCacheGpu>,
        cublas: &CublasHandle,
        stream: &CudaStream,
        mut allreduce_fn: Option<F>,
    ) -> Result<QwenImage21ExtractOutput>
    where
        F: FnMut(u64, usize, &CudaStream) -> Result<()>,
    {
        let _dev = CudaDevice::set_current(self.device_id)?;
        if img_shapes.is_empty() {
            return Err(CudaError::new(
                -10,
                "img_shapes must contain at least the target image shape",
            ));
        }
        let in_dim = self.cfg.in_channels * self.cfg.patch_size * self.cfg.patch_size;
        let out_dim = self.cfg.out_channels * self.cfg.patch_size * self.cfg.patch_size;
        let inner_dim = self.cfg.inner_dim();
        let ctx_dim = self.cfg.context_in_dim;
        let eps = self.cfg.eps;

        let cond_tokens: usize = img_shapes[..img_shapes.len() - 1]
            .iter()
            .map(|&(f, h, w)| f * h * w)
            .sum();
        let (tf, th, tw) = *img_shapes.last().unwrap();
        let target_tokens = tf * th * tw;
        let img_tokens = cond_tokens + target_tokens;

        if cond_latents_f16.len() != cond_tokens * in_dim {
            return Err(CudaError::with_detail(
                -10,
                "cond_latents_f16 length mismatch",
                format!(
                    "expected {} ({cond_tokens} * {in_dim}), got {}",
                    cond_tokens * in_dim,
                    cond_latents_f16.len()
                ),
            ));
        }
        if target_latents_f16.len() != target_tokens * in_dim {
            return Err(CudaError::with_detail(
                -10,
                "target_latents_f16 length mismatch",
                format!(
                    "expected {} ({target_tokens} * {in_dim}), got {}",
                    target_tokens * in_dim,
                    target_latents_f16.len()
                ),
            ));
        }
        if !encoder_hidden_f16.len().is_multiple_of(ctx_dim) || encoder_hidden_f16.is_empty() {
            return Err(CudaError::new(
                -10,
                "encoder_hidden_f16 must be non-empty multiple of context_in_dim",
            ));
        }
        let txt_seq_len = encoder_hidden_f16.len() / ctx_dim;
        let target_slots = target_tokens / IMG_TOKENS_PER_SLOT;

        let full_vlm_mask: Vec<bool> = if vlm_img_mask.len() == txt_seq_len {
            let mut m = Vec::with_capacity(txt_seq_len + target_slots);
            m.extend_from_slice(vlm_img_mask);
            m.resize(txt_seq_len + target_slots, true);
            m
        } else if vlm_img_mask.len() == txt_seq_len + target_slots {
            vlm_img_mask.to_vec()
        } else {
            return Err(CudaError::with_detail(
                -10,
                "vlm_img_mask length mismatch",
                format!(
                    "got {}, expected {txt_seq_len} or {}",
                    vlm_img_mask.len(),
                    txt_seq_len + target_slots
                ),
            ));
        };

        let shapes_arr: Vec<[usize; 3]> = img_shapes.iter().map(|&(f, h, w)| [f, h, w]).collect();
        let mut image_pad_mask = Vec::with_capacity(txt_seq_len + img_tokens);
        for &is_img in &full_vlm_mask {
            let count = if is_img { IMG_TOKENS_PER_SLOT } else { 1 };
            image_pad_mask.extend(std::iter::repeat_n(is_img, count));
        }
        let total_seq_len = image_pad_mask.len();
        let (image_ids, target_token_mask_bool) =
            build_token_metadata(&image_pad_mask, &shapes_arr)
                .map_err(|e| CudaError::with_detail(-10, "build_token_metadata", e.to_string()))?;
        let prefix_len = target_token_mask_bool.iter().filter(|&&b| !b).count();
        let segments = prefix_segments(&image_ids, prefix_len);
        let (cos_full, sin_full) = compute_qwen_image_21_rope_cos_sin(
            &shapes_arr,
            &image_pad_mask,
            self.cfg.axes_dims_rope,
            10000.0,
        )
        .map_err(|e| {
            CudaError::with_detail(-10, "compute_qwen_image_21_rope_cos_sin", e.to_string())
        })?;

        // Build `joint_row_map`: `>= 0` selects `txt_proj` row, `< 0` selects `img_proj` row `(-1 - code)`.
        let mut joint_row_map = Vec::with_capacity(total_seq_len);
        let mut img_cursor = 0i32;
        for (slot_idx, &is_img) in full_vlm_mask.iter().enumerate() {
            if is_img {
                for _ in 0..IMG_TOKENS_PER_SLOT {
                    joint_row_map.push(-1 - img_cursor);
                    img_cursor += 1;
                }
            } else {
                joint_row_map.push(slot_idx as i32);
            }
        }
        if img_cursor as usize != img_tokens {
            return Err(CudaError::with_detail(
                -10,
                "expanded image slots do not match total latent tokens",
                format!("slots expanded to {img_cursor}, but img_tokens = {img_tokens}"),
            ));
        }

        let target_mask_u8: Vec<u8> = target_token_mask_bool
            .iter()
            .map(|&b| u8::from(b))
            .collect();

        let mut img_all_f16 = Vec::with_capacity(img_tokens * in_dim);
        img_all_f16.extend_from_slice(cond_latents_f16);
        img_all_f16.extend_from_slice(target_latents_f16);

        if let Some(ref mut cache) = kv_cache {
            let half_head = self.cfg.attention_head_dim / 2;
            let cos_tgt = &cos_full[prefix_len * half_head..];
            let sin_tgt = &sin_full[prefix_len * half_head..];
            cache.ensure_geometry(prefix_len, target_tokens, cos_tgt, sin_tgt, stream)?;
        }

        let marlin_ws_ints = CudaDevice::props_for(self.device_id)?
            .multi_processor_count
            .max(1) as usize;
        let lse_scratch_bytes = total_seq_len * self.heads_local * std::mem::size_of::<f32>();
        let mut ws = self.workspace.lock().unwrap_or_else(|e| e.into_inner());
        let marlin_ws_addr = {
            let need = marlin_ws_ints * 4;
            let init_zero = ws.marlin_ws.as_ref().is_none_or(|b| b.len_bytes < need);
            let addr = QwenImage21Workspace::ensure_buf(&mut ws.marlin_ws, self.device_id, need)?;
            if init_zero {
                ws.marlin_ws.as_ref().unwrap().zero_async(stream)?;
            }
            addr
        };
        let had_scratch_bytes = self
            .blocks
            .first()
            .map(|b| {
                [
                    &b.to_q,
                    &b.to_k,
                    &b.to_v,
                    &b.to_out,
                    &b.mlp_gate_up,
                    &b.mlp_out,
                ]
                .into_iter()
                .filter_map(|p| p.input_transform_workspace_bytes(total_seq_len))
                .max()
                .unwrap_or(0)
            })
            .unwrap_or(0);
        let exec_scratch_bytes = lse_scratch_bytes.max(had_scratch_bytes).max(64 * 1024);
        let exec_scratch_addr = QwenImage21Workspace::ensure_buf(
            &mut ws.exec_scratch,
            self.device_id,
            exec_scratch_bytes,
        )?;
        let ex = stream.exec(exec_scratch_addr, exec_scratch_bytes);

        let num_mod_rows = if self.cfg.causal_condition {
            2usize
        } else {
            1usize
        };
        let timesteps_host: &[f32] = if self.cfg.causal_condition {
            &[timestep, 0.0]
        } else {
            std::slice::from_ref(&timestep)
        };

        let d_img_in = QwenImage21Workspace::ensure_buf(
            &mut ws.d_img_in,
            self.device_id,
            img_all_f16.len() * 2,
        )?;
        let d_img_proj = QwenImage21Workspace::ensure_buf(
            &mut ws.d_img_proj,
            self.device_id,
            img_tokens * inner_dim * 2,
        )?;
        let d_enc_in = QwenImage21Workspace::ensure_buf(
            &mut ws.d_enc_in,
            self.device_id,
            encoder_hidden_f16.len() * 2,
        )?;
        let d_txt_norm = QwenImage21Workspace::ensure_buf(
            &mut ws.d_txt_norm,
            self.device_id,
            txt_seq_len * ctx_dim * 2,
        )?;
        let d_txt_mid = QwenImage21Workspace::ensure_buf(
            &mut ws.d_txt_mid,
            self.device_id,
            txt_seq_len * inner_dim * 2,
        )?;
        let d_txt_proj = QwenImage21Workspace::ensure_buf(
            &mut ws.d_txt_proj,
            self.device_id,
            txt_seq_len * inner_dim * 2,
        )?;
        let d_row_map = QwenImage21Workspace::ensure_buf(
            &mut ws.d_row_map,
            self.device_id,
            joint_row_map.len() * 4,
        )?;
        let d_target_mask = QwenImage21Workspace::ensure_buf(
            &mut ws.d_target_mask,
            self.device_id,
            target_mask_u8.len(),
        )?;
        let d_cos_full = QwenImage21Workspace::ensure_buf(
            &mut ws.d_cos_full,
            self.device_id,
            cos_full.len() * 4,
        )?;
        let d_sin_full = QwenImage21Workspace::ensure_buf(
            &mut ws.d_sin_full,
            self.device_id,
            sin_full.len() * 4,
        )?;
        let d_timesteps = QwenImage21Workspace::ensure_buf(
            &mut ws.d_timesteps,
            self.device_id,
            num_mod_rows * 4,
        )?;
        let d_t_sin = QwenImage21Workspace::ensure_buf(
            &mut ws.d_t_sin,
            self.device_id,
            num_mod_rows * 256 * 2,
        )?;
        let d_temb_mid = QwenImage21Workspace::ensure_buf(
            &mut ws.d_temb_mid,
            self.device_id,
            num_mod_rows * inner_dim * 2,
        )?;
        let d_temb = QwenImage21Workspace::ensure_buf(
            &mut ws.d_temb,
            self.device_id,
            num_mod_rows * inner_dim * 2,
        )?;
        let d_temb_silu = QwenImage21Workspace::ensure_buf(
            &mut ws.d_temb_silu,
            self.device_id,
            num_mod_rows * inner_dim * 2,
        )?;
        let d_modulation = QwenImage21Workspace::ensure_buf(
            &mut ws.d_modulation,
            self.device_id,
            num_mod_rows * 4 * inner_dim * 2,
        )?;
        let d_norm_out_scale = QwenImage21Workspace::ensure_buf(
            &mut ws.d_norm_out_scale,
            self.device_id,
            num_mod_rows * inner_dim * 2,
        )?;
        let d_joint = QwenImage21Workspace::ensure_buf(
            &mut ws.d_joint,
            self.device_id,
            total_seq_len * inner_dim * 2,
        )?;
        let d_norm = QwenImage21Workspace::ensure_buf(
            &mut ws.d_norm,
            self.device_id,
            total_seq_len * inner_dim * 2,
        )?;
        let d_q = QwenImage21Workspace::ensure_buf(
            &mut ws.d_q,
            self.device_id,
            total_seq_len * self.qkv_dim_local * 2,
        )?;
        let d_k = QwenImage21Workspace::ensure_buf(
            &mut ws.d_k,
            self.device_id,
            total_seq_len * self.qkv_dim_local * 2,
        )?;
        let d_v = QwenImage21Workspace::ensure_buf(
            &mut ws.d_v,
            self.device_id,
            total_seq_len * self.qkv_dim_local * 2,
        )?;
        let d_attn_out = QwenImage21Workspace::ensure_buf(
            &mut ws.d_attn_out,
            self.device_id,
            total_seq_len * self.qkv_dim_local * 2,
        )?;
        let d_branch_out = QwenImage21Workspace::ensure_buf(
            &mut ws.d_branch_out,
            self.device_id,
            total_seq_len * inner_dim * 2,
        )?;
        let d_gate_up = QwenImage21Workspace::ensure_buf(
            &mut ws.d_gate_up,
            self.device_id,
            total_seq_len * 2 * self.mlp_local * 2,
        )?;
        let d_mlp_interm = QwenImage21Workspace::ensure_buf(
            &mut ws.d_mlp_interm,
            self.device_id,
            total_seq_len * self.mlp_local * 2,
        )?;
        let d_out = QwenImage21Workspace::ensure_buf(
            &mut ws.d_out,
            self.device_id,
            total_seq_len * out_dim * 2,
        )?;

        ws.d_img_in
            .as_ref()
            .unwrap()
            .copy_from_host(&img_all_f16, stream)?;
        ws.d_enc_in
            .as_ref()
            .unwrap()
            .copy_from_host(encoder_hidden_f16, stream)?;
        ws.d_row_map
            .as_ref()
            .unwrap()
            .copy_from_host(&joint_row_map, stream)?;
        ws.d_target_mask
            .as_ref()
            .unwrap()
            .copy_from_host(&target_mask_u8, stream)?;
        ws.d_cos_full
            .as_ref()
            .unwrap()
            .copy_from_host(&cos_full, stream)?;
        ws.d_sin_full
            .as_ref()
            .unwrap()
            .copy_from_host(&sin_full, stream)?;
        ws.d_timesteps
            .as_ref()
            .unwrap()
            .copy_from_host(timesteps_host, stream)?;

        // 1. Project image latents and text embeddings, then assemble joint_hidden_states
        self.img_in.forward(
            &ex,
            cublas,
            marlin_ws_addr,
            marlin_ws_ints,
            d_img_in,
            d_img_proj,
            img_tokens,
        )?;

        check_rc(
            // SAFETY: `d_enc_in` and `d_txt_norm` span `txt_seq_len * ctx_dim * 2` bytes; `txt_norm_w` spans `ctx_dim * 2` bytes.
            unsafe {
                ffi::warp34_gemma_rms_norm_f16(
                    &ex,
                    d_enc_in as *const c_void,
                    self.txt_norm_w as *const c_void,
                    d_txt_norm as *mut c_void,
                    core::ptr::null(),
                    txt_seq_len as i32,
                    ctx_dim as i32,
                    eps,
                )
            },
            "warp34_gemma_rms_norm_f16(txt_in.text_norm)",
        )?;
        self.txt_in_layer.forward(
            &ex,
            cublas,
            marlin_ws_addr,
            marlin_ws_ints,
            d_txt_norm,
            d_txt_mid,
            txt_seq_len,
        )?;
        check_rc(
            // SAFETY: `d_txt_mid` spans `txt_seq_len * inner_dim * 2` bytes; in-place tanh-GELU without bias.
            unsafe {
                ffi::warp34_vision_bias_gelu_f16(
                    &ex,
                    d_txt_mid as *const c_void,
                    core::ptr::null(),
                    d_txt_mid as *mut c_void,
                    txt_seq_len as i32,
                    inner_dim as i32,
                    1,
                )
            },
            "warp34_vision_bias_gelu_f16(txt_in.act)",
        )?;
        self.txt_out_layer.forward(
            &ex,
            cublas,
            marlin_ws_addr,
            marlin_ws_ints,
            d_txt_mid,
            d_txt_proj,
            txt_seq_len,
        )?;

        check_rc(
            // SAFETY: `d_txt_proj`, `d_img_proj`, `d_row_map`, and `d_joint` match the validated row counts.
            unsafe {
                ffi::warp34_qwen_image_build_joint_hidden_f16(
                    &ex,
                    d_txt_proj as *const c_void,
                    d_img_proj as *const c_void,
                    d_row_map as *const i32,
                    d_joint as *mut c_void,
                    total_seq_len as i32,
                    txt_seq_len as i32,
                    img_tokens as i32,
                    inner_dim as i32,
                )
            },
            "warp34_qwen_image_build_joint_hidden_f16",
        )?;

        // 2. Timestep embedding + shared modulation
        self.run_timestep_and_modulation(
            &ex,
            cublas,
            marlin_ws_addr,
            marlin_ws_ints,
            d_timesteps,
            d_t_sin,
            d_temb_mid,
            d_temb,
            d_temb_silu,
            d_modulation,
            d_norm_out_scale,
            num_mod_rows,
        )?;

        let mask_ptr = if self.cfg.causal_condition {
            d_target_mask as *const u8
        } else {
            core::ptr::null()
        };
        let cache_write_len = if kv_cache.is_some() { prefix_len } else { 0 };

        // 3. Transformer blocks
        for (layer_idx, block) in self.blocks.iter().enumerate() {
            check_rc(
                // SAFETY: `d_joint` and `d_norm` span `total_seq_len * inner_dim * 2` bytes; `d_modulation` has stride `4 * inner_dim`.
                unsafe {
                    ffi::warp34_qwen_image_adaln_modulate_f16(
                        &ex,
                        d_joint as *const c_void,
                        d_modulation as *const c_void,
                        mask_ptr,
                        d_norm as *mut c_void,
                        total_seq_len as i32,
                        inner_dim as i32,
                        (4 * inner_dim) as i32,
                        0,
                        eps,
                    )
                },
                "warp34_qwen_image_adaln_modulate_f16(img_norm1)",
            )?;

            let qkv_share_h64 = block.to_q.can_fuse_h64_from_producer()
                && block.to_k.can_fuse_h64_from_producer()
                && block.to_v.can_fuse_h64_from_producer();
            block.to_q.forward(
                &ex,
                cublas,
                marlin_ws_addr,
                marlin_ws_ints,
                d_norm,
                d_q,
                total_seq_len,
            )?;
            if qkv_share_h64 {
                block.to_k.forward_prerotated(
                    &ex,
                    cublas,
                    marlin_ws_addr,
                    marlin_ws_ints,
                    ex.scratch_ptr,
                    d_k,
                    total_seq_len,
                )?;
                block.to_v.forward_prerotated(
                    &ex,
                    cublas,
                    marlin_ws_addr,
                    marlin_ws_ints,
                    ex.scratch_ptr,
                    d_v,
                    total_seq_len,
                )?;
            } else {
                block.to_k.forward(
                    &ex,
                    cublas,
                    marlin_ws_addr,
                    marlin_ws_ints,
                    d_norm,
                    d_k,
                    total_seq_len,
                )?;
                block.to_v.forward(
                    &ex,
                    cublas,
                    marlin_ws_addr,
                    marlin_ws_ints,
                    d_norm,
                    d_v,
                    total_seq_len,
                )?;
            }

            let cache_k_ptr = kv_cache
                .as_ref()
                .map(|c| c.layer_k[layer_idx].addr() as *mut c_void)
                .unwrap_or(core::ptr::null_mut());

            check_rc(
                // SAFETY: `d_q` and `d_k` span `total_seq_len * qkv_dim_local * 2` bytes; `d_cos_full`/`d_sin_full` span `total_seq_len * 64 * 4` bytes.
                unsafe {
                    ffi::warp34_qwen_image_qk_rmsnorm_complex_rope_d128_f16(
                        &ex,
                        d_q as *const c_void,
                        d_k as *const c_void,
                        block.norm_q_w as *const c_void,
                        block.norm_k_w as *const c_void,
                        d_cos_full as *const f32,
                        d_sin_full as *const f32,
                        d_q as *mut c_void,
                        d_k as *mut c_void,
                        cache_k_ptr,
                        total_seq_len as i32,
                        self.heads_local as i32,
                        cache_write_len as i32,
                        eps,
                    )
                },
                "warp34_qwen_image_qk_rmsnorm_complex_rope_d128_f16(extract)",
            )?;

            if let Some(ref cache) = kv_cache
                && prefix_len > 0
            {
                let prefix_bytes = prefix_len * self.qkv_dim_local * 2;
                check_rc(
                    // SAFETY: `cache.layer_v[layer_idx]` and `d_v` both hold at least `prefix_bytes` bytes.
                    unsafe {
                        ffi::warp34_memcpy_d2d_async(
                            cache.layer_v[layer_idx].addr() as *mut c_void,
                            d_v as *const c_void,
                            prefix_bytes,
                            stream.handle,
                        )
                    },
                    "QwenImage21 prefix V cache write",
                )?;
            }

            // Block-causal FA2 across prefix segments + target suffix
            for &(start, end, is_text) in &segments {
                let seg_len = end - start;
                if seg_len == 0 {
                    continue;
                }
                let byte_off = (start * self.qkv_dim_local * 2) as u64;
                check_rc(
                    // SAFETY: `d_q + byte_off` and `d_attn_out + byte_off` are 16-byte aligned and span `seg_len` tokens;
                    // `d_k` and `d_v` span `end` tokens.
                    unsafe {
                        ffi::warp34_mha_fa2_d128_f16(
                            &ex,
                            (d_q + byte_off) as *const c_void,
                            d_k as *const c_void,
                            d_v as *const c_void,
                            (d_attn_out + byte_off) as *mut c_void,
                            1,
                            seg_len as i32,
                            end as i32,
                            self.heads_local as i32,
                            self.heads_local as i32,
                            i32::from(is_text),
                        )
                    },
                    "warp34_mha_fa2_d128_f16(prefix segment)",
                )?;
            }
            if target_tokens > 0 {
                let tgt_off = (prefix_len * self.qkv_dim_local * 2) as u64;
                check_rc(
                    // SAFETY: `d_q + tgt_off` and `d_attn_out + tgt_off` span `target_tokens` rows; `d_k` and `d_v` span `total_seq_len` rows.
                    unsafe {
                        ffi::warp34_mha_fa2_d128_f16(
                            &ex,
                            (d_q + tgt_off) as *const c_void,
                            d_k as *const c_void,
                            d_v as *const c_void,
                            (d_attn_out + tgt_off) as *mut c_void,
                            1,
                            target_tokens as i32,
                            total_seq_len as i32,
                            self.heads_local as i32,
                            self.heads_local as i32,
                            0,
                        )
                    },
                    "warp34_mha_fa2_d128_f16(target block)",
                )?;
            }

            block.to_out.forward(
                &ex,
                cublas,
                marlin_ws_addr,
                marlin_ws_ints,
                d_attn_out,
                d_branch_out,
                total_seq_len,
            )?;
            if self.tp_size > 1
                && let Some(ref mut ar) = allreduce_fn
            {
                ar(d_branch_out, total_seq_len * inner_dim, stream)?;
            }

            check_rc(
                // SAFETY: `d_joint` and `d_branch_out` span `total_seq_len * inner_dim * 2` bytes; `gate1` is at offset `inner_dim`.
                unsafe {
                    ffi::warp34_qwen_image_gated_residual_f16(
                        &ex,
                        d_joint as *mut c_void,
                        d_branch_out as *const c_void,
                        d_modulation as *const c_void,
                        mask_ptr,
                        total_seq_len as i32,
                        inner_dim as i32,
                        (4 * inner_dim) as i32,
                        inner_dim as i32,
                    )
                },
                "warp34_qwen_image_gated_residual_f16(attn)",
            )?;

            check_rc(
                // SAFETY: `d_joint` and `d_norm` span `total_seq_len * inner_dim * 2` bytes; `scale2` is at offset `2 * inner_dim`.
                unsafe {
                    ffi::warp34_qwen_image_adaln_modulate_f16(
                        &ex,
                        d_joint as *const c_void,
                        d_modulation as *const c_void,
                        mask_ptr,
                        d_norm as *mut c_void,
                        total_seq_len as i32,
                        inner_dim as i32,
                        (4 * inner_dim) as i32,
                        (2 * inner_dim) as i32,
                        eps,
                    )
                },
                "warp34_qwen_image_adaln_modulate_f16(img_norm2)",
            )?;

            block.mlp_gate_up.forward(
                &ex,
                cublas,
                marlin_ws_addr,
                marlin_ws_ints,
                d_norm,
                d_gate_up,
                total_seq_len,
            )?;
            check_rc(
                // SAFETY: `d_gate_up` spans `total_seq_len * 2 * mlp_local * 2` bytes; `d_mlp_interm` spans `total_seq_len * mlp_local * 2` bytes.
                unsafe {
                    ffi::warp34_silu_and_mul_f16(
                        &ex,
                        d_gate_up as *const c_void,
                        d_mlp_interm as *mut c_void,
                        core::ptr::null(),
                        total_seq_len as i32,
                        self.mlp_local as i32,
                    )
                },
                "warp34_silu_and_mul_f16(QwenImage21 MLP)",
            )?;
            block.mlp_out.forward(
                &ex,
                cublas,
                marlin_ws_addr,
                marlin_ws_ints,
                d_mlp_interm,
                d_branch_out,
                total_seq_len,
            )?;
            if self.tp_size > 1
                && let Some(ref mut ar) = allreduce_fn
            {
                ar(d_branch_out, total_seq_len * inner_dim, stream)?;
            }

            check_rc(
                // SAFETY: `d_joint` and `d_branch_out` span `total_seq_len * inner_dim * 2` bytes; `gate2` is at offset `3 * inner_dim`.
                unsafe {
                    ffi::warp34_qwen_image_gated_residual_f16(
                        &ex,
                        d_joint as *mut c_void,
                        d_branch_out as *const c_void,
                        d_modulation as *const c_void,
                        mask_ptr,
                        total_seq_len as i32,
                        inner_dim as i32,
                        (4 * inner_dim) as i32,
                        (3 * inner_dim) as i32,
                    )
                },
                "warp34_qwen_image_gated_residual_f16(mlp)",
            )?;
        }

        // 4. Final norm_out + proj_out
        check_rc(
            // SAFETY: `d_joint` and `d_norm` span `total_seq_len * inner_dim * 2` bytes; `d_norm_out_scale` has stride `inner_dim`.
            unsafe {
                ffi::warp34_qwen_image_adaln_modulate_f16(
                    &ex,
                    d_joint as *const c_void,
                    d_norm_out_scale as *const c_void,
                    mask_ptr,
                    d_norm as *mut c_void,
                    total_seq_len as i32,
                    inner_dim as i32,
                    inner_dim as i32,
                    0,
                    eps,
                )
            },
            "warp34_qwen_image_adaln_modulate_f16(norm_out)",
        )?;
        self.proj_out.forward(
            &ex,
            cublas,
            marlin_ws_addr,
            marlin_ws_ints,
            d_norm,
            d_out,
            total_seq_len,
        )?;

        let mut full_output_f16 = vec![0u16; total_seq_len * out_dim];
        check_rc(
            // SAFETY: `full_output_f16` and `d_out` both span `total_seq_len * out_dim * 2` bytes.
            unsafe {
                ffi::warp34_memcpy_d2h_async(
                    full_output_f16.as_mut_ptr() as *mut c_void,
                    d_out as *const c_void,
                    full_output_f16.len() * 2,
                    stream.handle,
                )
            },
            "QwenImage21 extract output D2H",
        )?;
        stream.synchronize()?;

        let target_output_f16 = full_output_f16[prefix_len * out_dim..].to_vec();
        Ok(QwenImage21ExtractOutput {
            full_output_f16,
            target_output_f16,
            prefix_len,
            target_tokens,
        })
    }

    /// Runs a `"cached"` target-only denoising step (`kv_cache_mode = "cached"`) against `kv_cache`,
    /// optionally capturing and replaying a full-step `CudaGraph` across Steps `1..N-1`.
    pub fn forward_cached(
        &self,
        target_latents_f16: &[u16],
        timestep: f32,
        kv_cache: &mut QwenImage21KvCacheGpu,
        use_cuda_graph: bool,
        cublas: &CublasHandle,
        stream: &CudaStream,
    ) -> Result<Vec<u16>> {
        self.forward_cached_with_allreduce::<fn(u64, usize, &CudaStream) -> Result<()>>(
            target_latents_f16,
            timestep,
            kv_cache,
            use_cuda_graph,
            cublas,
            stream,
            None,
        )
    }

    /// Runs a `"cached"` target-only denoising step with an optional tensor-parallel allreduce callback.
    #[allow(clippy::too_many_arguments)]
    pub fn forward_cached_with_allreduce<F>(
        &self,
        target_latents_f16: &[u16],
        timestep: f32,
        kv_cache: &mut QwenImage21KvCacheGpu,
        use_cuda_graph: bool,
        cublas: &CublasHandle,
        stream: &CudaStream,
        mut allreduce_fn: Option<F>,
    ) -> Result<Vec<u16>>
    where
        F: FnMut(u64, usize, &CudaStream) -> Result<()>,
    {
        let _dev = CudaDevice::set_current(self.device_id)?;
        let prefix_len = kv_cache.prefix_len;
        let target_tokens = kv_cache.target_tokens;
        let total_kv_tokens = prefix_len + target_tokens;
        let in_dim = self.cfg.in_channels * self.cfg.patch_size * self.cfg.patch_size;
        let out_dim = self.cfg.out_channels * self.cfg.patch_size * self.cfg.patch_size;
        let inner_dim = self.cfg.inner_dim();
        let eps = self.cfg.eps;

        if target_tokens == 0 || target_latents_f16.len() != target_tokens * in_dim {
            return Err(CudaError::with_detail(
                -10,
                "forward_cached target_latents_f16 length mismatch",
                format!(
                    "expected {} ({target_tokens} * {in_dim}), got {}",
                    target_tokens * in_dim,
                    target_latents_f16.len()
                ),
            ));
        }

        let d_cos_target = kv_cache
            .d_cos_target
            .as_ref()
            .ok_or_else(|| CudaError::new(-10, "kv_cache missing d_cos_target"))?
            .addr();
        let d_sin_target = kv_cache
            .d_sin_target
            .as_ref()
            .ok_or_else(|| CudaError::new(-10, "kv_cache missing d_sin_target"))?
            .addr();

        let marlin_ws_ints = CudaDevice::props_for(self.device_id)?
            .multi_processor_count
            .max(1) as usize;
        let lse_scratch_bytes = total_kv_tokens * self.heads_local * std::mem::size_of::<f32>();
        let mut ws = self.workspace.lock().unwrap_or_else(|e| e.into_inner());
        let marlin_ws_addr = {
            let need = marlin_ws_ints * 4;
            let init_zero = ws.marlin_ws.as_ref().is_none_or(|b| b.len_bytes < need);
            let addr = QwenImage21Workspace::ensure_buf(&mut ws.marlin_ws, self.device_id, need)?;
            if init_zero {
                ws.marlin_ws.as_ref().unwrap().zero_async(stream)?;
            }
            addr
        };
        let had_scratch_bytes = self
            .blocks
            .first()
            .map(|b| {
                [
                    &b.to_q,
                    &b.to_k,
                    &b.to_v,
                    &b.to_out,
                    &b.mlp_gate_up,
                    &b.mlp_out,
                ]
                .into_iter()
                .filter_map(|p| p.input_transform_workspace_bytes(target_tokens))
                .max()
                .unwrap_or(0)
            })
            .unwrap_or(0);
        let exec_scratch_bytes = lse_scratch_bytes.max(had_scratch_bytes).max(64 * 1024);
        let exec_scratch_addr = QwenImage21Workspace::ensure_buf(
            &mut ws.exec_scratch,
            self.device_id,
            exec_scratch_bytes,
        )?;
        let ex = stream.exec(exec_scratch_addr, exec_scratch_bytes);

        let d_img_in = QwenImage21Workspace::ensure_buf(
            &mut ws.d_img_in,
            self.device_id,
            target_latents_f16.len() * 2,
        )?;
        let d_timesteps = QwenImage21Workspace::ensure_buf(&mut ws.d_timesteps, self.device_id, 8)?;
        let d_t_sin = QwenImage21Workspace::ensure_buf(&mut ws.d_t_sin, self.device_id, 256 * 2)?;
        let d_temb_mid =
            QwenImage21Workspace::ensure_buf(&mut ws.d_temb_mid, self.device_id, inner_dim * 2)?;
        let d_temb =
            QwenImage21Workspace::ensure_buf(&mut ws.d_temb, self.device_id, inner_dim * 2)?;
        let d_temb_silu =
            QwenImage21Workspace::ensure_buf(&mut ws.d_temb_silu, self.device_id, inner_dim * 2)?;
        let d_modulation = QwenImage21Workspace::ensure_buf(
            &mut ws.d_modulation,
            self.device_id,
            4 * inner_dim * 2,
        )?;
        let d_norm_out_scale = QwenImage21Workspace::ensure_buf(
            &mut ws.d_norm_out_scale,
            self.device_id,
            inner_dim * 2,
        )?;
        let d_joint = QwenImage21Workspace::ensure_buf(
            &mut ws.d_joint,
            self.device_id,
            target_tokens * inner_dim * 2,
        )?;
        let d_norm = QwenImage21Workspace::ensure_buf(
            &mut ws.d_norm,
            self.device_id,
            target_tokens * inner_dim * 2,
        )?;
        let d_q = QwenImage21Workspace::ensure_buf(
            &mut ws.d_q,
            self.device_id,
            target_tokens * self.qkv_dim_local * 2,
        )?;
        let d_k = QwenImage21Workspace::ensure_buf(
            &mut ws.d_k,
            self.device_id,
            target_tokens * self.qkv_dim_local * 2,
        )?;
        let d_attn_out = QwenImage21Workspace::ensure_buf(
            &mut ws.d_attn_out,
            self.device_id,
            target_tokens * self.qkv_dim_local * 2,
        )?;
        let d_branch_out = QwenImage21Workspace::ensure_buf(
            &mut ws.d_branch_out,
            self.device_id,
            target_tokens * inner_dim * 2,
        )?;
        let d_gate_up = QwenImage21Workspace::ensure_buf(
            &mut ws.d_gate_up,
            self.device_id,
            target_tokens * 2 * self.mlp_local * 2,
        )?;
        let d_mlp_interm = QwenImage21Workspace::ensure_buf(
            &mut ws.d_mlp_interm,
            self.device_id,
            target_tokens * self.mlp_local * 2,
        )?;
        let d_out = QwenImage21Workspace::ensure_buf(
            &mut ws.d_out,
            self.device_id,
            target_tokens * out_dim * 2,
        )?;

        ws.d_img_in
            .as_ref()
            .unwrap()
            .copy_from_host(target_latents_f16, stream)?;
        ws.d_timesteps
            .as_ref()
            .unwrap()
            .copy_from_host(std::slice::from_ref(&timestep), stream)?;

        let layer_k_addrs: Vec<u64> = kv_cache.layer_k.iter().map(DeviceBuffer::addr).collect();
        let layer_v_addrs: Vec<u64> = kv_cache.layer_v.iter().map(DeviceBuffer::addr).collect();

        let mut emit_cached_step = || -> Result<()> {
            self.img_in.forward(
                &ex,
                cublas,
                marlin_ws_addr,
                marlin_ws_ints,
                d_img_in,
                d_joint,
                target_tokens,
            )?;

            self.run_timestep_and_modulation(
                &ex,
                cublas,
                marlin_ws_addr,
                marlin_ws_ints,
                d_timesteps,
                d_t_sin,
                d_temb_mid,
                d_temb,
                d_temb_silu,
                d_modulation,
                d_norm_out_scale,
                1,
            )?;

            let tail_byte_off = (prefix_len * self.qkv_dim_local * 2) as u64;
            for (layer_idx, block) in self.blocks.iter().enumerate() {
                check_rc(
                    // SAFETY: `d_joint` and `d_norm` span `target_tokens * inner_dim * 2` bytes; `target_token_mask` is null (all tokens use row 0).
                    unsafe {
                        ffi::warp34_qwen_image_adaln_modulate_f16(
                            &ex,
                            d_joint as *const c_void,
                            d_modulation as *const c_void,
                            core::ptr::null(),
                            d_norm as *mut c_void,
                            target_tokens as i32,
                            inner_dim as i32,
                            (4 * inner_dim) as i32,
                            0,
                            eps,
                        )
                    },
                    "warp34_qwen_image_adaln_modulate_f16(cached img_norm1)",
                )?;

                let k_cache_full = layer_k_addrs[layer_idx];
                let v_cache_full = layer_v_addrs[layer_idx];
                let k_cache_tail = k_cache_full + tail_byte_off;
                let v_cache_tail = v_cache_full + tail_byte_off;

                let qkv_share_h64 = block.to_q.can_fuse_h64_from_producer()
                    && block.to_k.can_fuse_h64_from_producer()
                    && block.to_v.can_fuse_h64_from_producer();
                block.to_q.forward(
                    &ex,
                    cublas,
                    marlin_ws_addr,
                    marlin_ws_ints,
                    d_norm,
                    d_q,
                    target_tokens,
                )?;
                if qkv_share_h64 {
                    block.to_k.forward_prerotated(
                        &ex,
                        cublas,
                        marlin_ws_addr,
                        marlin_ws_ints,
                        ex.scratch_ptr,
                        d_k,
                        target_tokens,
                    )?;
                    // Write target V directly into the tail of `layer_v[layer_idx]` (`[prefix_len..prefix_len + target_tokens]`).
                    block.to_v.forward_prerotated(
                        &ex,
                        cublas,
                        marlin_ws_addr,
                        marlin_ws_ints,
                        ex.scratch_ptr,
                        v_cache_tail,
                        target_tokens,
                    )?;
                } else {
                    block.to_k.forward(
                        &ex,
                        cublas,
                        marlin_ws_addr,
                        marlin_ws_ints,
                        d_norm,
                        d_k,
                        target_tokens,
                    )?;
                    // Write target V directly into the tail of `layer_v[layer_idx]` (`[prefix_len..prefix_len + target_tokens]`).
                    block.to_v.forward(
                        &ex,
                        cublas,
                        marlin_ws_addr,
                        marlin_ws_ints,
                        d_norm,
                        v_cache_tail,
                        target_tokens,
                    )?;
                }

                // Write rotated target K directly into `k_cache_tail`.
                check_rc(
                    // SAFETY: `d_q` and `d_k` span `target_tokens * qkv_dim_local * 2` bytes; `k_cache_tail` spans the remaining `target_tokens` rows of `layer_k`.
                    unsafe {
                        ffi::warp34_qwen_image_qk_rmsnorm_complex_rope_d128_f16(
                            &ex,
                            d_q as *const c_void,
                            d_k as *const c_void,
                            block.norm_q_w as *const c_void,
                            block.norm_k_w as *const c_void,
                            d_cos_target as *const f32,
                            d_sin_target as *const f32,
                            d_q as *mut c_void,
                            k_cache_tail as *mut c_void,
                            core::ptr::null_mut(),
                            target_tokens as i32,
                            self.heads_local as i32,
                            0,
                            eps,
                        )
                    },
                    "warp34_qwen_image_qk_rmsnorm_complex_rope_d128_f16(cached)",
                )?;

                check_rc(
                    // SAFETY: `d_q` (`target_tokens`), `k_cache_full` (`total_kv_tokens`), `v_cache_full` (`total_kv_tokens`), and `d_attn_out` (`target_tokens`) are valid 16-byte aligned device buffers.
                    unsafe {
                        ffi::warp34_mha_fa2_d128_f16(
                            &ex,
                            d_q as *const c_void,
                            k_cache_full as *const c_void,
                            v_cache_full as *const c_void,
                            d_attn_out as *mut c_void,
                            1,
                            target_tokens as i32,
                            total_kv_tokens as i32,
                            self.heads_local as i32,
                            self.heads_local as i32,
                            0,
                        )
                    },
                    "warp34_mha_fa2_d128_f16(cached full)",
                )?;

                block.to_out.forward(
                    &ex,
                    cublas,
                    marlin_ws_addr,
                    marlin_ws_ints,
                    d_attn_out,
                    d_branch_out,
                    target_tokens,
                )?;
                if self.tp_size > 1
                    && let Some(ref mut ar) = allreduce_fn
                {
                    ar(d_branch_out, target_tokens * inner_dim, stream)?;
                }

                check_rc(
                    // SAFETY: `d_joint` and `d_branch_out` span `target_tokens * inner_dim * 2` bytes.
                    unsafe {
                        ffi::warp34_qwen_image_gated_residual_f16(
                            &ex,
                            d_joint as *mut c_void,
                            d_branch_out as *const c_void,
                            d_modulation as *const c_void,
                            core::ptr::null(),
                            target_tokens as i32,
                            inner_dim as i32,
                            (4 * inner_dim) as i32,
                            inner_dim as i32,
                        )
                    },
                    "warp34_qwen_image_gated_residual_f16(cached attn)",
                )?;

                check_rc(
                    // SAFETY: `d_joint` and `d_norm` span `target_tokens * inner_dim * 2` bytes.
                    unsafe {
                        ffi::warp34_qwen_image_adaln_modulate_f16(
                            &ex,
                            d_joint as *const c_void,
                            d_modulation as *const c_void,
                            core::ptr::null(),
                            d_norm as *mut c_void,
                            target_tokens as i32,
                            inner_dim as i32,
                            (4 * inner_dim) as i32,
                            (2 * inner_dim) as i32,
                            eps,
                        )
                    },
                    "warp34_qwen_image_adaln_modulate_f16(cached img_norm2)",
                )?;

                block.mlp_gate_up.forward(
                    &ex,
                    cublas,
                    marlin_ws_addr,
                    marlin_ws_ints,
                    d_norm,
                    d_gate_up,
                    target_tokens,
                )?;
                check_rc(
                    // SAFETY: `d_gate_up` and `d_mlp_interm` match `target_tokens * mlp_local`.
                    unsafe {
                        ffi::warp34_silu_and_mul_f16(
                            &ex,
                            d_gate_up as *const c_void,
                            d_mlp_interm as *mut c_void,
                            core::ptr::null(),
                            target_tokens as i32,
                            self.mlp_local as i32,
                        )
                    },
                    "warp34_silu_and_mul_f16(cached MLP)",
                )?;
                block.mlp_out.forward(
                    &ex,
                    cublas,
                    marlin_ws_addr,
                    marlin_ws_ints,
                    d_mlp_interm,
                    d_branch_out,
                    target_tokens,
                )?;
                if self.tp_size > 1
                    && let Some(ref mut ar) = allreduce_fn
                {
                    ar(d_branch_out, target_tokens * inner_dim, stream)?;
                }

                check_rc(
                    // SAFETY: `d_joint` and `d_branch_out` span `target_tokens * inner_dim * 2` bytes.
                    unsafe {
                        ffi::warp34_qwen_image_gated_residual_f16(
                            &ex,
                            d_joint as *mut c_void,
                            d_branch_out as *const c_void,
                            d_modulation as *const c_void,
                            core::ptr::null(),
                            target_tokens as i32,
                            inner_dim as i32,
                            (4 * inner_dim) as i32,
                            (3 * inner_dim) as i32,
                        )
                    },
                    "warp34_qwen_image_gated_residual_f16(cached mlp)",
                )?;
            }

            check_rc(
                // SAFETY: `d_joint` and `d_norm` span `target_tokens * inner_dim * 2` bytes.
                unsafe {
                    ffi::warp34_qwen_image_adaln_modulate_f16(
                        &ex,
                        d_joint as *const c_void,
                        d_norm_out_scale as *const c_void,
                        core::ptr::null(),
                        d_norm as *mut c_void,
                        target_tokens as i32,
                        inner_dim as i32,
                        inner_dim as i32,
                        0,
                        eps,
                    )
                },
                "warp34_qwen_image_adaln_modulate_f16(cached norm_out)",
            )?;
            self.proj_out.forward(
                &ex,
                cublas,
                marlin_ws_addr,
                marlin_ws_ints,
                d_norm,
                d_out,
                target_tokens,
            )
        };

        let geom = (prefix_len, target_tokens);
        if use_cuda_graph && self.tp_size == 1 {
            if let Some((gp, gt, ref graph)) = kv_cache.cached_graph
                && (gp, gt) == geom
            {
                graph.launch(stream)?;
            } else if kv_cache.graph_warmed_geom == Some(geom) {
                let graph = CudaGraph::capture(stream, &mut emit_cached_step)?;
                graph.launch(stream)?;
                kv_cache.cached_graph = Some((prefix_len, target_tokens, graph));
            } else {
                emit_cached_step()?;
                kv_cache.graph_warmed_geom = Some(geom);
            }
        } else {
            emit_cached_step()?;
        }

        let mut target_output_f16 = vec![0u16; target_tokens * out_dim];
        check_rc(
            // SAFETY: `target_output_f16` and `d_out` both span `target_tokens * out_dim * 2` bytes.
            unsafe {
                ffi::warp34_memcpy_d2h_async(
                    target_output_f16.as_mut_ptr() as *mut c_void,
                    d_out as *const c_void,
                    target_output_f16.len() * 2,
                    stream.handle,
                )
            },
            "QwenImage21 cached output D2H",
        )?;
        stream.synchronize()?;
        Ok(target_output_f16)
    }

    #[allow(clippy::too_many_arguments)]
    fn run_timestep_and_modulation(
        &self,
        ex: &ffi::Warp34Exec,
        cublas: &CublasHandle,
        marlin_ws_addr: u64,
        marlin_ws_ints: usize,
        d_timesteps: u64,
        d_t_sin: u64,
        d_temb_mid: u64,
        d_temb: u64,
        d_temb_silu: u64,
        d_modulation: u64,
        d_norm_out_scale: u64,
        num_mod_rows: usize,
    ) -> Result<()> {
        let inner_dim = self.cfg.inner_dim();
        check_rc(
            // SAFETY: `d_timesteps` has `num_mod_rows` f32 elements and `d_t_sin` has `num_mod_rows * 256` FP16 elements.
            unsafe {
                ffi::warp34_qwen_image_timestep_sinusoidal_f16(
                    ex,
                    d_timesteps as *const f32,
                    d_t_sin as *mut c_void,
                    num_mod_rows as i32,
                    256,
                    10000.0,
                    1000.0,
                )
            },
            "warp34_qwen_image_timestep_sinusoidal_f16",
        )?;
        self.time_linear_1.forward(
            ex,
            cublas,
            marlin_ws_addr,
            marlin_ws_ints,
            d_t_sin,
            d_temb_mid,
            num_mod_rows,
        )?;
        check_rc(
            // SAFETY: `d_temb_mid` spans `num_mod_rows * inner_dim * 2` bytes.
            unsafe {
                ffi::warp34_qwen_image_silu_f16(
                    ex,
                    d_temb_mid as *const c_void,
                    d_temb_mid as *mut c_void,
                    (num_mod_rows * inner_dim) as i64,
                )
            },
            "warp34_qwen_image_silu_f16(timestep_embedder.act)",
        )?;
        self.time_linear_2.forward(
            ex,
            cublas,
            marlin_ws_addr,
            marlin_ws_ints,
            d_temb_mid,
            d_temb,
            num_mod_rows,
        )?;
        check_rc(
            // SAFETY: `d_temb` and `d_temb_silu` span `num_mod_rows * inner_dim * 2` bytes.
            unsafe {
                ffi::warp34_qwen_image_silu_f16(
                    ex,
                    d_temb as *const c_void,
                    d_temb_silu as *mut c_void,
                    (num_mod_rows * inner_dim) as i64,
                )
            },
            "warp34_qwen_image_silu_f16(modulation.0 / norm_out.silu)",
        )?;
        self.modulation_linear.forward(
            ex,
            cublas,
            marlin_ws_addr,
            marlin_ws_ints,
            d_temb_silu,
            d_modulation,
            num_mod_rows,
        )?;
        self.norm_out_linear.forward(
            ex,
            cublas,
            marlin_ws_addr,
            marlin_ws_ints,
            d_temb_silu,
            d_norm_out_scale,
            num_mod_rows,
        )
    }
}

/// Multi-GPU `TP=2` execution engine for `QwenImage21Transformer2DModel` across devices `0` and `1`.
#[cfg(feature = "cuda")]
pub struct QwenImage21TransformerTp2Engine {
    pub rank0: QwenImage21TransformerEngine,
    pub rank1: QwenImage21TransformerEngine,
    pub stream0: CudaStream,
    pub stream1: CudaStream,
    pub cublas0: CublasHandle,
    pub cublas1: CublasHandle,
    pub comm: Arc<Tp2Collectives>,
}

#[cfg(feature = "cuda")]
fn chunked_all_reduce_f16(
    comm: &Tp2Collectives,
    ex: &ffi::Warp34Exec,
    d_buf: u64,
    rank: i32,
    n_elems: usize,
) -> Result<()> {
    let max_chunk = (comm.max_stage_bytes_per_epoch / 2).min(i32::MAX as usize) & !7usize;
    if max_chunk == 0 {
        return Err(CudaError::new(
            -10,
            "Tp2Collectives max_stage_bytes_per_epoch too small for chunked_all_reduce_f16",
        ));
    }
    let mut off = 0usize;
    while off < n_elems {
        let chunk = (n_elems - off).min(max_chunk);
        let ptr = d_buf + (off * 2) as u64;
        comm.all_reduce_f16(ex, ptr, ptr, rank, chunk as i32)?;
        off += chunk;
    }
    Ok(())
}

#[cfg(feature = "cuda")]
impl QwenImage21TransformerTp2Engine {
    pub fn load_with_prefix(
        archive: &SafetensorsArchive,
        cfg: &QwenImage21TransformerConfig,
        quant: &QuantScheme,
        weight_prefix: &str,
        max_tokens: usize,
    ) -> Result<Self> {
        prepare_execution_plan(0)?;
        prepare_execution_plan(1)?;
        let stream0 = CudaStream::new(0)?;
        let cublas0 = CublasHandle::new(&stream0)?;
        let rank0 = QwenImage21TransformerEngine::load_with_prefix(
            archive,
            cfg,
            quant,
            weight_prefix,
            0,
            0,
            2,
            &stream0,
        )?;

        let stream1 = CudaStream::new(1)?;
        let cublas1 = CublasHandle::new(&stream1)?;
        let rank1 = QwenImage21TransformerEngine::load_with_prefix(
            archive,
            cfg,
            quant,
            weight_prefix,
            1,
            1,
            2,
            &stream1,
        )?;

        let max_stage_bytes = (max_tokens.max(16384) * cfg.inner_dim() * 2).max(256 * 1024);
        let comm = Arc::new(Tp2Collectives::new_for_tp(2, max_stage_bytes)?);
        Ok(Self {
            rank0,
            rank1,
            stream0,
            stream1,
            cublas0,
            cublas1,
            comm,
        })
    }

    pub fn new_kv_caches(&self) -> (QwenImage21KvCacheGpu, QwenImage21KvCacheGpu) {
        (self.rank0.new_kv_cache(), self.rank1.new_kv_cache())
    }

    #[allow(clippy::too_many_arguments)]
    pub fn forward_extract(
        &self,
        cond_latents_f16: &[u16],
        target_latents_f16: &[u16],
        encoder_hidden_f16: &[u16],
        vlm_img_mask: &[bool],
        img_shapes: &[(usize, usize, usize)],
        timestep: f32,
        kv_caches: Option<(&mut QwenImage21KvCacheGpu, &mut QwenImage21KvCacheGpu)>,
    ) -> Result<QwenImage21ExtractOutput> {
        let comm0 = Arc::clone(&self.comm);
        let comm1 = Arc::clone(&self.comm);
        let (c0, c1) = match kv_caches {
            Some((a, b)) => (Some(a), Some(b)),
            None => (None, None),
        };
        std::thread::scope(|s| {
            let h1 = s.spawn(|| {
                let _dev = CudaDevice::set_current(1)?;
                self.rank1.forward_extract_with_allreduce(
                    cond_latents_f16,
                    target_latents_f16,
                    encoder_hidden_f16,
                    vlm_img_mask,
                    img_shapes,
                    timestep,
                    c1,
                    &self.cublas1,
                    &self.stream1,
                    Some(|d_buf: u64, n_elems: usize, st: &CudaStream| {
                        let ex = st.exec(0, 0);
                        chunked_all_reduce_f16(&comm1, &ex, d_buf, 1, n_elems)
                    }),
                )
            });
            let _dev = CudaDevice::set_current(0)?;
            let r0 = self.rank0.forward_extract_with_allreduce(
                cond_latents_f16,
                target_latents_f16,
                encoder_hidden_f16,
                vlm_img_mask,
                img_shapes,
                timestep,
                c0,
                &self.cublas0,
                &self.stream0,
                Some(|d_buf: u64, n_elems: usize, st: &CudaStream| {
                    let ex = st.exec(0, 0);
                    chunked_all_reduce_f16(&comm0, &ex, d_buf, 0, n_elems)
                }),
            );
            let r1 = h1
                .join()
                .map_err(|_| CudaError::new(-10, "TP=2 rank 1 thread panicked"))?;
            r1?;
            r0
        })
    }

    pub fn forward_cached(
        &self,
        target_latents_f16: &[u16],
        timestep: f32,
        kv_caches: (&mut QwenImage21KvCacheGpu, &mut QwenImage21KvCacheGpu),
    ) -> Result<Vec<u16>> {
        let comm0 = Arc::clone(&self.comm);
        let comm1 = Arc::clone(&self.comm);
        let (c0, c1) = kv_caches;
        std::thread::scope(|s| {
            let h1 = s.spawn(|| {
                let _dev = CudaDevice::set_current(1)?;
                self.rank1.forward_cached_with_allreduce(
                    target_latents_f16,
                    timestep,
                    c1,
                    false,
                    &self.cublas1,
                    &self.stream1,
                    Some(|d_buf: u64, n_elems: usize, st: &CudaStream| {
                        let ex = st.exec(0, 0);
                        chunked_all_reduce_f16(&comm1, &ex, d_buf, 1, n_elems)
                    }),
                )
            });
            let _dev = CudaDevice::set_current(0)?;
            let r0 = self.rank0.forward_cached_with_allreduce(
                target_latents_f16,
                timestep,
                c0,
                false,
                &self.cublas0,
                &self.stream0,
                Some(|d_buf: u64, n_elems: usize, st: &CudaStream| {
                    let ex = st.exec(0, 0);
                    chunked_all_reduce_f16(&comm0, &ex, d_buf, 0, n_elems)
                }),
            );
            let r1 = h1
                .join()
                .map_err(|_| CudaError::new(-10, "TP=2 rank 1 thread panicked"))?;
            r1?;
            r0
        })
    }
}
