//! Native `FP16` CUDA execution engine for `AutoencoderKLQwenImage21` (64-channel RGBA VAE).
//!
//! Implements the full `diffusers` `AutoencoderKLQwenImage21` encoder and decoder for single-frame
//! images (`T = 1`):
//! - 2D `Conv2d` (`1x1` direct cuBLAS GEMM and `3x3` strip-mined `im2col + cuBLAS GEMM`)
//! - Channel-wise `QwenImage21RMS_norm` (`F.normalize(x.float(), dim=1, eps=1e-12).to(fp16) * sqrt(C) * gamma`)
//!   with optional fused `SiLU`
//! - `QwenImage21ResidualBlock` (`norm1 -> silu -> conv1 -> norm2 -> silu -> conv2 + conv_shortcut`)
//! - `QwenImage21AttentionBlock` (`norm -> to_qkv -> 2D single-head SDPA -> proj + residual`)
//! - `QwenImage21ResidualDownBlock` (`resnets -> optional ZeroPad2d(0,1,0,1)+Conv2d(s=2) + AvgDown3D`)
//! - `QwenImage21ResidualUpBlock` (`resnets -> optional Nearest2x+Conv2d(3) + DupUp3D(first_chunk=True)`)
//! - Per-channel `latents_mean` / `latents_std` normalization and denormalization
//! - Spatial tiled encode and decode (`tiled_encode` / `tiled_decode` with `blend_v` and `blend_h`)

use core::ffi::c_void;

use warp34_kernels::ffi;

use crate::cuda::{
    CublasHandle, CudaDevice, CudaError, CudaStream, DeviceBuffer, Result, check_rc, f16_to_f32,
    f32_to_f16, prepare_qwen_image_execution_plan,
};
use crate::loader::SafetensorsArchive;
use crate::qwen_image::QwenImage21VaeConfig;

const MAX_IM2COL_ELEMS_DEFAULT: usize = 192 * 1024 * 1024; // 384 MiB FP16 workspace cap when L2 < 48 MiB
const MAX_IM2COL_ELEMS_LARGE_L2: usize = 16 * 1024 * 1024; // 32 MiB FP16 L2-resident strip cap when L2 >= 48 MiB
const LARGE_L2_THRESHOLD_BYTES: i32 = 48 * 1024 * 1024;

/// Spatial tiling configuration for `AutoencoderKLQwenImage21`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct QwenImage21VaeTilingConfig {
    /// Whether spatial tiling is enabled.
    pub enabled: bool,
    /// Minimum sample height in pixels before tiling is triggered (`256` by default).
    pub tile_sample_min_height: usize,
    /// Minimum sample width in pixels before tiling is triggered (`256` by default).
    pub tile_sample_min_width: usize,
    /// Vertical stride between adjacent sample tiles in pixels (`192` by default).
    pub tile_sample_stride_height: usize,
    /// Horizontal stride between adjacent sample tiles in pixels (`192` by default).
    pub tile_sample_stride_width: usize,
}

impl Default for QwenImage21VaeTilingConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            tile_sample_min_height: 256,
            tile_sample_min_width: 256,
            tile_sample_stride_height: 192,
            tile_sample_stride_width: 192,
        }
    }
}

/// Normalizes raw VAE posterior mode latents `[z_dim, spatial]` by `(z - mean) / std` in `FP16`,
/// matching `QwenImage21Pipeline._encode_vae_image`.
///
/// # Panics
///
/// Panics if `z_chw.len() != z_dim * spatial` or if `mean.len() < z_dim` / `std.len() < z_dim`.
pub fn normalize_latents_f16(
    z_chw: &[u16],
    z_dim: usize,
    spatial: usize,
    mean: &[f32],
    std: &[f32],
) -> Vec<u16> {
    let total = z_dim
        .checked_mul(spatial)
        .expect("normalize_latents_f16 z_dim * spatial overflow");
    assert_eq!(z_chw.len(), total);
    assert!(mean.len() >= z_dim && std.len() >= z_dim);
    let mut out = vec![0u16; total];
    for c in 0..z_dim {
        let m_h = f16_to_f32(f32_to_f16(mean[c]));
        let s_h = f16_to_f32(f32_to_f16(std[c]));
        let row_in = &z_chw[c * spatial..(c + 1) * spatial];
        let row_out = &mut out[c * spatial..(c + 1) * spatial];
        for i in 0..spatial {
            let z = f16_to_f32(row_in[i]);
            let diff = f16_to_f32(f32_to_f16(z - m_h));
            row_out[i] = f32_to_f16(diff / s_h);
        }
    }
    out
}

/// Denormalizes packed/unpacked VAE latents `[z_dim, spatial]` by `z * std + mean` in `FP16`,
/// matching `QwenImage21Pipeline.__call__`.
///
/// # Panics
///
/// Panics if `z_chw.len() != z_dim * spatial` or if `mean.len() < z_dim` / `std.len() < z_dim`.
pub fn denormalize_latents_f16(
    z_chw: &[u16],
    z_dim: usize,
    spatial: usize,
    mean: &[f32],
    std: &[f32],
) -> Vec<u16> {
    let total = z_dim
        .checked_mul(spatial)
        .expect("denormalize_latents_f16 z_dim * spatial overflow");
    assert_eq!(z_chw.len(), total);
    assert!(mean.len() >= z_dim && std.len() >= z_dim);
    let mut out = vec![0u16; total];
    for c in 0..z_dim {
        let m_h = f16_to_f32(f32_to_f16(mean[c]));
        let s_h = f16_to_f32(f32_to_f16(std[c]));
        let row_in = &z_chw[c * spatial..(c + 1) * spatial];
        let row_out = &mut out[c * spatial..(c + 1) * spatial];
        for i in 0..spatial {
            let z = f16_to_f32(row_in[i]);
            let prod = f16_to_f32(f32_to_f16(z * s_h));
            row_out[i] = f32_to_f16(prod + m_h);
        }
    }
    out
}

/// Converts an `[H, W, 4]` `u8` RGBA image into a `[4, H, W]` `FP16` tensor in `[-1.0, 1.0]`,
/// matching `VaeImageProcessor.preprocess`.
///
/// # Panics
///
/// Panics if `rgba_hwc.len() != height * width * 4`.
pub fn rgba_u8_to_vae_chw_f16(rgba_hwc: &[u8], height: usize, width: usize) -> Vec<u16> {
    let spatial = height
        .checked_mul(width)
        .expect("rgba_u8_to_vae_chw_f16 height * width overflow");
    let total = spatial
        .checked_mul(4)
        .expect("rgba_u8_to_vae_chw_f16 spatial * 4 overflow");
    assert_eq!(rgba_hwc.len(), total);
    let mut out = vec![0u16; total];
    for idx in 0..spatial {
        for c in 0..4 {
            let v = (rgba_hwc[idx * 4 + c] as f32) / 255.0;
            out[c * spatial + idx] = f32_to_f16(2.0 * v - 1.0);
        }
    }
    out
}

/// Converts a `[4, H, W]` `FP16` VAE output tensor in `[-1.0, 1.0]` into an `[H, W, 4]` `u8` RGBA
/// (or `[H, W, 3]` `u8` RGB if `keep_alpha == false`) image buffer, matching `VaeImageProcessor.postprocess`.
///
/// # Panics
///
/// Panics if `chw_f16.len() < (if keep_alpha { 4 } else { 3 }) * height * width`.
pub fn vae_chw_f16_to_rgba_u8(
    chw_f16: &[u16],
    height: usize,
    width: usize,
    keep_alpha: bool,
) -> Vec<u8> {
    let spatial = height
        .checked_mul(width)
        .expect("vae_chw_f16_to_rgba_u8 height * width overflow");
    let out_c: usize = if keep_alpha { 4 } else { 3 };
    let required = out_c
        .checked_mul(spatial)
        .expect("vae_chw_f16_to_rgba_u8 channels * spatial overflow");
    assert!(chw_f16.len() >= required);
    let mut out = vec![0u8; required];
    for idx in 0..spatial {
        for c in 0..out_c {
            let v = f16_to_f32(chw_f16[c * spatial + idx]);
            let u = ((v * 0.5 + 0.5).clamp(0.0, 1.0) * 255.0).round() as u8;
            out[idx * out_c + c] = u;
        }
    }
    out
}

fn load_tensor_f16(archive: &SafetensorsArchive, name: &str) -> Result<Vec<u16>> {
    let key = archive
        .resolve_name(name)
        .ok_or_else(|| CudaError::with_detail(-2, "missing QwenImage21 VAE tensor", name))?;
    archive.read_f16_bits(&key).map_err(|e| {
        CudaError::with_detail(-3, "failed reading QwenImage21 VAE tensor", e.to_string())
    })
}

fn alloc_from_u16_slice(device_id: i32, stream: &CudaStream, host: &[u16]) -> Result<DeviceBuffer> {
    let bytes = std::mem::size_of_val(host);
    let buf = DeviceBuffer::alloc(device_id, bytes.max(16))?;
    if bytes > 0 {
        check_rc(
            // SAFETY: `buf.ptr` has `bytes` allocated on `device_id`, and `host` has `bytes` bytes.
            unsafe {
                ffi::warp34_memcpy_h2d_async(
                    buf.ptr,
                    host.as_ptr() as *const c_void,
                    bytes,
                    stream.handle,
                )
            },
            "warp34_memcpy_h2d_async(vae weight)",
        )?;
    }
    Ok(buf)
}

struct VaeConv2dGpu {
    weight: DeviceBuffer,
    bias: DeviceBuffer,
    in_c: usize,
    out_c: usize,
    k_h: usize,
    k_w: usize,
    pad_top: usize,
    pad_bottom: usize,
    pad_left: usize,
    pad_right: usize,
    stride_h: usize,
    stride_w: usize,
}

impl VaeConv2dGpu {
    #[allow(clippy::too_many_arguments)]
    fn load(
        archive: &SafetensorsArchive,
        prefix: &str,
        in_c: usize,
        out_c: usize,
        k_h: usize,
        k_w: usize,
        pad_top: usize,
        pad_left: usize,
        stride_h: usize,
        stride_w: usize,
        device_id: i32,
        stream: &CudaStream,
    ) -> Result<Self> {
        Self::load_asymmetric_pad(
            archive, prefix, in_c, out_c, k_h, k_w, pad_top, pad_top, pad_left, pad_left, stride_h,
            stride_w, device_id, stream,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn load_scaled(
        archive: &SafetensorsArchive,
        prefix: &str,
        in_c: usize,
        out_c: usize,
        k_h: usize,
        k_w: usize,
        pad_top: usize,
        pad_left: usize,
        stride_h: usize,
        stride_w: usize,
        w_scale: f32,
        b_scale: f32,
        device_id: i32,
        stream: &CudaStream,
    ) -> Result<Self> {
        Self::load_asymmetric_pad_scaled(
            archive, prefix, in_c, out_c, k_h, k_w, pad_top, pad_top, pad_left, pad_left, stride_h,
            stride_w, w_scale, b_scale, device_id, stream,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn load_asymmetric_pad(
        archive: &SafetensorsArchive,
        prefix: &str,
        in_c: usize,
        out_c: usize,
        k_h: usize,
        k_w: usize,
        pad_top: usize,
        pad_bottom: usize,
        pad_left: usize,
        pad_right: usize,
        stride_h: usize,
        stride_w: usize,
        device_id: i32,
        stream: &CudaStream,
    ) -> Result<Self> {
        Self::load_asymmetric_pad_scaled(
            archive, prefix, in_c, out_c, k_h, k_w, pad_top, pad_bottom, pad_left, pad_right,
            stride_h, stride_w, 1.0, 1.0, device_id, stream,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn load_asymmetric_pad_scaled(
        archive: &SafetensorsArchive,
        prefix: &str,
        in_c: usize,
        out_c: usize,
        k_h: usize,
        k_w: usize,
        pad_top: usize,
        pad_bottom: usize,
        pad_left: usize,
        pad_right: usize,
        stride_h: usize,
        stride_w: usize,
        w_scale: f32,
        b_scale: f32,
        device_id: i32,
        stream: &CudaStream,
    ) -> Result<Self> {
        let mut w_host = load_tensor_f16(archive, &format!("{prefix}.weight"))?;
        let mut b_host = load_tensor_f16(archive, &format!("{prefix}.bias"))?;
        if w_host.len() != out_c * in_c * k_h * k_w || b_host.len() != out_c {
            return Err(CudaError::new(-5, "VAE Conv2d weight/bias shape mismatch"));
        }
        if (w_scale - 1.0).abs() > 1e-7 {
            for w in &mut w_host {
                *w = f32_to_f16(f16_to_f32(*w) * w_scale);
            }
        }
        if (b_scale - 1.0).abs() > 1e-7 {
            for b in &mut b_host {
                *b = f32_to_f16(f16_to_f32(*b) * b_scale);
            }
        }
        Ok(Self {
            weight: alloc_from_u16_slice(device_id, stream, &w_host)?,
            bias: alloc_from_u16_slice(device_id, stream, &b_host)?,
            in_c,
            out_c,
            k_h,
            k_w,
            pad_top,
            pad_bottom,
            pad_left,
            pad_right,
            stride_h,
            stride_w,
        })
    }
}

struct VaeResnetBlockGpu {
    in_dim: usize,
    out_dim: usize,
    norm1_gamma: DeviceBuffer,
    conv1: VaeConv2dGpu,
    norm2_gamma: DeviceBuffer,
    conv2: VaeConv2dGpu,
    conv_shortcut: Option<VaeConv2dGpu>,
}

impl VaeResnetBlockGpu {
    fn load(
        archive: &SafetensorsArchive,
        prefix: &str,
        in_dim: usize,
        out_dim: usize,
        device_id: i32,
        stream: &CudaStream,
    ) -> Result<Self> {
        Self::load_scaled(archive, prefix, in_dim, out_dim, 1.0, device_id, stream)
    }

    fn load_scaled(
        archive: &SafetensorsArchive,
        prefix: &str,
        in_dim: usize,
        out_dim: usize,
        highway_scale: f32,
        device_id: i32,
        stream: &CudaStream,
    ) -> Result<Self> {
        let norm1_host = load_tensor_f16(archive, &format!("{prefix}.norm1.gamma"))?;
        let norm2_host = load_tensor_f16(archive, &format!("{prefix}.norm2.gamma"))?;
        if norm1_host.len() != in_dim || norm2_host.len() != out_dim {
            return Err(CudaError::new(-5, "VAE Resnet norm gamma shape mismatch"));
        }
        let conv1 = VaeConv2dGpu::load(
            archive,
            &format!("{prefix}.conv1"),
            in_dim,
            out_dim,
            3,
            3,
            1,
            1,
            1,
            1,
            device_id,
            stream,
        )?;
        let conv2 = VaeConv2dGpu::load_scaled(
            archive,
            &format!("{prefix}.conv2"),
            out_dim,
            out_dim,
            3,
            3,
            1,
            1,
            1,
            1,
            highway_scale,
            highway_scale,
            device_id,
            stream,
        )?;
        let conv_shortcut = if in_dim != out_dim {
            Some(VaeConv2dGpu::load_scaled(
                archive,
                &format!("{prefix}.conv_shortcut"),
                in_dim,
                out_dim,
                1,
                1,
                0,
                0,
                1,
                1,
                1.0,
                highway_scale,
                device_id,
                stream,
            )?)
        } else {
            None
        };
        Ok(Self {
            in_dim,
            out_dim,
            norm1_gamma: alloc_from_u16_slice(device_id, stream, &norm1_host)?,
            conv1,
            norm2_gamma: alloc_from_u16_slice(device_id, stream, &norm2_host)?,
            conv2,
            conv_shortcut,
        })
    }
}

struct VaeAttentionBlockGpu {
    dim: usize,
    norm_gamma: DeviceBuffer,
    to_qkv: VaeConv2dGpu,
    proj: VaeConv2dGpu,
}

impl VaeAttentionBlockGpu {
    fn load(
        archive: &SafetensorsArchive,
        prefix: &str,
        dim: usize,
        device_id: i32,
        stream: &CudaStream,
    ) -> Result<Self> {
        let norm_host = load_tensor_f16(archive, &format!("{prefix}.norm.gamma"))?;
        if norm_host.len() != dim {
            return Err(CudaError::new(
                -5,
                "VAE Attention norm gamma shape mismatch",
            ));
        }
        let to_qkv = VaeConv2dGpu::load(
            archive,
            &format!("{prefix}.to_qkv"),
            dim,
            dim * 3,
            1,
            1,
            0,
            0,
            1,
            1,
            device_id,
            stream,
        )?;
        let proj = VaeConv2dGpu::load(
            archive,
            &format!("{prefix}.proj"),
            dim,
            dim,
            1,
            1,
            0,
            0,
            1,
            1,
            device_id,
            stream,
        )?;
        Ok(Self {
            dim,
            norm_gamma: alloc_from_u16_slice(device_id, stream, &norm_host)?,
            to_qkv,
            proj,
        })
    }
}

struct VaeMidBlockGpu {
    resnet0: VaeResnetBlockGpu,
    attn0: VaeAttentionBlockGpu,
    resnet1: VaeResnetBlockGpu,
}

impl VaeMidBlockGpu {
    fn load(
        archive: &SafetensorsArchive,
        prefix: &str,
        dim: usize,
        device_id: i32,
        stream: &CudaStream,
    ) -> Result<Self> {
        Ok(Self {
            resnet0: VaeResnetBlockGpu::load(
                archive,
                &format!("{prefix}.resnets.0"),
                dim,
                dim,
                device_id,
                stream,
            )?,
            attn0: VaeAttentionBlockGpu::load(
                archive,
                &format!("{prefix}.attentions.0"),
                dim,
                device_id,
                stream,
            )?,
            resnet1: VaeResnetBlockGpu::load(
                archive,
                &format!("{prefix}.resnets.1"),
                dim,
                dim,
                device_id,
                stream,
            )?,
        })
    }
}

struct VaeDownBlockGpu {
    in_dim: usize,
    out_dim: usize,
    factor_t: usize,
    factor_s: usize,
    resnets: Vec<VaeResnetBlockGpu>,
    downsampler_conv: Option<VaeConv2dGpu>,
}

struct VaeUpBlockGpu {
    in_dim: usize,
    out_dim: usize,
    factor_t: usize,
    factor_s: usize,
    skip_scale: f32,
    resnets: Vec<VaeResnetBlockGpu>,
    upsampler_conv: Option<VaeConv2dGpu>,
}

/// Device-resident weights for `AutoencoderKLQwenImage21`.
pub struct QwenImage21VaeWeightsGpu {
    device_id: i32,
    cfg: QwenImage21VaeConfig,
    enc_conv_in: VaeConv2dGpu,
    enc_down_blocks: Vec<VaeDownBlockGpu>,
    enc_mid_block: VaeMidBlockGpu,
    enc_norm_out_gamma: DeviceBuffer,
    enc_conv_out: VaeConv2dGpu,
    quant_conv: VaeConv2dGpu,
    post_quant_conv: VaeConv2dGpu,
    dec_conv_in: VaeConv2dGpu,
    dec_mid_block: VaeMidBlockGpu,
    dec_up_blocks: Vec<VaeUpBlockGpu>,
    dec_norm_out_gamma: DeviceBuffer,
    dec_conv_out: VaeConv2dGpu,
}

impl QwenImage21VaeWeightsGpu {
    /// Loads `AutoencoderKLQwenImage21` weights onto `device_id`.
    ///
    /// Pass `prefix = ""` for standard `vae/diffusion_pytorch_model.safetensors` or
    /// `prefix = "vae.weight"` for synthetic oracle fixtures.
    pub fn from_archive(
        archive: &SafetensorsArchive,
        cfg: &QwenImage21VaeConfig,
        prefix: &str,
        device_id: i32,
    ) -> Result<Self> {
        let stream = CudaStream::new(device_id)?;
        let pfx = |name: &str| -> String {
            if prefix.is_empty() {
                name.to_string()
            } else {
                format!("{prefix}.{name}")
            }
        };

        // Encoder dimensions: [base_dim] + [base_dim * u for u in dim_mult]
        let mut enc_dims = Vec::with_capacity(cfg.dim_mult.len() + 1);
        enc_dims.push(cfg.base_dim);
        for &m in &cfg.dim_mult {
            enc_dims.push(cfg.base_dim * m);
        }
        let enc_conv_in = VaeConv2dGpu::load(
            archive,
            &pfx("encoder.conv_in"),
            cfg.in_channels,
            enc_dims[0],
            3,
            3,
            1,
            1,
            1,
            1,
            device_id,
            &stream,
        )?;

        let num_stages = cfg.dim_mult.len();
        let mut enc_down_blocks = Vec::with_capacity(num_stages);
        for i in 0..num_stages {
            let in_dim = enc_dims[i];
            let out_dim = enc_dims[i + 1];
            let down_flag = i != num_stages - 1;
            let temp_down = if down_flag {
                cfg.temperal_downsample.get(i).copied().unwrap_or(false)
            } else {
                false
            };
            let mut resnets = Vec::with_capacity(cfg.num_res_blocks);
            let mut cur_dim = in_dim;
            for r in 0..cfg.num_res_blocks {
                resnets.push(VaeResnetBlockGpu::load(
                    archive,
                    &pfx(&format!("encoder.down_blocks.{i}.resnets.{r}")),
                    cur_dim,
                    out_dim,
                    device_id,
                    &stream,
                )?);
                cur_dim = out_dim;
            }
            let downsampler_conv = if down_flag {
                // Matches `nn.Sequential(nn.ZeroPad2d((0, 1, 0, 1)), nn.Conv2d(..., 3, stride=2, padding=0))`.
                Some(VaeConv2dGpu::load_asymmetric_pad(
                    archive,
                    &pfx(&format!("encoder.down_blocks.{i}.downsampler.resample.1")),
                    out_dim,
                    out_dim,
                    3,
                    3,
                    0,
                    1,
                    0,
                    1,
                    2,
                    2,
                    device_id,
                    &stream,
                )?)
            } else {
                None
            };
            enc_down_blocks.push(VaeDownBlockGpu {
                in_dim,
                out_dim,
                factor_t: if temp_down { 2 } else { 1 },
                factor_s: if down_flag { 2 } else { 1 },
                resnets,
                downsampler_conv,
            });
        }

        let enc_last_dim = *enc_dims.last().unwrap_or(&cfg.base_dim);
        let enc_mid_block = VaeMidBlockGpu::load(
            archive,
            &pfx("encoder.mid_block"),
            enc_last_dim,
            device_id,
            &stream,
        )?;
        let enc_norm_out_host = load_tensor_f16(archive, &pfx("encoder.norm_out.gamma"))?;
        let enc_conv_out = VaeConv2dGpu::load(
            archive,
            &pfx("encoder.conv_out"),
            enc_last_dim,
            cfg.z_dim * 2,
            3,
            3,
            1,
            1,
            1,
            1,
            device_id,
            &stream,
        )?;
        let quant_conv = VaeConv2dGpu::load(
            archive,
            &pfx("quant_conv"),
            cfg.z_dim * 2,
            cfg.z_dim * 2,
            1,
            1,
            0,
            0,
            1,
            1,
            device_id,
            &stream,
        )?;
        let post_quant_conv = VaeConv2dGpu::load(
            archive,
            &pfx("post_quant_conv"),
            cfg.z_dim,
            cfg.z_dim,
            1,
            1,
            0,
            0,
            1,
            1,
            device_id,
            &stream,
        )?;

        // Decoder dimensions: [decoder_base_dim * dim_mult[-1]] + [decoder_base_dim * u for u in rev(dim_mult)]
        let mut dec_dims = Vec::with_capacity(num_stages + 1);
        dec_dims.push(cfg.decoder_base_dim * cfg.dim_mult.last().copied().unwrap_or(1));
        for &m in cfg.dim_mult.iter().rev() {
            dec_dims.push(cfg.decoder_base_dim * m);
        }
        let dec_conv_in = VaeConv2dGpu::load(
            archive,
            &pfx("decoder.conv_in"),
            cfg.z_dim,
            dec_dims[0],
            3,
            3,
            1,
            1,
            1,
            1,
            device_id,
            &stream,
        )?;
        let dec_mid_block = VaeMidBlockGpu::load(
            archive,
            &pfx("decoder.mid_block"),
            dec_dims[0],
            device_id,
            &stream,
        )?;

        let temperal_upsample: Vec<bool> = cfg.temperal_downsample.iter().copied().rev().collect();
        let mut dec_up_blocks = Vec::with_capacity(num_stages);
        // Full 5-stage `AutoencoderKLQwenImage21` (`num_stages >= 5`) reaches `3.62e6` in `up_blocks.3`
        // on transparent RGBA latents (`alpha = 0`), overflowing FP16 (`65,504.0`). Because every ResNet
        // branch (`norm1`) and `decoder.norm_out` start with 0-homogeneous `RMSNorm`, scaling the
        // residual highway by `2^-4 = 1/16` at `up_blocks[2]` and `2^-4 = 1/16` at `up_blocks[3]`
        // (`1/256` total) is an exact power-of-two exponent shift that keeps all activations `<= 14,144`
        // and cancels identically in `decoder.norm_out` (`54.02 dB` PSNR vs BF16, 0 NaNs).
        let mut highway_scale = 1.0f32;
        for i in 0..num_stages {
            let mut in_dim = dec_dims[i];
            let out_dim = dec_dims[i + 1];
            if i > 0 && !cfg.is_residual {
                in_dim /= 2;
            }
            let up_flag = i != num_stages - 1;
            let temp_up = if up_flag {
                temperal_upsample.get(i).copied().unwrap_or(false)
            } else {
                false
            };
            let step_scale = if num_stages >= 5 && up_flag && i + 3 >= num_stages {
                1.0 / 16.0
            } else {
                1.0
            };
            let mut resnets = Vec::with_capacity(cfg.num_res_blocks + 1);
            let mut cur_dim = in_dim;
            for r in 0..=cfg.num_res_blocks {
                resnets.push(VaeResnetBlockGpu::load_scaled(
                    archive,
                    &pfx(&format!("decoder.up_blocks.{i}.resnets.{r}")),
                    cur_dim,
                    out_dim,
                    highway_scale,
                    device_id,
                    &stream,
                )?);
                cur_dim = out_dim;
            }
            let next_highway_scale = highway_scale * step_scale;
            let upsampler_conv = if up_flag {
                Some(VaeConv2dGpu::load_scaled(
                    archive,
                    &pfx(&format!("decoder.up_blocks.{i}.upsampler.resample.1")),
                    out_dim,
                    out_dim,
                    3,
                    3,
                    1,
                    1,
                    1,
                    1,
                    step_scale,
                    next_highway_scale,
                    device_id,
                    &stream,
                )?)
            } else {
                None
            };
            dec_up_blocks.push(VaeUpBlockGpu {
                in_dim,
                out_dim,
                factor_t: if temp_up { 2 } else { 1 },
                factor_s: if up_flag { 2 } else { 1 },
                skip_scale: step_scale,
                resnets,
                upsampler_conv,
            });
            highway_scale = next_highway_scale;
        }

        let dec_last_dim = *dec_dims.last().unwrap_or(&cfg.decoder_base_dim);
        let dec_norm_out_host = load_tensor_f16(archive, &pfx("decoder.norm_out.gamma"))?;
        let dec_conv_out = VaeConv2dGpu::load(
            archive,
            &pfx("decoder.conv_out"),
            dec_last_dim,
            cfg.out_channels,
            3,
            3,
            1,
            1,
            1,
            1,
            device_id,
            &stream,
        )?;
        let enc_norm_out_gamma = alloc_from_u16_slice(device_id, &stream, &enc_norm_out_host)?;
        let dec_norm_out_gamma = alloc_from_u16_slice(device_id, &stream, &dec_norm_out_host)?;
        stream.synchronize()?;

        Ok(Self {
            device_id,
            cfg: cfg.clone(),
            enc_conv_in,
            enc_down_blocks,
            enc_mid_block,
            enc_norm_out_gamma,
            enc_conv_out,
            quant_conv,
            post_quant_conv,
            dec_conv_in,
            dec_mid_block,
            dec_up_blocks,
            dec_norm_out_gamma,
            dec_conv_out,
        })
    }
}

/// GPU execution engine for `AutoencoderKLQwenImage21`.
pub struct QwenImage21VaeEngine {
    stream: CudaStream,
    cublas: CublasHandle,
    weights: QwenImage21VaeWeightsGpu,
    tiling: QwenImage21VaeTilingConfig,
    im2col_ws: DeviceBuffer,
    buf_a: DeviceBuffer,
    buf_b: DeviceBuffer,
    buf_c: DeviceBuffer,
    buf_d: DeviceBuffer,
    buf_e: DeviceBuffer,
    qkv_ws: DeviceBuffer,
    scores_ws: DeviceBuffer,
    probs_ws: DeviceBuffer,
}

impl QwenImage21VaeEngine {
    /// Creates a new `QwenImage21VaeEngine` on `weights.device_id`.
    pub fn new(weights: QwenImage21VaeWeightsGpu) -> Result<Self> {
        let device_id = weights.device_id;
        prepare_qwen_image_execution_plan(device_id)?;
        let stream = CudaStream::new(device_id)?;
        let cublas = CublasHandle::new(&stream)?;
        Ok(Self {
            stream,
            cublas,
            weights,
            tiling: QwenImage21VaeTilingConfig::default(),
            im2col_ws: DeviceBuffer::alloc(device_id, 16)?,
            buf_a: DeviceBuffer::alloc(device_id, 16)?,
            buf_b: DeviceBuffer::alloc(device_id, 16)?,
            buf_c: DeviceBuffer::alloc(device_id, 16)?,
            buf_d: DeviceBuffer::alloc(device_id, 16)?,
            buf_e: DeviceBuffer::alloc(device_id, 16)?,
            qkv_ws: DeviceBuffer::alloc(device_id, 16)?,
            scores_ws: DeviceBuffer::alloc(device_id, 16)?,
            probs_ws: DeviceBuffer::alloc(device_id, 16)?,
        })
    }

    /// Returns the VAE configuration.
    #[must_use]
    pub fn config(&self) -> &QwenImage21VaeConfig {
        &self.weights.cfg
    }

    /// Returns the current spatial tiling configuration.
    #[must_use]
    pub fn tiling(&self) -> QwenImage21VaeTilingConfig {
        self.tiling
    }

    /// Enables or updates spatial tiling (`AutoencoderKLQwenImage21.enable_tiling`).
    pub fn set_tiling(&mut self, tiling: QwenImage21VaeTilingConfig) {
        self.tiling = tiling;
    }

    /// Releases transient GPU activation/im2col scratch buffers back to minimal placeholders.
    pub fn release_workspace(&mut self) -> Result<()> {
        let device_id = self.weights.device_id;
        let _dev = CudaDevice::set_current(device_id)?;
        self.im2col_ws = DeviceBuffer::alloc(device_id, 16)?;
        self.buf_a = DeviceBuffer::alloc(device_id, 16)?;
        self.buf_b = DeviceBuffer::alloc(device_id, 16)?;
        self.buf_c = DeviceBuffer::alloc(device_id, 16)?;
        self.buf_d = DeviceBuffer::alloc(device_id, 16)?;
        self.buf_e = DeviceBuffer::alloc(device_id, 16)?;
        self.qkv_ws = DeviceBuffer::alloc(device_id, 16)?;
        self.scores_ws = DeviceBuffer::alloc(device_id, 16)?;
        self.probs_ws = DeviceBuffer::alloc(device_id, 16)?;
        Ok(())
    }

    fn exec_ctx(&self) -> ffi::Warp34Exec {
        self.stream.exec(0, 0)
    }

    fn ensure_buf_u16(buf: &mut DeviceBuffer, elems: usize) -> Result<()> {
        let bytes = elems * 2;
        if buf.len_bytes < bytes {
            *buf = DeviceBuffer::alloc(buf.device_id, bytes)?;
        }
        Ok(())
    }

    fn ensure_buf_f32(buf: &mut DeviceBuffer, elems: usize) -> Result<()> {
        let bytes = elems * 4;
        if buf.len_bytes < bytes {
            *buf = DeviceBuffer::alloc(buf.device_id, bytes)?;
        }
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    fn run_conv2d(
        ex: &ffi::Warp34Exec,
        cublas: &CublasHandle,
        im2col_ws: &mut DeviceBuffer,
        conv: &VaeConv2dGpu,
        x_ptr: *const c_void,
        residual_ptr: *const c_void,
        out_ptr: *mut c_void,
        h_in: usize,
        w_in: usize,
        clamp_pm1: bool,
    ) -> Result<(usize, usize)> {
        let h_numer = h_in
            .checked_add(conv.pad_top + conv.pad_bottom)
            .and_then(|v| v.checked_sub(conv.k_h))
            .ok_or_else(|| CudaError::new(-10, "conv2d h_in smaller than kernel or overflow"))?;
        let w_numer = w_in
            .checked_add(conv.pad_left + conv.pad_right)
            .and_then(|v| v.checked_sub(conv.k_w))
            .ok_or_else(|| CudaError::new(-10, "conv2d w_in smaller than kernel or overflow"))?;
        let h_out = h_numer / conv.stride_h + 1;
        let w_out = w_numer / conv.stride_w + 1;
        let spatial_out = h_out
            .checked_mul(w_out)
            .ok_or_else(|| CudaError::new(-10, "conv2d spatial_out overflow"))?;

        if conv.k_h == 1
            && conv.k_w == 1
            && conv.pad_top == 0
            && conv.pad_bottom == 0
            && conv.pad_left == 0
            && conv.pad_right == 0
            && conv.stride_h == 1
            && conv.stride_w == 1
        {
            check_rc(
                // SAFETY: `x_ptr`, `conv.weight.ptr`, and `out_ptr` are valid device allocations on `cublas.handle`'s device.
                unsafe {
                    ffi::warp34_cublas_conv_gemm_f16(
                        cublas.handle,
                        x_ptr,
                        conv.weight.ptr as *const c_void,
                        out_ptr,
                        spatial_out as i32,
                        conv.out_c as i32,
                        conv.in_c as i32,
                        spatial_out as i32,
                    )
                },
                "warp34_cublas_conv_gemm_f16 (1x1)",
            )?;
        } else {
            let k_dim = conv.in_c * conv.k_h * conv.k_w;
            let row_col_elems = k_dim * w_out;
            let mut dev_props = ffi::Warp34DeviceProps::default();
            let is_large_l2 = unsafe {
                // SAFETY: `dev_props` is a valid stack `Warp34DeviceProps`.
                ffi::warp34_get_current_device_props(&raw mut dev_props)
            } == 0
                && dev_props.l2_cache_bytes >= LARGE_L2_THRESHOLD_BYTES;
            let max_im2col_elems = if is_large_l2 {
                MAX_IM2COL_ELEMS_LARGE_L2
            } else {
                MAX_IM2COL_ELEMS_DEFAULT
            };
            let raw_chunk_h = (max_im2col_elems / row_col_elems.max(1)).max(1).min(h_out);
            let max_chunk_h = if is_large_l2 && raw_chunk_h < h_out {
                1usize << raw_chunk_h.ilog2()
            } else {
                raw_chunk_h
            };
            let ws_needed = k_dim * max_chunk_h * w_out;
            Self::ensure_buf_u16(im2col_ws, ws_needed)?;

            let mut oh_start = 0usize;
            while oh_start < h_out {
                let chunk_h = (h_out - oh_start).min(max_chunk_h);
                let chunk_spatial = chunk_h * w_out;
                check_rc(
                    // SAFETY: `x_ptr` holds `[in_c, h_in, w_in]` and `im2col_ws` holds `k_dim * chunk_spatial` FP16 elements.
                    unsafe {
                        ffi::warp34_qwen_image_vae_im2col_f16(
                            ex,
                            x_ptr,
                            im2col_ws.ptr,
                            conv.in_c as i32,
                            h_in as i32,
                            w_in as i32,
                            conv.k_h as i32,
                            conv.k_w as i32,
                            conv.pad_top as i32,
                            conv.pad_left as i32,
                            conv.stride_h as i32,
                            conv.stride_w as i32,
                            oh_start as i32,
                            chunk_h as i32,
                            w_out as i32,
                        )
                    },
                    "warp34_qwen_image_vae_im2col_f16",
                )?;
                let dst_strip =
                    // SAFETY: `oh_start * w_out < spatial_out`, within `out_ptr` allocation of `out_c * spatial_out` u16s.
                    unsafe { (out_ptr as *mut u16).add(oh_start * w_out) as *mut c_void };
                check_rc(
                    // SAFETY: `im2col_ws`, `conv.weight`, and `dst_strip` (stride `spatial_out`) are valid device buffers.
                    unsafe {
                        ffi::warp34_cublas_conv_gemm_f16(
                            cublas.handle,
                            im2col_ws.ptr as *const c_void,
                            conv.weight.ptr as *const c_void,
                            dst_strip,
                            chunk_spatial as i32,
                            conv.out_c as i32,
                            k_dim as i32,
                            spatial_out as i32,
                        )
                    },
                    "warp34_cublas_conv_gemm_f16 (3x3)",
                )?;
                oh_start += chunk_h;
            }
        }

        check_rc(
            // SAFETY: `out_ptr`, `conv.bias.ptr`, and optional `residual_ptr` are valid device pointers for `[out_c, spatial_out]`.
            unsafe {
                ffi::warp34_qwen_image_vae_bias_residual_f16(
                    ex,
                    out_ptr as *const c_void,
                    conv.bias.ptr as *const c_void,
                    residual_ptr,
                    out_ptr,
                    conv.out_c as i32,
                    spatial_out as i32,
                    i32::from(clamp_pm1),
                )
            },
            "warp34_qwen_image_vae_bias_residual_f16",
        )?;
        Ok((h_out, w_out))
    }

    #[allow(clippy::too_many_arguments)]
    fn run_resnet(
        ex: &ffi::Warp34Exec,
        cublas: &CublasHandle,
        im2col_ws: &mut DeviceBuffer,
        res: &VaeResnetBlockGpu,
        x_ptr: *const c_void,
        tmp1_ptr: *mut c_void,
        tmp2_ptr: *mut c_void,
        out_ptr: *mut c_void,
        h: usize,
        w: usize,
    ) -> Result<()> {
        let spatial = h * w;
        // 1. norm1 + silu -> tmp1
        check_rc(
            // SAFETY: `x_ptr`, `res.norm1_gamma.ptr`, and `tmp1_ptr` are valid device pointers for `[in_dim, spatial]`.
            unsafe {
                ffi::warp34_qwen_image_vae_rms_norm_f16(
                    ex,
                    x_ptr,
                    res.norm1_gamma.ptr as *const c_void,
                    tmp1_ptr,
                    res.in_dim as i32,
                    spatial as i32,
                    1,
                )
            },
            "resnet norm1+silu",
        )?;
        // 2. conv1(tmp1) -> tmp2
        Self::run_conv2d(
            ex,
            cublas,
            im2col_ws,
            &res.conv1,
            tmp1_ptr as *const c_void,
            core::ptr::null(),
            tmp2_ptr,
            h,
            w,
            false,
        )?;
        // 3. norm2 + silu(tmp2) -> tmp2 (in-place)
        check_rc(
            // SAFETY: `tmp2_ptr` and `res.norm2_gamma.ptr` are valid device pointers for `[out_dim, spatial]`.
            unsafe {
                ffi::warp34_qwen_image_vae_rms_norm_f16(
                    ex,
                    tmp2_ptr as *const c_void,
                    res.norm2_gamma.ptr as *const c_void,
                    tmp2_ptr,
                    res.out_dim as i32,
                    spatial as i32,
                    1,
                )
            },
            "resnet norm2+silu",
        )?;
        // 4. shortcut -> tmp1 (if conv_shortcut or x_ptr aliases out_ptr) or x_ptr directly
        let shortcut_ptr = if let Some(sc) = &res.conv_shortcut {
            Self::run_conv2d(
                ex,
                cublas,
                im2col_ws,
                sc,
                x_ptr,
                core::ptr::null(),
                tmp1_ptr,
                h,
                w,
                false,
            )?;
            tmp1_ptr as *const c_void
        } else if core::ptr::eq(x_ptr, out_ptr as *const c_void) {
            check_rc(
                // SAFETY: `tmp1_ptr` and `x_ptr` hold at least `res.in_dim * spatial * 2` bytes on `ex.stream`.
                unsafe {
                    ffi::warp34_memcpy_d2d_async(
                        tmp1_ptr,
                        x_ptr,
                        res.in_dim * spatial * 2,
                        ex.stream,
                    )
                },
                "resnet shortcut d2d copy",
            )?;
            tmp1_ptr as *const c_void
        } else {
            x_ptr
        };
        // 5. conv2(tmp2) + shortcut_ptr -> out_ptr
        Self::run_conv2d(
            ex,
            cublas,
            im2col_ws,
            &res.conv2,
            tmp2_ptr as *const c_void,
            shortcut_ptr,
            out_ptr,
            h,
            w,
            false,
        )?;
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    fn run_attention(
        ex: &ffi::Warp34Exec,
        cublas: &CublasHandle,
        im2col_ws: &mut DeviceBuffer,
        qkv_ws: &mut DeviceBuffer,
        scores_ws: &mut DeviceBuffer,
        probs_ws: &mut DeviceBuffer,
        attn: &VaeAttentionBlockGpu,
        x_ptr: *const c_void,
        tmp_ptr: *mut c_void,
        out_ptr: *mut c_void,
        h: usize,
        w: usize,
    ) -> Result<()> {
        let spatial = h * w;
        let dim = attn.dim;
        Self::ensure_buf_u16(qkv_ws, 3 * dim * spatial)?;
        Self::ensure_buf_f32(scores_ws, spatial * spatial)?;
        Self::ensure_buf_u16(probs_ws, spatial * spatial)?;

        // 1. norm(x) -> tmp_ptr (no SiLU)
        check_rc(
            // SAFETY: `x_ptr`, `attn.norm_gamma.ptr`, and `tmp_ptr` are valid device pointers for `[dim, spatial]`.
            unsafe {
                ffi::warp34_qwen_image_vae_rms_norm_f16(
                    ex,
                    x_ptr,
                    attn.norm_gamma.ptr as *const c_void,
                    tmp_ptr,
                    dim as i32,
                    spatial as i32,
                    0,
                )
            },
            "attn norm",
        )?;
        // 2. to_qkv(tmp_ptr) -> qkv_ws [3 * dim, spatial]
        Self::run_conv2d(
            ex,
            cublas,
            im2col_ws,
            &attn.to_qkv,
            tmp_ptr as *const c_void,
            core::ptr::null(),
            qkv_ws.ptr,
            h,
            w,
            false,
        )?;
        let q_ptr = qkv_ws.ptr as *const u16;
        // SAFETY: `qkv_ws` was sized to `3 * dim * spatial` u16 elements above.
        let k_ptr = unsafe { q_ptr.add(dim * spatial) };
        // SAFETY: `qkv_ws` was sized to `3 * dim * spatial` u16 elements above.
        let v_ptr = unsafe { q_ptr.add(2 * dim * spatial) };
        let scale = 1.0f32 / (dim as f32).sqrt();

        // 3. scores_ws [spatial, spatial] = scale * (q^T @ k)
        check_rc(
            // SAFETY: `q_ptr`, `k_ptr`, and `scores_ws.ptr` are valid device allocations on `cublas.handle`'s device.
            unsafe {
                ffi::warp34_cublas_vae_qk_f16_f32(
                    cublas.handle,
                    q_ptr as *const c_void,
                    k_ptr as *const c_void,
                    scores_ws.ptr as *mut f32,
                    spatial as i32,
                    dim as i32,
                    scale,
                )
            },
            "warp34_cublas_vae_qk_f16_f32",
        )?;
        // 4. probs_ws [spatial, spatial] = softmax(scores_ws)
        check_rc(
            // SAFETY: `scores_ws` and `probs_ws` hold `spatial * spatial` elements on `ex.stream`.
            unsafe {
                ffi::warp34_qwen_image_vae_softmax_f32_to_f16(
                    ex,
                    scores_ws.ptr as *const f32,
                    probs_ws.ptr,
                    spatial as i32,
                )
            },
            "warp34_qwen_image_vae_softmax_f32_to_f16",
        )?;
        // 5. tmp_ptr [dim, spatial] = v @ probs^T
        check_rc(
            // SAFETY: `probs_ws`, `v_ptr`, and `tmp_ptr` are valid device buffers on `cublas.handle`'s device.
            unsafe {
                ffi::warp34_cublas_vae_sv_f16(
                    cublas.handle,
                    probs_ws.ptr as *const c_void,
                    v_ptr as *const c_void,
                    tmp_ptr,
                    spatial as i32,
                    dim as i32,
                )
            },
            "warp34_cublas_vae_sv_f16",
        )?;
        // 6. proj(tmp_ptr) + x_ptr -> out_ptr
        Self::run_conv2d(
            ex,
            cublas,
            im2col_ws,
            &attn.proj,
            tmp_ptr as *const c_void,
            x_ptr,
            out_ptr,
            h,
            w,
            false,
        )?;
        Ok(())
    }

    fn encode_single_tile(
        &mut self,
        tile_f16: &[u16],
        height: usize,
        width: usize,
    ) -> Result<Vec<u16>> {
        let _dev = CudaDevice::set_current(self.weights.device_id)?;
        let cfg = &self.weights.cfg;
        let max_c = cfg.dim_mult.iter().copied().max().unwrap_or(1).max(1)
            * cfg.base_dim.max(cfg.z_dim * 2);
        let max_elems = max_c * height * width;
        Self::ensure_buf_u16(&mut self.buf_a, max_elems)?;
        Self::ensure_buf_u16(&mut self.buf_b, max_elems)?;
        Self::ensure_buf_u16(&mut self.buf_c, max_elems)?;
        Self::ensure_buf_u16(&mut self.buf_d, max_elems)?;

        check_rc(
            // SAFETY: `buf_a` holds at least `tile_f16.len() * 2` bytes, and `tile_f16` is live.
            unsafe {
                ffi::warp34_memcpy_h2d_async(
                    self.buf_a.ptr,
                    tile_f16.as_ptr() as *const c_void,
                    tile_f16.len() * 2,
                    self.stream.handle,
                )
            },
            "warp34_memcpy_h2d_async(encode_single_tile)",
        )?;
        let ex = self.exec_ctx();

        // conv_in: buf_a -> buf_b
        let (mut cur_h, mut cur_w) = Self::run_conv2d(
            &ex,
            &self.cublas,
            &mut self.im2col_ws,
            &self.weights.enc_conv_in,
            self.buf_a.ptr as *const c_void,
            core::ptr::null(),
            self.buf_b.ptr,
            height,
            width,
            false,
        )?;

        // Current activation is in buf_b.
        for block in &self.weights.enc_down_blocks {
            // Save block input in buf_a for avg_shortcut; run resnets ping-ponging into buf_b.
            let in_bytes = block.in_dim * cur_h * cur_w * 2;
            check_rc(
                // SAFETY: `buf_a` and `buf_b` both hold at least `in_bytes` bytes on device.
                unsafe {
                    ffi::warp34_memcpy_d2d_async(
                        self.buf_a.ptr,
                        self.buf_b.ptr as *const c_void,
                        in_bytes,
                        self.stream.handle,
                    )
                },
                "warp34_memcpy_d2d_async(enc_down_block)",
            )?;

            for res in &block.resnets {
                Self::run_resnet(
                    &ex,
                    &self.cublas,
                    &mut self.im2col_ws,
                    res,
                    self.buf_b.ptr as *const c_void,
                    self.buf_c.ptr,
                    self.buf_d.ptr,
                    self.buf_b.ptr,
                    cur_h,
                    cur_w,
                )?;
            }

            if let Some(ds_conv) = &block.downsampler_conv {
                let (next_h, next_w) = Self::run_conv2d(
                    &ex,
                    &self.cublas,
                    &mut self.im2col_ws,
                    ds_conv,
                    self.buf_b.ptr as *const c_void,
                    core::ptr::null(),
                    self.buf_c.ptr,
                    cur_h,
                    cur_w,
                    false,
                )?;
                // avg_shortcut(buf_a) + buf_c -> buf_b
                check_rc(
                    // SAFETY: `buf_a`, `buf_c`, and `buf_b` are valid device allocations sized for `max_elems`.
                    unsafe {
                        ffi::warp34_qwen_image_vae_avg_down_f16(
                            &ex,
                            self.buf_a.ptr as *const c_void,
                            self.buf_c.ptr as *const c_void,
                            self.buf_b.ptr,
                            block.in_dim as i32,
                            block.out_dim as i32,
                            cur_h as i32,
                            cur_w as i32,
                            block.factor_t as i32,
                            block.factor_s as i32,
                        )
                    },
                    "warp34_qwen_image_vae_avg_down_f16 (down)",
                )?;
                cur_h = next_h;
                cur_w = next_w;
            } else {
                // avg_shortcut(buf_a) + buf_b -> buf_b
                check_rc(
                    // SAFETY: `buf_a` and `buf_b` are valid device allocations sized for `max_elems`.
                    unsafe {
                        ffi::warp34_qwen_image_vae_avg_down_f16(
                            &ex,
                            self.buf_a.ptr as *const c_void,
                            self.buf_b.ptr as *const c_void,
                            self.buf_b.ptr,
                            block.in_dim as i32,
                            block.out_dim as i32,
                            cur_h as i32,
                            cur_w as i32,
                            block.factor_t as i32,
                            block.factor_s as i32,
                        )
                    },
                    "warp34_qwen_image_vae_avg_down_f16 (no-down)",
                )?;
            }
        }

        // MidBlock: resnet0 -> attn0 -> resnet1 on buf_b
        Self::run_resnet(
            &ex,
            &self.cublas,
            &mut self.im2col_ws,
            &self.weights.enc_mid_block.resnet0,
            self.buf_b.ptr as *const c_void,
            self.buf_c.ptr,
            self.buf_d.ptr,
            self.buf_b.ptr,
            cur_h,
            cur_w,
        )?;
        Self::run_attention(
            &ex,
            &self.cublas,
            &mut self.im2col_ws,
            &mut self.qkv_ws,
            &mut self.scores_ws,
            &mut self.probs_ws,
            &self.weights.enc_mid_block.attn0,
            self.buf_b.ptr as *const c_void,
            self.buf_c.ptr,
            self.buf_a.ptr,
            cur_h,
            cur_w,
        )?;
        Self::run_resnet(
            &ex,
            &self.cublas,
            &mut self.im2col_ws,
            &self.weights.enc_mid_block.resnet1,
            self.buf_a.ptr as *const c_void,
            self.buf_c.ptr,
            self.buf_d.ptr,
            self.buf_b.ptr,
            cur_h,
            cur_w,
        )?;

        // norm_out + silu -> buf_c
        let last_dim = self.weights.enc_conv_out.in_c;
        let spatial = cur_h * cur_w;
        check_rc(
            // SAFETY: `buf_b`, `enc_norm_out_gamma`, and `buf_c` are valid device allocations.
            unsafe {
                ffi::warp34_qwen_image_vae_rms_norm_f16(
                    &ex,
                    self.buf_b.ptr as *const c_void,
                    self.weights.enc_norm_out_gamma.ptr as *const c_void,
                    self.buf_c.ptr,
                    last_dim as i32,
                    spatial as i32,
                    1,
                )
            },
            "enc norm_out+silu",
        )?;
        // conv_out -> buf_b
        Self::run_conv2d(
            &ex,
            &self.cublas,
            &mut self.im2col_ws,
            &self.weights.enc_conv_out,
            self.buf_c.ptr as *const c_void,
            core::ptr::null(),
            self.buf_b.ptr,
            cur_h,
            cur_w,
            false,
        )?;
        // quant_conv -> buf_a [2 * z_dim, cur_h, cur_w]
        Self::run_conv2d(
            &ex,
            &self.cublas,
            &mut self.im2col_ws,
            &self.weights.quant_conv,
            self.buf_b.ptr as *const c_void,
            core::ptr::null(),
            self.buf_a.ptr,
            cur_h,
            cur_w,
            false,
        )?;

        // Posterior mode is the first z_dim channels of quant_conv output!
        let mode_elems = cfg.z_dim * spatial;
        let mut out = vec![0u16; mode_elems];
        check_rc(
            // SAFETY: `out` holds `mode_elems * 2` bytes, and `buf_a` holds at least `2 * mode_elems * 2` bytes.
            unsafe {
                ffi::warp34_memcpy_d2h_async(
                    out.as_mut_ptr() as *mut c_void,
                    self.buf_a.ptr as *const c_void,
                    mode_elems * 2,
                    self.stream.handle,
                )
            },
            "warp34_memcpy_d2h_async(encode_single_tile)",
        )?;
        self.stream.synchronize()?;
        Ok(out)
    }

    fn decode_single_tile(
        &mut self,
        z_tile_f16: &[u16],
        latent_h: usize,
        latent_w: usize,
        clamp_final: bool,
    ) -> Result<Vec<u16>> {
        let _dev = CudaDevice::set_current(self.weights.device_id)?;
        let cfg = &self.weights.cfg;
        let scale = cfg.scale_factor_spatial();
        let out_h = latent_h * scale;
        let out_w = latent_w * scale;
        let mut max_elems = cfg.z_dim * latent_h * latent_w;
        let mut stage_h = latent_h;
        let mut stage_w = latent_w;
        for block in &self.weights.dec_up_blocks {
            max_elems = max_elems
                .max(block.in_dim * stage_h * stage_w)
                .max(block.out_dim * stage_h * stage_w);
            if block.upsampler_conv.is_some() {
                stage_h *= 2;
                stage_w *= 2;
                max_elems = max_elems.max(block.out_dim * stage_h * stage_w);
            }
        }
        max_elems = max_elems.max(cfg.decoder_base_dim * out_h * out_w);
        Self::ensure_buf_u16(&mut self.buf_a, max_elems)?;
        Self::ensure_buf_u16(&mut self.buf_b, max_elems)?;
        Self::ensure_buf_u16(&mut self.buf_c, max_elems)?;
        Self::ensure_buf_u16(&mut self.buf_d, max_elems)?;
        Self::ensure_buf_u16(&mut self.buf_e, max_elems)?;

        check_rc(
            // SAFETY: `buf_a` holds at least `z_tile_f16.len() * 2` bytes, and `z_tile_f16` is live.
            unsafe {
                ffi::warp34_memcpy_h2d_async(
                    self.buf_a.ptr,
                    z_tile_f16.as_ptr() as *const c_void,
                    z_tile_f16.len() * 2,
                    self.stream.handle,
                )
            },
            "warp34_memcpy_h2d_async(decode_single_tile)",
        )?;
        let ex = self.exec_ctx();

        // post_quant_conv: buf_a -> buf_c
        Self::run_conv2d(
            &ex,
            &self.cublas,
            &mut self.im2col_ws,
            &self.weights.post_quant_conv,
            self.buf_a.ptr as *const c_void,
            core::ptr::null(),
            self.buf_c.ptr,
            latent_h,
            latent_w,
            false,
        )?;
        // dec_conv_in: buf_c -> buf_b
        let (mut cur_h, mut cur_w) = Self::run_conv2d(
            &ex,
            &self.cublas,
            &mut self.im2col_ws,
            &self.weights.dec_conv_in,
            self.buf_c.ptr as *const c_void,
            core::ptr::null(),
            self.buf_b.ptr,
            latent_h,
            latent_w,
            false,
        )?;

        // MidBlock: resnet0 (buf_b -> buf_e) -> attn0 (buf_e -> buf_a) -> resnet1 (buf_a -> buf_b) — zero D2D copies!
        Self::run_resnet(
            &ex,
            &self.cublas,
            &mut self.im2col_ws,
            &self.weights.dec_mid_block.resnet0,
            self.buf_b.ptr as *const c_void,
            self.buf_c.ptr,
            self.buf_d.ptr,
            self.buf_e.ptr,
            cur_h,
            cur_w,
        )?;
        Self::run_attention(
            &ex,
            &self.cublas,
            &mut self.im2col_ws,
            &mut self.qkv_ws,
            &mut self.scores_ws,
            &mut self.probs_ws,
            &self.weights.dec_mid_block.attn0,
            self.buf_e.ptr as *const c_void,
            self.buf_c.ptr,
            self.buf_a.ptr,
            cur_h,
            cur_w,
        )?;
        Self::run_resnet(
            &ex,
            &self.cublas,
            &mut self.im2col_ws,
            &self.weights.dec_mid_block.resnet1,
            self.buf_a.ptr as *const c_void,
            self.buf_c.ptr,
            self.buf_d.ptr,
            self.buf_b.ptr,
            cur_h,
            cur_w,
        )?;

        // Ping-pong across `buf_a`, `buf_b`, `buf_e` so neither `block_skip` nor `run_resnet` ever needs a D2D copy!
        let mut cur_ptr = self.buf_b.ptr;
        for block in &self.weights.dec_up_blocks {
            let skip_ptr = cur_ptr;
            // Choose two ping-pong buffers disjoint from `skip_ptr` (when `up_flag` needs `skip_ptr` intact until `dup_up`).
            let (mut pp0, mut pp1) = if core::ptr::eq(skip_ptr, self.buf_a.ptr) {
                (self.buf_b.ptr, self.buf_e.ptr)
            } else if core::ptr::eq(skip_ptr, self.buf_b.ptr) {
                (self.buf_e.ptr, self.buf_a.ptr)
            } else {
                (self.buf_a.ptr, self.buf_b.ptr)
            };
            let mut r_in = cur_ptr;
            for (r_idx, res) in block.resnets.iter().enumerate() {
                let r_out = if r_idx == 0 { pp0 } else { pp1 };
                Self::run_resnet(
                    &ex,
                    &self.cublas,
                    &mut self.im2col_ws,
                    res,
                    r_in as *const c_void,
                    self.buf_c.ptr,
                    self.buf_d.ptr,
                    r_out,
                    cur_h,
                    cur_w,
                )?;
                r_in = r_out;
                if r_idx >= 1 {
                    core::mem::swap(&mut pp0, &mut pp1);
                }
            }
            cur_ptr = r_in;
            if let Some(up_conv) = &block.upsampler_conv {
                // 2x nearest upsample: cur_ptr -> buf_c
                check_rc(
                    // SAFETY: `cur_ptr` and `buf_c` are valid device allocations sized for `max_elems`.
                    unsafe {
                        ffi::warp34_qwen_image_vae_upsample2d_nearest2x_f16(
                            &ex,
                            cur_ptr as *const c_void,
                            self.buf_c.ptr,
                            block.out_dim as i32,
                            cur_h as i32,
                            cur_w as i32,
                        )
                    },
                    "warp34_qwen_image_vae_upsample2d_nearest2x_f16",
                )?;
                let (next_h, next_w) = Self::run_conv2d(
                    &ex,
                    &self.cublas,
                    &mut self.im2col_ws,
                    up_conv,
                    self.buf_c.ptr as *const c_void,
                    core::ptr::null(),
                    self.buf_d.ptr,
                    cur_h * 2,
                    cur_w * 2,
                    false,
                )?;
                // DupUp3D(skip_ptr) + buf_d -> cur_ptr (disjoint from `skip_ptr` since `resnets` has >= 1 block!)
                let dup_out = if core::ptr::eq(cur_ptr, skip_ptr) {
                    pp1
                } else {
                    cur_ptr
                };
                check_rc(
                    // SAFETY: `skip_ptr`, `buf_d`, and `dup_out` are disjoint valid device allocations sized for `max_elems`.
                    unsafe {
                        ffi::warp34_qwen_image_vae_dup_up_scaled_f16(
                            &ex,
                            skip_ptr as *const c_void,
                            self.buf_d.ptr as *const c_void,
                            dup_out,
                            block.in_dim as i32,
                            block.out_dim as i32,
                            cur_h as i32,
                            cur_w as i32,
                            block.factor_t as i32,
                            block.factor_s as i32,
                            block.skip_scale,
                        )
                    },
                    "warp34_qwen_image_vae_dup_up_scaled_f16",
                )?;
                cur_ptr = dup_out;
                cur_h = next_h;
                cur_w = next_w;
            }
        }

        // norm_out + silu -> buf_c
        let last_dim = self.weights.dec_conv_out.in_c;
        let spatial = cur_h * cur_w;
        check_rc(
            // SAFETY: `cur_ptr`, `dec_norm_out_gamma`, and `buf_c` are valid device allocations.
            unsafe {
                ffi::warp34_qwen_image_vae_rms_norm_f16(
                    &ex,
                    cur_ptr as *const c_void,
                    self.weights.dec_norm_out_gamma.ptr as *const c_void,
                    self.buf_c.ptr,
                    last_dim as i32,
                    spatial as i32,
                    1,
                )
            },
            "dec norm_out+silu",
        )?;
        // conv_out + optional clamp([-1, 1]) -> buf_a
        Self::run_conv2d(
            &ex,
            &self.cublas,
            &mut self.im2col_ws,
            &self.weights.dec_conv_out,
            self.buf_c.ptr as *const c_void,
            core::ptr::null(),
            self.buf_a.ptr,
            cur_h,
            cur_w,
            clamp_final,
        )?;

        let out_elems = cfg.out_channels * spatial;
        let mut out = vec![0u16; out_elems];
        check_rc(
            // SAFETY: `out` holds `out_elems * 2` bytes, and `buf_a` holds at least `out_elems * 2` bytes.
            unsafe {
                ffi::warp34_memcpy_d2h_async(
                    out.as_mut_ptr() as *mut c_void,
                    self.buf_a.ptr as *const c_void,
                    out_elems * 2,
                    self.stream.handle,
                )
            },
            "warp34_memcpy_d2h_async(decode_single_tile)",
        )?;
        self.stream.synchronize()?;
        Ok(out)
    }

    /// Encodes a `[in_channels, height, width]` `FP16` image into its posterior mode
    /// `[z_dim, height / scale_factor_spatial, width / scale_factor_spatial]` in `FP16`.
    pub fn encode_mode(
        &mut self,
        img_f16: &[u16],
        height: usize,
        width: usize,
    ) -> Result<Vec<u16>> {
        let cfg = &self.weights.cfg;
        let scale = cfg.scale_factor_spatial();
        if height == 0
            || width == 0
            || !height.is_multiple_of(scale)
            || !width.is_multiple_of(scale)
        {
            return Err(CudaError::new(
                -10,
                "height and width must be positive multiples of scale_factor_spatial",
            ));
        }
        let expected_len = cfg
            .in_channels
            .checked_mul(height)
            .and_then(|v| v.checked_mul(width))
            .ok_or_else(|| CudaError::new(-10, "encode_mode dimension overflow"))?;
        if img_f16.len() != expected_len {
            return Err(CudaError::new(
                -10,
                "img_f16 length mismatch in encode_mode",
            ));
        }
        if self.tiling.enabled
            && (width > self.tiling.tile_sample_min_width
                || height > self.tiling.tile_sample_min_height)
        {
            return self.tiled_encode_mode(img_f16, height, width);
        }
        self.encode_single_tile(img_f16, height, width)
    }

    /// Encodes an image and normalizes the posterior mode by `(z - latents_mean) / latents_std`
    /// (`QwenImage21Pipeline._encode_vae_image`).
    pub fn encode_normalized(
        &mut self,
        img_f16: &[u16],
        height: usize,
        width: usize,
    ) -> Result<Vec<u16>> {
        let z_mode = self.encode_mode(img_f16, height, width)?;
        let scale = self.weights.cfg.scale_factor_spatial();
        let latent_spatial = (height / scale) * (width / scale);
        Ok(normalize_latents_f16(
            &z_mode,
            self.weights.cfg.z_dim,
            latent_spatial,
            &self.weights.cfg.latents_mean,
            &self.weights.cfg.latents_std,
        ))
    }

    /// Decodes a raw `[z_dim, latent_h, latent_w]` `FP16` latent tensor into an
    /// `[out_channels, latent_h * scale, latent_w * scale]` `FP16` image clamped to `[-1.0, 1.0]`.
    pub fn decode(&mut self, z_f16: &[u16], latent_h: usize, latent_w: usize) -> Result<Vec<u16>> {
        let cfg = &self.weights.cfg;
        if latent_h == 0 || latent_w == 0 {
            return Err(CudaError::new(-10, "latent_h and latent_w must be > 0"));
        }
        let expected_len = cfg
            .z_dim
            .checked_mul(latent_h)
            .and_then(|v| v.checked_mul(latent_w))
            .ok_or_else(|| CudaError::new(-10, "decode dimension overflow"))?;
        if z_f16.len() != expected_len {
            return Err(CudaError::new(-10, "z_f16 length mismatch in decode"));
        }
        let scale = cfg.scale_factor_spatial();
        let tile_latent_min_h = self.tiling.tile_sample_min_height / scale;
        let tile_latent_min_w = self.tiling.tile_sample_min_width / scale;
        if self.tiling.enabled && (latent_w > tile_latent_min_w || latent_h > tile_latent_min_h) {
            return self.tiled_decode(z_f16, latent_h, latent_w);
        }
        self.decode_single_tile(z_f16, latent_h, latent_w, true)
    }

    /// Denormalizes `[z_dim, latent_h, latent_w]` latents by `z * latents_std + latents_mean`
    /// and decodes them into `[out_channels, latent_h * scale, latent_w * scale]` (`QwenImage21Pipeline.__call__`).
    pub fn decode_denormalized(
        &mut self,
        z_norm_f16: &[u16],
        latent_h: usize,
        latent_w: usize,
    ) -> Result<Vec<u16>> {
        if latent_h == 0 || latent_w == 0 {
            return Err(CudaError::new(-10, "latent_h and latent_w must be > 0"));
        }
        let spatial = latent_h
            .checked_mul(latent_w)
            .ok_or_else(|| CudaError::new(-10, "decode_denormalized spatial overflow"))?;
        let expected_len = self
            .weights
            .cfg
            .z_dim
            .checked_mul(spatial)
            .ok_or_else(|| CudaError::new(-10, "decode_denormalized length overflow"))?;
        if z_norm_f16.len() != expected_len {
            return Err(CudaError::new(
                -10,
                "z_norm_f16 length mismatch in decode_denormalized",
            ));
        }
        let z_raw = denormalize_latents_f16(
            z_norm_f16,
            self.weights.cfg.z_dim,
            spatial,
            &self.weights.cfg.latents_mean,
            &self.weights.cfg.latents_std,
        );
        self.decode(&z_raw, latent_h, latent_w)
    }

    fn tiled_encode_mode(
        &mut self,
        img_f16: &[u16],
        height: usize,
        width: usize,
    ) -> Result<Vec<u16>> {
        let scale = self.weights.cfg.scale_factor_spatial();
        let in_c = self.weights.cfg.in_channels;
        let z_dim = self.weights.cfg.z_dim;
        let latent_h = height / scale;
        let latent_w = width / scale;

        let tile_min_h = self.tiling.tile_sample_min_height;
        let tile_min_w = self.tiling.tile_sample_min_width;
        let tile_stride_h = self.tiling.tile_sample_stride_height;
        let tile_stride_w = self.tiling.tile_sample_stride_width;
        if tile_min_h < scale
            || tile_min_w < scale
            || tile_stride_h < scale
            || tile_stride_w < scale
            || !tile_min_h.is_multiple_of(scale)
            || !tile_min_w.is_multiple_of(scale)
            || !tile_stride_h.is_multiple_of(scale)
            || !tile_stride_w.is_multiple_of(scale)
        {
            return Err(CudaError::new(
                -10,
                "tiling min and stride dimensions must be positive multiples of scale_factor_spatial",
            ));
        }

        let tile_lat_min_h = tile_min_h / scale;
        let tile_lat_min_w = tile_min_w / scale;
        let tile_lat_stride_h = tile_stride_h / scale;
        let tile_lat_stride_w = tile_stride_w / scale;
        let blend_h = tile_lat_min_h.saturating_sub(tile_lat_stride_h);
        let blend_w = tile_lat_min_w.saturating_sub(tile_lat_stride_w);

        let mut rows: Vec<Vec<(Vec<u16>, usize, usize)>> = Vec::new();
        let mut i = 0usize;
        while i < height {
            let cur_h = (height - i).min(tile_min_h);
            let mut row = Vec::new();
            let mut j = 0usize;
            while j < width {
                let cur_w = (width - j).min(tile_min_w);
                let sub = extract_chw_subtile(img_f16, in_c, height, width, i, j, cur_h, cur_w);
                let z_sub = self.encode_single_tile(&sub, cur_h, cur_w)?;
                row.push((z_sub, cur_h / scale, cur_w / scale));
                j += tile_stride_w;
            }
            rows.push(row);
            i += tile_stride_h;
        }

        Ok(stitch_blended_tiles(
            &mut rows,
            z_dim,
            latent_h,
            latent_w,
            tile_lat_stride_h,
            tile_lat_stride_w,
            blend_h,
            blend_w,
        ))
    }

    fn tiled_decode(
        &mut self,
        z_f16: &[u16],
        latent_h: usize,
        latent_w: usize,
    ) -> Result<Vec<u16>> {
        let scale = self.weights.cfg.scale_factor_spatial();
        let z_dim = self.weights.cfg.z_dim;
        let out_c = self.weights.cfg.out_channels;
        let sample_h = latent_h * scale;
        let sample_w = latent_w * scale;

        let tile_lat_min_h = self.tiling.tile_sample_min_height / scale;
        let tile_lat_min_w = self.tiling.tile_sample_min_width / scale;
        let tile_lat_stride_h = self.tiling.tile_sample_stride_height / scale;
        let tile_lat_stride_w = self.tiling.tile_sample_stride_width / scale;
        if tile_lat_min_h == 0
            || tile_lat_min_w == 0
            || tile_lat_stride_h == 0
            || tile_lat_stride_w == 0
        {
            return Err(CudaError::new(
                -10,
                "tiling min and stride dimensions must be >= scale_factor_spatial",
            ));
        }
        let tile_sample_stride_h = self.tiling.tile_sample_stride_height;
        let tile_sample_stride_w = self.tiling.tile_sample_stride_width;
        let blend_h = self
            .tiling
            .tile_sample_min_height
            .saturating_sub(tile_sample_stride_h);
        let blend_w = self
            .tiling
            .tile_sample_min_width
            .saturating_sub(tile_sample_stride_w);

        let mut rows: Vec<Vec<(Vec<u16>, usize, usize)>> = Vec::new();
        let mut i = 0usize;
        while i < latent_h {
            let cur_lh = (latent_h - i).min(tile_lat_min_h);
            let mut row = Vec::new();
            let mut j = 0usize;
            while j < latent_w {
                let cur_lw = (latent_w - j).min(tile_lat_min_w);
                let sub =
                    extract_chw_subtile(z_f16, z_dim, latent_h, latent_w, i, j, cur_lh, cur_lw);
                let dec_sub = self.decode_single_tile(&sub, cur_lh, cur_lw, false)?;
                row.push((dec_sub, cur_lh * scale, cur_lw * scale));
                j += tile_lat_stride_w;
            }
            rows.push(row);
            i += tile_lat_stride_h;
        }

        let mut stitched = stitch_blended_tiles(
            &mut rows,
            out_c,
            sample_h,
            sample_w,
            tile_sample_stride_h,
            tile_sample_stride_w,
            blend_h,
            blend_w,
        );
        for v in &mut stitched {
            *v = f32_to_f16(f16_to_f32(*v).clamp(-1.0, 1.0));
        }
        Ok(stitched)
    }
}

#[allow(clippy::too_many_arguments)]
fn extract_chw_subtile(
    src: &[u16],
    channels: usize,
    full_h: usize,
    full_w: usize,
    y0: usize,
    x0: usize,
    sub_h: usize,
    sub_w: usize,
) -> Vec<u16> {
    let mut out = vec![0u16; channels * sub_h * sub_w];
    for c in 0..channels {
        for r in 0..sub_h {
            let src_off = (c * full_h + (y0 + r)) * full_w + x0;
            let dst_off = (c * sub_h + r) * sub_w;
            out[dst_off..dst_off + sub_w].copy_from_slice(&src[src_off..src_off + sub_w]);
        }
    }
    out
}

#[allow(clippy::too_many_arguments)]
fn blend_v_f16(
    a: &[u16],
    a_h: usize,
    a_w: usize,
    b: &mut [u16],
    b_h: usize,
    b_w: usize,
    channels: usize,
    blend_extent: usize,
) {
    let ext = blend_extent.min(a_h).min(b_h);
    let w_common = a_w.min(b_w);
    if ext == 0 || w_common == 0 {
        return;
    }
    for y in 0..ext {
        let wt_b = (y as f32) / (ext as f32);
        let wt_a = 1.0f32 - wt_b;
        let a_row = a_h - ext + y;
        for c in 0..channels {
            let a_base = (c * a_h + a_row) * a_w;
            let b_base = (c * b_h + y) * b_w;
            for x in 0..w_common {
                let va = f16_to_f32(f32_to_f16(f16_to_f32(a[a_base + x]) * wt_a));
                let vb = f16_to_f32(f32_to_f16(f16_to_f32(b[b_base + x]) * wt_b));
                b[b_base + x] = f32_to_f16(va + vb);
            }
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn blend_h_f16(
    a: &[u16],
    a_h: usize,
    a_w: usize,
    b: &mut [u16],
    b_h: usize,
    b_w: usize,
    channels: usize,
    blend_extent: usize,
) {
    let ext = blend_extent.min(a_w).min(b_w);
    let h_common = a_h.min(b_h);
    if ext == 0 || h_common == 0 {
        return;
    }
    for x in 0..ext {
        let wt_b = (x as f32) / (ext as f32);
        let wt_a = 1.0f32 - wt_b;
        let a_col = a_w - ext + x;
        for c in 0..channels {
            for y in 0..h_common {
                let va = f16_to_f32(f32_to_f16(
                    f16_to_f32(a[(c * a_h + y) * a_w + a_col]) * wt_a,
                ));
                let vb = f16_to_f32(f32_to_f16(f16_to_f32(b[(c * b_h + y) * b_w + x]) * wt_b));
                b[(c * b_h + y) * b_w + x] = f32_to_f16(va + vb);
            }
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn stitch_blended_tiles(
    rows: &mut [Vec<(Vec<u16>, usize, usize)>],
    channels: usize,
    target_h: usize,
    target_w: usize,
    stride_h: usize,
    stride_w: usize,
    blend_h: usize,
    blend_w: usize,
) -> Vec<u16> {
    let mut out = vec![0u16; channels * target_h * target_w];
    let mut y_cursor = 0usize;
    for i in 0..rows.len() {
        let mut x_cursor = 0usize;
        let mut row_h_advance = 0usize;
        let row_len = rows[i].len();
        for j in 0..row_len {
            let (mut tile, th, tw) = rows[i][j].clone();
            if i > 0 {
                let (above, ah, aw) = &rows[i - 1][j];
                blend_v_f16(above, *ah, *aw, &mut tile, th, tw, channels, blend_h);
            }
            if j > 0 {
                let (left, lh, lw) = &rows[i][j - 1];
                blend_h_f16(left, *lh, *lw, &mut tile, th, tw, channels, blend_w);
            }
            let crop_h = th.min(stride_h).min(target_h.saturating_sub(y_cursor));
            let crop_w = tw.min(stride_w).min(target_w.saturating_sub(x_cursor));
            for c in 0..channels {
                for r in 0..crop_h {
                    let src_off = (c * th + r) * tw;
                    let dst_off = (c * target_h + (y_cursor + r)) * target_w + x_cursor;
                    out[dst_off..dst_off + crop_w]
                        .copy_from_slice(&tile[src_off..src_off + crop_w]);
                }
            }
            rows[i][j].0 = tile;
            row_h_advance = crop_h;
            x_cursor += crop_w;
        }
        y_cursor += row_h_advance;
    }
    out
}
