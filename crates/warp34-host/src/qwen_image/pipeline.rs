//! End-to-end `QwenImage21Pipeline` (`Qwen/Qwen-Image-2.1` and `Qwen/Qwen-Image-2.1-Turbo`)
//! execution engine for text-to-image, single/multi-image editing, and RGBA generation.

#[cfg(feature = "cuda")]
use std::collections::HashMap;
use std::io::Cursor;
use std::path::Path;
#[cfg(feature = "cuda")]
use std::sync::Arc;
#[cfg(feature = "cuda")]
use std::time::Instant;

use image::codecs::png::{CompressionType, FilterType as PngFilterType, PngEncoder};
use image::{ColorType, ExtendedColorType, ImageEncoder};
#[cfg(feature = "cuda")]
use tokenizers::Tokenizer;

use crate::cuda::f32_to_f16;
#[cfg(feature = "cuda")]
use crate::cuda::{CublasHandle, CudaDevice, CudaError, CudaStream, Result};
#[cfg(feature = "cuda")]
use crate::loader::SafetensorsArchive;
#[cfg(feature = "cuda")]
use crate::qwen_image::text_encoder::{Qwen3VlConditioningEncoder, Qwen3VlConditioningOutput};
#[cfg(feature = "cuda")]
use crate::qwen_image::transformer::{
    QwenImage21KvCacheGpu, QwenImage21TransformerEngine, QwenImage21TransformerTp2Engine,
    pack_latents_chw_to_hwc, unpack_latents_hwc_to_chw,
};
#[cfg(feature = "cuda")]
use crate::qwen_image::vae::{
    QwenImage21VaeEngine, QwenImage21VaeTilingConfig, QwenImage21VaeWeightsGpu,
    rgba_u8_to_vae_chw_f16, vae_chw_f16_to_rgba_u8,
};
#[cfg(feature = "cuda")]
use crate::qwen_image::{
    IMAGE_PAD_TOKEN_ID, QwenImage21PipelineConfig, SYS_PROMPT, calculate_dimensions,
    composite_rgba_over_white, format_t2i_prompt, format_ti2i_prompt,
};
#[cfg(feature = "cuda")]
use crate::vision::{PreprocessedImage, preprocess_rgb_image};

/// Generates `n` standard normal `FP16` values bit-identically matching PyTorch CPU
/// `torch.randn(shape, generator=torch.Generator("cpu").manual_seed(seed), dtype=torch.float16)`.
///
/// Uses standard 32-bit `MT19937` seeded with `seed as u32` and PyTorch's `normal_fill_16`
/// vectorized Box-Muller transform in `FP32` before rounding to IEEE-754 binary16.
pub fn torch_randn_f16(n: usize, seed: u64) -> Vec<u16> {
    struct Mt19937 {
        mt: [u32; 624],
        idx: usize,
    }

    impl Mt19937 {
        fn new(seed: u64) -> Self {
            let mut mt = [0u32; 624];
            mt[0] = (seed & 0xffff_ffff) as u32;
            for i in 1..624 {
                let prev = mt[i - 1];
                mt[i] = 1_812_433_253u32
                    .wrapping_mul(prev ^ (prev >> 30))
                    .wrapping_add(i as u32);
            }
            Self { mt, idx: 624 }
        }

        #[inline]
        fn next_u32(&mut self) -> u32 {
            if self.idx >= 624 {
                for kk in 0..(624 - 397) {
                    let y = (self.mt[kk] & 0x8000_0000) | (self.mt[kk + 1] & 0x7fff_ffff);
                    self.mt[kk] =
                        self.mt[kk + 397] ^ (y >> 1) ^ if (y & 1) != 0 { 0x9908_b0df } else { 0 };
                }
                for kk in (624 - 397)..623 {
                    let y = (self.mt[kk] & 0x8000_0000) | (self.mt[kk + 1] & 0x7fff_ffff);
                    self.mt[kk] =
                        self.mt[kk - 227] ^ (y >> 1) ^ if (y & 1) != 0 { 0x9908_b0df } else { 0 };
                }
                let y = (self.mt[623] & 0x8000_0000) | (self.mt[0] & 0x7fff_ffff);
                self.mt[623] = self.mt[396] ^ (y >> 1) ^ if (y & 1) != 0 { 0x9908_b0df } else { 0 };
                self.idx = 0;
            }
            let mut y = self.mt[self.idx];
            self.idx += 1;
            y ^= y >> 11;
            y ^= (y << 7) & 0x9d2c_5680;
            y ^= (y << 15) & 0xefc6_0000;
            y ^= y >> 18;
            y
        }

        #[inline]
        fn next_f32_uniform(&mut self) -> f32 {
            ((self.next_u32() & 0x00ff_ffff) as f32) * (1.0f32 / 16_777_216.0f32)
        }
    }

    let mut rng = Mt19937::new(seed);
    let mut out = vec![0u16; n];
    let fill_16 = |rng: &mut Mt19937, dst: &mut [u16]| {
        let mut u = [0.0f32; 16];
        for slot in &mut u {
            *slot = rng.next_f32_uniform();
        }
        for j in 0..8 {
            let u1 = 1.0f32 - u[j];
            let u2 = u[j + 8];
            let radius = (-2.0f32 * ((u1 as f64).ln() as f32)).sqrt();
            let theta = (2.0f64 * core::f64::consts::PI * (u2 as f64)) as f32;
            dst[j] = f32_to_f16(radius * (theta as f64).cos() as f32);
            dst[j + 8] = f32_to_f16(radius * (theta as f64).sin() as f32);
        }
    };
    let mut i = 0usize;
    while i + 16 <= n {
        fill_16(&mut rng, &mut out[i..i + 16]);
        i += 16;
    }
    if i < n {
        if n >= 16 {
            // PyTorch `ATen/native/cpu/DistributionTemplates.h` (`normal_fill`) steps back to
            // `data + size - 16` and invokes `normal_fill_16(data + size - 16, mean, std, gen)`
            // whenever `size >= 16 && size % 16 != 0`. Matching this overlapping 16-wide tail
            // preserves bit-exact parity with `torch.randn(..., generator=torch.Generator("cpu"))`.
            fill_16(&mut rng, &mut out[n - 16..n]);
        } else {
            let mut cached: Option<f32> = None;
            while i < n {
                if let Some(val) = cached.take() {
                    out[i] = f32_to_f16(val);
                } else {
                    let u1 = 1.0f32 - rng.next_f32_uniform();
                    let u2 = rng.next_f32_uniform();
                    let radius = (-2.0f32 * ((u1 as f64).ln() as f32)).sqrt();
                    let theta = (2.0f64 * core::f64::consts::PI * (u2 as f64)) as f32;
                    out[i] = f32_to_f16(radius * (theta as f64).cos() as f32);
                    cached = Some(radius * (theta as f64).sin() as f32);
                }
                i += 1;
            }
        }
    }
    out
}

/// Condition image input for `QwenImage21Pipeline` (`ti2i` single or multi-image editing).
#[derive(Clone, Debug)]
pub struct QwenImage21ConditionImage {
    /// Raw RGBA8 pixels (`width * height * 4`).
    pub rgba: Vec<u8>,
    /// Image width in pixels.
    pub width: u32,
    /// Image height in pixels.
    pub height: u32,
}

impl QwenImage21ConditionImage {
    /// Creates a condition image from raw `RGBA8` pixels.
    pub fn from_rgba_u8(
        rgba: Vec<u8>,
        width: u32,
        height: u32,
    ) -> std::result::Result<Self, String> {
        let expected = (width as usize)
            .checked_mul(height as usize)
            .and_then(|px| px.checked_mul(4))
            .ok_or_else(|| "condition image dimension overflow".to_string())?;
        if width == 0 || height == 0 || rgba.len() != expected {
            return Err(format!(
                "invalid RGBA condition image dimensions {width}x{height} (len={})",
                rgba.len()
            ));
        }
        Ok(Self {
            rgba,
            width,
            height,
        })
    }

    /// Decodes a condition image from encoded image bytes (PNG, JPEG, WebP).
    pub fn from_image_bytes(bytes: &[u8]) -> std::result::Result<Self, String> {
        let dyn_img =
            image::load_from_memory(bytes).map_err(|e| format!("decode condition image: {e}"))?;
        let rgba_img = dyn_img.to_rgba8();
        let (width, height) = rgba_img.dimensions();
        Self::from_rgba_u8(rgba_img.into_raw(), width, height)
    }

    /// Loads and decodes a condition image from a file path.
    pub fn from_path(path: &Path) -> std::result::Result<Self, String> {
        let bytes = std::fs::read(path)
            .map_err(|e| format!("read condition image '{}': {e}", path.display()))?;
        Self::from_image_bytes(&bytes)
    }

    /// Resizes the condition image to `(target_width, target_height)` using Pillow-compatible
    /// premultiplied-alpha separable Lanczos-3 (`PIL.Image.resize(..., resample=LANCZOS)`),
    /// matching `VaeImageProcessor.resize(..., resample="lanczos")` without transparent-edge color bleed.
    pub fn resize_lanczos(&self, target_width: u32, target_height: u32) -> Self {
        let target_width = target_width.max(1);
        let target_height = target_height.max(1);
        if self.width == target_width && self.height == target_height {
            return self.clone();
        }
        let expected = (self.width as usize)
            .saturating_mul(self.height as usize)
            .saturating_mul(4);
        if self.rgba.len() != expected || self.width == 0 || self.height == 0 {
            return self.clone();
        }
        let rgba = resize_rgba_premultiplied_lanczos3(
            &self.rgba,
            self.width as usize,
            self.height as usize,
            target_width as usize,
            target_height as usize,
        );
        Self {
            rgba,
            width: target_width,
            height: target_height,
        }
    }
}

#[inline]
fn lanczos3_kernel(x: f64) -> f64 {
    let ax = x.abs();
    if ax < 1e-12 {
        1.0
    } else if ax >= 3.0 {
        0.0
    } else {
        let pix = core::f64::consts::PI * x;
        let pix3 = pix / 3.0;
        (pix.sin() / pix) * (pix3.sin() / pix3)
    }
}

const LANCZOS_PRECISION_BITS: i64 = 32 - 8 - 2;

fn build_lanczos3_taps(in_size: usize, out_size: usize) -> Vec<(usize, Vec<i64>)> {
    let scale = (in_size as f64) / (out_size as f64);
    let filter_scale = scale.max(1.0);
    let support = 3.0 * filter_scale;
    let inv_filter_scale = 1.0 / filter_scale;
    let scale_fp = (1i64 << LANCZOS_PRECISION_BITS) as f64;
    let mut table = Vec::with_capacity(out_size);
    for out_i in 0..out_size {
        let center = ((out_i as f64) + 0.5) * scale;
        let xmin = (center - support + 0.5).floor().max(0.0) as usize;
        let xmax = ((center + support + 0.5).floor().max(0.0) as usize).min(in_size);
        let mut weights = Vec::with_capacity(xmax.saturating_sub(xmin));
        let mut sum = 0.0f64;
        for src_i in xmin..xmax {
            let w = lanczos3_kernel(((src_i as f64) - center + 0.5) * inv_filter_scale);
            weights.push(w);
            sum += w;
        }
        let inv_sum = if sum.abs() > 1e-12 { 1.0 / sum } else { 0.0 };
        let weights_fp: Vec<i64> = weights
            .into_iter()
            .map(|w| (w * inv_sum * scale_fp).round() as i64)
            .collect();
        table.push((xmin, weights_fp));
    }
    table
}

fn resize_rgba_premultiplied_lanczos3(
    src: &[u8],
    in_w: usize,
    in_h: usize,
    out_w: usize,
    out_h: usize,
) -> Vec<u8> {
    let has_alpha = src.as_chunks::<4>().0.iter().any(|px| px[3] != 255);
    let half_bias = 1i64 << (LANCZOS_PRECISION_BITS - 1);

    // 1. RGBA -> RGBa premultiplication (`(c * a + 127) / 255` matching Pillow `rgb2rgb_a`).
    let pre: std::borrow::Cow<'_, [u8]> = if has_alpha {
        let mut buf = vec![0u8; src.len()];
        for (dst_px, src_px) in buf
            .as_chunks_mut::<4>()
            .0
            .iter_mut()
            .zip(src.as_chunks::<4>().0.iter())
        {
            let a = src_px[3] as u32;
            dst_px[0] = ((src_px[0] as u32 * a + 127) / 255) as u8;
            dst_px[1] = ((src_px[1] as u32 * a + 127) / 255) as u8;
            dst_px[2] = ((src_px[2] as u32 * a + 127) / 255) as u8;
            dst_px[3] = src_px[3];
        }
        std::borrow::Cow::Owned(buf)
    } else {
        std::borrow::Cow::Borrowed(src)
    };

    // 2. Horizontal pass (`in_w -> out_w`) in 22-bit fixed-point into `[in_h, out_w, 4]` u8.
    let horiz: Vec<u8> = if out_w != in_w {
        let x_taps = build_lanczos3_taps(in_w, out_w);
        let mut buf = vec![0u8; in_h * out_w * 4];
        for y in 0..in_h {
            let src_row = &pre[y * in_w * 4..(y + 1) * in_w * 4];
            let dst_row = &mut buf[y * out_w * 4..(y + 1) * out_w * 4];
            for (ox, (start, taps)) in x_taps.iter().enumerate() {
                let mut acc = [half_bias; 4];
                for (k, &w) in taps.iter().enumerate() {
                    let px = &src_row[(start + k) * 4..(start + k) * 4 + 4];
                    acc[0] += (px[0] as i64) * w;
                    acc[1] += (px[1] as i64) * w;
                    acc[2] += (px[2] as i64) * w;
                    acc[3] += (px[3] as i64) * w;
                }
                dst_row[ox * 4] = (acc[0] >> LANCZOS_PRECISION_BITS).clamp(0, 255) as u8;
                dst_row[ox * 4 + 1] = (acc[1] >> LANCZOS_PRECISION_BITS).clamp(0, 255) as u8;
                dst_row[ox * 4 + 2] = (acc[2] >> LANCZOS_PRECISION_BITS).clamp(0, 255) as u8;
                dst_row[ox * 4 + 3] = if has_alpha {
                    (acc[3] >> LANCZOS_PRECISION_BITS).clamp(0, 255) as u8
                } else {
                    255
                };
            }
        }
        buf
    } else {
        pre.into_owned()
    };

    // 3. Vertical pass (`in_h -> out_h`) in 22-bit fixed-point into `[out_h, out_w, 4]` u8.
    let mut out = if out_h != in_h {
        let y_taps = build_lanczos3_taps(in_h, out_h);
        let mut buf = vec![0u8; out_h * out_w * 4];
        for (oy, (start, taps)) in y_taps.iter().enumerate() {
            let dst_row = &mut buf[oy * out_w * 4..(oy + 1) * out_w * 4];
            for ox in 0..out_w {
                let mut acc = [half_bias; 4];
                for (k, &w) in taps.iter().enumerate() {
                    let p =
                        &horiz[((start + k) * out_w + ox) * 4..((start + k) * out_w + ox) * 4 + 4];
                    acc[0] += (p[0] as i64) * w;
                    acc[1] += (p[1] as i64) * w;
                    acc[2] += (p[2] as i64) * w;
                    acc[3] += (p[3] as i64) * w;
                }
                dst_row[ox * 4] = (acc[0] >> LANCZOS_PRECISION_BITS).clamp(0, 255) as u8;
                dst_row[ox * 4 + 1] = (acc[1] >> LANCZOS_PRECISION_BITS).clamp(0, 255) as u8;
                dst_row[ox * 4 + 2] = (acc[2] >> LANCZOS_PRECISION_BITS).clamp(0, 255) as u8;
                dst_row[ox * 4 + 3] = if has_alpha {
                    (acc[3] >> LANCZOS_PRECISION_BITS).clamp(0, 255) as u8
                } else {
                    255
                };
            }
        }
        buf
    } else {
        horiz
    };

    // 4. RGBa -> RGBA unpremultiplication (`(c * 255) / a` matching Pillow `rgb_a2rgba`).
    if has_alpha {
        for px in out.as_chunks_mut::<4>().0 {
            let a = px[3] as u32;
            px[0] = (px[0] as u32 * 255).checked_div(a).unwrap_or(0).min(255) as u8;
            px[1] = (px[1] as u32 * 255).checked_div(a).unwrap_or(0).min(255) as u8;
            px[2] = (px[2] as u32 * 255).checked_div(a).unwrap_or(0).min(255) as u8;
        }
    }
    out
}

/// Request parameters for `QwenImage21PipelineEngine::generate`.
#[derive(Clone, Debug)]
pub struct QwenImage21GenerationRequest {
    /// Positive text prompt.
    pub prompt: String,
    /// Optional negative text prompt (used when `true_cfg_scale > 1.0`).
    pub negative_prompt: Option<String>,
    /// Condition images (`empty` for text-to-image, `1+` for single/multi-image editing).
    pub condition_images: Vec<QwenImage21ConditionImage>,
    /// Optional explicit output height in pixels (rounded down to a multiple of 32).
    pub height: Option<u32>,
    /// Optional explicit output width in pixels (rounded down to a multiple of 32).
    pub width: Option<u32>,
    /// Target side length used to derive `height`/`width` and condition image sizes (`default = 1024`).
    pub output_resolution: u32,
    /// Number of Euler flow-matching steps (`None` uses model default: `8` for Turbo, `40` for standard).
    pub num_inference_steps: Option<usize>,
    /// Optional custom sigma schedule (excluding terminal `0.0`).
    pub sigmas: Option<Vec<f32>>,
    /// Classifier-free guidance scale (`1.0` disables CFG).
    pub true_cfg_scale: f32,
    /// Random seed for initial Gaussian latent noise (`torch.Generator("cpu").manual_seed(seed)`).
    pub seed: u64,
    /// Optional caller-supplied initial `[64, latent_h, latent_w]` `FP16` latents.
    pub initial_latents_chw_f16: Option<Vec<u16>>,
    /// Preserve the 4th alpha channel in the output PNG (`true` = RGBA, `false` = RGB).
    pub rgba_output: bool,
    /// Enable CUDA Graph capture & replay on Steps `1..N-1` (`true` by default).
    pub use_cuda_graph: bool,
}

impl Default for QwenImage21GenerationRequest {
    fn default() -> Self {
        Self {
            prompt: String::new(),
            negative_prompt: None,
            condition_images: Vec::new(),
            height: None,
            width: None,
            output_resolution: 1024,
            num_inference_steps: None,
            sigmas: None,
            true_cfg_scale: 1.0,
            seed: 42,
            initial_latents_chw_f16: None,
            rgba_output: false,
            use_cuda_graph: true,
        }
    }
}

/// Stage-by-stage latency breakdown for a `QwenImage21Pipeline` generation call.
#[derive(Clone, Debug, Default)]
pub struct QwenImage21GenerationTimings {
    /// Time spent in `Qwen3-VL-8B` conditioning encoder (`ms`).
    pub condition_encode_ms: f64,
    /// Time spent encoding condition images in `AutoencoderKLQwenImage21` (`ms`).
    pub vae_encode_ms: f64,
    /// Time spent in Step 0 (`extract` block-causal prefill + prefix KV write, `ms`).
    pub denoise_step0_ms: f64,
    /// Time spent across Steps `1..N-1` (`cached` decode with CUDA Graph replay, `ms`).
    pub denoise_cached_ms: f64,
    /// Time spent in `AutoencoderKLQwenImage21` decode (`ms`).
    pub vae_decode_ms: f64,
    /// Total end-to-end generation wall time (`ms`).
    pub total_ms: f64,
}

/// Output of `QwenImage21PipelineEngine::generate`.
#[derive(Clone, Debug)]
pub struct QwenImage21GenerationResult {
    /// Postprocessed `u8` pixel buffer (`width * height * channels`).
    pub pixels_u8: Vec<u8>,
    /// Encoded PNG image bytes.
    pub png_bytes: Vec<u8>,
    /// Normalized `[64, latent_h, latent_w]` `FP16` latents prior to VAE decode.
    pub latents_chw_f16: Vec<u16>,
    /// Raw `[4, height, width]` `FP16` VAE output clamped to `[-1.0, 1.0]`.
    pub decoded_chw_f16: Vec<u16>,
    /// Output image width in pixels.
    pub width: u32,
    /// Output image height in pixels.
    pub height: u32,
    /// Output color channels (`3` for RGB, `4` for RGBA).
    pub channels: u8,
    /// Number of denoising steps executed.
    pub steps: usize,
    /// Seed used for latent initialization.
    pub seed: u64,
    /// Stage latency breakdown.
    pub timings: QwenImage21GenerationTimings,
}

/// Encodes an `RGB8` (`channels == 3`) or `RGBA8` (`channels == 4`) pixel buffer as a PNG.
pub fn encode_png_bytes(
    pixels: &[u8],
    width: u32,
    height: u32,
    channels: u8,
) -> std::result::Result<Vec<u8>, String> {
    let color_type = match channels {
        3 => ExtendedColorType::Rgb8,
        4 => ExtendedColorType::Rgba8,
        _ => return Err(format!("unsupported PNG channel count: {channels}")),
    };
    if width == 0 || height == 0 {
        return Err(format!("invalid PNG dimensions {width}x{height}"));
    }
    let expected = (width as usize)
        .checked_mul(height as usize)
        .and_then(|n| n.checked_mul(channels as usize))
        .ok_or_else(|| "PNG pixel buffer size overflow".to_string())?;
    if pixels.len() != expected {
        return Err(format!(
            "PNG pixel buffer length mismatch: expected {expected} bytes for {width}x{height}x{channels}, got {}",
            pixels.len()
        ));
    }
    let mut buf = Vec::with_capacity(pixels.len() / 2);
    let encoder = PngEncoder::new_with_quality(
        Cursor::new(&mut buf),
        CompressionType::Fast,
        PngFilterType::Sub,
    );
    encoder
        .write_image(pixels, width, height, color_type)
        .map_err(|e| format!("PNG encode failed: {e}"))?;
    let _ = ColorType::Rgba8;
    Ok(buf)
}

/// Computes Peak Signal-to-Noise Ratio (PSNR, in dB) between two `[0..255]` `u8` images.
///
/// Returns `f64::NAN` if `a.len() != b.len()`.
pub fn compute_psnr_u8(a: &[u8], b: &[u8]) -> f64 {
    if a.len() != b.len() {
        return f64::NAN;
    }
    if a.is_empty() {
        return f64::INFINITY;
    }
    let mse: f64 = a
        .iter()
        .zip(b.iter())
        .map(|(&x, &y)| {
            let d = (x as f64) - (y as f64);
            d * d
        })
        .sum::<f64>()
        / (a.len() as f64);
    if mse <= 1e-12 {
        99.99
    } else {
        10.0 * ((255.0 * 255.0) / mse).log10()
    }
}

#[cfg(feature = "cuda")]
#[allow(clippy::large_enum_variant)]
enum TransformerBackend {
    Single {
        engine: QwenImage21TransformerEngine,
        stream: CudaStream,
        cublas: CublasHandle,
        cond_cache: QwenImage21KvCacheGpu,
        neg_cache: QwenImage21KvCacheGpu,
    },
    Tp2 {
        engine: QwenImage21TransformerTp2Engine,
        cond_caches: (QwenImage21KvCacheGpu, QwenImage21KvCacheGpu),
        neg_caches: (QwenImage21KvCacheGpu, QwenImage21KvCacheGpu),
    },
}

#[cfg(feature = "cuda")]
#[allow(clippy::large_enum_variant)]
enum ConditioningBackend {
    Resident {
        encoder: Qwen3VlConditioningEncoder,
        stream: CudaStream,
        cublas: CublasHandle,
    },
    OnDemand {
        device_id: i32,
        archive: Arc<SafetensorsArchive>,
    },
}

/// End-to-end GPU execution engine for `Qwen/Qwen-Image-2.1` and `Qwen/Qwen-Image-2.1-Turbo`.
#[cfg(feature = "cuda")]
pub struct QwenImage21PipelineEngine {
    /// Parsed pipeline configuration (`model_index.json` + component configs).
    pub cfg: QwenImage21PipelineConfig,
    tokenizer: Tokenizer,
    drop_idx: usize,
    transformer_archive: Arc<SafetensorsArchive>,
    conditioning: ConditioningBackend,
    transformer: Option<TransformerBackend>,
    vae: QwenImage21VaeEngine,
    tp_size: usize,
    conditioning_cache: HashMap<u64, Qwen3VlConditioningOutput>,
}

#[cfg(feature = "cuda")]
impl QwenImage21PipelineEngine {
    /// Loads `QwenImage21PipelineEngine` from a resolved checkpoint directory.
    ///
    /// - When `tp_size == 2` (or `0` auto on `>= 2` GPUs): keeps `TP=2` `transformer` across
    ///   GPU 0 + GPU 1, `vae` on GPU 0, and `text_encoder` on GPU 1 (`Resident` on quantized or
    ///   `>= 32 GiB` GPUs, `OnDemand` on 24 GiB unquantized `FP16`).
    /// - When `tp_size == 1` on `>= 2` GPUs: keeps `transformer` + `vae` on GPU 0 and
    ///   `text_encoder` on GPU 1 (100% resident in VRAM).
    /// - When only 1 GPU is available: keeps all 3 resident if VRAM permits (`>= 34 GiB`),
    ///   or automatically swaps `text_encoder` and `transformer` on GPU 0 with an in-memory
    ///   conditioning cache on 24 GiB cards.
    pub fn load(model_dir: &Path, requested_tp: usize) -> Result<Self> {
        let cfg = QwenImage21PipelineConfig::from_dir(model_dir).map_err(|e| {
            CudaError::with_detail(-10, "load QwenImage21PipelineConfig", e.to_string())
        })?;
        let tok_path = cfg.tokenizer_path();
        let tokenizer = Tokenizer::from_file(&tok_path).map_err(|e| {
            CudaError::with_detail(
                -10,
                "load processor/tokenizer.json",
                format!("{}: {e}", tok_path.display()),
            )
        })?;
        let sys_prefix = format!("<|im_start|>system\n{SYS_PROMPT}<|im_end|>\n");
        let sys_enc = tokenizer
            .encode(sys_prefix, false)
            .map_err(|e| CudaError::with_detail(-10, "tokenize system prefix", e.to_string()))?;
        let drop_idx = sys_enc.get_ids().len();

        let num_gpus = CudaDevice::count()?.max(1) as usize;
        let tp_size = if requested_tp == 0 {
            if num_gpus >= 2 { 2 } else { 1 }
        } else {
            requested_tp
        };
        if tp_size != 1 && tp_size != 2 {
            return Err(CudaError::with_detail(
                -10,
                "QwenImage21PipelineEngine supports tp_size 1 or 2",
                tp_size.to_string(),
            ));
        }
        if tp_size == 2 && num_gpus < 2 {
            return Err(CudaError::new(
                -10,
                "tp_size=2 requested but fewer than 2 CUDA devices are visible",
            ));
        }

        let text_archive = Arc::new(SafetensorsArchive::open(&cfg.text_encoder_dir).map_err(
            |e| CudaError::with_detail(-10, "open text_encoder safetensors", e.to_string()),
        )?);
        let transformer_archive =
            Arc::new(SafetensorsArchive::open(&cfg.transformer_dir).map_err(|e| {
                CudaError::with_detail(-10, "open transformer safetensors", e.to_string())
            })?);
        let vae_archive = SafetensorsArchive::open(&cfg.vae_dir)
            .map_err(|e| CudaError::with_detail(-10, "open vae safetensors", e.to_string()))?;

        // Enable spatial tiling above 1536x1536 so <=1536px runs untiled single-pass while >=2048px avoids OOM.
        // Keep the 64-ch RGBA VAE on GPU 0 so GPU 1 is reserved for `rank1` + `text_encoder` on `TP=2`.
        let vae_dev = 0;
        let vae_weights =
            QwenImage21VaeWeightsGpu::from_archive(&vae_archive, &cfg.vae, "", vae_dev)?;
        let mut vae = QwenImage21VaeEngine::new(vae_weights)?;
        vae.set_tiling(QwenImage21VaeTilingConfig {
            enabled: true,
            tile_sample_min_height: 1536,
            tile_sample_min_width: 1536,
            tile_sample_stride_height: 1152,
            tile_sample_stride_width: 1152,
        });

        let tf_quant = &cfg.transformer.quant;
        let te_quant = &cfg.text_encoder.quant;
        let (transformer, conditioning) = if tp_size == 2 {
            let tf_tp2 = QwenImage21TransformerTp2Engine::load_with_prefix(
                &transformer_archive,
                &cfg.transformer,
                tf_quant,
                "",
                16384,
            )?;
            let cond_caches = tf_tp2.new_kv_caches();
            let neg_caches = tf_tp2.new_kv_caches();
            // Place the VLM conditioning encoder on GPU 1 (since GPU 0 holds the 64-ch RGBA VAE + im2col workspace).
            // On 24 GB GPUs running unquantized FP16 (20B DiT + 8B VLM = 14.4 GiB), load the VLM on-demand on GPU 1
            // and release it before `denoise_loop` so both ranks retain ~16 GiB free for 1024x1024+ TI2I KV caches.
            let gpu1_total_bytes = CudaDevice::set_current(1)
                .and_then(|d| d.mem_info())
                .map(|(_, total)| total)
                .unwrap_or(0);
            let need_ondemand_te = *tf_quant == crate::loader::QuantScheme::Unquantized
                && *te_quant == crate::loader::QuantScheme::Unquantized
                && gpu1_total_bytes < (32usize << 30);
            let cond_backend = if need_ondemand_te {
                ConditioningBackend::OnDemand {
                    device_id: 1,
                    archive: text_archive,
                }
            } else {
                let te_stream = CudaStream::new(1)?;
                let te_cublas = CublasHandle::new(&te_stream)?;
                match Qwen3VlConditioningEncoder::load(
                    Arc::clone(&text_archive),
                    &cfg.text_encoder,
                    te_quant,
                    1,
                    &te_stream,
                ) {
                    Ok(te_encoder) => ConditioningBackend::Resident {
                        encoder: te_encoder,
                        stream: te_stream,
                        cublas: te_cublas,
                    },
                    Err(_) => ConditioningBackend::OnDemand {
                        device_id: 1,
                        archive: text_archive,
                    },
                }
            };
            (
                Some(TransformerBackend::Tp2 {
                    engine: tf_tp2,
                    cond_caches,
                    neg_caches,
                }),
                cond_backend,
            )
        } else if num_gpus >= 2 {
            let tf_stream = CudaStream::new(0)?;
            let tf_cublas = CublasHandle::new(&tf_stream)?;
            let tf_single = QwenImage21TransformerEngine::load(
                &transformer_archive,
                &cfg.transformer,
                tf_quant,
                0,
                &tf_stream,
            )?;
            let cond_cache = tf_single.new_kv_cache();
            let neg_cache = tf_single.new_kv_cache();
            let te_stream = CudaStream::new(1)?;
            let te_cublas = CublasHandle::new(&te_stream)?;
            let te_encoder = Qwen3VlConditioningEncoder::load(
                Arc::clone(&text_archive),
                &cfg.text_encoder,
                te_quant,
                1,
                &te_stream,
            )?;
            (
                Some(TransformerBackend::Single {
                    engine: tf_single,
                    stream: tf_stream,
                    cublas: tf_cublas,
                    cond_cache,
                    neg_cache,
                }),
                ConditioningBackend::Resident {
                    encoder: te_encoder,
                    stream: te_stream,
                    cublas: te_cublas,
                },
            )
        } else {
            let tf_stream = CudaStream::new(0)?;
            let tf_cublas = CublasHandle::new(&tf_stream)?;
            let tf_single = QwenImage21TransformerEngine::load(
                &transformer_archive,
                &cfg.transformer,
                tf_quant,
                0,
                &tf_stream,
            )?;
            let cond_cache = tf_single.new_kv_cache();
            let neg_cache = tf_single.new_kv_cache();
            let te_stream = CudaStream::new(0)?;
            let te_cublas = CublasHandle::new(&te_stream)?;
            match Qwen3VlConditioningEncoder::load(
                Arc::clone(&text_archive),
                &cfg.text_encoder,
                te_quant,
                0,
                &te_stream,
            ) {
                Ok(te_encoder) => (
                    Some(TransformerBackend::Single {
                        engine: tf_single,
                        stream: tf_stream,
                        cublas: tf_cublas,
                        cond_cache,
                        neg_cache,
                    }),
                    ConditioningBackend::Resident {
                        encoder: te_encoder,
                        stream: te_stream,
                        cublas: te_cublas,
                    },
                ),
                Err(_) => (
                    Some(TransformerBackend::Single {
                        engine: tf_single,
                        stream: tf_stream,
                        cublas: tf_cublas,
                        cond_cache,
                        neg_cache,
                    }),
                    ConditioningBackend::OnDemand {
                        device_id: 0,
                        archive: text_archive,
                    },
                ),
            }
        };

        Ok(Self {
            cfg,
            tokenizer,
            drop_idx,
            transformer_archive,
            conditioning,
            transformer,
            vae,
            tp_size,
            conditioning_cache: HashMap::new(),
        })
    }

    /// Returns the active tensor-parallel degree (`1` or `2`).
    pub const fn tp_size(&self) -> usize {
        self.tp_size
    }

    #[cfg(all(test, feature = "cuda"))]
    pub(crate) fn force_ondemand_conditioning_for_test(&mut self) {
        let device_id = if self.tp_size == 2 { 1 } else { 0 };
        let archive = Arc::new(
            SafetensorsArchive::open(&self.cfg.text_encoder_dir)
                .expect("open text_encoder archive"),
        );
        self.conditioning = ConditioningBackend::OnDemand { device_id, archive };
        self.conditioning_cache.clear();
    }

    fn hash_conditioning_key(prompt: &str, images: &[PreprocessedImage]) -> u64 {
        let mut h: u64 = 0xcbf2_9ce4_8422_2325;
        for &b in prompt.as_bytes() {
            h ^= b as u64;
            h = h.wrapping_mul(0x0100_0000_01b3);
        }
        h ^= 0xff;
        h = h.wrapping_mul(0x0100_0000_01b3);
        for img in images {
            h ^= img.content_hash;
            h = h.wrapping_mul(0x0100_0000_01b3);
        }
        h
    }

    fn build_prompt_token_ids(
        &self,
        prompt: &str,
        images: &[PreprocessedImage],
    ) -> Result<Vec<i32>> {
        let sanitized: std::borrow::Cow<'_, str> = if prompt.contains("<|image_pad|>") {
            std::borrow::Cow::Owned(prompt.replace("<|image_pad|>", ""))
        } else {
            std::borrow::Cow::Borrowed(prompt)
        };
        let non_empty = if sanitized.is_empty() {
            " "
        } else {
            sanitized.as_ref()
        };
        let formatted = if images.is_empty() {
            format_t2i_prompt(non_empty)
        } else {
            format_ti2i_prompt(non_empty, images.len())
        };
        let enc = self.tokenizer.encode(formatted, false).map_err(|e| {
            CudaError::with_detail(-10, "tokenize QwenImage21 prompt", e.to_string())
        })?;
        let raw_ids = enc.get_ids();
        let (spatial_merge_size, image_token_id) = self
            .cfg
            .text_encoder
            .vision_config
            .as_ref()
            .map(|v| (v.spatial_merge_size, v.image_token_id))
            .unwrap_or((2, IMAGE_PAD_TOKEN_ID));
        if images.is_empty() {
            return Ok(raw_ids
                .iter()
                .map(|&id| id as i32)
                .filter(|&id| id != image_token_id)
                .collect());
        }
        let mut expanded = Vec::with_capacity(
            raw_ids.len()
                + images
                    .iter()
                    .map(|i| i.num_merged_tokens(spatial_merge_size))
                    .sum::<usize>(),
        );
        let mut img_idx = 0usize;
        for &id in raw_ids {
            if (id as i32) == image_token_id {
                let count = images
                    .get(img_idx)
                    .map(|img| img.num_merged_tokens(spatial_merge_size))
                    .ok_or_else(|| {
                        CudaError::new(-10, "more <|image_pad|> tokens than condition images")
                    })?;
                expanded.extend(std::iter::repeat_n(image_token_id, count));
                img_idx += 1;
            } else {
                expanded.push(id as i32);
            }
        }
        if img_idx != images.len() {
            return Err(CudaError::with_detail(
                -10,
                "fewer <|image_pad|> tokens than condition images",
                format!("got {img_idx}, expected {}", images.len()),
            ));
        }
        Ok(expanded)
    }

    fn encode_prompt_cached(
        &mut self,
        prompt: &str,
        images: &[PreprocessedImage],
    ) -> Result<Qwen3VlConditioningOutput> {
        let key = Self::hash_conditioning_key(prompt, images);
        if let Some(cached) = self.conditioning_cache.get(&key) {
            return Ok(cached.clone());
        }
        let input_ids = self.build_prompt_token_ids(prompt, images)?;
        let out = match &self.conditioning {
            ConditioningBackend::Resident {
                encoder,
                stream,
                cublas,
            } => encoder.encode_conditioning(&input_ids, images, self.drop_idx, cublas, stream)?,
            ConditioningBackend::OnDemand { device_id, archive } => {
                // On single-GPU (`tp_size == 1`), free the transformer before loading the 14.4 GiB text encoder.
                // On `tp_size == 2`, GPU 1 only holds `rank1` (~7.0 GiB), so the 14.4 GiB text encoder fits
                // alongside `rank1` and is dropped at the end of this block before `denoise_loop` allocates KV cache.
                if self.tp_size == 1 {
                    self.transformer = None;
                    self.vae.release_workspace()?;
                } else if let Some(TransformerBackend::Tp2 {
                    engine,
                    cond_caches,
                    neg_caches,
                }) = &mut self.transformer
                {
                    *cond_caches = engine.new_kv_caches();
                    *neg_caches = engine.new_kv_caches();
                    engine.release_workspaces();
                }
                let stream = CudaStream::new(*device_id)?;
                let cublas = CublasHandle::new(&stream)?;
                let encoder = Qwen3VlConditioningEncoder::load(
                    Arc::clone(archive),
                    &self.cfg.text_encoder,
                    &self.cfg.text_encoder.quant,
                    *device_id,
                    &stream,
                )?;
                encoder.encode_conditioning(&input_ids, images, self.drop_idx, &cublas, &stream)?
            }
        };
        if self.conditioning_cache.len() >= 16 {
            self.conditioning_cache.clear();
        }
        self.conditioning_cache.insert(key, out.clone());
        Ok(out)
    }

    fn ensure_transformer_loaded(&mut self) -> Result<()> {
        if self.transformer.is_none() {
            let tf_stream = CudaStream::new(0)?;
            let tf_cublas = CublasHandle::new(&tf_stream)?;
            let tf_single = QwenImage21TransformerEngine::load(
                &self.transformer_archive,
                &self.cfg.transformer,
                &self.cfg.transformer.quant,
                0,
                &tf_stream,
            )?;
            let cond_cache = tf_single.new_kv_cache();
            let neg_cache = tf_single.new_kv_cache();
            self.transformer = Some(TransformerBackend::Single {
                engine: tf_single,
                stream: tf_stream,
                cublas: tf_cublas,
                cond_cache,
                neg_cache,
            });
        }
        Ok(())
    }

    /// Runs the full `QwenImage21Pipeline` generation / editing flow.
    pub fn generate(
        &mut self,
        req: &QwenImage21GenerationRequest,
    ) -> Result<QwenImage21GenerationResult> {
        let t_start = Instant::now();
        let scale = self.cfg.vae.scale_factor_spatial();
        let multiple_of = (scale * 2) as u32;
        let out_res = req.output_resolution.max(multiple_of);

        // 1. Resolve output dimensions and preprocess condition images (if any)
        let target_area = (out_res as usize) * (out_res as usize);
        let (calc_w, calc_h) = if let Some(last_img) = req.condition_images.last() {
            let (w, h) = calculate_dimensions(
                target_area,
                (last_img.width as f64) / (last_img.height.max(1) as f64),
            );
            (w as u32, h as u32)
        } else {
            (out_res, out_res)
        };
        let width = ((req.width.unwrap_or(calc_w) / multiple_of).max(1)) * multiple_of;
        let height = ((req.height.unwrap_or(calc_h) / multiple_of).max(1)) * multiple_of;
        let latent_h = (height as usize) / scale;
        let latent_w = (width as usize) / scale;
        let target_tokens = latent_h * latent_w;

        let mut vit_images = Vec::with_capacity(req.condition_images.len());
        let mut vae_cond_inputs = Vec::with_capacity(req.condition_images.len());
        let mut img_shapes = Vec::with_capacity(req.condition_images.len() + 1);

        let vis_cfg = self.cfg.text_encoder.vision_config.as_ref();
        let patch_size = vis_cfg.map(|v| v.patch_size).unwrap_or(16);
        let temporal_patch_size = vis_cfg.map(|v| v.temporal_patch_size).unwrap_or(2);
        let spatial_merge_size = vis_cfg.map(|v| v.spatial_merge_size).unwrap_or(2);

        for cond_img in &req.condition_images {
            let (in_w, in_h) = calculate_dimensions(
                target_area,
                (cond_img.width as f64) / (cond_img.height.max(1) as f64),
            );
            let in_w = (((in_w as u32) / multiple_of).max(1)) * multiple_of;
            let in_h = (((in_h as u32) / multiple_of).max(1)) * multiple_of;
            let resized = cond_img.resize_lanczos(in_w, in_h);

            let rgb_white = composite_rgba_over_white(&resized.rgba).map_err(|e| {
                CudaError::with_detail(-10, "composite RGBA over white", e.to_string())
            })?;
            let px = (in_w as usize) * (in_h as usize);
            let vit_pre = preprocess_rgb_image(
                &rgb_white,
                in_h as usize,
                in_w as usize,
                patch_size,
                temporal_patch_size,
                spatial_merge_size,
                px,
                px,
            );
            vit_images.push(vit_pre);

            let vae_chw = rgba_u8_to_vae_chw_f16(&resized.rgba, in_h as usize, in_w as usize);
            vae_cond_inputs.push((vae_chw, in_h as usize, in_w as usize));
            img_shapes.push((1usize, (in_h as usize) / scale, (in_w as usize) / scale));
        }
        img_shapes.push((1usize, latent_h, latent_w));

        // 2. Encode prompt (and optional negative prompt for CFG)
        let t_cond_start = Instant::now();
        let do_true_cfg = req.true_cfg_scale > 1.0 && req.negative_prompt.is_some();
        let cond_out = self.encode_prompt_cached(&req.prompt, &vit_images)?;
        let neg_out = if do_true_cfg {
            let neg_prompt = req.negative_prompt.as_deref().unwrap_or("");
            Some(self.encode_prompt_cached(neg_prompt, &vit_images)?)
        } else {
            None
        };
        self.ensure_transformer_loaded()?;
        let condition_encode_ms = t_cond_start.elapsed().as_secs_f64() * 1000.0;

        // 3. Encode condition images via VAE + prepare initial target latents
        let t_vae_enc_start = Instant::now();
        let z_dim = self.cfg.vae.z_dim;
        let mut cond_latents_packed = Vec::new();
        for (vae_chw, in_h, in_w) in &vae_cond_inputs {
            let cond_z_norm = self.vae.encode_normalized(vae_chw, *in_h, *in_w)?;
            let packed = pack_latents_chw_to_hwc(&cond_z_norm, z_dim, *in_h / scale, *in_w / scale);
            cond_latents_packed.extend_from_slice(&packed);
        }
        let vae_encode_ms = t_vae_enc_start.elapsed().as_secs_f64() * 1000.0;

        let init_chw = if let Some(custom) = &req.initial_latents_chw_f16 {
            if custom.len() != z_dim * latent_h * latent_w {
                return Err(CudaError::new(
                    -10,
                    "initial_latents_chw_f16 length mismatch",
                ));
            }
            custom.clone()
        } else {
            torch_randn_f16(z_dim * latent_h * latent_w, req.seed)
        };
        let init_latents = pack_latents_chw_to_hwc(&init_chw, z_dim, latent_h, latent_w);

        // 4. Build FlowMatchEulerDiscreteScheduler schedule
        let default_steps = self
            .cfg
            .sample_sigmas
            .as_ref()
            .map(|s| s.len())
            .unwrap_or(40);
        let requested_steps = req.num_inference_steps.unwrap_or(default_steps);
        let sigmas_override = if let Some(s) = &req.sigmas {
            Some(s.as_slice())
        } else if req.num_inference_steps.is_none() {
            self.cfg.sample_sigmas.as_deref()
        } else if let Some(sample_sigmas) = &self.cfg.sample_sigmas
            && sample_sigmas.len() == requested_steps
        {
            Some(sample_sigmas.as_slice())
        } else {
            None
        };
        let sched = self
            .cfg
            .scheduler
            .build_schedule(requested_steps, target_tokens, None, sigmas_override)
            .map_err(|e| CudaError::with_detail(-10, "build Euler schedule", e.to_string()))?;
        let num_steps = sched.num_steps();
        let timesteps: Vec<f32> = sched
            .timesteps
            .iter()
            .map(|&t| {
                let t_f16 = crate::cuda::f16_to_f32(f32_to_f16(t));
                crate::cuda::f16_to_f32(f32_to_f16(t_f16 / 1000.0))
            })
            .collect();
        let neg_ref = neg_out
            .as_ref()
            .map(|n| (n.prompt_embeds_f16.as_slice(), n.image_pad_mask.as_slice()));

        let is_ondemand = matches!(self.conditioning, ConditioningBackend::OnDemand { .. });
        let is_ondemand_single = self.tp_size == 1 && is_ondemand;
        if is_ondemand_single && target_tokens > 4096 {
            self.vae.release_workspace()?;
        }

        // 5. GPU-resident denoising loop (Step 0: block-causal extract; Steps 1..N-1: cached decode + CudaGraph)
        let (latents, denoise_step0_ms, denoise_cached_ms) =
            match self.transformer.as_mut().expect("transformer loaded") {
                TransformerBackend::Single {
                    engine,
                    stream,
                    cublas,
                    cond_cache,
                    neg_cache,
                } => engine.denoise_loop(
                    &cond_latents_packed,
                    &init_latents,
                    &cond_out.prompt_embeds_f16,
                    &cond_out.image_pad_mask,
                    neg_ref,
                    &img_shapes,
                    &timesteps,
                    &sched.sigmas,
                    req.true_cfg_scale,
                    req.use_cuda_graph,
                    cond_cache,
                    neg_cache,
                    cublas,
                    stream,
                )?,
                TransformerBackend::Tp2 {
                    engine,
                    cond_caches,
                    neg_caches,
                } => engine.denoise_loop(
                    &cond_latents_packed,
                    &init_latents,
                    &cond_out.prompt_embeds_f16,
                    &cond_out.image_pad_mask,
                    neg_ref,
                    &img_shapes,
                    &timesteps,
                    &sched.sigmas,
                    req.true_cfg_scale,
                    req.use_cuda_graph,
                    (&mut cond_caches.0, &mut cond_caches.1),
                    (&mut neg_caches.0, &mut neg_caches.1),
                )?,
            };

        if is_ondemand
            && target_tokens > 4096
            && let Some(TransformerBackend::Tp2 { engine, .. }) = &self.transformer
        {
            engine.release_workspaces();
        }

        // 6. Unpack latents & VAE decode
        let t_vae_dec_start = Instant::now();
        let latents_chw_f16 = unpack_latents_hwc_to_chw(&latents, z_dim, latent_h, latent_w);
        let decoded_chw_f16 = self
            .vae
            .decode_denormalized(&latents_chw_f16, latent_h, latent_w)?;
        if is_ondemand_single && target_tokens > 4096 {
            self.vae.release_workspace()?;
        }
        let channels: u8 = if req.rgba_output { 4 } else { 3 };
        let pixels_u8 = vae_chw_f16_to_rgba_u8(
            &decoded_chw_f16,
            height as usize,
            width as usize,
            req.rgba_output,
        );
        let png_bytes = encode_png_bytes(&pixels_u8, width, height, channels)
            .map_err(|e| CudaError::with_detail(-10, "encode PNG", e))?;
        let vae_decode_ms = t_vae_dec_start.elapsed().as_secs_f64() * 1000.0;
        let total_ms = t_start.elapsed().as_secs_f64() * 1000.0;

        Ok(QwenImage21GenerationResult {
            pixels_u8,
            png_bytes,
            latents_chw_f16,
            decoded_chw_f16,
            width,
            height,
            channels,
            steps: num_steps,
            seed: req.seed,
            timings: QwenImage21GenerationTimings {
                condition_encode_ms,
                vae_encode_ms,
                denoise_step0_ms,
                denoise_cached_ms,
                vae_decode_ms,
                total_ms,
            },
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn torch_randn_f16_matches_pytorch_cpu_generator_seed_42_bit_exact() {
        let vals = torch_randn_f16(23, 42);
        let expected: [u16; 23] = [
            16309, 15859, 15157, 49206, 14701, 48368, 43395, 16018, 45339, 47093, 14089, 47633,
            15440, 14952, 16057, 15646, 15664, 14562, 15703, 45929, 10584, 46086, 15073,
        ];
        assert_eq!(vals, expected);
    }

    #[test]
    fn condition_image_resize_and_psnr_work() {
        let mut rgba = Vec::with_capacity(4 * 4 * 4);
        for i in 0..16 {
            rgba.extend_from_slice(&[(i * 15) as u8, 128, 200, 255]);
        }
        let cond = QwenImage21ConditionImage::from_rgba_u8(rgba.clone(), 4, 4).unwrap();
        let png = encode_png_bytes(&rgba, 4, 4, 4).unwrap();
        let decoded = QwenImage21ConditionImage::from_image_bytes(&png).unwrap();
        assert_eq!(decoded.rgba, cond.rgba);
        assert!(compute_psnr_u8(&cond.rgba, &decoded.rgba) > 90.0);

        let resized = cond.resize_lanczos(8, 8);
        assert_eq!(resized.width, 8);
        assert_eq!(resized.height, 8);
        assert_eq!(resized.rgba.len(), 8 * 8 * 4);
    }

    #[test]
    fn premultiplied_lanczos_prevents_transparent_rgb_fringe_bleed() {
        // Left half opaque white (255,255,255,255), right half transparent green (0,255,0,0).
        // Straight-alpha Lanczos would bleed bright green into semi-transparent boundary pixels;
        // premultiplied Lanczos must keep RGB pure white (255,255,255) everywhere alpha > 0!
        let mut rgba = Vec::with_capacity(8 * 4 * 4);
        for _y in 0..4 {
            for x in 0..8 {
                if x < 4 {
                    rgba.extend_from_slice(&[255, 255, 255, 255]);
                } else {
                    rgba.extend_from_slice(&[0, 255, 0, 0]);
                }
            }
        }
        let cond = QwenImage21ConditionImage::from_rgba_u8(rgba, 8, 4).unwrap();
        let down = cond.resize_lanczos(4, 2);
        for px in down.rgba.as_chunks::<4>().0 {
            if px[3] > 0 {
                assert!(
                    px[0] >= 253 && px[1] >= 253 && px[2] >= 253,
                    "transparent RGB bled into semi-transparent pixel: {px:?}"
                );
            }
        }
    }

    #[test]
    fn condition_image_validation_and_png_encoding_edges() {
        // Invalid dimensions or byte lengths
        assert!(QwenImage21ConditionImage::from_rgba_u8(vec![0u8; 16], 0, 4).is_err());
        assert!(QwenImage21ConditionImage::from_rgba_u8(vec![0u8; 16], 4, 0).is_err());
        assert!(QwenImage21ConditionImage::from_rgba_u8(vec![0u8; 15], 2, 2).is_err());
        assert!(QwenImage21ConditionImage::from_image_bytes(b"not-an-image").is_err());

        // Same-size resize returns clone fast-path
        let cond = QwenImage21ConditionImage::from_rgba_u8(vec![200u8; 4 * 4 * 4], 4, 4).unwrap();
        let same = cond.resize_lanczos(4, 4);
        assert_eq!(same.width, 4);
        assert_eq!(same.height, 4);
        assert_eq!(same.rgba, cond.rgba);

        // Fully transparent pixels preserve RGB=0 when resizing
        let trans = QwenImage21ConditionImage::from_rgba_u8(vec![0u8; 4 * 4 * 4], 4, 4).unwrap();
        let trans_resized = trans.resize_lanczos(2, 2);
        assert!(trans_resized.rgba.iter().all(|&b| b == 0));

        // RGB (channels = 3) and invalid channel count PNG encoding
        let rgb_png = encode_png_bytes(&[128u8; 4 * 4 * 3], 4, 4, 3).unwrap();
        let decoded_rgb = QwenImage21ConditionImage::from_image_bytes(&rgb_png).unwrap();
        assert_eq!(decoded_rgb.width, 4);
        assert_eq!(decoded_rgb.height, 4);
        assert!(encode_png_bytes(&[0u8; 16], 2, 2, 2).is_err());
        assert!(encode_png_bytes(&[0u8; 10], 2, 2, 4).is_err());
        assert!(encode_png_bytes(&[], 0, 2, 4).is_err());

        // PSNR edge cases: empty / length mismatch vs noisy buffer
        assert_eq!(compute_psnr_u8(&[], &[]), f64::INFINITY);
        assert!(compute_psnr_u8(&[1, 2], &[1]).is_nan());
        let psnr_diff = compute_psnr_u8(&[100, 100, 100, 255], &[110, 90, 100, 255]);
        assert!(psnr_diff > 20.0 && psnr_diff < 50.0);
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn gpu_synthetic_pipeline_t2i_ti2i_cfg_rgba_cache_and_ondemand() {
        let _gpu_guard = crate::LIB_GPU_ENGINE_LOCK
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        if CudaDevice::count().unwrap_or(0) < 1 {
            return;
        }
        let tmp = tempfile::tempdir().unwrap();
        crate::qwen_image::tests::write_synthetic_qwen_image_pipeline_dir(tmp.path());

        // Invalid tp_size rejected
        assert!(QwenImage21PipelineEngine::load(tmp.path(), 3).is_err());

        // 1. Load TP=1 pipeline and test T2I RGB + RGBA + CFG + custom initial latents + custom sigmas
        let mut engine = QwenImage21PipelineEngine::load(tmp.path(), 1).unwrap();
        assert_eq!(engine.tp_size(), 1);

        let req_rgb = QwenImage21GenerationRequest {
            prompt: "a synthwave fox <|image_pad|>".to_string(),
            negative_prompt: None,
            condition_images: Vec::new(),
            width: Some(64),
            height: Some(64),
            output_resolution: 64,
            num_inference_steps: Some(2),
            true_cfg_scale: 1.0,
            seed: 42,
            rgba_output: false,
            use_cuda_graph: true,
            sigmas: None,
            initial_latents_chw_f16: None,
        };
        let out_rgb = engine.generate(&req_rgb).unwrap();
        assert_eq!(out_rgb.width, 64);
        assert_eq!(out_rgb.height, 64);
        assert_eq!(out_rgb.channels, 3);
        assert_eq!(out_rgb.steps, 2);
        assert_eq!(out_rgb.pixels_u8.len(), 64 * 64 * 3);
        assert!(!out_rgb.png_bytes.is_empty());
        assert!(
            out_rgb
                .decoded_chw_f16
                .iter()
                .all(|&bits| crate::cuda::f16_to_f32(bits).is_finite())
        );

        // Second call hits conditioning cache and supports RGBA + true CFG + custom sigmas + custom latents
        let mut req_rgba = req_rgb.clone();
        req_rgba.rgba_output = true;
        req_rgba.true_cfg_scale = 2.5;
        req_rgba.negative_prompt = Some("blurry low quality".to_string());
        req_rgba.sigmas = Some(vec![1.0, 0.4]);
        req_rgba.initial_latents_chw_f16 = Some(out_rgb.latents_chw_f16.clone());
        let out_rgba = engine.generate(&req_rgba).unwrap();
        assert_eq!(out_rgba.channels, 4);
        assert_eq!(out_rgba.pixels_u8.len(), 64 * 64 * 4);

        // Mismatched initial_latents_chw_f16 length fails cleanly
        let mut bad_latents = req_rgb.clone();
        bad_latents.initial_latents_chw_f16 = Some(vec![0u16; 3]);
        assert!(engine.generate(&bad_latents).is_err());

        // 2. Test TI2I with 1 and 2 condition images (including RGBA transparency)
        let cond_img =
            QwenImage21ConditionImage::from_rgba_u8(vec![180u8; 64 * 64 * 4], 64, 64).unwrap();
        let req_ti2i = QwenImage21GenerationRequest {
            prompt: "add neon rain".to_string(),
            negative_prompt: Some(String::new()),
            condition_images: vec![cond_img.clone(), cond_img],
            width: None,
            height: None,
            output_resolution: 64,
            num_inference_steps: None,
            true_cfg_scale: 2.0,
            seed: 7,
            rgba_output: true,
            use_cuda_graph: true,
            sigmas: None,
            initial_latents_chw_f16: None,
        };
        let out_ti2i = engine.generate(&req_ti2i).unwrap();
        assert_eq!(out_ti2i.width, 64);
        assert_eq!(out_ti2i.height, 64);
        assert_eq!(out_ti2i.channels, 4);

        // 3. Test ConditioningBackend::OnDemand on TP=1 (unloads transformer + vae workspace, reloads after VLM)
        engine.force_ondemand_conditioning_for_test();
        let out_ondemand = engine.generate(&req_rgb).unwrap();
        assert_eq!(out_ondemand.pixels_u8, out_rgb.pixels_u8);

        // 4. If 2 GPUs are visible, test TP=2 (auto tp=0) + OnDemand conditioning on TP=2
        if CudaDevice::count().unwrap_or(0) >= 2 {
            let mut engine_tp2 = QwenImage21PipelineEngine::load(tmp.path(), 0).unwrap();
            assert_eq!(engine_tp2.tp_size(), 2);
            let out_tp2 = engine_tp2.generate(&req_ti2i).unwrap();
            assert_eq!(out_tp2.width, 64);
            assert_eq!(out_tp2.height, 64);
            assert!(compute_psnr_u8(&out_ti2i.pixels_u8, &out_tp2.pixels_u8) >= 40.0);

            engine_tp2.force_ondemand_conditioning_for_test();
            let out_tp2_ondemand = engine_tp2.generate(&req_rgb).unwrap();
            assert!(compute_psnr_u8(&out_rgb.pixels_u8, &out_tp2_ondemand.pixels_u8) >= 40.0);
        }
    }
}
