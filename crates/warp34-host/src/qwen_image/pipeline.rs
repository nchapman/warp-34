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

use image::codecs::png::PngEncoder;
use image::imageops::FilterType;
use image::{ColorType, ExtendedColorType, ImageEncoder, RgbaImage};
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
    QwenImage21TransformerEngine, QwenImage21TransformerTp2Engine, euler_cfg_step_f16,
    pack_latents_chw_to_hwc, unpack_latents_hwc_to_chw,
};
#[cfg(feature = "cuda")]
use crate::qwen_image::vae::{
    QwenImage21VaeEngine, QwenImage21VaeWeightsGpu, rgba_u8_to_vae_chw_f16, vae_chw_f16_to_rgba_u8,
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

    /// Resizes the condition image to `(target_width, target_height)` using Lanczos3
    /// (matching `VaeImageProcessor.resize(..., resample="lanczos")`).
    pub fn resize_lanczos(&self, target_width: u32, target_height: u32) -> Self {
        let target_width = target_width.max(1);
        let target_height = target_height.max(1);
        if self.width == target_width && self.height == target_height {
            return self.clone();
        }
        let Some(src) = RgbaImage::from_raw(self.width, self.height, self.rgba.clone()) else {
            return self.clone();
        };
        let resized =
            image::imageops::resize(&src, target_width, target_height, FilterType::Lanczos3);
        Self {
            rgba: resized.into_raw(),
            width: target_width,
            height: target_height,
        }
    }
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
    let mut buf = Vec::new();
    let encoder = PngEncoder::new(Cursor::new(&mut buf));
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
    },
    Tp2 {
        engine: QwenImage21TransformerTp2Engine,
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
    /// - When `tp_size == 2` (or `0` auto on `>= 2` GPUs): keeps `text_encoder` on GPU 0,
    ///   `TP=2` `transformer` across GPU 0 + GPU 1, and `vae` on GPU 1 (100% resident in VRAM).
    /// - When `tp_size == 1` on `>= 2` GPUs: keeps `transformer` + `vae` on GPU 0 and
    ///   `text_encoder` on GPU 1 (100% resident in VRAM).
    /// - When only 1 GPU is available: keeps all 3 resident if VRAM permits (`>= 34 GiB`),
    ///   or automatically swaps `text_encoder` and `transformer` on GPU 0 with an in-memory
    ///   conditioning cache on 24 GiB cards.
    pub fn load(model_dir: &Path, requested_tp: usize) -> Result<Self> {
        let cfg = QwenImage21PipelineConfig::from_dir(model_dir).map_err(|e| {
            CudaError::with_detail(-10, "load QwenImage21PipelineConfig", e.to_string())
        })?;
        let tok_path = cfg.processor_dir.join("tokenizer.json");
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

        // Enable spatial tiling for >1024px images while keeping <=1024px untiled for maximum speed.
        let vae_dev = if num_gpus >= 2 && tp_size == 2 { 1 } else { 0 };
        let vae_weights =
            QwenImage21VaeWeightsGpu::from_archive(&vae_archive, &cfg.vae, "", vae_dev)?;
        let vae = QwenImage21VaeEngine::new(vae_weights)?;

        let tf_quant = &cfg.transformer.quant;
        let te_quant = &cfg.text_encoder.quant;
        let (transformer, conditioning) = if tp_size == 2 {
            let tf_tp2 = QwenImage21TransformerTp2Engine::load_with_prefix(
                &transformer_archive,
                &cfg.transformer,
                tf_quant,
                "",
                8192,
            )?;
            let te_stream = CudaStream::new(0)?;
            let te_cublas = CublasHandle::new(&te_stream)?;
            let te_encoder = Qwen3VlConditioningEncoder::load(
                Arc::clone(&text_archive),
                &cfg.text_encoder,
                te_quant,
                0,
                &te_stream,
            )?;
            (
                Some(TransformerBackend::Tp2 { engine: tf_tp2 }),
                ConditioningBackend::Resident {
                    encoder: te_encoder,
                    stream: te_stream,
                    cublas: te_cublas,
                },
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
        let non_empty = if prompt.is_empty() { " " } else { prompt };
        let formatted = if images.is_empty() {
            format_t2i_prompt(non_empty)
        } else {
            format_ti2i_prompt(non_empty, images.len())
        };
        let enc = self.tokenizer.encode(formatted, false).map_err(|e| {
            CudaError::with_detail(-10, "tokenize QwenImage21 prompt", e.to_string())
        })?;
        let raw_ids = enc.get_ids();
        if images.is_empty() {
            return Ok(raw_ids.iter().map(|&id| id as i32).collect());
        }
        let spatial_merge_size = self
            .cfg
            .text_encoder
            .vision_config
            .as_ref()
            .map(|v| v.spatial_merge_size)
            .unwrap_or(2);
        let mut expanded = Vec::with_capacity(
            raw_ids.len()
                + images
                    .iter()
                    .map(|i| i.num_merged_tokens(spatial_merge_size))
                    .sum::<usize>(),
        );
        let mut img_idx = 0usize;
        for &id in raw_ids {
            if (id as i32) == IMAGE_PAD_TOKEN_ID {
                let count = images
                    .get(img_idx)
                    .map(|img| img.num_merged_tokens(spatial_merge_size))
                    .ok_or_else(|| {
                        CudaError::new(-10, "more <|image_pad|> tokens than condition images")
                    })?;
                expanded.extend(std::iter::repeat_n(IMAGE_PAD_TOKEN_ID, count));
                img_idx += 1;
            } else {
                expanded.push(id as i32);
            }
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
                // Free the single-GPU transformer before loading the 14.4 GiB text encoder.
                self.transformer = None;
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
            self.transformer = Some(TransformerBackend::Single {
                engine: tf_single,
                stream: tf_stream,
                cublas: tf_cublas,
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
        let mut latents = pack_latents_chw_to_hwc(&init_chw, z_dim, latent_h, latent_w);

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

        // 5. Denoising loop (Step 0: block-causal extract; Steps 1..N-1: cached decode + CudaGraph)
        let mut denoise_step0_ms = 0.0f64;
        let mut denoise_cached_ms = 0.0f64;

        match self.transformer.as_mut().expect("transformer loaded") {
            TransformerBackend::Single {
                engine,
                stream,
                cublas,
            } => {
                let mut cond_cache = engine.new_kv_cache();
                let mut neg_cache = engine.new_kv_cache();
                for step in 0..num_steps {
                    let t_step = Instant::now();
                    let t_f16 = crate::cuda::f16_to_f32(f32_to_f16(sched.timesteps[step]));
                    let timestep = crate::cuda::f16_to_f32(f32_to_f16(t_f16 / 1000.0));
                    let (v_cond, v_uncond) = if step == 0 {
                        let out_c = engine.forward_extract(
                            &cond_latents_packed,
                            &latents,
                            &cond_out.prompt_embeds_f16,
                            &cond_out.image_pad_mask,
                            &img_shapes,
                            timestep,
                            Some(&mut cond_cache),
                            cublas,
                            stream,
                        )?;
                        let out_u = if let Some(neg) = &neg_out {
                            Some(
                                engine
                                    .forward_extract(
                                        &cond_latents_packed,
                                        &latents,
                                        &neg.prompt_embeds_f16,
                                        &neg.image_pad_mask,
                                        &img_shapes,
                                        timestep,
                                        Some(&mut neg_cache),
                                        cublas,
                                        stream,
                                    )?
                                    .target_output_f16,
                            )
                        } else {
                            None
                        };
                        denoise_step0_ms += t_step.elapsed().as_secs_f64() * 1000.0;
                        (out_c.target_output_f16, out_u)
                    } else {
                        let out_c = engine.forward_cached(
                            &latents,
                            timestep,
                            &mut cond_cache,
                            true,
                            cublas,
                            stream,
                        )?;
                        let out_u = if neg_out.is_some() {
                            Some(engine.forward_cached(
                                &latents,
                                timestep,
                                &mut neg_cache,
                                true,
                                cublas,
                                stream,
                            )?)
                        } else {
                            None
                        };
                        denoise_cached_ms += t_step.elapsed().as_secs_f64() * 1000.0;
                        (out_c, out_u)
                    };
                    euler_cfg_step_f16(
                        &mut latents,
                        &v_cond,
                        v_uncond.as_deref(),
                        req.true_cfg_scale,
                        sched.sigmas[step],
                        sched.sigmas[step + 1],
                    );
                }
            }
            TransformerBackend::Tp2 { engine } => {
                let (mut cond_c0, mut cond_c1) = engine.new_kv_caches();
                let (mut neg_c0, mut neg_c1) = engine.new_kv_caches();
                for step in 0..num_steps {
                    let t_step = Instant::now();
                    let t_f16 = crate::cuda::f16_to_f32(f32_to_f16(sched.timesteps[step]));
                    let timestep = crate::cuda::f16_to_f32(f32_to_f16(t_f16 / 1000.0));
                    let (v_cond, v_uncond) = if step == 0 {
                        let out_c = engine.forward_extract(
                            &cond_latents_packed,
                            &latents,
                            &cond_out.prompt_embeds_f16,
                            &cond_out.image_pad_mask,
                            &img_shapes,
                            timestep,
                            Some((&mut cond_c0, &mut cond_c1)),
                        )?;
                        let out_u = if let Some(neg) = &neg_out {
                            Some(
                                engine
                                    .forward_extract(
                                        &cond_latents_packed,
                                        &latents,
                                        &neg.prompt_embeds_f16,
                                        &neg.image_pad_mask,
                                        &img_shapes,
                                        timestep,
                                        Some((&mut neg_c0, &mut neg_c1)),
                                    )?
                                    .target_output_f16,
                            )
                        } else {
                            None
                        };
                        denoise_step0_ms += t_step.elapsed().as_secs_f64() * 1000.0;
                        (out_c.target_output_f16, out_u)
                    } else {
                        let out_c = engine.forward_cached(
                            &latents,
                            timestep,
                            (&mut cond_c0, &mut cond_c1),
                        )?;
                        let out_u = if neg_out.is_some() {
                            Some(engine.forward_cached(
                                &latents,
                                timestep,
                                (&mut neg_c0, &mut neg_c1),
                            )?)
                        } else {
                            None
                        };
                        denoise_cached_ms += t_step.elapsed().as_secs_f64() * 1000.0;
                        (out_c, out_u)
                    };
                    euler_cfg_step_f16(
                        &mut latents,
                        &v_cond,
                        v_uncond.as_deref(),
                        req.true_cfg_scale,
                        sched.sigmas[step],
                        sched.sigmas[step + 1],
                    );
                }
            }
        }

        // 6. Unpack latents & VAE decode
        let t_vae_dec_start = Instant::now();
        let latents_chw_f16 = unpack_latents_hwc_to_chw(&latents, z_dim, latent_h, latent_w);
        let decoded_chw_f16 = self
            .vae
            .decode_denormalized(&latents_chw_f16, latent_h, latent_w)?;
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
}
