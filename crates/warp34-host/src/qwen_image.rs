//! `Qwen/Qwen-Image-2.1` and `Qwen/Qwen-Image-2.1-Turbo` multi-component pipeline
//! configuration, `FlowMatchEulerDiscreteScheduler`, token/RoPE metadata, and prompt templates.

use crate::loader::{ModelConfig, QuantScheme, SafetensorsArchive, sanitize_diag};
use serde_json::Value as Json;
use std::fs;
use std::io;
use std::path::{Path, PathBuf};

#[cfg(feature = "cuda")]
#[path = "qwen_image/text_encoder.rs"]
pub mod text_encoder;
#[cfg(feature = "cuda")]
pub use text_encoder::{Qwen3VlConditioningEncoder, Qwen3VlConditioningOutput};

#[path = "qwen_image/transformer.rs"]
pub mod transformer;
#[cfg(feature = "cuda")]
pub use transformer::{
    QwenImage21ExtractOutput, QwenImage21KvCacheGpu, QwenImage21TransformerEngine,
    QwenImage21TransformerTp2Engine,
};
pub use transformer::{euler_cfg_step_f16, pack_latents_chw_to_hwc, unpack_latents_hwc_to_chw};

#[cfg(feature = "cuda")]
#[path = "qwen_image/vae.rs"]
pub mod vae;
#[cfg(feature = "cuda")]
pub use vae::{
    QwenImage21VaeEngine, QwenImage21VaeTilingConfig, QwenImage21VaeWeightsGpu,
    denormalize_latents_f16, normalize_latents_f16, rgba_u8_to_vae_chw_f16, vae_chw_f16_to_rgba_u8,
};

#[path = "qwen_image/pipeline.rs"]
pub mod pipeline;
#[cfg(feature = "cuda")]
pub use pipeline::QwenImage21PipelineEngine;
pub use pipeline::{
    QwenImage21ConditionImage, QwenImage21GenerationRequest, QwenImage21GenerationResult,
    QwenImage21GenerationTimings, compute_psnr_u8, encode_png_bytes, torch_randn_f16,
};

/// System prompt used by `QwenImage21Pipeline` in `diffusers`.
pub const SYS_PROMPT: &str = "Comprehend and analyze the provided prompt.";

/// Default text-to-image system + user chat template from `QwenImage21Pipeline`.
pub const PROMPT_TEMPLATE_T2I: &str = "<|im_start|>system\nComprehend and analyze the provided prompt.<|im_end|>\n<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n";

/// Default image-conditioned system + user chat template from `QwenImage21Pipeline`.
pub const PROMPT_TEMPLATE_TI2I: &str = "<|im_start|>system\nComprehend and analyze the provided prompt.<|im_end|>\n<|im_start|>user\n<image1><|vision_start|><|image_pad|><|vision_end|>{}<|im_end|>\n<|im_start|>assistant\n";

/// Number of leading system-role tokens dropped from the layer-35 hidden states (`_drop_idx = 14`).
pub const DEFAULT_DROP_IDX: usize = 14;

/// Token ID for `<|image_pad|>` in `Qwen3VLProcessor` (`151655`).
pub const IMAGE_PAD_TOKEN_ID: i32 = 151655;

/// Each vision-language `<|image_pad|>` slot stands for a `2x2` group (`4`) of latent tokens.
pub const IMG_TOKENS_PER_SLOT: usize = 4;

/// Default output resolution (`1024`) and target pixel area (`1024 * 1024 = 1_048_576`).
pub const DEFAULT_OUTPUT_RESOLUTION: usize = 1024;
pub const VAE_IMAGE_SIZE: usize = DEFAULT_OUTPUT_RESOLUTION * DEFAULT_OUTPUT_RESOLUTION;

/// Formats a user prompt using `PROMPT_TEMPLATE_T2I` (text-to-image mode).
/// Matches `QwenImage21Pipeline._get_qwen_prompt_embeds` (empty prompt becomes `" "`).
pub fn format_t2i_prompt(user_prompt: &str) -> String {
    let p = if user_prompt.is_empty() {
        " "
    } else {
        user_prompt
    };
    PROMPT_TEMPLATE_T2I.replace("{}", p)
}

/// Applies Qwen-Image-2.1's trained RGBA transparency trigger envelope when not already present:
/// `"This is an RGBA image with transparency. <prompt>. The image has alpha channel and the background is transparent."`
pub fn ensure_rgba_transparency_prompt(prompt: &str) -> String {
    let trimmed = prompt.trim();
    if trimmed.is_empty() {
        return String::from(
            "This is an RGBA image with transparency. The image has alpha channel and the background is transparent.",
        );
    }
    let lower = trimmed.to_ascii_lowercase();
    if lower.contains("rgba image") || lower.contains("alpha channel") {
        return prompt.to_string();
    }
    let body = trimmed.trim_end_matches('.');
    format!(
        "This is an RGBA image with transparency. {body}. The image has alpha channel and the background is transparent."
    )
}

/// Formats a user prompt and `num_images` vision placeholders using `PROMPT_TEMPLATE_TI2I`.
/// Matches `QwenImage21Pipeline._get_qwen_prompt_embeds`.
pub fn format_ti2i_prompt(user_prompt: &str, num_images: usize) -> String {
    let p = if user_prompt.is_empty() {
        " "
    } else {
        user_prompt
    };
    if num_images == 0 {
        return format_t2i_prompt(p);
    }
    let mut replace = String::from("<image1><|vision_start|><|image_pad|><|vision_end|>");
    for i in 2..=num_images {
        replace.push_str(&format!(
            " <image{i}><|vision_start|><|image_pad|><|vision_end|>"
        ));
    }
    let tpl = PROMPT_TEMPLATE_TI2I.replace(
        "<image1><|vision_start|><|image_pad|><|vision_end|>",
        &replace,
    );
    tpl.replace("{}", p)
}

/// Composites an `[H, W, 4]` RGBA `u8` buffer over opaque white `(255, 255, 255)` into `[H, W, 3]` RGB `u8`,
/// matching `PIL.Image.new("RGB", size, (255, 255, 255)); white.paste(img, mask=A)` in `QwenImage21Pipeline`.
pub fn composite_rgba_over_white(rgba: &[u8]) -> io::Result<Vec<u8>> {
    if !rgba.len().is_multiple_of(4) {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "RGBA buffer length must be a multiple of 4",
        ));
    }
    let num_px = rgba.len() / 4;
    let mut rgb = Vec::with_capacity(num_px * 3);
    for px in rgba.as_chunks::<4>().0 {
        let a = px[3] as u32;
        let inv_a = 255 - a;
        // PIL's integer paste formula: `(src * a + 255 * (255 - a) + 127) / 255`
        let r = ((px[0] as u32 * a + 255 * inv_a + 127) / 255) as u8;
        let g = ((px[1] as u32 * a + 255 * inv_a + 127) / 255) as u8;
        let b = ((px[2] as u32 * a + 255 * inv_a + 127) / 255) as u8;
        rgb.push(r);
        rgb.push(g);
        rgb.push(b);
    }
    Ok(rgb)
}

/// Computes `(width, height)` rounded to multiples of `32` for a target pixel area and aspect ratio
/// (`width / height`), matching `calculate_dimensions` in `diffusers`'s `pipeline_qwenimage21.py`.
pub fn calculate_dimensions(target_area: usize, ratio: f64) -> (usize, usize) {
    let width = (target_area as f64 * ratio).sqrt();
    let height = width / ratio;
    let width = ((width / 32.0).round_ties_even() as usize).max(1) * 32;
    let height = ((height / 32.0).round_ties_even() as usize).max(1) * 32;
    (width, height)
}

/// Labels every token of the expanded joint sequence with its image block ID (`-1` for text tokens,
/// `0..num_images` for each image block in `img_shapes`) and produces `target_token_mask` (`true` on
/// the final image block, which is the target image), matching
/// `QwenImage21Transformer2DModel.build_token_metadata`.
pub fn build_token_metadata(
    image_pad_mask: &[bool],
    img_shapes: &[[usize; 3]],
) -> io::Result<(Vec<i32>, Vec<bool>)> {
    if img_shapes.is_empty() {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "img_shapes must contain at least the target image shape",
        ));
    }
    let block_lengths: Vec<usize> = img_shapes.iter().map(|&[f, h, w]| f * h * w).collect();
    let total_img_tokens: usize = block_lengths.iter().sum();
    let mask_img_tokens = image_pad_mask.iter().filter(|&&b| b).count();
    if total_img_tokens != mask_img_tokens {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            format!(
                "img_shapes accounts for {total_img_tokens} image tokens but image_pad_mask marks {mask_img_tokens}"
            ),
        ));
    }

    let seq_len = image_pad_mask.len();
    let mut image_ids = vec![-1i32; seq_len];
    let mut target_token_mask = vec![false; seq_len];
    let target_block_id = (img_shapes.len() - 1) as i32;

    let mut cur_block = 0usize;
    let mut rem_in_block = block_lengths[0];
    for (idx, &is_img) in image_pad_mask.iter().enumerate() {
        if !is_img {
            continue;
        }
        while rem_in_block == 0 && cur_block + 1 < block_lengths.len() {
            cur_block += 1;
            rem_in_block = block_lengths[cur_block];
        }
        let bid = cur_block as i32;
        image_ids[idx] = bid;
        if bid == target_block_id {
            target_token_mask[idx] = true;
        }
        rem_in_block = rem_in_block.saturating_sub(1);
    }

    Ok((image_ids, target_token_mask))
}

/// Splits the prefix `[0..prefix_len)` into contiguous `(start, end, is_text)` segments of equal
/// `image_ids`, matching `_qwenimage21_prefix_segments` in `transformer_qwenimage21.py`.
pub fn prefix_segments(image_ids: &[i32], prefix_len: usize) -> Vec<(usize, usize, bool)> {
    let prefix_len = prefix_len.min(image_ids.len());
    if prefix_len == 0 {
        return Vec::new();
    }
    let mut segments = Vec::new();
    let mut start = 0usize;
    for index in 1..=prefix_len {
        if index == prefix_len || image_ids[index] != image_ids[start] {
            segments.push((start, index, image_ids[start] < 0));
            start = index;
        }
    }
    segments
}

/// Computes the 3-axis `(frame_index, height_index, width_index)` signed integer coordinates
/// for every token in the expanded joint sequence, matching `QwenImage21Rope.forward` in
/// `diffusers/models/transformers/transformer_qwenimage21.py`.
pub fn compute_qwen_image_21_rope_indices(
    img_shapes: &[[usize; 3]],
    image_pad_mask: &[bool],
) -> io::Result<[Vec<i32>; 3]> {
    let total_len = image_pad_mask.len();
    let mut frame_index = Vec::with_capacity(total_len);
    let mut image_height_index = Vec::new();
    let mut image_width_index = Vec::new();

    let mut cursor = 0usize;
    let mut position = 0i32;

    for &[frames, height, width] in img_shapes {
        if frames != 1 {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                format!("QwenImage21Rope expects frame==1 per image shape, got {frames}"),
            ));
        }
        let rel_start = image_pad_mask[cursor..]
            .iter()
            .position(|&b| b)
            .ok_or_else(|| {
                io::Error::new(
                    io::ErrorKind::InvalidInput,
                    "image_pad_mask ended before all img_shapes blocks were placed",
                )
            })?;
        let block_start = cursor + rel_start;
        let text_len = block_start - cursor;
        for t in 0..text_len {
            frame_index.push(position + t as i32);
        }
        position += text_len as i32;

        let block_tokens = height * width;
        cursor = block_start + block_tokens;
        if cursor > total_len {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "img_shapes block exceeds image_pad_mask length",
            ));
        }
        if image_pad_mask[block_start..cursor].iter().any(|&b| !b) {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "image_pad_mask must be contiguous across each img_shapes block",
            ));
        }
        frame_index.resize(frame_index.len() + block_tokens, position);
        position += height.max(width) as i32;

        let h_start = -((height - height / 2) as i32);
        let h_end = (height / 2) as i32;
        let w_start = -((width - width / 2) as i32);
        let w_end = (width / 2) as i32;
        for h in h_start..h_end {
            for _ in 0..width {
                image_height_index.push(h);
            }
        }
        for _ in 0..height {
            for w in w_start..w_end {
                image_width_index.push(w);
            }
        }
    }

    if cursor < total_len {
        if image_pad_mask[cursor..].iter().any(|&b| b) {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "image_pad_mask contains extra image tokens beyond img_shapes",
            ));
        }
        let rem = total_len - cursor;
        for t in 0..rem {
            frame_index.push(position + t as i32);
        }
    }

    let mut height_index = frame_index.clone();
    let mut width_index = frame_index.clone();
    let mut img_cursor = 0usize;
    for (idx, &is_img) in image_pad_mask.iter().enumerate() {
        if is_img {
            height_index[idx] = image_height_index[img_cursor];
            width_index[idx] = image_width_index[img_cursor];
            img_cursor += 1;
        }
    }
    Ok([frame_index, height_index, width_index])
}

/// Computes `[seq_len, head_dim / 2]` `(cos, sin)` `FP32` tables for the 3-axis `QwenImage21Rope`
/// (`axes_dims_rope = [16, 56, 56]`, `theta = 10000.0`), matching `QwenImage21Rope.forward` in
/// `diffusers/models/transformers/transformer_qwenimage21.py`.
pub fn compute_qwen_image_21_rope_cos_sin(
    img_shapes: &[[usize; 3]],
    image_pad_mask: &[bool],
    axes_dims_rope: [usize; 3],
    theta: f32,
) -> io::Result<(Vec<f32>, Vec<f32>)> {
    let indices = compute_qwen_image_21_rope_indices(img_shapes, image_pad_mask)?;
    let seq_len = image_pad_mask.len();
    let half_head_dim: usize = axes_dims_rope.iter().map(|&d| d / 2).sum();
    let mut cos_out = vec![0.0f32; seq_len * half_head_dim];
    let mut sin_out = vec![0.0f32; seq_len * half_head_dim];

    let mut inv_freqs: [Vec<f32>; 3] = [Vec::new(), Vec::new(), Vec::new()];
    for (inv_f_slot, &dim) in inv_freqs.iter_mut().zip(axes_dims_rope.iter()) {
        *inv_f_slot = (0..dim / 2)
            .map(|k| 1.0f32 / theta.powf(((2 * k) as f32) / (dim as f32)))
            .collect();
    }

    for (tok, (cos_row, sin_row)) in cos_out
        .chunks_exact_mut(half_head_dim)
        .zip(sin_out.chunks_exact_mut(half_head_dim))
        .enumerate()
    {
        let mut pair_off = 0usize;
        for (axis_indices, axis_inv_freqs) in indices.iter().zip(inv_freqs.iter()) {
            let coord = axis_indices[tok] as f32;
            for &inv_f in axis_inv_freqs {
                let angle = coord * inv_f;
                cos_row[pair_off] = angle.cos();
                sin_row[pair_off] = angle.sin();
                pair_off += 1;
            }
        }
    }
    Ok((cos_out, sin_out))
}

/// Configuration for `QwenImage21Transformer2DModel` (`transformer/config.json`).
#[derive(Clone, Debug, PartialEq)]
pub struct QwenImage21TransformerConfig {
    pub patch_size: usize,
    pub in_channels: usize,
    pub out_channels: usize,
    pub num_layers: usize,
    pub attention_head_dim: usize,
    pub num_attention_heads: usize,
    pub context_in_dim: usize,
    pub mlp_ratio: usize,
    pub axes_dims_rope: [usize; 3],
    pub causal_condition: bool,
    pub eps: f32,
    pub quant: QuantScheme,
}

impl QwenImage21TransformerConfig {
    pub const fn inner_dim(&self) -> usize {
        self.num_attention_heads * self.attention_head_dim
    }

    pub const fn mlp_hidden_dim(&self) -> usize {
        self.inner_dim() * self.mlp_ratio
    }

    pub fn from_dir(dir: &Path) -> io::Result<Self> {
        let text = fs::read_to_string(dir.join("config.json"))?;
        let root: Json = serde_json::from_str(&text).map_err(|e| {
            io::Error::new(
                io::ErrorKind::InvalidData,
                format!("bad transformer/config.json: {e}"),
            )
        })?;
        Self::from_json(&root)
    }

    pub fn from_json(root: &Json) -> io::Result<Self> {
        if let Some(cls) = root.get("_class_name").and_then(|v| v.as_str())
            && cls != "QwenImage21Transformer2DModel"
        {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!(
                    "unsupported transformer _class_name='{}' (expected 'QwenImage21Transformer2DModel')",
                    sanitize_diag(cls)
                ),
            ));
        }
        let patch_size = root.get("patch_size").and_then(|v| v.as_u64()).unwrap_or(1) as usize;
        let in_channels = root
            .get("in_channels")
            .and_then(|v| v.as_u64())
            .unwrap_or(64) as usize;
        let out_channels = root
            .get("out_channels")
            .and_then(|v| v.as_u64())
            .unwrap_or(in_channels as u64) as usize;
        let num_layers = root
            .get("num_layers")
            .and_then(|v| v.as_u64())
            .unwrap_or(32) as usize;
        let attention_head_dim = root
            .get("attention_head_dim")
            .and_then(|v| v.as_u64())
            .unwrap_or(128) as usize;
        let num_attention_heads = root
            .get("num_attention_heads")
            .and_then(|v| v.as_u64())
            .unwrap_or(32) as usize;
        let context_in_dim = root
            .get("context_in_dim")
            .and_then(|v| v.as_u64())
            .unwrap_or(4096) as usize;
        let mlp_ratio = root.get("mlp_ratio").and_then(|v| v.as_u64()).unwrap_or(3) as usize;
        let mut axes_dims_rope = [16usize, 56, 56];
        if let Some(arr) = root.get("axes_dims_rope").and_then(|v| v.as_array()) {
            if arr.len() != 3 {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    "transformer axes_dims_rope must have 3 elements",
                ));
            }
            for (i, v) in arr.iter().enumerate() {
                axes_dims_rope[i] = v.as_u64().ok_or_else(|| {
                    io::Error::new(io::ErrorKind::InvalidData, "invalid axes_dims_rope element")
                })? as usize;
            }
        }
        let rope_sum: usize = axes_dims_rope.iter().sum();
        if rope_sum != attention_head_dim || axes_dims_rope.iter().any(|&d| !d.is_multiple_of(2)) {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!(
                    "axes_dims_rope {:?} must be even and sum to attention_head_dim={attention_head_dim}",
                    axes_dims_rope
                ),
            ));
        }
        let causal_condition = root
            .get("causal_condition")
            .and_then(|v| v.as_bool())
            .unwrap_or(true);
        let eps = root.get("eps").and_then(|v| v.as_f64()).unwrap_or(1e-6) as f32;
        let quant = ModelConfig::parse_quant(root.get("quantization_config"))?;
        if patch_size == 0
            || in_channels == 0
            || out_channels == 0
            || num_layers == 0
            || attention_head_dim == 0
            || num_attention_heads == 0
        {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "transformer dimensions must all be > 0",
            ));
        }
        Ok(Self {
            patch_size,
            in_channels,
            out_channels,
            num_layers,
            attention_head_dim,
            num_attention_heads,
            context_in_dim,
            mlp_ratio,
            axes_dims_rope,
            causal_condition,
            eps,
            quant,
        })
    }
}

/// Configuration for `AutoencoderKLQwenImage21` (`vae/config.json`).
#[derive(Clone, Debug, PartialEq)]
pub struct QwenImage21VaeConfig {
    pub base_dim: usize,
    pub decoder_base_dim: usize,
    pub z_dim: usize,
    pub dim_mult: Vec<usize>,
    pub num_res_blocks: usize,
    pub attn_scales: Vec<f32>,
    pub temperal_downsample: Vec<bool>,
    pub in_channels: usize,
    pub out_channels: usize,
    pub is_residual: bool,
    pub latents_mean: Vec<f32>,
    pub latents_std: Vec<f32>,
}

impl QwenImage21VaeConfig {
    pub fn scale_factor_spatial(&self) -> usize {
        1usize << self.temperal_downsample.len()
    }

    pub fn scale_factor_temporal(&self) -> usize {
        1usize << self.temperal_downsample.iter().filter(|&&b| b).count()
    }

    pub fn from_dir(dir: &Path) -> io::Result<Self> {
        let text = fs::read_to_string(dir.join("config.json"))?;
        let root: Json = serde_json::from_str(&text).map_err(|e| {
            io::Error::new(
                io::ErrorKind::InvalidData,
                format!("bad vae/config.json: {e}"),
            )
        })?;
        Self::from_json(&root)
    }

    pub fn from_json(root: &Json) -> io::Result<Self> {
        if let Some(cls) = root.get("_class_name").and_then(|v| v.as_str())
            && cls != "AutoencoderKLQwenImage21"
        {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!(
                    "unsupported VAE _class_name='{}' (expected 'AutoencoderKLQwenImage21')",
                    sanitize_diag(cls)
                ),
            ));
        }
        let base_dim = root.get("base_dim").and_then(|v| v.as_u64()).unwrap_or(96) as usize;
        let decoder_base_dim = root
            .get("decoder_base_dim")
            .and_then(|v| v.as_u64())
            .unwrap_or(base_dim as u64) as usize;
        let z_dim = root.get("z_dim").and_then(|v| v.as_u64()).unwrap_or(64) as usize;
        let dim_mult = match root.get("dim_mult") {
            Some(v) => v
                .as_array()
                .ok_or_else(|| {
                    io::Error::new(io::ErrorKind::InvalidData, "vae dim_mult must be an array")
                })?
                .iter()
                .map(|x| {
                    x.as_u64().map(|u| u as usize).ok_or_else(|| {
                        io::Error::new(
                            io::ErrorKind::InvalidData,
                            "invalid integer in vae dim_mult",
                        )
                    })
                })
                .collect::<io::Result<Vec<_>>>()?,
            None => vec![1, 2, 4, 8, 8],
        };
        let num_res_blocks = root
            .get("num_res_blocks")
            .and_then(|v| v.as_u64())
            .unwrap_or(2) as usize;
        let attn_scales = match root.get("attn_scales") {
            Some(v) => v
                .as_array()
                .ok_or_else(|| {
                    io::Error::new(
                        io::ErrorKind::InvalidData,
                        "vae attn_scales must be an array",
                    )
                })?
                .iter()
                .map(|x| {
                    x.as_f64().map(|f| f as f32).ok_or_else(|| {
                        io::Error::new(
                            io::ErrorKind::InvalidData,
                            "invalid float in vae attn_scales",
                        )
                    })
                })
                .collect::<io::Result<Vec<_>>>()?,
            None => Vec::new(),
        };
        let temperal_downsample = match root.get("temperal_downsample") {
            Some(v) => v
                .as_array()
                .ok_or_else(|| {
                    io::Error::new(
                        io::ErrorKind::InvalidData,
                        "vae temperal_downsample must be an array",
                    )
                })?
                .iter()
                .map(|x| {
                    x.as_bool().ok_or_else(|| {
                        io::Error::new(
                            io::ErrorKind::InvalidData,
                            "invalid bool in vae temperal_downsample",
                        )
                    })
                })
                .collect::<io::Result<Vec<_>>>()?,
            None => vec![false, true, true, false],
        };
        if base_dim == 0
            || decoder_base_dim == 0
            || z_dim == 0
            || dim_mult.contains(&0)
            || temperal_downsample.len() > 8
        {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "vae dimensions must be > 0 and temperal_downsample length <= 8",
            ));
        }
        if dim_mult.len() != temperal_downsample.len() + 1 {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!(
                    "vae dim_mult length ({}) must equal temperal_downsample length + 1 ({})",
                    dim_mult.len(),
                    temperal_downsample.len() + 1
                ),
            ));
        }
        let in_channels = root
            .get("in_channels")
            .and_then(|v| v.as_u64())
            .unwrap_or(4) as usize;
        let out_channels = root
            .get("out_channels")
            .and_then(|v| v.as_u64())
            .unwrap_or(4) as usize;
        if in_channels == 0 || out_channels == 0 {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "vae in_channels and out_channels must be > 0",
            ));
        }
        let is_residual = root
            .get("is_residual")
            .and_then(|v| v.as_bool())
            .unwrap_or(true);
        let parse_latent_stats = |key: &str, default_val: f32| -> io::Result<Vec<f32>> {
            match root.get(key) {
                Some(v) => v
                    .as_array()
                    .ok_or_else(|| {
                        io::Error::new(
                            io::ErrorKind::InvalidData,
                            format!("vae {key} must be an array"),
                        )
                    })?
                    .iter()
                    .map(|x| {
                        let f = x.as_f64().ok_or_else(|| {
                            io::Error::new(
                                io::ErrorKind::InvalidData,
                                format!("invalid float in vae {key}"),
                            )
                        })? as f32;
                        if !f.is_finite() {
                            return Err(io::Error::new(
                                io::ErrorKind::InvalidData,
                                format!("non-finite float in vae {key}"),
                            ));
                        }
                        Ok(f)
                    })
                    .collect(),
                None => Ok(vec![default_val; z_dim]),
            }
        };
        let latents_mean = parse_latent_stats("latents_mean", 0.0)?;
        let latents_std = parse_latent_stats("latents_std", 1.0)?;
        if latents_std.contains(&0.0) {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "vae latents_std entries must be non-zero",
            ));
        }
        if latents_mean.len() != z_dim || latents_std.len() != z_dim {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!(
                    "vae latents_mean ({}) and latents_std ({}) must match z_dim={z_dim}",
                    latents_mean.len(),
                    latents_std.len()
                ),
            ));
        }
        Ok(Self {
            base_dim,
            decoder_base_dim,
            z_dim,
            dim_mult,
            num_res_blocks,
            attn_scales,
            temperal_downsample,
            in_channels,
            out_channels,
            is_residual,
            latents_mean,
            latents_std,
        })
    }
}

/// Configuration for `FlowMatchEulerDiscreteScheduler` (`scheduler/scheduler_config.json`).
#[derive(Clone, Debug, PartialEq)]
pub struct FlowMatchEulerSchedulerConfig {
    pub base_image_seq_len: usize,
    pub base_shift: f32,
    pub invert_sigmas: bool,
    pub max_image_seq_len: usize,
    pub max_shift: f32,
    pub num_train_timesteps: usize,
    pub shift: f32,
    pub shift_terminal: Option<f32>,
    pub stochastic_sampling: bool,
    pub time_shift_type: String,
    pub use_beta_sigmas: bool,
    pub use_dynamic_shifting: bool,
    pub use_exponential_sigmas: bool,
    pub use_karras_sigmas: bool,
}

impl FlowMatchEulerSchedulerConfig {
    pub fn from_dir(dir: &Path) -> io::Result<Self> {
        let text = fs::read_to_string(dir.join("scheduler_config.json"))?;
        let root: Json = serde_json::from_str(&text).map_err(|e| {
            io::Error::new(
                io::ErrorKind::InvalidData,
                format!("bad scheduler/scheduler_config.json: {e}"),
            )
        })?;
        Self::from_json(&root)
    }

    pub fn from_json(root: &Json) -> io::Result<Self> {
        let base_image_seq_len = root
            .get("base_image_seq_len")
            .and_then(|v| v.as_u64())
            .unwrap_or(256) as usize;
        let base_shift = root
            .get("base_shift")
            .and_then(|v| v.as_f64())
            .unwrap_or(0.5) as f32;
        let invert_sigmas = root
            .get("invert_sigmas")
            .and_then(|v| v.as_bool())
            .unwrap_or(false);
        let max_image_seq_len = root
            .get("max_image_seq_len")
            .and_then(|v| v.as_u64())
            .unwrap_or(8192) as usize;
        let max_shift = root
            .get("max_shift")
            .and_then(|v| v.as_f64())
            .unwrap_or(0.9) as f32;
        let num_train_timesteps = root
            .get("num_train_timesteps")
            .and_then(|v| v.as_u64())
            .unwrap_or(1000) as usize;
        let shift = root.get("shift").and_then(|v| v.as_f64()).unwrap_or(1.0) as f32;
        let shift_terminal = root
            .get("shift_terminal")
            .and_then(|v| v.as_f64())
            .map(|f| f as f32);
        let stochastic_sampling = root
            .get("stochastic_sampling")
            .and_then(|v| v.as_bool())
            .unwrap_or(false);
        let time_shift_type = root
            .get("time_shift_type")
            .and_then(|v| v.as_str())
            .unwrap_or("exponential")
            .to_string();
        if time_shift_type != "exponential" && time_shift_type != "linear" {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!(
                    "unsupported scheduler time_shift_type='{}' (expected 'exponential' or 'linear')",
                    sanitize_diag(&time_shift_type)
                ),
            ));
        }
        let use_beta_sigmas = root
            .get("use_beta_sigmas")
            .and_then(|v| v.as_bool())
            .unwrap_or(false);
        let use_dynamic_shifting = root
            .get("use_dynamic_shifting")
            .and_then(|v| v.as_bool())
            .unwrap_or(false);
        let use_exponential_sigmas = root
            .get("use_exponential_sigmas")
            .and_then(|v| v.as_bool())
            .unwrap_or(false);
        let use_karras_sigmas = root
            .get("use_karras_sigmas")
            .and_then(|v| v.as_bool())
            .unwrap_or(false);
        if use_beta_sigmas || use_exponential_sigmas || use_karras_sigmas || stochastic_sampling {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "unsupported non-Euler sigma schedule or stochastic_sampling in FlowMatchEulerDiscreteScheduler",
            ));
        }
        if base_image_seq_len == 0
            || max_image_seq_len <= base_image_seq_len
            || num_train_timesteps == 0
        {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "invalid FlowMatchEulerSchedulerConfig bounds",
            ));
        }
        Ok(Self {
            base_image_seq_len,
            base_shift,
            invert_sigmas,
            max_image_seq_len,
            max_shift,
            num_train_timesteps,
            shift,
            shift_terminal,
            stochastic_sampling,
            time_shift_type,
            use_beta_sigmas,
            use_dynamic_shifting,
            use_exponential_sigmas,
            use_karras_sigmas,
        })
    }

    /// Computes resolution-dependent `mu` matching `calculate_shift` in `pipeline_qwenimage21.py`.
    pub fn calculate_shift(&self, image_seq_len: usize) -> f64 {
        let base_seq = self.base_image_seq_len as f64;
        let max_seq = self.max_image_seq_len as f64;
        let base_shift = self.base_shift as f64;
        let max_shift = self.max_shift as f64;
        let m = (max_shift - base_shift) / (max_seq - base_seq);
        let b = base_shift - m * base_seq;
        (image_seq_len as f64) * m + b
    }

    /// Builds the discrete `(sigmas, timesteps)` schedule for a denoising run, matching
    /// `QwenImage21Pipeline` + `FlowMatchEulerDiscreteScheduler.set_timesteps`.
    pub fn build_schedule(
        &self,
        num_inference_steps: usize,
        image_seq_len: usize,
        custom_sigmas: Option<&[f32]>,
        sample_sigmas: Option<&[f32]>,
    ) -> io::Result<FlowMatchSchedule> {
        if num_inference_steps == 0 && custom_sigmas.is_none() && sample_sigmas.is_none() {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "num_inference_steps must be >= 1",
            ));
        }
        let mut sigmas: Vec<f32> = if let Some(s) = custom_sigmas {
            if s.is_empty() {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidInput,
                    "custom_sigmas must be non-empty",
                ));
            }
            s.to_vec()
        } else if let Some(s) = sample_sigmas {
            if s.is_empty() {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidInput,
                    "sample_sigmas must be non-empty",
                ));
            }
            s.to_vec()
        } else {
            // np.linspace(1.0, 1 / num_inference_steps, num_inference_steps)
            let n = num_inference_steps;
            let start = 1.0f64;
            let end = 1.0f64 / (n as f64);
            if n == 1 {
                vec![start as f32]
            } else {
                let step = (end - start) / ((n - 1) as f64);
                (0..n).map(|i| (start + (i as f64) * step) as f32).collect()
            }
        };

        let mu = self.calculate_shift(image_seq_len);
        if self.use_dynamic_shifting {
            let emu = (mu as f32).exp();
            for s in &mut sigmas {
                if *s >= 1.0 {
                    *s = 1.0;
                } else if *s <= 0.0 {
                    *s = 0.0;
                } else if self.time_shift_type == "linear" {
                    let lin_mu = mu as f32;
                    *s = lin_mu / (lin_mu + (1.0 / *s - 1.0));
                } else {
                    *s = emu / (emu + (1.0 / *s - 1.0));
                }
            }
        } else if (self.shift - 1.0).abs() > f32::EPSILON {
            let sh = self.shift;
            for s in &mut sigmas {
                *s = (sh * *s) / (1.0 + (sh - 1.0) * *s);
            }
        }

        if let Some(term) = self.shift_terminal.filter(|&t| t != 0.0)
            && let Some(&last_s) = sigmas.last()
        {
            let one_minus_last = 1.0f32 - last_s;
            let scale_factor = one_minus_last / (1.0f32 - term);
            if scale_factor.abs() > f32::EPSILON {
                for s in &mut sigmas {
                    let one_minus_z = 1.0f32 - *s;
                    *s = 1.0f32 - (one_minus_z / scale_factor);
                }
            }
        }

        let train_t = self.num_train_timesteps as f32;
        if self.invert_sigmas {
            for s in &mut sigmas {
                *s = 1.0f32 - *s;
            }
        }
        let timesteps: Vec<f32> = sigmas.iter().map(|&s| s * train_t).collect();
        sigmas.push(if self.invert_sigmas { 1.0 } else { 0.0 });

        Ok(FlowMatchSchedule {
            sigmas,
            timesteps,
            mu: mu as f32,
        })
    }
}

/// Resolved per-run `FlowMatchEulerDiscreteScheduler` schedule (`timesteps` and `sigmas`).
#[derive(Clone, Debug, PartialEq)]
pub struct FlowMatchSchedule {
    pub sigmas: Vec<f32>,
    pub timesteps: Vec<f32>,
    pub mu: f32,
}

impl FlowMatchSchedule {
    pub fn num_steps(&self) -> usize {
        self.timesteps.len()
    }

    pub fn dt(&self, step_idx: usize) -> io::Result<f32> {
        let s0 = *self
            .sigmas
            .get(step_idx)
            .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidInput, "step_idx out of bounds"))?;
        let s1 = *self.sigmas.get(step_idx + 1).ok_or_else(|| {
            io::Error::new(io::ErrorKind::InvalidInput, "step_idx+1 out of bounds")
        })?;
        Ok(s1 - s0)
    }

    /// Normalized `[0, 1]` timestep `t / 1000.0` passed into `QwenImage21Transformer2DModel`.
    pub fn normalized_timestep(&self, step_idx: usize) -> io::Result<f32> {
        let t = *self
            .timesteps
            .get(step_idx)
            .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidInput, "step_idx out of bounds"))?;
        Ok(t / 1000.0)
    }

    /// Performs one deterministic Euler step `sample <- sample + dt * model_output` in `FP32`.
    pub fn step_in_place(
        &self,
        step_idx: usize,
        latents: &mut [f32],
        model_output: &[f32],
    ) -> io::Result<()> {
        if latents.len() != model_output.len() {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "latents and model_output length mismatch",
            ));
        }
        let dt = self.dt(step_idx)?;
        for (x, &v) in latents.iter_mut().zip(model_output.iter()) {
            *x = v.mul_add(dt, *x);
        }
        Ok(())
    }
}

/// Applies Classifier-Free Guidance (`neg_pred + true_cfg_scale * (pos_pred - neg_pred)`),
/// matching `QwenImage21Pipeline` (`pipeline_qwenimage21.py`).
pub fn apply_cfg(
    pos_pred: &[f32],
    neg_pred: &[f32],
    true_cfg_scale: f32,
    out: &mut [f32],
) -> io::Result<()> {
    if pos_pred.len() != neg_pred.len() || pos_pred.len() != out.len() {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "apply_cfg slice length mismatch",
        ));
    }
    for ((&p, &n), o) in pos_pred.iter().zip(neg_pred.iter()).zip(out.iter_mut()) {
        *o = n + true_cfg_scale * (p - n);
    }
    Ok(())
}

/// Complete multi-component configuration for a `QwenImage21Pipeline` checkpoint directory.
#[derive(Clone, Debug)]
pub struct QwenImage21PipelineConfig {
    pub root_dir: PathBuf,
    pub class_name: String,
    pub sample_sigmas: Option<Vec<f32>>,
    pub transformer: QwenImage21TransformerConfig,
    pub vae: QwenImage21VaeConfig,
    pub scheduler: FlowMatchEulerSchedulerConfig,
    pub text_encoder: ModelConfig,
    pub processor_dir: PathBuf,
    pub text_encoder_dir: PathBuf,
    pub transformer_dir: PathBuf,
    pub vae_dir: PathBuf,
}

impl QwenImage21PipelineConfig {
    /// Returns `true` when `dir` has a `model_index.json` file (and no root `config.json`).
    pub fn is_pipeline_dir(dir: &Path) -> bool {
        dir.join("model_index.json").is_file() && !dir.join("config.json").is_file()
    }

    /// Loads and validates `model_index.json` and all 5 sub-component configurations under `dir`.
    pub fn from_dir(dir: &Path) -> io::Result<Self> {
        let index_path = dir.join("model_index.json");
        let index_text = fs::read_to_string(&index_path)?;
        let index_json: Json = serde_json::from_str(&index_text).map_err(|e| {
            io::Error::new(
                io::ErrorKind::InvalidData,
                format!("bad model_index.json: {e}"),
            )
        })?;
        let class_name = index_json
            .get("_class_name")
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .to_string();
        if class_name != "QwenImage21Pipeline" {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!(
                    "unsupported pipeline _class_name='{}' (expected 'QwenImage21Pipeline')",
                    sanitize_diag(&class_name)
                ),
            ));
        }
        let sample_sigmas = index_json
            .get("sample_sigmas")
            .and_then(|v| v.as_array())
            .map(|arr| {
                arr.iter()
                    .map(|x| {
                        x.as_f64().map(|f| f as f32).ok_or_else(|| {
                            io::Error::new(
                                io::ErrorKind::InvalidData,
                                "invalid float in model_index.json sample_sigmas",
                            )
                        })
                    })
                    .collect::<io::Result<Vec<f32>>>()
            })
            .transpose()?;

        if sample_sigmas.as_ref().is_some_and(Vec::is_empty) {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "model_index.json sample_sigmas must not be empty",
            ));
        }

        let processor_dir = dir.join("processor");
        let scheduler_dir = dir.join("scheduler");
        let text_encoder_dir = dir.join("text_encoder");
        let transformer_dir = dir.join("transformer");
        let vae_dir = dir.join("vae");

        let transformer = QwenImage21TransformerConfig::from_dir(&transformer_dir)?;
        let vae = QwenImage21VaeConfig::from_dir(&vae_dir)?;
        let scheduler = FlowMatchEulerSchedulerConfig::from_dir(&scheduler_dir)?;
        let text_encoder = ModelConfig::from_dir(&text_encoder_dir)?;
        if !text_encoder.is_qwen3_vl() {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!(
                    "unsupported text_encoder model_type='{}' in QwenImage21Pipeline (expected 'qwen3_vl')",
                    sanitize_diag(&text_encoder.model_type)
                ),
            ));
        }
        if transformer.in_channels != vae.z_dim || transformer.out_channels != vae.z_dim {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!(
                    "transformer channels ({}/{}) do not match vae z_dim ({})",
                    transformer.in_channels, transformer.out_channels, vae.z_dim
                ),
            ));
        }
        if transformer.context_in_dim != text_encoder.hidden_size {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!(
                    "transformer context_in_dim ({}) does not match text_encoder hidden_size ({})",
                    transformer.context_in_dim, text_encoder.hidden_size
                ),
            ));
        }

        Ok(Self {
            root_dir: dir.to_path_buf(),
            class_name,
            sample_sigmas,
            transformer,
            vae,
            scheduler,
            text_encoder,
            processor_dir,
            text_encoder_dir,
            transformer_dir,
            vae_dir,
        })
    }

    pub fn tokenizer_path(&self) -> PathBuf {
        let p = self.processor_dir.join("tokenizer.json");
        if p.is_file() {
            return p;
        }
        let t = self.root_dir.join("tokenizer").join("tokenizer.json");
        if t.is_file() {
            return t;
        }
        self.root_dir.join("tokenizer.json")
    }

    /// Returns the default number of inference steps (`sample_sigmas.len()` for Turbo, `40` for standard).
    pub fn default_steps(&self) -> usize {
        self.sample_sigmas.as_ref().map_or(40, |s| s.len().max(1))
    }

    /// Returns the default classifier-free guidance scale (`1.0` for distilled `sample_sigmas` / Turbo, `4.0` for standard `Qwen-Image-2.1`).
    pub fn default_cfg_scale(&self) -> f32 {
        if self.sample_sigmas.is_some() {
            1.0
        } else {
            4.0
        }
    }

    pub fn open_text_encoder_archive(&self) -> io::Result<SafetensorsArchive> {
        SafetensorsArchive::open(&self.text_encoder_dir)
    }

    pub fn open_transformer_archive(&self) -> io::Result<SafetensorsArchive> {
        SafetensorsArchive::open(&self.transformer_dir)
    }

    pub fn open_vae_archive(&self) -> io::Result<SafetensorsArchive> {
        SafetensorsArchive::open(&self.vae_dir)
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn test_calculate_dimensions_matches_diffusers() {
        assert_eq!(calculate_dimensions(VAE_IMAGE_SIZE, 1.0), (1024, 1024));
        assert_eq!(calculate_dimensions(384 * 384, 1.0), (384, 384));
        let (w, h) = calculate_dimensions(VAE_IMAGE_SIZE, 16.0 / 9.0);
        assert_eq!(w % 32, 0);
        assert_eq!(h % 32, 0);
        assert!(w > h);
    }

    #[test]
    fn test_prompt_templates_match_qwen_image_21() {
        let t2i = format_t2i_prompt("a cozy cabin");
        assert!(t2i.starts_with(
            "<|im_start|>system\nComprehend and analyze the provided prompt.<|im_end|>"
        ));
        assert!(t2i.ends_with("<|im_start|>user\na cozy cabin<|im_end|>\n<|im_start|>assistant\n"));

        let edit = format_ti2i_prompt("turn into sunset", 2);
        assert!(edit.contains("<image1><|vision_start|><|image_pad|><|vision_end|> <image2><|vision_start|><|image_pad|><|vision_end|>turn into sunset"));
    }

    #[test]
    fn test_token_metadata_and_rope_indices_match_diffusers() {
        // 2 text tokens, 1 condition image (1x2x2 = 4 tokens), 1 text token, 1 target image (1x2x4 = 8 tokens)
        let mut mask = vec![false, false];
        mask.extend([true; 4]);
        mask.push(false);
        mask.extend([true; 8]);
        let shapes = [[1, 2, 2], [1, 2, 4]];

        let (image_ids, target_mask) = build_token_metadata(&mask, &shapes).unwrap();
        assert_eq!(
            image_ids,
            vec![-1, -1, 0, 0, 0, 0, -1, 1, 1, 1, 1, 1, 1, 1, 1]
        );
        let prefix_len = target_mask.iter().filter(|&&b| !b).count();
        assert_eq!(prefix_len, 7);
        let segs = prefix_segments(&image_ids, prefix_len);
        assert_eq!(segs, vec![(0, 2, true), (2, 6, false), (6, 7, true)]);

        let [f_idx, h_idx, w_idx] = compute_qwen_image_21_rope_indices(&shapes, &mask).unwrap();
        // Text tokens [0, 1] -> pos 0, 1; cond image (2x2) -> frame=2, h in [-1, 0], w in [-1, 0];
        // position advances by max(2,2)=2 -> next text token at idx 6 has pos=4;
        // target image (2x4) -> frame=5, h in [-1, 0], w in [-2, -1, 0, 1].
        assert_eq!(f_idx, vec![0, 1, 2, 2, 2, 2, 4, 5, 5, 5, 5, 5, 5, 5, 5]);
        assert_eq!(
            h_idx,
            vec![0, 1, -1, -1, 0, 0, 4, -1, -1, -1, -1, 0, 0, 0, 0]
        );
        assert_eq!(
            w_idx,
            vec![0, 1, -1, 0, -1, 0, 4, -2, -1, 0, 1, -2, -1, 0, 1]
        );

        let (cos, sin) =
            compute_qwen_image_21_rope_cos_sin(&shapes, &mask, [16, 56, 56], 10_000.0).unwrap();
        assert_eq!(cos.len(), 960);
        assert_eq!(sin.len(), 960);
        for &(idx, exp_c, exp_s) in &[
            (0usize, 1.0f32, 0.0f32),
            (64, 0.5403023, 0.841471),
            (128, -0.4161468, 0.9092974),
            (136, 0.5403023, -0.841471),
            (448, 0.2836622, -0.9589243),
            (484, -0.4161468, -0.9092974),
            (959, 1.0, 0.0001389),
        ] {
            assert!(
                (cos[idx] - exp_c).abs() < 1e-5,
                "rope.cos[{idx}] mismatch: got {} expected {exp_c}",
                cos[idx]
            );
            assert!(
                (sin[idx] - exp_s).abs() < 1e-5,
                "rope.sin[{idx}] mismatch: got {} expected {exp_s}",
                sin[idx]
            );
        }
        let fixture_path = Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../tools/oracle/fixtures/qwen_image_ops.safetensors");
        if fixture_path.metadata().is_ok_and(|m| m.len() > 1024) {
            let arch = SafetensorsArchive::open(&fixture_path).unwrap();
            let ref_cos = arch.read_f32("rope.cos").unwrap();
            let ref_sin = arch.read_f32("rope.sin").unwrap();
            assert_eq!(cos.len(), ref_cos.len());
            assert_eq!(sin.len(), ref_sin.len());
            for (i, (&c, &rc)) in cos.iter().zip(ref_cos.iter()).enumerate() {
                assert!(
                    (c - rc).abs() < 1e-5,
                    "rope.cos[{i}] mismatch: got {c} expected {rc}"
                );
            }
            for (i, (&s, &rs)) in sin.iter().zip(ref_sin.iter()).enumerate() {
                assert!(
                    (s - rs).abs() < 1e-5,
                    "rope.sin[{i}] mismatch: got {s} expected {rs}"
                );
            }
        }
    }

    #[test]
    fn test_scheduler_matches_diffusers_turbo_and_standard() {
        let turbo_cfg = FlowMatchEulerSchedulerConfig::from_json(&json!({
            "base_image_seq_len": 256,
            "base_shift": 0.5,
            "invert_sigmas": false,
            "max_image_seq_len": 8192,
            "max_shift": 0.9,
            "num_train_timesteps": 1000,
            "shift": 1.0,
            "shift_terminal": null,
            "stochastic_sampling": false,
            "time_shift_type": "exponential",
            "use_beta_sigmas": false,
            "use_dynamic_shifting": false,
            "use_exponential_sigmas": false,
            "use_karras_sigmas": false
        }))
        .unwrap();
        let sample_sigmas = [
            1.0f32, 0.978453, 0.95418, 0.926626, 0.89508, 0.845148, 0.704534, 0.414568,
        ];
        let sched_turbo = turbo_cfg
            .build_schedule(8, 4096, None, Some(&sample_sigmas))
            .unwrap();
        assert_eq!(sched_turbo.num_steps(), 8);
        assert_eq!(sched_turbo.sigmas.len(), 9);
        for (a, b) in sched_turbo.sigmas[..8].iter().zip(sample_sigmas.iter()) {
            assert!((a - b).abs() < 1e-6);
        }
        assert_eq!(sched_turbo.sigmas[8], 0.0);
        assert!((sched_turbo.timesteps[0] - 1000.0).abs() < 1e-4);
        assert!((sched_turbo.timesteps[7] - 414.568).abs() < 1e-3);

        let std_cfg = FlowMatchEulerSchedulerConfig::from_json(&json!({
            "base_image_seq_len": 256,
            "base_shift": 0.5,
            "invert_sigmas": false,
            "max_image_seq_len": 8192,
            "max_shift": 0.9,
            "num_train_timesteps": 1000,
            "shift": 1.0,
            "shift_terminal": 0.02,
            "stochastic_sampling": false,
            "time_shift_type": "exponential",
            "use_beta_sigmas": false,
            "use_dynamic_shifting": true,
            "use_exponential_sigmas": false,
            "use_karras_sigmas": false
        }))
        .unwrap();
        let sched_std = std_cfg.build_schedule(40, 4096, None, None).unwrap();
        let expected_first8 = [
            1.0f32, 0.9869637, 0.973593, 0.9598749, 0.9457955, 0.9313406, 0.9164948, 0.9012421,
        ];
        for (i, &exp) in expected_first8.iter().enumerate() {
            assert!(
                (sched_std.sigmas[i] - exp).abs() < 1e-6,
                "step {i}: got {} expected {exp}",
                sched_std.sigmas[i]
            );
        }
        assert!((sched_std.sigmas[39] - 0.02).abs() < 1e-6);
        assert_eq!(sched_std.sigmas[40], 0.0);
    }

    #[cfg(feature = "cuda")]
    pub(crate) fn write_f16_safetensors_file(
        path: &Path,
        tensors: &[(&str, Vec<usize>, Vec<u16>)],
    ) {
        let mut header = serde_json::Map::new();
        let mut data = Vec::new();
        for (name, shape, vals) in tensors {
            let start = data.len();
            for &v in vals {
                data.extend_from_slice(&v.to_le_bytes());
            }
            let end = data.len();
            header.insert(
                (*name).to_string(),
                json!({
                    "dtype": "F16",
                    "shape": shape,
                    "data_offsets": [start, end],
                }),
            );
        }
        let hdr_bytes = serde_json::to_vec(&serde_json::Value::Object(header)).unwrap();
        let mut out = (hdr_bytes.len() as u64).to_le_bytes().to_vec();
        out.extend_from_slice(&hdr_bytes);
        out.extend_from_slice(&data);
        fs::write(path, out).unwrap();
    }

    /// Writes a miniature, oracle-derived `QwenImage21Pipeline` checkpoint directory (`root`)
    /// that loads and executes full GPU T2I + TI2I + RGBA + CFG generation in ~15 ms.
    #[cfg(feature = "cuda")]
    pub(crate) fn write_synthetic_qwen_image_pipeline_dir(root: &Path) {
        use crate::loader::SafetensorsArchive;

        for sub in [
            "processor",
            "scheduler",
            "text_encoder",
            "transformer",
            "vae",
        ] {
            fs::create_dir_all(root.join(sub)).unwrap();
        }

        let image_pad_id = {
            use tokenizers::AddedToken;
            use tokenizers::models::bpe::{BPE, Vocab};
            let mut vocab_map = Vocab::default();
            vocab_map.insert("[UNK]".to_owned(), 0);
            for (next_id, ch) in (1u32..).zip(
                ['\t', '\n', '\r']
                    .into_iter()
                    .chain((0x20u8..=0x7eu8).map(|b| b as char)),
            ) {
                vocab_map.insert(ch.to_string(), next_id);
            }
            let bpe = BPE::builder()
                .vocab_and_merges(vocab_map, Vec::new())
                .unk_token("[UNK]".to_owned())
                .build()
                .unwrap();
            let mut tok = tokenizers::Tokenizer::new(bpe);
            let added: Vec<AddedToken> = [
                "<|endoftext|>",
                "<|im_start|>",
                "<|im_end|>",
                "<|vision_start|>",
                "<|vision_end|>",
                "<|image_pad|>",
                SYS_PROMPT,
                "system\n",
                "user\n",
                "assistant\n",
                "Picture ",
            ]
            .iter()
            .map(|&s| AddedToken::from(s, true))
            .collect();
            tok.add_special_tokens(&added);
            let pad_id = tok.token_to_id("<|image_pad|>").unwrap() as i32;
            tok.save(root.join("processor/tokenizer.json"), false)
                .unwrap();
            pad_id
        };

        fs::write(
            root.join("model_index.json"),
            serde_json::to_vec(&json!({
                "_class_name": "QwenImage21Pipeline",
                "sample_sigmas": [1.0, 0.5]
            }))
            .unwrap(),
        )
        .unwrap();
        fs::write(
            root.join("scheduler/scheduler_config.json"),
            serde_json::to_vec(&json!({
                "_class_name": "FlowMatchEulerDiscreteScheduler",
                "base_image_seq_len": 256,
                "base_shift": 0.5,
                "invert_sigmas": false,
                "max_image_seq_len": 8192,
                "max_shift": 0.9,
                "num_train_timesteps": 1000,
                "shift": 1.0,
                "shift_terminal": 0.02,
                "stochastic_sampling": false,
                "time_shift_type": "exponential",
                "use_beta_sigmas": false,
                "use_dynamic_shifting": true,
                "use_exponential_sigmas": false,
                "use_karras_sigmas": false
            }))
            .unwrap(),
        )
        .unwrap();
        fs::write(
            root.join("text_encoder/config.json"),
            serde_json::to_vec(&json!({
                "architectures": ["Qwen3VLForConditionalGeneration"],
                "model_type": "qwen3_vl",
                "vision_config": {
                    "depth": 2,
                    "hidden_size": 64,
                    "intermediate_size": 128,
                    "num_heads": 4,
                    "in_channels": 3,
                    "out_hidden_size": 256,
                    "patch_size": 16,
                    "spatial_merge_size": 2,
                    "temporal_patch_size": 2,
                    "num_position_embeddings": 16,
                    "deepstack_visual_indexes": [0, 1],
                    "image_token_id": image_pad_id
                },
                "text_config": {
                    "model_type": "qwen3_vl_text",
                    "vocab_size": 128,
                    "hidden_size": 256,
                    "intermediate_size": 512,
                    "num_hidden_layers": 2,
                    "num_attention_heads": 2,
                    "num_key_value_heads": 1,
                    "head_dim": 128,
                    "rms_norm_eps": 1e-6,
                    "rope_theta": 5000000.0,
                    "rope_scaling": {
                        "mrope_interleaved": true,
                        "mrope_section": [24, 20, 20],
                        "rope_type": "default"
                    }
                }
            }))
            .unwrap(),
        )
        .unwrap();
        fs::write(
            root.join("transformer/config.json"),
            serde_json::to_vec(&json!({
                "_class_name": "QwenImage21Transformer2DModel",
                "patch_size": 1,
                "in_channels": 4,
                "out_channels": 4,
                "num_layers": 1,
                "attention_head_dim": 128,
                "num_attention_heads": 2,
                "context_in_dim": 256,
                "mlp_ratio": 2,
                "axes_dims_rope": [16, 56, 56],
                "causal_condition": true
            }))
            .unwrap(),
        )
        .unwrap();
        fs::write(
            root.join("vae/config.json"),
            serde_json::to_vec(&json!({
                "_class_name": "AutoencoderKLQwenImage21",
                "base_dim": 4,
                "decoder_base_dim": 4,
                "z_dim": 4,
                "dim_mult": [1, 2, 4, 4, 4],
                "num_res_blocks": 1,
                "attn_scales": [],
                "temperal_downsample": [false, false, true, true],
                "in_channels": 4,
                "out_channels": 4,
                "is_residual": true,
                "latents_mean": [0.0, 0.1, -0.1, 0.05],
                "latents_std": [1.0, 1.1, 0.9, 1.05]
            }))
            .unwrap(),
        )
        .unwrap();

        let fixture_path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../../tools/oracle/fixtures/qwen_image_ops.safetensors");
        let arch =
            SafetensorsArchive::open(&fixture_path).expect("open qwen_image_ops.safetensors");
        let mut entries: Vec<(String, Vec<usize>)> = arch
            .iter_entries()
            .map(|(k, e)| (k.to_owned(), e.shape.clone()))
            .collect();
        entries.sort_by(|a, b| a.0.cmp(&b.0));

        // 1. text_encoder/model.safetensors
        let mut vl_tensors = Vec::new();
        for (key, orig_shape) in &entries {
            if key.starts_with("model.") {
                let mut shape = orig_shape.clone();
                let mut bits = arch.read_f16_bits(key).unwrap();
                if key == "model.language_model.embed_tokens.weight" {
                    // Expand [64, 256] -> [128, 256] to cover the synthetic ASCII + special vocab
                    let copy = bits.clone();
                    bits.extend_from_slice(&copy);
                    shape[0] = 128;
                }
                vl_tensors.push((key.as_str(), shape, bits));
            }
        }
        write_f16_safetensors_file(&root.join("text_encoder/model.safetensors"), &vl_tensors);

        // 2. transformer/diffusion_pytorch_model.safetensors
        let mut dit_tensors = Vec::new();
        for (key, orig_shape) in &entries {
            if let Some(stripped) = key.strip_prefix("dit.weight.") {
                let mut shape = orig_shape.clone();
                let mut bits = arch.read_f16_bits(key).unwrap();
                match stripped {
                    "img_in.weight" => {
                        // Slice [256, 8] -> [256, 4] for z_dim = 4
                        let mut sliced = Vec::with_capacity(256 * 4);
                        for row in bits.as_chunks::<8>().0 {
                            sliced.extend_from_slice(&row[..4]);
                        }
                        bits = sliced;
                        shape = vec![256, 4];
                    }
                    "proj_out.weight" => {
                        // Slice [8, 256] -> [4, 256] for z_dim = 4
                        bits.truncate(4 * 256);
                        shape = vec![4, 256];
                    }
                    "txt_in.text_norm.weight" => {
                        // Repeat [64] -> [256] for context_in_dim = 256
                        bits = bits.repeat(4);
                        shape = vec![256];
                    }
                    "txt_in.in_layer.weight" => {
                        // Repeat cols [256, 64] -> [256, 256] for context_in_dim = 256
                        let mut expanded = Vec::with_capacity(256 * 256);
                        for row in bits.as_chunks::<64>().0 {
                            for _ in 0..4 {
                                expanded.extend_from_slice(row);
                            }
                        }
                        bits = expanded;
                        shape = vec![256, 256];
                    }
                    _ => {}
                }
                dit_tensors.push((stripped, shape, bits));
            }
        }
        write_f16_safetensors_file(
            &root.join("transformer/diffusion_pytorch_model.safetensors"),
            &dit_tensors,
        );

        // 3. vae/diffusion_pytorch_model.safetensors
        let mut vae_tensors = Vec::new();
        for (key, orig_shape) in &entries {
            if let Some(stripped) = key.strip_prefix("vae5.weight.") {
                let bits = arch.read_f16_bits(key).unwrap();
                vae_tensors.push((stripped, orig_shape.clone(), bits));
            }
        }
        write_f16_safetensors_file(
            &root.join("vae/diffusion_pytorch_model.safetensors"),
            &vae_tensors,
        );
    }

    #[test]
    fn test_pipeline_config_and_hub_resolve_synthetic_and_cached() {
        let tmp = tempfile::tempdir().unwrap();
        let root = tmp.path();
        for sub in [
            "processor",
            "scheduler",
            "text_encoder",
            "transformer",
            "vae",
        ] {
            fs::create_dir_all(root.join(sub)).unwrap();
        }
        let dummy_st = {
            let header = br#"{"weight":{"dtype":"F16","shape":[1],"data_offsets":[0,2]}}"#;
            let mut bytes = (header.len() as u64).to_le_bytes().to_vec();
            bytes.extend_from_slice(header);
            bytes.extend_from_slice(&[0, 0]);
            bytes
        };
        fs::write(
            root.join("model_index.json"),
            serde_json::to_vec(&json!({
                "_class_name": "QwenImage21Pipeline",
                "sample_sigmas": [1.0, 0.5]
            }))
            .unwrap(),
        )
        .unwrap();
        fs::write(root.join("processor/tokenizer.json"), b"{}").unwrap();
        fs::write(
            root.join("scheduler/scheduler_config.json"),
            serde_json::to_vec(&json!({
                "_class_name": "FlowMatchEulerDiscreteScheduler",
                "base_image_seq_len": 256,
                "max_image_seq_len": 8192,
                "num_train_timesteps": 1000
            }))
            .unwrap(),
        )
        .unwrap();
        fs::write(
            root.join("text_encoder/config.json"),
            serde_json::to_vec(&json!({
                "architectures": ["Qwen3VLForConditionalGeneration"],
                "model_type": "qwen3_vl",
                "text_config": {
                    "model_type": "qwen3_vl_text",
                    "hidden_size": 4096,
                    "intermediate_size": 12288,
                    "num_hidden_layers": 36,
                    "num_attention_heads": 32,
                    "num_key_value_heads": 8,
                    "head_dim": 128,
                    "vocab_size": 151936,
                    "rope_theta": 5000000,
                    "rope_scaling": {
                        "mrope_interleaved": true,
                        "mrope_section": [24, 20, 20],
                        "rope_type": "default"
                    }
                }
            }))
            .unwrap(),
        )
        .unwrap();
        fs::write(root.join("text_encoder/model.safetensors"), &dummy_st).unwrap();
        fs::write(
            root.join("transformer/config.json"),
            serde_json::to_vec(&json!({
                "_class_name": "QwenImage21Transformer2DModel",
                "in_channels": 64,
                "out_channels": 64,
                "num_layers": 32,
                "attention_head_dim": 128,
                "num_attention_heads": 32,
                "context_in_dim": 4096,
                "mlp_ratio": 3,
                "axes_dims_rope": [16, 56, 56],
                "causal_condition": true
            }))
            .unwrap(),
        )
        .unwrap();
        fs::write(
            root.join("transformer/diffusion_pytorch_model.safetensors.index.json"),
            serde_json::to_vec(&json!({
                "weight_map": {
                    "weight": "diffusion_pytorch_model-00001-of-00001.safetensors"
                }
            }))
            .unwrap(),
        )
        .unwrap();
        fs::write(
            root.join("transformer/diffusion_pytorch_model-00001-of-00001.safetensors"),
            &dummy_st,
        )
        .unwrap();
        fs::write(
            root.join("vae/config.json"),
            serde_json::to_vec(&json!({
                "_class_name": "AutoencoderKLQwenImage21",
                "z_dim": 64,
                "in_channels": 4,
                "out_channels": 4
            }))
            .unwrap(),
        )
        .unwrap();
        fs::write(
            root.join("vae/diffusion_pytorch_model.safetensors"),
            &dummy_st,
        )
        .unwrap();

        let pcfg = QwenImage21PipelineConfig::from_dir(root).unwrap();
        assert_eq!(pcfg.sample_sigmas, Some(vec![1.0, 0.5]));
        assert_eq!(pcfg.default_steps(), 2);
        assert_eq!(pcfg.default_cfg_scale(), 1.0);
        assert_eq!(pcfg.text_encoder.rotary_dim, 128);
        assert_eq!(pcfg.text_encoder.mrope_section, [24, 20, 20]);
        assert!(!pcfg.text_encoder.attn_output_gate);
        assert_eq!(pcfg.text_encoder.num_full_attention_layers(), 36);
        assert_eq!(pcfg.text_encoder.num_linear_attention_layers(), 0);
        assert!(pcfg.open_text_encoder_archive().is_ok());
        assert!(pcfg.open_transformer_archive().is_ok());
        assert!(pcfg.open_vae_archive().is_ok());

        let mut std_pcfg = pcfg.clone();
        std_pcfg.sample_sigmas = None;
        assert_eq!(std_pcfg.default_steps(), 40);
        assert_eq!(std_pcfg.default_cfg_scale(), 4.0);

        // If the real Qwen-Image-2.1-Turbo / Qwen-Image-2.1 snapshots are cached locally,
        // verify offline Hub resolution and archive opening on the real checkpoints too.
        for repo in ["Qwen/Qwen-Image-2.1-Turbo", "Qwen/Qwen-Image-2.1"] {
            let opts = crate::hub::HubOptions {
                revision: None,
                offline: true,
            };
            if let Ok(snap_dir) =
                crate::hub::resolve_model(repo, &opts, crate::hub::ModelRole::Target)
            {
                let real_cfg = QwenImage21PipelineConfig::from_dir(&snap_dir).unwrap();
                let te_arch = real_cfg.open_text_encoder_archive().unwrap();
                let tr_arch = real_cfg.open_transformer_archive().unwrap();
                let vae_arch = real_cfg.open_vae_archive().unwrap();
                assert_eq!(tr_arch.iter_entries().count(), 297);
                assert!(te_arch.iter_entries().count() > 300);
                assert!(vae_arch.iter_entries().count() > 100);
            }
        }
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn gpu_qwen3_vl_mrope_and_conditioning_encoder_match_oracle() {
        use crate::cuda::{
            CublasHandle, CudaDevice, CudaStream, DeviceBuffer, check_rc, f16_to_f32_slice,
            f32_to_f16,
        };
        use crate::vision::PreprocessedImage;
        use std::ffi::c_void;
        use std::sync::Arc;
        use warp34_kernels::ffi;

        if CudaDevice::count().unwrap_or(0) <= 0 {
            return;
        }
        let _gpu_guard = crate::LIB_GPU_ENGINE_LOCK
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        let _dev = CudaDevice::set_current(0).unwrap();
        let stream = CudaStream::new(0).unwrap();
        let cublas = CublasHandle::new(&stream).unwrap();

        let fixture_path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../../tools/oracle/fixtures/qwen_image_ops.safetensors");
        let arch = Arc::new(SafetensorsArchive::open(&fixture_path).unwrap());

        // 1. Verify warp34_fused_qk_rmsnorm_rope_f16 with unit norm weights against vl_mrope.q_out
        // Note: apply_qwen3_vl_mrope_interleaved in the oracle applies pure mRoPE without RMSNorm,
        // so we scale unit norm weight per head by the head's RMS so RMSNorm(q) * w == q.
        let pos_bytes = arch.read_bytes("vl_mrope.positions").unwrap();
        let positions: Vec<i32> = pos_bytes
            .as_chunks::<4>()
            .0
            .iter()
            .map(|&c| i32::from_le_bytes(c))
            .collect();
        let q_in_f32 = arch.read_f32("vl_mrope.q_in").unwrap();
        let q_out_oracle = arch.read_f32("vl_mrope.q_out").unwrap();
        let num_tokens = 8usize;
        let num_heads = 4usize;
        let head_dim = 128usize;

        // Normalize each head of q_in to unit RMS before calling the fused kernel with weight = 1.0,
        // then scale back by each head's RMS to isolate exact mRoPE parity.
        let mut q_unit_f16 = vec![0u16; q_in_f32.len()];
        let mut head_rms = vec![0.0f32; num_tokens * num_heads];
        for t in 0..num_tokens {
            for h in 0..num_heads {
                let base = (t * num_heads + h) * head_dim;
                let slice = &q_in_f32[base..base + head_dim];
                let mean_sq = slice.iter().map(|v| v * v).sum::<f32>() / (head_dim as f32);
                let rms = (mean_sq + 1e-6).sqrt();
                head_rms[t * num_heads + h] = rms;
                for d in 0..head_dim {
                    q_unit_f16[base + d] = f32_to_f16(slice[d] / rms);
                }
            }
        }
        let ones_f16 = vec![f32_to_f16(1.0); head_dim];
        let d_q = DeviceBuffer::alloc(0, q_unit_f16.len() * 2).unwrap();
        d_q.copy_from_host(&q_unit_f16, &stream).unwrap();
        let d_k = DeviceBuffer::alloc(0, q_unit_f16.len() * 2).unwrap();
        d_k.copy_from_host(&q_unit_f16, &stream).unwrap();
        let d_w = DeviceBuffer::alloc(0, ones_f16.len() * 2).unwrap();
        d_w.copy_from_host(&ones_f16, &stream).unwrap();
        let d_pos = DeviceBuffer::alloc(0, positions.len() * 4).unwrap();
        d_pos.copy_from_host(&positions, &stream).unwrap();
        let ex = stream.exec(0, 0);
        check_rc(
            // SAFETY: `d_q`, `d_k`, `d_w`, and `d_pos` are valid device allocations matching the launch dimensions.
            unsafe {
                ffi::warp34_fused_qk_rmsnorm_rope_f16(
                    &ex,
                    d_q.addr() as *const c_void,
                    d_k.addr() as *const c_void,
                    d_w.addr() as *const c_void,
                    d_w.addr() as *const c_void,
                    d_pos.addr() as *const i32,
                    core::ptr::null(),
                    d_q.addr() as *mut c_void,
                    d_k.addr() as *mut c_void,
                    num_tokens as i32,
                    num_tokens as i32,
                    num_heads as i32,
                    num_heads as i32,
                    head_dim as i32,
                    head_dim as i32,
                    24,
                    20,
                    20,
                    5_000_000.0,
                    1e-6,
                    0.0,
                )
            },
            "test warp34_fused_qk_rmsnorm_rope_f16",
        )
        .unwrap();
        let mut q_rot_f16 = vec![0u16; q_unit_f16.len()];
        d_q.copy_to_host(&mut q_rot_f16, &stream).unwrap();
        stream.synchronize().unwrap();
        let mut q_rot_f32 = vec![0.0f32; q_rot_f16.len()];
        f16_to_f32_slice(&q_rot_f16, &mut q_rot_f32);
        let mut max_mrope_diff = 0.0f32;
        for t in 0..num_tokens {
            for h in 0..num_heads {
                let rms = head_rms[t * num_heads + h];
                let base = (t * num_heads + h) * head_dim;
                for d in 0..head_dim {
                    let got = q_rot_f32[base + d] * rms;
                    let exp = q_out_oracle[base + d];
                    max_mrope_diff = max_mrope_diff.max((got - exp).abs());
                }
            }
        }
        assert!(
            max_mrope_diff < 5e-3,
            "mRoPE kernel max diff {max_mrope_diff} exceeded 5e-3"
        );

        // 2. Verify Qwen3VisionEncoder::encode_images_with_deepstack and Qwen3VlConditioningEncoder
        let cfg = ModelConfig::from_json(&json!({
            "model_type": "qwen3_vl",
            "image_token_id": 10,
            "vision_config": {
                "depth": 2,
                "hidden_size": 64,
                "intermediate_size": 128,
                "num_heads": 4,
                "in_channels": 3,
                "out_hidden_size": 256,
                "patch_size": 16,
                "spatial_merge_size": 2,
                "temporal_patch_size": 2,
                "num_position_embeddings": 16,
                "deepstack_visual_indexes": [0, 1]
            },
            "text_config": {
                "model_type": "qwen3_vl_text",
                "vocab_size": 64,
                "hidden_size": 256,
                "intermediate_size": 512,
                "num_hidden_layers": 2,
                "num_attention_heads": 2,
                "num_key_value_heads": 1,
                "head_dim": 128,
                "rms_norm_eps": 1e-6,
                "rope_theta": 5000000.0,
                "rope_scaling": {
                    "mrope_interleaved": true,
                    "mrope_section": [24, 20, 20],
                    "rope_type": "default"
                }
            }
        }))
        .unwrap();

        let quant = QuantScheme::Unquantized;
        let encoder =
            Qwen3VlConditioningEncoder::load(Arc::clone(&arch), &cfg, &quant, 0, &stream).unwrap();

        // 2a. Check ViT + deepstack outputs directly
        let mm_pixels = arch.read_f32("vl.mm_pixel_values").unwrap();
        let img = PreprocessedImage {
            patches_f32: mm_pixels.into(),
            grid_thw: [1, 4, 4],
            content_hash: 0,
        };
        let (vis_pooler_buf, ds_bufs) = encoder
            .vision
            .as_ref()
            .unwrap()
            .encode_images_with_deepstack(std::slice::from_ref(&img), &cublas, &stream)
            .unwrap();
        assert_eq!(ds_bufs.len(), 2);

        let check_vis_buf = |buf: &DeviceBuffer, oracle_name: &str, label: &str| {
            let expected = arch.read_f32(oracle_name).unwrap();
            let mut got_f16 = vec![0u16; expected.len()];
            buf.copy_to_host(&mut got_f16, &stream).unwrap();
            stream.synchronize().unwrap();
            let mut got_f32 = vec![0.0f32; expected.len()];
            f16_to_f32_slice(&got_f16, &mut got_f32);
            let max_diff = got_f32
                .iter()
                .zip(expected.iter())
                .map(|(a, b)| (a - b).abs())
                .fold(0.0f32, f32::max);
            assert!(
                max_diff < 5e-3,
                "{label} max abs diff {max_diff} exceeded 5e-3"
            );
        };
        check_vis_buf(&vis_pooler_buf, "vl.mm_vis_pooler", "vis_pooler");
        check_vis_buf(&ds_bufs[0], "vl.mm_vis_deepstack_0", "deepstack_0");
        check_vis_buf(&ds_bufs[1], "vl.mm_vis_deepstack_1", "deepstack_1");

        // 2b. Check text-only conditioning forward pass
        let txt_ids_bytes = arch.read_bytes("vl.txt_input_ids").unwrap();
        let txt_ids: Vec<i32> = txt_ids_bytes
            .as_chunks::<4>()
            .0
            .iter()
            .map(|&c| i32::from_le_bytes(c))
            .collect();
        let txt_expected = arch.read_f32("vl.txt_pre_norm").unwrap();
        let txt_out = encoder
            .encode_conditioning(&txt_ids, &[], 0, &cublas, &stream)
            .unwrap();
        assert_eq!(txt_out.seq_len, txt_ids.len());
        assert_eq!(txt_out.hidden_size, 256);
        let txt_max_diff = txt_out
            .prompt_embeds_f32
            .iter()
            .zip(txt_expected.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        assert!(
            txt_max_diff < 5e-3,
            "Qwen3-VL text-only pre-norm max diff {txt_max_diff} exceeded 5e-3"
        );

        // 2c. Check multimodal conditioning forward pass + drop_idx slicing
        let mm_ids_bytes = arch.read_bytes("vl.mm_input_ids").unwrap();
        let mm_ids: Vec<i32> = mm_ids_bytes
            .as_chunks::<4>()
            .0
            .iter()
            .map(|&c| i32::from_le_bytes(c))
            .collect();
        let mm_expected = arch.read_f32("vl.mm_pre_norm").unwrap();
        let mm_out = encoder
            .encode_conditioning(&mm_ids, std::slice::from_ref(&img), 1, &cublas, &stream)
            .unwrap();
        assert_eq!(mm_out.seq_len, mm_ids.len() - 1);
        assert_eq!(
            mm_out.image_pad_mask,
            vec![false, true, true, true, true, false, false]
        );
        let mm_expected_sliced = &mm_expected[256..];
        let mm_max_diff = mm_out
            .prompt_embeds_f32
            .iter()
            .zip(mm_expected_sliced.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        assert!(
            mm_max_diff < 5e-3,
            "Qwen3-VL multimodal pre-norm max diff {mm_max_diff} exceeded 5e-3"
        );

        // 2d. Check multi-image conditioning forward pass (2 images with distinct grid_thw: [1,4,4] and [1,2,4])
        let mm2_ids_bytes = arch.read_bytes("vl.mm2_input_ids").unwrap();
        let mm2_ids: Vec<i32> = mm2_ids_bytes
            .as_chunks::<4>()
            .0
            .iter()
            .map(|&c| i32::from_le_bytes(c))
            .collect();
        let mm2_px1 = arch.read_f32("vl.mm2_pixel_values_1").unwrap();
        let mm2_px2 = arch.read_f32("vl.mm2_pixel_values_2").unwrap();
        let imgs2 = [
            PreprocessedImage {
                patches_f32: mm2_px1.into(),
                grid_thw: [1, 4, 4],
                content_hash: 1,
            },
            PreprocessedImage {
                patches_f32: mm2_px2.into(),
                grid_thw: [1, 2, 4],
                content_hash: 2,
            },
        ];
        let mm2_expected = arch.read_f32("vl.mm2_pre_norm").unwrap();
        let mm2_out = encoder
            .encode_conditioning(&mm2_ids, &imgs2, 0, &cublas, &stream)
            .unwrap();
        assert_eq!(mm2_out.seq_len, mm2_ids.len());
        assert_eq!(
            mm2_out.image_pad_mask,
            vec![false, true, true, true, true, false, true, true, false]
        );
        let mm2_max_diff = mm2_out
            .prompt_embeds_f32
            .iter()
            .zip(mm2_expected.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        assert!(
            mm2_max_diff < 5e-3,
            "Qwen3-VL multi-image pre-norm max diff {mm2_max_diff} exceeded 5e-3"
        );
    }

    #[test]
    fn test_pack_unpack_latents_and_euler_cfg_step() {
        use crate::cuda::{f16_to_f32, f32_to_f16};

        let chw: Vec<u32> = (0..(4 * 2 * 3)).collect();
        let hwc = pack_latents_chw_to_hwc(&chw, 4, 2, 3);
        let roundtrip = unpack_latents_hwc_to_chw(&hwc, 4, 2, 3);
        assert_eq!(roundtrip, chw);
        // Token 1 (row 0, col 1), channel 2 -> chw[2 * 6 + 1] = 13
        assert_eq!(hwc[4 + 2], 13);

        let mut sample = [f32_to_f16(1.0), f32_to_f16(-0.5)];
        let cond = [f32_to_f16(0.4), f32_to_f16(-0.2)];
        let uncond = [f32_to_f16(0.2), f32_to_f16(-0.6)];
        euler_cfg_step_f16(&mut sample, &cond, Some(&uncond), 2.0, 1.0, 0.5);
        // guided_0 = 0.2 + 2.0 * (0.4 - 0.2) = 0.6 -> 1.0 + (-0.5) * 0.6 = 0.7
        // guided_1 = -0.6 + 2.0 * (-0.2 - (-0.6)) = 0.2 -> -0.5 + (-0.5) * 0.2 = -0.6
        assert!((f16_to_f32(sample[0]) - 0.7).abs() < 2e-3);
        assert!((f16_to_f32(sample[1]) - (-0.6)).abs() < 2e-3);
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn gpu_qwen_image_transformer_extract_cached_graph_and_tp2_match_oracle() {
        use crate::cuda::{
            CublasHandle, CudaDevice, CudaStream, f16_to_f32, f16_to_f32_slice, f32_to_f16,
        };
        use std::ffi::c_void;
        use warp34_kernels::ffi;

        let _gpu_guard = crate::LIB_GPU_ENGINE_LOCK
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        if CudaDevice::count().unwrap_or(0) <= 0 {
            return;
        }
        let _dev = CudaDevice::set_current(0).unwrap();
        let stream = CudaStream::new(0).unwrap();
        let cublas = CublasHandle::new(&stream).unwrap();

        let fixture_path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../../tools/oracle/fixtures/qwen_image_ops.safetensors");
        let arch = SafetensorsArchive::open(&fixture_path).unwrap();

        let cfg = QwenImage21TransformerConfig {
            patch_size: 1,
            in_channels: 8,
            out_channels: 8,
            num_layers: 1,
            attention_head_dim: 128,
            num_attention_heads: 2,
            context_in_dim: 64,
            mlp_ratio: 2,
            axes_dims_rope: [16, 56, 56],
            eps: 1e-6,
            causal_condition: true,
            quant: QuantScheme::Unquantized,
        };
        let quant = QuantScheme::Unquantized;
        let engine = QwenImage21TransformerEngine::load_with_prefix(
            &arch,
            &cfg,
            &quant,
            "dit.weight.",
            0,
            0,
            1,
            &stream,
        )
        .unwrap();

        let enc_hidden = arch.read_f16_bits("dit.enc_hidden").unwrap();
        let cond_latents = arch.read_f16_bits("dit.cond_latents").unwrap();
        let target_latents_0 = arch.read_f16_bits("dit.target_latents_0").unwrap();
        let target_latents_1 = arch.read_f16_bits("dit.target_latents_1").unwrap();
        let t0 = arch.read_f32("dit.t0").unwrap()[0];
        let t1 = arch.read_f32("dit.t1").unwrap()[0];

        let img_shapes = [(1usize, 2usize, 2usize), (1usize, 2usize, 4usize)];
        let vlm_img_mask = [false, false, true, false, true, true];

        let mut kv_cache = engine.new_kv_cache();
        let extract_out = engine
            .forward_extract(
                &cond_latents,
                &target_latents_0,
                &enc_hidden,
                &vlm_img_mask,
                &img_shapes,
                t0,
                Some(&mut kv_cache),
                &cublas,
                &stream,
            )
            .unwrap();
        assert_eq!(extract_out.prefix_len, 7);
        assert_eq!(extract_out.target_tokens, 8);

        let max_diff = |got_f16: &[u16], oracle_name: &str| -> f32 {
            let exp = arch.read_f32(oracle_name).unwrap();
            assert_eq!(got_f16.len(), exp.len(), "length mismatch on {oracle_name}");
            let mut got = vec![0.0f32; got_f16.len()];
            f16_to_f32_slice(got_f16, &mut got);
            got.iter()
                .zip(exp.iter())
                .map(|(a, b)| (a - b).abs())
                .fold(0.0f32, f32::max)
        };

        let diff_full = max_diff(&extract_out.full_output_f16, "dit.out_extract_full");
        assert!(
            diff_full < 5e-3,
            "dit.out_extract_full max diff {diff_full} exceeded 5e-3"
        );
        let diff_tgt0 = max_diff(&extract_out.target_output_f16, "dit.out_extract_target");
        assert!(
            diff_tgt0 < 5e-3,
            "dit.out_extract_target max diff {diff_tgt0} exceeded 5e-3"
        );

        let (cache_k0, cache_v0) = kv_cache.read_prefix_kv(0, &stream).unwrap();
        let diff_ck = max_diff(&cache_k0, "dit.cache_l0_k");
        let diff_cv = max_diff(&cache_v0, "dit.cache_l0_v");
        assert!(
            diff_ck < 5e-3,
            "dit.cache_l0_k max diff {diff_ck} exceeded 5e-3"
        );
        assert!(
            diff_cv < 5e-3,
            "dit.cache_l0_v max diff {diff_cv} exceeded 5e-3"
        );

        // Step 1 (warms graph), Step 2 (captures + launches graph), Step 3 (replays captured graph)
        let mut step0_cached = Vec::new();
        for step_idx in 0..3 {
            let cached_out = engine
                .forward_cached(&target_latents_1, t1, &mut kv_cache, true, &cublas, &stream)
                .unwrap();
            let diff_cached = max_diff(&cached_out, "dit.out_cached_target");
            assert!(
                diff_cached < 5e-3,
                "dit.out_cached_target (step {step_idx}) max diff {diff_cached} exceeded 5e-3"
            );
            if step_idx == 0 {
                step0_cached = cached_out;
            } else {
                // Captured and replayed CUDA graph must be bit-identical to uncaptured execution!
                assert_eq!(
                    cached_out, step0_cached,
                    "CUDA graph replay differed at step {step_idx}"
                );
            }
        }

        // Verify TP=2 sharding (2 ranks of 1 head each) matches the oracle
        let eng_r0 = QwenImage21TransformerEngine::load_with_prefix(
            &arch,
            &cfg,
            &quant,
            "dit.weight.",
            0,
            0,
            2,
            &stream,
        )
        .unwrap();
        let eng_r1 = QwenImage21TransformerEngine::load_with_prefix(
            &arch,
            &cfg,
            &quant,
            "dit.weight.",
            0,
            1,
            2,
            &stream,
        )
        .unwrap();

        // Collect rank-1's partial branch outputs before allreduce, then add them on rank-0 during rank-0's pass
        let mut r1_partials: Vec<Vec<u16>> = Vec::new();
        let _ = eng_r1
            .forward_extract_with_allreduce(
                &cond_latents,
                &target_latents_0,
                &enc_hidden,
                &vlm_img_mask,
                &img_shapes,
                t0,
                None,
                &cublas,
                &stream,
                Some(|d_buf: u64, n_elems: usize, st: &CudaStream| {
                    let mut h = vec![0u16; n_elems];
                    // SAFETY: `h` and `d_buf` both hold `n_elems * 2` bytes.
                    unsafe {
                        ffi::warp34_memcpy_d2h_async(
                            h.as_mut_ptr() as *mut c_void,
                            d_buf as *const c_void,
                            n_elems * 2,
                            st.handle,
                        );
                    }
                    st.synchronize()?;
                    r1_partials.push(h);
                    // Zero rank-1 buffer on first pass since we only need its partial projections
                    Ok(())
                }),
            )
            .unwrap();
        // Actually, because layer 0 MLP depends on the allreduced layer 0 attention output,
        // let's record rank-1 attention partial first, then run rank-0 + rank-1 in lockstep per allreduce!
        // With 1 layer, there are 2 allreduces: [0] = attn.to_out, [1] = img_mlp.out.
        let r1_attn_partial = r1_partials[0].clone();
        let mut r1_mlp_input_recorded: Vec<u16> = Vec::new();
        let mut ar_idx = 0usize;
        let tp2_out = eng_r0
            .forward_extract_with_allreduce(
                &cond_latents,
                &target_latents_0,
                &enc_hidden,
                &vlm_img_mask,
                &img_shapes,
                t0,
                None,
                &cublas,
                &stream,
                Some(|d_buf: u64, n_elems: usize, st: &CudaStream| {
                    let mut h0 = vec![0u16; n_elems];
                    // SAFETY: `h0` and `d_buf` both hold `n_elems * 2` bytes.
                    unsafe {
                        ffi::warp34_memcpy_d2h_async(
                            h0.as_mut_ptr() as *mut c_void,
                            d_buf as *const c_void,
                            n_elems * 2,
                            st.handle,
                        );
                    }
                    st.synchronize()?;
                    if ar_idx == 0 {
                        for (a, &b) in h0.iter_mut().zip(r1_attn_partial.iter()) {
                            *a = f32_to_f16(f16_to_f32(*a) + f16_to_f32(b));
                        }
                        r1_mlp_input_recorded = h0.clone();
                    } else {
                        // Replay rank-1 with the exact allreduced attention output to get rank-1's MLP partial
                        let mut r1_mlp_partial = Vec::new();
                        let mut sub_idx = 0usize;
                        let _ = eng_r1.forward_extract_with_allreduce(
                            &cond_latents,
                            &target_latents_0,
                            &enc_hidden,
                            &vlm_img_mask,
                            &img_shapes,
                            t0,
                            None,
                            &cublas,
                            st,
                            Some(|d_b1: u64, n1: usize, st1: &CudaStream| {
                                if sub_idx == 0 {
                                    // SAFETY: `r1_mlp_input_recorded` and `d_b1` both hold `n1 * 2` bytes.
                                    unsafe {
                                        ffi::warp34_memcpy_h2d_async(
                                            d_b1 as *mut c_void,
                                            r1_mlp_input_recorded.as_ptr() as *const c_void,
                                            n1 * 2,
                                            st1.handle,
                                        );
                                    }
                                    st1.synchronize()?;
                                } else {
                                    r1_mlp_partial.resize(n1, 0u16);
                                    // SAFETY: `r1_mlp_partial` and `d_b1` both hold `n1 * 2` bytes.
                                    unsafe {
                                        ffi::warp34_memcpy_d2h_async(
                                            r1_mlp_partial.as_mut_ptr() as *mut c_void,
                                            d_b1 as *const c_void,
                                            n1 * 2,
                                            st1.handle,
                                        );
                                    }
                                    st1.synchronize()?;
                                }
                                sub_idx += 1;
                                Ok(())
                            }),
                        )?;
                        for (a, &b) in h0.iter_mut().zip(r1_mlp_partial.iter()) {
                            *a = f32_to_f16(f16_to_f32(*a) + f16_to_f32(b));
                        }
                    }
                    // SAFETY: `d_buf` and `h0` both hold `n_elems * 2` bytes.
                    unsafe {
                        ffi::warp34_memcpy_h2d_async(
                            d_buf as *mut c_void,
                            h0.as_ptr() as *const c_void,
                            n_elems * 2,
                            st.handle,
                        );
                    }
                    st.synchronize()?;
                    ar_idx += 1;
                    Ok(())
                }),
            )
            .unwrap();

        let diff_tp2 = max_diff(&tp2_out.full_output_f16, "dit.out_extract_full");
        assert!(
            diff_tp2 < 5e-3,
            "TP=2 dit.out_extract_full max diff {diff_tp2} exceeded 5e-3"
        );

        // If 2 GPUs are present, also verify real 2-GPU hardware TP=2 execution with Tp2Collectives!
        if CudaDevice::count().unwrap_or(0) >= 2 {
            let tp2_engine = QwenImage21TransformerTp2Engine::load_with_prefix(
                &arch,
                &cfg,
                &quant,
                "dit.weight.",
                64,
            )
            .unwrap();
            let (mut c0, mut c1) = tp2_engine.new_kv_caches();
            let hw_extract = tp2_engine
                .forward_extract(
                    &cond_latents,
                    &target_latents_0,
                    &enc_hidden,
                    &vlm_img_mask,
                    &img_shapes,
                    t0,
                    Some((&mut c0, &mut c1)),
                )
                .unwrap();
            let diff_hw_extract = max_diff(&hw_extract.full_output_f16, "dit.out_extract_full");
            assert!(
                diff_hw_extract < 5e-3,
                "2-GPU TP=2 extract max diff {diff_hw_extract} exceeded 5e-3"
            );
            let hw_cached = tp2_engine
                .forward_cached(&target_latents_1, t1, (&mut c0, &mut c1))
                .unwrap();
            let diff_hw_cached = max_diff(&hw_cached, "dit.out_cached_target");
            assert!(
                diff_hw_cached < 5e-3,
                "2-GPU TP=2 cached max diff {diff_hw_cached} exceeded 5e-3"
            );
            // Exercise TP=2 CudaGraph capture (call 2) + replay (call 3), then release_workspaces()
            // and verify monotonic `alloc_epoch` invalidates the freed workspace graph without crash/stale pointers!
            for _ in 0..2 {
                let g_out = tp2_engine
                    .forward_cached(&target_latents_1, t1, (&mut c0, &mut c1))
                    .unwrap();
                assert!(max_diff(&g_out, "dit.out_cached_target") < 5e-3);
            }
            tp2_engine.release_workspaces();
            for _ in 0..3 {
                let post_rel = tp2_engine
                    .forward_cached(&target_latents_1, t1, (&mut c0, &mut c1))
                    .unwrap();
                assert!(max_diff(&post_rel, "dit.out_cached_target") < 5e-3);
            }

            // Also verify full GPU-resident denoise_loop (CFG=2.0, 3 steps, pipelined AllReduce + CUDA Graph)
            // matches single-GPU denoise_loop!
            let timesteps = [t0, t1, t1 * 0.5];
            let sigmas = [1.0f32, 0.7f32, 0.3f32, 0.0f32];
            let mut tp1_cond_cache = engine.new_kv_cache();
            let mut tp1_neg_cache = engine.new_kv_cache();
            let (tp1_loop_out, _, _) = engine
                .denoise_loop(
                    &cond_latents,
                    &target_latents_0,
                    &enc_hidden,
                    &vlm_img_mask,
                    Some((&enc_hidden, &vlm_img_mask)),
                    &img_shapes,
                    &timesteps,
                    &sigmas,
                    2.0,
                    true,
                    &mut tp1_cond_cache,
                    &mut tp1_neg_cache,
                    &cublas,
                    &stream,
                )
                .unwrap();
            let (mut tp2_cond0, mut tp2_cond1) = tp2_engine.new_kv_caches();
            let (mut tp2_neg0, mut tp2_neg1) = tp2_engine.new_kv_caches();
            let (tp2_loop_out, _, _) = tp2_engine
                .denoise_loop(
                    &cond_latents,
                    &target_latents_0,
                    &enc_hidden,
                    &vlm_img_mask,
                    Some((&enc_hidden, &vlm_img_mask)),
                    &img_shapes,
                    &timesteps,
                    &sigmas,
                    2.0,
                    true,
                    (&mut tp2_cond0, &mut tp2_cond1),
                    (&mut tp2_neg0, &mut tp2_neg1),
                )
                .unwrap();
            let loop_diff = tp1_loop_out
                .iter()
                .zip(tp2_loop_out.iter())
                .map(|(&a, &b)| (f16_to_f32(a) - f16_to_f32(b)).abs())
                .fold(0.0f32, f32::max);
            assert!(
                loop_diff < 5e-3,
                "2-GPU TP=2 denoise_loop max diff {loop_diff} vs TP=1 exceeded 5e-3"
            );

            // Multi-image conditioning (2 reference images [(1,2,2), (1,4,2)] + target (1,2,4))
            // + 3-step Euler + normed CFG (true_cfg_scale=4.0) with asymmetric positive (len=6)
            // vs negative (len=4) VLM sequences, verified directly against diffusers oracle!
            let multi_shapes = [
                (1usize, 2usize, 2usize),
                (1usize, 4usize, 2usize),
                (1usize, 2usize, 4usize),
            ];
            let pos_enc = arch.read_f16_bits("dit.multi.pos_enc_hidden").unwrap();
            let neg_enc = arch.read_f16_bits("dit.multi.neg_enc_hidden").unwrap();
            let pos_mask = [false, true, false, true, true, false];
            let neg_mask = [false, true, true, true];
            let multi_cond = arch.read_f16_bits("dit.multi.cond_latents").unwrap();
            let multi_init = arch.read_f16_bits("dit.multi.init_latents").unwrap();
            let cfg_timesteps = [0.978453f32, 0.845148f32, 0.422574f32];
            let cfg_sigmas = [1.0f32, 0.7f32, 0.3f32, 0.0f32];

            let mut m_tp1_cond = engine.new_kv_cache();
            let mut m_tp1_neg = engine.new_kv_cache();
            let m_ext = engine
                .forward_extract(
                    &multi_cond,
                    &multi_init,
                    &pos_enc,
                    &pos_mask,
                    &multi_shapes,
                    cfg_timesteps[0],
                    Some(&mut m_tp1_cond),
                    &cublas,
                    &stream,
                )
                .unwrap();
            let diff_m_ext = max_diff(&m_ext.target_output_f16, "dit.multi.extract_target");
            assert!(
                diff_m_ext < 5e-3,
                "multi-image extract_target max diff {diff_m_ext} exceeded 5e-3"
            );

            let (tp1_multi_final, _, _) = engine
                .denoise_loop(
                    &multi_cond,
                    &multi_init,
                    &pos_enc,
                    &pos_mask,
                    Some((&neg_enc, &neg_mask)),
                    &multi_shapes,
                    &cfg_timesteps,
                    &cfg_sigmas,
                    4.0,
                    true,
                    &mut m_tp1_cond,
                    &mut m_tp1_neg,
                    &cublas,
                    &stream,
                )
                .unwrap();
            let diff_m_tp1 = max_diff(&tp1_multi_final, "dit.multi.final_latents");
            assert!(
                diff_m_tp1 < 1e-2,
                "TP=1 multi-image + asymmetric CFG denoise_loop max diff {diff_m_tp1} vs diffusers exceeded 1e-2"
            );

            let (tp2_multi_final, _, _) = tp2_engine
                .denoise_loop(
                    &multi_cond,
                    &multi_init,
                    &pos_enc,
                    &pos_mask,
                    Some((&neg_enc, &neg_mask)),
                    &multi_shapes,
                    &cfg_timesteps,
                    &cfg_sigmas,
                    4.0,
                    true,
                    (&mut tp2_cond0, &mut tp2_cond1),
                    (&mut tp2_neg0, &mut tp2_neg1),
                )
                .unwrap();
            let diff_m_tp2 = max_diff(&tp2_multi_final, "dit.multi.final_latents");
            assert!(
                diff_m_tp2 < 1e-2,
                "TP=2 multi-image + asymmetric CFG denoise_loop max diff {diff_m_tp2} vs diffusers exceeded 1e-2"
            );
        }
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn gpu_qwen_image_vae_encode_decode_and_tiling_match_oracle() {
        use crate::cuda::{
            CudaDevice, CudaStream, DeviceBuffer, f16_to_f32, f32_to_f16,
            prepare_qwen_image_execution_plan,
        };
        use crate::qwen_image::vae::{
            QwenImage21VaeEngine, QwenImage21VaeTilingConfig, QwenImage21VaeWeightsGpu,
            denormalize_latents_f16, normalize_latents_f16, rgba_u8_to_vae_chw_f16,
            vae_chw_f16_to_rgba_u8,
        };
        use core::ffi::c_void;
        use warp34_kernels::ffi;

        if CudaDevice::count().unwrap_or(0) < 1 {
            return;
        }
        let _gpu_guard = crate::LIB_GPU_ENGINE_LOCK
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        prepare_qwen_image_execution_plan(0).unwrap();
        let stream = CudaStream::new(0).unwrap();
        let ex = stream.exec(0, 0);

        let fixture = Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../tools/oracle/fixtures/qwen_image_ops.safetensors");
        let arch = SafetensorsArchive::open(&fixture).unwrap();
        let load_f16 = |name: &str| -> Vec<u16> { arch.read_f16_bits(name).unwrap() };
        let max_diff = |got: &[u16], exp_name: &str| -> f32 {
            let exp = load_f16(exp_name);
            assert_eq!(got.len(), exp.len(), "length mismatch vs {exp_name}");
            got.iter()
                .zip(exp.iter())
                .map(|(&g, &e)| (f16_to_f32(g) - f16_to_f32(e)).abs())
                .fold(0.0f32, f32::max)
        };

        // 1. Standalone QwenImage21RMS_norm (16 channels, 8x8 = 64 spatial)
        let rms_in = load_f16("vae.rms_in");
        let rms_gamma = load_f16("vae.rms_gamma");
        let d_rms_in = DeviceBuffer::alloc(0, rms_in.len() * 2).unwrap();
        let d_rms_gamma = DeviceBuffer::alloc(0, rms_gamma.len() * 2).unwrap();
        let d_rms_out = DeviceBuffer::alloc(0, rms_in.len() * 2).unwrap();
        // SAFETY: host and device buffers are valid and sized for `rms_in` / `rms_gamma`.
        unsafe {
            ffi::warp34_memcpy_h2d_async(
                d_rms_in.ptr,
                rms_in.as_ptr() as *const c_void,
                rms_in.len() * 2,
                stream.handle,
            );
            ffi::warp34_memcpy_h2d_async(
                d_rms_gamma.ptr,
                rms_gamma.as_ptr() as *const c_void,
                rms_gamma.len() * 2,
                stream.handle,
            );
        }
        // SAFETY: `d_rms_in`, `d_rms_gamma`, and `d_rms_out` are valid device allocations on `ex.stream`.
        let rc = unsafe {
            ffi::warp34_qwen_image_vae_rms_norm_f16(
                &ex,
                d_rms_in.ptr as *const c_void,
                d_rms_gamma.ptr as *const c_void,
                d_rms_out.ptr,
                16,
                64,
                0,
            )
        };
        assert_eq!(rc, 0);
        let mut h_rms_out = vec![0u16; rms_in.len()];
        // SAFETY: `h_rms_out` and `d_rms_out` hold `rms_in.len() * 2` bytes.
        unsafe {
            ffi::warp34_memcpy_d2h_async(
                h_rms_out.as_mut_ptr() as *mut c_void,
                d_rms_out.ptr as *const c_void,
                rms_in.len() * 2,
                stream.handle,
            );
        }
        stream.synchronize().unwrap();
        let diff_rms = max_diff(&h_rms_out, "vae.rms_out");
        assert!(
            diff_rms < 1e-4,
            "vae.rms_out max diff {diff_rms} exceeded 1e-4"
        );

        // 2. Full AutoencoderKLQwenImage21 untiled encode & decode
        let cfg = QwenImage21VaeConfig {
            base_dim: 8,
            decoder_base_dim: 8,
            z_dim: 4,
            dim_mult: vec![1, 2, 2],
            num_res_blocks: 1,
            attn_scales: Vec::new(),
            temperal_downsample: vec![false, true],
            in_channels: 4,
            out_channels: 4,
            is_residual: true,
            latents_mean: (0..4).map(|i| 0.1 * (i as f32)).collect(),
            latents_std: (0..4).map(|i| 1.0 + 0.1 * (i as f32)).collect(),
        };
        let weights = QwenImage21VaeWeightsGpu::from_archive(&arch, &cfg, "vae.weight", 0).unwrap();
        let mut engine = QwenImage21VaeEngine::new(weights).unwrap();

        let img_in = load_f16("vae.img_in");
        let z_mode = engine.encode_mode(&img_in, 16, 16).unwrap();
        let diff_z = max_diff(&z_mode, "vae.z_mode");
        assert!(
            diff_z < 5e-3,
            "untiled vae.z_mode max diff {diff_z} exceeded 5e-3"
        );

        let z_oracle = load_f16("vae.z_mode");
        let dec_out = engine.decode(&z_oracle, 4, 4).unwrap();
        let diff_dec = max_diff(&dec_out, "vae.dec_out");
        assert!(
            diff_dec < 5e-3,
            "untiled vae.dec_out max diff {diff_dec} exceeded 5e-3"
        );

        // 3. Full AutoencoderKLQwenImage21 tiled encode & decode (8x8 tiles, stride 4x4)
        engine.set_tiling(QwenImage21VaeTilingConfig {
            enabled: true,
            tile_sample_min_height: 8,
            tile_sample_min_width: 8,
            tile_sample_stride_height: 4,
            tile_sample_stride_width: 4,
        });
        let tiled_z_mode = engine.encode_mode(&img_in, 16, 16).unwrap();
        let diff_tiled_z = max_diff(&tiled_z_mode, "vae.tiled_z_mode");
        assert!(
            diff_tiled_z < 5e-3,
            "tiled vae.tiled_z_mode max diff {diff_tiled_z} exceeded 5e-3"
        );

        let tiled_dec_out = engine.decode(&z_oracle, 4, 4).unwrap();
        let diff_tiled_dec = max_diff(&tiled_dec_out, "vae.tiled_dec_out");
        assert!(
            diff_tiled_dec < 5e-3,
            "tiled vae.tiled_dec_out max diff {diff_tiled_dec} exceeded 5e-3"
        );

        // 4. Latent normalization/denormalization and RGBA conversion helpers
        let z_norm = normalize_latents_f16(
            &z_oracle,
            cfg.z_dim,
            16,
            &cfg.latents_mean,
            &cfg.latents_std,
        );
        let z_denorm =
            denormalize_latents_f16(&z_norm, cfg.z_dim, 16, &cfg.latents_mean, &cfg.latents_std);
        for (&orig, &rt) in z_oracle.iter().zip(z_denorm.iter()) {
            let err = (f16_to_f32(orig) - f16_to_f32(rt)).abs();
            assert!(err < 5e-3, "latent norm/denorm roundtrip error {err}");
        }

        let rgba_in: Vec<u8> = (0..64).map(|i| (i * 4) as u8).collect();
        let vae_chw = rgba_u8_to_vae_chw_f16(&rgba_in, 4, 4);
        let rgba_rt = vae_chw_f16_to_rgba_u8(&vae_chw, 4, 4, true);
        assert_eq!(rgba_in, rgba_rt);
        let rgb_rt = vae_chw_f16_to_rgba_u8(&vae_chw, 4, 4, false);
        assert_eq!(rgb_rt.len(), 4 * 4 * 3);
        let _ = f32_to_f16(0.0);

        // 5. 5-stage AutoencoderKLQwenImage21 (dim_mult=[1, 2, 4, 4, 4]) high-dynamic-range overflow oracle:
        // Unscaled FP16 overflows past 65,504.0 to Inf/NaN in up_blocks.2 / up_blocks.3;
        // RMSNorm-homogeneity rescaled FP16 stays finite and matches the FP32 oracle.
        let cfg5 = QwenImage21VaeConfig {
            base_dim: 4,
            decoder_base_dim: 4,
            z_dim: 4,
            dim_mult: vec![1, 2, 4, 4, 4],
            num_res_blocks: 1,
            attn_scales: Vec::new(),
            temperal_downsample: vec![false, false, true, true],
            in_channels: 4,
            out_channels: 4,
            is_residual: true,
            latents_mean: vec![0.0; 4],
            latents_std: vec![1.0; 4],
        };
        let weights5 =
            QwenImage21VaeWeightsGpu::from_archive(&arch, &cfg5, "vae5.weight", 0).unwrap();
        let mut engine5 = QwenImage21VaeEngine::new(weights5).unwrap();
        let z5 = load_f16("vae5.z_in");
        let dec5 = engine5.decode(&z5, 2, 2).unwrap();
        assert!(
            dec5.iter().all(|&bits| f16_to_f32(bits).is_finite()),
            "5-stage VAE decode produced non-finite (NaN/Inf) output"
        );
        let diff_dec5 = max_diff(&dec5, "vae5.dec_out");
        assert!(
            diff_dec5 < 1.5e-2,
            "5-stage high-dynamic-range vae5.dec_out max diff {diff_dec5} exceeded 1.5e-2"
        );
    }

    #[test]
    fn parses_w4a16_hgptq_transformer_and_text_encoder_quant_configs() {
        let tf_cfg_json = serde_json::json!({
            "_class_name": "QwenImage21Transformer2DModel",
            "patch_size": 1,
            "in_channels": 64,
            "out_channels": 64,
            "num_layers": 32,
            "attention_head_dim": 128,
            "num_attention_heads": 32,
            "context_in_dim": 4096,
            "mlp_ratio": 3,
            "axes_dims_rope": [16, 56, 56],
            "causal_condition": true,
            "eps": 1e-6,
            "quantization_config": {
                "quant_method": "compressed-tensors",
                "format": "pack-quantized",
                "config_groups": {
                    "group_0": {
                        "targets": ["Linear"],
                        "weights": {
                            "num_bits": 4,
                            "type": "int",
                            "symmetric": true,
                            "strategy": "group",
                            "group_size": 64,
                            "actorder": "static"
                        }
                    }
                },
                "ignore": [
                    "img_in",
                    "txt_in",
                    "proj_out",
                    "norm_out.linear",
                    "time_text_embed.timestep_embedder.linear_1",
                    "time_text_embed.timestep_embedder.linear_2",
                    "re:^transformer_blocks\\.\\d+\\.modulation\\.1$"
                ],
                "transform_config": {
                    "config_groups": {
                        "h64_in": {
                            "type": "hadamard",
                            "head_dim": 64,
                            "randomize": false,
                            "apply": [
                                {
                                    "location": "weight_input",
                                    "targets": ["re:^transformer_blocks\\.\\d+\\.(attn\\.(to_q|to_k|to_v|to_out\\.0)|mlp\\.(gate_up_proj|down_proj))$"],
                                    "inverse": true
                                },
                                {
                                    "location": "input",
                                    "targets": ["re:^transformer_blocks\\.\\d+\\.(attn\\.(to_q|to_k|to_v|to_out\\.0)|mlp\\.(gate_up_proj|down_proj))$"],
                                    "inverse": false
                                }
                            ]
                        }
                    }
                }
            }
        });
        let tf_cfg = QwenImage21TransformerConfig::from_json(&tf_cfg_json).unwrap();
        assert_ne!(tf_cfg.quant, QuantScheme::Unquantized);
        assert_eq!(
            tf_cfg.quant.for_linear("img_in").unwrap(),
            QuantScheme::Unquantized
        );
        assert_eq!(
            tf_cfg
                .quant
                .for_linear("transformer_blocks.0.modulation.1")
                .unwrap(),
            QuantScheme::Unquantized
        );
        assert_ne!(
            tf_cfg
                .quant
                .for_linear("transformer_blocks.0.attn.to_q")
                .unwrap(),
            QuantScheme::Unquantized
        );
        assert_eq!(
            tf_cfg
                .quant
                .input_hadamard("transformer_blocks.0.attn.to_q")
                .unwrap(),
            Some(64)
        );
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn gpu_qwen_image_real_model_matches_diffusers_oracle_suite() {
        use crate::cuda::{CudaDevice, f16_to_f32};

        if std::env::var_os("WARP34_RUN_QWEN_IMAGE_ORACLE_SUITE").is_none() {
            return;
        }
        let _gpu_guard = crate::LIB_GPU_ENGINE_LOCK
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        let num_gpus = CudaDevice::count().unwrap_or(0);
        if num_gpus < 1 {
            return;
        }
        let tp = if num_gpus >= 2 { 2 } else { 1 };
        let golden_dir = PathBuf::from(
            std::env::var("WARP34_QWEN_IMAGE_GOLDEN_DIR")
                .unwrap_or_else(|_| "/tmp/oracle_verify".to_string()),
        );
        if !golden_dir.join("turbo_512.png").is_file() {
            return;
        }
        let opts = crate::hub::HubOptions {
            revision: None,
            offline: true,
        };
        let Ok(turbo_dir) = crate::hub::resolve_model(
            "Qwen/Qwen-Image-2.1-Turbo",
            &opts,
            crate::hub::ModelRole::Target,
        ) else {
            return;
        };
        let mut engine = QwenImage21PipelineEngine::load(&turbo_dir, tp).unwrap();
        let prompt = "A serene alpine lake at golden hour reflecting snow-capped peaks, crisp foreground wildflowers, photorealistic 8k";

        let load_golden_rgb = |name: &str| -> Vec<u8> {
            let bytes = std::fs::read(golden_dir.join(name)).unwrap();
            image::load_from_memory(&bytes)
                .unwrap()
                .to_rgb8()
                .into_raw()
        };
        let load_golden_rgba = |name: &str| -> Vec<u8> {
            let bytes = std::fs::read(golden_dir.join(name)).unwrap();
            image::load_from_memory(&bytes)
                .unwrap()
                .to_rgba8()
                .into_raw()
        };
        let premul_rgba_psnr = |a: &[u8], b: &[u8]| -> f64 {
            let mut sum_sq = 0.0f64;
            for (pa, pb) in a.as_chunks::<4>().0.iter().zip(b.as_chunks::<4>().0.iter()) {
                let aa = (pa[3] as f64) / 255.0;
                let ab = (pb[3] as f64) / 255.0;
                for c in 0..3 {
                    let d = (pa[c] as f64) * aa - (pb[c] as f64) * ab;
                    sum_sq += d * d;
                }
                let da = (pa[3] as f64) - (pb[3] as f64);
                sum_sq += da * da;
            }
            10.0 * ((255.0 * 255.0) / (sum_sq / (a.len() as f64))).log10()
        };

        // 1. T2I 512x512 RGB
        let r1 = engine
            .generate(&QwenImage21GenerationRequest {
                prompt: prompt.to_string(),
                negative_prompt: None,
                condition_images: Vec::new(),
                width: Some(512),
                height: Some(512),
                output_resolution: 1024,
                num_inference_steps: Some(8),
                sigmas: None,
                true_cfg_scale: 1.0,
                seed: 42,
                initial_latents_chw_f16: None,
                rgba_output: false,
                use_cuda_graph: true,
            })
            .unwrap();
        let psnr1 = compute_psnr_u8(&r1.pixels_u8, &load_golden_rgb("turbo_512.png"));
        assert!(psnr1 >= 45.0, "T2I 512x512 PSNR {psnr1:.2} dB < 45.0 dB");

        // 2. T2I 768x512 Non-Square
        let r2 = engine
            .generate(&QwenImage21GenerationRequest {
                prompt: prompt.to_string(),
                negative_prompt: None,
                condition_images: Vec::new(),
                width: Some(768),
                height: Some(512),
                output_resolution: 1024,
                num_inference_steps: Some(8),
                sigmas: None,
                true_cfg_scale: 1.0,
                seed: 42,
                initial_latents_chw_f16: None,
                rgba_output: false,
                use_cuda_graph: true,
            })
            .unwrap();
        let psnr2 = compute_psnr_u8(&r2.pixels_u8, &load_golden_rgb("turbo_768x512.png"));
        assert!(psnr2 >= 45.0, "T2I 768x512 PSNR {psnr2:.2} dB < 45.0 dB");

        // 3. T2I 1024x1024 RGBA Transparent + 4. Single-Image RGBA Edit
        if golden_dir
            .join("diffusers_rgba_transparent_1024.png")
            .is_file()
        {
            let r3 = engine
                .generate(&QwenImage21GenerationRequest {
                    prompt: ensure_rgba_transparency_prompt(
                        "A cute isometric 3D icon of a glowing crystal potion bottle with a cork stopper",
                    ),
                    negative_prompt: None,
                    condition_images: Vec::new(),
                    width: Some(1024),
                    height: Some(1024),
                    output_resolution: 1024,
                    num_inference_steps: Some(8),
                    sigmas: None,
                    true_cfg_scale: 1.0,
                    seed: 42,
                    initial_latents_chw_f16: None,
                    rgba_output: true,
                    use_cuda_graph: true,
                })
                .unwrap();
            assert!(
                r3.decoded_chw_f16
                    .iter()
                    .all(|&b| f16_to_f32(b).is_finite()),
                "1024x1024 RGBA T2I produced non-finite (NaN/Inf) VAE output"
            );
            let trans_px = r3
                .pixels_u8
                .as_chunks::<4>()
                .0
                .iter()
                .filter(|px| px[3] < 10)
                .count();
            assert!(
                trans_px > (1024 * 1024) / 2,
                "expected >50% transparent pixels on 1024x1024 RGBA T2I, got {trans_px}"
            );
            let psnr3 = premul_rgba_psnr(
                &r3.pixels_u8,
                &load_golden_rgba("diffusers_rgba_transparent_1024.png"),
            );
            assert!(
                psnr3 >= 35.0,
                "T2I 1024x1024 RGBA premultiplied PSNR {psnr3:.2} dB < 35.0 dB"
            );

            if golden_dir.join("diffusers_rgba_edit_1024.png").is_file() {
                let cond_rgba = QwenImage21ConditionImage::from_path(
                    &golden_dir.join("diffusers_rgba_transparent_1024.png"),
                )
                .unwrap();
                let r4 = engine
                    .generate(&QwenImage21GenerationRequest {
                        prompt: "Change the glowing potion inside the crystal bottle from purple to bright emerald green and add golden runes etched on the glass".to_string(),
                        negative_prompt: None,
                        condition_images: vec![cond_rgba],
                        width: Some(1024),
                        height: Some(1024),
                        output_resolution: 1024,
                        num_inference_steps: Some(8),
                        sigmas: None,
                        true_cfg_scale: 1.0,
                        seed: 42,
                        initial_latents_chw_f16: None,
                        rgba_output: true,
                        use_cuda_graph: true,
                    })
                    .unwrap();
                assert!(
                    r4.decoded_chw_f16
                        .iter()
                        .all(|&b| f16_to_f32(b).is_finite()),
                    "1024x1024 RGBA Edit produced non-finite (NaN/Inf) VAE output"
                );
                let psnr4 = premul_rgba_psnr(
                    &r4.pixels_u8,
                    &load_golden_rgba("diffusers_rgba_edit_1024.png"),
                );
                assert!(
                    psnr4 >= 38.0,
                    "Edit 1024x1024 RGBA premultiplied PSNR {psnr4:.2} dB < 38.0 dB"
                );
            }
        }

        // 5. Raw 640x480 COCO Single-Image Edit -> 576x448 @ res=512
        let coco_path = Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../tools/oracle/fixtures/coco_sample.png");
        let cond_coco = QwenImage21ConditionImage::from_path(&coco_path).unwrap();
        let r5 = engine
            .generate(&QwenImage21GenerationRequest {
                prompt: "Turn this scene into a watercolor painting".to_string(),
                negative_prompt: None,
                condition_images: vec![cond_coco],
                width: Some(576),
                height: Some(448),
                output_resolution: 512,
                num_inference_steps: Some(8),
                sigmas: None,
                true_cfg_scale: 1.0,
                seed: 42,
                initial_latents_chw_f16: None,
                rgba_output: false,
                use_cuda_graph: true,
            })
            .unwrap();
        let psnr5 = compute_psnr_u8(
            &r5.pixels_u8,
            &load_golden_rgb("diffusers_raw_640x480_edit.png"),
        );
        assert!(
            psnr5 >= 45.0,
            "Raw 640x480 single-image edit PSNR {psnr5:.2} dB < 45.0 dB"
        );

        // 6. Multi-Image Edit ([coco_512x512, coco_576x448] -> 512x512)
        if golden_dir.join("diffusers_512_576_seed123.png").is_file() {
            let c1 =
                QwenImage21ConditionImage::from_path(&golden_dir.join("coco_512x512.png")).unwrap();
            let c2 =
                QwenImage21ConditionImage::from_path(&golden_dir.join("coco_576x448.png")).unwrap();
            let r6 = engine
                .generate(&QwenImage21GenerationRequest {
                    prompt: "Combine lighting from image 2 onto image 1".to_string(),
                    negative_prompt: None,
                    condition_images: vec![c1, c2],
                    width: Some(512),
                    height: Some(512),
                    output_resolution: 512,
                    num_inference_steps: Some(8),
                    sigmas: None,
                    true_cfg_scale: 1.0,
                    seed: 123,
                    initial_latents_chw_f16: None,
                    rgba_output: false,
                    use_cuda_graph: true,
                })
                .unwrap();
            let psnr6 = compute_psnr_u8(
                &r6.pixels_u8,
                &load_golden_rgb("diffusers_512_576_seed123.png"),
            );
            assert!(
                psnr6 >= 45.0,
                "Multi-image edit PSNR {psnr6:.2} dB < 45.0 dB"
            );
        }
    }

    #[test]
    fn test_vae_config_rejects_malformed_arrays_and_zero_std() {
        let bad_mean = json!({
            "_class_name": "AutoencoderKLQwenImage21",
            "z_dim": 2,
            "latents_mean": [0.0, "not_a_float"],
            "latents_std": [1.0, 1.0],
        });
        assert!(QwenImage21VaeConfig::from_json(&bad_mean).is_err());

        let zero_std = json!({
            "_class_name": "AutoencoderKLQwenImage21",
            "z_dim": 2,
            "latents_mean": [0.0, 0.0],
            "latents_std": [1.0, 0.0],
        });
        assert!(QwenImage21VaeConfig::from_json(&zero_std).is_err());

        let zero_dim_mult = json!({
            "_class_name": "AutoencoderKLQwenImage21",
            "dim_mult": [1, 0, 4, 8, 8],
        });
        assert!(QwenImage21VaeConfig::from_json(&zero_dim_mult).is_err());
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn gpu_qwen_image_gated_residual_over_65535_rows_and_pipelined_tp2_allreduce() {
        use crate::cuda::{
            CudaDevice, CudaGraph, CudaStream, DeviceBuffer, Tp2Collectives, check_rc, f16_to_f32,
            f32_to_f16, prepare_qwen_image_execution_plan,
        };
        use std::ffi::c_void;
        use warp34_kernels::ffi;

        if CudaDevice::count().unwrap_or(0) < 1 {
            return;
        }
        let _gpu_guard = crate::LIB_GPU_ENGINE_LOCK
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        prepare_qwen_image_execution_plan(0).unwrap();
        let _dev = CudaDevice::set_current(0).unwrap();
        let stream = CudaStream::new(0).unwrap();
        let ex = stream.exec(0, 0);

        // 1. Test `warp34_qwen_image_gated_residual_f16` at `num_tokens = 65_600` (> 65,535 grid.y cap)
        // across both scalar (`hidden_size = 12`) and `vec8` (`hidden_size = 16`) paths.
        let num_tokens = 65_600usize;
        for hidden_size in [12usize, 16usize] {
            let elems = num_tokens * hidden_size;
            let res_init = vec![f32_to_f16(1.25); elems];
            let branch = vec![f32_to_f16(2.0); elems];
            let mod_row = vec![f32_to_f16(0.5); hidden_size * 2];

            let d_res = DeviceBuffer::alloc(0, elems * 2).unwrap();
            let d_branch = DeviceBuffer::alloc(0, elems * 2).unwrap();
            let d_mod = DeviceBuffer::alloc(0, mod_row.len() * 2).unwrap();
            d_res.copy_from_host(&res_init, &stream).unwrap();
            d_branch.copy_from_host(&branch, &stream).unwrap();
            d_mod.copy_from_host(&mod_row, &stream).unwrap();

            check_rc(
                // SAFETY: `d_res`, `d_branch`, and `d_mod` are live device allocations on device 0
                // sized for `num_tokens * hidden_size` FP16 elements and `hidden_size * 2` modulation cols.
                unsafe {
                    ffi::warp34_qwen_image_gated_residual_f16(
                        &ex,
                        d_res.addr() as *mut c_void,
                        d_branch.addr() as *const c_void,
                        d_mod.addr() as *const c_void,
                        core::ptr::null(),
                        num_tokens as i32,
                        hidden_size as i32,
                        (hidden_size * 2) as i32,
                        hidden_size as i32,
                    )
                },
                "gated_residual >65535 rows",
            )
            .unwrap();

            let mut out = vec![0u16; elems];
            d_res.copy_to_host(&mut out, &stream).unwrap();
            stream.synchronize().unwrap();

            let expected = 1.25f32 + 0.5f32.tanh() * 2.0f32;
            for idx in [
                0usize,
                65_534 * hidden_size,
                (num_tokens - 1) * hidden_size + hidden_size - 1,
            ] {
                let got = f16_to_f32(out[idx]);
                assert!(
                    (got - expected).abs() < 5e-3,
                    "hidden_size={hidden_size} idx={idx}: got {got}, expected {expected}"
                );
            }
        }

        // 2. Test >= 512 KiB `Tp2Collectives::all_reduce_pipelined_f16` (1 MiB = 524,288 FP16 elems)
        // including CUDA Graph capture + replay across 2 GPUs when available.
        if CudaDevice::count().unwrap_or(0) >= 2 {
            prepare_qwen_image_execution_plan(1).unwrap();
            let n_elems = 524_288usize; // 1 MiB -> exercises 8-chunk pipelined DMA + RX kernel
            let comm = std::sync::Arc::new(Tp2Collectives::new_for_tp(2, n_elems * 2).unwrap());
            let in0: Vec<u16> = (0..n_elems)
                .map(|i| f32_to_f16(((i % 31) as f32) * 0.125))
                .collect();
            let in1: Vec<u16> = (0..n_elems)
                .map(|i| f32_to_f16(((i % 17) as f32) * 0.25))
                .collect();

            let comm0 = std::sync::Arc::clone(&comm);
            let comm1 = std::sync::Arc::clone(&comm);
            let (out0, out1) = std::thread::scope(|s| {
                let h1 = s.spawn(move || {
                    let _d1 = CudaDevice::set_current(1).unwrap();
                    let st1 = CudaStream::new(1).unwrap();
                    let d_in1 = DeviceBuffer::alloc(1, n_elems * 2).unwrap();
                    let d_out1 = DeviceBuffer::alloc(1, n_elems * 2).unwrap();
                    d_in1.copy_from_host(&in1, &st1).unwrap();
                    let ex1 = st1.exec(0, 0);
                    comm1
                        .all_reduce_pipelined_f16(
                            &ex1,
                            d_in1.addr(),
                            d_out1.addr(),
                            1,
                            n_elems as i32,
                        )
                        .unwrap();
                    let g1 = CudaGraph::capture(&st1, || {
                        comm1.all_reduce_pipelined_f16(
                            &ex1,
                            d_in1.addr(),
                            d_out1.addr(),
                            1,
                            n_elems as i32,
                        )
                    })
                    .unwrap();
                    g1.launch(&st1).unwrap();
                    let mut host1 = vec![0u16; n_elems];
                    d_out1.copy_to_host(&mut host1, &st1).unwrap();
                    st1.synchronize().unwrap();
                    host1
                });
                let _d0 = CudaDevice::set_current(0).unwrap();
                let st0 = CudaStream::new(0).unwrap();
                let d_in0 = DeviceBuffer::alloc(0, n_elems * 2).unwrap();
                let d_out0 = DeviceBuffer::alloc(0, n_elems * 2).unwrap();
                d_in0.copy_from_host(&in0, &st0).unwrap();
                let ex0 = st0.exec(0, 0);
                comm0
                    .all_reduce_pipelined_f16(&ex0, d_in0.addr(), d_out0.addr(), 0, n_elems as i32)
                    .unwrap();
                let g0 = CudaGraph::capture(&st0, || {
                    comm0.all_reduce_pipelined_f16(
                        &ex0,
                        d_in0.addr(),
                        d_out0.addr(),
                        0,
                        n_elems as i32,
                    )
                })
                .unwrap();
                g0.launch(&st0).unwrap();
                let mut host0 = vec![0u16; n_elems];
                d_out0.copy_to_host(&mut host0, &st0).unwrap();
                st0.synchronize().unwrap();
                (host0, h1.join().unwrap())
            });
            assert_eq!(out0, out1);
            for i in [0usize, 131_071, 262_144, n_elems - 1] {
                let exp = ((i % 31) as f32) * 0.125 + ((i % 17) as f32) * 0.25;
                assert!((f16_to_f32(out0[i]) - exp).abs() < 1e-3);
            }
        }
    }
}
