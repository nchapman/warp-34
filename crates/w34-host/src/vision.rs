//! Phase 5.2: `Qwen3_VisionTransformer` (`model.visual.*`), `smart_resize` + bicubic +
//! normalize preprocessing, 2D bilinear `pos_embed` interpolation, 2D axial RoPE,
//! `PatchMerger`, 3D `mRoPE` position builder, and multimodal `MambaRadixCache` hashing.
#![allow(clippy::too_many_arguments, clippy::undocumented_unsafe_blocks)]

use serde::Deserialize;

#[cfg(feature = "cuda")]
use crate::cuda::{
    CublasHandle, CudaDevice, CudaError, CudaStream, DeviceArena, DeviceBuffer, Result, check_rc,
    f16_to_f32, f32_to_f16,
};
#[cfg(feature = "cuda")]
use crate::loader::SafetensorsArchive;
#[cfg(feature = "cuda")]
use std::ffi::c_void;
#[cfg(feature = "cuda")]
use w34_kernels::ffi;

#[derive(Clone, Debug, Deserialize, PartialEq, Eq)]
pub struct VisionConfig {
    #[serde(default = "default_depth")]
    pub depth: usize,
    #[serde(default = "default_vis_hidden")]
    pub hidden_size: usize,
    #[serde(default = "default_vis_interm")]
    pub intermediate_size: usize,
    #[serde(default = "default_vis_heads")]
    pub num_heads: usize,
    #[serde(default = "default_in_channels")]
    pub in_channels: usize,
    #[serde(default = "default_patch_size")]
    pub patch_size: usize,
    #[serde(default = "default_temporal_patch_size")]
    pub temporal_patch_size: usize,
    #[serde(default = "default_spatial_merge_size")]
    pub spatial_merge_size: usize,
    #[serde(default = "default_num_pos_embeds")]
    pub num_position_embeddings: usize,
    #[serde(default = "default_out_hidden")]
    pub out_hidden_size: usize,
    #[serde(default = "default_hidden_act")]
    pub hidden_act: String,
    #[serde(default)]
    pub deepstack_visual_indexes: Vec<usize>,
    #[serde(default = "default_image_token_id")]
    pub image_token_id: i32,
    #[serde(default = "default_vision_start_token_id")]
    pub vision_start_token_id: i32,
    #[serde(default = "default_vision_end_token_id")]
    pub vision_end_token_id: i32,
}

fn default_depth() -> usize {
    24
}
fn default_vis_hidden() -> usize {
    1024
}
fn default_vis_interm() -> usize {
    4096
}
fn default_vis_heads() -> usize {
    16
}
fn default_in_channels() -> usize {
    3
}
fn default_patch_size() -> usize {
    16
}
fn default_temporal_patch_size() -> usize {
    2
}
fn default_spatial_merge_size() -> usize {
    2
}
fn default_num_pos_embeds() -> usize {
    2304
}
fn default_out_hidden() -> usize {
    2560
}
fn default_hidden_act() -> String {
    "gelu_pytorch_tanh".to_string()
}
pub fn default_image_token_id() -> i32 {
    248056
}
pub fn default_vision_start_token_id() -> i32 {
    248053
}
pub fn default_vision_end_token_id() -> i32 {
    248054
}

impl VisionConfig {
    pub fn head_dim(&self) -> usize {
        self.hidden_size / self.num_heads
    }

    pub fn patch_input_dim(&self) -> usize {
        self.in_channels * self.temporal_patch_size * self.patch_size * self.patch_size
    }

    pub fn num_grid_per_side(&self) -> usize {
        (self.num_position_embeddings as f64).sqrt().round() as usize
    }
}

/// Preprocessed image patches ready for `Qwen3VisionEncoder::encode_images`.
#[derive(Clone, Debug)]
pub struct PreprocessedImage {
    /// Flattened patches `[num_patches, patch_input_dim]` in `2x2` merge-grouped order.
    pub patches_f32: Vec<f32>,
    /// `[t, h_grid, w_grid]` in patch units (`h_grid` and `w_grid` are multiples of `spatial_merge_size`).
    pub grid_thw: [usize; 3],
    /// Deterministic 64-bit content digest of the image for `MambaRadixCache` multimodal keys.
    pub content_hash: u64,
}

impl PreprocessedImage {
    pub fn num_patches(&self) -> usize {
        self.grid_thw[0] * self.grid_thw[1] * self.grid_thw[2]
    }

    pub fn num_merged_tokens(&self, spatial_merge_size: usize) -> usize {
        let m2 = spatial_merge_size * spatial_merge_size;
        self.num_patches() / m2
    }
}

/// `Qwen2VLImageProcessorFast` / `Qwen3VL` `smart_resize`:
/// Rounds `(height, width)` to multiples of `factor = patch_size * spatial_merge_size` (32)
/// while clamping total pixels into `[min_pixels, max_pixels]`.
pub fn smart_resize(
    height: usize,
    width: usize,
    factor: usize,
    min_pixels: usize,
    max_pixels: usize,
) -> (usize, usize) {
    let f = factor.max(1) as f64;
    let round_factor =
        |v: f64| -> usize { ((v / f).round_ties_even() as usize).max(1) * factor.max(1) };
    let mut h_bar = round_factor(height as f64);
    let mut w_bar = round_factor(width as f64);
    if h_bar * w_bar > max_pixels {
        let beta = ((height * width) as f64 / max_pixels as f64).sqrt();
        h_bar = (((height as f64 / beta / f).floor() as usize).max(1)) * factor.max(1);
        w_bar = (((width as f64 / beta / f).floor() as usize).max(1)) * factor.max(1);
    } else if h_bar * w_bar < min_pixels {
        let beta = (min_pixels as f64 / (height * width) as f64).sqrt();
        h_bar = (((height as f64 * beta / f).ceil() as usize).max(1)) * factor.max(1);
        w_bar = (((width as f64 * beta / f).ceil() as usize).max(1)) * factor.max(1);
    }
    (h_bar, w_bar)
}

#[inline]
fn cubic_weight(x: f32) -> f32 {
    // PyTorch / Torchvision bicubic filter (a = -0.75)
    let a = -0.75f32;
    let ax = x.abs();
    if ax <= 1.0 {
        ((a + 2.0) * ax - (a + 3.0)) * ax * ax + 1.0
    } else if ax < 2.0 {
        (((ax - 5.0) * ax + 8.0) * ax - 4.0) * a
    } else {
        0.0
    }
}

fn fnv1a_hash_bytes(bytes: &[u8]) -> u64 {
    let mut h: u64 = 0xcbf29ce484222325;
    for &b in bytes {
        h ^= b as u64;
        h = h.wrapping_mul(0x100000001b3);
    }
    h
}

/// Resizes an `[H, W, 3]` RGB `u8` image via `smart_resize` + bicubic interpolation,
/// normalizes with `mean = [0.5, 0.5, 0.5], std = [0.5, 0.5, 0.5]`, duplicates across
/// `temporal_patch_size = 2`, and extracts `[N_patches, 1536]` in `2x2` merge-grouped order.
pub fn preprocess_rgb_image(
    rgb: &[u8],
    height: usize,
    width: usize,
    patch_size: usize,
    temporal_patch_size: usize,
    spatial_merge_size: usize,
    min_pixels: usize,
    max_pixels: usize,
) -> PreprocessedImage {
    assert!(
        height >= 1 && width >= 1,
        "image dimensions must be non-zero"
    );
    assert_eq!(rgb.len(), height * width * 3, "RGB buffer size mismatch");
    let factor = patch_size * spatial_merge_size;
    let (out_h, out_w) = smart_resize(height, width, factor, min_pixels, max_pixels);

    let mut content_hash = fnv1a_hash_bytes(rgb);
    content_hash ^= ((out_h as u64) << 32) | (out_w as u64);
    content_hash = content_hash.wrapping_mul(0x100000001b3);

    // Bicubic resize + normalize `(val / 255.0 - 0.5) / 0.5 = val / 127.5 - 1.0`
    let mut norm_chw = vec![0.0f32; 3 * out_h * out_w];
    let scale_y = height as f32 / out_h as f32;
    let scale_x = width as f32 / out_w as f32;

    for oy in 0..out_h {
        let sy = (oy as f32 + 0.5) * scale_y - 0.5;
        let iy0 = sy.floor() as isize;
        let wy: [f32; 4] = [
            cubic_weight(sy - (iy0 - 1) as f32),
            cubic_weight(sy - iy0 as f32),
            cubic_weight(sy - (iy0 + 1) as f32),
            cubic_weight(sy - (iy0 + 2) as f32),
        ];
        for ox in 0..out_w {
            let sx = (ox as f32 + 0.5) * scale_x - 0.5;
            let ix0 = sx.floor() as isize;
            let wx: [f32; 4] = [
                cubic_weight(sx - (ix0 - 1) as f32),
                cubic_weight(sx - ix0 as f32),
                cubic_weight(sx - (ix0 + 1) as f32),
                cubic_weight(sx - (ix0 + 2) as f32),
            ];
            for c in 0..3 {
                let mut acc = 0.0f32;
                let mut w_sum = 0.0f32;
                for (dy, &wy_val) in wy.iter().enumerate() {
                    let y = (iy0 - 1 + dy as isize).clamp(0, height as isize - 1) as usize;
                    for (dx, &wx_val) in wx.iter().enumerate() {
                        let x = (ix0 - 1 + dx as isize).clamp(0, width as isize - 1) as usize;
                        let w = wy_val * wx_val;
                        acc += w * (rgb[(y * width + x) * 3 + c] as f32);
                        w_sum += w;
                    }
                }
                let val = (acc / w_sum).clamp(0.0, 255.0);
                norm_chw[(c * out_h + oy) * out_w + ox] = val / 127.5 - 1.0;
            }
        }
    }

    let grid_t = 1usize;
    let grid_h = out_h / patch_size;
    let grid_w = out_w / patch_size;
    let num_patches = grid_t * grid_h * grid_w;
    let patch_dim = 3 * temporal_patch_size * patch_size * patch_size;
    let mut patches = Vec::with_capacity(num_patches * patch_dim);

    // Order patches in (bh, bw, mh, mw) order and layout each patch as [C=3, T=2, ph=16, pw=16]
    let bh_max = grid_h / spatial_merge_size;
    let bw_max = grid_w / spatial_merge_size;
    for bh in 0..bh_max {
        for bw in 0..bw_max {
            for mh in 0..spatial_merge_size {
                for mw in 0..spatial_merge_size {
                    let ph_idx = bh * spatial_merge_size + mh;
                    let pw_idx = bw * spatial_merge_size + mw;
                    for c in 0..3 {
                        for _tp in 0..temporal_patch_size {
                            for py in 0..patch_size {
                                let y = ph_idx * patch_size + py;
                                let row_off = (c * out_h + y) * out_w + pw_idx * patch_size;
                                patches.extend_from_slice(&norm_chw[row_off..row_off + patch_size]);
                            }
                        }
                    }
                }
            }
        }
    }

    PreprocessedImage {
        patches_f32: patches,
        grid_thw: [grid_t, grid_h, grid_w],
        content_hash,
    }
}

fn decode_base64(input: &str) -> Option<Vec<u8>> {
    let mut out = Vec::with_capacity(input.len() * 3 / 4);
    let mut buf = 0u32;
    let mut bits = 0u32;
    for b in input.bytes() {
        let val = match b {
            b'A'..=b'Z' => b - b'A',
            b'a'..=b'z' => b - b'a' + 26,
            b'0'..=b'9' => b - b'0' + 52,
            b'+' | b'-' => 62,
            b'/' | b'_' => 63,
            b'=' | b'\r' | b'\n' | b' ' | b'\t' => continue,
            _ => return None,
        };
        buf = (buf << 6) | (val as u32);
        bits += 6;
        if bits >= 8 {
            bits -= 8;
            out.push(((buf >> bits) & 0xFF) as u8);
        }
    }
    Some(out)
}

fn decode_standard_raster(
    raw_bytes: &[u8],
    fmt: image::ImageFormat,
) -> std::result::Result<(Vec<u8>, usize, usize), String> {
    use image::ImageDecoder;
    // Bound peak decoder workspace independently of HTTP ingress concurrency.
    static DECODE_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());
    let _guard = DECODE_LOCK
        .lock()
        .map_err(|_| "image decoder unavailable")?;
    let mut reader = image::ImageReader::with_format(std::io::Cursor::new(raw_bytes), fmt);
    let mut limits = image::Limits::default();
    limits.max_image_width = Some(16_384);
    limits.max_image_height = Some(16_384);
    limits.max_alloc = Some(128 * 1024 * 1024);
    reader.limits(limits);
    let decoder = reader
        .into_decoder()
        .map_err(|e| format!("invalid {fmt:?} image: {e}"))?;
    let (width, height) = decoder.dimensions();
    if u64::from(width) * u64::from(height) > 16 * 1024 * 1024 {
        return Err("source image exceeds 16 megapixel limit".into());
    }
    let dyn_img = image::DynamicImage::from_decoder(decoder)
        .map_err(|e| format!("failed to decode {fmt:?} image: {e}"))?;
    let rgb = dyn_img.into_rgb8();
    let (w, h) = (rgb.width() as usize, rgb.height() as usize);
    if w == 0 || h == 0 {
        return Err("decoded image has zero width or height".to_string());
    }
    Ok((rgb.into_raw(), h, w))
}

fn decode_ppm_p6(raw_bytes: &[u8]) -> Option<(Vec<u8>, usize, usize)> {
    if !raw_bytes.starts_with(b"P6") {
        return None;
    }
    let mut pos = 2;
    let skip_ws_and_comments = |data: &[u8], p: &mut usize| {
        loop {
            while *p < data.len() && data[*p].is_ascii_whitespace() {
                *p += 1;
            }
            if *p < data.len() && data[*p] == b'#' {
                while *p < data.len() && data[*p] != b'\n' {
                    *p += 1;
                }
            } else {
                break;
            }
        }
    };
    let next_token = |data: &[u8], p: &mut usize| -> Option<usize> {
        skip_ws_and_comments(data, p);
        let start = *p;
        while *p < data.len() && data[*p].is_ascii_digit() {
            *p += 1;
        }
        std::str::from_utf8(&data[start..*p]).ok()?.parse().ok()
    };
    let w = next_token(raw_bytes, &mut pos)?;
    let h = next_token(raw_bytes, &mut pos)?;
    let maxv = next_token(raw_bytes, &mut pos)?;
    if !(1..=16_384).contains(&w) || !(1..=16_384).contains(&h) || maxv != 255 {
        return None;
    }
    if pos < raw_bytes.len() && raw_bytes[pos].is_ascii_whitespace() {
        pos += 1;
    } else {
        return None;
    }
    let needed = w.checked_mul(h)?.checked_mul(3)?;
    let end = pos.checked_add(needed)?;
    if raw_bytes.len() < end {
        return None;
    }
    Some((raw_bytes[pos..end].to_vec(), h, w))
}

/// Decodes PNG, JPEG, or binary PPM (`P6`) bytes into `(rgb_bytes, height, width)`.
/// When `require_known_format` is `false` and neither `mime_hint` nor magic bytes identify
/// PNG/JPEG/PPM, falls back to synthetic RGB bytes for raw test payloads.
pub fn decode_image_bytes_to_rgb(
    raw_bytes: &[u8],
    mime_hint: Option<&str>,
    require_known_format: bool,
) -> std::result::Result<(Vec<u8>, usize, usize), String> {
    if raw_bytes.is_empty() {
        return Err("empty image payload".to_string());
    }
    let mime = mime_hint.unwrap_or("").to_ascii_lowercase();
    if raw_bytes.starts_with(b"\x89PNG\r\n\x1a\n") || mime == "image/png" {
        return decode_standard_raster(raw_bytes, image::ImageFormat::Png);
    }
    if raw_bytes.starts_with(&[0xFF, 0xD8, 0xFF]) || mime == "image/jpeg" || mime == "image/jpg" {
        return decode_standard_raster(raw_bytes, image::ImageFormat::Jpeg);
    }
    if raw_bytes.starts_with(b"P6") || mime == "image/x-portable-pixmap" {
        return decode_ppm_p6(raw_bytes).ok_or_else(|| "invalid binary PPM (P6) image".to_string());
    }
    if require_known_format {
        return Err(
            "unsupported or corrupt image format (expected PNG, JPEG, or binary PPM P6)"
                .to_string(),
        );
    }

    // Fallback for raw non-PNG/JPEG/PPM data-URI test payloads
    let pixels = (raw_bytes.len() / 3).max(1);
    let side = (pixels as f64).sqrt().floor() as usize;
    let (h, w) = if (16..=4096).contains(&side) && side * side * 3 <= raw_bytes.len() {
        (side, side)
    } else {
        (64usize, 64usize)
    };
    let mut rgb = vec![0u8; h * w * 3];
    for (i, px) in rgb.iter_mut().enumerate() {
        *px = raw_bytes[i % raw_bytes.len()];
    }
    Ok((rgb, h, w))
}

/// Parses an OpenAI `image_url` (`data:image/{png,jpeg,x-portable-pixmap};base64,...`) into
/// `(rgb_bytes, height, width)`. Rejects local `file://` URIs, bare filesystem paths, and
/// unrecognized formats. Remote fetching is disabled; `rgb://` is an explicit offline fixture.
pub fn parse_image_url_to_rgb(url: &str) -> std::result::Result<(Vec<u8>, usize, usize), String> {
    parse_image_url_with_policy(url, false)
}

/// Bound aggregate image work before decoding or fetching any parts of a request.
pub const MAX_IMAGES_PER_REQUEST: usize = 4;

/// Fetch public HTTP(S) images only when explicitly enabled by the server operator.
pub fn parse_image_url_with_policy(
    url: &str,
    allow_remote_images: bool,
) -> std::result::Result<(Vec<u8>, usize, usize), String> {
    parse_image_url_with_deadline(
        url,
        allow_remote_images,
        std::time::Instant::now() + crate::remote_images::REMOTE_IMAGE_TIMEOUT,
    )
}

pub(crate) fn parse_image_url_with_deadline(
    url: &str,
    allow_remote_images: bool,
    deadline: std::time::Instant,
) -> std::result::Result<(Vec<u8>, usize, usize), String> {
    if url.starts_with("file://") {
        return Err("file:// URLs are not permitted in image_url".to_string());
    }
    if let Some(idx) = url.find(";base64,") {
        if !url.starts_with("data:") {
            return Err("base64 image_url must start with data:".to_string());
        }
        let mime_hint = &url[5..idx];
        let b64 = &url[idx + 8..];
        if b64.len() as u64 > crate::remote_images::MAX_REMOTE_IMAGE_BYTES.div_ceil(3) * 4 {
            return Err("base64 image exceeds 16 MiB limit".into());
        }
        let raw_bytes =
            decode_base64(b64).ok_or_else(|| "invalid base64 in image_url".to_string())?;
        if raw_bytes.is_empty() {
            return Err("empty decoded payload in image_url".to_string());
        }
        if raw_bytes.len() as u64 > crate::remote_images::MAX_REMOTE_IMAGE_BYTES {
            return Err("decoded image exceeds 16 MiB limit".into());
        }
        decode_image_bytes_to_rgb(&raw_bytes, Some(mime_hint), true)
    } else if url.starts_with("http://") || url.starts_with("https://") {
        if !allow_remote_images {
            return Err("remote images are disabled; enable --allow-remote-images to fetch public HTTP(S) URLs".into());
        }
        let bytes = crate::remote_images::fetch_image_bytes(
            url,
            deadline.saturating_duration_since(std::time::Instant::now()),
        )?;
        decode_image_bytes_to_rgb(&bytes, None, true)
    } else if let Some(seed) = url.strip_prefix("rgb://") {
        // The suffix is an opaque fixture seed, never a nested URL to fetch.
        // This lets migrated oracle fixtures preserve their original pixel bytes.
        let digest = fnv1a_hash_bytes(seed.as_bytes());
        let (h, w) = (64usize, 64usize);
        let mut rgb = vec![0u8; h * w * 3];
        for (i, px) in rgb.iter_mut().enumerate() {
            *px = ((digest.wrapping_add((i as u64).wrapping_mul(0x9E3779B97F4A7C15))) >> 24) as u8;
        }
        Ok((rgb, h, w))
    } else {
        Err("unsupported image_url scheme (expected data:...;base64, or https://)".to_string())
    }
}

/// Loads a CLI `--image <path|url>` argument into `(rgb_bytes, height, width)`.
/// Supports local PNG, JPEG, and PPM (`P6`) files on disk, `data:image/{png,jpeg,x-portable-pixmap};base64,...`
/// data URIs, and offline deterministic `rgb://` fixtures. Remote fetching is disabled.
pub fn load_cli_image_to_rgb(spec: &str) -> std::result::Result<(Vec<u8>, usize, usize), String> {
    load_cli_image_with_policy(spec, false)
}

pub fn load_cli_image_with_policy(
    spec: &str,
    allow_remote_images: bool,
) -> std::result::Result<(Vec<u8>, usize, usize), String> {
    if spec.starts_with("data:")
        || spec.starts_with("http://")
        || spec.starts_with("https://")
        || spec.starts_with("rgb://")
    {
        return parse_image_url_with_policy(spec, allow_remote_images);
    }
    let path_str = spec.strip_prefix("file://").unwrap_or(spec);
    let meta = std::fs::metadata(path_str)
        .map_err(|e| format!("failed to stat image file '{path_str}': {e}"))?;
    if meta.len() > 64 * 1024 * 1024 {
        return Err(format!(
            "image file '{path_str}' exceeds 64 MiB CLI limit ({} bytes)",
            meta.len()
        ));
    }
    let raw_bytes = std::fs::read(path_str)
        .map_err(|e| format!("failed to read image file '{path_str}': {e}"))?;
    decode_image_bytes_to_rgb(&raw_bytes, None, true)
}

/// Computes the 2D bilinearly interpolated `pos_embed` `[total_patches, hidden_size]`
/// and 2D axial RoPE `[total_patches, 2]` `(h_pos, w_pos)` matching `Qwen3_VisionTransformer`
/// (`fast_pos_embed_interpolate` and `rot_pos_ids` in `vllm/model_executor/models/qwen3_vl.py`).
pub fn interpolate_pos_embed_and_rot_ids(
    pos_embed_weight_f32: &[f32],
    num_grid_per_side: usize,
    hidden_size: usize,
    spatial_merge_size: usize,
    grid_thw_list: &[[usize; 3]],
) -> (Vec<f32>, Vec<i32>, Vec<i32>) {
    let num_grid_per_side = num_grid_per_side.max(1);
    let mut total_patches = 0usize;
    let mut cu_seqlens = Vec::with_capacity(grid_thw_list.len() + 1);
    cu_seqlens.push(0i32);
    for &[t, h, w] in grid_thw_list {
        total_patches += t * h * w;
        cu_seqlens.push(total_patches as i32);
    }

    let mut pos_interp = Vec::with_capacity(total_patches * hidden_size);
    let mut pos_hw = Vec::with_capacity(total_patches * 2);

    for &[t, h, w] in grid_thw_list {
        let h_idxs: Vec<f32> = (0..h)
            .map(|i| {
                if h <= 1 {
                    0.0
                } else {
                    (i as f32) * ((num_grid_per_side - 1) as f32) / ((h - 1) as f32)
                }
            })
            .collect();
        let w_idxs: Vec<f32> = (0..w)
            .map(|i| {
                if w <= 1 {
                    0.0
                } else {
                    (i as f32) * ((num_grid_per_side - 1) as f32) / ((w - 1) as f32)
                }
            })
            .collect();

        let bh_max = h / spatial_merge_size;
        let bw_max = w / spatial_merge_size;
        for _frame in 0..t {
            for bh in 0..bh_max {
                for bw in 0..bw_max {
                    for mh in 0..spatial_merge_size {
                        for mw in 0..spatial_merge_size {
                            let ih = bh * spatial_merge_size + mh;
                            let iw = bw * spatial_merge_size + mw;
                            pos_hw.push(ih as i32);
                            pos_hw.push(iw as i32);

                            let hy = h_idxs[ih];
                            let wx = w_idxs[iw];
                            let hf = (hy.floor() as usize).min(num_grid_per_side - 1);
                            let hc = (hf + 1).min(num_grid_per_side - 1);
                            let wf = (wx.floor() as usize).min(num_grid_per_side - 1);
                            let wc = (wf + 1).min(num_grid_per_side - 1);

                            let dh = hy - (hf as f32);
                            let dw = wx - (wf as f32);
                            let w11 = dh * dw;
                            let w10 = dh - w11;
                            let w01 = dw - w11;
                            let w00 = 1.0 - dh - w01;

                            let r00 = &pos_embed_weight_f32
                                [(hf * num_grid_per_side + wf) * hidden_size..];
                            let r01 = &pos_embed_weight_f32
                                [(hf * num_grid_per_side + wc) * hidden_size..];
                            let r10 = &pos_embed_weight_f32
                                [(hc * num_grid_per_side + wf) * hidden_size..];
                            let r11 = &pos_embed_weight_f32
                                [(hc * num_grid_per_side + wc) * hidden_size..];

                            for c in 0..hidden_size {
                                let v = w00 * r00[c] + w01 * r01[c] + w10 * r10[c] + w11 * r11[c];
                                pos_interp.push(v);
                            }
                        }
                    }
                }
            }
        }
    }

    (pos_interp, pos_hw, cu_seqlens)
}

/// Computes `[3, seq_len]` 3D `mRoPE` `(T, H, W)` position IDs and `mrope_position_delta`
/// matching `Qwen3VLForConditionalGeneration._get_mrope_input_positions` in `vLLM`.
pub fn compute_mrope_positions(
    input_ids: &[i32],
    image_grids: &[[usize; 3]],
    image_token_id: i32,
    spatial_merge_size: usize,
) -> ([Vec<i32>; 3], i32) {
    let seq_len = input_ids.len();
    let mut plane_t = Vec::with_capacity(seq_len);
    let mut plane_h = Vec::with_capacity(seq_len);
    let mut plane_w = Vec::with_capacity(seq_len);

    let mut cur_tok = 0usize;
    let mut st_idx = 0i32;

    for &[t_grid, h_grid, w_grid] in image_grids {
        let llm_h = (h_grid / spatial_merge_size).max(1);
        let llm_w = (w_grid / spatial_merge_size).max(1);
        let num_img_tokens = t_grid.max(1) * llm_h * llm_w;

        // Find start of next `<|image_pad|>` run
        let mut img_start = cur_tok;
        while img_start < seq_len && input_ids[img_start] != image_token_id {
            img_start += 1;
        }
        if img_start >= seq_len {
            break;
        }

        // Text segment [cur_tok .. img_start)
        let text_len = img_start - cur_tok;
        for i in 0..text_len {
            let p = st_idx + i as i32;
            plane_t.push(p);
            plane_h.push(p);
            plane_w.push(p);
        }
        st_idx += text_len as i32;

        // Count actual consecutive image_token_id tokens (up to num_img_tokens)
        let mut actual = 0usize;
        while img_start + actual < seq_len
            && input_ids[img_start + actual] == image_token_id
            && actual < num_img_tokens
        {
            actual += 1;
        }

        let mut emitted = 0usize;
        let mut max_coord = st_idx;
        'outer: for ft in 0..t_grid.max(1) {
            for ih in 0..llm_h {
                for iw in 0..llm_w {
                    if emitted >= actual {
                        break 'outer;
                    }
                    let pt = st_idx + ft as i32;
                    let ph = st_idx + ih as i32;
                    let pw = st_idx + iw as i32;
                    plane_t.push(pt);
                    plane_h.push(ph);
                    plane_w.push(pw);
                    max_coord = max_coord.max(pt).max(ph).max(pw);
                    emitted += 1;
                }
            }
        }
        if actual > 0 {
            st_idx = max_coord + 1;
        }
        cur_tok = img_start + actual;
    }

    // Trailing text segment [cur_tok .. seq_len)
    for i in 0..(seq_len - cur_tok) {
        let p = st_idx + i as i32;
        plane_t.push(p);
        plane_h.push(p);
        plane_w.push(p);
    }

    let max_pos = plane_t
        .iter()
        .chain(plane_h.iter())
        .chain(plane_w.iter())
        .copied()
        .max()
        .unwrap_or(-1);
    let mrope_delta = (max_pos + 1) - (seq_len as i32);
    ([plane_t, plane_h, plane_w], mrope_delta)
}

/// Builds a multimodal-aware token sequence key for `MambaRadixCache` by mixing each
/// image's 64-bit `content_hash` into the `<|image_pad|>` token positions so distinct
/// images never alias in the prefix tree while identical images hit the warm radix cache.
pub fn compute_multimodal_radix_key(
    input_ids: &[i32],
    images: &[PreprocessedImage],
    image_token_id: i32,
    spatial_merge_size: usize,
) -> Vec<i32> {
    let mut key = input_ids.to_vec();
    let mut cur_tok = 0usize;
    for img in images {
        let num_vis = img.num_merged_tokens(spatial_merge_size);
        while cur_tok < key.len() && key[cur_tok] != image_token_id {
            cur_tok += 1;
        }
        for v_idx in 0..num_vis {
            if cur_tok >= key.len() || key[cur_tok] != image_token_id {
                break;
            }
            let h = img
                .content_hash
                .wrapping_add((v_idx as u64).wrapping_mul(0x9E3779B97F4A7C15));
            // Negative synthetic token ID in [-2_000_000_000, -1000] to avoid colliding with real vocab IDs
            let hashed_id = -1000 - ((h & 0x3FFF_FFFF) as i32);
            key[cur_tok] = hashed_id;
            cur_tok += 1;
        }
    }
    key
}

#[cfg(feature = "cuda")]
struct VisionBlockGpu {
    norm1_w: u64,
    norm1_b: u64,
    qkv_w: u64,
    qkv_b: u64,
    proj_w: u64,
    proj_b: u64,
    norm2_w: u64,
    norm2_b: u64,
    fc1_w: u64,
    fc1_b: u64,
    fc2_w: u64,
    fc2_b: u64,
}

#[cfg(feature = "cuda")]
pub struct Qwen3VisionEncoder {
    pub cfg: VisionConfig,
    pub device_id: i32,
    pub arena: DeviceArena,
    patch_proj_w: u64,
    patch_proj_b: u64,
    pos_embed_f32: Vec<f32>,
    blocks: Vec<VisionBlockGpu>,
    merger_norm_w: u64,
    merger_norm_b: u64,
    merger_fc1_w: u64,
    merger_fc1_b: u64,
    merger_fc2_w: u64,
    merger_fc2_b: u64,
}

#[cfg(feature = "cuda")]
impl Qwen3VisionEncoder {
    pub fn try_load(
        archive: &SafetensorsArchive,
        cfg: &VisionConfig,
        device_id: i32,
        stream: &CudaStream,
    ) -> Result<Option<Self>> {
        if !cfg.deepstack_visual_indexes.is_empty() {
            return Err(CudaError {
                code: -10,
                context: "non-empty deepstack_visual_indexes not supported in Qwen3_5VisionEncoder",
            });
        }
        let pfx = if archive
            .resolve_name("model.visual.patch_embed.proj.weight")
            .is_some()
        {
            "model.visual"
        } else if archive
            .resolve_name("visual.patch_embed.proj.weight")
            .is_some()
        {
            "visual"
        } else {
            return Ok(None);
        };

        let _dev = CudaDevice::set_current(device_id)?;
        // Estimate weight arena bytes (~1.2 GB max for 27-layer FP16 ViT)
        let est_bytes = 1280 * 1024 * 1024;
        let mut arena = DeviceArena::new(device_id, est_bytes)?;

        let upload_f16 = |arena: &mut DeviceArena, name: &str| -> Result<u64> {
            let key = archive.resolve_name(name).ok_or(CudaError {
                code: -10,
                context: "missing vision tensor",
            })?;
            let bits = archive.read_f16_bits(&key).map_err(|_| CudaError {
                code: -10,
                context: "read vision tensor",
            })?;
            arena.upload_slice(&bits, stream)
        };

        let patch_proj_w = upload_f16(&mut arena, &format!("{pfx}.patch_embed.proj.weight"))?;
        let patch_proj_b = upload_f16(&mut arena, &format!("{pfx}.patch_embed.proj.bias"))?;

        let pos_key = archive
            .resolve_name(&format!("{pfx}.pos_embed.weight"))
            .ok_or(CudaError {
                code: -10,
                context: "missing pos_embed.weight",
            })?;
        let pos_f16 = archive.read_f16_bits(&pos_key).map_err(|_| CudaError {
            code: -10,
            context: "read pos_embed.weight",
        })?;
        let pos_embed_f32: Vec<f32> = pos_f16.iter().copied().map(f16_to_f32).collect();

        let mut blocks = Vec::with_capacity(cfg.depth);
        for i in 0..cfg.depth {
            let bp = format!("{pfx}.blocks.{i}");
            blocks.push(VisionBlockGpu {
                norm1_w: upload_f16(&mut arena, &format!("{bp}.norm1.weight"))?,
                norm1_b: upload_f16(&mut arena, &format!("{bp}.norm1.bias"))?,
                qkv_w: upload_f16(&mut arena, &format!("{bp}.attn.qkv.weight"))?,
                qkv_b: upload_f16(&mut arena, &format!("{bp}.attn.qkv.bias"))?,
                proj_w: upload_f16(&mut arena, &format!("{bp}.attn.proj.weight"))?,
                proj_b: upload_f16(&mut arena, &format!("{bp}.attn.proj.bias"))?,
                norm2_w: upload_f16(&mut arena, &format!("{bp}.norm2.weight"))?,
                norm2_b: upload_f16(&mut arena, &format!("{bp}.norm2.bias"))?,
                fc1_w: upload_f16(&mut arena, &format!("{bp}.mlp.linear_fc1.weight"))?,
                fc1_b: upload_f16(&mut arena, &format!("{bp}.mlp.linear_fc1.bias"))?,
                fc2_w: upload_f16(&mut arena, &format!("{bp}.mlp.linear_fc2.weight"))?,
                fc2_b: upload_f16(&mut arena, &format!("{bp}.mlp.linear_fc2.bias"))?,
            });
        }

        let merger_norm_w = upload_f16(&mut arena, &format!("{pfx}.merger.norm.weight"))?;
        let merger_norm_b = upload_f16(&mut arena, &format!("{pfx}.merger.norm.bias"))?;
        let merger_fc1_w = upload_f16(&mut arena, &format!("{pfx}.merger.linear_fc1.weight"))?;
        let merger_fc1_b = upload_f16(&mut arena, &format!("{pfx}.merger.linear_fc1.bias"))?;
        let merger_fc2_w = upload_f16(&mut arena, &format!("{pfx}.merger.linear_fc2.weight"))?;
        let merger_fc2_b = upload_f16(&mut arena, &format!("{pfx}.merger.linear_fc2.bias"))?;

        stream.synchronize()?;
        let _ = &arena;
        Ok(Some(Self {
            cfg: cfg.clone(),
            device_id,
            arena,
            patch_proj_w,
            patch_proj_b,
            pos_embed_f32,
            blocks,
            merger_norm_w,
            merger_norm_b,
            merger_fc1_w,
            merger_fc1_b,
            merger_fc2_w,
            merger_fc2_b,
        }))
    }

    /// Runs the full `Qwen3_VisionTransformer` forward pass on `images` and returns
    /// `[total_merged_tokens, out_hidden_size]` `FP16` (`u16`) embeddings on GPU.
    pub fn encode_images(
        &self,
        images: &[PreprocessedImage],
        cublas: &CublasHandle,
        stream: &CudaStream,
    ) -> Result<DeviceBuffer> {
        let _dev = CudaDevice::set_current(self.device_id)?;
        let ex = stream.exec(0, 0);

        let patch_dim = self.cfg.patch_input_dim();
        let hidden = self.cfg.hidden_size;
        let interm = self.cfg.intermediate_size;
        let num_heads = self.cfg.num_heads;
        let head_dim = self.cfg.head_dim();
        let merge_unit = self.cfg.spatial_merge_size * self.cfg.spatial_merge_size;
        let merger_hidden = hidden * merge_unit;
        let out_hidden = self.cfg.out_hidden_size;

        let mut grid_thw_list = Vec::with_capacity(images.len());
        let mut total_patches = 0usize;
        for img in images {
            grid_thw_list.push(img.grid_thw);
            total_patches += img.num_patches();
        }
        let total_merged = total_patches / merge_unit;

        let mut patches_f16 = Vec::with_capacity(total_patches * patch_dim);
        for img in images {
            for &v in &img.patches_f32 {
                patches_f16.push(f32_to_f16(v));
            }
        }

        let (pos_interp_f32, pos_hw, cu_seqlens) = interpolate_pos_embed_and_rot_ids(
            &self.pos_embed_f32,
            self.cfg.num_grid_per_side(),
            hidden,
            self.cfg.spatial_merge_size,
            &grid_thw_list,
        );
        let pos_interp_f16: Vec<u16> = pos_interp_f32.iter().copied().map(f32_to_f16).collect();

        let d_patches = DeviceBuffer::alloc(self.device_id, patches_f16.len() * 2)?;
        let d_pos_interp = DeviceBuffer::alloc(self.device_id, pos_interp_f16.len() * 2)?;
        let d_pos_hw = DeviceBuffer::alloc(self.device_id, pos_hw.len() * 4)?;
        let d_cu_seqlens = DeviceBuffer::alloc(self.device_id, cu_seqlens.len() * 4)?;
        d_patches.copy_from_host(&patches_f16, stream)?;
        d_pos_interp.copy_from_host(&pos_interp_f16, stream)?;
        d_pos_hw.copy_from_host(&pos_hw, stream)?;
        d_cu_seqlens.copy_from_host(&cu_seqlens, stream)?;

        let d_x = DeviceBuffer::alloc(self.device_id, total_patches * hidden * 2)?;
        let d_norm = DeviceBuffer::alloc(self.device_id, total_patches * hidden * 2)?;
        let d_qkv = DeviceBuffer::alloc(self.device_id, total_patches * 3 * hidden * 2)?;
        let d_attn_out = DeviceBuffer::alloc(self.device_id, total_patches * hidden * 2)?;
        let d_proj_out = DeviceBuffer::alloc(self.device_id, total_patches * hidden * 2)?;
        let d_mlp_interm = DeviceBuffer::alloc(self.device_id, total_patches * interm * 2)?;

        // 1. patch_embed.proj + bias + pos_embed
        cublas.gemm_f16(
            d_patches.addr(),
            self.patch_proj_w,
            d_x.addr(),
            total_patches as i32,
            hidden as i32,
            patch_dim as i32,
        )?;
        check_rc(
            unsafe {
                ffi::w34_vision_bias_add_f16(
                    &ex,
                    d_x.addr() as *const c_void,
                    self.patch_proj_b as *const c_void,
                    d_pos_interp.addr() as *const c_void,
                    d_x.addr() as *mut c_void,
                    total_patches as i32,
                    hidden as i32,
                )
            },
            "w34_vision_bias_add_f16(patch_embed)",
        )?;

        let use_tanh = i32::from(self.cfg.hidden_act.contains("tanh"));

        // 2. Transformer blocks 0..depth
        for blk in &self.blocks {
            check_rc(
                unsafe {
                    ffi::w34_vision_layernorm_f16(
                        &ex,
                        d_x.addr() as *const c_void,
                        blk.norm1_w as *const c_void,
                        blk.norm1_b as *const c_void,
                        d_norm.addr() as *mut c_void,
                        total_patches as i32,
                        hidden as i32,
                        1e-6,
                    )
                },
                "w34_vision_layernorm_f16(norm1)",
            )?;

            cublas.gemm_f16(
                d_norm.addr(),
                blk.qkv_w,
                d_qkv.addr(),
                total_patches as i32,
                (3 * hidden) as i32,
                hidden as i32,
            )?;

            check_rc(
                unsafe {
                    ffi::w34_vision_qkv_rope_attn_f16(
                        &ex,
                        d_qkv.addr() as *mut c_void,
                        blk.qkv_b as *const c_void,
                        d_pos_hw.addr() as *const i32,
                        d_cu_seqlens.addr() as *const i32,
                        d_attn_out.addr() as *mut c_void,
                        total_patches as i32,
                        images.len() as i32,
                        num_heads as i32,
                        head_dim as i32,
                        10000.0,
                    )
                },
                "w34_vision_qkv_rope_attn_f16",
            )?;

            cublas.gemm_f16(
                d_attn_out.addr(),
                blk.proj_w,
                d_proj_out.addr(),
                total_patches as i32,
                hidden as i32,
                hidden as i32,
            )?;

            check_rc(
                unsafe {
                    ffi::w34_vision_bias_add_f16(
                        &ex,
                        d_proj_out.addr() as *const c_void,
                        blk.proj_b as *const c_void,
                        d_x.addr() as *const c_void,
                        d_x.addr() as *mut c_void,
                        total_patches as i32,
                        hidden as i32,
                    )
                },
                "w34_vision_bias_add_f16(attn.proj)",
            )?;

            check_rc(
                unsafe {
                    ffi::w34_vision_layernorm_f16(
                        &ex,
                        d_x.addr() as *const c_void,
                        blk.norm2_w as *const c_void,
                        blk.norm2_b as *const c_void,
                        d_norm.addr() as *mut c_void,
                        total_patches as i32,
                        hidden as i32,
                        1e-6,
                    )
                },
                "w34_vision_layernorm_f16(norm2)",
            )?;

            cublas.gemm_f16(
                d_norm.addr(),
                blk.fc1_w,
                d_mlp_interm.addr(),
                total_patches as i32,
                interm as i32,
                hidden as i32,
            )?;

            check_rc(
                unsafe {
                    ffi::w34_vision_bias_gelu_f16(
                        &ex,
                        d_mlp_interm.addr() as *const c_void,
                        blk.fc1_b as *const c_void,
                        d_mlp_interm.addr() as *mut c_void,
                        total_patches as i32,
                        interm as i32,
                        use_tanh,
                    )
                },
                "w34_vision_bias_gelu_f16(mlp.fc1)",
            )?;

            cublas.gemm_f16(
                d_mlp_interm.addr(),
                blk.fc2_w,
                d_proj_out.addr(),
                total_patches as i32,
                hidden as i32,
                interm as i32,
            )?;

            check_rc(
                unsafe {
                    ffi::w34_vision_bias_add_f16(
                        &ex,
                        d_proj_out.addr() as *const c_void,
                        blk.fc2_b as *const c_void,
                        d_x.addr() as *const c_void,
                        d_x.addr() as *mut c_void,
                        total_patches as i32,
                        hidden as i32,
                    )
                },
                "w34_vision_bias_add_f16(mlp.fc2)",
            )?;
        }

        // 3. PatchMerger: LayerNorm(hidden_size) -> view [total_merged, 4 * hidden_size] -> fc1 -> exact GELU -> fc2
        check_rc(
            unsafe {
                ffi::w34_vision_layernorm_f16(
                    &ex,
                    d_x.addr() as *const c_void,
                    self.merger_norm_w as *const c_void,
                    self.merger_norm_b as *const c_void,
                    d_norm.addr() as *mut c_void,
                    total_patches as i32,
                    hidden as i32,
                    1e-6,
                )
            },
            "w34_vision_layernorm_f16(merger.norm)",
        )?;

        let d_merger_mid = DeviceBuffer::alloc(self.device_id, total_merged * merger_hidden * 2)?;
        let d_visual_out = DeviceBuffer::alloc(self.device_id, total_merged * out_hidden * 2)?;

        cublas.gemm_f16(
            d_norm.addr(),
            self.merger_fc1_w,
            d_merger_mid.addr(),
            total_merged as i32,
            merger_hidden as i32,
            merger_hidden as i32,
        )?;

        // Note: Qwen3_VisionPatchMerger uses nn.GELU() (exact erf GELU, use_tanh = 0)
        check_rc(
            unsafe {
                ffi::w34_vision_bias_gelu_f16(
                    &ex,
                    d_merger_mid.addr() as *const c_void,
                    self.merger_fc1_b as *const c_void,
                    d_merger_mid.addr() as *mut c_void,
                    total_merged as i32,
                    merger_hidden as i32,
                    0,
                )
            },
            "w34_vision_bias_gelu_f16(merger.fc1)",
        )?;

        cublas.gemm_f16(
            d_merger_mid.addr(),
            self.merger_fc2_w,
            d_visual_out.addr(),
            total_merged as i32,
            out_hidden as i32,
            merger_hidden as i32,
        )?;

        check_rc(
            unsafe {
                ffi::w34_vision_bias_add_f16(
                    &ex,
                    d_visual_out.addr() as *const c_void,
                    self.merger_fc2_b as *const c_void,
                    core::ptr::null(),
                    d_visual_out.addr() as *mut c_void,
                    total_merged as i32,
                    out_hidden as i32,
                )
            },
            "w34_vision_bias_add_f16(merger.fc2)",
        )?;

        Ok(d_visual_out)
    }
}
