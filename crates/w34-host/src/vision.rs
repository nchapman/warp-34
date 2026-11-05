//! `Qwen3_VisionTransformer` (`model.visual.*`), `smart_resize` + bicubic +
//! normalize preprocessing, 2D bilinear `pos_embed` interpolation, 2D axial RoPE,
//! `PatchMerger`, 3D `mRoPE` position builder, and multimodal `MambaRadixCache` hashing.

use serde::Deserialize;
use std::sync::Arc;

#[cfg(feature = "cuda")]
use crate::cuda::{
    CublasHandle, CudaDevice, CudaError, CudaStream, DeviceArena, DeviceBuffer, Result, check_rc,
    f16_to_f32, f32_to_f16,
};
#[cfg(feature = "cuda")]
use crate::loader::SafetensorsArchive;

/// Bounded diagnostic samples keyed by vision stage: shape and host FP32 values.
#[cfg(feature = "cuda")]
pub type VisionTrace = std::collections::BTreeMap<String, (Vec<usize>, Vec<f32>)>;
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
    /// Shared via `Arc<[f32]>` so multi-query fan-out clones are O(1) without duplicating patch memory.
    pub patches_f32: Arc<[f32]>,
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

pub const MAX_IMAGE_ASPECT_RATIO: usize = 200;

/// Validates that an image's aspect ratio does not exceed `MAX_IMAGE_ASPECT_RATIO` (200:1),
/// matching official Qwen2-VL / Qwen3-VL processor bounds.
pub fn validate_image_aspect_ratio(height: usize, width: usize) -> std::result::Result<(), String> {
    let min_dim = height.min(width).max(1);
    let max_dim = height.max(width);
    if max_dim > min_dim * MAX_IMAGE_ASPECT_RATIO {
        return Err(format!(
            "image aspect ratio ({max_dim}:{min_dim}) exceeds maximum allowed ratio of {MAX_IMAGE_ASPECT_RATIO}:1"
        ));
    }
    Ok(())
}

/// `Qwen2VLImageProcessorFast` / `Qwen3VL` `smart_resize`:
/// Rounds `(height, width)` to multiples of `factor = patch_size * spatial_merge_size` (32)
/// while clamping total pixels into `[min_pixels, max_pixels]`.
///
/// # Examples
///
/// ```
/// use w34_host::vision::smart_resize;
///
/// let (h, w) = smart_resize(128, 256, 32, 1024, 1024 * 1024);
/// assert_eq!(h % 32, 0);
/// assert_eq!(w % 32, 0);
/// ```
pub fn smart_resize(
    height: usize,
    width: usize,
    factor: usize,
    min_pixels: usize,
    max_pixels: usize,
) -> (usize, usize) {
    let h_in = height.max(1);
    let w_in = width.max(1);
    // Clamp extreme aspect ratio to MAX_IMAGE_ASPECT_RATIO (200:1) to prevent pathological grid allocations
    let (h_in, w_in) = if h_in > w_in * MAX_IMAGE_ASPECT_RATIO {
        (w_in * MAX_IMAGE_ASPECT_RATIO, w_in)
    } else if w_in > h_in * MAX_IMAGE_ASPECT_RATIO {
        (h_in, h_in * MAX_IMAGE_ASPECT_RATIO)
    } else {
        (h_in, w_in)
    };
    let f = factor.max(1) as f64;
    let round_factor =
        |v: f64| -> usize { ((v / f).round_ties_even() as usize).max(1) * factor.max(1) };
    let mut h_bar = round_factor(h_in as f64);
    let mut w_bar = round_factor(w_in as f64);
    if h_bar * w_bar > max_pixels {
        let beta = ((h_in * w_in) as f64 / max_pixels as f64).sqrt();
        h_bar = (((h_in as f64 / beta / f).floor() as usize).max(1)) * factor.max(1);
        w_bar = (((w_in as f64 / beta / f).floor() as usize).max(1)) * factor.max(1);
    } else if h_bar * w_bar < min_pixels {
        let beta = (min_pixels as f64 / (h_in * w_in) as f64).sqrt();
        h_bar = (((h_in as f64 * beta / f).ceil() as usize).max(1)) * factor.max(1);
        w_bar = (((w_in as f64 * beta / f).ceil() as usize).max(1)) * factor.max(1);
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
#[allow(clippy::too_many_arguments)]
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
        patches_f32: patches.into(),
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
    if fmt == image::ImageFormat::Jpeg {
        return decode_jpeg_rgb(raw_bytes);
    }
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

// libjpeg-turbo's accurate integer IDCT and fancy chroma upsampling match Pillow.
// The v3 API exposes limits missing from the high-level turbojpeg wrapper.
fn decode_jpeg_rgb(raw: &[u8]) -> std::result::Result<(Vec<u8>, usize, usize), String> {
    use turbojpeg_sys::*;
    struct Decoder(tjhandle);
    impl Drop for Decoder {
        fn drop(&mut self) {
            // SAFETY: this owns the non-null handle, destroyed exactly once.
            unsafe { tj3Destroy(self.0) }
        }
    }
    // SAFETY: all calls use a live, exclusively owned decompressor. Input slices
    // remain live for each call. Header dimensions are checked before allocating
    // an RGB buffer of exactly width * height * 3; pitch 0 means packed rows.
    unsafe {
        let handle = tj3Init(TJINIT_TJINIT_DECOMPRESS as i32);
        if handle.is_null() {
            return Err("JPEG decoder unavailable".into());
        }
        let decoder = Decoder(handle);
        let error = || {
            let message = std::ffi::CStr::from_ptr(tj3GetErrorStr(decoder.0));
            format!("invalid JPEG image: {}", message.to_string_lossy())
        };
        for (param, value) in [
            (TJPARAM_TJPARAM_MAXMEMORY, 128),
            (TJPARAM_TJPARAM_MAXPIXELS, 16 * 1024 * 1024),
            (TJPARAM_TJPARAM_SCANLIMIT, 100),
            (TJPARAM_TJPARAM_STOPONWARNING, 1),
            (TJPARAM_TJPARAM_FASTDCT, 0),
            (TJPARAM_TJPARAM_FASTUPSAMPLE, 0),
        ] {
            if tj3Set(decoder.0, param as i32, value) != 0 {
                return Err(error());
            }
        }
        if tj3DecompressHeader(decoder.0, raw.as_ptr(), raw.len() as _) != 0 {
            return Err(error());
        }
        let w = tj3Get(decoder.0, TJPARAM_TJPARAM_JPEGWIDTH as i32);
        let h = tj3Get(decoder.0, TJPARAM_TJPARAM_JPEGHEIGHT as i32);
        if !(1..=16_384).contains(&w) || !(1..=16_384).contains(&h) {
            return Err("invalid JPEG dimensions (maximum 16384 per axis)".into());
        }
        let (w, h) = (w as usize, h as usize);
        if w * h > 16 * 1024 * 1024 {
            return Err("source image exceeds 16 megapixel limit".into());
        }
        let mut rgb = vec![0; w * h * 3];
        if tj3Decompress8(
            decoder.0,
            raw.as_ptr(),
            raw.len() as _,
            rgb.as_mut_ptr(),
            0,
            TJPF_TJPF_RGB,
        ) != 0
        {
            return Err(error());
        }
        Ok((rgb, h, w))
    }
}

fn decode_ppm_p6(raw_bytes: &[u8]) -> Option<(Vec<u8>, usize, usize)> {
    if !raw_bytes.starts_with(b"P6") {
        return None;
    }
    let mut pos = 2;
    let skip_ws_and_comments = |data: &[u8], p: &mut usize| loop {
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
    // Local files retain their 64 MiB allowance; HTTP ingress separately caps
    // both base64 and fetched payloads at 16 MiB before reaching this decoder.
    if raw_bytes.len() > 64 * 1024 * 1024 {
        return Err("image exceeds 64 MiB input limit".into());
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

/// Single-rounded FP16 FMA for operands already representable as FP16.
#[inline]
fn fused_half_mul_add(a: f32, b: f32, c: f32) -> f32 {
    // FP64 preserves the information needed to round finite half operands to FP16.
    // Round-to-odd at the intermediate FP32 boundary preserves the sticky bit:
    // ordinary FP32 rounding can manufacture a half-way tie before FP16 rounding.
    let exact = (a as f64) * (b as f64) + c as f64;
    let rounded = exact as f32;
    let sticky = if exact.is_finite() && f64::from(rounded) != exact {
        let bits = rounded.to_bits();
        let odd_bits = if f64::from(rounded).abs() < exact.abs() {
            bits | 1
        } else {
            (bits - 1) | 1
        };
        f32::from_bits(odd_bits)
    } else {
        rounded
    };
    crate::cuda::f16_to_f32(crate::cuda::f32_to_f16(sticky))
}

/// Computes the 2D bilinearly interpolated `pos_embed` `[total_patches, hidden_size]`
/// and 2D axial RoPE `[total_patches, 2]` `(h_pos, w_pos)` matching `Qwen3_VisionTransformer`
/// (`fast_pos_embed_interpolate` and `rot_pos_ids` in `vllm/model_executor/models/qwen3_vl.py`).
/// Weights must be FP16 model parameters widened to FP32, as in the model loader.
pub fn interpolate_pos_embed_and_rot_ids(
    pos_embed_weight_f32: &[f32],
    num_grid_per_side: usize,
    hidden_size: usize,
    spatial_merge_size: usize,
    grid_thw_list: &[[usize; 3]],
) -> (Vec<f32>, Vec<i32>, Vec<i32>) {
    // vLLM's Triton kernel casts coefficients to the FP16 output dtype.
    // The compiled kernel contracts terms 01 and 11 into FP16 FMAs; term 10
    // depends on the thread layout below. Keep the native single rounding.
    // These public conversion helpers also work without CUDA.
    let half = |x: f32| crate::cuda::f16_to_f32(crate::cuda::f32_to_f16(x));
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
        // Python computes scales in double, then passes FP32 kernel arguments.
        let scale = |size: usize| {
            if size <= 1 {
                0.0
            } else {
                ((num_grid_per_side - 1) as f64 / (size - 1) as f64) as f32
            }
        };
        let h_scale = scale(h);
        let w_scale = scale(w);

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

                            let hy = ih as f32 * h_scale;
                            let wx = iw as f32 * w_scale;
                            let hf = (hy.floor() as usize).min(num_grid_per_side - 1);
                            let hc = (hf + 1).min(num_grid_per_side - 1);
                            let wf = (wx.floor() as usize).min(num_grid_per_side - 1);
                            let wc = (wf + 1).min(num_grid_per_side - 1);

                            // Triton contracts coordinate subtraction and w10/w01
                            // into FP32 FMAs; floor still uses the rounded product.
                            let dh = (ih as f32).mul_add(h_scale, -(hf as f32));
                            let dw = (iw as f32).mul_add(w_scale, -(wf as f32));
                            let w11 = dh * dw;
                            let w10 = (-dh).mul_add(dw, dh);
                            let w01 = (-dh).mul_add(dw, dw);
                            let w00 = 1.0 - dh - w01;

                            let r00 = &pos_embed_weight_f32
                                [(hf * num_grid_per_side + wf) * hidden_size..];
                            let r01 = &pos_embed_weight_f32
                                [(hf * num_grid_per_side + wc) * hidden_size..];
                            let r10 = &pos_embed_weight_f32
                                [(hc * num_grid_per_side + wf) * hidden_size..];
                            let r11 = &pos_embed_weight_f32
                                [(hc * num_grid_per_side + wc) * hidden_size..];

                            let (w00, w01, w10, w11) = (half(w00), half(w01), half(w10), half(w11));
                            for c in 0..hidden_size {
                                let p00 = half(w00 * r00[c]);
                                let s01 = fused_half_mul_add(w01, r01[c], p00);
                                // Native Triton uses four warps (128 threads).
                                // With multiple elements per thread its compiled
                                // FP16 chain contracts all three additions; the
                                // single-element layout leaves the middle product
                                // materialized. Verified against installed 3.7.1
                                // sm86 PTX at sampled widths 3..128 and 129..1152, plus
                                // independent full-grid model-width fingerprints.
                                let s10 = if hidden_size > 128 {
                                    fused_half_mul_add(w10, r10[c], s01)
                                } else {
                                    half(s01 + half(w10 * r10[c]))
                                };
                                let v = fused_half_mul_add(w11, r11[c], s10);
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
            return Err(CudaError::new(
                -10,
                "non-empty deepstack_visual_indexes not supported in Qwen3_5VisionEncoder",
            ));
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
            let key = archive
                .resolve_name(name)
                .ok_or_else(|| CudaError::with_detail(-10, "missing vision tensor", name))?;
            let bits = archive
                .read_f16_bits(&key)
                .map_err(|e| CudaError::with_detail(-10, "read vision tensor", e.to_string()))?;
            arena.upload_slice(&bits, stream)
        };

        let patch_proj_w = upload_f16(&mut arena, &format!("{pfx}.patch_embed.proj.weight"))?;
        let patch_proj_b = upload_f16(&mut arena, &format!("{pfx}.patch_embed.proj.bias"))?;

        let pos_key = archive
            .resolve_name(&format!("{pfx}.pos_embed.weight"))
            .ok_or(CudaError::new(-10, "missing pos_embed.weight"))?;
        let pos_f16 = archive
            .read_f16_bits(&pos_key)
            .map_err(|e| CudaError::with_detail(-10, "read pos_embed.weight", e.to_string()))?;
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
        self.encode_images_with_trace(images, cublas, stream, None)
    }

    /// Diagnostic replay only: three rows per stage, widened on the host.
    /// Shapes are normalized to [rows, channels], removing native singleton axes.
    pub fn encode_images_with_trace(
        &self,
        images: &[PreprocessedImage],
        cublas: &CublasHandle,
        stream: &CudaStream,
        mut trace: Option<&mut VisionTrace>,
    ) -> Result<DeviceBuffer> {
        macro_rules! sample {
            ($name:expr, $buf:expr, $rows:expr, $cols:expr) => {
                if let Some(trace_out) = trace.as_deref_mut() {
                    Self::record_vision_trace_sample(
                        trace_out,
                        stream,
                        &format!("vision_trace.{}", $name),
                        &$buf,
                        $rows,
                        $cols,
                    )?;
                }
            };
        }
        let _dev = CudaDevice::set_current(self.device_id)?;

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
        // FA2 writes per-head logsumexp separately from QKV/output. Keep this
        // workspace alive and reuse it across all vision blocks on this stream.
        let attention_scratch_bytes = total_patches
            .checked_mul(num_heads)
            .and_then(|n| n.checked_mul(std::mem::size_of::<f32>()))
            .ok_or(CudaError::new(-1, "vision attention scratch overflow"))?;
        let attention_scratch =
            DeviceBuffer::alloc(self.device_id, attention_scratch_bytes.max(4))?;
        let ex = stream.exec(attention_scratch.addr(), attention_scratch_bytes.max(4));

        let mut patches_f16 = Vec::with_capacity(total_patches * patch_dim);
        for img in images {
            for &v in img.patches_f32.iter() {
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
        cublas.gemm_f16_bias(
            d_patches.addr(),
            self.patch_proj_w,
            self.patch_proj_b,
            d_x.addr(),
            total_patches as i32,
            hidden as i32,
            patch_dim as i32,
        )?;
        sample!("patch_embed", d_x, total_patches, hidden);
        check_rc(
            // SAFETY: `d_x` and `d_pos_interp` are valid device buffers of length `total_patches * hidden * 2`
            // bytes, execution stream is valid, and null bias indicates no additive bias vector.
            unsafe {
                ffi::w34_vision_bias_add_f16(
                    &ex,
                    d_x.addr() as *const c_void,
                    core::ptr::null(),
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
        for (index, blk) in self.blocks.iter().enumerate() {
            sample!(format!("blocks.{index}.input"), d_x, total_patches, hidden);
            check_rc(
                // SAFETY: `d_x` and `d_norm` are valid device buffers with capacity for `total_patches * hidden * 2` bytes,
                // norm1 weight and bias pointers are valid weights loaded from safetensors, and epsilon is positive.
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
            sample!(
                format!("blocks.{index}.norm1"),
                d_norm,
                total_patches,
                hidden
            );

            cublas.gemm_f16_bias(
                d_norm.addr(),
                blk.qkv_w,
                blk.qkv_b,
                d_qkv.addr(),
                total_patches as i32,
                (3 * hidden) as i32,
                hidden as i32,
            )?;
            sample!(
                format!("blocks.{index}.qkv"),
                d_qkv,
                total_patches,
                3 * hidden
            );

            check_rc(
                // SAFETY: `d_qkv` contains valid packed QKV data, `d_pos_hw` and `d_cu_seqlens` are valid device integer
                // arrays for the image batch, `d_attn_out` has capacity for `total_patches * hidden * 2` bytes, and theta is 10000.0.
                unsafe {
                    ffi::w34_vision_qkv_rope_attn_f16(
                        &ex,
                        d_qkv.addr() as *mut c_void,
                        core::ptr::null(),
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
            sample!(
                format!("blocks.{index}.attn_output"),
                d_attn_out,
                total_patches,
                hidden
            );

            cublas.gemm_f16_bias(
                d_attn_out.addr(),
                blk.proj_w,
                blk.proj_b,
                d_proj_out.addr(),
                total_patches as i32,
                hidden as i32,
                hidden as i32,
            )?;
            sample!(
                format!("blocks.{index}.proj"),
                d_proj_out,
                total_patches,
                hidden
            );

            check_rc(
                // SAFETY: `d_proj_out` and `d_x` are valid device buffers of length `total_patches * hidden * 2` bytes,
                // executed on the active stream context.
                unsafe {
                    ffi::w34_vision_bias_add_f16(
                        &ex,
                        d_proj_out.addr() as *const c_void,
                        core::ptr::null(),
                        d_x.addr() as *const c_void,
                        d_x.addr() as *mut c_void,
                        total_patches as i32,
                        hidden as i32,
                    )
                },
                "w34_vision_bias_add_f16(attn.proj)",
            )?;

            check_rc(
                // SAFETY: `d_x` and `d_norm` are valid device buffers of length `total_patches * hidden * 2` bytes,
                // norm2 weight and bias pointers are valid device weights.
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
            sample!(
                format!("blocks.{index}.norm2"),
                d_norm,
                total_patches,
                hidden
            );

            cublas.gemm_f16_bias(
                d_norm.addr(),
                blk.fc1_w,
                blk.fc1_b,
                d_mlp_interm.addr(),
                total_patches as i32,
                interm as i32,
                hidden as i32,
            )?;
            sample!(
                format!("blocks.{index}.fc1"),
                d_mlp_interm,
                total_patches,
                interm
            );

            check_rc(
                // SAFETY: `d_mlp_interm` is a valid device buffer of length `total_patches * interm * 2` bytes,
                // in-place activation execution on active stream.
                unsafe {
                    ffi::w34_vision_bias_gelu_f16(
                        &ex,
                        d_mlp_interm.addr() as *const c_void,
                        core::ptr::null(),
                        d_mlp_interm.addr() as *mut c_void,
                        total_patches as i32,
                        interm as i32,
                        use_tanh,
                    )
                },
                "w34_vision_bias_gelu_f16(mlp.fc1)",
            )?;

            cublas.gemm_f16_bias(
                d_mlp_interm.addr(),
                blk.fc2_w,
                blk.fc2_b,
                d_proj_out.addr(),
                total_patches as i32,
                hidden as i32,
                interm as i32,
            )?;
            sample!(
                format!("blocks.{index}.fc2"),
                d_proj_out,
                total_patches,
                hidden
            );

            check_rc(
                // SAFETY: `d_proj_out` and `d_x` are valid device buffers of length `total_patches * hidden * 2` bytes,
                // in-place residual addition on active stream.
                unsafe {
                    ffi::w34_vision_bias_add_f16(
                        &ex,
                        d_proj_out.addr() as *const c_void,
                        core::ptr::null(),
                        d_x.addr() as *const c_void,
                        d_x.addr() as *mut c_void,
                        total_patches as i32,
                        hidden as i32,
                    )
                },
                "w34_vision_bias_add_f16(mlp.fc2)",
            )?;
            sample!(format!("blocks.{index}.output"), d_x, total_patches, hidden);
        }

        sample!("merger.input", d_x, total_patches, hidden);

        // 3. PatchMerger: LayerNorm(hidden_size) -> view [total_merged, 4 * hidden_size] -> fc1 -> exact GELU -> fc2
        check_rc(
            // SAFETY: `d_x` and `d_norm` are valid device buffers with capacity for `total_patches * hidden * 2` bytes,
            // merger norm weights are valid device weights.
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

        cublas.gemm_f16_bias(
            d_norm.addr(),
            self.merger_fc1_w,
            self.merger_fc1_b,
            d_merger_mid.addr(),
            total_merged as i32,
            merger_hidden as i32,
            merger_hidden as i32,
        )?;

        check_rc(
            // SAFETY: `d_merger_mid` is a valid device buffer of length `total_merged * merger_hidden * 2` bytes,
            // in-place exact GELU on active stream.
            unsafe {
                ffi::w34_vision_bias_gelu_f16(
                    &ex,
                    d_merger_mid.addr() as *const c_void,
                    core::ptr::null(),
                    d_merger_mid.addr() as *mut c_void,
                    total_merged as i32,
                    merger_hidden as i32,
                    0,
                )
            },
            "w34_vision_bias_gelu_f16(merger.fc1)",
        )?;

        cublas.gemm_f16_bias(
            d_merger_mid.addr(),
            self.merger_fc2_w,
            self.merger_fc2_b,
            d_visual_out.addr(),
            total_merged as i32,
            out_hidden as i32,
            merger_hidden as i32,
        )?;

        sample!("merger.output", d_visual_out, total_merged, out_hidden);
        Ok(d_visual_out)
    }

    fn record_vision_trace_sample(
        trace: &mut VisionTrace,
        stream: &CudaStream,
        key: &str,
        buf: &DeviceBuffer,
        rows: usize,
        cols: usize,
    ) -> Result<()> {
        if rows == 0 || trace.contains_key(key) {
            return Ok(());
        }
        let mut data = Vec::with_capacity(3 * cols);
        for row in [0, rows / 2, rows - 1] {
            let mut host = vec![0u16; cols];
            check_rc(
                // SAFETY: host is allocated for `cols` half elements (`cols * 2` bytes),
                // device address is within the source device buffer, and stream is synchronized before read.
                unsafe {
                    ffi::w34_memcpy_d2h_async(
                        host.as_mut_ptr() as *mut c_void,
                        (buf.addr() + (row * cols * 2) as u64) as *const c_void,
                        cols * 2,
                        stream.handle,
                    )
                },
                "vision trace row",
            )?;
            stream.synchronize()?;
            data.extend(host.into_iter().map(f16_to_f32));
        }
        trace.insert(key.to_string(), (vec![3, cols], data));
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_smart_resize_rounding_and_clamps() {
        let factor = 32;
        let min_pixels = 32 * 32;
        let max_pixels = 512 * 512;

        // Image already multiple of 32
        let (h, w) = smart_resize(128, 256, factor, min_pixels, max_pixels);
        assert_eq!(h % 32, 0);
        assert_eq!(w % 32, 0);
        assert_eq!((h, w), (128, 256));

        // Image with arbitrary size rounded to factor
        let (h2, w2) = smart_resize(130, 250, factor, min_pixels, max_pixels);
        assert_eq!(h2 % 32, 0);
        assert_eq!(w2 % 32, 0);

        // Clamping to max_pixels
        let (h_big, w_big) = smart_resize(2000, 2000, factor, min_pixels, max_pixels);
        assert!(h_big * w_big <= max_pixels);
        assert_eq!(h_big % 32, 0);
        assert_eq!(w_big % 32, 0);

        // Clamping to min_pixels
        let (h_small, w_small) = smart_resize(10, 10, factor, min_pixels, max_pixels);
        assert!(h_small * w_small >= min_pixels);
        assert_eq!(h_small % 32, 0);
        assert_eq!(w_small % 32, 0);
    }

    #[test]
    fn test_fnv1a_hash_stability() {
        let b1 = b"hello world";
        let b2 = b"hello world";
        let b3 = b"hello world!";

        assert_eq!(fnv1a_hash_bytes(b1), fnv1a_hash_bytes(b2));
        assert_ne!(fnv1a_hash_bytes(b1), fnv1a_hash_bytes(b3));
    }

    #[test]
    fn test_decode_base64() {
        assert_eq!(decode_base64("aGVsbG8="), Some(b"hello".to_vec()));
        assert_eq!(decode_base64("d2FycC0zNA=="), Some(b"warp-34".to_vec()));
        assert_eq!(decode_base64(""), Some(vec![]));
    }

    #[test]
    fn test_preprocess_rgb_image_geometry() {
        // Small 64x64 RGB image
        let rgb = vec![128u8; 64 * 64 * 3];
        let prep = preprocess_rgb_image(
            &rgb,
            64,
            64,
            16, // patch_size
            2,  // spatial_merge_size
            2,  // temporal_patch_size
            32 * 32,
            512 * 512,
        );

        assert_eq!(prep.grid_thw[0], 1); // T
        assert_eq!(prep.grid_thw[1], 4); // H = 64 / 16 = 4
        assert_eq!(prep.grid_thw[2], 4); // W = 64 / 16 = 4
        assert_eq!(prep.num_patches(), 16);
        assert_eq!(prep.num_merged_tokens(2), 4); // 16 / 4 = 4
    }

    #[test]
    fn test_validate_image_aspect_ratio() {
        assert!(validate_image_aspect_ratio(100, 200).is_ok());
        assert!(validate_image_aspect_ratio(1, 200).is_ok());
        assert!(validate_image_aspect_ratio(1, 201).is_err());
        assert!(validate_image_aspect_ratio(201, 1).is_err());
        assert!(validate_image_aspect_ratio(2, 400).is_ok());
        assert!(validate_image_aspect_ratio(2, 401).is_err());
    }

    #[test]
    fn test_smart_resize_extreme_aspect_ratio_clamped() {
        // Does not panic; clamps long dimension to 200 * min_dim
        let (h, w) = smart_resize(1, 500, 32, 1024, 1024 * 1024);
        assert_eq!((h, w), (32, 192));
        assert!(h.max(w) <= h.min(w) * MAX_IMAGE_ASPECT_RATIO);

        let (h_large, w_large) = smart_resize(32, 70_000, 32, 1024, 1024 * 1024);
        assert!(h_large.max(w_large) <= h_large.min(w_large) * MAX_IMAGE_ASPECT_RATIO);
    }
}
