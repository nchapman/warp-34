//! OpenAI `/v1/images/generations` and `/v1/images/edits` HTTP handlers and worker thread
//! for `Qwen/Qwen-Image-2.1` and `Qwen/Qwen-Image-2.1-Turbo`.

use crate::{
    engine_worker::WorkerError,
    qwen_image::{
        QwenImage21ConditionImage, QwenImage21GenerationRequest, QwenImage21GenerationResult,
        QwenImage21PipelineEngine,
    },
    telemetry::{Config as TelemetryConfig, Telemetry},
};
use axum::{
    Json, Router,
    body::to_bytes,
    extract::{Request, State},
    http::{StatusCode, header},
    response::{IntoResponse, Response},
    routing::{get, post},
};
use image::{ColorType, GenericImageView, ImageEncoder, codecs::png::PngEncoder};
use serde_json::{Value, json};
use std::{
    sync::{
        Arc, OnceLock,
        atomic::{AtomicBool, Ordering},
        mpsc,
    },
    thread::JoinHandle,
    time::{Duration, SystemTime, UNIX_EPOCH},
};
use tokio::sync::{Semaphore, oneshot};

const IMAGE_BODY_LIMIT: usize = 64 * 1024 * 1024;
const IMAGE_BODY_TIMEOUT: Duration = Duration::from_secs(30);
const MAX_BATCH_IMAGES: usize = 4;
const MAX_CONDITION_IMAGE_DIM: u32 = 4096;
const MAX_CONDITION_IMAGE_PIXELS: u64 = 16 * 1024 * 1024;
const MAX_CONDITION_IMAGE_ALLOC: u64 = 64 * 1024 * 1024;

enum ImageWorkerCommand {
    Generate {
        req: QwenImage21GenerationRequest,
        tx: oneshot::Sender<Result<QwenImage21GenerationResult, WorkerError>>,
    },
    Shutdown,
}

pub struct QwenImage21Worker {
    tx: mpsc::SyncSender<ImageWorkerCommand>,
    handle: std::sync::Mutex<Option<JoinHandle<()>>>,
    stopping: Arc<AtomicBool>,
    panicked: Arc<AtomicBool>,
    panic_reason: Arc<OnceLock<String>>,
    pub telemetry: Arc<Telemetry>,
    pub default_steps: usize,
    pub tp_size: usize,
}

impl QwenImage21Worker {
    pub fn from_loader(
        model_name: &str,
        loader: impl FnOnce() -> Result<QwenImage21PipelineEngine, String> + Send + 'static,
    ) -> Result<Self, String> {
        let (init_tx, init_rx) = mpsc::sync_channel::<Result<(usize, usize), String>>(1);
        let (tx, rx) = mpsc::sync_channel::<ImageWorkerCommand>(16);
        let stopping = Arc::new(AtomicBool::new(false));
        let panicked = Arc::new(AtomicBool::new(false));
        let panic_reason = Arc::new(OnceLock::new());
        let stopping_bg = stopping.clone();
        let panicked_bg = panicked.clone();
        let panic_reason_bg = panic_reason.clone();
        let telemetry = Telemetry::new(TelemetryConfig {
            tp_size: 1,
            kv_format: "fp16".to_string(),
            speculation: "none".to_string(),
        });
        telemetry.set_model(model_name.to_string());

        let handle = std::thread::Builder::new()
            .name("warp34-qwen-image-worker".into())
            .spawn(move || {
                let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                    let mut engine = match loader() {
                        Ok(eng) => {
                            let default_steps = eng
                                .cfg
                                .sample_sigmas
                                .as_ref()
                                .map(|s| s.len().max(1))
                                .unwrap_or(40);
                            let _ = init_tx.send(Ok((default_steps, eng.tp_size())));
                            eng
                        }
                        Err(e) => {
                            let _ = init_tx.send(Err(e));
                            return;
                        }
                    };
                    while let Ok(cmd) = rx.recv() {
                        if stopping_bg.load(Ordering::Acquire) {
                            break;
                        }
                        match cmd {
                            ImageWorkerCommand::Generate { req, tx } => {
                                let res = engine.generate(&req).map_err(WorkerError::Cuda);
                                let _ = tx.send(res);
                            }
                            ImageWorkerCommand::Shutdown => break,
                        }
                    }
                }));
                if let Err(payload) = result {
                    let msg = if let Some(s) = payload.downcast_ref::<&str>() {
                        (*s).to_string()
                    } else if let Some(s) = payload.downcast_ref::<String>() {
                        s.clone()
                    } else {
                        "image worker thread panicked".to_string()
                    };
                    let _ = panic_reason_bg.set(msg);
                    panicked_bg.store(true, Ordering::Release);
                }
                stopping_bg.store(true, Ordering::Release);
            })
            .map_err(|e| format!("failed to spawn image worker thread: {e}"))?;

        let (default_steps, tp_size) = init_rx
            .recv()
            .map_err(|_| "image worker thread exited during initialization".to_string())??;
        Ok(Self {
            tx,
            handle: std::sync::Mutex::new(Some(handle)),
            stopping,
            panicked,
            panic_reason,
            telemetry,
            default_steps,
            tp_size,
        })
    }

    pub fn is_panicked(&self) -> bool {
        self.panicked.load(Ordering::Acquire)
    }

    pub fn panic_reason(&self) -> Option<String> {
        self.panic_reason.get().cloned()
    }

    pub fn is_stopping(&self) -> bool {
        self.stopping.load(Ordering::Acquire)
    }

    pub async fn generate(
        &self,
        req: QwenImage21GenerationRequest,
    ) -> Result<QwenImage21GenerationResult, WorkerError> {
        if self.is_panicked() {
            return Err(WorkerError::worker_panicked(
                self.panic_reason()
                    .unwrap_or_else(|| "image worker panicked".into()),
            ));
        }
        if self.is_stopping() {
            return Err(WorkerError::worker_stopped("image worker shutting down"));
        }
        let (tx, rx) = oneshot::channel();
        self.tx
            .try_send(ImageWorkerCommand::Generate { req, tx })
            .map_err(|err| match err {
                mpsc::TrySendError::Full(_) => {
                    WorkerError::overload("image worker queue capacity exceeded")
                }
                mpsc::TrySendError::Disconnected(_) => {
                    if self.is_panicked() {
                        WorkerError::worker_panicked(
                            self.panic_reason()
                                .unwrap_or_else(|| "image worker panicked".into()),
                        )
                    } else {
                        WorkerError::worker_stopped("image worker channel closed")
                    }
                }
            })?;
        rx.await.map_err(|_| {
            if self.is_panicked() {
                WorkerError::worker_panicked(
                    self.panic_reason()
                        .unwrap_or_else(|| "image worker panicked".into()),
                )
            } else {
                WorkerError::worker_stopped("image worker dropped response channel")
            }
        })?
    }

    pub async fn shutdown(&self) -> Result<(), String> {
        self.stopping.store(true, Ordering::Release);
        let tx = self.tx.clone();
        let handle = self.handle.lock().unwrap().take();
        if let Some(handle) = handle {
            tokio::task::spawn_blocking(move || {
                let _ = tx.send(ImageWorkerCommand::Shutdown);
                let _ = handle.join();
            })
            .await
            .map_err(|e| format!("failed to join image worker: {e}"))?;
        }
        Ok(())
    }
}

pub struct QwenImage21ServerState {
    pub model_name: String,
    api_key: Option<crate::api_auth::ApiKey>,
    pub worker: QwenImage21Worker,
    pub metrics: Arc<Telemetry>,
    pub allow_remote_images: bool,
    pub use_cuda_graph: bool,
    admission: Arc<Semaphore>,
}

impl QwenImage21ServerState {
    pub fn from_loader_with_options(
        model_name: String,
        allow_remote_images: bool,
        use_cuda_graph: bool,
        options: super::ServeOptions,
        loader: impl FnOnce() -> Result<QwenImage21PipelineEngine, String> + Send + 'static,
    ) -> Result<Arc<Self>, String> {
        options.validate()?;
        let worker = QwenImage21Worker::from_loader(&model_name, loader)?;
        let metrics = worker.telemetry.clone();
        metrics.set_model(model_name.clone());
        Ok(Arc::new(Self {
            model_name,
            api_key: options.api_key,
            worker,
            metrics,
            allow_remote_images,
            use_cuda_graph,
            admission: Arc::new(Semaphore::new(options.max_num_seqs.max(1))),
        }))
    }
}

#[derive(Clone, Debug)]
pub struct ParsedImageRequest {
    pub prompt: String,
    pub negative_prompt: Option<String>,
    pub condition_images: Vec<QwenImage21ConditionImage>,
    pub width: Option<u32>,
    pub height: Option<u32>,
    pub output_resolution: u32,
    pub num_inference_steps: Option<usize>,
    pub guidance_scale: f32,
    pub seed: u64,
    pub rgba_output: bool,
    pub n: usize,
    pub response_format: String,
}

const B64_TABLE: &[u8; 64] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";

pub fn encode_base64(bytes: &[u8]) -> String {
    let mut out = String::with_capacity(bytes.len().div_ceil(3) * 4);
    for chunk in bytes.chunks(3) {
        let b0 = chunk[0] as u32;
        let b1 = if chunk.len() > 1 { chunk[1] as u32 } else { 0 };
        let b2 = if chunk.len() > 2 { chunk[2] as u32 } else { 0 };
        let triple = (b0 << 16) | (b1 << 8) | b2;
        out.push(B64_TABLE[((triple >> 18) & 0x3F) as usize] as char);
        out.push(B64_TABLE[((triple >> 12) & 0x3F) as usize] as char);
        if chunk.len() > 1 {
            out.push(B64_TABLE[((triple >> 6) & 0x3F) as usize] as char);
        } else {
            out.push('=');
        }
        if chunk.len() > 2 {
            out.push(B64_TABLE[(triple & 0x3F) as usize] as char);
        } else {
            out.push('=');
        }
    }
    out
}

pub fn encode_png_bytes(
    pixels: &[u8],
    width: u32,
    height: u32,
    channels: usize,
) -> Result<Vec<u8>, String> {
    let color = match channels {
        3 => ColorType::Rgb8,
        4 => ColorType::Rgba8,
        other => return Err(format!("unsupported PNG channel count {other}")),
    };
    let expected = (width as usize)
        .checked_mul(height as usize)
        .and_then(|px| px.checked_mul(channels))
        .ok_or_else(|| "PNG dimensions overflow".to_string())?;
    if pixels.len() != expected {
        return Err(format!(
            "PNG pixel buffer length mismatch: got {}, expected {expected}",
            pixels.len()
        ));
    }
    let mut buf = Vec::with_capacity(expected / 2 + 1024);
    PngEncoder::new(&mut buf)
        .write_image(pixels, width, height, color.into())
        .map_err(|e| format!("PNG encode failed: {e}"))?;
    Ok(buf)
}

pub fn decode_condition_image_bytes(bytes: &[u8]) -> Result<QwenImage21ConditionImage, String> {
    let mut reader = image::ImageReader::new(std::io::Cursor::new(bytes))
        .with_guessed_format()
        .map_err(|e| format!("failed to inspect input image: {e}"))?;
    let mut limits = image::Limits::default();
    limits.max_image_width = Some(MAX_CONDITION_IMAGE_DIM);
    limits.max_image_height = Some(MAX_CONDITION_IMAGE_DIM);
    limits.max_alloc = Some(MAX_CONDITION_IMAGE_ALLOC);
    reader.limits(limits);
    let img = reader
        .decode()
        .map_err(|e| format!("failed to decode input image: {e}"))?;
    let (width, height) = img.dimensions();
    if width == 0
        || height == 0
        || width > MAX_CONDITION_IMAGE_DIM
        || height > MAX_CONDITION_IMAGE_DIM
        || (width as u64) * (height as u64) > MAX_CONDITION_IMAGE_PIXELS
    {
        return Err(format!(
            "condition image dimensions {width}x{height} exceed maximum allowed ({MAX_CONDITION_IMAGE_DIM}x{MAX_CONDITION_IMAGE_DIM}, 16 MP)"
        ));
    }
    crate::vision::validate_image_aspect_ratio(height as usize, width as usize)?;
    QwenImage21ConditionImage::from_rgba_u8(img.to_rgba8().into_raw(), width, height)
}

pub fn load_condition_image_str(
    spec: &str,
    allow_remote_images: bool,
) -> Result<QwenImage21ConditionImage, String> {
    let trimmed = spec.trim();
    if let Some(rest) = trimmed.strip_prefix("data:") {
        let (_, b64) = rest
            .split_once(";base64,")
            .ok_or_else(|| "data: image URI must use ;base64, encoding".to_string())?;
        let raw = crate::vision::decode_base64(b64)
            .ok_or_else(|| "invalid base64 in data: image URI".to_string())?;
        return decode_condition_image_bytes(&raw);
    }
    if trimmed.starts_with("http://") || trimmed.starts_with("https://") {
        let (rgb, height, width) =
            crate::vision::parse_image_url_with_policy(trimmed, allow_remote_images)?;
        let mut rgba = Vec::with_capacity(width * height * 4);
        for px in rgb.as_chunks::<3>().0 {
            rgba.extend_from_slice(&[px[0], px[1], px[2], 255]);
        }
        return QwenImage21ConditionImage::from_rgba_u8(rgba, width as u32, height as u32);
    }
    // Raw base64 payload (standard in OpenAI JSON image edit requests)
    if let Some(raw) = crate::vision::decode_base64(trimmed)
        && raw.len() >= 8
    {
        return decode_condition_image_bytes(&raw);
    }
    Err(
        "condition image must be a base64 string, data:image/*;base64,... URI, or allowed URL"
            .into(),
    )
}

pub fn parse_size_spec(
    size: Option<&str>,
    width: Option<u32>,
    height: Option<u32>,
) -> Result<(Option<u32>, Option<u32>), String> {
    if width.is_some() || height.is_some() {
        if let Some(w) = width
            && !(64..=2048).contains(&w)
        {
            return Err("width must be in 64..=2048".into());
        }
        if let Some(h) = height
            && !(64..=2048).contains(&h)
        {
            return Err("height must be in 64..=2048".into());
        }
        return Ok((width, height));
    }
    if let Some(s) = size
        .map(str::trim)
        .filter(|s| !s.is_empty() && !s.eq_ignore_ascii_case("auto"))
    {
        let (w_str, h_str) = s
            .split_once('x')
            .or_else(|| s.split_once('X'))
            .ok_or_else(|| format!("invalid size '{s}': expected '<width>x<height>' or 'auto'"))?;
        let w: u32 = w_str
            .parse()
            .map_err(|_| format!("invalid width in size '{s}'"))?;
        let h: u32 = h_str
            .parse()
            .map_err(|_| format!("invalid height in size '{s}'"))?;
        if !(64..=2048).contains(&w) || !(64..=2048).contains(&h) {
            return Err("image size dimensions must be in 64..=2048".into());
        }
        return Ok((Some(w), Some(h)));
    }
    Ok((None, None))
}

pub fn parse_json_image_request(
    body: &[u8],
    allow_remote_images: bool,
    require_images: bool,
) -> Result<ParsedImageRequest, String> {
    let val: Value =
        serde_json::from_slice(body).map_err(|e| format!("invalid JSON request body: {e}"))?;
    let obj = val
        .as_object()
        .ok_or_else(|| "request body must be a JSON object".to_string())?;

    let prompt = obj
        .get("prompt")
        .and_then(Value::as_str)
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .ok_or_else(|| "'prompt' is required and must be a non-empty string".to_string())?
        .to_string();

    // Preserve explicit `""` or `" "` negative prompts so `guidance_scale > 1.0` enables CFG
    // identically to `QwenImage21Pipeline.__call__(negative_prompt="", true_cfg_scale=4.0)`.
    let negative_prompt = obj
        .get("negative_prompt")
        .and_then(Value::as_str)
        .map(str::to_string);

    let mut condition_images = Vec::new();
    for key in ["image", "images"] {
        if let Some(img_val) = obj.get(key) {
            match img_val {
                Value::String(s) => {
                    if condition_images.len() >= MAX_BATCH_IMAGES {
                        return Err(format!(
                            "at most {MAX_BATCH_IMAGES} condition images are supported"
                        ));
                    }
                    condition_images.push(load_condition_image_str(s, allow_remote_images)?);
                }
                Value::Array(arr) => {
                    for item in arr {
                        if condition_images.len() >= MAX_BATCH_IMAGES {
                            return Err(format!(
                                "at most {MAX_BATCH_IMAGES} condition images are supported"
                            ));
                        }
                        let s = item
                            .as_str()
                            .ok_or_else(|| format!("'{key}' array elements must be strings"))?;
                        condition_images.push(load_condition_image_str(s, allow_remote_images)?);
                    }
                }
                Value::Null => {}
                _ => {
                    return Err(format!(
                        "'{key}' must be a string or array of strings (base64 or data URL)"
                    ));
                }
            }
        }
    }
    if require_images && condition_images.is_empty() {
        return Err("'/v1/images/edits' requires at least one input 'image'".into());
    }

    let width = obj
        .get("width")
        .and_then(Value::as_u64)
        .map(|v| u32::try_from(v).map_err(|_| "invalid width".to_string()))
        .transpose()?;
    let height = obj
        .get("height")
        .and_then(Value::as_u64)
        .map(|v| u32::try_from(v).map_err(|_| "invalid height".to_string()))
        .transpose()?;
    let size = obj.get("size").and_then(Value::as_str);
    let (w, h) = parse_size_spec(size, width, height)?;
    let output_resolution = match obj
        .get("resolution")
        .or_else(|| obj.get("output_resolution"))
        .and_then(Value::as_u64)
    {
        Some(v) => {
            let res = u32::try_from(v).map_err(|_| "invalid 'resolution'".to_string())?;
            if !(64..=2048).contains(&res) {
                return Err("'resolution' must be in 64..=2048".into());
            }
            res
        }
        None => 1024,
    };

    let num_inference_steps = obj
        .get("steps")
        .or_else(|| obj.get("num_inference_steps"))
        .and_then(Value::as_u64)
        .map(|v| v as usize);
    if num_inference_steps == Some(0) || num_inference_steps.is_some_and(|s| s > 100) {
        return Err("'steps' must be in 1..=100".into());
    }

    let guidance_scale = obj
        .get("guidance_scale")
        .or_else(|| obj.get("true_cfg_scale"))
        .or_else(|| obj.get("cfg_scale"))
        .and_then(Value::as_f64)
        .map(|v| v as f32)
        .unwrap_or(1.0);
    if !guidance_scale.is_finite() || !(0.0..=30.0).contains(&guidance_scale) {
        return Err("'guidance_scale' must be in 0.0..=30.0".into());
    }

    let seed = obj.get("seed").and_then(Value::as_u64).unwrap_or(42);

    let background = obj.get("background").and_then(Value::as_str);
    let rgba_flag = obj.get("rgba").and_then(Value::as_bool).unwrap_or(false);
    let rgba_output = match background {
        Some("transparent") => true,
        Some("opaque" | "auto") | None => rgba_flag,
        Some(other) => {
            return Err(format!(
                "invalid 'background' value '{other}': expected 'transparent', 'opaque', or 'auto'"
            ));
        }
    };

    let n = obj.get("n").and_then(Value::as_u64).unwrap_or(1) as usize;
    if !(1..=MAX_BATCH_IMAGES).contains(&n) {
        return Err(format!("'n' must be in 1..={MAX_BATCH_IMAGES}"));
    }

    let response_format = obj
        .get("response_format")
        .and_then(Value::as_str)
        .unwrap_or("b64_json")
        .to_string();
    if response_format != "b64_json" {
        return Err(format!(
            "unsupported response_format '{response_format}': only 'b64_json' is supported"
        ));
    }

    Ok(ParsedImageRequest {
        prompt,
        negative_prompt,
        condition_images,
        width: w,
        height: h,
        output_resolution,
        num_inference_steps,
        guidance_scale,
        seed,
        rgba_output,
        n,
        response_format,
    })
}

fn find_subslice(haystack: &[u8], needle: &[u8]) -> Option<usize> {
    if needle.is_empty() || haystack.len() < needle.len() {
        return None;
    }
    haystack.windows(needle.len()).position(|w| w == needle)
}

pub fn parse_multipart_image_request(
    content_type: &str,
    body: &[u8],
    require_images: bool,
) -> Result<ParsedImageRequest, String> {
    let boundary = content_type
        .split(';')
        .skip(1)
        .find_map(|param| {
            let (k, v) = param.trim().split_once('=')?;
            if k.trim().eq_ignore_ascii_case("boundary") {
                Some(v.trim().trim_matches('"').to_string())
            } else {
                None
            }
        })
        .filter(|b| !b.is_empty())
        .ok_or_else(|| "multipart/form-data Content-Type missing boundary".to_string())?;

    let delimiter = format!("--{boundary}").into_bytes();
    let mut pos = find_subslice(body, &delimiter)
        .ok_or_else(|| "multipart body missing initial boundary".to_string())?
        + delimiter.len();

    let mut prompt: Option<String> = None;
    let mut negative_prompt: Option<String> = None;
    let mut size: Option<String> = None;
    let mut width: Option<u32> = None;
    let mut height: Option<u32> = None;
    let mut output_resolution: u32 = 1024;
    let mut steps: Option<usize> = None;
    let mut guidance_scale: f32 = 1.0;
    let mut seed: u64 = 42;
    let mut background: Option<String> = None;
    let mut rgba_flag = false;
    let mut n: usize = 1;
    let mut response_format = "b64_json".to_string();
    let mut condition_images = Vec::new();

    while pos < body.len() {
        if body[pos..].starts_with(b"--") {
            break;
        }
        if body[pos..].starts_with(b"\r\n") {
            pos += 2;
        } else if body[pos..].starts_with(b"\n") {
            pos += 1;
        }
        let next_delim = find_subslice(&body[pos..], &delimiter)
            .ok_or_else(|| "unterminated multipart part".to_string())?;
        let part = &body[pos..pos + next_delim];
        pos += next_delim + delimiter.len();

        let (hdr_bytes, mut content) = if let Some(h_end) = find_subslice(part, b"\r\n\r\n") {
            (&part[..h_end], &part[h_end + 4..])
        } else if let Some(h_end) = find_subslice(part, b"\n\n") {
            (&part[..h_end], &part[h_end + 2..])
        } else {
            return Err("multipart part missing header separator".into());
        };
        if let Some(stripped) = content.strip_suffix(b"\r\n") {
            content = stripped;
        } else if let Some(stripped) = content.strip_suffix(b"\n") {
            content = stripped;
        }

        let headers = std::str::from_utf8(hdr_bytes)
            .map_err(|_| "multipart headers are not valid UTF-8".to_string())?;
        let mut field_name: Option<String> = None;
        for line in headers.lines() {
            if let Some((hk, hv)) = line.split_once(':')
                && hk.trim().eq_ignore_ascii_case("content-disposition")
            {
                for param in hv.split(';') {
                    if let Some((pk, pv)) = param.trim().split_once('=')
                        && pk.trim().eq_ignore_ascii_case("name")
                    {
                        field_name = Some(pv.trim().trim_matches('"').to_string());
                    }
                }
            }
        }
        let Some(name) = field_name else {
            continue;
        };
        let as_text = || {
            std::str::from_utf8(content)
                .map(|s| s.trim().to_string())
                .map_err(|_| format!("multipart field '{name}' is not valid UTF-8"))
        };
        match name.as_str() {
            "image" | "image[]" | "images" | "images[]" => {
                if condition_images.len() >= MAX_BATCH_IMAGES {
                    return Err(format!(
                        "at most {MAX_BATCH_IMAGES} condition images are supported"
                    ));
                }
                condition_images.push(decode_condition_image_bytes(content)?);
            }
            "prompt" => prompt = Some(as_text()?),
            "negative_prompt" => {
                let raw = std::str::from_utf8(content)
                    .map(str::to_string)
                    .map_err(|_| {
                        "multipart field 'negative_prompt' is not valid UTF-8".to_string()
                    })?;
                negative_prompt = Some(raw);
            }
            "size" => size = Some(as_text()?),
            "resolution" | "output_resolution" => {
                let res: u32 = as_text()?
                    .parse()
                    .map_err(|_| "invalid multipart 'resolution'".to_string())?;
                if !(64..=2048).contains(&res) {
                    return Err("'resolution' must be in 64..=2048".into());
                }
                output_resolution = res;
            }
            "width" => {
                width = Some(
                    as_text()?
                        .parse()
                        .map_err(|_| "invalid multipart 'width'".to_string())?,
                )
            }
            "height" => {
                height = Some(
                    as_text()?
                        .parse()
                        .map_err(|_| "invalid multipart 'height'".to_string())?,
                )
            }
            "steps" | "num_inference_steps" => {
                steps = Some(
                    as_text()?
                        .parse()
                        .map_err(|_| "invalid multipart 'steps'".to_string())?,
                )
            }
            "guidance_scale" | "true_cfg_scale" | "cfg_scale" => {
                guidance_scale = as_text()?
                    .parse()
                    .map_err(|_| "invalid multipart 'guidance_scale'".to_string())?
            }
            "seed" => {
                seed = as_text()?
                    .parse()
                    .map_err(|_| "invalid multipart 'seed'".to_string())?
            }
            "background" => background = Some(as_text()?),
            "rgba" => rgba_flag = matches!(as_text()?.as_str(), "true" | "1"),
            "n" => {
                n = as_text()?
                    .parse()
                    .map_err(|_| "invalid multipart 'n'".to_string())?
            }
            "response_format" => response_format = as_text()?,
            _ => {}
        }
    }

    let prompt = prompt
        .filter(|s| !s.is_empty())
        .ok_or_else(|| "'prompt' field is required".to_string())?;
    if require_images && condition_images.is_empty() {
        return Err("'/v1/images/edits' requires at least one 'image' part".into());
    }
    if steps == Some(0) || steps.is_some_and(|s| s > 100) {
        return Err("'steps' must be in 1..=100".into());
    }
    if !guidance_scale.is_finite() || !(0.0..=30.0).contains(&guidance_scale) {
        return Err("'guidance_scale' must be in 0.0..=30.0".into());
    }
    let (w, h) = parse_size_spec(size.as_deref(), width, height)?;
    let rgba_output = match background.as_deref() {
        Some("transparent") => true,
        Some("opaque" | "auto") | None => rgba_flag,
        Some(other) => {
            return Err(format!(
                "invalid 'background' value '{other}': expected 'transparent', 'opaque', or 'auto'"
            ));
        }
    };
    if !(1..=MAX_BATCH_IMAGES).contains(&n) {
        return Err(format!("'n' must be in 1..={MAX_BATCH_IMAGES}"));
    }
    if response_format != "b64_json" {
        return Err(format!(
            "unsupported response_format '{response_format}': only 'b64_json' is supported"
        ));
    }

    Ok(ParsedImageRequest {
        prompt,
        negative_prompt,
        condition_images,
        width: w,
        height: h,
        output_resolution,
        num_inference_steps: steps,
        guidance_scale,
        seed,
        rgba_output,
        n,
        response_format,
    })
}

async fn read_image_request(
    req: Request,
    allow_remote_images: bool,
    require_images: bool,
) -> Result<ParsedImageRequest, super::http::ApiError> {
    let content_type = req
        .headers()
        .get(header::CONTENT_TYPE)
        .and_then(|v| v.to_str().ok())
        .unwrap_or("application/json")
        .to_string();
    let body = tokio::time::timeout(
        IMAGE_BODY_TIMEOUT,
        to_bytes(req.into_body(), IMAGE_BODY_LIMIT),
    )
    .await
    .map_err(|_| super::http::bad("request body read timed out"))?
    .map_err(|e| super::http::bad(format!("failed to read request body: {e}")))?;

    if content_type
        .to_ascii_lowercase()
        .starts_with("multipart/form-data")
    {
        parse_multipart_image_request(&content_type, &body, require_images)
            .map_err(super::http::bad)
    } else {
        parse_json_image_request(&body, allow_remote_images, require_images)
            .map_err(super::http::bad)
    }
}

async fn handle_image_request(
    state: Arc<QwenImage21ServerState>,
    req: Request,
    require_images: bool,
) -> Result<Json<Value>, super::http::ApiError> {
    let parsed = read_image_request(req, state.allow_remote_images, require_images).await?;
    let _permit = state
        .admission
        .clone()
        .try_acquire_owned()
        .map_err(super::http::admission_error)?;
    let steps = parsed
        .num_inference_steps
        .unwrap_or(state.worker.default_steps);

    let mut data = Vec::with_capacity(parsed.n);
    for idx in 0..parsed.n {
        let item_seed = parsed.seed.wrapping_add(idx as u64);
        let gen_req = QwenImage21GenerationRequest {
            prompt: parsed.prompt.clone(),
            negative_prompt: parsed.negative_prompt.clone(),
            condition_images: parsed.condition_images.clone(),
            width: parsed.width,
            height: parsed.height,
            output_resolution: parsed.output_resolution,
            num_inference_steps: Some(steps),
            sigmas: None,
            true_cfg_scale: parsed.guidance_scale,
            seed: item_seed,
            initial_latents_chw_f16: None,
            rgba_output: parsed.rgba_output,
            use_cuda_graph: state.use_cuda_graph,
        };
        let res = state
            .worker
            .generate(gen_req)
            .await
            .map_err(super::http::worker_api_err)?;
        let b64 = encode_base64(&res.png_bytes);
        data.push(json!({
            "b64_json": b64,
            "revised_prompt": parsed.prompt,
            "width": res.width,
            "height": res.height,
            "channels": res.channels,
            "seed": item_seed,
            "steps": res.steps,
            "timings": {
                "condition_encode_ms": res.timings.condition_encode_ms,
                "vae_encode_ms": res.timings.vae_encode_ms,
                "denoise_step0_ms": res.timings.denoise_step0_ms,
                "denoise_cached_ms": res.timings.denoise_cached_ms,
                "vae_decode_ms": res.timings.vae_decode_ms,
                "total_ms": res.timings.total_ms,
            }
        }));
    }

    let created = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs();
    Ok(Json(json!({
        "created": created,
        "data": data,
    })))
}

async fn images_generations(
    State(s): State<Arc<QwenImage21ServerState>>,
    req: Request,
) -> Result<Json<Value>, super::http::ApiError> {
    handle_image_request(s, req, false).await
}

async fn images_edits(
    State(s): State<Arc<QwenImage21ServerState>>,
    req: Request,
) -> Result<Json<Value>, super::http::ApiError> {
    handle_image_request(s, req, true).await
}

async fn image_models(State(s): State<Arc<QwenImage21ServerState>>) -> Json<Value> {
    Json(json!({
        "object": "list",
        "data": [{
            "id": s.model_name,
            "object": "model",
            "owned_by": "warp34",
            "pipeline": "QwenImage21Pipeline",
        }]
    }))
}

async fn image_health(State(s): State<Arc<QwenImage21ServerState>>) -> Response {
    if s.worker.is_panicked() {
        let detail = s
            .worker
            .panic_reason()
            .unwrap_or_else(|| "image worker panicked".into());
        return (
            StatusCode::SERVICE_UNAVAILABLE,
            Json(json!({
                "status": "error",
                "error": format!("image worker terminated after panic: {detail}"),
            })),
        )
            .into_response();
    }
    if s.admission.is_closed() || s.worker.is_stopping() {
        return (
            StatusCode::SERVICE_UNAVAILABLE,
            Json(json!({"status": "stopping"})),
        )
            .into_response();
    }
    Json(json!({"status": "ok"})).into_response()
}

async fn image_metrics(State(s): State<Arc<QwenImage21ServerState>>) -> impl IntoResponse {
    (
        [(header::CONTENT_TYPE, "text/plain; version=0.0.4")],
        s.metrics.snapshot().prometheus(),
    )
}

async fn image_metrics_json(State(s): State<Arc<QwenImage21ServerState>>) -> impl IntoResponse {
    (
        [(header::CACHE_CONTROL, "no-store")],
        Json(s.metrics.snapshot()),
    )
}

async fn count_image_responses(
    State(s): State<Arc<QwenImage21ServerState>>,
    request: Request,
    next: axum::middleware::Next,
) -> Response {
    let count = *request.method() == axum::http::Method::POST
        && matches!(
            request.uri().path(),
            "/v1/images/generations" | "/images/generations" | "/v1/images/edits" | "/images/edits"
        );
    let response = next.run(request).await;
    if count {
        s.metrics.http_response(response.status().as_u16());
    }
    response
}

pub fn image_router(state: Arc<QwenImage21ServerState>) -> Router {
    Router::new()
        .route("/v1/models", get(image_models))
        .route("/v1/images/generations", post(images_generations))
        .route("/images/generations", post(images_generations))
        .route("/v1/images/edits", post(images_edits))
        .route("/images/edits", post(images_edits))
        .route_layer(axum::middleware::from_fn_with_state(
            state.api_key.clone(),
            crate::api_auth::authenticate,
        ))
        .route("/health", get(image_health))
        .route("/metrics", get(image_metrics))
        .route("/api/metrics", get(image_metrics_json))
        .merge(crate::dashboard::router())
        .layer(axum::middleware::from_fn_with_state(
            state.clone(),
            count_image_responses,
        ))
        .fallback(|| async { super::http::error(StatusCode::NOT_FOUND, "not found") })
        .with_state(state)
}

pub fn run_blocking_image_server(
    addr: &str,
    state: Arc<QwenImage21ServerState>,
) -> std::io::Result<()> {
    let runtime = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()?;
    runtime.block_on(async {
        let listener = match tokio::net::TcpListener::bind(addr).await {
            Ok(listener) => listener,
            Err(error) => {
                state
                    .worker
                    .shutdown()
                    .await
                    .map_err(std::io::Error::other)?;
                return Err(error);
            }
        };
        eprintln!(
            "[warp34 serve] QwenImage21 ready on http://{}",
            listener.local_addr()?
        );
        let app = image_router(state.clone());
        let res = axum::serve(listener, app)
            .with_graceful_shutdown(async {
                #[cfg(unix)]
                {
                    let mut terminate =
                        tokio::signal::unix::signal(tokio::signal::unix::SignalKind::terminate())
                            .expect("SIGTERM handler");
                    tokio::select! { _ = tokio::signal::ctrl_c() => {}, _ = terminate.recv() => {} }
                }
                #[cfg(not(unix))]
                {
                    let _ = tokio::signal::ctrl_c().await;
                }
            })
            .await;
        state
            .worker
            .shutdown()
            .await
            .map_err(std::io::Error::other)?;
        res
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn base64_and_png_roundtrip_rgb_and_rgba() {
        let rgba = vec![
            255, 0, 0, 255, // red opaque
            0, 255, 0, 128, // green semi-transparent
            0, 0, 255, 0, // blue transparent
            255, 255, 255, 255, // white opaque
        ];
        let png = encode_png_bytes(&rgba, 2, 2, 4).unwrap();
        let b64 = encode_base64(&png);
        let decoded_png = crate::vision::decode_base64(&b64).unwrap();
        assert_eq!(decoded_png, png);
        let cond = decode_condition_image_bytes(&decoded_png).unwrap();
        assert_eq!(cond.width, 2);
        assert_eq!(cond.height, 2);
        assert_eq!(cond.rgba, rgba);
    }

    #[test]
    fn json_and_multipart_image_requests_parse_cleanly() {
        let rgb = vec![120u8; 4 * 4 * 3];
        let png = encode_png_bytes(&rgb, 4, 4, 3).unwrap();
        let data_uri = format!("data:image/png;base64,{}", encode_base64(&png));

        let json_body = serde_json::to_vec(&json!({
            "prompt": "A glowing crystal cube",
            "negative_prompt": "blurry",
            "image": [data_uri],
            "size": "1024x768",
            "steps": 8,
            "guidance_scale": 1.0,
            "seed": 123,
            "background": "transparent"
        }))
        .unwrap();
        let parsed = parse_json_image_request(&json_body, false, true).unwrap();
        assert_eq!(parsed.prompt, "A glowing crystal cube");
        assert_eq!(parsed.negative_prompt.as_deref(), Some("blurry"));
        assert_eq!(parsed.condition_images.len(), 1);
        assert_eq!(parsed.width, Some(1024));
        assert_eq!(parsed.height, Some(768));
        assert_eq!(parsed.output_resolution, 1024);
        assert_eq!(parsed.num_inference_steps, Some(8));
        assert_eq!(parsed.seed, 123);
        assert!(parsed.rgba_output);

        // Explicit empty `negative_prompt: ""` must remain `Some("")` so CFG stays enabled!
        let empty_neg_body = serde_json::to_vec(&json!({
            "prompt": "A mountain lake",
            "negative_prompt": "",
            "guidance_scale": 4.0
        }))
        .unwrap();
        let empty_neg = parse_json_image_request(&empty_neg_body, false, false).unwrap();
        assert_eq!(empty_neg.negative_prompt.as_deref(), Some(""));
        assert_eq!(empty_neg.width, None);
        assert_eq!(empty_neg.height, None);
        assert_eq!(empty_neg.output_resolution, 1024);

        let boundary = "----warp34boundary";
        let mut mp = Vec::new();
        mp.extend_from_slice(
            format!(
                "--{boundary}\r\nContent-Disposition: form-data; name=\"prompt\"\r\n\r\nEdit the sky\r\n"
            )
            .as_bytes(),
        );
        mp.extend_from_slice(
            format!(
                "--{boundary}\r\nContent-Disposition: form-data; name=\"size\"\r\n\r\n512x512\r\n"
            )
            .as_bytes(),
        );
        mp.extend_from_slice(
            format!(
                "--{boundary}\r\nContent-Disposition: form-data; name=\"image\"; filename=\"in.png\"\r\nContent-Type: image/png\r\n\r\n"
            )
            .as_bytes(),
        );
        mp.extend_from_slice(&png);
        mp.extend_from_slice(format!("\r\n--{boundary}--\r\n").as_bytes());

        let mp_parsed = parse_multipart_image_request(
            &format!("multipart/form-data; boundary={boundary}"),
            &mp,
            true,
        )
        .unwrap();
        assert_eq!(mp_parsed.prompt, "Edit the sky");
        assert_eq!(mp_parsed.width, Some(512));
        assert_eq!(mp_parsed.height, Some(512));
        assert_eq!(mp_parsed.condition_images.len(), 1);
    }
}
