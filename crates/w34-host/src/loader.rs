//! Unified Safetensors & HuggingFace Hub checkpoint loader (`w34-host::loader`).
#![allow(clippy::chunks_exact_to_as_chunks)]
//!
//! Supports:
//! - Single-file (`model.safetensors`) and sharded (`model.safetensors.index.json`) archives,
//!   plus standalone `.safetensors` fixture files.
//! - HuggingFace Hub cache resolution (`refs/main` -> `snapshots/<sha>`).
//! - Quantization formats: `compressed-tensors` (`pack-quantized` INT4/INT8), `gptq` / `gptq_marlin`,
//!   `awq`, `auto-round`, and unquantized `FP16` / `BF16` / `FP32`.
//! - Model families: Qwen3.5 dense (`4B`, `9B`), Qwen3.8 dense (`27B`), Qwen3.6 MoE (`35B-A3B` + MTP),
//!   and `Qwen3DFlashDraftModel` (`DFlash2`).

use crate::cuda::{bf16_to_f16, bf16_to_f32, f16_to_f32, f32_to_bf16, f32_to_f16};
use serde_json::Value as Json;
use std::collections::HashMap;
use std::fs::{self, File};
use std::io;
use std::os::unix::fs::FileExt;
use std::path::{Path, PathBuf};

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TensorInfo {
    pub name: String,
    pub shard_file: String,
    pub offset: u64,
    pub len_bytes: u64,
    pub dtype: String,
    pub shape: Vec<usize>,
}

impl TensorInfo {
    pub fn num_elems(&self) -> usize {
        self.shape.iter().copied().product()
    }
}

pub struct SafetensorsArchive {
    files: HashMap<String, File>,
    entries: HashMap<String, TensorInfo>,
}

impl SafetensorsArchive {
    /// Opens either a checkpoint directory (`model.safetensors` or `model.safetensors.index.json`)
    /// or a standalone `.safetensors` file directly.
    pub fn open(path: &Path) -> io::Result<Self> {
        if path.is_file() {
            return Self::open_single_file(path);
        }
        let index_path = path.join("model.safetensors.index.json");
        if index_path.exists() {
            Self::open_indexed_dir(path, &index_path)
        } else {
            let single = path.join("model.safetensors");
            if !single.exists() {
                return Err(io::Error::new(
                    io::ErrorKind::NotFound,
                    format!("No model.safetensors or index in {}", path.display()),
                ));
            }
            Self::open_single_file(&single)
        }
    }

    fn parse_safetensors_header(file: &File) -> io::Result<(u64, u64, Json)> {
        let file_len = file.metadata()?.len();
        if file_len < 8 {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "safetensors file shorter than 8 bytes",
            ));
        }
        let mut len_bytes = [0u8; 8];
        file.read_exact_at(&mut len_bytes, 0)?;
        let header_len = u64::from_le_bytes(len_bytes);
        let data_base = 8u64.checked_add(header_len).ok_or_else(|| {
            io::Error::new(io::ErrorKind::InvalidData, "safetensors header overflow")
        })?;
        if data_base > file_len {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "safetensors header exceeds file length",
            ));
        }
        let mut header_buf = vec![0u8; header_len as usize];
        file.read_exact_at(&mut header_buf, 8)?;
        let header: Json = serde_json::from_slice(&header_buf).map_err(|e| {
            io::Error::new(
                io::ErrorKind::InvalidData,
                format!("invalid safetensors JSON header: {e}"),
            )
        })?;
        Ok((data_base, file_len, header))
    }

    fn insert_entries_from_header(
        shard_key: &str,
        data_base: u64,
        file_len: u64,
        header: &Json,
        entries: &mut HashMap<String, TensorInfo>,
    ) -> io::Result<()> {
        let obj = header.as_object().ok_or_else(|| {
            io::Error::new(
                io::ErrorKind::InvalidData,
                "safetensors header is not an object",
            )
        })?;
        for (name, meta) in obj {
            if name == "__metadata__" || meta.get("dtype").is_none() {
                continue;
            }
            let dtype = meta["dtype"]
                .as_str()
                .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidData, "missing dtype"))?
                .to_string();
            let shape_arr = meta["shape"]
                .as_array()
                .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidData, "missing shape"))?;
            let mut shape = Vec::with_capacity(shape_arr.len());
            for dim in shape_arr {
                let d = dim
                    .as_u64()
                    .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidData, "invalid dim"))?;
                shape.push(d as usize);
            }
            let offsets = meta["data_offsets"].as_array().ok_or_else(|| {
                io::Error::new(io::ErrorKind::InvalidData, "missing data_offsets")
            })?;
            let start = offsets.first().and_then(|v| v.as_u64()).ok_or_else(|| {
                io::Error::new(io::ErrorKind::InvalidData, "invalid start offset")
            })?;
            let end = offsets
                .get(1)
                .and_then(|v| v.as_u64())
                .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidData, "invalid end offset"))?;
            if end < start {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    "safetensors end < start",
                ));
            }
            let abs_off = data_base.checked_add(start).ok_or_else(|| {
                io::Error::new(io::ErrorKind::InvalidData, "safetensors offset overflow")
            })?;
            let len_bytes = end - start;
            if abs_off.saturating_add(len_bytes) > file_len {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    format!("tensor {name} exceeds file end"),
                ));
            }
            entries.insert(
                name.clone(),
                TensorInfo {
                    name: name.clone(),
                    shard_file: shard_key.to_string(),
                    offset: abs_off,
                    len_bytes,
                    dtype,
                    shape,
                },
            );
        }
        Ok(())
    }

    fn open_single_file(file_path: &Path) -> io::Result<Self> {
        let f = File::open(file_path)?;
        let (data_base, file_len, header) = Self::parse_safetensors_header(&f)?;
        let shard_key = file_path
            .file_name()
            .and_then(|s| s.to_str())
            .unwrap_or("model.safetensors")
            .to_string();
        let mut entries = HashMap::new();
        Self::insert_entries_from_header(&shard_key, data_base, file_len, &header, &mut entries)?;
        let mut files = HashMap::new();
        files.insert(shard_key, f);
        Ok(Self { files, entries })
    }

    fn open_indexed_dir(dir: &Path, index_path: &Path) -> io::Result<Self> {
        let index_text = fs::read_to_string(index_path)?;
        let index_json: Json = serde_json::from_str(&index_text).map_err(|e| {
            io::Error::new(
                io::ErrorKind::InvalidData,
                format!("bad model.safetensors.index.json: {e}"),
            )
        })?;
        let weight_map = index_json["weight_map"].as_object().ok_or_else(|| {
            io::Error::new(
                io::ErrorKind::InvalidData,
                "missing weight_map in safetensors index",
            )
        })?;
        let mut files = HashMap::new();
        let mut entries = HashMap::new();
        for shard_val in weight_map.values() {
            let shard_name = shard_val.as_str().ok_or_else(|| {
                io::Error::new(io::ErrorKind::InvalidData, "shard filename not a string")
            })?;
            if files.contains_key(shard_name) {
                continue;
            }
            let shard_path = dir.join(shard_name);
            let f = File::open(&shard_path)?;
            let (data_base, file_len, header) = Self::parse_safetensors_header(&f)?;
            Self::insert_entries_from_header(
                shard_name,
                data_base,
                file_len,
                &header,
                &mut entries,
            )?;
            files.insert(shard_name.to_string(), f);
        }
        let extra_name = "model_extra_tensors.safetensors";
        let extra_path = dir.join(extra_name);
        if !files.contains_key(extra_name) && extra_path.exists() {
            let f = File::open(&extra_path)?;
            let (data_base, file_len, header) = Self::parse_safetensors_header(&f)?;
            Self::insert_entries_from_header(
                extra_name,
                data_base,
                file_len,
                &header,
                &mut entries,
            )?;
            files.insert(extra_name.to_string(), f);
        }
        Ok(Self { files, entries })
    }

    pub fn contains(&self, name: &str) -> bool {
        self.entries.contains_key(name)
    }

    pub fn len(&self) -> usize {
        self.entries.len()
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    pub fn tensor_names(&self) -> impl Iterator<Item = &str> {
        self.entries.keys().map(String::as_str)
    }

    pub fn tensor_info(&self, name: &str) -> io::Result<&TensorInfo> {
        self.entries.get(name).ok_or_else(|| {
            io::Error::new(
                io::ErrorKind::NotFound,
                format!("tensor '{name}' not found in archive"),
            )
        })
    }

    pub fn iter_entries(&self) -> impl Iterator<Item = (&str, &TensorInfo)> {
        self.entries.iter().map(|(k, v)| (k.as_str(), v))
    }

    /// Resolves either `<prefix>.<suffix>` or `model.language_model.<stripped>.<suffix>`.
    pub fn resolve_name(&self, canonical: &str) -> Option<String> {
        if self.entries.contains_key(canonical) {
            return Some(canonical.to_string());
        }
        if let Some(rest) = canonical.strip_prefix("model.") {
            let alt = format!("model.language_model.{rest}");
            if self.entries.contains_key(&alt) {
                return Some(alt);
            }
        }
        None
    }

    pub fn read_bytes(&self, name: &str) -> io::Result<Vec<u8>> {
        let info = self.tensor_info(name)?;
        let f = self.files.get(&info.shard_file).ok_or_else(|| {
            io::Error::new(
                io::ErrorKind::NotFound,
                format!("shard '{}' not open", info.shard_file),
            )
        })?;
        let mut buf = vec![0u8; info.len_bytes as usize];
        f.read_exact_at(&mut buf, info.offset)?;
        Ok(buf)
    }

    pub fn read_u32(&self, name: &str) -> io::Result<Vec<u32>> {
        let bytes = self.read_bytes(name)?;
        if !bytes.len().is_multiple_of(4) {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!("tensor {name} byte length not a multiple of 4"),
            ));
        }
        Ok(bytes
            .chunks_exact(4)
            .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect())
    }

    pub fn read_i32(&self, name: &str) -> io::Result<Vec<i32>> {
        let bytes = self.read_bytes(name)?;
        if !bytes.len().is_multiple_of(4) {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!("tensor {name} byte length not a multiple of 4"),
            ));
        }
        Ok(bytes
            .chunks_exact(4)
            .map(|c| i32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect())
    }

    pub fn read_f32(&self, name: &str) -> io::Result<Vec<f32>> {
        let info = self.tensor_info(name)?;
        let bytes = self.read_bytes(name)?;
        match info.dtype.as_str() {
            "F32" => Ok(bytes
                .chunks_exact(4)
                .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                .collect()),
            "F16" => Ok(bytes
                .chunks_exact(2)
                .map(|c| f16_to_f32(u16::from_le_bytes([c[0], c[1]])))
                .collect()),
            "BF16" => Ok(bytes
                .chunks_exact(2)
                .map(|c| bf16_to_f32(u16::from_le_bytes([c[0], c[1]])))
                .collect()),
            other => Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!("unsupported dtype {other} for read_f32({name})"),
            )),
        }
    }

    /// Reads a single row `row` (`[cols]`) of a 2-D floating-point tensor (`F32`, `F16`, `BF16`)
    /// via `pread` without reading the entire tensor into host memory.
    pub fn read_row_f32(&self, name: &str, row: usize) -> io::Result<Vec<f32>> {
        let info = self.tensor_info(name)?;
        if info.shape.len() != 2 {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!("tensor {name} is not 2-D: {:?}", info.shape),
            ));
        }
        let rows = info.shape[0];
        let cols = info.shape[1];
        if row >= rows {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                format!("row {row} out of range for {name} ([{rows}, {cols}])"),
            ));
        }
        let elem_bytes: usize = match info.dtype.as_str() {
            "F32" => 4,
            "F16" | "BF16" => 2,
            other => {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    format!("unsupported dtype {other} for read_row_f32({name})"),
                ));
            }
        };
        let row_bytes = cols.checked_mul(elem_bytes).ok_or_else(|| {
            io::Error::new(io::ErrorKind::InvalidData, "row byte length overflow")
        })?;
        let rel_off = (row as u64)
            .checked_mul(row_bytes as u64)
            .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidData, "row offset overflow"))?;
        if rel_off.saturating_add(row_bytes as u64) > info.len_bytes {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!("row {row} of {name} exceeds tensor byte slice"),
            ));
        }
        let f = self.files.get(&info.shard_file).ok_or_else(|| {
            io::Error::new(
                io::ErrorKind::NotFound,
                format!("shard '{}' not open", info.shard_file),
            )
        })?;
        let mut buf = vec![0u8; row_bytes];
        f.read_exact_at(&mut buf, info.offset + rel_off)?;
        match info.dtype.as_str() {
            "F32" => Ok(buf
                .chunks_exact(4)
                .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                .collect()),
            "F16" => Ok(buf
                .chunks_exact(2)
                .map(|c| f16_to_f32(u16::from_le_bytes([c[0], c[1]])))
                .collect()),
            "BF16" => Ok(buf
                .chunks_exact(2)
                .map(|c| bf16_to_f32(u16::from_le_bytes([c[0], c[1]])))
                .collect()),
            _ => unreachable!(),
        }
    }

    /// Reads any floating-point tensor (`F16`, `BF16`, `F32`) and converts to IEEE-754 `FP16` bits.
    pub fn read_f16_bits(&self, name: &str) -> io::Result<Vec<u16>> {
        let info = self.tensor_info(name)?;
        let bytes = self.read_bytes(name)?;
        match info.dtype.as_str() {
            "F16" => Ok(bytes
                .chunks_exact(2)
                .map(|c| u16::from_le_bytes([c[0], c[1]]))
                .collect()),
            "BF16" => Ok(bytes
                .chunks_exact(2)
                .map(|c| bf16_to_f16(u16::from_le_bytes([c[0], c[1]])))
                .collect()),
            "F32" => Ok(bytes
                .chunks_exact(4)
                .map(|c| f32_to_f16(f32::from_le_bytes([c[0], c[1], c[2], c[3]])))
                .collect()),
            other => Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!("unsupported dtype {other} for read_f16_bits({name})"),
            )),
        }
    }

    /// Reads any floating-point tensor (`BF16`, `F16`, `F32`) and converts to `BF16` bits.
    pub fn read_bf16_bits(&self, name: &str) -> io::Result<Vec<u16>> {
        let info = self.tensor_info(name)?;
        let bytes = self.read_bytes(name)?;
        match info.dtype.as_str() {
            "BF16" => Ok(bytes
                .chunks_exact(2)
                .map(|c| u16::from_le_bytes([c[0], c[1]]))
                .collect()),
            "F16" => Ok(bytes
                .chunks_exact(2)
                .map(|c| f32_to_bf16(f16_to_f32(u16::from_le_bytes([c[0], c[1]]))))
                .collect()),
            "F32" => Ok(bytes
                .chunks_exact(4)
                .map(|c| f32_to_bf16(f32::from_le_bytes([c[0], c[1], c[2], c[3]])))
                .collect()),
            other => Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!("unsupported dtype {other} for read_bf16_bits({name})"),
            )),
        }
    }
}

/// Resolves a model path or `<org>/<repo>` HuggingFace Hub identifier to a local snapshot directory.
///
/// Checks `refs/main` first when resolving from `~/.cache/huggingface/hub/models--<org>--<repo>/`
/// so multi-snapshot cache directories always pick the active revision.
pub fn resolve_hf_model_dir(model_spec: &str) -> io::Result<PathBuf> {
    let direct = Path::new(model_spec);
    if direct.exists() {
        return Ok(direct.to_path_buf());
    }
    let home = std::env::var("HOME").unwrap_or_else(|_| "/home/nchapman".to_string());
    let hub_root = std::env::var("HF_HUB_CACHE")
        .map(PathBuf::from)
        .unwrap_or_else(|_| Path::new(&home).join(".cache/huggingface/hub"));

    let folder_name = if model_spec.starts_with("models--") {
        model_spec.to_string()
    } else {
        format!("models--{}", model_spec.replace('/', "--"))
    };
    let repo_dir = hub_root.join(&folder_name);
    if !repo_dir.exists() {
        return Err(io::Error::new(
            io::ErrorKind::NotFound,
            format!(
                "HF model '{}' not found at {}",
                model_spec,
                repo_dir.display()
            ),
        ));
    }
    let snapshots_dir = repo_dir.join("snapshots");
    let ref_main = repo_dir.join("refs/main");
    if ref_main.exists()
        && let Ok(rev) = fs::read_to_string(&ref_main)
    {
        let rev = rev.trim();
        let candidate = snapshots_dir.join(rev);
        if candidate.join("config.json").exists() {
            return Ok(candidate);
        }
    }
    let mut candidates = Vec::new();
    if snapshots_dir.exists() {
        for entry in fs::read_dir(&snapshots_dir)? {
            let entry = entry?;
            let p = entry.path();
            if p.is_dir() && p.join("config.json").exists() {
                candidates.push(p);
            }
        }
    }
    candidates.sort();
    candidates.pop().ok_or_else(|| {
        io::Error::new(
            io::ErrorKind::NotFound,
            format!(
                "no valid snapshot with config.json under {}",
                repo_dir.display()
            ),
        )
    })
}

pub fn sanitize_diag(s: &str) -> String {
    let mut out = String::with_capacity(s.len().min(64));
    for c in s.chars().take(64) {
        if c.is_ascii_graphic() || c == ' ' {
            out.push(c);
        } else {
            out.push_str(&format!("\\u{{{:04x}}}", c as u32));
        }
    }
    out
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum QuantScheme {
    Unquantized,
    CompressedTensors {
        num_bits: usize,
        /// `0` denotes per-channel quantization (`group_size == size_k`, `group_blocks == -1`).
        group_size: usize,
        symmetric: bool,
        ignore: Vec<String>,
    },
    Gptq {
        bits: usize,
        /// `0` denotes per-channel quantization (`group_size == -1`).
        group_size: usize,
        sym: bool,
        desc_act: bool,
    },
    Awq {
        bits: usize,
        group_size: usize,
        zero_point: bool,
    },
    AutoRound {
        bits: usize,
        /// `0` denotes per-channel quantization (`group_size == -1`).
        group_size: usize,
        sym: bool,
        awq_packing: bool,
    },
}

impl QuantScheme {
    pub fn bits(&self) -> usize {
        match self {
            Self::Unquantized => 0,
            Self::CompressedTensors { num_bits, .. } => *num_bits,
            Self::Gptq { bits, .. } | Self::Awq { bits, .. } | Self::AutoRound { bits, .. } => {
                *bits
            }
        }
    }

    /// Raw configured group size (`0` for channelwise `W8A16` or `Unquantized`).
    pub fn raw_group_size(&self) -> usize {
        match self {
            Self::Unquantized => 0,
            Self::CompressedTensors { group_size, .. }
            | Self::Gptq { group_size, .. }
            | Self::Awq { group_size, .. }
            | Self::AutoRound { group_size, .. } => *group_size,
        }
    }

    pub fn is_channelwise(&self) -> bool {
        self.bits() > 0 && self.raw_group_size() == 0
    }

    pub fn group_size_or(&self, fallback: usize) -> usize {
        let g = self.raw_group_size();
        if g == 0 { fallback } else { g }
    }

    pub fn is_awq_packing(&self) -> bool {
        matches!(
            self,
            Self::Awq { .. }
                | Self::AutoRound {
                    awq_packing: true,
                    ..
                }
        )
    }

    pub fn expects_zero_points(&self) -> bool {
        matches!(
            self,
            Self::CompressedTensors {
                symmetric: false,
                ..
            } | Self::Awq {
                zero_point: true,
                ..
            } | Self::AutoRound {
                sym: false,
                awq_packing: true,
                ..
            }
        )
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LayerKind {
    LinearAttention,
    FullAttention,
}

#[derive(Clone, Debug, Default)]
pub struct DecisionConfigMeta {
    pub kev_head_dim: Option<usize>,
    pub kev_temperature: Option<f64>,
    pub nimble_temperature: Option<f64>,
    pub nimble_max_length: Option<usize>,
    pub decision_type: Option<String>,
    pub decision_temperature: Option<f64>,
}

#[derive(Clone, Debug)]
pub struct ModelConfig {
    pub architectures: Vec<String>,
    pub model_type: String,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub head_dim: usize,
    pub vocab_size: usize,
    pub rms_norm_eps: f32,
    pub rope_theta: f32,
    pub rotary_dim: usize,
    pub mrope_section: [usize; 3],
    pub tie_word_embeddings: bool,
    pub attn_output_gate: bool,
    pub linear_num_key_heads: usize,
    pub linear_num_value_heads: usize,
    pub linear_key_head_dim: usize,
    pub linear_value_head_dim: usize,
    pub linear_conv_kernel_dim: usize,
    pub layer_types: Vec<LayerKind>,
    pub num_experts: usize,
    pub num_experts_per_tok: usize,
    pub moe_intermediate_size: usize,
    pub shared_expert_intermediate_size: usize,
    pub norm_topk_prob: bool,
    pub scoring_func: String,
    pub mlp_only_layers: Vec<usize>,
    pub vision_config: Option<crate::vision::VisionConfig>,
    pub mtp_num_hidden_layers: usize,
    pub dflash_block_size: Option<usize>,
    pub dflash_target_layer_ids: Vec<usize>,
    pub eos_token_ids: Vec<i32>,
    pub quant: QuantScheme,
    pub decision_meta: DecisionConfigMeta,
}

impl ModelConfig {
    pub fn from_dir(dir: &Path) -> io::Result<Self> {
        let cfg_text = fs::read_to_string(dir.join("config.json"))?;
        let root: Json = serde_json::from_str(&cfg_text).map_err(|e| {
            io::Error::new(io::ErrorKind::InvalidData, format!("bad config.json: {e}"))
        })?;
        let mut cfg = Self::from_json(&root)?;
        let tc = root.get("text_config").unwrap_or(&root);
        if root.get("quantization_config").is_none_or(Json::is_null)
            && tc.get("quantization_config").is_none_or(Json::is_null)
        {
            let standalone_qcfg = dir.join("quantize_config.json");
            if standalone_qcfg.exists() {
                let qtext = fs::read_to_string(&standalone_qcfg).map_err(|e| {
                    io::Error::new(
                        io::ErrorKind::InvalidData,
                        format!("read {}: {e}", standalone_qcfg.display()),
                    )
                })?;
                let mut qjson: Json = serde_json::from_str(&qtext).map_err(|e| {
                    io::Error::new(
                        io::ErrorKind::InvalidData,
                        format!("bad {}: {e}", standalone_qcfg.display()),
                    )
                })?;
                if qjson.get("quant_method").is_none()
                    && let Some(obj) = qjson.as_object_mut()
                {
                    if let Some(b) = obj.get("w_bit").cloned() {
                        obj.insert("bits".to_string(), b);
                    }
                    if let Some(g) = obj.get("q_group_size").cloned() {
                        obj.insert("group_size".to_string(), g);
                    }
                    if obj.contains_key("zero_point") {
                        obj.insert("quant_method".to_string(), Json::String("awq".into()));
                    } else if obj.contains_key("bits")
                        || obj.contains_key("w_bit")
                        || obj.contains_key("group_size")
                    {
                        obj.insert("quant_method".to_string(), Json::String("gptq".into()));
                    } else {
                        return Err(io::Error::new(
                            io::ErrorKind::InvalidData,
                            format!(
                                "unrecognized standalone {} (missing 'bits', 'group_size', or 'zero_point')",
                                standalone_qcfg.display()
                            ),
                        ));
                    }
                }
                cfg.quant = Self::parse_quant(Some(&qjson))?;
            }
        }
        Ok(cfg)
    }

    pub fn from_json(root: &Json) -> io::Result<Self> {
        let tc = root.get("text_config").unwrap_or(root);
        let is_truthy = |v: &serde_json::Value| -> bool {
            match v {
                serde_json::Value::Bool(b) => *b,
                serde_json::Value::Null => false,
                serde_json::Value::Number(n) => n.as_f64().is_some_and(|x| x != 0.0),
                serde_json::Value::String(s) => s != "false" && s != "0" && !s.is_empty(),
                _ => true,
            }
        };
        if tc.get("layer_scale").is_some_and(is_truthy)
            || root.get("layer_scale").is_some_and(is_truthy)
        {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "layer_scale is not supported on this engine",
            ));
        }
        let architectures = root
            .get("architectures")
            .and_then(|v| v.as_array())
            .map(|arr| {
                arr.iter()
                    .filter_map(|x| x.as_str().map(String::from))
                    .collect::<Vec<_>>()
            })
            .unwrap_or_default();
        let model_type = tc
            .get("model_type")
            .or_else(|| root.get("model_type"))
            .and_then(|v| v.as_str())
            .unwrap_or("qwen3_5_text")
            .to_string();

        let hidden_size = tc["hidden_size"].as_u64().unwrap_or(2560) as usize;
        let intermediate_size = tc
            .get("intermediate_size")
            .and_then(|v| v.as_u64())
            .unwrap_or(0) as usize;
        let num_hidden_layers = tc["num_hidden_layers"].as_u64().unwrap_or(32) as usize;
        let num_attention_heads = tc["num_attention_heads"].as_u64().unwrap_or(16) as usize;
        if hidden_size == 0 || num_attention_heads == 0 {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "hidden_size and num_attention_heads must be > 0",
            ));
        }
        let num_key_value_heads = tc
            .get("num_key_value_heads")
            .and_then(|v| v.as_u64())
            .unwrap_or(num_attention_heads as u64) as usize;
        let head_dim = tc
            .get("head_dim")
            .and_then(|v| v.as_u64())
            .unwrap_or((hidden_size / num_attention_heads) as u64) as usize;
        let vocab_size = tc["vocab_size"].as_u64().unwrap_or(248320) as usize;
        let rms_norm_eps = tc
            .get("rms_norm_eps")
            .and_then(|v| v.as_f64())
            .unwrap_or(1e-6) as f32;

        let rope_params = tc
            .get("rope_parameters")
            .or_else(|| root.get("rope_parameters"));

        let rope_scaling = tc
            .get("rope_scaling")
            .or_else(|| root.get("rope_scaling"))
            .or_else(|| rope_params.and_then(|rp| rp.get("rope_scaling")));
        if let Some(rs) = rope_scaling
            && !rs.is_null()
        {
            let rtype = rs
                .get("type")
                .or_else(|| rs.get("rope_type"))
                .and_then(|v| v.as_str())
                .unwrap_or("unknown");
            if !rtype.eq_ignore_ascii_case("default") {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    format!(
                        "unsupported rope_scaling type='{}' (only standard unscaled RoPE is supported)",
                        sanitize_diag(rtype)
                    ),
                ));
            }
        }

        let rope_type = tc
            .get("rope_type")
            .or_else(|| root.get("rope_type"))
            .or_else(|| rope_params.and_then(|rp| rp.get("rope_type")))
            .or_else(|| rope_params.and_then(|rp| rp.get("type")))
            .and_then(|v| v.as_str());
        if let Some(rt) = rope_type
            && !rt.eq_ignore_ascii_case("default")
        {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!(
                    "unsupported rope_type='{}' (only standard unscaled RoPE is supported)",
                    sanitize_diag(rt)
                ),
            ));
        }
        let rope_theta = rope_params
            .and_then(|rp| rp.get("rope_theta"))
            .or_else(|| tc.get("rope_theta"))
            .and_then(|v| v.as_f64())
            .unwrap_or(10_000_000.0) as f32;
        let partial_rotary_factor = rope_params
            .and_then(|rp| rp.get("partial_rotary_factor"))
            .or_else(|| tc.get("partial_rotary_factor"))
            .and_then(|v| v.as_f64())
            .unwrap_or(0.25);
        let rotary_dim = ((head_dim as f64) * partial_rotary_factor).round() as usize;
        let mut mrope_section = [11usize, 11, 10];
        if let Some(arr) = rope_params
            .and_then(|rp| rp.get("mrope_section"))
            .and_then(|v| v.as_array())
            && arr.len() == 3
        {
            mrope_section = [
                arr[0].as_u64().unwrap_or(11) as usize,
                arr[1].as_u64().unwrap_or(11) as usize,
                arr[2].as_u64().unwrap_or(10) as usize,
            ];
        }

        let tie_word_embeddings = tc
            .get("tie_word_embeddings")
            .or_else(|| root.get("tie_word_embeddings"))
            .and_then(|v| v.as_bool())
            .unwrap_or(false);
        let attention_bias = tc
            .get("attention_bias")
            .or_else(|| root.get("attention_bias"))
            .and_then(|v| v.as_bool())
            .unwrap_or(false);
        if attention_bias {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "unsupported attention_bias=true (Qwen3.5/3.6/3.8 backbone layers are bias-free)",
            ));
        }
        let attn_output_gate = tc
            .get("attn_output_gate")
            .and_then(|v| v.as_bool())
            .unwrap_or(true);

        let linear_num_key_heads = tc
            .get("linear_num_key_heads")
            .and_then(|v| v.as_u64())
            .unwrap_or(16) as usize;
        let linear_num_value_heads = tc
            .get("linear_num_value_heads")
            .and_then(|v| v.as_u64())
            .unwrap_or(32) as usize;
        let linear_key_head_dim = tc
            .get("linear_key_head_dim")
            .and_then(|v| v.as_u64())
            .unwrap_or(128) as usize;
        if linear_key_head_dim != 128 {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!(
                    "unsupported linear_key_head_dim={linear_key_head_dim} (op_gdn kernel requires linear_key_head_dim == 128)"
                ),
            ));
        }
        let linear_value_head_dim = tc
            .get("linear_value_head_dim")
            .and_then(|v| v.as_u64())
            .unwrap_or(128) as usize;
        let linear_conv_kernel_dim = tc
            .get("linear_conv_kernel_dim")
            .and_then(|v| v.as_u64())
            .unwrap_or(4) as usize;

        let layer_types = if let Some(arr) = tc.get("layer_types").and_then(|v| v.as_array()) {
            arr.iter()
                .map(|s| match s.as_str().unwrap_or("full_attention") {
                    "linear_attention" => LayerKind::LinearAttention,
                    _ => LayerKind::FullAttention,
                })
                .collect()
        } else {
            let interval = tc
                .get("full_attention_interval")
                .and_then(|v| v.as_u64())
                .unwrap_or(4) as usize;
            (0..num_hidden_layers)
                .map(|i| {
                    if interval > 1 && (i + 1) % interval == 0 {
                        LayerKind::FullAttention
                    } else {
                        LayerKind::LinearAttention
                    }
                })
                .collect()
        };

        let num_experts = tc.get("num_experts").and_then(|v| v.as_u64()).unwrap_or(0) as usize;
        let num_experts_per_tok = tc
            .get("num_experts_per_tok")
            .and_then(|v| v.as_u64())
            .unwrap_or(0) as usize;
        let moe_intermediate_size = tc
            .get("moe_intermediate_size")
            .and_then(|v| v.as_u64())
            .unwrap_or(0) as usize;
        let shared_expert_intermediate_size = tc
            .get("shared_expert_intermediate_size")
            .and_then(|v| v.as_u64())
            .unwrap_or(0) as usize;
        let norm_topk_prob = tc
            .get("norm_topk_prob")
            .and_then(|v| v.as_bool())
            .unwrap_or(true);
        let scoring_func = tc
            .get("scoring_func")
            .and_then(|v| v.as_str())
            .unwrap_or("softmax")
            .to_string();
        let mlp_only_layers = tc
            .get("mlp_only_layers")
            .and_then(|v| v.as_array())
            .map(|arr| {
                arr.iter()
                    .filter_map(|x| x.as_u64().map(|u| u as usize))
                    .collect()
            })
            .unwrap_or_default();

        let vision_config = root.get("vision_config").and_then(|vc| {
            let mut vcfg: crate::vision::VisionConfig = serde_json::from_value(vc.clone()).ok()?;
            if let Some(id) = root.get("image_token_id").and_then(|v| v.as_i64()) {
                vcfg.image_token_id = id as i32;
            }
            if let Some(id) = root.get("vision_start_token_id").and_then(|v| v.as_i64()) {
                vcfg.vision_start_token_id = id as i32;
            }
            if let Some(id) = root.get("vision_end_token_id").and_then(|v| v.as_i64()) {
                vcfg.vision_end_token_id = id as i32;
            }
            Some(vcfg)
        });
        let mtp_num_hidden_layers = tc
            .get("mtp_num_hidden_layers")
            .and_then(|v| v.as_u64())
            .unwrap_or(0) as usize;

        let dflash_cfg = root.get("dflash_config");
        let dflash_block_size = dflash_cfg
            .and_then(|d| d.get("block_size"))
            .or_else(|| root.get("block_size"))
            .and_then(|v| v.as_u64())
            .map(|v| v as usize);
        let dflash_target_layer_ids = dflash_cfg
            .and_then(|d| d.get("target_layer_ids"))
            .and_then(|v| v.as_array())
            .map(|arr| {
                arr.iter()
                    .filter_map(|x| x.as_u64().map(|u| u as usize))
                    .collect()
            })
            .unwrap_or_default();

        let mut eos_token_ids = Vec::new();
        if let Some(eos) = tc.get("eos_token_id").or_else(|| root.get("eos_token_id")) {
            if let Some(id) = eos.as_i64() {
                eos_token_ids.push(id as i32);
            } else if let Some(arr) = eos.as_array() {
                for x in arr {
                    if let Some(id) = x.as_i64() {
                        eos_token_ids.push(id as i32);
                    }
                }
            }
        }
        if !eos_token_ids.contains(&248044) {
            eos_token_ids.push(248044);
        }
        if !eos_token_ids.contains(&248046) {
            eos_token_ids.push(248046);
        }

        let quant = Self::parse_quant(
            root.get("quantization_config")
                .or_else(|| tc.get("quantization_config")),
        )?;
        let get_f64 = |key: &str| {
            root.get(key)
                .or_else(|| tc.get(key))
                .and_then(|v| v.as_f64())
        };
        let get_usize = |key: &str| {
            root.get(key)
                .or_else(|| tc.get(key))
                .and_then(|v| v.as_u64())
                .map(|v| v as usize)
        };
        let dc = root
            .get("decision_config")
            .or_else(|| tc.get("decision_config"));
        let dc_f64 = |key: &str| dc.and_then(|d| d.get(key)).and_then(|v| v.as_f64());
        let dc_usize = |key: &str| {
            dc.and_then(|d| d.get(key))
                .and_then(|v| v.as_u64())
                .map(|v| v as usize)
        };
        let decision_meta = DecisionConfigMeta {
            kev_head_dim: get_usize("kev_head_dim").or_else(|| dc_usize("head_dim")),
            kev_temperature: get_f64("kev_temperature"),
            nimble_temperature: get_f64("nimble_temperature"),
            nimble_max_length: get_usize("nimble_max_length").or_else(|| dc_usize("max_length")),
            decision_type: root
                .get("decision_type")
                .or_else(|| tc.get("decision_type"))
                .or_else(|| dc.and_then(|d| d.get("type")))
                .and_then(|v| v.as_str())
                .map(String::from),
            decision_temperature: get_f64("decision_temperature").or_else(|| dc_f64("temperature")),
        };

        Ok(Self {
            architectures,
            model_type,
            hidden_size,
            intermediate_size,
            num_hidden_layers,
            num_attention_heads,
            num_key_value_heads,
            head_dim,
            vocab_size,
            rms_norm_eps,
            rope_theta,
            rotary_dim,
            mrope_section,
            tie_word_embeddings,
            attn_output_gate,
            linear_num_key_heads,
            linear_num_value_heads,
            linear_key_head_dim,
            linear_value_head_dim,
            linear_conv_kernel_dim,
            layer_types,
            num_experts,
            num_experts_per_tok,
            moe_intermediate_size,
            shared_expert_intermediate_size,
            norm_topk_prob,
            scoring_func,
            mlp_only_layers,
            vision_config,
            mtp_num_hidden_layers,
            dflash_block_size,
            dflash_target_layer_ids,
            eos_token_ids,
            quant,
            decision_meta,
        })
    }

    fn parse_group_size_i64(v: Option<&Json>, default_gs: i64) -> io::Result<usize> {
        let raw = match v {
            None | Some(Json::Null) => default_gs,
            Some(val) => val.as_i64().ok_or_else(|| {
                io::Error::new(
                    io::ErrorKind::InvalidData,
                    format!("invalid quantization group_size {val} (expected integer)"),
                )
            })?,
        };
        if raw == -1 {
            Ok(0)
        } else if raw > 0 {
            Ok(raw as usize)
        } else {
            Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!("unsupported quantization group_size={raw} (expected 32, 64, 128, or -1)"),
            ))
        }
    }

    fn validate_bits_and_group_size(
        method: &str,
        bits: usize,
        group_size: usize,
        allow_channelwise_8bit: bool,
    ) -> io::Result<()> {
        match bits {
            4 => {
                if !matches!(group_size, 32 | 64 | 128) {
                    return Err(io::Error::new(
                        io::ErrorKind::InvalidData,
                        format!(
                            "unsupported {method} 4-bit group_size={} (supported on sm_86 Marlin: 32, 64, 128)",
                            if group_size == 0 {
                                -1
                            } else {
                                group_size as i64
                            }
                        ),
                    ));
                }
                Ok(())
            }
            8 => {
                let ok = matches!(group_size, 32 | 64 | 128)
                    || (allow_channelwise_8bit && group_size == 0);
                if !ok {
                    return Err(io::Error::new(
                        io::ErrorKind::InvalidData,
                        format!(
                            "unsupported {method} 8-bit group_size={} (supported on sm_86 Marlin: 32, 64, 128{})",
                            if group_size == 0 {
                                -1
                            } else {
                                group_size as i64
                            },
                            if allow_channelwise_8bit {
                                ", or -1/channel"
                            } else {
                                ""
                            }
                        ),
                    ));
                }
                Ok(())
            }
            _ => Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!(
                    "unsupported {method} bit-width bits={bits} (supported on sm_86: 4-bit W4A16 or 8-bit W8A16)"
                ),
            )),
        }
    }

    fn parse_quant(qcfg: Option<&Json>) -> io::Result<QuantScheme> {
        let Some(q) = qcfg else {
            return Ok(QuantScheme::Unquantized);
        };
        if q.is_null() {
            return Ok(QuantScheme::Unquantized);
        }
        let raw_method = q
            .get("quant_method")
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .trim();
        let method_lc = raw_method.to_ascii_lowercase();
        match method_lc.as_str() {
            "compressed-tensors" | "compressed_tensors" => {
                if let Some(fmt) = q.get("format").and_then(|v| v.as_str())
                    && fmt != "pack-quantized"
                {
                    return Err(io::Error::new(
                        io::ErrorKind::InvalidData,
                        format!(
                            "unsupported compressed-tensors format '{fmt}' (RTX 3090 Ti sm_86 requires weight-only 'pack-quantized' W4A16/W8A16; FP8/NVFP4/W8A8 are not supported)"
                        ),
                    ));
                }
                if let Some(sp_fmt) = q
                    .get("sparsity_config")
                    .and_then(|sc| sc.get("format"))
                    .and_then(|v| v.as_str())
                    && !sp_fmt.is_empty()
                    && sp_fmt != "dense"
                {
                    return Err(io::Error::new(
                        io::ErrorKind::InvalidData,
                        format!(
                            "unsupported compressed-tensors sparsity format '{sp_fmt}' (sm_86 dense Marlin only)"
                        ),
                    ));
                }
                let groups_obj = q
                    .get("config_groups")
                    .and_then(|cg| cg.as_object())
                    .ok_or_else(|| {
                        io::Error::new(
                            io::ErrorKind::InvalidData,
                            "compressed-tensors quantization_config is missing config_groups",
                        )
                    })?;
                if groups_obj.is_empty() {
                    return Err(io::Error::new(
                        io::ErrorKind::InvalidData,
                        "compressed-tensors config_groups is empty",
                    ));
                }

                let mut chosen_linear: Option<(usize, usize, bool)> = None;
                let mut fallback_group: Option<(usize, usize, bool)> = None;
                let mut group_keys: Vec<&String> = groups_obj.keys().collect();
                group_keys.sort();
                for gk in group_keys {
                    let grp = &groups_obj[gk];
                    if let Some(gfmt) = grp.get("format").and_then(|v| v.as_str())
                        && gfmt != "pack-quantized"
                    {
                        return Err(io::Error::new(
                            io::ErrorKind::InvalidData,
                            format!(
                                "unsupported compressed-tensors group '{}' format '{}' (expected 'pack-quantized')",
                                sanitize_diag(gk),
                                sanitize_diag(gfmt)
                            ),
                        ));
                    }
                    if grp.get("input_activations").is_some_and(|v| !v.is_null())
                        || grp.get("output_activations").is_some_and(|v| !v.is_null())
                    {
                        return Err(io::Error::new(
                            io::ErrorKind::InvalidData,
                            format!(
                                "unsupported compressed-tensors activation quantization in '{}' (sm_86 engine supports weight-only W4A16/W8A16)",
                                sanitize_diag(gk)
                            ),
                        ));
                    }
                    let weights = grp.get("weights").filter(|v| !v.is_null()).ok_or_else(|| {
                        io::Error::new(
                            io::ErrorKind::InvalidData,
                            format!(
                                "compressed-tensors group '{}' is missing weights descriptor",
                                sanitize_diag(gk)
                            ),
                        )
                    })?;
                    if let Some(wtype) = weights.get("type").and_then(|v| v.as_str())
                        && wtype != "int"
                    {
                        return Err(io::Error::new(
                            io::ErrorKind::InvalidData,
                            format!(
                                "unsupported compressed-tensors weights.type='{}' in '{}' (expected 'int'; FP8/FP4 are not native on sm_86)",
                                sanitize_diag(wtype),
                                sanitize_diag(gk)
                            ),
                        ));
                    }
                    if let Some(ao) = weights.get("actorder") {
                        let dynamic_ao = match ao {
                            Json::Null | Json::Bool(false) => false,
                            Json::Bool(true) => true,
                            Json::String(s) => {
                                !matches!(s.as_str(), "none" | "static" | "false" | "")
                            }
                            _ => true,
                        };
                        if dynamic_ao {
                            return Err(io::Error::new(
                                io::ErrorKind::InvalidData,
                                format!(
                                    "unsupported compressed-tensors weights.actorder={} in '{}' (channel-permuted actorder is not supported)",
                                    sanitize_diag(&ao.to_string()),
                                    sanitize_diag(gk)
                                ),
                            ));
                        }
                    }
                    let num_bits = weights
                        .get("num_bits")
                        .and_then(|v| v.as_u64())
                        .unwrap_or(4) as usize;
                    let strategy = weights
                        .get("strategy")
                        .and_then(|v| v.as_str())
                        .unwrap_or("group");
                    let group_size = match strategy {
                        "channel" => 0,
                        "group" => Self::parse_group_size_i64(weights.get("group_size"), 32)?,
                        other => {
                            return Err(io::Error::new(
                                io::ErrorKind::InvalidData,
                                format!(
                                    "unsupported compressed-tensors weights.strategy='{}' in '{}' (expected 'group' or 8-bit 'channel')",
                                    sanitize_diag(other),
                                    sanitize_diag(gk)
                                ),
                            ));
                        }
                    };
                    let symmetric = weights
                        .get("symmetric")
                        .and_then(|v| v.as_bool())
                        .unwrap_or(true);
                    if num_bits == 8 && !symmetric {
                        return Err(io::Error::new(
                            io::ErrorKind::InvalidData,
                            format!(
                                "unsupported compressed-tensors asymmetric 8-bit weights in '{}' (8-bit W8A16 requires symmetric=true)",
                                sanitize_diag(gk)
                            ),
                        ));
                    }
                    Self::validate_bits_and_group_size(
                        "compressed-tensors",
                        num_bits,
                        group_size,
                        true,
                    )?;
                    let is_embedding_only = grp
                        .get("targets")
                        .and_then(|v| v.as_array())
                        .is_some_and(|arr| {
                            !arr.is_empty()
                                && arr
                                    .iter()
                                    .filter_map(|x| x.as_str())
                                    .all(|t| t == "Embedding" || t.contains("embed"))
                        });
                    if !is_embedding_only {
                        if let Some((prev_b, prev_g, prev_s)) = chosen_linear {
                            if (prev_b, prev_g, prev_s) != (num_bits, group_size, symmetric) {
                                return Err(io::Error::new(
                                    io::ErrorKind::InvalidData,
                                    format!(
                                        "conflicting compressed-tensors quantization in '{}': ({num_bits}-bit, group_size={group_size}, symmetric={symmetric}) vs previous ({prev_b}-bit, group_size={prev_g}, symmetric={prev_s})",
                                        sanitize_diag(gk)
                                    ),
                                ));
                            }
                        } else {
                            chosen_linear = Some((num_bits, group_size, symmetric));
                        }
                    } else if fallback_group.is_none() {
                        fallback_group = Some((num_bits, group_size, symmetric));
                    }
                }
                let (num_bits, group_size, symmetric) = chosen_linear
                    .or(fallback_group)
                    .ok_or_else(|| {
                        io::Error::new(
                            io::ErrorKind::InvalidData,
                            "compressed-tensors config_groups did not define any valid quantization group",
                        )
                    })?;
                let ignore = q
                    .get("ignore")
                    .and_then(|v| v.as_array())
                    .map(|arr| {
                        arr.iter()
                            .filter_map(|x| x.as_str().map(String::from))
                            .collect()
                    })
                    .unwrap_or_default();
                Ok(QuantScheme::CompressedTensors {
                    num_bits,
                    group_size,
                    symmetric,
                    ignore,
                })
            }
            "gptq" | "gptq_marlin" => {
                let bits = q.get("bits").and_then(|v| v.as_u64()).unwrap_or(4) as usize;
                let group_size = Self::parse_group_size_i64(q.get("group_size"), 128)?;
                let sym = q.get("sym").and_then(|v| v.as_bool()).unwrap_or(true);
                let desc_act = q.get("desc_act").and_then(|v| v.as_bool()).unwrap_or(false);
                if !sym {
                    return Err(io::Error::new(
                        io::ErrorKind::InvalidData,
                        "unsupported GPTQ config: sym=false (asymmetric GPTQ is not supported; use sym=true GPTQ/AutoRound or AWQ)",
                    ));
                }
                if desc_act && group_size != 0 {
                    return Err(io::Error::new(
                        io::ErrorKind::InvalidData,
                        "unsupported GPTQ config: desc_act=true (activation-order channel permutation is not supported; use desc_act=false)",
                    ));
                }
                Self::validate_bits_and_group_size("gptq", bits, group_size, true)?;
                if let Some(dyn_obj) = q.get("dynamic").and_then(|v| v.as_object()) {
                    for (pat, rule) in dyn_obj {
                        if pat.trim_start().starts_with("-:") {
                            continue;
                        }
                        if let Some(ro) = rule.as_object() {
                            let r_bits = ro
                                .get("bits")
                                .and_then(|v| v.as_u64())
                                .map(|x| x as usize)
                                .unwrap_or(bits);
                            let r_gs = if ro.contains_key("group_size") {
                                Self::parse_group_size_i64(ro.get("group_size"), 128)?
                            } else {
                                group_size
                            };
                            let r_sym = ro.get("sym").and_then(|v| v.as_bool()).unwrap_or(sym);
                            let r_desc = ro
                                .get("desc_act")
                                .and_then(|v| v.as_bool())
                                .unwrap_or(desc_act);
                            if !r_sym {
                                return Err(io::Error::new(
                                    io::ErrorKind::InvalidData,
                                    format!("unsupported GPTQ dynamic rule '{pat}': sym=false"),
                                ));
                            }
                            if r_desc && r_gs != 0 {
                                return Err(io::Error::new(
                                    io::ErrorKind::InvalidData,
                                    format!("unsupported GPTQ dynamic rule '{pat}': desc_act=true"),
                                ));
                            }
                            if r_bits != bits || r_gs != group_size {
                                return Err(io::Error::new(
                                    io::ErrorKind::InvalidData,
                                    format!(
                                        "unsupported GPTQ mixed bit/group dynamic override '{pat}' (bits={r_bits}, group_size={r_gs}; only '-:' FP16/BF16 skip rules are supported)"
                                    ),
                                ));
                            }
                        }
                    }
                }
                Ok(QuantScheme::Gptq {
                    bits,
                    group_size,
                    sym,
                    desc_act,
                })
            }
            "awq" | "awq_marlin" => {
                let bits = q.get("bits").and_then(|v| v.as_u64()).unwrap_or(4) as usize;
                let group_size = Self::parse_group_size_i64(q.get("group_size"), 128)?;
                let zero_point = q
                    .get("zero_point")
                    .and_then(|v| v.as_bool())
                    .unwrap_or(true);
                let version = q
                    .get("version")
                    .and_then(|v| v.as_str())
                    .unwrap_or("gemm")
                    .to_ascii_lowercase();
                if !matches!(version.as_str(), "gemm" | "marlin") {
                    return Err(io::Error::new(
                        io::ErrorKind::InvalidData,
                        format!(
                            "unsupported AWQ version '{version}' (expected 'gemm' or 'marlin')"
                        ),
                    ));
                }
                Self::validate_bits_and_group_size("awq", bits, group_size, false)?;
                if bits != 4 {
                    return Err(io::Error::new(
                        io::ErrorKind::InvalidData,
                        format!("unsupported AWQ bits={bits} (expected 4-bit AWQ)"),
                    ));
                }
                Ok(QuantScheme::Awq {
                    bits,
                    group_size,
                    zero_point,
                })
            }
            "auto-round" | "autoround" | "auto_round" => {
                if let Some(dt) = q.get("data_type").and_then(|v| v.as_str())
                    && !dt.to_ascii_lowercase().starts_with("int")
                {
                    return Err(io::Error::new(
                        io::ErrorKind::InvalidData,
                        format!(
                            "unsupported auto-round data_type='{dt}' (expected integer W4A16/W8A16)"
                        ),
                    ));
                }
                let packing_fmt = q
                    .get("packing_format")
                    .or_else(|| q.get("backend"))
                    .and_then(|v| v.as_str())
                    .unwrap_or("auto_round:auto_gptq")
                    .to_ascii_lowercase();
                if packing_fmt.contains("fp8")
                    || packing_fmt.contains("mxfp")
                    || packing_fmt.contains("nvfp")
                {
                    return Err(io::Error::new(
                        io::ErrorKind::InvalidData,
                        format!("unsupported auto-round packing_format='{packing_fmt}'"),
                    ));
                }
                let awq_packing = packing_fmt.contains("awq");
                let bits = q.get("bits").and_then(|v| v.as_u64()).unwrap_or(4) as usize;
                if awq_packing && bits != 4 {
                    return Err(io::Error::new(
                        io::ErrorKind::InvalidData,
                        format!(
                            "unsupported auto-round AWQ packing with bits={bits} (AWQ packing requires 4-bit)"
                        ),
                    ));
                }
                if let Some(ec) = q
                    .get("extra_config")
                    .or_else(|| q.get("layer_config"))
                    .and_then(|v| v.as_object())
                {
                    for (k, v) in ec {
                        if let Some(b) = v.get("bits").and_then(|x| x.as_u64())
                            && b as usize != bits
                            && b != 16
                        {
                            return Err(io::Error::new(
                                io::ErrorKind::InvalidData,
                                format!(
                                    "unsupported mixed-bit AutoRound override in '{}': {b}-bit (model base is {bits}-bit)",
                                    sanitize_diag(k)
                                ),
                            ));
                        }
                    }
                }
                let group_size = Self::parse_group_size_i64(q.get("group_size"), 128)?;
                let sym = q.get("sym").and_then(|v| v.as_bool()).unwrap_or(true);
                let desc_act = q.get("desc_act").and_then(|v| v.as_bool()).unwrap_or(false);
                if desc_act && group_size != 0 {
                    return Err(io::Error::new(
                        io::ErrorKind::InvalidData,
                        "unsupported auto-round config: desc_act=true",
                    ));
                }
                if !sym && !(awq_packing && bits == 4) {
                    return Err(io::Error::new(
                        io::ErrorKind::InvalidData,
                        "unsupported auto-round config: sym=false with GPTQ packing (requires sym=true or AWQ packing)",
                    ));
                }
                Self::validate_bits_and_group_size("auto-round", bits, group_size, !awq_packing)?;
                Ok(QuantScheme::AutoRound {
                    bits,
                    group_size,
                    sym,
                    awq_packing,
                })
            }
            _ => Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!(
                    "unsupported quantization method '{}' (RTX 3090 Ti sm_86 supports: unquantized FP16/BF16/FP32, compressed-tensors pack-quantized W4A16/W8A16, gptq/auto-round symmetric W4A16/W8A16, and awq gemm W4A16)",
                    if raw_method.is_empty() {
                        "<missing quant_method>".to_string()
                    } else {
                        sanitize_diag(raw_method)
                    }
                ),
            )),
        }
    }

    pub fn num_full_attention_layers(&self) -> usize {
        self.layer_types
            .iter()
            .filter(|&&k| k == LayerKind::FullAttention)
            .count()
    }

    pub fn num_linear_attention_layers(&self) -> usize {
        self.layer_types
            .iter()
            .filter(|&&k| k == LayerKind::LinearAttention)
            .count()
    }
}

/// Permutes a `[K / group_size, N]` FP16 scale matrix into `vLLM` Marlin scale layout.
pub fn marlin_permute_scales_k_by_n(
    scales_k_by_n: &[u16],
    size_k: usize,
    size_n: usize,
    group_size: usize,
) -> Vec<u16> {
    let num_groups = size_k / group_size;
    assert_eq!(scales_k_by_n.len(), num_groups * size_n);
    let mut out = vec![0u16; scales_k_by_n.len()];
    if group_size < size_k {
        let mut scale_perm = [0usize; 64];
        for i in 0..8 {
            for j in 0..8 {
                scale_perm[i * 8 + j] = i + 8 * j;
            }
        }
        for (chunk_idx, src_chunk) in scales_k_by_n.chunks_exact(64).enumerate() {
            let dst_base = chunk_idx * 64;
            for (i, &p) in scale_perm.iter().enumerate() {
                out[dst_base + i] = src_chunk[p];
            }
        }
    } else {
        let mut scale_perm_single = [0usize; 32];
        const BASE: [usize; 8] = [0, 1, 8, 9, 16, 17, 24, 25];
        for i in 0..4 {
            for (j, &b) in BASE.iter().enumerate() {
                scale_perm_single[i * 8 + j] = 2 * i + b;
            }
        }
        for (chunk_idx, src_chunk) in scales_k_by_n.chunks_exact(32).enumerate() {
            let dst_base = chunk_idx * 32;
            for (i, &p) in scale_perm_single.iter().enumerate() {
                out[dst_base + i] = src_chunk[p];
            }
        }
    }
    out
}

/// Transposes an N-major `[N, K / group_size]` scale tensor into `[K / group_size, N]`
/// and applies `marlin_permute_scales_k_by_n`.
pub fn marlin_permute_scales_from_n_major(
    scales_n_by_kg: &[u16],
    size_k: usize,
    size_n: usize,
    group_size: usize,
) -> Vec<u16> {
    let num_groups = size_k / group_size;
    assert_eq!(scales_n_by_kg.len(), size_n * num_groups);
    let mut k_by_n = vec![0u16; num_groups * size_n];
    for n in 0..size_n {
        for g in 0..num_groups {
            k_by_n[g * size_n + n] = scales_n_by_kg[n * num_groups + g];
        }
    }
    marlin_permute_scales_k_by_n(&k_by_n, size_k, size_n, group_size)
}

/// Dequantizes an N-major `[N, K/8]` `U4B8` weight + `[N, K/group_size]` `FP16` scale tensor
/// into an N-major `[N, K]` `FP16` matrix (used for skinny projections with `N < 64` or `N % 64 != 0`).
pub fn dequant_u4b8_n_major_to_f16(
    packed_n_by_k8: &[u32],
    scales_n_by_kg: &[u16],
    size_n: usize,
    size_k: usize,
    group_size: usize,
) -> Vec<u16> {
    let words_per_row = size_k / 8;
    let groups_per_row = size_k / group_size;
    assert_eq!(packed_n_by_k8.len(), size_n * words_per_row);
    assert_eq!(scales_n_by_kg.len(), size_n * groups_per_row);
    let mut out = vec![0u16; size_n * size_k];
    for n in 0..size_n {
        for w_idx in 0..words_per_row {
            let word = packed_n_by_k8[n * words_per_row + w_idx];
            for b in 0..8 {
                let k = w_idx * 8 + b;
                let nib = ((word >> (b * 4)) & 0xF) as i32 - 8;
                let scale = f16_to_f32(scales_n_by_kg[n * groups_per_row + (k / group_size)]);
                out[n * size_k + k] = f32_to_f16((nib as f32) * scale);
            }
        }
    }
    out
}

/// Transposes a K-major `[K/8, N]` 4-bit GPTQ `qweight` tensor into N-major `[N, K/8]` `u32` words.
pub fn transpose_gptq_qweight_k8_n_to_n_major(
    qweight_k8_by_n: &[u32],
    size_k: usize,
    size_n: usize,
) -> Vec<u32> {
    let k8 = size_k / 8;
    assert_eq!(qweight_k8_by_n.len(), k8 * size_n);
    let mut out = vec![0u32; size_n * k8];
    for kr in 0..k8 {
        for n in 0..size_n {
            out[n * k8 + kr] = qweight_k8_by_n[kr * size_n + n];
        }
    }
    out
}

/// Transposes a K-major `[K/4, N]` 8-bit GPTQ `qweight` tensor into N-major `[N, K/4]` `u32` words.
pub fn transpose_gptq_qweight_k4_n_to_n_major(
    qweight_k4_by_n: &[u32],
    size_k: usize,
    size_n: usize,
) -> Vec<u32> {
    let k4 = size_k / 4;
    assert_eq!(qweight_k4_by_n.len(), k4 * size_n);
    let mut out = vec![0u32; size_n * k4];
    for kr in 0..k4 {
        for n in 0..size_n {
            out[n * k4 + kr] = qweight_k4_by_n[kr * size_n + n];
        }
    }
    out
}

/// Unpacks an AutoAWQ `gemm` `[K, N/8]` `qweight` tensor (interleaved `[0, 2, 4, 6, 1, 3, 5, 7]`
/// along `N`) into N-major `[N, K/8]` `u32` words (8 consecutive K-nibbles per `u32`).
pub fn unpack_awq_qweight_k_n8_to_n_major(
    qweight_k_by_n8: &[u32],
    size_k: usize,
    size_n: usize,
) -> Vec<u32> {
    const UNDO_AWQ: [usize; 8] = [0, 4, 1, 5, 2, 6, 3, 7];
    let k8 = size_k / 8;
    let n8 = size_n / 8;
    assert_eq!(qweight_k_by_n8.len(), size_k * n8);
    let mut out = vec![0u32; size_n * k8];
    for k_word in 0..k8 {
        for b in 0..8 {
            let k = k_word * 8 + b;
            let src_row = &qweight_k_by_n8[k * n8..(k + 1) * n8];
            let shift_out = b * 4;
            for (n_pack, &awq_word) in src_row.iter().enumerate() {
                let n_base = n_pack * 8;
                for (c, &src_slot) in UNDO_AWQ.iter().enumerate() {
                    let nib = (awq_word >> (src_slot * 4)) & 0xF;
                    out[(n_base + c) * k8 + k_word] |= nib << shift_out;
                }
            }
        }
    }
    out
}

/// Unpacks an AutoAWQ `gemm` `[K / group_size, N / 8]` `qzeros` tensor (interleaved
/// `[0, 2, 4, 6, 1, 3, 5, 7]` along `N`) into N-major `[N / 8, K / group_size]` `u32` words
/// matching `compressed-tensors` `.weight_zero_point` layout (8 consecutive N-rows per `u32`).
pub fn unpack_awq_qzeros_kg_n8_to_n_major(
    qzeros_kg_by_n8: &[u32],
    size_k: usize,
    size_n: usize,
    group_size: usize,
) -> Vec<u32> {
    const UNDO_AWQ: [usize; 8] = [0, 4, 1, 5, 2, 6, 3, 7];
    let num_groups = size_k / group_size;
    let n8 = size_n / 8;
    assert_eq!(qzeros_kg_by_n8.len(), num_groups * n8);
    let mut out = vec![0u32; n8 * num_groups];
    for g in 0..num_groups {
        for n_pack in 0..n8 {
            let awq_word = qzeros_kg_by_n8[g * n8 + n_pack];
            let mut packed_col = 0u32;
            for (c, &src_slot) in UNDO_AWQ.iter().enumerate() {
                let nib = (awq_word >> (src_slot * 4)) & 0xF;
                packed_col |= nib << (c * 4);
            }
            out[n_pack * num_groups + g] = packed_col;
        }
    }
    out
}

/// Permutes an N-major `[N / 8, K / group_size]` packed 4-bit zero-point tensor into `vLLM`
/// Marlin `kU4` zero-point layout (`[K / group_size, N / 8]` `u32` words with `scale_perm` +
/// `[0, 2, 4, 6, 1, 3, 5, 7]` interleave).
pub fn marlin_permute_zero_points_from_n_major(
    qzeros_n8_by_kg: &[u32],
    size_k: usize,
    size_n: usize,
    group_size: usize,
) -> Vec<u32> {
    const INTERLEAVE: [usize; 8] = [0, 2, 4, 6, 1, 3, 5, 7];
    let num_groups = size_k / group_size;
    let n8 = size_n / 8;
    assert_eq!(size_n % 64, 0);
    assert_eq!(qzeros_n8_by_kg.len(), n8 * num_groups);

    let mut zp_g_by_n = vec![0u8; num_groups * size_n];
    for n_pack in 0..n8 {
        for g in 0..num_groups {
            let w = qzeros_n8_by_kg[n_pack * num_groups + g];
            let n_base = n_pack * 8;
            for b in 0..8 {
                zp_g_by_n[g * size_n + n_base + b] = ((w >> (b * 4)) & 0xF) as u8;
            }
        }
    }

    let mut permed = vec![0u8; zp_g_by_n.len()];
    let mut scale_perm = [0usize; 64];
    for i in 0..8 {
        for j in 0..8 {
            scale_perm[i * 8 + j] = i + 8 * j;
        }
    }
    for (chunk_idx, src_chunk) in zp_g_by_n.chunks_exact(64).enumerate() {
        let dst_base = chunk_idx * 64;
        for (i, &p) in scale_perm.iter().enumerate() {
            permed[dst_base + i] = src_chunk[p];
        }
    }

    let mut out = vec![0u32; num_groups * n8];
    for (word_idx, chunk8) in permed.chunks_exact(8).enumerate() {
        let mut packed = 0u32;
        for (i, &src_i) in INTERLEAVE.iter().enumerate() {
            packed |= ((chunk8[src_i] & 0xF) as u32) << (i * 4);
        }
        out[word_idx] = packed;
    }
    out
}

/// Dequantizes an N-major `[N, K/8]` asymmetric `U4` weight + `[N, K/group_size]` `FP16` scale
/// + `[N/8, K/group_size]` packed 4-bit zero-point tensor into an N-major `[N, K]` `FP16` matrix.
pub fn dequant_u4_zp_n_major_to_f16(
    packed_n_by_k8: &[u32],
    scales_n_by_kg: &[u16],
    qzeros_n8_by_kg: &[u32],
    size_n: usize,
    size_k: usize,
    group_size: usize,
) -> Vec<u16> {
    let words_per_row = size_k / 8;
    let groups_per_row = size_k / group_size;
    let n8 = size_n / 8;
    assert_eq!(size_n % 8, 0);
    assert_eq!(packed_n_by_k8.len(), size_n * words_per_row);
    assert_eq!(scales_n_by_kg.len(), size_n * groups_per_row);
    assert_eq!(qzeros_n8_by_kg.len(), n8 * groups_per_row);
    let mut out = vec![0u16; size_n * size_k];
    for n in 0..size_n {
        let n_pack = n >> 3;
        let n_shift = (n & 7) * 4;
        for w_idx in 0..words_per_row {
            let word = packed_n_by_k8[n * words_per_row + w_idx];
            for b in 0..8 {
                let k = w_idx * 8 + b;
                let g = k / group_size;
                let nib = ((word >> (b * 4)) & 0xF) as i32;
                let zp = ((qzeros_n8_by_kg[n_pack * groups_per_row + g] >> n_shift) & 0xF) as i32;
                let scale = f16_to_f32(scales_n_by_kg[n * groups_per_row + g]);
                out[n * size_k + k] = f32_to_f16(((nib - zp) as f32) * scale);
            }
        }
    }
    out
}

/// Dequantizes an N-major `[N, K/4]` symmetric `U8B128` weight + `[N, K/group_size]` `FP16` scale
/// tensor into an N-major `[N, K]` `FP16` matrix.
pub fn dequant_u8b128_n_major_to_f16(
    packed_n_by_k4: &[u32],
    scales_n_by_kg: &[u16],
    size_n: usize,
    size_k: usize,
    group_size: usize,
) -> Vec<u16> {
    let words_per_row = size_k / 4;
    let groups_per_row = size_k / group_size;
    assert_eq!(packed_n_by_k4.len(), size_n * words_per_row);
    assert_eq!(scales_n_by_kg.len(), size_n * groups_per_row);
    let mut out = vec![0u16; size_n * size_k];
    for n in 0..size_n {
        for w_idx in 0..words_per_row {
            let word = packed_n_by_k4[n * words_per_row + w_idx];
            for b in 0..4 {
                let k = w_idx * 4 + b;
                let q = ((word >> (b * 8)) & 0xFF) as i32 - 128;
                let scale = f16_to_f32(scales_n_by_kg[n * groups_per_row + (k / group_size)]);
                out[n * size_k + k] = f32_to_f16((q as f32) * scale);
            }
        }
    }
    out
}

/// Validates that an optional GPTQ `.g_idx` tensor represents static sequential group assignment
/// (`g_idx[k] == k / group_size`, or all `0` for channelwise) rather than an unapplied `desc_act`
/// channel permutation.
pub fn validate_static_gptq_g_idx(
    g_idx_words: &[u32],
    size_k: usize,
    group_size: usize,
    tensor_name: &str,
) -> io::Result<()> {
    if g_idx_words.len() != size_k {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!(
                "invalid g_idx length {} for '{tensor_name}' (expected K={size_k})",
                g_idx_words.len()
            ),
        ));
    }
    if g_idx_words.iter().all(|&x| x == 0) {
        return Ok(());
    }
    let eff_gs = if group_size == 0 { size_k } else { group_size };
    for (k, &raw_g) in g_idx_words.iter().enumerate() {
        let expected = (k / eff_gs) as u32;
        if raw_g != expected {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!(
                    "tensor '{tensor_name}' contains non-sequential g_idx[{k}]={} (expected {expected}); activation-order (desc_act) channel permutation is not supported",
                    raw_g as i32
                ),
            ));
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tensor_info_num_elems() {
        let t = TensorInfo {
            name: "test.weight".into(),
            shard_file: "model.safetensors".into(),
            offset: 0,
            len_bytes: 1024,
            dtype: "F16".into(),
            shape: vec![4, 8, 16],
        };
        assert_eq!(t.num_elems(), 4 * 8 * 16);
    }

    #[test]
    fn test_transpose_gptq_qweight_k8_n_to_n_major() {
        // size_k = 16 (k8 = 2), size_n = 3
        // input is [k8, size_n] = [2, 3]
        // row 0: [1, 2, 3]
        // row 1: [4, 5, 6]
        let input = vec![1u32, 2, 3, 4, 5, 6];
        let out = transpose_gptq_qweight_k8_n_to_n_major(&input, 16, 3);
        // output should be [size_n, k8] = [3, 2]
        // col 0: [1, 4]
        // col 1: [2, 5]
        // col 2: [3, 6]
        assert_eq!(out, vec![1, 4, 2, 5, 3, 6]);
    }

    #[test]
    fn test_model_config_from_json_parsing() {
        let json = serde_json::json!({
            "model_type": "qwen3_5",
            "hidden_size": 2560,
            "num_hidden_layers": 32,
            "num_attention_heads": 20,
            "num_key_value_heads": 4,
            "head_dim": 128,
            "vocab_size": 152064,
            "rms_norm_eps": 1e-6,
            "rope_theta": 1000000.0,
            "layer_types": ["linear_attention", "linear_attention", "linear_attention", "full_attention"]
        });

        let cfg = ModelConfig::from_json(&json).unwrap();
        assert_eq!(cfg.model_type, "qwen3_5");
        assert_eq!(cfg.hidden_size, 2560);
        assert_eq!(cfg.num_hidden_layers, 32);
        assert_eq!(cfg.num_attention_heads, 20);
        assert_eq!(cfg.num_key_value_heads, 4);
        assert_eq!(cfg.head_dim, 128);
        assert_eq!(cfg.vocab_size, 152064);
        assert_eq!(cfg.rms_norm_eps, 1e-6);
        assert_eq!(cfg.layer_types.len(), 4);
    }

    #[test]
    fn test_model_config_layer_scale_rejected() {
        let json_bool = serde_json::json!({
            "model_type": "qwen3_5",
            "hidden_size": 2560,
            "num_hidden_layers": 32,
            "num_attention_heads": 20,
            "layer_scale": true
        });
        assert!(ModelConfig::from_json(&json_bool).is_err());

        let json_num = serde_json::json!({
            "model_type": "qwen3_5",
            "hidden_size": 2560,
            "num_hidden_layers": 32,
            "num_attention_heads": 20,
            "layer_scale": 1
        });
        assert!(ModelConfig::from_json(&json_num).is_err());
    }

    #[test]
    fn test_read_row_f32_dtypes_and_bounds() {
        use crate::cuda::{f32_to_bf16, f32_to_f16};
        let tmp = std::env::temp_dir().join(format!("w34_loader_row_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&tmp);
        std::fs::create_dir_all(&tmp).unwrap();
        let st_path = tmp.join("model.safetensors");

        let f32_vals: [f32; 6] = [1.0, -2.5, 3.25, -4.0, 5.5, 6.75];
        let f32_bytes: Vec<u8> = f32_vals.iter().flat_map(|v| v.to_le_bytes()).collect();
        let bf16_bytes: Vec<u8> = f32_vals
            .iter()
            .flat_map(|&v| f32_to_bf16(v).to_le_bytes())
            .collect();
        let f16_bytes: Vec<u8> = f32_vals
            .iter()
            .flat_map(|&v| f32_to_f16(v).to_le_bytes())
            .collect();
        let i32_bytes: Vec<u8> = [10i32, 20, 30, 40]
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect();

        let tensors: &[(&str, &str, &[usize], &[u8])] = &[
            ("mat_f32", "F32", &[3, 2], &f32_bytes),
            ("mat_bf16", "BF16", &[3, 2], &bf16_bytes),
            ("mat_f16", "F16", &[3, 2], &f16_bytes),
            ("vec_1d", "F32", &[6], &f32_bytes),
            ("mat_i32", "I32", &[2, 2], &i32_bytes),
            ("mat_trunc", "F32", &[3, 2], &f32_bytes[..16]),
        ];
        let mut header = serde_json::Map::new();
        let mut off = 0usize;
        for (name, dtype, shape, bytes) in tensors {
            header.insert(
                (*name).to_owned(),
                serde_json::json!({
                    "dtype": dtype,
                    "shape": shape,
                    "data_offsets": [off, off + bytes.len()],
                }),
            );
            off += bytes.len();
        }
        let mut hdr_bytes = serde_json::to_vec(&serde_json::Value::Object(header)).unwrap();
        while !hdr_bytes.len().is_multiple_of(8) {
            hdr_bytes.push(b' ');
        }
        let mut file_bytes = (hdr_bytes.len() as u64).to_le_bytes().to_vec();
        file_bytes.extend_from_slice(&hdr_bytes);
        for (_, _, _, bytes) in tensors {
            file_bytes.extend_from_slice(bytes);
        }
        std::fs::write(&st_path, file_bytes).unwrap();

        let archive = SafetensorsArchive::open(&st_path).unwrap();
        assert_eq!(archive.read_row_f32("mat_f32", 0).unwrap(), vec![1.0, -2.5]);
        assert_eq!(
            archive.read_row_f32("mat_f32", 1).unwrap(),
            vec![3.25, -4.0]
        );
        assert_eq!(archive.read_row_f32("mat_f32", 2).unwrap(), vec![5.5, 6.75]);
        assert_eq!(
            archive.read_row_f32("mat_bf16", 1).unwrap(),
            vec![3.25, -4.0]
        );
        assert_eq!(archive.read_row_f32("mat_f16", 2).unwrap(), vec![5.5, 6.75]);

        assert!(archive.read_row_f32("mat_f32", 3).is_err());
        assert!(archive.read_row_f32("vec_1d", 0).is_err());
        assert!(archive.read_row_f32("mat_i32", 0).is_err());
        assert!(archive.read_row_f32("mat_trunc", 2).is_err());
        let _ = std::fs::remove_dir_all(&tmp);
    }

    fn cfg_with_quant(qcfg: serde_json::Value) -> io::Result<QuantScheme> {
        let root = serde_json::json!({
            "model_type": "qwen3_5_text",
            "hidden_size": 1024,
            "num_hidden_layers": 2,
            "num_attention_heads": 8,
            "num_key_value_heads": 2,
            "head_dim": 128,
            "vocab_size": 1024,
            "layer_types": ["linear_attention", "full_attention"],
            "quantization_config": qcfg,
        });
        ModelConfig::from_json(&root).map(|c| c.quant)
    }

    #[test]
    fn test_parse_quant_supported_schemes() {
        for gs in [32usize, 64, 128] {
            for sym in [true, false] {
                let q = cfg_with_quant(serde_json::json!({
                    "quant_method": "compressed-tensors",
                    "format": "pack-quantized",
                    "config_groups": {
                        "group_0": {
                            "format": "pack-quantized",
                            "targets": ["Linear"],
                            "weights": {
                                "num_bits": 4,
                                "type": "int",
                                "strategy": "group",
                                "group_size": gs,
                                "symmetric": sym,
                                "actorder": "static"
                            }
                        }
                    },
                    "ignore": ["lm_head"]
                }))
                .unwrap();
                assert_eq!(
                    q,
                    QuantScheme::CompressedTensors {
                        num_bits: 4,
                        group_size: gs,
                        symmetric: sym,
                        ignore: vec!["lm_head".into()],
                    }
                );
                assert_eq!(q.expects_zero_points(), !sym);
            }
        }

        // Compressed-tensors W8A16 group & channelwise + multi-group (Linear + Embedding)
        let q_w8_chan = cfg_with_quant(serde_json::json!({
            "quant_method": "compressed-tensors",
            "format": "pack-quantized",
            "config_groups": {
                "group_0": {
                    "targets": ["Embedding"],
                    "weights": { "num_bits": 8, "type": "int", "strategy": "channel", "symmetric": true }
                },
                "group_1": {
                    "targets": ["Linear"],
                    "weights": { "num_bits": 8, "type": "int", "strategy": "channel", "symmetric": true }
                }
            }
        }))
        .unwrap();
        assert_eq!(q_w8_chan.bits(), 8);
        assert!(q_w8_chan.is_channelwise());

        // GPTQ W4A16 & W8A16 with `-:` dynamic skip overrides
        for (bits, gs) in [(4usize, 32i64), (4, 64), (4, 128), (8, 128), (8, -1)] {
            let q = cfg_with_quant(serde_json::json!({
                "quant_method": "gptq",
                "bits": bits,
                "group_size": gs,
                "sym": true,
                "desc_act": false,
                "dynamic": { "-:.*in_proj_a.*": {} }
            }))
            .unwrap();
            assert_eq!(q.bits(), bits);
            assert_eq!(q.is_channelwise(), gs == -1);
        }

        // AWQ gemm W4A16 (zero_point = true & false)
        for zp in [true, false] {
            let q = cfg_with_quant(serde_json::json!({
                "quant_method": "awq",
                "bits": 4,
                "group_size": 64,
                "zero_point": zp,
                "version": "gemm"
            }))
            .unwrap();
            assert_eq!(
                q,
                QuantScheme::Awq {
                    bits: 4,
                    group_size: 64,
                    zero_point: zp
                }
            );
            assert!(q.is_awq_packing());
            assert_eq!(q.expects_zero_points(), zp);
        }

        // AutoRound GPTQ & AWQ packing
        let ar_gptq = cfg_with_quant(serde_json::json!({
            "quant_method": "auto-round",
            "bits": 4,
            "group_size": 128,
            "sym": true,
            "data_type": "int",
            "packing_format": "auto_round:auto_gptq"
        }))
        .unwrap();
        assert!(!ar_gptq.is_awq_packing());

        let ar_awq = cfg_with_quant(serde_json::json!({
            "quant_method": "auto-round",
            "bits": 4,
            "group_size": 32,
            "sym": false,
            "data_type": "int",
            "packing_format": "auto_round:auto_awq"
        }))
        .unwrap();
        assert!(ar_awq.is_awq_packing());
        assert!(ar_awq.expects_zero_points());
    }

    #[test]
    fn test_parse_quant_rejects_unsupported_schemes_with_clear_diagnostics() {
        let bad_cases: &[(&str, serde_json::Value, &str)] = &[
            (
                "fp8 method",
                serde_json::json!({ "quant_method": "fp8" }),
                "unsupported quantization method 'fp8'",
            ),
            (
                "mxfp4 method",
                serde_json::json!({ "quant_method": "mxfp4" }),
                "unsupported quantization method 'mxfp4'",
            ),
            (
                "bitsandbytes method",
                serde_json::json!({ "quant_method": "bitsandbytes", "load_in_4bit": true }),
                "unsupported quantization method 'bitsandbytes'",
            ),
            (
                "ct float-quantized",
                serde_json::json!({
                    "quant_method": "compressed-tensors",
                    "format": "float-quantized",
                    "config_groups": { "group_0": { "weights": { "num_bits": 8, "type": "float" } } }
                }),
                "float-quantized",
            ),
            (
                "ct activation quant W8A8",
                serde_json::json!({
                    "quant_method": "compressed-tensors",
                    "format": "pack-quantized",
                    "config_groups": {
                        "group_0": {
                            "input_activations": { "num_bits": 8, "type": "int" },
                            "weights": { "num_bits": 8, "type": "int", "strategy": "channel", "symmetric": true }
                        }
                    }
                }),
                "activation quantization",
            ),
            (
                "ct asymmetric 8-bit",
                serde_json::json!({
                    "quant_method": "compressed-tensors",
                    "format": "pack-quantized",
                    "config_groups": {
                        "group_0": {
                            "weights": { "num_bits": 8, "type": "int", "strategy": "group", "group_size": 128, "symmetric": false }
                        }
                    }
                }),
                "asymmetric 8-bit",
            ),
            (
                "ct dynamic actorder",
                serde_json::json!({
                    "quant_method": "compressed-tensors",
                    "format": "pack-quantized",
                    "config_groups": {
                        "group_0": {
                            "weights": { "num_bits": 4, "type": "int", "strategy": "group", "group_size": 128, "actorder": "weight" }
                        }
                    }
                }),
                "actorder",
            ),
            (
                "gptq asymmetric",
                serde_json::json!({ "quant_method": "gptq", "bits": 4, "group_size": 128, "sym": false }),
                "sym=false",
            ),
            (
                "gptq desc_act",
                serde_json::json!({ "quant_method": "gptq", "bits": 4, "group_size": 128, "sym": true, "desc_act": true }),
                "desc_act=true",
            ),
            (
                "gptq mixed dynamic bits",
                serde_json::json!({
                    "quant_method": "gptq",
                    "bits": 4,
                    "group_size": 128,
                    "sym": true,
                    "dynamic": { "+:.*mlp.*": { "bits": 8 } }
                }),
                "dynamic override",
            ),
            (
                "awq gemv",
                serde_json::json!({ "quant_method": "awq", "bits": 4, "group_size": 128, "version": "gemv" }),
                "AWQ version 'gemv'",
            ),
            (
                "unsupported 4-bit group_size=16",
                serde_json::json!({ "quant_method": "awq", "bits": 4, "group_size": 16, "version": "gemm" }),
                "group_size=16",
            ),
            (
                "autoround mxfp4",
                serde_json::json!({ "quant_method": "auto-round", "bits": 4, "data_type": "mx_fp4" }),
                "data_type='mx_fp4'",
            ),
        ];

        for (label, qcfg, needle) in bad_cases {
            let err = cfg_with_quant(qcfg.clone()).expect_err(label);
            assert!(
                err.to_string().contains(needle),
                "case '{label}' error '{}' did not contain '{needle}'",
                err
            );
        }
    }

    #[test]
    fn test_standalone_quantize_config_fallback_and_g_idx_validation() {
        let tmp = std::env::temp_dir().join(format!("w34_qcfg_fallback_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&tmp);
        std::fs::create_dir_all(&tmp).unwrap();
        std::fs::write(
            tmp.join("config.json"),
            serde_json::to_vec(&serde_json::json!({
                "model_type": "qwen3_5_text",
                "hidden_size": 1024,
                "num_hidden_layers": 2,
                "num_attention_heads": 8,
                "num_key_value_heads": 2,
                "head_dim": 128,
                "vocab_size": 1024,
                "layer_types": ["linear_attention", "full_attention"],
            }))
            .unwrap(),
        )
        .unwrap();
        std::fs::write(
            tmp.join("quantize_config.json"),
            serde_json::to_vec(&serde_json::json!({
                "bits": 4,
                "group_size": 64,
                "zero_point": true,
                "version": "gemm"
            }))
            .unwrap(),
        )
        .unwrap();
        let cfg = ModelConfig::from_dir(&tmp).unwrap();
        assert_eq!(
            cfg.quant,
            QuantScheme::Awq {
                bits: 4,
                group_size: 64,
                zero_point: true,
            }
        );
        let _ = std::fs::remove_dir_all(&tmp);

        // Static sequential & all-zero g_idx succeed; permuted g_idx fails fast.
        let seq_g: Vec<u32> = (0..128).map(|k| (k / 32) as u32).collect();
        validate_static_gptq_g_idx(&seq_g, 128, 32, "blk.g_idx").unwrap();
        validate_static_gptq_g_idx(&[0u32; 128], 128, 32, "blk.g_idx").unwrap();
        let mut perm_g = seq_g;
        perm_g.swap(0, 32);
        let err = validate_static_gptq_g_idx(&perm_g, 128, 32, "blk.g_idx").unwrap_err();
        assert!(err.to_string().contains("non-sequential g_idx"));
    }

    #[test]
    fn test_awq_and_u8b128_unpack_and_dequant_roundtrip() {
        const AWQ_ORDER: [usize; 8] = [0, 2, 4, 6, 1, 3, 5, 7];
        let size_k = 64usize;
        let size_n = 16usize;
        let group_size = 32usize;
        let num_groups = size_k / group_size;
        let n8 = size_n / 8;

        let mut raw_q = vec![0u8; size_k * size_n];
        for k in 0..size_k {
            for n in 0..size_n {
                raw_q[k * size_n + n] = ((k * 7 + n * 3 + 5) & 0xF) as u8;
            }
        }
        let mut raw_zp = vec![0u8; num_groups * size_n];
        for g in 0..num_groups {
            for n in 0..size_n {
                raw_zp[g * size_n + n] = ((g * 5 + n * 11 + 1) & 0xF) as u8;
            }
        }
        let mut scales_n_kg = vec![0u16; size_n * num_groups];
        for n in 0..size_n {
            for g in 0..num_groups {
                scales_n_kg[n * num_groups + g] = f32_to_f16(0.0625 * ((n + g + 1) as f32));
            }
        }

        // Pack in AutoAWQ [K, N/8] and [K/g, N/8] interleaved order
        let mut awq_qw = vec![0u32; size_k * n8];
        for k in 0..size_k {
            for np in 0..n8 {
                let mut w = 0u32;
                for (slot, &c) in AWQ_ORDER.iter().enumerate() {
                    w |= (raw_q[k * size_n + np * 8 + c] as u32) << (slot * 4);
                }
                awq_qw[k * n8 + np] = w;
            }
        }
        let mut awq_qz = vec![0u32; num_groups * n8];
        for g in 0..num_groups {
            for np in 0..n8 {
                let mut w = 0u32;
                for (slot, &c) in AWQ_ORDER.iter().enumerate() {
                    w |= (raw_zp[g * size_n + np * 8 + c] as u32) << (slot * 4);
                }
                awq_qz[g * n8 + np] = w;
            }
        }

        let unpacked_w = unpack_awq_qweight_k_n8_to_n_major(&awq_qw, size_k, size_n);
        let unpacked_z = unpack_awq_qzeros_kg_n8_to_n_major(&awq_qz, size_k, size_n, group_size);
        let deq = dequant_u4_zp_n_major_to_f16(
            &unpacked_w,
            &scales_n_kg,
            &unpacked_z,
            size_n,
            size_k,
            group_size,
        );
        for n in 0..size_n {
            for k in 0..size_k {
                let g = k / group_size;
                let q = raw_q[k * size_n + n] as i32;
                let z = raw_zp[g * size_n + n] as i32;
                let s = f16_to_f32(scales_n_kg[n * num_groups + g]);
                let expected = f32_to_f16(((q - z) as f32) * s);
                assert_eq!(deq[n * size_k + k], expected, "mismatch at n={n}, k={k}");
            }
        }

        // U8B128 GPTQ [K/4, N] -> [N, K/4] -> FP16 roundtrip
        let k4 = size_k / 4;
        let mut gptq_u8 = vec![0u32; k4 * size_n];
        for kw in 0..k4 {
            for n in 0..size_n {
                let mut w = 0u32;
                for b in 0..4 {
                    let k = kw * 4 + b;
                    let u = ((k * 13 + n * 17) & 0xFF) as u32;
                    w |= u << (b * 8);
                }
                gptq_u8[kw * size_n + n] = w;
            }
        }
        let n_maj_u8 = transpose_gptq_qweight_k4_n_to_n_major(&gptq_u8, size_k, size_n);
        let deq_u8 =
            dequant_u8b128_n_major_to_f16(&n_maj_u8, &scales_n_kg, size_n, size_k, group_size);
        for n in 0..size_n {
            for k in 0..size_k {
                let g = k / group_size;
                let q = (((k * 13 + n * 17) & 0xFF) as i32) - 128;
                let s = f16_to_f32(scales_n_kg[n * num_groups + g]);
                assert_eq!(deq_u8[n * size_k + k], f32_to_f16((q as f32) * s));
            }
        }
    }

    #[test]
    fn test_model_config_rejects_unsupported_rope_scaling_and_biases() {
        let base_json = serde_json::json!({
            "hidden_size": 2560,
            "intermediate_size": 6912,
            "num_attention_heads": 20,
            "num_key_value_heads": 4,
            "num_hidden_layers": 28,
            "vocab_size": 152064
        });

        // Valid base parses cleanly
        ModelConfig::from_json(&base_json).unwrap();

        // rope_scaling: default is allowed
        let mut ok_rope = base_json.clone();
        ok_rope["rope_scaling"] = serde_json::json!({ "type": "default" });
        ModelConfig::from_json(&ok_rope).unwrap();

        // rope_scaling: yarn / dynamic fails fast
        let mut bad_rope = base_json.clone();
        bad_rope["rope_scaling"] = serde_json::json!({ "type": "yarn" });
        let err = ModelConfig::from_json(&bad_rope).unwrap_err();
        assert!(
            err.to_string()
                .contains("unsupported rope_scaling type='yarn'")
        );

        // attention_bias=true fails fast
        let mut bad_bias = base_json.clone();
        bad_bias["attention_bias"] = serde_json::json!(true);
        let err = ModelConfig::from_json(&bad_bias).unwrap_err();
        assert!(err.to_string().contains("unsupported attention_bias=true"));

        // linear_key_head_dim != 128 fails fast
        let mut bad_gdn = base_json.clone();
        bad_gdn["linear_key_head_dim"] = serde_json::json!(64);
        let err = ModelConfig::from_json(&bad_gdn).unwrap_err();
        assert!(
            err.to_string()
                .contains("unsupported linear_key_head_dim=64")
        );
    }
}
