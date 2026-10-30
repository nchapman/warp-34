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

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum QuantScheme {
    Unquantized,
    CompressedTensors {
        num_bits: usize,
        group_size: usize,
        symmetric: bool,
        ignore: Vec<String>,
    },
    Gptq {
        bits: usize,
        group_size: usize,
        sym: bool,
        desc_act: bool,
    },
    Awq {
        bits: usize,
        group_size: usize,
    },
    AutoRound {
        bits: usize,
        group_size: usize,
        sym: bool,
    },
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LayerKind {
    LinearAttention,
    FullAttention,
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
}

impl ModelConfig {
    pub fn from_dir(dir: &Path) -> io::Result<Self> {
        let cfg_text = fs::read_to_string(dir.join("config.json"))?;
        let root: Json = serde_json::from_str(&cfg_text).map_err(|e| {
            io::Error::new(io::ErrorKind::InvalidData, format!("bad config.json: {e}"))
        })?;
        Self::from_json(&root)
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
        })
    }

    fn parse_quant(qcfg: Option<&Json>) -> io::Result<QuantScheme> {
        let Some(q) = qcfg else {
            return Ok(QuantScheme::Unquantized);
        };
        let method = q.get("quant_method").and_then(|v| v.as_str()).unwrap_or("");
        let scheme = match method {
            "compressed-tensors" => {
                let weights = q
                    .get("config_groups")
                    .and_then(|cg| cg.get("group_0"))
                    .and_then(|g0| g0.get("weights"));
                let num_bits = weights
                    .and_then(|w| w.get("num_bits"))
                    .and_then(|v| v.as_u64())
                    .unwrap_or(4) as usize;
                let group_size = weights
                    .and_then(|w| w.get("group_size"))
                    .and_then(|v| v.as_u64())
                    .unwrap_or(32) as usize;
                let symmetric = weights
                    .and_then(|w| w.get("symmetric"))
                    .and_then(|v| v.as_bool())
                    .unwrap_or(true);
                let ignore = q
                    .get("ignore")
                    .and_then(|v| v.as_array())
                    .map(|arr| {
                        arr.iter()
                            .filter_map(|x| x.as_str().map(String::from))
                            .collect()
                    })
                    .unwrap_or_default();
                QuantScheme::CompressedTensors {
                    num_bits,
                    group_size,
                    symmetric,
                    ignore,
                }
            }
            "gptq" | "gptq_marlin" => QuantScheme::Gptq {
                bits: q.get("bits").and_then(|v| v.as_u64()).unwrap_or(4) as usize,
                group_size: q.get("group_size").and_then(|v| v.as_u64()).unwrap_or(128) as usize,
                sym: q.get("sym").and_then(|v| v.as_bool()).unwrap_or(true),
                desc_act: q.get("desc_act").and_then(|v| v.as_bool()).unwrap_or(false),
            },
            "awq" | "awq_marlin" => QuantScheme::Awq {
                bits: q.get("bits").and_then(|v| v.as_u64()).unwrap_or(4) as usize,
                group_size: q.get("group_size").and_then(|v| v.as_u64()).unwrap_or(128) as usize,
            },
            "auto-round" | "autoround" => QuantScheme::AutoRound {
                bits: q.get("bits").and_then(|v| v.as_u64()).unwrap_or(4) as usize,
                group_size: q.get("group_size").and_then(|v| v.as_u64()).unwrap_or(128) as usize,
                sym: q.get("sym").and_then(|v| v.as_bool()).unwrap_or(true),
            },
            _ => QuantScheme::Unquantized,
        };
        match &scheme {
            QuantScheme::CompressedTensors {
                num_bits: b,
                group_size: g,
                ..
            }
            | QuantScheme::Gptq {
                bits: b,
                group_size: g,
                ..
            }
            | QuantScheme::Awq {
                bits: b,
                group_size: g,
            }
            | QuantScheme::AutoRound {
                bits: b,
                group_size: g,
                ..
            } => {
                if *b != 4 || *g == 0 {
                    return Err(io::Error::new(
                        io::ErrorKind::InvalidData,
                        format!(
                            "unsupported quantization width bits={b}, group_size={g} (expected 4-bit)"
                        ),
                    ));
                }
            }
            QuantScheme::Unquantized => {}
        }
        Ok(scheme)
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

/// Transposes a K-major `[K/8, N]` GPTQ `qweight` tensor into N-major `[N, K/8]` `u32` words.
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
}
