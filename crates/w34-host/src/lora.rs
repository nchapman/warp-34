//! Multi-LoRA (`PEFT`) adapter configuration, safetensors loader, TP sharding,
//! Hadamard-basis pre-rotation, and per-rank GPU execution state.
//!
//! Donor reference: `vLLM 0.31.0` (`vllm/lora/peft_helper.py`, `vllm/lora/models.py`,
//! `vllm/model_executor/models/qwen3_5.py`, `qwen3_next.py`).

use crate::{
    cuda::{f16_to_f32, f32_to_f16},
    loader::SafetensorsArchive,
};
use serde::Deserialize;
use std::{
    collections::{BTreeMap, HashMap},
    fs, io,
    path::{Path, PathBuf},
};

#[cfg(feature = "cuda")]
use crate::{
    cuda::{CudaStream, DeviceArena, DeviceBuffer, Result, check_rc},
    loader::ModelConfig,
    model::{LinearProj, loader_err, loader_err_msg},
};
#[cfg(feature = "cuda")]
use core::{
    ffi::c_void,
    sync::atomic::{AtomicI32, AtomicU64, Ordering},
};
#[cfg(feature = "cuda")]
use w34_kernels::ffi::{self, W34Exec};

/// Maximum padded LoRA rank per projection (`kLoraMaxRankPad` in `op_lora.cu`).
pub const MAX_LORA_RANK_PAD: usize = 512;

/// Maximum number of simultaneously loaded LoRA adapters (`64`, bitmask in `u64`).
pub const MAX_LORA_ADAPTERS: usize = 62;

/// Sentinel `lora_id` indicating the unadapted base model (`-1`).
pub const BASE_MODEL_LORA_ID: i32 = -1;

/// Parsed CLI/engine LoRA adapter specification (`--lora-modules` / `--lora`).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct LoraModuleSpec {
    /// Public adapter model ID exposed in `/v1/models` and matched by request `"model"`.
    pub name: String,
    /// Original path or Hugging Face repo spec (`owner/repo`, `owner/repo/subfolder`, or local path).
    pub path: String,
    /// Optional git revision (`@rev` suffix if provided).
    pub revision: Option<String>,
    /// Resolved local directory containing `adapter_config.json` and `adapter_model.safetensors`.
    pub resolved_dir: Option<PathBuf>,
}

#[derive(Deserialize)]
struct JsonLoraSpec {
    name: String,
    path: String,
    #[serde(default)]
    revision: Option<String>,
}

impl LoraModuleSpec {
    /// Parses a `--lora-modules` / `--lora` argument in `vLLM` format:
    /// - `name=path[@revision]`
    /// - JSON `{"name": "...", "path": "..."}`
    /// - Shorthand `path[@revision]` (uses `path` without `@revision` as `name`)
    pub fn parse(raw: &str) -> io::Result<Self> {
        let trimmed = raw.trim();
        if trimmed.is_empty() {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "empty --lora-modules specification",
            ));
        }
        if trimmed.starts_with('{') {
            let parsed: JsonLoraSpec = serde_json::from_str(trimmed).map_err(|e| {
                io::Error::new(
                    io::ErrorKind::InvalidInput,
                    format!("invalid JSON in --lora-modules '{trimmed}': {e}"),
                )
            })?;
            let (clean_path, rev) = split_spec_revision(parsed.path.trim());
            let revision = parsed.revision.or(rev);
            return Self::new_validated(parsed.name.trim(), clean_path, revision);
        }
        if let Some((left, right)) = trimmed.split_once('=') {
            let (clean_path, rev) = split_spec_revision(right.trim());
            return Self::new_validated(left.trim(), clean_path, rev);
        }
        let (clean_path, rev) = split_spec_revision(trimmed);
        Self::new_validated(clean_path, clean_path, rev)
    }

    fn new_validated(name: &str, path: &str, revision: Option<String>) -> io::Result<Self> {
        if name.is_empty() || path.is_empty() {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "LoRA adapter name and path must be non-empty",
            ));
        }
        if let Some(ref rev) = revision
            && rev.trim().is_empty()
        {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "LoRA adapter @revision suffix must be non-empty",
            ));
        }
        Ok(Self {
            name: name.to_string(),
            path: path.to_string(),
            revision,
            resolved_dir: None,
        })
    }
}

fn split_spec_revision(spec: &str) -> (&str, Option<String>) {
    if Path::new(spec).exists() {
        return (spec, None);
    }
    if let Some((base, rev)) = spec.rsplit_once('@') {
        (base, Some(rev.to_string()))
    } else {
        (spec, None)
    }
}

/// Splits a Hugging Face LoRA repo specification (`owner/repo` or `owner/repo/sub/folder`)
/// into `("owner/repo", Optional("sub/folder"))`.
pub fn split_hf_repo_and_subfolder(spec: &str) -> io::Result<(String, Option<String>)> {
    let normalized = spec
        .strip_prefix("models--")
        .map(|s| s.replace("--", "/"))
        .unwrap_or_else(|| spec.to_string());
    let parts: Vec<&str> = normalized.split('/').collect();
    if parts.len() < 2 {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            format!(
                "expected a local adapter directory or Hugging Face repo ID (owner/repo[/subfolder]), got '{spec}'"
            ),
        ));
    }
    for part in &parts[..2] {
        if part.is_empty()
            || part.starts_with(['.', '-'])
            || part.ends_with(['.', '-'])
            || part.contains("..")
            || part.contains("--")
            || !part
                .bytes()
                .all(|b| b.is_ascii_alphanumeric() || matches!(b, b'_' | b'-' | b'.'))
        {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                format!("invalid Hugging Face repository ID in '{spec}'"),
            ));
        }
    }
    let repo_id = format!("{}/{}", parts[0], parts[1]);
    if parts.len() == 2 {
        return Ok((repo_id, None));
    }
    let sub_parts = &parts[2..];
    for p in sub_parts {
        if p.is_empty()
            || *p == "."
            || *p == ".."
            || !p
                .bytes()
                .all(|b| b.is_ascii_alphanumeric() || matches!(b, b'_' | b'-' | b'.'))
        {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                format!("unsafe or invalid LoRA subfolder in '{spec}'"),
            ));
        }
    }
    Ok((repo_id, Some(sub_parts.join("/"))))
}

/// Resolves a LoRA adapter directory from a local path or the local Hugging Face cache.
pub fn resolve_cached_or_local_lora_dir(spec: &str, revision: Option<&str>) -> io::Result<PathBuf> {
    let local = Path::new(spec);
    if local.exists() {
        if !local.is_dir() {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                format!("LoRA path '{}' is not a directory", local.display()),
            ));
        }
        return Ok(local.to_path_buf());
    }
    if local.is_absolute() || spec.starts_with('.') || spec.starts_with('~') || spec.contains('\\')
    {
        return Err(io::Error::new(
            io::ErrorKind::NotFound,
            format!("local LoRA adapter directory '{spec}' does not exist"),
        ));
    }
    let (repo_id, subfolder) = split_hf_repo_and_subfolder(spec)?;
    let repo_cache =
        crate::hub::cache_dir().join(format!("models--{}", repo_id.replace('/', "--")));
    let snapshots = repo_cache.join("snapshots");
    if !snapshots.is_dir() {
        return Err(io::Error::new(
            io::ErrorKind::NotFound,
            format!(
                "LoRA adapter '{spec}' not found in local Hugging Face cache ({})",
                snapshots.display()
            ),
        ));
    }
    let resolve_candidate = |snap: PathBuf| -> PathBuf {
        match &subfolder {
            Some(sub) => snap.join(sub),
            None => snap,
        }
    };
    if let Some(rev) = revision {
        if rev.is_empty()
            || rev.starts_with('-')
            || rev.ends_with('/')
            || rev.ends_with(".lock")
            || rev.contains("..")
            || rev.contains("@{")
            || rev.chars().any(|c| {
                c.is_control()
                    || c.is_whitespace()
                    || matches!(c, '~' | '^' | ':' | '?' | '*' | '[' | '\\')
            })
            || rev
                .split('/')
                .any(|seg| seg.is_empty() || seg == "." || seg == "..")
        {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                format!("invalid LoRA revision '{rev}'"),
            ));
        }
        let direct = resolve_candidate(snapshots.join(rev));
        if direct.join("adapter_config.json").is_file() {
            return Ok(direct);
        }
        let ref_file = repo_cache.join("refs").join(rev);
        if let Ok(commit) = fs::read_to_string(&ref_file) {
            let sha = commit.trim();
            if sha.len() == 40 && sha.bytes().all(|b| b.is_ascii_hexdigit()) {
                let cand = resolve_candidate(snapshots.join(sha));
                if cand.join("adapter_config.json").is_file() {
                    return Ok(cand);
                }
            }
        }
        return Err(io::Error::new(
            io::ErrorKind::NotFound,
            format!("LoRA adapter '{spec}' revision '{rev}' not found in local Hugging Face cache"),
        ));
    }
    let main_ref = repo_cache.join("refs").join("main");
    if let Ok(commit) = fs::read_to_string(&main_ref) {
        let sha = commit.trim();
        if sha.len() == 40 && sha.bytes().all(|b| b.is_ascii_hexdigit()) {
            let cand = resolve_candidate(snapshots.join(sha));
            if cand.join("adapter_config.json").is_file() {
                return Ok(cand);
            }
        }
    }
    let mut entries: Vec<PathBuf> = fs::read_dir(&snapshots)?
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| p.is_dir())
        .collect();
    entries.sort();
    for snap in entries.into_iter().rev() {
        let cand = resolve_candidate(snap);
        if cand.join("adapter_config.json").is_file() {
            return Ok(cand);
        }
    }
    Err(io::Error::new(
        io::ErrorKind::NotFound,
        format!("no complete cached LoRA adapter snapshot found for '{spec}'"),
    ))
}

/// Mixes `lora_id` (`0..62`) into the root token of a `MambaRadixCache` key so that
/// prefix KV pages and GDN `StateBundle` checkpoints never alias across different adapters
/// (or between an adapter and the base model `lora_id == -1`), while requests targeting
/// the same adapter share full prefix-cache hits.
#[inline]
pub fn salt_radix_key_for_lora(tokens: &[i32], lora_id: i32) -> Vec<i32> {
    let mut out = tokens.to_vec();
    salt_radix_key_in_place(&mut out, lora_id);
    out
}

/// In-place variant of [`salt_radix_key_for_lora`].
/// If `tokens[0]` already carries a valid LoRA tag (`bits 30..31 == 0b10` and `bits 24..29 in 1..=MAX_LORA_ADAPTERS`),
/// it is preserved unchanged so a per-request salted key is never overwritten by a default `MambaRadixCache` adapter ID.
#[inline]
pub fn salt_radix_key_in_place(tokens: &mut [i32], lora_id: i32) {
    if lora_id < 0 || tokens.is_empty() {
        return;
    }
    let tok0 = tokens[0] as u32;
    let existing_tag = (tok0 >> 24) & 0x3F;
    if (tok0 & 0xC000_0000) == 0x8000_0000 && (1..=MAX_LORA_ADAPTERS as u32).contains(&existing_tag)
    {
        return;
    }
    let low = tok0 & 0x00FF_FFFF;
    let tag = (((lora_id as u32) + 1) & 0x3F) << 24;
    // Bit 31 = 1, bits 24..29 = lora_id + 1 (1..=62), placing the salted token in
    // [-2_130_706_432, -1_090_519_041], strictly disjoint from >= 0 vocab tokens and
    // [-1_073_742_823, -1000] multimodal image hashes.
    tokens[0] = (0x8000_0000u32 | tag | low) as i32;
}

/// PEFT `adapter_config.json` metadata (`vllm/lora/peft_helper.py`).
#[derive(Clone, Debug, Deserialize)]
pub struct PeftConfig {
    pub r: usize,
    pub lora_alpha: f32,
    #[serde(default)]
    pub peft_type: Option<String>,
    #[serde(default)]
    pub bias: Option<String>,
    #[serde(default)]
    pub use_rslora: bool,
    #[serde(default)]
    pub use_dora: bool,
    #[serde(default)]
    pub fan_in_fan_out: bool,
    #[serde(default)]
    pub rank_pattern: HashMap<String, usize>,
    #[serde(default)]
    pub alpha_pattern: HashMap<String, f32>,
    #[serde(default)]
    pub modules_to_save: Option<Vec<String>>,
    #[serde(default)]
    pub base_model_name_or_path: Option<String>,
}

impl PeftConfig {
    pub fn from_dir(dir: &Path) -> io::Result<Self> {
        let path = dir.join("adapter_config.json");
        let bytes = fs::read(&path).map_err(|e| {
            io::Error::new(e.kind(), format!("failed to read {}: {e}", path.display()))
        })?;
        let cfg: Self = serde_json::from_slice(&bytes).map_err(|e| {
            io::Error::new(
                io::ErrorKind::InvalidData,
                format!("invalid {}: {e}", path.display()),
            )
        })?;
        cfg.validate()?;
        Ok(cfg)
    }

    pub fn validate(&self) -> io::Result<()> {
        if let Some(ref kind) = self.peft_type
            && !kind.eq_ignore_ascii_case("LORA")
        {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!("unsupported peft_type '{kind}'; only 'LORA' is supported"),
            ));
        }
        if self.use_dora {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "DoRA adapters (use_dora=true) are not supported",
            ));
        }
        if self.fan_in_fan_out {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "Conv1D-style LoRA adapters (fan_in_fan_out=true) are not supported",
            ));
        }
        if let Some(ref bias) = self.bias
            && !bias.eq_ignore_ascii_case("none")
        {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!("LoRA bias mode '{bias}' is not supported; expected 'none'"),
            ));
        }
        if let Some(ref mods) = self.modules_to_save
            && !mods.is_empty()
        {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "PEFT modules_to_save is not supported",
            ));
        }
        if self.r == 0 || self.r > MAX_LORA_RANK_PAD {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!("LoRA rank r={} must be in 1..={MAX_LORA_RANK_PAD}", self.r),
            ));
        }
        if !self.lora_alpha.is_finite() || self.lora_alpha <= 0.0 {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!("LoRA lora_alpha={} must be finite and > 0", self.lora_alpha),
            ));
        }
        for &r_val in self.rank_pattern.values() {
            if r_val == 0 || r_val > MAX_LORA_RANK_PAD {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    format!("rank_pattern rank {r_val} must be in 1..={MAX_LORA_RANK_PAD}"),
                ));
            }
        }
        for &a_val in self.alpha_pattern.values() {
            if !a_val.is_finite() || a_val <= 0.0 {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    format!("alpha_pattern alpha {a_val} must be finite and > 0"),
                ));
            }
        }
        Ok(())
    }

    /// Computes the PEFT scaling factor `alpha / r` (or `alpha / sqrt(r)` for `use_rslora`),
    /// honoring `rank_pattern` and `alpha_pattern` overrides (`model.layers.{l}.{suffix}`,
    /// `layers.{l}.{suffix}`, `{suffix}`, or leaf module name e.g. `q_proj`).
    pub fn scaling_for_module(
        &self,
        layer_id: Option<usize>,
        module_suffix: &str,
        actual_r: usize,
    ) -> f32 {
        let mut alpha = self.lora_alpha;
        let mut r = actual_r.max(1);
        let leaf = module_suffix
            .rsplit_once('.')
            .map_or(module_suffix, |(_, l)| l);
        if let Some(l) = layer_id {
            let model_key = format!("model.layers.{l}.{module_suffix}");
            let full_key = format!("layers.{l}.{module_suffix}");
            if let Some(&pat_r) = self
                .rank_pattern
                .get(&model_key)
                .or_else(|| self.rank_pattern.get(&full_key))
                .or_else(|| self.rank_pattern.get(module_suffix))
                .or_else(|| self.rank_pattern.get(leaf))
            {
                r = pat_r.max(1);
            }
            if let Some(&pat_a) = self
                .alpha_pattern
                .get(&model_key)
                .or_else(|| self.alpha_pattern.get(&full_key))
                .or_else(|| self.alpha_pattern.get(module_suffix))
                .or_else(|| self.alpha_pattern.get(leaf))
            {
                alpha = pat_a;
            }
        } else {
            if let Some(&pat_r) = self
                .rank_pattern
                .get(module_suffix)
                .or_else(|| self.rank_pattern.get(leaf))
            {
                r = pat_r.max(1);
            }
            if let Some(&pat_a) = self
                .alpha_pattern
                .get(module_suffix)
                .or_else(|| self.alpha_pattern.get(leaf))
            {
                alpha = pat_a;
            }
        }
        if self.use_rslora {
            alpha / (r as f32).sqrt()
        } else {
            alpha / (r as f32)
        }
    }
}

/// Supported Qwen 3.5 / 3.6 / 3.8 decoder linear module targets for PEFT LoRA.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum TargetLoraModule {
    // Full Attention
    QProj,
    KProj,
    VProj,
    QkvProj,
    OProj,
    // GDN Linear Attention
    InProjQkv,
    InProjZ,
    InProjQkvz,
    InProjB,
    InProjA,
    InProjBa,
    OutProj,
    // Dense FFN
    GateProj,
    UpProj,
    GateUpProj,
    DownProj,
    // MoE Shared Expert FFN
    SharedGateProj,
    SharedUpProj,
    SharedGateUpProj,
    SharedDownProj,
    // Output Head
    LmHead,
}

impl TargetLoraModule {
    pub fn from_suffix(suffix: &str) -> Option<Self> {
        match suffix {
            "self_attn.q_proj" => Some(Self::QProj),
            "self_attn.k_proj" => Some(Self::KProj),
            "self_attn.v_proj" => Some(Self::VProj),
            "self_attn.qkv_proj" => Some(Self::QkvProj),
            "self_attn.o_proj" => Some(Self::OProj),
            "linear_attn.in_proj_qkv" => Some(Self::InProjQkv),
            "linear_attn.in_proj_z" => Some(Self::InProjZ),
            "linear_attn.in_proj_qkvz" => Some(Self::InProjQkvz),
            "linear_attn.in_proj_b" => Some(Self::InProjB),
            "linear_attn.in_proj_a" => Some(Self::InProjA),
            "linear_attn.in_proj_ba" => Some(Self::InProjBa),
            "linear_attn.out_proj" => Some(Self::OutProj),
            "mlp.gate_proj" => Some(Self::GateProj),
            "mlp.up_proj" => Some(Self::UpProj),
            "mlp.gate_up_proj" => Some(Self::GateUpProj),
            "mlp.down_proj" => Some(Self::DownProj),
            "mlp.shared_expert.gate_proj" => Some(Self::SharedGateProj),
            "mlp.shared_expert.up_proj" => Some(Self::SharedUpProj),
            "mlp.shared_expert.gate_up_proj" => Some(Self::SharedGateUpProj),
            "mlp.shared_expert.down_proj" => Some(Self::SharedDownProj),
            "lm_head" => Some(Self::LmHead),
            _ => None,
        }
    }
}

/// Host-side loaded `(lora_A, lora_B)` weight pair for one module slice, with `lora_B`
/// pre-multiplied by the PEFT scaling factor `alpha / r` (or `alpha / sqrt(r)`).
#[derive(Clone, Debug)]
pub struct LoraSliceWeight {
    pub r: usize,
    pub in_features: usize,
    pub out_features: usize,
    /// Row-major `[r, in_features]` in `FP32`.
    pub a_f32: Vec<f32>,
    /// Row-major `[out_features, r]` in `FP32`, pre-scaled by `scale`.
    pub b_f32: Vec<f32>,
}

/// Host-side parsed LoRA adapter ready to be sharded and uploaded to one or more TP ranks.
#[derive(Clone, Debug)]
pub struct LoraAdapterHost {
    pub name: String,
    pub raw_path: String,
    pub dir: PathBuf,
    pub config: PeftConfig,
    /// Per-layer `(layer_id, TargetLoraModule)` weights.
    pub layer_weights: BTreeMap<(usize, TargetLoraModule), LoraSliceWeight>,
    /// Optional `lm_head` LoRA weight.
    pub lm_head_weight: Option<LoraSliceWeight>,
    /// Optional per-adapter damped-Euler `midloop` `(start_layer, end_layer, passes)` specs
    /// parsed from `decision_config.json` (e.g. `GLiDE-9B` / `GLiDE-27B`).
    pub midloop: Vec<(usize, usize, usize)>,
    /// Whether this adapter configures full-prompt non-causal attention (`decision_config.json`).
    pub noncausal_full_attention: bool,
}

impl LoraAdapterHost {
    /// Loads a PEFT LoRA adapter from its resolved directory (containing `adapter_config.json`
    /// and `adapter_model.safetensors`).
    pub fn load_from_dir(
        name: impl Into<String>,
        raw_path: impl Into<String>,
        dir: &Path,
    ) -> io::Result<Self> {
        let name = name.into();
        let raw_path = raw_path.into();
        let config = PeftConfig::from_dir(dir)?;
        let st_path = dir.join("adapter_model.safetensors");
        if !st_path.is_file() {
            return Err(io::Error::new(
                io::ErrorKind::NotFound,
                format!("LoRA adapter '{}' is missing {}", name, st_path.display()),
            ));
        }
        let archive = SafetensorsArchive::open(&st_path)?;

        type LoraPairMap = BTreeMap<(Option<usize>, String), (Option<String>, Option<String>)>;
        let mut pairs: LoraPairMap = BTreeMap::new();
        for tensor_name in archive.tensor_names() {
            if matches!(
                tensor_name,
                "readout.weight" | "readout.bias" | "weight" | "bias"
            ) {
                continue;
            }
            let (layer_opt, module_suffix, is_b) = parse_peft_tensor_key(tensor_name)?;
            let entry = pairs.entry((layer_opt, module_suffix)).or_default();
            let slot = if is_b { &mut entry.1 } else { &mut entry.0 };
            if slot.is_some() {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    format!("LoRA adapter '{name}': duplicate tensor key '{tensor_name}'"),
                ));
            }
            *slot = Some(tensor_name.to_string());
        }
        if pairs.is_empty() {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!(
                    "LoRA adapter '{}' ({}) contains no lora_A/lora_B tensors",
                    name,
                    st_path.display()
                ),
            ));
        }

        let mut layer_weights = BTreeMap::new();
        let mut lm_head_weight = None;

        for ((layer_opt, suffix), (a_key_opt, b_key_opt)) in pairs {
            let a_key = a_key_opt.ok_or_else(|| {
                io::Error::new(
                    io::ErrorKind::InvalidData,
                    format!("LoRA adapter '{name}': missing lora_A.weight for '{suffix}'"),
                )
            })?;
            let b_key = b_key_opt.ok_or_else(|| {
                io::Error::new(
                    io::ErrorKind::InvalidData,
                    format!("LoRA adapter '{name}': missing lora_B.weight for '{suffix}'"),
                )
            })?;
            let target = TargetLoraModule::from_suffix(&suffix).ok_or_else(|| {
                io::Error::new(
                    io::ErrorKind::InvalidData,
                    format!("LoRA adapter '{name}': unsupported target module '{suffix}'"),
                )
            })?;

            let (a_shape, a_f32) = read_2d_tensor_f32(&archive, &a_key)?;
            let (b_shape, mut b_f32) = read_2d_tensor_f32(&archive, &b_key)?;
            let (r_a, in_features) = (a_shape[0], a_shape[1]);
            let (out_features, r_b) = (b_shape[0], b_shape[1]);
            if r_a != r_b || r_a == 0 || r_a > MAX_LORA_RANK_PAD {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    format!(
                        "LoRA adapter '{name}': rank mismatch on '{suffix}': lora_A shape {:?} vs lora_B shape {:?}",
                        a_shape, b_shape
                    ),
                ));
            }
            let scale = config.scaling_for_module(layer_opt, &suffix, r_a);
            for v in &mut b_f32 {
                *v *= scale;
            }
            let slice = LoraSliceWeight {
                r: r_a,
                in_features,
                out_features,
                a_f32,
                b_f32,
            };
            if let Some(layer_id) = layer_opt {
                layer_weights.insert((layer_id, target), slice);
            } else if target == TargetLoraModule::LmHead {
                lm_head_weight = Some(slice);
            } else {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    format!("LoRA adapter '{name}': module '{suffix}' is missing a layer index"),
                ));
            }
        }

        let (midloop, noncausal_full_attention) = {
            let dc_path = dir.join("decision_config.json");
            if dc_path.is_file() {
                let text = fs::read_to_string(&dc_path)?;
                let dc_val: serde_json::Value = serde_json::from_str(&text).map_err(|e| {
                    io::Error::new(
                        io::ErrorKind::InvalidData,
                        format!("LoRA adapter '{name}': bad {}: {e}", dc_path.display()),
                    )
                })?;
                let ml = if let Some(ml_val) = dc_val.get("midloop").filter(|v| !v.is_null()) {
                    crate::loader::ModelConfig::parse_midloop_specs(ml_val, usize::MAX)?
                } else {
                    Vec::new()
                };
                let noncausal = dc_val
                    .get("noncausal_full_attention")
                    .and_then(|v| v.as_bool())
                    .unwrap_or_else(|| {
                        dc_val
                            .get("attention_mode")
                            .and_then(|v| v.as_str())
                            .is_some_and(|m| m.eq_ignore_ascii_case("noncausal_full_attention"))
                    });
                (ml, noncausal)
            } else {
                (Vec::new(), false)
            }
        };

        Ok(Self {
            name,
            raw_path,
            dir: dir.to_path_buf(),
            config,
            layer_weights,
            lm_head_weight,
            midloop,
            noncausal_full_attention,
        })
    }

    pub fn get_layer_module(
        &self,
        layer_id: usize,
        module: TargetLoraModule,
    ) -> Option<&LoraSliceWeight> {
        self.layer_weights.get(&(layer_id, module))
    }
}

fn read_2d_tensor_f32(
    archive: &SafetensorsArchive,
    name: &str,
) -> io::Result<([usize; 2], Vec<f32>)> {
    let info = archive.tensor_info(name)?;
    if info.shape.len() != 2 {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!(
                "expected 2D LoRA tensor for '{name}', got shape {:?}",
                info.shape
            ),
        ));
    }
    let shape = [info.shape[0], info.shape[1]];
    let vals = match info.dtype.as_str() {
        "F32" => archive.read_f32(name)?,
        "F16" => {
            let bits = archive.read_f16_bits(name)?;
            bits.into_iter().map(f16_to_f32).collect()
        }
        "BF16" => {
            let bytes = archive.read_bytes(name)?;
            if !bytes.len().is_multiple_of(2) {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    format!("BF16 tensor '{name}' byte length is not a multiple of 2"),
                ));
            }
            bytes
                .as_chunks::<2>()
                .0
                .iter()
                .map(|&[b0, b1]| f32::from_bits((u16::from_le_bytes([b0, b1]) as u32) << 16))
                .collect()
        }
        other => {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!("unsupported LoRA tensor dtype '{other}' for '{name}'"),
            ));
        }
    };
    Ok((shape, vals))
}

/// Parses a PEFT safetensors key such as:
/// `base_model.model.model.language_model.layers.0.self_attn.o_proj.lora_A.weight`
/// or `base_model.model.lm_head.lora_B.weight`
/// into `(Option<layer_id>, canonical_module_suffix, is_lora_b)`.
pub fn parse_peft_tensor_key(key: &str) -> io::Result<(Option<usize>, String, bool)> {
    let (stem, is_b) = if let Some(s) = key
        .strip_suffix(".lora_A.weight")
        .or_else(|| key.strip_suffix(".lora_A.default.weight"))
    {
        (s, false)
    } else if let Some(s) = key
        .strip_suffix(".lora_B.weight")
        .or_else(|| key.strip_suffix(".lora_B.default.weight"))
    {
        (s, true)
    } else {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!(
                "unsupported tensor '{key}' in LoRA archive (expected *.lora_A.weight or *.lora_B.weight)"
            ),
        ));
    };

    if let Some(idx) = stem.find("layers.") {
        let prefix = &stem[..idx];
        if !prefix.is_empty() && !prefix.ends_with("model.") && !prefix.ends_with("language_model.")
        {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!("unsupported non-decoder layer prefix '{prefix}' in '{key}'"),
            ));
        }
        let after = &stem[idx + "layers.".len()..];
        let (layer_str, mod_suffix) = after.split_once('.').ok_or_else(|| {
            io::Error::new(
                io::ErrorKind::InvalidData,
                format!("malformed layer key '{key}'"),
            )
        })?;
        let layer_id: usize = layer_str.parse().map_err(|_| {
            io::Error::new(
                io::ErrorKind::InvalidData,
                format!("invalid layer index '{layer_str}' in '{key}'"),
            )
        })?;
        return Ok((Some(layer_id), mod_suffix.to_string(), is_b));
    }

    if stem == "lm_head" || stem.ends_with(".lm_head") {
        return Ok((None, "lm_head".to_string(), is_b));
    }

    Err(io::Error::new(
        io::ErrorKind::InvalidData,
        format!("unsupported non-decoder LoRA tensor key '{key}'"),
    ))
}

/// Registry of loaded LoRA adapters attached to a [` crate::model::Qwen3_5Engine `].
#[derive(Clone, Debug, Default)]
pub struct LoraAdapterRegistry {
    entries: Vec<LoraAdapterEntry>,
    by_name: HashMap<String, i32>,
}

/// Metadata for a single loaded LoRA adapter in [`LoraAdapterRegistry`].
#[derive(Clone, Debug)]
pub struct LoraAdapterEntry {
    pub lora_id: i32,
    pub name: String,
    pub root: String,
    pub decision_profile: Option<crate::decision::DecisionModelProfile>,
}

impl LoraAdapterRegistry {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    pub fn len(&self) -> usize {
        self.entries.len()
    }

    pub fn entries(&self) -> &[LoraAdapterEntry] {
        &self.entries
    }

    pub fn register(&mut self, name: String, root: String) -> io::Result<i32> {
        self.register_with_decision_profile(name, root, None)
    }

    pub fn register_with_decision_profile(
        &mut self,
        name: String,
        root: String,
        decision_profile: Option<crate::decision::DecisionModelProfile>,
    ) -> io::Result<i32> {
        if self.by_name.contains_key(&name) {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                format!("duplicate LoRA adapter name '{name}'"),
            ));
        }
        if self.entries.len() >= MAX_LORA_ADAPTERS {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                format!("cannot load more than {MAX_LORA_ADAPTERS} LoRA adapters"),
            ));
        }
        let id = self.entries.len() as i32;
        self.by_name.insert(name.clone(), id);
        self.entries.push(LoraAdapterEntry {
            lora_id: id,
            name,
            root,
            decision_profile,
        });
        Ok(id)
    }

    pub fn resolve_name(&self, model_name: &str) -> Option<i32> {
        self.by_name.get(model_name).copied()
    }

    pub fn decision_profile(&self, lora_id: i32) -> Option<&crate::decision::DecisionModelProfile> {
        if lora_id < 0 {
            return None;
        }
        self.entries
            .get(lora_id as usize)
            .and_then(|e| e.decision_profile.as_ref())
    }
}

/// Per-rank atomic runtime binding read by [`DeviceLoraStack::forward`] during eager execution
/// and CUDA graph capture.
#[cfg(feature = "cuda")]
pub struct RankLoraState {
    pub lowrank_ws_ptr: u64,
    pub lowrank_ws_elems: usize,
    pub token_lora_ids_ptr: AtomicU64,
    pub default_lora_id: AtomicI32,
}

#[cfg(feature = "cuda")]
impl RankLoraState {
    pub fn new(lowrank_ws_ptr: u64, lowrank_ws_elems: usize) -> Self {
        Self {
            lowrank_ws_ptr,
            lowrank_ws_elems,
            token_lora_ids_ptr: AtomicU64::new(0),
            default_lora_id: AtomicI32::new(BASE_MODEL_LORA_ID),
        }
    }

    #[inline]
    pub fn set_uniform(&self, lora_id: i32) {
        self.token_lora_ids_ptr.store(0, Ordering::Relaxed);
        self.default_lora_id.store(lora_id, Ordering::Relaxed);
    }

    #[inline]
    pub fn set_device_ids(&self, d_token_lora_ids_ptr: u64) {
        self.token_lora_ids_ptr
            .store(d_token_lora_ids_ptr, Ordering::Relaxed);
        self.default_lora_id
            .store(BASE_MODEL_LORA_ID, Ordering::Relaxed);
    }
}

/// Stacked GPU LoRA weights (`lora_A` and `lora_B`) attached to one [`LinearProj`].
#[cfg(feature = "cuda")]
#[derive(Clone)]
pub struct DeviceLoraStack {
    pub lora_a_ptr: u64,
    pub lora_b_ptr: u64,
    pub size_n: usize,
    pub size_k: usize,
    pub rank_pad: usize,
    pub num_adapters: usize,
    pub adapter_mask: u64,
    pub num_b_slices: u8,
    pub b_slices4: [i32; 12],
    pub state: std::sync::Arc<RankLoraState>,
}

#[cfg(feature = "cuda")]
impl DeviceLoraStack {
    #[inline]
    pub fn is_currently_noop(&self) -> bool {
        if self.adapter_mask == 0 {
            return true;
        }
        let d_ids_ptr = self.state.token_lora_ids_ptr.load(Ordering::Relaxed);
        let def_id = self.state.default_lora_id.load(Ordering::Relaxed);
        d_ids_ptr == 0
            && (def_id < 0
                || (def_id as usize) >= self.num_adapters
                || ((self.adapter_mask >> def_id) & 1) == 0)
    }

    #[inline]
    pub fn forward(&self, ex: &W34Exec, x_ptr: u64, y_inout_ptr: u64, rows: usize) -> Result<()> {
        self.forward_with_cublas(ex, 0, x_ptr, y_inout_ptr, rows)
    }

    #[inline]
    pub fn forward_with_cublas(
        &self,
        ex: &W34Exec,
        cublas_handle: usize,
        x_ptr: u64,
        y_inout_ptr: u64,
        rows: usize,
    ) -> Result<()> {
        if rows == 0 || self.adapter_mask == 0 {
            return Ok(());
        }
        let d_ids_ptr = self.state.token_lora_ids_ptr.load(Ordering::Relaxed);
        let def_id = self.state.default_lora_id.load(Ordering::Relaxed);
        if d_ids_ptr == 0
            && (def_id < 0
                || (def_id as usize) >= self.num_adapters
                || ((self.adapter_mask >> def_id) & 1) == 0)
        {
            return Ok(());
        }
        let needed_ws = rows.saturating_mul(self.rank_pad);
        if needed_ws > self.state.lowrank_ws_elems || self.state.lowrank_ws_ptr == 0 {
            return Err(loader_err_msg(format!(
                "LoRA low-rank workspace overflow: need {} f32 elems (rows={rows}, rank_pad={}), have {}",
                needed_ws, self.rank_pad, self.state.lowrank_ws_elems
            )));
        }
        let rows_i32 = i32::try_from(rows).map_err(|_| loader_err_msg("LoRA rows exceeds i32"))?;
        let size_n_i32 =
            i32::try_from(self.size_n).map_err(|_| loader_err_msg("LoRA size_n exceeds i32"))?;
        let size_k_i32 =
            i32::try_from(self.size_k).map_err(|_| loader_err_msg("LoRA size_k exceeds i32"))?;
        let rank_pad_i32 = i32::try_from(self.rank_pad)
            .map_err(|_| loader_err_msg("LoRA rank_pad exceeds i32"))?;
        let num_adapters_i32 = i32::try_from(self.num_adapters)
            .map_err(|_| loader_err_msg("LoRA num_adapters exceeds i32"))?;
        if d_ids_ptr == 0 && def_id >= 0 && cublas_handle != 0 && rows >= 8 && self.rank_pad >= 64 {
            return check_rc(
                // SAFETY: `cublas_handle` is bound to `ex.stream`; `x_ptr`, `y_inout_ptr`,
                // `lora_a_ptr`, `lora_b_ptr`, and `lowrank_ws_ptr` are valid device allocations.
                unsafe {
                    ffi::w34_cublas_lora_shrink_expand_f16(
                        cublas_handle,
                        x_ptr as *const c_void,
                        y_inout_ptr as *mut c_void,
                        self.lora_a_ptr as *const c_void,
                        self.lora_b_ptr as *const c_void,
                        self.state.lowrank_ws_ptr as *mut c_void,
                        def_id,
                        rows_i32,
                        size_n_i32,
                        size_k_i32,
                        rank_pad_i32,
                        i32::from(self.num_b_slices),
                        self.b_slices4.as_ptr(),
                    )
                },
                "w34_cublas_lora_shrink_expand_f16",
            );
        }
        check_rc(
            // SAFETY: `x_ptr` holds `[rows, size_k]` FP16, `y_inout_ptr` holds `[rows, size_n]` FP16,
            // `lora_a_ptr` holds `[num_adapters, rank_pad, size_k]` FP16, `lora_b_ptr` holds
            // `[num_adapters, size_n, rank_pad]` FP16, and `lowrank_ws_ptr` holds at least
            // `rows * rank_pad` FP32 elements.
            unsafe {
                ffi::w34_lora_shrink_expand_f16(
                    ex,
                    x_ptr as *const c_void,
                    self.lora_a_ptr as *const c_void,
                    self.lora_b_ptr as *const c_void,
                    d_ids_ptr as *const i32,
                    self.state.lowrank_ws_ptr as *mut f32,
                    y_inout_ptr as *mut c_void,
                    rows_i32,
                    size_n_i32,
                    size_k_i32,
                    rank_pad_i32,
                    num_adapters_i32,
                    def_id,
                    self.adapter_mask,
                )
            },
            "w34_lora_shrink_expand_f16",
        )
    }
}

/// Host-side packed LoRA matrices for a single `(layer, projection, tp_rank)` before GPU upload.
fn pad_hadamard_cols_f32(
    src: &[f32],
    rows: usize,
    logical_cols: usize,
    phys_cols: usize,
) -> Vec<f32> {
    if phys_cols <= logical_cols {
        return src.to_vec();
    }
    let mut out = vec![0.0f32; rows * phys_cols];
    for r in 0..rows {
        let s = r * logical_cols;
        let d = r * phys_cols;
        out[d..d + logical_cols].copy_from_slice(&src[s..s + logical_cols]);
    }
    out
}

fn hadamard_rows_f32(data: &mut [f32], cols: usize, block_size: usize) {
    assert!(
        block_size.is_power_of_two() && block_size > 0 && cols.is_multiple_of(block_size),
        "invalid Hadamard block_size={block_size} for cols={cols}"
    );
    let scale = (block_size as f32).sqrt().recip();
    for row in data.chunks_exact_mut(cols) {
        for blk in row.chunks_exact_mut(block_size) {
            let mut h = 1usize;
            while h < block_size {
                let step = h * 2;
                let mut i = 0usize;
                while i < block_size {
                    for j in i..i + h {
                        let a = blk[j];
                        let b = blk[j + h];
                        blk[j] = a + b;
                        blk[j + h] = a - b;
                    }
                    i += step;
                }
                h = step;
            }
            for x in blk.iter_mut() {
                *x *= scale;
            }
        }
    }
}

#[derive(Clone, Debug)]
pub struct PackedProjLoraHost {
    /// `[num_adapters, rank_pad, phys_k]` in `FP32` (before optional Hadamard rotation).
    pub a_f32: Vec<f32>,
    /// `[num_adapters, phys_n, rank_pad]` in `FP32`.
    pub b_f32: Vec<f32>,
    pub logical_n: usize,
    pub logical_k: usize,
    pub rank_pad: usize,
    pub num_adapters: usize,
    pub adapter_mask: u64,
    pub num_b_slices: u8,
    pub b_slices4: [i32; 12],
}

impl PackedProjLoraHost {
    /// Converts `a_f32` and `b_f32` to `FP16`, applying Hadamard pre-rotation to `a_f32`
    /// and zero-padding to `(phys_n, phys_k)` when wrapping a `LinearProj::Hadamard`'s inner
    /// projection so producer-fused Hadamard (`can_fuse_h64_from_producer`) remains exact.
    pub fn to_device_f16(
        &self,
        phys_n: usize,
        phys_k: usize,
        hadamard_block_size: Option<usize>,
    ) -> (Vec<u16>, Vec<u16>) {
        let total_a_rows = self.num_adapters * self.rank_pad;
        let mut a_work = if let Some(block_size) = hadamard_block_size {
            let mut padded = if phys_k > self.logical_k {
                pad_hadamard_cols_f32(&self.a_f32, total_a_rows, self.logical_k, phys_k)
            } else {
                self.a_f32.clone()
            };
            hadamard_rows_f32(&mut padded, phys_k, block_size);
            padded
        } else if phys_k > self.logical_k {
            pad_hadamard_cols_f32(&self.a_f32, total_a_rows, self.logical_k, phys_k)
        } else {
            self.a_f32.clone()
        };
        let a_f16: Vec<u16> = a_work.drain(..).map(f32_to_f16).collect();

        let b_f16: Vec<u16> = if phys_n > self.logical_n {
            let mut out = vec![0u16; self.num_adapters * phys_n * self.rank_pad];
            for ad in 0..self.num_adapters {
                let src_base = ad * self.logical_n * self.rank_pad;
                let dst_base = ad * phys_n * self.rank_pad;
                for i in 0..(self.logical_n * self.rank_pad) {
                    out[dst_base + i] = f32_to_f16(self.b_f32[src_base + i]);
                }
            }
            out
        } else {
            self.b_f32.iter().copied().map(f32_to_f16).collect()
        };
        (a_f16, b_f16)
    }
}

/// Slice specification for packing one or more sharded LoRA sub-projections into a single
/// target `LinearProj`.
pub struct LoraSubSlice<'a> {
    pub weight: Option<&'a LoraSliceWeight>,
    pub local_n: usize,
}

fn f32_slice_bits_eq(a: &[f32], b: &[f32]) -> bool {
    a.len() == b.len()
        && a.iter()
            .zip(b.iter())
            .all(|(x, y)| x.to_bits() == y.to_bits())
}

/// Packs per-adapter `LoraSubSlice`s into a single `PackedProjLoraHost`, deduplicating
/// any sub-slices within an adapter that share an identical `lora_A` matrix (e.g., joint-SVD
/// exports of `q_proj`/`k_proj`/`v_proj`, `gate_proj`/`up_proj`, or `in_proj_qkv`/`in_proj_z`).
pub fn pack_multi_slice_lora(
    per_adapter_slices: &[Vec<LoraSubSlice<'_>>],
    local_n: usize,
    local_k: usize,
) -> io::Result<Option<PackedProjLoraHost>> {
    let num_adapters = per_adapter_slices.len();
    let mut max_total_r = 0usize;
    let mut adapter_mask = 0u64;
    for (ad_idx, slices) in per_adapter_slices.iter().enumerate() {
        let sum_n: usize = slices.iter().map(|s| s.local_n).sum();
        if sum_n != local_n {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!("LoRA slice local_n sum {sum_n} != target local_n {local_n}"),
            ));
        }
        let mut total_r = 0usize;
        let mut unique_as: Vec<(usize, &[f32])> = Vec::with_capacity(slices.len());
        for s in slices {
            if let Some(w) = s.weight {
                if w.in_features != local_k || w.out_features != s.local_n {
                    return Err(io::Error::new(
                        io::ErrorKind::InvalidData,
                        format!(
                            "LoRA slice shape mismatch: got (out={}, in={}), expected (out={}, in={local_k})",
                            w.out_features, w.in_features, s.local_n
                        ),
                    ));
                }
                if !unique_as
                    .iter()
                    .any(|&(prev_r, prev_a)| prev_r == w.r && f32_slice_bits_eq(prev_a, &w.a_f32))
                {
                    unique_as.push((w.r, &w.a_f32));
                    total_r += w.r;
                }
            }
        }
        if total_r > 0 {
            adapter_mask |= 1u64 << ad_idx;
            max_total_r = max_total_r.max(total_r);
        }
    }
    if adapter_mask == 0 || max_total_r == 0 {
        return Ok(None);
    }
    let rank_pad = max_total_r.div_ceil(4) * 4;
    if rank_pad > MAX_LORA_RANK_PAD {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!("packed LoRA rank {rank_pad} exceeds maximum {MAX_LORA_RANK_PAD}"),
        ));
    }

    let mut a_f32 = vec![0.0f32; num_adapters * rank_pad * local_k];
    let mut b_f32 = vec![0.0f32; num_adapters * local_n * rank_pad];
    let mut common_b_slices: Option<Vec<[i32; 4]>> = None;
    let mut b_slices_uniform = true;

    for (ad_idx, slices) in per_adapter_slices.iter().enumerate() {
        let mut r_off = 0usize;
        let mut n_off = 0usize;
        let a_ad_base = ad_idx * rank_pad * local_k;
        let b_ad_base = ad_idx * local_n * rank_pad;
        let mut placed_as: Vec<(usize, usize, &[f32])> = Vec::with_capacity(slices.len());
        let mut ad_b_slices: Vec<[i32; 4]> = Vec::with_capacity(slices.len());
        for s in slices {
            if let Some(w) = s.weight {
                let use_r_off = if let Some(&(existing_r_off, _, _)) =
                    placed_as.iter().find(|&&(_, prev_r, prev_a)| {
                        prev_r == w.r && f32_slice_bits_eq(prev_a, &w.a_f32)
                    }) {
                    existing_r_off
                } else {
                    // Copy A: [w.r, local_k] -> a_f32[ad_idx, r_off .. r_off + w.r, 0..local_k]
                    let dst_a = &mut a_f32
                        [a_ad_base + r_off * local_k..a_ad_base + (r_off + w.r) * local_k];
                    dst_a.copy_from_slice(&w.a_f32);
                    let placed_off = r_off;
                    placed_as.push((placed_off, w.r, &w.a_f32));
                    r_off += w.r;
                    placed_off
                };
                // Copy B: [s.local_n, w.r] -> b_f32[ad_idx, n_off .. n_off + s.local_n, use_r_off .. use_r_off + w.r]
                for row in 0..s.local_n {
                    let src_row = &w.b_f32[row * w.r..(row + 1) * w.r];
                    let dst_row_start = b_ad_base + (n_off + row) * rank_pad + use_r_off;
                    b_f32[dst_row_start..dst_row_start + w.r].copy_from_slice(src_row);
                }
                ad_b_slices.push([n_off as i32, s.local_n as i32, use_r_off as i32, w.r as i32]);
            }
            n_off += s.local_n;
        }
        if ((adapter_mask >> ad_idx) & 1) != 0 {
            match &common_b_slices {
                None => common_b_slices = Some(ad_b_slices),
                Some(prev) if prev != &ad_b_slices => b_slices_uniform = false,
                Some(_) => {}
            }
        }
    }

    let mut num_b_slices = 0u8;
    let mut b_slices4 = [0i32; 12];
    if b_slices_uniform
        && let Some(ref bs) = common_b_slices
        && (2..=3).contains(&bs.len())
        && bs.iter().all(|&[n_off, sn, r_off, sr]| {
            n_off % 8 == 0 && sn >= 8 && sn % 8 == 0 && r_off % 8 == 0 && sr >= 8 && sr % 8 == 0
        })
    {
        // Only split B into sub-slices when they do not all share the same r_off (i.e. block-diagonal)
        let all_same_r = bs
            .windows(2)
            .all(|w| w[0][2] == w[1][2] && w[0][3] == w[1][3]);
        if !all_same_r {
            num_b_slices = bs.len() as u8;
            for (i, s) in bs.iter().enumerate() {
                b_slices4[i * 4..i * 4 + 4].copy_from_slice(s);
            }
        }
    }

    Ok(Some(PackedProjLoraHost {
        a_f32,
        b_f32,
        logical_n: local_n,
        logical_k: local_k,
        rank_pad,
        num_adapters,
        adapter_mask,
        num_b_slices,
        b_slices4,
    }))
}

impl LoraSliceWeight {
    /// Extracts a row shard `row_start..row_end` along `out_features` (column-parallel projection:
    /// `A` is replicated `[r, in_features]`, `B` is row-sliced `[row_end - row_start, r]`).
    pub fn slice_rows(&self, row_start: usize, row_end: usize) -> io::Result<Self> {
        if row_end > self.out_features || row_start > row_end {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!(
                    "LoRA row slice {}..{} out of bounds for out_features={}",
                    row_start, row_end, self.out_features
                ),
            ));
        }
        let new_out = row_end - row_start;
        Ok(Self {
            r: self.r,
            in_features: self.in_features,
            out_features: new_out,
            a_f32: self.a_f32.clone(),
            b_f32: self.b_f32[row_start * self.r..row_end * self.r].to_vec(),
        })
    }

    /// Extracts multiple disjoint row ranges from `B` and concatenates them along `out_features`
    /// while keeping a single shared `A` (`[r, in_features]`). Used for TP-sharded `in_proj_qkv`.
    pub fn slice_concat_rows(&self, ranges: &[(usize, usize)]) -> io::Result<Self> {
        let mut total_out = 0usize;
        for &(start, end) in ranges {
            if end > self.out_features || start > end {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    format!(
                        "LoRA row range {}..{} out of bounds for out_features={}",
                        start, end, self.out_features
                    ),
                ));
            }
            total_out += end - start;
        }
        let mut b_f32 = Vec::with_capacity(total_out * self.r);
        for &(start, end) in ranges {
            b_f32.extend_from_slice(&self.b_f32[start * self.r..end * self.r]);
        }
        Ok(Self {
            r: self.r,
            in_features: self.in_features,
            out_features: total_out,
            a_f32: self.a_f32.clone(),
            b_f32,
        })
    }

    /// Extracts a column shard `col_start..col_end` along `in_features` (row-parallel projection:
    /// `A` is column-sliced `[r, col_end - col_start]`, `B` is replicated `[out_features, r]`).
    pub fn slice_cols(&self, col_start: usize, col_end: usize) -> io::Result<Self> {
        if col_end > self.in_features || col_start > col_end {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!(
                    "LoRA col slice {}..{} out of bounds for in_features={}",
                    col_start, col_end, self.in_features
                ),
            ));
        }
        let new_in = col_end - col_start;
        let mut a_f32 = Vec::with_capacity(self.r * new_in);
        for row in 0..self.r {
            let base = row * self.in_features;
            a_f32.extend_from_slice(&self.a_f32[base + col_start..base + col_end]);
        }
        Ok(Self {
            r: self.r,
            in_features: new_in,
            out_features: self.out_features,
            a_f32,
            b_f32: self.b_f32.clone(),
        })
    }
}

/// GPU allocations and atomic execution state owned by a [`crate::model::Qwen3_5ModelRank`]
/// when one or more LoRA adapters are attached.
#[cfg(feature = "cuda")]
pub struct RankLoraRuntime {
    pub weight_arena: DeviceArena,
    pub lowrank_ws: DeviceBuffer,
    pub state: std::sync::Arc<RankLoraState>,
    pub num_adapters: usize,
    pub max_rank_pad: usize,
    pub adapter_midloops: Vec<Vec<(usize, usize, usize)>>,
    pub adapter_noncausal: Vec<bool>,
}

#[cfg(feature = "cuda")]
fn pack_single_slice_for_adapters(
    adapters: &[LoraAdapterHost],
    local_n: usize,
    local_k: usize,
    mut extract: impl FnMut(&LoraAdapterHost) -> io::Result<Option<LoraSliceWeight>>,
) -> Result<Option<PackedProjLoraHost>> {
    let mut owned: Vec<Option<LoraSliceWeight>> = Vec::with_capacity(adapters.len());
    for ad in adapters {
        owned.push(extract(ad).map_err(|e| loader_err_msg(e.to_string()))?);
    }
    let refs: Vec<Vec<LoraSubSlice<'_>>> = owned
        .iter()
        .map(|w| {
            vec![LoraSubSlice {
                weight: w.as_ref(),
                local_n,
            }]
        })
        .collect();
    pack_multi_slice_lora(&refs, local_n, local_k).map_err(|e| loader_err_msg(e.to_string()))
}

#[cfg(feature = "cuda")]
fn pack_fused_or_split_for_adapters(
    adapters: &[LoraAdapterHost],
    slice_local_ns: &[usize],
    local_n: usize,
    local_k: usize,
    mut extract_fused: impl FnMut(&LoraAdapterHost) -> io::Result<Option<LoraSliceWeight>>,
    mut extract_split: impl FnMut(&LoraAdapterHost, usize) -> io::Result<Option<LoraSliceWeight>>,
) -> Result<Option<PackedProjLoraHost>> {
    let mut owned: Vec<Vec<(Option<LoraSliceWeight>, usize)>> = Vec::with_capacity(adapters.len());
    for ad in adapters {
        let fused = extract_fused(ad).map_err(|e| loader_err_msg(e.to_string()))?;
        let mut split_row = Vec::with_capacity(slice_local_ns.len());
        let mut any_split = false;
        for (idx, &sn) in slice_local_ns.iter().enumerate() {
            let sw = extract_split(ad, idx).map_err(|e| loader_err_msg(e.to_string()))?;
            any_split |= sw.is_some();
            split_row.push((sw, sn));
        }
        if fused.is_some() && any_split {
            return Err(loader_err_msg(format!(
                "LoRA adapter '{}' defines both fused and split target weights for the same projection",
                ad.name
            )));
        }
        if fused.is_some() {
            owned.push(vec![(fused, local_n)]);
        } else {
            owned.push(split_row);
        }
    }
    let refs: Vec<Vec<LoraSubSlice<'_>>> = owned
        .iter()
        .map(|row| {
            row.iter()
                .map(|(w, sn)| LoraSubSlice {
                    weight: w.as_ref(),
                    local_n: *sn,
                })
                .collect()
        })
        .collect();
    pack_multi_slice_lora(&refs, local_n, local_k).map_err(|e| loader_err_msg(e.to_string()))
}

/// Validates and attaches `adapters` onto a rank's decoder `layers` and `lm_head`,
/// returning the newly allocated [`RankLoraRuntime`].
#[cfg(feature = "cuda")]
#[allow(clippy::cognitive_complexity, clippy::too_many_arguments)]
pub fn attach_lora_to_rank(
    cfg: &ModelConfig,
    rank: usize,
    tp_size: usize,
    device_id: i32,
    stream: &CudaStream,
    layers: &mut [crate::model::DecoderLayerWeights],
    lm_head: &mut Option<LinearProj>,
    adapters: &[LoraAdapterHost],
) -> Result<Option<RankLoraRuntime>> {
    use crate::loader::LayerKind;
    use crate::model::{FfnBlockWeights, LayerBlockWeights};

    for layer in layers.iter_mut() {
        match &mut layer.block {
            LayerBlockWeights::FullAttention(fa) => {
                fa.qkv_proj.strip_lora();
                fa.o_proj.strip_lora();
            }
            LayerBlockWeights::LinearAttention(gdn) => {
                gdn.in_proj_qkvz.strip_lora();
                gdn.in_proj_ba.strip_lora();
                gdn.out_proj.strip_lora();
            }
        }
        match &mut layer.ffn {
            FfnBlockWeights::Dense {
                gate_up_proj,
                down_proj,
            } => {
                gate_up_proj.strip_lora();
                down_proj.strip_lora();
            }
            FfnBlockWeights::Moe(moe) => {
                if let Some(p) = &mut moe.shared_gate_up_proj {
                    p.strip_lora();
                }
                if let Some(p) = &mut moe.shared_down_proj {
                    p.strip_lora();
                }
            }
        }
    }
    if let Some(lm) = lm_head.as_mut() {
        lm.strip_lora();
    }

    if adapters.is_empty() {
        return Ok(None);
    }
    if adapters.len() > MAX_LORA_ADAPTERS {
        return Err(loader_err_msg(format!(
            "cannot attach {} LoRA adapters (max {MAX_LORA_ADAPTERS})",
            adapters.len()
        )));
    }

    for ad in adapters {
        for &(a, b, passes) in &ad.midloop {
            if passes < 1 || a > b || b >= cfg.num_hidden_layers {
                return Err(loader_err_msg(format!(
                    "LoRA adapter '{}': invalid midloop triple [{a}, {b}, {passes}] for num_hidden_layers={}",
                    ad.name, cfg.num_hidden_layers
                )));
            }
        }
        for &(layer_id, target) in ad.layer_weights.keys() {
            if layer_id >= cfg.num_hidden_layers {
                return Err(loader_err_msg(format!(
                    "LoRA adapter '{}' targets layer {layer_id}, but model has {} layers",
                    ad.name, cfg.num_hidden_layers
                )));
            }
            let kind = cfg.layer_types[layer_id];
            match target {
                TargetLoraModule::QProj
                | TargetLoraModule::KProj
                | TargetLoraModule::VProj
                | TargetLoraModule::QkvProj
                | TargetLoraModule::OProj
                    if kind != LayerKind::FullAttention =>
                {
                    return Err(loader_err_msg(format!(
                        "LoRA adapter '{}' targets {target:?} on non-FullAttention layer {layer_id}",
                        ad.name
                    )));
                }
                TargetLoraModule::InProjQkv
                | TargetLoraModule::InProjZ
                | TargetLoraModule::InProjQkvz
                | TargetLoraModule::InProjB
                | TargetLoraModule::InProjA
                | TargetLoraModule::InProjBa
                | TargetLoraModule::OutProj
                    if kind != LayerKind::LinearAttention =>
                {
                    return Err(loader_err_msg(format!(
                        "LoRA adapter '{}' targets {target:?} on non-LinearAttention layer {layer_id}",
                        ad.name
                    )));
                }
                _ => {}
            }
        }
    }

    let h_dim = cfg.hidden_size;
    let local_q_heads = cfg.num_attention_heads / tp_size;
    let local_kv_heads = cfg.local_kv_heads(tp_size);
    let kv_shard_idx = cfg.kv_head_shard_index(rank, tp_size);
    let q_gate_total = cfg.num_attention_heads * cfg.head_dim * 2;
    let q_gate_local = local_q_heads * cfg.head_dim * 2;
    let kv_total = cfg.num_key_value_heads * cfg.head_dim;
    let kv_local = local_kv_heads * cfg.head_dim;
    let o_cols = local_q_heads * cfg.head_dim;

    let gdn_k_total = cfg.linear_num_key_heads * cfg.linear_key_head_dim;
    let gdn_v_total = cfg.linear_num_value_heads * cfg.linear_value_head_dim;
    let gdn_k_local = gdn_k_total / tp_size;
    let gdn_v_local = gdn_v_total / tp_size;
    let total_v_heads = cfg.linear_num_value_heads;
    let local_v_heads = total_v_heads / tp_size;
    let local_interm = cfg.intermediate_size / tp_size;
    let local_shared_interm = cfg.shared_expert_intermediate_size / tp_size;
    let local_vocab_size = cfg.vocab_size / tp_size;

    struct PendingLayerLora {
        qkv: Option<PackedProjLoraHost>,
        o: Option<PackedProjLoraHost>,
        qkvz: Option<PackedProjLoraHost>,
        ba: Option<PackedProjLoraHost>,
        out: Option<PackedProjLoraHost>,
        ffn_gu: Option<PackedProjLoraHost>,
        ffn_down: Option<PackedProjLoraHost>,
    }

    let mut pending_layers = Vec::with_capacity(layers.len());
    let mut total_arena_bytes = 0usize;
    let mut max_rank_pad = 4usize;

    let mut account_packed = |p: &Option<PackedProjLoraHost>, proj: &LinearProj| {
        if let Some(pk) = p {
            max_rank_pad = max_rank_pad.max(pk.rank_pad);
            let (phys_n, phys_k) = match proj {
                LinearProj::Hadamard { inner, .. } => (inner.size_n(), inner.size_k()),
                _ => (proj.size_n(), proj.size_k()),
            };
            let a_bytes = (pk.num_adapters * pk.rank_pad * phys_k * 2 + 255) & !255;
            let b_bytes = (pk.num_adapters * phys_n * pk.rank_pad * 2 + 255) & !255;
            total_arena_bytes = total_arena_bytes.saturating_add(a_bytes + b_bytes);
        }
    };

    for (layer_id, layer) in layers.iter().enumerate() {
        let mut pl = PendingLayerLora {
            qkv: None,
            o: None,
            qkvz: None,
            ba: None,
            out: None,
            ffn_gu: None,
            ffn_down: None,
        };
        match &layer.block {
            LayerBlockWeights::FullAttention(fa) => {
                let qkv_local_n = q_gate_local + 2 * kv_local;
                pl.qkv = pack_fused_or_split_for_adapters(
                    adapters,
                    &[q_gate_local, kv_local, kv_local],
                    qkv_local_n,
                    h_dim,
                    |ad| {
                        ad.get_layer_module(layer_id, TargetLoraModule::QkvProj)
                            .map(|w| {
                                w.slice_concat_rows(&[
                                    (rank * q_gate_local, (rank + 1) * q_gate_local),
                                    (
                                        q_gate_total + kv_shard_idx * kv_local,
                                        q_gate_total + (kv_shard_idx + 1) * kv_local,
                                    ),
                                    (
                                        q_gate_total + kv_total + kv_shard_idx * kv_local,
                                        q_gate_total + kv_total + (kv_shard_idx + 1) * kv_local,
                                    ),
                                ])
                            })
                            .transpose()
                    },
                    |ad, idx| match idx {
                        0 => ad
                            .get_layer_module(layer_id, TargetLoraModule::QProj)
                            .map(|w| w.slice_rows(rank * q_gate_local, (rank + 1) * q_gate_local))
                            .transpose(),
                        1 => ad
                            .get_layer_module(layer_id, TargetLoraModule::KProj)
                            .map(|w| {
                                w.slice_rows(kv_shard_idx * kv_local, (kv_shard_idx + 1) * kv_local)
                            })
                            .transpose(),
                        _ => ad
                            .get_layer_module(layer_id, TargetLoraModule::VProj)
                            .map(|w| {
                                w.slice_rows(kv_shard_idx * kv_local, (kv_shard_idx + 1) * kv_local)
                            })
                            .transpose(),
                    },
                )?;
                pl.o = pack_single_slice_for_adapters(adapters, h_dim, o_cols, |ad| {
                    ad.get_layer_module(layer_id, TargetLoraModule::OProj)
                        .map(|w| w.slice_cols(rank * o_cols, (rank + 1) * o_cols))
                        .transpose()
                })?;
                account_packed(&pl.qkv, &fa.qkv_proj);
                account_packed(&pl.o, &fa.o_proj);
            }
            LayerBlockWeights::LinearAttention(gdn) => {
                let qkv_local_n = 2 * gdn_k_local + gdn_v_local;
                let qkvz_local_n = qkv_local_n + gdn_v_local;
                pl.qkvz = pack_fused_or_split_for_adapters(
                    adapters,
                    &[qkv_local_n, gdn_v_local],
                    qkvz_local_n,
                    h_dim,
                    |ad| {
                        ad.get_layer_module(layer_id, TargetLoraModule::InProjQkvz)
                            .map(|w| {
                                w.slice_concat_rows(&[
                                    (rank * gdn_k_local, (rank + 1) * gdn_k_local),
                                    (
                                        gdn_k_total + rank * gdn_k_local,
                                        gdn_k_total + (rank + 1) * gdn_k_local,
                                    ),
                                    (
                                        2 * gdn_k_total + rank * gdn_v_local,
                                        2 * gdn_k_total + (rank + 1) * gdn_v_local,
                                    ),
                                    (
                                        2 * gdn_k_total + gdn_v_total + rank * gdn_v_local,
                                        2 * gdn_k_total + gdn_v_total + (rank + 1) * gdn_v_local,
                                    ),
                                ])
                            })
                            .transpose()
                    },
                    |ad, idx| match idx {
                        0 => ad
                            .get_layer_module(layer_id, TargetLoraModule::InProjQkv)
                            .map(|w| {
                                w.slice_concat_rows(&[
                                    (rank * gdn_k_local, (rank + 1) * gdn_k_local),
                                    (
                                        gdn_k_total + rank * gdn_k_local,
                                        gdn_k_total + (rank + 1) * gdn_k_local,
                                    ),
                                    (
                                        2 * gdn_k_total + rank * gdn_v_local,
                                        2 * gdn_k_total + (rank + 1) * gdn_v_local,
                                    ),
                                ])
                            })
                            .transpose(),
                        _ => ad
                            .get_layer_module(layer_id, TargetLoraModule::InProjZ)
                            .map(|w| w.slice_rows(rank * gdn_v_local, (rank + 1) * gdn_v_local))
                            .transpose(),
                    },
                )?;

                let ba_local_n = 2 * local_v_heads;
                pl.ba = pack_fused_or_split_for_adapters(
                    adapters,
                    &[local_v_heads, local_v_heads],
                    ba_local_n,
                    h_dim,
                    |ad| {
                        ad.get_layer_module(layer_id, TargetLoraModule::InProjBa)
                            .map(|w| {
                                w.slice_concat_rows(&[
                                    (rank * local_v_heads, (rank + 1) * local_v_heads),
                                    (
                                        total_v_heads + rank * local_v_heads,
                                        total_v_heads + (rank + 1) * local_v_heads,
                                    ),
                                ])
                            })
                            .transpose()
                    },
                    |ad, idx| match idx {
                        0 => ad
                            .get_layer_module(layer_id, TargetLoraModule::InProjB)
                            .map(|w| w.slice_rows(rank * local_v_heads, (rank + 1) * local_v_heads))
                            .transpose(),
                        _ => ad
                            .get_layer_module(layer_id, TargetLoraModule::InProjA)
                            .map(|w| w.slice_rows(rank * local_v_heads, (rank + 1) * local_v_heads))
                            .transpose(),
                    },
                )?;

                pl.out = pack_single_slice_for_adapters(adapters, h_dim, gdn_v_local, |ad| {
                    ad.get_layer_module(layer_id, TargetLoraModule::OutProj)
                        .map(|w| w.slice_cols(rank * gdn_v_local, (rank + 1) * gdn_v_local))
                        .transpose()
                })?;

                account_packed(&pl.qkvz, &gdn.in_proj_qkvz);
                account_packed(&pl.ba, &gdn.in_proj_ba);
                account_packed(&pl.out, &gdn.out_proj);
            }
        }

        match &layer.ffn {
            FfnBlockWeights::Dense {
                gate_up_proj,
                down_proj,
            } => {
                let gu_local_n = 2 * local_interm;
                pl.ffn_gu = pack_fused_or_split_for_adapters(
                    adapters,
                    &[local_interm, local_interm],
                    gu_local_n,
                    h_dim,
                    |ad| {
                        ad.get_layer_module(layer_id, TargetLoraModule::GateUpProj)
                            .map(|w| {
                                w.slice_concat_rows(&[
                                    (rank * local_interm, (rank + 1) * local_interm),
                                    (
                                        cfg.intermediate_size + rank * local_interm,
                                        cfg.intermediate_size + (rank + 1) * local_interm,
                                    ),
                                ])
                            })
                            .transpose()
                    },
                    |ad, idx| match idx {
                        0 => ad
                            .get_layer_module(layer_id, TargetLoraModule::GateProj)
                            .map(|w| w.slice_rows(rank * local_interm, (rank + 1) * local_interm))
                            .transpose(),
                        _ => ad
                            .get_layer_module(layer_id, TargetLoraModule::UpProj)
                            .map(|w| w.slice_rows(rank * local_interm, (rank + 1) * local_interm))
                            .transpose(),
                    },
                )?;
                pl.ffn_down =
                    pack_single_slice_for_adapters(adapters, h_dim, local_interm, |ad| {
                        ad.get_layer_module(layer_id, TargetLoraModule::DownProj)
                            .map(|w| w.slice_cols(rank * local_interm, (rank + 1) * local_interm))
                            .transpose()
                    })?;
                account_packed(&pl.ffn_gu, gate_up_proj);
                account_packed(&pl.ffn_down, down_proj);
            }
            FfnBlockWeights::Moe(moe) => {
                if let Some(sh_gu) = &moe.shared_gate_up_proj {
                    let gu_local_n = 2 * local_shared_interm;
                    pl.ffn_gu = pack_fused_or_split_for_adapters(
                        adapters,
                        &[local_shared_interm, local_shared_interm],
                        gu_local_n,
                        h_dim,
                        |ad| {
                            ad.get_layer_module(layer_id, TargetLoraModule::SharedGateUpProj)
                                .map(|w| {
                                    w.slice_concat_rows(&[
                                        (
                                            rank * local_shared_interm,
                                            (rank + 1) * local_shared_interm,
                                        ),
                                        (
                                            cfg.shared_expert_intermediate_size
                                                + rank * local_shared_interm,
                                            cfg.shared_expert_intermediate_size
                                                + (rank + 1) * local_shared_interm,
                                        ),
                                    ])
                                })
                                .transpose()
                        },
                        |ad, idx| match idx {
                            0 => ad
                                .get_layer_module(layer_id, TargetLoraModule::SharedGateProj)
                                .map(|w| {
                                    w.slice_rows(
                                        rank * local_shared_interm,
                                        (rank + 1) * local_shared_interm,
                                    )
                                })
                                .transpose(),
                            _ => ad
                                .get_layer_module(layer_id, TargetLoraModule::SharedUpProj)
                                .map(|w| {
                                    w.slice_rows(
                                        rank * local_shared_interm,
                                        (rank + 1) * local_shared_interm,
                                    )
                                })
                                .transpose(),
                        },
                    )?;
                    account_packed(&pl.ffn_gu, sh_gu);
                }
                if let Some(sh_down) = &moe.shared_down_proj {
                    pl.ffn_down = pack_single_slice_for_adapters(
                        adapters,
                        h_dim,
                        local_shared_interm,
                        |ad| {
                            ad.get_layer_module(layer_id, TargetLoraModule::SharedDownProj)
                                .map(|w| {
                                    w.slice_cols(
                                        rank * local_shared_interm,
                                        (rank + 1) * local_shared_interm,
                                    )
                                })
                                .transpose()
                        },
                    )?;
                    account_packed(&pl.ffn_down, sh_down);
                }
            }
        }
        pending_layers.push(pl);
    }

    let pending_lm = if let Some(lm) = lm_head.as_ref() {
        let p = pack_single_slice_for_adapters(adapters, local_vocab_size, h_dim, |ad| {
            ad.lm_head_weight
                .as_ref()
                .map(|w| w.slice_rows(rank * local_vocab_size, (rank + 1) * local_vocab_size))
                .transpose()
        })?;
        account_packed(&p, lm);
        p
    } else {
        None
    };

    let arena_cap = total_arena_bytes.saturating_add(64 * 1024).max(64 * 1024);
    let mut weight_arena = DeviceArena::new(device_id, arena_cap)?;
    let max_tokens = crate::scheduler::DEFAULT_MAX_NUM_BATCHED_TOKENS.max(6144);
    let lowrank_ws_elems = max_tokens.saturating_mul(max_rank_pad);
    let lowrank_ws = DeviceBuffer::alloc(device_id, lowrank_ws_elems * 4)?;
    let state = std::sync::Arc::new(RankLoraState::new(lowrank_ws.addr(), lowrank_ws_elems));

    for (layer, pl) in layers.iter_mut().zip(pending_layers) {
        match &mut layer.block {
            LayerBlockWeights::FullAttention(fa) => {
                if let Some(pk) = &pl.qkv {
                    fa.qkv_proj
                        .wrap_with_lora(pk, &mut weight_arena, stream, &state)?;
                }
                if let Some(pk) = &pl.o {
                    fa.o_proj
                        .wrap_with_lora(pk, &mut weight_arena, stream, &state)?;
                }
            }
            LayerBlockWeights::LinearAttention(gdn) => {
                if let Some(pk) = &pl.qkvz {
                    gdn.in_proj_qkvz
                        .wrap_with_lora(pk, &mut weight_arena, stream, &state)?;
                }
                if let Some(pk) = &pl.ba {
                    gdn.in_proj_ba
                        .wrap_with_lora(pk, &mut weight_arena, stream, &state)?;
                }
                if let Some(pk) = &pl.out {
                    gdn.out_proj
                        .wrap_with_lora(pk, &mut weight_arena, stream, &state)?;
                }
            }
        }
        match &mut layer.ffn {
            FfnBlockWeights::Dense {
                gate_up_proj,
                down_proj,
            } => {
                if let Some(pk) = &pl.ffn_gu {
                    gate_up_proj.wrap_with_lora(pk, &mut weight_arena, stream, &state)?;
                }
                if let Some(pk) = &pl.ffn_down {
                    down_proj.wrap_with_lora(pk, &mut weight_arena, stream, &state)?;
                }
            }
            FfnBlockWeights::Moe(moe) => {
                if let (Some(sh_gu), Some(pk)) = (&mut moe.shared_gate_up_proj, &pl.ffn_gu) {
                    sh_gu.wrap_with_lora(pk, &mut weight_arena, stream, &state)?;
                }
                if let (Some(sh_down), Some(pk)) = (&mut moe.shared_down_proj, &pl.ffn_down) {
                    sh_down.wrap_with_lora(pk, &mut weight_arena, stream, &state)?;
                }
            }
        }
    }
    if let (Some(lm), Some(pk)) = (lm_head.as_mut(), &pending_lm) {
        lm.wrap_with_lora(pk, &mut weight_arena, stream, &state)?;
    }

    let _ = loader_err;
    stream.synchronize()?;
    Ok(Some(RankLoraRuntime {
        weight_arena,
        lowrank_ws,
        state,
        num_adapters: adapters.len(),
        max_rank_pad,
        adapter_midloops: adapters.iter().map(|ad| ad.midloop.clone()).collect(),
        adapter_noncausal: adapters
            .iter()
            .map(|ad| ad.noncausal_full_attention)
            .collect(),
    }))
}

#[cfg(test)]
mod tests;
