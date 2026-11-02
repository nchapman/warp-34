use std::{collections::BTreeSet, fs, io, path::Path};

use super::{ModelRole, invalid, read_json, safe_path};
use crate::loader::{ModelConfig, QuantScheme};

const METADATA: &[&str] = &[
    "config.json",
    "generation_config.json",
    "model.safetensors.index.json",
    "quantize_config.json",
    "quant_config.json",
    "quantization_config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "added_tokens.json",
    "vocab.json",
    "vocab.txt",
    "merges.txt",
    "tokenizer.model",
    "spiece.model",
    "chat_template.jinja",
    "chat_template.json",
    "processor_config.json",
    "preprocessor_config.json",
    "video_preprocessor_config.json",
    "joint_head_config.json",
];

pub(super) fn metadata_files(files: &BTreeSet<String>) -> io::Result<Vec<String>> {
    if !files.contains("config.json") {
        return Err(invalid(
            "repository has no root config.json; specify a supported checkpoint repository",
        ));
    }
    let selected: Vec<_> = files
        .iter()
        .filter(|name| {
            METADATA.contains(&name.as_str())
                || (name.starts_with("chat_templates/") && name.ends_with(".jinja"))
        })
        .cloned()
        .collect();
    for name in &selected {
        safe_path(name)?;
    }
    Ok(selected)
}

pub(super) fn weight_files(dir: &Path, files: &BTreeSet<String>) -> io::Result<Vec<String>> {
    let mut selected = BTreeSet::new();
    if files.contains("model.safetensors.index.json") {
        let index = read_json(&dir.join("model.safetensors.index.json"))?;
        let map = index["weight_map"]
            .as_object()
            .filter(|m| !m.is_empty())
            .ok_or_else(|| invalid("model.safetensors.index.json has no nonempty weight_map"))?;
        for value in map.values() {
            let name = value
                .as_str()
                .ok_or_else(|| invalid("invalid safetensors shard filename"))?;
            safe_path(name)?;
            if !name.ends_with(".safetensors") || !files.contains(name) {
                return Err(invalid(format!(
                    "checkpoint is missing indexed safetensors shard {name}"
                )));
            }
            selected.insert(name.to_owned());
        }
    } else if files.contains("model.safetensors") {
        selected.insert("model.safetensors".to_owned());
    } else {
        return Err(invalid(
            "repository has no model.safetensors or model.safetensors.index.json; only Safetensors checkpoints are supported (not pickle, GGUF, or remote Python code)",
        ));
    }
    // Decision heads are loaded separately from the backbone archive.
    for name in ["head.safetensors", "joint_head.safetensors"] {
        if files.contains(name) {
            selected.insert(name.into());
        }
    }
    Ok(selected.into_iter().collect())
}

pub(super) fn validate_config(dir: &Path, role: ModelRole) -> io::Result<()> {
    let root = read_json(&dir.join("config.json"))?;
    match role {
        ModelRole::Target => {
            let body = root.get("text_config").unwrap_or(&root);
            let kind = body
                .get("model_type")
                .or_else(|| root.get("model_type"))
                .and_then(|v| v.as_str())
                .unwrap_or("");
            if !matches!(
                kind,
                "qwen3_5" | "qwen3_5_text" | "qwen3_5_moe" | "qwen3_5_moe_text"
            ) {
                return Err(invalid(format!(
                    "unsupported target model_type '{}'; expected a Qwen 3.5/3.6/3.8 hybrid checkpoint; remote model code is never executed",
                    crate::loader::sanitize_diag(kind)
                )));
            }
            ModelConfig::from_dir(dir)?;
        }
        ModelRole::Draft => {
            let body = root.get("transformer_layer_config").unwrap_or(&root);
            let config = ModelConfig::from_drafter_json(body)?;
            if config.quant != QuantScheme::Unquantized
                || root
                    .get("quantization_config")
                    .is_some_and(|v| !v.is_null())
                || [
                    "quantize_config.json",
                    "quant_config.json",
                    "quantization_config.json",
                ]
                .iter()
                .any(|name| dir.join(name).exists())
            {
                return Err(invalid(
                    "quantized DFlash2 drafters are not supported; use native BF16/FP16 drafter weights",
                ));
            }
            if !root["architectures"]
                .as_array()
                .is_some_and(|a| a.iter().any(|v| v == "DFlash2DraftModel"))
            {
                return Err(invalid(
                    "--draft-model requires a DFlash2DraftModel checkpoint",
                ));
            }
            #[cfg(feature = "cuda")]
            crate::dflash::DFlash2Config::load(dir).map_err(|e| invalid(e.to_string()))?;
        }
    }
    Ok(())
}

pub(super) fn validate_files(dir: &Path, names: &[String]) -> io::Result<()> {
    for name in names {
        safe_path(name)?;
        let path = dir.join(name);
        if !fs::metadata(&path).is_ok_and(|m| m.is_file() && m.len() > 0) {
            return Err(io::Error::new(
                io::ErrorKind::NotFound,
                format!(
                    "incomplete model snapshot: missing or empty {name}; rerun without --offline / HF_HUB_OFFLINE to download it"
                ),
            ));
        }
    }
    Ok(())
}

pub(super) fn validate_checkpoint(
    dir: &Path,
    role: ModelRole,
    names: &BTreeSet<String>,
) -> io::Result<()> {
    validate_config(dir, role)?;
    if role == ModelRole::Target {
        validate_files(dir, &["tokenizer.json".into()])?;
    }
    validate_files(dir, &weight_files(dir, names)?)?;
    // Parse headers and check tensor bounds, without loading weights into RAM or touching CUDA.
    crate::loader::SafetensorsArchive::open(dir)?;
    for name in ["head.safetensors", "joint_head.safetensors"] {
        if names.contains(name) {
            crate::loader::SafetensorsArchive::open(&dir.join(name))?;
        }
    }
    Ok(())
}
