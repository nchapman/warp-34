use std::{collections::BTreeSet, fs, io, path::Path};

use super::{ModelRole, invalid, read_json, safe_path};
use crate::loader::{ModelConfig, QuantScheme};

const METADATA: &[&str] = &[
    "config.json",
    "model_index.json",
    "generation_config.json",
    "model.safetensors.index.json",
    "diffusion_pytorch_model.safetensors.index.json",
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
    "decision_config.json",
    "joint_config.json",
    "static_hot_cache_rankings.json",
    "runtime/static_hot_cache_rankings.json",
    "runtime/mtp-int4-g32/config.json",
    "runtime/mtp-int4-g32/model.safetensors.index.json",
];

pub(super) const PIPELINE_SUBDIRS: &[&str] = &[
    "processor",
    "scheduler",
    "text_encoder",
    "tokenizer",
    "transformer",
    "vae",
];

const PIPELINE_SUBDIR_METADATA: &[&str] = &[
    "config.json",
    "scheduler_config.json",
    "generation_config.json",
    "model.safetensors.index.json",
    "diffusion_pytorch_model.safetensors.index.json",
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
    "chat_template.jinja",
    "chat_template.json",
    "processor_config.json",
    "preprocessor_config.json",
    "video_preprocessor_config.json",
];

pub(super) fn is_pipeline_files(files: &BTreeSet<String>) -> bool {
    files.contains("model_index.json") && !files.contains("config.json")
}

pub(super) fn metadata_files(files: &BTreeSet<String>) -> io::Result<Vec<String>> {
    if !files.contains("config.json") && !files.contains("model_index.json") {
        return Err(invalid(
            "repository has no root config.json; specify a supported checkpoint repository",
        ));
    }
    let is_pipeline = is_pipeline_files(files);
    let selected: Vec<_> = files
        .iter()
        .filter(|name| {
            METADATA.contains(&name.as_str())
                || (name.starts_with("chat_templates/") && name.ends_with(".jinja"))
                || (is_pipeline
                    && name.split_once('/').is_some_and(|(sub, leaf)| {
                        PIPELINE_SUBDIRS.contains(&sub) && PIPELINE_SUBDIR_METADATA.contains(&leaf)
                    }))
        })
        .cloned()
        .collect();
    for name in &selected {
        safe_path(name)?;
    }
    Ok(selected)
}

fn collect_component_weights(
    dir: &Path,
    subdir: &str,
    files: &BTreeSet<String>,
    selected: &mut BTreeSet<String>,
) -> io::Result<()> {
    for idx_leaf in [
        "model.safetensors.index.json",
        "diffusion_pytorch_model.safetensors.index.json",
    ] {
        let rel_idx = format!("{subdir}/{idx_leaf}");
        if files.contains(&rel_idx) {
            let index = read_json(&dir.join(&rel_idx))?;
            let map = index["weight_map"]
                .as_object()
                .filter(|m| !m.is_empty())
                .ok_or_else(|| invalid(format!("{rel_idx} has no nonempty weight_map")))?;
            for value in map.values() {
                let shard = value
                    .as_str()
                    .ok_or_else(|| invalid("invalid safetensors shard filename"))?;
                let rel_shard = format!("{subdir}/{shard}");
                safe_path(&rel_shard)?;
                if !rel_shard.ends_with(".safetensors") || !files.contains(&rel_shard) {
                    return Err(invalid(format!(
                        "checkpoint is missing indexed safetensors shard {rel_shard}"
                    )));
                }
                selected.insert(rel_shard);
            }
            return Ok(());
        }
    }
    for single_leaf in ["model.safetensors", "diffusion_pytorch_model.safetensors"] {
        let rel_single = format!("{subdir}/{single_leaf}");
        if files.contains(&rel_single) {
            selected.insert(rel_single);
            return Ok(());
        }
    }
    Err(invalid(format!(
        "pipeline component '{subdir}' has no safetensors weights or index"
    )))
}

pub(super) fn weight_files(dir: &Path, files: &BTreeSet<String>) -> io::Result<Vec<String>> {
    if is_pipeline_files(files)
        || crate::qwen_image::QwenImage21PipelineConfig::is_pipeline_dir(dir)
    {
        let mut selected = BTreeSet::new();
        for comp in ["text_encoder", "transformer", "vae"] {
            collect_component_weights(dir, comp, files, &mut selected)?;
        }
        return Ok(selected.into_iter().collect());
    }
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
    // Decision readout heads are loaded separately from the backbone archive.
    if files.contains("readout.safetensors") {
        selected.insert("readout.safetensors".into());
    }
    // Embedded Qwen4 MTP INT4 draft checkpoint shards (when published inside the target repo).
    if files.contains("runtime/mtp-int4-g32/model.safetensors.index.json") {
        let mtp_idx_path = dir.join("runtime/mtp-int4-g32/model.safetensors.index.json");
        if mtp_idx_path.exists() {
            let mtp_idx = read_json(&mtp_idx_path)?;
            if let Some(map) = mtp_idx["weight_map"].as_object() {
                for value in map.values() {
                    let shard = value
                        .as_str()
                        .ok_or_else(|| invalid("invalid MTP safetensors shard filename"))?;
                    let rel = format!("runtime/mtp-int4-g32/{shard}");
                    safe_path(&rel)?;
                    if files.contains(&rel) {
                        selected.insert(rel);
                    }
                }
            }
        }
    } else if files.contains("runtime/mtp-int4-g32/model.safetensors") {
        selected.insert("runtime/mtp-int4-g32/model.safetensors".to_string());
    }
    Ok(selected.into_iter().collect())
}

pub(super) fn validate_config(dir: &Path, role: ModelRole) -> io::Result<()> {
    if crate::qwen_image::QwenImage21PipelineConfig::is_pipeline_dir(dir) {
        if role != ModelRole::Target {
            return Err(invalid(
                "QwenImage21Pipeline checkpoints can only be loaded as a target model",
            ));
        }
        crate::qwen_image::QwenImage21PipelineConfig::from_dir(dir)?;
        return Ok(());
    }
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
                "qwen3_5"
                    | "qwen3_5_text"
                    | "qwen3_5_moe"
                    | "qwen3_5_moe_text"
                    | "qwen4_exp"
                    | "qwen4_exp_text"
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
            let mut config = ModelConfig::from_drafter_json(body)?;
            if config.quant == QuantScheme::Unquantized
                && root
                    .get("quantization_config")
                    .is_some_and(|v| !v.is_null())
            {
                config.quant = ModelConfig::parse_quant(root.get("quantization_config"))?;
            }
            if !matches!(
                config.quant,
                QuantScheme::Unquantized
                    | QuantScheme::CompressedTensors {
                        num_bits: 4,
                        symmetric: true,
                        ..
                    }
            ) || [
                "quantize_config.json",
                "quant_config.json",
                "quantization_config.json",
            ]
            .iter()
            .any(|name| dir.join(name).exists())
            {
                return Err(invalid(
                    "unsupported quantized DFlash2 drafter; only compressed-tensors W4A16 / W4A16-HGPTQ or unquantized BF16/FP16 drafter weights are supported",
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
        ModelRole::Lora => {
            crate::hub::validate_lora_dir(dir)?;
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
    if crate::qwen_image::QwenImage21PipelineConfig::is_pipeline_dir(dir) {
        let tok_rel = [
            "processor/tokenizer.json",
            "tokenizer/tokenizer.json",
            "tokenizer.json",
        ]
        .into_iter()
        .find(|&cand| names.contains(cand) || dir.join(cand).is_file())
        .unwrap_or("processor/tokenizer.json");
        validate_files(dir, &[tok_rel.into()])?;
        validate_files(dir, &weight_files(dir, names)?)?;
        let pcfg = crate::qwen_image::QwenImage21PipelineConfig::from_dir(dir)?;
        pcfg.open_text_encoder_archive()?;
        pcfg.open_transformer_archive()?;
        pcfg.open_vae_archive()?;
        return Ok(());
    }
    if role == ModelRole::Target {
        validate_files(dir, &["tokenizer.json".into()])?;
    }
    validate_files(dir, &weight_files(dir, names)?)?;
    // Parse headers and check tensor bounds, without loading weights into RAM or touching CUDA.
    crate::loader::SafetensorsArchive::open(dir)?;
    if names.contains("readout.safetensors") {
        crate::loader::SafetensorsArchive::open(&dir.join("readout.safetensors"))?;
    }
    Ok(())
}
