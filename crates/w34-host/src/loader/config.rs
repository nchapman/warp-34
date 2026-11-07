use super::*;

#[path = "config_rules.rs"]
mod config_rules;
pub use config_rules::{
    CompressedTensorRules, HADAMARD_BLOCK_16, HADAMARD_BLOCK_64, SUPPORTED_HADAMARD_BLOCKS,
    SUPPORTED_MARLIN_U4_GROUP_SIZES, is_valid_hadamard_block_for_dim,
};

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum QuantScheme {
    Unquantized,
    CompressedTensors {
        num_bits: usize,
        /// `0` denotes per-channel quantization (`group_size == size_k`, `group_blocks == -1`).
        group_size: usize,
        symmetric: bool,
        ignore: Vec<String>,
        rules: Option<Box<CompressedTensorRules>>,
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

    /// Returns the group size if this scheme uses symmetric 4-bit `U4B8` packing
    /// (`Gptq`, `AutoRound`, or `CompressedTensors`).
    pub fn symmetric_gptq_u4_group_size(&self) -> Option<usize> {
        match self {
            Self::CompressedTensors {
                num_bits: 4,
                group_size,
                symmetric: true,
                ..
            }
            | Self::Gptq {
                bits: 4,
                group_size,
                sym: true,
                ..
            }
            | Self::AutoRound {
                bits: 4,
                group_size,
                sym: true,
                awq_packing: false,
            } if *group_size > 0 => Some(*group_size),
            _ => None,
        }
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
    pub mtp_use_dedicated_embeddings: bool,
    pub dflash_block_size: Option<usize>,
    pub dflash_target_layer_ids: Vec<usize>,
    pub eos_token_ids: Vec<i32>,
    pub quant: QuantScheme,
    pub decision_meta: DecisionConfigMeta,
    // Qwen 3.8 Flash-Next (`qwen4_exp`) extensions:
    pub hc_count: usize,
    pub hc_lowrank: usize,
    /// 1-based layer IDs from `config.json` (`ple_layer_ids`, e.g. `[2]` -> 0-based `layer_idx == 1`).
    pub ple_layer_ids: Vec<usize>,
    pub ple_embed_dim: usize,
    pub ple_conv_kernel_size: usize,
    pub ngram_size: usize,
    pub heads_per_ngram: usize,
    pub ngram_vocab_size_base: usize,
    pub make_ngram_vocab_size_divisible_by: usize,
    pub split_ngram_parts: usize,
    pub ple_embedding_dtype: Option<String>,
    pub ple_seed: u64,
    pub indexer_n_heads: usize,
    pub indexer_kv_heads: usize,
    pub indexer_head_dim: usize,
    pub indexer_budget: usize,
    pub indexer_compress_ratio: usize,
    pub output_gate_type: String,
}

impl ModelConfig {
    pub fn from_dir(dir: &Path) -> io::Result<Self> {
        let cfg_text = fs::read_to_string(dir.join("config.json"))?;
        let root: Json = serde_json::from_str(&cfg_text).map_err(|e| {
            io::Error::new(io::ErrorKind::InvalidData, format!("bad config.json: {e}"))
        })?;
        // This is the target-model loader. Drafter callers explicitly opt into
        // windowed attention through from_drafter_json, never artifact metadata.
        let mut cfg = Self::from_json(&root)?;
        let tc = root.get("text_config").unwrap_or(&root);
        if root.get("quantization_config").is_none_or(Json::is_null)
            && tc.get("quantization_config").is_none_or(Json::is_null)
            && let Some(standalone_qcfg) = [
                "quantize_config.json",
                "quant_config.json",
                "quantization_config.json",
            ]
            .into_iter()
            .map(|name| dir.join(name))
            .find(|p| p.exists())
        {
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
            if let Some(obj) = qjson.as_object_mut() {
                if !obj.contains_key("bits")
                    && let Some(b) = obj.get("w_bit").cloned()
                {
                    obj.insert("bits".to_string(), b);
                }
                if !obj.contains_key("group_size")
                    && let Some(g) = obj.get("q_group_size").cloned()
                {
                    obj.insert("group_size".to_string(), g);
                }
                if !obj.contains_key("quant_method") {
                    if obj.contains_key("zero_point") {
                        obj.insert("quant_method".to_string(), Json::String("awq".into()));
                    } else if obj.contains_key("bits") || obj.contains_key("group_size") {
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
            }
            cfg.quant = Self::parse_quant(Some(&qjson))?;
        }
        Ok(cfg)
    }

    /// Parses a backbone (target-model) `config.json` tree.
    ///
    /// Backbones must use full causal attention; `sliding_window` fails fast.
    pub fn from_json(root: &Json) -> io::Result<Self> {
        Self::from_json_with_policy(root, /* backbone_sliding_window_fatal = */ true)
    }

    /// Parses a DFlash2 drafter `transformer_layer_config` body.
    ///
    /// DFlash2 drafters legitimately pair sliding-window attention with
    /// full-attention target layers, and `crate::dflash::DFlash2Config`
    /// implements the windowed attention path, so the backbone-only
    /// sliding-window fail-fast is not applied here.
    pub fn from_drafter_json(root: &Json) -> io::Result<Self> {
        Self::from_json_with_policy(root, /* backbone_sliding_window_fatal = */ false)
    }

    fn from_json_with_policy(root: &Json, backbone_sliding_window_fatal: bool) -> io::Result<Self> {
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

        if let Some(act) = tc
            .get("hidden_act")
            .or_else(|| root.get("hidden_act"))
            .and_then(|v| v.as_str())
            && !act.eq_ignore_ascii_case("silu")
            && !act.eq_ignore_ascii_case("swiglu")
        {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!(
                    "unsupported hidden_act='{}' (expected 'silu')",
                    sanitize_diag(act)
                ),
            ));
        }

        if backbone_sliding_window_fatal
            && (tc
                .get("use_sliding_window")
                .or_else(|| root.get("use_sliding_window"))
                .and_then(|v| v.as_bool())
                == Some(true)
                || tc
                    .get("sliding_window")
                    .or_else(|| root.get("sliding_window"))
                    .and_then(|v| v.as_i64())
                    .unwrap_or(0)
                    > 0)
        {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "unsupported sliding_window in backbone attention (warp-34 targets full causal attention)",
            ));
        }

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
            .or_else(|| tc.get("mtp").and_then(|m| m.get("num_hidden_layers")))
            .and_then(|v| v.as_u64())
            .unwrap_or(0) as usize;
        let mtp_use_dedicated_embeddings = tc
            .get("mtp_use_dedicated_embeddings")
            .and_then(|v| v.as_bool())
            .unwrap_or(true);

        let hc_count = tc.get("hc_count").and_then(|v| v.as_u64()).unwrap_or(1) as usize;
        if hc_count == 0 {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "hc_count must be >= 1",
            ));
        }
        let hc_lowrank = tc
            .get("hc_lowrank")
            .and_then(|v| v.as_u64())
            .unwrap_or(if hc_count > 1 { 320 } else { 0 }) as usize;

        let ple_layer_ids: Vec<usize> = tc
            .get("ple_layer_ids")
            .and_then(|v| v.as_array())
            .map(|arr| {
                arr.iter()
                    .filter_map(|x| x.as_u64().map(|u| u as usize))
                    .collect()
            })
            .unwrap_or_default();
        for &abs_id in &ple_layer_ids {
            if abs_id == 0 || abs_id > num_hidden_layers {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    format!(
                        "invalid 1-based ple_layer_id={abs_id} for num_hidden_layers={num_hidden_layers}"
                    ),
                ));
            }
        }
        let ple_embed_dim = tc.get("ple_embed_dim").and_then(|v| v.as_u64()).unwrap_or(
            if ple_layer_ids.is_empty() {
                0
            } else {
                hidden_size as u64
            },
        ) as usize;
        let ple_conv_kernel_size =
            tc.get("ple_conv_kernel_size")
                .and_then(|v| v.as_u64())
                .unwrap_or(if ple_layer_ids.is_empty() { 0 } else { 4 }) as usize;
        let ngram_size =
            tc.get("ngram_size")
                .and_then(|v| v.as_u64())
                .unwrap_or(if ple_layer_ids.is_empty() { 0 } else { 3 }) as usize;
        let heads_per_ngram =
            tc.get("heads_per_ngram")
                .and_then(|v| v.as_u64())
                .unwrap_or(if ple_layer_ids.is_empty() { 0 } else { 8 }) as usize;
        let ngram_vocab_size_base = tc
            .get("ngram_vocab_size_base")
            .and_then(|v| v.as_u64())
            .unwrap_or(if ple_layer_ids.is_empty() {
                0
            } else {
                20_000_000
            }) as usize;
        let make_ngram_vocab_size_divisible_by = tc
            .get("make_ngram_vocab_size_divisible_by")
            .and_then(|v| v.as_u64())
            .unwrap_or(128) as usize;
        let split_ngram_parts =
            tc.get("split_ngram_parts")
                .and_then(|v| v.as_u64())
                .unwrap_or(if ple_layer_ids.is_empty() { 0 } else { 128 }) as usize;
        let ple_embedding_dtype = tc
            .get("ple_embedding_dtype")
            .and_then(|v| v.as_str())
            .map(String::from);
        let ple_seed = tc
            .get("seed")
            .or_else(|| root.get("seed"))
            .and_then(|v| v.as_u64())
            .unwrap_or(1234);

        let indexer_n_heads = tc
            .get("indexer_n_heads")
            .and_then(|v| v.as_u64())
            .unwrap_or(0) as usize;
        let indexer_kv_heads =
            tc.get("indexer_kv_heads")
                .and_then(|v| v.as_u64())
                .unwrap_or(if indexer_n_heads > 0 { 1 } else { 0 }) as usize;
        let indexer_head_dim =
            tc.get("indexer_head_dim")
                .and_then(|v| v.as_u64())
                .unwrap_or(if indexer_n_heads > 0 { 128 } else { 0 }) as usize;
        let indexer_budget =
            tc.get("indexer_budget")
                .and_then(|v| v.as_u64())
                .unwrap_or(if indexer_n_heads > 0 { 2048 } else { 0 }) as usize;
        let indexer_compress_ratio =
            tc.get("indexer_compress_ratio")
                .and_then(|v| v.as_u64())
                .unwrap_or(if indexer_n_heads > 0 { 4 } else { 0 }) as usize;
        if indexer_n_heads > 0 {
            if indexer_kv_heads != 1 || indexer_head_dim != 128 {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    format!(
                        "unsupported QSA indexer geometry (indexer_kv_heads={indexer_kv_heads}, indexer_head_dim={indexer_head_dim}; expected 1 and 128)"
                    ),
                ));
            }
            if indexer_compress_ratio != 4
                || indexer_budget == 0
                || !indexer_budget.is_multiple_of(indexer_compress_ratio)
            {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    format!(
                        "invalid QSA indexer budget/compression (indexer_budget={indexer_budget}, indexer_compress_ratio={indexer_compress_ratio}; expected compress_ratio=4)"
                    ),
                ));
            }
        }

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
        for default_eos in [248044i32, 248046i32] {
            if (default_eos as usize) < vocab_size && !eos_token_ids.contains(&default_eos) {
                eos_token_ids.push(default_eos);
            }
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

        let output_gate_type = tc
            .get("output_gate_type")
            .or_else(|| root.get("output_gate_type"))
            .and_then(|v| v.as_str())
            .unwrap_or("silu")
            .to_string();

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
            mtp_use_dedicated_embeddings,
            dflash_block_size,
            dflash_target_layer_ids,
            eos_token_ids,
            quant,
            decision_meta,
            hc_count,
            hc_lowrank,
            ple_layer_ids,
            ple_embed_dim,
            ple_conv_kernel_size,
            ngram_size,
            heads_per_ngram,
            ngram_vocab_size_base,
            make_ngram_vocab_size_divisible_by,
            split_ngram_parts,
            ple_embedding_dtype,
            ple_seed,
            indexer_n_heads,
            indexer_kv_heads,
            indexer_head_dim,
            indexer_budget,
            indexer_compress_ratio,
            output_gate_type,
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
                            "unsupported {method} 4-bit group_size={} (supported on sm_86/sm_89 Marlin: 32, 64, 128)",
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
                            "unsupported {method} 8-bit group_size={} (supported on sm_86/sm_89 Marlin: 32, 64, 128{})",
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
                    "unsupported {method} bit-width bits={bits} (supported on sm_86/sm_89: 4-bit W4A16 or 8-bit W8A16)"
                ),
            )),
        }
    }

    pub(crate) fn parse_quant(qcfg: Option<&Json>) -> io::Result<QuantScheme> {
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
            "compressed-tensors" | "compressed_tensors" => Self::parse_quant_compressed_tensors(q),
            "gptq" | "gptq_marlin" => Self::parse_quant_gptq(q),
            "awq" | "awq_marlin" => Self::parse_quant_awq(q),
            "auto-round" | "autoround" | "auto_round" => Self::parse_quant_auto_round(q),
            _ => Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!(
                    "unsupported quantization method '{}' (warp-34 sm_86/sm_89 supports: unquantized FP16/BF16/FP32, compressed-tensors pack-quantized W4A16/W8A16, gptq/auto-round symmetric W4A16/W8A16, and awq gemm W4A16)",
                    if raw_method.is_empty() {
                        "<missing quant_method>".to_string()
                    } else {
                        sanitize_diag(raw_method)
                    }
                ),
            )),
        }
    }

    fn parse_quant_compressed_tensors(q: &Json) -> io::Result<QuantScheme> {
        if let Some(fmt) = q.get("format").and_then(|v| v.as_str())
            && fmt != "pack-quantized"
        {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!(
                    "unsupported compressed-tensors format '{fmt}' (warp-34 sm_86/sm_89 requires weight-only 'pack-quantized' W4A16/W8A16; FP8/NVFP4/W8A8 are not supported)"
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
                    "unsupported compressed-tensors sparsity format '{sp_fmt}' (sm_86/sm_89 dense Marlin only)"
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

        let rules = CompressedTensorRules::parse(q)?;
        let mut chosen_linear: Option<(usize, usize, bool)> = None;
        let mut fallback_group: Option<(usize, usize, bool)> = None;
        let mut group_keys: Vec<&String> = groups_obj.keys().collect();
        group_keys.sort();
        for gk in group_keys {
            let grp = &groups_obj[gk];
            let (num_bits, group_size, symmetric, is_embedding_only) =
                Self::parse_compressed_tensors_group(gk, grp)?;
            if !is_embedding_only {
                if let Some((prev_b, prev_g, prev_s)) = chosen_linear {
                    if rules.is_none()
                        && (prev_b, prev_g, prev_s) != (num_bits, group_size, symmetric)
                    {
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
        let (num_bits, group_size, symmetric) =
            chosen_linear.or(fallback_group).ok_or_else(|| {
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
            rules: rules.map(Box::new),
        })
    }

    fn parse_compressed_tensors_group(
        gk: &str,
        grp: &Json,
    ) -> io::Result<(usize, usize, bool, bool)> {
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
                    "unsupported compressed-tensors activation quantization in '{}' (sm_86/sm_89 engine supports weight-only W4A16/W8A16)",
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
                    "unsupported compressed-tensors weights.type='{}' in '{}' (expected 'int'; FP8/FP4 are not enabled on sm_86/sm_89)",
                    sanitize_diag(wtype),
                    sanitize_diag(gk)
                ),
            ));
        }
        if let Some(ao) = weights.get("actorder") {
            let dynamic_ao = match ao {
                Json::Null | Json::Bool(false) => false,
                Json::Bool(true) => true,
                Json::String(s) => !matches!(s.as_str(), "none" | "static" | "false" | ""),
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
        if !(num_bits == 4 && group_size == 0 && strategy == "channel" && symmetric) {
            Self::validate_bits_and_group_size("compressed-tensors", num_bits, group_size, true)?;
        }
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
        Ok((num_bits, group_size, symmetric, is_embedding_only))
    }

    fn parse_quant_gptq(q: &Json) -> io::Result<QuantScheme> {
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

    fn parse_quant_awq(q: &Json) -> io::Result<QuantScheme> {
        let bits = q
            .get("bits")
            .or_else(|| q.get("w_bit"))
            .and_then(|v| v.as_u64())
            .unwrap_or(4) as usize;
        let group_size =
            Self::parse_group_size_i64(q.get("group_size").or_else(|| q.get("q_group_size")), 128)?;
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
                format!("unsupported AWQ version '{version}' (expected 'gemm' or 'marlin')"),
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

    fn parse_quant_auto_round(q: &Json) -> io::Result<QuantScheme> {
        if let Some(dt) = q.get("data_type").and_then(|v| v.as_str())
            && !dt.to_ascii_lowercase().starts_with("int")
        {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!("unsupported auto-round data_type='{dt}' (expected integer W4A16/W8A16)"),
            ));
        }
        if let Some(act_b) = q.get("act_bits").and_then(|v| v.as_u64())
            && act_b < 16
        {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!(
                    "unsupported auto-round act_bits={act_b} (sm_86/sm_89 engine requires weight-only W4A16/W8A16 with act_bits >= 16)"
                ),
            ));
        }
        let packing_fmt = q
            .get("packing_format")
            .or_else(|| q.get("backend"))
            .or_else(|| q.get("vllm_backend"))
            .and_then(|v| v.as_str())
            .unwrap_or("auto_round:auto_gptq")
            .trim()
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
        if let Some(ec) = q
            .get("extra_config")
            .or_else(|| q.get("layer_config"))
            .and_then(|v| v.as_object())
        {
            Self::validate_auto_round_extra_config(ec, bits, group_size, sym, awq_packing)?;
        }
        Ok(QuantScheme::AutoRound {
            bits,
            group_size,
            sym,
            awq_packing,
        })
    }

    fn validate_auto_round_extra_config(
        ec: &serde_json::Map<String, Json>,
        bits: usize,
        group_size: usize,
        sym: bool,
        awq_packing: bool,
    ) -> io::Result<()> {
        for (k, v) in ec {
            let layer_bits = v
                .get("bits")
                .and_then(|x| x.as_u64())
                .map(|x| x as usize)
                .unwrap_or(bits);
            if layer_bits >= 16 {
                continue;
            }
            if layer_bits != bits {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    format!(
                        "unsupported mixed-bit AutoRound override in '{}': {layer_bits}-bit (model base is {bits}-bit)",
                        sanitize_diag(k)
                    ),
                ));
            }
            if let Some(dt) = v.get("data_type").and_then(|x| x.as_str())
                && !dt.to_ascii_lowercase().starts_with("int")
            {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    format!(
                        "unsupported auto-round data_type='{dt}' in '{}' (expected integer W4A16/W8A16)",
                        sanitize_diag(k)
                    ),
                ));
            }
            if let Some(act_b) = v.get("act_bits").and_then(|x| x.as_u64())
                && act_b < 16
            {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    format!(
                        "unsupported auto-round act_bits={act_b} in '{}'",
                        sanitize_diag(k)
                    ),
                ));
            }
            let layer_gs = if v.get("group_size").is_some() {
                Self::parse_group_size_i64(v.get("group_size"), group_size as i64)?
            } else {
                group_size
            };
            Self::validate_bits_and_group_size("auto-round", layer_bits, layer_gs, !awq_packing)?;
            let layer_sym = v.get("sym").and_then(|x| x.as_bool()).unwrap_or(sym);
            if !layer_sym && !(awq_packing && layer_bits == 4) {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    format!(
                        "unsupported auto-round sym=false override in '{}' with GPTQ packing",
                        sanitize_diag(k)
                    ),
                ));
            }
        }
        Ok(())
    }

    pub fn num_full_attention_layers(&self) -> usize {
        self.layer_types
            .iter()
            .filter(|&&k| k == LayerKind::FullAttention)
            .count()
    }

    pub const fn kv_heads_compatible_with_tp(&self, tp_size: usize) -> bool {
        if self.num_key_value_heads == 0
            || tp_size == 0
            || (tp_size > 1 && self.num_key_value_heads < 2)
        {
            false
        } else if self.num_key_value_heads >= tp_size {
            self.num_key_value_heads.is_multiple_of(tp_size)
        } else {
            tp_size.is_multiple_of(self.num_key_value_heads)
        }
    }

    pub const fn local_kv_heads(&self, tp_size: usize) -> usize {
        if tp_size <= 1 || self.num_key_value_heads == 0 {
            self.num_key_value_heads
        } else if self.num_key_value_heads >= tp_size {
            self.num_key_value_heads / tp_size
        } else {
            1
        }
    }

    pub const fn kv_head_shard_index(&self, rank: usize, tp_size: usize) -> usize {
        if tp_size <= 1 || self.num_key_value_heads == 0 || self.num_key_value_heads >= tp_size {
            rank
        } else {
            let rep = tp_size / self.num_key_value_heads;
            rank / rep
        }
    }

    pub fn num_linear_attention_layers(&self) -> usize {
        self.layer_types
            .iter()
            .filter(|&&k| k == LayerKind::LinearAttention)
            .count()
    }

    pub fn is_qwen4_exp(&self) -> bool {
        matches!(self.model_type.as_str(), "qwen4_exp" | "qwen4_exp_text")
            || self
                .architectures
                .iter()
                .any(|a| a == "Qwen4ExpForConditionalGeneration")
    }

    pub fn uses_hyper_connection(&self) -> bool {
        self.hc_count > 1
    }

    pub fn hyper_hidden_size(&self) -> usize {
        self.hidden_size * self.hc_count.max(1)
    }

    pub fn uses_qsa(&self) -> bool {
        self.indexer_n_heads > 0 && self.indexer_head_dim > 0
    }

    pub fn uses_ple(&self) -> bool {
        self.ple_embed_dim > 0 && !self.ple_layer_ids.is_empty()
    }

    /// Resolves the effective `QuantScheme` for routed MoE experts (`mlp.experts.0.gate_proj`).
    pub fn routed_expert_quant_scheme(&self) -> QuantScheme {
        for candidate in [
            "model.language_model.layers.0.mlp.experts.0.gate_proj",
            "model.layers.0.mlp.experts.0.gate_proj",
        ] {
            if let Ok(resolved) = self.quant.for_linear(candidate)
                && resolved != QuantScheme::Unquantized
            {
                return resolved;
            }
        }
        self.quant.clone()
    }

    /// Returns the `ExpertPackLayout` if this model's MoE geometry supports host/VRAM
    /// tiered `U4B8` expert storage.
    pub fn moe_expert_pack_layout(&self) -> Option<crate::moe_offload::ExpertPackLayout> {
        if self.num_experts == 0 {
            return None;
        }
        let expert_quant = self.routed_expert_quant_scheme();
        let group_size = expert_quant
            .symmetric_gptq_u4_group_size()
            .unwrap_or_else(|| expert_quant.group_size_or(32));
        crate::moe_offload::ExpertPackLayout::new(
            self.moe_intermediate_size,
            self.hidden_size,
            group_size,
        )
        .ok()
    }

    /// Total byte footprint of all routed experts across all MoE decoder layers
    /// when stored in symmetric `U4B8` `ExpertPackLayout` slots.
    pub fn routed_expert_total_bytes(&self) -> usize {
        let Some(layout) = self.moe_expert_pack_layout() else {
            return 0;
        };
        let moe_layers = self
            .num_hidden_layers
            .saturating_sub(self.mlp_only_layers.len());
        moe_layers
            .saturating_mul(self.num_experts)
            .saturating_mul(layout.stride)
    }

    pub fn uses_tiered_moe(&self) -> bool {
        // Use host/VRAM tiered MoE storage whenever all layers are MoE layers with a compatible
        // symmetric U4B8 expert layout and either:
        // - the total routed-expert footprint exceeds a single 24 GiB GPU budget (> 20 GiB), or
        // - the model uses the Qwen4 high-expert-count architecture (including small synthetic test fixtures).
        const TIERED_MOE_MIN_TOTAL_EXPERT_BYTES: usize = 20 * 1024 * 1024 * 1024;
        self.mlp_only_layers.is_empty()
            && self.moe_expert_pack_layout().is_some()
            && ((self
                .routed_expert_quant_scheme()
                .symmetric_gptq_u4_group_size()
                .is_some()
                && self.routed_expert_total_bytes() > TIERED_MOE_MIN_TOTAL_EXPERT_BYTES)
                || self.is_qwen4_exp())
    }

    pub fn qsa_output_width(&self) -> usize {
        if self.uses_qsa() {
            self.indexer_budget + self.indexer_compress_ratio.saturating_sub(1)
        } else {
            0
        }
    }

    /// Sorted unique 0-based layer indices that host a PLE module.
    pub fn ple_layer_indices(&self) -> Vec<usize> {
        let mut indices: Vec<usize> = self
            .ple_layer_ids
            .iter()
            .filter_map(|&abs_id| abs_id.checked_sub(1))
            .collect();
        indices.sort_unstable();
        indices.dedup();
        indices
    }

    pub fn num_ple_layers(&self) -> usize {
        self.ple_layer_indices().len()
    }

    /// Returns the dense `0..num_ple_layers` index if `layer_idx` (0-based) owns a PLE layer.
    pub fn ple_dense_layer_index(&self, layer_idx: usize) -> Option<usize> {
        self.ple_layer_indices().binary_search(&layer_idx).ok()
    }

    /// Number of prior dilated conv steps retained in `StateBundle::ple_conv_state`:
    /// `(ple_conv_kernel_size - 1) * ngram_size` (`(4 - 1) * 3 = 9` for `Qwen3.8-Flash-Next`).
    pub fn ple_conv_state_len(&self) -> usize {
        if self.ple_conv_kernel_size <= 1 || self.ngram_size == 0 {
            0
        } else {
            (self.ple_conv_kernel_size - 1) * self.ngram_size
        }
    }
}
