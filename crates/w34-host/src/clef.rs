//! Cloudflare Clef (`Cloudflare/clef-flash`, `Cloudflare/clef`) joint-schema decision head
//! and record encoder (`joint_schema_model.py`, `joint_head_config.json`, `joint_head.safetensors`).
//!
//! Clef packs all questions in a `/v1/systemone` or `/v1/decisions` request into a single prompt
//! (`STATE` + optional `<|vision_start|><|image_pad|>...<|vision_end|>\n` media prefix +
//! `SCHEMA FIELDS` with tracked question/option token spans + `JOINT SCHEMA DECISIONS:` assistant
//! boundary), runs one causal backbone prefill pass, and scores all options of all questions
//! jointly via `JointSchemaHead` (evidence-routing cross-attention + joint field transformer
//! decoder + lexical prior & residual cosine scorer).

use std::collections::HashMap;
use std::fs;
use std::path::Path;
use std::sync::{Arc, RwLock};

use serde_json::{Map, Value};
use tokenizers::Tokenizer;

use crate::decision::{DecisionError, DecisionField, kev_sanitize_special_tokens};
use crate::loader::{ModelConfig, SafetensorsArchive};

pub const CLEF_DEFAULT_MAX_LENGTH: usize = 16_384;
pub const CLEF_DEFAULT_MIN_PIXELS: usize = 65_536;
pub const CLEF_DEFAULT_MAX_PIXELS: usize = 1024 * 1024;

pub const CLEF_SYSTEM_PROMPT: &str = "Read the complete state and schema. Decide every field jointly. Each answer must be exactly one of that field's allowed options.";

const CLEF_NOUL_TRUE_DEFAULT: &str = "The proposition is true or the answer is yes.";
const CLEF_NOUL_FALSE_DEFAULT: &str = "The proposition is false or the answer is no.";

unsafe extern "C" {
    fn erf(x: f64) -> f64;
}

#[inline]
fn exact_gelu(x: f32) -> f32 {
    let xd = x as f64;
    // SAFETY: libm `erf` is a pure scalar `f64 -> f64` math function with no pointer arguments.
    let cdf = 0.5 * (1.0 + unsafe { erf(xd * std::f64::consts::FRAC_1_SQRT_2) });
    (xd * cdf) as f32
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ClefHeadConfig {
    pub hidden_size: usize,
    pub width: usize,
    pub routing_layers: usize,
    pub layers: usize,
    pub heads: usize,
    pub feedforward: usize,
}

impl ClefHeadConfig {
    pub fn from_json(v: &Value) -> Result<Self, String> {
        let obj = v
            .as_object()
            .ok_or_else(|| "joint_head_config.json must be a JSON object".to_string())?;
        let req_usize = |key: &str| -> Result<usize, String> {
            obj.get(key)
                .and_then(Value::as_u64)
                .map(|n| n as usize)
                .ok_or_else(|| format!("joint_head_config.json missing integer '{key}'"))
        };
        let cfg = Self {
            hidden_size: req_usize("hidden_size")?,
            width: req_usize("width")?,
            routing_layers: req_usize("routing_layers")?,
            layers: req_usize("layers")?,
            heads: req_usize("heads")?,
            feedforward: req_usize("feedforward")?,
        };
        cfg.validate()?;
        Ok(cfg)
    }

    pub fn from_path(path: &Path) -> Result<Self, String> {
        let text =
            fs::read_to_string(path).map_err(|e| format!("clef read {}: {e}", path.display()))?;
        let val: Value = serde_json::from_str(&text)
            .map_err(|e| format!("clef parse {}: {e}", path.display()))?;
        Self::from_json(&val)
    }

    pub fn validate(&self) -> Result<(), String> {
        if self.hidden_size == 0
            || self.width == 0
            || self.heads == 0
            || self.feedforward == 0
            || !self.width.is_multiple_of(self.heads)
        {
            return Err(format!(
                "clef invalid head geometry: hidden_size={}, width={}, heads={}, feedforward={}",
                self.hidden_size, self.width, self.heads, self.feedforward
            ));
        }
        if self.routing_layers > 32 || self.layers > 32 {
            return Err(format!(
                "clef unsupported layer counts: routing_layers={}, layers={}",
                self.routing_layers, self.layers
            ));
        }
        Ok(())
    }
}

/// Inspects `processor_config.json` or `preprocessor_config.json` for `shortest_edge`
/// (`min_pixels`) and `longest_edge` (`max_pixels`), clamping `max_pixels` to
/// `CLEF_DEFAULT_MAX_PIXELS` (`1024 * 1024`) to prevent pathological image allocations.
pub fn detect_clef_image_pixel_bounds(dir: &Path) -> (usize, usize) {
    for fname in ["processor_config.json", "preprocessor_config.json"] {
        let path = dir.join(fname);
        let Ok(text) = fs::read_to_string(&path) else {
            continue;
        };
        let Ok(val) = serde_json::from_str::<Value>(&text) else {
            continue;
        };
        let size = val
            .pointer("/image_processor/size")
            .or_else(|| val.get("size"));
        if let Some(sz) = size {
            let min_px = sz
                .get("shortest_edge")
                .and_then(Value::as_u64)
                .map(|n| (n as usize).clamp(1024, CLEF_DEFAULT_MAX_PIXELS))
                .unwrap_or(CLEF_DEFAULT_MIN_PIXELS);
            let max_px = sz
                .get("longest_edge")
                .and_then(Value::as_u64)
                .map(|n| (n as usize).clamp(min_px, CLEF_DEFAULT_MAX_PIXELS))
                .unwrap_or(CLEF_DEFAULT_MAX_PIXELS);
            return (min_px, max_px);
        }
    }
    (CLEF_DEFAULT_MIN_PIXELS, CLEF_DEFAULT_MAX_PIXELS)
}

const MAX_CACHED_LEXICAL_ROWS: usize = 4096;

/// On-demand lookup into `base_model.get_output_embeddings().weight` (`lm_head.weight` or
/// tied `model.embed_tokens.weight`) so `ClefJointHead` reads only the ~10–100 unique option
/// token rows per request via `pread` without holding a 2–4 GiB dense host copy.
pub enum ClefOutputEmbeddings {
    Archive {
        archive: Arc<SafetensorsArchive>,
        tensor_name: String,
        vocab_size: usize,
        hidden_size: usize,
        row_cache: RwLock<HashMap<usize, Arc<[f32]>>>,
    },
    InMemory {
        weights: Vec<f32>,
        vocab_size: usize,
        hidden_size: usize,
    },
}

impl std::fmt::Debug for ClefOutputEmbeddings {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Archive {
                tensor_name,
                vocab_size,
                hidden_size,
                ..
            } => f
                .debug_struct("ClefOutputEmbeddings::Archive")
                .field("tensor_name", tensor_name)
                .field("vocab_size", vocab_size)
                .field("hidden_size", hidden_size)
                .finish(),
            Self::InMemory {
                vocab_size,
                hidden_size,
                ..
            } => f
                .debug_struct("ClefOutputEmbeddings::InMemory")
                .field("vocab_size", vocab_size)
                .field("hidden_size", hidden_size)
                .finish(),
        }
    }
}

impl ClefOutputEmbeddings {
    pub fn from_in_memory(
        weights: Vec<f32>,
        vocab_size: usize,
        hidden_size: usize,
    ) -> Result<Self, String> {
        if vocab_size == 0
            || hidden_size == 0
            || weights.len() != vocab_size.saturating_mul(hidden_size)
        {
            return Err(format!(
                "clef in-memory output_embeddings len {} != vocab_size ({vocab_size}) * hidden_size ({hidden_size})",
                weights.len()
            ));
        }
        Ok(Self::InMemory {
            weights,
            vocab_size,
            hidden_size,
        })
    }

    pub fn from_archive_and_config(
        archive: Arc<SafetensorsArchive>,
        cfg: &ModelConfig,
    ) -> Result<Self, String> {
        let candidates: &[&str] = if cfg.tie_word_embeddings {
            &[
                "model.embed_tokens.weight",
                "model.language_model.embed_tokens.weight",
                "lm_head.weight",
            ]
        } else {
            &[
                "lm_head.weight",
                "model.language_model.lm_head.weight",
                "model.embed_tokens.weight",
                "model.language_model.embed_tokens.weight",
            ]
        };
        let tensor_name = candidates
            .iter()
            .find_map(|&name| {
                archive
                    .contains(name)
                    .then(|| name.to_string())
                    .or_else(|| archive.resolve_name(name))
            })
            .ok_or_else(|| {
                "clef: missing lm_head.weight / embed_tokens.weight in backbone archive".to_string()
            })?;
        let info = archive
            .tensor_info(&tensor_name)
            .map_err(|e| format!("clef tensor_info({tensor_name}): {e}"))?;
        if info.shape.len() != 2 || info.shape[1] != cfg.hidden_size || info.shape[0] == 0 {
            return Err(format!(
                "clef {tensor_name} shape {:?} incompatible with hidden_size={}",
                info.shape, cfg.hidden_size
            ));
        }
        Ok(Self::Archive {
            vocab_size: info.shape[0],
            hidden_size: info.shape[1],
            archive,
            tensor_name,
            row_cache: RwLock::new(HashMap::new()),
        })
    }

    pub fn hidden_size(&self) -> usize {
        match self {
            Self::Archive { hidden_size, .. } | Self::InMemory { hidden_size, .. } => *hidden_size,
        }
    }

    pub fn mean_rows(&self, token_ids: &[i32]) -> Result<Vec<f32>, String> {
        if token_ids.is_empty() {
            return Err("clef: empty option span for lexical embedding lookup".into());
        }
        let d = self.hidden_size();
        let mut acc = vec![0.0f64; d];
        match self {
            Self::InMemory {
                weights,
                vocab_size,
                ..
            } => {
                for &tid in token_ids {
                    if tid < 0 || (tid as usize) >= *vocab_size {
                        return Err(format!(
                            "clef: token_id {tid} out of range for vocab_size {vocab_size}"
                        ));
                    }
                    let off = (tid as usize) * d;
                    for (a, &w) in acc.iter_mut().zip(&weights[off..off + d]) {
                        *a += w as f64;
                    }
                }
            }
            Self::Archive {
                archive,
                tensor_name,
                vocab_size,
                row_cache,
                ..
            } => {
                for &tid in token_ids {
                    if tid < 0 || (tid as usize) >= *vocab_size {
                        return Err(format!(
                            "clef: token_id {tid} out of range for vocab_size {vocab_size}"
                        ));
                    }
                    let row_idx = tid as usize;
                    let cached = row_cache.read().ok().and_then(|g| g.get(&row_idx).cloned());
                    let row: Arc<[f32]> = match cached {
                        Some(r) => r,
                        None => {
                            let vec = archive.read_row_f32(tensor_name, row_idx).map_err(|e| {
                                format!("clef read_row_f32({tensor_name}, {row_idx}): {e}")
                            })?;
                            let arc: Arc<[f32]> = vec.into();
                            if let Ok(mut g) = row_cache.write() {
                                if g.len() >= MAX_CACHED_LEXICAL_ROWS {
                                    g.clear();
                                }
                                g.insert(row_idx, arc.clone());
                            }
                            arc
                        }
                    };
                    for (a, &w) in acc.iter_mut().zip(row.iter()) {
                        *a += w as f64;
                    }
                }
            }
        }
        let inv = 1.0 / (token_ids.len() as f64);
        Ok(acc.into_iter().map(|v| (v * inv) as f32).collect())
    }
}

#[derive(Clone, Debug)]
struct LayerNormWeights {
    weight: Vec<f32>,
    bias: Vec<f32>,
}

impl LayerNormWeights {
    fn load(archive: &SafetensorsArchive, prefix: &str, dim: usize) -> Result<Self, String> {
        let w_name = format!("{prefix}.weight");
        let b_name = format!("{prefix}.bias");
        let weight = read_exact_1d(archive, &w_name, dim)?;
        let bias = read_exact_1d(archive, &b_name, dim)?;
        Ok(Self { weight, bias })
    }

    fn apply_row(&self, input: &[f32], out: &mut [f32]) {
        let d = self.weight.len();
        debug_assert_eq!(input.len(), d);
        debug_assert_eq!(out.len(), d);
        let inv_d = 1.0 / (d as f64);
        let mean = input.iter().map(|&x| x as f64).sum::<f64>() * inv_d;
        let var = input
            .iter()
            .map(|&x| {
                let diff = (x as f64) - mean;
                diff * diff
            })
            .sum::<f64>()
            * inv_d;
        let inv_std = 1.0 / (var + 1e-5).sqrt();
        for i in 0..d {
            let norm = ((input[i] as f64) - mean) * inv_std;
            out[i] = (norm * (self.weight[i] as f64) + (self.bias[i] as f64)) as f32;
        }
    }

    fn apply_matrix(&self, input: &[f32], rows: usize) -> Vec<f32> {
        let d = self.weight.len();
        debug_assert_eq!(input.len(), rows * d);
        let mut out = vec![0.0f32; rows * d];
        for r in 0..rows {
            let off = r * d;
            self.apply_row(&input[off..off + d], &mut out[off..off + d]);
        }
        out
    }
}

#[derive(Clone, Debug)]
struct LinearWeights {
    out_dim: usize,
    in_dim: usize,
    weight: Vec<f32>,
    bias: Option<Vec<f32>>,
}

impl LinearWeights {
    fn load(
        archive: &SafetensorsArchive,
        prefix: &str,
        out_dim: usize,
        in_dim: usize,
        has_bias: bool,
    ) -> Result<Self, String> {
        let w_name = format!("{prefix}.weight");
        let weight = read_exact_2d(archive, &w_name, out_dim, in_dim)?;
        let bias = if has_bias {
            let b_name = format!("{prefix}.bias");
            Some(read_exact_1d(archive, &b_name, out_dim)?)
        } else {
            None
        };
        Ok(Self {
            out_dim,
            in_dim,
            weight,
            bias,
        })
    }

    fn apply(&self, input: &[f32], rows: usize) -> Vec<f32> {
        matmul_nt(
            input,
            rows,
            self.in_dim,
            &self.weight,
            self.out_dim,
            self.bias.as_deref(),
        )
    }
}

#[derive(Clone, Debug)]
struct MultiheadAttnWeights {
    width: usize,
    heads: usize,
    q_proj: LinearWeights,
    kv_proj: LinearWeights,
    out_proj: LinearWeights,
}

impl MultiheadAttnWeights {
    fn load(
        archive: &SafetensorsArchive,
        prefix: &str,
        width: usize,
        heads: usize,
    ) -> Result<Self, String> {
        let w_name = format!("{prefix}.in_proj_weight");
        let b_name = format!("{prefix}.in_proj_bias");
        let in_w = read_exact_2d(archive, &w_name, 3 * width, width)?;
        let in_b = read_exact_1d(archive, &b_name, 3 * width)?;
        let q_proj = LinearWeights {
            out_dim: width,
            in_dim: width,
            weight: in_w[..width * width].to_vec(),
            bias: Some(in_b[..width].to_vec()),
        };
        let kv_proj = LinearWeights {
            out_dim: 2 * width,
            in_dim: width,
            weight: in_w[width * width..].to_vec(),
            bias: Some(in_b[width..].to_vec()),
        };
        let out_proj =
            LinearWeights::load(archive, &format!("{prefix}.out_proj"), width, width, true)?;
        Ok(Self {
            width,
            heads,
            q_proj,
            kv_proj,
            out_proj,
        })
    }

    fn forward(&self, queries: &[f32], q_len: usize, memory: &[f32], m_len: usize) -> Vec<f32> {
        let w = self.width;
        let h = self.heads;
        let hd = w / h;
        let scale = 1.0 / ((hd as f64).sqrt());

        let q = self.q_proj.apply(queries, q_len);
        let kv = self.kv_proj.apply(memory, m_len);

        let mut context = vec![0.0f32; q_len * w];
        let mut scores = vec![0.0f64; m_len];

        for qi in 0..q_len {
            let q_row = &q[qi * w..(qi + 1) * w];
            let ctx_row = &mut context[qi * w..(qi + 1) * w];
            for hi in 0..h {
                let h_off = hi * hd;
                let q_head = &q_row[h_off..h_off + hd];
                let mut max_s = f64::NEG_INFINITY;
                for mi in 0..m_len {
                    let k_head = &kv[mi * (2 * w) + h_off..mi * (2 * w) + h_off + hd];
                    let s = (dot_f32(q_head, k_head) as f64) * scale;
                    scores[mi] = s;
                    if s > max_s {
                        max_s = s;
                    }
                }
                let mut sum_exp = 0.0f64;
                for s in &mut scores {
                    let e = (*s - max_s).exp();
                    *s = e;
                    sum_exp += e;
                }
                let inv_sum = 1.0 / sum_exp;
                let mut out_head = vec![0.0f64; hd];
                for (mi, &e) in scores.iter().enumerate() {
                    let prob = e * inv_sum;
                    let v_base = mi * (2 * w) + w + h_off;
                    let v_head = &kv[v_base..v_base + hd];
                    for d in 0..hd {
                        out_head[d] += prob * (v_head[d] as f64);
                    }
                }
                for d in 0..hd {
                    ctx_row[h_off + d] = out_head[d] as f32;
                }
            }
        }
        self.out_proj.apply(&context, q_len)
    }
}

#[derive(Clone, Debug)]
struct EvidenceRoutingLayerWeights {
    query_norm: LayerNormWeights,
    memory_norm: LayerNormWeights,
    attention: MultiheadAttnWeights,
    feedforward_norm: LayerNormWeights,
    ff0: LinearWeights,
    ff3: LinearWeights,
}

impl EvidenceRoutingLayerWeights {
    fn load(
        archive: &SafetensorsArchive,
        prefix: &str,
        width: usize,
        heads: usize,
        feedforward: usize,
    ) -> Result<Self, String> {
        Ok(Self {
            query_norm: LayerNormWeights::load(archive, &format!("{prefix}.query_norm"), width)?,
            memory_norm: LayerNormWeights::load(archive, &format!("{prefix}.memory_norm"), width)?,
            attention: MultiheadAttnWeights::load(
                archive,
                &format!("{prefix}.attention"),
                width,
                heads,
            )?,
            feedforward_norm: LayerNormWeights::load(
                archive,
                &format!("{prefix}.feedforward_norm"),
                width,
            )?,
            ff0: LinearWeights::load(
                archive,
                &format!("{prefix}.feedforward.0"),
                feedforward,
                width,
                true,
            )?,
            ff3: LinearWeights::load(
                archive,
                &format!("{prefix}.feedforward.3"),
                width,
                feedforward,
                true,
            )?,
        })
    }

    fn forward(&self, queries: &[f32], q_len: usize, memory: &[f32], m_len: usize) -> Vec<f32> {
        let norm_q = self.query_norm.apply_matrix(queries, q_len);
        let norm_m = self.memory_norm.apply_matrix(memory, m_len);
        let routed = self.attention.forward(&norm_q, q_len, &norm_m, m_len);
        let mut x: Vec<f32> = queries.iter().zip(&routed).map(|(&a, &b)| a + b).collect();
        let ff_in = self.feedforward_norm.apply_matrix(&x, q_len);
        let mut h = self.ff0.apply(&ff_in, q_len);
        for v in &mut h {
            *v = exact_gelu(*v);
        }
        let ff_out = self.ff3.apply(&h, q_len);
        for (xi, &fi) in x.iter_mut().zip(&ff_out) {
            *xi += fi;
        }
        x
    }
}

#[derive(Clone, Debug)]
struct DecoderLayerWeights {
    norm1: LayerNormWeights,
    self_attn: MultiheadAttnWeights,
    norm2: LayerNormWeights,
    multihead_attn: MultiheadAttnWeights,
    norm3: LayerNormWeights,
    linear1: LinearWeights,
    linear2: LinearWeights,
}

impl DecoderLayerWeights {
    fn load(
        archive: &SafetensorsArchive,
        prefix: &str,
        width: usize,
        heads: usize,
        feedforward: usize,
    ) -> Result<Self, String> {
        Ok(Self {
            norm1: LayerNormWeights::load(archive, &format!("{prefix}.norm1"), width)?,
            self_attn: MultiheadAttnWeights::load(
                archive,
                &format!("{prefix}.self_attn"),
                width,
                heads,
            )?,
            norm2: LayerNormWeights::load(archive, &format!("{prefix}.norm2"), width)?,
            multihead_attn: MultiheadAttnWeights::load(
                archive,
                &format!("{prefix}.multihead_attn"),
                width,
                heads,
            )?,
            norm3: LayerNormWeights::load(archive, &format!("{prefix}.norm3"), width)?,
            linear1: LinearWeights::load(
                archive,
                &format!("{prefix}.linear1"),
                feedforward,
                width,
                true,
            )?,
            linear2: LinearWeights::load(
                archive,
                &format!("{prefix}.linear2"),
                width,
                feedforward,
                true,
            )?,
        })
    }

    fn forward(&self, tgt: &[f32], q_len: usize, memory: &[f32], m_len: usize) -> Vec<f32> {
        let n1 = self.norm1.apply_matrix(tgt, q_len);
        let sa = self.self_attn.forward(&n1, q_len, &n1, q_len);
        let mut x: Vec<f32> = tgt.iter().zip(&sa).map(|(&a, &b)| a + b).collect();

        let n2 = self.norm2.apply_matrix(&x, q_len);
        let ca = self.multihead_attn.forward(&n2, q_len, memory, m_len);
        for (xi, &ci) in x.iter_mut().zip(&ca) {
            *xi += ci;
        }

        let n3 = self.norm3.apply_matrix(&x, q_len);
        let mut h = self.linear1.apply(&n3, q_len);
        for v in &mut h {
            *v = exact_gelu(*v);
        }
        let ff = self.linear2.apply(&h, q_len);
        for (xi, &fi) in x.iter_mut().zip(&ff) {
            *xi += fi;
        }
        x
    }
}

#[derive(Debug)]
pub struct ClefJointHead {
    pub config: ClefHeadConfig,
    hidden_norm: LayerNormWeights,
    memory_projection: LinearWeights,
    question_projection: LinearWeights,
    option_question_projection: LinearWeights,
    global_projection: LinearWeights,
    option_context_projection: LinearWeights,
    option_lexical_projection: LinearWeights,
    type_embedding: Vec<f32>,
    evidence_layers: Vec<EvidenceRoutingLayerWeights>,
    option_summary_norm: LayerNormWeights,
    layers: Vec<DecoderLayerWeights>,
    field_norm: LayerNormWeights,
    option_norm: LayerNormWeights,
    residual_0: LinearWeights,
    residual_3: LinearWeights,
    prior_logit_scale: f32,
    joint_logit_scale: f32,
    residual_gate: f32,
    output_embeddings: ClefOutputEmbeddings,
}

impl ClefJointHead {
    pub fn from_dir_and_config(dir: &Path, model_cfg: &ModelConfig) -> Result<Self, String> {
        let cfg_path = dir.join("joint_head_config.json");
        let head_cfg = ClefHeadConfig::from_path(&cfg_path)?;
        if head_cfg.hidden_size != model_cfg.hidden_size {
            return Err(format!(
                "clef joint_head_config hidden_size ({}) != backbone hidden_size ({})",
                head_cfg.hidden_size, model_cfg.hidden_size
            ));
        }
        let head_path = dir.join("joint_head.safetensors");
        let head_archive = SafetensorsArchive::open(&head_path)
            .map_err(|e| format!("clef open {}: {e}", head_path.display()))?;
        let backbone_archive = Arc::new(
            SafetensorsArchive::open(dir)
                .map_err(|e| format!("clef open backbone archive {}: {e}", dir.display()))?,
        );
        let out_emb = ClefOutputEmbeddings::from_archive_and_config(backbone_archive, model_cfg)?;
        Self::from_archive(&head_archive, head_cfg, out_emb)
    }

    pub fn from_archive(
        archive: &SafetensorsArchive,
        config: ClefHeadConfig,
        output_embeddings: ClefOutputEmbeddings,
    ) -> Result<Self, String> {
        config.validate()?;
        if output_embeddings.hidden_size() != config.hidden_size {
            return Err(format!(
                "clef output_embeddings hidden_size ({}) != config hidden_size ({})",
                output_embeddings.hidden_size(),
                config.hidden_size
            ));
        }
        let h = config.hidden_size;
        let w = config.width;
        let ff = config.feedforward;
        let heads = config.heads;

        let hidden_norm = LayerNormWeights::load(archive, "hidden_norm", h)?;
        let memory_projection = LinearWeights::load(archive, "memory_projection", w, h, false)?;
        let question_projection = LinearWeights::load(archive, "question_projection", w, h, false)?;
        let option_question_projection =
            LinearWeights::load(archive, "option_question_projection", w, h, false)?;
        let global_projection = LinearWeights::load(archive, "global_projection", w, h, false)?;
        let option_context_projection =
            LinearWeights::load(archive, "option_context_projection", w, h, false)?;
        let option_lexical_projection =
            LinearWeights::load(archive, "option_lexical_projection", w, h, false)?;
        let type_embedding = read_exact_2d(archive, "type_embedding.weight", 3, w)?;

        let mut evidence_layers = Vec::with_capacity(config.routing_layers);
        for i in 0..config.routing_layers {
            evidence_layers.push(EvidenceRoutingLayerWeights::load(
                archive,
                &format!("evidence_layers.{i}"),
                w,
                heads,
                ff,
            )?);
        }

        let option_summary_norm = LayerNormWeights::load(archive, "option_summary_norm", w)?;
        let mut layers = Vec::with_capacity(config.layers);
        for i in 0..config.layers {
            layers.push(DecoderLayerWeights::load(
                archive,
                &format!("layers.{i}"),
                w,
                heads,
                ff,
            )?);
        }

        let field_norm = LayerNormWeights::load(archive, "field_norm", w)?;
        let option_norm = LayerNormWeights::load(archive, "option_norm", w)?;
        let residual_0 = LinearWeights::load(archive, "residual_scorer.0", w, 4 * w, true)?;
        let residual_3 = LinearWeights::load(archive, "residual_scorer.3", 1, w, true)?;
        let prior_logit_scale = read_scalar_f32(archive, "prior_logit_scale")?;
        let joint_logit_scale = read_scalar_f32(archive, "joint_logit_scale")?;
        let residual_gate = read_scalar_f32(archive, "residual_gate")?;

        Ok(Self {
            config,
            hidden_norm,
            memory_projection,
            question_projection,
            option_question_projection,
            global_projection,
            option_context_projection,
            option_lexical_projection,
            type_embedding,
            evidence_layers,
            option_summary_norm,
            layers,
            field_norm,
            option_norm,
            residual_0,
            residual_3,
            prior_logit_scale,
            joint_logit_scale,
            residual_gate,
            output_embeddings,
        })
    }

    /// Evaluates `JointSchemaHead.forward` for one encoded record, returning one logit vector
    /// per question (ordered matching `questions[i].option_ids`).
    pub fn forward(
        &self,
        hidden_rows: &[Vec<f32>],
        input_ids: &[i32],
        questions: &[ClefEncodedQuestion],
    ) -> Result<Vec<Vec<f64>>, String> {
        let seq_len = hidden_rows.len();
        let h = self.config.hidden_size;
        let w = self.config.width;
        if seq_len == 0 || seq_len != input_ids.len() {
            return Err(format!(
                "clef forward: hidden_rows len ({seq_len}) != input_ids len ({}) or empty",
                input_ids.len()
            ));
        }
        if questions.is_empty() {
            return Err("clef forward: questions must be non-empty".into());
        }

        let mut sequence_hidden = vec![0.0f32; seq_len * h];
        for (i, row) in hidden_rows.iter().enumerate() {
            if row.len() != h {
                return Err(format!(
                    "clef forward: hidden row {i} len {} != hidden_size {h}",
                    row.len()
                ));
            }
            self.hidden_norm
                .apply_row(row, &mut sequence_hidden[i * h..(i + 1) * h]);
        }

        let memory = self.memory_projection.apply(&sequence_hidden, seq_len);
        let global_vector = &sequence_hidden[(seq_len - 1) * h..seq_len * h];
        let global_proj = self.global_projection.apply(global_vector, 1);

        let num_q = questions.len();
        let mut question_vectors = vec![0.0f32; num_q * h];
        let mut option_contexts: Vec<Vec<f32>> = Vec::with_capacity(num_q);
        let mut lexical_options: Vec<Vec<f32>> = Vec::with_capacity(num_q);
        let mut option_counts: Vec<usize> = Vec::with_capacity(num_q);

        for (qi, q) in questions.iter().enumerate() {
            if q.question_type >= 3 {
                return Err(format!(
                    "clef forward: invalid question_type {} for '{}'",
                    q.question_type, q.question_id
                ));
            }
            mean_span_into(
                &sequence_hidden,
                seq_len,
                h,
                q.question_span,
                &mut question_vectors[qi * h..(qi + 1) * h],
            )?;
            let n_opts = q.option_spans.len();
            if n_opts == 0 || n_opts != q.option_ids.len() {
                return Err(format!(
                    "clef forward: invalid option spans for '{}'",
                    q.question_id
                ));
            }
            let mut ctx_mat = vec![0.0f32; n_opts * h];
            let mut lex_mat = vec![0.0f32; n_opts * h];
            for (oi, &span) in q.option_spans.iter().enumerate() {
                mean_span_into(
                    &sequence_hidden,
                    seq_len,
                    h,
                    span,
                    &mut ctx_mat[oi * h..(oi + 1) * h],
                )?;
                let (s, e) = span;
                let lex = self.output_embeddings.mean_rows(&input_ids[s..e])?;
                lex_mat[oi * h..(oi + 1) * h].copy_from_slice(&lex);
            }
            option_contexts.push(ctx_mat);
            lexical_options.push(lex_mat);
            option_counts.push(n_opts);
        }

        let total_opts: usize = option_counts.iter().sum();
        let opt_q_proj_all = self
            .option_question_projection
            .apply(&question_vectors, num_q);
        let mut routed_options = Vec::with_capacity(total_opts * w);

        for qi in 0..num_q {
            let n_opts = option_counts[qi];
            let ctx_proj = self
                .option_context_projection
                .apply(&option_contexts[qi], n_opts);
            let lex_proj = self
                .option_lexical_projection
                .apply(&lexical_options[qi], n_opts);
            let q_proj = &opt_q_proj_all[qi * w..(qi + 1) * w];
            for oi in 0..n_opts {
                for d in 0..w {
                    routed_options.push(ctx_proj[oi * w + d] + lex_proj[oi * w + d] + q_proj[d]);
                }
            }
        }

        for layer in &self.evidence_layers {
            routed_options = layer.forward(&routed_options, total_opts, &memory, seq_len);
        }

        let base_fields = self.question_projection.apply(&question_vectors, num_q);
        let mut option_summaries = vec![0.0f32; num_q * w];
        let inv_sqrt_w = 1.0 / ((w as f64).sqrt());
        let mut opt_offset = 0usize;

        for qi in 0..num_q {
            let n_opts = option_counts[qi];
            let field_row = &base_fields[qi * w..(qi + 1) * w];
            let opts = &routed_options[opt_offset * w..(opt_offset + n_opts) * w];
            let mut weights = vec![0.0f64; n_opts];
            let mut max_s = f64::NEG_INFINITY;
            for oi in 0..n_opts {
                let s = (dot_f32(&opts[oi * w..(oi + 1) * w], field_row) as f64) * inv_sqrt_w;
                weights[oi] = s;
                if s > max_s {
                    max_s = s;
                }
            }
            let mut sum_exp = 0.0f64;
            for wt in &mut weights {
                let e = (*wt - max_s).exp();
                *wt = e;
                sum_exp += e;
            }
            let inv_sum = 1.0 / sum_exp;
            let summary = &mut option_summaries[qi * w..(qi + 1) * w];
            for (oi, &e) in weights.iter().enumerate() {
                let p = (e * inv_sum) as f32;
                let opt_row = &opts[oi * w..(oi + 1) * w];
                for d in 0..w {
                    summary[d] += p * opt_row[d];
                }
            }
            opt_offset += n_opts;
        }

        let norm_summaries = self
            .option_summary_norm
            .apply_matrix(&option_summaries, num_q);
        let mut fields = vec![0.0f32; num_q * w];
        for (qi, q) in questions.iter().enumerate() {
            let type_row = &self.type_embedding[q.question_type * w..(q.question_type + 1) * w];
            for d in 0..w {
                fields[qi * w + d] = base_fields[qi * w + d]
                    + norm_summaries[qi * w + d]
                    + global_proj[d]
                    + type_row[d];
            }
        }

        for layer in &self.layers {
            fields = layer.forward(&fields, num_q, &memory, seq_len);
        }
        let fields = self.field_norm.apply_matrix(&fields, num_q);

        let max_clamp = 100.0f64.ln() as f32;
        let prior_scale = (self.prior_logit_scale.min(max_clamp) as f64).exp();
        let joint_scale = (self.joint_logit_scale.min(max_clamp) as f64).exp();
        let gate = 1.0 / (1.0 + (-(self.residual_gate as f64)).exp());

        let mut results = Vec::with_capacity(num_q);
        opt_offset = 0;
        for qi in 0..num_q {
            let n_opts = option_counts[qi];
            let q_vec = &question_vectors[qi * h..(qi + 1) * h];
            let mut anchor = vec![0.0f32; h];
            for d in 0..h {
                anchor[d] = q_vec[d] + global_vector[d];
            }
            l2_normalize_in_place(&mut anchor, 1e-12);

            let field_row = &fields[qi * w..(qi + 1) * w];
            let opts_routed = &routed_options[opt_offset * w..(opt_offset + n_opts) * w];
            let options = self.option_norm.apply_matrix(opts_routed, n_opts);

            let mut features = vec![0.0f32; n_opts * 4 * w];
            for oi in 0..n_opts {
                let opt_row = &options[oi * w..(oi + 1) * w];
                let dst = &mut features[oi * 4 * w..(oi + 1) * 4 * w];
                dst[..w].copy_from_slice(field_row);
                dst[w..2 * w].copy_from_slice(opt_row);
                for d in 0..w {
                    dst[2 * w + d] = field_row[d] * opt_row[d];
                    dst[3 * w + d] = (field_row[d] - opt_row[d]).abs();
                }
            }
            let mut res_h = self.residual_0.apply(&features, n_opts);
            for v in &mut res_h {
                *v = exact_gelu(*v);
            }
            let residuals = self.residual_3.apply(&res_h, n_opts);

            let mut q_logits = Vec::with_capacity(n_opts);
            for (oi, &res_val) in residuals.iter().enumerate().take(n_opts) {
                let mut lex_row = lexical_options[qi][oi * h..(oi + 1) * h].to_vec();
                l2_normalize_in_place(&mut lex_row, 1e-12);
                let prior = prior_scale * (dot_f32(&lex_row, &anchor) as f64);

                let opt_row = &options[oi * w..(oi + 1) * w];
                let cosine = cosine_similarity_f32(field_row, opt_row, 1e-8) as f64;
                let joint = joint_scale * cosine + (res_val as f64);
                let logit = prior + gate * joint;
                if !logit.is_finite() {
                    return Err("clef forward: non-finite logit produced".into());
                }
                q_logits.push(logit);
            }
            results.push(q_logits);
            opt_offset += n_opts;
        }
        Ok(results)
    }
}

fn read_exact_1d(
    archive: &SafetensorsArchive,
    name: &str,
    expected_len: usize,
) -> Result<Vec<f32>, String> {
    let info = archive
        .tensor_info(name)
        .map_err(|e| format!("clef missing tensor '{name}': {e}"))?;
    if info.shape.as_slice() != [expected_len] {
        return Err(format!(
            "clef tensor '{name}' shape {:?} != [{expected_len}]",
            info.shape
        ));
    }
    let vals = archive
        .read_f32(name)
        .map_err(|e| format!("clef read_f32('{name}'): {e}"))?;
    if vals.len() != expected_len {
        return Err(format!(
            "clef tensor '{name}' length {} != expected {expected_len}",
            vals.len()
        ));
    }
    Ok(vals)
}

fn read_exact_2d(
    archive: &SafetensorsArchive,
    name: &str,
    rows: usize,
    cols: usize,
) -> Result<Vec<f32>, String> {
    let info = archive
        .tensor_info(name)
        .map_err(|e| format!("clef missing tensor '{name}': {e}"))?;
    if info.shape.as_slice() != [rows, cols] {
        return Err(format!(
            "clef tensor '{name}' shape {:?} != [{rows}, {cols}]",
            info.shape
        ));
    }
    let expected_len = rows
        .checked_mul(cols)
        .ok_or_else(|| format!("clef tensor '{name}' element count overflow"))?;
    let vals = archive
        .read_f32(name)
        .map_err(|e| format!("clef read_f32('{name}'): {e}"))?;
    if vals.len() != expected_len {
        return Err(format!(
            "clef tensor '{name}' length {} != expected {expected_len}",
            vals.len()
        ));
    }
    Ok(vals)
}

fn read_scalar_f32(archive: &SafetensorsArchive, name: &str) -> Result<f32, String> {
    let info = archive
        .tensor_info(name)
        .map_err(|e| format!("clef missing scalar '{name}': {e}"))?;
    if !info.shape.is_empty() && info.shape.as_slice() != [1] {
        return Err(format!(
            "clef scalar '{name}' unexpected shape {:?}",
            info.shape
        ));
    }
    let vals = archive
        .read_f32(name)
        .map_err(|e| format!("clef read_f32('{name}'): {e}"))?;
    if vals.len() != 1 {
        return Err(format!(
            "clef scalar '{name}' length {} != expected 1",
            vals.len()
        ));
    }
    Ok(vals[0])
}

fn mean_span_into(
    matrix: &[f32],
    rows: usize,
    cols: usize,
    span: (usize, usize),
    dst: &mut [f32],
) -> Result<(), String> {
    let (start, end) = span;
    if start >= end || end > rows {
        return Err(format!(
            "clef span ({start}, {end}) out of bounds for sequence len {rows}"
        ));
    }
    let inv = 1.0 / ((end - start) as f64);
    let mut acc = vec![0.0f64; cols];
    for r in start..end {
        let row = &matrix[r * cols..(r + 1) * cols];
        for (a, &v) in acc.iter_mut().zip(row) {
            *a += v as f64;
        }
    }
    for (d, a) in dst.iter_mut().zip(acc) {
        *d = (a * inv) as f32;
    }
    Ok(())
}

fn l2_normalize_in_place(v: &mut [f32], eps: f32) {
    let norm = (v
        .iter()
        .map(|&x| (x as f64) * (x as f64))
        .sum::<f64>()
        .sqrt() as f32)
        .max(eps);
    let inv = 1.0 / norm;
    for x in v {
        *x *= inv;
    }
}

fn cosine_similarity_f32(a: &[f32], b: &[f32], eps: f32) -> f32 {
    debug_assert_eq!(a.len(), b.len());
    let mut dot = 0.0f64;
    let mut na = 0.0f64;
    let mut nb = 0.0f64;
    for (&x, &y) in a.iter().zip(b) {
        let xf = x as f64;
        let yf = y as f64;
        dot += xf * yf;
        na += xf * xf;
        nb += yf * yf;
    }
    let denom = ((na.sqrt() as f32) * (nb.sqrt() as f32)).max(eps) as f64;
    (dot / denom) as f32
}

#[inline]
fn dot_f32(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len());
    #[cfg(target_arch = "x86_64")]
    {
        if std::arch::is_x86_feature_detected!("avx2") && std::arch::is_x86_feature_detected!("fma")
        {
            // SAFETY: AVX2 and FMA availability was checked dynamically above, and `a`/`b` have equal length.
            return unsafe { dot_f32_avx2(a, b) };
        }
    }
    dot_f32_scalar(a, b)
}

#[inline]
fn dot_f32_scalar(a: &[f32], b: &[f32]) -> f32 {
    let n = a.len();
    let chunks = n / 8;
    let mut s0 = 0.0f32;
    let mut s1 = 0.0f32;
    let mut s2 = 0.0f32;
    let mut s3 = 0.0f32;
    let mut s4 = 0.0f32;
    let mut s5 = 0.0f32;
    let mut s6 = 0.0f32;
    let mut s7 = 0.0f32;
    for c in 0..chunks {
        let i = c * 8;
        s0 += a[i] * b[i];
        s1 += a[i + 1] * b[i + 1];
        s2 += a[i + 2] * b[i + 2];
        s3 += a[i + 3] * b[i + 3];
        s4 += a[i + 4] * b[i + 4];
        s5 += a[i + 5] * b[i + 5];
        s6 += a[i + 6] * b[i + 6];
        s7 += a[i + 7] * b[i + 7];
    }
    let mut sum = ((s0 + s1) + (s2 + s3)) + ((s4 + s5) + (s6 + s7));
    for i in (chunks * 8)..n {
        sum += a[i] * b[i];
    }
    sum
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn dot_f32_avx2(a: &[f32], b: &[f32]) -> f32 {
    use std::arch::x86_64::*;
    let n = a.len();
    let ap = a.as_ptr();
    let bp = b.as_ptr();
    let mut acc0 = _mm256_setzero_ps();
    let mut acc1 = _mm256_setzero_ps();
    let mut i = 0usize;
    while i + 16 <= n {
        // SAFETY: `i + 16 <= n` bounds both `a` and `b` reads.
        unsafe {
            acc0 = _mm256_fmadd_ps(_mm256_loadu_ps(ap.add(i)), _mm256_loadu_ps(bp.add(i)), acc0);
            acc1 = _mm256_fmadd_ps(
                _mm256_loadu_ps(ap.add(i + 8)),
                _mm256_loadu_ps(bp.add(i + 8)),
                acc1,
            );
        }
        i += 16;
    }
    // SAFETY: AVX2+FMA enabled on this function.
    let mut out = unsafe { hsum256_ps(_mm256_add_ps(acc0, acc1)) };
    while i < n {
        out += a[i] * b[i];
        i += 1;
    }
    out
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn hsum256_ps(v: std::arch::x86_64::__m256) -> f32 {
    use std::arch::x86_64::*;
    let hi = _mm256_extractf128_ps(v, 1);
    let lo = _mm256_castps256_ps128(v);
    let s128 = _mm_add_ps(lo, hi);
    let shuf = _mm_movehdup_ps(s128);
    let sums = _mm_add_ps(s128, shuf);
    let shuf2 = _mm_movehl_ps(shuf, sums);
    _mm_cvtss_f32(_mm_add_ss(sums, shuf2))
}

/// Computes `C[m, n] = A[m, k] @ B[n, k]^T + bias[n]` (`A` is `[m, k]`, `B` is `[n, k]`).
fn matmul_nt(a: &[f32], m: usize, k: usize, b: &[f32], n: usize, bias: Option<&[f32]>) -> Vec<f32> {
    assert_eq!(a.len(), m * k);
    assert_eq!(b.len(), n * k);
    if let Some(bs) = bias {
        assert_eq!(bs.len(), n);
    }
    #[cfg(target_arch = "x86_64")]
    {
        if std::arch::is_x86_feature_detected!("avx2") && std::arch::is_x86_feature_detected!("fma")
        {
            // SAFETY: AVX2 and FMA availability was checked dynamically above, and slice bounds are asserted.
            return unsafe { matmul_nt_avx2(a, m, k, b, n, bias) };
        }
    }
    matmul_nt_scalar(a, m, k, b, n, bias)
}

fn matmul_nt_scalar(
    a: &[f32],
    m: usize,
    k: usize,
    b: &[f32],
    n: usize,
    bias: Option<&[f32]>,
) -> Vec<f32> {
    let mut c = vec![0.0f32; m * n];
    for row in 0..m {
        let a_row = &a[row * k..(row + 1) * k];
        for col in 0..n {
            let b_row = &b[col * k..(col + 1) * k];
            let bi = bias.map_or(0.0, |bs| bs[col]);
            c[row * n + col] = dot_f32_scalar(a_row, b_row) + bi;
        }
    }
    c
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn matmul_nt_avx2(
    a: &[f32],
    m: usize,
    k: usize,
    b: &[f32],
    n: usize,
    bias: Option<&[f32]>,
) -> Vec<f32> {
    use std::arch::x86_64::*;
    let mut c = vec![0.0f32; m * n];
    if let Some(bs) = bias {
        for row in 0..m {
            c[row * n..(row + 1) * n].copy_from_slice(bs);
        }
    }
    const KC: usize = 512;
    let m2 = m / 2;
    let n4 = n / 4;
    let k8 = k / 8 * 8;

    let mut kb = 0usize;
    while kb < k8 {
        let kend = (kb + KC).min(k8);
        for ni in 0..n4 {
            let c0 = ni * 4;
            // SAFETY: `c0 + 3 < n`, so `(c0 + 3) * k <= b.len()`.
            let (bp0, bp1, bp2, bp3) = unsafe {
                (
                    b.as_ptr().add(c0 * k),
                    b.as_ptr().add((c0 + 1) * k),
                    b.as_ptr().add((c0 + 2) * k),
                    b.as_ptr().add((c0 + 3) * k),
                )
            };

            for mi in 0..m2 {
                let r0 = mi * 2;
                // SAFETY: `r0 + 1 < m`, so `(r0 + 1) * k <= a.len()`.
                let (ap0, ap1) = unsafe { (a.as_ptr().add(r0 * k), a.as_ptr().add((r0 + 1) * k)) };

                let mut acc00 = _mm256_setzero_ps();
                let mut acc01 = _mm256_setzero_ps();
                let mut acc02 = _mm256_setzero_ps();
                let mut acc03 = _mm256_setzero_ps();
                let mut acc10 = _mm256_setzero_ps();
                let mut acc11 = _mm256_setzero_ps();
                let mut acc12 = _mm256_setzero_ps();
                let mut acc13 = _mm256_setzero_ps();

                let mut p = kb;
                while p < kend {
                    // SAFETY: `r0 + 1 < m`, `c0 + 3 < n`, and `p + 8 <= kend <= k8 <= k`.
                    unsafe {
                        let va0 = _mm256_loadu_ps(ap0.add(p));
                        let va1 = _mm256_loadu_ps(ap1.add(p));
                        let vb0 = _mm256_loadu_ps(bp0.add(p));
                        let vb1 = _mm256_loadu_ps(bp1.add(p));
                        acc00 = _mm256_fmadd_ps(va0, vb0, acc00);
                        acc10 = _mm256_fmadd_ps(va1, vb0, acc10);
                        acc01 = _mm256_fmadd_ps(va0, vb1, acc01);
                        acc11 = _mm256_fmadd_ps(va1, vb1, acc11);
                        let vb2 = _mm256_loadu_ps(bp2.add(p));
                        let vb3 = _mm256_loadu_ps(bp3.add(p));
                        acc02 = _mm256_fmadd_ps(va0, vb2, acc02);
                        acc12 = _mm256_fmadd_ps(va1, vb2, acc12);
                        acc03 = _mm256_fmadd_ps(va0, vb3, acc03);
                        acc13 = _mm256_fmadd_ps(va1, vb3, acc13);
                    }
                    p += 8;
                }
                // SAFETY: `hsum256_ps` requires AVX2+FMA which is enabled on this function.
                unsafe {
                    c[r0 * n + c0] += hsum256_ps(acc00);
                    c[r0 * n + c0 + 1] += hsum256_ps(acc01);
                    c[r0 * n + c0 + 2] += hsum256_ps(acc02);
                    c[r0 * n + c0 + 3] += hsum256_ps(acc03);
                    c[(r0 + 1) * n + c0] += hsum256_ps(acc10);
                    c[(r0 + 1) * n + c0 + 1] += hsum256_ps(acc11);
                    c[(r0 + 1) * n + c0 + 2] += hsum256_ps(acc12);
                    c[(r0 + 1) * n + c0 + 3] += hsum256_ps(acc13);
                }
            }
        }
        kb = kend;
    }

    for row in 0..(m2 * 2) {
        let a_row = &a[row * k..row * k + k8];
        for col in (n4 * 4)..n {
            let b_row = &b[col * k..col * k + k8];
            // SAFETY: AVX2+FMA enabled on this function.
            c[row * n + col] += unsafe { dot_f32_avx2(a_row, b_row) };
        }
    }
    for row in (m2 * 2)..m {
        let a_row = &a[row * k..row * k + k8];
        for col in 0..n {
            let b_row = &b[col * k..col * k + k8];
            // SAFETY: AVX2+FMA enabled on this function.
            c[row * n + col] += unsafe { dot_f32_avx2(a_row, b_row) };
        }
    }
    if k8 < k {
        for row in 0..m {
            let a_tail = &a[row * k + k8..(row + 1) * k];
            for col in 0..n {
                let b_tail = &b[col * k + k8..(col + 1) * k];
                c[row * n + col] += dot_f32_scalar(a_tail, b_tail);
            }
        }
    }
    c
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ClefEncodedQuestion {
    pub question_id: String,
    pub question_type: usize,
    pub question_span: (usize, usize),
    pub option_spans: Vec<(usize, usize)>,
    pub option_ids: Vec<String>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ClefEncodedRecord {
    pub input_ids: Vec<i32>,
    pub questions: Vec<ClefEncodedQuestion>,
}

#[derive(Clone, Copy, Debug)]
pub struct ClefMediaTokens {
    pub vision_start_id: i32,
    pub image_pad_id: i32,
    pub vision_end_id: i32,
}

/// Clef `render(value)` (`joint_schema_model.py:37-45`):
/// returns strings verbatim and serializes non-strings as compact UTF-8 JSON
/// (`separators=(",", ":")`, `sort_keys=True`, `ensure_ascii=False`).
pub fn clef_render_value(v: &Value) -> Result<String, DecisionError> {
    if let Some(s) = v.as_str() {
        return Ok(s.to_owned());
    }
    let mut out = String::new();
    clef_dump_compact_sorted(v, &mut out)?;
    Ok(out)
}

fn clef_dump_compact_sorted(v: &Value, out: &mut String) -> Result<(), DecisionError> {
    match v {
        Value::Array(arr) => {
            out.push('[');
            for (i, item) in arr.iter().enumerate() {
                if i > 0 {
                    out.push(',');
                }
                clef_dump_compact_sorted(item, out)?;
            }
            out.push(']');
        }
        Value::Object(obj) => {
            let mut entries: Vec<(&String, &Value)> = obj.iter().collect();
            entries.sort_by(|a, b| a.0.cmp(b.0));
            out.push('{');
            for (i, (k, val)) in entries.into_iter().enumerate() {
                if i > 0 {
                    out.push(',');
                }
                out.push_str(
                    &serde_json::to_string(k)
                        .map_err(|e| DecisionError::bad_request(e.to_string()))?,
                );
                out.push(':');
                clef_dump_compact_sorted(val, out)?;
            }
            out.push('}');
        }
        Value::Number(n) => {
            out.push_str(&format_python_json_number(n));
        }
        _ => {
            out.push_str(
                &serde_json::to_string(v).map_err(|e| DecisionError::bad_request(e.to_string()))?,
            );
        }
    }
    Ok(())
}

/// Formats a `serde_json::Number` identically to Python `json.dumps(...)`:
/// integers render verbatim; floats use shortest round-trip decimal (`ryu`) in fixed-point
/// when `f == 0.0` or `-4 <= exp < 16`, and `Xe+YY` / `Xe-YY` (explicit sign, `>= 2` exponent
/// digits) outside `1e-4..1e16`.
pub(crate) fn format_python_json_number(n: &serde_json::Number) -> String {
    if n.is_i64() || n.is_u64() {
        return n.to_string();
    }
    let raw = n.to_string();
    let (neg, unsigned) = match raw.strip_prefix('-') {
        Some(rest) => (true, rest),
        None => (false, raw.as_str()),
    };
    let sign_prefix = if neg { "-" } else { "" };
    let (mant, exp_part) = match unsigned.split_once(['e', 'E']) {
        Some((m, e)) => (m, e.parse::<i32>().unwrap_or(0)),
        None => (unsigned, 0),
    };
    let (int_str, frac_str) = match mant.split_once('.') {
        Some((i, f)) => (i, f),
        None => (mant, ""),
    };
    let mut digits = String::with_capacity(int_str.len() + frac_str.len());
    digits.push_str(int_str);
    digits.push_str(frac_str);
    let trimmed_leading = digits.trim_start_matches('0');
    if trimmed_leading.is_empty() {
        return format!("{sign_prefix}0.0");
    }
    let leading_zeros = digits.len() - trimmed_leading.len();
    let decpt = (int_str.len() as i32) - (leading_zeros as i32) + exp_part;
    let sig = trimmed_leading.trim_end_matches('0');
    let exp = decpt - 1;
    if (-4..16).contains(&exp) {
        if decpt <= 0 {
            format!("{sign_prefix}0.{}{sig}", "0".repeat((-decpt) as usize))
        } else if (decpt as usize) >= sig.len() {
            format!(
                "{sign_prefix}{sig}{}.0",
                "0".repeat((decpt as usize) - sig.len())
            )
        } else {
            let split = decpt as usize;
            format!("{sign_prefix}{}.{}", &sig[..split], &sig[split..])
        }
    } else {
        let mantissa = if sig.len() == 1 {
            sig.to_owned()
        } else {
            format!("{}.{}", &sig[..1], &sig[1..])
        };
        let esign = if exp >= 0 { '+' } else { '-' };
        format!("{sign_prefix}{mantissa}e{esign}{:02}", exp.unsigned_abs())
    }
}

fn collect_clef_added_control_tokens(tokenizer: &Tokenizer) -> Vec<String> {
    let mut toks: Vec<String> = tokenizer
        .get_added_tokens_decoder()
        .into_values()
        .map(|t| t.content)
        .filter(|s| !s.is_empty() && !s.contains("<|") && !s.contains("|>"))
        .collect();
    toks.sort_by(|a, b| b.len().cmp(&a.len()).then_with(|| a.cmp(b)));
    toks.dedup();
    toks
}

fn sanitize_clef_untrusted(text: &str, added_tokens: &[String]) -> String {
    let mut clean = kev_sanitize_special_tokens(text);
    for tok in added_tokens {
        if clean.contains(tok.as_str()) {
            let neutralized = if tok.contains('<') || tok.contains('>') {
                tok.replace('<', "<\u{00a6}").replace('>', "\u{00a6}>")
            } else {
                let mut ch = tok.chars();
                match ch.next() {
                    Some(first) => format!("{first}\u{200b}{}", ch.as_str()),
                    None => String::new(),
                }
            };
            if neutralized != *tok {
                clean = clean.replace(tok.as_str(), &neutralized);
            }
        }
    }
    clean
}

fn slice_utf8_prefix(s: &str, max_bytes: usize) -> &str {
    if s.len() <= max_bytes {
        return s;
    }
    let mut end = max_bytes;
    while end > 0 && !s.is_char_boundary(end) {
        end -= 1;
    }
    &s[..end]
}

fn encode_plain(tokenizer: &Tokenizer, text: &str) -> Result<Vec<i32>, DecisionError> {
    tokenizer
        .encode(text, false)
        .map(|enc| enc.get_ids().iter().map(|&id| id as i32).collect())
        .map_err(|e| DecisionError::internal(format!("clef tokenize: {e}")))
}

fn encode_untrusted(
    tokenizer: &Tokenizer,
    text: &str,
    added_tokens: &[String],
) -> Result<Vec<i32>, DecisionError> {
    let clean = sanitize_clef_untrusted(text, added_tokens);
    encode_plain(tokenizer, &clean)
}

/// Port of `joint_schema_model.py` `encode_record`, preserving exact token IDs,
/// `question_span`, and `option_spans` across text, JSON, and multimodal image records.
pub fn clef_encode_record(
    tokenizer: &Tokenizer,
    body: &Value,
    max_length: usize,
    media: Option<(ClefMediaTokens, &[usize])>,
) -> Result<ClefEncodedRecord, DecisionError> {
    let obj = body
        .as_object()
        .ok_or_else(|| DecisionError::bad_request("request body must be an object"))?;
    let state_val = obj
        .get("state")
        .ok_or_else(|| DecisionError::bad_request("state is required"))?;
    let q_obj = obj
        .get("questions")
        .and_then(Value::as_object)
        .filter(|m| !m.is_empty())
        .ok_or_else(|| DecisionError::bad_request("questions must be a non-empty object"))?;

    let added_control_tokens = collect_clef_added_control_tokens(tokenizer);
    let mut schema_ids = encode_plain(tokenizer, "\n\nSCHEMA FIELDS:\n")?;
    let mut questions = Vec::with_capacity(q_obj.len());

    for (q_idx, (question_id, question_val)) in q_obj.iter().enumerate() {
        let qm = question_val
            .as_object()
            .ok_or_else(|| DecisionError::bad_request("question must be an object"))?;
        let q_type_str = qm
            .get("type")
            .and_then(Value::as_str)
            .ok_or_else(|| DecisionError::bad_request("question missing string type"))?;
        let q_type_id = match q_type_str {
            "noul" => 0usize,
            "choice" => 1usize,
            "score" => 2usize,
            other => {
                return Err(DecisionError::bad_request(format!(
                    "question '{question_id}' has unknown type: {other}"
                )));
            }
        };

        let safe_qid = sanitize_clef_untrusted(question_id, &added_control_tokens);
        let field_header = format!(
            "\nFIELD {}\nID: {safe_qid}\nTYPE: {q_type_str}\nINSTRUCTION: ",
            q_idx + 1
        );
        schema_ids.extend(encode_plain(tokenizer, &field_header)?);

        let question_start = schema_ids.len();
        let instr_rendered = match qm.get("instructions") {
            Some(v) if !v.is_null() && v.as_str() != Some("") => clef_render_value(v)?,
            _ => question_id.clone(),
        };
        let instr_ids = encode_untrusted(tokenizer, &instr_rendered, &added_control_tokens)?;
        if instr_ids.is_empty() {
            return Err(DecisionError::bad_request(format!(
                "question '{question_id}': instructions produced zero tokens"
            )));
        }
        schema_ids.extend(instr_ids);
        let question_end = schema_ids.len();
        schema_ids.extend(encode_plain(tokenizer, "\nALLOWED OPTIONS:\n")?);

        let options: Vec<(String, Option<Value>)> = match q_type_str {
            "noul" => {
                let mut t_desc = Value::String(CLEF_NOUL_TRUE_DEFAULT.to_owned());
                let mut f_desc = Value::String(CLEF_NOUL_FALSE_DEFAULT.to_owned());
                if let Some(cm) = qm.get("criteria").and_then(Value::as_object) {
                    if let Some(tv) = cm.get("true").filter(|v| !v.is_null()) {
                        t_desc = tv.clone();
                    }
                    if let Some(fv) = cm.get("false").filter(|v| !v.is_null()) {
                        f_desc = fv.clone();
                    }
                }
                vec![
                    ("true".to_owned(), Some(t_desc)),
                    ("false".to_owned(), Some(f_desc)),
                ]
            }
            "choice" => {
                let cm = qm
                    .get("criteria")
                    .and_then(Value::as_object)
                    .ok_or_else(|| {
                        DecisionError::bad_request(format!(
                            "question '{question_id}': choice requires a criteria object"
                        ))
                    })?;
                let mut pairs: Vec<(String, Option<Value>)> = cm
                    .iter()
                    .map(|(k, v)| (k.clone(), (!v.is_null()).then(|| v.clone())))
                    .collect();
                pairs.sort_by(|a, b| a.0.cmp(&b.0));
                pairs
            }
            "score" => {
                let ca = qm
                    .get("criteria")
                    .and_then(Value::as_array)
                    .ok_or_else(|| {
                        DecisionError::bad_request(format!(
                            "question '{question_id}': score requires a criteria array"
                        ))
                    })?;
                ca.iter()
                    .enumerate()
                    .map(|(idx, v)| (idx.to_string(), (!v.is_null()).then(|| v.clone())))
                    .collect()
            }
            _ => unreachable!(),
        };

        let mut option_spans = Vec::with_capacity(options.len());
        let mut option_ids = Vec::with_capacity(options.len());
        for (opt_idx, (opt_id, desc_opt)) in options.into_iter().enumerate() {
            schema_ids.extend(encode_plain(
                tokenizer,
                &format!("OPTION {}: ", opt_idx + 1),
            )?);
            let option_start = schema_ids.len();
            let mut sem = Map::new();
            sem.insert("option_id".to_owned(), Value::String(opt_id.clone()));
            if let Some(d) = desc_opt {
                sem.insert("description".to_owned(), d);
            }
            let sem_str = clef_render_value(&Value::Object(sem))?;
            let opt_toks = encode_untrusted(tokenizer, &sem_str, &added_control_tokens)?;
            if opt_toks.is_empty() {
                return Err(DecisionError::bad_request(format!(
                    "question '{question_id}': option '{opt_id}' produced zero tokens"
                )));
            }
            schema_ids.extend(opt_toks);
            let option_end = schema_ids.len();
            option_spans.push((option_start, option_end));
            option_ids.push(opt_id);
            schema_ids.extend(encode_plain(tokenizer, "\n")?);
        }
        schema_ids.extend(encode_plain(tokenizer, "END FIELD\n")?);
        questions.push(ClefEncodedQuestion {
            question_id: question_id.clone(),
            question_type: q_type_id,
            question_span: (question_start, question_end),
            option_spans,
            option_ids,
        });
    }

    let mut prefix_ids = encode_plain(
        tokenizer,
        &format!("<|im_start|>system\n{CLEF_SYSTEM_PROMPT}<|im_end|>\n<|im_start|>user\nSTATE:\n"),
    )?;
    if let Some((mtoks, counts)) = media
        && !counts.is_empty()
    {
        for &n_merged in counts {
            prefix_ids.push(mtoks.vision_start_id);
            prefix_ids.resize(prefix_ids.len() + n_merged, mtoks.image_pad_id);
            prefix_ids.push(mtoks.vision_end_id);
        }
        prefix_ids.extend(encode_plain(tokenizer, "\n")?);
    }
    let suffix_ids = encode_plain(
        tokenizer,
        "\n<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\nJOINT SCHEMA DECISIONS:",
    )?;

    let fixed_length = prefix_ids.len() + schema_ids.len() + suffix_ids.len();
    if fixed_length >= max_length {
        return Err(DecisionError::bad_request(format!(
            "schema requires {fixed_length} tokens before state; maximum is {max_length}"
        )));
    }
    let state_budget = max_length - fixed_length;
    let state_rendered = clef_render_value(state_val)?;
    // Bound UTF-8 input length before BPE tokenization (Qwen3.5 max single-token UTF-8 span is
    // 256 bytes, so `state_budget * 256 + 256` is lossless for the kept `state_budget` prefix).
    let max_state_bytes = state_budget.saturating_mul(256).saturating_add(256);
    let state_slice = slice_utf8_prefix(&state_rendered, max_state_bytes);
    let mut state_ids = encode_untrusted(tokenizer, state_slice, &added_control_tokens)?;
    if state_ids.is_empty() {
        return Err(DecisionError::bad_request("state produced zero tokens"));
    }
    state_ids.truncate(state_budget);

    let schema_offset = prefix_ids.len() + state_ids.len();
    for q in &mut questions {
        q.question_span.0 += schema_offset;
        q.question_span.1 += schema_offset;
        for sp in &mut q.option_spans {
            sp.0 += schema_offset;
            sp.1 += schema_offset;
        }
    }

    let mut input_ids = Vec::with_capacity(schema_offset + schema_ids.len() + suffix_ids.len());
    input_ids.extend(prefix_ids);
    input_ids.extend(state_ids);
    input_ids.extend(schema_ids);
    input_ids.extend(suffix_ids);

    Ok(ClefEncodedRecord {
        input_ids,
        questions,
    })
}

/// Maps logits produced in `question.option_ids` order (which sorts `choice` keys alphabetically)
/// back onto `field.keys` order (caller's `criteria` key order).
pub fn reorder_clef_question_logits(
    question: &ClefEncodedQuestion,
    field: &DecisionField,
    raw_logits: &[f64],
) -> Result<Vec<f64>, DecisionError> {
    if raw_logits.len() != question.option_ids.len() || raw_logits.len() != field.keys.len() {
        return Err(DecisionError::internal(format!(
            "clef: option count mismatch for '{}'",
            field.name
        )));
    }
    if question.option_ids == field.keys {
        return Ok(raw_logits.to_vec());
    }
    let mut out = Vec::with_capacity(field.keys.len());
    for key in &field.keys {
        let idx = question
            .option_ids
            .iter()
            .position(|oid| oid == key)
            .ok_or_else(|| {
                DecisionError::internal(format!(
                    "clef: option '{key}' missing from encoded question '{}'",
                    field.name
                ))
            })?;
        out.push(raw_logits[idx]);
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn make_safetensors(path: &Path, tensors: &[(&str, &str, &[usize], &[u8])]) {
        let mut header = Map::new();
        let mut offset = 0usize;
        for (name, dtype, shape, bytes) in tensors {
            let mut m = Map::new();
            m.insert("dtype".into(), json!(dtype));
            m.insert("shape".into(), json!(shape));
            m.insert("data_offsets".into(), json!([offset, offset + bytes.len()]));
            header.insert((*name).to_string(), Value::Object(m));
            offset += bytes.len();
        }
        let mut hbytes = serde_json::to_vec(&Value::Object(header)).unwrap();
        while !hbytes.len().is_multiple_of(8) {
            hbytes.push(b' ');
        }
        let mut out = Vec::with_capacity(8 + hbytes.len() + offset);
        out.extend_from_slice(&(hbytes.len() as u64).to_le_bytes());
        out.extend_from_slice(&hbytes);
        for (_, _, _, bytes) in tensors {
            out.extend_from_slice(bytes);
        }
        fs::write(path, out).unwrap();
    }

    fn f32_bytes(vals: &[f32]) -> Vec<u8> {
        vals.iter().flat_map(|v| v.to_le_bytes()).collect()
    }

    #[test]
    fn clef_render_sorts_keys_compactly_and_formats_python_floats() {
        let v = json!({
            "zeta": [1, 2, {"b": true, "a": null}],
            "alpha": "raw <text>",
            "empty_arr": [],
            "empty_obj": {}
        });
        assert_eq!(
            clef_render_value(&v).unwrap(),
            r#"{"alpha":"raw <text>","empty_arr":[],"empty_obj":{},"zeta":[1,2,{"a":null,"b":true}]}"#
        );
        assert_eq!(clef_render_value(&json!("plain")).unwrap(), "plain");

        for (val, expected) in [
            (0.0f64, "0.0"),
            (-0.0f64, "-0.0"),
            (1.0f64, "1.0"),
            (12.34f64, "12.34"),
            (1e-4f64, "0.0001"),
            (1e-5f64, "1e-05"),
            (1.5e-7f64, "1.5e-07"),
            (1e15f64, "1000000000000000.0"),
            (1e16f64, "1e+16"),
            (1e20f64, "1e+20"),
            (1.5e20f64, "1.5e+20"),
            (-3.25e-12f64, "-3.25e-12"),
        ] {
            let n = serde_json::Number::from_f64(val).unwrap();
            assert_eq!(
                clef_render_value(&Value::Number(n)).unwrap(),
                expected,
                "float {val:e}"
            );
        }
    }

    #[test]
    fn clef_matmul_nt_and_dot_avx2_match_scalar_across_tail_and_blocking_shapes() {
        for len in 0..=35 {
            let a = splitmix_vec("dot_a", len, 1.0);
            let b = splitmix_vec("dot_b", len, 1.0);
            let got = dot_f32(&a, &b);
            let exp = dot_f32_scalar(&a, &b);
            assert!(
                (got - exp).abs() <= 1e-5,
                "dot_f32 mismatch at len={len}: got {got}, exp {exp}"
            );
        }

        for m in [1usize, 2, 3, 5] {
            for n in [1usize, 3, 4, 5, 7] {
                for k in [1usize, 7, 8, 9, 17, 512, 520, 1029] {
                    let a = splitmix_vec("mm_a", m * k, 0.5);
                    let b = splitmix_vec("mm_b", n * k, 0.5);
                    let bias = splitmix_vec("mm_bias", n, 0.25);
                    for opt_bias in [None, Some(bias.as_slice())] {
                        let c_disp = matmul_nt(&a, m, k, &b, n, opt_bias);
                        let c_ref = matmul_nt_scalar(&a, m, k, &b, n, opt_bias);
                        for (idx, (&g, &e)) in c_disp.iter().zip(&c_ref).enumerate() {
                            assert!(
                                (g - e).abs() <= 1e-4,
                                "matmul_nt mismatch at (m={m}, n={n}, k={k}, bias={}, idx={idx}): got {g}, exp {e}",
                                opt_bias.is_some()
                            );
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn clef_joint_head_forward_and_perturbation_gates() {
        let cfg = ClefHeadConfig {
            hidden_size: 8,
            width: 4,
            routing_layers: 1,
            layers: 1,
            heads: 2,
            feedforward: 8,
        };
        let tmp = std::env::temp_dir().join(format!("w34_clef_unit_{}", std::process::id()));
        let _ = fs::remove_dir_all(&tmp);
        fs::create_dir_all(&tmp).unwrap();
        let head_path = tmp.join("joint_head.safetensors");

        let mut named: Vec<(String, Vec<usize>, Vec<u8>)> = Vec::new();
        let mut add = |name: &str, shape: &[usize], scale: f32| {
            let n: usize = shape.iter().product::<usize>().max(1);
            let bytes = f32_bytes(&splitmix_vec(name, n, scale as f64));
            named.push((name.to_owned(), shape.to_vec(), bytes));
        };

        add("hidden_norm.weight", &[8], 1.0);
        add("hidden_norm.bias", &[8], 0.1);
        add("memory_projection.weight", &[4, 8], 0.3);
        add("question_projection.weight", &[4, 8], 0.3);
        add("option_question_projection.weight", &[4, 8], 0.3);
        add("global_projection.weight", &[4, 8], 0.3);
        add("option_context_projection.weight", &[4, 8], 0.3);
        add("option_lexical_projection.weight", &[4, 8], 0.3);
        add("type_embedding.weight", &[3, 4], 0.3);

        add("evidence_layers.0.query_norm.weight", &[4], 1.0);
        add("evidence_layers.0.query_norm.bias", &[4], 0.05);
        add("evidence_layers.0.memory_norm.weight", &[4], 1.0);
        add("evidence_layers.0.memory_norm.bias", &[4], 0.05);
        add("evidence_layers.0.attention.in_proj_weight", &[12, 4], 0.3);
        add("evidence_layers.0.attention.in_proj_bias", &[12], 0.05);
        add("evidence_layers.0.attention.out_proj.weight", &[4, 4], 0.3);
        add("evidence_layers.0.attention.out_proj.bias", &[4], 0.05);
        add("evidence_layers.0.feedforward_norm.weight", &[4], 1.0);
        add("evidence_layers.0.feedforward_norm.bias", &[4], 0.05);
        add("evidence_layers.0.feedforward.0.weight", &[8, 4], 0.3);
        add("evidence_layers.0.feedforward.0.bias", &[8], 0.05);
        add("evidence_layers.0.feedforward.3.weight", &[4, 8], 0.3);
        add("evidence_layers.0.feedforward.3.bias", &[4], 0.05);

        add("option_summary_norm.weight", &[4], 1.0);
        add("option_summary_norm.bias", &[4], 0.05);

        add("layers.0.norm1.weight", &[4], 1.0);
        add("layers.0.norm1.bias", &[4], 0.05);
        add("layers.0.self_attn.in_proj_weight", &[12, 4], 0.3);
        add("layers.0.self_attn.in_proj_bias", &[12], 0.05);
        add("layers.0.self_attn.out_proj.weight", &[4, 4], 0.3);
        add("layers.0.self_attn.out_proj.bias", &[4], 0.05);
        add("layers.0.norm2.weight", &[4], 1.0);
        add("layers.0.norm2.bias", &[4], 0.05);
        add("layers.0.multihead_attn.in_proj_weight", &[12, 4], 0.3);
        add("layers.0.multihead_attn.in_proj_bias", &[12], 0.05);
        add("layers.0.multihead_attn.out_proj.weight", &[4, 4], 0.3);
        add("layers.0.multihead_attn.out_proj.bias", &[4], 0.05);
        add("layers.0.norm3.weight", &[4], 1.0);
        add("layers.0.norm3.bias", &[4], 0.05);
        add("layers.0.linear1.weight", &[8, 4], 0.3);
        add("layers.0.linear1.bias", &[8], 0.05);
        add("layers.0.linear2.weight", &[4, 8], 0.3);
        add("layers.0.linear2.bias", &[4], 0.05);

        add("field_norm.weight", &[4], 1.0);
        add("field_norm.bias", &[4], 0.05);
        add("option_norm.weight", &[4], 1.0);
        add("option_norm.bias", &[4], 0.05);
        add("residual_scorer.0.weight", &[4, 16], 0.3);
        add("residual_scorer.0.bias", &[4], 0.05);
        add("residual_scorer.3.weight", &[1, 4], 0.3);
        add("residual_scorer.3.bias", &[1], 0.05);
        add("prior_logit_scale", &[], 0.5);
        add("joint_logit_scale", &[], 0.5);
        add("residual_gate", &[], 0.5);

        let refs: Vec<(&str, &str, &[usize], &[u8])> = named
            .iter()
            .map(|(n, s, b)| (n.as_str(), "F32", s.as_slice(), b.as_slice()))
            .collect();
        make_safetensors(&head_path, &refs);

        let arc = SafetensorsArchive::open(&head_path).unwrap();
        let out_emb =
            ClefOutputEmbeddings::from_in_memory(splitmix_vec("unit_emb", 16 * 8, 0.4), 16, 8)
                .unwrap();
        let head = ClefJointHead::from_archive(&arc, cfg.clone(), out_emb).unwrap();

        let hidden_rows: Vec<Vec<f32>> = (0..10)
            .map(|r| splitmix_vec(&format!("unit_h_{r}"), 8, 0.5))
            .collect();
        let input_ids = vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10];
        let questions = vec![
            ClefEncodedQuestion {
                question_id: "q1".into(),
                question_type: 0,
                question_span: (1, 3),
                option_spans: vec![(3, 5), (5, 7)],
                option_ids: vec!["true".into(), "false".into()],
            },
            ClefEncodedQuestion {
                question_id: "q2".into(),
                question_type: 1,
                question_span: (7, 8),
                option_spans: vec![(8, 9), (9, 10)],
                option_ids: vec!["alpha".into(), "zeta".into()],
            },
        ];

        let out = head.forward(&hidden_rows, &input_ids, &questions).unwrap();
        assert_eq!(out.len(), 2);
        assert_eq!(out[0].len(), 2);
        assert_eq!(out[1].len(), 2);
        assert_ne!(out[0][0], out[0][1]);
        assert_ne!(out[1][0], out[1][1]);

        // Cross-question coupling gate: changing q2's type embedding must shift q1's joint logits
        // via DecoderLayer self_attn across fields.
        let mut q_perturbed = questions.clone();
        q_perturbed[1].question_type = 2;
        let out_perturbed = head
            .forward(&hidden_rows, &input_ids, &q_perturbed)
            .unwrap();
        assert!(
            (out[0][0] - out_perturbed[0][0]).abs() > 1e-6,
            "joint field self-attention did not propagate cross-question information"
        );

        // Error validations
        assert!(head.forward(&[], &[], &questions).is_err());
        assert!(head.forward(&hidden_rows, &[1, 2], &questions).is_err());
        assert!(head.forward(&hidden_rows, &input_ids, &[]).is_err());

        let _ = fs::remove_dir_all(&tmp);
    }

    #[test]
    fn clef_head_config_and_logit_reordering_validation() {
        assert!(
            ClefHeadConfig::from_json(&json!({
                "hidden_size": 4096,
                "width": 1024,
                "routing_layers": 2,
                "layers": 4,
                "heads": 16,
                "feedforward": 4096
            }))
            .is_ok()
        );
        // width not divisible by heads
        assert!(
            ClefHeadConfig::from_json(&json!({
                "hidden_size": 4096,
                "width": 1024,
                "routing_layers": 2,
                "layers": 4,
                "heads": 15,
                "feedforward": 4096
            }))
            .is_err()
        );

        let q = ClefEncodedQuestion {
            question_id: "team".into(),
            question_type: 1,
            question_span: (1, 2),
            option_spans: vec![(2, 3), (3, 4)],
            option_ids: vec!["alpha".into(), "zeta".into()],
        };
        let field = DecisionField {
            name: "team".into(),
            field_type: "choice".into(),
            description: "Pick team".into(),
            values: vec![json!("zeta"), json!("alpha")],
            value_descriptions: vec!["Zeta".into(), "Alpha".into()],
            keys: vec!["zeta".into(), "alpha".into()],
        };
        let reordered = reorder_clef_question_logits(&q, &field, &[1.25, -0.75]).unwrap();
        assert_eq!(reordered, vec![-0.75, 1.25]);
    }

    fn splitmix_vec(seed_tag: &str, count: usize, scale: f64) -> Vec<f32> {
        let mut h: u64 = 0xCBF2_9CE4_8422_2325;
        for &b in seed_tag.as_bytes() {
            h ^= b as u64;
            h = h.wrapping_mul(0x0000_0100_0000_01B3);
        }
        let mut out = Vec::with_capacity(count);
        for i in 0..count {
            let mut z = h.wrapping_add(i as u64).wrapping_add(0x9E37_79B9_7F4A_7C15);
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
            z ^= z >> 31;
            let u = ((z >> 11) as f64) * (1.0 / ((1u64 << 53) as f64));
            out.push(((u * 2.0 - 1.0) * scale) as f32);
        }
        out
    }

    #[test]
    fn clef_joint_head_matches_pytorch_joint_schema_head_goldens() {
        let tmp = std::env::temp_dir().join(format!("w34_clef_pt_golden_{}", std::process::id()));
        let _ = fs::remove_dir_all(&tmp);
        fs::create_dir_all(&tmp).unwrap();
        let head_path = tmp.join("joint_head.safetensors");
        let model_path = tmp.join("model.safetensors");

        let mut specs: Vec<(String, Vec<usize>)> = vec![
            ("hidden_norm.weight".into(), vec![8]),
            ("hidden_norm.bias".into(), vec![8]),
            ("memory_projection.weight".into(), vec![4, 8]),
            ("question_projection.weight".into(), vec![4, 8]),
            ("option_question_projection.weight".into(), vec![4, 8]),
            ("global_projection.weight".into(), vec![4, 8]),
            ("option_context_projection.weight".into(), vec![4, 8]),
            ("option_lexical_projection.weight".into(), vec![4, 8]),
            ("type_embedding.weight".into(), vec![3, 4]),
        ];
        for i in 0..2 {
            specs.extend([
                (format!("evidence_layers.{i}.query_norm.weight"), vec![4]),
                (format!("evidence_layers.{i}.query_norm.bias"), vec![4]),
                (format!("evidence_layers.{i}.memory_norm.weight"), vec![4]),
                (format!("evidence_layers.{i}.memory_norm.bias"), vec![4]),
                (
                    format!("evidence_layers.{i}.attention.in_proj_weight"),
                    vec![12, 4],
                ),
                (
                    format!("evidence_layers.{i}.attention.in_proj_bias"),
                    vec![12],
                ),
                (
                    format!("evidence_layers.{i}.attention.out_proj.weight"),
                    vec![4, 4],
                ),
                (
                    format!("evidence_layers.{i}.attention.out_proj.bias"),
                    vec![4],
                ),
                (
                    format!("evidence_layers.{i}.feedforward_norm.weight"),
                    vec![4],
                ),
                (
                    format!("evidence_layers.{i}.feedforward_norm.bias"),
                    vec![4],
                ),
                (
                    format!("evidence_layers.{i}.feedforward.0.weight"),
                    vec![8, 4],
                ),
                (format!("evidence_layers.{i}.feedforward.0.bias"), vec![8]),
                (
                    format!("evidence_layers.{i}.feedforward.3.weight"),
                    vec![4, 8],
                ),
                (format!("evidence_layers.{i}.feedforward.3.bias"), vec![4]),
            ]);
        }
        specs.extend([
            ("option_summary_norm.weight".into(), vec![4]),
            ("option_summary_norm.bias".into(), vec![4]),
        ]);
        for i in 0..2 {
            specs.extend([
                (format!("layers.{i}.norm1.weight"), vec![4]),
                (format!("layers.{i}.norm1.bias"), vec![4]),
                (format!("layers.{i}.self_attn.in_proj_weight"), vec![12, 4]),
                (format!("layers.{i}.self_attn.in_proj_bias"), vec![12]),
                (format!("layers.{i}.self_attn.out_proj.weight"), vec![4, 4]),
                (format!("layers.{i}.self_attn.out_proj.bias"), vec![4]),
                (format!("layers.{i}.norm2.weight"), vec![4]),
                (format!("layers.{i}.norm2.bias"), vec![4]),
                (
                    format!("layers.{i}.multihead_attn.in_proj_weight"),
                    vec![12, 4],
                ),
                (format!("layers.{i}.multihead_attn.in_proj_bias"), vec![12]),
                (
                    format!("layers.{i}.multihead_attn.out_proj.weight"),
                    vec![4, 4],
                ),
                (format!("layers.{i}.multihead_attn.out_proj.bias"), vec![4]),
                (format!("layers.{i}.norm3.weight"), vec![4]),
                (format!("layers.{i}.norm3.bias"), vec![4]),
                (format!("layers.{i}.linear1.weight"), vec![8, 4]),
                (format!("layers.{i}.linear1.bias"), vec![8]),
                (format!("layers.{i}.linear2.weight"), vec![4, 8]),
                (format!("layers.{i}.linear2.bias"), vec![4]),
            ]);
        }
        specs.extend([
            ("field_norm.weight".into(), vec![4]),
            ("field_norm.bias".into(), vec![4]),
            ("option_norm.weight".into(), vec![4]),
            ("option_norm.bias".into(), vec![4]),
            ("residual_scorer.0.weight".into(), vec![4, 16]),
            ("residual_scorer.0.bias".into(), vec![4]),
            ("residual_scorer.3.weight".into(), vec![1, 4]),
            ("residual_scorer.3.bias".into(), vec![1]),
            ("prior_logit_scale".into(), vec![]),
            ("joint_logit_scale".into(), vec![]),
            ("residual_gate".into(), vec![]),
        ]);

        let named: Vec<(String, Vec<usize>, Vec<u8>)> = specs
            .into_iter()
            .map(|(k, shape)| {
                let scale = if k.ends_with(".bias") {
                    0.08
                } else if k.ends_with(".weight") && shape.len() == 1 {
                    1.0
                } else if shape.is_empty() {
                    0.5
                } else {
                    0.3
                };
                let n = shape.iter().product::<usize>().max(1);
                let bytes = f32_bytes(&splitmix_vec(&k, n, scale));
                (k, shape, bytes)
            })
            .collect();
        let refs: Vec<(&str, &str, &[usize], &[u8])> = named
            .iter()
            .map(|(n, s, b)| (n.as_str(), "F32", s.as_slice(), b.as_slice()))
            .collect();
        make_safetensors(&head_path, &refs);
        fs::write(
            tmp.join("joint_head_config.json"),
            r#"{"hidden_size":8,"width":4,"routing_layers":2,"layers":2,"heads":2,"feedforward":8}"#,
        )
        .unwrap();

        // Also test Archive-backed on-demand `read_row_f32` via `lm_head.weight`!
        let out_emb_bytes = f32_bytes(&splitmix_vec("out_emb", 16 * 8, 0.4));
        make_safetensors(
            &model_path,
            &[("lm_head.weight", "F32", &[16, 8], &out_emb_bytes)],
        );
        fs::write(
            tmp.join("config.json"),
            r#"{"architectures":["ClefModel"],"model_type":"qwen3_5_text","hidden_size":8,"num_hidden_layers":1,"num_attention_heads":2,"num_key_value_heads":2,"head_dim":4,"intermediate_size":16,"vocab_size":16,"tie_word_embeddings":false}"#,
        )
        .unwrap();

        let model_cfg = ModelConfig::from_dir(&tmp).unwrap();
        let head = ClefJointHead::from_dir_and_config(&tmp, &model_cfg).unwrap();

        let seq_flat = splitmix_vec("seq_hidden", 12 * 8, 0.6);
        let hidden_rows: Vec<Vec<f32>> = seq_flat
            .as_chunks::<8>()
            .0
            .iter()
            .map(|c| c.to_vec())
            .collect();
        let input_ids = vec![2, 5, 1, 7, 3, 9, 4, 8, 6, 11, 10, 13];
        let questions = vec![
            ClefEncodedQuestion {
                question_id: "urgent".into(),
                question_type: 0,
                question_span: (1, 3),
                option_spans: vec![(3, 5), (5, 7)],
                option_ids: vec!["true".into(), "false".into()],
            },
            ClefEncodedQuestion {
                question_id: "route".into(),
                question_type: 1,
                question_span: (7, 8),
                option_spans: vec![(8, 9), (9, 11), (11, 12)],
                option_ids: vec!["alpha".into(), "beta".into(), "zeta".into()],
            },
        ];

        let logits = head.forward(&hidden_rows, &input_ids, &questions).unwrap();
        let expected_q0 = [0.79531544f64, 0.57161945f64];
        let expected_q1 = [0.13990906f64, 0.55448157f64, 0.00338013f64];
        for (got, exp) in logits[0].iter().zip(expected_q0) {
            assert!(
                (got - exp).abs() < 2e-6,
                "q0 logit mismatch vs PyTorch JointSchemaHead: got {got}, expected {exp}"
            );
        }
        for (got, exp) in logits[1].iter().zip(expected_q1) {
            assert!(
                (got - exp).abs() < 2e-6,
                "q1 logit mismatch vs PyTorch JointSchemaHead: got {got}, expected {exp}"
            );
        }
        let _ = fs::remove_dir_all(&tmp);
    }

    fn build_synthetic_clef_tokenizer() -> Tokenizer {
        use tokenizers::AddedToken;
        use tokenizers::models::bpe::{BPE, Vocab};

        let mut vocab = Vocab::default();
        vocab.insert("[UNK]".to_owned(), 0);
        for (next_id, ch) in (1u32..).zip(
            ['\t', '\n', '\r']
                .into_iter()
                .chain((0x20u8..=0x7eu8).map(|b| b as char)),
        ) {
            vocab.insert(ch.to_string(), next_id);
        }
        let bpe = BPE::builder()
            .vocab_and_merges(vocab, Vec::new())
            .unk_token("[UNK]".to_owned())
            .build()
            .unwrap();
        let mut tok = Tokenizer::new(bpe);
        let specials = [
            "<|endoftext|>",
            "<|im_start|>",
            "<|im_end|>",
            "<|vision_start|>",
            "<|vision_end|>",
            "<|image_pad|>",
            "<think>",
            "</think>",
            "<tool_call>",
            "</tool_call>",
            "<tool_response>",
            "</tool_response>",
        ];
        let added: Vec<AddedToken> = specials
            .iter()
            .map(|&s| AddedToken::from(s, true))
            .collect();
        tok.add_special_tokens(&added);
        // Also register a non-special added control token to verify non-bracket sanitization.
        tok.add_tokens(&[AddedToken::from("@@SECRET_ADDED@@", false)]);
        tok
    }

    #[test]
    fn clef_joint_head_rejects_truncated_safetensors_payload() {
        let tmp = std::env::temp_dir().join(format!("w34_clef_trunc_{}", std::process::id()));
        let _ = fs::remove_dir_all(&tmp);
        fs::create_dir_all(&tmp).unwrap();
        let short_bytes = f32_bytes(&[1.0, 2.0]);
        let head_path = tmp.join("joint_head.safetensors");
        make_safetensors(
            &head_path,
            &[("hidden_norm.weight", "F32", &[4], &short_bytes)],
        );
        let archive = SafetensorsArchive::open(&head_path).unwrap();
        let err = read_exact_1d(&archive, "hidden_norm.weight", 4).unwrap_err();
        assert!(err.contains("length 2 != expected 4"), "{err}");

        let mat_path = tmp.join("mat_trunc.safetensors");
        make_safetensors(
            &mat_path,
            &[("memory_projection.weight", "F32", &[2, 2], &short_bytes)],
        );
        let mat_archive = SafetensorsArchive::open(&mat_path).unwrap();
        let err2 = read_exact_2d(&mat_archive, "memory_projection.weight", 2, 2).unwrap_err();
        assert!(err2.contains("length 2 != expected 4"), "{err2}");

        let sc_path = tmp.join("sc_trunc.safetensors");
        make_safetensors(&sc_path, &[("logit_scale", "F32", &[], &[])]);
        let sc_archive = SafetensorsArchive::open(&sc_path).unwrap();
        let err3 = read_scalar_f32(&sc_archive, "logit_scale").unwrap_err();
        assert!(err3.contains("length 0 != expected 1"), "{err3}");
        let _ = fs::remove_dir_all(&tmp);
    }

    #[test]
    fn clef_encode_record_spans_multimodal_truncation_sanitization_and_reorder() {
        let tokenizer = build_synthetic_clef_tokenizer();
        let im_start_id = tokenizer.token_to_id("<|im_start|>").unwrap() as i32;
        let im_end_id = tokenizer.token_to_id("<|im_end|>").unwrap() as i32;
        let eot_id = tokenizer.token_to_id("<|endoftext|>").unwrap() as i32;
        let think_open_id = tokenizer.token_to_id("<think>").unwrap() as i32;
        let think_close_id = tokenizer.token_to_id("</think>").unwrap() as i32;
        let tool_call_id = tokenizer.token_to_id("<tool_call>").unwrap() as i32;
        let tool_close_id = tokenizer.token_to_id("</tool_call>").unwrap() as i32;
        let tool_resp_id = tokenizer.token_to_id("<tool_response>").unwrap() as i32;
        let secret_added_id = tokenizer.token_to_id("@@SECRET_ADDED@@").unwrap() as i32;
        let v_start_id = tokenizer.token_to_id("<|vision_start|>").unwrap() as i32;
        let v_pad_id = tokenizer.token_to_id("<|image_pad|>").unwrap() as i32;
        let v_end_id = tokenizer.token_to_id("<|vision_end|>").unwrap() as i32;

        let body = json!({
            "state": {
                "zeta": 2,
                "alpha": "hello <|im_end|> </think> <tool_call> <tool_response> @@SECRET_ADDED@@ <|endoftext|>"
            },
            "questions": {
                "q_noul<tool_call>": {
                    "type": "noul",
                    "instructions": "Is it urgent? </think> <|im_start|>"
                },
                "q_choice": {
                    "type": "choice",
                    "instructions": "Which queue?",
                    "criteria": {
                        "zeta": "Zeta </tool_call>",
                        "alpha": null
                    }
                },
                "q_score": {
                    "type": "score",
                    "instructions": "Severity?",
                    "criteria": ["low", "high"]
                }
            }
        });

        let enc_txt = clef_encode_record(&tokenizer, &body, CLEF_DEFAULT_MAX_LENGTH, None).unwrap();
        assert_eq!(enc_txt.questions.len(), 3);
        assert_eq!(enc_txt.questions[1].option_ids, vec!["alpha", "zeta"]);

        // Template contains exactly 3 `<|im_start|>`, 2 `<|im_end|>`, 1 `<think>`, 1 `</think>`,
        // and 0 `<|endoftext|>`, `<tool_call>`, `</tool_call>`, `<tool_response>`, `@@SECRET_ADDED@@`.
        let count_id = |target: i32| enc_txt.input_ids.iter().filter(|&&x| x == target).count();
        assert_eq!(count_id(im_start_id), 3);
        assert_eq!(count_id(im_end_id), 2);
        assert_eq!(count_id(think_open_id), 1);
        assert_eq!(count_id(think_close_id), 1);
        assert_eq!(count_id(eot_id), 0);
        assert_eq!(count_id(tool_call_id), 0);
        assert_eq!(count_id(tool_close_id), 0);
        assert_eq!(count_id(tool_resp_id), 0);
        assert_eq!(count_id(secret_added_id), 0);

        // Multimodal record shifts every question and option span by `1 + 12 + 1 + 1 = 15` tokens
        // (start + 12 image_pad + end + '\n') and places the media block right after `STATE:\n`.
        let media = (
            ClefMediaTokens {
                vision_start_id: v_start_id,
                image_pad_id: v_pad_id,
                vision_end_id: v_end_id,
            },
            &[12usize][..],
        );
        let enc_img =
            clef_encode_record(&tokenizer, &body, CLEF_DEFAULT_MAX_LENGTH, Some(media)).unwrap();
        let shift = 15usize;
        assert_eq!(enc_img.input_ids.len(), enc_txt.input_ids.len() + shift);
        let vs_pos = enc_img
            .input_ids
            .iter()
            .position(|&t| t == v_start_id)
            .expect("vision_start present");
        assert!(
            enc_img.input_ids[vs_pos + 1..vs_pos + 13]
                .iter()
                .all(|&t| t == v_pad_id)
        );
        assert_eq!(enc_img.input_ids[vs_pos + 13], v_end_id);
        for (qt, qi) in enc_txt.questions.iter().zip(&enc_img.questions) {
            assert_eq!(
                qi.question_span,
                (qt.question_span.0 + shift, qt.question_span.1 + shift)
            );
            for (&spt, &spi) in qt.option_spans.iter().zip(&qi.option_spans) {
                assert_eq!(spi, (spt.0 + shift, spt.1 + shift));
            }
        }

        // State truncation: tightening `max_length` to `enc_txt.input_ids.len() - 20` trims
        // exactly 20 state tokens and shifts all schema spans left by 20; setting `max_length`
        // to `fixed_length` (leaving 0 state tokens) fails with 400 Bad Request.
        let trimmed_max = enc_txt.input_ids.len() - 20;
        let enc_trim = clef_encode_record(&tokenizer, &body, trimmed_max, None).unwrap();
        assert_eq!(enc_trim.input_ids.len(), trimmed_max);
        for (qt, qr) in enc_txt.questions.iter().zip(&enc_trim.questions) {
            assert_eq!(
                qr.question_span,
                (qt.question_span.0 - 20, qt.question_span.1 - 20)
            );
        }
        assert!(clef_encode_record(&tokenizer, &body, 32, None).is_err());

        // `reorder_clef_question_logits` maps alphabetical ["alpha", "zeta"] back to caller ["zeta", "alpha"]
        // and rejects length mismatches or unknown option keys.
        let field = DecisionField {
            name: "q_choice".into(),
            field_type: "choice".into(),
            description: "Which queue?".into(),
            values: vec![json!("zeta"), json!("alpha")],
            value_descriptions: vec!["Zeta".into(), "alpha".into()],
            keys: vec!["zeta".into(), "alpha".into()],
        };
        let reordered =
            reorder_clef_question_logits(&enc_txt.questions[1], &field, &[1.25, 4.5]).unwrap();
        assert_eq!(reordered, vec![4.5, 1.25]);
        assert!(reorder_clef_question_logits(&enc_txt.questions[1], &field, &[1.0]).is_err());

        let mut bad_field = field.clone();
        bad_field.keys = vec!["zeta".into(), "missing".into()];
        assert!(
            reorder_clef_question_logits(&enc_txt.questions[1], &bad_field, &[1.25, 4.5]).is_err()
        );
    }

    fn resolve_clef_flash_hf_tokenizer() -> Option<std::path::PathBuf> {
        if let Ok(p) = std::env::var("W34_CLEF_TOKENIZER") {
            let pb = std::path::PathBuf::from(p);
            if pb.is_file() {
                return Some(pb);
            }
        }
        let hub_root = std::env::var_os("HF_HUB_CACHE")
            .map(std::path::PathBuf::from)
            .or_else(|| {
                std::env::var_os("HF_HOME").map(|h| std::path::PathBuf::from(h).join("hub"))
            })
            .or_else(|| {
                std::env::var_os("HOME")
                    .map(|h| std::path::PathBuf::from(h).join(".cache/huggingface/hub"))
            })?;
        let repo = hub_root.join("models--Cloudflare--clef-flash");
        if let Ok(rev) = fs::read_to_string(repo.join("refs/main")) {
            let cand = repo
                .join("snapshots")
                .join(rev.trim())
                .join("tokenizer.json");
            if cand.is_file() {
                return Some(cand);
            }
        }
        None
    }

    #[test]
    #[ignore = "requires Cloudflare/clef-flash tokenizer in local HF cache or W34_CLEF_TOKENIZER"]
    fn clef_encode_record_matches_clef_flash_hf_tokenizer_goldens() {
        let tok_path = resolve_clef_flash_hf_tokenizer()
            .expect("set W34_CLEF_TOKENIZER or cache Cloudflare/clef-flash");
        let tokenizer = Tokenizer::from_file(&tok_path).unwrap();
        let body = json!({
            "state": {"zeta": 2, "alpha": "hello"},
            "questions": {
                "q_noul": {"type": "noul", "instructions": "Is it urgent?"},
                "q_choice": {
                    "type": "choice",
                    "instructions": "Which queue?",
                    "criteria": {"zeta": "Zeta queue", "alpha": null}
                },
                "q_score": {
                    "type": "score",
                    "instructions": "Severity?",
                    "criteria": ["low", "high"]
                }
            }
        });

        let enc_txt = clef_encode_record(&tokenizer, &body, CLEF_DEFAULT_MAX_LENGTH, None).unwrap();
        assert_eq!(enc_txt.input_ids.len(), 274);
        assert_eq!(
            &enc_txt.input_ids[..10],
            &[248045, 8678, 198, 4274, 279, 4434, 1528, 321, 10485, 13]
        );
        assert_eq!(
            &enc_txt.input_ids[enc_txt.input_ids.len() - 10..],
            &[248069, 271, 25310, 3118, 7295, 34651, 41402, 23451, 50, 25]
        );
        assert_eq!(enc_txt.questions[0].question_span, (73, 77));
        assert_eq!(
            enc_txt.questions[0].option_spans,
            vec![(88, 106), (112, 130)]
        );
        assert_eq!(enc_txt.questions[1].question_span, (152, 155));
        assert_eq!(
            enc_txt.questions[1].option_spans,
            vec![(166, 172), (178, 191)]
        );
        assert_eq!(enc_txt.questions[1].option_ids, vec!["alpha", "zeta"]);
        assert_eq!(enc_txt.questions[2].question_span, (213, 215));
        assert_eq!(
            enc_txt.questions[2].option_spans,
            vec![(226, 236), (242, 252)]
        );

        let media = (
            ClefMediaTokens {
                vision_start_id: 248053,
                image_pad_id: 248056,
                vision_end_id: 248054,
            },
            &[64usize][..],
        );
        let enc_img =
            clef_encode_record(&tokenizer, &body, CLEF_DEFAULT_MAX_LENGTH, Some(media)).unwrap();
        assert_eq!(enc_img.input_ids.len(), 341);
        assert_eq!(enc_img.questions[0].question_span, (140, 144));
        assert_eq!(
            enc_img.questions[0].option_spans,
            vec![(155, 173), (179, 197)]
        );
        assert_eq!(enc_img.questions[1].question_span, (219, 222));
        assert_eq!(
            enc_img.questions[1].option_spans,
            vec![(233, 239), (245, 258)]
        );
        assert_eq!(enc_img.questions[2].question_span, (280, 282));
        assert_eq!(
            enc_img.questions[2].option_spans,
            vec![(293, 303), (309, 319)]
        );
    }
}
