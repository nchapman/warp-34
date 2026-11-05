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
        let head_dim = self.width / self.heads;
        if self.width > 2048 || head_dim > 256 {
            return Err(format!(
                "clef head geometry exceeds GPU kernel bounds (width={} <= 2048, head_dim={} <= 256)",
                self.width, head_dim
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
        matmul_nt_scalar(
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
                    let s = (dot_f32_scalar(q_head, k_head) as f64) * scale;
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
    output_embeddings: Arc<ClefOutputEmbeddings>,
}

impl ClefJointHead {
    /// Exact 256B-aligned `FP16` `DeviceArena` capacity required to hold all head weights.
    pub fn weight_arena_bytes(&self) -> usize {
        let mut total_bytes = 0usize;
        fn count_slice(total: &mut usize, elems: usize) {
            let aligned = (*total + 255) & !255usize;
            *total = aligned + (elems * 2).max(16);
        }
        fn count_ln(total: &mut usize, ln: &LayerNormWeights) {
            count_slice(total, ln.weight.len());
            count_slice(total, ln.bias.len());
        }
        fn count_lin(total: &mut usize, lin: &LinearWeights) {
            count_slice(total, lin.weight.len());
            if let Some(b) = &lin.bias {
                count_slice(total, b.len());
            }
        }
        count_ln(&mut total_bytes, &self.hidden_norm);
        count_lin(&mut total_bytes, &self.memory_projection);
        count_lin(&mut total_bytes, &self.question_projection);
        count_lin(&mut total_bytes, &self.option_question_projection);
        count_lin(&mut total_bytes, &self.global_projection);
        count_lin(&mut total_bytes, &self.option_context_projection);
        count_lin(&mut total_bytes, &self.option_lexical_projection);
        count_slice(&mut total_bytes, self.type_embedding.len());
        for ev in &self.evidence_layers {
            count_ln(&mut total_bytes, &ev.query_norm);
            count_ln(&mut total_bytes, &ev.memory_norm);
            count_lin(&mut total_bytes, &ev.attention.q_proj);
            count_lin(&mut total_bytes, &ev.attention.kv_proj);
            count_lin(&mut total_bytes, &ev.attention.out_proj);
            count_ln(&mut total_bytes, &ev.feedforward_norm);
            count_lin(&mut total_bytes, &ev.ff0);
            count_lin(&mut total_bytes, &ev.ff3);
        }
        count_ln(&mut total_bytes, &self.option_summary_norm);
        for dec in &self.layers {
            count_ln(&mut total_bytes, &dec.norm1);
            count_lin(&mut total_bytes, &dec.self_attn.q_proj);
            count_lin(&mut total_bytes, &dec.self_attn.kv_proj);
            count_lin(&mut total_bytes, &dec.self_attn.out_proj);
            count_ln(&mut total_bytes, &dec.norm2);
            count_lin(&mut total_bytes, &dec.multihead_attn.q_proj);
            count_lin(&mut total_bytes, &dec.multihead_attn.kv_proj);
            count_lin(&mut total_bytes, &dec.multihead_attn.out_proj);
            count_ln(&mut total_bytes, &dec.norm3);
            count_lin(&mut total_bytes, &dec.linear1);
            count_lin(&mut total_bytes, &dec.linear2);
        }
        count_ln(&mut total_bytes, &self.field_norm);
        count_ln(&mut total_bytes, &self.option_norm);
        count_lin(&mut total_bytes, &self.residual_0);
        count_lin(&mut total_bytes, &self.residual_3);
        total_bytes + 4096
    }

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
            output_embeddings: Arc::new(output_embeddings),
        })
    }

    /// Reference scalar `JointSchemaHead.forward` oracle used by unit tests to verify
    /// `ClefJointHeadGpu` against PyTorch goldens.
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
                let s =
                    (dot_f32_scalar(&opts[oi * w..(oi + 1) * w], field_row) as f64) * inv_sqrt_w;
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
                let prior = prior_scale * (dot_f32_scalar(&lex_row, &anchor) as f64);

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

fn matmul_nt_scalar(
    a: &[f32],
    m: usize,
    k: usize,
    b: &[f32],
    n: usize,
    bias: Option<&[f32]>,
) -> Vec<f32> {
    assert_eq!(a.len(), m * k);
    assert_eq!(b.len(), n * k);
    if let Some(bs) = bias {
        assert_eq!(bs.len(), n);
    }
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

mod encoding;
pub use encoding::*;

#[cfg(feature = "cuda")]
mod gpu;
#[cfg(feature = "cuda")]
pub use gpu::*;

#[cfg(test)]
mod tests;
