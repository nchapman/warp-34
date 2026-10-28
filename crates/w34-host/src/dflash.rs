//! Native BF16 DFlash2 rank. Donor: SGLang v0.5.19 models/dflash.py.
//! Committed target taps are projected once into a separate paged KV namespace.
//! A proposal writes only a private, disposable M-row suffix; it never advances
//! committed context. Attention/MLP are TP-sharded; context FC and convs replicated.
#![allow(clippy::too_many_arguments, clippy::undocumented_unsafe_blocks)]
use crate::cache::{KvFormatPolicy, KvPool};
use crate::cuda::{DeviceArena, DeviceBuffer, Result, Tp2Collectives, check_rc};
use crate::loader::{ModelConfig, SafetensorsArchive};
use crate::model::{Qwen3_5ModelRank, loader_err};
use core::ffi::c_void;
use serde_json::Value;
use std::path::Path;
use w34_kernels::ffi::{self, W34AttentionBatch};

#[derive(Clone, Debug)]
pub struct DFlash2Config {
    pub model: ModelConfig,
    pub block_size: usize,
    pub target_layers: Vec<usize>,
    pub group: usize,
    pub taps: usize,
    pub selector_rank: usize,
    pub top_k: usize,
    pub mask_token: i32,
    pub window: usize,
}
impl DFlash2Config {
    pub fn load(dir: &Path) -> Result<Self> {
        let text = std::fs::read_to_string(dir.join("config.json"))
            .map_err(|_| loader_err("read DFlash config"))?;
        let root: Value =
            serde_json::from_str(&text).map_err(|_| loader_err("parse DFlash config"))?;
        let body = root.get("transformer_layer_config").unwrap_or(&root);
        let model = ModelConfig::from_json(body).map_err(|_| loader_err("DFlash model config"))?;
        let dc = root.get("dflash_config").unwrap_or(&root);
        let integer = |key: &str| -> Result<usize> {
            dc.get(key)
                .and_then(Value::as_u64)
                .and_then(|v| usize::try_from(v).ok())
                .ok_or_else(|| loader_err("missing DFlash integer config"))
        };
        let target_layers = if let Some(ids) = dc.get("target_layer_ids").and_then(Value::as_array)
        {
            ids.iter()
                .map(|v| {
                    v.as_u64()
                        .and_then(|v| usize::try_from(v).ok())
                        .ok_or_else(|| loader_err("invalid target layer"))
                })
                .collect::<Result<Vec<_>>>()?
        } else {
            // Speculators export hidden-state tuple indices; index 0 is embedding.
            root.get("aux_hidden_state_layer_ids")
                .and_then(Value::as_array)
                .ok_or_else(|| loader_err("missing DFlash taps"))?
                .iter()
                .map(|v| {
                    v.as_u64()
                        .and_then(|v| v.checked_sub(1))
                        .and_then(|v| usize::try_from(v).ok())
                        .ok_or_else(|| loader_err("invalid auxiliary hidden layer"))
                })
                .collect::<Result<Vec<_>>>()?
        };
        let window = if body.get("use_sliding_window").and_then(Value::as_bool) == Some(true) {
            let value = body
                .get("sliding_window")
                .and_then(Value::as_u64)
                .filter(|&v| v > 0 && v <= i32::MAX as u64)
                .ok_or_else(|| loader_err("invalid DFlash sliding window"))?;
            value as usize
        } else {
            0
        };
        let cfg = Self {
            model,
            block_size: integer("block_size")?,
            target_layers,
            group: integer("conv_group_size")?,
            taps: integer("conv_kernel_size")?,
            selector_rank: integer("selector_rank")?,
            top_k: integer("selector_top_k")?,
            mask_token: i32::try_from(integer("mask_token_id")?)
                .map_err(|_| loader_err("mask ID overflow"))?,
            window,
        };
        let m = &cfg.model;
        if !matches!(cfg.block_size, 8 | 16)
            || m.num_hidden_layers != 5
            || cfg.target_layers.len() != 5
            || cfg.target_layers.windows(2).any(|v| v[0] >= v[1])
            || cfg.group == 0
            || !m.hidden_size.is_multiple_of(cfg.group)
            || cfg.taps == 0
            || cfg.taps > cfg.block_size
            || cfg.top_k == 0
            || cfg.top_k > 16
            || cfg.selector_rank == 0
            || cfg.selector_rank > 1024
            || cfg.mask_token < 0
            || cfg.mask_token as usize >= m.vocab_size
            || !matches!(m.hidden_size, 2560 | 5120)
            || !matches!(m.head_dim, 128 | 256)
            || m.num_attention_heads == 0
            || m.num_key_value_heads == 0
            || !m.num_attention_heads.is_multiple_of(m.num_key_value_heads)
            || m.intermediate_size == 0
            || body
                .get("is_causal")
                .or_else(|| root.get("is_causal"))
                .and_then(Value::as_bool)
                == Some(true)
            || body.get("attention_bias").and_then(Value::as_bool) == Some(true)
        {
            return Err(loader_err("unsupported DFlash2 checkpoint geometry"));
        }
        Ok(cfg)
    }
}
struct Linear {
    weight: u64,
    n: usize,
    k: usize,
}
impl Linear {
    fn run(&self, t: &Qwen3_5ModelRank, x: u64, y: u64, rows: usize) -> Result<()> {
        t.cublas
            .gemm_bf16(x, self.weight, y, rows as i32, self.n as i32, self.k as i32)
    }
}
struct Conv {
    projection: Linear,
    base: u64,
}
struct Layer {
    input_norm: u64,
    post_norm: u64,
    q_norm: u64,
    k_norm: u64,
    qkv: Linear,
    out: Linear,
    gate_up: Linear,
    down: Linear,
    attn_conv: Conv,
    mlp_conv: Conv,
}

/// Local vocabulary candidates; IDs already carry the target's TP vocab offset.
#[derive(Debug, Clone)]
pub struct DraftCandidates {
    pub ids: Vec<i32>,
    pub scores: Vec<f32>,
    pub rows: usize,
    pub top_k: usize,
}
impl DraftCandidates {
    pub fn merge(shards: &[Self]) -> Result<Self> {
        let first = shards
            .first()
            .ok_or_else(|| loader_err("empty draft candidate shards"))?;
        let (rows, k) = (first.rows, first.top_k);
        if k == 0
            || rows == 0
            || shards.iter().any(|s| {
                s.rows != rows
                    || s.top_k != k
                    || s.ids.len() != rows * k
                    || s.scores.len() != rows * k
                    || s.ids.iter().any(|&x| x < 0)
                    || s.scores.iter().any(|x| !x.is_finite())
            })
        {
            return Err(loader_err("invalid draft candidates"));
        }
        let mut ids = Vec::with_capacity(rows * k);
        let mut scores = Vec::with_capacity(rows * k);
        for row in 0..rows {
            let mut values: Vec<(f32, i32)> = shards
                .iter()
                .flat_map(|s| (row * k..(row + 1) * k).map(|j| (s.scores[j], s.ids[j])))
                .collect();
            values.sort_by(|a, b| b.0.total_cmp(&a.0).then(a.1.cmp(&b.1)));
            for (score, id) in values.into_iter().take(k) {
                scores.push(score);
                ids.push(id);
            }
        }
        Ok(Self {
            ids,
            scores,
            rows,
            top_k: k,
        })
    }
}

pub struct DFlash2Rank {
    pub cfg: DFlash2Config,
    rank: usize,
    tp: usize,
    target_embedding: u64,
    _weights: DeviceArena,
    scratch: DeviceArena,
    kv: KvPool,
    pages: Vec<i32>,
    capacity: usize,
    max_rows: usize,
    committed: usize,
    proposal_anchor: Option<i32>,
    fc: Linear,
    hidden_norm: u64,
    final_norm: u64,
    layers: Vec<Layer>,
    rope: u64,
    rope_rows: usize,
    hp: Linear,
    pred: u64,
    succ: u64,
    pub hidden: DeviceBuffer,
    projected: DeviceBuffer,
    /// Explicit diagnostic destination; disabled by default, with no device copies.
    pub trace_dir: Option<std::path::PathBuf>,
    /// Counterfactual diagnostic only: substitute captured donor attention outputs.
    /// None in normal execution; never a parity result for the native model.
    pub diagnostic_attention_oracle: Option<std::path::PathBuf>,
}
impl DFlash2Rank {
    pub fn load(
        t: &Qwen3_5ModelRank,
        dir: &Path,
        capacity: usize,
        max_rows: usize,
    ) -> Result<Self> {
        let cfg = DFlash2Config::load(dir)?;
        let c = &cfg.model;
        let (h, d, tp, r) = (c.hidden_size, c.head_dim, t.tp_size, t.rank);
        if !matches!(tp, 1 | 2)
            || r >= tp
            || h != t.cfg.hidden_size
            || c.vocab_size != t.cfg.vocab_size
            || !c.num_attention_heads.is_multiple_of(tp)
            || !c.num_key_value_heads.is_multiple_of(tp)
            || !c.intermediate_size.is_multiple_of(tp)
            || cfg
                .target_layers
                .iter()
                .any(|&x| x >= t.cfg.num_hidden_layers)
            || capacity == 0
            || capacity > 262144
            || max_rows == 0
            || max_rows > 256
            || capacity < max_rows
        {
            return Err(loader_err("DFlash target/rank/capacity mismatch"));
        }
        t.device.make_current()?;
        let archive =
            SafetensorsArchive::open(dir).map_err(|_| loader_err("open DFlash weights"))?;
        let (q, kv, i) = (
            c.num_attention_heads * d / tp,
            c.num_key_value_heads * d / tp,
            c.intermediate_size / tp,
        );
        let coeff = 2 * cfg.taps * (h / cfg.group);
        let elems = h * h * cfg.target_layers.len()
            + 2 * h
            + c.num_hidden_layers
                * ((q + 2 * kv) * h
                    + h * q
                    + 3 * h * i
                    + 2 * coeff * h
                    + 4 * cfg.taps * h
                    + 2 * h
                    + 2 * d)
            + cfg.selector_rank * h
            + 2 * c.vocab_size * cfg.selector_rank;
        let rope_rows = capacity + cfg.block_size;
        let mut weights = DeviceArena::new(
            t.device.device_id,
            elems * 2 + rope_rows * d * 4 + 1024 * 1024,
        )?;
        fn read(a: &SafetensorsArchive, name: &str, shape: &[usize]) -> Result<Vec<u16>> {
            let info = a
                .tensor_info(name)
                .map_err(|_| loader_err("missing DFlash tensor"))?;
            if info.shape != shape || info.dtype != "BF16" {
                return Err(loader_err("DFlash tensor shape/dtype"));
            }
            a.read_bf16_bits(name)
                .map_err(|_| loader_err("read DFlash BF16 tensor"))
        }
        let upload = |a: &mut DeviceArena, name: &str, shape: &[usize]| -> Result<u64> {
            a.upload_slice(&read(&archive, name, shape)?, &t.stream)
        };
        let matrix = |a: &mut DeviceArena,
                      name: &str,
                      n: usize,
                      k: usize,
                      rs: usize,
                      rn: usize,
                      cs: usize,
                      cn: usize|
         -> Result<Linear> {
            let full = read(&archive, &format!("{name}.weight"), &[n, k])?;
            let values: Vec<u16> = full[rs * k..(rs + rn) * k]
                .chunks_exact(k)
                .flat_map(|row| row[cs..cs + cn].iter().copied())
                .collect();
            Ok(Linear {
                weight: a.upload_slice(&values, &t.stream)?,
                n: rn,
                k: cn,
            })
        };
        let fc = matrix(
            &mut weights,
            "fc",
            h,
            h * cfg.target_layers.len(),
            0,
            h,
            0,
            h * cfg.target_layers.len(),
        )?;
        let hidden_norm = upload(&mut weights, "hidden_norm.weight", &[h])?;
        let final_norm = upload(&mut weights, "norm.weight", &[h])?;
        let mut layers = Vec::with_capacity(5);
        for l in 0..5 {
            let prefix = format!("layers.{l}");
            let norm = |arena: &mut DeviceArena, name: &str, size: usize| {
                upload(arena, &format!("{prefix}.{name}.weight"), &[size])
            };
            let mut packed = Vec::with_capacity((q + 2 * kv) * h);
            for (name, local) in [("q_proj", q), ("k_proj", kv), ("v_proj", kv)] {
                let data = read(
                    &archive,
                    &format!("{prefix}.self_attn.{name}.weight"),
                    &[local * tp, h],
                )?;
                packed.extend_from_slice(&data[r * local * h..(r + 1) * local * h]);
            }
            let qkv = Linear {
                weight: weights.upload_slice(&packed, &t.stream)?,
                n: q + 2 * kv,
                k: h,
            };
            let out = matrix(
                &mut weights,
                &format!("{prefix}.self_attn.o_proj"),
                h,
                q * tp,
                0,
                h,
                r * q,
                q,
            )?;
            let mut packed = Vec::with_capacity(2 * i * h);
            for name in ["gate_proj", "up_proj"] {
                let data = read(
                    &archive,
                    &format!("{prefix}.mlp.{name}.weight"),
                    &[i * tp, h],
                )?;
                packed.extend_from_slice(&data[r * i * h..(r + 1) * i * h]);
            }
            let gate_up = Linear {
                weight: weights.upload_slice(&packed, &t.stream)?,
                n: 2 * i,
                k: h,
            };
            let down = matrix(
                &mut weights,
                &format!("{prefix}.mlp.down_proj"),
                h,
                i * tp,
                0,
                h,
                r * i,
                i,
            )?;
            let conv = |arena: &mut DeviceArena, name: &str| -> Result<Conv> {
                Ok(Conv {
                    projection: matrix(
                        arena,
                        &format!("{prefix}.{name}.kernel_projection"),
                        coeff,
                        h,
                        0,
                        coeff,
                        0,
                        h,
                    )?,
                    base: upload(
                        arena,
                        &format!("{prefix}.{name}.base_kernel"),
                        &[2, cfg.taps, h],
                    )?,
                })
            };
            layers.push(Layer {
                input_norm: norm(&mut weights, "input_layernorm", h)?,
                post_norm: norm(&mut weights, "post_attention_layernorm", h)?,
                q_norm: norm(&mut weights, "self_attn.q_norm", d)?,
                k_norm: norm(&mut weights, "self_attn.k_norm", d)?,
                qkv,
                out,
                gate_up,
                down,
                attn_conv: conv(&mut weights, "attention_conv")?,
                mlp_conv: conv(&mut weights, "mlp_conv")?,
            });
        }
        let hp = matrix(
            &mut weights,
            "candidate_selector.hidden_projection",
            cfg.selector_rank,
            h,
            0,
            cfg.selector_rank,
            0,
            h,
        )?;
        let pred = upload(
            &mut weights,
            "candidate_selector.predecessor_codebook",
            &[c.vocab_size, cfg.selector_rank],
        )?;
        let succ = upload(
            &mut weights,
            "candidate_selector.successor_codebook",
            &[c.vocab_size, cfg.selector_rank],
        )?;
        // SGLang RotaryEmbedding deliberately retains its CUDA cache in FP32,
        // even when Q/K and model weights are BF16. Quantizing this table loses
        // ~0.2% at the first Q projection and amplifies across the five layers.
        let mut table = vec![0f32; rope_rows * d];
        for pos in 0..rope_rows {
            for j in 0..d / 2 {
                let angle = pos as f32 * (1.0 / c.rope_theta.powf((2 * j) as f32 / d as f32));
                table[pos * d + j] = angle.cos();
                table[pos * d + d / 2 + j] = angle.sin();
            }
        }
        let rope = weights.upload_slice(&table, &t.stream)?;
        let mut kv = KvPool::new(
            t.device.device_id,
            ffi::W34_KV_NAMESPACE_DFLASH_DRAFT_CTX,
            &[0, 1, 2, 3, 4],
            c.num_key_value_heads / tp,
            d,
            16,
            rope_rows.div_ceil(16),
            KvFormatPolicy::Bf16,
        )?;
        let pages = (0..kv.num_pages)
            .map(|_| kv.alloc_page().unwrap())
            .collect();
        let hidden = DeviceBuffer::alloc(t.device.device_id, cfg.block_size * h * 2)?;
        let projected = DeviceBuffer::alloc(
            t.device.device_id,
            (cfg.block_size - 1) * cfg.selector_rank * 2,
        )?;
        t.stream.synchronize()?;
        Ok(Self {
            cfg,
            rank: r,
            tp,
            target_embedding: t.embed_tokens_ptr,
            _weights: weights,
            scratch: DeviceArena::new(t.device.device_id, 64 * 1024 * 1024)?,
            kv,
            pages,
            capacity,
            max_rows,
            committed: 0,
            proposal_anchor: None,
            fc,
            hidden_norm,
            final_norm,
            layers,
            rope,
            rope_rows,
            hp,
            pred,
            succ,
            hidden,
            projected,
            trace_dir: None,
            diagnostic_attention_oracle: None,
        })
    }
    fn trace(&self, t: &Qwen3_5ModelRank, name: &str, ptr: u64, n: usize) -> Result<()> {
        let Some(dir) = &self.trace_dir else {
            return Ok(());
        };
        let mut bytes = vec![0u8; n * 2];
        check_rc(
            unsafe {
                ffi::w34_memcpy_d2h_async(
                    bytes.as_mut_ptr().cast(),
                    ptr as *const c_void,
                    bytes.len(),
                    t.stream.handle,
                )
            },
            "DFlash trace",
        )?;
        t.stream.synchronize()?;
        std::fs::create_dir_all(dir).map_err(|_| loader_err("DFlash trace directory"))?;
        std::fs::write(dir.join(format!("{name}.bf16")), bytes)
            .map_err(|_| loader_err("DFlash trace write"))
    }
    pub fn committed_len(&self) -> usize {
        self.committed
    }
    pub fn kv_layouts(&self) -> Vec<ffi::W34KvLayerLayout> {
        self.kv.layers.iter().map(|l| l.layout).collect()
    }
    pub fn truncate(&mut self, len: usize) -> Result<()> {
        if len > self.committed {
            return Err(loader_err("DFlash truncate grows context"));
        }
        self.committed = len;
        self.proposal_anchor = None;
        Ok(())
    }
    fn validate_target(&self, t: &Qwen3_5ModelRank) -> Result<()> {
        if self.target_embedding != t.embed_tokens_ptr
            || self.rank != t.rank
            || self.tp != t.tp_size
        {
            return Err(loader_err("DFlash target rank mismatch"));
        }
        t.device.make_current()
    }
    fn metadata(&mut self, t: &Qwen3_5ModelRank, rows: usize) -> Result<(W34AttentionBatch, u64)> {
        let before = self.committed;
        let after = before + rows;
        let positions: Vec<i32> = (before..after).map(|v| v as i32).collect();
        let slots: Vec<i32> = (before..after)
            .map(|p| self.pages[p / 16] * 16 + (p % 16) as i32)
            .collect();
        let pos = self.scratch.upload_slice(&positions, &t.stream)?;
        let batch = W34AttentionBatch {
            query_start_loc: self.scratch.upload_slice(&[0i32, rows as i32], &t.stream)?
                as *const i32,
            context_lens_before: self.scratch.upload_slice(&[before as i32], &t.stream)?
                as *const i32,
            seq_lens_after: self.scratch.upload_slice(&[after as i32], &t.stream)? as *const i32,
            block_tables: self.scratch.upload_slice(&self.pages, &t.stream)? as *const i32,
            write_slot_mapping: self.scratch.upload_slice(&slots, &t.stream)? as *const i32,
            valid_token_mask: core::ptr::null(),
            valid_req_mask: core::ptr::null(),
            num_reqs: 1,
            total_tokens: rows as i32,
            max_blocks_per_req: self.pages.len() as i32,
            max_q_len: rows as i32,
            max_seq_len: after as i32,
            sliding_window: self.cfg.window as i32,
            mask_mode: ffi::W34_ATTN_MASK_NONCAUSAL_WINDOW,
            sm_scale: 1.0 / (self.cfg.model.head_dim as f32).sqrt(),
            reserved: 0,
        };
        Ok((batch, pos))
    }
    fn norm(
        &self,
        t: &Qwen3_5ModelRank,
        x: u64,
        residual: u64,
        w: u64,
        y: u64,
        rows: usize,
    ) -> Result<()> {
        check_rc(
            unsafe {
                ffi::w34_rms_norm_bf16(
                    &t.stream.exec(0, 0),
                    x as *const c_void,
                    residual as *mut c_void,
                    w as *const c_void,
                    y as *mut c_void,
                    rows as i32,
                    self.cfg.model.hidden_size as i32,
                    self.cfg.model.rms_norm_eps,
                )
            },
            "DFlash norm",
        )
    }
    fn rope(
        &self,
        t: &Qwen3_5ModelRank,
        qkv: u64,
        pos: u64,
        qheads: usize,
        layer: &Layer,
        rows: usize,
    ) -> Result<()> {
        check_rc(
            unsafe {
                ffi::w34_dflash_qk_norm_rope_bf16_f32_table(
                    &t.stream.exec(0, 0),
                    qkv as *mut c_void,
                    pos as *const i32,
                    layer.q_norm as *const c_void,
                    layer.k_norm as *const c_void,
                    self.rope as *const c_void,
                    rows as i32,
                    qheads as i32,
                    (self.cfg.model.num_key_value_heads / self.tp) as i32,
                    self.cfg.model.head_dim as i32,
                    self.rope_rows as i32,
                    self.cfg.model.rms_norm_eps,
                )
            },
            "DFlash QK norm/RoPE",
        )
    }
    fn copy(
        t: &Qwen3_5ModelRank,
        src: u64,
        dst: u64,
        rows: usize,
        stride: usize,
        start: usize,
        cols: usize,
    ) -> Result<()> {
        check_rc(
            unsafe {
                ffi::w34_copy_cols_bf16(
                    &t.stream.exec(0, 0),
                    src as *const c_void,
                    dst as *mut c_void,
                    rows as i32,
                    stride as i32,
                    cols as i32,
                    start as i32,
                    0,
                    cols as i32,
                )
            },
            "DFlash column copy",
        )
    }
    /// `tap_f32` is a live device span [rows,5*H] of committed target taps only.
    /// No unverified rows may be passed here. This advances committed context.
    pub fn ingest(&mut self, t: &Qwen3_5ModelRank, tap_f32: u64, rows: usize) -> Result<()> {
        self.validate_target(t)?;
        if rows == 0 {
            return Ok(());
        }
        if rows > self.max_rows
            || rows > self.capacity - self.committed
            || tap_f32 == 0
            || !tap_f32.is_multiple_of(4)
            || tap_f32
                .checked_add(
                    (rows * self.cfg.target_layers.len() * self.cfg.model.hidden_size * 4) as u64,
                )
                .is_none()
        {
            return Err(loader_err("invalid DFlash ingest"));
        }
        self.proposal_anchor = None;
        self.scratch.reset();
        let h = self.cfg.model.hidden_size;
        let q = self.cfg.model.num_attention_heads * self.cfg.model.head_dim / self.tp;
        let kv = self.cfg.model.num_key_value_heads * self.cfg.model.head_dim / self.tp;
        let taps = self
            .scratch
            .alloc_aligned(rows * self.cfg.target_layers.len() * h * 2)?;
        let projected = self.scratch.alloc_aligned(rows * h * 2)?;
        let ctx = self.scratch.alloc_aligned(rows * h * 2)?;
        let packed = self.scratch.alloc_aligned(rows * 2 * kv * 2)?;
        let k = self.scratch.alloc_aligned(rows * kv * 2)?;
        let v = self.scratch.alloc_aligned(rows * kv * 2)?;
        let (batch, pos) = self.metadata(t, rows)?;
        let ex = t.stream.exec(0, 0);
        check_rc(
            unsafe {
                ffi::w34_cast_f32_bf16(
                    &ex,
                    tap_f32 as *const f32,
                    taps as *mut c_void,
                    (rows * self.cfg.target_layers.len() * h) as i64,
                )
            },
            "DFlash tap cast",
        )?;
        self.fc.run(t, taps, projected, rows)?;
        self.norm(t, projected, 0, self.hidden_norm, ctx, rows)?;
        self.trace(t, "context.fc", projected, rows * h)?;
        self.trace(t, "context.hidden", ctx, rows * h)?;
        for (l, layer) in self.layers.iter().enumerate() {
            // Slice K/V from the single resident fused QKV weight, no duplicate.
            let kv_proj = Linear {
                weight: layer.qkv.weight + (q * h * 2) as u64,
                n: 2 * kv,
                k: h,
            };
            kv_proj.run(t, ctx, packed, rows)?;
            self.rope(t, packed, pos, 0, layer, rows)?;
            Self::copy(t, packed, k, rows, 2 * kv, 0, kv)?;
            Self::copy(t, packed, v, rows, 2 * kv, kv, kv)?;
            self.trace(t, &format!("layers.{l}.context_k"), k, rows * kv)?;
            self.trace(t, &format!("layers.{l}.context_v"), v, rows * kv)?;
            check_rc(
                unsafe {
                    ffi::w34_kv_cache_write_bf16(
                        &ex,
                        &self.kv.layers[l].layout,
                        &batch,
                        k as *const c_void,
                        v as *const c_void,
                    )
                },
                "DFlash context KV",
            )?;
        }
        t.stream.synchronize()?;
        self.committed += rows;
        Ok(())
    }
    fn conv(
        &self,
        t: &Qwen3_5ModelRank,
        c: &Conv,
        x: u64,
        coeff: u64,
        y: u64,
        side: i32,
    ) -> Result<()> {
        if side == 0 {
            c.projection.run(t, x, coeff, self.cfg.block_size)?;
        }
        check_rc(
            unsafe {
                ffi::w34_dflash_conv_bf16(
                    &t.stream.exec(0, 0),
                    x as *const c_void,
                    coeff as *const c_void,
                    c.base as *const c_void,
                    y as *mut c_void,
                    self.cfg.block_size as i32,
                    self.cfg.model.hidden_size as i32,
                    self.cfg.block_size as i32,
                    self.cfg.group as i32,
                    self.cfg.taps as i32,
                    side,
                )
            },
            "DFlash grouped conv",
        )
    }
    pub fn forward(
        &mut self,
        t: &Qwen3_5ModelRank,
        anchor: i32,
        comm: Option<&Tp2Collectives>,
    ) -> Result<DraftCandidates> {
        self.validate_target(t)?;
        if anchor < 0
            || anchor as usize >= self.cfg.model.vocab_size
            || (self.tp == 2) != comm.is_some()
        {
            return Err(loader_err("invalid DFlash forward"));
        }
        self.proposal_anchor = None;
        self.scratch.reset();
        let c = &self.cfg.model;
        let (h, m, d) = (c.hidden_size, self.cfg.block_size, c.head_dim);
        let (q, kv, i) = (
            c.num_attention_heads * d / self.tp,
            c.num_key_value_heads * d / self.tp,
            c.intermediate_size / self.tp,
        );
        let residual = self.scratch.alloc_aligned(m * h * 2)?;
        let x = self.scratch.alloc_aligned(m * h * 2)?;
        let normed = self.scratch.alloc_aligned(m * h * 2)?;
        let filtered = self.scratch.alloc_aligned(m * h * 2)?;
        let coeff = self
            .scratch
            .alloc_aligned(m * 2 * self.cfg.taps * (h / self.cfg.group) * 2)?;
        let packed = self.scratch.alloc_aligned(m * (q + 2 * kv) * 2)?;
        let qb = self.scratch.alloc_aligned(m * q * 2)?;
        let kb = self.scratch.alloc_aligned(m * kv * 2)?;
        let vb = self.scratch.alloc_aligned(m * kv * 2)?;
        let attn = self.scratch.alloc_aligned(m * q * 2)?;
        let tmp = self.scratch.alloc_aligned(m * h * 2)?;
        let gu = self.scratch.alloc_aligned(m * 2 * i * 2)?;
        let act = self.scratch.alloc_aligned(m * i * 2)?;
        let mut ids = vec![self.cfg.mask_token; m];
        ids[0] = anchor;
        let ids = self.scratch.upload_slice(&ids, &t.stream)?;
        let (batch, pos) = self.metadata(t, m)?;
        let ex = t.stream.exec(0, 0);
        check_rc(
            unsafe {
                ffi::w34_embed_lookup_f16_to_bf16(
                    &ex,
                    ids as *const i32,
                    t.embed_tokens_ptr as *const c_void,
                    residual as *mut c_void,
                    m as i32,
                    h as i32,
                    t.cfg.vocab_size as i32,
                )
            },
            "DFlash shared embedding",
        )?;
        for (l, layer) in self.layers.iter().enumerate() {
            self.norm(
                t,
                if l == 0 { residual } else { x },
                if l == 0 { 0 } else { residual },
                layer.input_norm,
                normed,
                m,
            )?;
            self.trace(
                t,
                &format!("ops.layers.{l}.input_layernorm.output0"),
                normed,
                m * h,
            )?;
            self.conv(t, &layer.attn_conv, normed, coeff, filtered, 0)?;
            self.trace(
                t,
                &format!("ops.layers.{l}.self_attn.qkv_proj.input0"),
                filtered,
                m * h,
            )?;
            layer.qkv.run(t, filtered, packed, m)?;
            self.trace(
                t,
                &format!("ops.layers.{l}.self_attn.qkv_proj.output0"),
                packed,
                m * (q + 2 * kv),
            )?;
            self.rope(t, packed, pos, q / d, layer, m)?;
            Self::copy(t, packed, qb, m, q + 2 * kv, 0, q)?;
            Self::copy(t, packed, kb, m, q + 2 * kv, q, kv)?;
            Self::copy(t, packed, vb, m, q + 2 * kv, q + kv, kv)?;
            check_rc(
                unsafe {
                    ffi::w34_kv_cache_write_bf16(
                        &ex,
                        &self.kv.layers[l].layout,
                        &batch,
                        kb as *const c_void,
                        vb as *const c_void,
                    )
                },
                "DFlash query KV",
            )?;
            check_rc(
                unsafe {
                    ffi::w34_paged_attention_bf16(
                        &ex,
                        &self.kv.layers[l].layout,
                        &batch,
                        qb as *const c_void,
                        attn as *mut c_void,
                        (q / d) as i32,
                    )
                },
                "DFlash paged attention",
            )?;
            self.trace(t, &format!("layers.{l}.q"), qb, m * q)?;
            self.trace(t, &format!("layers.{l}.attention"), attn, m * q)?;
            if let Some(path) = &self.diagnostic_attention_oracle {
                let archive = SafetensorsArchive::open(path)
                    .map_err(|_| loader_err("attention diagnostic archive"))?;
                let bytes = archive
                    .read_bytes(&format!("layers.{l}.attention"))
                    .map_err(|_| loader_err("attention diagnostic tensor"))?;
                if bytes.len() != m * q * 2 {
                    return Err(loader_err("attention diagnostic shape"));
                }
                check_rc(
                    unsafe {
                        ffi::w34_memcpy_h2d_async(
                            attn as *mut c_void,
                            bytes.as_ptr().cast(),
                            bytes.len(),
                            t.stream.handle,
                        )
                    },
                    "attention diagnostic injection",
                )?;
                t.stream.synchronize()?;
            }
            layer.out.run(t, attn, tmp, m)?;
            if let Some(comm) = comm {
                comm.all_reduce_bf16(&ex, tmp, tmp, self.rank as i32, (m * h) as i32)?;
            }
            self.trace(
                t,
                &format!("gemms.layers.{l}.self_attn.o_proj.output"),
                tmp,
                m * h,
            )?;
            self.conv(t, &layer.attn_conv, tmp, coeff, x, 1)?;
            self.trace(
                t,
                &format!("norms.layers.{l}.post_attention_layernorm.call0.input0"),
                x,
                m * h,
            )?;
            self.norm(t, x, residual, layer.post_norm, normed, m)?;
            self.trace(
                t,
                &format!("ops.layers.{l}.post_attention_layernorm.output0"),
                normed,
                m * h,
            )?;
            self.conv(t, &layer.mlp_conv, normed, coeff, filtered, 0)?;
            self.trace(
                t,
                &format!("ops.layers.{l}.mlp.gate_up_proj.input0"),
                filtered,
                m * h,
            )?;
            layer.gate_up.run(t, filtered, gu, m)?;
            self.trace(
                t,
                &format!("ops.layers.{l}.mlp.gate_up_proj.output0"),
                gu,
                m * 2 * i,
            )?;
            check_rc(
                unsafe {
                    ffi::w34_silu_and_mul_bf16(
                        &ex,
                        gu as *const c_void,
                        act as *mut c_void,
                        m as i32,
                        i as i32,
                    )
                },
                "DFlash SiLU",
            )?;
            self.trace(
                t,
                &format!("ops.layers.{l}.mlp.down_proj.input0"),
                act,
                m * i,
            )?;
            layer.down.run(t, act, tmp, m)?;
            self.trace(
                t,
                &format!("ops.layers.{l}.mlp.down_proj.output0"),
                tmp,
                m * h,
            )?;
            if let Some(comm) = comm {
                comm.all_reduce_bf16(&ex, tmp, tmp, self.rank as i32, (m * h) as i32)?;
            }
            self.conv(t, &layer.mlp_conv, tmp, coeff, x, 1)?;
            self.trace(t, &format!("layers.{l}.hidden"), x, m * h)?;
            self.trace(t, &format!("layers.{l}.residual"), residual, m * h)?;
        }
        self.norm(t, x, residual, self.final_norm, self.hidden.addr(), m)?;
        let draft_hidden = self.hidden.addr() + (h * 2) as u64;
        self.hp.run(t, draft_hidden, self.projected.addr(), m - 1)?;
        let head_input = self.scratch.alloc_aligned((m - 1) * h * 2)?;
        let logits = self
            .scratch
            .alloc_aligned((m - 1) * t.local_vocab_size * 2)?;
        // Explicit dtype boundary into the target's single resident FP16 head.
        check_rc(
            unsafe {
                ffi::w34_cast_bf16_to_f16(
                    &ex,
                    draft_hidden as *const c_void,
                    head_input as *mut c_void,
                    ((m - 1) * h) as i64,
                )
            },
            "DFlash shared head cast",
        )?;
        t.lm_head.forward(
            &ex,
            &t.cublas,
            t.marlin_ws.addr(),
            t.marlin_ws.len_bytes / 4,
            head_input,
            logits,
            m - 1,
        )?;
        let ids = DeviceBuffer::alloc(t.device.device_id, (m - 1) * self.cfg.top_k * 4)?;
        let scores = DeviceBuffer::alloc(t.device.device_id, (m - 1) * self.cfg.top_k * 4)?;
        check_rc(
            unsafe {
                ffi::w34_topk_rows_f16(
                    &ex,
                    logits as *const c_void,
                    ids.addr() as *mut i32,
                    scores.addr() as *mut f32,
                    (m - 1) as i32,
                    t.local_vocab_size as i32,
                    t.vocab_offset as i32,
                    self.cfg.top_k as i32,
                )
            },
            "DFlash top-k",
        )?;
        let mut out = DraftCandidates {
            ids: vec![0; (m - 1) * self.cfg.top_k],
            scores: vec![0.0; (m - 1) * self.cfg.top_k],
            rows: m - 1,
            top_k: self.cfg.top_k,
        };
        ids.copy_to_host(&mut out.ids, &t.stream)?;
        scores.copy_to_host(&mut out.scores, &t.stream)?;
        if out.scores.iter().any(|x| !x.is_finite()) {
            return Err(loader_err("nonfinite DFlash logits"));
        }
        self.proposal_anchor = Some(anchor);
        Ok(out)
    }
    /// Greedy conditional selector walk over globally merged candidates. The
    /// returned draft tokens are predictions, never target-substituted tokens.
    pub fn select_greedy(
        &mut self,
        t: &Qwen3_5ModelRank,
        anchor: i32,
        candidates: &DraftCandidates,
    ) -> Result<Vec<i32>> {
        self.validate_target(t)?;
        let (k, n) = (self.cfg.block_size - 1, self.cfg.top_k);
        if self.proposal_anchor != Some(anchor)
            || candidates.rows != k
            || candidates.top_k != n
            || candidates.ids.len() != k * n
            || candidates.scores.len() != k * n
            || candidates
                .ids
                .iter()
                .any(|&id| id < 0 || id as usize >= self.cfg.model.vocab_size)
            || candidates.scores.iter().any(|x| !x.is_finite())
        {
            return Err(loader_err("invalid merged DFlash candidates"));
        }
        self.scratch.reset();
        let ids = self.scratch.upload_slice(&candidates.ids, &t.stream)?;
        let vals = self.scratch.upload_slice(&candidates.scores, &t.stream)?;
        let edges = DeviceBuffer::alloc(t.device.device_id, k * n * n * 4)?;
        check_rc(
            unsafe {
                ffi::w34_dflash_edges_bf16(
                    &t.stream.exec(0, 0),
                    ids as *const i32,
                    vals as *const f32,
                    self.projected.addr() as *const c_void,
                    self.pred as *const c_void,
                    self.succ as *const c_void,
                    edges.addr() as *mut f32,
                    anchor,
                    k as i32,
                    n as i32,
                    self.cfg.selector_rank as i32,
                    self.cfg.model.vocab_size as i32,
                )
            },
            "DFlash selector",
        )?;
        let mut scores = vec![0f32; k * n * n];
        edges.copy_to_host(&mut scores, &t.stream)?;
        if scores.iter().any(|x| !x.is_finite()) {
            return Err(loader_err("nonfinite DFlash edges"));
        }
        let mut path = Vec::with_capacity(k);
        let mut predecessor = 0;
        for slot in 0..k {
            let row = &scores[(slot * n + predecessor) * n..(slot * n + predecessor + 1) * n];
            let mut winner = 0;
            for j in 1..n {
                if row[j] > row[winner] {
                    winner = j;
                }
            }
            path.push(candidates.ids[slot * n + winner]);
            predecessor = winner;
        }
        Ok(path)
    }
}
