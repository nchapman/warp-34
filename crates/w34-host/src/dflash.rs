//! Native BF16 DFlash2 rank. Donor: SGLang v0.5.19 models/dflash.py.
//! Committed target taps are projected once into a separate paged KV namespace.
//! A proposal writes only a private, disposable M-row suffix; it never advances
//! committed context. Attention/MLP are TP-sharded; context FC and convs replicated.
use crate::cache::{KvFormatPolicy, KvPool};
use crate::cuda::{CudaGraph, DeviceArena, DeviceBuffer, Result, Tp2Collectives, check_rc};
use crate::loader::{
    ModelConfig, QuantScheme, SUPPORTED_HADAMARD_BLOCKS, SafetensorsArchive,
    marlin_permute_scales_from_n_major,
};
use crate::model::{
    DEVICE_SCRATCH_ALIGNMENT_BYTES, MARLIN_CHANNELWISE_GROUP_BLOCKS, MARLIN_GROUP_BLOCK_K,
    MARLIN_TILE_K_W4, MARLIN_TILE_N, Qwen3_5ModelRank, U4_NIBBLES_PER_U32, loader_err,
    loader_err_ctx,
};
use core::ffi::c_void;
use serde_json::Value;
use std::path::Path;
use w34_kernels::ffi::{self, W34AttentionBatch};

/// Required number of transformer layers (`5`) and target hidden-state taps in a `DFlash2` drafter.
pub const DFLASH2_NUM_LAYERS: usize = 5;

/// Maximum candidate selector `top_k` (`16`) supported by the native `DFlash2` selector kernel.
pub const DFLASH2_MAX_TOP_K: usize = 16;

/// Maximum low-rank codebook dimension (`1,024`) supported by `DFlash2` candidate selection.
pub const DFLASH2_MAX_SELECTOR_RANK: usize = 1024;

/// Maximum context token capacity (`262,144` tokens) supported by `DFlash2Rank`.
pub const DFLASH2_MAX_KV_CAPACITY: usize = 262_144;

/// Maximum batch rows (`256` tokens) supported in a single `DFlash2Rank` forward pass.
pub const DFLASH2_MAX_BATCH_ROWS: usize = 256;

/// Paged-attention block size (`16` tokens/page) for the `DFlash2` draft context KV pool.
const DFLASH2_KV_PAGE_BLOCK_TOKENS: usize = 16;

/// Alignment/padding slack (`1 MiB`) added to the `DFlash2Rank` weight arena capacity.
const DFLASH2_WEIGHT_ARENA_SLACK_BYTES: usize = 1024 * 1024;

/// Capacity (`32 MiB`) of the fixed-address scratch arena used during `DFlash2` CUDA Graph capture.
const DFLASH2_FORWARD_GRAPH_ARENA_BYTES: usize = 32 * 1024 * 1024;

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
            .map_err(|e| loader_err_ctx("read DFlash config", e.to_string()))?;
        let root: Value = serde_json::from_str(&text)
            .map_err(|e| loader_err_ctx("parse DFlash config", e.to_string()))?;
        let body = root.get("transformer_layer_config").unwrap_or(&root);
        let mut model = ModelConfig::from_drafter_json(body)
            .map_err(|e| loader_err_ctx("DFlash model config", e.to_string()))?;
        if model.quant == QuantScheme::Unquantized
            && root
                .get("quantization_config")
                .is_some_and(|v| !v.is_null())
        {
            model.quant = ModelConfig::parse_quant(root.get("quantization_config"))
                .map_err(|e| loader_err_ctx("DFlash quantization config", e.to_string()))?;
        }
        let has_standalone_qcfg = [
            "quantize_config.json",
            "quant_config.json",
            "quantization_config.json",
        ]
        .iter()
        .any(|name| dir.join(name).exists());
        if has_standalone_qcfg
            || !matches!(
                model.quant,
                QuantScheme::Unquantized
                    | QuantScheme::CompressedTensors {
                        num_bits: 4,
                        symmetric: true,
                        ..
                    }
            )
        {
            return Err(loader_err_ctx(
                "DFlash model config",
                "unsupported quantized DFlash2 drafter; only compressed-tensors W4A16 / W4A16-HGPTQ or unquantized BF16/FP16 drafter weights are supported",
            ));
        }
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
            || m.num_hidden_layers != DFLASH2_NUM_LAYERS
            || cfg.target_layers.len() != DFLASH2_NUM_LAYERS
            || cfg.target_layers.windows(2).any(|v| v[0] >= v[1])
            || cfg.group == 0
            || !m.hidden_size.is_multiple_of(cfg.group)
            || cfg.taps == 0
            || cfg.taps > cfg.block_size
            || cfg.top_k == 0
            || cfg.top_k > DFLASH2_MAX_TOP_K
            || cfg.selector_rank == 0
            || cfg.selector_rank > DFLASH2_MAX_SELECTOR_RANK
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
            || body
                .get("output_multiplier")
                .or_else(|| root.get("output_multiplier"))
                .and_then(Value::as_f64)
                .is_some_and(|v| (v - 1.0).abs() > 1e-5)
            || body
                .get("final_logit_softcapping")
                .or_else(|| root.get("final_logit_softcapping"))
                .and_then(Value::as_f64)
                .is_some_and(|v| v > 0.0)
        {
            return Err(loader_err("unsupported DFlash2 checkpoint geometry"));
        }
        Ok(cfg)
    }
}
#[derive(Clone, Copy)]
enum Linear {
    DenseBf16 {
        weight: u64,
        n: usize,
        k: usize,
    },
    MarlinBf16U4B8 {
        weight: u64,
        scales: u64,
        n: usize,
        k: usize,
        group_blocks: i32,
        hadamard_block: usize,
    },
}
impl Linear {
    fn run(
        &self,
        t: &Qwen3_5ModelRank,
        x: u64,
        y: u64,
        rows: usize,
        had_scratch: u64,
    ) -> Result<()> {
        match *self {
            Self::DenseBf16 { weight, n, k } => {
                t.cublas
                    .gemm_bf16(x, weight, y, rows as i32, n as i32, k as i32)
            }
            Self::MarlinBf16U4B8 {
                weight,
                scales,
                n,
                k,
                group_blocks,
                hadamard_block,
            } => {
                let ex = t.stream.exec(0, 0);
                let a_ptr = if hadamard_block != 0 {
                    if had_scratch == 0 || had_scratch == x {
                        return Err(loader_err("missing DFlash Hadamard scratch buffer"));
                    }
                    check_rc(
                        // SAFETY: x and had_scratch are valid disjoint device buffers for rows * k BF16 elements on t.stream.
                        unsafe {
                            ffi::w34_block_hadamard_bf16(
                                &ex,
                                x as *const c_void,
                                had_scratch as *mut c_void,
                                rows as i32,
                                k as i32,
                                k as i32,
                                hadamard_block as i32,
                            )
                        },
                        "DFlash block_hadamard_bf16",
                    )?;
                    had_scratch
                } else {
                    x
                };
                check_rc(
                    // SAFETY: a_ptr, weight, scales, y, and t.marlin_ws are valid device buffers on t.stream.
                    unsafe {
                        ffi::w34_marlin_gemm_bf16u4(
                            &ex,
                            a_ptr as *const c_void,
                            weight as *const c_void,
                            scales as *const c_void,
                            y as *mut c_void,
                            t.marlin_ws.addr() as *mut i32,
                            rows as i64,
                            n as i64,
                            k as i64,
                            (t.marlin_ws.len_bytes / 4) as i64,
                            group_blocks,
                        )
                    },
                    "DFlash marlin_gemm_bf16u4",
                )
            }
        }
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
    kv: Linear,
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

// Metadata-only accounting shared by serving admission and the actual loader.
fn product(values: &[usize]) -> Result<usize> {
    values.iter().try_fold(1usize, |a, &b| {
        a.checked_mul(b)
            .ok_or_else(|| loader_err("DFlash allocation size overflow"))
    })
}
fn sum(values: &[usize]) -> Result<usize> {
    values.iter().try_fold(0usize, |a, &b| {
        a.checked_add(b)
            .ok_or_else(|| loader_err("DFlash allocation size overflow"))
    })
}
fn dense_metadata(a: &SafetensorsArchive, name: &str, shape: &[usize]) -> Result<usize> {
    let info = a
        .tensor_info(name)
        .map_err(|e| loader_err_ctx("DFlash metadata", format!("{name}: {e}")))?;
    if info.shape != shape
        || !matches!(info.dtype.as_str(), "BF16" | "F16" | "F32")
        || info.len_bytes
            != product(&[product(shape)?, if info.dtype == "F32" { 4 } else { 2 }])? as u64
    {
        return Err(loader_err_ctx("DFlash tensor shape/dtype", name));
    }
    product(&[product(shape)?, 2])
}

fn linear_metadata(
    cfg: &DFlash2Config,
    a: &SafetensorsArchive,
    name: &str,
    n: usize,
    k: usize,
    rn: usize,
    cn: usize,
) -> Result<usize> {
    let quant = cfg
        .model
        .quant
        .for_linear(name)
        .map_err(|e| loader_err_ctx("DFlash metadata", format!("{name}: {e}")))?;
    match quant {
        QuantScheme::Unquantized => {
            dense_metadata(a, &format!("{name}.weight"), &[n, k])?;
            product(&[rn, cn, 2])
        }
        QuantScheme::CompressedTensors {
            num_bits: 4,
            group_size,
            symmetric: true,
            ..
        } => {
            let p = a
                .tensor_info(&format!("{name}.weight_packed"))
                .map_err(|e| loader_err_ctx("DFlash metadata", format!("{name}: {e}")))?;
            let s = a
                .tensor_info(&format!("{name}.weight_scale"))
                .map_err(|e| loader_err_ctx("DFlash metadata", format!("{name}: {e}")))?;
            if a.resolve_name(&format!("{name}.weight_zero_point"))
                .is_some()
                || !k.is_multiple_of(8)
                || p.shape != [n, k / 8]
                || !matches!(p.dtype.as_str(), "I32" | "U32")
                || s.shape.len() != 2
                || s.shape[0] != n
                || s.shape[1] == 0
                || !matches!(s.dtype.as_str(), "BF16" | "F16" | "F32")
                || !k.is_multiple_of(s.shape[1])
            {
                return Err(loader_err_ctx("DFlash packed/scales metadata", name));
            }
            if p.len_bytes != product(&[n, k / 8, 4])? as u64
                || s.len_bytes
                    != product(&[n, s.shape[1], if s.dtype == "F32" { 4 } else { 2 }])? as u64
            {
                return Err(loader_err_ctx("DFlash tensor payload length", name));
            }
            let groups = s.shape[1];
            let gs = k / groups;
            let channelwise = groups == 1 && (group_size == 0 || k > MARLIN_TILE_K_W4);
            if (channelwise && cn != k)
                || (!channelwise
                    && (!matches!(gs, 32 | 64 | 128)
                        || (group_size != 0 && group_size != gs)
                        || !cn.is_multiple_of(gs)))
                || !rn.is_multiple_of(MARLIN_TILE_N)
                || !cn.is_multiple_of(32)
            {
                return Err(loader_err_ctx(
                    "DFlash unsupported scale groups/shard alignment",
                    name,
                ));
            }
            if let Some(block) = cfg
                .model
                .quant
                .input_hadamard(name)
                .map_err(|e| loader_err_ctx("DFlash metadata", format!("{name}: {e}")))?
                && (!SUPPORTED_HADAMARD_BLOCKS.contains(&block) || !cn.is_multiple_of(block))
            {
                return Err(loader_err_ctx("DFlash Hadamard shard alignment", name));
            }
            sum(&[
                product(&[rn, cn])? / 2,
                product(&[rn, cn / gs, 2])?,
                2 * DEVICE_SCRATCH_ALIGNMENT_BYTES,
            ])
        }
        _ => Err(loader_err_ctx(
            "unsupported DFlash linear quantization",
            name,
        )),
    }
}

fn weight_arena_bytes(
    cfg: &DFlash2Config,
    a: &SafetensorsArchive,
    tp: usize,
    capacity: usize,
) -> Result<usize> {
    let c = &cfg.model;
    if !matches!(tp, 1 | 2 | 4)
        || capacity == 0
        || capacity > DFLASH2_MAX_KV_CAPACITY
        || !c.num_attention_heads.is_multiple_of(tp)
        || !c.num_key_value_heads.is_multiple_of(tp)
        || !c.intermediate_size.is_multiple_of(tp)
    {
        return Err(loader_err("DFlash target/rank/capacity mismatch"));
    }
    let (h, d) = (c.hidden_size, c.head_dim);
    let q = product(&[c.num_attention_heads, d])?;
    let kv = product(&[c.num_key_value_heads, d])?;
    let i = c.intermediate_size;
    if [q, kv, i, c.vocab_size]
        .iter()
        .any(|&n| n > i32::MAX as usize)
    {
        return Err(loader_err("DFlash dimensions exceed native kernel limits"));
    }
    let coeff = product(&[2, cfg.taps, h / cfg.group])?;
    let rows = sum(&[capacity, cfg.block_size])?;
    let mut bytes = vec![
        DFLASH2_WEIGHT_ARENA_SLACK_BYTES,
        product(&[rows, d, 4])?,
        product(&[rows.div_ceil(16), 4])?,
    ];
    for name in ["hidden_norm.weight", "norm.weight"] {
        bytes.push(dense_metadata(a, name, &[h])?);
    }
    for name in ["predecessor_codebook", "successor_codebook"] {
        bytes.push(dense_metadata(
            a,
            &format!("candidate_selector.{name}"),
            &[c.vocab_size, cfg.selector_rank],
        )?);
    }
    let fc_k = product(&[h, cfg.target_layers.len()])?;
    bytes.push(linear_metadata(cfg, a, "fc", h, fc_k, h, fc_k)?);
    bytes.push(linear_metadata(
        cfg,
        a,
        "candidate_selector.hidden_projection",
        cfg.selector_rank,
        h,
        cfg.selector_rank,
        h,
    )?);
    for l in 0..c.num_hidden_layers {
        let p = format!("layers.{l}");
        // concat_n_matrix requires identical effective quantization across parts.
        for parts in [
            vec!["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj"],
            vec!["mlp.gate_proj", "mlp.up_proj"],
        ] {
            let mut first = None;
            for part in parts {
                let name = format!("{p}.{part}");
                let quant = c
                    .quant
                    .for_linear(&name)
                    .map_err(|e| loader_err_ctx("DFlash fused quantization", e.to_string()))?;
                let signature = if matches!(quant, QuantScheme::Unquantized) {
                    (false, 0, None)
                } else {
                    let s = a
                        .tensor_info(&format!("{name}.weight_scale"))
                        .map_err(|e| loader_err_ctx("DFlash fused scales", e.to_string()))?;
                    let groups = s.shape.get(1).copied().unwrap_or(0);
                    (
                        true,
                        groups,
                        c.quant
                            .input_hadamard(&name)
                            .map_err(|e| loader_err_ctx("DFlash fused Hadamard", e.to_string()))?,
                    )
                };
                if first.as_ref().is_some_and(|s| s != &signature) {
                    return Err(loader_err_ctx("mismatched fused DFlash projections", name));
                }
                first = Some(signature);
            }
        }
        for (name, size) in [
            ("input_layernorm", h),
            ("post_attention_layernorm", h),
            ("self_attn.q_norm", d),
            ("self_attn.k_norm", d),
        ] {
            bytes.push(dense_metadata(a, &format!("{p}.{name}.weight"), &[size])?);
        }
        for conv in ["attention_conv", "mlp_conv"] {
            bytes.push(dense_metadata(
                a,
                &format!("{p}.{conv}.base_kernel"),
                &[2, cfg.taps, h],
            )?);
            bytes.push(linear_metadata(
                cfg,
                a,
                &format!("{p}.{conv}.kernel_projection"),
                coeff,
                h,
                coeff,
                h,
            )?);
        }
        for (name, n, k, rn, cn) in [
            ("self_attn.q_proj", q, h, q / tp, h),
            ("self_attn.k_proj", kv, h, kv / tp, h),
            ("self_attn.v_proj", kv, h, kv / tp, h),
            ("self_attn.o_proj", h, q, h, q / tp),
            ("mlp.gate_proj", i, h, i / tp, h),
            ("mlp.up_proj", i, h, i / tp, h),
            ("mlp.down_proj", h, i, h, i / tp),
        ] {
            let name = format!("{p}.{name}");
            let b = linear_metadata(cfg, a, &name, n, k, rn, cn)?;
            bytes.push(b);
            if (name.ends_with("k_proj") || name.ends_with("v_proj"))
                && !matches!(
                    c.quant
                        .for_linear(&name)
                        .map_err(|e| loader_err_ctx("DFlash metadata", format!("{name}: {e}")))?,
                    QuantScheme::Unquantized
                )
            {
                bytes.push(b); // quantized fused QKV also owns a separate KV projection
            }
        }
    }
    sum(&bytes)
}

fn mandatory_bytes(
    cfg: &DFlash2Config,
    target: &ModelConfig,
    tp: usize,
    capacity: usize,
    weights: usize,
) -> Result<usize> {
    let rows = sum(&[capacity, cfg.block_size])?;
    let h = cfg.model.hidden_size;
    let b = cfg.block_size;
    // KvPool owns separate K/V buffers; descriptors and page bitmap are host-only.
    // Page-table device storage is in the weight arena. Draft KV is ALWAYS BF16.
    sum(&[
        weights,
        product(&[
            rows.div_ceil(16),
            16,
            DFLASH2_NUM_LAYERS,
            cfg.model.num_key_value_heads / tp,
            cfg.model.head_dim,
            4,
        ])?,
        32 * 1024 * 1024,
        DFLASH2_FORWARD_GRAPH_ARENA_BYTES,
        product(&[b, h, 2])?,
        product(&[b - 1, cfg.selector_rank, 2])?,
        product(&[sum(&[product(&[3, b])?, 4])?, 4])?,
        product(&[b - 1, cfg.top_k, 8])?,
        product(&[b - 1, 4])?,
        // Preserve NativeProposer's larger target tap capture for tiered MoE;
        // the drafter forward and shifted staging remain capped at 256 rows.
        product(&[
            capacity.min(if target.uses_tiered_moe() { 2048 } else { 256 }),
            target.hidden_size,
            DFLASH2_NUM_LAYERS,
            2,
        ])?,
        product(&[capacity.min(256), target.hidden_size, DFLASH2_NUM_LAYERS, 2])?,
        product(&[target.hidden_size, target.hc_count.max(1), 2])?,
        product(&[b, target.vocab_size / tp, 2])?,
    ])
}

pub(crate) struct ServingReserve {
    pub resident_bytes: usize,
    pub min_state_slots: usize,
}

// Three private request states, M-1 verification extras, and one locked radix
// checkpoint. The preferred 24-slot serving pool is not an execution minimum.
fn minimum_serving_state_slots(block_size: usize) -> Result<usize> {
    block_size
        .checked_add(3)
        .ok_or_else(|| loader_err("DFlash state-slot budget overflow"))
}

pub(crate) fn serving_reserve(
    dir: &Path,
    target: &ModelConfig,
    tp: usize,
    capacity: usize,
) -> Result<ServingReserve> {
    let cfg = DFlash2Config::load(dir)?;
    if cfg.model.hidden_size != target.hidden_size
        || cfg.model.vocab_size != target.vocab_size
        || cfg
            .target_layers
            .iter()
            .any(|&l| l >= target.num_hidden_layers)
    {
        return Err(loader_err("DFlash target hidden/vocabulary/tap mismatch"));
    }
    let a = SafetensorsArchive::open(dir)
        .map_err(|e| loader_err_ctx("open DFlash weights", e.to_string()))?;
    let weights = weight_arena_bytes(&cfg, &a, tp, capacity)?;
    Ok(ServingReserve {
        resident_bytes: mandatory_bytes(&cfg, target, tp, capacity, weights)?,
        min_state_slots: minimum_serving_state_slots(cfg.block_size)?,
    })
}

struct DFlashGraphIo {
    forward_graph: Option<CudaGraph>,
    select_greedy_graph: Option<CudaGraph>,
    forward_arena: DeviceArena,
    meta_dev: DeviceBuffer,
    topk_ids_dev: DeviceBuffer,
    topk_scores_dev: DeviceBuffer,
    out_path_dev: DeviceBuffer,
}

pub struct DFlash2Rank {
    pub cfg: DFlash2Config,
    pub use_cuda_graph: bool,
    rank: usize,
    tp: usize,
    target_embedding: u64,
    graph_io: DFlashGraphIo,
    _weights: DeviceArena,
    scratch: DeviceArena,
    kv: KvPool,
    pages: Vec<i32>,
    pages_dev: u64,
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
        if !matches!(tp, 1 | 2 | 4)
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
            || capacity > DFLASH2_MAX_KV_CAPACITY
            || max_rows == 0
            || max_rows > DFLASH2_MAX_BATCH_ROWS
            || capacity < max_rows
        {
            return Err(loader_err("DFlash target/rank/capacity mismatch"));
        }
        t.device.make_current()?;
        let archive = SafetensorsArchive::open(dir)
            .map_err(|e| loader_err_ctx("open DFlash weights", e.to_string()))?;
        let total_weight_bytes = weight_arena_bytes(&cfg, &archive, tp, capacity)?;
        mandatory_bytes(&cfg, &t.cfg, tp, capacity, total_weight_bytes)?;
        let (q, kv, i) = (
            c.num_attention_heads * d / tp,
            c.num_key_value_heads * d / tp,
            c.intermediate_size / tp,
        );
        let coeff = 2 * cfg.taps * (h / cfg.group);
        let rope_rows = capacity + cfg.block_size;
        let mut weights = DeviceArena::new(t.device.device_id, total_weight_bytes)?;
        fn read(a: &SafetensorsArchive, name: &str, shape: &[usize]) -> Result<Vec<u16>> {
            let info = a
                .tensor_info(name)
                .map_err(|e| loader_err_ctx("missing DFlash tensor", e.to_string()))?;
            if info.shape != shape || !matches!(info.dtype.as_str(), "BF16" | "F16" | "F32") {
                return Err(loader_err_ctx(
                    "DFlash tensor shape/dtype",
                    format!(
                        "tensor '{name}' has shape {:?} dtype {} (expected {shape:?} BF16/F16/F32)",
                        info.shape, info.dtype
                    ),
                ));
            }
            a.read_bf16_bits(name)
                .map_err(|e| loader_err_ctx("read DFlash BF16 tensor", e.to_string()))
        }
        enum HostSlice {
            DenseBf16(Vec<u16>),
            U4B8 {
                packed: Vec<u32>,
                scales: Vec<u16>,
                eff_gs: usize,
                group_blocks: i32,
                hadamard_block: usize,
            },
        }
        let read_slice = |name: &str,
                          n: usize,
                          k: usize,
                          rs: usize,
                          rn: usize,
                          cs: usize,
                          cn: usize|
         -> Result<HostSlice> {
            match c
                .quant
                .for_linear(name)
                .map_err(|e| loader_err_ctx("DFlash linear quant rule", e.to_string()))?
            {
                QuantScheme::Unquantized => {
                    let full = read(&archive, &format!("{name}.weight"), &[n, k])?;
                    let values: Vec<u16> = full[rs * k..(rs + rn) * k]
                        .chunks_exact(k)
                        .flat_map(|row| row[cs..cs + cn].iter().copied())
                        .collect();
                    Ok(HostSlice::DenseBf16(values))
                }
                QuantScheme::CompressedTensors {
                    num_bits: 4,
                    group_size,
                    symmetric: true,
                    ..
                } => {
                    if archive
                        .resolve_name(&format!("{name}.weight_zero_point"))
                        .is_some()
                    {
                        return Err(loader_err_ctx(
                            "DFlash quantized tensor",
                            format!("asymmetric weight_zero_point is not supported on '{name}'"),
                        ));
                    }
                    let p_key = format!("{name}.weight_packed");
                    let s_key = format!("{name}.weight_scale");
                    let p_info = archive.tensor_info(&p_key).map_err(|e| {
                        loader_err_ctx("missing DFlash weight_packed", e.to_string())
                    })?;
                    let s_info = archive.tensor_info(&s_key).map_err(|e| {
                        loader_err_ctx("missing DFlash weight_scale", e.to_string())
                    })?;
                    if !matches!(p_info.dtype.as_str(), "I32" | "U32")
                        || p_info.shape.as_slice() != [n, k / U4_NIBBLES_PER_U32]
                        || !matches!(s_info.dtype.as_str(), "BF16" | "F16" | "F32")
                        || s_info.shape.len() != 2
                        || s_info.shape[0] != n
                    {
                        return Err(loader_err_ctx(
                            "DFlash quantized tensor shape/dtype",
                            format!(
                                "'{name}' packed {:?} {} / scale {:?} {} (expected [{n}, {}] I32 and [{n}, G] BF16)",
                                p_info.shape,
                                p_info.dtype,
                                s_info.shape,
                                s_info.dtype,
                                k / U4_NIBBLES_PER_U32
                            ),
                        ));
                    }
                    let num_groups = s_info.shape[1];
                    if num_groups == 0 || !k.is_multiple_of(num_groups) {
                        return Err(loader_err_ctx(
                            "DFlash quantized scale groups",
                            format!(
                                "invalid scale shape {:?} for K={k} on '{name}'",
                                s_info.shape
                            ),
                        ));
                    }
                    let eff_gs = k / num_groups;
                    let channelwise = num_groups == 1 && (group_size == 0 || k > MARLIN_TILE_K_W4);
                    let group_blocks = if channelwise {
                        if cs != 0 || cn != k {
                            return Err(loader_err_ctx(
                                "DFlash channelwise shard",
                                format!("cannot K-shard channelwise weight '{name}'"),
                            ));
                        }
                        MARLIN_CHANNELWISE_GROUP_BLOCKS
                    } else {
                        if !matches!(eff_gs, 32 | 64 | 128)
                            || (group_size != 0 && eff_gs != group_size)
                            || !cs.is_multiple_of(eff_gs)
                            || !cn.is_multiple_of(eff_gs)
                        {
                            return Err(loader_err_ctx(
                                "DFlash group size",
                                format!("unsupported group_size={eff_gs} on '{name}'"),
                            ));
                        }
                        (eff_gs / MARLIN_GROUP_BLOCK_K) as i32
                    };
                    let hadamard_block = match c
                        .quant
                        .input_hadamard(name)
                        .map_err(|e| loader_err_ctx("DFlash input Hadamard rule", e.to_string()))?
                    {
                        Some(block_size) => {
                            if !SUPPORTED_HADAMARD_BLOCKS.contains(&block_size)
                                || !cs.is_multiple_of(block_size)
                                || !cn.is_multiple_of(block_size)
                            {
                                return Err(loader_err_ctx(
                                    "DFlash Hadamard block",
                                    format!(
                                        "invalid Hadamard block_size={block_size} for K-slice [{cs}..{}] on '{name}'",
                                        cs + cn
                                    ),
                                ));
                            }
                            block_size
                        }
                        None => 0,
                    };
                    if !rn.is_multiple_of(MARLIN_TILE_N)
                        || !cn.is_multiple_of(32)
                        || !cs.is_multiple_of(U4_NIBBLES_PER_U32)
                    {
                        return Err(loader_err_ctx(
                            "DFlash Marlin alignment",
                            format!("unaligned slice [{rn}, {cn}] at col {cs} on '{name}'"),
                        ));
                    }
                    let full_p = archive
                        .read_u32(&p_key)
                        .map_err(|e| loader_err_ctx("read DFlash weight_packed", e.to_string()))?;
                    let full_s = archive
                        .read_bf16_bits(&s_key)
                        .map_err(|e| loader_err_ctx("read DFlash weight_scale", e.to_string()))?;
                    let words_k = k / U4_NIBBLES_PER_U32;
                    let ws = cs / U4_NIBBLES_PER_U32;
                    let wn = cn / U4_NIBBLES_PER_U32;
                    let packed: Vec<u32> = full_p[rs * words_k..(rs + rn) * words_k]
                        .chunks_exact(words_k)
                        .flat_map(|row| row[ws..ws + wn].iter().copied())
                        .collect();
                    let gs = cs / eff_gs;
                    let gn = cn / eff_gs;
                    let scales: Vec<u16> = full_s[rs * num_groups..(rs + rn) * num_groups]
                        .chunks_exact(num_groups)
                        .flat_map(|row| row[gs..gs + gn].iter().copied())
                        .collect();
                    Ok(HostSlice::U4B8 {
                        packed,
                        scales,
                        eff_gs,
                        group_blocks,
                        hadamard_block,
                    })
                }
                _ => Err(loader_err("unsupported DFlash linear quantization")),
            }
        };
        let upload_slice_linear =
            |a: &mut DeviceArena, slice: HostSlice, rn: usize, cn: usize| -> Result<Linear> {
                match slice {
                    HostSlice::DenseBf16(values) => Ok(Linear::DenseBf16 {
                        weight: a.upload_slice(&values, &t.stream)?,
                        n: rn,
                        k: cn,
                    }),
                    HostSlice::U4B8 {
                        packed,
                        scales,
                        eff_gs,
                        group_blocks,
                        hadamard_block,
                    } => {
                        let tmp = DeviceBuffer::alloc(t.device.device_id, packed.len() * 4)?;
                        tmp.copy_from_host(&packed, &t.stream)?;
                        let weight = a.alloc_aligned(packed.len() * 4)?;
                        let ex = t.stream.exec(0, 0);
                        check_rc(
                            // SAFETY: tmp and weight are valid device allocations of packed.len() u32 words on t.stream.
                            unsafe {
                                ffi::w34_marlin_repack_int4(
                                    &ex,
                                    tmp.addr() as *const u32,
                                    weight as *mut u32,
                                    cn as i64,
                                    rn as i64,
                                )
                            },
                            "DFlash marlin_repack_int4",
                        )?;
                        t.stream.synchronize()?;
                        let perm_scales = marlin_permute_scales_from_n_major(
                            &scales,
                            cn,
                            rn,
                            if group_blocks == -1 { 0 } else { eff_gs },
                        );
                        let scales_ptr = a.upload_slice(&perm_scales, &t.stream)?;
                        Ok(Linear::MarlinBf16U4B8 {
                            weight,
                            scales: scales_ptr,
                            n: rn,
                            k: cn,
                            group_blocks,
                            hadamard_block,
                        })
                    }
                }
            };
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
            let slice = read_slice(name, n, k, rs, rn, cs, cn)?;
            upload_slice_linear(a, slice, rn, cn)
        };
        let concat_n_matrix =
            |a: &mut DeviceArena, parts: &[(&str, usize)], k: usize| -> Result<Linear> {
                let mut total_rn = 0usize;
                let mut acc: Option<HostSlice> = None;
                for &(name, local) in parts {
                    let part = read_slice(name, local * tp, k, r * local, local, 0, k)?;
                    total_rn += local;
                    match (&mut acc, part) {
                        (None, p) => acc = Some(p),
                        (Some(HostSlice::DenseBf16(dst)), HostSlice::DenseBf16(src)) => {
                            dst.extend_from_slice(&src);
                        }
                        (
                            Some(HostSlice::U4B8 {
                                packed: dp,
                                scales: ds,
                                eff_gs: dg,
                                group_blocks: dgb,
                                hadamard_block: dhb,
                            }),
                            HostSlice::U4B8 {
                                packed: sp,
                                scales: ss,
                                eff_gs: sg,
                                group_blocks: sgb,
                                hadamard_block: shb,
                            },
                        ) if *dg == sg && *dgb == sgb && *dhb == shb => {
                            dp.extend_from_slice(&sp);
                            ds.extend_from_slice(&ss);
                        }
                        _ => {
                            return Err(loader_err(
                                "mismatched quantization schemes across fused DFlash projections",
                            ));
                        }
                    }
                }
                upload_slice_linear(a, acc.expect("non-empty parts"), total_rn, k)
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
        let mut layers = Vec::with_capacity(c.num_hidden_layers);
        for l in 0..c.num_hidden_layers {
            let prefix = format!("layers.{l}");
            let norm = |arena: &mut DeviceArena, name: &str, size: usize| {
                upload(arena, &format!("{prefix}.{name}.weight"), &[size])
            };
            let q_name = format!("{prefix}.self_attn.q_proj");
            let k_name = format!("{prefix}.self_attn.k_proj");
            let v_name = format!("{prefix}.self_attn.v_proj");
            let qkv = concat_n_matrix(
                &mut weights,
                &[(&q_name, q), (&k_name, kv), (&v_name, kv)],
                h,
            )?;
            let kv_proj = match qkv {
                Linear::DenseBf16 { weight, .. } => Linear::DenseBf16 {
                    weight: weight + (q * h * 2) as u64,
                    n: 2 * kv,
                    k: h,
                },
                Linear::MarlinBf16U4B8 { .. } => {
                    concat_n_matrix(&mut weights, &[(&k_name, kv), (&v_name, kv)], h)?
                }
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
            let gate_name = format!("{prefix}.mlp.gate_proj");
            let up_name = format!("{prefix}.mlp.up_proj");
            let gate_up = concat_n_matrix(&mut weights, &[(&gate_name, i), (&up_name, i)], h)?;
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
                kv: kv_proj,
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
        let layer_ids: Vec<usize> = (0..c.num_hidden_layers).collect();
        let mut kv = KvPool::new(
            t.device.device_id,
            ffi::W34_KV_NAMESPACE_DFLASH_DRAFT_CTX,
            &layer_ids,
            c.num_key_value_heads / tp,
            d,
            DFLASH2_KV_PAGE_BLOCK_TOKENS,
            rope_rows.div_ceil(DFLASH2_KV_PAGE_BLOCK_TOKENS),
            KvFormatPolicy::Bf16,
        )?;
        let pages: Vec<i32> = (0..kv.num_pages)
            .map(|_| kv.alloc_page().unwrap())
            .collect();
        let pages_dev = weights.upload_slice(&pages, &t.stream)?;
        let hidden = DeviceBuffer::alloc(t.device.device_id, cfg.block_size * h * 2)?;
        let projected = DeviceBuffer::alloc(
            t.device.device_id,
            (cfg.block_size - 1) * cfg.selector_rank * 2,
        )?;
        let k_draft = cfg.block_size - 1;
        let topk_elems = k_draft * cfg.top_k;
        let graph_io = DFlashGraphIo {
            forward_graph: None,
            select_greedy_graph: None,
            forward_arena: DeviceArena::new(t.device.device_id, DFLASH2_FORWARD_GRAPH_ARENA_BYTES)?,
            meta_dev: DeviceBuffer::alloc(t.device.device_id, (3 * cfg.block_size + 4) * 4)?,
            topk_ids_dev: DeviceBuffer::alloc(t.device.device_id, topk_elems * 4)?,
            topk_scores_dev: DeviceBuffer::alloc(t.device.device_id, topk_elems * 4)?,
            out_path_dev: DeviceBuffer::alloc(t.device.device_id, k_draft * 4)?,
        };
        t.stream.synchronize()?;
        Ok(Self {
            cfg,
            use_cuda_graph: true,
            rank: r,
            tp,
            target_embedding: t.embed_tokens_ptr,
            graph_io,
            _weights: weights,
            scratch: DeviceArena::new(t.device.device_id, 32 * 1024 * 1024)?,
            kv,
            pages,
            pages_dev,
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
        })
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
        let mut meta = Vec::with_capacity(2 * rows + 4);
        meta.extend((before..after).map(|v| v as i32));
        meta.extend((before..after).map(|p| self.pages[p / 16] * 16 + (p % 16) as i32));
        meta.extend_from_slice(&[0i32, rows as i32, before as i32, after as i32]);
        let base = self.scratch.upload_slice(&meta, &t.stream)?;
        let pos = base;
        let slots = base + (rows * 4) as u64;
        let q_start = slots + (rows * 4) as u64;
        let ctx_before = q_start + 8;
        let seq_after = ctx_before + 4;
        let batch = W34AttentionBatch {
            query_start_loc: q_start as *const i32,
            context_lens_before: ctx_before as *const i32,
            seq_lens_after: seq_after as *const i32,
            block_tables: self.pages_dev as *const i32,
            write_slot_mapping: slots as *const i32,
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
            // SAFETY: device pointers x, residual, w, and y are valid allocations for rows * hidden_size BF16 elements on t.stream.
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
    #[allow(clippy::too_many_arguments)]
    fn rope_kv_write(
        &self,
        t: &Qwen3_5ModelRank,
        qkv: u64,
        q_out: u64,
        pos: u64,
        qheads: usize,
        layer: &Layer,
        kv_layout: &ffi::W34KvLayerLayout,
        write_slot_mapping: *const i32,
        rows: usize,
    ) -> Result<()> {
        check_rc(
            // SAFETY: device pointers and write_slot_mapping are valid allocations for rows BF16 QKV entries according to kv_layout.
            unsafe {
                ffi::w34_dflash_qk_norm_rope_kv_write_bf16_f32_table(
                    &t.stream.exec(0, 0),
                    qkv as *mut c_void,
                    q_out as *mut c_void,
                    pos as *const i32,
                    layer.q_norm as *const c_void,
                    layer.k_norm as *const c_void,
                    self.rope as *const c_void,
                    kv_layout,
                    write_slot_mapping,
                    rows as i32,
                    qheads as i32,
                    (self.cfg.model.num_key_value_heads / self.tp) as i32,
                    self.cfg.model.head_dim as i32,
                    self.rope_rows as i32,
                    self.cfg.model.rms_norm_eps,
                )
            },
            "DFlash QK norm/RoPE + KV write",
        )
    }
    /// `tap_bf16` is a live device span [rows, 5*H] of committed target taps in BF16.
    /// No unverified rows may be passed here. Enqueues context KV ingestion on `t.stream`
    /// without synchronizing the host thread so a subsequent `forward` on the same stream
    /// can pipeline back-to-back.
    pub fn ingest_no_sync(
        &mut self,
        t: &Qwen3_5ModelRank,
        tap_bf16: u64,
        rows: usize,
    ) -> Result<()> {
        self.validate_target(t)?;
        if rows == 0 {
            return Ok(());
        }
        if rows > self.max_rows
            || rows > self.capacity - self.committed
            || tap_bf16 == 0
            || !tap_bf16.is_multiple_of(2)
            || tap_bf16
                .checked_add(
                    (rows * self.cfg.target_layers.len() * self.cfg.model.hidden_size * 2) as u64,
                )
                .is_none()
        {
            return Err(loader_err("invalid DFlash ingest"));
        }
        self.proposal_anchor = None;
        self.scratch.reset();
        let h = self.cfg.model.hidden_size;
        let kv = self.cfg.model.num_key_value_heads * self.cfg.model.head_dim / self.tp;
        let projected = self.scratch.alloc_aligned(rows * h * 2)?;
        let ctx = self.scratch.alloc_aligned(rows * h * 2)?;
        let had_scratch = self
            .scratch
            .alloc_aligned(rows * h * self.cfg.target_layers.len() * 2)?;
        let packed = self.scratch.alloc_aligned(rows * 2 * kv * 2)?;
        let (batch, pos) = self.metadata(t, rows)?;
        self.fc.run(t, tap_bf16, projected, rows, had_scratch)?;
        self.norm(t, projected, 0, self.hidden_norm, ctx, rows)?;
        for (l, layer) in self.layers.iter().enumerate() {
            layer.kv.run(t, ctx, packed, rows, had_scratch)?;
            self.rope_kv_write(
                t,
                packed,
                0,
                pos,
                0,
                layer,
                &self.kv.layers[l].layout,
                batch.write_slot_mapping,
                rows,
            )?;
        }
        self.committed += rows;
        Ok(())
    }

    /// `tap_bf16` is a live device span [rows, 5*H] of committed target taps in BF16.
    /// No unverified rows may be passed here. This advances committed context.
    pub fn ingest(&mut self, t: &Qwen3_5ModelRank, tap_bf16: u64, rows: usize) -> Result<()> {
        if rows == 0 {
            self.validate_target(t)?;
            return Ok(());
        }
        self.ingest_no_sync(t, tap_bf16, rows)?;
        t.stream.synchronize()
    }
    #[allow(clippy::too_many_arguments)]
    fn conv(
        &self,
        t: &Qwen3_5ModelRank,
        c: &Conv,
        x: u64,
        coeff: u64,
        y: u64,
        side: i32,
        had_scratch: u64,
    ) -> Result<()> {
        if side == 0 {
            c.projection
                .run(t, x, coeff, self.cfg.block_size, had_scratch)?;
        }
        check_rc(
            // SAFETY: device pointers x, coeff, c.base, and y are valid allocations for block_size * hidden_size BF16 elements.
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
    fn fixed_forward_batch_ptrs(&self) -> (W34AttentionBatch, u64, u64) {
        let m = self.cfg.block_size;
        let base = self.graph_io.meta_dev.addr();
        let ids_ptr = base;
        let pos_ptr = ids_ptr + (m * 4) as u64;
        let slots_ptr = pos_ptr + (m * 4) as u64;
        let q_start_ptr = slots_ptr + (m * 4) as u64;
        let ctx_before_ptr = q_start_ptr + 8;
        let seq_after_ptr = ctx_before_ptr + 4;
        let batch = W34AttentionBatch {
            query_start_loc: q_start_ptr as *const i32,
            context_lens_before: ctx_before_ptr as *const i32,
            seq_lens_after: seq_after_ptr as *const i32,
            block_tables: self.pages_dev as *const i32,
            write_slot_mapping: slots_ptr as *const i32,
            valid_token_mask: core::ptr::null(),
            valid_req_mask: core::ptr::null(),
            num_reqs: 1,
            total_tokens: m as i32,
            max_blocks_per_req: self.pages.len() as i32,
            max_q_len: m as i32,
            max_seq_len: self.capacity as i32,
            sliding_window: self.cfg.window as i32,
            mask_mode: ffi::W34_ATTN_MASK_NONCAUSAL_WINDOW,
            sm_scale: 1.0 / (self.cfg.model.head_dim as f32).sqrt(),
            reserved: 0,
        };
        (batch, ids_ptr, pos_ptr)
    }

    fn upload_forward_meta(&self, t: &Qwen3_5ModelRank, anchor: i32) -> Result<()> {
        let m = self.cfg.block_size;
        let before = self.committed;
        let after = before + m;
        if after > self.capacity {
            return Err(loader_err("DFlash KV capacity exceeded"));
        }
        let mut meta = Vec::with_capacity(3 * m + 4);
        meta.push(anchor);
        meta.resize(m, self.cfg.mask_token);
        meta.extend((before..after).map(|v| v as i32));
        meta.extend((before..after).map(|p| self.pages[p / 16] * 16 + (p % 16) as i32));
        meta.extend_from_slice(&[0i32, m as i32, before as i32, after as i32]);
        self.graph_io.meta_dev.copy_from_host(&meta, &t.stream)
    }

    fn enqueue_forward_device_topk(
        &mut self,
        t: &Qwen3_5ModelRank,
        batch: &W34AttentionBatch,
        ids_ptr: u64,
        pos_ptr: u64,
        comm: Option<&Tp2Collectives>,
    ) -> Result<()> {
        self.graph_io.forward_arena.reset();
        let c = &self.cfg.model;
        let (h, m, d) = (c.hidden_size, self.cfg.block_size, c.head_dim);
        let (q, kv, i) = (
            c.num_attention_heads * d / self.tp,
            c.num_key_value_heads * d / self.tp,
            c.intermediate_size / self.tp,
        );
        let residual = self.graph_io.forward_arena.alloc_aligned(m * h * 2)?;
        let x = self.graph_io.forward_arena.alloc_aligned(m * h * 2)?;
        let normed = self.graph_io.forward_arena.alloc_aligned(m * h * 2)?;
        let filtered = self.graph_io.forward_arena.alloc_aligned(m * h * 2)?;
        let coeff = self
            .graph_io
            .forward_arena
            .alloc_aligned(m * 2 * self.cfg.taps * (h / self.cfg.group) * 2)?;
        let packed = self
            .graph_io
            .forward_arena
            .alloc_aligned(m * (q + 2 * kv) * 2)?;
        let qb = self.graph_io.forward_arena.alloc_aligned(m * q * 2)?;
        let attn = self.graph_io.forward_arena.alloc_aligned(m * q * 2)?;
        let tmp = self.graph_io.forward_arena.alloc_aligned(m * h * 2)?;
        let gu = self.graph_io.forward_arena.alloc_aligned(m * 2 * i * 2)?;
        let act = self.graph_io.forward_arena.alloc_aligned(m * i * 2)?;
        let had_scratch = self
            .graph_io
            .forward_arena
            .alloc_aligned(m * h.max(q).max(i) * 2)?;
        let ex = t.stream.exec(0, 0);
        check_rc(
            // SAFETY: ids_ptr contains m valid token IDs, embed_tokens_ptr is valid device memory of vocab_size * h half elements, residual has capacity for m * h BF16 elements.
            unsafe {
                ffi::w34_embed_lookup_f16_to_bf16(
                    &ex,
                    ids_ptr as *const i32,
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
            self.conv(t, &layer.attn_conv, normed, coeff, filtered, 0, had_scratch)?;
            layer.qkv.run(t, filtered, packed, m, had_scratch)?;
            self.rope_kv_write(
                t,
                packed,
                qb,
                pos_ptr,
                q / d,
                layer,
                &self.kv.layers[l].layout,
                batch.write_slot_mapping,
                m,
            )?;
            check_rc(
                // SAFETY: kv layout and attention batch describe valid paged memory, qb and attn are valid device buffers for m * q elements.
                unsafe {
                    ffi::w34_paged_attention_bf16(
                        &ex,
                        &self.kv.layers[l].layout,
                        batch,
                        qb as *const c_void,
                        attn as *mut c_void,
                        (q / d) as i32,
                    )
                },
                "DFlash paged attention",
            )?;
            layer.out.run(t, attn, tmp, m, had_scratch)?;
            if let Some(comm) = comm {
                comm.all_reduce_bf16(&ex, tmp, tmp, self.rank as i32, (m * h) as i32)?;
            }
            self.conv(t, &layer.attn_conv, tmp, coeff, x, 1, had_scratch)?;
            self.norm(t, x, residual, layer.post_norm, normed, m)?;
            self.conv(t, &layer.mlp_conv, normed, coeff, filtered, 0, had_scratch)?;
            layer.gate_up.run(t, filtered, gu, m, had_scratch)?;
            check_rc(
                // SAFETY: gu is valid input buffer with m * 2 * i elements, act has capacity for m * i BF16 elements.
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
            layer.down.run(t, act, tmp, m, had_scratch)?;
            if let Some(comm) = comm {
                comm.all_reduce_bf16(&ex, tmp, tmp, self.rank as i32, (m * h) as i32)?;
            }
            self.conv(t, &layer.mlp_conv, tmp, coeff, x, 1, had_scratch)?;
        }
        self.norm(t, x, residual, self.final_norm, self.hidden.addr(), m)?;
        let draft_hidden = self.hidden.addr() + (h * 2) as u64;
        self.hp
            .run(t, draft_hidden, self.projected.addr(), m - 1, had_scratch)?;
        let head_input = self.graph_io.forward_arena.alloc_aligned((m - 1) * h * 2)?;
        let logits = self
            .graph_io
            .forward_arena
            .alloc_aligned((m - 1) * t.local_vocab_size * 2)?;
        // Explicit dtype boundary into the target's single resident FP16 head.
        check_rc(
            // SAFETY: draft_hidden contains (m - 1) * h BF16 elements, head_input has capacity for (m - 1) * h FP16 elements.
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
        let topk_elems = (m - 1) * self.cfg.top_k;
        let num_splits = (256 / self.cfg.top_k).clamp(1, 16);
        let split_ids_ws = self
            .graph_io
            .forward_arena
            .alloc_aligned(topk_elems * num_splits * 4)?;
        let split_vals_ws = self
            .graph_io
            .forward_arena
            .alloc_aligned(topk_elems * num_splits * 4)?;
        check_rc(
            // SAFETY: logits contains valid FP16 rows, workspace and output device pointers are valid and aligned.
            unsafe {
                ffi::w34_topk_rows_split_f16(
                    &ex,
                    logits as *const c_void,
                    split_ids_ws as *mut i32,
                    split_vals_ws as *mut f32,
                    self.graph_io.topk_ids_dev.addr() as *mut i32,
                    self.graph_io.topk_scores_dev.addr() as *mut f32,
                    (m - 1) as i32,
                    t.local_vocab_size as i32,
                    t.vocab_offset as i32,
                    self.cfg.top_k as i32,
                    num_splits as i32,
                )
            },
            "DFlash top-k",
        )
    }

    pub fn has_forward_graph(&self) -> bool {
        self.graph_io.forward_graph.is_some()
    }

    fn ensure_forward_graph(
        &mut self,
        t: &Qwen3_5ModelRank,
        comm: Option<&Tp2Collectives>,
    ) -> Result<()> {
        if self.graph_io.forward_graph.is_some() {
            return Ok(());
        }
        let (batch, ids_ptr, pos_ptr) = self.fixed_forward_batch_ptrs();
        t.stream.synchronize()?;
        let graph = CudaGraph::capture(&t.stream, || {
            self.enqueue_forward_device_topk(t, &batch, ids_ptr, pos_ptr, comm)
        })?;
        self.graph_io.forward_graph = Some(graph);
        Ok(())
    }

    pub fn can_capture_select_greedy(&self) -> bool {
        let (k, n) = (self.cfg.block_size - 1, self.cfg.top_k);
        self.cfg.selector_rank <= 256 && n <= 32 && k <= 16
    }

    pub fn ensure_select_greedy_graph(
        &mut self,
        t: &Qwen3_5ModelRank,
        comm: Option<&Tp2Collectives>,
    ) -> Result<()> {
        if self.graph_io.select_greedy_graph.is_some() {
            return Ok(());
        }
        let (batch, ids_ptr, pos_ptr) = self.fixed_forward_batch_ptrs();
        let (k, n) = (self.cfg.block_size - 1, self.cfg.top_k);
        let topk_ids = self.graph_io.topk_ids_dev.addr();
        let topk_scores = self.graph_io.topk_scores_dev.addr();
        let out_path = self.graph_io.out_path_dev.addr();
        let projected = self.projected.addr();
        let pred = self.pred;
        let succ = self.succ;
        let selector_rank = self.cfg.selector_rank as i32;
        let vocab_size = self.cfg.model.vocab_size as i32;
        let rank_i32 = self.rank as i32;
        t.stream.synchronize()?;
        let graph = CudaGraph::capture(&t.stream, || {
            self.enqueue_forward_device_topk(t, &batch, ids_ptr, pos_ptr, comm)?;
            let ex = t.stream.exec(0, 0);
            if let Some(comm) = comm {
                comm.all_gather_topk_f32_i32(
                    &ex,
                    topk_scores,
                    topk_ids,
                    topk_scores,
                    topk_ids,
                    rank_i32,
                    k as i32,
                    n as i32,
                )?;
            }
            check_rc(
                // SAFETY: device arrays topk_ids, topk_scores, projected, pred, succ, out_path, and ids_ptr are valid and sized for k proposals and n candidates.
                unsafe {
                    ffi::w34_dflash_select_greedy_dev_anchor_bf16(
                        &ex,
                        topk_ids as *const i32,
                        topk_scores as *const f32,
                        projected as *const c_void,
                        pred as *const c_void,
                        succ as *const c_void,
                        out_path as *mut i32,
                        ids_ptr as *const i32,
                        k as i32,
                        n as i32,
                        selector_rank,
                        vocab_size,
                    )
                },
                "DFlash greedy selector",
            )
        })?;
        self.graph_io.select_greedy_graph = Some(graph);
        Ok(())
    }

    pub fn prepare_select_greedy_graph_inputs_async(
        &mut self,
        t: &Qwen3_5ModelRank,
        anchor: i32,
    ) -> Result<()> {
        self.validate_target(t)?;
        if anchor < 0 || anchor as usize >= self.cfg.model.vocab_size {
            return Err(loader_err("invalid DFlash forward"));
        }
        self.proposal_anchor = None;
        self.upload_forward_meta(t, anchor)
    }

    pub fn replay_select_greedy_graph_async(&self, t: &Qwen3_5ModelRank) -> Result<()> {
        let graph = self
            .graph_io
            .select_greedy_graph
            .as_ref()
            .ok_or_else(|| loader_err("DFlash select_greedy graph not captured"))?;
        graph.launch(&t.stream)
    }

    pub fn collect_select_greedy_host(
        &mut self,
        t: &Qwen3_5ModelRank,
        anchor: i32,
    ) -> Result<Vec<i32>> {
        let k = self.cfg.block_size - 1;
        let mut path = vec![0i32; k];
        self.graph_io
            .out_path_dev
            .copy_to_host(&mut path, &t.stream)?;
        if path.first().is_some_and(|&id| id < 0) {
            return Err(loader_err("nonfinite DFlash edges"));
        }
        self.proposal_anchor = Some(anchor);
        Ok(path)
    }

    pub fn launch_forward_graph_async(&mut self, t: &Qwen3_5ModelRank, anchor: i32) -> Result<()> {
        self.validate_target(t)?;
        if anchor < 0 || anchor as usize >= self.cfg.model.vocab_size {
            return Err(loader_err("invalid DFlash forward"));
        }
        self.proposal_anchor = None;
        self.upload_forward_meta(t, anchor)?;
        let graph = self
            .graph_io
            .forward_graph
            .as_ref()
            .ok_or_else(|| loader_err("DFlash forward graph not captured"))?;
        graph.launch(&t.stream)
    }

    pub fn collect_forward_topk_host(
        &mut self,
        t: &Qwen3_5ModelRank,
        anchor: i32,
    ) -> Result<DraftCandidates> {
        let m = self.cfg.block_size;
        let topk_elems = (m - 1) * self.cfg.top_k;
        let mut out = DraftCandidates {
            ids: vec![0; topk_elems],
            scores: vec![0.0; topk_elems],
            rows: m - 1,
            top_k: self.cfg.top_k,
        };
        check_rc(
            // SAFETY: out.ids has capacity for topk_elems i32 elements, source is valid device memory, stream is active.
            unsafe {
                ffi::w34_memcpy_d2h_async(
                    out.ids.as_mut_ptr().cast(),
                    self.graph_io.topk_ids_dev.addr() as *const c_void,
                    topk_elems * 4,
                    t.stream.handle,
                )
            },
            "DFlash top-k ids copy",
        )?;
        check_rc(
            // SAFETY: out.scores has capacity for topk_elems f32 elements, source is valid device memory, stream is active.
            unsafe {
                ffi::w34_memcpy_d2h_async(
                    out.scores.as_mut_ptr().cast(),
                    self.graph_io.topk_scores_dev.addr() as *const c_void,
                    topk_elems * 4,
                    t.stream.handle,
                )
            },
            "DFlash top-k scores copy",
        )?;
        t.stream.synchronize()?;
        if out.scores.iter().any(|x| !x.is_finite()) {
            return Err(loader_err("nonfinite DFlash logits"));
        }
        self.proposal_anchor = Some(anchor);
        Ok(out)
    }

    fn forward_device_topk(
        &mut self,
        t: &Qwen3_5ModelRank,
        anchor: i32,
        comm: Option<&Tp2Collectives>,
    ) -> Result<(u64, u64)> {
        self.validate_target(t)?;
        if anchor < 0
            || anchor as usize >= self.cfg.model.vocab_size
            || (self.tp > 1) != comm.is_some()
        {
            return Err(loader_err("invalid DFlash forward"));
        }
        self.proposal_anchor = None;
        self.upload_forward_meta(t, anchor)?;
        if self.use_cuda_graph {
            self.ensure_forward_graph(t, comm)?;
            self.graph_io
                .forward_graph
                .as_ref()
                .unwrap()
                .launch(&t.stream)?;
        } else {
            let (batch, ids_ptr, pos_ptr) = self.fixed_forward_batch_ptrs();
            self.enqueue_forward_device_topk(t, &batch, ids_ptr, pos_ptr, comm)?;
        }
        Ok((
            self.graph_io.topk_ids_dev.addr(),
            self.graph_io.topk_scores_dev.addr(),
        ))
    }

    pub fn forward(
        &mut self,
        t: &Qwen3_5ModelRank,
        anchor: i32,
        comm: Option<&Tp2Collectives>,
    ) -> Result<DraftCandidates> {
        let _ = self.forward_device_topk(t, anchor, comm)?;
        self.collect_forward_topk_host(t, anchor)
    }

    /// Single-rank (`TP=1`) fused forward + greedy selection on device without
    /// intermediate `DraftCandidates` D2H/H2D round-trips.
    pub fn forward_select_greedy(&mut self, t: &Qwen3_5ModelRank, anchor: i32) -> Result<Vec<i32>> {
        if self.tp != 1 {
            return Err(loader_err("forward_select_greedy requires TP=1"));
        }
        self.forward_select_greedy_tp(t, anchor, None)
    }

    /// Fused forward + (optional TP top-k merge) + greedy selection on device without
    /// intermediate `DraftCandidates` D2H/H2D round-trips.
    pub fn forward_select_greedy_tp(
        &mut self,
        t: &Qwen3_5ModelRank,
        anchor: i32,
        comm: Option<&Tp2Collectives>,
    ) -> Result<Vec<i32>> {
        if (self.tp > 1) != comm.is_some() {
            return Err(loader_err("invalid DFlash forward"));
        }
        if self.use_cuda_graph && self.can_capture_select_greedy() {
            self.prepare_select_greedy_graph_inputs_async(t, anchor)?;
            self.ensure_select_greedy_graph(t, comm)?;
            self.replay_select_greedy_graph_async(t)?;
            return self.collect_select_greedy_host(t, anchor);
        }
        let (ids_dev, scores_dev) = self.forward_device_topk(t, anchor, comm)?;
        if let Some(comm) = comm {
            let (k, n) = (self.cfg.block_size - 1, self.cfg.top_k);
            comm.all_gather_topk_f32_i32(
                &t.stream.exec(0, 0),
                scores_dev,
                ids_dev,
                scores_dev,
                ids_dev,
                self.rank as i32,
                k as i32,
                n as i32,
            )?;
        }
        self.proposal_anchor = Some(anchor);
        self.select_greedy_from_device(t, anchor, ids_dev, scores_dev)
    }

    fn select_greedy_from_device(
        &mut self,
        t: &Qwen3_5ModelRank,
        anchor: i32,
        ids: u64,
        vals: u64,
    ) -> Result<Vec<i32>> {
        let (k, n) = (self.cfg.block_size - 1, self.cfg.top_k);
        if self.cfg.selector_rank <= 256 && n <= 32 && k <= 16 {
            let path_dev = self.scratch.alloc_aligned(k * 4)?;
            check_rc(
                // SAFETY: ids, vals, projected, pred, succ, and path_dev are valid device buffers on t.stream.
                unsafe {
                    ffi::w34_dflash_select_greedy_bf16(
                        &t.stream.exec(0, 0),
                        ids as *const i32,
                        vals as *const f32,
                        self.projected.addr() as *const c_void,
                        self.pred as *const c_void,
                        self.succ as *const c_void,
                        path_dev as *mut i32,
                        anchor,
                        k as i32,
                        n as i32,
                        self.cfg.selector_rank as i32,
                        self.cfg.model.vocab_size as i32,
                    )
                },
                "DFlash greedy selector",
            )?;
            let mut path = vec![0i32; k];
            check_rc(
                // SAFETY: path has capacity for k i32 elements, path_dev is valid device buffer of k * 4 bytes.
                unsafe {
                    ffi::w34_memcpy_d2h_async(
                        path.as_mut_ptr().cast(),
                        path_dev as *const c_void,
                        k * 4,
                        t.stream.handle,
                    )
                },
                "DFlash greedy path copy",
            )?;
            t.stream.synchronize()?;
            if path.first().is_some_and(|&id| id < 0) {
                return Err(loader_err("nonfinite DFlash edges"));
            }
            return Ok(path);
        }
        let edges = self.scratch.alloc_aligned(k * n * n * 4)?;
        check_rc(
            // SAFETY: ids, vals, projected, pred, succ, and edges are valid device allocations on t.stream.
            unsafe {
                ffi::w34_dflash_edges_bf16(
                    &t.stream.exec(0, 0),
                    ids as *const i32,
                    vals as *const f32,
                    self.projected.addr() as *const c_void,
                    self.pred as *const c_void,
                    self.succ as *const c_void,
                    edges as *mut f32,
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
        check_rc(
            // SAFETY: scores has capacity for k * n * n f32 elements, edges is valid device memory of k * n * n * 4 bytes.
            unsafe {
                ffi::w34_memcpy_d2h_async(
                    scores.as_mut_ptr().cast(),
                    edges as *const c_void,
                    k * n * n * 4,
                    t.stream.handle,
                )
            },
            "DFlash edges copy",
        )?;
        t.stream.synchronize()?;
        if scores.iter().any(|x| !x.is_finite()) {
            return Err(loader_err("nonfinite DFlash edges"));
        }
        let mut candidate_ids = vec![0i32; k * n];
        check_rc(
            // SAFETY: candidate_ids has capacity for k * n i32 elements, ids is valid device memory of k * n * 4 bytes.
            unsafe {
                ffi::w34_memcpy_d2h_async(
                    candidate_ids.as_mut_ptr().cast(),
                    ids as *const c_void,
                    k * n * 4,
                    t.stream.handle,
                )
            },
            "DFlash candidate ids copy",
        )?;
        t.stream.synchronize()?;
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
            path.push(candidate_ids[slot * n + winner]);
            predecessor = winner;
        }
        Ok(path)
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
        self.select_greedy_from_device(t, anchor, ids, vals)
    }
}

#[cfg(test)]
mod memory_tests {
    use super::*;
    use std::io::Write;

    #[test]
    fn state_floor_includes_locked_prefix_without_requiring_preferred_pool() {
        assert_eq!(minimum_serving_state_slots(8).unwrap(), 11);
        assert_eq!(minimum_serving_state_slots(16).unwrap(), 19);
        assert!(minimum_serving_state_slots(usize::MAX).is_err());
    }

    fn config() -> DFlash2Config {
        let model = ModelConfig::from_drafter_json(&serde_json::json!({
            "hidden_size": 256, "intermediate_size": 512,
            "num_attention_heads": 4, "num_key_value_heads": 4,
            "head_dim": 128, "num_hidden_layers": 5, "vocab_size": 1024
        }))
        .unwrap();
        DFlash2Config {
            model,
            block_size: 8,
            target_layers: vec![0, 1, 2, 3, 4],
            group: 8,
            taps: 4,
            selector_rank: 128,
            top_k: 8,
            mask_token: 0,
            window: 0,
        }
    }

    fn archive(
        tensors: Vec<(String, Vec<usize>, &'static str)>,
    ) -> (tempfile::TempDir, SafetensorsArchive) {
        let dir = tempfile::tempdir().unwrap();
        let mut header = serde_json::Map::new();
        let mut offset = 0usize;
        for (name, shape, dtype) in tensors {
            let bytes = shape.iter().product::<usize>() * if dtype == "I32" { 4 } else { 2 };
            header.insert(
                name,
                serde_json::json!({"shape": shape, "dtype": dtype,
                "data_offsets": [offset, offset + bytes]}),
            );
            offset += bytes;
        }
        let header = serde_json::to_vec(&header).unwrap();
        let mut file = std::fs::File::create(dir.path().join("model.safetensors")).unwrap();
        file.write_all(&(header.len() as u64).to_le_bytes())
            .unwrap();
        file.write_all(&header).unwrap();
        file.set_len((8 + header.len() + offset) as u64).unwrap(); // sparse, no weight payload reads
        let a = SafetensorsArchive::open(dir.path()).unwrap();
        (dir, a)
    }

    fn dense_archive(c: &DFlash2Config) -> (tempfile::TempDir, SafetensorsArchive) {
        let h = c.model.hidden_size;
        let q = c.model.num_attention_heads * c.model.head_dim;
        let kv = c.model.num_key_value_heads * c.model.head_dim;
        let i = c.model.intermediate_size;
        let mut tensors = vec![];
        let mut add = |name: String, shape: Vec<usize>| tensors.push((name, shape, "BF16"));
        for name in ["hidden_norm.weight", "norm.weight"] {
            add(name.into(), vec![h]);
        }
        for name in ["predecessor_codebook", "successor_codebook"] {
            add(
                format!("candidate_selector.{name}"),
                vec![c.model.vocab_size, c.selector_rank],
            );
        }
        add("fc.weight".into(), vec![h, h * 5]);
        add(
            "candidate_selector.hidden_projection.weight".into(),
            vec![c.selector_rank, h],
        );
        for l in 0..5 {
            for (name, n) in [
                ("input_layernorm", h),
                ("post_attention_layernorm", h),
                ("self_attn.q_norm", 128),
                ("self_attn.k_norm", 128),
            ] {
                add(format!("layers.{l}.{name}.weight"), vec![n]);
            }
            for conv in ["attention_conv", "mlp_conv"] {
                add(format!("layers.{l}.{conv}.base_kernel"), vec![2, c.taps, h]);
                add(
                    format!("layers.{l}.{conv}.kernel_projection.weight"),
                    vec![2 * c.taps * h / c.group, h],
                );
            }
            for (name, n, k) in [
                ("self_attn.q_proj", q, h),
                ("self_attn.k_proj", kv, h),
                ("self_attn.v_proj", kv, h),
                ("self_attn.o_proj", h, q),
                ("mlp.gate_proj", i, h),
                ("mlp.up_proj", i, h),
                ("mlp.down_proj", h, i),
            ] {
                add(format!("layers.{l}.{name}.weight"), vec![n, k]);
            }
        }
        if c.model.quant != QuantScheme::Unquantized {
            tensors = tensors
                .into_iter()
                .flat_map(|(name, shape, dtype)| {
                    if shape.len() == 2 && name.ends_with(".weight") {
                        let base = name.strip_suffix(".weight").unwrap();
                        vec![
                            (
                                format!("{base}.weight_packed"),
                                vec![shape[0], shape[1] / 8],
                                "I32",
                            ),
                            (
                                format!("{base}.weight_scale"),
                                vec![shape[0], shape[1] / 64],
                                "BF16",
                            ),
                        ]
                    } else {
                        vec![(name, shape, dtype)]
                    }
                })
                .collect();
        }
        archive(tensors)
    }

    #[test]
    fn dflash_dense_shared_arena_tp_and_suffix() {
        let mut c = config();
        let (_dir, a) = dense_archive(&c);
        for tp in [1, 2, 4] {
            for block in [8, 16] {
                c.block_size = block;
                let weights = weight_arena_bytes(&c, &a, tp, 262144).unwrap();
                let short = weight_arena_bytes(&c, &a, tp, 256).unwrap();
                assert_eq!(
                    weights - short,
                    (262144 - 256) * 128 * 4 + (262144 - 256) / 16 * 4
                );
                assert!(
                    mandatory_bytes(&c, &c.model, tp, 262144, weights).unwrap()
                        > weights + 64 * 1024 * 1024
                );
                assert!(weight_arena_bytes(&c, &a, tp, 262145).is_err());
                assert!(weight_arena_bytes(&c, &a, tp, 0).is_err());
            }
        }
        // Only attention and MLP shard: codebooks, FC, convs and RoPE replicate.
        let w1 = weight_arena_bytes(&c, &a, 1, 256).unwrap();
        let w2 = weight_arena_bytes(&c, &a, 2, 256).unwrap();
        let w4 = weight_arena_bytes(&c, &a, 4, 256).unwrap();
        assert_eq!(w1 - w2, 2 * (w2 - w4));
        assert!(w2 > w1 / 2);
    }

    #[test]
    fn dflash_mandatory_native_kv_graphs_and_capped_io() {
        let c = config();
        let base = mandatory_bytes(&c, &c.model, 1, 256, 0).unwrap();
        let next = mandatory_bytes(&c, &c.model, 1, 272, 0).unwrap();
        // Taps/shifted do not grow past 256. No target KV format argument:
        // target TQ4 cannot change the draft's native BF16 payload.
        assert_eq!(next - base, 16 * 5 * 4 * 128 * 4);
        let kv = (256usize + 8).div_ceil(16) * 16 * 5 * 4 * 128 * 4;
        let expected = kv
            + 64 * 1024 * 1024
            + 8 * 256 * 2
            + 7 * 128 * 2
            + (3 * 8 + 4) * 4
            + 7 * 8 * 8
            + 7 * 4
            + 256 * 256 * 5 * 2 * 2
            + c.model.hyper_hidden_size() * 2
            + 8 * 1024 * 2;
        assert_eq!(base, expected); // no MTP batch-slot buffers or target GDN states
        assert!(product(&[usize::MAX, 2]).is_err());
        assert!(sum(&[usize::MAX, 1]).is_err());
        assert!(mandatory_bytes(&c, &c.model, 1, usize::MAX, 0).is_err());
    }

    #[test]
    fn dflash_quantized_metadata_errors_are_not_zero_reserves() {
        let mut c = config();
        c.model.quant = ModelConfig::parse_quant(Some(&serde_json::json!({
            "quant_method": "compressed-tensors", "format": "pack-quantized",
            "config_groups": {"group_0": {"targets": ["Linear"], "weights": {
                "num_bits": 4, "type": "int", "symmetric": true, "strategy": "group", "group_size": 64
            }}}
        }))).unwrap();
        let (_full_dir, full) = dense_archive(&c);
        for tp in [1, 2, 4] {
            for block in [8, 16] {
                c.block_size = block;
                let linear = |n: usize, k: usize| {
                    n * k / 2 + n * (k / 64) * 2 + 2 * DEVICE_SCRATCH_ALIGNMENT_BYTES
                };
                let expected = DFLASH2_WEIGHT_ARENA_SLACK_BYTES
                    + (256 + block) * 128 * 4
                    + (256 + block).div_ceil(16) * 4
                    + 2 * 256 * 2
                    + 2 * 1024 * 128 * 2
                    + linear(256, 256 * 5)
                    + linear(128, 256)
                    + 5 * ((2 * 256 + 2 * 128) * 2
                        + 4 * 4 * 256 * 2
                        + 2 * linear(256, 256)
                        + linear(512 / tp, 256)
                        + 4 * linear(512 / tp, 256)
                        + linear(256, 512 / tp)
                        + 2 * linear(512 / tp, 256)
                        + linear(256, 512 / tp));
                // Four KV projections: K/V in fused QKV plus separate quantized KV.
                assert_eq!(weight_arena_bytes(&c, &full, tp, 256).unwrap(), expected);
            }
        }
        let (_dir, a) = archive(vec![
            ("fc.weight_packed".into(), vec![128, 32], "I32"),
            ("fc.weight_scale".into(), vec![128, 4], "BF16"),
        ]);
        assert_eq!(
            linear_metadata(&c, &a, "fc", 128, 256, 128, 128).unwrap(),
            128 * 128 / 2 + 128 * 2 * 2 + 2 * DEVICE_SCRATCH_ALIGNMENT_BYTES
        );
        assert!(linear_metadata(&c, &a, "missing", 128, 256, 128, 256).is_err());
        let (_dir, bad) = archive(vec![
            ("fc.weight_packed".into(), vec![128, 32], "I32"),
            ("fc.weight_scale".into(), vec![128, 8], "BF16"),
        ]);
        assert!(linear_metadata(&c, &bad, "fc", 128, 256, 128, 128).is_err());
        let (_dir, channel) = archive(vec![
            ("fc.weight_packed".into(), vec![128, 32], "I32"),
            ("fc.weight_scale".into(), vec![128, 1], "BF16"),
        ]);
        assert!(linear_metadata(&c, &channel, "fc", 128, 256, 128, 128).is_err());
        c.model.quant = ModelConfig::parse_quant(Some(&serde_json::json!({
            "quant_method": "compressed-tensors", "format": "pack-quantized",
            "config_groups": {"group_0": {"targets": ["Linear"], "weights": {
                "num_bits": 4, "type": "int", "symmetric": true, "strategy": "group", "group_size": 64
            }}},
            "transform_config": {"config_groups": {"h64": {
                "type": "hadamard", "head_dim": 64, "randomize": false,
                "apply": [
                    {"location": "input", "inverse": false, "targets": ["re:.*"], "ignore": []},
                    {"location": "weight_input", "inverse": true, "targets": ["re:.*"], "ignore": []}
                ]
            }}}
        }))).unwrap();
        assert_eq!(c.model.quant.input_hadamard("fc").unwrap(), Some(64));
        assert!(linear_metadata(&c, &a, "fc", 128, 256, 128, 128).is_ok());
        assert!(linear_metadata(&c, &a, "fc", 128, 256, 128, 32).is_err());
        for tp in [1, 2, 4] {
            assert!(weight_arena_bytes(&c, &full, tp, 256).is_ok());
        }
    }
}

#[cfg(all(test, feature = "cuda"))]
#[path = "dflash_tests.rs"]
mod tests;
