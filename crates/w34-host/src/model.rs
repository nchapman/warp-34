//! Qwen 3.5 / 3.8 hybrid model backbone, graph replay, and GPU sampling.
//!
//! The implementation is divided by responsibility; public types remain available
//! through `crate::model` for callers of the original model API.

use crate::cache::{KvFormatPolicy, KvPool, StateBundlePool, StateTx};
use crate::cuda::{
    CapacityPlan, CapacityPlanner, CublasHandle, CudaDevice, CudaError, CudaGraph, CudaStream,
    DeviceArena, DeviceBuffer, Result, Tp2Collectives, check_rc, f16_to_f32,
    prepare_execution_plan,
};
use crate::loader::{
    LayerKind, ModelConfig, QuantScheme, SafetensorsArchive, dequant_u4_zp_n_major_to_f16,
    dequant_u4b8_n_major_to_f16, dequant_u8b128_n_major_to_f16, marlin_permute_scales_from_n_major,
    marlin_permute_zero_points_from_n_major, resolve_hf_model_dir,
    transpose_gptq_qweight_k4_n_to_n_major, transpose_gptq_qweight_k8_n_to_n_major,
    unpack_awq_qweight_k_n8_to_n_major, unpack_awq_qzeros_kg_n8_to_n_major,
    validate_static_gptq_g_idx,
};
use crate::radix::{MambaRadixCache, PlannedCheckpoint, ReqRadixState};
use crate::scheduler::{SchedulerConfig, SchedulerV1};
use core::ffi::c_void;
use std::collections::HashMap;
use std::path::Path;
use std::sync::Arc;
use tokenizers::Tokenizer;
use w34_kernels::ffi::{self, W34_ATTN_MASK_CAUSAL, W34AttentionBatch, W34Exec, W34GdnBatch};

mod batch;
mod decision;
mod diagnostics;
mod engine;
mod execution;
mod forward;
mod generation;
mod graph_io;
mod graphs;
mod rank;
mod sampling;
pub(crate) mod weights;

pub use engine::*;
pub use graph_io::*;
#[cfg(feature = "cuda")]
pub use rank::Qwen3_5ModelRank;
#[cfg(all(test, feature = "cuda"))]
use rank::clamp_state_slots;
#[cfg(feature = "cuda")]
use rank::{PoolCapacity, serving_state_slots};
pub(crate) use weights::RawLinearHost;
#[cfg(feature = "cuda")]
pub(crate) use weights::load_moe_layer_weights;
pub use weights::*;

#[cfg(any(feature = "cuda", test))]
#[path = "model_scratch.rs"]
mod model_scratch;

type LayerScratchAddrs = ([u64; 6], [u64; 5], [u64; 10], [u64; 3], [u64; 7], u64);

/// Reserve one high-water region; all offsets within a phase remain disjoint.
/// Main-stream ordering separates FA/GDN from FFN and subsequent layers. GDN's
/// ba auxiliary stream is joined before recurrent/out_proj and thus before reuse.
/// Dumps copy and synchronize in record_dump_f16, retaining no scratch pointers.
#[cfg(feature = "cuda")]
fn alloc_layer_scratch(
    arena: &mut DeviceArena,
    cfg: &ModelConfig,
    tp: usize,
    tokens: usize,
) -> Result<LayerScratchAddrs> {
    let (bytes, fa_offsets, gdn_offsets, ffn_offsets, ple_hc_offsets, qsa_offsets) =
        layer_scratch_layout(cfg, tp, tokens)?;
    let base = arena.alloc_aligned(bytes)?;
    let overlay_end = base
        .checked_add(bytes as u64)
        .ok_or_else(|| loader_err("layer scratch layout overflow"))?;
    Ok((
        fa_offsets.map(|o| base + o as u64),
        gdn_offsets.map(|o| base + o as u64),
        ffn_offsets.map(|o| o.map_or(0, |o| base + o as u64)),
        ple_hc_offsets.map(|o| base + o as u64),
        qsa_offsets.map(|o| base + o as u64),
        overlay_end,
    ))
}

#[cfg(feature = "cuda")]
fn layer_scratch_layout(
    cfg: &ModelConfig,
    tp: usize,
    tokens: usize,
) -> Result<model_scratch::LayerLayout> {
    if tp == 0 {
        return Err(loader_err("zero scratch TP size"));
    }
    let overflow = || loader_err("layer scratch layout overflow");
    let mul = |a: usize, b: usize| a.checked_mul(b).ok_or_else(overflow);
    let moe = if cfg.uses_tiered_moe() {
        cfg.moe_intermediate_size
    } else {
        cfg.moe_intermediate_size / tp
    };
    model_scratch::layer_layout(
        model_scratch::Dimensions {
            q: mul(cfg.num_attention_heads / tp, cfg.head_dim)?,
            kv: mul(cfg.num_key_value_heads / tp, cfg.head_dim)?,
            v_heads: cfg.linear_num_value_heads / tp,
            v: mul(cfg.linear_num_value_heads / tp, cfg.linear_value_head_dim)?,
            k: mul(cfg.linear_num_key_heads / tp, cfg.linear_key_head_dim)?,
            interm: cfg.intermediate_size / tp,
            experts: cfg.num_experts,
            top_k: cfg.num_experts_per_tok,
            moe,
            shared: cfg.shared_expert_intermediate_size / tp,
            hidden: cfg.hidden_size,
            hc_count: cfg.hc_count,
            uses_ple: cfg.uses_ple(),
            indexer_head_dim: cfg.indexer_head_dim,
        },
        tokens,
    )
    .ok_or_else(overflow)
}

#[inline]
pub(crate) fn loader_err(context: &'static str) -> CudaError {
    CudaError::new(-10, context)
}

#[inline]
pub(crate) fn loader_err_msg(detail: impl Into<String>) -> CudaError {
    CudaError::with_detail(-10, "model loader error", detail)
}

#[inline]
pub(crate) fn loader_err_ctx(context: &'static str, detail: impl Into<String>) -> CudaError {
    CudaError::with_detail(-10, context, detail)
}

/// Static tap plan specifying layer indices whose post-layer hidden states are captured.
#[derive(Clone, Debug, Default)]
pub struct TapPlan {
    pub layer_ids: Vec<usize>,
}

/// Same-input layer-0 MoE replay report against a captured native `vLLM 0.31.0` fixture.
#[derive(Clone, Debug, serde::Serialize)]
pub struct MoeLayer0GoldenReport {
    pub num_tokens: usize,
    pub hidden_size: usize,
    pub num_experts: usize,
    pub top_k: usize,
    pub limit: f64,
    pub routed_gemv_rel_l2: f64,
    pub routed_sparse_mma_rel_l2: f64,
    pub same_route_gemv_rel_l2: f64,
    pub same_route_sparse_mma_rel_l2: f64,
    pub full_ffn_rel_l2: f64,
    pub router_slot_matches: usize,
    pub router_total_slots: usize,
    pub passed: bool,
}

/// Named intermediate activations captured during a diagnostic `dump_forward` pass.
#[derive(Clone, Debug, Default)]
pub struct ForwardDump {
    pub tensors_f32: HashMap<String, (Vec<usize>, Vec<f32>)>,
    pub greedy_token_id: i32,
    pub top5_ids: [i32; 5],
    pub top5_logprobs: [f32; 5],
    pub draft_proposal_ids: Vec<i32>,
}

/// Strip stale HF tokenizer truncation/padding before KV-pool admission checks.
fn normalize_hf_tokenizer(mut tok: Tokenizer) -> Tokenizer {
    let _ = tok.with_truncation(None);
    tok.with_padding(None);
    tok
}

#[cfg(test)]
#[path = "model_quant_tests.rs"]
mod quant_tests;
#[cfg(test)]
#[path = "model/tests.rs"]
mod tests;
