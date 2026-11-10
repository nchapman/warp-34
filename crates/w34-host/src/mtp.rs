//! Native single-layer dense & MoE Qwen3.5 / Qwen3.6 MTP (`Qwen3_5MTP` + `Qwen3_5MoeMTP`),
//! using the target stream and shared vocabulary weights.
//!
//! Donor: vLLM qwen3_5_mtp.py and llm_base_proposer.py. At MTP position p,
//! the inputs are target FINAL-normalized `hidden[p]` and `token[p+1]`, NOT
//! `hidden[p-1]` and `token[p]`. Positions stay p when input IDs shift left.
//! Ingest prompt pairs up to p-1, then propose with `hidden[p]` and the target's
//! newly sampled anchor `token[p+1]`. The output predicts `token[p+2]`. Later
//! steps feed the previous MTP final-normalized hidden and sampled draft,
//! increasing position by one. No target GDN state is touched.
//!
//! This is a single-sequence eager component. All rows use FP16; FC is
//! replicated (algebraically the donor's column-shard + gather), attention
//! and dense/MoE FFN are TP-sharded. Vocabulary storage is borrowed, never copied.
//! TP2 callers must run both ranks concurrently with the same collective order.
use crate::cache::{KvFormatPolicy, KvPool};
use crate::cuda::{CudaGraph, DeviceArena, DeviceBuffer, Result, Tp2Collectives};
use crate::loader::{ModelConfig, SafetensorsArchive};
use crate::model::{
    FfnBlockWeights, FinalResidual, FullAttentionLayerWeights, LayerResidual, LinearProj,
    MARLIN_TILE_K_W4, MARLIN_TILE_N, Qwen3_5ModelRank, RawLinearHost, load_hc_block_weights,
    load_moe_layer_weights, load_qsa_indexer_weights, loader_err,
};
use std::path::Path;
use w34_kernels::ffi::{self, W34_KV_NAMESPACE_MTP_FULL_ATTN};

mod propose;

// Shared by pre-placement serving admission and the actual loader. The arena remains
// FP16-sized even for quantized checkpoints; do not budget only its used bytes.
fn weight_arena_bytes(c: &ModelConfig, tp: usize, capacity: usize, ffn: usize) -> usize {
    let h = c.hidden_size;
    let q = c.num_attention_heads * c.head_dim;
    let local_kv = c.local_kv_heads(tp) * c.head_dim;
    let hc = if c.uses_hyper_connection() {
        c.hc_count.max(1)
    } else {
        1
    };
    (2 * (2 * h * h
        + 2 * q * h / tp
        + 2 * local_kv * h
        + h * q / tp
        + (5 + hc) * h
        + 2 * c.head_dim))
        .saturating_add(ffn)
        .saturating_add(capacity.div_ceil(16).saturating_mul(4))
        .saturating_add(32 * 1024 * 1024)
}

fn repack_staging_bytes(c: &ModelConfig, tp: usize) -> usize {
    (64 * 1024 * 1024).max(
        2 * (c.intermediate_size / tp).div_ceil(MARLIN_TILE_N)
            * MARLIN_TILE_N
            * c.hidden_size.div_ceil(MARLIN_TILE_K_W4)
            * MARLIN_TILE_K_W4
            * c.quant.bits()
            / 8,
    )
}

fn scratch_row_elements(c: &ModelConfig, tp: usize, ffn: usize) -> usize {
    let hc = if c.uses_hyper_connection() {
        c.hc_count.max(1)
    } else {
        1
    };
    (9 + 4 * hc) * c.hidden_size
        + c.hc_lowrank.max(1)
        + 3 * hc
        + 8 * c.num_attention_heads * c.head_dim / tp
        + 6 * c.local_kv_heads(tp) * c.head_dim
        + 2 * (c.indexer_n_heads + c.indexer_kv_heads) * c.indexer_head_dim
        + ffn
}

const MTP_SCRATCH_SLACK: usize = 2 * 1024 * 1024 + 131072;

/// Mandatory dense MTP allocation peak, excluding the opportunistic draft LM head.
/// MoE/QSA retain their existing archive-dependent expert/cache contracts: this
/// dense-only correction must not double-charge their existing target reserves.
pub(crate) fn dense_serving_reserve(c: &ModelConfig, tp: usize, capacity: usize) -> usize {
    if c.num_experts > 0 || c.uses_qsa() {
        return 0;
    }
    let pages = capacity.div_ceil(16);
    let rows = capacity.min(256);
    let hc = if c.uses_hyper_connection() {
        c.hc_count.max(1)
    } else {
        1
    };
    let row = scratch_row_elements(c, tp, 3 * (c.intermediate_size / tp));
    let logits = c.vocab_size.div_ceil(tp) * 2;
    // Proposer IO: final-hidden tap, shifted rows, tail, four verify logits,
    // and per-slot tail + four captured verify rows (maximum supported MTP-3).
    // Using the tap row ceiling also bounds the proposer's staging row count.
    let tap_rows = capacity.min(crate::dflash::DFLASH2_MAX_BATCH_ROWS);
    let proposer_io =
        (2 * tap_rows + 1 + MAX_MTP_BATCH_SLOTS * 5) * c.hyper_hidden_size() * 2 + 4 * logits;
    // Capacity originates in --kv-pages. Saturate its products and the total so
    // impossible requests fail admission rather than wrapping to a small reserve.
    let kv_bytes = pages
        .saturating_mul(16)
        .saturating_mul(c.local_kv_heads(tp))
        .saturating_mul(c.head_dim)
        .saturating_mul(4);
    [
        proposer_io,
        weight_arena_bytes(
            c,
            tp,
            capacity,
            6 * c.hidden_size * (c.intermediate_size / tp),
        ),
        repack_staging_bytes(c, tp),
        kv_bytes,
        (rows + 1) * row * 2,
        2 * logits,
        2 * MTP_SCRATCH_SLACK,
        pages.saturating_mul(4),
        (rows + 1) * hc * c.hidden_size * 2,
        16 * 4,
        4 * 4,
    ]
    .into_iter()
    .fold(0usize, usize::saturating_add)
}

/// Target GDN state-slot budget for native MTP serving.
///
/// Single-sequence serving keeps the existing 8-slot pool (`3` private + `1` locked
/// radix checkpoint + `3` MTP-3 verify extras + `1` spare). Multi-sequence serving
/// budgets `3` private + `1` locked checkpoint per active sequence, batched MTP
/// verification extras (`2 * k` at `B = 2`, `min(B, 4) * min(k, 1)` at `B >= 3`),
/// plus `1` cached radix checkpoint per sequence for burst/multi-turn reuse.
pub(crate) fn serving_state_slots(max_num_seqs: usize, mtp_k: usize) -> crate::cuda::Result<usize> {
    if max_num_seqs == 0 {
        return Err(crate::model::loader_err(
            "serving max_num_seqs must be nonzero and fit four state slots per sequence",
        ));
    }
    let k = mtp_k.clamp(1, 3);
    if max_num_seqs == 1 {
        return Ok((4 + k).max(8));
    }
    let active_and_cached = max_num_seqs.checked_mul(5).ok_or_else(|| {
        crate::model::loader_err(
            "serving max_num_seqs must be nonzero and fit four state slots per sequence",
        )
    })?;
    let verify_extras = if max_num_seqs == 2 {
        2 * k
    } else if max_num_seqs <= 4 {
        max_num_seqs.min(4) * k.min(1)
    } else {
        max_num_seqs * k
    };
    active_and_cached.checked_add(verify_extras).ok_or_else(|| {
        crate::model::loader_err(
            "serving max_num_seqs must be nonzero and fit four state slots per sequence",
        )
    })
}

/// Double-buffered H2D staging capacity (`32` experts) for the single-layer `MTP` MoE workspace.
const MTP_MOE_STAGE_CAPACITY: usize = 32;

/// Minimum batch token capacity (`64` rows) allocated for the `MTP` tiered-MoE workspace.
const MTP_MOE_MIN_WORKSPACE_BATCH_ROWS: usize = 64;

/// Group size (`64`, `0.5625 B/weight`) used when online-quantizing a `DenseF16` `target.lm_head`
/// into a private `MarlinW4A16` (`U4B8`) proposal head for `MTP` (`vLLM` PR #59973
/// `vllm/model_executor/layers/quantization/online/lm_head.py`: `quantized_lm_head_copy`).
pub(crate) const MTP_DRAFT_LM_HEAD_GROUP_SIZE: usize = 64;

/// Alignment slack (`4 KiB`) added to the `MTP` `draft_lm_head` `DeviceArena` for 256-byte
/// aligned `b_q_weight` and `b_scales` sub-allocations.
pub(crate) const MTP_DRAFT_LM_HEAD_ARENA_SLACK_BYTES: usize = 4 * 1024;

/// Minimum free VRAM (`256 MiB`) that must remain after allocating all `MTP` weights, hot
/// experts, KV pool, scratch arenas, and the quantized `draft_lm_head` copy.
pub(crate) const MTP_DRAFT_LM_HEAD_POST_HEADROOM_BYTES: usize = 256 * 1024 * 1024;

/// Pure hardware-aware planner for an online-quantized `MarlinW4A16` copy of `target.lm_head`
/// (`vLLM` PR #59973 `quantized_lm_head_copy`).
///
/// Returns `Some((qweight_bytes, scales_bytes, arena_bytes))` when:
/// 1. `target_lm_head` is unquantized `LinearProj::DenseF16 { size_n, size_k, .. }`
///    (already-quantized `MarlinW4A16` / `MarlinW8A16` heads are shared at `0 B` extra VRAM),
/// 2. `(size_n, size_k)` satisfies Marlin's `U4B8` `group_size = 64` tile alignment
///    (`size_n % 64 == 0 && size_k % 128 == 0`), and
/// 3. `free_vram_after_mtp_init >= arena_bytes + MTP_DRAFT_LM_HEAD_POST_HEADROOM_BYTES`.
pub(crate) fn plan_mtp_draft_lm_head_bytes(
    target_lm_head: &LinearProj,
    free_vram_after_mtp_init: usize,
) -> Option<(usize, usize, usize)> {
    let LinearProj::DenseF16 { size_n, size_k, .. } = *target_lm_head else {
        return None;
    };
    if size_n == 0
        || size_k == 0
        || !size_n.is_multiple_of(64)
        || !size_k.is_multiple_of(128)
        || !size_k.is_multiple_of(MTP_DRAFT_LM_HEAD_GROUP_SIZE)
    {
        return None;
    }
    let qweight_bytes = (size_n / 2).checked_mul(size_k)?;
    let scales_bytes = size_n
        .checked_mul(size_k / MTP_DRAFT_LM_HEAD_GROUP_SIZE)?
        .checked_mul(2)?;
    let arena_bytes = qweight_bytes
        .checked_add(scales_bytes)?
        .checked_add(MTP_DRAFT_LM_HEAD_ARENA_SLACK_BYTES)?;
    let required_vram = arena_bytes.checked_add(MTP_DRAFT_LM_HEAD_POST_HEADROOM_BYTES)?;
    if free_vram_after_mtp_init >= required_vram {
        Some((qweight_bytes, scales_bytes, arena_bytes))
    } else {
        None
    }
}

/// Validate before slicing weights: both dense (`Qwen3_5MTP`) and MoE (`Qwen3_5MoeMTP`) geometries are checked explicitly.
pub fn validate_geometry(c: &ModelConfig, rank: usize, tp: usize) -> Result<()> {
    let ffn_ok = if c.num_experts == 0 {
        c.intermediate_size > 0 && c.intermediate_size.is_multiple_of(tp)
    } else {
        c.num_experts <= 512
            && c.num_experts_per_tok > 0
            && c.num_experts_per_tok <= 16
            && c.num_experts_per_tok <= c.num_experts
            && c.moe_intermediate_size > 0
            && (if c.uses_tiered_moe() {
                c.num_experts.is_multiple_of(tp) && c.moe_intermediate_size.is_multiple_of(8)
            } else {
                c.moe_intermediate_size.is_multiple_of(tp)
            })
            && c.shared_expert_intermediate_size.is_multiple_of(tp)
    };
    if !matches!(tp, 1 | 2 | 4)
        || rank >= tp
        || !ffn_ok
        || c.mtp_num_hidden_layers != 1
        || !c.attn_output_gate
        || c.hidden_size == 0
        || !matches!(c.head_dim, 128 | 256)
        || c.rotary_dim > c.head_dim
        || !c.rotary_dim.is_multiple_of(2)
        || c.vocab_size == 0
        || c.num_key_value_heads == 0
        || c.num_attention_heads == 0
        || !c.hidden_size.is_multiple_of(tp)
        || !c.num_attention_heads.is_multiple_of(tp)
        || !c.kv_heads_compatible_with_tp(tp)
        || !c.vocab_size.is_multiple_of(tp)
        || !(c.num_attention_heads / tp).is_multiple_of(c.local_kv_heads(tp))
    {
        return Err(loader_err("unsupported MTP TP geometry"));
    }
    Ok(())
}

pub const MAX_MTP_BATCH_SLOTS: usize = 8;

struct MtpGraphIo {
    propose_graphs: [Option<(usize, bool, CudaGraph)>; MAX_MTP_BATCH_SLOTS],
    propose_arena: DeviceArena,
    meta_dev: DeviceBuffer,
    out_ids_dev: DeviceBuffer,
}

enum MtpInputFusion {
    ConcatLinear {
        fc: LinearProj,
    },
    DualStreamBroadcast {
        fc_embedding: LinearProj,
        fc_hidden: LinearProj,
    },
}

struct MtpQsaWorkspace {
    qsa_sel_phys_buf: DeviceBuffer,
    qsa_sel_log_buf: DeviceBuffer,
    qsa_num_sel_buf: DeviceBuffer,
    qsa_seq_pos_buf: DeviceBuffer,
    skip_topk: bool,
    step0_page_idx: usize,
}

/// Owns only predictor weights and predictor KV. Pass the same target rank to
/// each operation; it is not borrowed across calls, allowing target verification
/// between rounds. The target must remain loaded (and its config unchanged).
pub struct Qwen3_5MtpRank {
    pub use_cuda_graph: bool,
    target_embedding: u64,
    hidden_size: usize,
    capacity: usize,
    total_capacity: usize,
    graph_io: MtpGraphIo,
    _weights: DeviceArena,
    scratch: DeviceArena,
    kv: KvPool,
    all_pages: Vec<i32>,
    all_pages_dev: u64,
    pages: Vec<i32>,
    pages_dev: u64,
    len: usize,
    committed: usize,
    active_slot: usize,
    batch_mode: bool,
    slot_state: [(usize, usize); MAX_MTP_BATCH_SLOTS],
    input_fusion: MtpInputFusion,
    residual: LayerResidual,
    final_residual: FinalResidual,
    fa: FullAttentionLayerWeights,
    ffn: FfnBlockWeights,
    norms: [u64; 2],
    tiered_moe: Option<crate::moe_offload::LayerTieredMoe>,
    qsa: Option<MtpQsaWorkspace>,
    _draft_lm_head_arena: Option<DeviceArena>,
    draft_lm_head: Option<LinearProj>,
    /// Recurrent proposal input: final-normalized for single-stream residuals,
    /// or the uncollapsed multi-stream residual for HyperConnection fusion.
    pub hidden: DeviceBuffer,
    /// Contiguous outputs of the last forward; only batch_hidden_rows rows are live.
    pub batch_hidden: DeviceBuffer,
    pub batch_hidden_rows: usize,
    max_batch_rows: usize,
}
impl Qwen3_5MtpRank {
    fn concat_fc(&self) -> Result<&LinearProj> {
        match &self.input_fusion {
            MtpInputFusion::ConcatLinear { fc } => Ok(fc),
            MtpInputFusion::DualStreamBroadcast { .. } => {
                Err(loader_err("MTP concat fusion required"))
            }
        }
    }

    /// Returns the online-quantized `MarlinW4A16` `draft_lm_head` copy when enabled, or
    /// falls back to borrowing `target.lm_head` (`vLLM` PR #59973 `quantized_lm_head_copy`).
    #[inline]
    pub(crate) fn lm_head<'a>(&'a self, target: &'a Qwen3_5ModelRank) -> &'a LinearProj {
        self.draft_lm_head.as_ref().unwrap_or(&target.lm_head)
    }

    /// Returns `true` when this `MTP` rank owns an online-quantized `MarlinW4A16` copy of `target.lm_head`.
    pub fn has_quantized_draft_lm_head(&self) -> bool {
        self.draft_lm_head.is_some()
    }

    /// Temporarily detaches the online-quantized `draft_lm_head` (falling back to `target.lm_head`)
    /// and invalidates any captured `propose_graphs`.
    pub fn take_quantized_draft_lm_head(&mut self) -> Option<LinearProj> {
        self.graph_io.propose_graphs = std::array::from_fn(|_| None);
        self.draft_lm_head.take()
    }

    /// Restores a previously detached `draft_lm_head` and invalidates any captured `propose_graphs`.
    pub fn restore_quantized_draft_lm_head(&mut self, head: Option<LinearProj>) {
        self.graph_io.propose_graphs = std::array::from_fn(|_| None);
        self.draft_lm_head = head;
    }

    pub fn supports_cuda_graph(&self) -> bool {
        matches!(self.input_fusion, MtpInputFusion::ConcatLinear { .. })
            && !self.final_residual.is_hyper_connection()
            && self.qsa.is_none()
            && self.tiered_moe.is_none()
    }

    #[allow(clippy::cognitive_complexity)]
    pub fn load(target: &Qwen3_5ModelRank, dir: &Path, capacity: usize) -> Result<Self> {
        let c = &target.cfg;
        validate_geometry(c, target.rank, target.tp_size)?;
        if capacity == 0 || capacity > i32::MAX as usize - 16 {
            return Err(loader_err("invalid MTP capacity"));
        }
        target.device.make_current()?;
        let mut archive =
            SafetensorsArchive::open(dir).map_err(|_| loader_err("open MTP archive"))?;
        if archive.resolve_name("mtp.norm.weight").is_none()
            && archive
                .resolve_name("mtp.hyper_connection_mixer.hc_norm.weight")
                .is_none()
        {
            let compact_dir = dir.join("runtime/mtp-int4-g32");
            if compact_dir.is_dir() {
                archive = SafetensorsArchive::open(&compact_dir)
                    .map_err(|_| loader_err("open compact MTP archive"))?;
            }
        }
        let r = target.rank;
        let tp = target.tp_size;
        let h = c.hidden_size;
        let q = c.num_attention_heads * c.head_dim;
        let kv = c.num_key_value_heads * c.head_dim;
        let local_kv = c.local_kv_heads(tp) * c.head_dim;
        let kv_shard_idx = c.kv_head_shard_index(r, tp);
        let i = c.intermediate_size;
        let uses_hc = c.uses_hyper_connection();
        let hc_count = if uses_hc { c.hc_count.max(1) } else { 1 };
        let mtp_row_elems = hc_count * h;
        let is_moe_mtp = c.num_experts > 0
            && (archive
                .resolve_name("mtp.layers.0.mlp.gate.weight")
                .is_some()
                || archive
                    .resolve_name("mtp.layers.0.mlp.gate.qweight")
                    .is_some()
                || archive
                    .resolve_name("mtp.layers.0.mlp.gate.weight_packed")
                    .is_some());
        let is_tiered_moe_mtp = uses_hc
            && c.uses_tiered_moe()
            && is_moe_mtp
            && (archive
                .resolve_name("mtp.layers.0.mlp.experts.0.gate_proj.qweight")
                .is_some()
                || archive
                    .resolve_name("mtp.layers.0.mlp.experts.0.gate_proj.weight_packed")
                    .is_some());
        let ffn_weight_bytes = if is_tiered_moe_mtp {
            let local_sh_i = c.shared_expert_intermediate_size / tp;
            2 * (c.num_experts * h + 3 * h * local_sh_i + h) + 32 * 1024 * 1024
        } else if is_moe_mtp {
            let local_moe_i = c.moe_intermediate_size / tp;
            let local_sh_i = c.shared_expert_intermediate_size / tp;
            let experts_are_w4 = c.quant.bits() == 4
                && (archive
                    .resolve_name("mtp.layers.0.mlp.experts.0.gate_proj.qweight")
                    .is_some()
                    || archive
                        .resolve_name("mtp.layers.0.mlp.experts.0.gate_proj.weight_packed")
                        .is_some());
            let expert_elem_bytes = if experts_are_w4 { 1usize } else { 2usize };
            c.num_experts * 3 * h * local_moe_i * expert_elem_bytes
                + 2 * (c.num_experts * h + 3 * h * local_sh_i + h)
                + 16 * 1024 * 1024
        } else {
            2 * 3 * h * (i / tp)
        };
        let mtp_capacity = if (65536..131072).contains(&capacity) && !is_tiered_moe_mtp {
            131072
        } else {
            capacity
        };
        let bytes = weight_arena_bytes(c, tp, mtp_capacity, ffn_weight_bytes);
        let mut weights = DeviceArena::new(target.device.device_id, bytes)?;
        let staging_bytes = repack_staging_bytes(c, tp);
        let staging = DeviceBuffer::alloc(target.device.device_id, staging_bytes)?;
        let load = |name: &str, n: usize, k: usize| -> Result<RawLinearHost> {
            let raw = RawLinearHost::load(&archive, &format!("mtp.{name}"), &c.quant)?;
            if raw.dims() == (n, k) {
                Ok(raw)
            } else {
                Err(crate::cuda::CudaError::with_detail(
                    -301,
                    "MTP requires correctly shaped weights",
                    format!("mtp.{name} has dims {:?}, expected ({n}, {k})", raw.dims()),
                ))
            }
        };
        let input_fusion = if uses_hc {
            MtpInputFusion::DualStreamBroadcast {
                fc_embedding: load("fc_embedding", h, h)?.upload(
                    &mut weights,
                    &target.stream,
                    &staging,
                )?,
                fc_hidden: load("fc_hidden", h, h)?.upload(
                    &mut weights,
                    &target.stream,
                    &staging,
                )?,
            }
        } else {
            MtpInputFusion::ConcatLinear {
                fc: load("fc", h, 2 * h)?.upload(&mut weights, &target.stream, &staging)?,
            }
        };
        let qs = load("layers.0.self_attn.q_proj", 2 * q, h)?
            .slice_rows(r * 2 * q / tp, (r + 1) * 2 * q / tp);
        let ks = load("layers.0.self_attn.k_proj", kv, h)?
            .slice_rows(kv_shard_idx * local_kv, (kv_shard_idx + 1) * local_kv);
        let vs = load("layers.0.self_attn.v_proj", kv, h)?
            .slice_rows(kv_shard_idx * local_kv, (kv_shard_idx + 1) * local_kv);
        let qkv_proj = RawLinearHost::concat_rows(&[qs, ks, vs])?.upload(
            &mut weights,
            &target.stream,
            &staging,
        )?;
        let o_proj = load("layers.0.self_attn.o_proj", h, q)?
            .slice_cols(r * q / tp, (r + 1) * q / tp)?
            .upload(&mut weights, &target.stream, &staging)?;
        let ffn = if is_moe_mtp {
            FfnBlockWeights::Moe(Box::new(load_moe_layer_weights(
                &archive,
                c,
                is_tiered_moe_mtp.then_some(c.num_hidden_layers),
                "mtp.layers.0",
                [r, tp],
                &mut weights,
                (&target.stream, &staging),
            )?))
        } else {
            let gs = load("layers.0.mlp.gate_proj", i, h)?.slice_rows(r * i / tp, (r + 1) * i / tp);
            let us = load("layers.0.mlp.up_proj", i, h)?.slice_rows(r * i / tp, (r + 1) * i / tp);
            let gate_up_proj = RawLinearHost::concat_rows(&[gs, us])?.upload(
                &mut weights,
                &target.stream,
                &staging,
            )?;
            let down_proj = load("layers.0.mlp.down_proj", h, i)?
                .slice_cols(r * i / tp, (r + 1) * i / tp)?
                .upload(&mut weights, &target.stream, &staging)?;
            FfnBlockWeights::Dense {
                gate_up_proj,
                down_proj,
            }
        };
        let qsa_indexer = if c.uses_qsa() {
            Some(load_qsa_indexer_weights(
                &archive,
                c,
                "mtp.layers.0",
                &mut weights,
                &target.stream,
                &staging,
            )?)
        } else {
            None
        };
        let norm = |weights: &mut DeviceArena, name: &str, size: usize| -> Result<u64> {
            let key = archive
                .resolve_name(&format!("mtp.{name}.weight"))
                .ok_or_else(|| loader_err("missing MTP norm"))?;
            let v = archive
                .read_f16_bits(&key)
                .map_err(|_| loader_err("read MTP norm"))?;
            if v.len() != size {
                return Err(loader_err("MTP norm shape"));
            }
            weights.upload_slice(&v, &target.stream)
        };
        let max_batch_rows = capacity.min(256);
        let (norms, residual, final_residual) = if uses_hc {
            let attn_hc = load_hc_block_weights(
                &archive,
                c,
                "mtp.layers.0.attn_hyper_connection",
                true,
                &mut weights,
                &target.stream,
                &staging,
            )?;
            let mlp_hc = load_hc_block_weights(
                &archive,
                c,
                "mtp.layers.0.mlp_hyper_connection",
                true,
                &mut weights,
                &target.stream,
                &staging,
            )?;
            let final_hc_mixer = load_hc_block_weights(
                &archive,
                c,
                "mtp.hyper_connection_mixer",
                false,
                &mut weights,
                &target.stream,
                &staging,
            )?;
            drop(staging);
            let pre_fc_norm_embedding_ptr = norm(&mut weights, "pre_fc_norm_embedding", h)?;
            let pre_fc_norm_hidden_ptr = norm(&mut weights, "pre_fc_norm_hidden", hc_count * h)?;
            (
                [pre_fc_norm_embedding_ptr, pre_fc_norm_hidden_ptr, 0, 0, 0],
                LayerResidual::HyperConnection { attn_hc, mlp_hc },
                FinalResidual::HyperConnection {
                    mixer: final_hc_mixer,
                },
            )
        } else {
            drop(staging);
            let norms = [
                norm(&mut weights, "pre_fc_norm_embedding", h)?,
                norm(&mut weights, "pre_fc_norm_hidden", h)?,
                norm(&mut weights, "layers.0.input_layernorm", h)?,
                norm(&mut weights, "layers.0.post_attention_layernorm", h)?,
                norm(&mut weights, "norm", h)?,
            ];
            (
                norms,
                LayerResidual::SingleStream {
                    input_layernorm_ptr: norms[2],
                    post_attention_layernorm_ptr: norms[3],
                },
                FinalResidual::SingleStream { norm_ptr: norms[4] },
            )
        };
        let qsa = if qsa_indexer.is_some() {
            let max_select_groups =
                (c.indexer_budget / c.indexer_compress_ratio.max(1)).clamp(1, 512) + 1;
            let qsa_sel_phys_buf = DeviceBuffer::alloc(
                target.device.device_id,
                max_batch_rows * max_select_groups * 4,
            )?;
            let qsa_sel_log_buf = DeviceBuffer::alloc(
                target.device.device_id,
                max_batch_rows * max_select_groups * 4,
            )?;
            let qsa_num_sel_buf = DeviceBuffer::alloc(target.device.device_id, max_batch_rows * 4)?;
            let qsa_seq_pos_buf = DeviceBuffer::alloc(target.device.device_id, max_batch_rows * 4)?;
            Some(MtpQsaWorkspace {
                qsa_sel_phys_buf,
                qsa_sel_log_buf,
                qsa_num_sel_buf,
                qsa_seq_pos_buf,
                skip_topk: false,
                step0_page_idx: usize::MAX,
            })
        } else {
            None
        };
        let fa = FullAttentionLayerWeights {
            layer_id: c.num_hidden_layers,
            kv_layer_ordinal: 0,
            qkv_proj,
            o_proj,
            q_norm_ptr: norm(&mut weights, "layers.0.self_attn.q_norm", c.head_dim)?,
            k_norm_ptr: norm(&mut weights, "layers.0.self_attn.k_norm", c.head_dim)?,
            qsa_indexer,
        };
        let mut kv = KvPool::new(
            target.device.device_id,
            W34_KV_NAMESPACE_MTP_FULL_ATTN,
            &[c.num_hidden_layers],
            c.local_kv_heads(tp),
            c.head_dim,
            16,
            mtp_capacity.div_ceil(16),
            KvFormatPolicy::Fp16,
        )?;
        if c.uses_qsa() {
            kv.init_qsa_indexer_caches(c.indexer_head_dim, c.indexer_compress_ratio.max(1))?;
        }
        let pages: Vec<i32> = (0..kv.num_pages)
            .map(|_| kv.alloc_page().unwrap())
            .collect();
        let pages_dev = weights.upload_slice(&pages, &target.stream)?;
        // All FP16 intermediates (including split QKV and full vocabulary logits),
        // plus alignment and per-request/token metadata.
        let ffn_row_elements = match &ffn {
            FfnBlockWeights::Dense { .. } => 3 * (i / tp),
            FfnBlockWeights::Moe(moe) => {
                let routed_interm =
                    if matches!(moe.experts, crate::model::MoeExpertStorage::Tiered { .. }) {
                        0
                    } else {
                        moe.top_k * moe.local_moe_interm
                    };
                moe.num_experts
                    + 4 * moe.top_k
                    + routed_interm
                    + 3 * moe.local_shared_interm
                    + h
                    + 64
            }
        };
        let row_elements = scratch_row_elements(c, tp, ffn_row_elements);
        let single_row_logits_bytes = target.local_vocab_size * 2;
        let scratch_bytes = max_batch_rows
            .checked_mul(row_elements)
            .and_then(|n| n.checked_mul(2))
            .and_then(|n| n.checked_add(single_row_logits_bytes))
            .and_then(|n| n.checked_add(MTP_SCRATCH_SLACK + capacity.div_ceil(16) * 4))
            .ok_or_else(|| loader_err("MTP scratch size overflow"))?;
        let propose_arena_bytes = row_elements
            .checked_mul(2)
            .and_then(|n| n.checked_add(single_row_logits_bytes))
            .and_then(|n| n.checked_add(MTP_SCRATCH_SLACK))
            .ok_or_else(|| loader_err("MTP propose arena overflow"))?;
        let graph_io = MtpGraphIo {
            propose_graphs: std::array::from_fn(|_| None),
            propose_arena: DeviceArena::new(target.device.device_id, propose_arena_bytes)?,
            meta_dev: DeviceBuffer::alloc(target.device.device_id, 16 * 4)?,
            out_ids_dev: DeviceBuffer::alloc(target.device.device_id, 4 * 4)?,
        };
        let scratch = DeviceArena::new(target.device.device_id, scratch_bytes)?;
        let hidden = DeviceBuffer::alloc(target.device.device_id, mtp_row_elems * 2)?;
        let batch_hidden =
            DeviceBuffer::alloc(target.device.device_id, max_batch_rows * mtp_row_elems * 2)?;
        let tiered_moe = if is_tiered_moe_mtp {
            let (group_size, mtp_quant) = if let Some(ws_key) =
                archive.resolve_name("mtp.layers.0.mlp.experts.0.gate_proj.weight_scale")
                && let Ok(ws_info) = archive.tensor_info(&ws_key)
                && ws_info.shape.len() == 2
                && ws_info.shape[1] > 0
                && h.is_multiple_of(ws_info.shape[1])
            {
                let gs = h / ws_info.shape[1];
                let (ignore, rules) = match &c.quant {
                    crate::loader::QuantScheme::CompressedTensors { ignore, rules, .. } => {
                        (ignore.clone(), rules.clone())
                    }
                    _ => (Vec::new(), None),
                };
                (
                    gs,
                    crate::loader::QuantScheme::CompressedTensors {
                        num_bits: 4,
                        group_size: gs,
                        symmetric: true,
                        ignore,
                        rules,
                    },
                )
            } else if let Some(sc_key) =
                archive.resolve_name("mtp.layers.0.mlp.experts.0.gate_proj.scales")
                && let Ok(sc_info) = archive.tensor_info(&sc_key)
                && sc_info.shape.len() == 2
                && sc_info.shape[0] > 0
                && h.is_multiple_of(sc_info.shape[0])
            {
                let gs = h / sc_info.shape[0];
                (
                    gs,
                    crate::loader::QuantScheme::AutoRound {
                        bits: 4,
                        group_size: gs,
                        sym: true,
                        awq_packing: false,
                    },
                )
            } else {
                (
                    c.routed_expert_quant_scheme().group_size_or(128),
                    c.quant.clone(),
                )
            };
            let down_had = archive
                .resolve_name("mtp.layers.0.mlp.experts.0.down_proj.weight_packed")
                .and_then(|k| k.strip_suffix(".weight_packed").map(str::to_owned))
                .or_else(|| {
                    archive
                        .resolve_name("mtp.layers.0.mlp.experts.0.down_proj.qweight")
                        .and_then(|k| k.strip_suffix(".qweight").map(str::to_owned))
                })
                .map_or_else(
                    || {
                        mtp_quant
                            .input_hadamard("mtp.layers.0.mlp.experts.0.down_proj")
                            .ok()
                            .flatten()
                            .unwrap_or(0)
                    },
                    |name| mtp_quant.input_hadamard(&name).ok().flatten().unwrap_or(0),
                );
            let layout =
                crate::moe_offload::ExpertPackLayout::new(c.moe_intermediate_size, h, group_size)
                    .and_then(|l| l.with_down_hadamard(down_had))
                    .map_err(|e| {
                        crate::cuda::CudaError::with_detail(
                            -1,
                            "MTP ExpertPackLayout::new",
                            e.to_string(),
                        )
                    })?;
            let local_experts = c.num_experts / tp;
            let stage_cap = MTP_MOE_STAGE_CAPACITY.min(local_experts).max(1);
            let mut ws = crate::moe_offload::TieredMoeWorkspace::new(
                target.device.device_id,
                layout.clone(),
                stage_cap,
                max_batch_rows.max(MTP_MOE_MIN_WORKSPACE_BATCH_ROWS),
                c.num_experts_per_tok.max(1),
            )?;
            let free_vram = target.device.mem_info().map_or(0, |(free, _)| free);
            let k_hot = crate::model::plan_mtp_tiered_moe_hot_experts(
                tp,
                local_experts,
                layout.stride,
                free_vram,
            );
            let fully_resident = k_hot >= local_experts;
            let mut store = crate::moe_offload::HostExpertStore::load_with_pin(
                &archive,
                "mtp.layers.0.mlp.experts",
                &mtp_quant,
                layout.clone(),
                c.num_experts,
                r,
                tp,
                usize::MAX,
                !fully_resident,
            )
            .map_err(|e| {
                crate::cuda::CudaError::with_detail(-1, "MTP HostExpertStore::load", e.to_string())
            })?;
            let mut cache = crate::moe_offload::HotExpertCache::new(
                target.device.device_id,
                k_hot,
                store.expert_ids(),
                layout,
            )?;
            let initial =
                crate::moe_offload::HotExpertRankings::from_model_dir(dir).and_then(|rk| {
                    rk.select_for_layer(
                        c.num_hidden_layers.saturating_sub(1),
                        store.expert_ids(),
                        k_hot,
                    )
                });
            cache.populate_initial(&store, initial.as_deref(), &mut ws, &target.stream)?;
            if fully_resident {
                target.stream.synchronize()?;
                store.release_host_storage();
            } else if tp == 1
                && store.layout().supports_cpu_bridge()
                && let Some(ref tmoe) = target.tiered_moe
                && let Some(rt) = tmoe.bridge_runtime()
            {
                let k_static = if k_hot >= 64 { k_hot - 24 } else { k_hot };
                cache.bridge_runtime = Some(rt);
                cache.register_cpu_bridge(&store, k_static, &target.stream)?;
            }
            Some(crate::moe_offload::LayerTieredMoe::new(store, cache, ws))
        } else {
            None
        };
        let free_vram_post_mtp = target.device.mem_info().map_or(0, |(free, _)| free);
        let (_draft_lm_head_arena, draft_lm_head) = if let LinearProj::DenseF16 {
            weight_ptr,
            size_n,
            size_k,
        } = target.lm_head
            && let Some((qweight_bytes, scales_bytes, arena_bytes)) =
                plan_mtp_draft_lm_head_bytes(&target.lm_head, free_vram_post_mtp)
            && let Ok(mut arena) = DeviceArena::new(target.device.device_id, arena_bytes)
            && let Ok(b_q_weight_ptr) = arena.alloc_aligned(qweight_bytes)
            && let Ok(b_scales_ptr) = arena.alloc_aligned(scales_bytes)
        {
            let ex = target.stream.exec(0, 0);
            // SAFETY: `weight_ptr` holds `size_n * size_k` FP16 elements on `target.device`,
            // and `b_q_weight_ptr` / `b_scales_ptr` are freshly allocated in `arena` with
            // `qweight_bytes` and `scales_bytes` capacity on `target.stream`.
            let rc = unsafe {
                ffi::w34_marlin_quantize_f16_u4b8(
                    &ex,
                    weight_ptr as *const core::ffi::c_void,
                    b_q_weight_ptr as *mut u32,
                    b_scales_ptr as *mut core::ffi::c_void,
                    size_k as i64,
                    size_n as i64,
                    MTP_DRAFT_LM_HEAD_GROUP_SIZE as i32,
                )
            };
            if rc == 0 {
                (
                    Some(arena),
                    Some(LinearProj::MarlinW4A16 {
                        b_q_weight_ptr,
                        b_scales_ptr,
                        size_n,
                        size_k,
                        group_blocks: (MTP_DRAFT_LM_HEAD_GROUP_SIZE / 16) as i32,
                    }),
                )
            } else if rc < 0 {
                (None, None)
            } else {
                crate::cuda::check_rc(rc, "w34_marlin_quantize_f16_u4b8 (MTP draft_lm_head)")?;
                unreachable!()
            }
        } else {
            (None, None)
        };
        target.stream.synchronize()?;
        let all_pages = pages.clone();
        Ok(Self {
            use_cuda_graph: !uses_hc && !is_tiered_moe_mtp && qsa.is_none(),
            target_embedding: target.embed_tokens_ptr,
            hidden_size: mtp_row_elems,
            capacity,
            total_capacity: capacity,
            graph_io,
            _weights: weights,
            scratch,
            kv,
            all_pages,
            all_pages_dev: pages_dev,
            pages,
            pages_dev,
            len: 0,
            committed: 0,
            active_slot: 0,
            batch_mode: false,
            slot_state: [(0, 0); MAX_MTP_BATCH_SLOTS],
            input_fusion,
            residual,
            final_residual,
            fa,
            ffn,
            norms: [norms[0], norms[1]],
            tiered_moe,
            qsa,
            _draft_lm_head_arena,
            draft_lm_head,
            hidden,
            batch_hidden,
            batch_hidden_rows: 0,
            max_batch_rows,
        })
    }
    pub fn len(&self) -> usize {
        self.len
    }
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }
    pub fn committed_len(&self) -> usize {
        self.committed
    }
    pub fn kv_layout(&self) -> &ffi::W34KvLayerLayout {
        &self.kv.layers[0].layout
    }
    pub fn kv_pool(&self) -> &KvPool {
        &self.kv
    }
    pub fn batch_slot_count(&self) -> usize {
        if self.all_pages.len() >= 8192 {
            MAX_MTP_BATCH_SLOTS
        } else {
            4
        }
    }
    pub fn batch_slot_capacity(&self) -> usize {
        (self.all_pages.len() / self.batch_slot_count()) * 16
    }
    pub fn is_batch_mode(&self) -> bool {
        self.batch_mode
    }
    pub fn set_batch_mode(&mut self, batch_mode: bool) -> Result<()> {
        if batch_mode && self.batch_slot_capacity() < 32 {
            return Err(loader_err("insufficient MTP KV pages for batched slots"));
        }
        if self.batch_mode != batch_mode {
            self.graph_io.propose_graphs = std::array::from_fn(|_| None);
            self.batch_mode = batch_mode;
        }
        self.slot_state = [(0, 0); MAX_MTP_BATCH_SLOTS];
        self.active_slot = 0;
        self.len = 0;
        self.committed = 0;
        if batch_mode {
            let slot_pages = self.all_pages.len() / self.batch_slot_count();
            self.pages = self.all_pages[..slot_pages].to_vec();
            self.pages_dev = self.all_pages_dev;
            self.capacity = slot_pages * 16;
        } else {
            self.pages = self.all_pages.clone();
            self.pages_dev = self.all_pages_dev;
            self.capacity = self.total_capacity;
        }
        Ok(())
    }
    pub fn promote_single_to_batch_slot0(&mut self) -> Result<()> {
        let slot_cap = self.batch_slot_capacity();
        if slot_cap < 32 || self.len > slot_cap {
            return Err(loader_err(
                "single-slot MTP state exceeds batch slot 0 capacity",
            ));
        }
        if !self.batch_mode {
            self.graph_io.propose_graphs = std::array::from_fn(|_| None);
            self.batch_mode = true;
        }
        self.slot_state = [(0, 0); MAX_MTP_BATCH_SLOTS];
        self.slot_state[0] = (self.len, self.committed);
        self.active_slot = 0;
        let slot_pages = self.all_pages.len() / self.batch_slot_count();
        self.pages = self.all_pages[..slot_pages].to_vec();
        self.pages_dev = self.all_pages_dev;
        self.capacity = slot_cap;
        Ok(())
    }
    pub fn select_batch_slot(&mut self, slot: usize) -> Result<()> {
        let slot_count = self.batch_slot_count();
        if !self.batch_mode || slot >= slot_count {
            return Err(loader_err("invalid MTP batch slot"));
        }
        if slot == self.active_slot {
            return Ok(());
        }
        self.slot_state[self.active_slot] = (self.len, self.committed);
        self.active_slot = slot;
        (self.len, self.committed) = self.slot_state[slot];
        let slot_pages = self.all_pages.len() / slot_count;
        self.pages = self.all_pages[slot * slot_pages..(slot + 1) * slot_pages].to_vec();
        self.pages_dev = self.all_pages_dev + (slot * slot_pages * 4) as u64;
        self.capacity = slot_pages * 16;
        Ok(())
    }
    /// Discard provisional suffix. Stale bytes remain unreachable and are overwritten.
    /// Even accepted draft rows must be replayed using TARGET hidden rows before
    /// another round: recurrent draft hidden is not the committed target hidden.
    pub fn truncate(&mut self, len: usize) -> Result<()> {
        if len > self.len {
            return Err(loader_err("MTP truncate grows cache"));
        }
        self.len = len;
        self.committed = self.committed.min(len);
        self.slot_state[self.active_slot] = (self.len, self.committed);
        Ok(())
    }
    pub fn copy_kv_from_slot(
        &mut self,
        target: &Qwen3_5ModelRank,
        src_slot: usize,
        dst_slot: usize,
        tokens: usize,
    ) -> Result<()> {
        let slot_count = self.batch_slot_count();
        let slot_pages = self.all_pages.len() / slot_count;
        let pages_to_copy = tokens.div_ceil(16).min(slot_pages);
        if src_slot != dst_slot && pages_to_copy > 0 {
            target.device.make_current()?;
            for layer in &self.kv.layers {
                let k_stride = layer.layout.k_page_stride_bytes as usize;
                let v_stride = layer.layout.v_page_stride_bytes as usize;
                let k_src = layer.k_buf.addr() + (src_slot * slot_pages * k_stride) as u64;
                let k_dst = layer.k_buf.addr() + (dst_slot * slot_pages * k_stride) as u64;
                let v_src = layer.v_buf.addr() + (src_slot * slot_pages * v_stride) as u64;
                let v_dst = layer.v_buf.addr() + (dst_slot * slot_pages * v_stride) as u64;
                crate::cuda::check_rc(
                    // SAFETY: Each slot owns a disjoint `slot_pages * stride` region inside `k_buf`.
                    unsafe {
                        ffi::w34_memcpy_d2d_async(
                            k_dst as *mut _,
                            k_src as *const _,
                            pages_to_copy * k_stride,
                            target.stream.handle,
                        )
                    },
                    "MTP slot K copy",
                )?;
                crate::cuda::check_rc(
                    // SAFETY: Each slot owns a disjoint `slot_pages * stride` region inside `v_buf`.
                    unsafe {
                        ffi::w34_memcpy_d2d_async(
                            v_dst as *mut _,
                            v_src as *const _,
                            pages_to_copy * v_stride,
                            target.stream.handle,
                        )
                    },
                    "MTP slot V copy",
                )?;
                let total_pages = self.kv.num_pages.max(1);
                if let Some(ref raw_buf) = layer.qsa_raw_k_buf {
                    let raw_stride = raw_buf.len_bytes / total_pages;
                    let r_src = raw_buf.addr() + (src_slot * slot_pages * raw_stride) as u64;
                    let r_dst = raw_buf.addr() + (dst_slot * slot_pages * raw_stride) as u64;
                    crate::cuda::check_rc(
                        // SAFETY: Each slot owns a disjoint `slot_pages * raw_stride` region inside `qsa_raw_k_buf`.
                        unsafe {
                            ffi::w34_memcpy_d2d_async(
                                r_dst as *mut _,
                                r_src as *const _,
                                pages_to_copy * raw_stride,
                                target.stream.handle,
                            )
                        },
                        "MTP slot QSA raw K copy",
                    )?;
                }
                if let Some(ref comp_buf) = layer.qsa_comp_k_buf {
                    let comp_stride = comp_buf.len_bytes / total_pages;
                    let c_src = comp_buf.addr() + (src_slot * slot_pages * comp_stride) as u64;
                    let c_dst = comp_buf.addr() + (dst_slot * slot_pages * comp_stride) as u64;
                    crate::cuda::check_rc(
                        // SAFETY: Each slot owns a disjoint `slot_pages * comp_stride` region inside `qsa_comp_k_buf`.
                        unsafe {
                            ffi::w34_memcpy_d2d_async(
                                c_dst as *mut _,
                                c_src as *const _,
                                pages_to_copy * comp_stride,
                                target.stream.handle,
                            )
                        },
                        "MTP slot QSA comp K copy",
                    )?;
                }
            }
        }
        self.slot_state[dst_slot] = (tokens, tokens);
        if self.active_slot == dst_slot {
            self.len = tokens;
            self.committed = tokens;
        }
        Ok(())
    }
    /// Ingest shifted committed pairs: `next_tokens[j]` is `token[start+j+1]`,
    /// `target_hidden` contains final-normalized target `hidden[start+j]`, FP16.
    /// Caller must keep `next_tokens.len() * hidden_size * 2` readable device
    /// bytes at this pointer on the target device through this call. The span
    /// may be a separate restored-prefix buffer, not necessarily the tap buffer.
    /// At most min(capacity, 256) rows per call. On error no rows are committed.
    pub fn ingest_no_sync(
        &mut self,
        target: &Qwen3_5ModelRank,
        next_tokens: &[i32],
        target_hidden: u64,
        tp2: Option<&Tp2Collectives>,
    ) -> Result<()> {
        self.ingest_inner(target, next_tokens, target_hidden, tp2, false)
    }
    /// Populate only `mtp.layers.0` KV cache rows for committed historical tokens when the
    /// caller will seed the next proposal directly from the target model's final hidden state.
    /// Because MTP has a single transformer layer, post-KV attention, `o_proj`, MoE/FFN, and
    /// `TP=2` all-reduces on ingested rows have zero effect on `self.kv` or subsequent proposals.
    pub fn ingest_kv_only_no_sync(
        &mut self,
        target: &Qwen3_5ModelRank,
        next_tokens: &[i32],
        target_hidden: u64,
        tp2: Option<&Tp2Collectives>,
    ) -> Result<()> {
        self.ingest_inner(target, next_tokens, target_hidden, tp2, true)
    }
    pub fn ingest_kv_only(
        &mut self,
        target: &Qwen3_5ModelRank,
        next_tokens: &[i32],
        target_hidden: u64,
        tp2: Option<&Tp2Collectives>,
    ) -> Result<()> {
        self.ingest_kv_only_no_sync(target, next_tokens, target_hidden, tp2)?;
        if !next_tokens.is_empty() {
            target.device.make_current()?;
            target.stream.synchronize()?;
        }
        Ok(())
    }
    fn ingest_inner(
        &mut self,
        target: &Qwen3_5ModelRank,
        next_tokens: &[i32],
        target_hidden: u64,
        tp2: Option<&Tp2Collectives>,
        kv_only: bool,
    ) -> Result<()> {
        if self.len != self.committed {
            return Err(loader_err("truncate provisional MTP rows before ingest"));
        }
        if next_tokens.len() > self.capacity - self.len
            || next_tokens
                .iter()
                .any(|&id| id < 0 || id as usize >= target.cfg.vocab_size)
            || (!next_tokens.is_empty() && target_hidden == 0)
        {
            return Err(loader_err("invalid MTP ingest rows"));
        }
        if !next_tokens.is_empty() {
            self.forward_inner(
                target,
                next_tokens,
                target_hidden,
                tp2,
                false,
                false,
                kv_only,
            )?;
            self.committed = self.len;
        }
        Ok(())
    }
    pub fn ingest(
        &mut self,
        target: &Qwen3_5ModelRank,
        next_tokens: &[i32],
        target_hidden: u64,
        tp2: Option<&Tp2Collectives>,
    ) -> Result<()> {
        self.ingest_no_sync(target, next_tokens, target_hidden, tp2)?;
        if !next_tokens.is_empty() {
            target.device.make_current()?;
            target.stream.synchronize()?;
        }
        Ok(())
    }
}
