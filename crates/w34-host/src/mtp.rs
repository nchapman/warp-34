//! Native single-layer dense & MoE Qwen3.5 / Qwen3.6 MTP (`Qwen3_5MTP` + `Qwen3_5MoeMTP`),
//! using the target stream and shared vocabulary weights.
//!
//! Donor: vLLM qwen3_5_mtp.py and llm_base_proposer.py. At MTP position p,
//! the inputs are target FINAL-normalized hidden[p] and token[p+1], NOT
//! hidden[p-1] and token[p]. Positions stay p when input IDs shift left.
//! Ingest prompt pairs up to p-1, then propose with hidden[p] and the target's
//! newly sampled anchor token[p+1]. The output predicts token[p+2]. Later
//! steps feed the previous MTP final-normalized hidden and sampled draft,
//! increasing position by one. No target GDN state is touched.
//!
//! This is a single-sequence eager component. All rows use FP16; FC is
//! replicated (algebraically the donor's column-shard + gather), attention
//! and dense/MoE FFN are TP-sharded. Vocabulary storage is borrowed, never copied.
//! TP2 callers must run both ranks concurrently with the same collective order.
#![allow(clippy::too_many_arguments)]
use crate::cache::{KvFormatPolicy, KvPool};
use crate::cuda::{CudaGraph, DeviceArena, DeviceBuffer, Result, Tp2Collectives, check_rc};
use crate::loader::{ModelConfig, SafetensorsArchive};
use crate::model::{
    FfnBlockWeights, FinalResidual, FullAttentionLayerWeights, HcStepContext, LayerResidual,
    LinearProj, Qwen3_5ModelRank, RawLinearHost, TapExport, load_hc_block_weights,
    load_moe_layer_weights, load_qsa_indexer_weights, loader_err,
};
use core::ffi::c_void;
use std::path::Path;
use w34_kernels::ffi::{
    self, W34_ATTN_MASK_CAUSAL, W34_KV_NAMESPACE_MTP_FULL_ATTN, W34AttentionBatch, W34Exec,
};

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

pub const MAX_MTP_BATCH_SLOTS: usize = 4;

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

type MtpTieredMoe = (
    crate::moe_offload::HostExpertStore,
    crate::moe_offload::HotExpertCache,
    crate::moe_offload::TieredMoeWorkspace,
);

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
    tiered_moe: Option<MtpTieredMoe>,
    qsa: Option<MtpQsaWorkspace>,
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

    pub fn supports_cuda_graph(&self) -> bool {
        matches!(self.input_fusion, MtpInputFusion::ConcatLinear { .. })
            && !self.final_residual.is_hyper_connection()
            && self.qsa.is_none()
            && self.tiered_moe.is_none()
    }

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
        let bytes = 2
            * (2 * h * h
                + (2 * q) * h / tp
                + (2 * local_kv) * h
                + h * q / tp
                + (5 + hc_count) * h
                + 2 * c.head_dim)
            + ffn_weight_bytes
            + capacity.div_ceil(16) * 4
            + 32 * 1024 * 1024;
        let mut weights = DeviceArena::new(target.device.device_id, bytes)?;
        let staging = DeviceBuffer::alloc(target.device.device_id, 64 * 1024 * 1024)?;
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
            .slice_cols(r * q / tp, (r + 1) * q / tp)
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
                .slice_cols(r * i / tp, (r + 1) * i / tp)
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
        let tiered_moe = if is_tiered_moe_mtp {
            let (group_size, mtp_quant) = if let Some(ws_key) =
                archive.resolve_name("mtp.layers.0.mlp.experts.0.gate_proj.weight_scale")
                && let Ok(ws_info) = archive.tensor_info(&ws_key)
                && ws_info.shape.len() == 2
                && ws_info.shape[1] > 0
                && h.is_multiple_of(ws_info.shape[1])
            {
                let gs = h / ws_info.shape[1];
                (
                    gs,
                    crate::loader::QuantScheme::CompressedTensors {
                        num_bits: 4,
                        group_size: gs,
                        symmetric: true,
                        ignore: Vec::new(),
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
                (c.quant.group_size_or(128), c.quant.clone())
            };
            let layout =
                crate::moe_offload::ExpertPackLayout::new(c.moe_intermediate_size, h, group_size)
                    .map_err(|e| {
                    crate::cuda::CudaError::with_detail(
                        -1,
                        "MTP ExpertPackLayout::new",
                        e.to_string(),
                    )
                })?;
            let local_experts = c.num_experts / tp;
            let free_vram = target.device.mem_info().map_or(0, |(free, _)| free);
            let slots_128_need = 128usize
                .saturating_mul(layout.stride)
                .saturating_add(256 * 1024 * 1024);
            let k_hot = if tp == 1 && local_experts >= 192 && free_vram >= 2048 * 1024 * 1024 {
                192
            } else if local_experts >= 128 && free_vram >= slots_128_need.max(512 * 1024 * 1024) {
                128
            } else if local_experts >= 64 {
                64
            } else {
                16.min(local_experts).max(1)
            };
            let fully_resident = k_hot >= local_experts;
            let stage_cap = 32.min(local_experts).max(1);
            let mut ws = crate::moe_offload::TieredMoeWorkspace::new(
                target.device.device_id,
                layout.clone(),
                stage_cap,
                max_batch_rows.max(64),
                c.num_experts_per_tok.max(1),
            )?;
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
            }
            Some((store, cache, ws))
        } else {
            None
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
            capacity.div_ceil(16),
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
        let row_elements = (9 + 4 * hc_count) * h
            + c.hc_lowrank.max(1)
            + 3 * hc_count
            + 8 * q / tp
            + 6 * local_kv
            + 2 * (c.indexer_n_heads + c.indexer_kv_heads) * c.indexer_head_dim
            + ffn_row_elements
            + target.local_vocab_size;
        let scratch_bytes = max_batch_rows
            .checked_mul(row_elements)
            .and_then(|n| n.checked_mul(2))
            .and_then(|n| n.checked_add(2 * 1024 * 1024 + 131072 + capacity.div_ceil(16) * 4))
            .ok_or_else(|| loader_err("MTP scratch size overflow"))?;
        let propose_arena_bytes = row_elements
            .checked_mul(2)
            .and_then(|n| n.checked_add(2 * 1024 * 1024 + 131072))
            .ok_or_else(|| loader_err("MTP propose arena overflow"))?;
        let graph_io = MtpGraphIo {
            propose_graphs: std::array::from_fn(|_| None),
            propose_arena: DeviceArena::new(target.device.device_id, propose_arena_bytes)?,
            meta_dev: DeviceBuffer::alloc(target.device.device_id, 16 * 4)?,
            out_ids_dev: DeviceBuffer::alloc(target.device.device_id, 4 * 4)?,
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
            scratch: DeviceArena::new(target.device.device_id, scratch_bytes)?,
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
            hidden: DeviceBuffer::alloc(target.device.device_id, mtp_row_elems * 2)?,
            batch_hidden: DeviceBuffer::alloc(
                target.device.device_id,
                max_batch_rows * mtp_row_elems * 2,
            )?,
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
    pub fn batch_slot_capacity(&self) -> usize {
        (self.all_pages.len() / MAX_MTP_BATCH_SLOTS) * 16
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
            let slot_pages = self.all_pages.len() / MAX_MTP_BATCH_SLOTS;
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
        let slot_pages = self.all_pages.len() / MAX_MTP_BATCH_SLOTS;
        self.pages = self.all_pages[..slot_pages].to_vec();
        self.pages_dev = self.all_pages_dev;
        self.capacity = slot_cap;
        Ok(())
    }
    pub fn select_batch_slot(&mut self, slot: usize) -> Result<()> {
        if !self.batch_mode || slot >= MAX_MTP_BATCH_SLOTS {
            return Err(loader_err("invalid MTP batch slot"));
        }
        if slot == self.active_slot {
            return Ok(());
        }
        self.slot_state[self.active_slot] = (self.len, self.committed);
        self.active_slot = slot;
        (self.len, self.committed) = self.slot_state[slot];
        let slot_pages = self.all_pages.len() / MAX_MTP_BATCH_SLOTS;
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
    /// Ingest shifted committed pairs: next_tokens[j] is token[start+j+1],
    /// target_hidden contains final-normalized target hidden[start+j], FP16.
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
    pub fn has_propose_graph(&self, k: usize) -> bool {
        let long_ctx = self.len + k > 1024;
        let idx = if self.batch_mode { self.active_slot } else { 0 };
        self.graph_io.propose_graphs[idx]
            .as_ref()
            .is_some_and(|(cap_k, cap_long, _)| *cap_k == k && *cap_long == long_ctx)
    }
    fn validate_propose_args(
        &self,
        target: &Qwen3_5ModelRank,
        anchor: i32,
        target_hidden: u64,
        k: usize,
        tp2_expected: Option<bool>,
    ) -> Result<()> {
        if target.embed_tokens_ptr != self.target_embedding {
            return Err(loader_err("MTP target rank mismatch"));
        }
        if let Some(has_tp2) = tp2_expected
            && (target.tp_size > 1) != has_tp2
        {
            return Err(loader_err("invalid MTP TP2 collective state"));
        }
        let span = self.hidden_size.checked_mul(2);
        if !(1..=3).contains(&k)
            || self.len != self.committed
            || self.len + k > self.capacity
            || anchor < 0
            || anchor as usize >= target.cfg.vocab_size
            || target_hidden == 0
            || !target_hidden.is_multiple_of(2)
            || span
                .and_then(|n| target_hidden.checked_add(n as u64))
                .is_none()
        {
            return Err(loader_err("invalid MTP proposal state or k"));
        }
        Ok(())
    }
    fn enqueue_propose_steps(
        &mut self,
        t: &Qwen3_5ModelRank,
        k: usize,
        long_ctx: bool,
        tp2: Option<&Tp2Collectives>,
    ) -> Result<()> {
        self.graph_io.propose_arena.reset();
        let h_dim = t.cfg.hidden_size;
        let head_dim = t.cfg.head_dim;
        let local_q_heads = t.cfg.num_attention_heads / t.tp_size;
        let local_kv_heads = t.cfg.local_kv_heads(t.tp_size);
        let local_interm = t.cfg.intermediate_size / t.tp_size;
        let q_gate_dim = local_q_heads * head_dim * 2;
        let kv_dim = local_kv_heads * head_dim;
        let arena = &mut self.graph_io.propose_arena;
        let residual_buf = arena.alloc_aligned(h_dim * 2)?;
        let normed_buf = arena.alloc_aligned(h_dim * 2)?;
        let embed = arena.alloc_aligned(h_dim * 2)?;
        let concat = arena.alloc_aligned((2 * h_dim) * 2)?;
        let qkv_proj_buf = arena.alloc_aligned((q_gate_dim + 2 * kv_dim) * 2)?;
        let q_buf = arena.alloc_aligned((local_q_heads * head_dim) * 2)?;
        let k_buf = arena.alloc_aligned(kv_dim * 2)?;
        let gate_buf = arena.alloc_aligned((local_q_heads * head_dim) * 2)?;
        let attn_out_buf = arena.alloc_aligned((local_q_heads * head_dim) * 2)?;
        let gated_attn_buf = arena.alloc_aligned((local_q_heads * head_dim) * 2)?;
        let block_out_buf = arena.alloc_aligned(h_dim * 2)?;
        let mlp_out_buf = arena.alloc_aligned(h_dim * 2)?;
        let (
            gate_up_buf,
            act_buf,
            moe_router_logits_buf,
            moe_topk_weights_buf,
            moe_topk_ids_buf,
            moe_interm_buf,
            moe_sh_gu_buf,
            moe_sh_act_buf,
            moe_sh_out_buf,
            moe_sh_gate_buf,
        ) = match &self.ffn {
            FfnBlockWeights::Dense { .. } => (
                arena.alloc_aligned((2 * local_interm) * 2)?,
                arena.alloc_aligned(local_interm * 2)?,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
            ),
            FfnBlockWeights::Moe(moe) => {
                let sh_gu = if moe.local_shared_interm > 0 {
                    arena.alloc_aligned(2 * moe.local_shared_interm * 2)?
                } else {
                    0
                };
                let sh_act = if moe.local_shared_interm > 0 {
                    arena.alloc_aligned(moe.local_shared_interm * 2)?
                } else {
                    0
                };
                let sh_out = if moe.local_shared_interm > 0 {
                    arena.alloc_aligned(h_dim * 2)?
                } else {
                    0
                };
                let sh_gate = if moe.shared_expert_gate.is_some() {
                    arena.alloc_aligned(2)?
                } else {
                    0
                };
                let moe_interm =
                    if matches!(moe.experts, crate::model::MoeExpertStorage::Tiered { .. }) {
                        0
                    } else {
                        arena.alloc_aligned(moe.top_k * moe.local_moe_interm * 2)?
                    };
                (
                    0,
                    0,
                    arena.alloc_aligned(moe.num_experts * 2)?,
                    arena.alloc_aligned(moe.top_k * 4)?,
                    arena.alloc_aligned(moe.top_k * 4)?,
                    moe_interm,
                    sh_gu,
                    sh_act,
                    sh_out,
                    sh_gate,
                )
            }
        };
        let logits_buf = arena.alloc_aligned(t.local_vocab_size * 2)?;
        let local_id = arena.alloc_aligned(4)?;
        let local_val = arena.alloc_aligned(4)?;
        let global_val = arena.alloc_aligned(4)?;
        let (ws_ptr, ws_cap) = arena.remaining_workspace();
        let ex = t.stream.exec(ws_ptr, ws_cap);

        let meta_base = self.graph_io.meta_dev.addr();
        let token_ids_ptr = meta_base;
        let qs_ptr = meta_base + 10 * 4;
        let qkv_row_stride = (q_gate_dim + 2 * kv_dim) as i32;
        let q_gate_ptr = qkv_proj_buf;
        let k_raw_ptr = qkv_proj_buf + (q_gate_dim * 2) as u64;
        let v_ptr = qkv_proj_buf + ((q_gate_dim + kv_dim) * 2) as u64;
        let kv_layout = &self.kv.layers[self.fa.kv_layer_ordinal].layout;

        for step in 0..k {
            let positions_ptr = meta_base + ((1 + 3 * step) * 4) as u64;
            let slot_ptr = meta_base + ((2 + 3 * step) * 4) as u64;
            let after_ptr = meta_base + ((3 + 3 * step) * 4) as u64;
            let out_id = self.graph_io.out_ids_dev.addr() + (step * 4) as u64;
            let attn_batch = W34AttentionBatch {
                query_start_loc: qs_ptr as *const i32,
                context_lens_before: positions_ptr as *const i32,
                seq_lens_after: after_ptr as *const i32,
                block_tables: self.pages_dev as *const i32,
                write_slot_mapping: slot_ptr as *const i32,
                valid_token_mask: core::ptr::null(),
                valid_req_mask: core::ptr::null(),
                num_reqs: 1,
                total_tokens: 1,
                max_blocks_per_req: self.pages.len() as i32,
                max_q_len: 1,
                max_seq_len: if long_ctx {
                    self.capacity as i32
                } else {
                    self.capacity.min(1024) as i32
                },
                mask_mode: W34_ATTN_MASK_CAUSAL,
                sliding_window: 0,
                sm_scale: (head_dim as f32).powf(-0.5),
                reserved: 0,
            };
            check_rc(
                // SAFETY: token_ids_ptr contains valid token ID, embed_tokens_ptr is valid device memory of vocab_size * h_dim half elements, embed has capacity for h_dim half elements.
                unsafe {
                    ffi::w34_embed_lookup_f16(
                        &ex,
                        token_ids_ptr as *const i32,
                        t.embed_tokens_ptr as *const c_void,
                        embed as *mut c_void,
                        core::ptr::null(),
                        1,
                        h_dim as i32,
                        t.cfg.vocab_size as i32,
                    )
                },
                "MTP embed",
            )?;
            for (input, weight, output) in [
                (embed, self.norms[0], concat),
                (
                    self.hidden.addr(),
                    self.norms[1],
                    concat + (h_dim * 2) as u64,
                ),
            ] {
                check_rc(
                    // SAFETY: input and output are valid device pointers with capacity for h_dim half elements, weight is valid norm weight.
                    unsafe {
                        ffi::w34_gemma_rms_norm_f16(
                            &ex,
                            input as *const c_void,
                            weight as *const c_void,
                            output as *mut c_void,
                            core::ptr::null(),
                            1,
                            h_dim as i32,
                            t.cfg.rms_norm_eps,
                        )
                    },
                    "MTP pre FC norm",
                )?;
            }
            Self::project(self.concat_fc()?, &ex, t, concat, residual_buf, 1)?;
            check_rc(
                // SAFETY: residual_buf and normed_buf are valid device allocations for h_dim half elements, weight is valid norm weight.
                unsafe {
                    ffi::w34_gemma_rms_norm_f16(
                        &ex,
                        residual_buf as *const c_void,
                        self.residual
                            .single_stream_input_norm_ptr()
                            .ok_or_else(|| {
                                loader_err("MTP graph requires single-stream residual")
                            })? as *const c_void,
                        normed_buf as *mut c_void,
                        core::ptr::null(),
                        1,
                        h_dim as i32,
                        t.cfg.rms_norm_eps,
                    )
                },
                "MTP input norm",
            )?;
            let fa = &self.fa;
            Self::project(&fa.qkv_proj, &ex, t, normed_buf, qkv_proj_buf, 1)?;
            check_rc(
                // SAFETY: q_gate_ptr, k_raw_ptr, and output buffers are valid strided device buffers for 1 token, positions_ptr contains valid position on t.stream.
                unsafe {
                    ffi::w34_fused_qk_rmsnorm_rope_gate_strided_f16(
                        &ex,
                        q_gate_ptr as *const c_void,
                        k_raw_ptr as *const c_void,
                        fa.q_norm_ptr as *const c_void,
                        fa.k_norm_ptr as *const c_void,
                        positions_ptr as *const i32,
                        core::ptr::null(),
                        q_buf as *mut c_void,
                        k_buf as *mut c_void,
                        gate_buf as *mut c_void,
                        1,
                        0,
                        local_q_heads as i32,
                        local_kv_heads as i32,
                        head_dim as i32,
                        t.cfg.rotary_dim as i32,
                        t.cfg.mrope_section[0] as i32,
                        t.cfg.mrope_section[1] as i32,
                        t.cfg.mrope_section[2] as i32,
                        qkv_row_stride,
                        qkv_row_stride,
                        t.cfg.rope_theta,
                        t.cfg.rms_norm_eps,
                    )
                },
                "w34_fused_qk_rmsnorm_rope_gate_strided_f16",
            )?;
            check_rc(
                // SAFETY: kv_layout describes valid paged memory, k_buf and v_ptr are valid device buffers with capacity for kv_dim elements.
                unsafe {
                    ffi::w34_kv_cache_write_strided(
                        &ex,
                        kv_layout,
                        &attn_batch,
                        k_buf as *const c_void,
                        v_ptr as *const c_void,
                        kv_dim as i32,
                        qkv_row_stride,
                    )
                },
                "w34_kv_cache_write_strided",
            )?;
            check_rc(
                // SAFETY: kv_layout and attn_batch describe valid paged attention batch, q_buf and attn_out_buf have capacity for local_q_heads * head_dim elements.
                unsafe {
                    ffi::w34_paged_attention_f16(
                        &ex,
                        kv_layout,
                        &attn_batch,
                        q_buf as *const c_void,
                        attn_out_buf as *mut c_void,
                        local_q_heads as i32,
                    )
                },
                "w34_paged_attention_f16",
            )?;
            check_rc(
                // SAFETY: attn_out_buf and gate_buf are valid device buffers of local_q_heads * head_dim half elements, gated_attn_buf has equal capacity.
                unsafe {
                    ffi::w34_apply_attn_output_gate_f16(
                        &ex,
                        attn_out_buf as *const c_void,
                        gate_buf as *const c_void,
                        gated_attn_buf as *mut c_void,
                        core::ptr::null(),
                        1,
                        (local_q_heads * head_dim) as i32,
                    )
                },
                "w34_apply_attn_output_gate_f16",
            )?;
            Self::project(&fa.o_proj, &ex, t, gated_attn_buf, block_out_buf, 1)?;
            let post_ln_ptr = self
                .residual
                .single_stream_post_attn_norm_ptr()
                .ok_or_else(|| loader_err("MTP graph requires single-stream residual"))?;
            if let Some(comm) = tp2 {
                comm.all_reduce_residual_gemma_rms_norm_f16(
                    &ex,
                    block_out_buf,
                    0,
                    residual_buf,
                    post_ln_ptr,
                    normed_buf,
                    0,
                    t.rank as i32,
                    1,
                    h_dim as i32,
                    t.cfg.rms_norm_eps,
                    0,
                    0,
                )?;
            } else {
                check_rc(
                    // SAFETY: block_out_buf and residual_buf are valid device buffers for h_dim half elements, post_ln_ptr is valid norm weight, normed_buf has capacity for h_dim half elements.
                    unsafe {
                        ffi::w34_fused_add_gemma_rms_norm_f16(
                            &ex,
                            block_out_buf as *const c_void,
                            residual_buf as *mut c_void,
                            post_ln_ptr as *const c_void,
                            normed_buf as *mut c_void,
                            core::ptr::null(),
                            1,
                            h_dim as i32,
                            t.cfg.rms_norm_eps,
                        )
                    },
                    "w34_fused_add_gemma_rms_norm_f16 (post_attn)",
                )?;
            }
            match &self.ffn {
                FfnBlockWeights::Dense {
                    gate_up_proj,
                    down_proj,
                } => {
                    Self::project(gate_up_proj, &ex, t, normed_buf, gate_up_buf, 1)?;
                    check_rc(
                        // SAFETY: gate_up_buf and act_buf are valid device buffers for 1 row of local_interm elements on active stream.
                        unsafe {
                            ffi::w34_silu_and_mul_f16(
                                &ex,
                                gate_up_buf as *const c_void,
                                act_buf as *mut c_void,
                                core::ptr::null(),
                                1,
                                local_interm as i32,
                            )
                        },
                        "w34_silu_and_mul_f16",
                    )?;
                    Self::project(down_proj, &ex, t, act_buf, mlp_out_buf, 1)?;
                }
                FfnBlockWeights::Moe(_) => {
                    self.ffn.forward(
                        &ex,
                        &t.cublas,
                        t.marlin_ws.addr(),
                        t.marlin_ws.len_bytes / 4,
                        normed_buf,
                        mlp_out_buf,
                        1,
                        h_dim,
                        local_interm,
                        gate_up_buf,
                        act_buf,
                        moe_router_logits_buf,
                        moe_topk_weights_buf,
                        moe_topk_ids_buf,
                        moe_interm_buf,
                        moe_sh_gu_buf,
                        moe_sh_act_buf,
                        moe_sh_out_buf,
                        moe_sh_gate_buf,
                    )?;
                }
            }
            if let Some(comm) = tp2 {
                comm.all_reduce_residual_gemma_rms_norm_f16(
                    &ex,
                    mlp_out_buf,
                    0,
                    residual_buf,
                    self.final_residual
                        .single_stream_norm_ptr()
                        .ok_or_else(|| loader_err("MTP graph requires single-stream final norm"))?,
                    self.hidden.addr(),
                    0,
                    t.rank as i32,
                    1,
                    h_dim as i32,
                    t.cfg.rms_norm_eps,
                    0,
                    0,
                )?;
            } else {
                check_rc(
                    // SAFETY: mlp_out_buf and residual_buf are valid device buffers for h_dim half elements, final norm weight is valid, hidden buffer has capacity for h_dim half elements.
                    unsafe {
                        ffi::w34_fused_add_gemma_rms_norm_f16(
                            &ex,
                            mlp_out_buf as *const c_void,
                            residual_buf as *mut c_void,
                            self.final_residual
                                .single_stream_norm_ptr()
                                .ok_or_else(|| {
                                    loader_err("MTP graph requires single-stream final norm")
                                })? as *const c_void,
                            self.hidden.addr() as *mut c_void,
                            core::ptr::null(),
                            1,
                            h_dim as i32,
                            t.cfg.rms_norm_eps,
                        )
                    },
                    "MTP final norm",
                )?;
            }
            if step + 1 == k {
                check_rc(
                    // SAFETY: batch_hidden and hidden are valid device buffers with capacity for h_dim * 2 bytes on t.stream.
                    unsafe {
                        ffi::w34_memcpy_d2d_async(
                            self.batch_hidden.addr() as *mut c_void,
                            self.hidden.addr() as *const c_void,
                            h_dim * 2,
                            t.stream.handle,
                        )
                    },
                    "MTP batch hidden tail",
                )?;
            }
            Self::project(&t.lm_head, &ex, t, self.hidden.addr(), logits_buf, 1)?;
            check_rc(
                // SAFETY: logits_buf contains local_vocab_size valid half logits, output pointers have capacity for argmax results.
                unsafe {
                    ffi::w34_sampler_greedy_top5_f16(
                        &ex,
                        logits_buf as *const c_void,
                        if tp2.is_some() { local_id } else { out_id } as *mut i32,
                        core::ptr::null_mut(),
                        if tp2.is_some() {
                            local_val as *mut f32
                        } else {
                            core::ptr::null_mut()
                        },
                        1,
                        t.local_vocab_size as i32,
                        t.vocab_offset as i32,
                    )
                },
                "MTP greedy",
            )?;
            if let Some(comm) = tp2 {
                comm.all_gather_top1_f32_i32(
                    &ex,
                    local_val,
                    local_id,
                    global_val,
                    out_id,
                    t.rank as i32,
                    1,
                )?;
            }
            if step + 1 < k {
                check_rc(
                    // SAFETY: token_ids_ptr and out_id are valid device pointers with capacity for 4 bytes on t.stream.
                    unsafe {
                        ffi::w34_memcpy_d2d_async(
                            token_ids_ptr as *mut c_void,
                            out_id as *const c_void,
                            4,
                            t.stream.handle,
                        )
                    },
                    "MTP recurrent draft token",
                )?;
            }
        }
        Ok(())
    }
    pub fn ensure_propose_graph(
        &mut self,
        target: &Qwen3_5ModelRank,
        k: usize,
        tp2: Option<&Tp2Collectives>,
    ) -> Result<()> {
        if self.has_propose_graph(k) {
            return Ok(());
        }
        let long_ctx = self.len + k > 1024;
        target.device.make_current()?;
        target.stream.synchronize()?;
        let graph = CudaGraph::capture(&target.stream, || {
            self.enqueue_propose_steps(target, k, long_ctx, tp2)
        })?;
        let idx = if self.batch_mode { self.active_slot } else { 0 };
        self.graph_io.propose_graphs[idx] = Some((k, long_ctx, graph));
        Ok(())
    }
    pub fn prepare_propose_graph_inputs_async(
        &mut self,
        target: &Qwen3_5ModelRank,
        anchor: i32,
        target_hidden: u64,
        k: usize,
    ) -> Result<()> {
        self.validate_propose_args(target, anchor, target_hidden, k, None)?;
        target.device.make_current()?;
        let mut meta = [0i32; 12];
        meta[0] = anchor;
        for s in 0..3 {
            let p = self.len + s.min(k - 1);
            meta[1 + 3 * s] = p as i32;
            meta[2 + 3 * s] = self.pages[p / 16] * 16 + (p % 16) as i32;
            meta[3 + 3 * s] = (p + 1) as i32;
        }
        meta[10] = 0;
        meta[11] = 1;
        self.graph_io
            .meta_dev
            .copy_from_host(&meta, &target.stream)?;
        if target_hidden != self.hidden.addr() {
            check_rc(
                // SAFETY: hidden and target_hidden are valid device buffers with capacity for hidden_size * 2 bytes on target.stream.
                unsafe {
                    ffi::w34_memcpy_d2d_async(
                        self.hidden.addr() as *mut c_void,
                        target_hidden as *const c_void,
                        self.hidden_size * 2,
                        target.stream.handle,
                    )
                },
                "MTP seed hidden",
            )?;
        }
        Ok(())
    }
    pub fn replay_propose_graph_async(&self, target: &Qwen3_5ModelRank, k: usize) -> Result<()> {
        let long_ctx = self.len + k > 1024;
        target.device.make_current()?;
        let idx = if self.batch_mode { self.active_slot } else { 0 };
        let (_, _, graph) = self.graph_io.propose_graphs[idx]
            .as_ref()
            .filter(|(cap_k, cap_long, _)| *cap_k == k && *cap_long == long_ctx)
            .ok_or_else(|| loader_err("MTP propose graph not captured"))?;
        graph.launch(&target.stream)
    }
    pub fn launch_propose_graph_async(
        &mut self,
        target: &Qwen3_5ModelRank,
        anchor: i32,
        target_hidden: u64,
        k: usize,
    ) -> Result<()> {
        self.prepare_propose_graph_inputs_async(target, anchor, target_hidden, k)?;
        self.replay_propose_graph_async(target, k)
    }
    pub fn collect_propose_host(
        &mut self,
        target: &Qwen3_5ModelRank,
        k: usize,
    ) -> Result<Vec<i32>> {
        target.device.make_current()?;
        let mut out = vec![0i32; k];
        self.graph_io
            .out_ids_dev
            .copy_to_host(&mut out, &target.stream)?;
        self.batch_hidden_rows = 1;
        self.len += k;
        Ok(out)
    }
    /// Anchor is token[p+1], target_hidden is target final hidden[p], and len must
    /// equal p. Returns k genuine greedy MTP drafts; cache grows by k provisional
    /// rows. The anchor itself is not returned. Truncate to the round's starting
    /// length and ingest accepted target pairs before the next proposal round.
    pub fn propose(
        &mut self,
        target: &Qwen3_5ModelRank,
        anchor: i32,
        target_hidden: u64,
        k: usize,
        tp2: Option<&Tp2Collectives>,
    ) -> Result<Vec<i32>> {
        if !(1..=3).contains(&k) || self.len != self.committed || self.len + k > self.capacity {
            return Err(loader_err("invalid MTP proposal state or k"));
        }
        if self.use_cuda_graph && self.supports_cuda_graph() {
            self.validate_propose_args(target, anchor, target_hidden, k, Some(tp2.is_some()))?;
            self.ensure_propose_graph(target, k, tp2)?;
            self.launch_propose_graph_async(target, anchor, target_hidden, k)?;
            return self.collect_propose_host(target, k);
        }
        let mut token = anchor;
        let mut hidden = target_hidden;
        let mut out = Vec::with_capacity(k);
        let step0_group = self.len / 4;
        for step in 0..k {
            if let Some(qsa_ws) = &mut self.qsa {
                qsa_ws.step0_page_idx = step0_group;
                qsa_ws.skip_topk = step > 0;
            }
            let res = self.forward(target, &[token], hidden, tp2);
            if let Some(qsa_ws) = &mut self.qsa {
                qsa_ws.skip_topk = false;
            }
            token = match res {
                Ok(token) => token,
                Err(error) => {
                    self.len = self.committed;
                    return Err(error);
                }
            };
            out.push(token);
            hidden = self.hidden.addr();
        }
        Ok(out)
    }
    // Shared LinearProj dispatches small-N dense projections via w34_small_n_gemm_f16.
    // MTP keeps cuBLAS GEMM for DenseF16 (matching vLLM donor M=5 prefill) while
    // dispatching MarlinW4A16 projections for quantized MTP checkpoints.
    fn project(
        proj: &LinearProj,
        ex: &W34Exec,
        t: &Qwen3_5ModelRank,
        input: u64,
        output: u64,
        rows: usize,
    ) -> Result<()> {
        match *proj {
            LinearProj::DenseF16 {
                weight_ptr,
                size_n,
                size_k,
            } => t.cublas.gemm_f16(
                input,
                weight_ptr,
                output,
                rows as i32,
                size_n as i32,
                size_k as i32,
            ),
            LinearProj::MarlinW4A16 { .. }
            | LinearProj::MarlinW4A16Zp { .. }
            | LinearProj::MarlinW8A16 { .. } => proj.forward(
                ex,
                &t.cublas,
                t.marlin_ws.addr(),
                t.marlin_ws.len_bytes / 4,
                input,
                output,
                rows,
            ),
        }
    }
    fn forward(
        &mut self,
        target: &Qwen3_5ModelRank,
        tokens: &[i32],
        input_hidden: u64,
        tp2: Option<&Tp2Collectives>,
    ) -> Result<i32> {
        self.forward_inner(target, tokens, input_hidden, tp2, true, true, false)?
            .ok_or_else(|| loader_err("MTP forward did not sample token"))
    }
    #[allow(clippy::too_many_arguments, clippy::cognitive_complexity)]
    fn forward_inner(
        &mut self,
        target: &Qwen3_5ModelRank,
        tokens: &[i32],
        input_hidden: u64,
        tp2: Option<&Tp2Collectives>,
        sample_last: bool,
        sync_stream: bool,
        kv_only: bool,
    ) -> Result<Option<i32>> {
        let t = target;
        if t.embed_tokens_ptr != self.target_embedding {
            return Err(loader_err("MTP target rank mismatch"));
        }
        let num_tokens = tokens.len();
        let span = num_tokens
            .checked_mul(self.hidden_size)
            .and_then(|n| n.checked_mul(2));
        if num_tokens == 0
            || num_tokens > self.max_batch_rows
            || num_tokens > self.capacity - self.len
            || tokens
                .iter()
                .any(|&id| id < 0 || id as usize >= t.cfg.vocab_size)
            || input_hidden == 0
            || !input_hidden.is_multiple_of(2)
            || span
                .and_then(|n| input_hidden.checked_add(n as u64))
                .is_none()
            || (t.tp_size > 1) != tp2.is_some()
        {
            return Err(loader_err("invalid MTP batch or input span"));
        }
        t.device.make_current()?;
        self.scratch.reset();
        let h_dim = t.cfg.hidden_size;
        let hc_count = self.hidden_size / h_dim;
        let hc_lowrank = t.cfg.hc_lowrank.max(1);
        let head_dim = t.cfg.head_dim;
        let local_q_heads = t.cfg.num_attention_heads / t.tp_size;
        let local_kv_heads = t.cfg.local_kv_heads(t.tp_size);
        let local_interm = t.cfg.intermediate_size / t.tp_size;
        let q_gate_dim = local_q_heads * head_dim * 2;
        let kv_dim = local_kv_heads * head_dim;

        let embed = self.scratch.alloc_aligned(num_tokens * h_dim * 2)?;
        let normed_buf = self.scratch.alloc_aligned(num_tokens * h_dim * 2)?;
        let fc_embed_buf = self.scratch.alloc_aligned(num_tokens * h_dim * 2)?;
        let concat = self.scratch.alloc_aligned(num_tokens * 2 * h_dim * 2)?;
        let hc_norm_buf = self
            .scratch
            .alloc_aligned(num_tokens * hc_count * h_dim * 2)?;
        let hc_residual_buf = self
            .scratch
            .alloc_aligned(num_tokens * hc_count * h_dim * 2)?;
        let hc_inj_weights_buf = self.scratch.alloc_aligned(num_tokens * hc_count * 4)?;
        let hc_scratch_a = self
            .scratch
            .alloc_aligned(num_tokens * hc_count * h_dim * 2)?;
        let hc_scratch_b = self
            .scratch
            .alloc_aligned(num_tokens * hc_count * h_dim * 2)?;
        let hc_scratch_c = self
            .scratch
            .alloc_aligned(num_tokens * (hc_lowrank + hc_count) * 2)?;
        let qkv_proj_buf = self
            .scratch
            .alloc_aligned(num_tokens * (q_gate_dim + 2 * kv_dim) * 2)?;
        let q_buf = self
            .scratch
            .alloc_aligned(num_tokens * (local_q_heads * head_dim) * 2)?;
        let k_buf = self.scratch.alloc_aligned(num_tokens * kv_dim * 2)?;
        let gate_buf = self
            .scratch
            .alloc_aligned(num_tokens * (local_q_heads * head_dim) * 2)?;
        let attn_out_buf = self
            .scratch
            .alloc_aligned(num_tokens * (local_q_heads * head_dim) * 2)?;
        let gated_attn_buf = self
            .scratch
            .alloc_aligned(num_tokens * (local_q_heads * head_dim) * 2)?;
        let block_out_buf = self.scratch.alloc_aligned(num_tokens * h_dim * 2)?;
        let mlp_out_buf = self.scratch.alloc_aligned(num_tokens * h_dim * 2)?;

        let qsa_qk_cols = (t.cfg.indexer_n_heads + t.cfg.indexer_kv_heads) * t.cfg.indexer_head_dim;
        let qsa_qk_buf = if qsa_qk_cols > 0 {
            self.scratch.alloc_aligned(num_tokens * qsa_qk_cols * 2)?
        } else {
            0
        };
        let qsa_q_buf = if qsa_qk_cols > 0 {
            self.scratch
                .alloc_aligned(num_tokens * t.cfg.indexer_n_heads * t.cfg.indexer_head_dim * 2)?
        } else {
            0
        };
        let qsa_k_buf = if qsa_qk_cols > 0 {
            self.scratch
                .alloc_aligned(num_tokens * t.cfg.indexer_kv_heads * t.cfg.indexer_head_dim * 2)?
        } else {
            0
        };

        let (
            gate_up_buf,
            act_buf,
            moe_router_logits_buf,
            moe_topk_weights_buf,
            moe_topk_ids_buf,
            moe_interm_buf,
            moe_sh_gu_buf,
            moe_sh_act_buf,
            moe_sh_out_buf,
            moe_sh_gate_buf,
        ) = match &self.ffn {
            FfnBlockWeights::Dense { .. } => (
                self.scratch
                    .alloc_aligned(num_tokens * (2 * local_interm) * 2)?,
                self.scratch.alloc_aligned(num_tokens * local_interm * 2)?,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
            ),
            FfnBlockWeights::Moe(moe) => {
                let sh_gu = if moe.local_shared_interm > 0 {
                    self.scratch
                        .alloc_aligned(num_tokens * 2 * moe.local_shared_interm * 2)?
                } else {
                    0
                };
                let sh_act = if moe.local_shared_interm > 0 {
                    self.scratch
                        .alloc_aligned(num_tokens * moe.local_shared_interm * 2)?
                } else {
                    0
                };
                let sh_out = if moe.local_shared_interm > 0 {
                    self.scratch.alloc_aligned(num_tokens * h_dim * 2)?
                } else {
                    0
                };
                let sh_gate = if moe.shared_expert_gate.is_some() {
                    self.scratch.alloc_aligned(num_tokens * 2)?
                } else {
                    0
                };
                let moe_interm =
                    if matches!(moe.experts, crate::model::MoeExpertStorage::Tiered { .. }) {
                        0
                    } else {
                        self.scratch
                            .alloc_aligned(num_tokens * moe.top_k * moe.local_moe_interm * 2)?
                    };
                (
                    0,
                    0,
                    self.scratch
                        .alloc_aligned(num_tokens * moe.num_experts * 2)?,
                    self.scratch.alloc_aligned(num_tokens * moe.top_k * 4)?,
                    self.scratch.alloc_aligned(num_tokens * moe.top_k * 4)?,
                    moe_interm,
                    sh_gu,
                    sh_act,
                    sh_out,
                    sh_gate,
                )
            }
        };

        let mut meta = Vec::with_capacity(3 * num_tokens + 3);
        meta.extend_from_slice(tokens);
        meta.extend((self.len..self.len + num_tokens).map(|p| p as i32));
        meta.extend(
            (self.len..self.len + num_tokens).map(|p| self.pages[p / 16] * 16 + (p % 16) as i32),
        );
        meta.extend_from_slice(&[0i32, num_tokens as i32, (self.len + num_tokens) as i32]);
        let meta_base = self.scratch.upload_slice(&meta, &t.stream)?;
        let (ws_ptr, ws_cap) = self.scratch.remaining_workspace();
        let ex = t.stream.exec(ws_ptr, ws_cap);
        let token_ids_ptr = meta_base;
        let positions_ptr = token_ids_ptr + (num_tokens * 4) as u64;
        let slot = positions_ptr + (num_tokens * 4) as u64;
        let qs = slot + (num_tokens * 4) as u64;
        let after = qs + 8;
        let attn_batch = &W34AttentionBatch {
            query_start_loc: qs as *const i32,
            context_lens_before: positions_ptr as *const i32,
            seq_lens_after: after as *const i32,
            block_tables: self.pages_dev as *const i32,
            write_slot_mapping: slot as *const i32,
            valid_token_mask: core::ptr::null(),
            valid_req_mask: core::ptr::null(),
            num_reqs: 1,
            total_tokens: num_tokens as i32,
            max_blocks_per_req: self.pages.len() as i32,
            max_q_len: num_tokens as i32,
            max_seq_len: (self.len + num_tokens) as i32,
            mask_mode: W34_ATTN_MASK_CAUSAL,
            sliding_window: 0,
            sm_scale: (head_dim as f32).powf(-0.5),
            reserved: 0,
        };

        // Predictor input fusion: concat-linear or dual-stream broadcast-add.
        check_rc(
            // SAFETY: token_ids_ptr contains num_tokens valid token IDs, embed_tokens_ptr is valid device memory.
            unsafe {
                ffi::w34_embed_lookup_f16(
                    &ex,
                    token_ids_ptr as *const i32,
                    t.embed_tokens_ptr as *const c_void,
                    embed as *mut c_void,
                    core::ptr::null(),
                    num_tokens as i32,
                    h_dim as i32,
                    t.cfg.vocab_size as i32,
                )
            },
            "MTP embed",
        )?;
        match &self.input_fusion {
            MtpInputFusion::ConcatLinear { fc } => {
                for (input, weight, output) in [
                    (embed, self.norms[0], normed_buf),
                    (input_hidden, self.norms[1], hc_norm_buf),
                ] {
                    check_rc(
                        // SAFETY: inputs and outputs contain num_tokens * h_dim FP16 elements.
                        unsafe {
                            ffi::w34_gemma_rms_norm_f16(
                                &ex,
                                input as *const c_void,
                                weight as *const c_void,
                                output as *mut c_void,
                                core::ptr::null(),
                                num_tokens as i32,
                                h_dim as i32,
                                t.cfg.rms_norm_eps,
                            )
                        },
                        "MTP pre FC norm",
                    )?;
                }
                check_rc(
                    // SAFETY: concat has room for both normalized num_tokens * h_dim inputs.
                    unsafe {
                        ffi::w34_concat_cols_2_f16(
                            &ex,
                            normed_buf as *const c_void,
                            hc_norm_buf as *const c_void,
                            concat as *mut c_void,
                            num_tokens as i32,
                            h_dim as i32,
                            h_dim as i32,
                        )
                    },
                    "MTP concat",
                )?;
                Self::project(fc, &ex, t, concat, hc_residual_buf, num_tokens)?;
            }
            MtpInputFusion::DualStreamBroadcast {
                fc_embedding,
                fc_hidden,
            } => {
                check_rc(
                    // SAFETY: embed and normed_buf hold num_tokens * h_dim FP16 elements.
                    unsafe {
                        ffi::w34_gemma_rms_norm_f16(
                            &ex,
                            embed as *const c_void,
                            self.norms[0] as *const c_void,
                            normed_buf as *mut c_void,
                            core::ptr::null(),
                            num_tokens as i32,
                            h_dim as i32,
                            t.cfg.rms_norm_eps,
                        )
                    },
                    "MTP pre_fc_norm_embedding",
                )?;
                Self::project(fc_embedding, &ex, t, normed_buf, fc_embed_buf, num_tokens)?;
                check_rc(
                    // SAFETY: input_hidden and hc_norm_buf hold num_tokens * hc_count * h_dim FP16 elements.
                    unsafe {
                        ffi::w34_gemma_rms_norm_f16(
                            &ex,
                            input_hidden as *const c_void,
                            self.norms[1] as *const c_void,
                            hc_norm_buf as *mut c_void,
                            core::ptr::null(),
                            num_tokens as i32,
                            (hc_count * h_dim) as i32,
                            t.cfg.rms_norm_eps,
                        )
                    },
                    "MTP pre_fc_norm_hidden",
                )?;
                Self::project(
                    fc_hidden,
                    &ex,
                    t,
                    hc_norm_buf,
                    hc_residual_buf,
                    num_tokens * hc_count,
                )?;
                check_rc(
                    // SAFETY: fc_embed_buf holds [num_tokens, h_dim] and hc_residual_buf holds [num_tokens * hc_count, h_dim].
                    unsafe {
                        ffi::w34_qwen4_mtp_broadcast_add_f16(
                            &ex,
                            fc_embed_buf as *const c_void,
                            hc_residual_buf as *const c_void,
                            hc_residual_buf as *mut c_void,
                            num_tokens as i32,
                            hc_count as i32,
                            h_dim as i32,
                        )
                    },
                    "w34_qwen4_mtp_broadcast_add_f16",
                )?;
            }
        }

        // Normalize / mix the residual before attention.
        let marlin_ws_ptr = t.marlin_ws.addr();
        let marlin_ws_ints = t.marlin_ws.len_bytes / 4;
        let hc_ctx = HcStepContext {
            ex: &ex,
            cublas: &t.cublas,
            marlin_ws_ptr,
            marlin_ws_ints,
            residual_buf: hc_residual_buf,
            hc_residual_buf,
            hc_inj_weights_buf,
            ple_bufs: [hc_scratch_a, hc_scratch_b, hc_scratch_c],
            num_tokens,
            hc_count,
            hc_lowrank,
            h_dim,
            eps: t.cfg.rms_norm_eps,
        };
        self.residual.pre_attn(0, &hc_ctx, normed_buf)?;

        // Full attention, with optional QSA indexing and block selection.
        let fa = &self.fa;
        Self::project(&fa.qkv_proj, &ex, t, normed_buf, qkv_proj_buf, num_tokens)?;
        let qkv_row_stride = (q_gate_dim + 2 * kv_dim) as i32;
        let q_gate_ptr = qkv_proj_buf;
        let k_raw_ptr = qkv_proj_buf + (q_gate_dim * 2) as u64;
        let v_ptr = qkv_proj_buf + ((q_gate_dim + kv_dim) * 2) as u64;
        check_rc(
            // SAFETY: q_gate_ptr, k_raw_ptr, and output buffers are valid strided device buffers.
            unsafe {
                ffi::w34_fused_qk_rmsnorm_rope_gate_strided_f16(
                    &ex,
                    q_gate_ptr as *const c_void,
                    k_raw_ptr as *const c_void,
                    fa.q_norm_ptr as *const c_void,
                    fa.k_norm_ptr as *const c_void,
                    positions_ptr as *const i32,
                    core::ptr::null(),
                    q_buf as *mut c_void,
                    k_buf as *mut c_void,
                    gate_buf as *mut c_void,
                    num_tokens as i32,
                    0,
                    local_q_heads as i32,
                    local_kv_heads as i32,
                    head_dim as i32,
                    t.cfg.rotary_dim as i32,
                    t.cfg.mrope_section[0] as i32,
                    t.cfg.mrope_section[1] as i32,
                    t.cfg.mrope_section[2] as i32,
                    qkv_row_stride,
                    qkv_row_stride,
                    t.cfg.rope_theta,
                    t.cfg.rms_norm_eps,
                )
            },
            "w34_fused_qk_rmsnorm_rope_gate_strided_f16 (MTP)",
        )?;
        let kv_bufs = &self.kv.layers[fa.kv_layer_ordinal];
        let kv_layout = &kv_bufs.layout;
        check_rc(
            // SAFETY: kv_layout describes valid paged memory, k_buf and v_ptr are valid device buffers.
            unsafe {
                ffi::w34_kv_cache_write_strided(
                    &ex,
                    kv_layout,
                    attn_batch,
                    k_buf as *const c_void,
                    v_ptr as *const c_void,
                    kv_dim as i32,
                    qkv_row_stride,
                )
            },
            "w34_kv_cache_write_strided (MTP)",
        )?;
        let reuse_step0_topk = self.qsa.as_ref().is_some_and(|helpers| {
            helpers.skip_topk && num_tokens == 1 && self.len / 4 == helpers.step0_page_idx
        });
        let select_k_blocks =
            (t.cfg.indexer_budget / t.cfg.indexer_compress_ratio.max(1)).clamp(1, 512) + 1;
        if let Some(qsa) = &fa.qsa_indexer {
            let qsa_ws = self
                .qsa
                .as_ref()
                .ok_or_else(|| loader_err("missing MTP QSA helpers"))?;
            qsa.index_qk_proj.forward(
                &ex,
                &t.cublas,
                marlin_ws_ptr,
                marlin_ws_ints,
                normed_buf,
                qsa_qk_buf,
                num_tokens,
            )?;
            crate::ple::gpu_ops::qsa_indexer_norm_rope_f16(
                &ex,
                qsa_qk_buf,
                qsa.q_layernorm_ptr,
                qsa.k_layernorm_ptr,
                positions_ptr,
                qsa_q_buf,
                qsa_k_buf,
                0,
                num_tokens,
                t.cfg.indexer_n_heads,
                t.cfg.indexer_kv_heads,
                t.cfg.indexer_head_dim,
                t.cfg.rotary_dim,
                t.cfg.rope_theta,
                t.cfg.rms_norm_eps,
                t.cfg.mrope_section[0],
                t.cfg.mrope_section[1],
                0,
            )?;
            let raw_k_cache_ptr = kv_bufs
                .qsa_raw_k_buf
                .as_ref()
                .ok_or_else(|| loader_err("missing MTP qsa_raw_k_buf"))?
                .addr();
            let comp_k_cache_ptr = kv_bufs
                .qsa_comp_k_buf
                .as_ref()
                .ok_or_else(|| loader_err("missing MTP qsa_comp_k_buf"))?
                .addr();
            crate::ple::gpu_ops::qsa_indexer_cache_update_f16(
                &ex,
                qsa_k_buf,
                raw_k_cache_ptr,
                comp_k_cache_ptr,
                qsa.k_layernorm_ptr,
                positions_ptr,
                slot,
                0,
                num_tokens,
                t.cfg.indexer_head_dim,
                t.cfg.rotary_dim,
                t.cfg.rope_theta,
                t.cfg.rms_norm_eps,
                t.cfg.mrope_section[0],
                t.cfg.mrope_section[1],
                0,
            )?;
            if !kv_only && !reuse_step0_topk {
                crate::ple::gpu_ops::qsa_select_blocks_f16(
                    &ex,
                    qsa_q_buf,
                    comp_k_cache_ptr,
                    attn_batch,
                    qsa_ws.qsa_sel_phys_buf.addr(),
                    qsa_ws.qsa_sel_log_buf.addr(),
                    qsa_ws.qsa_num_sel_buf.addr(),
                    qsa_ws.qsa_seq_pos_buf.addr(),
                    t.cfg.indexer_n_heads,
                    t.cfg.indexer_head_dim,
                    16,
                    select_k_blocks,
                    0,
                    0,
                )?;
            }
        }
        if kv_only {
            if sync_stream {
                t.stream.synchronize()?;
            }
            self.batch_hidden_rows = 0;
            self.len += num_tokens;
            return Ok(None);
        }
        if fa.qsa_indexer.is_some() {
            let qsa_ws = self
                .qsa
                .as_ref()
                .ok_or_else(|| loader_err("missing MTP QSA helpers"))?;
            let seq_pos_ptr = if reuse_step0_topk {
                positions_ptr
            } else {
                qsa_ws.qsa_seq_pos_buf.addr()
            };
            crate::ple::gpu_ops::qsa_sparse_paged_attn_f16(
                &ex,
                q_buf,
                gate_buf,
                kv_layout,
                qsa_ws.qsa_sel_phys_buf.addr(),
                qsa_ws.qsa_sel_log_buf.addr(),
                qsa_ws.qsa_num_sel_buf.addr(),
                seq_pos_ptr,
                0,
                gated_attn_buf,
                num_tokens,
                local_q_heads,
                select_k_blocks,
                attn_batch.sm_scale,
            )?;
        } else {
            check_rc(
                // SAFETY: `kv_layout`, `attn_batch`, `q_buf`, and `attn_out_buf` are valid
                // device allocations for `num_tokens`.
                unsafe {
                    ffi::w34_paged_attention_f16(
                        &ex,
                        kv_layout,
                        attn_batch,
                        q_buf as *const c_void,
                        attn_out_buf as *mut c_void,
                        local_q_heads as i32,
                    )
                },
                "w34_paged_attention_f16 (MTP)",
            )?;
            check_rc(
                // SAFETY: `attn_out_buf`, `gate_buf`, and `gated_attn_buf` are valid device
                // buffers for `num_tokens * local_q_heads * head_dim` FP16 elements.
                unsafe {
                    ffi::w34_apply_attn_output_gate_f16(
                        &ex,
                        attn_out_buf as *const c_void,
                        gate_buf as *const c_void,
                        gated_attn_buf as *mut c_void,
                        core::ptr::null(),
                        num_tokens as i32,
                        (local_q_heads * head_dim) as i32,
                    )
                },
                "w34_apply_attn_output_gate_f16 (MTP)",
            )?;
        }
        Self::project(
            &fa.o_proj,
            &ex,
            t,
            gated_attn_buf,
            block_out_buf,
            num_tokens,
        )?;
        self.residual
            .post_attn_pre_mlp(&hc_ctx, tp2, t.rank, block_out_buf, normed_buf, false)?;

        // Keep the concat predictor's donor-compatible cuBLAS dense FFN path.
        if let FfnBlockWeights::Dense {
            gate_up_proj,
            down_proj,
        } = &self.ffn
            && matches!(self.input_fusion, MtpInputFusion::ConcatLinear { .. })
        {
            Self::project(gate_up_proj, &ex, t, normed_buf, gate_up_buf, num_tokens)?;
            check_rc(
                // SAFETY: gate_up_buf and act_buf contain num_tokens rows of 2*local_interm and local_interm FP16 elements.
                unsafe {
                    ffi::w34_silu_and_mul_f16(
                        &ex,
                        gate_up_buf as *const c_void,
                        act_buf as *mut c_void,
                        core::ptr::null(),
                        num_tokens as i32,
                        local_interm as i32,
                    )
                },
                "w34_silu_and_mul_f16",
            )?;
            Self::project(down_proj, &ex, t, act_buf, mlp_out_buf, num_tokens)?;
        } else {
            self.ffn.forward(
                &ex,
                &t.cublas,
                marlin_ws_ptr,
                marlin_ws_ints,
                normed_buf,
                mlp_out_buf,
                num_tokens,
                h_dim,
                local_interm,
                gate_up_buf,
                act_buf,
                moe_router_logits_buf,
                moe_topk_weights_buf,
                moe_topk_ids_buf,
                moe_interm_buf,
                moe_sh_gu_buf,
                moe_sh_act_buf,
                moe_sh_out_buf,
                moe_sh_gate_buf,
            )?;
        }
        let tiered_info = match &self.ffn {
            FfnBlockWeights::Moe(moe)
                if matches!(moe.experts, crate::model::MoeExpertStorage::Tiered { .. }) =>
            {
                Some((
                    moe.top_k,
                    if moe.shared_down_proj.is_some() {
                        moe_sh_out_buf
                    } else {
                        0
                    },
                    if moe.shared_expert_gate.is_some() {
                        moe_sh_gate_buf
                    } else {
                        0
                    },
                ))
            }
            _ => None,
        };
        if let Some((top_k, sh_out_ptr, sh_gate_ptr)) = tiered_info
            && let Some((store, cache, ws)) = &mut self.tiered_moe
        {
            if cache.slots.capacity >= cache.expert_ids.len()
                || (num_tokens * top_k <= cache.slots.capacity && store.mapped_dev_ptr() != 0)
            {
                crate::moe_offload::TieredMoeEngine::execute_layer_device_lru(
                    store,
                    cache,
                    ws,
                    &t.stream,
                    normed_buf,
                    moe_topk_ids_buf,
                    moe_topk_weights_buf,
                    sh_out_ptr,
                    sh_gate_ptr,
                    mlp_out_buf,
                    num_tokens,
                    top_k,
                )?;
            } else {
                let pairs = num_tokens * top_k;
                let mut h_ids = vec![0i32; pairs];
                check_rc(
                    // SAFETY: `h_ids` and `moe_topk_ids_buf` hold `pairs` i32 elements.
                    unsafe {
                        ffi::w34_memcpy_d2h_async(
                            h_ids.as_mut_ptr().cast(),
                            moe_topk_ids_buf as *const c_void,
                            pairs * 4,
                            t.stream.handle,
                        )
                    },
                    "MTP h_ids",
                )?;
                t.stream.synchronize()?;
                crate::moe_offload::TieredMoeEngine::execute_layer(
                    crate::moe_offload::MoeOffloadMode::Auto,
                    store,
                    cache,
                    ws,
                    &t.stream,
                    normed_buf,
                    moe_topk_ids_buf,
                    moe_topk_weights_buf,
                    &h_ids,
                    &[],
                    sh_out_ptr,
                    sh_gate_ptr,
                    mlp_out_buf,
                    num_tokens,
                    top_k,
                )?;
            }
        }
        self.residual.post_mlp(
            &hc_ctx,
            tp2,
            t.rank,
            mlp_out_buf,
            [
                self.final_residual.single_stream_norm_ptr().unwrap_or(0),
                self.batch_hidden.addr(),
            ],
            TapExport::default(),
        )?;
        let multi_row_bytes = hc_count * h_dim * 2;
        if self.final_residual.is_hyper_connection() {
            check_rc(
                // SAFETY: batch_hidden and hc_residual_buf hold num_tokens * multi_row_bytes.
                unsafe {
                    ffi::w34_memcpy_d2d_async(
                        self.batch_hidden.addr() as *mut c_void,
                        hc_residual_buf as *const c_void,
                        num_tokens * multi_row_bytes,
                        t.stream.handle,
                    )
                },
                "MTP batch_hidden copy",
            )?;
        }
        let last_multi_ptr = self.batch_hidden.addr() + ((num_tokens - 1) * multi_row_bytes) as u64;
        check_rc(
            // SAFETY: self.hidden and last_multi_ptr hold multi_row_bytes.
            unsafe {
                ffi::w34_memcpy_d2d_async(
                    self.hidden.addr() as *mut c_void,
                    last_multi_ptr as *const c_void,
                    multi_row_bytes,
                    t.stream.handle,
                )
            },
            "MTP last multi_hidden copy",
        )?;

        // Collapse multi-stream residuals only for the LM head. Single-stream
        // post_mlp has already written the final-normalized recurrent hidden.
        let sampled_id = if sample_last {
            let mut final_hc_ctx = hc_ctx;
            final_hc_ctx.residual_buf = last_multi_ptr;
            final_hc_ctx.hc_residual_buf = last_multi_ptr;
            final_hc_ctx.num_tokens = 1;
            self.final_residual
                .finish_stack(&final_hc_ctx, &t.stream, normed_buf, None)?;
            let logits_buf = self.scratch.alloc_aligned(t.local_vocab_size * 2)?;
            let lm_input = if self.final_residual.is_hyper_connection() {
                normed_buf
            } else {
                last_multi_ptr
            };
            Self::project(&t.lm_head, &ex, t, lm_input, logits_buf, 1)?;
            let out_id = self.scratch.alloc_aligned(4)?;
            let local_id = self.scratch.alloc_aligned(4)?;
            let local_val = self.scratch.alloc_aligned(4)?;
            let global_val = self.scratch.alloc_aligned(4)?;
            check_rc(
                // SAFETY: `logits_buf` holds `local_vocab_size` FP16 elements and `out_id`/
                // `local_id`/`local_val` are 4-byte device allocations.
                unsafe {
                    ffi::w34_sampler_greedy_top5_f16(
                        &ex,
                        logits_buf as *const c_void,
                        if tp2.is_some() { local_id } else { out_id } as *mut i32,
                        core::ptr::null_mut(),
                        if tp2.is_some() {
                            local_val as *mut f32
                        } else {
                            core::ptr::null_mut()
                        },
                        1,
                        t.local_vocab_size as i32,
                        t.vocab_offset as i32,
                    )
                },
                "MTP greedy",
            )?;
            if let Some(comm) = tp2 {
                comm.all_gather_top1_f32_i32(
                    &ex,
                    local_val,
                    local_id,
                    global_val,
                    out_id,
                    t.rank as i32,
                    1,
                )?;
            }
            let mut out = [0i32];
            check_rc(
                // SAFETY: `out` and `out_id` hold 4 bytes and the stream is synchronized next.
                unsafe {
                    ffi::w34_memcpy_d2h_async(
                        out.as_mut_ptr().cast(),
                        out_id as *const c_void,
                        4,
                        t.stream.handle,
                    )
                },
                "MTP greedy copy",
            )?;
            t.stream.synchronize()?;
            Some(out[0])
        } else {
            if sync_stream {
                t.stream.synchronize()?;
            }
            None
        };
        self.batch_hidden_rows = num_tokens;
        self.len += num_tokens;
        Ok(sampled_id)
    }
}
