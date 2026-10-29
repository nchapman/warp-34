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
#![allow(clippy::too_many_arguments, clippy::undocumented_unsafe_blocks)]
use crate::cache::{KvFormatPolicy, KvPool};
use crate::cuda::{CudaGraph, DeviceArena, DeviceBuffer, Result, Tp2Collectives, check_rc};
use crate::loader::{ModelConfig, QuantScheme, SafetensorsArchive};
use crate::model::{
    FfnBlockWeights, FullAttentionLayerWeights, LinearProj, Qwen3_5ModelRank, RawLinearHost,
    load_moe_layer_weights, loader_err,
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
            && c.moe_intermediate_size.is_multiple_of(tp)
            && c.shared_expert_intermediate_size.is_multiple_of(tp)
    };
    if !matches!(tp, 1 | 2)
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
        || !c.num_key_value_heads.is_multiple_of(tp)
        || !c.vocab_size.is_multiple_of(tp)
        || !c.num_attention_heads.is_multiple_of(c.num_key_value_heads)
    {
        return Err(loader_err("unsupported MTP TP geometry"));
    }
    Ok(())
}

struct MtpGraphIo {
    propose_graph: Option<(usize, CudaGraph)>,
    propose_arena: DeviceArena,
    meta_dev: DeviceBuffer,
    out_ids_dev: DeviceBuffer,
}

/// Owns only predictor weights and predictor KV. Pass the same target rank to
/// each operation; it is not borrowed across calls, allowing target verification
/// between rounds. The target must remain loaded (and its config unchanged).
pub struct Qwen3_5MtpRank {
    pub use_cuda_graph: bool,
    target_embedding: u64,
    hidden_size: usize,
    capacity: usize,
    graph_io: MtpGraphIo,
    _weights: DeviceArena,
    scratch: DeviceArena,
    kv: KvPool,
    pages: Vec<i32>,
    pages_dev: u64,
    len: usize,
    committed: usize,
    fc: LinearProj,
    fa: FullAttentionLayerWeights,
    ffn: FfnBlockWeights,
    norms: [u64; 5],
    /// Last MTP final-normalized hidden row, also the recurrent proposal input.
    pub hidden: DeviceBuffer,
    /// Contiguous outputs of the last forward; only batch_hidden_rows rows are live.
    pub batch_hidden: DeviceBuffer,
    pub batch_hidden_rows: usize,
    max_batch_rows: usize,
}
impl Qwen3_5MtpRank {
    pub fn load(target: &Qwen3_5ModelRank, dir: &Path, capacity: usize) -> Result<Self> {
        let c = &target.cfg;
        validate_geometry(c, target.rank, target.tp_size)?;
        if capacity == 0 || capacity > i32::MAX as usize - 16 {
            return Err(loader_err("invalid MTP capacity"));
        }
        target.device.make_current()?;
        let archive = SafetensorsArchive::open(dir).map_err(|_| loader_err("open MTP archive"))?;
        let r = target.rank;
        let tp = target.tp_size;
        let h = c.hidden_size;
        let q = c.num_attention_heads * c.head_dim;
        let kv = c.num_key_value_heads * c.head_dim;
        let i = c.intermediate_size;
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
        let ffn_weight_bytes = if is_moe_mtp {
            let local_moe_i = c.moe_intermediate_size / tp;
            let local_sh_i = c.shared_expert_intermediate_size / tp;
            let experts_are_w4 = archive
                .resolve_name("mtp.layers.0.mlp.experts.0.gate_proj.qweight")
                .is_some()
                || archive
                    .resolve_name("mtp.layers.0.mlp.experts.0.gate_proj.weight_packed")
                    .is_some();
            let expert_elem_bytes = if experts_are_w4 { 1usize } else { 2usize };
            c.num_experts * 3 * h * local_moe_i * expert_elem_bytes
                + 2 * (c.num_experts * h + 3 * h * local_sh_i + h)
                + 16 * 1024 * 1024
        } else {
            2 * 3 * h * (i / tp)
        };
        let bytes = 2
            * (2 * h * h + (2 * q + 2 * kv) * h / tp + h * q / tp + 5 * h + 2 * c.head_dim)
            + ffn_weight_bytes
            + capacity.div_ceil(16) * 4
            + 8 * 1024 * 1024;
        let mut weights = DeviceArena::new(target.device.device_id, bytes)?;
        let staging = DeviceBuffer::alloc(target.device.device_id, 64 * 1024 * 1024)?;
        let default_group_size = match &c.quant {
            QuantScheme::CompressedTensors { group_size, .. }
            | QuantScheme::Gptq { group_size, .. }
            | QuantScheme::Awq { group_size, .. }
            | QuantScheme::AutoRound { group_size, .. } => *group_size,
            QuantScheme::Unquantized => 32,
        };
        let load = |name: &str, n: usize, k: usize| -> Result<RawLinearHost> {
            let raw = RawLinearHost::load(&archive, &format!("mtp.{name}"), default_group_size)?;
            match &raw {
                RawLinearHost::DenseF16 { n: rn, k: rk, .. }
                | RawLinearHost::U4B8NMajor { n: rn, k: rk, .. }
                    if *rn == n && *rk == k =>
                {
                    Ok(raw)
                }
                _ => Err(loader_err("MTP requires correctly shaped weights")),
            }
        };
        let fc = load("fc", h, 2 * h)?.upload(&mut weights, &target.stream, &staging)?;
        let qs = load("layers.0.self_attn.q_proj", 2 * q, h)?
            .slice_rows(r * 2 * q / tp, (r + 1) * 2 * q / tp);
        let ks =
            load("layers.0.self_attn.k_proj", kv, h)?.slice_rows(r * kv / tp, (r + 1) * kv / tp);
        let vs =
            load("layers.0.self_attn.v_proj", kv, h)?.slice_rows(r * kv / tp, (r + 1) * kv / tp);
        let qkv_proj = RawLinearHost::concat_rows(&[qs, ks, vs]).upload(
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
                "mtp.layers.0",
                r,
                tp,
                default_group_size,
                &mut weights,
                &target.stream,
                &staging,
            )?))
        } else {
            let gs = load("layers.0.mlp.gate_proj", i, h)?.slice_rows(r * i / tp, (r + 1) * i / tp);
            let us = load("layers.0.mlp.up_proj", i, h)?.slice_rows(r * i / tp, (r + 1) * i / tp);
            let gate_up_proj = RawLinearHost::concat_rows(&[gs, us]).upload(
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
        drop(staging);
        let mut norm = |name: &str, size: usize| -> Result<u64> {
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
        let norms = [
            norm("pre_fc_norm_embedding", h)?,
            norm("pre_fc_norm_hidden", h)?,
            norm("layers.0.input_layernorm", h)?,
            norm("layers.0.post_attention_layernorm", h)?,
            norm("norm", h)?,
        ];
        let fa = FullAttentionLayerWeights {
            layer_id: c.num_hidden_layers,
            kv_layer_ordinal: 0,
            qkv_proj,
            o_proj,
            q_norm_ptr: norm("layers.0.self_attn.q_norm", c.head_dim)?,
            k_norm_ptr: norm("layers.0.self_attn.k_norm", c.head_dim)?,
        };
        let mut kv = KvPool::new(
            target.device.device_id,
            W34_KV_NAMESPACE_MTP_FULL_ATTN,
            &[c.num_hidden_layers],
            c.num_key_value_heads / tp,
            c.head_dim,
            16,
            capacity.div_ceil(16),
            KvFormatPolicy::Fp16,
        )?;
        let pages: Vec<i32> = (0..kv.num_pages)
            .map(|_| kv.alloc_page().unwrap())
            .collect();
        let pages_dev = weights.upload_slice(&pages, &target.stream)?;
        // Bound eager batch memory independently of the KV capacity.
        let max_batch_rows = capacity.min(256);
        // All FP16 intermediates (including split QKV and full vocabulary logits),
        // plus alignment and per-request/token metadata.
        let ffn_row_elements = match &ffn {
            FfnBlockWeights::Dense { .. } => 3 * (i / tp),
            FfnBlockWeights::Moe(moe) => {
                moe.num_experts
                    + 4 * moe.top_k
                    + moe.top_k * moe.local_moe_interm
                    + 3 * moe.local_shared_interm
                    + h
                    + 64
            }
        };
        let row_elements = 9 * h
            + 8 * q / tp
            + 6 * c.num_key_value_heads * c.head_dim / tp
            + ffn_row_elements
            + target.local_vocab_size;
        let scratch_bytes = max_batch_rows
            .checked_mul(row_elements)
            .and_then(|n| n.checked_mul(2))
            .and_then(|n| n.checked_add(131072 + capacity.div_ceil(16) * 4))
            .ok_or_else(|| loader_err("MTP scratch size overflow"))?;
        let propose_arena_bytes = row_elements
            .checked_mul(2)
            .and_then(|n| n.checked_add(131072))
            .ok_or_else(|| loader_err("MTP propose arena overflow"))?;
        let graph_io = MtpGraphIo {
            propose_graph: None,
            propose_arena: DeviceArena::new(target.device.device_id, propose_arena_bytes)?,
            meta_dev: DeviceBuffer::alloc(target.device.device_id, 16 * 4)?,
            out_ids_dev: DeviceBuffer::alloc(target.device.device_id, 4 * 4)?,
        };
        target.stream.synchronize()?;
        Ok(Self {
            use_cuda_graph: true,
            target_embedding: target.embed_tokens_ptr,
            hidden_size: h,
            capacity,
            graph_io,
            _weights: weights,
            scratch: DeviceArena::new(target.device.device_id, scratch_bytes)?,
            kv,
            pages,
            pages_dev,
            len: 0,
            committed: 0,
            fc,
            fa,
            ffn,
            norms,
            hidden: DeviceBuffer::alloc(target.device.device_id, h * 2)?,
            batch_hidden: DeviceBuffer::alloc(target.device.device_id, max_batch_rows * h * 2)?,
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
    /// Discard provisional suffix. Stale bytes remain unreachable and are overwritten.
    /// Even accepted draft rows must be replayed using TARGET hidden rows before
    /// another round: recurrent draft hidden is not the committed target hidden.
    pub fn truncate(&mut self, len: usize) -> Result<()> {
        if len > self.len {
            return Err(loader_err("MTP truncate grows cache"));
        }
        self.len = len;
        self.committed = self.committed.min(len);
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
            self.forward_inner(target, next_tokens, target_hidden, tp2, false, false)?;
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
        self.graph_io
            .propose_graph
            .as_ref()
            .is_some_and(|(cap_k, _)| *cap_k == k)
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
            && (target.tp_size == 2) != has_tp2
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
        tp2: Option<&Tp2Collectives>,
    ) -> Result<()> {
        self.graph_io.propose_arena.reset();
        let ex = t.stream.exec(0, 0);
        let h_dim = t.cfg.hidden_size;
        let head_dim = t.cfg.head_dim;
        let local_q_heads = t.cfg.num_attention_heads / t.tp_size;
        let local_kv_heads = t.cfg.num_key_value_heads / t.tp_size;
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
                (
                    0,
                    0,
                    arena.alloc_aligned(moe.num_experts * 2)?,
                    arena.alloc_aligned(moe.top_k * 4)?,
                    arena.alloc_aligned(moe.top_k * 4)?,
                    arena.alloc_aligned(moe.top_k * moe.local_moe_interm * 2)?,
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
                max_seq_len: self.capacity as i32,
                mask_mode: W34_ATTN_MASK_CAUSAL,
                sliding_window: 0,
                sm_scale: (head_dim as f32).powf(-0.5),
                reserved: 0,
            };
            check_rc(
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
            Self::project(&self.fc, &ex, t, concat, residual_buf, 1)?;
            check_rc(
                unsafe {
                    ffi::w34_gemma_rms_norm_f16(
                        &ex,
                        residual_buf as *const c_void,
                        self.norms[2] as *const c_void,
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
            let post_ln_ptr = self.norms[3];
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
                    self.norms[4],
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
                    unsafe {
                        ffi::w34_fused_add_gemma_rms_norm_f16(
                            &ex,
                            mlp_out_buf as *const c_void,
                            residual_buf as *mut c_void,
                            self.norms[4] as *const c_void,
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
        target.device.make_current()?;
        target.stream.synchronize()?;
        let graph = CudaGraph::capture(&target.stream, || {
            self.enqueue_propose_steps(target, k, tp2)
        })?;
        self.graph_io.propose_graph = Some((k, graph));
        Ok(())
    }
    pub fn launch_propose_graph_async(
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
        let (_, graph) = self
            .graph_io
            .propose_graph
            .as_ref()
            .filter(|(cap_k, _)| *cap_k == k)
            .ok_or_else(|| loader_err("MTP propose graph not captured"))?;
        graph.launch(&target.stream)
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
        if self.use_cuda_graph {
            self.validate_propose_args(target, anchor, target_hidden, k, Some(tp2.is_some()))?;
            self.ensure_propose_graph(target, k, tp2)?;
            self.launch_propose_graph_async(target, anchor, target_hidden, k)?;
            return self.collect_propose_host(target, k);
        }
        let mut token = anchor;
        let mut hidden = target_hidden;
        let mut out = Vec::with_capacity(k);
        for _ in 0..k {
            token = match self.forward(target, &[token], hidden, tp2) {
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
            LinearProj::MarlinW4A16 { .. } => proj.forward(
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
        self.forward_inner(target, tokens, input_hidden, tp2, true, true)?
            .ok_or_else(|| loader_err("MTP forward did not sample token"))
    }
    fn forward_inner(
        &mut self,
        target: &Qwen3_5ModelRank,
        tokens: &[i32],
        input_hidden: u64,
        tp2: Option<&Tp2Collectives>,
        sample_last: bool,
        sync_stream: bool,
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
            || (t.tp_size == 2) != tp2.is_some()
        {
            return Err(loader_err("invalid MTP batch or input span"));
        }
        t.device.make_current()?;
        self.scratch.reset();
        let h_dim = t.cfg.hidden_size;
        let head_dim = t.cfg.head_dim;
        let local_q_heads = t.cfg.num_attention_heads / t.tp_size;
        let local_kv_heads = t.cfg.num_key_value_heads / t.tp_size;
        let local_interm = t.cfg.intermediate_size / t.tp_size;
        let q_gate_dim = local_q_heads * head_dim * 2;
        let kv_dim = local_kv_heads * head_dim;
        let residual_buf = self.scratch.alloc_aligned(num_tokens * h_dim * 2)?;
        let normed_buf = self.scratch.alloc_aligned(num_tokens * h_dim * 2)?;
        let embed = self.scratch.alloc_aligned(num_tokens * h_dim * 2)?;
        let concat = self.scratch.alloc_aligned(num_tokens * (2 * h_dim) * 2)?;
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
                (
                    0,
                    0,
                    self.scratch
                        .alloc_aligned(num_tokens * moe.num_experts * 2)?,
                    self.scratch.alloc_aligned(num_tokens * moe.top_k * 4)?,
                    self.scratch.alloc_aligned(num_tokens * moe.top_k * 4)?,
                    self.scratch
                        .alloc_aligned(num_tokens * moe.top_k * moe.local_moe_interm * 2)?,
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
        check_rc(
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
        if num_tokens == 1 {
            for (input, weight, output) in [
                (embed, self.norms[0], concat),
                (input_hidden, self.norms[1], concat + (h_dim * 2) as u64),
            ] {
                check_rc(
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
        } else {
            let hidden_norm = self.scratch.alloc_aligned(num_tokens * h_dim * 2)?;
            for (input, weight, output) in [
                (embed, self.norms[0], normed_buf),
                (input_hidden, self.norms[1], hidden_norm),
            ] {
                check_rc(
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
                unsafe {
                    ffi::w34_concat_cols_2_f16(
                        &ex,
                        normed_buf as *const c_void,
                        hidden_norm as *const c_void,
                        concat as *mut c_void,
                        num_tokens as i32,
                        h_dim as i32,
                        h_dim as i32,
                    )
                },
                "MTP concat",
            )?;
        }
        Self::project(&self.fc, &ex, t, concat, residual_buf, num_tokens)?;
        check_rc(
            unsafe {
                ffi::w34_gemma_rms_norm_f16(
                    &ex,
                    residual_buf as *const c_void,
                    self.norms[2] as *const c_void,
                    normed_buf as *mut c_void,
                    core::ptr::null(),
                    num_tokens as i32,
                    h_dim as i32,
                    t.cfg.rms_norm_eps,
                )
            },
            "MTP input norm",
        )?;
        let fa = &self.fa;
        Self::project(&fa.qkv_proj, &ex, t, normed_buf, qkv_proj_buf, num_tokens)?;
        let qkv_row_stride = (q_gate_dim + 2 * kv_dim) as i32;
        let q_gate_ptr = qkv_proj_buf;
        let k_raw_ptr = qkv_proj_buf + (q_gate_dim * 2) as u64;
        let v_ptr = qkv_proj_buf + ((q_gate_dim + kv_dim) * 2) as u64;
        check_rc(
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
                    0, // 1D text positions: all 3 mRoPE planes share plane 0
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
        let kv_layout = &self.kv.layers[fa.kv_layer_ordinal].layout;
        check_rc(
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
            "w34_kv_cache_write_strided",
        )?;
        check_rc(
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
            "w34_paged_attention_f16",
        )?;
        check_rc(
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
            "w34_apply_attn_output_gate_f16",
        )?;
        Self::project(
            &fa.o_proj,
            &ex,
            t,
            gated_attn_buf,
            block_out_buf,
            num_tokens,
        )?;
        let post_ln_ptr = self.norms[3];
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
                num_tokens as i32,
                h_dim as i32,
                t.cfg.rms_norm_eps,
                0,
                0,
            )?;
        } else {
            // Post-attention RMSNorm + MLP
            check_rc(
                unsafe {
                    ffi::w34_fused_add_gemma_rms_norm_f16(
                        &ex,
                        block_out_buf as *const c_void,
                        residual_buf as *mut c_void,
                        post_ln_ptr as *const c_void,
                        normed_buf as *mut c_void,
                        core::ptr::null(),
                        num_tokens as i32,
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
                Self::project(gate_up_proj, &ex, t, normed_buf, gate_up_buf, num_tokens)?;
                check_rc(
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
            }
            FfnBlockWeights::Moe(_) => {
                self.ffn.forward(
                    &ex,
                    &t.cublas,
                    t.marlin_ws.addr(),
                    t.marlin_ws.len_bytes / 4,
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
        }
        if let Some(comm) = tp2 {
            comm.all_reduce_residual_gemma_rms_norm_f16(
                &ex,
                mlp_out_buf,
                0,
                residual_buf,
                self.norms[4],
                self.batch_hidden.addr(),
                0,
                t.rank as i32,
                num_tokens as i32,
                h_dim as i32,
                t.cfg.rms_norm_eps,
                0,
                0,
            )?;
        } else {
            check_rc(
                unsafe {
                    ffi::w34_fused_add_gemma_rms_norm_f16(
                        &ex,
                        mlp_out_buf as *const c_void,
                        residual_buf as *mut c_void,
                        self.norms[4] as *const c_void,
                        self.batch_hidden.addr() as *mut c_void,
                        core::ptr::null(),
                        num_tokens as i32,
                        h_dim as i32,
                        t.cfg.rms_norm_eps,
                    )
                },
                "MTP final norm",
            )?;
        }
        let last_hidden_ptr = self.batch_hidden.addr() + ((num_tokens - 1) * h_dim * 2) as u64;
        check_rc(
            unsafe {
                ffi::w34_memcpy_d2d_async(
                    self.hidden.addr() as *mut c_void,
                    last_hidden_ptr as *const c_void,
                    h_dim * 2,
                    t.stream.handle,
                )
            },
            "MTP last hidden",
        )?;
        let sampled_id = if sample_last {
            let logits_buf = self.scratch.alloc_aligned(t.local_vocab_size * 2)?;
            Self::project(&t.lm_head, &ex, t, last_hidden_ptr, logits_buf, 1)?;
            let out_id = self.scratch.alloc_aligned(4)?;
            let local_id = self.scratch.alloc_aligned(4)?;
            let local_val = self.scratch.alloc_aligned(4)?;
            let global_val = self.scratch.alloc_aligned(4)?;
            check_rc(
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
