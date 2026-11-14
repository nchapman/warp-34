//! Per-device model residency, capacity planning, and loading.

pub(super) use super::rank_capacity::*;
use super::*;

#[cfg(feature = "cuda")]
pub(crate) type SpecVerifyGraphKey = (usize, bool, DecodeGraphSampleMode, GraphLoraMode);
#[cfg(feature = "cuda")]
pub(crate) type BatchedSpecVerifyGraphKey = (
    usize,
    usize,
    usize,
    bool,
    DecodeGraphSampleMode,
    GraphLoraMode,
);
#[cfg(feature = "cuda")]
pub(crate) type BatchedDecodeGraphKey = (usize, bool, DecodeGraphSampleMode, GraphLoraMode);

/// Shared PLE lookup data and rank-local staging buffers.
#[cfg(feature = "cuda")]
pub struct PleRankRuntime {
    pub hasher: Arc<crate::ple::PleNgramHasher>,
    pub mmap_table: Arc<crate::ple::PleMmapTable>,
    pub f8_staging_dev: DeviceBuffer,
    pub f8_host_scratch: crate::cuda::MappedHostBuffer,
    pub f8_host_slot: usize,
    pub ids_scratch: Vec<i64>,
    pub slice_tokens_scratch: Vec<i32>,
    pub slice_q_start_scratch: Vec<i32>,
    pub slice_st_in_scratch: Vec<i32>,
    pub slice_st_out_scratch: Vec<i32>,
    pub slice_has_init_scratch: Vec<u8>,
    pub slice_ckpt_dst_scratch: Vec<i32>,
    pub slice_ckpt_off_scratch: Vec<i32>,
}

/// One rank of a Qwen 3.5 / 3.8 model resident on a single GPU.
#[cfg(feature = "cuda")]
pub struct Qwen3_5ModelRank {
    pub rank: usize,
    pub tp_size: usize,
    pub device: CudaDevice,
    pub stream: CudaStream,
    pub cublas: CublasHandle,
    pub cfg: ModelConfig,
    pub weight_arena: DeviceArena,
    pub scratch_arena: DeviceArena,
    pub batch_meta_arena: DeviceArena,
    pub marlin_ws: DeviceBuffer,
    pub embed_tokens_ptr: u64,
    pub host_embed_tokens: Option<crate::cuda::MappedHostBuffer>,
    pub final_residual: FinalResidual,
    pub lm_head: Option<LinearProj>,
    pub local_vocab_size: usize,
    pub vocab_offset: usize,
    pub layers: Vec<DecoderLayerWeights>,
    pub kv_pool: KvPool,
    pub state_pool: StateBundlePool,
    pub decode_io: DecodeGraphIo,
    pub decode_graph: Vec<((bool, GraphLoraMode), u64, CudaGraph)>,
    pub spec_verify_io: SpecVerifyGraphIo,
    pub spec_verify_graphs: Vec<(SpecVerifyGraphKey, u64, CudaGraph)>,
    pub failed_spec_verify_keys: Vec<SpecVerifyGraphKey>,
    pub batched_spec_verify_io: BatchedSpecVerifyGraphIo,
    pub batched_spec_verify_graphs: Vec<(BatchedSpecVerifyGraphKey, u64, CudaGraph)>,
    pub failed_batched_spec_verify_keys: Vec<BatchedSpecVerifyGraphKey>,
    pub batched_decode_io: BatchedDecodeGraphIo,
    pub batched_decode_graphs: Vec<(BatchedDecodeGraphKey, u64, CudaGraph)>,
    pub aux_stream: CudaStream,
    pub aux_marlin_ws: DeviceBuffer,
    pub aux_fork_event: crate::cuda::CudaEvent,
    pub aux_join_event: crate::cuda::CudaEvent,
    pub capacity_plan: CapacityPlan,
    pub tap_layer_ids: Vec<usize>,
    pub tap_aux_bf16: Option<DeviceBuffer>,
    /// Selected-row local-vocabulary logits for native processed verification.
    pub speculative_logits: Option<DeviceBuffer>,
    pub speculative_logits_rows: usize,
    pub last_logits_ptr: u64,
    pub last_logits_rows: usize,
    pub captured_decode_logits_ptr: u64,
    pub last_final_hidden_f16: Option<DeviceBuffer>,
    pub max_tap_rows: usize,
    pub current_pos_plane_stride: i32,
    pub pending_visual_splice: Option<(u64, u64, usize)>,
    pub(crate) vision_splice_idx: Option<DeviceBuffer>,
    pub(crate) vision_splice_emb: Option<DeviceBuffer>,
    pub last_norm_hidden_ptr: u64,
    pub last_norm_hidden_rows: usize,
    pub last_batch_state_slots: Vec<usize>,
    pub tiered_moe: Option<crate::moe_offload::RankTieredMoe>,
    pub tiered_moe_hot_per_layer: usize,
    pub ple_runtime: Option<PleRankRuntime>,
    pub lora_runtime: Option<crate::lora::RankLoraRuntime>,
    pub is_capturing_graph: bool,
}

#[cfg(feature = "cuda")]
enum PreloadedAttnShards {
    FullAttention {
        qkv_shard: RawLinearHost,
        o_shard: RawLinearHost,
    },
    LinearAttention {
        qkvz_shard: RawLinearHost,
        ba_shard: RawLinearHost,
        out_shard: RawLinearHost,
    },
}

#[cfg(feature = "cuda")]
struct PreloadedLayerShards {
    attn: PreloadedAttnShards,
    dense_ffn: Option<(RawLinearHost, RawLinearHost)>,
}

#[cfg(feature = "cuda")]
fn prepare_layer_linear_shards(
    archive: &SafetensorsArchive,
    cfg: &ModelConfig,
    layer_id: usize,
    rank: usize,
    tp_size: usize,
) -> Result<PreloadedLayerShards> {
    let pfx = format!("model.layers.{layer_id}");
    let local_q_heads = cfg.num_attention_heads / tp_size;
    let local_kv_heads = cfg.local_kv_heads(tp_size);
    let kv_shard_idx = cfg.kv_head_shard_index(rank, tp_size);
    let q_gate_total = cfg.num_attention_heads * cfg.head_dim * 2;
    let q_gate_local = local_q_heads * cfg.head_dim * 2;
    let kv_local = local_kv_heads * cfg.head_dim;

    let gdn_k_total = cfg.linear_num_key_heads * cfg.linear_key_head_dim;
    let gdn_v_total = cfg.linear_num_value_heads * cfg.linear_value_head_dim;
    let gdn_k_local = gdn_k_total / tp_size;
    let gdn_v_local = gdn_v_total / tp_size;
    let local_v_heads = cfg.linear_num_value_heads / tp_size;
    let local_interm = cfg.intermediate_size / tp_size;

    let is_moe_layer = cfg.num_experts > 0
        && !cfg.mlp_only_layers.contains(&layer_id)
        && (archive
            .resolve_name(&format!("{pfx}.mlp.gate.weight"))
            .is_some()
            || archive
                .resolve_name(&format!("{pfx}.mlp.gate.weight_packed"))
                .is_some()
            || archive
                .resolve_name(&format!("{pfx}.mlp.gate.qweight"))
                .is_some());

    let pfx_ref = pfx.as_str();
    std::thread::scope(|s| -> Result<PreloadedLayerShards> {
        let ffn_handle = (!is_moe_layer).then(|| {
            let gate_h = s.spawn(move || -> Result<RawLinearHost> {
                Ok(
                    RawLinearHost::load(archive, &format!("{pfx_ref}.mlp.gate_proj"), &cfg.quant)?
                        .slice_rows(rank * local_interm, (rank + 1) * local_interm),
                )
            });
            let up_h = s.spawn(move || -> Result<RawLinearHost> {
                Ok(
                    RawLinearHost::load(archive, &format!("{pfx_ref}.mlp.up_proj"), &cfg.quant)?
                        .slice_rows(rank * local_interm, (rank + 1) * local_interm),
                )
            });
            let down_h = s.spawn(move || -> Result<RawLinearHost> {
                RawLinearHost::load(archive, &format!("{pfx_ref}.mlp.down_proj"), &cfg.quant)?
                    .slice_cols(rank * local_interm, (rank + 1) * local_interm)
            });
            (gate_h, up_h, down_h)
        });

        let attn = match cfg.layer_types[layer_id] {
            LayerKind::FullAttention => {
                let q_h = s.spawn(move || -> Result<RawLinearHost> {
                    let raw_q = RawLinearHost::load(
                        archive,
                        &format!("{pfx_ref}.self_attn.q_proj"),
                        &cfg.quant,
                    )?;
                    debug_assert_eq!(raw_q.slice_rows(0, q_gate_total).size_check(), q_gate_total);
                    Ok(raw_q.slice_rows(rank * q_gate_local, (rank + 1) * q_gate_local))
                });
                let k_h = s.spawn(move || -> Result<RawLinearHost> {
                    Ok(RawLinearHost::load(
                        archive,
                        &format!("{pfx_ref}.self_attn.k_proj"),
                        &cfg.quant,
                    )?
                    .slice_rows(kv_shard_idx * kv_local, (kv_shard_idx + 1) * kv_local))
                });
                let v_h = s.spawn(move || -> Result<RawLinearHost> {
                    Ok(RawLinearHost::load(
                        archive,
                        &format!("{pfx_ref}.self_attn.v_proj"),
                        &cfg.quant,
                    )?
                    .slice_rows(kv_shard_idx * kv_local, (kv_shard_idx + 1) * kv_local))
                });
                let o_cols = local_q_heads * cfg.head_dim;
                let o_h = s.spawn(move || -> Result<RawLinearHost> {
                    RawLinearHost::load(
                        archive,
                        &format!("{pfx_ref}.self_attn.o_proj"),
                        &cfg.quant,
                    )?
                    .slice_cols(rank * o_cols, (rank + 1) * o_cols)
                });
                let q_shard = q_h
                    .join()
                    .map_err(|_| loader_err("q_proj loader panicked"))??;
                let k_shard = k_h
                    .join()
                    .map_err(|_| loader_err("k_proj loader panicked"))??;
                let v_shard = v_h
                    .join()
                    .map_err(|_| loader_err("v_proj loader panicked"))??;
                let o_shard = o_h
                    .join()
                    .map_err(|_| loader_err("o_proj loader panicked"))??;
                let qkv_shard = RawLinearHost::concat_rows(&[q_shard, k_shard, v_shard])?;
                PreloadedAttnShards::FullAttention { qkv_shard, o_shard }
            }
            LayerKind::LinearAttention => {
                let qkv_h = s.spawn(
                    move || -> Result<(RawLinearHost, RawLinearHost, RawLinearHost)> {
                        let raw_qkv = RawLinearHost::load(
                            archive,
                            &format!("{pfx_ref}.linear_attn.in_proj_qkv"),
                            &cfg.quant,
                        )?;
                        let q_part =
                            raw_qkv.slice_rows(rank * gdn_k_local, (rank + 1) * gdn_k_local);
                        let k_part = raw_qkv.slice_rows(
                            gdn_k_total + rank * gdn_k_local,
                            gdn_k_total + (rank + 1) * gdn_k_local,
                        );
                        let v_part = raw_qkv.slice_rows(
                            2 * gdn_k_total + rank * gdn_v_local,
                            2 * gdn_k_total + (rank + 1) * gdn_v_local,
                        );
                        Ok((q_part, k_part, v_part))
                    },
                );
                let z_h = s.spawn(move || -> Result<RawLinearHost> {
                    Ok(RawLinearHost::load(
                        archive,
                        &format!("{pfx_ref}.linear_attn.in_proj_z"),
                        &cfg.quant,
                    )?
                    .slice_rows(rank * gdn_v_local, (rank + 1) * gdn_v_local))
                });
                let b_h = s.spawn(move || -> Result<RawLinearHost> {
                    Ok(RawLinearHost::load(
                        archive,
                        &format!("{pfx_ref}.linear_attn.in_proj_b"),
                        &cfg.quant,
                    )?
                    .slice_rows(rank * local_v_heads, (rank + 1) * local_v_heads))
                });
                let a_h = s.spawn(move || -> Result<RawLinearHost> {
                    Ok(RawLinearHost::load(
                        archive,
                        &format!("{pfx_ref}.linear_attn.in_proj_a"),
                        &cfg.quant,
                    )?
                    .slice_rows(rank * local_v_heads, (rank + 1) * local_v_heads))
                });
                let out_h = s.spawn(move || -> Result<RawLinearHost> {
                    RawLinearHost::load(
                        archive,
                        &format!("{pfx_ref}.linear_attn.out_proj"),
                        &cfg.quant,
                    )?
                    .slice_cols(rank * gdn_v_local, (rank + 1) * gdn_v_local)
                });
                let (q_part, k_part, v_part) = qkv_h
                    .join()
                    .map_err(|_| loader_err("in_proj_qkv loader panicked"))??;
                let z_shard = z_h
                    .join()
                    .map_err(|_| loader_err("in_proj_z loader panicked"))??;
                let b_shard = b_h
                    .join()
                    .map_err(|_| loader_err("in_proj_b loader panicked"))??;
                let a_shard = a_h
                    .join()
                    .map_err(|_| loader_err("in_proj_a loader panicked"))??;
                let out_shard = out_h
                    .join()
                    .map_err(|_| loader_err("out_proj loader panicked"))??;
                let qkvz_shard = RawLinearHost::concat_rows(&[q_part, k_part, v_part, z_shard])?;
                let ba_shard = RawLinearHost::concat_rows(&[b_shard, a_shard])?;
                PreloadedAttnShards::LinearAttention {
                    qkvz_shard,
                    ba_shard,
                    out_shard,
                }
            }
        };

        let dense_ffn = if let Some((gate_h, up_h, down_h)) = ffn_handle {
            let gate_shard = gate_h
                .join()
                .map_err(|_| loader_err("gate_proj loader panicked"))??;
            let up_shard = up_h
                .join()
                .map_err(|_| loader_err("up_proj loader panicked"))??;
            let down_shard = down_h
                .join()
                .map_err(|_| loader_err("down_proj loader panicked"))??;
            let gate_up_shard = RawLinearHost::concat_rows(&[gate_shard, up_shard])?;
            Some((gate_up_shard, down_shard))
        } else {
            None
        };

        Ok(PreloadedLayerShards { attn, dense_ffn })
    })
}

/// Phase-1 rank state after target non-MoE weights, scratch arena, and auxiliary streams
/// have been allocated on the GPU and temporary repack staging has been freed, prior to
/// initializing `PleRankRuntime`, `RankTieredMoe`, `StateBundlePool`, and `KvPool`.
#[cfg(feature = "cuda")]
pub(super) struct PrePoolModelRank {
    rank: usize,
    tp_size: usize,
    device_id: i32,
    device: CudaDevice,
    stream: CudaStream,
    cublas: CublasHandle,
    cfg: ModelConfig,
    total_vram: usize,
    free_vram: usize,
    est_weight_bytes: usize,
    local_kv_heads: usize,
    local_v_heads: usize,
    gdn_k_local: usize,
    gdn_v_local: usize,
    local_vocab_size: usize,
    vocab_offset: usize,
    full_attn_layer_ids: Vec<usize>,
    gdn_layer_ids: Vec<usize>,
    weight_arena: DeviceArena,
    scratch_arena: DeviceArena,
    batch_meta_arena: DeviceArena,
    marlin_ws: DeviceBuffer,
    embed_tokens_ptr: u64,
    host_embed_tokens: Option<crate::cuda::MappedHostBuffer>,
    final_residual: FinalResidual,
    lm_head: Option<LinearProj>,
    layers: Vec<DecoderLayerWeights>,
    aux_stream: CudaStream,
    aux_marlin_ws: DeviceBuffer,
    aux_fork_event: crate::cuda::CudaEvent,
    aux_join_event: crate::cuda::CudaEvent,
}

#[cfg(feature = "cuda")]
impl Qwen3_5ModelRank {
    pub fn attach_lora_adapters(
        &mut self,
        adapters: &[crate::lora::LoraAdapterHost],
    ) -> Result<()> {
        self.device.make_current()?;
        self.decode_graph.clear();
        self.spec_verify_graphs.clear();
        self.failed_spec_verify_keys.clear();
        self.batched_spec_verify_graphs.clear();
        self.failed_batched_spec_verify_keys.clear();
        self.batched_decode_graphs.clear();
        if !adapters.is_empty() {
            let target_scratch = if adapters
                .iter()
                .any(|ad| ad.midloop.iter().any(|&(_, _, passes)| passes > 1))
            {
                960 * 1024 * 1024
            } else {
                crate::scheduler::TIERED_MOE_SCRATCH_ARENA_BYTES
            };
            if self.scratch_arena.capacity() < target_scratch {
                self.scratch_arena = DeviceArena::new(self.device.device_id, 4096)?;
                self.scratch_arena = DeviceArena::new(self.device.device_id, target_scratch)?;
            }
        }
        self.lora_runtime = crate::lora::attach_lora_to_rank(
            &self.cfg,
            self.rank,
            self.tp_size,
            self.device.device_id,
            &self.stream,
            &mut self.layers,
            &mut self.lm_head,
            adapters,
        )?;
        Ok(())
    }

    #[inline]
    pub fn set_lora_uniform(&self, lora_id: i32) {
        if let Some(rt) = &self.lora_runtime {
            rt.state.set_uniform(lora_id);
        }
    }

    #[inline]
    pub fn bind_graph_lora_mode(&self, lora_mode: GraphLoraMode, token_lora_ids_ptr: u64) {
        let Some(rt) = &self.lora_runtime else {
            return;
        };
        match lora_mode {
            GraphLoraMode::BaseOnly => rt.state.set_uniform(crate::lora::BASE_MODEL_LORA_ID),
            GraphLoraMode::Uniform(lid) => rt.state.set_uniform(lid),
            GraphLoraMode::Mixed => rt.state.set_device_ids(token_lora_ids_ptr),
        }
    }

    pub fn bind_lora_for_batch(
        &mut self,
        uniform_lora_id: Option<i32>,
        flat_token_lora_ids: Option<&[i32]>,
    ) -> Result<()> {
        let Some(rt) = &self.lora_runtime else {
            return Ok(());
        };
        if let Some(ids) = flat_token_lora_ids {
            let d_ptr = self.batch_meta_arena.upload_slice(ids, &self.stream)?;
            rt.state.set_device_ids(d_ptr);
        } else {
            rt.state
                .set_uniform(uniform_lora_id.unwrap_or(crate::lora::BASE_MODEL_LORA_ID));
        }
        Ok(())
    }

    pub fn load_rank(
        archive: &SafetensorsArchive,
        cfg: &ModelConfig,
        rank: usize,
        tp_size: usize,
        device_id: i32,
        kv_policy: KvFormatPolicy,
        requested_kv_pages: Option<usize>,
    ) -> Result<Self> {
        Self::load_rank_with_capacity(
            archive,
            cfg,
            rank,
            tp_size,
            device_id,
            kv_policy,
            PoolCapacity {
                kv_pages: requested_kv_pages,
                ..PoolCapacity::default()
            },
            None,
        )
    }

    fn validate_tp_geometry(cfg: &ModelConfig, tp_size: usize) -> Result<()> {
        if !matches!(tp_size, 1 | 2 | 4) {
            return Err(loader_err_msg(format!(
                "unsupported tp_size={tp_size} (warp-34 supports TP=1, TP=2, or TP=4)"
            )));
        }
        if cfg.vocab_size == 0 || !cfg.vocab_size.is_multiple_of(tp_size) {
            return Err(loader_err_msg(format!(
                "vocab_size={} is not divisible by tp_size={tp_size}",
                cfg.vocab_size
            )));
        }
        if cfg.num_full_attention_layers() > 0 {
            if cfg.num_attention_heads == 0
                || cfg.num_key_value_heads == 0
                || !cfg.num_attention_heads.is_multiple_of(tp_size)
                || !cfg.kv_heads_compatible_with_tp(tp_size)
            {
                return Err(loader_err_msg(format!(
                    "attention heads (q={}, kv={}) are not compatible with tp_size={tp_size}; retry with --tp 1",
                    cfg.num_attention_heads, cfg.num_key_value_heads
                )));
            }
            let local_kv = cfg.local_kv_heads(tp_size);
            if !(cfg.num_attention_heads / tp_size).is_multiple_of(local_kv) {
                return Err(loader_err_msg(format!(
                    "local Q heads ({}) must be a multiple of local KV heads ({local_kv})",
                    cfg.num_attention_heads / tp_size,
                )));
            }
        }
        if cfg.num_linear_attention_layers() > 0 {
            if cfg.linear_num_key_heads == 0
                || cfg.linear_num_value_heads == 0
                || !cfg.linear_num_key_heads.is_multiple_of(tp_size)
                || !cfg.linear_num_value_heads.is_multiple_of(tp_size)
            {
                return Err(loader_err_msg(format!(
                    "GDN linear_attn heads (k={}, v={}) are not divisible by tp_size={tp_size}; retry with --tp 1",
                    cfg.linear_num_key_heads, cfg.linear_num_value_heads
                )));
            }
            let lk = cfg.linear_num_key_heads / tp_size;
            let lv = cfg.linear_num_value_heads / tp_size;
            if !lv.is_multiple_of(lk) {
                return Err(loader_err_msg(format!(
                    "GDN local value heads ({lv}) must be an integer multiple of local key heads ({lk})"
                )));
            }
        }
        if cfg.num_experts > 0 {
            if cfg.uses_tiered_moe() {
                if !cfg.num_experts.is_multiple_of(tp_size)
                    || cfg.moe_intermediate_size == 0
                    || !cfg.moe_intermediate_size.is_multiple_of(8)
                {
                    return Err(loader_err_msg(format!(
                        "tiered MoE num_experts={} / moe_intermediate_size={} incompatible with TP={tp_size}",
                        cfg.num_experts, cfg.moe_intermediate_size
                    )));
                }
            } else if cfg.moe_intermediate_size == 0
                || !cfg.moe_intermediate_size.is_multiple_of(tp_size * 8)
            {
                return Err(loader_err_msg(format!(
                    "moe_intermediate_size={} must be divisible by {} for TP={tp_size}",
                    cfg.moe_intermediate_size,
                    tp_size * 8
                )));
            }
            if cfg.shared_expert_intermediate_size > 0
                && !cfg.shared_expert_intermediate_size.is_multiple_of(tp_size)
            {
                return Err(loader_err_msg(format!(
                    "shared_expert_intermediate_size={} is not divisible by tp_size={tp_size}",
                    cfg.shared_expert_intermediate_size
                )));
            }
        }
        if (cfg.num_experts == 0 || !cfg.mlp_only_layers.is_empty())
            && cfg.intermediate_size > 0
            && !cfg.intermediate_size.is_multiple_of(tp_size)
        {
            return Err(loader_err_msg(format!(
                "intermediate_size={} is not divisible by tp_size={tp_size}",
                cfg.intermediate_size
            )));
        }
        Ok(())
    }

    fn fused_row_siblings(base_name: &str) -> Option<(&str, &'static [&'static str])> {
        for (suffix, group) in [
            (".self_attn.q_proj", &["q_proj", "k_proj", "v_proj"][..]),
            (".self_attn.k_proj", &["q_proj", "k_proj", "v_proj"][..]),
            (".self_attn.v_proj", &["q_proj", "k_proj", "v_proj"][..]),
            (
                ".linear_attn.in_proj_qkv",
                &["in_proj_qkv", "in_proj_z"][..],
            ),
            (".linear_attn.in_proj_z", &["in_proj_qkv", "in_proj_z"][..]),
            (".linear_attn.in_proj_b", &["in_proj_b", "in_proj_a"][..]),
            (".linear_attn.in_proj_a", &["in_proj_b", "in_proj_a"][..]),
            (".mlp.gate_proj", &["gate_proj", "up_proj"][..]),
            (".mlp.up_proj", &["gate_proj", "up_proj"][..]),
            (".shared_expert.gate_proj", &["gate_proj", "up_proj"][..]),
            (".shared_expert.up_proj", &["gate_proj", "up_proj"][..]),
        ] {
            if base_name.ends_with(suffix) {
                let stem_end = base_name.rfind('.').unwrap_or(base_name.len());
                return Some((&base_name[..stem_end], group));
            }
        }
        None
    }

    pub(super) fn estimate_weight_arena_bytes(
        cfg: &ModelConfig,
        archive: &SafetensorsArchive,
        tp_size: usize,
        requested_kv_pages: Option<usize>,
    ) -> usize {
        let embed_fp16_aligned = (cfg
            .vocab_size
            .saturating_mul(cfg.hidden_size)
            .saturating_mul(2))
        .saturating_add(255)
            & !255;
        let mut replicated_bytes = if cfg.uses_tiered_moe() && !cfg.tie_word_embeddings {
            0
        } else {
            embed_fp16_aligned
        };
        let mut sharded_bytes = 0usize;
        if cfg.tie_word_embeddings && tp_size > 1 {
            let lm_shard = ((cfg.vocab_size / tp_size)
                .saturating_mul(cfg.hidden_size)
                .saturating_mul(2))
            .saturating_add(255)
                & !255;
            replicated_bytes = replicated_bytes.saturating_add(lm_shard);
        }

        let bits = cfg.quant.bits();
        let pack_factor = if bits == 8 { 4usize } else { 8usize };

        for (name, info) in archive.iter_entries() {
            if name.starts_with("visual.")
                || name.contains(".visual.")
                || name.starts_with("mtp.")
                || name.contains(".mtp.")
                || name.contains("embed_tokens.")
                || (cfg.tie_word_embeddings
                    && (name.starts_with("lm_head.") || name.contains(".lm_head.")))
                || name.ends_with(".g_idx")
                || name.ends_with(".weight_g_idx")
                || name.ends_with(".weight_shape")
                || (name.ends_with(".qzeros") && !cfg.quant.expects_zero_points())
                || (cfg.uses_tiered_moe() && name.contains(".experts."))
                || (name.contains(".ple_embedding.")
                    && (!name.ends_with(".weight_scale") || info.num_elems() > 1))
            {
                continue;
            }
            let is_replicated = name.contains("norm.")
                || name.contains("hyper_connection")
                || name.contains(".ple.")
                || name.contains(".indexer.")
                || name.ends_with(".mlp.gate.weight")
                || name.ends_with(".mlp.gate.weight_packed")
                || name.ends_with(".mlp.gate.weight_scale")
                || name.ends_with(".mlp.gate.weight_zero_point")
                || name.ends_with(".mlp.gate.qweight")
                || name.ends_with(".mlp.gate.scales")
                || name.ends_with(".mlp.gate.qzeros")
                || name.contains(".shared_expert_gate.");
            let is_col_sharded = name.contains(".o_proj.")
                || name.contains(".out_proj.")
                || name.contains(".down_proj.");
            let raw_gpu_bytes = match info.dtype.as_str() {
                "F16" | "BF16" => info.num_elems().saturating_mul(2),
                "F32" => {
                    if name.ends_with(".A_log") {
                        info.num_elems().saturating_mul(4)
                    } else {
                        info.num_elems().saturating_mul(2)
                    }
                }
                "I32" | "U32" => {
                    let is_expert = name.contains(".experts.");
                    if is_expert
                        && info.shape.len() == 2
                        && (name.ends_with(".weight_packed") || name.ends_with(".qweight"))
                    {
                        info.len_bytes as usize
                    } else {
                        let skinny_dequant = if !is_expert
                            && info.shape.len() == 2
                            && (name.ends_with(".weight_packed") || name.ends_with(".qweight"))
                        {
                            let base_name = name
                                .trim_end_matches(".weight_packed")
                                .trim_end_matches(".qweight");
                            cfg.quant.for_linear(base_name).map(|t_quant| {
                                let t_bits = t_quant.bits();
                                let t_pf = match t_bits {
                                    2 => 16usize,
                                    8 => 4usize,
                                    _ => pack_factor,
                                };
                                let (n, k) =
                                    if t_quant.is_awq_packing() && name.ends_with(".qweight") {
                                        (info.shape[1] * 8, info.shape[0])
                                    } else if name.ends_with(".qweight") {
                                        (info.shape[1], info.shape[0] * t_pf)
                                    } else {
                                        (info.shape[0], info.shape[1] * t_pf)
                                    };
                                let (local_n, local_k) = if is_replicated {
                                    (n, k)
                                } else if is_col_sharded {
                                    (n, k.div_ceil(tp_size))
                                } else {
                                    (n.div_ceil(tp_size), k)
                                };
                                let marlin_ok = (local_n % 64 == 0 && local_k % 128 == 0)
                                    || (local_n % 128 == 0 && local_k % 64 == 0);
                                let gs = t_quant.group_size_or(local_k);
                                let gs_ok = (t_quant.is_channelwise()
                                    || matches!(gs, 32 | 64 | 128))
                                    && (gs == 0 || local_k.is_multiple_of(gs));
                                if !marlin_ok || !gs_ok {
                                    Some(n * k * 2)
                                } else if let Some((parent, group)) =
                                    Self::fused_row_siblings(base_name)
                                {
                                    let mut dequant_f16 = false;
                                    let mut widen_w2_to_w4 = false;
                                    for sib in group {
                                        let sib_name = format!("{parent}.{sib}");
                                        let sib_has_packed = archive
                                            .resolve_name(&format!("{sib_name}.weight_packed"))
                                            .is_some()
                                            || archive
                                                .resolve_name(&format!("{sib_name}.qweight"))
                                                .is_some();
                                        let sib_has_f16 = archive
                                            .resolve_name(&format!("{sib_name}.weight"))
                                            .is_some();
                                        if !sib_has_packed && sib_has_f16 {
                                            dequant_f16 = true;
                                            break;
                                        }
                                        let Ok(sib_quant) = cfg.quant.for_linear(&sib_name) else {
                                            dequant_f16 = true;
                                            break;
                                        };
                                        if sib_quant == t_quant {
                                            continue;
                                        }
                                        if matches!(t_bits, 2 | 4)
                                            && matches!(sib_quant.bits(), 2 | 4)
                                            && !t_quant.expects_zero_points()
                                            && !sib_quant.expects_zero_points()
                                            && sib_quant.group_size_or(local_k) == gs
                                        {
                                            if t_bits == 2 && sib_quant.bits() == 4 {
                                                widen_w2_to_w4 = true;
                                            }
                                        } else {
                                            dequant_f16 = true;
                                            break;
                                        }
                                    }
                                    if dequant_f16 {
                                        Some(n * k * 2)
                                    } else if widen_w2_to_w4 {
                                        Some(n * k / 2)
                                    } else {
                                        None
                                    }
                                } else {
                                    None
                                }
                            })
                        } else {
                            Ok(None)
                        };
                        match skinny_dequant {
                            Ok(Some(bytes)) => bytes,
                            Ok(None) => info.len_bytes as usize,
                            Err(_) => (info.len_bytes as usize).saturating_mul(4),
                        }
                    }
                }
                _ => info.len_bytes as usize,
            };
            if is_replicated {
                let aligned = raw_gpu_bytes.saturating_add(255) & !255;
                replicated_bytes = replicated_bytes.saturating_add(aligned);
            } else {
                sharded_bytes = sharded_bytes.saturating_add(raw_gpu_bytes);
            }
        }

        let per_rank_raw = replicated_bytes.saturating_add(sharded_bytes.div_ceil(tp_size));
        let extreme_longctx = tp_size == 1
            && cfg.num_hidden_layers >= EXTREME_LONGCTX_MIN_LAYERS
            && requested_kv_pages.unwrap_or(0) >= EXTREME_LONGCTX_MIN_PAGES;
        let margin = if extreme_longctx {
            WEIGHT_ARENA_EXTREME_LONGCTX_MARGIN_BYTES
        } else {
            WEIGHT_ARENA_DEFAULT_MARGIN_BYTES
        };
        let rounded = per_rank_raw
            .saturating_add(margin)
            .div_ceil(WEIGHT_ARENA_ALIGNMENT_BYTES)
            .saturating_mul(WEIGHT_ARENA_ALIGNMENT_BYTES);
        rounded.max(MIN_WEIGHT_ARENA_BYTES)
    }

    fn load_embed_tokens_f16(archive: &SafetensorsArchive, cfg: &ModelConfig) -> Result<Vec<u16>> {
        let expected_elems = cfg
            .vocab_size
            .checked_mul(cfg.hidden_size)
            .ok_or_else(|| loader_err("embed_tokens size overflow"))?;
        if let Some(embed_key) = archive.resolve_name("model.embed_tokens.weight") {
            if archive
                .resolve_name("model.embed_tokens.weight_scale")
                .is_some()
                || archive
                    .resolve_name("model.embed_tokens.weight_scale_inv")
                    .is_some()
            {
                return Err(loader_err_msg(format!(
                    "unsupported FP8/float-quantized embedding tensor '{embed_key}'"
                )));
            }
            let embed_f16 = archive
                .read_f16_bits(&embed_key)
                .map_err(|e| loader_err_msg(format!("read '{embed_key}' failed: {e}")))?;
            if embed_f16.len() != expected_elems {
                return Err(loader_err_msg(format!(
                    "embed_tokens element count {} does not match vocab_size*hidden_size ({}*{}={expected_elems})",
                    embed_f16.len(),
                    cfg.vocab_size,
                    cfg.hidden_size
                )));
            }
            return Ok(embed_f16);
        }
        if archive
            .resolve_name("model.embed_tokens.weight_packed")
            .is_some()
            || archive.resolve_name("model.embed_tokens.qweight").is_some()
        {
            let mut embed_quant = cfg.quant.clone();
            if let Some(pk) = archive.resolve_name("model.embed_tokens.weight_packed")
                && let Ok(p_info) = archive.tensor_info(&pk)
                && p_info.shape.len() == 2
                && p_info.shape[0] == cfg.vocab_size
            {
                let has_zp = archive
                    .resolve_name("model.embed_tokens.weight_zero_point")
                    .is_some();
                if p_info.shape[1].saturating_mul(4) == cfg.hidden_size {
                    embed_quant = crate::loader::QuantScheme::CompressedTensors {
                        num_bits: 8,
                        group_size: cfg.quant.raw_group_size(),
                        symmetric: true,
                        ignore: Vec::new(),
                        rules: None,
                    };
                } else if p_info.shape[1].saturating_mul(8) == cfg.hidden_size {
                    embed_quant = crate::loader::QuantScheme::CompressedTensors {
                        num_bits: 4,
                        group_size: cfg.quant.raw_group_size(),
                        symmetric: !has_zp,
                        ignore: Vec::new(),
                        rules: None,
                    };
                }
            }
            let raw_embed = RawLinearHost::load(archive, "model.embed_tokens", &embed_quant)?;
            if raw_embed.dims() != (cfg.vocab_size, cfg.hidden_size) {
                return Err(loader_err_msg(format!(
                    "quantized embed_tokens dims {:?} do not match ({}, {})",
                    raw_embed.dims(),
                    cfg.vocab_size,
                    cfg.hidden_size
                )));
            }
            return Ok(raw_embed.to_f16_vec());
        }
        Err(loader_err_msg("missing model.embed_tokens.weight"))
    }

    #[allow(clippy::cognitive_complexity, clippy::too_many_arguments)]
    pub(super) fn load_rank_with_capacity(
        archive: &SafetensorsArchive,
        cfg: &ModelConfig,
        rank: usize,
        tp_size: usize,
        device_id: i32,
        kv_policy: KvFormatPolicy,
        capacity: PoolCapacity,
        shared_vocab: Option<&SharedVocabHost>,
    ) -> Result<Self> {
        let pre = Self::load_rank_pre_pool(
            archive,
            cfg,
            rank,
            tp_size,
            device_id,
            kv_policy,
            capacity,
            shared_vocab,
        )?;
        pre.finalize_runtime_and_pools(archive, kv_policy, capacity, None)
    }

    #[allow(clippy::cognitive_complexity, clippy::too_many_arguments)]
    pub(super) fn load_rank_pre_pool(
        archive: &SafetensorsArchive,
        cfg: &ModelConfig,
        rank: usize,
        tp_size: usize,
        device_id: i32,
        kv_policy: KvFormatPolicy,
        capacity: PoolCapacity,
        shared_vocab: Option<&SharedVocabHost>,
    ) -> Result<PrePoolModelRank> {
        let requested_kv_pages = capacity.kv_pages;
        Self::validate_tp_geometry(cfg, tp_size)?;
        if cfg.head_dim != 256 && !matches!(kv_policy, KvFormatPolicy::Fp16 | KvFormatPolicy::Bf16)
        {
            return Err(loader_err_msg(format!(
                "unsupported KV format {kv_policy:?} for head_dim={} (TQ4 and Q8_0 require head_dim=256; use --kv fp16)",
                cfg.head_dim
            )));
        }
        prepare_execution_plan(device_id)?;
        let device = CudaDevice::set_current(device_id)?;
        let dev_props = device.props()?;
        dev_props.validate_supported()?;
        let supports_fp8_kv = dev_props.supports_fp8_kv();
        if matches!(kv_policy, KvFormatPolicy::Fp8) && !supports_fp8_kv {
            return Err(loader_err_msg(format!(
                "--kv-format fp8 requires sm_89+ hardware FP8 (40-series or newer); GPU {device_id} ({}) is {}",
                dev_props.name,
                dev_props.sm_tag()
            )));
        }
        let marlin_ws_bytes = dev_props.marlin_workspace_bytes();
        let stream = CudaStream::new(device_id)?;
        let cublas = CublasHandle::new(&stream)?;
        let (free_vram, total_vram) = device.mem_info()?;

        let raw_est_weight_bytes =
            Self::estimate_weight_arena_bytes(cfg, archive, tp_size, requested_kv_pages);
        let max_packed_tensor_bytes = archive
            .iter_entries()
            .filter(|(name, info)| {
                matches!(info.dtype.as_str(), "I32" | "U32")
                    && !name.contains(".ple_embedding.")
                    && !(cfg.uses_tiered_moe() && name.contains(".experts."))
            })
            .map(|(name, info)| {
                let bytes = info.len_bytes as usize;
                if name.starts_with("lm_head.")
                    || name.contains(".lm_head.")
                    || name.contains(".experts.")
                {
                    bytes.div_ceil(tp_size)
                } else {
                    bytes
                        .div_ceil(tp_size)
                        .saturating_mul(REPACK_STAGING_TENSOR_MULTIPLIER)
                }
            })
            .max()
            .unwrap_or(0);
        let staging_bytes = MIN_REPACK_STAGING_BYTES.max(max_packed_tensor_bytes);
        let mut scratch_bytes = plan_scratch_arena_bytes_with_lora(
            cfg,
            tp_size,
            raw_est_weight_bytes,
            requested_kv_pages,
            capacity.lora_reserve,
        );
        if cfg.uses_tiered_moe() && free_vram < TIERED_MOE_COMPACT_VRAM_THRESHOLD_BYTES {
            scratch_bytes = crate::scheduler::DEFAULT_SCRATCH_ARENA_BYTES;
        }
        let pages = requested_kv_pages.unwrap_or(DEFAULT_TARGET_KV_PAGES);
        let (page_bytes, state_bytes) =
            target_pool_geometry_with_fp8(cfg, tp_size, kv_policy, supports_fp8_kv);
        let archive_has_vision = has_vision_weights(cfg, archive);
        let min_state_floor = if capacity.dflash_reserve_bytes != 0 {
            capacity.dflash_min_state_slots
        } else {
            MIN_ACTIVE_STATE_SLOTS
        };
        let steady_without_states = BATCH_META_ARENA_BYTES
            .saturating_add(marlin_ws_bytes.saturating_mul(2))
            .saturating_add(page_bytes.saturating_mul(pages))
            .saturating_add(target_base_headroom(cfg, tp_size, pages, free_vram))
            .saturating_add(capacity.dflash_reserve_bytes)
            .saturating_add(capacity.lora_reserve.device_bytes_per_rank)
            .saturating_add(if capacity.native_mtp {
                crate::mtp::dense_serving_reserve(cfg, tp_size, pages.saturating_mul(16))
            } else {
                0
            })
            .saturating_add(if rank == 0 {
                capacity.rank0_extra_headroom_bytes
            } else {
                0
            });
        // This is a placement admission floor, not a promise that all requested
        // serving slots fit: the final live-memory planner retains its clamp.
        // Repack staging is freed before pools; only the two Marlin workspaces
        // and batch metadata coexist with them. Architecture-specific runtime
        // allocations (e.g. PLE staging) remain subject to final live admission.
        let steady_bytes =
            steady_without_states.saturating_add(state_bytes.saturating_mul(min_state_floor));
        let mtp_experts_w4 = cfg.quant.bits() == 4
            && (archive
                .resolve_name("mtp.layers.0.mlp.experts.0.gate_proj.qweight")
                .is_some()
                || archive
                    .resolve_name("mtp.layers.0.mlp.experts.0.gate_proj.weight_packed")
                    .is_some());
        let resident_moe_mtp_reserve = resident_moe_mtp_serving_reserve(
            cfg,
            capacity.state_slots.is_some(),
            capacity.native_mtp,
            mtp_experts_w4,
        );
        let desired_gpu_steady_bytes = capacity.state_slots.map(|req_slots| {
            let desired_slots = req_slots.max(min_state_floor);
            let vision = if rank == 0 && archive_has_vision && pages < EXTREME_LONGCTX_MIN_PAGES {
                VISION_ENCODER_LAZY_RESERVE_BYTES
            } else {
                0
            };
            steady_without_states
                .saturating_add(resident_moe_mtp_reserve)
                .saturating_add(state_bytes.saturating_mul(desired_slots))
                .saturating_add(vision)
        });
        let (est_weight_bytes, use_host_embed) = plan_host_embed_with_serving_budget(
            cfg,
            raw_est_weight_bytes,
            scratch_bytes,
            staging_bytes,
            free_vram,
            BATCH_META_ARENA_BYTES.saturating_add(marlin_ws_bytes),
            Some(steady_bytes),
            desired_gpu_steady_bytes,
        );
        let min_boot_bytes = est_weight_bytes
            .saturating_add(scratch_bytes)
            .saturating_add(staging_bytes)
            .saturating_add(MIN_BOOT_MARGIN_BYTES);
        if min_boot_bytes > free_vram {
            let gib = 1024.0 * 1024.0 * 1024.0;
            let hint = if tp_size == 1 {
                "; retry with --tp 2 across 2 GPUs or use a 4-bit W4A16 checkpoint"
            } else {
                "; use a 4-bit W4A16 / 8-bit W8A16 quantized checkpoint or free GPU memory"
            };
            return Err(loader_err_msg(format!(
                "insufficient VRAM on GPU {device_id} (rank {rank}/{tp_size}): weights + scratch require {:.2} GiB/rank, but only {:.2} GiB is free out of {:.2} GiB{hint}",
                min_boot_bytes as f64 / gib,
                free_vram as f64 / gib,
                total_vram as f64 / gib,
            )));
        }

        let mut weight_arena = DeviceArena::new(device_id, est_weight_bytes)?;
        let scratch_arena = DeviceArena::new(device_id, scratch_bytes)?;
        let _ = stream
            .set_persisting_l2_window(scratch_arena.allocation_base(), scratch_arena.capacity());
        let batch_meta_arena = DeviceArena::new(device_id, BATCH_META_ARENA_BYTES)?;
        let marlin_ws = DeviceBuffer::alloc(device_id, marlin_ws_bytes)?;
        let repack_staging = DeviceBuffer::alloc(device_id, staging_bytes)?;

        // 1. embed_tokens (replicated in FP16; dequantized on host when packed)
        //    and optional untied lm_head (loaded once and shared across TP ranks when `shared_vocab` is provided).
        let has_untied_lm_head = !cfg.tie_word_embeddings
            && ["lm_head.weight", "lm_head.weight_packed", "lm_head.qweight"]
                .iter()
                .any(|k| archive.resolve_name(k).is_some());
        let owned_vocab;
        let (embed_f16, shared_lm_opt): (&[u16], Option<&RawLinearHost>) =
            if let Some(cell) = shared_vocab {
                let res = cell.get_or_init(|| {
                    let emb = Self::load_embed_tokens_f16(archive, cfg)?;
                    let lm = if has_untied_lm_head {
                        Some(RawLinearHost::load(archive, "lm_head", &cfg.quant)?)
                    } else {
                        None
                    };
                    Ok((emb, lm))
                });
                let (emb, lm) = res.as_ref().map_err(Clone::clone)?;
                (emb.as_slice(), lm.as_ref())
            } else {
                owned_vocab = Self::load_embed_tokens_f16(archive, cfg)?;
                (owned_vocab.as_slice(), None)
            };
        let (embed_tokens_ptr, host_embed_tokens) = if use_host_embed {
            let bytes = core::mem::size_of_val(embed_f16);
            let buf = crate::cuda::MappedHostBuffer::alloc(bytes)?;
            // SAFETY: `buf.host_ptr` is a freshly allocated pinned buffer of `bytes` bytes.
            unsafe {
                core::ptr::copy_nonoverlapping(
                    embed_f16.as_ptr().cast::<u8>(),
                    buf.host_ptr.cast::<u8>(),
                    bytes,
                );
            }
            (buf.dev_ptr as u64, Some(buf))
        } else {
            (weight_arena.upload_slice(embed_f16, &stream)?, None)
        };

        // 2. final norm or multi-stream HyperConnection mixer
        let final_residual = if cfg.uses_hyper_connection() {
            let mixer = load_hc_block_weights(
                archive,
                cfg,
                "model.hyper_connection_mixer",
                false,
                &mut weight_arena,
                &stream,
                &repack_staging,
            )?;
            FinalResidual::HyperConnection { mixer }
        } else {
            let norm_key = archive
                .resolve_name("model.norm.weight")
                .ok_or_else(|| loader_err("missing model.norm.weight"))?;
            let norm_f16 = archive
                .read_f16_bits(&norm_key)
                .map_err(|_| loader_err("read model.norm.weight"))?;
            FinalResidual::SingleStream {
                norm_ptr: weight_arena.upload_slice(&norm_f16, &stream)?,
            }
        };

        // 3. lm_head (sharded along vocab N across TP ranks; None for readout-only decision models)
        let local_vocab_size = cfg.vocab_size / tp_size;
        let vocab_offset = rank * local_vocab_size;
        let lm_head = if cfg.tie_word_embeddings {
            let row_start = vocab_offset;
            let row_end = vocab_offset + local_vocab_size;
            let slice = &embed_f16[row_start * cfg.hidden_size..row_end * cfg.hidden_size];
            let ptr = if tp_size == 1 {
                embed_tokens_ptr
            } else {
                weight_arena.upload_slice(slice, &stream)?
            };
            Some(LinearProj::DenseF16 {
                weight_ptr: ptr,
                size_n: local_vocab_size,
                size_k: cfg.hidden_size,
            })
        } else if let Some(raw_lm) = shared_lm_opt {
            let sharded = raw_lm.slice_rows(vocab_offset, vocab_offset + local_vocab_size);
            Some(sharded.upload(&mut weight_arena, &stream, &repack_staging)?)
        } else if has_untied_lm_head {
            let raw_lm = RawLinearHost::load(archive, "lm_head", &cfg.quant)?;
            let sharded = raw_lm.slice_rows(vocab_offset, vocab_offset + local_vocab_size);
            Some(sharded.upload(&mut weight_arena, &stream, &repack_staging)?)
        } else {
            None
        };

        // 4. Decoder layers
        let mut layers = Vec::with_capacity(cfg.num_hidden_layers);
        let mut full_attn_layer_ids = Vec::new();
        let mut gdn_layer_ids = Vec::new();

        let local_kv_heads = cfg.local_kv_heads(tp_size);
        let gdn_k_total = cfg.linear_num_key_heads * cfg.linear_key_head_dim;
        let gdn_v_total = cfg.linear_num_value_heads * cfg.linear_value_head_dim;
        let gdn_k_local = gdn_k_total / tp_size;
        let gdn_v_local = gdn_v_total / tp_size;
        let local_v_heads = cfg.linear_num_value_heads / tp_size;

        for layer_id in 0..cfg.num_hidden_layers {
            let preloaded = prepare_layer_linear_shards(archive, cfg, layer_id, rank, tp_size)?;
            {
                let pfx = format!("model.layers.{layer_id}");
                let residual = if cfg.uses_hyper_connection() {
                    let attn_hc = load_hc_block_weights(
                        archive,
                        cfg,
                        &format!("{pfx}.attn_hyper_connection"),
                        true,
                        &mut weight_arena,
                        &stream,
                        &repack_staging,
                    )?;
                    let mlp_hc = load_hc_block_weights(
                        archive,
                        cfg,
                        &format!("{pfx}.mlp_hyper_connection"),
                        true,
                        &mut weight_arena,
                        &stream,
                        &repack_staging,
                    )?;
                    LayerResidual::HyperConnection { attn_hc, mlp_hc }
                } else {
                    let in_ln_key = archive
                        .resolve_name(&format!("{pfx}.input_layernorm.weight"))
                        .ok_or_else(|| loader_err("missing input_layernorm"))?;
                    let in_ln = archive
                        .read_f16_bits(&in_ln_key)
                        .map_err(|_| loader_err("read input_layernorm"))?;
                    let input_layernorm_ptr = weight_arena.upload_slice(&in_ln, &stream)?;

                    let post_ln_key = archive
                        .resolve_name(&format!("{pfx}.post_attention_layernorm.weight"))
                        .ok_or_else(|| loader_err("missing post_attention_layernorm"))?;
                    let post_ln = archive
                        .read_f16_bits(&post_ln_key)
                        .map_err(|_| loader_err("read post_attention_layernorm"))?;
                    let post_attention_layernorm_ptr =
                        weight_arena.upload_slice(&post_ln, &stream)?;
                    LayerResidual::SingleStream {
                        input_layernorm_ptr,
                        post_attention_layernorm_ptr,
                    }
                };
                let ple = if cfg.uses_ple() && cfg.ple_layer_ids.contains(&(layer_id + 1)) {
                    Some(load_ple_layer_weights(
                        archive,
                        cfg,
                        &pfx,
                        &mut weight_arena,
                        &stream,
                        &repack_staging,
                    )?)
                } else {
                    None
                };

                let block = match preloaded.attn {
                    PreloadedAttnShards::FullAttention { qkv_shard, o_shard } => {
                        let kv_ord = full_attn_layer_ids.len();
                        full_attn_layer_ids.push(layer_id);

                        let qkv_proj =
                            qkv_shard.upload(&mut weight_arena, &stream, &repack_staging)?;
                        let o_proj = o_shard.upload(&mut weight_arena, &stream, &repack_staging)?;

                        let q_norm_key = archive
                            .resolve_name(&format!("{pfx}.self_attn.q_norm.weight"))
                            .ok_or_else(|| loader_err("missing q_norm"))?;
                        let q_norm = archive
                            .read_f16_bits(&q_norm_key)
                            .map_err(|_| loader_err("read q_norm"))?;
                        let q_norm_ptr = weight_arena.upload_slice(&q_norm, &stream)?;

                        let k_norm_key = archive
                            .resolve_name(&format!("{pfx}.self_attn.k_norm.weight"))
                            .ok_or_else(|| loader_err("missing k_norm"))?;
                        let k_norm = archive
                            .read_f16_bits(&k_norm_key)
                            .map_err(|_| loader_err("read k_norm"))?;
                        let k_norm_ptr = weight_arena.upload_slice(&k_norm, &stream)?;

                        let qsa_indexer = if cfg.uses_qsa() {
                            Some(load_qsa_indexer_weights(
                                archive,
                                cfg,
                                &pfx,
                                &mut weight_arena,
                                &stream,
                                &repack_staging,
                            )?)
                        } else {
                            None
                        };

                        LayerBlockWeights::FullAttention(FullAttentionLayerWeights {
                            layer_id,
                            kv_layer_ordinal: kv_ord,
                            qkv_proj,
                            o_proj,
                            q_norm_ptr,
                            k_norm_ptr,
                            qsa_indexer,
                        })
                    }
                    PreloadedAttnShards::LinearAttention {
                        qkvz_shard,
                        ba_shard,
                        out_shard,
                    } => {
                        let gdn_ord = gdn_layer_ids.len();
                        gdn_layer_ids.push(layer_id);

                        let in_proj_qkvz =
                            qkvz_shard.upload(&mut weight_arena, &stream, &repack_staging)?;
                        let in_proj_ba =
                            ba_shard.upload(&mut weight_arena, &stream, &repack_staging)?;
                        let out_proj =
                            out_shard.upload(&mut weight_arena, &stream, &repack_staging)?;

                        let conv_key = archive
                            .resolve_name(&format!("{pfx}.linear_attn.conv1d.weight"))
                            .ok_or_else(|| loader_err("missing conv1d.weight"))?;
                        let conv_all = archive
                            .read_f16_bits(&conv_key)
                            .map_err(|_| loader_err("read conv1d.weight"))?;
                        let k_sz = cfg.linear_conv_kernel_dim;
                        let mut conv_local =
                            Vec::with_capacity((2 * gdn_k_local + gdn_v_local) * k_sz);
                        conv_local.extend_from_slice(
                            &conv_all[rank * gdn_k_local * k_sz..(rank + 1) * gdn_k_local * k_sz],
                        );
                        conv_local.extend_from_slice(
                            &conv_all[(gdn_k_total + rank * gdn_k_local) * k_sz
                                ..(gdn_k_total + (rank + 1) * gdn_k_local) * k_sz],
                        );
                        conv_local.extend_from_slice(
                            &conv_all[(2 * gdn_k_total + rank * gdn_v_local) * k_sz
                                ..(2 * gdn_k_total + (rank + 1) * gdn_v_local) * k_sz],
                        );
                        let conv_weight_ptr = weight_arena.upload_slice(&conv_local, &stream)?;

                        let a_log_key = archive
                            .resolve_name(&format!("{pfx}.linear_attn.A_log"))
                            .ok_or_else(|| loader_err("missing A_log"))?;
                        let a_log_all = archive
                            .read_f32(&a_log_key)
                            .map_err(|_| loader_err("read A_log"))?;
                        let a_log_local =
                            &a_log_all[rank * local_v_heads..(rank + 1) * local_v_heads];
                        let a_log_ptr = weight_arena.upload_slice(a_log_local, &stream)?;

                        let dt_bias_key = archive
                            .resolve_name(&format!("{pfx}.linear_attn.dt_bias"))
                            .ok_or_else(|| loader_err("missing dt_bias"))?;
                        let dt_bias_all = archive
                            .read_f32(&dt_bias_key)
                            .map_err(|_| loader_err("read dt_bias"))?;
                        let dt_bias_local =
                            &dt_bias_all[rank * local_v_heads..(rank + 1) * local_v_heads];
                        let dt_bias_ptr = weight_arena.upload_slice(dt_bias_local, &stream)?;

                        let norm_w_key = archive
                            .resolve_name(&format!("{pfx}.linear_attn.norm.weight"))
                            .ok_or_else(|| loader_err("missing linear_attn.norm.weight"))?;
                        let norm_w_all = archive
                            .read_f16_bits(&norm_w_key)
                            .map_err(|_| loader_err("read linear_attn.norm.weight"))?;
                        let norm_weight_ptr = weight_arena.upload_slice(&norm_w_all, &stream)?;

                        LayerBlockWeights::LinearAttention(GdnLayerWeights {
                            layer_id,
                            gdn_layer_ordinal: gdn_ord,
                            in_proj_qkvz,
                            in_proj_ba,
                            out_proj,
                            conv_weight_ptr,
                            a_log_ptr,
                            dt_bias_ptr,
                            norm_weight_ptr,
                        })
                    }
                };

                let ffn = if let Some((gate_up_shard, down_shard)) = preloaded.dense_ffn {
                    let gate_up_proj =
                        gate_up_shard.upload(&mut weight_arena, &stream, &repack_staging)?;
                    let down_proj =
                        down_shard.upload(&mut weight_arena, &stream, &repack_staging)?;
                    FfnBlockWeights::Dense {
                        gate_up_proj,
                        down_proj,
                    }
                } else {
                    FfnBlockWeights::Moe(Box::new(load_moe_layer_weights(
                        archive,
                        cfg,
                        cfg.uses_tiered_moe().then_some(layer_id),
                        &pfx,
                        [rank, tp_size],
                        &mut weight_arena,
                        (&stream, &repack_staging),
                    )?))
                };

                layers.push(DecoderLayerWeights {
                    layer_id,
                    residual,
                    ple,
                    block,
                    ffn,
                });
            }
        }

        drop(repack_staging);
        let aux_stream = CudaStream::new(device_id)?;
        let aux_marlin_ws = DeviceBuffer::alloc(device_id, marlin_ws_bytes)?;
        let aux_fork_event = crate::cuda::CudaEvent::new(device_id, false)?;
        let aux_join_event = crate::cuda::CudaEvent::new(device_id, false)?;
        stream.synchronize()?;

        Ok(PrePoolModelRank {
            rank,
            tp_size,
            device_id,
            device,
            stream,
            cublas,
            cfg: cfg.clone(),
            total_vram,
            free_vram,
            est_weight_bytes,
            local_kv_heads,
            local_v_heads,
            gdn_k_local,
            gdn_v_local,
            local_vocab_size,
            vocab_offset,
            full_attn_layer_ids,
            gdn_layer_ids,
            weight_arena,
            scratch_arena,
            batch_meta_arena,
            marlin_ws,
            embed_tokens_ptr,
            host_embed_tokens,
            final_residual,
            lm_head,
            layers,
            aux_stream,
            aux_marlin_ws,
            aux_fork_event,
            aux_join_event,
        })
    }
}

#[cfg(feature = "cuda")]
impl PrePoolModelRank {
    /// Initializes optional `PleRankRuntime` and `RankTieredMoe` **before** sizing and
    /// allocating `StateBundlePool` and `KvPool` so pool budgets observe live GPU free VRAM
    /// after all model-resident caches and staging workspaces are already allocated.
    pub(super) fn finalize_runtime_and_pools(
        self,
        archive: &SafetensorsArchive,
        kv_policy: KvFormatPolicy,
        capacity: PoolCapacity,
        ple_shared: Option<&(
            std::sync::Arc<crate::ple::PleNgramHasher>,
            std::sync::Arc<crate::ple::PleMmapTable>,
        )>,
    ) -> Result<Qwen3_5ModelRank> {
        let Self {
            rank,
            tp_size,
            device_id,
            device,
            stream,
            cublas,
            cfg,
            total_vram,
            free_vram,
            est_weight_bytes,
            local_kv_heads,
            local_v_heads,
            gdn_k_local,
            gdn_v_local,
            local_vocab_size,
            vocab_offset,
            full_attn_layer_ids,
            gdn_layer_ids,
            weight_arena,
            scratch_arena,
            batch_meta_arena,
            marlin_ws,
            embed_tokens_ptr,
            host_embed_tokens,
            final_residual,
            lm_head,
            layers,
            aux_stream,
            aux_marlin_ws,
            aux_fork_event,
            aux_join_event,
        } = self;

        let _ = CudaDevice::set_current(device_id)?;

        // 1. Allocate PLE staging buffers first so live `mem_info` accounts for them.
        let ple_runtime = if let Some((hasher, table)) = ple_shared {
            Some(super::rank_runtime::alloc_ple_runtime(
                device_id,
                &cfg,
                std::sync::Arc::clone(hasher),
                std::sync::Arc::clone(table),
            )?)
        } else {
            None
        };

        let requested_kv_pages = capacity.kv_pages;
        let block_size = KV_PAGE_BLOCK_TOKENS;
        let supports_fp8_kv = device.props()?.supports_fp8_kv();
        let (bytes_per_kv_page_all_layers, bytes_per_state_slot) =
            target_pool_geometry_with_fp8(&cfg, tp_size, kv_policy, supports_fp8_kv);
        let default_state_slots =
            if cfg.uses_ple() || est_weight_bytes >= LARGE_MODEL_WEIGHT_THRESHOLD_BYTES {
                PLE_OR_LARGE_MODEL_DEFAULT_STATE_SLOTS
            } else {
                STANDARD_DEFAULT_STATE_SLOTS
            };
        let requested_state_slots = capacity.state_slots.unwrap_or(default_state_slots);
        let default_pages = requested_kv_pages.or(Some(DEFAULT_TARGET_KV_PAGES));
        let req_kv_bytes = default_pages
            .unwrap_or(DEFAULT_TARGET_KV_PAGES)
            .saturating_mul(bytes_per_kv_page_all_layers);
        let base_headroom =
            target_base_headroom(&cfg, tp_size, default_pages.unwrap_or(0), free_vram);
        let rank0_extra_headroom = if rank == 0 {
            capacity.rank0_extra_headroom_bytes
        } else {
            0
        };
        let min_state_slots_for_moe = if capacity.state_slots.is_some() {
            let cap = if tp_size >= 2 {
                if capacity.dflash_reserve_bytes > 0 {
                    capacity.dflash_min_state_slots.max(14)
                } else if capacity.native_mtp && default_pages.unwrap_or(0) < 2048 {
                    16
                } else {
                    18
                }
            } else {
                PLE_OR_LARGE_MODEL_DEFAULT_STATE_SLOTS
            };
            requested_state_slots.clamp(MIN_ACTIVE_STATE_SLOTS, cap)
        } else if tp_size >= 2 {
            requested_state_slots
        } else {
            requested_state_slots.min(SINGLE_GPU_DFLASH2_MIN_STATE_SLOTS)
        };

        // 2. Plan and allocate `RankTieredMoe` (GPU `HotExpertCache` + double-buffered H2D
        //    staging workspace) BEFORE sizing `StateBundlePool` and `KvPool`.
        let (tiered_moe, tiered_moe_hot_per_layer, actual_tiered_moe_bytes, post_pool_moe_reserve) =
            if cfg.uses_tiered_moe() {
                let post_weight_available = device.mem_info()?.0;
                let (hot_per_layer, total_moe_reserve) = plan_tiered_moe_hot_per_layer_with_dflash(
                    &cfg,
                    tp_size,
                    post_weight_available,
                    base_headroom
                        .saturating_add(req_kv_bytes)
                        .saturating_add(capacity.lora_reserve.device_bytes_per_rank)
                        .saturating_add(rank0_extra_headroom),
                    bytes_per_state_slot.saturating_mul(min_state_slots_for_moe),
                    capacity.dflash_reserve_bytes,
                );
                let stage_cap = select_tiered_moe_stage_capacity(&cfg, tp_size, hot_per_layer);
                let tm = crate::moe_offload::RankTieredMoe::load_for_rank(
                    archive,
                    &cfg,
                    device_id,
                    rank,
                    tp_size,
                    hot_per_layer,
                    stage_cap,
                    &stream,
                )?;
                stream.synchronize()?;
                let available_after_moe = device.mem_info()?.0;
                let allocated_bytes = post_weight_available.saturating_sub(available_after_moe);
                let post_pool_reserve = total_moe_reserve.saturating_sub(allocated_bytes);
                (Some(tm), hot_per_layer, allocated_bytes, post_pool_reserve)
            } else {
                (None, 0, 0, 0)
            };

        // 3. Size and allocate `StateBundlePool` & `KvPool` from live post-TieredMoE free VRAM.
        let mtp_experts_w4 = cfg.quant.bits() == 4
            && (archive
                .resolve_name("mtp.layers.0.mlp.experts.0.gate_proj.qweight")
                .is_some()
                || archive
                    .resolve_name("mtp.layers.0.mlp.experts.0.gate_proj.weight_packed")
                    .is_some());
        let resident_moe_mtp_reserve = resident_moe_mtp_serving_reserve(
            &cfg,
            capacity.state_slots.is_some(),
            capacity.native_mtp,
            mtp_experts_w4,
        );
        // When DFlash2 is active (`dflash_reserve_bytes > 0`), native MTP is not loaded,
        // so `dflash_reserve_bytes` replaces `post_pool_moe_reserve` rather than stacking on top.
        let post_pool_spec_or_moe_reserve = if capacity.dflash_reserve_bytes > 0 {
            capacity.dflash_reserve_bytes
        } else {
            post_pool_moe_reserve
        };
        let required_headroom = base_headroom
            .saturating_add(if capacity.native_mtp {
                crate::mtp::dense_serving_reserve(
                    &cfg,
                    tp_size,
                    default_pages
                        .unwrap_or(DEFAULT_TARGET_KV_PAGES)
                        .saturating_mul(16),
                )
            } else {
                0
            })
            .saturating_add(resident_moe_mtp_reserve)
            .saturating_add(post_pool_spec_or_moe_reserve)
            .saturating_add(capacity.lora_reserve.device_bytes_per_rank)
            .saturating_add(rank0_extra_headroom);
        let use_live_pool_budget = capacity.state_slots.is_some() || cfg.uses_tiered_moe();
        let pool_weight_bytes = pool_weight_budget(
            use_live_pool_budget,
            weight_arena.used(),
            weight_arena.capacity(),
        );
        let available_for_pools = if use_live_pool_budget {
            device.mem_info()?.0
        } else {
            free_vram.saturating_sub(pool_weight_bytes.saturating_add(scratch_arena.capacity()))
        };
        let base_fixed = pool_weight_bytes
            .saturating_add(scratch_arena.capacity())
            .saturating_add(required_headroom)
            .saturating_add(req_kv_bytes);
        let effective_requested_state_slots =
            if cfg.uses_tiered_moe() && capacity.state_slots.is_some() {
                let max_tiered_serving_slots = if tp_size >= 2 {
                    if capacity.native_mtp && default_pages.unwrap_or(0) < 2048 {
                        16
                    } else {
                        18
                    }
                } else {
                    20
                };
                requested_state_slots.min(max_tiered_serving_slots)
            } else {
                requested_state_slots
            };
        let (state_budget_free, state_budget_fixed) = if capacity.state_slots.is_some() {
            // Serving can request many more slots than single-request defaults.
            // `available_for_pools` already reflects live arena capacity, `PleRankRuntime`,
            // and `RankTieredMoe`. `Qwen3VisionEncoder` is loaded lazily on the first
            // multimodal request (`ensure_vision_encoder_loaded`) and fits inside
            // `MIN_HEADROOM_BYTES` (1 GiB) when needed; only reserve dedicated ViT VRAM when
            // `vision_reserve` still leaves room for `effective_requested_state_slots`.
            let base_reserved = required_headroom.saturating_add(req_kv_bytes);
            let protected_state_slots = effective_requested_state_slots.max(MIN_ACTIVE_STATE_SLOTS);
            let min_state_bytes = bytes_per_state_slot.saturating_mul(protected_state_slots);
            let vision_reserve = if rank == 0
                && has_vision_weights(&cfg, archive)
                && default_pages.unwrap_or(0) < EXTREME_LONGCTX_MIN_PAGES
            {
                available_for_pools
                    .saturating_sub(base_reserved.saturating_add(min_state_bytes))
                    .min(VISION_ENCODER_LAZY_RESERVE_BYTES)
            } else {
                0
            };
            let reserved = base_reserved.saturating_add(vision_reserve);
            if available_for_pools
                < reserved
                    .saturating_add(bytes_per_state_slot.saturating_mul(MIN_ACTIVE_STATE_SLOTS))
            {
                return Err(loader_err(
                    "insufficient VRAM for serving state pool and headroom",
                ));
            }
            (available_for_pools, reserved)
        } else if cfg.uses_tiered_moe() {
            (
                available_for_pools,
                required_headroom.saturating_add(req_kv_bytes),
            )
        } else {
            (free_vram, base_fixed)
        };
        let total_state_slots = clamp_state_slots(
            effective_requested_state_slots,
            state_budget_free,
            state_budget_fixed,
            bytes_per_state_slot,
        );
        if total_state_slots < capacity.dflash_min_state_slots {
            return Err(loader_err_msg(format!(
                "insufficient memory for DFlash2: requires at least {} target state slots plus draft resident reserve; reduce KV capacity",
                capacity.dflash_min_state_slots,
            )));
        }
        let conv_dim = 2 * gdn_k_local + gdn_v_local;
        let mut state_pool = StateBundlePool::new(
            device_id,
            &gdn_layer_ids,
            total_state_slots,
            conv_dim,
            cfg.linear_conv_kernel_dim,
            local_v_heads,
            cfg.linear_key_head_dim,
            cfg.linear_value_head_dim,
        )?;
        if cfg.uses_ple() {
            state_pool.init_ple_state(
                cfg.hc_count * cfg.hidden_size,
                cfg.eos_token_ids.first().copied().unwrap_or(248044),
            )?;
        }
        let state_pool_bytes = total_state_slots * bytes_per_state_slot;

        let (plan_free_vram, plan_weight_bytes) = if use_live_pool_budget {
            let resident_model_bytes = weight_arena.used().saturating_add(actual_tiered_moe_bytes);
            let synthetic_free = available_for_pools
                .saturating_add(resident_model_bytes)
                .saturating_add(scratch_arena.capacity());
            (synthetic_free, resident_model_bytes)
        } else {
            (free_vram, pool_weight_bytes)
        };
        let fixed_without_kv = plan_weight_bytes
            .saturating_add(scratch_arena.capacity())
            .saturating_add(state_pool_bytes);
        let planned_kv_pages = plan_target_kv_pages(
            requested_kv_pages,
            capacity.state_slots.is_some(),
            capacity.dflash_reserve_bytes,
            capacity.native_mtp,
            plan_free_vram,
            fixed_without_kv,
            required_headroom,
            bytes_per_kv_page_all_layers,
        );
        let capacity_plan = CapacityPlanner::plan_with_headroom(
            total_vram,
            plan_free_vram,
            plan_weight_bytes,
            scratch_arena.capacity(),
            state_pool_bytes,
            bytes_per_kv_page_all_layers,
            Some(planned_kv_pages),
            required_headroom,
        )
        .map_err(|e| loader_err_ctx("CapacityPlanner failed", e))?;

        let mut kv_pool = KvPool::new_target_full_attn(
            device_id,
            &full_attn_layer_ids,
            local_kv_heads,
            cfg.head_dim,
            block_size,
            capacity_plan.max_kv_pages_per_layer,
            kv_policy,
        )?;
        if cfg.uses_qsa() {
            kv_pool.init_qsa_indexer_caches(cfg.indexer_head_dim, cfg.indexer_compress_ratio)?;
        }

        let decode_io =
            DecodeGraphIo::new(device_id, capacity_plan.max_kv_pages_per_layer, &stream)?;
        let spec_verify_io =
            SpecVerifyGraphIo::new(device_id, capacity_plan.max_kv_pages_per_layer, &stream)?;
        let batched_spec_verify_io = BatchedSpecVerifyGraphIo::new(device_id, &stream)?;
        let batched_decode_io =
            BatchedDecodeGraphIo::new(device_id, capacity_plan.max_kv_pages_per_layer, &stream)?;
        device.synchronize()?;

        Ok(Qwen3_5ModelRank {
            rank,
            tp_size,
            device,
            stream,
            cublas,
            cfg,
            weight_arena,
            scratch_arena,
            batch_meta_arena,
            marlin_ws,
            embed_tokens_ptr,
            host_embed_tokens,
            final_residual,
            lm_head,
            local_vocab_size,
            vocab_offset,
            layers,
            kv_pool,
            state_pool,
            decode_io,
            decode_graph: Vec::new(),
            spec_verify_io,
            spec_verify_graphs: Vec::new(),
            failed_spec_verify_keys: Vec::new(),
            batched_spec_verify_io,
            batched_spec_verify_graphs: Vec::new(),
            failed_batched_spec_verify_keys: Vec::new(),
            batched_decode_io,
            batched_decode_graphs: Vec::new(),
            aux_stream,
            aux_marlin_ws,
            aux_fork_event,
            aux_join_event,
            capacity_plan,
            tap_layer_ids: Vec::new(),
            tap_aux_bf16: None,
            speculative_logits: None,
            speculative_logits_rows: 0,
            last_logits_ptr: 0,
            last_logits_rows: 0,
            captured_decode_logits_ptr: 0,
            last_final_hidden_f16: None,
            max_tap_rows: 0,
            current_pos_plane_stride: 0,
            pending_visual_splice: None,
            vision_splice_idx: None,
            vision_splice_emb: None,
            last_norm_hidden_ptr: 0,
            last_norm_hidden_rows: 0,
            last_batch_state_slots: vec![0],
            tiered_moe,
            tiered_moe_hot_per_layer,
            ple_runtime,
            lora_runtime: None,
            is_capturing_graph: false,
        })
    }
}
