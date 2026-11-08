//! Per-device model residency, capacity planning, and loading.

use super::*;

/// Shared PLE lookup data and rank-local staging buffers.
#[cfg(feature = "cuda")]
pub struct PleRankRuntime {
    pub hasher: Arc<crate::ple::PleNgramHasher>,
    pub mmap_table: Arc<crate::ple::PleMmapTable>,
    pub f8_staging_dev: DeviceBuffer,
    pub f8_host_scratch: crate::cuda::MappedHostBuffer,
    pub f8_host_slot: usize,
    pub ids_scratch: Vec<i64>,
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
    pub lm_head: LinearProj,
    pub local_vocab_size: usize,
    pub vocab_offset: usize,
    pub layers: Vec<DecoderLayerWeights>,
    pub kv_pool: KvPool,
    pub state_pool: StateBundlePool,
    pub decode_io: DecodeGraphIo,
    pub decode_graph: Option<(bool, CudaGraph)>,
    pub spec_verify_io: SpecVerifyGraphIo,
    pub spec_verify_graph: Option<((usize, bool), CudaGraph)>,
    pub batched_decode_io: BatchedDecodeGraphIo,
    pub batched_decode_graphs: Vec<((usize, bool, DecodeGraphSampleMode), u64, CudaGraph)>,
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
    pub captured_spec_logits_ptr: u64,
    pub last_final_hidden_f16: Option<DeviceBuffer>,
    pub max_tap_rows: usize,
    pub current_pos_plane_stride: i32,
    pub pending_visual_splice: Option<(u64, u64, usize)>,
    pub last_norm_hidden_ptr: u64,
    pub last_norm_hidden_rows: usize,
    pub tiered_moe: Option<crate::moe_offload::RankTieredMoe>,
    pub tiered_moe_hot_per_layer: usize,
    pub ple_runtime: Option<PleRankRuntime>,
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

#[cfg(feature = "cuda")]
pub(super) type SharedVocabHost = std::sync::OnceLock<Result<(Vec<u16>, Option<RawLinearHost>)>>;
#[cfg(feature = "cuda")]
#[derive(Clone, Copy, Default)]
pub(super) struct PoolCapacity {
    pub(super) kv_pages: Option<usize>,
    pub(super) state_slots: Option<usize>,
    pub(super) rank0_extra_headroom_bytes: usize,
}

/// Number of state slots allocated per active sequence in serving mode (three private
/// decode/speculative rollback slots plus one borrowed radix-prefix checkpoint slot).
#[cfg(any(test, feature = "cuda"))]
const SERVING_STATE_SLOTS_PER_SEQ: usize = 4;

/// Minimum total serving state slots across the pool even when `max_num_seqs == 1`.
#[cfg(any(test, feature = "cuda"))]
const MIN_SERVING_STATE_SLOTS_TOTAL: usize = 8;

/// Minimum state-bundle slots required to execute single-sequence prefill + speculative verify.
#[cfg(any(test, feature = "cuda"))]
const MIN_ACTIVE_STATE_SLOTS: usize = 3;

/// Default target KV page count (`256 * 16 = 4,096` tokens) when `--kv-pages` is omitted.
#[cfg(feature = "cuda")]
const DEFAULT_TARGET_KV_PAGES: usize = 256;

/// Tokens per paged-attention KV block (`KvPool` page size).
#[cfg(feature = "cuda")]
const KV_PAGE_BLOCK_TOKENS: usize = 16;

/// Default state-bundle slot count for PLE / large (`>= 15 GiB`) checkpoints in non-serving mode.
#[cfg(feature = "cuda")]
const PLE_OR_LARGE_MODEL_DEFAULT_STATE_SLOTS: usize = 12;

/// Default state-bundle slot count for smaller non-PLE checkpoints in non-serving mode.
#[cfg(feature = "cuda")]
const STANDARD_DEFAULT_STATE_SLOTS: usize = 24;

/// Estimated weight footprint threshold (`15 GiB`) above which non-serving mode uses
/// `PLE_OR_LARGE_MODEL_DEFAULT_STATE_SLOTS` and single-GPU scratch uses
/// [`crate::scheduler::LARGE_MODEL_TP1_SCRATCH_ARENA_BYTES`].
pub const LARGE_MODEL_WEIGHT_THRESHOLD_BYTES: usize = 15 * 1024 * 1024 * 1024;

/// Dedicated VRAM reserve (`1.5 GiB`) set aside on rank 0 for lazy `Qwen3VisionEncoder`
/// loading when serving multimodal checkpoints with sufficient free VRAM.
#[cfg(feature = "cuda")]
const VISION_ENCODER_LAZY_RESERVE_BYTES: usize = 1536 * 1024 * 1024;

/// Layer-count threshold (`64` layers) for applying [`crate::cuda::EXTREME_LONGCTX_HEADROOM_BYTES`].
pub const EXTREME_LONGCTX_MIN_LAYERS: usize = 64;

/// Page-count threshold (`14,336` pages = `224K` tokens) for extreme long-context headroom.
pub const EXTREME_LONGCTX_MIN_PAGES: usize = 14_336;

/// Rounding quantum (`2 MiB`) for `weight_arena` capacity estimation.
#[cfg(feature = "cuda")]
const WEIGHT_ARENA_ALIGNMENT_BYTES: usize = 2 * 1024 * 1024;

/// Minimum `weight_arena` allocation (`128 MiB`) for tiny synthetic models.
#[cfg(feature = "cuda")]
const MIN_WEIGHT_ARENA_BYTES: usize = 128 * 1024 * 1024;

/// Default alignment/padding margin (`96 MiB`) added to raw estimated weight bytes.
#[cfg(feature = "cuda")]
const WEIGHT_ARENA_DEFAULT_MARGIN_BYTES: usize = 96 * 1024 * 1024;

/// Compact alignment/padding margin (`32 MiB`) added to raw estimated weight bytes under extreme long-context budgets.
#[cfg(feature = "cuda")]
const WEIGHT_ARENA_EXTREME_LONGCTX_MARGIN_BYTES: usize = 32 * 1024 * 1024;

/// Minimum temporary repack staging buffer (`128 MiB`) allocated during weight upload.
#[cfg(feature = "cuda")]
const MIN_REPACK_STAGING_BYTES: usize = 128 * 1024 * 1024;

/// Multiplier (`3×` largest packed tensor) used to size the temporary repack staging buffer.
#[cfg(feature = "cuda")]
const REPACK_STAGING_TENSOR_MULTIPLIER: usize = 3;

/// Extra safety margin (`16 MiB`) checked before initial weight + scratch arena boot allocations.
#[cfg(feature = "cuda")]
const MIN_BOOT_MARGIN_BYTES: usize = 16 * 1024 * 1024;

/// Capacity (`16 MiB`) of the per-rank `batch_meta_arena` used for batch descriptors and page tables.
#[cfg(feature = "cuda")]
const BATCH_META_ARENA_BYTES: usize = 16 * 1024 * 1024;

/// Alignment quantum (`8` slots) for per-layer GPU hot expert cache sizing.
pub const TIERED_MOE_HOT_SLOT_ALIGNMENT: usize = 8;

/// Minimum total model experts (`64`) required before applying full tiered-MoE VRAM planning.
pub const TIERED_MOE_MIN_MODEL_EXPERTS: usize = 64;

/// Maximum local experts per rank (`80`) treated as a tiny synthetic test model (`hot = local / 2`).
pub const TIERED_MOE_SYNTHETIC_LOCAL_EXPERTS_MAX: usize = 80;

/// Minimum per-layer GPU hot expert cache capacity on `TP=1` under extreme KV pressure.
pub const TIERED_MOE_TP1_MIN_HOT_EXPERTS: usize = 16;

/// Minimum per-layer GPU hot expert cache capacity per rank on `TP >= 2` under extreme KV pressure.
pub const TIERED_MOE_TP2_MIN_HOT_EXPERTS: usize = 32;

/// `base_headroom + kv_pool` threshold (`1.5 GiB`) separating compact `TP=1` KV budgets (`256` pages)
/// from multi-thousand-page serving KV budgets (`4,096` pages).
pub const TIERED_MOE_TP1_COMPACT_KV_THRESHOLD_BYTES: usize = 1536 * 1024 * 1024;

/// Auto-tuned `TP=1` per-layer GPU hot expert ceiling for compact KV budgets (`<= 1.5 GiB`),
/// leaving `>= 1.35 GiB` post-target VRAM for `MTP` (`k_hot = 192`) and CUDA graphs on 24 GiB cards.
pub const TIERED_MOE_TP1_COMPACT_KV_DEFAULT_HOT: usize = 104;

/// Auto-tuned `TP=1` per-layer GPU hot expert ceiling for larger serving KV budgets (`> 1.5 GiB`).
pub const TIERED_MOE_TP1_SERVING_KV_DEFAULT_HOT: usize = 96;

/// Auto-tuned `TP=2` per-rank GPU hot expert ceiling (`256 / 512` experts across 2 GPUs),
/// leaving room for `100%` VRAM-resident `MTP` (`256 / 256` local experts) plus `>= 1.4 GiB` headroom.
pub const TIERED_MOE_TP2_DEFAULT_HOT: usize = 128;

/// Extra expert-stride slots (`512`) reserved on `TP=1` / `TP=2` for `TieredMoeWorkspace` staging
/// and downstream `MTP` expert cache allocation.
pub const TIERED_MOE_STAGE_RESERVE_SLOTS: usize = 512;

/// Extra expert-stride slots (`256`) reserved on `TP >= 4` where all local experts fit in VRAM.
pub const TIERED_MOE_TP4_STAGE_RESERVE_SLOTS: usize = 256;

/// Fixed post-target VRAM reserve (`640 MiB`) on `TP=1` / `TP=2` for `MTP` non-expert weights,
/// `QSA` workspace buffers, and CUDA graph capture pools.
pub const TIERED_MOE_MTP_AND_GRAPH_RESERVE_BYTES: usize = 640 * 1024 * 1024;

/// Fixed extra VRAM reserve (`64 MiB`) for tiny synthetic tiered-MoE unit-test models.
pub const TIERED_MOE_SYNTHETIC_EXTRA_RESERVE_BYTES: usize = 64 * 1024 * 1024;

/// Double-buffered H2D staging capacity (`96` experts) for `RankTieredMoe` on `TP=1` / `TP=2`.
#[cfg(feature = "cuda")]
pub(super) const TIERED_MOE_STAGE_CAPACITY_DEFAULT: usize = 96;

/// Double-buffered H2D staging capacity (`32` experts) for `RankTieredMoe` on `TP >= 4`.
#[cfg(feature = "cuda")]
pub(super) const TIERED_MOE_STAGE_CAPACITY_TP4: usize = 32;

/// Minimum per-row byte stride (`2,560 B`) for `PLE` FP8 staging buffers.
#[cfg(feature = "cuda")]
pub(super) const MIN_PLE_ROW_STAGING_BYTES: usize = 2560;

/// Free VRAM headroom (`512 MiB`) preserved after placing `100%` of `MTP` local experts in VRAM on `TP >= 2`.
pub const MTP_FULL_RESIDENCY_HEADROOM_BYTES: usize = 512 * 1024 * 1024;

/// Free VRAM headroom (`256 MiB`) preserved after allocating a partial `MTP` `HotExpertCache`
/// so `QSA` workspaces, `FullAttentionLayerWeights`, and `MTP` CUDA graphs fit comfortably.
pub const MTP_PARTIAL_CACHE_HEADROOM_BYTES: usize = 256 * 1024 * 1024;

/// Maximum `MTP` GPU hot expert count (`192`) on `TP=1` (`local_experts = 512`).
///
/// Keeping `k_hot < local_experts` on `TP=1` preserves `StreamedPrefill` chunking during
/// `MTP::ingest_kv_only` instead of dispatching `num_experts = 512` with `group_size = 32`
/// into `w34_moe_w4a16_mlp_f16` (which falls back to GEMV when `total_pairs < 512`).
pub const MTP_TP1_STREAMED_PREFILL_MAX_HOT: usize = 192;

/// Intermediate `MTP` GPU hot expert count (`128`) when free VRAM cannot fit `192` slots.
pub const MTP_MEDIUM_HOT_EXPERTS: usize = 128;

/// Minimum free VRAM (`512 MiB`) required to select [`MTP_MEDIUM_HOT_EXPERTS`].
pub const MTP_MEDIUM_MIN_FREE_VRAM_BYTES: usize = 512 * 1024 * 1024;

/// Compact `MTP` GPU hot expert count (`64`) under tight VRAM budgets.
pub const MTP_COMPACT_HOT_EXPERTS: usize = 64;

/// Fallback `MTP` GPU hot expert count (`16`) for tiny synthetic models (`local_experts < 64`).
pub const MTP_FALLBACK_HOT_EXPERTS: usize = 16;

/// Returns the auto-tuned per-layer GPU hot expert ceiling for a tiered-MoE target rank.
#[inline]
pub fn default_tiered_moe_hot_ceiling(
    tp_size: usize,
    local_experts: usize,
    base_headroom_plus_kv_bytes: usize,
) -> usize {
    if tp_size >= 4 {
        local_experts
    } else if tp_size == 2 {
        TIERED_MOE_TP2_DEFAULT_HOT.min(local_experts)
    } else if base_headroom_plus_kv_bytes <= TIERED_MOE_TP1_COMPACT_KV_THRESHOLD_BYTES {
        TIERED_MOE_TP1_COMPACT_KV_DEFAULT_HOT.min(local_experts)
    } else {
        TIERED_MOE_TP1_SERVING_KV_DEFAULT_HOT.min(local_experts)
    }
}

/// Computes the `MTP` proposer's GPU hot expert cache capacity (`k_hot`) from live free VRAM.
///
/// On `TP >= 2` (`local_experts <= 256`), places `100%` of local `MTP` experts in VRAM when
/// `free_vram_bytes` covers all local experts plus [`MTP_FULL_RESIDENCY_HEADROOM_BYTES`].
/// On `TP == 1` (`local_experts == 512`), caps at [`MTP_TP1_STREAMED_PREFILL_MAX_HOT`] (`192`)
/// when affordable so `MTP` prompt ingestion retains fast `StreamedPrefill` chunking.
pub fn plan_mtp_tiered_moe_hot_experts(
    tp_size: usize,
    local_experts: usize,
    expert_stride_bytes: usize,
    free_vram_bytes: usize,
) -> usize {
    let local_experts = local_experts.max(1);
    let slots_all_need = local_experts
        .saturating_mul(expert_stride_bytes)
        .saturating_add(MTP_FULL_RESIDENCY_HEADROOM_BYTES);
    let slots_tp1_max_need = MTP_TP1_STREAMED_PREFILL_MAX_HOT
        .saturating_mul(expert_stride_bytes)
        .saturating_add(MTP_PARTIAL_CACHE_HEADROOM_BYTES);
    let slots_medium_need = MTP_MEDIUM_HOT_EXPERTS
        .saturating_mul(expert_stride_bytes)
        .saturating_add(MTP_PARTIAL_CACHE_HEADROOM_BYTES);
    if tp_size > 1 && free_vram_bytes >= slots_all_need {
        local_experts
    } else if tp_size == 1
        && local_experts >= MTP_TP1_STREAMED_PREFILL_MAX_HOT
        && free_vram_bytes >= slots_tp1_max_need
    {
        MTP_TP1_STREAMED_PREFILL_MAX_HOT
    } else if local_experts >= MTP_MEDIUM_HOT_EXPERTS
        && free_vram_bytes >= slots_medium_need.max(MTP_MEDIUM_MIN_FREE_VRAM_BYTES)
    {
        MTP_MEDIUM_HOT_EXPERTS
    } else if local_experts >= MTP_COMPACT_HOT_EXPERTS {
        MTP_COMPACT_HOT_EXPERTS
    } else {
        MTP_FALLBACK_HOT_EXPERTS.min(local_experts).max(1)
    }
}

/// Selects the per-rank transient `scratch_arena` capacity (`336 MiB`, `384 MiB`,
/// `512 MiB`, or `1,056 MiB`) based on model geometry, `tp_size`, estimated weight
/// footprint, and requested KV pages.
pub fn plan_scratch_arena_bytes(
    cfg: &ModelConfig,
    tp_size: usize,
    est_weight_bytes: usize,
    requested_kv_pages: Option<usize>,
) -> usize {
    if tp_size == 1
        && cfg.num_hidden_layers >= EXTREME_LONGCTX_MIN_LAYERS
        && requested_kv_pages.unwrap_or(0) >= EXTREME_LONGCTX_MIN_PAGES
    {
        crate::scheduler::EXTREME_LONGCTX_SCRATCH_ARENA_BYTES
    } else if tp_size == 1 && est_weight_bytes >= LARGE_MODEL_WEIGHT_THRESHOLD_BYTES {
        crate::scheduler::LARGE_MODEL_TP1_SCRATCH_ARENA_BYTES
    } else if cfg.uses_tiered_moe()
        && requested_kv_pages.unwrap_or(0) <= crate::scheduler::TIERED_MOE_FULL_SCRATCH_MAX_KV_PAGES
    {
        crate::scheduler::TIERED_MOE_SCRATCH_ARENA_BYTES
    } else {
        crate::scheduler::DEFAULT_SCRATCH_ARENA_BYTES
    }
}

#[cfg(any(test, feature = "cuda"))]
pub(super) fn serving_state_slots(max_num_seqs: usize) -> Result<usize> {
    max_num_seqs
        .checked_mul(SERVING_STATE_SLOTS_PER_SEQ)
        .filter(|&slots| slots != 0)
        .map(|slots| slots.max(MIN_SERVING_STATE_SLOTS_TOTAL))
        .ok_or_else(|| {
            loader_err("serving max_num_seqs must be nonzero and fit four state slots per sequence")
        })
}

/// Computes the per-layer hot expert cache capacity (`K_hot`) and total VRAM reserve
/// for tiered MoE models (`qwen4_exp`), adapting `K_hot` down from `112` on `TP=2`
/// when a larger KV/state pool is requested so multi-thousand-page contexts fit in 24 GiB.
pub fn plan_tiered_moe_hot_per_layer(
    cfg: &ModelConfig,
    tp_size: usize,
    available_vram_after_weights_and_scratch: usize,
    base_headroom_plus_kv_bytes: usize,
    min_state_pool_bytes: usize,
) -> (usize, usize) {
    if !cfg.uses_tiered_moe() {
        return (0, 0);
    }
    let local_experts = (cfg.num_experts / tp_size.max(1)).max(1);
    let default_hot =
        default_tiered_moe_hot_ceiling(tp_size, local_experts, base_headroom_plus_kv_bytes);
    let exp_bytes = cfg.moe_expert_pack_layout().map_or(0, |l| l.stride);
    if (tp_size < 4 && local_experts <= TIERED_MOE_SYNTHETIC_LOCAL_EXPERTS_MAX)
        || cfg.num_experts < TIERED_MOE_MIN_MODEL_EXPERTS
        || exp_bytes == 0
        || cfg.num_hidden_layers == 0
    {
        let hot = (local_experts / 2).max(1);
        let reserve = (cfg.num_hidden_layers * hot + TIERED_MOE_STAGE_RESERVE_SLOTS) * exp_bytes
            + TIERED_MOE_SYNTHETIC_EXTRA_RESERVE_BYTES;
        return (hot, reserve);
    }
    let min_hot = if tp_size >= 2 {
        TIERED_MOE_TP2_MIN_HOT_EXPERTS.min(default_hot)
    } else {
        TIERED_MOE_TP1_MIN_HOT_EXPERTS.min(default_hot)
    };
    let extra_reserve = if tp_size >= 4 {
        TIERED_MOE_TP4_STAGE_RESERVE_SLOTS * exp_bytes
    } else {
        TIERED_MOE_STAGE_RESERVE_SLOTS * exp_bytes + TIERED_MOE_MTP_AND_GRAPH_RESERVE_BYTES
    };
    let fixed_overhead = base_headroom_plus_kv_bytes
        .saturating_add(min_state_pool_bytes)
        .saturating_add(extra_reserve);
    let per_hot_step = cfg.num_hidden_layers * exp_bytes;
    let hot = if available_vram_after_weights_and_scratch > fixed_overhead && per_hot_step > 0 {
        let max_affordable =
            (available_vram_after_weights_and_scratch - fixed_overhead) / per_hot_step;
        let aligned =
            (max_affordable / TIERED_MOE_HOT_SLOT_ALIGNMENT) * TIERED_MOE_HOT_SLOT_ALIGNMENT;
        aligned.clamp(min_hot, default_hot)
    } else {
        min_hot
    };
    let reserve = (cfg.num_hidden_layers * hot) * exp_bytes + extra_reserve;
    (hot, reserve)
}

// Preserve the minimum three-slot execution budget and the existing VRAM clamp.
#[cfg(any(test, feature = "cuda"))]
pub(super) fn clamp_state_slots(
    requested: usize,
    free: usize,
    fixed: usize,
    per_slot: usize,
) -> usize {
    if per_slot == 0 {
        requested
    } else if free > fixed {
        ((free - fixed) / per_slot).clamp(MIN_ACTIVE_STATE_SLOTS, requested)
    } else {
        MIN_ACTIVE_STATE_SLOTS.min(requested)
    }
}

#[cfg(feature = "cuda")]
impl Qwen3_5ModelRank {
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
                    let skinny_dequant = if !is_expert
                        && info.shape.len() == 2
                        && (name.ends_with(".weight_packed") || name.ends_with(".qweight"))
                    {
                        let (n, k) = if cfg.quant.is_awq_packing() && name.ends_with(".qweight") {
                            (info.shape[1] * 8, info.shape[0])
                        } else if name.ends_with(".qweight") {
                            (info.shape[1], info.shape[0] * pack_factor)
                        } else {
                            (info.shape[0], info.shape[1] * pack_factor)
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
                        let gs = cfg.quant.group_size_or(local_k);
                        let gs_ok = ((cfg.quant.is_channelwise() && cfg.quant.bits() == 8)
                            || matches!(gs, 32 | 64 | 128))
                            && (gs == 0 || local_k.is_multiple_of(gs));
                        if !marlin_ok || !gs_ok {
                            Some(n * k * 2)
                        } else {
                            None
                        }
                    } else {
                        None
                    };
                    skinny_dequant.unwrap_or(info.len_bytes as usize)
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
        Self::validate_tp_geometry(cfg, tp_size)?;
        if cfg.head_dim != 256 && !matches!(kv_policy, KvFormatPolicy::Fp16 | KvFormatPolicy::Bf16)
        {
            return Err(loader_err_msg(format!(
                "unsupported KV format {kv_policy:?} for head_dim={} (TQ4 and Q8_0 require head_dim=256; use --kv fp16)",
                cfg.head_dim
            )));
        }
        let requested_kv_pages = capacity.kv_pages;
        prepare_execution_plan(device_id)?;
        let device = CudaDevice::set_current(device_id)?;
        let dev_props = device.props()?;
        dev_props.validate_supported()?;
        let marlin_ws_bytes = dev_props.marlin_workspace_bytes();
        let stream = CudaStream::new(device_id)?;
        let cublas = CublasHandle::new(&stream)?;
        let (free_vram, total_vram) = device.mem_info()?;

        let est_weight_bytes =
            Self::estimate_weight_arena_bytes(cfg, archive, tp_size, requested_kv_pages);
        let max_packed_tensor_bytes = archive
            .iter_entries()
            .filter(|(name, info)| {
                matches!(info.dtype.as_str(), "I32" | "U32")
                    && !name.contains(".ple_embedding.")
                    && !(cfg.uses_tiered_moe() && name.contains(".experts."))
            })
            .map(|(_, info)| info.len_bytes as usize)
            .max()
            .unwrap_or(0);
        let staging_bytes = MIN_REPACK_STAGING_BYTES
            .max(max_packed_tensor_bytes.saturating_mul(REPACK_STAGING_TENSOR_MULTIPLIER));
        let scratch_bytes =
            plan_scratch_arena_bytes(cfg, tp_size, est_weight_bytes, requested_kv_pages);
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
        let owned_vocab;
        let (embed_f16, shared_lm_opt): (&[u16], Option<&RawLinearHost>) =
            if let Some(cell) = shared_vocab {
                let res = cell.get_or_init(|| {
                    let emb = Self::load_embed_tokens_f16(archive, cfg)?;
                    let lm = if cfg.tie_word_embeddings {
                        None
                    } else {
                        Some(RawLinearHost::load(archive, "lm_head", &cfg.quant)?)
                    };
                    Ok((emb, lm))
                });
                let (emb, lm) = res.as_ref().map_err(Clone::clone)?;
                (emb.as_slice(), lm.as_ref())
            } else {
                owned_vocab = Self::load_embed_tokens_f16(archive, cfg)?;
                (owned_vocab.as_slice(), None)
            };
        let (embed_tokens_ptr, host_embed_tokens) =
            if cfg.uses_tiered_moe() && !cfg.tie_word_embeddings {
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

        // 3. lm_head (sharded along vocab N across TP ranks)
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
            LinearProj::DenseF16 {
                weight_ptr: ptr,
                size_n: local_vocab_size,
                size_k: cfg.hidden_size,
            }
        } else if let Some(raw_lm) = shared_lm_opt {
            let sharded = raw_lm.slice_rows(vocab_offset, vocab_offset + local_vocab_size);
            sharded.upload(&mut weight_arena, &stream, &repack_staging)?
        } else {
            let raw_lm = RawLinearHost::load(archive, "lm_head", &cfg.quant)?;
            let sharded = raw_lm.slice_rows(vocab_offset, vocab_offset + local_vocab_size);
            sharded.upload(&mut weight_arena, &stream, &repack_staging)?
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
        stream.synchronize()?;

        // 5. StateBundlePool & KvPool via CapacityPlanner
        let block_size = KV_PAGE_BLOCK_TOKENS;
        let mut bytes_per_kv_page_all_layers = 0usize;
        for ord in 0..full_attn_layer_ids.len() {
            let (kf, vf, _, _) = kv_policy.resolve_layer(ord, full_attn_layer_ids.len());
            let kb = crate::cache::kv_format_row_bytes(kf, cfg.head_dim);
            let vb = crate::cache::kv_format_row_bytes(vf, cfg.head_dim);
            bytes_per_kv_page_all_layers += local_kv_heads * block_size * (kb + vb);
            if cfg.uses_qsa() {
                let comp_rows = block_size / cfg.indexer_compress_ratio.max(1);
                bytes_per_kv_page_all_layers +=
                    cfg.indexer_kv_heads * (block_size + comp_rows) * cfg.indexer_head_dim * 2;
            }
        }

        let conv_dim = 2 * gdn_k_local + gdn_v_local;
        let ple_slot_bytes = if cfg.uses_ple() {
            cfg.ple_layer_ids.len() * cfg.hc_count * cfg.hidden_size * 9 * 2
        } else {
            0
        };
        let bytes_per_state_slot = gdn_layer_ids.len()
            * (conv_dim * (cfg.linear_conv_kernel_dim - 1) * 2
                + local_v_heads * cfg.linear_value_head_dim * cfg.linear_key_head_dim * 4)
            + ple_slot_bytes;
        let default_state_slots =
            if cfg.uses_ple() || est_weight_bytes >= LARGE_MODEL_WEIGHT_THRESHOLD_BYTES {
                PLE_OR_LARGE_MODEL_DEFAULT_STATE_SLOTS
            } else {
                STANDARD_DEFAULT_STATE_SLOTS
            };
        let requested_state_slots = capacity.state_slots.unwrap_or(default_state_slots);
        let default_pages = requested_kv_pages.or(Some(DEFAULT_TARGET_KV_PAGES));
        let req_kv_bytes =
            default_pages.unwrap_or(DEFAULT_TARGET_KV_PAGES) * bytes_per_kv_page_all_layers;
        let base_headroom = if tp_size == 1
            && cfg.num_hidden_layers >= EXTREME_LONGCTX_MIN_LAYERS
            && default_pages.unwrap_or(0) >= EXTREME_LONGCTX_MIN_PAGES
        {
            crate::cuda::EXTREME_LONGCTX_HEADROOM_BYTES
        } else {
            crate::cuda::MIN_HEADROOM_BYTES
        };
        let post_weight_available = if capacity.state_slots.is_some() || cfg.uses_tiered_moe() {
            device.mem_info()?.0
        } else {
            free_vram.saturating_sub(weight_arena.used().saturating_add(scratch_arena.capacity()))
        };
        let min_state_slots_for_moe = if capacity.state_slots.is_some() {
            requested_state_slots.clamp(
                MIN_ACTIVE_STATE_SLOTS,
                PLE_OR_LARGE_MODEL_DEFAULT_STATE_SLOTS,
            )
        } else if tp_size >= 2 {
            requested_state_slots
        } else {
            requested_state_slots.min(PLE_OR_LARGE_MODEL_DEFAULT_STATE_SLOTS / 2)
        };
        let (tiered_moe_hot_per_layer, tiered_moe_reserve) = plan_tiered_moe_hot_per_layer(
            cfg,
            tp_size,
            post_weight_available,
            base_headroom
                .saturating_add(req_kv_bytes)
                .saturating_add(if rank == 0 {
                    capacity.rank0_extra_headroom_bytes
                } else {
                    0
                }),
            bytes_per_state_slot.saturating_mul(min_state_slots_for_moe),
        );
        let required_headroom = base_headroom
            .saturating_add(tiered_moe_reserve)
            .saturating_add(if rank == 0 {
                capacity.rank0_extra_headroom_bytes
            } else {
                0
            });
        let base_fixed = weight_arena
            .used()
            .saturating_add(scratch_arena.capacity())
            .saturating_add(required_headroom)
            .saturating_add(req_kv_bytes);
        let (state_budget_free, state_budget_fixed) = if capacity.state_slots.is_some() {
            // Serving can request many more slots than single-request defaults.
            // Account for live arena capacity (not just used weight bytes) and
            // all auxiliary allocations before spending the remaining VRAM.
            // Note: `Qwen3VisionEncoder` is loaded lazily on the first multimodal
            // request (`ensure_vision_encoder_loaded`) and fits inside `MIN_HEADROOM_BYTES`
            // (1 GiB) when needed; only reserve dedicated ViT VRAM when `vision_reserve`
            // still leaves room for all `requested_state_slots`.
            let available = post_weight_available;
            let archive_has_vision = cfg.vision_config.is_some()
                && (archive
                    .resolve_name("model.visual.patch_embed.proj.weight")
                    .is_some()
                    || archive
                        .resolve_name("visual.patch_embed.proj.weight")
                        .is_some());
            let base_reserved = required_headroom.saturating_add(req_kv_bytes);
            let min_state_bytes = bytes_per_state_slot.saturating_mul(min_state_slots_for_moe);
            let vision_reserve = if rank == 0
                && archive_has_vision
                && default_pages.unwrap_or(0) < EXTREME_LONGCTX_MIN_PAGES
            {
                available
                    .saturating_sub(base_reserved.saturating_add(min_state_bytes))
                    .min(VISION_ENCODER_LAZY_RESERVE_BYTES)
            } else {
                0
            };
            let reserved = base_reserved.saturating_add(vision_reserve);
            if available
                < reserved
                    .saturating_add(bytes_per_state_slot.saturating_mul(MIN_ACTIVE_STATE_SLOTS))
            {
                return Err(loader_err(
                    "insufficient VRAM for serving state pool and headroom",
                ));
            }
            (available, reserved)
        } else {
            (free_vram, base_fixed)
        };
        let total_state_slots = clamp_state_slots(
            requested_state_slots,
            state_budget_free,
            state_budget_fixed,
            bytes_per_state_slot,
        );
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

        let capacity_plan = CapacityPlanner::plan_with_headroom(
            total_vram,
            free_vram,
            weight_arena.used(),
            scratch_arena.capacity(),
            state_pool_bytes,
            bytes_per_kv_page_all_layers,
            default_pages,
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
        let batched_decode_io =
            BatchedDecodeGraphIo::new(device_id, capacity_plan.max_kv_pages_per_layer, &stream)?;
        let aux_stream = CudaStream::new(device_id)?;
        let aux_marlin_ws = DeviceBuffer::alloc(device_id, marlin_ws_bytes)?;
        let aux_fork_event = crate::cuda::CudaEvent::new(device_id, false)?;
        let aux_join_event = crate::cuda::CudaEvent::new(device_id, false)?;
        device.synchronize()?;

        Ok(Self {
            rank,
            tp_size,
            device,
            stream,
            cublas,
            cfg: cfg.clone(),
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
            decode_graph: None,
            spec_verify_io,
            spec_verify_graph: None,
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
            captured_spec_logits_ptr: 0,
            last_final_hidden_f16: None,
            max_tap_rows: 0,
            current_pos_plane_stride: 0,
            pending_visual_splice: None,
            last_norm_hidden_ptr: 0,
            last_norm_hidden_rows: 0,
            tiered_moe: None,
            tiered_moe_hot_per_layer,
            ple_runtime: None,
            is_capturing_graph: false,
        })
    }
}
