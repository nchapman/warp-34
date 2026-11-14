//! Per-rank VRAM capacity, tiered-MoE hot cache, and host-embedding offload planners.

use super::*;

#[cfg(feature = "cuda")]
pub(super) type SharedVocabHost = std::sync::OnceLock<Result<(Vec<u16>, Option<RawLinearHost>)>>;

pub use crate::lora::LoraCapacityReserve;

#[cfg(feature = "cuda")]
#[derive(Clone, Copy, Default)]
pub(super) struct PoolCapacity {
    pub(super) kv_pages: Option<usize>,
    pub(super) state_slots: Option<usize>,
    pub(super) rank0_extra_headroom_bytes: usize,
    pub(super) native_mtp: bool,
    /// Metadata-validated per-rank DFlash resident reserve at the requested page bound.
    pub(super) dflash_reserve_bytes: usize,
    pub(super) dflash_min_state_slots: usize,
    /// Pre-computed per-rank LoRA VRAM reserve and execution traits.
    pub(super) lora_reserve: LoraCapacityReserve,
}

/// Number of state slots allocated per active sequence in serving mode (three private
/// decode/speculative rollback slots plus one borrowed radix-prefix checkpoint slot).
#[cfg(any(test, feature = "cuda"))]
pub(super) const SERVING_STATE_SLOTS_PER_SEQ: usize = 4;

/// Minimum total serving state slots across the pool even when `max_num_seqs == 1`.
#[cfg(any(test, feature = "cuda"))]
pub(super) const MIN_SERVING_STATE_SLOTS_TOTAL: usize = 8;

/// Minimum state-bundle slots required to execute single-sequence prefill + speculative verify.
#[cfg(any(test, feature = "cuda"))]
pub(super) const MIN_ACTIVE_STATE_SLOTS: usize = 3;

/// Default target KV page count (`256 * 16 = 4,096` tokens) when `--kv-pages` is omitted in non-serving mode.
pub const DEFAULT_TARGET_KV_PAGES: usize = 256;

/// Default serving KV page target (`2,048 * 16 = 32,768` tokens) when `--kv-pages` is omitted in serving mode.
pub const DEFAULT_SERVING_TARGET_KV_PAGES: usize = 2048;

/// Extra post-pool VRAM breathing room (`1.5 GiB`) preserved when auto-sizing serving KV pages.
pub const AUTO_SERVING_KV_EXTRA_HEADROOM_BYTES: usize = 1536 * 1024 * 1024;

/// Per-rank transient `scratch_arena` capacity (`960 MiB`) when a multi-pass `midloop` LoRA adapter is attached.
pub const MIDLOOP_LORA_SCRATCH_ARENA_BYTES: usize = 960 * 1024 * 1024;

/// Tokens per paged-attention KV block (`KvPool` page size).
#[cfg(any(test, feature = "cuda"))]
pub(super) const KV_PAGE_BLOCK_TOKENS: usize = 16;

/// Default state-bundle slot count for PLE / large (`>= 15 GiB`) checkpoints in non-serving mode.
#[cfg(feature = "cuda")]
pub(super) const PLE_OR_LARGE_MODEL_DEFAULT_STATE_SLOTS: usize = 12;

/// Default state-bundle slot count for smaller non-PLE checkpoints in non-serving mode.
#[cfg(feature = "cuda")]
pub(super) const STANDARD_DEFAULT_STATE_SLOTS: usize = 24;

/// Estimated weight footprint threshold (`15 GiB`) above which non-serving mode uses
/// `PLE_OR_LARGE_MODEL_DEFAULT_STATE_SLOTS` and single-GPU scratch uses
/// [`crate::scheduler::LARGE_MODEL_TP1_SCRATCH_ARENA_BYTES`].
pub const LARGE_MODEL_WEIGHT_THRESHOLD_BYTES: usize = 15 * 1024 * 1024 * 1024;

/// Dedicated VRAM reserve (`1.5 GiB`) set aside on rank 0 for lazy `Qwen3VisionEncoder`
/// loading when serving multimodal checkpoints with sufficient free VRAM.
#[cfg(feature = "cuda")]
pub(super) const VISION_ENCODER_LAZY_RESERVE_BYTES: usize = 1536 * 1024 * 1024;

/// Layer-count threshold (`64` layers) for applying [`crate::cuda::EXTREME_LONGCTX_HEADROOM_BYTES`].
pub const EXTREME_LONGCTX_MIN_LAYERS: usize = 64;

/// Page-count threshold (`14,336` pages = `224K` tokens) for extreme long-context headroom.
pub const EXTREME_LONGCTX_MIN_PAGES: usize = 14_336;

/// Rounding quantum (`2 MiB`) for `weight_arena` capacity estimation.
pub const WEIGHT_ARENA_ALIGNMENT_BYTES: usize = 2 * 1024 * 1024;

/// Minimum `weight_arena` allocation (`128 MiB`) for tiny synthetic models.
pub const MIN_WEIGHT_ARENA_BYTES: usize = 128 * 1024 * 1024;

/// Default alignment/padding margin (`96 MiB`) added to raw estimated weight bytes.
#[cfg(feature = "cuda")]
pub(super) const WEIGHT_ARENA_DEFAULT_MARGIN_BYTES: usize = 96 * 1024 * 1024;

/// Compact alignment/padding margin (`32 MiB`) added to raw estimated weight bytes under extreme long-context budgets.
#[cfg(feature = "cuda")]
pub(super) const WEIGHT_ARENA_EXTREME_LONGCTX_MARGIN_BYTES: usize = 32 * 1024 * 1024;

/// Minimum temporary repack staging buffer (`128 MiB`) allocated during weight upload.
#[cfg(feature = "cuda")]
pub(super) const MIN_REPACK_STAGING_BYTES: usize = 128 * 1024 * 1024;

/// Multiplier (`3×` largest packed tensor) used to size the temporary repack staging buffer.
#[cfg(feature = "cuda")]
pub(super) const REPACK_STAGING_TENSOR_MULTIPLIER: usize = 3;

/// Extra safety margin (`16 MiB`) checked before initial weight + scratch arena boot allocations.
pub const MIN_BOOT_MARGIN_BYTES: usize = 16 * 1024 * 1024;

/// Alignment quantum (`256 B`) for GPU tensor sub-allocations inside `DeviceArena`.
pub const DEVICE_ARENA_TENSOR_ALIGNMENT_BYTES: usize = 256;

/// Minimum post-boot VRAM slack (`2 GiB`) for `StateBundlePool`, `KvPool`, and runtime headroom
/// required to keep untied `embed_tokens` inside GPU `weight_arena`. When keeping `embed_tokens`
/// in GPU VRAM would leave less than this slack, `embed_tokens` is offloaded to zero-copy
/// `MappedHostBuffer` if doing so allows the rank to fit within `free_vram` with at least
/// [`crate::cuda::MIN_HEADROOM_BYTES`] remaining.
pub const HOST_EMBED_GPU_RESIDENCY_MIN_SLACK_BYTES: usize = 2 * 1024 * 1024 * 1024;

/// Capacity (`16 MiB`) of the per-rank `batch_meta_arena` used for batch descriptors and page tables.
#[cfg(feature = "cuda")]
pub(super) const BATCH_META_ARENA_BYTES: usize = 16 * 1024 * 1024;

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

/// Maximum per-layer GPU hot expert count (`24`) in the compact `< 12 GiB` tiered-MoE regime
/// that selects compact `32`-slot H2D staging ([`TIERED_MOE_STAGE_CAPACITY_TP4`]).
pub const TIERED_MOE_COMPACT_MAX_HOT_EXPERTS: usize = 24;

/// Double-buffered H2D expert staging reserve (`64` expert slots = `2 × 32`) for compact
/// tiered-MoE ranks where [`TIERED_MOE_STAGE_CAPACITY_TP4`] is active.
pub const TIERED_MOE_COMPACT_STAGE_RESERVE_SLOTS: usize = 64;

/// Free VRAM threshold (`12 GiB`) below which tiered-MoE ranks select the standard
/// `512 MiB` scratch arena ([`crate::scheduler::DEFAULT_SCRATCH_ARENA_BYTES`] instead of
/// `1,056 MiB` [`crate::scheduler::TIERED_MOE_SCRATCH_ARENA_BYTES`]) and `640 MiB` runtime
/// headroom ([`crate::cuda::EXTREME_LONGCTX_HEADROOM_BYTES`]).
pub const TIERED_MOE_COMPACT_VRAM_THRESHOLD_BYTES: usize = 12 * 1024 * 1024 * 1024;

/// Minimum state slots (`8` = `1` active + `7` draft verify slots) reserved on `TP=1`
/// when planning tiered-MoE hot experts so `DFlash2` (`M=8`) verification fits.
pub const SINGLE_GPU_DFLASH2_MIN_STATE_SLOTS: usize = 8;

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
pub const TIERED_MOE_STAGE_CAPACITY_DEFAULT: usize = 96;

/// Double-buffered H2D staging capacity (`32` experts) for `RankTieredMoe` on `TP >= 4` and compact `< 12 GiB` budgets.
pub const TIERED_MOE_STAGE_CAPACITY_TP4: usize = 32;

/// Minimum per-row byte stride (`2,560 B`) for `PLE` FP8 staging buffers.
#[cfg(feature = "cuda")]
pub(super) const MIN_PLE_ROW_STAGING_BYTES: usize = 2560;

/// Free VRAM headroom (`512 MiB`) preserved after placing `100%` of `MTP` local experts in VRAM on `TP >= 2`.
pub const MTP_FULL_RESIDENCY_HEADROOM_BYTES: usize = 512 * 1024 * 1024;

/// Free VRAM headroom (`256 MiB`) preserved after allocating a partial `MTP` `HotExpertCache`
/// so `QSA` workspaces, `FullAttentionLayerWeights`, and `MTP` CUDA graphs fit comfortably.
pub const MTP_PARTIAL_CACHE_HEADROOM_BYTES: usize = 256 * 1024 * 1024;

/// Maximum `MTP` GPU hot expert count (`192`) on `TP=1` (`local_experts = 512`).
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

/// `TP=1` `MTP` hot expert count (`320 / 512`) when `>= 1,700 MiB` free VRAM is available.
pub const MTP_TP1_SERVING_HOT_EXPERTS: usize = 320;
pub const MTP_TP1_SERVING_MIN_FREE_VRAM_BYTES: usize = 1700 * 1024 * 1024;

/// Computes the `MTP` proposer's GPU hot expert cache capacity (`k_hot`) from live free VRAM.
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
    let slots_tp1_serving_need = MTP_TP1_SERVING_HOT_EXPERTS
        .saturating_mul(expert_stride_bytes)
        .saturating_add(MTP_PARTIAL_CACHE_HEADROOM_BYTES);
    let slots_tp1_max_need = MTP_TP1_STREAMED_PREFILL_MAX_HOT
        .saturating_mul(expert_stride_bytes)
        .saturating_add(MTP_PARTIAL_CACHE_HEADROOM_BYTES);
    let slots_medium_need = MTP_MEDIUM_HOT_EXPERTS
        .saturating_mul(expert_stride_bytes)
        .saturating_add(MTP_PARTIAL_CACHE_HEADROOM_BYTES);
    if tp_size > 1 && free_vram_bytes >= slots_all_need {
        local_experts
    } else if tp_size == 1
        && local_experts >= MTP_TP1_SERVING_HOT_EXPERTS
        && free_vram_bytes >= slots_tp1_serving_need.max(MTP_TP1_SERVING_MIN_FREE_VRAM_BYTES)
    {
        MTP_TP1_SERVING_HOT_EXPERTS
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

/// Select the per-step `TieredMoeWorkspace` H2D staging slot capacity (`stage_cap`) for a rank.
pub(super) fn select_tiered_moe_stage_capacity(
    cfg: &ModelConfig,
    tp_size: usize,
    hot_per_layer: usize,
) -> usize {
    let local_experts = (cfg.num_experts / tp_size.max(1)).max(1);
    if tp_size >= 4
        || (cfg.num_experts >= TIERED_MOE_MIN_MODEL_EXPERTS
            && hot_per_layer <= TIERED_MOE_COMPACT_MAX_HOT_EXPERTS)
    {
        local_experts.clamp(1, TIERED_MOE_STAGE_CAPACITY_TP4)
    } else {
        local_experts.clamp(1, TIERED_MOE_STAGE_CAPACITY_DEFAULT)
    }
}

/// Selects the per-rank transient `scratch_arena` capacity (`336 MiB`, `384 MiB`,
/// `512 MiB`, `960 MiB`, or `1,056 MiB`) based on model geometry, `tp_size`, estimated weight
/// footprint, requested KV pages, and any pre-planned LoRA adapter traits.
pub fn plan_scratch_arena_bytes_with_lora(
    cfg: &ModelConfig,
    tp_size: usize,
    est_weight_bytes: usize,
    requested_kv_pages: Option<usize>,
    lora: LoraCapacityReserve,
) -> usize {
    if tp_size == 1
        && cfg.num_hidden_layers >= EXTREME_LONGCTX_MIN_LAYERS
        && requested_kv_pages.unwrap_or(0) >= EXTREME_LONGCTX_MIN_PAGES
    {
        crate::scheduler::EXTREME_LONGCTX_SCRATCH_ARENA_BYTES
    } else if lora.has_midloop
        && requested_kv_pages.unwrap_or(0) <= crate::scheduler::TIERED_MOE_FULL_SCRATCH_MAX_KV_PAGES
    {
        MIDLOOP_LORA_SCRATCH_ARENA_BYTES
    } else if (cfg.uses_tiered_moe()
        || cfg.requires_atomic_prefix_prefill()
        || lora.requires_atomic_prefix
        || lora.device_bytes_per_rank > 0)
        && requested_kv_pages.unwrap_or(0) <= crate::scheduler::TIERED_MOE_FULL_SCRATCH_MAX_KV_PAGES
    {
        crate::scheduler::TIERED_MOE_SCRATCH_ARENA_BYTES
    } else if tp_size == 1 && est_weight_bytes >= LARGE_MODEL_WEIGHT_THRESHOLD_BYTES {
        crate::scheduler::LARGE_MODEL_TP1_SCRATCH_ARENA_BYTES
    } else {
        crate::scheduler::DEFAULT_SCRATCH_ARENA_BYTES
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
    plan_scratch_arena_bytes_with_lora(
        cfg,
        tp_size,
        est_weight_bytes,
        requested_kv_pages,
        LoraCapacityReserve::default(),
    )
}

/// Selects the target `KvPool` page count:
/// - When `requested_kv_pages` is `Some(req)`, returns `req` verbatim.
/// - In non-serving CLI/test mode (`!is_serving`) or when `dflash_reserve_bytes > 0` / `native_mtp`,
///   returns [`DEFAULT_TARGET_KV_PAGES`] (`256` pages = `4,096` tokens).
/// - In serving mode (`is_serving`) when `--kv-pages` is omitted (`None`), auto-sizes between
///   [`DEFAULT_TARGET_KV_PAGES`] (`256` pages = `4,096` tokens) and
///   [`DEFAULT_SERVING_TARGET_KV_PAGES`] (`2,048` pages = `32,768` tokens) in 64-page increments
///   while preserving `required_headroom + AUTO_SERVING_KV_EXTRA_HEADROOM_BYTES`.
#[allow(clippy::too_many_arguments)]
pub fn plan_target_kv_pages(
    requested_kv_pages: Option<usize>,
    is_serving: bool,
    dflash_reserve_bytes: usize,
    native_mtp: bool,
    plan_free_vram: usize,
    fixed_without_kv: usize,
    required_headroom: usize,
    bytes_per_kv_page_all_layers: usize,
) -> usize {
    if let Some(req) = requested_kv_pages {
        return req;
    }
    if !is_serving || dflash_reserve_bytes > 0 || native_mtp || bytes_per_kv_page_all_layers == 0 {
        return DEFAULT_TARGET_KV_PAGES;
    }
    let auto_reserved = fixed_without_kv
        .saturating_add(required_headroom)
        .saturating_add(AUTO_SERVING_KV_EXTRA_HEADROOM_BYTES);
    if plan_free_vram <= auto_reserved {
        return DEFAULT_TARGET_KV_PAGES;
    }
    let max_auto_pages = (plan_free_vram - auto_reserved) / bytes_per_kv_page_all_layers;
    let aligned_pages = (max_auto_pages / 64) * 64;
    aligned_pages.clamp(DEFAULT_TARGET_KV_PAGES, DEFAULT_SERVING_TARGET_KV_PAGES)
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
/// for tiered MoE models (`qwen4_exp`).
pub fn plan_tiered_moe_hot_per_layer(
    cfg: &ModelConfig,
    tp_size: usize,
    available_vram_after_weights_and_scratch: usize,
    base_headroom_plus_kv_bytes: usize,
    min_state_pool_bytes: usize,
) -> (usize, usize) {
    plan_tiered_moe_hot_per_layer_with_dflash(
        cfg,
        tp_size,
        available_vram_after_weights_and_scratch,
        base_headroom_plus_kv_bytes,
        min_state_pool_bytes,
        0,
    )
}

/// Computes the per-layer hot expert cache capacity (`K_hot`) and total VRAM reserve
/// for tiered MoE models (`qwen4_exp`), accounting for an explicit `DFlash2` resident
/// drafter + sliding-window KV reserve when `dflash_reserve_bytes > 0`.
pub fn plan_tiered_moe_hot_per_layer_with_dflash(
    cfg: &ModelConfig,
    tp_size: usize,
    available_vram_after_weights_and_scratch: usize,
    base_headroom_plus_kv_bytes: usize,
    min_state_pool_bytes: usize,
    dflash_reserve_bytes: usize,
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
            + TIERED_MOE_SYNTHETIC_EXTRA_RESERVE_BYTES
            + dflash_reserve_bytes;
        return (hot, reserve);
    }
    let min_hot = if tp_size >= 2 {
        TIERED_MOE_TP2_MIN_HOT_EXPERTS.min(default_hot)
    } else {
        TIERED_MOE_TP1_MIN_HOT_EXPERTS.min(default_hot)
    };
    let extra_reserve = if tp_size >= 4 {
        (TIERED_MOE_TP4_STAGE_RESERVE_SLOTS * exp_bytes).saturating_add(dflash_reserve_bytes)
    } else {
        (TIERED_MOE_STAGE_RESERVE_SLOTS * exp_bytes)
            .saturating_add(TIERED_MOE_MTP_AND_GRAPH_RESERVE_BYTES.max(dflash_reserve_bytes))
    };
    let fixed_overhead = base_headroom_plus_kv_bytes
        .saturating_add(min_state_pool_bytes)
        .saturating_add(extra_reserve);
    let per_hot_step = cfg.num_hidden_layers * exp_bytes;
    let actual_per_hot_step: usize = (0..cfg.num_hidden_layers)
        .filter(|l| !cfg.mlp_only_layers.contains(l))
        .map(|l| {
            cfg.layer_moe_expert_pack_layout(l)
                .map_or(exp_bytes, |lay| lay.stride)
        })
        .sum();
    let min_standard_total = fixed_overhead.saturating_add(min_hot.saturating_mul(per_hot_step));
    let min_base_and_state = base_headroom_plus_kv_bytes.saturating_add(min_state_pool_bytes);
    if available_vram_after_weights_and_scratch < min_standard_total
        && available_vram_after_weights_and_scratch > min_base_and_state
        && per_hot_step > 0
    {
        let compact_extra_reserve =
            TIERED_MOE_COMPACT_STAGE_RESERVE_SLOTS.saturating_mul(exp_bytes);
        let hot = TIERED_MOE_TP1_MIN_HOT_EXPERTS.min(default_hot);
        let reserve = hot.saturating_mul(actual_per_hot_step) + compact_extra_reserve;
        return (hot, reserve);
    }
    let hot = if available_vram_after_weights_and_scratch > fixed_overhead && per_hot_step > 0 {
        let max_affordable =
            (available_vram_after_weights_and_scratch - fixed_overhead) / per_hot_step;
        let aligned =
            (max_affordable / TIERED_MOE_HOT_SLOT_ALIGNMENT) * TIERED_MOE_HOT_SLOT_ALIGNMENT;
        aligned.clamp(min_hot, default_hot)
    } else {
        min_hot
    };
    let reserve = hot.saturating_mul(actual_per_hot_step) + extra_reserve;
    (hot, reserve)
}

/// Determines whether untied `embed_tokens` should be placed in zero-copy `MappedHostBuffer`
/// instead of GPU `weight_arena`, returning `(adjusted_est_weight_bytes, use_host_embed)`.
pub fn plan_host_embed_offload(
    cfg: &ModelConfig,
    est_weight_bytes: usize,
    scratch_bytes: usize,
    staging_bytes: usize,
    free_vram_bytes: usize,
) -> (usize, bool) {
    plan_host_embed_with_budget(
        cfg,
        est_weight_bytes,
        scratch_bytes,
        staging_bytes,
        free_vram_bytes,
        0,
        None,
    )
}

pub(super) fn has_vision_weights(cfg: &ModelConfig, archive: &SafetensorsArchive) -> bool {
    cfg.vision_config.is_some()
        && (archive
            .resolve_name("model.visual.patch_embed.proj.weight")
            .is_some()
            || archive
                .resolve_name("visual.patch_embed.proj.weight")
                .is_some())
}

/// Boot staging is transient: it must never be charged alongside target pools.
/// `steady_bytes` includes persistent metadata/workspaces, minimum pools and headroom.
#[allow(clippy::too_many_arguments)]
pub(super) fn plan_host_embed_with_budget(
    cfg: &ModelConfig,
    est_weight_bytes: usize,
    scratch_bytes: usize,
    staging_bytes: usize,
    free_vram_bytes: usize,
    boot_persistent_bytes: usize,
    steady_bytes: Option<usize>,
) -> (usize, bool) {
    plan_host_embed_with_serving_budget(
        cfg,
        est_weight_bytes,
        scratch_bytes,
        staging_bytes,
        free_vram_bytes,
        boot_persistent_bytes,
        steady_bytes,
        None,
    )
}

#[allow(clippy::too_many_arguments)]
pub(super) fn plan_host_embed_with_serving_budget(
    cfg: &ModelConfig,
    est_weight_bytes: usize,
    scratch_bytes: usize,
    staging_bytes: usize,
    free_vram_bytes: usize,
    boot_persistent_bytes: usize,
    steady_bytes: Option<usize>,
    desired_gpu_steady_bytes: Option<usize>,
) -> (usize, bool) {
    let fits_budget = |weights: usize, budget: Option<usize>| {
        budget.is_none_or(|bytes| {
            (weights as u128 + scratch_bytes as u128 + bytes as u128) <= free_vram_bytes as u128
        })
    };
    if cfg.tie_word_embeddings {
        return (est_weight_bytes, false);
    }
    if cfg.uses_tiered_moe() {
        return (est_weight_bytes, true);
    }
    let boot_need = |weights: usize| {
        weights as u128
            + scratch_bytes as u128
            + staging_bytes as u128
            + boot_persistent_bytes as u128
            + MIN_BOOT_MARGIN_BYTES as u128
    };
    if boot_need(est_weight_bytes) + HOST_EMBED_GPU_RESIDENCY_MIN_SLACK_BYTES as u128
        <= free_vram_bytes as u128
        && fits_budget(est_weight_bytes, steady_bytes)
        && fits_budget(est_weight_bytes, desired_gpu_steady_bytes)
    {
        return (est_weight_bytes, false);
    }
    let embed_fp16_aligned = (cfg
        .vocab_size
        .saturating_mul(cfg.hidden_size)
        .saturating_mul(core::mem::size_of::<u16>()))
    .div_ceil(DEVICE_ARENA_TENSOR_ALIGNMENT_BYTES)
    .saturating_mul(DEVICE_ARENA_TENSOR_ALIGNMENT_BYTES);
    let est_without_embed = est_weight_bytes
        .saturating_sub(embed_fp16_aligned)
        .div_ceil(WEIGHT_ARENA_ALIGNMENT_BYTES)
        .saturating_mul(WEIGHT_ARENA_ALIGNMENT_BYTES)
        .max(MIN_WEIGHT_ARENA_BYTES);
    let boot_reserve = if steady_bytes.is_some() {
        0
    } else {
        crate::cuda::MIN_HEADROOM_BYTES
    };
    if boot_need(est_without_embed) + boot_reserve as u128 <= free_vram_bytes as u128
        && fits_budget(est_without_embed, steady_bytes)
    {
        (est_without_embed, true)
    } else {
        (est_weight_bytes, false)
    }
}

#[cfg(any(test, feature = "cuda"))]
pub(super) fn pool_weight_budget(use_live_budget: bool, used: usize, capacity: usize) -> usize {
    if use_live_budget { used } else { capacity }
}

/// Shared target pool geometry; QSA indexer KV is replicated, ordinary KV follows
/// `local_kv_heads` (including replication when TP exceeds the KV head count).
#[cfg(test)]
pub(super) fn target_pool_geometry(
    cfg: &ModelConfig,
    tp: usize,
    policy: KvFormatPolicy,
) -> (usize, usize) {
    target_pool_geometry_with_fp8(cfg, tp, policy, false)
}

#[cfg(any(test, feature = "cuda"))]
pub(super) fn target_pool_geometry_with_fp8(
    cfg: &ModelConfig,
    tp: usize,
    policy: KvFormatPolicy,
    supports_fp8_kv: bool,
) -> (usize, usize) {
    let full = cfg
        .layer_types
        .iter()
        .filter(|kind| matches!(kind, LayerKind::FullAttention))
        .count();
    let gdn = cfg.layer_types.len() - full;
    let block = KV_PAGE_BLOCK_TOKENS;
    let mut page = 0usize;
    for ord in 0..full {
        let (kf, vf, _, _) = policy.resolve_layer_with_fp8(ord, full, supports_fp8_kv);
        let rows = crate::cache::kv_format_row_bytes(kf, cfg.head_dim)
            .saturating_add(crate::cache::kv_format_row_bytes(vf, cfg.head_dim));
        page = page.saturating_add(
            cfg.local_kv_heads(tp)
                .saturating_mul(block)
                .saturating_mul(rows),
        );
        if cfg.uses_qsa() {
            page = page.saturating_add(
                cfg.indexer_kv_heads
                    .saturating_mul(block + block / cfg.indexer_compress_ratio.max(1))
                    .saturating_mul(cfg.indexer_head_dim)
                    .saturating_mul(2),
            );
        }
    }
    let k = cfg
        .linear_num_key_heads
        .saturating_mul(cfg.linear_key_head_dim)
        / tp;
    let v = cfg
        .linear_num_value_heads
        .saturating_mul(cfg.linear_value_head_dim)
        / tp;
    let conv = k
        .saturating_mul(2)
        .saturating_add(v)
        .saturating_mul(cfg.linear_conv_kernel_dim.saturating_sub(1))
        .saturating_mul(2);
    let recurrent = (cfg.linear_num_value_heads / tp)
        .saturating_mul(cfg.linear_value_head_dim)
        .saturating_mul(cfg.linear_key_head_dim)
        .saturating_mul(4);
    let ple = if cfg.uses_ple() {
        cfg.ple_layer_ids
            .len()
            .saturating_mul(cfg.hc_count)
            .saturating_mul(cfg.hidden_size)
            .saturating_mul(18)
    } else {
        0
    };
    (
        page,
        gdn.saturating_mul(conv.saturating_add(recurrent))
            .saturating_add(ple),
    )
}

#[cfg(any(test, feature = "cuda"))]
pub(super) fn target_base_headroom(
    cfg: &ModelConfig,
    tp: usize,
    pages: usize,
    free: usize,
) -> usize {
    if (tp == 1
        && cfg.num_hidden_layers >= EXTREME_LONGCTX_MIN_LAYERS
        && pages >= EXTREME_LONGCTX_MIN_PAGES)
        || (cfg.uses_tiered_moe() && free < TIERED_MOE_COMPACT_VRAM_THRESHOLD_BYTES)
    {
        crate::cuda::EXTREME_LONGCTX_HEADROOM_BYTES
    } else {
        crate::cuda::MIN_HEADROOM_BYTES
    }
}

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

#[cfg(any(test, feature = "cuda"))]
pub(super) fn resident_moe_mtp_serving_reserve(
    cfg: &ModelConfig,
    is_serving: bool,
    native_mtp: bool,
    mtp_experts_w4: bool,
) -> usize {
    if is_serving && native_mtp && cfg.num_experts > 0 && !cfg.uses_tiered_moe() && !cfg.uses_qsa()
    {
        if mtp_experts_w4 {
            768 * 1024 * 1024
        } else {
            1600 * 1024 * 1024
        }
    } else {
        0
    }
}
