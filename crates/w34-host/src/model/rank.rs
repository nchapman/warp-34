//! Per-device model residency, capacity planning, and loading.

use super::*;

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
    pub final_norm_ptr: u64,
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
}

#[cfg(feature = "cuda")]
#[derive(Clone, Copy, Default)]
pub(super) struct PoolCapacity {
    pub(super) kv_pages: Option<usize>,
    pub(super) state_slots: Option<usize>,
    pub(super) rank0_extra_headroom_bytes: usize,
}

#[cfg(feature = "cuda")]
pub(super) fn serving_state_slots(max_num_seqs: usize) -> Result<usize> {
    max_num_seqs
        .checked_mul(4)
        .filter(|&slots| slots != 0)
        .ok_or_else(|| {
            loader_err("serving max_num_seqs must be nonzero and fit four state slots per sequence")
        })
}

// Preserve the minimum three-slot execution budget and the existing VRAM clamp.
#[cfg(feature = "cuda")]
pub(super) fn clamp_state_slots(
    requested: usize,
    free: usize,
    fixed: usize,
    per_slot: usize,
) -> usize {
    if per_slot == 0 {
        requested
    } else if free > fixed {
        ((free - fixed) / per_slot).clamp(3, requested)
    } else {
        3usize.min(requested)
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
        )
    }

    fn validate_tp_geometry(cfg: &ModelConfig, tp_size: usize) -> Result<()> {
        if !matches!(tp_size, 1 | 2) {
            return Err(loader_err_msg(format!(
                "unsupported tp_size={tp_size} (warp-34 supports TP=1 or TP=2)"
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
                || !cfg.num_key_value_heads.is_multiple_of(tp_size)
            {
                return Err(loader_err_msg(format!(
                    "attention heads (q={}, kv={}) are not divisible by tp_size={tp_size}; retry with --tp 1",
                    cfg.num_attention_heads, cfg.num_key_value_heads
                )));
            }
            if !(cfg.num_attention_heads / tp_size)
                .is_multiple_of(cfg.num_key_value_heads / tp_size)
            {
                return Err(loader_err_msg(format!(
                    "local Q heads ({}) must be a multiple of local KV heads ({})",
                    cfg.num_attention_heads / tp_size,
                    cfg.num_key_value_heads / tp_size
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
            if cfg.moe_intermediate_size == 0
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

    fn estimate_weight_arena_bytes(
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
        let mut replicated_bytes = embed_fp16_aligned;
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
                || name.ends_with(".g_idx")
                || name.ends_with(".weight_g_idx")
                || name.ends_with(".weight_shape")
                || (name.ends_with(".qzeros") && !cfg.quant.expects_zero_points())
            {
                continue;
            }
            let is_replicated = name.contains("norm.")
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
            && cfg.num_hidden_layers >= 64
            && requested_kv_pages.unwrap_or(0) >= 14_336;
        let margin = if extreme_longctx {
            32 * 1024 * 1024
        } else {
            96 * 1024 * 1024
        };
        let two_mib = 2 * 1024 * 1024;
        let rounded = per_rank_raw
            .saturating_add(margin)
            .div_ceil(two_mib)
            .saturating_mul(two_mib);
        rounded.max(128 * 1024 * 1024)
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
                    };
                } else if p_info.shape[1].saturating_mul(8) == cfg.hidden_size {
                    embed_quant = crate::loader::QuantScheme::CompressedTensors {
                        num_bits: 4,
                        group_size: cfg.quant.raw_group_size(),
                        symmetric: !has_zp,
                        ignore: Vec::new(),
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

    pub(super) fn load_rank_with_capacity(
        archive: &SafetensorsArchive,
        cfg: &ModelConfig,
        rank: usize,
        tp_size: usize,
        device_id: i32,
        kv_policy: KvFormatPolicy,
        capacity: PoolCapacity,
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
            .filter(|(_, info)| matches!(info.dtype.as_str(), "I32" | "U32"))
            .map(|(_, info)| info.len_bytes as usize)
            .max()
            .unwrap_or(0);
        let staging_bytes = (128 * 1024 * 1024).max(max_packed_tensor_bytes.saturating_mul(3));
        let scratch_bytes = if tp_size == 1
            && cfg.num_hidden_layers >= 64
            && requested_kv_pages.unwrap_or(0) >= 14_336
        {
            336 * 1024 * 1024
        } else if tp_size == 1 && est_weight_bytes >= 16 * 1024 * 1024 * 1024 {
            384 * 1024 * 1024
        } else {
            512 * 1024 * 1024
        };
        let min_boot_bytes = est_weight_bytes
            .saturating_add(scratch_bytes)
            .saturating_add(staging_bytes)
            .saturating_add(16 * 1024 * 1024);
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
        let batch_meta_arena = DeviceArena::new(device_id, 16 * 1024 * 1024)?;
        let marlin_ws = DeviceBuffer::alloc(device_id, marlin_ws_bytes)?;
        let repack_staging = DeviceBuffer::alloc(device_id, staging_bytes)?;

        // 1. embed_tokens (replicated in FP16; dequantized on host when packed)
        let embed_f16 = Self::load_embed_tokens_f16(archive, cfg)?;
        let embed_tokens_ptr = weight_arena.upload_slice(&embed_f16, &stream)?;

        // 2. final norm
        let norm_key = archive
            .resolve_name("model.norm.weight")
            .ok_or_else(|| loader_err("missing model.norm.weight"))?;
        let norm_f16 = archive
            .read_f16_bits(&norm_key)
            .map_err(|_| loader_err("read model.norm.weight"))?;
        let final_norm_ptr = weight_arena.upload_slice(&norm_f16, &stream)?;

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
        } else {
            let raw_lm = RawLinearHost::load(archive, "lm_head", &cfg.quant)?;
            let sharded = raw_lm.slice_rows(vocab_offset, vocab_offset + local_vocab_size);
            sharded.upload(&mut weight_arena, &stream, &repack_staging)?
        };

        // 4. Decoder layers
        let mut layers = Vec::with_capacity(cfg.num_hidden_layers);
        let mut full_attn_layer_ids = Vec::new();
        let mut gdn_layer_ids = Vec::new();

        let local_q_heads = cfg.num_attention_heads / tp_size;
        let local_kv_heads = cfg.num_key_value_heads / tp_size;
        let q_gate_total = cfg.num_attention_heads * cfg.head_dim * 2;
        let q_gate_local = local_q_heads * cfg.head_dim * 2;
        let kv_local = local_kv_heads * cfg.head_dim;

        let gdn_k_total = cfg.linear_num_key_heads * cfg.linear_key_head_dim;
        let gdn_v_total = cfg.linear_num_value_heads * cfg.linear_value_head_dim;
        let gdn_k_local = gdn_k_total / tp_size;
        let gdn_v_local = gdn_v_total / tp_size;
        let local_v_heads = cfg.linear_num_value_heads / tp_size;

        let local_interm = cfg.intermediate_size / tp_size;

        for layer_id in 0..cfg.num_hidden_layers {
            let pfx = format!("model.layers.{layer_id}");
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
            let post_attention_layernorm_ptr = weight_arena.upload_slice(&post_ln, &stream)?;

            let block = match cfg.layer_types[layer_id] {
                LayerKind::FullAttention => {
                    let kv_ord = full_attn_layer_ids.len();
                    full_attn_layer_ids.push(layer_id);

                    let raw_q = RawLinearHost::load(
                        archive,
                        &format!("{pfx}.self_attn.q_proj"),
                        &cfg.quant,
                    )?;
                    debug_assert_eq!(raw_q.slice_rows(0, q_gate_total).size_check(), q_gate_total);
                    let q_shard = raw_q.slice_rows(rank * q_gate_local, (rank + 1) * q_gate_local);

                    let raw_k = RawLinearHost::load(
                        archive,
                        &format!("{pfx}.self_attn.k_proj"),
                        &cfg.quant,
                    )?;
                    let k_shard = raw_k.slice_rows(rank * kv_local, (rank + 1) * kv_local);

                    let raw_v = RawLinearHost::load(
                        archive,
                        &format!("{pfx}.self_attn.v_proj"),
                        &cfg.quant,
                    )?;
                    let v_shard = raw_v.slice_rows(rank * kv_local, (rank + 1) * kv_local);

                    let qkv_shard = RawLinearHost::concat_rows(&[q_shard, k_shard, v_shard])?;
                    let qkv_proj = qkv_shard.upload(&mut weight_arena, &stream, &repack_staging)?;

                    let raw_o = RawLinearHost::load(
                        archive,
                        &format!("{pfx}.self_attn.o_proj"),
                        &cfg.quant,
                    )?;
                    let o_cols = local_q_heads * cfg.head_dim;
                    let o_shard = raw_o.slice_cols(rank * o_cols, (rank + 1) * o_cols);
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

                    LayerBlockWeights::FullAttention(FullAttentionLayerWeights {
                        layer_id,
                        kv_layer_ordinal: kv_ord,
                        qkv_proj,
                        o_proj,
                        q_norm_ptr,
                        k_norm_ptr,
                    })
                }
                LayerKind::LinearAttention => {
                    let gdn_ord = gdn_layer_ids.len();
                    gdn_layer_ids.push(layer_id);

                    let raw_qkv = RawLinearHost::load(
                        archive,
                        &format!("{pfx}.linear_attn.in_proj_qkv"),
                        &cfg.quant,
                    )?;
                    let q_part = raw_qkv.slice_rows(rank * gdn_k_local, (rank + 1) * gdn_k_local);
                    let k_part = raw_qkv.slice_rows(
                        gdn_k_total + rank * gdn_k_local,
                        gdn_k_total + (rank + 1) * gdn_k_local,
                    );
                    let v_part = raw_qkv.slice_rows(
                        2 * gdn_k_total + rank * gdn_v_local,
                        2 * gdn_k_total + (rank + 1) * gdn_v_local,
                    );

                    let raw_z = RawLinearHost::load(
                        archive,
                        &format!("{pfx}.linear_attn.in_proj_z"),
                        &cfg.quant,
                    )?;
                    let z_shard = raw_z.slice_rows(rank * gdn_v_local, (rank + 1) * gdn_v_local);

                    let qkvz_shard =
                        RawLinearHost::concat_rows(&[q_part, k_part, v_part, z_shard])?;
                    let in_proj_qkvz =
                        qkvz_shard.upload(&mut weight_arena, &stream, &repack_staging)?;

                    let raw_b = RawLinearHost::load(
                        archive,
                        &format!("{pfx}.linear_attn.in_proj_b"),
                        &cfg.quant,
                    )?;
                    let b_shard =
                        raw_b.slice_rows(rank * local_v_heads, (rank + 1) * local_v_heads);

                    let raw_a = RawLinearHost::load(
                        archive,
                        &format!("{pfx}.linear_attn.in_proj_a"),
                        &cfg.quant,
                    )?;
                    let a_shard =
                        raw_a.slice_rows(rank * local_v_heads, (rank + 1) * local_v_heads);

                    let ba_shard = RawLinearHost::concat_rows(&[b_shard, a_shard])?;
                    let in_proj_ba =
                        ba_shard.upload(&mut weight_arena, &stream, &repack_staging)?;

                    let raw_out = RawLinearHost::load(
                        archive,
                        &format!("{pfx}.linear_attn.out_proj"),
                        &cfg.quant,
                    )?;
                    let out_shard =
                        raw_out.slice_cols(rank * gdn_v_local, (rank + 1) * gdn_v_local);
                    let out_proj = out_shard.upload(&mut weight_arena, &stream, &repack_staging)?;

                    let conv_key = archive
                        .resolve_name(&format!("{pfx}.linear_attn.conv1d.weight"))
                        .ok_or_else(|| loader_err("missing conv1d.weight"))?;
                    let conv_all = archive
                        .read_f16_bits(&conv_key)
                        .map_err(|_| loader_err("read conv1d.weight"))?;
                    let k_sz = cfg.linear_conv_kernel_dim;
                    let mut conv_local = Vec::with_capacity((2 * gdn_k_local + gdn_v_local) * k_sz);
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
                    let a_log_local = &a_log_all[rank * local_v_heads..(rank + 1) * local_v_heads];
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
                    // Note: linear_attn.norm.weight has shape [head_v_dim] (128), broadcast across all v_heads
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

            let ffn = if is_moe_layer {
                FfnBlockWeights::Moe(Box::new(load_moe_layer_weights(
                    archive,
                    cfg,
                    &pfx,
                    rank,
                    tp_size,
                    &mut weight_arena,
                    &stream,
                    &repack_staging,
                )?))
            } else {
                let raw_gate =
                    RawLinearHost::load(archive, &format!("{pfx}.mlp.gate_proj"), &cfg.quant)?;
                let gate_shard =
                    raw_gate.slice_rows(rank * local_interm, (rank + 1) * local_interm);

                let raw_up =
                    RawLinearHost::load(archive, &format!("{pfx}.mlp.up_proj"), &cfg.quant)?;
                let up_shard = raw_up.slice_rows(rank * local_interm, (rank + 1) * local_interm);

                let gate_up_shard = RawLinearHost::concat_rows(&[gate_shard, up_shard])?;
                let gate_up_proj =
                    gate_up_shard.upload(&mut weight_arena, &stream, &repack_staging)?;

                let raw_down =
                    RawLinearHost::load(archive, &format!("{pfx}.mlp.down_proj"), &cfg.quant)?;
                let down_shard =
                    raw_down.slice_cols(rank * local_interm, (rank + 1) * local_interm);
                let down_proj = down_shard.upload(&mut weight_arena, &stream, &repack_staging)?;

                FfnBlockWeights::Dense {
                    gate_up_proj,
                    down_proj,
                }
            };

            layers.push(DecoderLayerWeights {
                layer_id,
                input_layernorm_ptr,
                post_attention_layernorm_ptr,
                block,
                ffn,
            });
        }

        drop(repack_staging);
        stream.synchronize()?;

        // 5. StateBundlePool & KvPool via CapacityPlanner
        let block_size = 16usize;
        let mut bytes_per_kv_page_all_layers = 0usize;
        for ord in 0..full_attn_layer_ids.len() {
            let (kf, vf, _, _) = kv_policy.resolve_layer(ord, full_attn_layer_ids.len());
            let kb = crate::cache::kv_format_row_bytes(kf, cfg.head_dim);
            let vb = crate::cache::kv_format_row_bytes(vf, cfg.head_dim);
            bytes_per_kv_page_all_layers += local_kv_heads * block_size * (kb + vb);
        }

        let conv_dim = 2 * gdn_k_local + gdn_v_local;
        let bytes_per_state_slot = gdn_layer_ids.len()
            * (conv_dim * (cfg.linear_conv_kernel_dim - 1) * 2
                + local_v_heads * cfg.linear_value_head_dim * cfg.linear_key_head_dim * 4);
        let default_state_slots = if est_weight_bytes >= 17 * 1024 * 1024 * 1024 {
            10usize
        } else {
            24usize
        };
        let requested_state_slots = capacity.state_slots.unwrap_or(default_state_slots);
        let default_pages = requested_kv_pages.or(Some(256));
        let req_kv_bytes = default_pages.unwrap_or(256) * bytes_per_kv_page_all_layers;
        let base_headroom = if tp_size == 1
            && cfg.num_hidden_layers >= 64
            && default_pages.unwrap_or(0) >= 14_336
        {
            crate::cuda::EXTREME_LONGCTX_HEADROOM_BYTES
        } else {
            crate::cuda::MIN_HEADROOM_BYTES
        };
        let required_headroom = base_headroom.saturating_add(if rank == 0 {
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
            let available = device.mem_info()?.0;
            let archive_has_vision = cfg.vision_config.is_some()
                && (archive
                    .resolve_name("model.visual.patch_embed.proj.weight")
                    .is_some()
                    || archive
                        .resolve_name("visual.patch_embed.proj.weight")
                        .is_some());
            let vision_reserve =
                if rank == 0 && archive_has_vision && default_pages.unwrap_or(0) < 14_336 {
                    1536 * 1024 * 1024
                } else {
                    0
                };
            let reserved = required_headroom
                .saturating_add(vision_reserve)
                .saturating_add(req_kv_bytes);
            if available < reserved.saturating_add(bytes_per_state_slot.saturating_mul(3)) {
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
        let state_pool = StateBundlePool::new(
            device_id,
            &gdn_layer_ids,
            total_state_slots,
            conv_dim,
            cfg.linear_conv_kernel_dim,
            local_v_heads,
            cfg.linear_key_head_dim,
            cfg.linear_value_head_dim,
        )?;
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

        let kv_pool = KvPool::new_target_full_attn(
            device_id,
            &full_attn_layer_ids,
            local_kv_heads,
            cfg.head_dim,
            block_size,
            capacity_plan.max_kv_pages_per_layer,
            kv_policy,
        )?;

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
            final_norm_ptr,
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
        })
    }

    /// Configures replicated `TapPlan` auxiliary hidden-state capture for TP
    /// DFlash2 and/or post-final-norm hidden export for native MTP. Each target
    /// rank already owns the all-reduced activation; local capture avoids a
    /// second PCIe transfer before the replicated drafter context projection.
    /// Call on every TP rank with identical arguments before any forward or
    /// graph capture, enforcing the same row capacity. Engine callers must also
    /// propagate configuration failures before dispatching any rank forward.
    pub fn configure_speculative_taps(
        &mut self,
        tap_layer_ids: &[usize],
        export_final_hidden: bool,
        max_tap_rows: usize,
    ) -> Result<()> {
        // Captured graphs own these destination addresses and the tap geometry.
        // Require configuration before capture rather than freeing live targets.
        if self.decode_graph.is_some()
            || self.spec_verify_graph.is_some()
            || !self.batched_decode_graphs.is_empty()
        {
            return Err(loader_err(
                "configure speculative taps before decode graph capture",
            ));
        }
        if (export_final_hidden || !tap_layer_ids.is_empty()) && max_tap_rows == 0 {
            return Err(loader_err("speculative tap capacity must be positive"));
        }
        if tap_layer_ids
            .iter()
            .enumerate()
            .any(|(i, &id)| id >= self.layers.len() || tap_layer_ids[..i].contains(&id))
        {
            return Err(loader_err("tap layer ids must be unique and in range"));
        }
        self.device.make_current()?;
        self.stream.synchronize()?;
        let row_elems = max_tap_rows
            .checked_mul(self.cfg.hidden_size)
            .ok_or_else(|| loader_err("tap capacity overflow"))?;
        let tap_aux_bf16 = if !tap_layer_ids.is_empty() {
            let bytes = row_elems
                .checked_mul(tap_layer_ids.len())
                .and_then(|n| n.checked_mul(2))
                .ok_or_else(|| loader_err("tap capacity overflow"))?;
            Some(DeviceBuffer::alloc(self.device.device_id, bytes)?)
        } else {
            None
        };
        let last_final_hidden_f16 = if export_final_hidden {
            let bytes = row_elems
                .checked_mul(2)
                .ok_or_else(|| loader_err("final hidden capacity overflow"))?;
            Some(DeviceBuffer::alloc(self.device.device_id, bytes)?)
        } else {
            None
        };
        self.tap_layer_ids = tap_layer_ids.to_vec();
        self.max_tap_rows = max_tap_rows;
        self.tap_aux_bf16 = tap_aux_bf16;
        self.last_final_hidden_f16 = last_final_hidden_f16;
        Ok(())
    }

    pub(super) fn validate_tap_rows(&self, num_tokens: usize) -> Result<()> {
        if (!self.tap_layer_ids.is_empty() || self.last_final_hidden_f16.is_some())
            && num_tokens > self.max_tap_rows
        {
            return Err(loader_err("forward rows exceed speculative tap capacity"));
        }
        Ok(())
    }

    /// Maximum aligned ordinary prefill budget, including worst-case selected outputs.
    pub fn ordinary_prefill_token_capacity(&self, max_num_seqs: usize) -> Result<usize> {
        model_scratch::token_capacity(
            self.scratch_arena.capacity(),
            self.cfg.hidden_size,
            self.local_vocab_size,
            max_num_seqs,
            |tokens| {
                layer_scratch_layout(&self.cfg, self.tp_size, tokens)
                    .ok()
                    .map(|l| l.0)
            },
        )
        .ok_or_else(|| loader_err("scratch cannot fit one 64-token prefill chunk"))
    }
}

// SAFETY: moving a rank transfers its CUDA owners, not host references into device
// storage. Forward/load entrypoints select the device on the executing thread;
// scoped TP workers receive exclusive rank borrows and join before returning.
// Moving is not synchronization: callers must still order queued work and must
// not use copied raw addresses concurrently with the new owner.
#[cfg(feature = "cuda")]
unsafe impl Send for Qwen3_5ModelRank {}
