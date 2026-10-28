//! Qwen 3.5 / 3.8 Hybrid Model Backbone (`TP=1` & `TP=2`), `TapPlan`, GPU `SamplerV1`,
//! and `CUDAGraphRunner`.
#![allow(clippy::undocumented_unsafe_blocks, clippy::too_many_arguments)]

use crate::cache::{KvFormatPolicy, KvPool, StateBundlePool, StateTx};
use crate::cuda::{
    CapacityPlan, CapacityPlanner, CublasHandle, CudaDevice, CudaError, CudaGraph, CudaStream,
    DeviceArena, DeviceBuffer, Result, Tp2Collectives, check_rc, f16_to_f32,
    prepare_execution_plan,
};
use crate::loader::{
    LayerKind, ModelConfig, QuantScheme, SafetensorsArchive, dequant_u4b8_n_major_to_f16,
    marlin_permute_scales_from_n_major, resolve_hf_model_dir,
    transpose_gptq_qweight_k8_n_to_n_major,
};
use core::ffi::c_void;
use std::collections::HashMap;
use std::path::Path;
use std::sync::Arc;
use tokenizers::Tokenizer;
use w34_kernels::ffi::{self, W34_ATTN_MASK_CAUSAL, W34AttentionBatch, W34Exec, W34GdnBatch};

#[inline]
fn loader_err(context: &'static str) -> CudaError {
    CudaError { code: -10, context }
}

/// Static tap plan specifying layer indices whose post-layer hidden states are captured.
#[derive(Clone, Debug, Default)]
pub struct TapPlan {
    pub layer_ids: Vec<usize>,
}

/// Named intermediate activations captured during a diagnostic `dump_forward` pass.
#[derive(Clone, Debug, Default)]
pub struct ForwardDump {
    pub tensors_f32: HashMap<String, (Vec<usize>, Vec<f32>)>,
    pub greedy_token_id: i32,
    pub top5_ids: [i32; 5],
    pub top5_logprobs: [f32; 5],
}

#[cfg(feature = "cuda")]
pub enum LinearProj {
    MarlinW4A16 {
        b_q_weight_ptr: u64,
        b_scales_ptr: u64,
        size_n: usize,
        size_k: usize,
        group_blocks: i32,
    },
    DenseF16 {
        weight_ptr: u64,
        size_n: usize,
        size_k: usize,
    },
}

#[cfg(feature = "cuda")]
impl LinearProj {
    pub fn size_n(&self) -> usize {
        match self {
            Self::MarlinW4A16 { size_n, .. } | Self::DenseF16 { size_n, .. } => *size_n,
        }
    }

    pub fn size_k(&self) -> usize {
        match self {
            Self::MarlinW4A16 { size_k, .. } | Self::DenseF16 { size_k, .. } => *size_k,
        }
    }

    pub fn byte_footprint(&self) -> usize {
        match self {
            Self::MarlinW4A16 {
                size_n,
                size_k,
                group_blocks,
                ..
            } => {
                let g = (*group_blocks as usize) * 16;
                (size_n * size_k / 2) + (size_n * (size_k / g) * 2)
            }
            Self::DenseF16 { size_n, size_k, .. } => size_n * size_k * 2,
        }
    }

    #[allow(clippy::too_many_arguments)]
    pub fn forward(
        &self,
        ex: &W34Exec,
        cublas: &CublasHandle,
        marlin_ws_ptr: u64,
        marlin_ws_ints: usize,
        x_ptr: u64,
        out_ptr: u64,
        m: usize,
    ) -> Result<()> {
        match *self {
            Self::MarlinW4A16 {
                b_q_weight_ptr,
                b_scales_ptr,
                size_n,
                size_k,
                group_blocks,
            } => check_rc(
                unsafe {
                    ffi::w34_marlin_gemm_f16u4(
                        ex,
                        x_ptr as *const c_void,
                        b_q_weight_ptr as *const c_void,
                        b_scales_ptr as *const c_void,
                        out_ptr as *mut c_void,
                        marlin_ws_ptr as *mut i32,
                        m as i64,
                        size_n as i64,
                        size_k as i64,
                        marlin_ws_ints as i64,
                        group_blocks,
                    )
                },
                "w34_marlin_gemm_f16u4",
            ),
            Self::DenseF16 {
                weight_ptr,
                size_n,
                size_k,
            } => cublas.gemm_f16(
                x_ptr,
                weight_ptr,
                out_ptr,
                m as i32,
                size_n as i32,
                size_k as i32,
            ),
        }
    }
}

/// Raw host representation of a linear weight before sharding/uploading.
enum RawLinearHost {
    U4B8NMajor {
        packed: Vec<u32>,
        scales: Vec<u16>,
        n: usize,
        k: usize,
        group_size: usize,
    },
    DenseF16 {
        weight: Vec<u16>,
        n: usize,
        k: usize,
    },
}

impl RawLinearHost {
    fn load(
        archive: &SafetensorsArchive,
        base_prefix: &str,
        default_group_size: usize,
    ) -> Result<Self> {
        let packed_key = format!("{base_prefix}.weight_packed");
        if let Some(real_packed) = archive.resolve_name(&packed_key) {
            let scale_key = archive
                .resolve_name(&format!("{base_prefix}.weight_scale"))
                .ok_or_else(|| loader_err("missing weight_scale"))?;
            let p_info = archive
                .tensor_info(&real_packed)
                .map_err(|_| loader_err("tensor_info weight_packed"))?;
            let s_info = archive
                .tensor_info(&scale_key)
                .map_err(|_| loader_err("tensor_info weight_scale"))?;
            let n = p_info.shape[0];
            let k = p_info.shape[1] * 8;
            let num_groups = s_info.shape[1];
            let group_size = k / num_groups;
            let packed = archive
                .read_u32(&real_packed)
                .map_err(|_| loader_err("read_u32 weight_packed"))?;
            let scales = archive
                .read_f16_bits(&scale_key)
                .map_err(|_| loader_err("read_f16_bits weight_scale"))?;
            return Ok(Self::U4B8NMajor {
                packed,
                scales,
                n,
                k,
                group_size,
            });
        }

        let qweight_key = format!("{base_prefix}.qweight");
        if let Some(real_qw) = archive.resolve_name(&qweight_key) {
            let scale_key = archive
                .resolve_name(&format!("{base_prefix}.scales"))
                .ok_or_else(|| loader_err("missing GPTQ scales"))?;
            let qw_info = archive
                .tensor_info(&real_qw)
                .map_err(|_| loader_err("tensor_info qweight"))?;
            let k = qw_info.shape[0] * 8;
            let n = qw_info.shape[1];
            let qw_k8_n = archive
                .read_u32(&real_qw)
                .map_err(|_| loader_err("read_u32 qweight"))?;
            let packed = transpose_gptq_qweight_k8_n_to_n_major(&qw_k8_n, k, n);
            let scales_kg_n = archive
                .read_f16_bits(&scale_key)
                .map_err(|_| loader_err("read_f16_bits scales"))?;
            let s_info = archive
                .tensor_info(&scale_key)
                .map_err(|_| loader_err("tensor_info scales"))?;
            let num_groups = s_info.shape[0];
            let group_size = k.checked_div(num_groups).unwrap_or(default_group_size);
            let mut scales_n_kg = vec![0u16; n * num_groups];
            for g in 0..num_groups {
                for c in 0..n {
                    scales_n_kg[c * num_groups + g] = scales_kg_n[g * n + c];
                }
            }
            return Ok(Self::U4B8NMajor {
                packed,
                scales: scales_n_kg,
                n,
                k,
                group_size,
            });
        }

        let weight_key = archive
            .resolve_name(&format!("{base_prefix}.weight"))
            .ok_or_else(|| loader_err("missing weight"))?;
        let w_info = archive
            .tensor_info(&weight_key)
            .map_err(|_| loader_err("tensor_info weight"))?;
        let n = w_info.shape[0];
        let k = w_info.shape[1];
        let weight = archive
            .read_f16_bits(&weight_key)
            .map_err(|_| loader_err("read_f16_bits weight"))?;
        Ok(Self::DenseF16 { weight, n, k })
    }

    fn slice_rows(&self, row_start: usize, row_end: usize) -> Self {
        let out_n = row_end - row_start;
        match self {
            Self::U4B8NMajor {
                packed,
                scales,
                k,
                group_size,
                ..
            } => {
                let words_per_row = k / 8;
                let groups_per_row = k / group_size;
                Self::U4B8NMajor {
                    packed: packed[row_start * words_per_row..row_end * words_per_row].to_vec(),
                    scales: scales[row_start * groups_per_row..row_end * groups_per_row].to_vec(),
                    n: out_n,
                    k: *k,
                    group_size: *group_size,
                }
            }
            Self::DenseF16 { weight, k, .. } => Self::DenseF16 {
                weight: weight[row_start * k..row_end * k].to_vec(),
                n: out_n,
                k: *k,
            },
        }
    }

    fn slice_cols(&self, col_start: usize, col_end: usize) -> Self {
        let out_k = col_end - col_start;
        match self {
            Self::U4B8NMajor {
                packed,
                scales,
                n,
                k,
                group_size,
            } => {
                let in_words = k / 8;
                let out_words = out_k / 8;
                let w_start = col_start / 8;
                let in_groups = k / group_size;
                let out_groups = out_k / group_size;
                let g_start = col_start / group_size;
                let mut new_packed = vec![0u32; n * out_words];
                let mut new_scales = vec![0u16; n * out_groups];
                for r in 0..*n {
                    new_packed[r * out_words..(r + 1) * out_words].copy_from_slice(
                        &packed[r * in_words + w_start..r * in_words + w_start + out_words],
                    );
                    new_scales[r * out_groups..(r + 1) * out_groups].copy_from_slice(
                        &scales[r * in_groups + g_start..r * in_groups + g_start + out_groups],
                    );
                }
                Self::U4B8NMajor {
                    packed: new_packed,
                    scales: new_scales,
                    n: *n,
                    k: out_k,
                    group_size: *group_size,
                }
            }
            Self::DenseF16 { weight, n, k } => {
                let mut new_weight = vec![0u16; n * out_k];
                for r in 0..*n {
                    new_weight[r * out_k..(r + 1) * out_k]
                        .copy_from_slice(&weight[r * k + col_start..r * k + col_end]);
                }
                Self::DenseF16 {
                    weight: new_weight,
                    n: *n,
                    k: out_k,
                }
            }
        }
    }

    fn concat_rows(parts: &[Self]) -> Self {
        assert!(!parts.is_empty());
        let k = match &parts[0] {
            Self::U4B8NMajor { k, .. } | Self::DenseF16 { k, .. } => *k,
        };
        let all_u4 = parts.iter().all(|p| matches!(p, Self::U4B8NMajor { .. }));
        if all_u4 {
            let group_size = match &parts[0] {
                Self::U4B8NMajor { group_size, .. } => *group_size,
                _ => unreachable!(),
            };
            let mut packed = Vec::new();
            let mut scales = Vec::new();
            let mut total_n = 0;
            for p in parts {
                if let Self::U4B8NMajor {
                    packed: pk,
                    scales: sc,
                    n,
                    ..
                } = p
                {
                    packed.extend_from_slice(pk);
                    scales.extend_from_slice(sc);
                    total_n += *n;
                }
            }
            Self::U4B8NMajor {
                packed,
                scales,
                n: total_n,
                k,
                group_size,
            }
        } else {
            let mut weight = Vec::new();
            let mut total_n = 0;
            for p in parts {
                match p {
                    Self::DenseF16 { weight: w, n, .. } => {
                        weight.extend_from_slice(w);
                        total_n += *n;
                    }
                    Self::U4B8NMajor {
                        packed,
                        scales,
                        n,
                        k,
                        group_size,
                    } => {
                        let dq = dequant_u4b8_n_major_to_f16(packed, scales, *n, *k, *group_size);
                        weight.extend_from_slice(&dq);
                        total_n += *n;
                    }
                }
            }
            Self::DenseF16 {
                weight,
                n: total_n,
                k,
            }
        }
    }

    #[cfg(feature = "cuda")]
    fn upload(
        &self,
        arena: &mut DeviceArena,
        stream: &CudaStream,
        repack_staging: &DeviceBuffer,
    ) -> Result<LinearProj> {
        match self {
            Self::U4B8NMajor {
                packed,
                scales,
                n,
                k,
                group_size,
            } => {
                if n.is_multiple_of(64)
                    && k.is_multiple_of(128)
                    && (*group_size == 32 || *group_size == 128)
                {
                    repack_staging.copy_from_host(packed, stream)?;
                    let b_q_weight_ptr = arena.alloc_aligned(packed.len() * 4)?;
                    let ex = stream.exec(0, 0);
                    check_rc(
                        unsafe {
                            ffi::w34_marlin_repack_int4(
                                &ex,
                                repack_staging.addr() as *const u32,
                                b_q_weight_ptr as *mut u32,
                                *k as i64,
                                *n as i64,
                            )
                        },
                        "w34_marlin_repack_int4",
                    )?;
                    let perm_scales =
                        marlin_permute_scales_from_n_major(scales, *k, *n, *group_size);
                    let b_scales_ptr = arena.upload_slice(&perm_scales, stream)?;
                    Ok(LinearProj::MarlinW4A16 {
                        b_q_weight_ptr,
                        b_scales_ptr,
                        size_n: *n,
                        size_k: *k,
                        group_blocks: (*group_size / 16) as i32,
                    })
                } else {
                    let f16_w = dequant_u4b8_n_major_to_f16(packed, scales, *n, *k, *group_size);
                    let weight_ptr = arena.upload_slice(&f16_w, stream)?;
                    Ok(LinearProj::DenseF16 {
                        weight_ptr,
                        size_n: *n,
                        size_k: *k,
                    })
                }
            }
            Self::DenseF16 { weight, n, k } => {
                let weight_ptr = arena.upload_slice(weight, stream)?;
                Ok(LinearProj::DenseF16 {
                    weight_ptr,
                    size_n: *n,
                    size_k: *k,
                })
            }
        }
    }
}

#[cfg(feature = "cuda")]
pub struct FullAttentionLayerWeights {
    pub layer_id: usize,
    pub kv_layer_ordinal: usize,
    pub qkv_proj: LinearProj,
    pub o_proj: LinearProj,
    pub q_norm_ptr: u64,
    pub k_norm_ptr: u64,
}

#[cfg(feature = "cuda")]
pub struct GdnLayerWeights {
    pub layer_id: usize,
    pub gdn_layer_ordinal: usize,
    pub in_proj_qkvz: LinearProj,
    pub in_proj_ba: LinearProj,
    pub out_proj: LinearProj,
    pub conv_weight_ptr: u64,
    pub a_log_ptr: u64,
    pub dt_bias_ptr: u64,
    pub norm_weight_ptr: u64,
}

#[cfg(feature = "cuda")]
pub enum LayerBlockWeights {
    FullAttention(FullAttentionLayerWeights),
    LinearAttention(GdnLayerWeights),
}

#[cfg(feature = "cuda")]
pub struct DecoderLayerWeights {
    pub layer_id: usize,
    pub input_layernorm_ptr: u64,
    pub post_attention_layernorm_ptr: u64,
    pub block: LayerBlockWeights,
    pub gate_up_proj: LinearProj,
    pub down_proj: LinearProj,
}

/// Static device buffers used by [`CUDAGraphRunner`] for `M=1` single-request decode.
#[cfg(feature = "cuda")]
pub struct DecodeGraphIo {
    pub token_ids: DeviceBuffer,
    pub positions: DeviceBuffer,
    pub q_start_loc: DeviceBuffer,
    pub ctx_lens_before: DeviceBuffer,
    pub seq_lens_after: DeviceBuffer,
    pub block_tables: DeviceBuffer,
    pub write_slot_mapping: DeviceBuffer,
    pub state_in_indices: DeviceBuffer,
    pub state_out_indices: DeviceBuffer,
    pub has_initial_state: DeviceBuffer,
    pub out_greedy_id: DeviceBuffer,
    pub out_top5_ids: DeviceBuffer,
    pub out_top5_logprobs: DeviceBuffer,
    pub local_max_val: DeviceBuffer,
    pub local_max_id: DeviceBuffer,
    pub global_max_val: DeviceBuffer,
    pub global_max_id: DeviceBuffer,
    pub max_blocks_per_req: usize,
}

#[cfg(feature = "cuda")]
impl DecodeGraphIo {
    pub fn new(device_id: i32, max_blocks_per_req: usize, stream: &CudaStream) -> Result<Self> {
        let token_ids = DeviceBuffer::alloc(device_id, 4)?;
        let positions = DeviceBuffer::alloc(device_id, 4)?;
        let q_start_loc = DeviceBuffer::alloc(device_id, 8)?;
        q_start_loc.copy_from_host(&[0i32, 1], stream)?;
        let ctx_lens_before = DeviceBuffer::alloc(device_id, 4)?;
        let seq_lens_after = DeviceBuffer::alloc(device_id, 4)?;
        let block_tables = DeviceBuffer::alloc(device_id, max_blocks_per_req * 4)?;
        let write_slot_mapping = DeviceBuffer::alloc(device_id, 4)?;
        let state_in_indices = DeviceBuffer::alloc(device_id, 4)?;
        let state_out_indices = DeviceBuffer::alloc(device_id, 4)?;
        let has_initial_state = DeviceBuffer::alloc(device_id, 4)?;
        has_initial_state.copy_from_host(&[1u8, 0, 0, 0], stream)?;
        let out_greedy_id = DeviceBuffer::alloc(device_id, 4)?;
        let out_top5_ids = DeviceBuffer::alloc(device_id, 5 * 4)?;
        let out_top5_logprobs = DeviceBuffer::alloc(device_id, 5 * 4)?;
        let local_max_val = DeviceBuffer::alloc(device_id, 4)?;
        let local_max_id = DeviceBuffer::alloc(device_id, 4)?;
        let global_max_val = DeviceBuffer::alloc(device_id, 4)?;
        let global_max_id = DeviceBuffer::alloc(device_id, 4)?;
        Ok(Self {
            token_ids,
            positions,
            q_start_loc,
            ctx_lens_before,
            seq_lens_after,
            block_tables,
            write_slot_mapping,
            state_in_indices,
            state_out_indices,
            has_initial_state,
            out_greedy_id,
            out_top5_ids,
            out_top5_logprobs,
            local_max_val,
            local_max_id,
            global_max_val,
            global_max_id,
            max_blocks_per_req,
        })
    }
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
    pub decode_graph: Option<CudaGraph>,
    pub capacity_plan: CapacityPlan,
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
        prepare_execution_plan(device_id)?;
        let device = CudaDevice::set_current(device_id)?;
        let stream = CudaStream::new(device_id)?;
        let cublas = CublasHandle::new(&stream)?;
        let (free_vram, total_vram) = device.mem_info()?;

        // Estimate weight arena size generously based on model size & TP degree
        let est_weight_bytes = match (cfg.num_hidden_layers, tp_size) {
            (64, 2) => 12 * 1024 * 1024 * 1024, // 27B TP=2 (~9.5 GiB/rank)
            (64, _) => 19 * 1024 * 1024 * 1024,
            (_, 2) => 5 * 1024 * 1024 * 1024, // 4B/9B TP=2
            _ => 8 * 1024 * 1024 * 1024,      // 4B/9B TP=1
        };
        let mut weight_arena = DeviceArena::new(device_id, est_weight_bytes)?;
        let scratch_arena = DeviceArena::new(device_id, 512 * 1024 * 1024)?;
        let marlin_ws = DeviceBuffer::alloc(device_id, 84 * 64 * 4)?;
        let repack_staging = DeviceBuffer::alloc(device_id, 128 * 1024 * 1024)?;

        let default_group_size = match &cfg.quant {
            QuantScheme::CompressedTensors { group_size, .. }
            | QuantScheme::Gptq { group_size, .. }
            | QuantScheme::Awq { group_size, .. }
            | QuantScheme::AutoRound { group_size, .. } => *group_size,
            QuantScheme::Unquantized => 32,
        };

        // 1. embed_tokens (replicated in FP16)
        let embed_key = archive
            .resolve_name("model.embed_tokens.weight")
            .ok_or_else(|| loader_err("missing model.embed_tokens.weight"))?;
        let embed_f16 = archive
            .read_f16_bits(&embed_key)
            .map_err(|_| loader_err("read embed_tokens"))?;
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
            let raw_lm = RawLinearHost::load(archive, "lm_head", default_group_size)?;
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
                        default_group_size,
                    )?;
                    debug_assert_eq!(raw_q.slice_rows(0, q_gate_total).size_check(), q_gate_total);
                    let q_shard = raw_q.slice_rows(rank * q_gate_local, (rank + 1) * q_gate_local);

                    let raw_k = RawLinearHost::load(
                        archive,
                        &format!("{pfx}.self_attn.k_proj"),
                        default_group_size,
                    )?;
                    let k_shard = raw_k.slice_rows(rank * kv_local, (rank + 1) * kv_local);

                    let raw_v = RawLinearHost::load(
                        archive,
                        &format!("{pfx}.self_attn.v_proj"),
                        default_group_size,
                    )?;
                    let v_shard = raw_v.slice_rows(rank * kv_local, (rank + 1) * kv_local);

                    let qkv_shard = RawLinearHost::concat_rows(&[q_shard, k_shard, v_shard]);
                    let qkv_proj = qkv_shard.upload(&mut weight_arena, &stream, &repack_staging)?;

                    let raw_o = RawLinearHost::load(
                        archive,
                        &format!("{pfx}.self_attn.o_proj"),
                        default_group_size,
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
                        default_group_size,
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
                        default_group_size,
                    )?;
                    let z_shard = raw_z.slice_rows(rank * gdn_v_local, (rank + 1) * gdn_v_local);

                    let qkvz_shard = RawLinearHost::concat_rows(&[q_part, k_part, v_part, z_shard]);
                    let in_proj_qkvz =
                        qkvz_shard.upload(&mut weight_arena, &stream, &repack_staging)?;

                    let raw_b = RawLinearHost::load(
                        archive,
                        &format!("{pfx}.linear_attn.in_proj_b"),
                        default_group_size,
                    )?;
                    let b_shard =
                        raw_b.slice_rows(rank * local_v_heads, (rank + 1) * local_v_heads);

                    let raw_a = RawLinearHost::load(
                        archive,
                        &format!("{pfx}.linear_attn.in_proj_a"),
                        default_group_size,
                    )?;
                    let a_shard =
                        raw_a.slice_rows(rank * local_v_heads, (rank + 1) * local_v_heads);

                    let ba_shard = RawLinearHost::concat_rows(&[b_shard, a_shard]);
                    let in_proj_ba =
                        ba_shard.upload(&mut weight_arena, &stream, &repack_staging)?;

                    let raw_out = RawLinearHost::load(
                        archive,
                        &format!("{pfx}.linear_attn.out_proj"),
                        default_group_size,
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

            let raw_gate =
                RawLinearHost::load(archive, &format!("{pfx}.mlp.gate_proj"), default_group_size)?;
            let gate_shard = raw_gate.slice_rows(rank * local_interm, (rank + 1) * local_interm);

            let raw_up =
                RawLinearHost::load(archive, &format!("{pfx}.mlp.up_proj"), default_group_size)?;
            let up_shard = raw_up.slice_rows(rank * local_interm, (rank + 1) * local_interm);

            let gate_up_shard = RawLinearHost::concat_rows(&[gate_shard, up_shard]);
            let gate_up_proj = gate_up_shard.upload(&mut weight_arena, &stream, &repack_staging)?;

            let raw_down =
                RawLinearHost::load(archive, &format!("{pfx}.mlp.down_proj"), default_group_size)?;
            let down_shard = raw_down.slice_cols(rank * local_interm, (rank + 1) * local_interm);
            let down_proj = down_shard.upload(&mut weight_arena, &stream, &repack_staging)?;

            layers.push(DecoderLayerWeights {
                layer_id,
                input_layernorm_ptr,
                post_attention_layernorm_ptr,
                block,
                gate_up_proj,
                down_proj,
            });
        }

        drop(repack_staging);
        stream.synchronize()?;

        // 5. StateBundlePool & KvPool via CapacityPlanner
        let total_state_slots = 24usize;
        let conv_dim = 2 * gdn_k_local + gdn_v_local;
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
        let state_pool_bytes = gdn_layer_ids.len()
            * total_state_slots
            * (conv_dim * (cfg.linear_conv_kernel_dim - 1) * 2
                + local_v_heads * cfg.linear_value_head_dim * cfg.linear_key_head_dim * 4);

        let block_size = 16usize;
        let mut bytes_per_kv_page_all_layers = 0usize;
        for ord in 0..full_attn_layer_ids.len() {
            let (kf, vf, _, _) = kv_policy.resolve_layer(ord, full_attn_layer_ids.len());
            let kb = crate::cache::kv_format_row_bytes(kf, cfg.head_dim);
            let vb = crate::cache::kv_format_row_bytes(vf, cfg.head_dim);
            bytes_per_kv_page_all_layers += local_kv_heads * block_size * (kb + vb);
        }

        let default_pages = requested_kv_pages.or(Some(256));
        let capacity_plan = CapacityPlanner::plan(
            total_vram,
            free_vram,
            weight_arena.used(),
            scratch_arena.capacity(),
            state_pool_bytes,
            bytes_per_kv_page_all_layers,
            default_pages,
        )
        .map_err(|_| loader_err("CapacityPlanner failed"))?;

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
            capacity_plan,
        })
    }

    /// Executes the full model spine for `num_tokens` tokens on this rank.
    #[allow(clippy::too_many_arguments)]
    pub fn forward_device_batches(
        &mut self,
        token_ids_ptr: u64,
        positions_ptr: u64,
        num_tokens: usize,
        attn_batch: &W34AttentionBatch,
        gdn_batch: &W34GdnBatch,
        tp2: Option<&Tp2Collectives>,
        out_greedy_id_ptr: u64,
        out_top5_ids_ptr: u64,
        out_top5_logprobs_ptr: u64,
        mut dump_out: Option<&mut ForwardDump>,
    ) -> Result<i32> {
        self.device.make_current()?;
        self.scratch_arena.reset();
        let ex = self.stream.exec(
            self.scratch_arena.allocation_base(),
            self.scratch_arena.capacity(),
        );

        let h_dim = self.cfg.hidden_size;
        let local_q_heads = self.cfg.num_attention_heads / self.tp_size;
        let local_kv_heads = self.cfg.num_key_value_heads / self.tp_size;
        let head_dim = self.cfg.head_dim;
        let local_k_heads = self.cfg.linear_num_key_heads / self.tp_size;
        let local_v_heads = self.cfg.linear_num_value_heads / self.tp_size;
        let head_k_dim = self.cfg.linear_key_head_dim;
        let head_v_dim = self.cfg.linear_value_head_dim;
        let conv_dim = 2 * local_k_heads * head_k_dim + local_v_heads * head_v_dim;
        let local_interm = self.cfg.intermediate_size / self.tp_size;
        let marlin_ws_ptr = self.marlin_ws.addr();
        let marlin_ws_ints = self.marlin_ws.len_bytes / 4;

        let residual_buf = self.scratch_arena.alloc_aligned(num_tokens * h_dim * 2)?;
        let normed_buf = self.scratch_arena.alloc_aligned(num_tokens * h_dim * 2)?;
        let block_out_buf = self.scratch_arena.alloc_aligned(num_tokens * h_dim * 2)?;
        let mlp_out_buf = self.scratch_arena.alloc_aligned(num_tokens * h_dim * 2)?;

        let q_gate_dim = local_q_heads * head_dim * 2;
        let kv_dim = local_kv_heads * head_dim;
        let gdn_v_dim = local_v_heads * head_v_dim;

        let qkv_proj_buf = self
            .scratch_arena
            .alloc_aligned(num_tokens * (q_gate_dim + 2 * kv_dim) * 2)?;
        let q_gate_buf = self
            .scratch_arena
            .alloc_aligned(num_tokens * q_gate_dim * 2)?;
        let k_raw_buf = self.scratch_arena.alloc_aligned(num_tokens * kv_dim * 2)?;
        let v_buf = self.scratch_arena.alloc_aligned(num_tokens * kv_dim * 2)?;
        let q_buf = self
            .scratch_arena
            .alloc_aligned(num_tokens * local_q_heads * head_dim * 2)?;
        let k_buf = self.scratch_arena.alloc_aligned(num_tokens * kv_dim * 2)?;
        let gate_buf = self
            .scratch_arena
            .alloc_aligned(num_tokens * local_q_heads * head_dim * 2)?;
        let attn_out_buf = self
            .scratch_arena
            .alloc_aligned(num_tokens * local_q_heads * head_dim * 2)?;
        let gated_attn_buf = self
            .scratch_arena
            .alloc_aligned(num_tokens * local_q_heads * head_dim * 2)?;

        let qkvz_buf = self
            .scratch_arena
            .alloc_aligned(num_tokens * (conv_dim + gdn_v_dim) * 2)?;
        let ba_buf = self
            .scratch_arena
            .alloc_aligned(num_tokens * 2 * local_v_heads * 2)?;
        let qkv_buf = self
            .scratch_arena
            .alloc_aligned(num_tokens * conv_dim * 2)?;
        let qkv_conv_out_buf = self
            .scratch_arena
            .alloc_aligned(num_tokens * conv_dim * 2)?;
        let z_buf = self
            .scratch_arena
            .alloc_aligned(num_tokens * gdn_v_dim * 2)?;
        let b_buf = self
            .scratch_arena
            .alloc_aligned(num_tokens * local_v_heads * 2)?;
        let a_buf = self
            .scratch_arena
            .alloc_aligned(num_tokens * local_v_heads * 2)?;
        let gdn_raw_out_buf = self
            .scratch_arena
            .alloc_aligned(num_tokens * gdn_v_dim * 2)?;
        let gdn_normed_buf = self
            .scratch_arena
            .alloc_aligned(num_tokens * gdn_v_dim * 2)?;

        let gate_up_buf = self
            .scratch_arena
            .alloc_aligned(num_tokens * local_interm * 2 * 2)?;
        let act_buf = self
            .scratch_arena
            .alloc_aligned(num_tokens * local_interm * 2)?;
        let logits_buf = self
            .scratch_arena
            .alloc_aligned(self.local_vocab_size * 2)?;

        // 1. Embedding lookup into residual_buf
        check_rc(
            unsafe {
                ffi::w34_embed_lookup_f16(
                    &ex,
                    token_ids_ptr as *const i32,
                    self.embed_tokens_ptr as *const c_void,
                    residual_buf as *mut c_void,
                    core::ptr::null(),
                    num_tokens as i32,
                    h_dim as i32,
                    self.cfg.vocab_size as i32,
                )
            },
            "w34_embed_lookup_f16",
        )?;

        if let Some(dump) = dump_out.as_deref_mut()
            && self.rank == 0
        {
            Self::record_dump_f16(
                &self.stream,
                residual_buf,
                vec![num_tokens, h_dim],
                "embed",
                dump,
            )?;
        }

        // 2. Layers
        for layer_idx in 0..self.layers.len() {
            let in_ln_ptr = self.layers[layer_idx].input_layernorm_ptr;
            let post_ln_ptr = self.layers[layer_idx].post_attention_layernorm_ptr;
            if layer_idx == 0 {
                check_rc(
                    unsafe {
                        ffi::w34_gemma_rms_norm_f16(
                            &ex,
                            residual_buf as *const c_void,
                            in_ln_ptr as *const c_void,
                            normed_buf as *mut c_void,
                            core::ptr::null(),
                            num_tokens as i32,
                            h_dim as i32,
                            self.cfg.rms_norm_eps,
                        )
                    },
                    "w34_gemma_rms_norm_f16",
                )?;
            } else {
                check_rc(
                    unsafe {
                        ffi::w34_fused_add_gemma_rms_norm_f16(
                            &ex,
                            mlp_out_buf as *const c_void,
                            residual_buf as *mut c_void,
                            in_ln_ptr as *const c_void,
                            normed_buf as *mut c_void,
                            core::ptr::null(),
                            num_tokens as i32,
                            h_dim as i32,
                            self.cfg.rms_norm_eps,
                        )
                    },
                    "w34_fused_add_gemma_rms_norm_f16 (input)",
                )?;
                if let Some(dump) = dump_out.as_deref_mut()
                    && self.rank == 0
                {
                    Self::record_dump_f16(
                        &self.stream,
                        residual_buf,
                        vec![num_tokens, h_dim],
                        &format!("layer_{}_hidden_after", layer_idx - 1),
                        dump,
                    )?;
                }
            }

            match &self.layers[layer_idx].block {
                LayerBlockWeights::FullAttention(fa) => {
                    fa.qkv_proj.forward(
                        &ex,
                        &self.cublas,
                        marlin_ws_ptr,
                        marlin_ws_ints,
                        normed_buf,
                        qkv_proj_buf,
                        num_tokens,
                    )?;
                    let (q_gate_ptr, k_raw_ptr, v_ptr) = if num_tokens == 1 {
                        (
                            qkv_proj_buf,
                            qkv_proj_buf + (q_gate_dim * 2) as u64,
                            qkv_proj_buf + ((q_gate_dim + kv_dim) * 2) as u64,
                        )
                    } else {
                        check_rc(
                            unsafe {
                                ffi::w34_split_cols_3_f16(
                                    &ex,
                                    qkv_proj_buf as *const c_void,
                                    q_gate_buf as *mut c_void,
                                    k_raw_buf as *mut c_void,
                                    v_buf as *mut c_void,
                                    num_tokens as i32,
                                    q_gate_dim as i32,
                                    kv_dim as i32,
                                    kv_dim as i32,
                                )
                            },
                            "w34_split_cols_3_f16",
                        )?;
                        (q_gate_buf, k_raw_buf, v_buf)
                    };
                    check_rc(
                        unsafe {
                            ffi::w34_fused_qk_rmsnorm_rope_gate_f16(
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
                                self.cfg.rotary_dim as i32,
                                self.cfg.mrope_section[0] as i32,
                                self.cfg.mrope_section[1] as i32,
                                self.cfg.mrope_section[2] as i32,
                                self.cfg.rope_theta,
                                self.cfg.rms_norm_eps,
                            )
                        },
                        "w34_fused_qk_rmsnorm_rope_gate_f16",
                    )?;
                    let kv_layout = &self.kv_pool.layers[fa.kv_layer_ordinal].layout;
                    check_rc(
                        unsafe {
                            ffi::w34_kv_cache_write(
                                &ex,
                                kv_layout,
                                attn_batch,
                                k_buf as *const c_void,
                                v_ptr as *const c_void,
                            )
                        },
                        "w34_kv_cache_write",
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
                    fa.o_proj.forward(
                        &ex,
                        &self.cublas,
                        marlin_ws_ptr,
                        marlin_ws_ints,
                        gated_attn_buf,
                        block_out_buf,
                        num_tokens,
                    )?;
                }
                LayerBlockWeights::LinearAttention(gdn) => {
                    gdn.in_proj_qkvz.forward(
                        &ex,
                        &self.cublas,
                        marlin_ws_ptr,
                        marlin_ws_ints,
                        normed_buf,
                        qkvz_buf,
                        num_tokens,
                    )?;
                    gdn.in_proj_ba.forward(
                        &ex,
                        &self.cublas,
                        marlin_ws_ptr,
                        marlin_ws_ints,
                        normed_buf,
                        ba_buf,
                        num_tokens,
                    )?;
                    let (qkv_ptr, z_ptr, b_ptr, a_ptr) = if num_tokens == 1 {
                        (
                            qkvz_buf,
                            qkvz_buf + (conv_dim * 2) as u64,
                            ba_buf,
                            ba_buf + (local_v_heads * 2) as u64,
                        )
                    } else {
                        check_rc(
                            unsafe {
                                ffi::w34_split_cols_2_f16(
                                    &ex,
                                    qkvz_buf as *const c_void,
                                    qkv_buf as *mut c_void,
                                    z_buf as *mut c_void,
                                    num_tokens as i32,
                                    conv_dim as i32,
                                    gdn_v_dim as i32,
                                )
                            },
                            "w34_split_cols_2_f16 (qkvz)",
                        )?;
                        check_rc(
                            unsafe {
                                ffi::w34_split_cols_2_f16(
                                    &ex,
                                    ba_buf as *const c_void,
                                    b_buf as *mut c_void,
                                    a_buf as *mut c_void,
                                    num_tokens as i32,
                                    local_v_heads as i32,
                                    local_v_heads as i32,
                                )
                            },
                            "w34_split_cols_2_f16 (ba)",
                        )?;
                        (qkv_buf, z_buf, b_buf, a_buf)
                    };
                    let st_bufs = &self.state_pool.layers[gdn.gdn_layer_ordinal];
                    check_rc(
                        unsafe {
                            ffi::w34_gdn_causal_conv1d_f16(
                                &ex,
                                gdn_batch,
                                qkv_ptr as *const c_void,
                                gdn.conv_weight_ptr as *const c_void,
                                core::ptr::null(),
                                st_bufs.conv_state.addr() as *mut c_void,
                                qkv_conv_out_buf as *mut c_void,
                                conv_dim as i32,
                                self.cfg.linear_conv_kernel_dim as i32,
                                self.state_pool.total_slots as i32,
                                1,
                            )
                        },
                        "w34_gdn_causal_conv1d_f16",
                    )?;
                    check_rc(
                        unsafe {
                            ffi::w34_gdn_recurrent_f16(
                                &ex,
                                gdn_batch,
                                qkv_conv_out_buf as *const c_void,
                                core::ptr::null(),
                                core::ptr::null(),
                                core::ptr::null(),
                                b_ptr as *const c_void,
                                a_ptr as *const c_void,
                                gdn.a_log_ptr as *const f32,
                                gdn.dt_bias_ptr as *const f32,
                                st_bufs.ssm_state.addr() as *mut f32,
                                gdn_raw_out_buf as *mut c_void,
                                local_k_heads as i32,
                                head_k_dim as i32,
                                local_v_heads as i32,
                                head_v_dim as i32,
                                self.state_pool.total_slots as i32,
                            )
                        },
                        "w34_gdn_recurrent_f16",
                    )?;
                    check_rc(
                        unsafe {
                            ffi::w34_rms_norm_gated_silu_f16(
                                &ex,
                                gdn_raw_out_buf as *const c_void,
                                z_ptr as *const c_void,
                                gdn.norm_weight_ptr as *const c_void,
                                gdn_normed_buf as *mut c_void,
                                core::ptr::null(),
                                num_tokens as i32,
                                local_v_heads as i32,
                                head_v_dim as i32,
                                self.cfg.rms_norm_eps,
                            )
                        },
                        "w34_rms_norm_gated_silu_f16",
                    )?;
                    gdn.out_proj.forward(
                        &ex,
                        &self.cublas,
                        marlin_ws_ptr,
                        marlin_ws_ints,
                        gdn_normed_buf,
                        block_out_buf,
                        num_tokens,
                    )?;
                }
            }

            if let Some(comm) = tp2 {
                comm.all_reduce_f16(
                    &ex,
                    block_out_buf,
                    block_out_buf,
                    self.rank as i32,
                    (num_tokens * h_dim) as i32,
                )?;
            }

            if let Some(dump) = dump_out.as_deref_mut()
                && self.rank == 0
            {
                Self::record_dump_f16(
                    &self.stream,
                    block_out_buf,
                    vec![num_tokens, h_dim],
                    &format!("layer_{layer_idx}_attn_or_gdn_out"),
                    dump,
                )?;
            }

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
                        self.cfg.rms_norm_eps,
                    )
                },
                "w34_fused_add_gemma_rms_norm_f16 (post_attn)",
            )?;

            self.layers[layer_idx].gate_up_proj.forward(
                &ex,
                &self.cublas,
                marlin_ws_ptr,
                marlin_ws_ints,
                normed_buf,
                gate_up_buf,
                num_tokens,
            )?;
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
            self.layers[layer_idx].down_proj.forward(
                &ex,
                &self.cublas,
                marlin_ws_ptr,
                marlin_ws_ints,
                act_buf,
                mlp_out_buf,
                num_tokens,
            )?;
            if let Some(comm) = tp2 {
                comm.all_reduce_f16(
                    &ex,
                    mlp_out_buf,
                    mlp_out_buf,
                    self.rank as i32,
                    (num_tokens * h_dim) as i32,
                )?;
            }
        }

        // 3. Final RMSNorm
        check_rc(
            unsafe {
                ffi::w34_fused_add_gemma_rms_norm_f16(
                    &ex,
                    mlp_out_buf as *const c_void,
                    residual_buf as *mut c_void,
                    self.final_norm_ptr as *const c_void,
                    normed_buf as *mut c_void,
                    core::ptr::null(),
                    num_tokens as i32,
                    h_dim as i32,
                    self.cfg.rms_norm_eps,
                )
            },
            "w34_fused_add_gemma_rms_norm_f16 (final)",
        )?;

        if let Some(dump) = dump_out.as_deref_mut()
            && self.rank == 0
        {
            Self::record_dump_f16(
                &self.stream,
                residual_buf,
                vec![num_tokens, h_dim],
                &format!("layer_{}_hidden_after", self.layers.len() - 1),
                dump,
            )?;
            Self::record_dump_f16(
                &self.stream,
                normed_buf,
                vec![num_tokens, h_dim],
                "final_norm",
                dump,
            )?;
        }

        // 4. LM Head on last token row `[1, h_dim]`
        let last_hidden_ptr = normed_buf + ((num_tokens - 1) * h_dim * 2) as u64;
        self.lm_head.forward(
            &ex,
            &self.cublas,
            marlin_ws_ptr,
            marlin_ws_ints,
            last_hidden_ptr,
            logits_buf,
            1,
        )?;

        // 5. Sampler
        if let Some(comm) = tp2 {
            // Fast 1-pass local argmax + raw max logit in FP32, then TP=2 top-1 gather
            check_rc(
                unsafe {
                    ffi::w34_sampler_greedy_top5_f16(
                        &ex,
                        logits_buf as *const c_void,
                        self.decode_io.local_max_id.addr() as *mut i32,
                        core::ptr::null_mut(),
                        self.decode_io.local_max_val.addr() as *mut f32,
                        1,
                        self.local_vocab_size as i32,
                        self.vocab_offset as i32,
                    )
                },
                "w34_sampler_greedy_top5_f16 (tp2 local)",
            )?;
            comm.all_gather_top1_f32_i32(
                &ex,
                self.decode_io.local_max_val.addr(),
                self.decode_io.local_max_id.addr(),
                self.decode_io.global_max_val.addr(),
                out_greedy_id_ptr,
                self.rank as i32,
                1,
            )?;
            if out_top5_ids_ptr != 0 && out_top5_logprobs_ptr != 0 {
                check_rc(
                    unsafe {
                        ffi::w34_sampler_greedy_top5_f16(
                            &ex,
                            logits_buf as *const c_void,
                            self.decode_io.local_max_id.addr() as *mut i32,
                            out_top5_ids_ptr as *mut i32,
                            out_top5_logprobs_ptr as *mut f32,
                            1,
                            self.local_vocab_size as i32,
                            self.vocab_offset as i32,
                        )
                    },
                    "w34_sampler_greedy_top5_f16 (tp2 top5)",
                )?;
            }
        } else {
            check_rc(
                unsafe {
                    ffi::w34_sampler_greedy_top5_f16(
                        &ex,
                        logits_buf as *const c_void,
                        out_greedy_id_ptr as *mut i32,
                        out_top5_ids_ptr as *mut i32,
                        out_top5_logprobs_ptr as *mut f32,
                        1,
                        self.local_vocab_size as i32,
                        self.vocab_offset as i32,
                    )
                },
                "w34_sampler_greedy_top5_f16",
            )?;
        }

        if let Some(dump) = dump_out
            && self.rank == 0
        {
            Self::record_dump_f16(
                &self.stream,
                logits_buf,
                vec![self.local_vocab_size],
                "logits_last",
                dump,
            )?;
        }

        Ok(0)
    }

    fn record_dump_f16(
        stream: &CudaStream,
        dev_ptr: u64,
        shape: Vec<usize>,
        name: &str,
        dump: &mut ForwardDump,
    ) -> Result<()> {
        let count: usize = shape.iter().copied().product();
        let mut host_u16 = vec![0u16; count];
        check_rc(
            unsafe {
                ffi::w34_memcpy_d2h_async(
                    host_u16.as_mut_ptr() as *mut c_void,
                    dev_ptr as *const c_void,
                    count * 2,
                    stream.handle,
                )
            },
            "record_dump_f16 d2h",
        )?;
        stream.synchronize()?;
        let vals_f32: Vec<f32> = host_u16.into_iter().map(f16_to_f32).collect();
        dump.tensors_f32.insert(name.to_string(), (shape, vals_f32));
        Ok(())
    }

    /// Captures the `M=1` decode forward pass into `self.decode_graph`.
    pub fn ensure_decode_graph(&mut self, tp2: Option<&Tp2Collectives>) -> Result<()> {
        if self.decode_graph.is_some() {
            return Ok(());
        }
        let attn_batch = W34AttentionBatch {
            query_start_loc: self.decode_io.q_start_loc.addr() as *const i32,
            context_lens_before: self.decode_io.ctx_lens_before.addr() as *const i32,
            seq_lens_after: self.decode_io.seq_lens_after.addr() as *const i32,
            block_tables: self.decode_io.block_tables.addr() as *const i32,
            write_slot_mapping: self.decode_io.write_slot_mapping.addr() as *const i32,
            valid_token_mask: core::ptr::null(),
            valid_req_mask: core::ptr::null(),
            num_reqs: 1,
            total_tokens: 1,
            max_blocks_per_req: self.decode_io.max_blocks_per_req as i32,
            max_q_len: 1,
            max_seq_len: (self.decode_io.max_blocks_per_req * 16) as i32,
            mask_mode: W34_ATTN_MASK_CAUSAL,
            sliding_window: 0,
            sm_scale: (self.cfg.head_dim as f32).powf(-0.5),
            reserved: 0,
        };
        let gdn_batch = W34GdnBatch {
            query_start_loc: self.decode_io.q_start_loc.addr() as *const i32,
            state_in_indices: self.decode_io.state_in_indices.addr() as *const i32,
            state_out_indices: self.decode_io.state_out_indices.addr() as *const i32,
            spec_state_indices: core::ptr::null(),
            checkpoint_dst_indices: core::ptr::null(),
            checkpoint_token_offsets: core::ptr::null(),
            has_initial_state: self.decode_io.has_initial_state.addr() as *const u8,
            valid_req_mask: core::ptr::null(),
            num_reqs: 1,
            total_tokens: 1,
            max_verify_tokens: 0,
            max_checkpoints_per_req: 0,
        };
        let tok_ptr = self.decode_io.token_ids.addr();
        let pos_ptr = self.decode_io.positions.addr();
        let out_id_ptr = self.decode_io.out_greedy_id.addr();

        self.stream.synchronize()?;
        let stream_handle = self.stream.handle;
        let dev_id = self.device.device_id;
        check_rc(unsafe { ffi::w34_set_device(dev_id) }, "w34_set_device")?;
        check_rc(
            unsafe { ffi::w34_graph_begin_capture(stream_handle) },
            "w34_graph_begin_capture",
        )?;
        let res = self.forward_device_batches(
            tok_ptr,
            pos_ptr,
            1,
            &attn_batch,
            &gdn_batch,
            tp2,
            out_id_ptr,
            0,
            0,
            None,
        );
        let mut graph = 0usize;
        let mut graph_exec = 0usize;
        let end_rc = unsafe {
            ffi::w34_graph_end_capture(stream_handle, &raw mut graph, &raw mut graph_exec)
        };
        res?;
        check_rc(end_rc, "w34_graph_end_capture")?;
        self.decode_graph = Some(CudaGraph {
            device_id: dev_id,
            graph,
            graph_exec,
        });
        Ok(())
    }
}

impl RawLinearHost {
    fn size_check(&self) -> usize {
        match self {
            Self::U4B8NMajor { n, .. } | Self::DenseF16 { n, .. } => *n,
        }
    }
}

#[derive(Clone, Debug)]
pub struct BenchMetrics {
    pub prompt_len: usize,
    pub gen_len: usize,
    pub ttft_ms: f64,
    pub prefill_tok_per_s: f64,
    pub decode_ms: f64,
    pub decode_tok_per_s: f64,
    pub itl_ms: f64,
    pub generated_token_ids: Vec<i32>,
}

#[cfg(feature = "cuda")]
unsafe impl Send for Qwen3_5ModelRank {}

/// Top-level `TP=1` / `TP=2` Qwen 3.5 / 3.8 inference engine.
#[cfg(feature = "cuda")]
pub struct Qwen3_5Engine {
    pub cfg: ModelConfig,
    pub tp_size: usize,
    pub ranks: Vec<Qwen3_5ModelRank>,
    pub tp2_comm: Option<Arc<Tp2Collectives>>,
    pub tokenizer: Option<Tokenizer>,
    pub use_cuda_graph: bool,
}

#[cfg(feature = "cuda")]
impl Qwen3_5Engine {
    pub fn load(
        model_spec: &str,
        tp_size: usize,
        kv_policy: KvFormatPolicy,
        requested_kv_pages: Option<usize>,
        use_cuda_graph: bool,
    ) -> Result<Self> {
        assert!(tp_size == 1 || tp_size == 2, "tp_size must be 1 or 2");
        let dir =
            resolve_hf_model_dir(model_spec).map_err(|_| loader_err("resolve_hf_model_dir"))?;
        Self::load_from_dir(&dir, tp_size, kv_policy, requested_kv_pages, use_cuda_graph)
    }

    pub fn load_from_dir(
        dir: &Path,
        tp_size: usize,
        kv_policy: KvFormatPolicy,
        requested_kv_pages: Option<usize>,
        use_cuda_graph: bool,
    ) -> Result<Self> {
        let cfg = ModelConfig::from_dir(dir).map_err(|_| loader_err("ModelConfig::from_dir"))?;
        let archive =
            SafetensorsArchive::open(dir).map_err(|_| loader_err("SafetensorsArchive::open"))?;

        let tp2_comm = if tp_size == 2 {
            // Max collective payload: 4096 tokens * 5120 hidden_size * 2 bytes = 40 MiB
            Some(Arc::new(Tp2Collectives::new(40 * 1024 * 1024)?))
        } else {
            None
        };

        let mut ranks = Vec::with_capacity(tp_size);
        for r in 0..tp_size {
            let rank_model = Qwen3_5ModelRank::load_rank(
                &archive,
                &cfg,
                r,
                tp_size,
                r as i32,
                kv_policy,
                requested_kv_pages,
            )?;
            ranks.push(rank_model);
        }

        let tok_path = dir.join("tokenizer.json");
        let tokenizer = if tok_path.exists() {
            Tokenizer::from_file(&tok_path).ok()
        } else {
            None
        };

        Ok(Self {
            cfg,
            tp_size,
            ranks,
            tp2_comm,
            tokenizer,
            use_cuda_graph,
        })
    }

    pub fn encode(&self, text: &str) -> Result<Vec<i32>> {
        let tok = self
            .tokenizer
            .as_ref()
            .ok_or_else(|| loader_err("tokenizer.json not loaded"))?;
        let enc = tok
            .encode(text, false)
            .map_err(|_| loader_err("Tokenizer::encode failed"))?;
        Ok(enc.get_ids().iter().map(|&id| id as i32).collect())
    }

    pub fn decode(&self, ids: &[i32]) -> Result<String> {
        let tok = self
            .tokenizer
            .as_ref()
            .ok_or_else(|| loader_err("tokenizer.json not loaded"))?;
        let u32_ids: Vec<u32> = ids.iter().map(|&id| id as u32).collect();
        tok.decode(&u32_ids, true)
            .map_err(|_| loader_err("Tokenizer::decode failed"))
    }

    pub fn reset_all_states(&mut self) -> Result<()> {
        for r in &mut self.ranks {
            r.device.make_current()?;
            r.kv_pool.reset_all_pages();
            r.state_pool.reset_all_slots();
            for st in &r.state_pool.layers {
                st.conv_state.zero_async(&r.stream)?;
                st.ssm_state.zero_async(&r.stream)?;
            }
            r.stream.synchronize()?;
        }
        Ok(())
    }

    /// Runs a single prefill or multi-token step from `ctx_before` to `ctx_before + tokens.len()`,
    /// returning `(next_greedy_token_id, top5_ids, top5_logprobs)` and optionally capturing a `ForwardDump`.
    pub fn step_tokens(
        &mut self,
        tokens: &[i32],
        ctx_before: usize,
        block_table: &[i32],
        state_in_slot: i32,
        state_out_slot: i32,
        has_initial_state: bool,
        mut dump_out: Option<&mut ForwardDump>,
    ) -> Result<(i32, [i32; 5], [f32; 5])> {
        let num_tokens = tokens.len();
        if num_tokens == 0 {
            return Err(loader_err("step_tokens requires at least 1 token"));
        }
        let seq_after = ctx_before + num_tokens;
        let max_pages = self.ranks[0].kv_pool.num_pages;
        if seq_after.div_ceil(16) > block_table.len()
            || block_table.len() > max_pages
            || block_table
                .iter()
                .any(|&p| p < 0 || (p as usize) >= max_pages)
        {
            return Err(loader_err("KV cache page capacity exceeded in step_tokens"));
        }
        let positions: Vec<i32> = (ctx_before..seq_after).map(|p| p as i32).collect();
        let write_slots: Vec<i32> = (ctx_before..seq_after)
            .map(|pos| {
                let page_idx = pos / 16;
                let page_off = pos % 16;
                block_table[page_idx] * 16 + (page_off as i32)
            })
            .collect();
        let q_start = [0i32, num_tokens as i32];
        let ctx_arr = [ctx_before as i32];
        let seq_arr = [seq_after as i32];
        let st_in_arr = [state_in_slot];
        let st_out_arr = [state_out_slot];
        let has_init_arr = [u8::from(has_initial_state)];

        if self.tp_size == 1 {
            let r0 = &mut self.ranks[0];
            r0.device.make_current()?;
            let tok_dev = DeviceBuffer::alloc(r0.device.device_id, num_tokens * 4)?;
            let pos_dev = DeviceBuffer::alloc(r0.device.device_id, num_tokens * 4)?;
            let q_start_dev = DeviceBuffer::alloc(r0.device.device_id, 8)?;
            let ctx_dev = DeviceBuffer::alloc(r0.device.device_id, 4)?;
            let seq_dev = DeviceBuffer::alloc(r0.device.device_id, 4)?;
            let blk_dev = DeviceBuffer::alloc(r0.device.device_id, block_table.len() * 4)?;
            let slot_dev = DeviceBuffer::alloc(r0.device.device_id, num_tokens * 4)?;
            let st_in_dev = DeviceBuffer::alloc(r0.device.device_id, 4)?;
            let st_out_dev = DeviceBuffer::alloc(r0.device.device_id, 4)?;
            let has_init_dev = DeviceBuffer::alloc(r0.device.device_id, 4)?;

            tok_dev.copy_from_host(tokens, &r0.stream)?;
            pos_dev.copy_from_host(&positions, &r0.stream)?;
            q_start_dev.copy_from_host(&q_start, &r0.stream)?;
            ctx_dev.copy_from_host(&ctx_arr, &r0.stream)?;
            seq_dev.copy_from_host(&seq_arr, &r0.stream)?;
            blk_dev.copy_from_host(block_table, &r0.stream)?;
            slot_dev.copy_from_host(&write_slots, &r0.stream)?;
            st_in_dev.copy_from_host(&st_in_arr, &r0.stream)?;
            st_out_dev.copy_from_host(&st_out_arr, &r0.stream)?;
            has_init_dev.copy_from_host(&has_init_arr, &r0.stream)?;

            let attn_batch = W34AttentionBatch {
                query_start_loc: q_start_dev.addr() as *const i32,
                context_lens_before: ctx_dev.addr() as *const i32,
                seq_lens_after: seq_dev.addr() as *const i32,
                block_tables: blk_dev.addr() as *const i32,
                write_slot_mapping: slot_dev.addr() as *const i32,
                valid_token_mask: core::ptr::null(),
                valid_req_mask: core::ptr::null(),
                num_reqs: 1,
                total_tokens: num_tokens as i32,
                max_blocks_per_req: block_table.len() as i32,
                max_q_len: num_tokens as i32,
                max_seq_len: seq_after as i32,
                mask_mode: W34_ATTN_MASK_CAUSAL,
                sliding_window: 0,
                sm_scale: (r0.cfg.head_dim as f32).powf(-0.5),
                reserved: 0,
            };
            let gdn_batch = W34GdnBatch {
                query_start_loc: q_start_dev.addr() as *const i32,
                state_in_indices: st_in_dev.addr() as *const i32,
                state_out_indices: st_out_dev.addr() as *const i32,
                spec_state_indices: core::ptr::null(),
                checkpoint_dst_indices: core::ptr::null(),
                checkpoint_token_offsets: core::ptr::null(),
                has_initial_state: has_init_dev.addr() as *const u8,
                valid_req_mask: core::ptr::null(),
                num_reqs: 1,
                total_tokens: num_tokens as i32,
                max_verify_tokens: 0,
                max_checkpoints_per_req: 0,
            };

            let out_id_ptr = r0.decode_io.out_greedy_id.addr();
            let top5_id_ptr = r0.decode_io.out_top5_ids.addr();
            let top5_lp_ptr = r0.decode_io.out_top5_logprobs.addr();
            r0.forward_device_batches(
                tok_dev.addr(),
                pos_dev.addr(),
                num_tokens,
                &attn_batch,
                &gdn_batch,
                None,
                out_id_ptr,
                top5_id_ptr,
                top5_lp_ptr,
                dump_out.as_deref_mut(),
            )?;
            let mut greedy_id = [0i32; 1];
            let mut top5_ids = [0i32; 5];
            let mut top5_lps = [0.0f32; 5];
            r0.decode_io
                .out_greedy_id
                .copy_to_host(&mut greedy_id, &r0.stream)?;
            r0.decode_io
                .out_top5_ids
                .copy_to_host(&mut top5_ids, &r0.stream)?;
            r0.decode_io
                .out_top5_logprobs
                .copy_to_host(&mut top5_lps, &r0.stream)?;
            r0.stream.synchronize()?;
            if let Some(d) = dump_out {
                d.greedy_token_id = greedy_id[0];
                d.top5_ids = top5_ids;
                d.top5_logprobs = top5_lps;
            }
            Ok((greedy_id[0], top5_ids, top5_lps))
        } else {
            let comm = self.tp2_comm.as_ref().unwrap().clone();
            let [r0, r1] = &mut self.ranks[..] else {
                unreachable!()
            };
            let mut dump0 = if dump_out.is_some() {
                Some(ForwardDump::default())
            } else {
                None
            };
            let pos_slice = positions.as_slice();
            let slots_slice = write_slots.as_slice();
            let (res0, res1) = std::thread::scope(|sc| {
                let c0 = comm.clone();
                let d0_ref = &mut dump0;
                let h0 = sc.spawn(move || -> Result<(i32, f32, [i32; 5], [f32; 5])> {
                    r0.device.make_current()?;
                    let tok_dev = DeviceBuffer::alloc(r0.device.device_id, num_tokens * 4)?;
                    let pos_dev = DeviceBuffer::alloc(r0.device.device_id, num_tokens * 4)?;
                    let q_start_dev = DeviceBuffer::alloc(r0.device.device_id, 8)?;
                    let ctx_dev = DeviceBuffer::alloc(r0.device.device_id, 4)?;
                    let seq_dev = DeviceBuffer::alloc(r0.device.device_id, 4)?;
                    let blk_dev = DeviceBuffer::alloc(r0.device.device_id, block_table.len() * 4)?;
                    let slot_dev = DeviceBuffer::alloc(r0.device.device_id, num_tokens * 4)?;
                    let st_in_dev = DeviceBuffer::alloc(r0.device.device_id, 4)?;
                    let st_out_dev = DeviceBuffer::alloc(r0.device.device_id, 4)?;
                    let has_init_dev = DeviceBuffer::alloc(r0.device.device_id, 4)?;

                    tok_dev.copy_from_host(tokens, &r0.stream)?;
                    pos_dev.copy_from_host(pos_slice, &r0.stream)?;
                    q_start_dev.copy_from_host(&q_start, &r0.stream)?;
                    ctx_dev.copy_from_host(&ctx_arr, &r0.stream)?;
                    seq_dev.copy_from_host(&seq_arr, &r0.stream)?;
                    blk_dev.copy_from_host(block_table, &r0.stream)?;
                    slot_dev.copy_from_host(slots_slice, &r0.stream)?;
                    st_in_dev.copy_from_host(&st_in_arr, &r0.stream)?;
                    st_out_dev.copy_from_host(&st_out_arr, &r0.stream)?;
                    has_init_dev.copy_from_host(&has_init_arr, &r0.stream)?;

                    let attn_batch = W34AttentionBatch {
                        query_start_loc: q_start_dev.addr() as *const i32,
                        context_lens_before: ctx_dev.addr() as *const i32,
                        seq_lens_after: seq_dev.addr() as *const i32,
                        block_tables: blk_dev.addr() as *const i32,
                        write_slot_mapping: slot_dev.addr() as *const i32,
                        valid_token_mask: core::ptr::null(),
                        valid_req_mask: core::ptr::null(),
                        num_reqs: 1,
                        total_tokens: num_tokens as i32,
                        max_blocks_per_req: block_table.len() as i32,
                        max_q_len: num_tokens as i32,
                        max_seq_len: seq_after as i32,
                        mask_mode: W34_ATTN_MASK_CAUSAL,
                        sliding_window: 0,
                        sm_scale: (r0.cfg.head_dim as f32).powf(-0.5),
                        reserved: 0,
                    };
                    let gdn_batch = W34GdnBatch {
                        query_start_loc: q_start_dev.addr() as *const i32,
                        state_in_indices: st_in_dev.addr() as *const i32,
                        state_out_indices: st_out_dev.addr() as *const i32,
                        spec_state_indices: core::ptr::null(),
                        checkpoint_dst_indices: core::ptr::null(),
                        checkpoint_token_offsets: core::ptr::null(),
                        has_initial_state: has_init_dev.addr() as *const u8,
                        valid_req_mask: core::ptr::null(),
                        num_reqs: 1,
                        total_tokens: num_tokens as i32,
                        max_verify_tokens: 0,
                        max_checkpoints_per_req: 0,
                    };
                    let out_id_ptr = r0.decode_io.out_greedy_id.addr();
                    let top5_id_ptr = r0.decode_io.out_top5_ids.addr();
                    let top5_lp_ptr = r0.decode_io.out_top5_logprobs.addr();
                    r0.forward_device_batches(
                        tok_dev.addr(),
                        pos_dev.addr(),
                        num_tokens,
                        &attn_batch,
                        &gdn_batch,
                        Some(&c0),
                        out_id_ptr,
                        top5_id_ptr,
                        top5_lp_ptr,
                        d0_ref.as_mut(),
                    )?;
                    let mut greedy_id = [0i32; 1];
                    let mut local_max = [0.0f32; 1];
                    let mut top5_ids = [0i32; 5];
                    let mut top5_lps = [0.0f32; 5];
                    r0.decode_io
                        .out_greedy_id
                        .copy_to_host(&mut greedy_id, &r0.stream)?;
                    r0.decode_io
                        .local_max_val
                        .copy_to_host(&mut local_max, &r0.stream)?;
                    r0.decode_io
                        .out_top5_ids
                        .copy_to_host(&mut top5_ids, &r0.stream)?;
                    r0.decode_io
                        .out_top5_logprobs
                        .copy_to_host(&mut top5_lps, &r0.stream)?;
                    r0.stream.synchronize()?;
                    Ok((greedy_id[0], local_max[0], top5_ids, top5_lps))
                });

                let c1 = comm.clone();
                let h1 = sc.spawn(move || -> Result<(f32, [i32; 5], [f32; 5])> {
                    r1.device.make_current()?;
                    let tok_dev = DeviceBuffer::alloc(r1.device.device_id, num_tokens * 4)?;
                    let pos_dev = DeviceBuffer::alloc(r1.device.device_id, num_tokens * 4)?;
                    let q_start_dev = DeviceBuffer::alloc(r1.device.device_id, 8)?;
                    let ctx_dev = DeviceBuffer::alloc(r1.device.device_id, 4)?;
                    let seq_dev = DeviceBuffer::alloc(r1.device.device_id, 4)?;
                    let blk_dev = DeviceBuffer::alloc(r1.device.device_id, block_table.len() * 4)?;
                    let slot_dev = DeviceBuffer::alloc(r1.device.device_id, num_tokens * 4)?;
                    let st_in_dev = DeviceBuffer::alloc(r1.device.device_id, 4)?;
                    let st_out_dev = DeviceBuffer::alloc(r1.device.device_id, 4)?;
                    let has_init_dev = DeviceBuffer::alloc(r1.device.device_id, 4)?;

                    tok_dev.copy_from_host(tokens, &r1.stream)?;
                    pos_dev.copy_from_host(pos_slice, &r1.stream)?;
                    q_start_dev.copy_from_host(&q_start, &r1.stream)?;
                    ctx_dev.copy_from_host(&ctx_arr, &r1.stream)?;
                    seq_dev.copy_from_host(&seq_arr, &r1.stream)?;
                    blk_dev.copy_from_host(block_table, &r1.stream)?;
                    slot_dev.copy_from_host(slots_slice, &r1.stream)?;
                    st_in_dev.copy_from_host(&st_in_arr, &r1.stream)?;
                    st_out_dev.copy_from_host(&st_out_arr, &r1.stream)?;
                    has_init_dev.copy_from_host(&has_init_arr, &r1.stream)?;

                    let attn_batch = W34AttentionBatch {
                        query_start_loc: q_start_dev.addr() as *const i32,
                        context_lens_before: ctx_dev.addr() as *const i32,
                        seq_lens_after: seq_dev.addr() as *const i32,
                        block_tables: blk_dev.addr() as *const i32,
                        write_slot_mapping: slot_dev.addr() as *const i32,
                        valid_token_mask: core::ptr::null(),
                        valid_req_mask: core::ptr::null(),
                        num_reqs: 1,
                        total_tokens: num_tokens as i32,
                        max_blocks_per_req: block_table.len() as i32,
                        max_q_len: num_tokens as i32,
                        max_seq_len: seq_after as i32,
                        mask_mode: W34_ATTN_MASK_CAUSAL,
                        sliding_window: 0,
                        sm_scale: (r1.cfg.head_dim as f32).powf(-0.5),
                        reserved: 0,
                    };
                    let gdn_batch = W34GdnBatch {
                        query_start_loc: q_start_dev.addr() as *const i32,
                        state_in_indices: st_in_dev.addr() as *const i32,
                        state_out_indices: st_out_dev.addr() as *const i32,
                        spec_state_indices: core::ptr::null(),
                        checkpoint_dst_indices: core::ptr::null(),
                        checkpoint_token_offsets: core::ptr::null(),
                        has_initial_state: has_init_dev.addr() as *const u8,
                        valid_req_mask: core::ptr::null(),
                        num_reqs: 1,
                        total_tokens: num_tokens as i32,
                        max_verify_tokens: 0,
                        max_checkpoints_per_req: 0,
                    };
                    let out_id_ptr = r1.decode_io.out_greedy_id.addr();
                    let top5_id_ptr = r1.decode_io.out_top5_ids.addr();
                    let top5_lp_ptr = r1.decode_io.out_top5_logprobs.addr();
                    r1.forward_device_batches(
                        tok_dev.addr(),
                        pos_dev.addr(),
                        num_tokens,
                        &attn_batch,
                        &gdn_batch,
                        Some(&c1),
                        out_id_ptr,
                        top5_id_ptr,
                        top5_lp_ptr,
                        None,
                    )?;
                    let mut local_max = [0.0f32; 1];
                    let mut top5_ids = [0i32; 5];
                    let mut top5_lps = [0.0f32; 5];
                    r1.decode_io
                        .local_max_val
                        .copy_to_host(&mut local_max, &r1.stream)?;
                    r1.decode_io
                        .out_top5_ids
                        .copy_to_host(&mut top5_ids, &r1.stream)?;
                    r1.decode_io
                        .out_top5_logprobs
                        .copy_to_host(&mut top5_lps, &r1.stream)?;
                    r1.stream.synchronize()?;
                    Ok((local_max[0], top5_ids, top5_lps))
                });
                (h0.join().unwrap(), h1.join().unwrap())
            });
            let (gid, max0, ids0, lps0) = res0?;
            let (max1, ids1, lps1) = res1?;
            // Reconstruct exact global log-sum-exp across both shards and merge top-5
            let log_z0 = max0 - lps0[0];
            let log_z1 = max1 - lps1[0];
            let m_z = log_z0.max(log_z1);
            let log_z_global = m_z + ((log_z0 - m_z).exp() + (log_z1 - m_z).exp()).ln();
            let mut cands = [(0.0f32, 0i32); 10];
            for k in 0..5 {
                cands[k] = (lps0[k] + log_z0 - log_z_global, ids0[k]);
                cands[5 + k] = (lps1[k] + log_z1 - log_z_global, ids1[k]);
            }
            cands.sort_by(|a, b| b.0.total_cmp(&a.0).then_with(|| a.1.cmp(&b.1)));
            let mut top5_ids = [0i32; 5];
            let mut top5_lps = [0.0f32; 5];
            for k in 0..5 {
                top5_lps[k] = cands[k].0;
                top5_ids[k] = cands[k].1;
            }
            if let (Some(dst), Some(src)) = (dump_out, dump0) {
                *dst = src;
                dst.greedy_token_id = gid;
                dst.top5_ids = top5_ids;
                dst.top5_logprobs = top5_lps;
            }
            Ok((gid, top5_ids, top5_lps))
        }
    }

    /// Executes a single `M=1` decode step using `DecodeGraphIo` (and `CudaGraph` replay when enabled).
    pub fn decode_one_token(
        &mut self,
        token_id: i32,
        ctx_before: usize,
        block_table: &[i32],
        state_slot: i32,
    ) -> Result<i32> {
        let seq_after = ctx_before + 1;
        let max_pages = self.ranks[0].kv_pool.num_pages;
        if ctx_before / 16 >= block_table.len()
            || block_table.len() > self.ranks[0].decode_io.max_blocks_per_req
            || block_table
                .iter()
                .any(|&p| p < 0 || (p as usize) >= max_pages)
        {
            return Err(loader_err(
                "KV cache page capacity exceeded in decode_one_token",
            ));
        }
        let write_slot = block_table[ctx_before / 16] * 16 + ((ctx_before % 16) as i32);
        let tok_arr = [token_id];
        let pos_arr = [ctx_before as i32];
        let ctx_arr = [ctx_before as i32];
        let seq_arr = [seq_after as i32];
        let slot_arr = [write_slot];
        let st_arr = [state_slot];

        if self.tp_size == 1 {
            let r0 = &mut self.ranks[0];
            r0.device.make_current()?;
            r0.decode_io
                .token_ids
                .copy_from_host(&tok_arr, &r0.stream)?;
            r0.decode_io
                .positions
                .copy_from_host(&pos_arr, &r0.stream)?;
            r0.decode_io
                .ctx_lens_before
                .copy_from_host(&ctx_arr, &r0.stream)?;
            r0.decode_io
                .seq_lens_after
                .copy_from_host(&seq_arr, &r0.stream)?;
            r0.decode_io
                .block_tables
                .copy_from_host(block_table, &r0.stream)?;
            r0.decode_io
                .write_slot_mapping
                .copy_from_host(&slot_arr, &r0.stream)?;
            r0.decode_io
                .state_in_indices
                .copy_from_host(&st_arr, &r0.stream)?;
            r0.decode_io
                .state_out_indices
                .copy_from_host(&st_arr, &r0.stream)?;

            if self.use_cuda_graph {
                r0.ensure_decode_graph(None)?;
                r0.decode_graph.as_ref().unwrap().launch(&r0.stream)?;
            } else {
                let attn_batch = W34AttentionBatch {
                    query_start_loc: r0.decode_io.q_start_loc.addr() as *const i32,
                    context_lens_before: r0.decode_io.ctx_lens_before.addr() as *const i32,
                    seq_lens_after: r0.decode_io.seq_lens_after.addr() as *const i32,
                    block_tables: r0.decode_io.block_tables.addr() as *const i32,
                    write_slot_mapping: r0.decode_io.write_slot_mapping.addr() as *const i32,
                    valid_token_mask: core::ptr::null(),
                    valid_req_mask: core::ptr::null(),
                    num_reqs: 1,
                    total_tokens: 1,
                    max_blocks_per_req: r0.decode_io.max_blocks_per_req as i32,
                    max_q_len: 1,
                    max_seq_len: seq_after as i32,
                    mask_mode: W34_ATTN_MASK_CAUSAL,
                    sliding_window: 0,
                    sm_scale: (r0.cfg.head_dim as f32).powf(-0.5),
                    reserved: 0,
                };
                let gdn_batch = W34GdnBatch {
                    query_start_loc: r0.decode_io.q_start_loc.addr() as *const i32,
                    state_in_indices: r0.decode_io.state_in_indices.addr() as *const i32,
                    state_out_indices: r0.decode_io.state_out_indices.addr() as *const i32,
                    spec_state_indices: core::ptr::null(),
                    checkpoint_dst_indices: core::ptr::null(),
                    checkpoint_token_offsets: core::ptr::null(),
                    has_initial_state: r0.decode_io.has_initial_state.addr() as *const u8,
                    valid_req_mask: core::ptr::null(),
                    num_reqs: 1,
                    total_tokens: 1,
                    max_verify_tokens: 0,
                    max_checkpoints_per_req: 0,
                };
                let tok_ptr = r0.decode_io.token_ids.addr();
                let pos_ptr = r0.decode_io.positions.addr();
                let out_id_ptr = r0.decode_io.out_greedy_id.addr();
                r0.forward_device_batches(
                    tok_ptr,
                    pos_ptr,
                    1,
                    &attn_batch,
                    &gdn_batch,
                    None,
                    out_id_ptr,
                    0,
                    0,
                    None,
                )?;
            }
            let mut gid = [0i32; 1];
            r0.decode_io
                .out_greedy_id
                .copy_to_host(&mut gid, &r0.stream)?;
            r0.stream.synchronize()?;
            Ok(gid[0])
        } else {
            let comm = self.tp2_comm.as_ref().unwrap().clone();
            let use_graph = self.use_cuda_graph;
            let [r0, r1] = &mut self.ranks[..] else {
                unreachable!()
            };
            let (res0, res1) = std::thread::scope(|sc| {
                let c0 = comm.clone();
                let h0 = sc.spawn(move || -> Result<i32> {
                    r0.device.make_current()?;
                    r0.decode_io
                        .token_ids
                        .copy_from_host(&tok_arr, &r0.stream)?;
                    r0.decode_io
                        .positions
                        .copy_from_host(&pos_arr, &r0.stream)?;
                    r0.decode_io
                        .ctx_lens_before
                        .copy_from_host(&ctx_arr, &r0.stream)?;
                    r0.decode_io
                        .seq_lens_after
                        .copy_from_host(&seq_arr, &r0.stream)?;
                    r0.decode_io
                        .block_tables
                        .copy_from_host(block_table, &r0.stream)?;
                    r0.decode_io
                        .write_slot_mapping
                        .copy_from_host(&slot_arr, &r0.stream)?;
                    r0.decode_io
                        .state_in_indices
                        .copy_from_host(&st_arr, &r0.stream)?;
                    r0.decode_io
                        .state_out_indices
                        .copy_from_host(&st_arr, &r0.stream)?;
                    if use_graph {
                        r0.ensure_decode_graph(Some(&c0))?;
                        r0.decode_graph.as_ref().unwrap().launch(&r0.stream)?;
                    } else {
                        let attn_batch = W34AttentionBatch {
                            query_start_loc: r0.decode_io.q_start_loc.addr() as *const i32,
                            context_lens_before: r0.decode_io.ctx_lens_before.addr() as *const i32,
                            seq_lens_after: r0.decode_io.seq_lens_after.addr() as *const i32,
                            block_tables: r0.decode_io.block_tables.addr() as *const i32,
                            write_slot_mapping: r0.decode_io.write_slot_mapping.addr()
                                as *const i32,
                            valid_token_mask: core::ptr::null(),
                            valid_req_mask: core::ptr::null(),
                            num_reqs: 1,
                            total_tokens: 1,
                            max_blocks_per_req: r0.decode_io.max_blocks_per_req as i32,
                            max_q_len: 1,
                            max_seq_len: seq_after as i32,
                            mask_mode: W34_ATTN_MASK_CAUSAL,
                            sliding_window: 0,
                            sm_scale: (r0.cfg.head_dim as f32).powf(-0.5),
                            reserved: 0,
                        };
                        let gdn_batch = W34GdnBatch {
                            query_start_loc: r0.decode_io.q_start_loc.addr() as *const i32,
                            state_in_indices: r0.decode_io.state_in_indices.addr() as *const i32,
                            state_out_indices: r0.decode_io.state_out_indices.addr() as *const i32,
                            spec_state_indices: core::ptr::null(),
                            checkpoint_dst_indices: core::ptr::null(),
                            checkpoint_token_offsets: core::ptr::null(),
                            has_initial_state: r0.decode_io.has_initial_state.addr() as *const u8,
                            valid_req_mask: core::ptr::null(),
                            num_reqs: 1,
                            total_tokens: 1,
                            max_verify_tokens: 0,
                            max_checkpoints_per_req: 0,
                        };
                        let tok_ptr = r0.decode_io.token_ids.addr();
                        let pos_ptr = r0.decode_io.positions.addr();
                        let out_id_ptr = r0.decode_io.out_greedy_id.addr();
                        r0.forward_device_batches(
                            tok_ptr,
                            pos_ptr,
                            1,
                            &attn_batch,
                            &gdn_batch,
                            Some(&c0),
                            out_id_ptr,
                            0,
                            0,
                            None,
                        )?;
                    }
                    let mut gid = [0i32; 1];
                    r0.decode_io
                        .out_greedy_id
                        .copy_to_host(&mut gid, &r0.stream)?;
                    r0.stream.synchronize()?;
                    Ok(gid[0])
                });

                let c1 = comm.clone();
                let h1 = sc.spawn(move || -> Result<()> {
                    r1.device.make_current()?;
                    r1.decode_io
                        .token_ids
                        .copy_from_host(&tok_arr, &r1.stream)?;
                    r1.decode_io
                        .positions
                        .copy_from_host(&pos_arr, &r1.stream)?;
                    r1.decode_io
                        .ctx_lens_before
                        .copy_from_host(&ctx_arr, &r1.stream)?;
                    r1.decode_io
                        .seq_lens_after
                        .copy_from_host(&seq_arr, &r1.stream)?;
                    r1.decode_io
                        .block_tables
                        .copy_from_host(block_table, &r1.stream)?;
                    r1.decode_io
                        .write_slot_mapping
                        .copy_from_host(&slot_arr, &r1.stream)?;
                    r1.decode_io
                        .state_in_indices
                        .copy_from_host(&st_arr, &r1.stream)?;
                    r1.decode_io
                        .state_out_indices
                        .copy_from_host(&st_arr, &r1.stream)?;
                    if use_graph {
                        r1.ensure_decode_graph(Some(&c1))?;
                        r1.decode_graph.as_ref().unwrap().launch(&r1.stream)?;
                    } else {
                        let attn_batch = W34AttentionBatch {
                            query_start_loc: r1.decode_io.q_start_loc.addr() as *const i32,
                            context_lens_before: r1.decode_io.ctx_lens_before.addr() as *const i32,
                            seq_lens_after: r1.decode_io.seq_lens_after.addr() as *const i32,
                            block_tables: r1.decode_io.block_tables.addr() as *const i32,
                            write_slot_mapping: r1.decode_io.write_slot_mapping.addr()
                                as *const i32,
                            valid_token_mask: core::ptr::null(),
                            valid_req_mask: core::ptr::null(),
                            num_reqs: 1,
                            total_tokens: 1,
                            max_blocks_per_req: r1.decode_io.max_blocks_per_req as i32,
                            max_q_len: 1,
                            max_seq_len: seq_after as i32,
                            mask_mode: W34_ATTN_MASK_CAUSAL,
                            sliding_window: 0,
                            sm_scale: (r1.cfg.head_dim as f32).powf(-0.5),
                            reserved: 0,
                        };
                        let gdn_batch = W34GdnBatch {
                            query_start_loc: r1.decode_io.q_start_loc.addr() as *const i32,
                            state_in_indices: r1.decode_io.state_in_indices.addr() as *const i32,
                            state_out_indices: r1.decode_io.state_out_indices.addr() as *const i32,
                            spec_state_indices: core::ptr::null(),
                            checkpoint_dst_indices: core::ptr::null(),
                            checkpoint_token_offsets: core::ptr::null(),
                            has_initial_state: r1.decode_io.has_initial_state.addr() as *const u8,
                            valid_req_mask: core::ptr::null(),
                            num_reqs: 1,
                            total_tokens: 1,
                            max_verify_tokens: 0,
                            max_checkpoints_per_req: 0,
                        };
                        let tok_ptr = r1.decode_io.token_ids.addr();
                        let pos_ptr = r1.decode_io.positions.addr();
                        let out_id_ptr = r1.decode_io.out_greedy_id.addr();
                        r1.forward_device_batches(
                            tok_ptr,
                            pos_ptr,
                            1,
                            &attn_batch,
                            &gdn_batch,
                            Some(&c1),
                            out_id_ptr,
                            0,
                            0,
                            None,
                        )?;
                    }
                    r1.stream.synchronize()?;
                    Ok(())
                });
                (h0.join().unwrap(), h1.join().unwrap())
            });
            res1?;
            res0
        }
    }

    /// Generates `max_new_tokens` greedily from `prompt_token_ids`, returning `(generated_ids, BenchMetrics)`.
    pub fn generate_tokens(
        &mut self,
        prompt_token_ids: &[i32],
        max_new_tokens: usize,
        stop_on_eos: bool,
    ) -> Result<BenchMetrics> {
        self.reset_all_states()?;
        let total_needed_tokens = prompt_token_ids.len() + max_new_tokens + 4;
        let needed_pages = total_needed_tokens.div_ceil(16);
        let max_pages = self.ranks[0]
            .kv_pool
            .num_pages
            .min(self.ranks[0].decode_io.max_blocks_per_req);
        if needed_pages > max_pages {
            return Err(loader_err(
                "prompt + max_new_tokens exceeds allocated KV page capacity",
            ));
        }
        let mut block_table = Vec::with_capacity(needed_pages);
        for p in 0..needed_pages {
            block_table.push(p as i32);
        }
        let mut tx = StateTx::new(0);
        tx.begin_prefill();

        let t0 = std::time::Instant::now();
        let (first_tok, _, _) = self.step_tokens(
            prompt_token_ids,
            0,
            &block_table,
            tx.committed_slot,
            tx.committed_slot,
            false,
            None,
        )?;
        let ttft_ms = t0.elapsed().as_secs_f64() * 1000.0;
        tx.commit_prefill(prompt_token_ids.len());

        let mut gen_ids = Vec::with_capacity(max_new_tokens);
        if max_new_tokens > 0 {
            gen_ids.push(first_tok);
        }

        let t_dec_start = std::time::Instant::now();
        let mut cur_tok = first_tok;
        for (ctx_len, _) in (prompt_token_ids.len()..).zip(1..max_new_tokens) {
            if stop_on_eos && self.cfg.eos_token_ids.contains(&cur_tok) {
                break;
            }
            let next_tok =
                self.decode_one_token(cur_tok, ctx_len, &block_table, tx.committed_slot)?;
            gen_ids.push(next_tok);
            cur_tok = next_tok;
        }
        let decode_ms = t_dec_start.elapsed().as_secs_f64() * 1000.0;
        let decode_steps = gen_ids.len().saturating_sub(1);
        let decode_tok_per_s = if decode_ms > 0.0 && decode_steps > 0 {
            (decode_steps as f64) / (decode_ms / 1000.0)
        } else {
            0.0
        };
        let itl_ms = if decode_steps > 0 {
            decode_ms / (decode_steps as f64)
        } else {
            0.0
        };
        let prefill_tok_per_s = if ttft_ms > 0.0 {
            (prompt_token_ids.len() as f64) / (ttft_ms / 1000.0)
        } else {
            0.0
        };

        Ok(BenchMetrics {
            prompt_len: prompt_token_ids.len(),
            gen_len: gen_ids.len(),
            ttft_ms,
            prefill_tok_per_s,
            decode_ms,
            decode_tok_per_s,
            itl_ms,
            generated_token_ids: gen_ids,
        })
    }

    /// Runs a single diagnostic prefill pass and returns [`ForwardDump`] for `dump-diff`.
    pub fn dump_prompt(&mut self, prompt_token_ids: &[i32]) -> Result<ForwardDump> {
        self.reset_all_states()?;
        let needed_pages = (prompt_token_ids.len() + 4).div_ceil(16);
        if needed_pages > self.ranks[0].kv_pool.num_pages {
            return Err(loader_err("prompt exceeds allocated KV page capacity"));
        }
        let block_table: Vec<i32> = (0..needed_pages as i32).collect();
        let mut dump = ForwardDump::default();
        self.step_tokens(
            prompt_token_ids,
            0,
            &block_table,
            0,
            0,
            false,
            Some(&mut dump),
        )?;
        Ok(dump)
    }
}
