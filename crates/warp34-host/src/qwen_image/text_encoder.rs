//! `Qwen3-VL-8B` conditioning encoder (`text_encoder/` in `QwenImage21Pipeline`):
//! - ViT (`model.visual.*`) with `deepstack_visual_indexes = [8, 16, 24]` and `deepstack_merger_list.{0..2}`
//! - Standard `RMSNorm` (`weight`, not `1 + weight`) across `input_layernorm`, `post_attention_layernorm`, `q_norm`, `k_norm`
//! - Ungated `head_dim = 128` causal GQA (`32` Q heads, `8` KV heads) with full-rotary `mRoPE = [24, 20, 20]` (`theta = 5e6`)
//! - DeepStack visual feature injection (`_deepstack_process`) after decoder layers `0..len(deepstack_visual_indexes)`
//! - Pre-final-norm hidden tap after layer `num_hidden_layers - 1` + `_drop_idx` slicing (`DEFAULT_DROP_IDX = 14`)

use std::ffi::c_void;
use std::sync::{Arc, Mutex};

use crate::cuda::{
    CublasHandle, CudaDevice, CudaError, CudaStream, DeviceArena, DeviceBuffer, Result, check_rc,
    f16_to_f32_slice, f32_to_f16_slice, prepare_execution_plan,
};
use crate::loader::{ModelConfig, QuantScheme, SafetensorsArchive};
use crate::model::{LinearProj, MARLIN_TILE_K_W4, MARLIN_TILE_N, RawLinearHost};
use crate::vision::{PreprocessedImage, Qwen3VisionEncoder, compute_mrope_positions};
use warp34_kernels::ffi;

/// Encoded conditioning output from `Qwen3VlConditioningEncoder::encode_conditioning`.
#[derive(Clone, Debug)]
pub struct Qwen3VlConditioningOutput {
    /// Pre-final-norm hidden states `[seq_len, hidden_size]` after dropping `drop_idx` prefix tokens (`FP16` bits).
    pub prompt_embeds_f16: Vec<u16>,
    /// Pre-final-norm hidden states `[seq_len, hidden_size]` widened to `FP32`.
    pub prompt_embeds_f32: Vec<f32>,
    /// Boolean mask `[seq_len]` marking `<|image_pad|>` positions in the kept sequence.
    pub image_pad_mask: Vec<bool>,
    /// Kept token count (`input_ids.len() - drop_idx`).
    pub seq_len: usize,
    /// Model hidden dimension (`4096` on `Qwen3-VL-8B`).
    pub hidden_size: usize,
}

struct Qwen3VlDecoderLayerGpu {
    input_norm_w: u64,
    q_proj: LinearProj,
    k_proj: LinearProj,
    v_proj: LinearProj,
    q_norm_w: u64,
    k_norm_w: u64,
    o_proj: LinearProj,
    post_attn_norm_w: u64,
    gate_up_proj: LinearProj,
    down_proj: LinearProj,
}

#[derive(Default)]
struct Qwen3VlWorkspace {
    marlin_ws: Option<DeviceBuffer>,
    exec_scratch: Option<DeviceBuffer>,
    d_residual: Option<DeviceBuffer>,
    d_norm: Option<DeviceBuffer>,
    d_q: Option<DeviceBuffer>,
    d_k: Option<DeviceBuffer>,
    d_v: Option<DeviceBuffer>,
    d_attn_out: Option<DeviceBuffer>,
    d_proj_out: Option<DeviceBuffer>,
    d_gate_up: Option<DeviceBuffer>,
    d_mlp_interm: Option<DeviceBuffer>,
    d_positions: Option<DeviceBuffer>,
    d_vis_indices: Option<DeviceBuffer>,
}

impl Qwen3VlWorkspace {
    fn ensure_buf(
        slot: &mut Option<DeviceBuffer>,
        device_id: i32,
        min_bytes: usize,
    ) -> Result<u64> {
        let need = min_bytes.max(16);
        if slot.as_ref().is_none_or(|b| b.len_bytes < need) {
            *slot = Some(DeviceBuffer::alloc(device_id, need)?);
        }
        Ok(slot.as_ref().unwrap().addr())
    }
}

/// GPU conditioning encoder for `Qwen/Qwen-Image-2.1` (`Qwen3-VL-8B`).
pub struct Qwen3VlConditioningEncoder {
    pub cfg: ModelConfig,
    pub device_id: i32,
    pub archive: Arc<SafetensorsArchive>,
    embed_key: String,
    pub vision: Option<Qwen3VisionEncoder>,
    pub arena: DeviceArena,
    layers: Vec<Qwen3VlDecoderLayerGpu>,
    workspace: Mutex<Qwen3VlWorkspace>,
}

impl Qwen3VlConditioningEncoder {
    /// Loads `Qwen3-VL` vision + language layers onto `device_id`.
    ///
    /// Token embeddings (`model.language_model.embed_tokens.weight`) are read on-demand per prompt
    /// row via `pread` and `lm_head.weight` is omitted because `QwenImage21Pipeline` taps the
    /// hidden states before the final language-model norm, saving ~2.46 GiB of GPU VRAM.
    pub fn load(
        archive: Arc<SafetensorsArchive>,
        cfg: &ModelConfig,
        quant: &QuantScheme,
        device_id: i32,
        stream: &CudaStream,
    ) -> Result<Self> {
        prepare_execution_plan(device_id)?;
        let _dev = CudaDevice::set_current(device_id)?;
        let vision = if let Some(ref vcfg) = cfg.vision_config {
            Qwen3VisionEncoder::try_load(&archive, vcfg, quant, device_id, stream)?
        } else {
            None
        };

        let embed_key = archive
            .resolve_name("model.embed_tokens.weight")
            .ok_or_else(|| CudaError::new(-10, "missing model.embed_tokens.weight"))?;

        let hidden = cfg.hidden_size;
        let interm = cfg.intermediate_size;
        let head_dim = cfg.head_dim;
        let q_dim = cfg.num_attention_heads * head_dim;
        let kv_dim = cfg.num_key_value_heads * head_dim;

        // Estimate arena size for language model layers only (excluding embed_tokens, lm_head, and visual).
        let raw_layer_bytes: usize = archive
            .iter_entries()
            .filter(|(name, _)| name.contains("layers."))
            .map(|(_, info)| info.len_bytes as usize)
            .sum();
        let per_layer_f16 = q_dim
            .saturating_mul(hidden)
            .saturating_add(2usize.saturating_mul(kv_dim).saturating_mul(hidden))
            .saturating_add(hidden.saturating_mul(q_dim))
            .saturating_add(3usize.saturating_mul(interm).saturating_mul(hidden))
            .saturating_add(2usize.saturating_mul(hidden))
            .saturating_add(2usize.saturating_mul(head_dim))
            .saturating_mul(2);
        let dense_bytes = cfg.num_hidden_layers.saturating_mul(per_layer_f16);
        let base_bytes = if *quant == QuantScheme::Unquantized || raw_layer_bytes == 0 {
            dense_bytes
        } else {
            raw_layer_bytes
                .saturating_add(raw_layer_bytes / 4)
                .min(dense_bytes)
        };
        let est_bytes = base_bytes.saturating_add(64 * 1024 * 1024);
        let mut arena = DeviceArena::new(device_id, est_bytes)?;

        let upload_f16 =
            |arena: &mut DeviceArena, name: &str, expected_len: usize| -> Result<u64> {
                let key = archive
                    .resolve_name(name)
                    .ok_or_else(|| CudaError::with_detail(-10, "missing Qwen3-VL tensor", name))?;
                let bits = archive.read_f16_bits(&key).map_err(|e| {
                    CudaError::with_detail(-10, "read Qwen3-VL tensor", e.to_string())
                })?;
                if bits.len() != expected_len {
                    return Err(CudaError::with_detail(
                        -10,
                        "Qwen3-VL tensor length mismatch",
                        format!("{name}: expected {expected_len}, got {}", bits.len()),
                    ));
                }
                arena.upload_slice(&bits, stream)
            };

        let mut repack_staging: Option<DeviceBuffer> = None;
        let upload_raw = |arena: &mut DeviceArena,
                          staging: &mut Option<DeviceBuffer>,
                          raw: &RawLinearHost|
         -> Result<LinearProj> {
            if let RawLinearHost::DenseF16 { weight, n, k } = raw {
                return Ok(LinearProj::DenseF16 {
                    weight_ptr: arena.upload_slice(weight, stream)?,
                    size_n: *n,
                    size_k: *k,
                });
            }
            let (n, k) = raw.dims();
            let need = n
                .div_ceil(MARLIN_TILE_N)
                .saturating_mul(MARLIN_TILE_N)
                .saturating_mul(
                    k.div_ceil(MARLIN_TILE_K_W4)
                        .saturating_mul(MARLIN_TILE_K_W4),
                );
            if need > 0 && staging.as_ref().is_none_or(|b| b.len_bytes < need) {
                *staging = Some(DeviceBuffer::alloc(device_id, need)?);
            }
            raw.upload(arena, stream, staging.as_ref().unwrap())
        };

        let load_proj = |arena: &mut DeviceArena,
                         staging: &mut Option<DeviceBuffer>,
                         prefix: &str,
                         expected_n: usize,
                         expected_k: usize|
         -> Result<LinearProj> {
            let raw = RawLinearHost::load(&archive, prefix, quant)?;
            if raw.dims() != (expected_n, expected_k) {
                return Err(CudaError::with_detail(
                    -10,
                    "Qwen3-VL linear shape mismatch",
                    format!(
                        "{prefix}: expected ({expected_n}, {expected_k}), got {:?}",
                        raw.dims()
                    ),
                ));
            }
            upload_raw(arena, staging, &raw)
        };

        let mut layers = Vec::with_capacity(cfg.num_hidden_layers);
        for i in 0..cfg.num_hidden_layers {
            let lp = format!("model.layers.{i}");
            let input_norm_w =
                upload_f16(&mut arena, &format!("{lp}.input_layernorm.weight"), hidden)?;
            let q_proj = load_proj(
                &mut arena,
                &mut repack_staging,
                &format!("{lp}.self_attn.q_proj"),
                q_dim,
                hidden,
            )?;
            let k_proj = load_proj(
                &mut arena,
                &mut repack_staging,
                &format!("{lp}.self_attn.k_proj"),
                kv_dim,
                hidden,
            )?;
            let v_proj = load_proj(
                &mut arena,
                &mut repack_staging,
                &format!("{lp}.self_attn.v_proj"),
                kv_dim,
                hidden,
            )?;
            let q_norm_w = upload_f16(
                &mut arena,
                &format!("{lp}.self_attn.q_norm.weight"),
                head_dim,
            )?;
            let k_norm_w = upload_f16(
                &mut arena,
                &format!("{lp}.self_attn.k_norm.weight"),
                head_dim,
            )?;
            let o_proj = load_proj(
                &mut arena,
                &mut repack_staging,
                &format!("{lp}.self_attn.o_proj"),
                hidden,
                q_dim,
            )?;
            let post_attn_norm_w = upload_f16(
                &mut arena,
                &format!("{lp}.post_attention_layernorm.weight"),
                hidden,
            )?;

            let gate_raw = RawLinearHost::load(&archive, &format!("{lp}.mlp.gate_proj"), quant)?;
            let up_raw = RawLinearHost::load(&archive, &format!("{lp}.mlp.up_proj"), quant)?;
            if gate_raw.dims() != (interm, hidden) || up_raw.dims() != (interm, hidden) {
                return Err(CudaError::with_detail(
                    -10,
                    "Qwen3-VL MLP gate/up shape mismatch",
                    lp,
                ));
            }
            let gate_up_raw = RawLinearHost::concat_rows(&[gate_raw, up_raw])?;
            let gate_up_proj = upload_raw(&mut arena, &mut repack_staging, &gate_up_raw)?;
            let down_proj = load_proj(
                &mut arena,
                &mut repack_staging,
                &format!("{lp}.mlp.down_proj"),
                hidden,
                interm,
            )?;

            layers.push(Qwen3VlDecoderLayerGpu {
                input_norm_w,
                q_proj,
                k_proj,
                v_proj,
                q_norm_w,
                k_norm_w,
                o_proj,
                post_attn_norm_w,
                gate_up_proj,
                down_proj,
            });
        }

        stream.synchronize()?;
        Ok(Self {
            cfg: cfg.clone(),
            device_id,
            archive,
            embed_key,
            vision,
            arena,
            layers,
            workspace: Mutex::new(Qwen3VlWorkspace::default()),
        })
    }

    /// Runs the full `Qwen3-VL` conditioning forward pass on `input_ids` and optional `images`,
    /// tapping the pre-final-norm hidden states after the last decoder layer and dropping the
    /// first `drop_idx` system-template tokens.
    pub fn encode_conditioning(
        &self,
        input_ids: &[i32],
        images: &[PreprocessedImage],
        drop_idx: usize,
        cublas: &CublasHandle,
        stream: &CudaStream,
    ) -> Result<Qwen3VlConditioningOutput> {
        let _dev = CudaDevice::set_current(self.device_id)?;
        let seq_len = input_ids.len();
        let hidden = self.cfg.hidden_size;
        if seq_len == 0 {
            return Ok(Qwen3VlConditioningOutput {
                prompt_embeds_f16: Vec::new(),
                prompt_embeds_f32: Vec::new(),
                image_pad_mask: Vec::new(),
                seq_len: 0,
                hidden_size: hidden,
            });
        }

        let interm = self.cfg.intermediate_size;
        let num_q_heads = self.cfg.num_attention_heads;
        let num_kv_heads = self.cfg.num_key_value_heads;
        let head_dim = self.cfg.head_dim;
        let q_dim = num_q_heads * head_dim;
        let kv_dim = num_kv_heads * head_dim;
        let rotary_dim = self.cfg.rotary_dim;
        if head_dim != 128 {
            return Err(CudaError::with_detail(
                -10,
                "Qwen3VlConditioningEncoder requires head_dim == 128",
                head_dim.to_string(),
            ));
        }

        let image_token_id = self
            .cfg
            .vision_config
            .as_ref()
            .map(|v| v.image_token_id)
            .unwrap_or(crate::qwen_image::IMAGE_PAD_TOKEN_ID);
        let spatial_merge_size = self
            .cfg
            .vision_config
            .as_ref()
            .map(|v| v.spatial_merge_size)
            .unwrap_or(2);

        // 1. Optional ViT + DeepStack encoding
        let (vis_pooler, deepstack_embeds, vis_token_indices, image_grids) = if !images.is_empty() {
            let vision = self.vision.as_ref().ok_or_else(|| {
                CudaError::new(
                    -10,
                    "images provided to Qwen3VlConditioningEncoder without vision encoder",
                )
            })?;
            let (pooler, deepstack) =
                vision.encode_images_with_deepstack(images, cublas, stream)?;
            let expected_vis_tokens: usize = images
                .iter()
                .map(|img| img.num_merged_tokens(spatial_merge_size))
                .sum();
            let indices: Vec<i32> = input_ids
                .iter()
                .enumerate()
                .filter_map(|(idx, &tok)| (tok == image_token_id).then_some(idx as i32))
                .collect();
            if indices.len() != expected_vis_tokens {
                return Err(CudaError::with_detail(
                    -10,
                    "visual placeholder token count mismatch",
                    format!(
                        "input_ids has {} <|image_pad|> tokens, but images produced {expected_vis_tokens} merged tokens",
                        indices.len()
                    ),
                ));
            }
            let grids: Vec<[usize; 3]> = images.iter().map(|img| img.grid_thw).collect();
            (Some(pooler), deepstack, indices, grids)
        } else {
            (None, Vec::new(), Vec::new(), Vec::new())
        };

        // 2. Host pread token embedding lookup -> FP16 upload
        let mut embed_f16 = vec![0u16; seq_len * hidden];
        for (row_idx, &tok) in input_ids.iter().enumerate() {
            let vocab_row = usize::try_from(tok)
                .map_err(|_| CudaError::with_detail(-10, "negative token id", tok.to_string()))?;
            let row_f32 = self
                .archive
                .read_row_f32(&self.embed_key, vocab_row)
                .map_err(|e| CudaError::with_detail(-10, "embed_tokens lookup", e.to_string()))?;
            f32_to_f16_slice(
                &row_f32,
                &mut embed_f16[row_idx * hidden..(row_idx + 1) * hidden],
            );
        }

        // 3. Compute 3D mRoPE positions [3, seq_len]
        let (planes, _delta) =
            compute_mrope_positions(input_ids, &image_grids, image_token_id, spatial_merge_size);
        let mut positions_flat = Vec::with_capacity(3 * seq_len);
        for plane in &planes {
            positions_flat.extend_from_slice(plane);
        }

        let marlin_ws_ints = CudaDevice::props_for(self.device_id)?
            .multi_processor_count
            .max(1) as usize;
        let lse_scratch_bytes = seq_len * num_q_heads * std::mem::size_of::<f32>();
        let mut ws = self.workspace.lock().unwrap_or_else(|e| e.into_inner());
        let marlin_ws_addr = {
            let need = marlin_ws_ints * 4;
            let init_zero = ws.marlin_ws.as_ref().is_none_or(|b| b.len_bytes < need);
            let addr = Qwen3VlWorkspace::ensure_buf(&mut ws.marlin_ws, self.device_id, need)?;
            if init_zero {
                ws.marlin_ws.as_ref().unwrap().zero_async(stream)?;
            }
            addr
        };
        let had_scratch_bytes = self
            .layers
            .first()
            .map(|l| {
                [
                    &l.q_proj,
                    &l.k_proj,
                    &l.v_proj,
                    &l.o_proj,
                    &l.gate_up_proj,
                    &l.down_proj,
                ]
                .into_iter()
                .filter_map(|p| p.input_transform_workspace_bytes(seq_len))
                .max()
                .unwrap_or(0)
            })
            .unwrap_or(0);
        let exec_scratch_bytes = lse_scratch_bytes.max(had_scratch_bytes).max(64 * 1024);
        let exec_scratch_addr =
            Qwen3VlWorkspace::ensure_buf(&mut ws.exec_scratch, self.device_id, exec_scratch_bytes)?;
        let ex = stream.exec(exec_scratch_addr, exec_scratch_bytes);

        let d_residual =
            Qwen3VlWorkspace::ensure_buf(&mut ws.d_residual, self.device_id, seq_len * hidden * 2)?;
        let d_norm =
            Qwen3VlWorkspace::ensure_buf(&mut ws.d_norm, self.device_id, seq_len * hidden * 2)?;
        let d_q = Qwen3VlWorkspace::ensure_buf(&mut ws.d_q, self.device_id, seq_len * q_dim * 2)?;
        let d_k = Qwen3VlWorkspace::ensure_buf(&mut ws.d_k, self.device_id, seq_len * kv_dim * 2)?;
        let d_v = Qwen3VlWorkspace::ensure_buf(&mut ws.d_v, self.device_id, seq_len * kv_dim * 2)?;
        let d_attn_out =
            Qwen3VlWorkspace::ensure_buf(&mut ws.d_attn_out, self.device_id, seq_len * q_dim * 2)?;
        let d_proj_out =
            Qwen3VlWorkspace::ensure_buf(&mut ws.d_proj_out, self.device_id, seq_len * hidden * 2)?;
        let d_gate_up = Qwen3VlWorkspace::ensure_buf(
            &mut ws.d_gate_up,
            self.device_id,
            seq_len * 2 * interm * 2,
        )?;
        let d_mlp_interm = Qwen3VlWorkspace::ensure_buf(
            &mut ws.d_mlp_interm,
            self.device_id,
            seq_len * interm * 2,
        )?;
        let d_positions = Qwen3VlWorkspace::ensure_buf(
            &mut ws.d_positions,
            self.device_id,
            positions_flat.len() * 4,
        )?;

        ws.d_residual
            .as_ref()
            .unwrap()
            .copy_from_host(&embed_f16, stream)?;
        ws.d_positions
            .as_ref()
            .unwrap()
            .copy_from_host(&positions_flat, stream)?;

        let num_vis = vis_token_indices.len();
        let d_vis_indices = if num_vis > 0 {
            let addr =
                Qwen3VlWorkspace::ensure_buf(&mut ws.d_vis_indices, self.device_id, num_vis * 4)?;
            ws.d_vis_indices
                .as_ref()
                .unwrap()
                .copy_from_host(&vis_token_indices, stream)?;
            if let Some(ref pooler) = vis_pooler {
                check_rc(
                    // SAFETY: `d_residual` has `seq_len * hidden` FP16 elements, `pooler` has `num_vis * hidden`
                    // FP16 elements, and `d_vis_indices` has `num_vis` valid indices in `0..seq_len`.
                    unsafe {
                        ffi::warp34_embedding_splice_f16(
                            &ex,
                            d_residual as *mut c_void,
                            pooler.addr() as *const c_void,
                            addr as *const i32,
                            num_vis as i32,
                            hidden as i32,
                        )
                    },
                    "warp34_embedding_splice_f16(Qwen3-VL)",
                )?;
            }
            addr
        } else {
            0
        };

        let [mrope_s0, mrope_s1, mrope_s2] = self.cfg.mrope_section;
        let rope_theta = self.cfg.rope_theta;
        let rms_eps = self.cfg.rms_norm_eps;

        // 4. Decoder layers 0..num_hidden_layers
        for (layer_idx, layer) in self.layers.iter().enumerate() {
            check_rc(
                // SAFETY: `d_residual` and `d_norm` span `seq_len * hidden * 2` bytes; `input_norm_w` spans `hidden * 2` bytes.
                unsafe {
                    ffi::warp34_rms_norm_f16(
                        &ex,
                        d_residual as *const c_void,
                        layer.input_norm_w as *const c_void,
                        d_norm as *mut c_void,
                        core::ptr::null(),
                        seq_len as i32,
                        hidden as i32,
                        rms_eps,
                    )
                },
                "warp34_rms_norm_f16(input_layernorm)",
            )?;

            let qkv_share_h64 = layer.q_proj.can_fuse_h64_from_producer()
                && layer.k_proj.can_fuse_h64_from_producer()
                && layer.v_proj.can_fuse_h64_from_producer();
            layer.q_proj.forward(
                &ex,
                cublas,
                marlin_ws_addr,
                marlin_ws_ints,
                d_norm,
                d_q,
                seq_len,
            )?;
            if qkv_share_h64 {
                layer.k_proj.forward_prerotated(
                    &ex,
                    cublas,
                    marlin_ws_addr,
                    marlin_ws_ints,
                    ex.scratch_ptr,
                    d_k,
                    seq_len,
                )?;
                layer.v_proj.forward_prerotated(
                    &ex,
                    cublas,
                    marlin_ws_addr,
                    marlin_ws_ints,
                    ex.scratch_ptr,
                    d_v,
                    seq_len,
                )?;
            } else {
                layer.k_proj.forward(
                    &ex,
                    cublas,
                    marlin_ws_addr,
                    marlin_ws_ints,
                    d_norm,
                    d_k,
                    seq_len,
                )?;
                layer.v_proj.forward(
                    &ex,
                    cublas,
                    marlin_ws_addr,
                    marlin_ws_ints,
                    d_norm,
                    d_v,
                    seq_len,
                )?;
            }

            check_rc(
                // SAFETY: `d_q` (`[seq_len, q_dim]`) and `d_k` (`[seq_len, kv_dim]`) are normalized and rotated in-place
                // using shared memory per head; `d_positions` has `3 * seq_len` i32 elements.
                unsafe {
                    ffi::warp34_fused_qk_rmsnorm_rope_f16(
                        &ex,
                        d_q as *const c_void,
                        d_k as *const c_void,
                        layer.q_norm_w as *const c_void,
                        layer.k_norm_w as *const c_void,
                        d_positions as *const i32,
                        core::ptr::null(),
                        d_q as *mut c_void,
                        d_k as *mut c_void,
                        seq_len as i32,
                        seq_len as i32,
                        num_q_heads as i32,
                        num_kv_heads as i32,
                        head_dim as i32,
                        rotary_dim as i32,
                        mrope_s0 as i32,
                        mrope_s1 as i32,
                        mrope_s2 as i32,
                        rope_theta,
                        rms_eps,
                        0.0,
                    )
                },
                "warp34_fused_qk_rmsnorm_rope_f16(Qwen3-VL)",
            )?;

            check_rc(
                // SAFETY: `d_q`, `d_k`, `d_v`, `d_attn_out` are 16-byte aligned device buffers and `ex` scratch
                // holds at least `seq_len * num_q_heads * 4` bytes for softmax LSE.
                unsafe {
                    ffi::warp34_mha_fa2_d128_f16(
                        &ex,
                        d_q as *const c_void,
                        d_k as *const c_void,
                        d_v as *const c_void,
                        d_attn_out as *mut c_void,
                        1,
                        seq_len as i32,
                        seq_len as i32,
                        num_q_heads as i32,
                        num_kv_heads as i32,
                        1,
                    )
                },
                "warp34_mha_fa2_d128_f16(Qwen3-VL causal)",
            )?;

            layer.o_proj.forward(
                &ex,
                cublas,
                marlin_ws_addr,
                marlin_ws_ints,
                d_attn_out,
                d_proj_out,
                seq_len,
            )?;

            check_rc(
                // SAFETY: `d_proj_out`, `d_residual`, `d_norm` span `seq_len * hidden * 2` bytes.
                unsafe {
                    ffi::warp34_fused_add_rms_norm_f16(
                        &ex,
                        d_proj_out as *const c_void,
                        d_residual as *mut c_void,
                        layer.post_attn_norm_w as *const c_void,
                        d_norm as *mut c_void,
                        core::ptr::null(),
                        seq_len as i32,
                        hidden as i32,
                        rms_eps,
                    )
                },
                "warp34_fused_add_rms_norm_f16(post_attention_layernorm)",
            )?;

            layer.gate_up_proj.forward(
                &ex,
                cublas,
                marlin_ws_addr,
                marlin_ws_ints,
                d_norm,
                d_gate_up,
                seq_len,
            )?;

            if layer.down_proj.can_fuse_h64_from_producer() && (interm & 63) == 0 {
                check_rc(
                    // SAFETY: `d_gate_up` spans `seq_len * 2 * interm * 2` bytes and `d_mlp_interm` spans `seq_len * interm * 2` bytes.
                    unsafe {
                        ffi::warp34_silu_and_mul_hadamard_f16(
                            &ex,
                            d_gate_up as *const c_void,
                            d_mlp_interm as *mut c_void,
                            core::ptr::null(),
                            seq_len as i32,
                            interm as i32,
                        )
                    },
                    "warp34_silu_and_mul_hadamard_f16(Qwen3-VL)",
                )?;
                layer.down_proj.forward_prerotated(
                    &ex,
                    cublas,
                    marlin_ws_addr,
                    marlin_ws_ints,
                    d_mlp_interm,
                    d_proj_out,
                    seq_len,
                )?;
            } else {
                check_rc(
                    // SAFETY: `d_gate_up` spans `seq_len * 2 * interm * 2` bytes and `d_mlp_interm` spans `seq_len * interm * 2` bytes.
                    unsafe {
                        ffi::warp34_silu_and_mul_f16(
                            &ex,
                            d_gate_up as *const c_void,
                            d_mlp_interm as *mut c_void,
                            core::ptr::null(),
                            seq_len as i32,
                            interm as i32,
                        )
                    },
                    "warp34_silu_and_mul_f16(Qwen3-VL)",
                )?;
                layer.down_proj.forward(
                    &ex,
                    cublas,
                    marlin_ws_addr,
                    marlin_ws_ints,
                    d_mlp_interm,
                    d_proj_out,
                    seq_len,
                )?;
            }

            check_rc(
                // SAFETY: `d_residual` and `d_proj_out` are disjoint 16-byte-aligned buffers of `seq_len * hidden` FP16 elements.
                unsafe {
                    ffi::warp34_add_inplace_f16(
                        &ex,
                        d_residual as *mut c_void,
                        d_proj_out as *const c_void,
                        (seq_len * hidden) as i64,
                    )
                },
                "warp34_add_inplace_f16(Qwen3-VL mlp residual)",
            )?;

            // DeepStack additive visual injection after the first `deepstack_embeds.len()` decoder layers
            if num_vis > 0
                && let Some(ds_buf) = deepstack_embeds.get(layer_idx)
            {
                check_rc(
                    // SAFETY: `d_residual` has `seq_len * hidden` elements, `ds_buf` has `num_vis * hidden` elements,
                    // and `d_vis_indices` has `num_vis` valid token indices in `0..seq_len`.
                    unsafe {
                        ffi::warp34_embedding_add_f16(
                            &ex,
                            d_residual as *mut c_void,
                            ds_buf.addr() as *const c_void,
                            d_vis_indices as *const i32,
                            num_vis as i32,
                            seq_len as i32,
                            hidden as i32,
                        )
                    },
                    "warp34_embedding_add_f16(Qwen3-VL deepstack)",
                )?;
            }
        }

        // 5. Slice `[drop_idx..seq_len]` from pre-final-norm `d_residual`
        let start_tok = drop_idx.min(seq_len);
        let kept_len = seq_len - start_tok;
        let kept_elems = kept_len * hidden;
        let mut prompt_embeds_f16 = vec![0u16; kept_elems];
        if kept_elems > 0 {
            let byte_offset = (start_tok * hidden * 2) as u64;
            check_rc(
                // SAFETY: `prompt_embeds_f16` has `kept_elems * 2` bytes and `d_residual + byte_offset`
                // addresses the remaining `kept_elems * 2` bytes in `d_residual`.
                unsafe {
                    ffi::warp34_memcpy_d2h_async(
                        prompt_embeds_f16.as_mut_ptr() as *mut c_void,
                        (d_residual + byte_offset) as *const c_void,
                        kept_elems * 2,
                        stream.handle,
                    )
                },
                "Qwen3-VL conditioning D2H",
            )?;
            stream.synchronize()?;
        }
        let mut prompt_embeds_f32 = vec![0.0f32; kept_elems];
        f16_to_f32_slice(&prompt_embeds_f16, &mut prompt_embeds_f32);
        let image_pad_mask = input_ids[start_tok..]
            .iter()
            .map(|&id| id == image_token_id)
            .collect();

        Ok(Qwen3VlConditioningOutput {
            prompt_embeds_f16,
            prompt_embeds_f32,
            image_pad_mask,
            seq_len: kept_len,
            hidden_size: hidden,
        })
    }
}
