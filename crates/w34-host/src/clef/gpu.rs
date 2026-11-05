use super::*;

#[cfg(feature = "cuda")]
#[derive(Clone, Copy)]
pub(super) struct DevLayerNorm {
    weight: u64,
    bias: u64,
}

#[cfg(feature = "cuda")]
#[derive(Clone, Copy)]
pub(super) struct DevLinear {
    weight: u64,
    bias: u64,
}

#[cfg(feature = "cuda")]
#[derive(Clone, Copy)]
pub(super) struct DevMha {
    q_proj: DevLinear,
    kv_proj: DevLinear,
    out_proj: DevLinear,
}

#[cfg(feature = "cuda")]
#[derive(Clone, Copy)]
pub(super) struct DevEvidenceLayer {
    query_norm: DevLayerNorm,
    memory_norm: DevLayerNorm,
    attention: DevMha,
    feedforward_norm: DevLayerNorm,
    ff0: DevLinear,
    ff3: DevLinear,
}

#[cfg(feature = "cuda")]
#[derive(Clone, Copy)]
pub(super) struct DevDecoderLayer {
    norm1: DevLayerNorm,
    self_attn: DevMha,
    norm2: DevLayerNorm,
    multihead_attn: DevMha,
    norm3: DevLayerNorm,
    linear1: DevLinear,
    linear2: DevLinear,
}

#[cfg(feature = "cuda")]
pub(super) struct ClefDeviceSlot {
    d_memory: crate::cuda::DeviceBuffer,
    d_span_bounds: crate::cuda::DeviceBuffer,
    d_spans_f32: crate::cuda::DeviceBuffer,
}

/// Per-request streaming GPU state for `ClefJointHeadGpu`: projects each prefill chunk
/// directly into `[total_tokens, width]` `FP16` `memory` and accumulates `[1 + Q + O, H]`
/// `FP32` span means on device, avoiding any `[total_tokens, hidden_size]` host/device copy.
#[cfg(feature = "cuda")]
pub struct ClefDeviceRequestState {
    slot: Option<ClefDeviceSlot>,
    total_tokens: usize,
    next_offset: usize,
    num_q: usize,
    total_opts: usize,
    num_spans: usize,
}

#[cfg(feature = "cuda")]
impl ClefDeviceRequestState {
    #[inline]
    pub fn next_offset(&self) -> usize {
        self.next_offset
    }

    #[inline]
    pub fn total_tokens(&self) -> usize {
        self.total_tokens
    }
}

#[cfg(feature = "cuda")]
pub struct ClefJointHeadGpu {
    device_id: i32,
    config: ClefHeadConfig,
    output_embeddings: Arc<ClefOutputEmbeddings>,
    prior_logit_scale: f32,
    joint_logit_scale: f32,
    residual_gate: f32,
    _weight_arena: crate::cuda::DeviceArena,
    scratch_arena: crate::cuda::DeviceArena,
    free_slots: Vec<ClefDeviceSlot>,
    hidden_norm: DevLayerNorm,
    memory_projection: DevLinear,
    question_projection: DevLinear,
    option_question_projection: DevLinear,
    global_projection: DevLinear,
    option_context_projection: DevLinear,
    option_lexical_projection: DevLinear,
    type_embedding: u64,
    evidence_layers: Vec<DevEvidenceLayer>,
    option_summary_norm: DevLayerNorm,
    layers: Vec<DevDecoderLayer>,
    field_norm: DevLayerNorm,
    option_norm: DevLayerNorm,
    residual_0: DevLinear,
    residual_3: DevLinear,
}

#[cfg(feature = "cuda")]
pub(super) const CLEF_GPU_DEFAULT_SCRATCH_BYTES: usize = 64 * 1024 * 1024;
#[cfg(feature = "cuda")]
pub(super) const CLEF_GPU_MAX_CACHED_SLOTS: usize = 4;
#[cfg(feature = "cuda")]
pub(super) const CLEF_GPU_MAX_CACHED_SLOT_BYTES: usize = 36 * 1024 * 1024;

#[cfg(feature = "cuda")]
impl ClefJointHeadGpu {
    pub fn weight_arena_bytes(host_head: &ClefJointHead) -> usize {
        host_head.weight_arena_bytes()
    }

    /// Rank-0 VRAM headroom covering the weight arena, resting scratch arena, and bounded
    /// recycled slot pool.
    pub fn reserved_rank0_headroom_bytes(host_head: &ClefJointHead) -> usize {
        Self::weight_arena_bytes(host_head)
            .saturating_add(CLEF_GPU_DEFAULT_SCRATCH_BYTES)
            .saturating_add(CLEF_GPU_MAX_CACHED_SLOTS * CLEF_GPU_MAX_CACHED_SLOT_BYTES)
    }

    pub(super) fn finalize_f16_elem_counts(
        seq_len: usize,
        num_q: usize,
        total_opts: usize,
        w: usize,
        ff: usize,
    ) -> [usize; 24] {
        let max_q = total_opts.max(num_q);
        let max_m = seq_len.max(num_q);
        [
            w,                  // d_global_proj
            num_q * w,          // d_base_fields
            num_q * w,          // d_opt_q_proj
            total_opts * w,     // d_ctx_proj
            total_opts * w,     // d_lex_proj
            total_opts * w,     // d_routed_opts
            total_opts * w,     // d_routed_alt
            max_q * w,          // d_norm_q
            seq_len * w,        // d_norm_m
            max_q * w,          // d_q
            max_m * 2 * w,      // d_kv
            max_q * w,          // d_attn_ctx
            max_q * w,          // d_attn_out
            max_q * ff,         // d_ff_raw
            max_q * ff,         // d_ff_act
            max_q * w,          // d_ff_out
            num_q * w,          // d_summaries
            num_q * w,          // d_norm_summaries
            num_q * w,          // d_fields
            num_q * w,          // d_fields_alt
            total_opts * w,     // d_options_normed
            total_opts * 4 * w, // d_res_features
            total_opts,         // d_res_raw
            total_opts,         // d_residuals
        ]
    }

    pub(super) fn finalize_scratch_bytes(
        seq_len: usize,
        num_q: usize,
        total_opts: usize,
        h: usize,
        w: usize,
        ff: usize,
    ) -> usize {
        let num_spans = 1 + num_q + total_opts;
        let mut total = 0usize;
        let mut bump = |bytes: usize| {
            let aligned = (total + 255) & !255usize;
            total = aligned + bytes.max(1);
        };
        bump(total_opts * 4); // d_opt_q_idx
        bump((num_q + 1) * 4); // d_q_opt_offsets
        bump(num_q * 4); // d_q_types
        bump(total_opts * h * 2); // d_lex_f16
        bump(num_spans * h * 2); // d_spans_f16
        for elems in Self::finalize_f16_elem_counts(seq_len, num_q, total_opts, w, ff) {
            bump((elems * 2).max(16));
        }
        total + 4096
    }

    pub fn from_host(
        device_id: i32,
        stream: &crate::cuda::CudaStream,
        host_head: &ClefJointHead,
    ) -> Result<Self, String> {
        host_head.config.validate()?;
        let arena_bytes = Self::weight_arena_bytes(host_head);
        let mut arena = crate::cuda::DeviceArena::new(device_id, arena_bytes)
            .map_err(|e| format!("clef gpu weight arena alloc ({arena_bytes} B): {e}"))?;

        fn up_f16(
            arena: &mut crate::cuda::DeviceArena,
            stream: &crate::cuda::CudaStream,
            vals: &[f32],
        ) -> Result<u64, String> {
            let f16: Vec<u16> = vals.iter().copied().map(crate::cuda::f32_to_f16).collect();
            arena
                .upload_slice(&f16, stream)
                .map_err(|e| format!("clef gpu weight upload: {e}"))
        }
        fn up_ln(
            arena: &mut crate::cuda::DeviceArena,
            stream: &crate::cuda::CudaStream,
            ln: &LayerNormWeights,
        ) -> Result<DevLayerNorm, String> {
            Ok(DevLayerNorm {
                weight: up_f16(arena, stream, &ln.weight)?,
                bias: up_f16(arena, stream, &ln.bias)?,
            })
        }
        fn up_lin(
            arena: &mut crate::cuda::DeviceArena,
            stream: &crate::cuda::CudaStream,
            lin: &LinearWeights,
        ) -> Result<DevLinear, String> {
            Ok(DevLinear {
                weight: up_f16(arena, stream, &lin.weight)?,
                bias: match &lin.bias {
                    Some(b) => up_f16(arena, stream, b)?,
                    None => 0,
                },
            })
        }
        fn up_mha(
            arena: &mut crate::cuda::DeviceArena,
            stream: &crate::cuda::CudaStream,
            mha: &MultiheadAttnWeights,
        ) -> Result<DevMha, String> {
            Ok(DevMha {
                q_proj: up_lin(arena, stream, &mha.q_proj)?,
                kv_proj: up_lin(arena, stream, &mha.kv_proj)?,
                out_proj: up_lin(arena, stream, &mha.out_proj)?,
            })
        }

        let hidden_norm = up_ln(&mut arena, stream, &host_head.hidden_norm)?;
        let memory_projection = up_lin(&mut arena, stream, &host_head.memory_projection)?;
        let question_projection = up_lin(&mut arena, stream, &host_head.question_projection)?;
        let option_question_projection =
            up_lin(&mut arena, stream, &host_head.option_question_projection)?;
        let global_projection = up_lin(&mut arena, stream, &host_head.global_projection)?;
        let option_context_projection =
            up_lin(&mut arena, stream, &host_head.option_context_projection)?;
        let option_lexical_projection =
            up_lin(&mut arena, stream, &host_head.option_lexical_projection)?;
        let type_embedding = up_f16(&mut arena, stream, &host_head.type_embedding)?;

        let mut evidence_layers = Vec::with_capacity(host_head.evidence_layers.len());
        for ev in &host_head.evidence_layers {
            evidence_layers.push(DevEvidenceLayer {
                query_norm: up_ln(&mut arena, stream, &ev.query_norm)?,
                memory_norm: up_ln(&mut arena, stream, &ev.memory_norm)?,
                attention: up_mha(&mut arena, stream, &ev.attention)?,
                feedforward_norm: up_ln(&mut arena, stream, &ev.feedforward_norm)?,
                ff0: up_lin(&mut arena, stream, &ev.ff0)?,
                ff3: up_lin(&mut arena, stream, &ev.ff3)?,
            });
        }
        let option_summary_norm = up_ln(&mut arena, stream, &host_head.option_summary_norm)?;
        let mut layers = Vec::with_capacity(host_head.layers.len());
        for dec in &host_head.layers {
            layers.push(DevDecoderLayer {
                norm1: up_ln(&mut arena, stream, &dec.norm1)?,
                self_attn: up_mha(&mut arena, stream, &dec.self_attn)?,
                norm2: up_ln(&mut arena, stream, &dec.norm2)?,
                multihead_attn: up_mha(&mut arena, stream, &dec.multihead_attn)?,
                norm3: up_ln(&mut arena, stream, &dec.norm3)?,
                linear1: up_lin(&mut arena, stream, &dec.linear1)?,
                linear2: up_lin(&mut arena, stream, &dec.linear2)?,
            });
        }
        let field_norm = up_ln(&mut arena, stream, &host_head.field_norm)?;
        let option_norm = up_ln(&mut arena, stream, &host_head.option_norm)?;
        let residual_0 = up_lin(&mut arena, stream, &host_head.residual_0)?;
        let residual_3 = up_lin(&mut arena, stream, &host_head.residual_3)?;

        stream
            .synchronize()
            .map_err(|e| format!("clef gpu weight sync: {e}"))?;

        // Initial scratch arena sized modestly (grows on demand for larger sequences).
        let init_scratch = (512 * host_head.config.hidden_size * 2)
            .max(512 * host_head.config.width * 10)
            .max(256 * 1024);
        let scratch_arena = crate::cuda::DeviceArena::new(device_id, init_scratch)
            .map_err(|e| format!("clef gpu scratch alloc: {e}"))?;

        Ok(Self {
            device_id,
            config: host_head.config.clone(),
            output_embeddings: host_head.output_embeddings.clone(),
            prior_logit_scale: host_head.prior_logit_scale,
            joint_logit_scale: host_head.joint_logit_scale,
            residual_gate: host_head.residual_gate,
            _weight_arena: arena,
            scratch_arena,
            free_slots: Vec::new(),
            hidden_norm,
            memory_projection,
            question_projection,
            option_question_projection,
            global_projection,
            option_context_projection,
            option_lexical_projection,
            type_embedding,
            evidence_layers,
            option_summary_norm,
            layers,
            field_norm,
            option_norm,
            residual_0,
            residual_3,
        })
    }

    pub fn recycle_request_state(&mut self, mut state: ClefDeviceRequestState) {
        if let Some(slot) = state.slot.take() {
            let slot_bytes = slot
                .d_memory
                .len_bytes
                .saturating_add(slot.d_spans_f32.len_bytes)
                .saturating_add(slot.d_span_bounds.len_bytes);
            if self.free_slots.len() < CLEF_GPU_MAX_CACHED_SLOTS
                && slot_bytes <= CLEF_GPU_MAX_CACHED_SLOT_BYTES
            {
                self.free_slots.push(slot);
            }
        }
    }

    fn ensure_scratch(
        &mut self,
        needed_bytes: usize,
        stream: &crate::cuda::CudaStream,
    ) -> Result<(), String> {
        if self.scratch_arena.capacity() < needed_bytes {
            stream
                .synchronize()
                .map_err(|e| format!("clef gpu scratch sync: {e}"))?;
            let new_cap = needed_bytes
                .checked_next_power_of_two()
                .unwrap_or(needed_bytes);
            self.scratch_arena = crate::cuda::DeviceArena::new(self.device_id, new_cap)
                .map_err(|e| format!("clef gpu scratch realloc ({new_cap} B): {e}"))?;
        }
        self.scratch_arena.reset();
        Ok(())
    }

    pub fn begin_request(
        &mut self,
        record: &ClefEncodedRecord,
        stream: &crate::cuda::CudaStream,
    ) -> Result<ClefDeviceRequestState, String> {
        let seq_len = record.input_ids.len();
        if seq_len == 0 {
            return Err("clef gpu: input_ids must be non-empty".into());
        }
        if record.questions.is_empty() {
            return Err("clef gpu: questions must be non-empty".into());
        }
        let num_q = record.questions.len();
        let mut total_opts = 0usize;
        for q in &record.questions {
            if q.question_type >= 3 {
                return Err(format!(
                    "clef gpu: invalid question_type {} for '{}'",
                    q.question_type, q.question_id
                ));
            }
            let (qs, qe) = q.question_span;
            if qs >= qe || qe > seq_len {
                return Err(format!(
                    "clef gpu: invalid question_span ({qs}, {qe}) for seq_len {seq_len}"
                ));
            }
            if q.option_spans.is_empty() || q.option_spans.len() != q.option_ids.len() {
                return Err(format!(
                    "clef gpu: invalid option_spans for '{}'",
                    q.question_id
                ));
            }
            for &(os, oe) in &q.option_spans {
                if os >= oe || oe > seq_len {
                    return Err(format!(
                        "clef gpu: invalid option_span ({os}, {oe}) for seq_len {seq_len}"
                    ));
                }
            }
            total_opts += q.option_spans.len();
        }
        let num_spans = 1 + num_q + total_opts;
        let mut span_bounds = Vec::with_capacity(2 * num_spans);
        span_bounds.push((seq_len - 1) as i32);
        span_bounds.push(seq_len as i32);
        for q in &record.questions {
            span_bounds.push(q.question_span.0 as i32);
            span_bounds.push(q.question_span.1 as i32);
        }
        for q in &record.questions {
            for &(os, oe) in &q.option_spans {
                span_bounds.push(os as i32);
                span_bounds.push(oe as i32);
            }
        }

        let h = self.config.hidden_size;
        let w = self.config.width;
        let mem_bytes = seq_len * w * 2;
        let bounds_bytes = span_bounds.len() * 4;
        let spans_bytes = num_spans * h * 4;

        let mut slot = match self.free_slots.pop() {
            Some(s) => s,
            None => ClefDeviceSlot {
                d_memory: crate::cuda::DeviceBuffer::alloc(self.device_id, mem_bytes.max(1024))
                    .map_err(|e| format!("clef gpu d_memory alloc: {e}"))?,
                d_span_bounds: crate::cuda::DeviceBuffer::alloc(
                    self.device_id,
                    bounds_bytes.max(256),
                )
                .map_err(|e| format!("clef gpu d_span_bounds alloc: {e}"))?,
                d_spans_f32: crate::cuda::DeviceBuffer::alloc(
                    self.device_id,
                    spans_bytes.max(4096),
                )
                .map_err(|e| format!("clef gpu d_spans_f32 alloc: {e}"))?,
            },
        };
        if slot.d_memory.len_bytes < mem_bytes
            || slot.d_span_bounds.len_bytes < bounds_bytes
            || slot.d_spans_f32.len_bytes < spans_bytes
        {
            stream
                .synchronize()
                .map_err(|e| format!("clef gpu slot resize sync: {e}"))?;
            if slot.d_memory.len_bytes < mem_bytes {
                slot.d_memory =
                    crate::cuda::DeviceBuffer::alloc(self.device_id, mem_bytes.next_power_of_two())
                        .map_err(|e| format!("clef gpu d_memory realloc: {e}"))?;
            }
            if slot.d_span_bounds.len_bytes < bounds_bytes {
                slot.d_span_bounds = crate::cuda::DeviceBuffer::alloc(
                    self.device_id,
                    bounds_bytes.next_power_of_two(),
                )
                .map_err(|e| format!("clef gpu d_span_bounds realloc: {e}"))?;
            }
            if slot.d_spans_f32.len_bytes < spans_bytes {
                slot.d_spans_f32 = crate::cuda::DeviceBuffer::alloc(
                    self.device_id,
                    spans_bytes.next_power_of_two(),
                )
                .map_err(|e| format!("clef gpu d_spans_f32 realloc: {e}"))?;
            }
        }

        slot.d_span_bounds
            .copy_from_host(&span_bounds, stream)
            .map_err(|e| format!("clef gpu upload span_bounds: {e}"))?;
        crate::cuda::check_rc(
            // SAFETY: `d_spans_f32` has capacity >= `spans_bytes` on `stream.device_id`.
            unsafe {
                w34_kernels::ffi::w34_memset_async(
                    slot.d_spans_f32.ptr,
                    0,
                    spans_bytes,
                    stream.handle,
                )
            },
            "clef w34_memset_async spans",
        )
        .map_err(|e| e.to_string())?;

        Ok(ClefDeviceRequestState {
            slot: Some(slot),
            total_tokens: seq_len,
            next_offset: 0,
            num_q,
            total_opts,
            num_spans,
        })
    }

    pub fn ingest_chunk(
        &mut self,
        state: &mut ClefDeviceRequestState,
        norm_hidden_chunk_addr: u64,
        ctx_len_before: usize,
        chunk_len: usize,
        cublas: &crate::cuda::CublasHandle,
        stream: &crate::cuda::CudaStream,
    ) -> Result<(), String> {
        if state.next_offset != ctx_len_before {
            return Err(format!(
                "clef gpu: non-contiguous prefill chunk: expected offset {}, got {ctx_len_before}",
                state.next_offset
            ));
        }
        if chunk_len == 0 {
            return Ok(());
        }
        if ctx_len_before + chunk_len > state.total_tokens {
            return Err(format!(
                "clef gpu: chunk end {} exceeds total_tokens {}",
                ctx_len_before + chunk_len,
                state.total_tokens
            ));
        }
        let h = self.config.hidden_size;
        let w = self.config.width;
        self.ensure_scratch(chunk_len * h * 2 + 4096, stream)?;
        let d_chunk_normed = self
            .scratch_arena
            .alloc_aligned(chunk_len * h * 2)
            .map_err(|e| e.to_string())?;
        let slot = state
            .slot
            .as_ref()
            .ok_or_else(|| "clef gpu: missing request slot".to_string())?;

        let ex = stream.exec(0, 0);
        crate::cuda::check_rc(
            // SAFETY: `norm_hidden_chunk_addr` and `d_chunk_normed` are distinct valid FP16
            // buffers of `chunk_len * h` elements; `hidden_norm` weights have `h` elements.
            unsafe {
                w34_kernels::ffi::w34_vision_layernorm_f16(
                    &ex,
                    norm_hidden_chunk_addr as *const core::ffi::c_void,
                    self.hidden_norm.weight as *const core::ffi::c_void,
                    self.hidden_norm.bias as *const core::ffi::c_void,
                    d_chunk_normed as *mut core::ffi::c_void,
                    chunk_len as i32,
                    h as i32,
                    1e-5,
                )
            },
            "clef hidden_norm",
        )
        .map_err(|e| e.to_string())?;

        crate::cuda::check_rc(
            // SAFETY: `d_chunk_normed` (`[chunk_len, h]`), `d_span_bounds` (`[2 * num_spans]`),
            // and `d_spans_f32` (`[num_spans, h]`) are valid device allocations on `stream`.
            unsafe {
                w34_kernels::ffi::w34_clef_accum_spans_f16_to_f32(
                    &ex,
                    d_chunk_normed as *const core::ffi::c_void,
                    slot.d_span_bounds.ptr as *const i32,
                    slot.d_spans_f32.ptr as *mut f32,
                    ctx_len_before as i32,
                    chunk_len as i32,
                    state.num_spans as i32,
                    h as i32,
                )
            },
            "clef accum_spans",
        )
        .map_err(|e| e.to_string())?;

        let d_mem_dst = slot.d_memory.addr() + (ctx_len_before * w * 2) as u64;
        cublas
            .gemm_f16(
                d_chunk_normed,
                self.memory_projection.weight,
                d_mem_dst,
                chunk_len as i32,
                w as i32,
                h as i32,
            )
            .map_err(|e| format!("clef memory_projection gemm: {e}"))?;

        state.next_offset += chunk_len;
        Ok(())
    }

    pub fn finalize(
        &mut self,
        state: ClefDeviceRequestState,
        record: &ClefEncodedRecord,
        cublas: &crate::cuda::CublasHandle,
        stream: &crate::cuda::CudaStream,
    ) -> Result<Vec<Vec<f64>>, String> {
        let res = self.finalize_inner(&state, record, cublas, stream);
        self.recycle_request_state(state);
        if self.scratch_arena.capacity() > CLEF_GPU_DEFAULT_SCRATCH_BYTES {
            let _ = stream.synchronize();
            if let Ok(shrunk) =
                crate::cuda::DeviceArena::new(self.device_id, CLEF_GPU_DEFAULT_SCRATCH_BYTES)
            {
                self.scratch_arena = shrunk;
            }
        }
        res
    }

    fn finalize_inner(
        &mut self,
        state: &ClefDeviceRequestState,
        record: &ClefEncodedRecord,
        cublas: &crate::cuda::CublasHandle,
        stream: &crate::cuda::CudaStream,
    ) -> Result<Vec<Vec<f64>>, String> {
        if state.next_offset != state.total_tokens {
            return Err(format!(
                "clef gpu: incomplete prefill coverage: ingested {}, expected {}",
                state.next_offset, state.total_tokens
            ));
        }
        let slot = state
            .slot
            .as_ref()
            .ok_or_else(|| "clef gpu: missing request slot".to_string())?;

        let seq_len = state.total_tokens;
        let num_q = state.num_q;
        let total_opts = state.total_opts;
        let num_spans = state.num_spans;
        let h = self.config.hidden_size;
        let w = self.config.width;
        let ff = self.config.feedforward;
        let heads = self.config.heads;
        let hd = w / heads;

        // Fetch lexical embeddings on host via cached Safetensors pread (~10–20 µs).
        let mut lexical_f32 = Vec::with_capacity(total_opts * h);
        let mut opt_q_idx = Vec::with_capacity(total_opts);
        let mut q_opt_offsets = Vec::with_capacity(num_q + 1);
        let mut q_types = Vec::with_capacity(num_q);
        q_opt_offsets.push(0i32);
        for (qi, q) in record.questions.iter().enumerate() {
            q_types.push(q.question_type as i32);
            for &(s, e) in &q.option_spans {
                let lex = self.output_embeddings.mean_rows(&record.input_ids[s..e])?;
                lexical_f32.extend_from_slice(&lex);
                opt_q_idx.push(qi as i32);
            }
            q_opt_offsets.push(opt_q_idx.len() as i32);
        }
        let lexical_f16: Vec<u16> = lexical_f32
            .iter()
            .copied()
            .map(crate::cuda::f32_to_f16)
            .collect();

        let needed_scratch = Self::finalize_scratch_bytes(seq_len, num_q, total_opts, h, w, ff);
        self.ensure_scratch(needed_scratch, stream)?;

        let d_opt_q_idx = self
            .scratch_arena
            .upload_slice(&opt_q_idx, stream)
            .map_err(|e| e.to_string())?;
        let d_q_opt_offsets = self
            .scratch_arena
            .upload_slice(&q_opt_offsets, stream)
            .map_err(|e| e.to_string())?;
        let d_q_types = self
            .scratch_arena
            .upload_slice(&q_types, stream)
            .map_err(|e| e.to_string())?;
        let d_lex_f16 = self
            .scratch_arena
            .upload_slice(&lexical_f16, stream)
            .map_err(|e| e.to_string())?;

        // Pack spans without 256-byte gaps so sub-slices [0], [1..1+Q], [1+Q..] are contiguous.
        let d_spans_f16 = self
            .scratch_arena
            .alloc_aligned(num_spans * h * 2)
            .map_err(|e| e.to_string())?;
        let d_global_f16 = d_spans_f16;
        let d_q_vecs_f16 = d_spans_f16 + (h * 2) as u64;
        let d_opt_ctx_f16 = d_spans_f16 + ((1 + num_q) * h * 2) as u64;

        let f16_counts = Self::finalize_f16_elem_counts(seq_len, num_q, total_opts, w, ff);
        let mut f16_ptrs = [0u64; 24];
        for (dst, elems) in f16_ptrs.iter_mut().zip(f16_counts) {
            *dst = self
                .scratch_arena
                .alloc_aligned((elems * 2).max(16))
                .map_err(|e| e.to_string())?;
        }
        let [
            d_global_proj,
            d_base_fields,
            d_opt_q_proj,
            d_ctx_proj,
            d_lex_proj,
            mut d_routed_opts,
            mut d_routed_alt,
            d_norm_q,
            d_norm_m,
            d_q,
            d_kv,
            d_attn_ctx,
            d_attn_out,
            d_ff_raw,
            d_ff_act,
            d_ff_out,
            d_summaries,
            d_norm_summaries,
            mut d_fields,
            mut d_fields_alt,
            d_options_normed,
            d_res_features,
            d_res_raw,
            d_residuals,
        ] = f16_ptrs;

        let ex = stream.exec(0, 0);
        let layernorm = |src: u64, ln: DevLayerNorm, dst: u64, rows: usize, cols: usize| {
            crate::cuda::check_rc(
                // SAFETY: Caller passes distinct, arena-allocated `src` and `dst` FP16 buffers
                // of `rows * cols` elements and `ln` weight/bias of `cols` elements.
                unsafe {
                    w34_kernels::ffi::w34_vision_layernorm_f16(
                        &ex,
                        src as *const core::ffi::c_void,
                        ln.weight as *const core::ffi::c_void,
                        ln.bias as *const core::ffi::c_void,
                        dst as *mut core::ffi::c_void,
                        rows as i32,
                        cols as i32,
                        1e-5,
                    )
                },
                "clef layernorm",
            )
            .map_err(|e| e.to_string())
        };
        let bias_add = |x: u64, bias: u64, residual: u64, dst: u64, rows: usize, cols: usize| {
            crate::cuda::check_rc(
                // SAFETY: Caller passes distinct `x`, `residual` (or 0 -> null), and `dst`
                // FP16 buffers of `rows * cols` elements and `bias` of `cols` elements.
                unsafe {
                    w34_kernels::ffi::w34_vision_bias_add_f16(
                        &ex,
                        x as *const core::ffi::c_void,
                        bias as *const core::ffi::c_void,
                        if residual == 0 {
                            core::ptr::null()
                        } else {
                            residual as *const core::ffi::c_void
                        },
                        dst as *mut core::ffi::c_void,
                        rows as i32,
                        cols as i32,
                    )
                },
                "clef bias_add",
            )
            .map_err(|e| e.to_string())
        };
        let bias_gelu = |x: u64, bias: u64, dst: u64, rows: usize, cols: usize| {
            crate::cuda::check_rc(
                // SAFETY: Caller passes distinct `x` and `dst` FP16 buffers of `rows * cols`
                // elements and `bias` of `cols` elements.
                unsafe {
                    w34_kernels::ffi::w34_vision_bias_gelu_f16(
                        &ex,
                        x as *const core::ffi::c_void,
                        bias as *const core::ffi::c_void,
                        dst as *mut core::ffi::c_void,
                        rows as i32,
                        cols as i32,
                        0, // exact erf GELU
                    )
                },
                "clef bias_gelu",
            )
            .map_err(|e| e.to_string())
        };
        let run_mha = |q_in: u64,
                       q_len: usize,
                       m_in: u64,
                       m_len: usize,
                       mha: DevMha,
                       out: u64|
         -> Result<(), String> {
            cublas
                .gemm_f16(
                    q_in,
                    mha.q_proj.weight,
                    d_q,
                    q_len as i32,
                    w as i32,
                    w as i32,
                )
                .map_err(|e| e.to_string())?;
            cublas
                .gemm_f16(
                    m_in,
                    mha.kv_proj.weight,
                    d_kv,
                    m_len as i32,
                    (2 * w) as i32,
                    w as i32,
                )
                .map_err(|e| e.to_string())?;
            crate::cuda::check_rc(
                // SAFETY: `d_q` (`[q_len, w]`), `d_kv` (`[m_len, 2*w]`), `d_attn_ctx` (`[q_len, w]`),
                // and biases (`[w]`, `[2*w]`) are valid device allocations with `w == heads * hd`.
                unsafe {
                    w34_kernels::ffi::w34_clef_mha_attn_f16(
                        &ex,
                        d_q as *const core::ffi::c_void,
                        mha.q_proj.bias as *const core::ffi::c_void,
                        d_kv as *const core::ffi::c_void,
                        mha.kv_proj.bias as *const core::ffi::c_void,
                        d_attn_ctx as *mut core::ffi::c_void,
                        q_len as i32,
                        m_len as i32,
                        heads as i32,
                        hd as i32,
                    )
                },
                "clef mha_attn",
            )
            .map_err(|e| e.to_string())?;
            cublas
                .gemm_f16(
                    d_attn_ctx,
                    mha.out_proj.weight,
                    out,
                    q_len as i32,
                    w as i32,
                    w as i32,
                )
                .map_err(|e| e.to_string())
        };

        // 1. Cast accumulated FP32 span means to FP16 and run query/option projections.
        crate::cuda::check_rc(
            // SAFETY: `d_spans_f32` and `d_spans_f16` hold `num_spans * h` elements on `stream`.
            unsafe {
                w34_kernels::ffi::w34_clef_cast_f32_to_f16(
                    &ex,
                    slot.d_spans_f32.ptr as *const f32,
                    d_spans_f16 as *mut core::ffi::c_void,
                    (num_spans * h) as i64,
                )
            },
            "clef cast_f32_to_f16",
        )
        .map_err(|e| e.to_string())?;

        cublas
            .gemm_f16(
                d_global_f16,
                self.global_projection.weight,
                d_global_proj,
                1,
                w as i32,
                h as i32,
            )
            .map_err(|e| e.to_string())?;
        cublas
            .gemm_f16(
                d_q_vecs_f16,
                self.question_projection.weight,
                d_base_fields,
                num_q as i32,
                w as i32,
                h as i32,
            )
            .map_err(|e| e.to_string())?;
        cublas
            .gemm_f16(
                d_q_vecs_f16,
                self.option_question_projection.weight,
                d_opt_q_proj,
                num_q as i32,
                w as i32,
                h as i32,
            )
            .map_err(|e| e.to_string())?;
        cublas
            .gemm_f16(
                d_opt_ctx_f16,
                self.option_context_projection.weight,
                d_ctx_proj,
                total_opts as i32,
                w as i32,
                h as i32,
            )
            .map_err(|e| e.to_string())?;
        cublas
            .gemm_f16(
                d_lex_f16,
                self.option_lexical_projection.weight,
                d_lex_proj,
                total_opts as i32,
                w as i32,
                h as i32,
            )
            .map_err(|e| e.to_string())?;

        crate::cuda::check_rc(
            // SAFETY: `d_ctx_proj`, `d_lex_proj`, and `d_routed_opts` have `total_opts * w`
            // FP16 elements; `d_opt_q_proj` has `num_q * w` FP16 elements; `d_opt_q_idx` has `total_opts` i32s.
            unsafe {
                w34_kernels::ffi::w34_clef_build_option_queries_f16(
                    &ex,
                    d_ctx_proj as *const core::ffi::c_void,
                    d_lex_proj as *const core::ffi::c_void,
                    d_opt_q_proj as *const core::ffi::c_void,
                    d_opt_q_idx as *const i32,
                    d_routed_opts as *mut core::ffi::c_void,
                    total_opts as i32,
                    num_q as i32,
                    w as i32,
                )
            },
            "clef build_option_queries",
        )
        .map_err(|e| e.to_string())?;

        // 2. Evidence-routing cross-attention layers over full sequence `d_memory [seq_len, W]`.
        let d_memory = slot.d_memory.addr();
        for ev in &self.evidence_layers {
            layernorm(d_routed_opts, ev.query_norm, d_norm_q, total_opts, w)?;
            layernorm(d_memory, ev.memory_norm, d_norm_m, seq_len, w)?;
            run_mha(
                d_norm_q,
                total_opts,
                d_norm_m,
                seq_len,
                ev.attention,
                d_attn_out,
            )?;
            bias_add(
                d_attn_out,
                ev.attention.out_proj.bias,
                d_routed_opts,
                d_routed_alt,
                total_opts,
                w,
            )?;
            std::mem::swap(&mut d_routed_opts, &mut d_routed_alt);

            layernorm(d_routed_opts, ev.feedforward_norm, d_norm_q, total_opts, w)?;
            cublas
                .gemm_f16(
                    d_norm_q,
                    ev.ff0.weight,
                    d_ff_raw,
                    total_opts as i32,
                    ff as i32,
                    w as i32,
                )
                .map_err(|e| e.to_string())?;
            bias_gelu(d_ff_raw, ev.ff0.bias, d_ff_act, total_opts, ff)?;
            cublas
                .gemm_f16(
                    d_ff_act,
                    ev.ff3.weight,
                    d_ff_out,
                    total_opts as i32,
                    w as i32,
                    ff as i32,
                )
                .map_err(|e| e.to_string())?;
            bias_add(
                d_ff_out,
                ev.ff3.bias,
                d_routed_opts,
                d_routed_alt,
                total_opts,
                w,
            )?;
            std::mem::swap(&mut d_routed_opts, &mut d_routed_alt);
        }

        // 3. Option summaries & joint field queries.
        crate::cuda::check_rc(
            // SAFETY: `d_routed_opts` (`[total_opts, w]`), `d_base_fields` (`[num_q, w]`),
            // `d_q_opt_offsets` (`[num_q + 1]`), and `d_summaries` (`[num_q, w]`) are valid device buffers.
            unsafe {
                w34_kernels::ffi::w34_clef_option_summaries_f16(
                    &ex,
                    d_routed_opts as *const core::ffi::c_void,
                    d_base_fields as *const core::ffi::c_void,
                    d_q_opt_offsets as *const i32,
                    d_summaries as *mut core::ffi::c_void,
                    num_q as i32,
                    total_opts as i32,
                    w as i32,
                )
            },
            "clef option_summaries",
        )
        .map_err(|e| e.to_string())?;
        layernorm(
            d_summaries,
            self.option_summary_norm,
            d_norm_summaries,
            num_q,
            w,
        )?;
        crate::cuda::check_rc(
            // SAFETY: `d_base_fields`, `d_norm_summaries`, and `d_fields` have `num_q * w` FP16
            // elements; `d_global_proj` has `w`; `type_embedding` has `3 * w`; `d_q_types` has `num_q`.
            unsafe {
                w34_kernels::ffi::w34_clef_build_fields_f16(
                    &ex,
                    d_base_fields as *const core::ffi::c_void,
                    d_norm_summaries as *const core::ffi::c_void,
                    d_global_proj as *const core::ffi::c_void,
                    self.type_embedding as *const core::ffi::c_void,
                    d_q_types as *const i32,
                    d_fields as *mut core::ffi::c_void,
                    num_q as i32,
                    w as i32,
                )
            },
            "clef build_fields",
        )
        .map_err(|e| e.to_string())?;

        // 4. Joint field transformer decoder layers (self-attn over fields + cross-attn over memory).
        for dec in &self.layers {
            layernorm(d_fields, dec.norm1, d_norm_q, num_q, w)?;
            run_mha(d_norm_q, num_q, d_norm_q, num_q, dec.self_attn, d_attn_out)?;
            bias_add(
                d_attn_out,
                dec.self_attn.out_proj.bias,
                d_fields,
                d_fields_alt,
                num_q,
                w,
            )?;
            std::mem::swap(&mut d_fields, &mut d_fields_alt);

            layernorm(d_fields, dec.norm2, d_norm_q, num_q, w)?;
            run_mha(
                d_norm_q,
                num_q,
                d_memory,
                seq_len,
                dec.multihead_attn,
                d_attn_out,
            )?;
            bias_add(
                d_attn_out,
                dec.multihead_attn.out_proj.bias,
                d_fields,
                d_fields_alt,
                num_q,
                w,
            )?;
            std::mem::swap(&mut d_fields, &mut d_fields_alt);

            layernorm(d_fields, dec.norm3, d_norm_q, num_q, w)?;
            cublas
                .gemm_f16(
                    d_norm_q,
                    dec.linear1.weight,
                    d_ff_raw,
                    num_q as i32,
                    ff as i32,
                    w as i32,
                )
                .map_err(|e| e.to_string())?;
            bias_gelu(d_ff_raw, dec.linear1.bias, d_ff_act, num_q, ff)?;
            cublas
                .gemm_f16(
                    d_ff_act,
                    dec.linear2.weight,
                    d_ff_out,
                    num_q as i32,
                    w as i32,
                    ff as i32,
                )
                .map_err(|e| e.to_string())?;
            bias_add(d_ff_out, dec.linear2.bias, d_fields, d_fields_alt, num_q, w)?;
            std::mem::swap(&mut d_fields, &mut d_fields_alt);
        }

        layernorm(d_fields, self.field_norm, d_fields_alt, num_q, w)?;
        std::mem::swap(&mut d_fields, &mut d_fields_alt);

        // 5. Batched option_norm + residual_scorer MLP across all `total_opts`.
        layernorm(
            d_routed_opts,
            self.option_norm,
            d_options_normed,
            total_opts,
            w,
        )?;
        crate::cuda::check_rc(
            // SAFETY: `d_fields` (`[num_q, w]`), `d_options_normed` (`[total_opts, w]`),
            // `d_opt_q_idx` (`[total_opts]`), and `d_res_features` (`[total_opts, 4 * w]`) are valid device buffers.
            unsafe {
                w34_kernels::ffi::w34_clef_build_residual_features_f16(
                    &ex,
                    d_fields as *const core::ffi::c_void,
                    d_options_normed as *const core::ffi::c_void,
                    d_opt_q_idx as *const i32,
                    d_res_features as *mut core::ffi::c_void,
                    total_opts as i32,
                    num_q as i32,
                    w as i32,
                )
            },
            "clef build_residual_features",
        )
        .map_err(|e| e.to_string())?;
        cublas
            .gemm_f16(
                d_res_features,
                self.residual_0.weight,
                d_norm_q,
                total_opts as i32,
                w as i32,
                (4 * w) as i32,
            )
            .map_err(|e| e.to_string())?;
        bias_gelu(d_norm_q, self.residual_0.bias, d_routed_alt, total_opts, w)?;
        cublas
            .gemm_f16(
                d_routed_alt,
                self.residual_3.weight,
                d_res_raw,
                total_opts as i32,
                1,
                w as i32,
            )
            .map_err(|e| e.to_string())?;
        bias_add(
            d_res_raw,
            self.residual_3.bias,
            0,
            d_residuals,
            total_opts,
            1,
        )?;

        // 6. Single small D2H sync (independent of `seq_len`) + IEEE-754 f64 prior/cosine finish.
        let mut h_gq_spans = vec![0.0f32; (1 + num_q) * h];
        let mut h_fields_f16 = vec![0u16; num_q * w];
        let mut h_opts_f16 = vec![0u16; total_opts * w];
        let mut h_res_f16 = vec![0u16; total_opts];

        fn copy_d2h_async<T: Copy>(
            src: u64,
            dst: &mut [T],
            stream: &crate::cuda::CudaStream,
        ) -> Result<(), String> {
            let bytes = core::mem::size_of_val(dst);
            if bytes == 0 {
                return Ok(());
            }
            crate::cuda::check_rc(
                // SAFETY: `stream.device_id` is a valid CUDA device index initialized for this rank.
                unsafe { w34_kernels::ffi::w34_set_device(stream.device_id) },
                "clef d2h set_device",
            )
            .map_err(|e| e.to_string())?;
            crate::cuda::check_rc(
                // SAFETY: `src` points to at least `bytes` valid device memory and `dst` remains
                // alive until `stream.synchronize()` completes immediately below.
                unsafe {
                    w34_kernels::ffi::w34_memcpy_d2h_async(
                        dst.as_mut_ptr() as *mut core::ffi::c_void,
                        src as *const core::ffi::c_void,
                        bytes,
                        stream.handle,
                    )
                },
                "clef d2h memcpy",
            )
            .map_err(|e| e.to_string())
        }

        copy_d2h_async(slot.d_spans_f32.addr(), &mut h_gq_spans, stream)?;
        copy_d2h_async(d_fields, &mut h_fields_f16, stream)?;
        copy_d2h_async(d_options_normed, &mut h_opts_f16, stream)?;
        copy_d2h_async(d_residuals, &mut h_res_f16, stream)?;
        stream
            .synchronize()
            .map_err(|e| format!("clef gpu finalize sync: {e}"))?;

        let global_vec = &h_gq_spans[..h];
        let h_fields: Vec<f32> = h_fields_f16
            .into_iter()
            .map(crate::cuda::f16_to_f32)
            .collect();
        let h_opts: Vec<f32> = h_opts_f16
            .into_iter()
            .map(crate::cuda::f16_to_f32)
            .collect();

        let max_clamp = 100.0f64.ln() as f32;
        let prior_scale = (self.prior_logit_scale.min(max_clamp) as f64).exp();
        let joint_scale = (self.joint_logit_scale.min(max_clamp) as f64).exp();
        let gate = 1.0 / (1.0 + (-(self.residual_gate as f64)).exp());

        let mut results = Vec::with_capacity(num_q);
        let mut opt_offset = 0usize;
        for (qi, q) in record.questions.iter().enumerate() {
            let n_opts = q.option_spans.len();
            let q_vec = &h_gq_spans[(1 + qi) * h..(2 + qi) * h];
            let mut anchor = vec![0.0f32; h];
            for d in 0..h {
                anchor[d] = q_vec[d] + global_vec[d];
            }
            l2_normalize_in_place(&mut anchor, 1e-12);

            let field_row = &h_fields[qi * w..(qi + 1) * w];
            let mut q_logits = Vec::with_capacity(n_opts);
            for oi in 0..n_opts {
                let g_oi = opt_offset + oi;
                let mut lex_row = lexical_f32[g_oi * h..(g_oi + 1) * h].to_vec();
                l2_normalize_in_place(&mut lex_row, 1e-12);
                let prior = prior_scale * (dot_f32_scalar(&lex_row, &anchor) as f64);

                let opt_row = &h_opts[g_oi * w..(g_oi + 1) * w];
                let cosine = cosine_similarity_f32(field_row, opt_row, 1e-8) as f64;
                let res_val = crate::cuda::f16_to_f32(h_res_f16[g_oi]) as f64;
                let joint = joint_scale * cosine + res_val;
                let logit = prior + gate * joint;
                if !logit.is_finite() {
                    return Err("clef gpu: non-finite logit produced".into());
                }
                q_logits.push(logit);
            }
            results.push(q_logits);
            opt_offset += n_opts;
        }

        Ok(results)
    }
}
