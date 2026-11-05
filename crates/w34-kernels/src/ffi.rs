//! FFI bindings to `csrc/w34_cuda.h`.

#[cfg(feature = "cuda")]
use core::ffi::c_void;

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct W34Exec {
    pub stream: usize,
    pub scratch_ptr: u64,
    pub scratch_size: usize,
}

const _: [(); 24] = [(); core::mem::size_of::<W34Exec>()];

pub const W34_DTYPE_F16: i32 = 0;
pub const W34_DTYPE_BF16: i32 = 1;
pub const W34_DTYPE_F32: i32 = 2;
pub const W34_DTYPE_I32: i32 = 3;
pub const W34_DTYPE_U8: i32 = 4;
pub const W34_DTYPE_U32: i32 = 5;

pub const W34_KV_NAMESPACE_TARGET_FULL_ATTN: i32 = 0;
pub const W34_KV_NAMESPACE_MTP_FULL_ATTN: i32 = 1;
pub const W34_KV_NAMESPACE_DFLASH_DRAFT_CTX: i32 = 2;

pub const W34_KV_FORMAT_FP16: i32 = 0;
pub const W34_KV_FORMAT_BF16: i32 = 1;
pub const W34_KV_FORMAT_TQ4: i32 = 2;
pub const W34_KV_FORMAT_Q8_0: i32 = 3;

pub const W34_KV_TRANSFORM_IDENTITY: i32 = 0;
pub const W34_KV_TRANSFORM_RADEMACHER_WHT256: i32 = 1;

pub const W34_ATTN_MASK_CAUSAL: i32 = 0;
/// Legacy name: query-relative left window, non-causal (no right bound).
pub const W34_ATTN_MASK_SYMMETRIC_WINDOW: i32 = 1;
pub const W34_ATTN_MASK_NONCAUSAL_WINDOW: i32 = 1;

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct W34KvLayerLayout {
    pub k_base: u64,
    pub v_base: u64,
    pub k_page_stride_bytes: u64,
    pub k_head_stride_bytes: u64,
    pub k_token_stride_bytes: u64,
    pub v_page_stride_bytes: u64,
    pub v_head_stride_bytes: u64,
    pub v_token_stride_bytes: u64,
    pub namespace_id: i32,
    pub layer_id: i32,
    pub local_kv_heads: i32,
    pub head_dim: i32,
    pub block_size: i32,
    pub k_format: i32,
    pub v_format: i32,
    pub k_transform: i32,
    pub v_transform: i32,
    pub reserved: i32,
}

const _: [(); 104] = [(); core::mem::size_of::<W34KvLayerLayout>()];

#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct W34AttentionBatch {
    pub query_start_loc: *const i32,
    pub context_lens_before: *const i32,
    pub seq_lens_after: *const i32,
    pub block_tables: *const i32,
    pub write_slot_mapping: *const i32,
    pub valid_token_mask: *const u8,
    pub valid_req_mask: *const u8,
    pub num_reqs: i32,
    pub total_tokens: i32,
    pub max_blocks_per_req: i32,
    pub max_q_len: i32,
    pub max_seq_len: i32,
    pub mask_mode: i32,
    pub sliding_window: i32,
    pub sm_scale: f32,
    pub reserved: i32,
}

const _: [(); 96] = [(); core::mem::size_of::<W34AttentionBatch>()];

#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct W34GdnBatch {
    pub query_start_loc: *const i32,
    pub state_in_indices: *const i32,
    pub state_out_indices: *const i32,
    pub spec_state_indices: *const i32,
    pub checkpoint_dst_indices: *const i32,
    pub checkpoint_token_offsets: *const i32,
    pub has_initial_state: *const u8,
    pub valid_req_mask: *const u8,
    pub num_reqs: i32,
    pub total_tokens: i32,
    pub max_verify_tokens: i32,
    pub max_checkpoints_per_req: i32,
}

const _: [(); 80] = [(); core::mem::size_of::<W34GdnBatch>()];

#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct W34DeviceProps {
    pub device_id: i32,
    pub compute_major: i32,
    pub compute_minor: i32,
    pub multi_processor_count: i32,
    pub max_shared_mem_per_block_optin: i32,
    pub max_shared_mem_per_multiprocessor: i32,
    pub l2_cache_bytes: i32,
    pub persisting_l2_max_bytes: i32,
    pub name: [u8; 64],
}

impl Default for W34DeviceProps {
    fn default() -> Self {
        Self {
            device_id: 0,
            compute_major: 0,
            compute_minor: 0,
            multi_processor_count: 0,
            max_shared_mem_per_block_optin: 0,
            max_shared_mem_per_multiprocessor: 0,
            l2_cache_bytes: 0,
            persisting_l2_max_bytes: 0,
            name: [0u8; 64],
        }
    }
}

const _: [(); 96] = [(); core::mem::size_of::<W34DeviceProps>()];

pub const W34_MAX_TOP_LOGPROBS: usize = 20;

#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct W34SamplerRowCfg {
    pub temperature: f32,
    pub top_k: i32,
    pub top_p: f32,
    pub min_p: f32,
    pub repetition_penalty: f32,
    pub frequency_penalty: f32,
    pub presence_penalty: f32,
    pub uniform_sample: f32,
    pub recovery_uniform: f32,
    pub draft_token_id: i32,
    pub has_bitmask: i32,
    pub row_bitmask_word_offset: i32,
    pub sparse_offset: i32,
    pub sparse_count: i32,
    pub top_logprobs_k: i32,
    pub use_token_id_cdf: i32,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct W34SamplerSparseEntry {
    pub token_id: i32,
    pub output_count: u32,
    pub logit_bias: f32,
    pub in_prompt: u16,
    pub banned: u16,
}

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct W34SamplerRowOut {
    pub sampled_id: i32,
    pub draft_accepted: i32,
    pub status: i32,
    pub sampled_logprob: f32,
    pub draft_prob: f32,
    pub reserved: i32,
    pub top_ids: [i32; W34_MAX_TOP_LOGPROBS],
    pub top_logprobs: [f32; W34_MAX_TOP_LOGPROBS],
}

impl Default for W34SamplerRowOut {
    fn default() -> Self {
        Self {
            sampled_id: -1,
            draft_accepted: 0,
            status: 0,
            sampled_logprob: 0.0,
            draft_prob: 0.0,
            reserved: 0,
            top_ids: [-1; W34_MAX_TOP_LOGPROBS],
            top_logprobs: [f32::NEG_INFINITY; W34_MAX_TOP_LOGPROBS],
        }
    }
}

const _: [(); 64] = [(); core::mem::size_of::<W34SamplerRowCfg>()];
const _: [(); 16] = [(); core::mem::size_of::<W34SamplerSparseEntry>()];
const _: [(); 184] = [(); core::mem::size_of::<W34SamplerRowOut>()];

#[cfg(feature = "cuda")]
unsafe extern "C" {
    pub fn w34_cuda_error_string(code: i32) -> *const core::ffi::c_char;
    pub fn w34_cublas_runtime_version(out_version: *mut i32) -> i32;
    pub fn w34_get_device_count(out_count: *mut i32) -> i32;
    pub fn w34_get_device_props(device_id: i32, out_props: *mut W34DeviceProps) -> i32;
    pub fn w34_get_current_device_props(out_props: *mut W34DeviceProps) -> i32;
    pub fn w34_get_tp2_min_sm_count(out_min_sms: *mut i32) -> i32;
    pub fn w34_set_device(device_id: i32) -> i32;
    pub fn w34_device_synchronize() -> i32;
    pub fn w34_get_mem_info(out_free: *mut usize, out_total: *mut usize) -> i32;
    pub fn w34_malloc(out_ptr: *mut *mut c_void, bytes: usize) -> i32;
    pub fn w34_free(ptr: *mut c_void) -> i32;
    pub fn w34_host_alloc_mapped(
        out_host_ptr: *mut *mut c_void,
        out_dev_ptr: *mut *mut c_void,
        bytes: usize,
    ) -> i32;
    pub fn w34_host_free(host_ptr: *mut c_void) -> i32;
    pub fn w34_memcpy_h2d_async(
        dst: *mut c_void,
        src: *const c_void,
        bytes: usize,
        stream: usize,
    ) -> i32;
    pub fn w34_memcpy_d2h_async(
        dst: *mut c_void,
        src: *const c_void,
        bytes: usize,
        stream: usize,
    ) -> i32;
    pub fn w34_memcpy_d2d_async(
        dst: *mut c_void,
        src: *const c_void,
        bytes: usize,
        stream: usize,
    ) -> i32;
    pub fn w34_memset_async(dst: *mut c_void, value: i32, bytes: usize, stream: usize) -> i32;

    pub fn w34_stream_create(out_stream: *mut usize) -> i32;
    pub fn w34_stream_destroy(stream: usize) -> i32;
    pub fn w34_stream_synchronize(stream: usize) -> i32;
    pub fn w34_stream_set_persisting_l2_window(
        stream: usize,
        base_ptr: *const c_void,
        num_bytes: usize,
    ) -> i32;

    pub fn w34_event_create(out_event: *mut usize, enable_timing: i32) -> i32;
    pub fn w34_event_destroy(event: usize) -> i32;
    pub fn w34_event_record(event: usize, stream: usize) -> i32;
    pub fn w34_stream_wait_event(stream: usize, event: usize) -> i32;
    pub fn w34_event_synchronize(event: usize) -> i32;
    pub fn w34_event_elapsed_ms(out_ms: *mut f32, start_event: usize, end_event: usize) -> i32;

    pub fn w34_graph_begin_capture(stream: usize) -> i32;
    pub fn w34_graph_end_capture(
        stream: usize,
        out_graph: *mut usize,
        out_graph_exec: *mut usize,
    ) -> i32;
    pub fn w34_graph_launch(graph_exec: usize, stream: usize) -> i32;
    pub fn w34_graph_destroy(graph: usize, graph_exec: usize) -> i32;

    pub fn w34_cublas_create(out_handle: *mut usize, stream: usize) -> i32;
    pub fn w34_cublas_destroy(handle: usize) -> i32;
    pub fn w34_cublas_gemm_f16_bias(
        handle: usize,
        a: *const c_void,
        b: *const c_void,
        bias: *const c_void,
        c: *mut c_void,
        m: i32,
        n: i32,
        k: i32,
    ) -> i32;
    pub fn w34_cublas_gemm_f16(
        handle: usize,
        a: *const c_void,
        b: *const c_void,
        c: *mut c_void,
        m: i32,
        n: i32,
        k: i32,
    ) -> i32;
    pub fn w34_cublas_gemm_bf16(
        handle: usize,
        a: *const c_void,
        b: *const c_void,
        c: *mut c_void,
        m: i32,
        n: i32,
        k: i32,
    ) -> i32;

    pub fn w34_embed_lookup_f16(
        ex: *const W34Exec,
        token_ids: *const i32,
        embed_weight: *const c_void,
        out_hidden: *mut c_void,
        valid_token_mask: *const u8,
        num_tokens: i32,
        hidden_size: i32,
        vocab_size: i32,
    ) -> i32;

    pub fn w34_copy_masked_rows_f16(
        ex: *const W34Exec,
        src: *const c_void,
        dst: *mut c_void,
        valid_token_mask: *const u8,
        num_tokens: i32,
        row_elems: i32,
    ) -> i32;

    pub fn w34_gemma_rms_norm_f16(
        ex: *const W34Exec,
        x: *const c_void,
        weight: *const c_void,
        out: *mut c_void,
        valid_token_mask: *const u8,
        num_tokens: i32,
        hidden_size: i32,
        eps: f32,
    ) -> i32;

    pub fn w34_fused_add_gemma_rms_norm_f16(
        ex: *const W34Exec,
        x: *const c_void,
        residual: *mut c_void,
        weight: *const c_void,
        out: *mut c_void,
        valid_token_mask: *const u8,
        num_tokens: i32,
        hidden_size: i32,
        eps: f32,
    ) -> i32;

    pub fn w34_fused_add_gemma_rms_norm_tap_f16(
        ex: *const W34Exec,
        x: *const c_void,
        residual: *mut c_void,
        weight: *const c_void,
        out: *mut c_void,
        tap_dst_bf16: *mut c_void,
        valid_token_mask: *const u8,
        num_tokens: i32,
        hidden_size: i32,
        eps: f32,
        tap_row_stride: i32,
        tap_col_offset: i32,
    ) -> i32;

    pub fn w34_rms_norm_gated_silu_f16(
        ex: *const W34Exec,
        x: *const c_void,
        z: *const c_void,
        weight: *const c_void,
        out: *mut c_void,
        valid_token_mask: *const u8,
        num_tokens: i32,
        num_v_heads: i32,
        head_dim: i32,
        eps: f32,
    ) -> i32;

    pub fn w34_rms_norm_gated_silu_strided_f16(
        ex: *const W34Exec,
        x: *const c_void,
        z: *const c_void,
        weight: *const c_void,
        out: *mut c_void,
        valid_token_mask: *const u8,
        num_tokens: i32,
        num_v_heads: i32,
        head_dim: i32,
        z_token_stride: i32,
        eps: f32,
    ) -> i32;

    pub fn w34_rms_norm_gated_sigmoid_strided_f16(
        ex: *const W34Exec,
        x: *const c_void,
        z: *const c_void,
        weight: *const c_void,
        out: *mut c_void,
        valid_token_mask: *const u8,
        num_tokens: i32,
        num_v_heads: i32,
        head_dim: i32,
        z_token_stride: i32,
        eps: f32,
    ) -> i32;

    pub fn w34_fused_qk_rmsnorm_rope_gate_f16(
        ex: *const W34Exec,
        q_gate_in: *const c_void,
        k_in: *const c_void,
        q_norm_w: *const c_void,
        k_norm_w: *const c_void,
        positions: *const i32,
        valid_token_mask: *const u8,
        q_out: *mut c_void,
        k_out: *mut c_void,
        gate_out: *mut c_void,
        num_tokens: i32,
        pos_plane_stride: i32,
        num_q_heads: i32,
        num_kv_heads: i32,
        head_dim: i32,
        rotary_dim: i32,
        mrope_s0: i32,
        mrope_s1: i32,
        mrope_s2: i32,
        rope_theta: f32,
        rms_eps: f32,
    ) -> i32;

    pub fn w34_fused_qk_rmsnorm_rope_gate_strided_f16(
        ex: *const W34Exec,
        q_gate_in: *const c_void,
        k_in: *const c_void,
        q_norm_w: *const c_void,
        k_norm_w: *const c_void,
        positions: *const i32,
        valid_token_mask: *const u8,
        q_out: *mut c_void,
        k_out: *mut c_void,
        gate_out: *mut c_void,
        num_tokens: i32,
        pos_plane_stride: i32,
        num_q_heads: i32,
        num_kv_heads: i32,
        head_dim: i32,
        rotary_dim: i32,
        mrope_s0: i32,
        mrope_s1: i32,
        mrope_s2: i32,
        q_gate_row_stride: i32,
        k_row_stride: i32,
        rope_theta: f32,
        rms_eps: f32,
    ) -> i32;

    pub fn w34_silu_and_mul_f16(
        ex: *const W34Exec,
        gate_up: *const c_void,
        out: *mut c_void,
        valid_token_mask: *const u8,
        num_tokens: i32,
        intermediate_size: i32,
    ) -> i32;

    pub fn w34_silu_and_mul_split_f16(
        ex: *const W34Exec,
        gate: *const c_void,
        up: *const c_void,
        out: *mut c_void,
        valid_token_mask: *const u8,
        num_tokens: i32,
        intermediate_size: i32,
    ) -> i32;

    pub fn w34_apply_attn_output_gate_f16(
        ex: *const W34Exec,
        attn_out: *const c_void,
        gate: *const c_void,
        out: *mut c_void,
        valid_token_mask: *const u8,
        num_tokens: i32,
        total_dim: i32,
    ) -> i32;

    pub fn w34_split_cols_2_f16(
        ex: *const W34Exec,
        src: *const c_void,
        dst0: *mut c_void,
        dst1: *mut c_void,
        num_tokens: i32,
        dim0: i32,
        dim1: i32,
    ) -> i32;

    pub fn w34_split_cols_3_f16(
        ex: *const W34Exec,
        src: *const c_void,
        dst0: *mut c_void,
        dst1: *mut c_void,
        dst2: *mut c_void,
        num_tokens: i32,
        dim0: i32,
        dim1: i32,
        dim2: i32,
    ) -> i32;

    pub fn w34_small_n_gemm_f16(
        ex: *const W34Exec,
        x: *const c_void,
        w: *const c_void,
        out: *mut c_void,
        m: i32,
        size_n: i32,
        size_k: i32,
    ) -> i32;
    pub fn w34_small_n_gemm_dual_f16(
        ex: *const W34Exec,
        x: *const c_void,
        w0: *const c_void,
        out0: *mut c_void,
        size_n0: i32,
        w1: *const c_void,
        out1: *mut c_void,
        size_n1: i32,
        m: i32,
        size_k: i32,
    ) -> i32;

    pub fn w34_zero_i32(ex: *const W34Exec, ptr: *mut c_void, count: i64) -> i32;
    pub fn w34_marlin_repack_int4(
        ex: *const W34Exec,
        src_n_major: *const u32,
        dst_marlin: *mut u32,
        k: i64,
        n: i64,
    ) -> i32;
    pub fn w34_marlin_repack_int8(
        ex: *const W34Exec,
        src_n_major: *const u32,
        dst_marlin: *mut u32,
        k: i64,
        n: i64,
    ) -> i32;
    pub fn w34_marlin_gemm_f16u4(
        ex: *const W34Exec,
        a: *const c_void,
        b: *const c_void,
        scales: *const c_void,
        c: *mut c_void,
        workspace: *mut i32,
        m: i64,
        n: i64,
        k: i64,
        workspace_ints: i64,
        group_blocks: i32,
    ) -> i32;
    pub fn w34_marlin_gemm_f16u4_zp(
        ex: *const W34Exec,
        a: *const c_void,
        b: *const c_void,
        scales: *const c_void,
        zeros: *const c_void,
        c: *mut c_void,
        workspace: *mut i32,
        m: i64,
        n: i64,
        k: i64,
        workspace_ints: i64,
        group_blocks: i32,
    ) -> i32;
    pub fn w34_marlin_gemm_f16u8(
        ex: *const W34Exec,
        a: *const c_void,
        b: *const c_void,
        scales: *const c_void,
        c: *mut c_void,
        workspace: *mut i32,
        m: i64,
        n: i64,
        k: i64,
        workspace_ints: i64,
        group_blocks: i32,
    ) -> i32;

    pub fn w34_kv_cache_write(
        ex: *const W34Exec,
        layout: *const W34KvLayerLayout,
        batch: *const W34AttentionBatch,
        k_in: *const c_void,
        v_in: *const c_void,
    ) -> i32;

    pub fn w34_kv_cache_write_strided(
        ex: *const W34Exec,
        layout: *const W34KvLayerLayout,
        batch: *const W34AttentionBatch,
        k_in: *const c_void,
        v_in: *const c_void,
        k_row_stride: i32,
        v_row_stride: i32,
    ) -> i32;

    pub fn w34_kv_cache_write_bf16(
        ex: *const W34Exec,
        layout: *const W34KvLayerLayout,
        batch: *const W34AttentionBatch,
        k_in: *const c_void,
        v_in: *const c_void,
    ) -> i32;
    pub fn w34_kv_cache_write_strided_bf16(
        ex: *const W34Exec,
        layout: *const W34KvLayerLayout,
        batch: *const W34AttentionBatch,
        k_in: *const c_void,
        v_in: *const c_void,
        row_stride_elems: i32,
    ) -> i32;
    pub fn w34_paged_attention_bf16(
        ex: *const W34Exec,
        layout: *const W34KvLayerLayout,
        batch: *const W34AttentionBatch,
        q_in: *const c_void,
        attn_out: *mut c_void,
        num_q_heads: i32,
    ) -> i32;

    pub fn w34_paged_attention_f16(
        ex: *const W34Exec,
        layout: *const W34KvLayerLayout,
        batch: *const W34AttentionBatch,
        q_in: *const c_void,
        attn_out: *mut c_void,
        num_q_heads: i32,
    ) -> i32;

    pub fn w34_paged_attention_f16_force_kernel_for_test(
        ex: *const W34Exec,
        layout: *const W34KvLayerLayout,
        batch: *const W34AttentionBatch,
        q_in: *const c_void,
        attn_out: *mut c_void,
        num_q_heads: i32,
        use_64q: i32,
    ) -> i32;

    pub fn w34_gdn_causal_conv1d_f16(
        ex: *const W34Exec,
        batch: *const W34GdnBatch,
        x_in: *const c_void,
        conv_weight: *const c_void,
        conv_bias: *const c_void,
        conv_state_pool: *mut c_void,
        x_out: *mut c_void,
        conv_dim: i32,
        kernel_size: i32,
        total_state_slots: i32,
        silu_activation: i32,
    ) -> i32;

    pub fn w34_gdn_causal_conv1d_strided_f16(
        ex: *const W34Exec,
        batch: *const W34GdnBatch,
        x_in: *const c_void,
        conv_weight: *const c_void,
        conv_bias: *const c_void,
        conv_state_pool: *mut c_void,
        x_out: *mut c_void,
        conv_dim: i32,
        x_in_row_stride: i32,
        kernel_size: i32,
        total_state_slots: i32,
        silu_activation: i32,
    ) -> i32;

    pub fn w34_gdn_recurrent_f16(
        ex: *const W34Exec,
        batch: *const W34GdnBatch,
        qkv_conv_out: *const c_void,
        q_in: *const c_void,
        k_in: *const c_void,
        v_in: *const c_void,
        b_in: *const c_void,
        a_in: *const c_void,
        a_log: *const f32,
        dt_bias: *const f32,
        ssm_state_pool: *mut f32,
        out_f16: *mut c_void,
        num_k_heads: i32,
        head_k_dim: i32,
        num_v_heads: i32,
        head_v_dim: i32,
        total_state_slots: i32,
    ) -> i32;

    pub fn w34_gdn_recurrent_strided_f16(
        ex: *const W34Exec,
        batch: *const W34GdnBatch,
        qkv_conv_out: *const c_void,
        q_in: *const c_void,
        k_in: *const c_void,
        v_in: *const c_void,
        b_in: *const c_void,
        a_in: *const c_void,
        a_log: *const f32,
        dt_bias: *const f32,
        ssm_state_pool: *mut f32,
        out_f16: *mut c_void,
        num_k_heads: i32,
        head_k_dim: i32,
        num_v_heads: i32,
        head_v_dim: i32,
        ba_row_stride: i32,
        total_state_slots: i32,
    ) -> i32;

    pub fn w34_tp2_signal_bytes() -> usize;
    pub fn w34_tp2_all_reduce_f16(
        ex: *const W34Exec,
        local_in: *const c_void,
        out: *mut c_void,
        stage0: *mut c_void,
        stage1: *mut c_void,
        sig0: *mut c_void,
        sig1: *mut c_void,
        rank: i32,
        num_elems: i32,
    ) -> i32;
    pub fn w34_tp2_all_reduce_residual_gemma_rms_norm_f16(
        ex: *const W34Exec,
        local_in: *const c_void,
        ar_out: *mut c_void,
        residual: *mut c_void,
        weight: *const c_void,
        out: *mut c_void,
        tap_dst_bf16: *mut c_void,
        stage0: *mut c_void,
        stage1: *mut c_void,
        sig0: *mut c_void,
        sig1: *mut c_void,
        rank: i32,
        num_tokens: i32,
        hidden_size: i32,
        eps: f32,
        tap_row_stride: i32,
        tap_col_offset: i32,
    ) -> i32;
    pub fn w34_tp2_all_reduce_bf16(
        ex: *const W34Exec,
        local_in: *const c_void,
        out: *mut c_void,
        stage0: *mut c_void,
        stage1: *mut c_void,
        sig0: *mut c_void,
        sig1: *mut c_void,
        rank: i32,
        num_elems: i32,
    ) -> i32;
    pub fn w34_tp2_all_reduce_f32(
        ex: *const W34Exec,
        local_in: *const f32,
        out: *mut f32,
        stage0: *mut f32,
        stage1: *mut f32,
        sig0: *mut c_void,
        sig1: *mut c_void,
        rank: i32,
        num_elems: i32,
    ) -> i32;
    pub fn w34_tp2_broadcast_i32(
        ex: *const W34Exec,
        src_rank0: *const i32,
        out: *mut i32,
        stage0: *mut i32,
        stage1: *mut i32,
        sig0: *mut c_void,
        sig1: *mut c_void,
        rank: i32,
        num_elems: i32,
    ) -> i32;
    pub fn w34_tp2_all_gather_top1_f32_i32(
        ex: *const W34Exec,
        local_max_val: *const f32,
        local_max_id: *const i32,
        global_max_val: *mut f32,
        global_max_id: *mut i32,
        stage0: *mut c_void,
        stage1: *mut c_void,
        sig0: *mut c_void,
        sig1: *mut c_void,
        rank: i32,
        num_rows: i32,
    ) -> i32;
    pub fn w34_tp2_gather_vocab_f16(
        ex: *const W34Exec,
        local_logits_f16: *const c_void,
        rank0_full_logits_f16: *mut c_void,
        stage0: *mut c_void,
        stage1: *mut c_void,
        sig0: *mut c_void,
        sig1: *mut c_void,
        rank: i32,
        num_rows: i32,
        local_vocab_size: i32,
    ) -> i32;
    pub fn w34_tp2_warmup() -> i32;
    pub fn w34_device_can_enable_p2p(
        dev_a: i32,
        dev_b: i32,
        require_native_atomic: i32,
        out_can_p2p: *mut i32,
    ) -> i32;
    pub fn w34_tp4_all_reduce_f16(
        ex: *const W34Exec,
        local_in: *const c_void,
        out: *mut c_void,
        stage0: *mut c_void,
        stage1: *mut c_void,
        stage2: *mut c_void,
        stage3: *mut c_void,
        sig0: *mut c_void,
        sig1: *mut c_void,
        sig2: *mut c_void,
        sig3: *mut c_void,
        rank: i32,
        num_elems: i32,
    ) -> i32;
    pub fn w34_tp4_all_reduce_residual_gemma_rms_norm_f16(
        ex: *const W34Exec,
        local_in: *const c_void,
        ar_out: *mut c_void,
        residual: *mut c_void,
        weight: *const c_void,
        out: *mut c_void,
        tap_dst_bf16: *mut c_void,
        stage0: *mut c_void,
        stage1: *mut c_void,
        stage2: *mut c_void,
        stage3: *mut c_void,
        sig0: *mut c_void,
        sig1: *mut c_void,
        sig2: *mut c_void,
        sig3: *mut c_void,
        rank: i32,
        num_tokens: i32,
        hidden_size: i32,
        eps: f32,
        tap_row_stride: i32,
        tap_col_offset: i32,
    ) -> i32;
    pub fn w34_tp4_all_reduce_bf16(
        ex: *const W34Exec,
        local_in: *const c_void,
        out: *mut c_void,
        stage0: *mut c_void,
        stage1: *mut c_void,
        stage2: *mut c_void,
        stage3: *mut c_void,
        sig0: *mut c_void,
        sig1: *mut c_void,
        sig2: *mut c_void,
        sig3: *mut c_void,
        rank: i32,
        num_elems: i32,
    ) -> i32;
    pub fn w34_tp4_all_reduce_f32(
        ex: *const W34Exec,
        local_in: *const f32,
        out: *mut f32,
        stage0: *mut f32,
        stage1: *mut f32,
        stage2: *mut f32,
        stage3: *mut f32,
        sig0: *mut c_void,
        sig1: *mut c_void,
        sig2: *mut c_void,
        sig3: *mut c_void,
        rank: i32,
        num_elems: i32,
    ) -> i32;
    pub fn w34_tp4_broadcast_i32(
        ex: *const W34Exec,
        src_rank0: *const i32,
        out: *mut i32,
        stage0: *mut i32,
        stage1: *mut i32,
        stage2: *mut i32,
        stage3: *mut i32,
        sig0: *mut c_void,
        sig1: *mut c_void,
        sig2: *mut c_void,
        sig3: *mut c_void,
        rank: i32,
        num_elems: i32,
    ) -> i32;
    pub fn w34_tp4_all_gather_top1_f32_i32(
        ex: *const W34Exec,
        local_max_val: *const f32,
        local_max_id: *const i32,
        global_max_val: *mut f32,
        global_max_id: *mut i32,
        stage0: *mut c_void,
        stage1: *mut c_void,
        stage2: *mut c_void,
        stage3: *mut c_void,
        sig0: *mut c_void,
        sig1: *mut c_void,
        sig2: *mut c_void,
        sig3: *mut c_void,
        rank: i32,
        num_rows: i32,
    ) -> i32;
    pub fn w34_tp4_gather_vocab_f16(
        ex: *const W34Exec,
        local_logits_f16: *const c_void,
        rank0_full_logits_f16: *mut c_void,
        stage0: *mut c_void,
        stage1: *mut c_void,
        stage2: *mut c_void,
        stage3: *mut c_void,
        sig0: *mut c_void,
        sig1: *mut c_void,
        sig2: *mut c_void,
        sig3: *mut c_void,
        rank: i32,
        num_rows: i32,
        local_vocab_size: i32,
    ) -> i32;

    pub fn w34_sampler_greedy_top5_f16(
        ex: *const W34Exec,
        logits_f16: *const c_void,
        out_greedy_ids: *mut i32,
        out_top5_ids: *mut i32,
        out_top5_logprobs: *mut f32,
        num_rows: i32,
        local_vocab_size: i32,
        vocab_offset: i32,
    ) -> i32;
    pub fn w34_sampler_step_f16(
        ex: *const W34Exec,
        logits_f16: *const c_void,
        row_cfgs: *const W34SamplerRowCfg,
        bitmask_u32: *const u32,
        sparse_entries: *const W34SamplerSparseEntry,
        row_outs: *mut W34SamplerRowOut,
        num_rows: i32,
        vocab_size: i32,
        total_sparse_entries: i32,
    ) -> i32;
    pub fn w34_sampler_step_fast_greedy_f16(
        ex: *const W34Exec,
        logits_f16: *const c_void,
        row_cfgs: *const W34SamplerRowCfg,
        sparse_entries: *const W34SamplerSparseEntry,
        row_outs: *mut W34SamplerRowOut,
        num_rows: i32,
        vocab_size: i32,
    ) -> i32;
    pub fn w34_sampler_step_fast_greedy_masked_f16(
        ex: *const W34Exec,
        logits_f16: *const c_void,
        row_cfgs: *const W34SamplerRowCfg,
        bitmask_u32: *const u32,
        sparse_entries: *const W34SamplerSparseEntry,
        row_outs: *mut W34SamplerRowOut,
        num_rows: i32,
        vocab_size: i32,
    ) -> i32;
    pub fn w34_sampler_scatter_ids_to_tokens(
        ex: *const W34Exec,
        row_outs: *const W34SamplerRowOut,
        greedy_ids: *const i32,
        dst_token_ids: *mut i32,
        num_rows: i32,
    ) -> i32;

    pub fn w34_embed_lookup_f16_to_bf16(
        ex: *const W34Exec,
        ids: *const i32,
        weight: *const c_void,
        out: *mut c_void,
        rows: i32,
        hidden: i32,
        vocab: i32,
    ) -> i32;
    pub fn w34_cast_bf16_to_f16(
        ex: *const W34Exec,
        x: *const c_void,
        y: *mut c_void,
        n: i64,
    ) -> i32;
    pub fn w34_dflash_edges_bf16(
        ex: *const W34Exec,
        ids: *const i32,
        unary: *const f32,
        hidden: *const c_void,
        pred: *const c_void,
        succ: *const c_void,
        out: *mut f32,
        anchor: i32,
        k: i32,
        topk: i32,
        rank: i32,
        vocab: i32,
    ) -> i32;

    pub fn w34_rms_norm_bf16(
        ex: *const W34Exec,
        x: *const c_void,
        residual: *mut c_void,
        weight: *const c_void,
        y: *mut c_void,
        rows: i32,
        hidden: i32,
        eps: f32,
    ) -> i32;
    pub fn w34_silu_and_mul_bf16(
        ex: *const W34Exec,
        gate_up: *const c_void,
        y: *mut c_void,
        rows: i32,
        hidden: i32,
    ) -> i32;
    pub fn w34_dflash_conv_bf16(
        ex: *const W34Exec,
        x: *const c_void,
        delta: *const c_void,
        base: *const c_void,
        y: *mut c_void,
        rows: i32,
        hidden: i32,
        block: i32,
        group: i32,
        taps: i32,
        side: i32,
    ) -> i32;
    pub fn w34_dflash_qk_norm_rope_bf16_f32_table(
        ex: *const W34Exec,
        qkv: *mut c_void,
        positions: *const i32,
        qw: *const c_void,
        kw: *const c_void,
        table: *const c_void,
        rows: i32,
        q_heads: i32,
        kv_heads: i32,
        dim: i32,
        table_rows: i32,
        eps: f32,
    ) -> i32;
    pub fn w34_dflash_qk_norm_rope_bf16_f32_table_qout(
        ex: *const W34Exec,
        qkv: *mut c_void,
        q_out: *mut c_void,
        positions: *const i32,
        qw: *const c_void,
        kw: *const c_void,
        table: *const c_void,
        rows: i32,
        q_heads: i32,
        kv_heads: i32,
        dim: i32,
        table_rows: i32,
        eps: f32,
    ) -> i32;
    pub fn w34_dflash_qk_norm_rope_kv_write_bf16_f32_table(
        ex: *const W34Exec,
        qkv: *mut c_void,
        q_out: *mut c_void,
        positions: *const i32,
        qw: *const c_void,
        kw: *const c_void,
        table: *const c_void,
        kv_layout: *const W34KvLayerLayout,
        write_slot_mapping: *const i32,
        rows: i32,
        q_heads: i32,
        kv_heads: i32,
        dim: i32,
        table_rows: i32,
        eps: f32,
    ) -> i32;
    pub fn w34_dflash_select_greedy_bf16(
        ex: *const W34Exec,
        ids: *const i32,
        unary: *const f32,
        hidden: *const c_void,
        pred: *const c_void,
        succ: *const c_void,
        out_path: *mut i32,
        anchor: i32,
        k: i32,
        topk: i32,
        rank: i32,
        vocab: i32,
    ) -> i32;
    pub fn w34_dflash_select_greedy_dev_anchor_bf16(
        ex: *const W34Exec,
        ids: *const i32,
        unary: *const f32,
        hidden: *const c_void,
        pred: *const c_void,
        succ: *const c_void,
        out_path: *mut i32,
        anchor_dev: *const i32,
        k: i32,
        topk: i32,
        rank: i32,
        vocab: i32,
    ) -> i32;
    pub fn w34_dflash_qk_norm_rope_bf16(
        ex: *const W34Exec,
        qkv: *mut c_void,
        positions: *const i32,
        qw: *const c_void,
        kw: *const c_void,
        table: *const c_void,
        rows: i32,
        q_heads: i32,
        kv_heads: i32,
        dim: i32,
        table_rows: i32,
        eps: f32,
    ) -> i32;
    pub fn w34_copy_cols_bf16(
        ex: *const W34Exec,
        src: *const c_void,
        dst: *mut c_void,
        rows: i32,
        src_stride: i32,
        dst_stride: i32,
        src_col: i32,
        dst_col: i32,
        cols: i32,
    ) -> i32;

    pub fn w34_rejection_greedy(
        ex: *const W34Exec,
        drafts: *const i32,
        targets: *const i32,
        valid: *const u8,
        emitted: *mut i32,
        accepted: *mut i32,
        status: *mut i32,
        num_reqs: i32,
        k: i32,
    ) -> i32;

    pub fn w34_rejection_stochastic(
        ex: *const W34Exec,
        drafts: *const i32,
        target: *const f32,
        proposal: *const f32,
        uniforms: *const f64,
        recovery: *const f64,
        valid: *const u8,
        emitted: *mut i32,
        accepted: *mut i32,
        status: *mut i32,
        num_reqs: i32,
        k: i32,
        vocab: i32,
    ) -> i32;

    pub fn w34_concat_cols_2_f16(
        ex: *const W34Exec,
        src0: *const c_void,
        src1: *const c_void,
        dst: *mut c_void,
        num_tokens: i32,
        dim0: i32,
        dim1: i32,
    ) -> i32;

    pub fn w34_tap_add_f16_to_bf16(
        ex: *const W34Exec,
        res_f16: *const c_void,
        mlp_f16: *const c_void,
        dst_bf16: *mut c_void,
        num_tokens: i32,
        hidden_size: i32,
        dst_row_stride: i32,
        dst_col_offset: i32,
    ) -> i32;

    pub fn w34_topk_rows_f16(
        ex: *const W34Exec,
        logits_f16: *const c_void,
        out_ids: *mut i32,
        out_vals: *mut f32,
        num_rows: i32,
        local_vocab_size: i32,
        vocab_offset: i32,
        top_k: i32,
    ) -> i32;

    pub fn w34_topk_rows_split_f16(
        ex: *const W34Exec,
        logits_f16: *const c_void,
        split_ids_ws: *mut i32,
        split_vals_ws: *mut f32,
        out_ids: *mut i32,
        out_vals: *mut f32,
        num_rows: i32,
        local_vocab_size: i32,
        vocab_offset: i32,
        top_k: i32,
        num_splits: i32,
    ) -> i32;

    pub fn w34_vision_bias_add_f16(
        ex: *const W34Exec,
        x: *const c_void,
        bias: *const c_void,
        residual: *const c_void,
        out: *mut c_void,
        rows: i32,
        cols: i32,
    ) -> i32;

    pub fn w34_vision_layernorm_f16(
        ex: *const W34Exec,
        x: *const c_void,
        weight: *const c_void,
        bias: *const c_void,
        out: *mut c_void,
        rows: i32,
        cols: i32,
        eps: f32,
    ) -> i32;

    pub fn w34_vision_bias_gelu_f16(
        ex: *const W34Exec,
        x: *const c_void,
        bias: *const c_void,
        out: *mut c_void,
        rows: i32,
        cols: i32,
        use_tanh: i32,
    ) -> i32;

    pub fn w34_vision_qkv_rope_attn_f16(
        ex: *const W34Exec,
        qkv: *mut c_void,
        qkv_bias: *const c_void,
        pos_hw: *const i32,
        cu_seqlens: *const i32,
        out: *mut c_void,
        total_patches: i32,
        num_images: i32,
        num_heads: i32,
        head_dim: i32,
        rope_theta: f32,
    ) -> i32;

    pub fn w34_embedding_splice_f16(
        ex: *const W34Exec,
        dst_embeds: *mut c_void,
        src_visual_embeds: *const c_void,
        token_indices: *const i32,
        num_visual_tokens: i32,
        hidden_size: i32,
    ) -> i32;

    pub fn w34_moe_topk_router_f16(
        ex: *const W34Exec,
        router_logits: *const c_void,
        topk_weights: *mut f32,
        topk_ids: *mut i32,
        num_tokens: i32,
        num_experts: i32,
        top_k: i32,
        norm_topk_prob: i32,
        scoring_func: i32,
    ) -> i32;

    pub fn w34_moe_w4a16_gate_up_silu_f16(
        ex: *const W34Exec,
        x: *const c_void,
        gate_up_packed: *const u32,
        gate_up_scales: *const c_void,
        topk_ids: *const i32,
        interm_out: *mut c_void,
        num_tokens: i32,
        top_k: i32,
        num_experts: i32,
        n_interm: i32,
        k_hidden: i32,
        group_size: i32,
    ) -> i32;

    pub fn w34_moe_w4a16_down_accum_f16(
        ex: *const W34Exec,
        interm_in: *const c_void,
        down_packed: *const u32,
        down_scales: *const c_void,
        topk_ids: *const i32,
        topk_weights: *const f32,
        shared_expert_out: *const c_void,
        shared_expert_gate_logits: *const c_void,
        out: *mut c_void,
        num_tokens: i32,
        top_k: i32,
        num_experts: i32,
        k_hidden: i32,
        n_interm: i32,
        group_size: i32,
    ) -> i32;

    pub fn w34_moe_w4a16_mlp_f16(
        ex: *const W34Exec,
        x: *const c_void,
        gate_up_packed: *const u32,
        gate_up_scales: *const c_void,
        down_packed: *const u32,
        down_scales: *const c_void,
        topk_ids: *const i32,
        topk_weights: *const f32,
        shared_expert_out: *const c_void,
        shared_expert_gate_logits: *const c_void,
        interm_buf: *mut c_void,
        out: *mut c_void,
        num_tokens: i32,
        top_k: i32,
        num_experts: i32,
        n_interm: i32,
        k_hidden: i32,
        group_size: i32,
    ) -> i32;

    pub fn w34_moe_w4a16_mlp_zp_f16(
        ex: *const W34Exec,
        x: *const c_void,
        gate_up_packed: *const u32,
        gate_up_scales: *const c_void,
        gate_up_qzeros: *const u32,
        down_packed: *const u32,
        down_scales: *const c_void,
        down_qzeros: *const u32,
        topk_ids: *const i32,
        topk_weights: *const f32,
        shared_expert_out: *const c_void,
        shared_expert_gate_logits: *const c_void,
        interm_buf: *mut c_void,
        out: *mut c_void,
        num_tokens: i32,
        top_k: i32,
        num_experts: i32,
        n_interm: i32,
        k_hidden: i32,
        group_size: i32,
    ) -> i32;

    pub fn w34_moe_f16_gate_up_silu(
        ex: *const W34Exec,
        x: *const c_void,
        gate_up_w: *const c_void,
        topk_ids: *const i32,
        interm_out: *mut c_void,
        num_tokens: i32,
        top_k: i32,
        num_experts: i32,
        n_interm: i32,
        k_hidden: i32,
    ) -> i32;

    pub fn w34_moe_f16_down_accum(
        ex: *const W34Exec,
        interm_in: *const c_void,
        down_w: *const c_void,
        topk_ids: *const i32,
        topk_weights: *const f32,
        shared_expert_out: *const c_void,
        shared_expert_gate_logits: *const c_void,
        out: *mut c_void,
        num_tokens: i32,
        top_k: i32,
        num_experts: i32,
        k_hidden: i32,
        n_interm: i32,
    ) -> i32;

    pub fn w34_clef_accum_spans_f16_to_f32(
        ex: *const W34Exec,
        chunk_normed_f16: *const c_void,
        span_bounds: *const i32,
        span_accum_f32: *mut f32,
        chunk_start: i32,
        chunk_len: i32,
        num_spans: i32,
        hidden_size: i32,
    ) -> i32;

    pub fn w34_clef_cast_f32_to_f16(
        ex: *const W34Exec,
        src_f32: *const f32,
        dst_f16: *mut c_void,
        num_elems: i64,
    ) -> i32;

    pub fn w34_clef_build_option_queries_f16(
        ex: *const W34Exec,
        ctx_proj_f16: *const c_void,
        lex_proj_f16: *const c_void,
        opt_q_proj_f16: *const c_void,
        opt_q_idx: *const i32,
        out_f16: *mut c_void,
        total_opts: i32,
        num_q: i32,
        width: i32,
    ) -> i32;

    pub fn w34_clef_mha_attn_f16(
        ex: *const W34Exec,
        q_f16: *const c_void,
        q_bias_f16: *const c_void,
        kv_f16: *const c_void,
        kv_bias_f16: *const c_void,
        out_f16: *mut c_void,
        q_len: i32,
        m_len: i32,
        heads: i32,
        head_dim: i32,
    ) -> i32;

    pub fn w34_clef_option_summaries_f16(
        ex: *const W34Exec,
        routed_opts_f16: *const c_void,
        base_fields_f16: *const c_void,
        q_opt_offsets: *const i32,
        out_summaries_f16: *mut c_void,
        num_q: i32,
        total_opts: i32,
        width: i32,
    ) -> i32;

    pub fn w34_clef_build_fields_f16(
        ex: *const W34Exec,
        base_fields_f16: *const c_void,
        norm_summaries_f16: *const c_void,
        global_proj_f16: *const c_void,
        type_embed_f16: *const c_void,
        q_types: *const i32,
        out_fields_f16: *mut c_void,
        num_q: i32,
        width: i32,
    ) -> i32;

    pub fn w34_clef_build_residual_features_f16(
        ex: *const W34Exec,
        fields_f16: *const c_void,
        options_normed_f16: *const c_void,
        opt_q_idx: *const i32,
        out_features_f16: *mut c_void,
        total_opts: i32,
        num_q: i32,
        width: i32,
    ) -> i32;

    pub fn w34_hc_init_streams_f16(
        ex: *const W34Exec,
        embed_in: *const c_void,
        residual_out: *mut c_void,
        valid_token_mask: *const u8,
        num_tokens: i32,
        hc_count: i32,
        hidden_size: i32,
    ) -> i32;

    pub fn w34_hc_rmsnorm_f16(
        ex: *const W34Exec,
        residual_in: *const c_void,
        norm_weight: *const c_void,
        normed_out: *mut c_void,
        valid_token_mask: *const u8,
        num_tokens: i32,
        hc_count: i32,
        hidden_size: i32,
        eps: f32,
    ) -> i32;

    pub fn w34_hc_down_silu_inject_f16(
        ex: *const W34Exec,
        down_in: *const c_void,
        down_out: *mut c_void,
        inj_in: *const c_void,
        inject_weights_out: *mut f32,
        valid_token_mask: *const u8,
        num_tokens: i32,
        hc_rank_dim: i32,
        hc_count: i32,
    ) -> i32;

    pub fn w34_hc_mix_streams_f16(
        ex: *const W34Exec,
        residual_in: *const c_void,
        up_raw_in: *const c_void,
        mixed_out: *mut c_void,
        valid_token_mask: *const u8,
        num_tokens: i32,
        hc_count: i32,
        hidden_size: i32,
    ) -> i32;

    pub fn w34_hc_combine_f16(
        ex: *const W34Exec,
        residual_in: *const c_void,
        block_out: *const c_void,
        inject_weights: *const f32,
        ple_delta: *const c_void,
        residual_out: *mut c_void,
        valid_token_mask: *const u8,
        num_tokens: i32,
        hc_count: i32,
        hidden_size: i32,
    ) -> i32;

    pub fn w34_qwen4_mtp_broadcast_add_f16(
        ex: *const W34Exec,
        fc_embed: *const c_void,
        fc_hidden: *const c_void,
        out_multi_hidden: *mut c_void,
        num_tokens: i32,
        hc_count: i32,
        hidden_size: i32,
    ) -> i32;

    pub fn w34_ple_dequant_f8_to_f16(
        ex: *const W34Exec,
        f8_in: *const u8,
        f16_out: *mut c_void,
        valid_token_mask: *const u8,
        num_tokens: i32,
        embed_dim: i32,
        weight_scale: f32,
    ) -> i32;

    pub fn w34_ple_dilated_conv1d_f16(
        ex: *const W34Exec,
        key_in: *const c_void,
        conv_weight: *const c_void,
        conv_state_pool: *mut c_void,
        key_out: *mut c_void,
        batch: *const W34GdnBatch,
        conv_dim: i32,
    ) -> i32;

    pub fn w34_ple_signed_sqrt_gate_f16(
        ex: *const W34Exec,
        norm_query: *const c_void,
        norm_key: *const c_void,
        value_in: *const c_void,
        ple_delta_out: *mut c_void,
        valid_token_mask: *const u8,
        num_tokens: i32,
        hc_count: i32,
        embed_dim: i32,
        eps: f32,
    ) -> i32;

    pub fn w34_qsa_indexer_norm_rope_f16(
        ex: *const W34Exec,
        qk_in: *const c_void,
        q_norm_weight: *const c_void,
        k_norm_weight: *const c_void,
        positions: *const i32,
        q_out: *mut c_void,
        k_out: *mut c_void,
        valid_token_mask: *const u8,
        num_tokens: i32,
        q_heads: i32,
        kv_heads: i32,
        head_dim: i32,
        rotary_dim: i32,
        rope_theta: f32,
        eps: f32,
        mrope_s1: i32,
        mrope_s2: i32,
        pos_plane_stride: i32,
    ) -> i32;

    pub fn w34_qsa_indexer_cache_update_f16(
        ex: *const W34Exec,
        k_in: *const c_void,
        raw_k_cache: *mut c_void,
        comp_k_cache: *mut c_void,
        k_norm_weight: *const c_void,
        positions: *const i32,
        write_slot_mapping: *const i32,
        valid_token_mask: *const u8,
        num_tokens: i32,
        head_dim: i32,
        rotary_dim: i32,
        rope_theta: f32,
        eps: f32,
        mrope_s1: i32,
        mrope_s2: i32,
        pos_plane_stride: i32,
    ) -> i32;

    pub fn w34_qsa_select_blocks_f16(
        ex: *const W34Exec,
        q_indexer: *const c_void,
        comp_k_cache: *const c_void,
        batch: *const W34AttentionBatch,
        out_selected_phys_blocks: *mut i32,
        out_selected_logical_blocks: *mut i32,
        out_num_selected_blocks: *mut i32,
        out_token_seq_pos: *mut i32,
        q_heads: i32,
        head_dim: i32,
        block_size: i32,
        select_k_blocks: i32,
        sink_blocks: i32,
        recent_blocks: i32,
    ) -> i32;

    pub fn w34_qsa_sparse_paged_attn_f16(
        ex: *const W34Exec,
        q: *const c_void,
        gate: *const c_void,
        kv_layout: *const W34KvLayerLayout,
        selected_phys_blocks: *const i32,
        selected_logical_blocks: *const i32,
        num_selected_blocks: *const i32,
        token_seq_pos: *const i32,
        valid_token_mask: *const u8,
        out: *mut c_void,
        num_tokens: i32,
        local_q_heads: i32,
        select_k_blocks: i32,
        sm_scale: f32,
    ) -> i32;

    pub fn w34_moe_unpack_staged_experts(
        ex: *const W34Exec,
        staged_packs: *const c_void,
        dst_slots: *const i32,
        gate_up_packed_base: *mut u32,
        gate_up_scales_base: *mut c_void,
        down_packed_base: *mut u32,
        down_scales_base: *mut c_void,
        num_staged: i32,
        dst_capacity: i32,
        pack_stride_bytes: usize,
        n_interm: i32,
        k_hidden: i32,
        group_size: i32,
    ) -> i32;

    pub fn w34_moe_pull_mapped_experts(
        ex: *const W34Exec,
        mapped_host_packs: *const c_void,
        src_local_experts: *const i32,
        dst_slots: *const i32,
        gate_up_packed_base: *mut u32,
        gate_up_scales_base: *mut c_void,
        down_packed_base: *mut u32,
        down_scales_base: *mut c_void,
        num_staged: i32,
        local_num_experts: i32,
        dst_capacity: i32,
        pack_stride_bytes: usize,
        n_interm: i32,
        k_hidden: i32,
        group_size: i32,
    ) -> i32;

    pub fn w34_moe_remap_topk_ids_i32(
        ex: *const W34Exec,
        global_topk_ids: *const i32,
        expert_to_slot: *const i32,
        remapped_topk_ids: *mut i32,
        total_pairs: i32,
        global_expert_offset: i32,
        local_num_experts: i32,
    ) -> i32;

    pub fn w34_add_inplace_f16(
        ex: *const W34Exec,
        dst_f16: *mut c_void,
        src_f16: *const c_void,
        num_elems: i64,
    ) -> i32;

    pub fn w34_moe_gpu_lru_update_and_pull(
        ex: *const W34Exec,
        global_topk_ids: *const i32,
        expert_to_slot: *mut i32,
        slot_to_expert: *mut i32,
        slot_last_used: *mut u64,
        step_counter: *mut u64,
        remapped_topk_ids: *mut i32,
        miss_local_experts: *mut i32,
        miss_victim_slots: *mut i32,
        num_misses_buf: *mut i32,
        mapped_host_packs: *const c_void,
        gate_up_packed_base: *mut u32,
        gate_up_scales_base: *mut c_void,
        down_packed_base: *mut u32,
        down_scales_base: *mut c_void,
        total_pairs: i32,
        global_expert_offset: i32,
        local_num_experts: i32,
        k_hot: i32,
        max_misses: i32,
        pack_stride_bytes: usize,
        n_interm: i32,
        k_hidden: i32,
        group_size: i32,
    ) -> i32;

    pub fn w34_moe_hybrid_submit_and_pull(
        ex: *const W34Exec,
        layer_id: i32,
        d_req_slot: *mut c_void,
        x_f16: *const c_void,
        global_topk_ids: *const i32,
        topk_weights: *const f32,
        expert_to_slot: *const i32,
        remapped_topk_ids: *mut i32,
        miss_local_experts: *mut i32,
        miss_victim_slots: *mut i32,
        num_misses_buf: *mut i32,
        mapped_host_packs: *const c_void,
        gate_up_packed_base: *mut u32,
        gate_up_scales_base: *mut c_void,
        down_packed_base: *mut u32,
        down_scales_base: *mut c_void,
        num_tokens: i32,
        top_k: i32,
        global_expert_offset: i32,
        local_num_experts: i32,
        k_static: i32,
        k_hot: i32,
        pack_stride_bytes: usize,
        n_interm: i32,
        k_hidden: i32,
        group_size: i32,
        gpu_share_frac: f32,
    ) -> i32;

    pub fn w34_moe_hybrid_wait_and_add(
        ex: *const W34Exec,
        d_req_slot: *const c_void,
        d_resp_slot: *const c_void,
        out_f16: *mut c_void,
        num_misses_buf: *const i32,
        num_tokens: i32,
        k_hidden: i32,
    ) -> i32;
}

/// Returns the linked cuBLAS runtime version (`major * 10000 + minor * 100 + patch`).
#[cfg(feature = "cuda")]
pub fn cublas_runtime_version() -> Result<i32, i32> {
    let mut version = 0i32;
    // SAFETY: `version` is a valid stack-allocated `i32` pointer for the duration of the call.
    let rc = unsafe { w34_cublas_runtime_version(&raw mut version) };
    if rc == 0 { Ok(version) } else { Err(rc) }
}
