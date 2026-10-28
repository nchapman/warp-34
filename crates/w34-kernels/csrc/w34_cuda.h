// warp-34 C FFI contract. Mirrored in crates/w34-kernels/src/ffi.rs.
#pragma once
#include <assert.h>
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

enum {
    W34_DTYPE_F16  = 0,
    W34_DTYPE_BF16 = 1,
    W34_DTYPE_F32  = 2,
    W34_DTYPE_I32  = 3,
    W34_DTYPE_U8   = 4,
    W34_DTYPE_U32  = 5,
};

enum {
    W34_KV_NAMESPACE_TARGET_FULL_ATTN = 0,
    W34_KV_NAMESPACE_MTP_FULL_ATTN    = 1,
    W34_KV_NAMESPACE_DFLASH_DRAFT_CTX = 2,
};

enum {
    W34_KV_FORMAT_FP16 = 0,
    W34_KV_FORMAT_BF16 = 1,
    W34_KV_FORMAT_TQ4  = 2,
    W34_KV_FORMAT_Q8_0 = 3,
};

enum {
    W34_KV_TRANSFORM_IDENTITY          = 0,
    W34_KV_TRANSFORM_RADEMACHER_WHT256 = 1,
};

enum {
    W34_ATTN_MASK_CAUSAL           = 0,
    W34_ATTN_MASK_SYMMETRIC_WINDOW = 1,
};

typedef struct {
    uintptr_t stream;       // cudaStream_t
    uint64_t  scratch_ptr;  // device workspace pointer
    size_t    scratch_size; // workspace capacity in bytes
} w34_exec;

static_assert(sizeof(w34_exec) == 24, "w34_exec ABI size mismatch");

typedef struct {
    uint64_t k_base;
    uint64_t v_base;
    uint64_t k_page_stride_bytes;
    uint64_t k_head_stride_bytes;
    uint64_t k_token_stride_bytes;
    uint64_t v_page_stride_bytes;
    uint64_t v_head_stride_bytes;
    uint64_t v_token_stride_bytes;
    int32_t  namespace_id;
    int32_t  layer_id;
    int32_t  local_kv_heads;
    int32_t  head_dim;
    int32_t  block_size;
    int32_t  k_format;
    int32_t  v_format;
    int32_t  k_transform;
    int32_t  v_transform;
    int32_t  reserved;
} w34_kv_layer_layout;

static_assert(sizeof(w34_kv_layer_layout) == 104, "w34_kv_layer_layout ABI size mismatch");

typedef struct {
    const int32_t* query_start_loc;
    const int32_t* context_lens_before;
    const int32_t* seq_lens_after;
    const int32_t* block_tables;
    const int32_t* write_slot_mapping;
    const uint8_t* valid_token_mask;
    const uint8_t* valid_req_mask;
    int32_t        num_reqs;
    int32_t        total_tokens;
    int32_t        max_blocks_per_req;
    int32_t        max_q_len;
    int32_t        max_seq_len;
    int32_t        mask_mode;
    int32_t        sliding_window;
    float          sm_scale;
    int32_t        reserved;
} w34_attention_batch;

static_assert(sizeof(w34_attention_batch) == 96, "w34_attention_batch ABI size mismatch");

typedef struct {
    const int32_t* query_start_loc;
    const int32_t* state_in_indices;
    const int32_t* state_out_indices;
    const int32_t* spec_state_indices;
    const int32_t* checkpoint_dst_indices;
    const int32_t* checkpoint_token_offsets;
    const uint8_t* has_initial_state;
    const uint8_t* valid_req_mask;
    int32_t        num_reqs;
    int32_t        total_tokens;
    int32_t        max_verify_tokens;
    int32_t        max_checkpoints_per_req;
} w34_gdn_batch;

static_assert(sizeof(w34_gdn_batch) == 80, "w34_gdn_batch ABI size mismatch");

// --- Runtime, Stream, Event, Graph, & cuBLAS primitives (op_init.cu) ---
int w34_cublas_runtime_version(int* out_version);
int w34_get_device_count(int* out_count);
int w34_set_device(int device_id);
int w34_device_synchronize(void);
int w34_get_mem_info(size_t* out_free, size_t* out_total);
int w34_malloc(void** out_ptr, size_t bytes);
int w34_free(void* ptr);
int w34_host_alloc_mapped(void** out_host_ptr, void** out_dev_ptr, size_t bytes);
int w34_host_free(void* host_ptr);
int w34_memcpy_h2d_async(void* dst, const void* src, size_t bytes, uintptr_t stream);
int w34_memcpy_d2h_async(void* dst, const void* src, size_t bytes, uintptr_t stream);
int w34_memcpy_d2d_async(void* dst, const void* src, size_t bytes, uintptr_t stream);
int w34_memset_async(void* dst, int value, size_t bytes, uintptr_t stream);

int w34_stream_create(uintptr_t* out_stream);
int w34_stream_destroy(uintptr_t stream);
int w34_stream_synchronize(uintptr_t stream);

int w34_event_create(uintptr_t* out_event, int enable_timing);
int w34_event_destroy(uintptr_t event);
int w34_event_record(uintptr_t event, uintptr_t stream);
int w34_stream_wait_event(uintptr_t stream, uintptr_t event);
int w34_event_synchronize(uintptr_t event);
int w34_event_elapsed_ms(float* out_ms, uintptr_t start_event, uintptr_t end_event);

int w34_graph_begin_capture(uintptr_t stream);
int w34_graph_end_capture(uintptr_t stream, uintptr_t* out_graph, uintptr_t* out_graph_exec);
int w34_graph_launch(uintptr_t graph_exec, uintptr_t stream);
int w34_graph_destroy(uintptr_t graph, uintptr_t graph_exec);

int w34_cublas_create(uintptr_t* out_handle, uintptr_t stream);
int w34_cublas_destroy(uintptr_t handle);
// Computes C [M, N] = A [M, K] @ B^T [N, K] in row-major (B is [N, K] row-major).
int w34_cublas_gemm_f16(uintptr_t handle, const void* a, const void* b, void* c,
                        int m, int n, int k);
int w34_cublas_gemm_bf16(uintptr_t handle, const void* a, const void* b, void* c,
                         int m, int n, int k);

// --- Norms, RoPE, Embedding, & Activations (op_norms_rope_act.cu) ---
int w34_embed_lookup_f16(const w34_exec* ex, const int32_t* token_ids,
                         const void* embed_weight, void* out_hidden,
                         const uint8_t* valid_token_mask,
                         int num_tokens, int hidden_size, int vocab_size);

int w34_copy_masked_rows_f16(const w34_exec* ex, const void* src, void* dst,
                             const uint8_t* valid_token_mask,
                             int num_tokens, int row_elems);

int w34_gemma_rms_norm_f16(const w34_exec* ex, const void* x, const void* weight,
                           void* out, const uint8_t* valid_token_mask,
                           int num_tokens, int hidden_size, float eps);

int w34_fused_add_gemma_rms_norm_f16(const w34_exec* ex, const void* x, void* residual,
                                     const void* weight, void* out,
                                     const uint8_t* valid_token_mask,
                                     int num_tokens, int hidden_size, float eps);

int w34_rms_norm_gated_silu_f16(const w34_exec* ex, const void* x, const void* z,
                                const void* weight, void* out,
                                const uint8_t* valid_token_mask,
                                int num_tokens, int num_v_heads, int head_dim, float eps);

int w34_fused_qk_rmsnorm_rope_gate_f16(
    const w34_exec* ex,
    const void* q_gate_in,   // [num_tokens, num_q_heads, 2 * head_dim]
    const void* k_in,        // [num_tokens, num_kv_heads, head_dim]
    const void* q_norm_w,    // [head_dim] (GemmaRMSNorm 1.0 + w)
    const void* k_norm_w,    // [head_dim] (GemmaRMSNorm 1.0 + w)
    const int32_t* positions,// [3, pos_plane_stride]
    const uint8_t* valid_token_mask,
    void* q_out,             // [num_tokens, num_q_heads, head_dim]
    void* k_out,             // [num_tokens, num_kv_heads, head_dim]
    void* gate_out,          // [num_tokens, num_q_heads, head_dim]
    int num_tokens,
    int pos_plane_stride,
    int num_q_heads,
    int num_kv_heads,
    int head_dim,
    int rotary_dim,
    int mrope_s0,
    int mrope_s1,
    int mrope_s2,
    float rope_theta,
    float rms_eps);

int w34_silu_and_mul_f16(const w34_exec* ex, const void* gate_up, void* out,
                         const uint8_t* valid_token_mask,
                         int num_tokens, int intermediate_size);

int w34_silu_and_mul_split_f16(const w34_exec* ex, const void* gate, const void* up,
                               void* out, const uint8_t* valid_token_mask,
                               int num_tokens, int intermediate_size);

int w34_apply_attn_output_gate_f16(const w34_exec* ex, const void* attn_out,
                                   const void* gate, void* out,
                                   const uint8_t* valid_token_mask,
                                   int num_tokens, int total_dim);

int w34_split_cols_2_f16(const w34_exec* ex, const void* src, void* dst0, void* dst1,
                         int num_tokens, int dim0, int dim1);
int w34_split_cols_3_f16(const w34_exec* ex, const void* src, void* dst0, void* dst1, void* dst2,
                         int num_tokens, int dim0, int dim1, int dim2);

// --- Marlin Repack & GEMMs (op_marlin.cu) ---
int w34_zero_i32(const w34_exec* ex, void* ptr, int64_t count);
int w34_marlin_repack_int4(const w34_exec* ex, const uint32_t* src_n_major,
                           uint32_t* dst_marlin, int64_t k, int64_t n);
int w34_marlin_gemm_f16u4(const w34_exec* ex, const void* a, const void* b,
                          const void* scales, void* c, int32_t* workspace,
                          int64_t m, int64_t n, int64_t k,
                          int64_t workspace_ints, int group_blocks);
int w34_marlin_gemm_f16u8(const w34_exec* ex, const void* a, const void* b,
                          const void* scales, void* c, int32_t* workspace,
                          int64_t m, int64_t n, int64_t k,
                          int64_t workspace_ints, int group_blocks);

// --- Descriptor-Driven Paged Attention (op_paged_attn.cu) ---
int w34_kv_cache_write(const w34_exec* ex, const w34_kv_layer_layout* layout,
                       const w34_attention_batch* batch,
                       const void* k_in, const void* v_in);

int w34_paged_attention_f16(const w34_exec* ex, const w34_kv_layer_layout* layout,
                            const w34_attention_batch* batch,
                            const void* q_in, void* attn_out,
                            int num_q_heads);

// --- Gated DeltaNet (op_gdn.cu) ---
int w34_gdn_causal_conv1d_f16(const w34_exec* ex, const w34_gdn_batch* batch,
                              const void* x_in, const void* conv_weight,
                              const void* conv_bias, void* conv_state_pool,
                              void* x_out, int conv_dim, int kernel_size,
                              int total_state_slots, int silu_activation);

int w34_gdn_recurrent_f16(const w34_exec* ex, const w34_gdn_batch* batch,
                          const void* qkv_conv_out, // [total_tokens, 2*k_dim + v_dim] or split q,k,v
                          const void* q_in,         // [total_tokens, num_k_heads, head_k_dim] (if split)
                          const void* k_in,         // [total_tokens, num_k_heads, head_k_dim] (if split)
                          const void* v_in,         // [total_tokens, num_v_heads, head_v_dim] (if split)
                          const void* b_in,         // [total_tokens, num_v_heads] fp16
                          const void* a_in,         // [total_tokens, num_v_heads] fp16
                          const float* a_log,       // [num_v_heads] fp32
                          const float* dt_bias,     // [num_v_heads] fp32
                          float* ssm_state_pool,    // [total_state_slots, num_v_heads, head_v_dim, head_k_dim] fp32
                          void* out_f16,            // [total_tokens, num_v_heads, head_v_dim] fp16
                          int num_k_heads, int head_k_dim,
                          int num_v_heads, int head_v_dim,
                          int total_state_slots);

// --- Pinned-Sysmem TP=2 Collectives (op_collectives.cu) ---
size_t w34_tp2_signal_bytes(void);
int w34_tp2_all_reduce_f16(const w34_exec* ex, const void* local_in, void* out,
                           void* stage0, void* stage1,
                           void* sig0, void* sig1, int rank, int num_elems);
int w34_tp2_all_reduce_bf16(const w34_exec* ex, const void* local_in, void* out,
                            void* stage0, void* stage1,
                            void* sig0, void* sig1, int rank, int num_elems);
int w34_tp2_all_reduce_f32(const w34_exec* ex, const float* local_in, float* out,
                           float* stage0, float* stage1,
                           void* sig0, void* sig1, int rank, int num_elems);
int w34_tp2_broadcast_i32(const w34_exec* ex, const int32_t* src_rank0, int32_t* out,
                          int32_t* stage0, int32_t* stage1,
                          void* sig0, void* sig1, int rank, int num_elems);
int w34_tp2_all_gather_top1_f32_i32(const w34_exec* ex,
                                    const float* local_max_val,
                                    const int32_t* local_max_id,
                                    float* global_max_val,
                                    int32_t* global_max_id,
                                    void* stage0, void* stage1,
                                    void* sig0, void* sig1, int rank, int num_rows);

// --- GPU Sampler (op_sampler.cu) ---
int w34_sampler_greedy_top5_f16(const w34_exec* ex,
                                const void* logits_f16,     // [num_rows, local_vocab_size]
                                int32_t* out_greedy_ids,    // [num_rows]
                                int32_t* out_top5_ids,      // [num_rows, 5] (nullable)
                                float* out_top5_logprobs,   // [num_rows, 5] (nullable)
                                int num_rows,
                                int local_vocab_size,
                                int vocab_offset);

#ifdef __cplusplus
}
#endif
