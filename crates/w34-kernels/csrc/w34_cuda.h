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
    // Legacy name for non-causal query-relative left-window attention.
    W34_ATTN_MASK_SYMMETRIC_WINDOW = 1,
    W34_ATTN_MASK_NONCAUSAL_WINDOW = 1,
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

typedef struct {
    int32_t device_id;
    int32_t compute_major;
    int32_t compute_minor;
    int32_t multi_processor_count;
    int32_t max_shared_mem_per_block_optin;
    int32_t max_shared_mem_per_multiprocessor;
    int32_t l2_cache_bytes;
    int32_t persisting_l2_max_bytes;
    char    name[64];
} w34_device_props;

static_assert(sizeof(w34_device_props) == 96, "w34_device_props ABI size mismatch");

// --- Runtime, Stream, Event, Graph, & cuBLAS primitives (op_init.cu) ---
const char* w34_cuda_error_string(int code);
int w34_cublas_runtime_version(int* out_version);
int w34_get_device_count(int* out_count);
int w34_get_device_props(int device_id, w34_device_props* out_props);
int w34_get_current_device_props(w34_device_props* out_props);
int w34_get_tp2_min_sm_count(int* out_min_sms);
int w34_get_tp_min_sm_count(int tp_size, int* out_min_sms);
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
int w34_stream_set_persisting_l2_window(uintptr_t stream, const void* base_ptr, size_t num_bytes);

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
// C[M,N] = half(A[M,K] @ B[N,K]^T + bias[N]); positive dimensions required.
int w34_cublas_gemm_f16_bias(uintptr_t handle, const void* a, const void* b,
                            const void* bias, void* c, int m, int n, int k);
int w34_cublas_gemm_f16(uintptr_t handle, const void* a, const void* b, void* c,
                        int m, int n, int k);
int w34_cublas_lora_shrink_expand_f16(
    uintptr_t handle, const void* x, void* y_inout, const void* lora_a, const void* lora_b,
    void* lowrank_ws, int adapter_id, int m, int size_n, int size_k, int rank_pad,
    int num_b_slices, const int32_t* b_slices4);
int w34_cublas_gemm_bf16(uintptr_t handle, const void* a, const void* b, void* c,
                         int m, int n, int k);

// --- Norms, RoPE, Embedding, & Activations (op_norms_rope_act.cu) ---
int w34_embed_lookup_f16(const w34_exec* ex, const int32_t* token_ids,
                         const void* embed_weight, void* out_hidden,
                         const uint8_t* valid_token_mask,
                         int num_tokens, int hidden_size, int vocab_size);

// Out-of-place normalized Sylvester H16/H64 on contiguous FP16 [rows,k].
// Writes contiguous [rows,padded_k], zero-filling the tail after transforming
// logical blocks. Both widths must be multiples of block_size. No aliasing.
// FP16 crop with optional FP32 bias add. In-place only for equal widths.
int w34_linear_epilogue_f16(const w34_exec* ex, const void* src,
                            const void* bias, void* dst, int rows,
                            int logical_n, int physical_n);

int w34_block_hadamard_f16(const w34_exec* ex, const void* src, void* dst,
                           int rows, int k, int padded_k, int block_size);
int w34_block_hadamard_inplace_f16(const w34_exec* ex, void* buf,
                                   int rows, int k, int block_size);
int w34_block_hadamard_bf16(const w34_exec* ex, const void* src, void* dst,
                            int rows, int k, int padded_k, int block_size);

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

int w34_fused_add_gemma_rms_norm_tap_f16(const w34_exec* ex, const void* x, void* residual,
                                         const void* weight, void* out, void* tap_dst_bf16,
                                         const uint8_t* valid_token_mask,
                                         int num_tokens, int hidden_size, float eps,
                                         int tap_row_stride, int tap_col_offset);

int w34_rms_norm_gated_silu_f16(const w34_exec* ex, const void* x, const void* z,
                                const void* weight, void* out,
                                const uint8_t* valid_token_mask,
                                int num_tokens, int num_v_heads, int head_dim, float eps);
int w34_rms_norm_gated_silu_strided_f16(const w34_exec* ex, const void* x, const void* z,
                                        const void* weight, void* out,
                                        const uint8_t* valid_token_mask,
                                        int num_tokens, int num_v_heads, int head_dim,
                                        int z_token_stride, float eps);
int w34_rms_norm_gated_sigmoid_strided_f16(const w34_exec* ex, const void* x, const void* z,
                                           const void* weight, void* out,
                                           const uint8_t* valid_token_mask,
                                           int num_tokens, int num_v_heads, int head_dim,
                                           int z_token_stride, float eps);

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

int w34_fused_qk_rmsnorm_rope_gate_strided_f16(
    const w34_exec* ex,
    const void* q_gate_in,
    const void* k_in,
    const void* q_norm_w,
    const void* k_norm_w,
    const int32_t* positions,
    const uint8_t* valid_token_mask,
    void* q_out,
    void* k_out,
    void* gate_out,
    int num_tokens,
    int pos_plane_stride,
    int num_q_heads,
    int num_kv_heads,
    int head_dim,
    int rotary_dim,
    int mrope_s0,
    int mrope_s1,
    int mrope_s2,
    int q_gate_row_stride,
    int k_row_stride,
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
int w34_small_n_gemm_f16(const w34_exec* ex, const void* x, const void* w, void* out,
                         int m, int size_n, int size_k);
int w34_small_n_gemm_dual_f16(const w34_exec* ex, const void* x,
                              const void* w0, void* out0, int size_n0,
                              const void* w1, void* out1, int size_n1,
                              int m, int size_k);

// --- Marlin Repack & GEMMs (op_marlin.cu) ---
int w34_zero_i32(const w34_exec* ex, void* ptr, int64_t count);
int w34_marlin_warmup_specializations(void);
int w34_marlin_repack_int2(const w34_exec* ex, const uint32_t* src_n_major,
                           uint32_t* dst_marlin, int64_t k, int64_t n);
int w34_marlin_repack_int4(const w34_exec* ex, const uint32_t* src_n_major,
                           uint32_t* dst_marlin, int64_t k, int64_t n);
int w34_marlin_repack_int8(const w34_exec* ex, const uint32_t* src_n_major,
                           uint32_t* dst_marlin, int64_t k, int64_t n);
int w34_marlin_quantize_f16_u4b8(const w34_exec* ex, const void* src_f16,
                                 uint32_t* dst_marlin, void* dst_scales_perm,
                                 int64_t k, int64_t n, int group_size);
int w34_marlin_gemm_f16u2(const w34_exec* ex, const void* a, const void* b,
                          const void* scales, void* c, int32_t* workspace,
                          int64_t m, int64_t n, int64_t k,
                          int64_t workspace_ints, int group_blocks);
int w34_marlin_gemm_f16u4(const w34_exec* ex, const void* a, const void* b,
                          const void* scales, void* c, int32_t* workspace,
                          int64_t m, int64_t n, int64_t k,
                          int64_t workspace_ints, int group_blocks);
int w34_marlin_gemm_bf16u4(const w34_exec* ex, const void* a, const void* b,
                           const void* scales, void* c, int32_t* workspace,
                           int64_t m, int64_t n, int64_t k,
                           int64_t workspace_ints, int group_blocks);
int w34_marlin_gemm_f16u4_zp(const w34_exec* ex, const void* a, const void* b,
                             const void* scales, const void* zeros, void* c,
                             int32_t* workspace, int64_t m, int64_t n, int64_t k,
                             int64_t workspace_ints, int group_blocks);
int w34_marlin_gemm_f16u8(const w34_exec* ex, const void* a, const void* b,
                          const void* scales, void* c, int32_t* workspace,
                          int64_t m, int64_t n, int64_t k,
                          int64_t workspace_ints, int group_blocks);

// --- Descriptor-Driven KV Cache Write & Paged Attention (op_kv_cache.cu, op_paged_attn.cu) ---
int w34_kv_cache_write(const w34_exec* ex, const w34_kv_layer_layout* layout,
                       const w34_attention_batch* batch,
                       const void* k_in, const void* v_in);
int w34_kv_cache_write_strided(const w34_exec* ex, const w34_kv_layer_layout* layout,
                               const w34_attention_batch* batch,
                               const void* k_in, const void* v_in,
                               int k_row_stride, int v_row_stride);

// Native BF16 I/O and BF16 descriptor storage; identity transforms required.
int w34_kv_cache_write_bf16(const w34_exec* ex, const w34_kv_layer_layout* layout,
    const w34_attention_batch* batch, const void* k_in, const void* v_in);
int w34_kv_cache_write_strided_bf16(const w34_exec* ex, const w34_kv_layer_layout* layout,
    const w34_attention_batch* batch, const void* k_in, const void* v_in, int row_stride_elems);
int w34_paged_attention_bf16(const w34_exec* ex, const w34_kv_layer_layout* layout,
    const w34_attention_batch* batch, const void* q_in, void* attn_out, int num_q_heads);

int w34_paged_attention_f16(const w34_exec* ex, const w34_kv_layer_layout* layout,
                            const w34_attention_batch* batch,
                            const void* q_in, void* attn_out,
                            int num_q_heads);

// --- Gated DeltaNet (op_gdn.cu, op_gdn_wy.cu) ---
int w34_gdn_causal_conv1d_f16(const w34_exec* ex, const w34_gdn_batch* batch,
                              const void* x_in, const void* conv_weight,
                              const void* conv_bias, void* conv_state_pool,
                              void* x_out, int conv_dim, int kernel_size,
                              int total_state_slots, int silu_activation);
int w34_gdn_causal_conv1d_strided_f16(const w34_exec* ex, const w34_gdn_batch* batch,
                                      const void* x_in, const void* conv_weight,
                                      const void* conv_bias, void* conv_state_pool,
                                      void* x_out, int conv_dim, int x_in_row_stride,
                                      int kernel_size, int total_state_slots,
                                      int silu_activation);

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
int w34_gdn_recurrent_strided_f16(const w34_exec* ex, const w34_gdn_batch* batch,
                                  const void* qkv_conv_out,
                                  const void* q_in,
                                  const void* k_in,
                                  const void* v_in,
                                  const void* b_in,
                                  const void* a_in,
                                  const float* a_log,
                                  const float* dt_bias,
                                  float* ssm_state_pool,
                                  void* out_f16,
                                  int num_k_heads, int head_k_dim,
                                  int num_v_heads, int head_v_dim,
                                  int ba_row_stride, int total_state_slots);

// --- Pinned-Sysmem TP=2 Collectives (op_collectives.cu) ---
size_t w34_tp2_signal_bytes(void);
int w34_tp2_all_reduce_f16(const w34_exec* ex, const void* local_in, void* out,
                           void* stage0, void* stage1,
                           void* sig0, void* sig1, int rank, int num_elems);
int w34_tp2_all_reduce_residual_gemma_rms_norm_f16(
    const w34_exec* ex, const void* local_in, void* ar_out, void* residual,
    const void* weight, void* out, void* tap_dst_bf16,
    void* stage0, void* stage1,
    void* sig0, void* sig1, int rank,
    int num_tokens, int hidden_size, float eps,
    int tap_row_stride, int tap_col_offset);
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
int w34_tp2_gather_vocab_f16(const w34_exec* ex,
                             const void* local_logits_f16,
                             void* rank0_full_logits_f16,
                             void* stage0, void* stage1,
                             void* sig0, void* sig1,
                             int rank, int num_rows, int local_vocab_size);
int w34_tp2_warmup(void);

// --- GPU Sampler (op_sampler.cu) ---
#define W34_MAX_TOP_LOGPROBS 20

typedef struct w34_sampler_row_cfg {
    float    temperature;
    int32_t  top_k;
    float    top_p;
    float    min_p;
    float    repetition_penalty;
    float    frequency_penalty;
    float    presence_penalty;
    float    uniform_sample;
    float    recovery_uniform;
    int32_t  draft_token_id;
    int32_t  has_bitmask;
    int32_t  row_bitmask_word_offset;
    int32_t  sparse_offset;
    int32_t  sparse_count;
    int32_t  top_logprobs_k;
    int32_t  use_token_id_cdf;
} w34_sampler_row_cfg;

typedef struct w34_sampler_sparse_entry {
    int32_t  token_id;
    uint32_t output_count;
    float    logit_bias;
    uint16_t in_prompt;
    uint16_t banned;
} w34_sampler_sparse_entry;

typedef struct w34_sampler_row_out {
    int32_t  sampled_id;
    int32_t  draft_accepted;
    int32_t  status;
    float    sampled_logprob;
    float    draft_prob;
    int32_t  reserved;
    int32_t  top_ids[W34_MAX_TOP_LOGPROBS];
    float    top_logprobs[W34_MAX_TOP_LOGPROBS];
} w34_sampler_row_out;

static_assert(sizeof(w34_sampler_row_cfg) == 64, "w34_sampler_row_cfg ABI size mismatch");
static_assert(sizeof(w34_sampler_sparse_entry) == 16, "w34_sampler_sparse_entry ABI size mismatch");
static_assert(sizeof(w34_sampler_row_out) == 184, "w34_sampler_row_out ABI size mismatch");

int w34_sampler_greedy_top5_f16(const w34_exec* ex,
                                const void* logits_f16,     // [num_rows, local_vocab_size]
                                int32_t* out_greedy_ids,    // [num_rows]
                                int32_t* out_top5_ids,      // [num_rows, 5] (nullable)
                                float* out_top5_logprobs,   // [num_rows, 5] (nullable)
                                int num_rows,
                                int local_vocab_size,
                                int vocab_offset);

int w34_sampler_step_f16(const w34_exec* ex,
                         const void* logits_f16,
                         const w34_sampler_row_cfg* row_cfgs,
                         const uint32_t* bitmask_u32,
                         const w34_sampler_sparse_entry* sparse_entries,
                         w34_sampler_row_out* row_outs,
                         int num_rows,
                         int vocab_size,
                         int total_sparse_entries);

int w34_sampler_step_fast_greedy_f16(const w34_exec* ex,
                                     const void* logits_f16,
                                     const w34_sampler_row_cfg* row_cfgs,
                                     const w34_sampler_sparse_entry* sparse_entries,
                                     w34_sampler_row_out* row_outs,
                                     int num_rows,
                                     int vocab_size);

int w34_sampler_step_fast_greedy_masked_f16(const w34_exec* ex,
                                            const void* logits_f16,
                                            const w34_sampler_row_cfg* row_cfgs,
                                            const uint32_t* bitmask_u32,
                                            const w34_sampler_sparse_entry* sparse_entries,
                                            w34_sampler_row_out* row_outs,
                                            int num_rows,
                                            int vocab_size);

int w34_sampler_scatter_ids_to_tokens(const w34_exec* ex,
                                      const w34_sampler_row_out* row_outs,
                                      const int32_t* greedy_ids,
                                      int32_t* dst_token_ids,
                                      int num_rows);

// Rejection outputs: emitted [R,k+1], accepted [R], status [R].
// Only emitted[0..accepted+1] is live; suffix is -1. Masked requests return
// accepted=-1,status=0; invalid request data returns accepted=-1,status=1.
// k is in [0,16]. Inputs are contiguous; valid [R] is optional.
int w34_rejection_greedy(const w34_exec* ex, const int32_t* drafts,
    const int32_t* targets, const uint8_t* valid, int32_t* emitted,
    int32_t* accepted, int32_t* status, int num_reqs, int k);
// target [R,k+1,V], proposal [R,k,V], drafts/uniforms [R,k], recovery [R].
// Probability rows must be finite, nonnegative, and sum to 1 within 1e-5.
// Uniforms are independent draws in [0,1). No sampler/RNG state is mutated.
int w34_rejection_stochastic(const w34_exec* ex, const int32_t* drafts,
    const float* target, const float* proposal, const double* uniforms,
    const double* recovery, const uint8_t* valid, int32_t* emitted,
    int32_t* accepted, int32_t* status, int num_reqs, int k, int vocab);

// --- MTP & DFlash2 Speculative Decoding Kernels (op_dflash2.cu) ---
// BF16 activation path. RMSNorm may update a distinct residual in place;
// x==y is supported, but residual must not alias either. Other outputs must not
// overlap inputs. RoPE table [table_rows,dim] contains [cos || sin].
int w34_concat_cols_2_f16(const w34_exec* ex, const void* src0, const void* src1,
                          void* dst, int num_tokens, int dim0, int dim1);
int w34_tap_add_f16_to_bf16(const w34_exec* ex, const void* res_f16, const void* mlp_f16,
                            void* dst_bf16, int num_tokens, int hidden_size,
                            int dst_row_stride, int dst_col_offset);
int w34_embed_lookup_f16_to_bf16(const w34_exec*, const int32_t* ids,
    const void* weight, void* out, int rows, int hidden, int vocab);
int w34_cast_bf16_to_f16(const w34_exec*, const void* x, void* y, int64_t n);
int w34_dflash_edges_bf16(const w34_exec*, const int32_t* ids, const float* unary,
    const void* hidden, const void* pred, const void* succ, float* out,
    int anchor, int k, int topk, int rank, int vocab);
int w34_rms_norm_bf16(const w34_exec*, const void* x, void* residual,
    const void* weight, void* y, int rows, int hidden, float eps);
int w34_silu_and_mul_bf16(const w34_exec*, const void* gate_up, void* y, int rows, int hidden);
int w34_dflash_conv_bf16(const w34_exec*, const void* x, const void* delta,
    const void* base, void* y, int rows, int hidden, int block, int group, int taps, int side);
// q_heads==0 selects K-only norm/RoPE over packed [K,V] context projection.
// *_f32_table reads FP32 [table_rows, dim] cos/sin tables (production DFlash2Rank);
// w34_dflash_qk_norm_rope_bf16 reads BF16 [table_rows, dim] tables.
// *_qout writes rotated Q heads directly to contiguous q_out [rows, q_heads * dim]
// when q_out != nullptr (or in-place into qkv when q_out == nullptr).
int w34_dflash_qk_norm_rope_bf16_f32_table(const w34_exec*, void* qkv, const int32_t* positions,
    const void* qw, const void* kw, const void* table, int rows, int q_heads, int kv_heads,
    int dim, int table_rows, float eps);
int w34_dflash_qk_norm_rope_bf16_f32_table_qout(const w34_exec*, void* qkv, void* q_out,
    const int32_t* positions, const void* qw, const void* kw, const void* table,
    int rows, int q_heads, int kv_heads, int dim, int table_rows, float eps);
int w34_dflash_qk_norm_rope_kv_write_bf16_f32_table(const w34_exec* ex, void* qkv, void* q_out,
    const int32_t* positions, const void* qw, const void* kw, const void* table,
    const w34_kv_layer_layout* kv_layout, const int32_t* write_slot_mapping,
    int rows, int q_heads, int kv_heads, int dim, int table_rows, float eps);
int w34_dflash_select_greedy_bf16(const w34_exec* ex, const int32_t* ids, const float* unary,
    const void* hidden, const void* pred, const void* succ, int32_t* out_path,
    int anchor, int k, int topk, int rank, int vocab);
int w34_dflash_select_greedy_dev_anchor_bf16(const w34_exec* ex, const int32_t* ids,
    const float* unary, const void* hidden, const void* pred, const void* succ,
    int32_t* out_path, const int32_t* anchor_dev, int k, int topk, int rank, int vocab);
int w34_dflash_qk_norm_rope_bf16(const w34_exec*, void* qkv, const int32_t* positions,
    const void* qw, const void* kw, const void* table, int rows, int q_heads, int kv_heads,
    int dim, int table_rows, float eps);
int w34_copy_cols_bf16(const w34_exec*, const void* src, void* dst,
    int rows, int src_stride, int dst_stride, int src_col, int dst_col, int cols);
int w34_topk_rows_f16(const w34_exec* ex, const void* logits_f16,
                      int32_t* out_ids, float* out_vals,
                      int num_rows, int local_vocab_size, int vocab_offset, int top_k);
int w34_topk_rows_split_f16(const w34_exec* ex, const void* logits_f16,
                            int32_t* split_ids_ws, float* split_vals_ws,
                            int32_t* out_ids, float* out_vals,
                            int num_rows, int local_vocab_size, int vocab_offset,
                            int top_k, int num_splits);

// --- Qwen3_VisionTransformer & Multimodal Kernels (op_vision.cu) ---
int w34_vision_bias_add_f16(const w34_exec* ex, const void* x, const void* bias,
                            const void* residual, void* out, int rows, int cols);
int w34_vision_layernorm_f16(const w34_exec* ex, const void* x, const void* weight,
                             const void* bias, void* out, int rows, int cols, float eps);
int w34_vision_bias_gelu_f16(const w34_exec* ex, const void* x, const void* bias,
                             void* out, int rows, int cols, int use_tanh);
int w34_vision_qkv_rope_attn_f16(const w34_exec* ex, void* qkv, const void* qkv_bias,
                                 const int32_t* pos_hw, const int32_t* cu_seqlens,
                                 void* out, int total_patches, int num_images,
                                 int num_heads, int head_dim, float rope_theta);
int w34_embedding_splice_f16(const w34_exec* ex, void* dst_embeds,
                             const void* src_visual_embeds, const int32_t* token_indices,
                             int num_visual_tokens, int hidden_size);

// --- Qwen3.5 / 3.6 / 3.8 Hybrid MoE Kernels (op_moe.cu, op_moe_router.cu) ---
int w34_moe_topk_router_f16(const w34_exec* ex, const void* router_logits,
                            float* topk_weights, int32_t* topk_ids,
                            int num_tokens, int num_experts, int top_k,
                            int norm_topk_prob, int scoring_func);
int w34_moe_w4a16_gate_up_silu_f16(const w34_exec* ex, const void* x,
                                   const uint32_t* gate_up_packed, const void* gate_up_scales,
                                   const int32_t* topk_ids, void* interm_out,
                                   int num_tokens, int top_k, int num_experts,
                                   int n_interm, int k_hidden, int group_size);
int w34_moe_w4a16_down_accum_f16(const w34_exec* ex, const void* interm_in,
                                 const uint32_t* down_packed, const void* down_scales,
                                 const int32_t* topk_ids, const float* topk_weights,
                                 const void* shared_expert_out,
                                 const void* shared_expert_gate_logits,
                                 void* out, int num_tokens, int top_k, int num_experts,
                                 int k_hidden, int n_interm, int group_size);
int w34_moe_w4a16_mlp_f16(const w34_exec* ex, const void* x,
                          const uint32_t* gate_up_packed, const void* gate_up_scales,
                          const uint32_t* down_packed, const void* down_scales,
                          const int32_t* topk_ids, const float* topk_weights,
                          const void* shared_expert_out,
                          const void* shared_expert_gate_logits,
                          void* interm_buf, void* out,
                          int num_tokens, int top_k, int num_experts,
                          int n_interm, int k_hidden, int group_size);
int w34_moe_w4a16_mlp_ex_f16(const w34_exec* ex, const void* x,
                             const uint32_t* gate_up_packed, const void* gate_up_scales,
                             const uint32_t* down_packed, const void* down_scales,
                             const int32_t* topk_ids, const float* topk_weights,
                             const void* shared_expert_out,
                             const void* shared_expert_gate_logits,
                             void* interm_buf, void* out,
                             int num_tokens, int top_k, int num_experts,
                             int n_interm, int k_hidden, int group_size,
                             int down_hadamard_block_size);
int w34_moe_w2a16_gate_up_silu_f16(const w34_exec* ex, const void* x,
                                   const uint32_t* gate_up_packed, const void* gate_up_scales,
                                   const int32_t* topk_ids, void* interm_out,
                                   int num_tokens, int top_k, int num_experts,
                                   int n_interm, int k_hidden, int group_size);
int w34_moe_w2a16_down_accum_f16(const w34_exec* ex, const void* interm_in,
                                 const uint32_t* down_packed, const void* down_scales,
                                 const int32_t* topk_ids, const float* topk_weights,
                                 const void* shared_expert_out,
                                 const void* shared_expert_gate_logits,
                                 void* out, int num_tokens, int top_k, int num_experts,
                                 int k_hidden, int n_interm, int group_size);
int w34_moe_w2a16_mlp_f16(const w34_exec* ex, const void* x,
                          const uint32_t* gate_up_packed, const void* gate_up_scales,
                          const uint32_t* down_packed, const void* down_scales,
                          const int32_t* topk_ids, const float* topk_weights,
                          const void* shared_expert_out,
                          const void* shared_expert_gate_logits,
                          void* interm_buf, void* out,
                          int num_tokens, int top_k, int num_experts,
                          int n_interm, int k_hidden, int group_size);
int w34_moe_w2a16_mlp_ex_f16(const w34_exec* ex, const void* x,
                             const uint32_t* gate_up_packed, const void* gate_up_scales,
                             const uint32_t* down_packed, const void* down_scales,
                             const int32_t* topk_ids, const float* topk_weights,
                             const void* shared_expert_out,
                             const void* shared_expert_gate_logits,
                             void* interm_buf, void* out,
                             int num_tokens, int top_k, int num_experts,
                             int n_interm, int k_hidden, int group_size,
                             int down_hadamard_block_size);
int w34_moe_w4a16_mlp_zp_f16(const w34_exec* ex, const void* x,
                             const uint32_t* gate_up_packed, const void* gate_up_scales,
                             const uint32_t* gate_up_qzeros,
                             const uint32_t* down_packed, const void* down_scales,
                             const uint32_t* down_qzeros,
                             const int32_t* topk_ids, const float* topk_weights,
                             const void* shared_expert_out,
                             const void* shared_expert_gate_logits,
                             void* interm_buf, void* out,
                             int num_tokens, int top_k, int num_experts,
                             int n_interm, int k_hidden, int group_size);
int w34_moe_w4a16_mlp_zp_ex_f16(const w34_exec* ex, const void* x,
                                const uint32_t* gate_up_packed, const void* gate_up_scales,
                                const uint32_t* gate_up_qzeros,
                                const uint32_t* down_packed, const void* down_scales,
                                const uint32_t* down_qzeros,
                                const int32_t* topk_ids, const float* topk_weights,
                                const void* shared_expert_out,
                                const void* shared_expert_gate_logits,
                                void* interm_buf, void* out,
                                int num_tokens, int top_k, int num_experts,
                                int n_interm, int k_hidden, int group_size,
                                int down_hadamard_block_size);
int w34_moe_f16_gate_up_silu(const w34_exec* ex, const void* x, const void* gate_up_w,
                             const int32_t* topk_ids, void* interm_out,
                             int num_tokens, int top_k, int num_experts,
                             int n_interm, int k_hidden);
int w34_moe_f16_down_accum(const w34_exec* ex, const void* interm_in, const void* down_w,
                           const int32_t* topk_ids, const float* topk_weights,
                           const void* shared_expert_out,
                           const void* shared_expert_gate_logits,
                           void* out, int num_tokens, int top_k, int num_experts,
                           int k_hidden, int n_interm);

// --- Qwen3.8-Flash-Next (qwen4_exp): HyperConnection, PLE, and QSA Kernels ---
int w34_hc_init_streams_f16(const w34_exec* ex, const void* embed_in, void* residual_out,
                            const uint8_t* valid_token_mask, int num_tokens,
                            int hc_count, int hidden_size);
int w34_hc_rmsnorm_f16(const w34_exec* ex, const void* residual_in, const void* norm_weight,
                       void* normed_out, const uint8_t* valid_token_mask,
                       int num_tokens, int hc_count, int hidden_size, float eps);
int w34_hc_down_silu_inject_f16(const w34_exec* ex, const void* down_in, void* down_out,
                                const void* inj_in, float* inject_weights_out,
                                const uint8_t* valid_token_mask, int num_tokens,
                                int hc_rank_dim, int hc_count);
int w34_hc_mix_streams_f16(const w34_exec* ex, const void* residual_in, const void* up_raw_in,
                           void* mixed_out, const uint8_t* valid_token_mask,
                           int num_tokens, int hc_count, int hidden_size);
int w34_hc_mix_streams_hadamard_f16(const w34_exec* ex, const void* residual_in,
                                    const void* up_raw_in, void* mixed_out, void* had64_out,
                                    const uint8_t* valid_token_mask, int num_tokens,
                                    int hc_count, int hidden_size);
int w34_hc_combine_f16(const w34_exec* ex, const void* residual_in, const void* block_out,
                       const float* inject_weights, const void* ple_delta,
                       void* residual_out, const uint8_t* valid_token_mask,
                       int num_tokens, int hc_count, int hidden_size);
int w34_hc_combine_tap_f16(const w34_exec* ex, const void* residual_in, const void* block_out,
                           const float* inject_weights, const void* ple_delta,
                           void* residual_out, void* tap_bf16_out,
                           const uint8_t* valid_token_mask, int num_tokens,
                           int hc_count, int hidden_size, int tap_stride_elems,
                           int tap_offset_elems);
int w34_qwen4_mtp_broadcast_add_f16(const w34_exec* ex, const void* fc_embed,
                                    const void* fc_hidden, void* out_multi_hidden,
                                    int num_tokens, int hc_count, int hidden_size);

int w34_ple_dequant_f8_to_f16(const w34_exec* ex, const uint8_t* f8_in, void* f16_out,
                              const uint8_t* valid_token_mask, int num_tokens,
                              int embed_dim, float weight_scale);
int w34_ple_dequant_f8_hadamard_to_f16(const w34_exec* ex, const uint8_t* f8_in, void* f16_out,
                                       const uint8_t* valid_token_mask, int num_tokens,
                                       int embed_dim, float weight_scale);
int w34_ple_dequant_lut4_to_f16(const w34_exec* ex, const uint8_t* raw_in, const float* lut16,
                                void* f16_out, const uint8_t* valid_token_mask, int num_tokens,
                                int num_heads, int head_dim, int group_size);
int w34_ple_dequant_lut4_hadamard_to_f16(const w34_exec* ex, const uint8_t* raw_in,
                                         const float* lut16, void* f16_out,
                                         const uint8_t* valid_token_mask, int num_tokens,
                                         int num_heads, int head_dim, int group_size);
int w34_ple_dilated_conv1d_f16(const w34_exec* ex, const void* key_in, const void* conv_weight,
                               void* conv_state_pool, void* key_out,
                               const w34_gdn_batch* batch, int conv_dim);
int w34_ple_dilated_conv1d_add_f16(const w34_exec* ex, const void* key_in,
                                   const void* conv_weight, void* conv_state_pool,
                                   void* key_out, void* residual_inout,
                                   const w34_gdn_batch* batch, int conv_dim);
int w34_ple_signed_sqrt_gate_f16(const w34_exec* ex, const void* norm_query,
                                 const void* norm_key, const void* value_in,
                                 void* ple_delta_out, const uint8_t* valid_token_mask,
                                 int num_tokens, int hc_count, int embed_dim, float eps);
int w34_ple_signed_sqrt_gate_norm_add_f16(
    const w34_exec* ex, const void* norm_query, const void* norm_key, const void* value_in,
    void* ple_delta_out, void* residual_inout, const void* norm_conv_weight,
    void* norm_conv_out, const uint8_t* valid_token_mask, int num_tokens, int hc_count,
    int embed_dim, float eps, float norm_eps);

int w34_qsa_indexer_norm_rope_f16(const w34_exec* ex, const void* qk_in,
                                  const void* q_norm_weight, const void* k_norm_weight,
                                  const int32_t* positions, void* q_out, void* k_out,
                                  const uint8_t* valid_token_mask, int num_tokens,
                                  int q_heads, int kv_heads, int head_dim, int rotary_dim,
                                  float rope_theta, float eps, int mrope_s1, int mrope_s2,
                                  int pos_plane_stride);
int w34_qsa_indexer_cache_update_f16(const w34_exec* ex, const void* k_in,
                                     void* raw_k_cache, void* comp_k_cache,
                                     const void* k_norm_weight,
                                     const int32_t* positions,
                                     const int32_t* write_slot_mapping,
                                     const uint8_t* valid_token_mask,
                                     int num_tokens, int head_dim, int rotary_dim,
                                     float rope_theta, float eps, int mrope_s1,
                                     int mrope_s2, int pos_plane_stride);
int w34_qsa_select_blocks_f16(const w34_exec* ex, const void* q_indexer,
                              const void* comp_k_cache, const w34_attention_batch* batch,
                              int32_t* out_selected_phys_blocks,
                              int32_t* out_selected_logical_blocks,
                              int32_t* out_num_selected_blocks,
                              int32_t* out_token_seq_pos,
                              int q_heads, int head_dim, int block_size,
                              int select_k_blocks, int sink_blocks, int recent_blocks);
int w34_qsa_sparse_paged_attn_f16(const w34_exec* ex, const void* q, const void* gate,
                                  const w34_kv_layer_layout* kv_layout,
                                  const int32_t* selected_phys_blocks,
                                  const int32_t* selected_logical_blocks,
                                  const int32_t* num_selected_blocks,
                                  const int32_t* token_seq_pos,
                                  const uint8_t* valid_token_mask,
                                  void* out, int num_tokens, int local_q_heads,
                                  int select_k_blocks, float sm_scale);
int w34_qsa_sparse_paged_attn_hadamard_f16(
    const w34_exec* ex, const void* q, const void* gate,
    const w34_kv_layer_layout* kv_layout, const int32_t* selected_phys_blocks,
    const int32_t* selected_logical_blocks, const int32_t* num_selected_blocks,
    const int32_t* token_seq_pos, const uint8_t* valid_token_mask, void* out,
    int num_tokens, int local_q_heads, int select_k_blocks, float sm_scale);

// Pack and destination bases must be 16-byte aligned. Four destinations each
// contain dst_capacity expert regions; nonnegative dst_slots must be unique.
// Slots outside [0,dst_capacity) are skipped. Source packs use pack_stride_bytes.
int w34_moe_unpack_staged_experts(const w34_exec* ex, const void* staged_packs,
                                  const int32_t* dst_slots, uint32_t* gate_up_packed_base,
                                  void* gate_up_scales_base, uint32_t* down_packed_base,
                                  void* down_scales_base, int num_staged, int dst_capacity,
                                  size_t pack_stride_bytes, int n_interm, int k_hidden,
                                  int group_size, int bits);
int w34_moe_pull_mapped_experts(const w34_exec* ex, const void* mapped_host_packs,
                                const int32_t* src_local_experts, const int32_t* dst_slots,
                                uint32_t* gate_up_packed_base, void* gate_up_scales_base,
                                uint32_t* down_packed_base, void* down_scales_base,
                                int num_staged, int local_num_experts, int dst_capacity,
                                size_t pack_stride_bytes, int n_interm, int k_hidden,
                                int group_size, int bits);
int w34_moe_pull_mapped_experts_with_count(
    const w34_exec* ex, const void* mapped_host_packs, const int32_t* src_local_experts,
    const int32_t* dst_slots, const int32_t* num_staged_buf, uint32_t* gate_up_packed_base,
    void* gate_up_scales_base, uint32_t* down_packed_base, void* down_scales_base,
    int max_staged, int local_num_experts, int dst_capacity, size_t pack_stride_bytes,
    int n_interm, int k_hidden, int group_size, int bits);
int w34_moe_remap_topk_ids_i32(const w34_exec* ex, const int32_t* global_topk_ids,
                               const int32_t* expert_to_slot, int32_t* remapped_topk_ids,
                               int total_pairs, int global_expert_offset,
                               int local_num_experts);
// Source and destination must be disjoint and 4-byte aligned (half2).
int w34_add_inplace_f16(const w34_exec* ex, void* dst_f16, const void* src_f16,
                        int64_t num_elems);
int w34_midloop_damped_euler_f16(const w34_exec* ex, void* dst_f16,
                                 const void* x_prev_f16, const void* g_x_f16,
                                 int64_t num_elems, float alpha);
int w34_moe_gpu_lru_update_and_pull(
    const w34_exec* ex, const int32_t* global_topk_ids, int32_t* expert_to_slot,
    int32_t* slot_to_expert, uint64_t* slot_last_used, uint64_t* step_counter,
    int32_t* remapped_topk_ids, int32_t* miss_local_experts,
    int32_t* miss_victim_slots, int32_t* num_misses_buf,
    const void* mapped_host_packs, uint32_t* gate_up_packed_base,
    void* gate_up_scales_base, uint32_t* down_packed_base,
    void* down_scales_base, int total_pairs, int global_expert_offset,
    int local_num_experts, int k_hot, int max_misses,
    size_t pack_stride_bytes, int n_interm, int k_hidden, int group_size, int bits);

int w34_gemma_rms_norm_hadamard_f16(
    const w34_exec* ex, const void* x, const void* weight, void* out, void* out_had64,
    const uint8_t* valid_token_mask, int num_tokens, int hidden_size, float eps);
int w34_fused_add_gemma_rms_norm_tap_hadamard_f16(
    const w34_exec* ex, const void* x, void* residual, const void* weight, void* out,
    void* out_had64, void* tap_dst_bf16, const uint8_t* valid_token_mask,
    int num_tokens, int hidden_size, float eps, int tap_row_stride, int tap_col_offset);
int w34_tp2_all_reduce_residual_gemma_rms_norm_hadamard_f16(
    const w34_exec* ex, const void* local_in, void* ar_out, void* residual,
    const void* weight, void* out, void* out_had64, void* tap_dst_bf16,
    void* stage0, void* stage1, void* sig0, void* sig1, int rank, int num_tokens,
    int hidden_size, float eps, int tap_row_stride, int tap_col_offset);
int w34_tp4_all_reduce_residual_gemma_rms_norm_hadamard_f16(
    const w34_exec* ex, const void* local_in, void* ar_out, void* residual,
    const void* weight, void* out, void* out_had64, void* tap_dst_bf16,
    void* stage0, void* stage1, void* stage2, void* stage3,
    void* sig0, void* sig1, void* sig2, void* sig3, int rank, int num_tokens,
    int hidden_size, float eps, int tap_row_stride, int tap_col_offset);
int w34_rms_norm_gated_silu_strided_hadamard_f16(
    const w34_exec* ex, const void* x, const void* z, const void* weight, void* out,
    const uint8_t* valid_token_mask, int num_tokens, int num_v_heads, int head_dim,
    int z_token_stride, float eps);
int w34_rms_norm_gated_sigmoid_strided_hadamard_f16(
    const w34_exec* ex, const void* x, const void* z, const void* weight, void* out,
    const uint8_t* valid_token_mask, int num_tokens, int num_v_heads, int head_dim,
    int z_token_stride, float eps);
int w34_silu_and_mul_hadamard_f16(
    const w34_exec* ex, const void* gate_up, void* out, const uint8_t* valid_token_mask,
    int num_tokens, int intermediate_size);
int w34_apply_attn_output_gate_hadamard_f16(
    const w34_exec* ex, const void* attn_out, const void* gate, void* out,
    const uint8_t* valid_token_mask, int num_tokens, int total_dim);
int w34_tp2_all_gather_topk_f32_i32(
    const w34_exec* ex, const float* local_vals, const int32_t* local_ids,
    float* global_vals, int32_t* global_ids, void* stage0, void* stage1,
    void* sig0, void* sig1, int rank, int num_rows, int top_k);
int w34_tp4_all_gather_topk_f32_i32(
    const w34_exec* ex, const float* local_vals, const int32_t* local_ids,
    float* global_vals, int32_t* global_ids, void* stage0, void* stage1,
    void* stage2, void* stage3, void* sig0, void* sig1, void* sig2, void* sig3,
    int rank, int num_rows, int top_k);
int w34_tp2_all_reduce_hc_combine_tap_f16(
    const w34_exec* ex, const void* residual_in, const void* local_block_out,
    void* ar_out, const float* inject_weights, const void* ple_delta,
    void* residual_out, void* tap_bf16_out, const uint8_t* valid_token_mask,
    void* stage0, void* stage1, void* sig0, void* sig1, int rank, int num_tokens,
    int hc_count, int hidden_size, int tap_stride_elems, int tap_offset_elems);
int w34_tp4_all_reduce_hc_combine_tap_f16(
    const w34_exec* ex, const void* residual_in, const void* local_block_out,
    void* ar_out, const float* inject_weights, const void* ple_delta,
    void* residual_out, void* tap_bf16_out, const uint8_t* valid_token_mask,
    void* stage0, void* stage1, void* stage2, void* stage3,
    void* sig0, void* sig1, void* sig2, void* sig3, int rank, int num_tokens,
    int hc_count, int hidden_size, int tap_stride_elems, int tap_offset_elems);
int w34_lora_shrink_expand_f16(
    const w34_exec* ex, const void* x_f16, const void* lora_a_stacked,
    const void* lora_b_stacked, const int32_t* token_lora_ids,
    float* lowrank_f32_ws, void* y_inout_f16, int m, int size_n, int size_k,
    int rank_pad, int num_adapters, int default_lora_id, uint64_t adapter_mask);
int w34_sampler_shard_fast_greedy_f16(
    const w34_exec* ex, const void* logits_f16, const w34_sampler_row_cfg* row_cfgs,
    const uint32_t* bitmask_u32, const w34_sampler_sparse_entry* sparse_entries,
    float* out_local_vals, int32_t* out_local_ids, int num_rows,
    int local_vocab_size, int vocab_offset);

#ifdef __cplusplus
}
#endif
