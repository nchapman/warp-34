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
// C[M,N] = half(A[M,K] @ B[N,K]^T + bias[N]); positive dimensions required.
int w34_cublas_gemm_f16_bias(uintptr_t handle, const void* a, const void* b,
                            const void* bias, void* c, int m, int n, int k);
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

// --- Gated DeltaNet (op_gdn.cu) ---
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

// --- Phase 4: MTP & DFlash2 Speculative Decoding Kernels (op_dflash2.cu) ---
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

// --- Phase 5.2: Qwen3_VisionTransformer & Multimodal Kernels (op_vision.cu) ---
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

// --- Phase 5.3: Qwen3.5 / 3.6 Hybrid MoE Kernels (op_moe.cu) ---
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

// --- Cloudflare Clef JointSchemaHead Kernels (op_clef.cu) ---
int w34_clef_accum_spans_f16_to_f32(const w34_exec* ex, const void* chunk_normed_f16,
                                    const int32_t* span_bounds, float* span_accum_f32,
                                    int chunk_start, int chunk_len, int num_spans,
                                    int hidden_size);
int w34_clef_cast_f32_to_f16(const w34_exec* ex, const float* src_f32, void* dst_f16,
                             int64_t num_elems);
int w34_clef_build_option_queries_f16(const w34_exec* ex, const void* ctx_proj_f16,
                                      const void* lex_proj_f16, const void* opt_q_proj_f16,
                                      const int32_t* opt_q_idx, void* out_f16,
                                      int total_opts, int num_q, int width);
int w34_clef_mha_attn_f16(const w34_exec* ex, const void* q_f16, const void* q_bias_f16,
                          const void* kv_f16, const void* kv_bias_f16, void* out_f16,
                          int q_len, int m_len, int heads, int head_dim);
int w34_clef_option_summaries_f16(const w34_exec* ex, const void* routed_opts_f16,
                                  const void* base_fields_f16, const int32_t* q_opt_offsets,
                                  void* out_summaries_f16, int num_q, int total_opts,
                                  int width);
int w34_clef_build_fields_f16(const w34_exec* ex, const void* base_fields_f16,
                              const void* norm_summaries_f16, const void* global_proj_f16,
                              const void* type_embed_f16, const int32_t* q_types,
                              void* out_fields_f16, int num_q, int width);
int w34_clef_build_residual_features_f16(const w34_exec* ex, const void* fields_f16,
                                         const void* options_normed_f16,
                                         const int32_t* opt_q_idx, void* out_features_f16,
                                         int total_opts, int num_q, int width);

#ifdef __cplusplus
}
#endif
