// Descriptor-Driven Paged Attention and KV Cache Write kernels for warp-34.
// Supports w34_kv_layer_layout (FP16, BF16, TQ4 132B/row, Q8_0 272B/row) and
// w34_attention_batch (context_lens_before vs seq_lens_after, block_tables,
// write_slot_mapping, valid_token_mask, Causal and SymmetricWindow masks).
#include "../w34_cuda.h"
#include "../tq/tq4_codebook.cuh"

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <float.h>
#include <math.h>

namespace w34_attn {

// block_tq4_0: 66 bytes per 128 elements (2 blocks = 132 B for head_dim = 256).
#pragma pack(push, 1)
struct block_tq4_0 {
    __half  d;
    uint8_t qs[64];
};
static_assert(sizeof(block_tq4_0) == 66, "block_tq4_0 must be 66 bytes");

// block_q8_0_64: 68 bytes per 64 elements (4 blocks = 272 B for head_dim = 256).
struct block_q8_0_64 {
    float  d;
    int8_t qs[64];
};
static_assert(sizeof(block_q8_0_64) == 68, "block_q8_0_64 must be 68 bytes");
#pragma pack(pop)

__device__ __forceinline__ float warp_reduce_sum(float v) {
#pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
        v += __shfl_xor_sync(0xffffffffu, v, off);
    }
    return v;
}

__device__ __forceinline__ float block_reduce_sum_128(float v, float* s_warp) {
    const int lane = threadIdx.x & 31;
    const int wid = threadIdx.x >> 5;
    v = warp_reduce_sum(v);
    if (lane == 0) s_warp[wid] = v;
    __syncthreads();
    v = (threadIdx.x < (blockDim.x >> 5)) ? s_warp[lane] : 0.0f;
    if (wid == 0) {
        v = warp_reduce_sum(v);
        if (lane == 0) s_warp[0] = v;
    }
    __syncthreads();
    return s_warp[0];
}

__device__ __forceinline__ float block_reduce_max_128(float v, float* s_warp) {
    const int lane = threadIdx.x & 31;
    const int wid = threadIdx.x >> 5;
#pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
        v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, off));
    }
    if (lane == 0) s_warp[wid] = v;
    __syncthreads();
    v = (threadIdx.x < (blockDim.x >> 5)) ? s_warp[lane] : -FLT_MAX;
    if (wid == 0) {
#pragma unroll
        for (int off = 16; off > 0; off >>= 1) {
            v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, off));
        }
        if (lane == 0) s_warp[0] = v;
    }
    __syncthreads();
    return s_warp[0];
}

// Writes one (token, kv_head) row into a paged slab according to w34_kv_layer_layout.
__device__ void write_kv_row(
    uint8_t* dst_row,
    int format,
    const __half* src_row,
    int head_dim,
    float* s_warp) {
    if (format == W34_KV_FORMAT_FP16) {
        __half* dst = (__half*)dst_row;
        for (int c = threadIdx.x; c < head_dim; c += blockDim.x) {
            dst[c] = src_row[c];
        }
        return;
    }
    if (format == W34_KV_FORMAT_BF16) {
        __nv_bfloat16* dst = (__nv_bfloat16*)dst_row;
        for (int c = threadIdx.x; c < head_dim; c += blockDim.x) {
            dst[c] = __float2bfloat16(__half2float(src_row[c]));
        }
        return;
    }
    if (format == W34_KV_FORMAT_TQ4) {
        // Quantize each 128-element sub-block into a 66-byte block_tq4_0
        const int num_sub = head_dim / 128;
        for (int sb = 0; sb < num_sub; sb++) {
            const __half* sub_src = src_row + sb * 128;
            block_tq4_0* blk = (block_tq4_0*)(dst_row + sb * sizeof(block_tq4_0));
            float sq = 0.0f;
            if (threadIdx.x < 128) {
                float x = __half2float(sub_src[threadIdx.x]);
                sq = x * x;
            }
            float norm_x = sqrtf(block_reduce_sum_128(sq, s_warp));
            float inv_norm = (norm_x > 1e-12f) ? (1.0f / norm_x) : 0.0f;

            // Find nearest Beta(256) centroid for unit-normalized x
            uint8_t q_idx = 8;
            float c_val = 0.0f;
            if (threadIdx.x < 128) {
                float u = __half2float(sub_src[threadIdx.x]) * inv_norm;
                q_idx = 0;
#pragma unroll
                for (int b = 0; b < 15; b++) {
                    if (u > kTq4D256Boundaries[b]) q_idx = (uint8_t)(b + 1);
                }
                c_val = kTq4D256Centroids[q_idx];
            }
            float c_sq = block_reduce_sum_128(c_val * c_val, s_warp);
            float norm_c = sqrtf(c_sq);
            float scale_d = (norm_c > 1e-12f) ? (norm_x / norm_c) : 0.0f;
            if (threadIdx.x == 0) {
                blk->d = __float2half(scale_d);
            }
            // Pack pairs of nibbles (thread i in 0..64 packs elem 2*i and 2*i+1)
            __shared__ uint8_t s_q[128];
            if (threadIdx.x < 128) s_q[threadIdx.x] = q_idx;
            __syncthreads();
            if (threadIdx.x < 64) {
                uint8_t lo = s_q[2 * threadIdx.x] & 0x0Fu;
                uint8_t hi = (s_q[2 * threadIdx.x + 1] & 0x0Fu) << 4;
                blk->qs[threadIdx.x] = lo | hi;
            }
            __syncthreads();
        }
        return;
    }
    if (format == W34_KV_FORMAT_Q8_0) {
        const int num_sub = head_dim / 64;
        for (int sb = 0; sb < num_sub; sb++) {
            const __half* sub_src = src_row + sb * 64;
            block_q8_0_64* blk = (block_q8_0_64*)(dst_row + sb * sizeof(block_q8_0_64));
            float abs_v = 0.0f;
            if (threadIdx.x < 64) {
                abs_v = fabsf(__half2float(sub_src[threadIdx.x]));
            }
            float amax = block_reduce_max_128(abs_v, s_warp);
            float d = amax / 127.0f;
            float inv_d = (d > 1e-12f) ? (1.0f / d) : 0.0f;
            if (threadIdx.x == 0) blk->d = d;
            if (threadIdx.x < 64) {
                float qf = roundf(__half2float(sub_src[threadIdx.x]) * inv_d);
                qf = fminf(127.0f, fmaxf(-127.0f, qf));
                blk->qs[threadIdx.x] = (int8_t)qf;
            }
            __syncthreads();
        }
    }
}

__device__ __forceinline__ float read_kv_elem(
    const uint8_t* row_ptr,
    int format,
    int col) {
    if (format == W34_KV_FORMAT_FP16) {
        return __half2float(((const __half*)row_ptr)[col]);
    }
    if (format == W34_KV_FORMAT_BF16) {
        return __bfloat162float(((const __nv_bfloat16*)row_ptr)[col]);
    }
    if (format == W34_KV_FORMAT_TQ4) {
        const int sb = col >> 7;        // col / 128
        const int idx = col & 127;       // col % 128
        const block_tq4_0* blk = (const block_tq4_0*)(row_ptr + sb * sizeof(block_tq4_0));
        const float d = __half2float(blk->d);
        const uint8_t packed = blk->qs[idx >> 1];
        const uint8_t nib = (idx & 1) ? (packed >> 4) : (packed & 0x0Fu);
        return d * kTq4D256Centroids[nib];
    }
    if (format == W34_KV_FORMAT_Q8_0) {
        const int sb = col >> 6;        // col / 64
        const int idx = col & 63;        // col % 64
        const block_q8_0_64* blk = (const block_q8_0_64*)(row_ptr + sb * sizeof(block_q8_0_64));
        return blk->d * (float)blk->qs[idx];
    }
    return 0.0f;
}

// Grid: (total_tokens, local_kv_heads), Block: 128 threads
__global__ void kv_cache_write_kernel(
    w34_kv_layer_layout layout,
    w34_attention_batch batch,
    const __half* __restrict__ k_in,
    const __half* __restrict__ v_in) {
    const int tok = blockIdx.x;
    if (tok >= batch.total_tokens) return;
    if (batch.valid_token_mask != nullptr && batch.valid_token_mask[tok] == 0) return;

    const int32_t slot = batch.write_slot_mapping[tok];
    if (slot < 0) return;

    const int kv_head = blockIdx.y;
    const int page_id = slot / layout.block_size;
    const int page_off = slot % layout.block_size;

    uint8_t* k_dst = (uint8_t*)(uintptr_t)layout.k_base
                     + (size_t)page_id * layout.k_page_stride_bytes
                     + (size_t)kv_head * layout.k_head_stride_bytes
                     + (size_t)page_off * layout.k_token_stride_bytes;
    uint8_t* v_dst = (uint8_t*)(uintptr_t)layout.v_base
                     + (size_t)page_id * layout.v_page_stride_bytes
                     + (size_t)kv_head * layout.v_head_stride_bytes
                     + (size_t)page_off * layout.v_token_stride_bytes;

    const __half* k_src = k_in + ((size_t)tok * layout.local_kv_heads + kv_head) * layout.head_dim;
    const __half* v_src = v_in + ((size_t)tok * layout.local_kv_heads + kv_head) * layout.head_dim;

    __shared__ float s_warp[8];
    write_kv_row(k_dst, layout.k_format, k_src, layout.head_dim, s_warp);
    write_kv_row(v_dst, layout.v_format, v_src, layout.head_dim, s_warp);
}

// Grid: (total_tokens, num_q_heads), Block: 128 threads.
// Uses online FP32 softmax across the paged sequence, deterministic per-step reduction
// so output is bit-identical regardless of physical page permutation in block_tables!
__global__ void paged_attention_f16_kernel(
    w34_kv_layer_layout layout,
    w34_attention_batch batch,
    const __half* __restrict__ q_in,
    __half* __restrict__ attn_out,
    int num_q_heads) {
    const int tok = blockIdx.x;
    if (tok >= batch.total_tokens) return;
    if (batch.valid_token_mask != nullptr && batch.valid_token_mask[tok] == 0) return;

    // Locate owning request r via query_start_loc[r] <= tok < query_start_loc[r+1]
    int req = -1;
    int q_start = 0;
    int q_end = 0;
    for (int r = 0; r < batch.num_reqs; r++) {
        int lo = batch.query_start_loc[r];
        int hi = batch.query_start_loc[r + 1];
        if (tok >= lo && tok < hi) {
            req = r;
            q_start = lo;
            q_end = hi;
            break;
        }
    }
    if (req < 0) return;
    if (batch.valid_req_mask != nullptr && batch.valid_req_mask[req] == 0) return;

    const int q_head = blockIdx.y;
    const int gqa_ratio = num_q_heads / layout.local_kv_heads;
    const int kv_head = q_head / gqa_ratio;
    const int head_dim = layout.head_dim;

    const int q_offset_in_req = tok - q_start;
    const int ctx_before = batch.context_lens_before != nullptr
                               ? batch.context_lens_before[req]
                               : (batch.seq_lens_after[req] - (q_end - q_start));
    const int seq_after = batch.seq_lens_after[req];

    // Causal vs symmetric non-causal sliding window (DFlash2)
    int max_kv_pos = (batch.mask_mode == W34_ATTN_MASK_SYMMETRIC_WINDOW)
                         ? (seq_after - 1)
                         : (ctx_before + q_offset_in_req);
    if (max_kv_pos >= seq_after) max_kv_pos = seq_after - 1;

    int min_kv_pos = 0;
    if (batch.sliding_window > 0) {
        int ref_pos = (batch.mask_mode == W34_ATTN_MASK_SYMMETRIC_WINDOW)
                          ? (seq_after - 1)
                          : max_kv_pos;
        int win_start = ref_pos - batch.sliding_window + 1;
        if (win_start > min_kv_pos) min_kv_pos = win_start;
    }

    const __half* q_row = q_in + ((size_t)tok * num_q_heads + q_head) * head_dim;
    __half* out_row = attn_out + ((size_t)tok * num_q_heads + q_head) * head_dim;

    // 4 warps (128 threads): each warp `wid` (0..3) processes kv_pos = min_kv_pos + wid + 4*step
    // using register-only warp shuffles (0 __syncthreads() inside the KV loop), then merges
    // online-softmax states across warps 0..3 at the end.
    const int lane = threadIdx.x & 31;
    const int wid = threadIdx.x >> 5;  // 0..3

    float q_reg[8];
    float acc[8];
#pragma unroll
    for (int j = 0; j < 8; j++) {
        const int col = lane + 32 * j;
        q_reg[j] = (col < head_dim) ? __half2float(q_row[col]) : 0.0f;
        acc[j] = 0.0f;
    }

    float m_i = -FLT_MAX;
    float l_i = 0.0f;

    const int32_t* req_blocks = batch.block_tables + (size_t)req * batch.max_blocks_per_req;

    for (int kv_pos = min_kv_pos + wid; kv_pos <= max_kv_pos; kv_pos += 4) {
        const int logical_block = kv_pos / layout.block_size;
        const int page_off = kv_pos % layout.block_size;
        const int32_t phys_page = req_blocks[logical_block];

        const uint8_t* k_ptr = (const uint8_t*)(uintptr_t)layout.k_base
                               + (size_t)phys_page * layout.k_page_stride_bytes
                               + (size_t)kv_head * layout.k_head_stride_bytes
                               + (size_t)page_off * layout.k_token_stride_bytes;
        const uint8_t* v_ptr = (const uint8_t*)(uintptr_t)layout.v_base
                               + (size_t)phys_page * layout.v_page_stride_bytes
                               + (size_t)kv_head * layout.v_head_stride_bytes
                               + (size_t)page_off * layout.v_token_stride_bytes;

        float dot = 0.0f;
#pragma unroll
        for (int j = 0; j < 8; j++) {
            const int col = lane + 32 * j;
            if (col < head_dim) {
                dot += q_reg[j] * read_kv_elem(k_ptr, layout.k_format, col);
            }
        }
#pragma unroll
        for (int off = 16; off > 0; off >>= 1) {
            dot += __shfl_xor_sync(0xffffffffu, dot, off);
        }
        const float score = dot * batch.sm_scale;

        const float m_new = fmaxf(m_i, score);
        const float alpha = expf(m_i - m_new);
        const float p = expf(score - m_new);
        l_i = l_i * alpha + p;
        m_i = m_new;

#pragma unroll
        for (int j = 0; j < 8; j++) {
            const int col = lane + 32 * j;
            if (col < head_dim) {
                acc[j] = acc[j] * alpha + p * read_kv_elem(v_ptr, layout.v_format, col);
            }
        }
    }

    __shared__ float s_m[4];
    __shared__ float s_l[4];
    __shared__ float s_acc[4][256];

    if (lane == 0) {
        s_m[wid] = m_i;
        s_l[wid] = l_i;
    }
#pragma unroll
    for (int j = 0; j < 8; j++) {
        const int col = lane + 32 * j;
        if (col < head_dim) {
            s_acc[wid][col] = acc[j];
        }
    }
    __syncthreads();

    float m_tot = -FLT_MAX;
#pragma unroll
    for (int w = 0; w < 4; w++) {
        if (s_l[w] > 0.0f) {
            m_tot = fmaxf(m_tot, s_m[w]);
        }
    }
    float l_tot = 0.0f;
    float w_scale[4];
#pragma unroll
    for (int w = 0; w < 4; w++) {
        if (s_l[w] > 0.0f) {
            float s = expf(s_m[w] - m_tot);
            w_scale[w] = s;
            l_tot += s_l[w] * s;
        } else {
            w_scale[w] = 0.0f;
        }
    }
    const float inv_l = (l_tot > 0.0f) ? (1.0f / l_tot) : 0.0f;
    for (int col = threadIdx.x; col < head_dim; col += blockDim.x) {
        float sum_v = s_acc[0][col] * w_scale[0]
                    + s_acc[1][col] * w_scale[1]
                    + s_acc[2][col] * w_scale[2]
                    + s_acc[3][col] * w_scale[3];
        out_row[col] = __float2half(sum_v * inv_l);
    }
}

}  // namespace w34_attn

extern "C" int w34_kv_cache_write(const w34_exec* ex, const w34_kv_layer_layout* layout,
                                  const w34_attention_batch* batch,
                                  const void* k_in, const void* v_in) {
    if (ex == nullptr || layout == nullptr || batch == nullptr || k_in == nullptr || v_in == nullptr) {
        return (int)cudaErrorInvalidValue;
    }
    if (batch->total_tokens <= 0 || layout->local_kv_heads <= 0) return 0;
    dim3 grid((unsigned)batch->total_tokens, (unsigned)layout->local_kv_heads);
    w34_attn::kv_cache_write_kernel<<<grid, 128, 0, (cudaStream_t)ex->stream>>>(
        *layout, *batch, (const __half*)k_in, (const __half*)v_in);
    return (int)cudaGetLastError();
}

extern "C" int w34_paged_attention_f16(const w34_exec* ex, const w34_kv_layer_layout* layout,
                                       const w34_attention_batch* batch,
                                       const void* q_in, void* attn_out,
                                       int num_q_heads) {
    if (ex == nullptr || layout == nullptr || batch == nullptr || q_in == nullptr || attn_out == nullptr) {
        return (int)cudaErrorInvalidValue;
    }
    if (batch->total_tokens <= 0 || num_q_heads <= 0) return 0;
    dim3 grid((unsigned)batch->total_tokens, (unsigned)num_q_heads);
    w34_attn::paged_attention_f16_kernel<<<grid, 128, 0, (cudaStream_t)ex->stream>>>(
        *layout, *batch, (const __half*)q_in, (__half*)attn_out, num_q_heads);
    return (int)cudaGetLastError();
}
