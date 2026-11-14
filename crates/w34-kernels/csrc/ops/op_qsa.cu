// Qwen3.8-Flash-Next (qwen4_exp) Query-Sparse Attention (QSA) & Indexer CUDA Kernels.
// Reference: nvidia/qsa_indexer.py, nvidia/qsa.py, nvidia/ops/qsa.py, nvidia/ops/qsa_indexer.py
//   - Pre-indexer GemmaRMSNorm + partial interleaved NeoX mRoPE (head_dim=128, rotary_dim=64, [11,11,10])
//   - Paged raw key cache [num_pages, 16, 128] + 4:1 compressed key cache [num_pages, 4, 128]
//   - ReLU-sum MQA scoring + top-k block selector (k=64 blocks = 1024 tokens)
//   - Sparse paged GQA over selected blocks (FP16, Q8_0 272B/row, and TQ4 132B/row with Rademacher+WHT256)
//     with causal boundary masking & optional sigmoid output gate
#include "../w34_cuda.h"
#include "../tq/tq4_codebook.cuh"
#include "hadamard_inline.cuh"

#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>
#include <float.h>
#include <math.h>

extern "C" __device__ float __nv_sinf(float);
extern "C" __device__ float __nv_cosf(float);
extern "C" __device__ float __nv_powf(float, float);

namespace w34 {

// Naturally 2-byte aligned (`__half d` + 64-byte `qs`); do NOT use `#pragma pack(push, 1)`
// here because 1-byte packing triggers a CUDA 13.0 `ptxas` peephole bug (`F2I.U8.F16.TRUNC.NTZ`)
// when storing the low byte of `__half d`.
struct qsa_block_tq4_0 {
    __half  d;
    uint8_t qs[64];
};
static_assert(sizeof(qsa_block_tq4_0) == 66, "qsa_block_tq4_0 must be 66 bytes");
static_assert(alignof(qsa_block_tq4_0) == 2, "qsa_block_tq4_0 must be 2-byte aligned");

struct qsa_block_q8_0_64 {
    float  d;
    int8_t qs[64];
};
static_assert(sizeof(qsa_block_q8_0_64) == 68, "qsa_block_q8_0_64 must be 68 bytes");
static_assert(alignof(qsa_block_q8_0_64) == 4, "qsa_block_q8_0_64 must be 4-byte aligned");

struct qsa_block_fp8_e4m3_256 {
    uint8_t qs[256];
    float   d0;
    float   d1;
};
static_assert(sizeof(qsa_block_fp8_e4m3_256) == 264, "qsa_block_fp8_e4m3_256 must be 264 bytes");
static_assert(alignof(qsa_block_fp8_e4m3_256) == 4, "qsa_block_fp8_e4m3_256 must be 4-byte aligned");

inline bool qsa_current_device_supports_fp8_kv() {
    w34_device_props p{};
    if (w34_get_current_device_props(&p) != 0) return false;
    return p.compute_major > 8 || (p.compute_major == 8 && p.compute_minor >= 9);
}

__device__ __forceinline__ float qsa_warp_reduce_sum(float val) {
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        val += __shfl_xor_sync(0xffffffffu, val, offset);
    }
    return val;
}

__device__ __forceinline__ void qsa_wht256_warp(float (&v)[8], int lane) {
#pragma unroll
    for (int j = 0; j < 8; ++j) {
        float val = v[j];
#pragma unroll
        for (int h = 1; h < 32; h <<= 1) {
            float o = __shfl_xor_sync(0xffffffffu, val, h);
            val = (lane & h) ? (o - val) : (val + o);
        }
        v[j] = val;
    }
#pragma unroll
    for (int j = 0; j < 8; j += 2) {
        float a = v[j], b = v[j + 1];
        v[j] = a + b;
        v[j + 1] = a - b;
    }
#pragma unroll
    for (int g = 0; g < 8; g += 4) {
#pragma unroll
        for (int i = 0; i < 2; ++i) {
            float a = v[g + i], b = v[g + i + 2];
            v[g + i] = a + b;
            v[g + i + 2] = a - b;
        }
    }
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        float a = v[i], b = v[i + 4];
        v[i] = (a + b) * 0.0625f;
        v[i + 4] = (a - b) * 0.0625f;
    }
}

// Grid: (num_tokens, q_heads + kv_heads), Block: 128 (1 thread per dim of head_dim=128).
__global__ void qsa_indexer_norm_rope_f16_kernel(
    const __half* __restrict__ qk_in,
    const __half* __restrict__ q_norm_weight,
    const __half* __restrict__ k_norm_weight,
    const int32_t* __restrict__ positions,
    __half* __restrict__ q_out,
    __half* __restrict__ k_out,
    const uint8_t* __restrict__ valid_token_mask,
    int num_tokens,
    int q_heads,
    int kv_heads,
    int head_dim,
    int rotary_dim,
    float rope_theta,
    float eps,
    int mrope_s1,
    int mrope_s2,
    int pos_plane_stride) {
    const int tok = blockIdx.x;
    const int h = blockIdx.y;
    const int total_heads = q_heads + kv_heads;
    if (tok >= num_tokens || h >= total_heads) return;

    const bool is_q = (h < q_heads);
    __half* dst_head = is_q
        ? (q_out + ((size_t)tok * q_heads + h) * head_dim)
        : (k_out + ((size_t)tok * kv_heads + (h - q_heads)) * head_dim);

    if (valid_token_mask != nullptr && valid_token_mask[tok] == 0) {
        for (int d = threadIdx.x; d < head_dim; d += blockDim.x) {
            dst_head[d] = __float2half(0.0f);
        }
        return;
    }

    const __half* src_head = qk_in + ((size_t)tok * total_heads + h) * head_dim;
    if (!is_q) {
        // Upstream vLLM stores raw (pre-norm, pre-RoPE) keys in raw_key_cache;
        // k_layernorm + RoPE(first_position) are applied AFTER 4:1 mean-pooling.
        for (int d = threadIdx.x; d < head_dim; d += blockDim.x) {
            dst_head[d] = src_head[d];
        }
        return;
    }
    const __half* norm_w = q_norm_weight;

    __shared__ float s_warp[8];
    __shared__ float s_normed[128];

    float sum_sq = 0.0f;
    for (int d = threadIdx.x; d < head_dim; d += blockDim.x) {
        const float v = __half2float(src_head[d]);
        sum_sq += v * v;
    }
    sum_sq = qsa_warp_reduce_sum(sum_sq);
    const int lane = threadIdx.x & 31;
    const int wid = threadIdx.x >> 5;
    if (lane == 0) s_warp[wid] = sum_sq;
    __syncthreads();
    const int num_warps = blockDim.x >> 5;
    sum_sq = (threadIdx.x < num_warps) ? s_warp[lane] : 0.0f;
    if (wid == 0) {
        sum_sq = qsa_warp_reduce_sum(sum_sq);
        if (lane == 0) s_warp[0] = sum_sq;
    }
    __syncthreads();
    const float inv_rms = rsqrtf(s_warp[0] / (float)head_dim + eps);

    for (int d = threadIdx.x; d < head_dim; d += blockDim.x) {
        const float v = __half2float(src_head[d]) * inv_rms;
        const float w = 1.0f + __half2float(norm_w[d]);
        s_normed[d] = __half2float(__float2half(v * w));
    }
    __syncthreads();

    const int half_rot = rotary_dim / 2;
    for (int i = threadIdx.x; i < half_rot; i += blockDim.x) {
        int plane = 0;
        if ((i % 3) == 1 && i < mrope_s1 * 3) {
            plane = 1;
        } else if ((i % 3) == 2 && i < mrope_s2 * 3) {
            plane = 2;
        }
        const float pos = (float)positions[(size_t)plane * pos_plane_stride + tok];
        const float inv_freq =
            __fdiv_rn(1.0f, __nv_powf(rope_theta, (float)(2 * i) / (float)rotary_dim));
        const float angle = pos * inv_freq;
        float sin_a = __half2float(__float2half_rn(__nv_sinf(angle)));
        float cos_a = __half2float(__float2half_rn(__nv_cosf(angle)));

        const float x1 = s_normed[i];
        const float x2 = s_normed[i + half_rot];
        dst_head[i] = __float2half(x1 * cos_a - x2 * sin_a);
        dst_head[i + half_rot] = __float2half(x2 * cos_a + x1 * sin_a);
    }
    for (int d = rotary_dim + threadIdx.x; d < head_dim; d += blockDim.x) {
        dst_head[d] = __float2half(s_normed[d]);
    }
}

// Writes indexer keys [num_tokens, head_dim] into raw_k_cache [num_pages, 16, head_dim + 8]
// at write_slot_mapping[tok], and on group-leader slots (`slot & 3 == 0`) caches the 3-plane
// mRoPE coordinates of the group's first token in the 8-half metadata tail.
__global__ void qsa_indexer_write_raw_k_f16_kernel(
    const __half* __restrict__ k_in,
    __half* __restrict__ raw_k_cache,
    const int32_t* __restrict__ positions,
    const int32_t* __restrict__ write_slot_mapping,
    const uint8_t* __restrict__ valid_token_mask,
    int num_tokens,
    int head_dim,
    int pos_plane_stride) {
    const int tok = blockIdx.x;
    if (tok >= num_tokens) return;
    if (valid_token_mask != nullptr && valid_token_mask[tok] == 0) return;
    const int32_t slot = write_slot_mapping[tok];
    if (slot < 0) return;

    const int raw_stride = head_dim + 8;
    const __half* src = k_in + (size_t)tok * head_dim;
    __half* dst = raw_k_cache + (size_t)slot * raw_stride;
    for (int d = threadIdx.x; d < head_dim; d += blockDim.x) {
        dst[d] = src[d];
    }
    if ((slot & 3) == 0 && positions != nullptr) {
        int32_t* meta = reinterpret_cast<int32_t*>(dst + head_dim);
        if (threadIdx.x < 3) {
            const int plane = (pos_plane_stride > 0) ? threadIdx.x : 0;
            meta[threadIdx.x] = positions[(size_t)plane * pos_plane_stride + tok];
        } else if (threadIdx.x == 3) {
            meta[3] = 0x51534130; // cached group-first coordinates valid marker
        }
    }
}

// Updates 4:1 compressed key cache [num_pages, 4, head_dim] from raw_k_cache [num_pages, 16, head_dim + 8]
// using a single writer per completed 4-token compressed group (`(slot & 3) == 3`):
//   1. Mean-pools the 4 raw keys in the sub-block: pooled = 0.25 * (v0 + v1 + v2 + v3)
//   2. Applies GemmaRMSNorm(k_norm_weight, eps) + partial mRoPE at the sub-block's cached first
//      mRoPE coordinates (matching upstream vLLM QSAIndexer.forward / _qsa_prepare_kernel).
__global__ void qsa_indexer_update_comp_k_f16_kernel(
    const __half* __restrict__ raw_k_cache,
    __half* __restrict__ comp_k_cache,
    const __half* __restrict__ k_norm_weight,
    const int32_t* __restrict__ positions,
    const int32_t* __restrict__ write_slot_mapping,
    const uint8_t* __restrict__ valid_token_mask,
    int num_tokens,
    int head_dim,
    int rotary_dim,
    float rope_theta,
    float eps,
    int mrope_s1,
    int mrope_s2,
    int pos_plane_stride) {
    const int tok = blockIdx.x;
    if (tok >= num_tokens) return;
    if (valid_token_mask != nullptr && valid_token_mask[tok] == 0) return;
    const int32_t slot = write_slot_mapping[tok];
    if (slot < 0) return;

    const int raw_stride = head_dim + 8;
    const int32_t sub_block = slot >> 2; // slot / 4
    const int32_t base_slot = sub_block << 2;
    const __half* raw_base = raw_k_cache + (size_t)base_slot * raw_stride;
    __half* comp_dst = comp_k_cache + (size_t)sub_block * head_dim;

    // Single writer per 4-token compressed group: only the CTA writing the 4th slot (`slot & 3 == 3`)
    // commits the compressed group into `comp_k_cache`.
    if ((slot & 3) != 3) {
        return;
    }

    if (k_norm_weight == nullptr || positions == nullptr) {
        for (int d = threadIdx.x; d < head_dim; d += blockDim.x) {
            const float v0 = __half2float(raw_base[0 * raw_stride + d]);
            const float v1 = __half2float(raw_base[1 * raw_stride + d]);
            const float v2 = __half2float(raw_base[2 * raw_stride + d]);
            const float v3 = __half2float(raw_base[3 * raw_stride + d]);
            comp_dst[d] = __float2half(0.25f * (v0 + v1 + v2 + v3));
        }
        return;
    }

    __shared__ float s_warp[8];
    __shared__ float s_normed[128];

    float sum_sq = 0.0f;
    for (int d = threadIdx.x; d < head_dim; d += blockDim.x) {
        const float v0 = __half2float(raw_base[0 * raw_stride + d]);
        const float v1 = __half2float(raw_base[1 * raw_stride + d]);
        const float v2 = __half2float(raw_base[2 * raw_stride + d]);
        const float v3 = __half2float(raw_base[3 * raw_stride + d]);
        const float pooled = __half2float(__float2half(0.25f * (v0 + v1 + v2 + v3)));
        s_normed[d] = pooled;
        sum_sq += pooled * pooled;
    }
    sum_sq = qsa_warp_reduce_sum(sum_sq);
    const int lane = threadIdx.x & 31;
    const int wid = threadIdx.x >> 5;
    if (lane == 0) s_warp[wid] = sum_sq;
    __syncthreads();
    const int num_warps = blockDim.x >> 5;
    sum_sq = (threadIdx.x < num_warps) ? s_warp[lane] : 0.0f;
    if (wid == 0) {
        sum_sq = qsa_warp_reduce_sum(sum_sq);
        if (lane == 0) s_warp[0] = sum_sq;
    }
    __syncthreads();
    const float inv_rms = rsqrtf(s_warp[0] / (float)head_dim + eps);

    for (int d = threadIdx.x; d < head_dim; d += blockDim.x) {
        const float v = s_normed[d] * inv_rms;
        const float w = 1.0f + __half2float(k_norm_weight[d]);
        s_normed[d] = __half2float(__float2half(v * w));
    }
    __syncthreads();

    const int32_t* meta = reinterpret_cast<const int32_t*>(raw_base + head_dim);
    const bool has_cached_first = (meta[3] == 0x51534130);
    const int32_t offset_in_group = slot & 3;
    const int half_rot = rotary_dim / 2;
    for (int i = threadIdx.x; i < half_rot; i += blockDim.x) {
        int plane = 0;
        if ((i % 3) == 1 && i < mrope_s1 * 3) {
            plane = 1;
        } else if ((i % 3) == 2 && i < mrope_s2 * 3) {
            plane = 2;
        }
        const int eff_plane = (pos_plane_stride > 0) ? plane : 0;
        const float first_pos = has_cached_first
            ? (float)meta[eff_plane]
            : (float)(positions[(size_t)eff_plane * pos_plane_stride + tok] - offset_in_group);
        const float inv_freq =
            __fdiv_rn(1.0f, __nv_powf(rope_theta, (float)(2 * i) / (float)rotary_dim));
        const float angle = first_pos * inv_freq;
        float sin_a = __half2float(__float2half_rn(__nv_sinf(angle)));
        float cos_a = __half2float(__float2half_rn(__nv_cosf(angle)));

        const float x1 = s_normed[i];
        const float x2 = s_normed[i + half_rot];
        comp_dst[i] = __float2half(x1 * cos_a - x2 * sin_a);
        comp_dst[i + half_rot] = __float2half(x2 * cos_a + x1 * sin_a);
    }
    for (int d = rotary_dim + threadIdx.x; d < head_dim; d += blockDim.x) {
        comp_dst[d] = __float2half(s_normed[d]);
    }
}

__device__ __forceinline__ bool qsa_cand_better(
    float sc_a,
    int32_t blk_a,
    float sc_b,
    int32_t blk_b) {
    if (blk_a < 0) return false;
    if (blk_b < 0) return true;
    return (sc_a > sc_b) || (sc_a == sc_b && blk_a < blk_b);
}

// Parallel 512-element bitonic merge step in descending `qsa_cand_better` order across 256 threads.
__device__ __forceinline__ void qsa_bitonic_merge_512_desc(
    float* s_sc,
    int32_t* s_blk,
    int tid) {
#pragma unroll
    for (int j = 256; j > 0; j >>= 1) {
        const int low = tid & (j - 1);
        const int ix = ((tid - low) << 1) + low;
        const int iy = ix ^ j;
        const float sa = s_sc[ix];
        const float sb = s_sc[iy];
        const int32_t ba = s_blk[ix];
        const int32_t bb = s_blk[iy];
        if (qsa_cand_better(sb, bb, sa, ba)) {
            s_sc[ix] = sb;
            s_blk[ix] = bb;
            s_sc[iy] = sa;
            s_blk[iy] = ba;
        }
        __syncthreads();
    }
}

// Parallel 512-element bitonic sort in descending `qsa_cand_better` order across 256 threads.
__device__ __forceinline__ void qsa_bitonic_sort_512_desc(
    float* s_sc,
    int32_t* s_blk,
    int tid) {
    for (int k = 2; k <= 512; k <<= 1) {
        for (int j = k >> 1; j > 0; j >>= 1) {
            const int low = tid & (j - 1);
            const int ix = ((tid - low) << 1) + low;
            const int iy = ix ^ j;
            const bool desc = ((ix & k) == 0);
            const float sa = s_sc[ix];
            const float sb = s_sc[iy];
            const int32_t ba = s_blk[ix];
            const int32_t bb = s_blk[iy];
            const bool swap = desc
                ? qsa_cand_better(sb, bb, sa, ba)
                : qsa_cand_better(sa, ba, sb, bb);
            if (swap) {
                s_sc[ix] = sb;
                s_blk[ix] = bb;
                s_sc[iy] = sa;
                s_blk[iy] = ba;
            }
            __syncthreads();
        }
    }
}

// Parallel 512-element bitonic sort in ascending int32 order across 256 threads.
__device__ __forceinline__ void qsa_bitonic_sort_512_i32_asc(
    int32_t* s_blk,
    int tid) {
    for (int k = 2; k <= 512; k <<= 1) {
        for (int j = k >> 1; j > 0; j >>= 1) {
            const int low = tid & (j - 1);
            const int ix = ((tid - low) << 1) + low;
            const int iy = ix ^ j;
            const bool asc = ((ix & k) == 0);
            const int32_t ba = s_blk[ix];
            const int32_t bb = s_blk[iy];
            if (asc ? (ba > bb) : (ba < bb)) {
                s_blk[ix] = bb;
                s_blk[iy] = ba;
            }
            __syncthreads();
        }
    }
}

// Selects up to `select_k_blocks` 4-token compressed groups per query token using the QSA Indexer:
//   - Each entry in `out_selected_logical_blocks` is a logical 4-token group index `g` (covering tokens `4*g .. 4*g+4`).
//   - Each entry in `out_selected_phys_blocks` is the resolved physical 4-token group index
//     `phys_page * (block_size / 4) + (g % (block_size / 4))`.
//   - If all valid 4-token groups fit within `select_k_blocks` (and completed groups <= 512): selects all `0..num_valid_groups`.
//   - Otherwise: scores completed 4-token groups in 512-group batches across 8 warps (256 threads),
//     maintains the exact top-`need_mid` groups via parallel bitonic merge in shared memory, sorts the selected
//     logical group indices ascending in parallel, and appends any incomplete causal tail group.
// Grid: num_tokens, Block: 256.
__global__ void qsa_select_blocks_f16_kernel(
    const __half* __restrict__ q_indexer,
    const __half* __restrict__ comp_k_cache,
    w34_attention_batch batch,
    int32_t* __restrict__ out_selected_phys_blocks,
    int32_t* __restrict__ out_selected_logical_blocks,
    int32_t* __restrict__ out_num_selected_blocks,
    int32_t* __restrict__ out_token_seq_pos,
    int q_heads,
    int head_dim,
    int block_size,
    int select_k_blocks,
    int sink_blocks,
    int recent_blocks) {
    const int tok = blockIdx.x;
    if (tok >= batch.total_tokens) return;

    int32_t* dst_phys = out_selected_phys_blocks + (size_t)tok * select_k_blocks;
    int32_t* dst_log = out_selected_logical_blocks + (size_t)tok * select_k_blocks;

    if (batch.valid_token_mask != nullptr && batch.valid_token_mask[tok] == 0) {
        if (threadIdx.x == 0) {
            out_num_selected_blocks[tok] = 0;
            out_token_seq_pos[tok] = 0;
        }
        for (int i = threadIdx.x; i < select_k_blocks; i += blockDim.x) {
            dst_phys[i] = -1;
            dst_log[i] = -1;
        }
        return;
    }

    // Locate request owning `tok`.
    int req = -1;
    for (int r = 0; r < batch.num_reqs; ++r) {
        if (tok >= batch.query_start_loc[r] && tok < batch.query_start_loc[r + 1]) {
            req = r;
            break;
        }
    }
    if (req < 0 || (batch.valid_req_mask != nullptr && batch.valid_req_mask[req] == 0)) {
        if (threadIdx.x == 0) {
            out_num_selected_blocks[tok] = 0;
            out_token_seq_pos[tok] = 0;
        }
        for (int i = threadIdx.x; i < select_k_blocks; i += blockDim.x) {
            dst_phys[i] = -1;
            dst_log[i] = -1;
        }
        return;
    }
    const int q_start = batch.query_start_loc[req];
    const int ctx_before = (batch.context_lens_before != nullptr) ? batch.context_lens_before[req] : 0;
    const int token_pos = ctx_before + (tok - q_start);
    const int groups_per_page = block_size >> 2;
    const int num_comp_groups = (token_pos + 1) >> 2;
    const bool has_tail_group = ((token_pos + 1) & 3) != 0;
    const int num_valid_groups = num_comp_groups + (has_tail_group ? 1 : 0);
    const int32_t* req_block_table = batch.block_tables + (size_t)req * batch.max_blocks_per_req;

    if (threadIdx.x == 0) {
        out_token_seq_pos[tok] = token_pos;
    }

    const int max_comp = (select_k_blocks > 512) ? 512 : select_k_blocks;
    if (num_valid_groups <= select_k_blocks && num_comp_groups <= max_comp) {
        if (threadIdx.x == 0) {
            out_num_selected_blocks[tok] = num_valid_groups;
        }
        for (int g = threadIdx.x; g < select_k_blocks; g += blockDim.x) {
            if (g < num_valid_groups) {
                const int32_t phys_page = req_block_table[g / groups_per_page];
                const int sub_group = g % groups_per_page;
                dst_phys[g] = (phys_page >= 0) ? (phys_page * groups_per_page + sub_group) : -1;
                dst_log[g] = g;
            } else {
                dst_phys[g] = -1;
                dst_log[g] = -1;
            }
        }
        return;
    }

    const int tid = threadIdx.x;
    const int wid = tid >> 5;
    const int lane = tid & 31;
    const int num_warps = blockDim.x >> 5; // 8 warps

    // Load q_indexer[tok, 0..4, 0..128] into shared memory.
    __shared__ float s_q[4][128];
    const __half* q_tok = q_indexer + (size_t)tok * q_heads * head_dim;
    for (int idx = tid; idx < 4 * head_dim; idx += blockDim.x) {
        const int h = idx / head_dim;
        const int d = idx % head_dim;
        s_q[h][d] = (h < q_heads) ? __half2float(q_tok[idx]) : 0.0f;
    }

    __shared__ float s_top_sc[512];
    __shared__ int32_t s_top_blk[512];
    __shared__ float s_cand_sc[512];
    __shared__ int32_t s_cand_blk[512];

    s_top_sc[tid] = -FLT_MAX;
    s_top_blk[tid] = -1;
    s_top_sc[tid + 256] = -FLT_MAX;
    s_top_blk[tid + 256] = -1;
    __syncthreads();

    const float scale = rsqrtf((float)head_dim);
    const int comp_budget = has_tail_group
        ? ((select_k_blocks - 1 < max_comp) ? (select_k_blocks - 1) : max_comp)
        : max_comp;

    // Candidate middle completed groups are in [mid_start, mid_end).
    const int mid_start = (sink_blocks < num_comp_groups) ? sink_blocks : num_comp_groups;
    const int mid_end = ((num_comp_groups - recent_blocks) > mid_start)
        ? (num_comp_groups - recent_blocks)
        : mid_start;
    const int num_forced = mid_start + (num_comp_groups - mid_end);
    int need_mid = (comp_budget > num_forced) ? (comp_budget - num_forced) : 0;
    if (need_mid > 512) need_mid = 512;

    if (need_mid > 0 && mid_end > mid_start) {
        for (int chunk_base = mid_start; chunk_base < mid_end; chunk_base += 512) {
            const int rem = mid_end - chunk_base;
            const int chunk_cnt = (rem < 512) ? rem : 512;
            const bool is_first_chunk = (chunk_base == mid_start);
            float* dst_sc = is_first_chunk ? s_top_sc : s_cand_sc;
            int32_t* dst_blk = is_first_chunk ? s_top_blk : s_cand_blk;

            // 8 warps score up to 512 groups in 64 warp steps without inner barriers.
            for (int c_off = wid; c_off < chunk_cnt; c_off += num_warps) {
                const int g = chunk_base + c_off;
                float grp_score = -FLT_MAX;
                const int32_t phys_page = req_block_table[g / groups_per_page];
                const int sub_group = g % groups_per_page;
                if (phys_page >= 0) {
                    const __half* ck =
                        comp_k_cache + ((size_t)phys_page * groups_per_page + sub_group) * head_dim;
                    float dot_h[4] = {0.0f, 0.0f, 0.0f, 0.0f};
#pragma unroll
                    for (int i = 0; i < 4; ++i) {
                        const int col = lane + i * 32;
                        const float kv = __half2float(ck[col]);
                        dot_h[0] += s_q[0][col] * kv;
                        dot_h[1] += s_q[1][col] * kv;
                        dot_h[2] += s_q[2][col] * kv;
                        dot_h[3] += s_q[3][col] * kv;
                    }
                    float sum_relu = 0.0f;
#pragma unroll
                    for (int h = 0; h < 4; ++h) {
                        const float logit = qsa_warp_reduce_sum(dot_h[h]);
                        sum_relu += fmaxf(logit, 0.0f);
                    }
                    grp_score = sum_relu * scale;
                }
                if (lane == 0) {
                    dst_sc[c_off] = grp_score;
                    dst_blk[c_off] = g;
                }
            }

            if (is_first_chunk) {
                __syncthreads();
                qsa_bitonic_sort_512_desc(s_top_sc, s_top_blk, tid);
            } else {
                if (tid >= chunk_cnt) {
                    s_cand_sc[tid] = -FLT_MAX;
                    s_cand_blk[tid] = -1;
                }
                if (tid + 256 >= chunk_cnt) {
                    s_cand_sc[tid + 256] = -FLT_MAX;
                    s_cand_blk[tid + 256] = -1;
                }
                __syncthreads();

                const float thresh_sc = s_top_sc[need_mid - 1];
                const int32_t thresh_blk = s_top_blk[need_mid - 1];
                const bool b0 =
                    qsa_cand_better(s_cand_sc[tid], s_cand_blk[tid], thresh_sc, thresh_blk);
                const bool b1 = qsa_cand_better(
                    s_cand_sc[tid + 256], s_cand_blk[tid + 256], thresh_sc, thresh_blk);
                if (__syncthreads_or(b0 || b1)) {
                    qsa_bitonic_sort_512_desc(s_cand_sc, s_cand_blk, tid);
#pragma unroll
                    for (int half_idx = 0; half_idx < 512; half_idx += 256) {
                        const int i = tid + half_idx;
                        const int mirror = 511 - i;
                        if (qsa_cand_better(
                                s_cand_sc[mirror], s_cand_blk[mirror], s_top_sc[i], s_top_blk[i])) {
                            s_top_sc[i] = s_cand_sc[mirror];
                            s_top_blk[i] = s_cand_blk[mirror];
                        }
                    }
                    __syncthreads();
                    qsa_bitonic_merge_512_desc(s_top_sc, s_top_blk, tid);
                }
            }
        }
    }

    // Mask entries beyond `need_mid` (or invalid `< 0`) to INT32_MAX and sort ascending by logical group index.
#pragma unroll
    for (int half_idx = 0; half_idx < 512; half_idx += 256) {
        const int i = tid + half_idx;
        if (i >= need_mid || s_top_blk[i] < 0) {
            s_top_blk[i] = 0x7fffffff;
        }
    }
    __syncthreads();
    if (need_mid > 0 && mid_end > mid_start) {
        qsa_bitonic_sort_512_i32_asc(s_top_blk, tid);
    }

    const int avail_mid = (mid_end > mid_start) ? (mid_end - mid_start) : 0;
    const int valid_mid = (need_mid < avail_mid) ? need_mid : avail_mid;
    const int prefix_cnt = (mid_start < comp_budget) ? mid_start : comp_budget;
    const int rem_after_prefix = comp_budget - prefix_cnt;
    const int mid_cnt = (valid_mid < rem_after_prefix) ? valid_mid : rem_after_prefix;
    const int rem_after_mid = rem_after_prefix - mid_cnt;
    const int suffix_avail = num_comp_groups - mid_end;
    const int suffix_cnt = (suffix_avail < rem_after_mid) ? suffix_avail : rem_after_mid;
    const int comp_out = prefix_cnt + mid_cnt + suffix_cnt;
    const int total_out =
        (has_tail_group && comp_out < select_k_blocks) ? (comp_out + 1) : comp_out;

    if (tid == 0) {
        out_num_selected_blocks[tok] = total_out;
    }

    for (int out_idx = tid; out_idx < select_k_blocks; out_idx += blockDim.x) {
        if (out_idx < total_out) {
            int32_t g;
            if (out_idx < prefix_cnt) {
                g = out_idx;
            } else if (out_idx < prefix_cnt + mid_cnt) {
                g = s_top_blk[out_idx - prefix_cnt];
            } else if (out_idx < comp_out) {
                g = mid_end + (out_idx - prefix_cnt - mid_cnt);
            } else {
                g = num_comp_groups;
            }
            const int32_t phys_page = req_block_table[g / groups_per_page];
            dst_log[out_idx] = g;
            dst_phys[out_idx] =
                (phys_page >= 0) ? (phys_page * groups_per_page + (g % groups_per_page)) : -1;
        } else {
            dst_log[out_idx] = -1;
            dst_phys[out_idx] = -1;
        }
    }
}

// GQA-grouped + Split-K Sparse Paged Attention over the selected 4-token groups per token for
template <bool APPLY_HAD64 = false>
__device__ __forceinline__ void qsa_store_head256_warp(
    __half* __restrict__ dst_head,
    __half* __restrict__ s_head,
    int lane) {
    if constexpr (APPLY_HAD64) {
        __syncwarp();
#pragma unroll
        for (int blk = 0; blk < 2; ++blk) {
            const int idx4 = blk * 32 + lane;
            const uint2 v = reinterpret_cast<const uint2*>(s_head)[idx4];
            reinterpret_cast<uint2*>(dst_head)[idx4] = hadamard64_vec4_f16(v);
        }
    }
}

// FP16, Q8_0, and TQ4 KV caches (head_dim == 256, 1 <= q_group_size <= 6).
// All `q_group_size` Q heads in the sub-group share the same `kv_head` and run in the same CTA so
// each selected KV row is fetched and dequantized once instead of `q_group_size` times.
// Grid: (num_tokens, local_q_heads / q_group_size, num_splits), Block: 128 (4 warps).
template <bool APPLY_HAD64 = false>
__global__ void qsa_sparse_paged_attn_gqa_f16_kernel(
    const __half* __restrict__ q,
    const __half* __restrict__ gate,
    w34_kv_layer_layout kv_layout,
    const int32_t* __restrict__ selected_phys_blocks,
    const int32_t* __restrict__ selected_logical_blocks,
    const int32_t* __restrict__ num_selected_blocks,
    const int32_t* __restrict__ token_seq_pos,
    const uint8_t* __restrict__ valid_token_mask,
    __half* __restrict__ out,
    float* __restrict__ partial_m,
    float* __restrict__ partial_d,
    float* __restrict__ partial_acc,
    int num_tokens,
    int local_q_heads,
    int q_group_size,
    int select_k_blocks,
    int num_splits,
    float sm_scale) {
    constexpr int kMaxGqa = 6;
    const int tok = blockIdx.x;
    const int q_group_idx = blockIdx.y;
    const int split_id = blockIdx.z;
    const int q_head_base = q_group_idx * q_group_size;
    if (tok >= num_tokens || q_head_base >= local_q_heads || split_id >= num_splits) return;
    const int full_gqa = local_q_heads / kv_layout.local_kv_heads;
    const int kv_head = q_head_base / full_gqa;
    const int gqa_group = q_group_size;

    if (valid_token_mask != nullptr && valid_token_mask[tok] == 0) {
        if (num_splits == 1) {
            for (int h = 0; h < gqa_group; ++h) {
                const int q_head = q_head_base + h;
                __half* dst_head = out + ((size_t)tok * local_q_heads + q_head) * 256;
                for (int d = threadIdx.x; d < 256; d += blockDim.x) {
                    dst_head[d] = __float2half(0.0f);
                }
            }
        }
        return;
    }

    const int wid = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;

    // 16 KB shared workspace: aliased as `s_q[6][256]` during the KV loop and `s_work[4][4][256]`
    // during the two-pass 4-warp reduction at the end.
    __shared__ float s_work[4][4][256];
    __shared__ float s_m[kMaxGqa][4];
    __shared__ float s_d[kMaxGqa][4];
    float (*s_q)[256] = &s_work[0][0];

    for (int h = wid; h < gqa_group; h += 4) {
        const int q_head = q_head_base + h;
        const __half* q_head_ptr = q + ((size_t)tok * local_q_heads + q_head) * 256;
        float q_reg[8];
#pragma unroll
        for (int j = 0; j < 8; ++j) {
            const int col = lane + 32 * j;
            q_reg[j] = __half2float(q_head_ptr[col]) * sm_scale;
            if (kv_layout.k_transform == W34_KV_TRANSFORM_RADEMACHER_WHT256) {
                q_reg[j] *= kTq4Signs256[col];
            }
        }
        if (kv_layout.k_transform == W34_KV_TRANSFORM_RADEMACHER_WHT256) {
            qsa_wht256_warp(q_reg, lane);
        }
#pragma unroll
        for (int j = 0; j < 8; ++j) {
            s_q[h][lane + 32 * j] = q_reg[j];
        }
    }
    __syncthreads();

    const int n_sel = num_selected_blocks[tok];
    const int seq_pos = token_seq_pos[tok];
    const int groups_per_page = kv_layout.block_size >> 2;
    const int groups_per_split = (n_sel + num_splits - 1) / num_splits;
    const int grp_start = split_id * groups_per_split;
    const int grp_end =
        ((grp_start + groups_per_split) < n_sel) ? (grp_start + groups_per_split) : n_sel;
    const int t_start = grp_start * 4;
    const int t_end = (grp_end > grp_start) ? (grp_end * 4) : t_start;

    float acc[kMaxGqa][8];
    float m_i[kMaxGqa];
    float d_i[kMaxGqa];
#pragma unroll
    for (int h = 0; h < kMaxGqa; ++h) {
        m_i[h] = -FLT_MAX;
        d_i[h] = 0.0f;
#pragma unroll
        for (int j = 0; j < 8; ++j) {
            acc[h][j] = 0.0f;
        }
    }

    const int32_t* tok_phys = selected_phys_blocks + (size_t)tok * select_k_blocks;
    const int32_t* tok_log = selected_logical_blocks + (size_t)tok * select_k_blocks;
    const uint8_t* k_base = reinterpret_cast<const uint8_t*>((uintptr_t)kv_layout.k_base);
    const uint8_t* v_base = reinterpret_cast<const uint8_t*>((uintptr_t)kv_layout.v_base);

    for (int t_idx = t_start + wid; t_idx < t_end; t_idx += 4) {
        const int grp_idx = t_idx >> 2;
        const int grp_off = t_idx & 3;
        const int32_t phys_grp = tok_phys[grp_idx];
        const int32_t log_grp = tok_log[grp_idx];
        const int kv_pos = log_grp * 4 + grp_off;
        if (phys_grp < 0 || kv_pos > seq_pos) continue;

        const int32_t phys_page = phys_grp / groups_per_page;
        const int page_off = (phys_grp % groups_per_page) * 4 + grp_off;
        const uint8_t* k_ptr =
            k_base + (size_t)phys_page * kv_layout.k_page_stride_bytes +
            (size_t)kv_head * kv_layout.k_head_stride_bytes +
            (size_t)page_off * kv_layout.k_token_stride_bytes;
        const uint8_t* v_ptr =
            v_base + (size_t)phys_page * kv_layout.v_page_stride_bytes +
            (size_t)kv_head * kv_layout.v_head_stride_bytes +
            (size_t)page_off * kv_layout.v_token_stride_bytes;

        float k_val[8];
        if (kv_layout.k_format == W34_KV_FORMAT_TQ4) {
            const qsa_block_tq4_0* k_blks = reinterpret_cast<const qsa_block_tq4_0*>(k_ptr);
            const float kd0 = __half2float(k_blks[0].d);
            const float kd1 = __half2float(k_blks[1].d);
            const int odd = lane & 1;
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                const uint8_t p0 = k_blks[0].qs[(lane + 32 * j) >> 1];
                const uint8_t p1 = k_blks[1].qs[(lane + 32 * j) >> 1];
                const uint8_t n0 = odd ? (p0 >> 4) : (p0 & 0x0Fu);
                const uint8_t n1 = odd ? (p1 >> 4) : (p1 & 0x0Fu);
                k_val[j] = kd0 * kTq4D256Centroids[n0];
                k_val[j + 4] = kd1 * kTq4D256Centroids[n1];
            }
        } else if (kv_layout.k_format == W34_KV_FORMAT_Q8_0) {
            const qsa_block_q8_0_64* k_blks = reinterpret_cast<const qsa_block_q8_0_64*>(k_ptr);
#pragma unroll
            for (int sb = 0; sb < 4; ++sb) {
                const float kd = k_blks[sb].d;
                k_val[2 * sb] = kd * (float)k_blks[sb].qs[lane];
                k_val[2 * sb + 1] = kd * (float)k_blks[sb].qs[lane + 32];
            }
#if __CUDA_ARCH__ >= 890
        } else if (kv_layout.k_format == W34_KV_FORMAT_FP8_E4M3) {
            const qsa_block_fp8_e4m3_256* k_blk = reinterpret_cast<const qsa_block_fp8_e4m3_256*>(k_ptr);
            const float kd0 = k_blk->d0;
            const float kd1 = k_blk->d1;
#pragma unroll
            for (int sb = 0; sb < 4; ++sb) {
                const float kd = (sb < 2) ? kd0 : kd1;
                const uint16_t bits = static_cast<uint16_t>(k_blk->qs[sb * 64 + lane]) |
                                      (static_cast<uint16_t>(k_blk->qs[sb * 64 + lane + 32]) << 8);
                const __half2_raw hr = __nv_cvt_fp8x2_to_halfraw2(bits, __NV_E4M3);
                const float2 f2 = __half22float2(*reinterpret_cast<const __half2*>(&hr));
                k_val[2 * sb]     = kd * f2.x;
                k_val[2 * sb + 1] = kd * f2.y;
            }
#endif
        } else {
            const __half* k_tok = reinterpret_cast<const __half*>(k_ptr);
#pragma unroll
            for (int j = 0; j < 8; ++j) {
                k_val[j] = __half2float(k_tok[lane + 32 * j]);
            }
        }

        float alpha[kMaxGqa];
        float beta[kMaxGqa];
#pragma unroll
        for (int h = 0; h < kMaxGqa; ++h) {
            if (h < gqa_group) {
                float dot = 0.0f;
#pragma unroll
                for (int j = 0; j < 8; ++j) {
                    dot += s_q[h][lane + 32 * j] * k_val[j];
                }
                const float score = qsa_warp_reduce_sum(dot);
                const float m_new = fmaxf(m_i[h], score);
                alpha[h] = expf(m_i[h] - m_new);
                beta[h] = expf(score - m_new);
                d_i[h] = d_i[h] * alpha[h] + beta[h];
                m_i[h] = m_new;
            }
        }

        float v_val[8];
        if (kv_layout.v_format == W34_KV_FORMAT_TQ4) {
            const qsa_block_tq4_0* v_blks = reinterpret_cast<const qsa_block_tq4_0*>(v_ptr);
            const float vd0 = __half2float(v_blks[0].d);
            const float vd1 = __half2float(v_blks[1].d);
            const int odd = lane & 1;
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                const uint8_t p0 = v_blks[0].qs[(lane + 32 * j) >> 1];
                const uint8_t p1 = v_blks[1].qs[(lane + 32 * j) >> 1];
                const uint8_t n0 = odd ? (p0 >> 4) : (p0 & 0x0Fu);
                const uint8_t n1 = odd ? (p1 >> 4) : (p1 & 0x0Fu);
                v_val[j] = vd0 * kTq4D256Centroids[n0];
                v_val[j + 4] = vd1 * kTq4D256Centroids[n1];
            }
        } else if (kv_layout.v_format == W34_KV_FORMAT_Q8_0) {
            const qsa_block_q8_0_64* v_blks = reinterpret_cast<const qsa_block_q8_0_64*>(v_ptr);
#pragma unroll
            for (int sb = 0; sb < 4; ++sb) {
                const float vd = v_blks[sb].d;
                v_val[2 * sb] = vd * (float)v_blks[sb].qs[lane];
                v_val[2 * sb + 1] = vd * (float)v_blks[sb].qs[lane + 32];
            }
#if __CUDA_ARCH__ >= 890
        } else if (kv_layout.v_format == W34_KV_FORMAT_FP8_E4M3) {
            const qsa_block_fp8_e4m3_256* v_blk = reinterpret_cast<const qsa_block_fp8_e4m3_256*>(v_ptr);
            const float vd0 = v_blk->d0;
            const float vd1 = v_blk->d1;
#pragma unroll
            for (int sb = 0; sb < 4; ++sb) {
                const float vd = (sb < 2) ? vd0 : vd1;
                const uint16_t bits = static_cast<uint16_t>(v_blk->qs[sb * 64 + lane]) |
                                      (static_cast<uint16_t>(v_blk->qs[sb * 64 + lane + 32]) << 8);
                const __half2_raw hr = __nv_cvt_fp8x2_to_halfraw2(bits, __NV_E4M3);
                const float2 f2 = __half22float2(*reinterpret_cast<const __half2*>(&hr));
                v_val[2 * sb]     = vd * f2.x;
                v_val[2 * sb + 1] = vd * f2.y;
            }
#endif
        } else {
            const __half* v_tok = reinterpret_cast<const __half*>(v_ptr);
#pragma unroll
            for (int j = 0; j < 8; ++j) {
                v_val[j] = __half2float(v_tok[lane + 32 * j]);
            }
        }

#pragma unroll
        for (int h = 0; h < kMaxGqa; ++h) {
            if (h < gqa_group) {
#pragma unroll
                for (int j = 0; j < 8; ++j) {
                    acc[h][j] = acc[h][j] * alpha[h] + beta[h] * v_val[j];
                }
            }
        }
    }

    if (lane == 0) {
#pragma unroll
        for (int h = 0; h < kMaxGqa; ++h) {
            if (h < gqa_group) {
                s_m[h][wid] = m_i[h];
                s_d[h][wid] = d_i[h];
            }
        }
    }
    __syncthreads();

    for (int pass_base = 0; pass_base < gqa_group; pass_base += 4) {
        if (pass_base > 0) {
            __syncthreads();
        }
#pragma unroll
        for (int h = 0; h < kMaxGqa; ++h) {
            if (h >= pass_base && h < pass_base + 4 && h < gqa_group) {
                const int slot = h - pass_base;
#pragma unroll
                for (int j = 0; j < 8; ++j) {
                    s_work[slot][wid][lane + 32 * j] = acc[h][j];
                }
            }
        }
        __syncthreads();

        const int h_out = pass_base + wid;
        if (h_out < gqa_group) {
            const int q_head = q_head_base + h_out;
            float m_all = -FLT_MAX;
#pragma unroll
            for (int w = 0; w < 4; ++w) {
                if (s_d[h_out][w] > 0.0f) {
                    m_all = fmaxf(m_all, s_m[h_out][w]);
                }
            }
            float d_all = 0.0f;
            float w_scale[4];
#pragma unroll
            for (int w = 0; w < 4; ++w) {
                if (s_d[h_out][w] > 0.0f) {
                    w_scale[w] = expf(s_m[h_out][w] - m_all);
                    d_all += s_d[h_out][w] * w_scale[w];
                } else {
                    w_scale[w] = 0.0f;
                }
            }

            if (num_splits > 1) {
                const size_t split_idx =
                    ((size_t)tok * local_q_heads + q_head) * num_splits + split_id;
                if (lane == 0) {
                    partial_m[split_idx] = m_all;
                    partial_d[split_idx] = d_all;
                }
                float* dst_acc = partial_acc + split_idx * 256;
#pragma unroll
                for (int j = 0; j < 8; ++j) {
                    const int col = lane + 32 * j;
                    dst_acc[col] = s_work[wid][0][col] * w_scale[0]
                                 + s_work[wid][1][col] * w_scale[1]
                                 + s_work[wid][2][col] * w_scale[2]
                                 + s_work[wid][3][col] * w_scale[3];
                }
            } else {
                const float inv_d = (d_all > 0.0f) ? (1.0f / d_all) : 0.0f;
                float out_v[8];
#pragma unroll
                for (int j = 0; j < 8; ++j) {
                    const int col = lane + 32 * j;
                    const float sum_v = s_work[wid][0][col] * w_scale[0]
                                      + s_work[wid][1][col] * w_scale[1]
                                      + s_work[wid][2][col] * w_scale[2]
                                      + s_work[wid][3][col] * w_scale[3];
                    out_v[j] = sum_v * inv_d;
                }
                if (kv_layout.v_transform == W34_KV_TRANSFORM_RADEMACHER_WHT256) {
                    qsa_wht256_warp(out_v, lane);
#pragma unroll
                    for (int j = 0; j < 8; ++j) {
                        out_v[j] *= kTq4Signs256[lane + 32 * j];
                    }
                }
                const __half* gate_head = (gate != nullptr)
                    ? (gate + ((size_t)tok * local_q_heads + q_head) * 256)
                    : nullptr;
                __half* dst_head = out + ((size_t)tok * local_q_heads + q_head) * 256;
                __half* s_head = APPLY_HAD64
                    ? reinterpret_cast<__half*>(s_work[wid][0])
                    : dst_head;
                if constexpr (APPLY_HAD64) {
                    // Ensure all lanes finish reading float `s_work[wid][0][0..255]` (0..1023 B)
                    // before any lane overwrites `__half s_head[0..255]` (0..511 B).
                    __syncwarp();
                }
#pragma unroll
                for (int j = 0; j < 8; ++j) {
                    const int col = lane + 32 * j;
                    float val = out_v[j];
                    if (gate_head != nullptr) {
                        const float g = __half2float(gate_head[col]);
                        val *= (1.0f / (1.0f + expf(-g)));
                    }
                    s_head[col] = __float2half(val);
                }
                qsa_store_head256_warp<APPLY_HAD64>(dst_head, s_head, lane);
            }
        }
    }
}

// 1-warp Split-K reduction kernel for `qsa_sparse_paged_attn_gqa_f16_kernel` (`num_splits` in 2..=32).
// Grid: (num_tokens, local_q_heads), Block: 32.
template <bool APPLY_HAD64 = false>
__global__ void qsa_sparse_paged_attn_merge_f16_kernel(
    const float* __restrict__ partial_m,
    const float* __restrict__ partial_d,
    const float* __restrict__ partial_acc,
    const __half* __restrict__ gate,
    const uint8_t* __restrict__ valid_token_mask,
    __half* __restrict__ out,
    int num_tokens,
    int local_q_heads,
    int num_splits,
    uint32_t v_transform) {
    const int tok = blockIdx.x;
    const int q_head = blockIdx.y;
    if (tok >= num_tokens || q_head >= local_q_heads) return;

    const int lane = threadIdx.x;
    __half* dst_head = out + ((size_t)tok * local_q_heads + q_head) * 256;
    if (valid_token_mask != nullptr && valid_token_mask[tok] == 0) {
#pragma unroll
        for (int j = 0; j < 8; ++j) {
            dst_head[lane + 32 * j] = __float2half(0.0f);
        }
        return;
    }

    const size_t base_idx = ((size_t)tok * local_q_heads + q_head) * num_splits;
    const float m_s = (lane < num_splits) ? partial_m[base_idx + lane] : -FLT_MAX;
    const float d_s = (lane < num_splits) ? partial_d[base_idx + lane] : 0.0f;

    float m_all = (d_s > 0.0f) ? m_s : -FLT_MAX;
#pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
        m_all = fmaxf(m_all, __shfl_xor_sync(0xffffffffu, m_all, off));
    }
    const float w_my = (d_s > 0.0f) ? expf(m_s - m_all) : 0.0f;
    const float d_all = qsa_warp_reduce_sum(d_s * w_my);
    const float inv_d = (d_all > 0.0f) ? (1.0f / d_all) : 0.0f;

    float out_v[8] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    for (int s = 0; s < num_splits; ++s) {
        const float ws = __shfl_sync(0xffffffffu, w_my, s);
        if (ws > 0.0f) {
            const float* acc_s = partial_acc + (base_idx + s) * 256;
#pragma unroll
            for (int j = 0; j < 8; ++j) {
                out_v[j] += acc_s[lane + 32 * j] * ws;
            }
        }
    }
#pragma unroll
    for (int j = 0; j < 8; ++j) {
        out_v[j] *= inv_d;
    }

    if (v_transform == W34_KV_TRANSFORM_RADEMACHER_WHT256) {
        qsa_wht256_warp(out_v, lane);
#pragma unroll
        for (int j = 0; j < 8; ++j) {
            out_v[j] *= kTq4Signs256[lane + 32 * j];
        }
    }

    const __half* gate_head = (gate != nullptr)
        ? (gate + ((size_t)tok * local_q_heads + q_head) * 256)
        : nullptr;
    __shared__ __align__(16) __half s_head_buf[256];
    __half* s_head = APPLY_HAD64 ? s_head_buf : dst_head;
#pragma unroll
    for (int j = 0; j < 8; ++j) {
        const int col = lane + 32 * j;
        float val = out_v[j];
        if (gate_head != nullptr) {
            const float g = __half2float(gate_head[col]);
            val *= (1.0f / (1.0f + expf(-g)));
        }
        s_head[col] = __float2half(val);
    }
    qsa_store_head256_warp<APPLY_HAD64>(dst_head, s_head, lane);
}

// Generic Sparse Paged GQA fallback (used when `local_kv_heads == 0`, `local_q_heads` is not
// divisible by `local_kv_heads`, or `head_dim != 256`; divisible GQA with `head_dim == 256`
// uses `qsa_sparse_paged_attn_gqa_f16_kernel` with `q_group_size` in `1..=6`).
// Grid: (num_tokens, local_q_heads), Block: 128 (4 warps, 8 dims per lane of head_dim=256).
template <bool APPLY_HAD64 = false>
__global__ void qsa_sparse_paged_attn_f16_kernel(
    const __half* __restrict__ q,
    const __half* __restrict__ gate,
    w34_kv_layer_layout kv_layout,
    const int32_t* __restrict__ selected_phys_blocks,
    const int32_t* __restrict__ selected_logical_blocks,
    const int32_t* __restrict__ num_selected_blocks,
    const int32_t* __restrict__ token_seq_pos,
    const uint8_t* __restrict__ valid_token_mask,
    __half* __restrict__ out,
    int num_tokens,
    int local_q_heads,
    int select_k_blocks,
    float sm_scale) {
    const int tok = blockIdx.x;
    const int q_head = blockIdx.y;
    const int head_dim = kv_layout.head_dim;
    if (tok >= num_tokens || q_head >= local_q_heads) return;

    __half* dst_head = out + ((size_t)tok * local_q_heads + q_head) * head_dim;
    if (valid_token_mask != nullptr && valid_token_mask[tok] == 0) {
        for (int d = threadIdx.x; d < head_dim; d += blockDim.x) {
            dst_head[d] = __float2half(0.0f);
        }
        return;
    }

    const int n_sel = num_selected_blocks[tok];
    const int seq_pos = token_seq_pos[tok];
    const int block_size = kv_layout.block_size;
    const int groups_per_page = block_size >> 2;
    const int gqa_group = local_q_heads / kv_layout.local_kv_heads;
    const int kv_head = q_head / gqa_group;

    const __half* q_head_ptr = q + ((size_t)tok * local_q_heads + q_head) * head_dim;
    const int wid = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const int num_warps = blockDim.x >> 5;

    float q_reg[8];
    float acc[8] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
#pragma unroll
    for (int j = 0; j < 8; ++j) {
        const int d = lane + j * 32;
        q_reg[j] = (d < head_dim) ? (__half2float(q_head_ptr[d]) * sm_scale) : 0.0f;
    }
    if (head_dim == 256 && kv_layout.k_transform == W34_KV_TRANSFORM_RADEMACHER_WHT256) {
#pragma unroll
        for (int j = 0; j < 8; ++j) {
            q_reg[j] *= kTq4Signs256[lane + 32 * j];
        }
        qsa_wht256_warp(q_reg, lane);
    }

    float m_i = -FLT_MAX;
    float d_i = 0.0f;

    const int32_t* tok_phys = selected_phys_blocks + (size_t)tok * select_k_blocks;
    const int32_t* tok_log = selected_logical_blocks + (size_t)tok * select_k_blocks;

    const uint8_t* k_base = reinterpret_cast<const uint8_t*>((uintptr_t)kv_layout.k_base);
    const uint8_t* v_base = reinterpret_cast<const uint8_t*>((uintptr_t)kv_layout.v_base);

    const int total_sel_tokens = n_sel * 4;
    for (int t_idx = wid; t_idx < total_sel_tokens; t_idx += num_warps) {
        const int grp_idx = t_idx >> 2;
        const int grp_off = t_idx & 3;
        const int32_t phys_grp = tok_phys[grp_idx];
        const int32_t log_grp = tok_log[grp_idx];
        const int kv_pos = log_grp * 4 + grp_off;
        if (phys_grp < 0 || kv_pos > seq_pos) continue;
        const int32_t phys_page = phys_grp / groups_per_page;
        const int page_off = (phys_grp % groups_per_page) * 4 + grp_off;

        const uint8_t* k_ptr =
            k_base + (size_t)phys_page * kv_layout.k_page_stride_bytes +
            (size_t)kv_head * kv_layout.k_head_stride_bytes +
            (size_t)page_off * kv_layout.k_token_stride_bytes;
        const uint8_t* v_ptr =
            v_base + (size_t)phys_page * kv_layout.v_page_stride_bytes +
            (size_t)kv_head * kv_layout.v_head_stride_bytes +
            (size_t)page_off * kv_layout.v_token_stride_bytes;

        float dot = 0.0f;
        if (head_dim == 256 && kv_layout.k_format == W34_KV_FORMAT_TQ4) {
            const qsa_block_tq4_0* k_blks = reinterpret_cast<const qsa_block_tq4_0*>(k_ptr);
            const float d0 = __half2float(k_blks[0].d);
            const float d1 = __half2float(k_blks[1].d);
            const int odd = lane & 1;
            float dot0 = 0.0f;
            float dot1 = 0.0f;
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                const uint8_t p0 = k_blks[0].qs[(lane + 32 * j) >> 1];
                const uint8_t p1 = k_blks[1].qs[(lane + 32 * j) >> 1];
                const uint8_t n0 = odd ? (p0 >> 4) : (p0 & 0x0Fu);
                const uint8_t n1 = odd ? (p1 >> 4) : (p1 & 0x0Fu);
                dot0 += q_reg[j] * kTq4D256Centroids[n0];
                dot1 += q_reg[j + 4] * kTq4D256Centroids[n1];
            }
            dot = d0 * dot0 + d1 * dot1;
        } else if (head_dim == 256 && kv_layout.k_format == W34_KV_FORMAT_Q8_0) {
            const qsa_block_q8_0_64* k_blks = reinterpret_cast<const qsa_block_q8_0_64*>(k_ptr);
#pragma unroll
            for (int sb = 0; sb < 4; ++sb) {
                const float d = k_blks[sb].d;
                const float k0 = (float)k_blks[sb].qs[lane];
                const float k1 = (float)k_blks[sb].qs[lane + 32];
                dot += d * (q_reg[2 * sb] * k0 + q_reg[2 * sb + 1] * k1);
            }
#if __CUDA_ARCH__ >= 890
        } else if (head_dim == 256 && kv_layout.k_format == W34_KV_FORMAT_FP8_E4M3) {
            const qsa_block_fp8_e4m3_256* k_blk = reinterpret_cast<const qsa_block_fp8_e4m3_256*>(k_ptr);
            const float d0 = k_blk->d0;
            const float d1 = k_blk->d1;
#pragma unroll
            for (int sb = 0; sb < 4; ++sb) {
                const float d = (sb < 2) ? d0 : d1;
                const uint16_t bits = static_cast<uint16_t>(k_blk->qs[sb * 64 + lane]) |
                                      (static_cast<uint16_t>(k_blk->qs[sb * 64 + lane + 32]) << 8);
                const __half2_raw hr = __nv_cvt_fp8x2_to_halfraw2(bits, __NV_E4M3);
                const float2 f2 = __half22float2(*reinterpret_cast<const __half2*>(&hr));
                dot += d * (q_reg[2 * sb] * f2.x + q_reg[2 * sb + 1] * f2.y);
            }
#endif
        } else {
            const __half* k_tok = reinterpret_cast<const __half*>(k_ptr);
#pragma unroll
            for (int j = 0; j < 8; ++j) {
                const int d = lane + j * 32;
                if (d < head_dim) {
                    dot += q_reg[j] * __half2float(k_tok[d]);
                }
            }
        }
        const float score = qsa_warp_reduce_sum(dot);
        const float m_new = fmaxf(m_i, score);
        const float alpha = expf(m_i - m_new);
        const float beta = expf(score - m_new);
        d_i = d_i * alpha + beta;
        m_i = m_new;

        if (head_dim == 256 && kv_layout.v_format == W34_KV_FORMAT_TQ4) {
            const qsa_block_tq4_0* v_blks = reinterpret_cast<const qsa_block_tq4_0*>(v_ptr);
            const float pd0 = beta * __half2float(v_blks[0].d);
            const float pd1 = beta * __half2float(v_blks[1].d);
            const int odd = lane & 1;
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                const uint8_t p0 = v_blks[0].qs[(lane + 32 * j) >> 1];
                const uint8_t p1 = v_blks[1].qs[(lane + 32 * j) >> 1];
                const uint8_t n0 = odd ? (p0 >> 4) : (p0 & 0x0Fu);
                const uint8_t n1 = odd ? (p1 >> 4) : (p1 & 0x0Fu);
                acc[j] = acc[j] * alpha + pd0 * kTq4D256Centroids[n0];
                acc[j + 4] = acc[j + 4] * alpha + pd1 * kTq4D256Centroids[n1];
            }
        } else if (head_dim == 256 && kv_layout.v_format == W34_KV_FORMAT_Q8_0) {
            const qsa_block_q8_0_64* v_blks = reinterpret_cast<const qsa_block_q8_0_64*>(v_ptr);
#pragma unroll
            for (int sb = 0; sb < 4; ++sb) {
                const float pd = beta * v_blks[sb].d;
                acc[2 * sb] = acc[2 * sb] * alpha + pd * (float)v_blks[sb].qs[lane];
                acc[2 * sb + 1] = acc[2 * sb + 1] * alpha + pd * (float)v_blks[sb].qs[lane + 32];
            }
#if __CUDA_ARCH__ >= 890
        } else if (head_dim == 256 && kv_layout.v_format == W34_KV_FORMAT_FP8_E4M3) {
            const qsa_block_fp8_e4m3_256* v_blk = reinterpret_cast<const qsa_block_fp8_e4m3_256*>(v_ptr);
            const float d0 = v_blk->d0;
            const float d1 = v_blk->d1;
#pragma unroll
            for (int sb = 0; sb < 4; ++sb) {
                const float pd = beta * ((sb < 2) ? d0 : d1);
                const uint16_t bits = static_cast<uint16_t>(v_blk->qs[sb * 64 + lane]) |
                                      (static_cast<uint16_t>(v_blk->qs[sb * 64 + lane + 32]) << 8);
                const __half2_raw hr = __nv_cvt_fp8x2_to_halfraw2(bits, __NV_E4M3);
                const float2 f2 = __half22float2(*reinterpret_cast<const __half2*>(&hr));
                acc[2 * sb]     = acc[2 * sb] * alpha + pd * f2.x;
                acc[2 * sb + 1] = acc[2 * sb + 1] * alpha + pd * f2.y;
            }
#endif
        } else {
            const __half* v_tok = reinterpret_cast<const __half*>(v_ptr);
#pragma unroll
            for (int j = 0; j < 8; ++j) {
                const int d = lane + j * 32;
                if (d < head_dim) {
                    acc[j] = acc[j] * alpha + beta * __half2float(v_tok[d]);
                }
            }
        }
    }

    __shared__ float s_m[4];
    __shared__ float s_d[4];
    __shared__ float s_acc[4][256];

    if (lane == 0) {
        s_m[wid] = m_i;
        s_d[wid] = d_i;
    }
#pragma unroll
    for (int j = 0; j < 8; ++j) {
        const int d = lane + j * 32;
        if (d < head_dim) {
            s_acc[wid][d] = acc[j];
        }
    }
    __syncthreads();

    float m_all = -FLT_MAX;
#pragma unroll
    for (int w = 0; w < 4; ++w) {
        if (s_d[w] > 0.0f) {
            m_all = fmaxf(m_all, s_m[w]);
        }
    }
    float d_all = 0.0f;
    float w_scale[4];
#pragma unroll
    for (int w = 0; w < 4; ++w) {
        if (s_d[w] > 0.0f) {
            w_scale[w] = expf(s_m[w] - m_all);
            d_all += s_d[w] * w_scale[w];
        } else {
            w_scale[w] = 0.0f;
        }
    }
    const float inv_d = (d_all > 0.0f) ? (1.0f / d_all) : 0.0f;

    const __half* gate_head = (gate != nullptr)
        ? (gate + ((size_t)tok * local_q_heads + q_head) * head_dim)
        : nullptr;

    if (head_dim == 256 && kv_layout.v_transform == W34_KV_TRANSFORM_RADEMACHER_WHT256) {
        if (wid == 0) {
            float out_v[8];
#pragma unroll
            for (int j = 0; j < 8; ++j) {
                const int col = lane + 32 * j;
                float sum_v = s_acc[0][col] * w_scale[0]
                            + s_acc[1][col] * w_scale[1]
                            + s_acc[2][col] * w_scale[2]
                            + s_acc[3][col] * w_scale[3];
                out_v[j] = sum_v * inv_d;
            }
            qsa_wht256_warp(out_v, lane);
            __half* s_head = APPLY_HAD64 ? reinterpret_cast<__half*>(s_acc[0]) : dst_head;
#pragma unroll
            for (int j = 0; j < 8; ++j) {
                const int col = lane + 32 * j;
                float val = out_v[j] * kTq4Signs256[col];
                if (gate_head != nullptr) {
                    const float g = __half2float(gate_head[col]);
                    val *= (1.0f / (1.0f + expf(-g)));
                }
                s_head[col] = __float2half(val);
            }
            qsa_store_head256_warp<APPLY_HAD64>(dst_head, s_head, lane);
        }
        return;
    }

    __shared__ __align__(16) __half s_head_fallback[256];
    __half* s_head = APPLY_HAD64 ? s_head_fallback : dst_head;
    for (int d = threadIdx.x; d < head_dim; d += blockDim.x) {
        float sum_v = 0.0f;
#pragma unroll
        for (int w = 0; w < 4; ++w) {
            sum_v += s_acc[w][d] * w_scale[w];
        }
        float val = sum_v * inv_d;
        if (gate_head != nullptr) {
            const float g = __half2float(gate_head[d]);
            val *= (1.0f / (1.0f + expf(-g)));
        }
        s_head[d] = __float2half(val);
    }
    if constexpr (APPLY_HAD64) {
        __syncthreads();
        if (wid == 0) {
            qsa_store_head256_warp<true>(dst_head, s_head, lane);
        }
    }
}

template <bool APPLY_HAD64 = false>
static int launch_qsa_sparse_paged_attn_f16(
    const w34_exec* ex,
    const void* q,
    const void* gate,
    const w34_kv_layer_layout* kv_layout,
    const int32_t* selected_phys_blocks,
    const int32_t* selected_logical_blocks,
    const int32_t* num_selected_blocks,
    const int32_t* token_seq_pos,
    const uint8_t* valid_token_mask,
    void* out,
    int num_tokens,
    int local_q_heads,
    int select_k_blocks,
    float sm_scale) {
    if (!ex || !q || !kv_layout || !selected_phys_blocks || !selected_logical_blocks ||
        !num_selected_blocks || !token_seq_pos || !out || num_tokens < 0 || local_q_heads <= 0 ||
        kv_layout->local_kv_heads <= 0 || (local_q_heads % kv_layout->local_kv_heads) != 0 ||
        kv_layout->head_dim != 256 || select_k_blocks < 0) {
        return (int)cudaErrorInvalidValue;
    }
    if ((kv_layout->k_format == W34_KV_FORMAT_FP8_E4M3 || kv_layout->v_format == W34_KV_FORMAT_FP8_E4M3) &&
        !qsa_current_device_supports_fp8_kv()) {
        return (int)cudaErrorNotSupported;
    }
    if (num_tokens == 0) return 0;
    cudaStream_t stream = reinterpret_cast<cudaStream_t>(ex->stream);
    const int local_kv_heads = kv_layout->local_kv_heads;
    int q_group_size = 0;
    if (local_kv_heads > 0 && (local_q_heads % local_kv_heads) == 0) {
        const int full_gqa = local_q_heads / local_kv_heads;
        for (int cand = 6; cand >= 1; --cand) {
            if ((full_gqa % cand) == 0) {
                q_group_size = cand;
                break;
            }
        }
    }
    if (q_group_size >= 1 && q_group_size <= 6 && select_k_blocks > 0) {
        const int num_q_groups = local_q_heads / q_group_size;
        // Choose `target_splits` from `num_tokens` (independent of `local_q_heads` and constant
        // across `M = 1..16`) so Plain (`M = 1`), `MTP-3` (`M = 4`), and `DFlash2` (`M = 8..16`)
        // as well as `TP=1` and `TP=2` execute the exact same Split-K reduction tree.
        constexpr int kSplit8MaxTokens = 16;
        constexpr int kSplit4MaxTokens = 64;
        constexpr int kSplit2MaxTokens = 128;
        int target_splits = 1;
        if (num_tokens <= kSplit8MaxTokens) {
            target_splits = 8;
        } else if (num_tokens <= kSplit4MaxTokens) {
            target_splits = 4;
        } else if (num_tokens <= kSplit2MaxTokens) {
            target_splits = 2;
        }
        const int max_by_blocks = select_k_blocks >> 1;
        if (target_splits > max_by_blocks) {
            target_splits = (max_by_blocks >= 1) ? max_by_blocks : 1;
        }
        const size_t total_splits = (size_t)num_tokens * local_q_heads * target_splits;
        const size_t needed_bytes = total_splits * 258 * sizeof(float);
        const int num_splits =
            (target_splits > 1 && ex->scratch_ptr != 0 && ex->scratch_size >= needed_bytes)
                ? target_splits
                : 1;
        float* partial_m =
            (num_splits > 1) ? reinterpret_cast<float*>((uintptr_t)ex->scratch_ptr) : nullptr;
        float* partial_d = (num_splits > 1) ? (partial_m + total_splits) : nullptr;
        float* partial_acc = (num_splits > 1) ? (partial_d + total_splits) : nullptr;

        dim3 grid(num_tokens, num_q_groups, num_splits);
        qsa_sparse_paged_attn_gqa_f16_kernel<APPLY_HAD64><<<grid, 128, 0, stream>>>(
            static_cast<const __half*>(q),
            static_cast<const __half*>(gate),
            *kv_layout,
            selected_phys_blocks,
            selected_logical_blocks,
            num_selected_blocks,
            token_seq_pos,
            valid_token_mask,
            static_cast<__half*>(out),
            partial_m,
            partial_d,
            partial_acc,
            num_tokens,
            local_q_heads,
            q_group_size,
            select_k_blocks,
            num_splits,
            sm_scale);
        if (num_splits > 1) {
            dim3 merge_grid(num_tokens, local_q_heads);
            qsa_sparse_paged_attn_merge_f16_kernel<APPLY_HAD64><<<merge_grid, 32, 0, stream>>>(
                partial_m,
                partial_d,
                partial_acc,
                static_cast<const __half*>(gate),
                valid_token_mask,
                static_cast<__half*>(out),
                num_tokens,
                local_q_heads,
                num_splits,
                kv_layout->v_transform);
        }
        return (cudaGetLastError() == cudaSuccess) ? 0 : -2;
    }

    dim3 grid(num_tokens, local_q_heads);
    qsa_sparse_paged_attn_f16_kernel<APPLY_HAD64><<<grid, 128, 0, stream>>>(
        static_cast<const __half*>(q),
        static_cast<const __half*>(gate),
        *kv_layout,
        selected_phys_blocks,
        selected_logical_blocks,
        num_selected_blocks,
        token_seq_pos,
        valid_token_mask,
        static_cast<__half*>(out),
        num_tokens,
        local_q_heads,
        select_k_blocks,
        sm_scale);
    return (cudaGetLastError() == cudaSuccess) ? 0 : -2;
}

} // namespace w34

extern "C" {

int w34_qsa_indexer_norm_rope_f16(
    const w34_exec* ex,
    const void* qk_in,
    const void* q_norm_weight,
    const void* k_norm_weight,
    const int32_t* positions,
    void* q_out,
    void* k_out,
    const uint8_t* valid_token_mask,
    int num_tokens,
    int q_heads,
    int kv_heads,
    int head_dim,
    int rotary_dim,
    float rope_theta,
    float eps,
    int mrope_s1,
    int mrope_s2,
    int pos_plane_stride) {
    if (!ex || !qk_in || !q_norm_weight || !k_norm_weight || !positions || !q_out || !k_out ||
        num_tokens < 0 || q_heads <= 0 || kv_heads <= 0 || head_dim != 128) {
        return -1;
    }
    if (num_tokens == 0) return 0;
    cudaStream_t stream = reinterpret_cast<cudaStream_t>(ex->stream);
    dim3 grid(num_tokens, q_heads + kv_heads);
    w34::qsa_indexer_norm_rope_f16_kernel<<<grid, 128, 0, stream>>>(
        static_cast<const __half*>(qk_in),
        static_cast<const __half*>(q_norm_weight),
        static_cast<const __half*>(k_norm_weight),
        positions,
        static_cast<__half*>(q_out),
        static_cast<__half*>(k_out),
        valid_token_mask,
        num_tokens,
        q_heads,
        kv_heads,
        head_dim,
        rotary_dim,
        rope_theta,
        eps,
        mrope_s1,
        mrope_s2,
        pos_plane_stride);
    return (cudaGetLastError() == cudaSuccess) ? 0 : -2;
}

int w34_qsa_indexer_cache_update_f16(
    const w34_exec* ex,
    const void* k_in,
    void* raw_k_cache,
    void* comp_k_cache,
    const void* k_norm_weight,
    const int32_t* positions,
    const int32_t* write_slot_mapping,
    const uint8_t* valid_token_mask,
    int num_tokens,
    int head_dim,
    int rotary_dim,
    float rope_theta,
    float eps,
    int mrope_s1,
    int mrope_s2,
    int pos_plane_stride) {
    if (!ex || !k_in || !raw_k_cache || !comp_k_cache || !write_slot_mapping || num_tokens < 0 ||
        head_dim != 128) {
        return -1;
    }
    if (num_tokens == 0) return 0;
    cudaStream_t stream = reinterpret_cast<cudaStream_t>(ex->stream);
    w34::qsa_indexer_write_raw_k_f16_kernel<<<num_tokens, 128, 0, stream>>>(
        static_cast<const __half*>(k_in),
        static_cast<__half*>(raw_k_cache),
        positions,
        write_slot_mapping,
        valid_token_mask,
        num_tokens,
        head_dim,
        pos_plane_stride);
    w34::qsa_indexer_update_comp_k_f16_kernel<<<num_tokens, 128, 0, stream>>>(
        static_cast<const __half*>(raw_k_cache),
        static_cast<__half*>(comp_k_cache),
        static_cast<const __half*>(k_norm_weight),
        positions,
        write_slot_mapping,
        valid_token_mask,
        num_tokens,
        head_dim,
        rotary_dim,
        rope_theta,
        eps,
        mrope_s1,
        mrope_s2,
        pos_plane_stride);
    return (cudaGetLastError() == cudaSuccess) ? 0 : -2;
}

int w34_qsa_select_blocks_f16(
    const w34_exec* ex,
    const void* q_indexer,
    const void* comp_k_cache,
    const w34_attention_batch* batch,
    int32_t* out_selected_phys_blocks,
    int32_t* out_selected_logical_blocks,
    int32_t* out_num_selected_blocks,
    int32_t* out_token_seq_pos,
    int q_heads,
    int head_dim,
    int block_size,
    int select_k_blocks,
    int sink_blocks,
    int recent_blocks) {
    if (!ex || !q_indexer || !comp_k_cache || !batch || !out_selected_phys_blocks ||
        !out_selected_logical_blocks || !out_num_selected_blocks || !out_token_seq_pos ||
        q_heads != 4 || head_dim != 128 || block_size != 16 || select_k_blocks <= 0) {
        return -1;
    }
    if (batch->total_tokens <= 0) return 0;
    cudaStream_t stream = reinterpret_cast<cudaStream_t>(ex->stream);
    w34::qsa_select_blocks_f16_kernel<<<batch->total_tokens, 256, 0, stream>>>(
        static_cast<const __half*>(q_indexer),
        static_cast<const __half*>(comp_k_cache),
        *batch,
        out_selected_phys_blocks,
        out_selected_logical_blocks,
        out_num_selected_blocks,
        out_token_seq_pos,
        q_heads,
        head_dim,
        block_size,
        select_k_blocks,
        sink_blocks,
        recent_blocks);
    return (cudaGetLastError() == cudaSuccess) ? 0 : -2;
}

int w34_qsa_sparse_paged_attn_f16(
    const w34_exec* ex,
    const void* q,
    const void* gate,
    const w34_kv_layer_layout* kv_layout,
    const int32_t* selected_phys_blocks,
    const int32_t* selected_logical_blocks,
    const int32_t* num_selected_blocks,
    const int32_t* token_seq_pos,
    const uint8_t* valid_token_mask,
    void* out,
    int num_tokens,
    int local_q_heads,
    int select_k_blocks,
    float sm_scale) {
    return w34::launch_qsa_sparse_paged_attn_f16<false>(
        ex, q, gate, kv_layout, selected_phys_blocks, selected_logical_blocks,
        num_selected_blocks, token_seq_pos, valid_token_mask, out, num_tokens,
        local_q_heads, select_k_blocks, sm_scale);
}

int w34_qsa_sparse_paged_attn_hadamard_f16(
    const w34_exec* ex,
    const void* q,
    const void* gate,
    const w34_kv_layer_layout* kv_layout,
    const int32_t* selected_phys_blocks,
    const int32_t* selected_logical_blocks,
    const int32_t* num_selected_blocks,
    const int32_t* token_seq_pos,
    const uint8_t* valid_token_mask,
    void* out,
    int num_tokens,
    int local_q_heads,
    int select_k_blocks,
    float sm_scale) {
    return w34::launch_qsa_sparse_paged_attn_f16<true>(
        ex, q, gate, kv_layout, selected_phys_blocks, selected_logical_blocks,
        num_selected_blocks, token_seq_pos, valid_token_mask, out, num_tokens,
        local_q_heads, select_k_blocks, sm_scale);
}

} // extern "C"
