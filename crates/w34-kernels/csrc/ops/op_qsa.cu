// Qwen3.8-Flash-Next (qwen4_exp) Query-Sparse Attention (QSA) & Indexer CUDA Kernels.
// Reference: nvidia/qsa_indexer.py, nvidia/qsa.py, nvidia/ops/qsa.py, nvidia/ops/qsa_indexer.py
//   - Pre-indexer GemmaRMSNorm + partial interleaved NeoX mRoPE (head_dim=128, rotary_dim=64, [11,11,10])
//   - Paged raw key cache [num_pages, 16, 128] + 4:1 compressed key cache [num_pages, 4, 128]
//   - ReLU-sum MQA scoring + top-k block selector (k=64 blocks = 1024 tokens)
//   - Sparse paged GQA over selected blocks (FP16, Q8_0 272B/row, and TQ4 132B/row with Rademacher+WHT256)
//     with causal boundary masking & optional sigmoid output gate
#include "../w34_cuda.h"
#include "../tq/tq4_codebook.cuh"

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <float.h>
#include <math.h>

extern "C" __device__ float __nv_sinf(float);
extern "C" __device__ float __nv_cosf(float);
extern "C" __device__ float __nv_powf(float, float);

namespace w34 {

#pragma pack(push, 1)
struct qsa_block_tq4_0 {
    __half  d;
    uint8_t qs[64];
};
static_assert(sizeof(qsa_block_tq4_0) == 66, "qsa_block_tq4_0 must be 66 bytes");

struct qsa_block_q8_0_64 {
    float  d;
    int8_t qs[64];
};
static_assert(sizeof(qsa_block_q8_0_64) == 68, "qsa_block_q8_0_64 must be 68 bytes");
#pragma pack(pop)

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

    // Single writer per 4-token compressed group: only the CTA writing the 4th slot (`slot & 3 == 3`)
    // commits the compressed group into `comp_k_cache`.
    if ((slot & 3) != 3) {
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

// Selects up to `select_k_blocks` 4-token compressed groups per query token using the QSA Indexer:
//   - Each entry in `out_selected_logical_blocks` is a logical 4-token group index `g` (covering tokens `4*g .. 4*g+4`).
//   - Each entry in `out_selected_phys_blocks` is the resolved physical 4-token group index
//     `phys_page * (block_size / 4) + (g % (block_size / 4))`.
//   - If all valid 4-token groups fit within `select_k_blocks` (and completed groups <= 512): selects all `0..num_valid_groups`.
//   - Otherwise: scores each completed 4-token group `g in [0..num_comp_groups)` directly at 4-token group granularity,
//     selects the top completed groups (plus optional forced sink/recent completed groups), and always appends the
//     incomplete causal tail group when `(token_pos + 1) % 4 != 0`.
// Grid: num_tokens, Block: 128.
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
    int req = 0;
    for (int r = 0; r < batch.num_reqs; ++r) {
        if (tok >= batch.query_start_loc[r] && tok < batch.query_start_loc[r + 1]) {
            req = r;
            break;
        }
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

    // Load q_indexer[tok, 0..4, 0..128] into shared memory.
    __shared__ float s_q[4][128];
    const __half* q_tok = q_indexer + (size_t)tok * q_heads * head_dim;
    for (int idx = threadIdx.x; idx < 4 * head_dim; idx += blockDim.x) {
        const int h = idx / head_dim;
        const int d = idx % head_dim;
        s_q[h][d] = (h < q_heads) ? __half2float(q_tok[idx]) : 0.0f;
    }
    __syncthreads();

    const float scale = rsqrtf((float)head_dim);
    const int wid = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const int num_warps = blockDim.x >> 5;

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
    constexpr int kMaxSelectGroups = 513;
    if (need_mid > kMaxSelectGroups) need_mid = kMaxSelectGroups;

    // Maintain warp-collaborative top-need_mid candidate middle groups in shared memory.
    __shared__ float s_top_scores[kMaxSelectGroups];
    __shared__ int32_t s_top_blocks[kMaxSelectGroups];
    for (int i = threadIdx.x; i < kMaxSelectGroups; i += blockDim.x) {
        s_top_scores[i] = -FLT_MAX;
        s_top_blocks[i] = -1;
    }
    __syncthreads();

    __shared__ float s_chunk_scores[4];
    __shared__ int32_t s_chunk_blocks[4];
    __shared__ float s_w_sc[4];
    __shared__ int32_t s_w_blk[4];
    __shared__ int s_w_pos[4];

    for (int g_base = mid_start; g_base < mid_end; g_base += num_warps) {
        const int g = g_base + wid;
        float grp_score = -FLT_MAX;
        if (g < mid_end && g < num_comp_groups) {
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
        }
        if (lane == 0) {
            s_chunk_scores[wid] = grp_score;
            s_chunk_blocks[wid] = (g < mid_end) ? g : -1;
        }
        __syncthreads();

        if (need_mid > 0) {
            // Compute the minimum-score slot in s_top_scores[0..need_mid) once using all 128 threads,
            // breaking ties by larger logical block index (so earliest logical group wins ties).
            float my_min_sc = FLT_MAX;
            int32_t my_min_blk = 0;
            int my_min_pos = need_mid;
            for (int i = threadIdx.x; i < need_mid; i += blockDim.x) {
                const float sc = s_top_scores[i];
                const int32_t blk = s_top_blocks[i];
                const bool take_i = (my_min_pos >= need_mid) ||
                    (blk < 0 && (my_min_blk >= 0 || i < my_min_pos)) ||
                    (blk >= 0 && my_min_blk >= 0 &&
                     (sc < my_min_sc || (sc == my_min_sc && blk > my_min_blk)));
                if (take_i) {
                    my_min_sc = sc;
                    my_min_blk = blk;
                    my_min_pos = i;
                }
            }
            // Warp-reduce minimum victim slot.
#pragma unroll
            for (int off = 16; off > 0; off >>= 1) {
                const float o_sc = __shfl_down_sync(0xffffffffu, my_min_sc, off);
                const int32_t o_blk = __shfl_down_sync(0xffffffffu, my_min_blk, off);
                const int o_pos = __shfl_down_sync(0xffffffffu, my_min_pos, off);
                if (o_pos < need_mid) {
                    const bool take_o = (my_min_pos >= need_mid) ||
                        (o_blk < 0 && (my_min_blk >= 0 || o_pos < my_min_pos)) ||
                        (o_blk >= 0 && my_min_blk >= 0 &&
                         (o_sc < my_min_sc || (o_sc == my_min_sc && o_blk > my_min_blk)));
                    if (take_o) {
                        my_min_sc = o_sc;
                        my_min_blk = o_blk;
                        my_min_pos = o_pos;
                    }
                }
            }
            if (lane == 0) {
                s_w_sc[wid] = my_min_sc;
                s_w_blk[wid] = my_min_blk;
                s_w_pos[wid] = my_min_pos;
            }
            __syncthreads();

            if (threadIdx.x == 0) {
                for (int w = 0; w < num_warps; ++w) {
                    const float sc = s_chunk_scores[w];
                    const int32_t lg = s_chunk_blocks[w];
                    if (lg < 0) continue;
                    int v_pos = s_w_pos[0];
                    float v_sc = s_w_sc[0];
                    int32_t v_blk = s_w_blk[0];
                    for (int rw = 1; rw < 4; ++rw) {
                        const int op = s_w_pos[rw];
                        if (op >= need_mid) continue;
                        const float osc = s_w_sc[rw];
                        const int32_t oblk = s_w_blk[rw];
                        const bool take_o = (v_pos >= need_mid) ||
                            (oblk < 0 && (v_blk >= 0 || op < v_pos)) ||
                            (oblk >= 0 && v_blk >= 0 &&
                             (osc < v_sc || (osc == v_sc && oblk > v_blk)));
                        if (take_o) {
                            v_pos = op;
                            v_sc = osc;
                            v_blk = oblk;
                        }
                    }
                    if (v_pos < need_mid && (v_blk < 0 || sc > v_sc)) {
                        s_top_scores[v_pos] = sc;
                        s_top_blocks[v_pos] = lg;
                        if (w + 1 < num_warps) {
                            int best_p = 0;
                            float best_s = s_top_scores[0];
                            int32_t best_b = s_top_blocks[0];
                            for (int i = 1; i < need_mid; ++i) {
                                const float isc = s_top_scores[i];
                                const int32_t iblk = s_top_blocks[i];
                                const bool take_i =
                                    (iblk < 0 && (best_b >= 0 || i < best_p)) ||
                                    (iblk >= 0 && best_b >= 0 &&
                                     (isc < best_s || (isc == best_s && iblk > best_b)));
                                if (take_i) {
                                    best_p = i;
                                    best_s = isc;
                                    best_b = iblk;
                                }
                            }
                            s_w_pos[0] = best_p;
                            s_w_sc[0] = best_s;
                            s_w_blk[0] = best_b;
                            s_w_pos[1] = need_mid;
                            s_w_pos[2] = need_mid;
                            s_w_pos[3] = need_mid;
                        }
                    }
                }
            }
            __syncthreads();
        }
    }

    if (threadIdx.x == 0) {
        // Compact and insertion-sort the selected middle groups by logical group index ascending.
        int valid_mid = 0;
        for (int i = 0; i < need_mid; ++i) {
            if (s_top_blocks[i] >= 0) {
                s_top_blocks[valid_mid++] = s_top_blocks[i];
            }
        }
        for (int i = 1; i < valid_mid; ++i) {
            const int32_t key_g = s_top_blocks[i];
            int j = i - 1;
            while (j >= 0 && s_top_blocks[j] > key_g) {
                s_top_blocks[j + 1] = s_top_blocks[j];
                --j;
            }
            s_top_blocks[j + 1] = key_g;
        }

        int out_idx = 0;
        for (int g = 0; g < mid_start && out_idx < comp_budget; ++g) {
            const int32_t phys_page = req_block_table[g / groups_per_page];
            dst_log[out_idx] = g;
            dst_phys[out_idx] =
                (phys_page >= 0) ? (phys_page * groups_per_page + (g % groups_per_page)) : -1;
            ++out_idx;
        }
        for (int i = 0; i < valid_mid && out_idx < comp_budget; ++i) {
            const int32_t g = s_top_blocks[i];
            const int32_t phys_page = req_block_table[g / groups_per_page];
            dst_log[out_idx] = g;
            dst_phys[out_idx] =
                (phys_page >= 0) ? (phys_page * groups_per_page + (g % groups_per_page)) : -1;
            ++out_idx;
        }
        for (int g = mid_end; g < num_comp_groups && out_idx < comp_budget; ++g) {
            const int32_t phys_page = req_block_table[g / groups_per_page];
            dst_log[out_idx] = g;
            dst_phys[out_idx] =
                (phys_page >= 0) ? (phys_page * groups_per_page + (g % groups_per_page)) : -1;
            ++out_idx;
        }
        if (has_tail_group && out_idx < select_k_blocks) {
            const int32_t g = num_comp_groups;
            const int32_t phys_page = req_block_table[g / groups_per_page];
            dst_log[out_idx] = g;
            dst_phys[out_idx] =
                (phys_page >= 0) ? (phys_page * groups_per_page + (g % groups_per_page)) : -1;
            ++out_idx;
        }
        out_num_selected_blocks[tok] = out_idx;
        for (int i = out_idx; i < select_k_blocks; ++i) {
            dst_log[i] = -1;
            dst_phys[i] = -1;
        }
    }
}

// Sparse Paged GQA over the selected 4-token groups per token for FP16, Q8_0, and TQ4 KV caches.
// Grid: (num_tokens, local_q_heads), Block: 128 (4 warps, 8 dims per lane of head_dim=256).
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
#pragma unroll
            for (int j = 0; j < 8; ++j) {
                const int col = lane + 32 * j;
                float val = out_v[j] * kTq4Signs256[col];
                if (gate_head != nullptr) {
                    const float g = __half2float(gate_head[col]);
                    val *= (1.0f / (1.0f + expf(-g)));
                }
                dst_head[col] = __float2half(val);
            }
        }
        return;
    }

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
        dst_head[d] = __float2half(val);
    }
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
    w34::qsa_select_blocks_f16_kernel<<<batch->total_tokens, 128, 0, stream>>>(
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
    if (!ex || !q || !kv_layout || !selected_phys_blocks || !selected_logical_blocks ||
        !num_selected_blocks || !token_seq_pos || !out || num_tokens < 0 || local_q_heads <= 0 ||
        kv_layout->head_dim != 256) {
        return -1;
    }
    if (num_tokens == 0) return 0;
    cudaStream_t stream = reinterpret_cast<cudaStream_t>(ex->stream);
    dim3 grid(num_tokens, local_q_heads);
    w34::qsa_sparse_paged_attn_f16_kernel<<<grid, 128, 0, stream>>>(
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

} // extern "C"
