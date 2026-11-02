// Qwen3.8-Flash-Next (qwen4_exp) Query-Sparse Attention (QSA) & Indexer CUDA Kernels.
// Reference: nvidia/indexer_qsa.py, nvidia/qsa.py, nvidia/ops/qsa.py, nvidia/ops/qsa_special_kernels.py
//   - Pre-indexer GemmaRMSNorm + partial interleaved NeoX mRoPE (head_dim=128, rotary_dim=32, [6,5,5])
//   - Paged raw key cache [num_pages, 16, 128] + 4:1 compressed key cache [num_pages, 4, 128]
//   - Exact online-LSE per-head softmax + max-pool block scoring & top-k block selector (k=64 blocks = 1024 tokens,
//     8 sink blocks + 16 recent blocks + 40 top-scored middle blocks)
//   - Sparse paged GQA over selected blocks with causal boundary masking & optional sigmoid output gate
#include "../w34_cuda.h"

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <float.h>
#include <math.h>

extern "C" __device__ float __nv_sinf(float);
extern "C" __device__ float __nv_cosf(float);
extern "C" __device__ float __nv_powf(float, float);

namespace w34 {

__device__ __forceinline__ float qsa_warp_reduce_sum(float val) {
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        val += __shfl_xor_sync(0xffffffffu, val, offset);
    }
    return val;
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
    const __half* norm_w = is_q ? q_norm_weight : k_norm_weight;

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

// Writes indexer keys [num_tokens, head_dim] into raw_k_cache [num_pages, 16, head_dim]
// at write_slot_mapping[tok], then updates the corresponding 4:1 compressed sub-block
// in comp_k_cache [num_pages, 4, head_dim] whenever a 4-token sub-block is populated.
__global__ void qsa_indexer_write_raw_k_f16_kernel(
    const __half* __restrict__ k_in,
    __half* __restrict__ raw_k_cache,
    const int32_t* __restrict__ write_slot_mapping,
    const uint8_t* __restrict__ valid_token_mask,
    int num_tokens,
    int head_dim) {
    const int tok = blockIdx.x;
    if (tok >= num_tokens) return;
    if (valid_token_mask != nullptr && valid_token_mask[tok] == 0) return;
    const int32_t slot = write_slot_mapping[tok];
    if (slot < 0) return;

    const __half* src = k_in + (size_t)tok * head_dim;
    __half* dst = raw_k_cache + (size_t)slot * head_dim;
    for (int d = threadIdx.x; d < head_dim; d += blockDim.x) {
        dst[d] = src[d];
    }
}

// Updates 4:1 compressed key cache [num_pages, 4, head_dim] from raw_k_cache [num_pages, 16, head_dim]
// for every touched slot's 4-token aligned sub-block.
__global__ void qsa_indexer_update_comp_k_f16_kernel(
    const __half* __restrict__ raw_k_cache,
    __half* __restrict__ comp_k_cache,
    const int32_t* __restrict__ write_slot_mapping,
    const uint8_t* __restrict__ valid_token_mask,
    int num_tokens,
    int head_dim) {
    const int tok = blockIdx.x;
    if (tok >= num_tokens) return;
    if (valid_token_mask != nullptr && valid_token_mask[tok] == 0) return;
    const int32_t slot = write_slot_mapping[tok];
    if (slot < 0) return;

    const int32_t sub_block = slot >> 2; // slot / 4
    const int32_t base_slot = sub_block << 2;
    const __half* raw_base = raw_k_cache + (size_t)base_slot * head_dim;
    __half* comp_dst = comp_k_cache + (size_t)sub_block * head_dim;

    for (int d = threadIdx.x; d < head_dim; d += blockDim.x) {
        const float v0 = __half2float(raw_base[0 * head_dim + d]);
        const float v1 = __half2float(raw_base[1 * head_dim + d]);
        const float v2 = __half2float(raw_base[2 * head_dim + d]);
        const float v3 = __half2float(raw_base[3 * head_dim + d]);
        comp_dst[d] = __float2half(0.25f * (v0 + v1 + v2 + v3));
    }
}

// Selects up to select_k_blocks (64) logical blocks per query token using the QSA Indexer:
//   - If num_valid_blocks <= select_k_blocks: selects all 0..num_valid_blocks in ascending order.
//   - Otherwise: forces sink blocks [0..sink_blocks) and recent blocks [num_valid_blocks-recent_blocks..num_valid_blocks),
//     scores candidate middle blocks via exact per-head softmax + 4:1 max-pooling + head sum,
//     picks top (select_k_blocks - num_forced) middle blocks, and sorts selected logical blocks ascending.
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
    const int current_block = token_pos / block_size;
    const int num_valid_blocks = current_block + 1;
    const int32_t* req_block_table = batch.block_tables + (size_t)req * batch.max_blocks_per_req;

    if (threadIdx.x == 0) {
        out_token_seq_pos[tok] = token_pos;
    }

    if (num_valid_blocks <= select_k_blocks) {
        if (threadIdx.x == 0) {
            out_num_selected_blocks[tok] = num_valid_blocks;
        }
        for (int i = threadIdx.x; i < select_k_blocks; i += blockDim.x) {
            if (i < num_valid_blocks) {
                dst_phys[i] = req_block_table[i];
                dst_log[i] = i;
            } else {
                dst_phys[i] = -1;
                dst_log[i] = -1;
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
    // Compressed length ends at the last completed 4-token group before or at token_pos.
    const int num_comp_groups = token_pos / 4;
    const int num_full_comp_blocks = num_comp_groups / 4;
    const int rem_comp_groups = num_comp_groups % 4;

    // Pass 1: Compute per-head log-sum-exp over all valid compressed groups 0..num_comp_groups.
    // Warp 0..3 handles blocks in parallel (4 warps in blockDim=128).
    const int wid = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const int num_warps = blockDim.x >> 5;

    float m_h[4] = {-FLT_MAX, -FLT_MAX, -FLT_MAX, -FLT_MAX};
    float d_h[4] = {0.0f, 0.0f, 0.0f, 0.0f};

    const int total_comp_blocks = num_full_comp_blocks + ((rem_comp_groups > 0) ? 1 : 0);
    for (int b = wid; b < total_comp_blocks; b += num_warps) {
        const int32_t phys_page = req_block_table[b];
        if (phys_page < 0) continue;
        const int c_limit = (b < num_full_comp_blocks) ? 4 : rem_comp_groups;
        for (int c = 0; c < c_limit; ++c) {
            const __half* ck = comp_k_cache + ((size_t)phys_page * 4 + c) * head_dim;
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
#pragma unroll
            for (int h = 0; h < 4; ++h) {
                const float logit = qsa_warp_reduce_sum(dot_h[h]) * scale;
                const float m_new = fmaxf(m_h[h], logit);
                d_h[h] = d_h[h] * expf(m_h[h] - m_new) + expf(logit - m_new);
                m_h[h] = m_new;
            }
        }
    }

    __shared__ float s_m[8][4];
    __shared__ float s_d[8][4];
    __shared__ float s_lse[4];
    if (lane == 0) {
#pragma unroll
        for (int h = 0; h < 4; ++h) {
            s_m[wid][h] = m_h[h];
            s_d[wid][h] = d_h[h];
        }
    }
    __syncthreads();

    if (threadIdx.x < 4) {
        const int h = threadIdx.x;
        float m_all = -FLT_MAX;
        float d_all = 0.0f;
        for (int w = 0; w < num_warps; ++w) {
            const float mw = s_m[w][h];
            const float dw = s_d[w][h];
            if (dw > 0.0f) {
                const float m_new = fmaxf(m_all, mw);
                d_all = d_all * expf(m_all - m_new) + dw * expf(mw - m_new);
                m_all = m_new;
            }
        }
        s_lse[h] = (d_all > 0.0f) ? (m_all + logf(d_all)) : 0.0f;
    }
    __syncthreads();

    // Candidate middle blocks are in [mid_start, mid_end).
    const int mid_start = (sink_blocks < num_valid_blocks) ? sink_blocks : num_valid_blocks;
    const int mid_end = ((num_valid_blocks - recent_blocks) > mid_start)
        ? (num_valid_blocks - recent_blocks)
        : mid_start;
    const int num_forced = mid_start + (num_valid_blocks - mid_end);
    const int need_mid = (select_k_blocks > num_forced) ? (select_k_blocks - num_forced) : 0;

    // Maintain warp-collaborative top-need_mid candidate middle blocks in shared memory.
    __shared__ float s_top_scores[128];
    __shared__ int32_t s_top_blocks[128];
    for (int i = threadIdx.x; i < 128; i += blockDim.x) {
        s_top_scores[i] = -FLT_MAX;
        s_top_blocks[i] = -1;
    }
    __syncthreads();

    __shared__ float s_chunk_scores[4];
    __shared__ int32_t s_chunk_blocks[4];

    for (int b_base = mid_start; b_base < mid_end; b_base += num_warps) {
        const int b = b_base + wid;
        float blk_score = -FLT_MAX;
        if (b < mid_end && b < total_comp_blocks) {
            const int32_t phys_page = req_block_table[b];
            const int c_limit = (b < num_full_comp_blocks) ? 4 : rem_comp_groups;
            if (phys_page >= 0 && c_limit > 0) {
                float max_logit[4] = {-FLT_MAX, -FLT_MAX, -FLT_MAX, -FLT_MAX};
                for (int c = 0; c < c_limit; ++c) {
                    const __half* ck = comp_k_cache + ((size_t)phys_page * 4 + c) * head_dim;
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
#pragma unroll
                    for (int h = 0; h < 4; ++h) {
                        const float logit = qsa_warp_reduce_sum(dot_h[h]) * scale;
                        max_logit[h] = fmaxf(max_logit[h], logit);
                    }
                }
                blk_score = expf(max_logit[0] - s_lse[0]) + expf(max_logit[1] - s_lse[1]) +
                            expf(max_logit[2] - s_lse[2]) + expf(max_logit[3] - s_lse[3]);
            }
        }
        if (lane == 0) {
            s_chunk_scores[wid] = blk_score;
            s_chunk_blocks[wid] = (b < mid_end) ? b : -1;
        }
        __syncthreads();

        if (threadIdx.x == 0 && need_mid > 0) {
            for (int w = 0; w < num_warps; ++w) {
                const float sc = s_chunk_scores[w];
                const int32_t lb = s_chunk_blocks[w];
                if (lb >= 0 && sc > s_top_scores[need_mid - 1]) {
                    int pos = need_mid - 1;
                    while (pos > 0 && sc > s_top_scores[pos - 1]) {
                        s_top_scores[pos] = s_top_scores[pos - 1];
                        s_top_blocks[pos] = s_top_blocks[pos - 1];
                        --pos;
                    }
                    s_top_scores[pos] = sc;
                    s_top_blocks[pos] = lb;
                }
            }
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        // Insertion-sort the selected middle blocks by logical block index ascending.
        int valid_mid = 0;
        while (valid_mid < need_mid && s_top_blocks[valid_mid] >= 0) {
            ++valid_mid;
        }
        for (int i = 1; i < valid_mid; ++i) {
            const int32_t key_b = s_top_blocks[i];
            int j = i - 1;
            while (j >= 0 && s_top_blocks[j] > key_b) {
                s_top_blocks[j + 1] = s_top_blocks[j];
                --j;
            }
            s_top_blocks[j + 1] = key_b;
        }

        int out_idx = 0;
        for (int b = 0; b < mid_start && out_idx < select_k_blocks; ++b) {
            dst_log[out_idx] = b;
            dst_phys[out_idx] = req_block_table[b];
            ++out_idx;
        }
        for (int i = 0; i < valid_mid && out_idx < select_k_blocks; ++i) {
            const int32_t b = s_top_blocks[i];
            dst_log[out_idx] = b;
            dst_phys[out_idx] = req_block_table[b];
            ++out_idx;
        }
        for (int b = mid_end; b < num_valid_blocks && out_idx < select_k_blocks; ++b) {
            dst_log[out_idx] = b;
            dst_phys[out_idx] = req_block_table[b];
            ++out_idx;
        }
        out_num_selected_blocks[tok] = out_idx;
        for (int i = out_idx; i < select_k_blocks; ++i) {
            dst_log[i] = -1;
            dst_phys[i] = -1;
        }
    }
}

// Sparse Paged GQA over the selected blocks per token for FP16/BF16 KV cache.
// Grid: (num_tokens, local_q_heads), Block: 128 (4 warps, 2 dims per lane of head_dim=256).
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
    const int gqa_group = local_q_heads / kv_layout.local_kv_heads;
    const int kv_head = q_head / gqa_group;

    const __half* q_head_ptr = q + ((size_t)tok * local_q_heads + q_head) * head_dim;
    __shared__ float s_q[256];
    for (int d = threadIdx.x; d < head_dim; d += blockDim.x) {
        s_q[d] = __half2float(q_head_ptr[d]) * sm_scale;
    }
    __syncthreads();

    const int wid = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const int num_warps = blockDim.x >> 5;

    float m_i = -FLT_MAX;
    float d_i = 0.0f;
    float acc[8] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};

    const int32_t* tok_phys = selected_phys_blocks + (size_t)tok * select_k_blocks;
    const int32_t* tok_log = selected_logical_blocks + (size_t)tok * select_k_blocks;

    const uint8_t* k_base = reinterpret_cast<const uint8_t*>((uintptr_t)kv_layout.k_base);
    const uint8_t* v_base = reinterpret_cast<const uint8_t*>((uintptr_t)kv_layout.v_base);

    const int total_sel_tokens = n_sel * block_size;
    for (int t_idx = wid; t_idx < total_sel_tokens; t_idx += num_warps) {
        const int blk_idx = t_idx / block_size;
        const int page_off = t_idx % block_size;
        const int32_t phys_page = tok_phys[blk_idx];
        const int32_t log_blk = tok_log[blk_idx];
        const int kv_pos = log_blk * block_size + page_off;
        if (phys_page < 0 || kv_pos > seq_pos) continue;

        const __half* k_tok = reinterpret_cast<const __half*>(
            k_base + (size_t)phys_page * kv_layout.k_page_stride_bytes +
            (size_t)kv_head * kv_layout.k_head_stride_bytes +
            (size_t)page_off * kv_layout.k_token_stride_bytes);
        const __half* v_tok = reinterpret_cast<const __half*>(
            v_base + (size_t)phys_page * kv_layout.v_page_stride_bytes +
            (size_t)kv_head * kv_layout.v_head_stride_bytes +
            (size_t)page_off * kv_layout.v_token_stride_bytes);

        float dot = 0.0f;
#pragma unroll
        for (int j = 0; j < 8; ++j) {
            const int d = lane + j * 32;
            if (d < head_dim) {
                dot += s_q[d] * __half2float(k_tok[d]);
            }
        }
        const float score = qsa_warp_reduce_sum(dot);
        const float m_new = fmaxf(m_i, score);
        const float alpha = expf(m_i - m_new);
        const float beta = expf(score - m_new);
        d_i = d_i * alpha + beta;
#pragma unroll
        for (int j = 0; j < 8; ++j) {
            const int d = lane + j * 32;
            if (d < head_dim) {
                acc[j] = acc[j] * alpha + beta * __half2float(v_tok[d]);
            }
        }
        m_i = m_new;
    }

    __shared__ float s_m[8];
    __shared__ float s_d[8];
    __shared__ float s_acc[8][256];

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
    for (int w = 0; w < num_warps; ++w) {
        if (s_d[w] > 0.0f) {
            m_all = fmaxf(m_all, s_m[w]);
        }
    }
    float d_all = 0.0f;
    float w_scale[8];
    for (int w = 0; w < num_warps; ++w) {
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

    for (int d = threadIdx.x; d < head_dim; d += blockDim.x) {
        float sum_v = 0.0f;
        for (int w = 0; w < num_warps; ++w) {
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
    const int32_t* write_slot_mapping,
    const uint8_t* valid_token_mask,
    int num_tokens,
    int head_dim) {
    if (!ex || !k_in || !raw_k_cache || !comp_k_cache || !write_slot_mapping || num_tokens < 0 ||
        head_dim <= 0) {
        return -1;
    }
    if (num_tokens == 0) return 0;
    cudaStream_t stream = reinterpret_cast<cudaStream_t>(ex->stream);
    w34::qsa_indexer_write_raw_k_f16_kernel<<<num_tokens, 128, 0, stream>>>(
        static_cast<const __half*>(k_in),
        static_cast<__half*>(raw_k_cache),
        write_slot_mapping,
        valid_token_mask,
        num_tokens,
        head_dim);
    w34::qsa_indexer_update_comp_k_f16_kernel<<<num_tokens, 128, 0, stream>>>(
        static_cast<const __half*>(raw_k_cache),
        static_cast<__half*>(comp_k_cache),
        write_slot_mapping,
        valid_token_mask,
        num_tokens,
        head_dim);
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
