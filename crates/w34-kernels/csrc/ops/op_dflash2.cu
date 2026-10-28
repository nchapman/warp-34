// Phase 4 speculative decoding kernels:
// - MTP horizontal feature concatenation (`w34_concat_cols_2_f16`)
// - TapPlan auxiliary hidden-state capture (`w34_tap_add_f16_to_f32`)
// - Native BF16 / FP32 DFlash2 block-diffusion drafter kernels
//   (`w34_gemm_bf16_rows`, `w34_rms_norm_f32`, `w34_dflash_conv_f32`,
//    `w34_dflash_rope_f32`, `w34_dflash_kv_write_f32_to_bf16`,
//    `w34_dflash_attention_f32`, `w34_dflash_edges_f32`, `w34_topk_rows_f16`)
#include "../w34_cuda.h"
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <float.h>
#include <math.h>

namespace {

__global__ void concat_cols_2_f16_kernel(
    const __half* __restrict__ src0,
    const __half* __restrict__ src1,
    __half* __restrict__ dst,
    int num_tokens,
    int dim0,
    int dim1) {
    const int row = blockIdx.y;
    if (row >= num_tokens) return;
    const int total = dim0 + dim1;
    for (int col = blockIdx.x * blockDim.x + threadIdx.x; col < total;
         col += blockDim.x * gridDim.x) {
        if (col < dim0) {
            dst[row * total + col] = src0[row * dim0 + col];
        } else {
            dst[row * total + col] = src1[row * dim1 + (col - dim0)];
        }
    }
}

__global__ void tap_add_f16_to_f32_kernel(
    const __half* __restrict__ res_f16,
    const __half* __restrict__ mlp_f16,
    float* __restrict__ dst_f32,
    int num_tokens,
    int hidden_size,
    int dst_row_stride,
    int dst_col_offset) {
    const int row = blockIdx.y;
    if (row >= num_tokens) return;
    for (int col = blockIdx.x * blockDim.x + threadIdx.x; col < hidden_size;
         col += blockDim.x * gridDim.x) {
        const float r = __half2float(res_f16[(int64_t)row * hidden_size + col]);
        const float m = __half2float(mlp_f16[(int64_t)row * hidden_size + col]);
        dst_f32[(int64_t)row * dst_row_stride + dst_col_offset + col] = r + m;
    }
}

__global__ void cast_f32_bf16_kernel(
    const float* __restrict__ src,
    __nv_bfloat16* __restrict__ dst,
    int64_t n) {
    const int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        dst[i] = __float2bfloat16(src[i]);
    }
}

__global__ void cast_f32_f16_kernel(
    const float* __restrict__ src,
    __half* __restrict__ dst,
    int64_t n) {
    const int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        dst[i] = __float2half(src[i]);
    }
}

__global__ void embed_lookup_f16_to_f32_kernel(
    const int32_t* __restrict__ token_ids,
    const __half* __restrict__ embed_weight,
    float* __restrict__ out_f32,
    int num_tokens,
    int hidden_size,
    int vocab_size) {
    const int row = blockIdx.y;
    if (row >= num_tokens) return;
    int32_t tok = token_ids[row];
    if (tok < 0 || tok >= vocab_size) tok = 0;
    const __half* src_row = embed_weight + (int64_t)tok * hidden_size;
    float* dst_row = out_f32 + (int64_t)row * hidden_size;
    for (int col = blockIdx.x * blockDim.x + threadIdx.x; col < hidden_size;
         col += blockDim.x * gridDim.x) {
        dst_row[col] = __half2float(src_row[col]);
    }
}

__global__ void embed_lookup_bf16_to_f32_kernel(
    const int32_t* __restrict__ token_ids,
    const __nv_bfloat16* __restrict__ embed_weight,
    float* __restrict__ out_f32,
    int num_tokens,
    int hidden_size,
    int vocab_size) {
    const int row = blockIdx.y;
    if (row >= num_tokens) return;
    int32_t tok = token_ids[row];
    if (tok < 0 || tok >= vocab_size) tok = 0;
    const __nv_bfloat16* src_row = embed_weight + (int64_t)tok * hidden_size;
    float* dst_row = out_f32 + (int64_t)row * hidden_size;
    for (int col = blockIdx.x * blockDim.x + threadIdx.x; col < hidden_size;
         col += blockDim.x * gridDim.x) {
        dst_row[col] = __bfloat162float(src_row[col]);
    }
}

// Multi-row BF16 GEMV/GEMM with FP32 accumulation:
// y [rows, n] = x16 [rows, k] @ w [n, k]^T
template <int ROWS>
__global__ void gemm_bf16_rows_xsmem_kernel(
    float* __restrict__ y,
    const __nv_bfloat16* __restrict__ x16,
    const __nv_bfloat16* __restrict__ w,
    int64_t n,
    int64_t k) {
    extern __shared__ __nv_bfloat162 xs[];
    const int warp = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const int n_warps = blockDim.x >> 5;
    const int64_t col = (int64_t)blockIdx.x * n_warps + warp;
    const int64_t k2 = k >> 1;
    const __nv_bfloat162* x2_in = reinterpret_cast<const __nv_bfloat162*>(x16);

    const int64_t total_x2 = (int64_t)ROWS * k2;
    for (int64_t i = threadIdx.x; i < total_x2; i += blockDim.x) {
        xs[i] = x2_in[i];
    }
    __syncthreads();
    if (col >= n) return;

    const __nv_bfloat162* w2 = reinterpret_cast<const __nv_bfloat162*>(w + col * k);
    float acc[ROWS];
    #pragma unroll
    for (int r = 0; r < ROWS; ++r) acc[r] = 0.0f;

    for (int64_t j2 = lane; j2 < k2; j2 += 32) {
        const float2 wf = __bfloat1622float2(w2[j2]);
        #pragma unroll
        for (int r = 0; r < ROWS; ++r) {
            const float2 xf = __bfloat1622float2(xs[r * k2 + j2]);
            acc[r] = fmaf(wf.x, xf.x, acc[r]);
            acc[r] = fmaf(wf.y, xf.y, acc[r]);
        }
    }
    if ((k & 1) != 0 && lane == 0) {
        const float wl = __bfloat162float(w[col * k + (k - 1)]);
        #pragma unroll
        for (int r = 0; r < ROWS; ++r) {
            acc[r] = fmaf(wl, __bfloat162float(x16[r * k + (k - 1)]), acc[r]);
        }
    }
    #pragma unroll
    for (int r = 0; r < ROWS; ++r) {
        float v = acc[r];
        #pragma unroll
        for (int off = 16; off > 0; off >>= 1) {
            v += __shfl_down_sync(0xffffffffu, v, off);
        }
        if (lane == 0) {
            y[r * n + col] = v;
        }
    }
}

template <int ROWS>
__global__ void gemm_bf16_rows_kernel(
    float* __restrict__ y,
    const __nv_bfloat16* __restrict__ x16,
    const __nv_bfloat16* __restrict__ w,
    int64_t n,
    int64_t k) {
    const int warp = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const int n_warps = blockDim.x >> 5;
    const int64_t col = (int64_t)blockIdx.x * n_warps + warp;
    if (col >= n) return;

    const int64_t k2 = k >> 1;
    const __nv_bfloat162* w2 = reinterpret_cast<const __nv_bfloat162*>(w + col * k);
    const __nv_bfloat162* x2 = reinterpret_cast<const __nv_bfloat162*>(x16);

    float acc[ROWS];
    #pragma unroll
    for (int r = 0; r < ROWS; ++r) acc[r] = 0.0f;

    for (int64_t j2 = lane; j2 < k2; j2 += 32) {
        const float2 wf = __bfloat1622float2(w2[j2]);
        #pragma unroll
        for (int r = 0; r < ROWS; ++r) {
            const float2 xf = __bfloat1622float2(x2[r * k2 + j2]);
            acc[r] = fmaf(wf.x, xf.x, acc[r]);
            acc[r] = fmaf(wf.y, xf.y, acc[r]);
        }
    }
    if ((k & 1) != 0 && lane == 0) {
        const float wl = __bfloat162float(w[col * k + (k - 1)]);
        #pragma unroll
        for (int r = 0; r < ROWS; ++r) {
            acc[r] = fmaf(wl, __bfloat162float(x16[r * k + (k - 1)]), acc[r]);
        }
    }
    #pragma unroll
    for (int r = 0; r < ROWS; ++r) {
        float v = acc[r];
        #pragma unroll
        for (int off = 16; off > 0; off >>= 1) {
            v += __shfl_down_sync(0xffffffffu, v, off);
        }
        if (lane == 0) {
            y[r * n + col] = v;
        }
    }
}

__global__ void rms_norm_f32_kernel(
    const float* __restrict__ x,
    const float* __restrict__ weight,
    float* __restrict__ out,
    int num_rows,
    int dim,
    float eps) {
    const int row = blockIdx.x;
    if (row >= num_rows) return;
    const float* x_row = x + (int64_t)row * dim;
    float* out_row = out + (int64_t)row * dim;

    float sum_sq = 0.0f;
    for (int i = threadIdx.x; i < dim; i += blockDim.x) {
        const float v = x_row[i];
        sum_sq = fmaf(v, v, sum_sq);
    }

    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
        sum_sq += __shfl_down_sync(0xffffffffu, sum_sq, off);
    }
    __shared__ float warp_sums[8];
    if (lane == 0) warp_sums[warp] = sum_sq;
    __syncthreads();

    if (warp == 0) {
        const int num_warps = blockDim.x >> 5;
        float total = (lane < num_warps) ? warp_sums[lane] : 0.0f;
        #pragma unroll
        for (int off = 16; off > 0; off >>= 1) {
            total += __shfl_down_sync(0xffffffffu, total, off);
        }
        if (lane == 0) {
            warp_sums[0] = rsqrtf(total / (float)dim + eps);
        }
    }
    __syncthreads();
    const float inv_rms = warp_sums[0];

    for (int i = threadIdx.x; i < dim; i += blockDim.x) {
        out_row[i] = x_row[i] * inv_rms * weight[i];
    }
}

__global__ void add_f32_kernel(
    const float* __restrict__ a,
    const float* __restrict__ b,
    float* __restrict__ out,
    int64_t n) {
    const int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        out[i] = a[i] + b[i];
    }
}

__global__ void silu_and_mul_f32_kernel(
    const float* __restrict__ gate,
    const float* __restrict__ up,
    float* __restrict__ out,
    int64_t n) {
    const int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        const float g = gate[i];
        const float silu_g = g / (1.0f + expf(-g));
        out[i] = silu_g * up[i];
    }
}

__global__ void split_cols_2_f32_kernel(
    const float* __restrict__ src,
    float* __restrict__ dst0,
    float* __restrict__ dst1,
    int num_rows,
    int dim0,
    int dim1) {
    const int row = blockIdx.y;
    if (row >= num_rows) return;
    const int total = dim0 + dim1;
    for (int col = blockIdx.x * blockDim.x + threadIdx.x; col < total;
         col += blockDim.x * gridDim.x) {
        const float v = src[(int64_t)row * total + col];
        if (col < dim0) {
            dst0[(int64_t)row * dim0 + col] = v;
        } else {
            dst1[(int64_t)row * dim1 + (col - dim0)] = v;
        }
    }
}

__global__ void split_cols_3_f32_kernel(
    const float* __restrict__ src,
    float* __restrict__ dst0,
    float* __restrict__ dst1,
    float* __restrict__ dst2,
    int num_rows,
    int dim0,
    int dim1,
    int dim2) {
    const int row = blockIdx.y;
    if (row >= num_rows) return;
    const int total = dim0 + dim1 + dim2;
    for (int col = blockIdx.x * blockDim.x + threadIdx.x; col < total;
         col += blockDim.x * gridDim.x) {
        const float v = src[(int64_t)row * total + col];
        if (col < dim0) {
            dst0[(int64_t)row * dim0 + col] = v;
        } else if (col < dim0 + dim1) {
            dst1[(int64_t)row * dim1 + (col - dim0)] = v;
        } else {
            dst2[(int64_t)row * dim2 + (col - dim0 - dim1)] = v;
        }
    }
}

__global__ void dflash_conv_f32_kernel(
    float* __restrict__ y,
    const float* __restrict__ x,
    const float* __restrict__ kp,
    const float* __restrict__ base,
    int64_t rows,
    int64_t hidden,
    int64_t block,
    int64_t group,
    int64_t taps,
    int64_t conv) {
    const int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= rows * hidden) return;
    const int64_t r = idx / hidden;
    const int64_t c = idx - r * hidden;
    const int64_t ng = hidden / group;
    const int64_t g = c / group;
    const int64_t pos = r % block;
    const int64_t kp_stride = 2 * taps * ng;
    float acc = 0.0f;
    for (int64_t tap = 0; tap < taps; ++tap) {
        if (pos >= tap && r >= tap) {
            const float coeff = base[(conv * taps + tap) * hidden + c]
                              + kp[r * kp_stride + (conv * taps + tap) * ng + g];
            acc = fmaf(coeff, x[(r - tap) * hidden + c], acc);
        }
    }
    y[idx] = acc;
}

__global__ void dflash_rope_f32_kernel(
    float* __restrict__ x,
    const float* __restrict__ pos,
    int64_t rows,
    int64_t heads,
    int64_t d,
    float base) {
    const int64_t half = d >> 1;
    const int64_t total = rows * heads * half;
    const int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) return;
    const int64_t i = idx % half;
    const int64_t rh = idx / half;
    const int64_t r = rh / heads;
    const float p = pos[r];
    const float inv_freq = powf(base, -2.0f * (float)i / (float)d);
    const float ang = p * inv_freq;
    float s, c;
    sincosf(ang, &s, &c);
    float* row = x + rh * d;
    const float x0 = row[i];
    const float x1 = row[i + half];
    row[i]        = x0 * c - x1 * s;
    row[i + half] = x1 * c + x0 * s;
}

__global__ void dflash_kv_write_f32_to_bf16_kernel(
    const float* __restrict__ k_in,
    const float* __restrict__ v_in,
    __nv_bfloat16* __restrict__ k_cache,
    __nv_bfloat16* __restrict__ v_cache,
    int pos_base,
    int num_rows,
    int row_elems) {
    const int row = blockIdx.y;
    if (row >= num_rows) return;
    const int64_t dst_off = (int64_t)(pos_base + row) * row_elems;
    const int64_t src_off = (int64_t)row * row_elems;
    for (int col = blockIdx.x * blockDim.x + threadIdx.x; col < row_elems;
         col += blockDim.x * gridDim.x) {
        k_cache[dst_off + col] = __float2bfloat16(k_in[src_off + col]);
        v_cache[dst_off + col] = __float2bfloat16(v_in[src_off + col]);
    }
}

// Non-causal / symmetric-sliding-window block attention for DFlash2:
// Q:       [block_rows, num_q_heads, head_dim] (FP32)
// K_cache: [total_kv_len, num_kv_heads, head_dim] (BF16)
// V_cache: [total_kv_len, num_kv_heads, head_dim] (BF16)
// Out:     [block_rows, num_q_heads, head_dim] (FP32)
// Grid: (block_rows, num_q_heads), Block: 128 threads (4 warps)
__global__ void dflash_attention_f32_kernel(
    const float* __restrict__ q_in,
    const __nv_bfloat16* __restrict__ k_cache,
    const __nv_bfloat16* __restrict__ v_cache,
    float* __restrict__ attn_out,
    int block_rows,
    int q_pos_base,
    int total_kv_len,
    int num_q_heads,
    int num_kv_heads,
    int head_dim,
    int sliding_window,
    float sm_scale) {
    const int q_row = blockIdx.x;
    const int q_head = blockIdx.y;
    if (q_row >= block_rows || q_head >= num_q_heads) return;

    const int gqa_ratio = num_q_heads / num_kv_heads;
    const int kv_head = q_head / gqa_ratio;
    const int warp_id = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const int num_warps = blockDim.x >> 5;

    // Each lane owns up to 8 elements of head_dim (256 / 32 = 8).
    float q_reg[8];
    float acc_v[8];
    const float* q_ptr = q_in + ((int64_t)q_row * num_q_heads + q_head) * head_dim;
    #pragma unroll
    for (int i = 0; i < 8; ++i) {
        const int d = lane + i * 32;
        q_reg[i] = (d < head_dim) ? (q_ptr[d] * sm_scale) : 0.0f;
        acc_v[i] = 0.0f;
    }

    const int ii = q_pos_base + q_row;
    const int lo = (sliding_window > 0) ? max(0, ii - (sliding_window - 1)) : 0;
    // SGLang FlashInfer ENCODER_ONLY: causal=false, window_left=window-1.
    // Future keys in the materialized draft block have no right-window bound.
    const int hi = total_kv_len - 1;

    float m_w = -FLT_MAX;
    float l_w = 0.0f;

    for (int kv_pos = lo + warp_id; kv_pos <= hi; kv_pos += num_warps) {
        const __nv_bfloat16* k_ptr = k_cache + ((int64_t)kv_pos * num_kv_heads + kv_head) * head_dim;
        const __nv_bfloat16* v_ptr = v_cache + ((int64_t)kv_pos * num_kv_heads + kv_head) * head_dim;

        float dot = 0.0f;
        #pragma unroll
        for (int i = 0; i < 8; ++i) {
            const int d = lane + i * 32;
            if (d < head_dim) {
                dot = fmaf(q_reg[i], __bfloat162float(k_ptr[d]), dot);
            }
        }
        #pragma unroll
        for (int off = 16; off > 0; off >>= 1) {
            dot += __shfl_xor_sync(0xffffffffu, dot, off);
        }

        const float m_new = fmaxf(m_w, dot);
        const float alpha = (m_w == -FLT_MAX) ? 0.0f : expf(m_w - m_new);
        const float beta = expf(dot - m_new);
        l_w = l_w * alpha + beta;
        m_w = m_new;

        #pragma unroll
        for (int i = 0; i < 8; ++i) {
            const int d = lane + i * 32;
            if (d < head_dim) {
                acc_v[i] = fmaf(beta, __bfloat162float(v_ptr[d]), acc_v[i] * alpha);
            }
        }
    }

    // Cross-warp online softmax merge in shared memory.
    __shared__ float smem_m[4];
    __shared__ float smem_l[4];
    __shared__ float smem_v[4][256];

    if (lane == 0) {
        smem_m[warp_id] = m_w;
        smem_l[warp_id] = l_w;
    }
    #pragma unroll
    for (int i = 0; i < 8; ++i) {
        const int d = lane + i * 32;
        if (d < head_dim) {
            smem_v[warp_id][d] = acc_v[i];
        }
    }
    __syncthreads();

    if (warp_id == 0) {
        float m_global = -FLT_MAX;
        #pragma unroll
        for (int w = 0; w < 4; ++w) {
            m_global = fmaxf(m_global, smem_m[w]);
        }
        float l_global = 0.0f;
        float scales[4];
        #pragma unroll
        for (int w = 0; w < 4; ++w) {
            scales[w] = (smem_m[w] == -FLT_MAX || m_global == -FLT_MAX)
                ? 0.0f
                : expf(smem_m[w] - m_global);
            l_global += smem_l[w] * scales[w];
        }
        const float inv_l = (l_global > 0.0f) ? (1.0f / l_global) : 0.0f;
        float* out_ptr = attn_out + ((int64_t)q_row * num_q_heads + q_head) * head_dim;
        #pragma unroll
        for (int i = 0; i < 8; ++i) {
            const int d = lane + i * 32;
            if (d < head_dim) {
                float v_sum = 0.0f;
                #pragma unroll
                for (int w = 0; w < 4; ++w) {
                    v_sum = fmaf(smem_v[w][d], scales[w], v_sum);
                }
                out_ptr[d] = v_sum * inv_l;
            }
        }
    }
}

__global__ void dflash_edges_f32_kernel(
    float* __restrict__ edges,
    const __nv_bfloat16* __restrict__ pred_cb,
    const __nv_bfloat16* __restrict__ succ_cb,
    const float* __restrict__ hp,
    const int32_t* __restrict__ cand_ids,
    const float* __restrict__ cand_vals,
    int32_t bonus_id,
    int64_t k,
    int64_t top_k,
    int64_t rank) {
    const int64_t cell = blockIdx.x;
    if (cell >= k * top_k * top_k) return;
    const int64_t c = cell % top_k;
    const int64_t sp = cell / top_k;
    const int64_t p = sp % top_k;
    const int64_t s = sp / top_k;

    const int32_t pred_tok = (s == 0) ? bonus_id : cand_ids[(s - 1) * top_k + p];
    const int32_t succ_tok = cand_ids[s * top_k + c];
    const __nv_bfloat16* pr = pred_cb + (int64_t)pred_tok * rank;
    const __nv_bfloat16* su = succ_cb + (int64_t)succ_tok * rank;
    const float* h = hp + s * rank;

    float acc = 0.0f;
    for (int64_t r = threadIdx.x; r < rank; r += 32) {
        const float pv = __bfloat162float(pr[r]);
        const float sv = __bfloat162float(su[r]);
        acc = fmaf(pv * h[r], sv, acc);
    }
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
        acc += __shfl_down_sync(0xffffffffu, acc, off);
    }
    if (threadIdx.x == 0) {
        edges[cell] = cand_vals[s * top_k + c] + acc;
    }
}

__global__ void topk_rows_f16_kernel(
    const __half* __restrict__ logits,
    int32_t* __restrict__ out_ids,
    float* __restrict__ out_vals,
    int num_rows,
    int local_vocab_size,
    int vocab_offset,
    int top_k) {
    constexpr int MAX_K = 16;
    const int row = blockIdx.x;
    if (row >= num_rows) return;
    const __half* row_logits = logits + (int64_t)row * local_vocab_size;

    float tv[MAX_K];
    int32_t ti[MAX_K];
    #pragma unroll
    for (int m = 0; m < MAX_K; ++m) {
        tv[m] = -FLT_MAX;
        ti[m] = -1;
    }

    for (int col = threadIdx.x; col < local_vocab_size; col += blockDim.x) {
        const float v = __half2float(row_logits[col]);
        if (v > tv[top_k - 1]) {
            int pos = top_k - 1;
            while (pos > 0 && v > tv[pos - 1]) {
                tv[pos] = tv[pos - 1];
                ti[pos] = ti[pos - 1];
                --pos;
            }
            tv[pos] = v;
            ti[pos] = vocab_offset + col;
        }
    }

    extern __shared__ char smem[];
    float* sv = reinterpret_cast<float*>(smem);
    int32_t* si = reinterpret_cast<int32_t*>(sv + (size_t)blockDim.x * MAX_K);
    for (int m = 0; m < top_k; ++m) {
        sv[threadIdx.x * MAX_K + m] = tv[m];
        si[threadIdx.x * MAX_K + m] = ti[m];
    }
    __syncthreads();

    for (int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if ((int)threadIdx.x < stride) {
            const int a = threadIdx.x;
            const int b = threadIdx.x + stride;
            float mv[MAX_K];
            int32_t mi[MAX_K];
            int ia = 0;
            int ib = 0;
            for (int m = 0; m < top_k; ++m) {
                const float va = sv[a * MAX_K + ia];
                const float vb = sv[b * MAX_K + ib];
                const int32_t ida = si[a * MAX_K + ia];
                const int32_t idb = si[b * MAX_K + ib];
                if (va > vb || (va == vb && ida < idb)) {
                    mv[m] = va;
                    mi[m] = ida;
                    ++ia;
                } else {
                    mv[m] = vb;
                    mi[m] = idb;
                    ++ib;
                }
            }
            for (int m = 0; m < top_k; ++m) {
                sv[a * MAX_K + m] = mv[m];
                si[a * MAX_K + m] = mi[m];
            }
        }
        __syncthreads();
    }

    if ((int)threadIdx.x < top_k) {
        out_vals[(int64_t)row * top_k + threadIdx.x] = sv[threadIdx.x];
        out_ids[(int64_t)row * top_k + threadIdx.x] = si[threadIdx.x];
    }
}

__global__ void topk_rows_f32_kernel(
    const float* __restrict__ logits,
    int32_t* __restrict__ out_ids,
    float* __restrict__ out_vals,
    int num_rows,
    int local_vocab_size,
    int vocab_offset,
    int top_k) {
    constexpr int MAX_K = 16;
    const int row = blockIdx.x;
    if (row >= num_rows) return;
    const float* row_logits = logits + (int64_t)row * local_vocab_size;

    float tv[MAX_K];
    int32_t ti[MAX_K];
    #pragma unroll
    for (int m = 0; m < MAX_K; ++m) {
        tv[m] = -FLT_MAX;
        ti[m] = -1;
    }

    for (int col = threadIdx.x; col < local_vocab_size; col += blockDim.x) {
        const float v = row_logits[col];
        if (v > tv[top_k - 1]) {
            int pos = top_k - 1;
            while (pos > 0 && v > tv[pos - 1]) {
                tv[pos] = tv[pos - 1];
                ti[pos] = ti[pos - 1];
                --pos;
            }
            tv[pos] = v;
            ti[pos] = vocab_offset + col;
        }
    }

    extern __shared__ char smem[];
    float* sv = reinterpret_cast<float*>(smem);
    int32_t* si = reinterpret_cast<int32_t*>(sv + (size_t)blockDim.x * MAX_K);
    for (int m = 0; m < top_k; ++m) {
        sv[threadIdx.x * MAX_K + m] = tv[m];
        si[threadIdx.x * MAX_K + m] = ti[m];
    }
    __syncthreads();

    for (int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if ((int)threadIdx.x < stride) {
            const int a = threadIdx.x;
            const int b = threadIdx.x + stride;
            float mv[MAX_K];
            int32_t mi[MAX_K];
            int ia = 0;
            int ib = 0;
            for (int m = 0; m < top_k; ++m) {
                const float va = sv[a * MAX_K + ia];
                const float vb = sv[b * MAX_K + ib];
                const int32_t ida = si[a * MAX_K + ia];
                const int32_t idb = si[b * MAX_K + ib];
                if (va > vb || (va == vb && ida < idb)) {
                    mv[m] = va;
                    mi[m] = ida;
                    ++ia;
                } else {
                    mv[m] = vb;
                    mi[m] = idb;
                    ++ib;
                }
            }
            for (int m = 0; m < top_k; ++m) {
                sv[a * MAX_K + m] = mv[m];
                si[a * MAX_K + m] = mi[m];
            }
        }
        __syncthreads();
    }

    if ((int)threadIdx.x < top_k) {
        out_vals[(int64_t)row * top_k + threadIdx.x] = sv[threadIdx.x];
        out_ids[(int64_t)row * top_k + threadIdx.x] = si[threadIdx.x];
    }
}

int launch_gemm_bf16_rows_chunk(
    cudaStream_t stream,
    float* y,
    const __nv_bfloat16* x16,
    const __nv_bfloat16* w,
    int64_t rows,
    int64_t n,
    int64_t k) {
    const int threads = 256;
    const int n_warps = threads >> 5;
    const unsigned blocks = (unsigned)((n + n_warps - 1) / n_warps);
    const size_t smem = (size_t)rows * (size_t)k * sizeof(__nv_bfloat16);
    const bool use_smem = ((k & 1) == 0) && (smem <= 48 * 1024);

    #define CASE_R(R) \
        case R: \
            if (use_smem) { \
                gemm_bf16_rows_xsmem_kernel<R><<<blocks, threads, smem, stream>>>(y, x16, w, n, k); \
            } else { \
                gemm_bf16_rows_kernel<R><<<blocks, threads, 0, stream>>>(y, x16, w, n, k); \
            } \
            break

    switch (rows) {
        CASE_R(1);  CASE_R(2);  CASE_R(3);  CASE_R(4);
        CASE_R(5);  CASE_R(6);  CASE_R(7);  CASE_R(8);
        CASE_R(9);  CASE_R(10); CASE_R(11); CASE_R(12);
        CASE_R(13); CASE_R(14); CASE_R(15); CASE_R(16);
        default:
            return (int)cudaErrorInvalidValue;
    }
    #undef CASE_R
    return (int)cudaGetLastError();
}

} // namespace

extern "C" {

int w34_concat_cols_2_f16(
    const w34_exec* ex,
    const void* src0,
    const void* src1,
    void* dst,
    int num_tokens,
    int dim0,
    int dim1) {
    if (!ex || !src0 || !src1 || !dst || num_tokens < 0 || dim0 <= 0 || dim1 <= 0) {
        return (int)cudaErrorInvalidValue;
    }
    if (num_tokens == 0) return 0;
    const int total = dim0 + dim1;
    const dim3 block(256);
    const dim3 grid((unsigned)((total + 255) / 256), (unsigned)num_tokens);
    concat_cols_2_f16_kernel<<<grid, block, 0, (cudaStream_t)ex->stream>>>(
        static_cast<const __half*>(src0),
        static_cast<const __half*>(src1),
        static_cast<__half*>(dst),
        num_tokens,
        dim0,
        dim1);
    return (int)cudaGetLastError();
}

int w34_tap_add_f16_to_f32(
    const w34_exec* ex,
    const void* res_f16,
    const void* mlp_f16,
    float* dst_f32,
    int num_tokens,
    int hidden_size,
    int dst_row_stride,
    int dst_col_offset) {
    if (!ex || !res_f16 || !mlp_f16 || !dst_f32 || num_tokens < 0 || hidden_size <= 0 ||
        dst_col_offset < 0 || dst_row_stride < hidden_size ||
        dst_col_offset > dst_row_stride - hidden_size) {
        return (int)cudaErrorInvalidValue;
    }
    if (num_tokens == 0) return 0;
    const dim3 block(256);
    const dim3 grid((unsigned)((hidden_size + 255) / 256), (unsigned)num_tokens);
    tap_add_f16_to_f32_kernel<<<grid, block, 0, (cudaStream_t)ex->stream>>>(
        static_cast<const __half*>(res_f16),
        static_cast<const __half*>(mlp_f16),
        dst_f32,
        num_tokens,
        hidden_size,
        dst_row_stride,
        dst_col_offset);
    return (int)cudaGetLastError();
}

int w34_cast_f32_bf16(
    const w34_exec* ex,
    const float* src,
    void* dst_bf16,
    int64_t n) {
    if (!ex || !src || !dst_bf16 || n < 0) {
        return (int)cudaErrorInvalidValue;
    }
    if (n == 0) return 0;
    const int threads = 256;
    const unsigned blocks = (unsigned)((n + threads - 1) / threads);
    cast_f32_bf16_kernel<<<blocks, threads, 0, (cudaStream_t)ex->stream>>>(
        src, static_cast<__nv_bfloat16*>(dst_bf16), n);
    return (int)cudaGetLastError();
}

int w34_cast_f32_f16(
    const w34_exec* ex,
    const float* src,
    void* dst_f16,
    int64_t n) {
    if (!ex || !src || !dst_f16 || n < 0) {
        return (int)cudaErrorInvalidValue;
    }
    if (n == 0) return 0;
    const int threads = 256;
    const unsigned blocks = (unsigned)((n + threads - 1) / threads);
    cast_f32_f16_kernel<<<blocks, threads, 0, (cudaStream_t)ex->stream>>>(
        src, static_cast<__half*>(dst_f16), n);
    return (int)cudaGetLastError();
}

int w34_embed_lookup_f16_to_f32(
    const w34_exec* ex,
    const int32_t* token_ids,
    const void* embed_weight_f16,
    float* out_f32,
    int num_tokens,
    int hidden_size,
    int vocab_size) {
    if (!ex || !token_ids || !embed_weight_f16 || !out_f32 || num_tokens < 0 || hidden_size <= 0) {
        return (int)cudaErrorInvalidValue;
    }
    if (num_tokens == 0) return 0;
    const dim3 block(256);
    const dim3 grid((unsigned)((hidden_size + 255) / 256), (unsigned)num_tokens);
    embed_lookup_f16_to_f32_kernel<<<grid, block, 0, (cudaStream_t)ex->stream>>>(
        token_ids,
        static_cast<const __half*>(embed_weight_f16),
        out_f32,
        num_tokens,
        hidden_size,
        vocab_size);
    return (int)cudaGetLastError();
}

int w34_embed_lookup_bf16_to_f32(
    const w34_exec* ex,
    const int32_t* token_ids,
    const void* embed_weight_bf16,
    float* out_f32,
    int num_tokens,
    int hidden_size,
    int vocab_size) {
    if (!ex || !token_ids || !embed_weight_bf16 || !out_f32 || num_tokens < 0 || hidden_size <= 0) {
        return (int)cudaErrorInvalidValue;
    }
    if (num_tokens == 0) return 0;
    const dim3 block(256);
    const dim3 grid((unsigned)((hidden_size + 255) / 256), (unsigned)num_tokens);
    embed_lookup_bf16_to_f32_kernel<<<grid, block, 0, (cudaStream_t)ex->stream>>>(
        token_ids,
        static_cast<const __nv_bfloat16*>(embed_weight_bf16),
        out_f32,
        num_tokens,
        hidden_size,
        vocab_size);
    return (int)cudaGetLastError();
}

int w34_gemm_bf16_rows(
    const w34_exec* ex,
    float* y_f32,
    const void* x_bf16,
    const void* w_bf16,
    int64_t rows,
    int64_t n,
    int64_t k) {
    // Packed bfloat162 loads require even row strides for both operands.
    if (!ex || !y_f32 || !x_bf16 || !w_bf16 || rows < 0 || n <= 0 || k <= 0 || (k & 1)) {
        return (int)cudaErrorInvalidValue;
    }
    if (rows == 0) return 0;
    const __nv_bfloat16* x_ptr = static_cast<const __nv_bfloat16*>(x_bf16);
    const __nv_bfloat16* w_ptr = static_cast<const __nv_bfloat16*>(w_bf16);
    cudaStream_t stream = (cudaStream_t)ex->stream;

    int64_t off = 0;
    while (off < rows) {
        const int64_t chunk = (rows - off > 16) ? 16 : (rows - off);
        int rc = launch_gemm_bf16_rows_chunk(
            stream,
            y_f32 + off * n,
            x_ptr + off * k,
            w_ptr,
            chunk,
            n,
            k);
        if (rc != 0) return rc;
        off += chunk;
    }
    return 0;
}

int w34_rms_norm_f32(
    const w34_exec* ex,
    const float* x,
    const float* weight,
    float* out,
    int num_rows,
    int dim,
    float eps) {
    if (!ex || !x || !weight || !out || num_rows < 0 || dim <= 0) {
        return (int)cudaErrorInvalidValue;
    }
    if (num_rows == 0) return 0;
    rms_norm_f32_kernel<<<(unsigned)num_rows, 256, 0, (cudaStream_t)ex->stream>>>(
        x, weight, out, num_rows, dim, eps);
    return (int)cudaGetLastError();
}

int w34_add_f32(
    const w34_exec* ex,
    const float* a,
    const float* b,
    float* out,
    int64_t n) {
    if (!ex || !a || !b || !out || n < 0) {
        return (int)cudaErrorInvalidValue;
    }
    if (n == 0) return 0;
    const int threads = 256;
    const unsigned blocks = (unsigned)((n + threads - 1) / threads);
    add_f32_kernel<<<blocks, threads, 0, (cudaStream_t)ex->stream>>>(a, b, out, n);
    return (int)cudaGetLastError();
}

int w34_silu_and_mul_f32(
    const w34_exec* ex,
    const float* gate,
    const float* up,
    float* out,
    int64_t n) {
    if (!ex || !gate || !up || !out || n < 0) {
        return (int)cudaErrorInvalidValue;
    }
    if (n == 0) return 0;
    const int threads = 256;
    const unsigned blocks = (unsigned)((n + threads - 1) / threads);
    silu_and_mul_f32_kernel<<<blocks, threads, 0, (cudaStream_t)ex->stream>>>(gate, up, out, n);
    return (int)cudaGetLastError();
}

int w34_split_cols_2_f32(
    const w34_exec* ex,
    const float* src,
    float* dst0,
    float* dst1,
    int num_rows,
    int dim0,
    int dim1) {
    if (!ex || !src || !dst0 || !dst1 || num_rows < 0 || dim0 <= 0 || dim1 <= 0) {
        return (int)cudaErrorInvalidValue;
    }
    if (num_rows == 0) return 0;
    const int total = dim0 + dim1;
    const dim3 block(256);
    const dim3 grid((unsigned)((total + 255) / 256), (unsigned)num_rows);
    split_cols_2_f32_kernel<<<grid, block, 0, (cudaStream_t)ex->stream>>>(
        src, dst0, dst1, num_rows, dim0, dim1);
    return (int)cudaGetLastError();
}

int w34_split_cols_3_f32(
    const w34_exec* ex,
    const float* src,
    float* dst0,
    float* dst1,
    float* dst2,
    int num_rows,
    int dim0,
    int dim1,
    int dim2) {
    if (!ex || !src || !dst0 || !dst1 || !dst2 || num_rows < 0 || dim0 <= 0 || dim1 <= 0 || dim2 <= 0) {
        return (int)cudaErrorInvalidValue;
    }
    if (num_rows == 0) return 0;
    const int total = dim0 + dim1 + dim2;
    const dim3 block(256);
    const dim3 grid((unsigned)((total + 255) / 256), (unsigned)num_rows);
    split_cols_3_f32_kernel<<<grid, block, 0, (cudaStream_t)ex->stream>>>(
        src, dst0, dst1, dst2, num_rows, dim0, dim1, dim2);
    return (int)cudaGetLastError();
}

int w34_dflash_conv_f32(
    const w34_exec* ex,
    float* y,
    const float* x,
    const float* kp,
    const float* base,
    int64_t rows,
    int64_t hidden,
    int64_t block,
    int64_t group,
    int64_t taps,
    int64_t conv) {
    if (!ex || !y || !x || !kp || !base || rows < 0 || hidden <= 0 || block <= 0 ||
        group <= 0 || taps <= 0 || hidden % group != 0 || (conv != 0 && conv != 1)) {
        return (int)cudaErrorInvalidValue;
    }
    if (rows == 0) return 0;
    const int64_t total = rows * hidden;
    const int threads = 256;
    const unsigned blocks = (unsigned)((total + threads - 1) / threads);
    dflash_conv_f32_kernel<<<blocks, threads, 0, (cudaStream_t)ex->stream>>>(
        y, x, kp, base, rows, hidden, block, group, taps, conv);
    return (int)cudaGetLastError();
}

int w34_dflash_rope_f32(
    const w34_exec* ex,
    float* x,
    const float* pos,
    int64_t rows,
    int64_t heads,
    int64_t d,
    float base) {
    if (!ex || !x || !pos || rows < 0 || heads <= 0 || d <= 0 || (d & 1) != 0) {
        return (int)cudaErrorInvalidValue;
    }
    if (rows == 0) return 0;
    const int64_t total = rows * heads * (d >> 1);
    const int threads = 256;
    const unsigned blocks = (unsigned)((total + threads - 1) / threads);
    dflash_rope_f32_kernel<<<blocks, threads, 0, (cudaStream_t)ex->stream>>>(
        x, pos, rows, heads, d, base);
    return (int)cudaGetLastError();
}

int w34_dflash_kv_write_f32_to_bf16(
    const w34_exec* ex,
    const float* k_in,
    const float* v_in,
    void* k_cache_bf16,
    void* v_cache_bf16,
    int pos_base,
    int num_rows,
    int row_elems) {
    if (!ex || !k_in || !v_in || !k_cache_bf16 || !v_cache_bf16 || pos_base < 0 || num_rows < 0 || row_elems <= 0) {
        return (int)cudaErrorInvalidValue;
    }
    if (num_rows == 0) return 0;
    const dim3 block(256);
    const dim3 grid((unsigned)((row_elems + 255) / 256), (unsigned)num_rows);
    dflash_kv_write_f32_to_bf16_kernel<<<grid, block, 0, (cudaStream_t)ex->stream>>>(
        k_in,
        v_in,
        static_cast<__nv_bfloat16*>(k_cache_bf16),
        static_cast<__nv_bfloat16*>(v_cache_bf16),
        pos_base,
        num_rows,
        row_elems);
    return (int)cudaGetLastError();
}

int w34_dflash_attention_f32(
    const w34_exec* ex,
    const float* q_in,
    const void* k_cache_bf16,
    const void* v_cache_bf16,
    float* attn_out,
    int block_rows,
    int q_pos_base,
    int total_kv_len,
    int num_q_heads,
    int num_kv_heads,
    int head_dim,
    int sliding_window,
    float sm_scale) {
    if (!ex || !q_in || !k_cache_bf16 || !v_cache_bf16 || !attn_out ||
        block_rows <= 0 || total_kv_len <= 0 || num_q_heads <= 0 ||
        num_kv_heads <= 0 || head_dim <= 0 || head_dim > 256 ||
        num_q_heads % num_kv_heads != 0 || q_pos_base < 0 || sliding_window < 0 ||
        block_rows > total_kv_len || q_pos_base > total_kv_len - block_rows) {
        return (int)cudaErrorInvalidValue;
    }
    const dim3 grid((unsigned)block_rows, (unsigned)num_q_heads);
    const dim3 block(128);
    dflash_attention_f32_kernel<<<grid, block, 0, (cudaStream_t)ex->stream>>>(
        q_in,
        static_cast<const __nv_bfloat16*>(k_cache_bf16),
        static_cast<const __nv_bfloat16*>(v_cache_bf16),
        attn_out,
        block_rows,
        q_pos_base,
        total_kv_len,
        num_q_heads,
        num_kv_heads,
        head_dim,
        sliding_window,
        sm_scale);
    return (int)cudaGetLastError();
}

int w34_dflash_edges_f32(
    const w34_exec* ex,
    float* edges,
    const void* pred_cb_bf16,
    const void* succ_cb_bf16,
    const float* hp,
    const int32_t* cand_ids,
    const float* cand_vals,
    int32_t bonus_id,
    int64_t k,
    int64_t top_k,
    int64_t rank) {
    if (!ex || !edges || !pred_cb_bf16 || !succ_cb_bf16 || !hp || !cand_ids || !cand_vals ||
        k <= 0 || top_k <= 0 || rank <= 0) {
        return (int)cudaErrorInvalidValue;
    }
    const int64_t cells = k * top_k * top_k;
    dflash_edges_f32_kernel<<<(unsigned)cells, 32, 0, (cudaStream_t)ex->stream>>>(
        edges,
        static_cast<const __nv_bfloat16*>(pred_cb_bf16),
        static_cast<const __nv_bfloat16*>(succ_cb_bf16),
        hp,
        cand_ids,
        cand_vals,
        bonus_id,
        k,
        top_k,
        rank);
    return (int)cudaGetLastError();
}

int w34_topk_rows_f16(
    const w34_exec* ex,
    const void* logits_f16,
    int32_t* out_ids,
    float* out_vals,
    int num_rows,
    int local_vocab_size,
    int vocab_offset,
    int top_k) {
    if (!ex || !logits_f16 || !out_ids || !out_vals || num_rows < 0 || local_vocab_size <= 0 || top_k <= 0 || top_k > 16) {
        return (int)cudaErrorInvalidValue;
    }
    if (num_rows == 0) return 0;
    const int threads = 256;
    const size_t smem = (size_t)threads * 16 * (sizeof(float) + sizeof(int32_t));
    topk_rows_f16_kernel<<<(unsigned)num_rows, threads, smem, (cudaStream_t)ex->stream>>>(
        static_cast<const __half*>(logits_f16),
        out_ids,
        out_vals,
        num_rows,
        local_vocab_size,
        vocab_offset,
        top_k);
    return (int)cudaGetLastError();
}

int w34_topk_rows_f32(
    const w34_exec* ex,
    const float* logits_f32,
    int32_t* out_ids,
    float* out_vals,
    int num_rows,
    int local_vocab_size,
    int vocab_offset,
    int top_k) {
    if (!ex || !logits_f32 || !out_ids || !out_vals || num_rows < 0 || local_vocab_size <= 0 || top_k <= 0 || top_k > 16) {
        return (int)cudaErrorInvalidValue;
    }
    if (num_rows == 0) return 0;
    const int threads = 256;
    const size_t smem = (size_t)threads * 16 * (sizeof(float) + sizeof(int32_t));
    topk_rows_f32_kernel<<<(unsigned)num_rows, threads, smem, (cudaStream_t)ex->stream>>>(
        logits_f32,
        out_ids,
        out_vals,
        num_rows,
        local_vocab_size,
        vocab_offset,
        top_k);
    return (int)cudaGetLastError();
}

} // extern "C"
