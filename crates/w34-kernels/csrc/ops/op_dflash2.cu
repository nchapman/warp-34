// Native MTP (FP16) and DFlash2 (BF16) speculative decoding kernels for sm_86 / sm_89.
// Target spine stays FP16; DFlash2 preserves native BF16 weights and activations with
// FP32 reductions. Semantics match SGLang v0.5.19 models/dflash.py and dflash_utils.py.
#include "../w34_cuda.h"

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <float.h>
#include <limits.h>
#include <math.h>
#include <stdint.h>

namespace {

__device__ __forceinline__ float sum256(float v, float* scratch) {
    const int t = threadIdx.x;
    scratch[t] = v;
    __syncthreads();
    for (int s = 128; s; s >>= 1) {
        if (t < s) scratch[t] += scratch[t + s];
        __syncthreads();
    }
    return scratch[0];
}

// Bit-for-bit (0-ULP) identical to a 256-thread sum256 tree reduction when executed
// by a single 32-thread warp (lane = threadIdx.x in 0..31), with 0 shared memory and
// 0 __syncthreads() barriers.
__device__ __forceinline__ float reduce_sum256_exact_warp(
    float v0, float v32, float v64, float v96,
    float v128, float v160, float v192, float v224) {
    float s = ((v0 + v128) + (v64 + v192)) + ((v32 + v160) + (v96 + v224));
#pragma unroll
    for (int mask = 16; mask >= 1; mask >>= 1) {
        s += __shfl_xor_sync(0xffffffffu, s, mask);
    }
    return s;
}

__global__ void concat_cols_2_f16_kernel(
    const __half* __restrict__ src0,
    const __half* __restrict__ src1,
    __half* __restrict__ dst,
    int num_tokens,
    int dim0,
    int dim1) {
    const int tok = blockIdx.y;
    const int col = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = dim0 + dim1;
    if (tok >= num_tokens || col >= total) return;
    if (col < dim0) {
        dst[(int64_t)tok * total + col] = src0[(int64_t)tok * dim0 + col];
    } else {
        dst[(int64_t)tok * total + col] = src1[(int64_t)tok * dim1 + (col - dim0)];
    }
}

__global__ void tap_add_f16_to_bf16_kernel(
    const __half* __restrict__ res_f16,
    const __half* __restrict__ mlp_f16,
    __nv_bfloat16* __restrict__ dst_bf16,
    int num_tokens,
    int hidden_size,
    int dst_row_stride,
    int dst_col_offset) {
    const int tok = blockIdx.y;
    const int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (tok >= num_tokens || col >= hidden_size) return;
    const int64_t src_idx = (int64_t)tok * hidden_size + col;
    const int64_t dst_idx = (int64_t)tok * dst_row_stride + dst_col_offset + col;
    const float sum = __half2float(res_f16[src_idx]) + __half2float(mlp_f16[src_idx]);
    dst_bf16[dst_idx] = __float2bfloat16(sum);
}

union Bf16Vec8 {
    uint4 u;
    __nv_bfloat16 b[8];
};

union HalfVec8 {
    uint4 u;
    __half h[8];
};

// 1-warp (1 row/CTA), compile-time unrolled, register-resident (zero .local stack spill),
// 128-bit vectorized BF16 RMSNorm preserving the exact 32-lane x vec=8 FP32 reduction tree
// of FlashInfer's <=3072-width norm while spreading `rows` across `rows` SMs.
template <int ROUNDS>
__global__ void norm_bf16_vec8_1warp(
    const __nv_bfloat16* __restrict__ x,
    __nv_bfloat16* __restrict__ residual,
    const __nv_bfloat16* __restrict__ weight,
    __nv_bfloat16* __restrict__ y,
    int rows,
    float eps) {
    constexpr int HIDDEN = ROUNDS << 8;
    const int lane = threadIdx.x & 31;
    const int row = (int)blockIdx.x;
    if (row >= rows) return;

    const int64_t base = (int64_t)row * HIDDEN;
    float vals[ROUNDS][8];
    float sq = 0.0f;

#pragma unroll
    for (int i = 0; i < ROUNDS; ++i) {
        const int col0 = ((i << 5) + lane) << 3;
        Bf16Vec8 xv;
        xv.u = *reinterpret_cast<const uint4*>(x + base + col0);
        if (residual) {
            Bf16Vec8 rv;
            rv.u = *reinterpret_cast<const uint4*>(residual + base + col0);
#pragma unroll
            for (int j = 0; j < 8; ++j) {
                const float v = (float)xv.b[j] + (float)rv.b[j];
                vals[i][j] = v;
                sq += v * v;
            }
        } else {
#pragma unroll
            for (int j = 0; j < 8; ++j) {
                const float v = (float)xv.b[j];
                vals[i][j] = v;
                sq += v * v;
            }
        }
    }

#pragma unroll
    for (int offset = 1; offset < 32; offset <<= 1) {
        sq += __shfl_xor_sync(0xffffffffu, sq, offset);
    }

    float inv;
    const float var_eps = sq / (float)HIDDEN + eps;
    asm volatile("rsqrt.approx.ftz.f32 %0, %1;" : "=f"(inv) : "f"(var_eps));

#pragma unroll
    for (int i = 0; i < ROUNDS; ++i) {
        const int col0 = ((i << 5) + lane) << 3;
        Bf16Vec8 wv;
        wv.u = *reinterpret_cast<const uint4*>(weight + col0);
        if (residual) {
            Bf16Vec8 rv_out;
#pragma unroll
            for (int j = 0; j < 8; ++j) {
                rv_out.b[j] = __float2bfloat16(vals[i][j]);
            }
            *reinterpret_cast<uint4*>(residual + base + col0) = rv_out.u;
        }
        Bf16Vec8 yv;
#pragma unroll
        for (int j = 0; j < 8; ++j) {
            yv.b[j] = __float2bfloat16(vals[i][j] * inv * (float)wv.b[j]);
        }
        *reinterpret_cast<uint4*>(y + base + col0) = yv.u;
    }
}

// 4-warp (4 rows/CTA), compile-time unrolled, zero-stack register-resident 128-bit vectorized
// BF16 RMSNorm preserving the exact 32-lane x vec=8 FP32 reduction tree of FlashInfer's <=3072-width norm.
template <int ROUNDS>
__global__ void __launch_bounds__(128) norm_bf16_vec8_4warp(
    const __nv_bfloat16* __restrict__ x,
    __nv_bfloat16* __restrict__ residual,
    const __nv_bfloat16* __restrict__ weight,
    __nv_bfloat16* __restrict__ y,
    int rows,
    float eps) {
    constexpr int HIDDEN = ROUNDS << 8;
    const int wid = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const int row = (int)blockIdx.x * 4 + wid;
    if (row >= rows) return;

    const int64_t base = (int64_t)row * HIDDEN;
    float vals[ROUNDS][8];
    float sq = 0.0f;

#pragma unroll
    for (int i = 0; i < ROUNDS; ++i) {
        const int col0 = ((i << 5) + lane) << 3;
        Bf16Vec8 xv;
        xv.u = *reinterpret_cast<const uint4*>(x + base + col0);
        if (residual) {
            Bf16Vec8 rv;
            rv.u = *reinterpret_cast<const uint4*>(residual + base + col0);
#pragma unroll
            for (int j = 0; j < 8; ++j) {
                const float v = (float)xv.b[j] + (float)rv.b[j];
                vals[i][j] = v;
                sq += v * v;
            }
        } else {
#pragma unroll
            for (int j = 0; j < 8; ++j) {
                const float v = (float)xv.b[j];
                vals[i][j] = v;
                sq += v * v;
            }
        }
    }

#pragma unroll
    for (int offset = 1; offset < 32; offset <<= 1) {
        sq += __shfl_xor_sync(0xffffffffu, sq, offset);
    }

    float inv;
    const float var_eps = sq / (float)HIDDEN + eps;
    asm volatile("rsqrt.approx.ftz.f32 %0, %1;" : "=f"(inv) : "f"(var_eps));

#pragma unroll
    for (int i = 0; i < ROUNDS; ++i) {
        const int col0 = ((i << 5) + lane) << 3;
        Bf16Vec8 wv;
        wv.u = *reinterpret_cast<const uint4*>(weight + col0);
        if (residual) {
            Bf16Vec8 rv_out;
#pragma unroll
            for (int j = 0; j < 8; ++j) {
                rv_out.b[j] = __float2bfloat16(vals[i][j]);
            }
            *reinterpret_cast<uint4*>(residual + base + col0) = rv_out.u;
        }
        Bf16Vec8 yv;
#pragma unroll
        for (int j = 0; j < 8; ++j) {
            yv.b[j] = __float2bfloat16(vals[i][j] * inv * (float)wv.b[j]);
        }
        *reinterpret_cast<uint4*>(y + base + col0) = yv.u;
    }
}

__global__ void norm_bf16(
    const __nv_bfloat16* __restrict__ x,
    __nv_bfloat16* __restrict__ residual,
    const __nv_bfloat16* __restrict__ weight,
    __nv_bfloat16* __restrict__ y,
    int hidden,
    float eps) {
    const int64_t base = (int64_t)blockIdx.x * hidden;
    __shared__ float scratch[32];
    const int tid = threadIdx.x;
    const int num_threads = blockDim.x;
    const int num_warps = num_threads >> 5;
    const int vec = (hidden % 8 == 0) ? 8 : 1;
    const int rounds = (hidden + vec * num_threads - 1) / (vec * num_threads);
    float sq = 0.0f;
    for (int i = 0; i < rounds; ++i) {
        const int col0 = (i * num_threads + tid) * vec;
        for (int j = 0; j < vec; ++j) {
            const int c = col0 + j;
            if (c < hidden) {
                float v = (float)x[base + c];
                if (residual) v += (float)residual[base + c];
                sq += v * v;
            }
        }
    }
    for (int offset = 1; offset < 32; offset <<= 1) {
        sq += __shfl_xor_sync(0xffffffffu, sq, offset);
    }
    if ((tid & 31) == 0) scratch[tid >> 5] = sq;
    __syncthreads();
    if (tid < 32) {
        sq = (tid < num_warps) ? scratch[tid] : 0.0f;
        for (int offset = 1; offset < 32; offset <<= 1) {
            sq += __shfl_xor_sync(0xffffffffu, sq, offset);
        }
        if (tid == 0) scratch[0] = sq;
    }
    __syncthreads();
    float inv;
    const float var_eps = scratch[0] / (float)hidden + eps;
    asm volatile("rsqrt.approx.ftz.f32 %0, %1;" : "=f"(inv) : "f"(var_eps));
    for (int c = tid; c < hidden; c += num_threads) {
        float v = (float)x[base + c];
        if (residual) {
            v += (float)residual[base + c];
            residual[base + c] = __float2bfloat16(v);
        }
        y[base + c] = __float2bfloat16(v * inv * (float)weight[c]);
    }
}

__global__ void silu_mul_bf16(
    const __nv_bfloat16* __restrict__ gu,
    __nv_bfloat16* __restrict__ y,
    int n,
    int hidden) {
    const int i = (int)blockIdx.x * 256 + (int)threadIdx.x;
    if (i >= n) return;
    const int row = i / hidden;
    const int col = i - row * hidden;
    const int64_t base = (int64_t)row * (2 * hidden);
    const float g = (float)gu[base + col];
    const float u = (float)gu[base + hidden + col];
    y[i] = __float2bfloat16((g / (1.0f + expf(-g))) * u);
}

__global__ void grouped_conv_bf16(
    const __nv_bfloat16* __restrict__ x,
    const __nv_bfloat16* __restrict__ delta,
    const __nv_bfloat16* __restrict__ base,
    __nv_bfloat16* __restrict__ y,
    int rows,
    int hidden,
    int block,
    int group,
    int taps,
    int side) {
    const int i = (int)blockIdx.x * 256 + (int)threadIdx.x;
    if (i >= rows * hidden) return;
    const int row = i / hidden;
    const int col = i - row * hidden;
    const int groups = hidden / group;
    const int grp = col / group;
    const int row_in_blk = row % block;
    float acc = 0.0f;
    for (int t = 0; t < taps && t <= row_in_blk; ++t) {
        const float b = (float)base[((int64_t)side * taps + t) * hidden + col];
        const float d = (float)delta[((int64_t)row * 2 * taps + side * taps + t) * groups + grp];
        const float coeff = __fadd_rn(b, d);
        const float xv = (float)x[(int64_t)(row - t) * hidden + col];
        acc = __fadd_rn(acc, __fmul_rn(coeff, xv));
    }
    y[i] = __float2bfloat16(acc);
}

// 1-warp (32-thread) QK RMSNorm + RoPE for dim <= 256.
// When q_out != nullptr and head < q_heads, writes rotated Q directly to contiguous
// q_out [rows, q_heads * dim]; otherwise writes in-place into qkv.
// When write_slot_mapping != nullptr and head >= q_heads, writes rotated K and
// copies V directly into the BF16 paged KV cache described by kv_layout.
template <typename TableT>
__global__ void qk_norm_rope_bf16_warp(
    __nv_bfloat16* __restrict__ qkv,
    __nv_bfloat16* __restrict__ q_out,
    const int32_t* __restrict__ positions,
    const __nv_bfloat16* __restrict__ qw,
    const __nv_bfloat16* __restrict__ kw,
    const TableT* __restrict__ table,
    int q_heads,
    int kv_heads,
    int dim,
    int table_rows,
    float eps,
    w34_kv_layer_layout kv_layout,
    const int32_t* __restrict__ write_slot_mapping) {
    const int row = blockIdx.x;
    const int head = blockIdx.y;
    const int half = dim >> 1;
    const int pos = positions[row];
    if (pos < 0 || pos >= table_rows) return;

    const int64_t offset = (int64_t)row * (q_heads + 2 * kv_heads) * dim + (int64_t)head * dim;
    const __nv_bfloat16* w = (head < q_heads) ? qw : kw;
    const int lane = threadIdx.x;

    float x_vals[8];
    float sq_vals[8];
#pragma unroll
    for (int k = 0; k < 8; ++k) {
        const int c = lane + k * 32;
        if (c < dim) {
            const float x = (float)qkv[offset + c];
            x_vals[k] = x;
            sq_vals[k] = x * x;
        } else {
            x_vals[k] = 0.0f;
            sq_vals[k] = 0.0f;
        }
    }

    const float sum_sq = reduce_sum256_exact_warp(
        sq_vals[0], sq_vals[1], sq_vals[2], sq_vals[3],
        sq_vals[4], sq_vals[5], sq_vals[6], sq_vals[7]);
    const float inv = rsqrtf(sum_sq / (float)dim + eps);

    __nv_bfloat16* dst = (q_out != nullptr && head < q_heads)
        ? (q_out + ((int64_t)row * q_heads + head) * dim)
        : (qkv + offset);
    __nv_bfloat16* v_dst = nullptr;
    const __nv_bfloat16* v_src = nullptr;
    if (write_slot_mapping != nullptr && head >= q_heads) {
        const int slot = write_slot_mapping[row];
        if (slot >= 0) {
            const int kv_head = head - q_heads;
            const int page = slot / kv_layout.block_size;
            const int off = slot % kv_layout.block_size;
            dst = (__nv_bfloat16*)((uint8_t*)(uintptr_t)kv_layout.k_base +
                (size_t)page * kv_layout.k_page_stride_bytes +
                (size_t)kv_head * kv_layout.k_head_stride_bytes +
                (size_t)off * kv_layout.k_token_stride_bytes);
            v_dst = (__nv_bfloat16*)((uint8_t*)(uintptr_t)kv_layout.v_base +
                (size_t)page * kv_layout.v_page_stride_bytes +
                (size_t)kv_head * kv_layout.v_head_stride_bytes +
                (size_t)off * kv_layout.v_token_stride_bytes);
            v_src = qkv + offset + (int64_t)kv_heads * dim;
        }
    }
    const TableT* row_table = table + (int64_t)pos * dim;

    for (int c = lane; c < half; c += 32) {
        const float a = (float)__float2bfloat16((float)qkv[offset + c] * inv * (float)w[c]);
        const float b = (float)__float2bfloat16((float)qkv[offset + half + c] * inv * (float)w[half + c]);
        const float cosine = (float)row_table[c];
        const float sine = (float)row_table[half + c];
        dst[c] = __float2bfloat16(a * cosine - b * sine);
        dst[half + c] = __float2bfloat16(b * cosine + a * sine);
        if (v_dst != nullptr) {
            v_dst[c] = v_src[c];
            v_dst[half + c] = v_src[half + c];
        }
    }
}

__global__ void copy_cols_bf16(
    const __nv_bfloat16* __restrict__ src,
    __nv_bfloat16* __restrict__ dst,
    int rows,
    int src_stride,
    int dst_stride,
    int src_col,
    int dst_col,
    int cols) {
    const int i = (int)blockIdx.x * 256 + (int)threadIdx.x;
    if (i < rows * cols) {
        const int row = i / cols;
        const int col = i - row * cols;
        dst[(int64_t)row * dst_stride + dst_col + col] =
            src[(int64_t)row * src_stride + src_col + col];
    }
}

__global__ void embed_bf16(
    const int32_t* __restrict__ ids,
    const __half* __restrict__ weight,
    __nv_bfloat16* __restrict__ out,
    int rows,
    int hidden,
    int vocab) {
    const int i = (int)blockIdx.x * 256 + (int)threadIdx.x;
    if (i >= rows * hidden) return;
    const int row = i / hidden;
    const int col = i - row * hidden;
    const int token = ids[row];
    out[i] = __float2bfloat16(
        (token >= 0 && token < vocab)
            ? __half2float(weight[(int64_t)token * hidden + col])
            : 0.0f);
}

__global__ void bf16_to_f16(
    const __nv_bfloat16* __restrict__ x,
    __half* __restrict__ y,
    int64_t n) {
    const int64_t i = (int64_t)blockIdx.x * 256 + threadIdx.x;
    if (i < n) y[i] = __float2half((float)x[i]);
}

// 1-warp (32-thread) DFlash2 bilinear edge scoring when rank <= 256 (0-ULP identical to sum256).
__global__ void edges_bf16_warp(
    const int32_t* __restrict__ ids,
    const float* __restrict__ unary,
    const __nv_bfloat16* __restrict__ hidden,
    const __nv_bfloat16* __restrict__ pred,
    const __nv_bfloat16* __restrict__ succ,
    float* __restrict__ edges,
    int anchor,
    int k,
    int topk,
    int rank,
    int vocab) {
    const int i = blockIdx.x;
    const int topk2 = topk * topk;
    const int slot = i / topk2;
    const int rem = i - slot * topk2;
    const int p = rem / topk;
    const int c = rem - p * topk;
    const int pred_id = (slot == 0) ? anchor : ids[(slot - 1) * topk + p];
    const int succ_id = ids[slot * topk + c];
    if (pred_id < 0 || pred_id >= vocab || succ_id < 0 || succ_id >= vocab) {
        if (threadIdx.x == 0) edges[i] = -INFINITY;
        return;
    }

    const int lane = threadIdx.x;
    const __nv_bfloat16* pred_row = pred + (int64_t)pred_id * rank;
    const __nv_bfloat16* hidden_row = hidden + (int64_t)slot * rank;
    const __nv_bfloat16* succ_row = succ + (int64_t)succ_id * rank;

    float prod[8];
#pragma unroll
    for (int m = 0; m < 8; ++m) {
        const int r = lane + m * 32;
        if (r < rank) {
            const float ph = (float)__float2bfloat16((float)pred_row[r] * (float)hidden_row[r]);
            prod[m] = ph * (float)succ_row[r];
        } else {
            prod[m] = 0.0f;
        }
    }

    const float sum = reduce_sum256_exact_warp(
        prod[0], prod[1], prod[2], prod[3],
        prod[4], prod[5], prod[6], prod[7]);
    if (lane == 0) {
        edges[i] = (float)__float2bfloat16(sum) + unary[slot * topk + c];
    }
}

// Single-CTA (32 * topk threads) greedy path walk over DFlash2 candidates when rank <= 256.
// Evaluates only the active predecessor row at each step slot = 0..k-1 with the exact same
// 0-ULP warp reduction and tie-break order as edges_bf16_warp + host select_greedy.
__global__ void dflash_select_greedy_bf16_kernel(
    const int32_t* __restrict__ ids,
    const float* __restrict__ unary,
    const __nv_bfloat16* __restrict__ hidden,
    const __nv_bfloat16* __restrict__ pred,
    const __nv_bfloat16* __restrict__ succ,
    int32_t* __restrict__ out_path,
    const int32_t* __restrict__ anchor_dev,
    int anchor,
    int k,
    int topk,
    int rank,
    int vocab) {
    __shared__ float s_scores[32];
    __shared__ int32_t s_pred_id;
    __shared__ int s_invalid;

    const int lane = threadIdx.x & 31;
    const int c = threadIdx.x >> 5;

    if (threadIdx.x == 0) {
        const int anc = anchor_dev ? anchor_dev[0] : anchor;
        s_pred_id = anc;
        s_invalid = (anc < 0 || anc >= vocab) ? 1 : 0;
    }
    __syncthreads();
    if (s_invalid) {
        if (threadIdx.x == 0) out_path[0] = -1;
        return;
    }

    for (int slot = 0; slot < k; ++slot) {
        if (c < topk) {
            const int32_t pred_id = s_pred_id;
            const int32_t succ_id = ids[slot * topk + c];
            const float u_val = unary[slot * topk + c];
            if (pred_id < 0 || pred_id >= vocab || succ_id < 0 || succ_id >= vocab || !isfinite(u_val)) {
                if (lane == 0) {
                    s_scores[c] = -INFINITY;
                    atomicExch(&s_invalid, 1);
                }
            } else {
                const __nv_bfloat16* pred_row = pred + (int64_t)pred_id * rank;
                const __nv_bfloat16* hidden_row = hidden + (int64_t)slot * rank;
                const __nv_bfloat16* succ_row = succ + (int64_t)succ_id * rank;

                float prod[8];
#pragma unroll
                for (int m = 0; m < 8; ++m) {
                    const int r = lane + m * 32;
                    if (r < rank) {
                        const float ph = (float)__float2bfloat16((float)pred_row[r] * (float)hidden_row[r]);
                        prod[m] = ph * (float)succ_row[r];
                    } else {
                        prod[m] = 0.0f;
                    }
                }
                const float sum = reduce_sum256_exact_warp(
                    prod[0], prod[1], prod[2], prod[3],
                    prod[4], prod[5], prod[6], prod[7]);
                if (lane == 0) {
                    const float edge_score = (float)__float2bfloat16(sum) + u_val;
                    s_scores[c] = edge_score;
                    if (!isfinite(edge_score)) {
                        atomicExch(&s_invalid, 1);
                    }
                }
            }
        }
        __syncthreads();
        if (s_invalid) {
            if (threadIdx.x == 0) out_path[0] = -1;
            return;
        }
        if (threadIdx.x == 0) {
            int winner = 0;
            for (int j = 1; j < topk; ++j) {
                if (s_scores[j] > s_scores[winner]) {
                    winner = j;
                }
            }
            const int32_t win_id = ids[slot * topk + winner];
            out_path[slot] = win_id;
            s_pred_id = win_id;
        }
        __syncthreads();
    }
}

__global__ void edges_bf16_generic(
    const int32_t* __restrict__ ids,
    const float* __restrict__ unary,
    const __nv_bfloat16* __restrict__ hidden,
    const __nv_bfloat16* __restrict__ pred,
    const __nv_bfloat16* __restrict__ succ,
    float* __restrict__ edges,
    int anchor,
    int k,
    int topk,
    int rank,
    int vocab) {
    const int i = blockIdx.x;
    const int slot = i / (topk * topk);
    const int p = (i / topk) % topk;
    const int c = i % topk;
    const int pred_id = (slot == 0) ? anchor : ids[(slot - 1) * topk + p];
    const int succ_id = ids[slot * topk + c];
    if (pred_id < 0 || pred_id >= vocab || succ_id < 0 || succ_id >= vocab) {
        if (threadIdx.x == 0) edges[i] = -INFINITY;
        return;
    }
    __shared__ float scratch[256];
    float sum = 0.0f;
    for (int r = threadIdx.x; r < rank; r += 256) {
        const float ph = (float)__float2bfloat16(
            (float)pred[(int64_t)pred_id * rank + r] *
            (float)hidden[(int64_t)slot * rank + r]);
        sum += ph * (float)succ[(int64_t)succ_id * rank + r];
    }
    sum = sum256(sum, scratch);
    if (threadIdx.x == 0) {
        edges[i] = (float)__float2bfloat16(sum) + unary[slot * topk + c];
    }
}

struct TopkPair {
    float val;
    int32_t id;
};

__device__ __forceinline__ TopkPair better_topk_pair(TopkPair a, TopkPair b) {
    if (b.val > a.val || (b.val == a.val && b.id >= 0 && (a.id < 0 || b.id < a.id))) {
        return b;
    }
    return a;
}

__device__ __forceinline__ TopkPair warp_reduce_topk_pair(TopkPair v) {
#pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
        TopkPair o;
        o.val = __shfl_xor_sync(0xffffffffu, v.val, off);
        o.id = __shfl_xor_sync(0xffffffffu, v.id, off);
        v = better_topk_pair(v, o);
    }
    return v;
}

__device__ __forceinline__ void topk_insert_smem(
    float v,
    int32_t id,
    int tid,
    int last,
    float (&sv)[16][256],
    int32_t (&si)[16][256],
    float& min_top) {
    if (v > min_top) {
        int pos = last;
        while (pos > 0 && (v > sv[pos - 1][tid] || (v == sv[pos - 1][tid] && id < si[pos - 1][tid]))) {
            sv[pos][tid] = sv[pos - 1][tid];
            si[pos][tid] = si[pos - 1][tid];
            --pos;
        }
        sv[pos][tid] = v;
        si[pos][tid] = id;
        min_top = sv[last][tid];
    }
}

// Encodes an FP16 `__half` bit pattern + local column index `local_col < 65535` into a
// strictly monotonic `uint32_t` key such that `key_a > key_b` matches
// `(val_a > val_b) || (val_a == val_b && id_a < id_b)` across all 65,536 FP16 bit patterns:
// - `NaN` (`(u & 0x7fff) > 0x7c00`) and `-Inf` (`u == 0xfc00`) map to `0u` (never > `min_top >= 0u`,
//   matching IEEE `v > -FLT_MAX == false` in `topk_insert_smem`).
// - `-0.0` (`u == 0x8000`) normalizes to `+0.0` (`0x0000`) so signed-zero ties break by lowest ID.
// - Finite negatives (`0x8001..0xfbff`) map to `ord = (~u) & 0x7fff` in `0x0400..0x7ffe`.
// - `+0.0`, finite positives, and `+Inf` (`0x0000..0x7c00`) map to `ord = u | 0x8000` in `0x8000..0xfc00`.
__device__ __forceinline__ uint32_t pack_f16_topk_key(uint16_t u, int local_col) {
    if ((u & 0x7fffu) > 0x7c00u || u == 0xfc00u) return 0u;
    if (u == 0x8000u) u = 0x0000u;
    const uint32_t ord = ((u & 0x8000u) == 0u)
        ? (uint32_t)(u | 0x8000u)
        : (uint32_t)((~u) & 0x7fffu);
    return (ord << 16) | (uint32_t)(0xffffu - (uint32_t)local_col);
}

__device__ __forceinline__ float unpack_f16_topk_val(uint32_t key) {
    const uint32_t ord = key >> 16;
    if (ord == 0u) return -FLT_MAX;
    const uint16_t u = (ord >= 0x8000u)
        ? (uint16_t)(ord & 0x7fffu)
        : (uint16_t)((~ord) | 0x8000u);
    return __half2float(__ushort_as_half(u));
}

__device__ __forceinline__ void topk_insert_packed_smem(
    uint32_t key,
    int tid,
    int last,
    uint32_t (&sk)[16][128],
    uint32_t& min_top) {
    if (key > min_top) {
        int pos = last;
        while (pos > 0 && key > sk[pos - 1][tid]) {
            sk[pos][tid] = sk[pos - 1][tid];
            --pos;
        }
        sk[pos][tid] = key;
        min_top = sk[last][tid];
    }
}

__device__ __forceinline__ uint32_t warp_reduce_max_u32(uint32_t v) {
#pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
        const uint32_t o = __shfl_xor_sync(0xffffffffu, v, off);
        if (o > v) v = o;
    }
    return v;
}

// High-occupancy (8 KB smem, 128 threads = 4 warps, 5-6 CTAs/SM) packed-u32 top-k extraction
// for splits with `chunk_elems <= 65535` (covers all `num_splits >= 4` up to 1M local vocab).
__global__ void topk_rows_packed_u32_f16_kernel(
    const __half* __restrict__ logits,
    int32_t* __restrict__ out_ids,
    float* __restrict__ out_vals,
    int num_rows,
    int local_vocab_size,
    int vocab_offset,
    int top_k,
    int num_splits) {
    constexpr int MAX_K = 16;
    const int split_idx = blockIdx.x;
    const int row = blockIdx.y;
    if (row >= num_rows || split_idx >= num_splits) return;
    const __half* row_logits = logits + (int64_t)row * local_vocab_size;
    const int tid = threadIdx.x;

    const int chunk_elems = ((local_vocab_size + num_splits - 1) / num_splits + 7) & ~7;
    const int col_start = split_idx * chunk_elems;
    const int col_end = min(local_vocab_size, col_start + chunk_elems);
    const int span = col_end - col_start;

    __shared__ uint32_t sk[MAX_K][128];
    __shared__ uint32_t warp_k[4];
    __shared__ uint32_t s_win_k;

#pragma unroll
    for (int m = 0; m < MAX_K; ++m) {
        sk[m][tid] = 0u;
    }

    uint32_t min_top = 0u;
    const int last = top_k - 1;
    const uint16_t* split_u16 = reinterpret_cast<const uint16_t*>(row_logits + col_start);
    const bool aligned16 = (((uintptr_t)split_u16) & 15u) == 0u;
    const int lane = tid & 31;
    const int wid = tid >> 5;
    if (aligned16 && span > 0) {
        const int vec_end = span & ~7;
        // Pre-seed CTA-wide `top_k`-th largest threshold from the first 1,024 tokens so >98%
        // of remaining `uint4` loads reject all 8 keys in registers via a single `max8 <= min_top`.
        int start_lc = (tid << 3);
        if (vec_end >= 1024) {
            const uint4 u4 = *reinterpret_cast<const uint4*>(split_u16 + start_lc);
            const uint32_t k0 = pack_f16_topk_key((uint16_t)(u4.x & 0xffffu), start_lc + 0);
            const uint32_t k1 = pack_f16_topk_key((uint16_t)(u4.x >> 16), start_lc + 1);
            const uint32_t k2 = pack_f16_topk_key((uint16_t)(u4.y & 0xffffu), start_lc + 2);
            const uint32_t k3 = pack_f16_topk_key((uint16_t)(u4.y >> 16), start_lc + 3);
            const uint32_t k4 = pack_f16_topk_key((uint16_t)(u4.z & 0xffffu), start_lc + 4);
            const uint32_t k5 = pack_f16_topk_key((uint16_t)(u4.z >> 16), start_lc + 5);
            const uint32_t k6 = pack_f16_topk_key((uint16_t)(u4.w & 0xffffu), start_lc + 6);
            const uint32_t k7 = pack_f16_topk_key((uint16_t)(u4.w >> 16), start_lc + 7);
            topk_insert_packed_smem(k0, tid, last, sk, min_top);
            topk_insert_packed_smem(k1, tid, last, sk, min_top);
            topk_insert_packed_smem(k2, tid, last, sk, min_top);
            topk_insert_packed_smem(k3, tid, last, sk, min_top);
            topk_insert_packed_smem(k4, tid, last, sk, min_top);
            topk_insert_packed_smem(k5, tid, last, sk, min_top);
            topk_insert_packed_smem(k6, tid, last, sk, min_top);
            topk_insert_packed_smem(k7, tid, last, sk, min_top);
            __syncthreads();
            int seed_cur = 0;
            uint32_t seed_kth = 0u;
            for (int m = 0; m < top_k; ++m) {
                uint32_t cand = (seed_cur < top_k) ? sk[seed_cur][tid] : 0u;
                cand = warp_reduce_max_u32(cand);
                if (lane == 0) warp_k[wid] = cand;
                __syncthreads();
                if (wid == 0) {
                    uint32_t w = (lane < 4) ? warp_k[lane] : 0u;
                    w = warp_reduce_max_u32(w);
                    if (lane == 0) s_win_k = w;
                }
                __syncthreads();
                seed_kth = s_win_k;
                if (seed_cur < top_k && seed_kth != 0u && sk[seed_cur][tid] == seed_kth) {
                    ++seed_cur;
                }
            }
            if (seed_kth > min_top) min_top = seed_kth;
            start_lc += 1024;
        }
        for (int lc = start_lc; lc < vec_end; lc += 128 * 8) {
            const uint4 u4 = *reinterpret_cast<const uint4*>(split_u16 + lc);
            const uint32_t k0 = pack_f16_topk_key((uint16_t)(u4.x & 0xffffu), lc + 0);
            const uint32_t k1 = pack_f16_topk_key((uint16_t)(u4.x >> 16), lc + 1);
            const uint32_t k2 = pack_f16_topk_key((uint16_t)(u4.y & 0xffffu), lc + 2);
            const uint32_t k3 = pack_f16_topk_key((uint16_t)(u4.y >> 16), lc + 3);
            const uint32_t k4 = pack_f16_topk_key((uint16_t)(u4.z & 0xffffu), lc + 4);
            const uint32_t k5 = pack_f16_topk_key((uint16_t)(u4.z >> 16), lc + 5);
            const uint32_t k6 = pack_f16_topk_key((uint16_t)(u4.w & 0xffffu), lc + 6);
            const uint32_t k7 = pack_f16_topk_key((uint16_t)(u4.w >> 16), lc + 7);
            const uint32_t m01 = (k0 > k1) ? k0 : k1;
            const uint32_t m23 = (k2 > k3) ? k2 : k3;
            const uint32_t m45 = (k4 > k5) ? k4 : k5;
            const uint32_t m67 = (k6 > k7) ? k6 : k7;
            const uint32_t m03 = (m01 > m23) ? m01 : m23;
            const uint32_t m47 = (m45 > m67) ? m45 : m67;
            const uint32_t max8 = (m03 > m47) ? m03 : m47;
            if (max8 > min_top) {
                topk_insert_packed_smem(k0, tid, last, sk, min_top);
                topk_insert_packed_smem(k1, tid, last, sk, min_top);
                topk_insert_packed_smem(k2, tid, last, sk, min_top);
                topk_insert_packed_smem(k3, tid, last, sk, min_top);
                topk_insert_packed_smem(k4, tid, last, sk, min_top);
                topk_insert_packed_smem(k5, tid, last, sk, min_top);
                topk_insert_packed_smem(k6, tid, last, sk, min_top);
                topk_insert_packed_smem(k7, tid, last, sk, min_top);
            }
        }
        for (int lc = vec_end + tid; lc < span; lc += 128) {
            topk_insert_packed_smem(pack_f16_topk_key(split_u16[lc], lc), tid, last, sk, min_top);
        }
    } else {
        for (int lc = tid; lc < span; lc += 128) {
            topk_insert_packed_smem(pack_f16_topk_key(split_u16[lc], lc), tid, last, sk, min_top);
        }
    }
    __syncthreads();

    int cur = 0;
    const int64_t out_base = ((int64_t)row * num_splits + split_idx) * top_k;
    for (int m = 0; m < top_k; ++m) {
        uint32_t cand = (cur < top_k) ? sk[cur][tid] : 0u;
        cand = warp_reduce_max_u32(cand);
        if (lane == 0) {
            warp_k[wid] = cand;
        }
        __syncthreads();

        if (wid == 0) {
            uint32_t w = (lane < 4) ? warp_k[lane] : 0u;
            w = warp_reduce_max_u32(w);
            if (lane == 0) {
                s_win_k = w;
                if (w != 0u) {
                    const int local_col = (int)(0xffffu - (w & 0xffffu));
                    out_vals[out_base + m] = unpack_f16_topk_val(w);
                    out_ids[out_base + m] = vocab_offset + col_start + local_col;
                } else {
                    out_vals[out_base + m] = -FLT_MAX;
                    out_ids[out_base + m] = -1;
                }
            }
        }
        __syncthreads();

        const uint32_t win_k = s_win_k;
        if (cur < top_k && win_k != 0u && sk[cur][tid] == win_k) {
            ++cur;
        }
    }
}

// Bank-conflict-free, spill-free top-k extraction (top_k <= 16) over [col_start, col_end).
// Uses 128-bit uint4 (8x FP16) vectorized global loads when 16-byte aligned.
__global__ void topk_rows_range_f16_kernel(
    const __half* __restrict__ logits,
    int32_t* __restrict__ out_ids,
    float* __restrict__ out_vals,
    int num_rows,
    int local_vocab_size,
    int vocab_offset,
    int top_k,
    int num_splits) {
    constexpr int MAX_K = 16;
    const int split_idx = blockIdx.x;
    const int row = blockIdx.y;
    if (row >= num_rows || split_idx >= num_splits) return;
    const __half* row_logits = logits + (int64_t)row * local_vocab_size;
    const int tid = threadIdx.x;

    const int chunk_elems = ((local_vocab_size + num_splits - 1) / num_splits + 7) & ~7;
    const int col_start = split_idx * chunk_elems;
    const int col_end = min(local_vocab_size, col_start + chunk_elems);

    __shared__ float sv[MAX_K][256];
    __shared__ int32_t si[MAX_K][256];
    __shared__ float warp_v[8];
    __shared__ int32_t warp_i[8];
    __shared__ int32_t s_win_i;

#pragma unroll
    for (int m = 0; m < MAX_K; ++m) {
        sv[m][tid] = -FLT_MAX;
        si[m][tid] = -1;
    }

    float min_top = -FLT_MAX;
    const int last = top_k - 1;
    const bool aligned16 = (((uintptr_t)(row_logits + col_start)) & 15u) == 0u;
    if (aligned16 && col_start < col_end) {
        const int vec_end = col_start + ((col_end - col_start) & ~7);
        for (int col = col_start + (tid << 3); col < vec_end; col += 256 * 8) {
            HalfVec8 hv;
            hv.u = *reinterpret_cast<const uint4*>(row_logits + col);
#pragma unroll
            for (int j = 0; j < 8; ++j) {
                const float v = __half2float(hv.h[j]);
                topk_insert_smem(v, vocab_offset + col + j, tid, last, sv, si, min_top);
            }
        }
        for (int col = vec_end + tid; col < col_end; col += 256) {
            const float v = __half2float(row_logits[col]);
            topk_insert_smem(v, vocab_offset + col, tid, last, sv, si, min_top);
        }
    } else {
        for (int col = col_start + tid; col < col_end; col += 256) {
            const float v = __half2float(row_logits[col]);
            topk_insert_smem(v, vocab_offset + col, tid, last, sv, si, min_top);
        }
    }
    __syncthreads();

    int cur = 0;
    const int lane = tid & 31;
    const int wid = tid >> 5;
    const int64_t out_base = ((int64_t)row * num_splits + split_idx) * top_k;
    for (int m = 0; m < top_k; ++m) {
        TopkPair cand = (cur < top_k)
            ? TopkPair{sv[cur][tid], si[cur][tid]}
            : TopkPair{-FLT_MAX, -1};
        cand = warp_reduce_topk_pair(cand);
        if (lane == 0) {
            warp_v[wid] = cand.val;
            warp_i[wid] = cand.id;
        }
        __syncthreads();

        if (wid == 0) {
            TopkPair w = (lane < 8)
                ? TopkPair{warp_v[lane], warp_i[lane]}
                : TopkPair{-FLT_MAX, -1};
            w = warp_reduce_topk_pair(w);
            if (lane == 0) {
                s_win_i = w.id;
                out_vals[out_base + m] = w.val;
                out_ids[out_base + m] = w.id;
            }
        }
        __syncthreads();

        const int32_t win_i = s_win_i;
        if (cur < top_k && win_i >= 0 && si[cur][tid] == win_i) {
            ++cur;
        }
    }
}

// Merges `num_splits * top_k <= 256` candidates per row in registers (256 threads/block).
__global__ void topk_rows_merge_kernel(
    const int32_t* __restrict__ split_ids,
    const float* __restrict__ split_vals,
    int32_t* __restrict__ out_ids,
    float* __restrict__ out_vals,
    int num_rows,
    int top_k,
    int total_cands) {
    const int row = blockIdx.x;
    if (row >= num_rows) return;
    const int tid = threadIdx.x;
    const int lane = tid & 31;
    const int wid = tid >> 5;

    __shared__ float warp_v[8];
    __shared__ int32_t warp_i[8];
    __shared__ int32_t s_win_i;

    TopkPair my_cand = (tid < total_cands)
        ? TopkPair{split_vals[(int64_t)row * total_cands + tid], split_ids[(int64_t)row * total_cands + tid]}
        : TopkPair{-FLT_MAX, -1};

    for (int m = 0; m < top_k; ++m) {
        TopkPair w = warp_reduce_topk_pair(my_cand);
        if (lane == 0) {
            warp_v[wid] = w.val;
            warp_i[wid] = w.id;
        }
        __syncthreads();
        if (wid == 0) {
            TopkPair g = (lane < 8) ? TopkPair{warp_v[lane], warp_i[lane]} : TopkPair{-FLT_MAX, -1};
            g = warp_reduce_topk_pair(g);
            if (lane == 0) {
                s_win_i = g.id;
                out_vals[(int64_t)row * top_k + m] = g.val;
                out_ids[(int64_t)row * top_k + m] = g.id;
            }
        }
        __syncthreads();
        const int32_t win_i = s_win_i;
        if (win_i >= 0 && my_cand.id == win_i) {
            my_cand = TopkPair{-FLT_MAX, -1};
        }
    }
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

int w34_tap_add_f16_to_bf16(
    const w34_exec* ex,
    const void* res_f16,
    const void* mlp_f16,
    void* dst_bf16,
    int num_tokens,
    int hidden_size,
    int dst_row_stride,
    int dst_col_offset) {
    if (!ex || !res_f16 || !mlp_f16 || !dst_bf16 || num_tokens < 0 || hidden_size <= 0 ||
        dst_col_offset < 0 || dst_row_stride < hidden_size ||
        dst_col_offset > dst_row_stride - hidden_size) {
        return (int)cudaErrorInvalidValue;
    }
    if (num_tokens == 0) return 0;
    const dim3 block(256);
    const dim3 grid((unsigned)((hidden_size + 255) / 256), (unsigned)num_tokens);
    tap_add_f16_to_bf16_kernel<<<grid, block, 0, (cudaStream_t)ex->stream>>>(
        static_cast<const __half*>(res_f16),
        static_cast<const __half*>(mlp_f16),
        static_cast<__nv_bfloat16*>(dst_bf16),
        num_tokens,
        hidden_size,
        dst_row_stride,
        dst_col_offset);
    return (int)cudaGetLastError();
}

int w34_embed_lookup_f16_to_bf16(
    const w34_exec* ex,
    const int32_t* ids,
    const void* weight,
    void* out,
    int rows,
    int hidden,
    int vocab) {
    if (!ex || !ids || !weight || !out || rows < 0 || hidden <= 0 || vocab <= 0) {
        return (int)cudaErrorInvalidValue;
    }
    const int64_t n = (int64_t)rows * hidden;
    if (n == 0) return 0;
    if (n > INT_MAX) return (int)cudaErrorInvalidValue;
    embed_bf16<<<(unsigned)((n + 255) / 256), 256, 0, (cudaStream_t)ex->stream>>>(
        ids, (const __half*)weight, (__nv_bfloat16*)out, rows, hidden, vocab);
    return (int)cudaGetLastError();
}

int w34_cast_bf16_to_f16(const w34_exec* ex, const void* x, void* y, int64_t n) {
    if (!ex || !x || !y || n < 0) return (int)cudaErrorInvalidValue;
    if (n == 0) return 0;
    bf16_to_f16<<<(unsigned)((n + 255) / 256), 256, 0, (cudaStream_t)ex->stream>>>(
        (const __nv_bfloat16*)x, (__half*)y, n);
    return (int)cudaGetLastError();
}

int w34_dflash_edges_bf16(
    const w34_exec* ex,
    const int32_t* ids,
    const float* unary,
    const void* hidden,
    const void* pred,
    const void* succ,
    float* out,
    int anchor,
    int k,
    int topk,
    int rank,
    int vocab) {
    if (!ex || !ids || !unary || !hidden || !pred || !succ || !out ||
        anchor < 0 || anchor >= vocab || k <= 0 || k > 16 ||
        topk <= 0 || topk > 32 || rank <= 0 || vocab <= 0) {
        return (int)cudaErrorInvalidValue;
    }
    const unsigned cells = (unsigned)(k * topk * topk);
    if (rank <= 256) {
        edges_bf16_warp<<<cells, 32, 0, (cudaStream_t)ex->stream>>>(
            ids, unary, (const __nv_bfloat16*)hidden,
            (const __nv_bfloat16*)pred, (const __nv_bfloat16*)succ,
            out, anchor, k, topk, rank, vocab);
    } else {
        edges_bf16_generic<<<cells, 256, 0, (cudaStream_t)ex->stream>>>(
            ids, unary, (const __nv_bfloat16*)hidden,
            (const __nv_bfloat16*)pred, (const __nv_bfloat16*)succ,
            out, anchor, k, topk, rank, vocab);
    }
    return (int)cudaGetLastError();
}

int w34_rms_norm_bf16(
    const w34_exec* ex,
    const void* x,
    void* residual,
    const void* weight,
    void* y,
    int rows,
    int hidden,
    float eps) {
    if (!ex || !x || !weight || !y || rows < 0 || hidden <= 0 ||
        !(eps > 0.0f) || !isfinite(eps) ||
        (residual && (residual == x || residual == y))) {
        return (int)cudaErrorInvalidValue;
    }
    if (rows == 0) return 0;
    const bool aligned16 =
        (((uintptr_t)x | (uintptr_t)weight | (uintptr_t)y | (uintptr_t)residual) & 15u) == 0u;
    if (aligned16 && hidden == 2560) {
        norm_bf16_vec8_1warp<10><<<(unsigned)rows, 32, 0, (cudaStream_t)ex->stream>>>(
            (const __nv_bfloat16*)x, (__nv_bfloat16*)residual,
            (const __nv_bfloat16*)weight, (__nv_bfloat16*)y,
            rows, eps);
        return (int)cudaGetLastError();
    }
    if (aligned16 && hidden >= 256 && hidden <= 3072 && (hidden & 255) == 0) {
        const unsigned blocks = (unsigned)((rows + 3) / 4);
        const cudaStream_t stream = (cudaStream_t)ex->stream;
        #define LAUNCH_NORM_4WARP(R) \
            norm_bf16_vec8_4warp<(R)><<<blocks, 128, 0, stream>>>( \
                (const __nv_bfloat16*)x, (__nv_bfloat16*)residual, \
                (const __nv_bfloat16*)weight, (__nv_bfloat16*)y, rows, eps)
        switch (hidden >> 8) {
            case 1: LAUNCH_NORM_4WARP(1); break;
            case 2: LAUNCH_NORM_4WARP(2); break;
            case 3: LAUNCH_NORM_4WARP(3); break;
            case 4: LAUNCH_NORM_4WARP(4); break;
            case 5: LAUNCH_NORM_4WARP(5); break;
            case 6: LAUNCH_NORM_4WARP(6); break;
            case 7: LAUNCH_NORM_4WARP(7); break;
            case 8: LAUNCH_NORM_4WARP(8); break;
            case 9: LAUNCH_NORM_4WARP(9); break;
            case 10: LAUNCH_NORM_4WARP(10); break;
            case 11: LAUNCH_NORM_4WARP(11); break;
            default: LAUNCH_NORM_4WARP(12); break;
        }
        #undef LAUNCH_NORM_4WARP
        return (int)cudaGetLastError();
    }
    const int threads = (hidden <= 3072) ? 32 : ((hidden <= 6144) ? 64 : 128);
    norm_bf16<<<(unsigned)rows, threads, 0, (cudaStream_t)ex->stream>>>(
        (const __nv_bfloat16*)x, (__nv_bfloat16*)residual,
        (const __nv_bfloat16*)weight, (__nv_bfloat16*)y, hidden, eps);
    return (int)cudaGetLastError();
}

int w34_silu_and_mul_bf16(
    const w34_exec* ex,
    const void* gu,
    void* y,
    int rows,
    int hidden) {
    if (!ex || !gu || !y || rows < 0 || hidden <= 0 || gu == y) {
        return (int)cudaErrorInvalidValue;
    }
    const int64_t n = (int64_t)rows * hidden;
    if (n == 0) return 0;
    if (n > INT_MAX) return (int)cudaErrorInvalidValue;
    silu_mul_bf16<<<(unsigned)((n + 255) / 256), 256, 0, (cudaStream_t)ex->stream>>>(
        (const __nv_bfloat16*)gu, (__nv_bfloat16*)y, (int)n, hidden);
    return (int)cudaGetLastError();
}

int w34_dflash_conv_bf16(
    const w34_exec* ex,
    const void* x,
    const void* delta,
    const void* base,
    void* y,
    int rows,
    int hidden,
    int block,
    int group,
    int taps,
    int side) {
    if (!ex || !x || !delta || !base || !y || x == y || rows < 0 || hidden <= 0 ||
        block <= 0 || group <= 0 || hidden % group || taps <= 0 || (side != 0 && side != 1)) {
        return (int)cudaErrorInvalidValue;
    }
    const int64_t n = (int64_t)rows * hidden;
    if (n == 0) return 0;
    if (n > INT_MAX) return (int)cudaErrorInvalidValue;
    grouped_conv_bf16<<<(unsigned)((n + 255) / 256), 256, 0, (cudaStream_t)ex->stream>>>(
        (const __nv_bfloat16*)x, (const __nv_bfloat16*)delta,
        (const __nv_bfloat16*)base, (__nv_bfloat16*)y,
        rows, hidden, block, group, taps, side);
    return (int)cudaGetLastError();
}

int w34_dflash_qk_norm_rope_bf16(
    const w34_exec* ex,
    void* qkv,
    const int32_t* positions,
    const void* qw,
    const void* kw,
    const void* table,
    int rows,
    int q_heads,
    int kv_heads,
    int dim,
    int table_rows,
    float eps) {
    if (!ex || !qkv || !positions || !qw || !kw || !table || rows < 0 ||
        q_heads < 0 || kv_heads <= 0 || q_heads % kv_heads ||
        dim <= 0 || dim > 256 || dim % 2 || table_rows <= 0 ||
        !(eps > 0.0f) || !isfinite(eps)) {
        return (int)cudaErrorInvalidValue;
    }
    if (rows == 0) return 0;
    w34_kv_layer_layout empty_layout = {};
    qk_norm_rope_bf16_warp<<<dim3((unsigned)rows, (unsigned)(q_heads + kv_heads)), 32, 0, (cudaStream_t)ex->stream>>>(
        (__nv_bfloat16*)qkv, nullptr, positions,
        (const __nv_bfloat16*)qw, (const __nv_bfloat16*)kw,
        (const __nv_bfloat16*)table, q_heads, kv_heads, dim, table_rows, eps,
        empty_layout, nullptr);
    return (int)cudaGetLastError();
}

int w34_dflash_qk_norm_rope_bf16_f32_table_qout(
    const w34_exec* ex,
    void* qkv,
    void* q_out,
    const int32_t* positions,
    const void* qw,
    const void* kw,
    const void* table,
    int rows,
    int q_heads,
    int kv_heads,
    int dim,
    int table_rows,
    float eps) {
    if (!ex || !qkv || !positions || !qw || !kw || !table || rows < 0 ||
        q_heads < 0 || kv_heads <= 0 || q_heads % kv_heads ||
        dim <= 0 || dim > 256 || dim % 2 || table_rows <= 0 ||
        !(eps > 0.0f) || !isfinite(eps) || q_out == qkv) {
        return (int)cudaErrorInvalidValue;
    }
    if (rows == 0) return 0;
    w34_kv_layer_layout empty_layout = {};
    qk_norm_rope_bf16_warp<<<dim3((unsigned)rows, (unsigned)(q_heads + kv_heads)), 32, 0, (cudaStream_t)ex->stream>>>(
        (__nv_bfloat16*)qkv, (__nv_bfloat16*)q_out, positions,
        (const __nv_bfloat16*)qw, (const __nv_bfloat16*)kw,
        (const float*)table, q_heads, kv_heads, dim, table_rows, eps,
        empty_layout, nullptr);
    return (int)cudaGetLastError();
}

int w34_dflash_qk_norm_rope_kv_write_bf16_f32_table(
    const w34_exec* ex,
    void* qkv,
    void* q_out,
    const int32_t* positions,
    const void* qw,
    const void* kw,
    const void* table,
    const w34_kv_layer_layout* kv_layout,
    const int32_t* write_slot_mapping,
    int rows,
    int q_heads,
    int kv_heads,
    int dim,
    int table_rows,
    float eps) {
    if (!ex || !qkv || !positions || !qw || !kw || !table || !kv_layout || !write_slot_mapping ||
        rows < 0 || q_heads < 0 || kv_heads <= 0 || q_heads % kv_heads ||
        dim <= 0 || dim > 256 || dim % 2 || table_rows <= 0 ||
        !(eps > 0.0f) || !isfinite(eps) || q_out == qkv ||
        kv_layout->k_base == 0 || kv_layout->v_base == 0 ||
        kv_layout->block_size <= 0 || kv_layout->local_kv_heads != kv_heads ||
        kv_layout->head_dim != dim ||
        kv_layout->k_format != W34_KV_FORMAT_BF16 ||
        kv_layout->v_format != W34_KV_FORMAT_BF16) {
        return (int)cudaErrorInvalidValue;
    }
    if (rows == 0) return 0;
    qk_norm_rope_bf16_warp<<<dim3((unsigned)rows, (unsigned)(q_heads + kv_heads)), 32, 0, (cudaStream_t)ex->stream>>>(
        (__nv_bfloat16*)qkv, (__nv_bfloat16*)q_out, positions,
        (const __nv_bfloat16*)qw, (const __nv_bfloat16*)kw,
        (const float*)table, q_heads, kv_heads, dim, table_rows, eps,
        *kv_layout, write_slot_mapping);
    return (int)cudaGetLastError();
}

int w34_dflash_select_greedy_bf16(
    const w34_exec* ex,
    const int32_t* ids,
    const float* unary,
    const void* hidden,
    const void* pred,
    const void* succ,
    int32_t* out_path,
    int anchor,
    int k,
    int topk,
    int rank,
    int vocab) {
    if (!ex || !ids || !unary || !hidden || !pred || !succ || !out_path ||
        anchor < 0 || anchor >= vocab || k <= 0 || k > 16 ||
        topk <= 0 || topk > 32 || rank <= 0 || rank > 256 || vocab <= 0) {
        return (int)cudaErrorInvalidValue;
    }
    dflash_select_greedy_bf16_kernel<<<1, (unsigned)(32 * topk), 0, (cudaStream_t)ex->stream>>>(
        ids, unary, (const __nv_bfloat16*)hidden,
        (const __nv_bfloat16*)pred, (const __nv_bfloat16*)succ,
        out_path, nullptr, anchor, k, topk, rank, vocab);
    return (int)cudaGetLastError();
}

int w34_dflash_select_greedy_dev_anchor_bf16(
    const w34_exec* ex,
    const int32_t* ids,
    const float* unary,
    const void* hidden,
    const void* pred,
    const void* succ,
    int32_t* out_path,
    const int32_t* anchor_dev,
    int k,
    int topk,
    int rank,
    int vocab) {
    if (!ex || !ids || !unary || !hidden || !pred || !succ || !out_path || !anchor_dev ||
        k <= 0 || k > 16 || topk <= 0 || topk > 32 || rank <= 0 || rank > 256 || vocab <= 0) {
        return (int)cudaErrorInvalidValue;
    }
    dflash_select_greedy_bf16_kernel<<<1, (unsigned)(32 * topk), 0, (cudaStream_t)ex->stream>>>(
        ids, unary, (const __nv_bfloat16*)hidden,
        (const __nv_bfloat16*)pred, (const __nv_bfloat16*)succ,
        out_path, anchor_dev, 0, k, topk, rank, vocab);
    return (int)cudaGetLastError();
}

int w34_dflash_qk_norm_rope_bf16_f32_table(
    const w34_exec* ex,
    void* qkv,
    const int32_t* positions,
    const void* qw,
    const void* kw,
    const void* table,
    int rows,
    int q_heads,
    int kv_heads,
    int dim,
    int table_rows,
    float eps) {
    return w34_dflash_qk_norm_rope_bf16_f32_table_qout(
        ex, qkv, nullptr, positions, qw, kw, table,
        rows, q_heads, kv_heads, dim, table_rows, eps);
}

int w34_copy_cols_bf16(
    const w34_exec* ex,
    const void* src,
    void* dst,
    int rows,
    int src_stride,
    int dst_stride,
    int src_col,
    int dst_col,
    int cols) {
    if (!ex || !src || !dst || src == dst || rows < 0 || cols <= 0 ||
        src_col < 0 || dst_col < 0 || src_stride < cols || dst_stride < cols ||
        src_col > src_stride - cols || dst_col > dst_stride - cols) {
        return (int)cudaErrorInvalidValue;
    }
    const int64_t n = (int64_t)rows * cols;
    if (n == 0) return 0;
    if (n > INT_MAX) return (int)cudaErrorInvalidValue;
    copy_cols_bf16<<<(unsigned)((n + 255) / 256), 256, 0, (cudaStream_t)ex->stream>>>(
        (const __nv_bfloat16*)src, (__nv_bfloat16*)dst,
        rows, src_stride, dst_stride, src_col, dst_col, cols);
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
    if (!ex || !logits_f16 || !out_ids || !out_vals || num_rows < 0 ||
        local_vocab_size <= 0 || top_k <= 0 || top_k > 16) {
        return (int)cudaErrorInvalidValue;
    }
    if (num_rows == 0) return 0;
    topk_rows_range_f16_kernel<<<dim3(1u, (unsigned)num_rows), 256, 0, (cudaStream_t)ex->stream>>>(
        static_cast<const __half*>(logits_f16),
        out_ids,
        out_vals,
        num_rows,
        local_vocab_size,
        vocab_offset,
        top_k,
        1);
    return (int)cudaGetLastError();
}

int w34_topk_rows_split_f16(
    const w34_exec* ex,
    const void* logits_f16,
    int32_t* split_ids_ws,
    float* split_vals_ws,
    int32_t* out_ids,
    float* out_vals,
    int num_rows,
    int local_vocab_size,
    int vocab_offset,
    int top_k,
    int num_splits) {
    if (!ex || !logits_f16 || !out_ids || !out_vals || num_rows < 0 ||
        local_vocab_size <= 0 || top_k <= 0 || top_k > 16 ||
        num_splits <= 0 || num_splits > 32 || num_splits * top_k > 256 ||
        (num_splits > 1 && (!split_ids_ws || !split_vals_ws))) {
        return (int)cudaErrorInvalidValue;
    }
    if (num_rows == 0) return 0;
    if (num_splits == 1) {
        return w34_topk_rows_f16(
            ex, logits_f16, out_ids, out_vals, num_rows, local_vocab_size, vocab_offset, top_k);
    }
    const int chunk_elems = ((local_vocab_size + num_splits - 1) / num_splits + 7) & ~7;
    if (chunk_elems <= 65535) {
        topk_rows_packed_u32_f16_kernel<<<dim3((unsigned)num_splits, (unsigned)num_rows), 128, 0, (cudaStream_t)ex->stream>>>(
            static_cast<const __half*>(logits_f16),
            split_ids_ws,
            split_vals_ws,
            num_rows,
            local_vocab_size,
            vocab_offset,
            top_k,
            num_splits);
    } else {
        topk_rows_range_f16_kernel<<<dim3((unsigned)num_splits, (unsigned)num_rows), 256, 0, (cudaStream_t)ex->stream>>>(
            static_cast<const __half*>(logits_f16),
            split_ids_ws,
            split_vals_ws,
            num_rows,
            local_vocab_size,
            vocab_offset,
            top_k,
            num_splits);
    }
    topk_rows_merge_kernel<<<(unsigned)num_rows, 256, 0, (cudaStream_t)ex->stream>>>(
        split_ids_ws,
        split_vals_ws,
        out_ids,
        out_vals,
        num_rows,
        top_k,
        num_splits * top_k);
    return (int)cudaGetLastError();
}

} // extern "C"
