// Phase 5.3: Qwen3.5 / Qwen3.6 Hybrid MoE Kernels (Top-K Softmax Router,
// Fused W4A16 & FP16 Gate+Up+SiLU and Down+TopK-Accum+SharedExpert Gate).
#include "../w34_cuda.h"

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <float.h>
#include <limits.h>
#include <math.h>
#include <stdint.h>

// Match donor vLLM C++ custom ops (activation_kernels.cu, moe_wna16) which are
// compiled without --use_fast_math.
extern "C" __device__ float __nv_expf(float);

namespace w34_moe {

// Donor vLLM SiluAndMul (activation_kernels.cu): gate & up are FP16; SiLU(gate)
// is computed in FP32 with IEEE div + __nv_expf, rounded to FP16, then multiplied
// by up in FP16.
__device__ __forceinline__ __half marlin_silu_and_mul_f16(__half gate_h, __half up_h) {
    const float gf = __half2float(gate_h);
    const __half silu_h = __float2half_rn(__fdiv_rn(gf, 1.0f + __nv_expf(-gf)));
    return __hmul(silu_h, up_h);
}

// Fast 0-ULP IEEE-754 binary16 magic-bias unpack of 2 consecutive U4B8 nibbles
// (zero-point 8) from `packed_w` (byte `tid_in_grp`), pre-multiplied in FP16 by
// group `scale` to match donor vLLM Marlin / moe_wna16 dequantization.
__device__ __forceinline__ __half2 unpack_u4b8_pair_scaled_h2(
    uint32_t packed_w,
    int tid_in_grp,
    __half2 scale_h2) {
    const uint32_t byte_v = (packed_w >> (tid_in_grp << 3)) & 0xFFu;
    const uint32_t h2_bits = ((byte_v & 0x0Fu) | ((byte_v & 0xF0u) << 12)) | 0x64006400u;
    const uint32_t bias_bits = 0x64086408u; // (1032.0, 1032.0) in FP16
    const __half2 q_h2 = __hsub2(
        *reinterpret_cast<const __half2*>(&h2_bits),
        *reinterpret_cast<const __half2*>(&bias_bits));
    return __hmul2(q_h2, scale_h2);
}

__device__ __forceinline__ uint32_t unpack_u4b8_pair_scaled(
    uint32_t packed_w,
    int tid_in_grp,
    __half2 scale_h2) {
    const __half2 h2 = unpack_u4b8_pair_scaled_h2(packed_w, tid_in_grp, scale_h2);
    return *reinterpret_cast<const uint32_t*>(&h2);
}

__device__ __forceinline__ __half2 unpack_u4_zp_pair_scaled_h2(
    uint32_t packed_w,
    int tid_in_grp,
    uint32_t zp_nibble,
    __half2 scale_h2) {
    const uint32_t byte_v = (packed_w >> (tid_in_grp << 3)) & 0xFFu;
    const uint32_t h2_bits = ((byte_v & 0x0Fu) | ((byte_v & 0xF0u) << 12)) | 0x64006400u;
    const uint32_t bias_bits = 0x64006400u | zp_nibble | (zp_nibble << 16);
    const __half2 q_h2 = __hsub2(
        *reinterpret_cast<const __half2*>(&h2_bits),
        *reinterpret_cast<const __half2*>(&bias_bits));
    return __hmul2(q_h2, scale_h2);
}

__device__ __forceinline__ uint32_t unpack_u4_zp_pair_scaled(
    uint32_t packed_w,
    int tid_in_grp,
    uint32_t zp_nibble,
    __half2 scale_h2) {
    const __half2 h2 = unpack_u4_zp_pair_scaled_h2(packed_w, tid_in_grp, zp_nibble, scale_h2);
    return *reinterpret_cast<const uint32_t*>(&h2);
}

__device__ __forceinline__ uint32_t load_zp_nibble(
    const uint32_t* qzeros_exp,
    int row_in_expert,
    int groups_per_row,
    int g_idx) {
    const uint32_t packed_zp = __ldg(&qzeros_exp[(size_t)(row_in_expert >> 3) * groups_per_row + g_idx]);
    return (packed_zp >> ((row_in_expert & 7) << 2)) & 0xFu;
}

__device__ __forceinline__ float warp_reduce_sum(float v) {
#pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
        v += __shfl_xor_sync(0xffffffffu, v, off);
    }
    return v;
}

__device__ __forceinline__ float warp_reduce_max(float v) {
#pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
        v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, off));
    }
    return v;
}

// Specialized 1-warp (32-thread) register-only router for num_experts == 256 and top_k == 8
// with 16-byte aligned uint4 load (1 transaction per thread) and winner-only rescan.
__global__ void __launch_bounds__(32) moe_topk_router_256_k8_f16_kernel(
    const __half* __restrict__ router_logits,
    float* __restrict__ topk_weights,
    int32_t* __restrict__ topk_ids,
    int num_tokens,
    int norm_topk_prob,
    int scoring_func) {
    const int tok = blockIdx.x;
    if (tok >= num_tokens) return;

    const int lane = threadIdx.x & 31;
    const uint4* row_v4 = reinterpret_cast<const uint4*>(router_logits + (size_t)tok * 256);
    const uint4 v = __ldg(&row_v4[lane]);
    const __half* h = reinterpret_cast<const __half*>(&v);
    const int base_e = lane * 8;

    float p[8];
#pragma unroll
    for (int i = 0; i < 8; ++i) {
        p[i] = __half2float(h[i]);
    }

    if (scoring_func == 1) {
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            p[i] = 1.0f / (1.0f + expf(-p[i]));
        }
    } else {
        float local_max = p[0];
#pragma unroll
        for (int i = 1; i < 8; ++i) {
            local_max = fmaxf(local_max, p[i]);
        }
        const float max_val = warp_reduce_max(local_max);

        float local_sum = 0.0f;
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            const float ev = expf(p[i] - max_val);
            p[i] = ev;
            local_sum += ev;
        }
        const float sum_val = warp_reduce_sum(local_sum);
        const float inv_sum = (sum_val > 0.0f) ? (1.0f / sum_val) : 0.0f;
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            p[i] *= inv_sum;
        }
    }

    float best_v = p[0];
    int best_e = base_e;
#pragma unroll
    for (int i = 1; i < 8; ++i) {
        if (p[i] > best_v) {
            best_v = p[i];
            best_e = base_e + i;
        }
    }

    float w_sum = 0.0f;
    float my_w = 0.0f;
    int32_t my_id = 0;

#pragma unroll
    for (int k = 0; k < 8; ++k) {
        float cur_v = best_v;
        int cur_e = best_e;
#pragma unroll
        for (int off = 16; off > 0; off >>= 1) {
            const float ov = __shfl_xor_sync(0xffffffffu, cur_v, off);
            const int oe = __shfl_xor_sync(0xffffffffu, cur_e, off);
            if (ov > cur_v || (ov == cur_v && oe < cur_e)) {
                cur_v = ov;
                cur_e = oe;
            }
        }
        const float win_v = __shfl_sync(0xffffffffu, cur_v, 0);
        const int win_e = __shfl_sync(0xffffffffu, cur_e, 0);
        w_sum += win_v;
        if (lane == k) {
            my_w = win_v;
            my_id = win_e;
        }
        if ((win_e >> 3) == lane && k < 7) {
            best_v = -1.0f;
            best_e = 256;
#pragma unroll
            for (int i = 0; i < 8; ++i) {
                if (base_e + i == win_e) {
                    p[i] = -2.0f;
                }
                if (p[i] > best_v) {
                    best_v = p[i];
                    best_e = base_e + i;
                }
            }
        }
    }

    if (lane < 8) {
        const float scale = (norm_topk_prob != 0 && w_sum > 1e-12f) ? (1.0f / w_sum) : 1.0f;
        topk_weights[(size_t)tok * 8 + lane] = my_w * scale;
        topk_ids[(size_t)tok * 8 + lane] = my_id;
    }
}

// Fast 1-warp (32-thread) register-only router for num_experts <= 256 and top_k <= 32.
// Eliminates all shared memory and CTA barriers.
__global__ void moe_topk_router_warp256_f16_kernel(
    const __half* __restrict__ router_logits,
    float* __restrict__ topk_weights,
    int32_t* __restrict__ topk_ids,
    int num_tokens,
    int num_experts,
    int top_k,
    int norm_topk_prob,
    int scoring_func) {
    const int tok = blockIdx.x;
    if (tok >= num_tokens) return;

    const int lane = threadIdx.x & 31;
    const __half* row = router_logits + (size_t)tok * num_experts;

    float p[8];
#pragma unroll
    for (int i = 0; i < 8; ++i) {
        const int e = lane + i * 32;
        p[i] = (e < num_experts) ? __half2float(__ldg(&row[e])) : -FLT_MAX;
    }

    if (scoring_func == 1) {
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            const int e = lane + i * 32;
            p[i] = (e < num_experts) ? (1.0f / (1.0f + expf(-p[i]))) : -2.0f;
        }
    } else {
        float local_max = -FLT_MAX;
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            local_max = fmaxf(local_max, p[i]);
        }
        const float max_val = warp_reduce_max(local_max);

        float local_sum = 0.0f;
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            const int e = lane + i * 32;
            if (e < num_experts) {
                const float v = expf(p[i] - max_val);
                p[i] = v;
                local_sum += v;
            } else {
                p[i] = -2.0f;
            }
        }
        const float sum_val = warp_reduce_sum(local_sum);
        const float inv_sum = (sum_val > 0.0f) ? (1.0f / sum_val) : 0.0f;
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            const int e = lane + i * 32;
            if (e < num_experts) {
                p[i] *= inv_sum;
            }
        }
    }

    float w_sum = 0.0f;
    float my_w = 0.0f;
    int32_t my_id = 0;

    for (int k = 0; k < top_k; ++k) {
        float best_v = -1.0f;
        int best_e = num_experts;
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            const int e = lane + i * 32;
            if (e < num_experts) {
                const float v = p[i];
                if (v > best_v || (v == best_v && e < best_e)) {
                    best_v = v;
                    best_e = e;
                }
            }
        }
#pragma unroll
        for (int off = 16; off > 0; off >>= 1) {
            const float ov = __shfl_xor_sync(0xffffffffu, best_v, off);
            const int oe = __shfl_xor_sync(0xffffffffu, best_e, off);
            if (ov > best_v || (ov == best_v && oe < best_e)) {
                best_v = ov;
                best_e = oe;
            }
        }
        const float win_v = __shfl_sync(0xffffffffu, best_v, 0);
        const int win_e = __shfl_sync(0xffffffffu, best_e, 0);
        w_sum += win_v;
        if (lane == k) {
            my_w = win_v;
            my_id = win_e;
        }
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            if (lane + i * 32 == win_e) {
                p[i] = -2.0f;
            }
        }
    }

    if (lane < top_k) {
        const float scale = (norm_topk_prob != 0 && w_sum > 1e-12f) ? (1.0f / w_sum) : 1.0f;
        topk_weights[(size_t)tok * top_k + lane] = my_w * scale;
        topk_ids[(size_t)tok * top_k + lane] = my_id;
    }
}

// One block (128 threads) per token row.
// Supports arbitrary num_experts <= 1024 and top_k <= 32.
// Computes softmax (scoring_func == 0) or sigmoid (scoring_func == 1) over router_logits,
// selects top_k experts in descending order (ties broken by smaller expert_id),
// and optionally normalizes topk_weights so they sum to 1.
__global__ void moe_topk_router_f16_kernel(
    const __half* __restrict__ router_logits,
    float* __restrict__ topk_weights,
    int32_t* __restrict__ topk_ids,
    int num_tokens,
    int num_experts,
    int top_k,
    int norm_topk_prob,
    int scoring_func) {
    const int tok = blockIdx.x;
    if (tok >= num_tokens) return;

    const __half* row = router_logits + (size_t)tok * num_experts;
    __shared__ float s_probs[1024];
    __shared__ float s_warp[32];

    const int lane = threadIdx.x & 31;
    const int wid = threadIdx.x >> 5;

    if (scoring_func == 1) {
        // Sigmoid scoring
        for (int e = threadIdx.x; e < num_experts; e += blockDim.x) {
            float x = __half2float(row[e]);
            s_probs[e] = 1.0f / (1.0f + expf(-x));
        }
        __syncthreads();
    } else {
        // Softmax scoring
        float local_max = -FLT_MAX;
        for (int e = threadIdx.x; e < num_experts; e += blockDim.x) {
            local_max = fmaxf(local_max, __half2float(row[e]));
        }
        local_max = warp_reduce_max(local_max);
        if (lane == 0) s_warp[wid] = local_max;
        __syncthreads();
        if (wid == 0) {
            float m = (lane < (blockDim.x >> 5)) ? s_warp[lane] : -FLT_MAX;
            m = warp_reduce_max(m);
            if (lane == 0) s_warp[0] = m;
        }
        __syncthreads();
        const float max_val = s_warp[0];
        __syncthreads();

        float local_sum = 0.0f;
        for (int e = threadIdx.x; e < num_experts; e += blockDim.x) {
            float p = expf(__half2float(row[e]) - max_val);
            s_probs[e] = p;
            local_sum += p;
        }
        local_sum = warp_reduce_sum(local_sum);
        if (lane == 0) s_warp[wid] = local_sum;
        __syncthreads();
        if (wid == 0) {
            float s = (lane < (blockDim.x >> 5)) ? s_warp[lane] : 0.0f;
            s = warp_reduce_sum(s);
            if (lane == 0) s_warp[0] = s;
        }
        __syncthreads();
        const float inv_sum = (s_warp[0] > 0.0f) ? (1.0f / s_warp[0]) : 0.0f;
        for (int e = threadIdx.x; e < num_experts; e += blockDim.x) {
            s_probs[e] *= inv_sum;
        }
        __syncthreads();
    }

    // Thread 0..31 (Warp 0) finds top_k iteratively using warp reductions
    if (wid == 0) {
        float w_sum = 0.0f;
        float chosen_w[32];
        int32_t chosen_id[32];

        for (int k = 0; k < top_k; ++k) {
            float best_v = -1.0f;
            int best_e = num_experts;
            for (int e = lane; e < num_experts; e += 32) {
                float v = s_probs[e];
                if (v > best_v || (v == best_v && e < best_e)) {
                    best_v = v;
                    best_e = e;
                }
            }
#pragma unroll
            for (int off = 16; off > 0; off >>= 1) {
                float ov = __shfl_xor_sync(0xffffffffu, best_v, off);
                int oe = __shfl_xor_sync(0xffffffffu, best_e, off);
                if (ov > best_v || (ov == best_v && oe < best_e)) {
                    best_v = ov;
                    best_e = oe;
                }
            }
            if (lane == 0) {
                chosen_w[k] = best_v;
                chosen_id[k] = best_e;
                w_sum += best_v;
                if (best_e < num_experts) {
                    s_probs[best_e] = -2.0f;
                }
            }
            __syncwarp();
        }

        if (lane == 0) {
            const float scale = (norm_topk_prob != 0 && w_sum > 1e-12f) ? (1.0f / w_sum) : 1.0f;
            float* out_w = topk_weights + (size_t)tok * top_k;
            int32_t* out_id = topk_ids + (size_t)tok * top_k;
            for (int k = 0; k < top_k; ++k) {
                out_w[k] = chosen_w[k] * scale;
                out_id[k] = chosen_id[k];
            }
        }
    }
}

// Stage 1 (W4A16 U4B8 / U4+ZP N-major): Fused gate_proj + up_proj + SiLU(gate) * up.
// gate_up_packed: [num_experts, 2 * n_interm, k_hidden / 8] uint32_t
// gate_up_scales: [num_experts, 2 * n_interm, k_hidden / group_size] __half
// gate_up_qzeros: [num_experts, (2 * n_interm) / 8, k_hidden / group_size] uint32_t (when HAS_ZP)
// Grid: ((n_interm + 3) / 4, num_tokens * top_k), Block: 128 threads (4 warps).
template <bool HAS_ZP = false>
__global__ void __launch_bounds__(128, 8) moe_w4a16_gate_up_silu_f16_kernel(
    const __half* __restrict__ x,
    const uint32_t* __restrict__ gate_up_packed,
    const __half* __restrict__ gate_up_scales,
    const uint32_t* __restrict__ gate_up_qzeros,
    const int32_t* __restrict__ topk_ids,
    __half* __restrict__ interm_out,
    int num_tokens,
    int top_k,
    int num_experts,
    int n_interm,
    int k_hidden,
    int group_size) {
    const int pair_idx = blockIdx.y;
    if (pair_idx >= num_tokens * top_k) return;
    const int tok = pair_idx / top_k;
    const int expert_id = __ldg(&topk_ids[pair_idx]);
    if (expert_id < 0 || expert_id >= num_experts) return;

    const int lane = threadIdx.x & 31;
    const int wid = threadIdx.x >> 5; // 0..3
    const int col = blockIdx.x * 4 + wid;
    if (col >= n_interm) return;

    const int words_per_row = k_hidden >> 3;
    const int words_per_group = group_size >> 3;
    const int group_shift = ((words_per_group & (words_per_group - 1)) == 0)
        ? (__ffs(words_per_group) - 1)
        : -1;
    const int groups_per_row = (k_hidden + group_size - 1) / group_size;
    const int n_gate_up = 2 * n_interm;

    const __half* x_row = x + (size_t)tok * k_hidden;
    const uint4* x_vec4 = (const uint4*)x_row;

    const size_t exp_row_base = (size_t)expert_id * n_gate_up;
    const uint32_t* gate_w = gate_up_packed + (exp_row_base + col) * words_per_row;
    const uint32_t* up_w = gate_up_packed + (exp_row_base + n_interm + col) * words_per_row;
    const __half* gate_s = gate_up_scales + (exp_row_base + col) * groups_per_row;
    const __half* up_s = gate_up_scales + (exp_row_base + n_interm + col) * groups_per_row;
    const uint32_t* zp_exp = HAS_ZP
        ? (gate_up_qzeros + (size_t)expert_id * (n_gate_up >> 3) * groups_per_row)
        : nullptr;

    float gate_acc = 0.0f;
    float up_acc = 0.0f;

#pragma unroll 4
    for (int w = lane; w < words_per_row; w += 32) {
        const uint4 xv = __ldg(&x_vec4[w]);
        const uint32_t* x2 = reinterpret_cast<const uint32_t*>(&xv);
        const uint32_t gw = __ldg(&gate_w[w]);
        const uint32_t uw = __ldg(&up_w[w]);
        const int g_idx = (group_shift >= 0) ? (w >> group_shift) : (w / words_per_group);
        const __half2 gs_h2 = __half2half2(__ldg(&gate_s[g_idx]));
        const __half2 us_h2 = __half2half2(__ldg(&up_s[g_idx]));
        uint32_t g_zp = 8u;
        uint32_t u_zp = 8u;
        if constexpr (HAS_ZP) {
            g_zp = load_zp_nibble(zp_exp, col, groups_per_row, g_idx);
            u_zp = load_zp_nibble(zp_exp, n_interm + col, groups_per_row, g_idx);
        }

#pragma unroll
        for (int p = 0; p < 4; ++p) {
            const __half2 xh2 = *reinterpret_cast<const __half2*>(&x2[p]);
            const float x_lo = __low2float(xh2);
            const float x_hi = __high2float(xh2);

            const __half2 gwh2 = HAS_ZP
                ? unpack_u4_zp_pair_scaled_h2(gw, p, g_zp, gs_h2)
                : unpack_u4b8_pair_scaled_h2(gw, p, gs_h2);
            gate_acc += __low2float(gwh2) * x_lo;
            gate_acc += __high2float(gwh2) * x_hi;

            const __half2 uwh2 = HAS_ZP
                ? unpack_u4_zp_pair_scaled_h2(uw, p, u_zp, us_h2)
                : unpack_u4b8_pair_scaled_h2(uw, p, us_h2);
            up_acc += __low2float(uwh2) * x_lo;
            up_acc += __high2float(uwh2) * x_hi;
        }
    }

    gate_acc = warp_reduce_sum(gate_acc);
    up_acc = warp_reduce_sum(up_acc);

    if (lane == 0) {
        interm_out[(size_t)pair_idx * n_interm + col] =
            marlin_silu_and_mul_f16(__float2half_rn(gate_acc), __float2half_rn(up_acc));
    }
}

// Stage 2 (W4A16 U4B8 / U4+ZP N-major): Fused down_proj + topk_weights sum + optional shared_expert * sigmoid(gate).
// down_packed: [num_experts, k_hidden, n_interm / 8] uint32_t
// down_scales: [num_experts, k_hidden, n_interm / group_size] __half
// down_qzeros: [num_experts, k_hidden / 8, n_interm / group_size] uint32_t (when HAS_ZP)
// Grid: ((k_hidden + 3) / 4, num_tokens), Block: 128 threads (4 warps).
template <bool HAS_ZP = false, int FIXED_WORDS_PER_ROW = 0, int FIXED_TOP_K = 0>
__global__ void __launch_bounds__(128, 8) moe_w4a16_down_accum_f16_kernel(
    const __half* __restrict__ interm_in,
    const uint32_t* __restrict__ down_packed,
    const __half* __restrict__ down_scales,
    const uint32_t* __restrict__ down_qzeros,
    const int32_t* __restrict__ topk_ids,
    const float* __restrict__ topk_weights,
    const __half* __restrict__ shared_expert_out,
    const __half* __restrict__ shared_expert_gate_logits,
    __half* __restrict__ out,
    int num_tokens,
    int top_k,
    int num_experts,
    int k_hidden,
    int n_interm,
    int group_size) {
    const int tok = blockIdx.y;
    if (tok >= num_tokens) return;

    const int lane = threadIdx.x & 31;
    const int wid = threadIdx.x >> 5; // 0..3
    const int out_col = blockIdx.x * 4 + wid;
    if (out_col >= k_hidden) return;

    const int words_per_row = (FIXED_WORDS_PER_ROW > 0) ? FIXED_WORDS_PER_ROW : (n_interm >> 3);
    const int words_per_group = group_size >> 3;
    const int group_shift = ((words_per_group & (words_per_group - 1)) == 0)
        ? (__ffs(words_per_group) - 1)
        : -1;
    const int groups_per_row = (n_interm + group_size - 1) / group_size;

    float total_sum = 0.0f;

    if constexpr (FIXED_WORDS_PER_ROW > 0 && FIXED_TOP_K > 0) {
#pragma unroll
        for (int slot = 0; slot < FIXED_TOP_K; ++slot) {
            const int pair_idx = tok * FIXED_TOP_K + slot;
            const int expert_id = __ldg(&topk_ids[pair_idx]);
            if (expert_id < 0 || expert_id >= num_experts) continue;
            const __half tw_h = __float2half_rn(__ldg(&topk_weights[pair_idx]));

            const uint4* x_vec4 = (const uint4*)(interm_in + (size_t)pair_idx * n_interm);
            const size_t row_idx = (size_t)expert_id * k_hidden + out_col;
            const uint32_t* dw_ptr = down_packed + row_idx * FIXED_WORDS_PER_ROW;
            const __half* ds_ptr = down_scales + row_idx * groups_per_row;
            const uint32_t* zp_exp = HAS_ZP
                ? (down_qzeros + (size_t)expert_id * (k_hidden >> 3) * groups_per_row)
                : nullptr;

            float dot = 0.0f;
#pragma unroll
            for (int iter = 0; iter < (FIXED_WORDS_PER_ROW / 32); ++iter) {
                const int w = lane + iter * 32;
                const uint4 xv = __ldg(&x_vec4[w]);
                const uint32_t* x2 = reinterpret_cast<const uint32_t*>(&xv);
                const uint32_t dw = __ldg(&dw_ptr[w]);
                const int g_idx = (group_shift >= 0) ? (w >> group_shift) : (w / words_per_group);
                const __half2 ds_h2 = __half2half2(__ldg(&ds_ptr[g_idx]));
                uint32_t d_zp = 8u;
                if constexpr (HAS_ZP) {
                    d_zp = load_zp_nibble(zp_exp, out_col, groups_per_row, g_idx);
                }

#pragma unroll
                for (int p = 0; p < 4; ++p) {
                    const __half2 xh2 = *reinterpret_cast<const __half2*>(&x2[p]);
                    const __half2 dwh2 = HAS_ZP
                        ? unpack_u4_zp_pair_scaled_h2(dw, p, d_zp, ds_h2)
                        : unpack_u4b8_pair_scaled_h2(dw, p, ds_h2);
                    dot += __low2float(dwh2) * __low2float(xh2);
                    dot += __high2float(dwh2) * __high2float(xh2);
                }
            }
            dot = warp_reduce_sum(dot);
            total_sum += __half2float(__hmul(__float2half_rn(dot), tw_h));
        }
    } else {
        for (int slot = 0; slot < top_k; ++slot) {
            const int pair_idx = tok * top_k + slot;
            const int expert_id = __ldg(&topk_ids[pair_idx]);
            if (expert_id < 0 || expert_id >= num_experts) continue;
            const __half tw_h = __float2half_rn(__ldg(&topk_weights[pair_idx]));

            const uint4* x_vec4 = (const uint4*)(interm_in + (size_t)pair_idx * n_interm);
            const size_t row_idx = (size_t)expert_id * k_hidden + out_col;
            const uint32_t* dw_ptr = down_packed + row_idx * words_per_row;
            const __half* ds_ptr = down_scales + row_idx * groups_per_row;
            const uint32_t* zp_exp = HAS_ZP
                ? (down_qzeros + (size_t)expert_id * (k_hidden >> 3) * groups_per_row)
                : nullptr;

            float dot = 0.0f;
            for (int w = lane; w < words_per_row; w += 32) {
                const uint4 xv = __ldg(&x_vec4[w]);
                const uint32_t* x2 = reinterpret_cast<const uint32_t*>(&xv);
                const uint32_t dw = __ldg(&dw_ptr[w]);
                const int g_idx = (group_shift >= 0) ? (w >> group_shift) : (w / words_per_group);
                const __half2 ds_h2 = __half2half2(__ldg(&ds_ptr[g_idx]));
                uint32_t d_zp = 8u;
                if constexpr (HAS_ZP) {
                    d_zp = load_zp_nibble(zp_exp, out_col, groups_per_row, g_idx);
                }

#pragma unroll
                for (int p = 0; p < 4; ++p) {
                    const __half2 xh2 = *reinterpret_cast<const __half2*>(&x2[p]);
                    const __half2 dwh2 = HAS_ZP
                        ? unpack_u4_zp_pair_scaled_h2(dw, p, d_zp, ds_h2)
                        : unpack_u4b8_pair_scaled_h2(dw, p, ds_h2);
                    dot += __low2float(dwh2) * __low2float(xh2);
                    dot += __high2float(dwh2) * __high2float(xh2);
                }
            }
            dot = warp_reduce_sum(dot);
            total_sum += __half2float(__hmul(__float2half_rn(dot), tw_h));
        }
    }

    if (lane == 0) {
        __half res_h = __float2half_rn(total_sum);
        if (shared_expert_out != nullptr) {
            __half sh_h = shared_expert_out[(size_t)tok * k_hidden + out_col];
            if (shared_expert_gate_logits != nullptr) {
                const float gl = __half2float(shared_expert_gate_logits[tok]);
                const __half sig_h = __float2half_rn(__fdiv_rn(1.0f, 1.0f + __nv_expf(-gl)));
                sh_h = __hmul(sig_h, sh_h);
            }
            res_h = __hadd(res_h, sh_h);
        }
        out[(size_t)tok * k_hidden + out_col] = res_h;
    }
}

// Dense FP16 Stage 1 (for unquantized MoE models & unit tests):
// gate_up_w: [num_experts, 2 * n_interm, k_hidden] __half
__global__ void moe_f16_gate_up_silu_kernel(
    const __half* __restrict__ x,
    const __half* __restrict__ gate_up_w,
    const int32_t* __restrict__ topk_ids,
    __half* __restrict__ interm_out,
    int num_tokens,
    int top_k,
    int num_experts,
    int n_interm,
    int k_hidden) {
    const int pair_idx = blockIdx.y;
    if (pair_idx >= num_tokens * top_k) return;
    const int tok = pair_idx / top_k;
    const int expert_id = topk_ids[pair_idx];
    if (expert_id < 0 || expert_id >= num_experts) return;

    const int lane = threadIdx.x & 31;
    const int wid = threadIdx.x >> 5;
    const int col = blockIdx.x * 4 + wid;
    if (col >= n_interm) return;

    const __half* x_row = x + (size_t)tok * k_hidden;
    const size_t exp_base = (size_t)expert_id * (2 * n_interm);
    const __half* gw = gate_up_w + (exp_base + col) * k_hidden;
    const __half* uw = gate_up_w + (exp_base + n_interm + col) * k_hidden;

    float g_acc = 0.0f;
    float u_acc = 0.0f;
#pragma unroll 4
    for (int c = lane; c < k_hidden; c += 32) {
        const float xf = __half2float(__ldg(&x_row[c]));
        g_acc += xf * __half2float(__ldg(&gw[c]));
        u_acc += xf * __half2float(__ldg(&uw[c]));
    }
    g_acc = warp_reduce_sum(g_acc);
    u_acc = warp_reduce_sum(u_acc);
    if (lane == 0) {
        const float silu_g = g_acc / (1.0f + expf(-g_acc));
        interm_out[(size_t)pair_idx * n_interm + col] = __float2half(silu_g * u_acc);
    }
}

// Dense FP16 Stage 2:
// down_w: [num_experts, k_hidden, n_interm] __half
__global__ void moe_f16_down_accum_kernel(
    const __half* __restrict__ interm_in,
    const __half* __restrict__ down_w,
    const int32_t* __restrict__ topk_ids,
    const float* __restrict__ topk_weights,
    const __half* __restrict__ shared_expert_out,
    const __half* __restrict__ shared_expert_gate_logits,
    __half* __restrict__ out,
    int num_tokens,
    int top_k,
    int num_experts,
    int k_hidden,
    int n_interm) {
    const int tok = blockIdx.y;
    if (tok >= num_tokens) return;

    const int lane = threadIdx.x & 31;
    const int wid = threadIdx.x >> 5;
    const int out_col = blockIdx.x * 4 + wid;
    if (out_col >= k_hidden) return;

    float total_sum = 0.0f;
    for (int slot = 0; slot < top_k; ++slot) {
        const int pair_idx = tok * top_k + slot;
        const int expert_id = topk_ids[pair_idx];
        if (expert_id < 0 || expert_id >= num_experts) continue;
        const float tw = topk_weights[pair_idx];
        const __half* x_row = interm_in + (size_t)pair_idx * n_interm;
        const __half* dw_row = down_w + ((size_t)expert_id * k_hidden + out_col) * n_interm;

        float dot = 0.0f;
#pragma unroll 4
        for (int c = lane; c < n_interm; c += 32) {
            dot += __half2float(__ldg(&x_row[c])) * __half2float(__ldg(&dw_row[c]));
        }
        dot = warp_reduce_sum(dot);
        total_sum += tw * dot;
    }

    if (lane == 0) {
        if (shared_expert_out != nullptr) {
            float sh = __half2float(shared_expert_out[(size_t)tok * k_hidden + out_col]);
            if (shared_expert_gate_logits != nullptr) {
                float gl = __half2float(shared_expert_gate_logits[tok]);
                sh *= 1.0f / (1.0f + expf(-gl));
            }
            total_sum += sh;
        }
        out[(size_t)tok * k_hidden + out_col] = __float2half(total_sum);
    }
}

// --- Phase 6.3: Token-grouped (expert_id, token_slot) Marlin/moe_wna16 mma.sync MoE (M >= 4) ---

// Templated CUDA-Graph-safe single-CTA shared-memory counting sort grouping
// `total_pairs = num_tokens * top_k` routed (token, topk_slot) pairs into homogeneous
// BLOCK_M (`4` for sparse M=4..31 batches, `16` for dense mma.sync tiles) expert tiles.
template <int BLOCK_M>
__global__ void moe_align_block_size_kernel(
    const int32_t* __restrict__ topk_ids,
    int32_t* __restrict__ sorted_pair_ids,
    int32_t* __restrict__ tile_expert_ids,
    int total_pairs,
    int num_experts,
    int max_tiles) {
    __shared__ int s_counts[1024];
    __shared__ int s_write_off[1024];

    const int tid = (int)threadIdx.x;
    for (int e = tid; e < num_experts; e += 256) {
        s_counts[e] = 0;
    }
    const int total_slots = max_tiles * BLOCK_M;
    for (int i = tid; i < total_slots; i += 256) {
        sorted_pair_ids[i] = -1;
    }
    for (int t = tid; t < max_tiles; t += 256) {
        tile_expert_ids[t] = -1;
    }
    __syncthreads();

    for (int p = tid; p < total_pairs; p += 256) {
        const int e = topk_ids[p];
        if (e >= 0 && e < num_experts) {
            atomicAdd(&s_counts[e], 1);
        }
    }
    __syncthreads();

    if (num_experts <= 256) {
        const int c = (tid < num_experts) ? s_counts[tid] : 0;
        const int nt = (c > 0) ? ((c + BLOCK_M - 1) / BLOCK_M) : 0;
        int scan = nt;
        const int lane = tid & 31;
        const int wid = tid >> 5;
#pragma unroll
        for (int off = 1; off < 32; off <<= 1) {
            const int v = __shfl_up_sync(0xffffffffu, scan, off);
            if (lane >= off) scan += v;
        }
        __shared__ int s_warp_tot[8];
        if (lane == 31) s_warp_tot[wid] = scan;
        __syncthreads();
        if (wid == 0) {
            int wscan = (lane < 8) ? s_warp_tot[lane] : 0;
#pragma unroll
            for (int off = 1; off < 8; off <<= 1) {
                const int v = __shfl_up_sync(0xffffffffu, wscan, off);
                if (lane >= off) wscan += v;
            }
            if (lane < 8) s_warp_tot[lane] = wscan;
        }
        __syncthreads();
        const int warp_base = (wid > 0) ? s_warp_tot[wid - 1] : 0;
        const int start_tile = warp_base + (scan - nt);
        if (tid < num_experts) {
            s_write_off[tid] = start_tile * BLOCK_M;
            for (int t = 0; t < nt; ++t) {
                tile_expert_ids[start_tile + t] = tid;
            }
        }
    } else if (tid == 0) {
        int cur_tile = 0;
        for (int e = 0; e < num_experts; ++e) {
            const int c = s_counts[e];
            s_write_off[e] = cur_tile * BLOCK_M;
            if (c > 0) {
                const int nt = (c + BLOCK_M - 1) / BLOCK_M;
                for (int t = 0; t < nt; ++t) {
                    tile_expert_ids[cur_tile + t] = e;
                }
                cur_tile += nt;
            }
        }
    }
    __syncthreads();

    for (int p = tid; p < total_pairs; p += 256) {
        const int e = topk_ids[p];
        if (e >= 0 && e < num_experts) {
            const int pos = atomicAdd(&s_write_off[e], 1);
            sorted_pair_ids[pos] = p;
        }
    }
}

// Grouped Stage 1 (W4A16 U4B8 / U4+ZP -> m16n8k16 mma.sync):
// Computes gate_proj AND up_proj simultaneously (2x activation A register reuse) + SiLU(gate) * up.
// Grid: ((n_interm + 31) / 32, max_tiles), Block: 128 (4 warps; 8 gate + 8 up cols/warp = 32 cols/CTA).
template <bool HAS_ZP = false>
__global__ void moe_w4a16_grouped_gate_up_silu_mma_f16_kernel(
    const __half* __restrict__ x,
    const uint32_t* __restrict__ gate_up_packed,
    const __half* __restrict__ gate_up_scales,
    const uint32_t* __restrict__ gate_up_qzeros,
    const int32_t* __restrict__ sorted_pair_ids,
    const int32_t* __restrict__ tile_expert_ids,
    __half* __restrict__ interm_out,
    int top_k,
    int n_interm,
    int k_hidden,
    int group_size) {
    const int tile_idx = (int)blockIdx.y;
    const int expert_id = tile_expert_ids[tile_idx];
    if (expert_id < 0) return;

    __shared__ __align__(16) __half s_a[16][72];
    __shared__ int s_pairs[16];

    const int tid = (int)threadIdx.x;
    const int lane = tid & 31;
    const int wid = tid >> 5;
    const int group_id = lane >> 2;
    const int tid_in_grp = lane & 3;

    if (tid < 16) {
        s_pairs[tid] = sorted_pair_ids[(tile_idx << 4) + tid];
    }
    __syncthreads();

    const int n_warp_base = (int)blockIdx.x * 32 + wid * 8;
    const int n_col = n_warp_base + group_id;
    const bool n_valid = (n_col < n_interm);

    const int words_per_row = k_hidden >> 3;
    const int groups_per_row = k_hidden / group_size;
    const size_t exp_base = (size_t)expert_id * (size_t)(2 * n_interm);
    const uint32_t* gate_w_row = n_valid
        ? (gate_up_packed + (exp_base + (size_t)n_col) * words_per_row)
        : nullptr;
    const uint32_t* up_w_row = n_valid
        ? (gate_up_packed + (exp_base + (size_t)(n_interm + n_col)) * words_per_row)
        : nullptr;
    const __half* gate_s_row = n_valid
        ? (gate_up_scales + (exp_base + (size_t)n_col) * groups_per_row)
        : nullptr;
    const __half* up_s_row = n_valid
        ? (gate_up_scales + (exp_base + (size_t)(n_interm + n_col)) * groups_per_row)
        : nullptr;
    const uint32_t* zp_exp = HAS_ZP
        ? (gate_up_qzeros + (size_t)expert_id * ((size_t)(2 * n_interm) >> 3) * groups_per_row)
        : nullptr;

    float g0 = 0.0f, g1 = 0.0f, g2 = 0.0f, g3 = 0.0f;
    float u0 = 0.0f, u1 = 0.0f, u2 = 0.0f, u3 = 0.0f;

    const int load_r = tid >> 3;
    const int load_c8 = (tid & 7) << 3;
    const int load_pair = s_pairs[load_r];
    const int load_tok = (load_pair >= 0) ? (load_pair / top_k) : -1;

    for (int kc = 0; kc < k_hidden; kc += 64) {
        if (load_tok >= 0) {
            *reinterpret_cast<uint4*>(&s_a[load_r][load_c8]) =
                *reinterpret_cast<const uint4*>(&x[(size_t)load_tok * k_hidden + kc + load_c8]);
        } else {
            *reinterpret_cast<uint4*>(&s_a[load_r][load_c8]) = make_uint4(0u, 0u, 0u, 0u);
        }
        __syncthreads();

        const int g_idx = kc / group_size;
        const __half2 gs_h2 = n_valid ? __half2half2(gate_s_row[g_idx]) : __float2half2_rn(0.0f);
        const __half2 us_h2 = n_valid ? __half2half2(up_s_row[g_idx]) : __float2half2_rn(0.0f);
        uint32_t g_zp_64 = 8u;
        uint32_t u_zp_64 = 8u;
        if constexpr (HAS_ZP) {
            if (n_valid && group_size >= 64) {
                g_zp_64 = load_zp_nibble(zp_exp, n_col, groups_per_row, g_idx);
                u_zp_64 = load_zp_nibble(zp_exp, n_interm + n_col, groups_per_row, g_idx);
            }
        }

#pragma unroll
        for (int ks = 0; ks < 4; ++ks) {
            const int k_off = ks << 4;
            const int col_base = kc + k_off;
            const uint32_t a0 = *reinterpret_cast<const uint32_t*>(&s_a[group_id][k_off + 2 * tid_in_grp]);
            const uint32_t a1 = *reinterpret_cast<const uint32_t*>(&s_a[group_id + 8][k_off + 2 * tid_in_grp]);
            const uint32_t a2 = *reinterpret_cast<const uint32_t*>(&s_a[group_id][k_off + 8 + 2 * tid_in_grp]);
            const uint32_t a3 = *reinterpret_cast<const uint32_t*>(&s_a[group_id + 8][k_off + 8 + 2 * tid_in_grp]);

            uint32_t gb0 = 0u, gb1 = 0u, ub0 = 0u, ub1 = 0u;
            if (n_valid) {
                const int w_idx = col_base >> 3;
                const int cur_g = col_base / group_size;
                const __half2 gs_cur = (group_size >= 64)
                    ? gs_h2
                    : __half2half2(gate_s_row[cur_g]);
                const __half2 us_cur = (group_size >= 64)
                    ? us_h2
                    : __half2half2(up_s_row[cur_g]);
                if constexpr (HAS_ZP) {
                    const uint32_t g_zp = (group_size >= 64)
                        ? g_zp_64
                        : load_zp_nibble(zp_exp, n_col, groups_per_row, cur_g);
                    const uint32_t u_zp = (group_size >= 64)
                        ? u_zp_64
                        : load_zp_nibble(zp_exp, n_interm + n_col, groups_per_row, cur_g);
                    gb0 = unpack_u4_zp_pair_scaled(gate_w_row[w_idx], tid_in_grp, g_zp, gs_cur);
                    gb1 = unpack_u4_zp_pair_scaled(gate_w_row[w_idx + 1], tid_in_grp, g_zp, gs_cur);
                    ub0 = unpack_u4_zp_pair_scaled(up_w_row[w_idx], tid_in_grp, u_zp, us_cur);
                    ub1 = unpack_u4_zp_pair_scaled(up_w_row[w_idx + 1], tid_in_grp, u_zp, us_cur);
                } else {
                    gb0 = unpack_u4b8_pair_scaled(gate_w_row[w_idx], tid_in_grp, gs_cur);
                    gb1 = unpack_u4b8_pair_scaled(gate_w_row[w_idx + 1], tid_in_grp, gs_cur);
                    ub0 = unpack_u4b8_pair_scaled(up_w_row[w_idx], tid_in_grp, us_cur);
                    ub1 = unpack_u4b8_pair_scaled(up_w_row[w_idx + 1], tid_in_grp, us_cur);
                }
            }
            asm volatile(
                "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
                "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                : "+f"(g0), "+f"(g1), "+f"(g2), "+f"(g3)
                : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(gb0), "r"(gb1));
            asm volatile(
                "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
                "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                : "+f"(u0), "+f"(u1), "+f"(u2), "+f"(u3)
                : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(ub0), "r"(ub1));
        }
        __syncthreads();
    }

    const int out_c0 = n_warp_base + 2 * tid_in_grp;
    if (out_c0 + 1 < n_interm) {
        const int p0 = s_pairs[group_id];
        if (p0 >= 0) {
            const __half v0 = marlin_silu_and_mul_f16(__float2half_rn(g0), __float2half_rn(u0));
            const __half v1 = marlin_silu_and_mul_f16(__float2half_rn(g1), __float2half_rn(u1));
            *reinterpret_cast<__half2*>(&interm_out[(size_t)p0 * n_interm + out_c0]) =
                __halves2half2(v0, v1);
        }
        const int p1 = s_pairs[group_id + 8];
        if (p1 >= 0) {
            const __half v2 = marlin_silu_and_mul_f16(__float2half_rn(g2), __float2half_rn(u2));
            const __half v3 = marlin_silu_and_mul_f16(__float2half_rn(g3), __float2half_rn(u3));
            *reinterpret_cast<__half2*>(&interm_out[(size_t)p1 * n_interm + out_c0]) =
                __halves2half2(v2, v3);
        }
    }
}

// Grouped Stage 2a (W4A16 U4B8 / U4+ZP -> m16n8k16 mma.sync with donor Marlin FP16 pre-scaling):
// Computes FP16 topk_weights[pair] * FP16(interm_in[pair] @ W_down[expert]^T) into FP16 `stage2_pair_out`.
// Grid: ((k_hidden + 63) / 64, max_tiles), Block: 128 (4 warps; 2x8 = 16 cols/warp = 64 cols/CTA).
template <bool HAS_ZP = false>
__global__ void moe_w4a16_grouped_down_mma_f16_kernel(
    const __half* __restrict__ interm_in,
    const uint32_t* __restrict__ down_packed,
    const __half* __restrict__ down_scales,
    const uint32_t* __restrict__ down_qzeros,
    const int32_t* __restrict__ sorted_pair_ids,
    const int32_t* __restrict__ tile_expert_ids,
    const float* __restrict__ topk_weights,
    __half* __restrict__ stage2_pair_out,
    int k_hidden,
    int n_interm,
    int group_size) {
    const int tile_idx = (int)blockIdx.y;
    const int expert_id = tile_expert_ids[tile_idx];
    if (expert_id < 0) return;

    __shared__ __align__(16) __half s_a[16][72];
    __shared__ int s_pairs[16];
    __shared__ __half s_tw[16];

    const int tid = (int)threadIdx.x;
    const int lane = tid & 31;
    const int wid = tid >> 5;
    const int group_id = lane >> 2;
    const int tid_in_grp = lane & 3;

    if (tid < 16) {
        const int p = sorted_pair_ids[(tile_idx << 4) + tid];
        s_pairs[tid] = p;
        s_tw[tid] = (p >= 0) ? __float2half_rn(topk_weights[p]) : __float2half_rn(0.0f);
    }
    __syncthreads();

    const int out_warp_base = (int)blockIdx.x * 64 + wid * 16;
    const int col_a = out_warp_base + group_id;
    const int col_b = out_warp_base + 8 + group_id;
    const bool valid_a = (col_a < k_hidden);
    const bool valid_b = (col_b < k_hidden);

    const int words_per_row = n_interm >> 3;
    const int groups_per_row = n_interm / group_size;
    const size_t exp_base = (size_t)expert_id * (size_t)k_hidden;
    const uint32_t* dw_a = valid_a ? (down_packed + (exp_base + (size_t)col_a) * words_per_row) : nullptr;
    const uint32_t* dw_b = valid_b ? (down_packed + (exp_base + (size_t)col_b) * words_per_row) : nullptr;
    const __half* ds_a = valid_a ? (down_scales + (exp_base + (size_t)col_a) * groups_per_row) : nullptr;
    const __half* ds_b = valid_b ? (down_scales + (exp_base + (size_t)col_b) * groups_per_row) : nullptr;
    const uint32_t* zp_exp = HAS_ZP
        ? (down_qzeros + (size_t)expert_id * ((size_t)k_hidden >> 3) * groups_per_row)
        : nullptr;

    float d0_0 = 0.0f, d0_1 = 0.0f, d0_2 = 0.0f, d0_3 = 0.0f;
    float d1_0 = 0.0f, d1_1 = 0.0f, d1_2 = 0.0f, d1_3 = 0.0f;

    const int load_r = tid >> 3;
    const int load_c8 = (tid & 7) << 3;
    const int load_pair = s_pairs[load_r];

    for (int kc = 0; kc < n_interm; kc += 64) {
        if (load_pair >= 0) {
            *reinterpret_cast<uint4*>(&s_a[load_r][load_c8]) =
                *reinterpret_cast<const uint4*>(&interm_in[(size_t)load_pair * n_interm + kc + load_c8]);
        } else {
            *reinterpret_cast<uint4*>(&s_a[load_r][load_c8]) = make_uint4(0u, 0u, 0u, 0u);
        }
        __syncthreads();

        const int g_idx = kc / group_size;
        const __half2 sa_h2 = valid_a ? __half2half2(ds_a[g_idx]) : __float2half2_rn(0.0f);
        const __half2 sb_h2 = valid_b ? __half2half2(ds_b[g_idx]) : __float2half2_rn(0.0f);
        uint32_t za_64 = 8u;
        uint32_t zb_64 = 8u;
        if constexpr (HAS_ZP) {
            if (group_size >= 64) {
                if (valid_a) za_64 = load_zp_nibble(zp_exp, col_a, groups_per_row, g_idx);
                if (valid_b) zb_64 = load_zp_nibble(zp_exp, col_b, groups_per_row, g_idx);
            }
        }

#pragma unroll
        for (int ks = 0; ks < 4; ++ks) {
            const int k_off = ks << 4;
            const int col_base = kc + k_off;
            const uint32_t a0 = *reinterpret_cast<const uint32_t*>(&s_a[group_id][k_off + 2 * tid_in_grp]);
            const uint32_t a1 = *reinterpret_cast<const uint32_t*>(&s_a[group_id + 8][k_off + 2 * tid_in_grp]);
            const uint32_t a2 = *reinterpret_cast<const uint32_t*>(&s_a[group_id][k_off + 8 + 2 * tid_in_grp]);
            const uint32_t a3 = *reinterpret_cast<const uint32_t*>(&s_a[group_id + 8][k_off + 8 + 2 * tid_in_grp]);

            const int w_idx = col_base >> 3;
            const int cur_g = col_base / group_size;
            uint32_t ba0 = 0u, ba1 = 0u, bb0 = 0u, bb1 = 0u;
            if (valid_a) {
                const __half2 sa_cur = (group_size >= 64)
                    ? sa_h2
                    : __half2half2(ds_a[cur_g]);
                if constexpr (HAS_ZP) {
                    const uint32_t za = (group_size >= 64)
                        ? za_64
                        : load_zp_nibble(zp_exp, col_a, groups_per_row, cur_g);
                    ba0 = unpack_u4_zp_pair_scaled(dw_a[w_idx], tid_in_grp, za, sa_cur);
                    ba1 = unpack_u4_zp_pair_scaled(dw_a[w_idx + 1], tid_in_grp, za, sa_cur);
                } else {
                    ba0 = unpack_u4b8_pair_scaled(dw_a[w_idx], tid_in_grp, sa_cur);
                    ba1 = unpack_u4b8_pair_scaled(dw_a[w_idx + 1], tid_in_grp, sa_cur);
                }
            }
            if (valid_b) {
                const __half2 sb_cur = (group_size >= 64)
                    ? sb_h2
                    : __half2half2(ds_b[cur_g]);
                if constexpr (HAS_ZP) {
                    const uint32_t zb = (group_size >= 64)
                        ? zb_64
                        : load_zp_nibble(zp_exp, col_b, groups_per_row, cur_g);
                    bb0 = unpack_u4_zp_pair_scaled(dw_b[w_idx], tid_in_grp, zb, sb_cur);
                    bb1 = unpack_u4_zp_pair_scaled(dw_b[w_idx + 1], tid_in_grp, zb, sb_cur);
                } else {
                    bb0 = unpack_u4b8_pair_scaled(dw_b[w_idx], tid_in_grp, sb_cur);
                    bb1 = unpack_u4b8_pair_scaled(dw_b[w_idx + 1], tid_in_grp, sb_cur);
                }
            }
            asm volatile(
                "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
                "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                : "+f"(d0_0), "+f"(d0_1), "+f"(d0_2), "+f"(d0_3)
                : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(ba0), "r"(ba1));
            asm volatile(
                "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
                "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                : "+f"(d1_0), "+f"(d1_1), "+f"(d1_2), "+f"(d1_3)
                : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(bb0), "r"(bb1));
        }
        __syncthreads();
    }

    const int c0_a = out_warp_base + 2 * tid_in_grp;
    const int c0_b = out_warp_base + 8 + 2 * tid_in_grp;
    const int p0 = s_pairs[group_id];
    if (p0 >= 0) {
        const __half2 tw0_h2 = __half2half2(s_tw[group_id]);
        if (c0_a + 1 < k_hidden) {
            *reinterpret_cast<__half2*>(&stage2_pair_out[(size_t)p0 * k_hidden + c0_a]) =
                __hmul2(__floats2half2_rn(d0_0, d0_1), tw0_h2);
        }
        if (c0_b + 1 < k_hidden) {
            *reinterpret_cast<__half2*>(&stage2_pair_out[(size_t)p0 * k_hidden + c0_b]) =
                __hmul2(__floats2half2_rn(d1_0, d1_1), tw0_h2);
        }
    }
    const int p1 = s_pairs[group_id + 8];
    if (p1 >= 0) {
        const __half2 tw1_h2 = __half2half2(s_tw[group_id + 8]);
        if (c0_a + 1 < k_hidden) {
            *reinterpret_cast<__half2*>(&stage2_pair_out[(size_t)p1 * k_hidden + c0_a]) =
                __hmul2(__floats2half2_rn(d0_2, d0_3), tw1_h2);
        }
        if (c0_b + 1 < k_hidden) {
            *reinterpret_cast<__half2*>(&stage2_pair_out[(size_t)p1 * k_hidden + c0_b]) =
                __hmul2(__floats2half2_rn(d1_2, d1_3), tw1_h2);
        }
    }
}


// Coalesced sub-warp uint4 expert-grouped kernel for M = 4..7 (BLOCK_M = 4):
// Eliminates 40x strided-row L1 MSHR pressure at very small M by assigning
// `lanes_per_row = min(32, K / 32)` contiguous uint4 weight loads per row
// (512 contiguous bytes = 4 full 128B cache lines per warp instruction for all
// K in {128, 256, 512, 1024, 2048}), with a 4-row register-cached `valid_cnt == 1`
// fast path and a low-register per-slot loop for `valid_cnt >= 2` to preserve SM occupancy.
template <bool GATE>
__global__ __launch_bounds__(128, 4) void moe_sparse_vec4_kernel(
    const __half * __restrict__ x,
    const uint32_t * __restrict__ w,
    const __half * __restrict__ scales,
    const int32_t * __restrict__ pairs,
    const int32_t * __restrict__ experts,
    const float * __restrict__ tw,
    __half * __restrict__ intermediate,
    __half * __restrict__ down,
    int top_k, int N, int K) {
    const int e = __ldg(&experts[blockIdx.y]);
    if (e < 0)
        return;

    int pair_v[4], xr_v[4];
    __half tw_v[4];
    int valid_cnt = 0;
#pragma unroll
    for (int m = 0; m < 4; ++m) {
        const int p = __ldg(&pairs[blockIdx.y * 4 + m]);
        pair_v[m] = p;
        xr_v[m] = GATE ? (p >= 0 ? p / top_k : -1) : p;
        tw_v[m] = (!GATE && p >= 0) ? __float2half_rn(__ldg(&tw[p])) : __float2half_rn(0.0f);
        if (p >= 0) ++valid_cnt;
    }
    if (valid_cnt == 0)
        return;

    const int tid = (int)threadIdx.x;
    const int wid = tid >> 5;
    const int lane = tid & 31;
    const int k_u4_per_row = K >> 5;
    const int lpr = (k_u4_per_row >= 32) ? 32 : k_u4_per_row; // 4, 8, 16, or 32
    const int rpw = 32 / lpr;                                 // 8, 4, 2, or 1
    const int sub_lane = lane & (lpr - 1);
    const int sub_row = lane / lpr;
    const int wave_rows = 4 * rpw;
    const int base_row = (int)blockIdx.x * (4 * wave_rows) + wid * rpw + sub_row;
    if (base_row >= N)
        return;

    const int total_rows = GATE ? (N * 2) : N;
    const uint4 *w_u4 = reinterpret_cast<const uint4 *>(w);
    const uint4 *x_u4 = reinterpret_cast<const uint4 *>(x);
    const int x_u4_per_row = K >> 3;
    const int groups_per_row = K >> 7;

    if (valid_cnt == 1) {
        float g_r[4] = {};
        float u_r[4] = {};
        const size_t x0_base = (size_t)xr_v[0] * x_u4_per_row;
        for (int w_col = sub_lane; w_col < k_u4_per_row; w_col += lpr) {
            const int g_idx = w_col >> 2;
            const int x_col0 = w_col * 4;
            const uint4 xvs[4] = {
                __ldg(&x_u4[x0_base + x_col0]),
                __ldg(&x_u4[x0_base + x_col0 + 1]),
                __ldg(&x_u4[x0_base + x_col0 + 2]),
                __ldg(&x_u4[x0_base + x_col0 + 3]),
            };
#pragma unroll
            for (int ri = 0; ri < 4; ++ri) {
                const int row = base_row + ri * wave_rows;
                if (row < N) {
                    const size_t g_row_idx = (size_t)e * total_rows + row;
                    const uint4 gw4 = __ldg(&w_u4[g_row_idx * k_u4_per_row + w_col]);
                    const __half2 gs_h2 =
                        __half2half2(__ldg(&scales[g_row_idx * groups_per_row + g_idx]));
                    uint4 uw4;
                    __half2 us_h2;
                    if (GATE) {
                        const size_t u_row_idx = g_row_idx + N;
                        uw4 = __ldg(&w_u4[u_row_idx * k_u4_per_row + w_col]);
                        us_h2 = __half2half2(__ldg(&scales[u_row_idx * groups_per_row + g_idx]));
                    }
                    const uint32_t *gw_words = reinterpret_cast<const uint32_t *>(&gw4);
                    const uint32_t *uw_words = reinterpret_cast<const uint32_t *>(&uw4);
#pragma unroll
                    for (int word_i = 0; word_i < 4; ++word_i) {
                        const uint32_t gw_v = gw_words[word_i];
                        const uint32_t uw_v = GATE ? uw_words[word_i] : 0u;
                        const __half2 *xh2 = reinterpret_cast<const __half2 *>(&xvs[word_i]);
#pragma unroll
                        for (int p = 0; p < 4; ++p) {
                            const float x0 = __low2float(xh2[p]);
                            const float x1 = __high2float(xh2[p]);
                            const __half2 gd = unpack_u4b8_pair_scaled_h2(gw_v, p, gs_h2);
                            g_r[ri] += __low2float(gd) * x0 + __high2float(gd) * x1;
                            if (GATE) {
                                const __half2 ud = unpack_u4b8_pair_scaled_h2(uw_v, p, us_h2);
                                u_r[ri] += __low2float(ud) * x0 + __high2float(ud) * x1;
                            }
                        }
                    }
                }
            }
        }
        const int p0 = pair_v[0];
#pragma unroll
        for (int ri = 0; ri < 4; ++ri) {
            const int row = base_row + ri * wave_rows;
            float gv = g_r[ri];
            float uv = GATE ? u_r[ri] : 0.0f;
            if (lpr >= 32) {
                gv += __shfl_xor_sync(0xffffffffu, gv, 16);
                if (GATE) uv += __shfl_xor_sync(0xffffffffu, uv, 16);
            }
            if (lpr >= 16) {
                gv += __shfl_xor_sync(0xffffffffu, gv, 8);
                if (GATE) uv += __shfl_xor_sync(0xffffffffu, uv, 8);
            }
            if (lpr >= 8) {
                gv += __shfl_xor_sync(0xffffffffu, gv, 4);
                if (GATE) uv += __shfl_xor_sync(0xffffffffu, uv, 4);
            }
            gv += __shfl_xor_sync(0xffffffffu, gv, 2);
            gv += __shfl_xor_sync(0xffffffffu, gv, 1);
            if (GATE) {
                uv += __shfl_xor_sync(0xffffffffu, uv, 2);
                uv += __shfl_xor_sync(0xffffffffu, uv, 1);
            }
            if (sub_lane == 0 && row < N) {
                if (GATE) {
                    intermediate[(size_t)p0 * N + row] =
                        marlin_silu_and_mul_f16(__float2half_rn(gv), __float2half_rn(uv));
                } else {
                    down[(size_t)p0 * N + row] = __hmul(__float2half_rn(gv), tw_v[0]);
                }
            }
        }
        return;
    }

    for (int ri = 0; ri < 4; ++ri) {
        const int row = base_row + ri * wave_rows;
        if (row >= N)
            break;
        const size_t g_row_idx = (size_t)e * total_rows + row;
        const size_t u_row_idx = g_row_idx + N;

        for (int mb = 0; mb < valid_cnt; mb += 2) {
            const int p0 = (mb == 0) ? pair_v[0] : pair_v[2];
            const int p1 = (mb == 0) ? pair_v[1] : pair_v[3];
            const int xr0 = (mb == 0) ? xr_v[0] : xr_v[2];
            const int xr1 = (mb == 0) ? xr_v[1] : xr_v[3];
            const __half tw0 = (mb == 0) ? tw_v[0] : tw_v[2];
            const __half tw1 = (mb == 0) ? tw_v[1] : tw_v[3];
            const bool has1 = (mb + 1 < valid_cnt);
            const size_t xm0 = (size_t)xr0 * x_u4_per_row;
            const size_t xm1 = has1 ? ((size_t)xr1 * x_u4_per_row) : 0;

            float gv0 = 0.0f, uv0 = 0.0f;
            float gv1 = 0.0f, uv1 = 0.0f;

            for (int w_col = sub_lane; w_col < k_u4_per_row; w_col += lpr) {
                const int g_idx = w_col >> 2;
                const int x_col0 = w_col * 4;
                const uint4 gw4 = __ldg(&w_u4[g_row_idx * k_u4_per_row + w_col]);
                const __half2 gs_h2 =
                    __half2half2(__ldg(&scales[g_row_idx * groups_per_row + g_idx]));
                uint4 uw4;
                __half2 us_h2;
                if (GATE) {
                    uw4 = __ldg(&w_u4[u_row_idx * k_u4_per_row + w_col]);
                    us_h2 = __half2half2(__ldg(&scales[u_row_idx * groups_per_row + g_idx]));
                }
                const uint32_t *gw_words = reinterpret_cast<const uint32_t *>(&gw4);
                const uint32_t *uw_words = reinterpret_cast<const uint32_t *>(&uw4);

#pragma unroll
                for (int word_i = 0; word_i < 4; ++word_i) {
                    const uint32_t gw_v = gw_words[word_i];
                    const uint32_t uw_v = GATE ? uw_words[word_i] : 0u;
                    const uint4 xv0 = __ldg(&x_u4[xm0 + x_col0 + word_i]);
                    const uint4 xv1 = has1 ? __ldg(&x_u4[xm1 + x_col0 + word_i]) : make_uint4(0, 0, 0, 0);
                    const __half2 *xh0 = reinterpret_cast<const __half2 *>(&xv0);
                    const __half2 *xh1 = reinterpret_cast<const __half2 *>(&xv1);
#pragma unroll
                    for (int p = 0; p < 4; ++p) {
                        const __half2 gd = unpack_u4b8_pair_scaled_h2(gw_v, p, gs_h2);
                        const float gd_l = __low2float(gd), gd_h = __high2float(gd);
                        float ud_l = 0.0f, ud_h = 0.0f;
                        if (GATE) {
                            const __half2 ud = unpack_u4b8_pair_scaled_h2(uw_v, p, us_h2);
                            ud_l = __low2float(ud);
                            ud_h = __high2float(ud);
                        }
                        const float x0_l = __low2float(xh0[p]), x0_h = __high2float(xh0[p]);
                        gv0 += gd_l * x0_l + gd_h * x0_h;
                        if (GATE) uv0 += ud_l * x0_l + ud_h * x0_h;
                        if (has1) {
                            const float x1_l = __low2float(xh1[p]), x1_h = __high2float(xh1[p]);
                            gv1 += gd_l * x1_l + gd_h * x1_h;
                            if (GATE) uv1 += ud_l * x1_l + ud_h * x1_h;
                        }
                    }
                }
            }

            for (int off = lpr >> 1; off > 0; off >>= 1) {
                gv0 += __shfl_xor_sync(0xffffffffu, gv0, off);
                if (GATE) uv0 += __shfl_xor_sync(0xffffffffu, uv0, off);
                if (has1) {
                    gv1 += __shfl_xor_sync(0xffffffffu, gv1, off);
                    if (GATE) uv1 += __shfl_xor_sync(0xffffffffu, uv1, off);
                }
            }
            if (sub_lane == 0) {
                if (GATE) {
                    intermediate[(size_t)p0 * N + row] =
                        marlin_silu_and_mul_f16(__float2half_rn(gv0), __float2half_rn(uv0));
                    if (has1) {
                        intermediate[(size_t)p1 * N + row] =
                            marlin_silu_and_mul_f16(__float2half_rn(gv1), __float2half_rn(uv1));
                    }
                } else {
                    down[(size_t)p0 * N + row] = __hmul(__float2half_rn(gv0), tw0);
                    if (has1) {
                        down[(size_t)p1 * N + row] = __hmul(__float2half_rn(gv1), tw1);
                    }
                }
            }
        }
    }
}

// Sparse g128 expert tiles (128 threads = 4 warps/CTA -> 32..48 warps/SM on sm_86):
// 16 output channels occupy MMA A, eight token slots occupy B (the small-M tile
// also selected by vLLM marlin_moe).
// - GATE=true  (small N, large K): 4 warps split K 4-way (16 rows/CTA).
// - GATE=false (large N, small K): 2 row-tiles x 2 K-split warps (32 rows/CTA).
// Vectorized uint4 weight loads are exchanged within four-lane row groups and
// pre-scaled in FP16 to match donor vLLM Marlin / moe_wna16 rounding.
// N is a multiple of 64 and K a multiple of 128 at the dispatch below.
template <bool GATE>
__global__ void moe_sparse_mma_kernel(
    const __half * __restrict__ x,
    const uint32_t * __restrict__ w,
    const __half * __restrict__ scales,
    const int32_t * __restrict__ pairs,
    const int32_t * __restrict__ experts,
    const float * __restrict__ tw,
    __half * __restrict__ intermediate,
    __half * __restrict__ down,
    int top_k, int N, int K) {
    constexpr int GROUP_SIZE = 128;
    const int e = __ldg(&experts[blockIdx.y]);
    if (e < 0)
        return;
    constexpr int K_WARPS = GATE ? 4 : 2;
    constexpr int ROW_TILES = 4 / K_WARPS;
    const int wid = threadIdx.x >> 5;
    const int tile_m = GATE ? 0 : (wid >> 1);
    const int warp_k = GATE ? wid : (wid & 1);
    const int lane = threadIdx.x & 31;
    const int r = lane >> 2, c = lane & 3;
    const int base = ((int)blockIdx.x * ROW_TILES + tile_m) * 16;
    const int nr = base + r;
    const int pair = __ldg(&pairs[blockIdx.y * 8 + r]);
    const int out_p0 = __ldg(&pairs[blockIdx.y * 8 + c * 2]);
    const int out_p1 = __ldg(&pairs[blockIdx.y * 8 + c * 2 + 1]);
    const int xr = GATE ? (pair >= 0 ? pair / top_k : -1) : pair;
    const int rows = GATE ? N * 2 : N;
    float acc[4] = {}, up[4] = {};
    for (int kg = warp_k * GROUP_SIZE; kg < K; kg += K_WARPS * GROUP_SIZE) {
        uint4 gw[2], uw[2];
        __half2 gs_h2[2], us_h2[2];
#pragma unroll
        for (int j = 0; j < 2; ++j) {
            const int row = nr + j * 8;
            gw[j] = __ldg(reinterpret_cast<const uint4 *>(w + ((size_t)e * rows + row) * (K / 8) +
                                                          kg / 8 + c * 4));
            gs_h2[j] = __half2half2(
                __ldg(&scales[((size_t)e * rows + row) * (K / GROUP_SIZE) + kg / GROUP_SIZE]));
            if (GATE) {
                uw[j] = __ldg(reinterpret_cast<const uint4 *>(
                    w + ((size_t)e * rows + N + row) * (K / 8) + kg / 8 + c * 4));
                us_h2[j] = __half2half2(__ldg(
                    &scales[((size_t)e * rows + N + row) * (K / GROUP_SIZE) + kg / GROUP_SIZE]));
            }
        }
#pragma unroll
        for (int kk = 0; kk < GROUP_SIZE; kk += 16) {
            const int k = kg + kk;
            uint32_t av[4] = {}, uv[4] = {};
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                const int word = kk / 8 + (j >> 1);
                const uint32_t gv = reinterpret_cast<uint32_t *>(&gw[j & 1])[word % 4];
                av[j] = unpack_u4b8_pair_scaled(
                    __shfl_sync(0xffffffffu, gv, (lane & ~3) + word / 4), c, gs_h2[j & 1]);
                if (GATE) {
                    const uint32_t uvv = reinterpret_cast<uint32_t *>(&uw[j & 1])[word % 4];
                    uv[j] = unpack_u4b8_pair_scaled(
                        __shfl_sync(0xffffffffu, uvv, (lane & ~3) + word / 4), c, us_h2[j & 1]);
                }
            }
            const uint32_t b0 =
                xr >= 0 ? __ldg(reinterpret_cast<const uint32_t *>(x + (size_t)xr * K + k + 2 * c)) : 0;
            const uint32_t b1 =
                xr >= 0 ? __ldg(reinterpret_cast<const uint32_t *>(x + (size_t)xr * K + k + 8 + 2 * c))
                        : 0;
            asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, "
                         "{%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
                         : "+f"(acc[0]), "+f"(acc[1]), "+f"(acc[2]), "+f"(acc[3])
                         : "r"(av[0]), "r"(av[1]), "r"(av[2]), "r"(av[3]), "r"(b0), "r"(b1));
            if (GATE)
                asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, "
                             "{%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
                             : "+f"(up[0]), "+f"(up[1]), "+f"(up[2]), "+f"(up[3])
                             : "r"(uv[0]), "r"(uv[1]), "r"(uv[2]), "r"(uv[3]), "r"(b0), "r"(b1));
        }
    }
    __shared__ float partial[4][GATE ? 8 : 4][32];
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        partial[wid][j][lane] = acc[j];
        if (GATE)
            partial[wid][j + 4][lane] = up[j];
    }
    __syncthreads();
    if (warp_k != 0)
        return;
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        if (GATE) {
            acc[j] = (partial[0][j][lane] + partial[1][j][lane]) +
                     (partial[2][j][lane] + partial[3][j][lane]);
            up[j] = (partial[0][j + 4][lane] + partial[1][j + 4][lane]) +
                    (partial[2][j + 4][lane] + partial[3][j + 4][lane]);
        } else {
            const int w0 = tile_m * 2;
            acc[j] = partial[w0][j][lane] + partial[w0 + 1][j][lane];
        }
    }
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        const int p = (j & 1) ? out_p1 : out_p0;
        const int row = nr + (j / 2) * 8;
        if (p >= 0) {
            if (GATE)
                intermediate[(size_t)p * N + row] =
                    marlin_silu_and_mul_f16(__float2half_rn(acc[j]), __float2half_rn(up[j]));
            else
                down[(size_t)p * N + row] =
                    __hmul(__float2half_rn(acc[j]), __float2half_rn(__ldg(&tw[p])));
        }
    }
}

// Deterministic Stage 2b reduction in slot order (0..top_k-1) + FP16 shared_expert * sigmoid(gate).
// Grid: ((k_hidden + 511) / 512, num_tokens), Block: 128 (4 columns/thread via uint2 = 2x __half2).
__global__ void moe_down_topk_reduce_f16_kernel(
    const __half* __restrict__ stage2_pair_out,
    const int32_t* __restrict__ topk_ids,
    const __half* __restrict__ shared_expert_out,
    const __half* __restrict__ shared_expert_gate_logits,
    __half* __restrict__ out,
    int num_tokens,
    int top_k,
    int num_experts,
    int k_hidden) {
    const int tok = (int)blockIdx.y;
    if (tok >= num_tokens) return;
    const int col4 = (int)blockIdx.x * 512 + (int)threadIdx.x * 4;
    if (col4 + 3 >= k_hidden) return;

    float s0 = 0.0f, s1 = 0.0f, s2 = 0.0f, s3 = 0.0f;
    for (int slot = 0; slot < top_k; ++slot) {
        const int pair_idx = tok * top_k + slot;
        const int e = topk_ids[pair_idx];
        if (e >= 0 && e < num_experts) {
            const uint2 v =
                *reinterpret_cast<const uint2*>(&stage2_pair_out[(size_t)pair_idx * k_hidden + col4]);
            const __half2 vlo = *reinterpret_cast<const __half2*>(&v.x);
            const __half2 vhi = *reinterpret_cast<const __half2*>(&v.y);
            s0 += __low2float(vlo);
            s1 += __high2float(vlo);
            s2 += __low2float(vhi);
            s3 += __high2float(vhi);
        }
    }

    __half2 out_lo = __floats2half2_rn(s0, s1);
    __half2 out_hi = __floats2half2_rn(s2, s3);

    if (shared_expert_out != nullptr) {
        const uint2 sh2 =
            *reinterpret_cast<const uint2*>(&shared_expert_out[(size_t)tok * k_hidden + col4]);
        __half2 sh_lo = *reinterpret_cast<const __half2*>(&sh2.x);
        __half2 sh_hi = *reinterpret_cast<const __half2*>(&sh2.y);
        if (shared_expert_gate_logits != nullptr) {
            const float gl = __half2float(shared_expert_gate_logits[tok]);
            const __half sig_h = __float2half_rn(__fdiv_rn(1.0f, 1.0f + __nv_expf(-gl)));
            const __half2 sig_h2 = __half2half2(sig_h);
            sh_lo = __hmul2(sig_h2, sh_lo);
            sh_hi = __hmul2(sig_h2, sh_hi);
        }
        out_lo = __hadd2(out_lo, sh_lo);
        out_hi = __hadd2(out_hi, sh_hi);
    }

    uint2 out_u2;
    out_u2.x = *reinterpret_cast<const uint32_t*>(&out_lo);
    out_u2.y = *reinterpret_cast<const uint32_t*>(&out_hi);
    *reinterpret_cast<uint2*>(&out[(size_t)tok * k_hidden + col4]) = out_u2;
}

template <int BLOCK_M>
static inline int compute_moe_max_tiles(int total_pairs, int num_experts) {
    const int e_star = (total_pairs < num_experts) ? total_pairs : num_experts;
    return e_star + ((total_pairs - e_star) / BLOCK_M);
}

static inline bool can_use_grouped_mma_moe_shape(
    const w34_exec* ex,
    int num_tokens,
    int total_pairs,
    int num_experts,
    int n_interm,
    int k_hidden,
    int group_size) {
    return num_tokens >= 4 &&
           total_pairs >= num_experts &&
           num_experts > 0 && num_experts <= 1024 &&
           (n_interm % 64) == 0 && (n_interm % group_size) == 0 &&
           (k_hidden % 64) == 0 && (k_hidden % group_size) == 0 &&
           (group_size % 16) == 0 &&
           ex->scratch_ptr != 0 &&
           (ex->scratch_ptr & 15u) == 0u;
}

static inline bool can_use_grouped_mma_moe(
    const w34_exec* ex,
    int num_tokens,
    int total_pairs,
    int num_experts,
    int n_interm,
    int k_hidden,
    int group_size,
    size_t needed_bytes) {
    return can_use_grouped_mma_moe_shape(
               ex, num_tokens, total_pairs, num_experts, n_interm, k_hidden, group_size) &&
           ex->scratch_size >= needed_bytes;
}

static inline size_t grouped_mma_stage2_scratch_bytes(
    int chunk_tokens,
    int top_k,
    int num_experts,
    int k_hidden,
    int* out_max_tiles16,
    size_t* out_pairs16,
    size_t* out_tiles16) {
    const int cp = chunk_tokens * top_k;
    const int mt = compute_moe_max_tiles<16>(cp, num_experts);
    const size_t p16 = (size_t)mt * 16u * sizeof(int32_t);
    const size_t t16 = (((size_t)mt * sizeof(int32_t)) + 15u) & ~15u;
    const size_t s2 = (size_t)cp * (size_t)k_hidden * sizeof(__half);
    if (out_max_tiles16) *out_max_tiles16 = mt;
    if (out_pairs16) *out_pairs16 = p16;
    if (out_tiles16) *out_tiles16 = t16;
    return p16 + t16 + s2;
}

// Returns balanced chunk count >= 1 if `num_tokens` can be partitioned into
// Grouped-MMA-eligible chunks that each fit in `ex->scratch_size` with zero
// extra VRAM allocation, or 0 if scratch is too small even for a minimal chunk.
static inline int plan_grouped_mma_chunks(
    const w34_exec* ex,
    int num_tokens,
    int top_k,
    int num_experts,
    int n_interm,
    int k_hidden,
    int group_size) {
    if (!can_use_grouped_mma_moe_shape(
            ex, num_tokens, num_tokens * top_k, num_experts, n_interm, k_hidden, group_size)) {
        return 0;
    }
    const int min_chunk_m = (num_experts + top_k - 1) / top_k;
    const int floor_m = (min_chunk_m > 4) ? min_chunk_m : 4;
    if (grouped_mma_stage2_scratch_bytes(floor_m, top_k, num_experts, k_hidden, nullptr, nullptr, nullptr) >
        ex->scratch_size) {
        return 0;
    }
    const int max_chunks = num_tokens / floor_m;
    for (int chunks = 1; chunks <= max_chunks; ++chunks) {
        const int ceil_m = (num_tokens + chunks - 1) / chunks;
        if (grouped_mma_stage2_scratch_bytes(ceil_m, top_k, num_experts, k_hidden, nullptr, nullptr, nullptr) <=
            ex->scratch_size) {
            return chunks;
        }
    }
    return 0;
}

} // namespace w34_moe

extern "C" int w34_moe_topk_router_f16(
    const w34_exec* ex,
    const void* router_logits,
    float* topk_weights,
    int32_t* topk_ids,
    int num_tokens,
    int num_experts,
    int top_k,
    int norm_topk_prob,
    int scoring_func) {
    if (ex == nullptr || router_logits == nullptr || topk_weights == nullptr || topk_ids == nullptr ||
        num_tokens < 0 || num_experts <= 0 || num_experts > 1024 || top_k <= 0 || top_k > 32) {
        return (int)cudaErrorInvalidValue;
    }
    if (num_tokens == 0) return 0;
    if (num_experts == 256 && top_k == 8 && (((uintptr_t)router_logits & 15u) == 0u)) {
        w34_moe::moe_topk_router_256_k8_f16_kernel<<<(unsigned)num_tokens, 32, 0, (cudaStream_t)ex->stream>>>(
            (const __half*)router_logits, topk_weights, topk_ids, num_tokens, norm_topk_prob, scoring_func);
    } else if (num_experts <= 256) {
        w34_moe::moe_topk_router_warp256_f16_kernel<<<(unsigned)num_tokens, 32, 0, (cudaStream_t)ex->stream>>>(
            (const __half*)router_logits, topk_weights, topk_ids, num_tokens, num_experts, top_k,
            norm_topk_prob, scoring_func);
    } else {
        w34_moe::moe_topk_router_f16_kernel<<<(unsigned)num_tokens, 128, 0, (cudaStream_t)ex->stream>>>(
            (const __half*)router_logits, topk_weights, topk_ids, num_tokens, num_experts, top_k,
            norm_topk_prob, scoring_func);
    }
    return (int)cudaGetLastError();
}

extern "C" int w34_moe_w4a16_gate_up_silu_f16(
    const w34_exec* ex,
    const void* x,
    const uint32_t* gate_up_packed,
    const void* gate_up_scales,
    const int32_t* topk_ids,
    void* interm_out,
    int num_tokens,
    int top_k,
    int num_experts,
    int n_interm,
    int k_hidden,
    int group_size) {
    if (ex == nullptr || x == nullptr || gate_up_packed == nullptr || gate_up_scales == nullptr ||
        topk_ids == nullptr || interm_out == nullptr || num_tokens < 0 || top_k <= 0 ||
        num_experts <= 0 || n_interm <= 0 || k_hidden <= 0 || (k_hidden % 8) != 0 ||
        group_size <= 0 || (group_size % 8) != 0) {
        return (int)cudaErrorInvalidValue;
    }
    if (num_tokens == 0) return 0;
    const int total_pairs = num_tokens * top_k;
    const int max_tiles16 = w34_moe::compute_moe_max_tiles<16>(total_pairs, num_experts);
    const size_t pairs16 = (size_t)max_tiles16 * 16u * sizeof(int32_t);
    const size_t tiles16 = (((size_t)max_tiles16 * sizeof(int32_t)) + 15u) & ~15u;
    if (w34_moe::can_use_grouped_mma_moe(
            ex, num_tokens, total_pairs, num_experts, n_interm, k_hidden, group_size,
            pairs16 + tiles16)) {
        uint8_t* ws = reinterpret_cast<uint8_t*>((uintptr_t)ex->scratch_ptr);
        int32_t* sorted_pair_ids = reinterpret_cast<int32_t*>(ws);
        int32_t* tile_expert_ids = reinterpret_cast<int32_t*>(ws + pairs16);
        cudaStream_t st = (cudaStream_t)ex->stream;
        w34_moe::moe_align_block_size_kernel<16><<<1, 256, 0, st>>>(
            topk_ids, sorted_pair_ids, tile_expert_ids, total_pairs, num_experts, max_tiles16);
        dim3 grid((unsigned)((n_interm + 31) / 32), (unsigned)max_tiles16);
        w34_moe::moe_w4a16_grouped_gate_up_silu_mma_f16_kernel<false><<<grid, 128, 0, st>>>(
            (const __half*)x, gate_up_packed, (const __half*)gate_up_scales, nullptr,
            sorted_pair_ids, tile_expert_ids, (__half*)interm_out,
            top_k, n_interm, k_hidden, group_size);
        return (int)cudaGetLastError();
    }
    dim3 grid((unsigned)((n_interm + 3) / 4), (unsigned)(num_tokens * top_k));
    w34_moe::moe_w4a16_gate_up_silu_f16_kernel<false><<<grid, 128, 0, (cudaStream_t)ex->stream>>>(
        (const __half*)x, gate_up_packed, (const __half*)gate_up_scales, nullptr, topk_ids,
        (__half*)interm_out, num_tokens, top_k, num_experts, n_interm, k_hidden, group_size);
    return (int)cudaGetLastError();
}

extern "C" int w34_moe_w4a16_down_accum_f16(
    const w34_exec* ex,
    const void* interm_in,
    const uint32_t* down_packed,
    const void* down_scales,
    const int32_t* topk_ids,
    const float* topk_weights,
    const void* shared_expert_out,
    const void* shared_expert_gate_logits,
    void* out,
    int num_tokens,
    int top_k,
    int num_experts,
    int k_hidden,
    int n_interm,
    int group_size) {
    if (ex == nullptr || interm_in == nullptr || down_packed == nullptr || down_scales == nullptr ||
        topk_ids == nullptr || topk_weights == nullptr || out == nullptr || num_tokens < 0 ||
        top_k <= 0 || num_experts <= 0 || k_hidden <= 0 || n_interm <= 0 || (n_interm % 8) != 0 ||
        group_size <= 0 || (group_size % 8) != 0) {
        return (int)cudaErrorInvalidValue;
    }
    if (num_tokens == 0) return 0;
    const int num_chunks = w34_moe::plan_grouped_mma_chunks(
        ex, num_tokens, top_k, num_experts, n_interm, k_hidden, group_size);
    if (num_chunks > 0) {
        uint8_t* ws = reinterpret_cast<uint8_t*>((uintptr_t)ex->scratch_ptr);
        cudaStream_t st = (cudaStream_t)ex->stream;
        const int base_m = num_tokens / num_chunks;
        const int rem_m = num_tokens % num_chunks;
        int tok_off = 0;
        for (int c = 0; c < num_chunks; ++c) {
            const int chunk_m = base_m + ((c < rem_m) ? 1 : 0);
            int max_tiles16 = 0;
            size_t pairs16 = 0, tiles16 = 0;
            w34_moe::grouped_mma_stage2_scratch_bytes(
                chunk_m, top_k, num_experts, k_hidden, &max_tiles16, &pairs16, &tiles16);
            int32_t* sorted_pair_ids = reinterpret_cast<int32_t*>(ws);
            int32_t* tile_expert_ids = reinterpret_cast<int32_t*>(ws + pairs16);
            __half* stage2_pair_out = reinterpret_cast<__half*>(ws + pairs16 + tiles16);
            const int chunk_pairs = chunk_m * top_k;
            const int32_t* ids_chunk = topk_ids + (size_t)tok_off * top_k;
            const float* tw_chunk = topk_weights + (size_t)tok_off * top_k;
            const __half* interm_chunk =
                (const __half*)interm_in + (size_t)tok_off * top_k * n_interm;
            const __half* sh_out_chunk = (shared_expert_out != nullptr)
                ? ((const __half*)shared_expert_out + (size_t)tok_off * k_hidden)
                : nullptr;
            const __half* sh_gate_chunk = (shared_expert_gate_logits != nullptr)
                ? ((const __half*)shared_expert_gate_logits + tok_off)
                : nullptr;
            __half* out_chunk = (__half*)out + (size_t)tok_off * k_hidden;

            w34_moe::moe_align_block_size_kernel<16><<<1, 256, 0, st>>>(
                ids_chunk, sorted_pair_ids, tile_expert_ids, chunk_pairs, num_experts, max_tiles16);
            dim3 g_down((unsigned)((k_hidden + 63) / 64), (unsigned)max_tiles16);
            w34_moe::moe_w4a16_grouped_down_mma_f16_kernel<false><<<g_down, 128, 0, st>>>(
                interm_chunk, down_packed, (const __half*)down_scales, nullptr,
                sorted_pair_ids, tile_expert_ids, tw_chunk, stage2_pair_out,
                k_hidden, n_interm, group_size);
            dim3 g_red((unsigned)((k_hidden + 511) / 512), (unsigned)chunk_m);
            w34_moe::moe_down_topk_reduce_f16_kernel<<<g_red, 128, 0, st>>>(
                stage2_pair_out, ids_chunk, sh_out_chunk, sh_gate_chunk, out_chunk,
                chunk_m, top_k, num_experts, k_hidden);
            tok_off += chunk_m;
        }
        return (int)cudaGetLastError();
    }
    dim3 grid((unsigned)((k_hidden + 3) / 4), (unsigned)num_tokens);
    if (n_interm == 256 && top_k == 8) {
        w34_moe::moe_w4a16_down_accum_f16_kernel<false, 32, 8><<<grid, 128, 0, (cudaStream_t)ex->stream>>>(
            (const __half*)interm_in, down_packed, (const __half*)down_scales, nullptr, topk_ids, topk_weights,
            (const __half*)shared_expert_out, (const __half*)shared_expert_gate_logits, (__half*)out,
            num_tokens, top_k, num_experts, k_hidden, n_interm, group_size);
    } else if (n_interm == 512 && top_k == 8) {
        w34_moe::moe_w4a16_down_accum_f16_kernel<false, 64, 8><<<grid, 128, 0, (cudaStream_t)ex->stream>>>(
            (const __half*)interm_in, down_packed, (const __half*)down_scales, nullptr, topk_ids, topk_weights,
            (const __half*)shared_expert_out, (const __half*)shared_expert_gate_logits, (__half*)out,
            num_tokens, top_k, num_experts, k_hidden, n_interm, group_size);
    } else {
        w34_moe::moe_w4a16_down_accum_f16_kernel<false><<<grid, 128, 0, (cudaStream_t)ex->stream>>>(
            (const __half*)interm_in, down_packed, (const __half*)down_scales, nullptr, topk_ids, topk_weights,
            (const __half*)shared_expert_out, (const __half*)shared_expert_gate_logits, (__half*)out,
            num_tokens, top_k, num_experts, k_hidden, n_interm, group_size);
    }
    return (int)cudaGetLastError();
}

extern "C" int w34_moe_w4a16_select_regime(
    const w34_exec* ex,
    int num_tokens,
    int top_k,
    int num_experts,
    int n_interm,
    int k_hidden,
    int group_size) {
    if (ex == nullptr || num_tokens <= 0 || top_k <= 0 || num_experts <= 0 ||
        n_interm <= 0 || k_hidden <= 0 || group_size <= 0) {
        return 0;
    }
    if (num_tokens >= 4 && num_tokens <= 7 && num_tokens * top_k <= num_experts &&
        num_experts <= 1024 && n_interm % 64 == 0 && k_hidden % 64 == 0 &&
        n_interm % group_size == 0 && k_hidden % group_size == 0 && group_size == 128) {
        const int np = num_tokens * top_k;
        const int nt = w34_moe::compute_moe_max_tiles<4>(np, num_experts);
        const size_t pb = nt * 4 * sizeof(int32_t), eb = (nt * sizeof(int32_t) + 15) & ~size_t(15);
        if (ex->scratch_ptr && !(ex->scratch_ptr & 15u) &&
            ex->scratch_size >= pb + eb + (size_t)np * k_hidden * sizeof(__half)) {
            return 1;
        }
    }
    if (num_tokens >= 8 && num_tokens <= 128 && num_tokens * top_k <= 4 * num_experts &&
        num_experts <= 1024 && n_interm % 64 == 0 && k_hidden % 64 == 0 &&
        n_interm % group_size == 0 && k_hidden % group_size == 0 && group_size == 128) {
        const int np = num_tokens * top_k;
        const int nt = w34_moe::compute_moe_max_tiles<8>(np, num_experts);
        const size_t pb = nt * 8 * sizeof(int32_t), eb = (nt * sizeof(int32_t) + 15) & ~size_t(15);
        if (ex->scratch_ptr && !(ex->scratch_ptr & 15u) &&
            ex->scratch_size >= pb + eb + (size_t)np * k_hidden * sizeof(__half)) {
            return 2;
        }
    }
    if (w34_moe::plan_grouped_mma_chunks(
            ex, num_tokens, top_k, num_experts, n_interm, k_hidden, group_size) > 0) {
        return 3;
    }
    return 0;
}

extern "C" int w34_moe_w4a16_mlp_f16(
    const w34_exec* ex,
    const void* x,
    const uint32_t* gate_up_packed,
    const void* gate_up_scales,
    const uint32_t* down_packed,
    const void* down_scales,
    const int32_t* topk_ids,
    const float* topk_weights,
    const void* shared_expert_out,
    const void* shared_expert_gate_logits,
    void* interm_buf,
    void* out,
    int num_tokens,
    int top_k,
    int num_experts,
    int n_interm,
    int k_hidden,
    int group_size) {
    if (ex == nullptr || x == nullptr || gate_up_packed == nullptr || gate_up_scales == nullptr ||
        down_packed == nullptr || down_scales == nullptr || topk_ids == nullptr ||
        topk_weights == nullptr || interm_buf == nullptr || out == nullptr || num_tokens < 0 ||
        top_k <= 0 || num_experts <= 0 || n_interm <= 0 || k_hidden <= 0 ||
        (k_hidden % 8) != 0 || (n_interm % 8) != 0 || group_size <= 0 || (group_size % 8) != 0) {
        return (int)cudaErrorInvalidValue;
    }
    if (num_tokens == 0) return 0;

    const int regime = w34_moe_w4a16_select_regime(
        ex, num_tokens, top_k, num_experts, n_interm, k_hidden, group_size);

    // Coalesced sub-warp uint4 expert-grouped regime for M = 4..7 (preserving GEMV for M = 1..3).
    if (regime == 1) {
        const int np = num_tokens * top_k;
        const int nt = w34_moe::compute_moe_max_tiles<4>(np, num_experts);
        const size_t pb = nt * 4 * sizeof(int32_t), eb = (nt * sizeof(int32_t) + 15) & ~size_t(15);
        auto pairs = reinterpret_cast<int32_t *>((uintptr_t)ex->scratch_ptr);
        auto experts = reinterpret_cast<int32_t *>((char *)pairs + pb);
        auto down = reinterpret_cast<__half *>((char *)experts + eb);
        cudaStream_t st = (cudaStream_t)ex->stream;
        const int lpr_gu = ((k_hidden >> 5) >= 32) ? 32 : (k_hidden >> 5);
        const int cta_rows_gu = 4 * 4 * (32 / lpr_gu);
        const int lpr_dn = ((n_interm >> 5) >= 32) ? 32 : (n_interm >> 5);
        const int cta_rows_dn = 4 * 4 * (32 / lpr_dn);
        w34_moe::moe_align_block_size_kernel<4>
            <<<1, 256, 0, st>>>(topk_ids, pairs, experts, np, num_experts, nt);
        w34_moe::moe_sparse_vec4_kernel<true>
            <<<dim3((n_interm + cta_rows_gu - 1) / cta_rows_gu, nt), 128, 0, st>>>(
                (const __half *)x, gate_up_packed, (const __half *)gate_up_scales, pairs,
                experts, topk_weights, (__half *)interm_buf, down, top_k, n_interm, k_hidden);
        w34_moe::moe_sparse_vec4_kernel<false>
            <<<dim3((k_hidden + cta_rows_dn - 1) / cta_rows_dn, nt), 128, 0, st>>>(
                (const __half *)interm_buf, down_packed, (const __half *)down_scales, pairs,
                experts, topk_weights, (__half *)interm_buf, down, top_k, k_hidden, n_interm);
        w34_moe::moe_down_topk_reduce_f16_kernel<<<dim3((k_hidden + 511) / 512, num_tokens),
                                                   128, 0, st>>>(
            down, topk_ids, (const __half *)shared_expert_out,
            (const __half *)shared_expert_gate_logits, (__half *)out, num_tokens, top_k,
            num_experts, k_hidden);
        return (int)cudaGetLastError();
    }

    // Sparse token-grouped mma.sync regime for M = 8..128 when total_pairs <= 4 * num_experts.
    if (regime == 2) {
        const int np = num_tokens * top_k;
        const int nt = w34_moe::compute_moe_max_tiles<8>(np, num_experts);
        const size_t pb = nt * 8 * sizeof(int32_t), eb = (nt * sizeof(int32_t) + 15) & ~size_t(15);
        auto pairs = reinterpret_cast<int32_t *>((uintptr_t)ex->scratch_ptr);
        auto experts = reinterpret_cast<int32_t *>((char *)pairs + pb);
        auto down = reinterpret_cast<__half *>((char *)experts + eb);
        cudaStream_t st = (cudaStream_t)ex->stream;
        w34_moe::moe_align_block_size_kernel<8>
            <<<1, 256, 0, st>>>(topk_ids, pairs, experts, np, num_experts, nt);
        w34_moe::moe_sparse_mma_kernel<true><<<dim3(n_interm / 16, nt), 128, 0, st>>>(
            (const __half *)x, gate_up_packed, (const __half *)gate_up_scales, pairs, experts,
            topk_weights, (__half *)interm_buf, down, top_k, n_interm, k_hidden);
        w34_moe::moe_sparse_mma_kernel<false><<<dim3(k_hidden / 32, nt), 128, 0, st>>>(
            (const __half *)interm_buf, down_packed, (const __half *)down_scales, pairs,
            experts, topk_weights, (__half *)interm_buf, down, top_k, k_hidden, n_interm);
        w34_moe::moe_down_topk_reduce_f16_kernel<<<dim3((k_hidden + 511) / 512, num_tokens),
                                                   128, 0, st>>>(
            down, topk_ids, (const __half *)shared_expert_out,
            (const __half *)shared_expert_gate_logits, (__half *)out, num_tokens, top_k,
            num_experts, k_hidden);
        return (int)cudaGetLastError();
    }

    if (regime == 3) {
        // Prefer coalesced BLOCK_M=8 sparse_mma chunks whenever geometry matches g128
        // (1.7-2x faster than BLOCK_M=16 strided grouped_mma on 256-expert MoE prefill).
        if (num_experts <= 1024 && (n_interm % 64) == 0 && (k_hidden % 64) == 0 &&
            (n_interm % group_size) == 0 && (k_hidden % group_size) == 0 && group_size == 128 &&
            ex->scratch_ptr && !(ex->scratch_ptr & 15u)) {
            int sparse_chunks = 0;
            for (int cand = 1; cand <= num_tokens; ++cand) {
                const int cm = (num_tokens + cand - 1) / cand;
                const int np = cm * top_k;
                const int nt = w34_moe::compute_moe_max_tiles<8>(np, num_experts);
                const size_t pb = (size_t)nt * 8 * sizeof(int32_t);
                const size_t eb = ((size_t)nt * sizeof(int32_t) + 15) & ~size_t(15);
                const size_t need = pb + eb + (size_t)np * k_hidden * sizeof(__half);
                if (need <= ex->scratch_size) {
                    sparse_chunks = cand;
                    break;
                }
            }
            if (sparse_chunks > 0) {
                cudaStream_t st = (cudaStream_t)ex->stream;
                const int base_m = num_tokens / sparse_chunks;
                const int rem_m = num_tokens % sparse_chunks;
                int tok_off = 0;
                for (int c = 0; c < sparse_chunks; ++c) {
                    const int cm = base_m + ((c < rem_m) ? 1 : 0);
                    const int np = cm * top_k;
                    const int nt = w34_moe::compute_moe_max_tiles<8>(np, num_experts);
                    const size_t pb = (size_t)nt * 8 * sizeof(int32_t);
                    const size_t eb = ((size_t)nt * sizeof(int32_t) + 15) & ~size_t(15);
                    auto pairs = reinterpret_cast<int32_t *>((uintptr_t)ex->scratch_ptr);
                    auto experts = reinterpret_cast<int32_t *>((char *)pairs + pb);
                    auto down = reinterpret_cast<__half *>((char *)experts + eb);
                    const __half *x_c = (const __half *)x + (size_t)tok_off * k_hidden;
                    const int32_t *ids_c = topk_ids + (size_t)tok_off * top_k;
                    const float *tw_c = topk_weights + (size_t)tok_off * top_k;
                    __half *interm_c = (__half *)interm_buf + (size_t)tok_off * top_k * n_interm;
                    const __half *sh_out_c = (shared_expert_out != nullptr)
                        ? ((const __half *)shared_expert_out + (size_t)tok_off * k_hidden)
                        : nullptr;
                    const __half *sh_gate_c = (shared_expert_gate_logits != nullptr)
                        ? ((const __half *)shared_expert_gate_logits + tok_off)
                        : nullptr;
                    __half *out_c = (__half *)out + (size_t)tok_off * k_hidden;

                    w34_moe::moe_align_block_size_kernel<8>
                        <<<1, 256, 0, st>>>(ids_c, pairs, experts, np, num_experts, nt);
                    w34_moe::moe_sparse_mma_kernel<true><<<dim3(n_interm / 16, nt), 128, 0, st>>>(
                        x_c, gate_up_packed, (const __half *)gate_up_scales, pairs, experts,
                        tw_c, interm_c, down, top_k, n_interm, k_hidden);
                    w34_moe::moe_sparse_mma_kernel<false><<<dim3(k_hidden / 32, nt), 128, 0, st>>>(
                        interm_c, down_packed, (const __half *)down_scales, pairs, experts,
                        tw_c, interm_c, down, top_k, k_hidden, n_interm);
                    w34_moe::moe_down_topk_reduce_f16_kernel
                        <<<dim3((k_hidden + 511) / 512, cm), 128, 0, st>>>(
                            down, ids_c, sh_out_c, sh_gate_c, out_c, cm, top_k, num_experts,
                            k_hidden);
                    tok_off += cm;
                }
                return (int)cudaGetLastError();
            }
        }
        const int num_chunks = w34_moe::plan_grouped_mma_chunks(
            ex, num_tokens, top_k, num_experts, n_interm, k_hidden, group_size);
        uint8_t* ws = reinterpret_cast<uint8_t*>((uintptr_t)ex->scratch_ptr);
        cudaStream_t st = (cudaStream_t)ex->stream;
        const int base_m = num_tokens / num_chunks;
        const int rem_m = num_tokens % num_chunks;
        int tok_off = 0;
        for (int c = 0; c < num_chunks; ++c) {
            const int chunk_m = base_m + ((c < rem_m) ? 1 : 0);
            int max_tiles16 = 0;
            size_t pairs16 = 0, tiles16 = 0;
            w34_moe::grouped_mma_stage2_scratch_bytes(
                chunk_m, top_k, num_experts, k_hidden, &max_tiles16, &pairs16, &tiles16);
            int32_t* sorted_pair_ids = reinterpret_cast<int32_t*>(ws);
            int32_t* tile_expert_ids = reinterpret_cast<int32_t*>(ws + pairs16);
            __half* stage2_pair_out = reinterpret_cast<__half*>(ws + pairs16 + tiles16);
            const int chunk_pairs = chunk_m * top_k;
            const __half* x_chunk = (const __half*)x + (size_t)tok_off * k_hidden;
            const int32_t* ids_chunk = topk_ids + (size_t)tok_off * top_k;
            const float* tw_chunk = topk_weights + (size_t)tok_off * top_k;
            __half* interm_chunk = (__half*)interm_buf + (size_t)tok_off * top_k * n_interm;
            const __half* sh_out_chunk = (shared_expert_out != nullptr)
                ? ((const __half*)shared_expert_out + (size_t)tok_off * k_hidden)
                : nullptr;
            const __half* sh_gate_chunk = (shared_expert_gate_logits != nullptr)
                ? ((const __half*)shared_expert_gate_logits + tok_off)
                : nullptr;
            __half* out_chunk = (__half*)out + (size_t)tok_off * k_hidden;

            w34_moe::moe_align_block_size_kernel<16><<<1, 256, 0, st>>>(
                ids_chunk, sorted_pair_ids, tile_expert_ids, chunk_pairs, num_experts, max_tiles16);
            dim3 g_gu((unsigned)((n_interm + 31) / 32), (unsigned)max_tiles16);
            w34_moe::moe_w4a16_grouped_gate_up_silu_mma_f16_kernel<false><<<g_gu, 128, 0, st>>>(
                x_chunk, gate_up_packed, (const __half*)gate_up_scales, nullptr,
                sorted_pair_ids, tile_expert_ids, interm_chunk,
                top_k, n_interm, k_hidden, group_size);
            dim3 g_down((unsigned)((k_hidden + 63) / 64), (unsigned)max_tiles16);
            w34_moe::moe_w4a16_grouped_down_mma_f16_kernel<false><<<g_down, 128, 0, st>>>(
                interm_chunk, down_packed, (const __half*)down_scales, nullptr,
                sorted_pair_ids, tile_expert_ids, tw_chunk, stage2_pair_out,
                k_hidden, n_interm, group_size);
            dim3 g_red((unsigned)((k_hidden + 511) / 512), (unsigned)chunk_m);
            w34_moe::moe_down_topk_reduce_f16_kernel<<<g_red, 128, 0, st>>>(
                stage2_pair_out, ids_chunk, sh_out_chunk, sh_gate_chunk, out_chunk,
                chunk_m, top_k, num_experts, k_hidden);
            tok_off += chunk_m;
        }
        return (int)cudaGetLastError();
    }

    cudaStream_t st = (cudaStream_t)ex->stream;
    dim3 g1((unsigned)((n_interm + 3) / 4), (unsigned)(num_tokens * top_k));
    w34_moe::moe_w4a16_gate_up_silu_f16_kernel<false><<<g1, 128, 0, st>>>(
        (const __half*)x, gate_up_packed, (const __half*)gate_up_scales, nullptr, topk_ids,
        (__half*)interm_buf, num_tokens, top_k, num_experts, n_interm, k_hidden, group_size);
    dim3 g2((unsigned)((k_hidden + 3) / 4), (unsigned)num_tokens);
    if (n_interm == 256 && top_k == 8) {
        w34_moe::moe_w4a16_down_accum_f16_kernel<false, 32, 8><<<g2, 128, 0, st>>>(
            (const __half*)interm_buf, down_packed, (const __half*)down_scales, nullptr, topk_ids, topk_weights,
            (const __half*)shared_expert_out, (const __half*)shared_expert_gate_logits, (__half*)out,
            num_tokens, top_k, num_experts, k_hidden, n_interm, group_size);
    } else if (n_interm == 512 && top_k == 8) {
        w34_moe::moe_w4a16_down_accum_f16_kernel<false, 64, 8><<<g2, 128, 0, st>>>(
            (const __half*)interm_buf, down_packed, (const __half*)down_scales, nullptr, topk_ids, topk_weights,
            (const __half*)shared_expert_out, (const __half*)shared_expert_gate_logits, (__half*)out,
            num_tokens, top_k, num_experts, k_hidden, n_interm, group_size);
    } else {
        w34_moe::moe_w4a16_down_accum_f16_kernel<false><<<g2, 128, 0, st>>>(
            (const __half*)interm_buf, down_packed, (const __half*)down_scales, nullptr, topk_ids, topk_weights,
            (const __half*)shared_expert_out, (const __half*)shared_expert_gate_logits, (__half*)out,
            num_tokens, top_k, num_experts, k_hidden, n_interm, group_size);
    }
    return (int)cudaGetLastError();
}

extern "C" int w34_moe_w4a16_mlp_zp_f16(
    const w34_exec* ex,
    const void* x,
    const uint32_t* gate_up_packed,
    const void* gate_up_scales,
    const uint32_t* gate_up_qzeros,
    const uint32_t* down_packed,
    const void* down_scales,
    const uint32_t* down_qzeros,
    const int32_t* topk_ids,
    const float* topk_weights,
    const void* shared_expert_out,
    const void* shared_expert_gate_logits,
    void* interm_buf,
    void* out,
    int num_tokens,
    int top_k,
    int num_experts,
    int n_interm,
    int k_hidden,
    int group_size) {
    if (ex == nullptr || x == nullptr || gate_up_packed == nullptr || gate_up_scales == nullptr ||
        gate_up_qzeros == nullptr || down_packed == nullptr || down_scales == nullptr ||
        down_qzeros == nullptr || topk_ids == nullptr || topk_weights == nullptr ||
        interm_buf == nullptr || out == nullptr || num_tokens < 0 || top_k <= 0 ||
        num_experts <= 0 || n_interm <= 0 || k_hidden <= 0 || (k_hidden % 8) != 0 ||
        (n_interm % 8) != 0 || group_size <= 0 || (group_size % 8) != 0 ||
        (k_hidden % group_size) != 0 || (n_interm % group_size) != 0) {
        return (int)cudaErrorInvalidValue;
    }
    if (num_tokens == 0) return 0;

    const int num_chunks = w34_moe::plan_grouped_mma_chunks(
        ex, num_tokens, top_k, num_experts, n_interm, k_hidden, group_size);
    if (num_chunks > 0) {
        uint8_t* ws = reinterpret_cast<uint8_t*>((uintptr_t)ex->scratch_ptr);
        cudaStream_t st = (cudaStream_t)ex->stream;
        const int base_m = num_tokens / num_chunks;
        const int rem_m = num_tokens % num_chunks;
        int tok_off = 0;
        for (int c = 0; c < num_chunks; ++c) {
            const int chunk_m = base_m + ((c < rem_m) ? 1 : 0);
            int max_tiles16 = 0;
            size_t pairs16 = 0, tiles16 = 0;
            w34_moe::grouped_mma_stage2_scratch_bytes(
                chunk_m, top_k, num_experts, k_hidden, &max_tiles16, &pairs16, &tiles16);
            int32_t* sorted_pair_ids = reinterpret_cast<int32_t*>(ws);
            int32_t* tile_expert_ids = reinterpret_cast<int32_t*>(ws + pairs16);
            __half* stage2_pair_out = reinterpret_cast<__half*>(ws + pairs16 + tiles16);
            const int chunk_pairs = chunk_m * top_k;
            const __half* x_chunk = (const __half*)x + (size_t)tok_off * k_hidden;
            const int32_t* ids_chunk = topk_ids + (size_t)tok_off * top_k;
            const float* tw_chunk = topk_weights + (size_t)tok_off * top_k;
            __half* interm_chunk = (__half*)interm_buf + (size_t)tok_off * top_k * n_interm;
            const __half* sh_out_chunk = (shared_expert_out != nullptr)
                ? ((const __half*)shared_expert_out + (size_t)tok_off * k_hidden)
                : nullptr;
            const __half* sh_gate_chunk = (shared_expert_gate_logits != nullptr)
                ? ((const __half*)shared_expert_gate_logits + tok_off)
                : nullptr;
            __half* out_chunk = (__half*)out + (size_t)tok_off * k_hidden;

            w34_moe::moe_align_block_size_kernel<16><<<1, 256, 0, st>>>(
                ids_chunk, sorted_pair_ids, tile_expert_ids, chunk_pairs, num_experts, max_tiles16);
            dim3 g_gu((unsigned)((n_interm + 31) / 32), (unsigned)max_tiles16);
            w34_moe::moe_w4a16_grouped_gate_up_silu_mma_f16_kernel<true><<<g_gu, 128, 0, st>>>(
                x_chunk, gate_up_packed, (const __half*)gate_up_scales, gate_up_qzeros,
                sorted_pair_ids, tile_expert_ids, interm_chunk,
                top_k, n_interm, k_hidden, group_size);
            dim3 g_down((unsigned)((k_hidden + 63) / 64), (unsigned)max_tiles16);
            w34_moe::moe_w4a16_grouped_down_mma_f16_kernel<true><<<g_down, 128, 0, st>>>(
                interm_chunk, down_packed, (const __half*)down_scales, down_qzeros,
                sorted_pair_ids, tile_expert_ids, tw_chunk, stage2_pair_out,
                k_hidden, n_interm, group_size);
            dim3 g_red((unsigned)((k_hidden + 511) / 512), (unsigned)chunk_m);
            w34_moe::moe_down_topk_reduce_f16_kernel<<<g_red, 128, 0, st>>>(
                stage2_pair_out, ids_chunk, sh_out_chunk, sh_gate_chunk, out_chunk,
                chunk_m, top_k, num_experts, k_hidden);
            tok_off += chunk_m;
        }
        return (int)cudaGetLastError();
    }

    cudaStream_t st = (cudaStream_t)ex->stream;
    dim3 g1((unsigned)((n_interm + 3) / 4), (unsigned)(num_tokens * top_k));
    w34_moe::moe_w4a16_gate_up_silu_f16_kernel<true><<<g1, 128, 0, st>>>(
        (const __half*)x, gate_up_packed, (const __half*)gate_up_scales, gate_up_qzeros, topk_ids,
        (__half*)interm_buf, num_tokens, top_k, num_experts, n_interm, k_hidden, group_size);
    dim3 g2((unsigned)((k_hidden + 3) / 4), (unsigned)num_tokens);
    w34_moe::moe_w4a16_down_accum_f16_kernel<true><<<g2, 128, 0, st>>>(
        (const __half*)interm_buf, down_packed, (const __half*)down_scales, down_qzeros,
        topk_ids, topk_weights, (const __half*)shared_expert_out,
        (const __half*)shared_expert_gate_logits, (__half*)out,
        num_tokens, top_k, num_experts, k_hidden, n_interm, group_size);
    return (int)cudaGetLastError();
}

extern "C" int w34_moe_f16_gate_up_silu(
    const w34_exec* ex,
    const void* x,
    const void* gate_up_w,
    const int32_t* topk_ids,
    void* interm_out,
    int num_tokens,
    int top_k,
    int num_experts,
    int n_interm,
    int k_hidden) {
    if (ex == nullptr || x == nullptr || gate_up_w == nullptr || topk_ids == nullptr ||
        interm_out == nullptr || num_tokens < 0 || top_k <= 0 || num_experts <= 0 ||
        n_interm <= 0 || k_hidden <= 0) {
        return (int)cudaErrorInvalidValue;
    }
    if (num_tokens == 0) return 0;
    dim3 grid((unsigned)((n_interm + 3) / 4), (unsigned)(num_tokens * top_k));
    w34_moe::moe_f16_gate_up_silu_kernel<<<grid, 128, 0, (cudaStream_t)ex->stream>>>(
        (const __half*)x, (const __half*)gate_up_w, topk_ids, (__half*)interm_out, num_tokens,
        top_k, num_experts, n_interm, k_hidden);
    return (int)cudaGetLastError();
}

extern "C" int w34_moe_f16_down_accum(
    const w34_exec* ex,
    const void* interm_in,
    const void* down_w,
    const int32_t* topk_ids,
    const float* topk_weights,
    const void* shared_expert_out,
    const void* shared_expert_gate_logits,
    void* out,
    int num_tokens,
    int top_k,
    int num_experts,
    int k_hidden,
    int n_interm) {
    if (ex == nullptr || interm_in == nullptr || down_w == nullptr || topk_ids == nullptr ||
        topk_weights == nullptr || out == nullptr || num_tokens < 0 || top_k <= 0 ||
        num_experts <= 0 || k_hidden <= 0 || n_interm <= 0) {
        return (int)cudaErrorInvalidValue;
    }
    if (num_tokens == 0) return 0;
    dim3 grid((unsigned)((k_hidden + 3) / 4), (unsigned)num_tokens);
    w34_moe::moe_f16_down_accum_kernel<<<grid, 128, 0, (cudaStream_t)ex->stream>>>(
        (const __half*)interm_in, (const __half*)down_w, topk_ids, topk_weights,
        (const __half*)shared_expert_out, (const __half*)shared_expert_gate_logits, (__half*)out,
        num_tokens, top_k, num_experts, k_hidden, n_interm);
    return (int)cudaGetLastError();
}

namespace w34_moe {

__global__ void moe_unpack_staged_experts_kernel(
    const uint8_t* __restrict__ staged_packs,
    const int32_t* __restrict__ dst_slots,
    uint32_t* __restrict__ gate_up_packed_base,
    uint16_t* __restrict__ gate_up_scales_base,
    uint32_t* __restrict__ down_packed_base,
    uint16_t* __restrict__ down_scales_base,
    int num_staged,
    int dst_capacity,
    size_t pack_stride_bytes,
    size_t gu_packed_bytes,
    size_t gu_scales_bytes,
    size_t dn_packed_bytes,
    size_t dn_scales_bytes) {
    const int s = (int)blockIdx.y;
    if (s >= num_staged) return;
    const int dst_slot = dst_slots[s];
    if (dst_slot < 0 || dst_slot >= dst_capacity) return;

    const uint8_t* src_pack = staged_packs + (size_t)s * pack_stride_bytes;
    const size_t tid = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    const size_t stride = (size_t)gridDim.x * blockDim.x;

    // 1. Copy gate_up_packed (16B uint4 chunks)
    const size_t gu_p_vec = gu_packed_bytes >> 4;
    const uint4* src_gu_p = reinterpret_cast<const uint4*>(src_pack);
    uint4* dst_gu_p = reinterpret_cast<uint4*>(
        reinterpret_cast<uint8_t*>(gate_up_packed_base) + (size_t)dst_slot * gu_packed_bytes);
    for (size_t i = tid; i < gu_p_vec; i += stride) {
        dst_gu_p[i] = src_gu_p[i];
    }

    // 2. Copy gate_up_scales
    const size_t gu_s_vec = gu_scales_bytes >> 4;
    const uint4* src_gu_s = reinterpret_cast<const uint4*>(src_pack + gu_packed_bytes);
    uint4* dst_gu_s = reinterpret_cast<uint4*>(
        reinterpret_cast<uint8_t*>(gate_up_scales_base) + (size_t)dst_slot * gu_scales_bytes);
    for (size_t i = tid; i < gu_s_vec; i += stride) {
        dst_gu_s[i] = src_gu_s[i];
    }

    // 3. Copy down_packed
    const size_t dn_p_vec = dn_packed_bytes >> 4;
    const uint4* src_dn_p =
        reinterpret_cast<const uint4*>(src_pack + gu_packed_bytes + gu_scales_bytes);
    uint4* dst_dn_p = reinterpret_cast<uint4*>(
        reinterpret_cast<uint8_t*>(down_packed_base) + (size_t)dst_slot * dn_packed_bytes);
    for (size_t i = tid; i < dn_p_vec; i += stride) {
        dst_dn_p[i] = src_dn_p[i];
    }

    // 4. Copy down_scales
    const size_t dn_s_vec = dn_scales_bytes >> 4;
    const uint4* src_dn_s = reinterpret_cast<const uint4*>(
        src_pack + gu_packed_bytes + gu_scales_bytes + dn_packed_bytes);
    uint4* dst_dn_s = reinterpret_cast<uint4*>(
        reinterpret_cast<uint8_t*>(down_scales_base) + (size_t)dst_slot * dn_scales_bytes);
    for (size_t i = tid; i < dn_s_vec; i += stride) {
        dst_dn_s[i] = src_dn_s[i];
    }
}

__global__ void moe_remap_topk_ids_i32_kernel(
    const int32_t* __restrict__ global_topk_ids,
    const int32_t* __restrict__ expert_to_slot,
    int32_t* __restrict__ remapped_topk_ids,
    int total_pairs,
    int global_expert_offset,
    int local_num_experts) {
    const int idx = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    if (idx >= total_pairs) return;
    const int ge = global_topk_ids[idx];
    const int64_t le = (int64_t)ge - global_expert_offset;
    int32_t slot = -1;
    if (le >= 0 && le < local_num_experts) {
        slot = expert_to_slot[le];
    }
    remapped_topk_ids[idx] = slot;
}

__global__ void add_inplace_f16_kernel(
    __half* __restrict__ dst,
    const __half* __restrict__ src,
    int64_t num_elems) {
    const int64_t idx2 = ((int64_t)blockIdx.x * blockDim.x + threadIdx.x) * 2;
    if (idx2 + 1 < num_elems) {
        __half2* d2 = reinterpret_cast<__half2*>(dst + idx2);
        const __half2* s2 = reinterpret_cast<const __half2*>(src + idx2);
        *d2 = __hadd2(*d2, *s2);
    } else if (idx2 < num_elems) {
        dst[idx2] = __hadd(dst[idx2], src[idx2]);
    }
}

} // namespace w34_moe

extern "C" int w34_moe_unpack_staged_experts(
    const w34_exec* ex,
    const void* staged_packs,
    const int32_t* dst_slots,
    uint32_t* gate_up_packed_base,
    void* gate_up_scales_base,
    uint32_t* down_packed_base,
    void* down_scales_base,
    int num_staged,
    int dst_capacity,
    size_t pack_stride_bytes,
    int n_interm,
    int k_hidden,
    int group_size) {
    if (!ex || !staged_packs || !dst_slots || !gate_up_packed_base || !gate_up_scales_base ||
        !down_packed_base || !down_scales_base || num_staged < 0 || dst_capacity <= 0 || n_interm <= 0 ||
        (((uintptr_t)staged_packs | (uintptr_t)gate_up_packed_base |
          (uintptr_t)gate_up_scales_base | (uintptr_t)down_packed_base |
          (uintptr_t)down_scales_base) & 15) != 0 ||
        k_hidden <= 0 || group_size <= 0 || num_staged > 65535 ||
        n_interm > INT_MAX / 2 || n_interm % 8 != 0 || k_hidden % 8 != 0 ||
        n_interm % group_size != 0 || k_hidden % group_size != 0 ||
        (pack_stride_bytes & 15) != 0) {
        return (int)cudaErrorInvalidValue;
    }
    if (num_staged == 0) return 0;

    const size_t gu_packed_bytes = (size_t)(2 * n_interm) * (size_t)(k_hidden / 8) * sizeof(uint32_t);
    const size_t gu_scales_bytes =
        (size_t)(2 * n_interm) * (size_t)(k_hidden / group_size) * sizeof(uint16_t);
    const size_t dn_packed_bytes = (size_t)k_hidden * (size_t)(n_interm / 8) * sizeof(uint32_t);
    const size_t dn_scales_bytes =
        (size_t)k_hidden * (size_t)(n_interm / group_size) * sizeof(uint16_t);
    if ((gu_packed_bytes & 15) != 0 || (gu_scales_bytes & 15) != 0 ||
        (dn_packed_bytes & 15) != 0 || (dn_scales_bytes & 15) != 0 ||
        pack_stride_bytes < gu_packed_bytes + gu_scales_bytes + dn_packed_bytes + dn_scales_bytes) {
        return (int)cudaErrorInvalidValue;
    }

    dim3 grid(64, (unsigned)num_staged);
    w34_moe::moe_unpack_staged_experts_kernel<<<grid, 256, 0, (cudaStream_t)ex->stream>>>(
        static_cast<const uint8_t*>(staged_packs),
        dst_slots,
        gate_up_packed_base,
        static_cast<uint16_t*>(gate_up_scales_base),
        down_packed_base,
        static_cast<uint16_t*>(down_scales_base),
        num_staged,
        dst_capacity,
        pack_stride_bytes,
        gu_packed_bytes,
        gu_scales_bytes,
        dn_packed_bytes,
        dn_scales_bytes);
    return (int)cudaGetLastError();
}

extern "C" int w34_moe_remap_topk_ids_i32(
    const w34_exec* ex,
    const int32_t* global_topk_ids,
    const int32_t* expert_to_slot,
    int32_t* remapped_topk_ids,
    int total_pairs,
    int global_expert_offset,
    int local_num_experts) {
    if (!ex || !global_topk_ids || !expert_to_slot || !remapped_topk_ids || total_pairs < 0 ||
        local_num_experts <= 0) {
        return (int)cudaErrorInvalidValue;
    }
    if (total_pairs == 0) return 0;
    const int threads = 256;
    const int blocks = (int)(((int64_t)total_pairs + threads - 1) / threads);
    w34_moe::moe_remap_topk_ids_i32_kernel<<<blocks, threads, 0, (cudaStream_t)ex->stream>>>(
        global_topk_ids,
        expert_to_slot,
        remapped_topk_ids,
        total_pairs,
        global_expert_offset,
        local_num_experts);
    return (int)cudaGetLastError();
}

extern "C" int w34_add_inplace_f16(
    const w34_exec* ex,
    void* dst_f16,
    const void* src_f16,
    int64_t num_elems) {
    if (!ex || !dst_f16 || !src_f16 || num_elems < 0 ||
        (((uintptr_t)dst_f16 | (uintptr_t)src_f16) & 3) != 0) {
        return (int)cudaErrorInvalidValue;
    }
    if (num_elems == 0) return 0;
    const int64_t pairs = num_elems / 2 + num_elems % 2;
    if (pairs > (int64_t)INT_MAX * 256) return (int)cudaErrorInvalidValue;
    const int threads = 256;
    const int blocks = (int)((pairs + threads - 1) / threads);
    w34_moe::add_inplace_f16_kernel<<<blocks, threads, 0, (cudaStream_t)ex->stream>>>(
        static_cast<__half*>(dst_f16),
        static_cast<const __half*>(src_f16),
        num_elems);
    return (int)cudaGetLastError();
}
