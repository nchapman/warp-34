// Phase 5.3: Qwen3.5 / Qwen3.6 Hybrid MoE Kernels (Top-K Softmax Router,
// Fused W4A16 & FP16 Gate+Up+SiLU and Down+TopK-Accum+SharedExpert Gate).
#include "../w34_cuda.h"

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <float.h>
#include <math.h>
#include <stdint.h>

namespace w34_moe {

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

// Stage 1 (W4A16 U4B8 N-major): Fused gate_proj + up_proj + SiLU(gate) * up.
// gate_up_packed: [num_experts, 2 * n_interm, k_hidden / 8] uint32_t
// gate_up_scales: [num_experts, 2 * n_interm, k_hidden / group_size] __half
// Grid: ((n_interm + 3) / 4, num_tokens * top_k), Block: 128 threads (4 warps).
__global__ void moe_w4a16_gate_up_silu_f16_kernel(
    const __half* __restrict__ x,
    const uint32_t* __restrict__ gate_up_packed,
    const __half* __restrict__ gate_up_scales,
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
    const int expert_id = topk_ids[pair_idx];
    if (expert_id < 0 || expert_id >= num_experts) return;

    const int lane = threadIdx.x & 31;
    const int wid = threadIdx.x >> 5; // 0..3
    const int col = blockIdx.x * 4 + wid;
    if (col >= n_interm) return;

    const int words_per_row = k_hidden / 8;
    const int groups_per_row = (k_hidden + group_size - 1) / group_size;
    const int words_per_group = group_size / 8;
    const int n_gate_up = 2 * n_interm;

    const __half* x_row = x + (size_t)tok * k_hidden;
    const uint4* x_vec4 = (const uint4*)x_row;

    const size_t exp_row_base = (size_t)expert_id * n_gate_up;
    const uint32_t* gate_w = gate_up_packed + (exp_row_base + col) * words_per_row;
    const uint32_t* up_w = gate_up_packed + (exp_row_base + n_interm + col) * words_per_row;
    const __half* gate_s = gate_up_scales + (exp_row_base + col) * groups_per_row;
    const __half* up_s = gate_up_scales + (exp_row_base + n_interm + col) * groups_per_row;

    float gate_acc = 0.0f;
    float up_acc = 0.0f;

    for (int w = lane; w < words_per_row; w += 32) {
        const uint4 xv = x_vec4[w];
        const __half* x8 = (const __half*)&xv;
        const uint32_t gw = gate_w[w];
        const uint32_t uw = up_w[w];
        const int g_idx = w / words_per_group;
        const float gs = __half2float(gate_s[g_idx]);
        const float us = __half2float(up_s[g_idx]);

        float g_sub = 0.0f;
        float u_sub = 0.0f;
#pragma unroll
        for (int b = 0; b < 8; ++b) {
            const float xf = __half2float(x8[b]);
            const int qg = (int)((gw >> (4 * b)) & 0x0Fu) - 8;
            const int qu = (int)((uw >> (4 * b)) & 0x0Fu) - 8;
            g_sub += (float)qg * xf;
            u_sub += (float)qu * xf;
        }
        gate_acc += g_sub * gs;
        up_acc += u_sub * us;
    }

    gate_acc = warp_reduce_sum(gate_acc);
    up_acc = warp_reduce_sum(up_acc);

    if (lane == 0) {
        const float silu_g = gate_acc / (1.0f + expf(-gate_acc));
        interm_out[(size_t)pair_idx * n_interm + col] = __float2half(silu_g * up_acc);
    }
}

// Stage 2 (W4A16 U4B8 N-major): Fused down_proj + topk_weights sum + optional shared_expert * sigmoid(gate).
// down_packed: [num_experts, k_hidden, n_interm / 8] uint32_t
// down_scales: [num_experts, k_hidden, n_interm / group_size] __half
// Grid: ((k_hidden + 3) / 4, num_tokens), Block: 128 threads (4 warps).
__global__ void moe_w4a16_down_accum_f16_kernel(
    const __half* __restrict__ interm_in,
    const uint32_t* __restrict__ down_packed,
    const __half* __restrict__ down_scales,
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

    const int words_per_row = n_interm / 8;
    const int groups_per_row = (n_interm + group_size - 1) / group_size;
    const int words_per_group = group_size / 8;

    float total_sum = 0.0f;

    for (int slot = 0; slot < top_k; ++slot) {
        const int pair_idx = tok * top_k + slot;
        const int expert_id = topk_ids[pair_idx];
        if (expert_id < 0 || expert_id >= num_experts) continue;
        const float tw = topk_weights[pair_idx];

        const uint4* x_vec4 = (const uint4*)(interm_in + (size_t)pair_idx * n_interm);
        const size_t row_idx = (size_t)expert_id * k_hidden + out_col;
        const uint32_t* dw_ptr = down_packed + row_idx * words_per_row;
        const __half* ds_ptr = down_scales + row_idx * groups_per_row;

        float dot = 0.0f;
        for (int w = lane; w < words_per_row; w += 32) {
            const uint4 xv = x_vec4[w];
            const __half* x8 = (const __half*)&xv;
            const uint32_t dw = dw_ptr[w];
            const float ds = __half2float(ds_ptr[w / words_per_group]);

            float sub = 0.0f;
#pragma unroll
            for (int b = 0; b < 8; ++b) {
                const int qd = (int)((dw >> (4 * b)) & 0x0Fu) - 8;
                sub += (float)qd * __half2float(x8[b]);
            }
            dot += sub * ds;
        }
        dot = warp_reduce_sum(dot);
        total_sum += tw * dot;
    }

    if (lane == 0) {
        if (shared_expert_out != nullptr) {
            float sh = __half2float(shared_expert_out[(size_t)tok * k_hidden + out_col]);
            if (shared_expert_gate_logits != nullptr) {
                float gl = __half2float(shared_expert_gate_logits[tok]);
                float sig = 1.0f / (1.0f + expf(-gl));
                sh *= sig;
            }
            total_sum += sh;
        }
        out[(size_t)tok * k_hidden + out_col] = __float2half(total_sum);
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
    for (int c = lane; c < k_hidden; c += 32) {
        const float xf = __half2float(x_row[c]);
        g_acc += xf * __half2float(gw[c]);
        u_acc += xf * __half2float(uw[c]);
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
        for (int c = lane; c < n_interm; c += 32) {
            dot += __half2float(x_row[c]) * __half2float(dw_row[c]);
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
    w34_moe::moe_topk_router_f16_kernel<<<(unsigned)num_tokens, 128, 0, (cudaStream_t)ex->stream>>>(
        (const __half*)router_logits, topk_weights, topk_ids, num_tokens, num_experts, top_k,
        norm_topk_prob, scoring_func);
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
    dim3 grid((unsigned)((n_interm + 3) / 4), (unsigned)(num_tokens * top_k));
    w34_moe::moe_w4a16_gate_up_silu_f16_kernel<<<grid, 128, 0, (cudaStream_t)ex->stream>>>(
        (const __half*)x, gate_up_packed, (const __half*)gate_up_scales, topk_ids,
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
    dim3 grid((unsigned)((k_hidden + 3) / 4), (unsigned)num_tokens);
    w34_moe::moe_w4a16_down_accum_f16_kernel<<<grid, 128, 0, (cudaStream_t)ex->stream>>>(
        (const __half*)interm_in, down_packed, (const __half*)down_scales, topk_ids, topk_weights,
        (const __half*)shared_expert_out, (const __half*)shared_expert_gate_logits, (__half*)out,
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
