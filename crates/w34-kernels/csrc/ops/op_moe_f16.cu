// Dense FP16 Mixture-of-Experts (MoE) kernels for unquantized MoE models & unit tests.
// Split from op_moe.cu for modularity and parallel nvcc compilation.
#include "../w34_cuda.h"

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <math.h>

namespace w34_moe_f16 {

__device__ __forceinline__ float warp_reduce_sum(float v) {
#pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
        v += __shfl_xor_sync(0xffffffffu, v, off);
    }
    return v;
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

}  // namespace w34_moe_f16

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
    w34_moe_f16::moe_f16_gate_up_silu_kernel<<<grid, 128, 0, (cudaStream_t)ex->stream>>>(
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
    w34_moe_f16::moe_f16_down_accum_kernel<<<grid, 128, 0, (cudaStream_t)ex->stream>>>(
        (const __half*)interm_in, (const __half*)down_w, topk_ids, topk_weights,
        (const __half*)shared_expert_out, (const __half*)shared_expert_gate_logits, (__half*)out,
        num_tokens, top_k, num_experts, k_hidden, n_interm);
    return (int)cudaGetLastError();
}
