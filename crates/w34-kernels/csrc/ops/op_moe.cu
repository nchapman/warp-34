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
    const uint32_t bias_bits = 0x64086408u;

    for (int w = lane; w < words_per_row; w += 32) {
        const uint4 xv = __ldg(&x_vec4[w]);
        const uint32_t* x2 = reinterpret_cast<const uint32_t*>(&xv);
        const uint32_t gw = __ldg(&gate_w[w]);
        const uint32_t uw = __ldg(&up_w[w]);
        const int g_idx = w / words_per_group;
        const float gs = __half2float(__ldg(&gate_s[g_idx]));
        const float us = __half2float(__ldg(&up_s[g_idx]));

        float g_sub = 0.0f;
        float u_sub = 0.0f;
#pragma unroll
        for (int p = 0; p < 4; ++p) {
            const __half2 xh2 = *reinterpret_cast<const __half2*>(&x2[p]);
            const float x_lo = __low2float(xh2);
            const float x_hi = __high2float(xh2);

            const uint32_t gb = (gw >> (p << 3)) & 0xFFu;
            const uint32_t gh2_bits = ((gb & 0x0Fu) | ((gb & 0xF0u) << 12)) | 0x64006400u;
            const __half2 gqh2 = __hsub2(
                *reinterpret_cast<const __half2*>(&gh2_bits),
                *reinterpret_cast<const __half2*>(&bias_bits));
            g_sub += __low2float(gqh2) * x_lo;
            g_sub += __high2float(gqh2) * x_hi;

            const uint32_t ub = (uw >> (p << 3)) & 0xFFu;
            const uint32_t uh2_bits = ((ub & 0x0Fu) | ((ub & 0xF0u) << 12)) | 0x64006400u;
            const __half2 uqh2 = __hsub2(
                *reinterpret_cast<const __half2*>(&uh2_bits),
                *reinterpret_cast<const __half2*>(&bias_bits));
            u_sub += __low2float(uqh2) * x_lo;
            u_sub += __high2float(uqh2) * x_hi;
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
        const uint32_t bias_bits = 0x64086408u;
        for (int w = lane; w < words_per_row; w += 32) {
            const uint4 xv = __ldg(&x_vec4[w]);
            const uint32_t* x2 = reinterpret_cast<const uint32_t*>(&xv);
            const uint32_t dw = __ldg(&dw_ptr[w]);
            const float ds = __half2float(__ldg(&ds_ptr[w / words_per_group]));

            float sub = 0.0f;
#pragma unroll
            for (int p = 0; p < 4; ++p) {
                const __half2 xh2 = *reinterpret_cast<const __half2*>(&x2[p]);
                const uint32_t db = (dw >> (p << 3)) & 0xFFu;
                const uint32_t dh2_bits = ((db & 0x0Fu) | ((db & 0xF0u) << 12)) | 0x64006400u;
                const __half2 dqh2 = __hsub2(
                    *reinterpret_cast<const __half2*>(&dh2_bits),
                    *reinterpret_cast<const __half2*>(&bias_bits));
                sub += __low2float(dqh2) * __low2float(xh2);
                sub += __high2float(dqh2) * __high2float(xh2);
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

// --- Phase 6.3: Token-grouped (expert_id, token_slot) Marlin/moe_wna16 mma.sync MoE (M >= 4) ---

// Fast 0-ULP IEEE-754 binary16 magic-bias unpack of 2 consecutive U4B8 nibbles (zero-point 8)
// from `packed_w` (byte `tid_in_grp`) into exact signed FP16 `(q_lo - 8.0, q_hi - 8.0)`.
// Group scales are applied in FP32 AFTER `mma.sync` accumulation over each quantization group,
// eliminating pre-MMA FP16 scale-rounding drift and saving 7/8 scale loads for group_size=128.
__device__ __forceinline__ uint32_t unpack_u4b8_pair_unscaled(
    uint32_t packed_w,
    int tid_in_grp) {
    const uint32_t byte_v = (packed_w >> (tid_in_grp << 3)) & 0xFFu;
    const uint32_t h2_bits = ((byte_v & 0x0Fu) | ((byte_v & 0xF0u) << 12)) | 0x64006400u;
    const uint32_t bias_bits = 0x64086408u; // (1032.0, 1032.0) in FP16
    const __half2 h2 = __hsub2(
        *reinterpret_cast<const __half2*>(&h2_bits),
        *reinterpret_cast<const __half2*>(&bias_bits));
    return *reinterpret_cast<const uint32_t*>(&h2);
}

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

    if (tid == 0) {
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

// Grouped Stage 1 (W4A16 U4B8 -> m16n8k16 mma.sync with FP32 post-group scale multiplication):
// Computes gate_proj AND up_proj simultaneously (2x activation A register reuse) + SiLU(gate) * up.
// Grid: ((n_interm + 31) / 32, max_tiles), Block: 128 (4 warps; 8 gate + 8 up cols/warp = 32 cols/CTA).
__global__ void moe_w4a16_grouped_gate_up_silu_mma_f16_kernel(
    const __half* __restrict__ x,
    const uint32_t* __restrict__ gate_up_packed,
    const __half* __restrict__ gate_up_scales,
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

    float g0 = 0.0f, g1 = 0.0f, g2 = 0.0f, g3 = 0.0f;
    float u0 = 0.0f, u1 = 0.0f, u2 = 0.0f, u3 = 0.0f;
    float sg0 = 0.0f, sg1 = 0.0f, sg2 = 0.0f, sg3 = 0.0f;
    float su0 = 0.0f, su1 = 0.0f, su2 = 0.0f, su3 = 0.0f;

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
                gb0 = unpack_u4b8_pair_unscaled(gate_w_row[w_idx], tid_in_grp);
                gb1 = unpack_u4b8_pair_unscaled(gate_w_row[w_idx + 1], tid_in_grp);
                ub0 = unpack_u4b8_pair_unscaled(up_w_row[w_idx], tid_in_grp);
                ub1 = unpack_u4b8_pair_unscaled(up_w_row[w_idx + 1], tid_in_grp);
            }
            asm volatile(
                "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
                "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                : "+f"(sg0), "+f"(sg1), "+f"(sg2), "+f"(sg3)
                : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(gb0), "r"(gb1));
            asm volatile(
                "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
                "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                : "+f"(su0), "+f"(su1), "+f"(su2), "+f"(su3)
                : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(ub0), "r"(ub1));

            if (((col_base + 16) % group_size) == 0) {
                const int g_idx = col_base / group_size;
                const float gs_my = n_valid ? __half2float(gate_s_row[g_idx]) : 0.0f;
                const float us_my = n_valid ? __half2float(up_s_row[g_idx]) : 0.0f;
                const int lane_c0 = (2 * tid_in_grp) << 2;
                const int lane_c1 = (2 * tid_in_grp + 1) << 2;
                const float gs0 = __shfl_sync(0xffffffffu, gs_my, lane_c0);
                const float gs1 = __shfl_sync(0xffffffffu, gs_my, lane_c1);
                const float us0 = __shfl_sync(0xffffffffu, us_my, lane_c0);
                const float us1 = __shfl_sync(0xffffffffu, us_my, lane_c1);
                g0 += sg0 * gs0;
                g1 += sg1 * gs1;
                g2 += sg2 * gs0;
                g3 += sg3 * gs1;
                u0 += su0 * us0;
                u1 += su1 * us1;
                u2 += su2 * us0;
                u3 += su3 * us1;
                sg0 = 0.0f;
                sg1 = 0.0f;
                sg2 = 0.0f;
                sg3 = 0.0f;
                su0 = 0.0f;
                su1 = 0.0f;
                su2 = 0.0f;
                su3 = 0.0f;
            }
        }
        __syncthreads();
    }

    const int out_c0 = n_warp_base + 2 * tid_in_grp;
    if (out_c0 + 1 < n_interm) {
        const int p0 = s_pairs[group_id];
        if (p0 >= 0) {
            const float v0 = (g0 / (1.0f + expf(-g0))) * u0;
            const float v1 = (g1 / (1.0f + expf(-g1))) * u1;
            *reinterpret_cast<__half2*>(&interm_out[(size_t)p0 * n_interm + out_c0]) =
                __floats2half2_rn(v0, v1);
        }
        const int p1 = s_pairs[group_id + 8];
        if (p1 >= 0) {
            const float v2 = (g2 / (1.0f + expf(-g2))) * u2;
            const float v3 = (g3 / (1.0f + expf(-g3))) * u3;
            *reinterpret_cast<__half2*>(&interm_out[(size_t)p1 * n_interm + out_c0]) =
                __floats2half2_rn(v2, v3);
        }
    }
}

// Grouped Stage 2a (W4A16 U4B8 -> m16n8k16 mma.sync with FP32 post-group scale multiplication):
// Computes topk_weights[pair] * (interm_in[pair] @ W_down[expert]^T) into FP32 `stage2_pair_out`.
// Grid: ((k_hidden + 63) / 64, max_tiles), Block: 128 (4 warps; 2x8 = 16 cols/warp = 64 cols/CTA).
__global__ void moe_w4a16_grouped_down_mma_f16_kernel(
    const __half* __restrict__ interm_in,
    const uint32_t* __restrict__ down_packed,
    const __half* __restrict__ down_scales,
    const int32_t* __restrict__ sorted_pair_ids,
    const int32_t* __restrict__ tile_expert_ids,
    const float* __restrict__ topk_weights,
    float* __restrict__ stage2_pair_out,
    int k_hidden,
    int n_interm,
    int group_size) {
    const int tile_idx = (int)blockIdx.y;
    const int expert_id = tile_expert_ids[tile_idx];
    if (expert_id < 0) return;

    __shared__ __align__(16) __half s_a[16][72];
    __shared__ int s_pairs[16];
    __shared__ float s_tw[16];

    const int tid = (int)threadIdx.x;
    const int lane = tid & 31;
    const int wid = tid >> 5;
    const int group_id = lane >> 2;
    const int tid_in_grp = lane & 3;

    if (tid < 16) {
        const int p = sorted_pair_ids[(tile_idx << 4) + tid];
        s_pairs[tid] = p;
        s_tw[tid] = (p >= 0) ? topk_weights[p] : 0.0f;
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

    float d0_0 = 0.0f, d0_1 = 0.0f, d0_2 = 0.0f, d0_3 = 0.0f;
    float d1_0 = 0.0f, d1_1 = 0.0f, d1_2 = 0.0f, d1_3 = 0.0f;
    float sd0_0 = 0.0f, sd0_1 = 0.0f, sd0_2 = 0.0f, sd0_3 = 0.0f;
    float sd1_0 = 0.0f, sd1_1 = 0.0f, sd1_2 = 0.0f, sd1_3 = 0.0f;

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

#pragma unroll
        for (int ks = 0; ks < 4; ++ks) {
            const int k_off = ks << 4;
            const int col_base = kc + k_off;
            const uint32_t a0 = *reinterpret_cast<const uint32_t*>(&s_a[group_id][k_off + 2 * tid_in_grp]);
            const uint32_t a1 = *reinterpret_cast<const uint32_t*>(&s_a[group_id + 8][k_off + 2 * tid_in_grp]);
            const uint32_t a2 = *reinterpret_cast<const uint32_t*>(&s_a[group_id][k_off + 8 + 2 * tid_in_grp]);
            const uint32_t a3 = *reinterpret_cast<const uint32_t*>(&s_a[group_id + 8][k_off + 8 + 2 * tid_in_grp]);

            const int w_idx = col_base >> 3;
            uint32_t ba0 = 0u, ba1 = 0u, bb0 = 0u, bb1 = 0u;
            if (valid_a) {
                ba0 = unpack_u4b8_pair_unscaled(dw_a[w_idx], tid_in_grp);
                ba1 = unpack_u4b8_pair_unscaled(dw_a[w_idx + 1], tid_in_grp);
            }
            if (valid_b) {
                bb0 = unpack_u4b8_pair_unscaled(dw_b[w_idx], tid_in_grp);
                bb1 = unpack_u4b8_pair_unscaled(dw_b[w_idx + 1], tid_in_grp);
            }
            asm volatile(
                "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
                "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                : "+f"(sd0_0), "+f"(sd0_1), "+f"(sd0_2), "+f"(sd0_3)
                : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(ba0), "r"(ba1));
            asm volatile(
                "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
                "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                : "+f"(sd1_0), "+f"(sd1_1), "+f"(sd1_2), "+f"(sd1_3)
                : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(bb0), "r"(bb1));

            if (((col_base + 16) % group_size) == 0) {
                const int g_idx = col_base / group_size;
                const float sa_my = valid_a ? __half2float(ds_a[g_idx]) : 0.0f;
                const float sb_my = valid_b ? __half2float(ds_b[g_idx]) : 0.0f;
                const int lane_c0 = (2 * tid_in_grp) << 2;
                const int lane_c1 = (2 * tid_in_grp + 1) << 2;
                const float sa0 = __shfl_sync(0xffffffffu, sa_my, lane_c0);
                const float sa1 = __shfl_sync(0xffffffffu, sa_my, lane_c1);
                const float sb0 = __shfl_sync(0xffffffffu, sb_my, lane_c0);
                const float sb1 = __shfl_sync(0xffffffffu, sb_my, lane_c1);
                d0_0 += sd0_0 * sa0;
                d0_1 += sd0_1 * sa1;
                d0_2 += sd0_2 * sa0;
                d0_3 += sd0_3 * sa1;
                d1_0 += sd1_0 * sb0;
                d1_1 += sd1_1 * sb1;
                d1_2 += sd1_2 * sb0;
                d1_3 += sd1_3 * sb1;
                sd0_0 = 0.0f;
                sd0_1 = 0.0f;
                sd0_2 = 0.0f;
                sd0_3 = 0.0f;
                sd1_0 = 0.0f;
                sd1_1 = 0.0f;
                sd1_2 = 0.0f;
                sd1_3 = 0.0f;
            }
        }
        __syncthreads();
    }

    const int c0_a = out_warp_base + 2 * tid_in_grp;
    const int c0_b = out_warp_base + 8 + 2 * tid_in_grp;
    const int p0 = s_pairs[group_id];
    if (p0 >= 0) {
        const float tw0 = s_tw[group_id];
        if (c0_a + 1 < k_hidden) {
            *reinterpret_cast<float2*>(&stage2_pair_out[(size_t)p0 * k_hidden + c0_a]) =
                make_float2(d0_0 * tw0, d0_1 * tw0);
        }
        if (c0_b + 1 < k_hidden) {
            *reinterpret_cast<float2*>(&stage2_pair_out[(size_t)p0 * k_hidden + c0_b]) =
                make_float2(d1_0 * tw0, d1_1 * tw0);
        }
    }
    const int p1 = s_pairs[group_id + 8];
    if (p1 >= 0) {
        const float tw1 = s_tw[group_id + 8];
        if (c0_a + 1 < k_hidden) {
            *reinterpret_cast<float2*>(&stage2_pair_out[(size_t)p1 * k_hidden + c0_a]) =
                make_float2(d0_2 * tw1, d0_3 * tw1);
        }
        if (c0_b + 1 < k_hidden) {
            *reinterpret_cast<float2*>(&stage2_pair_out[(size_t)p1 * k_hidden + c0_b]) =
                make_float2(d1_2 * tw1, d1_3 * tw1);
        }
    }
}

// Deterministic Stage 2b reduction in slot order (0..top_k-1) + shared_expert * sigmoid(gate).
// Grid: ((k_hidden + 511) / 512, num_tokens), Block: 128 (4 columns/thread via float4).
__global__ void moe_down_topk_reduce_f16_kernel(
    const float* __restrict__ stage2_pair_out,
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
            const float4 v =
                *reinterpret_cast<const float4*>(&stage2_pair_out[(size_t)pair_idx * k_hidden + col4]);
            s0 += v.x;
            s1 += v.y;
            s2 += v.z;
            s3 += v.w;
        }
    }

    if (shared_expert_out != nullptr) {
        float sig = 1.0f;
        if (shared_expert_gate_logits != nullptr) {
            const float gl = __half2float(shared_expert_gate_logits[tok]);
            sig = 1.0f / (1.0f + expf(-gl));
        }
        const uint2 sh2 =
            *reinterpret_cast<const uint2*>(&shared_expert_out[(size_t)tok * k_hidden + col4]);
        const __half2 sh_lo = *reinterpret_cast<const __half2*>(&sh2.x);
        const __half2 sh_hi = *reinterpret_cast<const __half2*>(&sh2.y);
        s0 += __low2float(sh_lo) * sig;
        s1 += __high2float(sh_lo) * sig;
        s2 += __low2float(sh_hi) * sig;
        s3 += __high2float(sh_hi) * sig;
    }

    const __half2 out_lo = __floats2half2_rn(s0, s1);
    const __half2 out_hi = __floats2half2_rn(s2, s3);
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

static inline bool can_use_grouped_mma_moe(
    const w34_exec* ex,
    int num_tokens,
    int total_pairs,
    int num_experts,
    int n_interm,
    int k_hidden,
    int group_size,
    size_t needed_bytes) {
    return num_tokens >= 4 &&
           total_pairs >= num_experts &&
           num_experts > 0 && num_experts <= 1024 &&
           (n_interm % 64) == 0 && (n_interm % group_size) == 0 &&
           (k_hidden % 64) == 0 && (k_hidden % group_size) == 0 &&
           (group_size % 16) == 0 &&
           ex->scratch_ptr != 0 &&
           (ex->scratch_ptr & 15u) == 0u &&
           ex->scratch_size >= needed_bytes;
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
        w34_moe::moe_w4a16_grouped_gate_up_silu_mma_f16_kernel<<<grid, 128, 0, st>>>(
            (const __half*)x, gate_up_packed, (const __half*)gate_up_scales,
            sorted_pair_ids, tile_expert_ids, (__half*)interm_out,
            top_k, n_interm, k_hidden, group_size);
        return (int)cudaGetLastError();
    }
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
    const int total_pairs = num_tokens * top_k;
    const size_t stage2_bytes = (size_t)total_pairs * (size_t)k_hidden * sizeof(float);
    const int max_tiles16 = w34_moe::compute_moe_max_tiles<16>(total_pairs, num_experts);
    const size_t pairs16 = (size_t)max_tiles16 * 16u * sizeof(int32_t);
    const size_t tiles16 = (((size_t)max_tiles16 * sizeof(int32_t)) + 15u) & ~15u;
    if (w34_moe::can_use_grouped_mma_moe(
            ex, num_tokens, total_pairs, num_experts, n_interm, k_hidden, group_size,
            pairs16 + tiles16 + stage2_bytes)) {
        uint8_t* ws = reinterpret_cast<uint8_t*>((uintptr_t)ex->scratch_ptr);
        int32_t* sorted_pair_ids = reinterpret_cast<int32_t*>(ws);
        int32_t* tile_expert_ids = reinterpret_cast<int32_t*>(ws + pairs16);
        float* stage2_pair_out = reinterpret_cast<float*>(ws + pairs16 + tiles16);
        cudaStream_t st = (cudaStream_t)ex->stream;
        w34_moe::moe_align_block_size_kernel<16><<<1, 256, 0, st>>>(
            topk_ids, sorted_pair_ids, tile_expert_ids, total_pairs, num_experts, max_tiles16);
        dim3 g_down((unsigned)((k_hidden + 63) / 64), (unsigned)max_tiles16);
        w34_moe::moe_w4a16_grouped_down_mma_f16_kernel<<<g_down, 128, 0, st>>>(
            (const __half*)interm_in, down_packed, (const __half*)down_scales,
            sorted_pair_ids, tile_expert_ids, topk_weights, stage2_pair_out,
            k_hidden, n_interm, group_size);
        dim3 g_red((unsigned)((k_hidden + 511) / 512), (unsigned)num_tokens);
        w34_moe::moe_down_topk_reduce_f16_kernel<<<g_red, 128, 0, st>>>(
            stage2_pair_out, topk_ids, (const __half*)shared_expert_out,
            (const __half*)shared_expert_gate_logits, (__half*)out,
            num_tokens, top_k, num_experts, k_hidden);
        return (int)cudaGetLastError();
    }
    dim3 grid((unsigned)((k_hidden + 3) / 4), (unsigned)num_tokens);
    w34_moe::moe_w4a16_down_accum_f16_kernel<<<grid, 128, 0, (cudaStream_t)ex->stream>>>(
        (const __half*)interm_in, down_packed, (const __half*)down_scales, topk_ids, topk_weights,
        (const __half*)shared_expert_out, (const __half*)shared_expert_gate_logits, (__half*)out,
        num_tokens, top_k, num_experts, k_hidden, n_interm, group_size);
    return (int)cudaGetLastError();
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

    const int total_pairs = num_tokens * top_k;
    const size_t stage2_bytes = (size_t)total_pairs * (size_t)k_hidden * sizeof(float);
    const int max_tiles16 = w34_moe::compute_moe_max_tiles<16>(total_pairs, num_experts);
    const size_t pairs16 = (size_t)max_tiles16 * 16u * sizeof(int32_t);
    const size_t tiles16 = (((size_t)max_tiles16 * sizeof(int32_t)) + 15u) & ~15u;
    if (w34_moe::can_use_grouped_mma_moe(
            ex, num_tokens, total_pairs, num_experts, n_interm, k_hidden, group_size,
            pairs16 + tiles16 + stage2_bytes)) {
        uint8_t* ws = reinterpret_cast<uint8_t*>((uintptr_t)ex->scratch_ptr);
        int32_t* sorted_pair_ids = reinterpret_cast<int32_t*>(ws);
        int32_t* tile_expert_ids = reinterpret_cast<int32_t*>(ws + pairs16);
        float* stage2_pair_out = reinterpret_cast<float*>(ws + pairs16 + tiles16);
        cudaStream_t st = (cudaStream_t)ex->stream;

        w34_moe::moe_align_block_size_kernel<16><<<1, 256, 0, st>>>(
            topk_ids, sorted_pair_ids, tile_expert_ids, total_pairs, num_experts, max_tiles16);
        dim3 g_gu((unsigned)((n_interm + 31) / 32), (unsigned)max_tiles16);
        w34_moe::moe_w4a16_grouped_gate_up_silu_mma_f16_kernel<<<g_gu, 128, 0, st>>>(
            (const __half*)x, gate_up_packed, (const __half*)gate_up_scales,
            sorted_pair_ids, tile_expert_ids, (__half*)interm_buf,
            top_k, n_interm, k_hidden, group_size);
        dim3 g_down((unsigned)((k_hidden + 63) / 64), (unsigned)max_tiles16);
        w34_moe::moe_w4a16_grouped_down_mma_f16_kernel<<<g_down, 128, 0, st>>>(
            (const __half*)interm_buf, down_packed, (const __half*)down_scales,
            sorted_pair_ids, tile_expert_ids, topk_weights, stage2_pair_out,
            k_hidden, n_interm, group_size);
        dim3 g_red((unsigned)((k_hidden + 511) / 512), (unsigned)num_tokens);
        w34_moe::moe_down_topk_reduce_f16_kernel<<<g_red, 128, 0, st>>>(
            stage2_pair_out, topk_ids, (const __half*)shared_expert_out,
            (const __half*)shared_expert_gate_logits, (__half*)out,
            num_tokens, top_k, num_experts, k_hidden);
        return (int)cudaGetLastError();
    }

    cudaStream_t st = (cudaStream_t)ex->stream;
    dim3 g1((unsigned)((n_interm + 3) / 4), (unsigned)(num_tokens * top_k));
    w34_moe::moe_w4a16_gate_up_silu_f16_kernel<<<g1, 128, 0, st>>>(
        (const __half*)x, gate_up_packed, (const __half*)gate_up_scales, topk_ids,
        (__half*)interm_buf, num_tokens, top_k, num_experts, n_interm, k_hidden, group_size);
    dim3 g2((unsigned)((k_hidden + 3) / 4), (unsigned)num_tokens);
    w34_moe::moe_w4a16_down_accum_f16_kernel<<<g2, 128, 0, st>>>(
        (const __half*)interm_buf, down_packed, (const __half*)down_scales, topk_ids, topk_weights,
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
