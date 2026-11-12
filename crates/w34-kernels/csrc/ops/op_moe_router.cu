// Qwen3.5 / Qwen3.6 / Qwen3.8 Hybrid MoE Router & Expert Cache Staging Kernels.
// Split from op_moe.cu for modularity and parallel nvcc compilation.
#include "../w34_cuda.h"

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <float.h>
#include <limits.h>
#include <math.h>
#include <stdint.h>

extern "C" __device__ float __nv_expf(float);

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

__global__ void moe_gpu_lru_plan_and_remap_kernel(
    const int32_t* __restrict__ global_topk_ids,
    int32_t* __restrict__ expert_to_slot,
    int32_t* __restrict__ slot_to_expert,
    uint64_t* __restrict__ slot_last_used,
    uint64_t* __restrict__ step_counter,
    int32_t* __restrict__ remapped_topk_ids,
    int32_t* __restrict__ miss_local_experts,
    int32_t* __restrict__ miss_victim_slots,
    int32_t* __restrict__ num_misses_out,
    int total_pairs,
    int global_expert_offset,
    int local_num_experts,
    int k_hot,
    int max_misses) {
    const int lane = (int)threadIdx.x;
    uint64_t step = 0;
    if (lane == 0) {
        step = (*step_counter) + 1;
        *step_counter = step;
    }
    step = __shfl_sync(0xffffffffu, step, 0);

    for (int i = lane; i < total_pairs; i += 32) {
        const int ge = global_topk_ids[i];
        const int64_t le = (int64_t)ge - global_expert_offset;
        if (le < 0 || le >= local_num_experts) {
            remapped_topk_ids[i] = -1;
        } else {
            const int slot = expert_to_slot[le];
            if (slot >= 0 && slot < k_hot) {
                slot_last_used[slot] = step;
                remapped_topk_ids[i] = slot;
            } else {
                remapped_topk_ids[i] = -2;
            }
        }
    }
    __syncwarp();

    // Protect the static hot prefix of `k_hot` during decode and speculative verify (`M=1..8`):
    // reserve up to `total_pairs` (`<= 40` for `M <= 4`, capped at `min(40, k_hot / 2)` for `M=8`)
    // tail slots for dynamic cold misses so speculative tree misses never evict anchor slots `0..k_protected`.
    const int tail_budget = (total_pairs <= 40) ? 40 : ((40 < (k_hot / 2)) ? 40 : (k_hot / 2));
    const int k_protected = (k_hot > tail_budget) ? (k_hot - tail_budget) : 0;

    int n_miss = 0;
    for (int base = 0; base < total_pairs; base += 32) {
        const int idx = base + lane;
        const bool is_cand = (idx < total_pairs) && (remapped_topk_ids[idx] == -2);
        unsigned mask = __ballot_sync(0xffffffffu, is_cand);
        while (mask != 0u) {
            const int src_lane = __ffs(mask) - 1;
            mask &= mask - 1u;
            const int i = base + src_lane;

            int slot = 0;
            int ge = 0;
            int le = 0;
            if (lane == 0) {
                ge = global_topk_ids[i];
                le = (int)((int64_t)ge - global_expert_offset);
                slot = expert_to_slot[le];
                if (slot >= 0 && slot < k_hot) {
                    remapped_topk_ids[i] = slot;
                    slot_last_used[slot] = step;
                } else {
                    slot = -2;
                }
            }
            slot = __shfl_sync(0xffffffffu, slot, 0);
            if (slot != -2) continue;
            if (n_miss >= max_misses) {
                if (lane == 0) remapped_topk_ids[i] = -1;
                continue;
            }

            uint64_t best_ts = UINT64_MAX;
            int best_slot = k_protected;
            for (int s = k_protected + lane; s < k_hot; s += 32) {
                const uint64_t ts = slot_last_used[s];
                if (ts < best_ts || (ts == best_ts && s < best_slot)) {
                    best_ts = ts;
                    best_slot = s;
                }
            }
#pragma unroll
            for (int off = 16; off > 0; off >>= 1) {
                const uint64_t other_ts = __shfl_down_sync(0xffffffffu, best_ts, off);
                const int other_slot = __shfl_down_sync(0xffffffffu, best_slot, off);
                if (other_ts < best_ts || (other_ts == best_ts && other_slot < best_slot)) {
                    best_ts = other_ts;
                    best_slot = other_slot;
                }
            }
            best_ts = __shfl_sync(0xffffffffu, best_ts, 0);

            // If every slot in `k_protected .. k_hot` is already active in this step
            // (`best_ts >= step`), spill to the oldest inactive slot in `0 .. k_protected`
            // so an active expert in the current batch is never overwritten.
            if (best_ts >= step && k_protected > 0) {
                best_ts = UINT64_MAX;
                best_slot = 0;
                for (int s = lane; s < k_protected; s += 32) {
                    const uint64_t ts = slot_last_used[s];
                    if (ts < best_ts || (ts == best_ts && s < best_slot)) {
                        best_ts = ts;
                        best_slot = s;
                    }
                }
#pragma unroll
                for (int off = 16; off > 0; off >>= 1) {
                    const uint64_t other_ts = __shfl_down_sync(0xffffffffu, best_ts, off);
                    const int other_slot = __shfl_down_sync(0xffffffffu, best_slot, off);
                    if (other_ts < best_ts || (other_ts == best_ts && other_slot < best_slot)) {
                        best_ts = other_ts;
                        best_slot = other_slot;
                    }
                }
                best_ts = __shfl_sync(0xffffffffu, best_ts, 0);
            }
            if (best_ts >= step) {
                if (lane == 0) remapped_topk_ids[i] = -1;
                continue;
            }

            if (lane == 0) {
                const int victim_slot = best_slot;
                const int old_ge = slot_to_expert[victim_slot];
                const int64_t old_le = (int64_t)old_ge - global_expert_offset;
                if (old_le >= 0 && old_le < local_num_experts) {
                    expert_to_slot[old_le] = -1;
                }
                expert_to_slot[le] = victim_slot;
                slot_to_expert[victim_slot] = ge;
                slot_last_used[victim_slot] = step;
                miss_local_experts[n_miss] = le;
                miss_victim_slots[n_miss] = victim_slot;
                remapped_topk_ids[i] = victim_slot;
            }
            __syncwarp();
            n_miss++;
        }
    }
    if (lane == 0) {
        *num_misses_out = n_miss;
    }
}

__global__ void moe_pull_mapped_experts_kernel(
    const uint8_t* __restrict__ mapped_host_packs,
    const int32_t* __restrict__ src_local_experts,
    const int32_t* __restrict__ dst_slots,
    const int32_t* __restrict__ num_staged_buf,
    uint32_t* __restrict__ gate_up_packed_base,
    uint16_t* __restrict__ gate_up_scales_base,
    uint32_t* __restrict__ down_packed_base,
    uint16_t* __restrict__ down_scales_base,
    int num_staged_imm,
    int local_num_experts,
    int dst_capacity,
    size_t pack_stride_bytes,
    size_t gu_packed_bytes,
    size_t gu_scales_bytes,
    size_t dn_packed_bytes,
    size_t dn_scales_bytes) {
    const int s = (int)blockIdx.y;
    const int num_staged = (num_staged_buf != nullptr) ? num_staged_buf[0] : num_staged_imm;
    if (s >= num_staged) return;
    const int le = src_local_experts[s];
    const int dst_slot = dst_slots[s];
    if (le < 0 || le >= local_num_experts || dst_slot < 0 || dst_slot >= dst_capacity) return;

    const uint8_t* src_pack = mapped_host_packs + (size_t)le * pack_stride_bytes;
    const size_t tid = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    const size_t stride = (size_t)gridDim.x * blockDim.x;

    const size_t gu_p_vec = gu_packed_bytes >> 4;
    const uint4* src_gu_p = reinterpret_cast<const uint4*>(src_pack);
    uint4* dst_gu_p = reinterpret_cast<uint4*>(
        reinterpret_cast<uint8_t*>(gate_up_packed_base) + (size_t)dst_slot * gu_packed_bytes);
    for (size_t i = tid; i < gu_p_vec; i += stride) {
        dst_gu_p[i] = src_gu_p[i];
    }

    const size_t gu_s_vec = gu_scales_bytes >> 4;
    const uint4* src_gu_s = reinterpret_cast<const uint4*>(src_pack + gu_packed_bytes);
    uint4* dst_gu_s = reinterpret_cast<uint4*>(
        reinterpret_cast<uint8_t*>(gate_up_scales_base) + (size_t)dst_slot * gu_scales_bytes);
    for (size_t i = tid; i < gu_s_vec; i += stride) {
        dst_gu_s[i] = src_gu_s[i];
    }

    const size_t dn_p_vec = dn_packed_bytes >> 4;
    const uint4* src_dn_p =
        reinterpret_cast<const uint4*>(src_pack + gu_packed_bytes + gu_scales_bytes);
    uint4* dst_dn_p = reinterpret_cast<uint4*>(
        reinterpret_cast<uint8_t*>(down_packed_base) + (size_t)dst_slot * dn_packed_bytes);
    for (size_t i = tid; i < dn_p_vec; i += stride) {
        dst_dn_p[i] = src_dn_p[i];
    }

    const size_t dn_s_vec = dn_scales_bytes >> 4;
    const uint4* src_dn_s = reinterpret_cast<const uint4*>(
        src_pack + gu_packed_bytes + gu_scales_bytes + dn_packed_bytes);
    uint4* dst_dn_s = reinterpret_cast<uint4*>(
        reinterpret_cast<uint8_t*>(down_scales_base) + (size_t)dst_slot * dn_scales_bytes);
    for (size_t i = tid; i < dn_s_vec; i += stride) {
        dst_dn_s[i] = src_dn_s[i];
    }
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

    size_t gu_packed_bytes = (size_t)(2 * n_interm) * (size_t)(k_hidden / 8) * sizeof(uint32_t);
    const size_t gu_scales_bytes =
        (size_t)(2 * n_interm) * (size_t)(k_hidden / group_size) * sizeof(uint16_t);
    size_t dn_packed_bytes = (size_t)k_hidden * (size_t)(n_interm / 8) * sizeof(uint32_t);
    const size_t dn_scales_bytes =
        (size_t)k_hidden * (size_t)(n_interm / group_size) * sizeof(uint16_t);
    const size_t w4_raw_bytes =
        gu_packed_bytes + gu_scales_bytes + dn_packed_bytes + dn_scales_bytes;
    const size_t w2_raw_bytes =
        (gu_packed_bytes >> 1) + gu_scales_bytes + (dn_packed_bytes >> 1) + dn_scales_bytes;
    const size_t w2_aligned_stride = (w2_raw_bytes + 4095u) & ~size_t(4095u);
    if (pack_stride_bytes < w4_raw_bytes &&
        (pack_stride_bytes == w2_aligned_stride || pack_stride_bytes == w2_raw_bytes) &&
        (n_interm % 16) == 0 && (k_hidden % 16) == 0) {
        gu_packed_bytes >>= 1;
        dn_packed_bytes >>= 1;
    }
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

extern "C" int w34_moe_pull_mapped_experts_with_count(
    const w34_exec* ex,
    const void* mapped_host_packs,
    const int32_t* src_local_experts,
    const int32_t* dst_slots,
    const int32_t* num_staged_buf,
    uint32_t* gate_up_packed_base,
    void* gate_up_scales_base,
    uint32_t* down_packed_base,
    void* down_scales_base,
    int max_staged,
    int local_num_experts,
    int dst_capacity,
    size_t pack_stride_bytes,
    int n_interm,
    int k_hidden,
    int group_size) {
    if (!ex || !mapped_host_packs || !src_local_experts || !dst_slots || !gate_up_packed_base ||
        !gate_up_scales_base || !down_packed_base || !down_scales_base || max_staged < 0 ||
        local_num_experts <= 0 || dst_capacity <= 0 || n_interm <= 0 ||
        (((uintptr_t)mapped_host_packs | (uintptr_t)gate_up_packed_base |
          (uintptr_t)gate_up_scales_base | (uintptr_t)down_packed_base |
          (uintptr_t)down_scales_base) & 15) != 0 ||
        k_hidden <= 0 || group_size <= 0 || max_staged > 65535 ||
        n_interm > INT_MAX / 2 || n_interm % 8 != 0 || k_hidden % 8 != 0 ||
        n_interm % group_size != 0 || k_hidden % group_size != 0 ||
        (pack_stride_bytes & 15) != 0) {
        return (int)cudaErrorInvalidValue;
    }
    if (max_staged == 0) return 0;

    size_t gu_packed_bytes = (size_t)(2 * n_interm) * (size_t)(k_hidden / 8) * sizeof(uint32_t);
    const size_t gu_scales_bytes =
        (size_t)(2 * n_interm) * (size_t)(k_hidden / group_size) * sizeof(uint16_t);
    size_t dn_packed_bytes = (size_t)k_hidden * (size_t)(n_interm / 8) * sizeof(uint32_t);
    const size_t dn_scales_bytes =
        (size_t)k_hidden * (size_t)(n_interm / group_size) * sizeof(uint16_t);
    const size_t w4_raw_bytes =
        gu_packed_bytes + gu_scales_bytes + dn_packed_bytes + dn_scales_bytes;
    const size_t w2_raw_bytes =
        (gu_packed_bytes >> 1) + gu_scales_bytes + (dn_packed_bytes >> 1) + dn_scales_bytes;
    const size_t w2_aligned_stride = (w2_raw_bytes + 4095u) & ~size_t(4095u);
    if (pack_stride_bytes < w4_raw_bytes &&
        (pack_stride_bytes == w2_aligned_stride || pack_stride_bytes == w2_raw_bytes) &&
        (n_interm % 16) == 0 && (k_hidden % 16) == 0) {
        gu_packed_bytes >>= 1;
        dn_packed_bytes >>= 1;
    }
    if ((gu_packed_bytes & 15) != 0 || (gu_scales_bytes & 15) != 0 ||
        (dn_packed_bytes & 15) != 0 || (dn_scales_bytes & 15) != 0 ||
        pack_stride_bytes < gu_packed_bytes + gu_scales_bytes + dn_packed_bytes + dn_scales_bytes) {
        return (int)cudaErrorInvalidValue;
    }

    dim3 grid(64, (unsigned)max_staged);
    w34_moe::moe_pull_mapped_experts_kernel<<<grid, 256, 0, (cudaStream_t)ex->stream>>>(
        static_cast<const uint8_t*>(mapped_host_packs),
        src_local_experts,
        dst_slots,
        num_staged_buf,
        gate_up_packed_base,
        static_cast<uint16_t*>(gate_up_scales_base),
        down_packed_base,
        static_cast<uint16_t*>(down_scales_base),
        max_staged,
        local_num_experts,
        dst_capacity,
        pack_stride_bytes,
        gu_packed_bytes,
        gu_scales_bytes,
        dn_packed_bytes,
        dn_scales_bytes);
    return (int)cudaGetLastError();
}

extern "C" int w34_moe_pull_mapped_experts(
    const w34_exec* ex,
    const void* mapped_host_packs,
    const int32_t* src_local_experts,
    const int32_t* dst_slots,
    uint32_t* gate_up_packed_base,
    void* gate_up_scales_base,
    uint32_t* down_packed_base,
    void* down_scales_base,
    int num_staged,
    int local_num_experts,
    int dst_capacity,
    size_t pack_stride_bytes,
    int n_interm,
    int k_hidden,
    int group_size) {
    return w34_moe_pull_mapped_experts_with_count(
        ex,
        mapped_host_packs,
        src_local_experts,
        dst_slots,
        nullptr,
        gate_up_packed_base,
        gate_up_scales_base,
        down_packed_base,
        down_scales_base,
        num_staged,
        local_num_experts,
        dst_capacity,
        pack_stride_bytes,
        n_interm,
        k_hidden,
        group_size);
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

extern "C" int w34_moe_gpu_lru_update_and_pull(
    const w34_exec* ex,
    const int32_t* global_topk_ids,
    int32_t* expert_to_slot,
    int32_t* slot_to_expert,
    uint64_t* slot_last_used,
    uint64_t* step_counter,
    int32_t* remapped_topk_ids,
    int32_t* miss_local_experts,
    int32_t* miss_victim_slots,
    int32_t* num_misses_buf,
    const void* mapped_host_packs,
    uint32_t* gate_up_packed_base,
    void* gate_up_scales_base,
    uint32_t* down_packed_base,
    void* down_scales_base,
    int total_pairs,
    int global_expert_offset,
    int local_num_experts,
    int k_hot,
    int max_misses,
    size_t pack_stride_bytes,
    int n_interm,
    int k_hidden,
    int group_size) {
    if (!ex || !global_topk_ids || !expert_to_slot || !slot_to_expert || !slot_last_used ||
        !step_counter || !remapped_topk_ids || !miss_local_experts || !miss_victim_slots ||
        !num_misses_buf || total_pairs < 0 || local_num_experts <= 0 || k_hot <= 0 ||
        max_misses < 0 || max_misses > 65535 || (max_misses > 0 && mapped_host_packs == nullptr)) {
        return (int)cudaErrorInvalidValue;
    }
    if (total_pairs == 0) return 0;

    cudaStream_t stream = (cudaStream_t)ex->stream;
    w34_moe::moe_gpu_lru_plan_and_remap_kernel<<<1, 32, 0, stream>>>(
        global_topk_ids,
        expert_to_slot,
        slot_to_expert,
        slot_last_used,
        step_counter,
        remapped_topk_ids,
        miss_local_experts,
        miss_victim_slots,
        num_misses_buf,
        total_pairs,
        global_expert_offset,
        local_num_experts,
        k_hot,
        max_misses);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) return (int)err;

    if (mapped_host_packs != nullptr && max_misses > 0) {
        return w34_moe_pull_mapped_experts_with_count(
            ex,
            mapped_host_packs,
            miss_local_experts,
            miss_victim_slots,
            num_misses_buf,
            gate_up_packed_base,
            gate_up_scales_base,
            down_packed_base,
            down_scales_base,
            max_misses,
            local_num_experts,
            k_hot,
            pack_stride_bytes,
            n_interm,
            k_hidden,
            group_size);
    }
    return 0;
}
