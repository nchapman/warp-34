// op_moe_cpu_bridge.cu — CUDA-Graph-safe GPU mailbox kernels for hybrid CPU+GPU MoE offload.
//
// Device-side half of the hybrid MoE bridge (`crates/w34-host/src/moe_cpu_bridge.rs`):
//   1. `w34_moe_hybrid_submit_and_pull`:
//      - `moe_hybrid_plan_and_submit_kernel` splits routed `top_k` experts per token into hot
//        VRAM hits (`0 <= slot < k_static`), optional GPU-staged cold misses (`k_static..k_hot`),
//        and CPU DDR5 cold misses, publishing the cold request to mapped pinned host memory
//        (`MoeBridgeReqSlot`) via a system-scope release store (`st.release.sys.global.b64`).
//      - `w34_moe_pull_mapped_experts_with_count` (`op_moe_router.cu`) optionally pulls
//        GPU-staged cold misses over zero-copy PCIe when `k_hot > k_static`.
//   2. `w34_moe_hybrid_wait_and_add`:
//      - `moe_hybrid_wait_and_add_kernel` waits on the Rust host worker pool's system-scope
//        response (`ld.acquire.sys.global.b64` on `MoeBridgeRespSlot::seq`) and adds the FP32
//        cold-expert output (`__ldcv`) into the GPU hot-expert output buffer `[num_tokens, hidden]`.
//
// All host-side thread pooling, core pinning, and AVX-512 `u4b8` expert evaluation live in Rust
// (`crates/w34-host/src/moe_cpu_bridge.rs`).

#include "../w34_cuda.h"
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cstddef>
#include <cstdint>

namespace w34_moe_bridge {

static constexpr int kMaxTokens = 8;
static constexpr int kMaxTopK = 16;
static constexpr int kMaxPairs = kMaxTokens * kMaxTopK; // 128
static constexpr int kMaxHidden = 4096;
static constexpr int kMaxGpuStagePull = 8;
static constexpr int kDefaultSingleTokenMaxGpuPull = 2;
static constexpr float kHighGpuShareThreshold = 0.20f;
static constexpr float kMultiTokenDefaultGpuShare = 0.36f;

__host__ __device__ __forceinline__ int hybrid_max_gpu_pull(
    int k_static,
    int k_hot,
    int num_tokens,
    float gpu_share_frac) {
    const int max_gpu_stage = (k_hot > k_static) ? (k_hot - k_static) : 0;
    const int cap_stage = (max_gpu_stage < kMaxGpuStagePull) ? max_gpu_stage : kMaxGpuStagePull;
    if (gpu_share_frac > kHighGpuShareThreshold || num_tokens > 1) {
        return cap_stage;
    }
    return (max_gpu_stage < kDefaultSingleTokenMaxGpuPull)
        ? max_gpu_stage
        : kDefaultSingleTokenMaxGpuPull;
}

struct alignas(64) MoeBridgeReqSlot {
    int64_t seq;
    int32_t num_tokens;
    int32_t top_k;
    int32_t _reserved0;
    int32_t num_cpu_pairs;
    int32_t hidden;
    int32_t intermediate;
    int32_t local_num_experts;
    int32_t _pad0[7];
    int32_t local_expert_ids[kMaxPairs];
    float topk_weights[kMaxPairs];
    uint16_t x_f16[kMaxTokens * kMaxHidden];
};

static_assert(offsetof(MoeBridgeReqSlot, x_f16) == 1088);
static_assert(sizeof(MoeBridgeReqSlot) == 66624);

struct alignas(64) MoeBridgeRespSlot {
    int64_t seq;
    int32_t status; // nonzero: host failure, published before seq; do not consume y
    int32_t _pad[13];
    float y_f32[kMaxTokens * kMaxHidden];
};
static_assert(offsetof(MoeBridgeRespSlot, status) == 8);
static_assert(offsetof(MoeBridgeRespSlot, y_f32) == 64);
static_assert(sizeof(MoeBridgeRespSlot) == 131136);

__global__ void moe_hybrid_plan_and_submit_kernel(
    const __half* __restrict__ x_f16,
    const int32_t* __restrict__ global_topk_ids,
    const float* __restrict__ topk_weights,
    int32_t* __restrict__ expert_to_slot,
    int32_t* __restrict__ slot_to_expert,
    uint64_t* __restrict__ slot_last_used,
    uint64_t* __restrict__ step_counter,
    int32_t* __restrict__ remapped_topk_ids,
    int32_t* __restrict__ miss_local_experts,
    int32_t* __restrict__ miss_victim_slots,
    int32_t* __restrict__ num_misses_out,
    MoeBridgeReqSlot* __restrict__ req_slot,
    int num_tokens,
    int top_k,
    int hidden,
    int intermediate,
    int global_expert_offset,
    int local_num_experts,
    int k_static,
    int k_hot,
    float gpu_share_frac) {
    __shared__ int32_t s_cpu_le[kMaxPairs];
    __shared__ int32_t s_n_cpu_cold;

    const int total_pairs = num_tokens * top_k;
    if (threadIdx.x == 0) {
        const uint64_t step = (*step_counter) + 1;
        *step_counter = step;

        int32_t distinct_cold[kMaxPairs];
        int32_t distinct_ge[kMaxPairs];
        int32_t cold_hits[kMaxPairs];
        int32_t cold_score[kMaxPairs];
        int n_cold = 0;

        for (int p = 0; p < total_pairs; ++p) {
            const int ge = global_topk_ids[p];
            const int64_t le64 = (int64_t)ge - global_expert_offset;
            if (le64 < 0 || le64 >= local_num_experts) {
                remapped_topk_ids[p] = -1;
                s_cpu_le[p] = -1;
                continue;
            }
            const int le = (int)le64;
            const int slot = expert_to_slot[le];
            if (slot >= 0 && slot < k_hot) {
                remapped_topk_ids[p] = slot;
                if (slot >= k_static) {
                    slot_last_used[slot] = step;
                }
                s_cpu_le[p] = -1;
            } else {
                int c_idx = 0;
                while (c_idx < n_cold && distinct_cold[c_idx] != le) ++c_idx;
                if (c_idx == n_cold) {
                    distinct_cold[n_cold] = le;
                    distinct_ge[n_cold] = ge;
                    cold_hits[n_cold] = 0;
                    cold_score[n_cold] = 0;
                    n_cold++;
                }
                cold_hits[c_idx] += 1;
                cold_score[c_idx] += (p < top_k) ? 4 : 2;
                // Temporarily store `-2 - c_idx` to mark the cold rank `c_idx`.
                remapped_topk_ids[p] = -2 - c_idx;
            }
        }

        const int max_pull = hybrid_max_gpu_pull(k_static, k_hot, num_tokens, gpu_share_frac);
        int want_gpu_cold = 0;
        if (max_pull > 0 && gpu_share_frac > 0.0f) {
            if (n_cold >= 7) {
                want_gpu_cold = 2;
            } else if (n_cold >= 4) {
                want_gpu_cold = 1;
            } else {
                for (int c = 0; c < n_cold; ++c) {
                    if (cold_hits[c] >= 2) {
                        want_gpu_cold++;
                    }
                }
            }
            if (gpu_share_frac > kHighGpuShareThreshold || num_tokens > 1) {
                const float eff_frac = (gpu_share_frac > kHighGpuShareThreshold)
                    ? gpu_share_frac
                    : kMultiTokenDefaultGpuShare;
                const int scaled = static_cast<int>(static_cast<float>(n_cold) * eff_frac + 0.5f);
                if (scaled > want_gpu_cold) {
                    want_gpu_cold = scaled;
                }
            }
            if (want_gpu_cold > max_pull) want_gpu_cold = max_pull;
            if (want_gpu_cold > n_cold) want_gpu_cold = n_cold;
        }

        int32_t cold_assigned_slot[kMaxPairs];
        for (int c = 0; c < n_cold; ++c) {
            cold_assigned_slot[c] = -1;
        }

        int n_gpu_cold = 0;
        for (int m = 0; m < want_gpu_cold; ++m) {
            int best_c = -1;
            int best_score = -1;
            for (int c = 0; c < n_cold; ++c) {
                if (cold_assigned_slot[c] < 0 && cold_score[c] > best_score) {
                    best_score = cold_score[c];
                    best_c = c;
                }
            }
            if (best_c < 0) break;

            uint64_t best_ts = step;
            int victim_slot = -1;
            for (int s = k_static; s < k_hot; ++s) {
                const uint64_t ts = slot_last_used[s];
                if (ts < best_ts) {
                    best_ts = ts;
                    victim_slot = s;
                }
            }
            if (victim_slot < 0) break;

            const int le = distinct_cold[best_c];
            const int ge = distinct_ge[best_c];
            const int old_ge = slot_to_expert[victim_slot];
            const int64_t old_le = (int64_t)old_ge - global_expert_offset;
            if (old_le >= 0 && old_le < local_num_experts) {
                expert_to_slot[old_le] = -1;
            }
            expert_to_slot[le] = victim_slot;
            slot_to_expert[victim_slot] = ge;
            slot_last_used[victim_slot] = step;
            cold_assigned_slot[best_c] = victim_slot;
            miss_local_experts[n_gpu_cold] = le;
            miss_victim_slots[n_gpu_cold] = victim_slot;
            n_gpu_cold++;
        }

        const int n_cpu_cold = n_cold - n_gpu_cold;
        num_misses_out[0] = n_gpu_cold;
        num_misses_out[1] = n_cpu_cold;
        s_n_cpu_cold = n_cpu_cold;

        for (int p = 0; p < total_pairs; ++p) {
            const int r = remapped_topk_ids[p];
            if (r <= -2) {
                const int c_idx = -2 - r;
                const int v_slot = cold_assigned_slot[c_idx];
                if (v_slot >= 0) {
                    remapped_topk_ids[p] = v_slot;
                    s_cpu_le[p] = -1;
                } else {
                    remapped_topk_ids[p] = -1;
                    s_cpu_le[p] = distinct_cold[c_idx];
                }
            }
        }

        if (n_cpu_cold > 0) {
            req_slot->num_tokens = num_tokens;
            req_slot->top_k = top_k;
            req_slot->_reserved0 = 0;
            req_slot->num_cpu_pairs = total_pairs;
            req_slot->hidden = hidden;
            req_slot->intermediate = intermediate;
            req_slot->local_num_experts = local_num_experts;
        }
    }
    __syncthreads();

    if (s_n_cpu_cold <= 0) return;

    for (int p = threadIdx.x; p < kMaxPairs; p += blockDim.x) {
        if (p < total_pairs) {
            const int32_t cle = s_cpu_le[p];
            req_slot->local_expert_ids[p] = cle;
            req_slot->topk_weights[p] = (cle >= 0) ? topk_weights[p] : 0.0f;
        } else {
            req_slot->local_expert_ids[p] = -1;
            req_slot->topk_weights[p] = 0.0f;
        }
    }

    const int total_elems = num_tokens * hidden;
    if (((total_elems & 7) == 0) && ((((uintptr_t)x_f16 | (uintptr_t)req_slot->x_f16) & 15u) == 0)) {
        const int vec_elems = total_elems >> 3;
        const uint4* src_v = reinterpret_cast<const uint4*>(x_f16);
        uint4* dst_v = reinterpret_cast<uint4*>(req_slot->x_f16);
        for (int i = threadIdx.x; i < vec_elems; i += blockDim.x) {
            dst_v[i] = src_v[i];
        }
    } else {
        const uint16_t* src_u16 = reinterpret_cast<const uint16_t*>(x_f16);
        for (int i = threadIdx.x; i < total_elems; i += blockDim.x) {
            req_slot->x_f16[i] = src_u16[i];
        }
    }
    __syncthreads();

    if (threadIdx.x == 0) {
        __threadfence_system();
        const int64_t next_seq = req_slot->seq + 1;
        asm volatile("st.release.sys.global.b64 [%0], %1;" ::"l"(&req_slot->seq), "l"(next_seq)
                     : "memory");
    }
}

__global__ void moe_hybrid_wait_and_add_kernel(
    __half* __restrict__ out_f16,
    const int32_t* __restrict__ num_misses_buf,
    const MoeBridgeReqSlot* __restrict__ req_slot,
    const MoeBridgeRespSlot* __restrict__ resp_slot,
    int total_elems) {
    if (num_misses_buf[1] <= 0) return;

    __shared__ int failed;
    if (threadIdx.x == 0) {
        const int64_t expected = *reinterpret_cast<const volatile int64_t*>(&req_slot->seq);
        int64_t v = 0;
        for (;;) {
            asm volatile("ld.acquire.sys.global.b64 %0, [%1];"
                         : "=l"(v)
                         : "l"(&resp_slot->seq)
                         : "memory");
            if (v == expected) break;
            __nanosleep(128);
        }
        failed = __ldcv(&resp_slot->status);
    }
    __syncthreads();
    // The Rust stream/event synchronization reports the sticky host error before
    // outputs can be consumed. Release the graph rather than waiting forever or
    // adding stale mailbox data. Its model state must not be reused after failure.
    if (failed) return;

    const int i4 = blockIdx.x * blockDim.x + threadIdx.x;
    if (4 * i4 < total_elems) {
        const float4 y = __ldcv(reinterpret_cast<const float4*>(resp_slot->y_f32) + i4);
        __half* o = out_f16 + 4 * i4;
        o[0] = __float2half(__half2float(o[0]) + y.x);
        o[1] = __float2half(__half2float(o[1]) + y.y);
        o[2] = __float2half(__half2float(o[2]) + y.z);
        o[3] = __float2half(__half2float(o[3]) + y.w);
    }
}

} // namespace w34_moe_bridge

extern "C" int w34_moe_hybrid_submit_and_pull(
    const w34_exec* ex,
    void* d_req_slot,
    const void* x_f16,
    const int32_t* global_topk_ids,
    const float* topk_weights,
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
    int num_tokens,
    int top_k,
    int global_expert_offset,
    int local_num_experts,
    int k_static,
    int k_hot,
    size_t pack_stride_bytes,
    int n_interm,
    int k_hidden,
    int group_size,
    float gpu_share_frac) {
    if (!ex || !d_req_slot || !x_f16 || !global_topk_ids || !topk_weights ||
        !expert_to_slot || !slot_to_expert || !slot_last_used || !step_counter ||
        !remapped_topk_ids || !miss_local_experts || !miss_victim_slots ||
        !num_misses_buf || !gate_up_packed_base || !gate_up_scales_base ||
        !down_packed_base || !down_scales_base || num_tokens <= 0 ||
        num_tokens > w34_moe_bridge::kMaxTokens || top_k <= 0 ||
        top_k > w34_moe_bridge::kMaxTopK || k_static <= 0 || k_hot < k_static || k_hidden <= 0 ||
        k_hidden > w34_moe_bridge::kMaxHidden) {
        return (int)cudaErrorInvalidValue;
    }

    const float effective_gpu_share = (mapped_host_packs != nullptr) ? gpu_share_frac : 0.0f;
    cudaStream_t stream = (cudaStream_t)ex->stream;
    w34_moe_bridge::moe_hybrid_plan_and_submit_kernel<<<1, 256, 0, stream>>>(
        static_cast<const __half*>(x_f16),
        global_topk_ids,
        topk_weights,
        expert_to_slot,
        slot_to_expert,
        slot_last_used,
        step_counter,
        remapped_topk_ids,
        miss_local_experts,
        miss_victim_slots,
        num_misses_buf,
        static_cast<w34_moe_bridge::MoeBridgeReqSlot*>(d_req_slot),
        num_tokens,
        top_k,
        k_hidden,
        n_interm,
        global_expert_offset,
        local_num_experts,
        k_static,
        k_hot,
        effective_gpu_share);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) return (int)err;

    const int max_pull = w34_moe_bridge::hybrid_max_gpu_pull(
        k_static, k_hot, num_tokens, effective_gpu_share);
    if (max_pull > 0 && mapped_host_packs != nullptr && effective_gpu_share > 0.0f) {
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
            max_pull,
            local_num_experts,
            k_hot,
            pack_stride_bytes,
            n_interm,
            k_hidden,
            group_size);
    }

    return 0;
}

extern "C" int w34_moe_hybrid_wait_and_add(
    const w34_exec* ex,
    const void* d_req_slot,
    const void* d_resp_slot,
    void* out_f16,
    const int32_t* num_misses_buf,
    int num_tokens,
    int k_hidden) {
    if (!ex || !d_req_slot || !d_resp_slot || !out_f16 || !num_misses_buf || num_tokens <= 0 ||
        k_hidden <= 0 || (k_hidden & 3) != 0) {
        return (int)cudaErrorInvalidValue;
    }
    const int total_elems = num_tokens * k_hidden;
    const int n4 = total_elems / 4;
    const int blocks = (n4 + 255) / 256;
    cudaStream_t stream = (cudaStream_t)ex->stream;
    w34_moe_bridge::moe_hybrid_wait_and_add_kernel<<<blocks, 256, 0, stream>>>(
        static_cast<__half*>(out_f16),
        num_misses_buf,
        static_cast<const w34_moe_bridge::MoeBridgeReqSlot*>(d_req_slot),
        static_cast<const w34_moe_bridge::MoeBridgeRespSlot*>(d_resp_slot),
        total_elems);
    return (int)cudaGetLastError();
}
