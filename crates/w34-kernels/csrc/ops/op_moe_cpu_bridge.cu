// op_moe_cpu_bridge.cu — CUDA-Graph-safe GPU mailbox kernels for hybrid CPU+GPU MoE offload.
//
// Device-side half of the hybrid MoE bridge (`crates/w34-host/src/moe_cpu_bridge.rs`):
//   1. `w34_moe_hybrid_submit_and_pull`:
//      - `moe_hybrid_plan_and_submit_kernel` splits routed `top_k` experts per token into hot
//        VRAM hits (`0 <= slot < k_static`), optional GPU-staged cold misses (`k_static..k_hot`),
//        and CPU DDR5 cold misses, publishing the cold request to mapped pinned host memory
//        (`MoeBridgeReqSlot`) via a system-scope release store (`st.release.sys.global.b64`).
//      - `moe_hybrid_pull_mapped_experts_kernel` optionally pulls GPU-staged cold misses over
//        zero-copy PCIe when `k_hot > k_static`.
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

static constexpr int kMaxTokens = 4;
static constexpr int kMaxTopK = 16;
static constexpr int kMaxPairs = kMaxTokens * kMaxTopK; // 64
static constexpr int kMaxHidden = 4096;

struct alignas(64) MoeBridgeReqSlot {
    int64_t seq;
    int32_t num_tokens;
    int32_t top_k;
    int32_t layer_id;
    int32_t num_cpu_pairs;
    int32_t hidden;
    int32_t intermediate;
    int32_t local_num_experts;
    int32_t _pad0[5];
    int32_t local_expert_ids[kMaxPairs];
    float topk_weights[kMaxPairs];
    float x_f32[kMaxTokens * kMaxHidden];
};

static_assert(offsetof(MoeBridgeReqSlot, x_f32) == 568);
static_assert(sizeof(MoeBridgeReqSlot) == 66112);

struct alignas(64) MoeBridgeRespSlot {
    int64_t seq;
    int32_t status; // nonzero: host failure, published before seq; do not consume y
    int32_t _pad[13];
    float y_f32[kMaxTokens * kMaxHidden];
};
static_assert(offsetof(MoeBridgeRespSlot, status) == 8);
static_assert(offsetof(MoeBridgeRespSlot, y_f32) == 64);
static_assert(sizeof(MoeBridgeRespSlot) == 65600);

__global__ void moe_hybrid_plan_and_submit_kernel(
    const __half* __restrict__ x_f16,
    const int32_t* __restrict__ global_topk_ids,
    const float* __restrict__ topk_weights,
    const int32_t* __restrict__ expert_to_slot,
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
    int layer_id,
    float gpu_share_frac) {
    __shared__ int32_t s_cpu_le[kMaxPairs];
    __shared__ int32_t s_n_cpu_cold;

    const int total_pairs = num_tokens * top_k;
    if (threadIdx.x == 0) {
        int32_t distinct_cold[kMaxPairs];
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
            if (slot >= 0 && slot < k_static) {
                remapped_topk_ids[p] = slot;
                s_cpu_le[p] = -1;
            } else {
                int c_idx = 0;
                while (c_idx < n_cold && distinct_cold[c_idx] != le) ++c_idx;
                if (c_idx == n_cold) {
                    distinct_cold[n_cold++] = le;
                }
                // Temporarily store `-2 - c_idx` to mark the cold rank `c_idx`.
                remapped_topk_ids[p] = -2 - c_idx;
            }
        }

        const int max_gpu_stage = (k_hot > k_static) ? (k_hot - k_static) : 0;
        int n_gpu_cold = (n_cold >= 7) ? (int)(n_cold * gpu_share_frac) : 0;
        if (n_gpu_cold > 2) n_gpu_cold = 2;
        if (n_gpu_cold > max_gpu_stage) n_gpu_cold = max_gpu_stage;
        if (n_gpu_cold > n_cold) n_gpu_cold = n_cold;
        const int n_cpu_cold = n_cold - n_gpu_cold;

        for (int m = 0; m < n_gpu_cold; ++m) {
            miss_local_experts[m] = distinct_cold[m];
            miss_victim_slots[m] = k_static + m;
        }
        num_misses_out[0] = n_gpu_cold;
        num_misses_out[1] = n_cpu_cold;
        s_n_cpu_cold = n_cpu_cold;

        for (int p = 0; p < total_pairs; ++p) {
            const int r = remapped_topk_ids[p];
            if (r <= -2) {
                const int c_idx = -2 - r;
                if (c_idx < n_gpu_cold) {
                    remapped_topk_ids[p] = k_static + c_idx;
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
            req_slot->layer_id = layer_id;
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
    for (int i = threadIdx.x; i < total_elems; i += blockDim.x) {
        req_slot->x_f32[i] = __half2float(x_f16[i]);
    }
    __threadfence_system();
    __syncthreads();

    if (threadIdx.x == 0) {
        const int64_t next_seq = req_slot->seq + 1;
        asm volatile("st.release.sys.global.b64 [%0], %1;" ::"l"(&req_slot->seq), "l"(next_seq)
                     : "memory");
    }
}

__global__ void moe_hybrid_pull_mapped_experts_kernel(
    const uint8_t* __restrict__ mapped_host_packs,
    const int32_t* __restrict__ src_local_experts,
    const int32_t* __restrict__ dst_slots,
    const int32_t* __restrict__ num_misses_buf,
    uint32_t* __restrict__ gate_up_packed_base,
    uint16_t* __restrict__ gate_up_scales_base,
    uint32_t* __restrict__ down_packed_base,
    uint16_t* __restrict__ down_scales_base,
    int local_num_experts,
    int dst_capacity,
    size_t pack_stride_bytes,
    size_t gu_packed_bytes,
    size_t gu_scales_bytes,
    size_t dn_packed_bytes,
    size_t dn_scales_bytes) {
    const int s = (int)blockIdx.y;
    const int num_misses = num_misses_buf[0];
    if (s >= num_misses) return;
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
    int layer_id,
    void* d_req_slot,
    const void* x_f16,
    const int32_t* global_topk_ids,
    const float* topk_weights,
    const int32_t* expert_to_slot,
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
    if (!ex || layer_id < 0 || !d_req_slot || !x_f16 || !global_topk_ids || !topk_weights ||
        !expert_to_slot || !remapped_topk_ids || !miss_local_experts || !miss_victim_slots ||
        !num_misses_buf || !mapped_host_packs || !gate_up_packed_base || !gate_up_scales_base ||
        !down_packed_base || !down_scales_base || num_tokens <= 0 ||
        num_tokens > w34_moe_bridge::kMaxTokens || top_k <= 0 ||
        top_k > w34_moe_bridge::kMaxTopK || k_static <= 0 || k_hot < k_static || k_hidden <= 0 ||
        k_hidden > w34_moe_bridge::kMaxHidden) {
        return (int)cudaErrorInvalidValue;
    }

    cudaStream_t stream = (cudaStream_t)ex->stream;
    w34_moe_bridge::moe_hybrid_plan_and_submit_kernel<<<1, 256, 0, stream>>>(
        static_cast<const __half*>(x_f16),
        global_topk_ids,
        topk_weights,
        expert_to_slot,
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
        layer_id,
        gpu_share_frac);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) return (int)err;

    const int max_gpu_stage = k_hot - k_static;
    if (max_gpu_stage > 0) {
        const size_t gu_packed_bytes =
            (size_t)(2 * n_interm) * (size_t)(k_hidden / 8) * sizeof(uint32_t);
        const size_t gu_scales_bytes =
            (size_t)(2 * n_interm) * (size_t)(k_hidden / group_size) * sizeof(uint16_t);
        const size_t dn_packed_bytes =
            (size_t)k_hidden * (size_t)(n_interm / 8) * sizeof(uint32_t);
        const size_t dn_scales_bytes =
            (size_t)k_hidden * (size_t)(n_interm / group_size) * sizeof(uint16_t);
        dim3 grid(64, (unsigned)max_gpu_stage);
        w34_moe_bridge::moe_hybrid_pull_mapped_experts_kernel<<<grid, 256, 0, stream>>>(
            static_cast<const uint8_t*>(mapped_host_packs),
            miss_local_experts,
            miss_victim_slots,
            num_misses_buf,
            gate_up_packed_base,
            static_cast<uint16_t*>(gate_up_scales_base),
            down_packed_base,
            static_cast<uint16_t*>(down_scales_base),
            local_num_experts,
            k_hot,
            pack_stride_bytes,
            gu_packed_bytes,
            gu_scales_bytes,
            dn_packed_bytes,
            dn_scales_bytes);
        err = cudaGetLastError();
        if (err != cudaSuccess) return (int)err;
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
