// Pinned-Sysmem TP=2 Collectives for warp-34 (2x RTX 3090 Ti, PCIe Gen4 x8, topo PHB).
// Donor: warp3090 op_all_reduce.cu + vLLM 0.31.0 custom_all_reduce.cuh.
//
// Invariants enforced (docs/contracts/collectives.md):
// - System-scope release/acquire fences (st.release.sys.global.u32 / ld.acquire.sys.global.u32)
// - Double-buffered staging keyed by monotonic per-block epoch parity (1 barrier/call,
//   CUDA Graph capture-safe)
// - Distinct typed entrypoints for FP16, BF16, FP32, INT32, and (FP32, INT32) top-1 argmax
//   (never casting token IDs or integer counters through FP16).
#include "../w34_cuda.h"

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <stdint.h>

namespace w34_comm {

constexpr int kMaxBlocks = 36;
constexpr int kRanks = 16;

struct Signal {
    alignas(128) uint32_t start[kMaxBlocks][kRanks];
    alignas(128) uint32_t end[kMaxBlocks][kRanks];
    alignas(128) uint32_t flag[kMaxBlocks];
};
static_assert(sizeof(Signal) == 4864, "Signal ABI size mismatch");

__device__ __forceinline__ void st_flag_release(uint32_t* addr, uint32_t val) {
    asm volatile("st.release.sys.global.u32 [%1], %0;" ::"r"(val), "l"(addr) : "memory");
}

__device__ __forceinline__ uint32_t ld_flag_acquire(uint32_t* addr) {
    uint32_t val;
    asm volatile("ld.acquire.sys.global.u32 %0, [%1];" : "=r"(val) : "l"(addr) : "memory");
    return val;
}

__device__ __forceinline__ void barrier_epoch_release(
    Signal* sig0,
    Signal* sig1,
    Signal* self_sg,
    int rank,
    uint32_t* out_epoch) {
    Signal* signals[2] = {sig0, sig1};
    __syncthreads();
    uint32_t epoch = self_sg->flag[blockIdx.x] + 1;
    if (threadIdx.x < 2) {
        uint32_t* peer_counter = &signals[threadIdx.x]->start[blockIdx.x][rank];
        uint32_t* self_counter = &self_sg->start[blockIdx.x][threadIdx.x];
        st_flag_release(peer_counter, epoch);
        while (ld_flag_acquire(self_counter) != epoch) {
            __nanosleep(128);
        }
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        self_sg->flag[blockIdx.x] = epoch;
    }
    *out_epoch = epoch;
}

__global__ void __launch_bounds__(512, 1) tp2_all_reduce_f16_kernel(
    const __half* __restrict__ local_in,
    __half* __restrict__ out,
    __half* s0,
    __half* s1,
    Signal* sig0,
    Signal* sig1,
    int rank,
    int num_elems) {
    Signal* self_sg = (rank == 0) ? sig0 : sig1;
    uint32_t epoch_pre = self_sg->flag[blockIdx.x] + 1;
    bool odd = (epoch_pre & 1u) != 0;

    __half* s0_buf = odd ? (s0 + num_elems) : s0;
    __half* s1_buf = odd ? (s1 + num_elems) : s1;
    __half* my_s = (rank == 0) ? s0_buf : s1_buf;

    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < num_elems; i += gridDim.x * blockDim.x) {
        my_s[i] = local_in[i];
    }

    uint32_t epoch = 0;
    barrier_epoch_release(sig0, sig1, self_sg, rank, &epoch);

    const __half* r0 = s0_buf;
    const __half* r1 = s1_buf;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < num_elems; i += gridDim.x * blockDim.x) {
        float sum = __half2float(r0[i]) + __half2float(r1[i]);
        out[i] = __float2half(sum);
    }
}

__global__ void __launch_bounds__(512, 1) tp2_all_reduce_bf16_kernel(
    const __nv_bfloat16* __restrict__ local_in,
    __nv_bfloat16* __restrict__ out,
    __nv_bfloat16* s0,
    __nv_bfloat16* s1,
    Signal* sig0,
    Signal* sig1,
    int rank,
    int num_elems) {
    Signal* self_sg = (rank == 0) ? sig0 : sig1;
    uint32_t epoch_pre = self_sg->flag[blockIdx.x] + 1;
    bool odd = (epoch_pre & 1u) != 0;

    __nv_bfloat16* s0_buf = odd ? (s0 + num_elems) : s0;
    __nv_bfloat16* s1_buf = odd ? (s1 + num_elems) : s1;
    __nv_bfloat16* my_s = (rank == 0) ? s0_buf : s1_buf;

    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < num_elems; i += gridDim.x * blockDim.x) {
        my_s[i] = local_in[i];
    }

    uint32_t epoch = 0;
    barrier_epoch_release(sig0, sig1, self_sg, rank, &epoch);

    const __nv_bfloat16* r0 = s0_buf;
    const __nv_bfloat16* r1 = s1_buf;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < num_elems; i += gridDim.x * blockDim.x) {
        float sum = __bfloat162float(r0[i]) + __bfloat162float(r1[i]);
        out[i] = __float2bfloat16(sum);
    }
}

__global__ void __launch_bounds__(512, 1) tp2_all_reduce_f32_kernel(
    const float* __restrict__ local_in,
    float* __restrict__ out,
    float* s0,
    float* s1,
    Signal* sig0,
    Signal* sig1,
    int rank,
    int num_elems) {
    Signal* self_sg = (rank == 0) ? sig0 : sig1;
    uint32_t epoch_pre = self_sg->flag[blockIdx.x] + 1;
    bool odd = (epoch_pre & 1u) != 0;

    float* s0_buf = odd ? (s0 + num_elems) : s0;
    float* s1_buf = odd ? (s1 + num_elems) : s1;
    float* my_s = (rank == 0) ? s0_buf : s1_buf;

    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < num_elems; i += gridDim.x * blockDim.x) {
        my_s[i] = local_in[i];
    }

    uint32_t epoch = 0;
    barrier_epoch_release(sig0, sig1, self_sg, rank, &epoch);

    const float* r0 = s0_buf;
    const float* r1 = s1_buf;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < num_elems; i += gridDim.x * blockDim.x) {
        out[i] = r0[i] + r1[i];
    }
}

__global__ void __launch_bounds__(256, 1) tp2_broadcast_i32_kernel(
    const int32_t* __restrict__ src_rank0,
    int32_t* __restrict__ out,
    int32_t* s0,
    int32_t* s1,
    Signal* sig0,
    Signal* sig1,
    int rank,
    int num_elems) {
    (void)s1;
    Signal* self_sg = (rank == 0) ? sig0 : sig1;
    uint32_t epoch_pre = self_sg->flag[blockIdx.x] + 1;
    bool odd = (epoch_pre & 1u) != 0;
    int32_t* s0_buf = odd ? (s0 + num_elems) : s0;

    if (rank == 0) {
        for (int i = threadIdx.x; i < num_elems; i += blockDim.x) {
            s0_buf[i] = src_rank0[i];
        }
    }

    uint32_t epoch = 0;
    barrier_epoch_release(sig0, sig1, self_sg, rank, &epoch);

    for (int i = threadIdx.x; i < num_elems; i += blockDim.x) {
        out[i] = s0_buf[i];
    }
}

struct Top1Pair {
    float   val;
    int32_t id;
};

__global__ void __launch_bounds__(256, 1) tp2_all_gather_top1_f32_i32_kernel(
    const float* __restrict__ local_max_val,
    const int32_t* __restrict__ local_max_id,
    float* __restrict__ global_max_val,
    int32_t* __restrict__ global_max_id,
    Top1Pair* s0,
    Top1Pair* s1,
    Signal* sig0,
    Signal* sig1,
    int rank,
    int num_rows) {
    Signal* self_sg = (rank == 0) ? sig0 : sig1;
    uint32_t epoch_pre = self_sg->flag[blockIdx.x] + 1;
    bool odd = (epoch_pre & 1u) != 0;

    Top1Pair* s0_buf = odd ? (s0 + num_rows) : s0;
    Top1Pair* s1_buf = odd ? (s1 + num_rows) : s1;
    Top1Pair* my_s = (rank == 0) ? s0_buf : s1_buf;

    for (int i = threadIdx.x; i < num_rows; i += blockDim.x) {
        my_s[i] = Top1Pair{local_max_val[i], local_max_id[i]};
    }

    uint32_t epoch = 0;
    barrier_epoch_release(sig0, sig1, self_sg, rank, &epoch);

    for (int i = threadIdx.x; i < num_rows; i += blockDim.x) {
        Top1Pair p0 = s0_buf[i];
        Top1Pair p1 = s1_buf[i];
        // Deterministic tie-break: lower token ID wins on exact float equality
        bool take0 = (p0.val > p1.val) || (p0.val == p1.val && p0.id <= p1.id);
        global_max_val[i] = take0 ? p0.val : p1.val;
        global_max_id[i] = take0 ? p0.id : p1.id;
    }
}

}  // namespace w34_comm

extern "C" size_t w34_tp2_signal_bytes(void) {
    return sizeof(w34_comm::Signal);
}

extern "C" int w34_tp2_all_reduce_f16(const w34_exec* ex, const void* local_in, void* out,
                                      void* stage0, void* stage1,
                                      void* sig0, void* sig1, int rank, int num_elems) {
    if (ex == nullptr || local_in == nullptr || out == nullptr ||
        stage0 == nullptr || stage1 == nullptr || sig0 == nullptr || sig1 == nullptr ||
        (rank != 0 && rank != 1) || num_elems <= 0) {
        return (int)cudaErrorInvalidValue;
    }
    int blocks = (num_elems + 511) / 512;
    if (blocks > w34_comm::kMaxBlocks) blocks = w34_comm::kMaxBlocks;
    if (blocks < 1) blocks = 1;
    w34_comm::tp2_all_reduce_f16_kernel<<<blocks, 512, 0, (cudaStream_t)ex->stream>>>(
        (const __half*)local_in, (__half*)out,
        (__half*)stage0, (__half*)stage1,
        (w34_comm::Signal*)sig0, (w34_comm::Signal*)sig1,
        rank, num_elems);
    return (int)cudaGetLastError();
}

extern "C" int w34_tp2_all_reduce_bf16(const w34_exec* ex, const void* local_in, void* out,
                                       void* stage0, void* stage1,
                                       void* sig0, void* sig1, int rank, int num_elems) {
    if (ex == nullptr || local_in == nullptr || out == nullptr ||
        stage0 == nullptr || stage1 == nullptr || sig0 == nullptr || sig1 == nullptr ||
        (rank != 0 && rank != 1) || num_elems <= 0) {
        return (int)cudaErrorInvalidValue;
    }
    int blocks = (num_elems + 511) / 512;
    if (blocks > w34_comm::kMaxBlocks) blocks = w34_comm::kMaxBlocks;
    if (blocks < 1) blocks = 1;
    w34_comm::tp2_all_reduce_bf16_kernel<<<blocks, 512, 0, (cudaStream_t)ex->stream>>>(
        (const __nv_bfloat16*)local_in, (__nv_bfloat16*)out,
        (__nv_bfloat16*)stage0, (__nv_bfloat16*)stage1,
        (w34_comm::Signal*)sig0, (w34_comm::Signal*)sig1,
        rank, num_elems);
    return (int)cudaGetLastError();
}

extern "C" int w34_tp2_all_reduce_f32(const w34_exec* ex, const float* local_in, float* out,
                                      float* stage0, float* stage1,
                                      void* sig0, void* sig1, int rank, int num_elems) {
    if (ex == nullptr || local_in == nullptr || out == nullptr ||
        stage0 == nullptr || stage1 == nullptr || sig0 == nullptr || sig1 == nullptr ||
        (rank != 0 && rank != 1) || num_elems <= 0) {
        return (int)cudaErrorInvalidValue;
    }
    int blocks = (num_elems + 511) / 512;
    if (blocks > w34_comm::kMaxBlocks) blocks = w34_comm::kMaxBlocks;
    if (blocks < 1) blocks = 1;
    w34_comm::tp2_all_reduce_f32_kernel<<<blocks, 512, 0, (cudaStream_t)ex->stream>>>(
        local_in, out, stage0, stage1,
        (w34_comm::Signal*)sig0, (w34_comm::Signal*)sig1,
        rank, num_elems);
    return (int)cudaGetLastError();
}

extern "C" int w34_tp2_broadcast_i32(const w34_exec* ex, const int32_t* src_rank0, int32_t* out,
                                     int32_t* stage0, int32_t* stage1,
                                     void* sig0, void* sig1, int rank, int num_elems) {
    if (ex == nullptr || (rank == 0 && src_rank0 == nullptr) || out == nullptr ||
        stage0 == nullptr || stage1 == nullptr || sig0 == nullptr || sig1 == nullptr ||
        (rank != 0 && rank != 1) || num_elems <= 0) {
        return (int)cudaErrorInvalidValue;
    }
    w34_comm::tp2_broadcast_i32_kernel<<<1, 256, 0, (cudaStream_t)ex->stream>>>(
        src_rank0, out, stage0, stage1,
        (w34_comm::Signal*)sig0, (w34_comm::Signal*)sig1,
        rank, num_elems);
    return (int)cudaGetLastError();
}

extern "C" int w34_tp2_all_gather_top1_f32_i32(
    const w34_exec* ex,
    const float* local_max_val,
    const int32_t* local_max_id,
    float* global_max_val,
    int32_t* global_max_id,
    void* stage0,
    void* stage1,
    void* sig0,
    void* sig1,
    int rank,
    int num_rows) {
    if (ex == nullptr || local_max_val == nullptr || local_max_id == nullptr ||
        global_max_val == nullptr || global_max_id == nullptr ||
        stage0 == nullptr || stage1 == nullptr || sig0 == nullptr || sig1 == nullptr ||
        (rank != 0 && rank != 1) || num_rows <= 0) {
        return (int)cudaErrorInvalidValue;
    }
    w34_comm::tp2_all_gather_top1_f32_i32_kernel<<<1, 256, 0, (cudaStream_t)ex->stream>>>(
        local_max_val, local_max_id, global_max_val, global_max_id,
        (w34_comm::Top1Pair*)stage0, (w34_comm::Top1Pair*)stage1,
        (w34_comm::Signal*)sig0, (w34_comm::Signal*)sig1,
        rank, num_rows);
    return (int)cudaGetLastError();
}
