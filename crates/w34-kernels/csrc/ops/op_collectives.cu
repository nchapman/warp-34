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
#include "hadamard_inline.cuh"

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <stdint.h>

namespace w34_comm {

// Sized for up to 144 SMs (covering all 30-series 28..84 SMs and 40-series 24..142 SMs).
constexpr int kMaxBlocks = 144;
constexpr int kSmallMaxBlocks = 36;
constexpr int kRanks = 16;
// Crossover where Copy-Engine DMA TX (`cudaMemcpyAsync` D2H at ~20.5 GB/s) + 2-barrier SM RX
// beats 1-barrier in-kernel SM write-back TX (`st.global.wb.v4.u32`) on PCIe Gen4 mapped sysmem.
constexpr size_t kDmaThresholdBytes = 65536;

struct Signal {
    alignas(128) uint32_t start[kMaxBlocks][kRanks];
    alignas(128) uint32_t end[kMaxBlocks][kRanks];
    alignas(128) uint32_t flag[kMaxBlocks];
};
static_assert(sizeof(Signal) == 19072, "Signal ABI size mismatch");

__device__ __forceinline__ void st_global_wb_v4(uint4* addr, uint4 val) {
    asm volatile("st.global.wb.v4.u32 [%0], {%1, %2, %3, %4};"
                 :: "l"(addr), "r"(val.x), "r"(val.y), "r"(val.z), "r"(val.w)
                 : "memory");
}

template <typename T>
__device__ __forceinline__ T* stage_epoch_ptr(T* base, const Signal* sig, bool odd, size_t fallback_elems) {
    size_t cap_bytes = (size_t)sig->end[128][0] & ~15ULL;
    size_t min_bytes = fallback_elems * sizeof(T);
    if (cap_bytes >= kDmaThresholdBytes && min_bytes < kDmaThresholdBytes) {
        size_t base_off = (cap_bytes * 2u) + (odd ? kDmaThresholdBytes : 0u);
        return reinterpret_cast<T*>(reinterpret_cast<char*>(base) + base_off);
    }
    if (!odd) return base;
    size_t off_bytes = (cap_bytes >= min_bytes) ? cap_bytes : min_bytes;
    return reinterpret_cast<T*>(reinterpret_cast<char*>(base) + off_bytes);
}

template <typename T>
__device__ __forceinline__ T* small_epoch_ptr(T* fallback_base, Signal* sig, bool odd, size_t num_elems) {
    if (num_elems * sizeof(T) <= 4096) {
        return reinterpret_cast<T*>(odd ? &sig->end[64][0] : &sig->end[0][0]);
    }
    return stage_epoch_ptr(fallback_base, sig, odd, num_elems);
}

inline int tp2_block_cap(bool is_small) {
    int min_sms = 84;
    if (w34_get_tp2_min_sm_count(&min_sms) != 0 || min_sms < 1) {
        min_sms = 1;
    }
    int cap = is_small ? kSmallMaxBlocks : kMaxBlocks;
    if (cap > min_sms) cap = min_sms;
    if (cap < 1) cap = 1;
    return cap;
}

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
    __syncthreads();
    uint32_t epoch = self_sg->flag[blockIdx.x] + 1;
    if (threadIdx.x == 0) {
        // Only write to peer's Signal struct (`peer_sg->start[blockIdx.x][rank]`) and poll
        // our own slot written by the peer (`self_sg->start[blockIdx.x][1 - rank]`).
        // Writing to `self_sg->start[blockIdx.x][rank]` would dirty the same 128-byte host
        // cache line that the peer GPU's PCIe Posted Write updates, causing a coherence
        // bounce across the CPU root complex on every barrier.
        Signal* peer_sg = (rank == 0) ? sig1 : sig0;
        uint32_t* peer_counter = &peer_sg->start[blockIdx.x][rank];
        uint32_t* self_counter = &self_sg->start[blockIdx.x][1 - rank];
        st_flag_release(peer_counter, epoch);
        while (static_cast<int32_t>(ld_flag_acquire(self_counter) - epoch) < 0) {
        }
        self_sg->flag[blockIdx.x] = epoch;
    }
    __syncthreads();
    *out_epoch = epoch;
}

__device__ __forceinline__ float warp_reduce_sum_f32(float val) {
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        val += __shfl_xor_sync(0xffffffffu, val, offset);
    }
    return val;
}

template <int THREADS = 256>
__device__ __forceinline__ float block_reduce_sum(float val, float* s_warp) {
    constexpr int NUM_WARPS = THREADS / 32;
    const int lane = threadIdx.x & 31;
    const int wid = threadIdx.x >> 5;
    val = warp_reduce_sum_f32(val);
    if (lane == 0) {
        s_warp[wid] = val;
    }
    __syncthreads();
    val = (threadIdx.x < NUM_WARPS) ? s_warp[lane] : 0.0f;
    if (wid == 0) {
        val = warp_reduce_sum_f32(val);
        if (lane == 0) {
            s_warp[0] = val;
        }
    }
    __syncthreads();
    const float total = s_warp[0];
    __syncthreads();
    return total;
}

__global__ void __launch_bounds__(512, 1) tp2_all_reduce_f16_kernel(
    const __half* local_in,
    __half* out,
    __half* __restrict__ s0,
    __half* __restrict__ s1,
    Signal* __restrict__ sig0,
    Signal* __restrict__ sig1,
    int rank,
    int num_elems) {
    Signal* self_sg = (rank == 0) ? sig0 : sig1;
    const bool dma_tx = ((size_t)num_elems * sizeof(__half) >= kDmaThresholdBytes);
    uint32_t epoch_pre = self_sg->flag[blockIdx.x] + 1;
    bool odd = (epoch_pre & 1u) != 0;

    __half* s0_buf = dma_tx ? s0 : stage_epoch_ptr(s0, self_sg, odd, num_elems);
    __half* s1_buf = dma_tx ? s1 : stage_epoch_ptr(s1, self_sg, odd, num_elems);
    __half* __restrict__ my_s = (rank == 0) ? s0_buf : s1_buf;
    const __half* __restrict__ peer_s = (rank == 0) ? s1_buf : s0_buf;

    const bool vec8 = ((num_elems & 7) == 0) &&
                      ((((uintptr_t)local_in | (uintptr_t)out | (uintptr_t)my_s | (uintptr_t)peer_s) & 15u) == 0);
    if (!dma_tx) {
        if (vec8) {
            const int vec_elems = num_elems >> 3;
            const uint4* local_v = reinterpret_cast<const uint4*>(local_in);
            uint4* my_v = reinterpret_cast<uint4*>(my_s);
            for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < vec_elems; i += gridDim.x * blockDim.x) {
                st_global_wb_v4(my_v + i, local_v[i]);
            }
        } else {
            for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < num_elems; i += gridDim.x * blockDim.x) {
                my_s[i] = local_in[i];
            }
        }
    }

    uint32_t epoch = 0;
    barrier_epoch_release(sig0, sig1, self_sg, rank, &epoch);

    if (vec8) {
        const int vec_elems = num_elems >> 3;
        const uint4* local_v = reinterpret_cast<const uint4*>(local_in);
        const uint4* peer_v = reinterpret_cast<const uint4*>(peer_s);
        uint4* out_v = reinterpret_cast<uint4*>(out);
        for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < vec_elems; i += gridDim.x * blockDim.x) {
            const uint4 lv = local_v[i];
            const uint4 pv = peer_v[i];
            const __half* lh = reinterpret_cast<const __half*>(&lv);
            const __half* ph = reinterpret_cast<const __half*>(&pv);
            uint4 ov;
            __half* oh = reinterpret_cast<__half*>(&ov);
#pragma unroll
            for (int d = 0; d < 8; ++d) {
                const float fl = __half2float(lh[d]);
                const float fp = __half2float(ph[d]);
                oh[d] = __float2half((rank == 0) ? (fl + fp) : (fp + fl));
            }
            out_v[i] = ov;
        }
    } else {
        for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < num_elems; i += gridDim.x * blockDim.x) {
            const float fl = __half2float(local_in[i]);
            const float fp = __half2float(peer_s[i]);
            out[i] = __float2half((rank == 0) ? (fl + fp) : (fp + fl));
        }
    }
    if (dma_tx) {
        barrier_epoch_release(sig0, sig1, self_sg, rank, &epoch);
    }
}

template <int THREADS, int MAX_ITERS>
__global__ void __launch_bounds__(THREADS, 2) tp2_all_reduce_residual_gemma_rms_norm_f16_kernel(
    const __half* local_in,
    __half* ar_out,
    __half* __restrict__ residual,
    const __half* __restrict__ weight,
    __half* __restrict__ out,
    __half* __restrict__ out_had64,
    __nv_bfloat16* __restrict__ tap_dst_bf16,
    __half* __restrict__ s0,
    __half* __restrict__ s1,
    Signal* __restrict__ sig0,
    Signal* __restrict__ sig1,
    int rank,
    int num_tokens,
    int hidden_size,
    float eps,
    int tap_row_stride,
    int tap_col_offset) {
    const int num_elems = num_tokens * hidden_size;
    const bool dma_tx = ((size_t)num_elems * sizeof(__half) >= kDmaThresholdBytes);
    Signal* self_sg = (rank == 0) ? sig0 : sig1;
    uint32_t epoch_pre = self_sg->flag[blockIdx.x] + 1;
    bool odd = (epoch_pre & 1u) != 0;

    __half* s0_buf = dma_tx ? s0 : stage_epoch_ptr(s0, self_sg, odd, num_elems);
    __half* s1_buf = dma_tx ? s1 : stage_epoch_ptr(s1, self_sg, odd, num_elems);
    __half* __restrict__ my_s = (rank == 0) ? s0_buf : s1_buf;
    const __half* __restrict__ peer_s = (rank == 0) ? s1_buf : s0_buf;

    constexpr int VEC_ITERS = (MAX_ITERS + 7) / 8;
    const bool vec8 = ((hidden_size & 7) == 0) &&
                      ((((uintptr_t)local_in | (uintptr_t)my_s | (uintptr_t)peer_s |
                         (uintptr_t)residual | (uintptr_t)weight | (uintptr_t)out |
                         (uintptr_t)out_had64 | (uintptr_t)ar_out) & 15u) == 0);
    // When `dma_tx` is active (`>= 64 KiB`), Copy-Engine `cudaMemcpyAsync` has already staged
    // all rows into `my_s`, so we run a single RX pass bracketed by two barriers.
    const int num_stages = dma_tx
        ? 1
        : ((num_tokens > (int)gridDim.x * 12)
               ? 5
               : ((num_tokens > (int)gridDim.x * 4) ? 3 : 1));
    const int stage_rows = (num_tokens + num_stages - 1) / num_stages;

    __shared__ float s_warp[32];

    auto tx_stage = [&](int st) {
        const int rb = st * stage_rows;
        int re = rb + stage_rows;
        if (re > num_tokens) re = num_tokens;
        if (vec8) {
            const int vec_cols = hidden_size >> 3;
            for (int row = rb + (int)blockIdx.x; row < re; row += (int)gridDim.x) {
                const uint4* src_v = reinterpret_cast<const uint4*>(local_in + (size_t)row * hidden_size);
                uint4* dst_v = reinterpret_cast<uint4*>(my_s + (size_t)row * hidden_size);
#pragma unroll
                for (int k = 0; k < VEC_ITERS; ++k) {
                    const int c = threadIdx.x + k * THREADS;
                    if (c < vec_cols) {
                        st_global_wb_v4(dst_v + c, src_v[c]);
                    }
                }
            }
        } else {
            for (int row = rb + (int)blockIdx.x; row < re; row += (int)gridDim.x) {
                const __half* src_r = local_in + (size_t)row * hidden_size;
                __half* dst_r = my_s + (size_t)row * hidden_size;
                for (int c = threadIdx.x; c < hidden_size; c += THREADS) {
                    dst_r[c] = src_r[c];
                }
            }
        }
    };

    if (!dma_tx) {
        tx_stage(0);
    }

    for (int stage = 0; stage < num_stages; ++stage) {
        const int row_begin = stage * stage_rows;
        int row_end = row_begin + stage_rows;
        if (row_end > num_tokens) row_end = num_tokens;

        uint32_t epoch = 0;
        barrier_epoch_release(sig0, sig1, self_sg, rank, &epoch);

        // Software-pipeline TX of `stage + 1` onto PCIe Posted Write queues immediately
        // after `stage` barrier release so TX(stage + 1) overlaps in full duplex with
        // RX(stage) non-posted reads across PCIe.
        if (!dma_tx && stage + 1 < num_stages) {
            tx_stage(stage + 1);
        }

        if (vec8) {
            const int vec_cols = hidden_size >> 3;
            const uint4* weight_v = reinterpret_cast<const uint4*>(weight);
            for (int row = row_begin + (int)blockIdx.x; row < row_end; row += (int)gridDim.x) {
                const uint4* local_v = reinterpret_cast<const uint4*>(local_in + (size_t)row * hidden_size);
                const uint4* peer_v = reinterpret_cast<const uint4*>(peer_s + (size_t)row * hidden_size);
                uint4* ar_v = (ar_out != nullptr)
                    ? reinterpret_cast<uint4*>(ar_out + (size_t)row * hidden_size)
                    : nullptr;
                uint4* res_v = reinterpret_cast<uint4*>(residual + (size_t)row * hidden_size);
                uint4* out_v = (out != nullptr)
                    ? reinterpret_cast<uint4*>(out + (size_t)row * hidden_size)
                    : nullptr;
                uint4* out_had64_v = (out_had64 != nullptr)
                    ? reinterpret_cast<uint4*>(out_had64 + (size_t)row * hidden_size)
                    : nullptr;
                __nv_bfloat16* tap_row = (tap_dst_bf16 != nullptr)
                    ? (tap_dst_bf16 + (size_t)row * tap_row_stride + tap_col_offset)
                    : nullptr;

                float v_reg[VEC_ITERS][8];
                float sum_sq = 0.0f;
#pragma unroll
                for (int k = 0; k < VEC_ITERS; ++k) {
                    const int c = threadIdx.x + k * THREADS;
                    if (c < vec_cols) {
                        const uint4 lv = local_v[c];
                        const uint4 pv = peer_v[c];
                        const uint4 rv = res_v[c];
                        const __half* lh = reinterpret_cast<const __half*>(&lv);
                        const __half* ph = reinterpret_cast<const __half*>(&pv);
                        const __half* rh = reinterpret_cast<const __half*>(&rv);
                        uint4 arv, sumv;
                        __half* arh = reinterpret_cast<__half*>(&arv);
                        __half* sh = reinterpret_cast<__half*>(&sumv);
                        const int base_i = c << 3;
#pragma unroll
                        for (int d = 0; d < 8; ++d) {
                            const float lf = __half2float(lh[d]);
                            const float pf = __half2float(ph[d]);
                            const float ar_f = (rank == 0) ? (lf + pf) : (pf + lf);
                            const __half ar_h = __float2half(ar_f);
                            arh[d] = ar_h;
                            const __half res_old = rh[d];
                            const float ar_hf = __half2float(ar_h);
                            const float res_hf = __half2float(res_old);
                            if (tap_row != nullptr) {
                                tap_row[base_i + d] = __float2bfloat16(res_hf + ar_hf);
                            }
                            sh[d] = __hadd(ar_h, res_old);
                            const float v = ar_hf + res_hf;
                            v_reg[k][d] = v;
                            sum_sq += v * v;
                        }
                        if (ar_v != nullptr) {
                            ar_v[c] = arv;
                        }
                        res_v[c] = sumv;
                    }
                }
                sum_sq = block_reduce_sum<THREADS>(sum_sq, s_warp);
                const float inv_rms = rsqrtf(sum_sq / (float)hidden_size + eps);

#pragma unroll
                for (int k = 0; k < VEC_ITERS; ++k) {
                    const int c = threadIdx.x + k * THREADS;
                    if (c < vec_cols) {
                        const uint4 wv = weight_v[c];
                        const __half* wh = reinterpret_cast<const __half*>(&wv);
                        uint4 ov;
                        __half* oh = reinterpret_cast<__half*>(&ov);
#pragma unroll
                        for (int d = 0; d < 8; ++d) {
                            const float v = v_reg[k][d] * inv_rms;
                            const float w = 1.0f + __half2float(wh[d]);
                            oh[d] = __float2half(v * w);
                        }
                        if (out_v != nullptr) {
                            out_v[c] = ov;
                        }
                        if (out_had64_v != nullptr) {
                            out_had64_v[c] = w34::hadamard64_vec8_f16(ov, (int)threadIdx.x);
                        }
                    }
                }
            }
        } else {
            for (int row = row_begin + (int)blockIdx.x; row < row_end; row += (int)gridDim.x) {
                const __half* local_row = local_in + (size_t)row * hidden_size;
                const __half* peer_row = peer_s + (size_t)row * hidden_size;
                __half* ar_row = (ar_out != nullptr) ? (ar_out + (size_t)row * hidden_size) : nullptr;
                __half* res_row = residual + (size_t)row * hidden_size;
                __half* out_row = out + (size_t)row * hidden_size;
                __nv_bfloat16* tap_row = (tap_dst_bf16 != nullptr)
                    ? (tap_dst_bf16 + (size_t)row * tap_row_stride + tap_col_offset)
                    : nullptr;

                float v_scalar[MAX_ITERS];
                float sum_sq = 0.0f;
#pragma unroll
                for (int k = 0; k < MAX_ITERS; k++) {
                    const int i = threadIdx.x + k * THREADS;
                    if (i < hidden_size) {
                        const float local_f = __half2float(local_row[i]);
                        const float peer_f = __half2float(peer_row[i]);
                        const float ar_f = (rank == 0) ? (local_f + peer_f) : (peer_f + local_f);
                        const __half ar_h = __float2half(ar_f);
                        if (ar_row != nullptr) {
                            ar_row[i] = ar_h;
                        }
                        const __half res_old = res_row[i];
                        if (tap_row != nullptr) {
                            tap_row[i] = __float2bfloat16(__half2float(res_old) + __half2float(ar_h));
                        }
                        const __half sum_h = __hadd(ar_h, res_old);
                        res_row[i] = sum_h;
                        const float v = __half2float(ar_h) + __half2float(res_old);
                        v_scalar[k] = v;
                        sum_sq += v * v;
                    }
                }
                sum_sq = block_reduce_sum<THREADS>(sum_sq, s_warp);
                const float inv_rms = rsqrtf(sum_sq / (float)hidden_size + eps);

#pragma unroll
                for (int k = 0; k < MAX_ITERS; k++) {
                    const int i = threadIdx.x + k * THREADS;
                    if (i < hidden_size) {
                        const float v = v_scalar[k] * inv_rms;
                        const float w = 1.0f + __half2float(weight[i]);
                        out_row[i] = __float2half(v * w);
                    }
                }
            }
        }
        if (dma_tx) {
            barrier_epoch_release(sig0, sig1, self_sg, rank, &epoch);
        }
    }
}

__global__ void __launch_bounds__(512, 1) tp2_all_reduce_bf16_kernel(
    const __nv_bfloat16* local_in,
    __nv_bfloat16* out,
    __nv_bfloat16* __restrict__ s0,
    __nv_bfloat16* __restrict__ s1,
    Signal* __restrict__ sig0,
    Signal* __restrict__ sig1,
    int rank,
    int num_elems) {
    Signal* self_sg = (rank == 0) ? sig0 : sig1;
    const bool dma_tx = ((size_t)num_elems * sizeof(__nv_bfloat16) >= kDmaThresholdBytes);
    uint32_t epoch_pre = self_sg->flag[blockIdx.x] + 1;
    bool odd = (epoch_pre & 1u) != 0;

    __nv_bfloat16* s0_buf = dma_tx ? s0 : stage_epoch_ptr(s0, self_sg, odd, num_elems);
    __nv_bfloat16* s1_buf = dma_tx ? s1 : stage_epoch_ptr(s1, self_sg, odd, num_elems);
    __nv_bfloat16* __restrict__ my_s = (rank == 0) ? s0_buf : s1_buf;
    const __nv_bfloat16* __restrict__ peer_s = (rank == 0) ? s1_buf : s0_buf;

    const bool vec8 = ((num_elems & 7) == 0) &&
                      ((((uintptr_t)local_in | (uintptr_t)out | (uintptr_t)my_s | (uintptr_t)peer_s) & 15u) == 0);
    if (!dma_tx) {
        if (vec8) {
            const int vec_elems = num_elems >> 3;
            const uint4* local_v = reinterpret_cast<const uint4*>(local_in);
            uint4* my_v = reinterpret_cast<uint4*>(my_s);
            for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < vec_elems; i += gridDim.x * blockDim.x) {
                st_global_wb_v4(my_v + i, local_v[i]);
            }
        } else {
            for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < num_elems; i += gridDim.x * blockDim.x) {
                my_s[i] = local_in[i];
            }
        }
    }

    uint32_t epoch = 0;
    barrier_epoch_release(sig0, sig1, self_sg, rank, &epoch);

    if (vec8) {
        const int vec_elems = num_elems >> 3;
        const uint4* local_v = reinterpret_cast<const uint4*>(local_in);
        const uint4* peer_v = reinterpret_cast<const uint4*>(peer_s);
        uint4* out_v = reinterpret_cast<uint4*>(out);
        for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < vec_elems; i += gridDim.x * blockDim.x) {
            const uint4 lv = local_v[i];
            const uint4 pv = peer_v[i];
            const __nv_bfloat16* lb = reinterpret_cast<const __nv_bfloat16*>(&lv);
            const __nv_bfloat16* pb = reinterpret_cast<const __nv_bfloat16*>(&pv);
            uint4 ov;
            __nv_bfloat16* ob = reinterpret_cast<__nv_bfloat16*>(&ov);
#pragma unroll
            for (int d = 0; d < 8; ++d) {
                const float fl = __bfloat162float(lb[d]);
                const float fp = __bfloat162float(pb[d]);
                ob[d] = __float2bfloat16((rank == 0) ? (fl + fp) : (fp + fl));
            }
            out_v[i] = ov;
        }
    } else {
        for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < num_elems; i += gridDim.x * blockDim.x) {
            const float fl = __bfloat162float(local_in[i]);
            const float fp = __bfloat162float(peer_s[i]);
            out[i] = __float2bfloat16((rank == 0) ? (fl + fp) : (fp + fl));
        }
    }
    if (dma_tx) {
        barrier_epoch_release(sig0, sig1, self_sg, rank, &epoch);
    }
}

__global__ void __launch_bounds__(512, 1) tp2_all_reduce_f32_kernel(
    const float* local_in,
    float* out,
    float* s0,
    float* s1,
    Signal* sig0,
    Signal* sig1,
    int rank,
    int num_elems) {
    Signal* self_sg = (rank == 0) ? sig0 : sig1;
    const bool dma_tx = ((size_t)num_elems * sizeof(float) >= kDmaThresholdBytes);
    uint32_t epoch_pre = self_sg->flag[blockIdx.x] + 1;
    bool odd = (epoch_pre & 1u) != 0;

    float* s0_buf = dma_tx ? s0 : stage_epoch_ptr(s0, self_sg, odd, num_elems);
    float* s1_buf = dma_tx ? s1 : stage_epoch_ptr(s1, self_sg, odd, num_elems);
    float* my_s = (rank == 0) ? s0_buf : s1_buf;
    const float* peer_s = (rank == 0) ? s1_buf : s0_buf;

    const bool vec4 = ((num_elems & 3) == 0) &&
                      ((((uintptr_t)local_in | (uintptr_t)out | (uintptr_t)my_s | (uintptr_t)peer_s) & 15u) == 0);
    if (!dma_tx) {
        if (vec4) {
            const int vec_elems = num_elems >> 2;
            const uint4* local_v = reinterpret_cast<const uint4*>(local_in);
            uint4* my_v = reinterpret_cast<uint4*>(my_s);
            for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < vec_elems; i += gridDim.x * blockDim.x) {
                st_global_wb_v4(my_v + i, local_v[i]);
            }
        } else {
            for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < num_elems; i += gridDim.x * blockDim.x) {
                my_s[i] = local_in[i];
            }
        }
    }

    uint32_t epoch = 0;
    barrier_epoch_release(sig0, sig1, self_sg, rank, &epoch);

    if (vec4) {
        const int vec_elems = num_elems >> 2;
        const uint4* local_v = reinterpret_cast<const uint4*>(local_in);
        const uint4* peer_v = reinterpret_cast<const uint4*>(peer_s);
        uint4* out_v = reinterpret_cast<uint4*>(out);
        for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < vec_elems; i += gridDim.x * blockDim.x) {
            const uint4 lv = local_v[i];
            const uint4 pv = peer_v[i];
            const float* lf = reinterpret_cast<const float*>(&lv);
            const float* pf = reinterpret_cast<const float*>(&pv);
            uint4 ov;
            float* of = reinterpret_cast<float*>(&ov);
#pragma unroll
            for (int d = 0; d < 4; ++d) {
                of[d] = (rank == 0) ? (lf[d] + pf[d]) : (pf[d] + lf[d]);
            }
            out_v[i] = ov;
        }
    } else {
        for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < num_elems; i += gridDim.x * blockDim.x) {
            const float lf = local_in[i];
            const float pf = peer_s[i];
            out[i] = (rank == 0) ? (lf + pf) : (pf + lf);
        }
    }
    if (dma_tx) {
        barrier_epoch_release(sig0, sig1, self_sg, rank, &epoch);
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
    int32_t* s0_buf = small_epoch_ptr(s0, sig0, odd, num_elems);

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

    Top1Pair* s0_buf = small_epoch_ptr(s0, sig0, odd, num_rows);
    Top1Pair* s1_buf = small_epoch_ptr(s1, sig1, odd, num_rows);
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

__global__ void __launch_bounds__(256, 1) tp2_all_gather_topk_f32_i32_kernel(
    const float* local_vals, const int32_t* local_ids,
    float* global_vals, int32_t* global_ids,
    Top1Pair* s0, Top1Pair* s1, Signal* sig0, Signal* sig1,
    int rank, int num_rows, int k) {
    const int num_elems = num_rows * k;
    Signal* self_sg = (rank == 0) ? sig0 : sig1;
    bool odd = ((self_sg->flag[blockIdx.x] + 1) & 1u) != 0;
    Top1Pair* s0_buf = small_epoch_ptr(s0, sig0, odd, num_elems);
    Top1Pair* s1_buf = small_epoch_ptr(s1, sig1, odd, num_elems);
    Top1Pair* my_s = (rank == 0) ? s0_buf : s1_buf;
    for (int i = threadIdx.x; i < num_elems; i += blockDim.x) {
        my_s[i] = Top1Pair{local_vals[i], local_ids[i]};
    }
    uint32_t epoch = 0;
    barrier_epoch_release(sig0, sig1, self_sg, rank, &epoch);
    for (int row = threadIdx.x; row < num_rows; row += blockDim.x) {
        const Top1Pair* r0 = s0_buf + (size_t)row * k;
        const Top1Pair* r1 = s1_buf + (size_t)row * k;
        float* out_v = global_vals + (size_t)row * k;
        int32_t* out_i = global_ids + (size_t)row * k;
        int i0 = 0, i1 = 0;
        for (int j = 0; j < k; ++j) {
            const Top1Pair p0 = r0[i0];
            const Top1Pair p1 = r1[i1];
            if (!isfinite(p0.val) || !isfinite(p1.val) || p0.id < 0 || p1.id < 0) {
                out_v[j] = NAN;
                out_i[j] = -1;
                continue;
            }
            const bool take0 = (p0.val > p1.val) || (p0.val == p1.val && p0.id <= p1.id);
            out_v[j] = take0 ? p0.val : p1.val;
            out_i[j] = take0 ? p0.id : p1.id;
            if (take0) ++i0; else ++i1;
        }
    }
}

__global__ void __launch_bounds__(256, 1) tp2_all_reduce_hc_combine_tap_f16_kernel(
    const __half* residual_in, const __half* local_in, __half* ar_out,
    const float* inject_weights, const __half* ple_delta, __half* residual_out,
    __nv_bfloat16* tap_bf16_out, const uint8_t* valid_token_mask,
    __half* s0, __half* s1, Signal* sig0, Signal* sig1,
    int rank, int num_tokens, int hc_count, int hidden_size,
    int tap_stride_elems, int tap_offset_elems) {
    const int num_elems = num_tokens * hidden_size;
    const bool dma_tx = ((size_t)num_elems * sizeof(__half) >= kDmaThresholdBytes);
    Signal* self_sg = (rank == 0) ? sig0 : sig1;
    bool odd = ((self_sg->flag[blockIdx.x] + 1) & 1u) != 0;
    __half* s0_buf = dma_tx ? s0 : stage_epoch_ptr(s0, self_sg, odd, num_elems);
    __half* s1_buf = dma_tx ? s1 : stage_epoch_ptr(s1, self_sg, odd, num_elems);
    __half* my_s = (rank == 0) ? s0_buf : s1_buf;
    const __half* peer_s = (rank == 0) ? s1_buf : s0_buf;
    const bool vec8 = ((hidden_size & 7) == 0) &&
                      ((((uintptr_t)residual_in | (uintptr_t)local_in | (uintptr_t)ar_out |
                         (uintptr_t)ple_delta | (uintptr_t)residual_out | (uintptr_t)my_s |
                         (uintptr_t)peer_s) & 15u) == 0);
    if (!dma_tx) {
        if (vec8) {
            const int vec_cols = hidden_size >> 3;
            for (int tok = blockIdx.x; tok < num_tokens; tok += gridDim.x) {
                const uint4* src_v = reinterpret_cast<const uint4*>(local_in + (size_t)tok * hidden_size);
                uint4* dst_v = reinterpret_cast<uint4*>(my_s + (size_t)tok * hidden_size);
                for (int c = threadIdx.x; c < vec_cols; c += blockDim.x) {
                    st_global_wb_v4(dst_v + c, src_v[c]);
                }
            }
        } else {
            for (int tok = blockIdx.x; tok < num_tokens; tok += gridDim.x) {
                const size_t roff = (size_t)tok * hidden_size;
                for (int d = threadIdx.x; d < hidden_size; d += blockDim.x) {
                    my_s[roff + d] = local_in[roff + d];
                }
            }
        }
    }
    uint32_t epoch = 0;
    barrier_epoch_release(sig0, sig1, self_sg, rank, &epoch);
    const float inv_hc = 1.0f / static_cast<float>(hc_count);
    for (int tok = blockIdx.x; tok < num_tokens; tok += gridDim.x) {
        if (valid_token_mask != nullptr && valid_token_mask[tok] == 0) continue;
        const size_t base = (size_t)tok * hc_count * hidden_size;
        const __half* res_in_tok = residual_in + base;
        __half* res_out_tok = residual_out + base;
        const __half* ple_tok = (ple_delta != nullptr) ? (ple_delta + base) : nullptr;
        const float* inj_tok = inject_weights + (size_t)tok * hc_count;
        const __half* loc_row = local_in + (size_t)tok * hidden_size;
        const __half* peer_row = peer_s + (size_t)tok * hidden_size;
        __half* ar_row = (ar_out != nullptr) ? (ar_out + (size_t)tok * hidden_size) : nullptr;
        __nv_bfloat16* tap_row = (tap_bf16_out != nullptr)
            ? (tap_bf16_out + (size_t)tok * tap_stride_elems + tap_offset_elems)
            : nullptr;
        if (vec8) {
            const int vec_cols = hidden_size >> 3;
            const uint4* loc_v = reinterpret_cast<const uint4*>(loc_row);
            const uint4* peer_v = reinterpret_cast<const uint4*>(peer_row);
            uint4* ar_v = (ar_row != nullptr) ? reinterpret_cast<uint4*>(ar_row) : nullptr;
            const bool tap_vec8 = (tap_row != nullptr) && ((((uintptr_t)tap_row) & 15u) == 0);
            for (int c = threadIdx.x; c < vec_cols; c += blockDim.x) {
                const uint4 lv = loc_v[c];
                const uint4 pv = peer_v[c];
                const __half* lh = reinterpret_cast<const __half*>(&lv);
                const __half* ph = reinterpret_cast<const __half*>(&pv);
                uint4 arv;
                __half* arh = reinterpret_cast<__half*>(&arv);
                float y_arr[8];
                float tap_sum[8];
#pragma unroll
                for (int j = 0; j < 8; ++j) {
                    const float lf = __half2float(lh[j]);
                    const float pf = __half2float(ph[j]);
                    const __half ar_h = __float2half((rank == 0) ? (lf + pf) : (pf + lf));
                    arh[j] = ar_h;
                    y_arr[j] = __half2float(ar_h);
                    tap_sum[j] = 0.0f;
                }
                if (ar_v != nullptr) ar_v[c] = arv;
                for (int s = 0; s < hc_count; ++s) {
                    const float inj_s = inj_tok[s];
                    const uint4* rin_v = reinterpret_cast<const uint4*>(res_in_tok + (size_t)s * hidden_size);
                    uint4* rout_v = reinterpret_cast<uint4*>(res_out_tok + (size_t)s * hidden_size);
                    const uint4 rv = rin_v[c];
                    const __half* rh = reinterpret_cast<const __half*>(&rv);
                    uint4 pv_ple = make_uint4(0, 0, 0, 0);
                    if (ple_tok != nullptr) {
                        pv_ple = reinterpret_cast<const uint4*>(ple_tok + (size_t)s * hidden_size)[c];
                    }
                    const __half* plh = reinterpret_cast<const __half*>(&pv_ple);
                    uint4 outv;
                    __half* outh = reinterpret_cast<__half*>(&outv);
#pragma unroll
                    for (int j = 0; j < 8; ++j) {
                        float val = __half2float(rh[j]) + inj_s * y_arr[j];
                        if (ple_tok != nullptr) val += __half2float(plh[j]);
                        const __half out_h = __float2half(val);
                        outh[j] = out_h;
                        if (tap_row != nullptr) tap_sum[j] += __half2float(out_h);
                    }
                    rout_v[c] = outv;
                }
                if (tap_row != nullptr) {
                    if (tap_vec8) {
                        uint4 tv;
                        __nv_bfloat16* th = reinterpret_cast<__nv_bfloat16*>(&tv);
#pragma unroll
                        for (int j = 0; j < 8; ++j) {
                            th[j] = __float2bfloat16_rn(tap_sum[j] * inv_hc);
                        }
                        reinterpret_cast<uint4*>(tap_row)[c] = tv;
                    } else {
                        const int d_base = c << 3;
#pragma unroll
                        for (int j = 0; j < 8; ++j) {
                            tap_row[d_base + j] = __float2bfloat16_rn(tap_sum[j] * inv_hc);
                        }
                    }
                }
            }
        } else {
            for (int d = threadIdx.x; d < hidden_size; d += blockDim.x) {
                const float lf = __half2float(loc_row[d]);
                const float pf = __half2float(peer_row[d]);
                const __half ar_h = __float2half((rank == 0) ? (lf + pf) : (pf + lf));
                if (ar_row != nullptr) ar_row[d] = ar_h;
                const float y = __half2float(ar_h);
                float tap_sum = 0.0f;
                for (int s = 0; s < hc_count; ++s) {
                    const size_t idx = (size_t)s * hidden_size + d;
                    float val = __half2float(res_in_tok[idx]) + inj_tok[s] * y;
                    if (ple_tok != nullptr) val += __half2float(ple_tok[idx]);
                    const __half out_h = __float2half(val);
                    res_out_tok[idx] = out_h;
                    if (tap_row != nullptr) tap_sum += __half2float(out_h);
                }
                if (tap_row != nullptr) tap_row[d] = __float2bfloat16_rn(tap_sum * inv_hc);
            }
        }
    }
    if (dma_tx) {
        barrier_epoch_release(sig0, sig1, self_sg, rank, &epoch);
    }
}

__global__ void __launch_bounds__(512, 1) tp2_gather_vocab_f16_kernel(
    const __half* __restrict__ local_in,
    __half* __restrict__ rank0_full_out,
    __half* __restrict__ s1,
    Signal* __restrict__ sig0,
    Signal* __restrict__ sig1,
    int rank,
    int num_rows,
    int local_vocab_size) {
    const int num_elems = num_rows * local_vocab_size;
    Signal* self_sg = (rank == 0) ? sig0 : sig1;
    uint32_t epoch_pre = self_sg->flag[blockIdx.x] + 1;
    bool odd = (epoch_pre & 1u) != 0;
    __half* s1_buf = stage_epoch_ptr(s1, self_sg, odd, num_elems);
    const int full_vocab = local_vocab_size * 2;

    const bool vec8 = ((local_vocab_size & 7) == 0) &&
                      ((((uintptr_t)local_in | (uintptr_t)s1_buf | (uintptr_t)rank0_full_out) & 15u) == 0);
    if (vec8) {
        const int vec_vocab = local_vocab_size >> 3;
        const int vec_full = full_vocab >> 3;
        const int vec_elems = num_rows * vec_vocab;
        const uint4* in_v = reinterpret_cast<const uint4*>(local_in);
        uint4* s1_v = reinterpret_cast<uint4*>(s1_buf);
        uint4* out_v = reinterpret_cast<uint4*>(rank0_full_out);
        if (rank == 1) {
            for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < vec_elems; i += gridDim.x * blockDim.x) {
                s1_v[i] = in_v[i];
            }
        } else {
            for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < vec_elems; i += gridDim.x * blockDim.x) {
                const int row = i / vec_vocab;
                const int col = i - row * vec_vocab;
                out_v[(size_t)row * vec_full + col] = in_v[i];
            }
        }
    } else if (rank == 1) {
        for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < num_elems; i += gridDim.x * blockDim.x) {
            s1_buf[i] = local_in[i];
        }
    } else {
        for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < num_elems; i += gridDim.x * blockDim.x) {
            const int row = i / local_vocab_size;
            const int col = i - row * local_vocab_size;
            rank0_full_out[(size_t)row * full_vocab + col] = local_in[i];
        }
    }

    uint32_t epoch = 0;
    barrier_epoch_release(sig0, sig1, self_sg, rank, &epoch);

    if (rank == 0) {
        if (vec8) {
            const int vec_vocab = local_vocab_size >> 3;
            const int vec_full = full_vocab >> 3;
            const int vec_elems = num_rows * vec_vocab;
            const uint4* s1_v = reinterpret_cast<const uint4*>(s1_buf);
            uint4* out_v = reinterpret_cast<uint4*>(rank0_full_out);
            for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < vec_elems; i += gridDim.x * blockDim.x) {
                const int row = i / vec_vocab;
                const int col = i - row * vec_vocab;
                out_v[(size_t)row * vec_full + vec_vocab + col] = s1_v[i];
            }
        } else {
            for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < num_elems; i += gridDim.x * blockDim.x) {
                const int row = i / local_vocab_size;
                const int col = i - row * local_vocab_size;
                rank0_full_out[(size_t)row * full_vocab + local_vocab_size + col] = s1_buf[i];
            }
        }
    }
}

struct Tp4Ptrs {
    void* s[4];
    Signal* sig[4];
};

__device__ __forceinline__ void barrier_tp4_epoch_release(
    const Tp4Ptrs& ptrs,
    Signal* self_sg,
    int rank,
    uint32_t* out_epoch) {
    __syncthreads();
    uint32_t epoch = self_sg->flag[blockIdx.x] + 1;
    if (threadIdx.x == 0) {
#pragma unroll
        for (int peer = 0; peer < 4; ++peer) {
            if (peer != rank) {
                st_flag_release(&ptrs.sig[peer]->start[blockIdx.x][rank], epoch);
            }
        }
#pragma unroll
        for (int peer = 0; peer < 4; ++peer) {
            if (peer != rank) {
                uint32_t* self_counter = &self_sg->start[blockIdx.x][peer];
                while (static_cast<int32_t>(ld_flag_acquire(self_counter) - epoch) < 0) {
                }
            }
        }
        self_sg->flag[blockIdx.x] = epoch;
    }
    __syncthreads();
    *out_epoch = epoch;
}

__global__ void __launch_bounds__(512, 1) tp4_all_reduce_f16_kernel(
    const __half* local_in,
    __half* out,
    Tp4Ptrs ptrs,
    int rank,
    int num_elems) {
    Signal* self_sg = ptrs.sig[rank];
    uint32_t epoch_pre = self_sg->flag[blockIdx.x] + 1;
    bool odd = (epoch_pre & 1u) != 0;

    __half* s_buf[4];
#pragma unroll
    for (int r = 0; r < 4; ++r) {
        __half* base = reinterpret_cast<__half*>(ptrs.s[r]);
        s_buf[r] = stage_epoch_ptr(base, self_sg, odd, num_elems);
    }
    __half* __restrict__ my_s = s_buf[rank];

    const bool vec8 = ((num_elems & 7) == 0) &&
                      ((((uintptr_t)local_in | (uintptr_t)out | (uintptr_t)s_buf[0] |
                         (uintptr_t)s_buf[1] | (uintptr_t)s_buf[2] | (uintptr_t)s_buf[3]) & 15u) == 0);
    if (vec8) {
        const int vec_elems = num_elems >> 3;
        const uint4* local_v = reinterpret_cast<const uint4*>(local_in);
        uint4* my_v = reinterpret_cast<uint4*>(my_s);
        for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < vec_elems; i += gridDim.x * blockDim.x) {
            my_v[i] = local_v[i];
        }
    } else {
        for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < num_elems; i += gridDim.x * blockDim.x) {
            my_s[i] = local_in[i];
        }
    }

    uint32_t epoch = 0;
    barrier_tp4_epoch_release(ptrs, self_sg, rank, &epoch);

    if (vec8) {
        const int vec_elems = num_elems >> 3;
        const uint4* local_v = reinterpret_cast<const uint4*>(local_in);
        const uint4* s0_v = reinterpret_cast<const uint4*>(s_buf[0]);
        const uint4* s1_v = reinterpret_cast<const uint4*>(s_buf[1]);
        const uint4* s2_v = reinterpret_cast<const uint4*>(s_buf[2]);
        const uint4* s3_v = reinterpret_cast<const uint4*>(s_buf[3]);
        uint4* out_v = reinterpret_cast<uint4*>(out);
        for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < vec_elems; i += gridDim.x * blockDim.x) {
            const uint4 lv = local_v[i];
            const uint4 v0 = (rank == 0) ? lv : s0_v[i];
            const uint4 v1 = (rank == 1) ? lv : s1_v[i];
            const uint4 v2 = (rank == 2) ? lv : s2_v[i];
            const uint4 v3 = (rank == 3) ? lv : s3_v[i];
            const __half* h0 = reinterpret_cast<const __half*>(&v0);
            const __half* h1 = reinterpret_cast<const __half*>(&v1);
            const __half* h2 = reinterpret_cast<const __half*>(&v2);
            const __half* h3 = reinterpret_cast<const __half*>(&v3);
            uint4 ov;
            __half* oh = reinterpret_cast<__half*>(&ov);
#pragma unroll
            for (int d = 0; d < 8; ++d) {
                const float f0 = __half2float(h0[d]);
                const float f1 = __half2float(h1[d]);
                const float f2 = __half2float(h2[d]);
                const float f3 = __half2float(h3[d]);
                oh[d] = __float2half((f0 + f1) + (f2 + f3));
            }
            out_v[i] = ov;
        }
    } else {
        for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < num_elems; i += gridDim.x * blockDim.x) {
            const float lf = __half2float(local_in[i]);
            const float f0 = (rank == 0) ? lf : __half2float(s_buf[0][i]);
            const float f1 = (rank == 1) ? lf : __half2float(s_buf[1][i]);
            const float f2 = (rank == 2) ? lf : __half2float(s_buf[2][i]);
            const float f3 = (rank == 3) ? lf : __half2float(s_buf[3][i]);
            out[i] = __float2half((f0 + f1) + (f2 + f3));
        }
    }
}

template <int THREADS, int MAX_ITERS>
__global__ void __launch_bounds__(THREADS, 2) tp4_all_reduce_residual_gemma_rms_norm_f16_kernel(
    const __half* local_in,
    __half* ar_out,
    __half* __restrict__ residual,
    const __half* __restrict__ weight,
    __half* __restrict__ out,
    __half* __restrict__ out_had64,
    __nv_bfloat16* __restrict__ tap_dst_bf16,
    Tp4Ptrs ptrs,
    int rank,
    int num_tokens,
    int hidden_size,
    float eps,
    int tap_row_stride,
    int tap_col_offset) {
    const int num_elems = num_tokens * hidden_size;
    Signal* self_sg = ptrs.sig[rank];
    uint32_t epoch_pre = self_sg->flag[blockIdx.x] + 1;
    bool odd = (epoch_pre & 1u) != 0;

    __half* s_buf[4];
#pragma unroll
    for (int r = 0; r < 4; ++r) {
        __half* base = reinterpret_cast<__half*>(ptrs.s[r]);
        s_buf[r] = stage_epoch_ptr(base, self_sg, odd, num_elems);
    }
    __half* __restrict__ my_s = s_buf[rank];

    constexpr int VEC_ITERS = (MAX_ITERS + 7) / 8;
    const bool vec8 = ((hidden_size & 7) == 0) &&
                      ((((uintptr_t)local_in | (uintptr_t)s_buf[0] | (uintptr_t)s_buf[1] |
                         (uintptr_t)s_buf[2] | (uintptr_t)s_buf[3] |
                         (uintptr_t)residual | (uintptr_t)weight | (uintptr_t)out |
                         (uintptr_t)out_had64 | (uintptr_t)ar_out) & 15u) == 0);

    __shared__ float s_warp[32];

    if (vec8) {
        const int vec_cols = hidden_size >> 3;
        for (int row = (int)blockIdx.x; row < num_tokens; row += (int)gridDim.x) {
            const uint4* src_v = reinterpret_cast<const uint4*>(local_in + (size_t)row * hidden_size);
            uint4* dst_v = reinterpret_cast<uint4*>(my_s + (size_t)row * hidden_size);
#pragma unroll
            for (int k = 0; k < VEC_ITERS; ++k) {
                const int c = threadIdx.x + k * THREADS;
                if (c < vec_cols) {
                    dst_v[c] = src_v[c];
                }
            }
        }
    } else {
        for (int row = (int)blockIdx.x; row < num_tokens; row += (int)gridDim.x) {
            const __half* src_r = local_in + (size_t)row * hidden_size;
            __half* dst_r = my_s + (size_t)row * hidden_size;
            for (int c = threadIdx.x; c < hidden_size; c += THREADS) {
                dst_r[c] = src_r[c];
            }
        }
    }

    uint32_t epoch = 0;
    barrier_tp4_epoch_release(ptrs, self_sg, rank, &epoch);

    if (vec8) {
        const int vec_cols = hidden_size >> 3;
        const uint4* weight_v = reinterpret_cast<const uint4*>(weight);
        for (int row = (int)blockIdx.x; row < num_tokens; row += (int)gridDim.x) {
            const uint4* local_v = reinterpret_cast<const uint4*>(local_in + (size_t)row * hidden_size);
            const uint4* s0_v = reinterpret_cast<const uint4*>(s_buf[0] + (size_t)row * hidden_size);
            const uint4* s1_v = reinterpret_cast<const uint4*>(s_buf[1] + (size_t)row * hidden_size);
            const uint4* s2_v = reinterpret_cast<const uint4*>(s_buf[2] + (size_t)row * hidden_size);
            const uint4* s3_v = reinterpret_cast<const uint4*>(s_buf[3] + (size_t)row * hidden_size);
            uint4* ar_v = (ar_out != nullptr)
                ? reinterpret_cast<uint4*>(ar_out + (size_t)row * hidden_size)
                : nullptr;
            uint4* res_v = reinterpret_cast<uint4*>(residual + (size_t)row * hidden_size);
            uint4* out_v = (out != nullptr)
                ? reinterpret_cast<uint4*>(out + (size_t)row * hidden_size)
                : nullptr;
            uint4* out_had_v = (out_had64 != nullptr)
                ? reinterpret_cast<uint4*>(out_had64 + (size_t)row * hidden_size)
                : nullptr;
            __nv_bfloat16* tap_row = (tap_dst_bf16 != nullptr)
                ? (tap_dst_bf16 + (size_t)row * tap_row_stride + tap_col_offset)
                : nullptr;

            float v_reg[VEC_ITERS][8];
            float sum_sq = 0.0f;
#pragma unroll
            for (int k = 0; k < VEC_ITERS; ++k) {
                const int c = threadIdx.x + k * THREADS;
                if (c < vec_cols) {
                    const uint4 lv = local_v[c];
                    const uint4 v0 = (rank == 0) ? lv : s0_v[c];
                    const uint4 v1 = (rank == 1) ? lv : s1_v[c];
                    const uint4 v2 = (rank == 2) ? lv : s2_v[c];
                    const uint4 v3 = (rank == 3) ? lv : s3_v[c];
                    const uint4 rv = res_v[c];
                    const __half* h0 = reinterpret_cast<const __half*>(&v0);
                    const __half* h1 = reinterpret_cast<const __half*>(&v1);
                    const __half* h2 = reinterpret_cast<const __half*>(&v2);
                    const __half* h3 = reinterpret_cast<const __half*>(&v3);
                    const __half* rh = reinterpret_cast<const __half*>(&rv);
                    uint4 arv, sumv;
                    __half* arh = reinterpret_cast<__half*>(&arv);
                    __half* sh = reinterpret_cast<__half*>(&sumv);
                    const int base_i = c << 3;
#pragma unroll
                    for (int d = 0; d < 8; ++d) {
                        const float f0 = __half2float(h0[d]);
                        const float f1 = __half2float(h1[d]);
                        const float f2 = __half2float(h2[d]);
                        const float f3 = __half2float(h3[d]);
                        const float ar_f = (f0 + f1) + (f2 + f3);
                        const __half ar_h = __float2half(ar_f);
                        arh[d] = ar_h;
                        const __half res_old = rh[d];
                        const float ar_hf = __half2float(ar_h);
                        const float res_hf = __half2float(res_old);
                        if (tap_row != nullptr) {
                            tap_row[base_i + d] = __float2bfloat16(res_hf + ar_hf);
                        }
                        sh[d] = __hadd(ar_h, res_old);
                        const float v = ar_hf + res_hf;
                        v_reg[k][d] = v;
                        sum_sq += v * v;
                    }
                    if (ar_v != nullptr) {
                        ar_v[c] = arv;
                    }
                    res_v[c] = sumv;
                }
            }
            sum_sq = block_reduce_sum<THREADS>(sum_sq, s_warp);
            const float inv_rms = rsqrtf(sum_sq / (float)hidden_size + eps);

#pragma unroll
            for (int k = 0; k < VEC_ITERS; ++k) {
                const int c = threadIdx.x + k * THREADS;
                if (c < vec_cols) {
                    const uint4 wv = weight_v[c];
                    const __half* wh = reinterpret_cast<const __half*>(&wv);
                    uint4 ov;
                    __half* oh = reinterpret_cast<__half*>(&ov);
#pragma unroll
                    for (int d = 0; d < 8; ++d) {
                        const float v = v_reg[k][d] * inv_rms;
                        const float w = 1.0f + __half2float(wh[d]);
                        oh[d] = __float2half(v * w);
                    }
                    if (out_v != nullptr) {
                        out_v[c] = ov;
                    }
                    if (out_had_v != nullptr) {
                        out_had_v[c] = w34::hadamard64_vec8_f16(ov, (int)threadIdx.x & 7);
                    }
                }
            }
        }
    } else {
        for (int row = (int)blockIdx.x; row < num_tokens; row += (int)gridDim.x) {
            const __half* local_row = local_in + (size_t)row * hidden_size;
            const __half* s0_row = s_buf[0] + (size_t)row * hidden_size;
            const __half* s1_row = s_buf[1] + (size_t)row * hidden_size;
            const __half* s2_row = s_buf[2] + (size_t)row * hidden_size;
            const __half* s3_row = s_buf[3] + (size_t)row * hidden_size;
            __half* ar_row = (ar_out != nullptr) ? (ar_out + (size_t)row * hidden_size) : nullptr;
            __half* res_row = residual + (size_t)row * hidden_size;
            __half* out_row = out + (size_t)row * hidden_size;
            __nv_bfloat16* tap_row = (tap_dst_bf16 != nullptr)
                ? (tap_dst_bf16 + (size_t)row * tap_row_stride + tap_col_offset)
                : nullptr;

            float v_scalar[MAX_ITERS];
            float sum_sq = 0.0f;
#pragma unroll
            for (int k = 0; k < MAX_ITERS; k++) {
                const int i = threadIdx.x + k * THREADS;
                if (i < hidden_size) {
                    const float lf = __half2float(local_row[i]);
                    const float f0 = (rank == 0) ? lf : __half2float(s0_row[i]);
                    const float f1 = (rank == 1) ? lf : __half2float(s1_row[i]);
                    const float f2 = (rank == 2) ? lf : __half2float(s2_row[i]);
                    const float f3 = (rank == 3) ? lf : __half2float(s3_row[i]);
                    const float ar_f = (f0 + f1) + (f2 + f3);
                    const __half ar_h = __float2half(ar_f);
                    if (ar_row != nullptr) {
                        ar_row[i] = ar_h;
                    }
                    const __half res_old = res_row[i];
                    if (tap_row != nullptr) {
                        tap_row[i] = __float2bfloat16(__half2float(res_old) + __half2float(ar_h));
                    }
                    const __half sum_h = __hadd(ar_h, res_old);
                    res_row[i] = sum_h;
                    const float v = __half2float(ar_h) + __half2float(res_old);
                    v_scalar[k] = v;
                    sum_sq += v * v;
                }
            }
            sum_sq = block_reduce_sum<THREADS>(sum_sq, s_warp);
            const float inv_rms = rsqrtf(sum_sq / (float)hidden_size + eps);

#pragma unroll
            for (int k = 0; k < MAX_ITERS; k++) {
                const int i = threadIdx.x + k * THREADS;
                if (i < hidden_size) {
                    const float v = v_scalar[k] * inv_rms;
                    const float w = 1.0f + __half2float(weight[i]);
                    out_row[i] = __float2half(v * w);
                }
            }
        }
    }
}

__global__ void __launch_bounds__(512, 1) tp4_all_reduce_bf16_kernel(
    const __nv_bfloat16* local_in,
    __nv_bfloat16* out,
    Tp4Ptrs ptrs,
    int rank,
    int num_elems) {
    Signal* self_sg = ptrs.sig[rank];
    uint32_t epoch_pre = self_sg->flag[blockIdx.x] + 1;
    bool odd = (epoch_pre & 1u) != 0;

    __nv_bfloat16* s_buf[4];
#pragma unroll
    for (int r = 0; r < 4; ++r) {
        __nv_bfloat16* base = reinterpret_cast<__nv_bfloat16*>(ptrs.s[r]);
        s_buf[r] = stage_epoch_ptr(base, self_sg, odd, num_elems);
    }
    __nv_bfloat16* __restrict__ my_s = s_buf[rank];

    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < num_elems; i += gridDim.x * blockDim.x) {
        my_s[i] = local_in[i];
    }

    uint32_t epoch = 0;
    barrier_tp4_epoch_release(ptrs, self_sg, rank, &epoch);

    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < num_elems; i += gridDim.x * blockDim.x) {
        const float lf = __bfloat162float(local_in[i]);
        const float f0 = (rank == 0) ? lf : __bfloat162float(s_buf[0][i]);
        const float f1 = (rank == 1) ? lf : __bfloat162float(s_buf[1][i]);
        const float f2 = (rank == 2) ? lf : __bfloat162float(s_buf[2][i]);
        const float f3 = (rank == 3) ? lf : __bfloat162float(s_buf[3][i]);
        out[i] = __float2bfloat16((f0 + f1) + (f2 + f3));
    }
}

__global__ void __launch_bounds__(512, 1) tp4_all_reduce_f32_kernel(
    const float* local_in,
    float* out,
    Tp4Ptrs ptrs,
    int rank,
    int num_elems) {
    Signal* self_sg = ptrs.sig[rank];
    uint32_t epoch_pre = self_sg->flag[blockIdx.x] + 1;
    bool odd = (epoch_pre & 1u) != 0;

    float* s_buf[4];
#pragma unroll
    for (int r = 0; r < 4; ++r) {
        float* base = reinterpret_cast<float*>(ptrs.s[r]);
        s_buf[r] = stage_epoch_ptr(base, self_sg, odd, num_elems);
    }

    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < num_elems; i += gridDim.x * blockDim.x) {
        s_buf[rank][i] = local_in[i];
    }

    uint32_t epoch = 0;
    barrier_tp4_epoch_release(ptrs, self_sg, rank, &epoch);

    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < num_elems; i += gridDim.x * blockDim.x) {
        const float lf = local_in[i];
        const float f0 = (rank == 0) ? lf : s_buf[0][i];
        const float f1 = (rank == 1) ? lf : s_buf[1][i];
        const float f2 = (rank == 2) ? lf : s_buf[2][i];
        const float f3 = (rank == 3) ? lf : s_buf[3][i];
        out[i] = (f0 + f1) + (f2 + f3);
    }
}

__global__ void __launch_bounds__(256, 1) tp4_broadcast_i32_kernel(
    const int32_t* __restrict__ src_rank0,
    int32_t* __restrict__ out,
    Tp4Ptrs ptrs,
    int rank,
    int num_elems) {
    Signal* self_sg = ptrs.sig[rank];
    uint32_t epoch_pre = self_sg->flag[blockIdx.x] + 1;
    bool odd = (epoch_pre & 1u) != 0;
    int32_t* s0_base = reinterpret_cast<int32_t*>(ptrs.s[0]);
    int32_t* s0_buf = small_epoch_ptr(s0_base, ptrs.sig[0], odd, num_elems);

    if (rank == 0) {
        for (int i = threadIdx.x; i < num_elems; i += blockDim.x) {
            s0_buf[i] = src_rank0[i];
        }
    }

    uint32_t epoch = 0;
    barrier_tp4_epoch_release(ptrs, self_sg, rank, &epoch);

    for (int i = threadIdx.x; i < num_elems; i += blockDim.x) {
        out[i] = s0_buf[i];
    }
}

__global__ void __launch_bounds__(256, 1) tp4_all_gather_top1_f32_i32_kernel(
    const float* __restrict__ local_max_val,
    const int32_t* __restrict__ local_max_id,
    float* __restrict__ global_max_val,
    int32_t* __restrict__ global_max_id,
    Tp4Ptrs ptrs,
    int rank,
    int num_rows) {
    Signal* self_sg = ptrs.sig[rank];
    uint32_t epoch_pre = self_sg->flag[blockIdx.x] + 1;
    bool odd = (epoch_pre & 1u) != 0;

    Top1Pair* s_buf[4];
#pragma unroll
    for (int r = 0; r < 4; ++r) {
        Top1Pair* base = reinterpret_cast<Top1Pair*>(ptrs.s[r]);
        s_buf[r] = small_epoch_ptr(base, ptrs.sig[r], odd, num_rows);
    }

    for (int i = threadIdx.x; i < num_rows; i += blockDim.x) {
        s_buf[rank][i] = Top1Pair{local_max_val[i], local_max_id[i]};
    }

    uint32_t epoch = 0;
    barrier_tp4_epoch_release(ptrs, self_sg, rank, &epoch);

    for (int i = threadIdx.x; i < num_rows; i += blockDim.x) {
        Top1Pair best = s_buf[0][i];
#pragma unroll
        for (int r = 1; r < 4; ++r) {
            Top1Pair cand = s_buf[r][i];
            if ((cand.val > best.val) || (cand.val == best.val && cand.id < best.id)) {
                best = cand;
            }
        }
        global_max_val[i] = best.val;
        global_max_id[i] = best.id;
    }
}

__global__ void __launch_bounds__(256, 1) tp4_all_gather_topk_f32_i32_kernel(
    const float* local_vals, const int32_t* local_ids,
    float* global_vals, int32_t* global_ids,
    Tp4Ptrs ptrs, int rank, int num_rows, int k) {
    const int num_elems = num_rows * k;
    Signal* self_sg = ptrs.sig[rank];
    bool odd = ((self_sg->flag[blockIdx.x] + 1) & 1u) != 0;
    Top1Pair* s_buf[4];
#pragma unroll
    for (int r = 0; r < 4; ++r) {
        Top1Pair* base = reinterpret_cast<Top1Pair*>(ptrs.s[r]);
        s_buf[r] = small_epoch_ptr(base, ptrs.sig[r], odd, num_elems);
    }
    for (int i = threadIdx.x; i < num_elems; i += blockDim.x) {
        s_buf[rank][i] = Top1Pair{local_vals[i], local_ids[i]};
    }
    uint32_t epoch = 0;
    barrier_tp4_epoch_release(ptrs, self_sg, rank, &epoch);
    for (int row = threadIdx.x; row < num_rows; row += blockDim.x) {
        float* out_v = global_vals + (size_t)row * k;
        int32_t* out_i = global_ids + (size_t)row * k;
        int cur[4] = {0, 0, 0, 0};
        for (int j = 0; j < k; ++j) {
            int best_r = -1;
            Top1Pair best{-INFINITY, -1};
            bool bad = false;
#pragma unroll
            for (int r = 0; r < 4; ++r) {
                if (cur[r] < k) {
                    const Top1Pair cand = s_buf[r][(size_t)row * k + cur[r]];
                    if (!isfinite(cand.val) || cand.id < 0) { bad = true; break; }
                    if (best_r < 0 || cand.val > best.val || (cand.val == best.val && cand.id < best.id)) {
                        best = cand;
                        best_r = r;
                    }
                }
            }
            if (bad || best_r < 0) {
                out_v[j] = NAN;
                out_i[j] = -1;
            } else {
                out_v[j] = best.val;
                out_i[j] = best.id;
                ++cur[best_r];
            }
        }
    }
}

__global__ void __launch_bounds__(256, 1) tp4_all_reduce_hc_combine_tap_f16_kernel(
    const __half* residual_in, const __half* local_in, __half* ar_out,
    const float* inject_weights, const __half* ple_delta, __half* residual_out,
    __nv_bfloat16* tap_bf16_out, const uint8_t* valid_token_mask,
    Tp4Ptrs ptrs, int rank, int num_tokens, int hc_count, int hidden_size,
    int tap_stride_elems, int tap_offset_elems) {
    const int num_elems = num_tokens * hidden_size;
    Signal* self_sg = ptrs.sig[rank];
    bool odd = ((self_sg->flag[blockIdx.x] + 1) & 1u) != 0;
    __half* s_buf[4];
#pragma unroll
    for (int r = 0; r < 4; ++r) {
        __half* base = reinterpret_cast<__half*>(ptrs.s[r]);
        s_buf[r] = stage_epoch_ptr(base, self_sg, odd, num_elems);
    }
    for (int tok = blockIdx.x; tok < num_tokens; tok += gridDim.x) {
        const size_t roff = (size_t)tok * hidden_size;
        for (int d = threadIdx.x; d < hidden_size; d += blockDim.x) {
            s_buf[rank][roff + d] = local_in[roff + d];
        }
    }
    uint32_t epoch = 0;
    barrier_tp4_epoch_release(ptrs, self_sg, rank, &epoch);
    const float inv_hc = 1.0f / static_cast<float>(hc_count);
    for (int tok = blockIdx.x; tok < num_tokens; tok += gridDim.x) {
        if (valid_token_mask != nullptr && valid_token_mask[tok] == 0) continue;
        const size_t base = (size_t)tok * hc_count * hidden_size;
        const __half* res_in_tok = residual_in + base;
        __half* res_out_tok = residual_out + base;
        const __half* ple_tok = (ple_delta != nullptr) ? (ple_delta + base) : nullptr;
        const float* inj_tok = inject_weights + (size_t)tok * hc_count;
        const __half* loc_row = local_in + (size_t)tok * hidden_size;
        __half* ar_row = (ar_out != nullptr) ? (ar_out + (size_t)tok * hidden_size) : nullptr;
        __nv_bfloat16* tap_row = (tap_bf16_out != nullptr)
            ? (tap_bf16_out + (size_t)tok * tap_stride_elems + tap_offset_elems)
            : nullptr;
        for (int d = threadIdx.x; d < hidden_size; d += blockDim.x) {
            const size_t roff = (size_t)tok * hidden_size + d;
            const float lf = __half2float(loc_row[d]);
            const float f0 = (rank == 0) ? lf : __half2float(s_buf[0][roff]);
            const float f1 = (rank == 1) ? lf : __half2float(s_buf[1][roff]);
            const float f2 = (rank == 2) ? lf : __half2float(s_buf[2][roff]);
            const float f3 = (rank == 3) ? lf : __half2float(s_buf[3][roff]);
            const __half ar_h = __float2half((f0 + f1) + (f2 + f3));
            if (ar_row != nullptr) ar_row[d] = ar_h;
            const float y = __half2float(ar_h);
            float tap_sum = 0.0f;
            for (int s = 0; s < hc_count; ++s) {
                const size_t idx = (size_t)s * hidden_size + d;
                float val = __half2float(res_in_tok[idx]) + inj_tok[s] * y;
                if (ple_tok != nullptr) val += __half2float(ple_tok[idx]);
                const __half out_h = __float2half(val);
                res_out_tok[idx] = out_h;
                if (tap_row != nullptr) tap_sum += __half2float(out_h);
            }
            if (tap_row != nullptr) tap_row[d] = __float2bfloat16_rn(tap_sum * inv_hc);
        }
    }
}

__global__ void __launch_bounds__(512, 1) tp4_gather_vocab_f16_kernel(
    const __half* __restrict__ local_in,
    __half* __restrict__ rank0_full_out,
    Tp4Ptrs ptrs,
    int rank,
    int num_rows,
    int local_vocab_size) {
    const int num_elems = num_rows * local_vocab_size;
    Signal* self_sg = ptrs.sig[rank];
    uint32_t epoch_pre = self_sg->flag[blockIdx.x] + 1;
    bool odd = (epoch_pre & 1u) != 0;

    __half* s_buf[4];
#pragma unroll
    for (int r = 0; r < 4; ++r) {
        __half* base = reinterpret_cast<__half*>(ptrs.s[r]);
        s_buf[r] = stage_epoch_ptr(base, self_sg, odd, num_elems);
    }
    const int full_vocab = local_vocab_size * 4;

    const bool vec8 = ((local_vocab_size & 7) == 0) &&
                      ((((uintptr_t)local_in | (uintptr_t)s_buf[1] | (uintptr_t)s_buf[2] |
                         (uintptr_t)s_buf[3] | (uintptr_t)rank0_full_out) & 15u) == 0);
    if (rank != 0) {
        if (vec8) {
            const int vec_elems = num_elems >> 3;
            const uint4* in_v = reinterpret_cast<const uint4*>(local_in);
            uint4* my_v = reinterpret_cast<uint4*>(s_buf[rank]);
            for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < vec_elems; i += gridDim.x * blockDim.x) {
                my_v[i] = in_v[i];
            }
        } else {
            for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < num_elems; i += gridDim.x * blockDim.x) {
                s_buf[rank][i] = local_in[i];
            }
        }
    } else if (vec8) {
        const int vec_vocab = local_vocab_size >> 3;
        const int vec_full = full_vocab >> 3;
        const int vec_elems = num_rows * vec_vocab;
        const uint4* in_v = reinterpret_cast<const uint4*>(local_in);
        uint4* out_v = reinterpret_cast<uint4*>(rank0_full_out);
        for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < vec_elems; i += gridDim.x * blockDim.x) {
            const int row = i / vec_vocab;
            const int col = i - row * vec_vocab;
            out_v[(size_t)row * vec_full + col] = in_v[i];
        }
    } else {
        for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < num_elems; i += gridDim.x * blockDim.x) {
            const int row = i / local_vocab_size;
            const int col = i - row * local_vocab_size;
            rank0_full_out[(size_t)row * full_vocab + col] = local_in[i];
        }
    }

    uint32_t epoch = 0;
    barrier_tp4_epoch_release(ptrs, self_sg, rank, &epoch);

    if (rank == 0) {
        if (vec8) {
            const int vec_vocab = local_vocab_size >> 3;
            const int vec_full = full_vocab >> 3;
            const int vec_elems = num_rows * vec_vocab;
            const uint4* s1_v = reinterpret_cast<const uint4*>(s_buf[1]);
            const uint4* s2_v = reinterpret_cast<const uint4*>(s_buf[2]);
            const uint4* s3_v = reinterpret_cast<const uint4*>(s_buf[3]);
            uint4* out_v = reinterpret_cast<uint4*>(rank0_full_out);
            for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < vec_elems; i += gridDim.x * blockDim.x) {
                const int row = i / vec_vocab;
                const int col = i - row * vec_vocab;
                out_v[(size_t)row * vec_full + vec_vocab + col] = s1_v[i];
                out_v[(size_t)row * vec_full + 2 * vec_vocab + col] = s2_v[i];
                out_v[(size_t)row * vec_full + 3 * vec_vocab + col] = s3_v[i];
            }
        } else {
            for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < num_elems; i += gridDim.x * blockDim.x) {
                const int row = i / local_vocab_size;
                const int col = i - row * local_vocab_size;
                rank0_full_out[(size_t)row * full_vocab + local_vocab_size + col] = s_buf[1][i];
                rank0_full_out[(size_t)row * full_vocab + 2 * local_vocab_size + col] = s_buf[2][i];
                rank0_full_out[(size_t)row * full_vocab + 3 * local_vocab_size + col] = s_buf[3][i];
            }
        }
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
    const size_t num_bytes = (size_t)num_elems * sizeof(__half);
    if (num_bytes >= w34_comm::kDmaThresholdBytes) {
        cudaError_t cpy_err = cudaMemcpyAsync(
            (rank == 0) ? stage0 : stage1, local_in, num_bytes,
            cudaMemcpyDefault, (cudaStream_t)ex->stream);
        if (cpy_err != cudaSuccess) return (int)cpy_err;
    }
    const bool vec8 = ((num_elems & 7) == 0) &&
                      ((((uintptr_t)local_in | (uintptr_t)out | (uintptr_t)stage0 | (uintptr_t)stage1) & 15u) == 0);
    const int elems_per_block = vec8 ? 4096 : 512;
    int blocks = (num_elems + elems_per_block - 1) / elems_per_block;
    int cap = w34_comm::tp2_block_cap(true);
    if (cap > 16) cap = 16;
    if (blocks > cap) blocks = cap;
    if (blocks < 1) blocks = 1;
    w34_comm::tp2_all_reduce_f16_kernel<<<blocks, 512, 0, (cudaStream_t)ex->stream>>>(
        (const __half*)local_in, (__half*)out,
        (__half*)stage0, (__half*)stage1,
        (w34_comm::Signal*)sig0, (w34_comm::Signal*)sig1,
        rank, num_elems);
    return (int)cudaGetLastError();
}

extern "C" int w34_tp2_all_reduce_residual_gemma_rms_norm_hadamard_f16(
    const w34_exec* ex, const void* local_in, void* ar_out, void* residual,
    const void* weight, void* out, void* out_had64, void* tap_dst_bf16,
    void* stage0, void* stage1, void* sig0, void* sig1, int rank, int num_tokens,
    int hidden_size, float eps, int tap_row_stride, int tap_col_offset) {
    if (ex == nullptr || local_in == nullptr || residual == nullptr ||
        weight == nullptr || (out == nullptr && out_had64 == nullptr) ||
        stage0 == nullptr || stage1 == nullptr || sig0 == nullptr || sig1 == nullptr ||
        (rank != 0 && rank != 1) || num_tokens < 0 || hidden_size <= 0 ||
        hidden_size > 6144) {
        return (int)cudaErrorInvalidValue;
    }
    if (out_had64 != nullptr &&
        ((hidden_size & 63) != 0 ||
         ((((uintptr_t)local_in | (uintptr_t)stage0 | (uintptr_t)stage1 |
            (uintptr_t)residual | (uintptr_t)weight | (uintptr_t)out |
            (uintptr_t)out_had64 | (uintptr_t)ar_out) & 15u) != 0))) {
        return (int)cudaErrorInvalidValue;
    }
    if (tap_dst_bf16 != nullptr &&
        (tap_col_offset < 0 || tap_row_stride < hidden_size ||
         tap_col_offset > tap_row_stride - hidden_size)) {
        return (int)cudaErrorInvalidValue;
    }
    if (num_tokens == 0) return 0;
    const size_t num_bytes = (size_t)num_tokens * (size_t)hidden_size * sizeof(__half);
    if (num_bytes >= w34_comm::kDmaThresholdBytes) {
        cudaError_t cpy_err = cudaMemcpyAsync(
            (rank == 0) ? stage0 : stage1, local_in, num_bytes,
            cudaMemcpyDefault, (cudaStream_t)ex->stream);
        if (cpy_err != cudaSuccess) return (int)cpy_err;
    }
    int blocks = num_tokens;
    // Cap blocks at 16 (`16 * 512 = 8192` threads across 16 CTAs easily saturates PCIe Gen4
    // x4/x8/x16 while avoiding 36..128-CTA host cache-line polling storms across the root complex).
    int cap = w34_comm::tp2_block_cap(true);
    if (cap > 16) cap = 16;
    if (blocks > cap) blocks = cap;
    if (hidden_size <= 2560) {
        w34_comm::tp2_all_reduce_residual_gemma_rms_norm_f16_kernel<256, 10>
            <<<blocks, 256, 0, (cudaStream_t)ex->stream>>>(
                (const __half*)local_in, (__half*)ar_out, (__half*)residual,
                (const __half*)weight, (__half*)out, (__half*)out_had64,
                (__nv_bfloat16*)tap_dst_bf16, (__half*)stage0, (__half*)stage1,
                (w34_comm::Signal*)sig0, (w34_comm::Signal*)sig1,
                rank, num_tokens, hidden_size, eps, tap_row_stride, tap_col_offset);
    } else if (hidden_size <= 4096) {
        w34_comm::tp2_all_reduce_residual_gemma_rms_norm_f16_kernel<256, 16>
            <<<blocks, 256, 0, (cudaStream_t)ex->stream>>>(
                (const __half*)local_in, (__half*)ar_out, (__half*)residual,
                (const __half*)weight, (__half*)out, (__half*)out_had64,
                (__nv_bfloat16*)tap_dst_bf16, (__half*)stage0, (__half*)stage1,
                (w34_comm::Signal*)sig0, (w34_comm::Signal*)sig1,
                rank, num_tokens, hidden_size, eps, tap_row_stride, tap_col_offset);
    } else if (hidden_size <= 5120) {
        w34_comm::tp2_all_reduce_residual_gemma_rms_norm_f16_kernel<512, 10>
            <<<blocks, 512, 0, (cudaStream_t)ex->stream>>>(
                (const __half*)local_in, (__half*)ar_out, (__half*)residual,
                (const __half*)weight, (__half*)out, (__half*)out_had64,
                (__nv_bfloat16*)tap_dst_bf16, (__half*)stage0, (__half*)stage1,
                (w34_comm::Signal*)sig0, (w34_comm::Signal*)sig1,
                rank, num_tokens, hidden_size, eps, tap_row_stride, tap_col_offset);
    } else {
        w34_comm::tp2_all_reduce_residual_gemma_rms_norm_f16_kernel<512, 12>
            <<<blocks, 512, 0, (cudaStream_t)ex->stream>>>(
                (const __half*)local_in, (__half*)ar_out, (__half*)residual,
                (const __half*)weight, (__half*)out, (__half*)out_had64,
                (__nv_bfloat16*)tap_dst_bf16, (__half*)stage0, (__half*)stage1,
                (w34_comm::Signal*)sig0, (w34_comm::Signal*)sig1,
                rank, num_tokens, hidden_size, eps, tap_row_stride, tap_col_offset);
    }
    return (int)cudaGetLastError();
}

extern "C" int w34_tp2_all_reduce_residual_gemma_rms_norm_f16(
    const w34_exec* ex, const void* local_in, void* ar_out, void* residual,
    const void* weight, void* out, void* tap_dst_bf16,
    void* stage0, void* stage1, void* sig0, void* sig1, int rank, int num_tokens,
    int hidden_size, float eps, int tap_row_stride, int tap_col_offset) {
    if (out == nullptr) return (int)cudaErrorInvalidValue;
    return w34_tp2_all_reduce_residual_gemma_rms_norm_hadamard_f16(
        ex, local_in, ar_out, residual, weight, out, nullptr, tap_dst_bf16,
        stage0, stage1, sig0, sig1, rank, num_tokens, hidden_size, eps,
        tap_row_stride, tap_col_offset);
}

extern "C" int w34_tp2_all_reduce_bf16(const w34_exec* ex, const void* local_in, void* out,
                                       void* stage0, void* stage1,
                                       void* sig0, void* sig1, int rank, int num_elems) {
    if (ex == nullptr || local_in == nullptr || out == nullptr ||
        stage0 == nullptr || stage1 == nullptr || sig0 == nullptr || sig1 == nullptr ||
        (rank != 0 && rank != 1) || num_elems <= 0) {
        return (int)cudaErrorInvalidValue;
    }
    const size_t num_bytes = (size_t)num_elems * sizeof(__nv_bfloat16);
    if (num_bytes >= w34_comm::kDmaThresholdBytes) {
        cudaError_t cpy_err = cudaMemcpyAsync(
            (rank == 0) ? stage0 : stage1, local_in, num_bytes,
            cudaMemcpyDefault, (cudaStream_t)ex->stream);
        if (cpy_err != cudaSuccess) return (int)cpy_err;
    }
    const bool vec8 = ((num_elems & 7) == 0) &&
                      ((((uintptr_t)local_in | (uintptr_t)out | (uintptr_t)stage0 | (uintptr_t)stage1) & 15u) == 0);
    const int elems_per_block = vec8 ? 4096 : 512;
    int blocks = (num_elems + elems_per_block - 1) / elems_per_block;
    int cap = w34_comm::tp2_block_cap(true);
    if (cap > 16) cap = 16;
    if (blocks > cap) blocks = cap;
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
    const size_t num_bytes = (size_t)num_elems * sizeof(float);
    if (num_bytes >= w34_comm::kDmaThresholdBytes) {
        cudaError_t cpy_err = cudaMemcpyAsync(
            (rank == 0) ? stage0 : stage1, local_in, num_bytes,
            cudaMemcpyDefault, (cudaStream_t)ex->stream);
        if (cpy_err != cudaSuccess) return (int)cpy_err;
    }
    const bool vec4 = ((num_elems & 3) == 0) &&
                      ((((uintptr_t)local_in | (uintptr_t)out | (uintptr_t)stage0 | (uintptr_t)stage1) & 15u) == 0);
    const int elems_per_block = vec4 ? 2048 : 512;
    int blocks = (num_elems + elems_per_block - 1) / elems_per_block;
    int cap = w34_comm::tp2_block_cap(true);
    if (cap > 16) cap = 16;
    if (blocks > cap) blocks = cap;
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
    const w34_exec* ex, const float* local_max_val, const int32_t* local_max_id,
    float* global_max_val, int32_t* global_max_id, void* stage0, void* stage1,
    void* sig0, void* sig1, int rank, int num_rows) {
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

extern "C" int w34_tp2_gather_vocab_f16(
    const w34_exec* ex, const void* local_logits_f16, void* rank0_full_logits_f16,
    void* stage0, void* stage1, void* sig0, void* sig1, int rank, int num_rows, int local_vocab_size) {
    (void)stage0;
    if (ex == nullptr || local_logits_f16 == nullptr ||
        (rank == 0 && rank0_full_logits_f16 == nullptr) ||
        stage1 == nullptr || sig0 == nullptr || sig1 == nullptr ||
        (rank != 0 && rank != 1) || num_rows <= 0 || local_vocab_size <= 0) {
        return (int)cudaErrorInvalidValue;
    }
    const int num_elems = num_rows * local_vocab_size;
    int blocks = (num_elems + 511) / 512;
    const int cap = w34_comm::tp2_block_cap(num_elems <= 36 * 5120);
    if (blocks > cap) blocks = cap;
    if (blocks < 1) blocks = 1;
    w34_comm::tp2_gather_vocab_f16_kernel<<<blocks, 512, 0, (cudaStream_t)ex->stream>>>(
        (const __half*)local_logits_f16, (__half*)rank0_full_logits_f16, (__half*)stage1,
        (w34_comm::Signal*)sig0, (w34_comm::Signal*)sig1, rank, num_rows, local_vocab_size);
    return (int)cudaGetLastError();
}

extern "C" int w34_device_can_enable_p2p(int dev_a, int dev_b, int require_native_atomic, int* out_can_p2p) {
    if (out_can_p2p == nullptr) return (int)cudaErrorInvalidValue;
    *out_can_p2p = 0;
    if (dev_a < 0 || dev_b < 0) return (int)cudaErrorInvalidDevice;
    if (dev_a == dev_b) {
        *out_can_p2p = 1;
        return 0;
    }
    int can_ab = 0;
    int can_ba = 0;
    cudaError_t err = cudaDeviceCanAccessPeer(&can_ab, dev_a, dev_b);
    if (err != cudaSuccess) return (int)err;
    err = cudaDeviceCanAccessPeer(&can_ba, dev_b, dev_a);
    if (err != cudaSuccess) return (int)err;
    if (!can_ab || !can_ba) {
        return 0;
    }
    if (require_native_atomic) {
        int atomic_ab = 0;
        int atomic_ba = 0;
        err = cudaDeviceGetP2PAttribute(&atomic_ab, cudaDevP2PAttrNativeAtomicSupported, dev_a, dev_b);
        if (err != cudaSuccess) return (int)err;
        err = cudaDeviceGetP2PAttribute(&atomic_ba, cudaDevP2PAttrNativeAtomicSupported, dev_b, dev_a);
        if (err != cudaSuccess) return (int)err;
        if (!atomic_ab || !atomic_ba) {
            return 0;
        }
    }
    int prev_dev = 0;
    (void)cudaGetDevice(&prev_dev);
    err = cudaSetDevice(dev_a);
    if (err != cudaSuccess) return (int)err;
    err = cudaDeviceEnablePeerAccess(dev_b, 0);
    if (err == cudaErrorPeerAccessAlreadyEnabled) {
        (void)cudaGetLastError();
    } else if (err != cudaSuccess) {
        (void)cudaSetDevice(prev_dev);
        return (int)err;
    }
    err = cudaSetDevice(dev_b);
    if (err != cudaSuccess) {
        (void)cudaSetDevice(prev_dev);
        return (int)err;
    }
    err = cudaDeviceEnablePeerAccess(dev_a, 0);
    if (err == cudaErrorPeerAccessAlreadyEnabled) {
        (void)cudaGetLastError();
    } else if (err != cudaSuccess) {
        (void)cudaSetDevice(prev_dev);
        return (int)err;
    }
    (void)cudaSetDevice(prev_dev);
    *out_can_p2p = 1;
    return 0;
}

extern "C" int w34_tp4_all_reduce_f16(
    const w34_exec* ex, const void* local_in, void* out,
    void* stage0, void* stage1, void* stage2, void* stage3,
    void* sig0, void* sig1, void* sig2, void* sig3, int rank, int num_elems) {
    if (ex == nullptr || local_in == nullptr || out == nullptr ||
        stage0 == nullptr || stage1 == nullptr || stage2 == nullptr || stage3 == nullptr ||
        sig0 == nullptr || sig1 == nullptr || sig2 == nullptr || sig3 == nullptr ||
        rank < 0 || rank >= 4 || num_elems <= 0) {
        return (int)cudaErrorInvalidValue;
    }
    const bool vec8 = ((num_elems & 7) == 0) &&
                      ((((uintptr_t)local_in | (uintptr_t)out | (uintptr_t)stage0 | (uintptr_t)stage1 | (uintptr_t)stage2 | (uintptr_t)stage3) & 15u) == 0);
    const int elems_per_block = vec8 ? 4096 : 512;
    int blocks = (num_elems + elems_per_block - 1) / elems_per_block;
    int cap = w34_comm::tp2_block_cap(num_elems <= 36 * 5120);
    if (num_elems <= 36 * 5120 && cap > 16) cap = 16;
    if (blocks > cap) blocks = cap;
    if (blocks < 1) blocks = 1;
    w34_comm::Tp4Ptrs ptrs{
        {stage0, stage1, stage2, stage3},
        {(w34_comm::Signal*)sig0, (w34_comm::Signal*)sig1, (w34_comm::Signal*)sig2, (w34_comm::Signal*)sig3},
    };
    w34_comm::tp4_all_reduce_f16_kernel<<<blocks, 512, 0, (cudaStream_t)ex->stream>>>(
        (const __half*)local_in, (__half*)out, ptrs, rank, num_elems);
    return (int)cudaGetLastError();
}

extern "C" int w34_tp4_all_reduce_residual_gemma_rms_norm_hadamard_f16(
    const w34_exec* ex, const void* local_in, void* ar_out, void* residual,
    const void* weight, void* out, void* out_had64, void* tap_dst_bf16,
    void* stage0, void* stage1, void* stage2, void* stage3,
    void* sig0, void* sig1, void* sig2, void* sig3, int rank, int num_tokens,
    int hidden_size, float eps, int tap_row_stride, int tap_col_offset) {
    if (ex == nullptr || local_in == nullptr || residual == nullptr ||
        weight == nullptr || (out == nullptr && out_had64 == nullptr) ||
        stage0 == nullptr || stage1 == nullptr || stage2 == nullptr || stage3 == nullptr ||
        sig0 == nullptr || sig1 == nullptr || sig2 == nullptr || sig3 == nullptr ||
        rank < 0 || rank >= 4 || num_tokens < 0 || hidden_size <= 0 || hidden_size > 6144) {
        return (int)cudaErrorInvalidValue;
    }
    if (out_had64 != nullptr &&
        ((hidden_size & 63) != 0 ||
         ((((uintptr_t)local_in | (uintptr_t)stage0 | (uintptr_t)stage1 |
            (uintptr_t)stage2 | (uintptr_t)stage3 | (uintptr_t)residual |
            (uintptr_t)weight | (uintptr_t)out | (uintptr_t)out_had64 |
            (uintptr_t)ar_out) & 15u) != 0))) {
        return (int)cudaErrorInvalidValue;
    }
    if (tap_dst_bf16 != nullptr &&
        (tap_col_offset < 0 || tap_row_stride < hidden_size ||
         tap_col_offset > tap_row_stride - hidden_size)) {
        return (int)cudaErrorInvalidValue;
    }
    if (num_tokens == 0) return 0;
    int blocks = num_tokens;
    int cap = w34_comm::tp2_block_cap(true);
    if (cap > 16) cap = 16;
    if (blocks > cap) blocks = cap;
    w34_comm::Tp4Ptrs ptrs{
        {stage0, stage1, stage2, stage3},
        {(w34_comm::Signal*)sig0, (w34_comm::Signal*)sig1, (w34_comm::Signal*)sig2, (w34_comm::Signal*)sig3},
    };
    if (hidden_size <= 2560) {
        w34_comm::tp4_all_reduce_residual_gemma_rms_norm_f16_kernel<256, 10>
            <<<blocks, 256, 0, (cudaStream_t)ex->stream>>>(
                (const __half*)local_in, (__half*)ar_out, (__half*)residual,
                (const __half*)weight, (__half*)out, (__half*)out_had64,
                (__nv_bfloat16*)tap_dst_bf16, ptrs,
                rank, num_tokens, hidden_size, eps, tap_row_stride, tap_col_offset);
    } else if (hidden_size <= 4096) {
        w34_comm::tp4_all_reduce_residual_gemma_rms_norm_f16_kernel<256, 16>
            <<<blocks, 256, 0, (cudaStream_t)ex->stream>>>(
                (const __half*)local_in, (__half*)ar_out, (__half*)residual,
                (const __half*)weight, (__half*)out, (__half*)out_had64,
                (__nv_bfloat16*)tap_dst_bf16, ptrs,
                rank, num_tokens, hidden_size, eps, tap_row_stride, tap_col_offset);
    } else if (hidden_size <= 5120) {
        w34_comm::tp4_all_reduce_residual_gemma_rms_norm_f16_kernel<512, 10>
            <<<blocks, 512, 0, (cudaStream_t)ex->stream>>>(
                (const __half*)local_in, (__half*)ar_out, (__half*)residual,
                (const __half*)weight, (__half*)out, (__half*)out_had64,
                (__nv_bfloat16*)tap_dst_bf16, ptrs,
                rank, num_tokens, hidden_size, eps, tap_row_stride, tap_col_offset);
    } else {
        w34_comm::tp4_all_reduce_residual_gemma_rms_norm_f16_kernel<512, 12>
            <<<blocks, 512, 0, (cudaStream_t)ex->stream>>>(
                (const __half*)local_in, (__half*)ar_out, (__half*)residual,
                (const __half*)weight, (__half*)out, (__half*)out_had64,
                (__nv_bfloat16*)tap_dst_bf16, ptrs,
                rank, num_tokens, hidden_size, eps, tap_row_stride, tap_col_offset);
    }
    return (int)cudaGetLastError();
}

extern "C" int w34_tp4_all_reduce_residual_gemma_rms_norm_f16(
    const w34_exec* ex, const void* local_in, void* ar_out, void* residual,
    const void* weight, void* out, void* tap_dst_bf16,
    void* stage0, void* stage1, void* stage2, void* stage3,
    void* sig0, void* sig1, void* sig2, void* sig3, int rank, int num_tokens,
    int hidden_size, float eps, int tap_row_stride, int tap_col_offset) {
    if (out == nullptr) return (int)cudaErrorInvalidValue;
    return w34_tp4_all_reduce_residual_gemma_rms_norm_hadamard_f16(
        ex, local_in, ar_out, residual, weight, out, nullptr, tap_dst_bf16,
        stage0, stage1, stage2, stage3, sig0, sig1, sig2, sig3,
        rank, num_tokens, hidden_size, eps, tap_row_stride, tap_col_offset);
}

extern "C" int w34_tp4_all_reduce_bf16(
    const w34_exec* ex, const void* local_in, void* out,
    void* stage0, void* stage1, void* stage2, void* stage3,
    void* sig0, void* sig1, void* sig2, void* sig3, int rank, int num_elems) {
    if (ex == nullptr || local_in == nullptr || out == nullptr ||
        stage0 == nullptr || stage1 == nullptr || stage2 == nullptr || stage3 == nullptr ||
        sig0 == nullptr || sig1 == nullptr || sig2 == nullptr || sig3 == nullptr ||
        rank < 0 || rank >= 4 || num_elems <= 0) {
        return (int)cudaErrorInvalidValue;
    }
    const bool vec8 = ((num_elems & 7) == 0) &&
                      ((((uintptr_t)local_in | (uintptr_t)out | (uintptr_t)stage0 | (uintptr_t)stage1 | (uintptr_t)stage2 | (uintptr_t)stage3) & 15u) == 0);
    const int elems_per_block = vec8 ? 4096 : 512;
    int blocks = (num_elems + elems_per_block - 1) / elems_per_block;
    int cap = w34_comm::tp2_block_cap(num_elems <= 36 * 5120);
    if (num_elems <= 36 * 5120 && cap > 16) cap = 16;
    if (blocks > cap) blocks = cap;
    if (blocks < 1) blocks = 1;
    w34_comm::Tp4Ptrs ptrs{
        {stage0, stage1, stage2, stage3},
        {(w34_comm::Signal*)sig0, (w34_comm::Signal*)sig1, (w34_comm::Signal*)sig2, (w34_comm::Signal*)sig3},
    };
    w34_comm::tp4_all_reduce_bf16_kernel<<<blocks, 512, 0, (cudaStream_t)ex->stream>>>(
        (const __nv_bfloat16*)local_in, (__nv_bfloat16*)out, ptrs, rank, num_elems);
    return (int)cudaGetLastError();
}

extern "C" int w34_tp4_all_reduce_f32(
    const w34_exec* ex, const float* local_in, float* out,
    float* stage0, float* stage1, float* stage2, float* stage3,
    void* sig0, void* sig1, void* sig2, void* sig3, int rank, int num_elems) {
    if (ex == nullptr || local_in == nullptr || out == nullptr ||
        stage0 == nullptr || stage1 == nullptr || stage2 == nullptr || stage3 == nullptr ||
        sig0 == nullptr || sig1 == nullptr || sig2 == nullptr || sig3 == nullptr ||
        rank < 0 || rank >= 4 || num_elems <= 0) {
        return (int)cudaErrorInvalidValue;
    }
    const bool vec4 = ((num_elems & 3) == 0) &&
                      ((((uintptr_t)local_in | (uintptr_t)out | (uintptr_t)stage0 | (uintptr_t)stage1 | (uintptr_t)stage2 | (uintptr_t)stage3) & 15u) == 0);
    const int elems_per_block = vec4 ? 2048 : 512;
    int blocks = (num_elems + elems_per_block - 1) / elems_per_block;
    int cap = w34_comm::tp2_block_cap(num_elems <= 36 * 5120);
    if (num_elems <= 36 * 5120 && cap > 16) cap = 16;
    if (blocks > cap) blocks = cap;
    if (blocks < 1) blocks = 1;
    w34_comm::Tp4Ptrs ptrs{
        {stage0, stage1, stage2, stage3},
        {(w34_comm::Signal*)sig0, (w34_comm::Signal*)sig1, (w34_comm::Signal*)sig2, (w34_comm::Signal*)sig3},
    };
    w34_comm::tp4_all_reduce_f32_kernel<<<blocks, 512, 0, (cudaStream_t)ex->stream>>>(
        local_in, out, ptrs, rank, num_elems);
    return (int)cudaGetLastError();
}

extern "C" int w34_tp4_broadcast_i32(
    const w34_exec* ex, const int32_t* src_rank0, int32_t* out,
    int32_t* stage0, int32_t* stage1, int32_t* stage2, int32_t* stage3,
    void* sig0, void* sig1, void* sig2, void* sig3, int rank, int num_elems) {
    if (ex == nullptr || (rank == 0 && src_rank0 == nullptr) || out == nullptr ||
        stage0 == nullptr || stage1 == nullptr || stage2 == nullptr || stage3 == nullptr ||
        sig0 == nullptr || sig1 == nullptr || sig2 == nullptr || sig3 == nullptr ||
        rank < 0 || rank >= 4 || num_elems <= 0) {
        return (int)cudaErrorInvalidValue;
    }
    w34_comm::Tp4Ptrs ptrs{
        {stage0, stage1, stage2, stage3},
        {(w34_comm::Signal*)sig0, (w34_comm::Signal*)sig1, (w34_comm::Signal*)sig2, (w34_comm::Signal*)sig3},
    };
    w34_comm::tp4_broadcast_i32_kernel<<<1, 256, 0, (cudaStream_t)ex->stream>>>(
        src_rank0, out, ptrs, rank, num_elems);
    return (int)cudaGetLastError();
}

extern "C" int w34_tp4_all_gather_top1_f32_i32(
    const w34_exec* ex, const float* local_max_val, const int32_t* local_max_id,
    float* global_max_val, int32_t* global_max_id,
    void* stage0, void* stage1, void* stage2, void* stage3,
    void* sig0, void* sig1, void* sig2, void* sig3, int rank, int num_rows) {
    if (ex == nullptr || local_max_val == nullptr || local_max_id == nullptr ||
        global_max_val == nullptr || global_max_id == nullptr ||
        stage0 == nullptr || stage1 == nullptr || stage2 == nullptr || stage3 == nullptr ||
        sig0 == nullptr || sig1 == nullptr || sig2 == nullptr || sig3 == nullptr ||
        rank < 0 || rank >= 4 || num_rows <= 0) {
        return (int)cudaErrorInvalidValue;
    }
    w34_comm::Tp4Ptrs ptrs{
        {stage0, stage1, stage2, stage3},
        {(w34_comm::Signal*)sig0, (w34_comm::Signal*)sig1, (w34_comm::Signal*)sig2, (w34_comm::Signal*)sig3},
    };
    w34_comm::tp4_all_gather_top1_f32_i32_kernel<<<1, 256, 0, (cudaStream_t)ex->stream>>>(
        local_max_val, local_max_id, global_max_val, global_max_id, ptrs, rank, num_rows);
    return (int)cudaGetLastError();
}

extern "C" int w34_tp4_gather_vocab_f16(
    const w34_exec* ex, const void* local_logits_f16, void* rank0_full_logits_f16,
    void* stage0, void* stage1, void* stage2, void* stage3,
    void* sig0, void* sig1, void* sig2, void* sig3, int rank, int num_rows, int local_vocab_size) {
    if (ex == nullptr || local_logits_f16 == nullptr ||
        (rank == 0 && rank0_full_logits_f16 == nullptr) ||
        stage1 == nullptr || stage2 == nullptr || stage3 == nullptr ||
        sig0 == nullptr || sig1 == nullptr || sig2 == nullptr || sig3 == nullptr ||
        rank < 0 || rank >= 4 || num_rows <= 0 || local_vocab_size <= 0) {
        return (int)cudaErrorInvalidValue;
    }
    const int num_elems = num_rows * local_vocab_size;
    int blocks = (num_elems + 511) / 512;
    const int cap = w34_comm::tp2_block_cap(num_elems <= 36 * 5120);
    if (blocks > cap) blocks = cap;
    if (blocks < 1) blocks = 1;
    w34_comm::Tp4Ptrs ptrs{
        {stage0, stage1, stage2, stage3},
        {(w34_comm::Signal*)sig0, (w34_comm::Signal*)sig1, (w34_comm::Signal*)sig2, (w34_comm::Signal*)sig3},
    };
    w34_comm::tp4_gather_vocab_f16_kernel<<<blocks, 512, 0, (cudaStream_t)ex->stream>>>(
        (const __half*)local_logits_f16, (__half*)rank0_full_logits_f16, ptrs, rank, num_rows, local_vocab_size);
    return (int)cudaGetLastError();
}

extern "C" int w34_tp2_all_gather_topk_f32_i32(
    const w34_exec* ex, const float* local_vals, const int32_t* local_ids,
    float* global_vals, int32_t* global_ids, void* stage0, void* stage1,
    void* sig0, void* sig1, int rank, int num_rows, int top_k) {
    if (!ex || !local_vals || !local_ids || !global_vals || !global_ids ||
        !stage0 || !stage1 || !sig0 || !sig1 || (rank != 0 && rank != 1) ||
        num_rows <= 0 || top_k <= 0 || top_k > 32 ||
        (int64_t)num_rows * top_k > 65536) {
        return (int)cudaErrorInvalidValue;
    }
    w34_comm::tp2_all_gather_topk_f32_i32_kernel<<<1, 256, 0, (cudaStream_t)ex->stream>>>(
        local_vals, local_ids, global_vals, global_ids,
        (w34_comm::Top1Pair*)stage0, (w34_comm::Top1Pair*)stage1,
        (w34_comm::Signal*)sig0, (w34_comm::Signal*)sig1, rank, num_rows, top_k);
    return (int)cudaGetLastError();
}

extern "C" int w34_tp4_all_gather_topk_f32_i32(
    const w34_exec* ex, const float* local_vals, const int32_t* local_ids,
    float* global_vals, int32_t* global_ids, void* stage0, void* stage1,
    void* stage2, void* stage3, void* sig0, void* sig1, void* sig2, void* sig3,
    int rank, int num_rows, int top_k) {
    if (!ex || !local_vals || !local_ids || !global_vals || !global_ids ||
        !stage0 || !stage1 || !stage2 || !stage3 || !sig0 || !sig1 || !sig2 || !sig3 ||
        rank < 0 || rank >= 4 || num_rows <= 0 || top_k <= 0 || top_k > 32 ||
        (int64_t)num_rows * top_k > 65536) {
        return (int)cudaErrorInvalidValue;
    }
    w34_comm::Tp4Ptrs ptrs{
        {stage0, stage1, stage2, stage3},
        {(w34_comm::Signal*)sig0, (w34_comm::Signal*)sig1, (w34_comm::Signal*)sig2, (w34_comm::Signal*)sig3},
    };
    w34_comm::tp4_all_gather_topk_f32_i32_kernel<<<1, 256, 0, (cudaStream_t)ex->stream>>>(
        local_vals, local_ids, global_vals, global_ids, ptrs, rank, num_rows, top_k);
    return (int)cudaGetLastError();
}

extern "C" int w34_tp2_all_reduce_hc_combine_tap_f16(
    const w34_exec* ex, const void* residual_in, const void* local_block_out,
    void* ar_out, const float* inject_weights, const void* ple_delta,
    void* residual_out, void* tap_bf16_out, const uint8_t* valid_token_mask,
    void* stage0, void* stage1, void* sig0, void* sig1, int rank, int num_tokens,
    int hc_count, int hidden_size, int tap_stride_elems, int tap_offset_elems) {
    if (!ex || !residual_in || !local_block_out || !inject_weights || !residual_out ||
        !stage0 || !stage1 || !sig0 || !sig1 || (rank != 0 && rank != 1) ||
        num_tokens <= 0 || hc_count <= 0 || hc_count > 8 || hidden_size <= 0 || hidden_size > 6144 ||
        (int64_t)num_tokens * hidden_size > (INT32_MAX / 2) ||
        (tap_bf16_out != nullptr && (tap_offset_elems < 0 || tap_stride_elems < tap_offset_elems + hidden_size))) {
        return (int)cudaErrorInvalidValue;
    }
    const size_t num_bytes = (size_t)num_tokens * (size_t)hidden_size * sizeof(__half);
    if (num_bytes >= w34_comm::kDmaThresholdBytes) {
        cudaError_t cpy_err = cudaMemcpyAsync(
            (rank == 0) ? stage0 : stage1, local_block_out, num_bytes,
            cudaMemcpyDefault, (cudaStream_t)ex->stream);
        if (cpy_err != cudaSuccess) return (int)cpy_err;
    }
    int blocks = num_tokens;
    int cap = w34_comm::tp2_block_cap(true);
    if (cap > 16) cap = 16;
    if (blocks > cap) blocks = cap;
    w34_comm::tp2_all_reduce_hc_combine_tap_f16_kernel<<<blocks, 256, 0, (cudaStream_t)ex->stream>>>(
        (const __half*)residual_in, (const __half*)local_block_out, (__half*)ar_out,
        inject_weights, (const __half*)ple_delta, (__half*)residual_out,
        (__nv_bfloat16*)tap_bf16_out, valid_token_mask,
        (__half*)stage0, (__half*)stage1, (w34_comm::Signal*)sig0, (w34_comm::Signal*)sig1,
        rank, num_tokens, hc_count, hidden_size, tap_stride_elems, tap_offset_elems);
    return (int)cudaGetLastError();
}

extern "C" int w34_tp4_all_reduce_hc_combine_tap_f16(
    const w34_exec* ex, const void* residual_in, const void* local_block_out,
    void* ar_out, const float* inject_weights, const void* ple_delta,
    void* residual_out, void* tap_bf16_out, const uint8_t* valid_token_mask,
    void* stage0, void* stage1, void* stage2, void* stage3,
    void* sig0, void* sig1, void* sig2, void* sig3, int rank, int num_tokens,
    int hc_count, int hidden_size, int tap_stride_elems, int tap_offset_elems) {
    if (!ex || !residual_in || !local_block_out || !inject_weights || !residual_out ||
        !stage0 || !stage1 || !stage2 || !stage3 || !sig0 || !sig1 || !sig2 || !sig3 ||
        rank < 0 || rank >= 4 || num_tokens <= 0 || hc_count <= 0 || hc_count > 8 ||
        hidden_size <= 0 || hidden_size > 6144 || (int64_t)num_tokens * hidden_size > (INT32_MAX / 2) ||
        (tap_bf16_out != nullptr && (tap_offset_elems < 0 || tap_stride_elems < tap_offset_elems + hidden_size))) {
        return (int)cudaErrorInvalidValue;
    }
    int blocks = num_tokens;
    int cap = w34_comm::tp2_block_cap(num_tokens * hidden_size <= 36 * 5120);
    if (num_tokens * hidden_size <= 36 * 5120 && cap > 16) cap = 16;
    if (blocks > cap) blocks = cap;
    w34_comm::Tp4Ptrs ptrs{
        {stage0, stage1, stage2, stage3},
        {(w34_comm::Signal*)sig0, (w34_comm::Signal*)sig1, (w34_comm::Signal*)sig2, (w34_comm::Signal*)sig3},
    };
    w34_comm::tp4_all_reduce_hc_combine_tap_f16_kernel<<<blocks, 256, 0, (cudaStream_t)ex->stream>>>(
        (const __half*)residual_in, (const __half*)local_block_out, (__half*)ar_out,
        inject_weights, (const __half*)ple_delta, (__half*)residual_out,
        (__nv_bfloat16*)tap_bf16_out, valid_token_mask, ptrs,
        rank, num_tokens, hc_count, hidden_size, tap_stride_elems, tap_offset_elems);
    return (int)cudaGetLastError();
}

extern "C" int w34_tp2_warmup(void) {
    cudaFuncAttributes attr;
    #define WARM(fn) do { cudaError_t err = cudaFuncGetAttributes(&attr, (fn)); if (err != cudaSuccess) return (int)err; } while (0)
    WARM(w34_comm::tp2_all_reduce_f16_kernel);
    WARM((w34_comm::tp2_all_reduce_residual_gemma_rms_norm_f16_kernel<256, 10>));
    WARM((w34_comm::tp2_all_reduce_residual_gemma_rms_norm_f16_kernel<256, 16>));
    WARM((w34_comm::tp2_all_reduce_residual_gemma_rms_norm_f16_kernel<512, 10>));
    WARM((w34_comm::tp2_all_reduce_residual_gemma_rms_norm_f16_kernel<512, 12>));
    WARM(w34_comm::tp2_all_gather_top1_f32_i32_kernel);
    WARM(w34_comm::tp2_gather_vocab_f16_kernel);
    WARM(w34_comm::tp2_broadcast_i32_kernel);
    WARM(w34_comm::tp4_all_reduce_f16_kernel);
    WARM((w34_comm::tp4_all_reduce_residual_gemma_rms_norm_f16_kernel<256, 10>));
    WARM((w34_comm::tp4_all_reduce_residual_gemma_rms_norm_f16_kernel<256, 16>));
    WARM((w34_comm::tp4_all_reduce_residual_gemma_rms_norm_f16_kernel<512, 10>));
    WARM((w34_comm::tp4_all_reduce_residual_gemma_rms_norm_f16_kernel<512, 12>));
    WARM(w34_comm::tp4_all_reduce_bf16_kernel);
    WARM(w34_comm::tp4_all_reduce_f32_kernel);
    WARM(w34_comm::tp4_all_gather_top1_f32_i32_kernel);
    WARM(w34_comm::tp4_gather_vocab_f16_kernel);
    WARM(w34_comm::tp4_broadcast_i32_kernel);
    WARM(w34_comm::tp2_all_gather_topk_f32_i32_kernel);
    WARM(w34_comm::tp4_all_gather_topk_f32_i32_kernel);
    WARM(w34_comm::tp2_all_reduce_hc_combine_tap_f16_kernel);
    WARM(w34_comm::tp4_all_reduce_hc_combine_tap_f16_kernel);
    #undef WARM
    return (int)cudaSuccess;
}
