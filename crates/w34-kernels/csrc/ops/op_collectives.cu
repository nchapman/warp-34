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

constexpr int kMaxBlocks = 84;
constexpr int kSmallMaxBlocks = 36;
constexpr int kRanks = 16;

struct Signal {
    alignas(128) uint32_t start[kMaxBlocks][kRanks];
    alignas(128) uint32_t end[kMaxBlocks][kRanks];
    alignas(128) uint32_t flag[kMaxBlocks];
};
static_assert(sizeof(Signal) == 11136, "Signal ABI size mismatch");

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
    if (threadIdx.x < 2) {
        Signal* target_sg = (threadIdx.x == 0) ? sig0 : sig1;
        uint32_t* peer_counter = &target_sg->start[blockIdx.x][rank];
        uint32_t* self_counter = &self_sg->start[blockIdx.x][threadIdx.x];
        st_flag_release(peer_counter, epoch);
        while (static_cast<int32_t>(ld_flag_acquire(self_counter) - epoch) < 0) {
            __nanosleep(128);
        }
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        self_sg->flag[blockIdx.x] = epoch;
    }
    *out_epoch = epoch;
}

__device__ __forceinline__ float warp_reduce_sum_f32(float val) {
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        val += __shfl_xor_sync(0xffffffffu, val, offset);
    }
    return val;
}

__device__ __forceinline__ float block_reduce_sum_256(float val, float* s_warp) {
    const int lane = threadIdx.x & 31;
    const int wid = threadIdx.x >> 5;
    val = warp_reduce_sum_f32(val);
    if (lane == 0) {
        s_warp[wid] = val;
    }
    __syncthreads();
    val = (threadIdx.x < 8) ? s_warp[lane] : 0.0f;
    if (wid == 0) {
        val = warp_reduce_sum_f32(val);
        if (lane == 0) {
            s_warp[0] = val;
        }
    }
    __syncthreads();
    return s_warp[0];
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
    uint32_t epoch_pre = self_sg->flag[blockIdx.x] + 1;
    bool odd = (epoch_pre & 1u) != 0;

    __half* s0_buf = odd ? (s0 + num_elems) : s0;
    __half* s1_buf = odd ? (s1 + num_elems) : s1;
    __half* __restrict__ my_s = (rank == 0) ? s0_buf : s1_buf;
    const __half* __restrict__ peer_s = (rank == 0) ? s1_buf : s0_buf;

    const bool vec8 = ((num_elems & 7) == 0) &&
                      ((((uintptr_t)local_in | (uintptr_t)out | (uintptr_t)my_s | (uintptr_t)peer_s) & 15u) == 0);
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
}

template <int MAX_ITERS>
__global__ void __launch_bounds__(256, 2) tp2_all_reduce_residual_gemma_rms_norm_f16_kernel(
    const __half* local_in,
    __half* ar_out,
    __half* __restrict__ residual,
    const __half* __restrict__ weight,
    __half* __restrict__ out,
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
    Signal* self_sg = (rank == 0) ? sig0 : sig1;
    uint32_t epoch_pre = self_sg->flag[blockIdx.x] + 1;
    bool odd = (epoch_pre & 1u) != 0;

    __half* s0_buf = odd ? (s0 + num_elems) : s0;
    __half* s1_buf = odd ? (s1 + num_elems) : s1;
    __half* __restrict__ my_s = (rank == 0) ? s0_buf : s1_buf;
    const __half* __restrict__ peer_s = (rank == 0) ? s1_buf : s0_buf;

    constexpr int VEC_ITERS = (MAX_ITERS + 7) / 8;
    const bool vec8 = ((hidden_size & 7) == 0) &&
                      ((((uintptr_t)local_in | (uintptr_t)my_s | (uintptr_t)peer_s |
                         (uintptr_t)residual | (uintptr_t)weight | (uintptr_t)out |
                         (uintptr_t)ar_out) & 15u) == 0);
    // Odd stage count (1, 3, or 5) overlaps PCIe Gen4 x8 TX and RX in full duplex across
    // disjoint row slices on large-M prefill while guaranteeing `self_sg->flag[blockIdx.x]`
    // flips odd/even double-buffer parity on every kernel invocation (CUDA Graph safe).
    const int num_stages = (num_tokens > (int)gridDim.x * 12)
        ? 5
        : ((num_tokens > (int)gridDim.x * 4) ? 3 : 1);
    const int stage_rows = (num_tokens + num_stages - 1) / num_stages;

    __shared__ float s_warp[32];
    __shared__ float s_v[MAX_ITERS * 256];

    for (int stage = 0; stage < num_stages; ++stage) {
        const int row_begin = stage * stage_rows;
        int row_end = row_begin + stage_rows;
        if (row_end > num_tokens) row_end = num_tokens;

        if (vec8) {
            const int vec_cols = hidden_size >> 3;
            for (int row = row_begin + (int)blockIdx.x; row < row_end; row += (int)gridDim.x) {
                const uint4* src_v = reinterpret_cast<const uint4*>(local_in + (size_t)row * hidden_size);
                uint4* dst_v = reinterpret_cast<uint4*>(my_s + (size_t)row * hidden_size);
#pragma unroll
                for (int k = 0; k < VEC_ITERS; ++k) {
                    const int c = threadIdx.x + k * 256;
                    if (c < vec_cols) {
                        dst_v[c] = src_v[c];
                    }
                }
            }
        } else {
            for (int row = row_begin + (int)blockIdx.x; row < row_end; row += (int)gridDim.x) {
                const __half* src_r = local_in + (size_t)row * hidden_size;
                __half* dst_r = my_s + (size_t)row * hidden_size;
                for (int c = threadIdx.x; c < hidden_size; c += 256) {
                    dst_r[c] = src_r[c];
                }
            }
        }

        uint32_t epoch = 0;
        barrier_epoch_release(sig0, sig1, self_sg, rank, &epoch);

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
                uint4* out_v = reinterpret_cast<uint4*>(out + (size_t)row * hidden_size);
                __nv_bfloat16* tap_row = (tap_dst_bf16 != nullptr)
                    ? (tap_dst_bf16 + (size_t)row * tap_row_stride + tap_col_offset)
                    : nullptr;

                float sum_sq = 0.0f;
#pragma unroll
                for (int k = 0; k < VEC_ITERS; ++k) {
                    const int c = threadIdx.x + k * 256;
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
                            s_v[base_i + d] = v;
                            sum_sq += v * v;
                        }
                        if (ar_v != nullptr) {
                            ar_v[c] = arv;
                        }
                        res_v[c] = sumv;
                    }
                }
                sum_sq = block_reduce_sum_256(sum_sq, s_warp);
                const float inv_rms = rsqrtf(sum_sq / (float)hidden_size + eps);

#pragma unroll
                for (int k = 0; k < VEC_ITERS; ++k) {
                    const int c = threadIdx.x + k * 256;
                    if (c < vec_cols) {
                        const uint4 wv = weight_v[c];
                        const __half* wh = reinterpret_cast<const __half*>(&wv);
                        uint4 ov;
                        __half* oh = reinterpret_cast<__half*>(&ov);
                        const int base_i = c << 3;
#pragma unroll
                        for (int d = 0; d < 8; ++d) {
                            const float v = s_v[base_i + d] * inv_rms;
                            const float w = 1.0f + __half2float(wh[d]);
                            oh[d] = __float2half(v * w);
                        }
                        out_v[c] = ov;
                    }
                }
                __syncthreads();
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

                float sum_sq = 0.0f;
#pragma unroll
                for (int k = 0; k < MAX_ITERS; k++) {
                    const int i = threadIdx.x + k * 256;
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
                        s_v[i] = v;
                        sum_sq += v * v;
                    }
                }
                sum_sq = block_reduce_sum_256(sum_sq, s_warp);
                const float inv_rms = rsqrtf(sum_sq / (float)hidden_size + eps);

#pragma unroll
                for (int k = 0; k < MAX_ITERS; k++) {
                    const int i = threadIdx.x + k * 256;
                    if (i < hidden_size) {
                        const float v = s_v[i] * inv_rms;
                        const float w = 1.0f + __half2float(weight[i]);
                        out_row[i] = __float2half(v * w);
                    }
                }
                __syncthreads();
            }
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
    uint32_t epoch_pre = self_sg->flag[blockIdx.x] + 1;
    bool odd = (epoch_pre & 1u) != 0;

    __nv_bfloat16* s0_buf = odd ? (s0 + num_elems) : s0;
    __nv_bfloat16* s1_buf = odd ? (s1 + num_elems) : s1;
    __nv_bfloat16* __restrict__ my_s = (rank == 0) ? s0_buf : s1_buf;
    const __nv_bfloat16* __restrict__ peer_s = (rank == 0) ? s1_buf : s0_buf;

    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < num_elems; i += gridDim.x * blockDim.x) {
        my_s[i] = local_in[i];
    }

    uint32_t epoch = 0;
    barrier_epoch_release(sig0, sig1, self_sg, rank, &epoch);

    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < num_elems; i += gridDim.x * blockDim.x) {
        const float fl = __bfloat162float(local_in[i]);
        const float fp = __bfloat162float(peer_s[i]);
        out[i] = __float2bfloat16((rank == 0) ? (fl + fp) : (fp + fl));
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
    __half* s1_buf = odd ? (s1 + num_elems) : s1;
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
    const int cap = (num_elems <= 36 * 5120) ? w34_comm::kSmallMaxBlocks : w34_comm::kMaxBlocks;
    if (blocks > cap) blocks = cap;
    if (blocks < 1) blocks = 1;
    w34_comm::tp2_all_reduce_f16_kernel<<<blocks, 512, 0, (cudaStream_t)ex->stream>>>(
        (const __half*)local_in, (__half*)out,
        (__half*)stage0, (__half*)stage1,
        (w34_comm::Signal*)sig0, (w34_comm::Signal*)sig1,
        rank, num_elems);
    return (int)cudaGetLastError();
}

extern "C" int w34_tp2_all_reduce_residual_gemma_rms_norm_f16(
    const w34_exec* ex,
    const void* local_in,
    void* ar_out,
    void* residual,
    const void* weight,
    void* out,
    void* tap_dst_bf16,
    void* stage0,
    void* stage1,
    void* sig0,
    void* sig1,
    int rank,
    int num_tokens,
    int hidden_size,
    float eps,
    int tap_row_stride,
    int tap_col_offset) {
    if (ex == nullptr || local_in == nullptr || residual == nullptr ||
        weight == nullptr || out == nullptr || stage0 == nullptr ||
        stage1 == nullptr || sig0 == nullptr || sig1 == nullptr ||
        (rank != 0 && rank != 1) || num_tokens < 0 || hidden_size <= 0 ||
        hidden_size > 6144) {
        return (int)cudaErrorInvalidValue;
    }
    if (tap_dst_bf16 != nullptr &&
        (tap_col_offset < 0 || tap_row_stride < hidden_size ||
         tap_col_offset > tap_row_stride - hidden_size)) {
        return (int)cudaErrorInvalidValue;
    }
    if (num_tokens == 0) return 0;
    int blocks = num_tokens;
    const int cap = (num_tokens <= 36) ? w34_comm::kSmallMaxBlocks : w34_comm::kMaxBlocks;
    if (blocks > cap) blocks = cap;
    if (hidden_size <= 2560) {
        w34_comm::tp2_all_reduce_residual_gemma_rms_norm_f16_kernel<10>
            <<<blocks, 256, 0, (cudaStream_t)ex->stream>>>(
                (const __half*)local_in, (__half*)ar_out, (__half*)residual,
                (const __half*)weight, (__half*)out,
                (__nv_bfloat16*)tap_dst_bf16, (__half*)stage0, (__half*)stage1,
                (w34_comm::Signal*)sig0, (w34_comm::Signal*)sig1,
                rank, num_tokens, hidden_size, eps, tap_row_stride, tap_col_offset);
    } else if (hidden_size <= 4096) {
        w34_comm::tp2_all_reduce_residual_gemma_rms_norm_f16_kernel<16>
            <<<blocks, 256, 0, (cudaStream_t)ex->stream>>>(
                (const __half*)local_in, (__half*)ar_out, (__half*)residual,
                (const __half*)weight, (__half*)out,
                (__nv_bfloat16*)tap_dst_bf16, (__half*)stage0, (__half*)stage1,
                (w34_comm::Signal*)sig0, (w34_comm::Signal*)sig1,
                rank, num_tokens, hidden_size, eps, tap_row_stride, tap_col_offset);
    } else if (hidden_size <= 5120) {
        w34_comm::tp2_all_reduce_residual_gemma_rms_norm_f16_kernel<20>
            <<<blocks, 256, 0, (cudaStream_t)ex->stream>>>(
                (const __half*)local_in, (__half*)ar_out, (__half*)residual,
                (const __half*)weight, (__half*)out,
                (__nv_bfloat16*)tap_dst_bf16, (__half*)stage0, (__half*)stage1,
                (w34_comm::Signal*)sig0, (w34_comm::Signal*)sig1,
                rank, num_tokens, hidden_size, eps, tap_row_stride, tap_col_offset);
    } else {
        w34_comm::tp2_all_reduce_residual_gemma_rms_norm_f16_kernel<24>
            <<<blocks, 256, 0, (cudaStream_t)ex->stream>>>(
                (const __half*)local_in, (__half*)ar_out, (__half*)residual,
                (const __half*)weight, (__half*)out,
                (__nv_bfloat16*)tap_dst_bf16, (__half*)stage0, (__half*)stage1,
                (w34_comm::Signal*)sig0, (w34_comm::Signal*)sig1,
                rank, num_tokens, hidden_size, eps, tap_row_stride, tap_col_offset);
    }
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
    const int cap = (num_elems <= 36 * 5120) ? w34_comm::kSmallMaxBlocks : w34_comm::kMaxBlocks;
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
    int blocks = (num_elems + 511) / 512;
    const int cap = (num_elems <= 36 * 5120) ? w34_comm::kSmallMaxBlocks : w34_comm::kMaxBlocks;
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

extern "C" int w34_tp2_gather_vocab_f16(
    const w34_exec* ex,
    const void* local_logits_f16,
    void* rank0_full_logits_f16,
    void* stage0,
    void* stage1,
    void* sig0,
    void* sig1,
    int rank,
    int num_rows,
    int local_vocab_size) {
    (void)stage0;
    if (ex == nullptr || local_logits_f16 == nullptr ||
        (rank == 0 && rank0_full_logits_f16 == nullptr) ||
        stage1 == nullptr || sig0 == nullptr || sig1 == nullptr ||
        (rank != 0 && rank != 1) || num_rows <= 0 || local_vocab_size <= 0) {
        return (int)cudaErrorInvalidValue;
    }
    const int num_elems = num_rows * local_vocab_size;
    int blocks = (num_elems + 511) / 512;
    const int cap = (num_elems <= 36 * 5120) ? w34_comm::kSmallMaxBlocks : w34_comm::kMaxBlocks;
    if (blocks > cap) blocks = cap;
    if (blocks < 1) blocks = 1;
    w34_comm::tp2_gather_vocab_f16_kernel<<<blocks, 512, 0, (cudaStream_t)ex->stream>>>(
        (const __half*)local_logits_f16,
        (__half*)rank0_full_logits_f16,
        (__half*)stage1,
        (w34_comm::Signal*)sig0,
        (w34_comm::Signal*)sig1,
        rank,
        num_rows,
        local_vocab_size);
    return (int)cudaGetLastError();
}

extern "C" int w34_tp2_warmup(void) {
    cudaFuncAttributes attr;
    cudaError_t err = cudaFuncGetAttributes(&attr, w34_comm::tp2_all_reduce_f16_kernel);
    if (err != cudaSuccess) return (int)err;
    err = cudaFuncGetAttributes(&attr, w34_comm::tp2_all_gather_top1_f32_i32_kernel);
    if (err != cudaSuccess) return (int)err;
    err = cudaFuncGetAttributes(&attr, w34_comm::tp2_gather_vocab_f16_kernel);
    if (err != cudaSuccess) return (int)err;
    err = cudaFuncGetAttributes(&attr, w34_comm::tp2_broadcast_i32_kernel);
    return (int)err;
}
