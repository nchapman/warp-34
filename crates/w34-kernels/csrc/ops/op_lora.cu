#include "../w34_cuda.h"
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cstdint>

namespace {

constexpr int kLoraMaxRankPad = 512;
constexpr int kLoraFusedMaxRankPad = 16;
constexpr int kLoraFusedMaxBlocks = 8;
constexpr int kLoraMaxAdapters = 64;

__global__ __launch_bounds__(512, 2) void lora_fused_shrink_expand_512t_f16_kernel(
    const __half* __restrict__ x,
    const __half* __restrict__ lora_a,
    const __half* __restrict__ lora_b,
    const int32_t* __restrict__ token_lora_ids,
    __half* __restrict__ y_inout,
    int size_n,
    int size_k,
    int rank_pad,
    int num_adapters,
    int default_lora_id,
    uint64_t adapter_mask,
    bool vec8_aligned,
    bool vec4_aligned) {
    const int tok = (int)blockIdx.y;
    const int32_t lora_id = (token_lora_ids != nullptr) ? token_lora_ids[tok] : default_lora_id;
    if (lora_id < 0 || lora_id >= num_adapters || ((adapter_mask >> lora_id) & 1ULL) == 0ULL) {
        return;
    }

    __shared__ alignas(16) float s_lowrank[kLoraFusedMaxRankPad];
    __shared__ float4 s_warp[16];
    const __half* x_row = x + (size_t)tok * size_k;
    const int lane = (int)threadIdx.x & 31;
    const int warp = (int)threadIdx.x >> 5;

    for (int r_base = 0; r_base < rank_pad; r_base += 4) {
        const __half* a_base = lora_a + ((size_t)lora_id * rank_pad + r_base) * size_k;
        const int rem_r = rank_pad - r_base;
        float acc0 = 0.0f;
        float acc1 = 0.0f;
        float acc2 = 0.0f;
        float acc3 = 0.0f;

        if (vec8_aligned && rem_r >= 4) {
            const __half* a_row0 = a_base;
            const __half* a_row1 = a_base + size_k;
            const __half* a_row2 = a_base + (size_t)2 * size_k;
            const __half* a_row3 = a_base + (size_t)3 * size_k;
#pragma unroll 4
            for (int k = (int)threadIdx.x * 8; k < size_k; k += 4096) {
                const uint4 xv = *reinterpret_cast<const uint4*>(x_row + k);
                const uint4 av0 = *reinterpret_cast<const uint4*>(a_row0 + k);
                const uint4 av1 = *reinterpret_cast<const uint4*>(a_row1 + k);
                const uint4 av2 = *reinterpret_cast<const uint4*>(a_row2 + k);
                const uint4 av3 = *reinterpret_cast<const uint4*>(a_row3 + k);
                const __half* xh = reinterpret_cast<const __half*>(&xv);
                const __half* ah0 = reinterpret_cast<const __half*>(&av0);
                const __half* ah1 = reinterpret_cast<const __half*>(&av1);
                const __half* ah2 = reinterpret_cast<const __half*>(&av2);
                const __half* ah3 = reinterpret_cast<const __half*>(&av3);
#pragma unroll
                for (int j = 0; j < 8; j++) {
                    const float xf = __half2float(xh[j]);
                    acc0 = fmaf(xf, __half2float(ah0[j]), acc0);
                    acc1 = fmaf(xf, __half2float(ah1[j]), acc1);
                    acc2 = fmaf(xf, __half2float(ah2[j]), acc2);
                    acc3 = fmaf(xf, __half2float(ah3[j]), acc3);
                }
            }
        } else {
            for (int k = (int)threadIdx.x; k < size_k; k += 512) {
                const float xf = __half2float(x_row[k]);
                acc0 = fmaf(xf, __half2float(a_base[k]), acc0);
                if (rem_r > 1) acc1 = fmaf(xf, __half2float(a_base[(size_t)size_k + k]), acc1);
                if (rem_r > 2) acc2 = fmaf(xf, __half2float(a_base[(size_t)2 * size_k + k]), acc2);
                if (rem_r > 3) acc3 = fmaf(xf, __half2float(a_base[(size_t)3 * size_k + k]), acc3);
            }
        }

#pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1) {
            acc0 += __shfl_down_sync(0xffffffffu, acc0, offset);
            acc1 += __shfl_down_sync(0xffffffffu, acc1, offset);
            acc2 += __shfl_down_sync(0xffffffffu, acc2, offset);
            acc3 += __shfl_down_sync(0xffffffffu, acc3, offset);
        }

        if (lane == 0) {
            s_warp[warp] = make_float4(acc0, acc1, acc2, acc3);
        }
        __syncthreads();

        if (warp == 0) {
            float4 v = (lane < 16) ? s_warp[lane] : make_float4(0.0f, 0.0f, 0.0f, 0.0f);
#pragma unroll
            for (int offset = 8; offset > 0; offset >>= 1) {
                v.x += __shfl_down_sync(0xffffffffu, v.x, offset);
                v.y += __shfl_down_sync(0xffffffffu, v.y, offset);
                v.z += __shfl_down_sync(0xffffffffu, v.z, offset);
                v.w += __shfl_down_sync(0xffffffffu, v.w, offset);
            }
            if (lane == 0) {
                if (rem_r >= 4) {
                    *reinterpret_cast<float4*>(s_lowrank + r_base) = v;
                } else {
                    s_lowrank[r_base] = v.x;
                    if (rem_r > 1) s_lowrank[r_base + 1] = v.y;
                    if (rem_r > 2) s_lowrank[r_base + 2] = v.z;
                    if (rem_r > 3) s_lowrank[r_base + 3] = v.w;
                }
            }
        }
        __syncthreads();
    }

    const int col0 = ((int)blockIdx.x * 512 + (int)threadIdx.x) * 2;
    if (col0 >= size_n) return;

    const __half* b_row0 = lora_b + ((size_t)lora_id * size_n + col0) * rank_pad;
    float sum0 = 0.0f;
    float sum1 = 0.0f;

    if (col0 + 1 < size_n) {
        const __half* b_row1 = b_row0 + rank_pad;
        if (vec4_aligned) {
            const float4* s_lr4 = reinterpret_cast<const float4*>(s_lowrank);
            const int vec_ranks = rank_pad / 4;
            for (int r4 = 0; r4 < vec_ranks; r4++) {
                const float4 hv = s_lr4[r4];
                const uint2 bv0 = *reinterpret_cast<const uint2*>(b_row0 + r4 * 4);
                const uint2 bv1 = *reinterpret_cast<const uint2*>(b_row1 + r4 * 4);
                const __half* bh0 = reinterpret_cast<const __half*>(&bv0);
                const __half* bh1 = reinterpret_cast<const __half*>(&bv1);
                sum0 = fmaf(hv.x, __half2float(bh0[0]), sum0);
                sum0 = fmaf(hv.y, __half2float(bh0[1]), sum0);
                sum0 = fmaf(hv.z, __half2float(bh0[2]), sum0);
                sum0 = fmaf(hv.w, __half2float(bh0[3]), sum0);
                sum1 = fmaf(hv.x, __half2float(bh1[0]), sum1);
                sum1 = fmaf(hv.y, __half2float(bh1[1]), sum1);
                sum1 = fmaf(hv.z, __half2float(bh1[2]), sum1);
                sum1 = fmaf(hv.w, __half2float(bh1[3]), sum1);
            }
        } else {
            for (int r = 0; r < rank_pad; r++) {
                const float hv = s_lowrank[r];
                sum0 = fmaf(hv, __half2float(b_row0[r]), sum0);
                sum1 = fmaf(hv, __half2float(b_row1[r]), sum1);
            }
        }
        __half* y_ptr = y_inout + (size_t)tok * size_n + col0;
        if ((reinterpret_cast<uintptr_t>(y_ptr) & 3u) == 0u) {
            uint32_t y_pair = *reinterpret_cast<const uint32_t*>(y_ptr);
            __half* yh = reinterpret_cast<__half*>(&y_pair);
            yh[0] = __float2half_rn(__half2float(yh[0]) + sum0);
            yh[1] = __float2half_rn(__half2float(yh[1]) + sum1);
            *reinterpret_cast<uint32_t*>(y_ptr) = y_pair;
        } else {
            y_ptr[0] = __float2half_rn(__half2float(y_ptr[0]) + sum0);
            y_ptr[1] = __float2half_rn(__half2float(y_ptr[1]) + sum1);
        }
    } else {
        if (vec4_aligned) {
            const float4* s_lr4 = reinterpret_cast<const float4*>(s_lowrank);
            const int vec_ranks = rank_pad / 4;
            for (int r4 = 0; r4 < vec_ranks; r4++) {
                const float4 hv = s_lr4[r4];
                const uint2 bv0 = *reinterpret_cast<const uint2*>(b_row0 + r4 * 4);
                const __half* bh0 = reinterpret_cast<const __half*>(&bv0);
                sum0 = fmaf(hv.x, __half2float(bh0[0]), sum0);
                sum0 = fmaf(hv.y, __half2float(bh0[1]), sum0);
                sum0 = fmaf(hv.z, __half2float(bh0[2]), sum0);
                sum0 = fmaf(hv.w, __half2float(bh0[3]), sum0);
            }
        } else {
            for (int r = 0; r < rank_pad; r++) {
                sum0 = fmaf(s_lowrank[r], __half2float(b_row0[r]), sum0);
            }
        }
        __half* y_ptr = y_inout + (size_t)tok * size_n + col0;
        y_ptr[0] = __float2half_rn(__half2float(y_ptr[0]) + sum0);
    }
}

__global__ void lora_shrink_f16_kernel(
    const __half* __restrict__ x,
    const __half* __restrict__ lora_a,
    const int32_t* __restrict__ token_lora_ids,
    float* __restrict__ lowrank_out,
    int m,
    int size_k,
    int rank_pad,
    int num_adapters,
    int default_lora_id,
    uint64_t adapter_mask,
    bool vec8_aligned) {
    const int tok_base = (int)blockIdx.y * 4;
    if (tok_base >= m) return;
    const int tile_m = (m - tok_base < 4) ? (m - tok_base) : 4;

    const int r_base = (int)blockIdx.x * 4;
    if (r_base >= rank_pad) return;
    const int rem_r = rank_pad - r_base;

    int32_t lids[4];
    bool all_same = true;
#pragma unroll
    for (int t = 0; t < 4; t++) {
        if (t < tile_m) {
            lids[t] = (token_lora_ids != nullptr) ? token_lora_ids[tok_base + t] : default_lora_id;
            if (lids[t] != lids[0]) all_same = false;
        } else {
            lids[t] = -1;
        }
    }

    __shared__ float4 s_warp[4][8];
    const int lane = (int)threadIdx.x & 31;
    const int warp = (int)threadIdx.x >> 5;

    if (all_same) {
        const int32_t lora_id = lids[0];
        if (lora_id < 0 || lora_id >= num_adapters || ((adapter_mask >> lora_id) & 1ULL) == 0ULL) {
            return;
        }
        const __half* a_base = lora_a + ((size_t)lora_id * rank_pad + r_base) * size_k;
        float acc[4][4] = {{0.0f}};

        if (vec8_aligned && rem_r >= 4) {
            const __half* a_row0 = a_base;
            const __half* a_row1 = a_base + size_k;
            const __half* a_row2 = a_base + (size_t)2 * size_k;
            const __half* a_row3 = a_base + (size_t)3 * size_k;
            for (int k = (int)threadIdx.x * 8; k < size_k; k += 2048) {
                const uint4 av0 = *reinterpret_cast<const uint4*>(a_row0 + k);
                const uint4 av1 = *reinterpret_cast<const uint4*>(a_row1 + k);
                const uint4 av2 = *reinterpret_cast<const uint4*>(a_row2 + k);
                const uint4 av3 = *reinterpret_cast<const uint4*>(a_row3 + k);
                const __half* ah0 = reinterpret_cast<const __half*>(&av0);
                const __half* ah1 = reinterpret_cast<const __half*>(&av1);
                const __half* ah2 = reinterpret_cast<const __half*>(&av2);
                const __half* ah3 = reinterpret_cast<const __half*>(&av3);
                float af0[8], af1[8], af2[8], af3[8];
#pragma unroll
                for (int j = 0; j < 8; j++) {
                    af0[j] = __half2float(ah0[j]);
                    af1[j] = __half2float(ah1[j]);
                    af2[j] = __half2float(ah2[j]);
                    af3[j] = __half2float(ah3[j]);
                }
#pragma unroll
                for (int t = 0; t < 4; t++) {
                    if (t < tile_m) {
                        const uint4 xv = *reinterpret_cast<const uint4*>(
                            x + (size_t)(tok_base + t) * size_k + k);
                        const __half* xh = reinterpret_cast<const __half*>(&xv);
#pragma unroll
                        for (int j = 0; j < 8; j++) {
                            const float xf = __half2float(xh[j]);
                            acc[t][0] = fmaf(xf, af0[j], acc[t][0]);
                            acc[t][1] = fmaf(xf, af1[j], acc[t][1]);
                            acc[t][2] = fmaf(xf, af2[j], acc[t][2]);
                            acc[t][3] = fmaf(xf, af3[j], acc[t][3]);
                        }
                    }
                }
            }
        } else {
            for (int k = (int)threadIdx.x; k < size_k; k += 256) {
                const float a0 = __half2float(a_base[k]);
                const float a1 = (rem_r > 1) ? __half2float(a_base[(size_t)size_k + k]) : 0.0f;
                const float a2 = (rem_r > 2) ? __half2float(a_base[(size_t)2 * size_k + k]) : 0.0f;
                const float a3 = (rem_r > 3) ? __half2float(a_base[(size_t)3 * size_k + k]) : 0.0f;
#pragma unroll
                for (int t = 0; t < 4; t++) {
                    if (t < tile_m) {
                        const float xf = __half2float(x[(size_t)(tok_base + t) * size_k + k]);
                        acc[t][0] = fmaf(xf, a0, acc[t][0]);
                        if (rem_r > 1) acc[t][1] = fmaf(xf, a1, acc[t][1]);
                        if (rem_r > 2) acc[t][2] = fmaf(xf, a2, acc[t][2]);
                        if (rem_r > 3) acc[t][3] = fmaf(xf, a3, acc[t][3]);
                    }
                }
            }
        }

#pragma unroll
        for (int t = 0; t < 4; t++) {
            if (t < tile_m) {
#pragma unroll
                for (int offset = 16; offset > 0; offset >>= 1) {
                    acc[t][0] += __shfl_down_sync(0xffffffffu, acc[t][0], offset);
                    acc[t][1] += __shfl_down_sync(0xffffffffu, acc[t][1], offset);
                    acc[t][2] += __shfl_down_sync(0xffffffffu, acc[t][2], offset);
                    acc[t][3] += __shfl_down_sync(0xffffffffu, acc[t][3], offset);
                }
                if (lane == 0) {
                    s_warp[t][warp] = make_float4(acc[t][0], acc[t][1], acc[t][2], acc[t][3]);
                }
            }
        }
        __syncthreads();

        if (threadIdx.x < (unsigned int)tile_m) {
            const int t = (int)threadIdx.x;
            float4 sum = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
#pragma unroll
            for (int w = 0; w < 8; w++) {
                const float4 v = s_warp[t][w];
                sum.x += v.x;
                sum.y += v.y;
                sum.z += v.z;
                sum.w += v.w;
            }
            float* dst = lowrank_out + (size_t)(tok_base + t) * rank_pad + r_base;
            if (rem_r >= 4 && ((reinterpret_cast<uintptr_t>(dst) & 15u) == 0u)) {
                *reinterpret_cast<float4*>(dst) = sum;
            } else {
                dst[0] = sum.x;
                if (rem_r > 1) dst[1] = sum.y;
                if (rem_r > 2) dst[2] = sum.z;
                if (rem_r > 3) dst[3] = sum.w;
            }
        }
        return;
    }

    for (int t = 0; t < tile_m; t++) {
        const int32_t lora_id = lids[t];
        if (lora_id < 0 || lora_id >= num_adapters || ((adapter_mask >> lora_id) & 1ULL) == 0ULL) {
            continue;
        }
        const __half* x_row = x + (size_t)(tok_base + t) * size_k;
        const __half* a_base = lora_a + ((size_t)lora_id * rank_pad + r_base) * size_k;
        float acc0 = 0.0f, acc1 = 0.0f, acc2 = 0.0f, acc3 = 0.0f;
        if (vec8_aligned && rem_r >= 4) {
            const __half* a_row0 = a_base;
            const __half* a_row1 = a_base + size_k;
            const __half* a_row2 = a_base + (size_t)2 * size_k;
            const __half* a_row3 = a_base + (size_t)3 * size_k;
            for (int k = (int)threadIdx.x * 8; k < size_k; k += 2048) {
                const uint4 xv = *reinterpret_cast<const uint4*>(x_row + k);
                const uint4 av0 = *reinterpret_cast<const uint4*>(a_row0 + k);
                const uint4 av1 = *reinterpret_cast<const uint4*>(a_row1 + k);
                const uint4 av2 = *reinterpret_cast<const uint4*>(a_row2 + k);
                const uint4 av3 = *reinterpret_cast<const uint4*>(a_row3 + k);
                const __half* xh = reinterpret_cast<const __half*>(&xv);
                const __half* ah0 = reinterpret_cast<const __half*>(&av0);
                const __half* ah1 = reinterpret_cast<const __half*>(&av1);
                const __half* ah2 = reinterpret_cast<const __half*>(&av2);
                const __half* ah3 = reinterpret_cast<const __half*>(&av3);
#pragma unroll
                for (int j = 0; j < 8; j++) {
                    const float xf = __half2float(xh[j]);
                    acc0 = fmaf(xf, __half2float(ah0[j]), acc0);
                    acc1 = fmaf(xf, __half2float(ah1[j]), acc1);
                    acc2 = fmaf(xf, __half2float(ah2[j]), acc2);
                    acc3 = fmaf(xf, __half2float(ah3[j]), acc3);
                }
            }
        } else {
            for (int k = (int)threadIdx.x; k < size_k; k += 256) {
                const float xf = __half2float(x_row[k]);
                acc0 = fmaf(xf, __half2float(a_base[k]), acc0);
                if (rem_r > 1) acc1 = fmaf(xf, __half2float(a_base[(size_t)size_k + k]), acc1);
                if (rem_r > 2) acc2 = fmaf(xf, __half2float(a_base[(size_t)2 * size_k + k]), acc2);
                if (rem_r > 3) acc3 = fmaf(xf, __half2float(a_base[(size_t)3 * size_k + k]), acc3);
            }
        }
#pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1) {
            acc0 += __shfl_down_sync(0xffffffffu, acc0, offset);
            acc1 += __shfl_down_sync(0xffffffffu, acc1, offset);
            acc2 += __shfl_down_sync(0xffffffffu, acc2, offset);
            acc3 += __shfl_down_sync(0xffffffffu, acc3, offset);
        }
        if (lane == 0) {
            s_warp[0][warp] = make_float4(acc0, acc1, acc2, acc3);
        }
        __syncthreads();
        if (threadIdx.x == 0) {
            float4 sum = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
#pragma unroll
            for (int w = 0; w < 8; w++) {
                const float4 v = s_warp[0][w];
                sum.x += v.x;
                sum.y += v.y;
                sum.z += v.z;
                sum.w += v.w;
            }
            float* dst = lowrank_out + (size_t)(tok_base + t) * rank_pad + r_base;
            if (rem_r >= 4 && ((reinterpret_cast<uintptr_t>(dst) & 15u) == 0u)) {
                *reinterpret_cast<float4*>(dst) = sum;
            } else {
                dst[0] = sum.x;
                if (rem_r > 1) dst[1] = sum.y;
                if (rem_r > 2) dst[2] = sum.z;
                if (rem_r > 3) dst[3] = sum.w;
            }
        }
        __syncthreads();
    }
}

__global__ void lora_expand_add_f16_kernel(
    const float* __restrict__ lowrank_in,
    const __half* __restrict__ lora_b,
    const int32_t* __restrict__ token_lora_ids,
    __half* __restrict__ y_inout,
    int m,
    int size_n,
    int rank_pad,
    int num_adapters,
    int default_lora_id,
    uint64_t adapter_mask,
    bool vec4_aligned) {
    const int tok_base = (int)blockIdx.y * 8;
    if (tok_base >= m) return;
    const int tile_m = (m - tok_base < 8) ? (m - tok_base) : 8;

    int32_t lids[8];
    bool all_same = true;
#pragma unroll
    for (int t = 0; t < 8; t++) {
        if (t < tile_m) {
            lids[t] = (token_lora_ids != nullptr) ? token_lora_ids[tok_base + t] : default_lora_id;
            if (lids[t] != lids[0]) all_same = false;
        } else {
            lids[t] = -1;
        }
    }

    __shared__ alignas(16) float s_lowrank[8][kLoraMaxRankPad];

    if (all_same) {
        const int32_t lora_id = lids[0];
        if (lora_id < 0 || lora_id >= num_adapters || ((adapter_mask >> lora_id) & 1ULL) == 0ULL) {
            return;
        }
        for (int t = 0; t < tile_m; t++) {
            const float* lr_row = lowrank_in + (size_t)(tok_base + t) * rank_pad;
            for (int r = (int)threadIdx.x; r < rank_pad; r += 256) {
                s_lowrank[t][r] = lr_row[r];
            }
        }
        __syncthreads();

        const int col0 = ((int)blockIdx.x * 256 + (int)threadIdx.x) * 2;
        if (col0 >= size_n) return;

        const __half* b_row0 = lora_b + ((size_t)lora_id * size_n + col0) * rank_pad;
        float sum0[8] = {0.0f};
        float sum1[8] = {0.0f};

        if (col0 + 1 < size_n) {
            const __half* b_row1 = b_row0 + rank_pad;
            if (vec4_aligned) {
                const int vec_ranks = rank_pad / 4;
                for (int r4 = 0; r4 < vec_ranks; r4++) {
                    const uint2 bv0 = *reinterpret_cast<const uint2*>(b_row0 + r4 * 4);
                    const uint2 bv1 = *reinterpret_cast<const uint2*>(b_row1 + r4 * 4);
                    const __half* bh0 = reinterpret_cast<const __half*>(&bv0);
                    const __half* bh1 = reinterpret_cast<const __half*>(&bv1);
                    const float b00 = __half2float(bh0[0]);
                    const float b01 = __half2float(bh0[1]);
                    const float b02 = __half2float(bh0[2]);
                    const float b03 = __half2float(bh0[3]);
                    const float b10 = __half2float(bh1[0]);
                    const float b11 = __half2float(bh1[1]);
                    const float b12 = __half2float(bh1[2]);
                    const float b13 = __half2float(bh1[3]);
#pragma unroll
                    for (int t = 0; t < 8; t++) {
                        if (t < tile_m) {
                            const float4 hv =
                                reinterpret_cast<const float4*>(s_lowrank[t])[r4];
                            sum0[t] = fmaf(hv.x, b00, sum0[t]);
                            sum0[t] = fmaf(hv.y, b01, sum0[t]);
                            sum0[t] = fmaf(hv.z, b02, sum0[t]);
                            sum0[t] = fmaf(hv.w, b03, sum0[t]);
                            sum1[t] = fmaf(hv.x, b10, sum1[t]);
                            sum1[t] = fmaf(hv.y, b11, sum1[t]);
                            sum1[t] = fmaf(hv.z, b12, sum1[t]);
                            sum1[t] = fmaf(hv.w, b13, sum1[t]);
                        }
                    }
                }
            } else {
                for (int r = 0; r < rank_pad; r++) {
                    const float b0 = __half2float(b_row0[r]);
                    const float b1 = __half2float(b_row1[r]);
#pragma unroll
                    for (int t = 0; t < 8; t++) {
                        if (t < tile_m) {
                            const float hv = s_lowrank[t][r];
                            sum0[t] = fmaf(hv, b0, sum0[t]);
                            sum1[t] = fmaf(hv, b1, sum1[t]);
                        }
                    }
                }
            }
            for (int t = 0; t < tile_m; t++) {
                __half* y_ptr = y_inout + (size_t)(tok_base + t) * size_n + col0;
                if ((reinterpret_cast<uintptr_t>(y_ptr) & 3u) == 0u) {
                    uint32_t y_pair = *reinterpret_cast<const uint32_t*>(y_ptr);
                    __half* yh = reinterpret_cast<__half*>(&y_pair);
                    yh[0] = __float2half_rn(__half2float(yh[0]) + sum0[t]);
                    yh[1] = __float2half_rn(__half2float(yh[1]) + sum1[t]);
                    *reinterpret_cast<uint32_t*>(y_ptr) = y_pair;
                } else {
                    y_ptr[0] = __float2half_rn(__half2float(y_ptr[0]) + sum0[t]);
                    y_ptr[1] = __float2half_rn(__half2float(y_ptr[1]) + sum1[t]);
                }
            }
        } else {
            for (int r = 0; r < rank_pad; r++) {
                const float b0 = __half2float(b_row0[r]);
#pragma unroll
                for (int t = 0; t < 8; t++) {
                    if (t < tile_m) {
                        sum0[t] = fmaf(s_lowrank[t][r], b0, sum0[t]);
                    }
                }
            }
            for (int t = 0; t < tile_m; t++) {
                __half* y_ptr = y_inout + (size_t)(tok_base + t) * size_n + col0;
                y_ptr[0] = __float2half_rn(__half2float(y_ptr[0]) + sum0[t]);
            }
        }
        return;
    }

    const int col0 = ((int)blockIdx.x * 256 + (int)threadIdx.x) * 2;
    for (int t = 0; t < tile_m; t++) {
        const int32_t lora_id = lids[t];
        if (lora_id < 0 || lora_id >= num_adapters || ((adapter_mask >> lora_id) & 1ULL) == 0ULL) {
            continue;
        }
        const float* lr_row = lowrank_in + (size_t)(tok_base + t) * rank_pad;
        for (int r = (int)threadIdx.x; r < rank_pad; r += 256) {
            s_lowrank[0][r] = lr_row[r];
        }
        __syncthreads();
        if (col0 < size_n) {
            const __half* b_row0 = lora_b + ((size_t)lora_id * size_n + col0) * rank_pad;
            float sum0 = 0.0f;
            float sum1 = 0.0f;
            if (col0 + 1 < size_n) {
                const __half* b_row1 = b_row0 + rank_pad;
                if (vec4_aligned) {
                    const float4* s_lr4 = reinterpret_cast<const float4*>(s_lowrank[0]);
                    const int vec_ranks = rank_pad / 4;
                    for (int r4 = 0; r4 < vec_ranks; r4++) {
                        const float4 hv = s_lr4[r4];
                        const uint2 bv0 = *reinterpret_cast<const uint2*>(b_row0 + r4 * 4);
                        const uint2 bv1 = *reinterpret_cast<const uint2*>(b_row1 + r4 * 4);
                        const __half* bh0 = reinterpret_cast<const __half*>(&bv0);
                        const __half* bh1 = reinterpret_cast<const __half*>(&bv1);
                        sum0 = fmaf(hv.x, __half2float(bh0[0]), sum0);
                        sum0 = fmaf(hv.y, __half2float(bh0[1]), sum0);
                        sum0 = fmaf(hv.z, __half2float(bh0[2]), sum0);
                        sum0 = fmaf(hv.w, __half2float(bh0[3]), sum0);
                        sum1 = fmaf(hv.x, __half2float(bh1[0]), sum1);
                        sum1 = fmaf(hv.y, __half2float(bh1[1]), sum1);
                        sum1 = fmaf(hv.z, __half2float(bh1[2]), sum1);
                        sum1 = fmaf(hv.w, __half2float(bh1[3]), sum1);
                    }
                } else {
                    for (int r = 0; r < rank_pad; r++) {
                        const float hv = s_lowrank[0][r];
                        sum0 = fmaf(hv, __half2float(b_row0[r]), sum0);
                        sum1 = fmaf(hv, __half2float(b_row1[r]), sum1);
                    }
                }
                __half* y_ptr = y_inout + (size_t)(tok_base + t) * size_n + col0;
                if ((reinterpret_cast<uintptr_t>(y_ptr) & 3u) == 0u) {
                    uint32_t y_pair = *reinterpret_cast<const uint32_t*>(y_ptr);
                    __half* yh = reinterpret_cast<__half*>(&y_pair);
                    yh[0] = __float2half_rn(__half2float(yh[0]) + sum0);
                    yh[1] = __float2half_rn(__half2float(yh[1]) + sum1);
                    *reinterpret_cast<uint32_t*>(y_ptr) = y_pair;
                } else {
                    y_ptr[0] = __float2half_rn(__half2float(y_ptr[0]) + sum0);
                    y_ptr[1] = __float2half_rn(__half2float(y_ptr[1]) + sum1);
                }
            } else {
                for (int r = 0; r < rank_pad; r++) {
                    sum0 = fmaf(s_lowrank[0][r], __half2float(b_row0[r]), sum0);
                }
                __half* y_ptr = y_inout + (size_t)(tok_base + t) * size_n + col0;
                y_ptr[0] = __float2half_rn(__half2float(y_ptr[0]) + sum0);
            }
        }
        __syncthreads();
    }
}

}  // namespace

extern "C" int w34_lora_shrink_expand_f16(
    const w34_exec* ex,
    const void* x_f16,
    const void* lora_a_stacked,
    const void* lora_b_stacked,
    const int32_t* token_lora_ids,
    float* lowrank_f32_ws,
    void* y_inout_f16,
    int m,
    int size_n,
    int size_k,
    int rank_pad,
    int num_adapters,
    int default_lora_id,
    uint64_t adapter_mask) {
    if (ex == nullptr) return int(cudaErrorInvalidValue);
    if (m == 0) return int(cudaSuccess);
    if (m < 0 || m > 65535 || size_n <= 0 || size_k <= 0 || rank_pad <= 0 ||
        rank_pad > kLoraMaxRankPad || num_adapters <= 0 ||
        num_adapters > kLoraMaxAdapters) {
        return int(cudaErrorInvalidValue);
    }
    if (x_f16 == nullptr || lora_a_stacked == nullptr || lora_b_stacked == nullptr ||
        lowrank_f32_ws == nullptr || y_inout_f16 == nullptr) {
        return int(cudaErrorInvalidValue);
    }
    if ((reinterpret_cast<uintptr_t>(x_f16) & 1u) != 0u ||
        (reinterpret_cast<uintptr_t>(lora_a_stacked) & 1u) != 0u ||
        (reinterpret_cast<uintptr_t>(lora_b_stacked) & 1u) != 0u ||
        (reinterpret_cast<uintptr_t>(y_inout_f16) & 1u) != 0u ||
        (reinterpret_cast<uintptr_t>(lowrank_f32_ws) & 3u) != 0u) {
        return int(cudaErrorInvalidValue);
    }
    if (adapter_mask == 0ULL) return int(cudaSuccess);
    if (token_lora_ids == nullptr &&
        (default_lora_id < 0 || default_lora_id >= num_adapters ||
         ((adapter_mask >> default_lora_id) & 1ULL) == 0ULL)) {
        return int(cudaSuccess);
    }

    const bool vec8_aligned = ((size_k % 8) == 0) &&
                              ((reinterpret_cast<uintptr_t>(x_f16) & 15u) == 0u) &&
                              ((reinterpret_cast<uintptr_t>(lora_a_stacked) & 15u) == 0u);
    const bool vec4_aligned = ((rank_pad % 4) == 0) &&
                              ((reinterpret_cast<uintptr_t>(lora_b_stacked) & 7u) == 0u);

    cudaStream_t stream = reinterpret_cast<cudaStream_t>(ex->stream);
    const int fused_blocks = (size_n + 1023) / 1024;
    if (m <= 4 && rank_pad <= kLoraFusedMaxRankPad &&
        fused_blocks <= kLoraFusedMaxBlocks) {
        const dim3 fused_grid((unsigned int)fused_blocks, (unsigned int)m, 1);
        lora_fused_shrink_expand_512t_f16_kernel<<<fused_grid, 512, 0, stream>>>(
            static_cast<const __half*>(x_f16),
            static_cast<const __half*>(lora_a_stacked),
            static_cast<const __half*>(lora_b_stacked),
            token_lora_ids,
            static_cast<__half*>(y_inout_f16),
            size_n,
            size_k,
            rank_pad,
            num_adapters,
            default_lora_id,
            adapter_mask,
            vec8_aligned,
            vec4_aligned);
        return int(cudaGetLastError());
    }

    const int expand_blocks = (size_n + 511) / 512;
    const dim3 expand_grid((unsigned int)expand_blocks, (unsigned int)((m + 7) / 8), 1);
    const dim3 shrink_grid((unsigned int)((rank_pad + 3) / 4), (unsigned int)((m + 3) / 4), 1);

    lora_shrink_f16_kernel<<<shrink_grid, 256, 0, stream>>>(
        static_cast<const __half*>(x_f16),
        static_cast<const __half*>(lora_a_stacked),
        token_lora_ids,
        lowrank_f32_ws,
        m,
        size_k,
        rank_pad,
        num_adapters,
        default_lora_id,
        adapter_mask,
        vec8_aligned);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) return int(err);

    lora_expand_add_f16_kernel<<<expand_grid, 256, 0, stream>>>(
        lowrank_f32_ws,
        static_cast<const __half*>(lora_b_stacked),
        token_lora_ids,
        static_cast<__half*>(y_inout_f16),
        m,
        size_n,
        rank_pad,
        num_adapters,
        default_lora_id,
        adapter_mask,
        vec4_aligned);
    return int(cudaGetLastError());
}
