#pragma once
// Shared TQ4 / Q8_0 KV block layouts and warp/block reduction + WHT-256 helpers.
#include "../w34_cuda.h"
#include "tq4_codebook.cuh"

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <float.h>
#include <math.h>
#include <stdint.h>

namespace w34_attn {

// block_tq4_0: 66 bytes per 128 elements (2 blocks = 132 B for head_dim = 256).
#pragma pack(push, 1)
struct block_tq4_0 {
    __half  d;
    uint8_t qs[64];
};
static_assert(sizeof(block_tq4_0) == 66, "block_tq4_0 must be 66 bytes");

// block_q8_0_64: 68 bytes per 64 elements (4 blocks = 272 B for head_dim = 256).
struct block_q8_0_64 {
    float  d;
    int8_t qs[64];
};
static_assert(sizeof(block_q8_0_64) == 68, "block_q8_0_64 must be 68 bytes");
#pragma pack(pop)

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

// 256-point Walsh-Hadamard transform on a single warp holding v[0..7] at
// col = lane + 32 * j (j = 0..7). Stages h = 1..16 use intra-warp shuffles;
// stages h = 32, 64, 128 operate directly on each thread's register pair
// v[j] and v[j ^ (h >> 5)] with zero shared memory and zero __syncthreads().
// Matches warp3090's wht256 rounding order to 0 ULP.
__device__ __forceinline__ void wht256_warp(float (&v)[8], int lane) {
#pragma unroll
    for (int j = 0; j < 8; ++j) {
        float val = v[j];
#pragma unroll
        for (int h = 1; h < 32; h <<= 1) {
            float o = __shfl_xor_sync(0xffffffffu, val, h);
            val = (lane & h) ? (o - val) : (val + o);
        }
        v[j] = val;
    }
#pragma unroll
    for (int j = 0; j < 8; j += 2) {
        float a = v[j], b = v[j + 1];
        v[j] = a + b;
        v[j + 1] = a - b;
    }
#pragma unroll
    for (int g = 0; g < 8; g += 4) {
#pragma unroll
        for (int i = 0; i < 2; ++i) {
            float a = v[g + i], b = v[g + i + 2];
            v[g + i] = a + b;
            v[g + i + 2] = a - b;
        }
    }
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        float a = v[i], b = v[i + 4];
        v[i] = (a + b) * 0.0625f;
        v[i + 4] = (a - b) * 0.0625f;
    }
}

__device__ __forceinline__ float block_reduce_sum_128(float v, float* s_warp) {
    const int lane = threadIdx.x & 31;
    const int wid = threadIdx.x >> 5;
    v = warp_reduce_sum(v);
    if (lane == 0) s_warp[wid] = v;
    __syncthreads();
    v = (threadIdx.x < (blockDim.x >> 5)) ? s_warp[lane] : 0.0f;
    if (wid == 0) {
        v = warp_reduce_sum(v);
        if (lane == 0) s_warp[0] = v;
    }
    __syncthreads();
    return s_warp[0];
}

__device__ __forceinline__ float block_reduce_max_128(float v, float* s_warp) {
    const int lane = threadIdx.x & 31;
    const int wid = threadIdx.x >> 5;
#pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
        v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, off));
    }
    if (lane == 0) s_warp[wid] = v;
    __syncthreads();
    v = (threadIdx.x < (blockDim.x >> 5)) ? s_warp[lane] : -FLT_MAX;
    if (wid == 0) {
#pragma unroll
        for (int off = 16; off > 0; off >>= 1) {
            v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, off));
        }
        if (lane == 0) s_warp[0] = v;
    }
    __syncthreads();
    return s_warp[0];
}

}  // namespace w34_attn
