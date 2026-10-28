// GPU Sampler kernels for warp-34 (Greedy argmax + Top-5 logprobs in FP32).
#include "../w34_cuda.h"

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <float.h>
#include <math.h>

namespace w34_sampler {

struct ValIdx {
    float   val;
    int32_t idx;
};

__device__ __forceinline__ ValIdx better_pair(ValIdx a, ValIdx b) {
    if (b.val > a.val || (b.val == a.val && b.idx >= 0 && (a.idx < 0 || b.idx < a.idx))) {
        return b;
    }
    return a;
}

__device__ __forceinline__ ValIdx warp_reduce_max_pair(ValIdx v) {
#pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
        ValIdx o;
        o.val = __shfl_xor_sync(0xffffffffu, v.val, off);
        o.idx = __shfl_xor_sync(0xffffffffu, v.idx, off);
        v = better_pair(v, o);
    }
    return v;
}

__device__ __forceinline__ ValIdx block_reduce_max_pair(ValIdx v, float* s_val, int32_t* s_idx) {
    const int lane = threadIdx.x & 31;
    const int wid = threadIdx.x >> 5;
    v = warp_reduce_max_pair(v);
    if (lane == 0) {
        s_val[wid] = v.val;
        s_idx[wid] = v.idx;
    }
    __syncthreads();
    const int num_warps = blockDim.x >> 5;
    ValIdx wv = (threadIdx.x < num_warps) ? ValIdx{s_val[lane], s_idx[lane]} : ValIdx{-FLT_MAX, -1};
    if (wid == 0) {
        wv = warp_reduce_max_pair(wv);
        if (lane == 0) {
            s_val[0] = wv.val;
            s_idx[0] = wv.idx;
        }
    }
    __syncthreads();
    return ValIdx{s_val[0], s_idx[0]};
}

__device__ __forceinline__ float block_reduce_sum(float v, float* s_val) {
    const int lane = threadIdx.x & 31;
    const int wid = threadIdx.x >> 5;
#pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
        v += __shfl_xor_sync(0xffffffffu, v, off);
    }
    if (lane == 0) s_val[wid] = v;
    __syncthreads();
    const int num_warps = blockDim.x >> 5;
    v = (threadIdx.x < num_warps) ? s_val[lane] : 0.0f;
    if (wid == 0) {
#pragma unroll
        for (int off = 16; off > 0; off >>= 1) {
            v += __shfl_xor_sync(0xffffffffu, v, off);
        }
        if (lane == 0) s_val[0] = v;
    }
    __syncthreads();
    return s_val[0];
}

// Grid: num_rows, Block: 256 threads
__global__ void sampler_greedy_top5_f16_kernel(
    const __half* __restrict__ logits_f16,
    int32_t* __restrict__ out_greedy_ids,
    int32_t* __restrict__ out_top5_ids,
    float* __restrict__ out_top5_logprobs,
    int local_vocab_size,
    int vocab_offset) {
    const int row = blockIdx.x;
    const __half* row_logits = logits_f16 + (size_t)row * local_vocab_size;

    __shared__ float s_val[32];
    __shared__ int32_t s_idx[32];

    const bool want_top5 = (out_top5_ids != nullptr && out_top5_logprobs != nullptr);
    const int k_iters = want_top5 ? 5 : 1;

    int32_t picked_ids[5] = {-1, -1, -1, -1, -1};
    float picked_vals[5] = {-FLT_MAX, -FLT_MAX, -FLT_MAX, -FLT_MAX, -FLT_MAX};

    for (int k = 0; k < k_iters; k++) {
        ValIdx local_best = {-FLT_MAX, -1};
        for (int i = threadIdx.x; i < local_vocab_size; i += blockDim.x) {
            int32_t gid = vocab_offset + i;
            bool already = false;
#pragma unroll
            for (int p = 0; p < 5; p++) {
                if (p < k && picked_ids[p] == gid) already = true;
            }
            if (!already) {
                float v = __half2float(row_logits[i]);
                local_best = better_pair(local_best, ValIdx{v, gid});
            }
        }
        ValIdx winner = block_reduce_max_pair(local_best, s_val, s_idx);
        picked_ids[k] = winner.idx;
        picked_vals[k] = winner.val;
    }

    if (threadIdx.x == 0) {
        out_greedy_ids[row] = picked_ids[0];
        if (!want_top5 && out_top5_logprobs != nullptr) {
            out_top5_logprobs[row] = picked_vals[0];
        }
    }

    if (want_top5) {
        const float max_logit = picked_vals[0];
        float sum_exp = 0.0f;
        for (int i = threadIdx.x; i < local_vocab_size; i += blockDim.x) {
            sum_exp += expf(__half2float(row_logits[i]) - max_logit);
        }
        sum_exp = block_reduce_sum(sum_exp, s_val);
        const float log_z = max_logit + logf(fmaxf(sum_exp, 1e-20f));
        if (threadIdx.x < 5) {
            out_top5_ids[row * 5 + threadIdx.x] = picked_ids[threadIdx.x];
            out_top5_logprobs[row * 5 + threadIdx.x] = picked_vals[threadIdx.x] - log_z;
        }
    }
}

}  // namespace w34_sampler

extern "C" int w34_sampler_greedy_top5_f16(
    const w34_exec* ex,
    const void* logits_f16,
    int32_t* out_greedy_ids,
    int32_t* out_top5_ids,
    float* out_top5_logprobs,
    int num_rows,
    int local_vocab_size,
    int vocab_offset) {
    if (ex == nullptr || logits_f16 == nullptr || out_greedy_ids == nullptr ||
        num_rows <= 0 || local_vocab_size <= 0) {
        return (int)cudaErrorInvalidValue;
    }
    w34_sampler::sampler_greedy_top5_f16_kernel<<<num_rows, 1024, 0, (cudaStream_t)ex->stream>>>(
        (const __half*)logits_f16, out_greedy_ids, out_top5_ids, out_top5_logprobs,
        local_vocab_size, vocab_offset);
    return (int)cudaGetLastError();
}
