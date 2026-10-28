// Canonical token-wise speculative rejection sampling.
// Oracle: vLLM v1/worker/gpu/spec_decode/rejection_sampler_utils.py.
// Probabilities are already processed (temperature/penalties/grammar/top-k/p).
// Uniform draws are supplied explicitly; kernels never mutate request RNG state.
#include "../w34_cuda.h"
#include <cuda_runtime.h>
#include <math.h>

namespace {
constexpr int THREADS = 256;
constexpr int MAX_DRAFT = 16;

__global__ void rejection_greedy_kernel(
    const int32_t* drafts, const int32_t* targets, const uint8_t* valid,
    int32_t* emitted, int32_t* accepted, int32_t* status, int k) {
    const int req = blockIdx.x;
    // One thread per request; token IDs remain INT32 throughout.
    const int64_t out_base = (int64_t)req * (k + 1);
    accepted[req] = -1;
    status[req] = 0;
    for (int i = 0; i <= k; ++i) emitted[out_base + i] = -1;
    if (valid && !valid[req]) return;
    for (int i = 0; i <= k; ++i) {
        if (targets[out_base + i] < 0 || (i < k && drafts[(int64_t)req * k + i] < 0)) {
            status[req] = 1;
            return;
        }
    }
    int a = 0;
    while (a < k && drafts[(int64_t)req * k + a] == targets[out_base + a]) ++a;
    for (int i = 0; i < a; ++i) emitted[out_base + i] = drafts[(int64_t)req * k + i];
    emitted[out_base + a] = targets[out_base + a];
    accepted[req] = a;
}

__device__ double block_sum(double x, double* scratch) {
    scratch[threadIdx.x] = x;
    __syncthreads();
    for (int stride = THREADS / 2; stride; stride >>= 1) {
        if (threadIdx.x < stride) scratch[threadIdx.x] += scratch[threadIdx.x + stride];
        __syncthreads();
    }
    const double sum = scratch[0];
    __syncthreads();
    return sum;
}

__device__ bool valid_draw(double u) {
    return isfinite(u) && u >= 0.0 && u < 1.0;
}

__global__ void rejection_stochastic_kernel(
    const int32_t* drafts, const float* target, const float* proposal,
    const double* uniforms, const double* recovery_uniforms, const uint8_t* valid,
    int32_t* emitted, int32_t* accepted, int32_t* status, int k, int vocab) {
    const int req = blockIdx.x;
    const int tid = threadIdx.x;
    const int64_t out_base = (int64_t)req * (k + 1);
    const int64_t draft_base = (int64_t)req * k;
    __shared__ double scratch[THREADS];
    __shared__ double p_sum[MAX_DRAFT + 1];
    __shared__ double q_sum[MAX_DRAFT];
    __shared__ int error;
    __shared__ int a;
    if (tid == 0) {
        accepted[req] = -1;
        status[req] = 0;
        error = 0;
    }
    if (tid <= k) emitted[out_base + tid] = -1;
    // Mask is uniform across the entire CTA. Invalid requests read no inputs.
    if (valid && !valid[req]) return;
    __syncthreads();

    // Validate all rows, not only the accepted prefix, before writing a result.
    for (int i = 0; i <= k; ++i) {
        double ps = 0.0, qs = 0.0;
        for (int col = tid; col < vocab; col += THREADS) {
            const float p = target[(out_base + i) * vocab + col];
            if (!isfinite(p) || p < 0.0f) atomicExch(&error, 1);
            ps += (double)p;
            if (i < k) {
                const float q = proposal[(draft_base + i) * vocab + col];
                if (!isfinite(q) || q < 0.0f) atomicExch(&error, 1);
                qs += (double)q;
            }
        }
        ps = block_sum(ps, scratch);
        qs = block_sum(qs, scratch);
        if (tid == 0) {
            p_sum[i] = ps;
            if (!isfinite(ps) || fabs(ps - 1.0) > 1e-5) atomicExch(&error, 1);
            if (i < k) {
                q_sum[i] = qs;
                if (!isfinite(qs) || fabs(qs - 1.0) > 1e-5) atomicExch(&error, 1);
            }
        }
    }
    if (tid == 0) {
        if (!valid_draw(recovery_uniforms[req])) atomicExch(&error, 1);
        for (int i = 0; i < k; ++i) {
            const int id = drafts[draft_base + i];
            if (id < 0 || id >= vocab || !valid_draw(uniforms[draft_base + i])) {
                atomicExch(&error, 1);
            } else if (!(proposal[(draft_base + i) * vocab + id] > 0.0f)) {
                atomicExch(&error, 1);
            }
        }
    }
    __syncthreads();
    if (error) {
        if (tid == 0) status[req] = 1;
        return;
    }
    if (tid == 0) {
        a = 0;
        for (; a < k; ++a) {
            const int id = drafts[draft_base + a];
            const double p = target[(out_base + a) * vocab + id] / p_sum[a];
            const double q = proposal[(draft_base + a) * vocab + id] / q_sum[a];
            if (!(uniforms[draft_base + a] < fmin(1.0, p / q))) break;
        }
    }
    __syncthreads();

    // Contiguous vocabulary partitions preserve token-ID CDF ordering. Each
    // thread sums its partition; only the selected partition is scanned again.
    const int start = (int)((int64_t)vocab * tid / THREADS);
    const int end = (int)((int64_t)vocab * (tid + 1) / THREADS);
    double mass = 0.0;
    for (int col = start; col < end; ++col) {
        const double p = target[(out_base + a) * vocab + col] / p_sum[a];
        const double q = a < k ? proposal[(draft_base + a) * vocab + col] / q_sum[a] : 0.0;
        mass += fmax(p - q, 0.0);
    }
    scratch[tid] = mass;
    __syncthreads();
    if (tid == 0) {
        double sum = 0.0;
        for (int t = 0; t < THREADS; ++t) {
            sum += scratch[t];
            scratch[t] = sum;
        }
        if (!(sum > 0.0) || !isfinite(sum)) error = 1;
    }
    __syncthreads();
    if (error) {
        if (tid == 0) status[req] = 1;
        return;
    }
    const double total = scratch[THREADS - 1];
    // A rounded product at the upper endpoint must still select the last
    // positive-mass bin (the host categorical reference has the same fallback).
    const double threshold = fmin(recovery_uniforms[req] * total, nextafter(total, 0.0));
    double before = tid == 0 ? 0.0 : scratch[tid - 1];
    if (threshold >= before && threshold < scratch[tid]) {
        int last_positive = -1;
        for (int col = start; col < end; ++col) {
            const double p = target[(out_base + a) * vocab + col] / p_sum[a];
            const double q = a < k ? proposal[(draft_base + a) * vocab + col] / q_sum[a] : 0.0;
            const double weight = fmax(p - q, 0.0);
            if (weight > 0.0) {
                last_positive = col;
                before += weight;
                if (threshold < before) break;
            }
        }
        emitted[out_base + a] = last_positive;
    }
    if (tid < a) emitted[out_base + tid] = drafts[draft_base + tid];
    if (tid == 0) accepted[req] = a;
}
} // namespace

extern "C" int w34_rejection_greedy(
    const w34_exec* ex, const int32_t* drafts, const int32_t* targets,
    const uint8_t* valid, int32_t* emitted, int32_t* accepted, int32_t* status,
    int num_reqs, int k) {
    if (!ex || !targets || !emitted || !accepted || !status || num_reqs < 0 ||
        k < 0 || k > MAX_DRAFT || (k > 0 && !drafts)) return (int)cudaErrorInvalidValue;
    if (!num_reqs) return 0;
    rejection_greedy_kernel<<<num_reqs, 1, 0, (cudaStream_t)ex->stream>>>(
        drafts, targets, valid, emitted, accepted, status, k);
    return (int)cudaGetLastError();
}

extern "C" int w34_rejection_stochastic(
    const w34_exec* ex, const int32_t* drafts, const float* target, const float* proposal,
    const double* uniforms, const double* recovery_uniforms, const uint8_t* valid,
    int32_t* emitted, int32_t* accepted, int32_t* status, int num_reqs, int k, int vocab) {
    if (!ex || !target || !recovery_uniforms || !emitted || !accepted || !status ||
        num_reqs < 0 || k < 0 || k > MAX_DRAFT || vocab <= 0 ||
        (k > 0 && (!drafts || !proposal || !uniforms))) return (int)cudaErrorInvalidValue;
    if (!num_reqs) return 0;
    rejection_stochastic_kernel<<<num_reqs, THREADS, 0, (cudaStream_t)ex->stream>>>(
        drafts, target, proposal, uniforms, recovery_uniforms, valid,
        emitted, accepted, status, k, vocab);
    return (int)cudaGetLastError();
}
