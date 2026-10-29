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

union HalfVec8 {
    uint4 u;
    __half h[8];
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

__device__ __forceinline__ double block_reduce_sum_f64(double v, double* s_val) {
    const int lane = threadIdx.x & 31;
    const int wid = threadIdx.x >> 5;
#pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
        v += __shfl_xor_sync(0xffffffffu, v, off);
    }
    if (lane == 0) s_val[wid] = v;
    __syncthreads();
    const int num_warps = blockDim.x >> 5;
    v = (threadIdx.x < num_warps) ? s_val[lane] : 0.0;
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

__device__ __forceinline__ uint32_t block_reduce_sum_u32(uint32_t v, uint32_t* s_val) {
    const int lane = threadIdx.x & 31;
    const int wid = threadIdx.x >> 5;
#pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
        v += __shfl_xor_sync(0xffffffffu, v, off);
    }
    if (lane == 0) s_val[wid] = v;
    __syncthreads();
    const int num_warps = blockDim.x >> 5;
    v = (threadIdx.x < num_warps) ? s_val[lane] : 0u;
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

static_assert(sizeof(w34_sampler_row_cfg) == 64, "w34_sampler_row_cfg ABI mismatch");
static_assert(sizeof(w34_sampler_sparse_entry) == 16, "w34_sampler_sparse_entry ABI mismatch");
static_assert(sizeof(w34_sampler_row_out) == 184, "w34_sampler_row_out ABI mismatch");

// Maps finite float -> uint32_t >= 0x00800000u strictly preserving `<`, and masked (-inf) -> 0u.
// Normalizes -0.0f to +0.0f so zero ties break by token ID ascending.
__device__ __forceinline__ uint32_t float_to_ord_u32(float x) {
    if (!(x >= -FLT_MAX)) return 0u;
    if (x == 0.0f) x = 0.0f;
    uint32_t u = __float_as_uint(x);
    return (u & 0x80000000u) ? (~u) : (u ^ 0x80000000u);
}

__device__ __forceinline__ uint64_t make_rank_key(float x, int32_t tid) {
    return (uint64_t(float_to_ord_u32(x)) << 32) | uint64_t(~uint32_t(tid));
}

__global__ void sampler_scatter_sparse_kernel(
    const w34_sampler_row_cfg* __restrict__ row_cfgs,
    const w34_sampler_sparse_entry* __restrict__ entries,
    uint32_t* __restrict__ work_counts_u32,
    float* __restrict__ work_bias_f32,
    int vocab_size) {
    const int row = blockIdx.x;
    const w34_sampler_row_cfg cfg = row_cfgs[row];
    if (cfg.sparse_offset < 0 || cfg.sparse_count <= 0) return;
    const size_t row_base = (size_t)row * vocab_size;
    for (int i = threadIdx.x; i < cfg.sparse_count; i += blockDim.x) {
        const w34_sampler_sparse_entry e = entries[cfg.sparse_offset + i];
        if (e.token_id >= 0 && e.token_id < vocab_size) {
            const uint32_t packed = ((e.banned ? 1u : 0u) << 31) |
                                    ((e.in_prompt ? 1u : 0u) << 30) |
                                    (e.output_count & 0x3fffffffu);
            work_counts_u32[row_base + e.token_id] = packed;
            work_bias_f32[row_base + e.token_id] = e.logit_bias;
        }
    }
}

__device__ uint64_t radix_kth_largest_key(
    const float* __restrict__ row_logit,
    int vocab_size,
    uint64_t min_key,
    uint32_t k_target,
    uint32_t* s_hist,
    uint64_t* s_prefix,
    uint32_t* s_rem_k) {
    if (threadIdx.x == 0) {
        *s_prefix = 0ull;
        *s_rem_k = k_target;
    }
    __syncthreads();

    for (int pass = 0; pass < 8; ++pass) {
        if (threadIdx.x < 256) s_hist[threadIdx.x] = 0u;
        __syncthreads();

        const uint64_t pref = *s_prefix;
        const int shift = 56 - pass * 8;
        const int hi_shift = 64 - pass * 8;
        for (int tid = threadIdx.x; tid < vocab_size; tid += blockDim.x) {
            const uint64_t key = make_rank_key(row_logit[tid], tid);
            if (key >= min_key && (pass == 0 || (key >> hi_shift) == (pref >> hi_shift))) {
                atomicAdd(&s_hist[(key >> shift) & 0xffu], 1u);
            }
        }
        __syncthreads();

        if (threadIdx.x == 0) {
            uint32_t rem = *s_rem_k;
            uint32_t chosen = 0u;
            uint32_t last_nonempty = 0u;
            bool found = false;
            for (int b = 255; b >= 0; --b) {
                const uint32_t cnt = s_hist[b];
                if (cnt > 0u) {
                    last_nonempty = (uint32_t)b;
                    if (rem <= cnt) {
                        chosen = (uint32_t)b;
                        found = true;
                        break;
                    }
                    rem -= cnt;
                }
            }
            if (!found) {
                chosen = last_nonempty;
                rem = 1u;
            }
            *s_prefix = pref | (uint64_t(chosen) << shift);
            *s_rem_k = rem;
        }
        __syncthreads();
    }
    return *s_prefix;
}

__device__ uint64_t radix_descending_mass_key(
    const float* __restrict__ row_logit,
    const float* __restrict__ row_prob,
    int vocab_size,
    uint64_t min_key,
    bool require_positive_prob,
    double target_mass,
    double* s_mass,
    uint32_t* s_hist,
    uint64_t* s_prefix,
    double* s_rem_mass) {
    if (threadIdx.x == 0) {
        *s_prefix = 0ull;
        *s_rem_mass = target_mass;
    }
    __syncthreads();

    for (int pass = 0; pass < 8; ++pass) {
        if (threadIdx.x < 256) {
            s_mass[threadIdx.x] = 0.0;
            s_hist[threadIdx.x] = 0u;
        }
        __syncthreads();

        const uint64_t pref = *s_prefix;
        const int shift = 56 - pass * 8;
        const int hi_shift = 64 - pass * 8;
        for (int tid = threadIdx.x; tid < vocab_size; tid += blockDim.x) {
            const float p = row_prob[tid];
            if (require_positive_prob && !(p > 0.0f)) continue;
            const uint64_t key = make_rank_key(row_logit[tid], tid);
            if (key >= min_key && (pass == 0 || (key >> hi_shift) == (pref >> hi_shift))) {
                const uint32_t b = (uint32_t)((key >> shift) & 0xffu);
                atomicAdd(&s_mass[b], (double)p);
                atomicAdd(&s_hist[b], 1u);
            }
        }
        __syncthreads();

        if (threadIdx.x == 0) {
            double rem = *s_rem_mass;
            uint32_t chosen = 0u;
            uint32_t last_nonempty = 0u;
            bool found = false;
            for (int b = 255; b >= 0; --b) {
                if (s_hist[b] > 0u) {
                    last_nonempty = (uint32_t)b;
                    const double m = s_mass[b];
                    if (rem <= m) {
                        chosen = (uint32_t)b;
                        found = true;
                        break;
                    }
                    rem -= m;
                }
            }
            if (!found) {
                chosen = last_nonempty;
                rem = 0.0;
            }
            *s_prefix = pref | (uint64_t(chosen) << shift);
            *s_rem_mass = rem;
        }
        __syncthreads();
    }
    return *s_prefix;
}

__device__ int32_t radix_ascending_tid_cdf(
    const float* __restrict__ row_logit,
    const float* __restrict__ row_prob,
    int vocab_size,
    uint64_t min_key,
    int32_t exclude_tid,
    double target_mass,
    double* s_mass,
    uint32_t* s_hist,
    uint64_t* s_prefix,
    double* s_rem_mass) {
    if (threadIdx.x == 0) {
        *s_prefix = 0ull;
        *s_rem_mass = target_mass;
    }
    __syncthreads();

    for (int pass = 0; pass < 4; ++pass) {
        if (threadIdx.x < 256) {
            s_mass[threadIdx.x] = 0.0;
            s_hist[threadIdx.x] = 0u;
        }
        __syncthreads();

        const uint32_t pref = (uint32_t)(*s_prefix);
        const int shift = 24 - pass * 8;
        const int hi_shift = 32 - pass * 8;
        for (int tid = threadIdx.x; tid < vocab_size; tid += blockDim.x) {
            if (tid == exclude_tid) continue;
            const float p = row_prob[tid];
            if (!(p > 0.0f)) continue;
            const uint64_t key = make_rank_key(row_logit[tid], tid);
            if (key < min_key) continue;
            const uint32_t u_tid = (uint32_t)tid;
            if (pass == 0 || (u_tid >> hi_shift) == (pref >> hi_shift)) {
                const uint32_t b = (u_tid >> shift) & 0xffu;
                atomicAdd(&s_mass[b], (double)p);
                atomicAdd(&s_hist[b], 1u);
            }
        }
        __syncthreads();

        if (threadIdx.x == 0) {
            double rem = *s_rem_mass;
            uint32_t chosen = 0u;
            uint32_t last_nonempty = 0u;
            bool found = false;
            for (int b = 0; b < 256; ++b) {
                if (s_hist[b] > 0u) {
                    last_nonempty = (uint32_t)b;
                    const double m = s_mass[b];
                    if (rem < m) {
                        chosen = (uint32_t)b;
                        found = true;
                        break;
                    }
                    rem -= m;
                }
            }
            if (!found) {
                chosen = last_nonempty;
                rem = 0.0;
            }
            *s_prefix = uint64_t(pref | (chosen << shift));
            *s_rem_mass = rem;
        }
        __syncthreads();
    }
    return (int32_t)(uint32_t)(*s_prefix);
}

// Grid: num_rows, Block: 1024 threads
__global__ void __launch_bounds__(1024, 1) sampler_step_f16_kernel(
    const __half* __restrict__ logits_f16,
    const w34_sampler_row_cfg* __restrict__ row_cfgs,
    const uint32_t* __restrict__ bitmask_u32,
    const uint32_t* __restrict__ work_counts_u32,
    const float* __restrict__ work_bias_f32,
    float* __restrict__ work_logit_all,
    float* __restrict__ work_prob_all,
    w34_sampler_row_out* __restrict__ row_outs,
    int vocab_size) {
    const int row = blockIdx.x;
    const w34_sampler_row_cfg cfg = row_cfgs[row];
    const size_t row_base = (size_t)row * vocab_size;
    const __half* row_logits = logits_f16 + row_base;
    const uint32_t* row_counts = (work_counts_u32 != nullptr) ? (work_counts_u32 + row_base) : nullptr;
    const float* row_bias = (work_bias_f32 != nullptr) ? (work_bias_f32 + row_base) : nullptr;
    const uint32_t* row_mask = (cfg.has_bitmask != 0 && cfg.row_bitmask_word_offset >= 0 && bitmask_u32 != nullptr)
                                   ? (bitmask_u32 + (size_t)cfg.row_bitmask_word_offset)
                                   : nullptr;
    float* row_logit = work_logit_all + row_base;
    float* row_prob = work_prob_all + row_base;

    __shared__ double s_mass[256];
    __shared__ uint32_t s_hist[256];
    __shared__ float s_val[32];
    __shared__ int32_t s_idx[32];
    __shared__ uint32_t s_u32[32];
    __shared__ double s_f64[32];
    __shared__ uint64_t s_prefix;
    __shared__ uint32_t s_rem_k;
    __shared__ double s_rem_mass;

    // 1. Check whether soft EOS/stop bans leave at least one allowed token.
    bool honor_bans = (row_counts != nullptr && cfg.sparse_count > 0);
    if (honor_bans) {
        uint32_t unbanned_cnt = 0u;
        for (int tid = threadIdx.x; tid < vocab_size; tid += blockDim.x) {
            const bool bit_ok = (row_mask == nullptr) ||
                                ((row_mask[tid >> 5] & (1u << (tid & 31))) != 0u);
            const bool is_banned = (row_counts[tid] >> 31) != 0u;
            if (bit_ok && !is_banned) unbanned_cnt += 1u;
        }
        honor_bans = (block_reduce_sum_u32(unbanned_cnt, s_u32) > 0u);
    }

    // 2. Optional raw/biased logprobs + top_logprobs (0..=20)
    float log_z = 0.0f;
    const bool want_logprobs = (cfg.top_logprobs_k >= 0);
    const int k_lp = want_logprobs
                         ? (cfg.top_logprobs_k < W34_MAX_TOP_LOGPROBS
                                ? cfg.top_logprobs_k
                                : W34_MAX_TOP_LOGPROBS)
                         : 0;
    const int actual_lp = (k_lp < vocab_size) ? k_lp : vocab_size;
    int32_t picked_lp_ids[W34_MAX_TOP_LOGPROBS];
    float picked_lp_vals[W34_MAX_TOP_LOGPROBS];
    if (want_logprobs) {
        const int iters = (actual_lp > 1) ? actual_lp : 1;
        float raw_max = -FLT_MAX;
        for (int k = 0; k < iters; ++k) {
            ValIdx best = {-FLT_MAX, -1};
            for (int tid = threadIdx.x; tid < vocab_size; tid += blockDim.x) {
                bool dup = false;
                for (int p = 0; p < k; ++p) {
                    if (picked_lp_ids[p] == tid) {
                        dup = true;
                        break;
                    }
                }
                if (!dup) {
                    const float b = (row_bias != nullptr) ? row_bias[tid] : 0.0f;
                    const float v = __half2float(row_logits[tid]) + b;
                    best = better_pair(best, ValIdx{v, tid});
                }
            }
            ValIdx win = block_reduce_max_pair(best, s_val, s_idx);
            if (k == 0) raw_max = win.val;
            if (k < actual_lp) {
                picked_lp_ids[k] = win.idx;
                picked_lp_vals[k] = win.val;
            }
        }
        double sum_exp_raw = 0.0;
        for (int tid = threadIdx.x; tid < vocab_size; tid += blockDim.x) {
            const float b = (row_bias != nullptr) ? row_bias[tid] : 0.0f;
            const float v = __half2float(row_logits[tid]) + b;
            sum_exp_raw += exp((double)(v - raw_max));
        }
        sum_exp_raw = block_reduce_sum_f64(sum_exp_raw, s_f64);
        log_z = raw_max + (float)log(fmax(sum_exp_raw, 1e-30));
    }

    // 3. Pass 0: bitmask, bans, bias, penalties, temperature
    const bool is_greedy_policy = (cfg.temperature <= 1e-6f || cfg.top_k == 1);
    const float inv_temp = is_greedy_policy ? 1.0f : (1.0f / cfg.temperature);
    const bool has_rep = fabsf(cfg.repetition_penalty - 1.0f) > 1e-6f;
    uint32_t local_valid = 0u;
    uint32_t local_nonfinite = 0u;
    ValIdx local_best = {-FLT_MAX, -1};

    for (int tid = threadIdx.x; tid < vocab_size; tid += blockDim.x) {
        bool allowed = (row_mask == nullptr) ||
                       ((row_mask[tid >> 5] & (1u << (tid & 31))) != 0u);
        uint32_t packed_cnt = (row_counts != nullptr) ? row_counts[tid] : 0u;
        if (allowed && honor_bans && ((packed_cnt >> 31) != 0u)) {
            allowed = false;
        }
        if (!allowed) {
            row_logit[tid] = -INFINITY;
            row_prob[tid] = 0.0f;
            continue;
        }
        const float raw = __half2float(row_logits[tid]);
        if (!isfinite(raw)) {
            local_nonfinite = 1u;
            row_logit[tid] = -INFINITY;
            row_prob[tid] = 0.0f;
            continue;
        }
        float l = raw + ((row_bias != nullptr) ? row_bias[tid] : 0.0f);
        const bool in_prompt = ((packed_cnt >> 30) & 1u) != 0u;
        const uint32_t out_cnt = packed_cnt & 0x3fffffffu;
        if ((in_prompt || out_cnt > 0u) && has_rep) {
            l = (l > 0.0f) ? (l / cfg.repetition_penalty) : (l * cfg.repetition_penalty);
        }
        if (out_cnt > 0u) {
            l -= cfg.frequency_penalty * (float)out_cnt;
            l -= cfg.presence_penalty;
        }
        if (!is_greedy_policy) {
            l *= inv_temp;
        }
        if (!isfinite(l)) {
            local_nonfinite = 1u;
            row_logit[tid] = -INFINITY;
            row_prob[tid] = 0.0f;
            continue;
        }
        row_logit[tid] = l;
        local_valid += 1u;
        local_best = better_pair(local_best, ValIdx{l, tid});
    }

    const uint32_t valid_count = block_reduce_sum_u32(local_valid, s_u32);
    const uint32_t any_nonfinite = block_reduce_sum_u32(local_nonfinite, s_u32);
    const ValIdx winner = block_reduce_max_pair(local_best, s_val, s_idx);

    if (any_nonfinite > 0u || valid_count == 0u) {
        if (threadIdx.x == 0) {
            w34_sampler_row_out out = {};
            out.sampled_id = -1;
            out.draft_accepted = 0;
            out.status = (any_nonfinite > 0u) ? 2 : 1;
            for (int j = 0; j < W34_MAX_TOP_LOGPROBS; ++j) {
                out.top_ids[j] = -1;
                out.top_logprobs[j] = -INFINITY;
            }
            row_outs[row] = out;
        }
        return;
    }

    int32_t sampled_id = winner.idx;
    int32_t draft_accepted = 0;
    float draft_prob = 0.0f;

    if (is_greedy_policy || valid_count == 1u) {
        if (cfg.draft_token_id >= 0) {
            const bool match = (winner.idx == cfg.draft_token_id);
            draft_prob = match ? 1.0f : 0.0f;
            draft_accepted = match ? 1 : 0;
            sampled_id = winner.idx;
        } else {
            sampled_id = winner.idx;
        }
    } else {
        // Stage A: top_k
        uint64_t min_key = (uint64_t(1u) << 32);
        if (cfg.top_k > 0 && (uint32_t)cfg.top_k < valid_count) {
            min_key = radix_kth_largest_key(
                row_logit, vocab_size, min_key, (uint32_t)cfg.top_k, s_hist, &s_prefix, &s_rem_k);
        }

        // Stage B: softmax over K >= min_key
        const float max_l = winner.val;
        double sum_p_d = 0.0;
        for (int tid = threadIdx.x; tid < vocab_size; tid += blockDim.x) {
            const uint64_t key = make_rank_key(row_logit[tid], tid);
            if (key >= min_key) {
                const float w = expf(row_logit[tid] - max_l);
                row_prob[tid] = w;
                sum_p_d += (double)w;
            } else {
                row_prob[tid] = 0.0f;
            }
        }
        const float sum_p = (float)block_reduce_sum_f64(sum_p_d, s_f64);
        if (sum_p > 0.0f) {
            const float inv_sum = 1.0f / sum_p;
            for (int tid = threadIdx.x; tid < vocab_size; tid += blockDim.x) {
                if (row_prob[tid] > 0.0f) row_prob[tid] *= inv_sum;
            }
        }
        __syncthreads();

        // Stage B2: min_p relative to top-1 probability
        if (cfg.min_p > 0.0f && sum_p > 0.0f) {
            const float thresh = (1.0f / sum_p) * cfg.min_p;
            double renorm_d = 0.0;
            for (int tid = threadIdx.x; tid < vocab_size; tid += blockDim.x) {
                const uint64_t key = make_rank_key(row_logit[tid], tid);
                if (key >= min_key) {
                    const float p = row_prob[tid];
                    if (tid == winner.idx || p >= thresh) {
                        renorm_d += (double)p;
                    } else {
                        row_logit[tid] = -INFINITY;
                        row_prob[tid] = 0.0f;
                    }
                }
            }
            const float renorm = (float)block_reduce_sum_f64(renorm_d, s_f64);
            if (renorm > 0.0f) {
                const float inv_renorm = 1.0f / renorm;
                for (int tid = threadIdx.x; tid < vocab_size; tid += blockDim.x) {
                    if (row_prob[tid] > 0.0f) row_prob[tid] *= inv_renorm;
                }
            }
            __syncthreads();
        }

        // Stage C: top_p nucleus cutoff
        uint32_t surv_local = 0u;
        for (int tid = threadIdx.x; tid < vocab_size; tid += blockDim.x) {
            if (make_rank_key(row_logit[tid], tid) >= min_key) surv_local += 1u;
        }
        const uint32_t surv_count = block_reduce_sum_u32(surv_local, s_u32);
        if (cfg.top_p < 1.0f && surv_count > 1u) {
            min_key = radix_descending_mass_key(
                row_logit, row_prob, vocab_size, min_key, false, (double)cfg.top_p,
                s_mass, s_hist, &s_prefix, &s_rem_mass);
            for (int tid = threadIdx.x; tid < vocab_size; tid += blockDim.x) {
                if (make_rank_key(row_logit[tid], tid) < min_key) {
                    row_prob[tid] = 0.0f;
                }
            }
            __syncthreads();
        }

        // Stage D: Sample (direct sorted CDF, bonus ascending-tid CDF, or draft rejection)
        double norm_sum_d = 0.0;
        for (int tid = threadIdx.x; tid < vocab_size; tid += blockDim.x) {
            if (make_rank_key(row_logit[tid], tid) >= min_key) {
                norm_sum_d += (double)row_prob[tid];
            }
        }
        const double norm_sum = block_reduce_sum_f64(norm_sum_d, s_f64);

        if (cfg.draft_token_id >= 0) {
            const int32_t d = cfg.draft_token_id;
            const double raw_pd = (d < vocab_size && make_rank_key(row_logit[d], d) >= min_key)
                                      ? (double)row_prob[d]
                                      : 0.0;
            const double p_d = (norm_sum > 0.0) ? (raw_pd / norm_sum) : 0.0;
            draft_prob = (float)p_d;
            if ((double)cfg.uniform_sample < fmin(1.0, p_d)) {
                draft_accepted = 1;
                sampled_id = d;
            } else {
                draft_accepted = 0;
                const double res_sum = fmax(norm_sum - raw_pd, 0.0);
                if (res_sum > 0.0) {
                    sampled_id = radix_ascending_tid_cdf(
                        row_logit, row_prob, vocab_size, min_key, d,
                        (double)cfg.recovery_uniform * res_sum,
                        s_mass, s_hist, &s_prefix, &s_rem_mass);
                } else {
                    sampled_id = winner.idx;
                }
            }
        } else if (cfg.use_token_id_cdf != 0) {
            sampled_id = radix_ascending_tid_cdf(
                row_logit, row_prob, vocab_size, min_key, -1,
                (double)cfg.recovery_uniform * norm_sum,
                s_mass, s_hist, &s_prefix, &s_rem_mass);
        } else {
            const double target = (double)cfg.uniform_sample * (double)((float)norm_sum);
            const uint64_t chosen_key = radix_descending_mass_key(
                row_logit, row_prob, vocab_size, min_key, true, target,
                s_mass, s_hist, &s_prefix, &s_rem_mass);
            sampled_id = (int32_t)(~uint32_t(chosen_key));
        }
    }

    if (threadIdx.x == 0) {
        w34_sampler_row_out out = {};
        out.sampled_id = sampled_id;
        out.draft_accepted = draft_accepted;
        out.status = 0;
        out.draft_prob = draft_prob;
        if (want_logprobs) {
            const int32_t emitted_id =
                (cfg.draft_token_id >= 0 && draft_accepted != 0) ? cfg.draft_token_id : sampled_id;
            float em_raw = -FLT_MAX;
            if (emitted_id >= 0 && emitted_id < vocab_size) {
                const float b = (row_bias != nullptr) ? row_bias[emitted_id] : 0.0f;
                em_raw = __half2float(row_logits[emitted_id]) + b;
            }
            out.sampled_logprob = em_raw - log_z;
            for (int j = 0; j < W34_MAX_TOP_LOGPROBS; ++j) {
                if (j < actual_lp) {
                    out.top_ids[j] = picked_lp_ids[j];
                    out.top_logprobs[j] = picked_lp_vals[j] - log_z;
                } else {
                    out.top_ids[j] = -1;
                    out.top_logprobs[j] = -INFINITY;
                }
            }
        } else {
            out.sampled_logprob = 0.0f;
            for (int j = 0; j < W34_MAX_TOP_LOGPROBS; ++j) {
                out.top_ids[j] = -1;
                out.top_logprobs[j] = -INFINITY;
            }
        }
        row_outs[row] = out;
    }
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

    const bool aligned16 = (((uintptr_t)row_logits) & 15u) == 0u;
    const int vec_end = aligned16 ? (local_vocab_size & ~7) : 0;

    for (int k = 0; k < k_iters; k++) {
        ValIdx local_best = {-FLT_MAX, -1};
        if (k == 0 && aligned16) {
            for (int col = (int)threadIdx.x * 8; col < vec_end; col += (int)blockDim.x * 8) {
                HalfVec8 hv;
                hv.u = *reinterpret_cast<const uint4*>(row_logits + col);
#pragma unroll
                for (int j = 0; j < 8; ++j) {
                    const float v = __half2float(hv.h[j]);
                    local_best = better_pair(local_best, ValIdx{v, vocab_offset + col + j});
                }
            }
            for (int i = vec_end + threadIdx.x; i < local_vocab_size; i += blockDim.x) {
                const float v = __half2float(row_logits[i]);
                local_best = better_pair(local_best, ValIdx{v, vocab_offset + i});
            }
        } else {
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

extern "C" int w34_sampler_step_f16(
    const w34_exec* ex,
    const void* logits_f16,
    const w34_sampler_row_cfg* row_cfgs,
    const uint32_t* bitmask_u32,
    const w34_sampler_sparse_entry* sparse_entries,
    w34_sampler_row_out* row_outs,
    int num_rows,
    int vocab_size,
    int total_sparse_entries) {
    if (ex == nullptr || logits_f16 == nullptr || row_cfgs == nullptr || row_outs == nullptr ||
        num_rows <= 0 || vocab_size <= 0 || total_sparse_entries < 0 ||
        (total_sparse_entries > 0 && sparse_entries == nullptr)) {
        return (int)cudaErrorInvalidValue;
    }
    const size_t elems = (size_t)num_rows * (size_t)vocab_size;
    const size_t slice_bytes = ((elems * sizeof(float)) + 255u) & ~size_t(255u);
    const size_t needed_bytes = (total_sparse_entries > 0) ? (slice_bytes * 4u) : (slice_bytes * 2u);
    if (ex->scratch_ptr == 0 || ex->scratch_size < needed_bytes) {
        return (int)cudaErrorMemoryAllocation;
    }
    uint8_t* ws = (uint8_t*)(uintptr_t)ex->scratch_ptr;
    float* work_logit = (float*)ws;
    float* work_prob = (float*)(ws + slice_bytes);
    uint32_t* work_counts = nullptr;
    float* work_bias = nullptr;
    cudaStream_t stream = (cudaStream_t)ex->stream;
    if (total_sparse_entries > 0) {
        work_counts = (uint32_t*)(ws + slice_bytes * 2u);
        work_bias = (float*)(ws + slice_bytes * 3u);
        cudaError_t err = cudaMemsetAsync(work_counts, 0, slice_bytes * 2u, stream);
        if (err != cudaSuccess) return (int)err;
        w34_sampler::sampler_scatter_sparse_kernel<<<num_rows, 256, 0, stream>>>(
            row_cfgs, sparse_entries, work_counts, work_bias, vocab_size);
        err = cudaGetLastError();
        if (err != cudaSuccess) return (int)err;
    }
    w34_sampler::sampler_step_f16_kernel<<<num_rows, 1024, 0, stream>>>(
        (const __half*)logits_f16,
        row_cfgs,
        bitmask_u32,
        work_counts,
        work_bias,
        work_logit,
        work_prob,
        row_outs,
        vocab_size);
    return (int)cudaGetLastError();
}
