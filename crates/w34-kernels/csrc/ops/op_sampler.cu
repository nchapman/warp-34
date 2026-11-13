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
    const ValIdx out{s_val[0], s_idx[0]};
    __syncthreads();
    return out;
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
    const float out = s_val[0];
    __syncthreads();
    return out;
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
    const double out = s_val[0];
    __syncthreads();
    return out;
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
    const uint32_t out = s_val[0];
    __syncthreads();
    return out;
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

__device__ __forceinline__ uint64_t radix_kth_largest_key(
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
                const uint32_t b = (uint32_t)((key >> shift) & 0xffu);
                const uint32_t peers = __match_any_sync(__activemask(), b);
                if (((int)(threadIdx.x & 31)) == (__ffs((int)peers) - 1)) {
                    atomicAdd(&s_hist[b], (uint32_t)__popc(peers));
                }
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

__device__ __forceinline__ void warp_atomic_add_bin_mass_and_hist(
    uint32_t b,
    double val,
    double* s_mass,
    uint32_t* s_hist) {
    const uint32_t active = __activemask();
    const uint32_t peers = __match_any_sync(active, b);
    const int lane = (int)(threadIdx.x & 31);
    const int leader = __ffs((int)peers) - 1;
    if (peers == (1u << lane)) {
        atomicAdd(&s_mass[b], val);
        atomicAdd(&s_hist[b], 1u);
        return;
    }
    const uint64_t bits = __double_as_longlong(val);
    const uint32_t lo = (uint32_t)bits;
    const uint32_t hi = (uint32_t)(bits >> 32);
    double sum = 0.0;
    uint32_t m = peers;
    while (m != 0u) {
        const int src = __ffs((int)m) - 1;
        const uint32_t s_lo = __shfl_sync(peers, lo, src);
        const uint32_t s_hi = __shfl_sync(peers, hi, src);
        if (lane == leader) {
            sum += __longlong_as_double(((int64_t)s_hi << 32) | (int64_t)s_lo);
        }
        m &= (m - 1u);
    }
    if (lane == leader) {
        atomicAdd(&s_mass[b], sum);
        atomicAdd(&s_hist[b], (uint32_t)__popc(peers));
    }
}

__device__ __forceinline__ uint64_t radix_descending_mass_key(
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
                warp_atomic_add_bin_mass_and_hist(b, (double)p, s_mass, s_hist);
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

__device__ __forceinline__ int32_t radix_ascending_tid_cdf(
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

    const int start_pass =
        (vocab_size <= 256) ? 3 : ((vocab_size <= 65536) ? 2 : ((vocab_size <= (1 << 24)) ? 1 : 0));
    for (int pass = start_pass; pass < 4; ++pass) {
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
            if (pass == start_pass || (u_tid >> hi_shift) == (pref >> hi_shift)) {
                const uint32_t b = (u_tid >> shift) & 0xffu;
                warp_atomic_add_bin_mass_and_hist(b, (double)p, s_mass, s_hist);
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

static constexpr int kFastGreedySplits = 16;
static constexpr int kFastGreedyMaxSparse = 32;

struct __align__(16) FastGreedySplitPart {
    float    sum_exp;
    float    raw_max;
    float    best_val;
    int32_t  best_idx;
    uint32_t valid_cnt;
    uint32_t nonfinite;
    uint64_t reserved;
};

__device__ __forceinline__ bool is_row_fast_greedy(const w34_sampler_row_cfg& cfg) {
    const bool is_greedy = (cfg.temperature <= 1e-6f || cfg.top_k == 1);
    const bool no_mask = (cfg.has_bitmask == 0 || cfg.row_bitmask_word_offset < 0);
    const bool small_sparse = (cfg.sparse_count >= 0 && cfg.sparse_count <= kFastGreedyMaxSparse);
    const bool lp_at_most_zero = (cfg.top_logprobs_k <= 0);
    return is_greedy && no_mask && small_sparse && lp_at_most_zero;
}

__device__ __forceinline__ bool is_row_fast_greedy_masked(
    const w34_sampler_row_cfg& cfg,
    const uint32_t* bitmask_u32) {
    if (is_row_fast_greedy(cfg)) return true;
    const bool is_greedy = (cfg.temperature <= 1e-6f || cfg.top_k == 1);
    const bool has_mask =
        (bitmask_u32 != nullptr && cfg.has_bitmask != 0 && cfg.row_bitmask_word_offset >= 0);
    return is_greedy && has_mask && cfg.sparse_count == 0 && cfg.top_logprobs_k < 0;
}

__device__ __forceinline__ void eval_fast_greedy_elem(
    __half h,
    int col,
    const w34_sampler_row_cfg& cfg,
    bool has_rep,
    const w34_sampler_sparse_entry* s_local,
    int local_cnt,
    float& local_raw_max,
    float& local_sum,
    bool want_lp,
    ValIdx& local_best,
    uint32_t& local_valid,
    uint32_t& local_nonfinite) {
    float bias = 0.0f;
    uint32_t out_cnt = 0u;
    bool in_prompt = false;
    bool banned = false;
    for (int i = 0; i < local_cnt; ++i) {
        if (s_local[i].token_id == col) {
            bias = s_local[i].logit_bias;
            out_cnt = (s_local[i].output_count > 0) ? (uint32_t)s_local[i].output_count : 0u;
            in_prompt = (s_local[i].in_prompt != 0);
            banned = (s_local[i].banned != 0);
            break;
        }
    }
    const float raw = __half2float(h);
    const float biased = raw + bias;
    if (isfinite(biased)) {
        if (biased > local_raw_max) {
            if (want_lp) {
                local_sum = isfinite(local_raw_max)
                    ? (local_sum * expf(local_raw_max - biased) + 1.0f)
                    : 1.0f;
            }
            local_raw_max = biased;
        } else if (want_lp) {
            local_sum += expf(biased - local_raw_max);
        }
    }
    if (banned) return;
    if (!isfinite(raw)) {
        local_nonfinite = 1u;
        return;
    }
    float l = biased;
    if ((in_prompt || out_cnt > 0u) && has_rep) {
        l = (l > 0.0f) ? (l / cfg.repetition_penalty) : (l * cfg.repetition_penalty);
    }
    if (out_cnt > 0u) {
        l -= cfg.frequency_penalty * (float)out_cnt;
        l -= cfg.presence_penalty;
    }
    if (!isfinite(l)) {
        local_nonfinite = 1u;
        return;
    }
    local_valid += 1u;
    local_best = better_pair(local_best, ValIdx{l, col});
}

// Parallel split-vocab (`grid = (16, num_rows), block = 256`) vectorized `uint4` fast path
// for unmasked greedy rows (`top_logprobs_k <= 0`, `sparse_count <= 32`), spreading 16-row
// decode across 256 CTAs (all 84 SMs) with hardware `expf`, shared-memory sparse filtering,
// and zero global `work_logit`/`work_counts` DRAM writes or memsets.
__global__ void sampler_fast_greedy_split_f16_kernel(
    const __half* __restrict__ logits_f16,
    const w34_sampler_row_cfg* __restrict__ row_cfgs,
    const uint32_t* __restrict__ bitmask_u32,
    const w34_sampler_sparse_entry* __restrict__ sparse_entries,
    FastGreedySplitPart* __restrict__ out_parts,
    int num_rows,
    int vocab_size) {
    const int split_idx = (int)blockIdx.x;
    const int row = (int)blockIdx.y;
    if (row >= num_rows || split_idx >= kFastGreedySplits) return;
    const w34_sampler_row_cfg cfg = row_cfgs[row];
    if (!is_row_fast_greedy_masked(cfg, bitmask_u32)) return;

    __shared__ float s_val[8];
    __shared__ int32_t s_idx[8];
    __shared__ uint32_t s_u32[8];
    __shared__ w34_sampler_sparse_entry s_local[kFastGreedyMaxSparse];
    __shared__ int s_local_cnt;
    __shared__ int s_min_tid;
    __shared__ int s_max_tid;

    const int chunk_elems = ((vocab_size + kFastGreedySplits - 1) / kFastGreedySplits + 7) & ~7;
    const int col_start = split_idx * chunk_elems;
    const int col_end = (col_start + chunk_elems < vocab_size) ? (col_start + chunk_elems) : vocab_size;
    const int span = col_end - col_start;

    const int tid = threadIdx.x;
    if (tid == 0) {
        int cnt = 0;
        int min_t = vocab_size;
        int max_t = -1;
        if (sparse_entries != nullptr && cfg.sparse_count > 0 && cfg.sparse_offset >= 0) {
            const w34_sampler_sparse_entry* row_sp = sparse_entries + cfg.sparse_offset;
            for (int i = 0; i < cfg.sparse_count; ++i) {
                const w34_sampler_sparse_entry e = row_sp[i];
                if (e.token_id >= col_start && e.token_id < col_end && cnt < kFastGreedyMaxSparse) {
                    s_local[cnt++] = e;
                    if (e.token_id < min_t) min_t = e.token_id;
                    if (e.token_id > max_t) max_t = e.token_id;
                }
            }
        }
        s_local_cnt = cnt;
        s_min_tid = min_t;
        s_max_tid = max_t;
    }
    __syncthreads();

    const int local_cnt = s_local_cnt;
    const int min_t = s_min_tid;
    const int max_t = s_max_tid;
    const bool want_lp = (cfg.top_logprobs_k >= 0);
    const bool has_rep = fabsf(cfg.repetition_penalty - 1.0f) > 1e-6f;

    const __half* row_logits = logits_f16 + (size_t)row * vocab_size;
    const __half* split_ptr = row_logits + col_start;
    const uint32_t* row_mask =
        (bitmask_u32 != nullptr && cfg.has_bitmask != 0 && cfg.row_bitmask_word_offset >= 0)
            ? (bitmask_u32 + (size_t)cfg.row_bitmask_word_offset)
            : nullptr;

    ValIdx local_best = {-FLT_MAX, -1};
    float local_raw_max = -FLT_MAX;
    float local_sum = 0.0f;
    uint32_t local_valid = 0u;
    uint32_t local_nonfinite = 0u;

    const bool aligned16 = (((uintptr_t)split_ptr) & 15u) == 0u;
    if (span > 0) {
        if (aligned16) {
            const int vec_end = span & ~7;
            for (int lc = (tid << 3); lc < vec_end; lc += 256 * 8) {
                const int col0 = col_start + lc;
                if (row_mask != nullptr) {
                    const uint32_t mbyte = (row_mask[col0 >> 5] >> (col0 & 24)) & 0xffu;
                    if (mbyte == 0u) continue;
                    HalfVec8 hv;
                    hv.u = *reinterpret_cast<const uint4*>(split_ptr + lc);
#pragma unroll
                    for (int j = 0; j < 8; ++j) {
                        if (((mbyte >> j) & 1u) != 0u) {
                            const float v = __half2float(hv.h[j]);
                            if (!isfinite(v)) {
                                local_nonfinite = 1u;
                            } else {
                                local_valid += 1u;
                                local_best = better_pair(local_best, ValIdx{v, col0 + j});
                            }
                        }
                    }
                    continue;
                }
                HalfVec8 hv;
                hv.u = *reinterpret_cast<const uint4*>(split_ptr + lc);
                if (local_cnt == 0 || col0 + 7 < min_t || col0 > max_t) {
#pragma unroll
                    for (int j = 0; j < 8; ++j) {
                        const float v = __half2float(hv.h[j]);
                        if (!isfinite(v)) {
                            local_nonfinite = 1u;
                        } else {
                            if (v > local_raw_max) {
                                if (want_lp) {
                                    local_sum = isfinite(local_raw_max)
                                        ? (local_sum * expf(local_raw_max - v) + 1.0f)
                                        : 1.0f;
                                }
                                local_raw_max = v;
                            } else if (want_lp) {
                                local_sum += expf(v - local_raw_max);
                            }
                            local_valid += 1u;
                            local_best = better_pair(local_best, ValIdx{v, col0 + j});
                        }
                    }
                } else {
#pragma unroll
                    for (int j = 0; j < 8; ++j) {
                        eval_fast_greedy_elem(
                            hv.h[j], col0 + j, cfg, has_rep, s_local, local_cnt,
                            local_raw_max, local_sum, want_lp,
                            local_best, local_valid, local_nonfinite);
                    }
                }
            }
            for (int lc = vec_end + tid; lc < span; lc += 256) {
                const int col = col_start + lc;
                if (row_mask != nullptr && ((row_mask[col >> 5] & (1u << (col & 31))) == 0u)) {
                    continue;
                }
                eval_fast_greedy_elem(
                    split_ptr[lc], col, cfg, has_rep, s_local, local_cnt,
                    local_raw_max, local_sum, want_lp,
                    local_best, local_valid, local_nonfinite);
            }
        } else {
            for (int lc = tid; lc < span; lc += 256) {
                const int col = col_start + lc;
                if (row_mask != nullptr && ((row_mask[col >> 5] & (1u << (col & 31))) == 0u)) {
                    continue;
                }
                eval_fast_greedy_elem(
                    split_ptr[lc], col, cfg, has_rep, s_local, local_cnt,
                    local_raw_max, local_sum, want_lp,
                    local_best, local_valid, local_nonfinite);
            }
        }
    }

    const ValIdx split_best = block_reduce_max_pair(local_best, s_val, s_idx);
    const ValIdx split_raw_pair = block_reduce_max_pair(ValIdx{local_raw_max, 0}, s_val, s_idx);
    const float split_raw_max = split_raw_pair.val;
    const uint32_t split_valid = block_reduce_sum_u32(local_valid, s_u32);
    const uint32_t split_nonfinite = block_reduce_sum_u32(local_nonfinite, s_u32);

    float split_sum_exp = 0.0f;
    if (want_lp && split_nonfinite == 0u && span > 0 && isfinite(split_raw_max)) {
        const float scaled = (isfinite(local_raw_max) && local_sum > 0.0f)
            ? (local_sum * expf(local_raw_max - split_raw_max))
            : 0.0f;
        split_sum_exp = block_reduce_sum(scaled, s_val);
    }

    if (tid == 0) {
        FastGreedySplitPart part;
        part.sum_exp = split_sum_exp;
        part.raw_max = split_raw_max;
        part.best_val = split_best.val;
        part.best_idx = split_best.idx;
        part.valid_cnt = split_valid;
        part.nonfinite = split_nonfinite;
        part.reserved = 0ull;
        out_parts[row * kFastGreedySplits + split_idx] = part;
    }
}

__device__ __forceinline__ void finalize_fast_greedy_row(
    int row,
    const __half* __restrict__ logits_f16,
    const w34_sampler_row_cfg& cfg,
    const w34_sampler_sparse_entry* __restrict__ sparse_entries,
    ValIdx winner,
    float global_raw_max,
    float sum_exp_raw,
    uint32_t valid_count,
    uint32_t any_nonfinite,
    w34_sampler_row_out* __restrict__ row_outs,
    int vocab_size) {
    w34_sampler_row_out out = {};
    for (int j = 0; j < W34_MAX_TOP_LOGPROBS; ++j) {
        out.top_ids[j] = -1;
        out.top_logprobs[j] = -INFINITY;
    }
    if (any_nonfinite > 0u || valid_count == 0u) {
        out.sampled_id = -1;
        out.draft_accepted = 0;
        out.status = (any_nonfinite > 0u) ? 2 : 1;
        row_outs[row] = out;
        return;
    }
    int32_t sampled_id = winner.idx;
    int32_t draft_accepted = 0;
    float draft_prob = 0.0f;
    if (cfg.draft_token_id >= 0) {
        const bool match = (winner.idx == cfg.draft_token_id);
        draft_prob = match ? 1.0f : 0.0f;
        draft_accepted = match ? 1 : 0;
    }
    out.sampled_id = sampled_id;
    out.draft_accepted = draft_accepted;
    out.status = 0;
    out.draft_prob = draft_prob;
    if (cfg.top_logprobs_k >= 0) {
        const float log_z = global_raw_max + logf(fmaxf(sum_exp_raw, 1e-30f));
        const int32_t emitted_id =
            (cfg.draft_token_id >= 0 && draft_accepted != 0) ? cfg.draft_token_id : sampled_id;
        float em_bias = 0.0f;
        if (sparse_entries != nullptr && cfg.sparse_count > 0 && cfg.sparse_offset >= 0) {
            const w34_sampler_sparse_entry* row_sp = sparse_entries + cfg.sparse_offset;
            for (int i = 0; i < cfg.sparse_count; ++i) {
                if (row_sp[i].token_id == emitted_id) {
                    em_bias = row_sp[i].logit_bias;
                    break;
                }
            }
        }
        const __half* row_logits = logits_f16 + (size_t)row * vocab_size;
        const float em_raw = (emitted_id >= 0 && emitted_id < vocab_size)
            ? (__half2float(row_logits[emitted_id]) + em_bias)
            : -FLT_MAX;
        out.sampled_logprob = em_raw - log_z;
    } else {
        out.sampled_logprob = 0.0f;
    }
    row_outs[row] = out;
}

__device__ __forceinline__ void merge_fast_greedy_row(
    int row,
    const __half* __restrict__ logits_f16,
    const w34_sampler_row_cfg& cfg,
    const w34_sampler_sparse_entry* __restrict__ sparse_entries,
    const FastGreedySplitPart* __restrict__ fast_split_parts,
    w34_sampler_row_out* __restrict__ row_outs,
    int vocab_size) {
    const FastGreedySplitPart* parts = fast_split_parts + row * kFastGreedySplits;
    ValIdx winner = {-FLT_MAX, -1};
    float global_raw_max = -FLT_MAX;
    uint32_t valid_count = 0u;
    uint32_t any_nonfinite = 0u;
    for (int s = 0; s < kFastGreedySplits; ++s) {
        const FastGreedySplitPart p = parts[s];
        valid_count += p.valid_cnt;
        any_nonfinite |= p.nonfinite;
        if (p.raw_max > global_raw_max) global_raw_max = p.raw_max;
        if (p.valid_cnt > 0u) {
            winner = better_pair(winner, ValIdx{p.best_val, p.best_idx});
        }
    }
    float sum_exp_raw = 0.0f;
    if (cfg.top_logprobs_k >= 0 && any_nonfinite == 0u && valid_count > 0u) {
        for (int s = 0; s < kFastGreedySplits; ++s) {
            const FastGreedySplitPart p = parts[s];
            if (isfinite(p.raw_max) && p.sum_exp > 0.0f) {
                sum_exp_raw += p.sum_exp * expf(p.raw_max - global_raw_max);
            }
        }
    }
    finalize_fast_greedy_row(
        row, logits_f16, cfg, sparse_entries,
        winner, global_raw_max, sum_exp_raw, valid_count, any_nonfinite,
        row_outs, vocab_size);
}

__global__ void sampler_fast_greedy_merge_f16_kernel(
    const __half* __restrict__ logits_f16,
    const w34_sampler_row_cfg* __restrict__ row_cfgs,
    const uint32_t* __restrict__ bitmask_u32,
    const w34_sampler_sparse_entry* __restrict__ sparse_entries,
    const FastGreedySplitPart* __restrict__ fast_split_parts,
    w34_sampler_row_out* __restrict__ row_outs,
    int num_rows,
    int vocab_size) {
    const int row = (int)blockIdx.x;
    if (row >= num_rows) return;
    const w34_sampler_row_cfg cfg = row_cfgs[row];
    const int lane = (int)threadIdx.x;
    if (!is_row_fast_greedy_masked(cfg, bitmask_u32)) {
        if (lane == 0) {
            w34_sampler_row_out out = {};
            out.sampled_id = -1;
            out.status = 3;
            for (int j = 0; j < W34_MAX_TOP_LOGPROBS; ++j) {
                out.top_ids[j] = -1;
                out.top_logprobs[j] = -INFINITY;
            }
            row_outs[row] = out;
        }
        return;
    }
    static_assert(kFastGreedySplits <= 32, "kFastGreedySplits must fit in one warp");
    FastGreedySplitPart p = {};
    p.raw_max = -FLT_MAX;
    p.best_val = -FLT_MAX;
    p.best_idx = -1;
    if (lane < kFastGreedySplits) {
        p = fast_split_parts[row * kFastGreedySplits + lane];
    }
    ValIdx winner = (p.valid_cnt > 0u) ? ValIdx{p.best_val, p.best_idx} : ValIdx{-FLT_MAX, -1};
    float global_raw_max = p.raw_max;
    uint32_t valid_count = p.valid_cnt;
    uint32_t any_nonfinite = p.nonfinite;
#pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
        const float ov = __shfl_xor_sync(0xffffffffu, winner.val, off);
        const int32_t oi = __shfl_xor_sync(0xffffffffu, winner.idx, off);
        winner = better_pair(winner, ValIdx{ov, oi});
        const float orm = __shfl_xor_sync(0xffffffffu, global_raw_max, off);
        if (orm > global_raw_max) global_raw_max = orm;
        valid_count += __shfl_xor_sync(0xffffffffu, valid_count, off);
        any_nonfinite |= __shfl_xor_sync(0xffffffffu, any_nonfinite, off);
    }
    float sum_exp_raw = 0.0f;
    if (cfg.top_logprobs_k >= 0 && any_nonfinite == 0u && valid_count > 0u) {
        float part_scaled = (lane < kFastGreedySplits && isfinite(p.raw_max) && p.sum_exp > 0.0f)
            ? (p.sum_exp * expf(p.raw_max - global_raw_max))
            : 0.0f;
#pragma unroll
        for (int off = 16; off > 0; off >>= 1) {
            part_scaled += __shfl_xor_sync(0xffffffffu, part_scaled, off);
        }
        sum_exp_raw = part_scaled;
    }
    if (lane == 0) {
        finalize_fast_greedy_row(
            row, logits_f16, cfg, sparse_entries,
            winner, global_raw_max, sum_exp_raw, valid_count, any_nonfinite,
            row_outs, vocab_size);
    }
}

__global__ void sampler_scatter_ids_to_tokens_kernel(
    const w34_sampler_row_out* __restrict__ row_outs,
    const int32_t* __restrict__ greedy_ids,
    int32_t* __restrict__ dst_token_ids,
    int num_rows) {
    const int i = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    if (i >= num_rows) return;
    const int32_t id = (row_outs != nullptr) ? row_outs[i].sampled_id : greedy_ids[i];
    dst_token_ids[i] = (id >= 0) ? id : 0;
}

// Grid: num_rows, Block: 512 threads (up to 128 regs/thread with 0 stack spill)
__global__ void __launch_bounds__(512) sampler_step_f16_kernel(
    const __half* __restrict__ logits_f16,
    const w34_sampler_row_cfg* __restrict__ row_cfgs,
    const uint32_t* __restrict__ bitmask_u32,
    const uint32_t* __restrict__ work_counts_u32,
    const float* __restrict__ work_bias_f32,
    float* __restrict__ work_logit_all,
    float* __restrict__ work_prob_all,
    const FastGreedySplitPart* __restrict__ fast_split_parts,
    w34_sampler_row_out* __restrict__ row_outs,
    int vocab_size) {
    const int row = blockIdx.x;
    const w34_sampler_row_cfg cfg = row_cfgs[row];
    const size_t row_base = (size_t)row * vocab_size;
    const __half* row_logits = logits_f16 + row_base;

    if (fast_split_parts != nullptr && work_counts_u32 == nullptr && work_bias_f32 == nullptr &&
        cfg.sparse_count <= 0 && is_row_fast_greedy_masked(cfg, bitmask_u32)) {
        if (threadIdx.x == 0) {
            merge_fast_greedy_row(
                row, logits_f16, cfg, nullptr, fast_split_parts, row_outs, vocab_size);
        }
        return;
    }
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
    __shared__ int32_t picked_lp_ids[W34_MAX_TOP_LOGPROBS];
    __shared__ float picked_lp_vals[W34_MAX_TOP_LOGPROBS];

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
                if (threadIdx.x == 0) {
                    picked_lp_ids[k] = win.idx;
                    picked_lp_vals[k] = win.val;
                }
                __syncthreads();
            }
        }
        double sum_exp_raw = 0.0;
        for (int tid = threadIdx.x; tid < vocab_size; tid += blockDim.x) {
            const float b = (row_bias != nullptr) ? row_bias[tid] : 0.0f;
            const float v = __half2float(row_logits[tid]) + b;
            sum_exp_raw += (double)expf(v - raw_max);
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
            if (!is_greedy_policy) {
                row_logit[tid] = -INFINITY;
                row_prob[tid] = 0.0f;
            }
            continue;
        }
        const float raw = __half2float(row_logits[tid]);
        if (!isfinite(raw)) {
            local_nonfinite = 1u;
            if (!is_greedy_policy) {
                row_logit[tid] = -INFINITY;
                row_prob[tid] = 0.0f;
            }
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
            if (!is_greedy_policy) {
                row_logit[tid] = -INFINITY;
                row_prob[tid] = 0.0f;
            }
            continue;
        }
        if (!is_greedy_policy) {
            row_logit[tid] = l;
        }
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
    w34_sampler::FastGreedySplitPart* fast_parts = nullptr;
    const size_t parts_bytes =
        (((size_t)num_rows * w34_sampler::kFastGreedySplits * sizeof(w34_sampler::FastGreedySplitPart)) +
         255u) &
        ~size_t(255u);
    if (total_sparse_entries == 0 && vocab_size >= 1024 &&
        ex->scratch_size >= needed_bytes + parts_bytes) {
        // Place `fast_parts` in a disjoint scratch region past `work_prob` (`ws + needed_bytes`)
        // so non-fast rows writing `work_logit` / `work_prob` in the same launch cannot clobber
        // fast-greedy split summaries.
        fast_parts = reinterpret_cast<w34_sampler::FastGreedySplitPart*>(ws + needed_bytes);
        w34_sampler::sampler_fast_greedy_split_f16_kernel<<<
            dim3((unsigned)w34_sampler::kFastGreedySplits, (unsigned)num_rows), 256, 0, stream>>>(
            (const __half*)logits_f16,
            row_cfgs,
            bitmask_u32,
            nullptr,
            fast_parts,
            num_rows,
            vocab_size);
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) return (int)err;
    }
    w34_sampler::sampler_step_f16_kernel<<<num_rows, 512, 0, stream>>>(
        (const __half*)logits_f16,
        row_cfgs,
        bitmask_u32,
        work_counts,
        work_bias,
        work_logit,
        work_prob,
        fast_parts,
        row_outs,
        vocab_size);
    return (int)cudaGetLastError();
}

extern "C" int w34_sampler_step_fast_greedy_masked_f16(
    const w34_exec* ex,
    const void* logits_f16,
    const w34_sampler_row_cfg* row_cfgs,
    const uint32_t* bitmask_u32,
    const w34_sampler_sparse_entry* sparse_entries,
    w34_sampler_row_out* row_outs,
    int num_rows,
    int vocab_size) {
    if (ex == nullptr || logits_f16 == nullptr || row_cfgs == nullptr || row_outs == nullptr ||
        num_rows <= 0 || vocab_size < 1024) {
        return (int)cudaErrorInvalidValue;
    }
    const size_t parts_bytes =
        (size_t)num_rows * w34_sampler::kFastGreedySplits * sizeof(w34_sampler::FastGreedySplitPart);
    if (ex->scratch_ptr == 0 || ex->scratch_size < parts_bytes) {
        return (int)cudaErrorMemoryAllocation;
    }
    cudaStream_t stream = (cudaStream_t)ex->stream;
    auto* fast_parts = reinterpret_cast<w34_sampler::FastGreedySplitPart*>((uintptr_t)ex->scratch_ptr);
    w34_sampler::sampler_fast_greedy_split_f16_kernel<<<
        dim3((unsigned)w34_sampler::kFastGreedySplits, (unsigned)num_rows), 256, 0, stream>>>(
        (const __half*)logits_f16,
        row_cfgs,
        bitmask_u32,
        sparse_entries,
        fast_parts,
        num_rows,
        vocab_size);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) return (int)err;
    w34_sampler::sampler_fast_greedy_merge_f16_kernel<<<(unsigned)num_rows, 32, 0, stream>>>(
        (const __half*)logits_f16,
        row_cfgs,
        bitmask_u32,
        sparse_entries,
        fast_parts,
        row_outs,
        num_rows,
        vocab_size);
    return (int)cudaGetLastError();
}

extern "C" int w34_sampler_step_fast_greedy_f16(
    const w34_exec* ex,
    const void* logits_f16,
    const w34_sampler_row_cfg* row_cfgs,
    const w34_sampler_sparse_entry* sparse_entries,
    w34_sampler_row_out* row_outs,
    int num_rows,
    int vocab_size) {
    return w34_sampler_step_fast_greedy_masked_f16(
        ex, logits_f16, row_cfgs, nullptr, sparse_entries, row_outs, num_rows, vocab_size);
}

extern "C" int w34_sampler_scatter_ids_to_tokens(
    const w34_exec* ex,
    const w34_sampler_row_out* row_outs,
    const int32_t* greedy_ids,
    int32_t* dst_token_ids,
    int num_rows) {
    if (ex == nullptr || dst_token_ids == nullptr || num_rows <= 0 ||
        (row_outs == nullptr && greedy_ids == nullptr)) {
        return (int)cudaErrorInvalidValue;
    }
    const unsigned blocks = (unsigned)((num_rows + 31) / 32);
    w34_sampler::sampler_scatter_ids_to_tokens_kernel<<<blocks, 32, 0, (cudaStream_t)ex->stream>>>(
        row_outs, greedy_ids, dst_token_ids, num_rows);
    return (int)cudaGetLastError();
}
