#include "w34_cuda.h"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <functional>
#include <mutex>
#include <thread>
#include <vector>

#if defined(__x86_64__) || defined(_M_X64)
#include <immintrin.h>
#include <linux/futex.h>
#include <pthread.h>
#include <sched.h>
#include <sys/syscall.h>
#include <unistd.h>
#endif

namespace w34_moe_bridge {

constexpr int kMaxLayers = 64;
constexpr int kMaxTokens = 4;
constexpr int kMaxPrefillJobTokens = 96;
constexpr int kMaxTopK = 16;
constexpr int kMaxPairs = kMaxTokens * kMaxTopK; // 64
constexpr int kMaxHidden = 4096;
constexpr int kMaxInterm = 2048;
constexpr int kGroupSize = 128;

struct alignas(64) MoeBridgeReqSlot {
    int64_t seq;
    int32_t num_tokens;
    int32_t top_k;
    int32_t layer_id;
    int32_t num_cpu_pairs;
    int32_t hidden;
    int32_t intermediate;
    int32_t local_num_experts;
    int32_t pad[5];
    int32_t local_expert_ids[kMaxPairs];
    float topk_weights[kMaxPairs];
    float x_f32[kMaxTokens * kMaxHidden];
};

struct alignas(64) MoeBridgeRespSlot {
    int64_t seq;
    int32_t pad[14];
    float y_f32[kMaxTokens * kMaxHidden];
};

struct LayerHostDesc {
    const uint8_t* packs_base = nullptr;
    size_t pack_stride_bytes = 0;
    size_t gu_packed_bytes = 0;
    size_t gu_scales_bytes = 0;
    size_t dn_packed_bytes = 0;
    size_t dn_scales_bytes = 0;
    int local_num_experts = 0;
    int intermediate = 0;
    int hidden = 0;
    int group_size = 128;
};

#if defined(__x86_64__) || defined(_M_X64)

__attribute__((target("avx2,fma,f16c,avx512f")))
static inline float hsum512_ps(__m512 v) {
    __m256 lo = _mm512_castps512_ps256(v);
    __m256 hi = _mm256_castpd_ps(_mm512_extractf64x4_pd(_mm512_castps_pd(v), 1));
    __m256 s256 = _mm256_add_ps(lo, hi);
    __m128 s128 = _mm_add_ps(_mm256_castps256_ps128(s256), _mm256_extractf128_ps(s256, 1));
    s128 = _mm_hadd_ps(s128, s128);
    s128 = _mm_hadd_ps(s128, s128);
    return _mm_cvtss_f32(s128);
}

__attribute__((target("avx2,fma,f16c,avx512f")))
static inline void prepare_x_perm(
    const float* __restrict__ x,
    float* __restrict__ xp,
    float* __restrict__ xg,
    int K) {
    const int num_groups = K / kGroupSize;
    for (int g = 0; g < num_groups; ++g) {
        const float* src = x + g * 128;
        float* dst = xp + g * 128;
        float gsum = 0.0f;
        for (int l = 0; l < 16; ++l) {
            for (int b = 0; b < 8; ++b) {
                const float v = src[l * 8 + b];
                dst[b * 16 + l] = v;
                gsum += v;
            }
        }
        xg[g] = gsum;
    }
}

template <int T>
__attribute__((target("avx2,fma,f16c,avx512f,avx512vl")))
static inline void eval_row_u4b8(
    const uint32_t* __restrict__ row_w,
    const uint16_t* __restrict__ row_s,
    const float* __restrict__ xp,
    const float* __restrict__ xg,
    int num_groups,
    int xp_stride,
    int xg_stride,
    float* __restrict__ out_t) {
    const __m512i mask = _mm512_set1_epi32(0xF);
    __m512 tot[T];
    float bias[T];
#pragma unroll
    for (int t = 0; t < T; ++t) {
        tot[t] = _mm512_setzero_ps();
        bias[t] = 0.0f;
    }

    for (int g = 0; g < num_groups; ++g) {
        const uint32_t* gw = row_w + g * 16;
        _mm_prefetch(reinterpret_cast<const char*>(gw + 64), _MM_HINT_T0);
        const __m512i q = _mm512_loadu_si512(reinterpret_cast<const void*>(gw));

        __m512 acc[T];
#pragma unroll
        for (int t = 0; t < T; ++t) acc[t] = _mm512_setzero_ps();

#define W34_UNROLL_NIB(B)                                                                   \
    {                                                                                       \
        const __m512 w = _mm512_cvtepi32_ps(                                                \
            _mm512_and_si512((B) == 0 ? q : _mm512_srli_epi32(q, 4 * (B)), mask));          \
        for (int t = 0; t < T; ++t) {                                                       \
            const float* xpt = xp + t * xp_stride + g * 128;                                \
            acc[t] = _mm512_fmadd_ps(w, _mm512_loadu_ps(xpt + 16 * (B)), acc[t]);           \
        }                                                                                   \
    }
        W34_UNROLL_NIB(0)
        W34_UNROLL_NIB(1)
        W34_UNROLL_NIB(2)
        W34_UNROLL_NIB(3)
        W34_UNROLL_NIB(4)
        W34_UNROLL_NIB(5)
        W34_UNROLL_NIB(6)
        W34_UNROLL_NIB(7)
#undef W34_UNROLL_NIB

        const float scale = _cvtsh_ss(row_s[g]);
        const __m512 vscale = _mm512_set1_ps(scale);
#pragma unroll
        for (int t = 0; t < T; ++t) {
            tot[t] = _mm512_fmadd_ps(acc[t], vscale, tot[t]);
            bias[t] += scale * xg[t * xg_stride + g];
        }
    }
#pragma unroll
    for (int t = 0; t < T; ++t) {
        out_t[t] = hsum512_ps(tot[t]) - 8.0f * bias[t];
    }
}

template <int T>
__attribute__((target("avx2,fma,f16c,avx512f,avx512vl")))
static inline void eval_two_rows_u4b8(
    const uint32_t* __restrict__ row_wa,
    const uint16_t* __restrict__ row_sa,
    const uint32_t* __restrict__ row_wb,
    const uint16_t* __restrict__ row_sb,
    const float* __restrict__ xp,
    const float* __restrict__ xg,
    int num_groups,
    int xp_stride,
    int xg_stride,
    float* __restrict__ out_a,
    float* __restrict__ out_b) {
    const __m512i mask = _mm512_set1_epi32(0xF);
    __m512 tot_a[T], tot_b[T];
    float bias_a[T], bias_b[T];
#pragma unroll
    for (int t = 0; t < T; ++t) {
        tot_a[t] = _mm512_setzero_ps();
        tot_b[t] = _mm512_setzero_ps();
        bias_a[t] = 0.0f;
        bias_b[t] = 0.0f;
    }

    for (int g = 0; g < num_groups; ++g) {
        const uint32_t* gwa = row_wa + g * 16;
        const uint32_t* gwb = row_wb + g * 16;
        _mm_prefetch(reinterpret_cast<const char*>(gwa + 64), _MM_HINT_T0);
        _mm_prefetch(reinterpret_cast<const char*>(gwb + 64), _MM_HINT_T0);
        const __m512i qa = _mm512_loadu_si512(reinterpret_cast<const void*>(gwa));
        const __m512i qb = _mm512_loadu_si512(reinterpret_cast<const void*>(gwb));

        __m512 acc_a[T], acc_b[T];
#pragma unroll
        for (int t = 0; t < T; ++t) {
            acc_a[t] = _mm512_setzero_ps();
            acc_b[t] = _mm512_setzero_ps();
        }

#define W34_UNROLL_NIB2(B)                                                                  \
    {                                                                                       \
        const __m512 wa = _mm512_cvtepi32_ps(                                               \
            _mm512_and_si512((B) == 0 ? qa : _mm512_srli_epi32(qa, 4 * (B)), mask));        \
        const __m512 wb = _mm512_cvtepi32_ps(                                               \
            _mm512_and_si512((B) == 0 ? qb : _mm512_srli_epi32(qb, 4 * (B)), mask));        \
        for (int t = 0; t < T; ++t) {                                                       \
            const float* xpt = xp + t * xp_stride + g * 128;                                \
            const __m512 x = _mm512_loadu_ps(xpt + 16 * (B));                               \
            acc_a[t] = _mm512_fmadd_ps(wa, x, acc_a[t]);                                    \
            acc_b[t] = _mm512_fmadd_ps(wb, x, acc_b[t]);                                    \
        }                                                                                   \
    }
        W34_UNROLL_NIB2(0)
        W34_UNROLL_NIB2(1)
        W34_UNROLL_NIB2(2)
        W34_UNROLL_NIB2(3)
        W34_UNROLL_NIB2(4)
        W34_UNROLL_NIB2(5)
        W34_UNROLL_NIB2(6)
        W34_UNROLL_NIB2(7)
#undef W34_UNROLL_NIB2

        const float sa = _cvtsh_ss(row_sa[g]);
        const float sb = _cvtsh_ss(row_sb[g]);
        const __m512 vsa = _mm512_set1_ps(sa);
        const __m512 vsb = _mm512_set1_ps(sb);
#pragma unroll
        for (int t = 0; t < T; ++t) {
            const float xgv = xg[t * xg_stride + g];
            tot_a[t] = _mm512_fmadd_ps(acc_a[t], vsa, tot_a[t]);
            tot_b[t] = _mm512_fmadd_ps(acc_b[t], vsb, tot_b[t]);
            bias_a[t] += sa * xgv;
            bias_b[t] += sb * xgv;
        }
    }
#pragma unroll
    for (int t = 0; t < T; ++t) {
        out_a[t] = hsum512_ps(tot_a[t]) - 8.0f * bias_a[t];
        out_b[t] = hsum512_ps(tot_b[t]) - 8.0f * bias_b[t];
    }
}

class CpuWorkerPool {
public:
    static constexpr int kSpinIters = 1 << 16;

    explicit CpuWorkerPool(int num_threads) : n_(num_threads) {
        for (int i = 1; i < n_; ++i) {
            threads_.emplace_back([this, i] {
                pin_thread(i);
                worker_loop(i);
            });
        }
        pin_thread(0);
    }

    ~CpuWorkerPool() {
        stop_.store(true, std::memory_order_release);
        epoch_.fetch_add(1, std::memory_order_acq_rel);
        futex_wake_all();
        for (auto& t : threads_) {
            if (t.joinable()) t.join();
        }
    }

    int size() const { return n_; }

    template <class F>
    void run(int count, F&& fn) {
        fn_ = [&fn](int item, int worker_id) { fn(item, worker_id); };
        count_ = count;
        next_.store(0, std::memory_order_relaxed);
        done_.store(0, std::memory_order_relaxed);
        const uint32_t prev = epoch_.fetch_add(1, std::memory_order_acq_rel);
        if (sleeping_.load(std::memory_order_acquire) > 0) {
            futex_wake_all();
        }
        (void)prev;
        do_work(0);
        while (done_.load(std::memory_order_acquire) != n_ - 1) {
            _mm_pause();
        }
    }

private:
    void futex_wake_all() {
        syscall(SYS_futex, reinterpret_cast<uint32_t*>(&epoch_), FUTEX_WAKE_PRIVATE, INT32_MAX, nullptr, nullptr, 0);
    }

    void futex_wait(uint32_t val) {
        sleeping_.fetch_add(1, std::memory_order_acq_rel);
        syscall(SYS_futex, reinterpret_cast<uint32_t*>(&epoch_), FUTEX_WAIT_PRIVATE, val, nullptr, nullptr, 0);
        sleeping_.fetch_sub(1, std::memory_order_acq_rel);
    }
    static void pin_thread(int idx) {
        const int ncpu = (int)std::thread::hardware_concurrency();
        if (ncpu <= 0) return;
        cpu_set_t set;
        CPU_ZERO(&set);
        // On SMT systems (e.g. 16-core / 32-thread Zen 4), spread workers across physical cores first.
        const int phys = (ncpu >= 32) ? 16 : ncpu;
        const int target_cpu = (idx < phys) ? idx : (idx % ncpu);
        CPU_SET(target_cpu, &set);
        pthread_setaffinity_np(pthread_self(), sizeof(set), &set);
    }

    void do_work(int worker_id) {
        for (;;) {
            const int i = next_.fetch_add(1, std::memory_order_relaxed);
            if (i >= count_) break;
            fn_(i, worker_id);
        }
    }

    void worker_loop(int worker_id) {
        uint32_t seen = 0;
        for (;;) {
            uint32_t e;
            int spins = 0;
            while ((e = epoch_.load(std::memory_order_acquire)) == seen) {
                if (++spins < kSpinIters) {
                    _mm_pause();
                    continue;
                }
                futex_wait(seen);
            }
            seen = e;
            if (stop_.load(std::memory_order_acquire)) return;
            do_work(worker_id);
            done_.fetch_add(1, std::memory_order_acq_rel);
        }
    }

    int n_;
    std::vector<std::thread> threads_;
    alignas(64) std::atomic<uint32_t> epoch_{0};
    alignas(64) std::atomic<int> sleeping_{0};
    alignas(64) std::atomic<int> next_{0};
    alignas(64) std::atomic<int> done_{0};
    alignas(64) std::atomic<bool> stop_{false};
    int count_ = 0;
    std::function<void(int, int)> fn_;
};

struct ExpertJob {
    int le;
    int n;
    int tok[kMaxTokens];
    float wt[kMaxTokens];
};

template <int T>
__attribute__((target("avx2,fma,f16c,avx512f,avx512vl")))
static inline void eval_four_rows_u4b8(
    const uint32_t* __restrict__ row_w0,
    const uint16_t* __restrict__ row_s0,
    const uint32_t* __restrict__ row_w1,
    const uint16_t* __restrict__ row_s1,
    const uint32_t* __restrict__ row_w2,
    const uint16_t* __restrict__ row_s2,
    const uint32_t* __restrict__ row_w3,
    const uint16_t* __restrict__ row_s3,
    const float* __restrict__ xp,
    const float* __restrict__ xg,
    int num_groups,
    int xp_stride,
    int xg_stride,
    float* __restrict__ out_0,
    float* __restrict__ out_1,
    float* __restrict__ out_2,
    float* __restrict__ out_3) {
    const __m512i mask = _mm512_set1_epi32(0xF);
    __m512 tot_0[T], tot_1[T], tot_2[T], tot_3[T];
    float bias_0[T], bias_1[T], bias_2[T], bias_3[T];
#pragma unroll
    for (int t = 0; t < T; ++t) {
        tot_0[t] = _mm512_setzero_ps();
        tot_1[t] = _mm512_setzero_ps();
        tot_2[t] = _mm512_setzero_ps();
        tot_3[t] = _mm512_setzero_ps();
        bias_0[t] = 0.0f;
        bias_1[t] = 0.0f;
        bias_2[t] = 0.0f;
        bias_3[t] = 0.0f;
    }

    for (int g = 0; g < num_groups; ++g) {
        const uint32_t* gw0 = row_w0 + g * 16;
        const uint32_t* gw1 = row_w1 + g * 16;
        const uint32_t* gw2 = row_w2 + g * 16;
        const uint32_t* gw3 = row_w3 + g * 16;
        _mm_prefetch(reinterpret_cast<const char*>(gw0 + 64), _MM_HINT_T0);
        _mm_prefetch(reinterpret_cast<const char*>(gw1 + 64), _MM_HINT_T0);
        _mm_prefetch(reinterpret_cast<const char*>(gw2 + 64), _MM_HINT_T0);
        _mm_prefetch(reinterpret_cast<const char*>(gw3 + 64), _MM_HINT_T0);
        const __m512i q0 = _mm512_loadu_si512(reinterpret_cast<const void*>(gw0));
        const __m512i q1 = _mm512_loadu_si512(reinterpret_cast<const void*>(gw1));
        const __m512i q2 = _mm512_loadu_si512(reinterpret_cast<const void*>(gw2));
        const __m512i q3 = _mm512_loadu_si512(reinterpret_cast<const void*>(gw3));

        __m512 acc0[T], acc1[T], acc2[T], acc3[T];
#pragma unroll
        for (int t = 0; t < T; ++t) {
            acc0[t] = _mm512_setzero_ps();
            acc1[t] = _mm512_setzero_ps();
            acc2[t] = _mm512_setzero_ps();
            acc3[t] = _mm512_setzero_ps();
        }

#define W34_UNROLL_NIB4(B)                                                                  \
    {                                                                                       \
        const __m512 w0 = _mm512_cvtepi32_ps(                                               \
            _mm512_and_si512((B) == 0 ? q0 : _mm512_srli_epi32(q0, 4 * (B)), mask));        \
        const __m512 w1 = _mm512_cvtepi32_ps(                                               \
            _mm512_and_si512((B) == 0 ? q1 : _mm512_srli_epi32(q1, 4 * (B)), mask));        \
        const __m512 w2 = _mm512_cvtepi32_ps(                                               \
            _mm512_and_si512((B) == 0 ? q2 : _mm512_srli_epi32(q2, 4 * (B)), mask));        \
        const __m512 w3 = _mm512_cvtepi32_ps(                                               \
            _mm512_and_si512((B) == 0 ? q3 : _mm512_srli_epi32(q3, 4 * (B)), mask));        \
        for (int t = 0; t < T; ++t) {                                                       \
            const float* xpt = xp + (size_t)t * xp_stride + g * 128;                        \
            const __m512 x = _mm512_loadu_ps(xpt + 16 * (B));                               \
            acc0[t] = _mm512_fmadd_ps(w0, x, acc0[t]);                                      \
            acc1[t] = _mm512_fmadd_ps(w1, x, acc1[t]);                                      \
            acc2[t] = _mm512_fmadd_ps(w2, x, acc2[t]);                                      \
            acc3[t] = _mm512_fmadd_ps(w3, x, acc3[t]);                                      \
        }                                                                                   \
    }
        W34_UNROLL_NIB4(0)
        W34_UNROLL_NIB4(1)
        W34_UNROLL_NIB4(2)
        W34_UNROLL_NIB4(3)
        W34_UNROLL_NIB4(4)
        W34_UNROLL_NIB4(5)
        W34_UNROLL_NIB4(6)
        W34_UNROLL_NIB4(7)
#undef W34_UNROLL_NIB4

        const float s0 = _cvtsh_ss(row_s0[g]);
        const float s1 = _cvtsh_ss(row_s1[g]);
        const float s2 = _cvtsh_ss(row_s2[g]);
        const float s3 = _cvtsh_ss(row_s3[g]);
        const __m512 vs0 = _mm512_set1_ps(s0);
        const __m512 vs1 = _mm512_set1_ps(s1);
        const __m512 vs2 = _mm512_set1_ps(s2);
        const __m512 vs3 = _mm512_set1_ps(s3);
#pragma unroll
        for (int t = 0; t < T; ++t) {
            const float xgv = xg[t * xg_stride + g];
            tot_0[t] = _mm512_fmadd_ps(acc0[t], vs0, tot_0[t]);
            tot_1[t] = _mm512_fmadd_ps(acc1[t], vs1, tot_1[t]);
            tot_2[t] = _mm512_fmadd_ps(acc2[t], vs2, tot_2[t]);
            tot_3[t] = _mm512_fmadd_ps(acc3[t], vs3, tot_3[t]);
            bias_0[t] += s0 * xgv;
            bias_1[t] += s1 * xgv;
            bias_2[t] += s2 * xgv;
            bias_3[t] += s3 * xgv;
        }
    }
#pragma unroll
    for (int t = 0; t < T; ++t) {
        out_0[t] = hsum512_ps(tot_0[t]) - 8.0f * bias_0[t];
        out_1[t] = hsum512_ps(tot_1[t]) - 8.0f * bias_1[t];
        out_2[t] = hsum512_ps(tot_2[t]) - 8.0f * bias_2[t];
        out_3[t] = hsum512_ps(tot_3[t]) - 8.0f * bias_3[t];
    }
}

template <int T>
__attribute__((target("avx2,fma,f16c,avx512f,avx512vl")))
static void eval_expert_slice_u4b8(
    const LayerHostDesc& desc,
    const ExpertJob& jb,
    int g_interm_start,
    int g_interm_end,
    const float* __restrict__ job_xp,
    const float* __restrict__ job_xg,
    float* __restrict__ worker_y_acc) {
    const int H = desc.hidden;
    const int I = desc.intermediate;
    const int h_words = H / 8;
    const int h_groups = H / kGroupSize;
    const int i_words = I / 8;
    const int i_groups = I / kGroupSize;

    const uint8_t* pack = desc.packs_base + (size_t)jb.le * desc.pack_stride_bytes;
    const uint32_t* gu_w = reinterpret_cast<const uint32_t*>(pack);
    const uint16_t* gu_s = reinterpret_cast<const uint16_t*>(pack + desc.gu_packed_bytes);
    const uint32_t* dn_w =
        reinterpret_cast<const uint32_t*>(pack + desc.gu_packed_bytes + desc.gu_scales_bytes);
    const uint16_t* dn_s = reinterpret_cast<const uint16_t*>(
        pack + desc.gu_packed_bytes + desc.gu_scales_bytes + desc.dn_packed_bytes);

    const int c0 = g_interm_start * kGroupSize;
    const int c1 = g_interm_end * kGroupSize;
    const int slice_groups = g_interm_end - g_interm_start;
    const int slice_cols = slice_groups * kGroupSize;

    alignas(32) float act[T * kMaxInterm];
    alignas(32) float act_xp[T * kMaxInterm];
    alignas(32) float act_xg[T * (kMaxInterm / kGroupSize)];

    int n = c0;
    if constexpr (T == 1) {
        for (; n + 2 <= c1; n += 2) {
            float g_val0[1], g_val1[1], u_val0[1], u_val1[1];
            eval_four_rows_u4b8<1>(
                gu_w + (size_t)n * h_words,
                gu_s + (size_t)n * h_groups,
                gu_w + (size_t)(n + 1) * h_words,
                gu_s + (size_t)(n + 1) * h_groups,
                gu_w + (size_t)(I + n) * h_words,
                gu_s + (size_t)(I + n) * h_groups,
                gu_w + (size_t)(I + n + 1) * h_words,
                gu_s + (size_t)(I + n + 1) * h_groups,
                job_xp,
                job_xg,
                h_groups,
                H,
                h_groups,
                g_val0,
                g_val1,
                u_val0,
                u_val1);
            const int col = n - c0;
            const float gv0 = g_val0[0];
            const float gv1 = g_val1[0];
            act[col] = (gv0 / (1.0f + std::exp(-gv0))) * u_val0[0];
            act[col + 1] = (gv1 / (1.0f + std::exp(-gv1))) * u_val1[0];
        }
    }
    for (; n < c1; ++n) {
        float g_val[T];
        float u_val[T];
        eval_two_rows_u4b8<T>(
            gu_w + (size_t)n * h_words,
            gu_s + (size_t)n * h_groups,
            gu_w + (size_t)(I + n) * h_words,
            gu_s + (size_t)(I + n) * h_groups,
            job_xp,
            job_xg,
            h_groups,
            H,
            h_groups,
            g_val,
            u_val);
        const int col = n - c0;
#pragma unroll
        for (int t = 0; t < T; ++t) {
            const float gv = g_val[t];
            act[t * slice_cols + col] = (gv / (1.0f + std::exp(-gv))) * u_val[t];
        }
    }

    for (int t = 0; t < T; ++t) {
        prepare_x_perm(
            act + t * slice_cols,
            act_xp + t * slice_cols,
            act_xg + t * slice_groups,
            slice_cols);
    }

    const size_t dn_w_slice_off = (size_t)g_interm_start * 16;
    const size_t dn_s_slice_off = (size_t)g_interm_start;

    int h = 0;
    if constexpr (T == 1) {
        for (; h + 4 <= H; h += 4) {
            float d_val0[1], d_val1[1], d_val2[1], d_val3[1];
            eval_four_rows_u4b8<1>(
                dn_w + (size_t)h * i_words + dn_w_slice_off,
                dn_s + (size_t)h * i_groups + dn_s_slice_off,
                dn_w + (size_t)(h + 1) * i_words + dn_w_slice_off,
                dn_s + (size_t)(h + 1) * i_groups + dn_s_slice_off,
                dn_w + (size_t)(h + 2) * i_words + dn_w_slice_off,
                dn_s + (size_t)(h + 2) * i_groups + dn_s_slice_off,
                dn_w + (size_t)(h + 3) * i_words + dn_w_slice_off,
                dn_s + (size_t)(h + 3) * i_groups + dn_s_slice_off,
                act_xp,
                act_xg,
                slice_groups,
                slice_cols,
                slice_groups,
                d_val0,
                d_val1,
                d_val2,
                d_val3);
            const float wt = jb.wt[0];
            float* dst = worker_y_acc + jb.tok[0] * H + h;
            dst[0] += wt * d_val0[0];
            dst[1] += wt * d_val1[0];
            dst[2] += wt * d_val2[0];
            dst[3] += wt * d_val3[0];
        }
    }
    for (; h + 2 <= H; h += 2) {
        float d_val0[T];
        float d_val1[T];
        eval_two_rows_u4b8<T>(
            dn_w + (size_t)h * i_words + dn_w_slice_off,
            dn_s + (size_t)h * i_groups + dn_s_slice_off,
            dn_w + (size_t)(h + 1) * i_words + dn_w_slice_off,
            dn_s + (size_t)(h + 1) * i_groups + dn_s_slice_off,
            act_xp,
            act_xg,
            slice_groups,
            slice_cols,
            slice_groups,
            d_val0,
            d_val1);
#pragma unroll
        for (int t = 0; t < T; ++t) {
            worker_y_acc[jb.tok[t] * H + h] += jb.wt[t] * d_val0[t];
            worker_y_acc[jb.tok[t] * H + h + 1] += jb.wt[t] * d_val1[t];
        }
    }
    for (; h < H; ++h) {
        float d_val[T];
        eval_row_u4b8<T>(
            dn_w + (size_t)h * i_words + dn_w_slice_off,
            dn_s + (size_t)h * i_groups + dn_s_slice_off,
            act_xp,
            act_xg,
            slice_groups,
            slice_cols,
            slice_groups,
            d_val);
#pragma unroll
        for (int t = 0; t < T; ++t) {
            worker_y_acc[jb.tok[t] * H + h] += jb.wt[t] * d_val[t];
        }
    }
}

__attribute__((target("avx2,fma,f16c")))
static void compute_slot_cpu(
    CpuWorkerPool* pool,
    const LayerHostDesc& desc,
    const MoeBridgeReqSlot* rq,
    MoeBridgeRespSlot* rs,
    float* worker_scratch,
    float* workers_y_base) {
    const int T = rq->num_tokens;
    const int top_k = rq->top_k;
    const int H = desc.hidden;
    const int I = desc.intermediate;
    const int total_pairs = T * top_k;

    ExpertJob jobs[kMaxPairs];
    int J = 0;
    for (int p = 0; p < total_pairs; ++p) {
        const int le = rq->local_expert_ids[p];
        if (le < 0 || le >= desc.local_num_experts) continue;
        const int tok = p / top_k;
        const float wt = rq->topk_weights[p];
        int j = 0;
        while (j < J && jobs[j].le != le) ++j;
        if (j == J) {
            jobs[J].le = le;
            jobs[J].n = 0;
            ++J;
        }
        const int slot = jobs[j].n;
        if (slot > 0 && jobs[j].tok[slot - 1] == tok) {
            jobs[j].wt[slot - 1] += wt;
        } else if (slot < kMaxTokens) {
            jobs[j].tok[slot] = tok;
            jobs[j].wt[slot] = wt;
            jobs[j].n = slot + 1;
        }
    }

    if (J == 0) {
        std::memset(rs->y_f32, 0, sizeof(float) * T * H);
        return;
    }

    // Sort jobs in descending order of routed token count (`jb.n`) so multi-token cold experts
    // execute first in the work-stealing queue (Longest-Processing-Time-First).
    for (int i = 1; i < J; ++i) {
        ExpertJob key = jobs[i];
        int k = i - 1;
        while (k >= 0 && jobs[k].n < key.n) {
            jobs[k + 1] = jobs[k];
            --k;
        }
        jobs[k + 1] = key;
    }

    // Prepare transposed activations for all active tokens `0..T` once.
    alignas(32) float tok_xp[kMaxTokens * kMaxHidden];
    alignas(32) float tok_xg[kMaxTokens * (kMaxHidden / kGroupSize)];
    const int h_groups = H / kGroupSize;
    for (int t = 0; t < T; ++t) {
        prepare_x_perm(
            rq->x_f32 + t * H,
            tok_xp + t * H,
            tok_xg + t * h_groups,
            H);
    }

    // Pack per-job `job_xp` and `job_xg` contiguously for cache-friendly multi-token access.
    const size_t per_job_xp = (size_t)kMaxTokens * H;
    const size_t per_job_xg = (size_t)kMaxTokens * h_groups;
    float* jobs_xp_base = worker_scratch;
    float* jobs_xg_base = jobs_xp_base + (size_t)J * per_job_xp;
    for (int j = 0; j < J; ++j) {
        float* j_xp = jobs_xp_base + (size_t)j * per_job_xp;
        float* j_xg = jobs_xg_base + (size_t)j * per_job_xg;
        for (int t = 0; t < jobs[j].n; ++t) {
            const int src_t = jobs[j].tok[t];
            std::memcpy(j_xp + t * H, tok_xp + src_t * H, sizeof(float) * H);
            std::memcpy(j_xg + t * h_groups, tok_xg + src_t * h_groups, sizeof(float) * h_groups);
        }
    }

    const int nw = pool->size();
    const size_t per_worker_y = (size_t)kMaxTokens * H;
    for (int w = 0; w < nw; ++w) {
        std::memset(workers_y_base + (size_t)w * per_worker_y, 0, sizeof(float) * T * H);
    }

    const int i_groups = I / kGroupSize;
    int slice_bounds[3] = {0, i_groups, i_groups};
    int num_slices = 1;
    if (J < 8 && i_groups >= 2) {
        slice_bounds[0] = 0;
        slice_bounds[1] = i_groups / 2;
        slice_bounds[2] = i_groups;
        num_slices = 2;
    }

    pool->run(J * num_slices, [&](int item, int w) {
        const int j = item / num_slices;
        const int s = item % num_slices;
        const int g0 = slice_bounds[s];
        const int g1 = slice_bounds[s + 1];
        const ExpertJob& jb = jobs[j];
        const float* j_xp = jobs_xp_base + (size_t)j * per_job_xp;
        const float* j_xg = jobs_xg_base + (size_t)j * per_job_xg;
        float* wy = workers_y_base + (size_t)w * per_worker_y;
        switch (jb.n) {
            case 1:
                eval_expert_slice_u4b8<1>(desc, jb, g0, g1, j_xp, j_xg, wy);
                break;
            case 2:
                eval_expert_slice_u4b8<2>(desc, jb, g0, g1, j_xp, j_xg, wy);
                break;
            case 3:
                eval_expert_slice_u4b8<3>(desc, jb, g0, g1, j_xp, j_xg, wy);
                break;
            default:
                eval_expert_slice_u4b8<4>(desc, jb, g0, g1, j_xp, j_xg, wy);
                break;
        }
    });

    std::memcpy(rs->y_f32, workers_y_base, sizeof(float) * T * H);
    for (int w = 1; w < nw; ++w) {
        const float* wy = workers_y_base + (size_t)w * per_worker_y;
        int i = 0;
        for (; i + 8 <= T * H; i += 8) {
            __m256 a = _mm256_loadu_ps(rs->y_f32 + i);
            __m256 b = _mm256_loadu_ps(wy + i);
            _mm256_storeu_ps(rs->y_f32 + i, _mm256_add_ps(a, b));
        }
        for (; i < T * H; ++i) {
            rs->y_f32[i] += wy[i];
        }
    }
}

struct PrefillBridgeTask {
    alignas(64) std::atomic<uint64_t> req_seq{0};
    alignas(64) std::atomic<uint64_t> resp_seq{0};
    int layer_id = 0;
    const uint16_t* x_f16 = nullptr;
    int num_tokens = 0;
    const int32_t* job_le = nullptr;
    const int32_t* job_n_tok = nullptr;
    const int32_t* job_tok_ids = nullptr;
    const float* job_weights = nullptr;
    int num_jobs = 0;
    uint16_t* out_f16 = nullptr;
};

__attribute__((target("avx2,fma,f16c")))
static void prepare_prefill_tok_x(
    const uint16_t* __restrict__ src_f16,
    float* __restrict__ dst_xp,
    float* __restrict__ dst_xg,
    int H) {
    alignas(32) float x_f32[kMaxHidden];
    int h = 0;
    for (; h + 8 <= H; h += 8) {
        const __m128i vh = _mm_loadu_si128(reinterpret_cast<const __m128i*>(src_f16 + h));
        _mm256_store_ps(x_f32 + h, _mm256_cvtph_ps(vh));
    }
    for (; h < H; ++h) {
        x_f32[h] = _cvtsh_ss(src_f16[h]);
    }
    prepare_x_perm(x_f32, dst_xp, dst_xg, H);
}

constexpr int kMaxTileTokens = kMaxPrefillJobTokens;

__attribute__((target("avx2,fma,f16c,avx512f,avx512vl")))
static inline void eval_four_rows_tile_u4b8(
    const uint32_t* __restrict__ row_w0,
    const uint16_t* __restrict__ row_s0,
    const uint32_t* __restrict__ row_w1,
    const uint16_t* __restrict__ row_s1,
    const uint32_t* __restrict__ row_w2,
    const uint16_t* __restrict__ row_s2,
    const uint32_t* __restrict__ row_w3,
    const uint16_t* __restrict__ row_s3,
    const float* __restrict__ xp,
    const float* __restrict__ xg,
    int num_groups,
    int xp_stride,
    int xg_stride,
    int n_tok,
    float* __restrict__ out_0,
    float* __restrict__ out_1,
    float* __restrict__ out_2,
    float* __restrict__ out_3) {
    const __m512i mask = _mm512_set1_epi32(0xF);
    __m512 tot_0[kMaxTileTokens];
    __m512 tot_1[kMaxTileTokens];
    __m512 tot_2[kMaxTileTokens];
    __m512 tot_3[kMaxTileTokens];
    float bias_0[kMaxTileTokens];
    float bias_1[kMaxTileTokens];
    float bias_2[kMaxTileTokens];
    float bias_3[kMaxTileTokens];
    for (int t = 0; t < n_tok; ++t) {
        tot_0[t] = _mm512_setzero_ps();
        tot_1[t] = _mm512_setzero_ps();
        tot_2[t] = _mm512_setzero_ps();
        tot_3[t] = _mm512_setzero_ps();
        bias_0[t] = 0.0f;
        bias_1[t] = 0.0f;
        bias_2[t] = 0.0f;
        bias_3[t] = 0.0f;
    }

    for (int g = 0; g < num_groups; ++g) {
        const uint32_t* gw0 = row_w0 + g * 16;
        const uint32_t* gw1 = row_w1 + g * 16;
        const uint32_t* gw2 = row_w2 + g * 16;
        const uint32_t* gw3 = row_w3 + g * 16;
        _mm_prefetch(reinterpret_cast<const char*>(gw0 + 64), _MM_HINT_T0);
        _mm_prefetch(reinterpret_cast<const char*>(gw1 + 64), _MM_HINT_T0);
        _mm_prefetch(reinterpret_cast<const char*>(gw2 + 64), _MM_HINT_T0);
        _mm_prefetch(reinterpret_cast<const char*>(gw3 + 64), _MM_HINT_T0);
        const __m512i q0 = _mm512_loadu_si512(reinterpret_cast<const void*>(gw0));
        const __m512i q1 = _mm512_loadu_si512(reinterpret_cast<const void*>(gw1));
        const __m512i q2 = _mm512_loadu_si512(reinterpret_cast<const void*>(gw2));
        const __m512i q3 = _mm512_loadu_si512(reinterpret_cast<const void*>(gw3));

        alignas(64) __m512 w0[8], w1[8], w2[8], w3[8];
#define W34_UNPACK4_TILE_NIB(B)                                                             \
    w0[B] = _mm512_cvtepi32_ps(                                                             \
        _mm512_and_si512((B) == 0 ? q0 : _mm512_srli_epi32(q0, 4 * (B)), mask));            \
    w1[B] = _mm512_cvtepi32_ps(                                                             \
        _mm512_and_si512((B) == 0 ? q1 : _mm512_srli_epi32(q1, 4 * (B)), mask));            \
    w2[B] = _mm512_cvtepi32_ps(                                                             \
        _mm512_and_si512((B) == 0 ? q2 : _mm512_srli_epi32(q2, 4 * (B)), mask));            \
    w3[B] = _mm512_cvtepi32_ps(                                                             \
        _mm512_and_si512((B) == 0 ? q3 : _mm512_srli_epi32(q3, 4 * (B)), mask));
        W34_UNPACK4_TILE_NIB(0)
        W34_UNPACK4_TILE_NIB(1)
        W34_UNPACK4_TILE_NIB(2)
        W34_UNPACK4_TILE_NIB(3)
        W34_UNPACK4_TILE_NIB(4)
        W34_UNPACK4_TILE_NIB(5)
        W34_UNPACK4_TILE_NIB(6)
        W34_UNPACK4_TILE_NIB(7)
#undef W34_UNPACK4_TILE_NIB

        const float s0 = _cvtsh_ss(row_s0[g]);
        const float s1 = _cvtsh_ss(row_s1[g]);
        const float s2 = _cvtsh_ss(row_s2[g]);
        const float s3 = _cvtsh_ss(row_s3[g]);
        const __m512 vs0 = _mm512_set1_ps(s0);
        const __m512 vs1 = _mm512_set1_ps(s1);
        const __m512 vs2 = _mm512_set1_ps(s2);
        const __m512 vs3 = _mm512_set1_ps(s3);

        for (int t = 0; t < n_tok; ++t) {
            const float* xpt = xp + (size_t)t * xp_stride + g * 128;
            __m512 acc0 = _mm512_setzero_ps();
            __m512 acc1 = _mm512_setzero_ps();
            __m512 acc2 = _mm512_setzero_ps();
            __m512 acc3 = _mm512_setzero_ps();
#pragma unroll
            for (int b = 0; b < 8; ++b) {
                const __m512 x = _mm512_loadu_ps(xpt + 16 * b);
                acc0 = _mm512_fmadd_ps(w0[b], x, acc0);
                acc1 = _mm512_fmadd_ps(w1[b], x, acc1);
                acc2 = _mm512_fmadd_ps(w2[b], x, acc2);
                acc3 = _mm512_fmadd_ps(w3[b], x, acc3);
            }
            const float xgv = xg[t * xg_stride + g];
            tot_0[t] = _mm512_fmadd_ps(acc0, vs0, tot_0[t]);
            tot_1[t] = _mm512_fmadd_ps(acc1, vs1, tot_1[t]);
            tot_2[t] = _mm512_fmadd_ps(acc2, vs2, tot_2[t]);
            tot_3[t] = _mm512_fmadd_ps(acc3, vs3, tot_3[t]);
            bias_0[t] += s0 * xgv;
            bias_1[t] += s1 * xgv;
            bias_2[t] += s2 * xgv;
            bias_3[t] += s3 * xgv;
        }
    }
    for (int t = 0; t < n_tok; ++t) {
        out_0[t] = hsum512_ps(tot_0[t]) - 8.0f * bias_0[t];
        out_1[t] = hsum512_ps(tot_1[t]) - 8.0f * bias_1[t];
        out_2[t] = hsum512_ps(tot_2[t]) - 8.0f * bias_2[t];
        out_3[t] = hsum512_ps(tot_3[t]) - 8.0f * bias_3[t];
    }
}

__attribute__((target("avx2,fma,f16c,avx512f,avx512vl")))
static void eval_expert_prefill_u4b8(
    const LayerHostDesc& desc,
    int le,
    int n_tok,
    const float* __restrict__ job_xp,
    const float* __restrict__ job_xg,
    const float* __restrict__ job_wt,
    float* __restrict__ job_out) {
    const int H = desc.hidden;
    const int I = desc.intermediate;
    const int h_words = H / 8;
    const int h_groups = H / kGroupSize;
    const int i_words = I / 8;
    const int i_groups = I / kGroupSize;

    const uint8_t* pack = desc.packs_base + (size_t)le * desc.pack_stride_bytes;
    const uint32_t* gu_w = reinterpret_cast<const uint32_t*>(pack);
    const uint16_t* gu_s = reinterpret_cast<const uint16_t*>(pack + desc.gu_packed_bytes);
    const uint32_t* dn_w =
        reinterpret_cast<const uint32_t*>(pack + desc.gu_packed_bytes + desc.gu_scales_bytes);
    const uint16_t* dn_s = reinterpret_cast<const uint16_t*>(
        pack + desc.gu_packed_bytes + desc.gu_scales_bytes + desc.dn_packed_bytes);

    alignas(32) float act[kMaxPrefillJobTokens * kMaxInterm];
    alignas(32) float act_xp[kMaxPrefillJobTokens * kMaxInterm];
    alignas(32) float act_xg[kMaxPrefillJobTokens * (kMaxInterm / kGroupSize)];

    float g_val0[kMaxTileTokens], g_val1[kMaxTileTokens];
    float u_val0[kMaxTileTokens], u_val1[kMaxTileTokens];
    for (int n = 0; n + 2 <= I; n += 2) {
        eval_four_rows_tile_u4b8(
            gu_w + (size_t)n * h_words,
            gu_s + (size_t)n * h_groups,
            gu_w + (size_t)(n + 1) * h_words,
            gu_s + (size_t)(n + 1) * h_groups,
            gu_w + (size_t)(I + n) * h_words,
            gu_s + (size_t)(I + n) * h_groups,
            gu_w + (size_t)(I + n + 1) * h_words,
            gu_s + (size_t)(I + n + 1) * h_groups,
            job_xp,
            job_xg,
            h_groups,
            H,
            h_groups,
            n_tok,
            g_val0,
            g_val1,
            u_val0,
            u_val1);
        for (int t = 0; t < n_tok; ++t) {
            const float gv0 = g_val0[t];
            const float gv1 = g_val1[t];
            act[t * I + n] = (gv0 / (1.0f + std::exp(-gv0))) * u_val0[t];
            act[t * I + n + 1] = (gv1 / (1.0f + std::exp(-gv1))) * u_val1[t];
        }
    }

    for (int t = 0; t < n_tok; ++t) {
        prepare_x_perm(
            act + t * I,
            act_xp + (size_t)t * I,
            act_xg + t * i_groups,
            I);
    }

    float d_val0[kMaxTileTokens], d_val1[kMaxTileTokens];
    float d_val2[kMaxTileTokens], d_val3[kMaxTileTokens];
    for (int h = 0; h + 4 <= H; h += 4) {
        eval_four_rows_tile_u4b8(
            dn_w + (size_t)h * i_words,
            dn_s + (size_t)h * i_groups,
            dn_w + (size_t)(h + 1) * i_words,
            dn_s + (size_t)(h + 1) * i_groups,
            dn_w + (size_t)(h + 2) * i_words,
            dn_s + (size_t)(h + 2) * i_groups,
            dn_w + (size_t)(h + 3) * i_words,
            dn_s + (size_t)(h + 3) * i_groups,
            act_xp,
            act_xg,
            i_groups,
            I,
            i_groups,
            n_tok,
            d_val0,
            d_val1,
            d_val2,
            d_val3);
        for (int t = 0; t < n_tok; ++t) {
            const float wt = job_wt[t];
            float* dst = job_out + (size_t)t * H + h;
            dst[0] = wt * d_val0[t];
            dst[1] = wt * d_val1[t];
            dst[2] = wt * d_val2[t];
            dst[3] = wt * d_val3[t];
        }
    }
}

__attribute__((target("avx2,fma,f16c")))
static void reduce_prefill_tok_y(
    const int32_t* __restrict__ contrib_jt,
    int n_contrib,
    const float* __restrict__ job_out,
    uint16_t* __restrict__ dst_f16,
    int H) {
    alignas(32) float acc[kMaxHidden];
    if (n_contrib <= 0) {
        std::memset(dst_f16, 0, sizeof(uint16_t) * H);
        return;
    }
    std::memcpy(acc, job_out + (size_t)contrib_jt[0] * H, sizeof(float) * H);
    for (int idx = 1; idx < n_contrib; ++idx) {
        const int jt = contrib_jt[idx];
        const float* src = job_out + (size_t)jt * H;
        int h = 0;
        for (; h + 8 <= H; h += 8) {
            __m256 a = _mm256_load_ps(acc + h);
            __m256 b = _mm256_loadu_ps(src + h);
            _mm256_store_ps(acc + h, _mm256_add_ps(a, b));
        }
        for (; h < H; ++h) {
            acc[h] += src[h];
        }
    }
    int h = 0;
    for (; h + 8 <= H; h += 8) {
        const __m256 v = _mm256_load_ps(acc + h);
        const __m128i vh = _mm256_cvtps_ph(v, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
        _mm_storeu_si128(reinterpret_cast<__m128i*>(dst_f16 + h), vh);
    }
    for (; h < H; ++h) {
        dst_f16[h] = _cvtss_sh(acc[h], _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
    }
}

__attribute__((target("avx2,fma,f16c,avx512f,avx512vl")))
static void compute_prefill_cpu(
    CpuWorkerPool* pool,
    const LayerHostDesc& desc,
    const PrefillBridgeTask& task,
    std::vector<float>& prefill_xp,
    std::vector<float>& prefill_xg,
    std::vector<float>& job_out) {
    const int H = desc.hidden;
    const int h_groups = H / kGroupSize;
    const int J = task.num_jobs;
    const int M = task.num_tokens;

    if (J <= 0 || M <= 0) {
        if (M > 0) std::memset(task.out_f16, 0, sizeof(uint16_t) * (size_t)M * H);
        return;
    }

    std::vector<int32_t> tok_to_compact(M, -1);
    std::vector<int32_t> active_toks;
    active_toks.reserve(std::min(M, J * kMaxPrefillJobTokens));
    std::vector<int32_t> job_row_off(J + 1, 0);
    for (int j = 0; j < J; ++j) {
        const int le = task.job_le[j];
        const int n = std::min(kMaxPrefillJobTokens, std::max(0, task.job_n_tok[j]));
        job_row_off[j + 1] = job_row_off[j] + ((le >= 0 && le < desc.local_num_experts) ? n : 0);
        for (int t = 0; t < n; ++t) {
            const int tok = task.job_tok_ids[j * kMaxPrefillJobTokens + t];
            if (tok >= 0 && tok < M && tok_to_compact[tok] < 0) {
                tok_to_compact[tok] = (int32_t)active_toks.size();
                active_toks.push_back(tok);
            }
        }
    }
    const int n_act = (int)active_toks.size();
    const int total_job_rows = job_row_off[J];
    if (n_act == 0 || total_job_rows == 0) {
        std::memset(task.out_f16, 0, sizeof(uint16_t) * (size_t)M * H);
        return;
    }

    if (prefill_xp.size() < (size_t)n_act * H) {
        prefill_xp.resize((size_t)n_act * H);
    }
    if (prefill_xg.size() < (size_t)n_act * h_groups) {
        prefill_xg.resize((size_t)n_act * h_groups);
    }
    if (job_out.size() < (size_t)total_job_rows * H) {
        job_out.resize((size_t)total_job_rows * H);
    }

    pool->run(M, [&](int tok, int) {
        const int c_idx = tok_to_compact[tok];
        if (c_idx >= 0) {
            prepare_prefill_tok_x(
                task.x_f16 + (size_t)tok * H,
                prefill_xp.data() + (size_t)c_idx * H,
                prefill_xg.data() + (size_t)c_idx * h_groups,
                H);
        } else {
            std::memset(task.out_f16 + (size_t)tok * H, 0, sizeof(uint16_t) * H);
        }
    });

    pool->run(J, [&](int j, int) {
        const int le = task.job_le[j];
        const int n = std::min(kMaxPrefillJobTokens, std::max(0, task.job_n_tok[j]));
        if (le < 0 || le >= desc.local_num_experts || n <= 0) return;

        alignas(32) float local_xp[kMaxPrefillJobTokens * kMaxHidden];
        alignas(32) float local_xg[kMaxPrefillJobTokens * (kMaxHidden / kGroupSize)];
        alignas(32) float local_wt[kMaxPrefillJobTokens];
        for (int t = 0; t < n; ++t) {
            const int tok = task.job_tok_ids[j * kMaxPrefillJobTokens + t];
            const int c_idx = tok_to_compact[tok];
            local_wt[t] = task.job_weights[j * kMaxPrefillJobTokens + t];
            std::memcpy(
                local_xp + (size_t)t * H,
                prefill_xp.data() + (size_t)c_idx * H,
                sizeof(float) * H);
            std::memcpy(
                local_xg + (size_t)t * h_groups,
                prefill_xg.data() + (size_t)c_idx * h_groups,
                sizeof(float) * h_groups);
        }
        float* j_out = job_out.data() + (size_t)job_row_off[j] * H;
        eval_expert_prefill_u4b8(desc, le, n, local_xp, local_xg, local_wt, j_out);
    });

    std::vector<int32_t> tok_counts(n_act + 1, 0);
    for (int j = 0; j < J; ++j) {
        const int le = task.job_le[j];
        const int n = std::min(kMaxPrefillJobTokens, std::max(0, task.job_n_tok[j]));
        if (le < 0 || le >= desc.local_num_experts || n <= 0) continue;
        for (int t = 0; t < n; ++t) {
            const int tok = task.job_tok_ids[j * kMaxPrefillJobTokens + t];
            const int c_idx = tok_to_compact[tok];
            if (c_idx >= 0) {
                ++tok_counts[c_idx + 1];
            }
        }
    }
    for (int c = 0; c < n_act; ++c) {
        tok_counts[c + 1] += tok_counts[c];
    }
    std::vector<int32_t> tok_cursor(tok_counts.begin(), tok_counts.begin() + n_act);
    std::vector<int32_t> contrib_jt(tok_counts[n_act]);
    for (int j = 0; j < J; ++j) {
        const int le = task.job_le[j];
        const int n = std::min(kMaxPrefillJobTokens, std::max(0, task.job_n_tok[j]));
        if (le < 0 || le >= desc.local_num_experts || n <= 0) continue;
        const int base_row = job_row_off[j];
        for (int t = 0; t < n; ++t) {
            const int tok = task.job_tok_ids[j * kMaxPrefillJobTokens + t];
            const int c_idx = tok_to_compact[tok];
            if (c_idx >= 0) {
                contrib_jt[tok_cursor[c_idx]++] = base_row + t;
            }
        }
    }

    pool->run(n_act, [&](int c_idx, int) {
        const int tok = active_toks[c_idx];
        const int start = tok_counts[c_idx];
        const int count = tok_counts[c_idx + 1] - start;
        reduce_prefill_tok_y(
            contrib_jt.data() + start,
            count,
            job_out.data(),
            task.out_f16 + (size_t)tok * H,
            H);
    });
}

#endif // __x86_64__

struct GlobalBridgeState {
    std::mutex mu;
    bool initialized = false;
    MoeBridgeReqSlot* h_req = nullptr;
    MoeBridgeRespSlot* h_resp = nullptr;
    MoeBridgeReqSlot* d_req = nullptr;
    MoeBridgeRespSlot* d_resp = nullptr;
    LayerHostDesc layers[kMaxLayers];
    PrefillBridgeTask prefill_task;
    std::atomic<bool> stop{false};
    std::thread server_thread;
};

static GlobalBridgeState g_bridge;

static int ensure_bridge_started() {
    std::lock_guard<std::mutex> lock(g_bridge.mu);
    if (g_bridge.initialized) return 0;

#if defined(__x86_64__) || defined(_M_X64)
#if defined(__GNUC__) || defined(__clang__)
    if (!__builtin_cpu_supports("avx2") || !__builtin_cpu_supports("fma") ||
        !__builtin_cpu_supports("f16c") || !__builtin_cpu_supports("avx512f") ||
        !__builtin_cpu_supports("avx512vl")) {
        return -1;
    }
#endif
#else
    return -1;
#endif

    void* req_ptr = nullptr;
    void* resp_ptr = nullptr;
    cudaError_t e1 = cudaHostAlloc(
        &req_ptr, sizeof(MoeBridgeReqSlot) * kMaxLayers, cudaHostAllocMapped);
    if (e1 != cudaSuccess) return (int)e1;
    cudaError_t e2 = cudaHostAlloc(
        &resp_ptr, sizeof(MoeBridgeRespSlot) * kMaxLayers, cudaHostAllocMapped);
    if (e2 != cudaSuccess) {
        cudaFreeHost(req_ptr);
        return (int)e2;
    }
    std::memset(req_ptr, 0, sizeof(MoeBridgeReqSlot) * kMaxLayers);
    std::memset(resp_ptr, 0, sizeof(MoeBridgeRespSlot) * kMaxLayers);

    void* d_req_ptr = nullptr;
    void* d_resp_ptr = nullptr;
    cudaError_t e3 = cudaHostGetDevicePointer(&d_req_ptr, req_ptr, 0);
    cudaError_t e4 = cudaHostGetDevicePointer(&d_resp_ptr, resp_ptr, 0);
    if (e3 != cudaSuccess || e4 != cudaSuccess || !d_req_ptr || !d_resp_ptr) {
        cudaFreeHost(resp_ptr);
        cudaFreeHost(req_ptr);
        return (int)(e3 != cudaSuccess ? e3 : (e4 != cudaSuccess ? e4 : cudaErrorInvalidValue));
    }

    g_bridge.h_req = static_cast<MoeBridgeReqSlot*>(req_ptr);
    g_bridge.h_resp = static_cast<MoeBridgeRespSlot*>(resp_ptr);
    g_bridge.d_req = static_cast<MoeBridgeReqSlot*>(d_req_ptr);
    g_bridge.d_resp = static_cast<MoeBridgeRespSlot*>(d_resp_ptr);
    g_bridge.stop.store(false, std::memory_order_relaxed);

#if defined(__x86_64__) || defined(_M_X64)
    const int hw = (int)std::thread::hardware_concurrency();
    const int num_workers = (hw >= 32) ? 16 : std::max(1, std::min(hw, 16));
    g_bridge.server_thread = std::thread([num_workers] {
        auto* pool = new CpuWorkerPool(num_workers);
        std::vector<float> scratch(4 << 20, 0.0f);
        std::vector<float> workers_y((size_t)num_workers * kMaxTokens * kMaxHidden, 0.0f);
        std::vector<float> prefill_xp;
        std::vector<float> prefill_xg;
        std::vector<float> prefill_job_out;
        int64_t seen[kMaxLayers] = {0};
        uint64_t seen_prefill = 0;
        while (!g_bridge.stop.load(std::memory_order_relaxed)) {
            bool any = false;
            const uint64_t pq = g_bridge.prefill_task.req_seq.load(std::memory_order_acquire);
            if (pq != seen_prefill) {
                seen_prefill = pq;
                any = true;
                const int lid = g_bridge.prefill_task.layer_id;
                if (lid >= 0 && lid < kMaxLayers && g_bridge.layers[lid].packs_base != nullptr) {
                    compute_prefill_cpu(
                        pool,
                        g_bridge.layers[lid],
                        g_bridge.prefill_task,
                        prefill_xp,
                        prefill_xg,
                        prefill_job_out);
                }
                g_bridge.prefill_task.resp_seq.store(pq, std::memory_order_release);
            }
            for (int s = 0; s < kMaxLayers; ++s) {
                MoeBridgeReqSlot* rq = &g_bridge.h_req[s];
                const int64_t q = __atomic_load_n(&rq->seq, __ATOMIC_ACQUIRE);
                if (q == seen[s]) continue;
                seen[s] = q;
                any = true;
                MoeBridgeRespSlot* rs = &g_bridge.h_resp[s];
                const LayerHostDesc& desc = g_bridge.layers[s];
                const int T = rq->num_tokens;
                const int H = desc.hidden;
                const bool did_compute =
                    (desc.packs_base != nullptr && T >= 1 && T <= kMaxTokens);
                if (did_compute) {
                    compute_slot_cpu(pool, desc, rq, rs, scratch.data(), workers_y.data());
                } else {
                    std::memset(
                        rs->y_f32,
                        0,
                        sizeof(float) * std::max(1, T) * std::max(1, rq->hidden));
                }
                __atomic_store_n(&rs->seq, q, __ATOMIC_RELEASE);
            }
            if (!any) {
                _mm_pause();
            }
        }
    });
    g_bridge.server_thread.detach();
#endif

    g_bridge.initialized = true;
    return 0;
}

// 1-CTA kernel that splits active experts into:
//   1. Static hot experts (`0 <= slot < k_static`): run on GPU in `cache.slots[0..k_static)`.
//   2. GPU PCIe cold share (`n_gpu_cold <= k_hot - k_static`): pulled over PCIe into `cache.slots[k_static..k_hot)`
//      and run on GPU in the same `launch_moe_w4a16` call without evicting any static hot expert.
//   3. CPU DDR5 cold share (`n_cpu_cold = n_cold - n_gpu_cold`): published via `st.release.sys.global.b64`
//      to `req_slot` for parallel execution on the 16-core AVX2/FMA worker pool.
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
    }
    __syncthreads();

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

extern "C" int w34_moe_cpu_bridge_register_layer(
    int layer_id,
    const void* host_packs_base,
    size_t pack_stride_bytes,
    int local_num_experts,
    int intermediate,
    int hidden,
    int group_size) {
    if (layer_id < 0 || layer_id >= w34_moe_bridge::kMaxLayers || !host_packs_base ||
        local_num_experts <= 0 || intermediate <= 0 ||
        intermediate > w34_moe_bridge::kMaxInterm || hidden <= 0 ||
        hidden > w34_moe_bridge::kMaxHidden || group_size != w34_moe_bridge::kGroupSize) {
        return -1;
    }
    int rc = w34_moe_bridge::ensure_bridge_started();
    if (rc != 0) return rc;

    std::lock_guard<std::mutex> lock(w34_moe_bridge::g_bridge.mu);
    auto& d = w34_moe_bridge::g_bridge.layers[layer_id];
    d.packs_base = static_cast<const uint8_t*>(host_packs_base);
    d.pack_stride_bytes = pack_stride_bytes;
    d.gu_packed_bytes = (size_t)(2 * intermediate) * (size_t)(hidden / 8) * sizeof(uint32_t);
    d.gu_scales_bytes = (size_t)(2 * intermediate) * (size_t)(hidden / group_size) * sizeof(uint16_t);
    d.dn_packed_bytes = (size_t)hidden * (size_t)(intermediate / 8) * sizeof(uint32_t);
    d.dn_scales_bytes = (size_t)hidden * (size_t)(intermediate / group_size) * sizeof(uint16_t);
    d.local_num_experts = local_num_experts;
    d.intermediate = intermediate;
    d.hidden = hidden;
    d.group_size = group_size;
    return 0;
}

extern "C" int w34_moe_hybrid_submit_and_pull(
    const w34_exec* ex,
    int layer_id,
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
    if (!ex || layer_id < 0 || layer_id >= w34_moe_bridge::kMaxLayers || !x_f16 ||
        !global_topk_ids || !topk_weights || !expert_to_slot || !remapped_topk_ids ||
        !miss_local_experts || !miss_victim_slots || !num_misses_buf || !mapped_host_packs ||
        !gate_up_packed_base || !gate_up_scales_base || !down_packed_base || !down_scales_base ||
        num_tokens <= 0 || num_tokens > w34_moe_bridge::kMaxTokens || top_k <= 0 ||
        top_k > w34_moe_bridge::kMaxTopK || k_static <= 0 || k_hot < k_static ||
        !w34_moe_bridge::g_bridge.initialized) {
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
        &w34_moe_bridge::g_bridge.d_req[layer_id],
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
    int layer_id,
    void* out_f16,
    const int32_t* num_misses_buf,
    int num_tokens,
    int k_hidden) {
    if (!ex || layer_id < 0 || layer_id >= w34_moe_bridge::kMaxLayers || !out_f16 ||
        !num_misses_buf || num_tokens <= 0 || k_hidden <= 0 || (k_hidden & 3) != 0 ||
        !w34_moe_bridge::g_bridge.initialized) {
        return (int)cudaErrorInvalidValue;
    }
    const int total_elems = num_tokens * k_hidden;
    const int n4 = total_elems / 4;
    const int blocks = (n4 + 255) / 256;
    cudaStream_t stream = (cudaStream_t)ex->stream;
    w34_moe_bridge::moe_hybrid_wait_and_add_kernel<<<blocks, 256, 0, stream>>>(
        static_cast<__half*>(out_f16),
        num_misses_buf,
        &w34_moe_bridge::g_bridge.d_req[layer_id],
        &w34_moe_bridge::g_bridge.d_resp[layer_id],
        total_elems);
    return (int)cudaGetLastError();
}

extern "C" int w34_moe_cpu_eval_prefill_batch(
    int layer_id,
    const uint16_t* x_f16,
    int num_tokens,
    const int32_t* job_le,
    const int32_t* job_n_tok,
    const int32_t* job_tok_ids,
    const float* job_weights,
    int num_jobs,
    uint16_t* out_f16) {
    if (layer_id < 0 || layer_id >= w34_moe_bridge::kMaxLayers || !x_f16 || num_tokens <= 0 ||
        !job_le || !job_n_tok || !job_tok_ids || !job_weights || num_jobs <= 0 || !out_f16 ||
        !w34_moe_bridge::g_bridge.initialized) {
        return -1;
    }
    auto& pt = w34_moe_bridge::g_bridge.prefill_task;
    pt.layer_id = layer_id;
    pt.x_f16 = x_f16;
    pt.num_tokens = num_tokens;
    pt.job_le = job_le;
    pt.job_n_tok = job_n_tok;
    pt.job_tok_ids = job_tok_ids;
    pt.job_weights = job_weights;
    pt.num_jobs = num_jobs;
    pt.out_f16 = out_f16;
    const uint64_t seq = pt.req_seq.load(std::memory_order_relaxed) + 1;
    pt.req_seq.store(seq, std::memory_order_release);
    uint64_t spins = 0;
    const auto start = std::chrono::steady_clock::now();
    while (pt.resp_seq.load(std::memory_order_acquire) != seq) {
#if defined(__x86_64__) || defined(_M_X64)
        _mm_pause();
#endif
        if ((++spins & 0xFFFFu) == 0u) {
            if (std::chrono::steady_clock::now() - start > std::chrono::seconds(30)) {
                return -1;
            }
        }
    }
    return 0;
}
