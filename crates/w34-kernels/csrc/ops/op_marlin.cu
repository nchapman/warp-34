// Marlin W4A16 & W8A16 Repack and GEMM kernels for warp-34.
// Donor: vLLM 0.31.0 (8352b2427) marlin_template.h + gptq_marlin_repack.cu.
#include "../w34_cuda.h"

#include <cuda_runtime.h>
#include <algorithm>
#include <cstdint>

#define MARLIN_NAMESPACE_NAME marlin
#include "../marlin/kernel.h"
#include "../marlin/marlin_template.h"

namespace marlin {
static constexpr int kF16 = vllm::ST_ID_F16;
static constexpr int kU4B8 = vllm::ST_ID_U4B8;
static constexpr int kU8B128 = vllm::ST_ID_U8B128;
}  // namespace marlin

using marlin::Marlin;

namespace w34_marlin {

static constexpr int repack_threads = 256;
static constexpr int repack_stages = 8;
static constexpr int tile_k_size = 16;
static constexpr int tile_n_size = tile_k_size * 4;  // 64
static constexpr int stage_int4s = tile_n_size;      // 64

__host__ __device__ __forceinline__ int div_ceil(int a, int b) {
    return (a + b - 1) / b;
}

__device__ __forceinline__ void cp_async_fence() {
    asm volatile("cp.async.commit_group;");
}

template <int N>
__device__ __forceinline__ void cp_async_wait() {
    asm volatile("cp.async.wait_group %0;" ::"n"(N));
}

__device__ __forceinline__ void cp_async4(void* smem_ptr, const void* gmem_ptr) {
    const unsigned dst = (unsigned)__cvta_generic_to_shared(smem_ptr);
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" ::"r"(dst), "l"(gmem_ptr));
}

template <int const num_threads, int const num_bits>
__global__ void gptq_marlin_repack_n_major_kernel(
    uint32_t const* __restrict__ b_q_weight_ptr,
    uint32_t* __restrict__ out_ptr,
    int size_k,
    int size_n) {
    constexpr int pack_factor = 32 / num_bits;
    constexpr int tile_k = tile_k_size;

    int k_pairs = size_k / (tile_k * 2);
    int n_tiles = size_n / tile_n_size;
    int block_k_pairs = div_ceil(k_pairs, gridDim.x);

    int start_k_pair = blockIdx.x * block_k_pairs;
    if (start_k_pair >= k_pairs) return;
    int finish_k_pair = min(start_k_pair + block_k_pairs, k_pairs);

    auto wait_for_stage = [&]() {
        cp_async_wait<repack_stages - 2>();
        __syncthreads();
    };

    extern __shared__ int4 sh[];

    auto fetch_to_shared = [&](int pipe, int k_pair, int n_tile_id) {
        if (n_tile_id >= n_tiles) {
            cp_async_fence();
            return;
        }
        int first_n = n_tile_id * tile_n_size;
        int first_word = k_pair * 4;
        int4* sh_ptr = sh + stage_int4s * pipe;
        if (threadIdx.x < stage_int4s) {
            cp_async4(
                &sh_ptr[threadIdx.x],
                reinterpret_cast<int4 const*>(
                    &b_q_weight_ptr[(size_t)(first_n + threadIdx.x) * (size_k / pack_factor) + first_word]));
        }
        cp_async_fence();
    };

    auto repack_tile = [&](int pipe, int k_pair, int n_tile_id, int sub) {
        if (n_tile_id >= n_tiles) return;
        int warp_id = threadIdx.x / 32;
        int th_id = threadIdx.x % 32;
        if (warp_id >= 4) return;

        int tc_col = th_id / 4;
        int tc_row = (th_id % 4) * 2;
        constexpr int tc_offsets[4] = {0, 1, 8, 9};
        int cur_n = warp_id * 16 + tc_col;
        constexpr uint32_t mask = (1u << num_bits) - 1u;

        uint32_t* sh_stage_int_ptr = reinterpret_cast<uint32_t*>(sh + stage_int4s * pipe);
        constexpr int tile_ints = tile_k / pack_factor;

        uint32_t b1_vals[tile_ints];
        uint32_t b2_vals[tile_ints];
#pragma unroll
        for (int i = 0; i < tile_ints; i++) {
            b1_vals[i] = sh_stage_int_ptr[cur_n * 4 + sub * 2 + i];
            b2_vals[i] = sh_stage_int_ptr[(cur_n + 8) * 4 + sub * 2 + i];
        }

        uint32_t vals[8];
#pragma unroll
        for (int i = 0; i < 4; i++) {
            int cur_elem = tc_row + tc_offsets[i];
            int cur_int = cur_elem / pack_factor;
            int cur_pos = cur_elem % pack_factor;
            vals[i] = (b1_vals[cur_int] >> (cur_pos * num_bits)) & mask;
            vals[4 + i] = (b2_vals[cur_int] >> (cur_pos * num_bits)) & mask;
        }

        constexpr int tile_out_size = tile_k * tile_n_size / pack_factor;
        int k_tile_id = k_pair * 2 + sub;
        int out_offset = (k_tile_id * n_tiles + n_tile_id) * tile_out_size;
        int pack_idx[8] = {0, 2, 4, 6, 1, 3, 5, 7};

        uint32_t res = 0;
#pragma unroll
        for (int i = 0; i < 8; i++) {
            res |= vals[pack_idx[i]] << (i * 4);
        }
        out_ptr[out_offset + th_id * 4 + warp_id] = res;
    };

    auto start_pipes = [&](int k_pair, int n_tile_id) {
#pragma unroll
        for (int pipe = 0; pipe < repack_stages - 1; pipe++) {
            fetch_to_shared(pipe, k_pair, n_tile_id + pipe);
        }
        wait_for_stage();
    };

    for (int k_pair = start_k_pair; k_pair < finish_k_pair; k_pair++) {
        int n_tile_id = 0;
        start_pipes(k_pair, n_tile_id);
        while (n_tile_id < n_tiles) {
#pragma unroll
            for (int pipe = 0; pipe < repack_stages; pipe++) {
                fetch_to_shared((pipe + repack_stages - 1) % repack_stages, k_pair,
                                n_tile_id + pipe + repack_stages - 1);
                repack_tile(pipe, k_pair, n_tile_id + pipe, 0);
                repack_tile(pipe, k_pair, n_tile_id + pipe, 1);
                wait_for_stage();
            }
            n_tile_id += repack_stages;
        }
    }
}

__global__ void zero_ints_kernel(int32_t* p, int64_t n) {
    int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) p[i] = 0;
}

typedef struct {
    int thread_k;
    int thread_n;
    int num_threads;
} thread_config_t;

static thread_config_t small_batch_thread_configs[] = {
    {128, 128, 256}, {64, 128, 128}, {128, 64, 128}};

static thread_config_t large_batch_thread_configs[] = {
    {64, 256, 256}, {64, 128, 128}, {128, 64, 128}};

static int get_scales_cache_size(thread_config_t const& th_config, int group_size, int stages) {
    int tb_n = th_config.thread_n;
    int tb_k = th_config.thread_k;
    int tb_groups = group_size == -1 ? 1 : div_ceil(tb_k, group_size);
    return tb_groups * tb_n * 2 * stages;
}

static int get_kernel_cache_size(thread_config_t const& th_config, int thread_m_blocks,
                                 int num_bits, int group_size, int stages) {
    int pack_factor = 32 / num_bits;
    int tb_k = th_config.thread_k;
    int tb_n = th_config.thread_n;
    int tb_m = thread_m_blocks * 16;
    int sh_a_size = stages * (tb_m * tb_k) * 2;
    int sh_b_size = stages * (tb_k * tb_n / pack_factor) * 4;
    int sh_red_size = tb_m * (tb_n + 8) * 2;
    int sh_bias_size = tb_n * 2;
    int tmp_size = (sh_b_size > sh_red_size ? sh_red_size : sh_b_size) + sh_bias_size;
    tmp_size = std::max(std::max(sh_b_size, sh_red_size), tmp_size);
    int sh_s_size = get_scales_cache_size(th_config, group_size, stages);
    return tmp_size + sh_a_size + sh_s_size;
}

static bool is_valid_config(thread_config_t const& th_config, int thread_m_blocks,
                            int prob_m, int prob_n, int prob_k, int num_bits,
                            int group_size, int stages, int max_shared_mem) {
    (void)prob_m;
    if (th_config.thread_k == -1 || th_config.thread_n == -1 || th_config.num_threads == -1) {
        return false;
    }
    if (prob_k % th_config.thread_k != 0 || prob_n % th_config.thread_n != 0) {
        return false;
    }
    if (th_config.thread_n < marlin::min_thread_n || th_config.thread_k < marlin::min_thread_k) {
        return false;
    }
    int cache_size = get_kernel_cache_size(th_config, thread_m_blocks, num_bits, group_size, stages);
    return cache_size <= max_shared_mem;
}

typedef void (*MarlinFuncPtr)(MARLIN_KERNEL_PARAMS);

#define MARLIN_CASE_F16U4(T_K, T_N, THR, M_BLOCKS, M8, GBX)                   \
    if (th_config.thread_k == (T_K) && th_config.thread_n == (T_N) &&         \
        th_config.num_threads == (THR) && thread_m_blocks == (M_BLOCKS) &&    \
        m_block_size_8 == (M8)) {                                             \
        return &Marlin<marlin::kF16, marlin::kU4B8, marlin::kF16,             \
                       marlin::kF16, (THR), (M_BLOCKS), (T_N) / 16,           \
                       (T_K) / 16, (M8), 4, (GBX), false>;                    \
    }

#define MARLIN_CASE_F16U8(T_K, T_N, THR, M_BLOCKS, M8, GBX)                   \
    if (th_config.thread_k == (T_K) && th_config.thread_n == (T_N) &&         \
        th_config.num_threads == (THR) && thread_m_blocks == (M_BLOCKS) &&    \
        m_block_size_8 == (M8)) {                                             \
        return &Marlin<marlin::kF16, marlin::kU8B128, marlin::kF16,           \
                       marlin::kF16, (THR), (M_BLOCKS), (T_N) / 16,           \
                       (T_K) / 16, (M8), 4, (GBX), false>;                    \
    }

static MarlinFuncPtr get_marlin_kernel_f16u4(thread_config_t const& th_config,
                                             int thread_m_blocks,
                                             bool m_block_size_8,
                                             int group_blocks) {
    if (group_blocks == 2) {
        MARLIN_CASE_F16U4(128, 128, 256, 1, false, 2)
        MARLIN_CASE_F16U4(64, 128, 128, 1, false, 2)
        MARLIN_CASE_F16U4(128, 64, 128, 1, false, 2)
        MARLIN_CASE_F16U4(128, 128, 256, 1, true, 2)
        MARLIN_CASE_F16U4(64, 128, 128, 1, true, 2)
        MARLIN_CASE_F16U4(128, 64, 128, 1, true, 2)
        MARLIN_CASE_F16U4(64, 256, 256, 2, false, 2)
        MARLIN_CASE_F16U4(64, 128, 128, 2, false, 2)
        MARLIN_CASE_F16U4(128, 64, 128, 2, false, 2)
        MARLIN_CASE_F16U4(64, 256, 256, 3, false, 2)
        MARLIN_CASE_F16U4(64, 128, 128, 3, false, 2)
        MARLIN_CASE_F16U4(128, 64, 128, 3, false, 2)
        MARLIN_CASE_F16U4(64, 256, 256, 4, false, 2)
        MARLIN_CASE_F16U4(64, 128, 128, 4, false, 2)
        MARLIN_CASE_F16U4(128, 64, 128, 4, false, 2)
        return nullptr;
    }
    if (group_blocks == 8) {
        MARLIN_CASE_F16U4(128, 128, 256, 1, false, 8)
        MARLIN_CASE_F16U4(64, 128, 128, 1, false, 8)
        MARLIN_CASE_F16U4(128, 64, 128, 1, false, 8)
        MARLIN_CASE_F16U4(128, 128, 256, 1, true, 8)
        MARLIN_CASE_F16U4(64, 128, 128, 1, true, 8)
        MARLIN_CASE_F16U4(128, 64, 128, 1, true, 8)
        MARLIN_CASE_F16U4(64, 256, 256, 2, false, 8)
        MARLIN_CASE_F16U4(64, 128, 128, 2, false, 8)
        MARLIN_CASE_F16U4(128, 64, 128, 2, false, 8)
        MARLIN_CASE_F16U4(64, 256, 256, 3, false, 8)
        MARLIN_CASE_F16U4(64, 128, 128, 3, false, 8)
        MARLIN_CASE_F16U4(128, 64, 128, 3, false, 8)
        MARLIN_CASE_F16U4(64, 256, 256, 4, false, 8)
        MARLIN_CASE_F16U4(64, 128, 128, 4, false, 8)
        MARLIN_CASE_F16U4(128, 64, 128, 4, false, 8)
        return nullptr;
    }
    return nullptr;
}

static MarlinFuncPtr get_marlin_kernel_f16u8(thread_config_t const& th_config,
                                             int thread_m_blocks,
                                             bool m_block_size_8) {
    MARLIN_CASE_F16U8(128, 128, 256, 1, false, 2)
    MARLIN_CASE_F16U8(64, 128, 128, 1, false, 2)
    MARLIN_CASE_F16U8(128, 64, 128, 1, false, 2)
    MARLIN_CASE_F16U8(128, 128, 256, 1, true, 2)
    MARLIN_CASE_F16U8(64, 128, 128, 1, true, 2)
    MARLIN_CASE_F16U8(128, 64, 128, 1, true, 2)
    MARLIN_CASE_F16U8(64, 256, 256, 2, false, 2)
    MARLIN_CASE_F16U8(64, 128, 128, 2, false, 2)
    MARLIN_CASE_F16U8(128, 64, 128, 2, false, 2)
    MARLIN_CASE_F16U8(64, 256, 256, 3, false, 2)
    MARLIN_CASE_F16U8(64, 128, 128, 3, false, 2)
    MARLIN_CASE_F16U8(128, 64, 128, 3, false, 2)
    MARLIN_CASE_F16U8(64, 256, 256, 4, false, 2)
    MARLIN_CASE_F16U8(64, 128, 128, 4, false, 2)
    MARLIN_CASE_F16U8(128, 64, 128, 4, false, 2)
    return nullptr;
}

static MarlinFuncPtr get_marlin_kernel_f16(thread_config_t const& th_config,
                                           int num_bits,
                                           int thread_m_blocks,
                                           bool m_block_size_8,
                                           int group_blocks) {
    if (num_bits == 8) {
        if (group_blocks != 2) return nullptr;
        return get_marlin_kernel_f16u8(th_config, thread_m_blocks, m_block_size_8);
    }
    return get_marlin_kernel_f16u4(th_config, thread_m_blocks, m_block_size_8, group_blocks);
}

static thread_config_t choose_thread_config(int prob_m, int prob_n, int prob_k,
                                            int thread_m_blocks, bool m_block_size_8,
                                            int group_blocks, int num_bits,
                                            int stages, int max_shared_mem,
                                            int sms) {
    thread_config_t* table = thread_m_blocks > 1 ? large_batch_thread_configs : small_batch_thread_configs;
    int groupsize = group_blocks * 16;
    if (thread_m_blocks > 1) {
        const int m_tiles = div_ceil(prob_m, thread_m_blocks * 16);
        int want = -1;
        // Prefer `{64, 256, 256}` (2x A-tile reuse across 8 warps) whenever `prob_n >= 32768`
        // OR whenever `{64, 256, 256}` already saturates all SMs (`mn_tiles_256 >= sms`).
        // Step down to `{64, 128, 128}` only when `mn_tiles_256 < sms` (e.g. M=64, N=8192..17408).
        if (prob_n >= 32768 || ((prob_n % 256) == 0 && (prob_n / 256) * m_tiles >= sms)) {
            want = 0;
        } else if (prob_n >= 8192) {
            want = 1;
        }
        if (want >= 0 &&
            is_valid_config(table[want], thread_m_blocks, prob_m, prob_n, prob_k,
                            num_bits, groupsize, stages, max_shared_mem - 512) &&
            get_marlin_kernel_f16(table[want], num_bits, thread_m_blocks, m_block_size_8, group_blocks) != nullptr) {
            return table[want];
        }
    }
    for (int i = 0; i < 3; i++) {
        if (is_valid_config(table[i], thread_m_blocks, prob_m, prob_n, prob_k,
                            num_bits, groupsize, stages, max_shared_mem - 512) &&
            get_marlin_kernel_f16(table[i], num_bits, thread_m_blocks, m_block_size_8, group_blocks) != nullptr) {
            return table[i];
        }
    }
    return {-1, -1, -1};
}

static int marlin_gemm_impl(const w34_exec* ex, const void* A, const void* B,
                            const void* S, void* C, int32_t* workspace,
                            int64_t m, int64_t n, int64_t k,
                            int64_t workspace_ints, int group_blocks, int num_bits) {
    if (ex == nullptr || A == nullptr || B == nullptr || S == nullptr ||
        C == nullptr || workspace == nullptr) {
        return (int)cudaErrorInvalidValue;
    }
    int sms = 0;
    if (cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, 0) != cudaSuccess) {
        return (int)cudaGetLastError();
    }
    int max_shared_mem = 0;
    if (cudaDeviceGetAttribute(&max_shared_mem, cudaDevAttrMaxSharedMemoryPerBlockOptin, 0) != cudaSuccess) {
        return (int)cudaGetLastError();
    }
    if (workspace_ints < sms) return (int)cudaErrorInvalidValue;

    int prob_m = (int)m, prob_n = (int)n, prob_k = (int)k;
    if (prob_m <= 0 || prob_n <= 0 || prob_k <= 0 || (prob_k % 32) != 0 ||
        (prob_n % 64) != 0 || (group_blocks != 2 && group_blocks != 8)) {
        return (int)cudaErrorInvalidValue;
    }

    cudaStream_t stream = (cudaStream_t)ex->stream;
    const int4* A_ptr = (const int4*)A;
    const int4* B_ptr = (const int4*)B;
    int4* C_ptr = (int4*)C;
    const int4* S_ptr = (const int4*)S;

    constexpr int stages = 4;
    // When `(prob_n / 64) * 2 <= sms` and `prob_k >= 1024` (e.g. 4B `down_proj` [9728 -> 2560]
    // and 35B-A3B MoE projections [2048..4096 -> 512..2048]), `mn_tiles <= 42` leaves 50-90% of
    // the 84 SMs idle. Force `max_par = 1` + deterministic 2..4-slice Split-K across ALL
    // `M = 1..16` with lock-based sequential reduction (no atomicAdd) so `M = 1..16` stay
    // 0-ULP M-invariant!
    // For wider `N` where `(prob_n / 64) * 2 > sms` (e.g. 27B `N = 5120 -> 80` tiles), 1-slice
    // `cfg128 = {128, 64, 128}` already fills 80/84 SMs without Split-K reduction overhead.
    const bool enable_m_invariant_split_k =
        (prob_m <= 16) &&
        (((prob_n >= 2048) && ((prob_n / 64) * 2 <= sms) && (prob_k >= 4608)) ||
         ((prob_n >= 512) && (prob_n <= 2048) && (prob_k >= 1024) && (prob_k <= 2048)));
    int max_par = enable_m_invariant_split_k ? 1 : (prob_n <= 4096 ? 128 : 64);
    int rest_m = prob_m;

    while (rest_m > 0) {
        int par_count = rest_m / 64;
        if (par_count > max_par) par_count = max_par;
        int prob_m_split = par_count > 0 ? (par_count * 64) : rest_m;
        int thread_m_blocks = std::min(div_ceil(prob_m_split, 16), 4);
        bool m_block_size_8 = prob_m_split <= 8;

        thread_config_t th_cfg = choose_thread_config(
            prob_m_split, prob_n, prob_k, thread_m_blocks, m_block_size_8,
            group_blocks, num_bits, stages, max_shared_mem, sms);
        if (th_cfg.thread_n != -1) {
            const int cur_mn_tiles =
                (prob_n / th_cfg.thread_n) * div_ceil(prob_m_split, thread_m_blocks * 16);
            if (enable_m_invariant_split_k ||
                (prob_m_split <= 64 && cur_mn_tiles < sms && (prob_n / 64) * 2 > sms) ||
                cur_mn_tiles * 4 <= sms) {
                thread_config_t cfg128 = {128, 64, 128};
                if (is_valid_config(cfg128, thread_m_blocks, prob_m_split, prob_n, prob_k,
                                    num_bits, group_blocks * 16, stages, max_shared_mem) &&
                    get_marlin_kernel_f16(cfg128, num_bits, thread_m_blocks, m_block_size_8, group_blocks) != nullptr) {
                    th_cfg = cfg128;
                }
            }
        }
        if (th_cfg.thread_k == -1) return (int)cudaErrorInvalidValue;

        int mn_tiles = (prob_n / th_cfg.thread_n) * div_ceil(prob_m_split, thread_m_blocks * 16);
        int blocks = sms;
        if (enable_m_invariant_split_k && th_cfg.thread_k == 128 && th_cfg.thread_n == 64 && mn_tiles > 0) {
            int k_tiles = prob_k / th_cfg.thread_k;
            int cap_splits = (mn_tiles <= 20) ? 4 : 2;
            int max_k_splits = std::min(sms / mn_tiles, k_tiles / 4);
            if (max_k_splits > cap_splits) max_k_splits = cap_splits;
            if (max_k_splits < 1) max_k_splits = 1;
            blocks = mn_tiles * max_k_splits;
        } else if (blocks > mn_tiles) {
            blocks = mn_tiles;
        }
        if (blocks < 1) blocks = 1;

        auto kernel = get_marlin_kernel_f16(th_cfg, num_bits, thread_m_blocks, m_block_size_8, group_blocks);
        if (kernel == nullptr) return (int)cudaErrorInvalidValue;

        if (cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, max_shared_mem) != cudaSuccess) {
            return (int)cudaGetLastError();
        }
        bool part_use_atomic_add = !enable_m_invariant_split_k && (div_ceil(prob_m_split, 64) * prob_n <= 2048);

        kernel<<<blocks, th_cfg.num_threads, max_shared_mem, stream>>>(
            A_ptr, B_ptr, C_ptr, nullptr,
            nullptr, S_ptr, nullptr, nullptr, nullptr,
            prob_k / (group_blocks * 16), prob_m_split, prob_n,
            prob_k, prob_k, workspace,
            false, part_use_atomic_add, max_shared_mem);

        A_ptr += prob_m_split * (prob_k / 8);
        C_ptr += prob_m_split * (prob_n / 8);
        rest_m -= prob_m_split;
    }
    return (int)cudaGetLastError();
}

}  // namespace w34_marlin

extern "C" int w34_zero_i32(const w34_exec* ex, void* ptr, int64_t count) {
    if (ex == nullptr || ptr == nullptr) return (int)cudaErrorInvalidValue;
    if (count <= 0) return 0;
    int blocks = (int)((count + 255) / 256);
    w34_marlin::zero_ints_kernel<<<blocks, 256, 0, (cudaStream_t)ex->stream>>>((int32_t*)ptr, count);
    return (int)cudaGetLastError();
}

extern "C" int w34_marlin_repack_int4(const w34_exec* ex, const uint32_t* src_n_major,
                                      uint32_t* dst_marlin, int64_t k, int64_t n) {
    if (ex == nullptr || src_n_major == nullptr || dst_marlin == nullptr) {
        return (int)cudaErrorInvalidValue;
    }
    if (k <= 0 || n <= 0 || (k % 32) != 0 || (n % 64) != 0) {
        return (int)cudaErrorInvalidValue;
    }
    int blocks = 0;
    if (cudaDeviceGetAttribute(&blocks, cudaDevAttrMultiProcessorCount, 0) != cudaSuccess) {
        return (int)cudaGetLastError();
    }
    size_t smem = w34_marlin::stage_int4s * sizeof(int4) * w34_marlin::repack_stages;
    w34_marlin::gptq_marlin_repack_n_major_kernel<w34_marlin::repack_threads, 4>
        <<<(unsigned)blocks, w34_marlin::repack_threads, smem, (cudaStream_t)ex->stream>>>(
            src_n_major, dst_marlin, (int)k, (int)n);
    return (int)cudaGetLastError();
}

extern "C" int w34_marlin_gemm_f16u4(const w34_exec* ex, const void* a, const void* b,
                                     const void* scales, void* c, int32_t* workspace,
                                     int64_t m, int64_t n, int64_t k,
                                     int64_t workspace_ints, int group_blocks) {
    return w34_marlin::marlin_gemm_impl(
        ex, a, b, scales, c, workspace, m, n, k, workspace_ints, group_blocks, 4);
}

extern "C" int w34_marlin_gemm_f16u8(const w34_exec* ex, const void* a, const void* b,
                                     const void* scales, void* c, int32_t* workspace,
                                     int64_t m, int64_t n, int64_t k,
                                     int64_t workspace_ints, int group_blocks) {
    return w34_marlin::marlin_gemm_impl(
        ex, a, b, scales, c, workspace, m, n, k, workspace_ints, group_blocks, 8);
}
