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
static constexpr int kBF16 = vllm::ST_ID_BF16;
static constexpr int kU4 = vllm::ST_ID_U4;
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
    static_assert(num_bits == 4 || num_bits == 8, "num_bits must be 4 or 8");
    constexpr int pack_factor = 32 / num_bits;
    constexpr int tile_k = tile_k_size;
    constexpr int tile_ints = tile_k / pack_factor;
    constexpr int pair_ints = tile_ints * 2;
    constexpr int pair_int4s = pair_ints / 4;
    constexpr int cur_stage_int4s = tile_n_size * pair_int4s;
    static_assert(cur_stage_int4s <= num_threads, "stage copy exceeds CTA threads");

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
        int4* sh_ptr = sh + cur_stage_int4s * pipe;
        if (threadIdx.x < cur_stage_int4s) {
            int row_off = threadIdx.x / pair_int4s;
            int vec_off = threadIdx.x % pair_int4s;
            int first_word = k_pair * pair_ints + vec_off * 4;
            cp_async4(
                &sh_ptr[threadIdx.x],
                reinterpret_cast<int4 const*>(
                    &b_q_weight_ptr[(size_t)(first_n + row_off) * (size_k / pack_factor) + first_word]));
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

        uint32_t* sh_stage_int_ptr = reinterpret_cast<uint32_t*>(sh + cur_stage_int4s * pipe);

        uint32_t const* sh_b1 = sh_stage_int_ptr + cur_n * pair_ints + sub * tile_ints;
        uint32_t const* sh_b2 = sh_stage_int_ptr + (cur_n + 8) * pair_ints + sub * tile_ints;

        uint32_t vals[8];
#pragma unroll
        for (int i = 0; i < 4; i++) {
            int cur_elem = tc_row + tc_offsets[i];
            int cur_int = cur_elem / pack_factor;
            int cur_pos = cur_elem % pack_factor;
            vals[i] = (sh_b1[cur_int] >> (cur_pos * num_bits)) & mask;
            vals[4 + i] = (sh_b2[cur_int] >> (cur_pos * num_bits)) & mask;
        }

        constexpr int tile_out_size = tile_k * tile_n_size / pack_factor;
        int k_tile_id = k_pair * 2 + sub;
        int out_offset = (k_tile_id * n_tiles + n_tile_id) * tile_out_size;
        if constexpr (num_bits == 4) {
            constexpr int pack_idx[8] = {0, 2, 4, 6, 1, 3, 5, 7};
            uint32_t res = 0;
#pragma unroll
            for (int i = 0; i < 8; i++) {
                res |= vals[pack_idx[i]] << (i * 4);
            }
            out_ptr[out_offset + th_id * 4 + warp_id] = res;
        } else {
            constexpr int pack_idx[4] = {0, 2, 1, 3};
            uint32_t res1 = 0;
            uint32_t res2 = 0;
#pragma unroll
            for (int i = 0; i < 4; i++) {
                res1 |= vals[pack_idx[i]] << (i * 8);
                res2 |= vals[4 + pack_idx[i]] << (i * 8);
            }
            out_ptr[out_offset + th_id * 8 + warp_id * 2 + 0] = res1;
            out_ptr[out_offset + th_id * 8 + warp_id * 2 + 1] = res2;
        }
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

// Online weight-only FP16 -> Marlin U4B8 quantizer + tile repacker (`group_size` in {32, 64, 128}).
// Donor: vLLM PR #59973 (`vllm/model_executor/layers/quantization/online/lm_head.py`:
// `quantized_lm_head_copy`), adapted to sm_86/sm_89 Marlin U4B8 (`w34_marlin_gemm_f16u4`).
// Quantizes an N-major `[size_n, size_k]` FP16 weight matrix directly into:
//   - `dst_marlin`: `[size_k / 16, (size_n / 64) * 128]` uint32_t Marlin U4B8 tiles
//   - `dst_scales_perm`: `[size_k / group_size, size_n]` FP16 Marlin-permuted scales
// without allocating any intermediate FP16 or N-major packed staging buffer.
__global__ void marlin_quantize_f16_u4b8_kernel(
    half const* __restrict__ src_f16,
    uint32_t* __restrict__ dst_marlin,
    half* __restrict__ dst_scales_perm,
    int size_k,
    int size_n,
    int group_size) {
    constexpr int kTileN = 64;
    constexpr int kTileK = 16;
    const int n_tiles = size_n / kTileN;
    const int num_groups = size_k / group_size;
    const int total_tiles = n_tiles * num_groups;
    const int sub_tiles = group_size / kTileK;

    __shared__ uint8_t sh_q[kTileN][kTileK];

    const int r = threadIdx.x >> 1;
    const int sub_t = threadIdx.x & 1;
    const int warp_id = threadIdx.x >> 5;
    const int th_id = threadIdx.x & 31;
    const int tc_col = th_id >> 2;
    const int tc_row = (th_id & 3) * 2;
    constexpr int tc_offsets[4] = {0, 1, 8, 9};
    const int cur_n = warp_id * 16 + tc_col;

    for (int tile_idx = blockIdx.x; tile_idx < total_tiles; tile_idx += gridDim.x) {
        const int n_tile_id = tile_idx % n_tiles;
        const int group_id = tile_idx / n_tiles;
        const int first_n = n_tile_id * kTileN;
        const int first_k = group_id * group_size;
        half const* row_ptr = src_f16 + (size_t)(first_n + r) * size_k + first_k;

        float max_abs = 0.0f;
        for (int c = sub_t * 8; c < group_size; c += 16) {
            int4 vec = *reinterpret_cast<int4 const*>(row_ptr + c);
            half const* h = reinterpret_cast<half const*>(&vec);
#pragma unroll
            for (int b = 0; b < 8; ++b) {
                max_abs = fmaxf(max_abs, fabsf(__half2float(h[b])));
            }
        }
        max_abs = fmaxf(max_abs, __shfl_xor_sync(0xffffffffu, max_abs, 1));

        half scale_h = __float2half(max_abs / 7.0f);
        float scale_f = __half2float(scale_h);
        float inv_scale = (scale_f > 0.0f) ? (1.0f / scale_f) : 0.0f;

        if (sub_t == 0) {
            size_t scale_base = (size_t)group_id * size_n + (size_t)n_tile_id * kTileN;
            dst_scales_perm[scale_base + (r & 7) * 8 + (r >> 3)] = scale_h;
        }

        for (int sub_k = 0; sub_k < sub_tiles; ++sub_k) {
            int4 vec = *reinterpret_cast<int4 const*>(row_ptr + sub_k * kTileK + sub_t * 8);
            half const* h = reinterpret_cast<half const*>(&vec);
#pragma unroll
            for (int b = 0; b < 8; ++b) {
                float w = __half2float(h[b]);
                int q = __float2int_rn(w * inv_scale) + 8;
                q = max(0, min(15, q));
                sh_q[r][sub_t * 8 + b] = (uint8_t)q;
            }
            __syncthreads();

            uint32_t vals[8];
#pragma unroll
            for (int i = 0; i < 4; ++i) {
                int cur_elem = tc_row + tc_offsets[i];
                vals[i] = sh_q[cur_n][cur_elem];
                vals[4 + i] = sh_q[cur_n + 8][cur_elem];
            }
            constexpr int pack_idx[8] = {0, 2, 4, 6, 1, 3, 5, 7};
            uint32_t res = 0;
#pragma unroll
            for (int i = 0; i < 8; ++i) {
                res |= vals[pack_idx[i]] << (i * 4);
            }
            int k_tile_id = group_id * sub_tiles + sub_k;
            size_t out_offset = ((size_t)k_tile_id * n_tiles + n_tile_id) * 128;
            dst_marlin[out_offset + th_id * 4 + warp_id] = res;
            __syncthreads();
        }
    }
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

static int get_zp_cache_size(thread_config_t const& th_config, int group_size,
                             int num_bits, bool has_zp, int stages) {
    if (!has_zp) return 0;
    int pack_factor = 32 / num_bits;
    int tb_n = th_config.thread_n;
    int tb_k = th_config.thread_k;
    int tb_groups = group_size == -1 ? 1 : div_ceil(tb_k, group_size);
    return tb_groups * (tb_n / pack_factor) * 4 * stages;
}

static int get_kernel_cache_size(thread_config_t const& th_config, int thread_m_blocks,
                                 int num_bits, int group_size, bool has_zp, int stages) {
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
    int sh_zp_size = get_zp_cache_size(th_config, group_size, num_bits, has_zp, stages);
    return tmp_size + sh_a_size + sh_s_size + sh_zp_size;
}

static bool is_valid_config(thread_config_t const& th_config, int thread_m_blocks,
                            int prob_m, int prob_n, int prob_k, int num_bits,
                            int group_size, bool has_zp, int stages, int max_shared_mem) {
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
    if (group_size == -1 && prob_n >= 128 && th_config.thread_n < 128) {
        return false;
    }
    int cache_size = get_kernel_cache_size(
        th_config, thread_m_blocks, num_bits, group_size, has_zp, stages);
    return cache_size <= max_shared_mem;
}

typedef void (*MarlinFuncPtr)(MARLIN_KERNEL_PARAMS);

#define MARLIN_CASE_ATYPE_BTYPE(ATYPE, BTYPE, T_K, T_N, THR, M_BLOCKS, M8, GBX) \
    if (th_config.thread_k == (T_K) && th_config.thread_n == (T_N) &&           \
        th_config.num_threads == (THR) && thread_m_blocks == (M_BLOCKS) &&      \
        m_block_size_8 == (M8)) {                                               \
        return &Marlin<(ATYPE), (BTYPE), (ATYPE),                               \
                       (ATYPE), (THR), (M_BLOCKS), (T_N) / 16,                  \
                       (T_K) / 16, (M8), 4, (GBX), false>;                      \
    }

#define MARLIN_CASES_15_ATYPE(ATYPE, BTYPE, GBX)                              \
    MARLIN_CASE_ATYPE_BTYPE(ATYPE, BTYPE, 128, 128, 256, 1, false, GBX)       \
    MARLIN_CASE_ATYPE_BTYPE(ATYPE, BTYPE, 64, 128, 128, 1, false, GBX)        \
    MARLIN_CASE_ATYPE_BTYPE(ATYPE, BTYPE, 128, 64, 128, 1, false, GBX)        \
    MARLIN_CASE_ATYPE_BTYPE(ATYPE, BTYPE, 128, 128, 256, 1, true, GBX)        \
    MARLIN_CASE_ATYPE_BTYPE(ATYPE, BTYPE, 64, 128, 128, 1, true, GBX)         \
    MARLIN_CASE_ATYPE_BTYPE(ATYPE, BTYPE, 128, 64, 128, 1, true, GBX)         \
    MARLIN_CASE_ATYPE_BTYPE(ATYPE, BTYPE, 64, 256, 256, 2, false, GBX)        \
    MARLIN_CASE_ATYPE_BTYPE(ATYPE, BTYPE, 64, 128, 128, 2, false, GBX)        \
    MARLIN_CASE_ATYPE_BTYPE(ATYPE, BTYPE, 128, 64, 128, 2, false, GBX)        \
    MARLIN_CASE_ATYPE_BTYPE(ATYPE, BTYPE, 64, 256, 256, 3, false, GBX)        \
    MARLIN_CASE_ATYPE_BTYPE(ATYPE, BTYPE, 64, 128, 128, 3, false, GBX)        \
    MARLIN_CASE_ATYPE_BTYPE(ATYPE, BTYPE, 128, 64, 128, 3, false, GBX)        \
    MARLIN_CASE_ATYPE_BTYPE(ATYPE, BTYPE, 64, 256, 256, 4, false, GBX)        \
    MARLIN_CASE_ATYPE_BTYPE(ATYPE, BTYPE, 64, 128, 128, 4, false, GBX)        \
    MARLIN_CASE_ATYPE_BTYPE(ATYPE, BTYPE, 128, 64, 128, 4, false, GBX)

#define MARLIN_CASES_15(BTYPE, GBX) MARLIN_CASES_15_ATYPE(marlin::kF16, BTYPE, GBX)

static MarlinFuncPtr get_marlin_kernel_bf16u4(thread_config_t const& th_config,
                                              int thread_m_blocks,
                                              bool m_block_size_8,
                                              int group_blocks) {
    if (group_blocks == -1) {
        MARLIN_CASES_15_ATYPE(marlin::kBF16, marlin::kU4B8, -1)
        return nullptr;
    }
    if (group_blocks == 2) {
        MARLIN_CASES_15_ATYPE(marlin::kBF16, marlin::kU4B8, 2)
        return nullptr;
    }
    if (group_blocks == 4) {
        MARLIN_CASES_15_ATYPE(marlin::kBF16, marlin::kU4B8, 4)
        return nullptr;
    }
    if (group_blocks == 8) {
        MARLIN_CASES_15_ATYPE(marlin::kBF16, marlin::kU4B8, 8)
        return nullptr;
    }
    return nullptr;
}

static MarlinFuncPtr get_marlin_kernel_f16u4(thread_config_t const& th_config,
                                             int thread_m_blocks,
                                             bool m_block_size_8,
                                             int group_blocks) {
    if (group_blocks == -1) {
        MARLIN_CASES_15(marlin::kU4B8, -1)
        return nullptr;
    }
    if (group_blocks == 2) {
        MARLIN_CASES_15(marlin::kU4B8, 2)
        return nullptr;
    }
    if (group_blocks == 4) {
        MARLIN_CASES_15(marlin::kU4B8, 4)
        return nullptr;
    }
    if (group_blocks == 8) {
        MARLIN_CASES_15(marlin::kU4B8, 8)
        return nullptr;
    }
    return nullptr;
}

static MarlinFuncPtr get_marlin_kernel_f16u4_zp(thread_config_t const& th_config,
                                                int thread_m_blocks,
                                                bool m_block_size_8,
                                                int group_blocks) {
    if (group_blocks == 2) {
        MARLIN_CASES_15(marlin::kU4, 2)
        return nullptr;
    }
    if (group_blocks == 4) {
        MARLIN_CASES_15(marlin::kU4, 4)
        return nullptr;
    }
    if (group_blocks == 8) {
        MARLIN_CASES_15(marlin::kU4, 8)
        return nullptr;
    }
    return nullptr;
}

static MarlinFuncPtr get_marlin_kernel_f16u8(thread_config_t const& th_config,
                                             int thread_m_blocks,
                                             bool m_block_size_8,
                                             int group_blocks) {
    if (group_blocks == -1) {
        MARLIN_CASES_15(marlin::kU8B128, -1)
        return nullptr;
    }
    if (group_blocks == 2) {
        MARLIN_CASES_15(marlin::kU8B128, 2)
        return nullptr;
    }
    if (group_blocks == 4) {
        MARLIN_CASES_15(marlin::kU8B128, 4)
        return nullptr;
    }
    if (group_blocks == 8) {
        MARLIN_CASES_15(marlin::kU8B128, 8)
        return nullptr;
    }
    return nullptr;
}

static MarlinFuncPtr get_marlin_kernel_f16(thread_config_t const& th_config,
                                           int num_bits,
                                           bool has_zp,
                                           int thread_m_blocks,
                                           bool m_block_size_8,
                                           int group_blocks) {
    if (num_bits == 8) {
        if (has_zp) return nullptr;
        return get_marlin_kernel_f16u8(th_config, thread_m_blocks, m_block_size_8, group_blocks);
    }
    if (has_zp) {
        return get_marlin_kernel_f16u4_zp(th_config, thread_m_blocks, m_block_size_8, group_blocks);
    }
    return get_marlin_kernel_f16u4(th_config, thread_m_blocks, m_block_size_8, group_blocks);
}

static thread_config_t choose_thread_config(int prob_m, int prob_n, int prob_k,
                                            int thread_m_blocks, bool m_block_size_8,
                                            int group_blocks, int num_bits, bool has_zp,
                                            int stages, int max_shared_mem,
                                            int sms, bool is_bf16 = false) {
    thread_config_t* table = thread_m_blocks > 1 ? large_batch_thread_configs : small_batch_thread_configs;
    int groupsize = (group_blocks == -1) ? -1 : (group_blocks * 16);
    auto has_kernel = [&](const thread_config_t& cfg) {
        return is_bf16 ? (get_marlin_kernel_bf16u4(cfg, thread_m_blocks, m_block_size_8, group_blocks) != nullptr)
                       : (get_marlin_kernel_f16(cfg, num_bits, has_zp, thread_m_blocks, m_block_size_8, group_blocks) != nullptr);
    };
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
                            num_bits, groupsize, has_zp, stages, max_shared_mem - 512) &&
            has_kernel(table[want])) {
            return table[want];
        }
    }
    for (int i = 0; i < 3; i++) {
        if (is_valid_config(table[i], thread_m_blocks, prob_m, prob_n, prob_k,
                            num_bits, groupsize, has_zp, stages, max_shared_mem - 512) &&
            has_kernel(table[i])) {
            return table[i];
        }
    }
    return {-1, -1, -1};
}

static int marlin_gemm_impl(const w34_exec* ex, const void* A, const void* B,
                            const void* S, const void* ZP, void* C, int32_t* workspace,
                            int64_t m, int64_t n, int64_t k,
                            int64_t workspace_ints, int group_blocks, int num_bits,
                            bool has_zp, bool is_bf16 = false) {
    if (ex == nullptr || A == nullptr || B == nullptr || S == nullptr ||
        C == nullptr || workspace == nullptr || (has_zp && ZP == nullptr)) {
        return (int)cudaErrorInvalidValue;
    }
    w34_device_props dev_props{};
    int props_rc = w34_get_current_device_props(&dev_props);
    if (props_rc != 0) return props_rc;
    int sms = dev_props.multi_processor_count;
    int max_shared_mem = dev_props.max_shared_mem_per_block_optin;
    if (workspace_ints < sms) return (int)cudaErrorInvalidValue;

    int prob_m = (int)m, prob_n = (int)n, prob_k = (int)k;
    const bool valid_gb = (group_blocks == 2 || group_blocks == 4 || group_blocks == 8) ||
                          (!has_zp && group_blocks == -1);
    if (prob_m <= 0 || prob_n <= 0 || prob_k <= 0 || (prob_k % 32) != 0 ||
        (prob_n % 64) != 0 || !valid_gb ||
        (group_blocks > 0 && (prob_k % (group_blocks * 16)) != 0)) {
        return (int)cudaErrorInvalidValue;
    }

    cudaStream_t stream = (cudaStream_t)ex->stream;
    const int4* A_ptr = (const int4*)A;
    const int4* B_ptr = (const int4*)B;
    int4* C_ptr = (int4*)C;
    const int4* S_ptr = (const int4*)S;
    const int4* ZP_ptr = (const int4*)ZP;
    const int groupsize = (group_blocks == -1) ? -1 : (group_blocks * 16);
    const int num_groups = (group_blocks == -1) ? 1 : (prob_k / groupsize);

    constexpr int stages = 4;
    // When `(prob_n / 64) * 2 <= sms` and `prob_k >= 1024`, `mn_tiles <= sms / 2` leaves
    // 50-90% of the SMs idle. Force `max_par = 1` + deterministic 2..4-slice Split-K across
    // ALL `M = 1..16` with lock-based sequential reduction (no atomicAdd) so `M = 1..16` stay
    // 0-ULP M-invariant.
    // Keep `N = 2560, K <= 4096` on 1-slice `{128, 64, 128}` so 4B golden parity is preserved
    // across 84-SM (3090 Ti) and 128-SM (4090) GPUs while unlocking `N = 3072, K >= 2560`
    // (e.g. 9B TP=2 `qkv` / `in_z` on 128 SMs).
    const bool keep_2560_1slice = !is_bf16 && (prob_n == 2560 && prob_k <= 4096);
    const bool enable_m_invariant_split_k =
        (prob_m <= 16) && (group_blocks > 0) && !keep_2560_1slice &&
        ((is_bf16 && ((prob_n / 64) * 2 <= sms) && (prob_k >= 1024)) ||
         ((prob_n >= 2048) && ((prob_n / 64) * 2 <= sms) && (prob_k >= 4608)) ||
         ((prob_n == 3072) && ((prob_n / 64) * 2 <= sms) && (prob_k >= 2560)) ||
         ((prob_n >= 512) && (prob_n <= 2048) && (prob_k >= 1024) && (prob_k <= 2048)));
    // Scale wide-N (N > 4096) M-parallelism directly from the card's reported L2 cache size:
    // 16 (= 1024 M-rows) per 6 MiB of L2 cache so activations stay L2-resident across wide N
    // tiles (e.g. 3 MiB -> 8, 6 MiB -> 16, 24 MiB -> 64, 72 MiB -> 128).
    const int wide_n_max_par =
        std::min(128, std::max(8, dev_props.l2_cache_bytes / (384 * 1024)));
    int max_par = enable_m_invariant_split_k ? 1 : (prob_n <= 4096 ? 128 : wide_n_max_par);
    int rest_m = prob_m;

    while (rest_m > 0) {
        int par_count = rest_m / 64;
        if (par_count > max_par) par_count = max_par;
        int prob_m_split = par_count > 0 ? (par_count * 64) : rest_m;
        int thread_m_blocks = std::min(div_ceil(prob_m_split, 16), 4);
        bool m_block_size_8 = prob_m_split <= 8;

        thread_config_t th_cfg = choose_thread_config(
            prob_m_split, prob_n, prob_k, thread_m_blocks, m_block_size_8,
            group_blocks, num_bits, has_zp, stages, max_shared_mem, sms, is_bf16);
        if (th_cfg.thread_n != -1) {
            const int cur_mn_tiles =
                (prob_n / th_cfg.thread_n) * div_ceil(prob_m_split, thread_m_blocks * 16);
            if (enable_m_invariant_split_k ||
                (prob_m_split <= 64 && cur_mn_tiles < sms && (prob_n / 64) * 2 > sms) ||
                cur_mn_tiles * 4 <= sms) {
                thread_config_t cfg128 = {128, 64, 128};
                const bool cfg128_has_kernel =
                    is_bf16 ? (get_marlin_kernel_bf16u4(cfg128, thread_m_blocks, m_block_size_8, group_blocks) != nullptr)
                            : (get_marlin_kernel_f16(cfg128, num_bits, has_zp, thread_m_blocks, m_block_size_8, group_blocks) != nullptr);
                if (is_valid_config(cfg128, thread_m_blocks, prob_m_split, prob_n, prob_k,
                                    num_bits, groupsize, has_zp, stages, max_shared_mem) &&
                    cfg128_has_kernel) {
                    th_cfg = cfg128;
                }
            }
        }
        if (th_cfg.thread_k == -1) return (int)cudaErrorInvalidValue;

        int mn_tiles = (prob_n / th_cfg.thread_n) * div_ceil(prob_m_split, thread_m_blocks * 16);
        int k_tiles = prob_k / th_cfg.thread_k;
        // Prefill (M > 16) intentionally uses donor-style SM-wide Split-K to avoid idle
        // SMs on narrow projections. Its FP16 partial reductions need not be bit-identical
        // to unsplit GEMMs; the M=1..16 decode/verify invariant below remains unchanged.
        // workspace_ints >= sms is checked above. Callers supply zero-initialized locks,
        // serialize reuse on one stream (aux GEMMs have separate locks), and the donor's
        // final slice resets each lock. A failed CUDA execution invalidates that context.
        int blocks = sms;
        if (enable_m_invariant_split_k && th_cfg.thread_k == 128 && th_cfg.thread_n == 64 && mn_tiles > 0) {
            int cap_splits = (mn_tiles <= 20) ? 4 : 2;
            int max_k_splits = std::min(sms / mn_tiles, k_tiles / 4);
            if (max_k_splits > cap_splits) max_k_splits = cap_splits;
            if (max_k_splits < 1) max_k_splits = 1;
            blocks = mn_tiles * max_k_splits;
        } else if (prob_m <= 16 && blocks > mn_tiles) {
            // Preserve 1-slice M=1..16 decode/verify invariance outside explicit Split-K shapes.
            blocks = mn_tiles;
        } else if (blocks > mn_tiles * k_tiles) {
            blocks = mn_tiles * k_tiles;
        }
        if (blocks < 1) blocks = 1;

        auto kernel = is_bf16
            ? get_marlin_kernel_bf16u4(th_cfg, thread_m_blocks, m_block_size_8, group_blocks)
            : get_marlin_kernel_f16(
                  th_cfg, num_bits, has_zp, thread_m_blocks, m_block_size_8, group_blocks);
        if (kernel == nullptr) return (int)cudaErrorInvalidValue;

        if (cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, max_shared_mem) != cudaSuccess) {
            return (int)cudaGetLastError();
        }
        bool part_use_atomic_add = (blocks <= mn_tiles) && !enable_m_invariant_split_k &&
                                   (div_ceil(prob_m_split, 64) * prob_n <= 2048);

        kernel<<<blocks, th_cfg.num_threads, max_shared_mem, stream>>>(
            A_ptr, B_ptr, C_ptr, nullptr,
            nullptr, S_ptr, nullptr, ZP_ptr, nullptr,
            num_groups, prob_m_split, prob_n,
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
    w34_device_props dev_props{};
    int props_rc = w34_get_current_device_props(&dev_props);
    if (props_rc != 0) return props_rc;
    int blocks = dev_props.multi_processor_count;
    size_t smem = w34_marlin::stage_int4s * sizeof(int4) * w34_marlin::repack_stages;
    w34_marlin::gptq_marlin_repack_n_major_kernel<w34_marlin::repack_threads, 4>
        <<<(unsigned)blocks, w34_marlin::repack_threads, smem, (cudaStream_t)ex->stream>>>(
            src_n_major, dst_marlin, (int)k, (int)n);
    return (int)cudaGetLastError();
}

extern "C" int w34_marlin_repack_int8(const w34_exec* ex, const uint32_t* src_n_major,
                                      uint32_t* dst_marlin, int64_t k, int64_t n) {
    if (ex == nullptr || src_n_major == nullptr || dst_marlin == nullptr) {
        return (int)cudaErrorInvalidValue;
    }
    if (k <= 0 || n <= 0 || (k % 32) != 0 || (n % 64) != 0) {
        return (int)cudaErrorInvalidValue;
    }
    w34_device_props dev_props{};
    int props_rc = w34_get_current_device_props(&dev_props);
    if (props_rc != 0) return props_rc;
    int blocks = dev_props.multi_processor_count;
    size_t smem = (w34_marlin::stage_int4s * 2) * sizeof(int4) * w34_marlin::repack_stages;
    w34_marlin::gptq_marlin_repack_n_major_kernel<w34_marlin::repack_threads, 8>
        <<<(unsigned)blocks, w34_marlin::repack_threads, smem, (cudaStream_t)ex->stream>>>(
            src_n_major, dst_marlin, (int)k, (int)n);
    return (int)cudaGetLastError();
}

extern "C" int w34_marlin_quantize_f16_u4b8(
    const w34_exec* ex, const void* src_f16, uint32_t* dst_marlin,
    void* dst_scales_perm, int64_t k, int64_t n, int group_size) {
    if (ex == nullptr || src_f16 == nullptr || dst_marlin == nullptr ||
        dst_scales_perm == nullptr) {
        return (int)cudaErrorInvalidValue;
    }
    if (k <= 0 || n <= 0 || (k % 32) != 0 || (n % 64) != 0 ||
        (group_size != 32 && group_size != 64 && group_size != 128) ||
        (k % group_size) != 0) {
        return (int)cudaErrorInvalidValue;
    }
    if ((((uintptr_t)src_f16) & 15u) != 0u) {
        return (int)cudaErrorInvalidValue;
    }
    w34_device_props dev_props{};
    int props_rc = w34_get_current_device_props(&dev_props);
    if (props_rc != 0) return props_rc;
    int64_t total_tiles = (n / 64) * (k / group_size);
    int blocks = (int)std::min<int64_t>(total_tiles, (int64_t)dev_props.multi_processor_count * 8);
    if (blocks < 1) blocks = 1;
    w34_marlin::marlin_quantize_f16_u4b8_kernel<<<(unsigned)blocks, 128, 0, (cudaStream_t)ex->stream>>>(
        reinterpret_cast<half const*>(src_f16), dst_marlin,
        reinterpret_cast<half*>(dst_scales_perm), (int)k, (int)n, group_size);
    return (int)cudaGetLastError();
}

extern "C" int w34_marlin_gemm_f16u4(const w34_exec* ex, const void* a, const void* b,
                                     const void* scales, void* c, int32_t* workspace,
                                     int64_t m, int64_t n, int64_t k,
                                     int64_t workspace_ints, int group_blocks) {
    return w34_marlin::marlin_gemm_impl(
        ex, a, b, scales, nullptr, c, workspace, m, n, k, workspace_ints, group_blocks, 4, false, false);
}

extern "C" int w34_marlin_gemm_bf16u4(const w34_exec* ex, const void* a, const void* b,
                                      const void* scales, void* c, int32_t* workspace,
                                      int64_t m, int64_t n, int64_t k,
                                      int64_t workspace_ints, int group_blocks) {
    return w34_marlin::marlin_gemm_impl(
        ex, a, b, scales, nullptr, c, workspace, m, n, k, workspace_ints, group_blocks, 4, false, true);
}

extern "C" int w34_marlin_gemm_f16u4_zp(const w34_exec* ex, const void* a, const void* b,
                                        const void* scales, const void* zeros, void* c,
                                        int32_t* workspace, int64_t m, int64_t n, int64_t k,
                                        int64_t workspace_ints, int group_blocks) {
    return w34_marlin::marlin_gemm_impl(
        ex, a, b, scales, zeros, c, workspace, m, n, k, workspace_ints, group_blocks, 4, true);
}

extern "C" int w34_marlin_gemm_f16u8(const w34_exec* ex, const void* a, const void* b,
                                     const void* scales, void* c, int32_t* workspace,
                                     int64_t m, int64_t n, int64_t k,
                                     int64_t workspace_ints, int group_blocks) {
    return w34_marlin::marlin_gemm_impl(
        ex, a, b, scales, nullptr, c, workspace, m, n, k, workspace_ints, group_blocks, 8, false);
}
