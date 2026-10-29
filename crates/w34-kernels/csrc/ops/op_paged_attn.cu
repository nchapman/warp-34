// Descriptor-Driven Paged Attention and KV Cache Write kernels for warp-34.
// Supports w34_kv_layer_layout (FP16, BF16, TQ4 132B/row, Q8_0 272B/row) and
// w34_attention_batch (context_lens_before vs seq_lens_after, block_tables,
// write_slot_mapping, valid_token_mask, Causal and SymmetricWindow masks).
#include "../w34_cuda.h"
#include "../tq/tq4_codebook.cuh"

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <float.h>
#include <math.h>
#include <type_traits>

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

// Writes one (token, kv_head) row into a paged slab according to w34_kv_layer_layout.
__device__ void write_kv_row(
    uint8_t* dst_row,
    int format,
    int transform,
    const __half* src_row,
    int head_dim,
    float* s_warp,
    int target_warp) {
    if (format == W34_KV_FORMAT_FP16) {
        __half* dst = (__half*)dst_row;
        for (int c = threadIdx.x; c < head_dim; c += blockDim.x) {
            dst[c] = src_row[c];
        }
        return;
    }
    if (format == W34_KV_FORMAT_BF16) {
        __nv_bfloat16* dst = (__nv_bfloat16*)dst_row;
        for (int c = threadIdx.x; c < head_dim; c += blockDim.x) {
            dst[c] = __float2bfloat16(__half2float(src_row[c]));
        }
        return;
    }
    if (head_dim == 256 && format == W34_KV_FORMAT_Q8_0) {
        if ((int)(threadIdx.x >> 5) == target_warp) {
            const int lane = threadIdx.x & 31;
            block_q8_0_64* blks = (block_q8_0_64*)dst_row;
            float v[8];
            if (transform == W34_KV_TRANSFORM_RADEMACHER_WHT256) {
#pragma unroll
                for (int j = 0; j < 8; ++j) {
                    const int col = 32 * j + lane;
                    v[j] = __half2float(src_row[col]) * kTq4Signs256[col];
                }
                wht256_warp(v, lane);
            } else {
#pragma unroll
                for (int j = 0; j < 8; ++j) {
                    v[j] = __half2float(src_row[32 * j + lane]);
                }
            }
#pragma unroll
            for (int sb = 0; sb < 4; ++sb) {
                const float x0 = v[2 * sb];
                const float x1 = v[2 * sb + 1];
                const float amax = warp_reduce_max(fmaxf(fabsf(x0), fabsf(x1)));
                const float d = amax / 127.0f;
                const float inv_d = (d > 1e-12f) ? (1.0f / d) : 0.0f;
                if (lane == 0) blks[sb].d = d;
                float q0 = fminf(127.0f, fmaxf(-127.0f, roundf(x0 * inv_d)));
                float q1 = fminf(127.0f, fmaxf(-127.0f, roundf(x1 * inv_d)));
                blks[sb].qs[lane] = (int8_t)q0;
                blks[sb].qs[lane + 32] = (int8_t)q1;
            }
        }
        return;
    }
    if (head_dim == 256 && transform == W34_KV_TRANSFORM_RADEMACHER_WHT256 &&
        format == W34_KV_FORMAT_TQ4) {
        // Exact warp3090 tq4_encode_row (Seed-42 Rademacher + WHT-256 +
        // full-256 energy-preserving norm + Beta(256) codebook).
        // Runs on `target_warp` (0 for K, 1 for V) so K and V encode concurrently.
        if ((int)(threadIdx.x >> 5) == target_warp) {
            const int lane = threadIdx.x & 31;
            block_tq4_0* row2 = (block_tq4_0*)dst_row;
            float v[8];
            float s_sq = 0.0f;
#pragma unroll
            for (int j = 0; j < 8; ++j) {
                const int col = 32 * j + lane;
                float x = __half2float(src_row[col]);
                v[j] = x;
                s_sq += warp_reduce_sum(x * x);
            }
            const float grp_norm = __fsqrt_rn(s_sq);
            const float inv_norm = (grp_norm > 1e-10f) ? __fdiv_rn(1.0f, grp_norm) : 0.0f;
#pragma unroll
            for (int j = 0; j < 8; ++j) {
                const int col = 32 * j + lane;
                v[j] = v[j] * inv_norm * kTq4Signs256[col];
            }
            wht256_warp(v, lane);

            float vc_dot0 = 0.0f, c_sq0 = 0.0f;
            float vc_dot1 = 0.0f, c_sq1 = 0.0f;
#pragma unroll
            for (int j = 0; j < 8; ++j) {
                const float rv = v[j];
                int q = 0;
#pragma unroll
                for (int i = 0; i < 15; ++i) {
                    q += (rv > kTq4D256Boundaries[i]) ? 1 : 0;
                }
                const uint8_t nib = (uint8_t)q;
                const uint8_t partner =
                    (uint8_t)__shfl_sync(0xffffffffu, (unsigned)nib, lane ^ 1);
                if ((lane & 1) == 0) {
                    row2[j >> 2].qs[((j & 3) * 32 + lane) >> 1] =
                        (uint8_t)(nib | (partner << 4));
                }
                const float c = kTq4D256Centroids[q];
                const float vc = warp_reduce_sum(rv * c);
                const float cc = warp_reduce_sum(c * c);
                if (j < 4) {
                    vc_dot0 += vc;
                    c_sq0 += cc;
                } else {
                    vc_dot1 += vc;
                    c_sq1 += cc;
                }
            }
            if (lane == 0) {
                const float d0 = (c_sq0 > 1e-10f) ? (grp_norm * (vc_dot0 / c_sq0)) : 0.0f;
                const float d1 = (c_sq1 > 1e-10f) ? (grp_norm * (vc_dot1 / c_sq1)) : 0.0f;
                row2[0].d = __float2half(d0);
                row2[1].d = __float2half(d1);
            }
        }
        return;
    }
    if (format == W34_KV_FORMAT_TQ4) {
        // Quantize each 128-element sub-block into a 66-byte block_tq4_0
        const int num_sub = head_dim / 128;
        for (int sb = 0; sb < num_sub; sb++) {
            const __half* sub_src = src_row + sb * 128;
            block_tq4_0* blk = (block_tq4_0*)(dst_row + sb * sizeof(block_tq4_0));
            float sq = 0.0f;
            if (threadIdx.x < 128) {
                float x = __half2float(sub_src[threadIdx.x]);
                sq = x * x;
            }
            float norm_x = sqrtf(block_reduce_sum_128(sq, s_warp));
            float inv_norm = (norm_x > 1e-12f) ? (1.0f / norm_x) : 0.0f;

            // Find nearest Beta(256) centroid for unit-normalized x
            uint8_t q_idx = 8;
            float c_val = 0.0f;
            if (threadIdx.x < 128) {
                float u = __half2float(sub_src[threadIdx.x]) * inv_norm;
                q_idx = 0;
#pragma unroll
                for (int b = 0; b < 15; b++) {
                    if (u > kTq4D256Boundaries[b]) q_idx = (uint8_t)(b + 1);
                }
                c_val = kTq4D256Centroids[q_idx];
            }
            float c_sq = block_reduce_sum_128(c_val * c_val, s_warp);
            float norm_c = sqrtf(c_sq);
            float scale_d = (norm_c > 1e-12f) ? (norm_x / norm_c) : 0.0f;
            if (threadIdx.x == 0) {
                blk->d = __float2half(scale_d);
            }
            // Pack pairs of nibbles (thread i in 0..64 packs elem 2*i and 2*i+1)
            __shared__ uint8_t s_q[128];
            if (threadIdx.x < 128) s_q[threadIdx.x] = q_idx;
            __syncthreads();
            if (threadIdx.x < 64) {
                uint8_t lo = s_q[2 * threadIdx.x] & 0x0Fu;
                uint8_t hi = (s_q[2 * threadIdx.x + 1] & 0x0Fu) << 4;
                blk->qs[threadIdx.x] = lo | hi;
            }
            __syncthreads();
        }
        return;
    }
    if (format == W34_KV_FORMAT_Q8_0) {
        const int num_sub = head_dim / 64;
        for (int sb = 0; sb < num_sub; sb++) {
            const __half* sub_src = src_row + sb * 64;
            block_q8_0_64* blk = (block_q8_0_64*)(dst_row + sb * sizeof(block_q8_0_64));
            float abs_v = 0.0f;
            if (threadIdx.x < 64) {
                abs_v = fabsf(__half2float(sub_src[threadIdx.x]));
            }
            float amax = block_reduce_max_128(abs_v, s_warp);
            float d = amax / 127.0f;
            float inv_d = (d > 1e-12f) ? (1.0f / d) : 0.0f;
            if (threadIdx.x == 0) blk->d = d;
            if (threadIdx.x < 64) {
                float qf = roundf(__half2float(sub_src[threadIdx.x]) * inv_d);
                qf = fminf(127.0f, fmaxf(-127.0f, qf));
                blk->qs[threadIdx.x] = (int8_t)qf;
            }
            __syncthreads();
        }
    }
}

__device__ __forceinline__ float read_kv_elem(
    const uint8_t* row_ptr,
    int format,
    int col) {
    if (format == W34_KV_FORMAT_FP16) {
        return __half2float(((const __half*)row_ptr)[col]);
    }
    if (format == W34_KV_FORMAT_BF16) {
        return __bfloat162float(((const __nv_bfloat16*)row_ptr)[col]);
    }
    if (format == W34_KV_FORMAT_TQ4) {
        const int sb = col >> 7;        // col / 128
        const int idx = col & 127;       // col % 128
        const block_tq4_0* blk = (const block_tq4_0*)(row_ptr + sb * sizeof(block_tq4_0));
        const float d = __half2float(blk->d);
        const uint8_t packed = blk->qs[idx >> 1];
        const uint8_t nib = (idx & 1) ? (packed >> 4) : (packed & 0x0Fu);
        return d * kTq4D256Centroids[nib];
    }
    if (format == W34_KV_FORMAT_Q8_0) {
        const int sb = col >> 6;        // col / 64
        const int idx = col & 63;        // col % 64
        const block_q8_0_64* blk = (const block_q8_0_64*)(row_ptr + sb * sizeof(block_q8_0_64));
        return blk->d * (float)blk->qs[idx];
    }
    return 0.0f;
}

// Grid: (total_tokens, local_kv_heads), Block: 128 threads
__global__ void kv_cache_write_kernel(
    w34_kv_layer_layout layout,
    w34_attention_batch batch,
    const __half* __restrict__ k_in,
    const __half* __restrict__ v_in,
    int k_row_stride,
    int v_row_stride) {
    const int tok = blockIdx.x;
    if (tok >= batch.total_tokens) return;
    if (batch.valid_token_mask != nullptr && batch.valid_token_mask[tok] == 0) return;

    const int32_t slot = batch.write_slot_mapping[tok];
    if (slot < 0) return;

    const int kv_head = blockIdx.y;
    const int page_id = slot / layout.block_size;
    const int page_off = slot % layout.block_size;

    uint8_t* k_dst = (uint8_t*)(uintptr_t)layout.k_base
                     + (size_t)page_id * layout.k_page_stride_bytes
                     + (size_t)kv_head * layout.k_head_stride_bytes
                     + (size_t)page_off * layout.k_token_stride_bytes;
    uint8_t* v_dst = (uint8_t*)(uintptr_t)layout.v_base
                     + (size_t)page_id * layout.v_page_stride_bytes
                     + (size_t)kv_head * layout.v_head_stride_bytes
                     + (size_t)page_off * layout.v_token_stride_bytes;

    const __half* k_src = k_in + (size_t)tok * k_row_stride + (size_t)kv_head * layout.head_dim;
    const __half* v_src = v_in + (size_t)tok * v_row_stride + (size_t)kv_head * layout.head_dim;

    __shared__ float s_warp[8];
    write_kv_row(k_dst, layout.k_format, layout.k_transform, k_src, layout.head_dim, s_warp, 0);
    write_kv_row(v_dst, layout.v_format, layout.v_transform, v_src, layout.head_dim, s_warp, 1);
}

// Grid: (total_tokens, num_q_heads), Block: 128 threads.
// Uses online FP32 softmax across the paged sequence, deterministic per-step reduction
// so output is bit-identical regardless of physical page permutation in block_tables!
template <typename Scalar>
__global__ void paged_attention_kernel(
    w34_kv_layer_layout layout,
    w34_attention_batch batch,
    const Scalar* __restrict__ q_in,
    Scalar* __restrict__ attn_out,
    int num_q_heads) {
    const int tok = blockIdx.x;
    if (tok >= batch.total_tokens) return;
    if (batch.valid_token_mask != nullptr && batch.valid_token_mask[tok] == 0) return;

    // Locate owning request r via query_start_loc[r] <= tok < query_start_loc[r+1]
    int req = -1;
    int q_start = 0;
    int q_end = 0;
    for (int r = 0; r < batch.num_reqs; r++) {
        int lo = batch.query_start_loc[r];
        int hi = batch.query_start_loc[r + 1];
        if (tok >= lo && tok < hi) {
            req = r;
            q_start = lo;
            q_end = hi;
            break;
        }
    }
    if (req < 0) return;
    if (batch.valid_req_mask != nullptr && batch.valid_req_mask[req] == 0) return;

    const int q_head = blockIdx.y;
    const int gqa_ratio = num_q_heads / layout.local_kv_heads;
    const int kv_head = q_head / gqa_ratio;
    const int head_dim = layout.head_dim;

    const int q_offset_in_req = tok - q_start;
    const int ctx_before = batch.context_lens_before != nullptr
                               ? batch.context_lens_before[req]
                               : (batch.seq_lens_after[req] - (q_end - q_start));
    const int seq_after = batch.seq_lens_after[req];

    // Non-causal DFlash permits all materialized future keys. SGLang's
    // FlashInfer window_left applies relative to each query, NOT the block end.
    int max_kv_pos = (batch.mask_mode == W34_ATTN_MASK_SYMMETRIC_WINDOW)
                         ? (seq_after - 1)
                         : (ctx_before + q_offset_in_req);
    if (max_kv_pos >= seq_after) max_kv_pos = seq_after - 1;

    int min_kv_pos = 0;
    if (batch.sliding_window > 0) {
        int ref_pos = ctx_before + q_offset_in_req;
        int win_start = ref_pos - batch.sliding_window + 1;
        if (win_start > min_kv_pos) min_kv_pos = win_start;
    }

    const Scalar* q_row = q_in + ((size_t)tok * num_q_heads + q_head) * head_dim;
    Scalar* out_row = attn_out + ((size_t)tok * num_q_heads + q_head) * head_dim;

    // 4 warps (128 threads): each warp `wid` (0..3) processes kv_pos = min_kv_pos + wid + 4*step
    // using register-only warp shuffles (0 __syncthreads() inside the KV loop), then merges
    // online-softmax states across warps 0..3 at the end.
    const int lane = threadIdx.x & 31;
    const int wid = threadIdx.x >> 5;  // 0..3

    float q_reg[8];
    float acc[8];
#pragma unroll
    for (int j = 0; j < 8; j++) {
        const int col = lane + 32 * j;
        q_reg[j] = (col < head_dim) ? (float)q_row[col] : 0.0f;
        acc[j] = 0.0f;
    }
    if (head_dim == 256 && layout.k_transform == W34_KV_TRANSFORM_RADEMACHER_WHT256) {
        // Pre-rotate Q into the WHT-256 domain once per query row:
        // Q' = WHT256 * diag(s) * Q so <Q', K_rot> == <Q, K>.
#pragma unroll
        for (int j = 0; j < 8; ++j) {
            q_reg[j] *= kTq4Signs256[lane + 32 * j];
        }
        wht256_warp(q_reg, lane);
    }

    float m_i = -FLT_MAX;
    float l_i = 0.0f;

    const int32_t* req_blocks = batch.block_tables + (size_t)req * batch.max_blocks_per_req;

    // FlashInfer FA2 compute_sfm_v casts exp2 scores to DTypeQ before
    // BOTH PV MMA and m16k16_rowsum_f16f16f32. Keep the FP16 and quantized
    // policies below untouched. Tile maxima must precede probability rounding:
    // rounding each scalar online update is not equivalent.
    if constexpr (std::is_same<Scalar, __nv_bfloat16>::value) {
        if (layout.k_format == W34_KV_FORMAT_BF16 &&
            layout.v_format == W34_KV_FORMAT_BF16) {
            // Pinned sm86 FA2 uses NUM_MMA_KV=4 at HD=128, but =1 at
            // HD=256 (WKV=1 for both). Probability rounding follows the maximum
            // of the entire 64/16-key tile, before 16-key MMA contractions.
            // Split-KV donor dispatches can additionally round partial outputs.
            const int tile = head_dim <= 128 ? 64 : 16;
            __shared__ float scores[64];
            __shared__ __align__(4) uint16_t p_bits[64];
            __shared__ const uint8_t* v_ptrs[64];
            auto ptx_exp2 = [](float x) {
                float y;
                asm volatile("ex2.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x));
                return y;
            };
            auto ptx_rcp = [](float x) {
                float y;
                asm volatile("rcp.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x));
                return y;
            };
            float num_frag[8][2] = {{0.f}};
            float maximum = -FLT_MAX, denominator = 0.f;
            const float scale = batch.sm_scale * 1.4426950408889634f;
            const uint32_t* qp32 = (const uint32_t*)q_row;
            uint32_t qa0[16], qa2[16];
            const int k_steps = head_dim >> 4;
#pragma unroll
            for (int ks = 0; ks < 16; ++ks) {
                if (ks < k_steps && lane < 4) {
                    qa0[ks] = qp32[ks * 8 + lane];
                    qa2[ks] = qp32[ks * 8 + 4 + lane];
                } else {
                    qa0[ks] = 0u;
                    qa2[ks] = 0u;
                }
            }
            const int start_base = (min_kv_pos > 0) ? ((min_kv_pos / tile) * tile) : 0;
            for (int base = start_base; base <= max_kv_pos; base += tile) {
                for (int t = wid; t < tile; t += 4) {
                    int kv = base + t;
                    float dot = -FLT_MAX;
                    const uint8_t* vp = nullptr;
                    if (kv >= min_kv_pos && kv <= max_kv_pos) {
                        const uint8_t* kp = (const uint8_t*)(uintptr_t)layout.k_base
                            + (size_t)req_blocks[kv / layout.block_size] * layout.k_page_stride_bytes
                            + (size_t)kv_head * layout.k_head_stride_bytes
                            + (size_t)(kv % layout.block_size) * layout.k_token_stride_bytes;
                        vp = (const uint8_t*)(uintptr_t)layout.v_base
                            + (size_t)req_blocks[kv / layout.block_size] * layout.v_page_stride_bytes
                            + (size_t)kv_head * layout.v_head_stride_bytes
                            + (size_t)(kv % layout.block_size) * layout.v_token_stride_bytes;
                        dot = 0.f;
                        const uint32_t* kp32 = (const uint32_t*)kp;
                        for (int ks = 0; ks < k_steps; ++ks) {
                            uint32_t a0 = qa0[ks];
                            uint32_t a2 = qa2[ks];
                            uint32_t b0 = (lane < 4) ? kp32[ks * 8 + lane] : 0u;
                            uint32_t b1 = (lane < 4) ? kp32[ks * 8 + 4 + lane] : 0u;
                            uint32_t z = 0u;
                            float d1 = 0.f, d2 = 0.f, d3 = 0.f;
                            asm volatile(
                                "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
                                "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                                : "+f"(dot), "+f"(d1), "+f"(d2), "+f"(d3)
                                : "r"(a0), "r"(z), "r"(a2), "r"(z), "r"(b0), "r"(b1));
                        }
                    }
                    if (lane == 0) {
                        scores[t] = dot;
                        v_ptrs[t] = vp;
                    }
                }
                __syncthreads();
                float next = maximum;
                for (int t = 0; t < tile; ++t) next = fmaxf(next, scores[t]);
                float alpha = (next == -FLT_MAX) ? 0.f : ptx_exp2(maximum * scale - next * scale);
                denominator *= alpha;
#pragma unroll
                for (int step = 0; step < 8; ++step) {
                    num_frag[step][0] *= alpha;
                    num_frag[step][1] *= alpha;
                }
                maximum = next;
                if (threadIdx.x < tile) {
                    int kv = base + threadIdx.x;
                    p_bits[threadIdx.x] = (kv >= min_kv_pos && kv <= max_kv_pos)
                        ? __bfloat16_as_ushort(__float2bfloat16_rn(ptx_exp2(scores[threadIdx.x] * scale - maximum * scale)))
                        : 0u;
                }
                __syncthreads();
                for (int chunk = 0; chunk < tile; chunk += 16) {
                const uint32_t* p32 = (const uint32_t*)(p_bits + chunk);
                uint32_t pa0 = (lane < 4) ? p32[lane] : 0u;
                uint32_t pa2 = (lane < 4) ? p32[4 + lane] : 0u;
                uint32_t z = 0u;
                {
                    float d0 = denominator, d1 = 0.f, d2 = 0.f, d3 = 0.f;
                    uint32_t ones = (lane < 4) ? 0x3f803f80u : 0u;
                    asm volatile(
                        "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
                        "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                        : "+f"(d0), "+f"(d1), "+f"(d2), "+f"(d3)
                        : "r"(pa0), "r"(z), "r"(pa2), "r"(z), "r"(ones), "r"(ones));
                    denominator = __shfl_sync(0xffffffffu, d0, 0);
                }
                const int col_group = lane >> 2;
                const int k_pair = lane & 3;
#pragma unroll
                for (int step = 0; step < 8; ++step) {
                    int col = wid * 64 + step * 8 + col_group;
                    uint32_t vb0 = 0u, vb1 = 0u;
                    if (col < head_dim) {
                        const uint16_t* vp0 = (const uint16_t*)v_ptrs[chunk + 2 * k_pair];
                        const uint16_t* vp1 = (const uint16_t*)v_ptrs[chunk + 2 * k_pair + 1];
                        const uint16_t* vp2 = (const uint16_t*)v_ptrs[chunk + 8 + 2 * k_pair];
                        const uint16_t* vp3 = (const uint16_t*)v_ptrs[chunk + 8 + 2 * k_pair + 1];
                        uint32_t v0 = vp0 ? vp0[col] : 0u;
                        uint32_t v1 = vp1 ? vp1[col] : 0u;
                        uint32_t v2 = vp2 ? vp2[col] : 0u;
                        uint32_t v3 = vp3 ? vp3[col] : 0u;
                        vb0 = v0 | (v1 << 16);
                        vb1 = v2 | (v3 << 16);
                    }
                    float c0 = num_frag[step][0], c1 = num_frag[step][1], c2 = 0.f, c3 = 0.f;
                    asm volatile(
                        "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
                        "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                        : "+f"(c0), "+f"(c1), "+f"(c2), "+f"(c3)
                        : "r"(pa0), "r"(z), "r"(pa2), "r"(z), "r"(vb0), "r"(vb1));
                    num_frag[step][0] = c0;
                    num_frag[step][1] = c1;
                }
                }
                __syncthreads();
            }
            const float d_rcp = (maximum > -FLT_MAX && denominator > 0.f) ? ptx_rcp(denominator) : 0.f;
            if (lane < 4) {
#pragma unroll
                for (int step = 0; step < 8; ++step) {
                    int col0 = wid * 64 + step * 8 + 2 * lane;
                    if (col0 < head_dim) out_row[col0] = __float2bfloat16_rn(num_frag[step][0] * d_rcp);
                    if (col0 + 1 < head_dim) out_row[col0 + 1] = __float2bfloat16_rn(num_frag[step][1] * d_rcp);
                }
            }
            return;
        }
    }

    for (int kv_pos = min_kv_pos + wid; kv_pos <= max_kv_pos; kv_pos += 4) {
        const int logical_block = kv_pos / layout.block_size;
        const int page_off = kv_pos % layout.block_size;
        const int32_t phys_page = req_blocks[logical_block];

        const uint8_t* k_ptr = (const uint8_t*)(uintptr_t)layout.k_base
                               + (size_t)phys_page * layout.k_page_stride_bytes
                               + (size_t)kv_head * layout.k_head_stride_bytes
                               + (size_t)page_off * layout.k_token_stride_bytes;
        const uint8_t* v_ptr = (const uint8_t*)(uintptr_t)layout.v_base
                               + (size_t)phys_page * layout.v_page_stride_bytes
                               + (size_t)kv_head * layout.v_head_stride_bytes
                               + (size_t)page_off * layout.v_token_stride_bytes;

        float dot = 0.0f;
        if (head_dim == 256 && layout.k_format == W34_KV_FORMAT_FP16) {
            const __half* kh = (const __half*)k_ptr;
#pragma unroll
            for (int j = 0; j < 8; ++j) {
                dot += q_reg[j] * __half2float(kh[lane + 32 * j]);
            }
        } else if (head_dim == 256 && layout.k_format == W34_KV_FORMAT_TQ4) {
            const block_tq4_0* k_blks = (const block_tq4_0*)k_ptr;
            const float d0 = __half2float(k_blks[0].d);
            const float d1 = __half2float(k_blks[1].d);
            const int odd = lane & 1;
            float dot0 = 0.0f;
            float dot1 = 0.0f;
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                const uint8_t p0 = k_blks[0].qs[(lane + 32 * j) >> 1];
                const uint8_t p1 = k_blks[1].qs[(lane + 32 * j) >> 1];
                const uint8_t n0 = odd ? (p0 >> 4) : (p0 & 0x0Fu);
                const uint8_t n1 = odd ? (p1 >> 4) : (p1 & 0x0Fu);
                dot0 += q_reg[j] * kTq4D256Centroids[n0];
                dot1 += q_reg[j + 4] * kTq4D256Centroids[n1];
            }
            dot = d0 * dot0 + d1 * dot1;
        } else if (head_dim == 256 && layout.k_format == W34_KV_FORMAT_Q8_0) {
            const block_q8_0_64* k_blks = (const block_q8_0_64*)k_ptr;
#pragma unroll
            for (int sb = 0; sb < 4; ++sb) {
                const float d = k_blks[sb].d;
                const float k0 = (float)k_blks[sb].qs[lane];
                const float k1 = (float)k_blks[sb].qs[lane + 32];
                dot += d * (q_reg[2 * sb] * k0 + q_reg[2 * sb + 1] * k1);
            }
        } else {
#pragma unroll
            for (int j = 0; j < 8; j++) {
                const int col = lane + 32 * j;
                if (col < head_dim) {
                    dot += q_reg[j] * read_kv_elem(k_ptr, layout.k_format, col);
                }
            }
        }
#pragma unroll
        for (int off = 16; off > 0; off >>= 1) {
            dot += __shfl_xor_sync(0xffffffffu, dot, off);
        }
        const float score = dot * batch.sm_scale;

        const float m_new = fmaxf(m_i, score);
        const float alpha = expf(m_i - m_new);
        const float p = expf(score - m_new);
        l_i = l_i * alpha + p;
        m_i = m_new;

        if (head_dim == 256 && layout.v_format == W34_KV_FORMAT_FP16) {
            const __half* vh = (const __half*)v_ptr;
#pragma unroll
            for (int j = 0; j < 8; ++j) {
                acc[j] = acc[j] * alpha + p * __half2float(vh[lane + 32 * j]);
            }
        } else if (head_dim == 256 && layout.v_format == W34_KV_FORMAT_TQ4) {
            const block_tq4_0* v_blks = (const block_tq4_0*)v_ptr;
            const float pd0 = p * __half2float(v_blks[0].d);
            const float pd1 = p * __half2float(v_blks[1].d);
            const int odd = lane & 1;
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                const uint8_t p0 = v_blks[0].qs[(lane + 32 * j) >> 1];
                const uint8_t p1 = v_blks[1].qs[(lane + 32 * j) >> 1];
                const uint8_t n0 = odd ? (p0 >> 4) : (p0 & 0x0Fu);
                const uint8_t n1 = odd ? (p1 >> 4) : (p1 & 0x0Fu);
                acc[j] = acc[j] * alpha + pd0 * kTq4D256Centroids[n0];
                acc[j + 4] = acc[j + 4] * alpha + pd1 * kTq4D256Centroids[n1];
            }
        } else if (head_dim == 256 && layout.v_format == W34_KV_FORMAT_Q8_0) {
            const block_q8_0_64* v_blks = (const block_q8_0_64*)v_ptr;
#pragma unroll
            for (int sb = 0; sb < 4; ++sb) {
                const float pd = p * v_blks[sb].d;
                acc[2 * sb] = acc[2 * sb] * alpha + pd * (float)v_blks[sb].qs[lane];
                acc[2 * sb + 1] = acc[2 * sb + 1] * alpha + pd * (float)v_blks[sb].qs[lane + 32];
            }
        } else {
#pragma unroll
            for (int j = 0; j < 8; j++) {
                const int col = lane + 32 * j;
                if (col < head_dim) {
                    acc[j] = acc[j] * alpha + p * read_kv_elem(v_ptr, layout.v_format, col);
                }
            }
        }
    }

    __shared__ float s_m[4];
    __shared__ float s_l[4];
    __shared__ float s_acc[4][256];

    if (lane == 0) {
        s_m[wid] = m_i;
        s_l[wid] = l_i;
    }
#pragma unroll
    for (int j = 0; j < 8; j++) {
        const int col = lane + 32 * j;
        if (col < head_dim) {
            s_acc[wid][col] = acc[j];
        }
    }
    __syncthreads();

    float m_tot = -FLT_MAX;
#pragma unroll
    for (int w = 0; w < 4; w++) {
        if (s_l[w] > 0.0f) {
            m_tot = fmaxf(m_tot, s_m[w]);
        }
    }
    float l_tot = 0.0f;
    float w_scale[4];
#pragma unroll
    for (int w = 0; w < 4; w++) {
        if (s_l[w] > 0.0f) {
            float s = expf(s_m[w] - m_tot);
            w_scale[w] = s;
            l_tot += s_l[w] * s;
        } else {
            w_scale[w] = 0.0f;
        }
    }
    const float inv_l = (l_tot > 0.0f) ? (1.0f / l_tot) : 0.0f;
    if (head_dim == 256 && layout.v_transform == W34_KV_TRANSFORM_RADEMACHER_WHT256) {
        if (wid == 0) {
            float out_v[8];
#pragma unroll
            for (int j = 0; j < 8; ++j) {
                const int col = lane + 32 * j;
                float sum_v = s_acc[0][col] * w_scale[0]
                            + s_acc[1][col] * w_scale[1]
                            + s_acc[2][col] * w_scale[2]
                            + s_acc[3][col] * w_scale[3];
                out_v[j] = sum_v * inv_l;
            }
            // Inverse transform R^{-1} = diag(s) * WHT256: butterfly first, then signs.
            wht256_warp(out_v, lane);
#pragma unroll
            for (int j = 0; j < 8; ++j) {
                const int col = lane + 32 * j;
                out_row[col] = Scalar(out_v[j] * kTq4Signs256[col]);
            }
        }
        return;
    }
    for (int col = threadIdx.x; col < head_dim; col += blockDim.x) {
        float sum_v = s_acc[0][col] * w_scale[0]
                    + s_acc[1][col] * w_scale[1]
                    + s_acc[2][col] * w_scale[2]
                    + s_acc[3][col] * w_scale[3];
        out_row[col] = Scalar(sum_v * inv_l);
    }
}

// Cooperative 1-warp (32-lane) decode of one head_dim=256 KV row (FP16, TQ4, or Q8_0)
// into 16-byte-aligned shared memory row `dst[264]`.
__device__ __forceinline__ void decode_kv_row_256_to_smem_f16(
    const uint8_t* src_ptr,
    int format,
    __half* dst,
    int lane) {
    if (format == W34_KV_FORMAT_FP16) {
        *reinterpret_cast<uint4*>(dst + (lane << 3)) =
            *reinterpret_cast<const uint4*>(reinterpret_cast<const __half*>(src_ptr) + (lane << 3));
        return;
    }
    if (format == W34_KV_FORMAT_TQ4) {
        const block_tq4_0* blks = reinterpret_cast<const block_tq4_0*>(src_ptr);
        const float d0 = __half2float(blks[0].d);
        const float d1 = __half2float(blks[1].d);
        const uint16_t p0 = *reinterpret_cast<const uint16_t*>(&blks[0].qs[lane << 1]);
        const uint16_t p1 = *reinterpret_cast<const uint16_t*>(&blks[1].qs[lane << 1]);
        *reinterpret_cast<__half2*>(dst + (lane << 2)) = __floats2half2_rn(
            d0 * kTq4D256Centroids[p0 & 0x0Fu],
            d0 * kTq4D256Centroids[(p0 >> 4) & 0x0Fu]);
        *reinterpret_cast<__half2*>(dst + (lane << 2) + 2) = __floats2half2_rn(
            d0 * kTq4D256Centroids[(p0 >> 8) & 0x0Fu],
            d0 * kTq4D256Centroids[(p0 >> 12) & 0x0Fu]);
        *reinterpret_cast<__half2*>(dst + 128 + (lane << 2)) = __floats2half2_rn(
            d1 * kTq4D256Centroids[p1 & 0x0Fu],
            d1 * kTq4D256Centroids[(p1 >> 4) & 0x0Fu]);
        *reinterpret_cast<__half2*>(dst + 128 + (lane << 2) + 2) = __floats2half2_rn(
            d1 * kTq4D256Centroids[(p1 >> 8) & 0x0Fu],
            d1 * kTq4D256Centroids[(p1 >> 12) & 0x0Fu]);
        return;
    }
    if (format == W34_KV_FORMAT_Q8_0) {
        const block_q8_0_64* blks = reinterpret_cast<const block_q8_0_64*>(src_ptr);
#pragma unroll
        for (int sb = 0; sb < 4; ++sb) {
            const float d = blks[sb].d;
            const uint16_t q2 = *reinterpret_cast<const uint16_t*>(&blks[sb].qs[lane << 1]);
            const int8_t q_lo = static_cast<int8_t>(q2 & 0xFFu);
            const int8_t q_hi = static_cast<int8_t>(q2 >> 8);
            *reinterpret_cast<__half2*>(dst + sb * 64 + (lane << 1)) =
                __floats2half2_rn(d * (float)q_lo, d * (float)q_hi);
        }
        return;
    }
    *reinterpret_cast<uint4*>(dst + (lane << 3)) = make_uint4(0u, 0u, 0u, 0u);
}

// Phase 6.2: GQA-grouped + multi-token m16n8k16 mma.sync paged attention (head_dim=256,
// block_size=16) across TQ4, Q8_0, and FP16 KV caches with optional parallel Split-KV.
// Grid: (ceil(total_tokens / tok_tile), local_kv_heads * q_subgroups, num_splits), Block: 128.
__global__ void paged_attention_gqa_mma_f16_kernel(
    w34_kv_layer_layout layout,
    w34_attention_batch batch,
    const __half* __restrict__ q_in,
    __half* __restrict__ attn_out,
    float* __restrict__ partial_acc,
    float* __restrict__ partial_m,
    float* __restrict__ partial_l,
    int num_q_heads,
    int gqa_ratio,
    int q_group,
    int q_subgroups,
    int tok_tile,
    int num_splits) {
    constexpr int HEAD_DIM = 256;
    constexpr int SMEM_STRIDE = 264; // 264 * 2 = 528 B (16B-aligned + bank-conflict-free)
    constexpr int MIN_PAGES_PER_SPLIT = 64; // 1,024 KV tokens minimum per active split

    __shared__ __align__(16) __half s_q[16][SMEM_STRIDE];
    __shared__ __align__(16) __half s_kv[16][SMEM_STRIDE];
    __shared__ __align__(16) __half s_p[16][24];
    __shared__ float s_qk_part[2][16][16];
    __shared__ float s_m[16];
    __shared__ float s_l[16];
    __shared__ float s_alpha[16];
    __shared__ int s_row_tok[16];
    __shared__ int s_row_qhead[16];
    __shared__ int s_min_kv[16];
    __shared__ int s_max_kv[16];
    __shared__ uint8_t s_row_valid[16];
    __shared__ int s_req;
    __shared__ int s_union_min_kv;
    __shared__ int s_union_max_kv;
    __shared__ uint8_t s_any_valid;

    const int tid = threadIdx.x;
    const int lane = tid & 31;
    const int wid = tid >> 5;
    const int group_id = lane >> 2;
    const int tid_in_grp = lane & 3;

    const int tok_base = (int)blockIdx.x * tok_tile;
    const int kv_sub = (int)blockIdx.y;
    const int kv_head = kv_sub / q_subgroups;
    const int q_head_base = kv_head * gqa_ratio + (kv_sub % q_subgroups) * q_group;
    const int split_idx = (int)blockIdx.z;
    const int active_rows = tok_tile * q_group;

    if (tid < 16) {
        uint8_t valid = 0;
        int tok = 0;
        int qh = 0;
        int min_kv = 0;
        int max_kv = -1;
        if (tid < active_rows) {
            const int t_off = tid / q_group;
            const int q_off = tid % q_group;
            tok = tok_base + t_off;
            qh = q_head_base + q_off;
            if (tok < batch.total_tokens &&
                (batch.valid_token_mask == nullptr || batch.valid_token_mask[tok] != 0)) {
                int req = -1;
                int q_start = 0;
                int q_end = 0;
                for (int r = 0; r < batch.num_reqs; ++r) {
                    const int lo = batch.query_start_loc[r];
                    const int hi = batch.query_start_loc[r + 1];
                    if (tok >= lo && tok < hi) {
                        req = r;
                        q_start = lo;
                        q_end = hi;
                        break;
                    }
                }
                if (req >= 0 && (batch.valid_req_mask == nullptr || batch.valid_req_mask[req] != 0)) {
                    valid = 1;
                    s_req = req;
                    const int q_offset_in_req = tok - q_start;
                    const int ctx_before = (batch.context_lens_before != nullptr)
                        ? batch.context_lens_before[req]
                        : (batch.seq_lens_after[req] - (q_end - q_start));
                    const int seq_after = batch.seq_lens_after[req];
                    max_kv = (batch.mask_mode == W34_ATTN_MASK_SYMMETRIC_WINDOW)
                        ? (seq_after - 1)
                        : (ctx_before + q_offset_in_req);
                    if (max_kv >= seq_after) max_kv = seq_after - 1;
                    min_kv = 0;
                    if (batch.sliding_window > 0) {
                        const int ref_pos = ctx_before + q_offset_in_req;
                        const int win_start = ref_pos - batch.sliding_window + 1;
                        if (win_start > min_kv) min_kv = win_start;
                    }
                }
            }
        }
        s_row_valid[tid] = valid;
        s_row_tok[tid] = tok;
        s_row_qhead[tid] = qh;
        s_min_kv[tid] = min_kv;
        s_max_kv[tid] = max_kv;
        s_m[tid] = -FLT_MAX;
        s_l[tid] = 0.0f;
        s_alpha[tid] = 1.0f;
    }
    __syncthreads();

    if (tid == 0) {
        int u_min = INT_MAX;
        int u_max = -1;
        uint8_t any = 0;
        for (int r = 0; r < 16; ++r) {
            if (s_row_valid[r]) {
                any = 1;
                if (s_min_kv[r] <= s_max_kv[r]) {
                    if (s_min_kv[r] < u_min) u_min = s_min_kv[r];
                    if (s_max_kv[r] > u_max) u_max = s_max_kv[r];
                }
            }
        }
        if (u_min == INT_MAX) u_min = 0;
        s_union_min_kv = u_min;
        s_union_max_kv = u_max;
        s_any_valid = any;
    }
    __syncthreads();

    if (!s_any_valid) return;

    const int union_min_kv = s_union_min_kv;
    const int union_max_kv = s_union_max_kv;
    const int first_page = union_min_kv >> 4;
    const int last_page = (union_max_kv >= union_min_kv) ? (union_max_kv >> 4) : (first_page - 1);
    const int total_pages = (last_page >= first_page) ? (last_page - first_page + 1) : 0;

    int my_page_start = first_page;
    int my_page_end = first_page + total_pages;
    int active_splits = 1;
    if (num_splits > 1) {
        int pages_per_split = (total_pages + num_splits - 1) / num_splits;
        if (pages_per_split < MIN_PAGES_PER_SPLIT) pages_per_split = MIN_PAGES_PER_SPLIT;
        active_splits = (total_pages > 0)
            ? ((total_pages + pages_per_split - 1) / pages_per_split)
            : 1;
        if (active_splits > num_splits) active_splits = num_splits;
        if (split_idx >= active_splits) {
            return;
        }
        if (active_splits > 1) {
            my_page_start = first_page + split_idx * pages_per_split;
            my_page_end = my_page_start + pages_per_split;
            if (my_page_end > first_page + total_pages) my_page_end = first_page + total_pages;
        }
    }

    // Stage and pre-rotate Q rows into s_q[16][264].
    for (int r = wid; r < 16; r += 4) {
        if (s_row_valid[r]) {
            const __half* q_row = q_in + ((size_t)s_row_tok[r] * num_q_heads + s_row_qhead[r]) * HEAD_DIM;
            if (layout.k_transform == W34_KV_TRANSFORM_RADEMACHER_WHT256) {
                float q_reg[8];
#pragma unroll
                for (int j = 0; j < 8; ++j) {
                    const int col = lane + 32 * j;
                    q_reg[j] = __half2float(q_row[col]) * kTq4Signs256[col];
                }
                wht256_warp(q_reg, lane);
#pragma unroll
                for (int j = 0; j < 8; ++j) {
                    s_q[r][lane + 32 * j] = __float2half(q_reg[j]);
                }
            } else {
                *reinterpret_cast<uint4*>(&s_q[r][lane << 3]) =
                    *reinterpret_cast<const uint4*>(&q_row[lane << 3]);
            }
        } else {
            *reinterpret_cast<uint4*>(&s_q[r][lane << 3]) = make_uint4(0u, 0u, 0u, 0u);
        }
    }

    float o_frag[8][4];
#pragma unroll
    for (int step = 0; step < 8; ++step) {
        o_frag[step][0] = 0.0f;
        o_frag[step][1] = 0.0f;
        o_frag[step][2] = 0.0f;
        o_frag[step][3] = 0.0f;
    }
    __syncthreads();

    const int32_t* req_blocks = batch.block_tables + (size_t)s_req * batch.max_blocks_per_req;
    const int k_half = wid & 1;
    const int n_base = (wid >> 1) * 8;
    const int row_r = tid >> 3;      // 0..15 (8 threads per query row)
    const int sub_lane = tid & 7;    // 0..7  (2 keys per thread)
    const uint16_t (*s_kv_u16)[SMEM_STRIDE] = reinterpret_cast<const uint16_t (*)[SMEM_STRIDE]>(s_kv);

    for (int page_idx = my_page_start; page_idx < my_page_end; ++page_idx) {
        const int kv_base = page_idx << 4;
        const int32_t phys_page = req_blocks[page_idx];
        const uint8_t* k_page_head = (const uint8_t*)(uintptr_t)layout.k_base
                                     + (size_t)phys_page * layout.k_page_stride_bytes
                                     + (size_t)kv_head * layout.k_head_stride_bytes;
        const uint8_t* v_page_head = (const uint8_t*)(uintptr_t)layout.v_base
                                     + (size_t)phys_page * layout.v_page_stride_bytes
                                     + (size_t)kv_head * layout.v_head_stride_bytes;

        // 1. Cooperative 4-warp decode of 16 K rows into s_kv.
#pragma unroll
        for (int pass = 0; pass < 4; ++pass) {
            const int t = pass * 4 + wid;
            const int kv_pos = kv_base + t;
            if (kv_pos >= union_min_kv && kv_pos <= union_max_kv) {
                decode_kv_row_256_to_smem_f16(
                    k_page_head + (size_t)t * layout.k_token_stride_bytes,
                    layout.k_format,
                    s_kv[t],
                    lane);
            } else {
                *reinterpret_cast<uint4*>(&s_kv[t][lane << 3]) = make_uint4(0u, 0u, 0u, 0u);
            }
        }
        __syncthreads();

        // 2. Q @ K^T ([16, 256] x [16, 256]^T -> [16, 16] FP32) via 8x mma.sync per warp.
        float qk0 = 0.0f, qk1 = 0.0f, qk2 = 0.0f, qk3 = 0.0f;
#pragma unroll
        for (int ks = 0; ks < 8; ++ks) {
            const int col_base = k_half * 128 + ks * 16;
            const uint32_t a0 = *reinterpret_cast<const uint32_t*>(&s_q[group_id][col_base + 2 * tid_in_grp]);
            const uint32_t a1 = *reinterpret_cast<const uint32_t*>(&s_q[group_id + 8][col_base + 2 * tid_in_grp]);
            const uint32_t a2 = *reinterpret_cast<const uint32_t*>(&s_q[group_id][col_base + 8 + 2 * tid_in_grp]);
            const uint32_t a3 = *reinterpret_cast<const uint32_t*>(&s_q[group_id + 8][col_base + 8 + 2 * tid_in_grp]);
            const uint32_t b0 = *reinterpret_cast<const uint32_t*>(&s_kv[n_base + group_id][col_base + 2 * tid_in_grp]);
            const uint32_t b1 = *reinterpret_cast<const uint32_t*>(&s_kv[n_base + group_id][col_base + 8 + 2 * tid_in_grp]);
            asm volatile(
                "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
                "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                : "+f"(qk0), "+f"(qk1), "+f"(qk2), "+f"(qk3)
                : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
        }
        s_qk_part[k_half][group_id][n_base + 2 * tid_in_grp] = qk0;
        s_qk_part[k_half][group_id][n_base + 2 * tid_in_grp + 1] = qk1;
        s_qk_part[k_half][group_id + 8][n_base + 2 * tid_in_grp] = qk2;
        s_qk_part[k_half][group_id + 8][n_base + 2 * tid_in_grp + 1] = qk3;
        __syncthreads();

        // 3. Parallel online FP32 softmax across all 16 query rows (8 lanes/row).
        // Keep all 32 warp lanes converged across __shfl_xor_sync (sm_86 ITS safety).
        const int t0 = sub_lane << 1;
        const int t1 = t0 + 1;
        const bool row_ok = (s_row_valid[row_r] != 0);
        float sc0 = -FLT_MAX;
        float sc1 = -FLT_MAX;
        if (row_ok) {
            const int min_kv = s_min_kv[row_r];
            const int max_kv = s_max_kv[row_r];
            const int kv0 = kv_base + t0;
            const int kv1 = kv_base + t1;
            if (kv0 >= min_kv && kv0 <= max_kv) {
                sc0 = (s_qk_part[0][row_r][t0] + s_qk_part[1][row_r][t0]) * batch.sm_scale;
            }
            if (kv1 >= min_kv && kv1 <= max_kv) {
                sc1 = (s_qk_part[0][row_r][t1] + s_qk_part[1][row_r][t1]) * batch.sm_scale;
            }
        }
        float tile_max = fmaxf(sc0, sc1);
        tile_max = fmaxf(tile_max, __shfl_xor_sync(0xffffffffu, tile_max, 4));
        tile_max = fmaxf(tile_max, __shfl_xor_sync(0xffffffffu, tile_max, 2));
        tile_max = fmaxf(tile_max, __shfl_xor_sync(0xffffffffu, tile_max, 1));
        const bool has_tile = row_ok && (tile_max > -FLT_MAX);
        const float m_old = row_ok ? s_m[row_r] : -FLT_MAX;
        const float m_new = has_tile ? fmaxf(m_old, tile_max) : 0.0f;
        const float alpha = has_tile ? ((m_old > -FLT_MAX) ? expf(m_old - m_new) : 0.0f) : 1.0f;
        const float p0 = (has_tile && sc0 > -FLT_MAX) ? expf(sc0 - m_new) : 0.0f;
        const float p1 = (has_tile && sc1 > -FLT_MAX) ? expf(sc1 - m_new) : 0.0f;
        float p_sum = p0 + p1;
        p_sum += __shfl_xor_sync(0xffffffffu, p_sum, 4);
        p_sum += __shfl_xor_sync(0xffffffffu, p_sum, 2);
        p_sum += __shfl_xor_sync(0xffffffffu, p_sum, 1);
        *reinterpret_cast<__half2*>(&s_p[row_r][t0]) = __floats2half2_rn(p0, p1);
        if (sub_lane == 0) {
            if (has_tile) {
                s_m[row_r] = m_new;
                s_l[row_r] = s_l[row_r] * alpha + p_sum;
            }
            s_alpha[row_r] = alpha;
        }

        // 4. Cooperative 4-warp decode of 16 V rows into s_kv.
#pragma unroll
        for (int pass = 0; pass < 4; ++pass) {
            const int t = pass * 4 + wid;
            const int kv_pos = kv_base + t;
            if (kv_pos >= union_min_kv && kv_pos <= union_max_kv) {
                decode_kv_row_256_to_smem_f16(
                    v_page_head + (size_t)t * layout.v_token_stride_bytes,
                    layout.v_format,
                    s_kv[t],
                    lane);
            } else {
                *reinterpret_cast<uint4*>(&s_kv[t][lane << 3]) = make_uint4(0u, 0u, 0u, 0u);
            }
        }
        __syncthreads();

        // 5. Rescale output accumulators and accumulate P @ V ([16, 16] x [16, 256]) via 8x mma.sync/warp.
        const float al0 = s_alpha[group_id];
        const float al1 = s_alpha[group_id + 8];
        const uint32_t pa0 = *reinterpret_cast<const uint32_t*>(&s_p[group_id][2 * tid_in_grp]);
        const uint32_t pa1 = *reinterpret_cast<const uint32_t*>(&s_p[group_id + 8][2 * tid_in_grp]);
        const uint32_t pa2 = *reinterpret_cast<const uint32_t*>(&s_p[group_id][8 + 2 * tid_in_grp]);
        const uint32_t pa3 = *reinterpret_cast<const uint32_t*>(&s_p[group_id + 8][8 + 2 * tid_in_grp]);
        const int k0 = 2 * tid_in_grp;
        const int k1 = k0 + 1;
        const int k2 = 8 + k0;
        const int k3 = k2 + 1;
#pragma unroll
        for (int step = 0; step < 8; ++step) {
            o_frag[step][0] *= al0;
            o_frag[step][1] *= al0;
            o_frag[step][2] *= al1;
            o_frag[step][3] *= al1;
            const int col = wid * 64 + step * 8 + group_id;
            const uint32_t vb0 = (uint32_t)s_kv_u16[k0][col] | ((uint32_t)s_kv_u16[k1][col] << 16);
            const uint32_t vb1 = (uint32_t)s_kv_u16[k2][col] | ((uint32_t)s_kv_u16[k3][col] << 16);
            asm volatile(
                "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
                "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                : "+f"(o_frag[step][0]), "+f"(o_frag[step][1]), "+f"(o_frag[step][2]), "+f"(o_frag[step][3])
                : "r"(pa0), "r"(pa1), "r"(pa2), "r"(pa3), "r"(vb0), "r"(vb1));
        }
        __syncthreads();
    }

    // Epilogue: reuse s_kv (8,448 B) as float s_out_stage[8][256] (8,192 B) in 2 passes of 8 rows.
    float (*s_out_stage)[HEAD_DIM] = reinterpret_cast<float (*)[HEAD_DIM]>(s_kv);
#pragma unroll
    for (int half_idx = 0; half_idx < 2; ++half_idx) {
#pragma unroll
        for (int step = 0; step < 8; ++step) {
            const int c0 = wid * 64 + step * 8 + 2 * tid_in_grp;
            s_out_stage[group_id][c0] = o_frag[step][2 * half_idx];
            s_out_stage[group_id][c0 + 1] = o_frag[step][2 * half_idx + 1];
        }
        __syncthreads();

        for (int sub = 0; sub < 2; ++sub) {
            const int local_r = sub * 4 + wid;
            const int r = half_idx * 8 + local_r;
            if (s_row_valid[r]) {
                const size_t row_linear = (size_t)s_row_tok[r] * num_q_heads + s_row_qhead[r];
                if (active_splits > 1) {
                    const size_t split_linear = row_linear * num_splits + split_idx;
                    float* dst_acc = partial_acc + split_linear * HEAD_DIM;
#pragma unroll
                    for (int j = 0; j < 8; ++j) {
                        const int col = lane + 32 * j;
                        dst_acc[col] = s_out_stage[local_r][col];
                    }
                    if (lane == 0) {
                        partial_m[split_linear] = s_m[r];
                        partial_l[split_linear] = s_l[r];
                        if (split_idx + 1 == active_splits && active_splits < num_splits) {
                            partial_l[split_linear + 1] = -2.0f;
                        }
                    }
                } else {
                    if (num_splits > 1 && lane == 0) {
                        partial_l[row_linear * num_splits] = -1.0f;
                    }
                    __half* out_row = attn_out + row_linear * HEAD_DIM;
                    const float inv_l = (s_l[r] > 0.0f) ? (1.0f / s_l[r]) : 0.0f;
                    float out_v[8];
#pragma unroll
                    for (int j = 0; j < 8; ++j) {
                        out_v[j] = s_out_stage[local_r][lane + 32 * j] * inv_l;
                    }
                    if (layout.v_transform == W34_KV_TRANSFORM_RADEMACHER_WHT256) {
                        wht256_warp(out_v, lane);
#pragma unroll
                        for (int j = 0; j < 8; ++j) {
                            const int col = lane + 32 * j;
                            out_row[col] = __float2half(out_v[j] * kTq4Signs256[col]);
                        }
                    } else {
#pragma unroll
                        for (int j = 0; j < 8; ++j) {
                            out_row[lane + 32 * j] = __float2half(out_v[j]);
                        }
                    }
                }
            }
        }
        __syncthreads();
    }
}

// Parallel Split-KV online-softmax reduction + single inverse WHT-256 (1 warp = 32 threads
// per (tok, q_head); 4 warps = 4 (tok, q_head) rows per CTA).
__global__ void paged_attention_split_kv_reduce_f16_kernel(
    w34_kv_layer_layout layout,
    w34_attention_batch batch,
    const float* __restrict__ partial_acc,
    const float* __restrict__ partial_m,
    const float* __restrict__ partial_l,
    __half* __restrict__ attn_out,
    int num_q_heads,
    int num_splits) {
    constexpr int HEAD_DIM = 256;
    const int lane = threadIdx.x & 31;
    const int wid = threadIdx.x >> 5;
    const int q_head = (int)blockIdx.y * 4 + wid;
    const int tok = (int)blockIdx.x;
    if (tok >= batch.total_tokens || q_head >= num_q_heads) return;
    if (batch.valid_token_mask != nullptr && batch.valid_token_mask[tok] == 0) return;

    int req = -1;
    for (int r = 0; r < batch.num_reqs; ++r) {
        if (tok >= batch.query_start_loc[r] && tok < batch.query_start_loc[r + 1]) {
            req = r;
            break;
        }
    }
    if (req < 0) return;
    if (batch.valid_req_mask != nullptr && batch.valid_req_mask[req] == 0) return;

    const size_t row_linear = (size_t)tok * num_q_heads + q_head;
    const float* row_l = partial_l + row_linear * num_splits;
    if (row_l[0] < 0.0f) {
        // active_splits == 1 fast path: split 0 already finalized attn_out directly.
        return;
    }
    const float* row_m = partial_m + row_linear * num_splits;
    const float* row_acc = partial_acc + row_linear * num_splits * HEAD_DIM;

    int eff_splits = num_splits;
    float m_tot = -FLT_MAX;
    for (int s = 0; s < num_splits; ++s) {
        const float ls = row_l[s];
        if (ls < -1.5f) {
            eff_splits = s;
            break;
        }
        if (ls > 0.0f) {
            m_tot = fmaxf(m_tot, row_m[s]);
        }
    }

    float out_v[8] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    float l_tot = 0.0f;
    if (m_tot > -FLT_MAX) {
        for (int s = 0; s < eff_splits; ++s) {
            const float ls = row_l[s];
            if (ls > 0.0f) {
                const float w = expf(row_m[s] - m_tot);
                l_tot += ls * w;
                const float* acc_s = row_acc + (size_t)s * HEAD_DIM;
#pragma unroll
                for (int j = 0; j < 8; ++j) {
                    out_v[j] += acc_s[lane + 32 * j] * w;
                }
            }
        }
    }

    const float inv_l = (l_tot > 0.0f) ? (1.0f / l_tot) : 0.0f;
#pragma unroll
    for (int j = 0; j < 8; ++j) {
        out_v[j] *= inv_l;
    }

    __half* out_row = attn_out + row_linear * HEAD_DIM;
    if (layout.v_transform == W34_KV_TRANSFORM_RADEMACHER_WHT256) {
        wht256_warp(out_v, lane);
#pragma unroll
        for (int j = 0; j < 8; ++j) {
            const int col = lane + 32 * j;
            out_row[col] = __float2half(out_v[j] * kTq4Signs256[col]);
        }
    } else {
#pragma unroll
        for (int j = 0; j < 8; ++j) {
            out_row[lane + 32 * j] = __float2half(out_v[j]);
        }
    }
}

// Native BF16 write: no FP16 intermediary (4B drafter values may exceed 65504).
// Supports arbitrary row_stride_elems >= local_kv_heads * head_dim so packed [K, V]
// and [Q, K, V] activations can be written directly without intermediate column copies.
__global__ void kv_cache_write_bf16_kernel(
    w34_kv_layer_layout layout, w34_attention_batch batch,
    const __nv_bfloat16* k, const __nv_bfloat16* v, int row_stride_elems) {
    const int tok = blockIdx.x;
    if (batch.valid_token_mask && !batch.valid_token_mask[tok]) return;
    int req = -1;
    for (int r = 0; r < batch.num_reqs; ++r) {
        if (tok >= batch.query_start_loc[r] && tok < batch.query_start_loc[r + 1]) { req = r; break; }
    }
    if (req < 0 || (batch.valid_req_mask && !batch.valid_req_mask[req])) return;
    const int slot = batch.write_slot_mapping[tok];
    if (slot < 0) return;
    const int head = blockIdx.y;
    const int page = slot / layout.block_size, off = slot % layout.block_size;
    auto* kd = (__nv_bfloat16*)((uint8_t*)(uintptr_t)layout.k_base +
        (size_t)page * layout.k_page_stride_bytes + (size_t)head * layout.k_head_stride_bytes +
        (size_t)off * layout.k_token_stride_bytes);
    auto* vd = (__nv_bfloat16*)((uint8_t*)(uintptr_t)layout.v_base +
        (size_t)page * layout.v_page_stride_bytes + (size_t)head * layout.v_head_stride_bytes +
        (size_t)off * layout.v_token_stride_bytes);
    const size_t base = (size_t)tok * row_stride_elems + (size_t)head * layout.head_dim;
    for (int c = threadIdx.x; c < layout.head_dim; c += blockDim.x) {
        const size_t src = base + c;
        kd[c] = k[src];
        vd[c] = v[src];
    }
}

}  // namespace w34_attn

extern "C" int w34_kv_cache_write_strided(
    const w34_exec* ex,
    const w34_kv_layer_layout* layout,
    const w34_attention_batch* batch,
    const void* k_in,
    const void* v_in,
    int k_row_stride,
    int v_row_stride) {
    if (ex == nullptr || layout == nullptr || batch == nullptr || k_in == nullptr || v_in == nullptr) {
        return (int)cudaErrorInvalidValue;
    }
    if (batch->total_tokens <= 0 || layout->local_kv_heads <= 0) return 0;
    const int min_stride = layout->local_kv_heads * layout->head_dim;
    if (k_row_stride < min_stride || v_row_stride < min_stride) return (int)cudaErrorInvalidValue;
    dim3 grid((unsigned)batch->total_tokens, (unsigned)layout->local_kv_heads);
    w34_attn::kv_cache_write_kernel<<<grid, 128, 0, (cudaStream_t)ex->stream>>>(
        *layout, *batch, (const __half*)k_in, (const __half*)v_in, k_row_stride, v_row_stride);
    return (int)cudaGetLastError();
}

extern "C" int w34_kv_cache_write(const w34_exec* ex, const w34_kv_layer_layout* layout,
                                  const w34_attention_batch* batch,
                                  const void* k_in, const void* v_in) {
    if (layout == nullptr) return (int)cudaErrorInvalidValue;
    const int stride = layout->local_kv_heads * layout->head_dim;
    return w34_kv_cache_write_strided(ex, layout, batch, k_in, v_in, stride, stride);
}

static bool valid_bf16_layout(const w34_kv_layer_layout* l) {
    return l && l->local_kv_heads > 0 && l->block_size > 0 &&
        l->head_dim > 0 && l->head_dim <= 256 && l->k_base && l->v_base &&
        l->k_format == W34_KV_FORMAT_BF16 && l->v_format == W34_KV_FORMAT_BF16 &&
        l->k_transform == W34_KV_TRANSFORM_IDENTITY && l->v_transform == W34_KV_TRANSFORM_IDENTITY;
}

extern "C" int w34_kv_cache_write_strided_bf16(const w34_exec* ex, const w34_kv_layer_layout* layout,
    const w34_attention_batch* batch, const void* k, const void* v, int row_stride_elems) {
    if (!ex || !valid_bf16_layout(layout) || !batch || !k || !v ||
        row_stride_elems < layout->local_kv_heads * layout->head_dim ||
        batch->total_tokens < 0 || batch->num_reqs <= 0 || !batch->query_start_loc ||
        !batch->write_slot_mapping) return (int)cudaErrorInvalidValue;
    if (!batch->total_tokens) return 0;
    dim3 grid((unsigned)batch->total_tokens, (unsigned)layout->local_kv_heads);
    w34_attn::kv_cache_write_bf16_kernel<<<grid, 128, 0, (cudaStream_t)ex->stream>>>(
        *layout, *batch, (const __nv_bfloat16*)k, (const __nv_bfloat16*)v, row_stride_elems);
    return (int)cudaGetLastError();
}

extern "C" int w34_kv_cache_write_bf16(const w34_exec* ex, const w34_kv_layer_layout* layout,
    const w34_attention_batch* batch, const void* k, const void* v) {
    if (!valid_bf16_layout(layout)) return (int)cudaErrorInvalidValue;
    return w34_kv_cache_write_strided_bf16(
        ex, layout, batch, k, v, layout->local_kv_heads * layout->head_dim);
}

extern "C" int w34_paged_attention_bf16(const w34_exec* ex, const w34_kv_layer_layout* layout,
    const w34_attention_batch* batch, const void* q, void* out, int num_q_heads) {
    if (!ex || !valid_bf16_layout(layout) || !batch || !q || !out || num_q_heads <= 0 ||
        num_q_heads % layout->local_kv_heads != 0 || batch->total_tokens < 0 ||
        batch->num_reqs <= 0 || !batch->query_start_loc || !batch->seq_lens_after ||
        !batch->block_tables || batch->max_blocks_per_req <= 0 || batch->sliding_window < 0 ||
        (batch->mask_mode != W34_ATTN_MASK_CAUSAL && batch->mask_mode != W34_ATTN_MASK_SYMMETRIC_WINDOW))
        return (int)cudaErrorInvalidValue;
    if (!batch->total_tokens) return 0;
    dim3 grid((unsigned)batch->total_tokens, (unsigned)num_q_heads);
    w34_attn::paged_attention_kernel<__nv_bfloat16><<<grid, 128, 0, (cudaStream_t)ex->stream>>>(
        *layout, *batch, (const __nv_bfloat16*)q, (__nv_bfloat16*)out, num_q_heads);
    return (int)cudaGetLastError();
}

extern "C" int w34_paged_attention_f16(const w34_exec* ex, const w34_kv_layer_layout* layout,
                                       const w34_attention_batch* batch,
                                       const void* q_in, void* attn_out,
                                       int num_q_heads) {
    if (ex == nullptr || layout == nullptr || batch == nullptr || q_in == nullptr || attn_out == nullptr) {
        return (int)cudaErrorInvalidValue;
    }
    if (batch->total_tokens <= 0 || num_q_heads <= 0) return 0;

    const bool can_use_gqa_mma =
        (layout->head_dim == 256) &&
        (layout->block_size == 16) &&
        (layout->local_kv_heads > 0) &&
        (num_q_heads % layout->local_kv_heads == 0) &&
        (layout->k_format == W34_KV_FORMAT_TQ4 ||
         layout->k_format == W34_KV_FORMAT_Q8_0 ||
         layout->k_format == W34_KV_FORMAT_FP16) &&
        (layout->v_format == W34_KV_FORMAT_TQ4 ||
         layout->v_format == W34_KV_FORMAT_Q8_0 ||
         layout->v_format == W34_KV_FORMAT_FP16);

    if (can_use_gqa_mma) {
        const int gqa_ratio = num_q_heads / layout->local_kv_heads;

        // Parallel Split-KV for M <= 16 when max_seq_len > 1024 (> 64 pages)
        // and aligned scratch workspace is provided in ex->scratch_ptr.
        // Note: num_splits depends ONLY on (total_tokens, max_seq_len, max_blocks_per_req) —
        // never on local_kv_heads — preserving 0-ULP TP=1 == TP=2 parity!
        int num_splits = 1;
        float* partial_acc = nullptr;
        float* partial_m = nullptr;
        float* partial_l = nullptr;
        if (batch->total_tokens <= 16 && batch->max_seq_len > 1024 &&
            batch->max_blocks_per_req > 64 &&
            ex->scratch_ptr != 0 && (ex->scratch_ptr & 15u) == 0u) {
            const int cand_splits =
                (batch->max_blocks_per_req > 1024 && batch->total_tokens <= 4) ? 32 : 16;
            const size_t total_slots = (size_t)batch->total_tokens * (size_t)num_q_heads * (size_t)cand_splits;
            const size_t acc_bytes = total_slots * 256u * sizeof(float);
            const size_t ml_bytes = total_slots * sizeof(float);
            const size_t needed = acc_bytes + 2u * ml_bytes;
            if (ex->scratch_size >= needed) {
                num_splits = cand_splits;
                uint8_t* ws = reinterpret_cast<uint8_t*>((uintptr_t)ex->scratch_ptr);
                partial_acc = reinterpret_cast<float*>(ws);
                partial_m = reinterpret_cast<float*>(ws + acc_bytes);
                partial_l = reinterpret_cast<float*>(ws + acc_bytes + ml_bytes);
            }
        }

        if (batch->total_tokens == 1 && num_splits == 1) {
            dim3 grid(1u, (unsigned)num_q_heads);
            w34_attn::paged_attention_kernel<__half><<<grid, 128, 0, (cudaStream_t)ex->stream>>>(
                *layout, *batch, (const __half*)q_in, (__half*)attn_out, num_q_heads);
            return (int)cudaGetLastError();
        }

        int q_group = gqa_ratio;
        if (q_group > 8) {
            for (int d = 8; d >= 1; --d) {
                if (gqa_ratio % d == 0) {
                    q_group = d;
                    break;
                }
            }
        }
        const int q_subgroups = gqa_ratio / q_group;
        int tok_tile = 1;
        if (batch->num_reqs == 1 && batch->total_tokens > 1) {
            const int max_tt = 16 / q_group;
            tok_tile = (batch->total_tokens < max_tt) ? batch->total_tokens : max_tt;
            if (tok_tile < 1) tok_tile = 1;
        }

        const unsigned grid_x = (unsigned)((batch->total_tokens + tok_tile - 1) / tok_tile);
        const unsigned grid_y = (unsigned)(layout->local_kv_heads * q_subgroups);
        const dim3 grid(grid_x, grid_y, (unsigned)num_splits);
        w34_attn::paged_attention_gqa_mma_f16_kernel<<<grid, 128, 0, (cudaStream_t)ex->stream>>>(
            *layout,
            *batch,
            static_cast<const __half*>(q_in),
            static_cast<__half*>(attn_out),
            partial_acc,
            partial_m,
            partial_l,
            num_q_heads,
            gqa_ratio,
            q_group,
            q_subgroups,
            tok_tile,
            num_splits);
        if (num_splits > 1) {
            const dim3 red_grid(
                (unsigned)batch->total_tokens,
                (unsigned)((num_q_heads + 3) / 4));
            w34_attn::paged_attention_split_kv_reduce_f16_kernel<<<red_grid, 128, 0, (cudaStream_t)ex->stream>>>(
                *layout,
                *batch,
                partial_acc,
                partial_m,
                partial_l,
                static_cast<__half*>(attn_out),
                num_q_heads,
                num_splits);
        }
        return (int)cudaGetLastError();
    }

    dim3 grid((unsigned)batch->total_tokens, (unsigned)num_q_heads);
    w34_attn::paged_attention_kernel<__half><<<grid, 128, 0, (cudaStream_t)ex->stream>>>(
        *layout, *batch, (const __half*)q_in, (__half*)attn_out, num_q_heads);
    return (int)cudaGetLastError();
}
