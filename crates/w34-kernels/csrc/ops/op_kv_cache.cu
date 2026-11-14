// Descriptor-Driven KV Cache Write (FP16, BF16, TQ4, Q8_0), BF16 GQA Paged Attention,
// and Scalar/Vector Paged Attention Reference Fallback Kernels for warp-34.
#include "../tq/tq4_kv_common.cuh"

#include <type_traits>

namespace w34_attn {

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
    if (head_dim == 256 && format == W34_KV_FORMAT_FP8_E4M3) {
#if __CUDA_ARCH__ >= 890
        if ((int)(threadIdx.x >> 5) == target_warp) {
            const int lane = threadIdx.x & 31;
            block_fp8_e4m3_256* blk = reinterpret_cast<block_fp8_e4m3_256*>(dst_row);
            if (transform == W34_KV_TRANSFORM_RADEMACHER_WHT256) {
                float w[8];
#pragma unroll
                for (int j = 0; j < 8; ++j) {
                    const int col = 32 * j + lane;
                    w[j] = __half2float(src_row[col]) * kTq4Signs256[col];
                }
                wht256_warp(w, lane);
                float amax0 = fmaxf(fmaxf(fabsf(w[0]), fabsf(w[1])), fmaxf(fabsf(w[2]), fabsf(w[3])));
                float amax1 = fmaxf(fmaxf(fabsf(w[4]), fabsf(w[5])), fmaxf(fabsf(w[6]), fabsf(w[7])));
                amax0 = warp_reduce_max(amax0);
                amax1 = warp_reduce_max(amax1);
                const float d0 = amax0 * (1.0f / 448.0f);
                const float d1 = amax1 * (1.0f / 448.0f);
                const float inv_d0 = (d0 > 1e-30f) ? (1.0f / d0) : 0.0f;
                const float inv_d1 = (d1 > 1e-30f) ? (1.0f / d1) : 0.0f;
                if (lane == 0) {
                    blk->d0 = d0;
                    blk->d1 = d1;
                }
                const uint16_t p01 = __nv_cvt_float2_to_fp8x2(make_float2(w[0] * inv_d0, w[1] * inv_d0), __NV_SATFINITE, __NV_E4M3);
                const uint16_t p23 = __nv_cvt_float2_to_fp8x2(make_float2(w[2] * inv_d0, w[3] * inv_d0), __NV_SATFINITE, __NV_E4M3);
                const uint16_t p45 = __nv_cvt_float2_to_fp8x2(make_float2(w[4] * inv_d1, w[5] * inv_d1), __NV_SATFINITE, __NV_E4M3);
                const uint16_t p67 = __nv_cvt_float2_to_fp8x2(make_float2(w[6] * inv_d1, w[7] * inv_d1), __NV_SATFINITE, __NV_E4M3);
                blk->qs[lane]       = static_cast<uint8_t>(p01 & 0xFFu);
                blk->qs[lane + 32]  = static_cast<uint8_t>(p01 >> 8);
                blk->qs[lane + 64]  = static_cast<uint8_t>(p23 & 0xFFu);
                blk->qs[lane + 96]  = static_cast<uint8_t>(p23 >> 8);
                blk->qs[lane + 128] = static_cast<uint8_t>(p45 & 0xFFu);
                blk->qs[lane + 160] = static_cast<uint8_t>(p45 >> 8);
                blk->qs[lane + 192] = static_cast<uint8_t>(p67 & 0xFFu);
                blk->qs[lane + 224] = static_cast<uint8_t>(p67 >> 8);
            } else {
                float v[8];
                const __half* src8 = src_row + (lane << 3);
                if ((((uintptr_t)src8) & 15u) == 0u) {
                    const uint4 raw = *reinterpret_cast<const uint4*>(src8);
                    const __half2 h0 = *reinterpret_cast<const __half2*>(&raw.x);
                    const __half2 h1 = *reinterpret_cast<const __half2*>(&raw.y);
                    const __half2 h2 = *reinterpret_cast<const __half2*>(&raw.z);
                    const __half2 h3 = *reinterpret_cast<const __half2*>(&raw.w);
                    v[0] = __half2float(h0.x); v[1] = __half2float(h0.y);
                    v[2] = __half2float(h1.x); v[3] = __half2float(h1.y);
                    v[4] = __half2float(h2.x); v[5] = __half2float(h2.y);
                    v[6] = __half2float(h3.x); v[7] = __half2float(h3.y);
                } else {
#pragma unroll
                    for (int j = 0; j < 8; ++j) v[j] = __half2float(src8[j]);
                }
                // Lanes 0..15 hold [0..128), lanes 16..31 hold [128..256); xor 8,4,2,1 stays in half-warp.
                float amax = 0.0f;
#pragma unroll
                for (int j = 0; j < 8; ++j) amax = fmaxf(amax, fabsf(v[j]));
#pragma unroll
                for (int off = 8; off > 0; off >>= 1) {
                    amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, off));
                }
                const float d = amax * (1.0f / 448.0f);
                const float inv_d = (d > 1e-30f) ? (1.0f / d) : 0.0f;
                if (lane == 0) blk->d0 = d;
                if (lane == 16) blk->d1 = d;
                const uint16_t p0 = __nv_cvt_float2_to_fp8x2(make_float2(v[0] * inv_d, v[1] * inv_d), __NV_SATFINITE, __NV_E4M3);
                const uint16_t p1 = __nv_cvt_float2_to_fp8x2(make_float2(v[2] * inv_d, v[3] * inv_d), __NV_SATFINITE, __NV_E4M3);
                const uint16_t p2 = __nv_cvt_float2_to_fp8x2(make_float2(v[4] * inv_d, v[5] * inv_d), __NV_SATFINITE, __NV_E4M3);
                const uint16_t p3 = __nv_cvt_float2_to_fp8x2(make_float2(v[6] * inv_d, v[7] * inv_d), __NV_SATFINITE, __NV_E4M3);
                uint8_t* dst8 = blk->qs + (lane << 3);
                if ((((uintptr_t)dst8) & 7u) == 0u) {
                    *reinterpret_cast<uint2*>(dst8) = make_uint2(
                        static_cast<uint32_t>(p0) | (static_cast<uint32_t>(p1) << 16),
                        static_cast<uint32_t>(p2) | (static_cast<uint32_t>(p3) << 16));
                } else {
                    dst8[0] = static_cast<uint8_t>(p0 & 0xFFu); dst8[1] = static_cast<uint8_t>(p0 >> 8);
                    dst8[2] = static_cast<uint8_t>(p1 & 0xFFu); dst8[3] = static_cast<uint8_t>(p1 >> 8);
                    dst8[4] = static_cast<uint8_t>(p2 & 0xFFu); dst8[5] = static_cast<uint8_t>(p2 >> 8);
                    dst8[6] = static_cast<uint8_t>(p3 & 0xFFu); dst8[7] = static_cast<uint8_t>(p3 >> 8);
                }
            }
        }
#endif
        return;
    }
    if (head_dim == 256 && transform == W34_KV_TRANSFORM_RADEMACHER_WHT256 &&
        format == W34_KV_FORMAT_TQ4) {
        // TurboQuant Algorithm 1 (arXiv:2504.19874) at b=4 with Seed-42 Rademacher +
        // WHT-256 rotation, per-128 post-WHT sub-block RMS normalization (matching
        // Beta(256) codebook variance 1/256 with zero sub-block norm dispersion), and
        // per-128 least-squares optimal scaling d_b = <v_b, c_b> / ||c_b||^2.
        // Runs on `target_warp` (0 for K, 1 for V) so K and V encode concurrently.
        if ((int)(threadIdx.x >> 5) == target_warp) {
            const int lane = threadIdx.x & 31;
            block_tq4_0* row2 = (block_tq4_0*)dst_row;
            float v[8];
#pragma unroll
            for (int j = 0; j < 8; ++j) {
                const int col = 32 * j + lane;
                v[j] = __half2float(src_row[col]) * kTq4Signs256[col];
            }
            wht256_warp(v, lane);

            float s_sq0 = 0.0f, s_sq1 = 0.0f;
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                s_sq0 += warp_reduce_sum(v[j] * v[j]);
                s_sq1 += warp_reduce_sum(v[j + 4] * v[j + 4]);
            }
            const float inv_norm0 =
                (s_sq0 > 1e-20f) ? (0.7071067811865475f * rsqrtf(s_sq0)) : 0.0f;
            const float inv_norm1 =
                (s_sq1 > 1e-20f) ? (0.7071067811865475f * rsqrtf(s_sq1)) : 0.0f;

            float vc_dot0 = 0.0f, c_sq0 = 0.0f;
            float vc_dot1 = 0.0f, c_sq1 = 0.0f;
#pragma unroll
            for (int j = 0; j < 8; ++j) {
                const float raw_v = v[j];
                const float rv = raw_v * ((j < 4) ? inv_norm0 : inv_norm1);
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
                const float vc = warp_reduce_sum(raw_v * c);
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
                const float d0 = (c_sq0 > 1e-10f) ? (vc_dot0 / c_sq0) : 0.0f;
                const float d1 = (c_sq1 > 1e-10f) ? (vc_dot1 / c_sq1) : 0.0f;
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
            float x = 0.0f;
            if (threadIdx.x < 128) {
                x = __half2float(sub_src[threadIdx.x]);
                sq = x * x;
            }
            float norm_x = sqrtf(block_reduce_sum_128(sq, s_warp));
            float inv_norm = (norm_x > 1e-12f) ? (0.7071067811865475f / norm_x) : 0.0f;

            // Find nearest Beta(256) centroid for RMS-normalized 128-element sub-block
            uint8_t q_idx = 8;
            float c_val = 0.0f;
            if (threadIdx.x < 128) {
                float u = x * inv_norm;
                q_idx = 0;
#pragma unroll
                for (int b = 0; b < 15; b++) {
                    if (u > kTq4D256Boundaries[b]) q_idx = (uint8_t)(b + 1);
                }
                c_val = kTq4D256Centroids[q_idx];
            }
            float xc_dot = block_reduce_sum_128(x * c_val, s_warp);
            float c_sq = block_reduce_sum_128(c_val * c_val, s_warp);
            float scale_d = (c_sq > 1e-12f) ? (xc_dot / c_sq) : 0.0f;
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
#if __CUDA_ARCH__ >= 890
    if (format == W34_KV_FORMAT_FP8_E4M3) {
        const block_fp8_e4m3_256* blk = reinterpret_cast<const block_fp8_e4m3_256*>(row_ptr);
        const float d = (col < 128) ? blk->d0 : blk->d1;
        const __half_raw hr = __nv_cvt_fp8_to_halfraw(blk->qs[col], __NV_E4M3);
        return d * __half2float(*reinterpret_cast<const __half*>(&hr));
    }
#endif
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
    if (batch.valid_req_mask != nullptr && batch.query_start_loc != nullptr && batch.num_reqs > 0) {
        int req = -1;
        for (int r = 0; r < batch.num_reqs; ++r) {
            if (tok >= batch.query_start_loc[r] && tok < batch.query_start_loc[r + 1]) {
                req = r;
                break;
            }
        }
        if (req < 0 || batch.valid_req_mask[req] == 0) return;
    }

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
                // Use all eight N columns of each MMA for independent keys.
                // Each QK dot retains the same K=16 contraction order; only the
                // previously zero columns are populated, preserving FA2 rounding.
                for (int t = wid * 8; t < tile; t += 32) {
                    const int kv = base + t + (lane >> 2);
                    const bool valid_k = kv >= min_kv_pos && kv <= max_kv_pos;
                    const uint32_t* kp32 = nullptr;
                    if (valid_k) {
                        const uint8_t* kp = (const uint8_t*)(uintptr_t)layout.k_base
                            + (size_t)req_blocks[kv / layout.block_size] * layout.k_page_stride_bytes
                            + (size_t)kv_head * layout.k_head_stride_bytes
                            + (size_t)(kv % layout.block_size) * layout.k_token_stride_bytes;
                        kp32 = (const uint32_t*)kp;
                    }
                    float d0 = 0.f, d1 = 0.f, d2 = 0.f, d3 = 0.f;
                    for (int ks = 0; ks < k_steps; ++ks) {
                        const int pair = lane & 3;
                        uint32_t b0 = valid_k ? kp32[ks * 8 + pair] : 0u;
                        uint32_t b1 = valid_k ? kp32[ks * 8 + 4 + pair] : 0u;
                        uint32_t z = 0u;
                        asm volatile(
                            "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
                            "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                            : "+f"(d0), "+f"(d1), "+f"(d2), "+f"(d3)
                            : "r"(qa0[ks]), "r"(z), "r"(qa2[ks]), "r"(z), "r"(b0), "r"(b1));
                    }
                    if (lane < 4) {
                        const int kv0 = base + t + 2 * lane;
                        const int kv1 = kv0 + 1;
                        scores[t + 2 * lane] = (kv0 >= min_kv_pos && kv0 <= max_kv_pos) ? d0 : -FLT_MAX;
                        scores[t + 2 * lane + 1] = (kv1 >= min_kv_pos && kv1 <= max_kv_pos) ? d1 : -FLT_MAX;
                    }
                    if (lane < 8) {
                        const int pos = base + t + lane;
                        v_ptrs[t + lane] = (pos >= min_kv_pos && pos <= max_kv_pos)
                            ? (const uint8_t*)(uintptr_t)layout.v_base
                                + (size_t)req_blocks[pos / layout.block_size] * layout.v_page_stride_bytes
                                + (size_t)kv_head * layout.v_head_stride_bytes
                                + (size_t)(pos % layout.block_size) * layout.v_token_stride_bytes
                            : nullptr;
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
    if (layout->k_format == W34_KV_FORMAT_FP8_E4M3 || layout->v_format == W34_KV_FORMAT_FP8_E4M3) {
        if (layout->head_dim != 256) return (int)cudaErrorInvalidValue;
        if (!w34_attn::current_device_supports_fp8_kv()) return (int)cudaErrorNotSupported;
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

namespace w34_attn {

// GQA-grouped m16n8k16 BF16 tensor-core paged attention (head_dim in {128, 256}, block_size=16).
// Preserves 0-ULP FlashInfer sm86 FA2 rounding parity with `paged_attention_kernel<__nv_bfloat16>`
// while sharing coalesced 128-bit K/V page loads and m16n8k16 tiles across `q_group` GQA heads.
// Grid: (total_tokens, local_kv_heads * q_subgroups), Block: 128 threads (4 warps).
template <int HEAD_DIM>
__global__ void paged_attention_gqa_bf16_kernel(
    w34_kv_layer_layout layout,
    w34_attention_batch batch,
    const __nv_bfloat16* __restrict__ q_in,
    __nv_bfloat16* __restrict__ attn_out,
    int num_q_heads,
    int gqa_ratio,
    int q_group,
    int q_subgroups) {
    constexpr int TILE = (HEAD_DIM <= 128) ? 64 : 16;
    constexpr int K_STEPS = HEAD_DIM >> 4;
    constexpr int V_STEPS = HEAD_DIM >> 5;
    constexpr int WARP_COLS = HEAD_DIM >> 2;
    constexpr int VEC_PER_ROW = HEAD_DIM >> 3;
    constexpr int SMEM_STRIDE = HEAD_DIM + 8;
    constexpr int P_STRIDE = TILE + 8;

    const int tok = (int)blockIdx.x;
    if (tok >= batch.total_tokens) return;
    if (batch.valid_token_mask != nullptr && batch.valid_token_mask[tok] == 0) return;

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
    if (req < 0) return;
    if (batch.valid_req_mask != nullptr && batch.valid_req_mask[req] == 0) return;

    const int kv_sub = (int)blockIdx.y;
    const int kv_head = kv_sub / q_subgroups;
    const int q_head_base = kv_head * gqa_ratio + (kv_sub % q_subgroups) * q_group;

    const int q_offset_in_req = tok - q_start;
    const int ctx_before = (batch.context_lens_before != nullptr)
        ? batch.context_lens_before[req]
        : (batch.seq_lens_after[req] - (q_end - q_start));
    const int seq_after = batch.seq_lens_after[req];

    int max_kv_pos = (batch.mask_mode == W34_ATTN_MASK_SYMMETRIC_WINDOW)
        ? (seq_after - 1)
        : (ctx_before + q_offset_in_req);
    if (max_kv_pos >= seq_after) max_kv_pos = seq_after - 1;

    int min_kv_pos = 0;
    if (batch.sliding_window > 0) {
        const int ref_pos = ctx_before + q_offset_in_req;
        const int win_start = ref_pos - batch.sliding_window + 1;
        if (win_start > min_kv_pos) min_kv_pos = win_start;
    }

    const int tid = threadIdx.x;
    const int wid = tid >> 5;
    const int lane = tid & 31;
    const int group_id = lane >> 2;
    const int tid_in_grp = lane & 3;

    if (min_kv_pos > max_kv_pos) {
        for (int idx = tid; idx < q_group * VEC_PER_ROW; idx += 128) {
            const int r = idx / VEC_PER_ROW;
            const int v = idx % VEC_PER_ROW;
            __nv_bfloat16* out_row = attn_out + ((size_t)tok * num_q_heads + q_head_base + r) * HEAD_DIM;
            *reinterpret_cast<uint4*>(out_row + (v << 3)) = make_uint4(0u, 0u, 0u, 0u);
        }
        return;
    }

    __shared__ __align__(16) uint16_t s_q[16][SMEM_STRIDE];
    __shared__ __align__(16) uint16_t s_kv[TILE][SMEM_STRIDE];
    __shared__ float scores[16][TILE + 4];
    __shared__ __align__(16) uint16_t p_bits[16][P_STRIDE];
    __shared__ float s_max[16];
    __shared__ float s_alpha[16];
    __shared__ float s_denom[16];

    for (int idx = tid; idx < 16 * VEC_PER_ROW; idx += 128) {
        const int r = idx / VEC_PER_ROW;
        const int v = idx % VEC_PER_ROW;
        if (r < q_group) {
            const __nv_bfloat16* q_row = q_in + ((size_t)tok * num_q_heads + q_head_base + r) * HEAD_DIM;
            *reinterpret_cast<uint4*>(&s_q[r][v << 3]) =
                *reinterpret_cast<const uint4*>(q_row + (v << 3));
        } else {
            *reinterpret_cast<uint4*>(&s_q[r][v << 3]) = make_uint4(0u, 0u, 0u, 0u);
        }
    }
    if (tid < 16) {
        s_max[tid] = -FLT_MAX;
        s_alpha[tid] = 0.f;
        s_denom[tid] = 0.f;
    }
    __syncthreads();

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

    float num_frag[V_STEPS][4];
#pragma unroll
    for (int step = 0; step < V_STEPS; ++step) {
        num_frag[step][0] = 0.f;
        num_frag[step][1] = 0.f;
        num_frag[step][2] = 0.f;
        num_frag[step][3] = 0.f;
    }

    const int32_t* req_blocks = batch.block_tables + (size_t)req * batch.max_blocks_per_req;
    const float scale = batch.sm_scale * 1.4426950408889634f;
    const int start_base = (min_kv_pos > 0) ? ((min_kv_pos / TILE) * TILE) : 0;
    const int row_r = tid >> 3;
    const int sub_lane = tid & 7;

    for (int base = start_base; base <= max_kv_pos; base += TILE) {
        // 1. Coalesced 128-bit load of TILE K-rows into s_kv.
        for (int idx = tid; idx < TILE * VEC_PER_ROW; idx += 128) {
            const int t = idx / VEC_PER_ROW;
            const int v = idx % VEC_PER_ROW;
            const int kv = base + t;
            if (kv >= min_kv_pos && kv <= max_kv_pos) {
                const uint8_t* kp = (const uint8_t*)(uintptr_t)layout.k_base
                    + (size_t)req_blocks[kv >> 4] * layout.k_page_stride_bytes
                    + (size_t)kv_head * layout.k_head_stride_bytes
                    + (size_t)(kv & 15) * layout.k_token_stride_bytes;
                *reinterpret_cast<uint4*>(&s_kv[t][v << 3]) =
                    *reinterpret_cast<const uint4*>(kp + (v << 4));
            } else {
                *reinterpret_cast<uint4*>(&s_kv[t][v << 3]) = make_uint4(0u, 0u, 0u, 0u);
            }
        }
        __syncthreads();

        // 2. Q @ K^T ([16, HEAD_DIM] x [TILE, HEAD_DIM]^T) with identical sequential K_STEPS chain.
        for (int t = wid * 8; t < TILE; t += 32) {
            float d0 = 0.f, d1 = 0.f, d2 = 0.f, d3 = 0.f;
#pragma unroll
            for (int ks = 0; ks < K_STEPS; ++ks) {
                const int col = ks * 16 + 2 * tid_in_grp;
                const uint32_t a0 = *reinterpret_cast<const uint32_t*>(&s_q[group_id][col]);
                const uint32_t a1 = *reinterpret_cast<const uint32_t*>(&s_q[group_id + 8][col]);
                const uint32_t a2 = *reinterpret_cast<const uint32_t*>(&s_q[group_id][col + 8]);
                const uint32_t a3 = *reinterpret_cast<const uint32_t*>(&s_q[group_id + 8][col + 8]);
                const uint32_t b0 = *reinterpret_cast<const uint32_t*>(&s_kv[t + group_id][col]);
                const uint32_t b1 = *reinterpret_cast<const uint32_t*>(&s_kv[t + group_id][col + 8]);
                asm volatile(
                    "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
                    "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                    : "+f"(d0), "+f"(d1), "+f"(d2), "+f"(d3)
                    : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
            }
            const int kv0 = base + t + 2 * tid_in_grp;
            const int kv1 = kv0 + 1;
            const bool v0 = (kv0 >= min_kv_pos && kv0 <= max_kv_pos);
            const bool v1 = (kv1 >= min_kv_pos && kv1 <= max_kv_pos);
            scores[group_id][t + 2 * tid_in_grp] = v0 ? d0 : -FLT_MAX;
            scores[group_id][t + 2 * tid_in_grp + 1] = v1 ? d1 : -FLT_MAX;
            scores[group_id + 8][t + 2 * tid_in_grp] = v0 ? d2 : -FLT_MAX;
            scores[group_id + 8][t + 2 * tid_in_grp + 1] = v1 ? d3 : -FLT_MAX;
        }
        __syncthreads();

        // 3. Tile max, rescaling alpha, and BF16 probability rounding + 4. Coalesced load of TILE V-rows.
        const float prev_max = s_max[row_r];
        float next_max = prev_max;
#pragma unroll
        for (int t = sub_lane; t < TILE; t += 8) {
            next_max = fmaxf(next_max, scores[row_r][t]);
        }
        next_max = fmaxf(next_max, __shfl_xor_sync(0xffffffffu, next_max, 4));
        next_max = fmaxf(next_max, __shfl_xor_sync(0xffffffffu, next_max, 2));
        next_max = fmaxf(next_max, __shfl_xor_sync(0xffffffffu, next_max, 1));
        const float alpha = (next_max == -FLT_MAX) ? 0.f : ptx_exp2(prev_max * scale - next_max * scale);
        __syncwarp();
        if (sub_lane == 0) {
            s_max[row_r] = next_max;
            s_alpha[row_r] = alpha;
            s_denom[row_r] *= alpha;
        }
#pragma unroll
        for (int t = sub_lane; t < TILE; t += 8) {
            const int kv = base + t;
            p_bits[row_r][t] = (row_r < q_group && kv >= min_kv_pos && kv <= max_kv_pos)
                ? __bfloat16_as_ushort(__float2bfloat16_rn(ptx_exp2(scores[row_r][t] * scale - next_max * scale)))
                : 0u;
        }
        for (int idx = tid; idx < TILE * VEC_PER_ROW; idx += 128) {
            const int t = idx / VEC_PER_ROW;
            const int v = idx % VEC_PER_ROW;
            const int kv = base + t;
            if (kv >= min_kv_pos && kv <= max_kv_pos) {
                const uint8_t* vp = (const uint8_t*)(uintptr_t)layout.v_base
                    + (size_t)req_blocks[kv >> 4] * layout.v_page_stride_bytes
                    + (size_t)kv_head * layout.v_head_stride_bytes
                    + (size_t)(kv & 15) * layout.v_token_stride_bytes;
                *reinterpret_cast<uint4*>(&s_kv[t][v << 3]) =
                    *reinterpret_cast<const uint4*>(vp + (v << 4));
            } else {
                *reinterpret_cast<uint4*>(&s_kv[t][v << 3]) = make_uint4(0u, 0u, 0u, 0u);
            }
        }
        __syncthreads();

        // 5. Rescale accumulators once per tile, then run 16-key m16k16_rowsum + PV MMA contractions.
        const float al0 = s_alpha[group_id];
        const float al1 = s_alpha[group_id + 8];
#pragma unroll
        for (int step = 0; step < V_STEPS; ++step) {
            num_frag[step][0] *= al0;
            num_frag[step][1] *= al0;
            num_frag[step][2] *= al1;
            num_frag[step][3] *= al1;
        }
        float den0 = (wid == 0 && tid_in_grp == 0) ? s_denom[group_id] : 0.f;
        float den1 = (wid == 0 && tid_in_grp == 0) ? s_denom[group_id + 8] : 0.f;
        for (int chunk = 0; chunk < TILE; chunk += 16) {
            const uint32_t pa0 = *reinterpret_cast<const uint32_t*>(&p_bits[group_id][chunk + 2 * tid_in_grp]);
            const uint32_t pa1 = *reinterpret_cast<const uint32_t*>(&p_bits[group_id + 8][chunk + 2 * tid_in_grp]);
            const uint32_t pa2 = *reinterpret_cast<const uint32_t*>(&p_bits[group_id][chunk + 8 + 2 * tid_in_grp]);
            const uint32_t pa3 = *reinterpret_cast<const uint32_t*>(&p_bits[group_id + 8][chunk + 8 + 2 * tid_in_grp]);
            if (wid == 0) {
                float d0 = den0, d1 = 0.f, d2 = den1, d3 = 0.f;
                const uint32_t ones = (lane < 4) ? 0x3f803f80u : 0u;
                asm volatile(
                    "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
                    "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                    : "+f"(d0), "+f"(d1), "+f"(d2), "+f"(d3)
                    : "r"(pa0), "r"(pa1), "r"(pa2), "r"(pa3), "r"(ones), "r"(ones));
                if (tid_in_grp == 0) {
                    den0 = d0;
                    den1 = d2;
                }
            }
            const int k0 = chunk + 2 * tid_in_grp;
            const int k1 = k0 + 1;
            const int k2 = k0 + 8;
            const int k3 = k2 + 1;
#pragma unroll
            for (int step = 0; step < V_STEPS; ++step) {
                const int col = wid * WARP_COLS + step * 8 + group_id;
                const uint32_t vb0 = (uint32_t)s_kv[k0][col] | ((uint32_t)s_kv[k1][col] << 16);
                const uint32_t vb1 = (uint32_t)s_kv[k2][col] | ((uint32_t)s_kv[k3][col] << 16);
                asm volatile(
                    "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
                    "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                    : "+f"(num_frag[step][0]), "+f"(num_frag[step][1]),
                      "+f"(num_frag[step][2]), "+f"(num_frag[step][3])
                    : "r"(pa0), "r"(pa1), "r"(pa2), "r"(pa3), "r"(vb0), "r"(vb1));
            }
        }
        if (wid == 0 && tid_in_grp == 0) {
            s_denom[group_id] = den0;
            s_denom[group_id + 8] = den1;
        }
        __syncthreads();
    }

    if (group_id < q_group) {
        const float d_rcp = (s_max[group_id] > -FLT_MAX && s_denom[group_id] > 0.f)
            ? ptx_rcp(s_denom[group_id])
            : 0.f;
        __nv_bfloat16* out_row = attn_out + ((size_t)tok * num_q_heads + q_head_base + group_id) * HEAD_DIM;
#pragma unroll
        for (int step = 0; step < V_STEPS; ++step) {
            const int col0 = wid * WARP_COLS + step * 8 + 2 * tid_in_grp;
            out_row[col0] = __float2bfloat16_rn(num_frag[step][0] * d_rcp);
            out_row[col0 + 1] = __float2bfloat16_rn(num_frag[step][1] * d_rcp);
        }
    }
    if (group_id + 8 < q_group) {
        const float d_rcp = (s_max[group_id + 8] > -FLT_MAX && s_denom[group_id + 8] > 0.f)
            ? ptx_rcp(s_denom[group_id + 8])
            : 0.f;
        __nv_bfloat16* out_row = attn_out + ((size_t)tok * num_q_heads + q_head_base + group_id + 8) * HEAD_DIM;
#pragma unroll
        for (int step = 0; step < V_STEPS; ++step) {
            const int col0 = wid * WARP_COLS + step * 8 + 2 * tid_in_grp;
            out_row[col0] = __float2bfloat16_rn(num_frag[step][2] * d_rcp);
            out_row[col0 + 1] = __float2bfloat16_rn(num_frag[step][3] * d_rcp);
        }
    }
}

} // namespace w34_attn

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
    const bool can_use_gqa_bf16 =
        (layout->head_dim == 128 || layout->head_dim == 256) &&
        (layout->block_size == 16) &&
        ((layout->k_base & 15u) == 0u) &&
        ((layout->v_base & 15u) == 0u) &&
        ((layout->k_page_stride_bytes & 15u) == 0u) &&
        ((layout->k_head_stride_bytes & 15u) == 0u) &&
        ((layout->k_token_stride_bytes & 15u) == 0u) &&
        ((layout->v_page_stride_bytes & 15u) == 0u) &&
        ((layout->v_head_stride_bytes & 15u) == 0u) &&
        ((layout->v_token_stride_bytes & 15u) == 0u) &&
        ((((uintptr_t)q) & 15u) == 0u);
    if (can_use_gqa_bf16) {
        const int gqa_ratio = num_q_heads / layout->local_kv_heads;
        int q_group = gqa_ratio;
        if (q_group > 16) {
            for (int d = 16; d >= 1; --d) {
                if (gqa_ratio % d == 0) {
                    q_group = d;
                    break;
                }
            }
        }
        const int q_subgroups = gqa_ratio / q_group;
        dim3 grid((unsigned)batch->total_tokens, (unsigned)(layout->local_kv_heads * q_subgroups));
        if (layout->head_dim == 128) {
            w34_attn::paged_attention_gqa_bf16_kernel<128><<<grid, 128, 0, (cudaStream_t)ex->stream>>>(
                *layout, *batch, (const __nv_bfloat16*)q, (__nv_bfloat16*)out,
                num_q_heads, gqa_ratio, q_group, q_subgroups);
        } else {
            w34_attn::paged_attention_gqa_bf16_kernel<256><<<grid, 128, 0, (cudaStream_t)ex->stream>>>(
                *layout, *batch, (const __nv_bfloat16*)q, (__nv_bfloat16*)out,
                num_q_heads, gqa_ratio, q_group, q_subgroups);
        }
        return (int)cudaGetLastError();
    }
    dim3 grid((unsigned)batch->total_tokens, (unsigned)num_q_heads);
    w34_attn::paged_attention_kernel<__nv_bfloat16><<<grid, 128, 0, (cudaStream_t)ex->stream>>>(
        *layout, *batch, (const __nv_bfloat16*)q, (__nv_bfloat16*)out, num_q_heads);
    return (int)cudaGetLastError();
}


extern "C" int w34_paged_attention_scalar_f16(
    const w34_exec* ex,
    const w34_kv_layer_layout* layout,
    const w34_attention_batch* batch,
    const void* q_in,
    void* attn_out,
    int num_q_heads) {
    if (!ex || !layout || !batch || !q_in || !attn_out || num_q_heads <= 0 ||
        layout->local_kv_heads <= 0 || num_q_heads % layout->local_kv_heads != 0 ||
        batch->total_tokens < 0 || batch->num_reqs <= 0 || !batch->query_start_loc ||
        !batch->seq_lens_after || !batch->block_tables || batch->max_blocks_per_req <= 0) {
        return (int)cudaErrorInvalidValue;
    }
    if (layout->k_format == W34_KV_FORMAT_FP8_E4M3 || layout->v_format == W34_KV_FORMAT_FP8_E4M3) {
        if (layout->head_dim != 256) return (int)cudaErrorInvalidValue;
        if (!w34_attn::current_device_supports_fp8_kv()) return (int)cudaErrorNotSupported;
    }
    if (batch->total_tokens == 0) return 0;
    dim3 grid((unsigned)batch->total_tokens, (unsigned)num_q_heads);
    w34_attn::paged_attention_kernel<__half><<<grid, 128, 0, (cudaStream_t)ex->stream>>>(
        *layout, *batch, (const __half*)q_in, (__half*)attn_out, num_q_heads);
    return (int)cudaGetLastError();
}
