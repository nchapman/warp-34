// Descriptor-Driven GQA Tensor-Core (mma.sync.m16n8k16) Paged Attention & Split-KV Kernels
// across FP16, TQ4 (132B/row WHT-256), and Q8_0 (272B/row) KV caches.
#include "../tq/tq4_kv_common.cuh"

#include <atomic>
#include <type_traits>

extern "C" int w34_paged_attention_scalar_f16(
    const w34_exec* ex,
    const w34_kv_layer_layout* layout,
    const w34_attention_batch* batch,
    const void* q_in,
    void* attn_out,
    int num_q_heads);

namespace w34_attn {

// Cooperative 1-warp (32-lane) decode of one head_dim=256 KV row (FP16, TQ4, or Q8_0)
// into 16-byte-aligned shared memory row `dst[264]`.
// `tq4_lut` holds `kTq4D256Centroids[lane & 15]` in a register so all 8 nibble lookups
// execute via 1-cycle `__shfl_sync` instead of 16-way serialized `__constant__` loads.
__device__ __forceinline__ void decode_kv_row_256_to_smem_f16(
    const uint8_t* src_ptr,
    int format,
    __half* dst,
    int lane,
    float tq4_lut) {
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
        const float c00 = __shfl_sync(0xffffffffu, tq4_lut, (int)(p0 & 0x0Fu));
        const float c01 = __shfl_sync(0xffffffffu, tq4_lut, (int)((p0 >> 4) & 0x0Fu));
        const float c02 = __shfl_sync(0xffffffffu, tq4_lut, (int)((p0 >> 8) & 0x0Fu));
        const float c03 = __shfl_sync(0xffffffffu, tq4_lut, (int)((p0 >> 12) & 0x0Fu));
        const float c10 = __shfl_sync(0xffffffffu, tq4_lut, (int)(p1 & 0x0Fu));
        const float c11 = __shfl_sync(0xffffffffu, tq4_lut, (int)((p1 >> 4) & 0x0Fu));
        const float c12 = __shfl_sync(0xffffffffu, tq4_lut, (int)((p1 >> 8) & 0x0Fu));
        const float c13 = __shfl_sync(0xffffffffu, tq4_lut, (int)((p1 >> 12) & 0x0Fu));
        const __half2 h00 = __floats2half2_rn(d0 * c00, d0 * c01);
        const __half2 h01 = __floats2half2_rn(d0 * c02, d0 * c03);
        const __half2 h10 = __floats2half2_rn(d1 * c10, d1 * c11);
        const __half2 h11 = __floats2half2_rn(d1 * c12, d1 * c13);
        *reinterpret_cast<uint2*>(dst + (lane << 2)) =
            make_uint2(*reinterpret_cast<const uint32_t*>(&h00),
                       *reinterpret_cast<const uint32_t*>(&h01));
        *reinterpret_cast<uint2*>(dst + 128 + (lane << 2)) =
            make_uint2(*reinterpret_cast<const uint32_t*>(&h10),
                       *reinterpret_cast<const uint32_t*>(&h11));
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
#if __CUDA_ARCH__ >= 890
    if (format == W34_KV_FORMAT_FP8_E4M3) {
        const block_fp8_e4m3_256* blk = reinterpret_cast<const block_fp8_e4m3_256*>(src_ptr);
        const uint8_t* qs8 = blk->qs + (lane << 3);
        uint2 q8;
        if ((((uintptr_t)qs8) & 7u) == 0u) {
            q8 = *reinterpret_cast<const uint2*>(qs8);
        } else {
            q8 = make_uint2(
                static_cast<uint32_t>(qs8[0]) | (static_cast<uint32_t>(qs8[1]) << 8) |
                    (static_cast<uint32_t>(qs8[2]) << 16) | (static_cast<uint32_t>(qs8[3]) << 24),
                static_cast<uint32_t>(qs8[4]) | (static_cast<uint32_t>(qs8[5]) << 8) |
                    (static_cast<uint32_t>(qs8[6]) << 16) | (static_cast<uint32_t>(qs8[7]) << 24));
        }
        const float d = (lane < 16) ? blk->d0 : blk->d1;
        const __half2_raw r0 = __nv_cvt_fp8x2_to_halfraw2(static_cast<__nv_fp8x2_storage_t>(q8.x & 0xFFFFu), __NV_E4M3);
        const __half2_raw r1 = __nv_cvt_fp8x2_to_halfraw2(static_cast<__nv_fp8x2_storage_t>(q8.x >> 16), __NV_E4M3);
        const __half2_raw r2 = __nv_cvt_fp8x2_to_halfraw2(static_cast<__nv_fp8x2_storage_t>(q8.y & 0xFFFFu), __NV_E4M3);
        const __half2_raw r3 = __nv_cvt_fp8x2_to_halfraw2(static_cast<__nv_fp8x2_storage_t>(q8.y >> 16), __NV_E4M3);
        const float2 f0 = __half22float2(*reinterpret_cast<const __half2*>(&r0));
        const float2 f1 = __half22float2(*reinterpret_cast<const __half2*>(&r1));
        const float2 f2 = __half22float2(*reinterpret_cast<const __half2*>(&r2));
        const float2 f3 = __half22float2(*reinterpret_cast<const __half2*>(&r3));
        const __half2 h0 = __floats2half2_rn(f0.x * d, f0.y * d);
        const __half2 h1 = __floats2half2_rn(f1.x * d, f1.y * d);
        const __half2 h2 = __floats2half2_rn(f2.x * d, f2.y * d);
        const __half2 h3 = __floats2half2_rn(f3.x * d, f3.y * d);
        *reinterpret_cast<uint4*>(dst + (lane << 3)) = make_uint4(
            *reinterpret_cast<const uint32_t*>(&h0),
            *reinterpret_cast<const uint32_t*>(&h1),
            *reinterpret_cast<const uint32_t*>(&h2),
            *reinterpret_cast<const uint32_t*>(&h3));
        return;
    }
#endif
    *reinterpret_cast<uint4*>(dst + (lane << 3)) = make_uint4(0u, 0u, 0u, 0u);
}

template <int WARPS>
__device__ __forceinline__ void prefetch_tq4_rows(
    const uint8_t* page_head,
    size_t token_stride,
    int kv_base,
    int union_min_kv,
    int union_max_kv,
    int wid,
    int lane,
    uint32_t (&pf_d)[16 / WARPS],
    uint32_t (&pf_p)[16 / WARPS],
    uint32_t& pf_mask) {
    pf_mask = 0u;
#pragma unroll
    for (int pass = 0; pass < 16 / WARPS; ++pass) {
        const int t = pass * WARPS + wid;
        const int kv_pos = kv_base + t;
        if (kv_pos >= union_min_kv && kv_pos <= union_max_kv) {
            const block_tq4_0* blks =
                reinterpret_cast<const block_tq4_0*>(page_head + (size_t)t * token_stride);
            const uint16_t d0 = *reinterpret_cast<const uint16_t*>(&blks[0].d);
            const uint16_t d1 = *reinterpret_cast<const uint16_t*>(&blks[1].d);
            const uint16_t p0 = *reinterpret_cast<const uint16_t*>(&blks[0].qs[lane << 1]);
            const uint16_t p1 = *reinterpret_cast<const uint16_t*>(&blks[1].qs[lane << 1]);
            pf_d[pass] = (uint32_t)d0 | ((uint32_t)d1 << 16);
            pf_p[pass] = (uint32_t)p0 | ((uint32_t)p1 << 16);
            pf_mask |= (1u << pass);
        } else {
            pf_d[pass] = 0u;
            pf_p[pass] = 0u;
        }
    }
}

template <int WARPS>
__device__ __forceinline__ void unpack_tq4_rows_to_smem_f16(
    const uint32_t (&pf_d)[16 / WARPS],
    const uint32_t (&pf_p)[16 / WARPS],
    uint32_t pf_mask,
    __half (*s_kv)[264],
    int wid,
    int lane,
    float tq4_lut) {
#pragma unroll
    for (int pass = 0; pass < 16 / WARPS; ++pass) {
        const int t = pass * WARPS + wid;
        if ((pf_mask & (1u << pass)) != 0u) {
            const uint16_t d0_u16 = static_cast<uint16_t>(pf_d[pass] & 0xFFFFu);
            const uint16_t d1_u16 = static_cast<uint16_t>(pf_d[pass] >> 16);
            const float d0 = __half2float(*reinterpret_cast<const __half*>(&d0_u16));
            const float d1 = __half2float(*reinterpret_cast<const __half*>(&d1_u16));
            const uint32_t pp = pf_p[pass];
            // Each half-warp materializes one 16-entry scaled FP16 codebook. Moving
            // the identical multiply + round before the lookup saves seven per lane.
            const uint32_t lut = __half_as_ushort(__float2half_rn((lane < 16 ? d0 : d1) * tq4_lut));
            const uint32_t c00 = __shfl_sync(0xffffffffu, lut, pp & 15u);
            const uint32_t c01 = __shfl_sync(0xffffffffu, lut, (pp >> 4) & 15u);
            const uint32_t c02 = __shfl_sync(0xffffffffu, lut, (pp >> 8) & 15u);
            const uint32_t c03 = __shfl_sync(0xffffffffu, lut, (pp >> 12) & 15u);
            const uint32_t c10 = __shfl_sync(0xffffffffu, lut, 16 + ((pp >> 16) & 15u));
            const uint32_t c11 = __shfl_sync(0xffffffffu, lut, 16 + ((pp >> 20) & 15u));
            const uint32_t c12 = __shfl_sync(0xffffffffu, lut, 16 + ((pp >> 24) & 15u));
            const uint32_t c13 = __shfl_sync(0xffffffffu, lut, 16 + ((pp >> 28) & 15u));
            *reinterpret_cast<uint2*>(s_kv[t] + (lane << 2)) =
                make_uint2(c00 | (c01 << 16), c02 | (c03 << 16));
            *reinterpret_cast<uint2*>(s_kv[t] + 128 + (lane << 2)) =
                make_uint2(c10 | (c11 << 16), c12 | (c13 << 16));
        } else {
            *reinterpret_cast<uint4*>(&s_kv[t][lane << 3]) = make_uint4(0u, 0u, 0u, 0u);
        }
    }
}

// GQA-grouped + multi-token m16n8k16 mma.sync paged attention (head_dim=256,
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
    // Keep Split-KV partitions down to 4 pages (64 keys) across both short and long
    // max_seq_len CUDA-graph buckets so 1K-4K decode spreads across 52-64 active CTAs
    // instead of collapsing to 4-8 CTAs when max_seq_len > 1024.
    constexpr int MIN_PAGES_PER_SPLIT = 4;

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
    const float tq4_lut = kTq4D256Centroids[lane & 15];

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
                    lane,
                    tq4_lut);
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
        __syncwarp();
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
                    lane,
                    tq4_lut);
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

// 64/128-query-row paged prefill (4/8 warps x 16 independent query rows per warp).
// A shared 16-key page preserves the reference two-half Q@K and online-softmax order.
// The 128-row specialization halves KV reads/dequantization relative to 64 rows;
// dynamic Q storage allows its 76 KiB total smem footprint on sm_86.
template <int QROWS>
__global__ void __launch_bounds__(QROWS * 2, 1) paged_attention_prefill_mma_f16_kernel(
    w34_kv_layer_layout layout,
    w34_attention_batch batch,
    const __half* __restrict__ q_in,
    __half* __restrict__ attn_out,
    int num_q_heads,
    int gqa_ratio,
    int q_group,
    int q_subgroups,
    int tok_tile) {
    constexpr int HEAD_DIM = 256;
    constexpr int SMEM_STRIDE = 264; // 264 * 2 = 528 B (16B-aligned + bank-conflict-free)

    constexpr int WARPS = QROWS / 16;
    static_assert(QROWS == 64 || QROWS == 128);
    extern __shared__ __align__(16) __half s_q[][SMEM_STRIDE];
    __shared__ __align__(16) __half s_kv[16][SMEM_STRIDE];
    __shared__ int s_row_tok[QROWS];
    __shared__ int s_row_qhead[QROWS];
    __shared__ int s_min_kv[QROWS];
    __shared__ int s_max_kv[QROWS];
    __shared__ uint8_t s_row_valid[QROWS];
    __shared__ int s_req;
    __shared__ int s_req_q_start;
    __shared__ int s_req_q_end;
    __shared__ int s_tok_base;
    __shared__ int s_union_min_kv;
    __shared__ int s_union_max_kv;
    __shared__ uint8_t s_any_valid;

    const int tid = threadIdx.x;
    const int lane = tid & 31;
    const int wid = tid >> 5;
    const int group_id = lane >> 2;
    const int tid_in_grp = lane & 3;
    const int q_base = wid * 16;

    const int kv_sub = (int)blockIdx.y;
    const int kv_head = kv_sub / q_subgroups;
    const int q_head_base = kv_head * gqa_ratio + (kv_sub % q_subgroups) * q_group;
    const int active_rows = tok_tile * q_group;

    // Resolve (req, tok_base) per-request tile so multi-request mixed prefill+decode and
    // concurrent coding-agent batches (num_reqs >= 1) never mix two requests in one CTA.
    if (tid == 0) {
        int rem_tile = (int)blockIdx.x;
        int found_req = -1;
        int q_lo = 0;
        int q_hi = 0;
        int t_base = 0;
        for (int r = 0; r < batch.num_reqs; ++r) {
            if (batch.valid_req_mask != nullptr && batch.valid_req_mask[r] == 0) continue;
            const int lo = batch.query_start_loc[r];
            const int hi = batch.query_start_loc[r + 1];
            const int qlen = hi - lo;
            if (qlen <= 0) continue;
            const int n_tiles = (qlen + tok_tile - 1) / tok_tile;
            if (rem_tile < n_tiles) {
                found_req = r;
                q_lo = lo;
                q_hi = hi;
                t_base = lo + rem_tile * tok_tile;
                break;
            }
            rem_tile -= n_tiles;
        }
        s_req = found_req;
        s_req_q_start = q_lo;
        s_req_q_end = q_hi;
        s_tok_base = t_base;
    }
    __syncthreads();

    if (s_req < 0) return;

    const int req = s_req;
    const int q_start = s_req_q_start;
    const int q_end = s_req_q_end;
    const int tok_base = s_tok_base;

    if (tid < QROWS) {
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
            if (tok < q_end && tok < batch.total_tokens &&
                (batch.valid_token_mask == nullptr || batch.valid_token_mask[tok] != 0)) {
                valid = 1;
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
        s_row_valid[tid] = valid;
        s_row_tok[tid] = tok;
        s_row_qhead[tid] = qh;
        s_min_kv[tid] = min_kv;
        s_max_kv[tid] = max_kv;
    }
    __syncthreads();

    if (tid == 0) {
        int u_min = INT_MAX;
        int u_max = -1;
        uint8_t any = 0;
        for (int r = 0; r < QROWS; ++r) {
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
    const int my_page_start = first_page;
    const int my_page_end = (last_page >= first_page) ? (last_page + 1) : first_page;

    // Stage and pre-rotate Q rows, one warp per row.
    for (int r = wid; r < QROWS; r += WARPS) {
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

    int warp_min_kv = INT_MAX;
    int warp_max_kv = -1;
    for (int r = q_base; r < q_base + 16; ++r) {
        if (s_row_valid[r] && s_min_kv[r] <= s_max_kv[r]) {
            if (s_min_kv[r] < warp_min_kv) warp_min_kv = s_min_kv[r];
            if (s_max_kv[r] > warp_max_kv) warp_max_kv = s_max_kv[r];
        }
    }

    const int r0 = q_base + group_id;
    const int r1 = q_base + group_id + 8;
    const bool row0_ok = (s_row_valid[r0] != 0);
    const bool row1_ok = (s_row_valid[r1] != 0);
    const int min_kv0 = s_min_kv[r0];
    const int max_kv0 = s_max_kv[r0];
    const int min_kv1 = s_min_kv[r1];
    const int max_kv1 = s_max_kv[r1];

    float m_row0 = -FLT_MAX, l_row0 = 0.0f;
    float m_row1 = -FLT_MAX, l_row1 = 0.0f;

    float o_frag[32][4];
#pragma unroll
    for (int step = 0; step < 32; ++step) {
        o_frag[step][0] = 0.0f;
        o_frag[step][1] = 0.0f;
        o_frag[step][2] = 0.0f;
        o_frag[step][3] = 0.0f;
    }
    __syncthreads();

    const int32_t* req_blocks = batch.block_tables + (size_t)s_req * batch.max_blocks_per_req;
    const float tq4_lut = kTq4D256Centroids[lane & 15];
    const bool k_is_tq4 = (layout.k_format == W34_KV_FORMAT_TQ4);
    const bool v_is_tq4 = (layout.v_format == W34_KV_FORMAT_TQ4);

    uint32_t pf_d[16 / WARPS] = {};
    uint32_t pf_p[16 / WARPS] = {};
    uint32_t pf_mask = 0u;
    if (k_is_tq4 && my_page_start < my_page_end) {
        const int32_t phys0 = req_blocks[my_page_start];
        const uint8_t* k0_head = (const uint8_t*)(uintptr_t)layout.k_base
                                 + (size_t)phys0 * layout.k_page_stride_bytes
                                 + (size_t)kv_head * layout.k_head_stride_bytes;
        prefetch_tq4_rows<WARPS>(
            k0_head,
            layout.k_token_stride_bytes,
            my_page_start << 4,
            union_min_kv,
            union_max_kv,
            wid,
            lane,
            pf_d,
            pf_p,
            pf_mask);
    }

    for (int page_idx = my_page_start; page_idx < my_page_end; ++page_idx) {
        const int kv_base = page_idx << 4;
        const int32_t phys_page = req_blocks[page_idx];
        const uint8_t* k_page_head = (const uint8_t*)(uintptr_t)layout.k_base
                                     + (size_t)phys_page * layout.k_page_stride_bytes
                                     + (size_t)kv_head * layout.k_head_stride_bytes;
        const uint8_t* v_page_head = (const uint8_t*)(uintptr_t)layout.v_base
                                     + (size_t)phys_page * layout.v_page_stride_bytes
                                     + (size_t)kv_head * layout.v_head_stride_bytes;

        // 1. Cooperative decode of 16 K rows into s_kv (shared across all Q rows),
        // then immediately issue V[page_idx] HBM prefetch so V arrives during Q@K^T + softmax.
        if (k_is_tq4) {
            unpack_tq4_rows_to_smem_f16<WARPS>(pf_d, pf_p, pf_mask, s_kv, wid, lane, tq4_lut);
        } else {
#pragma unroll
            for (int pass = 0; pass < 16 / WARPS; ++pass) {
                const int t = pass * WARPS + wid;
                const int kv_pos = kv_base + t;
                if (kv_pos >= union_min_kv && kv_pos <= union_max_kv) {
                    decode_kv_row_256_to_smem_f16(
                        k_page_head + (size_t)t * layout.k_token_stride_bytes,
                        layout.k_format,
                        s_kv[t],
                        lane,
                        tq4_lut);
                } else {
                    *reinterpret_cast<uint4*>(&s_kv[t][lane << 3]) = make_uint4(0u, 0u, 0u, 0u);
                }
            }
        }
        if (v_is_tq4) {
            prefetch_tq4_rows<WARPS>(
                v_page_head,
                layout.v_token_stride_bytes,
                kv_base,
                union_min_kv,
                union_max_kv,
                wid,
                lane,
                pf_d,
                pf_p,
                pf_mask);
        }
        __syncthreads();

        const bool warp_active = (kv_base <= warp_max_kv) && (kv_base + 15 >= warp_min_kv);
        float alpha0 = 1.0f, alpha1 = 1.0f;
        uint32_t pa0 = 0u, pa1 = 0u, pa2 = 0u, pa3 = 0u;
        bool warp_has_tile = false;

        if (warp_active) {
            // 2. Per-warp Q_wid @ K^T ([16, 256] x [16, 256]^T -> [16, 16] FP32).
            // Accumulate cols 0..127 and cols 128..255 separately to preserve exact 0-ULP
            // floating-point addition order with the 16-row k_half={0,1} kernel.
            float qk0_h0[4] = {0.0f, 0.0f, 0.0f, 0.0f};
            float qk1_h0[4] = {0.0f, 0.0f, 0.0f, 0.0f};
#pragma unroll
            for (int ks = 0; ks < 8; ++ks) {
                const int col_base = ks * 16;
                uint32_t a0, a1, a2, a3, b00, b01, b10, b11;
                const uint32_t q_smem = static_cast<uint32_t>(__cvta_generic_to_shared(
                    &s_q[q_base + (lane & 15)][col_base + ((lane >> 4) << 3)]));
                const uint32_t k_smem = static_cast<uint32_t>(__cvta_generic_to_shared(
                    &s_kv[((lane >> 4) << 3) + (lane & 7)][col_base + (((lane >> 3) & 1) << 3)]));
                asm volatile(
                    "ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                    : "=r"(a0), "=r"(a1), "=r"(a2), "=r"(a3)
                    : "r"(q_smem));
                asm volatile(
                    "ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                    : "=r"(b00), "=r"(b01), "=r"(b10), "=r"(b11)
                    : "r"(k_smem));
                asm volatile(
                    "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
                    "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                    : "+f"(qk0_h0[0]), "+f"(qk0_h0[1]), "+f"(qk0_h0[2]), "+f"(qk0_h0[3])
                    : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b00), "r"(b01));
                asm volatile(
                    "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
                    "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                    : "+f"(qk1_h0[0]), "+f"(qk1_h0[1]), "+f"(qk1_h0[2]), "+f"(qk1_h0[3])
                    : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b10), "r"(b11));
            }

            float qk0_h1[4] = {0.0f, 0.0f, 0.0f, 0.0f};
            float qk1_h1[4] = {0.0f, 0.0f, 0.0f, 0.0f};
#pragma unroll
            for (int ks = 0; ks < 8; ++ks) {
                const int col_base = 128 + ks * 16;
                uint32_t a0, a1, a2, a3, b00, b01, b10, b11;
                const uint32_t q_smem = static_cast<uint32_t>(__cvta_generic_to_shared(
                    &s_q[q_base + (lane & 15)][col_base + ((lane >> 4) << 3)]));
                const uint32_t k_smem = static_cast<uint32_t>(__cvta_generic_to_shared(
                    &s_kv[((lane >> 4) << 3) + (lane & 7)][col_base + (((lane >> 3) & 1) << 3)]));
                asm volatile(
                    "ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                    : "=r"(a0), "=r"(a1), "=r"(a2), "=r"(a3)
                    : "r"(q_smem));
                asm volatile(
                    "ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                    : "=r"(b00), "=r"(b01), "=r"(b10), "=r"(b11)
                    : "r"(k_smem));
                asm volatile(
                    "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
                    "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                    : "+f"(qk0_h1[0]), "+f"(qk0_h1[1]), "+f"(qk0_h1[2]), "+f"(qk0_h1[3])
                    : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b00), "r"(b01));
                asm volatile(
                    "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
                    "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                    : "+f"(qk1_h1[0]), "+f"(qk1_h1[1]), "+f"(qk1_h1[2]), "+f"(qk1_h1[3])
                    : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b10), "r"(b11));
            }

            // 3. In-register online FP32 softmax across the 4-lane group_id quad.
            const int kv_lo0 = kv_base + 2 * tid_in_grp;
            const int kv_lo1 = kv_lo0 + 1;
            const int kv_hi0 = kv_lo0 + 8;
            const int kv_hi1 = kv_lo0 + 9;

            float s0_lo0 = -FLT_MAX, s0_lo1 = -FLT_MAX, s0_hi0 = -FLT_MAX, s0_hi1 = -FLT_MAX;
            if (row0_ok) {
                if (kv_lo0 >= min_kv0 && kv_lo0 <= max_kv0) s0_lo0 = (qk0_h0[0] + qk0_h1[0]) * batch.sm_scale;
                if (kv_lo1 >= min_kv0 && kv_lo1 <= max_kv0) s0_lo1 = (qk0_h0[1] + qk0_h1[1]) * batch.sm_scale;
                if (kv_hi0 >= min_kv0 && kv_hi0 <= max_kv0) s0_hi0 = (qk1_h0[0] + qk1_h1[0]) * batch.sm_scale;
                if (kv_hi1 >= min_kv0 && kv_hi1 <= max_kv0) s0_hi1 = (qk1_h0[1] + qk1_h1[1]) * batch.sm_scale;
            }
            float tmax0 = fmaxf(fmaxf(s0_lo0, s0_lo1), fmaxf(s0_hi0, s0_hi1));
            tmax0 = fmaxf(tmax0, __shfl_xor_sync(0xffffffffu, tmax0, 2));
            tmax0 = fmaxf(tmax0, __shfl_xor_sync(0xffffffffu, tmax0, 1));
            const bool has0 = row0_ok && (tmax0 > -FLT_MAX);
            const float m_new0 = has0 ? fmaxf(m_row0, tmax0) : 0.0f;
            alpha0 = has0 ? ((m_row0 > -FLT_MAX) ? expf(m_row0 - m_new0) : 0.0f) : 1.0f;
            const float p0_lo0 = (has0 && s0_lo0 > -FLT_MAX) ? expf(s0_lo0 - m_new0) : 0.0f;
            const float p0_lo1 = (has0 && s0_lo1 > -FLT_MAX) ? expf(s0_lo1 - m_new0) : 0.0f;
            const float p0_hi0 = (has0 && s0_hi0 > -FLT_MAX) ? expf(s0_hi0 - m_new0) : 0.0f;
            const float p0_hi1 = (has0 && s0_hi1 > -FLT_MAX) ? expf(s0_hi1 - m_new0) : 0.0f;
            float psum0 = (p0_lo0 + p0_lo1) + (p0_hi0 + p0_hi1);
            psum0 += __shfl_xor_sync(0xffffffffu, psum0, 2);
            psum0 += __shfl_xor_sync(0xffffffffu, psum0, 1);
            if (has0) {
                m_row0 = m_new0;
                l_row0 = l_row0 * alpha0 + psum0;
            }
            const __half2 h2_0_lo = __floats2half2_rn(p0_lo0, p0_lo1);
            const __half2 h2_0_hi = __floats2half2_rn(p0_hi0, p0_hi1);
            pa0 = *reinterpret_cast<const uint32_t*>(&h2_0_lo);
            pa2 = *reinterpret_cast<const uint32_t*>(&h2_0_hi);

            float s1_lo0 = -FLT_MAX, s1_lo1 = -FLT_MAX, s1_hi0 = -FLT_MAX, s1_hi1 = -FLT_MAX;
            if (row1_ok) {
                if (kv_lo0 >= min_kv1 && kv_lo0 <= max_kv1) s1_lo0 = (qk0_h0[2] + qk0_h1[2]) * batch.sm_scale;
                if (kv_lo1 >= min_kv1 && kv_lo1 <= max_kv1) s1_lo1 = (qk0_h0[3] + qk0_h1[3]) * batch.sm_scale;
                if (kv_hi0 >= min_kv1 && kv_hi0 <= max_kv1) s1_hi0 = (qk1_h0[2] + qk1_h1[2]) * batch.sm_scale;
                if (kv_hi1 >= min_kv1 && kv_hi1 <= max_kv1) s1_hi1 = (qk1_h0[3] + qk1_h1[3]) * batch.sm_scale;
            }
            float tmax1 = fmaxf(fmaxf(s1_lo0, s1_lo1), fmaxf(s1_hi0, s1_hi1));
            tmax1 = fmaxf(tmax1, __shfl_xor_sync(0xffffffffu, tmax1, 2));
            tmax1 = fmaxf(tmax1, __shfl_xor_sync(0xffffffffu, tmax1, 1));
            const bool has1 = row1_ok && (tmax1 > -FLT_MAX);
            const float m_new1 = has1 ? fmaxf(m_row1, tmax1) : 0.0f;
            alpha1 = has1 ? ((m_row1 > -FLT_MAX) ? expf(m_row1 - m_new1) : 0.0f) : 1.0f;
            const float p1_lo0 = (has1 && s1_lo0 > -FLT_MAX) ? expf(s1_lo0 - m_new1) : 0.0f;
            const float p1_lo1 = (has1 && s1_lo1 > -FLT_MAX) ? expf(s1_lo1 - m_new1) : 0.0f;
            const float p1_hi0 = (has1 && s1_hi0 > -FLT_MAX) ? expf(s1_hi0 - m_new1) : 0.0f;
            const float p1_hi1 = (has1 && s1_hi1 > -FLT_MAX) ? expf(s1_hi1 - m_new1) : 0.0f;
            float psum1 = (p1_lo0 + p1_lo1) + (p1_hi0 + p1_hi1);
            psum1 += __shfl_xor_sync(0xffffffffu, psum1, 2);
            psum1 += __shfl_xor_sync(0xffffffffu, psum1, 1);
            if (has1) {
                m_row1 = m_new1;
                l_row1 = l_row1 * alpha1 + psum1;
            }
            const __half2 h2_1_lo = __floats2half2_rn(p1_lo0, p1_lo1);
            const __half2 h2_1_hi = __floats2half2_rn(p1_hi0, p1_hi1);
            pa1 = *reinterpret_cast<const uint32_t*>(&h2_1_lo);
            pa3 = *reinterpret_cast<const uint32_t*>(&h2_1_hi);

            warp_has_tile = (__any_sync(0xffffffffu, has0 || has1) != 0);
        }
        __syncthreads();

        // 4. Cooperative decode of 16 V rows into s_kv, then immediately prefetch
        // K[page_idx + 1] from HBM so next page's K arrives during P@V mma.sync.
        if (v_is_tq4) {
            unpack_tq4_rows_to_smem_f16<WARPS>(pf_d, pf_p, pf_mask, s_kv, wid, lane, tq4_lut);
        } else {
#pragma unroll
            for (int pass = 0; pass < 16 / WARPS; ++pass) {
                const int t = pass * WARPS + wid;
                const int kv_pos = kv_base + t;
                if (kv_pos >= union_min_kv && kv_pos <= union_max_kv) {
                    decode_kv_row_256_to_smem_f16(
                        v_page_head + (size_t)t * layout.v_token_stride_bytes,
                        layout.v_format,
                        s_kv[t],
                        lane,
                        tq4_lut);
                } else {
                    *reinterpret_cast<uint4*>(&s_kv[t][lane << 3]) = make_uint4(0u, 0u, 0u, 0u);
                }
            }
        }
        if (k_is_tq4 && (page_idx + 1) < my_page_end) {
            const int32_t next_phys = req_blocks[page_idx + 1];
            const uint8_t* next_k_head = (const uint8_t*)(uintptr_t)layout.k_base
                                         + (size_t)next_phys * layout.k_page_stride_bytes
                                         + (size_t)kv_head * layout.k_head_stride_bytes;
            prefetch_tq4_rows<WARPS>(
                next_k_head,
                layout.k_token_stride_bytes,
                (page_idx + 1) << 4,
                union_min_kv,
                union_max_kv,
                wid,
                lane,
                pf_d,
                pf_p,
                pf_mask);
        }
        __syncthreads();

        // 5. Rescale output accumulators and accumulate P_wid @ V ([16, 16] x [16, 256])
        // via 16x ldmatrix.m8n8.x4.trans + 32x mma.sync per warp.
        if (warp_has_tile) {
#pragma unroll
            for (int pair = 0; pair < 16; ++pair) {
                const int s0 = pair * 2;
                const int s1 = s0 + 1;
                o_frag[s0][0] *= alpha0;
                o_frag[s0][1] *= alpha0;
                o_frag[s0][2] *= alpha1;
                o_frag[s0][3] *= alpha1;
                o_frag[s1][0] *= alpha0;
                o_frag[s1][1] *= alpha0;
                o_frag[s1][2] *= alpha1;
                o_frag[s1][3] *= alpha1;
                uint32_t v00, v01, v10, v11;
                const uint32_t v_smem = static_cast<uint32_t>(__cvta_generic_to_shared(
                    &s_kv[lane & 15][pair * 16 + ((lane >> 4) << 3)]));
                asm volatile(
                    "ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                    : "=r"(v00), "=r"(v01), "=r"(v10), "=r"(v11)
                    : "r"(v_smem));
                asm volatile(
                    "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
                    "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                    : "+f"(o_frag[s0][0]), "+f"(o_frag[s0][1]), "+f"(o_frag[s0][2]), "+f"(o_frag[s0][3])
                    : "r"(pa0), "r"(pa1), "r"(pa2), "r"(pa3), "r"(v00), "r"(v01));
                asm volatile(
                    "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
                    "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                    : "+f"(o_frag[s1][0]), "+f"(o_frag[s1][1]), "+f"(o_frag[s1][2]), "+f"(o_frag[s1][3])
                    : "r"(pa0), "r"(pa1), "r"(pa2), "r"(pa3), "r"(v10), "r"(v11));
            }
        }
        __syncthreads();
    }

    // Epilogue: reuse s_q as disjoint per-warp FP32 staging, in two passes of
    // eight rows per warp. QROWS * 264 * 2 bytes covers QROWS / 2 * 256 * 4.
    float (*s_out_stage)[HEAD_DIM] = reinterpret_cast<float (*)[HEAD_DIM]>(s_q);
#pragma unroll
    for (int half_idx = 0; half_idx < 2; ++half_idx) {
#pragma unroll
        for (int step = 0; step < 32; ++step) {
            const int c0 = step * 8 + 2 * tid_in_grp;
            s_out_stage[wid * 8 + group_id][c0] = o_frag[step][2 * half_idx];
            s_out_stage[wid * 8 + group_id][c0 + 1] = o_frag[step][2 * half_idx + 1];
        }
        __syncwarp();

        for (int local_r = 0; local_r < 8; ++local_r) {
            const int r = q_base + half_idx * 8 + local_r;
            const float row_l = __shfl_sync(
                0xffffffffu, (half_idx == 0) ? l_row0 : l_row1, local_r << 2);
            if (s_row_valid[r]) {
                const size_t row_linear = (size_t)s_row_tok[r] * num_q_heads + s_row_qhead[r];
                __half* out_row = attn_out + row_linear * HEAD_DIM;
                const float inv_l = (row_l > 0.0f) ? (1.0f / row_l) : 0.0f;
                float out_v[8];
#pragma unroll
                for (int j = 0; j < 8; ++j) {
                    out_v[j] = s_out_stage[wid * 8 + local_r][lane + 32 * j] * inv_l;
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
        __syncwarp();
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

template <int QROWS>
int launch_prefill(const w34_exec* ex, const w34_kv_layer_layout* layout,
    const w34_attention_batch* batch, const void* q_in, void* attn_out, int num_q_heads) {
    static std::atomic<uint32_t> configured_mask{0u};
    int dev = 0;
    const int device_rc = (int)cudaGetDevice(&dev);
    if (device_rc != 0) return device_rc;
    if (dev < 0 || dev >= 32) return (int)cudaErrorInvalidDevice;
    const uint32_t bit = 1u << dev;
    if ((configured_mask.load(std::memory_order_relaxed) & bit) == 0u) {
        int rc = (int)cudaFuncSetAttribute(
            paged_attention_prefill_mma_f16_kernel<QROWS>,
            cudaFuncAttributeMaxDynamicSharedMemorySize, QROWS * 264 * sizeof(__half));
        if (rc != 0) return rc;
        rc = (int)cudaFuncSetAttribute(
            paged_attention_prefill_mma_f16_kernel<QROWS>,
            cudaFuncAttributePreferredSharedMemoryCarveout, 100);
        if (rc != 0) return rc;
        configured_mask.fetch_or(bit, std::memory_order_relaxed);
    }
    const int gqa_ratio = num_q_heads / layout->local_kv_heads;
    int q_group = 1;
    for (int d = 16; d >= 1; --d) {
        if (gqa_ratio % d == 0 && QROWS % d == 0) {
            q_group = d;
            break;
        }
    }
    const int q_subgroups = gqa_ratio / q_group;
    const int tok_tile = QROWS / q_group;
    const unsigned grid_x = batch->num_reqs == 1
        ? (unsigned)((batch->total_tokens + tok_tile - 1) / tok_tile)
        : (unsigned)(batch->total_tokens / tok_tile + batch->num_reqs);
    paged_attention_prefill_mma_f16_kernel<QROWS><<<
        dim3(grid_x, layout->local_kv_heads * q_subgroups), QROWS * 2,
        QROWS * 264 * sizeof(__half), (cudaStream_t)ex->stream>>>(
        *layout, *batch, static_cast<const __half*>(q_in), static_cast<__half*>(attn_out),
        num_q_heads, gqa_ratio, q_group, q_subgroups, tok_tile);
    return (int)cudaGetLastError();
}

}  // namespace w34_attn

extern "C" int w34_paged_fa2_try(const w34_exec*, const w34_kv_layer_layout*,
    const w34_attention_batch*, const void*, void*, int);

extern "C" int w34_paged_attention_f16(const w34_exec* ex, const w34_kv_layer_layout* layout,
                                       const w34_attention_batch* batch,
                                       const void* q_in, void* attn_out,
                                       int num_q_heads) {
    if (ex == nullptr || layout == nullptr || batch == nullptr || q_in == nullptr || attn_out == nullptr) {
        return (int)cudaErrorInvalidValue;
    }
    if (layout->k_format == W34_KV_FORMAT_FP8_E4M3 || layout->v_format == W34_KV_FORMAT_FP8_E4M3) {
        if (layout->head_dim != 256) return (int)cudaErrorInvalidValue;
        if (!w34_attn::current_device_supports_fp8_kv()) return (int)cudaErrorNotSupported;
    }
    if (batch->total_tokens <= 0 || num_q_heads <= 0) return 0;

    const int fa2_rc = w34_paged_fa2_try(ex, layout, batch, q_in, attn_out, num_q_heads);
    if (fa2_rc != -1) return fa2_rc; // CUDA failure is not fallback eligibility.

    const bool can_use_gqa_mma =
        (layout->head_dim == 256) &&
        (layout->block_size == 16) &&
        (layout->local_kv_heads > 0) &&
        (num_q_heads % layout->local_kv_heads == 0) &&
        (layout->k_format == W34_KV_FORMAT_TQ4 ||
         layout->k_format == W34_KV_FORMAT_Q8_0 ||
         layout->k_format == W34_KV_FORMAT_FP8_E4M3 ||
         layout->k_format == W34_KV_FORMAT_FP16) &&
        (layout->v_format == W34_KV_FORMAT_TQ4 ||
         layout->v_format == W34_KV_FORMAT_Q8_0 ||
         layout->v_format == W34_KV_FORMAT_FP8_E4M3 ||
         layout->v_format == W34_KV_FORMAT_FP16);

    if (can_use_gqa_mma) {
        const int gqa_ratio = num_q_heads / layout->local_kv_heads;

        // Parallel Split-KV for short verification/batched decode (128..1024 keys),
        // or long contexts (>1024 keys), only when aligned scratch fits.
        // Note: `cand_splits` depends ONLY on TP-invariant batch geometry
        // (num_reqs, total_tokens, max_q_len, max_seq_len, max_blocks_per_req) —
        // never on local_kv_heads — and caps `needed <= 4.03 MiB` even at `num_q_heads = 32`,
        // preserving 0-ULP TP=1 == TP=2 parity under the engine's `>= 8 MiB` scratch invariant!
        int num_splits = 1;
        float* partial_acc = nullptr;
        float* partial_m = nullptr;
        float* partial_l = nullptr;
        const bool short_split =
            ((batch->max_q_len > 1 && batch->total_tokens > 1) ||
             (batch->max_q_len == 1 && batch->num_reqs >= 1 && batch->total_tokens <= 8)) &&
            batch->total_tokens <= 16 && batch->max_seq_len >= 128 && batch->max_seq_len <= 1024;
        if ((short_split || (batch->total_tokens <= 32 && batch->max_seq_len > 1024 &&
            batch->max_blocks_per_req > 64)) &&
            ex->scratch_ptr != 0 && (ex->scratch_ptr & 15u) == 0u) {
            const int max_short_splits = (batch->total_tokens <= 4) ? 8 : 4;
            int cand_splits =
                short_split
                    ? (((batch->max_seq_len + 63) / 64 < max_short_splits)
                           ? ((batch->max_seq_len + 63) / 64)
                           : max_short_splits)
                    : ((batch->max_blocks_per_req > 2048 && batch->total_tokens <= 2) ? 64 :
                       (batch->max_blocks_per_req > 1024 && batch->total_tokens <= 4) ? 32 :
                       (batch->total_tokens <= 8 ? 16 : (batch->total_tokens <= 16 ? 8 : 4)));
            if (short_split && batch->max_q_len == 1 && batch->total_tokens >= 6 &&
                cand_splits > 2) {
                cand_splits = 2;
            }
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

        if (batch->total_tokens == 1 && num_splits == 1 && gqa_ratio < 2) {
            return w34_paged_attention_scalar_f16(ex, layout, batch, q_in, attn_out, num_q_heads);
        }

        // Fast 64-query-row FlashAttention-2 prefill path for TQ4/Q8_0/FP16 chunked prefill
        // (including multi-request mixed prefill+decode and concurrent agent prefills):
        // 4 warps each own 16 independent query rows sharing the cooperatively decoded KV page,
        // cutting HBM KV bandwidth and TQ4/Q8_0 page dequantization by 4x-16x with 0-ULP parity.
        if (batch->num_reqs >= 1 && batch->total_tokens >= 64 && batch->max_q_len >= 16 &&
            num_splits == 1) {
            int q_group_64 = 1;
            for (int d = 16; d >= 1; --d) {
                if ((gqa_ratio % d) == 0 && (64 % d) == 0) {
                    q_group_64 = d;
                    break;
                }
            }
            const int q_subgroups_64 = gqa_ratio / q_group_64;
            const int tok_tile_64 = 64 / q_group_64;
            const unsigned grid_x_64 = (batch->num_reqs == 1)
                ? (unsigned)((batch->total_tokens + tok_tile_64 - 1) / tok_tile_64)
                : (unsigned)((batch->total_tokens / tok_tile_64) + batch->num_reqs);
            const unsigned grid_y_64 = (unsigned)(layout->local_kv_heads * q_subgroups_64);
            if ((uint64_t)grid_x_64 * (uint64_t)grid_y_64 >= 42u || batch->max_seq_len >= 4096) {
                // Larger tiles amortize Q8 dequantization, but measurements show
                // TQ4/FP16 favor two resident 64-row CTAs over one 128-row CTA.
                if (batch->total_tokens >= 512 && batch->max_seq_len >= 4096 &&
                    layout->k_format == W34_KV_FORMAT_Q8_0 &&
                    layout->v_format == W34_KV_FORMAT_Q8_0) {
                    return w34_attn::launch_prefill<128>(ex, layout, batch, q_in, attn_out, num_q_heads);
                }
                return w34_attn::launch_prefill<64>(ex, layout, batch, q_in, attn_out, num_q_heads);
            }
        }

        int q_group = gqa_ratio;
        if (batch->num_reqs == 1 && batch->total_tokens > 32 && num_splits == 1 &&
            (16 % gqa_ratio) != 0) {
            // For chunked prefill on non-power-of-2 GQA ratios (e.g. 27B gqa_ratio = 6),
            // picking a divisor of both `gqa_ratio` and 16 (q_group = 2, tok_tile = 8)
            // fills all 16/16 mma.sync.m16n8k16 rows instead of 12/16, cutting redundant
            // TQ4/WHT-256 KV page decodes by 25% with 0-ULP invariance.
            for (int d = 8; d >= 1; --d) {
                if ((gqa_ratio % d) == 0 && (16 % d) == 0) {
                    q_group = d;
                    break;
                }
            }
        } else if (q_group > 8) {
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
            // Short verification without Split-KV otherwise launches only a handful of
            // CTAs (e.g. 16 for M=16, GQA=4), so keep each query in its own CTA when
            // num_splits == 1. When short_split is active (num_splits == 2..4), packing
            // tok_tile = 16 / q_group queries per CTA fills all 16 m16n8k16 rows, avoids
            // 4x redundant TQ4/WHT-256 KV decoding across sibling query CTAs, and still
            // launches 64+ CTAs across the 84 SMs.
            if (batch->total_tokens <= 16 && num_splits == 1) tok_tile = 1;
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

    return w34_paged_attention_scalar_f16(ex, layout, batch, q_in, attn_out, num_q_heads);
}

// Test-only selector: 0 = 16 rows, 1 = 64 rows, 2 = 128 rows (same arithmetic).
extern "C" int w34_paged_attention_f16_force_kernel_for_test(
    const w34_exec* ex,
    const w34_kv_layer_layout* layout,
    const w34_attention_batch* batch,
    const void* q_in,
    void* attn_out,
    int num_q_heads,
    int use_64q) {
    if (ex == nullptr || layout == nullptr || batch == nullptr || q_in == nullptr || attn_out == nullptr) {
        return (int)cudaErrorInvalidValue;
    }
    if (layout->k_format == W34_KV_FORMAT_FP8_E4M3 || layout->v_format == W34_KV_FORMAT_FP8_E4M3) {
        if (layout->head_dim != 256) return (int)cudaErrorInvalidValue;
        if (!w34_attn::current_device_supports_fp8_kv()) return (int)cudaErrorNotSupported;
    }
    if (batch->total_tokens <= 0 || num_q_heads <= 0) return 0;
    if (layout->head_dim != 256 || layout->block_size != 16 || layout->local_kv_heads <= 0 ||
        (num_q_heads % layout->local_kv_heads) != 0 || batch->num_reqs < 1) {
        return (int)cudaErrorInvalidValue;
    }
    const int gqa_ratio = num_q_heads / layout->local_kv_heads;
    if (use_64q == 2) {
        return w34_attn::launch_prefill<128>(ex, layout, batch, q_in, attn_out, num_q_heads);
    }
    if (use_64q != 0) {
        return w34_attn::launch_prefill<64>(ex, layout, batch, q_in, attn_out, num_q_heads);
    }

    int q_group = gqa_ratio;
    if (batch->num_reqs == 1 && (16 % gqa_ratio) != 0) {
        for (int d = 8; d >= 1; --d) {
            if ((gqa_ratio % d) == 0 && (16 % d) == 0) {
                q_group = d;
                break;
            }
        }
    } else if (q_group > 8) {
        for (int d = 8; d >= 1; --d) {
            if (gqa_ratio % d == 0) {
                q_group = d;
                break;
            }
        }
    }
    const int q_subgroups = gqa_ratio / q_group;
    int tok_tile = (batch->num_reqs == 1) ? (16 / q_group) : 1;
    if (batch->total_tokens < tok_tile) tok_tile = batch->total_tokens;
    if (tok_tile < 1) tok_tile = 1;
    const dim3 grid_16(
        (unsigned)((batch->total_tokens + tok_tile - 1) / tok_tile),
        (unsigned)(layout->local_kv_heads * q_subgroups),
        1u);
    w34_attn::paged_attention_gqa_mma_f16_kernel<<<
        grid_16, 128, 0, (cudaStream_t)ex->stream>>>(
        *layout,
        *batch,
        static_cast<const __half*>(q_in),
        static_cast<__half*>(attn_out),
        nullptr,
        nullptr,
        nullptr,
        num_q_heads,
        gqa_ratio,
        q_group,
        q_subgroups,
        tok_tile,
        1);
    return (int)cudaGetLastError();
}
