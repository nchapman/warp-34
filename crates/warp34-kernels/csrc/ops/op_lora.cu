#include "../warp34_cuda.h"

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <stdint.h>

namespace {

constexpr int kLoraMaxRankPad = 512;
constexpr int kLoraMaxSplitK = 16;

struct LoraBSliceArgs {
    int num_slices;
    int n_off0, n_off1, n_off2;
    int slice_n0, slice_n1, slice_n2;
    int r_off0, r_off1, r_off2;
    int slice_r0, slice_r1, slice_r2;
    int tile_end0, tile_end1;
};

__device__ __forceinline__ void unpack_u2_f32(
    uint2 v, float& f0, float& f1, float& f2, float& f3) {
    f0 = __half2float(__ushort_as_half((unsigned short)(v.x & 0xffffu)));
    f1 = __half2float(__ushort_as_half((unsigned short)(v.x >> 16)));
    f2 = __half2float(__ushort_as_half((unsigned short)(v.y & 0xffffu)));
    f3 = __half2float(__ushort_as_half((unsigned short)(v.y >> 16)));
}

__device__ __forceinline__ uint32_t add_f32x2_to_packed_f16x2(uint32_t y_pair, float d0, float d1) {
    const float y0 = __half2float(__ushort_as_half((unsigned short)(y_pair & 0xffffu))) + d0;
    const float y1 = __half2float(__ushort_as_half((unsigned short)(y_pair >> 16))) + d1;
    const uint32_t u0 = (uint32_t)__half_as_ushort(__float2half_rn(y0));
    const uint32_t u1 = (uint32_t)__half_as_ushort(__float2half_rn(y1));
    return u0 | (u1 << 16);
}

template <int kGroupLanes>
__device__ __forceinline__ float reduce_group_sum(float val, uint32_t mask) {
    if constexpr (kGroupLanes >= 32) val += __shfl_down_sync(mask, val, 16, 32);
    if constexpr (kGroupLanes >= 16) val += __shfl_down_sync(mask, val, 8, kGroupLanes);
    if constexpr (kGroupLanes >= 8) val += __shfl_down_sync(mask, val, 4, kGroupLanes);
    if constexpr (kGroupLanes >= 4) val += __shfl_down_sync(mask, val, 2, kGroupLanes);
    if constexpr (kGroupLanes >= 2) val += __shfl_down_sync(mask, val, 1, kGroupLanes);
    return val;
}

// Split-K LoRA shrink:
//   lowrank_out[ks, tok, r] = sum_{k in split(ks)} x[tok, k] * lora_a[lid, r, k]
// Grid: (ceil(rank_pad / kRStep), ceil(m / m_tile_size), split_k)
// Block: 64 threads (2 warps), 512 FP16 elements (1 KiB) per vectorized loop step.
template <int kRStep>
__global__ __launch_bounds__(64, 4) void lora_shrink_splitk_f16_kernel(
    const __half* __restrict__ x,
    const __half* __restrict__ lora_a,
    const int32_t* __restrict__ token_lora_ids,
    float* __restrict__ lowrank_out,
    int m,
    int m_tile_size,
    int size_k,
    int k_chunk,
    int rank_pad,
    int num_adapters,
    int default_lora_id,
    uint64_t adapter_mask,
    bool vec8_aligned) {
    const int tok_base = (int)blockIdx.y * m_tile_size;
    if (tok_base >= m) return;
    const int tile_m = (m - tok_base < m_tile_size) ? (m - tok_base) : m_tile_size;

    const int r_base = (int)blockIdx.x * kRStep;
    if (r_base >= rank_pad) return;
    const int rem_r = rank_pad - r_base;

    const int ks = (int)blockIdx.z;
    const int k_start = ks * k_chunk;
    if (k_start >= size_k) return;
    const int k_end = (size_k - k_start < k_chunk) ? size_k : (k_start + k_chunk);

    __shared__ float s_warp[4][2][kRStep];
    const int tid = (int)threadIdx.x;
    const int warp = tid >> 5;
    const int lane = tid & 31;

    int32_t uniform_lid = default_lora_id;
    bool is_uniform = (token_lora_ids == nullptr || tile_m == 1);
    if (token_lora_ids != nullptr && tile_m == 1) {
        uniform_lid = token_lora_ids[tok_base];
    }

    __shared__ int32_t s_lids[4];
    __shared__ int32_t s_all_same;
    if (!is_uniform) {
        if (tid < 4) {
            s_lids[tid] = (tid < tile_m) ? token_lora_ids[tok_base + tid] : -1;
        }
        __syncthreads();
        if (tid == 0) {
            int32_t same = 1;
            const int32_t first = s_lids[0];
            for (int t = 1; t < tile_m; ++t) {
                if (s_lids[t] != first) {
                    same = 0;
                    break;
                }
            }
            s_all_same = same;
        }
        __syncthreads();
        if (s_all_same != 0) {
            is_uniform = true;
            uniform_lid = s_lids[0];
        }
    }

    if (is_uniform) {
        const int32_t lora_id = uniform_lid;
        if (lora_id < 0 || lora_id >= num_adapters ||
            ((adapter_mask >> lora_id) & 1ULL) == 0ULL) {
            return;
        }
        const __half* a_base =
            lora_a + ((size_t)lora_id * (size_t)rank_pad + (size_t)r_base) * (size_t)size_k;

        if (tile_m == 1) {
            const __half* x_row = x + (size_t)tok_base * (size_t)size_k;
            float acc[kRStep];
            #pragma unroll
            for (int r = 0; r < kRStep; ++r) acc[r] = 0.0f;

            if (vec8_aligned && rem_r >= kRStep) {
                for (int k = k_start + tid * 8; k < k_end; k += 512) {
                    const uint4 xv = *reinterpret_cast<const uint4*>(x_row + k);
                    const __half2* xh2 = reinterpret_cast<const __half2*>(&xv);
                    const float2 xf0 = __half22float2(xh2[0]);
                    const float2 xf1 = __half22float2(xh2[1]);
                    const float2 xf2 = __half22float2(xh2[2]);
                    const float2 xf3 = __half22float2(xh2[3]);
                    #pragma unroll
                    for (int r = 0; r < kRStep; ++r) {
                        const uint4 av =
                            *reinterpret_cast<const uint4*>(a_base + (size_t)r * size_k + k);
                        const __half2* ah2 = reinterpret_cast<const __half2*>(&av);
                        const float2 af0 = __half22float2(ah2[0]);
                        const float2 af1 = __half22float2(ah2[1]);
                        const float2 af2 = __half22float2(ah2[2]);
                        const float2 af3 = __half22float2(ah2[3]);
                        acc[r] = fmaf(xf0.x, af0.x, acc[r]);
                        acc[r] = fmaf(xf0.y, af0.y, acc[r]);
                        acc[r] = fmaf(xf1.x, af1.x, acc[r]);
                        acc[r] = fmaf(xf1.y, af1.y, acc[r]);
                        acc[r] = fmaf(xf2.x, af2.x, acc[r]);
                        acc[r] = fmaf(xf2.y, af2.y, acc[r]);
                        acc[r] = fmaf(xf3.x, af3.x, acc[r]);
                        acc[r] = fmaf(xf3.y, af3.y, acc[r]);
                    }
                }
            } else {
                for (int k = k_start + tid; k < k_end; k += 64) {
                    const float xv = __half2float(x_row[k]);
                    #pragma unroll
                    for (int r = 0; r < kRStep; ++r) {
                        if (r < rem_r) {
                            acc[r] = fmaf(xv, __half2float(a_base[(size_t)r * size_k + k]), acc[r]);
                        }
                    }
                }
            }

            #pragma unroll
            for (int offset = 16; offset > 0; offset >>= 1) {
                #pragma unroll
                for (int r = 0; r < kRStep; ++r) {
                    acc[r] += __shfl_down_sync(0xffffffffu, acc[r], offset);
                }
            }
            if (lane == 0) {
                #pragma unroll
                for (int r = 0; r < kRStep; ++r) s_warp[0][warp][r] = acc[r];
            }
            __syncthreads();
            if (tid == 0) {
                float* dst = lowrank_out +
                             ((size_t)ks * (size_t)m + (size_t)tok_base) * (size_t)rank_pad +
                             (size_t)r_base;
                #pragma unroll
                for (int r = 0; r < kRStep; ++r) {
                    if (r < rem_r) dst[r] = s_warp[0][0][r] + s_warp[0][1][r];
                }
            }
            return;
        }

        float acc[4][kRStep];
        #pragma unroll
        for (int t = 0; t < 4; ++t) {
            #pragma unroll
            for (int r = 0; r < kRStep; ++r) acc[t][r] = 0.0f;
        }

        if (vec8_aligned && rem_r >= kRStep) {
            for (int k = k_start + tid * 8; k < k_end; k += 512) {
                float2 af0[kRStep], af1[kRStep], af2[kRStep], af3[kRStep];
                #pragma unroll
                for (int r = 0; r < kRStep; ++r) {
                    const uint4 av =
                        *reinterpret_cast<const uint4*>(a_base + (size_t)r * size_k + k);
                    const __half2* ah2 = reinterpret_cast<const __half2*>(&av);
                    af0[r] = __half22float2(ah2[0]);
                    af1[r] = __half22float2(ah2[1]);
                    af2[r] = __half22float2(ah2[2]);
                    af3[r] = __half22float2(ah2[3]);
                }
                #pragma unroll
                for (int t = 0; t < 4; ++t) {
                    if (t < tile_m) {
                        const uint4 xv = *reinterpret_cast<const uint4*>(
                            x + (size_t)(tok_base + t) * size_k + k);
                        const __half2* xh2 = reinterpret_cast<const __half2*>(&xv);
                        const float2 xf0 = __half22float2(xh2[0]);
                        const float2 xf1 = __half22float2(xh2[1]);
                        const float2 xf2 = __half22float2(xh2[2]);
                        const float2 xf3 = __half22float2(xh2[3]);
                        #pragma unroll
                        for (int r = 0; r < kRStep; ++r) {
                            acc[t][r] = fmaf(xf0.x, af0[r].x, acc[t][r]);
                            acc[t][r] = fmaf(xf0.y, af0[r].y, acc[t][r]);
                            acc[t][r] = fmaf(xf1.x, af1[r].x, acc[t][r]);
                            acc[t][r] = fmaf(xf1.y, af1[r].y, acc[t][r]);
                            acc[t][r] = fmaf(xf2.x, af2[r].x, acc[t][r]);
                            acc[t][r] = fmaf(xf2.y, af2[r].y, acc[t][r]);
                            acc[t][r] = fmaf(xf3.x, af3[r].x, acc[t][r]);
                            acc[t][r] = fmaf(xf3.y, af3[r].y, acc[t][r]);
                        }
                    }
                }
            }
        } else {
            for (int k = k_start + tid; k < k_end; k += 64) {
                float av[kRStep];
                #pragma unroll
                for (int r = 0; r < kRStep; ++r) {
                    av[r] = (r < rem_r) ? __half2float(a_base[(size_t)r * size_k + k]) : 0.0f;
                }
                #pragma unroll
                for (int t = 0; t < 4; ++t) {
                    if (t < tile_m) {
                        const float xv = __half2float(x[(size_t)(tok_base + t) * size_k + k]);
                        #pragma unroll
                        for (int r = 0; r < kRStep; ++r) {
                            acc[t][r] = fmaf(xv, av[r], acc[t][r]);
                        }
                    }
                }
            }
        }

        #pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1) {
            #pragma unroll
            for (int t = 0; t < 4; ++t) {
                #pragma unroll
                for (int r = 0; r < kRStep; ++r) {
                    acc[t][r] += __shfl_down_sync(0xffffffffu, acc[t][r], offset);
                }
            }
        }
        if (lane == 0) {
            #pragma unroll
            for (int t = 0; t < 4; ++t) {
                #pragma unroll
                for (int r = 0; r < kRStep; ++r) s_warp[t][warp][r] = acc[t][r];
            }
        }
        __syncthreads();
        if (tid < tile_m) {
            float* dst = lowrank_out +
                         ((size_t)ks * (size_t)m + (size_t)(tok_base + tid)) * (size_t)rank_pad +
                         (size_t)r_base;
            #pragma unroll
            for (int r = 0; r < kRStep; ++r) {
                if (r < rem_r) dst[r] = s_warp[tid][0][r] + s_warp[tid][1][r];
            }
        }
        return;
    }

    for (int t = 0; t < tile_m; ++t) {
        const int32_t lora_id = s_lids[t];
        if (lora_id < 0 || lora_id >= num_adapters ||
            ((adapter_mask >> lora_id) & 1ULL) == 0ULL) {
            continue;
        }
        const __half* x_row = x + (size_t)(tok_base + t) * (size_t)size_k;
        const __half* a_base =
            lora_a + ((size_t)lora_id * (size_t)rank_pad + (size_t)r_base) * (size_t)size_k;
        float acc[kRStep];
        #pragma unroll
        for (int r = 0; r < kRStep; ++r) acc[r] = 0.0f;

        if (vec8_aligned && rem_r >= kRStep) {
            for (int k = k_start + tid * 8; k < k_end; k += 512) {
                const uint4 xv = *reinterpret_cast<const uint4*>(x_row + k);
                const __half2* xh2 = reinterpret_cast<const __half2*>(&xv);
                const float2 xf0 = __half22float2(xh2[0]);
                const float2 xf1 = __half22float2(xh2[1]);
                const float2 xf2 = __half22float2(xh2[2]);
                const float2 xf3 = __half22float2(xh2[3]);
                #pragma unroll
                for (int r = 0; r < kRStep; ++r) {
                    const uint4 av =
                        *reinterpret_cast<const uint4*>(a_base + (size_t)r * size_k + k);
                    const __half2* ah2 = reinterpret_cast<const __half2*>(&av);
                    const float2 af0 = __half22float2(ah2[0]);
                    const float2 af1 = __half22float2(ah2[1]);
                    const float2 af2 = __half22float2(ah2[2]);
                    const float2 af3 = __half22float2(ah2[3]);
                    acc[r] = fmaf(xf0.x, af0.x, acc[r]);
                    acc[r] = fmaf(xf0.y, af0.y, acc[r]);
                    acc[r] = fmaf(xf1.x, af1.x, acc[r]);
                    acc[r] = fmaf(xf1.y, af1.y, acc[r]);
                    acc[r] = fmaf(xf2.x, af2.x, acc[r]);
                    acc[r] = fmaf(xf2.y, af2.y, acc[r]);
                    acc[r] = fmaf(xf3.x, af3.x, acc[r]);
                    acc[r] = fmaf(xf3.y, af3.y, acc[r]);
                }
            }
        } else {
            for (int k = k_start + tid; k < k_end; k += 64) {
                const float xv = __half2float(x_row[k]);
                #pragma unroll
                for (int r = 0; r < kRStep; ++r) {
                    if (r < rem_r) {
                        acc[r] = fmaf(xv, __half2float(a_base[(size_t)r * size_k + k]), acc[r]);
                    }
                }
            }
        }

        #pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1) {
            #pragma unroll
            for (int r = 0; r < kRStep; ++r) {
                acc[r] += __shfl_down_sync(0xffffffffu, acc[r], offset);
            }
        }
        if (lane == 0) {
            #pragma unroll
            for (int r = 0; r < kRStep; ++r) s_warp[0][warp][r] = acc[r];
        }
        __syncthreads();
        if (tid == 0) {
            float* dst = lowrank_out +
                         ((size_t)ks * (size_t)m + (size_t)(tok_base + t)) * (size_t)rank_pad +
                         (size_t)r_base;
            #pragma unroll
            for (int r = 0; r < kRStep; ++r) {
                if (r < rem_r) dst[r] = s_warp[0][0][r] + s_warp[0][1][r];
            }
        }
        __syncthreads();
    }
}

template <int kGroupLanes>
__device__ __forceinline__ void eval_and_add_one_token(
    const float* __restrict__ lr_row,
    const __half* __restrict__ lora_b,
    __half* __restrict__ y_inout,
    int lora_id,
    int tok,
    int size_n,
    int rank_pad,
    int slice_n_off,
    int slice_n_len,
    int slice_r_off,
    int slice_r_len,
    int local_col0,
    int group_lane,
    uint32_t group_mask,
    bool vec4_aligned,
    bool vec8_b_aligned) {
    if (local_col0 >= slice_n_len) return;
    const int col0 = slice_n_off + local_col0;
    const __half* b_row0 =
        lora_b + ((size_t)lora_id * (size_t)size_n + (size_t)col0) * (size_t)rank_pad +
        (size_t)slice_r_off;
    float sum0 = 0.0f;
    float sum1 = 0.0f;
    const bool has_col1 = (local_col0 + 1 < slice_n_len);

    if (vec8_b_aligned && kGroupLanes >= 8) {
        const float4* s_lr4 = reinterpret_cast<const float4*>(lr_row);
        const int vec8_ranks = slice_r_len >> 3;
        if (has_col1) {
            const __half* b_row1 = b_row0 + rank_pad;
            for (int r8 = group_lane; r8 < vec8_ranks; r8 += kGroupLanes) {
                const float4 hv0 = s_lr4[r8 * 2];
                const float4 hv1 = s_lr4[r8 * 2 + 1];
                const uint4 bv0 = *reinterpret_cast<const uint4*>(b_row0 + r8 * 8);
                const uint4 bv1 = *reinterpret_cast<const uint4*>(b_row1 + r8 * 8);
                float b00, b01, b02, b03, b04, b05, b06, b07;
                float b10, b11, b12, b13, b14, b15, b16, b17;
                unpack_u2_f32(make_uint2(bv0.x, bv0.y), b00, b01, b02, b03);
                unpack_u2_f32(make_uint2(bv0.z, bv0.w), b04, b05, b06, b07);
                unpack_u2_f32(make_uint2(bv1.x, bv1.y), b10, b11, b12, b13);
                unpack_u2_f32(make_uint2(bv1.z, bv1.w), b14, b15, b16, b17);
                sum0 = fmaf(hv0.x, b00, sum0);
                sum0 = fmaf(hv0.y, b01, sum0);
                sum0 = fmaf(hv0.z, b02, sum0);
                sum0 = fmaf(hv0.w, b03, sum0);
                sum0 = fmaf(hv1.x, b04, sum0);
                sum0 = fmaf(hv1.y, b05, sum0);
                sum0 = fmaf(hv1.z, b06, sum0);
                sum0 = fmaf(hv1.w, b07, sum0);
                sum1 = fmaf(hv0.x, b10, sum1);
                sum1 = fmaf(hv0.y, b11, sum1);
                sum1 = fmaf(hv0.z, b12, sum1);
                sum1 = fmaf(hv0.w, b13, sum1);
                sum1 = fmaf(hv1.x, b14, sum1);
                sum1 = fmaf(hv1.y, b15, sum1);
                sum1 = fmaf(hv1.z, b16, sum1);
                sum1 = fmaf(hv1.w, b17, sum1);
            }
            sum0 = reduce_group_sum<kGroupLanes>(sum0, group_mask);
            sum1 = reduce_group_sum<kGroupLanes>(sum1, group_mask);
        } else {
            for (int r8 = group_lane; r8 < vec8_ranks; r8 += kGroupLanes) {
                const float4 hv0 = s_lr4[r8 * 2];
                const float4 hv1 = s_lr4[r8 * 2 + 1];
                const uint4 bv0 = *reinterpret_cast<const uint4*>(b_row0 + r8 * 8);
                float b00, b01, b02, b03, b04, b05, b06, b07;
                unpack_u2_f32(make_uint2(bv0.x, bv0.y), b00, b01, b02, b03);
                unpack_u2_f32(make_uint2(bv0.z, bv0.w), b04, b05, b06, b07);
                sum0 = fmaf(hv0.x, b00, sum0);
                sum0 = fmaf(hv0.y, b01, sum0);
                sum0 = fmaf(hv0.z, b02, sum0);
                sum0 = fmaf(hv0.w, b03, sum0);
                sum0 = fmaf(hv1.x, b04, sum0);
                sum0 = fmaf(hv1.y, b05, sum0);
                sum0 = fmaf(hv1.z, b06, sum0);
                sum0 = fmaf(hv1.w, b07, sum0);
            }
            sum0 = reduce_group_sum<kGroupLanes>(sum0, group_mask);
        }
    } else if (vec4_aligned) {
        const float4* s_lr4 = reinterpret_cast<const float4*>(lr_row);
        const int vec_ranks = slice_r_len >> 2;
        if (has_col1) {
            if constexpr (kGroupLanes == 1) {
                if (rank_pad == 4 && slice_r_len == 4 && slice_r_off == 0 &&
                    ((reinterpret_cast<uintptr_t>(b_row0) & 15u) == 0u)) {
                    const float4 hv = s_lr4[0];
                    const uint4 bv = *reinterpret_cast<const uint4*>(b_row0);
                    float b00, b01, b02, b03, b10, b11, b12, b13;
                    unpack_u2_f32(make_uint2(bv.x, bv.y), b00, b01, b02, b03);
                    unpack_u2_f32(make_uint2(bv.z, bv.w), b10, b11, b12, b13);
                    sum0 = fmaf(hv.x, b00, fmaf(hv.y, b01, fmaf(hv.z, b02, hv.w * b03)));
                    sum1 = fmaf(hv.x, b10, fmaf(hv.y, b11, fmaf(hv.z, b12, hv.w * b13)));
                } else {
                    const __half* b_row1 = b_row0 + rank_pad;
                    for (int r4 = 0; r4 < vec_ranks; ++r4) {
                        const float4 hv = s_lr4[r4];
                        const uint2 bv0 = *reinterpret_cast<const uint2*>(b_row0 + r4 * 4);
                        const uint2 bv1 = *reinterpret_cast<const uint2*>(b_row1 + r4 * 4);
                        float b00, b01, b02, b03, b10, b11, b12, b13;
                        unpack_u2_f32(bv0, b00, b01, b02, b03);
                        unpack_u2_f32(bv1, b10, b11, b12, b13);
                        sum0 = fmaf(hv.x, b00, sum0);
                        sum0 = fmaf(hv.y, b01, sum0);
                        sum0 = fmaf(hv.z, b02, sum0);
                        sum0 = fmaf(hv.w, b03, sum0);
                        sum1 = fmaf(hv.x, b10, sum1);
                        sum1 = fmaf(hv.y, b11, sum1);
                        sum1 = fmaf(hv.z, b12, sum1);
                        sum1 = fmaf(hv.w, b13, sum1);
                    }
                }
            } else {
                const __half* b_row1 = b_row0 + rank_pad;
                for (int r4 = group_lane; r4 < vec_ranks; r4 += kGroupLanes) {
                    const float4 hv = s_lr4[r4];
                    const uint2 bv0 = *reinterpret_cast<const uint2*>(b_row0 + r4 * 4);
                    const uint2 bv1 = *reinterpret_cast<const uint2*>(b_row1 + r4 * 4);
                    float b00, b01, b02, b03, b10, b11, b12, b13;
                    unpack_u2_f32(bv0, b00, b01, b02, b03);
                    unpack_u2_f32(bv1, b10, b11, b12, b13);
                    sum0 = fmaf(hv.x, b00, sum0);
                    sum0 = fmaf(hv.y, b01, sum0);
                    sum0 = fmaf(hv.z, b02, sum0);
                    sum0 = fmaf(hv.w, b03, sum0);
                    sum1 = fmaf(hv.x, b10, sum1);
                    sum1 = fmaf(hv.y, b11, sum1);
                    sum1 = fmaf(hv.z, b12, sum1);
                    sum1 = fmaf(hv.w, b13, sum1);
                }
                sum0 = reduce_group_sum<kGroupLanes>(sum0, group_mask);
                sum1 = reduce_group_sum<kGroupLanes>(sum1, group_mask);
            }
        } else {
            for (int r4 = group_lane; r4 < vec_ranks; r4 += kGroupLanes) {
                const float4 hv = s_lr4[r4];
                const uint2 bv0 = *reinterpret_cast<const uint2*>(b_row0 + r4 * 4);
                float b00, b01, b02, b03;
                unpack_u2_f32(bv0, b00, b01, b02, b03);
                sum0 = fmaf(hv.x, b00, sum0);
                sum0 = fmaf(hv.y, b01, sum0);
                sum0 = fmaf(hv.z, b02, sum0);
                sum0 = fmaf(hv.w, b03, sum0);
            }
            sum0 = reduce_group_sum<kGroupLanes>(sum0, group_mask);
        }
    } else {
        const __half* b_row1 = b_row0 + rank_pad;
        for (int r = group_lane; r < slice_r_len; r += kGroupLanes) {
            const float hv = lr_row[r];
            sum0 = fmaf(hv, __half2float(b_row0[r]), sum0);
            if (has_col1) sum1 = fmaf(hv, __half2float(b_row1[r]), sum1);
        }
        sum0 = reduce_group_sum<kGroupLanes>(sum0, group_mask);
        if (has_col1) sum1 = reduce_group_sum<kGroupLanes>(sum1, group_mask);
    }

    if (group_lane == 0) {
        __half* y_ptr = y_inout + (size_t)tok * (size_t)size_n + (size_t)col0;
        if (has_col1) {
            if ((reinterpret_cast<uintptr_t>(y_ptr) & 3u) == 0u) {
                const uint32_t y_pair = *reinterpret_cast<const uint32_t*>(y_ptr);
                *reinterpret_cast<uint32_t*>(y_ptr) =
                    add_f32x2_to_packed_f16x2(y_pair, sum0, sum1);
            } else {
                y_ptr[0] = __float2half_rn(__half2float(y_ptr[0]) + sum0);
                y_ptr[1] = __float2half_rn(__half2float(y_ptr[1]) + sum1);
            }
        } else {
            y_ptr[0] = __float2half_rn(__half2float(y_ptr[0]) + sum0);
        }
    }
}

// Warp-coalesced sliced LoRA expand + in-place FP16 accumulate:
//   1. Reduces Split-K partials across ks in 0..split_k into dynamic shared memory `s_lowrank`.
//   2. Maps CTA `blockIdx.x` directly onto active non-zero B slices `(n_off, slice_n, r_off, slice_r)`.
//   3. Coalesces row-major `lora_b` loads across `kGroupLanes` consecutive threads per column pair.
template <int kBlockThreads, int kGroupLanes>
__global__ __launch_bounds__(kBlockThreads, (512 / kBlockThreads))
void lora_expand_add_coalesced_f16_kernel(
    const float* __restrict__ lowrank_in,
    const __half* __restrict__ lora_b,
    const int32_t* __restrict__ token_lora_ids,
    __half* __restrict__ y_inout,
    LoraBSliceArgs slices,
    int tiles_per_cta,
    int m,
    int split_k,
    int size_n,
    int rank_pad,
    int num_adapters,
    int default_lora_id,
    uint64_t adapter_mask,
    bool vec4_aligned,
    bool vec8_b_aligned) {
    const int tok_base = (int)blockIdx.y * 8;
    if (tok_base >= m) return;
    const int tile_m = (m - tok_base < 8) ? (m - tok_base) : 8;

    int s_idx = 0;
    if (slices.num_slices > 1 && (int)blockIdx.x >= slices.tile_end0) {
        s_idx = (slices.num_slices > 2 && (int)blockIdx.x >= slices.tile_end1) ? 2 : 1;
    }
    const int tile_start =
        (s_idx == 0) ? 0 : ((s_idx == 1) ? slices.tile_end0 : slices.tile_end1);
    const int local_cta = (int)blockIdx.x - tile_start;
    const int slice_n_off =
        (s_idx == 0) ? slices.n_off0 : ((s_idx == 1) ? slices.n_off1 : slices.n_off2);
    const int slice_n_len =
        (s_idx == 0) ? slices.slice_n0 : ((s_idx == 1) ? slices.slice_n1 : slices.slice_n2);
    const int slice_r_off =
        (s_idx == 0) ? slices.r_off0 : ((s_idx == 1) ? slices.r_off1 : slices.r_off2);
    const int slice_r_len =
        (s_idx == 0) ? slices.slice_r0 : ((s_idx == 1) ? slices.slice_r1 : slices.slice_r2);
    const int slice_r_stride = vec8_b_aligned ? ((slice_r_len + 7) & ~7) : ((slice_r_len + 3) & ~3);

    constexpr int kGroupsPerBlock = kBlockThreads / kGroupLanes;
    constexpr int kColsPerBlock = kGroupsPerBlock * 2;
    const int tid = (int)threadIdx.x;
    const int lane_in_warp = tid & 31;
    const int group_id = tid / kGroupLanes;
    const int group_lane = tid & (kGroupLanes - 1);
    const uint32_t group_mask =
        (kGroupLanes == 32)
            ? 0xffffffffu
            : (((1u << kGroupLanes) - 1u) << ((lane_in_warp / kGroupLanes) * kGroupLanes));

    extern __shared__ __align__(16) float s_lowrank[];
    __shared__ int32_t s_lids[8];
    __shared__ int32_t s_all_same;

    int32_t uniform_lid = default_lora_id;
    bool is_uniform = (token_lora_ids == nullptr || tile_m == 1);
    if (token_lora_ids != nullptr && tile_m == 1) {
        uniform_lid = token_lora_ids[tok_base];
    }

    if (!is_uniform) {
        if (tid < 8) {
            s_lids[tid] = (tid < tile_m) ? token_lora_ids[tok_base + tid] : -1;
        }
        __syncthreads();
        if (tid == 0) {
            int32_t same = 1;
            const int32_t first = s_lids[0];
            for (int t = 1; t < tile_m; ++t) {
                if (s_lids[t] != first) {
                    same = 0;
                    break;
                }
            }
            s_all_same = same;
        }
        __syncthreads();
        if (s_all_same != 0) {
            is_uniform = true;
            uniform_lid = s_lids[0];
        }
    }

    const size_t stride_ks = (size_t)m * (size_t)rank_pad;

    if (is_uniform) {
        const int32_t lora_id = uniform_lid;
        if (lora_id < 0 || lora_id >= num_adapters ||
            ((adapter_mask >> lora_id) & 1ULL) == 0ULL) {
            return;
        }
        const int total_lr = tile_m * slice_r_stride;
        for (int idx = tid; idx < total_lr; idx += kBlockThreads) {
            const int t = idx / slice_r_stride;
            const int r = idx - t * slice_r_stride;
            float acc_lr = 0.0f;
            if (r < slice_r_len) {
                const float* src = lowrank_in +
                                   (size_t)(tok_base + t) * (size_t)rank_pad +
                                   (size_t)(slice_r_off + r);
                for (int ks = 0; ks < split_k; ++ks) {
                    acc_lr += src[(size_t)ks * stride_ks];
                }
            }
            s_lowrank[idx] = acc_lr;
        }
        __syncthreads();

        if (tile_m == 1) {
            for (int sub = 0; sub < tiles_per_cta; ++sub) {
                const int local_col0 =
                    (local_cta * tiles_per_cta + sub) * kColsPerBlock + group_id * 2;
                if (local_col0 >= slice_n_len) break;
                eval_and_add_one_token<kGroupLanes>(
                    s_lowrank,
                    lora_b,
                    y_inout,
                    lora_id,
                    tok_base,
                    size_n,
                    rank_pad,
                    slice_n_off,
                    slice_n_len,
                    slice_r_off,
                    slice_r_len,
                    local_col0,
                    group_lane,
                    group_mask,
                    vec4_aligned,
                    vec8_b_aligned);
            }
            return;
        }

        for (int sub = 0; sub < tiles_per_cta; ++sub) {
            const int local_col0 =
                (local_cta * tiles_per_cta + sub) * kColsPerBlock + group_id * 2;
            if (local_col0 >= slice_n_len) break;
            const int col0 = slice_n_off + local_col0;
            const bool has_col1 = (local_col0 + 1 < slice_n_len);
            const __half* b_row0 =
                lora_b + ((size_t)lora_id * (size_t)size_n + (size_t)col0) * (size_t)rank_pad +
                (size_t)slice_r_off;
            const __half* b_row1 = b_row0 + rank_pad;

            float sum0[8] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
            float sum1[8] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};

            if (vec8_b_aligned && kGroupLanes >= 8) {
                const int vec8_ranks = slice_r_len >> 3;
                if (has_col1) {
                    for (int r8 = group_lane; r8 < vec8_ranks; r8 += kGroupLanes) {
                        const uint4 bv0 = *reinterpret_cast<const uint4*>(b_row0 + r8 * 8);
                        const uint4 bv1 = *reinterpret_cast<const uint4*>(b_row1 + r8 * 8);
                        float b00, b01, b02, b03, b04, b05, b06, b07;
                        float b10, b11, b12, b13, b14, b15, b16, b17;
                        unpack_u2_f32(make_uint2(bv0.x, bv0.y), b00, b01, b02, b03);
                        unpack_u2_f32(make_uint2(bv0.z, bv0.w), b04, b05, b06, b07);
                        unpack_u2_f32(make_uint2(bv1.x, bv1.y), b10, b11, b12, b13);
                        unpack_u2_f32(make_uint2(bv1.z, bv1.w), b14, b15, b16, b17);
                        #pragma unroll
                        for (int t = 0; t < 8; ++t) {
                            if (t < tile_m) {
                                const float4* lr4 = reinterpret_cast<const float4*>(
                                    s_lowrank + (size_t)t * (size_t)slice_r_stride);
                                const float4 hv0 = lr4[r8 * 2];
                                const float4 hv1 = lr4[r8 * 2 + 1];
                                sum0[t] = fmaf(hv0.x, b00, sum0[t]);
                                sum0[t] = fmaf(hv0.y, b01, sum0[t]);
                                sum0[t] = fmaf(hv0.z, b02, sum0[t]);
                                sum0[t] = fmaf(hv0.w, b03, sum0[t]);
                                sum0[t] = fmaf(hv1.x, b04, sum0[t]);
                                sum0[t] = fmaf(hv1.y, b05, sum0[t]);
                                sum0[t] = fmaf(hv1.z, b06, sum0[t]);
                                sum0[t] = fmaf(hv1.w, b07, sum0[t]);
                                sum1[t] = fmaf(hv0.x, b10, sum1[t]);
                                sum1[t] = fmaf(hv0.y, b11, sum1[t]);
                                sum1[t] = fmaf(hv0.z, b12, sum1[t]);
                                sum1[t] = fmaf(hv0.w, b13, sum1[t]);
                                sum1[t] = fmaf(hv1.x, b14, sum1[t]);
                                sum1[t] = fmaf(hv1.y, b15, sum1[t]);
                                sum1[t] = fmaf(hv1.z, b16, sum1[t]);
                                sum1[t] = fmaf(hv1.w, b17, sum1[t]);
                            }
                        }
                    }
                } else {
                    for (int r8 = group_lane; r8 < vec8_ranks; r8 += kGroupLanes) {
                        const uint4 bv0 = *reinterpret_cast<const uint4*>(b_row0 + r8 * 8);
                        float b00, b01, b02, b03, b04, b05, b06, b07;
                        unpack_u2_f32(make_uint2(bv0.x, bv0.y), b00, b01, b02, b03);
                        unpack_u2_f32(make_uint2(bv0.z, bv0.w), b04, b05, b06, b07);
                        #pragma unroll
                        for (int t = 0; t < 8; ++t) {
                            if (t < tile_m) {
                                const float4* lr4 = reinterpret_cast<const float4*>(
                                    s_lowrank + (size_t)t * (size_t)slice_r_stride);
                                const float4 hv0 = lr4[r8 * 2];
                                const float4 hv1 = lr4[r8 * 2 + 1];
                                sum0[t] = fmaf(hv0.x, b00, sum0[t]);
                                sum0[t] = fmaf(hv0.y, b01, sum0[t]);
                                sum0[t] = fmaf(hv0.z, b02, sum0[t]);
                                sum0[t] = fmaf(hv0.w, b03, sum0[t]);
                                sum0[t] = fmaf(hv1.x, b04, sum0[t]);
                                sum0[t] = fmaf(hv1.y, b05, sum0[t]);
                                sum0[t] = fmaf(hv1.z, b06, sum0[t]);
                                sum0[t] = fmaf(hv1.w, b07, sum0[t]);
                            }
                        }
                    }
                }
            } else if (vec4_aligned) {
                const int vec_ranks = slice_r_len >> 2;
                if (has_col1) {
                    for (int r4 = group_lane; r4 < vec_ranks; r4 += kGroupLanes) {
                        const uint2 bv0 = *reinterpret_cast<const uint2*>(b_row0 + r4 * 4);
                        const uint2 bv1 = *reinterpret_cast<const uint2*>(b_row1 + r4 * 4);
                        float b00, b01, b02, b03, b10, b11, b12, b13;
                        unpack_u2_f32(bv0, b00, b01, b02, b03);
                        unpack_u2_f32(bv1, b10, b11, b12, b13);
                        #pragma unroll
                        for (int t = 0; t < 8; ++t) {
                            if (t < tile_m) {
                                const float4 hv = reinterpret_cast<const float4*>(
                                    s_lowrank + (size_t)t * (size_t)slice_r_stride)[r4];
                                sum0[t] = fmaf(hv.x, b00, sum0[t]);
                                sum0[t] = fmaf(hv.y, b01, sum0[t]);
                                sum0[t] = fmaf(hv.z, b02, sum0[t]);
                                sum0[t] = fmaf(hv.w, b03, sum0[t]);
                                sum1[t] = fmaf(hv.x, b10, sum1[t]);
                                sum1[t] = fmaf(hv.y, b11, sum1[t]);
                                sum1[t] = fmaf(hv.z, b12, sum1[t]);
                                sum1[t] = fmaf(hv.w, b13, sum1[t]);
                            }
                        }
                    }
                } else {
                    for (int r4 = group_lane; r4 < vec_ranks; r4 += kGroupLanes) {
                        const uint2 bv0 = *reinterpret_cast<const uint2*>(b_row0 + r4 * 4);
                        float b00, b01, b02, b03;
                        unpack_u2_f32(bv0, b00, b01, b02, b03);
                        #pragma unroll
                        for (int t = 0; t < 8; ++t) {
                            if (t < tile_m) {
                                const float4 hv = reinterpret_cast<const float4*>(
                                    s_lowrank + (size_t)t * (size_t)slice_r_stride)[r4];
                                sum0[t] = fmaf(hv.x, b00, sum0[t]);
                                sum0[t] = fmaf(hv.y, b01, sum0[t]);
                                sum0[t] = fmaf(hv.z, b02, sum0[t]);
                                sum0[t] = fmaf(hv.w, b03, sum0[t]);
                            }
                        }
                    }
                }
            } else {
                for (int r = group_lane; r < slice_r_len; r += kGroupLanes) {
                    const float b0 = __half2float(b_row0[r]);
                    const float b1 = has_col1 ? __half2float(b_row1[r]) : 0.0f;
                    #pragma unroll
                    for (int t = 0; t < 8; ++t) {
                        if (t < tile_m) {
                            const float hv = s_lowrank[(size_t)t * (size_t)slice_r_stride + r];
                            sum0[t] = fmaf(hv, b0, sum0[t]);
                            sum1[t] = fmaf(hv, b1, sum1[t]);
                        }
                    }
                }
            }

            #pragma unroll
            for (int t = 0; t < 8; ++t) {
                if (t < tile_m) {
                    sum0[t] = reduce_group_sum<kGroupLanes>(sum0[t], group_mask);
                    if (has_col1) {
                        sum1[t] = reduce_group_sum<kGroupLanes>(sum1[t], group_mask);
                    }
                }
            }

            if (group_lane == 0) {
                const bool y_aligned =
                    has_col1 && ((size_n & 1) == 0) &&
                    ((reinterpret_cast<uintptr_t>(y_inout + col0) & 3u) == 0u);
                if (y_aligned) {
                    uint32_t yp[8] = {0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u};
                    #pragma unroll
                    for (int t = 0; t < 8; ++t) {
                        if (t < tile_m) {
                            yp[t] = *reinterpret_cast<const uint32_t*>(
                                y_inout + (size_t)(tok_base + t) * (size_t)size_n + (size_t)col0);
                        }
                    }
                    #pragma unroll
                    for (int t = 0; t < 8; ++t) {
                        if (t < tile_m) {
                            *reinterpret_cast<uint32_t*>(
                                y_inout + (size_t)(tok_base + t) * (size_t)size_n + (size_t)col0) =
                                add_f32x2_to_packed_f16x2(yp[t], sum0[t], sum1[t]);
                        }
                    }
                } else {
                    #pragma unroll
                    for (int t = 0; t < 8; ++t) {
                        if (t < tile_m) {
                            __half* y_ptr = y_inout + (size_t)(tok_base + t) * (size_t)size_n +
                                            (size_t)col0;
                            y_ptr[0] = __float2half_rn(__half2float(y_ptr[0]) + sum0[t]);
                            if (has_col1) {
                                y_ptr[1] = __float2half_rn(__half2float(y_ptr[1]) + sum1[t]);
                            }
                        }
                    }
                }
            }
        }
        return;
    }

    for (int t = 0; t < tile_m; ++t) {
        const int32_t lora_id = s_lids[t];
        if (lora_id < 0 || lora_id >= num_adapters ||
            ((adapter_mask >> lora_id) & 1ULL) == 0ULL) {
            continue;
        }
        const float* src_base =
            lowrank_in + (size_t)(tok_base + t) * (size_t)rank_pad + (size_t)slice_r_off;
        for (int r = tid; r < slice_r_stride; r += kBlockThreads) {
            float acc_lr = 0.0f;
            if (r < slice_r_len) {
                for (int ks = 0; ks < split_k; ++ks) {
                    acc_lr += src_base[(size_t)ks * stride_ks + (size_t)r];
                }
            }
            s_lowrank[r] = acc_lr;
        }
        __syncthreads();
        for (int sub = 0; sub < tiles_per_cta; ++sub) {
            const int local_col0 =
                (local_cta * tiles_per_cta + sub) * kColsPerBlock + group_id * 2;
            if (local_col0 >= slice_n_len) break;
            eval_and_add_one_token<kGroupLanes>(
                s_lowrank,
                lora_b,
                y_inout,
                lora_id,
                tok_base + t,
                size_n,
                rank_pad,
                slice_n_off,
                slice_n_len,
                slice_r_off,
                slice_r_len,
                local_col0,
                group_lane,
                group_mask,
                vec4_aligned,
                vec8_b_aligned);
        }
        __syncthreads();
    }
}

template <int kBlockThreads, int kGroupLanes>
int launch_lora_expand(
    cudaStream_t stream,
    int sm_count,
    const float* lowrank_f32_ws,
    const __half* lora_b_stacked,
    const int32_t* token_lora_ids,
    __half* y_inout_f16,
    LoraBSliceArgs slices,
    int max_slice_r,
    int m,
    int split_k,
    int size_n,
    int rank_pad,
    int num_adapters,
    int default_lora_id,
    uint64_t adapter_mask,
    bool vec4_aligned,
    bool vec8_b_aligned) {
    constexpr int kColsPerBlock = (kBlockThreads / kGroupLanes) * 2;
    const int s_lens[3] = {slices.slice_n0, slices.slice_n1, slices.slice_n2};
    const int y_blocks = (m + 7) / 8;
    int natural_tiles = 0;
    for (int s = 0; s < slices.num_slices; ++s) {
        natural_tiles += (s_lens[s] + kColsPerBlock - 1) / kColsPerBlock;
    }
    if (natural_tiles == 0) return int(cudaSuccess);

    const int target_x_ctas = (2 * sm_count + y_blocks - 1) / y_blocks;
    const int tiles_per_cta =
        (target_x_ctas > 0 && natural_tiles > target_x_ctas)
            ? ((natural_tiles + target_x_ctas - 1) / target_x_ctas)
            : 1;
    const int cta_cols = kColsPerBlock * tiles_per_cta;

    int cum_ctas = 0;
    for (int s = 0; s < slices.num_slices; ++s) {
        const int ctas = (s_lens[s] + cta_cols - 1) / cta_cols;
        cum_ctas += ctas;
        if (s == 0) slices.tile_end0 = cum_ctas;
        if (s == 1) slices.tile_end1 = cum_ctas;
    }

    const int smem_rows = (token_lora_ids == nullptr && m == 1) ? 1 : ((m < 8) ? m : 8);
    const int max_r_stride = vec8_b_aligned ? ((max_slice_r + 7) & ~7) : ((max_slice_r + 3) & ~3);
    size_t smem_bytes = (size_t)smem_rows * (size_t)max_r_stride * sizeof(float);
    if (smem_bytes < 16u) smem_bytes = 16u;

    const dim3 expand_grid((unsigned int)cum_ctas, (unsigned int)y_blocks, 1u);
    lora_expand_add_coalesced_f16_kernel<kBlockThreads, kGroupLanes>
        <<<expand_grid, kBlockThreads, smem_bytes, stream>>>(
            lowrank_f32_ws,
            lora_b_stacked,
            token_lora_ids,
            y_inout_f16,
            slices,
            tiles_per_cta,
            m,
            split_k,
            size_n,
            rank_pad,
            num_adapters,
            default_lora_id,
            adapter_mask,
            vec4_aligned,
            vec8_b_aligned);
    return int(cudaGetLastError());
}

}  // namespace

extern "C" int warp34_lora_shrink_expand_f16(
    const warp34_exec* ex,
    const void* x_f16,
    const void* lora_a_stacked,
    const void* lora_b_stacked,
    const int32_t* token_lora_ids,
    float* lowrank_f32_ws,
    size_t lowrank_ws_elems,
    void* y_inout_f16,
    int m,
    int size_n,
    int size_k,
    int rank_pad,
    int num_adapters,
    int default_lora_id,
    uint64_t adapter_mask,
    int num_b_slices,
    const int32_t* b_slices4) {
    if (ex == nullptr || x_f16 == nullptr || lora_a_stacked == nullptr ||
        lora_b_stacked == nullptr || lowrank_f32_ws == nullptr || y_inout_f16 == nullptr ||
        m < 0 || size_n <= 0 || size_k <= 0 || rank_pad <= 0 ||
        rank_pad > kLoraMaxRankPad || num_adapters <= 0 || num_adapters > 64 ||
        num_b_slices < 0 || num_b_slices > 3) {
        return int(cudaErrorInvalidValue);
    }
    if (m == 0 || adapter_mask == 0ULL) {
        return int(cudaSuccess);
    }
    if (token_lora_ids == nullptr &&
        (default_lora_id < 0 || default_lora_id >= num_adapters ||
         ((adapter_mask >> default_lora_id) & 1ULL) == 0ULL)) {
        return int(cudaSuccess);
    }

    const size_t min_ws = (size_t)m * (size_t)rank_pad;
    if (lowrank_ws_elems < min_ws) {
        return int(cudaErrorInvalidValue);
    }

    LoraBSliceArgs slices{};
    bool slices_r_mod4 = true;
    bool slices_r_mod8 = true;
    int max_slice_r = rank_pad;
    int active_n_sum = 0;
    if (num_b_slices >= 1) {
        if (b_slices4 == nullptr) return int(cudaErrorInvalidValue);
        slices.num_slices = num_b_slices;
        max_slice_r = 0;
        int prev_end = 0;
        for (int s = 0; s < num_b_slices; ++s) {
            const int n_off = b_slices4[s * 4 + 0];
            const int slice_n = b_slices4[s * 4 + 1];
            const int r_off = b_slices4[s * 4 + 2];
            const int slice_r = b_slices4[s * 4 + 3];
            if (n_off < prev_end || slice_n <= 0 || r_off < 0 || slice_r <= 0 ||
                (int64_t)n_off + (int64_t)slice_n > (int64_t)size_n ||
                (int64_t)r_off + (int64_t)slice_r > (int64_t)rank_pad) {
                return int(cudaErrorInvalidValue);
            }
            prev_end = n_off + slice_n;
            active_n_sum += slice_n;
            if (s == 0) {
                slices.n_off0 = n_off;
                slices.slice_n0 = slice_n;
                slices.r_off0 = r_off;
                slices.slice_r0 = slice_r;
            } else if (s == 1) {
                slices.n_off1 = n_off;
                slices.slice_n1 = slice_n;
                slices.r_off1 = r_off;
                slices.slice_r1 = slice_r;
            } else {
                slices.n_off2 = n_off;
                slices.slice_n2 = slice_n;
                slices.r_off2 = r_off;
                slices.slice_r2 = slice_r;
            }
            if ((r_off % 4) != 0 || (slice_r % 4) != 0) slices_r_mod4 = false;
            if ((r_off % 8) != 0 || (slice_r % 8) != 0) slices_r_mod8 = false;
            if (slice_r > max_slice_r) max_slice_r = slice_r;
        }
    } else {
        slices.num_slices = 1;
        slices.n_off0 = 0;
        slices.slice_n0 = size_n;
        slices.r_off0 = 0;
        slices.slice_r0 = rank_pad;
        active_n_sum = size_n;
        slices_r_mod4 = ((rank_pad % 4) == 0);
        slices_r_mod8 = ((rank_pad % 8) == 0);
        max_slice_r = rank_pad;
    }

    warp34_device_props dev_props{};
    const int sm_count =
        (warp34_get_current_device_props(&dev_props) == 0 && dev_props.multi_processor_count > 0)
            ? dev_props.multi_processor_count
            : 84;

    const int r_step = (rank_pad <= 4) ? 1 : ((rank_pad <= 16) ? 2 : 4);
    const int m_tile_size = (m == 1 || (token_lora_ids != nullptr && m <= 16)) ? 1 : 4;
    const int grid_r = (rank_pad + r_step - 1) / r_step;
    const int grid_m = (m + m_tile_size - 1) / m_tile_size;
    const int base_ctas = grid_r * grid_m;

    const size_t max_ws_splits = lowrank_ws_elems / min_ws;
    const int max_k_splits = (size_k >= 1024) ? (size_k / 512) : 1;
    int split_k = (sm_count + base_ctas - 1) / base_ctas;
    if (split_k > kLoraMaxSplitK) split_k = kLoraMaxSplitK;
    if (split_k > max_k_splits) split_k = max_k_splits;
    if ((size_t)split_k > max_ws_splits) split_k = (int)max_ws_splits;
    if (split_k < 1) split_k = 1;

    int k_chunk = size_k;
    if (split_k > 1) {
        k_chunk = (((size_k + split_k - 1) / split_k) + 63) & ~63;
        split_k = (size_k + k_chunk - 1) / k_chunk;
        if (split_k < 1) split_k = 1;
    }

    const bool vec8_aligned =
        ((size_k % 8) == 0) && ((k_chunk % 8) == 0) &&
        ((reinterpret_cast<uintptr_t>(x_f16) & 15u) == 0u) &&
        ((reinterpret_cast<uintptr_t>(lora_a_stacked) & 15u) == 0u);
    const bool vec4_aligned =
        ((rank_pad % 4) == 0) && slices_r_mod4 &&
        ((reinterpret_cast<uintptr_t>(lora_b_stacked) & 7u) == 0u);
    const bool vec8_b_aligned =
        ((rank_pad % 8) == 0) && slices_r_mod8 &&
        ((reinterpret_cast<uintptr_t>(lora_b_stacked) & 15u) == 0u);

    cudaStream_t stream = reinterpret_cast<cudaStream_t>(ex->stream);
    const dim3 shrink_grid((unsigned int)grid_r, (unsigned int)grid_m, (unsigned int)split_k);
    if (r_step == 1) {
        lora_shrink_splitk_f16_kernel<1><<<shrink_grid, 64, 0, stream>>>(
            static_cast<const __half*>(x_f16),
            static_cast<const __half*>(lora_a_stacked),
            token_lora_ids,
            lowrank_f32_ws,
            m,
            m_tile_size,
            size_k,
            k_chunk,
            rank_pad,
            num_adapters,
            default_lora_id,
            adapter_mask,
            vec8_aligned);
    } else if (r_step == 2) {
        lora_shrink_splitk_f16_kernel<2><<<shrink_grid, 64, 0, stream>>>(
            static_cast<const __half*>(x_f16),
            static_cast<const __half*>(lora_a_stacked),
            token_lora_ids,
            lowrank_f32_ws,
            m,
            m_tile_size,
            size_k,
            k_chunk,
            rank_pad,
            num_adapters,
            default_lora_id,
            adapter_mask,
            vec8_aligned);
    } else {
        lora_shrink_splitk_f16_kernel<4><<<shrink_grid, 64, 0, stream>>>(
            static_cast<const __half*>(x_f16),
            static_cast<const __half*>(lora_a_stacked),
            token_lora_ids,
            lowrank_f32_ws,
            m,
            m_tile_size,
            size_k,
            k_chunk,
            rank_pad,
            num_adapters,
            default_lora_id,
            adapter_mask,
            vec8_aligned);
    }
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) return int(err);

    const __half* b_ptr = static_cast<const __half*>(lora_b_stacked);
    __half* y_ptr = static_cast<__half*>(y_inout_f16);
    if (!vec4_aligned || max_slice_r <= 4) {
        return launch_lora_expand<64, 1>(
            stream, sm_count, lowrank_f32_ws, b_ptr, token_lora_ids, y_ptr, slices, max_slice_r,
            m, split_k, size_n, rank_pad, num_adapters, default_lora_id, adapter_mask,
            vec4_aligned, false);
    }
    if (max_slice_r <= 8) {
        if (active_n_sum >= 4096) {
            return launch_lora_expand<256, 2>(
                stream, sm_count, lowrank_f32_ws, b_ptr, token_lora_ids, y_ptr, slices,
                max_slice_r, m, split_k, size_n, rank_pad, num_adapters, default_lora_id,
                adapter_mask, vec4_aligned, false);
        }
        return launch_lora_expand<64, 2>(
            stream, sm_count, lowrank_f32_ws, b_ptr, token_lora_ids, y_ptr, slices, max_slice_r,
            m, split_k, size_n, rank_pad, num_adapters, default_lora_id, adapter_mask,
            vec4_aligned, false);
    }
    if (max_slice_r <= 16) {
        return launch_lora_expand<128, 4>(
            stream, sm_count, lowrank_f32_ws, b_ptr, token_lora_ids, y_ptr, slices, max_slice_r,
            m, split_k, size_n, rank_pad, num_adapters, default_lora_id, adapter_mask,
            vec4_aligned, false);
    }
    if (max_slice_r <= 32) {
        return launch_lora_expand<128, 8>(
            stream, sm_count, lowrank_f32_ws, b_ptr, token_lora_ids, y_ptr, slices, max_slice_r,
            m, split_k, size_n, rank_pad, num_adapters, default_lora_id, adapter_mask,
            vec4_aligned, vec8_b_aligned);
    }
    if (max_slice_r <= 64) {
        return launch_lora_expand<256, 8>(
            stream, sm_count, lowrank_f32_ws, b_ptr, token_lora_ids, y_ptr, slices, max_slice_r,
            m, split_k, size_n, rank_pad, num_adapters, default_lora_id, adapter_mask,
            vec4_aligned, vec8_b_aligned);
    }
    if (vec8_b_aligned) {
        if (active_n_sum <= 128) {
            return launch_lora_expand<128, 16>(
                stream, sm_count, lowrank_f32_ws, b_ptr, token_lora_ids, y_ptr, slices,
                max_slice_r, m, split_k, size_n, rank_pad, num_adapters, default_lora_id,
                adapter_mask, vec4_aligned, true);
        }
        return launch_lora_expand<256, 16>(
            stream, sm_count, lowrank_f32_ws, b_ptr, token_lora_ids, y_ptr, slices, max_slice_r,
            m, split_k, size_n, rank_pad, num_adapters, default_lora_id, adapter_mask,
            vec4_aligned, true);
    }
    return launch_lora_expand<256, 32>(
        stream, sm_count, lowrank_f32_ws, b_ptr, token_lora_ids, y_ptr, slices, max_slice_r, m,
        split_k, size_n, rank_pad, num_adapters, default_lora_id, adapter_mask, vec4_aligned,
        false);
}
