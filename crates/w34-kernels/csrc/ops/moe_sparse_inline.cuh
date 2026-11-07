#pragma once

namespace w34_moe {

// Coalesced sub-warp uint4 expert-grouped kernel for M = 4..7 (BLOCK_M = 4):
// Eliminates 40x strided-row L1 MSHR pressure at very small M by assigning
// `lanes_per_row = min(32, K / 32)` contiguous uint4 weight loads per row
// (512 contiguous bytes = 4 full 128B cache lines per warp instruction for all
// K in {128, 256, 512, 1024, 2048}), with a 4-row register-cached `valid_cnt == 1`
// fast path and a low-register per-slot loop for `valid_cnt >= 2` to preserve SM occupancy.
static __host__ __device__ __forceinline__ int sparse_vec4_lpr(int k_dim) {
    const int k_u4 = k_dim >> 5;
    return (k_u4 >= 32) ? 32 : (k_u4 >= 16) ? 16 : (k_u4 >= 8) ? 8 : 4;
}

template <bool GATE, int GROUP_SIZE = 128>
__global__ __launch_bounds__(128, 4) void moe_sparse_vec4_kernel(
    const __half * __restrict__ x,
    const uint32_t * __restrict__ w,
    const __half * __restrict__ scales,
    const int32_t * __restrict__ pairs,
    const int32_t * __restrict__ experts,
    const float * __restrict__ tw,
    __half * __restrict__ intermediate,
    __half * __restrict__ down,
    int top_k, int N, int K) {
    static_assert(GROUP_SIZE == 64 || GROUP_SIZE == 128, "GROUP_SIZE must be 64 or 128");
    constexpr int W_COLS_PER_GROUP = GROUP_SIZE / 32;
    const int e = __ldg(&experts[blockIdx.y]);
    if (e < 0)
        return;

    int pair_v[4], xr_v[4];
    __half tw_v[4];
    int valid_cnt = 0;
#pragma unroll
    for (int m = 0; m < 4; ++m) {
        const int p = __ldg(&pairs[blockIdx.y * 4 + m]);
        pair_v[m] = p;
        xr_v[m] = GATE ? (p >= 0 ? p / top_k : -1) : p;
        tw_v[m] = (!GATE && p >= 0) ? __float2half_rn(__ldg(&tw[p])) : __float2half_rn(0.0f);
        if (p >= 0) ++valid_cnt;
    }
    if (valid_cnt == 0)
        return;

    const int tid = (int)threadIdx.x;
    const int wid = tid >> 5;
    const int lane = tid & 31;
    const int k_u4_per_row = K >> 5;
    const int lpr = sparse_vec4_lpr(K); // 4, 8, 16, or 32
    const int rpw = 32 / lpr;           // 8, 4, 2, or 1
    const int sub_lane = lane & (lpr - 1);
    const int sub_row = lane / lpr;
    const int wave_rows = 4 * rpw;
    const int base_row = (int)blockIdx.x * (4 * wave_rows) + wid * rpw + sub_row;
    if (base_row >= N)
        return;

    const int total_rows = GATE ? (N * 2) : N;
    const uint4 *w_u4 = reinterpret_cast<const uint4 *>(w);
    const uint4 *x_u4 = reinterpret_cast<const uint4 *>(x);
    const int x_u4_per_row = K >> 3;
    const int groups_per_row = K / GROUP_SIZE;

    if (valid_cnt == 1) {
        float g_r[4] = {};
        float u_r[4] = {};
        const size_t x0_base = (size_t)xr_v[0] * x_u4_per_row;
        for (int w_col = sub_lane; w_col < k_u4_per_row; w_col += lpr) {
            const int g_idx = w_col / W_COLS_PER_GROUP;
            const int x_col0 = w_col * 4;
            const uint4 xvs[4] = {
                __ldg(&x_u4[x0_base + x_col0]),
                __ldg(&x_u4[x0_base + x_col0 + 1]),
                __ldg(&x_u4[x0_base + x_col0 + 2]),
                __ldg(&x_u4[x0_base + x_col0 + 3]),
            };
#pragma unroll
            for (int ri = 0; ri < 4; ++ri) {
                const int row = base_row + ri * wave_rows;
                if (row < N) {
                    const size_t g_row_idx = (size_t)e * total_rows + row;
                    const uint4 gw4 = __ldg(&w_u4[g_row_idx * k_u4_per_row + w_col]);
                    const __half2 gs_h2 =
                        __half2half2(__ldg(&scales[g_row_idx * groups_per_row + g_idx]));
                    uint4 uw4;
                    __half2 us_h2;
                    if (GATE) {
                        const size_t u_row_idx = g_row_idx + N;
                        uw4 = __ldg(&w_u4[u_row_idx * k_u4_per_row + w_col]);
                        us_h2 = __half2half2(__ldg(&scales[u_row_idx * groups_per_row + g_idx]));
                    }
                    const uint32_t *gw_words = reinterpret_cast<const uint32_t *>(&gw4);
                    const uint32_t *uw_words = reinterpret_cast<const uint32_t *>(&uw4);
#pragma unroll
                    for (int word_i = 0; word_i < 4; ++word_i) {
                        const uint32_t gw_v = gw_words[word_i];
                        const uint32_t uw_v = GATE ? uw_words[word_i] : 0u;
                        const __half2 *xh2 = reinterpret_cast<const __half2 *>(&xvs[word_i]);
#pragma unroll
                        for (int p = 0; p < 4; ++p) {
                            const float x0 = __low2float(xh2[p]);
                            const float x1 = __high2float(xh2[p]);
                            const __half2 gd = unpack_u4b8_pair_scaled_h2(gw_v, p, gs_h2);
                            g_r[ri] += __low2float(gd) * x0 + __high2float(gd) * x1;
                            if (GATE) {
                                const __half2 ud = unpack_u4b8_pair_scaled_h2(uw_v, p, us_h2);
                                u_r[ri] += __low2float(ud) * x0 + __high2float(ud) * x1;
                            }
                        }
                    }
                }
            }
        }
        const int p0 = pair_v[0];
#pragma unroll
        for (int ri = 0; ri < 4; ++ri) {
            const int row = base_row + ri * wave_rows;
            float gv = g_r[ri];
            float uv = GATE ? u_r[ri] : 0.0f;
            if (lpr >= 32) {
                gv += __shfl_xor_sync(0xffffffffu, gv, 16);
                if (GATE) uv += __shfl_xor_sync(0xffffffffu, uv, 16);
            }
            if (lpr >= 16) {
                gv += __shfl_xor_sync(0xffffffffu, gv, 8);
                if (GATE) uv += __shfl_xor_sync(0xffffffffu, uv, 8);
            }
            if (lpr >= 8) {
                gv += __shfl_xor_sync(0xffffffffu, gv, 4);
                if (GATE) uv += __shfl_xor_sync(0xffffffffu, uv, 4);
            }
            gv += __shfl_xor_sync(0xffffffffu, gv, 2);
            gv += __shfl_xor_sync(0xffffffffu, gv, 1);
            if (GATE) {
                uv += __shfl_xor_sync(0xffffffffu, uv, 2);
                uv += __shfl_xor_sync(0xffffffffu, uv, 1);
            }
            if (sub_lane == 0 && row < N) {
                if (GATE) {
                    intermediate[(size_t)p0 * N + row] =
                        marlin_silu_and_mul_f16(__float2half_rn(gv), __float2half_rn(uv));
                } else {
                    down[(size_t)p0 * N + row] = __hmul(__float2half_rn(gv), tw_v[0]);
                }
            }
        }
        return;
    }

    for (int ri = 0; ri < 4; ++ri) {
        const int row = base_row + ri * wave_rows;
        if (row >= N)
            break;
        const size_t g_row_idx = (size_t)e * total_rows + row;
        const size_t u_row_idx = g_row_idx + N;

        for (int mb = 0; mb < valid_cnt; mb += 2) {
            const int p0 = (mb == 0) ? pair_v[0] : pair_v[2];
            const int p1 = (mb == 0) ? pair_v[1] : pair_v[3];
            const int xr0 = (mb == 0) ? xr_v[0] : xr_v[2];
            const int xr1 = (mb == 0) ? xr_v[1] : xr_v[3];
            const __half tw0 = (mb == 0) ? tw_v[0] : tw_v[2];
            const __half tw1 = (mb == 0) ? tw_v[1] : tw_v[3];
            const bool has1 = (mb + 1 < valid_cnt);
            const size_t xm0 = (size_t)xr0 * x_u4_per_row;
            const size_t xm1 = has1 ? ((size_t)xr1 * x_u4_per_row) : 0;

            float gv0 = 0.0f, uv0 = 0.0f;
            float gv1 = 0.0f, uv1 = 0.0f;

            for (int w_col = sub_lane; w_col < k_u4_per_row; w_col += lpr) {
                const int g_idx = w_col / W_COLS_PER_GROUP;
                const int x_col0 = w_col * 4;
                const uint4 gw4 = __ldg(&w_u4[g_row_idx * k_u4_per_row + w_col]);
                const __half2 gs_h2 =
                    __half2half2(__ldg(&scales[g_row_idx * groups_per_row + g_idx]));
                uint4 uw4;
                __half2 us_h2;
                if (GATE) {
                    uw4 = __ldg(&w_u4[u_row_idx * k_u4_per_row + w_col]);
                    us_h2 = __half2half2(__ldg(&scales[u_row_idx * groups_per_row + g_idx]));
                }
                const uint32_t *gw_words = reinterpret_cast<const uint32_t *>(&gw4);
                const uint32_t *uw_words = reinterpret_cast<const uint32_t *>(&uw4);

#pragma unroll
                for (int word_i = 0; word_i < 4; ++word_i) {
                    const uint32_t gw_v = gw_words[word_i];
                    const uint32_t uw_v = GATE ? uw_words[word_i] : 0u;
                    const uint4 xv0 = __ldg(&x_u4[xm0 + x_col0 + word_i]);
                    const uint4 xv1 = has1 ? __ldg(&x_u4[xm1 + x_col0 + word_i]) : make_uint4(0, 0, 0, 0);
                    const __half2 *xh0 = reinterpret_cast<const __half2 *>(&xv0);
                    const __half2 *xh1 = reinterpret_cast<const __half2 *>(&xv1);
#pragma unroll
                    for (int p = 0; p < 4; ++p) {
                        const __half2 gd = unpack_u4b8_pair_scaled_h2(gw_v, p, gs_h2);
                        const float gd_l = __low2float(gd), gd_h = __high2float(gd);
                        float ud_l = 0.0f, ud_h = 0.0f;
                        if (GATE) {
                            const __half2 ud = unpack_u4b8_pair_scaled_h2(uw_v, p, us_h2);
                            ud_l = __low2float(ud);
                            ud_h = __high2float(ud);
                        }
                        const float x0_l = __low2float(xh0[p]), x0_h = __high2float(xh0[p]);
                        gv0 += gd_l * x0_l + gd_h * x0_h;
                        if (GATE) uv0 += ud_l * x0_l + ud_h * x0_h;
                        if (has1) {
                            const float x1_l = __low2float(xh1[p]), x1_h = __high2float(xh1[p]);
                            gv1 += gd_l * x1_l + gd_h * x1_h;
                            if (GATE) uv1 += ud_l * x1_l + ud_h * x1_h;
                        }
                    }
                }
            }

            for (int off = lpr >> 1; off > 0; off >>= 1) {
                gv0 += __shfl_xor_sync(0xffffffffu, gv0, off);
                if (GATE) uv0 += __shfl_xor_sync(0xffffffffu, uv0, off);
                if (has1) {
                    gv1 += __shfl_xor_sync(0xffffffffu, gv1, off);
                    if (GATE) uv1 += __shfl_xor_sync(0xffffffffu, uv1, off);
                }
            }
            if (sub_lane == 0) {
                if (GATE) {
                    intermediate[(size_t)p0 * N + row] =
                        marlin_silu_and_mul_f16(__float2half_rn(gv0), __float2half_rn(uv0));
                    if (has1) {
                        intermediate[(size_t)p1 * N + row] =
                            marlin_silu_and_mul_f16(__float2half_rn(gv1), __float2half_rn(uv1));
                    }
                } else {
                    down[(size_t)p0 * N + row] = __hmul(__float2half_rn(gv0), tw0);
                    if (has1) {
                        down[(size_t)p1 * N + row] = __hmul(__float2half_rn(gv1), tw1);
                    }
                }
            }
        }
    }
}

// Sparse g128/g64 expert tiles (128 threads = 4 warps/CTA -> 32..48 warps/SM on sm_86):
// 16 output channels occupy MMA A, eight token slots occupy B (the small-M tile
// also selected by vLLM marlin_moe).
// - GATE=true  (small N, large K): 4 warps split K 4-way (16 rows/CTA).
// - GATE=false (large N, small K): 2 row-tiles x 2 K-split warps (32 rows/CTA).
// Vectorized uint4 weight loads are exchanged within four-lane row groups and
// pre-scaled in FP16 to match donor vLLM Marlin / moe_wna16 rounding.
// N is a multiple of 64 and K a multiple of 128 at the dispatch below.
template <bool GATE, int GROUP_SIZE = 128>
__global__ void moe_sparse_mma_kernel(
    const __half * __restrict__ x,
    const uint32_t * __restrict__ w,
    const __half * __restrict__ scales,
    const int32_t * __restrict__ pairs,
    const int32_t * __restrict__ experts,
    const float * __restrict__ tw,
    __half * __restrict__ intermediate,
    __half * __restrict__ down,
    int top_k, int N, int K) {
    static_assert(GROUP_SIZE == 64 || GROUP_SIZE == 128, "GROUP_SIZE must be 64 or 128");
    constexpr int TILE_K = 128;
    constexpr int SCALES_PER_TILE = TILE_K / GROUP_SIZE;
    const int e = __ldg(&experts[blockIdx.y]);
    if (e < 0)
        return;
    constexpr int K_WARPS = GATE ? 4 : 2;
    constexpr int ROW_TILES = 4 / K_WARPS;
    const int wid = threadIdx.x >> 5;
    const int tile_m = GATE ? 0 : (wid >> 1);
    const int warp_k = GATE ? wid : (wid & 1);
    const int lane = threadIdx.x & 31;
    const int r = lane >> 2, c = lane & 3;
    const int base = ((int)blockIdx.x * ROW_TILES + tile_m) * 16;
    const int nr = base + r;
    const int pair = __ldg(&pairs[blockIdx.y * 8 + r]);
    const int out_p0 = __ldg(&pairs[blockIdx.y * 8 + c * 2]);
    const int out_p1 = __ldg(&pairs[blockIdx.y * 8 + c * 2 + 1]);
    const int xr = GATE ? (pair >= 0 ? pair / top_k : -1) : pair;
    const int rows = GATE ? N * 2 : N;
    float acc[4] = {}, up[4] = {};
    for (int kg = warp_k * TILE_K; kg < K; kg += K_WARPS * TILE_K) {
        uint4 gw[2], uw[2];
        __half2 gs_h2[2][SCALES_PER_TILE], us_h2[2][SCALES_PER_TILE];
#pragma unroll
        for (int j = 0; j < 2; ++j) {
            const int row = nr + j * 8;
            gw[j] = __ldg(reinterpret_cast<const uint4 *>(w + ((size_t)e * rows + row) * (K / 8) +
                                                          kg / 8 + c * 4));
#pragma unroll
            for (int sg = 0; sg < SCALES_PER_TILE; ++sg) {
                gs_h2[j][sg] = __half2half2(
                    __ldg(&scales[((size_t)e * rows + row) * (K / GROUP_SIZE) + kg / GROUP_SIZE + sg]));
            }
            if (GATE) {
                uw[j] = __ldg(reinterpret_cast<const uint4 *>(
                    w + ((size_t)e * rows + N + row) * (K / 8) + kg / 8 + c * 4));
#pragma unroll
                for (int sg = 0; sg < SCALES_PER_TILE; ++sg) {
                    us_h2[j][sg] = __half2half2(__ldg(
                        &scales[((size_t)e * rows + N + row) * (K / GROUP_SIZE) + kg / GROUP_SIZE + sg]));
                }
            }
        }
#pragma unroll
        for (int kk = 0; kk < TILE_K; kk += 16) {
            const int k = kg + kk;
            const int sg = kk / GROUP_SIZE;
            uint32_t av[4] = {}, uv[4] = {};
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                const int word = kk / 8 + (j >> 1);
                const uint32_t gv = reinterpret_cast<uint32_t *>(&gw[j & 1])[word % 4];
                av[j] = unpack_u4b8_pair_scaled(
                    __shfl_sync(0xffffffffu, gv, (lane & ~3) + word / 4), c, gs_h2[j & 1][sg]);
                if (GATE) {
                    const uint32_t uvv = reinterpret_cast<uint32_t *>(&uw[j & 1])[word % 4];
                    uv[j] = unpack_u4b8_pair_scaled(
                        __shfl_sync(0xffffffffu, uvv, (lane & ~3) + word / 4), c, us_h2[j & 1][sg]);
                }
            }
            const uint32_t b0 =
                xr >= 0 ? __ldg(reinterpret_cast<const uint32_t *>(x + (size_t)xr * K + k + 2 * c)) : 0;
            const uint32_t b1 =
                xr >= 0 ? __ldg(reinterpret_cast<const uint32_t *>(x + (size_t)xr * K + k + 8 + 2 * c))
                        : 0;
            asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, "
                         "{%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
                         : "+f"(acc[0]), "+f"(acc[1]), "+f"(acc[2]), "+f"(acc[3])
                         : "r"(av[0]), "r"(av[1]), "r"(av[2]), "r"(av[3]), "r"(b0), "r"(b1));
            if (GATE)
                asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, "
                             "{%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
                             : "+f"(up[0]), "+f"(up[1]), "+f"(up[2]), "+f"(up[3])
                             : "r"(uv[0]), "r"(uv[1]), "r"(uv[2]), "r"(uv[3]), "r"(b0), "r"(b1));
        }
    }
    __shared__ float partial[4][GATE ? 8 : 4][32];
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        partial[wid][j][lane] = acc[j];
        if (GATE)
            partial[wid][j + 4][lane] = up[j];
    }
    __syncthreads();
    if (warp_k != 0)
        return;
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        if (GATE) {
            acc[j] = (partial[0][j][lane] + partial[1][j][lane]) +
                     (partial[2][j][lane] + partial[3][j][lane]);
            up[j] = (partial[0][j + 4][lane] + partial[1][j + 4][lane]) +
                    (partial[2][j + 4][lane] + partial[3][j + 4][lane]);
        } else {
            const int w0 = tile_m * 2;
            acc[j] = partial[w0][j][lane] + partial[w0 + 1][j][lane];
        }
    }
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        const int p = (j & 1) ? out_p1 : out_p0;
        const int row = nr + (j / 2) * 8;
        if (p >= 0) {
            if (GATE)
                intermediate[(size_t)p * N + row] =
                    marlin_silu_and_mul_f16(__float2half_rn(acc[j]), __float2half_rn(up[j]));
            else
                down[(size_t)p * N + row] =
                    __hmul(__float2half_rn(acc[j]), __float2half_rn(__ldg(&tw[p])));
        }
    }
}

} // namespace w34_moe
