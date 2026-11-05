// Gated DeltaNet (GDN) WY Chunked Prefill (BT=64) kernels for warp-34.
// Split from op_gdn.cu for modularity and parallel nvcc compilation.
#include "../w34_cuda.h"

#include <atomic>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <math.h>
#include <mma.h>

namespace w34_gdn {

namespace wmma = nvcuda::wmma;

__device__ __forceinline__ float silu_f32(float x) {
    return x / (1.0f + expf(-x));
}

__device__ __forceinline__ float softplus_f32(float x) {
    return (x > 20.0f) ? x : log1pf(expf(x));
}

__device__ __forceinline__ void gdn_norm_qk_warp(
    const __half* __restrict__ q_tok_hk,
    const __half* __restrict__ k_tok_hk,
    int lane,
    float scale,
    float4& bq4_out,
    float4& bk4_out) {
    const int c = lane << 2;
    float q0, q1, q2, q3;
    if ((((uintptr_t)(q_tok_hk + c)) & 7u) == 0u) {
        const uint2 qu = __ldg(reinterpret_cast<const uint2*>(q_tok_hk + c));
        const __half* qh = reinterpret_cast<const __half*>(&qu);
        q0 = __half2float(qh[0]);
        q1 = __half2float(qh[1]);
        q2 = __half2float(qh[2]);
        q3 = __half2float(qh[3]);
    } else {
        q0 = __half2float(q_tok_hk[c + 0]);
        q1 = __half2float(q_tok_hk[c + 1]);
        q2 = __half2float(q_tok_hk[c + 2]);
        q3 = __half2float(q_tok_hk[c + 3]);
    }
    float k0, k1, k2, k3;
    if ((((uintptr_t)(k_tok_hk + c)) & 7u) == 0u) {
        const uint2 ku = __ldg(reinterpret_cast<const uint2*>(k_tok_hk + c));
        const __half* kh = reinterpret_cast<const __half*>(&ku);
        k0 = __half2float(kh[0]);
        k1 = __half2float(kh[1]);
        k2 = __half2float(kh[2]);
        k3 = __half2float(kh[3]);
    } else {
        k0 = __half2float(k_tok_hk[c + 0]);
        k1 = __half2float(k_tok_hk[c + 1]);
        k2 = __half2float(k_tok_hk[c + 2]);
        k3 = __half2float(k_tok_hk[c + 3]);
    }
    float q_sq = __fadd_rn(__fmaf_rn(q0, q0, __fmul_rn(q1, q1)),
                           __fmaf_rn(q2, q2, __fmul_rn(q3, q3)));
    float k_sq = __fadd_rn(__fmaf_rn(k0, k0, __fmul_rn(k1, k1)),
                           __fmaf_rn(k2, k2, __fmul_rn(k3, k3)));
#pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
        q_sq = __fadd_rn(q_sq, __shfl_xor_sync(0xffffffffu, q_sq, off));
        k_sq = __fadd_rn(k_sq, __shfl_xor_sync(0xffffffffu, k_sq, off));
    }
    const float q_inv = __fmul_rn(rsqrtf(__fadd_rn(q_sq, 1e-6f)), scale);
    const float k_inv = rsqrtf(__fadd_rn(k_sq, 1e-6f));
    bq4_out = make_float4(__fmul_rn(q0, q_inv), __fmul_rn(q1, q_inv),
                          __fmul_rn(q2, q_inv), __fmul_rn(q3, q_inv));
    bk4_out = make_float4(__fmul_rn(k0, k_inv), __fmul_rn(k1, k_inv),
                          __fmul_rn(k2, k_inv), __fmul_rn(k3, k_inv));
}

__device__ __forceinline__ void gdn_norm_single_warp(
    const __half* __restrict__ x_tok,
    int lane,
    float out_scale,
    float4& bx4) {
    const int k_base = lane * 4;
    const uint2 xu = *reinterpret_cast<const uint2*>(x_tok + k_base);
    const __half2 x01 = *reinterpret_cast<const __half2*>(&xu.x);
    const __half2 x23 = *reinterpret_cast<const __half2*>(&xu.y);
    const float2 xf01 = __half22float2(x01);
    const float2 xf23 = __half22float2(x23);
    float sq = __fmaf_rn(xf01.x, xf01.x, __fmaf_rn(xf01.y, xf01.y,
               __fmaf_rn(xf23.x, xf23.x, __fmul_rn(xf23.y, xf23.y))));
#pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
        sq = __fadd_rn(sq, __shfl_xor_sync(0xffffffffu, sq, off));
    }
    const float inv = __fmul_rn(rsqrtf(__fadd_rn(sq, 1e-6f)), out_scale);
    bx4 = make_float4(__fmul_rn(xf01.x, inv), __fmul_rn(xf01.y, inv),
                      __fmul_rn(xf23.x, inv), __fmul_rn(xf23.y, inv));
}

struct WyPlanHeader {
    int32_t total_chunks;
    int32_t overflow;
    int32_t pad[2];
};

struct WyReqSpan {
    int32_t first_chunk;
    int32_t num_chunks;
};

struct WyChunkDesc {
    int32_t req;
    int32_t tok0;
    int32_t len;
    int32_t ckpt_slot;
    int32_t h_slot;
    int32_t pad[3];
};

__global__ void gdn_wy_plan_chunks_kernel(
    w34_gdn_batch batch,
    WyPlanHeader* __restrict__ hdr,
    WyReqSpan* __restrict__ req_spans,
    WyChunkDesc* __restrict__ chunks,
    int max_chunks_cap,
    int max_h_slots,
    int total_state_slots) {
    if (threadIdx.x != 0 || blockIdx.x != 0) return;
    const bool has_checkpoints =
        (batch.checkpoint_dst_indices != nullptr && batch.checkpoint_token_offsets != nullptr &&
         batch.max_checkpoints_per_req > 0);
    int32_t gc = 0;
    int32_t next_h_slot = 0;
    for (int req = 0; req < batch.num_reqs; ++req) {
        if (batch.valid_req_mask != nullptr && batch.valid_req_mask[req] == 0) {
            req_spans[req] = {gc, 0};
            continue;
        }
        const int lo = batch.query_start_loc[req];
        const int hi = batch.query_start_loc[req + 1];
        const int seqlen = hi - lo;
        if (seqlen <= 0) {
            req_spans[req] = {gc, 0};
            continue;
        }
        const bool has_init =
            (batch.has_initial_state == nullptr) || (batch.has_initial_state[req] != 0);
        const int32_t slot_in =
            (batch.state_in_indices != nullptr) ? batch.state_in_indices[req] : -1;
        const bool req_has_init_state =
            has_init && slot_in >= 0 && slot_in < total_state_slots;

        const int32_t first_c = gc;
        int base_s = 0;
        int local_c = 0;
        while (base_s < seqlen) {
            if (gc >= max_chunks_cap) {
                for (int r = 0; r < batch.num_reqs; ++r) req_spans[r] = {0, 0};
                hdr->total_chunks = 0;
                hdr->overflow = 1;
                return;
            }
            int tile_len = (seqlen - base_s < 64) ? (seqlen - base_s) : 64;
            int32_t hit_ckpt_slot = -1;
            if (has_checkpoints) {
                for (int e = 0; e < batch.max_checkpoints_per_req; ++e) {
                    const int32_t trig_off =
                        batch.checkpoint_token_offsets[req * batch.max_checkpoints_per_req + e];
                    if (trig_off > base_s && trig_off <= base_s + tile_len) {
                        const int cand_len = trig_off - base_s;
                        if (cand_len < tile_len) tile_len = cand_len;
                    }
                }
                for (int e = 0; e < batch.max_checkpoints_per_req; ++e) {
                    const int32_t trig_off =
                        batch.checkpoint_token_offsets[req * batch.max_checkpoints_per_req + e];
                    if (trig_off == base_s + tile_len) {
                        hit_ckpt_slot =
                            batch.checkpoint_dst_indices[req * batch.max_checkpoints_per_req + e];
                        break;
                    }
                }
            }
            int32_t h_slot = -1;
            if (local_c > 0 || req_has_init_state) {
                if (next_h_slot >= max_h_slots) {
                    for (int r = 0; r < batch.num_reqs; ++r) req_spans[r] = {0, 0};
                    hdr->total_chunks = 0;
                    hdr->overflow = 1;
                    return;
                }
                h_slot = next_h_slot++;
            }
            WyChunkDesc cd;
            cd.req = req;
            cd.tok0 = lo + base_s;
            cd.len = tile_len;
            cd.ckpt_slot = hit_ckpt_slot;
            cd.h_slot = h_slot;
            cd.pad[0] = 0;
            cd.pad[1] = 0;
            cd.pad[2] = 0;
            chunks[gc++] = cd;
            base_s += tile_len;
            ++local_c;
        }
        req_spans[req] = {first_c, gc - first_c};
    }
    hdr->total_chunks = gc;
    hdr->overflow = 0;
}

// Strict-upper staging slot (blocks (0,1), (0,2), (0,3)) of the padded [64, 68] FP32 s_T tile.
__device__ __forceinline__ float& wy_upper_pw_slot_padded(float* s_T, int idx) {
    const int blk = idx >> 8;          // 0..2 -> blocks (0,1), (0,2), (0,3)
    const int r = (idx >> 4) & 15;     // row 0..15
    const int c = idx & 15;            // col within block 0..15
    return s_T[r * 68 + 16 + blk * 16 + c];
}

static constexpr int kWyStage1SmemBytes = 36608;
static constexpr int kWyStage2SmemBytes = 43520;
static constexpr int kWyStage3SmemBytes = 46592;

// Stage 1 (T-parallel across ALL 64-token chunks x V-heads, 2 CTAs/SM co-resident):
// Computes K_norm (optionally cached to k_norm_out), cumulative log-decay g_cum,
// Gram = K @ K^T, T = (I + A)^{-1}, U = T @ (beta * V) -> u_out, W = T @ (beta * exp(g) * K) -> w_out.
__global__ __launch_bounds__(256, 2) void gdn_wy_stage1_wu_f16_kernel(
    const WyPlanHeader* __restrict__ hdr,
    const WyChunkDesc* __restrict__ chunks,
    const __half* __restrict__ qkv_conv_out,
    const __half* __restrict__ k_in,
    const __half* __restrict__ v_in,
    const __half* __restrict__ b_in,
    const __half* __restrict__ a_in,
    const float* __restrict__ a_log,
    const float* __restrict__ dt_bias,
    float* __restrict__ g_cum_out,
    __half* __restrict__ k_norm_out,
    __half* __restrict__ w_out,
    __half* __restrict__ u_out,
    int num_k_heads,
    int num_v_heads,
    int ba_row_stride) {
#if __CUDA_ARCH__ >= 800
    constexpr int BT = 64;
    constexpr int DK = 128;
    const int gc = blockIdx.x;
    const int hv = blockIdx.y;
    if (gc >= hdr->total_chunks) return;

    const WyChunkDesc cd = chunks[gc];
    const int tok0 = cd.tok0;
    const int tile_len = cd.len;
    const int gqa_ratio = num_v_heads / num_k_heads;
    const int hk = hv / gqa_ratio;
    const int tid = threadIdx.x;
    const int wid = tid >> 5;
    const int lane = tid & 31;

    constexpr int DK_PAD = 136;   // 272 B __half rows
    constexpr int AT_PAD = 68;    // 272 B float rows
    constexpr int BT_PAD16 = 72;  // 144 B __half rows

    // 36,608 B total dynamic smem -> 2 CTAs/SM co-resident. Row strides carry +16 B padding to
    // break the power-of-two bank conflicts of WMMA fragment loads/stores; K_norm is stashed in
    // 8 uint2 registers/thread after the Gram product so s_T can alias s_K:
    // [0 .. 17408): union { s_K [64, 136] __half, s_T [64, 68] float }
    // [17408 .. 35840): union { s_A [64, 68] float,
    //                           s_Bop [64, 72] __half (9,216 B) + s_T_f16 [64, 72] __half }
    // [35840 .. 36608): s_g [64] + s_eg [64] + s_beta [64] float
    extern __shared__ __align__(16) char smem_raw[];
    __half* s_K = reinterpret_cast<__half*>(smem_raw);
    float* s_T = reinterpret_cast<float*>(smem_raw);
    float* s_A = reinterpret_cast<float*>(smem_raw + 17408);
    float* s_g = reinterpret_cast<float*>(smem_raw + 35840);
    float* s_eg = s_g + 64;
    float* s_beta = s_g + 128;

    using Acc = wmma::fragment<wmma::accumulator, 16, 16, 16, float>;
    using FragA_Row = wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, wmma::row_major>;
    using FragB_Row = wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, wmma::row_major>;
    using FragB_Col = wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, wmma::col_major>;

    const float neg_exp_a = -expf(a_log[hv]);
    const float dt_bias_hv = dt_bias[hv];
    const int k_total_dim = num_k_heads * DK;
    const int v_total_dim = num_v_heads * DK;
    const int qkv_stride = 2 * k_total_dim + v_total_dim;

    for (int s_norm = wid; s_norm < tile_len; s_norm += 8) {
        const int tok = tok0 + s_norm;
        const __half* k_tok_hk = (qkv_conv_out != nullptr)
            ? (qkv_conv_out + (size_t)tok * qkv_stride + k_total_dim + (size_t)hk * DK)
            : (k_in + ((size_t)tok * num_k_heads + hk) * DK);
        float4 bk4;
        gdn_norm_single_warp(k_tok_hk, lane, 1.0f, bk4);
        const __half2 kh01 = __floats2half2_rn(bk4.x, bk4.y);
        const __half2 kh23 = __floats2half2_rn(bk4.z, bk4.w);
        __half2* k_dst2 = reinterpret_cast<__half2*>(s_K + s_norm * DK_PAD + lane * 4);
        k_dst2[0] = kh01;
        k_dst2[1] = kh23;
        if (k_norm_out != nullptr && (hv == hk * gqa_ratio) && hk < num_k_heads) {
            __half2* kn_g2 = reinterpret_cast<__half2*>(
                k_norm_out + ((size_t)tok * num_k_heads + hk) * DK + lane * 4);
            kn_g2[0] = kh01;
            kn_g2[1] = kh23;
        }
        if (lane == 0) {
            const float a_val = __half2float(a_in[(size_t)tok * ba_row_stride + hv]);
            const float b_val = __half2float(b_in[(size_t)tok * ba_row_stride + hv]);
            s_g[s_norm] = neg_exp_a * softplus_f32(a_val + dt_bias_hv);
            s_beta[s_norm] = 1.0f / (1.0f + expf(-b_val));
        }
    }
    for (int idx = tile_len * DK_PAD + tid; idx < BT * DK_PAD; idx += 256) {
        s_K[idx] = __float2half(0.0f);
    }
    if (tid >= tile_len && tid < BT) {
        s_g[tid] = 0.0f;
        s_eg[tid] = 1.0f;
        s_beta[tid] = 0.0f;
    }
    __syncthreads();
    if (wid == 0) {
        float v0 = (lane < tile_len) ? s_g[lane] : 0.0f;
        float v1 = (32 + lane < tile_len) ? s_g[32 + lane] : 0.0f;
#pragma unroll
        for (int off = 1; off < 32; off <<= 1) {
            const float n0 = __shfl_up_sync(0xffffffffu, v0, off);
            const float n1 = __shfl_up_sync(0xffffffffu, v1, off);
            if (lane >= off) {
                v0 += n0;
                v1 += n1;
            }
        }
        v1 += __shfl_sync(0xffffffffu, v0, 31);
        if (lane < tile_len) {
            s_g[lane] = v0;
            s_eg[lane] = expf(v0);
            g_cum_out[(size_t)(tok0 + lane) * num_v_heads + hv] = v0;
        }
        if (32 + lane < tile_len) {
            s_g[32 + lane] = v1;
            s_eg[32 + lane] = expf(v1);
            g_cum_out[(size_t)(tok0 + 32 + lane) * num_v_heads + hv] = v1;
        }
    }
    __syncthreads();

#pragma unroll
    for (int step_t = 0; step_t < 2; ++step_t) {
        const int t_id = wid + step_t * 8;
        const int it = t_id & 3;
        const int jt = t_id >> 2;
        Acc acc;
        wmma::fill_fragment(acc, 0.0f);
#pragma unroll
        for (int d = 0; d < DK; d += 16) {
            FragA_Row a;
            FragB_Col b;
            wmma::load_matrix_sync(a, s_K + (it * 16) * DK_PAD + d, DK_PAD);
            wmma::load_matrix_sync(b, s_K + (jt * 16) * DK_PAD + d, DK_PAD);
            wmma::mma_sync(acc, a, b, acc);
        }
        wmma::store_matrix_sync(s_A + (it * 16) * AT_PAD + (jt * 16), acc, AT_PAD, wmma::mem_row_major);
    }
    // Stash K_norm rows (wid + 8 * iter) as half2 pairs at columns {2 * lane, 64 + 2 * lane}.
    uint2 k_reg[8];
#pragma unroll
    for (int iter = 0; iter < 8; ++iter) {
        const int i = wid + iter * 8;
        k_reg[iter] = make_uint2(
            *reinterpret_cast<const uint32_t*>(s_K + i * DK_PAD + lane * 2),
            *reinterpret_cast<const uint32_t*>(s_K + i * DK_PAD + 64 + lane * 2));
    }
    __syncthreads();  // s_K is dead; s_T may now overwrite it.

    for (int idx = tid; idx < BT * AT_PAD; idx += 256) {
        s_T[idx] = 0.0f;
    }
    for (int idx = tid; idx < BT * BT; idx += 256) {
        const int i = idx >> 6;
        const int j = idx & 63;
        float& cell = s_A[i * AT_PAD + j];
        cell = (i < tile_len && j < i)
            ? (s_beta[i] * expf(s_g[i] - s_g[j]) * cell)
            : 0.0f;
    }
    __syncthreads();

    if (tid < 64) {
        const int bd = tid >> 4;
        const int base = bd * 16;
        const int c = base + (tid & 15);
        s_T[c * AT_PAD + c] = 1.0f;
        if (c < tile_len) {
            const int iend = (base + 16 < tile_len) ? (base + 16) : tile_len;
            for (int i = c + 1; i < iend; ++i) {
                float sfs = 0.0f;
                for (int m = c; m < i; ++m) {
                    sfs += s_A[i * AT_PAD + m] * s_T[m * AT_PAD + c];
                }
                s_T[i * AT_PAD + c] = -sfs;
            }
        }
    }
    __syncthreads();

    // Distance 1 Schur merges staged in strict-upper blocks (0,1..3) of s_T.
    for (int idx = tid; idx < 768; idx += 256) {
        const int blk = idx >> 8;
        const int bi = blk + 1;
        const int bj = blk;
        const int r = (idx >> 4) & 15;
        const int c = idx & 15;
        float sum = 0.0f;
#pragma unroll
        for (int m = 0; m < 16; ++m) {
            sum += s_A[(bi * 16 + r) * AT_PAD + (bj * 16 + m)] * s_T[(bj * 16 + m) * AT_PAD + (bj * 16 + c)];
        }
        wy_upper_pw_slot_padded(s_T, idx) = sum;
    }
    __syncthreads();
    for (int idx = tid; idx < 768; idx += 256) {
        const int blk = idx >> 8;
        const int bi = blk + 1;
        const int bj = blk;
        const int r = (idx >> 4) & 15;
        const int c = idx & 15;
        float sum = 0.0f;
#pragma unroll
        for (int m = 0; m < 16; ++m) {
            sum += s_T[(bi * 16 + r) * AT_PAD + (bi * 16 + m)] * wy_upper_pw_slot_padded(s_T, (blk << 8) + (m << 4) + c);
        }
        s_T[(bi * 16 + r) * AT_PAD + (bj * 16 + c)] = -sum;
    }
    __syncthreads();

    // Distance 2 Schur merges.
    for (int idx = tid; idx < 512; idx += 256) {
        const int blk = idx >> 8;
        const int bi = blk + 2;
        const int bj = blk;
        const int r = (idx >> 4) & 15;
        const int c = idx & 15;
        float sum = 0.0f;
#pragma unroll
        for (int bk = 0; bk < 2; ++bk) {
            const int kblk = bj + bk;
#pragma unroll
            for (int m = 0; m < 16; ++m) {
                sum += s_A[(bi * 16 + r) * AT_PAD + (kblk * 16 + m)] *
                       s_T[(kblk * 16 + m) * AT_PAD + (bj * 16 + c)];
            }
        }
        wy_upper_pw_slot_padded(s_T, idx) = sum;
    }
    __syncthreads();
    for (int idx = tid; idx < 512; idx += 256) {
        const int blk = idx >> 8;
        const int bi = blk + 2;
        const int bj = blk;
        const int r = (idx >> 4) & 15;
        const int c = idx & 15;
        float sum = 0.0f;
#pragma unroll
        for (int m = 0; m < 16; ++m) {
            sum += s_T[(bi * 16 + r) * AT_PAD + (bi * 16 + m)] * wy_upper_pw_slot_padded(s_T, (blk << 8) + (m << 4) + c);
        }
        s_T[(bi * 16 + r) * AT_PAD + (bj * 16 + c)] = -sum;
    }
    __syncthreads();

    // Distance 3 Schur merge (3, 0).
    {
        const int r = tid >> 4;
        const int c = tid & 15;
        float sum = 0.0f;
#pragma unroll
        for (int kblk = 0; kblk < 3; ++kblk) {
#pragma unroll
            for (int m = 0; m < 16; ++m) {
                sum += s_A[(48 + r) * AT_PAD + (kblk * 16 + m)] * s_T[(kblk * 16 + m) * AT_PAD + c];
            }
        }
        wy_upper_pw_slot_padded(s_T, tid) = sum;
    }
    __syncthreads();
    {
        const int r = tid >> 4;
        const int c = tid & 15;
        float sum = 0.0f;
#pragma unroll
        for (int m = 0; m < 16; ++m) {
            sum += s_T[(48 + r) * AT_PAD + (48 + m)] * wy_upper_pw_slot_padded(s_T, (m << 4) + c);
        }
        s_T[(48 + r) * AT_PAD + c] = -sum;
    }
    __syncthreads();
    for (int idx = tid; idx < 768; idx += 256) {
        wy_upper_pw_slot_padded(s_T, idx) = 0.0f;
    }
    __syncthreads();

    __half* s_Bop = reinterpret_cast<__half*>(s_A);
    __half* s_T_f16 = s_Bop + (BT * BT_PAD16);
    for (int idx = tid; idx < BT * BT; idx += 256) {
        const int r = idx >> 6;
        const int c = idx & 63;
        s_T_f16[r * BT_PAD16 + c] = __float2half(s_T[r * AT_PAD + c]);
    }
    __syncthreads();

    // Apply U = s_T_f16 @ (beta * V) -> u_out, then W = s_T_f16 @ (beta * exp(g) * K) -> w_out.
    for (int pass = 0; pass < 2; ++pass) {
        __half* dst_global = (pass == 0) ? u_out : w_out;
#pragma unroll
        for (int cb_idx = 0; cb_idx < 2; ++cb_idx) {
            const int cb = cb_idx * 64;
            // Row i = wid + 8 * iter, half2 column `lane` (matches the k_reg stash layout).
#pragma unroll
            for (int iter = 0; iter < 8; ++iter) {
                const int i = wid + iter * 8;
                __half2 val2 = __floats2half2_rn(0.0f, 0.0f);
                if (i < tile_len) {
                    if (pass == 0) {
                        const int tok = tok0 + i;
                        const __half* v_tok_hv = (qkv_conv_out != nullptr)
                            ? (qkv_conv_out + (size_t)tok * qkv_stride + 2 * k_total_dim + (size_t)hv * DK)
                            : (v_in + ((size_t)tok * num_v_heads + hv) * DK);
                        const float2 vf =
                            __half22float2(reinterpret_cast<const __half2*>(v_tok_hv + cb)[lane]);
                        const float sc = s_beta[i];
                        val2 = __floats2half2_rn(sc * vf.x, sc * vf.y);
                    } else {
                        const uint32_t raw = (cb_idx == 0) ? k_reg[iter].x : k_reg[iter].y;
                        const float2 kf = __half22float2(*reinterpret_cast<const __half2*>(&raw));
                        const float sc = s_beta[i] * s_eg[i];
                        val2 = __floats2half2_rn(sc * kf.x, sc * kf.y);
                    }
                }
                reinterpret_cast<__half2*>(s_Bop + i * BT_PAD16)[lane] = val2;
            }
            __syncthreads();
#pragma unroll
            for (int step_t = 0; step_t < 2; ++step_t) {
                const int t_id = wid + step_t * 8;
                const int it = t_id & 3;
                const int ct = t_id >> 2;
                Acc acc;
                wmma::fill_fragment(acc, 0.0f);
#pragma unroll
                for (int jj = 0; jj < BT; jj += 16) {
                    FragA_Row a;
                    FragB_Row b;
                    wmma::load_matrix_sync(a, s_T_f16 + (it * 16) * BT_PAD16 + jj, BT_PAD16);
                    wmma::load_matrix_sync(b, s_Bop + jj * BT_PAD16 + (ct * 16), BT_PAD16);
                    wmma::mma_sync(acc, a, b, acc);
                }
                wmma::store_matrix_sync(s_T + (it * 16) * AT_PAD + (ct * 16), acc, AT_PAD, wmma::mem_row_major);
            }
            __syncthreads();
            for (int idx2 = tid; idx2 < (tile_len * 64) / 2; idx2 += 256) {
                const int i = idx2 >> 5;
                const int c2 = idx2 & 31;
                reinterpret_cast<__half2*>(
                    dst_global + ((size_t)(tok0 + i) * num_v_heads + hv) * DK + cb)[c2] =
                    __floats2half2_rn(s_T[i * AT_PAD + c2 * 2 + 0], s_T[i * AT_PAD + c2 * 2 + 1]);
            }
            __syncthreads();
        }
    }
#else
    (void)hdr; (void)chunks; (void)qkv_conv_out; (void)k_in; (void)v_in;
    (void)b_in; (void)a_in; (void)a_log; (void)dt_bias; (void)g_cum_out; (void)k_norm_out;
    (void)w_out; (void)u_out; (void)num_k_heads; (void)num_v_heads; (void)ba_row_stride;
#endif
}

// Stage 2 (BV = 32, 256 threads = 8 warps -> 4 CTAs/V-head = 128 CTAs on 9B / 96 on 27B TP=2,
// 2 CTAs/SM co-resident):
// Saves per-chunk starting H to h_chunks, computes V_new = U - W @ H in-place in u_vnew_inout,
// updates H = exp(g_last) * H + K_scaled^T @ V_new, and flushes checkpoints / final ssm_state
// via bank-conflict-padded s_H_f32 [128, 36].
__global__ __launch_bounds__(256, 2) void gdn_wy_stage2_deltah_f16_kernel(
    w34_gdn_batch batch,
    const WyReqSpan* __restrict__ req_spans,
    const WyChunkDesc* __restrict__ chunks,
    const __half* __restrict__ qkv_conv_out,
    const __half* __restrict__ k_in,
    const __half* __restrict__ k_norm_in,
    const float* __restrict__ g_cum,
    const __half* __restrict__ w_in,
    __half* __restrict__ u_vnew_inout,
    __half* __restrict__ h_chunks_out,
    float* __restrict__ ssm_state_pool,
    int num_k_heads,
    int num_v_heads,
    int total_state_slots) {
#if __CUDA_ARCH__ >= 800
    constexpr int BT = 64;
    constexpr int BV = 32;
    constexpr int BV_PAD = 36;  // 16B-aligned stride (144 B) with gcd(36, 32) = 4 on transpose!
    constexpr int DK = 128;
    constexpr int DK_PAD = 136; // 16B-aligned stride (272 B) reduces 16-way bank conflict to 2-way!

    const int i_v = blockIdx.x;  // 0..3
    const int i_nh = blockIdx.y;
    const int req = i_nh / num_v_heads;
    const int hv = i_nh % num_v_heads;
    if (req >= batch.num_reqs) return;
    if (batch.valid_req_mask != nullptr && batch.valid_req_mask[req] == 0) return;
    const WyReqSpan span = req_spans[req];
    if (span.num_chunks <= 0) return;

    const int gqa_ratio = num_v_heads / num_k_heads;
    const int hk = hv / gqa_ratio;
    const int tid = threadIdx.x;
    const int wid = tid >> 5;
    const int lane = tid & 31;
    const int it = wid >> 1;       // 0..3
    const int vc_tile = wid & 1;   // 0..1
    const int v_off = vc_tile * 16;
    const int vbase = i_v * BV;
    const size_t head_state_elems = (size_t)DK * (size_t)DK;
    const int k_total_dim = num_k_heads * DK;
    const int v_total_dim = num_v_heads * DK;
    const int qkv_stride = 2 * k_total_dim + v_total_dim;

    constexpr int BH_PAD = 40;  // 80 B half rows for s_H / s_Vnew (breaks 64 B-stride conflicts)

    // 43,520 B total dynamic smem (<= 48 KiB default opt-in-free limit -> 2 CTAs/SM co-resident):
    // [0 .. 18432): union { s_H_f32 [128, 36] (18,432 B), s_W [64, 136] (17,408 B),
    //                       s_K [64, 136] (17,408 B) }
    // [18432 .. 23552): s_Vnew [64, 40] __half (5,120 B)
    // [23552 .. 33792): s_H [128, 40] __half (10,240 B)
    // [33792 .. 34048): s_g [64] float (256 B)
    // [34048 .. 34304): s_sc [64] float (256 B) = exp(g_last - g_i), computed once per row
    // [34304 .. 43520): s_wh_full [64, 36] float (9,216 B), disjoint from s_W/s_K so W@H store
    //                   and V_new + K_scaled staging fuse across two fewer block barriers/chunk
    extern __shared__ __align__(16) char smem_raw[];
    float* s_H_f32 = reinterpret_cast<float*>(smem_raw);
    __half* s_W = reinterpret_cast<__half*>(smem_raw);
    __half* s_K = reinterpret_cast<__half*>(smem_raw);
    __half* s_Vnew = reinterpret_cast<__half*>(smem_raw + 18432);
    __half* s_H = reinterpret_cast<__half*>(smem_raw + 23552);
    float* s_g = reinterpret_cast<float*>(smem_raw + 33792);
    float* s_sc = s_g + 64;
    float* s_wh_full = reinterpret_cast<float*>(smem_raw + 34304);

    using Acc = wmma::fragment<wmma::accumulator, 16, 16, 16, float>;
    using FragA_Row = wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, wmma::row_major>;
    using FragA_Col = wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, wmma::col_major>;
    using FragB_Row = wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, wmma::row_major>;

    Acc h_acc[2];
    const bool has_init = (batch.has_initial_state == nullptr) || (batch.has_initial_state[req] != 0);
    const int32_t slot_in = (batch.state_in_indices != nullptr) ? batch.state_in_indices[req] : -1;
    if (has_init && slot_in >= 0 && slot_in < total_state_slots) {
        const float* head_in = ssm_state_pool + ((size_t)slot_in * num_v_heads + hv) * head_state_elems;
        for (int idx = tid; idx < BV * DK; idx += 256) {
            const int vc = idx >> 7;
            const int k = idx & 127;
            const float val = head_in[(size_t)(vbase + vc) * DK + k];
            s_H_f32[k * BV_PAD + vc] = val;
            s_H[k * BH_PAD + vc] = __float2half(val);
        }
        __syncthreads();
#pragma unroll
        for (int kt = 0; kt < 2; ++kt) {
            wmma::load_matrix_sync(
                h_acc[kt],
                s_H_f32 + ((it * 2 + kt) * 16) * BV_PAD + v_off,
                BV_PAD,
                wmma::mem_row_major);
        }
        __syncthreads();
    } else {
        for (int idx = tid; idx < DK * BH_PAD; idx += 256) {
            s_H[idx] = __float2half(0.0f);
        }
#pragma unroll
        for (int kt = 0; kt < 2; ++kt) {
            wmma::fill_fragment(h_acc[kt], 0.0f);
        }
        __syncthreads();
    }

    for (int c_local = 0; c_local < span.num_chunks; ++c_local) {
        const WyChunkDesc cd = chunks[span.first_chunk + c_local];
        const int tok0 = cd.tok0;
        const int tile_len = cd.len;
        if (cd.h_slot >= 0) {
            __half* h_dst = h_chunks_out + ((size_t)cd.h_slot * num_v_heads + hv) * head_state_elems;
            for (int idx4 = tid; idx4 < (DK * BV) / 4; idx4 += 256) {
                const int k = idx4 >> 3;
                const int vc4 = idx4 & 7;
                reinterpret_cast<uint2*>(h_dst + (size_t)k * DK + vbase)[vc4] =
                    reinterpret_cast<const uint2*>(s_H + k * BH_PAD)[vc4];
            }
        }

        // Prefetch this thread's 4 U half2 values so their latency overlaps the W load + W @ H.
        __half2 u_pre[4];
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            const int idx2 = tid + j * 256;
            const int row = idx2 >> 4;
            const int col2 = idx2 & 15;
            u_pre[j] = (row < tile_len)
                ? reinterpret_cast<const __half2*>(
                      u_vnew_inout + ((size_t)(tok0 + row) * num_v_heads + hv) * DK + vbase)[col2]
                : __floats2half2_rn(0.0f, 0.0f);
        }

        // Vectorized 64-bit (half4) load of s_W [64, 128].
        for (int idx4 = tid; idx4 < (BT * DK) / 4; idx4 += 256) {
            const int i = idx4 >> 5;
            const int d4 = idx4 & 31;
            uint2 v = make_uint2(0u, 0u);
            if (i < tile_len) {
                v = reinterpret_cast<const uint2*>(
                    w_in + ((size_t)(tok0 + i) * num_v_heads + hv) * DK)[d4];
            }
            reinterpret_cast<uint2*>(s_W + i * DK_PAD)[d4] = v;
        }
        if (tid < BT) {
            const float gv =
                (tid < tile_len) ? g_cum[(size_t)(tok0 + tid) * num_v_heads + hv] : 0.0f;
            const float gl = g_cum[(size_t)(tok0 + tile_len - 1) * num_v_heads + hv];
            s_g[tid] = gv;
            s_sc[tid] = (tid < tile_len) ? expf(gl - gv) : 0.0f;
        }
        __syncthreads();

        // WH [64, 32] = s_W [64, 128] @ s_H [128, 32] (1 single 16x16 tile per warp across 8 warps).
        Acc wh_acc;
        wmma::fill_fragment(wh_acc, 0.0f);
#pragma unroll
        for (int d = 0; d < DK; d += 16) {
            FragA_Row a;
            FragB_Row b;
            wmma::load_matrix_sync(a, s_W + (it * 16) * DK_PAD + d, DK_PAD);
            wmma::load_matrix_sync(b, s_H + d * BH_PAD + v_off, BH_PAD);
            wmma::mma_sync(wh_acc, a, b, wh_acc);
        }
        wmma::store_matrix_sync(
            s_wh_full + (it * 16) * BV_PAD + v_off, wh_acc, BV_PAD, wmma::mem_row_major);
        // Prefetch raw K_norm (8 uint2/thread) while W@H stores retire before the single barrier.
        uint2 kn_pre[8];
        if (k_norm_in != nullptr) {
#pragma unroll
            for (int j = 0; j < 8; ++j) {
                const int idx4 = tid + j * 256;
                const int s_norm = idx4 >> 5;
                const int d4 = idx4 & 31;
                kn_pre[j] = (s_norm < tile_len)
                    ? reinterpret_cast<const uint2*>(
                          k_norm_in + ((size_t)(tok0 + s_norm) * num_k_heads + hk) * DK)[d4]
                    : make_uint2(0u, 0u);
            }
        }
        __syncthreads();  // s_wh_full is ready and s_W [0..17408) is now free for s_K!

        // Coalesced half2 read-subtract-write across all 32 V columns, fused in the same barrier
        // epoch with s_K scaling into disjoint [0..17408).
        __half2* s_Vnew_h2 = reinterpret_cast<__half2*>(s_Vnew);
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            const int idx2 = tid + j * 256;
            const int row = idx2 >> 4;
            const int col2 = idx2 & 15;
            __half2 vn2 = __floats2half2_rn(0.0f, 0.0f);
            if (row < tile_len) {
                __half2* u_ptr = reinterpret_cast<__half2*>(
                    u_vnew_inout + ((size_t)(tok0 + row) * num_v_heads + hv) * DK + vbase);
                const float2 uf = __half22float2(u_pre[j]);
                const float wh0 = s_wh_full[row * BV_PAD + col2 * 2 + 0];
                const float wh1 = s_wh_full[row * BV_PAD + col2 * 2 + 1];
                vn2 = __floats2half2_rn(uf.x - wh0, uf.y - wh1);
                u_ptr[col2] = vn2;
            }
            s_Vnew_h2[row * (BH_PAD / 2) + col2] = vn2;
        }

        const float g_last = s_g[tile_len - 1];
        const float decay_chunk = expf(g_last);
        if (k_norm_in != nullptr) {
#pragma unroll
            for (int j = 0; j < 8; ++j) {
                const int idx4 = tid + j * 256;
                const int s_norm = idx4 >> 5;
                const int d4 = idx4 & 31;
                uint2 v = make_uint2(0u, 0u);
                if (s_norm < tile_len) {
                    const uint2 kn_u2 = kn_pre[j];
                    const __half2* kn_h2 = reinterpret_cast<const __half2*>(&kn_u2);
                    const float sc = s_sc[s_norm];
                    const float2 k01 = __half22float2(kn_h2[0]);
                    const float2 k23 = __half22float2(kn_h2[1]);
                    const __half2 kv01 = __floats2half2_rn(k01.x * sc, k01.y * sc);
                    const __half2 kv23 = __floats2half2_rn(k23.x * sc, k23.y * sc);
                    __half2* v_h2 = reinterpret_cast<__half2*>(&v);
                    v_h2[0] = kv01;
                    v_h2[1] = kv23;
                }
                reinterpret_cast<uint2*>(s_K + s_norm * DK_PAD)[d4] = v;
            }
        } else {
            for (int s_norm = wid; s_norm < BT; s_norm += 8) {
                __half kh0 = __float2half(0.0f);
                __half kh1 = __float2half(0.0f);
                __half kh2 = __float2half(0.0f);
                __half kh3 = __float2half(0.0f);
                if (s_norm < tile_len) {
                    const int tok = tok0 + s_norm;
                    const __half* k_tok_hk = (qkv_conv_out != nullptr)
                        ? (qkv_conv_out + (size_t)tok * qkv_stride + k_total_dim + (size_t)hk * DK)
                        : (k_in + ((size_t)tok * num_k_heads + hk) * DK);
                    float4 bk4;
                    gdn_norm_single_warp(k_tok_hk, lane, 1.0f, bk4);
                    const float sc = s_sc[s_norm];
                    kh0 = __float2half(__half2float(__float2half(bk4.x)) * sc);
                    kh1 = __float2half(__half2float(__float2half(bk4.y)) * sc);
                    kh2 = __float2half(__half2float(__float2half(bk4.z)) * sc);
                    kh3 = __float2half(__half2float(__float2half(bk4.w)) * sc);
                }
                __half* k_dst = s_K + s_norm * DK_PAD + lane * 4;
                k_dst[0] = kh0;
                k_dst[1] = kh1;
                k_dst[2] = kh2;
                k_dst[3] = kh3;
            }
        }
        __syncthreads();

#pragma unroll
        for (int kt = 0; kt < 2; ++kt) {
#pragma unroll
            for (int e = 0; e < h_acc[kt].num_elements; ++e) {
                h_acc[kt].x[e] *= decay_chunk;
            }
            const int k0 = (it * 2 + kt) * 16;
#pragma unroll
            for (int jj = 0; jj < BT; jj += 16) {
                FragA_Col ak;
                FragB_Row bv;
                wmma::load_matrix_sync(ak, s_K + jj * DK_PAD + k0, DK_PAD);
                wmma::load_matrix_sync(bv, s_Vnew + jj * BH_PAD + v_off, BH_PAD);
                wmma::mma_sync(h_acc[kt], ak, bv, h_acc[kt]);
            }
        }
        __syncthreads();

        const bool is_last_chunk = (c_local == span.num_chunks - 1);
        const int32_t final_slot = (is_last_chunk && batch.state_out_indices != nullptr)
            ? batch.state_out_indices[req]
            : -1;
        const bool need_ckpt = (cd.ckpt_slot >= 0 && cd.ckpt_slot < total_state_slots);
        const bool need_final = (final_slot >= 0 && final_slot < total_state_slots);
        if (!is_last_chunk || need_ckpt || need_final) {
#pragma unroll
            for (int kt = 0; kt < 2; ++kt) {
                wmma::store_matrix_sync(
                    s_H_f32 + ((it * 2 + kt) * 16) * BV_PAD + v_off,
                    h_acc[kt],
                    BV_PAD,
                    wmma::mem_row_major);
            }
            __syncthreads();
            if (!is_last_chunk) {
                __half2* s_H_h2 = reinterpret_cast<__half2*>(s_H);
                for (int idx2 = tid; idx2 < (DK * BV) / 2; idx2 += 256) {
                    const int k = idx2 >> 4;
                    const int vc2 = idx2 & 15;
                    s_H_h2[k * (BH_PAD / 2) + vc2] = __floats2half2_rn(
                        s_H_f32[k * BV_PAD + vc2 * 2 + 0],
                        s_H_f32[k * BV_PAD + vc2 * 2 + 1]);
                }
            }
            if (need_ckpt) {
                float* head_dst =
                    ssm_state_pool + ((size_t)cd.ckpt_slot * num_v_heads + hv) * head_state_elems;
                for (int idx = tid; idx < BV * DK; idx += 256) {
                    const int vc = idx >> 7;
                    const int k = idx & 127;
                    head_dst[(size_t)(vbase + vc) * DK + k] = s_H_f32[k * BV_PAD + vc];
                }
            }
            if (need_final) {
                float* head_dst =
                    ssm_state_pool + ((size_t)final_slot * num_v_heads + hv) * head_state_elems;
                for (int idx = tid; idx < BV * DK; idx += 256) {
                    const int vc = idx >> 7;
                    const int k = idx & 127;
                    head_dst[(size_t)(vbase + vc) * DK + k] = s_H_f32[k * BV_PAD + vc];
                }
            }
            __syncthreads();
        }
    }
#else
    (void)batch; (void)req_spans; (void)chunks; (void)qkv_conv_out; (void)k_in; (void)k_norm_in;
    (void)g_cum; (void)w_in; (void)u_vnew_inout; (void)h_chunks_out;
    (void)ssm_state_pool; (void)num_k_heads; (void)num_v_heads; (void)total_state_slots;
#endif
}

// Stage 3 (T-parallel across ALL 64-token chunks x V-heads, BOTH V-halves fused per CTA, 2 CTAs/SM):
// Computes Q_norm, K_norm load, and causal(Q_norm @ K_norm^T * exp(g_i - g_j)) ONCE per (chunk, hv),
// then applies both V-halves: O_half = exp(g) * (Q_norm @ H_half) + QK_causal @ V_new_half.
// Never reads ssm_state_pool (Stage 2 snapshots incoming initial H into h_chunks_in before mutating
// in-place state_out == state_in).
__global__ __launch_bounds__(256, 2) void gdn_wy_stage3_out_f16_kernel(
    const WyPlanHeader* __restrict__ hdr,
    const WyChunkDesc* __restrict__ chunks,
    const __half* __restrict__ qkv_conv_out,
    const __half* __restrict__ q_in,
    const __half* __restrict__ k_in,
    const __half* __restrict__ k_norm_in,
    const float* __restrict__ g_cum,
    const __half* __restrict__ h_chunks_in,
    __half* __restrict__ vnew_out_inout,
    int num_k_heads,
    int num_v_heads) {
#if __CUDA_ARCH__ >= 800
    constexpr int BT = 64;
    constexpr int BV = 64;
    constexpr int DK = 128;

    const int gc = blockIdx.x;
    const int hv = blockIdx.y;
    if (gc >= hdr->total_chunks) return;

    const WyChunkDesc cd = chunks[gc];
    const int tok0 = cd.tok0;
    const int tile_len = cd.len;
    const int gqa_ratio = num_v_heads / num_k_heads;
    const int hk = hv / gqa_ratio;
    const int tid = threadIdx.x;
    const int wid = tid >> 5;
    const int lane = tid & 31;
    const size_t head_state_elems = (size_t)DK * (size_t)DK;
    const int k_total_dim = num_k_heads * DK;
    const int v_total_dim = num_v_heads * DK;
    const int qkv_stride = 2 * k_total_dim + v_total_dim;
    const float scale = rsqrtf((float)DK);

    // 42,496 B total dynamic smem -> 2 CTAs/SM co-resident:
    // 46,592 B total dynamic smem -> 2 CTAs/SM co-resident. All row strides carry +16 B padding
    // (QP = 136, HP = QKP = 72 __half; OP = 68 float) to break power-of-two bank conflicts:
    // [0 .. 17408): s_Q [64, 136] __half -> later QK_causal [64, 72] __half (9,216 B)
    // [17408 .. 35840): s_Buf union { H_half [128, 72], K_norm [64, 136], V_new_half [64, 72],
    //                                  s_O_full [64, 68] float }
    // [35840 .. 46080): s_Stage [8 * 16 * 20] float (per-warp 16x16 staging, SP = 20 padded)
    // [46080 .. 46336): s_g [64] float (256 B)
    // [46336 .. 46592): s_eg [64] float (256 B)
    constexpr int QP = 136;
    constexpr int HP = 72;
    constexpr int QKP = 72;
    constexpr int OP = 68;
    constexpr int SP = 20;
    extern __shared__ __align__(16) char smem_raw[];
    __half* s_Q = reinterpret_cast<__half*>(smem_raw);
    __half* s_Buf = reinterpret_cast<__half*>(smem_raw + 17408);
    float* s_Stage = reinterpret_cast<float*>(smem_raw + 35840) + wid * (16 * SP);
    float* s_g = reinterpret_cast<float*>(smem_raw + 46080);
    float* s_eg = s_g + 64;

    using Acc = wmma::fragment<wmma::accumulator, 16, 16, 16, float>;
    using FragA_Row = wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, wmma::row_major>;
    using FragB_Row = wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, wmma::row_major>;
    using FragB_Col = wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, wmma::col_major>;

    if (tid < BT) {
        const float gv = (tid < tile_len) ? g_cum[(size_t)(tok0 + tid) * num_v_heads + hv] : 0.0f;
        s_g[tid] = gv;
        s_eg[tid] = expf(gv);
    }
    for (int s_norm = wid; s_norm < tile_len; s_norm += 8) {
        const int tok = tok0 + s_norm;
        const __half* q_tok_hk = (qkv_conv_out != nullptr)
            ? (qkv_conv_out + (size_t)tok * qkv_stride + (size_t)hk * DK)
            : (q_in + ((size_t)tok * num_k_heads + hk) * DK);
        float4 bq4;
        gdn_norm_single_warp(q_tok_hk, lane, scale, bq4);
        __half* q_dst = s_Q + s_norm * QP + lane * 4;
        q_dst[0] = __float2half(bq4.x);
        q_dst[1] = __float2half(bq4.y);
        q_dst[2] = __float2half(bq4.z);
        q_dst[3] = __float2half(bq4.w);
    }
    for (int idx = tile_len * QP + tid; idx < BT * QP; idx += 256) {
        s_Q[idx] = __float2half(0.0f);
    }
    __syncthreads();

    Acc acc_o[2][2];  // [v_half 0..1][step_t 0..1]
#pragma unroll
    for (int vh = 0; vh < 2; ++vh) {
        const int vbase = vh * BV;
        if (cd.h_slot >= 0) {
            const __half* h_src = h_chunks_in + ((size_t)cd.h_slot * num_v_heads + hv) * head_state_elems;
            for (int idx4 = tid; idx4 < (DK * BV) / 4; idx4 += 256) {
                const int k = idx4 >> 4;
                const int vc4 = idx4 & 15;
                reinterpret_cast<uint2*>(s_Buf + k * HP)[vc4] =
                    reinterpret_cast<const uint2*>(h_src + (size_t)k * DK + vbase)[vc4];
            }
            __syncthreads();
#pragma unroll
            for (int step_t = 0; step_t < 2; ++step_t) {
                const int t_id = wid + step_t * 8;
                const int it = t_id & 3;
                const int vt = t_id >> 2;
                wmma::fill_fragment(acc_o[vh][step_t], 0.0f);
#pragma unroll
                for (int d = 0; d < DK; d += 16) {
                    FragA_Row aq;
                    FragB_Row bh;
                    wmma::load_matrix_sync(aq, s_Q + (it * 16) * QP + d, QP);
                    wmma::load_matrix_sync(bh, s_Buf + d * HP + (vt * 16), HP);
                    wmma::mma_sync(acc_o[vh][step_t], aq, bh, acc_o[vh][step_t]);
                }
            }
            __syncthreads();
        } else {
            wmma::fill_fragment(acc_o[vh][0], 0.0f);
            wmma::fill_fragment(acc_o[vh][1], 0.0f);
        }
    }

    // Load K_norm [64, 128] ONCE into s_Buf with DK_PAD = 136 and compute QK = s_Q @ K_norm^T ONCE!
    constexpr int DK_PAD = 136;
    if (k_norm_in != nullptr) {
        uint2* s_Buf_u2 = reinterpret_cast<uint2*>(s_Buf);
        for (int idx4 = tid; idx4 < (BT * DK_PAD) / 4; idx4 += 256) {
            const int s_norm = idx4 / 34;
            const int d4 = idx4 % 34;
            uint2 v = make_uint2(0u, 0u);
            if (s_norm < tile_len && d4 < 32) {
                v = reinterpret_cast<const uint2*>(
                    k_norm_in + ((size_t)(tok0 + s_norm) * num_k_heads + hk) * DK)[d4];
            }
            s_Buf_u2[idx4] = v;
        }
    } else {
        for (int s_norm = wid; s_norm < tile_len; s_norm += 8) {
            const int tok = tok0 + s_norm;
            const __half* k_tok_hk = (qkv_conv_out != nullptr)
                ? (qkv_conv_out + (size_t)tok * qkv_stride + k_total_dim + (size_t)hk * DK)
                : (k_in + ((size_t)tok * num_k_heads + hk) * DK);
            float4 bk4;
            gdn_norm_single_warp(k_tok_hk, lane, 1.0f, bk4);
            __half* k_dst = s_Buf + s_norm * DK_PAD + lane * 4;
            k_dst[0] = __float2half(bk4.x);
            k_dst[1] = __float2half(bk4.y);
            k_dst[2] = __float2half(bk4.z);
            k_dst[3] = __float2half(bk4.w);
            if (lane == 0) {
                *reinterpret_cast<uint4*>(s_Buf + s_norm * DK_PAD + 128) = make_uint4(0, 0, 0, 0);
            }
        }
        for (int idx = tile_len * DK_PAD + tid; idx < BT * DK_PAD; idx += 256) {
            s_Buf[idx] = __float2half(0.0f);
        }
    }
    __syncthreads();

    Acc acc_qk[2];
#pragma unroll
    for (int step_t = 0; step_t < 2; ++step_t) {
        const int t_id = wid + step_t * 8;
        const int it = t_id & 3;
        const int jt = t_id >> 2;
        wmma::fill_fragment(acc_qk[step_t], 0.0f);
#pragma unroll
        for (int d = 0; d < DK; d += 16) {
            FragA_Row aq;
            FragB_Col bk;
            wmma::load_matrix_sync(aq, s_Q + (it * 16) * QP + d, QP);
            wmma::load_matrix_sync(bk, s_Buf + (jt * 16) * DK_PAD + d, DK_PAD);
            wmma::mma_sync(acc_qk[step_t], aq, bk, acc_qk[step_t]);
        }
    }
    __syncthreads();  // s_Q and s_Buf are both free!

    // Stage acc_qk -> causal QK_f16 [64, 64] in s_Q, and scale acc_o[0..1][0..1] by exp(s_g[row]).
#pragma unroll
    for (int step_t = 0; step_t < 2; ++step_t) {
        const int t_id = wid + step_t * 8;
        const int it = t_id & 3;
        const int jt = t_id >> 2;
        wmma::store_matrix_sync(s_Stage, acc_qk[step_t], SP, wmma::mem_row_major);
        __syncwarp();
        for (int e = lane; e < 256; e += 32) {
            const int r = e >> 4;
            const int c = e & 15;
            const int row = it * 16 + r;
            const int col = jt * 16 + c;
            s_Q[row * QKP + col] = (row < tile_len && col <= row)
                ? __float2half(expf(s_g[row] - s_g[col]) * s_Stage[r * SP + c])
                : __float2half(0.0f);
        }
        __syncwarp();
    }
#pragma unroll
    for (int vh = 0; vh < 2; ++vh) {
#pragma unroll
        for (int step_t = 0; step_t < 2; ++step_t) {
            const int t_id = wid + step_t * 8;
            const int it = t_id & 3;
            wmma::store_matrix_sync(s_Stage, acc_o[vh][step_t], SP, wmma::mem_row_major);
            __syncwarp();
            for (int e = lane; e < 256; e += 32) {
                const int row = it * 16 + (e >> 4);
                const int si = (e >> 4) * SP + (e & 15);
                s_Stage[si] = (row < tile_len) ? (s_eg[row] * s_Stage[si]) : 0.0f;
            }
            __syncwarp();
            wmma::load_matrix_sync(acc_o[vh][step_t], s_Stage, SP, wmma::mem_row_major);
            __syncwarp();
        }
    }
    __syncthreads();

    // Apply QK_causal @ V_new_half and write O_half with coalesced half2 stores for vh = 0, 1.
    float* s_O_full = reinterpret_cast<float*>(s_Buf);  // [64, 68] FP32 (17,408 B)
#pragma unroll
    for (int vh = 0; vh < 2; ++vh) {
        const int vbase = vh * BV;
        for (int idx4 = tid; idx4 < (BT * BV) / 4; idx4 += 256) {
            const int i = idx4 >> 4;
            const int c4 = idx4 & 15;
            uint2 v = make_uint2(0u, 0u);
            if (i < tile_len) {
                v = reinterpret_cast<const uint2*>(
                    vnew_out_inout + ((size_t)(tok0 + i) * num_v_heads + hv) * DK + vbase)[c4];
            }
            reinterpret_cast<uint2*>(s_Buf + i * HP)[c4] = v;
        }
        __syncthreads();
#pragma unroll
        for (int step_t = 0; step_t < 2; ++step_t) {
            const int t_id = wid + step_t * 8;
            const int it = t_id & 3;
            const int vt = t_id >> 2;
#pragma unroll
            for (int jj = 0; jj < BT; jj += 16) {
                FragA_Row a;
                FragB_Row b;
                wmma::load_matrix_sync(a, s_Q + (it * 16) * QKP + jj, QKP);
                wmma::load_matrix_sync(b, s_Buf + jj * HP + (vt * 16), HP);
                wmma::mma_sync(acc_o[vh][step_t], a, b, acc_o[vh][step_t]);
            }
        }
        __syncthreads();
#pragma unroll
        for (int step_t = 0; step_t < 2; ++step_t) {
            const int t_id = wid + step_t * 8;
            const int it = t_id & 3;
            const int vt = t_id >> 2;
            wmma::store_matrix_sync(
                s_O_full + (it * 16) * OP + (vt * 16), acc_o[vh][step_t], OP, wmma::mem_row_major);
        }
        __syncthreads();
        for (int idx2 = tid; idx2 < (tile_len * BV) / 2; idx2 += 256) {
            const int i = idx2 >> 5;
            const int c2 = idx2 & 31;
            const float o0 = s_O_full[i * OP + c2 * 2 + 0];
            const float o1 = s_O_full[i * OP + c2 * 2 + 1];
            reinterpret_cast<__half2*>(
                vnew_out_inout + ((size_t)(tok0 + i) * num_v_heads + hv) * DK + vbase)[c2] =
                __floats2half2_rn(o0, o1);
        }
        __syncthreads();
    }
#else
    (void)hdr; (void)chunks; (void)qkv_conv_out; (void)q_in; (void)k_in; (void)k_norm_in;
    (void)g_cum; (void)h_chunks_in; (void)vnew_out_inout;
    (void)num_k_heads; (void)num_v_heads;
#endif
}

// Fused Tensor-Core WY chunked GDN prefill kernel (`BT = 64, BV = 64, DK = 128`, 8 warps = 256
// threads, 1 single wave of 64 CTAs on 9B TP=1 / 48 CTAs on 27B TP=2 across 84 SMs, 0 bytes global
// scratch, STACK: 0).
// Donor math: vllm.cpp (50d425bec) `GdnChunkWUWmmaVecKernel` + `GdnChunkDeltaHRegKernel` +
// `GdnChunkOWmmaKernel`, fused across chunks so persistent state `H [128, 64]` remains in FP32
// `wmma::accumulator` registers across the entire sequence while checkpoints clamp `tile_len` to
// hit exact token offsets. Used for large non-speculative prefill batches (`avg_seqlen >= 256`).
static constexpr int kWy64SmemBytes = 99072;

__global__ __launch_bounds__(256, 1) void gdn_wy_chunk64_prefill_f16_kernel(
    w34_gdn_batch batch,
    const __half* __restrict__ qkv_conv_out,
    const __half* __restrict__ q_in,
    const __half* __restrict__ k_in,
    const __half* __restrict__ v_in,
    const __half* __restrict__ b_in,
    const __half* __restrict__ a_in,
    const float* __restrict__ a_log,
    const float* __restrict__ dt_bias,
    float* __restrict__ ssm_state_pool,
    __half* __restrict__ out_f16,
    int num_k_heads,
    int num_v_heads,
    int ba_row_stride,
    int total_state_slots) {
#if __CUDA_ARCH__ >= 800
    constexpr int BT = 64;
    constexpr int BV = 64;
    constexpr int DK = 128;

    const int i_v = blockIdx.x;
    const int i_nh = blockIdx.y;
    const int req = i_nh / num_v_heads;
    const int hv = i_nh % num_v_heads;
    if (req >= batch.num_reqs) return;
    if (batch.valid_req_mask != nullptr && batch.valid_req_mask[req] == 0) return;

    const int lo = batch.query_start_loc[req];
    const int hi = batch.query_start_loc[req + 1];
    const int seqlen = hi - lo;
    if (seqlen <= 0) return;

    const int gqa_ratio = num_v_heads / num_k_heads;
    const int hk = hv / gqa_ratio;
    const int tid = threadIdx.x;
    const int wid = tid >> 5;
    const int lane = tid & 31;
    const int vbase = i_v * BV;
    const size_t head_state_elems = (size_t)DK * (size_t)DK;

    extern __shared__ __align__(16) char smem_raw[];
    // [0 .. 32768): f32_128x64 (32 KiB) OR { s_Q [64,128] (16 KiB), s_W [64,128] (16 KiB) }
    float* s_H_f32 = reinterpret_cast<float*>(smem_raw);
    __half* s_Q = reinterpret_cast<__half*>(smem_raw);
    __half* s_W = reinterpret_cast<__half*>(smem_raw + 16384);
    // [32768 .. 49152): s_H [128, 64] __half (16 KiB)
    __half* s_H = reinterpret_cast<__half*>(smem_raw + 32768);
    // [49152 .. 65536): s_K [64, 128] __half (16 KiB)
    __half* s_K = reinterpret_cast<__half*>(smem_raw + 49152);
    // [65536 .. 81920): s_A [64, 64] float (16 KiB)
    float* s_A = reinterpret_cast<float*>(smem_raw + 65536);
    // [81920 .. 98304): s_T [64, 64] float (16 KiB)
    float* s_T = reinterpret_cast<float*>(smem_raw + 81920);
    // [98304 .. 99072): s_g[64], s_eg[64], s_beta[64] (768 B)
    float* s_g = reinterpret_cast<float*>(smem_raw + 98304);
    float* s_eg = s_g + 64;
    float* s_beta = s_g + 128;

    using Acc = wmma::fragment<wmma::accumulator, 16, 16, 16, float>;
    using FragA_Row = wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, wmma::row_major>;
    using FragB_Row = wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, wmma::row_major>;
    using FragB_Col = wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, wmma::col_major>;

    // Warp `wid` (0..7) owns K-rows [wid*16 .. wid*16 + 16) across V-cols [0..64): 4 16x16 tiles.
    Acc h_acc[4];
    const bool has_init = (batch.has_initial_state == nullptr) || (batch.has_initial_state[req] != 0);
    const int32_t slot_in = (batch.state_in_indices != nullptr) ? batch.state_in_indices[req] : -1;
    if (has_init && slot_in >= 0 && slot_in < total_state_slots) {
        const float* head_in = ssm_state_pool + ((size_t)slot_in * num_v_heads + hv) * head_state_elems;
        for (int idx = tid; idx < BV * DK; idx += 256) {
            const int vc = idx >> 7;
            const int k = idx & 127;
            const float val = head_in[(size_t)(vbase + vc) * DK + k];
            s_H_f32[k * BV + vc] = val;
            s_H[k * BV + vc] = __float2half(val);
        }
        __syncthreads();
#pragma unroll
        for (int vt = 0; vt < 4; ++vt) {
            wmma::load_matrix_sync(
                h_acc[vt], s_H_f32 + (wid * 16) * BV + vt * 16, BV, wmma::mem_row_major);
        }
        __syncthreads();
    } else {
        for (int idx = tid; idx < DK * BV; idx += 256) {
            s_H[idx] = __float2half(0.0f);
        }
#pragma unroll
        for (int vt = 0; vt < 4; ++vt) {
            wmma::fill_fragment(h_acc[vt], 0.0f);
        }
        __syncthreads();
    }

    const float scale = rsqrtf((float)DK);
    const float neg_exp_a = -expf(a_log[hv]);
    const float dt_bias_hv = dt_bias[hv];
    const int k_total_dim = num_k_heads * DK;
    const int v_total_dim = num_v_heads * DK;
    const int qkv_stride = 2 * k_total_dim + v_total_dim;
    const bool has_checkpoints =
        (batch.checkpoint_dst_indices != nullptr && batch.checkpoint_token_offsets != nullptr &&
         batch.max_checkpoints_per_req > 0);

    int base_s = 0;
    while (base_s < seqlen) {
        int tile_len = (seqlen - base_s < BT) ? (seqlen - base_s) : BT;
        int32_t hit_ckpt_slot = -1;
        if (has_checkpoints) {
            for (int e = 0; e < batch.max_checkpoints_per_req; ++e) {
                const int32_t trig_off =
                    batch.checkpoint_token_offsets[req * batch.max_checkpoints_per_req + e];
                if (trig_off > base_s && trig_off <= base_s + tile_len) {
                    const int cand_len = trig_off - base_s;
                    if (cand_len < tile_len) tile_len = cand_len;
                }
            }
            for (int e = 0; e < batch.max_checkpoints_per_req; ++e) {
                const int32_t trig_off =
                    batch.checkpoint_token_offsets[req * batch.max_checkpoints_per_req + e];
                if (trig_off == base_s + tile_len) {
                    hit_ckpt_slot =
                        batch.checkpoint_dst_indices[req * batch.max_checkpoints_per_req + e];
                    break;
                }
            }
        }

        // Phase 1: L2-normalize Q and K across 8 warps (8 tokens/iter).
        for (int s_norm = wid; s_norm < tile_len; s_norm += 8) {
            const int tok = lo + base_s + s_norm;
            const __half* q_tok_hk = (qkv_conv_out != nullptr)
                ? (qkv_conv_out + (size_t)tok * qkv_stride + (size_t)hk * DK)
                : (q_in + ((size_t)tok * num_k_heads + hk) * DK);
            const __half* k_tok_hk = (qkv_conv_out != nullptr)
                ? (qkv_conv_out + (size_t)tok * qkv_stride + k_total_dim + (size_t)hk * DK)
                : (k_in + ((size_t)tok * num_k_heads + hk) * DK);
            float4 bq4, bk4;
            gdn_norm_qk_warp(q_tok_hk, k_tok_hk, lane, scale, bq4, bk4);
            __half* q_dst = s_Q + s_norm * DK + lane * 4;
            __half* k_dst = s_K + s_norm * DK + lane * 4;
            q_dst[0] = __float2half(bq4.x);
            q_dst[1] = __float2half(bq4.y);
            q_dst[2] = __float2half(bq4.z);
            q_dst[3] = __float2half(bq4.w);
            k_dst[0] = __float2half(bk4.x);
            k_dst[1] = __float2half(bk4.y);
            k_dst[2] = __float2half(bk4.z);
            k_dst[3] = __float2half(bk4.w);
            if (lane == 0) {
                const float a_val = __half2float(a_in[(size_t)tok * ba_row_stride + hv]);
                const float b_val = __half2float(b_in[(size_t)tok * ba_row_stride + hv]);
                s_g[s_norm] = neg_exp_a * softplus_f32(a_val + dt_bias_hv);
                s_beta[s_norm] = 1.0f / (1.0f + expf(-b_val));
            }
        }
        for (int idx = tile_len * DK + tid; idx < BT * DK; idx += 256) {
            s_Q[idx] = __float2half(0.0f);
            s_K[idx] = __float2half(0.0f);
        }
        if (tid >= tile_len && tid < BT) {
            s_g[tid] = 0.0f;
            s_eg[tid] = 1.0f;
            s_beta[tid] = 0.0f;
        }
        __syncthreads();
        // Parallel 2-warp inclusive prefix sum over s_g[0..63] using warp 0 & warp 1.
        if (wid == 0) {
            float v0 = (lane < tile_len) ? s_g[lane] : 0.0f;
            float v1 = (32 + lane < tile_len) ? s_g[32 + lane] : 0.0f;
#pragma unroll
            for (int off = 1; off < 32; off <<= 1) {
                const float n0 = __shfl_up_sync(0xffffffffu, v0, off);
                const float n1 = __shfl_up_sync(0xffffffffu, v1, off);
                if (lane >= off) {
                    v0 += n0;
                    v1 += n1;
                }
            }
            const float sum0 = __shfl_sync(0xffffffffu, v0, 31);
            v1 += sum0;
            if (lane < tile_len) {
                s_g[lane] = v0;
                s_eg[lane] = expf(v0);
            }
            if (32 + lane < tile_len) {
                s_g[32 + lane] = v1;
                s_eg[32 + lane] = expf(v1);
            }
        }
        __syncthreads();

        // Phase 2: Gram Am = s_K @ s_K^T [64, 64] (16 tiles across 8 warps = 2 tiles/warp).
#pragma unroll
        for (int step_t = 0; step_t < 2; ++step_t) {
            const int t_id = wid + step_t * 8;
            const int it = t_id & 3;
            const int jt = t_id >> 2;
            Acc acc;
            wmma::fill_fragment(acc, 0.0f);
#pragma unroll
            for (int d = 0; d < DK; d += 16) {
                FragA_Row a;
                FragB_Col b;
                wmma::load_matrix_sync(a, s_K + (it * 16) * DK + d, DK);
                wmma::load_matrix_sync(b, s_K + (jt * 16) * DK + d, DK);
                wmma::mma_sync(acc, a, b, acc);
            }
            wmma::store_matrix_sync(s_A + (it * 16) * BT + (jt * 16), acc, BT, wmma::mem_row_major);
        }
        for (int idx = tid; idx < BT * BT; idx += 256) {
            s_T[idx] = 0.0f;
        }
        __syncthreads();

        for (int idx = tid; idx < BT * BT; idx += 256) {
            const int i = idx >> 6;
            const int j = idx & 63;
            s_A[idx] = (i < tile_len && j < i)
                ? (s_beta[i] * expf(s_g[i] - s_g[j]) * s_A[idx])
                : 0.0f;
        }
        __syncthreads();

        // Invert T = (I + A)^{-1}: 4 parallel 16x16 diagonal blocks (64 threads) + 3 parallel
        // 256-thread exact FP32 Schur-merge phases staged in s_W (currently unused, 16 KiB).
        if (tid < 64) {
            const int bd = tid >> 4;
            const int base = bd * 16;
            const int c = base + (tid & 15);
            s_T[c * BT + c] = 1.0f;
            if (c < tile_len) {
                const int iend = (base + 16 < tile_len) ? (base + 16) : tile_len;
                for (int i = c + 1; i < iend; ++i) {
                    float sfs = 0.0f;
                    for (int m = c; m < i; ++m) {
                        sfs += s_A[i * BT + m] * s_T[m * BT + c];
                    }
                    s_T[i * BT + c] = -sfs;
                }
            }
        }
        __syncthreads();

        float* s_pw = reinterpret_cast<float*>(s_W);  // up to 3 * 256 floats (3 KiB <= 16 KiB)
        // Distance 1: (1,0), (2,1), (3,2) -> 3 * 256 = 768 elements.
        for (int idx = tid; idx < 768; idx += 256) {
            const int blk = idx >> 8;  // 0, 1, 2 -> (bi = blk+1, bj = blk)
            const int bi = blk + 1;
            const int bj = blk;
            const int r = (idx >> 4) & 15;
            const int c = idx & 15;
            float sum = 0.0f;
#pragma unroll
            for (int m = 0; m < 16; ++m) {
                sum += s_A[(bi * 16 + r) * BT + (bj * 16 + m)] * s_T[(bj * 16 + m) * BT + (bj * 16 + c)];
            }
            s_pw[idx] = sum;
        }
        __syncthreads();
        for (int idx = tid; idx < 768; idx += 256) {
            const int blk = idx >> 8;
            const int bi = blk + 1;
            const int bj = blk;
            const int r = (idx >> 4) & 15;
            const int c = idx & 15;
            const float* pw_blk = s_pw + blk * 256;
            float sum = 0.0f;
#pragma unroll
            for (int m = 0; m < 16; ++m) {
                sum += s_T[(bi * 16 + r) * BT + (bi * 16 + m)] * pw_blk[m * 16 + c];
            }
            s_T[(bi * 16 + r) * BT + (bj * 16 + c)] = -sum;
        }
        __syncthreads();

        // Distance 2: (2,0), (3,1) -> 2 * 256 = 512 elements.
        for (int idx = tid; idx < 512; idx += 256) {
            const int blk = idx >> 8;  // 0 -> (2,0), 1 -> (3,1)
            const int bi = blk + 2;
            const int bj = blk;
            const int r = (idx >> 4) & 15;
            const int c = idx & 15;
            float sum = 0.0f;
#pragma unroll
            for (int bk = 0; bk < 2; ++bk) {
                const int kblk = bj + bk;
#pragma unroll
                for (int m = 0; m < 16; ++m) {
                    sum += s_A[(bi * 16 + r) * BT + (kblk * 16 + m)] *
                           s_T[(kblk * 16 + m) * BT + (bj * 16 + c)];
                }
            }
            s_pw[idx] = sum;
        }
        __syncthreads();
        for (int idx = tid; idx < 512; idx += 256) {
            const int blk = idx >> 8;
            const int bi = blk + 2;
            const int bj = blk;
            const int r = (idx >> 4) & 15;
            const int c = idx & 15;
            const float* pw_blk = s_pw + blk * 256;
            float sum = 0.0f;
#pragma unroll
            for (int m = 0; m < 16; ++m) {
                sum += s_T[(bi * 16 + r) * BT + (bi * 16 + m)] * pw_blk[m * 16 + c];
            }
            s_T[(bi * 16 + r) * BT + (bj * 16 + c)] = -sum;
        }
        __syncthreads();

        // Distance 3: (3,0) -> 256 elements (1 element per thread!).
        {
            const int r = tid >> 4;
            const int c = tid & 15;
            float sum = 0.0f;
#pragma unroll
            for (int kblk = 0; kblk < 3; ++kblk) {
#pragma unroll
                for (int m = 0; m < 16; ++m) {
                    sum += s_A[(48 + r) * BT + (kblk * 16 + m)] * s_T[(kblk * 16 + m) * BT + c];
                }
            }
            s_pw[tid] = sum;
        }
        __syncthreads();
        {
            const int r = tid >> 4;
            const int c = tid & 15;
            float sum = 0.0f;
#pragma unroll
            for (int m = 0; m < 16; ++m) {
                sum += s_T[(48 + r) * BT + (48 + m)] * s_pw[m * 16 + c];
            }
            s_T[(48 + r) * BT + c] = -sum;
        }
        __syncthreads();

        // Store s_T_f16 [64, 64] (8 KiB) in upper half of s_A; lower half of s_A (8 KiB) stays free.
        __half* s_Bop = reinterpret_cast<__half*>(s_A);        // [0 .. 4096) __half (8 KiB)
        __half* s_T_f16 = s_Bop + (BT * BT);                   // [4096 .. 8192) __half (8 KiB)
        for (int idx = tid; idx < BT * BT; idx += 256) {
            s_T_f16[idx] = __float2half(s_T[idx]);
        }
        __syncthreads();

        // Phase 3a: U = s_T_f16 @ (beta * V) -> s_T [64, 64] FP32 (stage Bop_V in s_Bop).
        for (int idx = tid; idx < BT * BV; idx += 256) {
            const int i = idx >> 6;
            const int c = idx & 63;
            float val = 0.0f;
            if (i < tile_len) {
                const int tok = lo + base_s + i;
                const __half* v_tok_hv = (qkv_conv_out != nullptr)
                    ? (qkv_conv_out + (size_t)tok * qkv_stride + 2 * k_total_dim + (size_t)hv * DK)
                    : (v_in + ((size_t)tok * num_v_heads + hv) * DK);
                val = s_beta[i] * __half2float(v_tok_hv[vbase + c]);
            }
            s_Bop[idx] = __float2half(val);
        }
        __syncthreads();
#pragma unroll
        for (int step_t = 0; step_t < 2; ++step_t) {
            const int t_id = wid + step_t * 8;
            const int it = t_id & 3;
            const int ct = t_id >> 2;
            Acc acc;
            wmma::fill_fragment(acc, 0.0f);
#pragma unroll
            for (int jj = 0; jj < BT; jj += 16) {
                FragA_Row a;
                FragB_Row b;
                wmma::load_matrix_sync(a, s_T_f16 + (it * 16) * BT + jj, BT);
                wmma::load_matrix_sync(b, s_Bop + jj * BV + (ct * 16), BV);
                wmma::mma_sync(acc, a, b, acc);
            }
            wmma::store_matrix_sync(s_T + (it * 16) * BV + (ct * 16), acc, BV, wmma::mem_row_major);
        }
        __syncthreads();

        // Phase 3b: W = s_T_f16 @ (beta * exp(g) * K) -> s_W [64, 128] in two 64-col halves.
        for (int cb = 0; cb < DK; cb += 64) {
            for (int idx = tid; idx < BT * 64; idx += 256) {
                const int i = idx >> 6;
                const int c = idx & 63;
                const float val = (i < tile_len)
                    ? (s_beta[i] * s_eg[i] * __half2float(s_K[i * DK + cb + c]))
                    : 0.0f;
                s_Bop[idx] = __float2half(val);
            }
            __syncthreads();
            Acc acc_w[2];
#pragma unroll
            for (int step_t = 0; step_t < 2; ++step_t) {
                const int t_id = wid + step_t * 8;
                const int it = t_id & 3;
                const int ct = t_id >> 2;
                wmma::fill_fragment(acc_w[step_t], 0.0f);
#pragma unroll
                for (int jj = 0; jj < BT; jj += 16) {
                    FragA_Row a;
                    FragB_Row b;
                    wmma::load_matrix_sync(a, s_T_f16 + (it * 16) * BT + jj, BT);
                    wmma::load_matrix_sync(b, s_Bop + jj * 64 + (ct * 16), 64);
                    wmma::mma_sync(acc_w[step_t], a, b, acc_w[step_t]);
                }
            }
            __syncthreads();  // s_Bop (lower 8 KiB of s_A = 8 * 256 floats) is now free for staging!
            float* s_w_stage = s_A + wid * 256;
#pragma unroll
            for (int step_t = 0; step_t < 2; ++step_t) {
                const int t_id = wid + step_t * 8;
                const int it = t_id & 3;
                const int ct = t_id >> 2;
                wmma::store_matrix_sync(s_w_stage, acc_w[step_t], 16, wmma::mem_row_major);
                __syncwarp();
                for (int e = lane; e < 256; e += 32) {
                    const int r = e >> 4;
                    const int c = e & 15;
                    s_W[(it * 16 + r) * DK + cb + ct * 16 + c] = __float2half(s_w_stage[e]);
                }
                __syncwarp();
            }
            __syncthreads();
        }

        // Phase 4: WH = s_W @ s_H -> s_A [64, 64], then V_new = U (s_T) - WH -> s_Vnew (aliases s_W).
#pragma unroll
        for (int step_t = 0; step_t < 2; ++step_t) {
            const int t_id = wid + step_t * 8;
            const int it = t_id & 3;
            const int vt = t_id >> 2;
            Acc acc;
            wmma::fill_fragment(acc, 0.0f);
#pragma unroll
            for (int d = 0; d < DK; d += 16) {
                FragA_Row a;
                FragB_Row b;
                wmma::load_matrix_sync(a, s_W + (it * 16) * DK + d, DK);
                wmma::load_matrix_sync(b, s_H + d * BV + (vt * 16), BV);
                wmma::mma_sync(acc, a, b, acc);
            }
            wmma::store_matrix_sync(s_A + (it * 16) * BV + (vt * 16), acc, BV, wmma::mem_row_major);
        }
        __syncthreads();
        __half* s_Vnew = s_W;  // s_W is now dead; first 8 KiB holds V_new [64, 64] __half.
        for (int idx = tid; idx < BT * BV; idx += 256) {
            const int i = idx >> 6;
            s_Vnew[idx] = (i < tile_len) ? __float2half(s_T[idx] - s_A[idx]) : __float2half(0.0f);
        }
        __syncthreads();

        // Phase 5: O = exp(g) * (s_Q @ s_H) + causal(s_Q @ s_K^T * exp(g_i - g_j)) @ s_Vnew.
#pragma unroll
        for (int step_t = 0; step_t < 2; ++step_t) {
            const int t_id = wid + step_t * 8;
            const int it = t_id & 3;
            const int ct = t_id >> 2;
            Acc acc_qh, acc_qk;
            wmma::fill_fragment(acc_qh, 0.0f);
            wmma::fill_fragment(acc_qk, 0.0f);
#pragma unroll
            for (int d = 0; d < DK; d += 16) {
                FragA_Row aq;
                FragB_Row bh;
                FragB_Col bk;
                wmma::load_matrix_sync(aq, s_Q + (it * 16) * DK + d, DK);
                wmma::load_matrix_sync(bh, s_H + d * BV + (ct * 16), BV);
                wmma::load_matrix_sync(bk, s_K + (ct * 16) * DK + d, DK);
                wmma::mma_sync(acc_qh, aq, bh, acc_qh);
                wmma::mma_sync(acc_qk, aq, bk, acc_qk);
            }
            wmma::store_matrix_sync(s_T + (it * 16) * BV + (ct * 16), acc_qh, BV, wmma::mem_row_major);
            wmma::store_matrix_sync(s_A + (it * 16) * BT + (ct * 16), acc_qk, BT, wmma::mem_row_major);
        }
        __syncthreads();
        for (int idx = tid; idx < BT * BT; idx += 256) {
            const int i = idx >> 6;
            const int j = idx & 63;
            s_Q[idx] = (i < tile_len && j <= i)
                ? __float2half(expf(s_g[i] - s_g[j]) * s_A[idx])
                : __float2half(0.0f);
            s_T[idx] = (i < tile_len) ? (s_eg[i] * s_T[idx]) : 0.0f;
        }
        __syncthreads();
#pragma unroll
        for (int step_t = 0; step_t < 2; ++step_t) {
            const int t_id = wid + step_t * 8;
            const int it = t_id & 3;
            const int vt = t_id >> 2;
            Acc acc_o;
            wmma::load_matrix_sync(
                acc_o, s_T + (it * 16) * BV + (vt * 16), BV, wmma::mem_row_major);
#pragma unroll
            for (int jj = 0; jj < BT; jj += 16) {
                FragA_Row a;
                FragB_Row b;
                wmma::load_matrix_sync(a, s_Q + (it * 16) * BT + jj, BT);
                wmma::load_matrix_sync(b, s_Vnew + jj * BV + (vt * 16), BV);
                wmma::mma_sync(acc_o, a, b, acc_o);
            }
            wmma::store_matrix_sync(s_A + (it * 16) * BV + (vt * 16), acc_o, BV, wmma::mem_row_major);
        }
        __syncthreads();
        for (int idx = tid; idx < tile_len * BV; idx += 256) {
            const int i = idx >> 6;
            const int c = idx & 63;
            out_f16[((size_t)(lo + base_s + i) * num_v_heads + hv) * DK + vbase + c] =
                __float2half(s_A[idx]);
        }

        // Phase 6: State update h_acc = exp(g_last) * h_acc + K_scaled_T @ s_Vnew.
        const float g_last = s_g[tile_len - 1];
        const float decay_chunk = expf(g_last);
#pragma unroll
        for (int vt = 0; vt < 4; ++vt) {
#pragma unroll
            for (int e = 0; e < h_acc[vt].num_elements; ++e) {
                h_acc[vt].x[e] *= decay_chunk;
            }
        }
        // Build K_scaled_T [128, 64] __half (16 KiB) in s_Q (disjoint from s_Vnew in s_W!).
        for (int idx = tid; idx < DK * BT; idx += 256) {
            const int k = idx >> 6;
            const int i = idx & 63;
            const float kv = (i < tile_len)
                ? (expf(g_last - s_g[i]) * __half2float(s_K[i * DK + k]))
                : 0.0f;
            s_Q[k * BT + i] = __float2half(kv);
        }
        __syncthreads();
        {
            const int k0 = wid * 16;
#pragma unroll
            for (int vt = 0; vt < 4; ++vt) {
                const int v0 = vt * 16;
#pragma unroll
                for (int jj = 0; jj < BT; jj += 16) {
                    FragA_Row ak;
                    FragB_Row bv;
                    wmma::load_matrix_sync(ak, s_Q + k0 * BT + jj, BT);
                    wmma::load_matrix_sync(bv, s_Vnew + jj * BV + v0, BV);
                    wmma::mma_sync(h_acc[vt], ak, bv, h_acc[vt]);
                }
            }
        }
        __syncthreads();

        const bool is_last_chunk = (base_s + tile_len >= seqlen);
        const int32_t final_slot = (is_last_chunk && batch.state_out_indices != nullptr)
            ? batch.state_out_indices[req]
            : -1;
        const bool need_ckpt = (hit_ckpt_slot >= 0 && hit_ckpt_slot < total_state_slots);
        const bool need_final = (final_slot >= 0 && final_slot < total_state_slots);
        if (!is_last_chunk || need_ckpt || need_final) {
#pragma unroll
            for (int vt = 0; vt < 4; ++vt) {
                wmma::store_matrix_sync(
                    s_H_f32 + (wid * 16) * BV + vt * 16, h_acc[vt], BV, wmma::mem_row_major);
            }
            __syncthreads();
            if (!is_last_chunk) {
                for (int idx = tid; idx < DK * BV; idx += 256) {
                    s_H[idx] = __float2half(s_H_f32[idx]);
                }
            }
            if (need_ckpt) {
                float* head_dst =
                    ssm_state_pool + ((size_t)hit_ckpt_slot * num_v_heads + hv) * head_state_elems;
                for (int idx = tid; idx < BV * DK; idx += 256) {
                    const int vc = idx >> 7;
                    const int k = idx & 127;
                    head_dst[(size_t)(vbase + vc) * DK + k] = s_H_f32[k * BV + vc];
                }
            }
            if (need_final) {
                float* head_dst =
                    ssm_state_pool + ((size_t)final_slot * num_v_heads + hv) * head_state_elems;
                for (int idx = tid; idx < BV * DK; idx += 256) {
                    const int vc = idx >> 7;
                    const int k = idx & 127;
                    head_dst[(size_t)(vbase + vc) * DK + k] = s_H_f32[k * BV + vc];
                }
            }
            __syncthreads();
        }

        base_s += tile_len;
    }
#else
    (void)batch; (void)qkv_conv_out; (void)q_in; (void)k_in; (void)v_in;
    (void)b_in; (void)a_in; (void)a_log; (void)dt_bias; (void)ssm_state_pool;
    (void)out_f16; (void)num_k_heads; (void)num_v_heads; (void)ba_row_stride;
    (void)total_state_slots;
#endif
}

}  // namespace w34_gdn


extern "C" int w34_gdn_launch_wy_prefill_f16(
    const w34_exec* ex,
    const w34_gdn_batch* batch,
    const void* qkv_conv_out,
    const void* q_in,
    const void* k_in,
    const void* v_in,
    const void* b_in,
    const void* a_in,
    const float* a_log,
    const float* dt_bias,
    float* ssm_state_pool,
    void* out_f16,
    int num_k_heads,
    int num_v_heads,
    int head_v_dim,
    int ba_row_stride,
    int total_state_slots) {
    const int nh = batch->num_reqs * num_v_heads;
            const int T = batch->total_tokens;
            const int R = batch->num_reqs;
            const int grid_upper_chunks =
                (T + 63) / 64 + R * (batch->max_checkpoints_per_req + 1);
            // Every checkpoint can split a tile; an initialized request also needs an H
            // snapshot for its first chunk. Reserve the descriptor upper bound so the
            // device planner cannot overflow after the host has selected split WY.
            const int needed_h_slots = grid_upper_chunks;
            auto align256 = [](size_t x) -> size_t { return (x + 255u) & ~size_t(255u); };
            const size_t off_hdr = 0;
            const size_t off_spans = off_hdr + 256u;
            const size_t off_chunks = off_spans + align256((size_t)R * sizeof(w34_gdn::WyReqSpan));
            const size_t max_chunk_descs = (size_t)grid_upper_chunks;
            const size_t off_gcum =
                off_chunks + align256(max_chunk_descs * sizeof(w34_gdn::WyChunkDesc));
            const size_t off_w =
                off_gcum + align256((size_t)T * (size_t)num_v_heads * sizeof(float));
            const size_t off_h =
                off_w + align256((size_t)T * (size_t)num_v_heads * 128u * sizeof(__half));
            const size_t h_slot_bytes = (size_t)num_v_heads * 128u * 128u * sizeof(__half);
            const bool can_use_split_wy =
                (ex->scratch_ptr != 0) &&
                (batch->max_checkpoints_per_req <= (T / 64) + 2) &&
                (ex->scratch_size >= off_h + (size_t)needed_h_slots * h_slot_bytes);

            if (can_use_split_wy) {
                static std::atomic<uint32_t> s_split_wy_attr_mask{0};
                int dev_id = 0;
                cudaGetDevice(&dev_id);
                const uint32_t dev_bit = (dev_id >= 0 && dev_id < 32) ? (1u << dev_id) : 0u;
                if ((s_split_wy_attr_mask.load(std::memory_order_relaxed) & dev_bit) == 0) {
                    if (cudaFuncSetAttribute(
                            w34_gdn::gdn_wy_stage1_wu_f16_kernel,
                            cudaFuncAttributeMaxDynamicSharedMemorySize,
                            w34_gdn::kWyStage1SmemBytes) != cudaSuccess ||
                        cudaFuncSetAttribute(
                            w34_gdn::gdn_wy_chunk64_prefill_f16_kernel,
                            cudaFuncAttributeMaxDynamicSharedMemorySize,
                            w34_gdn::kWy64SmemBytes) != cudaSuccess) {
                        return (int)cudaGetLastError();
                    }
                    s_split_wy_attr_mask.fetch_or(dev_bit, std::memory_order_relaxed);
                }
                char* ws = reinterpret_cast<char*>((uintptr_t)ex->scratch_ptr);
                auto* hdr = reinterpret_cast<w34_gdn::WyPlanHeader*>(ws + off_hdr);
                auto* spans = reinterpret_cast<w34_gdn::WyReqSpan*>(ws + off_spans);
                auto* chunks = reinterpret_cast<w34_gdn::WyChunkDesc*>(ws + off_chunks);
                auto* g_cum = reinterpret_cast<float*>(ws + off_gcum);
                auto* w_buf = reinterpret_cast<__half*>(ws + off_w);
                const size_t k_norm_bytes =
                    align256((size_t)T * (size_t)num_k_heads * 128u * sizeof(__half));
                const bool has_knorm_ws =
                    (ex->scratch_size >= off_h + k_norm_bytes + (size_t)needed_h_slots * h_slot_bytes);
                __half* k_norm_buf = has_knorm_ws ? reinterpret_cast<__half*>(ws + off_h) : nullptr;
                const size_t off_h_actual = has_knorm_ws ? (off_h + k_norm_bytes) : off_h;
                auto* h_chunks = reinterpret_cast<__half*>(ws + off_h_actual);
                const int max_h_slots = (int)((ex->scratch_size - off_h_actual) / h_slot_bytes);
                cudaStream_t stream = (cudaStream_t)ex->stream;

                w34_gdn::gdn_wy_plan_chunks_kernel<<<1, 1, 0, stream>>>(
                    *batch, hdr, spans, chunks, grid_upper_chunks, max_h_slots, total_state_slots);
                w34_gdn::gdn_wy_stage1_wu_f16_kernel<<<
                    dim3((unsigned)grid_upper_chunks, (unsigned)num_v_heads),
                    256,
                    w34_gdn::kWyStage1SmemBytes,
                    stream>>>(
                    hdr,
                    chunks,
                    (const __half*)qkv_conv_out,
                    (const __half*)k_in,
                    (const __half*)v_in,
                    (const __half*)b_in,
                    (const __half*)a_in,
                    a_log,
                    dt_bias,
                    g_cum,
                    k_norm_buf,
                    w_buf,
                    (__half*)out_f16,
                    num_k_heads,
                    num_v_heads,
                    ba_row_stride);
                w34_gdn::gdn_wy_stage2_deltah_f16_kernel<<<
                    dim3(4u, (unsigned)nh),
                    256,
                    w34_gdn::kWyStage2SmemBytes,
                    stream>>>(
                    *batch,
                    spans,
                    chunks,
                    (const __half*)qkv_conv_out,
                    (const __half*)k_in,
                    k_norm_buf,
                    g_cum,
                    w_buf,
                    (__half*)out_f16,
                    h_chunks,
                    ssm_state_pool,
                    num_k_heads,
                    num_v_heads,
                    total_state_slots);
                w34_gdn::gdn_wy_stage3_out_f16_kernel<<<
                    dim3((unsigned)grid_upper_chunks, (unsigned)num_v_heads),
                    256,
                    w34_gdn::kWyStage3SmemBytes,
                    stream>>>(
                    hdr,
                    chunks,
                    (const __half*)qkv_conv_out,
                    (const __half*)q_in,
                    (const __half*)k_in,
                    k_norm_buf,
                    g_cum,
                    h_chunks,
                    (__half*)out_f16,
                    num_k_heads,
                    num_v_heads);
            } else {
                if (cudaFuncSetAttribute(
                        w34_gdn::gdn_wy_chunk64_prefill_f16_kernel,
                        cudaFuncAttributeMaxDynamicSharedMemorySize,
                        w34_gdn::kWy64SmemBytes) != cudaSuccess) {
                    return (int)cudaGetLastError();
                }
                dim3 grid((unsigned)(head_v_dim / 64), (unsigned)nh);
                w34_gdn::gdn_wy_chunk64_prefill_f16_kernel
                    <<<grid, 256, w34_gdn::kWy64SmemBytes, (cudaStream_t)ex->stream>>>(
                        *batch,
                        (const __half*)qkv_conv_out,
                        (const __half*)q_in,
                        (const __half*)k_in,
                        (const __half*)v_in,
                        (const __half*)b_in,
                        (const __half*)a_in,
                        a_log,
                        dt_bias,
                        ssm_state_pool,
                        (__half*)out_f16,
                        num_k_heads,
                        num_v_heads,
                        ba_row_stride,
                        total_state_slots);
            }
    return (int)cudaGetLastError();
}
