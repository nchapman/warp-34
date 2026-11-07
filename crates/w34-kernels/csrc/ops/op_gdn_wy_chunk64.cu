// Fused Tensor-Core WY chunk64 GDN prefill fallback kernel (BT=64, BV=64, DK=128) for warp-34.
// Split from op_gdn_wy.cu for modularity and parallel nvcc compilation.
#include "../w34_cuda.h"

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <math.h>
#include <mma.h>

namespace w34_gdn_chunk64 {

namespace wmma = nvcuda::wmma;

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

// Fused Tensor-Core WY chunked GDN prefill kernel (`BT = 64, BV = 64, DK = 128`, 8 warps = 256
// threads, 1 single wave of 64 CTAs on 9B TP=1 / 48 CTAs on 27B TP=2 across 84 SMs, 0 bytes global
// scratch, STACK: 0).
// Donor math: vllm.cpp (50d425bec) `GdnChunkWUWmmaVecKernel` + `GdnChunkDeltaHRegKernel` +
// `GdnChunkOWmmaKernel`, fused across chunks so persistent state `H [128, 64]` remains in FP32
// `wmma::accumulator` registers across the entire sequence while checkpoints clamp `tile_len` to
// hit exact token offsets. Used as the 0-scratch fallback when split WY workspace is unavailable.
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

}  // namespace w34_gdn_chunk64

extern "C" int w34_gdn_init_wy_chunk64_f16() {
    if (cudaFuncSetAttribute(
            w34_gdn_chunk64::gdn_wy_chunk64_prefill_f16_kernel,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            w34_gdn_chunk64::kWy64SmemBytes) != cudaSuccess) {
        return (int)cudaGetLastError();
    }
    return 0;
}

extern "C" int w34_gdn_launch_wy_chunk64_prefill_f16(
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
    const int rc = w34_gdn_init_wy_chunk64_f16();
    if (rc != 0) return rc;
    const int nh = batch->num_reqs * num_v_heads;
    dim3 grid((unsigned)(head_v_dim / 64), (unsigned)nh);
    w34_gdn_chunk64::gdn_wy_chunk64_prefill_f16_kernel
        <<<grid, 256, w34_gdn_chunk64::kWy64SmemBytes, (cudaStream_t)ex->stream>>>(
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
    return (int)cudaGetLastError();
}
