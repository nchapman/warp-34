// Gated DeltaNet (GDN) Linear Attention kernels for warp-34.
// Donor: vllm.cpp (50d425bec) src/vt/cuda/cuda_gdn.cu, cross-checked against
// vLLM 0.31.0 FLA golden dumps.
//
// Invariants enforced (docs/contracts/state-transaction.md):
// - conv_state is FP16 [num_slots, conv_dim, kernel_size - 1]
// - ssm_state is FP32 in global memory [num_slots, num_v_heads, head_v_dim, head_k_dim]
// - Explicit state_in_indices, state_out_indices, spec_state_indices, and
//   checkpoint_dst_indices / checkpoint_token_offsets
// - Identical FP32 accumulation tree across one-shot prefill, BT=64 chunked prefill,
//   decode (q_len=1), and speculative verify (q_len=1+k) -> 0-ULP continuation &
//   rejection parity (Decisive Gate Test #2).
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

// Grid: (ceil(conv_dim / 256), num_reqs, z_tiles), Block: 256 threads.
// Processes one (request, channel) sliding window across the request's query tokens
// (parallelized across `gridDim.z` token chunks of >= 32 tokens during prefill),
// writing intermediate snapshots to spec_state_indices / checkpoint_dst_indices and
// final state to state_out_indices.
__global__ void gdn_causal_conv1d_f16_kernel(
    w34_gdn_batch batch,
    const __half* __restrict__ x_in,
    const __half* __restrict__ conv_weight,
    const __half* __restrict__ conv_bias,
    __half* __restrict__ conv_state_pool,
    __half* __restrict__ x_out,
    int conv_dim,
    int x_in_row_stride,
    int kernel_size,
    int total_state_slots,
    int silu_activation) {
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    const int req = blockIdx.y;
    if (c >= conv_dim || req >= batch.num_reqs) return;
    if (batch.valid_req_mask != nullptr && batch.valid_req_mask[req] == 0) return;

    const int lo = batch.query_start_loc[req];
    const int hi = batch.query_start_loc[req + 1];
    const int seqlen = hi - lo;
    if (seqlen <= 0) return;

    const int state_len = kernel_size - 1;  // 3 for kernel_size = 4
    int step_start = 0;
    int step_end = seqlen;
    if (gridDim.z > 1) {
        int steps_per_z = (seqlen + (int)gridDim.z - 1) / (int)gridDim.z;
        if (steps_per_z < 32) steps_per_z = 32;
        step_start = (int)blockIdx.z * steps_per_z;
        if (step_start >= seqlen) return;
        step_end = step_start + steps_per_z;
        if (step_end > seqlen) step_end = seqlen;
    }

    float hist[3] = {0.0f, 0.0f, 0.0f};
    const bool has_init = (batch.has_initial_state == nullptr) || (batch.has_initial_state[req] != 0);
    const int32_t slot_in = (batch.state_in_indices != nullptr) ? batch.state_in_indices[req] : -1;
    if (step_start == 0) {
        if (has_init && slot_in >= 0 && slot_in < total_state_slots) {
            const __half* s_in = conv_state_pool + ((size_t)slot_in * conv_dim + c) * state_len;
#pragma unroll
            for (int j = 0; j < 3; j++) {
                if (j < state_len) hist[j] = __half2float(s_in[j]);
            }
        }
    } else {
        // `step_start >= 32 >= state_len`, so history comes strictly from immutable `x_in`.
#pragma unroll
        for (int j = 0; j < 3; j++) {
            if (j < state_len) {
                const int src_tok = lo + step_start - state_len + j;
                hist[j] = __half2float(x_in[(size_t)src_tok * x_in_row_stride + c]);
            }
        }
    }

    float w_reg[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    const __half* w_row = conv_weight + (size_t)c * kernel_size;
#pragma unroll
    for (int j = 0; j < 4; j++) {
        if (j < kernel_size) w_reg[j] = __half2float(w_row[j]);
    }
    const float bias_val = (conv_bias != nullptr) ? __half2float(conv_bias[c]) : 0.0f;

    const bool has_conv_ckpts =
        (batch.checkpoint_dst_indices != nullptr && batch.checkpoint_token_offsets != nullptr &&
         batch.max_checkpoints_per_req > 0);
    const bool fast_two_conv_ckpts = has_conv_ckpts && (batch.max_checkpoints_per_req <= 2);
    int32_t conv_ckpt0_off = -1, conv_ckpt0_slot = -1;
    int32_t conv_ckpt1_off = -1, conv_ckpt1_slot = -1;
    if (fast_two_conv_ckpts) {
        const int base_e = req * batch.max_checkpoints_per_req;
        conv_ckpt0_off = batch.checkpoint_token_offsets[base_e];
        conv_ckpt0_slot = batch.checkpoint_dst_indices[base_e];
        if (batch.max_checkpoints_per_req == 2) {
            conv_ckpt1_off = batch.checkpoint_token_offsets[base_e + 1];
            conv_ckpt1_slot = batch.checkpoint_dst_indices[base_e + 1];
        }
    }

    int32_t last_spec_slot = -1;
    for (int step = step_start; step < step_end; step++) {
        const int tok = lo + step;
        const float x_val = __half2float(x_in[(size_t)tok * x_in_row_stride + c]);

        float acc = bias_val;
        if (kernel_size == 4) {
            // Native Triton conv (prefill and update) materializes each product
            // in FP16 before adding to the FP32 accumulator. Keep this rule
            // independent of query length to preserve exact split/state parity.
            acc += __half2float(__float2half_rn(w_reg[0] * hist[0]));
            acc += __half2float(__float2half_rn(w_reg[1] * hist[1]));
            acc += __half2float(__float2half_rn(w_reg[2] * hist[2]));
            acc += __half2float(__float2half_rn(w_reg[3] * x_val));
            hist[0] = hist[1];
            hist[1] = hist[2];
            hist[2] = x_val;
        } else {
            for (int j = 0; j < state_len; j++)
                acc += __half2float(__float2half_rn(w_reg[j] * hist[j]));
            acc += __half2float(__float2half_rn(w_reg[state_len] * x_val));
            for (int j = 0; j < state_len - 1; j++) hist[j] = hist[j + 1];
            if (state_len > 0) hist[state_len - 1] = x_val;
        }

        const float out_val = silu_activation ? silu_f32(acc) : acc;
        x_out[(size_t)tok * conv_dim + c] = __float2half(out_val);

        // Speculative per-step snapshot
        if (batch.spec_state_indices != nullptr && step < batch.max_verify_tokens) {
            const int32_t spec_slot = batch.spec_state_indices[req * batch.max_verify_tokens + step];
            if (spec_slot >= 0 && spec_slot < total_state_slots) {
                __half* s_spec = conv_state_pool + ((size_t)spec_slot * conv_dim + c) * state_len;
                for (int j = 0; j < state_len; j++) s_spec[j] = __float2half(hist[j]);
                if (step == seqlen - 1) last_spec_slot = spec_slot;
            }
        }

        // Chunk-aligned radix/ping-pong checkpoint snapshot
        if (fast_two_conv_ckpts) {
            if (conv_ckpt0_off == step + 1 && conv_ckpt0_slot >= 0 && conv_ckpt0_slot < total_state_slots) {
                __half* s_chk = conv_state_pool + ((size_t)conv_ckpt0_slot * conv_dim + c) * state_len;
                for (int j = 0; j < state_len; j++) s_chk[j] = __float2half(hist[j]);
            }
            if (conv_ckpt1_off == step + 1 && conv_ckpt1_slot >= 0 && conv_ckpt1_slot < total_state_slots) {
                __half* s_chk = conv_state_pool + ((size_t)conv_ckpt1_slot * conv_dim + c) * state_len;
                for (int j = 0; j < state_len; j++) s_chk[j] = __float2half(hist[j]);
            }
        } else if (has_conv_ckpts) {
            for (int e = 0; e < batch.max_checkpoints_per_req; e++) {
                const int32_t trig_off = batch.checkpoint_token_offsets[req * batch.max_checkpoints_per_req + e];
                if (trig_off == step + 1) {
                    const int32_t chk_slot = batch.checkpoint_dst_indices[req * batch.max_checkpoints_per_req + e];
                    if (chk_slot >= 0 && chk_slot < total_state_slots) {
                        __half* s_chk = conv_state_pool + ((size_t)chk_slot * conv_dim + c) * state_len;
                        for (int j = 0; j < state_len; j++) s_chk[j] = __float2half(hist[j]);
                    }
                }
            }
        }
    }

    // Write final state from `blockIdx.z == 0` AFTER reading `slot_in`, avoiding any
    // cross-block race when `slot_in == slot_out` under `gridDim.z > 1`.
    if (blockIdx.z == 0 && batch.state_out_indices != nullptr) {
        const int32_t slot_out = batch.state_out_indices[req];
        if (slot_out >= 0 && slot_out < total_state_slots && slot_out != last_spec_slot) {
            __half* s_out = conv_state_pool + ((size_t)slot_out * conv_dim + c) * state_len;
            if (step_end == seqlen) {
                for (int j = 0; j < state_len; j++) s_out[j] = __float2half(hist[j]);
            } else {
                // `step_end < seqlen` implies `seqlen > 32 >= state_len`.
                for (int j = 0; j < state_len; j++) {
                    const int final_tok = lo + seqlen - state_len + j;
                    s_out[j] = x_in[(size_t)final_tok * x_in_row_stride + c];
                }
            }
        }
    }
}

// Standard four-tap prefill: paired channels keep the donor's FP16 products while
// widening memory transactions and halving multiply/conversion instruction count.
__global__ void gdn_causal_conv1d_pair_f16_kernel(
    w34_gdn_batch batch, const __half* __restrict__ x_in,
    const __half* __restrict__ weights, __half* __restrict__ state,
    __half* __restrict__ out, int channels, int stride, int slots, int activate) {
    const int c = 2 * (blockIdx.x * blockDim.x + threadIdx.x);
    const int req = blockIdx.y;
    if (c >= channels || req >= batch.num_reqs) return;
    if (batch.valid_req_mask != nullptr && batch.valid_req_mask[req] == 0) return;
    const int lo = batch.query_start_loc[req];
    const int len = batch.query_start_loc[req + 1] - lo;
    if (len <= 0) return;
    const int span = max(32, (len + (int)gridDim.z - 1) / (int)gridDim.z);
    const int begin = blockIdx.z * span;
    if (begin >= len) return;
    const int end = min(len, begin + span);
    __half2 hist[3];
    const int input_slot = batch.state_in_indices ? batch.state_in_indices[req] : -1;
    const bool initialized = (!batch.has_initial_state || batch.has_initial_state[req]) &&
        input_slot >= 0 && input_slot < slots;
#pragma unroll
    for (int j = 0; j < 3; ++j) {
        if (begin > 0) {
            hist[j] = *reinterpret_cast<const __half2*>(x_in + (size_t)(lo + begin - 3 + j) * stride + c);
        } else if (initialized) {
            const __half* src = state + ((size_t)input_slot * channels + c) * 3;
            hist[j] = __halves2half2(src[j], src[3 + j]);
        } else {
            hist[j] = __float2half2_rn(0.0f);
        }
    }
    __half2 w[4];
#pragma unroll
    for (int j = 0; j < 4; ++j) w[j] = __halves2half2(weights[c * 4 + j], weights[(c + 1) * 4 + j]);
    int ck_offset[2] = {-1, -1};
    int ck_slot[2] = {-1, -1};
    if (batch.checkpoint_dst_indices && batch.checkpoint_token_offsets) {
#pragma unroll
        for (int j = 0; j < 2; ++j) {
            if (j < batch.max_checkpoints_per_req) {
                const int i = req * batch.max_checkpoints_per_req + j;
                ck_offset[j] = batch.checkpoint_token_offsets[i];
                ck_slot[j] = batch.checkpoint_dst_indices[i];
            }
        }
    }
    for (int step = begin; step < end; ++step) {
        const __half2 x = *reinterpret_cast<const __half2*>(x_in + (size_t)(lo + step) * stride + c);
        float2 acc = make_float2(0.0f, 0.0f);
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            const float2 product = __half22float2(__hmul2(w[j], j < 3 ? hist[j] : x));
            acc.x += product.x;
            acc.y += product.y;
        }
        hist[0] = hist[1]; hist[1] = hist[2]; hist[2] = x;
        *reinterpret_cast<__half2*>(out + (size_t)(lo + step) * channels + c) =
            __floats2half2_rn(activate ? silu_f32(acc.x) : acc.x,
                             activate ? silu_f32(acc.y) : acc.y);
#pragma unroll
        for (int j = 0; j < 2; ++j) {
            if (ck_offset[j] == step + 1 && ck_slot[j] >= 0 && ck_slot[j] < slots) {
                __half* dst = state + ((size_t)ck_slot[j] * channels + c) * 3;
#pragma unroll
                for (int k = 0; k < 3; ++k) {
                    dst[k] = __low2half(hist[k]); dst[3 + k] = __high2half(hist[k]);
                }
            }
        }
    }
    // Only the first tile reads initial state, so it alone may overwrite that slot.
    if (blockIdx.z == 0 && batch.state_out_indices) {
        const int slot = batch.state_out_indices[req];
        if (slot >= 0 && slot < slots) {
            __half* dst = state + ((size_t)slot * channels + c) * 3;
#pragma unroll
            for (int k = 0; k < 3; ++k) {
                const __half2 value = end == len ? hist[k] :
                    *reinterpret_cast<const __half2*>(x_in + (size_t)(lo + len - 3 + k) * stride + c);
                dst[k] = __low2half(value); dst[3 + k] = __high2half(value);
            }
        }
    }
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

__device__ __forceinline__ float4 ldg_cs_f4(const float4* p) {
    float4 v;
    asm volatile("ld.global.cs.v4.f32 {%0, %1, %2, %3}, [%4];"
                 : "=f"(v.x), "=f"(v.y), "=f"(v.z), "=f"(v.w)
                 : "l"(p));
    return v;
}

__device__ __forceinline__ void stg_cs_f4(float4* p, float4 v) {
    asm volatile("st.global.cs.v4.f32 [%0], {%1, %2, %3, %4};"
                 :
                 : "l"(p), "f"(v.x), "f"(v.y), "f"(v.z), "f"(v.w));
}

template <int ROWS>
__device__ __forceinline__ void gdn_step_rows_warp(
    float4 (&rr)[ROWS],
    const float4& bq4,
    const float4& bk4,
    float decay,
    float beta,
    const float (&vv)[ROWS],
    __half* __restrict__ out_hv,
    int vrow_base,
    int head_v_dim,
    int lane) {
    float dot[ROWS];
#pragma unroll
    for (int r = 0; r < ROWS; ++r) {
        rr[r].x = __fmul_rn(rr[r].x, decay);
        rr[r].y = __fmul_rn(rr[r].y, decay);
        rr[r].z = __fmul_rn(rr[r].z, decay);
        rr[r].w = __fmul_rn(rr[r].w, decay);
        dot[r] = __fadd_rn(__fmaf_rn(rr[r].x, bk4.x, __fmul_rn(rr[r].y, bk4.y)),
                           __fmaf_rn(rr[r].z, bk4.z, __fmul_rn(rr[r].w, bk4.w)));
    }
#pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
#pragma unroll
        for (int r = 0; r < ROWS; ++r) {
            dot[r] = __fadd_rn(dot[r], __shfl_xor_sync(0xffffffffu, dot[r], off));
        }
    }
    float o[ROWS];
#pragma unroll
    for (int r = 0; r < ROWS; ++r) {
        const float vp = __fmul_rn(__fsub_rn(vv[r], dot[r]), beta);
        rr[r].x = __fmaf_rn(vp, bk4.x, rr[r].x);
        rr[r].y = __fmaf_rn(vp, bk4.y, rr[r].y);
        rr[r].z = __fmaf_rn(vp, bk4.z, rr[r].z);
        rr[r].w = __fmaf_rn(vp, bk4.w, rr[r].w);
        o[r] = __fadd_rn(__fmaf_rn(rr[r].x, bq4.x, __fmul_rn(rr[r].y, bq4.y)),
                         __fmaf_rn(rr[r].z, bq4.z, __fmul_rn(rr[r].w, bq4.w)));
    }
#pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
#pragma unroll
        for (int r = 0; r < ROWS; ++r) {
            o[r] = __fadd_rn(o[r], __shfl_xor_sync(0xffffffffu, o[r], off));
        }
    }
    if (lane == 0) {
        if constexpr ((ROWS % 4) == 0) {
            if (((vrow_base & 3) == 0) && (vrow_base + ROWS <= head_v_dim) &&
                ((((uintptr_t)(out_hv + vrow_base)) & 7u) == 0u)) {
#pragma unroll
                for (int r = 0; r < ROWS; r += 4) {
                    const __half2 h01 = __halves2half2(__float2half(o[r]), __float2half(o[r + 1]));
                    const __half2 h23 = __halves2half2(__float2half(o[r + 2]), __float2half(o[r + 3]));
                    uint2 packed;
                    packed.x = *reinterpret_cast<const uint32_t*>(&h01);
                    packed.y = *reinterpret_cast<const uint32_t*>(&h23);
                    *reinterpret_cast<uint2*>(out_hv + vrow_base + r) = packed;
                }
                return;
            }
        }
#pragma unroll
        for (int r = 0; r < ROWS; ++r) {
            if ((vrow_base + r) < head_v_dim) {
                out_hv[vrow_base + r] = __float2half(o[r]);
            }
        }
    }
}

template <int ROWS>
__device__ __forceinline__ void gdn_step_rows_smem_v_warp(
    float4 (&rr)[ROWS],
    const float4& bq4,
    const float4& bk4,
    float decay,
    float beta,
    const float* __restrict__ s_v_row,
    __half* out_hv,
    int vrow_base,
    int head_v_dim,
    int lane) {
    float dot[ROWS];
#pragma unroll
    for (int r = 0; r < ROWS; ++r) {
        rr[r].x = __fmul_rn(rr[r].x, decay);
        rr[r].y = __fmul_rn(rr[r].y, decay);
        rr[r].z = __fmul_rn(rr[r].z, decay);
        rr[r].w = __fmul_rn(rr[r].w, decay);
        dot[r] = __fadd_rn(__fmaf_rn(rr[r].x, bk4.x, __fmul_rn(rr[r].y, bk4.y)),
                           __fmaf_rn(rr[r].z, bk4.z, __fmul_rn(rr[r].w, bk4.w)));
    }
#pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
#pragma unroll
        for (int r = 0; r < ROWS; ++r) {
            dot[r] = __fadd_rn(dot[r], __shfl_xor_sync(0xffffffffu, dot[r], off));
        }
    }
    float o[ROWS];
#pragma unroll
    for (int r = 0; r < ROWS; ++r) {
        const float vp = __fmul_rn(__fsub_rn(s_v_row[r], dot[r]), beta);
        rr[r].x = __fmaf_rn(vp, bk4.x, rr[r].x);
        rr[r].y = __fmaf_rn(vp, bk4.y, rr[r].y);
        rr[r].z = __fmaf_rn(vp, bk4.z, rr[r].z);
        rr[r].w = __fmaf_rn(vp, bk4.w, rr[r].w);
        o[r] = __fadd_rn(__fmaf_rn(rr[r].x, bq4.x, __fmul_rn(rr[r].y, bq4.y)),
                         __fmaf_rn(rr[r].z, bq4.z, __fmul_rn(rr[r].w, bq4.w)));
    }
#pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
#pragma unroll
        for (int r = 0; r < ROWS; ++r) {
            o[r] = __fadd_rn(o[r], __shfl_xor_sync(0xffffffffu, o[r], off));
        }
    }
    if (lane == 0) {
        if constexpr ((ROWS % 4) == 0) {
            if ((vrow_base + ROWS <= head_v_dim) && ((vrow_base & 3) == 0) &&
                ((((uintptr_t)(out_hv + vrow_base)) & 7u) == 0u)) {
#pragma unroll
                for (int g = 0; g < ROWS / 4; ++g) {
                    const __half2 h01 = __floats2half2_rn(o[4 * g + 0], o[4 * g + 1]);
                    const __half2 h23 = __floats2half2_rn(o[4 * g + 2], o[4 * g + 3]);
                    uint2 pack;
                    pack.x = *reinterpret_cast<const uint32_t*>(&h01);
                    pack.y = *reinterpret_cast<const uint32_t*>(&h23);
                    *reinterpret_cast<uint2*>(out_hv + vrow_base + 4 * g) = pack;
                }
                return;
            }
        }
#pragma unroll
        for (int r = 0; r < ROWS; ++r) {
            const int vr = vrow_base + r;
            if (vr < head_v_dim) {
                out_hv[vr] = __float2half(o[r]);
            }
        }
    }
}

template <int ROWS>
__device__ __forceinline__ void gdn_store_rows_warp(
    float* __restrict__ ssm_state_pool,
    int32_t slot,
    int num_v_heads,
    int hv,
    size_t head_state_elems,
    int vrow_base,
    int head_v_dim,
    int lane,
    const float4 (&rr)[ROWS]) {
    float* head_dst = ssm_state_pool + ((size_t)slot * num_v_heads + hv) * head_state_elems;
#pragma unroll
    for (int r = 0; r < ROWS; ++r) {
        const int vr = vrow_base + r;
        if (vr < head_v_dim) {
            reinterpret_cast<float4*>(head_dst + (size_t)vr * 128)[lane] = rr[r];
        }
    }
}

// Zero-smem, zero-barrier warp-coalesced GDN decode kernel (`ROWS = 4` -> `BV = 16` at small batch,
// `ROWS = 8` -> `BV = 32` when `num_reqs * num_v_heads >= 96` to halve redundant Q/K norms and
// GDDR6X R/W turnaround at B>=2).
template <int ROWS, int MIN_BLOCKS>
__global__ __launch_bounds__(128, MIN_BLOCKS) void gdn_recurrent_decode_f16_kernel(
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
    int head_k_dim,
    int num_v_heads,
    int head_v_dim,
    int ba_row_stride,
    int total_state_slots) {
    constexpr int BV = ROWS * 4;
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
    const int vrow_base = vbase + wid * ROWS;

    float4 rr[ROWS];
    const bool has_init = (batch.has_initial_state == nullptr) || (batch.has_initial_state[req] != 0);
    const int32_t slot_in = (batch.state_in_indices != nullptr) ? batch.state_in_indices[req] : -1;
    const size_t head_state_elems = (size_t)head_v_dim * 128u;
    if (has_init && slot_in >= 0 && slot_in < total_state_slots) {
        const float* head_in = ssm_state_pool + ((size_t)slot_in * num_v_heads + hv) * head_state_elems;
#pragma unroll
        for (int r = 0; r < ROWS; ++r) {
            const int vr = vrow_base + r;
            rr[r] = (vr < head_v_dim)
                ? ldg_cs_f4(reinterpret_cast<const float4*>(head_in + (size_t)vr * 128) + lane)
                : make_float4(0.0f, 0.0f, 0.0f, 0.0f);
        }
    } else {
#pragma unroll
        for (int r = 0; r < ROWS; ++r) {
            rr[r] = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
        }
    }

    const float scale = rsqrtf((float)head_k_dim);
    const float neg_exp_a = -expf(a_log[hv]);
    const float dt_bias_hv = dt_bias[hv];
    const int k_total_dim = num_k_heads * 128;
    const int v_total_dim = num_v_heads * head_v_dim;
    const int qkv_stride = 2 * k_total_dim + v_total_dim;
    const bool has_checkpoints =
        (batch.checkpoint_dst_indices != nullptr && batch.checkpoint_token_offsets != nullptr &&
         batch.max_checkpoints_per_req > 0);

    int32_t last_spec_slot = -1;
    for (int step = 0; step < seqlen; ++step) {
        const int tok = lo + step;
        const __half* q_tok_hk = (qkv_conv_out != nullptr)
            ? (qkv_conv_out + (size_t)tok * qkv_stride + (size_t)hk * 128)
            : (q_in + ((size_t)tok * num_k_heads + hk) * 128);
        const __half* k_tok_hk = (qkv_conv_out != nullptr)
            ? (qkv_conv_out + (size_t)tok * qkv_stride + k_total_dim + (size_t)hk * 128)
            : (k_in + ((size_t)tok * num_k_heads + hk) * 128);
        const __half* v_tok_hv = (qkv_conv_out != nullptr)
            ? (qkv_conv_out + (size_t)tok * qkv_stride + 2 * k_total_dim + (size_t)hv * head_v_dim)
            : (v_in + ((size_t)tok * num_v_heads + hv) * head_v_dim);

        float4 bq4, bk4;
        gdn_norm_qk_warp(q_tok_hk, k_tok_hk, lane, scale, bq4, bk4);

        const float a_val = __half2float(a_in[(size_t)tok * ba_row_stride + hv]);
        const float b_val = __half2float(b_in[(size_t)tok * ba_row_stride + hv]);
        const float decay = expf(neg_exp_a * softplus_f32(a_val + dt_bias_hv));
        const float beta = 1.0f / (1.0f + expf(-b_val));
        float vv[ROWS];
#pragma unroll
        for (int r = 0; r < ROWS; ++r) {
            const int vr = vrow_base + r;
            vv[r] = (vr < head_v_dim) ? __half2float(v_tok_hv[vr]) : 0.0f;
        }
        gdn_step_rows_warp<ROWS>(
            rr, bq4, bk4, decay, beta, vv,
            out_f16 + ((size_t)tok * num_v_heads + hv) * head_v_dim,
            vrow_base, head_v_dim, lane);

        if (batch.spec_state_indices != nullptr && step < batch.max_verify_tokens) {
            const int32_t spec_slot = batch.spec_state_indices[req * batch.max_verify_tokens + step];
            if (spec_slot >= 0 && spec_slot < total_state_slots) {
                gdn_store_rows_warp<ROWS>(ssm_state_pool, spec_slot, num_v_heads, hv, head_state_elems,
                                          vrow_base, head_v_dim, lane, rr);
                if (step == seqlen - 1) last_spec_slot = spec_slot;
            }
        }
        if (has_checkpoints) {
            for (int e = 0; e < batch.max_checkpoints_per_req; e++) {
                const int32_t trig_off = batch.checkpoint_token_offsets[req * batch.max_checkpoints_per_req + e];
                if (trig_off == step + 1) {
                    const int32_t chk_slot = batch.checkpoint_dst_indices[req * batch.max_checkpoints_per_req + e];
                    if (chk_slot >= 0 && chk_slot < total_state_slots) {
                        gdn_store_rows_warp<ROWS>(ssm_state_pool, chk_slot, num_v_heads, hv, head_state_elems,
                                                  vrow_base, head_v_dim, lane, rr);
                    }
                }
            }
        }
    }

    if (batch.state_out_indices != nullptr) {
        const int32_t slot_out = batch.state_out_indices[req];
        if (slot_out >= 0 && slot_out < total_state_slots && slot_out != last_spec_slot) {
            gdn_store_rows_warp<ROWS>(ssm_state_pool, slot_out, num_v_heads, hv, head_state_elems,
                                      vrow_base, head_v_dim, lane, rr);
        }
    }
}

// Multi-step warp-parallel GDN prefill/verify kernel (`ROWS = 4` -> `BV = 16`, or `ROWS = 8` ->
// `BV = 32` when `num_reqs * num_v_heads >= 96`; `TILE_S = 16` steps normalized cooperatively per
// barrier in bank-conflict-free SoA shared memory).
// Shares the exact `gdn_norm_qk_warp` + `gdn_step_rows_warp` helpers with decode for 0-ULP parity.
template <int ROWS, int MIN_BLOCKS>
__global__ __launch_bounds__(128, MIN_BLOCKS) void gdn_recurrent_prefill_f16_kernel(
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
    int head_k_dim,
    int num_v_heads,
    int head_v_dim,
    int ba_row_stride,
    int total_state_slots) {
    constexpr int BV = ROWS * 4;
    constexpr int TILE_S = 16;
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
    const int vrow_base = vbase + wid * ROWS;

    float4 rr[ROWS];
    const bool has_init = (batch.has_initial_state == nullptr) || (batch.has_initial_state[req] != 0);
    const int32_t slot_in = (batch.state_in_indices != nullptr) ? batch.state_in_indices[req] : -1;
    const size_t head_state_elems = (size_t)head_v_dim * 128u;
    if (has_init && slot_in >= 0 && slot_in < total_state_slots) {
        const float* head_in = ssm_state_pool + ((size_t)slot_in * num_v_heads + hv) * head_state_elems;
#pragma unroll
        for (int r = 0; r < ROWS; ++r) {
            const int vr = vrow_base + r;
            rr[r] = (vr < head_v_dim)
                ? ldg_cs_f4(reinterpret_cast<const float4*>(head_in + (size_t)vr * 128) + lane)
                : make_float4(0.0f, 0.0f, 0.0f, 0.0f);
        }
    } else {
#pragma unroll
        for (int r = 0; r < ROWS; ++r) {
            rr[r] = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
        }
    }

    const float scale = rsqrtf((float)head_k_dim);
    const float neg_exp_a = -expf(a_log[hv]);
    const float dt_bias_hv = dt_bias[hv];
    const int k_total_dim = num_k_heads * 128;
    const int v_total_dim = num_v_heads * head_v_dim;
    const int qkv_stride = 2 * k_total_dim + v_total_dim;

    __shared__ float s_bq_x[TILE_S][32];
    __shared__ float s_bq_y[TILE_S][32];
    __shared__ float s_bq_z[TILE_S][32];
    __shared__ float s_bq_w[TILE_S][32];
    __shared__ float s_bk_x[TILE_S][32];
    __shared__ float s_bk_y[TILE_S][32];
    __shared__ float s_bk_z[TILE_S][32];
    __shared__ float s_bk_w[TILE_S][32];
    __shared__ float s_v[TILE_S][BV];
    __shared__ float s_decay[TILE_S];
    __shared__ float s_beta[TILE_S];

    const bool has_checkpoints =
        (batch.checkpoint_dst_indices != nullptr && batch.checkpoint_token_offsets != nullptr &&
         batch.max_checkpoints_per_req > 0);
    const bool fast_two_ckpts = has_checkpoints && (batch.max_checkpoints_per_req <= 2);
    int32_t ckpt0_off = -1, ckpt0_slot = -1;
    int32_t ckpt1_off = -1, ckpt1_slot = -1;
    if (fast_two_ckpts) {
        const int base_e = req * batch.max_checkpoints_per_req;
        ckpt0_off = batch.checkpoint_token_offsets[base_e];
        ckpt0_slot = batch.checkpoint_dst_indices[base_e];
        if (batch.max_checkpoints_per_req == 2) {
            ckpt1_off = batch.checkpoint_token_offsets[base_e + 1];
            ckpt1_slot = batch.checkpoint_dst_indices[base_e + 1];
        }
    }

    int32_t last_spec_slot = -1;
    for (int base_s = 0; base_s < seqlen; base_s += TILE_S) {
        if (base_s > 0) __syncthreads();
        const int tile_len = (seqlen - base_s < TILE_S) ? (seqlen - base_s) : TILE_S;
        for (int s_norm = wid; s_norm < tile_len; s_norm += 4) {
            const int tok = lo + base_s + s_norm;
            const __half* q_tok_hk = (qkv_conv_out != nullptr)
                ? (qkv_conv_out + (size_t)tok * qkv_stride + (size_t)hk * 128)
                : (q_in + ((size_t)tok * num_k_heads + hk) * 128);
            const __half* k_tok_hk = (qkv_conv_out != nullptr)
                ? (qkv_conv_out + (size_t)tok * qkv_stride + k_total_dim + (size_t)hk * 128)
                : (k_in + ((size_t)tok * num_k_heads + hk) * 128);
            const __half* v_tok_hv = (qkv_conv_out != nullptr)
                ? (qkv_conv_out + (size_t)tok * qkv_stride + 2 * k_total_dim + (size_t)hv * head_v_dim)
                : (v_in + ((size_t)tok * num_v_heads + hv) * head_v_dim);

            float4 bq4, bk4;
            gdn_norm_qk_warp(q_tok_hk, k_tok_hk, lane, scale, bq4, bk4);
            s_bq_x[s_norm][lane] = bq4.x;
            s_bq_y[s_norm][lane] = bq4.y;
            s_bq_z[s_norm][lane] = bq4.z;
            s_bq_w[s_norm][lane] = bq4.w;
            s_bk_x[s_norm][lane] = bk4.x;
            s_bk_y[s_norm][lane] = bk4.y;
            s_bk_z[s_norm][lane] = bk4.z;
            s_bk_w[s_norm][lane] = bk4.w;
#pragma unroll
            for (int c = lane; c < BV; c += 32) {
                const int vr = vbase + c;
                s_v[s_norm][c] = (vr < head_v_dim) ? __half2float(v_tok_hv[vr]) : 0.0f;
            }
            if (lane == 0) {
                const float a_val = __half2float(a_in[(size_t)tok * ba_row_stride + hv]);
                const float b_val = __half2float(b_in[(size_t)tok * ba_row_stride + hv]);
                s_decay[s_norm] = expf(neg_exp_a * softplus_f32(a_val + dt_bias_hv));
                s_beta[s_norm] = 1.0f / (1.0f + expf(-b_val));
            }
        }
        __syncthreads();

        const bool tile_has_spec =
            (batch.spec_state_indices != nullptr && base_s < batch.max_verify_tokens);
        bool tile_has_ckpt = false;
        if (fast_two_ckpts) {
            tile_has_ckpt = (ckpt0_off > base_s && ckpt0_off <= base_s + tile_len) ||
                            (ckpt1_off > base_s && ckpt1_off <= base_s + tile_len);
        } else if (has_checkpoints) {
            for (int e = 0; e < batch.max_checkpoints_per_req; e++) {
                const int32_t trig_off =
                    batch.checkpoint_token_offsets[req * batch.max_checkpoints_per_req + e];
                if (trig_off > base_s && trig_off <= base_s + tile_len) {
                    tile_has_ckpt = true;
                    break;
                }
            }
        }

        if (!tile_has_spec && !tile_has_ckpt) {
            for (int s = 0; s < tile_len; ++s) {
                const int tok = lo + base_s + s;
                const float4 bq4 = make_float4(
                    s_bq_x[s][lane], s_bq_y[s][lane], s_bq_z[s][lane], s_bq_w[s][lane]);
                const float4 bk4 = make_float4(
                    s_bk_x[s][lane], s_bk_y[s][lane], s_bk_z[s][lane], s_bk_w[s][lane]);
                gdn_step_rows_smem_v_warp<ROWS>(
                    rr, bq4, bk4, s_decay[s], s_beta[s], &s_v[s][wid * ROWS],
                    out_f16 + ((size_t)tok * num_v_heads + hv) * head_v_dim,
                    vrow_base, head_v_dim, lane);
            }
        } else {
            for (int s = 0; s < tile_len; ++s) {
                const int step = base_s + s;
                const int tok = lo + step;
                const float4 bq4 = make_float4(
                    s_bq_x[s][lane], s_bq_y[s][lane], s_bq_z[s][lane], s_bq_w[s][lane]);
                const float4 bk4 = make_float4(
                    s_bk_x[s][lane], s_bk_y[s][lane], s_bk_z[s][lane], s_bk_w[s][lane]);
                gdn_step_rows_smem_v_warp<ROWS>(
                    rr, bq4, bk4, s_decay[s], s_beta[s], &s_v[s][wid * ROWS],
                    out_f16 + ((size_t)tok * num_v_heads + hv) * head_v_dim,
                    vrow_base, head_v_dim, lane);

                if (tile_has_spec && step < batch.max_verify_tokens) {
                    const int32_t spec_slot =
                        batch.spec_state_indices[req * batch.max_verify_tokens + step];
                    if (spec_slot >= 0 && spec_slot < total_state_slots) {
                        gdn_store_rows_warp<ROWS>(ssm_state_pool, spec_slot, num_v_heads, hv,
                                                  head_state_elems, vrow_base, head_v_dim, lane, rr);
                        if (step == seqlen - 1) last_spec_slot = spec_slot;
                    }
                }
                if (tile_has_ckpt) {
                    if (fast_two_ckpts) {
                        if (ckpt0_off == step + 1 && ckpt0_slot >= 0 &&
                            ckpt0_slot < total_state_slots) {
                            gdn_store_rows_warp<ROWS>(ssm_state_pool, ckpt0_slot, num_v_heads, hv,
                                                      head_state_elems, vrow_base, head_v_dim, lane,
                                                      rr);
                        }
                        if (ckpt1_off == step + 1 && ckpt1_slot >= 0 &&
                            ckpt1_slot < total_state_slots) {
                            gdn_store_rows_warp<ROWS>(ssm_state_pool, ckpt1_slot, num_v_heads, hv,
                                                      head_state_elems, vrow_base, head_v_dim, lane,
                                                      rr);
                        }
                    } else {
                        for (int e = 0; e < batch.max_checkpoints_per_req; e++) {
                            const int32_t trig_off =
                                batch.checkpoint_token_offsets[req * batch.max_checkpoints_per_req + e];
                            if (trig_off == step + 1) {
                                const int32_t chk_slot =
                                    batch.checkpoint_dst_indices[req * batch.max_checkpoints_per_req + e];
                                if (chk_slot >= 0 && chk_slot < total_state_slots) {
                                    gdn_store_rows_warp<ROWS>(ssm_state_pool, chk_slot, num_v_heads,
                                                              hv, head_state_elems, vrow_base,
                                                              head_v_dim, lane, rr);
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    if (batch.state_out_indices != nullptr) {
        const int32_t slot_out = batch.state_out_indices[req];
        if (slot_out >= 0 && slot_out < total_state_slots && slot_out != last_spec_slot) {
            gdn_store_rows_warp<ROWS>(ssm_state_pool, slot_out, num_v_heads, hv, head_state_elems,
                                      vrow_base, head_v_dim, lane, rr);
        }
    }
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
    int total_state_slots);

extern "C" int w34_gdn_causal_conv1d_strided_f16(
    const w34_exec* ex,
    const w34_gdn_batch* batch,
    const void* x_in,
    const void* conv_weight,
    const void* conv_bias,
    void* conv_state_pool,
    void* x_out,
    int conv_dim,
    int x_in_row_stride,
    int kernel_size,
    int total_state_slots,
    int silu_activation) {
    if (ex == nullptr || batch == nullptr || x_in == nullptr || conv_weight == nullptr ||
        conv_state_pool == nullptr || x_out == nullptr) {
        return (int)cudaErrorInvalidValue;
    }
    if (batch->num_reqs <= 0 || batch->total_tokens <= 0 || conv_dim <= 0) return 0;
    if (x_in_row_stride < conv_dim) return (int)cudaErrorInvalidValue;
    unsigned z_tiles = 1u;
    if (batch->spec_state_indices == nullptr &&
        x_in != x_out &&
        kernel_size >= 1 && kernel_size <= 4 &&
        batch->total_tokens > batch->num_reqs * 32) {
        const int avg_t = (batch->total_tokens + batch->num_reqs - 1) / batch->num_reqs;
        int cand_z = (avg_t + 31) / 32;
        if (cand_z > 64) cand_z = 64;
        if (cand_z > 1) z_tiles = (unsigned)cand_z;
    }
    if (kernel_size == 4 && conv_bias == nullptr && batch->spec_state_indices == nullptr &&
        x_in != x_out && (((uintptr_t)x_in | (uintptr_t)x_out) & 3u) == 0u &&
        (conv_dim % 2) == 0 && (x_in_row_stride % 2) == 0 &&
        batch->max_checkpoints_per_req >= 0 && batch->max_checkpoints_per_req <= 2 &&
        batch->total_tokens >= 256 && batch->total_tokens >= batch->num_reqs * 64) {
        dim3 paired_grid((unsigned)((conv_dim + 511) / 512), (unsigned)batch->num_reqs, z_tiles);
        w34_gdn::gdn_causal_conv1d_pair_f16_kernel<<<paired_grid, 256, 0, (cudaStream_t)ex->stream>>>(
            *batch, (const __half*)x_in, (const __half*)conv_weight,
            (__half*)conv_state_pool, (__half*)x_out, conv_dim, x_in_row_stride,
            total_state_slots, silu_activation);
        return (int)cudaGetLastError();
    }
    dim3 grid((unsigned)((conv_dim + 255) / 256), (unsigned)batch->num_reqs, z_tiles);
    w34_gdn::gdn_causal_conv1d_f16_kernel<<<grid, 256, 0, (cudaStream_t)ex->stream>>>(
        *batch, (const __half*)x_in, (const __half*)conv_weight,
        (const __half*)conv_bias, (__half*)conv_state_pool, (__half*)x_out,
        conv_dim, x_in_row_stride, kernel_size, total_state_slots, silu_activation);
    return (int)cudaGetLastError();
}

extern "C" int w34_gdn_causal_conv1d_f16(
    const w34_exec* ex,
    const w34_gdn_batch* batch,
    const void* x_in,
    const void* conv_weight,
    const void* conv_bias,
    void* conv_state_pool,
    void* x_out,
    int conv_dim,
    int kernel_size,
    int total_state_slots,
    int silu_activation) {
    return w34_gdn_causal_conv1d_strided_f16(
        ex, batch, x_in, conv_weight, conv_bias, conv_state_pool, x_out,
        conv_dim, conv_dim, kernel_size, total_state_slots, silu_activation);
}

extern "C" int w34_gdn_recurrent_strided_f16(
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
    int head_k_dim,
    int num_v_heads,
    int head_v_dim,
    int ba_row_stride,
    int total_state_slots) {
    if (ex == nullptr || batch == nullptr || b_in == nullptr || a_in == nullptr ||
        a_log == nullptr || dt_bias == nullptr || ssm_state_pool == nullptr || out_f16 == nullptr) {
        return (int)cudaErrorInvalidValue;
    }
    if (qkv_conv_out == nullptr && (q_in == nullptr || k_in == nullptr || v_in == nullptr)) {
        return (int)cudaErrorInvalidValue;
    }
    if (batch->num_reqs <= 0 || batch->total_tokens <= 0) return 0;
    if (num_k_heads <= 0 || num_v_heads <= 0 || (num_v_heads % num_k_heads) != 0) {
        return (int)cudaErrorInvalidValue;
    }
    if (head_k_dim != 128 || ba_row_stride < num_v_heads) return (int)cudaErrorInvalidValue;

    if (batch->total_tokens <= batch->num_reqs * 2) {
        const int nh = batch->num_reqs * num_v_heads;
        if (nh >= 96 && head_v_dim >= 32) {
            constexpr int ROWS = 8;
            constexpr int BV = ROWS * 4;
            const int v_tiles = (head_v_dim + BV - 1) / BV;
            dim3 grid((unsigned)v_tiles, (unsigned)nh);
            w34_gdn::gdn_recurrent_decode_f16_kernel<ROWS, 6>
                <<<grid, 128, 0, (cudaStream_t)ex->stream>>>(
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
                    head_k_dim,
                    num_v_heads,
                    head_v_dim,
                    ba_row_stride,
                    total_state_slots);
        } else {
            constexpr int ROWS = 4;
            constexpr int BV = ROWS * 4;
            const int v_tiles = (head_v_dim + BV - 1) / BV;
            dim3 grid((unsigned)v_tiles, (unsigned)nh);
            w34_gdn::gdn_recurrent_decode_f16_kernel<ROWS, 12>
                <<<grid, 128, 0, (cudaStream_t)ex->stream>>>(
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
                    head_k_dim,
                    num_v_heads,
                    head_v_dim,
                    ba_row_stride,
                    total_state_slots);
        }
    } else {
        const int nh = batch->num_reqs * num_v_heads;
        w34_device_props dev_props{};
        const int sm_count =
            (w34_get_current_device_props(&dev_props) == 0 && dev_props.multi_processor_count > 0)
                ? dev_props.multi_processor_count
                : 84;
        if (batch->spec_state_indices == nullptr &&
            head_k_dim == 128 &&
            head_v_dim == 128 &&
            batch->total_tokens >= 256 &&
            batch->total_tokens >= batch->num_reqs * 64) {
            return w34_gdn_launch_wy_prefill_f16(
                ex,
                batch,
                qkv_conv_out,
                q_in,
                k_in,
                v_in,
                b_in,
                a_in,
                a_log,
                dt_bias,
                ssm_state_pool,
                out_f16,
                num_k_heads,
                num_v_heads,
                head_v_dim,
                ba_row_stride,
                total_state_slots);
        } else {
            // Select ROWS in {4, 7, 8, 11} by minimizing wave-quantized work
            // `waves * ROWS = ceil(nh * v_tiles / sm_count) * ROWS` using the GPU's
            // reported `sm_count` (breaking ties toward fewer waves).
            int best_rows = 4;
            int best_cost = INT32_MAX;
            int best_waves = INT32_MAX;
            for (int cand_rows : {4, 7, 8, 11}) {
                const int bv = cand_rows * 4;
                if (head_v_dim < bv && cand_rows != 4) continue;
                const int v_tiles = (head_v_dim + bv - 1) / bv;
                // Compute wave cost from per-request `num_v_heads * v_tiles` (not `nh = num_reqs * num_v_heads`)
                // so the selected `ROWS` template specialization is invariant to `batch->num_reqs`
                // while remaining identical for `num_reqs == 1`.
                const int ctas = num_v_heads * v_tiles;
                const int waves = (ctas + sm_count - 1) / sm_count;
                const int cost = waves * cand_rows;
                if (cost < best_cost || (cost == best_cost && waves < best_waves)) {
                    best_cost = cost;
                    best_waves = waves;
                    best_rows = cand_rows;
                }
            }
            const int bv = best_rows * 4;
            const int v_tiles = (head_v_dim + bv - 1) / bv;
            dim3 grid((unsigned)v_tiles, (unsigned)nh);
            auto launch_recurrent = [&](auto rows_tag, auto min_blocks_tag) {
                constexpr int ROWS = decltype(rows_tag)::value;
                constexpr int MIN_BLOCKS = decltype(min_blocks_tag)::value;
                w34_gdn::gdn_recurrent_prefill_f16_kernel<ROWS, MIN_BLOCKS>
                    <<<grid, 128, 0, (cudaStream_t)ex->stream>>>(
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
                        head_k_dim,
                        num_v_heads,
                        head_v_dim,
                        ba_row_stride,
                        total_state_slots);
            };
            if (best_rows == 11) {
                launch_recurrent(std::integral_constant<int, 11>{}, std::integral_constant<int, 3>{});
            } else if (best_rows == 8) {
                launch_recurrent(std::integral_constant<int, 8>{}, std::integral_constant<int, 5>{});
            } else if (best_rows == 7) {
                launch_recurrent(std::integral_constant<int, 7>{}, std::integral_constant<int, 5>{});
            } else {
                launch_recurrent(std::integral_constant<int, 4>{}, std::integral_constant<int, 5>{});
            }
        }
    }
    return (int)cudaGetLastError();
}

extern "C" int w34_gdn_recurrent_f16(
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
    int head_k_dim,
    int num_v_heads,
    int head_v_dim,
    int total_state_slots) {
    return w34_gdn_recurrent_strided_f16(
        ex, batch, qkv_conv_out, q_in, k_in, v_in, b_in, a_in, a_log, dt_bias,
        ssm_state_pool, out_f16, num_k_heads, head_k_dim, num_v_heads, head_v_dim,
        num_v_heads, total_state_slots);
}
