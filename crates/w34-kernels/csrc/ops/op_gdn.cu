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

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <math.h>

namespace w34_gdn {

__device__ __forceinline__ float silu_f32(float x) {
    return x / (1.0f + expf(-x));
}

__device__ __forceinline__ float softplus_f32(float x) {
    return (x > 20.0f) ? x : log1pf(expf(x));
}

// Grid: (ceil(conv_dim / 256), num_reqs), Block: 256 threads.
// Processes one (request, channel) sliding window across the request's query tokens,
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
    float hist[3] = {0.0f, 0.0f, 0.0f};

    const bool has_init = (batch.has_initial_state == nullptr) || (batch.has_initial_state[req] != 0);
    const int32_t slot_in = (batch.state_in_indices != nullptr) ? batch.state_in_indices[req] : -1;
    if (has_init && slot_in >= 0 && slot_in < total_state_slots) {
        const __half* s_in = conv_state_pool + ((size_t)slot_in * conv_dim + c) * state_len;
#pragma unroll
        for (int j = 0; j < 3; j++) {
            if (j < state_len) hist[j] = __half2float(s_in[j]);
        }
    }

    float w_reg[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    const __half* w_row = conv_weight + (size_t)c * kernel_size;
#pragma unroll
    for (int j = 0; j < 4; j++) {
        if (j < kernel_size) w_reg[j] = __half2float(w_row[j]);
    }
    const float bias_val = (conv_bias != nullptr) ? __half2float(conv_bias[c]) : 0.0f;

    int32_t last_spec_slot = -1;
    for (int step = 0; step < seqlen; step++) {
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
        if (batch.checkpoint_dst_indices != nullptr && batch.checkpoint_token_offsets != nullptr) {
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

    // Write final state to state_out_indices[req] (skip if already written to the same spec_slot)
    if (batch.state_out_indices != nullptr) {
        const int32_t slot_out = batch.state_out_indices[req];
        if (slot_out >= 0 && slot_out < total_state_slots && slot_out != last_spec_slot) {
            __half* s_out = conv_state_pool + ((size_t)slot_out * conv_dim + c) * state_len;
            for (int j = 0; j < state_len; j++) s_out[j] = __float2half(hist[j]);
        }
    }
}

template <int NW>
__device__ __forceinline__ void gdn_load_coalesced_s4(
    float4 (&smem_s4)[16][17],
    const float4* __restrict__ row_in4,
    int vi,
    int wk,
    float (&rr)[16]) {
    smem_s4[vi][0 * NW + wk] = __ldg(&row_in4[0 * NW + wk]);
    smem_s4[vi][1 * NW + wk] = __ldg(&row_in4[1 * NW + wk]);
    __syncwarp();
    if (wk < 4) {
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            const float4 v4 = smem_s4[vi][wk * 4 + j];
            rr[j * 4 + 0] = v4.x;
            rr[j * 4 + 1] = v4.y;
            rr[j * 4 + 2] = v4.z;
            rr[j * 4 + 3] = v4.w;
        }
    }
    __syncwarp();
    smem_s4[vi][0 * NW + wk] = __ldg(&row_in4[2 * NW + wk]);
    smem_s4[vi][1 * NW + wk] = __ldg(&row_in4[3 * NW + wk]);
    __syncwarp();
    if (wk >= 4) {
        const int half_wk = wk - 4;
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            const float4 v4 = smem_s4[vi][half_wk * 4 + j];
            rr[j * 4 + 0] = v4.x;
            rr[j * 4 + 1] = v4.y;
            rr[j * 4 + 2] = v4.z;
            rr[j * 4 + 3] = v4.w;
        }
    }
}

template <int NW>
__device__ __forceinline__ void gdn_store_coalesced_s4(
    float4 (&smem_s4)[16][17],
    float4* __restrict__ row_out4,
    int vi,
    int wk,
    const float (&rr)[16]) {
    __syncwarp();
    if (wk < 4) {
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            smem_s4[vi][wk * 4 + j] = make_float4(
                rr[j * 4 + 0], rr[j * 4 + 1], rr[j * 4 + 2], rr[j * 4 + 3]);
        }
    }
    __syncwarp();
    row_out4[0 * NW + wk] = smem_s4[vi][0 * NW + wk];
    row_out4[1 * NW + wk] = smem_s4[vi][1 * NW + wk];
    __syncwarp();
    if (wk >= 4) {
        const int half_wk = wk - 4;
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            smem_s4[vi][half_wk * 4 + j] = make_float4(
                rr[j * 4 + 0], rr[j * 4 + 1], rr[j * 4 + 2], rr[j * 4 + 3]);
        }
    }
    __syncwarp();
    row_out4[2 * NW + wk] = smem_s4[vi][0 * NW + wk];
    row_out4[3 * NW + wk] = smem_s4[vi][1 * NW + wk];
}

// Ported from vllm.cpp GdnDecodeFusedKernel / GdnSpecDecodeFusedKernel with in-kernel
// q/k L2-normalization and g/beta gating:
// Grid: (dv / BV, num_reqs * num_v_heads), where BV = 16, NW = 8, threads = 128.
// Each block owns a [BV=16, dk=128] slice of ssm_state for one (req, v_head).
// Each row vi in 0..16 is partitioned across NW=8 consecutive lanes (16 dk cols/lane),
// so the dot product and state update stay in registers (`rr[16]`) with an 8-lane
// `__shfl_xor_sync` butterfly reduction!
// `__launch_bounds__(BV * NW, 8)` + two-pass `smem_s4[16][17]` (`4.35 KiB` + `1.03 KiB` =
// `5.38 KiB`/CTA) fits 8 CTAs/SM (`1024` resident threads/SM, up from 5 CTAs/SM) on `sm_86`.
template <int BV, int NW>
__global__ __launch_bounds__(BV * NW, 8) void gdn_recurrent_f16_kernel(
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
    const int vbase = i_v * BV;
    const int tid = threadIdx.x;
    const int vi = tid / NW;       // 0..16
    const int wk = tid % NW;       // 0..8
    const int vrow = vbase + vi;
    const int ck = head_k_dim / NW; // 16 when head_k_dim == 128, NW == 8
    const int c0 = wk * ck;

    // Load initial [BV, head_k_dim] FP32 state slice into registers rr[16].
    // When NW == 8 and head_k_dim == 128, all 8 lanes wk = 0..7 of row vi reside in the
    // same warp, so two-pass staging through smem_s4[16][17] converts 64-byte-strided global
    // float4 accesses into 100%-coalesced contiguous 128-byte warp accesses while preserving
    // bit-exact 0-ULP register ownership rr[0..15] and fitting 8 CTAs/SM on sm_86.
    __shared__ float4 smem_s4[16][17];
    const bool coalesced_s4 = (NW == 8) && (BV == 16) && (head_k_dim == 128) && ((vbase + BV) <= head_v_dim);
    float rr[16];
#pragma unroll
    for (int j = 0; j < 16; j++) rr[j] = 0.0f;

    const bool has_init = (batch.has_initial_state == nullptr) || (batch.has_initial_state[req] != 0);
    const int32_t slot_in = (batch.state_in_indices != nullptr) ? batch.state_in_indices[req] : -1;
    const size_t head_state_elems = (size_t)head_v_dim * head_k_dim;
    if (has_init && slot_in >= 0 && slot_in < total_state_slots) {
        if (coalesced_s4) {
            const float4* row_in4 = reinterpret_cast<const float4*>(
                ssm_state_pool + ((size_t)slot_in * num_v_heads + hv) * head_state_elems +
                (size_t)vrow * head_k_dim);
            gdn_load_coalesced_s4<NW>(smem_s4, row_in4, vi, wk, rr);
        } else if (vrow < head_v_dim) {
            const float* s_in = ssm_state_pool
                                + ((size_t)slot_in * num_v_heads + hv) * head_state_elems
                                + (size_t)vrow * head_k_dim;
            const float4* s_in4 = reinterpret_cast<const float4*>(s_in + c0);
#pragma unroll
            for (int j = 0; j < 4; j++) {
                float4 v4 = s_in4[j];
                rr[j * 4 + 0] = v4.x;
                rr[j * 4 + 1] = v4.y;
                rr[j * 4 + 2] = v4.z;
                rr[j * 4 + 3] = v4.w;
            }
        }
    }

    __shared__ float smem_qk[256 + 8];
    float* bq = smem_qk;            // [128]
    float* bk = smem_qk + 128;      // [128]
    float* s_red = smem_qk + 256;   // [8]

    const float scale = rsqrtf((float)head_k_dim);
    const float a_log_hv = a_log[hv];
    const float dt_bias_hv = dt_bias[hv];
    const int k_total_dim = num_k_heads * head_k_dim;
    const int v_total_dim = num_v_heads * head_v_dim;
    const int qkv_stride = 2 * k_total_dim + v_total_dim;

    int32_t last_spec_slot = -1;
    for (int step = 0; step < seqlen; step++) {
        const int tok = lo + step;
        if (step > 0) __syncthreads();

        // Load q and k for (tok, hk) and L2-normalize in FP32
        const __half* q_tok_hk = (qkv_conv_out != nullptr)
                                     ? (qkv_conv_out + (size_t)tok * qkv_stride + (size_t)hk * head_k_dim)
                                     : (q_in + ((size_t)tok * num_k_heads + hk) * head_k_dim);
        const __half* k_tok_hk = (qkv_conv_out != nullptr)
                                     ? (qkv_conv_out + (size_t)tok * qkv_stride + k_total_dim + (size_t)hk * head_k_dim)
                                     : (k_in + ((size_t)tok * num_k_heads + hk) * head_k_dim);
        const __half* v_tok_hv = (qkv_conv_out != nullptr)
                                     ? (qkv_conv_out + (size_t)tok * qkv_stride + 2 * k_total_dim + (size_t)hv * head_v_dim)
                                     : (v_in + ((size_t)tok * num_v_heads + hv) * head_v_dim);

        float q_val = (tid < head_k_dim) ? __half2float(q_tok_hk[tid]) : 0.0f;
        float k_val = (tid < head_k_dim) ? __half2float(k_tok_hk[tid]) : 0.0f;

        // Warp-tree L2 norm over 128 threads
        float q_sq = q_val * q_val;
        float k_sq = k_val * k_val;
#pragma unroll
        for (int off = 16; off > 0; off >>= 1) {
            q_sq += __shfl_xor_sync(0xffffffffu, q_sq, off);
            k_sq += __shfl_xor_sync(0xffffffffu, k_sq, off);
        }
        if ((tid & 31) == 0) {
            s_red[tid >> 5] = q_sq;
            s_red[4 + (tid >> 5)] = k_sq;
        }
        __syncthreads();
        if (tid < head_k_dim) {
            const float q_inv = rsqrtf(s_red[0] + s_red[1] + s_red[2] + s_red[3] + 1e-6f) * scale;
            const float k_inv = rsqrtf(s_red[4] + s_red[5] + s_red[6] + s_red[7] + 1e-6f);
            bq[tid] = q_val * q_inv;
            bk[tid] = k_val * k_inv;
        }
        __syncthreads();

        const float a_val = __half2float(a_in[(size_t)tok * ba_row_stride + hv]);
        const float b_val = __half2float(b_in[(size_t)tok * ba_row_stride + hv]);
        const float g = -expf(a_log_hv) * softplus_f32(a_val + dt_bias_hv);
        const float decay = expf(g);
        const float beta = 1.0f / (1.0f + expf(-b_val));

        float pdot = 0.0f;
#pragma unroll
        for (int j = 0; j < 16; j++) {
            rr[j] *= decay;
            pdot += rr[j] * bk[c0 + j];
        }
        float dot = pdot;
#pragma unroll
        for (int off = 1; off < NW; off <<= 1) {
            dot += __shfl_xor_sync(0xffffffffu, dot, off);
        }

        const float vv = (vrow < head_v_dim) ? __half2float(v_tok_hv[vrow]) : 0.0f;
        const float vp = (vv - dot) * beta;

        float po = 0.0f;
#pragma unroll
        for (int j = 0; j < 16; j++) {
            rr[j] += vp * bk[c0 + j];
            po += rr[j] * bq[c0 + j];
        }
        float o = po;
#pragma unroll
        for (int off = 1; off < NW; off <<= 1) {
            o += __shfl_xor_sync(0xffffffffu, o, off);
        }
        if (vrow < head_v_dim && wk == 0) {
            out_f16[((size_t)tok * num_v_heads + hv) * head_v_dim + vrow] = __float2half(o);
        }

        // Speculative per-step snapshot into spec_state_indices[req, step]
        if (batch.spec_state_indices != nullptr && step < batch.max_verify_tokens) {
            const int32_t spec_slot = batch.spec_state_indices[req * batch.max_verify_tokens + step];
            if (spec_slot >= 0 && spec_slot < total_state_slots) {
                if (coalesced_s4) {
                    float4* row_spec4 = reinterpret_cast<float4*>(
                        ssm_state_pool + ((size_t)spec_slot * num_v_heads + hv) * head_state_elems +
                        (size_t)vrow * head_k_dim);
                    gdn_store_coalesced_s4<NW>(smem_s4, row_spec4, vi, wk, rr);
                    if (step == seqlen - 1) last_spec_slot = spec_slot;
                } else if (vrow < head_v_dim) {
                    float* s_spec = ssm_state_pool
                                    + ((size_t)spec_slot * num_v_heads + hv) * head_state_elems
                                    + (size_t)vrow * head_k_dim;
                    float4* s_spec4 = reinterpret_cast<float4*>(s_spec + c0);
#pragma unroll
                    for (int j = 0; j < 4; j++) {
                        s_spec4[j] = make_float4(
                            rr[j * 4 + 0], rr[j * 4 + 1], rr[j * 4 + 2], rr[j * 4 + 3]);
                    }
                    if (step == seqlen - 1) last_spec_slot = spec_slot;
                }
            }
        }

        // Chunk-aligned radix/ping-pong checkpoint snapshot
        if (batch.checkpoint_dst_indices != nullptr && batch.checkpoint_token_offsets != nullptr) {
            for (int e = 0; e < batch.max_checkpoints_per_req; e++) {
                const int32_t trig_off = batch.checkpoint_token_offsets[req * batch.max_checkpoints_per_req + e];
                if (trig_off == step + 1) {
                    const int32_t chk_slot = batch.checkpoint_dst_indices[req * batch.max_checkpoints_per_req + e];
                    if (chk_slot >= 0 && chk_slot < total_state_slots) {
                        if (coalesced_s4) {
                            float4* row_chk4 = reinterpret_cast<float4*>(
                                ssm_state_pool + ((size_t)chk_slot * num_v_heads + hv) * head_state_elems +
                                (size_t)vrow * head_k_dim);
                            gdn_store_coalesced_s4<NW>(smem_s4, row_chk4, vi, wk, rr);
                        } else if (vrow < head_v_dim) {
                            float* s_chk = ssm_state_pool
                                           + ((size_t)chk_slot * num_v_heads + hv) * head_state_elems
                                           + (size_t)vrow * head_k_dim;
                            float4* s_chk4 = reinterpret_cast<float4*>(s_chk + c0);
#pragma unroll
                            for (int j = 0; j < 4; j++) {
                                s_chk4[j] = make_float4(
                                    rr[j * 4 + 0], rr[j * 4 + 1], rr[j * 4 + 2], rr[j * 4 + 3]);
                            }
                        }
                    }
                }
            }
        }
    }

    // Write final state to state_out_indices[req] (skip if already written to the same spec_slot)
    if (batch.state_out_indices != nullptr) {
        const int32_t slot_out = batch.state_out_indices[req];
        if (slot_out >= 0 && slot_out < total_state_slots && slot_out != last_spec_slot) {
            if (coalesced_s4) {
                float4* row_out4 = reinterpret_cast<float4*>(
                    ssm_state_pool + ((size_t)slot_out * num_v_heads + hv) * head_state_elems +
                    (size_t)vrow * head_k_dim);
                gdn_store_coalesced_s4<NW>(smem_s4, row_out4, vi, wk, rr);
            } else if (vrow < head_v_dim) {
                float* s_out = ssm_state_pool
                               + ((size_t)slot_out * num_v_heads + hv) * head_state_elems
                               + (size_t)vrow * head_k_dim;
                float4* s_out4 = reinterpret_cast<float4*>(s_out + c0);
#pragma unroll
                for (int j = 0; j < 4; j++) {
                    s_out4[j] = make_float4(
                        rr[j * 4 + 0], rr[j * 4 + 1], rr[j * 4 + 2], rr[j * 4 + 3]);
                }
            }
        }
    }
}

}  // namespace w34_gdn

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
    dim3 grid((unsigned)((conv_dim + 255) / 256), (unsigned)batch->num_reqs);
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
    if (head_k_dim != 128 || ba_row_stride < num_v_heads) return (int)cudaErrorInvalidValue;

    constexpr int BV = 16;
    constexpr int NW = 8;
    const int v_tiles = (head_v_dim + BV - 1) / BV;
    dim3 grid((unsigned)v_tiles, (unsigned)(batch->num_reqs * num_v_heads));
    w34_gdn::gdn_recurrent_f16_kernel<BV, NW><<<grid, BV * NW, 0, (cudaStream_t)ex->stream>>>(
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
