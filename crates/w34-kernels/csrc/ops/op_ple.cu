// Qwen3.8-Flash-Next (qwen4_exp) Per-Layer Embedding (PLE) CUDA Kernels.
// Reference: nvidia/ple_layer.py
//   - F8_E4M3 (e4m3fn) dequantization with scalar weight_scale -> FP16 [num_tokens, 2560]
//   - Stateful causal dilated 1D depthwise convolution (kernel_size=4, dilation=3, span=9)
//     over [num_tokens, 10240] with StateBundlePool slot indexing & speculative step snapshots
//   - Signed-sqrt sigmoid gating:
//       gate = sigmoid(sign(q * k) * sqrt(|q * k| + 1e-6))
//       ple_delta[tok, s, d] = gate[tok, s, d] * value[tok, d]
#include "../w34_cuda.h"
#include "hadamard_inline.cuh"

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <math.h>

namespace w34 {

__device__ __forceinline__ float ple_f8_e4m3_to_f32(uint8_t byte) {
    const float sign = (byte & 0x80u) ? -1.0f : 1.0f;
    const uint32_t exp = (byte >> 3) & 0x0fu;
    const uint32_t mant = byte & 0x07u;
    if (exp == 0x0fu && mant == 0x07u) {
        return 0.0f;
    } else if (exp == 0u) {
        return sign * (float)mant * (1.0f / 512.0f);
    } else {
        const float scale = __uint_as_float((exp + 127u - 7u) << 23);
        return sign * (1.0f + (float)mant * 0.125f) * scale;
    }
}

__global__ void ple_dequant_f8_to_f16_kernel(
    const uint8_t* __restrict__ f8_in,
    __half* __restrict__ f16_out,
    const uint8_t* __restrict__ valid_token_mask,
    int num_tokens,
    int embed_dim,
    float weight_scale) {
    const int tok = blockIdx.x;
    if (tok >= num_tokens) return;
    const size_t base = (size_t)tok * embed_dim;
    __half* dst = f16_out + base;
    if (valid_token_mask != nullptr && valid_token_mask[tok] == 0) {
        for (int d = threadIdx.x; d < embed_dim; d += blockDim.x) {
            dst[d] = __float2half(0.0f);
        }
        return;
    }
    const uint8_t* src = f8_in + base;
    for (int d = threadIdx.x; d < embed_dim; d += blockDim.x) {
        const float val = ple_f8_e4m3_to_f32(src[d]) * weight_scale;
        dst[d] = __float2half(val);
    }
}

__global__ void ple_dequant_lut4_to_f16_kernel(
    const uint8_t* __restrict__ raw_in,
    const float* __restrict__ lut16,
    __half* __restrict__ f16_out,
    const uint8_t* __restrict__ valid_token_mask,
    int num_tokens,
    int num_heads,
    int head_dim,
    int group_size) {
    const int tok = blockIdx.x;
    if (tok >= num_tokens) return;
    const int embed_dim = num_heads * head_dim;
    __half* dst = f16_out + (size_t)tok * embed_dim;
    if (valid_token_mask != nullptr && valid_token_mask[tok] == 0) {
        for (int d = threadIdx.x; d < embed_dim; d += blockDim.x) {
            dst[d] = __float2half(0.0f);
        }
        return;
    }
    __shared__ float s_lut[16];
    if (threadIdx.x < 16) {
        s_lut[threadIdx.x] = lut16[threadIdx.x];
    }
    __syncthreads();

    const int packed_bytes = head_dim / 2;
    const int num_groups = head_dim / group_size;
    const int head_stride = packed_bytes + num_groups * 2;
    const uint8_t* tok_src = raw_in + (size_t)tok * num_heads * head_stride;

    for (int d = threadIdx.x; d < embed_dim; d += blockDim.x) {
        const int h = d / head_dim;
        const int col = d - h * head_dim;
        const uint8_t* head_ptr = tok_src + (size_t)h * head_stride;
        const uint8_t packed_byte = head_ptr[col >> 1];
        const uint32_t nibble = (col & 1) ? ((packed_byte >> 4) & 0x0fu) : (packed_byte & 0x0fu);
        const int grp = col / group_size;
        const uint8_t* scale_bytes = head_ptr + packed_bytes + grp * 2;
        const uint16_t scale_bits = (uint16_t)scale_bytes[0] | ((uint16_t)scale_bytes[1] << 8);
        const float scale = __half2float(__ushort_as_half(scale_bits));
        dst[d] = __float2half(s_lut[nibble] * scale);
    }
}

__global__ void ple_dequant_f8_hadamard_to_f16_kernel(
    const uint8_t* __restrict__ f8_in,
    __half* __restrict__ f16_out,
    const uint8_t* __restrict__ valid_token_mask,
    int num_tokens,
    int embed_dim,
    float weight_scale) {
    const int tok = blockIdx.x;
    if (tok >= num_tokens) return;
    const size_t base = (size_t)tok * embed_dim;
    __half* dst = f16_out + base;
    if (valid_token_mask != nullptr && valid_token_mask[tok] == 0) {
        for (int d = threadIdx.x; d < embed_dim; d += blockDim.x) {
            dst[d] = __float2half(0.0f);
        }
        return;
    }
    const uint8_t* src = f8_in + base;
    for (int base_col = 0; base_col < embed_dim; base_col += (int)blockDim.x * 4) {
        const int col0 = base_col + (int)threadIdx.x * 4;
        if (col0 < embed_dim) {
            __half h[4];
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                const float val = ple_f8_e4m3_to_f32(src[col0 + j]) * weight_scale;
                h[j] = __float2half(val);
            }
            const uint2 v = *reinterpret_cast<const uint2*>(h);
            *reinterpret_cast<uint2*>(&dst[col0]) = hadamard64_vec4_f16(v);
        }
    }
}

__global__ void ple_dequant_lut4_hadamard_to_f16_kernel(
    const uint8_t* __restrict__ raw_in,
    const float* __restrict__ lut16,
    __half* __restrict__ f16_out,
    const uint8_t* __restrict__ valid_token_mask,
    int num_tokens,
    int num_heads,
    int head_dim,
    int group_size) {
    const int tok = blockIdx.x;
    if (tok >= num_tokens) return;
    const int embed_dim = num_heads * head_dim;
    __half* dst = f16_out + (size_t)tok * embed_dim;
    if (valid_token_mask != nullptr && valid_token_mask[tok] == 0) {
        for (int d = threadIdx.x; d < embed_dim; d += blockDim.x) {
            dst[d] = __float2half(0.0f);
        }
        return;
    }
    __shared__ float s_lut[16];
    if (threadIdx.x < 16) {
        s_lut[threadIdx.x] = lut16[threadIdx.x];
    }
    __syncthreads();

    const int packed_bytes = head_dim / 2;
    const int num_groups = head_dim / group_size;
    const int head_stride = packed_bytes + num_groups * 2;
    const uint8_t* tok_src = raw_in + (size_t)tok * num_heads * head_stride;

    for (int base_col = 0; base_col < embed_dim; base_col += (int)blockDim.x * 4) {
        const int col0 = base_col + (int)threadIdx.x * 4;
        if (col0 < embed_dim) {
            __half h[4];
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                const int d = col0 + j;
                const int head = d / head_dim;
                const int col = d - head * head_dim;
                const uint8_t* head_ptr = tok_src + (size_t)head * head_stride;
                const uint8_t packed_byte = head_ptr[col >> 1];
                const uint32_t nibble =
                    (col & 1) ? ((packed_byte >> 4) & 0x0fu) : (packed_byte & 0x0fu);
                const int grp = col / group_size;
                const uint8_t* scale_bytes = head_ptr + packed_bytes + grp * 2;
                const uint16_t scale_bits =
                    (uint16_t)scale_bytes[0] | ((uint16_t)scale_bytes[1] << 8);
                const float scale = __half2float(__ushort_as_half(scale_bits));
                h[j] = __float2half(s_lut[nibble] * scale);
            }
            const uint2 v = *reinterpret_cast<const uint2*>(h);
            *reinterpret_cast<uint2*>(&dst[col0]) = hadamard64_vec4_f16(v);
        }
    }
}

// Causal dilated 1D depthwise convolution with kernel_size=4, dilation=3 (span=9 taps).
// Grid: (conv_dim + 255) / 256, Block: 256, Y-grid: num_reqs.
__global__ void ple_dilated_conv1d_f16_kernel(
    const __half* __restrict__ key_in,
    const __half* __restrict__ conv_weight,
    __half* __restrict__ conv_state_pool,
    __half* __restrict__ key_out,
    __half* __restrict__ residual_inout,
    w34_gdn_batch batch,
    int conv_dim) {
    const int d = blockIdx.x * blockDim.x + threadIdx.x;
    const int req = blockIdx.y;
    if (d >= conv_dim || req >= batch.num_reqs) return;
    if (batch.valid_req_mask != nullptr && batch.valid_req_mask[req] == 0) return;

    const int q_start = batch.query_start_loc[req];
    const int q_end = batch.query_start_loc[req + 1];
    const int seq_len = q_end - q_start;
    if (seq_len <= 0) return;

    const int state_in = (batch.state_in_indices != nullptr) ? batch.state_in_indices[req] : -1;
    const int state_out = (batch.state_out_indices != nullptr) ? batch.state_out_indices[req] : state_in;
    const bool has_init =
        (batch.has_initial_state != nullptr) && (batch.has_initial_state[req] != 0) && (state_in >= 0);

    constexpr int SPAN = 9; // dilation(3) * (kernel_size(4) - 1)
    float hist[SPAN];
#pragma unroll
    for (int s = 0; s < SPAN; ++s) {
        if (has_init && conv_state_pool != nullptr) {
            const size_t idx = ((size_t)state_in * conv_dim + d) * SPAN + s;
            hist[s] = __half2float(conv_state_pool[idx]);
        } else {
            hist[s] = 0.0f;
        }
    }

    const __half* w_d = conv_weight + (size_t)d * 4;
    const float w0 = __half2float(w_d[0]); // lag 9
    const float w1 = __half2float(w_d[1]); // lag 6
    const float w2 = __half2float(w_d[2]); // lag 3
    const float w3 = __half2float(w_d[3]); // lag 0

    for (int t = 0; t < seq_len; ++t) {
        const float v_lag0 = __half2float(key_in[(size_t)(q_start + t) * conv_dim + d]);
        const float v_lag3 = (t >= 3)
            ? __half2float(key_in[(size_t)(q_start + t - 3) * conv_dim + d])
            : hist[SPAN + t - 3];
        const float v_lag6 = (t >= 6)
            ? __half2float(key_in[(size_t)(q_start + t - 6) * conv_dim + d])
            : hist[SPAN + t - 6];
        const float v_lag9 = (t >= 9)
            ? __half2float(key_in[(size_t)(q_start + t - 9) * conv_dim + d])
            : hist[SPAN + t - 9];

        const float acc = w0 * v_lag9 + w1 * v_lag6 + w2 * v_lag3 + w3 * v_lag0;
        const float silu_acc = acc / (1.0f + expf(-acc));
        const size_t out_idx = (size_t)(q_start + t) * conv_dim + d;
        const __half out_h = __float2half(silu_acc);
        if (key_out != nullptr) {
            key_out[out_idx] = out_h;
        }
        if (residual_inout != nullptr) {
            residual_inout[out_idx] = __hadd(residual_inout[out_idx], out_h);
        }

        if (batch.spec_state_indices != nullptr && batch.max_verify_tokens > 0 &&
            t < batch.max_verify_tokens && conv_state_pool != nullptr) {
            const int spec_slot = batch.spec_state_indices[req * batch.max_verify_tokens + t];
            if (spec_slot >= 0) {
                __half* dst_state = conv_state_pool + ((size_t)spec_slot * conv_dim + d) * SPAN;
#pragma unroll
                for (int s = 0; s < SPAN; ++s) {
                    const int pos = (t + 1) - SPAN + s;
                    const float sv = (pos >= 0)
                        ? __half2float(key_in[(size_t)(q_start + pos) * conv_dim + d])
                        : hist[SPAN + pos];
                    dst_state[s] = __float2half(sv);
                }
            }
        }
        if (batch.checkpoint_dst_indices != nullptr && batch.checkpoint_token_offsets != nullptr &&
            batch.max_checkpoints_per_req > 0 && conv_state_pool != nullptr) {
            for (int e = 0; e < batch.max_checkpoints_per_req; ++e) {
                const int trig_off =
                    batch.checkpoint_token_offsets[req * batch.max_checkpoints_per_req + e];
                if (trig_off == t + 1) {
                    const int chk_slot =
                        batch.checkpoint_dst_indices[req * batch.max_checkpoints_per_req + e];
                    if (chk_slot >= 0) {
                        __half* dst_state =
                            conv_state_pool + ((size_t)chk_slot * conv_dim + d) * SPAN;
#pragma unroll
                        for (int s = 0; s < SPAN; ++s) {
                            const int pos = (t + 1) - SPAN + s;
                            const float sv = (pos >= 0)
                                ? __half2float(key_in[(size_t)(q_start + pos) * conv_dim + d])
                                : hist[SPAN + pos];
                            dst_state[s] = __float2half(sv);
                        }
                    }
                }
            }
        }
    }

    if (batch.spec_state_indices == nullptr && state_out >= 0 && conv_state_pool != nullptr) {
        __half* dst_state = conv_state_pool + ((size_t)state_out * conv_dim + d) * SPAN;
#pragma unroll
        for (int s = 0; s < SPAN; ++s) {
            const int pos = seq_len - SPAN + s;
            const float sv = (pos >= 0)
                ? __half2float(key_in[(size_t)(q_start + pos) * conv_dim + d])
                : hist[SPAN + pos];
            dst_state[s] = __float2half(sv);
        }
    }
}

// Signed-sqrt sigmoid gating over [num_tokens, hc_count, embed_dim]:
//   dot[tok, s] = sum_{d=0..embed_dim-1} (norm_query[tok, s, d] * norm_key[tok, s, d]) / sqrt(embed_dim)
//   gate[tok, s] = sigmoid(sign(dot[tok, s]) * sqrt(max(|dot[tok, s]|, eps)))
//   ple_delta[tok, s, d] = gate[tok, s] * value[tok, d]
__global__ void ple_signed_sqrt_gate_f16_kernel(
    const __half* __restrict__ norm_query,
    const __half* __restrict__ norm_key,
    const __half* __restrict__ value_in,
    __half* __restrict__ ple_delta_out,
    const uint8_t* __restrict__ valid_token_mask,
    int num_tokens,
    int hc_count,
    int embed_dim,
    float eps) {
    const int tok = blockIdx.x;
    if (tok >= num_tokens) return;

    const size_t stream_base = (size_t)tok * hc_count * embed_dim;
    __half* dst_tok = ple_delta_out + stream_base;
    if (valid_token_mask != nullptr && valid_token_mask[tok] == 0) {
        for (int i = threadIdx.x; i < hc_count * embed_dim; i += blockDim.x) {
            dst_tok[i] = __float2half(0.0f);
        }
        return;
    }

    const __half* q_tok = norm_query + stream_base;
    const __half* k_tok = norm_key + stream_base;
    const __half* v_tok = value_in + (size_t)tok * embed_dim;

    constexpr int MAX_HC = 8;
    float dot_s[MAX_HC] = {0.0f};
    for (int d = threadIdx.x; d < embed_dim; d += blockDim.x) {
#pragma unroll
        for (int s = 0; s < MAX_HC; ++s) {
            if (s < hc_count) {
                const size_t idx = (size_t)s * embed_dim + d;
                dot_s[s] += __half2float(q_tok[idx]) * __half2float(k_tok[idx]);
            }
        }
    }

    const int lane = threadIdx.x & 31;
    const int wid = threadIdx.x >> 5;
    const int num_warps = blockDim.x >> 5;
    __shared__ float s_warp[8][MAX_HC];
    __shared__ float s_gate[MAX_HC];

#pragma unroll
    for (int s = 0; s < MAX_HC; ++s) {
        float v = dot_s[s];
#pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1) {
            v += __shfl_xor_sync(0xffffffffu, v, offset);
        }
        if (lane == 0) {
            s_warp[wid][s] = v;
        }
    }
    __syncthreads();

    if (threadIdx.x < MAX_HC) {
        const int s = threadIdx.x;
        if (s < hc_count) {
            float sum = 0.0f;
            for (int w = 0; w < num_warps; ++w) {
                sum += s_warp[w][s];
            }
            const float dot = sum * rsqrtf((float)embed_dim);
            const float sign_d = (dot > 0.0f) ? 1.0f : ((dot < 0.0f) ? -1.0f : 0.0f);
            const float clamped = fmaxf(fabsf(dot), eps);
            const float logit = sign_d * sqrtf(clamped);
            s_gate[s] = 1.0f / (1.0f + expf(-logit));
        } else {
            s_gate[s] = 0.0f;
        }
    }
    __syncthreads();

    for (int d = threadIdx.x; d < embed_dim; d += blockDim.x) {
        const float val = __half2float(v_tok[d]);
#pragma unroll
        for (int s = 0; s < MAX_HC; ++s) {
            if (s < hc_count) {
                const size_t idx = (size_t)s * embed_dim + d;
                dst_tok[idx] = __float2half(s_gate[s] * val);
            }
        }
    }
}

__global__ void ple_signed_sqrt_gate_norm_add_f16_kernel(
    const __half* norm_query,
    const __half* norm_key,
    const __half* value_in,
    __half* ple_delta_out,
    __half* residual_inout,
    const __half* __restrict__ norm_conv_weight,
    __half* norm_conv_out,
    const uint8_t* __restrict__ valid_token_mask,
    int num_tokens,
    int hc_count,
    int embed_dim,
    float eps,
    float norm_eps) {
    const int tok = blockIdx.x;
    if (tok >= num_tokens) return;

    const size_t stream_base = (size_t)tok * hc_count * embed_dim;
    __half* dst_tok = (ple_delta_out != nullptr) ? (ple_delta_out + stream_base) : nullptr;
    __half* res_tok = (residual_inout != nullptr) ? (residual_inout + stream_base) : nullptr;
    __half* norm_tok = (norm_conv_out != nullptr) ? (norm_conv_out + stream_base) : nullptr;
    if (valid_token_mask != nullptr && valid_token_mask[tok] == 0) {
        for (int i = threadIdx.x; i < hc_count * embed_dim; i += blockDim.x) {
            if (dst_tok != nullptr) dst_tok[i] = __float2half(0.0f);
            if (norm_tok != nullptr) norm_tok[i] = __float2half(0.0f);
        }
        return;
    }

    const __half* q_tok = norm_query + stream_base;
    const __half* k_tok = norm_key + stream_base;
    const __half* v_tok = value_in + (size_t)tok * embed_dim;

    constexpr int MAX_HC = 8;
    float dot_s[MAX_HC] = {0.0f};
    for (int d = threadIdx.x; d < embed_dim; d += blockDim.x) {
#pragma unroll
        for (int s = 0; s < MAX_HC; ++s) {
            if (s < hc_count) {
                const size_t idx = (size_t)s * embed_dim + d;
                dot_s[s] += __half2float(q_tok[idx]) * __half2float(k_tok[idx]);
            }
        }
    }

    const int lane = threadIdx.x & 31;
    const int wid = threadIdx.x >> 5;
    const int num_warps = blockDim.x >> 5;
    __shared__ float s_warp[8][MAX_HC];
    __shared__ float s_gate[MAX_HC];
    __shared__ float s_inv_rms[MAX_HC];

#pragma unroll
    for (int s = 0; s < MAX_HC; ++s) {
        float v = dot_s[s];
#pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1) {
            v += __shfl_xor_sync(0xffffffffu, v, offset);
        }
        if (lane == 0) {
            s_warp[wid][s] = v;
        }
    }
    __syncthreads();

    if (threadIdx.x < MAX_HC) {
        const int s = threadIdx.x;
        if (s < hc_count) {
            float sum = 0.0f;
            for (int w = 0; w < num_warps; ++w) {
                sum += s_warp[w][s];
            }
            const float dot = sum * rsqrtf((float)embed_dim);
            const float sign_d = (dot > 0.0f) ? 1.0f : ((dot < 0.0f) ? -1.0f : 0.0f);
            const float clamped = fmaxf(fabsf(dot), eps);
            const float logit = sign_d * sqrtf(clamped);
            s_gate[s] = 1.0f / (1.0f + expf(-logit));
        } else {
            s_gate[s] = 0.0f;
        }
    }
    __syncthreads();

    float sum_sq_s[MAX_HC] = {0.0f};
    const bool do_norm = (norm_conv_weight != nullptr && norm_tok != nullptr);
    for (int d = threadIdx.x; d < embed_dim; d += blockDim.x) {
        const float val = __half2float(v_tok[d]);
#pragma unroll
        for (int s = 0; s < MAX_HC; ++s) {
            if (s < hc_count) {
                const size_t idx = (size_t)s * embed_dim + d;
                const __half delta_h = __float2half(s_gate[s] * val);
                if (dst_tok != nullptr) {
                    dst_tok[idx] = delta_h;
                }
                if (res_tok != nullptr) {
                    res_tok[idx] = __hadd(res_tok[idx], delta_h);
                }
                if (do_norm) {
                    const float df = __half2float(delta_h);
                    sum_sq_s[s] += df * df;
                }
            }
        }
    }

    if (do_norm) {
#pragma unroll
        for (int s = 0; s < MAX_HC; ++s) {
            float v = sum_sq_s[s];
#pragma unroll
            for (int offset = 16; offset > 0; offset >>= 1) {
                v += __shfl_xor_sync(0xffffffffu, v, offset);
            }
            if (lane == 0) {
                s_warp[wid][s] = v;
            }
        }
        __syncthreads();

        if (wid == 0) {
#pragma unroll
            for (int s = 0; s < MAX_HC; ++s) {
                if (s < hc_count) {
                    float v = (lane < num_warps) ? s_warp[lane][s] : 0.0f;
#pragma unroll
                    for (int offset = 16; offset > 0; offset >>= 1) {
                        v += __shfl_xor_sync(0xffffffffu, v, offset);
                    }
                    if (lane == 0) {
                        s_inv_rms[s] = rsqrtf(v / (float)embed_dim + norm_eps);
                    }
                }
            }
        }
        __syncthreads();

        for (int d = threadIdx.x; d < embed_dim; d += blockDim.x) {
            const float val = __half2float(v_tok[d]);
#pragma unroll
            for (int s = 0; s < MAX_HC; ++s) {
                if (s < hc_count) {
                    const size_t idx = (size_t)s * embed_dim + d;
                    const __half delta_h = __float2half(s_gate[s] * val);
                    const float v = __half2float(delta_h) * s_inv_rms[s];
                    const float wf = 1.0f + __half2float(norm_conv_weight[idx]);
                    norm_tok[idx] = __float2half(v * wf);
                }
            }
        }
    }
}

} // namespace w34

extern "C" {

int w34_ple_dequant_f8_to_f16(
    const w34_exec* ex,
    const uint8_t* f8_in,
    void* f16_out,
    const uint8_t* valid_token_mask,
    int num_tokens,
    int embed_dim,
    float weight_scale) {
    if (!ex || !f8_in || !f16_out || num_tokens < 0 || embed_dim <= 0) {
        return -1;
    }
    if (num_tokens == 0) return 0;
    cudaStream_t stream = reinterpret_cast<cudaStream_t>(ex->stream);
    w34::ple_dequant_f8_to_f16_kernel<<<num_tokens, 256, 0, stream>>>(
        f8_in,
        static_cast<__half*>(f16_out),
        valid_token_mask,
        num_tokens,
        embed_dim,
        weight_scale);
    return (cudaGetLastError() == cudaSuccess) ? 0 : -2;
}

int w34_ple_dequant_f8_hadamard_to_f16(
    const w34_exec* ex,
    const uint8_t* f8_in,
    void* f16_out,
    const uint8_t* valid_token_mask,
    int num_tokens,
    int embed_dim,
    float weight_scale) {
    if (!ex || !f8_in || !f16_out || num_tokens < 0 || embed_dim <= 0 || (embed_dim & 63) != 0) {
        return -1;
    }
    if (num_tokens == 0) return 0;
    cudaStream_t stream = reinterpret_cast<cudaStream_t>(ex->stream);
    w34::ple_dequant_f8_hadamard_to_f16_kernel<<<num_tokens, 256, 0, stream>>>(
        f8_in,
        static_cast<__half*>(f16_out),
        valid_token_mask,
        num_tokens,
        embed_dim,
        weight_scale);
    return (cudaGetLastError() == cudaSuccess) ? 0 : -2;
}

int w34_ple_dequant_lut4_to_f16(
    const w34_exec* ex,
    const uint8_t* raw_in,
    const float* lut16,
    void* f16_out,
    const uint8_t* valid_token_mask,
    int num_tokens,
    int num_heads,
    int head_dim,
    int group_size) {
    if (!ex || !raw_in || !lut16 || !f16_out || num_tokens < 0 ||
        num_heads <= 0 || head_dim <= 0 || (head_dim % 2) != 0 ||
        group_size <= 0 || (head_dim % group_size) != 0) {
        return -1;
    }
    if (num_tokens == 0) return 0;
    cudaStream_t stream = reinterpret_cast<cudaStream_t>(ex->stream);
    w34::ple_dequant_lut4_to_f16_kernel<<<num_tokens, 256, 0, stream>>>(
        raw_in,
        lut16,
        static_cast<__half*>(f16_out),
        valid_token_mask,
        num_tokens,
        num_heads,
        head_dim,
        group_size);
    return (cudaGetLastError() == cudaSuccess) ? 0 : -2;
}

int w34_ple_dequant_lut4_hadamard_to_f16(
    const w34_exec* ex,
    const uint8_t* raw_in,
    const float* lut16,
    void* f16_out,
    const uint8_t* valid_token_mask,
    int num_tokens,
    int num_heads,
    int head_dim,
    int group_size) {
    if (!ex || !raw_in || !lut16 || !f16_out || num_tokens < 0 ||
        num_heads <= 0 || head_dim <= 0 || (head_dim % 2) != 0 ||
        group_size <= 0 || (head_dim % group_size) != 0 ||
        ((num_heads * head_dim) & 63) != 0) {
        return -1;
    }
    if (num_tokens == 0) return 0;
    cudaStream_t stream = reinterpret_cast<cudaStream_t>(ex->stream);
    w34::ple_dequant_lut4_hadamard_to_f16_kernel<<<num_tokens, 256, 0, stream>>>(
        raw_in,
        lut16,
        static_cast<__half*>(f16_out),
        valid_token_mask,
        num_tokens,
        num_heads,
        head_dim,
        group_size);
    return (cudaGetLastError() == cudaSuccess) ? 0 : -2;
}

int w34_ple_dilated_conv1d_f16(
    const w34_exec* ex,
    const void* key_in,
    const void* conv_weight,
    void* conv_state_pool,
    void* key_out,
    const w34_gdn_batch* batch,
    int conv_dim) {
    return w34_ple_dilated_conv1d_add_f16(
        ex, key_in, conv_weight, conv_state_pool, key_out, nullptr, batch, conv_dim);
}

int w34_ple_dilated_conv1d_add_f16(
    const w34_exec* ex,
    const void* key_in,
    const void* conv_weight,
    void* conv_state_pool,
    void* key_out,
    void* residual_inout,
    const w34_gdn_batch* batch,
    int conv_dim) {
    if (!ex || !key_in || !conv_weight || (!key_out && !residual_inout) || !batch || conv_dim <= 0) {
        return -1;
    }
    if (batch->num_reqs <= 0 || batch->total_tokens <= 0) return 0;
    cudaStream_t stream = reinterpret_cast<cudaStream_t>(ex->stream);
    dim3 grid((conv_dim + 255) / 256, batch->num_reqs);
    w34::ple_dilated_conv1d_f16_kernel<<<grid, 256, 0, stream>>>(
        static_cast<const __half*>(key_in),
        static_cast<const __half*>(conv_weight),
        static_cast<__half*>(conv_state_pool),
        static_cast<__half*>(key_out),
        static_cast<__half*>(residual_inout),
        *batch,
        conv_dim);
    return (cudaGetLastError() == cudaSuccess) ? 0 : -2;
}

int w34_ple_signed_sqrt_gate_f16(
    const w34_exec* ex,
    const void* norm_query,
    const void* norm_key,
    const void* value_in,
    void* ple_delta_out,
    const uint8_t* valid_token_mask,
    int num_tokens,
    int hc_count,
    int embed_dim,
    float eps) {
    if (!ex || !norm_query || !norm_key || !value_in || !ple_delta_out || num_tokens < 0 ||
        hc_count <= 0 || embed_dim <= 0 || hc_count > 8) {
        return -1;
    }
    if (num_tokens == 0) return 0;
    cudaStream_t stream = reinterpret_cast<cudaStream_t>(ex->stream);
    w34::ple_signed_sqrt_gate_f16_kernel<<<num_tokens, 256, 0, stream>>>(
        static_cast<const __half*>(norm_query),
        static_cast<const __half*>(norm_key),
        static_cast<const __half*>(value_in),
        static_cast<__half*>(ple_delta_out),
        valid_token_mask,
        num_tokens,
        hc_count,
        embed_dim,
        eps);
    return (cudaGetLastError() == cudaSuccess) ? 0 : -2;
}

int w34_ple_signed_sqrt_gate_norm_add_f16(
    const w34_exec* ex,
    const void* norm_query,
    const void* norm_key,
    const void* value_in,
    void* ple_delta_out,
    void* residual_inout,
    const void* norm_conv_weight,
    void* norm_conv_out,
    const uint8_t* valid_token_mask,
    int num_tokens,
    int hc_count,
    int embed_dim,
    float eps,
    float norm_eps) {
    if (!ex || !norm_query || !norm_key || !value_in ||
        (!ple_delta_out && !residual_inout && !norm_conv_out) ||
        ((norm_conv_out == nullptr) != (norm_conv_weight == nullptr)) ||
        num_tokens < 0 || hc_count <= 0 || embed_dim <= 0 || hc_count > 8) {
        return -1;
    }
    if (num_tokens == 0) return 0;
    cudaStream_t stream = reinterpret_cast<cudaStream_t>(ex->stream);
    w34::ple_signed_sqrt_gate_norm_add_f16_kernel<<<num_tokens, 256, 0, stream>>>(
        static_cast<const __half*>(norm_query),
        static_cast<const __half*>(norm_key),
        static_cast<const __half*>(value_in),
        static_cast<__half*>(ple_delta_out),
        static_cast<__half*>(residual_inout),
        static_cast<const __half*>(norm_conv_weight),
        static_cast<__half*>(norm_conv_out),
        valid_token_mask,
        num_tokens,
        hc_count,
        embed_dim,
        eps,
        norm_eps);
    return (cudaGetLastError() == cudaSuccess) ? 0 : -2;
}

} // extern "C"
