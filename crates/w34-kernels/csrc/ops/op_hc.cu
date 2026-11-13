// Qwen3.8-Flash-Next (qwen4_exp) 4-Stream HyperConnection (GatedResidual) CUDA Kernels.
// Reference: nvidia/hc.py & nvidia/hyperconnection.py
//   - Per-stream GemmaRMSNorm (eps=1e-6, weight_offset=+1.0) over [M, 4, 2560]
//   - Low-rank down activation: SiLU(down / 4.0) over [M, 320]
//   - Stream injection weights: 2.0 * sigmoid(inj / 4.0) over [M, 4]
//   - Stream mixing: sum_{s=0..3} (sigmoid(up[m, s, d]) / 4.0) * residual[m, s, d] -> [M, 2560]
//   - Stream combine: residual[m, s, d] += inject_w[m, s] * block_out[m, d] (+ ple_delta[m, s, d])
#include "../w34_cuda.h"
#include "hadamard_inline.cuh"

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <math.h>

namespace w34 {

__device__ __forceinline__ float hc_warp_reduce_sum(float val) {
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        val += __shfl_xor_sync(0xffffffffu, val, offset);
    }
    return val;
}

__device__ __forceinline__ float hc_block_reduce_sum(float val, float* s_warp) {
    const int lane = threadIdx.x & 31;
    const int wid = threadIdx.x >> 5;
    val = hc_warp_reduce_sum(val);
    if (lane == 0) {
        s_warp[wid] = val;
    }
    __syncthreads();
    const int num_warps = blockDim.x >> 5;
    val = (threadIdx.x < num_warps) ? s_warp[lane] : 0.0f;
    if (wid == 0) {
        val = hc_warp_reduce_sum(val);
        if (lane == 0) {
            s_warp[0] = val;
        }
    }
    __syncthreads();
    const float out = s_warp[0];
    __syncthreads();
    return out;
}

// Replicates initial token embeddings [num_tokens, hidden_size] across hc_count streams
// into [num_tokens, hc_count, hidden_size].
__global__ void hc_init_streams_f16_kernel(
    const __half* __restrict__ embed_in,
    __half* __restrict__ residual_out,
    const uint8_t* __restrict__ valid_token_mask,
    int num_tokens,
    int hc_count,
    int hidden_size) {
    const int tok = blockIdx.x;
    if (tok >= num_tokens) return;
    const bool valid = (valid_token_mask == nullptr) || (valid_token_mask[tok] != 0);
    const __half* src = embed_in + (size_t)tok * hidden_size;
    __half* dst_base = residual_out + (size_t)tok * hc_count * hidden_size;
    for (int d = threadIdx.x; d < hidden_size; d += blockDim.x) {
        const __half v = valid ? src[d] : __float2half(0.0f);
        for (int s = 0; s < hc_count; ++s) {
            dst_base[(size_t)s * hidden_size + d] = v;
        }
    }
}

// Per-stream GemmaRMSNorm over [num_tokens, hc_count, hidden_size] with weight [hc_count, hidden_size].
// Grid: (num_tokens, hc_count), Block: 256.
__global__ void hc_rmsnorm_f16_kernel(
    const __half* residual_in,
    const __half* __restrict__ norm_weight,
    __half* normed_out,
    const uint8_t* __restrict__ valid_token_mask,
    int num_tokens,
    int hc_count,
    int hidden_size,
    float eps) {
    const int tok = blockIdx.x;
    const int stream_idx = blockIdx.y;
    if (tok >= num_tokens || stream_idx >= hc_count) return;

    const size_t stream_offset = ((size_t)tok * hc_count + stream_idx) * hidden_size;
    __half* dst = normed_out + stream_offset;
    if (valid_token_mask != nullptr && valid_token_mask[tok] == 0) {
        for (int d = threadIdx.x; d < hidden_size; d += blockDim.x) {
            dst[d] = __float2half(0.0f);
        }
        return;
    }

    const __half* src = residual_in + stream_offset;
    const __half* w = norm_weight + (size_t)stream_idx * hidden_size;

    __shared__ float s_warp[32];
    float sum_sq = 0.0f;
    for (int d = threadIdx.x; d < hidden_size; d += blockDim.x) {
        const float v = __half2float(src[d]);
        sum_sq += v * v;
    }
    sum_sq = hc_block_reduce_sum(sum_sq, s_warp);
    const float inv_rms = rsqrtf(sum_sq / (float)hidden_size + eps);

    for (int d = threadIdx.x; d < hidden_size; d += blockDim.x) {
        const float v = __half2float(src[d]) * inv_rms;
        const float wf = 1.0f + __half2float(w[d]);
        dst[d] = __float2half(v * wf);
    }
}

// Applies SiLU(down / hc_count) in-place or out-of-place on [num_tokens, hc_rank_dim]
// and optionally computes inject_weights = 2.0 * sigmoid(inj / hc_count) on [num_tokens, hc_count].
__global__ void hc_down_silu_inject_f16_kernel(
    const __half* down_in,
    __half* down_out,
    const __half* __restrict__ inj_in,
    float* __restrict__ inject_weights_out,
    const uint8_t* __restrict__ valid_token_mask,
    int num_tokens,
    int hc_rank_dim,
    int hc_count) {
    const int tok = blockIdx.x;
    if (tok >= num_tokens) return;
    const bool valid = (valid_token_mask == nullptr) || (valid_token_mask[tok] != 0);
    const float inv_hc = 1.0f / (float)hc_count;

    const __half* d_in = down_in + (size_t)tok * hc_rank_dim;
    __half* d_out = down_out + (size_t)tok * hc_rank_dim;
    for (int c = threadIdx.x; c < hc_rank_dim; c += blockDim.x) {
        if (!valid) {
            d_out[c] = __float2half(0.0f);
        } else {
            const float x = __half2float(d_in[c]) * inv_hc;
            const float silu = x / (1.0f + expf(-x));
            d_out[c] = __float2half(silu);
        }
    }

    if (inj_in != nullptr && inject_weights_out != nullptr && threadIdx.x < hc_count) {
        const int s = threadIdx.x;
        if (!valid) {
            inject_weights_out[(size_t)tok * hc_count + s] = 0.0f;
        } else {
            const float x = __half2float(inj_in[(size_t)tok * hc_count + s]) * inv_hc;
            const float sig = 1.0f / (1.0f + expf(-x));
            inject_weights_out[(size_t)tok * hc_count + s] = 2.0f * sig;
        }
    }
}

// Mixes hc_count streams into a single hidden state [num_tokens, hidden_size]:
//   out_mixed[tok, d] = sum_{s=0..hc_count-1} (sigmoid(up_raw[tok, s, d]) / hc_count) * residual[tok, s, d]
__global__ void hc_mix_streams_f16_kernel(
    const __half* __restrict__ residual_in,
    const __half* __restrict__ up_raw_in,
    __half* __restrict__ mixed_out,
    const uint8_t* __restrict__ valid_token_mask,
    int num_tokens,
    int hc_count,
    int hidden_size) {
    const int tok = blockIdx.x;
    if (tok >= num_tokens) return;
    __half* dst = mixed_out + (size_t)tok * hidden_size;
    if (valid_token_mask != nullptr && valid_token_mask[tok] == 0) {
        for (int d = threadIdx.x; d < hidden_size; d += blockDim.x) {
            dst[d] = __float2half(0.0f);
        }
        return;
    }

    const size_t base = (size_t)tok * hc_count * hidden_size;
    const __half* res_tok = residual_in + base;
    const __half* up_tok = up_raw_in + base;
    const float inv_hc = 1.0f / (float)hc_count;

    for (int d = threadIdx.x; d < hidden_size; d += blockDim.x) {
        float acc = 0.0f;
        for (int s = 0; s < hc_count; ++s) {
            const size_t idx = (size_t)s * hidden_size + d;
            const float u = __half2float(up_tok[idx]);
            const float gate = inv_hc / (1.0f + expf(-u));
            const float r = __half2float(res_tok[idx]);
            acc += gate * r;
        }
        dst[d] = __float2half(acc);
    }
}

__global__ void hc_mix_streams_hadamard_f16_kernel(
    const __half* __restrict__ residual_in,
    const __half* __restrict__ up_raw_in,
    __half* __restrict__ mixed_out,
    __half* __restrict__ had64_out,
    const uint8_t* __restrict__ valid_token_mask,
    int num_tokens,
    int hc_count,
    int hidden_size) {
    const int tok = blockIdx.x;
    if (tok >= num_tokens) return;
    __half* dst = (mixed_out != nullptr) ? (mixed_out + (size_t)tok * hidden_size) : nullptr;
    __half* dst_had = (had64_out != nullptr) ? (had64_out + (size_t)tok * hidden_size) : nullptr;
    if (valid_token_mask != nullptr && valid_token_mask[tok] == 0) {
        for (int d = threadIdx.x; d < hidden_size; d += blockDim.x) {
            if (dst != nullptr) dst[d] = __float2half(0.0f);
            if (dst_had != nullptr) dst_had[d] = __float2half(0.0f);
        }
        return;
    }

    const size_t base = (size_t)tok * hc_count * hidden_size;
    const __half* res_tok = residual_in + base;
    const __half* up_tok = up_raw_in + base;
    const float inv_hc = 1.0f / (float)hc_count;

    for (int base_col = 0; base_col < hidden_size; base_col += (int)blockDim.x * 4) {
        const int col0 = base_col + (int)threadIdx.x * 4;
        if (col0 < hidden_size) {
            __half h[4];
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                const int d = col0 + j;
                float acc = 0.0f;
                for (int s = 0; s < hc_count; ++s) {
                    const size_t idx = (size_t)s * hidden_size + d;
                    const float u = __half2float(up_tok[idx]);
                    const float gate = inv_hc / (1.0f + expf(-u));
                    const float r = __half2float(res_tok[idx]);
                    acc += gate * r;
                }
                h[j] = __float2half(acc);
            }
            const uint2 v = *reinterpret_cast<const uint2*>(h);
            if (dst != nullptr) {
                *reinterpret_cast<uint2*>(&dst[col0]) = v;
            }
            if (dst_had != nullptr) {
                *reinterpret_cast<uint2*>(&dst_had[col0]) = hadamard64_vec4_f16(v);
            }
        }
    }
}

// Combines block output [num_tokens, hidden_size] (and optional PLE delta [num_tokens, hc_count, hidden_size])
// back into the hc_count-stream residual state [num_tokens, hc_count, hidden_size]:
//   residual_out[tok, s, d] = residual_in[tok, s, d] + inject_w[tok, s] * block_out[tok, d] (+ ple_delta[tok, s, d])
// When `tap_bf16_out != nullptr`, also writes the completed layer's stream average:
//   tap_bf16_out[tok * tap_stride_elems + tap_offset_elems + d] =
//       bf16_rn((1 / hc_count) * sum_{s=0..hc_count-1} f32(residual_out[tok, s, d]))
__global__ void hc_combine_f16_kernel(
    const __half* residual_in,
    const __half* __restrict__ block_out,
    const float* __restrict__ inject_weights,
    const __half* ple_delta,
    __half* residual_out,
    __nv_bfloat16* __restrict__ tap_bf16_out,
    const uint8_t* __restrict__ valid_token_mask,
    int num_tokens,
    int hc_count,
    int hidden_size,
    int tap_stride_elems,
    int tap_offset_elems) {
    const int tok = blockIdx.x;
    if (tok >= num_tokens) return;
    if (valid_token_mask != nullptr && valid_token_mask[tok] == 0) {
        return;
    }

    const size_t base = (size_t)tok * hc_count * hidden_size;
    const __half* res_in_tok = residual_in + base;
    __half* res_out_tok = residual_out + base;
    const __half* blk_tok = block_out + (size_t)tok * hidden_size;
    const __half* ple_tok = (ple_delta != nullptr) ? (ple_delta + base) : nullptr;
    const float* inj_tok = inject_weights + (size_t)tok * hc_count;
    __nv_bfloat16* tap_row =
        (tap_bf16_out != nullptr)
            ? (tap_bf16_out + (size_t)tok * tap_stride_elems + tap_offset_elems)
            : nullptr;
    const float inv_hc = 1.0f / static_cast<float>(hc_count);

    for (int d = threadIdx.x; d < hidden_size; d += blockDim.x) {
        const float y = __half2float(blk_tok[d]);
        float tap_sum = 0.0f;
        for (int s = 0; s < hc_count; ++s) {
            const size_t idx = (size_t)s * hidden_size + d;
            float val = __half2float(res_in_tok[idx]) + inj_tok[s] * y;
            if (ple_tok != nullptr) {
                val += __half2float(ple_tok[idx]);
            }
            const __half out_h = __float2half(val);
            res_out_tok[idx] = out_h;
            if (tap_row != nullptr) {
                tap_sum += __half2float(out_h);
            }
        }
        if (tap_row != nullptr) {
            tap_row[d] = __float2bfloat16_rn(tap_sum * inv_hc);
        }
    }
}

} // namespace w34

extern "C" {

int w34_hc_init_streams_f16(
    const w34_exec* ex,
    const void* embed_in,
    void* residual_out,
    const uint8_t* valid_token_mask,
    int num_tokens,
    int hc_count,
    int hidden_size) {
    if (!ex || !embed_in || !residual_out || num_tokens < 0 || hc_count <= 0 || hidden_size <= 0) {
        return -1;
    }
    if (num_tokens == 0) return 0;
    cudaStream_t stream = reinterpret_cast<cudaStream_t>(ex->stream);
    w34::hc_init_streams_f16_kernel<<<num_tokens, 256, 0, stream>>>(
         static_cast<const __half*>(embed_in),
        static_cast<__half*>(residual_out),
        valid_token_mask,
        num_tokens,
        hc_count,
        hidden_size);
    return (cudaGetLastError() == cudaSuccess) ? 0 : -2;
}

int w34_hc_rmsnorm_f16(
    const w34_exec* ex,
    const void* residual_in,
    const void* norm_weight,
    void* normed_out,
    const uint8_t* valid_token_mask,
    int num_tokens,
    int hc_count,
    int hidden_size,
    float eps) {
    if (!ex || !residual_in || !norm_weight || !normed_out || num_tokens < 0 || hc_count <= 0 ||
        hidden_size <= 0) {
        return -1;
    }
    if (num_tokens == 0) return 0;
    cudaStream_t stream = reinterpret_cast<cudaStream_t>(ex->stream);
    dim3 grid(num_tokens, hc_count);
    w34::hc_rmsnorm_f16_kernel<<<grid, 256, 0, stream>>>(
        static_cast<const __half*>(residual_in),
        static_cast<const __half*>(norm_weight),
        static_cast<__half*>(normed_out),
        valid_token_mask,
        num_tokens,
        hc_count,
        hidden_size,
        eps);
    return (cudaGetLastError() == cudaSuccess) ? 0 : -2;
}

int w34_hc_down_silu_inject_f16(
    const w34_exec* ex,
    const void* down_in,
    void* down_out,
    const void* inj_in,
    float* inject_weights_out,
    const uint8_t* valid_token_mask,
    int num_tokens,
    int hc_rank_dim,
    int hc_count) {
    if (!ex || !down_in || !down_out || num_tokens < 0 || hc_rank_dim <= 0 || hc_count <= 0) {
        return -1;
    }
    if (num_tokens == 0) return 0;
    cudaStream_t stream = reinterpret_cast<cudaStream_t>(ex->stream);
    w34::hc_down_silu_inject_f16_kernel<<<num_tokens, 256, 0, stream>>>(
        static_cast<const __half*>(down_in),
        static_cast<__half*>(down_out),
        static_cast<const __half*>(inj_in),
        inject_weights_out,
        valid_token_mask,
        num_tokens,
        hc_rank_dim,
        hc_count);
    return (cudaGetLastError() == cudaSuccess) ? 0 : -2;
}

int w34_hc_mix_streams_f16(
    const w34_exec* ex,
    const void* residual_in,
    const void* up_raw_in,
    void* mixed_out,
    const uint8_t* valid_token_mask,
    int num_tokens,
    int hc_count,
    int hidden_size) {
    if (!ex || !residual_in || !up_raw_in || !mixed_out || num_tokens < 0 || hc_count <= 0 ||
        hidden_size <= 0 || hc_count > 8) {
        return -1;
    }
    if (num_tokens == 0) return 0;
    cudaStream_t stream = reinterpret_cast<cudaStream_t>(ex->stream);
    w34::hc_mix_streams_f16_kernel<<<num_tokens, 256, 0, stream>>>(
        static_cast<const __half*>(residual_in),
        static_cast<const __half*>(up_raw_in),
        static_cast<__half*>(mixed_out),
        valid_token_mask,
        num_tokens,
        hc_count,
        hidden_size);
    return (cudaGetLastError() == cudaSuccess) ? 0 : -2;
}

int w34_hc_mix_streams_hadamard_f16(
    const w34_exec* ex,
    const void* residual_in,
    const void* up_raw_in,
    void* mixed_out,
    void* had64_out,
    const uint8_t* valid_token_mask,
    int num_tokens,
    int hc_count,
    int hidden_size) {
    if (!had64_out) {
        return w34_hc_mix_streams_f16(
            ex, residual_in, up_raw_in, mixed_out, valid_token_mask,
            num_tokens, hc_count, hidden_size);
    }
    if (!ex || !residual_in || !up_raw_in || num_tokens < 0 || hc_count <= 0 ||
        hidden_size <= 0 || hc_count > 8 || (hidden_size & 63) != 0) {
        return -1;
    }
    if (num_tokens == 0) return 0;
    cudaStream_t stream = reinterpret_cast<cudaStream_t>(ex->stream);
    w34::hc_mix_streams_hadamard_f16_kernel<<<num_tokens, 256, 0, stream>>>(
        static_cast<const __half*>(residual_in),
        static_cast<const __half*>(up_raw_in),
        static_cast<__half*>(mixed_out),
        static_cast<__half*>(had64_out),
        valid_token_mask,
        num_tokens,
        hc_count,
        hidden_size);
    return (cudaGetLastError() == cudaSuccess) ? 0 : -2;
}

int w34_hc_combine_tap_f16(
    const w34_exec* ex,
    const void* residual_in,
    const void* block_out,
    const float* inject_weights,
    const void* ple_delta,
    void* residual_out,
    void* tap_bf16_out,
    const uint8_t* valid_token_mask,
    int num_tokens,
    int hc_count,
    int hidden_size,
    int tap_stride_elems,
    int tap_offset_elems) {
    if (!ex || !residual_in || !block_out || !inject_weights || !residual_out || num_tokens < 0 ||
        hc_count <= 0 || hidden_size <= 0 || hc_count > 8) {
        return -1;
    }
    if (tap_bf16_out != nullptr &&
        (tap_offset_elems < 0 || tap_stride_elems < tap_offset_elems + hidden_size)) {
        return -1;
    }
    if (num_tokens == 0) return 0;
    cudaStream_t stream = reinterpret_cast<cudaStream_t>(ex->stream);
    w34::hc_combine_f16_kernel<<<num_tokens, 256, 0, stream>>>(
        static_cast<const __half*>(residual_in),
        static_cast<const __half*>(block_out),
        inject_weights,
        static_cast<const __half*>(ple_delta),
        static_cast<__half*>(residual_out),
        static_cast<__nv_bfloat16*>(tap_bf16_out),
        valid_token_mask,
        num_tokens,
        hc_count,
        hidden_size,
        tap_stride_elems,
        tap_offset_elems);
    return (cudaGetLastError() == cudaSuccess) ? 0 : -2;
}

int w34_hc_combine_f16(
    const w34_exec* ex,
    const void* residual_in,
    const void* block_out,
    const float* inject_weights,
    const void* ple_delta,
    void* residual_out,
    const uint8_t* valid_token_mask,
    int num_tokens,
    int hc_count,
    int hidden_size) {
    return w34_hc_combine_tap_f16(
        ex,
        residual_in,
        block_out,
        inject_weights,
        ple_delta,
        residual_out,
        nullptr,
        valid_token_mask,
        num_tokens,
        hc_count,
        hidden_size,
        0,
        0);
}

__global__ void qwen4_mtp_broadcast_add_f16_kernel(
    const __half* __restrict__ fc_embed,
    const __half* __restrict__ fc_hidden,
    __half* __restrict__ out_multi_hidden,
    int num_tokens,
    int hc_count,
    int hidden_size) {
    const int tok = blockIdx.x;
    if (tok >= num_tokens) return;
    const __half* emb_row = fc_embed + static_cast<size_t>(tok) * hidden_size;
    for (int d = threadIdx.x; d < hidden_size; d += blockDim.x) {
        const __half e = emb_row[d];
        for (int b = 0; b < hc_count; ++b) {
            const size_t idx =
                (static_cast<size_t>(tok) * hc_count + b) * hidden_size + d;
            out_multi_hidden[idx] = __hadd(e, fc_hidden[idx]);
        }
    }
}

int w34_qwen4_mtp_broadcast_add_f16(
    const w34_exec* ex,
    const void* fc_embed,
    const void* fc_hidden,
    void* out_multi_hidden,
    int num_tokens,
    int hc_count,
    int hidden_size) {
    if (!ex || !fc_embed || !fc_hidden || !out_multi_hidden || num_tokens < 0 ||
        hc_count <= 0 || hidden_size <= 0 || hc_count > 8) {
        return -1;
    }
    if (num_tokens == 0) return 0;
    cudaStream_t stream = reinterpret_cast<cudaStream_t>(ex->stream);
    qwen4_mtp_broadcast_add_f16_kernel<<<num_tokens, 256, 0, stream>>>(
        static_cast<const __half*>(fc_embed),
        static_cast<const __half*>(fc_hidden),
        static_cast<__half*>(out_multi_hidden),
        num_tokens,
        hc_count,
        hidden_size);
    return (cudaGetLastError() == cudaSuccess) ? 0 : -2;
}

} // extern "C"
