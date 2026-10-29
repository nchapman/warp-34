// Norms, RoPE, Embedding, and Activation kernels for warp-34.
// Donor: vLLM 0.31.0 (GemmaRMSNorm 1.0 + weight, fused_qk_rmsnorm_rope_gate,
// RMSNormGated(norm_before_gate=True, activation="silu"), silu_and_mul).
#include "../w34_cuda.h"

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <math.h>

namespace w34 {

__device__ __forceinline__ float warp_reduce_sum(float val) {
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        val += __shfl_xor_sync(0xffffffffu, val, offset);
    }
    return val;
}

__device__ __forceinline__ float block_reduce_sum(float val, float* shared_warp_sums) {
    const int lane = threadIdx.x & 31;
    const int wid = threadIdx.x >> 5;
    val = warp_reduce_sum(val);
    if (lane == 0) {
        shared_warp_sums[wid] = val;
    }
    __syncthreads();
    const int num_warps = blockDim.x >> 5;
    val = (threadIdx.x < num_warps) ? shared_warp_sums[lane] : 0.0f;
    if (wid == 0) {
        val = warp_reduce_sum(val);
        if (lane == 0) {
            shared_warp_sums[0] = val;
        }
    }
    __syncthreads();
    return shared_warp_sums[0];
}

__global__ void embed_lookup_f16_kernel(
    const int32_t* __restrict__ token_ids,
    const __half* __restrict__ embed_weight,
    __half* __restrict__ out_hidden,
    const uint8_t* __restrict__ valid_token_mask,
    int num_tokens,
    int hidden_size,
    int vocab_size) {
    const int row = blockIdx.x;
    if (row >= num_tokens) return;
    if (valid_token_mask != nullptr && valid_token_mask[row] == 0) {
        for (int c = threadIdx.x; c < hidden_size; c += blockDim.x) {
            out_hidden[(size_t)row * hidden_size + c] = __float2half(0.0f);
        }
        return;
    }
    int32_t tok = token_ids[row];
    if (tok < 0 || tok >= vocab_size) tok = 0;
    const __half* src = embed_weight + (size_t)tok * hidden_size;
    __half* dst = out_hidden + (size_t)row * hidden_size;
    for (int c = threadIdx.x; c < hidden_size; c += blockDim.x) {
        dst[c] = src[c];
    }
}

__global__ void copy_masked_rows_f16_kernel(
    const __half* __restrict__ src,
    __half* __restrict__ dst,
    const uint8_t* __restrict__ valid_token_mask,
    int num_tokens,
    int row_elems) {
    const int row = blockIdx.x;
    if (row >= num_tokens) return;
    if (valid_token_mask != nullptr && valid_token_mask[row] == 0) return;
    const __half* s = src + (size_t)row * row_elems;
    __half* d = dst + (size_t)row * row_elems;
    for (int c = threadIdx.x; c < row_elems; c += blockDim.x) {
        d[c] = s[c];
    }
}

template <int MAX_ITERS>
__global__ void gemma_rms_norm_f16_kernel(
    const __half* __restrict__ x,
    const __half* __restrict__ weight,
    __half* __restrict__ out,
    const uint8_t* __restrict__ valid_token_mask,
    int hidden_size,
    float eps) {
    const int row = blockIdx.x;
    if (valid_token_mask != nullptr && valid_token_mask[row] == 0) return;
    const __half* x_row = x + (size_t)row * hidden_size;
    __half* out_row = out + (size_t)row * hidden_size;

    __shared__ float s_warp[32];
    float reg_v[MAX_ITERS];
    float sum_sq = 0.0f;
#pragma unroll
    for (int k = 0; k < MAX_ITERS; k++) {
        const int i = threadIdx.x + k * blockDim.x;
        if (i < hidden_size) {
            float v = __half2float(x_row[i]);
            reg_v[k] = v;
            sum_sq += v * v;
        }
    }
    sum_sq = block_reduce_sum(sum_sq, s_warp);
    const float inv_rms = rsqrtf(sum_sq / (float)hidden_size + eps);

#pragma unroll
    for (int k = 0; k < MAX_ITERS; k++) {
        const int i = threadIdx.x + k * blockDim.x;
        if (i < hidden_size) {
            float v = reg_v[k] * inv_rms;
            float w = 1.0f + __half2float(weight[i]);
            out_row[i] = __float2half(v * w);
        }
    }
}

template <int MAX_ITERS>
__global__ void fused_add_gemma_rms_norm_f16_kernel(
    const __half* __restrict__ x,
    __half* __restrict__ residual,
    const __half* __restrict__ weight,
    __half* __restrict__ out,
    __nv_bfloat16* __restrict__ tap_dst_bf16,
    const uint8_t* __restrict__ valid_token_mask,
    int hidden_size,
    float eps,
    int tap_row_stride,
    int tap_col_offset) {
    const int row = blockIdx.x;
    if (valid_token_mask != nullptr && valid_token_mask[row] == 0) return;
    const __half* x_row = x + (size_t)row * hidden_size;
    __half* res_row = residual + (size_t)row * hidden_size;
    __half* out_row = out + (size_t)row * hidden_size;
    __nv_bfloat16* tap_row = (tap_dst_bf16 != nullptr)
        ? (tap_dst_bf16 + (size_t)row * tap_row_stride + tap_col_offset)
        : nullptr;

    __shared__ float s_warp[32];
    float reg_v[MAX_ITERS];
    float sum_sq = 0.0f;
#pragma unroll
    for (int k = 0; k < MAX_ITERS; k++) {
        const int i = threadIdx.x + k * blockDim.x;
        if (i < hidden_size) {
            const __half x_val = x_row[i];
            const __half res_old = res_row[i];
            if (tap_row != nullptr) {
                tap_row[i] = __float2bfloat16(__half2float(res_old) + __half2float(x_val));
            }
            __half sum_h = __hadd(x_val, res_old);
            res_row[i] = sum_h;
            float v = __half2float(sum_h);
            reg_v[k] = v;
            sum_sq += v * v;
        }
    }
    sum_sq = block_reduce_sum(sum_sq, s_warp);
    const float inv_rms = rsqrtf(sum_sq / (float)hidden_size + eps);

#pragma unroll
    for (int k = 0; k < MAX_ITERS; k++) {
        const int i = threadIdx.x + k * blockDim.x;
        if (i < hidden_size) {
            float v = reg_v[k] * inv_rms;
            float w = 1.0f + __half2float(weight[i]);
            out_row[i] = __float2half(v * w);
        }
    }
}

// RMSNormGated(norm_before_gate=True, activation="silu"):
// Each block processes one (token, v_head) row of length head_dim (e.g. 128).
__global__ void rms_norm_gated_silu_f16_kernel(
    const __half* __restrict__ x,
    const __half* __restrict__ z,
    const __half* __restrict__ weight,
    __half* __restrict__ out,
    const uint8_t* __restrict__ valid_token_mask,
    int num_v_heads,
    int head_dim,
    int z_token_stride,
    float eps) {
    const int row = blockIdx.x;
    const int token_idx = row / num_v_heads;
    const int head_idx = row % num_v_heads;
    if (valid_token_mask != nullptr && valid_token_mask[token_idx] == 0) return;

    const __half* x_row = x + (size_t)row * head_dim;
    const __half* z_row = z + (size_t)token_idx * z_token_stride + (size_t)head_idx * head_dim;
    __half* out_row = out + (size_t)row * head_dim;

    __shared__ float s_warp[32];
    float sum_sq = 0.0f;
    for (int i = threadIdx.x; i < head_dim; i += blockDim.x) {
        float v = __half2float(x_row[i]);
        sum_sq += v * v;
    }
    sum_sq = block_reduce_sum(sum_sq, s_warp);
    const float inv_rms = rsqrtf(sum_sq / (float)head_dim + eps);

    for (int i = threadIdx.x; i < head_dim; i += blockDim.x) {
        float xv = __half2float(x_row[i]);
        float zv = __half2float(z_row[i]);
        float wv = __half2float(weight[i]);
        float normed = xv * inv_rms * wv;
        float gate = zv / (1.0f + expf(-zv));
        out_row[i] = __float2half(normed * gate);
    }
}

// Fused Q/Gate split + per-head GemmaRMSNorm on Q and K + partial interleaved 3D mRoPE.
// Grid: (num_tokens, num_q_heads + num_kv_heads), Block: 128 threads (or head_dim / 2).
__global__ void fused_qk_rmsnorm_rope_gate_f16_kernel(
    const __half* __restrict__ q_gate_in,
    const __half* __restrict__ k_in,
    const __half* __restrict__ q_norm_w,
    const __half* __restrict__ k_norm_w,
    const int32_t* __restrict__ positions,
    const uint8_t* __restrict__ valid_token_mask,
    __half* __restrict__ q_out,
    __half* __restrict__ k_out,
    __half* __restrict__ gate_out,
    int pos_plane_stride,
    int num_q_heads,
    int num_kv_heads,
    int head_dim,
    int rotary_dim,
    int mrope_s0,
    int mrope_s1,
    int mrope_s2,
    int q_gate_row_stride,
    int k_row_stride,
    float rope_theta,
    float rms_eps) {
    const int tok = blockIdx.x;
    if (valid_token_mask != nullptr && valid_token_mask[tok] == 0) return;

    const int head_idx = blockIdx.y;
    const bool is_q = head_idx < num_q_heads;
    const int local_head = is_q ? head_idx : (head_idx - num_q_heads);

    extern __shared__ float smem[];
    float* s_warp = smem;               // [32]
    float* s_normed = smem + 32;        // [head_dim]

    const __half* src_head = nullptr;
    const __half* norm_w = nullptr;
    __half* dst_head = nullptr;

    if (is_q) {
        const __half* qg_base =
            q_gate_in + (size_t)tok * q_gate_row_stride + (size_t)local_head * (2 * head_dim);
        src_head = qg_base;
        const __half* gate_src = qg_base + head_dim;
        __half* gate_dst = gate_out + ((size_t)tok * num_q_heads + local_head) * head_dim;
        for (int i = threadIdx.x; i < head_dim; i += blockDim.x) {
            gate_dst[i] = gate_src[i];
        }
        norm_w = q_norm_w;
        dst_head = q_out + ((size_t)tok * num_q_heads + local_head) * head_dim;
    } else {
        src_head = k_in + (size_t)tok * k_row_stride + (size_t)local_head * head_dim;
        norm_w = k_norm_w;
        dst_head = k_out + ((size_t)tok * num_kv_heads + local_head) * head_dim;
    }

    float sum_sq = 0.0f;
    for (int i = threadIdx.x; i < head_dim; i += blockDim.x) {
        float v = __half2float(src_head[i]);
        sum_sq += v * v;
    }
    sum_sq = block_reduce_sum(sum_sq, s_warp);
    const float inv_rms = rsqrtf(sum_sq / (float)head_dim + rms_eps);

    for (int i = threadIdx.x; i < head_dim; i += blockDim.x) {
        float v = __half2float(src_head[i]) * inv_rms;
        float w = 1.0f + __half2float(norm_w[i]);
        s_normed[i] = __half2float(__float2half(v * w));
    }
    __syncthreads();

    const int half_rot = rotary_dim / 2;
    for (int i = threadIdx.x; i < half_rot; i += blockDim.x) {
        int plane = 0;
        if ((i % 3) == 1 && i < mrope_s1 * 3) {
            plane = 1;
        } else if ((i % 3) == 2 && i < mrope_s2 * 3) {
            plane = 2;
        }
        const float pos = (float)positions[(size_t)plane * pos_plane_stride + tok];
        const float inv_freq = powf(rope_theta, -((float)(2 * i) / (float)rotary_dim));
        const float angle = pos * inv_freq;
        float sin_a, cos_a;
        sincosf(angle, &sin_a, &cos_a);

        const float x1 = s_normed[i];
        const float x2 = s_normed[i + half_rot];
        dst_head[i] = __float2half(x1 * cos_a - x2 * sin_a);
        dst_head[i + half_rot] = __float2half(x2 * cos_a + x1 * sin_a);
    }
    for (int i = rotary_dim + threadIdx.x; i < head_dim; i += blockDim.x) {
        dst_head[i] = __float2half(s_normed[i]);
    }
}

__global__ void silu_and_mul_f16_kernel(
    const __half* __restrict__ gate_up,
    __half* __restrict__ out,
    const uint8_t* __restrict__ valid_token_mask,
    int intermediate_size) {
    const int tok = blockIdx.y;
    if (valid_token_mask != nullptr && valid_token_mask[tok] == 0) return;
    const int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (col >= intermediate_size) return;

    const __half* row_in = gate_up + (size_t)tok * (2 * intermediate_size);
    const float g = __half2float(row_in[col]);
    const float u = __half2float(row_in[col + intermediate_size]);
    const float silu_g = g / (1.0f + expf(-g));
    out[(size_t)tok * intermediate_size + col] = __float2half(silu_g * u);
}

__global__ void silu_and_mul_split_f16_kernel(
    const __half* __restrict__ gate,
    const __half* __restrict__ up,
    __half* __restrict__ out,
    const uint8_t* __restrict__ valid_token_mask,
    int intermediate_size) {
    const int tok = blockIdx.y;
    if (valid_token_mask != nullptr && valid_token_mask[tok] == 0) return;
    const int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (col >= intermediate_size) return;

    const size_t idx = (size_t)tok * intermediate_size + col;
    const float g = __half2float(gate[idx]);
    const float u = __half2float(up[idx]);
    const float silu_g = g / (1.0f + expf(-g));
    out[idx] = __float2half(silu_g * u);
}

__global__ void apply_attn_output_gate_f16_kernel(
    const __half* __restrict__ attn_out,
    const __half* __restrict__ gate,
    __half* __restrict__ out,
    const uint8_t* __restrict__ valid_token_mask,
    int total_dim) {
    const int tok = blockIdx.y;
    if (valid_token_mask != nullptr && valid_token_mask[tok] == 0) return;
    const int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (col >= total_dim) return;

    const size_t idx = (size_t)tok * total_dim + col;
    const float a = __half2float(attn_out[idx]);
    const float g = __half2float(gate[idx]);
    const float sig = 1.0f / (1.0f + expf(-g));
    out[idx] = __float2half(a * sig);
}

}  // namespace w34

extern "C" int w34_embed_lookup_f16(const w34_exec* ex, const int32_t* token_ids,
                                    const void* embed_weight, void* out_hidden,
                                    const uint8_t* valid_token_mask,
                                    int num_tokens, int hidden_size, int vocab_size) {
    if (ex == nullptr || token_ids == nullptr || embed_weight == nullptr || out_hidden == nullptr) {
        return (int)cudaErrorInvalidValue;
    }
    if (num_tokens <= 0 || hidden_size <= 0) return 0;
    w34::embed_lookup_f16_kernel<<<num_tokens, 256, 0, (cudaStream_t)ex->stream>>>(
        token_ids, (const __half*)embed_weight, (__half*)out_hidden,
        valid_token_mask, num_tokens, hidden_size, vocab_size);
    return (int)cudaGetLastError();
}

extern "C" int w34_copy_masked_rows_f16(const w34_exec* ex, const void* src, void* dst,
                                        const uint8_t* valid_token_mask,
                                        int num_tokens, int row_elems) {
    if (ex == nullptr || src == nullptr || dst == nullptr) return (int)cudaErrorInvalidValue;
    if (num_tokens <= 0 || row_elems <= 0) return 0;
    w34::copy_masked_rows_f16_kernel<<<num_tokens, 256, 0, (cudaStream_t)ex->stream>>>(
        (const __half*)src, (__half*)dst, valid_token_mask, num_tokens, row_elems);
    return (int)cudaGetLastError();
}

extern "C" int w34_gemma_rms_norm_f16(const w34_exec* ex, const void* x, const void* weight,
                                      void* out, const uint8_t* valid_token_mask,
                                      int num_tokens, int hidden_size, float eps) {
    if (ex == nullptr || x == nullptr || weight == nullptr || out == nullptr) {
        return (int)cudaErrorInvalidValue;
    }
    if (num_tokens <= 0 || hidden_size <= 0) return 0;
    if (hidden_size <= 2560) {
        w34::gemma_rms_norm_f16_kernel<10><<<num_tokens, 256, 0, (cudaStream_t)ex->stream>>>(
            (const __half*)x, (const __half*)weight, (__half*)out,
            valid_token_mask, hidden_size, eps);
    } else if (hidden_size <= 4096) {
        w34::gemma_rms_norm_f16_kernel<16><<<num_tokens, 256, 0, (cudaStream_t)ex->stream>>>(
            (const __half*)x, (const __half*)weight, (__half*)out,
            valid_token_mask, hidden_size, eps);
    } else if (hidden_size <= 5120) {
        w34::gemma_rms_norm_f16_kernel<20><<<num_tokens, 256, 0, (cudaStream_t)ex->stream>>>(
            (const __half*)x, (const __half*)weight, (__half*)out,
            valid_token_mask, hidden_size, eps);
    } else {
        w34::gemma_rms_norm_f16_kernel<24><<<num_tokens, 256, 0, (cudaStream_t)ex->stream>>>(
            (const __half*)x, (const __half*)weight, (__half*)out,
            valid_token_mask, hidden_size, eps);
    }
    return (int)cudaGetLastError();
}

extern "C" int w34_fused_add_gemma_rms_norm_tap_f16(
    const w34_exec* ex,
    const void* x,
    void* residual,
    const void* weight,
    void* out,
    void* tap_dst_bf16,
    const uint8_t* valid_token_mask,
    int num_tokens,
    int hidden_size,
    float eps,
    int tap_row_stride,
    int tap_col_offset) {
    if (ex == nullptr || x == nullptr || residual == nullptr || weight == nullptr || out == nullptr) {
        return (int)cudaErrorInvalidValue;
    }
    if (num_tokens <= 0 || hidden_size <= 0) return 0;
    if (hidden_size > 6144) return (int)cudaErrorInvalidValue;
    if (tap_dst_bf16 != nullptr && (tap_col_offset < 0 || tap_row_stride < tap_col_offset + hidden_size)) {
        return (int)cudaErrorInvalidValue;
    }
    if (hidden_size <= 2560) {
        w34::fused_add_gemma_rms_norm_f16_kernel<10><<<num_tokens, 256, 0, (cudaStream_t)ex->stream>>>(
            (const __half*)x, (__half*)residual, (const __half*)weight, (__half*)out,
            (__nv_bfloat16*)tap_dst_bf16, valid_token_mask, hidden_size, eps,
            tap_row_stride, tap_col_offset);
    } else if (hidden_size <= 4096) {
        w34::fused_add_gemma_rms_norm_f16_kernel<16><<<num_tokens, 256, 0, (cudaStream_t)ex->stream>>>(
            (const __half*)x, (__half*)residual, (const __half*)weight, (__half*)out,
            (__nv_bfloat16*)tap_dst_bf16, valid_token_mask, hidden_size, eps,
            tap_row_stride, tap_col_offset);
    } else if (hidden_size <= 5120) {
        w34::fused_add_gemma_rms_norm_f16_kernel<20><<<num_tokens, 256, 0, (cudaStream_t)ex->stream>>>(
            (const __half*)x, (__half*)residual, (const __half*)weight, (__half*)out,
            (__nv_bfloat16*)tap_dst_bf16, valid_token_mask, hidden_size, eps,
            tap_row_stride, tap_col_offset);
    } else {
        w34::fused_add_gemma_rms_norm_f16_kernel<24><<<num_tokens, 256, 0, (cudaStream_t)ex->stream>>>(
            (const __half*)x, (__half*)residual, (const __half*)weight, (__half*)out,
            (__nv_bfloat16*)tap_dst_bf16, valid_token_mask, hidden_size, eps,
            tap_row_stride, tap_col_offset);
    }
    return (int)cudaGetLastError();
}

extern "C" int w34_fused_add_gemma_rms_norm_f16(const w34_exec* ex, const void* x, void* residual,
                                                const void* weight, void* out,
                                                const uint8_t* valid_token_mask,
                                                int num_tokens, int hidden_size, float eps) {
    return w34_fused_add_gemma_rms_norm_tap_f16(
        ex, x, residual, weight, out, nullptr, valid_token_mask, num_tokens, hidden_size, eps, 0, 0);
}

extern "C" int w34_rms_norm_gated_silu_strided_f16(
    const w34_exec* ex,
    const void* x,
    const void* z,
    const void* weight,
    void* out,
    const uint8_t* valid_token_mask,
    int num_tokens,
    int num_v_heads,
    int head_dim,
    int z_token_stride,
    float eps) {
    if (ex == nullptr || x == nullptr || z == nullptr || weight == nullptr || out == nullptr) {
        return (int)cudaErrorInvalidValue;
    }
    const int total_rows = num_tokens * num_v_heads;
    if (total_rows <= 0 || head_dim <= 0) return 0;
    if (z_token_stride < num_v_heads * head_dim) return (int)cudaErrorInvalidValue;
    w34::rms_norm_gated_silu_f16_kernel<<<total_rows, 128, 0, (cudaStream_t)ex->stream>>>(
        (const __half*)x, (const __half*)z, (const __half*)weight, (__half*)out,
        valid_token_mask, num_v_heads, head_dim, z_token_stride, eps);
    return (int)cudaGetLastError();
}

extern "C" int w34_rms_norm_gated_silu_f16(const w34_exec* ex, const void* x, const void* z,
                                           const void* weight, void* out,
                                           const uint8_t* valid_token_mask,
                                           int num_tokens, int num_v_heads, int head_dim, float eps) {
    return w34_rms_norm_gated_silu_strided_f16(
        ex, x, z, weight, out, valid_token_mask, num_tokens, num_v_heads, head_dim,
        num_v_heads * head_dim, eps);
}

extern "C" int w34_fused_qk_rmsnorm_rope_gate_strided_f16(
    const w34_exec* ex,
    const void* q_gate_in,
    const void* k_in,
    const void* q_norm_w,
    const void* k_norm_w,
    const int32_t* positions,
    const uint8_t* valid_token_mask,
    void* q_out,
    void* k_out,
    void* gate_out,
    int num_tokens,
    int pos_plane_stride,
    int num_q_heads,
    int num_kv_heads,
    int head_dim,
    int rotary_dim,
    int mrope_s0,
    int mrope_s1,
    int mrope_s2,
    int q_gate_row_stride,
    int k_row_stride,
    float rope_theta,
    float rms_eps) {
    if (ex == nullptr || q_gate_in == nullptr || k_in == nullptr || q_norm_w == nullptr ||
        k_norm_w == nullptr || positions == nullptr || q_out == nullptr ||
        k_out == nullptr || gate_out == nullptr) {
        return (int)cudaErrorInvalidValue;
    }
    if (num_tokens <= 0) return 0;
    if (q_gate_row_stride < num_q_heads * 2 * head_dim || k_row_stride < num_kv_heads * head_dim) {
        return (int)cudaErrorInvalidValue;
    }
    dim3 grid((unsigned)num_tokens, (unsigned)(num_q_heads + num_kv_heads));
    size_t smem = (32 + (size_t)head_dim) * sizeof(float);
    w34::fused_qk_rmsnorm_rope_gate_f16_kernel<<<grid, 128, smem, (cudaStream_t)ex->stream>>>(
        (const __half*)q_gate_in, (const __half*)k_in,
        (const __half*)q_norm_w, (const __half*)k_norm_w,
        positions, valid_token_mask,
        (__half*)q_out, (__half*)k_out, (__half*)gate_out,
        pos_plane_stride, num_q_heads, num_kv_heads,
        head_dim, rotary_dim, mrope_s0, mrope_s1, mrope_s2,
        q_gate_row_stride, k_row_stride,
        rope_theta, rms_eps);
    return (int)cudaGetLastError();
}

extern "C" int w34_fused_qk_rmsnorm_rope_gate_f16(
    const w34_exec* ex,
    const void* q_gate_in,
    const void* k_in,
    const void* q_norm_w,
    const void* k_norm_w,
    const int32_t* positions,
    const uint8_t* valid_token_mask,
    void* q_out,
    void* k_out,
    void* gate_out,
    int num_tokens,
    int pos_plane_stride,
    int num_q_heads,
    int num_kv_heads,
    int head_dim,
    int rotary_dim,
    int mrope_s0,
    int mrope_s1,
    int mrope_s2,
    float rope_theta,
    float rms_eps) {
    return w34_fused_qk_rmsnorm_rope_gate_strided_f16(
        ex, q_gate_in, k_in, q_norm_w, k_norm_w, positions, valid_token_mask,
        q_out, k_out, gate_out, num_tokens, pos_plane_stride, num_q_heads, num_kv_heads,
        head_dim, rotary_dim, mrope_s0, mrope_s1, mrope_s2,
        num_q_heads * 2 * head_dim, num_kv_heads * head_dim,
        rope_theta, rms_eps);
}

extern "C" int w34_silu_and_mul_f16(const w34_exec* ex, const void* gate_up, void* out,
                                    const uint8_t* valid_token_mask,
                                    int num_tokens, int intermediate_size) {
    if (ex == nullptr || gate_up == nullptr || out == nullptr) return (int)cudaErrorInvalidValue;
    if (num_tokens <= 0 || intermediate_size <= 0) return 0;
    dim3 grid((unsigned)((intermediate_size + 255) / 256), (unsigned)num_tokens);
    w34::silu_and_mul_f16_kernel<<<grid, 256, 0, (cudaStream_t)ex->stream>>>(
        (const __half*)gate_up, (__half*)out, valid_token_mask, intermediate_size);
    return (int)cudaGetLastError();
}

extern "C" int w34_silu_and_mul_split_f16(const w34_exec* ex, const void* gate, const void* up,
                                          void* out, const uint8_t* valid_token_mask,
                                          int num_tokens, int intermediate_size) {
    if (ex == nullptr || gate == nullptr || up == nullptr || out == nullptr) {
        return (int)cudaErrorInvalidValue;
    }
    if (num_tokens <= 0 || intermediate_size <= 0) return 0;
    dim3 grid((unsigned)((intermediate_size + 255) / 256), (unsigned)num_tokens);
    w34::silu_and_mul_split_f16_kernel<<<grid, 256, 0, (cudaStream_t)ex->stream>>>(
        (const __half*)gate, (const __half*)up, (__half*)out,
        valid_token_mask, intermediate_size);
    return (int)cudaGetLastError();
}

extern "C" int w34_apply_attn_output_gate_f16(const w34_exec* ex, const void* attn_out,
                                              const void* gate, void* out,
                                              const uint8_t* valid_token_mask,
                                              int num_tokens, int total_dim) {
    if (ex == nullptr || attn_out == nullptr || gate == nullptr || out == nullptr) {
        return (int)cudaErrorInvalidValue;
    }
    if (num_tokens <= 0 || total_dim <= 0) return 0;
    dim3 grid((unsigned)((total_dim + 255) / 256), (unsigned)num_tokens);
    w34::apply_attn_output_gate_f16_kernel<<<grid, 256, 0, (cudaStream_t)ex->stream>>>(
        (const __half*)attn_out, (const __half*)gate, (__half*)out,
        valid_token_mask, total_dim);
    return (int)cudaGetLastError();
}

namespace w34 {

__global__ void split_cols_2_f16_kernel(
    const __half* __restrict__ src,
    __half* __restrict__ dst0,
    __half* __restrict__ dst1,
    int dim0,
    int dim1) {
    const int tok = blockIdx.x;
    const int total = dim0 + dim1;
    const __half* row = src + (size_t)tok * total;
    __half* r0 = dst0 + (size_t)tok * dim0;
    __half* r1 = dst1 + (size_t)tok * dim1;
    for (int i = threadIdx.x; i < total; i += blockDim.x) {
        if (i < dim0) {
            r0[i] = row[i];
        } else {
            r1[i - dim0] = row[i];
        }
    }
}

__global__ void split_cols_3_f16_kernel(
    const __half* __restrict__ src,
    __half* __restrict__ dst0,
    __half* __restrict__ dst1,
    __half* __restrict__ dst2,
    int dim0,
    int dim1,
    int dim2) {
    const int tok = blockIdx.x;
    const int total = dim0 + dim1 + dim2;
    const __half* row = src + (size_t)tok * total;
    __half* r0 = dst0 + (size_t)tok * dim0;
    __half* r1 = dst1 + (size_t)tok * dim1;
    __half* r2 = dst2 + (size_t)tok * dim2;
    for (int i = threadIdx.x; i < total; i += blockDim.x) {
        if (i < dim0) {
            r0[i] = row[i];
        } else if (i < dim0 + dim1) {
            r1[i - dim0] = row[i];
        } else {
            r2[i - dim0 - dim1] = row[i];
        }
    }
}

}  // namespace w34

extern "C" int w34_split_cols_2_f16(const w34_exec* ex, const void* src, void* dst0, void* dst1,
                                    int num_tokens, int dim0, int dim1) {
    if (num_tokens <= 0) return 0;
    w34::split_cols_2_f16_kernel<<<num_tokens, 256, 0, (cudaStream_t)ex->stream>>>(
        (const __half*)src, (__half*)dst0, (__half*)dst1, dim0, dim1);
    return (int)cudaGetLastError();
}

extern "C" int w34_split_cols_3_f16(const w34_exec* ex, const void* src, void* dst0, void* dst1, void* dst2,
                                    int num_tokens, int dim0, int dim1, int dim2) {
    if (num_tokens <= 0) return 0;
    w34::split_cols_3_f16_kernel<<<num_tokens, 256, 0, (cudaStream_t)ex->stream>>>(
        (const __half*)src, (__half*)dst0, (__half*)dst1, (__half*)dst2, dim0, dim1, dim2);
    return (int)cudaGetLastError();
}

namespace w34 {

// Batch-size-invariant FP16 GEMM for small-N projections (e.g. GDN in_proj_ba, size_n <= 256):
// out[m, size_n] = x[m, size_k] * w[size_n, size_k]^T.
// Each warp computes one output column `n` across up to 8 rows `m_base .. m_base + 7`,
// accumulating in FP32 with identical per-row operation order for any `m >= 1`.
__global__ void small_n_gemm_f16_kernel(
    const __half* __restrict__ x,
    const __half* __restrict__ w,
    __half* __restrict__ out,
    int m,
    int size_n,
    int size_k) {
    const int wid = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const int n = blockIdx.x * 4 + wid;
    const int m_base = blockIdx.y * 8;
    if (n >= size_n || m_base >= m) return;

    const int rows = (m - m_base < 8) ? (m - m_base) : 8;
    const __half* w_row = w + (size_t)n * size_k;

    float acc[8] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    for (int k = lane * 8; k < size_k; k += 256) {
        const uint4 wv = *reinterpret_cast<const uint4*>(w_row + k);
        const __half* wh = reinterpret_cast<const __half*>(&wv);
        float wf[8];
#pragma unroll
        for (int j = 0; j < 8; j++) {
            wf[j] = __half2float(wh[j]);
        }
#pragma unroll
        for (int r = 0; r < 8; r++) {
            if (r < rows) {
                const uint4 xv = *reinterpret_cast<const uint4*>(
                    x + (size_t)(m_base + r) * size_k + k);
                const __half* xh = reinterpret_cast<const __half*>(&xv);
#pragma unroll
                for (int j = 0; j < 8; j++) {
                    acc[r] = fmaf(__half2float(xh[j]), wf[j], acc[r]);
                }
            }
        }
    }

#pragma unroll
    for (int r = 0; r < 8; r++) {
        if (r < rows) {
            float sum = acc[r];
#pragma unroll
            for (int off = 16; off > 0; off >>= 1) {
                sum += __shfl_xor_sync(0xffffffffu, sum, off);
            }
            if (lane == 0) {
                out[(size_t)(m_base + r) * size_n + n] = __float2half(sum);
            }
        }
    }
}

}  // namespace w34

extern "C" int w34_small_n_gemm_f16(
    const w34_exec* ex,
    const void* x,
    const void* w,
    void* out,
    int m,
    int size_n,
    int size_k) {
    if (ex == nullptr || x == nullptr || w == nullptr || out == nullptr) {
        return (int)cudaErrorInvalidValue;
    }
    if (m <= 0 || size_n <= 0 || size_k <= 0) return 0;
    if ((size_k & 7) != 0) return (int)cudaErrorInvalidValue;
    dim3 grid((unsigned)((size_n + 3) / 4), (unsigned)((m + 7) / 8));
    w34::small_n_gemm_f16_kernel<<<grid, 128, 0, (cudaStream_t)ex->stream>>>(
        (const __half*)x, (const __half*)w, (__half*)out, m, size_n, size_k);
    return (int)cudaGetLastError();
}
