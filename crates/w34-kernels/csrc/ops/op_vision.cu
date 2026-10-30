// Phase 5.2: Qwen3_VisionTransformer (ViT) + Multimodal Embedding Splice Kernels.
#include "../w34_cuda.h"

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <float.h>
#include <math.h>

// Call the accurate libdevice functions explicitly: -use_fast_math otherwise
// replaces sincosf with MUFU approximations whose absolute error loses FP16
// precision for the small angles in the native rotary cache.
extern "C" __device__ float __nv_sinf(float);
extern "C" __device__ float __nv_cosf(float);
extern "C" __device__ float __nv_tanhf(float);

namespace w34_vision {

__device__ __forceinline__ float warp_reduce_sum(float v) {
#pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
        v += __shfl_xor_sync(0xffffffffu, v, off);
    }
    return v;
}

__device__ __forceinline__ float block_reduce_sum(float v, float* s_warp) {
    const int lane = threadIdx.x & 31;
    const int wid = threadIdx.x >> 5;
    v = warp_reduce_sum(v);
    if (lane == 0) s_warp[wid] = v;
    __syncthreads();
    v = (threadIdx.x < (blockDim.x >> 5)) ? s_warp[lane] : 0.0f;
    if (wid == 0) {
        v = warp_reduce_sum(v);
        if (lane == 0) s_warp[0] = v;
    }
    __syncthreads();
    const float out = s_warp[0];
    __syncthreads();
    return out;
}

__global__ void vision_bias_add_f16_kernel(
    const __half* __restrict__ x,
    const __half* __restrict__ bias,
    const __half* __restrict__ residual,
    __half* __restrict__ out,
    int rows,
    int cols) {
    const int row = blockIdx.y;
    const int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= rows || col >= cols) return;
    const size_t idx = (size_t)row * cols + col;
    float v = __half2float(x[idx]);
    if (bias != nullptr) v += __half2float(bias[col]);
    // Native Linear materializes FP16 before the separate residual operation.
    v = __half2float(__float2half_rn(v));
    if (residual != nullptr) v += __half2float(residual[idx]);
    out[idx] = __float2half(v);
}

// Donor: PyTorch 2.13.0+cu130, cf30153c4c131c8164ee7798e5022d810682e2cb,
// aten/src/ATen/native/cuda/layer_norm_kernel.cu (cuWelfordOnlineSum,
// cuWelfordCombine, compute_stats); thread_constants.h selects 32*4 threads.
// Preserve reciprocal rounding despite this translation unit's --use_fast_math.
struct VisionWelford { float mean, sigma2, count; };
__device__ __forceinline__ VisionWelford vision_combine(VisionWelford b, VisionWelford a) {
    float delta = b.mean - a.mean;
    float count = a.count + b.count;
    if (count == 0.f) return {0.f, 0.f, 0.f};
    float coef = __frcp_rn(count);
    float na = a.count * coef, nb = b.count * coef;
    return {na * a.mean + nb * b.mean,
            a.sigma2 + b.sigma2 + delta * delta * a.count * nb, count};
}

__global__ void vision_layernorm_vectorized_f16_kernel(
    const __half* x, const __half* weight, const __half* bias, __half* out,
    int cols, float eps) {
    __shared__ float buf[6];
    const int lane = threadIdx.x, warp = threadIdx.y;
    const int tid = lane + 32 * warp;
    const size_t base = (size_t)blockIdx.x * cols;
    VisionWelford wd{0.f, 0.f, 0.f};
    for (int i = tid; i < cols / 4; i += 128) {
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            float v = __half2float(x[base + 4*i + j]);
            float delta = v - wd.mean;
            float count = wd.count + 1.f;
            float mean = wd.mean + delta * __frcp_rn(count);
            wd = {mean, wd.sigma2 + delta * (v - mean), count};
        }
    }
    for (int offset = 16; offset > 0; offset >>= 1) {
        VisionWelford b{__shfl_down_sync(0xffffffffu, wd.mean, offset),
                        __shfl_down_sync(0xffffffffu, wd.sigma2, offset),
                        __shfl_down_sync(0xffffffffu, wd.count, offset)};
        wd = vision_combine(wd, b);
    }
    for (int offset = 2; offset > 0; offset /= 2) {
        if (lane == 0 && warp >= offset && warp < 2*offset) {
            int y = warp - offset;
            buf[2*y] = wd.mean; buf[2*y+1] = wd.sigma2; buf[4+y] = wd.count;
        }
        __syncthreads();
        if (lane == 0 && warp < offset)
            wd = vision_combine(wd, {buf[2*warp], buf[2*warp+1], buf[4+warp]});
        __syncthreads();
    }
    if (tid == 0) { buf[0] = wd.mean; buf[1] = __fdiv_rn(wd.sigma2, (float)cols); }
    __syncthreads();
    float inv = rsqrtf(buf[1] + eps);
    for (int i = tid; i < cols / 4; i += 128) {
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            int c = 4*i+j;
            float n = inv * (__half2float(x[base+c]) - buf[0]);
            float w = weight ? __half2float(weight[c]) : 1.f;
            float b = bias ? __half2float(bias[c]) : 0.f;
            out[base+c] = __float2half_rn(w * n + b);
        }
    }
}

__global__ void vision_layernorm_f16_kernel(
    const __half* __restrict__ x,
    const __half* __restrict__ weight,
    const __half* __restrict__ bias,
    __half* __restrict__ out,
    int rows,
    int cols,
    float eps) {
    const int row = blockIdx.x;
    if (row >= rows) return;
    const __half* x_row = x + (size_t)row * cols;
    __half* out_row = out + (size_t)row * cols;

    __shared__ float s_warp[32];
    float sum = 0.0f;
    for (int c = threadIdx.x; c < cols; c += blockDim.x) {
        sum += __half2float(x_row[c]);
    }
    const float mean = block_reduce_sum(sum, s_warp) / (float)cols;

    float sq_sum = 0.0f;
    for (int c = threadIdx.x; c < cols; c += blockDim.x) {
        float d = __half2float(x_row[c]) - mean;
        sq_sum += d * d;
    }
    const float var = block_reduce_sum(sq_sum, s_warp) / (float)cols;
    const float inv_std = rsqrtf(var + eps);

    for (int c = threadIdx.x; c < cols; c += blockDim.x) {
        float n = (__half2float(x_row[c]) - mean) * inv_std;
        float w = (weight != nullptr) ? __half2float(weight[c]) : 1.0f;
        float b = (bias != nullptr) ? __half2float(bias[c]) : 0.0f;
        out_row[c] = __float2half(n * w + b);
    }
}

__global__ void vision_bias_gelu_f16_kernel(
    const __half* __restrict__ x,
    const __half* __restrict__ bias,
    __half* __restrict__ out,
    int rows,
    int cols,
    int use_tanh) {
    const int row = blockIdx.y;
    const int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= rows || col >= cols) return;
    const size_t idx = (size_t)row * cols + col;
    float v = __half2float(x[idx]);
    if (bias != nullptr) v += __half2float(bias[col]);
    // GELU consumes the rounded FP16 Linear output, not an unrounded bias sum.
    v = __half2float(__float2half_rn(v));
    float g;
    if (use_tanh != 0) {
        // gelu_pytorch_tanh: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
        const float cube = v * v * v;
        const float inner = 0.7978845608028654f * (v + 0.044715f * cube);
        // Native PyTorch uses accurate tanhf, not fast-math's MUFU approximation.
        g = 0.5f * v * (1.0f + __nv_tanhf(inner));
    } else {
        // Exact erf GELU: 0.5 * x * (1 + erf(x / sqrt(2)))
        g = 0.5f * v * (1.0f + erff(v * 0.7071067811865475f));
    }
    out[idx] = __float2half(g);
}

// Grid: (total_patches, num_heads), Block: 128 threads.
// Adds qkv_bias to Q, K, V and applies 2D axial RoPE in-place to Q and K.
// qkv layout per patch: [Q (num_heads * head_dim), K (num_heads * head_dim), V (num_heads * head_dim)].
__global__ void vision_qkv_bias_rope_inplace_f16_kernel(
    __half* __restrict__ qkv,
    const __half* __restrict__ qkv_bias,
    const int32_t* __restrict__ pos_hw,
    int total_patches,
    int num_heads,
    int head_dim,
    float rope_theta) {
    const int patch = blockIdx.x;
    const int head = blockIdx.y;
    if (patch >= total_patches || head >= num_heads) return;

    const int hidden = num_heads * head_dim;
    const int stride = 3 * hidden;
    __half* row = qkv + (size_t)patch * stride;

    const int h_pos = pos_hw[2 * patch];
    const int w_pos = pos_hw[2 * patch + 1];
    const int half_d = head_dim / 2;
    const int quarter_d = head_dim / 4;

    for (int i = threadIdx.x; i < half_d; i += blockDim.x) {
        const float pos_val = (i < quarter_d) ? (float)h_pos : (float)w_pos;
        const int freq_idx = (i < quarter_d) ? i : (i - quarter_d);
        const float inv_freq = powf(rope_theta, -((float)(2 * freq_idx) / (float)half_d));
        float sin_a = __nv_sinf(pos_val * inv_freq);
        float cos_a = __nv_cosf(pos_val * inv_freq);
        // vLLM RotaryEmbedding caches these coefficients in the model dtype.
        sin_a = __half2float(__float2half_rn(sin_a));
        cos_a = __half2float(__float2half_rn(cos_a));

        // Rotate Q and K (p = 0 for Q, p = 1 for K)
#pragma unroll
        for (int p = 0; p < 2; ++p) {
            const int base = p * hidden + head * head_dim;
            const float b1 = (qkv_bias != nullptr) ? __half2float(qkv_bias[base + i]) : 0.0f;
            const float b2 =
                (qkv_bias != nullptr) ? __half2float(qkv_bias[base + i + half_d]) : 0.0f;
            const float x1 = __half2float(__float2half_rn(__half2float(row[base + i]) + b1));
            const float x2 = __half2float(__float2half_rn(__half2float(row[base + i + half_d]) + b2));
            row[base + i] = __float2half(x1 * cos_a - x2 * sin_a);
            row[base + i + half_d] = __float2half(x2 * cos_a + x1 * sin_a);
        }
    }

    // Add bias to V (p = 2)
    const int v_base = 2 * hidden + head * head_dim;
    for (int c = threadIdx.x; c < head_dim; c += blockDim.x) {
        const float bv = (qkv_bias != nullptr) ? __half2float(qkv_bias[v_base + c]) : 0.0f;
        row[v_base + c] = __float2half(__half2float(row[v_base + c]) + bv);
    }
}

// Grid: (total_patches, num_heads), Block: 128 threads (4 warps).
// Bidirectional (non-causal) multi-head attention within each image segment [cu_seqlens[b], cu_seqlens[b+1]).
__global__ void vision_mha_f16_kernel(
    const __half* __restrict__ qkv,
    const int32_t* __restrict__ cu_seqlens,
    __half* __restrict__ out,
    int total_patches,
    int num_images,
    int num_heads,
    int head_dim) {
    const int patch = blockIdx.x;
    const int head = blockIdx.y;
    if (patch >= total_patches || head >= num_heads) return;

    int seq_start = 0;
    int seq_end = total_patches;
    for (int b = 0; b < num_images; ++b) {
        const int s0 = cu_seqlens[b];
        const int s1 = cu_seqlens[b + 1];
        if (patch >= s0 && patch < s1) {
            seq_start = s0;
            seq_end = s1;
            break;
        }
    }

    const int hidden = num_heads * head_dim;
    const int stride = 3 * hidden;
    const float sm_scale = rsqrtf((float)head_dim);

    const int lane = threadIdx.x & 31;
    const int wid = threadIdx.x >> 5; // 0..3

    const __half* q_ptr = qkv + (size_t)patch * stride + head * head_dim;
    float q_reg[4];
    float acc[4];
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        const int col = lane + 32 * j;
        q_reg[j] = (col < head_dim) ? __half2float(q_ptr[col]) : 0.0f;
        acc[j] = 0.0f;
    }

    float m_i = -FLT_MAX;
    float l_i = 0.0f;

    for (int kv_pos = seq_start + wid; kv_pos < seq_end; kv_pos += 4) {
        const __half* kv_row = qkv + (size_t)kv_pos * stride;
        const __half* k_ptr = kv_row + hidden + head * head_dim;
        const __half* v_ptr = kv_row + 2 * hidden + head * head_dim;

        float dot = 0.0f;
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            const int col = lane + 32 * j;
            if (col < head_dim) {
                dot += q_reg[j] * __half2float(k_ptr[col]);
            }
        }
        dot = warp_reduce_sum(dot);
        const float score = dot * sm_scale;

        const float m_new = fmaxf(m_i, score);
        const float alpha = expf(m_i - m_new);
        const float p = expf(score - m_new);
        l_i = l_i * alpha + p;
        m_i = m_new;

#pragma unroll
        for (int j = 0; j < 4; ++j) {
            const int col = lane + 32 * j;
            if (col < head_dim) {
                acc[j] = acc[j] * alpha + p * __half2float(v_ptr[col]);
            }
        }
    }

    __shared__ float s_m[4];
    __shared__ float s_l[4];
    __shared__ float s_acc[4][128];

    if (lane == 0) {
        s_m[wid] = m_i;
        s_l[wid] = l_i;
    }
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        const int col = lane + 32 * j;
        if (col < head_dim) {
            s_acc[wid][col] = acc[j];
        }
    }
    __syncthreads();

    float m_tot = -FLT_MAX;
#pragma unroll
    for (int w = 0; w < 4; ++w) {
        if (s_l[w] > 0.0f) m_tot = fmaxf(m_tot, s_m[w]);
    }
    float l_tot = 0.0f;
    float w_scale[4];
#pragma unroll
    for (int w = 0; w < 4; ++w) {
        if (s_l[w] > 0.0f) {
            float s = expf(s_m[w] - m_tot);
            w_scale[w] = s;
            l_tot += s_l[w] * s;
        } else {
            w_scale[w] = 0.0f;
        }
    }
    const float inv_l = (l_tot > 0.0f) ? (1.0f / l_tot) : 0.0f;
    __half* out_ptr = out + (size_t)patch * hidden + head * head_dim;
    for (int col = threadIdx.x; col < head_dim; col += blockDim.x) {
        float sum_v = s_acc[0][col] * w_scale[0]
                    + s_acc[1][col] * w_scale[1]
                    + s_acc[2][col] * w_scale[2]
                    + s_acc[3][col] * w_scale[3];
        out_ptr[col] = __float2half(sum_v * inv_l);
    }
}

__global__ void embedding_splice_f16_kernel(
    __half* __restrict__ dst_embeds,
    const __half* __restrict__ src_visual_embeds,
    const int32_t* __restrict__ token_indices,
    int num_visual_tokens,
    int hidden_size) {
    const int v_idx = blockIdx.x;
    if (v_idx >= num_visual_tokens) return;
    const int tok = token_indices[v_idx];
    if (tok < 0) return;
    __half* dst = dst_embeds + (size_t)tok * hidden_size;
    const __half* src = src_visual_embeds + (size_t)v_idx * hidden_size;
    for (int c = threadIdx.x; c < hidden_size; c += blockDim.x) {
        dst[c] = src[c];
    }
}

} // namespace w34_vision

extern "C" int w34_vision_bias_add_f16(
    const w34_exec* ex,
    const void* x,
    const void* bias,
    const void* residual,
    void* out,
    int rows,
    int cols) {
    if (ex == nullptr || x == nullptr || out == nullptr || rows < 0 || cols <= 0) {
        return (int)cudaErrorInvalidValue;
    }
    if (rows == 0) return 0;
    dim3 block(256);
    dim3 grid((unsigned)((cols + 255) / 256), (unsigned)rows);
    w34_vision::vision_bias_add_f16_kernel<<<grid, block, 0, (cudaStream_t)ex->stream>>>(
        (const __half*)x, (const __half*)bias, (const __half*)residual, (__half*)out, rows, cols);
    return (int)cudaGetLastError();
}

extern "C" int w34_vision_layernorm_f16(
    const w34_exec* ex,
    const void* x,
    const void* weight,
    const void* bias,
    void* out,
    int rows,
    int cols,
    float eps) {
    if (ex == nullptr || x == nullptr || out == nullptr || rows < 0 || cols <= 0) {
        return (int)cudaErrorInvalidValue;
    }
    if (rows == 0) return 0;
    if (cols % 4 == 0 && cols <= (1 << 24) &&
        (((uintptr_t)x | (uintptr_t)weight | (uintptr_t)bias | (uintptr_t)out) & 7) == 0) {
        w34_vision::vision_layernorm_vectorized_f16_kernel<<<(unsigned)rows, dim3(32, 4), 0, (cudaStream_t)ex->stream>>>(
            (const __half*)x, (const __half*)weight, (const __half*)bias, (__half*)out, cols, eps);
        return (int)cudaGetLastError();
    }
    w34_vision::vision_layernorm_f16_kernel<<<(unsigned)rows, 256, 0, (cudaStream_t)ex->stream>>>(
        (const __half*)x, (const __half*)weight, (const __half*)bias, (__half*)out, rows, cols, eps);
    return (int)cudaGetLastError();
}

extern "C" int w34_vision_bias_gelu_f16(
    const w34_exec* ex,
    const void* x,
    const void* bias,
    void* out,
    int rows,
    int cols,
    int use_tanh) {
    if (ex == nullptr || x == nullptr || out == nullptr || rows < 0 || cols <= 0) {
        return (int)cudaErrorInvalidValue;
    }
    if (rows == 0) return 0;
    dim3 block(256);
    dim3 grid((unsigned)((cols + 255) / 256), (unsigned)rows);
    w34_vision::vision_bias_gelu_f16_kernel<<<grid, block, 0, (cudaStream_t)ex->stream>>>(
        (const __half*)x, (const __half*)bias, (__half*)out, rows, cols, use_tanh);
    return (int)cudaGetLastError();
}

extern "C" int w34_vision_fa2_validate(const w34_exec*, const void*, const void*, int, int, int, int);
extern "C" int w34_vision_fa2(const w34_exec*, const void*, const int32_t*, void*, int, int, int, int);

extern "C" int w34_vision_qkv_rope_attn_f16(
    const w34_exec* ex,
    void* qkv,
    const void* qkv_bias,
    const int32_t* pos_hw,
    const int32_t* cu_seqlens,
    void* out,
    int total_patches,
    int num_images,
    int num_heads,
    int head_dim,
    float rope_theta) {
    if (ex == nullptr || qkv == nullptr || pos_hw == nullptr || cu_seqlens == nullptr ||
        out == nullptr || total_patches < 0 || num_images <= 0 || num_heads <= 0 ||
        head_dim <= 0 || head_dim > 128) {
        return (int)cudaErrorInvalidValue;
    }
    if (total_patches == 0) return 0;
    if (head_dim == 64 || head_dim == 72) {
        int rc = w34_vision_fa2_validate(ex, qkv, out, total_patches, num_images, num_heads, head_dim);
        if (rc) return rc;
    }
    dim3 grid((unsigned)total_patches, (unsigned)num_heads);
    w34_vision::vision_qkv_bias_rope_inplace_f16_kernel<<<grid, 128, 0, (cudaStream_t)ex->stream>>>(
        (__half*)qkv, (const __half*)qkv_bias, pos_hw, total_patches, num_heads, head_dim, rope_theta);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) return (int)err;
    if (head_dim == 64 || head_dim == 72)
        return w34_vision_fa2(ex, qkv, cu_seqlens, out, total_patches, num_images, num_heads, head_dim);
    w34_vision::vision_mha_f16_kernel<<<grid, 128, 0, (cudaStream_t)ex->stream>>>(
        (const __half*)qkv, cu_seqlens, (__half*)out, total_patches, num_images, num_heads, head_dim);
    return (int)cudaGetLastError();
}

extern "C" int w34_embedding_splice_f16(
    const w34_exec* ex,
    void* dst_embeds,
    const void* src_visual_embeds,
    const int32_t* token_indices,
    int num_visual_tokens,
    int hidden_size) {
    if (ex == nullptr || dst_embeds == nullptr || src_visual_embeds == nullptr ||
        token_indices == nullptr || num_visual_tokens < 0 || hidden_size <= 0) {
        return (int)cudaErrorInvalidValue;
    }
    if (num_visual_tokens == 0) return 0;
    w34_vision::embedding_splice_f16_kernel<<<(unsigned)num_visual_tokens, 256, 0, (cudaStream_t)ex->stream>>>(
        (__half*)dst_embeds, (const __half*)src_visual_embeds, token_indices, num_visual_tokens, hidden_size);
    return (int)cudaGetLastError();
}
