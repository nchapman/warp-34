// Cloudflare Clef JointSchemaHead GPU Kernels (FP16 Tensor-Core Spine + FP32 Reductions).
#include "../w34_cuda.h"

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <float.h>
#include <math.h>

namespace w34_clef {

__device__ __forceinline__ float warp_reduce_sum(float v) {
#pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
        v += __shfl_xor_sync(0xffffffffu, v, off);
    }
    return v;
}

__device__ __forceinline__ float block_reduce_sum_256(float v, float* s_warp) {
    const int lane = threadIdx.x & 31;
    const int wid = threadIdx.x >> 5;
    v = warp_reduce_sum(v);
    if (lane == 0) s_warp[wid] = v;
    __syncthreads();
    v = (threadIdx.x < 8) ? s_warp[lane] : 0.0f;
    if (wid == 0) {
        v = warp_reduce_sum(v);
        if (lane == 0) s_warp[0] = v;
    }
    __syncthreads();
    const float out = s_warp[0];
    __syncthreads();
    return out;
}

// Grid: (ceil_div(hidden_size, 256), num_spans), Block: 256.
// Accumulates mean over [span_start, span_end) from chunk [chunk_start, chunk_start + chunk_len).
__global__ void clef_accum_spans_f16_to_f32_kernel(
    const __half* __restrict__ chunk_normed,
    const int32_t* __restrict__ span_bounds,
    float* __restrict__ span_accum,
    int chunk_start,
    int chunk_len,
    int num_spans,
    int hidden_size) {
    const int span_idx = blockIdx.y;
    const int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (span_idx >= num_spans || col >= hidden_size) return;

    const int s = span_bounds[2 * span_idx];
    const int e = span_bounds[2 * span_idx + 1];
    if (e <= s) return;
    const int lo = (s > chunk_start) ? s : chunk_start;
    const int chunk_end = chunk_start + chunk_len;
    const int hi = (e < chunk_end) ? e : chunk_end;
    if (lo >= hi) return;

    const float inv_len = __frcp_rn((float)(e - s));
    float acc = 0.0f;
    for (int t = lo; t < hi; ++t) {
        acc += __half2float(chunk_normed[(size_t)(t - chunk_start) * hidden_size + col]);
    }
    span_accum[(size_t)span_idx * hidden_size + col] += acc * inv_len;
}

__global__ void clef_cast_f32_to_f16_kernel(
    const float* __restrict__ src,
    __half* __restrict__ dst,
    int64_t n) {
    const int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    dst[idx] = __float2half_rn(src[idx]);
}

// Grid: (ceil_div(width, 256), total_opts), Block: 256.
__global__ void clef_build_option_queries_f16_kernel(
    const __half* __restrict__ ctx_proj,
    const __half* __restrict__ lex_proj,
    const __half* __restrict__ opt_q_proj,
    const int32_t* __restrict__ opt_q_idx,
    __half* __restrict__ out,
    int total_opts,
    int num_q,
    int width) {
    const int oi = blockIdx.y;
    const int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (oi >= total_opts || col >= width) return;
    const int qi = opt_q_idx[oi];
    if (qi < 0 || qi >= num_q) return;
    const size_t off = (size_t)oi * width + col;
    const float v = __half2float(ctx_proj[off]) + __half2float(lex_proj[off]) +
                    __half2float(opt_q_proj[(size_t)qi * width + col]);
    out[off] = __float2half_rn(v);
}

// Multi-head cross/self-attention with online softmax across 4 warps (128 threads).
// Grid: (q_len, heads), Block: 128. Supports head_dim in [1, 256] and arbitrary m_len >= 1.
__global__ void clef_mha_attn_f16_kernel(
    const __half* __restrict__ q,
    const __half* __restrict__ q_bias,
    const __half* __restrict__ kv,
    const __half* __restrict__ kv_bias,
    __half* __restrict__ out,
    int q_len,
    int m_len,
    int heads,
    int head_dim) {
    const int qi = blockIdx.x;
    const int hi = blockIdx.y;
    if (qi >= q_len || hi >= heads) return;

    __shared__ float s_q[256];
    __shared__ float s_kb[256];
    __shared__ float s_vb[256];
    __shared__ float s_m[4];
    __shared__ float s_l[4];
    __shared__ float s_acc[4][256];

    const int tid = threadIdx.x;
    const int lane = tid & 31;
    const int wid = tid >> 5;
    const int width = heads * head_dim;
    const int h_off = hi * head_dim;

    for (int d = tid; d < head_dim; d += 128) {
        const float qb = (q_bias != nullptr) ? __half2float(q_bias[h_off + d]) : 0.0f;
        const float qv = __half2float(q[(size_t)qi * width + h_off + d]) + qb;
        s_q[d] = __half2float(__float2half_rn(qv));
        s_kb[d] = (kv_bias != nullptr) ? __half2float(kv_bias[h_off + d]) : 0.0f;
        s_vb[d] = (kv_bias != nullptr) ? __half2float(kv_bias[width + h_off + d]) : 0.0f;
    }
    __syncthreads();

    const float scale = rsqrtf((float)head_dim);
    float m_w = -FLT_MAX;
    float l_w = 0.0f;
    float acc[8] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};

    for (int mi = wid; mi < m_len; mi += 4) {
        const __half* kv_row = kv + (size_t)mi * (2 * width);
        const __half* k_head = kv_row + h_off;
        const __half* v_head = kv_row + width + h_off;

        float part = 0.0f;
#pragma unroll
        for (int r = 0; r < 8; ++r) {
            const int d = lane + 32 * r;
            if (d < head_dim) {
                const float kv_val = __half2float(__float2half_rn(__half2float(k_head[d]) + s_kb[d]));
                part += s_q[d] * kv_val;
            }
        }
        const float score = warp_reduce_sum(part) * scale;
        const float m_new = fmaxf(m_w, score);
        const float alpha = (l_w == 0.0f) ? 0.0f : expf(m_w - m_new);
        const float p = expf(score - m_new);
        l_w = l_w * alpha + p;
        m_w = m_new;

#pragma unroll
        for (int r = 0; r < 8; ++r) {
            const int d = lane + 32 * r;
            if (d < head_dim) {
                const float vv = __half2float(__float2half_rn(__half2float(v_head[d]) + s_vb[d]));
                acc[r] = acc[r] * alpha + p * vv;
            }
        }
    }

    if (lane == 0) {
        s_m[wid] = m_w;
        s_l[wid] = l_w;
    }
#pragma unroll
    for (int r = 0; r < 8; ++r) {
        const int d = lane + 32 * r;
        if (d < head_dim) {
            s_acc[wid][d] = acc[r];
        }
    }
    __syncthreads();

    if (wid == 0) {
        float m_all = -FLT_MAX;
#pragma unroll
        for (int w = 0; w < 4; ++w) {
            if (s_l[w] > 0.0f) m_all = fmaxf(m_all, s_m[w]);
        }
        float w_scale[4];
        float l_all = 0.0f;
#pragma unroll
        for (int w = 0; w < 4; ++w) {
            w_scale[w] = (s_l[w] > 0.0f) ? expf(s_m[w] - m_all) : 0.0f;
            l_all += s_l[w] * w_scale[w];
        }
        const float inv_l = (l_all > 0.0f) ? __frcp_rn(l_all) : 0.0f;
        __half* out_head = out + (size_t)qi * width + h_off;
#pragma unroll
        for (int r = 0; r < 8; ++r) {
            const int d = lane + 32 * r;
            if (d < head_dim) {
                const float sum = s_acc[0][d] * w_scale[0] + s_acc[1][d] * w_scale[1] +
                                  s_acc[2][d] * w_scale[2] + s_acc[3][d] * w_scale[3];
                out_head[d] = __float2half_rn(sum * inv_l);
            }
        }
    }
}

// Grid: num_q, Block: 256. Supports width in [1, 2048] and arbitrary n_opts >= 1.
__global__ void clef_option_summaries_f16_kernel(
    const __half* __restrict__ routed_opts,
    const __half* __restrict__ base_fields,
    const int32_t* __restrict__ q_opt_offsets,
    __half* __restrict__ out_summaries,
    int num_q,
    int total_opts,
    int width) {
    const int qi = blockIdx.x;
    if (qi >= num_q) return;

    const int opt_start = q_opt_offsets[qi];
    const int opt_end = q_opt_offsets[qi + 1];
    if (opt_start < 0 || opt_end <= opt_start || opt_end > total_opts) return;

    __shared__ float s_warp[8];
    const int tid = threadIdx.x;
    const __half* field_row = base_fields + (size_t)qi * width;
    const float inv_sqrt_w = rsqrtf((float)width);

    float m = -FLT_MAX;
    float l = 0.0f;
    float acc[8] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};

    for (int oi = opt_start; oi < opt_end; ++oi) {
        const __half* opt_row = routed_opts + (size_t)oi * width;
        float part = 0.0f;
#pragma unroll
        for (int r = 0; r < 8; ++r) {
            const int c = tid + 256 * r;
            if (c < width) {
                part += __half2float(opt_row[c]) * __half2float(field_row[c]);
            }
        }
        const float score = block_reduce_sum_256(part, s_warp) * inv_sqrt_w;
        const float m_new = fmaxf(m, score);
        const float alpha = (l == 0.0f) ? 0.0f : expf(m - m_new);
        const float p = expf(score - m_new);
        l = l * alpha + p;
        m = m_new;

#pragma unroll
        for (int r = 0; r < 8; ++r) {
            const int c = tid + 256 * r;
            if (c < width) {
                acc[r] = acc[r] * alpha + p * __half2float(opt_row[c]);
            }
        }
    }

    const float inv_l = (l > 0.0f) ? __frcp_rn(l) : 0.0f;
    __half* dst = out_summaries + (size_t)qi * width;
#pragma unroll
    for (int r = 0; r < 8; ++r) {
        const int c = tid + 256 * r;
        if (c < width) {
            dst[c] = __float2half_rn(acc[r] * inv_l);
        }
    }
}

// Grid: (ceil_div(width, 256), num_q), Block: 256.
__global__ void clef_build_fields_f16_kernel(
    const __half* __restrict__ base_fields,
    const __half* __restrict__ norm_summaries,
    const __half* __restrict__ global_proj,
    const __half* __restrict__ type_embed,
    const int32_t* __restrict__ q_types,
    __half* __restrict__ out_fields,
    int num_q,
    int width) {
    const int qi = blockIdx.y;
    const int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (qi >= num_q || col >= width) return;
    const int qt = q_types[qi];
    if (qt < 0 || qt >= 3) return;
    const size_t off = (size_t)qi * width + col;
    const float v = __half2float(base_fields[off]) + __half2float(norm_summaries[off]) +
                    __half2float(global_proj[col]) +
                    __half2float(type_embed[(size_t)qt * width + col]);
    out_fields[off] = __float2half_rn(v);
}

// Grid: (ceil_div(width, 256), total_opts), Block: 256.
__global__ void clef_build_residual_features_f16_kernel(
    const __half* __restrict__ fields,
    const __half* __restrict__ options_normed,
    const int32_t* __restrict__ opt_q_idx,
    __half* __restrict__ out_features,
    int total_opts,
    int num_q,
    int width) {
    const int oi = blockIdx.y;
    const int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (oi >= total_opts || col >= width) return;
    const int qi = opt_q_idx[oi];
    if (qi < 0 || qi >= num_q) return;

    const float f = __half2float(fields[(size_t)qi * width + col]);
    const float o = __half2float(options_normed[(size_t)oi * width + col]);
    __half* dst = out_features + (size_t)oi * (4 * width);
    dst[col] = __float2half_rn(f);
    dst[width + col] = __float2half_rn(o);
    dst[2 * width + col] = __float2half_rn(f * o);
    dst[3 * width + col] = __float2half_rn(fabsf(f - o));
}

}  // namespace w34_clef

extern "C" int w34_clef_accum_spans_f16_to_f32(
    const w34_exec* ex,
    const void* chunk_normed_f16,
    const int32_t* span_bounds,
    float* span_accum_f32,
    int chunk_start,
    int chunk_len,
    int num_spans,
    int hidden_size) {
    if (ex == nullptr || chunk_normed_f16 == nullptr || span_bounds == nullptr ||
        span_accum_f32 == nullptr || chunk_start < 0 || chunk_len < 0 ||
        num_spans < 0 || hidden_size <= 0) {
        return (int)cudaErrorInvalidValue;
    }
    if (chunk_len == 0 || num_spans == 0) return 0;
    dim3 block(256);
    dim3 grid((unsigned)((hidden_size + 255) / 256), (unsigned)num_spans);
    w34_clef::clef_accum_spans_f16_to_f32_kernel<<<grid, block, 0, (cudaStream_t)ex->stream>>>(
        (const __half*)chunk_normed_f16, span_bounds, span_accum_f32,
        chunk_start, chunk_len, num_spans, hidden_size);
    return (int)cudaGetLastError();
}

extern "C" int w34_clef_cast_f32_to_f16(
    const w34_exec* ex,
    const float* src_f32,
    void* dst_f16,
    int64_t num_elems) {
    if (ex == nullptr || src_f32 == nullptr || dst_f16 == nullptr || num_elems < 0) {
        return (int)cudaErrorInvalidValue;
    }
    if (num_elems == 0) return 0;
    const uint64_t blocks = (uint64_t)((num_elems + 255) / 256);
    if (blocks > 0xffffffffull) return (int)cudaErrorInvalidValue;
    w34_clef::clef_cast_f32_to_f16_kernel<<<(unsigned)blocks, 256, 0, (cudaStream_t)ex->stream>>>(
        src_f32, (__half*)dst_f16, num_elems);
    return (int)cudaGetLastError();
}

extern "C" int w34_clef_build_option_queries_f16(
    const w34_exec* ex,
    const void* ctx_proj_f16,
    const void* lex_proj_f16,
    const void* opt_q_proj_f16,
    const int32_t* opt_q_idx,
    void* out_f16,
    int total_opts,
    int num_q,
    int width) {
    if (ex == nullptr || ctx_proj_f16 == nullptr || lex_proj_f16 == nullptr ||
        opt_q_proj_f16 == nullptr || opt_q_idx == nullptr || out_f16 == nullptr ||
        total_opts <= 0 || num_q <= 0 || width <= 0) {
        return (int)cudaErrorInvalidValue;
    }
    dim3 block(256);
    dim3 grid((unsigned)((width + 255) / 256), (unsigned)total_opts);
    w34_clef::clef_build_option_queries_f16_kernel<<<grid, block, 0, (cudaStream_t)ex->stream>>>(
        (const __half*)ctx_proj_f16, (const __half*)lex_proj_f16,
        (const __half*)opt_q_proj_f16, opt_q_idx, (__half*)out_f16,
        total_opts, num_q, width);
    return (int)cudaGetLastError();
}

extern "C" int w34_clef_mha_attn_f16(
    const w34_exec* ex,
    const void* q_f16,
    const void* q_bias_f16,
    const void* kv_f16,
    const void* kv_bias_f16,
    void* out_f16,
    int q_len,
    int m_len,
    int heads,
    int head_dim) {
    if (ex == nullptr || q_f16 == nullptr || kv_f16 == nullptr || out_f16 == nullptr ||
        q_len <= 0 || m_len <= 0 || heads <= 0 || head_dim <= 0 || head_dim > 256) {
        return (int)cudaErrorInvalidValue;
    }
    dim3 grid((unsigned)q_len, (unsigned)heads);
    w34_clef::clef_mha_attn_f16_kernel<<<grid, 128, 0, (cudaStream_t)ex->stream>>>(
        (const __half*)q_f16, (const __half*)q_bias_f16,
        (const __half*)kv_f16, (const __half*)kv_bias_f16,
        (__half*)out_f16, q_len, m_len, heads, head_dim);
    return (int)cudaGetLastError();
}

extern "C" int w34_clef_option_summaries_f16(
    const w34_exec* ex,
    const void* routed_opts_f16,
    const void* base_fields_f16,
    const int32_t* q_opt_offsets,
    void* out_summaries_f16,
    int num_q,
    int total_opts,
    int width) {
    if (ex == nullptr || routed_opts_f16 == nullptr || base_fields_f16 == nullptr ||
        q_opt_offsets == nullptr || out_summaries_f16 == nullptr ||
        num_q <= 0 || total_opts <= 0 || width <= 0 || width > 2048) {
        return (int)cudaErrorInvalidValue;
    }
    w34_clef::clef_option_summaries_f16_kernel<<<(unsigned)num_q, 256, 0, (cudaStream_t)ex->stream>>>(
        (const __half*)routed_opts_f16, (const __half*)base_fields_f16,
        q_opt_offsets, (__half*)out_summaries_f16, num_q, total_opts, width);
    return (int)cudaGetLastError();
}

extern "C" int w34_clef_build_fields_f16(
    const w34_exec* ex,
    const void* base_fields_f16,
    const void* norm_summaries_f16,
    const void* global_proj_f16,
    const void* type_embed_f16,
    const int32_t* q_types,
    void* out_fields_f16,
    int num_q,
    int width) {
    if (ex == nullptr || base_fields_f16 == nullptr || norm_summaries_f16 == nullptr ||
        global_proj_f16 == nullptr || type_embed_f16 == nullptr || q_types == nullptr ||
        out_fields_f16 == nullptr || num_q <= 0 || width <= 0) {
        return (int)cudaErrorInvalidValue;
    }
    dim3 block(256);
    dim3 grid((unsigned)((width + 255) / 256), (unsigned)num_q);
    w34_clef::clef_build_fields_f16_kernel<<<grid, block, 0, (cudaStream_t)ex->stream>>>(
        (const __half*)base_fields_f16, (const __half*)norm_summaries_f16,
        (const __half*)global_proj_f16, (const __half*)type_embed_f16,
        q_types, (__half*)out_fields_f16, num_q, width);
    return (int)cudaGetLastError();
}

extern "C" int w34_clef_build_residual_features_f16(
    const w34_exec* ex,
    const void* fields_f16,
    const void* options_normed_f16,
    const int32_t* opt_q_idx,
    void* out_features_f16,
    int total_opts,
    int num_q,
    int width) {
    if (ex == nullptr || fields_f16 == nullptr || options_normed_f16 == nullptr ||
        opt_q_idx == nullptr || out_features_f16 == nullptr ||
        total_opts <= 0 || num_q <= 0 || width <= 0) {
        return (int)cudaErrorInvalidValue;
    }
    dim3 block(256);
    dim3 grid((unsigned)((width + 255) / 256), (unsigned)total_opts);
    w34_clef::clef_build_residual_features_f16_kernel<<<grid, block, 0, (cudaStream_t)ex->stream>>>(
        (const __half*)fields_f16, (const __half*)options_normed_f16,
        opt_q_idx, (__half*)out_features_f16, total_opts, num_q, width);
    return (int)cudaGetLastError();
}
