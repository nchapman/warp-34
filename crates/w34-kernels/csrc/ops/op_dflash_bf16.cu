// Native BF16 DFlash2 activation operators. Reductions stay FP32; persistent
// activation storage never passes through FP16. Semantics: SGLang v0.5.19
// models/dflash.py and speculative/dflash_utils.py::table_qk_norm_rope_.
#include "../w34_cuda.h"
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <math.h>

namespace {
__device__ float sum256(float v, float* scratch) {
    const int t = threadIdx.x;
    scratch[t] = v;
    __syncthreads();
    for (int s = 128; s; s >>= 1) {
        if (t < s) scratch[t] += scratch[t + s];
        __syncthreads();
    }
    return scratch[0];
}

__global__ void norm_bf16(const __nv_bfloat16* x, __nv_bfloat16* residual,
    const __nv_bfloat16* weight, __nv_bfloat16* y, int hidden, float eps) {
    const int64_t base = (int64_t)blockIdx.x * hidden;
    __shared__ float scratch[32];
    const int tid = threadIdx.x;
    const int num_threads = blockDim.x;
    const int num_warps = num_threads >> 5;
    const int vec = (hidden % 8 == 0) ? 8 : 1;
    const int rounds = (hidden + vec * num_threads - 1) / (vec * num_threads);
    float sq = 0.0f;
    for (int i = 0; i < rounds; ++i) {
        const int col0 = (i * num_threads + tid) * vec;
        for (int j = 0; j < vec; ++j) {
            const int c = col0 + j;
            if (c < hidden) {
                float v = (float)x[base + c];
                if (residual) v += (float)residual[base + c];
                sq += v * v;
            }
        }
    }
    for (int offset = 1; offset < 32; offset <<= 1) {
        sq += __shfl_xor_sync(0xffffffffu, sq, offset);
    }
    if ((tid & 31) == 0) scratch[tid >> 5] = sq;
    __syncthreads();
    if (tid < 32) {
        sq = (tid < num_warps) ? scratch[tid] : 0.0f;
        for (int offset = 1; offset < 32; offset <<= 1) {
            sq += __shfl_xor_sync(0xffffffffu, sq, offset);
        }
        if (tid == 0) scratch[0] = sq;
    }
    __syncthreads();
    float inv;
    const float var_eps = scratch[0] / (float)hidden + eps;
    asm volatile("rsqrt.approx.ftz.f32 %0, %1;" : "=f"(inv) : "f"(var_eps));
    // All input reads for the reduction complete before any in-place writes.
    for (int c = tid; c < hidden; c += num_threads) {
        float v = (float)x[base + c];
        if (residual) {
            v += (float)residual[base + c];
            residual[base + c] = __float2bfloat16(v);
        }
        y[base + c] = __float2bfloat16(v * inv * (float)weight[c]);
    }
}

__global__ void silu_mul_bf16(const __nv_bfloat16* gu, __nv_bfloat16* y,
    int64_t n, int hidden) {
    const int64_t i = (int64_t)blockIdx.x * 256 + threadIdx.x;
    if (i >= n) return;
    const int64_t row = i / hidden, col = i % hidden;
    const float g = (float)gu[row * 2 * hidden + col];
    const float u = (float)gu[row * 2 * hidden + hidden + col];
    y[i] = __float2bfloat16((g / (1.0f + expf(-g))) * u);
}

__global__ void grouped_conv_bf16(const __nv_bfloat16* x, const __nv_bfloat16* delta,
    const __nv_bfloat16* base, __nv_bfloat16* y, int rows, int hidden,
    int block, int group, int taps, int side) {
    const int64_t i = (int64_t)blockIdx.x * 256 + threadIdx.x;
    if (i >= (int64_t)rows * hidden) return;
    const int row = i / hidden, col = i % hidden, groups = hidden / group;
    float acc = 0.0f;
    for (int t = 0; t < taps && t <= row % block; ++t) {
        const float b = (float)base[((int64_t)side * taps + t) * hidden + col];
        const float d = (float)delta[((int64_t)row * 2 * taps + side * taps + t) * groups + col / group];
        const float coeff = __fadd_rn(b, d);
        const float xv = (float)x[(int64_t)(row - t) * hidden + col];
        acc = __fadd_rn(acc, __fmul_rn(coeff, xv));
    }
    y[i] = __float2bfloat16(acc);
}

template <typename TableT>
__global__ void qk_norm_rope_bf16(__nv_bfloat16* qkv, const int32_t* positions,
    const __nv_bfloat16* qw, const __nv_bfloat16* kw, const TableT* table,
    int q_heads, int kv_heads, int dim, int table_rows, float eps) {
    const int row = blockIdx.x, head = blockIdx.y, half = dim / 2;
    const int pos = positions[row];
    // Positions are request metadata, validated by the model. Guard memory even
    // for an invalid row; the model must never treat an unwritten row as valid.
    if (pos < 0 || pos >= table_rows) return;
    const int64_t offset = (int64_t)row * (q_heads + 2 * kv_heads) * dim + head * dim;
    const __nv_bfloat16* w = head < q_heads ? qw : kw;
    __shared__ float scratch[256];
    float sq = 0.0f;
    for (int c = threadIdx.x; c < dim; c += 256) {
        const float x = (float)qkv[offset + c];
        sq += x * x;
    }
    const float inv = rsqrtf(sum256(sq, scratch) / dim + eps);
    for (int c = threadIdx.x; c < half; c += 256) {
        // The donor explicitly rounds normalized Q/K before rotary arithmetic.
        const float a = (float)__float2bfloat16((float)qkv[offset + c] * inv * (float)w[c]);
        const float b = (float)__float2bfloat16((float)qkv[offset + half + c] * inv * (float)w[half + c]);
        const float cosine = (float)table[(int64_t)pos * dim + c];
        const float sine = (float)table[(int64_t)pos * dim + half + c];
        qkv[offset + c] = __float2bfloat16(a * cosine - b * sine);
        qkv[offset + half + c] = __float2bfloat16(b * cosine + a * sine);
    }
}

__global__ void copy_cols_bf16(const __nv_bfloat16* src, __nv_bfloat16* dst,
    int rows, int src_stride, int dst_stride, int src_col, int dst_col, int cols) {
    const int64_t i = (int64_t)blockIdx.x * 256 + threadIdx.x;
    if (i < (int64_t)rows * cols) {
        const int64_t row = i / cols, col = i % cols;
        dst[row * dst_stride + dst_col + col] = src[row * src_stride + src_col + col];
    }
}
__global__ void embed_bf16(const int32_t* ids, const __half* weight, __nv_bfloat16* out,
    int rows, int hidden, int vocab) {
    const int64_t i = (int64_t)blockIdx.x * 256 + threadIdx.x;
    if (i >= (int64_t)rows * hidden) return;
    const int token = ids[i / hidden];
    out[i] = __float2bfloat16(token >= 0 && token < vocab ? __half2float(weight[(int64_t)token * hidden + i % hidden]) : 0.0f);
}
__global__ void bf16_to_f16(const __nv_bfloat16* x, __half* y, int64_t n) {
    const int64_t i = (int64_t)blockIdx.x * 256 + threadIdx.x;
    if (i < n) y[i] = __float2half((float)x[i]);
}
__global__ void edges_bf16(const int32_t* ids, const float* unary, const __nv_bfloat16* hidden,
    const __nv_bfloat16* pred, const __nv_bfloat16* succ, float* edges,
    int anchor, int k, int topk, int rank, int vocab) {
    const int i = blockIdx.x, slot = i / (topk * topk);
    const int p = (i / topk) % topk, c = i % topk;
    const int pred_id = slot == 0 ? anchor : ids[(slot - 1) * topk + p];
    const int succ_id = ids[slot * topk + c];
    if (pred_id < 0 || pred_id >= vocab || succ_id < 0 || succ_id >= vocab) {
        if (threadIdx.x == 0) edges[i] = -INFINITY;
        return;
    }
    __shared__ float scratch[256];
    float sum = 0.0f;
    for (int r = threadIdx.x; r < rank; r += 256) {
        // SGLang: einsum(predecessor * projected_hidden, successor), where
        // the product is a BF16 operand of the FP32-accumulating contraction.
        const float ph = (float)__float2bfloat16((float)pred[(int64_t)pred_id * rank + r] *
            (float)hidden[(int64_t)slot * rank + r]);
        sum += ph * (float)succ[(int64_t)succ_id * rank + r];
    }
    sum = sum256(sum, scratch);
    if (threadIdx.x == 0) edges[i] = (float)__float2bfloat16(sum) + unary[slot * topk + c];
}
} // namespace

extern "C" int w34_embed_lookup_f16_to_bf16(const w34_exec* ex, const int32_t* ids,
    const void* weight, void* out, int rows, int hidden, int vocab) {
    if (!ex || !ids || !weight || !out || rows < 0 || hidden <= 0 || vocab <= 0) return (int)cudaErrorInvalidValue;
    const int64_t n = (int64_t)rows * hidden;
    if (!n) return 0;
    embed_bf16<<<(n+255)/256,256,0,(cudaStream_t)ex->stream>>>(ids,(const __half*)weight,(__nv_bfloat16*)out,rows,hidden,vocab);
    return (int)cudaGetLastError();
}
extern "C" int w34_cast_bf16_to_f16(const w34_exec* ex, const void* x, void* y, int64_t n) {
    if (!ex || !x || !y || n < 0) return (int)cudaErrorInvalidValue;
    if (!n) return 0;
    bf16_to_f16<<<(n+255)/256,256,0,(cudaStream_t)ex->stream>>>((const __nv_bfloat16*)x,(__half*)y,n);
    return (int)cudaGetLastError();
}
extern "C" int w34_dflash_edges_bf16(const w34_exec* ex, const int32_t* ids, const float* unary,
    const void* hidden, const void* pred, const void* succ, float* out,
    int anchor, int k, int topk, int rank, int vocab) {
    if (!ex || !ids || !unary || !hidden || !pred || !succ || !out || anchor < 0 || anchor >= vocab ||
        k <= 0 || k > 16 || topk <= 0 || topk > 32 || rank <= 0 || vocab <= 0) return (int)cudaErrorInvalidValue;
    edges_bf16<<<k*topk*topk,256,0,(cudaStream_t)ex->stream>>>(ids,unary,(const __nv_bfloat16*)hidden,
        (const __nv_bfloat16*)pred,(const __nv_bfloat16*)succ,out,anchor,k,topk,rank,vocab);
    return (int)cudaGetLastError();

}

extern "C" int w34_rms_norm_bf16(const w34_exec* ex, const void* x, void* residual,
    const void* weight, void* y, int rows, int hidden, float eps) {
    if (!ex || !x || !weight || !y || rows < 0 || hidden <= 0 || !(eps > 0.0f) || !isfinite(eps) ||
        (residual && (residual == x || residual == y))) return (int)cudaErrorInvalidValue;
    if (!rows) return 0;
    const int threads = (hidden <= 3072) ? 32 : ((hidden <= 6144) ? 64 : 128);
    norm_bf16<<<rows, threads, 0, (cudaStream_t)ex->stream>>>((const __nv_bfloat16*)x,
        (__nv_bfloat16*)residual, (const __nv_bfloat16*)weight, (__nv_bfloat16*)y, hidden, eps);
    return (int)cudaGetLastError();
}

extern "C" int w34_silu_and_mul_bf16(const w34_exec* ex, const void* gu, void* y, int rows, int hidden) {
    if (!ex || !gu || !y || rows < 0 || hidden <= 0 || gu == y) return (int)cudaErrorInvalidValue;
    const int64_t n = (int64_t)rows * hidden;
    if (!n) return 0;
    silu_mul_bf16<<<(n + 255) / 256, 256, 0, (cudaStream_t)ex->stream>>>(
        (const __nv_bfloat16*)gu, (__nv_bfloat16*)y, n, hidden);
    return (int)cudaGetLastError();
}

extern "C" int w34_dflash_conv_bf16(const w34_exec* ex, const void* x, const void* delta,
    const void* base, void* y, int rows, int hidden, int block, int group, int taps, int side) {
    if (!ex || !x || !delta || !base || !y || x == y || rows < 0 || hidden <= 0 ||
        block <= 0 || group <= 0 || hidden % group || taps <= 0 || (side != 0 && side != 1))
        return (int)cudaErrorInvalidValue;
    const int64_t n = (int64_t)rows * hidden;
    if (!n) return 0;
    grouped_conv_bf16<<<(n + 255) / 256, 256, 0, (cudaStream_t)ex->stream>>>(
        (const __nv_bfloat16*)x, (const __nv_bfloat16*)delta, (const __nv_bfloat16*)base,
        (__nv_bfloat16*)y, rows, hidden, block, group, taps, side);
    return (int)cudaGetLastError();
}

extern "C" int w34_dflash_qk_norm_rope_bf16(const w34_exec* ex, void* qkv, const int32_t* positions,
    const void* qw, const void* kw, const void* table, int rows, int q_heads, int kv_heads,
    int dim, int table_rows, float eps) {
    if (!ex || !qkv || !positions || !qw || !kw || !table || rows < 0 || q_heads < 0 || kv_heads <= 0 ||
        q_heads % kv_heads || dim <= 0 || dim > 256 || dim % 2 || table_rows <= 0 ||
        !(eps > 0.0f) || !isfinite(eps)) return (int)cudaErrorInvalidValue;
    if (!rows) return 0;
    qk_norm_rope_bf16<<<dim3(rows, q_heads + kv_heads), 256, 0, (cudaStream_t)ex->stream>>>(
        (__nv_bfloat16*)qkv, positions, (const __nv_bfloat16*)qw, (const __nv_bfloat16*)kw,
        (const __nv_bfloat16*)table, q_heads, kv_heads, dim, table_rows, eps);
    return (int)cudaGetLastError();
}

// Actual SGLang CUDA model retains its rotary cache in FP32.
extern "C" int w34_dflash_qk_norm_rope_bf16_f32_table(const w34_exec* ex, void* qkv, const int32_t* positions,
    const void* qw, const void* kw, const void* table, int rows, int q_heads, int kv_heads,
    int dim, int table_rows, float eps) {
    if (!ex || !qkv || !positions || !qw || !kw || !table || rows < 0 || q_heads < 0 || kv_heads <= 0 ||
        q_heads % kv_heads || dim <= 0 || dim > 256 || dim % 2 || table_rows <= 0 ||
        !(eps > 0.0f) || !isfinite(eps)) return (int)cudaErrorInvalidValue;
    if (!rows) return 0;
    qk_norm_rope_bf16<<<dim3(rows, q_heads + kv_heads), 256, 0, (cudaStream_t)ex->stream>>>(
        (__nv_bfloat16*)qkv, positions, (const __nv_bfloat16*)qw, (const __nv_bfloat16*)kw,
        (const float*)table, q_heads, kv_heads, dim, table_rows, eps);
    return (int)cudaGetLastError();
}

extern "C" int w34_copy_cols_bf16(const w34_exec* ex, const void* src, void* dst,
    int rows, int src_stride, int dst_stride, int src_col, int dst_col, int cols) {
    if (!ex || !src || !dst || src == dst || rows < 0 || cols <= 0 || src_col < 0 || dst_col < 0 ||
        src_stride < cols || dst_stride < cols || src_col > src_stride - cols || dst_col > dst_stride - cols)
        return (int)cudaErrorInvalidValue;
    const int64_t n = (int64_t)rows * cols;
    if (!n) return 0;
    copy_cols_bf16<<<(n + 255) / 256, 256, 0, (cudaStream_t)ex->stream>>>(
        (const __nv_bfloat16*)src, (__nv_bfloat16*)dst, rows, src_stride, dst_stride, src_col, dst_col, cols);
    return (int)cudaGetLastError();
}
