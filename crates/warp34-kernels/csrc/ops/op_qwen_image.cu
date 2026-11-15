// Clean-room CUDA kernels for Qwen-Image-2.1 (`QwenImage21Transformer2DModel`):
// 1. Sinusoidal timestep projection (`QwenImage21TemporalTimesteps`, FP16-rounded timestep * 1000).
// 2. Unary SiLU (`x / (1 + exp(-x))`) on FP16.
// 3. Joint text + image hidden sequence assembly (`joint_row_map` gather).
// 4. Parameterless LayerNorm + AdaLN `(1 + scale)` modulation (`QwenImage21TransformerBlock._modulate` / `norm_out`).
// 5. Gated `tanh(gate) * branch_out` residual add with FP16 clamp (`[-65504, 65504]`).
// 6. Fused per-head Q/K RMSNorm + 3-axis complex RoPE (`apply_rotary_emb_qwen(use_real=False)`) + optional prefix K cache write.
#include "../warp34_cuda.h"
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <math.h>
#include <stdint.h>

extern "C" __device__ float __nv_sinf(float);
extern "C" __device__ float __nv_cosf(float);
extern "C" __device__ float __nv_tanhf(float);
extern "C" __device__ float __nv_expf(float);
extern "C" __device__ float __nv_logf(float);

namespace warp34_qwen_image {

__device__ __forceinline__ float warp_reduce_sum(float v) {
    #pragma unroll
    for (int mask = 16; mask > 0; mask >>= 1) {
        v += __shfl_xor_sync(0xffffffffu, v, mask);
    }
    return v;
}

__global__ void timestep_sinusoidal_f16_kernel(
    const float* __restrict__ timesteps,
    __half* __restrict__ out,
    int num_rows,
    int timestep_dim,
    float max_period,
    float time_factor) {
    const int row = blockIdx.y;
    const int col = blockIdx.x * blockDim.x + threadIdx.x;
    const int half_dim = timestep_dim / 2;
    if (row >= num_rows || col >= half_dim) return;

    // Match `timestep = timestep.to(hidden_states.dtype)` before `time_factor * timestep.float()`.
    const float t_fp16 = __half2float(__float2half_rn(timesteps[row]));
    const float t_scaled = time_factor * t_fp16;
    const float freq = __nv_expf(-__nv_logf(max_period) * (float)col / (float)half_dim);
    const float arg = t_scaled * freq;

    __half* row_out = out + (size_t)row * timestep_dim;
    row_out[col] = __float2half_rn(__nv_cosf(arg));
    row_out[half_dim + col] = __float2half_rn(__nv_sinf(arg));
    if ((timestep_dim & 1) != 0 && col == 0) {
        row_out[timestep_dim - 1] = __float2half_rn(0.0f);
    }
}

__global__ void silu_f16_kernel(
    const __half* __restrict__ x,
    __half* __restrict__ out,
    int64_t n) {
    const int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    const float v = __half2float(x[idx]);
    const float s = v / (1.0f + __nv_expf(-v));
    out[idx] = __float2half_rn(s);
}

// Assembles `joint_hidden_states` (`[total_rows, hidden_size]`) from `txt_hidden` and `img_hidden`.
// For each row `r` in `0..total_rows`:
// - `row_map[r] >= 0`: copies row `row_map[r]` from `txt_hidden`
// - `row_map[r] <  0`: copies row `(-1 - row_map[r])` from `img_hidden`
__global__ void build_joint_hidden_f16_kernel(
    const __half* __restrict__ txt_hidden,
    const __half* __restrict__ img_hidden,
    const int32_t* __restrict__ row_map,
    __half* __restrict__ out_joint,
    int total_rows,
    int txt_rows,
    int img_rows,
    int hidden_size) {
    const int row = blockIdx.x;
    if (row >= total_rows) return;
    const int src_code = row_map[row];
    const __half* src_row = nullptr;
    if (src_code >= 0) {
        if (src_code >= txt_rows) return;
        src_row = txt_hidden + (size_t)src_code * hidden_size;
    } else {
        const int img_idx = -1 - src_code;
        if (img_idx < 0 || img_idx >= img_rows) return;
        src_row = img_hidden + (size_t)img_idx * hidden_size;
    }
    __half* dst_row = out_joint + (size_t)row * hidden_size;
    for (int c = threadIdx.x; c < hidden_size; c += blockDim.x) {
        dst_row[c] = src_row[c];
    }
}

// Parameterless LayerNorm (`elementwise_affine=False`) + AdaLN modulation `norm(x) * (1 + scale)`.
// `modulation` has stride `mod_stride` per modulation row and `scale` starts at `scale_col_offset`.
// When `target_token_mask != nullptr`, token `t` selects row `0` if `target_token_mask[t] != 0`, else row `1`.
// When `target_token_mask == nullptr`, all tokens select row `0`.
__global__ void adaln_modulate_f16_kernel(
    const __half* __restrict__ x,
    const __half* __restrict__ modulation,
    const uint8_t* __restrict__ target_token_mask,
    __half* __restrict__ out,
    int num_tokens,
    int hidden_size,
    int mod_stride,
    int scale_col_offset,
    float eps) {
    const int row = blockIdx.x;
    if (row >= num_tokens) return;
    const int tid = threadIdx.x;
    const int lane = tid & 31;
    const int warp_id = tid >> 5;
    const int num_warps = blockDim.x >> 5;

    __shared__ float s_sum[32];
    __shared__ float s_sq[32];

    const __half* x_row = x + (size_t)row * hidden_size;
    float local_sum = 0.0f;
    for (int i = tid; i < hidden_size; i += blockDim.x) {
        local_sum += __half2float(x_row[i]);
    }
    local_sum = warp_reduce_sum(local_sum);
    if (lane == 0) s_sum[warp_id] = local_sum;
    __syncthreads();

    if (warp_id == 0) {
        float v = (tid < num_warps) ? s_sum[tid] : 0.0f;
        v = warp_reduce_sum(v);
        if (tid == 0) s_sum[0] = v / (float)hidden_size;
    }
    __syncthreads();
    const float mean = s_sum[0];

    float local_sq = 0.0f;
    for (int i = tid; i < hidden_size; i += blockDim.x) {
        const float d = __half2float(x_row[i]) - mean;
        local_sq += d * d;
    }
    local_sq = warp_reduce_sum(local_sq);
    if (lane == 0) s_sq[warp_id] = local_sq;
    __syncthreads();

    if (warp_id == 0) {
        float v = (tid < num_warps) ? s_sq[tid] : 0.0f;
        v = warp_reduce_sum(v);
        if (tid == 0) s_sq[0] = rsqrtf(v / (float)hidden_size + eps);
    }
    __syncthreads();
    const float rstd = s_sq[0];

    const int mod_row = (target_token_mask != nullptr && target_token_mask[row] == 0) ? 1 : 0;
    const __half* scale_row = modulation + (size_t)mod_row * mod_stride + scale_col_offset;
    __half* out_row = out + (size_t)row * hidden_size;

    const __half one_h = __float2half_rn(1.0f);
    for (int i = tid; i < hidden_size; i += blockDim.x) {
        // Match PyTorch FP16: `self.img_norm(hidden_states)` rounds to FP16 before `* (1 + scale)`.
        const __half normed_h = __float2half_rn((__half2float(x_row[i]) - mean) * rstd);
        const __half scale_plus_1 = __hadd(one_h, scale_row[i]);
        out_row[i] = __hmul(normed_h, scale_plus_1);
    }
}

// Residual update: `residual[t, c] = clamp_fp16(residual[t, c] + tanh(gate[mod_row, c]) * branch_out[t, c])`.
__global__ void gated_residual_f16_kernel(
    __half* __restrict__ residual,
    const __half* __restrict__ branch_out,
    const __half* __restrict__ modulation,
    const uint8_t* __restrict__ target_token_mask,
    int num_tokens,
    int hidden_size,
    int mod_stride,
    int gate_col_offset) {
    const int row = blockIdx.y;
    const int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= num_tokens || col >= hidden_size) return;

    const int mod_row = (target_token_mask != nullptr && target_token_mask[row] == 0) ? 1 : 0;
    const __half gate_h = modulation[(size_t)mod_row * mod_stride + gate_col_offset + col];
    const __half gate_tanh = __float2half_rn(__nv_tanhf(__half2float(gate_h)));

    const size_t idx = (size_t)row * hidden_size + col;
    const __half prod = __hmul(gate_tanh, branch_out[idx]);
    float sum = __half2float(__hadd(residual[idx], prod));
    if (sum > 65504.0f) sum = 65504.0f;
    if (sum < -65504.0f) sum = -65504.0f;
    residual[idx] = __float2half_rn(sum);
}

// Fused per-head Q/K RMSNorm + complex 3-axis RoPE (`apply_rotary_emb_qwen(use_real=False)`).
// Grid: (num_tokens, 2 * num_heads), Block: 64 threads (one thread per complex pair p in 0..64, head_dim = 128).
// `cos_table` and `sin_table` have shape `[num_tokens, 64]` in FP32.
// If `cache_k_out != nullptr` and `token_idx < cache_write_len`, K heads also write rotated K into `cache_k_out`.
__global__ void qk_rmsnorm_complex_rope_d128_f16_kernel(
    const __half* __restrict__ q_in,
    const __half* __restrict__ k_in,
    const __half* __restrict__ norm_q_w,
    const __half* __restrict__ norm_k_w,
    const float* __restrict__ cos_table,
    const float* __restrict__ sin_table,
    __half* __restrict__ q_out,
    __half* __restrict__ k_out,
    __half* __restrict__ cache_k_out,
    int num_tokens,
    int num_heads,
    int cache_write_len,
    float eps) {
    const int tok = blockIdx.x;
    const int hk = blockIdx.y;
    const int pair_idx = threadIdx.x; // 0..63
    if (tok >= num_tokens || pair_idx >= 64) return;

    const bool is_k = (hk >= num_heads);
    const int head = is_k ? (hk - num_heads) : hk;
    const size_t head_offset = ((size_t)tok * num_heads + head) * 128;

    const __half* src = (is_k ? k_in : q_in) + head_offset;
    const __half* nw = is_k ? norm_k_w : norm_q_w;

    const float v0 = __half2float(src[2 * pair_idx]);
    const float v1 = __half2float(src[2 * pair_idx + 1]);
    float sq = v0 * v0 + v1 * v1;
    sq = warp_reduce_sum(sq);

    __shared__ float s_warp[2];
    const int lane = pair_idx & 31;
    const int warp_id = pair_idx >> 5;
    if (lane == 0) s_warp[warp_id] = sq;
    __syncthreads();

    const float total_sq = s_warp[0] + s_warp[1];
    const float rrms = rsqrtf(total_sq * (1.0f / 128.0f) + eps);

    // Match `attn.norm_q(query).to(value.dtype)` before complex RoPE in FP32.
    const float n0 = __half2float(__float2half_rn(v0 * rrms * __half2float(nw[2 * pair_idx])));
    const float n1 = __half2float(__float2half_rn(v1 * rrms * __half2float(nw[2 * pair_idx + 1])));

    const float c = cos_table[(size_t)tok * 64 + pair_idx];
    const float s = sin_table[(size_t)tok * 64 + pair_idx];

    const __half r0 = __float2half_rn(n0 * c - n1 * s);
    const __half r1 = __float2half_rn(n0 * s + n1 * c);

    __half* dst = (is_k ? k_out : q_out) + head_offset;
    dst[2 * pair_idx] = r0;
    dst[2 * pair_idx + 1] = r1;

    if (is_k && cache_k_out != nullptr && tok < cache_write_len) {
        __half* cdst = cache_k_out + head_offset;
        cdst[2 * pair_idx] = r0;
        cdst[2 * pair_idx + 1] = r1;
    }
}

} // namespace warp34_qwen_image

extern "C" int warp34_qwen_image_timestep_sinusoidal_f16(
    const warp34_exec* ex,
    const float* timesteps,
    void* out,
    int num_rows,
    int timestep_dim,
    float max_period,
    float time_factor) {
    if (ex == nullptr || timesteps == nullptr || out == nullptr ||
        num_rows < 0 || timestep_dim <= 0) {
        return (int)cudaErrorInvalidValue;
    }
    if (num_rows == 0) return 0;
    const int half_dim = timestep_dim / 2;
    dim3 block(128);
    dim3 grid((unsigned)((half_dim + 127) / 128), (unsigned)num_rows);
    warp34_qwen_image::timestep_sinusoidal_f16_kernel<<<grid, block, 0, (cudaStream_t)ex->stream>>>(
        timesteps, (__half*)out, num_rows, timestep_dim, max_period, time_factor);
    return (int)cudaGetLastError();
}

extern "C" int warp34_qwen_image_silu_f16(
    const warp34_exec* ex,
    const void* x,
    void* out,
    int64_t n) {
    if (ex == nullptr || x == nullptr || out == nullptr || n < 0) {
        return (int)cudaErrorInvalidValue;
    }
    if (n == 0) return 0;
    const int threads = 256;
    const unsigned blocks = (unsigned)((n + threads - 1) / threads);
    warp34_qwen_image::silu_f16_kernel<<<blocks, threads, 0, (cudaStream_t)ex->stream>>>(
        (const __half*)x, (__half*)out, n);
    return (int)cudaGetLastError();
}

extern "C" int warp34_qwen_image_build_joint_hidden_f16(
    const warp34_exec* ex,
    const void* txt_hidden,
    const void* img_hidden,
    const int32_t* row_map,
    void* out_joint,
    int total_rows,
    int txt_rows,
    int img_rows,
    int hidden_size) {
    if (ex == nullptr || txt_hidden == nullptr || img_hidden == nullptr ||
        row_map == nullptr || out_joint == nullptr ||
        total_rows < 0 || txt_rows < 0 || img_rows < 0 || hidden_size <= 0) {
        return (int)cudaErrorInvalidValue;
    }
    if (total_rows == 0) return 0;
    warp34_qwen_image::build_joint_hidden_f16_kernel<<<(unsigned)total_rows, 256, 0, (cudaStream_t)ex->stream>>>(
        (const __half*)txt_hidden,
        (const __half*)img_hidden,
        row_map,
        (__half*)out_joint,
        total_rows,
        txt_rows,
        img_rows,
        hidden_size);
    return (int)cudaGetLastError();
}

extern "C" int warp34_qwen_image_adaln_modulate_f16(
    const warp34_exec* ex,
    const void* x,
    const void* modulation,
    const uint8_t* target_token_mask,
    void* out,
    int num_tokens,
    int hidden_size,
    int mod_stride,
    int scale_col_offset,
    float eps) {
    if (ex == nullptr || x == nullptr || modulation == nullptr || out == nullptr ||
        num_tokens < 0 || hidden_size <= 0 || mod_stride < hidden_size ||
        scale_col_offset < 0 || scale_col_offset + hidden_size > mod_stride) {
        return (int)cudaErrorInvalidValue;
    }
    if (num_tokens == 0) return 0;
    warp34_qwen_image::adaln_modulate_f16_kernel<<<(unsigned)num_tokens, 256, 0, (cudaStream_t)ex->stream>>>(
        (const __half*)x,
        (const __half*)modulation,
        target_token_mask,
        (__half*)out,
        num_tokens,
        hidden_size,
        mod_stride,
        scale_col_offset,
        eps);
    return (int)cudaGetLastError();
}

extern "C" int warp34_qwen_image_gated_residual_f16(
    const warp34_exec* ex,
    void* residual,
    const void* branch_out,
    const void* modulation,
    const uint8_t* target_token_mask,
    int num_tokens,
    int hidden_size,
    int mod_stride,
    int gate_col_offset) {
    if (ex == nullptr || residual == nullptr || branch_out == nullptr || modulation == nullptr ||
        num_tokens < 0 || hidden_size <= 0 || mod_stride < hidden_size ||
        gate_col_offset < 0 || gate_col_offset + hidden_size > mod_stride) {
        return (int)cudaErrorInvalidValue;
    }
    if (num_tokens == 0) return 0;
    dim3 block(256);
    dim3 grid((unsigned)((hidden_size + 255) / 256), (unsigned)num_tokens);
    warp34_qwen_image::gated_residual_f16_kernel<<<grid, block, 0, (cudaStream_t)ex->stream>>>(
        (__half*)residual,
        (const __half*)branch_out,
        (const __half*)modulation,
        target_token_mask,
        num_tokens,
        hidden_size,
        mod_stride,
        gate_col_offset);
    return (int)cudaGetLastError();
}

extern "C" int warp34_qwen_image_qk_rmsnorm_complex_rope_d128_f16(
    const warp34_exec* ex,
    const void* q_in,
    const void* k_in,
    const void* norm_q_w,
    const void* norm_k_w,
    const float* cos_table,
    const float* sin_table,
    void* q_out,
    void* k_out,
    void* cache_k_out,
    int num_tokens,
    int num_heads,
    int cache_write_len,
    float eps) {
    if (ex == nullptr || q_in == nullptr || k_in == nullptr ||
        norm_q_w == nullptr || norm_k_w == nullptr ||
        cos_table == nullptr || sin_table == nullptr ||
        q_out == nullptr || k_out == nullptr ||
        num_tokens < 0 || num_heads <= 0 || cache_write_len < 0) {
        return (int)cudaErrorInvalidValue;
    }
    if (num_tokens == 0) return 0;
    dim3 grid((unsigned)num_tokens, (unsigned)(2 * num_heads));
    warp34_qwen_image::qk_rmsnorm_complex_rope_d128_f16_kernel<<<grid, 64, 0, (cudaStream_t)ex->stream>>>(
        (const __half*)q_in,
        (const __half*)k_in,
        (const __half*)norm_q_w,
        (const __half*)norm_k_w,
        cos_table,
        sin_table,
        (__half*)q_out,
        (__half*)k_out,
        (__half*)cache_k_out,
        num_tokens,
        num_heads,
        cache_write_len,
        eps);
    return (int)cudaGetLastError();
}
