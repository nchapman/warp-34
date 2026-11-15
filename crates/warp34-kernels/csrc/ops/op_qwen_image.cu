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

namespace warp34_qwen_image_vae {

// 1. QwenImage21RMS_norm over channels [channels, spatial] + optional SiLU.
// Matches PyTorch:
//   normalized = F.normalize(x.float(), dim=1, eps=1e-12).to(x.dtype)
//   out = normalized * sqrt(channels) * gamma
__global__ void vae_rms_norm_f16_kernel(
    const __half* __restrict__ x,
    const __half* __restrict__ gamma,
    __half* __restrict__ out,
    int channels,
    int spatial,
    int apply_silu) {
    const int s = (int)blockIdx.x * (int)blockDim.x + (int)threadIdx.x;
    if (s >= spatial) return;

    float sum_sq = 0.0f;
    for (int c = 0; c < channels; ++c) {
        const float v = __half2float(x[(int64_t)c * spatial + s]);
        sum_sq += v * v;
    }
    const float inv_norm = 1.0f / fmaxf(sqrtf(sum_sq), 1e-12f);
    const float scale = sqrtf((float)channels);

    for (int c = 0; c < channels; ++c) {
        const int64_t idx = (int64_t)c * spatial + s;
        const float v = __half2float(x[idx]);
        const __half n_h = __float2half(v * inv_norm);
        const __half s_h = __float2half(__half2float(n_h) * scale);
        __half o_h = __float2half(__half2float(s_h) * __half2float(gamma[c]));
        if (apply_silu != 0) {
            const float of = __half2float(o_h);
            o_h = __float2half(of / (1.0f + __nv_expf(-of)));
        }
        out[idx] = o_h;
    }
}

// 2. 2D im2col for horizontal output strip [oh_start .. oh_start + chunk_h]:
//    x [in_c, h_in, w_in] -> col [in_c * k_h * k_w, chunk_h * w_out]
__global__ void vae_im2col_f16_kernel(
    const __half* __restrict__ x,
    __half* __restrict__ col,
    int in_c,
    int h_in,
    int w_in,
    int k_h,
    int k_w,
    int pad_top,
    int pad_left,
    int stride_h,
    int stride_w,
    int oh_start,
    int chunk_h,
    int w_out) {
    const int chunk_spatial = chunk_h * w_out;
    const int k_dim = in_c * k_h * k_w;
    const int64_t total = (int64_t)k_dim * (int64_t)chunk_spatial;
    const int64_t idx = (int64_t)blockIdx.x * (int64_t)blockDim.x + (int64_t)threadIdx.x;
    if (idx >= total) return;

    const int s = (int)(idx % (int64_t)chunk_spatial);
    const int k = (int)(idx / (int64_t)chunk_spatial);
    const int local_oh = s / w_out;
    const int ow = s % w_out;
    const int oh = oh_start + local_oh;

    const int kw = k % k_w;
    const int kh = (k / k_w) % k_h;
    const int ic = k / (k_h * k_w);

    const int ih = oh * stride_h - pad_top + kh;
    const int iw = ow * stride_w - pad_left + kw;
    __half val = __float2half(0.0f);
    if (ih >= 0 && ih < h_in && iw >= 0 && iw < w_in) {
        val = x[((int64_t)ic * h_in + ih) * w_in + iw];
    }
    col[idx] = val;
}

// 3. Per-channel bias + optional residual add + optional [-1, 1] clamp on [channels, spatial].
__global__ void vae_bias_residual_f16_kernel(
    const __half* __restrict__ x,
    const __half* __restrict__ bias,
    const __half* __restrict__ residual,
    __half* __restrict__ out,
    int channels,
    int spatial,
    int clamp_pm1) {
    const int64_t total = (int64_t)channels * (int64_t)spatial;
    const int64_t idx = (int64_t)blockIdx.x * (int64_t)blockDim.x + (int64_t)threadIdx.x;
    if (idx >= total) return;

    const int c = (int)(idx / (int64_t)spatial);
    __half v_h = __float2half(__half2float(x[idx]) + __half2float(bias[c]));
    if (residual != nullptr) {
        v_h = __float2half(__half2float(v_h) + __half2float(residual[idx]));
    }
    if (clamp_pm1 != 0) {
        const float vf = fminf(fmaxf(__half2float(v_h), -1.0f), 1.0f);
        v_h = __float2half(vf);
    }
    out[idx] = v_h;
}

// 4. Nearest-exact 2x spatial upsample: [channels, h_in, w_in] -> [channels, 2*h_in, 2*w_in].
__global__ void vae_upsample2d_nearest2x_f16_kernel(
    const __half* __restrict__ x,
    __half* __restrict__ out,
    int channels,
    int h_in,
    int w_in) {
    const int h_out = h_in * 2;
    const int w_out = w_in * 2;
    const int spatial_out = h_out * w_out;
    const int64_t total = (int64_t)channels * (int64_t)spatial_out;
    const int64_t idx = (int64_t)blockIdx.x * (int64_t)blockDim.x + (int64_t)threadIdx.x;
    if (idx >= total) return;

    const int c = (int)(idx / (int64_t)spatial_out);
    const int rem = (int)(idx % (int64_t)spatial_out);
    const int oh = rem / w_out;
    const int ow = rem % w_out;
    const int ih = oh >> 1;
    const int iw = ow >> 1;
    out[idx] = x[((int64_t)c * h_in + ih) * w_in + iw];
}

// 5. QwenImage21AvgDown3D for single-frame image (T = 1) + optional residual add:
//    x [in_c, h_in, w_in] -> out [out_c, h_in / factor_s, w_in / factor_s]
__global__ void vae_avg_down_f16_kernel(
    const __half* __restrict__ x,
    const __half* __restrict__ residual,
    __half* __restrict__ out,
    int in_c,
    int out_c,
    int h_in,
    int w_in,
    int factor_t,
    int factor_s) {
    const int h_out = h_in / factor_s;
    const int w_out = w_in / factor_s;
    const int spatial_out = h_out * w_out;
    const int64_t total = (int64_t)out_c * (int64_t)spatial_out;
    const int64_t idx = (int64_t)blockIdx.x * (int64_t)blockDim.x + (int64_t)threadIdx.x;
    if (idx >= total) return;

    const int oc = (int)(idx / (int64_t)spatial_out);
    const int rem_s = (int)(idx % (int64_t)spatial_out);
    const int oh = rem_s / w_out;
    const int ow = rem_s % w_out;

    const int fs2 = factor_s * factor_s;
    const int factor = factor_t * fs2;
    const int group_size = (in_c * factor) / out_c;

    float acc = 0.0f;
    for (int k = 0; k < group_size; ++k) {
        const int flat = oc * group_size + k;
        const int ic = flat / factor;
        const int rem = flat % factor;
        const int t_idx = rem / fs2;
        if (t_idx == factor_t - 1) {
            const int sh = (rem / factor_s) % factor_s;
            const int sw = rem % factor_s;
            const int ih = oh * factor_s + sh;
            const int iw = ow * factor_s + sw;
            acc += __half2float(x[((int64_t)ic * h_in + ih) * w_in + iw]);
        }
    }
    __half v_h = __float2half(acc / (float)group_size);
    if (residual != nullptr) {
        v_h = __float2half(__half2float(v_h) + __half2float(residual[idx]));
    }
    out[idx] = v_h;
}

// 6. QwenImage21DupUp3D(first_chunk=True) for single-frame image (T = 1) + optional residual add:
//    x [in_c, h_in, w_in] -> out [out_c, h_in * factor_s, w_in * factor_s]
__global__ void vae_dup_up_f16_kernel(
    const __half* __restrict__ x,
    const __half* __restrict__ residual,
    __half* __restrict__ out,
    int in_c,
    int out_c,
    int h_in,
    int w_in,
    int factor_t,
    int factor_s) {
    const int h_out = h_in * factor_s;
    const int w_out = w_in * factor_s;
    const int spatial_out = h_out * w_out;
    const int64_t total = (int64_t)out_c * (int64_t)spatial_out;
    const int64_t idx = (int64_t)blockIdx.x * (int64_t)blockDim.x + (int64_t)threadIdx.x;
    if (idx >= total) return;

    const int oc = (int)(idx / (int64_t)spatial_out);
    const int rem_s = (int)(idx % (int64_t)spatial_out);
    const int oh = rem_s / w_out;
    const int ow = rem_s % w_out;
    const int ih = oh / factor_s;
    const int iw = ow / factor_s;
    const int sh = oh % factor_s;
    const int sw = ow % factor_s;

    const int factor = factor_t * factor_s * factor_s;
    const int repeats = (out_c * factor) / in_c;
    const int rep_c = ((oc * factor_t + (factor_t - 1)) * factor_s + sh) * factor_s + sw;
    const int ic = rep_c / repeats;

    __half v_h = x[((int64_t)ic * h_in + ih) * w_in + iw];
    if (residual != nullptr) {
        v_h = __float2half(__half2float(v_h) + __half2float(residual[idx]));
    }
    out[idx] = v_h;
}

// 7. Row-wise softmax over [seq_len, seq_len]: FP32 scores -> FP16 probabilities.
//    1 CTA (256 threads = 8 warps) per row.
__global__ void vae_softmax_f32_to_f16_kernel(
    const float* __restrict__ scores,
    __half* __restrict__ probs,
    int seq_len) {
    const int row = (int)blockIdx.x;
    if (row >= seq_len) return;
    const int tid = (int)threadIdx.x;
    const int warp_id = tid >> 5;
    const int lane = tid & 31;

    const float* s_row = scores + (int64_t)row * seq_len;
    __half* p_row = probs + (int64_t)row * seq_len;

    float max_val = -1e30f;
    for (int j = tid; j < seq_len; j += 256) {
        max_val = fmaxf(max_val, s_row[j]);
    }
    for (int offset = 16; offset > 0; offset >>= 1) {
        max_val = fmaxf(max_val, __shfl_xor_sync(0xffffffffu, max_val, offset));
    }
    __shared__ float smem_red[8];
    if (lane == 0) smem_red[warp_id] = max_val;
    __syncthreads();
    if (warp_id == 0) {
        float v = (lane < 8) ? smem_red[lane] : -1e30f;
        for (int offset = 4; offset > 0; offset >>= 1) {
            v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, offset));
        }
        if (lane == 0) smem_red[0] = v;
    }
    __syncthreads();
    max_val = smem_red[0];

    float sum_exp = 0.0f;
    for (int j = tid; j < seq_len; j += 256) {
        sum_exp += __nv_expf(s_row[j] - max_val);
    }
    for (int offset = 16; offset > 0; offset >>= 1) {
        sum_exp += __shfl_xor_sync(0xffffffffu, sum_exp, offset);
    }
    if (lane == 0) smem_red[warp_id] = sum_exp;
    __syncthreads();
    if (warp_id == 0) {
        float v = (lane < 8) ? smem_red[lane] : 0.0f;
        for (int offset = 4; offset > 0; offset >>= 1) {
            v += __shfl_xor_sync(0xffffffffu, v, offset);
        }
        if (lane == 0) smem_red[0] = v;
    }
    __syncthreads();
    const float inv_sum = 1.0f / smem_red[0];

    for (int j = tid; j < seq_len; j += 256) {
        p_row[j] = __float2half(__nv_expf(s_row[j] - max_val) * inv_sum);
    }
}

} // namespace warp34_qwen_image_vae

extern "C" int warp34_qwen_image_vae_rms_norm_f16(
    const warp34_exec* ex,
    const void* x,
    const void* gamma,
    void* out,
    int channels,
    int spatial,
    int apply_silu) {
    if (ex == nullptr || x == nullptr || gamma == nullptr || out == nullptr ||
        channels <= 0 || spatial < 0) {
        return (int)cudaErrorInvalidValue;
    }
    if (spatial == 0) return 0;
    const int threads = 256;
    const int blocks = (spatial + threads - 1) / threads;
    warp34_qwen_image_vae::vae_rms_norm_f16_kernel<<<blocks, threads, 0, (cudaStream_t)ex->stream>>>(
        (const __half*)x,
        (const __half*)gamma,
        (__half*)out,
        channels,
        spatial,
        apply_silu);
    return (int)cudaGetLastError();
}

extern "C" int warp34_qwen_image_vae_im2col_f16(
    const warp34_exec* ex,
    const void* x,
    void* col,
    int in_c,
    int h_in,
    int w_in,
    int k_h,
    int k_w,
    int pad_top,
    int pad_left,
    int stride_h,
    int stride_w,
    int oh_start,
    int chunk_h,
    int w_out) {
    if (ex == nullptr || x == nullptr || col == nullptr ||
        in_c <= 0 || h_in <= 0 || w_in <= 0 || k_h <= 0 || k_w <= 0 ||
        stride_h <= 0 || stride_w <= 0 || oh_start < 0 || chunk_h < 0 || w_out <= 0) {
        return (int)cudaErrorInvalidValue;
    }
    const int64_t total = (int64_t)in_c * k_h * k_w * (int64_t)chunk_h * w_out;
    if (total == 0) return 0;
    if (total > (int64_t)INT_MAX * 256) return (int)cudaErrorInvalidValue;
    const int threads = 256;
    const int blocks = (int)((total + threads - 1) / threads);
    warp34_qwen_image_vae::vae_im2col_f16_kernel<<<blocks, threads, 0, (cudaStream_t)ex->stream>>>(
        (const __half*)x,
        (__half*)col,
        in_c,
        h_in,
        w_in,
        k_h,
        k_w,
        pad_top,
        pad_left,
        stride_h,
        stride_w,
        oh_start,
        chunk_h,
        w_out);
    return (int)cudaGetLastError();
}

extern "C" int warp34_qwen_image_vae_bias_residual_f16(
    const warp34_exec* ex,
    const void* x,
    const void* bias,
    const void* residual,
    void* out,
    int channels,
    int spatial,
    int clamp_pm1) {
    if (ex == nullptr || x == nullptr || bias == nullptr || out == nullptr ||
        channels <= 0 || spatial < 0) {
        return (int)cudaErrorInvalidValue;
    }
    const int64_t total = (int64_t)channels * (int64_t)spatial;
    if (total == 0) return 0;
    if (total > (int64_t)INT_MAX * 256) return (int)cudaErrorInvalidValue;
    const int threads = 256;
    const int blocks = (int)((total + threads - 1) / threads);
    warp34_qwen_image_vae::vae_bias_residual_f16_kernel<<<blocks, threads, 0, (cudaStream_t)ex->stream>>>(
        (const __half*)x,
        (const __half*)bias,
        (const __half*)residual,
        (__half*)out,
        channels,
        spatial,
        clamp_pm1);
    return (int)cudaGetLastError();
}

extern "C" int warp34_qwen_image_vae_upsample2d_nearest2x_f16(
    const warp34_exec* ex,
    const void* x,
    void* out,
    int channels,
    int h_in,
    int w_in) {
    if (ex == nullptr || x == nullptr || out == nullptr ||
        channels <= 0 || h_in < 0 || w_in < 0 ||
        (int64_t)h_in * 2 > INT_MAX || (int64_t)w_in * 2 > INT_MAX) {
        return (int)cudaErrorInvalidValue;
    }
    const int64_t total = (int64_t)channels * ((int64_t)h_in * 2) * ((int64_t)w_in * 2);
    if (total == 0) return 0;
    if (total > (int64_t)INT_MAX * 256) return (int)cudaErrorInvalidValue;
    const int threads = 256;
    const int blocks = (int)((total + threads - 1) / threads);
    warp34_qwen_image_vae::vae_upsample2d_nearest2x_f16_kernel<<<blocks, threads, 0, (cudaStream_t)ex->stream>>>(
        (const __half*)x,
        (__half*)out,
        channels,
        h_in,
        w_in);
    return (int)cudaGetLastError();
}

extern "C" int warp34_qwen_image_vae_avg_down_f16(
    const warp34_exec* ex,
    const void* x,
    const void* residual,
    void* out,
    int in_c,
    int out_c,
    int h_in,
    int w_in,
    int factor_t,
    int factor_s) {
    if (ex == nullptr || x == nullptr || out == nullptr ||
        in_c <= 0 || out_c <= 0 || h_in <= 0 || w_in <= 0 ||
        factor_t <= 0 || factor_s <= 0 ||
        (int64_t)factor_t * factor_s > INT_MAX ||
        (int64_t)factor_t * factor_s * factor_s > INT_MAX ||
        (int64_t)in_c * factor_t * factor_s * factor_s > INT_MAX ||
        (h_in % factor_s) != 0 || (w_in % factor_s) != 0 ||
        (((int64_t)in_c * factor_t * factor_s * factor_s) % out_c) != 0) {
        return (int)cudaErrorInvalidValue;
    }
    const int h_out = h_in / factor_s;
    const int w_out = w_in / factor_s;
    const int64_t total = (int64_t)out_c * (int64_t)h_out * (int64_t)w_out;
    if (total > (int64_t)INT_MAX * 256) return (int)cudaErrorInvalidValue;
    const int threads = 256;
    const int blocks = (int)((total + threads - 1) / threads);
    warp34_qwen_image_vae::vae_avg_down_f16_kernel<<<blocks, threads, 0, (cudaStream_t)ex->stream>>>(
        (const __half*)x,
        (const __half*)residual,
        (__half*)out,
        in_c,
        out_c,
        h_in,
        w_in,
        factor_t,
        factor_s);
    return (int)cudaGetLastError();
}

extern "C" int warp34_qwen_image_vae_dup_up_f16(
    const warp34_exec* ex,
    const void* x,
    const void* residual,
    void* out,
    int in_c,
    int out_c,
    int h_in,
    int w_in,
    int factor_t,
    int factor_s) {
    if (ex == nullptr || x == nullptr || out == nullptr ||
        in_c <= 0 || out_c <= 0 || h_in <= 0 || w_in <= 0 ||
        factor_t <= 0 || factor_s <= 0 ||
        (int64_t)h_in * factor_s > INT_MAX || (int64_t)w_in * factor_s > INT_MAX ||
        (int64_t)factor_t * factor_s > INT_MAX ||
        (int64_t)factor_t * factor_s * factor_s > INT_MAX ||
        (int64_t)out_c * factor_t * factor_s * factor_s > INT_MAX ||
        (((int64_t)out_c * factor_t * factor_s * factor_s) % in_c) != 0) {
        return (int)cudaErrorInvalidValue;
    }
    const int h_out = h_in * factor_s;
    const int w_out = w_in * factor_s;
    const int64_t total = (int64_t)out_c * (int64_t)h_out * (int64_t)w_out;
    if (total > (int64_t)INT_MAX * 256) return (int)cudaErrorInvalidValue;
    const int threads = 256;
    const int blocks = (int)((total + threads - 1) / threads);
    warp34_qwen_image_vae::vae_dup_up_f16_kernel<<<blocks, threads, 0, (cudaStream_t)ex->stream>>>(
        (const __half*)x,
        (const __half*)residual,
        (__half*)out,
        in_c,
        out_c,
        h_in,
        w_in,
        factor_t,
        factor_s);
    return (int)cudaGetLastError();
}

extern "C" int warp34_qwen_image_vae_softmax_f32_to_f16(
    const warp34_exec* ex,
    const float* scores,
    void* probs,
    int seq_len) {
    if (ex == nullptr || scores == nullptr || probs == nullptr || seq_len < 0) {
        return (int)cudaErrorInvalidValue;
    }
    if (seq_len == 0) return 0;
    warp34_qwen_image_vae::vae_softmax_f32_to_f16_kernel<<<seq_len, 256, 0, (cudaStream_t)ex->stream>>>(
        scores,
        (__half*)probs,
        seq_len);
    return (int)cudaGetLastError();
}

