// Clean-room CUDA kernels for Qwen-Image-2.1 (`QwenImage21Transformer2DModel` & `AutoencoderKLQwenImage21`):
// 1. Sinusoidal timestep projection (`QwenImage21TemporalTimesteps`, FP16-rounded timestep * 1000).
// 2. Unary SiLU (`x / (1 + exp(-x))`) on FP16.
// 3. Joint text + image hidden sequence assembly (`joint_row_map` gather).
// 4. Parameterless LayerNorm + AdaLN `(1 + scale)` modulation + optional in-register Hadamard-64.
// 5. Gated `tanh(gate) * branch_out` residual add with FP16 clamp (`[-65504, 65504]`), standalone & fused with AdaLN.
// 6. Fused per-head Q/K RMSNorm + 3-axis complex RoPE (`apply_rotary_emb_qwen(use_real=False)`) + optional prefix K cache write.
// 7. GPU-side `FlowMatchEulerDiscreteScheduler` step with optional classifier-free guidance (`euler_cfg_step_f16`).
// 8. Vectorized zero-division `AutoencoderKLQwenImage21` (`RMS_norm`, `im2col`, `bias_residual`, `upsample2x`, `avg_down`, `dup_up`, `softmax`).
#include "../warp34_cuda.h"
#include "hadamard_inline.cuh"
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <limits.h>
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

__global__ void silu_vec8_f16_kernel(
    const uint4* __restrict__ x,
    uint4* __restrict__ out,
    int64_t n_vec8) {
    const int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_vec8) return;
    const uint4 xv = x[idx];
    const __half* xh = reinterpret_cast<const __half*>(&xv);
    uint4 ov;
    __half* oh = reinterpret_cast<__half*>(&ov);
    #pragma unroll
    for (int d = 0; d < 8; ++d) {
        const float v = __half2float(xh[d]);
        oh[d] = __float2half_rn(v / (1.0f + __nv_expf(-v)));
    }
    out[idx] = ov;
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
    if (((hidden_size & 7) == 0) && ((((uintptr_t)src_row | (uintptr_t)dst_row) & 15u) == 0)) {
        const uint4* sv = reinterpret_cast<const uint4*>(src_row);
        uint4* dv = reinterpret_cast<uint4*>(dst_row);
        const int vec_cols = hidden_size >> 3;
        for (int c = threadIdx.x; c < vec_cols; c += blockDim.x) {
            dv[c] = sv[c];
        }
    } else {
        for (int c = threadIdx.x; c < hidden_size; c += blockDim.x) {
            dst_row[c] = src_row[c];
        }
    }
}

// Generic parameterless LayerNorm + AdaLN modulation `norm(x) * (1 + scale)`.
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
        const __half normed_h = __float2half_rn((__half2float(x_row[i]) - mean) * rstd);
        const __half scale_plus_1 = __hadd(one_h, scale_row[i]);
        out_row[i] = __hmul(normed_h, scale_plus_1);
    }
}

// Single-pass register-resident vectorized `hidden_size == 4096` AdaLN + optional Hadamard-64.
// 512 threads * 8 FP16 (`uint4`) = 4096 elements per row.
__global__ void __launch_bounds__(512, 2) adaln_modulate_4096_vec8_f16_kernel(
    const __half* __restrict__ x,
    const __half* __restrict__ modulation,
    const uint8_t* __restrict__ target_token_mask,
    __half* __restrict__ out,
    int num_tokens,
    int mod_stride,
    int scale_col_offset,
    float eps,
    int apply_had64) {
    const int row = blockIdx.x;
    if (row >= num_tokens) return;
    const int tid = threadIdx.x; // 0..511
    const int lane = tid & 31;
    const int warp_id = tid >> 5;

    __shared__ float s_sum[16];
    __shared__ float s_sq[16];

    const uint4 xv = reinterpret_cast<const uint4*>(x + (size_t)row * 4096)[tid];
    const __half* xh = reinterpret_cast<const __half*>(&xv);
    float xf[8];
    float local_sum = 0.0f;
    #pragma unroll
    for (int d = 0; d < 8; ++d) {
        xf[d] = __half2float(xh[d]);
        local_sum += xf[d];
    }
    local_sum = warp_reduce_sum(local_sum);
    if (lane == 0) s_sum[warp_id] = local_sum;
    __syncthreads();

    if (warp_id == 0) {
        float v = (tid < 16) ? s_sum[tid] : 0.0f;
        v = warp_reduce_sum(v);
        if (tid == 0) s_sum[0] = v * (1.0f / 4096.0f);
    }
    __syncthreads();
    const float mean = s_sum[0];

    float local_sq = 0.0f;
    #pragma unroll
    for (int d = 0; d < 8; ++d) {
        const float diff = xf[d] - mean;
        local_sq += diff * diff;
    }
    local_sq = warp_reduce_sum(local_sq);
    if (lane == 0) s_sq[warp_id] = local_sq;
    __syncthreads();

    if (warp_id == 0) {
        float v = (tid < 16) ? s_sq[tid] : 0.0f;
        v = warp_reduce_sum(v);
        if (tid == 0) s_sq[0] = rsqrtf(v * (1.0f / 4096.0f) + eps);
    }
    __syncthreads();
    const float rstd = s_sq[0];

    const int mod_row = (target_token_mask != nullptr && target_token_mask[row] == 0) ? 1 : 0;
    const uint4 sv = reinterpret_cast<const uint4*>(
        modulation + (size_t)mod_row * mod_stride + scale_col_offset)[tid];
    const __half* sh = reinterpret_cast<const __half*>(&sv);

    uint4 ov;
    __half* oh = reinterpret_cast<__half*>(&ov);
    const __half one_h = __float2half_rn(1.0f);
    #pragma unroll
    for (int d = 0; d < 8; ++d) {
        const __half normed_h = __float2half_rn((xf[d] - mean) * rstd);
        oh[d] = __hmul(normed_h, __hadd(one_h, sh[d]));
    }
    if (apply_had64 != 0) {
        ov = warp34::hadamard64_vec8_f16(ov, lane);
    }
    reinterpret_cast<uint4*>(out + (size_t)row * 4096)[tid] = ov;
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
    const int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (col >= hidden_size) return;

    for (int row = blockIdx.y; row < num_tokens; row += gridDim.y) {
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
}

__global__ void gated_residual_vec8_f16_kernel(
    uint4* __restrict__ residual,
    const uint4* __restrict__ branch_out,
    const __half* __restrict__ modulation,
    const uint8_t* __restrict__ target_token_mask,
    int num_tokens,
    int vec_cols,
    int mod_stride,
    int gate_col_offset) {
    const int vcol = blockIdx.x * blockDim.x + threadIdx.x;
    if (vcol >= vec_cols) return;

    for (int row = blockIdx.y; row < num_tokens; row += gridDim.y) {
        const int mod_row = (target_token_mask != nullptr && target_token_mask[row] == 0) ? 1 : 0;
        const uint4 gv = reinterpret_cast<const uint4*>(
            modulation + (size_t)mod_row * mod_stride + gate_col_offset)[vcol];
        const size_t idx = (size_t)row * vec_cols + vcol;
        const uint4 rv = residual[idx];
        const uint4 bv = branch_out[idx];

        const __half* gh = reinterpret_cast<const __half*>(&gv);
        const __half* rh = reinterpret_cast<const __half*>(&rv);
        const __half* bh = reinterpret_cast<const __half*>(&bv);
        uint4 ov;
        __half* oh = reinterpret_cast<__half*>(&ov);
        #pragma unroll
        for (int d = 0; d < 8; ++d) {
            const __half gate_tanh = __float2half_rn(__nv_tanhf(__half2float(gh[d])));
            const __half prod = __hmul(gate_tanh, bh[d]);
            float sum = __half2float(__hadd(rh[d], prod));
            if (sum > 65504.0f) sum = 65504.0f;
            if (sum < -65504.0f) sum = -65504.0f;
            oh[d] = __float2half_rn(sum);
        }
        residual[idx] = ov;
    }
}

// Single-pass fused `gated_residual` + `adaln_modulate` + optional `hadamard64` for `hidden_size == 4096`.
// Updates `residual` in-place AND emits the modulated (and optionally H_64-rotated) next-branch input `norm_out`
// directly from registers without re-reading `residual` from HBM.
__global__ void __launch_bounds__(512, 2) fused_gated_residual_adaln_4096_vec8_f16_kernel(
    __half* __restrict__ residual,
    const __half* __restrict__ branch_out,
    const __half* __restrict__ gate_modulation,
    const uint8_t* __restrict__ gate_target_mask,
    int gate_mod_stride,
    int gate_col_offset,
    const __half* __restrict__ scale_modulation,
    const uint8_t* __restrict__ scale_target_mask,
    int scale_mod_stride,
    int scale_col_offset,
    __half* __restrict__ norm_out,
    int num_tokens,
    float eps,
    int apply_had64) {
    const int row = blockIdx.x;
    if (row >= num_tokens) return;
    const int tid = threadIdx.x; // 0..511
    const int lane = tid & 31;
    const int warp_id = tid >> 5;

    __shared__ float s_sum[16];
    __shared__ float s_sq[16];

    const int gate_mod_row = (gate_target_mask != nullptr && gate_target_mask[row] == 0) ? 1 : 0;
    const uint4 gv = reinterpret_cast<const uint4*>(
        gate_modulation + (size_t)gate_mod_row * gate_mod_stride + gate_col_offset)[tid];
    const uint4 rv_in = reinterpret_cast<const uint4*>(residual + (size_t)row * 4096)[tid];
    const uint4 bv = reinterpret_cast<const uint4*>(branch_out + (size_t)row * 4096)[tid];

    const __half* gh = reinterpret_cast<const __half*>(&gv);
    const __half* rh_in = reinterpret_cast<const __half*>(&rv_in);
    const __half* bh = reinterpret_cast<const __half*>(&bv);

    uint4 rv_out;
    __half* rh_out = reinterpret_cast<__half*>(&rv_out);
    float xf[8];
    float local_sum = 0.0f;
    #pragma unroll
    for (int d = 0; d < 8; ++d) {
        const __half gate_tanh = __float2half_rn(__nv_tanhf(__half2float(gh[d])));
        const __half prod = __hmul(gate_tanh, bh[d]);
        float sum = __half2float(__hadd(rh_in[d], prod));
        if (sum > 65504.0f) sum = 65504.0f;
        if (sum < -65504.0f) sum = -65504.0f;
        const __half r_new = __float2half_rn(sum);
        rh_out[d] = r_new;
        xf[d] = __half2float(r_new);
        local_sum += xf[d];
    }
    reinterpret_cast<uint4*>(residual + (size_t)row * 4096)[tid] = rv_out;

    local_sum = warp_reduce_sum(local_sum);
    if (lane == 0) s_sum[warp_id] = local_sum;
    __syncthreads();

    if (warp_id == 0) {
        float v = (tid < 16) ? s_sum[tid] : 0.0f;
        v = warp_reduce_sum(v);
        if (tid == 0) s_sum[0] = v * (1.0f / 4096.0f);
    }
    __syncthreads();
    const float mean = s_sum[0];

    float local_sq = 0.0f;
    #pragma unroll
    for (int d = 0; d < 8; ++d) {
        const float diff = xf[d] - mean;
        local_sq += diff * diff;
    }
    local_sq = warp_reduce_sum(local_sq);
    if (lane == 0) s_sq[warp_id] = local_sq;
    __syncthreads();

    if (warp_id == 0) {
        float v = (tid < 16) ? s_sq[tid] : 0.0f;
        v = warp_reduce_sum(v);
        if (tid == 0) s_sq[0] = rsqrtf(v * (1.0f / 4096.0f) + eps);
    }
    __syncthreads();
    const float rstd = s_sq[0];

    const int scale_mod_row = (scale_target_mask != nullptr && scale_target_mask[row] == 0) ? 1 : 0;
    const uint4 sv = reinterpret_cast<const uint4*>(
        scale_modulation + (size_t)scale_mod_row * scale_mod_stride + scale_col_offset)[tid];
    const __half* sh = reinterpret_cast<const __half*>(&sv);

    uint4 ov;
    __half* oh = reinterpret_cast<__half*>(&ov);
    const __half one_h = __float2half_rn(1.0f);
    #pragma unroll
    for (int d = 0; d < 8; ++d) {
        const __half normed_h = __float2half_rn((xf[d] - mean) * rstd);
        oh[d] = __hmul(normed_h, __hadd(one_h, sh[d]));
    }
    if (apply_had64 != 0) {
        ov = warp34::hadamard64_vec8_f16(ov, lane);
    }
    reinterpret_cast<uint4*>(norm_out + (size_t)row * 4096)[tid] = ov;
}

// Fused per-head Q/K RMSNorm + complex 3-axis RoPE (`apply_rotary_emb_qwen(use_real=False)`).
// Block: (64, 4) = 256 threads (4 heads per CTA, sharing `cos_table[tok]` and `sin_table[tok]` in smem).
// `cos_table` and `sin_table` have shape `[num_tokens, 64]` in FP32.
// If `cache_k_out != nullptr` and `token_idx < cache_write_len`, K heads also write rotated K into `cache_k_out`.
__global__ void __launch_bounds__(256, 2) qk_rmsnorm_complex_rope_d128_f16_kernel(
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
    const int pair_idx = threadIdx.x; // 0..63
    const int sub_h = threadIdx.y;    // 0..3
    if (tok >= num_tokens || pair_idx >= 64) return;

    __shared__ float s_cos[64];
    __shared__ float s_sin[64];
    __shared__ float s_warp[4][2];

    if (sub_h == 0) {
        s_cos[pair_idx] = cos_table[(size_t)tok * 64 + pair_idx];
    } else if (sub_h == 1) {
        s_sin[pair_idx] = sin_table[(size_t)tok * 64 + pair_idx];
    }

    const int hk = blockIdx.y * 4 + sub_h;
    const bool valid_hk = (hk < 2 * num_heads);

    float v0 = 0.0f, v1 = 0.0f, sq = 0.0f;
    bool is_k = false;
    int head = 0;
    size_t head_offset = 0;
    const __half* nw = nullptr;

    if (valid_hk) {
        is_k = (hk >= num_heads);
        head = is_k ? (hk - num_heads) : hk;
        head_offset = ((size_t)tok * num_heads + head) * 128;
        const __half2 src2 = reinterpret_cast<const __half2*>((is_k ? k_in : q_in) + head_offset)[pair_idx];
        const float2 sf = __half22float2(src2);
        v0 = sf.x;
        v1 = sf.y;
        sq = warp_reduce_sum(v0 * v0 + v1 * v1);
        nw = is_k ? norm_k_w : norm_q_w;
    }

    const int lane = pair_idx & 31;
    const int warp_id = pair_idx >> 5;
    if (valid_hk && lane == 0) {
        s_warp[sub_h][warp_id] = sq;
    }
    __syncthreads();

    if (!valid_hk) return;

    const float total_sq = s_warp[sub_h][0] + s_warp[sub_h][1];
    const float rrms = rsqrtf(total_sq * (1.0f / 128.0f) + eps);
    const __half2 nw2 = reinterpret_cast<const __half2*>(nw)[pair_idx];
    const float2 nwf = __half22float2(nw2);

    // Match `attn.norm_q(query).to(value.dtype)` before complex RoPE in FP32.
    const float n0 = __half2float(__float2half_rn(v0 * rrms * nwf.x));
    const float n1 = __half2float(__float2half_rn(v1 * rrms * nwf.y));

    const float c = s_cos[pair_idx];
    const float s = s_sin[pair_idx];

    const __half2 rot2 = __floats2half2_rn(n0 * c - n1 * s, n0 * s + n1 * c);
    reinterpret_cast<__half2*>((is_k ? k_out : q_out) + head_offset)[pair_idx] = rot2;

    if (is_k && cache_k_out != nullptr && tok < cache_write_len) {
        reinterpret_cast<__half2*>(cache_k_out + head_offset)[pair_idx] = rot2;
    }
}

// GPU-side `FlowMatchEulerDiscreteScheduler` step with optional classifier-free guidance (`true_cfg_scale`).
// 0-ULP identical to `euler_cfg_step_f16` on CPU.
__global__ void euler_cfg_step_f16_kernel(
    __half* __restrict__ sample,
    const __half* __restrict__ cond_pred,
    const __half* __restrict__ uncond_pred,
    float true_cfg_scale,
    float dt,
    int64_t n) {
    const int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    float pred_h;
    if (uncond_pred != nullptr && true_cfg_scale > 1.0f) {
        const float pos = __half2float(cond_pred[idx]);
        const float neg = __half2float(uncond_pred[idx]);
        const float diff_h = __half2float(__float2half_rn(pos - neg));
        const float scaled_h = __half2float(__float2half_rn(true_cfg_scale * diff_h));
        pred_h = __half2float(__float2half_rn(neg + scaled_h));
    } else {
        pred_h = __half2float(cond_pred[idx]);
    }
    sample[idx] = __float2half_rn(__half2float(sample[idx]) + dt * pred_h);
}

__global__ void euler_cfg_step_vec8_f16_kernel(
    uint4* __restrict__ sample,
    const uint4* __restrict__ cond_pred,
    const uint4* __restrict__ uncond_pred,
    float true_cfg_scale,
    float dt,
    int64_t n_vec8) {
    const int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_vec8) return;
    const uint4 sv = sample[idx];
    const uint4 cv = cond_pred[idx];
    const __half* sh = reinterpret_cast<const __half*>(&sv);
    const __half* ch = reinterpret_cast<const __half*>(&cv);
    uint4 ov;
    __half* oh = reinterpret_cast<__half*>(&ov);
    if (uncond_pred != nullptr && true_cfg_scale > 1.0f) {
        const uint4 uv = uncond_pred[idx];
        const __half* uh = reinterpret_cast<const __half*>(&uv);
        #pragma unroll
        for (int d = 0; d < 8; ++d) {
            const float pos = __half2float(ch[d]);
            const float neg = __half2float(uh[d]);
            const float diff_h = __half2float(__float2half_rn(pos - neg));
            const float scaled_h = __half2float(__float2half_rn(true_cfg_scale * diff_h));
            const float pred_h = __half2float(__float2half_rn(neg + scaled_h));
            oh[d] = __float2half_rn(__half2float(sh[d]) + dt * pred_h);
        }
    } else {
        #pragma unroll
        for (int d = 0; d < 8; ++d) {
            oh[d] = __float2half_rn(__half2float(sh[d]) + dt * __half2float(ch[d]));
        }
    }
    sample[idx] = ov;
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
    if (((n & 7) == 0) && ((((uintptr_t)x | (uintptr_t)out) & 15u) == 0)) {
        const int64_t n_vec8 = n >> 3;
        const unsigned blocks = (unsigned)((n_vec8 + threads - 1) / threads);
        warp34_qwen_image::silu_vec8_f16_kernel<<<blocks, threads, 0, (cudaStream_t)ex->stream>>>(
            (const uint4*)x, (uint4*)out, n_vec8);
    } else {
        const unsigned blocks = (unsigned)((n + threads - 1) / threads);
        warp34_qwen_image::silu_f16_kernel<<<blocks, threads, 0, (cudaStream_t)ex->stream>>>(
            (const __half*)x, (__half*)out, n);
    }
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

extern "C" int warp34_qwen_image_adaln_modulate_ex_f16(
    const warp34_exec* ex,
    const void* x,
    const void* modulation,
    const uint8_t* target_token_mask,
    void* out,
    int num_tokens,
    int hidden_size,
    int mod_stride,
    int scale_col_offset,
    float eps,
    int apply_had64) {
    if (ex == nullptr || x == nullptr || modulation == nullptr || out == nullptr ||
        num_tokens < 0 || hidden_size <= 0 || mod_stride < hidden_size ||
        scale_col_offset < 0 || scale_col_offset + hidden_size > mod_stride) {
        return (int)cudaErrorInvalidValue;
    }
    if (apply_had64 != 0 && (hidden_size & 63) != 0) {
        return (int)cudaErrorInvalidValue;
    }
    if (num_tokens == 0) return 0;
    if (hidden_size == 4096 &&
        (mod_stride & 7) == 0 &&
        (scale_col_offset & 7) == 0 &&
        ((((uintptr_t)x | (uintptr_t)modulation | (uintptr_t)out) & 15u) == 0)) {
        warp34_qwen_image::adaln_modulate_4096_vec8_f16_kernel<<<(unsigned)num_tokens, 512, 0, (cudaStream_t)ex->stream>>>(
            (const __half*)x,
            (const __half*)modulation,
            target_token_mask,
            (__half*)out,
            num_tokens,
            mod_stride,
            scale_col_offset,
            eps,
            apply_had64);
        return (int)cudaGetLastError();
    }
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
    int rc = (int)cudaGetLastError();
    if (rc != 0 || apply_had64 == 0) return rc;
    return warp34_block_hadamard_inplace_f16(ex, out, num_tokens, hidden_size, 64);
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
    return warp34_qwen_image_adaln_modulate_ex_f16(
        ex, x, modulation, target_token_mask, out,
        num_tokens, hidden_size, mod_stride, scale_col_offset, eps, 0);
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
    const unsigned grid_y = (unsigned)(num_tokens > 65535 ? 65535 : num_tokens);
    if (((hidden_size & 7) == 0) &&
        ((mod_stride & 7) == 0) &&
        ((gate_col_offset & 7) == 0) &&
        ((((uintptr_t)residual | (uintptr_t)branch_out | (uintptr_t)modulation) & 15u) == 0)) {
        const int vec_cols = hidden_size >> 3;
        dim3 block(128);
        dim3 grid((unsigned)((vec_cols + 127) / 128), grid_y);
        warp34_qwen_image::gated_residual_vec8_f16_kernel<<<grid, block, 0, (cudaStream_t)ex->stream>>>(
            (uint4*)residual,
            (const uint4*)branch_out,
            (const __half*)modulation,
            target_token_mask,
            num_tokens,
            vec_cols,
            mod_stride,
            gate_col_offset);
        return (int)cudaGetLastError();
    }
    dim3 block(256);
    dim3 grid((unsigned)((hidden_size + 255) / 256), grid_y);
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

extern "C" int warp34_qwen_image_fused_gated_residual_adaln_f16(
    const warp34_exec* ex,
    void* residual,
    const void* branch_out,
    const void* gate_modulation,
    const uint8_t* gate_target_mask,
    int gate_mod_stride,
    int gate_col_offset,
    const void* scale_modulation,
    const uint8_t* scale_target_mask,
    int scale_mod_stride,
    int scale_col_offset,
    void* norm_out,
    int num_tokens,
    int hidden_size,
    float eps,
    int apply_had64) {
    if (ex == nullptr || residual == nullptr || branch_out == nullptr ||
        gate_modulation == nullptr || scale_modulation == nullptr || norm_out == nullptr ||
        num_tokens < 0 || hidden_size <= 0 ||
        gate_mod_stride < hidden_size || gate_col_offset < 0 || gate_col_offset + hidden_size > gate_mod_stride ||
        scale_mod_stride < hidden_size || scale_col_offset < 0 || scale_col_offset + hidden_size > scale_mod_stride) {
        return (int)cudaErrorInvalidValue;
    }
    if (apply_had64 != 0 && (hidden_size & 63) != 0) {
        return (int)cudaErrorInvalidValue;
    }
    if (num_tokens == 0) return 0;
    if (hidden_size == 4096 &&
        (gate_mod_stride & 7) == 0 && (gate_col_offset & 7) == 0 &&
        (scale_mod_stride & 7) == 0 && (scale_col_offset & 7) == 0 &&
        ((((uintptr_t)residual | (uintptr_t)branch_out | (uintptr_t)gate_modulation |
           (uintptr_t)scale_modulation | (uintptr_t)norm_out) & 15u) == 0)) {
        warp34_qwen_image::fused_gated_residual_adaln_4096_vec8_f16_kernel<<<(unsigned)num_tokens, 512, 0, (cudaStream_t)ex->stream>>>(
            (__half*)residual,
            (const __half*)branch_out,
            (const __half*)gate_modulation,
            gate_target_mask,
            gate_mod_stride,
            gate_col_offset,
            (const __half*)scale_modulation,
            scale_target_mask,
            scale_mod_stride,
            scale_col_offset,
            (__half*)norm_out,
            num_tokens,
            eps,
            apply_had64);
        return (int)cudaGetLastError();
    }
    int rc = warp34_qwen_image_gated_residual_f16(
        ex, residual, branch_out, gate_modulation, gate_target_mask,
        num_tokens, hidden_size, gate_mod_stride, gate_col_offset);
    if (rc != 0) return rc;
    return warp34_qwen_image_adaln_modulate_ex_f16(
        ex, residual, scale_modulation, scale_target_mask, norm_out,
        num_tokens, hidden_size, scale_mod_stride, scale_col_offset, eps, apply_had64);
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
    dim3 block(64, 4);
    dim3 grid((unsigned)num_tokens, (unsigned)((2 * num_heads + 3) / 4));
    warp34_qwen_image::qk_rmsnorm_complex_rope_d128_f16_kernel<<<grid, block, 0, (cudaStream_t)ex->stream>>>(
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

extern "C" int warp34_qwen_image_euler_cfg_step_f16(
    const warp34_exec* ex,
    void* sample_f16,
    const void* cond_pred_f16,
    const void* uncond_pred_f16,
    float true_cfg_scale,
    float dt,
    int64_t n_elems) {
    if (ex == nullptr || sample_f16 == nullptr || cond_pred_f16 == nullptr || n_elems < 0) {
        return (int)cudaErrorInvalidValue;
    }
    if (n_elems == 0) return 0;
    const int threads = 256;
    const uintptr_t align_mask = (uintptr_t)sample_f16 | (uintptr_t)cond_pred_f16 | (uintptr_t)uncond_pred_f16;
    if (((n_elems & 7) == 0) && ((align_mask & 15u) == 0)) {
        const int64_t n_vec8 = n_elems >> 3;
        const unsigned blocks = (unsigned)((n_vec8 + threads - 1) / threads);
        warp34_qwen_image::euler_cfg_step_vec8_f16_kernel<<<blocks, threads, 0, (cudaStream_t)ex->stream>>>(
            (uint4*)sample_f16,
            (const uint4*)cond_pred_f16,
            (const uint4*)uncond_pred_f16,
            true_cfg_scale,
            dt,
            n_vec8);
    } else {
        const unsigned blocks = (unsigned)((n_elems + threads - 1) / threads);
        warp34_qwen_image::euler_cfg_step_f16_kernel<<<blocks, threads, 0, (cudaStream_t)ex->stream>>>(
            (__half*)sample_f16,
            (const __half*)cond_pred_f16,
            (const __half*)uncond_pred_f16,
            true_cfg_scale,
            dt,
            n_elems);
    }
    return (int)cudaGetLastError();
}

namespace warp34_qwen_image_vae {

// 1a. Scalar fallback QwenImage21RMS_norm over channels [channels, spatial] + optional SiLU.
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
        const float v = fminf(fmaxf(__half2float(x[(int64_t)c * spatial + s]), -65504.0f), 65504.0f);
        sum_sq += v * v;
    }
    const float inv_norm = 1.0f / fmaxf(sqrtf(sum_sq), 1e-12f);
    const float scale = sqrtf((float)channels);

    for (int c = 0; c < channels; ++c) {
        const int64_t idx = (int64_t)c * spatial + s;
        const float v = fminf(fmaxf(__half2float(x[idx]), -65504.0f), 65504.0f);
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

// 1b. Coalesced `uint4` (8 spatial pixels per thread) QwenImage21RMS_norm with shared-memory `gamma`.
__global__ void vae_rms_norm_vec8_f16_kernel(
    const uint4* __restrict__ x,
    const __half* __restrict__ gamma,
    uint4* __restrict__ out,
    int channels,
    int spatial_vec8,
    int apply_silu) {
    extern __shared__ float s_gamma[];
    for (int c = threadIdx.x; c < channels; c += blockDim.x) {
        s_gamma[c] = __half2float(gamma[c]);
    }
    __syncthreads();

    const int s8 = (int)blockIdx.x * (int)blockDim.x + (int)threadIdx.x;
    if (s8 >= spatial_vec8) return;

    float sum_sq[8] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
    for (int c = 0; c < channels; ++c) {
        const uint4 xv = x[(size_t)c * spatial_vec8 + s8];
        const __half* xh = reinterpret_cast<const __half*>(&xv);
        #pragma unroll
        for (int d = 0; d < 8; ++d) {
            const float v = fminf(fmaxf(__half2float(xh[d]), -65504.0f), 65504.0f);
            sum_sq[d] += v * v;
        }
    }

    float inv_norm[8];
    #pragma unroll
    for (int d = 0; d < 8; ++d) {
        inv_norm[d] = 1.0f / fmaxf(sqrtf(sum_sq[d]), 1e-12f);
    }
    const float scale = sqrtf((float)channels);

    for (int c = 0; c < channels; ++c) {
        const size_t idx = (size_t)c * spatial_vec8 + s8;
        const uint4 xv = x[idx];
        const __half* xh = reinterpret_cast<const __half*>(&xv);
        const float gc = s_gamma[c];
        uint4 ov;
        __half* oh = reinterpret_cast<__half*>(&ov);
        #pragma unroll
        for (int d = 0; d < 8; ++d) {
            const float v = fminf(fmaxf(__half2float(xh[d]), -65504.0f), 65504.0f);
            const __half n_h = __float2half(v * inv_norm[d]);
            const __half s_h = __float2half(__half2float(n_h) * scale);
            __half o_h = __float2half(__half2float(s_h) * gc);
            if (apply_silu != 0) {
                const float of = __half2float(o_h);
                o_h = __float2half(of / (1.0f + __nv_expf(-of)));
            }
            oh[d] = o_h;
        }
        out[idx] = ov;
    }
}

// 2a. Fast zero-division 2D `uint4` im2col for `stride_w == 1` & `(w_out & 7) == 0`:
//     `blockIdx.y` = `k_row` in `0 .. in_c * k_h * k_w` (`<= 65535`).
//     Each thread writes 8 contiguous `__half`s (`uint4`) to `col[k_row, local_oh * w_out + ow .. +8]`.
__global__ void vae_im2col_stride1_vec8_f16_kernel(
    const uint16_t* __restrict__ x,
    uint4* __restrict__ col,
    int h_in,
    int w_in,
    int k_h,
    int k_w,
    int pad_top,
    int pad_left,
    int stride_h,
    int oh_start,
    int chunk_h,
    int w_out_vec8) {
    const int k = (int)blockIdx.y;
    const int s8 = (int)blockIdx.x * (int)blockDim.x + (int)threadIdx.x;
    const int total_s8 = chunk_h * w_out_vec8;
    if (s8 >= total_s8) return;

    __shared__ int s_ic, s_kh, s_kw;
    if (threadIdx.x == 0) {
        const int khw = k_h * k_w;
        s_ic = k / khw;
        const int rem = k - s_ic * khw;
        s_kh = rem / k_w;
        s_kw = rem - s_kh * k_w;
    }
    __syncthreads();

    const int local_oh = s8 / w_out_vec8;
    const int ow_vec = s8 - local_oh * w_out_vec8;
    const int oh = oh_start + local_oh;
    const int ih = oh * stride_h - pad_top + s_kh;
    const size_t dst_idx = (size_t)k * total_s8 + s8;

    if (ih < 0 || ih >= h_in) {
        col[dst_idx] = make_uint4(0u, 0u, 0u, 0u);
        return;
    }

    const int iw = (ow_vec << 3) - pad_left + s_kw;
    const uint16_t* x_row = x + ((size_t)s_ic * h_in + ih) * w_in;
    uint4 ov;
    if (iw >= 0 && iw + 8 <= w_in) {
        const uint16_t* src = x_row + iw;
        if (((uintptr_t)src & 15u) == 0) {
            ov = *reinterpret_cast<const uint4*>(src);
        } else {
            uint16_t* oh = reinterpret_cast<uint16_t*>(&ov);
            #pragma unroll
            for (int d = 0; d < 8; ++d) {
                oh[d] = src[d];
            }
        }
    } else {
        uint16_t* oh = reinterpret_cast<uint16_t*>(&ov);
        #pragma unroll
        for (int d = 0; d < 8; ++d) {
            const int iwd = iw + d;
            oh[d] = (iwd >= 0 && iwd < w_in) ? x_row[iwd] : 0u;
        }
    }
    col[dst_idx] = ov;
}

// 2b. General 2D im2col fallback (2D grid avoids 64-bit signed division when `k_dim <= 65535`).
__global__ void vae_im2col_2d_f16_kernel(
    const __half* __restrict__ x,
    __half* __restrict__ col,
    int h_in,
    int w_in,
    int k_h,
    int k_w,
    int pad_top,
    int pad_left,
    int stride_h,
    int stride_w,
    int oh_start,
    int chunk_spatial,
    int w_out) {
    const int k = (int)blockIdx.y;
    const int s = (int)blockIdx.x * (int)blockDim.x + (int)threadIdx.x;
    if (s >= chunk_spatial) return;

    const int khw = k_h * k_w;
    const int ic = k / khw;
    const int rem = k - ic * khw;
    const int kh = rem / k_w;
    const int kw = rem - kh * k_w;

    const int local_oh = s / w_out;
    const int ow = s - local_oh * w_out;
    const int oh = oh_start + local_oh;

    const int ih = oh * stride_h - pad_top + kh;
    const int iw = ow * stride_w - pad_left + kw;
    __half val = __float2half(0.0f);
    if (ih >= 0 && ih < h_in && iw >= 0 && iw < w_in) {
        val = x[((int64_t)ic * h_in + ih) * w_in + iw];
    }
    col[(size_t)k * chunk_spatial + s] = val;
}

// 2c. Arbitrary `k_dim` fallback when `k_dim > 65535`.
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

// 3a. Vectorized 2D per-channel bias + optional residual + optional [-1, 1] clamp (`uint4`, zero division).
__global__ void vae_bias_residual_vec8_f16_kernel(
    const uint4* __restrict__ x,
    const __half* __restrict__ bias,
    const uint4* __restrict__ residual,
    uint4* __restrict__ out,
    int spatial_vec8,
    int clamp_pm1) {
    const int c = (int)blockIdx.y;
    const int s8 = (int)blockIdx.x * (int)blockDim.x + (int)threadIdx.x;
    if (s8 >= spatial_vec8) return;

    const float bc = __half2float(bias[c]);
    const size_t idx = (size_t)c * spatial_vec8 + s8;
    const uint4 xv = x[idx];
    const __half* xh = reinterpret_cast<const __half*>(&xv);
    uint4 ov;
    __half* oh = reinterpret_cast<__half*>(&ov);
    if (residual != nullptr) {
        const uint4 rv = residual[idx];
        const __half* rh = reinterpret_cast<const __half*>(&rv);
        #pragma unroll
        for (int d = 0; d < 8; ++d) {
            __half v_h = __float2half(__half2float(xh[d]) + bc);
            v_h = __float2half(__half2float(v_h) + __half2float(rh[d]));
            if (clamp_pm1 != 0) {
                v_h = __float2half(fminf(fmaxf(__half2float(v_h), -1.0f), 1.0f));
            }
            oh[d] = v_h;
        }
    } else {
        #pragma unroll
        for (int d = 0; d < 8; ++d) {
            __half v_h = __float2half(__half2float(xh[d]) + bc);
            if (clamp_pm1 != 0) {
                v_h = __float2half(fminf(fmaxf(__half2float(v_h), -1.0f), 1.0f));
            }
            oh[d] = v_h;
        }
    }
    out[idx] = ov;
}

// 3b. Scalar fallback per-channel bias + optional residual + optional [-1, 1] clamp.
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

// 4a. Vectorized nearest-exact 2x spatial upsample (`uint2` = 4 input pixels -> `uint4` = 8 output pixels, zero 64-bit div).
__global__ void vae_upsample2d_nearest2x_vec4_f16_kernel(
    const uint2* __restrict__ x,
    uint4* __restrict__ out,
    int h_in,
    int w_in_vec4) {
    const int c = (int)blockIdx.y;
    const int idx = (int)blockIdx.x * (int)blockDim.x + (int)threadIdx.x;
    const int h_out = h_in * 2;
    const int total = h_out * w_in_vec4;
    if (idx >= total) return;

    const int oh = idx / w_in_vec4;
    const int iw_vec = idx - oh * w_in_vec4;
    const int ih = oh >> 1;

    const uint2 sv = x[((size_t)c * h_in + ih) * w_in_vec4 + iw_vec];
    const uint16_t* sh = reinterpret_cast<const uint16_t*>(&sv);
    uint4 ov;
    uint16_t* dh = reinterpret_cast<uint16_t*>(&ov);
    dh[0] = sh[0]; dh[1] = sh[0];
    dh[2] = sh[1]; dh[3] = sh[1];
    dh[4] = sh[2]; dh[5] = sh[2];
    dh[6] = sh[3]; dh[7] = sh[3];
    out[((size_t)c * h_out + oh) * w_in_vec4 + iw_vec] = ov;
}

// 4b. Scalar fallback nearest-exact 2x spatial upsample.
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

// 5. QwenImage21AvgDown3D for single-frame image (T = 1) + optional residual add (2D grid avoids 64-bit div).
__global__ void vae_avg_down_2d_f16_kernel(
    const __half* __restrict__ x,
    const __half* __restrict__ residual,
    __half* __restrict__ out,
    int in_c,
    int out_c,
    int h_in,
    int w_in,
    int factor_t,
    int factor_s) {
    const int oc = (int)blockIdx.y;
    const int s_idx = (int)blockIdx.x * (int)blockDim.x + (int)threadIdx.x;
    const int h_out = h_in / factor_s;
    const int w_out = w_in / factor_s;
    const int spatial_out = h_out * w_out;
    if (oc >= out_c || s_idx >= spatial_out) return;

    const int oh = s_idx / w_out;
    const int ow = s_idx - oh * w_out;

    const int fs2 = factor_s * factor_s;
    const int factor = factor_t * fs2;
    const int group_size = (in_c * factor) / out_c;

    float acc = 0.0f;
    for (int k = 0; k < group_size; ++k) {
        const int flat = oc * group_size + k;
        const int ic = flat / factor;
        const int rem = flat - ic * factor;
        const int t_idx = rem / fs2;
        if (t_idx == factor_t - 1) {
            const int sh = (rem / factor_s) % factor_s;
            const int sw = rem % factor_s;
            const int ih = oh * factor_s + sh;
            const int iw = ow * factor_s + sw;
            acc += __half2float(x[((int64_t)ic * h_in + ih) * w_in + iw]);
        }
    }
    const size_t out_idx = (size_t)oc * spatial_out + s_idx;
    __half v_h = __float2half(acc / (float)group_size);
    if (residual != nullptr) {
        v_h = __float2half(__half2float(v_h) + __half2float(residual[out_idx]));
    }
    out[out_idx] = v_h;
}

// 6. QwenImage21DupUp3D(first_chunk=True) for single-frame image (T = 1) + optional residual add (2D grid avoids 64-bit div).
__global__ void vae_dup_up_2d_f16_kernel(
    const __half* __restrict__ x,
    const __half* __restrict__ residual,
    __half* __restrict__ out,
    int in_c,
    int out_c,
    int h_in,
    int w_in,
    int factor_t,
    int factor_s,
    float x_scale) {
    const int oc = (int)blockIdx.y;
    const int s_idx = (int)blockIdx.x * (int)blockDim.x + (int)threadIdx.x;
    const int h_out = h_in * factor_s;
    const int w_out = w_in * factor_s;
    const int spatial_out = h_out * w_out;
    if (oc >= out_c || s_idx >= spatial_out) return;

    const int oh = s_idx / w_out;
    const int ow = s_idx - oh * w_out;
    const int ih = oh / factor_s;
    const int iw = ow / factor_s;
    const int sh = oh - ih * factor_s;
    const int sw = ow - iw * factor_s;

    const int factor = factor_t * factor_s * factor_s;
    const int repeats = (out_c * factor) / in_c;
    const int rep_c = ((oc * factor_t + (factor_t - 1)) * factor_s + sh) * factor_s + sw;
    const int ic = rep_c / repeats;

    const size_t out_idx = (size_t)oc * spatial_out + s_idx;
    __half v_h = x[((int64_t)ic * h_in + ih) * w_in + iw];
    if (x_scale != 1.0f) {
        v_h = __float2half(__half2float(v_h) * x_scale);
    }
    if (residual != nullptr) {
        v_h = __float2half(__half2float(v_h) + __half2float(residual[out_idx]));
    }
    out[out_idx] = v_h;
}

// 7. Row-wise softmax over [seq_len, seq_len]: FP32 scores -> FP16 probabilities.
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
    __shared__ float smem_max[8];
    __shared__ float smem_sum[8];
    __shared__ float smem_max_total;
    __shared__ float smem_sum_total;
    if (lane == 0) smem_max[warp_id] = max_val;
    __syncthreads();
    if (warp_id == 0) {
        float v = (lane < 8) ? smem_max[lane] : -1e30f;
        for (int offset = 4; offset > 0; offset >>= 1) {
            v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, offset));
        }
        if (lane == 0) smem_max_total = v;
    }
    __syncthreads();
    max_val = smem_max_total;

    float sum_exp = 0.0f;
    for (int j = tid; j < seq_len; j += 256) {
        sum_exp += __nv_expf(s_row[j] - max_val);
    }
    for (int offset = 16; offset > 0; offset >>= 1) {
        sum_exp += __shfl_xor_sync(0xffffffffu, sum_exp, offset);
    }
    if (lane == 0) smem_sum[warp_id] = sum_exp;
    __syncthreads();
    if (warp_id == 0) {
        float v = (lane < 8) ? smem_sum[lane] : 0.0f;
        for (int offset = 4; offset > 0; offset >>= 1) {
            v += __shfl_xor_sync(0xffffffffu, v, offset);
        }
        if (lane == 0) smem_sum_total = v;
    }
    __syncthreads();
    const float inv_sum = 1.0f / smem_sum_total;

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
    if (((spatial & 7) == 0) &&
        channels <= 2048 &&
        ((((uintptr_t)x | (uintptr_t)out) & 15u) == 0)) {
        const int spatial_vec8 = spatial >> 3;
        const int blocks = (spatial_vec8 + threads - 1) / threads;
        const size_t smem = (size_t)channels * sizeof(float);
        warp34_qwen_image_vae::vae_rms_norm_vec8_f16_kernel<<<blocks, threads, smem, (cudaStream_t)ex->stream>>>(
            (const uint4*)x,
            (const __half*)gamma,
            (uint4*)out,
            channels,
            spatial_vec8,
            apply_silu);
        return (int)cudaGetLastError();
    }
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
    const int64_t k_dim64 = (int64_t)in_c * k_h * k_w;
    const int64_t chunk_spatial64 = (int64_t)chunk_h * w_out;
    const int64_t total = k_dim64 * chunk_spatial64;
    if (total == 0) return 0;
    if (total > (int64_t)INT_MAX * 256) return (int)cudaErrorInvalidValue;

    const int threads = 256;
    if (stride_w == 1 &&
        ((w_out & 7) == 0) &&
        k_dim64 <= 65535 &&
        chunk_spatial64 <= INT_MAX &&
        (((uintptr_t)col & 15u) == 0) &&
        (((uintptr_t)x & 1u) == 0)) {
        const int w_out_vec8 = w_out >> 3;
        const int total_s8 = chunk_h * w_out_vec8;
        dim3 grid((unsigned)((total_s8 + threads - 1) / threads), (unsigned)k_dim64);
        warp34_qwen_image_vae::vae_im2col_stride1_vec8_f16_kernel<<<grid, threads, 0, (cudaStream_t)ex->stream>>>(
            (const uint16_t*)x,
            (uint4*)col,
            h_in,
            w_in,
            k_h,
            k_w,
            pad_top,
            pad_left,
            stride_h,
            oh_start,
            chunk_h,
            w_out_vec8);
        return (int)cudaGetLastError();
    }

    if (k_dim64 <= 65535 && chunk_spatial64 <= INT_MAX) {
        dim3 grid((unsigned)(((int)chunk_spatial64 + threads - 1) / threads), (unsigned)k_dim64);
        warp34_qwen_image_vae::vae_im2col_2d_f16_kernel<<<grid, threads, 0, (cudaStream_t)ex->stream>>>(
            (const __half*)x,
            (__half*)col,
            h_in,
            w_in,
            k_h,
            k_w,
            pad_top,
            pad_left,
            stride_h,
            stride_w,
            oh_start,
            (int)chunk_spatial64,
            w_out);
        return (int)cudaGetLastError();
    }

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
    const uintptr_t align_mask = (uintptr_t)x | (uintptr_t)out | (uintptr_t)residual;
    if (((spatial & 7) == 0) && channels <= 65535 && ((align_mask & 15u) == 0)) {
        const int spatial_vec8 = spatial >> 3;
        dim3 grid((unsigned)((spatial_vec8 + threads - 1) / threads), (unsigned)channels);
        warp34_qwen_image_vae::vae_bias_residual_vec8_f16_kernel<<<grid, threads, 0, (cudaStream_t)ex->stream>>>(
            (const uint4*)x,
            (const __half*)bias,
            (const uint4*)residual,
            (uint4*)out,
            spatial_vec8,
            clamp_pm1);
        return (int)cudaGetLastError();
    }
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
    if (((w_in & 3) == 0) &&
        channels <= 65535 &&
        ((int64_t)h_in * 2 * (w_in >> 2)) <= INT_MAX &&
        (((uintptr_t)x & 7u) == 0) &&
        (((uintptr_t)out & 15u) == 0)) {
        const int w_in_vec4 = w_in >> 2;
        const int total_vec = (h_in * 2) * w_in_vec4;
        dim3 grid((unsigned)((total_vec + threads - 1) / threads), (unsigned)channels);
        warp34_qwen_image_vae::vae_upsample2d_nearest2x_vec4_f16_kernel<<<grid, threads, 0, (cudaStream_t)ex->stream>>>(
            (const uint2*)x,
            (uint4*)out,
            h_in,
            w_in_vec4);
        return (int)cudaGetLastError();
    }
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
    const int64_t spatial_out = (int64_t)h_out * (int64_t)w_out;
    if (spatial_out > INT_MAX || out_c > 65535) return (int)cudaErrorInvalidValue;
    const int threads = 256;
    dim3 grid((unsigned)(((int)spatial_out + threads - 1) / threads), (unsigned)out_c);
    warp34_qwen_image_vae::vae_avg_down_2d_f16_kernel<<<grid, threads, 0, (cudaStream_t)ex->stream>>>(
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

extern "C" int warp34_qwen_image_vae_dup_up_scaled_f16(
    const warp34_exec* ex,
    const void* x,
    const void* residual,
    void* out,
    int in_c,
    int out_c,
    int h_in,
    int w_in,
    int factor_t,
    int factor_s,
    float x_scale) {
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
    const int64_t spatial_out = (int64_t)h_out * (int64_t)w_out;
    if (spatial_out > INT_MAX || out_c > 65535) return (int)cudaErrorInvalidValue;
    const int threads = 256;
    dim3 grid((unsigned)(((int)spatial_out + threads - 1) / threads), (unsigned)out_c);
    warp34_qwen_image_vae::vae_dup_up_2d_f16_kernel<<<grid, threads, 0, (cudaStream_t)ex->stream>>>(
        (const __half*)x,
        (const __half*)residual,
        (__half*)out,
        in_c,
        out_c,
        h_in,
        w_in,
        factor_t,
        factor_s,
        x_scale);
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
    return warp34_qwen_image_vae_dup_up_scaled_f16(
        ex, x, residual, out, in_c, out_c, h_in, w_in, factor_t, factor_s, 1.0f);
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
