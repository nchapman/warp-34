// Compressed GDN checkpoint pack/unpack kernels (`WHT128-Q8_0` with BF16 block-32 scales).
//
// Checkpoints are quantized once on donation (`FP32` active `ssm_state` -> 1D WHT-128 along
// contiguous `head_k_dim = 128` + symmetric INT8 with BF16 scale per 32 elements) and
// dequantized once on prefix-cache hit (`WHT128-Q8_0` -> `FP32` active `ssm_state`), while
// `conv_state` (`FP16`) and optional `ple_conv_state` (`FP16`) are copied lossless into the
// same flat checkpoint frame.

#include "../warp34_cuda.h"
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <math.h>
#include <stdint.h>

namespace warp34_gdn_ckpt {

// Orthogonal normalized 1D Walsh-Hadamard transform of length 128 across one 32-lane warp
// (`lane` holds `float4` `[4*lane + 0 .. 4*lane + 3]`). Self-inverse (`H_128^2 = I_128`).
__device__ __forceinline__ void wht128_warp(float4& v, int lane) {
    // Stage 0 (stride 1) + Stage 1 (stride 2) inside each lane's float4.
    const float a0 = v.x + v.y;
    const float a1 = v.x - v.y;
    const float a2 = v.z + v.w;
    const float a3 = v.z - v.w;
    v.x = a0 + a2;
    v.y = a1 + a3;
    v.z = a0 - a2;
    v.w = a1 - a3;

    // Stages 2..6 (strides 4, 8, 16, 32, 64 -> lane XOR d = 1, 2, 4, 8, 16).
#pragma unroll
    for (int d = 1; d < 32; d <<= 1) {
        const float px = __shfl_xor_sync(0xffffffffu, v.x, d);
        const float py = __shfl_xor_sync(0xffffffffu, v.y, d);
        const float pz = __shfl_xor_sync(0xffffffffu, v.z, d);
        const float pw = __shfl_xor_sync(0xffffffffu, v.w, d);
        const float s = (lane & d) ? -1.0f : 1.0f;
        v.x = px + s * v.x;
        v.y = py + s * v.y;
        v.z = pz + s * v.z;
        v.w = pw + s * v.w;
    }

    constexpr float kInvSqrt128 = 0.08838834764831844655f;
    v.x *= kInvSqrt128;
    v.y *= kInvSqrt128;
    v.z *= kInvSqrt128;
    v.w *= kInvSqrt128;
}

__device__ __forceinline__ int8_t quant_sym_i8(float val, float inv_scale) {
    const float q = fminf(fmaxf(val * inv_scale, -127.0f), 127.0f);
    return static_cast<int8_t>(__float2int_rn(q));
}

__device__ __forceinline__ uint32_t pack4_i8(int8_t b0, int8_t b1, int8_t b2, int8_t b3) {
    return static_cast<uint32_t>(static_cast<uint8_t>(b0)) |
           (static_cast<uint32_t>(static_cast<uint8_t>(b1)) << 8) |
           (static_cast<uint32_t>(static_cast<uint8_t>(b2)) << 16) |
           (static_cast<uint32_t>(static_cast<uint8_t>(b3)) << 24);
}

__global__ __launch_bounds__(128, 4) void gdn_ckpt_pack_wht_q8_kernel(
    const uint64_t* __restrict__ ssm_layer_ptrs,
    int src_state_slot,
    int num_layers,
    int rows_per_layer,
    uint32_t* __restrict__ dst_qdata_u32,
    __nv_bfloat16* __restrict__ dst_scales_bf16) {
    const int warp_global = static_cast<int>(blockIdx.x) * 4 + (static_cast<int>(threadIdx.x) >> 5);
    const int total_rows = num_layers * rows_per_layer;
    if (warp_global >= total_rows) return;

    const int lane = static_cast<int>(threadIdx.x) & 31;
    const int layer = warp_global / rows_per_layer;
    const int row_in_layer = warp_global - layer * rows_per_layer;

    const float* layer_base = reinterpret_cast<const float*>(ssm_layer_ptrs[layer]);
    const size_t src_row_offset =
        (static_cast<size_t>(src_state_slot) * static_cast<size_t>(rows_per_layer) +
         static_cast<size_t>(row_in_layer)) *
        128u;
    float4 v = reinterpret_cast<const float4*>(layer_base + src_row_offset)[lane];

    wht128_warp(v, lane);

    // Block-32 amax reduction across 8-lane sub-warp (`8 * 4 = 32` contiguous elements).
    float amax = fmaxf(fmaxf(fabsf(v.x), fabsf(v.y)), fmaxf(fabsf(v.z), fabsf(v.w)));
    amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, 1));
    amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, 2));
    amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, 4));

    const __nv_bfloat16 scale_bf16 = __float2bfloat16_rn(amax * (1.0f / 127.0f));
    const float scale = __bfloat162float(scale_bf16);
    const float inv_scale = (scale > 0.0f) ? (1.0f / scale) : 0.0f;

    const uint32_t packed = pack4_i8(
        quant_sym_i8(v.x, inv_scale),
        quant_sym_i8(v.y, inv_scale),
        quant_sym_i8(v.z, inv_scale),
        quant_sym_i8(v.w, inv_scale));

    dst_qdata_u32[static_cast<size_t>(warp_global) * 32u + static_cast<size_t>(lane)] = packed;
    if ((lane & 7) == 0) {
        dst_scales_bf16[static_cast<size_t>(warp_global) * 4u + static_cast<size_t>(lane >> 3)] =
            scale_bf16;
    }
}

__global__ __launch_bounds__(128, 4) void gdn_ckpt_unpack_wht_q8_kernel(
    const uint32_t* __restrict__ src_qdata_u32,
    const __nv_bfloat16* __restrict__ src_scales_bf16,
    const uint64_t* __restrict__ ssm_layer_ptrs,
    int dst_state_slot,
    int num_layers,
    int rows_per_layer) {
    const int warp_global = static_cast<int>(blockIdx.x) * 4 + (static_cast<int>(threadIdx.x) >> 5);
    const int total_rows = num_layers * rows_per_layer;
    if (warp_global >= total_rows) return;

    const int lane = static_cast<int>(threadIdx.x) & 31;
    const int layer = warp_global / rows_per_layer;
    const int row_in_layer = warp_global - layer * rows_per_layer;

    const uint32_t packed =
        src_qdata_u32[static_cast<size_t>(warp_global) * 32u + static_cast<size_t>(lane)];
    const float scale = __bfloat162float(
        src_scales_bf16[static_cast<size_t>(warp_global) * 4u + static_cast<size_t>(lane >> 3)]);

    const int8_t b0 = static_cast<int8_t>(packed & 0xffu);
    const int8_t b1 = static_cast<int8_t>((packed >> 8) & 0xffu);
    const int8_t b2 = static_cast<int8_t>((packed >> 16) & 0xffu);
    const int8_t b3 = static_cast<int8_t>((packed >> 24) & 0xffu);
    float4 v = make_float4(
        static_cast<float>(b0) * scale,
        static_cast<float>(b1) * scale,
        static_cast<float>(b2) * scale,
        static_cast<float>(b3) * scale);

    wht128_warp(v, lane);

    float* layer_base = reinterpret_cast<float*>(ssm_layer_ptrs[layer]);
    const size_t dst_row_offset =
        (static_cast<size_t>(dst_state_slot) * static_cast<size_t>(rows_per_layer) +
         static_cast<size_t>(row_in_layer)) *
        128u;
    reinterpret_cast<float4*>(layer_base + dst_row_offset)[lane] = v;
}

__global__ __launch_bounds__(256, 4) void gdn_ckpt_copy_conv_pack_kernel(
    const uint64_t* __restrict__ conv_layer_ptrs,
    const uint16_t* __restrict__ ple_conv_pool,
    int src_state_slot,
    int num_layers,
    size_t conv_elems_per_slot,
    size_t ple_conv_elems_per_slot,
    uint16_t* __restrict__ dst_conv_u16,
    uint16_t* __restrict__ dst_ple_conv_u16) {
    const size_t total_gdn_conv = static_cast<size_t>(num_layers) * conv_elems_per_slot;
    const size_t total_elems = total_gdn_conv + ple_conv_elems_per_slot;
    for (size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         idx < total_elems;
         idx += static_cast<size_t>(gridDim.x) * blockDim.x) {
        if (idx < total_gdn_conv) {
            const size_t layer = idx / conv_elems_per_slot;
            const size_t elem = idx - layer * conv_elems_per_slot;
            const uint16_t* base = reinterpret_cast<const uint16_t*>(conv_layer_ptrs[layer]);
            dst_conv_u16[idx] =
                base[static_cast<size_t>(src_state_slot) * conv_elems_per_slot + elem];
        } else if (ple_conv_pool != nullptr && dst_ple_conv_u16 != nullptr) {
            const size_t ple_elem = idx - total_gdn_conv;
            dst_ple_conv_u16[ple_elem] =
                ple_conv_pool[static_cast<size_t>(src_state_slot) * ple_conv_elems_per_slot +
                              ple_elem];
        }
    }
}

__global__ __launch_bounds__(256, 4) void gdn_ckpt_copy_conv_unpack_kernel(
    const uint16_t* __restrict__ src_conv_u16,
    const uint16_t* __restrict__ src_ple_conv_u16,
    const uint64_t* __restrict__ conv_layer_ptrs,
    uint16_t* __restrict__ ple_conv_pool,
    int dst_state_slot,
    int num_layers,
    size_t conv_elems_per_slot,
    size_t ple_conv_elems_per_slot) {
    const size_t total_gdn_conv = static_cast<size_t>(num_layers) * conv_elems_per_slot;
    const size_t total_elems = total_gdn_conv + ple_conv_elems_per_slot;
    for (size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         idx < total_elems;
         idx += static_cast<size_t>(gridDim.x) * blockDim.x) {
        if (idx < total_gdn_conv) {
            const size_t layer = idx / conv_elems_per_slot;
            const size_t elem = idx - layer * conv_elems_per_slot;
            uint16_t* base = reinterpret_cast<uint16_t*>(conv_layer_ptrs[layer]);
            base[static_cast<size_t>(dst_state_slot) * conv_elems_per_slot + elem] =
                src_conv_u16[idx];
        } else if (ple_conv_pool != nullptr && src_ple_conv_u16 != nullptr) {
            const size_t ple_elem = idx - total_gdn_conv;
            ple_conv_pool[static_cast<size_t>(dst_state_slot) * ple_conv_elems_per_slot +
                          ple_elem] = src_ple_conv_u16[ple_elem];
        }
    }
}

}  // namespace warp34_gdn_ckpt

extern "C" int warp34_gdn_checkpoint_pack_wht_q8(
    const warp34_exec* ex,
    const uint64_t* ssm_layer_ptrs,
    const uint64_t* conv_layer_ptrs,
    const void* ple_conv_pool,
    int src_state_slot,
    int total_state_slots,
    int num_layers,
    int rows_per_layer,
    int head_k_dim,
    int conv_elems_per_slot,
    int ple_conv_elems_per_slot,
    void* dst_qdata,
    void* dst_scales_bf16,
    void* dst_conv_f16,
    void* dst_ple_conv_f16) {
    if (ex == nullptr) return (int)cudaErrorInvalidValue;
    if (num_layers < 0 || rows_per_layer < 0 || conv_elems_per_slot < 0 ||
        ple_conv_elems_per_slot < 0) {
        return (int)cudaErrorInvalidValue;
    }
    if (src_state_slot < 0 || src_state_slot >= total_state_slots) {
        return (int)cudaErrorInvalidValue;
    }
    const cudaStream_t stream = reinterpret_cast<cudaStream_t>(ex->stream);

    if (num_layers > 0 && rows_per_layer > 0) {
        if (head_k_dim != 128 || ssm_layer_ptrs == nullptr || dst_qdata == nullptr ||
            dst_scales_bf16 == nullptr) {
            return (int)cudaErrorInvalidValue;
        }
        const int total_rows = num_layers * rows_per_layer;
        const uint32_t blocks = static_cast<uint32_t>((total_rows + 3) / 4);
        warp34_gdn_ckpt::gdn_ckpt_pack_wht_q8_kernel<<<blocks, 128, 0, stream>>>(
            ssm_layer_ptrs,
            src_state_slot,
            num_layers,
            rows_per_layer,
            reinterpret_cast<uint32_t*>(dst_qdata),
            reinterpret_cast<__nv_bfloat16*>(dst_scales_bf16));
    }

    const size_t total_gdn_conv =
        static_cast<size_t>(num_layers) * static_cast<size_t>(conv_elems_per_slot);
    const size_t total_conv = total_gdn_conv + static_cast<size_t>(ple_conv_elems_per_slot);
    if (total_conv > 0) {
        if (total_gdn_conv > 0 && (conv_layer_ptrs == nullptr || dst_conv_f16 == nullptr)) {
            return (int)cudaErrorInvalidValue;
        }
        if (ple_conv_elems_per_slot > 0 &&
            (ple_conv_pool == nullptr || dst_ple_conv_f16 == nullptr)) {
            return (int)cudaErrorInvalidValue;
        }
        const size_t raw_blocks = (total_conv + 255u) / 256u;
        const uint32_t blocks = static_cast<uint32_t>(raw_blocks > 4096u ? 4096u : raw_blocks);
        warp34_gdn_ckpt::gdn_ckpt_copy_conv_pack_kernel<<<blocks, 256, 0, stream>>>(
            conv_layer_ptrs,
            reinterpret_cast<const uint16_t*>(ple_conv_pool),
            src_state_slot,
            num_layers,
            static_cast<size_t>(conv_elems_per_slot),
            static_cast<size_t>(ple_conv_elems_per_slot),
            reinterpret_cast<uint16_t*>(dst_conv_f16),
            reinterpret_cast<uint16_t*>(dst_ple_conv_f16));
    }

    return (int)cudaGetLastError();
}

extern "C" int warp34_gdn_checkpoint_unpack_wht_q8(
    const warp34_exec* ex,
    const void* src_qdata,
    const void* src_scales_bf16,
    const void* src_conv_f16,
    const void* src_ple_conv_f16,
    const uint64_t* ssm_layer_ptrs,
    const uint64_t* conv_layer_ptrs,
    void* ple_conv_pool,
    int dst_state_slot,
    int total_state_slots,
    int num_layers,
    int rows_per_layer,
    int head_k_dim,
    int conv_elems_per_slot,
    int ple_conv_elems_per_slot) {
    if (ex == nullptr) return (int)cudaErrorInvalidValue;
    if (num_layers < 0 || rows_per_layer < 0 || conv_elems_per_slot < 0 ||
        ple_conv_elems_per_slot < 0) {
        return (int)cudaErrorInvalidValue;
    }
    if (dst_state_slot < 0 || dst_state_slot >= total_state_slots) {
        return (int)cudaErrorInvalidValue;
    }
    const cudaStream_t stream = reinterpret_cast<cudaStream_t>(ex->stream);

    if (num_layers > 0 && rows_per_layer > 0) {
        if (head_k_dim != 128 || ssm_layer_ptrs == nullptr || src_qdata == nullptr ||
            src_scales_bf16 == nullptr) {
            return (int)cudaErrorInvalidValue;
        }
        const int total_rows = num_layers * rows_per_layer;
        const uint32_t blocks = static_cast<uint32_t>((total_rows + 3) / 4);
        warp34_gdn_ckpt::gdn_ckpt_unpack_wht_q8_kernel<<<blocks, 128, 0, stream>>>(
            reinterpret_cast<const uint32_t*>(src_qdata),
            reinterpret_cast<const __nv_bfloat16*>(src_scales_bf16),
            ssm_layer_ptrs,
            dst_state_slot,
            num_layers,
            rows_per_layer);
    }

    const size_t total_gdn_conv =
        static_cast<size_t>(num_layers) * static_cast<size_t>(conv_elems_per_slot);
    const size_t total_conv = total_gdn_conv + static_cast<size_t>(ple_conv_elems_per_slot);
    if (total_conv > 0) {
        if (total_gdn_conv > 0 && (conv_layer_ptrs == nullptr || src_conv_f16 == nullptr)) {
            return (int)cudaErrorInvalidValue;
        }
        if (ple_conv_elems_per_slot > 0 &&
            (ple_conv_pool == nullptr || src_ple_conv_f16 == nullptr)) {
            return (int)cudaErrorInvalidValue;
        }
        const size_t raw_blocks = (total_conv + 255u) / 256u;
        const uint32_t blocks = static_cast<uint32_t>(raw_blocks > 4096u ? 4096u : raw_blocks);
        warp34_gdn_ckpt::gdn_ckpt_copy_conv_unpack_kernel<<<blocks, 256, 0, stream>>>(
            reinterpret_cast<const uint16_t*>(src_conv_f16),
            reinterpret_cast<const uint16_t*>(src_ple_conv_f16),
            conv_layer_ptrs,
            reinterpret_cast<uint16_t*>(ple_conv_pool),
            dst_state_slot,
            num_layers,
            static_cast<size_t>(conv_elems_per_slot),
            static_cast<size_t>(ple_conv_elems_per_slot));
    }

    return (int)cudaGetLastError();
}
