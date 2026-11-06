#include "../w34_cuda.h"
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cstdint>

namespace {

// Each 4-lane subgroup owns H16 (4 values/lane); each 16-lane subgroup owns H64 (4 values/lane).
// Executing bits 0..1 via an in-register H4 butterfly followed by bits 2..log2(B)-1 via warp
// shuffles preserves strict bit-order FP32 summation while coalescing 8-byte (uint2) loads/stores.
template <int B>
__global__ void block_hadamard_f16(const __half* src, __half* dst,
                                  int k, int padded_k, int64_t groups) {
    static_assert(B == 16 || B == 64, "block_hadamard_f16 supports B=16 or B=64");
    constexpr int lanes = B / 4;
    constexpr float scale = (B == 16) ? 0.25f : 0.125f;
    const int lane = threadIdx.x & (lanes - 1);
    const int64_t first = (int64_t(blockIdx.x) * blockDim.x + threadIdx.x) / lanes;
    const int64_t step = (int64_t(gridDim.x) * blockDim.x) / lanes;
    const int blocks_per_row = padded_k / B;
    const bool vec_aligned =
        ((reinterpret_cast<uintptr_t>(src) | reinterpret_cast<uintptr_t>(dst)) & 7u) == 0u;

    for (int64_t group = first; group < groups; group += step) {
        const int64_t row = group / blocks_per_row;
        const int col = int(group % blocks_per_row) * B + lane * 4;
        float v0 = 0.0f, v1 = 0.0f, v2 = 0.0f, v3 = 0.0f;
        if (col < k) {
            const __half* sp = src + row * k + col;
            if (vec_aligned) {
                const uint2 raw = *reinterpret_cast<const uint2*>(sp);
                const __half2 h01 = *reinterpret_cast<const __half2*>(&raw.x);
                const __half2 h23 = *reinterpret_cast<const __half2*>(&raw.y);
                const float2 f01 = __half22float2(h01);
                const float2 f23 = __half22float2(h23);
                v0 = f01.x;
                v1 = f01.y;
                v2 = f23.x;
                v3 = f23.y;
            } else {
                v0 = __half2float(sp[0]);
                v1 = __half2float(sp[1]);
                v2 = __half2float(sp[2]);
                v3 = __half2float(sp[3]);
            }
        }

        // Stage h=1, 2 in registers (H4 butterfly)
        {
            const float a0 = v0 + v1;
            const float a1 = v0 - v1;
            const float a2 = v2 + v3;
            const float a3 = v2 - v3;
            v0 = a0 + a2;
            v1 = a1 + a3;
            v2 = a0 - a2;
            v3 = a1 - a3;
        }

        // Stage h=4..B/2 across subgroup lanes
        #pragma unroll
        for (int h = 1; h < lanes; h *= 2) {
            const float o0 = __shfl_xor_sync(0xffffffffu, v0, h, lanes);
            const float o1 = __shfl_xor_sync(0xffffffffu, v1, h, lanes);
            const float o2 = __shfl_xor_sync(0xffffffffu, v2, h, lanes);
            const float o3 = __shfl_xor_sync(0xffffffffu, v3, h, lanes);
            const bool sub = (lane & h) != 0;
            v0 = sub ? (o0 - v0) : (v0 + o0);
            v1 = sub ? (o1 - v1) : (v1 + o1);
            v2 = sub ? (o2 - v2) : (v2 + o2);
            v3 = sub ? (o3 - v3) : (v3 + o3);
        }

        __half* dp = dst + row * padded_k + col;
        const __half2 out01 = __floats2half2_rn(v0 * scale, v1 * scale);
        const __half2 out23 = __floats2half2_rn(v2 * scale, v3 * scale);
        if (vec_aligned) {
            uint2 packed_out;
            packed_out.x = *reinterpret_cast<const uint32_t*>(&out01);
            packed_out.y = *reinterpret_cast<const uint32_t*>(&out23);
            *reinterpret_cast<uint2*>(dp) = packed_out;
        } else {
            dp[0] = __low2half(out01);
            dp[1] = __high2half(out01);
            dp[2] = __low2half(out23);
            dp[3] = __high2half(out23);
        }
    }
}

} // namespace

extern "C" int w34_block_hadamard_f16(const w34_exec* ex, const void* src,
                                      void* dst, int rows, int k, int padded_k,
                                      int block_size) {
    if (!ex || rows < 0 || k <= 0 || padded_k < k ||
        (block_size != 16 && block_size != 64) ||
        k % block_size != 0 || padded_k % block_size != 0) {
        return int(cudaErrorInvalidValue);
    }
    if (rows == 0) return int(cudaSuccess);
    const uintptr_t s = reinterpret_cast<uintptr_t>(src);
    const uintptr_t d = reinterpret_cast<uintptr_t>(dst);
    const uint64_t src_bytes = uint64_t(rows) * k * sizeof(__half);
    const uint64_t dst_bytes = uint64_t(rows) * padded_k * sizeof(__half);
    // Reject overlapping ranges, including partial aliasing: callers must not
    // destroy shared inputs. Validate arithmetic before comparing end pointers.
    if (!s || !d || s % alignof(__half) || d % alignof(__half) ||
        s > UINTPTR_MAX - src_bytes || d > UINTPTR_MAX - dst_bytes ||
        (s < d + dst_bytes && d < s + src_bytes)) {
        return int(cudaErrorInvalidValue);
    }
    const int64_t groups = int64_t(rows) * (padded_k / block_size);
    const int groups_per_cta = 256 / (block_size / 4);
    const int64_t needed = (groups + groups_per_cta - 1) / groups_per_cta;
    const int grid = int(needed < 65535 ? needed : 65535);
    const auto stream = reinterpret_cast<cudaStream_t>(ex->stream);
    if (block_size == 16) {
        block_hadamard_f16<16><<<grid, 256, 0, stream>>>(
            static_cast<const __half*>(src), static_cast<__half*>(dst), k, padded_k, groups);
    } else {
        block_hadamard_f16<64><<<grid, 256, 0, stream>>>(
            static_cast<const __half*>(src), static_cast<__half*>(dst), k, padded_k, groups);
    }
    return int(cudaGetLastError());
}

namespace {
__global__ void linear_epilogue_f16(const __half* src, const __half* bias,
                                   __half* dst, int logical_n, int physical_n,
                                   int64_t count) {
    for (int64_t i = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
         i < count; i += int64_t(gridDim.x) * blockDim.x) {
        const int col = int(i % logical_n);
        const __half value = src[(i / logical_n) * physical_n + col];
        dst[i] = bias ? __float2half_rn(__half2float(value) + __half2float(bias[col])) : value;
    }
}
}

extern "C" int w34_linear_epilogue_f16(const w34_exec* ex, const void* src,
                                       const void* bias, void* dst, int rows,
                                       int logical_n, int physical_n) {
    if (!ex || rows < 0 || logical_n <= 0 || physical_n < logical_n)
        return int(cudaErrorInvalidValue);
    if (rows == 0) return int(cudaSuccess);
    const uintptr_t s = reinterpret_cast<uintptr_t>(src);
    const uintptr_t d = reinterpret_cast<uintptr_t>(dst);
    const uintptr_t b = reinterpret_cast<uintptr_t>(bias);
    const uint64_t sb = uint64_t(rows) * physical_n * sizeof(__half);
    const uint64_t db = uint64_t(rows) * logical_n * sizeof(__half);
    const uint64_t bb = uint64_t(logical_n) * sizeof(__half);
    if (!s || !d || s % 2 || d % 2 || b % 2 ||
        s > UINTPTR_MAX - sb || d > UINTPTR_MAX - db || b > UINTPTR_MAX - bb ||
        ((s < d + db && d < s + sb) && !(s == d && logical_n == physical_n)) ||
        (b && b < d + db && d < b + bb))
        return int(cudaErrorInvalidValue);
    const int64_t count = int64_t(rows) * logical_n;
    const int64_t needed = (count + 255) / 256;
    const int grid = int(needed < 65535 ? needed : 65535);
    linear_epilogue_f16<<<grid, 256, 0, reinterpret_cast<cudaStream_t>(ex->stream)>>>(
        static_cast<const __half*>(src), static_cast<const __half*>(bias),
        static_cast<__half*>(dst), logical_n, physical_n, count);
    return int(cudaGetLastError());
}
