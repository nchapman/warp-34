#include "../w34_cuda.h"
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cstdint>

namespace {

// Each 16-lane subgroup owns H16; each warp owns H64 (two values/lane).
// FP32 butterflies avoid intermediate FP16 overflow. Only the normalized
// result is rounded to FP16. The input remains available to unrotated branches.
template <int B>
__global__ void block_hadamard_f16(const __half* src, __half* dst,
                                  int k, int padded_k, int64_t groups) {
    constexpr int lanes = B == 16 ? 16 : 32;
    const int lane = threadIdx.x % lanes;
    const int64_t first = (int64_t(blockIdx.x) * blockDim.x + threadIdx.x) / lanes;
    const int64_t step = int64_t(gridDim.x) * blockDim.x / lanes;
    const int blocks_per_row = padded_k / B;
    for (int64_t group = first; group < groups; group += step) {
        const int64_t row = group / blocks_per_row;
        const int col = int(group % blocks_per_row) * B + lane;
        float lo = col < k ? __half2float(src[row * k + col]) : 0.0f;
        float hi = 0.0f;
        if constexpr (B == 64) {
            hi = col < k ? __half2float(src[row * k + col + 32]) : 0.0f;
        }
        const unsigned mask = __activemask();
        #pragma unroll
        for (int h = 1; h < lanes; h *= 2) {
            const float other = __shfl_xor_sync(mask, lo, h, lanes);
            lo = (lane & h) ? other - lo : lo + other;
            if constexpr (B == 64) {
                const float other_hi = __shfl_xor_sync(mask, hi, h, lanes);
                hi = (lane & h) ? other_hi - hi : hi + other_hi;
            }
        }
        if constexpr (B == 64) {
            dst[row * padded_k + col] = __float2half_rn((lo + hi) * 0.125f);
            dst[row * padded_k + col + 32] = __float2half_rn((lo - hi) * 0.125f);
        } else {
            dst[row * padded_k + col] = __float2half_rn(lo * 0.25f);
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
    const int groups_per_cta = block_size == 16 ? 16 : 8;
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
