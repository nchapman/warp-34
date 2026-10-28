// warp-34 C FFI contract. Mirrored in crates/w34-kernels/src/ffi.rs.
#pragma once
#include <assert.h>
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

enum {
    W34_DTYPE_F16  = 0,
    W34_DTYPE_BF16 = 1,
    W34_DTYPE_F32  = 2,
    W34_DTYPE_I32  = 3,
    W34_DTYPE_U8   = 4,
    W34_DTYPE_U32  = 5,
};

typedef struct {
    uintptr_t stream;       // cudaStream_t
    uint64_t  scratch_ptr;  // device workspace pointer
    size_t    scratch_size; // workspace capacity in bytes
} w34_exec;

static_assert(sizeof(w34_exec) == 24, "w34_exec ABI size mismatch");

// Returns linked cuBLAS runtime version (major*10000 + minor*100 + patch) in *out_version,
// and 0 on success or a CUDA/cuBLAS error code on failure.
int w34_cublas_runtime_version(int* out_version);

#ifdef __cplusplus
}
#endif
