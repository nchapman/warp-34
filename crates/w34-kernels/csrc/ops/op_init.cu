#include "../w34_cuda.h"
#include <cublas_v2.h>
#include <cuda_runtime.h>

extern "C" int w34_cublas_runtime_version(int* out_version) {
    if (out_version == nullptr) {
        return (int)cudaErrorInvalidValue;
    }
    cublasStatus_t st = cublasGetVersion_v2(nullptr, out_version);
    return (int)st;
}
