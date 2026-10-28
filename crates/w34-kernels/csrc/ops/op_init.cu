// Runtime, Stream, Event, Graph, and cuBLAS wrappers for warp-34.
#include "../w34_cuda.h"

#include <cublas_v2.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

extern "C" int w34_cublas_runtime_version(int* out_version) {
    if (out_version == nullptr) {
        return (int)cudaErrorInvalidValue;
    }
    cublasHandle_t handle = nullptr;
    cublasStatus_t st = cublasCreate_v2(&handle);
    if (st != CUBLAS_STATUS_SUCCESS) {
        return (int)st;
    }
    int ver = 0;
    st = cublasGetVersion_v2(handle, &ver);
    cublasDestroy_v2(handle);
    if (st != CUBLAS_STATUS_SUCCESS) {
        return (int)st;
    }
    *out_version = ver;
    return 0;
}

extern "C" int w34_get_device_count(int* out_count) {
    if (out_count == nullptr) return (int)cudaErrorInvalidValue;
    return (int)cudaGetDeviceCount(out_count);
}

extern "C" int w34_set_device(int device_id) {
    return (int)cudaSetDevice(device_id);
}

extern "C" int w34_device_synchronize(void) {
    return (int)cudaDeviceSynchronize();
}

extern "C" int w34_get_mem_info(size_t* out_free, size_t* out_total) {
    if (out_free == nullptr || out_total == nullptr) return (int)cudaErrorInvalidValue;
    return (int)cudaMemGetInfo(out_free, out_total);
}

extern "C" int w34_malloc(void** out_ptr, size_t bytes) {
    if (out_ptr == nullptr) return (int)cudaErrorInvalidValue;
    return (int)cudaMalloc(out_ptr, bytes);
}

extern "C" int w34_free(void* ptr) {
    if (ptr == nullptr) return 0;
    return (int)cudaFree(ptr);
}

extern "C" int w34_host_alloc_mapped(void** out_host_ptr, void** out_dev_ptr, size_t bytes) {
    if (out_host_ptr == nullptr || out_dev_ptr == nullptr) return (int)cudaErrorInvalidValue;
    cudaError_t err = cudaHostAlloc(out_host_ptr, bytes, cudaHostAllocMapped | cudaHostAllocPortable);
    if (err != cudaSuccess) return (int)err;
    err = cudaHostGetDevicePointer(out_dev_ptr, *out_host_ptr, 0);
    if (err != cudaSuccess) {
        cudaFreeHost(*out_host_ptr);
        *out_host_ptr = nullptr;
        return (int)err;
    }
    return 0;
}

extern "C" int w34_host_free(void* host_ptr) {
    if (host_ptr == nullptr) return 0;
    return (int)cudaFreeHost(host_ptr);
}

extern "C" int w34_memcpy_h2d_async(void* dst, const void* src, size_t bytes, uintptr_t stream) {
    return (int)cudaMemcpyAsync(dst, src, bytes, cudaMemcpyHostToDevice, (cudaStream_t)stream);
}

extern "C" int w34_memcpy_d2h_async(void* dst, const void* src, size_t bytes, uintptr_t stream) {
    return (int)cudaMemcpyAsync(dst, src, bytes, cudaMemcpyDeviceToHost, (cudaStream_t)stream);
}

extern "C" int w34_memcpy_d2d_async(void* dst, const void* src, size_t bytes, uintptr_t stream) {
    return (int)cudaMemcpyAsync(dst, src, bytes, cudaMemcpyDeviceToDevice, (cudaStream_t)stream);
}

extern "C" int w34_memset_async(void* dst, int value, size_t bytes, uintptr_t stream) {
    return (int)cudaMemsetAsync(dst, value, bytes, (cudaStream_t)stream);
}

extern "C" int w34_stream_create(uintptr_t* out_stream) {
    if (out_stream == nullptr) return (int)cudaErrorInvalidValue;
    cudaStream_t s = nullptr;
    cudaError_t err = cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking);
    if (err != cudaSuccess) return (int)err;
    *out_stream = (uintptr_t)s;
    return 0;
}

extern "C" int w34_stream_destroy(uintptr_t stream) {
    if (stream == 0) return 0;
    return (int)cudaStreamDestroy((cudaStream_t)stream);
}

extern "C" int w34_stream_synchronize(uintptr_t stream) {
    return (int)cudaStreamSynchronize((cudaStream_t)stream);
}

extern "C" int w34_event_create(uintptr_t* out_event, int enable_timing) {
    if (out_event == nullptr) return (int)cudaErrorInvalidValue;
    cudaEvent_t ev = nullptr;
    unsigned flags = enable_timing ? cudaEventDefault : cudaEventDisableTiming;
    cudaError_t err = cudaEventCreateWithFlags(&ev, flags);
    if (err != cudaSuccess) return (int)err;
    *out_event = (uintptr_t)ev;
    return 0;
}

extern "C" int w34_event_destroy(uintptr_t event) {
    if (event == 0) return 0;
    return (int)cudaEventDestroy((cudaEvent_t)event);
}

extern "C" int w34_event_record(uintptr_t event, uintptr_t stream) {
    return (int)cudaEventRecord((cudaEvent_t)event, (cudaStream_t)stream);
}

extern "C" int w34_stream_wait_event(uintptr_t stream, uintptr_t event) {
    return (int)cudaStreamWaitEvent((cudaStream_t)stream, (cudaEvent_t)event, 0);
}

extern "C" int w34_event_synchronize(uintptr_t event) {
    return (int)cudaEventSynchronize((cudaEvent_t)event);
}

extern "C" int w34_event_elapsed_ms(float* out_ms, uintptr_t start_event, uintptr_t end_event) {
    if (out_ms == nullptr) return (int)cudaErrorInvalidValue;
    return (int)cudaEventElapsedTime(out_ms, (cudaEvent_t)start_event, (cudaEvent_t)end_event);
}

extern "C" int w34_graph_begin_capture(uintptr_t stream) {
    return (int)cudaStreamBeginCapture((cudaStream_t)stream, cudaStreamCaptureModeThreadLocal);
}

extern "C" int w34_graph_end_capture(uintptr_t stream, uintptr_t* out_graph, uintptr_t* out_graph_exec) {
    if (out_graph == nullptr || out_graph_exec == nullptr) return (int)cudaErrorInvalidValue;
    cudaGraph_t g = nullptr;
    cudaError_t err = cudaStreamEndCapture((cudaStream_t)stream, &g);
    if (err != cudaSuccess) return (int)err;
    cudaGraphExec_t ge = nullptr;
    err = cudaGraphInstantiate(&ge, g, 0);
    if (err != cudaSuccess) {
        cudaGraphDestroy(g);
        return (int)err;
    }
    *out_graph = (uintptr_t)g;
    *out_graph_exec = (uintptr_t)ge;
    return 0;
}

extern "C" int w34_graph_launch(uintptr_t graph_exec, uintptr_t stream) {
    return (int)cudaGraphLaunch((cudaGraphExec_t)graph_exec, (cudaStream_t)stream);
}

extern "C" int w34_graph_destroy(uintptr_t graph, uintptr_t graph_exec) {
    if (graph_exec != 0) cudaGraphExecDestroy((cudaGraphExec_t)graph_exec);
    if (graph != 0) cudaGraphDestroy((cudaGraph_t)graph);
    return 0;
}

extern "C" int w34_cublas_create(uintptr_t* out_handle, uintptr_t stream) {
    if (out_handle == nullptr) return (int)cudaErrorInvalidValue;
    cublasHandle_t h = nullptr;
    cublasStatus_t st = cublasCreate_v2(&h);
    if (st != CUBLAS_STATUS_SUCCESS) return (int)st;
    st = cublasSetStream_v2(h, (cudaStream_t)stream);
    if (st != CUBLAS_STATUS_SUCCESS) {
        cublasDestroy_v2(h);
        return (int)st;
    }
    *out_handle = (uintptr_t)h;
    return 0;
}

extern "C" int w34_cublas_destroy(uintptr_t handle) {
    if (handle == 0) return 0;
    return (int)cublasDestroy_v2((cublasHandle_t)handle);
}

// Row-major C [M, N] = A [M, K] @ B^T [N, K].
// In column-major cuBLAS terms: C^T [N, M] = B [K, N]^T @ A^T [K, M].
extern "C" int w34_cublas_gemm_f16(uintptr_t handle, const void* a, const void* b, void* c,
                                   int m, int n, int k) {
    if (handle == 0 || a == nullptr || b == nullptr || c == nullptr) return (int)cudaErrorInvalidValue;
    const float alpha = 1.0f;
    const float beta = 0.0f;
    cublasStatus_t st = cublasGemmEx(
        (cublasHandle_t)handle,
        CUBLAS_OP_T, CUBLAS_OP_N,
        n, m, k,
        &alpha,
        b, CUDA_R_16F, k,
        a, CUDA_R_16F, k,
        &beta,
        c, CUDA_R_16F, n,
        CUBLAS_COMPUTE_32F,
        CUBLAS_GEMM_DEFAULT_TENSOR_OP);
    return (int)st;
}

extern "C" int w34_cublas_gemm_bf16(uintptr_t handle, const void* a, const void* b, void* c,
                                    int m, int n, int k) {
    if (handle == 0 || a == nullptr || b == nullptr || c == nullptr) return (int)cudaErrorInvalidValue;
    const float alpha = 1.0f;
    const float beta = 0.0f;
    cublasStatus_t st = cublasGemmEx(
        (cublasHandle_t)handle,
        CUBLAS_OP_T, CUBLAS_OP_N,
        n, m, k,
        &alpha,
        b, CUDA_R_16BF, k,
        a, CUDA_R_16BF, k,
        &beta,
        c, CUDA_R_16BF, n,
        CUBLAS_COMPUTE_32F,
        CUBLAS_GEMM_DEFAULT_TENSOR_OP);
    return (int)st;
}
