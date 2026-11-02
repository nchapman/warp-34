// Runtime, Stream, Event, Graph, and cuBLAS wrappers for warp-34.
#include "../w34_cuda.h"

#include <cublas_v2.h>
#include <cublasLt.h>
#include <cstring>
#include <mutex>
#include <new>

namespace {
constexpr size_t vision_workspace_bytes = 1024 * 1024;
constexpr int vision_plan_limit = 256;
struct VisionPlan {
    int m = 0, n = 0, k = 0;
    const void* bias = nullptr;
    uint32_t alignment[4] = {};
    cublasLtMatmulDesc_t desc = nullptr;
    cublasLtMatrixLayout_t a = nullptr, b = nullptr, c = nullptr;
    cublasLtMatmulPreference_t pref = nullptr;
    cublasLtMatmulHeuristicResult_t result = {};
    ~VisionPlan() {
        if (pref) cublasLtMatmulPreferenceDestroy(pref);
        if (a) cublasLtMatrixLayoutDestroy(a);
        if (b) cublasLtMatrixLayoutDestroy(b);
        if (c) cublasLtMatrixLayoutDestroy(c);
        if (desc) cublasLtMatmulDescDestroy(desc);
    }
};
// Opaque ABI, not a raw vendor handle. One immutable stream/device per owner.
// Cached host descriptors may be replaced after submission; graph nodes retain
// copied launch parameters, not these descriptors. Workspace is never evicted.
// The owner and tensors must outlive ALL graph replays, including other streams;
// concurrent replays sharing this workspace are not supported. Destroy only
// after graphs are retired. Host submissions are serialized; device submissions
// use the fixed stream. No global cache or dependency on vendor handle casts.
struct W34Blas {
    cublasHandle_t blas = nullptr;
    cublasLtHandle_t lt = nullptr;
    cudaStream_t stream = nullptr;
    int device = 0, count = 0, next_evict = 0;
    void* workspace = nullptr;
    VisionPlan* plans[vision_plan_limit] = {};
    std::mutex mutex;
};
uint32_t vision_alignment(const void* p) {
    uint32_t alignment = 256;
    while ((uintptr_t)p % alignment) alignment /= 2;
    return alignment;
}
} // namespace
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

extern "C" const char* w34_cuda_error_string(int code) {
    if (code < 0) return nullptr;
    const char* msg = cudaGetErrorString((cudaError_t)code);
    if (msg == nullptr || std::strcmp(msg, "unrecognized error code") == 0) {
        return nullptr;
    }
    return msg;
}

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

extern "C" int w34_vision_fa2_prepare();
extern "C" int w34_paged_fa2_prepare();
extern "C" int w34_stream_create(uintptr_t* out_stream) {
    if (out_stream == nullptr) return (int)cudaErrorInvalidValue;
    // Eagerly load and opt in vision and paged FA2 variants before captures.
    int prepare_rc = w34_vision_fa2_prepare();
    if (prepare_rc) return prepare_rc;
    prepare_rc = w34_paged_fa2_prepare();
    if (prepare_rc) return prepare_rc;
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
    auto owner = new (std::nothrow) W34Blas;
    if (!owner) { cublasDestroy_v2(h); return (int)cudaErrorMemoryAllocation; }
    cudaError_t err = cudaGetDevice(&owner->device);
    if (err != cudaSuccess) { delete owner; cublasDestroy_v2(h); return (int)err; }
    owner->blas = h;
    owner->stream = (cudaStream_t)stream;
    *out_handle = (uintptr_t)owner;
    return 0;
}

extern "C" int w34_cublas_destroy(uintptr_t handle) {
    if (handle == 0) return 0;
    auto owner = (W34Blas*)handle;
    int previous;
    cudaError_t err = cudaGetDevice(&previous);
    if (err != cudaSuccess) return (int)err;
    err = cudaSetDevice(owner->device);
    if (err != cudaSuccess) return (int)err;
    // The Rust owner's stream may already have been destroyed during teardown;
    // never dereference that stored stream here. Capture and resource teardown
    // must be isolated, and all graphs must be retired before destroying owners.
    int rc = (int)cudaDeviceSynchronize();
    for (int i = 0; i < owner->count; ++i) delete owner->plans[i];
    if (owner->workspace) { int s = (int)cudaFree(owner->workspace); if (!rc) rc = s; }
    if (owner->lt) { int s = (int)cublasLtDestroy(owner->lt); if (!rc) rc = s; }
    int s = (int)cublasDestroy_v2(owner->blas);
    if (!rc) rc = s;
    delete owner;
    err = cudaSetDevice(previous);
    return rc ? rc : (int)err;
}

// Row-major C [M, N] = A [M, K] @ B^T [N, K].
// In column-major cuBLAS terms: C^T [N, M] = B [K, N]^T @ A^T [K, M].
extern "C" int w34_cublas_gemm_f16(uintptr_t handle, const void* a, const void* b, void* c,
                                   int m, int n, int k) {
    if (handle == 0 || a == nullptr || b == nullptr || c == nullptr) return (int)cudaErrorInvalidValue;
    std::lock_guard<std::mutex> lock(((W34Blas*)handle)->mutex);
    const float alpha = 1.0f;
    const float beta = 0.0f;
    cublasStatus_t st = cublasGemmEx(
        ((W34Blas*)handle)->blas,
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

// Pinned native gemm_and_bias policy: FP32 scalars/compute, FP16 data,
// default reduction preference, one heuristic and a 1 MiB workspace limit.
static int prepare_vision_plan(W34Blas* owner, VisionPlan* p) {
#define LT_CHECK(expr) do { int rc = (int)(expr); if (rc) return rc; } while (0)
    LT_CHECK(cublasLtMatmulDescCreate(&p->desc, CUBLAS_COMPUTE_32F, CUDA_R_32F));
    cublasOperation_t ta = CUBLAS_OP_T, tb = CUBLAS_OP_N;
    LT_CHECK(cublasLtMatmulDescSetAttribute(p->desc, CUBLASLT_MATMUL_DESC_TRANSA, &ta, sizeof(ta)));
    LT_CHECK(cublasLtMatmulDescSetAttribute(p->desc, CUBLASLT_MATMUL_DESC_TRANSB, &tb, sizeof(tb)));
    cublasLtEpilogue_t ep = CUBLASLT_EPILOGUE_BIAS;
    LT_CHECK(cublasLtMatmulDescSetAttribute(p->desc, CUBLASLT_MATMUL_DESC_EPILOGUE, &ep, sizeof(ep)));
    LT_CHECK(cublasLtMatmulDescSetAttribute(p->desc, CUBLASLT_MATMUL_DESC_BIAS_POINTER, &p->bias, sizeof(p->bias)));
    LT_CHECK(cublasLtMatrixLayoutCreate(&p->a, CUDA_R_16F, p->k, p->n, p->k));
    LT_CHECK(cublasLtMatrixLayoutCreate(&p->b, CUDA_R_16F, p->k, p->m, p->k));
    LT_CHECK(cublasLtMatrixLayoutCreate(&p->c, CUDA_R_16F, p->n, p->m, p->n));
    LT_CHECK(cublasLtMatmulPreferenceCreate(&p->pref));
    LT_CHECK(cublasLtMatmulPreferenceSetAttribute(p->pref, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &vision_workspace_bytes, sizeof(vision_workspace_bytes)));
    const cublasLtMatmulPreferenceAttributes_t attrs[] = {
        CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_A_BYTES, CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_B_BYTES,
        CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_C_BYTES, CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_D_BYTES};
    // Native intentionally uses bias alignment for the D preference.
    for (int i = 0; i < 4; ++i)
        LT_CHECK(cublasLtMatmulPreferenceSetAttribute(p->pref, attrs[i], &p->alignment[i], sizeof(uint32_t)));
    int count = 0;
    LT_CHECK(cublasLtMatmulAlgoGetHeuristic(owner->lt, p->desc, p->a, p->b, p->c, p->c, p->pref, 1, &p->result, &count));
    return count == 1 ? (int)p->result.state : (int)CUBLAS_STATUS_NOT_SUPPORTED;
#undef LT_CHECK
}

extern "C" int w34_cublas_gemm_f16_bias(uintptr_t handle, const void* a, const void* b,
                                        const void* bias, void* c, int m, int n, int k) {
    if (!handle || !a || !b || !bias || !c || m <= 0 || n <= 0 || k <= 0)
        return (int)cudaErrorInvalidValue;
    auto owner = (W34Blas*)handle;
    std::lock_guard<std::mutex> lock(owner->mutex);
    int device;
    cudaError_t err = cudaGetDevice(&device);
    if (err != cudaSuccess) return (int)err;
    if (device != owner->device) return (int)cudaErrorInvalidDevice;
    const uint32_t align[] = {vision_alignment(b), vision_alignment(a), vision_alignment(c), vision_alignment(bias)};
    VisionPlan* p = nullptr;
    for (int i = 0; i < owner->count; ++i) {
        auto candidate = owner->plans[i];
        if (candidate->m != m || candidate->n != n || candidate->k != k || candidate->bias != bias) continue;
        bool same = true;
        for (int j = 0; j < 4; ++j) same &= candidate->alignment[j] == align[j];
        if (same) { p = candidate; break; }
    }
    if (!p) {
        cudaStreamCaptureStatus capture;
        err = cudaStreamIsCapturing(owner->stream, &capture);
        if (err != cudaSuccess) return (int)err;
        // Explicit warmup required, including each bias pointer/alignment.
        // Do not allocate, initialize Lt, or query heuristics during capture.
        if (capture != cudaStreamCaptureStatusNone) return (int)cudaErrorStreamCaptureUnsupported;
        if (!owner->lt) {
            int rc = (int)cublasLtCreate(&owner->lt);
            if (rc) return rc;
        }
        if (!owner->workspace) {
            err = cudaMalloc(&owner->workspace, vision_workspace_bytes);
            if (err != cudaSuccess) return (int)err;
        }
        p = new (std::nothrow) VisionPlan;
        if (!p) return (int)cudaErrorMemoryAllocation;
        p->m = m; p->n = n; p->k = k; p->bias = bias;
        for (int i = 0; i < 4; ++i) p->alignment[i] = align[i];
        int rc = prepare_vision_plan(owner, p);
        if (rc) { delete p; return rc; }
        int slot = owner->count;
        if (owner->count < vision_plan_limit) {
            ++owner->count;
        } else {
            // Keep storage bounded without imposing a lifetime limit on the
            // number of image geometries accepted by a serving handle.
            slot = owner->next_evict;
            owner->next_evict = (slot + 1) % vision_plan_limit;
            delete owner->plans[slot];
        }
        owner->plans[slot] = p;
    }
    const float alpha = 1.0f, beta = 0.0f;
    return (int)cublasLtMatmul(owner->lt, p->desc, &alpha, b, p->a, a, p->b,
        &beta, c, p->c, c, p->c, &p->result.algo,
        owner->workspace, vision_workspace_bytes, owner->stream);
}

extern "C" int w34_cublas_gemm_bf16(uintptr_t handle, const void* a, const void* b, void* c,
                                    int m, int n, int k) {
    if (handle == 0 || a == nullptr || b == nullptr || c == nullptr) return (int)cudaErrorInvalidValue;
    std::lock_guard<std::mutex> lock(((W34Blas*)handle)->mutex);
    const float alpha = 1.0f;
    const float beta = 0.0f;
    cublasStatus_t st = cublasGemmEx(
        ((W34Blas*)handle)->blas,
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
