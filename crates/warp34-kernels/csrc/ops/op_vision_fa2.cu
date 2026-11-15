// Source-faithful FA2 vision adapter. See vendor/vision-fa2/PROVENANCE.md.
#include "../warp34_cuda.h"
#include "flash_fwd_launch_template.h"
#include <climits>

namespace warp34_vision_fa2_detail {
using D64 = Flash_fwd_kernel_traits<64,128,128,4,false,false,cutlass::half_t>;
using D72 = Flash_fwd_kernel_traits<96,128,64,4,false,false,cutlass::half_t>;
using D128Causal = Flash_fwd_kernel_traits<128,64,64,4,false,false,cutlass::half_t>;
using D128Full = Flash_fwd_kernel_traits<128,64,64,4,false,false,cutlass::half_t>;
// Varlen forces even-MN=false, D64/D128 have even-K=true, logical D72 does not.
// Directly use the donor entrypoint, avoiding its error-swallowing stub wrapper.
template<class T, bool IsCausal, bool EvenK> cudaError_t prepare() {
    if constexpr (T::kSmemSize >= 48 * 1024) {
        return cudaFuncSetAttribute(
            flash::flash_fwd_kernel<T,false,IsCausal,false,false,false,EvenK,false,false>,
            cudaFuncAttributeMaxDynamicSharedMemorySize, T::kSmemSize);
    }
    return cudaSuccess;
}
template<class T, bool IsCausal, bool EvenK> int launch(flash::Flash_fwd_params& p, cudaStream_t s) {
    flash::flash_fwd_kernel<T,false,IsCausal,false,false,false,EvenK,false,false>
        <<<dim3((p.seqlen_q+T::kBlockM-1)/T::kBlockM,p.b,p.h),T::kNThreads,T::kSmemSize,s>>>(p);
    return (int)cudaGetLastError();
}
}
using namespace warp34_vision_fa2_detail;
// Called at stream initialization, outside capture, on each selected device.
extern "C" int warp34_vision_fa2_prepare() {
    cudaError_t e = prepare<D64,false,true>();
    if (e != cudaSuccess) return (int)e;
    e = prepare<D72,false,false>();
    if (e != cudaSuccess) return (int)e;
    e = prepare<D128Causal,true,true>();
    if (e != cudaSuccess) return (int)e;
    return (int)prepare<D128Full,false,true>();
}
extern "C" int warp34_vision_fa2_validate(const warp34_exec* ex, const void* qkv,
                                      const void* out, int n, int b, int h, int d) {
    if (!ex || !qkv || !out || n < 0 || n > INT_MAX-128 || b <= 0 || b > 65535 ||
        h <= 0 || h > 65535 || (d != 64 && d != 72) ||
        ((uintptr_t(qkv) | uintptr_t(out)) & 15)) return (int)cudaErrorInvalidValue;
    if (n && (!ex->scratch_ptr || (ex->scratch_ptr & 3) ||
        ex->scratch_size < size_t(n)*h*sizeof(float))) return (int)cudaErrorInvalidValue;
    return 0;
}
// QKV is already bias-added and post-RoPE, packed [N,3,H,D]. cu is device
// int32 [B+1], monotonic, cu[0]=0, cu[B]=N (caller-owned stream lifetime).
extern "C" int warp34_vision_fa2(const warp34_exec* ex, const void* qkv, const int32_t* cu,
                              void* out, int n, int b, int h, int d) {
    int rc = warp34_vision_fa2_validate(ex,qkv,out,n,b,h,d);
    if (rc || !cu) return rc ? rc : (int)cudaErrorInvalidValue;
    if (!n) return 0;
    flash::Flash_fwd_params p{};
    const size_t hidden = size_t(h)*d;
    p.q_ptr=const_cast<void*>(qkv);
    p.k_ptr=(char*)p.q_ptr+hidden*2; p.v_ptr=(char*)p.q_ptr+hidden*4;
    p.o_ptr=out; p.softmax_lse_ptr=(void*)ex->scratch_ptr;
    p.q_row_stride=p.k_row_stride=p.v_row_stride=3*hidden;
    p.o_row_stride=hidden;
    p.q_head_stride=p.k_head_stride=p.v_head_stride=p.o_head_stride=d;
    p.b=b; p.h=p.h_k=h; p.h_h_k_ratio=1;
    // N is a safe upper bound on every image length; extra CTAs return early.
    p.seqlen_q=p.seqlen_k=p.total_q=n; p.d=d;
    p.seqlen_q_rounded=p.seqlen_k_rounded=(n+127)/128*128;
    p.d_rounded=d==64 ? 64 : 96;
    p.cu_seqlens_q=p.cu_seqlens_k=const_cast<int32_t*>(cu);
    p.is_seqlens_k_cumulative=true; p.unpadded_lse=true;
    p.scale_softmax=float(1.0 / std::sqrt(double(d)));
    p.scale_softmax_log2=p.scale_softmax * 1.4426950408889634074;
    p.p_dropout=1.f; p.p_dropout_in_uint8_t=255; p.rp_dropout=1.f;
    p.scale_softmax_rp_dropout=p.scale_softmax;
    p.window_size_left=p.window_size_right=-1; p.num_splits=1;
    return d==64 ? launch<D64,false,true>(p,(cudaStream_t)ex->stream)
                 : launch<D72,false,false>(p,(cudaStream_t)ex->stream);
}

// Contiguous [B, Sq, H, 128] / [B, Sk, Hk, 128] FlashAttention-2 forward pass
// supporting both causal GQA (Qwen3-VL-8B) and bidirectional full/cached-prefix
// attention (QwenImage21Transformer2DModel).
extern "C" int warp34_mha_fa2_d128_f16(
    const warp34_exec* ex,
    const void* q,
    const void* k,
    const void* v,
    void* out,
    int b,
    int seqlen_q,
    int seqlen_k,
    int h,
    int h_k,
    int is_causal) {
    if (!ex || !q || !k || !v || !out || b <= 0 || b > 65535 ||
        seqlen_q < 0 || seqlen_q > INT_MAX - 128 ||
        seqlen_k < 0 || seqlen_k > INT_MAX - 128 ||
        h <= 0 || h > 65535 || h_k <= 0 || h % h_k != 0 ||
        ((uintptr_t(q) | uintptr_t(k) | uintptr_t(v) | uintptr_t(out)) & 15)) {
        return (int)cudaErrorInvalidValue;
    }
    if (seqlen_q == 0 || seqlen_k == 0) return 0;
    const size_t lse_bytes = size_t(b) * size_t(h) * size_t(seqlen_q) * sizeof(float);
    if (!ex->scratch_ptr || (ex->scratch_ptr & 3) || ex->scratch_size < lse_bytes) {
        return (int)cudaErrorInvalidValue;
    }
    flash::Flash_fwd_params p{};
    const size_t q_row = size_t(h) * 128;
    const size_t kv_row = size_t(h_k) * 128;
    p.q_ptr = const_cast<void*>(q);
    p.k_ptr = const_cast<void*>(k);
    p.v_ptr = const_cast<void*>(v);
    p.o_ptr = out;
    p.softmax_lse_ptr = (void*)ex->scratch_ptr;
    p.q_row_stride = q_row;
    p.k_row_stride = kv_row;
    p.v_row_stride = kv_row;
    p.o_row_stride = q_row;
    p.q_head_stride = p.k_head_stride = p.v_head_stride = p.o_head_stride = 128;
    p.q_batch_stride = size_t(seqlen_q) * q_row;
    p.k_batch_stride = size_t(seqlen_k) * kv_row;
    p.v_batch_stride = size_t(seqlen_k) * kv_row;
    p.o_batch_stride = size_t(seqlen_q) * q_row;
    p.b = b;
    p.h = h;
    p.h_k = h_k;
    p.h_h_k_ratio = h / h_k;
    p.seqlen_q = seqlen_q;
    p.seqlen_k = seqlen_k;
    p.total_q = b * seqlen_q;
    p.d = p.d_rounded = 128;
    p.seqlen_q_rounded = (seqlen_q + 127) / 128 * 128;
    p.seqlen_k_rounded = (seqlen_k + 127) / 128 * 128;
    p.cu_seqlens_q = nullptr;
    p.cu_seqlens_k = nullptr;
    p.is_seqlens_k_cumulative = true;
    p.unpadded_lse = false;
    p.scale_softmax = 0.08838834764831845f;
    p.scale_softmax_log2 = p.scale_softmax * 1.4426950408889634074f;
    p.p_dropout = 1.f;
    p.p_dropout_in_uint8_t = 255;
    p.rp_dropout = 1.f;
    p.scale_softmax_rp_dropout = p.scale_softmax;
    p.window_size_left = -1;
    p.window_size_right = is_causal ? 0 : -1;
    p.num_splits = 1;
    return is_causal
        ? launch<D128Causal,true,true>(p, (cudaStream_t)ex->stream)
        : launch<D128Full,false,true>(p, (cudaStream_t)ex->stream);
}
