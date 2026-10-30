// Source-faithful FA2 vision adapter. See vendor/vision-fa2/PROVENANCE.md.
#include "../w34_cuda.h"
#include "flash_fwd_launch_template.h"
#include <climits>

namespace w34_vision_fa2_detail {
using D64 = Flash_fwd_kernel_traits<64,128,128,4,false,false,cutlass::half_t>;
using D72 = Flash_fwd_kernel_traits<96,128,64,4,false,false,cutlass::half_t>;
// Varlen forces even-MN=false, D64 has even-K=true, logical D72 does not.
// Directly use the donor entrypoint, avoiding its error-swallowing stub wrapper.
template<class T, bool EvenK> cudaError_t prepare() {
    return cudaFuncSetAttribute(
        flash::flash_fwd_kernel<T,false,false,false,false,false,EvenK,false,false>,
        cudaFuncAttributeMaxDynamicSharedMemorySize, T::kSmemSize);
}
template<class T, bool EvenK> int launch(flash::Flash_fwd_params& p, cudaStream_t s) {
    flash::flash_fwd_kernel<T,false,false,false,false,false,EvenK,false,false>
        <<<dim3((p.seqlen_q+127)/128,p.b,p.h),T::kNThreads,T::kSmemSize,s>>>(p);
    return (int)cudaGetLastError();
}
}
using namespace w34_vision_fa2_detail;
// Called at stream initialization, outside capture, on each selected device.
extern "C" int w34_vision_fa2_prepare() {
    cudaError_t e = prepare<D64,true>();
    if (e != cudaSuccess) return (int)e;
    return (int)prepare<D72,false>();
}
extern "C" int w34_vision_fa2_validate(const w34_exec* ex, const void* qkv,
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
extern "C" int w34_vision_fa2(const w34_exec* ex, const void* qkv, const int32_t* cu,
                              void* out, int n, int b, int h, int d) {
    int rc = w34_vision_fa2_validate(ex,qkv,out,n,b,h,d);
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
    return d==64 ? launch<D64,true>(p,(cudaStream_t)ex->stream)
                 : launch<D72,false>(p,(cudaStream_t)ex->stream);
}
