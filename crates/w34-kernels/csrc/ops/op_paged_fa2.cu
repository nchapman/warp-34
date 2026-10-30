// Bounded target-prefill adapter. Metadata retains the cache-layout contract:
// seq_after = context_before + query length; page IDs must address allocated pages.
#include "../w34_cuda.h"
#include "flash_fwd_launch_template.h"
#include <climits>
#include <cmath>
using PagedD256 = Flash_fwd_kernel_traits<256,64,64,4,false,false,cutlass::half_t>;
static_assert(PagedD256::kSmemSize == 98304);
extern "C" int w34_paged_fa2_prepare() {
    return (int)cudaFuncSetAttribute(flash::flash_fwd_splitkv_kernel<PagedD256,true,false,false,false,true,false,false,false>,
        cudaFuncAttributeMaxDynamicSharedMemorySize, PagedD256::kSmemSize);
}
static bool aligned(uint64_t p, unsigned n) { return p && p % n == 0; }
// Tri-state: -1 ineligible, 0 launched, positive CUDA error. Never retry a failure.
extern "C" int w34_paged_fa2_try(const w34_exec* ex, const w34_kv_layer_layout* l,
    const w34_attention_batch* b, const void* q, void* o, int h) {
    if (!ex || !l || !b || l->namespace_id != W34_KV_NAMESPACE_TARGET_FULL_ATTN ||
        l->k_format != W34_KV_FORMAT_FP16 || l->v_format != W34_KV_FORMAT_FP16 ||
        l->k_transform != W34_KV_TRANSFORM_IDENTITY || l->v_transform != W34_KV_TRANSFORM_IDENTITY ||
        l->head_dim != 256 || l->block_size != 16 || l->local_kv_heads <= 0 ||
        h <= 0 || h > 65535 || h % l->local_kv_heads || b->num_reqs != 1 ||
        b->total_tokens <= 16 || b->max_q_len < b->total_tokens ||
        b->max_q_len > INT_MAX - 127 || b->max_seq_len < b->total_tokens || b->max_seq_len > INT_MAX - 127 ||
        b->max_blocks_per_req <= 0 || (int64_t)b->max_seq_len > (int64_t)b->max_blocks_per_req * 16 ||
        b->mask_mode != W34_ATTN_MASK_CAUSAL || b->sliding_window != 0 ||
        b->valid_token_mask || b->valid_req_mask || !std::isfinite(b->sm_scale) ||
        !aligned((uintptr_t)q,16) || !aligned((uintptr_t)o,16) ||
        !aligned(l->k_base,16) || !aligned(l->v_base,16) ||
        !aligned((uintptr_t)b->query_start_loc,4) || !aligned((uintptr_t)b->context_lens_before,4) ||
        !aligned((uintptr_t)b->seq_lens_after,4) || !aligned((uintptr_t)b->block_tables,4) ||
        !aligned(ex->scratch_ptr,16)) return -1;
    // The donor page helper narrows strides to int. Also bound row/head offset sums.
    const uint64_t strides[] = {l->k_page_stride_bytes,l->k_head_stride_bytes,l->k_token_stride_bytes,
        l->v_page_stride_bytes,l->v_head_stride_bytes,l->v_token_stride_bytes};
    for (auto s : strides) if (!s || s % 16 || s / 2 > INT_MAX) return -1;
    if (15 * (l->k_token_stride_bytes / 2) + (l->local_kv_heads - 1ULL) * (l->k_head_stride_bytes / 2) + 256 > INT_MAX ||
        15 * (l->v_token_stride_bytes / 2) + (l->local_kv_heads - 1ULL) * (l->v_head_stride_bytes / 2) + 256 > INT_MAX ||
        (uint64_t)b->max_q_len * h * 256 > INT_MAX) return -1;
    const uint64_t needed = (uint64_t)h * b->max_q_len * sizeof(float);
    if (ex->scratch_size < needed) return -1;
    flash::Flash_fwd_params p{};
    p.q_ptr=const_cast<void*>(q); p.o_ptr=o; p.k_ptr=(void*)l->k_base; p.v_ptr=(void*)l->v_base;
    p.softmax_lse_ptr=(void*)ex->scratch_ptr;
    p.q_row_stride=p.o_row_stride=h*256; p.q_head_stride=p.o_head_stride=256;
    p.k_batch_stride=l->k_page_stride_bytes/2; p.v_batch_stride=l->v_page_stride_bytes/2;
    p.k_row_stride=l->k_token_stride_bytes/2; p.v_row_stride=l->v_token_stride_bytes/2;
    p.k_head_stride=l->k_head_stride_bytes/2; p.v_head_stride=l->v_head_stride_bytes/2;
    p.block_table=const_cast<int*>(b->block_tables); p.block_table_batch_stride=b->max_blocks_per_req; p.page_block_size=16;
    p.cu_seqlens_q=const_cast<int*>(b->query_start_loc); p.seqused_k=const_cast<int*>(b->seq_lens_after);
    p.is_seqlens_k_cumulative=true;
    p.b=1; p.h=h; p.h_k=l->local_kv_heads; p.h_h_k_ratio=h/l->local_kv_heads;
    p.seqlen_q=b->max_q_len; p.total_q=b->total_tokens; p.seqlen_k=b->max_seq_len; p.d=p.d_rounded=256;
    p.seqlen_q_rounded=(p.seqlen_q+127)/128*128; p.seqlen_k_rounded=(p.seqlen_k+127)/128*128;
    p.scale_softmax=b->sm_scale; p.scale_softmax_log2=p.scale_softmax*1.4426950408889634074;
    p.p_dropout=1.f; p.p_dropout_in_uint8_t=255; p.rp_dropout=1.f; p.scale_softmax_rp_dropout=p.scale_softmax;
    p.window_size_left=-1; p.window_size_right=0; p.num_splits=1;
    flash::flash_fwd_splitkv_kernel<PagedD256,true,false,false,false,true,false,false,false>
        <<<dim3((p.seqlen_q+63)/64,1,h),PagedD256::kNThreads,PagedD256::kSmemSize,(cudaStream_t)ex->stream>>>(p);
    return (int)cudaGetLastError();
}
