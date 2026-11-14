//! `w34-kernels`: CUDA kernel library and FFI seam for `warp-34`.

pub mod ffi;

/// Returns whether `w34-kernels` was compiled with the `cuda` feature enabled.
///
/// # Examples
///
/// ```
/// let _ = w34_kernels::cuda_enabled();
/// ```
pub const fn cuda_enabled() -> bool {
    cfg!(feature = "cuda")
}

#[cfg(test)]
mod tests {
    use super::*;
    use core::mem::{align_of, size_of};
    use ffi::*;

    #[test]
    fn test_feature_flag_matches_cfg() {
        assert_eq!(cuda_enabled(), cfg!(feature = "cuda"));
    }

    #[test]
    fn test_ffi_struct_sizes_and_alignments() {
        assert_eq!(size_of::<W34Exec>(), 24);
        assert_eq!(align_of::<W34Exec>(), 8);

        assert_eq!(size_of::<W34KvLayerLayout>(), 104);
        assert_eq!(align_of::<W34KvLayerLayout>(), 8);

        assert_eq!(size_of::<W34AttentionBatch>(), 96);
        assert_eq!(align_of::<W34AttentionBatch>(), 8);

        assert_eq!(size_of::<W34GdnBatch>(), 80);
        assert_eq!(align_of::<W34GdnBatch>(), 8);

        assert_eq!(size_of::<W34DeviceProps>(), 96);
        assert_eq!(align_of::<W34DeviceProps>(), 4);

        assert_eq!(size_of::<W34SamplerRowCfg>(), 64);
        assert_eq!(align_of::<W34SamplerRowCfg>(), 4);

        assert_eq!(size_of::<W34SamplerSparseEntry>(), 16);
        assert_eq!(align_of::<W34SamplerSparseEntry>(), 4);

        assert_eq!(size_of::<W34SamplerRowOut>(), 184);
        assert_eq!(align_of::<W34SamplerRowOut>(), 4);
    }

    #[test]
    fn test_ffi_constants_contracts() {
        assert_eq!(W34_DTYPE_F16, 0);
        assert_eq!(W34_DTYPE_BF16, 1);
        assert_eq!(W34_DTYPE_F32, 2);
        assert_eq!(W34_DTYPE_I32, 3);
        assert_eq!(W34_DTYPE_U8, 4);
        assert_eq!(W34_DTYPE_U32, 5);

        assert_eq!(W34_KV_NAMESPACE_TARGET_FULL_ATTN, 0);
        assert_eq!(W34_KV_NAMESPACE_MTP_FULL_ATTN, 1);
        assert_eq!(W34_KV_NAMESPACE_DFLASH_DRAFT_CTX, 2);

        assert_eq!(W34_KV_FORMAT_FP16, 0);
        assert_eq!(W34_KV_FORMAT_BF16, 1);
        assert_eq!(W34_KV_FORMAT_TQ4, 2);
        assert_eq!(W34_KV_FORMAT_Q8_0, 3);
        assert_eq!(W34_KV_FORMAT_FP8_E4M3, 4);

        assert_eq!(W34_KV_TRANSFORM_IDENTITY, 0);
        assert_eq!(W34_KV_TRANSFORM_RADEMACHER_WHT256, 1);

        assert_eq!(W34_ATTN_MASK_CAUSAL, 0);
        assert_eq!(W34_ATTN_MASK_NONCAUSAL_WINDOW, 1);
        assert_eq!(W34_ATTN_MASK_SYMMETRIC_WINDOW, 1);

        assert_eq!(W34_MAX_TOP_LOGPROBS, 20);
    }

    #[test]
    fn test_ffi_struct_defaults() {
        let exec = W34Exec::default();
        assert_eq!(exec.stream, 0);
        assert_eq!(exec.scratch_ptr, 0);
        assert_eq!(exec.scratch_size, 0);

        let layout = W34KvLayerLayout::default();
        assert_eq!(layout.k_base, 0);
        assert_eq!(layout.v_base, 0);
        assert_eq!(layout.namespace_id, 0);
        assert_eq!(layout.layer_id, 0);

        let sampler_cfg = W34SamplerRowCfg::default();
        assert_eq!(sampler_cfg.temperature, 0.0);
        assert_eq!(sampler_cfg.top_k, 0);
        assert_eq!(sampler_cfg.draft_token_id, 0);

        let sampler_out = W34SamplerRowOut::default();
        assert_eq!(sampler_out.sampled_id, -1);
        assert_eq!(sampler_out.status, 0);
        assert_eq!(sampler_out.draft_accepted, 0);
        assert!(sampler_out.top_ids.iter().all(|&id| id == -1));
        assert!(
            sampler_out
                .top_logprobs
                .iter()
                .all(|&lp| lp == f32::NEG_INFINITY)
        );

        let sparse = W34SamplerSparseEntry::default();
        assert_eq!(sparse.token_id, 0);
        assert_eq!(sparse.output_count, 0);
        assert_eq!(sparse.banned, 0);

        let props = W34DeviceProps::default();
        assert_eq!(props.device_id, 0);
        assert_eq!(props.compute_major, 0);
        assert_eq!(props.multi_processor_count, 0);
        assert!(props.name.iter().all(|&b| b == 0));
    }
}
