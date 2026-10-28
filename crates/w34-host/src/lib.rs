//! `w34-host`: Rust host runtime for `warp-34`.

pub use w34_kernels::ffi;

#[cfg(test)]
mod tests {
    use super::ffi::W34Exec;

    #[test]
    fn exec_abi_size_is_24_bytes() {
        assert_eq!(core::mem::size_of::<W34Exec>(), 24);
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn cuda_seam_links_and_queries_cublas_version() {
        let ver = super::ffi::cublas_runtime_version().expect("cublasGetVersion_v2");
        assert!(ver >= 120000, "unexpected cuBLAS version {ver}");
    }
}
