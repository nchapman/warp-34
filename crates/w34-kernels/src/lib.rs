//! `w34-kernels`: CUDA kernel library and FFI seam for `warp-34`.

pub mod ffi;

/// Returns whether `w34-kernels` was compiled with the `cuda` feature enabled.
pub const fn cuda_enabled() -> bool {
    cfg!(feature = "cuda")
}

// Kept here alongside the hand-written FFI module to expose the biased linear ABI.
#[cfg(feature = "cuda")]
unsafe extern "C" {
    pub fn w34_cublas_gemm_f16_bias(
        handle: usize,
        a: *const core::ffi::c_void,
        b: *const core::ffi::c_void,
        bias: *const core::ffi::c_void,
        c: *mut core::ffi::c_void,
        m: i32,
        n: i32,
        k: i32,
    ) -> i32;
}
