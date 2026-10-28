//! `w34-kernels`: CUDA kernel library and FFI seam for `warp-34`.

pub mod ffi;

/// Returns whether `w34-kernels` was compiled with the `cuda` feature enabled.
pub const fn cuda_enabled() -> bool {
    cfg!(feature = "cuda")
}
