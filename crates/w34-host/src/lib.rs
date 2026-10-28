//! Host runtime, loader, cache, model, and CLI for `warp-34`.

pub mod cache;
pub mod cuda;
pub mod loader;
#[cfg(feature = "cuda")]
pub mod model;

/// Returns whether the crate was compiled with the `cuda` feature enabled.
pub const fn cuda_enabled() -> bool {
    w34_kernels::cuda_enabled()
}
