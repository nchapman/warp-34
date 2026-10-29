//! Host runtime, loader, cache, model, and CLI for `warp-34`.

pub mod cache;
pub mod chatml;
pub mod constraints;
pub mod cuda;
#[cfg(feature = "cuda")]
pub mod dflash;
pub mod loader;
#[cfg(feature = "cuda")]
pub mod model;
#[cfg(feature = "cuda")]
pub mod mtp;
pub mod parser;
#[cfg(feature = "cuda")]
pub mod proposer;
pub mod radix;
mod remote_images;
pub mod scheduler;
pub mod server;
pub mod speculative;
pub mod tool_constraints;
pub mod vision;

/// Returns whether the crate was compiled with the `cuda` feature enabled.
pub const fn cuda_enabled() -> bool {
    w34_kernels::cuda_enabled()
}
