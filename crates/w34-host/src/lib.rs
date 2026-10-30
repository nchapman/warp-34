//! Host runtime, loader, cache, model, and CLI for `warp-34`.

pub mod cache;
pub mod chatml;
pub mod constraints;
#[cfg(feature = "cuda")]
mod coordinated;
pub mod cuda;
#[cfg(feature = "cuda")]
pub mod dflash;
#[cfg(feature = "cuda")]
pub mod engine_worker;
pub mod loader;
#[cfg(feature = "cuda")]
pub mod model;
#[cfg(feature = "cuda")]
pub mod mtp;
pub mod parser;
pub mod profiling;
#[cfg(feature = "cuda")]
pub mod proposer;
pub mod radix;
mod remote_images;
pub mod request_constraints;
#[cfg(feature = "cuda")]
mod scheduled_vision;
pub mod scheduler;
pub mod server;
pub mod speculative;
pub mod tool_constraints;
pub mod vision;

/// Returns whether the crate was compiled with the `cuda` feature enabled.
pub const fn cuda_enabled() -> bool {
    w34_kernels::cuda_enabled()
}
