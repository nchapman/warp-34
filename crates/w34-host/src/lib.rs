//! Host runtime, loader, cache, model, and CLI for `warp-34`.

pub mod anthropic_api;
pub mod api_auth;
pub mod cache;
pub mod chatml;
pub mod clef;
pub mod constraints;
#[cfg(feature = "cuda")]
mod coordinated;
pub mod cuda;
#[cfg(feature = "cuda")]
pub mod dashboard;
pub mod decision;
#[cfg(feature = "cuda")]
pub mod dflash;
#[cfg(feature = "cuda")]
pub mod engine_worker;
pub mod hub;
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
pub mod responses_api;
#[cfg(feature = "cuda")]
mod scheduled_vision;
pub mod scheduler;
pub mod server;
pub mod speculative;
pub mod telemetry;
pub mod tool_constraints;
pub mod vision;

/// Returns whether the crate was compiled with the `cuda` feature enabled.
///
/// # Examples
///
/// ```
/// let _ = w34_host::cuda_enabled();
/// ```
pub const fn cuda_enabled() -> bool {
    w34_kernels::cuda_enabled()
}
