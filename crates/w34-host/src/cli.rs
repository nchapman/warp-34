#[cfg(feature = "cuda")]
use std::env;
#[cfg(feature = "cuda")]
use std::fs;
#[cfg(feature = "cuda")]
use std::path::Path;
#[cfg(feature = "cuda")]
use std::process::ExitCode;

#[cfg(feature = "cuda")]
use crate::cache::KvFormatPolicy;
#[cfg(feature = "cuda")]
use crate::hub::{HubOptions, ModelRole};
#[cfg(feature = "cuda")]
use crate::model::Qwen3_5Engine;

#[cfg(feature = "cuda")]
mod commands;
mod options;
#[cfg(feature = "cuda")]
mod oracle;

#[cfg(feature = "cuda")]
pub use commands::run_cuda_cli;
#[cfg(feature = "cuda")]
pub(crate) use commands::*;
pub use options::print_usage;
#[cfg(feature = "cuda")]
pub(crate) use options::*;
#[cfg(feature = "cuda")]
pub(crate) use oracle::*;

#[cfg(all(test, feature = "cuda"))]
mod tests;
