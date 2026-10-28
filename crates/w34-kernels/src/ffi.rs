//! FFI bindings to `csrc/w34_cuda.h`.

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct W34Exec {
    pub stream: usize,
    pub scratch_ptr: u64,
    pub scratch_size: usize,
}

const _: [(); 24] = [(); core::mem::size_of::<W34Exec>()];

pub const W34_DTYPE_F16: i32 = 0;
pub const W34_DTYPE_BF16: i32 = 1;
pub const W34_DTYPE_F32: i32 = 2;
pub const W34_DTYPE_I32: i32 = 3;
pub const W34_DTYPE_U8: i32 = 4;
pub const W34_DTYPE_U32: i32 = 5;

#[cfg(feature = "cuda")]
unsafe extern "C" {
    fn w34_cublas_runtime_version(out_version: *mut i32) -> i32;
}

/// Returns the linked cuBLAS runtime version (`major * 10000 + minor * 100 + patch`).
#[cfg(feature = "cuda")]
pub fn cublas_runtime_version() -> Result<i32, i32> {
    let mut version = 0i32;
    // SAFETY: `version` is a valid stack-allocated `i32` pointer for the duration of the call.
    let rc = unsafe { w34_cublas_runtime_version(&raw mut version) };
    if rc == 0 { Ok(version) } else { Err(rc) }
}
