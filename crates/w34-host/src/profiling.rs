//! Thread-local NVTX scopes; no-op without CUDA. No synchronization is introduced.
use std::{ffi::CStr, marker::PhantomData, rc::Rc};

#[cfg(feature = "cuda")]
unsafe extern "C" {
    fn w34_profile_push(label: *const std::ffi::c_char);
    fn w34_profile_pop();
}

/// A balanced NVTX push/pop, deliberately neither Send nor Sync.
#[must_use]
pub struct Range(PhantomData<Rc<()>>);

impl Range {
    pub fn new(label: &CStr) -> Self {
        #[cfg(feature = "cuda")]
        // SAFETY: label is NUL-terminated and NVTX copies it during the call.
        unsafe {
            w34_profile_push(label.as_ptr());
        }
        #[cfg(not(feature = "cuda"))]
        let _ = label;
        Self(PhantomData)
    }
}

impl Drop for Range {
    fn drop(&mut self) {
        #[cfg(feature = "cuda")]
        // SAFETY: this non-Send guard balances its push on the creating thread.
        unsafe {
            w34_profile_pop();
        }
    }
}
