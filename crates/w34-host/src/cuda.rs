//! CUDA runtime substrate, memory arenas, `TP=2` collectives, `CapacityPlanner`,
//! and `prepare_execution_plan()` kernel pre-warming.
//!
//! # Safety Contract
//!
//! These low-level wrappers rely on runtime callers to maintain the following
//! invariants; raw addresses and public handles do not enforce them in the type system.
//! 1. **Per-thread device context**: CUDA's active device is thread-local. Wrappers
//!    select their owning device where required; `*_in_current_ctx` operations and
//!    kernel/collective launches require the caller to select the device first.
//! 2. **Stack out-parameters**: scalar out-pointers (`&raw mut ...`) reference live
//!    initialized stack locals for the duration of the synchronous FFI call.
//! 3. **Device pointer lifetime**: device addresses passed into asynchronous launches
//!    are backed by `DeviceBuffer`, `DeviceArena`, or `MappedHostBuffer` owners
//!    that outlive every stream/graph execution referencing them.
//! 4. **Teardown synchronization**: `DeviceBuffer::drop` relies on `cudaFree`'s
//!    implicit device synchronization; never replace `w34_free` with an asynchronous
//!    allocator free without explicit stream synchronization.
//! 5. **Graph capture pairing**: `CudaGraph::capture` pairs
//!    `w34_graph_begin_capture` with `w34_graph_end_capture` when `body()` returns
//!    an error. This pairing does not cover panics in the capture body.
//! 6. **Cross-rank mapped staging**: `MappedHostBuffer` is `Send + Sync` only under
//!    rank-partitioned writes or explicit GPU/host fence protocols (`Tp2Collectives`
//!    epoch barriers and double-buffered decode staging).

#[cfg(feature = "cuda")]
use core::ffi::c_void;
use core::fmt;
#[cfg(feature = "cuda")]
use w34_kernels::ffi::{self, W34AttentionBatch, W34Exec, W34GdnBatch, W34KvLayerLayout};

/// Minimum free VRAM headroom required per rank after weight, scratch, state, and KV allocation.
pub const MIN_HEADROOM_BYTES: usize = 1024 * 1024 * 1024; // 1.0 GiB
/// Tighter post-pool headroom for single-card 24 GiB 27B 224K-262K (`>= 14,336` pages)
/// serving, where `weight_arena` and `scratch_arena` (336 MiB) are already pre-allocated
/// and `repack_staging` is already freed. This is the planner reserve, not a guarantee
/// of free driver memory after subsequent decode/verify graph capture.
pub const EXTREME_LONGCTX_HEADROOM_BYTES: usize = 512 * 1024 * 1024; // 512 MiB

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CudaError {
    pub code: i32,
    pub context: &'static str,
    pub detail: Option<String>,
}

impl CudaError {
    #[inline]
    pub const fn new(code: i32, context: &'static str) -> Self {
        Self {
            code,
            context,
            detail: None,
        }
    }

    #[inline]
    pub fn with_detail(code: i32, context: &'static str, detail: impl Into<String>) -> Self {
        Self {
            code,
            context,
            detail: Some(detail.into()),
        }
    }

    /// Returns the human-readable `cudaGetErrorString` description for genuine
    /// CUDA runtime error codes (`code >= 0`, excluding `w34_cublas_*` status
    /// codes and negative host-level synthetic codes).
    #[inline]
    pub fn runtime_error_string(&self) -> Option<&'static str> {
        #[cfg(feature = "cuda")]
        {
            if self.code >= 0 && !self.context.starts_with("w34_cublas_") {
                // SAFETY: `w34_cuda_error_string` accepts any integer and returns either null
                // or a static NUL-terminated string literal owned by `libcudart`.
                let ptr = unsafe { ffi::w34_cuda_error_string(self.code) };
                if !ptr.is_null() {
                    // SAFETY: `ptr` is non-null and points to a static NUL-terminated C string.
                    return unsafe { core::ffi::CStr::from_ptr(ptr) }.to_str().ok();
                }
            }
        }
        None
    }
}

impl fmt::Display for CudaError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match (self.runtime_error_string(), &self.detail) {
            (Some(rt), Some(detail)) => write!(
                f,
                "CUDA error {} ({}) in {}: {}",
                self.code, rt, self.context, detail
            ),
            (Some(rt), None) => {
                write!(f, "CUDA error {} ({}) in {}", self.code, rt, self.context)
            }
            (None, Some(detail)) => write!(
                f,
                "CUDA error {} in {}: {}",
                self.code, self.context, detail
            ),
            (None, None) => write!(f, "CUDA error {} in {}", self.code, self.context),
        }
    }
}

impl std::error::Error for CudaError {}

pub type Result<T> = core::result::Result<T, CudaError>;

#[inline]
pub fn check_rc(rc: i32, context: &'static str) -> Result<()> {
    if rc == 0 {
        Ok(())
    } else {
        Err(CudaError::new(rc, context))
    }
}

/// Converts IEEE-754 binary16 bits (`u16`) to `f32`.
///
/// # Examples
///
/// ```
/// use w34_host::cuda::{f16_to_f32, f32_to_f16};
///
/// assert_eq!(f16_to_f32(f32_to_f16(1.0)), 1.0);
/// ```
#[inline]
pub fn f16_to_f32(bits: u16) -> f32 {
    let sign = ((bits & 0x8000) as u32) << 16;
    let exp = ((bits >> 10) & 0x1f) as u32;
    let mant = (bits & 0x03ff) as u32;
    let f_bits = if exp == 0 {
        if mant == 0 {
            sign
        } else {
            let mut m = mant;
            let mut e = 127u32 - 14u32;
            while (m & 0x0400) == 0 {
                m <<= 1;
                e -= 1;
            }
            m &= 0x03ff;
            sign | (e << 23) | (m << 13)
        }
    } else if exp == 31 {
        sign | 0x7f80_0000 | (mant << 13)
    } else {
        sign | ((exp + 112) << 23) | (mant << 13)
    };
    f32::from_bits(f_bits)
}

/// Converts `f32` to IEEE-754 binary16 bits (`u16`) with round-to-nearest-even.
#[inline]
pub fn f32_to_f16(val: f32) -> u16 {
    let bits = val.to_bits();
    let sign = ((bits >> 16) & 0x8000) as u16;
    let exp = ((bits >> 23) & 0xff) as i32;
    let mant = bits & 0x007f_ffff;

    if exp == 255 {
        if mant != 0 {
            return sign | 0x7e00;
        }
        return sign | 0x7c00;
    }

    let new_exp = exp - 127 + 15;
    if new_exp >= 31 {
        return sign | 0x7c00;
    }
    if new_exp <= 0 {
        if new_exp < -10 {
            return sign;
        }
        let m = mant | 0x0080_0000;
        let shift = 14 - new_exp;
        let out_m = (m >> shift) as u16;
        let round_bit = (m >> (shift - 1)) & 1;
        let sticky = m & ((1u32 << (shift - 1)) - 1);
        let round_up = round_bit == 1 && (sticky != 0 || (out_m & 1) != 0);
        return sign | (out_m + u16::from(round_up));
    }

    let out_e = (new_exp as u16) << 10;
    let out_m = (mant >> 13) as u16;
    let round_bit = (mant >> 12) & 1;
    let sticky = mant & 0x0fff;
    let round_up = round_bit == 1 && (sticky != 0 || (out_m & 1) != 0);
    sign | (out_e + out_m + u16::from(round_up))
}

/// Converts BF16 bits (`u16`) to `f32`.
#[inline]
pub fn bf16_to_f32(bits: u16) -> f32 {
    f32::from_bits((bits as u32) << 16)
}

/// Converts `f32` to BF16 bits (`u16`) with round-to-nearest-even.
#[inline]
pub fn f32_to_bf16(val: f32) -> u16 {
    let bits = val.to_bits();
    if (bits & 0x7f80_0000) == 0x7f80_0000 && (bits & 0x007f_ffff) != 0 {
        return ((bits >> 16) as u16) | 0x0040;
    }
    let lsb = (bits >> 16) & 1;
    let rounding_bias = 0x7fff + lsb;
    ((bits.wrapping_add(rounding_bias)) >> 16) as u16
}

/// Converts BF16 bits (`u16`) to IEEE-754 FP16 bits (`u16`).
#[inline]
pub fn bf16_to_f16(bits: u16) -> u16 {
    f32_to_f16(bf16_to_f32(bits))
}

/// Memory budget calculated by [`CapacityPlanner`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CapacityPlan {
    pub total_vram_bytes: usize,
    pub free_vram_bytes: usize,
    pub weight_arena_bytes: usize,
    pub scratch_arena_bytes: usize,
    pub state_bundle_pool_bytes: usize,
    pub kv_pool_bytes: usize,
    pub headroom_bytes: usize,
    pub max_kv_pages_per_layer: usize,
}

/// Computes static memory budgets per GPU rank and enforces `MIN_HEADROOM_BYTES`.
pub struct CapacityPlanner;

impl CapacityPlanner {
    pub fn plan(
        total_vram_bytes: usize,
        free_vram_bytes: usize,
        weight_arena_bytes: usize,
        scratch_arena_bytes: usize,
        state_bundle_pool_bytes: usize,
        bytes_per_kv_page_across_layers: usize,
        requested_kv_pages: Option<usize>,
    ) -> std::result::Result<CapacityPlan, String> {
        Self::plan_with_headroom(
            total_vram_bytes,
            free_vram_bytes,
            weight_arena_bytes,
            scratch_arena_bytes,
            state_bundle_pool_bytes,
            bytes_per_kv_page_across_layers,
            requested_kv_pages,
            MIN_HEADROOM_BYTES,
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub fn plan_with_headroom(
        total_vram_bytes: usize,
        free_vram_bytes: usize,
        weight_arena_bytes: usize,
        scratch_arena_bytes: usize,
        state_bundle_pool_bytes: usize,
        bytes_per_kv_page_across_layers: usize,
        requested_kv_pages: Option<usize>,
        min_headroom_bytes: usize,
    ) -> std::result::Result<CapacityPlan, String> {
        let fixed = weight_arena_bytes
            .saturating_add(scratch_arena_bytes)
            .saturating_add(state_bundle_pool_bytes)
            .saturating_add(min_headroom_bytes);
        if fixed > free_vram_bytes {
            return Err(format!(
                "CapacityPlanner: fixed footprint ({:.2} GiB) exceeds free VRAM ({:.2} GiB); need >= {:.2} GiB headroom",
                fixed as f64 / (1024.0 * 1024.0 * 1024.0),
                free_vram_bytes as f64 / (1024.0 * 1024.0 * 1024.0),
                min_headroom_bytes as f64 / (1024.0 * 1024.0 * 1024.0),
            ));
        }
        let max_avail_for_kv = free_vram_bytes - fixed;
        let max_possible_pages = max_avail_for_kv
            .checked_div(bytes_per_kv_page_across_layers)
            .unwrap_or(0);
        let pages = match requested_kv_pages {
            Some(req) => {
                if req > max_possible_pages {
                    return Err(format!(
                        "CapacityPlanner: requested {req} KV pages exceeds capacity {max_possible_pages}"
                    ));
                }
                req
            }
            None => max_possible_pages,
        };
        let kv_pool_bytes = pages.saturating_mul(bytes_per_kv_page_across_layers);
        let headroom_bytes = free_vram_bytes
            .saturating_sub(weight_arena_bytes)
            .saturating_sub(scratch_arena_bytes)
            .saturating_sub(state_bundle_pool_bytes)
            .saturating_sub(kv_pool_bytes);
        Ok(CapacityPlan {
            total_vram_bytes,
            free_vram_bytes,
            weight_arena_bytes,
            scratch_arena_bytes,
            state_bundle_pool_bytes,
            kv_pool_bytes,
            headroom_bytes,
            max_kv_pages_per_layer: pages,
        })
    }
}

/// Architecture family for 30-series (`sm_86`) and 40-series (`sm_89`) GPUs.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CudaArchFamily {
    /// NVIDIA Ampere 30-series / workstation (`sm_86`: RTX 3060..3090 Ti, A4000..A6000, A40).
    AmpereSm86,
    /// NVIDIA Ada Lovelace 40-series / workstation (`sm_89`: RTX 4060..4090, RTX 6000 Ada, L40S).
    AdaSm89,
    /// Other Ampere-or-newer architecture (`>= 8.0`, e.g. `sm_80`, `sm_87`, `sm_90`).
    OtherAmpereOrNewer { major: i32, minor: i32 },
}

/// Minimum opt-in dynamic shared memory per block required by `W34PagedD256` (`98,304 B`).
/// Detected from `cudaDeviceProp::sharedMemPerBlockOptin` (`101,376 B` on all 30/40-series GPUs).
pub const MIN_REQUIRED_OPTIN_SMEM_BYTES: usize = 98_304;

/// Immutable hardware properties queried once per GPU rank from `cudaGetDeviceProperties`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CudaDeviceProps {
    pub device_id: i32,
    pub compute_major: i32,
    pub compute_minor: i32,
    pub multi_processor_count: i32,
    pub max_shared_mem_per_block_optin: usize,
    pub max_shared_mem_per_multiprocessor: usize,
    pub l2_cache_bytes: usize,
    pub persisting_l2_max_bytes: usize,
    pub name: String,
}

impl CudaDeviceProps {
    #[inline]
    pub fn sm_tag(&self) -> String {
        format!("sm_{}{}", self.compute_major, self.compute_minor)
    }

    pub fn arch_family(&self) -> Result<CudaArchFamily> {
        match (self.compute_major, self.compute_minor) {
            (8, 6) => Ok(CudaArchFamily::AmpereSm86),
            (8, 9) => Ok(CudaArchFamily::AdaSm89),
            (major, minor) if (major > 8 && minor >= 0) || (major == 8 && minor >= 6) => {
                Ok(CudaArchFamily::OtherAmpereOrNewer { major, minor })
            }
            (major, minor) => Err(CudaError::with_detail(
                -20,
                "unsupported GPU compute capability",
                format!(
                    "GPU {} ({}) has compute capability {}.{} (sm_{}{}); warp-34 requires 30-series (sm_86) or 40-series (sm_89) (>= 8.6)",
                    self.device_id, self.name, major, minor, major, minor
                ),
            )),
        }
    }

    pub fn validate_supported(&self) -> Result<CudaArchFamily> {
        let family = self.arch_family()?;
        if self.multi_processor_count <= 0 {
            return Err(CudaError::with_detail(
                -21,
                "invalid GPU multiProcessorCount",
                format!(
                    "GPU {} ({}) reported multi_processor_count={}",
                    self.device_id, self.name, self.multi_processor_count
                ),
            ));
        }
        if self.max_shared_mem_per_block_optin < MIN_REQUIRED_OPTIN_SMEM_BYTES {
            return Err(CudaError::with_detail(
                -22,
                "insufficient GPU opt-in shared memory",
                format!(
                    "GPU {} ({}, {}) provides {} B opt-in shared memory per block, need >= {} B",
                    self.device_id,
                    self.name,
                    self.sm_tag(),
                    self.max_shared_mem_per_block_optin,
                    MIN_REQUIRED_OPTIN_SMEM_BYTES
                ),
            ));
        }
        Ok(family)
    }

    #[inline]
    pub fn marlin_workspace_ints(&self) -> usize {
        (self.multi_processor_count.max(1) as usize).saturating_mul(64)
    }

    #[inline]
    pub fn marlin_workspace_bytes(&self) -> usize {
        self.marlin_workspace_ints()
            .saturating_mul(core::mem::size_of::<i32>())
    }
}

#[cfg(feature = "cuda")]
pub struct CudaDevice {
    pub device_id: i32,
}

#[cfg(feature = "cuda")]
impl CudaDevice {
    pub fn count() -> Result<i32> {
        let mut count = 0i32;
        check_rc(
            // SAFETY: `count` is a valid initialized stack `i32` out-parameter.
            unsafe { ffi::w34_get_device_count(&raw mut count) },
            "w34_get_device_count",
        )?;
        Ok(count)
    }

    pub fn props_for(device_id: i32) -> Result<CudaDeviceProps> {
        let mut raw_props = ffi::W34DeviceProps::default();
        check_rc(
            // SAFETY: `raw_props` is a valid initialized stack `W34DeviceProps` out-parameter.
            unsafe { ffi::w34_get_device_props(device_id, &raw mut raw_props) },
            "w34_get_device_props",
        )?;
        let nul_pos = raw_props
            .name
            .iter()
            .position(|&b| b == 0)
            .unwrap_or(raw_props.name.len());
        let name = String::from_utf8_lossy(&raw_props.name[..nul_pos]).into_owned();
        Ok(CudaDeviceProps {
            device_id: raw_props.device_id,
            compute_major: raw_props.compute_major,
            compute_minor: raw_props.compute_minor,
            multi_processor_count: raw_props.multi_processor_count,
            max_shared_mem_per_block_optin: raw_props.max_shared_mem_per_block_optin.max(0)
                as usize,
            max_shared_mem_per_multiprocessor: raw_props.max_shared_mem_per_multiprocessor.max(0)
                as usize,
            l2_cache_bytes: raw_props.l2_cache_bytes.max(0) as usize,
            persisting_l2_max_bytes: raw_props.persisting_l2_max_bytes.max(0) as usize,
            name,
        })
    }

    pub fn props(&self) -> Result<CudaDeviceProps> {
        Self::props_for(self.device_id)
    }

    pub fn set_current(device_id: i32) -> Result<Self> {
        // SAFETY: `w34_set_device` validates `device_id` via `cudaSetDevice` and returns an error code.
        check_rc(unsafe { ffi::w34_set_device(device_id) }, "w34_set_device")?;
        Ok(Self { device_id })
    }

    pub fn make_current(&self) -> Result<()> {
        check_rc(
            // SAFETY: `self.device_id` was validated on construction; return code is checked.
            unsafe { ffi::w34_set_device(self.device_id) },
            "w34_set_device",
        )
    }

    pub fn synchronize(&self) -> Result<()> {
        self.make_current()?;
        check_rc(
            // SAFETY: current thread's active device is set above; synchronizes device work.
            unsafe { ffi::w34_device_synchronize() },
            "w34_device_synchronize",
        )
    }

    pub fn mem_info(&self) -> Result<(usize, usize)> {
        self.make_current()?;
        let mut free = 0usize;
        let mut total = 0usize;
        check_rc(
            // SAFETY: `free` and `total` are valid stack `usize` out-parameters on the active device.
            unsafe { ffi::w34_get_mem_info(&raw mut free, &raw mut total) },
            "w34_get_mem_info",
        )?;
        Ok((free, total))
    }
}

/// Sticky host failures shared by streams and events through flat slot snapshots.
/// Register slots before launching any GPU work or graphs that use them. A host
/// worker must publish its error before acknowledging completion to the GPU.
#[cfg(any(feature = "cuda", test))]
#[derive(Default)]
pub(crate) struct HostFailures {
    slots: std::sync::Mutex<Vec<std::sync::Arc<std::sync::OnceLock<CudaError>>>>,
}

#[cfg(any(feature = "cuda", test))]
impl HostFailures {
    pub(crate) fn register(&self) -> std::sync::Arc<std::sync::OnceLock<CudaError>> {
        let slot = std::sync::Arc::new(std::sync::OnceLock::new());
        self.slots.lock().unwrap().push(slot.clone());
        slot
    }

    pub(crate) fn check(&self) -> Result<()> {
        for slot in self.slots.lock().unwrap().iter() {
            if let Some(error) = slot.get() {
                return Err(error.clone());
            }
        }
        Ok(())
    }

    fn inherit(&self, source: &Self) {
        // Release the source lock before taking ours, including for self-inheritance.
        // Sharing slots (not owners) preserves late publication without cycles.
        let snapshot = source.slots.lock().unwrap().clone();
        if snapshot.is_empty() {
            return;
        }
        let mut slots = self.slots.lock().unwrap();
        for slot in snapshot {
            if !slots
                .iter()
                .any(|existing| std::sync::Arc::ptr_eq(existing, &slot))
            {
                slots.push(slot);
            }
        }
    }
}

#[cfg(test)]
mod host_failure_tests {
    use super::{CudaError, HostFailures};

    #[test]
    fn late_publication_after_snapshot() {
        let source = HostFailures::default();
        let slot = source.register();
        let target = HostFailures::default();
        target.inherit(&source);
        assert_eq!(target.check(), Ok(()));
        let error = CudaError::new(-1, "host compute");
        std::thread::spawn(move || slot.set(error).unwrap())
            .join()
            .unwrap();
        assert_eq!(target.check(), Err(CudaError::new(-1, "host compute")));
    }

    #[test]
    fn failures_are_sticky_and_first_error_wins() {
        let failures = HostFailures::default();
        assert_eq!(failures.check(), Ok(()));
        let first = failures.register();
        let second = failures.register();
        let error = CudaError::with_detail(-1, "host compute", "failed response");
        first.set(error.clone()).unwrap();
        second.set(CudaError::new(-2, "later failure")).unwrap();
        for _ in 0..3 {
            assert_eq!(failures.check(), Err(error.clone()));
        }
        assert!(first.set(CudaError::new(-3, "replacement")).is_err());
        assert_eq!(failures.check(), Err(error));
    }

    #[test]
    fn repeated_and_bidirectional_inheritance_deduplicates() {
        let source = HostFailures::default();
        let first = source.register();
        let target = HostFailures::default();
        let second = target.register();
        for _ in 0..3 {
            target.inherit(&source);
            source.inherit(&target);
            target.inherit(&target);
        }
        assert_eq!(source.slots.lock().unwrap().len(), 2);
        assert_eq!(target.slots.lock().unwrap().len(), 2);
        drop(source);
        drop(target);
        assert_eq!(std::sync::Arc::strong_count(&first), 1);
        assert_eq!(std::sync::Arc::strong_count(&second), 1);
    }

    #[test]
    fn transitive_inheritance_outlives_source_owners() {
        let source = HostFailures::default();
        let slot = source.register();
        let event = HostFailures::default();
        event.inherit(&source);
        let dependent = HostFailures::default();
        dependent.inherit(&event);
        drop(source);
        drop(event);
        let error = CudaError::new(-1, "host compute");
        slot.set(error.clone()).unwrap();
        assert_eq!(dependent.check(), Err(error));
    }
}

#[cfg(feature = "cuda")]
pub struct CudaStream {
    pub device_id: i32,
    pub handle: usize,
    pub(crate) host_failures: HostFailures,
}

#[cfg(feature = "cuda")]
impl CudaStream {
    pub fn new(device_id: i32) -> Result<Self> {
        // SAFETY: selects `device_id` on this thread before creating the non-blocking stream.
        check_rc(unsafe { ffi::w34_set_device(device_id) }, "w34_set_device")?;
        let mut handle = 0usize;
        check_rc(
            // SAFETY: `handle` is a valid stack out-parameter for the new `cudaStream_t`.
            unsafe { ffi::w34_stream_create(&raw mut handle) },
            "w34_stream_create",
        )?;
        Ok(Self {
            device_id,
            handle,
            host_failures: HostFailures::default(),
        })
    }

    pub fn synchronize(&self) -> Result<()> {
        check_rc(
            // SAFETY: selects the stream's owning device on the calling thread.
            unsafe { ffi::w34_set_device(self.device_id) },
            "w34_set_device",
        )?;
        check_rc(
            // SAFETY: `self.handle` is a live `cudaStream_t` owned by `self`.
            unsafe { ffi::w34_stream_synchronize(self.handle) },
            "w34_stream_synchronize",
        )?;
        self.host_failures.check()
    }

    pub fn exec(&self, scratch_ptr: u64, scratch_size: usize) -> W34Exec {
        W34Exec {
            stream: self.handle,
            scratch_ptr,
            scratch_size,
        }
    }

    pub fn wait_event(&self, event: &CudaEvent) -> Result<()> {
        self.host_failures.inherit(&event.host_failures);
        check_rc(
            // SAFETY: both `self.handle` and `event.handle` are live CUDA handles owned by their wrappers.
            unsafe { ffi::w34_stream_wait_event(self.handle, event.handle) },
            "w34_stream_wait_event",
        )
    }

    pub fn set_persisting_l2_window(&self, base_ptr: u64, num_bytes: usize) -> Result<()> {
        check_rc(
            // SAFETY: selects the stream's owning device on the calling thread.
            unsafe { ffi::w34_set_device(self.device_id) },
            "w34_set_device",
        )?;
        check_rc(
            // SAFETY: `self.handle` is a live `cudaStream_t` and `base_ptr` is a device allocation
            // of at least `num_bytes` on `self.device_id` (or null when `num_bytes == 0`).
            unsafe {
                ffi::w34_stream_set_persisting_l2_window(
                    self.handle,
                    base_ptr as usize as *const c_void,
                    num_bytes,
                )
            },
            "w34_stream_set_persisting_l2_window",
        )
    }
}

#[cfg(feature = "cuda")]
impl Drop for CudaStream {
    fn drop(&mut self) {
        if self.handle != 0 {
            // SAFETY: `self.handle` is non-zero and uniquely owned by `self`; device context is set first.
            unsafe {
                let _ = ffi::w34_set_device(self.device_id);
                let _ = ffi::w34_stream_set_persisting_l2_window(self.handle, std::ptr::null(), 0);
                let _ = ffi::w34_stream_destroy(self.handle);
            }
        }
    }
}

#[cfg(feature = "cuda")]
pub struct CudaEvent {
    pub device_id: i32,
    pub handle: usize,
    pub(crate) host_failures: HostFailures,
}

#[cfg(feature = "cuda")]
impl CudaEvent {
    pub fn new(device_id: i32, enable_timing: bool) -> Result<Self> {
        // SAFETY: selects `device_id` on this thread before creating the CUDA event.
        check_rc(unsafe { ffi::w34_set_device(device_id) }, "w34_set_device")?;
        let mut handle = 0usize;
        check_rc(
            // SAFETY: `handle` is a valid stack out-parameter for the new `cudaEvent_t`.
            unsafe { ffi::w34_event_create(&raw mut handle, i32::from(enable_timing)) },
            "w34_event_create",
        )?;
        Ok(Self {
            device_id,
            handle,
            host_failures: HostFailures::default(),
        })
    }

    pub fn record(&self, stream: &CudaStream) -> Result<()> {
        check_rc(
            // SAFETY: selects the event's owning device before recording onto `stream`.
            unsafe { ffi::w34_set_device(self.device_id) },
            "w34_set_device",
        )?;
        self.record_in_current_ctx(stream)
    }

    pub fn record_in_current_ctx(&self, stream: &CudaStream) -> Result<()> {
        self.host_failures.inherit(&stream.host_failures);
        check_rc(
            // SAFETY: `self.handle` and `stream.handle` are live event/stream handles on the same device.
            unsafe { ffi::w34_event_record(self.handle, stream.handle) },
            "w34_event_record",
        )
    }

    pub fn synchronize(&self) -> Result<()> {
        check_rc(
            // SAFETY: `self.handle` is a live `cudaEvent_t` owned by `self`.
            unsafe { ffi::w34_event_synchronize(self.handle) },
            "w34_event_synchronize",
        )?;
        self.host_failures.check()
    }

    pub fn elapsed_ms(&self, end: &CudaEvent) -> Result<f32> {
        let mut ms = 0.0f32;
        check_rc(
            // SAFETY: `ms` is a stack `f32` out-parameter; `self` and `end` are live timing events.
            unsafe { ffi::w34_event_elapsed_ms(&raw mut ms, self.handle, end.handle) },
            "w34_event_elapsed_ms",
        )?;
        Ok(ms)
    }
}

#[cfg(feature = "cuda")]
impl Drop for CudaEvent {
    fn drop(&mut self) {
        if self.handle != 0 {
            // SAFETY: `self.handle` is non-zero and uniquely owned by `self`; device context is set first.
            unsafe {
                let _ = ffi::w34_set_device(self.device_id);
                let _ = ffi::w34_event_destroy(self.handle);
            }
        }
    }
}

#[cfg(feature = "cuda")]
pub struct CublasHandle {
    pub device_id: i32,
    pub handle: usize,
}

#[cfg(feature = "cuda")]
impl CublasHandle {
    pub fn new(stream: &CudaStream) -> Result<Self> {
        check_rc(
            // SAFETY: binds the calling thread to `stream.device_id` before creating `W34Blas`.
            unsafe { ffi::w34_set_device(stream.device_id) },
            "w34_set_device",
        )?;
        let mut handle = 0usize;
        check_rc(
            // SAFETY: `handle` is a stack out-parameter and `stream.handle` is a live stream on `device_id`.
            unsafe { ffi::w34_cublas_create(&raw mut handle, stream.handle) },
            "w34_cublas_create",
        )?;
        Ok(Self {
            device_id: stream.device_id,
            handle,
        })
    }

    #[allow(clippy::too_many_arguments)]
    pub fn gemm_f16(
        &self,
        a_ptr: u64,
        b_ptr: u64,
        c_ptr: u64,
        m: i32,
        n: i32,
        k: i32,
    ) -> Result<()> {
        check_rc(
            // SAFETY: `self.handle` is a live `W34Blas` handle; caller guarantees `a_ptr`, `b_ptr`,
            // and `c_ptr` reference valid FP16 device buffers sized for `[m, k]`, `[n, k]`, `[m, n]`.
            unsafe {
                ffi::w34_cublas_gemm_f16(
                    self.handle,
                    a_ptr as *const c_void,
                    b_ptr as *const c_void,
                    c_ptr as *mut c_void,
                    m,
                    n,
                    k,
                )
            },
            "w34_cublas_gemm_f16",
        )
    }

    /// Row-major native FP16 Linear using the cuBLASLt BIAS epilogue.
    ///
    /// Warm each shape, bias address and pointer-alignment combination immediately
    /// before capture (cold capture returns an error). The 256-entry FIFO plan
    /// cache can evict older combinations without invalidating existing graphs.
    /// The handle owns a persistent 1 MiB workspace: it and the tensors must
    /// outlive every captured graph replay. Graph replays and ordinary biased
    /// GEMMs sharing a handle must not overlap across different streams.
    /// Ordinary calls submit on the stream passed to `new`; keep it alive.
    /// The public handle value is opaque, not a vendor cuBLAS handle.
    #[allow(clippy::too_many_arguments)]
    pub fn gemm_f16_bias(
        &self,
        a_ptr: u64,
        b_ptr: u64,
        bias_ptr: u64,
        c_ptr: u64,
        m: i32,
        n: i32,
        k: i32,
    ) -> Result<()> {
        check_rc(
            // SAFETY: `self.handle` is a live `W34Blas` handle; caller guarantees `a_ptr`, `b_ptr`,
            // `bias_ptr`, and `c_ptr` reference valid FP16 device buffers outliving graph replay.
            unsafe {
                ffi::w34_cublas_gemm_f16_bias(
                    self.handle,
                    a_ptr as *const c_void,
                    b_ptr as *const c_void,
                    bias_ptr as *const c_void,
                    c_ptr as *mut c_void,
                    m,
                    n,
                    k,
                )
            },
            "w34_cublas_gemm_f16_bias",
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub fn gemm_bf16(
        &self,
        a_ptr: u64,
        b_ptr: u64,
        c_ptr: u64,
        m: i32,
        n: i32,
        k: i32,
    ) -> Result<()> {
        check_rc(
            // SAFETY: `self.handle` is a live `W34Blas` handle; caller guarantees `a_ptr`, `b_ptr`,
            // and `c_ptr` reference valid BF16 device buffers sized for `[m, k]`, `[n, k]`, `[m, n]`.
            unsafe {
                ffi::w34_cublas_gemm_bf16(
                    self.handle,
                    a_ptr as *const c_void,
                    b_ptr as *const c_void,
                    c_ptr as *mut c_void,
                    m,
                    n,
                    k,
                )
            },
            "w34_cublas_gemm_bf16",
        )
    }
}

#[cfg(feature = "cuda")]
impl Drop for CublasHandle {
    fn drop(&mut self) {
        if self.handle != 0 {
            // SAFETY: `self.handle` is non-zero and uniquely owned by `self`; device context is set first.
            unsafe {
                let _ = ffi::w34_set_device(self.device_id);
                let _ = ffi::w34_cublas_destroy(self.handle);
            }
        }
    }
}

#[cfg(feature = "cuda")]
pub struct CudaGraph {
    pub device_id: i32,
    pub graph: usize,
    pub graph_exec: usize,
}

#[cfg(feature = "cuda")]
impl CudaGraph {
    pub fn capture<F>(stream: &CudaStream, body: F) -> Result<Self>
    where
        F: FnOnce() -> Result<()>,
    {
        check_rc(
            // SAFETY: binds calling thread to `stream.device_id` prior to stream capture.
            unsafe { ffi::w34_set_device(stream.device_id) },
            "w34_set_device",
        )?;
        check_rc(
            // SAFETY: `stream.handle` is a live non-default stream on `stream.device_id`.
            unsafe { ffi::w34_graph_begin_capture(stream.handle) },
            "w34_graph_begin_capture",
        )?;
        let body_res = body();
        let mut graph = 0usize;
        let mut graph_exec = 0usize;
        // SAFETY: always pairs `w34_graph_begin_capture` on `stream.handle` even if `body()` failed;
        // `graph` and `graph_exec` are stack out-parameters transferred into `captured` for RAII cleanup.
        let end_rc = unsafe {
            ffi::w34_graph_end_capture(stream.handle, &raw mut graph, &raw mut graph_exec)
        };
        let captured = Self {
            device_id: stream.device_id,
            graph,
            graph_exec,
        };
        body_res?;
        check_rc(end_rc, "w34_graph_end_capture")?;
        Ok(captured)
    }

    pub fn launch(&self, stream: &CudaStream) -> Result<()> {
        check_rc(
            // SAFETY: binds calling thread to `self.device_id` before replaying `self.graph_exec`.
            unsafe { ffi::w34_set_device(self.device_id) },
            "w34_set_device",
        )?;
        check_rc(
            // SAFETY: `self.graph_exec` and `stream.handle` are live handles on `self.device_id`.
            unsafe { ffi::w34_graph_launch(self.graph_exec, stream.handle) },
            "w34_graph_launch",
        )
    }
}

#[cfg(feature = "cuda")]
impl Drop for CudaGraph {
    fn drop(&mut self) {
        if self.graph != 0 || self.graph_exec != 0 {
            // SAFETY: `self.graph` / `self.graph_exec` are uniquely owned by `self`; `w34_graph_destroy`
            // tolerates zero handles on partial capture failures.
            unsafe {
                let _ = ffi::w34_set_device(self.device_id);
                let _ = ffi::w34_graph_destroy(self.graph, self.graph_exec);
            }
        }
    }
}

#[cfg(feature = "cuda")]
pub struct DeviceBuffer {
    pub device_id: i32,
    pub ptr: *mut c_void,
    pub len_bytes: usize,
}

// SAFETY: `DeviceBuffer` owns a CUDA global-memory allocation (`cudaMalloc`) whose device
// virtual address is valid across host threads once `cudaSetDevice(self.device_id)` is bound;
// host code never dereferences `self.ptr`, and teardown uses `cudaFree`.
#[cfg(feature = "cuda")]
unsafe impl Send for DeviceBuffer {}
// SAFETY: sharing this wrapper does not expose host references into device memory.
// Callers must order GPU writes and copies with streams/events; sharing the wrapper
// does not itself establish ordering between different streams.
#[cfg(feature = "cuda")]
unsafe impl Sync for DeviceBuffer {}

#[cfg(feature = "cuda")]
impl DeviceBuffer {
    pub fn alloc(device_id: i32, len_bytes: usize) -> Result<Self> {
        // SAFETY: selects `device_id` on this thread before `cudaMalloc` + zero-init.
        check_rc(unsafe { ffi::w34_set_device(device_id) }, "w34_set_device")?;
        let alloc_bytes = len_bytes.max(16);
        let mut ptr: *mut c_void = core::ptr::null_mut();
        check_rc(
            // SAFETY: `ptr` is a stack out-parameter and `alloc_bytes >= 16`.
            unsafe { ffi::w34_malloc(&raw mut ptr, alloc_bytes) },
            "w34_malloc",
        )?;
        check_rc(
            // SAFETY: `ptr` was just allocated for `alloc_bytes` on `device_id`; stream 0 zero-fills it.
            unsafe { ffi::w34_memset_async(ptr, 0, alloc_bytes, 0) },
            "w34_memset_async",
        )?;
        check_rc(
            // SAFETY: waits for the zero-fill on stream 0 before exposing `Self`.
            unsafe { ffi::w34_stream_synchronize(0) },
            "w34_stream_synchronize(0)",
        )?;
        Ok(Self {
            device_id,
            ptr,
            len_bytes,
        })
    }

    #[inline]
    pub fn addr(&self) -> u64 {
        self.ptr as u64
    }

    pub fn zero_async(&self, stream: &CudaStream) -> Result<()> {
        if self.len_bytes == 0 {
            return Ok(());
        }
        check_rc(
            // SAFETY: binds calling thread to `self.device_id` before `cudaMemsetAsync`.
            unsafe { ffi::w34_set_device(self.device_id) },
            "w34_set_device",
        )?;
        check_rc(
            // SAFETY: `self.ptr` is live for `self.len_bytes` on `self.device_id`, and `stream` is live.
            unsafe { ffi::w34_memset_async(self.ptr, 0, self.len_bytes, stream.handle) },
            "w34_memset_async",
        )
    }

    pub fn copy_from_host<T: Copy>(&self, src: &[T], stream: &CudaStream) -> Result<()> {
        let bytes = core::mem::size_of_val(src);
        if bytes > self.len_bytes {
            return Err(CudaError::new(-11, "copy_from_host buffer overflow"));
        }
        if bytes == 0 {
            return Ok(());
        }
        check_rc(
            // SAFETY: binds calling thread to `self.device_id` before H2D copy.
            unsafe { ffi::w34_set_device(self.device_id) },
            "w34_set_device",
        )?;
        check_rc(
            // SAFETY: the destination bound is checked above and `src` supplies `bytes`.
            // Runtime callers must retain the host storage until the stream consumes it;
            // this low-level API does not encode an asynchronous source lifetime.
            unsafe {
                ffi::w34_memcpy_h2d_async(
                    self.ptr,
                    src.as_ptr() as *const c_void,
                    bytes,
                    stream.handle,
                )
            },
            "w34_memcpy_h2d_async",
        )
    }

    pub fn copy_to_host<T: Copy>(&self, dst: &mut [T], stream: &CudaStream) -> Result<()> {
        let bytes = core::mem::size_of_val(dst);
        if bytes > self.len_bytes {
            return Err(CudaError::new(-11, "copy_to_host buffer overflow"));
        }
        if bytes == 0 {
            return Ok(());
        }
        check_rc(
            // SAFETY: binds calling thread to `self.device_id` before D2H copy.
            unsafe { ffi::w34_set_device(self.device_id) },
            "w34_set_device",
        )?;
        check_rc(
            // SAFETY: `bytes <= self.len_bytes` is checked above, `dst` is uniquely borrowed for
            // `bytes`, and `stream.synchronize()` below completes the DMA before `dst` is released.
            unsafe {
                ffi::w34_memcpy_d2h_async(
                    dst.as_mut_ptr() as *mut c_void,
                    self.ptr,
                    bytes,
                    stream.handle,
                )
            },
            "w34_memcpy_d2h_async",
        )?;
        stream.synchronize()
    }
}

#[cfg(feature = "cuda")]
impl Drop for DeviceBuffer {
    fn drop(&mut self) {
        if !self.ptr.is_null() {
            // SAFETY: `self.ptr` is uniquely owned by `self` and allocated via `w34_malloc`
            // (`cudaMalloc`); `w34_free` (`cudaFree`) implicitly synchronizes pending GPU work.
            unsafe {
                let _ = ffi::w34_set_device(self.device_id);
                let _ = ffi::w34_free(self.ptr);
            }
        }
    }
}

/// Bump-allocated GPU memory arena with 256-byte sub-allocation alignment.
#[cfg(feature = "cuda")]
pub struct DeviceArena {
    buf: DeviceBuffer,
    offset: usize,
}

#[cfg(feature = "cuda")]
impl DeviceArena {
    pub fn new(device_id: i32, capacity_bytes: usize) -> Result<Self> {
        let buf = DeviceBuffer::alloc(device_id, capacity_bytes)?;
        Ok(Self { buf, offset: 0 })
    }

    #[inline]
    pub fn device_id(&self) -> i32 {
        self.buf.device_id
    }

    #[inline]
    pub fn capacity(&self) -> usize {
        self.buf.len_bytes
    }

    #[inline]
    pub fn allocation_base(&self) -> u64 {
        self.buf.addr()
    }

    #[inline]
    pub fn used(&self) -> usize {
        self.offset
    }

    #[inline]
    pub fn reset(&mut self) {
        self.offset = 0;
    }

    #[inline]
    pub fn remaining_workspace(&self) -> (u64, usize) {
        let aligned_off = (self.offset + 255) & !255usize;
        if aligned_off >= self.buf.len_bytes {
            (0, 0)
        } else {
            (
                self.buf.addr() + aligned_off as u64,
                self.buf.len_bytes - aligned_off,
            )
        }
    }

    pub fn alloc_aligned(&mut self, bytes: usize) -> Result<u64> {
        let aligned_off = (self.offset + 255) & !255usize;
        let next = aligned_off
            .checked_add(bytes)
            .ok_or(CudaError::new(-1, "DeviceArena::alloc_aligned overflow"))?;
        if next > self.buf.len_bytes {
            return Err(CudaError::with_detail(
                2, // cudaErrorMemoryAllocation
                "DeviceArena capacity exceeded",
                format!(
                    "requested {} MiB total > {} MiB arena capacity",
                    next / (1024 * 1024),
                    self.buf.len_bytes / (1024 * 1024)
                ),
            ));
        }
        self.offset = next;
        Ok(self.buf.addr() + aligned_off as u64)
    }

    pub fn upload_slice<T: Copy>(&mut self, src: &[T], stream: &CudaStream) -> Result<u64> {
        let bytes = core::mem::size_of_val(src);
        let addr = self.alloc_aligned(bytes.max(16))?;
        if bytes > 0 {
            check_rc(
                // SAFETY: binds calling thread to `self.buf.device_id` before H2D copy into `addr`.
                unsafe { ffi::w34_set_device(self.buf.device_id) },
                "w34_set_device",
            )?;
            check_rc(
                // SAFETY: `addr..addr + bytes` was just sub-allocated within `self.buf`.
                // As with `copy_from_host`, callers must retain source storage until DMA
                // completes; a borrowed slice alone does not prove that lifetime.
                unsafe {
                    ffi::w34_memcpy_h2d_async(
                        addr as *mut c_void,
                        src.as_ptr() as *const c_void,
                        bytes,
                        stream.handle,
                    )
                },
                "w34_memcpy_h2d_async",
            )?;
        }
        Ok(addr)
    }
}

/// Pinned portable mapped host buffer visible to all GPUs over PCIe.
#[cfg(feature = "cuda")]
pub struct MappedHostBuffer {
    pub host_ptr: *mut c_void,
    pub dev_ptr: *mut c_void,
    pub len_bytes: usize,
}

// SAFETY: `MappedHostBuffer` wraps `cudaHostAllocPortable | cudaHostAllocMapped` memory, which
// CUDA explicitly makes accessible from any host thread and any GPU context in the process.
#[cfg(feature = "cuda")]
unsafe impl Send for MappedHostBuffer {}
// SAFETY: Sharing `&MappedHostBuffer` across rank threads (`Arc<Tp2Collectives>`) is sound only
// because callers partition writes per rank (`stage0`/`sig0` vs `stage1`/`sig1`) or synchronize
// host/GPU access via CUDA event completion (`BatchedDecodeGraphIo` double-buffered slots).
#[cfg(feature = "cuda")]
unsafe impl Sync for MappedHostBuffer {}

#[cfg(feature = "cuda")]
impl MappedHostBuffer {
    pub fn alloc(len_bytes: usize) -> Result<Self> {
        let alloc_bytes = len_bytes.max(128);
        let mut host_ptr: *mut c_void = core::ptr::null_mut();
        let mut dev_ptr: *mut c_void = core::ptr::null_mut();
        check_rc(
            // SAFETY: `host_ptr` and `dev_ptr` are valid stack out-parameters and `alloc_bytes >= 128`.
            unsafe { ffi::w34_host_alloc_mapped(&raw mut host_ptr, &raw mut dev_ptr, alloc_bytes) },
            "w34_host_alloc_mapped",
        )?;
        Ok(Self {
            host_ptr,
            dev_ptr,
            len_bytes,
        })
    }

    #[inline]
    pub fn dev_addr(&self) -> u64 {
        self.dev_ptr as u64
    }
}

#[cfg(feature = "cuda")]
impl Drop for MappedHostBuffer {
    fn drop(&mut self) {
        if !self.host_ptr.is_null() {
            // SAFETY: `self.host_ptr` is uniquely owned by `self` and allocated by `w34_host_alloc_mapped`.
            unsafe {
                let _ = ffi::w34_host_free(self.host_ptr);
            }
        }
    }
}

#[cfg(feature = "cuda")]
mod collectives;
#[cfg(feature = "cuda")]
pub use collectives::*;

#[cfg(feature = "cuda")]
mod warmup;
#[cfg(feature = "cuda")]
pub use warmup::prepare_execution_plan;

#[cfg(test)]
mod tests;
