//! CUDA runtime substrate, memory arenas, `TP=2` collectives, `CapacityPlanner`,
//! and `prepare_execution_plan()` kernel pre-warming.
#![allow(clippy::undocumented_unsafe_blocks, clippy::too_many_arguments)]

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
}

impl fmt::Display for CudaError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if let Some(detail) = &self.detail {
            write!(
                f,
                "CUDA error {} in {}: {}",
                self.code, self.context, detail
            )
        } else {
            write!(f, "CUDA error {} in {}", self.code, self.context)
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

#[cfg(feature = "cuda")]
pub struct CudaDevice {
    pub device_id: i32,
}

#[cfg(feature = "cuda")]
impl CudaDevice {
    pub fn count() -> Result<i32> {
        let mut count = 0i32;
        check_rc(
            unsafe { ffi::w34_get_device_count(&raw mut count) },
            "w34_get_device_count",
        )?;
        Ok(count)
    }

    pub fn set_current(device_id: i32) -> Result<Self> {
        check_rc(unsafe { ffi::w34_set_device(device_id) }, "w34_set_device")?;
        Ok(Self { device_id })
    }

    pub fn make_current(&self) -> Result<()> {
        check_rc(
            unsafe { ffi::w34_set_device(self.device_id) },
            "w34_set_device",
        )
    }

    pub fn synchronize(&self) -> Result<()> {
        self.make_current()?;
        check_rc(
            unsafe { ffi::w34_device_synchronize() },
            "w34_device_synchronize",
        )
    }

    pub fn mem_info(&self) -> Result<(usize, usize)> {
        self.make_current()?;
        let mut free = 0usize;
        let mut total = 0usize;
        check_rc(
            unsafe { ffi::w34_get_mem_info(&raw mut free, &raw mut total) },
            "w34_get_mem_info",
        )?;
        Ok((free, total))
    }
}

#[cfg(feature = "cuda")]
pub struct CudaStream {
    pub device_id: i32,
    pub handle: usize,
}

#[cfg(feature = "cuda")]
impl CudaStream {
    pub fn new(device_id: i32) -> Result<Self> {
        check_rc(unsafe { ffi::w34_set_device(device_id) }, "w34_set_device")?;
        let mut handle = 0usize;
        check_rc(
            unsafe { ffi::w34_stream_create(&raw mut handle) },
            "w34_stream_create",
        )?;
        Ok(Self { device_id, handle })
    }

    pub fn synchronize(&self) -> Result<()> {
        check_rc(
            unsafe { ffi::w34_set_device(self.device_id) },
            "w34_set_device",
        )?;
        check_rc(
            unsafe { ffi::w34_stream_synchronize(self.handle) },
            "w34_stream_synchronize",
        )
    }

    pub fn exec(&self, scratch_ptr: u64, scratch_size: usize) -> W34Exec {
        W34Exec {
            stream: self.handle,
            scratch_ptr,
            scratch_size,
        }
    }

    pub fn wait_event(&self, event: &CudaEvent) -> Result<()> {
        check_rc(
            unsafe { ffi::w34_stream_wait_event(self.handle, event.handle) },
            "w34_stream_wait_event",
        )
    }
}

#[cfg(feature = "cuda")]
impl Drop for CudaStream {
    fn drop(&mut self) {
        if self.handle != 0 {
            unsafe {
                let _ = ffi::w34_set_device(self.device_id);
                let _ = ffi::w34_stream_destroy(self.handle);
            }
        }
    }
}

#[cfg(feature = "cuda")]
pub struct CudaEvent {
    pub device_id: i32,
    pub handle: usize,
}

#[cfg(feature = "cuda")]
impl CudaEvent {
    pub fn new(device_id: i32, enable_timing: bool) -> Result<Self> {
        check_rc(unsafe { ffi::w34_set_device(device_id) }, "w34_set_device")?;
        let mut handle = 0usize;
        check_rc(
            unsafe { ffi::w34_event_create(&raw mut handle, i32::from(enable_timing)) },
            "w34_event_create",
        )?;
        Ok(Self { device_id, handle })
    }

    pub fn record(&self, stream: &CudaStream) -> Result<()> {
        check_rc(
            unsafe { ffi::w34_set_device(self.device_id) },
            "w34_set_device",
        )?;
        self.record_in_current_ctx(stream)
    }

    pub fn record_in_current_ctx(&self, stream: &CudaStream) -> Result<()> {
        check_rc(
            unsafe { ffi::w34_event_record(self.handle, stream.handle) },
            "w34_event_record",
        )
    }

    pub fn synchronize(&self) -> Result<()> {
        check_rc(
            unsafe { ffi::w34_event_synchronize(self.handle) },
            "w34_event_synchronize",
        )
    }

    pub fn elapsed_ms(&self, end: &CudaEvent) -> Result<f32> {
        let mut ms = 0.0f32;
        check_rc(
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
            unsafe { ffi::w34_set_device(stream.device_id) },
            "w34_set_device",
        )?;
        let mut handle = 0usize;
        check_rc(
            unsafe { ffi::w34_cublas_create(&raw mut handle, stream.handle) },
            "w34_cublas_create",
        )?;
        Ok(Self {
            device_id: stream.device_id,
            handle,
        })
    }

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
            unsafe {
                w34_kernels::w34_cublas_gemm_f16_bias(
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
            unsafe { ffi::w34_set_device(stream.device_id) },
            "w34_set_device",
        )?;
        check_rc(
            unsafe { ffi::w34_graph_begin_capture(stream.handle) },
            "w34_graph_begin_capture",
        )?;
        let body_res = body();
        let mut graph = 0usize;
        let mut graph_exec = 0usize;
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
            unsafe { ffi::w34_set_device(self.device_id) },
            "w34_set_device",
        )?;
        check_rc(
            unsafe { ffi::w34_graph_launch(self.graph_exec, stream.handle) },
            "w34_graph_launch",
        )
    }
}

#[cfg(feature = "cuda")]
impl Drop for CudaGraph {
    fn drop(&mut self) {
        if self.graph != 0 || self.graph_exec != 0 {
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

#[cfg(feature = "cuda")]
unsafe impl Send for DeviceBuffer {}
#[cfg(feature = "cuda")]
unsafe impl Sync for DeviceBuffer {}

#[cfg(feature = "cuda")]
impl DeviceBuffer {
    pub fn alloc(device_id: i32, len_bytes: usize) -> Result<Self> {
        check_rc(unsafe { ffi::w34_set_device(device_id) }, "w34_set_device")?;
        let alloc_bytes = len_bytes.max(16);
        let mut ptr: *mut c_void = core::ptr::null_mut();
        check_rc(
            unsafe { ffi::w34_malloc(&raw mut ptr, alloc_bytes) },
            "w34_malloc",
        )?;
        check_rc(
            unsafe { ffi::w34_memset_async(ptr, 0, alloc_bytes, 0) },
            "w34_memset_async",
        )?;
        check_rc(
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
            unsafe { ffi::w34_set_device(self.device_id) },
            "w34_set_device",
        )?;
        check_rc(
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
            unsafe { ffi::w34_set_device(self.device_id) },
            "w34_set_device",
        )?;
        check_rc(
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
            unsafe { ffi::w34_set_device(self.device_id) },
            "w34_set_device",
        )?;
        check_rc(
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
                unsafe { ffi::w34_set_device(self.buf.device_id) },
                "w34_set_device",
            )?;
            check_rc(
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

#[cfg(feature = "cuda")]
unsafe impl Send for MappedHostBuffer {}
#[cfg(feature = "cuda")]
unsafe impl Sync for MappedHostBuffer {}

#[cfg(feature = "cuda")]
impl MappedHostBuffer {
    pub fn alloc(len_bytes: usize) -> Result<Self> {
        let alloc_bytes = len_bytes.max(128);
        let mut host_ptr: *mut c_void = core::ptr::null_mut();
        let mut dev_ptr: *mut c_void = core::ptr::null_mut();
        check_rc(
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
            unsafe {
                let _ = ffi::w34_host_free(self.host_ptr);
            }
        }
    }
}

/// Pinned-sysmem `TP=2` collective context shared between rank 0 and rank 1.
#[cfg(feature = "cuda")]
pub struct Tp2Collectives {
    pub sig0: MappedHostBuffer,
    pub sig1: MappedHostBuffer,
    pub stage0: MappedHostBuffer,
    pub stage1: MappedHostBuffer,
    pub max_stage_bytes_per_epoch: usize,
}

#[cfg(feature = "cuda")]
impl Tp2Collectives {
    /// Allocates double-buffered staging (`2 * max_stage_bytes_per_epoch`) and signal structs.
    pub fn new(max_stage_bytes_per_epoch: usize) -> Result<Self> {
        let sig_bytes = unsafe { ffi::w34_tp2_signal_bytes() };
        let sig0 = MappedHostBuffer::alloc(sig_bytes)?;
        let sig1 = MappedHostBuffer::alloc(sig_bytes)?;
        let stage_bytes = max_stage_bytes_per_epoch.saturating_mul(2).max(4096);
        let stage0 = MappedHostBuffer::alloc(stage_bytes)?;
        let stage1 = MappedHostBuffer::alloc(stage_bytes)?;
        Ok(Self {
            sig0,
            sig1,
            stage0,
            stage1,
            max_stage_bytes_per_epoch,
        })
    }

    pub fn all_reduce_f16(
        &self,
        ex: &W34Exec,
        local_in_ptr: u64,
        out_ptr: u64,
        rank: i32,
        num_elems: i32,
    ) -> Result<()> {
        assert!(
            (num_elems as usize) * 2 <= self.max_stage_bytes_per_epoch,
            "all_reduce_f16 exceeds max_stage_bytes_per_epoch"
        );
        check_rc(
            unsafe {
                ffi::w34_tp2_all_reduce_f16(
                    ex,
                    local_in_ptr as *const c_void,
                    out_ptr as *mut c_void,
                    self.stage0.dev_ptr,
                    self.stage1.dev_ptr,
                    self.sig0.dev_ptr,
                    self.sig1.dev_ptr,
                    rank,
                    num_elems,
                )
            },
            "w34_tp2_all_reduce_f16",
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub fn all_reduce_residual_gemma_rms_norm_f16(
        &self,
        ex: &W34Exec,
        local_in_ptr: u64,
        ar_out_ptr: u64,
        residual_ptr: u64,
        weight_ptr: u64,
        out_ptr: u64,
        tap_dst_bf16_ptr: u64,
        rank: i32,
        num_tokens: i32,
        hidden_size: i32,
        eps: f32,
        tap_row_stride: i32,
        tap_col_offset: i32,
    ) -> Result<()> {
        let num_elems = (num_tokens as usize) * (hidden_size as usize);
        assert!(
            num_elems * 2 <= self.max_stage_bytes_per_epoch,
            "all_reduce_residual_gemma_rms_norm_f16 exceeds max_stage_bytes_per_epoch"
        );
        check_rc(
            unsafe {
                ffi::w34_tp2_all_reduce_residual_gemma_rms_norm_f16(
                    ex,
                    local_in_ptr as *const c_void,
                    ar_out_ptr as *mut c_void,
                    residual_ptr as *mut c_void,
                    weight_ptr as *const c_void,
                    out_ptr as *mut c_void,
                    tap_dst_bf16_ptr as *mut c_void,
                    self.stage0.dev_ptr,
                    self.stage1.dev_ptr,
                    self.sig0.dev_ptr,
                    self.sig1.dev_ptr,
                    rank,
                    num_tokens,
                    hidden_size,
                    eps,
                    tap_row_stride,
                    tap_col_offset,
                )
            },
            "w34_tp2_all_reduce_residual_gemma_rms_norm_f16",
        )
    }

    pub fn all_reduce_bf16(
        &self,
        ex: &W34Exec,
        local_in_ptr: u64,
        out_ptr: u64,
        rank: i32,
        num_elems: i32,
    ) -> Result<()> {
        assert!(
            (num_elems as usize) * 2 <= self.max_stage_bytes_per_epoch,
            "all_reduce_bf16 exceeds max_stage_bytes_per_epoch"
        );
        check_rc(
            unsafe {
                ffi::w34_tp2_all_reduce_bf16(
                    ex,
                    local_in_ptr as *const c_void,
                    out_ptr as *mut c_void,
                    self.stage0.dev_ptr,
                    self.stage1.dev_ptr,
                    self.sig0.dev_ptr,
                    self.sig1.dev_ptr,
                    rank,
                    num_elems,
                )
            },
            "w34_tp2_all_reduce_bf16",
        )
    }

    pub fn all_reduce_f32(
        &self,
        ex: &W34Exec,
        local_in_ptr: u64,
        out_ptr: u64,
        rank: i32,
        num_elems: i32,
    ) -> Result<()> {
        assert!(
            (num_elems as usize) * 4 <= self.max_stage_bytes_per_epoch,
            "all_reduce_f32 exceeds max_stage_bytes_per_epoch"
        );
        check_rc(
            unsafe {
                ffi::w34_tp2_all_reduce_f32(
                    ex,
                    local_in_ptr as *const f32,
                    out_ptr as *mut f32,
                    self.stage0.dev_ptr as *mut f32,
                    self.stage1.dev_ptr as *mut f32,
                    self.sig0.dev_ptr,
                    self.sig1.dev_ptr,
                    rank,
                    num_elems,
                )
            },
            "w34_tp2_all_reduce_f32",
        )
    }

    pub fn broadcast_i32(
        &self,
        ex: &W34Exec,
        src_rank0_ptr: u64,
        out_ptr: u64,
        rank: i32,
        num_elems: i32,
    ) -> Result<()> {
        assert!(
            (num_elems as usize) * 4 <= self.max_stage_bytes_per_epoch,
            "broadcast_i32 exceeds max_stage_bytes_per_epoch"
        );
        check_rc(
            unsafe {
                ffi::w34_tp2_broadcast_i32(
                    ex,
                    src_rank0_ptr as *const i32,
                    out_ptr as *mut i32,
                    self.stage0.dev_ptr as *mut i32,
                    self.stage1.dev_ptr as *mut i32,
                    self.sig0.dev_ptr,
                    self.sig1.dev_ptr,
                    rank,
                    num_elems,
                )
            },
            "w34_tp2_broadcast_i32",
        )
    }

    pub fn all_gather_top1_f32_i32(
        &self,
        ex: &W34Exec,
        local_max_val_ptr: u64,
        local_max_id_ptr: u64,
        global_max_val_ptr: u64,
        global_max_id_ptr: u64,
        rank: i32,
        num_rows: i32,
    ) -> Result<()> {
        assert!(
            (num_rows as usize) * 8 <= self.max_stage_bytes_per_epoch,
            "all_gather_top1_f32_i32 exceeds max_stage_bytes_per_epoch"
        );
        check_rc(
            unsafe {
                ffi::w34_tp2_all_gather_top1_f32_i32(
                    ex,
                    local_max_val_ptr as *const f32,
                    local_max_id_ptr as *const i32,
                    global_max_val_ptr as *mut f32,
                    global_max_id_ptr as *mut i32,
                    self.stage0.dev_ptr,
                    self.stage1.dev_ptr,
                    self.sig0.dev_ptr,
                    self.sig1.dev_ptr,
                    rank,
                    num_rows,
                )
            },
            "w34_tp2_all_gather_top1_f32_i32",
        )
    }

    pub fn gather_vocab_f16(
        &self,
        ex: &W34Exec,
        local_logits_ptr: u64,
        rank0_full_logits_ptr: u64,
        rank: i32,
        num_rows: i32,
        local_vocab_size: i32,
    ) -> Result<()> {
        assert!(
            (num_rows as usize) * (local_vocab_size as usize) * 2 <= self.max_stage_bytes_per_epoch,
            "gather_vocab_f16 exceeds max_stage_bytes_per_epoch"
        );
        check_rc(
            unsafe {
                ffi::w34_tp2_gather_vocab_f16(
                    ex,
                    local_logits_ptr as *const c_void,
                    rank0_full_logits_ptr as *mut c_void,
                    self.stage0.dev_ptr,
                    self.stage1.dev_ptr,
                    self.sig0.dev_ptr,
                    self.sig1.dev_ptr,
                    rank,
                    num_rows,
                    local_vocab_size,
                )
            },
            "w34_tp2_gather_vocab_f16",
        )
    }
}

/// Pre-warms every kernel specialization on `device_id` and synchronizes the device.
///
/// Mandatory under `docs/contracts/execution-contract.md` and `docs/contracts/collectives.md`
/// before any `TP=2` collective barrier or `cudaStreamBeginCapture` to prevent CUDA 13
/// lazy-module-loading deadlocks.
#[cfg(feature = "cuda")]
pub fn prepare_execution_plan(device_id: i32) -> Result<()> {
    let dev = CudaDevice::set_current(device_id)?;
    let stream = CudaStream::new(device_id)?;
    let cublas = CublasHandle::new(&stream)?;
    let mut arena = DeviceArena::new(device_id, 16 * 1024 * 1024)?;
    let ex = stream.exec(arena.buf.addr(), arena.capacity());

    let buf_a = arena.alloc_aligned(64 * 1024)?;
    let buf_b = arena.alloc_aligned(64 * 1024)?;
    let buf_c = arena.alloc_aligned(64 * 1024)?;
    let buf_d = arena.alloc_aligned(64 * 1024)?;
    let buf_ws = arena.alloc_aligned(64 * 1024)?;

    // 1. Norms, RoPE, activations
    check_rc(
        unsafe {
            ffi::w34_embed_lookup_f16(
                &ex,
                buf_a as *const i32,
                buf_b as *const c_void,
                buf_c as *mut c_void,
                core::ptr::null(),
                1,
                128,
                256,
            )
        },
        "warmup w34_embed_lookup_f16",
    )?;
    for hs in [128i32, 3072, 5120, 6144] {
        for byte_off in [0u64, 4u64] {
            check_rc(
                unsafe {
                    ffi::w34_gemma_rms_norm_f16(
                        &ex,
                        (buf_a + byte_off) as *const c_void,
                        (buf_b + byte_off) as *const c_void,
                        (buf_c + byte_off) as *mut c_void,
                        core::ptr::null(),
                        1,
                        hs,
                        1e-6,
                    )
                },
                "warmup w34_gemma_rms_norm_f16",
            )?;
            check_rc(
                unsafe {
                    ffi::w34_fused_add_gemma_rms_norm_f16(
                        &ex,
                        (buf_a + byte_off) as *const c_void,
                        (buf_b + byte_off) as *mut c_void,
                        (buf_c + byte_off) as *const c_void,
                        (buf_d + byte_off) as *mut c_void,
                        core::ptr::null(),
                        1,
                        hs,
                        1e-6,
                    )
                },
                "warmup w34_fused_add_gemma_rms_norm_f16",
            )?;
        }
    }
    check_rc(
        unsafe {
            ffi::w34_rms_norm_gated_silu_f16(
                &ex,
                buf_a as *const c_void,
                buf_b as *const c_void,
                buf_c as *const c_void,
                buf_d as *mut c_void,
                core::ptr::null(),
                1,
                2,
                128,
                1e-6,
            )
        },
        "warmup w34_rms_norm_gated_silu_f16",
    )?;
    check_rc(
        unsafe {
            ffi::w34_fused_qk_rmsnorm_rope_gate_f16(
                &ex,
                buf_a as *const c_void,
                buf_b as *const c_void,
                buf_c as *const c_void,
                buf_c as *const c_void,
                buf_ws as *const i32,
                core::ptr::null(),
                buf_d as *mut c_void,
                buf_d as *mut c_void,
                buf_d as *mut c_void,
                1,
                1,
                2,
                1,
                256,
                64,
                11,
                11,
                10,
                10_000_000.0,
                1e-6,
            )
        },
        "warmup w34_fused_qk_rmsnorm_rope_gate_f16",
    )?;
    for d_warm in [128i32, 132] {
        check_rc(
            unsafe {
                ffi::w34_silu_and_mul_f16(
                    &ex,
                    buf_a as *const c_void,
                    buf_c as *mut c_void,
                    core::ptr::null(),
                    1,
                    d_warm,
                )
            },
            "warmup w34_silu_and_mul_f16",
        )?;
    }
    check_rc(
        unsafe {
            ffi::w34_apply_attn_output_gate_f16(
                &ex,
                buf_a as *const c_void,
                buf_b as *const c_void,
                buf_c as *mut c_void,
                core::ptr::null(),
                1,
                256,
            )
        },
        "warmup w34_apply_attn_output_gate_f16",
    )?;

    // 2. Marlin repack & GEMM + cuBLAS GEMM
    check_rc(
        unsafe {
            ffi::w34_marlin_repack_int4(&ex, buf_a as *const u32, buf_b as *mut u32, 128, 64)
        },
        "warmup w34_marlin_repack_int4",
    )?;
    check_rc(
        unsafe {
            ffi::w34_marlin_repack_int8(&ex, buf_a as *const u32, buf_b as *mut u32, 128, 64)
        },
        "warmup w34_marlin_repack_int8",
    )?;
    for m_warm in [1i64, 16, 40, 64] {
        for group_blocks in [8i32, 4, 2] {
            check_rc(
                unsafe {
                    ffi::w34_marlin_gemm_f16u4(
                        &ex,
                        buf_a as *const c_void,
                        buf_b as *const c_void,
                        buf_c as *const c_void,
                        buf_d as *mut c_void,
                        buf_ws as *mut i32,
                        m_warm,
                        64,
                        128,
                        2048,
                        group_blocks,
                    )
                },
                "warmup w34_marlin_gemm_f16u4",
            )?;
            check_rc(
                unsafe {
                    ffi::w34_marlin_gemm_f16u4_zp(
                        &ex,
                        buf_a as *const c_void,
                        buf_b as *const c_void,
                        buf_c as *const c_void,
                        buf_c as *const c_void,
                        buf_d as *mut c_void,
                        buf_ws as *mut i32,
                        m_warm,
                        64,
                        128,
                        2048,
                        group_blocks,
                    )
                },
                "warmup w34_marlin_gemm_f16u4_zp",
            )?;
        }
        for group_blocks in [8i32, 4, 2, -1] {
            check_rc(
                unsafe {
                    ffi::w34_marlin_gemm_f16u8(
                        &ex,
                        buf_a as *const c_void,
                        buf_b as *const c_void,
                        buf_c as *const c_void,
                        buf_d as *mut c_void,
                        buf_ws as *mut i32,
                        m_warm,
                        64,
                        128,
                        2048,
                        group_blocks,
                    )
                },
                "warmup w34_marlin_gemm_f16u8",
            )?;
        }
    }
    cublas.gemm_f16(buf_a, buf_b, buf_c, 1, 64, 128)?;
    // Mandatory lazy-module warmup before TP workers/capture.
    cublas.gemm_f16_bias(buf_a, buf_b, buf_d, buf_c, 1, 64, 128)?;
    cublas.gemm_bf16(buf_a, buf_b, buf_c, 1, 64, 128)?;
    for m_warm in [1i32, 4, 20] {
        check_rc(
            unsafe {
                ffi::w34_small_n_gemm_f16(
                    &ex,
                    buf_a as *const c_void,
                    buf_b as *const c_void,
                    buf_c as *mut c_void,
                    m_warm,
                    64,
                    128,
                )
            },
            "warmup w34_small_n_gemm_f16",
        )?;
        check_rc(
            unsafe {
                ffi::w34_small_n_gemm_dual_f16(
                    &ex,
                    buf_a as *const c_void,
                    buf_b as *const c_void,
                    buf_c as *mut c_void,
                    64,
                    buf_b as *const c_void,
                    buf_d as *mut c_void,
                    1,
                    m_warm,
                    128,
                )
            },
            "warmup w34_small_n_gemm_dual_f16",
        )?;
    }
    for hs in [128i32, 3072, 5120, 6144] {
        let sig_u32_len = (unsafe { ffi::w34_tp2_signal_bytes() } as usize).div_ceil(4);
        let mut fake_sig = vec![0u32; sig_u32_len.max(1216)];
        fake_sig[1] = 1; // pre-satisfy start[0][1] == epoch(1) for single-GPU warmup
        let sig_dev = arena.upload_slice(&fake_sig, &stream)?;
        check_rc(
            unsafe {
                ffi::w34_tp2_all_reduce_residual_gemma_rms_norm_f16(
                    &ex,
                    buf_a as *const c_void,
                    core::ptr::null_mut(),
                    buf_b as *mut c_void,
                    buf_c as *const c_void,
                    buf_d as *mut c_void,
                    core::ptr::null_mut(),
                    buf_ws as *mut c_void,
                    buf_ws as *mut c_void,
                    sig_dev as *mut c_void,
                    sig_dev as *mut c_void,
                    0,
                    1,
                    hs,
                    1e-6,
                    0,
                    0,
                )
            },
            "warmup w34_tp2_all_reduce_residual_gemma_rms_norm_f16",
        )?;
    }

    // 3. KV write & Paged Attention
    let q_start = arena.upload_slice(&[0i32, 1], &stream)?;
    let ctx_lens = arena.upload_slice(&[0i32], &stream)?;
    let seq_lens = arena.upload_slice(&[1i32], &stream)?;
    let blk_tbl = arena.upload_slice(&[0i32], &stream)?;
    let slot_map = arena.upload_slice(&[0i32], &stream)?;
    let layout = W34KvLayerLayout {
        k_base: buf_a,
        v_base: buf_b,
        k_page_stride_bytes: 16 * 256 * 2,
        k_head_stride_bytes: 16 * 256 * 2,
        k_token_stride_bytes: 256 * 2,
        v_page_stride_bytes: 16 * 256 * 2,
        v_head_stride_bytes: 16 * 256 * 2,
        v_token_stride_bytes: 256 * 2,
        namespace_id: ffi::W34_KV_NAMESPACE_TARGET_FULL_ATTN,
        layer_id: 0,
        local_kv_heads: 1,
        head_dim: 256,
        block_size: 16,
        k_format: ffi::W34_KV_FORMAT_FP16,
        v_format: ffi::W34_KV_FORMAT_FP16,
        k_transform: ffi::W34_KV_TRANSFORM_IDENTITY,
        v_transform: ffi::W34_KV_TRANSFORM_IDENTITY,
        reserved: 0,
    };
    let attn_batch = W34AttentionBatch {
        query_start_loc: q_start as *const i32,
        context_lens_before: ctx_lens as *const i32,
        seq_lens_after: seq_lens as *const i32,
        block_tables: blk_tbl as *const i32,
        write_slot_mapping: slot_map as *const i32,
        valid_token_mask: core::ptr::null(),
        valid_req_mask: core::ptr::null(),
        num_reqs: 1,
        total_tokens: 1,
        max_blocks_per_req: 1,
        max_q_len: 1,
        max_seq_len: 1,
        mask_mode: ffi::W34_ATTN_MASK_CAUSAL,
        sliding_window: 0,
        sm_scale: 0.0625,
        reserved: 0,
    };
    check_rc(
        unsafe {
            ffi::w34_kv_cache_write(
                &ex,
                &layout,
                &attn_batch,
                buf_c as *const c_void,
                buf_c as *const c_void,
            )
        },
        "warmup w34_kv_cache_write",
    )?;
    check_rc(
        unsafe {
            ffi::w34_paged_attention_f16(
                &ex,
                &layout,
                &attn_batch,
                buf_c as *const c_void,
                buf_d as *mut c_void,
                2,
            )
        },
        "warmup w34_paged_attention_f16",
    )?;
    // Exercise the eligible paged FA2 branch on this selected device before capture.
    // Dedicated zeroed inputs and LSE must not alias the generic arena workspace.
    let fa_q = arena.upload_slice(&vec![0u16; 17 * 2 * 256], &stream)?;
    let fa_k = arena.upload_slice(&vec![0u16; 32 * 256], &stream)?;
    let fa_v = arena.upload_slice(&vec![0u16; 32 * 256], &stream)?;
    let fa_o = arena.alloc_aligned(17 * 2 * 256 * 2)?;
    let fa_ws = arena.alloc_aligned(17 * 2 * 4)?;
    let fa_start = arena.upload_slice(&[0i32, 17], &stream)?;
    let fa_seq = arena.upload_slice(&[17i32], &stream)?;
    let fa_pages = arena.upload_slice(&[1i32, 0], &stream)?;
    let fa_layout = W34KvLayerLayout {
        k_base: fa_k,
        v_base: fa_v,
        ..layout
    };
    let fa_batch = W34AttentionBatch {
        query_start_loc: fa_start as *const i32,
        seq_lens_after: fa_seq as *const i32,
        block_tables: fa_pages as *const i32,
        total_tokens: 17,
        max_q_len: 17,
        max_seq_len: 17,
        max_blocks_per_req: 2,
        ..attn_batch
    };
    check_rc(
        unsafe {
            ffi::w34_paged_attention_f16(
                &stream.exec(fa_ws, 17 * 2 * 4),
                &fa_layout,
                &fa_batch,
                fa_q as *const c_void,
                fa_o as *mut c_void,
                2,
            )
        },
        "warmup w34_paged_attention_f16 FA2",
    )?;

    // Warm 64-query-row TQ4 FlashAttention-2 prefill specialization before TP capture.
    let fa64_q = arena.upload_slice(&vec![0u16; 64 * 8 * 256], &stream)?;
    let fa64_k = arena.upload_slice(&vec![0u8; 4 * 4 * 16 * 132], &stream)?;
    let fa64_v = arena.upload_slice(&vec![0u8; 4 * 4 * 16 * 132], &stream)?;
    let fa64_o = arena.alloc_aligned(64 * 8 * 256 * 2)?;
    let fa64_start = arena.upload_slice(&[0i32, 64], &stream)?;
    let fa64_seq = arena.upload_slice(&[64i32], &stream)?;
    let fa64_pages = arena.upload_slice(&[0i32, 1, 2, 3], &stream)?;
    let fa64_layout = W34KvLayerLayout {
        k_base: fa64_k,
        v_base: fa64_v,
        k_token_stride_bytes: 132,
        v_token_stride_bytes: 132,
        k_head_stride_bytes: 16 * 132,
        v_head_stride_bytes: 16 * 132,
        k_page_stride_bytes: 4 * 16 * 132,
        v_page_stride_bytes: 4 * 16 * 132,
        local_kv_heads: 4,
        k_format: ffi::W34_KV_FORMAT_TQ4,
        v_format: ffi::W34_KV_FORMAT_TQ4,
        k_transform: ffi::W34_KV_TRANSFORM_RADEMACHER_WHT256,
        v_transform: ffi::W34_KV_TRANSFORM_RADEMACHER_WHT256,
        ..layout
    };
    let fa64_batch = W34AttentionBatch {
        query_start_loc: fa64_start as *const i32,
        seq_lens_after: fa64_seq as *const i32,
        block_tables: fa64_pages as *const i32,
        total_tokens: 64,
        max_q_len: 64,
        max_seq_len: 64,
        max_blocks_per_req: 4,
        ..attn_batch
    };
    for variant in [1, 2] {
        check_rc(
            unsafe {
                ffi::w34_paged_attention_f16_force_kernel_for_test(
                    &ex,
                    &fa64_layout,
                    &fa64_batch,
                    fa64_q as *const c_void,
                    fa64_o as *mut c_void,
                    8,
                    variant,
                )
            },
            "warmup w34_paged_attention_prefill_mma_f16",
        )?;
    }

    // Warm GQA mma.sync + Split-KV reduction specialization (head_dim = 256)
    let layout_256 = W34KvLayerLayout {
        head_dim: 256,
        k_token_stride_bytes: 512,
        v_token_stride_bytes: 512,
        k_head_stride_bytes: 16 * 512,
        v_head_stride_bytes: 16 * 512,
        k_page_stride_bytes: 16 * 512,
        v_page_stride_bytes: 16 * 512,
        ..layout
    };
    let blk_tbl_128 = arena.upload_slice(&[0i32; 128], &stream)?;
    let seq_lens_2048 = arena.upload_slice(&[2048i32], &stream)?;
    let q_start_2 = arena.upload_slice(&[0i32, 2], &stream)?;
    let slot_map_2 = arena.upload_slice(&[0i32, 1], &stream)?;
    let mut attn_batch_m2 = attn_batch;
    attn_batch_m2.query_start_loc = q_start_2 as *const i32;
    attn_batch_m2.write_slot_mapping = slot_map_2 as *const i32;
    attn_batch_m2.total_tokens = 2;
    attn_batch_m2.max_q_len = 2;
    attn_batch_m2.seq_lens_after = seq_lens_2048 as *const i32;
    attn_batch_m2.block_tables = blk_tbl_128 as *const i32;
    attn_batch_m2.max_blocks_per_req = 128;
    attn_batch_m2.max_seq_len = 2048;
    let mut attn_batch_split = attn_batch;
    attn_batch_split.block_tables = blk_tbl_128 as *const i32;
    attn_batch_split.seq_lens_after = seq_lens_2048 as *const i32;
    attn_batch_split.max_blocks_per_req = 128;
    attn_batch_split.max_seq_len = 2048;
    let split_ws = arena.alloc_aligned(512 * 1024)?;
    let ex_split = stream.exec(split_ws, 512 * 1024);
    for q_heads in [4i32, 6, 8] {
        check_rc(
            unsafe {
                ffi::w34_paged_attention_f16(
                    &ex,
                    &layout_256,
                    &attn_batch,
                    buf_c as *const c_void,
                    buf_d as *mut c_void,
                    q_heads,
                )
            },
            "warmup w34_paged_attention_f16 gqa_vec_m1",
        )?;
        check_rc(
            unsafe {
                ffi::w34_paged_attention_f16(
                    &ex_split,
                    &layout_256,
                    &attn_batch_m2,
                    buf_c as *const c_void,
                    buf_d as *mut c_void,
                    q_heads,
                )
            },
            "warmup w34_paged_attention_f16 gqa_mma_split_kv",
        )?;
        check_rc(
            unsafe {
                ffi::w34_paged_attention_f16(
                    &ex_split,
                    &layout_256,
                    &attn_batch_split,
                    buf_c as *const c_void,
                    buf_d as *mut c_void,
                    q_heads,
                )
            },
            "warmup w34_paged_attention_f16 split_kv",
        )?;
    }
    let tq4_buf = arena.upload_slice(&vec![0u8; 16 * 132], &stream)?;
    let layout_256_tq4 = W34KvLayerLayout {
        k_base: tq4_buf,
        v_base: tq4_buf,
        k_token_stride_bytes: 132,
        v_token_stride_bytes: 132,
        k_head_stride_bytes: 16 * 132,
        v_head_stride_bytes: 16 * 132,
        k_page_stride_bytes: 16 * 132,
        v_page_stride_bytes: 16 * 132,
        k_format: ffi::W34_KV_FORMAT_TQ4,
        v_format: ffi::W34_KV_FORMAT_TQ4,
        k_transform: ffi::W34_KV_TRANSFORM_RADEMACHER_WHT256,
        v_transform: ffi::W34_KV_TRANSFORM_RADEMACHER_WHT256,
        ..layout_256
    };
    for q_heads in [4i32, 6, 8] {
        check_rc(
            unsafe {
                ffi::w34_paged_attention_f16(
                    &ex,
                    &layout_256_tq4,
                    &attn_batch,
                    buf_c as *const c_void,
                    buf_d as *mut c_void,
                    q_heads,
                )
            },
            "warmup w34_paged_attention_f16 tq4 gqa_vec_m1",
        )?;
        check_rc(
            unsafe {
                ffi::w34_paged_attention_f16(
                    &ex_split,
                    &layout_256_tq4,
                    &attn_batch_m2,
                    buf_c as *const c_void,
                    buf_d as *mut c_void,
                    q_heads,
                )
            },
            "warmup w34_paged_attention_f16 tq4 gqa_mma_split_kv",
        )?;
    }

    let bf16_layout = W34KvLayerLayout {
        namespace_id: ffi::W34_KV_NAMESPACE_DFLASH_DRAFT_CTX,
        k_format: ffi::W34_KV_FORMAT_BF16,
        v_format: ffi::W34_KV_FORMAT_BF16,
        ..layout
    };
    check_rc(
        unsafe {
            ffi::w34_kv_cache_write_bf16(
                &ex,
                &bf16_layout,
                &attn_batch,
                buf_c as *const c_void,
                buf_c as *const c_void,
            )
        },
        "warmup w34_kv_cache_write_bf16",
    )?;
    check_rc(
        unsafe {
            ffi::w34_paged_attention_bf16(
                &ex,
                &bf16_layout,
                &attn_batch,
                buf_c as *const c_void,
                buf_d as *mut c_void,
                2,
            )
        },
        "warmup w34_paged_attention_bf16 hd128",
    )?;
    let bf16_layout_256 = W34KvLayerLayout {
        namespace_id: ffi::W34_KV_NAMESPACE_DFLASH_DRAFT_CTX,
        k_format: ffi::W34_KV_FORMAT_BF16,
        v_format: ffi::W34_KV_FORMAT_BF16,
        ..layout_256
    };
    check_rc(
        unsafe {
            ffi::w34_paged_attention_bf16(
                &ex,
                &bf16_layout_256,
                &attn_batch,
                buf_c as *const c_void,
                buf_d as *mut c_void,
                2,
            )
        },
        "warmup w34_paged_attention_bf16 hd256",
    )?;
    check_rc(
        unsafe {
            ffi::w34_paged_attention_bf16(
                &ex,
                &bf16_layout,
                &attn_batch,
                (buf_c + 4) as *const c_void,
                buf_d as *mut c_void,
                2,
            )
        },
        "warmup w34_paged_attention_bf16 fallback",
    )?;

    // 4. GDN conv1d & recurrent
    let state_in = arena.upload_slice(&[0i32], &stream)?;
    let state_out = arena.upload_slice(&[0i32], &stream)?;
    let has_init = arena.upload_slice(&[0u8], &stream)?;
    let gdn_batch = W34GdnBatch {
        query_start_loc: q_start as *const i32,
        state_in_indices: state_in as *const i32,
        state_out_indices: state_out as *const i32,
        spec_state_indices: core::ptr::null(),
        checkpoint_dst_indices: core::ptr::null(),
        checkpoint_token_offsets: core::ptr::null(),
        has_initial_state: has_init as *const u8,
        valid_req_mask: core::ptr::null(),
        num_reqs: 1,
        total_tokens: 1,
        max_verify_tokens: 0,
        max_checkpoints_per_req: 0,
    };
    check_rc(
        unsafe {
            ffi::w34_gdn_causal_conv1d_f16(
                &ex,
                &gdn_batch,
                buf_a as *const c_void,
                buf_b as *const c_void,
                core::ptr::null(),
                buf_c as *mut c_void,
                buf_d as *mut c_void,
                384,
                4,
                1,
                1,
            )
        },
        "warmup w34_gdn_causal_conv1d_f16",
    )?;
    check_rc(
        unsafe {
            ffi::w34_gdn_recurrent_f16(
                &ex,
                &gdn_batch,
                buf_d as *const c_void,
                core::ptr::null(),
                core::ptr::null(),
                core::ptr::null(),
                buf_a as *const c_void,
                buf_a as *const c_void,
                buf_b as *const f32,
                buf_b as *const f32,
                buf_ws as *mut f32,
                buf_c as *mut c_void,
                1,
                128,
                1,
                128,
                1,
            )
        },
        "warmup w34_gdn_recurrent_f16 decode",
    )?;
    let gdn_ws_wide = arena.alloc_aligned(2 * 1024 * 1024)?;
    check_rc(
        unsafe {
            ffi::w34_gdn_recurrent_f16(
                &ex,
                &gdn_batch,
                buf_d as *const c_void,
                core::ptr::null(),
                core::ptr::null(),
                core::ptr::null(),
                buf_a as *const c_void,
                buf_a as *const c_void,
                buf_b as *const f32,
                buf_b as *const f32,
                gdn_ws_wide as *mut f32,
                buf_c as *mut c_void,
                1,
                128,
                96,
                32,
                1,
            )
        },
        "warmup w34_gdn_recurrent_f16 decode wide",
    )?;
    let q_start_prefill = arena.upload_slice(&[0i32, 3], &stream)?;
    let gdn_batch_prefill = W34GdnBatch {
        query_start_loc: q_start_prefill as *const i32,
        total_tokens: 3,
        ..gdn_batch
    };
    check_rc(
        unsafe {
            ffi::w34_gdn_recurrent_f16(
                &ex,
                &gdn_batch_prefill,
                buf_d as *const c_void,
                core::ptr::null(),
                core::ptr::null(),
                core::ptr::null(),
                buf_a as *const c_void,
                buf_a as *const c_void,
                buf_b as *const f32,
                buf_b as *const f32,
                buf_ws as *mut f32,
                buf_c as *mut c_void,
                1,
                128,
                1,
                128,
                1,
            )
        },
        "warmup w34_gdn_recurrent_f16 prefill",
    )?;
    let q_start_wy = arena.upload_slice(&[0i32, 256], &stream)?;
    let gdn_batch_wy = W34GdnBatch {
        query_start_loc: q_start_wy as *const i32,
        total_tokens: 256,
        ..gdn_batch
    };
    check_rc(
        unsafe {
            ffi::w34_gdn_causal_conv1d_f16(
                &ex,
                &gdn_batch_wy,
                buf_a as *const c_void,
                buf_b as *const c_void,
                core::ptr::null(),
                buf_c as *mut c_void,
                buf_d as *mut c_void,
                8,
                4,
                1,
                1,
            )
        },
        "warmup w34_gdn_causal_conv1d_f16 paired prefill",
    )?;
    let ex_wy_split = stream.exec(gdn_ws_wide + 1024 * 1024, 1024 * 1024);
    for ex_wy in [&ex, &ex_wy_split] {
        check_rc(
            unsafe {
                ffi::w34_gdn_recurrent_f16(
                    ex_wy,
                    &gdn_batch_wy,
                    gdn_ws_wide as *const c_void,
                    core::ptr::null(),
                    core::ptr::null(),
                    core::ptr::null(),
                    gdn_ws_wide as *const c_void,
                    gdn_ws_wide as *const c_void,
                    buf_b as *const f32,
                    buf_b as *const f32,
                    buf_ws as *mut f32,
                    (gdn_ws_wide + 512 * 1024) as *mut c_void,
                    1,
                    128,
                    1,
                    128,
                    1,
                )
            },
            "warmup w34_gdn_recurrent_f16 wy prefill",
        )?;
    }
    for (hv_warm, dv_warm) in [(96i32, 32i32), (24, 128), (32, 128)] {
        check_rc(
            unsafe {
                ffi::w34_gdn_recurrent_f16(
                    &ex,
                    &gdn_batch_prefill,
                    buf_d as *const c_void,
                    core::ptr::null(),
                    core::ptr::null(),
                    core::ptr::null(),
                    buf_a as *const c_void,
                    buf_a as *const c_void,
                    buf_b as *const f32,
                    buf_b as *const f32,
                    gdn_ws_wide as *mut f32,
                    buf_c as *mut c_void,
                    1,
                    128,
                    hv_warm,
                    dv_warm,
                    1,
                )
            },
            "warmup w34_gdn_recurrent_f16 prefill wide",
        )?;
    }

    // 5. Sampler & Phase 4 MTP / DFlash2 kernels
    check_rc(
        unsafe { ffi::w34_tp2_warmup() },
        "warmup op_collectives module",
    )?;
    check_rc(
        unsafe {
            ffi::w34_sampler_greedy_top5_f16(
                &ex,
                buf_a as *const c_void,
                buf_b as *mut i32,
                buf_c as *mut i32,
                buf_d as *mut f32,
                1,
                256,
                0,
            )
        },
        "warmup w34_sampler_greedy_top5_f16",
    )?;
    let sampler_cfg = ffi::W34SamplerRowCfg {
        temperature: 1.0,
        top_k: 8,
        top_p: 0.9,
        min_p: 0.05,
        repetition_penalty: 1.1,
        frequency_penalty: 0.1,
        presence_penalty: 0.1,
        uniform_sample: 0.5,
        recovery_uniform: 0.5,
        draft_token_id: -1,
        has_bitmask: 0,
        row_bitmask_word_offset: 0,
        sparse_offset: 0,
        sparse_count: 1,
        top_logprobs_k: 5,
        use_token_id_cdf: 0,
    };
    let sampler_sparse = ffi::W34SamplerSparseEntry {
        token_id: 0,
        output_count: 1,
        logit_bias: 0.0,
        in_prompt: 1,
        banned: 0,
    };
    let sampler_cfg_dev = arena.upload_slice(&[sampler_cfg], &stream)?;
    let sampler_sparse_dev = arena.upload_slice(&[sampler_sparse], &stream)?;
    let sampler_out_dev = arena.alloc_aligned(core::mem::size_of::<ffi::W34SamplerRowOut>())?;
    let sampler_ws = arena.alloc_aligned(64 * 1024)?;
    let ex_sampler = stream.exec(sampler_ws, 64 * 1024);
    check_rc(
        unsafe {
            ffi::w34_sampler_step_f16(
                &ex_sampler,
                buf_a as *const c_void,
                sampler_cfg_dev as *const ffi::W34SamplerRowCfg,
                core::ptr::null(),
                sampler_sparse_dev as *const ffi::W34SamplerSparseEntry,
                sampler_out_dev as *mut ffi::W34SamplerRowOut,
                1,
                256,
                1,
            )
        },
        "warmup w34_sampler_step_f16",
    )?;
    let greedy_sampler_cfg = ffi::W34SamplerRowCfg {
        temperature: 0.0,
        top_k: 1,
        sparse_count: 1,
        top_logprobs_k: 0,
        ..sampler_cfg
    };
    let greedy_sampler_cfg_dev = arena.upload_slice(&[greedy_sampler_cfg], &stream)?;
    check_rc(
        unsafe {
            ffi::w34_sampler_step_fast_greedy_f16(
                &ex_sampler,
                buf_a as *const c_void,
                greedy_sampler_cfg_dev as *const ffi::W34SamplerRowCfg,
                sampler_sparse_dev as *const ffi::W34SamplerSparseEntry,
                sampler_out_dev as *mut ffi::W34SamplerRowOut,
                1,
                1024,
            )
        },
        "warmup w34_sampler_step_fast_greedy_f16",
    )?;
    let plain_greedy_cfg = ffi::W34SamplerRowCfg {
        sparse_count: 0,
        ..greedy_sampler_cfg
    };
    let plain_greedy_cfg_dev = arena.upload_slice(&[plain_greedy_cfg], &stream)?;
    let masked_greedy_cfg = ffi::W34SamplerRowCfg {
        has_bitmask: 1,
        row_bitmask_word_offset: 0,
        sparse_count: 0,
        top_logprobs_k: -1,
        ..greedy_sampler_cfg
    };
    let masked_greedy_cfg_dev = arena.upload_slice(&[masked_greedy_cfg], &stream)?;
    let masked_words_dev = arena.upload_slice(&[u32::MAX; 32], &stream)?;
    check_rc(
        unsafe {
            ffi::w34_sampler_step_fast_greedy_masked_f16(
                &ex_sampler,
                buf_a as *const c_void,
                masked_greedy_cfg_dev as *const ffi::W34SamplerRowCfg,
                masked_words_dev as *const u32,
                core::ptr::null(),
                sampler_out_dev as *mut ffi::W34SamplerRowOut,
                1,
                1024,
            )
        },
        "warmup w34_sampler_step_fast_greedy_masked_f16",
    )?;
    check_rc(
        unsafe {
            ffi::w34_sampler_step_f16(
                &ex_sampler,
                buf_a as *const c_void,
                plain_greedy_cfg_dev as *const ffi::W34SamplerRowCfg,
                core::ptr::null(),
                core::ptr::null(),
                sampler_out_dev as *mut ffi::W34SamplerRowOut,
                1,
                1024,
                0,
            )
        },
        "warmup w34_sampler_step_f16 fast_greedy",
    )?;
    check_rc(
        unsafe {
            ffi::w34_sampler_scatter_ids_to_tokens(
                &ex_sampler,
                sampler_out_dev as *const ffi::W34SamplerRowOut,
                core::ptr::null(),
                buf_b as *mut i32,
                1,
            )
        },
        "warmup w34_sampler_scatter_ids_to_tokens",
    )?;
    check_rc(
        unsafe {
            ffi::w34_concat_cols_2_f16(
                &ex,
                buf_a as *const c_void,
                buf_b as *const c_void,
                buf_c as *mut c_void,
                1,
                128,
                128,
            )
        },
        "warmup w34_concat_cols_2_f16",
    )?;
    check_rc(
        unsafe {
            ffi::w34_tap_add_f16_to_bf16(
                &ex,
                buf_a as *const c_void,
                buf_b as *const c_void,
                buf_c as *mut c_void,
                1,
                128,
                640,
                0,
            )
        },
        "warmup w34_tap_add_f16_to_bf16",
    )?;
    // Selector IDs are indices, not the FP32 GDN state previously in buf_ws.
    let candidate_ids = arena.upload_slice(&[0i32; 16], &stream)?;
    check_rc(
        unsafe {
            ffi::w34_dflash_qk_norm_rope_kv_write_bf16_f32_table(
                &ex,
                buf_a as *mut c_void,
                buf_d as *mut c_void,
                candidate_ids as *const i32,
                buf_b as *const c_void,
                buf_b as *const c_void,
                buf_c as *const c_void,
                &bf16_layout,
                slot_map as *const i32,
                1,
                1,
                1,
                256,
                1,
                1e-6,
            )
        },
        "warmup BF16 QK/RoPE + KV write",
    )?;
    check_rc(
        unsafe {
            ffi::w34_dflash_select_greedy_bf16(
                &ex,
                candidate_ids as *const i32,
                buf_c as *const f32,
                buf_a as *const c_void,
                buf_a as *const c_void,
                buf_b as *const c_void,
                buf_d as *mut i32,
                0,
                1,
                16,
                256,
                1,
            )
        },
        "warmup dflash_select_greedy_bf16",
    )?;
    check_rc(
        unsafe {
            ffi::w34_embed_lookup_f16_to_bf16(
                &ex,
                candidate_ids as *const i32,
                buf_a as *const c_void,
                buf_b as *mut c_void,
                1,
                128,
                1,
            )
        },
        "warmup embed_f16_to_bf16",
    )?;
    check_rc(
        unsafe {
            ffi::w34_cast_bf16_to_f16(&ex, buf_a as *const c_void, buf_b as *mut c_void, 128)
        },
        "warmup cast_bf16_to_f16",
    )?;
    check_rc(
        unsafe {
            ffi::w34_dflash_edges_bf16(
                &ex,
                candidate_ids as *const i32,
                buf_c as *const f32,
                buf_a as *const c_void,
                buf_a as *const c_void,
                buf_b as *const c_void,
                buf_d as *mut f32,
                0,
                1,
                16,
                256,
                1,
            )
        },
        "warmup dflash_edges_bf16",
    )?;
    check_rc(
        unsafe {
            ffi::w34_topk_rows_f16(
                &ex,
                buf_a as *const c_void,
                buf_b as *mut i32,
                buf_c as *mut f32,
                1,
                256,
                0,
                16,
            )
        },
        "warmup w34_topk_rows_f16",
    )?;
    check_rc(
        unsafe {
            ffi::w34_topk_rows_split_f16(
                &ex,
                buf_a as *const c_void,
                buf_ws as *mut i32,
                (buf_ws + 4096) as *mut f32,
                buf_b as *mut i32,
                buf_c as *mut f32,
                1,
                2048,
                0,
                16,
                16,
            )
        },
        "warmup w34_topk_rows_split_f16",
    )?;
    let zero_i32 = arena.upload_slice(&[0i32; 16], &stream)?;
    check_rc(
        unsafe {
            ffi::w34_dflash_select_greedy_bf16(
                &ex,
                zero_i32 as *const i32,
                buf_c as *const f32,
                buf_a as *const c_void,
                buf_b as *const c_void,
                buf_b as *const c_void,
                buf_d as *mut i32,
                0,
                1,
                16,
                256,
                1,
            )
        },
        "warmup w34_dflash_select_greedy_bf16",
    )?;
    check_rc(
        unsafe {
            ffi::w34_dflash_select_greedy_dev_anchor_bf16(
                &ex,
                zero_i32 as *const i32,
                buf_c as *const f32,
                buf_a as *const c_void,
                buf_b as *const c_void,
                buf_b as *const c_void,
                buf_d as *mut i32,
                zero_i32 as *const i32,
                1,
                16,
                256,
                1,
            )
        },
        "warmup w34_dflash_select_greedy_dev_anchor_bf16",
    )?;

    check_rc(
        unsafe {
            ffi::w34_rms_norm_bf16(
                &ex,
                buf_a as *const c_void,
                buf_b as *mut c_void,
                buf_c as *const c_void,
                buf_d as *mut c_void,
                1,
                128,
                1e-6,
            )
        },
        "warmup rms_norm_bf16",
    )?;
    check_rc(
        unsafe {
            ffi::w34_rms_norm_bf16(
                &ex,
                buf_a as *const c_void,
                buf_b as *mut c_void,
                buf_c as *const c_void,
                buf_d as *mut c_void,
                1,
                256,
                1e-6,
            )
        },
        "warmup rms_norm_bf16 vec8",
    )?;
    check_rc(
        unsafe {
            ffi::w34_rms_norm_bf16(
                &ex,
                buf_a as *const c_void,
                buf_b as *mut c_void,
                buf_c as *const c_void,
                buf_d as *mut c_void,
                1,
                2560,
                1e-6,
            )
        },
        "warmup rms_norm_bf16 vec8 h2560",
    )?;
    check_rc(
        unsafe {
            ffi::w34_silu_and_mul_bf16(&ex, buf_a as *const c_void, buf_b as *mut c_void, 1, 128)
        },
        "warmup silu_and_mul_bf16",
    )?;
    check_rc(
        unsafe {
            ffi::w34_dflash_conv_bf16(
                &ex,
                buf_a as *const c_void,
                buf_b as *const c_void,
                buf_c as *const c_void,
                buf_d as *mut c_void,
                1,
                128,
                8,
                64,
                3,
                0,
            )
        },
        "warmup dflash_conv_bf16",
    )?;
    check_rc(
        unsafe {
            ffi::w34_dflash_qk_norm_rope_bf16(
                &ex,
                buf_a as *mut c_void,
                slot_map as *const i32,
                buf_b as *const c_void,
                buf_b as *const c_void,
                buf_c as *const c_void,
                1,
                2,
                1,
                128,
                1,
                1e-6,
            )
        },
        "warmup dflash_qk_norm_rope_bf16",
    )?;
    check_rc(
        unsafe {
            ffi::w34_copy_cols_bf16(
                &ex,
                buf_a as *const c_void,
                buf_b as *mut c_void,
                1,
                256,
                128,
                128,
                0,
                128,
            )
        },
        "warmup copy_cols_bf16",
    )?;

    // Rejection kernels consume explicit probabilities/draws, not raw scratch.
    let reject_ids = arena.upload_slice(&[0i32; 2], &stream)?;
    let reject_probs = arena.upload_slice(&[1.0f32; 2], &stream)?;
    let reject_draws = arena.upload_slice(&[0.5f64], &stream)?;
    check_rc(
        unsafe {
            ffi::w34_rejection_greedy(
                &ex,
                reject_ids as *const i32,
                reject_ids as *const i32,
                core::ptr::null(),
                buf_a as *mut i32,
                buf_b as *mut i32,
                buf_c as *mut i32,
                1,
                1,
            )
        },
        "warmup w34_rejection_greedy",
    )?;
    check_rc(
        unsafe {
            ffi::w34_rejection_stochastic(
                &ex,
                reject_ids as *const i32,
                reject_probs as *const f32,
                reject_probs as *const f32,
                reject_draws as *const f64,
                reject_draws as *const f64,
                core::ptr::null(),
                buf_a as *mut i32,
                buf_b as *mut i32,
                buf_c as *mut i32,
                1,
                1,
                1,
            )
        },
        "warmup w34_rejection_stochastic",
    )?;

    // Warm Phase 5.2 (Vision) & Phase 5.3 (MoE) kernels so CUDA 13 lazy loading
    // resolves all module symbols before any TP=2 collective or graph capture.
    let cu_seqlens = arena.upload_slice(&[0i32, 1i32], &stream)?;
    check_rc(
        unsafe {
            ffi::w34_vision_bias_add_f16(
                &ex,
                buf_a as *const c_void,
                buf_b as *const c_void,
                buf_a as *const c_void,
                buf_c as *mut c_void,
                1,
                64,
            )
        },
        "warmup w34_vision_bias_add_f16",
    )?;
    check_rc(
        unsafe {
            ffi::w34_vision_layernorm_f16(
                &ex,
                buf_a as *const c_void,
                buf_b as *const c_void,
                buf_b as *const c_void,
                buf_c as *mut c_void,
                1,
                64,
                1e-6,
            )
        },
        "warmup w34_vision_layernorm_f16",
    )?;
    check_rc(
        unsafe {
            ffi::w34_vision_bias_gelu_f16(
                &ex,
                buf_a as *const c_void,
                buf_b as *const c_void,
                buf_c as *mut c_void,
                1,
                64,
                1,
            )
        },
        "warmup w34_vision_bias_gelu_f16",
    )?;
    let ex_vision = stream.exec(buf_ws, 64 * 1024);
    for head_dim in [64, 72] {
        check_rc(
            unsafe {
                ffi::w34_vision_qkv_rope_attn_f16(
                    &ex_vision,
                    buf_a as *mut c_void,
                    buf_b as *const c_void,
                    reject_ids as *const i32,
                    cu_seqlens as *const i32,
                    buf_c as *mut c_void,
                    1,
                    1,
                    2,
                    head_dim,
                    10000.0,
                )
            },
            "warmup w34_vision_qkv_rope_attn_f16",
        )?;
    }
    check_rc(
        unsafe {
            ffi::w34_embedding_splice_f16(
                &ex,
                buf_c as *mut c_void,
                buf_a as *const c_void,
                reject_ids as *const i32,
                1,
                64,
            )
        },
        "warmup w34_embedding_splice_f16",
    )?;
    check_rc(
        unsafe {
            ffi::w34_moe_topk_router_f16(
                &ex,
                buf_a as *const c_void,
                buf_b as *mut f32,
                reject_ids as *mut i32,
                1,
                8,
                2,
                1,
                0,
            )
        },
        "warmup w34_moe_topk_router_f16 (warp256)",
    )?;
    check_rc(
        unsafe {
            ffi::w34_moe_topk_router_f16(
                &ex,
                buf_a as *const c_void,
                buf_b as *mut f32,
                reject_ids as *mut i32,
                1,
                512,
                2,
                1,
                0,
            )
        },
        "warmup w34_moe_topk_router_f16 (cta128)",
    )?;
    check_rc(
        unsafe {
            ffi::w34_moe_w4a16_gate_up_silu_f16(
                &ex,
                buf_a as *const c_void,
                buf_b as *const u32,
                buf_a as *const c_void,
                reject_ids as *const i32,
                buf_c as *mut c_void,
                1,
                1,
                2,
                8,
                64,
                64,
            )
        },
        "warmup w34_moe_w4a16_gate_up_silu_f16",
    )?;
    check_rc(
        unsafe {
            ffi::w34_moe_w4a16_down_accum_f16(
                &ex,
                buf_c as *const c_void,
                buf_b as *const u32,
                buf_a as *const c_void,
                reject_ids as *const i32,
                reject_probs as *const f32,
                buf_a as *const c_void,
                buf_a as *const c_void,
                buf_c as *mut c_void,
                1,
                1,
                2,
                64,
                64,
                64,
            )
        },
        "warmup w34_moe_w4a16_down_accum_f16",
    )?;
    let moe_ids_m4 = arena.upload_slice(&[0i32, 1, 0, 1], &stream)?;
    let moe_w_m4 = arena.upload_slice(&[1.0f32; 4], &stream)?;
    let moe_ws = arena.alloc_aligned(64 * 1024)?;
    let ex_moe = stream.exec(moe_ws, 64 * 1024);
    for (m_tok, n_exp) in [(1i32, 2i32), (4, 8), (4, 2)] {
        check_rc(
            unsafe {
                ffi::w34_moe_w4a16_mlp_f16(
                    &ex_moe,
                    buf_a as *const c_void,
                    buf_b as *const u32,
                    buf_a as *const c_void,
                    buf_b as *const u32,
                    buf_a as *const c_void,
                    moe_ids_m4 as *const i32,
                    moe_w_m4 as *const f32,
                    buf_a as *const c_void,
                    buf_a as *const c_void,
                    buf_c as *mut c_void,
                    buf_d as *mut c_void,
                    m_tok,
                    1,
                    n_exp,
                    64,
                    64,
                    64,
                )
            },
            "warmup w34_moe_w4a16_mlp_f16",
        )?;
        check_rc(
            unsafe {
                ffi::w34_moe_w4a16_mlp_zp_f16(
                    &ex_moe,
                    buf_a as *const c_void,
                    buf_b as *const u32,
                    buf_a as *const c_void,
                    buf_b as *const u32,
                    buf_b as *const u32,
                    buf_a as *const c_void,
                    buf_b as *const u32,
                    moe_ids_m4 as *const i32,
                    moe_w_m4 as *const f32,
                    buf_a as *const c_void,
                    buf_a as *const c_void,
                    buf_c as *mut c_void,
                    buf_d as *mut c_void,
                    m_tok,
                    1,
                    n_exp,
                    64,
                    64,
                    64,
                )
            },
            "warmup w34_moe_w4a16_mlp_zp_f16",
        )?;
    }
    // Sparse M4 (BLOCK_M=4 vec4) and M8..128 (BLOCK_M=8 MMA) g128 dispatches have
    // distinct sorting and kernel specializations. Warm both on each rank before
    // any TP collective capture or replay.
    let moe_ids_m8 = arena.upload_slice(&[0i32, 0, 2, 3, 4, 5, 6, 7], &stream)?;
    let moe_w_m8 = arena.upload_slice(&[1.0f32; 8], &stream)?;
    let moe_sparse_weights = arena.alloc_aligned(16 * 256 * 128 / 2)?;
    for m_sparse in [4i32, 8] {
        check_rc(
            unsafe {
                ffi::w34_moe_w4a16_mlp_f16(
                    &ex_moe,
                    buf_a as *const c_void,
                    moe_sparse_weights as *const u32,
                    buf_a as *const c_void,
                    moe_sparse_weights as *const u32,
                    buf_a as *const c_void,
                    moe_ids_m8 as *const i32,
                    moe_w_m8 as *const f32,
                    core::ptr::null(),
                    core::ptr::null(),
                    buf_c as *mut c_void,
                    buf_d as *mut c_void,
                    m_sparse,
                    1,
                    16,
                    128,
                    128,
                    128,
                )
            },
            "warmup sparse w34_moe_w4a16_mlp_f16",
        )?;
    }
    check_rc(
        unsafe {
            ffi::w34_moe_f16_gate_up_silu(
                &ex,
                buf_a as *const c_void,
                buf_b as *const c_void,
                reject_ids as *const i32,
                buf_c as *mut c_void,
                1,
                1,
                2,
                8,
                64,
            )
        },
        "warmup w34_moe_f16_gate_up_silu",
    )?;
    check_rc(
        unsafe {
            ffi::w34_moe_f16_down_accum(
                &ex,
                buf_c as *const c_void,
                buf_b as *const c_void,
                reject_ids as *const i32,
                reject_probs as *const f32,
                buf_a as *const c_void,
                buf_a as *const c_void,
                buf_c as *mut c_void,
                1,
                1,
                2,
                64,
                8,
            )
        },
        "warmup w34_moe_f16_down_accum",
    )?;

    // Warm Clef JointSchemaHead kernels across TP ranks before collective capture.
    check_rc(
        unsafe {
            ffi::w34_clef_accum_spans_f16_to_f32(
                &ex,
                buf_a as *const c_void,
                cu_seqlens as *const i32,
                buf_c as *mut f32,
                0,
                1,
                1,
                64,
            )
        },
        "warmup w34_clef_accum_spans_f16_to_f32",
    )?;
    check_rc(
        unsafe {
            ffi::w34_clef_cast_f32_to_f16(&ex, buf_c as *const f32, buf_d as *mut c_void, 64)
        },
        "warmup w34_clef_cast_f32_to_f16",
    )?;
    check_rc(
        unsafe {
            ffi::w34_clef_build_option_queries_f16(
                &ex,
                buf_a as *const c_void,
                buf_b as *const c_void,
                buf_a as *const c_void,
                zero_i32 as *const i32,
                buf_c as *mut c_void,
                1,
                1,
                64,
            )
        },
        "warmup w34_clef_build_option_queries_f16",
    )?;
    check_rc(
        unsafe {
            ffi::w34_clef_mha_attn_f16(
                &ex,
                buf_a as *const c_void,
                buf_b as *const c_void,
                buf_a as *const c_void,
                buf_b as *const c_void,
                buf_c as *mut c_void,
                1,
                1,
                2,
                32,
            )
        },
        "warmup w34_clef_mha_attn_f16",
    )?;
    check_rc(
        unsafe {
            ffi::w34_clef_option_summaries_f16(
                &ex,
                buf_a as *const c_void,
                buf_b as *const c_void,
                cu_seqlens as *const i32,
                buf_c as *mut c_void,
                1,
                1,
                64,
            )
        },
        "warmup w34_clef_option_summaries_f16",
    )?;
    check_rc(
        unsafe {
            ffi::w34_clef_build_fields_f16(
                &ex,
                buf_a as *const c_void,
                buf_b as *const c_void,
                buf_a as *const c_void,
                buf_b as *const c_void,
                zero_i32 as *const i32,
                buf_c as *mut c_void,
                1,
                64,
            )
        },
        "warmup w34_clef_build_fields_f16",
    )?;
    check_rc(
        unsafe {
            ffi::w34_clef_build_residual_features_f16(
                &ex,
                buf_a as *const c_void,
                buf_b as *const c_void,
                zero_i32 as *const i32,
                buf_c as *mut c_void,
                1,
                1,
                64,
            )
        },
        "warmup w34_clef_build_residual_features_f16",
    )?;

    dev.synchronize()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_f16_f32_conversions_and_specials() {
        // Zero and negative zero
        assert_eq!(f16_to_f32(0x0000), 0.0);
        assert_eq!(f16_to_f32(0x8000), -0.0);
        assert_eq!(f32_to_f16(0.0), 0x0000);
        assert_eq!(f32_to_f16(-0.0), 0x8000);

        // Exact powers of 2
        assert_eq!(f16_to_f32(f32_to_f16(1.0)), 1.0);
        assert_eq!(f16_to_f32(f32_to_f16(-1.0)), -1.0);
        assert_eq!(f16_to_f32(f32_to_f16(0.5)), 0.5);
        assert_eq!(f16_to_f32(f32_to_f16(2.0)), 2.0);

        // Max representable FP16: 65504.0
        assert_eq!(f16_to_f32(0x7bff), 65504.0);
        assert_eq!(f32_to_f16(65504.0), 0x7bff);

        // Infinities
        assert_eq!(f16_to_f32(0x7c00), f32::INFINITY);
        assert_eq!(f16_to_f32(0xfc00), f32::NEG_INFINITY);
        assert_eq!(f32_to_f16(f32::INFINITY), 0x7c00);
        assert_eq!(f32_to_f16(f32::NEG_INFINITY), 0xfc00);

        // NaN
        assert!(f16_to_f32(0x7e00).is_nan());
        let nan_f16 = f32_to_f16(f32::NAN);
        assert_eq!(nan_f16 & 0x7c00, 0x7c00);
        assert_ne!(nan_f16 & 0x03ff, 0);
    }

    #[test]
    fn test_bf16_f32_conversions() {
        assert_eq!(bf16_to_f32(0x0000), 0.0);
        assert_eq!(bf16_to_f32(f32_to_bf16(1.0)), 1.0);
        assert_eq!(bf16_to_f32(f32_to_bf16(-2.5)), -2.5);

        // Overflow of FP16 (e.g. 70000.0) is preserved in BF16!
        let big = 70000.0f32;
        let big_bf16 = f32_to_bf16(big);
        let roundtrip = bf16_to_f32(big_bf16);
        assert!((roundtrip - big).abs() / big < 0.01);
    }

    #[test]
    fn test_capacity_planner_budgeting_logic() {
        let total_vram = 24 * 1024 * 1024 * 1024; // 24 GiB
        let free_vram = 22 * 1024 * 1024 * 1024; // 22 GiB
        let weight_arena = 8 * 1024 * 1024 * 1024; // 8 GiB
        let scratch_arena = 1024 * 1024 * 1024; // 1 GiB
        let state_bundle = 512 * 1024 * 1024; // 512 MiB
        let bytes_per_page = 16 * 1024; // 16 KiB

        // Auto page calculation
        let plan = CapacityPlanner::plan(
            total_vram,
            free_vram,
            weight_arena,
            scratch_arena,
            state_bundle,
            bytes_per_page,
            None,
        )
        .unwrap();

        assert!(plan.max_kv_pages_per_layer > 0);
        assert!(plan.headroom_bytes >= MIN_HEADROOM_BYTES);

        // Explicit requested pages exceeding capacity should fail
        let huge_req = plan.max_kv_pages_per_layer + 1_000_000;
        let err = CapacityPlanner::plan(
            total_vram,
            free_vram,
            weight_arena,
            scratch_arena,
            state_bundle,
            bytes_per_page,
            Some(huge_req),
        );
        assert!(err.is_err());
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn test_prefill_64q_zero_ulp_parity_across_kv_formats() {
        prefill_variant_fixture(false);
    }

    #[cfg(feature = "cuda")]
    #[test]
    #[ignore = "GPU microbenchmark; run serially on an idle card with --nocapture"]
    fn bench_prefill_query_tiles() {
        prefill_variant_fixture(true);
    }

    #[cfg(feature = "cuda")]
    fn prefill_variant_fixture(benchmark: bool) {
        let _dev = CudaDevice::set_current(0).unwrap();
        let stream = CudaStream::new(0).unwrap();
        let mut arena = DeviceArena::new(0, 64 * 1024 * 1024).unwrap();
        let ex = stream.exec(arena.buf.addr(), 4 * 1024 * 1024);
        let _ = arena.alloc_aligned(4 * 1024 * 1024).unwrap();

        let num_pages = 32usize;
        let kv_heads = 4usize;
        let head_dim = 256usize;
        let max_tokens = num_pages * 16; // 512

        // Deterministic pseudo-random FP16 inputs for Q, K, and V across 512 tokens.
        let mut rng = 0x9E37_79B9_7F4A_7C15u64;
        let mut next_f16 = || -> u16 {
            rng ^= rng << 13;
            rng ^= rng >> 7;
            rng ^= rng << 17;
            let val = ((rng & 0xFFFF) as f32 / 65535.0 - 0.5) * 1.5;
            f32_to_f16(val)
        };

        let max_q_heads = 24usize;
        let query_capacity = if benchmark { 1024 } else { max_tokens };
        let q_host: Vec<u16> = (0..query_capacity * max_q_heads * head_dim)
            .map(|_| next_f16())
            .collect();
        let k_host: Vec<u16> = (0..max_tokens * kv_heads * head_dim)
            .map(|_| next_f16())
            .collect();
        let v_host: Vec<u16> = (0..max_tokens * kv_heads * head_dim)
            .map(|_| next_f16())
            .collect();

        let q_dev = arena.upload_slice(&q_host, &stream).unwrap();
        let k_in_dev = arena.upload_slice(&k_host, &stream).unwrap();
        let v_in_dev = arena.upload_slice(&v_host, &stream).unwrap();
        let out_16q = DeviceBuffer::alloc(0, query_capacity * max_q_heads * head_dim * 2).unwrap();
        let out_64q = DeviceBuffer::alloc(0, query_capacity * max_q_heads * head_dim * 2).unwrap();
        let out_128q = DeviceBuffer::alloc(0, query_capacity * max_q_heads * head_dim * 2).unwrap();

        // Permuted page table to exercise non-contiguous physical page lookups.
        let block_table_host: Vec<i32> = (0..num_pages as i32)
            .map(|i| (i * 7 + 3) % (num_pages as i32))
            .collect();
        let block_table_dev = arena.upload_slice(&block_table_host, &stream).unwrap();
        let write_slots_host: Vec<i32> = (0..max_tokens)
            .map(|t| block_table_host[t / 16] * 16 + (t % 16) as i32)
            .collect();
        let write_slots_dev = arena.upload_slice(&write_slots_host, &stream).unwrap();

        for (fmt, transform, row_bytes) in [
            (
                ffi::W34_KV_FORMAT_FP16,
                ffi::W34_KV_TRANSFORM_IDENTITY,
                512usize,
            ),
            (
                ffi::W34_KV_FORMAT_Q8_0,
                ffi::W34_KV_TRANSFORM_RADEMACHER_WHT256,
                272usize,
            ),
            (
                ffi::W34_KV_FORMAT_TQ4,
                ffi::W34_KV_TRANSFORM_RADEMACHER_WHT256,
                132usize,
            ),
        ] {
            let pool_bytes = num_pages * kv_heads * 16 * row_bytes;
            let k_pool = DeviceBuffer::alloc(0, pool_bytes).unwrap();
            let v_pool = DeviceBuffer::alloc(0, pool_bytes).unwrap();
            let layout = W34KvLayerLayout {
                k_base: k_pool.addr(),
                v_base: v_pool.addr(),
                k_page_stride_bytes: (kv_heads * 16 * row_bytes) as u64,
                v_page_stride_bytes: (kv_heads * 16 * row_bytes) as u64,
                k_head_stride_bytes: (16 * row_bytes) as u64,
                v_head_stride_bytes: (16 * row_bytes) as u64,
                k_token_stride_bytes: row_bytes as u64,
                v_token_stride_bytes: row_bytes as u64,
                namespace_id: ffi::W34_KV_NAMESPACE_TARGET_FULL_ATTN,
                layer_id: 0,
                local_kv_heads: kv_heads as i32,
                head_dim: head_dim as i32,
                block_size: 16,
                k_format: fmt,
                v_format: fmt,
                k_transform: transform,
                v_transform: transform,
                reserved: 0,
            };

            // Populate all 512 KV positions via the real w34_kv_cache_write_strided kernel.
            let q_start_all = arena
                .upload_slice(&[0i32, max_tokens as i32], &stream)
                .unwrap();
            let ctx_zero = arena.upload_slice(&[0i32], &stream).unwrap();
            let seq_all = arena.upload_slice(&[max_tokens as i32], &stream).unwrap();
            let write_batch = W34AttentionBatch {
                query_start_loc: q_start_all as *const i32,
                context_lens_before: ctx_zero as *const i32,
                seq_lens_after: seq_all as *const i32,
                block_tables: block_table_dev as *const i32,
                write_slot_mapping: write_slots_dev as *const i32,
                valid_token_mask: core::ptr::null(),
                valid_req_mask: core::ptr::null(),
                num_reqs: 1,
                total_tokens: max_tokens as i32,
                max_blocks_per_req: num_pages as i32,
                max_q_len: max_tokens as i32,
                max_seq_len: max_tokens as i32,
                mask_mode: ffi::W34_ATTN_MASK_CAUSAL,
                sliding_window: 0,
                sm_scale: 0.0625,
                reserved: 0,
            };
            check_rc(
                unsafe {
                    ffi::w34_kv_cache_write(
                        &ex,
                        &layout,
                        &write_batch,
                        k_in_dev as *const c_void,
                        v_in_dev as *const c_void,
                    )
                },
                "populate kv pool for 64q test",
            )
            .unwrap();

            if benchmark {
                // Repeated physical pages isolate tile compute/dequant costs without a model.
                // End-to-end cold-prefill benchmarks separately measure real HBM working sets.
                let table: Vec<i32> = (0..4096).map(|i| block_table_host[i % num_pages]).collect();
                let pages = arena.upload_slice(&table, &stream).unwrap();
                for heads in [16, 24] {
                    for (queries, context) in [(256, 8192), (1024, 8192), (1024, 64512)] {
                        let starts = arena.upload_slice(&[0i32, queries], &stream).unwrap();
                        let ctx = arena.upload_slice(&[context], &stream).unwrap();
                        let seq = arena.upload_slice(&[context + queries], &stream).unwrap();
                        let batch = W34AttentionBatch {
                            query_start_loc: starts as *const i32,
                            context_lens_before: ctx as *const i32,
                            seq_lens_after: seq as *const i32,
                            block_tables: pages as *const i32,
                            write_slot_mapping: core::ptr::null(),
                            total_tokens: queries,
                            max_blocks_per_req: 4096,
                            max_q_len: queries,
                            max_seq_len: context + queries,
                            ..write_batch
                        };
                        for variant in [1, 2] {
                            let invoke = || {
                                check_rc(
                                    unsafe {
                                        ffi::w34_paged_attention_f16_force_kernel_for_test(
                                            &ex,
                                            &layout,
                                            &batch,
                                            q_dev as *const c_void,
                                            out_128q.addr() as *mut c_void,
                                            heads,
                                            variant,
                                        )
                                    },
                                    "prefill tile benchmark",
                                )
                                .unwrap()
                            };
                            invoke();
                            stream.synchronize().unwrap();
                            let start = CudaEvent::new(0, true).unwrap();
                            let end = CudaEvent::new(0, true).unwrap();
                            start.record(&stream).unwrap();
                            for _ in 0..5 {
                                invoke();
                            }
                            end.record(&stream).unwrap();
                            stream.synchronize().unwrap();
                            eprintln!(
                                "tile_bench fmt={fmt} heads={heads} q={queries} ctx={context} rows={} ms={:.4}",
                                if variant == 1 { 64 } else { 128 },
                                start.elapsed_ms(&end).unwrap() / 5.0
                            );
                        }
                    }
                }
                continue;
            }

            // Test both 9B/35B GQA=4 (16 Q heads) and 27B GQA=6 (24 Q heads),
            // including odd chunk sizes and non-zero prefix offsets.
            for num_q_heads in [16usize, 24usize] {
                for (q_len, ctx_before) in [(64usize, 0usize), (65, 19), (137, 128), (256, 200)] {
                    let seq_after = ctx_before + q_len;
                    let q_start = arena.upload_slice(&[0i32, q_len as i32], &stream).unwrap();
                    let ctx_dev = arena.upload_slice(&[ctx_before as i32], &stream).unwrap();
                    let seq_dev = arena.upload_slice(&[seq_after as i32], &stream).unwrap();
                    let batch = W34AttentionBatch {
                        query_start_loc: q_start as *const i32,
                        context_lens_before: ctx_dev as *const i32,
                        seq_lens_after: seq_dev as *const i32,
                        block_tables: block_table_dev as *const i32,
                        write_slot_mapping: core::ptr::null(),
                        valid_token_mask: core::ptr::null(),
                        valid_req_mask: core::ptr::null(),
                        num_reqs: 1,
                        total_tokens: q_len as i32,
                        max_blocks_per_req: num_pages as i32,
                        max_q_len: q_len as i32,
                        max_seq_len: seq_after as i32,
                        mask_mode: ffi::W34_ATTN_MASK_CAUSAL,
                        sliding_window: 0,
                        sm_scale: 0.0625,
                        reserved: 0,
                    };
                    for (use_64q, dst) in [
                        (0i32, out_16q.addr()),
                        (1i32, out_64q.addr()),
                        (2i32, out_128q.addr()),
                    ] {
                        check_rc(
                            unsafe {
                                ffi::w34_paged_attention_f16_force_kernel_for_test(
                                    &ex,
                                    &layout,
                                    &batch,
                                    q_dev as *const c_void,
                                    dst as *mut c_void,
                                    num_q_heads as i32,
                                    use_64q,
                                )
                            },
                            "run 16q vs 64q paged attention",
                        )
                        .unwrap();
                    }

                    let elems = q_len * num_q_heads * head_dim;
                    let mut h16 = vec![0u16; elems];
                    let mut h64 = vec![0u16; elems];
                    let mut h128 = vec![0u16; elems];
                    out_128q.copy_to_host(&mut h128, &stream).unwrap();
                    out_16q.copy_to_host(&mut h16, &stream).unwrap();
                    out_64q.copy_to_host(&mut h64, &stream).unwrap();
                    stream.synchronize().unwrap();
                    assert_eq!(
                        h16, h128,
                        "128q mismatch fmt={fmt} heads={num_q_heads} q={q_len} ctx={ctx_before}"
                    );
                    for idx in 0..elems {
                        assert_eq!(
                            h16[idx], h64[idx],
                            "non-zero ULP mismatch fmt={fmt} q_heads={num_q_heads} q_len={q_len} ctx={ctx_before} idx={idx}: 16q=0x{:04x} vs 64q=0x{:04x}",
                            h16[idx], h64[idx]
                        );
                    }
                }

                // Multi-request mixed decode + concurrent prefill (num_reqs = 3: 1 decode + 99 prefill + 156 prefill)
                let q_starts_mr = [0i32, 1, 100, 256];
                let ctx_mr = [143i32, 64, 95];
                let seq_mr = [144i32, 163, 251];
                let max_blk_mr = 16usize;
                let mut bt_mr = Vec::with_capacity(3 * max_blk_mr);
                for r in 0..3usize {
                    for p in 0..max_blk_mr {
                        bt_mr.push(((p + r * 5) % num_pages) as i32);
                    }
                }
                let q_start_dev = arena.upload_slice(&q_starts_mr, &stream).unwrap();
                let ctx_mr_dev = arena.upload_slice(&ctx_mr, &stream).unwrap();
                let seq_mr_dev = arena.upload_slice(&seq_mr, &stream).unwrap();
                let bt_mr_dev = arena.upload_slice(&bt_mr, &stream).unwrap();
                let mr_batch = W34AttentionBatch {
                    query_start_loc: q_start_dev as *const i32,
                    context_lens_before: ctx_mr_dev as *const i32,
                    seq_lens_after: seq_mr_dev as *const i32,
                    block_tables: bt_mr_dev as *const i32,
                    write_slot_mapping: core::ptr::null(),
                    valid_token_mask: core::ptr::null(),
                    valid_req_mask: core::ptr::null(),
                    num_reqs: 3,
                    total_tokens: 256,
                    max_blocks_per_req: max_blk_mr as i32,
                    max_q_len: 156,
                    max_seq_len: 251,
                    mask_mode: ffi::W34_ATTN_MASK_CAUSAL,
                    sliding_window: 0,
                    sm_scale: 0.0625,
                    reserved: 0,
                };
                for (use_64q, dst) in [
                    (0i32, out_16q.addr()),
                    (1i32, out_64q.addr()),
                    (2i32, out_128q.addr()),
                ] {
                    check_rc(
                        unsafe {
                            ffi::w34_paged_attention_f16_force_kernel_for_test(
                                &ex,
                                &layout,
                                &mr_batch,
                                q_dev as *const c_void,
                                dst as *mut c_void,
                                num_q_heads as i32,
                                use_64q,
                            )
                        },
                        "run multi-req 16q vs 64q paged attention",
                    )
                    .unwrap();
                }
                let elems = 256 * num_q_heads * head_dim;
                let mut h16 = vec![0u16; elems];
                let mut h64 = vec![0u16; elems];
                let mut h128 = vec![0u16; elems];
                out_128q.copy_to_host(&mut h128, &stream).unwrap();
                out_16q.copy_to_host(&mut h16, &stream).unwrap();
                out_64q.copy_to_host(&mut h64, &stream).unwrap();
                stream.synchronize().unwrap();
                assert_eq!(
                    h16, h128,
                    "multi-req 128q mismatch fmt={fmt} heads={num_q_heads}"
                );
                for idx in 0..elems {
                    assert_eq!(
                        h16[idx], h64[idx],
                        "multi-req non-zero ULP mismatch fmt={fmt} q_heads={num_q_heads} idx={idx}: 16q=0x{:04x} vs 64q=0x{:04x}",
                        h16[idx], h64[idx]
                    );
                }

                // Skip the middle request entirely and mask rows around tile boundaries.
                // Both kernels must preserve inactive outputs, including all-masked tiles.
                let mut valid_tokens = vec![1u8; 256];
                valid_tokens[100..164].fill(0);
                valid_tokens[255] = 0;
                let valid_tokens_dev = arena.upload_slice(&valid_tokens, &stream).unwrap();
                let valid_reqs_dev = arena.upload_slice(&[1u8, 0, 1], &stream).unwrap();
                let masked_batch = W34AttentionBatch {
                    valid_token_mask: valid_tokens_dev as *const u8,
                    valid_req_mask: valid_reqs_dev as *const u8,
                    sliding_window: 33,
                    ..mr_batch
                };
                for (variant, dst) in [(0, &out_16q), (2, &out_128q)] {
                    dst.copy_from_host(&vec![0x5500u16; elems], &stream)
                        .unwrap();
                    check_rc(
                        unsafe {
                            ffi::w34_paged_attention_f16_force_kernel_for_test(
                                &ex,
                                &layout,
                                &masked_batch,
                                q_dev as *const c_void,
                                dst.addr() as *mut c_void,
                                num_q_heads as i32,
                                variant,
                            )
                        },
                        "masked sliding-window 16q vs 128q",
                    )
                    .unwrap();
                }
                out_16q.copy_to_host(&mut h16, &stream).unwrap();
                out_128q.copy_to_host(&mut h128, &stream).unwrap();
                stream.synchronize().unwrap();
                assert_eq!(
                    h16, h128,
                    "masked 128q mismatch fmt={fmt} heads={num_q_heads}"
                );
                for row in 0..256 {
                    if (1..100).contains(&row) || valid_tokens[row] == 0 {
                        assert!(
                            h128[row * num_q_heads * head_dim..(row + 1) * num_q_heads * head_dim]
                                .iter()
                                .all(|&v| v == 0x5500)
                        );
                    }
                }
            }
        }
    }
}
