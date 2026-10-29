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

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CudaError {
    pub code: i32,
    pub context: &'static str,
}

impl fmt::Display for CudaError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "CUDA error {} in {}", self.code, self.context)
    }
}

impl std::error::Error for CudaError {}

pub type Result<T> = core::result::Result<T, CudaError>;

#[inline]
pub fn check_rc(rc: i32, context: &'static str) -> Result<()> {
    if rc == 0 {
        Ok(())
    } else {
        Err(CudaError { code: rc, context })
    }
}

/// Converts IEEE-754 binary16 bits (`u16`) to `f32`.
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
        let fixed = weight_arena_bytes
            .saturating_add(scratch_arena_bytes)
            .saturating_add(state_bundle_pool_bytes)
            .saturating_add(MIN_HEADROOM_BYTES);
        if fixed > free_vram_bytes {
            return Err(format!(
                "CapacityPlanner: fixed footprint ({:.2} GiB) exceeds free VRAM ({:.2} GiB); need >= 1.0 GiB headroom",
                fixed as f64 / (1024.0 * 1024.0 * 1024.0),
                free_vram_bytes as f64 / (1024.0 * 1024.0 * 1024.0),
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
            return Err(CudaError {
                code: -11,
                context: "copy_from_host buffer overflow",
            });
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
            return Err(CudaError {
                code: -11,
                context: "copy_to_host buffer overflow",
            });
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
        let next = aligned_off.checked_add(bytes).ok_or(CudaError {
            code: -1,
            context: "DeviceArena::alloc_aligned overflow",
        })?;
        if next > self.buf.len_bytes {
            return Err(CudaError {
                code: 2, // cudaErrorMemoryAllocation
                context: "DeviceArena capacity exceeded",
            });
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
        check_rc(
            unsafe {
                ffi::w34_gemma_rms_norm_f16(
                    &ex,
                    buf_a as *const c_void,
                    buf_b as *const c_void,
                    buf_c as *mut c_void,
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
                    buf_a as *const c_void,
                    buf_b as *mut c_void,
                    buf_c as *const c_void,
                    buf_d as *mut c_void,
                    core::ptr::null(),
                    1,
                    hs,
                    1e-6,
                )
            },
            "warmup w34_fused_add_gemma_rms_norm_f16",
        )?;
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
    check_rc(
        unsafe {
            ffi::w34_silu_and_mul_f16(
                &ex,
                buf_a as *const c_void,
                buf_c as *mut c_void,
                core::ptr::null(),
                1,
                128,
            )
        },
        "warmup w34_silu_and_mul_f16",
    )?;
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
    for m_warm in [1i64, 16, 64] {
        for group_blocks in [8i32, 2] {
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
        }
    }
    cublas.gemm_f16(buf_a, buf_b, buf_c, 1, 64, 128)?;
    cublas.gemm_bf16(buf_a, buf_b, buf_c, 1, 64, 128)?;
    check_rc(
        unsafe {
            ffi::w34_small_n_gemm_f16(
                &ex,
                buf_a as *const c_void,
                buf_b as *const c_void,
                buf_c as *mut c_void,
                1,
                64,
                128,
            )
        },
        "warmup w34_small_n_gemm_f16",
    )?;
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
    check_rc(
        unsafe {
            ffi::w34_paged_attention_f16(
                &ex,
                &layout_256,
                &attn_batch,
                buf_c as *const c_void,
                buf_d as *mut c_void,
                4,
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
                4,
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
                4,
            )
        },
        "warmup w34_paged_attention_f16 split_kv",
    )?;

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
        "warmup w34_paged_attention_bf16",
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
        "warmup w34_gdn_recurrent_f16",
    )?;

    // 5. Sampler & Phase 4 MTP / DFlash2 kernels
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
                8,
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
    check_rc(
        unsafe {
            ffi::w34_vision_qkv_rope_attn_f16(
                &ex,
                buf_a as *mut c_void,
                buf_b as *const c_void,
                reject_ids as *const i32,
                cu_seqlens as *const i32,
                buf_c as *mut c_void,
                1,
                1,
                2,
                64,
                10000.0,
            )
        },
        "warmup w34_vision_qkv_rope_attn_f16",
    )?;
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
        "warmup w34_moe_topk_router_f16",
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

    dev.synchronize()
}
