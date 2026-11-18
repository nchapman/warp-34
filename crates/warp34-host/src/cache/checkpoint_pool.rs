//! Compressed GDN (`Mamba`) checkpoint pool (`WHT128-Q8_0` with `BF16` block-32 scales)
//! across L1 GPU VRAM and L2 Pinned Host RAM (`MappedHostBuffer` with pageable fallback).
//!
//! Checkpoints are tagged with [`CHECKPOINT_POOL_SLOT_BASE`] so [`crate::radix::MambaRadixCache`]
//! manages compressed checkpoints transparently without competing for active `FP32` slots in
//! [`super::StateBundlePool`].

#[cfg(feature = "cuda")]
use core::ffi::c_void;
#[cfg(feature = "cuda")]
use std::sync::atomic::{AtomicBool, Ordering};

#[cfg(feature = "cuda")]
use super::StateBundlePool;
#[cfg(feature = "cuda")]
use crate::cuda::{CudaError, CudaStream, DeviceBuffer, MappedHostBuffer, Result, check_rc};
#[cfg(feature = "cuda")]
use warp34_kernels::ffi;

/// Slot-ID offset distinguishing compressed [`GdnCheckpointPool`] handles (`>= 1_000_000`)
/// from active `FP32` [`super::StateBundlePool`] indices (`0 .. total_slots`).
pub const CHECKPOINT_POOL_SLOT_BASE: i32 = 1_000_000;

/// Default process-wide L2 Host-RAM budget (`4 GiB`) across all tensor-parallel ranks.
pub const DEFAULT_HOST_CHECKPOINT_BUDGET_BYTES: usize = 4 * 1024 * 1024 * 1024;

/// Maximum automatically planned L2 Host-RAM checkpoint slots (`128` prefixes/turns).
pub const MAX_AUTO_HOST_CHECKPOINT_SLOTS: usize = 128;

/// Maximum automatically planned L1 VRAM checkpoint slots (`16` hot checkpoints).
pub const MAX_AUTO_VRAM_CHECKPOINT_SLOTS: usize = 16;

const SECTION_ALIGN_BYTES: usize = 256;

#[inline]
const fn align_up(bytes: usize, align: usize) -> usize {
    let rem = bytes % align;
    if rem == 0 {
        bytes
    } else {
        bytes.saturating_add(align - rem)
    }
}

/// Returns `true` when `slot` addresses [`GdnCheckpointPool`] rather than [`super::StateBundlePool`].
#[inline]
pub const fn is_checkpoint_pool_slot(slot: i32) -> bool {
    slot >= CHECKPOINT_POOL_SLOT_BASE
}

/// Encodes a 0-based logical checkpoint index as a tagged `MambaRadixCache` slot ID.
#[inline]
pub const fn encode_checkpoint_pool_slot(logical_id: usize) -> i32 {
    CHECKPOINT_POOL_SLOT_BASE + (logical_id as i32)
}

/// Decodes a tagged `MambaRadixCache` slot ID into a 0-based logical checkpoint index.
#[inline]
pub const fn decode_checkpoint_pool_slot(slot: i32) -> Option<usize> {
    if slot >= CHECKPOINT_POOL_SLOT_BASE {
        Some((slot - CHECKPOINT_POOL_SLOT_BASE) as usize)
    } else {
        None
    }
}

/// Byte layout of a single flat `WHT128-Q8_0` GDN checkpoint bundle.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct GdnCheckpointLayout {
    pub num_layers: usize,
    pub rows_per_layer: usize,
    pub head_k_dim: usize,
    pub conv_elems_per_slot: usize,
    pub ple_conv_elems_per_slot: usize,
    pub qdata_offset: usize,
    pub qdata_bytes: usize,
    pub scales_offset: usize,
    pub scales_bytes: usize,
    pub conv_offset: usize,
    pub conv_bytes: usize,
    pub ple_conv_offset: usize,
    pub ple_conv_bytes: usize,
    pub bundle_bytes: usize,
}

impl GdnCheckpointLayout {
    pub const fn new(
        num_layers: usize,
        local_v_heads: usize,
        head_v_dim: usize,
        head_k_dim: usize,
        conv_dim: usize,
        conv_tap_count: usize,
        ple_conv_dim: usize,
    ) -> Self {
        let rows_per_layer = local_v_heads.saturating_mul(head_v_dim);
        let conv_elems_per_slot = conv_dim.saturating_mul(conv_tap_count);
        let ple_conv_elems_per_slot = ple_conv_dim.saturating_mul(9);

        let raw_qdata = num_layers
            .saturating_mul(rows_per_layer)
            .saturating_mul(head_k_dim);
        let qdata_bytes = align_up(raw_qdata, SECTION_ALIGN_BYTES);

        let scales_per_row = head_k_dim / 32;
        let raw_scales = num_layers
            .saturating_mul(rows_per_layer)
            .saturating_mul(scales_per_row)
            .saturating_mul(2);
        let scales_offset = qdata_bytes;
        let scales_bytes = align_up(raw_scales, SECTION_ALIGN_BYTES);

        let raw_conv = num_layers
            .saturating_mul(conv_elems_per_slot)
            .saturating_mul(2);
        let conv_offset = scales_offset.saturating_add(scales_bytes);
        let conv_bytes = align_up(raw_conv, SECTION_ALIGN_BYTES);

        let raw_ple = ple_conv_elems_per_slot.saturating_mul(2);
        let ple_conv_offset = conv_offset.saturating_add(conv_bytes);
        let ple_conv_bytes = align_up(raw_ple, SECTION_ALIGN_BYTES);

        let total = ple_conv_offset.saturating_add(ple_conv_bytes);
        let bundle_bytes = if total < SECTION_ALIGN_BYTES {
            SECTION_ALIGN_BYTES
        } else {
            total
        };

        Self {
            num_layers,
            rows_per_layer,
            head_k_dim,
            conv_elems_per_slot,
            ple_conv_elems_per_slot,
            qdata_offset: 0,
            qdata_bytes,
            scales_offset,
            scales_bytes,
            conv_offset,
            conv_bytes,
            ple_conv_offset,
            ple_conv_bytes,
            bundle_bytes,
        }
    }

    #[cfg(feature = "cuda")]
    pub fn from_state_pool(pool: &StateBundlePool) -> Self {
        Self::new(
            pool.layers.len(),
            pool.local_v_heads,
            pool.head_v_dim,
            pool.head_k_dim,
            pool.conv_dim,
            pool.conv_tap_count,
            if pool.ple_conv_state.is_some() {
                pool.ple_conv_dim
            } else {
                0
            },
        )
    }
}

/// Pure parser for Linux `/proc/meminfo` (`MemAvailable:`) + cgroup v2 (`memory.max` / `memory.current`).
pub fn parse_available_host_ram(
    meminfo: &str,
    cgroup_max: Option<&str>,
    cgroup_current: Option<&str>,
) -> Option<usize> {
    let mut mem_avail_bytes: Option<u64> = None;
    for line in meminfo.lines() {
        if let Some(rest) = line.strip_prefix("MemAvailable:") {
            let kib_str = rest.split_whitespace().next()?;
            let kib: u64 = kib_str.parse().ok()?;
            mem_avail_bytes = Some(kib.saturating_mul(1024));
            break;
        }
    }
    let mut avail = mem_avail_bytes?;
    if let (Some(max_raw), Some(cur_raw)) = (cgroup_max, cgroup_current) {
        let max_trim = max_raw.trim();
        if max_trim != "max"
            && let (Ok(max_b), Ok(cur_b)) = (max_trim.parse::<u64>(), cur_raw.trim().parse::<u64>())
            && max_b < (u64::MAX / 2)
        {
            avail = avail.min(max_b.saturating_sub(cur_b));
        }
    }
    usize::try_from(avail).ok()
}

/// Queries available host RAM on Linux while respecting container cgroup v2 limits.
pub fn detect_available_host_ram_bytes() -> Option<usize> {
    let meminfo = std::fs::read_to_string("/proc/meminfo").ok()?;
    let cgroup_max = std::fs::read_to_string("/sys/fs/cgroup/memory.max").ok();
    let cgroup_cur = std::fs::read_to_string("/sys/fs/cgroup/memory.current").ok();
    parse_available_host_ram(&meminfo, cgroup_max.as_deref(), cgroup_cur.as_deref())
}

/// Plans per-rank L2 Host-RAM checkpoint slot capacity (`<= 128` slots, `<= 4 GiB` across all TP ranks).
pub fn plan_auto_host_checkpoint_slots(
    bundle_bytes: usize,
    tp_size: usize,
    available_ram_bytes: Option<usize>,
) -> usize {
    if bundle_bytes == 0 {
        return 0;
    }
    let total_budget = available_ram_bytes
        .map_or(DEFAULT_HOST_CHECKPOINT_BUDGET_BYTES / 4, |avail| {
            (avail / 8).min(DEFAULT_HOST_CHECKPOINT_BUDGET_BYTES)
        });
    let per_rank_budget = total_budget / tp_size.max(1);
    (per_rank_budget / bundle_bytes).min(MAX_AUTO_HOST_CHECKPOINT_SLOTS)
}

#[cfg(feature = "cuda")]
static WARNED_CKPT_PINNED_FALLBACK: AtomicBool = AtomicBool::new(false);

#[cfg(feature = "cuda")]
enum HostCheckpointSlot {
    Pinned(MappedHostBuffer),
    Pageable(Box<[u8]>),
}

#[cfg(feature = "cuda")]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum CkptLocation {
    Vram(usize),
    Host(usize),
}

/// Tiered `WHT128-Q8_0` GDN checkpoint store (L1 GPU VRAM + lazy-allocated L2 Pinned Host RAM).
#[cfg(feature = "cuda")]
pub struct GdnCheckpointPool {
    pub device_id: i32,
    pub layout: GdnCheckpointLayout,
    pub vram_slots: usize,
    pub host_slots: usize,
    pub total_slots: usize,
    pub vram_hits: u64,
    pub host_hits: u64,
    pub host_spills: u64,
    ssm_layer_ptrs: DeviceBuffer,
    conv_layer_ptrs: DeviceBuffer,
    vram_buf: Option<DeviceBuffer>,
    staging_buf: Option<DeviceBuffer>,
    host_storage: Vec<Option<HostCheckpointSlot>>,
    locations: Vec<Option<CkptLocation>>,
    vram_lru: Vec<usize>,
    free_vram_slots: Vec<usize>,
    free_host_slots: Vec<usize>,
    free_logical_slots: Vec<usize>,
    ple_ngram_context: Vec<[i32; 2]>,
    ple_eos_token_id: i32,
}

#[cfg(feature = "cuda")]
impl GdnCheckpointPool {
    pub fn new(
        state_pool: &StateBundlePool,
        vram_slots: usize,
        host_slots: usize,
        stream: &CudaStream,
    ) -> Result<Self> {
        let device_id = state_pool.device_id;
        let layout = GdnCheckpointLayout::from_state_pool(state_pool);
        if layout.num_layers > 0 && layout.rows_per_layer > 0 && layout.head_k_dim != 128 {
            return Err(CudaError::new(
                -11,
                "GdnCheckpointPool requires head_k_dim == 128",
            ));
        }

        let ssm_addrs: Vec<u64> = state_pool
            .layers
            .iter()
            .map(|l| l.ssm_state.addr())
            .collect();
        let conv_addrs: Vec<u64> = state_pool
            .layers
            .iter()
            .map(|l| l.conv_state.addr())
            .collect();
        let ptr_bytes = (layout.num_layers * core::mem::size_of::<u64>()).max(16);
        let ssm_layer_ptrs = DeviceBuffer::alloc_on_stream(device_id, ptr_bytes, stream)?;
        let conv_layer_ptrs = DeviceBuffer::alloc_on_stream(device_id, ptr_bytes, stream)?;
        if !ssm_addrs.is_empty() {
            ssm_layer_ptrs.copy_from_host(&ssm_addrs, stream)?;
            conv_layer_ptrs.copy_from_host(&conv_addrs, stream)?;
            // `ssm_addrs` / `conv_addrs` are stack-local Vecs; ensure H2D completes before drop.
            stream.synchronize()?;
        }

        let vram_buf = if vram_slots > 0 {
            Some(DeviceBuffer::alloc_on_stream(
                device_id,
                vram_slots.saturating_mul(layout.bundle_bytes),
                stream,
            )?)
        } else {
            None
        };

        let total_slots = vram_slots.saturating_add(host_slots);
        let mut host_storage = Vec::with_capacity(host_slots);
        host_storage.resize_with(host_slots, || None);

        Ok(Self {
            device_id,
            layout,
            vram_slots,
            host_slots,
            total_slots,
            vram_hits: 0,
            host_hits: 0,
            host_spills: 0,
            ssm_layer_ptrs,
            conv_layer_ptrs,
            vram_buf,
            staging_buf: None,
            host_storage,
            locations: vec![None; total_slots],
            vram_lru: Vec::with_capacity(vram_slots),
            free_vram_slots: (0..vram_slots).rev().collect(),
            free_host_slots: (0..host_slots).rev().collect(),
            free_logical_slots: (0..total_slots).rev().collect(),
            ple_ngram_context: vec![
                [state_pool.ple_eos_token_id, state_pool.ple_eos_token_id];
                total_slots
            ],
            ple_eos_token_id: state_pool.ple_eos_token_id,
        })
    }

    #[inline]
    pub fn free_count(&self) -> usize {
        self.free_logical_slots.len()
    }

    #[inline]
    pub fn allocated_host_slots(&self) -> usize {
        self.host_storage.iter().filter(|s| s.is_some()).count()
    }

    #[inline]
    pub fn vram_used_slots(&self) -> usize {
        self.vram_slots.saturating_sub(self.free_vram_slots.len())
    }

    #[inline]
    pub fn host_used_slots(&self) -> usize {
        self.host_slots.saturating_sub(self.free_host_slots.len())
    }

    pub fn alloc_slot(&mut self) -> Option<i32> {
        self.free_logical_slots
            .pop()
            .map(encode_checkpoint_pool_slot)
    }

    pub fn free_slot(&mut self, tagged_slot: i32) {
        let Some(logical_id) = decode_checkpoint_pool_slot(tagged_slot) else {
            return;
        };
        if logical_id >= self.total_slots {
            return;
        }
        if let Some(loc) = self.locations[logical_id].take() {
            match loc {
                CkptLocation::Vram(v_idx) => {
                    if let Some(pos) = self.vram_lru.iter().position(|&id| id == logical_id) {
                        self.vram_lru.remove(pos);
                    }
                    self.free_vram_slots.push(v_idx);
                }
                CkptLocation::Host(h_idx) => {
                    self.free_host_slots.push(h_idx);
                }
            }
            self.free_logical_slots.push(logical_id);
        } else if !self.free_logical_slots.contains(&logical_id) {
            self.free_logical_slots.push(logical_id);
        }
    }

    pub fn reset_all_slots(&mut self) {
        self.locations.fill(None);
        self.vram_lru.clear();
        self.free_vram_slots = (0..self.vram_slots).rev().collect();
        self.free_host_slots = (0..self.host_slots).rev().collect();
        self.free_logical_slots = (0..self.total_slots).rev().collect();
        self.ple_ngram_context
            .fill([self.ple_eos_token_id, self.ple_eos_token_id]);
    }

    /// Clamps logical L1/L2 capacities downward so `TP >= 2` ranks execute identical eviction steps.
    pub fn truncate_slots(&mut self, max_vram_slots: usize, max_host_slots: usize) {
        let new_vram = self.vram_slots.min(max_vram_slots);
        let new_host = self.host_slots.min(max_host_slots);
        if new_vram < self.vram_slots || new_host < self.host_slots {
            self.vram_slots = new_vram;
            self.host_slots = new_host;
            self.total_slots = new_vram.saturating_add(new_host);
            self.host_storage.truncate(new_host);
            self.locations.truncate(self.total_slots);
            self.ple_ngram_context.truncate(self.total_slots);
            self.reset_all_slots();
        }
    }

    #[inline]
    pub fn is_slot_in_vram(&self, tagged_slot: i32) -> bool {
        decode_checkpoint_pool_slot(tagged_slot)
            .and_then(|id| self.locations.get(id).copied().flatten())
            .is_some_and(|loc| matches!(loc, CkptLocation::Vram(_)))
    }

    #[inline]
    pub fn is_slot_in_host(&self, tagged_slot: i32) -> bool {
        decode_checkpoint_pool_slot(tagged_slot)
            .and_then(|id| self.locations.get(id).copied().flatten())
            .is_some_and(|loc| matches!(loc, CkptLocation::Host(_)))
    }

    fn touch_vram_lru(&mut self, logical_id: usize) {
        if let Some(pos) = self.vram_lru.iter().position(|&id| id == logical_id) {
            self.vram_lru.remove(pos);
        }
        self.vram_lru.push(logical_id);
    }

    fn ensure_host_slot(&mut self, h_idx: usize) {
        if self.host_storage[h_idx].is_some() {
            return;
        }
        let bytes = self.layout.bundle_bytes;
        let slot = match MappedHostBuffer::alloc_unzeroed(bytes) {
            Ok(pinned) => HostCheckpointSlot::Pinned(pinned),
            Err(_) => {
                if !WARNED_CKPT_PINNED_FALLBACK.swap(true, Ordering::Relaxed) {
                    eprintln!(
                        "warning: pinned host GDN checkpoint pool fell back to pageable heap (check memlock / ulimit -l); prefix restores may synchronize stream"
                    );
                }
                HostCheckpointSlot::Pageable(vec![0u8; bytes].into_boxed_slice())
            }
        };
        self.host_storage[h_idx] = Some(slot);
    }

    fn ensure_staging_addr(&mut self, stream: &CudaStream) -> Result<u64> {
        if self.staging_buf.is_none() {
            self.staging_buf = Some(DeviceBuffer::alloc_on_stream(
                self.device_id,
                self.layout.bundle_bytes,
                stream,
            )?);
        }
        Ok(self.staging_buf.as_ref().unwrap().addr())
    }

    #[inline]
    fn vram_slot_addr(&self, v_idx: usize) -> u64 {
        self.vram_buf.as_ref().expect("vram_buf present").addr()
            + (v_idx * self.layout.bundle_bytes) as u64
    }

    fn copy_vram_to_host_slot(
        &mut self,
        v_idx: usize,
        h_idx: usize,
        stream: &CudaStream,
    ) -> Result<()> {
        self.ensure_host_slot(h_idx);
        let src_dev = self.vram_slot_addr(v_idx) as *const c_void;
        let bytes = self.layout.bundle_bytes;
        check_rc(
            // SAFETY: binds calling thread to `self.device_id` before D2H copy.
            unsafe { ffi::warp34_set_device(self.device_id) },
            "warp34_set_device",
        )?;
        match self.host_storage[h_idx].as_mut().unwrap() {
            HostCheckpointSlot::Pinned(pinned) => check_rc(
                // SAFETY: `pinned` is pinned portable host memory of `bytes`, and `src_dev` spans `bytes` on `stream`.
                unsafe {
                    ffi::warp34_memcpy_d2h_async(pinned.host_ptr, src_dev, bytes, stream.handle)
                },
                "spill GDN checkpoint L1->L2 pinned",
            ),
            HostCheckpointSlot::Pageable(boxed) => {
                check_rc(
                    // SAFETY: `boxed` is live for `bytes`; `stream.synchronize()` below completes before returning.
                    unsafe {
                        ffi::warp34_memcpy_d2h_async(
                            boxed.as_mut_ptr().cast(),
                            src_dev,
                            bytes,
                            stream.handle,
                        )
                    },
                    "spill GDN checkpoint L1->L2 pageable",
                )?;
                stream.synchronize()
            }
        }
    }

    fn launch_pack_to_dev_frame(
        &self,
        state_pool: &StateBundlePool,
        src_active_slot: i32,
        frame_dev_addr: u64,
        stream: &CudaStream,
    ) -> Result<()> {
        check_rc(
            // SAFETY: binds calling thread to `self.device_id` before kernel launch.
            unsafe { ffi::warp34_set_device(self.device_id) },
            "warp34_set_device",
        )?;
        let ex = stream.exec(0, 0);
        let ple_pool_ptr = state_pool
            .ple_conv_state
            .as_ref()
            .map_or(core::ptr::null(), |b| b.ptr as *const c_void);
        let dst_qdata = (frame_dev_addr + self.layout.qdata_offset as u64) as *mut c_void;
        let dst_scales = (frame_dev_addr + self.layout.scales_offset as u64) as *mut c_void;
        let dst_conv = (frame_dev_addr + self.layout.conv_offset as u64) as *mut c_void;
        let dst_ple = if self.layout.ple_conv_elems_per_slot > 0 {
            (frame_dev_addr + self.layout.ple_conv_offset as u64) as *mut c_void
        } else {
            core::ptr::null_mut()
        };
        check_rc(
            // SAFETY: pointer tables, `state_pool` buffers, and `frame_dev_addr` sections are sized by `self.layout`.
            unsafe {
                ffi::warp34_gdn_checkpoint_pack_wht_q8(
                    &ex,
                    self.ssm_layer_ptrs.addr() as *const u64,
                    self.conv_layer_ptrs.addr() as *const u64,
                    ple_pool_ptr,
                    src_active_slot,
                    state_pool.total_slots as i32,
                    self.layout.num_layers as i32,
                    self.layout.rows_per_layer as i32,
                    self.layout.head_k_dim as i32,
                    self.layout.conv_elems_per_slot as i32,
                    self.layout.ple_conv_elems_per_slot as i32,
                    dst_qdata,
                    dst_scales,
                    dst_conv,
                    dst_ple,
                )
            },
            "warp34_gdn_checkpoint_pack_wht_q8",
        )
    }

    fn launch_unpack_from_dev_frame(
        &self,
        state_pool: &mut StateBundlePool,
        frame_dev_addr: u64,
        dst_active_slot: i32,
        stream: &CudaStream,
    ) -> Result<()> {
        check_rc(
            // SAFETY: binds calling thread to `self.device_id` before kernel launch.
            unsafe { ffi::warp34_set_device(self.device_id) },
            "warp34_set_device",
        )?;
        let ex = stream.exec(0, 0);
        let ple_pool_ptr = state_pool
            .ple_conv_state
            .as_mut()
            .map_or(core::ptr::null_mut(), |b| b.ptr);
        let src_qdata = (frame_dev_addr + self.layout.qdata_offset as u64) as *const c_void;
        let src_scales = (frame_dev_addr + self.layout.scales_offset as u64) as *const c_void;
        let src_conv = (frame_dev_addr + self.layout.conv_offset as u64) as *const c_void;
        let src_ple = if self.layout.ple_conv_elems_per_slot > 0 {
            (frame_dev_addr + self.layout.ple_conv_offset as u64) as *const c_void
        } else {
            core::ptr::null()
        };
        check_rc(
            // SAFETY: `frame_dev_addr` sections and `state_pool` destination slot are valid for `self.layout`.
            unsafe {
                ffi::warp34_gdn_checkpoint_unpack_wht_q8(
                    &ex,
                    src_qdata,
                    src_scales,
                    src_conv,
                    src_ple,
                    self.ssm_layer_ptrs.addr() as *const u64,
                    self.conv_layer_ptrs.addr() as *const u64,
                    ple_pool_ptr,
                    dst_active_slot,
                    state_pool.total_slots as i32,
                    self.layout.num_layers as i32,
                    self.layout.rows_per_layer as i32,
                    self.layout.head_k_dim as i32,
                    self.layout.conv_elems_per_slot as i32,
                    self.layout.ple_conv_elems_per_slot as i32,
                )
            },
            "warp34_gdn_checkpoint_unpack_wht_q8",
        )
    }

    /// Quantizes `src_active_slot` (`FP32`) into `dst_ckpt_slot` (`WHT128-Q8_0`), spilling the
    /// oldest L1 VRAM checkpoint to L2 Host RAM asynchronously if L1 VRAM is full.
    pub fn pack_from_active(
        &mut self,
        state_pool: &StateBundlePool,
        src_active_slot: i32,
        dst_ckpt_slot: i32,
        stream: &CudaStream,
    ) -> Result<()> {
        let logical_id = decode_checkpoint_pool_slot(dst_ckpt_slot)
            .filter(|&id| id < self.total_slots)
            .ok_or_else(|| CudaError::new(-11, "invalid destination checkpoint slot"))?;
        if src_active_slot < 0 || (src_active_slot as usize) >= state_pool.total_slots {
            return Err(CudaError::new(-11, "invalid source active state slot"));
        }

        // Release any prior physical backing if overwriting an existing logical slot.
        if let Some(prev) = self.locations[logical_id].take() {
            match prev {
                CkptLocation::Vram(v) => {
                    if let Some(pos) = self.vram_lru.iter().position(|&id| id == logical_id) {
                        self.vram_lru.remove(pos);
                    }
                    self.free_vram_slots.push(v);
                }
                CkptLocation::Host(h) => self.free_host_slots.push(h),
            }
        }

        if !state_pool.ple_ngram_context.is_empty() {
            self.ple_ngram_context[logical_id] =
                state_pool.ple_ngram_context[src_active_slot as usize];
        }

        if self.vram_slots > 0 {
            let target_v_idx = if let Some(v_idx) = self.free_vram_slots.pop() {
                v_idx
            } else {
                // Evict oldest L1 VRAM slot to a free L2 Host slot on `stream`.
                let victim_logical =
                    self.vram_lru.first().copied().ok_or_else(|| {
                        CudaError::new(-11, "L1 VRAM LRU empty despite 0 free slots")
                    })?;
                let CkptLocation::Vram(v_idx) =
                    self.locations[victim_logical].expect("victim must be in VRAM")
                else {
                    return Err(CudaError::new(-11, "corrupt L1 VRAM LRU entry"));
                };
                let h_idx = self
                    .free_host_slots
                    .pop()
                    .ok_or_else(|| CudaError::new(-11, "no free L2 host slot for L1 spill"))?;
                self.copy_vram_to_host_slot(v_idx, h_idx, stream)?;
                self.vram_lru.remove(0);
                self.locations[victim_logical] = Some(CkptLocation::Host(h_idx));
                self.host_spills += 1;
                v_idx
            };
            let dev_addr = self.vram_slot_addr(target_v_idx);
            self.launch_pack_to_dev_frame(state_pool, src_active_slot, dev_addr, stream)?;
            self.locations[logical_id] = Some(CkptLocation::Vram(target_v_idx));
            self.touch_vram_lru(logical_id);
            Ok(())
        } else {
            let h_idx = self
                .free_host_slots
                .pop()
                .ok_or_else(|| CudaError::new(-11, "no free L2 host checkpoint slot"))?;
            self.ensure_host_slot(h_idx);
            let pinned_dev_addr = match self.host_storage[h_idx].as_ref().unwrap() {
                HostCheckpointSlot::Pinned(pinned) => Some(pinned.dev_addr()),
                HostCheckpointSlot::Pageable(_) => None,
            };
            if let Some(dev_addr) = pinned_dev_addr {
                self.launch_pack_to_dev_frame(state_pool, src_active_slot, dev_addr, stream)?;
            } else {
                let stage_addr = self.ensure_staging_addr(stream)?;
                self.launch_pack_to_dev_frame(state_pool, src_active_slot, stage_addr, stream)?;
                let bytes = self.layout.bundle_bytes;
                if let Some(HostCheckpointSlot::Pageable(boxed)) = &mut self.host_storage[h_idx] {
                    check_rc(
                        // SAFETY: `stage_addr` and `boxed` both span `bytes`; synced below.
                        unsafe {
                            ffi::warp34_memcpy_d2h_async(
                                boxed.as_mut_ptr().cast(),
                                stage_addr as *const c_void,
                                bytes,
                                stream.handle,
                            )
                        },
                        "pack GDN checkpoint stage->pageable",
                    )?;
                    stream.synchronize()?;
                }
            }
            self.locations[logical_id] = Some(CkptLocation::Host(h_idx));
            Ok(())
        }
    }

    /// Dequantizes `src_ckpt_slot` (`WHT128-Q8_0` in L1 VRAM or L2 Host RAM) directly into
    /// `dst_active_slot` (`FP32`) in `state_pool`.
    pub fn unpack_to_active(
        &mut self,
        state_pool: &mut StateBundlePool,
        src_ckpt_slot: i32,
        dst_active_slot: i32,
        stream: &CudaStream,
    ) -> Result<()> {
        let logical_id = decode_checkpoint_pool_slot(src_ckpt_slot)
            .filter(|&id| id < self.total_slots)
            .ok_or_else(|| CudaError::new(-11, "invalid source checkpoint slot"))?;
        if dst_active_slot < 0 || (dst_active_slot as usize) >= state_pool.total_slots {
            return Err(CudaError::new(-11, "invalid destination active state slot"));
        }
        let loc = self.locations[logical_id]
            .ok_or_else(|| CudaError::new(-11, "unpacking unpopulated checkpoint slot"))?;

        if !state_pool.ple_ngram_context.is_empty() {
            state_pool.ple_ngram_context[dst_active_slot as usize] =
                self.ple_ngram_context[logical_id];
        }

        match loc {
            CkptLocation::Vram(v_idx) => {
                self.vram_hits += 1;
                self.touch_vram_lru(logical_id);
                let dev_addr = self.vram_slot_addr(v_idx);
                self.launch_unpack_from_dev_frame(state_pool, dev_addr, dst_active_slot, stream)
            }
            CkptLocation::Host(h_idx) => {
                self.host_hits += 1;
                let pinned_dev_addr = match self.host_storage[h_idx]
                    .as_ref()
                    .ok_or_else(|| CudaError::new(-11, "missing host storage slot"))?
                {
                    HostCheckpointSlot::Pinned(pinned) => Some(pinned.dev_addr()),
                    HostCheckpointSlot::Pageable(_) => None,
                };
                if let Some(dev_addr) = pinned_dev_addr {
                    self.launch_unpack_from_dev_frame(state_pool, dev_addr, dst_active_slot, stream)
                } else {
                    let stage_addr = self.ensure_staging_addr(stream)?;
                    let bytes = self.layout.bundle_bytes;
                    check_rc(
                        // SAFETY: binds calling thread to `self.device_id` before H2D copy.
                        unsafe { ffi::warp34_set_device(self.device_id) },
                        "warp34_set_device",
                    )?;
                    if let Some(HostCheckpointSlot::Pageable(boxed)) = &self.host_storage[h_idx] {
                        check_rc(
                            // SAFETY: `boxed` and `stage_addr` span `bytes`; synced before `boxed` borrow ends.
                            unsafe {
                                ffi::warp34_memcpy_h2d_async(
                                    stage_addr as *mut c_void,
                                    boxed.as_ptr().cast(),
                                    bytes,
                                    stream.handle,
                                )
                            },
                            "unpack GDN checkpoint pageable->stage",
                        )?;
                        stream.synchronize()?;
                    }
                    self.launch_unpack_from_dev_frame(
                        state_pool,
                        stage_addr,
                        dst_active_slot,
                        stream,
                    )
                }
            }
        }
    }
}

#[cfg(feature = "cuda")]
impl Drop for GdnCheckpointPool {
    fn drop(&mut self) {
        if self.host_storage.iter().any(Option::is_some) {
            // SAFETY: synchronizes `self.device_id` before mapped pinned host buffers are freed.
            unsafe {
                if ffi::warp34_set_device(self.device_id) == 0 {
                    let _ = ffi::warp34_device_synchronize();
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn checkpoint_layout_27b_matches_41mib_compression_and_slot_tags() {
        // Qwen3.8-27B TP=1: 48 GDN layers, local_v_heads=48, d_v=128, d_k=128, conv_dim=10240 (taps=3).
        let layout = GdnCheckpointLayout::new(48, 48, 128, 128, 10240, 3, 0);
        let fp32_total_bytes = 48usize * 48 * 128 * 128 * 4 + layout.conv_bytes;
        assert!(
            layout.bundle_bytes < 42 * 1024 * 1024 && layout.bundle_bytes > 40 * 1024 * 1024,
            "27B WHT128-Q8_0 checkpoint bundle must be ~41 MiB, got {} bytes",
            layout.bundle_bytes
        );
        assert!(
            fp32_total_bytes as f64 / layout.bundle_bytes as f64 > 3.55,
            "expected > 3.55x compression vs FP32 GDN state"
        );
        assert_eq!(layout.bundle_bytes % SECTION_ALIGN_BYTES, 0);

        for id in [0usize, 1, 95] {
            let tagged = encode_checkpoint_pool_slot(id);
            assert!(is_checkpoint_pool_slot(tagged));
            assert_eq!(decode_checkpoint_pool_slot(tagged), Some(id));
        }
        assert!(!is_checkpoint_pool_slot(0));
        assert!(!is_checkpoint_pool_slot(31));
        assert_eq!(decode_checkpoint_pool_slot(7), None);
    }

    #[test]
    fn host_budget_planner_scales_across_tp_ranks_and_honors_cgroup_cap() {
        let layout = GdnCheckpointLayout::new(48, 48, 128, 128, 10240, 3, 0);
        let avail_64g = Some(64 * 1024 * 1024 * 1024usize);
        let slots_tp1 = plan_auto_host_checkpoint_slots(layout.bundle_bytes, 1, avail_64g);
        let slots_tp2 = plan_auto_host_checkpoint_slots(layout.bundle_bytes, 2, avail_64g);
        assert!(slots_tp1 >= 90);
        assert_eq!(slots_tp2, slots_tp1 / 2);

        // Tight 8 GiB cgroup cap -> 1 GiB checkpoint budget -> ~24 slots.
        let tight = parse_available_host_ram(
            "MemAvailable:   67108864 kB\n",
            Some("8589934592\n"),
            Some("0\n"),
        );
        assert_eq!(tight, Some(8 * 1024 * 1024 * 1024));
        let slots_tight = plan_auto_host_checkpoint_slots(layout.bundle_bytes, 1, tight);
        assert!((20..=26).contains(&slots_tight), "got {slots_tight}");
    }

    #[cfg(feature = "cuda")]
    fn populate_test_layers(
        active: &StateBundlePool,
        stream: &CudaStream,
    ) -> Vec<(Vec<u16>, Vec<f32>)> {
        let conv_elems = active.conv_dim * active.conv_tap_count;
        let ssm_rows = active.local_v_heads * active.head_v_dim;
        let ssm_elems = ssm_rows * 128;
        let mut layer_refs = Vec::with_capacity(active.layers.len());
        for (l_idx, layer) in active.layers.iter().enumerate() {
            let mut conv_all = vec![0u16; active.total_slots * conv_elems];
            for (i, dst) in conv_all[..conv_elems].iter_mut().enumerate() {
                *dst = 0x2c00u16 + (((i + l_idx * 7) * 13) % 3072) as u16;
            }
            layer.conv_state.copy_from_host(&conv_all, stream).unwrap();

            let mut ssm_all = vec![0.0f32; active.total_slots * ssm_elems];
            for r in 0..ssm_rows {
                let scale = match r % 4 {
                    0 => 0.0,   // exact-zero row
                    1 => 1e-12, // sub-FP16 normal row (tests BF16 E8M7 scales)
                    2 => 25.0,  // high-amplitude row
                    _ => 0.35,  // typical steady-state row
                };
                for c in 0..128 {
                    let phase = ((r * 31 + c * 17 + l_idx * 13) % 113) as f32 / 56.0 - 1.0;
                    let spike = if c == (r % 128) { 2.75 } else { 0.35 };
                    ssm_all[r * 128 + c] = scale * phase * spike;
                }
            }
            layer.ssm_state.copy_from_host(&ssm_all, stream).unwrap();
            layer_refs.push((
                conv_all[..conv_elems].to_vec(),
                ssm_all[..ssm_elems].to_vec(),
            ));
        }
        layer_refs
    }

    #[cfg(feature = "cuda")]
    fn verify_ssm_rows_snr(l_idx: usize, ssm_rows: usize, ssm_ref: &[f32], got_host: &[f32]) {
        let mut sig_all = 0.0f64;
        let mut err_all = 0.0f64;
        let mut sig_subnormal = 0.0f64;
        let mut err_subnormal = 0.0f64;
        for r in 0..ssm_rows {
            let row_ref = &ssm_ref[r * 128..(r + 1) * 128];
            let row_got = &got_host[r * 128..(r + 1) * 128];
            if r % 4 == 0 {
                assert!(
                    row_got.iter().all(|&x| x == 0.0),
                    "exact-zero row must unpack to exact 0.0"
                );
            } else if r % 4 == 1 {
                for (&a, &b) in row_ref.iter().zip(row_got) {
                    sig_subnormal += (a as f64) * (a as f64);
                    let d = (a - b) as f64;
                    err_subnormal += d * d;
                }
            } else {
                for (&a, &b) in row_ref.iter().zip(row_got) {
                    sig_all += (a as f64) * (a as f64);
                    let d = (a - b) as f64;
                    err_all += d * d;
                }
            }
        }
        let snr_db = 10.0 * (sig_all / err_all.max(1e-30)).log10();
        let sub_snr_db = 10.0 * (sig_subnormal / err_subnormal.max(1e-60)).log10();
        assert!(
            snr_db > 43.0,
            "layer {l_idx}: expected > 43 dB WHT128-Q8_0 SNR, got {snr_db:.2} dB"
        );
        assert!(
            sub_snr_db > 42.0,
            "layer {l_idx}: BF16 block scales must preserve 1e-12 rows (> 42 dB), got {sub_snr_db:.2} dB"
        );
    }

    #[cfg(feature = "cuda")]
    fn verify_unpacked_layers(
        active: &StateBundlePool,
        stream: &CudaStream,
        layer_refs: &[(Vec<u16>, Vec<f32>)],
        dst_host: usize,
        dst_vram: usize,
    ) {
        let conv_elems = active.conv_dim * active.conv_tap_count;
        let ssm_rows = active.local_v_heads * active.head_v_dim;
        let ssm_elems = ssm_rows * 128;
        for (l_idx, layer) in active.layers.iter().enumerate() {
            let (ref conv_ref, ref ssm_ref) = layer_refs[l_idx];
            let mut conv_u16 = vec![0u16; active.total_slots * conv_elems];
            layer
                .conv_state
                .copy_to_host(&mut conv_u16, stream)
                .unwrap();
            let mut ssm_f32 = vec![0.0f32; active.total_slots * ssm_elems];
            layer.ssm_state.copy_to_host(&mut ssm_f32, stream).unwrap();
            stream.synchronize().unwrap();

            assert_eq!(
                &conv_u16[dst_host * conv_elems..(dst_host + 1) * conv_elems],
                conv_ref.as_slice()
            );
            assert_eq!(
                &conv_u16[dst_vram * conv_elems..(dst_vram + 1) * conv_elems],
                conv_ref.as_slice()
            );

            let got_host = &ssm_f32[dst_host * ssm_elems..(dst_host + 1) * ssm_elems];
            let got_vram = &ssm_f32[dst_vram * ssm_elems..(dst_vram + 1) * ssm_elems];
            assert_eq!(got_host, got_vram);
            verify_ssm_rows_snr(l_idx, ssm_rows, ssm_ref, got_host);
        }
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn wht128_q8_roundtrip_and_l2_spill_restore() {
        let stream = CudaStream::new(0).expect("stream");
        let layer_ids = [0usize, 2, 5];
        let mut active = StateBundlePool::new(0, &layer_ids, 4, 128, 4, 4, 128, 64)
            .expect("active StateBundlePool");
        active
            .init_ple_state(32, 151645)
            .expect("init PLE conv + ngram state");

        let mut ckpt =
            GdnCheckpointPool::new(&active, 2, 3, &stream).expect("tiered GdnCheckpointPool");
        assert_eq!(ckpt.total_slots, 5);
        assert_eq!(ckpt.allocated_host_slots(), 0);

        let ple_elems = active.ple_conv_dim * 9;

        let src_slot = active.alloc_slot().unwrap();
        assert_eq!(src_slot, 0);
        let mut ple_all = vec![0u16; active.total_slots * ple_elems];
        for (i, dst) in ple_all[..ple_elems].iter_mut().enumerate() {
            *dst = 0x3000u16 + ((i * 19) % 2048) as u16;
        }
        active
            .ple_conv_state
            .as_ref()
            .unwrap()
            .copy_from_host(&ple_all, &stream)
            .unwrap();
        let ple_pattern = ple_all[..ple_elems].to_vec();
        let layer_refs = populate_test_layers(&active, &stream);

        // Pack 4 checkpoints into `vram_slots=2, host_slots=3` -> forces 2 L1->L2 LRU spills.
        let mut ckpt_slots = Vec::new();
        for step in 0..4 {
            active.ple_ngram_context[src_slot as usize] = [100 + step, 200 + step];
            let c_slot = ckpt.alloc_slot().expect("ckpt slot");
            ckpt.pack_from_active(&active, src_slot, c_slot, &stream)
                .expect("pack_from_active");
            ckpt_slots.push(c_slot);
        }
        stream.synchronize().unwrap();

        assert_eq!(
            (
                ckpt.vram_used_slots(),
                ckpt.host_used_slots(),
                ckpt.allocated_host_slots(),
                ckpt.host_spills,
            ),
            (2, 2, 2, 2)
        );
        assert_eq!(
            [
                ckpt.is_slot_in_host(ckpt_slots[0]),
                ckpt.is_slot_in_host(ckpt_slots[1]),
                ckpt.is_slot_in_vram(ckpt_slots[2]),
                ckpt.is_slot_in_vram(ckpt_slots[3]),
            ],
            [true, true, true, true]
        );

        let dst_host = active.alloc_slot().unwrap();
        let dst_vram = active.alloc_slot().unwrap();
        ckpt.unpack_to_active(&mut active, ckpt_slots[0], dst_host, &stream)
            .expect("unpack from L2 host");
        ckpt.unpack_to_active(&mut active, ckpt_slots[3], dst_vram, &stream)
            .expect("unpack from L1 VRAM");
        stream.synchronize().unwrap();

        assert_eq!(active.ple_ngram_context[dst_host as usize], [100, 200]);
        assert_eq!(active.ple_ngram_context[dst_vram as usize], [103, 203]);
        assert_eq!(ckpt.host_hits, 1);
        assert_eq!(ckpt.vram_hits, 1);

        let mut ple_u16 = vec![0u16; active.total_slots * ple_elems];
        active
            .ple_conv_state
            .as_ref()
            .unwrap()
            .copy_to_host(&mut ple_u16, &stream)
            .unwrap();
        stream.synchronize().unwrap();
        for dst in [dst_host as usize, dst_vram as usize] {
            assert_eq!(
                &ple_u16[dst * ple_elems..(dst + 1) * ple_elems],
                ple_pattern.as_slice()
            );
        }
        verify_unpacked_layers(
            &active,
            &stream,
            &layer_refs,
            dst_host as usize,
            dst_vram as usize,
        );

        for s in ckpt_slots {
            ckpt.free_slot(s);
        }
        assert_eq!(ckpt.vram_used_slots(), 0);
        assert_eq!(ckpt.host_used_slots(), 0);
        assert_eq!(ckpt.free_count(), 5);

        ckpt.truncate_slots(1, 2);
        assert_eq!(ckpt.vram_slots, 1);
        assert_eq!(ckpt.host_slots, 2);
        assert_eq!(ckpt.total_slots, 3);
    }
}
