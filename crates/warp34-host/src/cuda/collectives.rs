use super::*;

#[cfg(feature = "cuda")]
pub enum CommBuffer {
    MappedHost(MappedHostBuffer),
    DeviceVram(DeviceBuffer),
}

#[cfg(feature = "cuda")]
impl CommBuffer {
    #[inline]
    pub fn dev_ptr(&self) -> *mut c_void {
        match self {
            Self::MappedHost(b) => b.dev_ptr,
            Self::DeviceVram(b) => b.ptr,
        }
    }
}

/// Per-rank async Copy-Engine DMA pipeline for full-duplex PCIe `TP=2` AllReduce.
#[cfg(feature = "cuda")]
pub struct Tp2DmaPipe {
    pub stream: CudaStream,
    pub ev_ready: CudaEvent,
    pub ev_tx_done: [CudaEvent; 8],
    pub ev_tx_handles: [usize; 8],
}

/// Pinned-sysmem or P2P-VRAM (`NVLink`) `TP=2` / `TP=4` collective context shared across ranks.
#[cfg(feature = "cuda")]
pub struct Tp2Collectives {
    pub tp_size: usize,
    pub p2p_vram: bool,
    pub sigs: Vec<CommBuffer>,
    pub stages: Vec<CommBuffer>,
    pub max_stage_bytes_per_epoch: usize,
    pub dma_pipes: Vec<Tp2DmaPipe>,
}

/// Byte threshold (`64 KiB`) at and above which `TP=2` collectives use Copy-Engine
/// `cudaMemcpyAsync` TX into `[0, max_stage_bytes_per_epoch)` followed by 2-barrier SM RX.
pub const TP2_DMA_THRESHOLD_BYTES: usize = 65_536;
/// Byte threshold (`512 KiB`) at and above which `TP=2` PCIe collectives overlap
/// multi-chunk Copy-Engine DMA TX (`dma_pipes[rank].stream`) with SM RX (`ex.stream`).
pub const TP2_DMA_PIPELINE_MIN_BYTES: usize = 512 * 1024;
/// Number of `max_stage_bytes_per_epoch` slots allocated per rank (`[0, cap)` for 2-barrier
/// DMA TX, `[cap, 2 * cap)` for even 1-barrier epochs, and `[2 * cap, 3 * cap)` for odd 1-barrier epochs).
const TP2_STAGE_EPOCH_SLOTS: usize = 3;
/// Minimum per-rank staging buffer byte allocation (`4 KiB`).
const TP2_MIN_STAGE_BYTES: usize = 4_096;

#[cfg(feature = "cuda")]
impl Tp2Collectives {
    /// Allocates 3-slot staging (`3 * max_stage_bytes_per_epoch`) and signal structs for `TP=2`.
    pub fn new(max_stage_bytes_per_epoch: usize) -> Result<Self> {
        Self::new_for_tp(2, max_stage_bytes_per_epoch)
    }

    /// Allocates 3-slot staging (`3 * max_stage_bytes_per_epoch`) and signal structs for `tp_size` (`2` or `4`),
    /// automatically enabling P2P VRAM staging when all rank pairs have native NVLink P2P support.
    pub fn new_for_tp(tp_size: usize, max_stage_bytes_per_epoch: usize) -> Result<Self> {
        Self::new_for_tp_with_p2p(tp_size, max_stage_bytes_per_epoch, true)
    }

    /// Allocates collective buffers for `tp_size` (`2` or `4`), optionally allowing P2P VRAM staging.
    pub fn new_for_tp_with_p2p(
        tp_size: usize,
        max_stage_bytes_per_epoch: usize,
        allow_p2p: bool,
    ) -> Result<Self> {
        if !matches!(tp_size, 2 | 4) {
            return Err(CudaError::new(
                -1,
                "Tp2Collectives supports tp_size == 2 or tp_size == 4",
            ));
        }
        // SAFETY: `warp34_tp2_signal_bytes` is a pure `sizeof(Warp34Tp2Signal)` query with no preconditions.
        let sig_bytes = unsafe { ffi::warp34_tp2_signal_bytes() };
        let max_stage_bytes_per_epoch = max_stage_bytes_per_epoch.saturating_add(15) & !15;
        // Reserve `[0, max_stage_bytes_per_epoch)` exclusively for 2-barrier Copy-Engine DMA TX
        // (`>= 64 KiB`), and `[max_stage_bytes_per_epoch, 3 * max_stage_bytes_per_epoch)` for
        // 1-barrier `stage_epoch_ptr` even/odd ping-pong transfers (both `< 64 KiB` and `>= 64 KiB`
        // gathers) so 1-barrier and 2-barrier DMA transfers never share staging memory.
        let stage_bytes = max_stage_bytes_per_epoch
            .saturating_mul(TP2_STAGE_EPOCH_SLOTS)
            .max(TP2_MIN_STAGE_BYTES);
        // `Signal::end[128][0]` (u32 offset `144 * 16 + 128 * 16 = 4352`) stores the
        // per-epoch byte stride so `odd` and `even` staging halves are disjoint across
        // variable-size back-to-back collectives.
        const SIG_EPOCH_CAP_U32_IDX: usize = 144 * 16 + 128 * 16;
        let cap_u32 = u32::try_from(max_stage_bytes_per_epoch).unwrap_or(!15u32);

        let mut all_p2p = allow_p2p;
        if all_p2p {
            for i in 0..tp_size {
                for j in (i + 1)..tp_size {
                    let mut can_p2p = 0i32;
                    // SAFETY: queries and enables bidirectional peer access between `i` and `j`.
                    let rc = unsafe {
                        ffi::warp34_device_can_enable_p2p(i as i32, j as i32, 1, &raw mut can_p2p)
                    };
                    if rc != 0 || can_p2p == 0 {
                        all_p2p = false;
                        break;
                    }
                }
                if !all_p2p {
                    break;
                }
            }
        }

        let mut sigs = Vec::with_capacity(tp_size);
        let mut stages = Vec::with_capacity(tp_size);
        if all_p2p {
            let mut sig_init = vec![0u32; sig_bytes / 4];
            if let Some(slot) = sig_init.get_mut(SIG_EPOCH_CAP_U32_IDX) {
                *slot = cap_u32;
            }
            for r in 0..tp_size {
                let sig = DeviceBuffer::alloc(r as i32, sig_bytes)?;
                let stream = super::CudaStream::new(r as i32)?;
                sig.copy_from_host(&sig_init, &stream)?;
                stream.synchronize()?;
                sigs.push(CommBuffer::DeviceVram(sig));
                stages.push(CommBuffer::DeviceVram(DeviceBuffer::alloc(
                    r as i32,
                    stage_bytes,
                )?));
            }
        } else {
            let num_gpus = super::CudaDevice::count()?.max(1) as usize;
            for r in 0..tp_size {
                let dev_id = (r % num_gpus) as i32;
                check_rc(
                    // SAFETY: binds calling thread to a valid visible device before allocating
                    // portable mapped host memory for rank `r` (wraps modulo `num_gpus` so
                    // TP=4 mapped-host simulation tests work on 1- or 2-GPU hosts).
                    unsafe { ffi::warp34_set_device(dev_id) },
                    "warp34_set_device",
                )?;
                let sig = MappedHostBuffer::alloc(sig_bytes)?;
                if SIG_EPOCH_CAP_U32_IDX * 4 + 4 <= sig.len_bytes {
                    // SAFETY: `sig.host_ptr` is 128B-aligned mapped host memory of size `sig.len_bytes`,
                    // and `SIG_EPOCH_CAP_U32_IDX * 4 + 4 <= sig.len_bytes`.
                    unsafe {
                        sig.host_ptr
                            .cast::<u32>()
                            .add(SIG_EPOCH_CAP_U32_IDX)
                            .write(cap_u32);
                    }
                }
                sigs.push(CommBuffer::MappedHost(sig));
                stages.push(CommBuffer::MappedHost(MappedHostBuffer::alloc(
                    stage_bytes,
                )?));
            }
        }
        let mut dma_pipes = Vec::new();
        if tp_size == 2 && !all_p2p && super::CudaDevice::count().unwrap_or(0) >= 2 {
            for dev in 0..2i32 {
                let stream = super::CudaStream::new(dev)?;
                let ev_ready = super::CudaEvent::new(dev, false)?;
                let ev0 = super::CudaEvent::new(dev, false)?;
                let ev1 = super::CudaEvent::new(dev, false)?;
                let ev2 = super::CudaEvent::new(dev, false)?;
                let ev3 = super::CudaEvent::new(dev, false)?;
                let ev4 = super::CudaEvent::new(dev, false)?;
                let ev5 = super::CudaEvent::new(dev, false)?;
                let ev6 = super::CudaEvent::new(dev, false)?;
                let ev7 = super::CudaEvent::new(dev, false)?;
                let ev_tx_handles = [
                    ev0.handle, ev1.handle, ev2.handle, ev3.handle, ev4.handle, ev5.handle,
                    ev6.handle, ev7.handle,
                ];
                dma_pipes.push(Tp2DmaPipe {
                    stream,
                    ev_ready,
                    ev_tx_done: [ev0, ev1, ev2, ev3, ev4, ev5, ev6, ev7],
                    ev_tx_handles,
                });
            }
        }
        Ok(Self {
            tp_size,
            p2p_vram: all_p2p,
            sigs,
            stages,
            max_stage_bytes_per_epoch,
            dma_pipes,
        })
    }

    pub fn all_reduce_f16(
        &self,
        ex: &Warp34Exec,
        local_in_ptr: u64,
        out_ptr: u64,
        rank: i32,
        num_elems: i32,
    ) -> Result<()> {
        assert!(
            num_elems >= 0
                && (num_elems as usize).saturating_mul(2) <= self.max_stage_bytes_per_epoch,
            "all_reduce_f16 exceeds max_stage_bytes_per_epoch"
        );
        if self.tp_size == 4 {
            return check_rc(
                // SAFETY: staging capacity is asserted above; `stages[0..4]` and `sigs[0..4]` are live
                // buffers owned by `self`, and caller guarantees valid `local_in_ptr`/`out_ptr` for `num_elems` FP16s.
                unsafe {
                    ffi::warp34_tp4_all_reduce_f16(
                        ex,
                        local_in_ptr as *const c_void,
                        out_ptr as *mut c_void,
                        self.stages[0].dev_ptr(),
                        self.stages[1].dev_ptr(),
                        self.stages[2].dev_ptr(),
                        self.stages[3].dev_ptr(),
                        self.sigs[0].dev_ptr(),
                        self.sigs[1].dev_ptr(),
                        self.sigs[2].dev_ptr(),
                        self.sigs[3].dev_ptr(),
                        rank,
                        num_elems,
                    )
                },
                "warp34_tp4_all_reduce_f16",
            );
        }
        if self.tp_size == 2
            && !self.p2p_vram
            && self.dma_pipes.len() == 2
            && (rank == 0 || rank == 1)
            && (num_elems as usize).saturating_mul(2) >= TP2_DMA_PIPELINE_MIN_BYTES
        {
            return self.all_reduce_pipelined_f16(ex, local_in_ptr, out_ptr, rank, num_elems);
        }
        check_rc(
            // SAFETY: staging capacity is asserted above; `stages[0..2]` and `sigs[0..2]` are live
            // buffers owned by `self`, and caller guarantees valid `local_in_ptr`/`out_ptr` for `num_elems` FP16s.
            unsafe {
                ffi::warp34_tp2_all_reduce_f16(
                    ex,
                    local_in_ptr as *const c_void,
                    out_ptr as *mut c_void,
                    self.stages[0].dev_ptr(),
                    self.stages[1].dev_ptr(),
                    self.sigs[0].dev_ptr(),
                    self.sigs[1].dev_ptr(),
                    rank,
                    num_elems,
                )
            },
            "warp34_tp2_all_reduce_f16",
        )
    }

    /// Full-duplex pipelined `TP=2` `FP16` AllReduce for real 2-GPU execution (`rank` running on device `rank`).
    /// Overlaps Copy-Engine PCIe Upstream DMA TX (`dma_pipes[rank].stream`) with SM PCIe Downstream RX (`ex.stream`)
    /// across up to 8 chunks, while remaining 100% CUDA-Graph capturable.
    pub fn all_reduce_pipelined_f16(
        &self,
        ex: &Warp34Exec,
        local_in_ptr: u64,
        out_ptr: u64,
        rank: i32,
        num_elems: i32,
    ) -> Result<()> {
        if num_elems < 0 || (num_elems as usize).saturating_mul(2) > self.max_stage_bytes_per_epoch
        {
            return Err(CudaError::new(
                -10,
                "all_reduce_pipelined_f16 num_elems out of range for max_stage_bytes_per_epoch",
            ));
        }
        if rank < 0 || (rank as usize) >= self.tp_size {
            return Err(CudaError::new(
                -10,
                "all_reduce_pipelined_f16 rank out of range",
            ));
        }
        if self.tp_size != 2 || self.dma_pipes.len() < 2 {
            return check_rc(
                // SAFETY: `stages` and `sigs` are live buffers owned by `self` with sufficient capacity.
                unsafe {
                    if self.tp_size == 4 {
                        ffi::warp34_tp4_all_reduce_f16(
                            ex,
                            local_in_ptr as *const c_void,
                            out_ptr as *mut c_void,
                            self.stages[0].dev_ptr(),
                            self.stages[1].dev_ptr(),
                            self.stages[2].dev_ptr(),
                            self.stages[3].dev_ptr(),
                            self.sigs[0].dev_ptr(),
                            self.sigs[1].dev_ptr(),
                            self.sigs[2].dev_ptr(),
                            self.sigs[3].dev_ptr(),
                            rank,
                            num_elems,
                        )
                    } else {
                        ffi::warp34_tp2_all_reduce_f16(
                            ex,
                            local_in_ptr as *const c_void,
                            out_ptr as *mut c_void,
                            self.stages[0].dev_ptr(),
                            self.stages[1].dev_ptr(),
                            self.sigs[0].dev_ptr(),
                            self.sigs[1].dev_ptr(),
                            rank,
                            num_elems,
                        )
                    }
                },
                "warp34_tp2_all_reduce_f16",
            );
        }
        let pipe = &self.dma_pipes[rank as usize];
        check_rc(
            // SAFETY: `stages[0..2]`, `sigs[0..2]`, and `pipe` belong to `self` on device `rank`;
            // caller guarantees `local_in_ptr`/`out_ptr` span `num_elems` FP16 elements.
            unsafe {
                ffi::warp34_tp2_all_reduce_pipelined_f16(
                    ex,
                    local_in_ptr as *const c_void,
                    out_ptr as *mut c_void,
                    self.stages[0].dev_ptr(),
                    self.stages[1].dev_ptr(),
                    self.sigs[0].dev_ptr(),
                    self.sigs[1].dev_ptr(),
                    rank,
                    num_elems,
                    pipe.stream.handle,
                    pipe.ev_ready.handle,
                    pipe.ev_tx_handles.as_ptr(),
                    8,
                )
            },
            "warp34_tp2_all_reduce_pipelined_f16",
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub fn all_reduce_residual_gemma_rms_norm_f16(
        &self,
        ex: &Warp34Exec,
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
        self.all_reduce_residual_gemma_rms_norm_hadamard_f16(
            ex,
            local_in_ptr,
            ar_out_ptr,
            residual_ptr,
            weight_ptr,
            out_ptr,
            0,
            tap_dst_bf16_ptr,
            rank,
            num_tokens,
            hidden_size,
            eps,
            tap_row_stride,
            tap_col_offset,
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub fn all_reduce_residual_gemma_rms_norm_hadamard_f16(
        &self,
        ex: &Warp34Exec,
        local_in_ptr: u64,
        ar_out_ptr: u64,
        residual_ptr: u64,
        weight_ptr: u64,
        out_ptr: u64,
        out_had64_ptr: u64,
        tap_dst_bf16_ptr: u64,
        rank: i32,
        num_tokens: i32,
        hidden_size: i32,
        eps: f32,
        tap_row_stride: i32,
        tap_col_offset: i32,
    ) -> Result<()> {
        assert!(
            num_tokens >= 0 && hidden_size > 0,
            "all_reduce_residual_gemma_rms_norm_f16 invalid dimensions"
        );
        let num_elems = (num_tokens as usize).saturating_mul(hidden_size as usize);
        assert!(
            num_elems.saturating_mul(2) <= self.max_stage_bytes_per_epoch,
            "all_reduce_residual_gemma_rms_norm_f16 exceeds max_stage_bytes_per_epoch"
        );
        if self.tp_size == 4 {
            return check_rc(
                // SAFETY: staging capacity is asserted above; `stages[0..4]` and `sigs[0..4]` are live
                // buffers owned by `self`, and caller guarantees valid device pointers for `[num_tokens, hidden_size]`.
                unsafe {
                    ffi::warp34_tp4_all_reduce_residual_gemma_rms_norm_hadamard_f16(
                        ex,
                        local_in_ptr as *const c_void,
                        ar_out_ptr as *mut c_void,
                        residual_ptr as *mut c_void,
                        weight_ptr as *const c_void,
                        out_ptr as *mut c_void,
                        out_had64_ptr as *mut c_void,
                        tap_dst_bf16_ptr as *mut c_void,
                        self.stages[0].dev_ptr(),
                        self.stages[1].dev_ptr(),
                        self.stages[2].dev_ptr(),
                        self.stages[3].dev_ptr(),
                        self.sigs[0].dev_ptr(),
                        self.sigs[1].dev_ptr(),
                        self.sigs[2].dev_ptr(),
                        self.sigs[3].dev_ptr(),
                        rank,
                        num_tokens,
                        hidden_size,
                        eps,
                        tap_row_stride,
                        tap_col_offset,
                    )
                },
                "warp34_tp4_all_reduce_residual_gemma_rms_norm_hadamard_f16",
            );
        }
        if self.tp_size == 2
            && !self.p2p_vram
            && self.dma_pipes.len() == 2
            && (rank == 0 || rank == 1)
            && num_elems.saturating_mul(2) >= TP2_DMA_PIPELINE_MIN_BYTES
        {
            let pipe = &self.dma_pipes[rank as usize];
            return check_rc(
                // SAFETY: staging capacity is asserted above; `stages[0..2]`, `sigs[0..2]`, and `pipe`
                // belong to `self` on device `rank`, and caller guarantees valid device pointers.
                unsafe {
                    ffi::warp34_tp2_all_reduce_residual_gemma_rms_norm_hadamard_pipelined_f16(
                        ex,
                        local_in_ptr as *const c_void,
                        ar_out_ptr as *mut c_void,
                        residual_ptr as *mut c_void,
                        weight_ptr as *const c_void,
                        out_ptr as *mut c_void,
                        out_had64_ptr as *mut c_void,
                        tap_dst_bf16_ptr as *mut c_void,
                        self.stages[0].dev_ptr(),
                        self.stages[1].dev_ptr(),
                        self.sigs[0].dev_ptr(),
                        self.sigs[1].dev_ptr(),
                        rank,
                        num_tokens,
                        hidden_size,
                        eps,
                        tap_row_stride,
                        tap_col_offset,
                        pipe.stream.handle,
                        pipe.ev_ready.handle,
                        pipe.ev_tx_handles.as_ptr(),
                        8,
                    )
                },
                "warp34_tp2_all_reduce_residual_gemma_rms_norm_hadamard_pipelined_f16",
            );
        }
        check_rc(
            // SAFETY: staging capacity is asserted above; `stages[0..2]` and `sigs[0..2]` are live
            // buffers owned by `self`, and caller guarantees valid device pointers for `[num_tokens, hidden_size]`.
            unsafe {
                ffi::warp34_tp2_all_reduce_residual_gemma_rms_norm_hadamard_f16(
                    ex,
                    local_in_ptr as *const c_void,
                    ar_out_ptr as *mut c_void,
                    residual_ptr as *mut c_void,
                    weight_ptr as *const c_void,
                    out_ptr as *mut c_void,
                    out_had64_ptr as *mut c_void,
                    tap_dst_bf16_ptr as *mut c_void,
                    self.stages[0].dev_ptr(),
                    self.stages[1].dev_ptr(),
                    self.sigs[0].dev_ptr(),
                    self.sigs[1].dev_ptr(),
                    rank,
                    num_tokens,
                    hidden_size,
                    eps,
                    tap_row_stride,
                    tap_col_offset,
                )
            },
            "warp34_tp2_all_reduce_residual_gemma_rms_norm_f16",
        )
    }

    pub fn all_reduce_bf16(
        &self,
        ex: &Warp34Exec,
        local_in_ptr: u64,
        out_ptr: u64,
        rank: i32,
        num_elems: i32,
    ) -> Result<()> {
        assert!(
            (num_elems as usize) * 2 <= self.max_stage_bytes_per_epoch,
            "all_reduce_bf16 exceeds max_stage_bytes_per_epoch"
        );
        if self.tp_size == 4 {
            return check_rc(
                // SAFETY: staging capacity is asserted above; `stages[0..4]` and `sigs[0..4]` are live
                // buffers owned by `self`, and caller guarantees valid BF16 device buffers for `num_elems`.
                unsafe {
                    ffi::warp34_tp4_all_reduce_bf16(
                        ex,
                        local_in_ptr as *const c_void,
                        out_ptr as *mut c_void,
                        self.stages[0].dev_ptr(),
                        self.stages[1].dev_ptr(),
                        self.stages[2].dev_ptr(),
                        self.stages[3].dev_ptr(),
                        self.sigs[0].dev_ptr(),
                        self.sigs[1].dev_ptr(),
                        self.sigs[2].dev_ptr(),
                        self.sigs[3].dev_ptr(),
                        rank,
                        num_elems,
                    )
                },
                "warp34_tp4_all_reduce_bf16",
            );
        }
        check_rc(
            // SAFETY: staging capacity is asserted above; `stages[0..2]` and `sigs[0..2]` are live
            // buffers owned by `self`, and caller guarantees valid BF16 device buffers for `num_elems`.
            unsafe {
                ffi::warp34_tp2_all_reduce_bf16(
                    ex,
                    local_in_ptr as *const c_void,
                    out_ptr as *mut c_void,
                    self.stages[0].dev_ptr(),
                    self.stages[1].dev_ptr(),
                    self.sigs[0].dev_ptr(),
                    self.sigs[1].dev_ptr(),
                    rank,
                    num_elems,
                )
            },
            "warp34_tp2_all_reduce_bf16",
        )
    }

    pub fn all_reduce_f32(
        &self,
        ex: &Warp34Exec,
        local_in_ptr: u64,
        out_ptr: u64,
        rank: i32,
        num_elems: i32,
    ) -> Result<()> {
        assert!(
            (num_elems as usize) * 4 <= self.max_stage_bytes_per_epoch,
            "all_reduce_f32 exceeds max_stage_bytes_per_epoch"
        );
        if self.tp_size == 4 {
            return check_rc(
                // SAFETY: staging capacity is asserted above; `stages[0..4]` and `sigs[0..4]` are live
                // buffers owned by `self`, and caller guarantees valid FP32 device buffers for `num_elems`.
                unsafe {
                    ffi::warp34_tp4_all_reduce_f32(
                        ex,
                        local_in_ptr as *const f32,
                        out_ptr as *mut f32,
                        self.stages[0].dev_ptr() as *mut f32,
                        self.stages[1].dev_ptr() as *mut f32,
                        self.stages[2].dev_ptr() as *mut f32,
                        self.stages[3].dev_ptr() as *mut f32,
                        self.sigs[0].dev_ptr(),
                        self.sigs[1].dev_ptr(),
                        self.sigs[2].dev_ptr(),
                        self.sigs[3].dev_ptr(),
                        rank,
                        num_elems,
                    )
                },
                "warp34_tp4_all_reduce_f32",
            );
        }
        check_rc(
            // SAFETY: staging capacity is asserted above; `stages[0..2]` and `sigs[0..2]` are live
            // buffers owned by `self`, and caller guarantees valid FP32 device buffers for `num_elems`.
            unsafe {
                ffi::warp34_tp2_all_reduce_f32(
                    ex,
                    local_in_ptr as *const f32,
                    out_ptr as *mut f32,
                    self.stages[0].dev_ptr() as *mut f32,
                    self.stages[1].dev_ptr() as *mut f32,
                    self.sigs[0].dev_ptr(),
                    self.sigs[1].dev_ptr(),
                    rank,
                    num_elems,
                )
            },
            "warp34_tp2_all_reduce_f32",
        )
    }

    pub fn broadcast_i32(
        &self,
        ex: &Warp34Exec,
        src_rank0_ptr: u64,
        out_ptr: u64,
        rank: i32,
        num_elems: i32,
    ) -> Result<()> {
        assert!(
            (num_elems as usize) * 4 <= self.max_stage_bytes_per_epoch,
            "broadcast_i32 exceeds max_stage_bytes_per_epoch"
        );
        if self.tp_size == 4 {
            return check_rc(
                // SAFETY: staging capacity is asserted above; `stages[0..4]` and `sigs[0..4]` are live
                // buffers owned by `self`, and caller guarantees valid `i32` device buffers for `num_elems`.
                unsafe {
                    ffi::warp34_tp4_broadcast_i32(
                        ex,
                        src_rank0_ptr as *const i32,
                        out_ptr as *mut i32,
                        self.stages[0].dev_ptr() as *mut i32,
                        self.stages[1].dev_ptr() as *mut i32,
                        self.stages[2].dev_ptr() as *mut i32,
                        self.stages[3].dev_ptr() as *mut i32,
                        self.sigs[0].dev_ptr(),
                        self.sigs[1].dev_ptr(),
                        self.sigs[2].dev_ptr(),
                        self.sigs[3].dev_ptr(),
                        rank,
                        num_elems,
                    )
                },
                "warp34_tp4_broadcast_i32",
            );
        }
        check_rc(
            // SAFETY: staging capacity is asserted above; `stages[0..2]` and `sigs[0..2]` are live
            // buffers owned by `self`, and caller guarantees valid `i32` device buffers for `num_elems`.
            unsafe {
                ffi::warp34_tp2_broadcast_i32(
                    ex,
                    src_rank0_ptr as *const i32,
                    out_ptr as *mut i32,
                    self.stages[0].dev_ptr() as *mut i32,
                    self.stages[1].dev_ptr() as *mut i32,
                    self.sigs[0].dev_ptr(),
                    self.sigs[1].dev_ptr(),
                    rank,
                    num_elems,
                )
            },
            "warp34_tp2_broadcast_i32",
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub fn all_gather_top1_f32_i32(
        &self,
        ex: &Warp34Exec,
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
        if self.tp_size == 4 {
            return check_rc(
                // SAFETY: staging capacity is asserted above; `stages[0..4]` and `sigs[0..4]` are live
                // buffers owned by `self`, and caller guarantees valid `(f32, i32)` row buffers for `num_rows`.
                unsafe {
                    ffi::warp34_tp4_all_gather_top1_f32_i32(
                        ex,
                        local_max_val_ptr as *const f32,
                        local_max_id_ptr as *const i32,
                        global_max_val_ptr as *mut f32,
                        global_max_id_ptr as *mut i32,
                        self.stages[0].dev_ptr(),
                        self.stages[1].dev_ptr(),
                        self.stages[2].dev_ptr(),
                        self.stages[3].dev_ptr(),
                        self.sigs[0].dev_ptr(),
                        self.sigs[1].dev_ptr(),
                        self.sigs[2].dev_ptr(),
                        self.sigs[3].dev_ptr(),
                        rank,
                        num_rows,
                    )
                },
                "warp34_tp4_all_gather_top1_f32_i32",
            );
        }
        check_rc(
            // SAFETY: staging capacity is asserted above; `stages[0..2]` and `sigs[0..2]` are live
            // buffers owned by `self`, and caller guarantees valid `(f32, i32)` row buffers for `num_rows`.
            unsafe {
                ffi::warp34_tp2_all_gather_top1_f32_i32(
                    ex,
                    local_max_val_ptr as *const f32,
                    local_max_id_ptr as *const i32,
                    global_max_val_ptr as *mut f32,
                    global_max_id_ptr as *mut i32,
                    self.stages[0].dev_ptr(),
                    self.stages[1].dev_ptr(),
                    self.sigs[0].dev_ptr(),
                    self.sigs[1].dev_ptr(),
                    rank,
                    num_rows,
                )
            },
            "warp34_tp2_all_gather_top1_f32_i32",
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub fn all_gather_topk_f32_i32(
        &self,
        ex: &Warp34Exec,
        local_vals_ptr: u64,
        local_ids_ptr: u64,
        global_vals_ptr: u64,
        global_ids_ptr: u64,
        rank: i32,
        num_rows: i32,
        top_k: i32,
    ) -> Result<()> {
        assert!(
            (num_rows as usize) * (top_k as usize) * 8 <= self.max_stage_bytes_per_epoch,
            "all_gather_topk_f32_i32 exceeds max_stage_bytes_per_epoch"
        );
        if self.tp_size == 4 {
            return check_rc(
                // SAFETY: staging capacity is asserted above; `stages[0..4]` and `sigs[0..4]` are live
                // buffers owned by `self`, and caller guarantees valid `(f32, i32)` buffers for `num_rows * top_k`.
                unsafe {
                    ffi::warp34_tp4_all_gather_topk_f32_i32(
                        ex,
                        local_vals_ptr as *const f32,
                        local_ids_ptr as *const i32,
                        global_vals_ptr as *mut f32,
                        global_ids_ptr as *mut i32,
                        self.stages[0].dev_ptr(),
                        self.stages[1].dev_ptr(),
                        self.stages[2].dev_ptr(),
                        self.stages[3].dev_ptr(),
                        self.sigs[0].dev_ptr(),
                        self.sigs[1].dev_ptr(),
                        self.sigs[2].dev_ptr(),
                        self.sigs[3].dev_ptr(),
                        rank,
                        num_rows,
                        top_k,
                    )
                },
                "warp34_tp4_all_gather_topk_f32_i32",
            );
        }
        check_rc(
            // SAFETY: staging capacity is asserted above; `stages[0..2]` and `sigs[0..2]` are live
            // buffers owned by `self`, and caller guarantees valid `(f32, i32)` buffers for `num_rows * top_k`.
            unsafe {
                ffi::warp34_tp2_all_gather_topk_f32_i32(
                    ex,
                    local_vals_ptr as *const f32,
                    local_ids_ptr as *const i32,
                    global_vals_ptr as *mut f32,
                    global_ids_ptr as *mut i32,
                    self.stages[0].dev_ptr(),
                    self.stages[1].dev_ptr(),
                    self.sigs[0].dev_ptr(),
                    self.sigs[1].dev_ptr(),
                    rank,
                    num_rows,
                    top_k,
                )
            },
            "warp34_tp2_all_gather_topk_f32_i32",
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub fn all_reduce_hc_combine_tap_f16(
        &self,
        ex: &Warp34Exec,
        residual_in_ptr: u64,
        local_block_out_ptr: u64,
        ar_out_ptr: u64,
        inject_weights_ptr: u64,
        ple_delta_ptr: u64,
        residual_out_ptr: u64,
        tap_bf16_out_ptr: u64,
        valid_token_mask_ptr: u64,
        rank: i32,
        num_tokens: i32,
        hc_count: i32,
        hidden_size: i32,
        tap_stride_elems: i32,
        tap_offset_elems: i32,
    ) -> Result<()> {
        let num_elems = (num_tokens as usize) * (hidden_size as usize);
        assert!(
            num_elems * 2 <= self.max_stage_bytes_per_epoch,
            "all_reduce_hc_combine_tap_f16 exceeds max_stage_bytes_per_epoch"
        );
        if self.tp_size == 4 {
            return check_rc(
                // SAFETY: staging capacity is asserted above; `stages[0..4]` and `sigs[0..4]` are live
                // buffers owned by `self`, and caller guarantees valid device buffers for `num_tokens`.
                unsafe {
                    ffi::warp34_tp4_all_reduce_hc_combine_tap_f16(
                        ex,
                        residual_in_ptr as *const c_void,
                        local_block_out_ptr as *const c_void,
                        ar_out_ptr as *mut c_void,
                        inject_weights_ptr as *const f32,
                        ple_delta_ptr as *const c_void,
                        residual_out_ptr as *mut c_void,
                        tap_bf16_out_ptr as *mut c_void,
                        valid_token_mask_ptr as *const u8,
                        self.stages[0].dev_ptr(),
                        self.stages[1].dev_ptr(),
                        self.stages[2].dev_ptr(),
                        self.stages[3].dev_ptr(),
                        self.sigs[0].dev_ptr(),
                        self.sigs[1].dev_ptr(),
                        self.sigs[2].dev_ptr(),
                        self.sigs[3].dev_ptr(),
                        rank,
                        num_tokens,
                        hc_count,
                        hidden_size,
                        tap_stride_elems,
                        tap_offset_elems,
                    )
                },
                "warp34_tp4_all_reduce_hc_combine_tap_f16",
            );
        }
        check_rc(
            // SAFETY: staging capacity is asserted above; `stages[0..2]` and `sigs[0..2]` are live
            // buffers owned by `self`, and caller guarantees valid device buffers for `num_tokens`.
            unsafe {
                ffi::warp34_tp2_all_reduce_hc_combine_tap_f16(
                    ex,
                    residual_in_ptr as *const c_void,
                    local_block_out_ptr as *const c_void,
                    ar_out_ptr as *mut c_void,
                    inject_weights_ptr as *const f32,
                    ple_delta_ptr as *const c_void,
                    residual_out_ptr as *mut c_void,
                    tap_bf16_out_ptr as *mut c_void,
                    valid_token_mask_ptr as *const u8,
                    self.stages[0].dev_ptr(),
                    self.stages[1].dev_ptr(),
                    self.sigs[0].dev_ptr(),
                    self.sigs[1].dev_ptr(),
                    rank,
                    num_tokens,
                    hc_count,
                    hidden_size,
                    tap_stride_elems,
                    tap_offset_elems,
                )
            },
            "warp34_tp2_all_reduce_hc_combine_tap_f16",
        )
    }

    pub fn gather_vocab_f16(
        &self,
        ex: &Warp34Exec,
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
        if self.tp_size == 4 {
            return check_rc(
                // SAFETY: staging capacity is asserted above; `stages[0..4]` and `sigs[0..4]` are live
                // buffers owned by `self`, and caller guarantees valid FP16 shard/full buffers.
                unsafe {
                    ffi::warp34_tp4_gather_vocab_f16(
                        ex,
                        local_logits_ptr as *const c_void,
                        rank0_full_logits_ptr as *mut c_void,
                        self.stages[0].dev_ptr(),
                        self.stages[1].dev_ptr(),
                        self.stages[2].dev_ptr(),
                        self.stages[3].dev_ptr(),
                        self.sigs[0].dev_ptr(),
                        self.sigs[1].dev_ptr(),
                        self.sigs[2].dev_ptr(),
                        self.sigs[3].dev_ptr(),
                        rank,
                        num_rows,
                        local_vocab_size,
                    )
                },
                "warp34_tp4_gather_vocab_f16",
            );
        }
        check_rc(
            // SAFETY: staging capacity is asserted above; `stages[0..2]` and `sigs[0..2]` are live
            // buffers owned by `self`, and caller guarantees valid FP16 shard/full buffers.
            unsafe {
                ffi::warp34_tp2_gather_vocab_f16(
                    ex,
                    local_logits_ptr as *const c_void,
                    rank0_full_logits_ptr as *mut c_void,
                    self.stages[0].dev_ptr(),
                    self.stages[1].dev_ptr(),
                    self.sigs[0].dev_ptr(),
                    self.sigs[1].dev_ptr(),
                    rank,
                    num_rows,
                    local_vocab_size,
                )
            },
            "warp34_tp2_gather_vocab_f16",
        )
    }
}
