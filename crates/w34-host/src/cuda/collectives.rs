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

/// Pinned-sysmem or P2P-VRAM (`NVLink`) `TP=2` / `TP=4` collective context shared across ranks.
#[cfg(feature = "cuda")]
pub struct Tp2Collectives {
    pub tp_size: usize,
    pub p2p_vram: bool,
    pub sigs: Vec<CommBuffer>,
    pub stages: Vec<CommBuffer>,
    pub max_stage_bytes_per_epoch: usize,
}

/// Byte threshold (`64 KiB`) at and above which `TP=2` collectives use Copy-Engine
/// `cudaMemcpyAsync` TX into `[0, max_stage_bytes_per_epoch)` followed by 2-barrier SM RX.
pub const TP2_DMA_THRESHOLD_BYTES: usize = 65_536;
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
        // SAFETY: `w34_tp2_signal_bytes` is a pure `sizeof(W34Tp2Signal)` query with no preconditions.
        let sig_bytes = unsafe { ffi::w34_tp2_signal_bytes() };
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
                        ffi::w34_device_can_enable_p2p(i as i32, j as i32, 1, &raw mut can_p2p)
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
            for _ in 0..tp_size {
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
        Ok(Self {
            tp_size,
            p2p_vram: all_p2p,
            sigs,
            stages,
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
        if self.tp_size == 4 {
            return check_rc(
                // SAFETY: staging capacity is asserted above; `stages[0..4]` and `sigs[0..4]` are live
                // buffers owned by `self`, and caller guarantees valid `local_in_ptr`/`out_ptr` for `num_elems` FP16s.
                unsafe {
                    ffi::w34_tp4_all_reduce_f16(
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
                "w34_tp4_all_reduce_f16",
            );
        }
        check_rc(
            // SAFETY: staging capacity is asserted above; `stages[0..2]` and `sigs[0..2]` are live
            // buffers owned by `self`, and caller guarantees valid `local_in_ptr`/`out_ptr` for `num_elems` FP16s.
            unsafe {
                ffi::w34_tp2_all_reduce_f16(
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
        ex: &W34Exec,
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
        let num_elems = (num_tokens as usize) * (hidden_size as usize);
        assert!(
            num_elems * 2 <= self.max_stage_bytes_per_epoch,
            "all_reduce_residual_gemma_rms_norm_f16 exceeds max_stage_bytes_per_epoch"
        );
        if self.tp_size == 4 {
            return check_rc(
                // SAFETY: staging capacity is asserted above; `stages[0..4]` and `sigs[0..4]` are live
                // buffers owned by `self`, and caller guarantees valid device pointers for `[num_tokens, hidden_size]`.
                unsafe {
                    ffi::w34_tp4_all_reduce_residual_gemma_rms_norm_hadamard_f16(
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
                "w34_tp4_all_reduce_residual_gemma_rms_norm_hadamard_f16",
            );
        }
        check_rc(
            // SAFETY: staging capacity is asserted above; `stages[0..2]` and `sigs[0..2]` are live
            // buffers owned by `self`, and caller guarantees valid device pointers for `[num_tokens, hidden_size]`.
            unsafe {
                ffi::w34_tp2_all_reduce_residual_gemma_rms_norm_hadamard_f16(
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
        if self.tp_size == 4 {
            return check_rc(
                // SAFETY: staging capacity is asserted above; `stages[0..4]` and `sigs[0..4]` are live
                // buffers owned by `self`, and caller guarantees valid BF16 device buffers for `num_elems`.
                unsafe {
                    ffi::w34_tp4_all_reduce_bf16(
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
                "w34_tp4_all_reduce_bf16",
            );
        }
        check_rc(
            // SAFETY: staging capacity is asserted above; `stages[0..2]` and `sigs[0..2]` are live
            // buffers owned by `self`, and caller guarantees valid BF16 device buffers for `num_elems`.
            unsafe {
                ffi::w34_tp2_all_reduce_bf16(
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
        if self.tp_size == 4 {
            return check_rc(
                // SAFETY: staging capacity is asserted above; `stages[0..4]` and `sigs[0..4]` are live
                // buffers owned by `self`, and caller guarantees valid FP32 device buffers for `num_elems`.
                unsafe {
                    ffi::w34_tp4_all_reduce_f32(
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
                "w34_tp4_all_reduce_f32",
            );
        }
        check_rc(
            // SAFETY: staging capacity is asserted above; `stages[0..2]` and `sigs[0..2]` are live
            // buffers owned by `self`, and caller guarantees valid FP32 device buffers for `num_elems`.
            unsafe {
                ffi::w34_tp2_all_reduce_f32(
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
        if self.tp_size == 4 {
            return check_rc(
                // SAFETY: staging capacity is asserted above; `stages[0..4]` and `sigs[0..4]` are live
                // buffers owned by `self`, and caller guarantees valid `i32` device buffers for `num_elems`.
                unsafe {
                    ffi::w34_tp4_broadcast_i32(
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
                "w34_tp4_broadcast_i32",
            );
        }
        check_rc(
            // SAFETY: staging capacity is asserted above; `stages[0..2]` and `sigs[0..2]` are live
            // buffers owned by `self`, and caller guarantees valid `i32` device buffers for `num_elems`.
            unsafe {
                ffi::w34_tp2_broadcast_i32(
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
            "w34_tp2_broadcast_i32",
        )
    }

    #[allow(clippy::too_many_arguments)]
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
        if self.tp_size == 4 {
            return check_rc(
                // SAFETY: staging capacity is asserted above; `stages[0..4]` and `sigs[0..4]` are live
                // buffers owned by `self`, and caller guarantees valid `(f32, i32)` row buffers for `num_rows`.
                unsafe {
                    ffi::w34_tp4_all_gather_top1_f32_i32(
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
                "w34_tp4_all_gather_top1_f32_i32",
            );
        }
        check_rc(
            // SAFETY: staging capacity is asserted above; `stages[0..2]` and `sigs[0..2]` are live
            // buffers owned by `self`, and caller guarantees valid `(f32, i32)` row buffers for `num_rows`.
            unsafe {
                ffi::w34_tp2_all_gather_top1_f32_i32(
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
            "w34_tp2_all_gather_top1_f32_i32",
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub fn all_gather_topk_f32_i32(
        &self,
        ex: &W34Exec,
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
                    ffi::w34_tp4_all_gather_topk_f32_i32(
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
                "w34_tp4_all_gather_topk_f32_i32",
            );
        }
        check_rc(
            // SAFETY: staging capacity is asserted above; `stages[0..2]` and `sigs[0..2]` are live
            // buffers owned by `self`, and caller guarantees valid `(f32, i32)` buffers for `num_rows * top_k`.
            unsafe {
                ffi::w34_tp2_all_gather_topk_f32_i32(
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
            "w34_tp2_all_gather_topk_f32_i32",
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub fn all_reduce_hc_combine_tap_f16(
        &self,
        ex: &W34Exec,
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
                    ffi::w34_tp4_all_reduce_hc_combine_tap_f16(
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
                "w34_tp4_all_reduce_hc_combine_tap_f16",
            );
        }
        check_rc(
            // SAFETY: staging capacity is asserted above; `stages[0..2]` and `sigs[0..2]` are live
            // buffers owned by `self`, and caller guarantees valid device buffers for `num_tokens`.
            unsafe {
                ffi::w34_tp2_all_reduce_hc_combine_tap_f16(
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
            "w34_tp2_all_reduce_hc_combine_tap_f16",
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
        if self.tp_size == 4 {
            return check_rc(
                // SAFETY: staging capacity is asserted above; `stages[0..4]` and `sigs[0..4]` are live
                // buffers owned by `self`, and caller guarantees valid FP16 shard/full buffers.
                unsafe {
                    ffi::w34_tp4_gather_vocab_f16(
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
                "w34_tp4_gather_vocab_f16",
            );
        }
        check_rc(
            // SAFETY: staging capacity is asserted above; `stages[0..2]` and `sigs[0..2]` are live
            // buffers owned by `self`, and caller guarantees valid FP16 shard/full buffers.
            unsafe {
                ffi::w34_tp2_gather_vocab_f16(
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
            "w34_tp2_gather_vocab_f16",
        )
    }
}
