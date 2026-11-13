//! Fixed-layout decode and speculative graph input/output buffers.

use super::*;

/// Static device buffers used by [`Qwen3_5ModelRank`] for `M=1` single-request decode.
#[cfg(feature = "cuda")]
pub struct DecodeGraphIo {
    pub token_ids: DeviceBuffer,
    pub positions: DeviceBuffer,
    pub q_start_loc: DeviceBuffer,
    pub ctx_lens_before: DeviceBuffer,
    pub seq_lens_after: DeviceBuffer,
    pub block_tables: DeviceBuffer,
    pub write_slot_mapping: DeviceBuffer,
    pub state_in_indices: DeviceBuffer,
    pub state_out_indices: DeviceBuffer,
    pub has_initial_state: DeviceBuffer,
    pub out_greedy_id: DeviceBuffer,
    pub out_top5_ids: DeviceBuffer,
    pub out_top5_logprobs: DeviceBuffer,
    pub local_max_val: DeviceBuffer,
    pub local_max_id: DeviceBuffer,
    pub global_max_val: DeviceBuffer,
    pub global_max_id: DeviceBuffer,
    pub token_lora_ids: DeviceBuffer,
    pub max_blocks_per_req: usize,
}

#[cfg(feature = "cuda")]
impl DecodeGraphIo {
    pub fn new(device_id: i32, max_blocks_per_req: usize, stream: &CudaStream) -> Result<Self> {
        let token_ids = DeviceBuffer::alloc(device_id, 4)?;
        let positions = DeviceBuffer::alloc(device_id, 4)?;
        let q_start_loc = DeviceBuffer::alloc(device_id, 8)?;
        q_start_loc.copy_from_host(&[0i32, 1], stream)?;
        let ctx_lens_before = DeviceBuffer::alloc(device_id, 4)?;
        let seq_lens_after = DeviceBuffer::alloc(device_id, 4)?;
        let block_tables = DeviceBuffer::alloc(device_id, max_blocks_per_req * 4)?;
        let write_slot_mapping = DeviceBuffer::alloc(device_id, 4)?;
        let state_in_indices = DeviceBuffer::alloc(device_id, 4)?;
        let state_out_indices = DeviceBuffer::alloc(device_id, 4)?;
        let has_initial_state = DeviceBuffer::alloc(device_id, 4)?;
        has_initial_state.copy_from_host(&[1u8, 0, 0, 0], stream)?;
        let out_greedy_id = DeviceBuffer::alloc(device_id, 4)?;
        let out_top5_ids = DeviceBuffer::alloc(device_id, 5 * 4)?;
        let out_top5_logprobs = DeviceBuffer::alloc(device_id, 5 * 4)?;
        let local_max_val = DeviceBuffer::alloc(device_id, 4)?;
        let local_max_id = DeviceBuffer::alloc(device_id, 4)?;
        let global_max_val = DeviceBuffer::alloc(device_id, 4)?;
        let global_max_id = DeviceBuffer::alloc(device_id, 4)?;
        let token_lora_ids = DeviceBuffer::alloc(device_id, 4)?;
        token_lora_ids.copy_from_host(&[crate::lora::BASE_MODEL_LORA_ID], stream)?;
        Ok(Self {
            token_ids,
            positions,
            q_start_loc,
            ctx_lens_before,
            seq_lens_after,
            block_tables,
            write_slot_mapping,
            state_in_indices,
            state_out_indices,
            has_initial_state,
            out_greedy_id,
            out_top5_ids,
            out_top5_logprobs,
            local_max_val,
            local_max_id,
            global_max_val,
            global_max_id,
            token_lora_ids,
            max_blocks_per_req,
        })
    }
}

pub const MAX_SPEC_VERIFY_TOKENS: usize = 16;
pub const MAX_SPEC_VERIFY_CHECKPOINTS: usize = 4;
const SPEC_META_I32_LEN: usize = 5 * MAX_SPEC_VERIFY_TOKENS + 2 * MAX_SPEC_VERIFY_CHECKPOINTS + 6;

/// Static device buffers used for fixed-shape `M_verify = K + 1` (`2..=16`) speculative
/// verification CUDA Graph capture and replay (`MTP` & `DFlash2`, `TP=1` & `TP=2`).
#[cfg(feature = "cuda")]
pub struct SpecVerifyGraphIo {
    pub meta_i32: DeviceBuffer,
    pub block_tables: DeviceBuffer,
    pub has_initial_state: DeviceBuffer,
    pub out_greedy_ids: DeviceBuffer,
    pub max_blocks_per_req: usize,
}

#[cfg(feature = "cuda")]
impl SpecVerifyGraphIo {
    pub fn new(device_id: i32, max_blocks_per_req: usize, stream: &CudaStream) -> Result<Self> {
        let meta_i32 = DeviceBuffer::alloc(device_id, SPEC_META_I32_LEN * 4)?;
        let block_tables = DeviceBuffer::alloc(device_id, max_blocks_per_req.max(1) * 4)?;
        let has_initial_state = DeviceBuffer::alloc(device_id, 4)?;
        has_initial_state.copy_from_host(&[1u8, 0, 0, 0], stream)?;
        let out_greedy_ids = DeviceBuffer::alloc(device_id, MAX_SPEC_VERIFY_TOKENS * 4)?;
        Ok(Self {
            meta_i32,
            block_tables,
            has_initial_state,
            out_greedy_ids,
            max_blocks_per_req: max_blocks_per_req.max(1),
        })
    }

    #[inline]
    pub fn token_ids_ptr(&self) -> u64 {
        self.meta_i32.addr()
    }
    #[inline]
    pub fn positions_ptr(&self) -> u64 {
        self.meta_i32.addr() + (MAX_SPEC_VERIFY_TOKENS * 4) as u64
    }
    #[inline]
    pub fn write_slot_mapping_ptr(&self) -> u64 {
        self.meta_i32.addr() + (2 * MAX_SPEC_VERIFY_TOKENS * 4) as u64
    }
    #[inline]
    pub fn spec_state_indices_ptr(&self) -> u64 {
        self.meta_i32.addr() + (3 * MAX_SPEC_VERIFY_TOKENS * 4) as u64
    }
    #[inline]
    pub fn checkpoint_dst_indices_ptr(&self) -> u64 {
        self.meta_i32.addr() + (4 * MAX_SPEC_VERIFY_TOKENS * 4) as u64
    }
    #[inline]
    pub fn checkpoint_token_offsets_ptr(&self) -> u64 {
        self.meta_i32.addr()
            + ((4 * MAX_SPEC_VERIFY_TOKENS + MAX_SPEC_VERIFY_CHECKPOINTS) * 4) as u64
    }
    #[inline]
    pub fn q_start_loc_ptr(&self) -> u64 {
        self.meta_i32.addr()
            + ((4 * MAX_SPEC_VERIFY_TOKENS + 2 * MAX_SPEC_VERIFY_CHECKPOINTS) * 4) as u64
    }
    #[inline]
    pub fn ctx_lens_before_ptr(&self) -> u64 {
        self.q_start_loc_ptr() + 8
    }
    #[inline]
    pub fn seq_lens_after_ptr(&self) -> u64 {
        self.q_start_loc_ptr() + 12
    }
    #[inline]
    pub fn state_in_indices_ptr(&self) -> u64 {
        self.q_start_loc_ptr() + 16
    }
    #[inline]
    pub fn state_out_indices_ptr(&self) -> u64 {
        self.q_start_loc_ptr() + 20
    }
    #[inline]
    pub fn token_lora_ids_ptr(&self) -> u64 {
        self.q_start_loc_ptr() + 24
    }

    pub fn pack_meta(slice: &EngineBatchSlice<'_>) -> Result<[i32; SPEC_META_I32_LEN]> {
        Self::pack_meta_with_lora(slice, crate::lora::BASE_MODEL_LORA_ID)
    }

    pub fn pack_meta_with_lora(
        slice: &EngineBatchSlice<'_>,
        lora_id: i32,
    ) -> Result<[i32; SPEC_META_I32_LEN]> {
        let q_len = slice.tokens.len();
        let seq_after = slice
            .ctx_len_before
            .checked_add(q_len)
            .ok_or_else(|| loader_err("spec_verify sequence length overflow"))?;
        if q_len == 0
            || q_len > MAX_SPEC_VERIFY_TOKENS
            || slice.spec_slots.len() > MAX_SPEC_VERIFY_TOKENS
            || slice.checkpoints.len() > MAX_SPEC_VERIFY_CHECKPOINTS
            || slice.block_table.len() < seq_after.div_ceil(16)
        {
            return Err(loader_err(
                "slice exceeds SpecVerifyGraphIo metadata capacity",
            ));
        }
        let mut m = [-1i32; SPEC_META_I32_LEN];
        m[..q_len].copy_from_slice(slice.tokens);
        for step in 0..q_len {
            let pos = slice.ctx_len_before + step;
            m[MAX_SPEC_VERIFY_TOKENS + step] = pos as i32;
            let page_id = slice.block_table[pos / 16];
            let page_off = (pos % 16) as i32;
            m[2 * MAX_SPEC_VERIFY_TOKENS + step] = page_id * 16 + page_off;
        }
        m[3 * MAX_SPEC_VERIFY_TOKENS..3 * MAX_SPEC_VERIFY_TOKENS + slice.spec_slots.len()]
            .copy_from_slice(slice.spec_slots);
        let cp_dst_base = 4 * MAX_SPEC_VERIFY_TOKENS;
        let cp_off_base = cp_dst_base + MAX_SPEC_VERIFY_CHECKPOINTS;
        for (e, &(trig_off, dst_slot)) in slice.checkpoints.iter().enumerate() {
            m[cp_dst_base + e] = dst_slot;
            m[cp_off_base + e] = trig_off;
        }
        let q_base = cp_off_base + MAX_SPEC_VERIFY_CHECKPOINTS;
        m[q_base] = 0;
        m[q_base + 1] = q_len as i32;
        m[q_base + 2] = slice.ctx_len_before as i32;
        m[q_base + 3] = seq_after as i32;
        m[q_base + 4] = slice.state_in_slot;
        m[q_base + 5] = slice.state_out_slot;
        let lora_base = q_base + 6;
        for step in 0..q_len {
            m[lora_base + step] = lora_id;
        }
        Ok(m)
    }
}

pub const MAX_BATCHED_DECODE_REQS: usize = 16;
pub const DECODE_GRAPH_BUCKETS: [usize; 5] = [1, 2, 4, 8, 16];
const BATCHED_DECODE_META_I32_LEN: usize = 11 * MAX_BATCHED_DECODE_REQS + 1;
const BATCHED_DECODE_MASKS_U8_LEN: usize = 3 * MAX_BATCHED_DECODE_REQS;

#[inline]
pub fn decode_graph_bucket(num_reqs: usize) -> Option<usize> {
    match num_reqs {
        1 => Some(1),
        2 => Some(2),
        3..=4 => Some(4),
        5..=8 => Some(8),
        9..=16 => Some(16),
        _ => None,
    }
}

/// Sampler mode captured inside a `batched_decode_graph`.
#[cfg(feature = "cuda")]
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum DecodeGraphSampleMode {
    LogitsOnly,
    PlainGreedy,
    FastGreedy,
}

#[cfg(feature = "cuda")]
pub struct InFlightDecodeTicket {
    pub slot: usize,
    pub num_reqs: usize,
    pub fast_greedy: bool,
}

#[cfg(feature = "cuda")]
pub enum PipelinedDecodeOutput {
    PlainGreedy(Vec<i32>),
    FastGreedy(Vec<ffi::W34SamplerRowOut>),
}

/// Static device buffers used for padded multi-request `B in {1, 2, 4, 8, 16}` decode
/// CUDA Graph capture and replay (`valid_req_mask` + `valid_token_mask` padding).
#[cfg(feature = "cuda")]
pub struct BatchedDecodeGraphIo {
    pub meta_i32: DeviceBuffer,
    pub block_tables: DeviceBuffer,
    pub out_greedy_ids: DeviceBuffer,
    pub fast_greedy_cfgs: DeviceBuffer,
    pub fast_greedy_sparse: DeviceBuffer,
    pub fast_greedy_outs: DeviceBuffer,
    pub pinned_staging: crate::cuda::MappedHostBuffer,
    pub pinned_in_flight: std::sync::atomic::AtomicBool,
    pub pinned_sampler_in_flight: std::sync::atomic::AtomicBool,
    pub cached_block_tables: std::sync::Mutex<Vec<Vec<i32>>>,
    pub cached_fast_greedy_cfgs: std::sync::Mutex<Vec<u8>>,
    pub cached_fast_greedy_sparse: std::sync::Mutex<Vec<u8>>,
    pub pipeline_done_events: [crate::cuda::CudaEvent; 2],
    pub max_blocks_per_req: usize,
}

#[cfg(feature = "cuda")]
impl BatchedDecodeGraphIo {
    pub const TOKEN_IDS_BYTES: usize = MAX_BATCHED_DECODE_REQS * 4;
    pub const MASKS_DEV_OFFSET: usize = (BATCHED_DECODE_META_I32_LEN * 4 + 63) & !63;
    pub const META_AND_MASKS_BYTES: usize = Self::MASKS_DEV_OFFSET + BATCHED_DECODE_MASKS_U8_LEN;
    pub const MAX_FAST_GREEDY_SPARSE: usize = MAX_BATCHED_DECODE_REQS * 32;
    pub const PINNED_META_BYTES: usize = 160 * 1024;
    pub const PINNED_SAMPLER_OFFSET: usize = 160 * 1024;
    pub const PINNED_SAMPLER_BYTES: usize = 96 * 1024;
    pub const PINNED_PIPELINE_OUT_OFFSET: usize = 224 * 1024;
    pub const PINNED_SLOT_BYTES: usize = Self::PINNED_META_BYTES + Self::PINNED_SAMPLER_BYTES;

    pub fn new(device_id: i32, max_blocks_per_req: usize, _stream: &CudaStream) -> Result<Self> {
        let max_blocks = max_blocks_per_req.max(1);
        let meta_i32 = DeviceBuffer::alloc(device_id, Self::META_AND_MASKS_BYTES)?;
        let block_tables =
            DeviceBuffer::alloc(device_id, MAX_BATCHED_DECODE_REQS * max_blocks * 4)?;
        let out_greedy_ids = DeviceBuffer::alloc(device_id, MAX_BATCHED_DECODE_REQS * 4)?;
        let fast_greedy_cfgs = DeviceBuffer::alloc(
            device_id,
            MAX_BATCHED_DECODE_REQS * core::mem::size_of::<ffi::W34SamplerRowCfg>(),
        )?;
        let fast_greedy_sparse = DeviceBuffer::alloc(
            device_id,
            Self::MAX_FAST_GREEDY_SPARSE * core::mem::size_of::<ffi::W34SamplerSparseEntry>(),
        )?;
        let fast_greedy_outs = DeviceBuffer::alloc(
            device_id,
            MAX_BATCHED_DECODE_REQS * core::mem::size_of::<ffi::W34SamplerRowOut>(),
        )?;
        let pinned_staging = crate::cuda::MappedHostBuffer::alloc(2 * Self::PINNED_SLOT_BYTES)?;
        let pipeline_done_events = [
            crate::cuda::CudaEvent::new(device_id, false)?,
            crate::cuda::CudaEvent::new(device_id, false)?,
        ];
        // SAFETY: allocation contains two PINNED_SLOT_BYTES regions; each metadata
        // prefix fits within its slot. The fresh allocation has no in-flight users.
        unsafe {
            for slot in 0..2 {
                core::ptr::write_bytes(
                    pinned_staging
                        .host_ptr
                        .cast::<u8>()
                        .add(slot * Self::PINNED_SLOT_BYTES),
                    0,
                    Self::META_AND_MASKS_BYTES,
                );
            }
        }
        Ok(Self {
            meta_i32,
            block_tables,
            out_greedy_ids,
            fast_greedy_cfgs,
            fast_greedy_sparse,
            fast_greedy_outs,
            pinned_staging,
            pinned_in_flight: std::sync::atomic::AtomicBool::new(false),
            pinned_sampler_in_flight: std::sync::atomic::AtomicBool::new(false),
            cached_block_tables: std::sync::Mutex::new(Vec::new()),
            cached_fast_greedy_cfgs: std::sync::Mutex::new(Vec::new()),
            cached_fast_greedy_sparse: std::sync::Mutex::new(Vec::new()),
            pipeline_done_events,
            max_blocks_per_req: max_blocks,
        })
    }

    #[inline]
    pub fn token_ids_ptr(&self) -> u64 {
        self.meta_i32.addr()
    }
    #[inline]
    pub fn positions_ptr(&self) -> u64 {
        self.meta_i32.addr() + (MAX_BATCHED_DECODE_REQS * 4) as u64
    }
    #[inline]
    pub fn write_slot_mapping_ptr(&self) -> u64 {
        self.meta_i32.addr() + (2 * MAX_BATCHED_DECODE_REQS * 4) as u64
    }
    #[inline]
    pub fn q_start_loc_ptr(&self) -> u64 {
        self.meta_i32.addr() + (3 * MAX_BATCHED_DECODE_REQS * 4) as u64
    }
    #[inline]
    pub fn ctx_lens_before_ptr(&self) -> u64 {
        self.meta_i32.addr()
            + ((3 * MAX_BATCHED_DECODE_REQS + MAX_BATCHED_DECODE_REQS + 1) * 4) as u64
    }
    #[inline]
    pub fn seq_lens_after_ptr(&self) -> u64 {
        self.ctx_lens_before_ptr() + (MAX_BATCHED_DECODE_REQS * 4) as u64
    }
    #[inline]
    pub fn state_in_indices_ptr(&self) -> u64 {
        self.seq_lens_after_ptr() + (MAX_BATCHED_DECODE_REQS * 4) as u64
    }
    #[inline]
    pub fn state_out_indices_ptr(&self) -> u64 {
        self.state_in_indices_ptr() + (MAX_BATCHED_DECODE_REQS * 4) as u64
    }
    #[inline]
    pub fn checkpoint_dst_indices_ptr(&self) -> u64 {
        self.state_out_indices_ptr() + (MAX_BATCHED_DECODE_REQS * 4) as u64
    }
    #[inline]
    pub fn checkpoint_token_offsets_ptr(&self) -> u64 {
        self.checkpoint_dst_indices_ptr() + (MAX_BATCHED_DECODE_REQS * 4) as u64
    }
    #[inline]
    pub fn token_lora_ids_ptr(&self) -> u64 {
        self.checkpoint_token_offsets_ptr() + (MAX_BATCHED_DECODE_REQS * 4) as u64
    }
    #[inline]
    pub fn has_initial_state_ptr(&self) -> u64 {
        self.meta_i32.addr() + Self::MASKS_DEV_OFFSET as u64
    }
    #[inline]
    pub fn valid_req_mask_ptr(&self) -> u64 {
        self.has_initial_state_ptr() + MAX_BATCHED_DECODE_REQS as u64
    }
    #[inline]
    pub fn valid_token_mask_ptr(&self) -> u64 {
        self.has_initial_state_ptr() + (2 * MAX_BATCHED_DECODE_REQS) as u64
    }

    pub fn pack_meta(
        slices: &[EngineBatchSlice<'_>],
        rope_pos_deltas: Option<&[i32]>,
    ) -> Result<(
        [i32; BATCHED_DECODE_META_I32_LEN],
        [u8; BATCHED_DECODE_MASKS_U8_LEN],
    )> {
        Self::pack_meta_with_lora(
            slices,
            rope_pos_deltas,
            None,
            crate::lora::BASE_MODEL_LORA_ID,
        )
    }

    pub fn pack_meta_with_lora(
        slices: &[EngineBatchSlice<'_>],
        rope_pos_deltas: Option<&[i32]>,
        slice_lora_ids: Option<&[i32]>,
        default_lora_id: i32,
    ) -> Result<(
        [i32; BATCHED_DECODE_META_I32_LEN],
        [u8; BATCHED_DECODE_MASKS_U8_LEN],
    )> {
        let n = slices.len();
        if n == 0 || n > MAX_BATCHED_DECODE_REQS {
            return Err(loader_err("invalid slice count for BatchedDecodeGraphIo"));
        }
        let mut m = [0i32; BATCHED_DECODE_META_I32_LEN];
        let mut masks = [0u8; BATCHED_DECODE_MASKS_U8_LEN];
        let ws_base = 2 * MAX_BATCHED_DECODE_REQS;
        let qs_base = 3 * MAX_BATCHED_DECODE_REQS;
        let cb_base = qs_base + MAX_BATCHED_DECODE_REQS + 1;
        let sa_base = cb_base + MAX_BATCHED_DECODE_REQS;
        let si_base = sa_base + MAX_BATCHED_DECODE_REQS;
        let so_base = si_base + MAX_BATCHED_DECODE_REQS;
        let cd_base = so_base + MAX_BATCHED_DECODE_REQS;
        let co_base = cd_base + MAX_BATCHED_DECODE_REQS;
        let lora_base = co_base + MAX_BATCHED_DECODE_REQS;

        for i in 0..=MAX_BATCHED_DECODE_REQS {
            m[qs_base + i] = i as i32;
        }
        for r in 0..MAX_BATCHED_DECODE_REQS {
            m[ws_base + r] = -1;
            m[so_base + r] = -1;
            m[cd_base + r] = -1;
            m[co_base + r] = -1;
            m[lora_base + r] = crate::lora::BASE_MODEL_LORA_ID;
        }
        for (r, s) in slices.iter().enumerate() {
            if s.tokens.len() != 1
                || !s.spec_slots.is_empty()
                || s.checkpoints.len() > 1
                || s.block_table.len() < (s.ctx_len_before + 1).div_ceil(16)
            {
                return Err(loader_err(
                    "slice incompatible with 1-token BatchedDecodeGraphIo",
                ));
            }
            let pos = s.ctx_len_before;
            let delta = rope_pos_deltas.and_then(|d| d.get(r).copied()).unwrap_or(0);
            let page_id = s.block_table[pos / 16];
            let page_off = (pos % 16) as i32;
            m[r] = s.tokens[0];
            m[MAX_BATCHED_DECODE_REQS + r] = (pos as i32) + delta;
            m[ws_base + r] = page_id * 16 + page_off;
            m[cb_base + r] = pos as i32;
            m[sa_base + r] = (pos + 1) as i32;
            m[si_base + r] = s.state_in_slot;
            m[so_base + r] = s.state_out_slot;
            if let Some(&(trig_off, dst_slot)) = s.checkpoints.first() {
                m[cd_base + r] = dst_slot;
                m[co_base + r] = trig_off;
            }
            m[lora_base + r] = slice_lora_ids
                .and_then(|ids| ids.get(r).copied())
                .unwrap_or(default_lora_id);
            masks[r] = u8::from(s.has_initial_state);
            masks[MAX_BATCHED_DECODE_REQS + r] = 1;
            masks[2 * MAX_BATCHED_DECODE_REQS + r] = 1;
        }
        Ok((m, masks))
    }

    pub fn upload_inputs(
        &self,
        meta: &[i32; BATCHED_DECODE_META_I32_LEN],
        masks: &[u8; BATCHED_DECODE_MASKS_U8_LEN],
        slices: &[EngineBatchSlice<'_>],
        stream: &CudaStream,
    ) -> Result<()> {
        self.upload_inputs_slot(0, meta, masks, slices, false, false, stream)
    }

    #[allow(clippy::too_many_arguments)]
    pub fn upload_inputs_slot(
        &self,
        slot: usize,
        meta: &[i32; BATCHED_DECODE_META_I32_LEN],
        masks: &[u8; BATCHED_DECODE_MASKS_U8_LEN],
        slices: &[EngineBatchSlice<'_>],
        tokens_already_on_device: bool,
        pipelined: bool,
        stream: &CudaStream,
    ) -> Result<()> {
        let slot = slot & 1;
        let meta_bytes = BATCHED_DECODE_META_I32_LEN * 4;
        if Self::META_AND_MASKS_BYTES > self.meta_i32.len_bytes {
            return Err(loader_err("batched decode meta_i32 buffer too small"));
        }
        let masks_off = Self::MASKS_DEV_OFFSET;
        let bt_off = (Self::META_AND_MASKS_BYTES + 63) & !63;
        let row_bytes = self.max_blocks_per_req * 4;
        let mut last_bt_end = 0usize;
        for (r, s) in slices.iter().enumerate() {
            if s.block_table.len() > self.max_blocks_per_req {
                return Err(loader_err(
                    "batched decode row block_table exceeds max_blocks_per_req",
                ));
            }
            if !s.block_table.is_empty() {
                last_bt_end = r * row_bytes + s.block_table.len() * 4;
            }
        }
        if last_bt_end > self.block_tables.len_bytes {
            return Err(loader_err("batched decode block_tables buffer too small"));
        }
        let bt_unchanged = {
            let guard = self.cached_block_tables.lock().unwrap();
            guard.len() == slices.len()
                && guard
                    .iter()
                    .zip(slices)
                    .all(|(c, s)| c.as_slice() == s.block_table)
        };
        let slot_end = (slot + 1) * Self::PINNED_SLOT_BYTES;
        if bt_off + last_bt_end <= Self::PINNED_META_BYTES
            && slot_end <= self.pinned_staging.len_bytes
        {
            let _ = pipelined;
            if slot == 0
                && self
                    .pinned_in_flight
                    .swap(false, std::sync::atomic::Ordering::AcqRel)
            {
                stream.synchronize()?;
            }
            // SAFETY: slot is masked to 0/1 and slot_end is bounded by the owned
            // pinned allocation above. This computes a pointer, not a host borrow.
            let base = unsafe {
                self.pinned_staging
                    .host_ptr
                    .cast::<u8>()
                    .add(slot * Self::PINNED_SLOT_BYTES)
            };
            let h2d_off = if tokens_already_on_device {
                Self::TOKEN_IDS_BYTES
            } else {
                0
            };
            // SAFETY: the branch bounds metadata/masks and strided block-table
            // bytes within this slot and the destination device buffers. Host
            // sources are separate slices. Slot 0's previous upload is waited
            // above; slot 1 reuse relies on the pipeline caller waiting its ticket.
            // The &self API/flags alone do not serialize concurrent callers.
            unsafe {
                core::ptr::copy_nonoverlapping(
                    meta.as_ptr().cast::<u8>().add(h2d_off),
                    base.add(h2d_off),
                    meta_bytes - h2d_off,
                );
                core::ptr::copy_nonoverlapping(
                    masks.as_ptr(),
                    base.add(masks_off),
                    BATCHED_DECODE_MASKS_U8_LEN,
                );
                if !bt_unchanged {
                    for (r, s) in slices.iter().enumerate() {
                        if !s.block_table.is_empty() {
                            core::ptr::copy_nonoverlapping(
                                s.block_table.as_ptr().cast::<u8>(),
                                base.add(bt_off + r * row_bytes),
                                s.block_table.len() * 4,
                            );
                        }
                    }
                }
                if slot == 0 {
                    self.pinned_in_flight
                        .store(true, std::sync::atomic::Ordering::Release);
                }
                check_rc(
                    ffi::w34_memcpy_h2d_async(
                        (self.meta_i32.addr() + h2d_off as u64) as *mut c_void,
                        base.add(h2d_off).cast(),
                        Self::META_AND_MASKS_BYTES - h2d_off,
                        stream.handle,
                    ),
                    "upload pinned batched decode meta+masks",
                )?;
                if !bt_unchanged && last_bt_end > 0 {
                    check_rc(
                        ffi::w34_memcpy_h2d_async(
                            self.block_tables.addr() as *mut c_void,
                            base.add(bt_off).cast(),
                            last_bt_end,
                            stream.handle,
                        ),
                        "upload pinned batched decode block_tables",
                    )?;
                }
            }
            if !bt_unchanged {
                let mut guard = self.cached_block_tables.lock().unwrap();
                guard.clear();
                guard.extend(slices.iter().map(|s| s.block_table.to_vec()));
            }
            return Ok(());
        }
        let mut combined = [0u8; Self::META_AND_MASKS_BYTES];
        // SAFETY: fixed-size input arrays and combined's constant layout provide
        // disjoint, in-bounds byte ranges; every i32 byte is initialized.
        unsafe {
            core::ptr::copy_nonoverlapping(
                meta.as_ptr().cast::<u8>(),
                combined.as_mut_ptr(),
                meta_bytes,
            );
            core::ptr::copy_nonoverlapping(
                masks.as_ptr(),
                combined.as_mut_ptr().add(masks_off),
                BATCHED_DECODE_MASKS_U8_LEN,
            );
        }
        let h2d_off = if tokens_already_on_device {
            Self::TOKEN_IDS_BYTES
        } else {
            0
        };
        check_rc(
            // SAFETY: the device metadata extent was checked above and h2d_off
            // skips only token IDs. This fallback uses pageable stack storage,
            // relying on CUDA's host staging before that storage goes out of scope;
            // it is not an asynchronously retained pinned source.
            unsafe {
                ffi::w34_memcpy_h2d_async(
                    (self.meta_i32.addr() + h2d_off as u64) as *mut c_void,
                    combined.as_ptr().add(h2d_off).cast(),
                    Self::META_AND_MASKS_BYTES - h2d_off,
                    stream.handle,
                )
            },
            "upload batched decode meta+masks fallback",
        )?;
        if !bt_unchanged {
            let bt_base = self.block_tables.addr();
            for (r, s) in slices.iter().enumerate() {
                if !s.block_table.is_empty() {
                    check_rc(
                        // SAFETY: row and total table byte extents were checked
                        // above. Engine callers supply ordinary pageable host
                        // slices here, relying on CUDA host staging; this public
                        // method does not itself rule out registered/pinned sources.
                        unsafe {
                            ffi::w34_memcpy_h2d_async(
                                (bt_base + (r * row_bytes) as u64) as *mut c_void,
                                s.block_table.as_ptr().cast(),
                                s.block_table.len() * 4,
                                stream.handle,
                            )
                        },
                        "upload batched decode block_table row",
                    )?;
                }
            }
            let mut guard = self.cached_block_tables.lock().unwrap();
            guard.clear();
            guard.extend(slices.iter().map(|s| s.block_table.to_vec()));
        }
        Ok(())
    }

    pub fn upload_fast_greedy_slot(
        &self,
        slot: usize,
        row_cfgs: &[ffi::W34SamplerRowCfg],
        sparse_entries: &[ffi::W34SamplerSparseEntry],
        bucket_b: usize,
        stream: &CudaStream,
    ) -> Result<()> {
        if bucket_b > MAX_BATCHED_DECODE_REQS || row_cfgs.len() > bucket_b {
            return Err(loader_err(
                "fast_greedy row_cfgs exceed decode graph bucket",
            ));
        }
        if sparse_entries.len() > Self::MAX_FAST_GREEDY_SPARSE {
            return Err(loader_err(
                "fast_greedy sparse_entries exceed static graph buffer",
            ));
        }
        let pad_cfg = ffi::W34SamplerRowCfg {
            temperature: 0.0,
            top_k: 1,
            top_p: 1.0,
            min_p: 0.0,
            repetition_penalty: 1.0,
            frequency_penalty: 0.0,
            presence_penalty: 0.0,
            uniform_sample: 0.0,
            recovery_uniform: 0.0,
            draft_token_id: -1,
            has_bitmask: 0,
            row_bitmask_word_offset: -1,
            sparse_offset: 0,
            sparse_count: 0,
            top_logprobs_k: -1,
            use_token_id_cdf: 0,
        };
        let mut padded_cfgs = [pad_cfg; MAX_BATCHED_DECODE_REQS];
        padded_cfgs[..row_cfgs.len()].copy_from_slice(row_cfgs);
        let cfg_bytes = bucket_b * core::mem::size_of::<ffi::W34SamplerRowCfg>();
        // SAFETY: W34SamplerRowCfg is repr(C), sixteen initialized 4-byte
        // scalars without padding; bucket_b is bounded by padded_cfgs.len().
        let cfg_raw =
            unsafe { core::slice::from_raw_parts(padded_cfgs.as_ptr().cast::<u8>(), cfg_bytes) };
        let sparse_bytes = core::mem::size_of_val(sparse_entries);
        // SAFETY: W34SamplerSparseEntry is repr(C), with 4+4+4+2+2 initialized
        // bytes and no padding. The byte view covers exactly the borrowed slice.
        let sparse_raw = unsafe {
            core::slice::from_raw_parts(sparse_entries.as_ptr().cast::<u8>(), sparse_bytes)
        };

        let cfg_unchanged = {
            let guard = self.cached_fast_greedy_cfgs.lock().unwrap();
            guard.as_slice() == cfg_raw
        };
        let sparse_unchanged = {
            let guard = self.cached_fast_greedy_sparse.lock().unwrap();
            guard.as_slice() == sparse_raw
        };
        if cfg_unchanged && sparse_unchanged {
            return Ok(());
        }

        let slot = slot & 1;
        if slot == 0
            && self
                .pinned_sampler_in_flight
                .swap(false, std::sync::atomic::Ordering::AcqRel)
        {
            stream.synchronize()?;
        }
        // SAFETY: new() reserves two slots, each including PINNED_SAMPLER_BYTES;
        // slot is 0/1 and this offset addresses that slot's sampler region.
        let pbase = unsafe {
            self.pinned_staging
                .host_ptr
                .cast::<u8>()
                .add(slot * Self::PINNED_SLOT_BYTES + Self::PINNED_SAMPLER_OFFSET)
        };
        let sparse_off = (cfg_bytes + 63) & !63;
        // SAFETY: bounded cfg/sparse counts fit the slot's sampler region and
        // static device buffers; the host slices do not overlap pinned storage.
        // Slot 0 waits its previous upload above; slot 1 and concurrent access
        // require caller coordination. Flags mark queued work, not a host lock.
        unsafe {
            let mut queued_h2d = false;
            if !cfg_unchanged {
                core::ptr::copy_nonoverlapping(cfg_raw.as_ptr(), pbase, cfg_bytes);
                if slot == 0 {
                    self.pinned_sampler_in_flight
                        .store(true, std::sync::atomic::Ordering::Release);
                    queued_h2d = true;
                }
                check_rc(
                    ffi::w34_memcpy_h2d_async(
                        self.fast_greedy_cfgs.addr() as *mut c_void,
                        pbase.cast(),
                        cfg_bytes,
                        stream.handle,
                    ),
                    "upload fast_greedy_cfgs",
                )?;
            }
            if !sparse_unchanged && sparse_bytes > 0 {
                core::ptr::copy_nonoverlapping(
                    sparse_raw.as_ptr(),
                    pbase.add(sparse_off),
                    sparse_bytes,
                );
                if slot == 0 && !queued_h2d {
                    self.pinned_sampler_in_flight
                        .store(true, std::sync::atomic::Ordering::Release);
                }
                check_rc(
                    ffi::w34_memcpy_h2d_async(
                        self.fast_greedy_sparse.addr() as *mut c_void,
                        pbase.add(sparse_off).cast(),
                        sparse_bytes,
                        stream.handle,
                    ),
                    "upload fast_greedy_sparse",
                )?;
            }
        }
        if !cfg_unchanged {
            let mut guard = self.cached_fast_greedy_cfgs.lock().unwrap();
            guard.clear();
            guard.extend_from_slice(cfg_raw);
        }
        if !sparse_unchanged {
            let mut guard = self.cached_fast_greedy_sparse.lock().unwrap();
            guard.clear();
            guard.extend_from_slice(sparse_raw);
        }
        Ok(())
    }
}

#[cfg(all(test, feature = "cuda"))]
mod tests {
    use super::*;

    fn slice(tokens: &[i32]) -> EngineBatchSlice<'_> {
        EngineBatchSlice {
            tokens,
            ctx_len_before: 15,
            block_table: &[7, 3],
            state_in_slot: 2,
            state_out_slot: 5,
            has_initial_state: true,
            spec_slots: &[8, 9],
            checkpoints: &[(1, 6)],
            emit_all_query_tokens: true,
        }
    }

    #[test]
    fn spec_metadata_preserves_page_crossing_and_state_transaction() {
        let meta = SpecVerifyGraphIo::pack_meta(&slice(&[41, 42])).unwrap();
        let n = MAX_SPEC_VERIFY_TOKENS;
        let cp = MAX_SPEC_VERIFY_CHECKPOINTS;
        assert_eq!(&meta[..2], &[41, 42]);
        assert_eq!(&meta[n..n + 2], &[15, 16]);
        assert_eq!(&meta[2 * n..2 * n + 2], &[7 * 16 + 15, 3 * 16]);
        assert_eq!(&meta[3 * n..3 * n + 2], &[8, 9]);
        assert_eq!(meta[4 * n], 6);
        assert_eq!(meta[4 * n + cp], 1);
        assert_eq!(
            &meta[4 * n + 2 * cp..4 * n + 2 * cp + 6],
            &[0, 2, 15, 17, 2, 5]
        );
        assert_eq!(&meta[4 * n + 2 * cp + 6..], &[-1; MAX_SPEC_VERIFY_TOKENS]);
        let meta_lora = SpecVerifyGraphIo::pack_meta_with_lora(&slice(&[41, 42]), 3).unwrap();
        assert_eq!(&meta_lora[4 * n + 2 * cp + 6..4 * n + 2 * cp + 8], &[3, 3]);
        assert!(
            meta_lora[4 * n + 2 * cp + 8..]
                .iter()
                .all(|&v| v == crate::lora::BASE_MODEL_LORA_ID)
        );
        for base in [0, n, 2 * n, 3 * n] {
            assert!(meta[base + 2..base + n].iter().all(|&v| v == -1));
        }
        assert!(meta[4 * n + 1..4 * n + cp].iter().all(|&v| v == -1));
    }

    #[test]
    fn spec_metadata_rejects_overflow_and_insufficient_tables() {
        let tokens = [1; MAX_SPEC_VERIFY_TOKENS + 1];
        assert!(SpecVerifyGraphIo::pack_meta(&slice(&tokens)).is_err());
        assert!(SpecVerifyGraphIo::pack_meta(&slice(&[])).is_err());
        let mut s = slice(&[1, 2]);
        s.block_table = &[7];
        assert!(SpecVerifyGraphIo::pack_meta(&s).is_err());
        s.ctx_len_before = usize::MAX;
        assert!(SpecVerifyGraphIo::pack_meta(&s).is_err());
    }

    #[test]
    fn pinned_slots_have_room_for_all_pipeline_and_sampler_records() {
        type Io = BatchedDecodeGraphIo;
        let cfg = MAX_BATCHED_DECODE_REQS * core::mem::size_of::<ffi::W34SamplerRowCfg>();
        let sparse_end = ((cfg + 63) & !63)
            + Io::MAX_FAST_GREEDY_SPARSE * core::mem::size_of::<ffi::W34SamplerSparseEntry>();
        let out_bytes = MAX_BATCHED_DECODE_REQS * core::mem::size_of::<ffi::W34SamplerRowOut>();
        const { assert!(Io::META_AND_MASKS_BYTES <= Io::PINNED_META_BYTES) };
        assert!(Io::PINNED_SAMPLER_OFFSET + sparse_end <= Io::PINNED_PIPELINE_OUT_OFFSET);
        assert!(Io::PINNED_PIPELINE_OUT_OFFSET + out_bytes <= Io::PINNED_SLOT_BYTES);
        assert_eq!(
            Io::PINNED_PIPELINE_OUT_OFFSET % core::mem::align_of::<ffi::W34SamplerRowOut>(),
            0
        );
    }
}
