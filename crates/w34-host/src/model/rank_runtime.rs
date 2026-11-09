//! Post-load runtime configuration and batch staging for [`Qwen3_5ModelRank`]
//! (tiered MoE initialization, PLE staging, speculative tap configuration, and
//! prefill token capacity queries).

use super::rank::{MIN_PLE_ROW_STAGING_BYTES, PleRankRuntime, select_tiered_moe_stage_capacity};
use super::*;

#[cfg(feature = "cuda")]
pub(super) fn alloc_ple_runtime(
    device_id: i32,
    cfg: &ModelConfig,
    ple_hasher: Arc<crate::ple::PleNgramHasher>,
    ple_mmap_table: Arc<crate::ple::PleMmapTable>,
) -> Result<PleRankRuntime> {
    // Pre-allocate batch rows * ple_embed_dim F8 staging buffer at a fixed device address
    // so both eager forward and CUDA Graphs can read from `f8_staging_dev.addr()`.
    let max_ple_tokens = if cfg.uses_tiered_moe() {
        crate::moe_offload::TIERED_MOE_MAX_BATCH_TOKENS
    } else {
        crate::scheduler::DEFAULT_MAX_NUM_BATCHED_TOKENS
    };
    let staged_row_bytes = ple_hasher
        .ngram_heads
        .saturating_mul(ple_mmap_table.staged_bytes_per_head());
    let ple_row_bytes = cfg.ple_embed_dim.max(cfg.hidden_size).max(staged_row_bytes);
    let staging_cap = max_ple_tokens * ple_row_bytes.max(MIN_PLE_ROW_STAGING_BYTES);
    let f8_staging_dev = DeviceBuffer::alloc(device_id, staging_cap)?;
    let f8_host_scratch = crate::cuda::MappedHostBuffer::alloc(2 * staging_cap)?;
    Ok(PleRankRuntime {
        hasher: ple_hasher,
        mmap_table: ple_mmap_table,
        f8_staging_dev,
        f8_host_scratch,
        f8_host_slot: 0,
        ids_scratch: Vec::new(),
    })
}

#[cfg(feature = "cuda")]
impl Qwen3_5ModelRank {
    /// Returns whether all layers on this rank can execute `num_tokens` inside a captured CUDA Graph.
    pub fn supports_cuda_graph_for_tokens(&self, num_tokens: usize) -> bool {
        let top_k = self.cfg.num_experts_per_tok.max(1);
        self.tiered_moe
            .as_ref()
            .is_none_or(|tm| tm.supports_cuda_graph_for_tokens(num_tokens, top_k))
    }

    /// Initializes expert residency independently of residual topology and PLE.
    pub fn init_tiered_moe(&mut self, archive: &SafetensorsArchive) -> Result<()> {
        self.device.make_current()?;
        let local_experts = (self.cfg.num_experts / self.tp_size).max(1);
        let default_hot = default_tiered_moe_hot_ceiling(self.tp_size, local_experts, usize::MAX);
        let hot_per_layer = if self.tiered_moe_hot_per_layer > 0 {
            self.tiered_moe_hot_per_layer
        } else if self.cfg.num_experts < TIERED_MOE_MIN_MODEL_EXPERTS {
            (local_experts / 2).max(1)
        } else {
            default_hot
        };
        let stage_cap = select_tiered_moe_stage_capacity(&self.cfg, self.tp_size, hot_per_layer);
        let tiered_moe = crate::moe_offload::RankTieredMoe::load_for_rank(
            archive,
            &self.cfg,
            self.device.device_id,
            self.rank,
            self.tp_size,
            hot_per_layer,
            stage_cap,
            &self.stream,
        )?;
        self.tiered_moe = Some(tiered_moe);
        Ok(())
    }

    pub(super) fn mark_tiered_moe_device_lru_dirty(&mut self) {
        if let Some(tm) = &mut self.tiered_moe {
            for (store, c) in tm.stores.iter().zip(&mut tm.caches) {
                if c.slots.capacity < c.expert_ids.len()
                    && (!c.has_cpu_bridge()
                        || (c.k_static < c.slots.capacity && store.mapped_dev_ptr() != 0))
                {
                    c.device_lru_dirty = true;
                }
            }
        }
    }

    /// Initializes shared PLE tables and rank-local staging independently of MoE.
    pub fn init_ple(
        &mut self,
        ple_hasher: Arc<crate::ple::PleNgramHasher>,
        ple_mmap_table: Arc<crate::ple::PleMmapTable>,
    ) -> Result<()> {
        self.device.make_current()?;
        self.ple_runtime = Some(alloc_ple_runtime(
            self.device.device_id,
            &self.cfg,
            ple_hasher,
            ple_mmap_table,
        )?);
        Ok(())
    }

    /// Computes PLE n-gram hashes for a batch, updates `state_pool.ple_ngram_context` across
    /// `state_in`/`state_out`/`spec_state_indices`/`checkpoint_dst_indices`, gathers F8_E4M3 rows
    /// from `PleMmapTable`, and uploads them asynchronously into `ple_runtime.f8_staging_dev`.
    #[allow(clippy::too_many_arguments)]
    pub fn stage_ple_f8_for_batch(
        &mut self,
        input_ids: &[i32],
        query_start_loc: &[i32],
        state_in_indices: &[i32],
        state_out_indices: &[i32],
        has_initial_state: &[u8],
        spec_state_indices: &[i32],
        max_verify_tokens: usize,
        checkpoint_dst_indices: &[i32],
        checkpoint_token_offsets: &[i32],
        max_checkpoints_per_req: usize,
    ) -> Result<()> {
        let Some(ple) = &mut self.ple_runtime else {
            return Ok(());
        };
        if input_ids.is_empty() {
            return Ok(());
        }
        let num_tokens = input_ids.len();
        let num_heads = ple.hasher.ngram_heads;
        let ple_row_bytes = num_heads * ple.mmap_table.staged_bytes_per_head();
        if num_tokens * ple_row_bytes > ple.f8_staging_dev.len_bytes {
            return Err(loader_err(
                "PLE batch token count exceeds staging buffer capacity",
            ));
        }
        ple.ids_scratch.resize(num_tokens * num_heads, 0);
        ple.hasher.compute_ngram_ids_for_slots(
            input_ids,
            query_start_loc,
            state_in_indices,
            state_out_indices,
            has_initial_state,
            spec_state_indices,
            max_verify_tokens,
            checkpoint_dst_indices,
            checkpoint_token_offsets,
            max_checkpoints_per_req,
            &mut self.state_pool.ple_ngram_context,
            &mut ple.ids_scratch,
        );
        let byte_len = num_tokens * ple_row_bytes;
        let slot_cap = ple.f8_staging_dev.len_bytes;
        let slot_offset = (ple.f8_host_slot & 1) * slot_cap;
        ple.f8_host_slot = ple.f8_host_slot.wrapping_add(1);
        // SAFETY: `f8_host_scratch` owns `2 * slot_cap` pinned bytes and `byte_len <= slot_cap`.
        let host_slice = unsafe {
            core::slice::from_raw_parts_mut(
                ple.f8_host_scratch.host_ptr.cast::<u8>().add(slot_offset),
                byte_len,
            )
        };
        ple.mmap_table.gather_f8(&ple.ids_scratch, host_slice);
        self.device.make_current()?;
        ple.f8_staging_dev.copy_from_host(host_slice, &self.stream)
    }

    pub(super) fn stage_ple_f8_for_decode_slices(
        &mut self,
        slices: &[EngineBatchSlice<'_>],
    ) -> Result<()> {
        if self.ple_runtime.is_none() || slices.is_empty() {
            return Ok(());
        }
        let n = slices.len();
        let mut tokens = Vec::with_capacity(n);
        let mut q_start = Vec::with_capacity(n + 1);
        q_start.push(0i32);
        let mut st_in = Vec::with_capacity(n);
        let mut st_out = Vec::with_capacity(n);
        let mut has_init = Vec::with_capacity(n);
        let mut ckpt_dst = vec![-1i32; n];
        let mut ckpt_off = vec![-1i32; n];
        for (i, s) in slices.iter().enumerate() {
            tokens.extend_from_slice(s.tokens);
            q_start.push(tokens.len() as i32);
            st_in.push(s.state_in_slot);
            st_out.push(s.state_out_slot);
            has_init.push(u8::from(s.has_initial_state));
            if let Some(&(trig_off, dst_slot)) = s.checkpoints.first() {
                ckpt_off[i] = trig_off;
                ckpt_dst[i] = dst_slot;
            }
        }
        self.stage_ple_f8_for_batch(
            &tokens,
            &q_start,
            &st_in,
            &st_out,
            &has_init,
            &[],
            0,
            &ckpt_dst,
            &ckpt_off,
            1,
        )
    }

    /// Configures replicated `TapPlan` auxiliary hidden-state capture for TP
    /// DFlash2 and/or post-final-norm hidden export for native MTP. Each target
    /// rank already owns the all-reduced activation; local capture avoids a
    /// second PCIe transfer before the replicated drafter context projection.
    /// Call on every TP rank with identical arguments before any forward or
    /// graph capture, enforcing the same row capacity. Engine callers must also
    /// propagate configuration failures before dispatching any rank forward.
    pub fn configure_speculative_taps(
        &mut self,
        tap_layer_ids: &[usize],
        export_final_hidden: bool,
        max_tap_rows: usize,
    ) -> Result<()> {
        // Captured graphs own these destination addresses and the tap geometry.
        // Require configuration before capture rather than freeing live targets.
        if self.decode_graph.is_some()
            || self.spec_verify_graph.is_some()
            || !self.batched_decode_graphs.is_empty()
        {
            return Err(loader_err(
                "configure speculative taps before decode graph capture",
            ));
        }
        if (export_final_hidden || !tap_layer_ids.is_empty()) && max_tap_rows == 0 {
            return Err(loader_err("speculative tap capacity must be positive"));
        }
        if tap_layer_ids
            .iter()
            .enumerate()
            .any(|(i, &id)| id >= self.layers.len() || tap_layer_ids[..i].contains(&id))
        {
            return Err(loader_err("tap layer ids must be unique and in range"));
        }
        self.device.make_current()?;
        self.stream.synchronize()?;
        let row_elems = max_tap_rows
            .checked_mul(self.cfg.hidden_size)
            .ok_or_else(|| loader_err("tap capacity overflow"))?;
        let tap_aux_bf16 = if !tap_layer_ids.is_empty() {
            let bytes = row_elems
                .checked_mul(tap_layer_ids.len())
                .and_then(|n| n.checked_mul(2))
                .ok_or_else(|| loader_err("tap capacity overflow"))?;
            Some(DeviceBuffer::alloc(self.device.device_id, bytes)?)
        } else {
            None
        };
        let last_final_hidden_f16 = if export_final_hidden {
            let hc_mult = self.cfg.hc_count.max(1);
            let bytes = row_elems
                .checked_mul(hc_mult)
                .and_then(|n| n.checked_mul(2))
                .ok_or_else(|| loader_err("final hidden capacity overflow"))?;
            Some(DeviceBuffer::alloc(self.device.device_id, bytes)?)
        } else {
            None
        };
        self.tap_layer_ids = tap_layer_ids.to_vec();
        self.max_tap_rows = max_tap_rows;
        self.tap_aux_bf16 = tap_aux_bf16;
        self.last_final_hidden_f16 = last_final_hidden_f16;
        if tap_layer_ids.is_empty() && !export_final_hidden && max_tap_rows == 0 {
            self.captured_spec_logits_ptr = 0;
            self.speculative_logits_rows = 0;
            self.speculative_logits = None;
        }
        Ok(())
    }

    /// Maximum aligned ordinary prefill budget, including worst-case selected outputs.
    pub fn ordinary_prefill_token_capacity(&self, max_num_seqs: usize) -> Result<usize> {
        let max_chunks = if self.cfg.uses_tiered_moe() { 96 } else { 32 };
        scratch::token_capacity_up_to(
            self.scratch_arena.capacity(),
            self.cfg.hidden_size,
            self.local_vocab_size,
            max_num_seqs,
            max_chunks,
            |tokens| {
                let hc_extra = if self.final_residual.is_hyper_connection() {
                    let streams = self.cfg.hc_count.max(1);
                    tokens * (streams - 1) * self.cfg.hidden_size * 2 + tokens * streams * 4 + 512
                } else {
                    0
                };
                layer_scratch_layout(&self.cfg, self.tp_size, tokens)
                    .ok()
                    .map(|l| {
                        let transform = if matches!(
                            self.cfg.quant,
                            QuantScheme::CompressedTensors { rules: Some(_), .. }
                        ) {
                            let max_dim = self
                                .cfg
                                .hidden_size
                                .max(self.cfg.intermediate_size)
                                .max(self.cfg.num_attention_heads * self.cfg.head_dim)
                                .max(
                                    self.cfg.linear_num_value_heads
                                        * self.cfg.linear_value_head_dim,
                                )
                                .div_ceil(128)
                                * 128;
                            tokens
                                .saturating_mul(max_dim)
                                .saturating_mul(4)
                                .saturating_add(512)
                        } else {
                            0
                        };
                        l.0.saturating_add(hc_extra).saturating_add(transform)
                    })
            },
        )
        .ok_or_else(|| loader_err("scratch cannot fit one 64-token prefill chunk"))
    }
}

// SAFETY: moving a rank transfers its CUDA owners, not host references into device
// storage. Forward/load entrypoints select the device on the executing thread;
// scoped TP workers receive exclusive rank borrows and join before returning.
// Moving is not synchronization: callers must still order queued work and must
// not use copied raw addresses concurrently with the new owner.
#[cfg(feature = "cuda")]
unsafe impl Send for Qwen3_5ModelRank {}
