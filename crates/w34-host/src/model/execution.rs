//! Token stepping, single-request decode, and uncached generation.

use super::*;

#[cfg(feature = "cuda")]
#[allow(clippy::too_many_arguments)]
fn stage_step_tokens_meta_on_rank(
    r: &mut Qwen3_5ModelRank,
    tokens: &[i32],
    pos_host: &[i32],
    write_slots: &[i32],
    block_table: &[i32],
    ctx_before: usize,
    seq_after: usize,
    state_in_slot: i32,
    state_out_slot: i32,
    has_initial_state: bool,
) -> Result<(u64, u64, W34AttentionBatch, W34GdnBatch)> {
    let num_tokens = tokens.len();
    let tok_words = num_tokens;
    let pos_words = pos_host.len();
    let slot_words = write_slots.len();
    let blk_words = block_table.len();
    let tail_words = [
        0i32,
        num_tokens as i32,
        ctx_before as i32,
        seq_after as i32,
        state_in_slot,
        state_out_slot,
        i32::from(has_initial_state),
    ];
    let total_words = tok_words + pos_words + slot_words + blk_words + tail_words.len();
    let total_bytes = total_words * 4;

    r.batch_meta_arena.reset();
    let dev_base = r.batch_meta_arena.alloc_aligned(total_bytes.max(64))?;
    if total_bytes <= BatchedDecodeGraphIo::PINNED_META_BYTES {
        if r.batched_decode_io
            .pinned_in_flight
            .swap(false, std::sync::atomic::Ordering::AcqRel)
        {
            r.stream.synchronize()?;
        }
        let pbase = r.batched_decode_io.pinned_staging.host_ptr.cast::<i32>();
        // SAFETY: `total_bytes <= PINNED_META_BYTES <= pinned_staging.len_bytes` and
        // `host_ptr` is page-aligned (`>= align_of::<i32>()`). Prior slot-0 use is synced above.
        unsafe {
            let mut cur = 0usize;
            core::ptr::copy_nonoverlapping(tokens.as_ptr(), pbase.add(cur), tok_words);
            cur += tok_words;
            core::ptr::copy_nonoverlapping(pos_host.as_ptr(), pbase.add(cur), pos_words);
            cur += pos_words;
            core::ptr::copy_nonoverlapping(write_slots.as_ptr(), pbase.add(cur), slot_words);
            cur += slot_words;
            if blk_words > 0 {
                core::ptr::copy_nonoverlapping(block_table.as_ptr(), pbase.add(cur), blk_words);
                cur += blk_words;
            }
            core::ptr::copy_nonoverlapping(tail_words.as_ptr(), pbase.add(cur), tail_words.len());
            r.batched_decode_io
                .pinned_in_flight
                .store(true, std::sync::atomic::Ordering::Release);
            check_rc(
                ffi::w34_memcpy_h2d_async(
                    dev_base as *mut c_void,
                    pbase.cast(),
                    total_bytes,
                    r.stream.handle,
                ),
                "upload pinned step_tokens meta",
            )?;
        }
    } else {
        let mut packed = Vec::with_capacity(total_words);
        packed.extend_from_slice(tokens);
        packed.extend_from_slice(pos_host);
        packed.extend_from_slice(write_slots);
        packed.extend_from_slice(block_table);
        packed.extend_from_slice(&tail_words);
        check_rc(
            // SAFETY: `dev_base` spans `total_bytes == packed.len() * 4` on `r.stream`.
            unsafe {
                ffi::w34_memcpy_h2d_async(
                    dev_base as *mut c_void,
                    packed.as_ptr().cast(),
                    total_bytes,
                    r.stream.handle,
                )
            },
            "upload packed step_tokens meta",
        )?;
    }

    let tok_ptr = dev_base;
    let pos_ptr = tok_ptr + (tok_words * 4) as u64;
    let slot_ptr = pos_ptr + (pos_words * 4) as u64;
    let blk_ptr = slot_ptr + (slot_words * 4) as u64;
    let q_start_ptr = blk_ptr + (blk_words * 4) as u64;
    let ctx_ptr = q_start_ptr + 8;
    let seq_ptr = ctx_ptr + 4;
    let st_in_ptr = seq_ptr + 4;
    let st_out_ptr = st_in_ptr + 4;
    let has_init_ptr = st_out_ptr + 4;

    let attn_batch = W34AttentionBatch {
        query_start_loc: q_start_ptr as *const i32,
        context_lens_before: ctx_ptr as *const i32,
        seq_lens_after: seq_ptr as *const i32,
        block_tables: blk_ptr as *const i32,
        write_slot_mapping: slot_ptr as *const i32,
        valid_token_mask: core::ptr::null(),
        valid_req_mask: core::ptr::null(),
        num_reqs: 1,
        total_tokens: num_tokens as i32,
        max_blocks_per_req: block_table.len() as i32,
        max_q_len: num_tokens as i32,
        max_seq_len: seq_after as i32,
        mask_mode: if r.active_noncausal_full_attention() {
            w34_kernels::ffi::W34_ATTN_MASK_SYMMETRIC_WINDOW
        } else {
            W34_ATTN_MASK_CAUSAL
        },
        sliding_window: 0,
        sm_scale: (r.cfg.head_dim as f32).powf(-0.5),
        reserved: 0,
    };
    let gdn_batch = W34GdnBatch {
        query_start_loc: q_start_ptr as *const i32,
        state_in_indices: st_in_ptr as *const i32,
        state_out_indices: st_out_ptr as *const i32,
        spec_state_indices: core::ptr::null(),
        checkpoint_dst_indices: core::ptr::null(),
        checkpoint_token_offsets: core::ptr::null(),
        has_initial_state: has_init_ptr as *const u8,
        valid_req_mask: core::ptr::null(),
        num_reqs: 1,
        total_tokens: num_tokens as i32,
        max_verify_tokens: 0,
        max_checkpoints_per_req: 0,
    };
    Ok((tok_ptr, pos_ptr, attn_batch, gdn_batch))
}

#[cfg(feature = "cuda")]
fn run_decode_one_eager_on_rank(
    r: &mut Qwen3_5ModelRank,
    seq_after: usize,
    lora_id: i32,
    tp2: Option<&Tp2Collectives>,
) -> Result<()> {
    r.set_lora_uniform(lora_id);
    let attn_batch = W34AttentionBatch {
        query_start_loc: r.decode_io.q_start_loc.addr() as *const i32,
        context_lens_before: r.decode_io.ctx_lens_before_ptr() as *const i32,
        seq_lens_after: r.decode_io.seq_lens_after_ptr() as *const i32,
        block_tables: r.decode_io.block_tables.addr() as *const i32,
        write_slot_mapping: r.decode_io.write_slot_mapping_ptr() as *const i32,
        valid_token_mask: core::ptr::null(),
        valid_req_mask: core::ptr::null(),
        num_reqs: 1,
        total_tokens: 1,
        max_blocks_per_req: r.decode_io.max_blocks_per_req as i32,
        max_q_len: 1,
        max_seq_len: seq_after as i32,
        mask_mode: W34_ATTN_MASK_CAUSAL,
        sliding_window: 0,
        sm_scale: (r.cfg.head_dim as f32).powf(-0.5),
        reserved: 0,
    };
    let gdn_batch = W34GdnBatch {
        query_start_loc: r.decode_io.q_start_loc.addr() as *const i32,
        state_in_indices: r.decode_io.state_in_indices_ptr() as *const i32,
        state_out_indices: r.decode_io.state_out_indices_ptr() as *const i32,
        spec_state_indices: core::ptr::null(),
        checkpoint_dst_indices: core::ptr::null(),
        checkpoint_token_offsets: core::ptr::null(),
        has_initial_state: r.decode_io.has_initial_state.addr() as *const u8,
        valid_req_mask: core::ptr::null(),
        num_reqs: 1,
        total_tokens: 1,
        max_verify_tokens: 0,
        max_checkpoints_per_req: 0,
    };
    let tok_ptr = r.decode_io.token_ids_ptr();
    let pos_ptr = r.decode_io.positions_ptr();
    let out_id_ptr = r.decode_io.out_greedy_id_ptr();
    r.forward_device_batches(
        tok_ptr,
        pos_ptr,
        1,
        &attn_batch,
        &gdn_batch,
        tp2,
        out_id_ptr,
        0,
        0,
        None,
    )?;
    Ok(())
}

#[cfg(feature = "cuda")]
impl Qwen3_5Engine {
    pub fn encode(&self, text: &str) -> Result<Vec<i32>> {
        let tok = self
            .tokenizer
            .as_ref()
            .ok_or_else(|| loader_err("tokenizer.json not loaded"))?;
        let enc = tok
            .encode(text, false)
            .map_err(|_| loader_err("Tokenizer::encode failed"))?;
        Ok(enc.get_ids().iter().map(|&id| id as i32).collect())
    }

    pub fn decode(&self, ids: &[i32]) -> Result<String> {
        let tok = self
            .tokenizer
            .as_ref()
            .ok_or_else(|| loader_err("tokenizer.json not loaded"))?;
        let u32_ids: Vec<u32> = ids.iter().map(|&id| id as u32).collect();
        tok.decode(&u32_ids, true)
            .map_err(|_| loader_err("Tokenizer::decode failed"))
    }

    pub fn reset_all_states(&mut self) -> Result<()> {
        self.radix.reset();
        self.radix.active_lora_id = self.active_lora_id;
        self.vision_embed_cache.clear();
        if let Some(enc) = &self.vision_encoder {
            enc.clear_workspace();
        }
        if let Some(proposer) = &mut self.speculative {
            proposer.clear_all_caches()?;
        }
        for r in &mut self.ranks {
            r.device.make_current()?;
            r.kv_pool.reset_all_pages();
            r.state_pool.reset_all_slots();
            for st in &r.state_pool.layers {
                st.conv_state.zero_async(&r.stream)?;
                st.ssm_state.zero_async(&r.stream)?;
            }
            if let Some(ple_st) = &r.state_pool.ple_conv_state {
                ple_st.zero_async(&r.stream)?;
            }
            r.stream.synchronize()?;
        }
        Ok(())
    }

    /// Runs a single prefill or multi-token step from `ctx_before` to `ctx_before + tokens.len()`,
    /// returning `(next_greedy_token_id, top5_ids, top5_logprobs)` and optionally capturing a `ForwardDump`.
    #[allow(clippy::too_many_arguments)]
    pub fn step_tokens(
        &mut self,
        tokens: &[i32],
        ctx_before: usize,
        block_table: &[i32],
        state_in_slot: i32,
        state_out_slot: i32,
        has_initial_state: bool,
        dump_out: Option<&mut ForwardDump>,
    ) -> Result<(i32, [i32; 5], [f32; 5])> {
        self.step_tokens_with_positions(
            tokens,
            None,
            ctx_before,
            block_table,
            state_in_slot,
            state_out_slot,
            has_initial_state,
            dump_out,
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub fn step_tokens_with_positions(
        &mut self,
        tokens: &[i32],
        custom_positions: Option<&[i32]>,
        ctx_before: usize,
        block_table: &[i32],
        state_in_slot: i32,
        state_out_slot: i32,
        has_initial_state: bool,
        mut dump_out: Option<&mut ForwardDump>,
    ) -> Result<(i32, [i32; 5], [f32; 5])> {
        let num_tokens = tokens.len();
        if num_tokens == 0 {
            return Err(loader_err("step_tokens requires at least 1 token"));
        }
        let max_prefill_chunk = self
            .ranks
            .iter()
            .map(|rank| rank.ordinary_prefill_token_capacity(1))
            .collect::<Result<Vec<_>>>()?
            .into_iter()
            .min()
            .ok_or_else(|| loader_err("no model ranks"))?;
        if num_tokens > max_prefill_chunk
            && custom_positions.is_none()
            && dump_out.is_none()
            && self.ranks[0].tap_layer_ids.is_empty()
            && self.ranks[0].last_final_hidden_f16.is_none()
            && self.ranks[0].pending_visual_splice.is_none()
        {
            let mut offset = 0usize;
            let mut last = (0i32, [0i32; 5], [0.0f32; 5]);
            while offset < num_tokens {
                let end = (offset + max_prefill_chunk).min(num_tokens);
                let in_slot = if offset == 0 {
                    state_in_slot
                } else {
                    state_out_slot
                };
                let init = if offset == 0 { has_initial_state } else { true };
                if end < num_tokens {
                    let _ = self.step_batch_slices_for_sampler_filtered(
                        &[EngineBatchSlice {
                            tokens: &tokens[offset..end],
                            ctx_len_before: ctx_before + offset,
                            block_table,
                            state_in_slot: in_slot,
                            state_out_slot,
                            has_initial_state: init,
                            spec_slots: &[],
                            checkpoints: &[],
                            emit_all_query_tokens: false,
                        }],
                        None,
                        Some(&[false]),
                        true,
                    )?;
                } else {
                    last = self.step_tokens_with_positions(
                        &tokens[offset..end],
                        None,
                        ctx_before + offset,
                        block_table,
                        in_slot,
                        state_out_slot,
                        init,
                        None,
                    )?;
                }
                offset = end;
            }
            return Ok(last);
        }
        let seq_after = ctx_before + num_tokens;
        let max_pages = self.ranks[0].kv_pool.num_pages;
        if seq_after.div_ceil(16) > block_table.len()
            || block_table.len() > max_pages
            || block_table
                .iter()
                .any(|&p| p < 0 || (p as usize) >= max_pages)
        {
            return Err(loader_err("KV cache page capacity exceeded in step_tokens"));
        }
        let positions: Vec<i32> = (ctx_before..seq_after).map(|p| p as i32).collect();
        if let Some(cp) = custom_positions
            && cp.len() != num_tokens
            && cp.len() != 3 * num_tokens
        {
            return Err(loader_err(
                "custom_positions length must be num_tokens or 3 * num_tokens",
            ));
        }
        let pos_host: &[i32] = custom_positions.unwrap_or(&positions);
        let write_slots: Vec<i32> = (ctx_before..seq_after)
            .map(|pos| {
                let page_idx = pos / 16;
                let page_off = pos % 16;
                block_table[page_idx] * 16 + (page_off as i32)
            })
            .collect();
        let q_start = [0i32, num_tokens as i32];
        let st_in_arr = [state_in_slot];
        let st_out_arr = [state_out_slot];
        let has_init_arr = [u8::from(has_initial_state)];
        let active_lora_id = self.active_lora_id;
        for r in &mut self.ranks {
            r.last_batch_state_slots.clear();
            if has_initial_state
                && state_in_slot >= 0
                && (state_in_slot as usize) < r.state_pool.total_slots
            {
                r.last_batch_state_slots.push(state_in_slot as usize);
            }
        }

        if self.tp_size == 1 {
            let r0 = &mut self.ranks[0];
            r0.device.make_current()?;
            r0.set_lora_uniform(active_lora_id);
            r0.stage_ple_f8_for_batch(
                tokens,
                &q_start,
                &st_in_arr,
                &st_out_arr,
                &has_init_arr,
                &[],
                0,
                &[],
                &[],
                0,
            )?;
            let (tok_ptr, pos_ptr, attn_batch, gdn_batch) = stage_step_tokens_meta_on_rank(
                r0,
                tokens,
                pos_host,
                &write_slots,
                block_table,
                ctx_before,
                seq_after,
                state_in_slot,
                state_out_slot,
                has_initial_state,
            )?;
            let out_id_ptr = r0.decode_io.out_greedy_id_ptr();
            let top5_id_ptr = r0.decode_io.out_top5_ids_ptr();
            let top5_lp_ptr = r0.decode_io.out_top5_logprobs_ptr();
            r0.forward_device_batches(
                tok_ptr,
                pos_ptr,
                num_tokens,
                &attn_batch,
                &gdn_batch,
                None,
                out_id_ptr,
                top5_id_ptr,
                top5_lp_ptr,
                dump_out.as_deref_mut(),
            )?;
            let (greedy_id, _, top5_ids, top5_lps) = r0
                .decode_io
                .read_step_outputs_pinned(&r0.batched_decode_io, &r0.stream)?;
            if let Some(d) = dump_out {
                d.greedy_token_id = greedy_id;
                d.top5_ids = top5_ids;
                d.top5_logprobs = top5_lps;
            }
            Ok((greedy_id, top5_ids, top5_lps))
        } else {
            let comm = self.tp2_comm.as_ref().unwrap().clone();
            let comm_ref = &comm;
            let pool = self.tp_workers.as_ref().unwrap().clone();
            let mut dump0 = if dump_out.is_some() {
                Some(ForwardDump::default())
            } else {
                None
            };
            let pos_slice = pos_host;
            let slots_slice = write_slots.as_slice();
            let d0_ref = &mut dump0;
            let (res0, peer_results) = pool.run_split(
                self.ranks.as_mut_slice(),
                move |r0| -> Result<(i32, f32, [i32; 5], [f32; 5])> {
                    r0.device.make_current()?;
                    r0.set_lora_uniform(active_lora_id);
                    r0.stage_ple_f8_for_batch(
                        tokens,
                        &q_start,
                        &st_in_arr,
                        &st_out_arr,
                        &has_init_arr,
                        &[],
                        0,
                        &[],
                        &[],
                        0,
                    )?;
                    let (tok_ptr, pos_ptr, attn_batch, gdn_batch) = stage_step_tokens_meta_on_rank(
                        r0,
                        tokens,
                        pos_slice,
                        slots_slice,
                        block_table,
                        ctx_before,
                        seq_after,
                        state_in_slot,
                        state_out_slot,
                        has_initial_state,
                    )?;
                    let out_id_ptr = r0.decode_io.out_greedy_id_ptr();
                    let top5_id_ptr = r0.decode_io.out_top5_ids_ptr();
                    let top5_lp_ptr = r0.decode_io.out_top5_logprobs_ptr();
                    r0.forward_device_batches(
                        tok_ptr,
                        pos_ptr,
                        num_tokens,
                        &attn_batch,
                        &gdn_batch,
                        Some(comm_ref),
                        out_id_ptr,
                        top5_id_ptr,
                        top5_lp_ptr,
                        d0_ref.as_mut(),
                    )?;
                    r0.decode_io
                        .read_step_outputs_pinned(&r0.batched_decode_io, &r0.stream)
                },
                |_, r_peer| -> Result<(f32, [i32; 5], [f32; 5])> {
                    r_peer.device.make_current()?;
                    r_peer.set_lora_uniform(active_lora_id);
                    r_peer.stage_ple_f8_for_batch(
                        tokens,
                        &q_start,
                        &st_in_arr,
                        &st_out_arr,
                        &has_init_arr,
                        &[],
                        0,
                        &[],
                        &[],
                        0,
                    )?;
                    let (tok_ptr, pos_ptr, attn_batch, gdn_batch) = stage_step_tokens_meta_on_rank(
                        r_peer,
                        tokens,
                        pos_slice,
                        slots_slice,
                        block_table,
                        ctx_before,
                        seq_after,
                        state_in_slot,
                        state_out_slot,
                        has_initial_state,
                    )?;
                    let out_id_ptr = r_peer.decode_io.out_greedy_id_ptr();
                    let top5_id_ptr = r_peer.decode_io.out_top5_ids_ptr();
                    let top5_lp_ptr = r_peer.decode_io.out_top5_logprobs_ptr();
                    r_peer.forward_device_batches(
                        tok_ptr,
                        pos_ptr,
                        num_tokens,
                        &attn_batch,
                        &gdn_batch,
                        Some(comm_ref),
                        out_id_ptr,
                        top5_id_ptr,
                        top5_lp_ptr,
                        None,
                    )?;
                    let (_, local_max, top5_ids, top5_lps) = r_peer
                        .decode_io
                        .read_step_outputs_pinned(&r_peer.batched_decode_io, &r_peer.stream)?;
                    Ok((local_max, top5_ids, top5_lps))
                },
            );
            let (gid, max0, ids0, lps0) = res0?;
            let mut all_shards = Vec::with_capacity(1 + peer_results.len());
            all_shards.push((max0, ids0, lps0));
            for pr in peer_results {
                all_shards.push(pr?);
            }
            // Reconstruct exact global log-sum-exp across all shards and merge top-5
            let log_zs: Vec<f32> = all_shards.iter().map(|(m, _, lps)| *m - lps[0]).collect();
            let m_z = log_zs.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let sum_exp: f32 = log_zs.iter().map(|&lz| (lz - m_z).exp()).sum();
            let log_z_global = m_z + sum_exp.ln();
            let mut cands = Vec::with_capacity(all_shards.len() * 5);
            for ((_, ids, lps), &lz) in all_shards.iter().zip(&log_zs) {
                for k in 0..5 {
                    cands.push((lps[k] + lz - log_z_global, ids[k]));
                }
            }
            cands.sort_by(|a, b| b.0.total_cmp(&a.0).then_with(|| a.1.cmp(&b.1)));
            let mut top5_ids = [0i32; 5];
            let mut top5_lps = [0.0f32; 5];
            for k in 0..5 {
                top5_lps[k] = cands[k].0;
                top5_ids[k] = cands[k].1;
            }
            if let (Some(dst), Some(src)) = (dump_out, dump0) {
                dst.tensors_f32.extend(src.tensors_f32);
                dst.draft_proposal_ids = src.draft_proposal_ids;
                dst.greedy_token_id = gid;
                dst.top5_ids = top5_ids;
                dst.top5_logprobs = top5_lps;
            }
            Ok((gid, top5_ids, top5_lps))
        }
    }

    /// Executes a single `M=1` decode step using `DecodeGraphIo` (and `CudaGraph` replay when enabled).
    pub fn decode_one_token(
        &mut self,
        token_id: i32,
        ctx_before: usize,
        block_table: &[i32],
        state_slot: i32,
    ) -> Result<i32> {
        let seq_after = ctx_before + 1;
        let max_pages = self.ranks[0].kv_pool.num_pages;
        if ctx_before / 16 >= block_table.len()
            || block_table.len() > self.ranks[0].decode_io.max_blocks_per_req
            || block_table
                .iter()
                .any(|&p| p < 0 || (p as usize) >= max_pages)
        {
            return Err(loader_err(
                "KV cache page capacity exceeded in decode_one_token",
            ));
        }
        let write_slot = block_table[ctx_before / 16] * 16 + ((ctx_before % 16) as i32);
        let tok_arr = [token_id];
        let st_arr = [state_slot];
        let lora_id = self.active_lora_id;
        let meta = [
            token_id,
            ctx_before as i32,
            ctx_before as i32,
            seq_after as i32,
            write_slot,
            state_slot,
            state_slot,
            lora_id,
        ];
        for r in &mut self.ranks {
            r.last_batch_state_slots.clear();
            let slot_u = if state_slot >= 0 && (state_slot as usize) < r.state_pool.total_slots {
                state_slot as usize
            } else {
                0
            };
            r.last_batch_state_slots.push(slot_u);
            r.stage_ple_f8_for_batch(
                &tok_arr,
                &[0, 1],
                &st_arr,
                &st_arr,
                &[1u8],
                &[],
                0,
                &[],
                &[],
                0,
            )?;
        }

        let lora_mode = if self.lora_registry.is_empty() {
            GraphLoraMode::BaseOnly
        } else {
            GraphLoraMode::from_single(lora_id)
        };
        if self.tp_size == 1 {
            let r0 = &mut self.ranks[0];
            r0.device.make_current()?;
            r0.decode_io.upload_step_pinned(
                &meta,
                block_table,
                &r0.batched_decode_io,
                &r0.stream,
            )?;
            let long_ctx = !self.cfg.uses_qsa() && seq_after > 1024;
            if self.use_cuda_graph && r0.supports_cuda_graph_for_tokens_with_lora(1, lora_mode) {
                r0.ensure_decode_graph(long_ctx, lora_mode, None)?;
                r0.bind_graph_lora_mode(lora_mode, r0.decode_io.token_lora_ids_ptr());
                r0.decode_graph.last().unwrap().2.launch(&r0.stream)?;
                r0.mark_tiered_moe_device_lru_dirty();
                r0.last_logits_ptr = r0.captured_decode_logits_ptr;
                r0.last_logits_rows = 1;
            } else {
                run_decode_one_eager_on_rank(r0, seq_after, lora_id, None)?;
            }
            r0.decode_io
                .read_greedy_id_pinned(&r0.batched_decode_io, &r0.stream)
        } else {
            let comm = self.tp2_comm.as_ref().unwrap().clone();
            let use_graph = self.use_cuda_graph
                && self
                    .ranks
                    .iter()
                    .all(|r| r.supports_cuda_graph_for_tokens_with_lora(1, lora_mode));
            let long_ctx = !self.cfg.uses_qsa() && seq_after > 1024;
            if use_graph {
                use crate::coordinated::{ReplayPhase, replay_ranks};
                let res = replay_ranks(self.ranks.len(), |phase, rank| {
                    let r = &mut self.ranks[rank];
                    r.device.make_current()?;
                    match phase {
                        ReplayPhase::Upload => r.decode_io.upload_step_pinned(
                            &meta,
                            block_table,
                            &r.batched_decode_io,
                            &r.stream,
                        ),
                        ReplayPhase::Capture => {
                            r.ensure_decode_graph(long_ctx, lora_mode, Some(&comm))
                        }
                        ReplayPhase::Launch => {
                            r.bind_graph_lora_mode(lora_mode, r.decode_io.token_lora_ids_ptr());
                            r.decode_graph.last().unwrap().2.launch(&r.stream)?;
                            r.mark_tiered_moe_device_lru_dirty();
                            r.last_logits_ptr = r.captured_decode_logits_ptr;
                            r.last_logits_rows = 1;
                            Ok(())
                        }
                    }
                });
                res.map_err(|e| {
                    if e.partial_launch {
                        eprintln!(
                            "[w34] fatal: asymmetric TP decode_one_token graph launch failure: {}",
                            e.source
                        );
                        std::process::abort();
                    }
                    e.source
                })?;
                self.ranks[0].device.make_current()?;
                let gid = self.ranks[0].decode_io.read_greedy_id_pinned(
                    &self.ranks[0].batched_decode_io,
                    &self.ranks[0].stream,
                )?;
                for r in &self.ranks[1..] {
                    r.device.make_current()?;
                    r.stream.synchronize()?;
                    r.batched_decode_io
                        .pinned_in_flight
                        .store(false, std::sync::atomic::Ordering::Release);
                }
                return Ok(gid);
            }
            let comm_ref = &comm;
            let pool = self.tp_workers.as_ref().unwrap().clone();
            let (res0, peer_results) = pool.run_split(
                self.ranks.as_mut_slice(),
                |r0| -> Result<i32> {
                    r0.device.make_current()?;
                    r0.decode_io.upload_step_pinned(
                        &meta,
                        block_table,
                        &r0.batched_decode_io,
                        &r0.stream,
                    )?;
                    run_decode_one_eager_on_rank(r0, seq_after, lora_id, Some(comm_ref))?;
                    r0.decode_io
                        .read_greedy_id_pinned(&r0.batched_decode_io, &r0.stream)
                },
                |_, r_peer| -> Result<()> {
                    r_peer.device.make_current()?;
                    r_peer.decode_io.upload_step_pinned(
                        &meta,
                        block_table,
                        &r_peer.batched_decode_io,
                        &r_peer.stream,
                    )?;
                    run_decode_one_eager_on_rank(r_peer, seq_after, lora_id, Some(comm_ref))?;
                    r_peer.stream.synchronize()?;
                    r_peer
                        .batched_decode_io
                        .pinned_in_flight
                        .store(false, std::sync::atomic::Ordering::Release);
                    Ok(())
                },
            );
            for pr in peer_results {
                pr?;
            }
            res0
        }
    }

    /// Generates `max_new_tokens` greedily from `prompt_token_ids`, returning `(generated_ids, BenchMetrics)`.
    pub fn generate_tokens(
        &mut self,
        prompt_token_ids: &[i32],
        max_new_tokens: usize,
        stop_on_eos: bool,
    ) -> Result<BenchMetrics> {
        self.reset_all_states()?;
        if self.speculative.is_some() {
            let max_chunk = if self.cfg.uses_tiered_moe() {
                6144
            } else {
                2048
            };
            let chunk = self
                .ranks
                .iter()
                .filter_map(|r| r.ordinary_prefill_token_capacity(1).ok())
                .min()
                .unwrap_or(2048)
                .clamp(64, max_chunk);
            let m = self.generate_native_speculative(
                prompt_token_ids,
                max_new_tokens,
                chunk,
                stop_on_eos,
            )?;
            return Ok(BenchMetrics {
                prompt_len: m.prompt_len,
                gen_len: m.gen_len,
                ttft_ms: m.ttft_ms,
                prefill_tok_per_s: m.prefill_tok_per_s,
                decode_ms: m.decode_ms,
                decode_tok_per_s: m.decode_tok_per_s,
                itl_ms: m.itl_ms,
                generated_token_ids: m.generated_token_ids,
            });
        }
        let total_needed_tokens = prompt_token_ids.len() + max_new_tokens + 4;
        let needed_pages = total_needed_tokens.div_ceil(16);
        let max_pages = self.ranks[0]
            .kv_pool
            .num_pages
            .min(self.ranks[0].decode_io.max_blocks_per_req);
        if needed_pages > max_pages {
            return Err(loader_err(
                "prompt + max_new_tokens exceeds allocated KV page capacity",
            ));
        }
        let mut block_table = Vec::with_capacity(needed_pages);
        for p in 0..needed_pages {
            block_table.push(p as i32);
        }
        let mut tx = StateTx::new(0);
        tx.begin_prefill();

        let t0 = std::time::Instant::now();
        let (first_tok, _, _) = self.step_tokens(
            prompt_token_ids,
            0,
            &block_table,
            tx.committed_slot,
            tx.committed_slot,
            false,
            None,
        )?;
        let ttft_ms = t0.elapsed().as_secs_f64() * 1000.0;
        tx.commit_prefill(prompt_token_ids.len());

        let mut gen_ids = Vec::with_capacity(max_new_tokens);
        if max_new_tokens > 0 {
            gen_ids.push(first_tok);
        }

        let decode_range = crate::profiling::Range::new(c"w34.decode");
        let t_dec_start = std::time::Instant::now();
        let mut cur_tok = first_tok;
        for (ctx_len, _) in (prompt_token_ids.len()..).zip(1..max_new_tokens) {
            if stop_on_eos && self.cfg.eos_token_ids.contains(&cur_tok) {
                break;
            }
            let next_tok =
                self.decode_one_token(cur_tok, ctx_len, &block_table, tx.committed_slot)?;
            gen_ids.push(next_tok);
            cur_tok = next_tok;
        }
        drop(decode_range);
        let decode_ms = t_dec_start.elapsed().as_secs_f64() * 1000.0;
        let decode_steps = gen_ids.len().saturating_sub(1);
        let decode_tok_per_s = if decode_ms > 0.0 && decode_steps > 0 {
            (decode_steps as f64) / (decode_ms / 1000.0)
        } else {
            0.0
        };
        let itl_ms = if decode_steps > 0 {
            decode_ms / (decode_steps as f64)
        } else {
            0.0
        };
        let prefill_tok_per_s = if ttft_ms > 0.0 {
            (prompt_token_ids.len() as f64) / (ttft_ms / 1000.0)
        } else {
            0.0
        };

        Ok(BenchMetrics {
            prompt_len: prompt_token_ids.len(),
            gen_len: gen_ids.len(),
            ttft_ms,
            prefill_tok_per_s,
            decode_ms,
            decode_tok_per_s,
            itl_ms,
            generated_token_ids: gen_ids,
        })
    }
}
