//! Token stepping, single-request decode, and uncached generation.

use super::*;

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
        let ctx_arr = [ctx_before as i32];
        let seq_arr = [seq_after as i32];
        let st_in_arr = [state_in_slot];
        let st_out_arr = [state_out_slot];
        let has_init_arr = [u8::from(has_initial_state)];

        if self.tp_size == 1 {
            let r0 = &mut self.ranks[0];
            r0.device.make_current()?;
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
            let tok_dev = DeviceBuffer::alloc(r0.device.device_id, num_tokens * 4)?;
            let pos_dev = DeviceBuffer::alloc(r0.device.device_id, pos_host.len() * 4)?;
            let q_start_dev = DeviceBuffer::alloc(r0.device.device_id, 8)?;
            let ctx_dev = DeviceBuffer::alloc(r0.device.device_id, 4)?;
            let seq_dev = DeviceBuffer::alloc(r0.device.device_id, 4)?;
            let blk_dev = DeviceBuffer::alloc(r0.device.device_id, block_table.len() * 4)?;
            let slot_dev = DeviceBuffer::alloc(r0.device.device_id, num_tokens * 4)?;
            let st_in_dev = DeviceBuffer::alloc(r0.device.device_id, 4)?;
            let st_out_dev = DeviceBuffer::alloc(r0.device.device_id, 4)?;
            let has_init_dev = DeviceBuffer::alloc(r0.device.device_id, 4)?;

            tok_dev.copy_from_host(tokens, &r0.stream)?;
            pos_dev.copy_from_host(pos_host, &r0.stream)?;
            q_start_dev.copy_from_host(&q_start, &r0.stream)?;
            ctx_dev.copy_from_host(&ctx_arr, &r0.stream)?;
            seq_dev.copy_from_host(&seq_arr, &r0.stream)?;
            blk_dev.copy_from_host(block_table, &r0.stream)?;
            slot_dev.copy_from_host(&write_slots, &r0.stream)?;
            st_in_dev.copy_from_host(&st_in_arr, &r0.stream)?;
            st_out_dev.copy_from_host(&st_out_arr, &r0.stream)?;
            has_init_dev.copy_from_host(&has_init_arr, &r0.stream)?;

            let attn_batch = W34AttentionBatch {
                query_start_loc: q_start_dev.addr() as *const i32,
                context_lens_before: ctx_dev.addr() as *const i32,
                seq_lens_after: seq_dev.addr() as *const i32,
                block_tables: blk_dev.addr() as *const i32,
                write_slot_mapping: slot_dev.addr() as *const i32,
                valid_token_mask: core::ptr::null(),
                valid_req_mask: core::ptr::null(),
                num_reqs: 1,
                total_tokens: num_tokens as i32,
                max_blocks_per_req: block_table.len() as i32,
                max_q_len: num_tokens as i32,
                max_seq_len: seq_after as i32,
                mask_mode: W34_ATTN_MASK_CAUSAL,
                sliding_window: 0,
                sm_scale: (r0.cfg.head_dim as f32).powf(-0.5),
                reserved: 0,
            };
            let gdn_batch = W34GdnBatch {
                query_start_loc: q_start_dev.addr() as *const i32,
                state_in_indices: st_in_dev.addr() as *const i32,
                state_out_indices: st_out_dev.addr() as *const i32,
                spec_state_indices: core::ptr::null(),
                checkpoint_dst_indices: core::ptr::null(),
                checkpoint_token_offsets: core::ptr::null(),
                has_initial_state: has_init_dev.addr() as *const u8,
                valid_req_mask: core::ptr::null(),
                num_reqs: 1,
                total_tokens: num_tokens as i32,
                max_verify_tokens: 0,
                max_checkpoints_per_req: 0,
            };

            let out_id_ptr = r0.decode_io.out_greedy_id.addr();
            let top5_id_ptr = r0.decode_io.out_top5_ids.addr();
            let top5_lp_ptr = r0.decode_io.out_top5_logprobs.addr();
            r0.forward_device_batches(
                tok_dev.addr(),
                pos_dev.addr(),
                num_tokens,
                &attn_batch,
                &gdn_batch,
                None,
                out_id_ptr,
                top5_id_ptr,
                top5_lp_ptr,
                dump_out.as_deref_mut(),
            )?;
            let mut greedy_id = [0i32; 1];
            let mut top5_ids = [0i32; 5];
            let mut top5_lps = [0.0f32; 5];
            r0.decode_io
                .out_greedy_id
                .copy_to_host(&mut greedy_id, &r0.stream)?;
            r0.decode_io
                .out_top5_ids
                .copy_to_host(&mut top5_ids, &r0.stream)?;
            r0.decode_io
                .out_top5_logprobs
                .copy_to_host(&mut top5_lps, &r0.stream)?;
            r0.stream.synchronize()?;
            if let Some(d) = dump_out {
                d.greedy_token_id = greedy_id[0];
                d.top5_ids = top5_ids;
                d.top5_logprobs = top5_lps;
            }
            Ok((greedy_id[0], top5_ids, top5_lps))
        } else {
            let comm = self.tp2_comm.as_ref().unwrap().clone();
            let [r0, r1] = &mut self.ranks[..] else {
                unreachable!()
            };
            let mut dump0 = if dump_out.is_some() {
                Some(ForwardDump::default())
            } else {
                None
            };
            let pos_slice = pos_host;
            let slots_slice = write_slots.as_slice();
            let (res0, res1) = std::thread::scope(|sc| {
                let c0 = comm.clone();
                let d0_ref = &mut dump0;
                let h0 = sc.spawn(move || -> Result<(i32, f32, [i32; 5], [f32; 5])> {
                    r0.device.make_current()?;
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
                    let tok_dev = DeviceBuffer::alloc(r0.device.device_id, num_tokens * 4)?;
                    let pos_dev = DeviceBuffer::alloc(r0.device.device_id, pos_slice.len() * 4)?;
                    let q_start_dev = DeviceBuffer::alloc(r0.device.device_id, 8)?;
                    let ctx_dev = DeviceBuffer::alloc(r0.device.device_id, 4)?;
                    let seq_dev = DeviceBuffer::alloc(r0.device.device_id, 4)?;
                    let blk_dev = DeviceBuffer::alloc(r0.device.device_id, block_table.len() * 4)?;
                    let slot_dev = DeviceBuffer::alloc(r0.device.device_id, num_tokens * 4)?;
                    let st_in_dev = DeviceBuffer::alloc(r0.device.device_id, 4)?;
                    let st_out_dev = DeviceBuffer::alloc(r0.device.device_id, 4)?;
                    let has_init_dev = DeviceBuffer::alloc(r0.device.device_id, 4)?;

                    tok_dev.copy_from_host(tokens, &r0.stream)?;
                    pos_dev.copy_from_host(pos_slice, &r0.stream)?;
                    q_start_dev.copy_from_host(&q_start, &r0.stream)?;
                    ctx_dev.copy_from_host(&ctx_arr, &r0.stream)?;
                    seq_dev.copy_from_host(&seq_arr, &r0.stream)?;
                    blk_dev.copy_from_host(block_table, &r0.stream)?;
                    slot_dev.copy_from_host(slots_slice, &r0.stream)?;
                    st_in_dev.copy_from_host(&st_in_arr, &r0.stream)?;
                    st_out_dev.copy_from_host(&st_out_arr, &r0.stream)?;
                    has_init_dev.copy_from_host(&has_init_arr, &r0.stream)?;

                    let attn_batch = W34AttentionBatch {
                        query_start_loc: q_start_dev.addr() as *const i32,
                        context_lens_before: ctx_dev.addr() as *const i32,
                        seq_lens_after: seq_dev.addr() as *const i32,
                        block_tables: blk_dev.addr() as *const i32,
                        write_slot_mapping: slot_dev.addr() as *const i32,
                        valid_token_mask: core::ptr::null(),
                        valid_req_mask: core::ptr::null(),
                        num_reqs: 1,
                        total_tokens: num_tokens as i32,
                        max_blocks_per_req: block_table.len() as i32,
                        max_q_len: num_tokens as i32,
                        max_seq_len: seq_after as i32,
                        mask_mode: W34_ATTN_MASK_CAUSAL,
                        sliding_window: 0,
                        sm_scale: (r0.cfg.head_dim as f32).powf(-0.5),
                        reserved: 0,
                    };
                    let gdn_batch = W34GdnBatch {
                        query_start_loc: q_start_dev.addr() as *const i32,
                        state_in_indices: st_in_dev.addr() as *const i32,
                        state_out_indices: st_out_dev.addr() as *const i32,
                        spec_state_indices: core::ptr::null(),
                        checkpoint_dst_indices: core::ptr::null(),
                        checkpoint_token_offsets: core::ptr::null(),
                        has_initial_state: has_init_dev.addr() as *const u8,
                        valid_req_mask: core::ptr::null(),
                        num_reqs: 1,
                        total_tokens: num_tokens as i32,
                        max_verify_tokens: 0,
                        max_checkpoints_per_req: 0,
                    };
                    let out_id_ptr = r0.decode_io.out_greedy_id.addr();
                    let top5_id_ptr = r0.decode_io.out_top5_ids.addr();
                    let top5_lp_ptr = r0.decode_io.out_top5_logprobs.addr();
                    r0.forward_device_batches(
                        tok_dev.addr(),
                        pos_dev.addr(),
                        num_tokens,
                        &attn_batch,
                        &gdn_batch,
                        Some(&c0),
                        out_id_ptr,
                        top5_id_ptr,
                        top5_lp_ptr,
                        d0_ref.as_mut(),
                    )?;
                    let mut greedy_id = [0i32; 1];
                    let mut local_max = [0.0f32; 1];
                    let mut top5_ids = [0i32; 5];
                    let mut top5_lps = [0.0f32; 5];
                    r0.decode_io
                        .out_greedy_id
                        .copy_to_host(&mut greedy_id, &r0.stream)?;
                    r0.decode_io
                        .local_max_val
                        .copy_to_host(&mut local_max, &r0.stream)?;
                    r0.decode_io
                        .out_top5_ids
                        .copy_to_host(&mut top5_ids, &r0.stream)?;
                    r0.decode_io
                        .out_top5_logprobs
                        .copy_to_host(&mut top5_lps, &r0.stream)?;
                    r0.stream.synchronize()?;
                    Ok((greedy_id[0], local_max[0], top5_ids, top5_lps))
                });

                let c1 = comm.clone();
                let h1 = sc.spawn(move || -> Result<(f32, [i32; 5], [f32; 5])> {
                    r1.device.make_current()?;
                    r1.stage_ple_f8_for_batch(
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
                    let tok_dev = DeviceBuffer::alloc(r1.device.device_id, num_tokens * 4)?;
                    let pos_dev = DeviceBuffer::alloc(r1.device.device_id, pos_slice.len() * 4)?;
                    let q_start_dev = DeviceBuffer::alloc(r1.device.device_id, 8)?;
                    let ctx_dev = DeviceBuffer::alloc(r1.device.device_id, 4)?;
                    let seq_dev = DeviceBuffer::alloc(r1.device.device_id, 4)?;
                    let blk_dev = DeviceBuffer::alloc(r1.device.device_id, block_table.len() * 4)?;
                    let slot_dev = DeviceBuffer::alloc(r1.device.device_id, num_tokens * 4)?;
                    let st_in_dev = DeviceBuffer::alloc(r1.device.device_id, 4)?;
                    let st_out_dev = DeviceBuffer::alloc(r1.device.device_id, 4)?;
                    let has_init_dev = DeviceBuffer::alloc(r1.device.device_id, 4)?;

                    tok_dev.copy_from_host(tokens, &r1.stream)?;
                    pos_dev.copy_from_host(pos_slice, &r1.stream)?;
                    q_start_dev.copy_from_host(&q_start, &r1.stream)?;
                    ctx_dev.copy_from_host(&ctx_arr, &r1.stream)?;
                    seq_dev.copy_from_host(&seq_arr, &r1.stream)?;
                    blk_dev.copy_from_host(block_table, &r1.stream)?;
                    slot_dev.copy_from_host(slots_slice, &r1.stream)?;
                    st_in_dev.copy_from_host(&st_in_arr, &r1.stream)?;
                    st_out_dev.copy_from_host(&st_out_arr, &r1.stream)?;
                    has_init_dev.copy_from_host(&has_init_arr, &r1.stream)?;

                    let attn_batch = W34AttentionBatch {
                        query_start_loc: q_start_dev.addr() as *const i32,
                        context_lens_before: ctx_dev.addr() as *const i32,
                        seq_lens_after: seq_dev.addr() as *const i32,
                        block_tables: blk_dev.addr() as *const i32,
                        write_slot_mapping: slot_dev.addr() as *const i32,
                        valid_token_mask: core::ptr::null(),
                        valid_req_mask: core::ptr::null(),
                        num_reqs: 1,
                        total_tokens: num_tokens as i32,
                        max_blocks_per_req: block_table.len() as i32,
                        max_q_len: num_tokens as i32,
                        max_seq_len: seq_after as i32,
                        mask_mode: W34_ATTN_MASK_CAUSAL,
                        sliding_window: 0,
                        sm_scale: (r1.cfg.head_dim as f32).powf(-0.5),
                        reserved: 0,
                    };
                    let gdn_batch = W34GdnBatch {
                        query_start_loc: q_start_dev.addr() as *const i32,
                        state_in_indices: st_in_dev.addr() as *const i32,
                        state_out_indices: st_out_dev.addr() as *const i32,
                        spec_state_indices: core::ptr::null(),
                        checkpoint_dst_indices: core::ptr::null(),
                        checkpoint_token_offsets: core::ptr::null(),
                        has_initial_state: has_init_dev.addr() as *const u8,
                        valid_req_mask: core::ptr::null(),
                        num_reqs: 1,
                        total_tokens: num_tokens as i32,
                        max_verify_tokens: 0,
                        max_checkpoints_per_req: 0,
                    };
                    let out_id_ptr = r1.decode_io.out_greedy_id.addr();
                    let top5_id_ptr = r1.decode_io.out_top5_ids.addr();
                    let top5_lp_ptr = r1.decode_io.out_top5_logprobs.addr();
                    r1.forward_device_batches(
                        tok_dev.addr(),
                        pos_dev.addr(),
                        num_tokens,
                        &attn_batch,
                        &gdn_batch,
                        Some(&c1),
                        out_id_ptr,
                        top5_id_ptr,
                        top5_lp_ptr,
                        None,
                    )?;
                    let mut local_max = [0.0f32; 1];
                    let mut top5_ids = [0i32; 5];
                    let mut top5_lps = [0.0f32; 5];
                    r1.decode_io
                        .local_max_val
                        .copy_to_host(&mut local_max, &r1.stream)?;
                    r1.decode_io
                        .out_top5_ids
                        .copy_to_host(&mut top5_ids, &r1.stream)?;
                    r1.decode_io
                        .out_top5_logprobs
                        .copy_to_host(&mut top5_lps, &r1.stream)?;
                    r1.stream.synchronize()?;
                    Ok((local_max[0], top5_ids, top5_lps))
                });
                (h0.join().unwrap(), h1.join().unwrap())
            });
            let (gid, max0, ids0, lps0) = res0?;
            let (max1, ids1, lps1) = res1?;
            // Reconstruct exact global log-sum-exp across both shards and merge top-5
            let log_z0 = max0 - lps0[0];
            let log_z1 = max1 - lps1[0];
            let m_z = log_z0.max(log_z1);
            let log_z_global = m_z + ((log_z0 - m_z).exp() + (log_z1 - m_z).exp()).ln();
            let mut cands = [(0.0f32, 0i32); 10];
            for k in 0..5 {
                cands[k] = (lps0[k] + log_z0 - log_z_global, ids0[k]);
                cands[5 + k] = (lps1[k] + log_z1 - log_z_global, ids1[k]);
            }
            cands.sort_by(|a, b| b.0.total_cmp(&a.0).then_with(|| a.1.cmp(&b.1)));
            let mut top5_ids = [0i32; 5];
            let mut top5_lps = [0.0f32; 5];
            for k in 0..5 {
                top5_lps[k] = cands[k].0;
                top5_ids[k] = cands[k].1;
            }
            if let (Some(dst), Some(src)) = (dump_out, dump0) {
                *dst = src;
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
        let pos_arr = [ctx_before as i32];
        let ctx_arr = [ctx_before as i32];
        let seq_arr = [seq_after as i32];
        let slot_arr = [write_slot];
        let st_arr = [state_slot];
        for r in &mut self.ranks {
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

        if self.tp_size == 1 {
            let r0 = &mut self.ranks[0];
            r0.device.make_current()?;
            r0.decode_io
                .token_ids
                .copy_from_host(&tok_arr, &r0.stream)?;
            r0.decode_io
                .positions
                .copy_from_host(&pos_arr, &r0.stream)?;
            r0.decode_io
                .ctx_lens_before
                .copy_from_host(&ctx_arr, &r0.stream)?;
            r0.decode_io
                .seq_lens_after
                .copy_from_host(&seq_arr, &r0.stream)?;
            r0.decode_io
                .block_tables
                .copy_from_host(block_table, &r0.stream)?;
            r0.decode_io
                .write_slot_mapping
                .copy_from_host(&slot_arr, &r0.stream)?;
            r0.decode_io
                .state_in_indices
                .copy_from_host(&st_arr, &r0.stream)?;
            r0.decode_io
                .state_out_indices
                .copy_from_host(&st_arr, &r0.stream)?;

            let long_ctx = seq_after > 1024;
            if self.use_cuda_graph && r0.supports_cuda_graph_for_tokens(1) {
                r0.ensure_decode_graph(long_ctx, None)?;
                r0.decode_graph.as_ref().unwrap().1.launch(&r0.stream)?;
                r0.mark_tiered_moe_device_lru_dirty();
                r0.last_logits_ptr = r0.captured_decode_logits_ptr;
                r0.last_logits_rows = 1;
            } else {
                let attn_batch = W34AttentionBatch {
                    query_start_loc: r0.decode_io.q_start_loc.addr() as *const i32,
                    context_lens_before: r0.decode_io.ctx_lens_before.addr() as *const i32,
                    seq_lens_after: r0.decode_io.seq_lens_after.addr() as *const i32,
                    block_tables: r0.decode_io.block_tables.addr() as *const i32,
                    write_slot_mapping: r0.decode_io.write_slot_mapping.addr() as *const i32,
                    valid_token_mask: core::ptr::null(),
                    valid_req_mask: core::ptr::null(),
                    num_reqs: 1,
                    total_tokens: 1,
                    max_blocks_per_req: r0.decode_io.max_blocks_per_req as i32,
                    max_q_len: 1,
                    max_seq_len: seq_after as i32,
                    mask_mode: W34_ATTN_MASK_CAUSAL,
                    sliding_window: 0,
                    sm_scale: (r0.cfg.head_dim as f32).powf(-0.5),
                    reserved: 0,
                };
                let gdn_batch = W34GdnBatch {
                    query_start_loc: r0.decode_io.q_start_loc.addr() as *const i32,
                    state_in_indices: r0.decode_io.state_in_indices.addr() as *const i32,
                    state_out_indices: r0.decode_io.state_out_indices.addr() as *const i32,
                    spec_state_indices: core::ptr::null(),
                    checkpoint_dst_indices: core::ptr::null(),
                    checkpoint_token_offsets: core::ptr::null(),
                    has_initial_state: r0.decode_io.has_initial_state.addr() as *const u8,
                    valid_req_mask: core::ptr::null(),
                    num_reqs: 1,
                    total_tokens: 1,
                    max_verify_tokens: 0,
                    max_checkpoints_per_req: 0,
                };
                let tok_ptr = r0.decode_io.token_ids.addr();
                let pos_ptr = r0.decode_io.positions.addr();
                let out_id_ptr = r0.decode_io.out_greedy_id.addr();
                r0.forward_device_batches(
                    tok_ptr,
                    pos_ptr,
                    1,
                    &attn_batch,
                    &gdn_batch,
                    None,
                    out_id_ptr,
                    0,
                    0,
                    None,
                )?;
            }
            let mut gid = [0i32; 1];
            r0.decode_io
                .out_greedy_id
                .copy_to_host(&mut gid, &r0.stream)?;
            r0.stream.synchronize()?;
            Ok(gid[0])
        } else {
            let comm = self.tp2_comm.as_ref().unwrap().clone();
            let use_graph = self.use_cuda_graph
                && self
                    .ranks
                    .iter()
                    .all(|r| r.supports_cuda_graph_for_tokens(1));
            let [r0, r1] = &mut self.ranks[..] else {
                unreachable!()
            };
            let long_ctx = !self.cfg.uses_qsa() && seq_after > 1024;
            if use_graph
                && r0
                    .decode_graph
                    .as_ref()
                    .is_some_and(|(lc, _)| *lc == long_ctx)
                && r1
                    .decode_graph
                    .as_ref()
                    .is_some_and(|(lc, _)| *lc == long_ctx)
            {
                for r in [&mut *r0, &mut *r1] {
                    r.device.make_current()?;
                    r.decode_io.token_ids.copy_from_host(&tok_arr, &r.stream)?;
                    r.decode_io.positions.copy_from_host(&pos_arr, &r.stream)?;
                    r.decode_io
                        .ctx_lens_before
                        .copy_from_host(&ctx_arr, &r.stream)?;
                    r.decode_io
                        .seq_lens_after
                        .copy_from_host(&seq_arr, &r.stream)?;
                    r.decode_io
                        .block_tables
                        .copy_from_host(block_table, &r.stream)?;
                    r.decode_io
                        .write_slot_mapping
                        .copy_from_host(&slot_arr, &r.stream)?;
                    r.decode_io
                        .state_in_indices
                        .copy_from_host(&st_arr, &r.stream)?;
                    r.decode_io
                        .state_out_indices
                        .copy_from_host(&st_arr, &r.stream)?;
                    r.decode_graph.as_ref().unwrap().1.launch(&r.stream)?;
                    r.mark_tiered_moe_device_lru_dirty();
                    r.last_logits_ptr = r.captured_decode_logits_ptr;
                    r.last_logits_rows = 1;
                }
                r0.device.make_current()?;
                let mut gid = [0i32; 1];
                r0.decode_io
                    .out_greedy_id
                    .copy_to_host(&mut gid, &r0.stream)?;
                r0.stream.synchronize()?;
                r1.device.make_current()?;
                r1.stream.synchronize()?;
                return Ok(gid[0]);
            }
            let (res0, res1) = std::thread::scope(|sc| {
                let c0 = comm.clone();
                let h0 = sc.spawn(move || -> Result<i32> {
                    r0.device.make_current()?;
                    r0.decode_io
                        .token_ids
                        .copy_from_host(&tok_arr, &r0.stream)?;
                    r0.decode_io
                        .positions
                        .copy_from_host(&pos_arr, &r0.stream)?;
                    r0.decode_io
                        .ctx_lens_before
                        .copy_from_host(&ctx_arr, &r0.stream)?;
                    r0.decode_io
                        .seq_lens_after
                        .copy_from_host(&seq_arr, &r0.stream)?;
                    r0.decode_io
                        .block_tables
                        .copy_from_host(block_table, &r0.stream)?;
                    r0.decode_io
                        .write_slot_mapping
                        .copy_from_host(&slot_arr, &r0.stream)?;
                    r0.decode_io
                        .state_in_indices
                        .copy_from_host(&st_arr, &r0.stream)?;
                    r0.decode_io
                        .state_out_indices
                        .copy_from_host(&st_arr, &r0.stream)?;
                    if use_graph {
                        r0.ensure_decode_graph(long_ctx, Some(&c0))?;
                        r0.decode_graph.as_ref().unwrap().1.launch(&r0.stream)?;
                        r0.mark_tiered_moe_device_lru_dirty();
                        r0.last_logits_ptr = r0.captured_decode_logits_ptr;
                        r0.last_logits_rows = 1;
                    } else {
                        let attn_batch = W34AttentionBatch {
                            query_start_loc: r0.decode_io.q_start_loc.addr() as *const i32,
                            context_lens_before: r0.decode_io.ctx_lens_before.addr() as *const i32,
                            seq_lens_after: r0.decode_io.seq_lens_after.addr() as *const i32,
                            block_tables: r0.decode_io.block_tables.addr() as *const i32,
                            write_slot_mapping: r0.decode_io.write_slot_mapping.addr()
                                as *const i32,
                            valid_token_mask: core::ptr::null(),
                            valid_req_mask: core::ptr::null(),
                            num_reqs: 1,
                            total_tokens: 1,
                            max_blocks_per_req: r0.decode_io.max_blocks_per_req as i32,
                            max_q_len: 1,
                            max_seq_len: seq_after as i32,
                            mask_mode: W34_ATTN_MASK_CAUSAL,
                            sliding_window: 0,
                            sm_scale: (r0.cfg.head_dim as f32).powf(-0.5),
                            reserved: 0,
                        };
                        let gdn_batch = W34GdnBatch {
                            query_start_loc: r0.decode_io.q_start_loc.addr() as *const i32,
                            state_in_indices: r0.decode_io.state_in_indices.addr() as *const i32,
                            state_out_indices: r0.decode_io.state_out_indices.addr() as *const i32,
                            spec_state_indices: core::ptr::null(),
                            checkpoint_dst_indices: core::ptr::null(),
                            checkpoint_token_offsets: core::ptr::null(),
                            has_initial_state: r0.decode_io.has_initial_state.addr() as *const u8,
                            valid_req_mask: core::ptr::null(),
                            num_reqs: 1,
                            total_tokens: 1,
                            max_verify_tokens: 0,
                            max_checkpoints_per_req: 0,
                        };
                        let tok_ptr = r0.decode_io.token_ids.addr();
                        let pos_ptr = r0.decode_io.positions.addr();
                        let out_id_ptr = r0.decode_io.out_greedy_id.addr();
                        r0.forward_device_batches(
                            tok_ptr,
                            pos_ptr,
                            1,
                            &attn_batch,
                            &gdn_batch,
                            Some(&c0),
                            out_id_ptr,
                            0,
                            0,
                            None,
                        )?;
                    }
                    let mut gid = [0i32; 1];
                    r0.decode_io
                        .out_greedy_id
                        .copy_to_host(&mut gid, &r0.stream)?;
                    r0.stream.synchronize()?;
                    Ok(gid[0])
                });

                let c1 = comm.clone();
                let h1 = sc.spawn(move || -> Result<()> {
                    r1.device.make_current()?;
                    r1.decode_io
                        .token_ids
                        .copy_from_host(&tok_arr, &r1.stream)?;
                    r1.decode_io
                        .positions
                        .copy_from_host(&pos_arr, &r1.stream)?;
                    r1.decode_io
                        .ctx_lens_before
                        .copy_from_host(&ctx_arr, &r1.stream)?;
                    r1.decode_io
                        .seq_lens_after
                        .copy_from_host(&seq_arr, &r1.stream)?;
                    r1.decode_io
                        .block_tables
                        .copy_from_host(block_table, &r1.stream)?;
                    r1.decode_io
                        .write_slot_mapping
                        .copy_from_host(&slot_arr, &r1.stream)?;
                    r1.decode_io
                        .state_in_indices
                        .copy_from_host(&st_arr, &r1.stream)?;
                    r1.decode_io
                        .state_out_indices
                        .copy_from_host(&st_arr, &r1.stream)?;
                    if use_graph {
                        r1.ensure_decode_graph(long_ctx, Some(&c1))?;
                        r1.decode_graph.as_ref().unwrap().1.launch(&r1.stream)?;
                        r1.mark_tiered_moe_device_lru_dirty();
                        r1.last_logits_ptr = r1.captured_decode_logits_ptr;
                        r1.last_logits_rows = 1;
                    } else {
                        let attn_batch = W34AttentionBatch {
                            query_start_loc: r1.decode_io.q_start_loc.addr() as *const i32,
                            context_lens_before: r1.decode_io.ctx_lens_before.addr() as *const i32,
                            seq_lens_after: r1.decode_io.seq_lens_after.addr() as *const i32,
                            block_tables: r1.decode_io.block_tables.addr() as *const i32,
                            write_slot_mapping: r1.decode_io.write_slot_mapping.addr()
                                as *const i32,
                            valid_token_mask: core::ptr::null(),
                            valid_req_mask: core::ptr::null(),
                            num_reqs: 1,
                            total_tokens: 1,
                            max_blocks_per_req: r1.decode_io.max_blocks_per_req as i32,
                            max_q_len: 1,
                            max_seq_len: seq_after as i32,
                            mask_mode: W34_ATTN_MASK_CAUSAL,
                            sliding_window: 0,
                            sm_scale: (r1.cfg.head_dim as f32).powf(-0.5),
                            reserved: 0,
                        };
                        let gdn_batch = W34GdnBatch {
                            query_start_loc: r1.decode_io.q_start_loc.addr() as *const i32,
                            state_in_indices: r1.decode_io.state_in_indices.addr() as *const i32,
                            state_out_indices: r1.decode_io.state_out_indices.addr() as *const i32,
                            spec_state_indices: core::ptr::null(),
                            checkpoint_dst_indices: core::ptr::null(),
                            checkpoint_token_offsets: core::ptr::null(),
                            has_initial_state: r1.decode_io.has_initial_state.addr() as *const u8,
                            valid_req_mask: core::ptr::null(),
                            num_reqs: 1,
                            total_tokens: 1,
                            max_verify_tokens: 0,
                            max_checkpoints_per_req: 0,
                        };
                        let tok_ptr = r1.decode_io.token_ids.addr();
                        let pos_ptr = r1.decode_io.positions.addr();
                        let out_id_ptr = r1.decode_io.out_greedy_id.addr();
                        r1.forward_device_batches(
                            tok_ptr,
                            pos_ptr,
                            1,
                            &attn_batch,
                            &gdn_batch,
                            Some(&c1),
                            out_id_ptr,
                            0,
                            0,
                            None,
                        )?;
                    }
                    r1.stream.synchronize()?;
                    Ok(())
                });
                (h0.join().unwrap(), h1.join().unwrap())
            });
            res1?;
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
        self.radix = MambaRadixCache::new(16, 64);
        if self.speculative.is_some() {
            let m = self.generate_native_speculative(
                prompt_token_ids,
                max_new_tokens,
                2048,
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
