//! Radix-cached generation, vision, speculative verification, and scheduling.

use super::*;

#[cfg(feature = "cuda")]
impl Qwen3_5Engine {
    pub fn is_eos_token(&self, token_id: i32) -> bool {
        self.cfg.eos_token_ids.contains(&token_id)
    }

    pub(crate) fn alloc_three_state_slots(&mut self) -> Option<(i32, [i32; 2])> {
        let mut radix = std::mem::replace(&mut self.radix, MambaRadixCache::new(16, 64));
        let out = self.alloc_three_state_slots_from(&mut radix);
        self.radix = radix;
        out
    }

    pub(crate) fn alloc_three_state_slots_from(
        &mut self,
        radix: &mut MambaRadixCache,
    ) -> Option<(i32, [i32; 2])> {
        let s0 = self.alloc_state_slot_from(radix)?;
        let Some(s1) = self.alloc_state_slot_from(radix) else {
            self.free_state_slots(&[s0]);
            return None;
        };
        let Some(s2) = self.alloc_state_slot_from(radix) else {
            self.free_state_slots(&[s0, s1]);
            return None;
        };
        Some((s0, [s1, s2]))
    }

    pub(crate) fn ensure_kv_pages_for_len(
        &mut self,
        req_state: &mut ReqRadixState,
        target_len: usize,
    ) -> Result<()> {
        let needed_pages = target_len.div_ceil(16);
        let mut radix = std::mem::replace(&mut self.radix, MambaRadixCache::new(16, 64));
        let ok = self.reserve_kv_pages_from(&mut radix, &mut req_state.kv_pages, needed_pages);
        self.radix = radix;
        if !ok {
            return Err(loader_err("KvPool out of pages"));
        }
        Ok(())
    }

    /// Generates `max_new_tokens` using `MambaRadixCache` warm prefix lookup + chunked suffix prefill
    /// (`prefill_chunk_size`, e.g. `64` or `8`) + `extra_buffer` ping-pong checkpoint donation.
    pub fn generate_with_radix_cache(
        &mut self,
        prompt_token_ids: &[i32],
        max_new_tokens: usize,
        prefill_chunk_size: usize,
        stop_on_eos: bool,
    ) -> Result<RadixBenchMetrics> {
        if self.speculative.is_some() {
            return self.generate_native_speculative(
                prompt_token_ids,
                max_new_tokens,
                prefill_chunk_size,
                stop_on_eos,
            );
        }
        let params = crate::parser::SamplingParams {
            max_tokens: max_new_tokens,
            stop_token_ids: self.cfg.eos_token_ids.clone(),
            eos_token_ids: self.cfg.eos_token_ids.clone(),
            ..Default::default()
        };
        self.generate_plain_sampled_with_radix_cache(
            prompt_token_ids,
            params,
            None,
            prefill_chunk_size,
            stop_on_eos,
        )
    }

    pub fn generate_plain_sampled_with_radix_cache(
        &mut self,
        prompt_token_ids: &[i32],
        params: crate::parser::SamplingParams,
        constraint: Option<crate::parser::TokenBitmaskConstraint>,
        prefill_chunk_size: usize,
        stop_on_eos: bool,
    ) -> Result<RadixBenchMetrics> {
        params
            .validate()
            .map_err(|_| loader_err("invalid sampling parameters"))?;
        if prompt_token_ids.is_empty() {
            return Err(loader_err("empty prompt in generate_with_radix_cache"));
        }
        let max_new_tokens = params.max_tokens;
        let use_gpu_sampler = !params.is_plain_greedy(constraint.is_some());
        let mut sampler = crate::parser::SamplerV1::new(params, prompt_token_ids, constraint);
        let target_total_len = prompt_token_ids
            .len()
            .checked_add(max_new_tokens)
            .and_then(|v| v.checked_add(4))
            .ok_or_else(|| loader_err("token length overflow in generate_with_radix_cache"))?;
        let needed_pages = target_total_len.div_ceil(16);
        let max_pool_pages = self.ranks[0].kv_pool.num_pages;
        if needed_pages > max_pool_pages {
            return Err(loader_err(
                "requested sequence pages exceed KV pool capacity",
            ));
        }
        let (active_slot, [pp0, pp1]) = self
            .alloc_three_state_slots()
            .ok_or_else(|| loader_err("StateBundlePool out of slots"))?;

        let lookup_len = prompt_token_ids.len().saturating_sub(1);
        let match_res = self.radix.match_prefix(&prompt_token_ids[..lookup_len]);
        let cached_prefix_tokens = match_res.matched_tokens;

        let mut req_state = ReqRadixState::new(16, 64, active_slot, [pp0, pp1]);
        req_state.restore_prefix_hit(&mut self.radix, prompt_token_ids, &match_res);

        let step_res: Result<(f64, f64, Vec<i32>, Vec<crate::parser::StepLogprobs>)> = (|| {
            self.ensure_kv_pages_for_len(&mut req_state, target_total_len)?;

            let chunk_sz = prefill_chunk_size.max(1);
            let t0 = std::time::Instant::now();
            let mut first_tok = 0i32;
            let mut cursor = cached_prefix_tokens;
            while cursor < prompt_token_ids.len() {
                let rem = prompt_token_ids.len() - cursor;
                let take = rem.min(chunk_sz);
                let slice_tokens = &prompt_token_ids[cursor..cursor + take];
                let planned = req_state.plan_checkpoints(take);
                let ckpt_pairs: Vec<(i32, i32)> = planned
                    .iter()
                    .map(|c| (c.trigger_offset_1based, c.dst_slot))
                    .collect();
                let is_final_chunk = cursor + take >= prompt_token_ids.len();
                let batch_out = self.step_batch_slices_for_sampler_filtered(
                    &[EngineBatchSlice {
                        tokens: slice_tokens,
                        ctx_len_before: cursor,
                        block_table: &req_state.kv_pages,
                        state_in_slot: req_state.current_state_in_slot(),
                        state_out_slot: req_state.active_state_slot,
                        has_initial_state: req_state.has_initial_state,
                        spec_slots: &[],
                        checkpoints: &ckpt_pairs,
                        emit_all_query_tokens: false,
                    }],
                    None,
                    Some(&[is_final_chunk]),
                    is_final_chunk && use_gpu_sampler,
                )?;
                if is_final_chunk && !use_gpu_sampler {
                    first_tok = batch_out[0][0];
                }
                req_state.commit_tokens(slice_tokens, &planned);
                cursor += take;
            }
            let ttft_ms = t0.elapsed().as_secs_f64() * 1000.0;

            let mut gen_ids = Vec::with_capacity(max_new_tokens);
            let mut token_logprobs = Vec::new();
            if max_new_tokens > 0 {
                if use_gpu_sampler {
                    let (tok, lp) = self.sample_single_gpu_step(&mut sampler)?;
                    first_tok = tok;
                    if let Some(lp) = lp {
                        token_logprobs.push(lp);
                    }
                } else {
                    sampler.commit_token(first_tok);
                }
                gen_ids.push(first_tok);
            }

            let t_dec = std::time::Instant::now();
            let mut cur_tok = first_tok;
            for _ in 1..max_new_tokens {
                if stop_on_eos
                    && sampler.output_history.len() >= sampler.params.min_tokens
                    && sampler
                        .params
                        .is_stop_or_eos(cur_tok, &self.cfg.eos_token_ids)
                {
                    break;
                }
                let ctx_before = req_state.token_ids.len();
                let planned = req_state.plan_checkpoints(1);
                let ckpt_pairs: Vec<(i32, i32)> = planned
                    .iter()
                    .map(|c| (c.trigger_offset_1based, c.dst_slot))
                    .collect();
                let mut next_tok =
                    if ckpt_pairs.is_empty() && req_state.cow_src_state_slot.is_none() {
                        self.decode_one_token(
                            cur_tok,
                            ctx_before,
                            &req_state.kv_pages,
                            req_state.active_state_slot,
                        )?
                    } else {
                        let out = self.step_batch_slices(&[EngineBatchSlice {
                            tokens: &[cur_tok],
                            ctx_len_before: ctx_before,
                            block_table: &req_state.kv_pages,
                            state_in_slot: req_state.current_state_in_slot(),
                            state_out_slot: req_state.active_state_slot,
                            has_initial_state: req_state.has_initial_state,
                            spec_slots: &[],
                            checkpoints: &ckpt_pairs,
                            emit_all_query_tokens: false,
                        }])?;
                        out[0][0]
                    };
                if use_gpu_sampler {
                    let (tok, lp) = self.sample_single_gpu_step(&mut sampler)?;
                    next_tok = tok;
                    if let Some(lp) = lp {
                        token_logprobs.push(lp);
                    }
                } else {
                    sampler.commit_token(next_tok);
                }
                req_state.commit_tokens(&[cur_tok], &planned);
                gen_ids.push(next_tok);
                cur_tok = next_tok;
            }
            let decode_ms = t_dec.elapsed().as_secs_f64() * 1000.0;
            Ok((ttft_ms, decode_ms, gen_ids, token_logprobs))
        })();

        let (ttft_ms, decode_ms, gen_ids, token_logprobs) = match step_res {
            Ok(v) => v,
            Err(e) => {
                let aborted = req_state.abort_and_release(&mut self.radix);
                self.free_kv_pages(&aborted.freed_kv_pages);
                self.free_state_slots(&aborted.freed_mamba_slots);
                return Err(e);
            }
        };
        let donation = req_state.donate_committed_checkpoint(&mut self.radix, None, true);
        self.free_kv_pages(&donation.freed_kv_pages);
        self.free_state_slots(&donation.freed_mamba_slots);

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
        let suffix_len = (prompt_token_ids.len() - cached_prefix_tokens).max(1);
        let prefill_tok_per_s = if ttft_ms > 0.0 {
            (suffix_len as f64) / (ttft_ms / 1000.0)
        } else {
            0.0
        };

        Ok(RadixBenchMetrics {
            prompt_len: prompt_token_ids.len(),
            cached_prefix_tokens,
            donated_seq_len: donation.donated_seq_len,
            gen_len: gen_ids.len(),
            ttft_ms,
            prefill_tok_per_s,
            decode_ms,
            decode_tok_per_s,
            itl_ms,
            generated_token_ids: gen_ids,
            token_logprobs,
        })
    }

    /// Lazily loads [`crate::vision::Qwen3VisionEncoder`] on GPU 0 from `self.model_dir`.
    pub fn ensure_vision_encoder(&mut self) -> Result<()> {
        if self.vision_encoder.is_none() {
            let vcfg = self
                .cfg
                .vision_config
                .as_ref()
                .ok_or_else(|| loader_err("model does not have a vision_config"))?;
            let archive = SafetensorsArchive::open(&self.model_dir)
                .map_err(|_| loader_err("SafetensorsArchive::open for vision_encoder"))?;
            let r0 = &self.ranks[0];
            r0.device.make_current()?;
            let enc =
                crate::vision::Qwen3VisionEncoder::try_load(&archive, vcfg, 0, &r0.stream)?
                    .ok_or_else(|| loader_err("vision weights not found in safetensors archive"))?;
            self.vision_encoder = Some(enc);
        }
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn step_single_slice_with_mrope_and_vision(
        &mut self,
        slice: &EngineBatchSlice<'_>,
        custom_positions: &[i32],
        pos_plane_stride: i32,
        vis_splice_per_rank: Option<&[(u64, u64, usize)]>,
    ) -> Result<i32> {
        let q_len = slice.tokens.len();
        if q_len == 0 {
            return Err(loader_err(
                "empty slice in step_single_slice_with_mrope_and_vision",
            ));
        }
        let seq_after = slice.ctx_len_before + q_len;
        let max_pages = self.ranks[0].kv_pool.num_pages;
        if seq_after.div_ceil(16) > slice.block_table.len()
            || slice
                .block_table
                .iter()
                .any(|&p| p < 0 || (p as usize) >= max_pages)
        {
            return Err(loader_err(
                "KV cache page capacity exceeded in multimodal step",
            ));
        }
        let mut flat_write_slots = Vec::with_capacity(q_len);
        for step in 0..q_len {
            let pos = slice.ctx_len_before + step;
            let page_id = slice.block_table[pos / 16];
            let page_off = (pos % 16) as i32;
            flat_write_slots.push(page_id * 16 + page_off);
        }
        let q_start_loc = [0i32, q_len as i32];
        let ctx_lens_before = [slice.ctx_len_before as i32];
        let seq_lens_after = [seq_after as i32];
        let state_in_indices = [slice.state_in_slot];
        let state_out_indices = [slice.state_out_slot];
        let has_initial_state = [u8::from(slice.has_initial_state)];
        let max_checkpoints_per_req = slice.checkpoints.len();
        let mut checkpoint_token_offsets = Vec::with_capacity(max_checkpoints_per_req);
        let mut checkpoint_dst_indices = Vec::with_capacity(max_checkpoints_per_req);
        for &(trig, dst) in slice.checkpoints {
            checkpoint_token_offsets.push(trig);
            checkpoint_dst_indices.push(dst);
        }
        let selected_rows = [q_len - 1];

        for (r_idx, rank) in self.ranks.iter_mut().enumerate() {
            rank.current_pos_plane_stride = pos_plane_stride;
            rank.pending_visual_splice = vis_splice_per_rank.and_then(|s| s.get(r_idx).copied());
        }

        let run_res: Result<i32> = (|| {
            if self.tp_size == 1 {
                let r0 = &mut self.ranks[0];
                let (tok_ptr, pos_ptr, attn_batch, gdn_batch) = r0.upload_batch_meta(
                    slice.tokens,
                    custom_positions,
                    &q_start_loc,
                    &ctx_lens_before,
                    &seq_lens_after,
                    slice.block_table,
                    &flat_write_slots,
                    &state_in_indices,
                    &state_out_indices,
                    &has_initial_state,
                    &[],
                    &checkpoint_dst_indices,
                    &checkpoint_token_offsets,
                    1,
                    q_len,
                    slice.block_table.len(),
                    q_len,
                    seq_after,
                    0,
                    max_checkpoints_per_req,
                )?;
                let ids = r0.forward_selected_rows(
                    tok_ptr,
                    pos_ptr,
                    q_len,
                    &attn_batch,
                    &gdn_batch,
                    None,
                    &selected_rows,
                )?;
                Ok(ids[0])
            } else {
                let comm = self.tp2_comm.as_ref().unwrap().clone();
                let (r0, peer_ranks) = self.ranks.split_first_mut().unwrap();
                let ft_ref = slice.tokens;
                let fp_ref = custom_positions;
                let qs_ref = &q_start_loc;
                let cb_ref = &ctx_lens_before;
                let sa_ref = &seq_lens_after;
                let bt_ref = slice.block_table;
                let ws_ref = &flat_write_slots;
                let si_ref = &state_in_indices;
                let so_ref = &state_out_indices;
                let hi_ref = &has_initial_state;
                let cd_ref = &checkpoint_dst_indices;
                let co_ref = &checkpoint_token_offsets;
                let sr_ref = &selected_rows;
                let (res0, peer_results) = std::thread::scope(|sc| {
                    let c0 = comm.clone();
                    let h0 = sc.spawn(move || -> Result<Vec<i32>> {
                        let (tok_ptr, pos_ptr, attn_batch, gdn_batch) = r0.upload_batch_meta(
                            ft_ref,
                            fp_ref,
                            qs_ref,
                            cb_ref,
                            sa_ref,
                            bt_ref,
                            ws_ref,
                            si_ref,
                            so_ref,
                            hi_ref,
                            &[],
                            cd_ref,
                            co_ref,
                            1,
                            q_len,
                            bt_ref.len(),
                            q_len,
                            seq_after,
                            0,
                            max_checkpoints_per_req,
                        )?;
                        r0.forward_selected_rows(
                            tok_ptr,
                            pos_ptr,
                            q_len,
                            &attn_batch,
                            &gdn_batch,
                            Some(&c0),
                            sr_ref,
                        )
                    });
                    let mut peer_handles = Vec::with_capacity(peer_ranks.len());
                    for r_peer in peer_ranks {
                        let cp = comm.clone();
                        peer_handles.push(sc.spawn(move || -> Result<()> {
                            let (tok_ptr, pos_ptr, attn_batch, gdn_batch) = r_peer
                                .upload_batch_meta(
                                    ft_ref,
                                    fp_ref,
                                    qs_ref,
                                    cb_ref,
                                    sa_ref,
                                    bt_ref,
                                    ws_ref,
                                    si_ref,
                                    so_ref,
                                    hi_ref,
                                    &[],
                                    cd_ref,
                                    co_ref,
                                    1,
                                    q_len,
                                    bt_ref.len(),
                                    q_len,
                                    seq_after,
                                    0,
                                    max_checkpoints_per_req,
                                )?;
                            let _ = r_peer.forward_selected_rows(
                                tok_ptr,
                                pos_ptr,
                                q_len,
                                &attn_batch,
                                &gdn_batch,
                                Some(&cp),
                                sr_ref,
                            )?;
                            Ok(())
                        }));
                    }
                    let mut peers = Vec::with_capacity(peer_handles.len());
                    for hp in peer_handles {
                        peers.push(hp.join().unwrap());
                    }
                    (h0.join().unwrap(), peers)
                });
                for pr in peer_results {
                    pr?;
                }
                Ok(res0?[0])
            }
        })();

        for rank in &mut self.ranks {
            rank.current_pos_plane_stride = 0;
            rank.pending_visual_splice = None;
        }
        run_res
    }

    /// Multimodal generation with `Qwen3VisionEncoder`, 3D `mRoPE`, visual embedding splice,
    /// and content-hashed `MambaRadixCache` prefix reuse.
    pub fn generate_multimodal_with_radix_cache(
        &mut self,
        prompt_token_ids: &[i32],
        images: &[crate::vision::PreprocessedImage],
        max_new_tokens: usize,
        prefill_chunk_size: usize,
        stop_on_eos: bool,
    ) -> Result<RadixBenchMetrics> {
        let params = crate::parser::SamplingParams {
            max_tokens: max_new_tokens,
            stop_token_ids: self.cfg.eos_token_ids.clone(),
            eos_token_ids: self.cfg.eos_token_ids.clone(),
            ..Default::default()
        };
        self.generate_multimodal_sampled_with_radix_cache(
            prompt_token_ids,
            images,
            params,
            None,
            prefill_chunk_size,
            stop_on_eos,
        )
    }

    pub fn generate_multimodal_sampled_with_radix_cache(
        &mut self,
        prompt_token_ids: &[i32],
        images: &[crate::vision::PreprocessedImage],
        params: crate::parser::SamplingParams,
        constraint: Option<crate::parser::TokenBitmaskConstraint>,
        prefill_chunk_size: usize,
        stop_on_eos: bool,
    ) -> Result<RadixBenchMetrics> {
        params
            .validate()
            .map_err(|_| loader_err("invalid sampling parameters"))?;
        if images.is_empty() {
            if stop_on_eos {
                return self.generate_sampled_with_radix_cache(
                    prompt_token_ids,
                    params,
                    constraint,
                    prefill_chunk_size,
                );
            }
            return self.generate_plain_sampled_with_radix_cache(
                prompt_token_ids,
                params,
                constraint,
                prefill_chunk_size,
                false,
            );
        }
        let max_new_tokens = params.max_tokens;
        let use_gpu_sampler = !params.is_plain_greedy(constraint.is_some());
        let mut sampler = crate::parser::SamplerV1::new(params, prompt_token_ids, constraint);
        if prompt_token_ids.is_empty() {
            return Err(loader_err(
                "generate_multimodal_with_radix_cache requires at least 1 prompt token",
            ));
        }
        self.ensure_vision_encoder()?;
        let (merge_size, image_token_id) = self
            .cfg
            .vision_config
            .as_ref()
            .map(|v| (v.spatial_merge_size, v.image_token_id))
            .unwrap_or((2, 248056));
        let grids: Vec<[usize; 3]> = images.iter().map(|img| img.grid_thw).collect();
        let (mrope_planes, mrope_delta) = crate::vision::compute_mrope_positions(
            prompt_token_ids,
            &grids,
            image_token_id,
            merge_size,
        );
        let hashed_prompt_ids = crate::vision::compute_multimodal_radix_key(
            prompt_token_ids,
            images,
            image_token_id,
            merge_size,
        );

        // Map each image_token_id position in prompt_token_ids to its visual token index `0..total_vis`.
        let mut vis_idx_at_pos = vec![-1i32; prompt_token_ids.len()];
        let mut total_vis_expected = 0usize;
        for (pos, &tok) in prompt_token_ids.iter().enumerate() {
            if tok == image_token_id {
                vis_idx_at_pos[pos] = total_vis_expected as i32;
                total_vis_expected += 1;
            }
        }

        let target_total_len = prompt_token_ids
            .len()
            .checked_add(max_new_tokens)
            .and_then(|v| v.checked_add(4))
            .ok_or_else(|| loader_err("token length overflow in multimodal generate"))?;
        let needed_pages = target_total_len.div_ceil(16);
        if needed_pages > self.ranks[0].kv_pool.num_pages {
            return Err(loader_err(
                "requested sequence pages exceed KV pool capacity",
            ));
        }
        let (active_slot, [pp0, pp1]) = self
            .alloc_three_state_slots()
            .ok_or_else(|| loader_err("StateBundlePool out of slots"))?;

        let lookup_len = hashed_prompt_ids.len().saturating_sub(1);
        let match_res = self.radix.match_prefix(&hashed_prompt_ids[..lookup_len]);
        let cached_prefix_tokens = match_res.matched_tokens;

        let mut req_state = ReqRadixState::new(16, 64, active_slot, [pp0, pp1]);
        req_state.restore_prefix_hit(&mut self.radix, &hashed_prompt_ids, &match_res);

        let step_res: Result<(f64, f64, Vec<i32>, Vec<crate::parser::StepLogprobs>)> = (|| {
            self.ensure_kv_pages_for_len(&mut req_state, target_total_len)?;

            let t0 = std::time::Instant::now();
            // Run ViT only if the uncached suffix contains any visual tokens.
            let uncached_has_vis = vis_idx_at_pos[cached_prefix_tokens..]
                .iter()
                .any(|&idx| idx >= 0);
            let mut all_vis_embeds_host: Vec<u16> = Vec::new();
            if uncached_has_vis {
                let r0 = &self.ranks[0];
                r0.device.make_current()?;
                let vis_dev0 = self
                    .vision_encoder
                    .as_ref()
                    .unwrap()
                    .encode_images(images, &r0.cublas, &r0.stream)?;
                let num_u16 = vis_dev0.len_bytes / 2;
                all_vis_embeds_host.resize(num_u16, 0u16);
                vis_dev0.copy_to_host(&mut all_vis_embeds_host, &r0.stream)?;
                r0.stream.synchronize()?;
            }

            let h_dim = self.cfg.hidden_size;
            let chunk_sz = prefill_chunk_size.max(1);
            let mut first_tok = 0i32;
            let mut cursor = cached_prefix_tokens;
            while cursor < prompt_token_ids.len() {
                let rem = prompt_token_ids.len() - cursor;
                let take = rem.min(chunk_sz);
                let slice_tokens = &prompt_token_ids[cursor..cursor + take];
                let slice_hashed = &hashed_prompt_ids[cursor..cursor + take];
                let planned = req_state.plan_checkpoints(take);
                let ckpt_pairs: Vec<(i32, i32)> = planned
                    .iter()
                    .map(|c| (c.trigger_offset_1based, c.dst_slot))
                    .collect();

                // Pack 3-plane mRoPE positions `[3, take]`.
                let mut pos_3d = Vec::with_capacity(3 * take);
                pos_3d.extend_from_slice(&mrope_planes[0][cursor..cursor + take]);
                pos_3d.extend_from_slice(&mrope_planes[1][cursor..cursor + take]);
                pos_3d.extend_from_slice(&mrope_planes[2][cursor..cursor + take]);

                // Collect visual tokens that fall inside `cursor..cursor + take`.
                let mut slice_vis_indices = Vec::new();
                let mut slice_vis_embeds = Vec::new();
                for local_idx in 0..take {
                    let v_idx = vis_idx_at_pos[cursor + local_idx];
                    if v_idx >= 0 {
                        let v = v_idx as usize;
                        if (v + 1) * h_dim <= all_vis_embeds_host.len() {
                            slice_vis_indices.push(local_idx as i32);
                            slice_vis_embeds.extend_from_slice(
                                &all_vis_embeds_host[v * h_dim..(v + 1) * h_dim],
                            );
                        }
                    }
                }

                let mut rank_vis_bufs: Vec<(DeviceBuffer, DeviceBuffer)> = Vec::new();
                let mut rank_vis_ptrs: Vec<(u64, u64, usize)> = Vec::new();
                if !slice_vis_indices.is_empty() {
                    for rank in &self.ranks {
                        rank.device.make_current()?;
                        let emb_dev =
                            DeviceBuffer::alloc(rank.device.device_id, slice_vis_embeds.len() * 2)?;
                        let idx_dev = DeviceBuffer::alloc(
                            rank.device.device_id,
                            slice_vis_indices.len() * 4,
                        )?;
                        emb_dev.copy_from_host(&slice_vis_embeds, &rank.stream)?;
                        idx_dev.copy_from_host(&slice_vis_indices, &rank.stream)?;
                        rank_vis_ptrs.push((
                            emb_dev.addr(),
                            idx_dev.addr(),
                            slice_vis_indices.len(),
                        ));
                        rank_vis_bufs.push((emb_dev, idx_dev));
                    }
                }

                let eng_slice = EngineBatchSlice {
                    tokens: slice_tokens,
                    ctx_len_before: cursor,
                    block_table: &req_state.kv_pages,
                    state_in_slot: req_state.current_state_in_slot(),
                    state_out_slot: req_state.active_state_slot,
                    has_initial_state: req_state.has_initial_state,
                    spec_slots: &[],
                    checkpoints: &ckpt_pairs,
                    emit_all_query_tokens: false,
                };
                first_tok = self.step_single_slice_with_mrope_and_vision(
                    &eng_slice,
                    &pos_3d,
                    take as i32,
                    if rank_vis_ptrs.is_empty() {
                        None
                    } else {
                        Some(&rank_vis_ptrs)
                    },
                )?;
                drop(rank_vis_bufs);
                req_state.commit_tokens(slice_hashed, &planned);
                cursor += take;
            }
            let ttft_ms = t0.elapsed().as_secs_f64() * 1000.0;

            let mut gen_ids = Vec::with_capacity(max_new_tokens);
            let mut token_logprobs = Vec::new();
            if max_new_tokens > 0 {
                if use_gpu_sampler {
                    let (tok, lp) = self.sample_single_gpu_step(&mut sampler)?;
                    first_tok = tok;
                    if let Some(lp) = lp {
                        token_logprobs.push(lp);
                    }
                } else {
                    sampler.commit_token(first_tok);
                }
                gen_ids.push(first_tok);
            }

            let t_dec = std::time::Instant::now();
            let mut cur_tok = first_tok;
            for _dec_step in 1..max_new_tokens {
                if stop_on_eos
                    && sampler.output_history.len() >= sampler.params.min_tokens
                    && sampler
                        .params
                        .is_stop_or_eos(cur_tok, &self.cfg.eos_token_ids)
                {
                    break;
                }
                let ctx_before = req_state.token_ids.len();
                let planned = req_state.plan_checkpoints(1);
                let ckpt_pairs: Vec<(i32, i32)> = planned
                    .iter()
                    .map(|c| (c.trigger_offset_1based, c.dst_slot))
                    .collect();
                let rope_pos = (ctx_before as i32) + mrope_delta;
                let eng_slice = EngineBatchSlice {
                    tokens: &[cur_tok],
                    ctx_len_before: ctx_before,
                    block_table: &req_state.kv_pages,
                    state_in_slot: req_state.current_state_in_slot(),
                    state_out_slot: req_state.active_state_slot,
                    has_initial_state: req_state.has_initial_state,
                    spec_slots: &[],
                    checkpoints: &ckpt_pairs,
                    emit_all_query_tokens: false,
                };
                let mut next_tok =
                    self.step_single_slice_with_mrope_and_vision(&eng_slice, &[rope_pos], 0, None)?;
                if use_gpu_sampler {
                    let (tok, lp) = self.sample_single_gpu_step(&mut sampler)?;
                    next_tok = tok;
                    if let Some(lp) = lp {
                        token_logprobs.push(lp);
                    }
                } else {
                    sampler.commit_token(next_tok);
                }
                req_state.commit_tokens(&[cur_tok], &planned);
                gen_ids.push(next_tok);
                cur_tok = next_tok;
            }
            let decode_ms = t_dec.elapsed().as_secs_f64() * 1000.0;
            Ok((ttft_ms, decode_ms, gen_ids, token_logprobs))
        })();

        let (ttft_ms, decode_ms, gen_ids, token_logprobs) = match step_res {
            Ok(v) => v,
            Err(e) => {
                let aborted = req_state.abort_and_release(&mut self.radix);
                self.free_kv_pages(&aborted.freed_kv_pages);
                self.free_state_slots(&aborted.freed_mamba_slots);
                return Err(e);
            }
        };
        let donation = req_state.donate_committed_checkpoint(&mut self.radix, None, true);
        self.free_kv_pages(&donation.freed_kv_pages);
        self.free_state_slots(&donation.freed_mamba_slots);

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
        let suffix_len = (prompt_token_ids.len() - cached_prefix_tokens).max(1);
        let prefill_tok_per_s = if ttft_ms > 0.0 {
            (suffix_len as f64) / (ttft_ms / 1000.0)
        } else {
            0.0
        };

        Ok(RadixBenchMetrics {
            prompt_len: prompt_token_ids.len(),
            cached_prefix_tokens,
            donated_seq_len: donation.donated_seq_len,
            gen_len: gen_ids.len(),
            ttft_ms,
            prefill_tok_per_s,
            decode_ms,
            decode_tok_per_s,
            itl_ms,
            generated_token_ids: gen_ids,
            token_logprobs,
        })
    }

    /// Runs warm prefix lookup in `MambaRadixCache`, chunked suffix prefill, and speculative
    /// verification rounds (`[anchor, draft_1, ..., draft_k]`) with transactional `StateTx` rollback
    /// and cross-boundary `C = 64` ping-pong checkpoint commit/rejection (**Decisive Gate Test #5**).
    ///
    /// When `keep_active_slot_alive == true`, `active_state_slot` is retained (not freed on finish)
    /// so caller tests can inspect the exact post-generation `conv_state` and `ssm_state` in VRAM.
    pub fn verify_speculative_with_radix<F>(
        &mut self,
        prompt_token_ids: &[i32],
        max_new_tokens: usize,
        prefill_chunk_size: usize,
        keep_active_slot_alive: bool,
        mut draft_proposer: F,
    ) -> Result<SpecVerifyMetrics>
    where
        F: FnMut(usize, i32) -> Vec<i32>,
    {
        if prompt_token_ids.is_empty() {
            return Err(loader_err("empty prompt in verify_speculative_with_radix"));
        }
        let target_total_len = prompt_token_ids
            .len()
            .checked_add(max_new_tokens)
            .and_then(|v| v.checked_add(32))
            .ok_or_else(|| loader_err("token length overflow in verify_speculative_with_radix"))?;
        let (active_slot, [pp0, pp1]) = self
            .alloc_three_state_slots()
            .ok_or_else(|| loader_err("StateBundlePool out of slots"))?;

        let lookup_len = prompt_token_ids.len().saturating_sub(1);
        let match_res = self.radix.match_prefix(&prompt_token_ids[..lookup_len]);
        let cached_prefix_tokens = match_res.matched_tokens;

        let mut req_state = ReqRadixState::new(16, 64, active_slot, [pp0, pp1]);
        req_state.restore_prefix_hit(&mut self.radix, prompt_token_ids, &match_res);

        let mut tx = StateTx::new(req_state.active_state_slot);
        let step_res: Result<(usize, Vec<usize>, Vec<i32>)> = (|| {
            self.ensure_kv_pages_for_len(&mut req_state, target_total_len)?;

            let chunk_sz = prefill_chunk_size.max(1);
            let mut anchor_tok = 0i32;
            let mut cursor = cached_prefix_tokens;
            while cursor < prompt_token_ids.len() {
                let rem = prompt_token_ids.len() - cursor;
                let take = rem.min(chunk_sz);
                let slice_tokens = &prompt_token_ids[cursor..cursor + take];
                let planned = req_state.plan_checkpoints(take);
                let ckpt_pairs: Vec<(i32, i32)> = planned
                    .iter()
                    .map(|c| (c.trigger_offset_1based, c.dst_slot))
                    .collect();
                let is_final_chunk = cursor + take >= prompt_token_ids.len();
                let batch_out = self.step_batch_slices_for_sampler_filtered(
                    &[EngineBatchSlice {
                        tokens: slice_tokens,
                        ctx_len_before: cursor,
                        block_table: &req_state.kv_pages,
                        state_in_slot: req_state.current_state_in_slot(),
                        state_out_slot: req_state.active_state_slot,
                        has_initial_state: req_state.has_initial_state,
                        spec_slots: &[],
                        checkpoints: &ckpt_pairs,
                        emit_all_query_tokens: false,
                    }],
                    None,
                    Some(&[is_final_chunk]),
                    false,
                )?;
                if is_final_chunk {
                    anchor_tok = batch_out[0][0];
                }
                req_state.commit_tokens(slice_tokens, &planned);
                cursor += take;
            }

            let mut gen_ids = Vec::with_capacity(max_new_tokens);
            if max_new_tokens > 0 {
                gen_ids.push(anchor_tok);
            }

            tx.commit_prefill(req_state.token_ids.len());

            let mut verify_rounds = 0usize;
            let mut accepted_counts_per_round = Vec::new();

            while gen_ids.len() < max_new_tokens {
                let round_idx = verify_rounds;
                verify_rounds += 1;
                let drafts = draft_proposer(round_idx, anchor_tok);
                let k = drafts.len();

                let mut verify_tokens = Vec::with_capacity(1 + k);
                verify_tokens.push(anchor_tok);
                verify_tokens.extend_from_slice(&drafts);

                let mut extra_slots = Vec::with_capacity(k);
                for _ in 0..k {
                    let Some(s) = self.alloc_state_slot() else {
                        self.free_state_slots(&extra_slots);
                        return Err(loader_err("StateBundlePool out of slots for spec verify"));
                    };
                    extra_slots.push(s);
                }
                let spec_schedule = if k > 0 {
                    tx.begin_spec_verify(extra_slots);
                    tx.verify_slot_schedule()
                } else {
                    vec![tx.committed_slot]
                };

                let ctx_before = req_state.token_ids.len();
                self.ensure_kv_pages_for_len(&mut req_state, ctx_before + verify_tokens.len() + 4)?;
                let planned: Vec<PlannedCheckpoint> =
                    req_state.plan_checkpoints(verify_tokens.len());
                let ckpt_pairs: Vec<(i32, i32)> = planned
                    .iter()
                    .map(|c| (c.trigger_offset_1based, c.dst_slot))
                    .collect();

                let batch_out = self.step_batch_slices(&[EngineBatchSlice {
                    tokens: &verify_tokens,
                    ctx_len_before: ctx_before,
                    block_table: &req_state.kv_pages,
                    state_in_slot: tx.committed_slot,
                    state_out_slot: *spec_schedule.last().unwrap(),
                    has_initial_state: true,
                    spec_slots: &spec_schedule,
                    checkpoints: &ckpt_pairs,
                    emit_all_query_tokens: true,
                }])?;
                let target_next_ids = &batch_out[0];

                // Greedy verification: accept `a` drafts where `drafts[j] == target_next_ids[j]`,
                // bounded by the remaining tokens needed to reach `max_new_tokens`.
                let max_drafts_to_accept = max_new_tokens.saturating_sub(1 + gen_ids.len()).min(k);
                let mut accepted_drafts = 0usize;
                for j in 0..max_drafts_to_accept {
                    if drafts[j] == target_next_ids[j] {
                        accepted_drafts += 1;
                    } else {
                        break;
                    }
                }
                let processed_accepted = 1 + accepted_drafts;
                accepted_counts_per_round.push(accepted_drafts);

                if k > 0 {
                    let freed = tx.resolve_spec_verify(processed_accepted);
                    self.free_state_slots(&freed);
                } else {
                    tx.kv_seq_len_before += 1;
                }
                req_state.active_state_slot = tx.committed_slot;

                // Commit only the `1 + accepted_drafts` verified input tokens into `ReqRadixState`!
                // Any planned checkpoint at `abs_seq_len > ctx_before + 1 + accepted_drafts` is discarded!
                req_state.commit_tokens(&verify_tokens[..processed_accepted], &planned);
                let rolled_back_pages = req_state.rollback_uncommitted_kv_pages();
                self.free_kv_pages(&rolled_back_pages);

                // Emit the accepted drafts + the target's bonus/recovered token `target_next_ids[accepted_drafts]`,
                // capping at `max_new_tokens`.
                for &d_tok in &drafts[..accepted_drafts] {
                    if gen_ids.len() < max_new_tokens {
                        gen_ids.push(d_tok);
                    }
                }
                let bonus_tok = target_next_ids[accepted_drafts];
                if gen_ids.len() < max_new_tokens {
                    gen_ids.push(bonus_tok);
                }
                anchor_tok = bonus_tok;
            }
            Ok((verify_rounds, accepted_counts_per_round, gen_ids))
        })();

        let (verify_rounds, accepted_counts_per_round, gen_ids) = match step_res {
            Ok(v) => v,
            Err(e) => {
                if !tx.spec_slots.is_empty() {
                    let extra_freed = tx.resolve_spec_verify(1);
                    self.free_state_slots(&extra_freed);
                }
                req_state.active_state_slot = tx.committed_slot;
                let aborted = req_state.abort_and_release(&mut self.radix);
                self.free_kv_pages(&aborted.freed_kv_pages);
                self.free_state_slots(&aborted.freed_mamba_slots);
                return Err(e);
            }
        };

        let final_active_slot = req_state.active_state_slot;
        if keep_active_slot_alive {
            // Prevent `donate_committed_checkpoint` from freeing `active_state_slot`.
            req_state.active_state_slot = -1;
        }
        let donation = req_state.donate_committed_checkpoint(&mut self.radix, None, true);
        self.free_kv_pages(&donation.freed_kv_pages);
        self.free_state_slots(&donation.freed_mamba_slots);

        Ok(SpecVerifyMetrics {
            prompt_len: prompt_token_ids.len(),
            cached_prefix_tokens,
            donated_seq_len: donation.donated_seq_len,
            verify_rounds,
            accepted_counts_per_round,
            active_state_slot: final_active_slot,
            generated_token_ids: gen_ids,
        })
    }

    /// Executes `prompts` concurrently through `SchedulerV1` + `MambaRadixCache` + `step_batch_slices`.
    pub fn generate_batch_scheduled(
        &mut self,
        prompts: &[Vec<i32>],
        max_new_tokens: usize,
        mut sched_cfg: SchedulerConfig,
    ) -> Result<Vec<(Vec<i32>, usize)>> {
        if sched_cfg.page_size != 16 {
            return Err(loader_err("SchedulerConfig.page_size must be 16"));
        }
        for rank in &self.ranks {
            let cap = rank.ordinary_prefill_token_capacity(sched_cfg.max_num_seqs)?;
            sched_cfg.clamp_for_token_capacity(cap);
        }
        let mut sched = SchedulerV1::new(sched_cfg);
        std::mem::swap(&mut sched.radix, &mut self.radix);

        let mut req_ids = Vec::with_capacity(prompts.len());
        for p in prompts {
            if p.is_empty() {
                std::mem::swap(&mut sched.radix, &mut self.radix);
                return Err(loader_err("empty prompt in generate_batch_scheduled"));
            }
            let id = sched.submit_request(p.clone(), max_new_tokens, Vec::new());
            req_ids.push(id);
        }

        let mut results_by_id: HashMap<u64, (Vec<i32>, usize)> = HashMap::new();
        let loop_res: Result<()> = (|| {
            loop {
                sched.admit_waiting(|radix| self.alloc_three_state_slots_from(radix));

                if sched.active.is_empty() && sched.waiting.is_empty() {
                    break;
                }

                for seq in &mut sched.active {
                    let total_len = seq
                        .prompt_tokens
                        .len()
                        .checked_add(seq.max_new_tokens)
                        .and_then(|v| v.checked_add(4))
                        .ok_or_else(|| loader_err("token length overflow in scheduler"))?;
                    let needed = total_len.div_ceil(16);
                    if !self.reserve_kv_pages_from(
                        &mut sched.radix,
                        &mut seq.radix_state.kv_pages,
                        needed,
                    ) {
                        return Err(loader_err("KvPool out of pages in scheduler"));
                    }
                }

                let step_slices = sched.schedule_step();
                if step_slices.is_empty() {
                    break;
                }

                let mut engine_slices = Vec::with_capacity(step_slices.len());
                let mut ckpt_vecs: Vec<Vec<(i32, i32)>> = Vec::with_capacity(step_slices.len());
                let mut block_vecs: Vec<Vec<i32>> = Vec::with_capacity(step_slices.len());
                for sl in &step_slices {
                    let seq = sched.active.iter().find(|s| s.req_id == sl.req_id).unwrap();
                    block_vecs.push(seq.radix_state.kv_pages.clone());
                    ckpt_vecs.push(
                        sl.planned_checkpoints
                            .iter()
                            .map(|c| (c.trigger_offset_1based, c.dst_slot))
                            .collect(),
                    );
                }
                for ((sl, blks), ckpts) in step_slices
                    .iter()
                    .zip(block_vecs.iter())
                    .zip(ckpt_vecs.iter())
                {
                    engine_slices.push(EngineBatchSlice {
                        tokens: &sl.query_tokens,
                        ctx_len_before: sl.ctx_len_before,
                        block_table: blks,
                        state_in_slot: sl.state_in_slot,
                        state_out_slot: sl.state_out_slot,
                        has_initial_state: sl.has_initial_state,
                        spec_slots: &[],
                        checkpoints: ckpts,
                        emit_all_query_tokens: false,
                    });
                }

                let batch_outputs = self.step_batch_slices(&engine_slices)?;

                for (sl, out_ids) in step_slices.iter().zip(batch_outputs.iter()) {
                    let emitted = if !sl.is_prefill_chunk || sl.is_final_prefill_chunk {
                        Some(out_ids[0])
                    } else {
                        None
                    };
                    let mut rep_slot = if sl.is_prefill_chunk {
                        self.alloc_state_slot_from(&mut sched.radix)
                    } else {
                        None
                    };
                    if let Some(don) = sched.commit_slice(sl, emitted, || rep_slot.take()) {
                        self.free_kv_pages(&don.freed_kv_pages);
                        self.free_state_slots(&don.freed_mamba_slots);
                    }
                    if let Some(unused_slot) = rep_slot {
                        self.free_state_slots(&[unused_slot]);
                    }
                }

                for done_seq in sched.take_finished() {
                    results_by_id.insert(
                        done_seq.req_id,
                        (done_seq.generated_tokens, done_seq.cached_prefix_tokens),
                    );
                }
            }
            Ok(())
        })();

        for seq in &mut sched.active {
            let aborted = seq.radix_state.abort_and_release(&mut sched.radix);
            self.free_kv_pages(&aborted.freed_kv_pages);
            self.free_state_slots(&aborted.freed_mamba_slots);
        }
        std::mem::swap(&mut sched.radix, &mut self.radix);
        loop_res?;

        Ok(req_ids
            .into_iter()
            .map(|id| results_by_id.remove(&id).unwrap_or_default())
            .collect())
    }
}
