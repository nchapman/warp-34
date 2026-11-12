//! Forward dumps and MoE golden-fixture replay.

use super::*;

#[cfg(feature = "cuda")]
impl Qwen3_5Engine {
    /// Runs a single diagnostic prefill pass (plus optional speculative drafter pass)
    /// and returns [`ForwardDump`] for `dump-diff`.
    pub fn dump_prompt(&mut self, prompt_token_ids: &[i32]) -> Result<ForwardDump> {
        self.reset_all_states()?;
        let needed_pages = (prompt_token_ids.len() + 4).div_ceil(16);
        if needed_pages > self.ranks[0].kv_pool.num_pages {
            return Err(loader_err("prompt exceeds allocated KV page capacity"));
        }
        let block_table: Vec<i32> = (0..needed_pages as i32).collect();
        let mut dump = ForwardDump::default();
        self.step_tokens(
            prompt_token_ids,
            0,
            &block_table,
            0,
            0,
            false,
            Some(&mut dump),
        )?;
        if let Some(mut proposer) = self.speculative.take() {
            let res = proposer.dump_step(self, prompt_token_ids, dump.greedy_token_id, &mut dump);
            self.speculative = Some(proposer);
            res?;
        }
        Ok(dump)
    }

    /// Runs a multimodal diagnostic prefill pass (including `Qwen3VisionEncoder`, 3D `mRoPE`,
    /// and visual embedding splice) and returns [`ForwardDump`] for `dump-diff`.
    pub fn dump_multimodal_prompt(
        &mut self,
        prompt_token_ids: &[i32],
        images: &[crate::vision::PreprocessedImage],
    ) -> Result<ForwardDump> {
        if images.is_empty() {
            return self.dump_prompt(prompt_token_ids);
        }
        self.reset_all_states()?;
        self.ensure_vision_encoder()?;
        let num_tokens = prompt_token_ids.len();
        if num_tokens == 0 {
            return Err(loader_err(
                "dump_multimodal_prompt requires at least 1 token",
            ));
        }
        let needed_pages = (num_tokens + 4).div_ceil(16);
        if needed_pages > self.ranks[0].kv_pool.num_pages {
            return Err(loader_err("prompt exceeds allocated KV page capacity"));
        }
        let block_table: Vec<i32> = (0..needed_pages as i32).collect();

        let (merge_size, image_token_id) = self
            .cfg
            .vision_config
            .as_ref()
            .map(|v| (v.spatial_merge_size, v.image_token_id))
            .unwrap_or((2, 248056));
        let grids: Vec<[usize; 3]> = images.iter().map(|img| img.grid_thw).collect();
        let (mrope_planes, _) = crate::vision::compute_mrope_positions(
            prompt_token_ids,
            &grids,
            image_token_id,
            merge_size,
        );
        let mut pos_3d = Vec::with_capacity(3 * num_tokens);
        pos_3d.extend_from_slice(&mrope_planes[0]);
        pos_3d.extend_from_slice(&mrope_planes[1]);
        pos_3d.extend_from_slice(&mrope_planes[2]);

        let h_dim = self.cfg.hidden_size;
        let mut vision_trace = std::collections::BTreeMap::new();
        let vis_dev0 = {
            let r0 = &self.ranks[0];
            r0.device.make_current()?;
            self.vision_encoder
                .as_ref()
                .unwrap()
                .encode_images_with_trace(images, &r0.cublas, &r0.stream, Some(&mut vision_trace))?
        };
        let num_u16 = vis_dev0.len_bytes / 2;
        let mut all_vis_embeds_host = vec![0u16; num_u16];
        {
            let r0 = &self.ranks[0];
            r0.device.make_current()?;
            vis_dev0.copy_to_host(&mut all_vis_embeds_host, &r0.stream)?;
            r0.stream.synchronize()?;
        }

        let mut dump = ForwardDump::default();
        for (k, v) in vision_trace {
            dump.tensors_f32.insert(k, v);
        }
        let vis_f32: Vec<f32> = all_vis_embeds_host
            .iter()
            .copied()
            .map(crate::cuda::f16_to_f32)
            .collect();
        dump.tensors_f32.insert(
            "visual_embeddings".to_string(),
            (vec![num_u16 / h_dim.max(1), h_dim], vis_f32),
        );

        let total_merged = num_u16 / h_dim.max(1);
        let mut vis_indices = Vec::new();
        let mut vis_embeds = Vec::new();
        let mut v_cursor = 0usize;
        for (pos, &tok) in prompt_token_ids.iter().enumerate() {
            if tok == image_token_id {
                if (v_cursor + 1) * h_dim > all_vis_embeds_host.len() {
                    return Err(loader_err(
                        "prompt has more <|image_pad|> tokens than encoded visual embeddings",
                    ));
                }
                vis_indices.push(pos as i32);
                vis_embeds.extend_from_slice(
                    &all_vis_embeds_host[v_cursor * h_dim..(v_cursor + 1) * h_dim],
                );
                v_cursor += 1;
            }
        }
        if v_cursor != total_merged {
            return Err(loader_err(
                "prompt <|image_pad|> count does not match encoded visual token count",
            ));
        }

        let mut rank_vis_bufs = Vec::new();
        let step_res = (|| -> Result<()> {
            for rank in &mut self.ranks {
                rank.device.make_current()?;
                rank.current_pos_plane_stride = num_tokens as i32;
                if !vis_indices.is_empty() {
                    let emb_dev = DeviceBuffer::alloc(rank.device.device_id, vis_embeds.len() * 2)?;
                    let idx_dev =
                        DeviceBuffer::alloc(rank.device.device_id, vis_indices.len() * 4)?;
                    emb_dev.copy_from_host(&vis_embeds, &rank.stream)?;
                    idx_dev.copy_from_host(&vis_indices, &rank.stream)?;
                    rank.pending_visual_splice =
                        Some((emb_dev.addr(), idx_dev.addr(), vis_indices.len()));
                    rank_vis_bufs.push((emb_dev, idx_dev));
                } else {
                    rank.pending_visual_splice = None;
                }
            }
            self.step_tokens_with_positions(
                prompt_token_ids,
                Some(&pos_3d),
                0,
                &block_table,
                0,
                0,
                false,
                Some(&mut dump),
            )?;
            Ok(())
        })();
        for rank in &mut self.ranks {
            rank.current_pos_plane_stride = 0;
            rank.pending_visual_splice = None;
        }
        drop(rank_vis_bufs);
        step_res?;

        if let Some(mut proposer) = self.speculative.take() {
            let res = proposer.dump_step(self, prompt_token_ids, dump.greedy_token_id, &mut dump);
            self.speculative = Some(proposer);
            res?;
        }
        Ok(dump)
    }

    /// Replays captured native `vLLM 0.31.0` layer-0 MoE inputs and routes against both
    /// GEMV (`scratch = 0`) and sparse MMA (`scratch = 8 MiB`), plus full `FfnBlockWeights::forward`
    /// (`warp-34` router + shared expert + routed MoE), enforcing `rel_l2 <= limit`.
    pub fn verify_moe_layer0_golden(
        &self,
        archive: &crate::loader::SafetensorsArchive,
        limit: f64,
    ) -> Result<MoeLayer0GoldenReport> {
        if self.tp_size != 1 {
            return Err(loader_err("verify_moe_layer0_golden requires tp_size == 1"));
        }
        if !(limit.is_finite() && limit > 0.0) {
            return Err(loader_err(
                "verify_moe_layer0_golden requires finite positive limit",
            ));
        }
        let r0 = &self.ranks[0];
        let ffn = &r0
            .layers
            .first()
            .ok_or_else(|| loader_err("model has no layers"))?
            .ffn;
        let FfnBlockWeights::Moe(moe) = ffn else {
            return Err(loader_err("layer 0 is not a sparse MoE block"));
        };
        let MoeExpertStorage::W4A16 {
            gate_up_packed_ptr,
            gate_up_scales_ptr,
            down_packed_ptr,
            down_scales_ptr,
            group_size,
        } = &moe.experts
        else {
            return Err(loader_err("layer 0 MoE experts are not W4A16"));
        };

        let h = self.cfg.hidden_size;
        let k = moe.top_k;
        let e = moe.num_experts;
        let in_shape = &archive
            .tensor_info("hidden_input")
            .map_err(|_| loader_err("missing hidden_input in MoE golden fixture"))?
            .shape;
        if in_shape.len() != 2 || in_shape[0] == 0 || in_shape[1] != h {
            return Err(loader_err(
                "invalid hidden_input shape in MoE golden fixture",
            ));
        }
        let m = in_shape[0];
        let check_shape =
            |name: &'static str, expected: &[usize], msg: &'static str| -> Result<()> {
                let got = &archive
                    .tensor_info(name)
                    .map_err(|_| loader_err(msg))?
                    .shape;
                if got.as_slice() != expected {
                    return Err(loader_err(msg));
                }
                Ok(())
            };
        check_shape("route_ids", &[m, k], "invalid route_ids in MoE golden")?;
        check_shape(
            "route_weights",
            &[m, k],
            "invalid route_weights in MoE golden",
        )?;
        check_shape("shared_raw", &[m, h], "invalid shared_raw in MoE golden")?;
        check_shape(
            "shared_gate_logits",
            &[m, 1],
            "invalid shared_gate_logits in MoE golden",
        )?;
        check_shape(
            "native_routed",
            &[m, h],
            "invalid native_routed in MoE golden",
        )?;
        check_shape(
            "native_combined",
            &[m, h],
            "invalid native_combined in MoE golden",
        )?;

        let hidden_f16 = archive
            .read_f16_bits("hidden_input")
            .map_err(|_| loader_err("failed reading hidden_input in MoE golden"))?;
        let route_ids = archive
            .read_i32("route_ids")
            .map_err(|_| loader_err("failed reading route_ids in MoE golden"))?;
        let route_weights = archive
            .read_f32("route_weights")
            .map_err(|_| loader_err("failed reading route_weights in MoE golden"))?;
        let shared_raw_f16 = archive
            .read_f16_bits("shared_raw")
            .map_err(|_| loader_err("failed reading shared_raw in MoE golden"))?;
        let shared_gate_f16 = archive
            .read_f16_bits("shared_gate_logits")
            .map_err(|_| loader_err("failed reading shared_gate_logits in MoE golden"))?;
        let native_routed = archive
            .read_f32("native_routed")
            .map_err(|_| loader_err("failed reading native_routed in MoE golden"))?;
        let native_combined = archive
            .read_f32("native_combined")
            .map_err(|_| loader_err("failed reading native_combined in MoE golden"))?;

        if route_ids.iter().any(|&id| id < 0 || (id as usize) >= e)
            || route_weights.iter().any(|w| !w.is_finite())
            || native_routed
                .iter()
                .chain(&native_combined)
                .any(|v| !v.is_finite())
        {
            return Err(loader_err(
                "invalid route IDs or non-finite values in MoE golden",
            ));
        }

        r0.device.make_current()?;
        let dev_id = r0.device.device_id;
        let stream = &r0.stream;
        let d_x = DeviceBuffer::alloc(dev_id, m * h * 2)?;
        let d_ids = DeviceBuffer::alloc(dev_id, m * k * 4)?;
        let d_tw = DeviceBuffer::alloc(dev_id, m * k * 4)?;
        let d_sh_raw = DeviceBuffer::alloc(dev_id, m * h * 2)?;
        let d_sh_gate = DeviceBuffer::alloc(dev_id, m * 2)?;
        let d_interm = DeviceBuffer::alloc(dev_id, m * k * moe.local_moe_interm.max(1) * 2)?;
        let d_out = DeviceBuffer::alloc(dev_id, m * h * 2)?;
        let ws_cap = 8 * 1024 * 1024usize;
        let d_ws = DeviceBuffer::alloc(dev_id, ws_cap)?;

        d_x.copy_from_host(&hidden_f16, stream)?;
        d_ids.copy_from_host(&route_ids, stream)?;
        d_tw.copy_from_host(&route_weights, stream)?;
        d_sh_raw.copy_from_host(&shared_raw_f16, stream)?;
        d_sh_gate.copy_from_host(&shared_gate_f16, stream)?;

        let rel_l2_f16 = |got_bits: &[u16], reference: &[f32]| -> Result<f64> {
            let mut err_sq = 0.0f64;
            let mut ref_sq = 0.0f64;
            for (&bits, &r) in got_bits.iter().zip(reference) {
                let g = crate::cuda::f16_to_f32(bits);
                if !g.is_finite() {
                    return Err(loader_err("non-finite output in MoE layer-0 replay"));
                }
                err_sq += (f64::from(g) - f64::from(r)).powi(2);
                ref_sq += f64::from(r).powi(2);
            }
            Ok((err_sq / ref_sq.max(1e-30)).sqrt())
        };

        let d_x_routed = if moe.routed_input_hadamard > 0 {
            let buf = DeviceBuffer::alloc(dev_id, m * h * 2)?;
            buf.copy_from_host(&hidden_f16, stream)?;
            let ex0 = stream.exec(0, 0);
            check_rc(
                // SAFETY: `buf` is a distinct `m * h` FP16 buffer on `dev_id`.
                unsafe {
                    ffi::w34_block_hadamard_inplace_f16(
                        &ex0,
                        buf.addr() as *mut c_void,
                        m as i32,
                        h as i32,
                        moe.routed_input_hadamard as i32,
                    )
                },
                "w34_block_hadamard_inplace_f16 (verify_moe_layer0_golden)",
            )?;
            Some(buf)
        } else {
            None
        };

        let run_same_route =
            |use_scratch: bool, with_shared: bool, reference: &[f32]| -> Result<f64> {
                let ex = if use_scratch {
                    stream.exec(d_ws.addr(), ws_cap)
                } else {
                    stream.exec(0, 0)
                };
                let (sh_ptr, sg_ptr) = if with_shared {
                    (
                        d_sh_raw.addr() as *const c_void,
                        d_sh_gate.addr() as *const c_void,
                    )
                } else {
                    (core::ptr::null(), core::ptr::null())
                };
                let x_routed_ptr =
                    d_x_routed.as_ref().map_or(d_x.addr(), |buf| buf.addr()) as *const c_void;
                check_rc(
                    // SAFETY: fixture shapes and route IDs were checked above;
                    // these distinct DeviceBuffers have M/H/K-sized extents and
                    // remain live through the synchronized output read. Weights
                    // belong to the loaded layer; absent shared inputs are null.
                    unsafe {
                        ffi::w34_moe_w4a16_mlp_ex_f16(
                            &ex,
                            x_routed_ptr,
                            *gate_up_packed_ptr as *const u32,
                            *gate_up_scales_ptr as *const c_void,
                            *down_packed_ptr as *const u32,
                            *down_scales_ptr as *const c_void,
                            d_ids.addr() as *const i32,
                            d_tw.addr() as *const f32,
                            sh_ptr,
                            sg_ptr,
                            d_interm.addr() as *mut c_void,
                            d_out.addr() as *mut c_void,
                            m as i32,
                            k as i32,
                            e as i32,
                            moe.local_moe_interm as i32,
                            h as i32,
                            *group_size as i32,
                            moe.routed_down_hadamard as i32,
                        )
                    },
                    "w34_moe_w4a16_mlp_ex_f16 (verify_moe_layer0_golden)",
                )?;
                let mut got = vec![0u16; m * h];
                d_out.copy_to_host(&mut got, stream)?;
                stream.synchronize()?;
                rel_l2_f16(&got, reference)
            };

        let routed_gemv_rel_l2 = run_same_route(false, false, &native_routed)?;
        let routed_sparse_mma_rel_l2 = run_same_route(true, false, &native_routed)?;
        let same_route_gemv_rel_l2 = run_same_route(false, true, &native_combined)?;
        let same_route_sparse_mma_rel_l2 = run_same_route(true, true, &native_combined)?;

        let n_sh = moe.local_shared_interm.max(8);
        let marlin_ints = (h.max(n_sh * 2).div_ceil(64) * 16).max(1024);
        let d_marlin = DeviceBuffer::alloc(dev_id, marlin_ints * 4)?;
        d_marlin.zero_async(stream)?;
        let d_gu = DeviceBuffer::alloc(dev_id, 64)?;
        let d_act = DeviceBuffer::alloc(dev_id, 64)?;
        let d_logits = DeviceBuffer::alloc(dev_id, m * e * 2)?;
        let d_w34_tw = DeviceBuffer::alloc(dev_id, m * k * 4)?;
        let d_w34_ids = DeviceBuffer::alloc(dev_id, m * k * 4)?;
        let d_sh_gu = DeviceBuffer::alloc(dev_id, m * n_sh * 2 * 2)?;
        let d_sh_act = DeviceBuffer::alloc(dev_id, m * n_sh * 2)?;
        let d_sh_out = DeviceBuffer::alloc(dev_id, m * h * 2)?;
        let d_sh_g = DeviceBuffer::alloc(dev_id, m * 2)?;
        let ex_opt = stream.exec(d_ws.addr(), ws_cap);
        ffn.forward(
            &ex_opt,
            &r0.cublas,
            d_marlin.addr(),
            marlin_ints,
            d_x.addr(),
            d_out.addr(),
            m,
            h,
            0,
            d_gu.addr(),
            d_act.addr(),
            d_logits.addr(),
            d_w34_tw.addr(),
            d_w34_ids.addr(),
            d_interm.addr(),
            d_sh_gu.addr(),
            d_sh_act.addr(),
            d_sh_out.addr(),
            d_sh_g.addr(),
        )?;
        let mut full_got = vec![0u16; m * h];
        let mut got_ids = vec![0i32; m * k];
        d_out.copy_to_host(&mut full_got, stream)?;
        d_w34_ids.copy_to_host(&mut got_ids, stream)?;
        stream.synchronize()?;
        let full_ffn_rel_l2 = rel_l2_f16(&full_got, &native_combined)?;
        let mut router_slot_matches = 0usize;
        for row in 0..m {
            let ref_row = &route_ids[row * k..(row + 1) * k];
            let got_row = &got_ids[row * k..(row + 1) * k];
            let mut used = vec![false; k];
            for &id in got_row {
                if let Some(pos) = (0..k).find(|&i| !used[i] && ref_row[i] == id) {
                    used[pos] = true;
                    router_slot_matches += 1;
                }
            }
        }
        let passed = routed_gemv_rel_l2 <= limit
            && routed_sparse_mma_rel_l2 <= limit
            && same_route_gemv_rel_l2 <= limit
            && same_route_sparse_mma_rel_l2 <= limit
            && full_ffn_rel_l2 <= limit
            && router_slot_matches == m * k;
        Ok(MoeLayer0GoldenReport {
            num_tokens: m,
            hidden_size: h,
            num_experts: e,
            top_k: k,
            limit,
            routed_gemv_rel_l2,
            routed_sparse_mma_rel_l2,
            same_route_gemv_rel_l2,
            same_route_sparse_mma_rel_l2,
            full_ffn_rel_l2,
            router_slot_matches,
            router_total_slots: m * k,
            passed,
        })
    }
}
