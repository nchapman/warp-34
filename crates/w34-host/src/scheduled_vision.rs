//! Request-local multimodal prefill for the dedicated engine worker.
//!
//! The scheduler owns admission, KV allocation, commits and resource release. This
//! module executes one chunk only; it never commits or locks scheduler state.

use std::sync::Arc;

use crate::cuda::{self, DeviceBuffer};
use crate::model::{EngineBatchSlice, Qwen3_5Engine, loader_err};
use crate::scheduler::{ScheduledReqSlice, SeqEntry};
use crate::vision::{self, PreprocessedImage};

pub struct ScheduledVision {
    pub radix_key: Vec<i32>,
    /// Add to the absolute token position during decode (single-plane mRoPE).
    pub position_delta: i32,
    prompt: Vec<i32>,
    mrope_planes: [Vec<i32>; 3],
    // Prompt position -> (image index, merged visual row within that image).
    visual_indices: Vec<Option<(usize, usize)>>,
    images: Vec<PreprocessedImage>,
    visual_counts: Vec<usize>,
    rank_device_embeds: Vec<Option<Arc<Vec<DeviceBuffer>>>>,
    hidden_size: usize,
    valid: bool,
}

impl ScheduledVision {
    /// Validate client-owned geometry before scheduler admission; no CUDA work occurs here.
    pub fn new(
        engine: &Qwen3_5Engine,
        prompt: &[i32],
        images: Vec<PreprocessedImage>,
    ) -> Result<Self, String> {
        Self::from_vision_config(
            engine.cfg.vision_config.as_ref(),
            engine.cfg.hidden_size,
            prompt,
            images,
        )
    }

    /// Validate client-owned geometry using `VisionConfig` and `hidden_size` without requiring
    /// `&Qwen3_5Engine` (callable off the CUDA worker thread during request ingress).
    pub fn from_vision_config(
        vision_config: Option<&crate::vision::VisionConfig>,
        hidden_size: usize,
        prompt: &[i32],
        images: Vec<PreprocessedImage>,
    ) -> Result<Self, String> {
        let config = vision_config.ok_or("model does not support images")?;
        let (merge, image_token) = (config.spatial_merge_size, config.image_token_id);
        let mut valid = merge > 0 && !prompt.is_empty() && prompt.len() <= i32::MAX as usize;
        let mut visual_counts = Vec::with_capacity(images.len());
        for image in &images {
            let [t, h, w] = image.grid_thw;
            let count = if merge > 0 && t > 0 && h > 0 && w > 0 && h % merge == 0 && w % merge == 0
            {
                t.checked_mul(h / merge)
                    .and_then(|n| n.checked_mul(w / merge))
            } else {
                None
            };
            valid &= count.is_some();
            let patch_elems = t
                .checked_mul(h)
                .and_then(|n| n.checked_mul(w))
                .and_then(|n| n.checked_mul(config.patch_input_dim()));
            valid &= patch_elems == Some(image.patches_f32.len());
            visual_counts.push(count.unwrap_or(0));
        }
        let expected = visual_counts
            .iter()
            .try_fold(0usize, |n, &c| n.checked_add(c));
        valid &= expected == Some(prompt.iter().filter(|&&t| t == image_token).count());
        let mut visual_indices = vec![None; prompt.len()];
        let mut positions = prompt
            .iter()
            .enumerate()
            .filter_map(|(p, &t)| (t == image_token).then_some(p));
        if valid {
            for (image, &count) in visual_counts.iter().enumerate() {
                let mut previous = None;
                for row in 0..count {
                    let pos = positions.next().unwrap();
                    if let Some(prev) = previous {
                        valid &= pos == prev + 1;
                    }
                    previous = Some(pos);
                    visual_indices[pos] = Some((image, row));
                }
            }
        }
        let (mrope_planes, position_delta, radix_key) = if valid {
            let grids: Vec<_> = images.iter().map(|i| i.grid_thw).collect();
            let (planes, delta) =
                vision::compute_mrope_positions(prompt, &grids, image_token, merge);
            (
                planes,
                delta,
                vision::compute_multimodal_radix_key(prompt, &images, image_token, merge),
            )
        } else {
            ([Vec::new(), Vec::new(), Vec::new()], 0, prompt.to_vec())
        };
        if !valid {
            return Err("invalid scheduled vision layout or patch dimensions".into());
        }
        Ok(Self {
            radix_key,
            position_delta,
            prompt: prompt.to_vec(),
            mrope_planes,
            visual_indices,
            rank_device_embeds: (0..images.len()).map(|_| None).collect(),
            images,
            visual_counts,
            hidden_size,
            valid,
        })
    }

    /// Executes exactly one nonempty prefill chunk, using scheduler-owned pages
    /// and checkpoint slots. The caller must commit the returned slice before
    /// calling again. Pass `Some(top1)` to `commit_slice` only for the final
    /// prefill chunk (or sample from the engine's final logits instead).
    ///
    /// Images entirely in the cached prefix are never encoded. An image crossing
    /// a chunk/cache boundary is encoded once, in full, on first use.
    #[cfg(test)]
    pub fn step_prefill(
        &mut self,
        engine: &mut Qwen3_5Engine,
        seq: &SeqEntry,
        chunk_size: usize,
    ) -> cuda::Result<(i32, ScheduledReqSlice)> {
        let needs_final_logits = seq.max_new_tokens > 0 || engine.ranks[0].lm_head.is_some();
        self.step_prefill_with_emit(engine, seq, chunk_size, needs_final_logits)
    }

    /// Executes one nonempty multimodal prefill chunk, running `lm_head` + greedy sampling
    /// only when `slice.is_final_prefill_chunk && needs_final_logits`.
    pub fn step_prefill_with_emit(
        &mut self,
        engine: &mut Qwen3_5Engine,
        seq: &SeqEntry,
        chunk_size: usize,
        needs_final_logits: bool,
    ) -> cuda::Result<(i32, ScheduledReqSlice)> {
        if !self.valid || engine.cfg.hidden_size != self.hidden_size {
            return Err(loader_err(
                "invalid scheduled vision layout or embedding dimension",
            ));
        }
        let start = seq.prefilled_tokens;
        if seq.finished
            || seq.resources_released
            || start >= self.prompt.len()
            || seq.prompt_tokens != self.prompt
            || seq.radix_key_tokens.len() != self.radix_key.len()
            || seq.radix_key_tokens[1..] != self.radix_key[1..]
            || seq.radix_state.token_ids.len() != start
        {
            return Err(loader_err(
                "scheduled vision sequence does not match prefill state",
            ));
        }
        let take = if let Some(bidir_end) = seq.prefix_bidir_len
            && start < bidir_end
        {
            let need = bidir_end.min(self.prompt.len()) - start;
            if need > chunk_size.max(1) {
                return Err(loader_err(
                    "bidirectional prefix exceeds chunk_prefill_size in multimodal step",
                ));
            }
            need
        } else {
            let mut max_take = (self.prompt.len() - start).min(chunk_size.max(1));
            if let Some(branch_end) = seq.mamba_branching_seqlen {
                let aligned_branch = (branch_end / 64) * 64;
                if aligned_branch > start && aligned_branch < self.prompt.len() {
                    max_take = max_take.min(aligned_branch - start);
                }
            }
            max_take
        };
        let end = start + take;
        let planned = if seq.prefix_bidir_len.is_some() {
            Vec::new()
        } else {
            seq.radix_state.plan_checkpoints(take)
        };
        let slice = ScheduledReqSlice {
            req_id: seq.req_id,
            is_prefill_chunk: true,
            is_final_prefill_chunk: end == self.prompt.len(),
            ctx_len_before: start,
            query_tokens: self.prompt[start..end].to_vec(),
            state_in_slot: seq.radix_state.current_state_in_slot(),
            state_out_slot: seq.radix_state.active_state_slot,
            has_initial_state: seq.radix_state.has_initial_state,
            planned_checkpoints: planned,
            lora_id: seq.lora_id,
        };
        let mut positions = Vec::with_capacity(3 * take);
        for plane in &self.mrope_planes {
            positions.extend_from_slice(&plane[start..end]);
        }
        let mut indices = Vec::<i32>::new();
        // Contiguous spans `(image_idx, start_row, row_count)` within this prefill chunk.
        let mut spans = Vec::<(usize, usize, usize)>::new();
        for (local, visual) in self.visual_indices[start..end].iter().enumerate() {
            let Some((image, row)) = *visual else {
                continue;
            };
            if self.rank_device_embeds[image].is_none() {
                const MAX_VISION_EMBED_CACHE_ENTRIES: usize = 8;
                const MAX_VISION_EMBED_CACHE_BYTES: usize = 64 * 1024 * 1024;
                let cache_key = (
                    self.images[image].content_hash,
                    self.images[image].grid_thw,
                    self.images[image].patches_f32.len(),
                );
                if let Some(pos) = engine
                    .vision_embed_cache
                    .iter()
                    .position(|(k, _)| *k == cache_key)
                {
                    if pos + 1 != engine.vision_embed_cache.len() {
                        let hit = engine.vision_embed_cache.remove(pos);
                        engine.vision_embed_cache.push(hit);
                    }
                    self.rank_device_embeds[image] =
                        Some(engine.vision_embed_cache.last().unwrap().1.clone());
                } else {
                    engine.ensure_vision_encoder()?;
                    let rank0 = &engine.ranks[0];
                    rank0.device.make_current()?;
                    let enc = engine
                        .vision_encoder
                        .as_ref()
                        .ok_or_else(|| loader_err("vision encoder unavailable"))?;
                    let dev0 = enc.encode_images(
                        std::slice::from_ref(&self.images[image]),
                        &rank0.cublas,
                        &rank0.stream,
                    )?;
                    let count = self.visual_counts[image]
                        .checked_mul(self.hidden_size)
                        .ok_or_else(|| loader_err("visual embedding size overflow"))?;
                    if count.checked_mul(2) != Some(dev0.len_bytes) {
                        return Err(loader_err("visual token embedding count mismatch"));
                    }
                    let mut rank_bufs = Vec::with_capacity(engine.ranks.len());
                    if engine.ranks.len() == 1 {
                        rank_bufs.push(dev0);
                    } else {
                        let peers: Vec<(i32, &cuda::CudaStream)> = engine.ranks[1..]
                            .iter()
                            .map(|r| (r.device.device_id, &r.stream))
                            .collect();
                        let peer_bufs =
                            enc.replicate_to_peers_pinned(&dev0, &rank0.stream, &peers)?;
                        rank_bufs.push(dev0);
                        rank_bufs.extend(peer_bufs);
                    }
                    let shared = Arc::new(rank_bufs);
                    let entry_bytes = shared[0].len_bytes;
                    if entry_bytes <= MAX_VISION_EMBED_CACHE_BYTES {
                        while !engine.vision_embed_cache.is_empty()
                            && (engine.vision_embed_cache.len() >= MAX_VISION_EMBED_CACHE_ENTRIES
                                || engine
                                    .vision_embed_cache
                                    .iter()
                                    .map(|(_, v)| v[0].len_bytes)
                                    .sum::<usize>()
                                    + entry_bytes
                                    > MAX_VISION_EMBED_CACHE_BYTES)
                        {
                            engine.vision_embed_cache.remove(0);
                        }
                        engine.vision_embed_cache.push((cache_key, shared.clone()));
                    }
                    self.rank_device_embeds[image] = Some(shared);
                }
            }
            if row >= self.visual_counts[image] {
                return Err(loader_err("missing visual token embedding"));
            }
            indices.push(local as i32);
            if let Some(last) = spans.last_mut()
                && last.0 == image
                && last.1 + last.2 == row
            {
                last.2 += 1;
            } else {
                spans.push((image, row, 1));
            }
        }
        let mut pointers = Vec::new();
        let row_bytes = self.hidden_size * 2;
        for (rank_idx, rank) in engine.ranks.iter_mut().enumerate() {
            if indices.is_empty() {
                break;
            }
            rank.device.make_current()?;
            let idx_bytes = indices.len() * 4;
            if rank
                .vision_splice_idx
                .as_ref()
                .is_none_or(|b| b.len_bytes < idx_bytes)
            {
                rank.vision_splice_idx =
                    Some(DeviceBuffer::alloc(rank.device.device_id, idx_bytes)?);
            }
            let idx_buf = rank.vision_splice_idx.as_ref().unwrap();
            idx_buf.copy_from_host(&indices, &rank.stream)?;
            let idx_addr = idx_buf.addr();
            let emb_addr = if spans.len() == 1 {
                let (image, start_row, _) = spans[0];
                let dev_buf = &self.rank_device_embeds[image].as_ref().unwrap()[rank_idx];
                dev_buf.addr() + (start_row * row_bytes) as u64
            } else {
                let emb_bytes = indices.len() * row_bytes;
                if rank
                    .vision_splice_emb
                    .as_ref()
                    .is_none_or(|b| b.len_bytes < emb_bytes)
                {
                    rank.vision_splice_emb =
                        Some(DeviceBuffer::alloc(rank.device.device_id, emb_bytes)?);
                }
                let emb = rank.vision_splice_emb.as_ref().unwrap();
                let mut dst_off = 0usize;
                for &(image, start_row, row_count) in &spans {
                    let dev_buf = &self.rank_device_embeds[image].as_ref().unwrap()[rank_idx];
                    let src_ptr = dev_buf.addr() + (start_row * row_bytes) as u64;
                    let dst_ptr = emb.addr() + (dst_off * row_bytes) as u64;
                    let copy_bytes = row_count * row_bytes;
                    cuda::check_rc(
                        // SAFETY: `src_ptr` and `dst_ptr` are valid non-overlapping device spans of `copy_bytes`.
                        unsafe {
                            w34_kernels::ffi::w34_memcpy_d2d_async(
                                dst_ptr as *mut core::ffi::c_void,
                                src_ptr as *const core::ffi::c_void,
                                copy_bytes,
                                rank.stream.handle,
                            )
                        },
                        "ScheduledVision multi-image D2D splice",
                    )?;
                    dst_off += row_count;
                }
                emb.addr()
            };
            pointers.push((emb_addr, idx_addr, indices.len()));
        }
        let checkpoints: Vec<_> = slice
            .planned_checkpoints
            .iter()
            .map(|c| (c.trigger_offset_1based, c.dst_slot))
            .collect();
        let engine_slice = EngineBatchSlice {
            tokens: &slice.query_tokens,
            ctx_len_before: start,
            block_table: &seq.radix_state.kv_pages,
            state_in_slot: slice.state_in_slot,
            state_out_slot: slice.state_out_slot,
            has_initial_state: slice.has_initial_state,
            spec_slots: &[],
            checkpoints: &checkpoints,
            emit_all_query_tokens: false,
        };
        let noncausal_prefix = seq
            .prefix_bidir_len
            .is_some_and(|bidir_end| end <= bidir_end);
        let emit_last_row = slice.is_final_prefill_chunk && needs_final_logits;
        engine.set_slice_lora_ids(&[seq.lora_id]);
        let result = engine.step_single_slice_with_mrope_and_vision_bidir(
            &engine_slice,
            &positions,
            take as i32,
            if pointers.is_empty() {
                None
            } else {
                Some(&pointers)
            },
            noncausal_prefix,
            emit_last_row,
        );
        engine.clear_slice_lora_ids();
        let mut sync_result = Ok(());
        for rank in &engine.ranks {
            let synced = rank
                .device
                .make_current()
                .and_then(|_| rank.stream.synchronize());
            if sync_result.is_ok() {
                sync_result = synced;
            }
        }
        if slice.is_final_prefill_chunk {
            self.rank_device_embeds.clear();
        }
        let token = result?;
        sync_result?;
        Ok((token, slice))
    }
}
