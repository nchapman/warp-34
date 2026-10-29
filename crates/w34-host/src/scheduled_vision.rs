//! Request-local multimodal prefill for the dedicated engine worker.
//!
//! The scheduler owns admission, KV allocation, commits and resource release. This
//! module executes one chunk only; it never commits or locks scheduler state.

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
    host_embeds: Vec<Option<Vec<u16>>>,
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
        let config = engine
            .cfg
            .vision_config
            .as_ref()
            .ok_or("model does not support images")?;
        let (merge, image_token) = engine
            .cfg
            .vision_config
            .as_ref()
            .map(|v| (v.spatial_merge_size, v.image_token_id))
            .unwrap_or((2, 248056));
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
            host_embeds: vec![None; images.len()],
            images,
            visual_counts,
            hidden_size: engine.cfg.hidden_size,
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
    pub fn step_prefill(
        &mut self,
        engine: &mut Qwen3_5Engine,
        seq: &SeqEntry,
        chunk_size: usize,
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
            || seq.radix_key_tokens != self.radix_key
            || seq.radix_state.token_ids.len() != start
        {
            return Err(loader_err(
                "scheduled vision sequence does not match prefill state",
            ));
        }
        let take = (self.prompt.len() - start).min(chunk_size.max(1));
        let end = start + take;
        let planned = seq.radix_state.plan_checkpoints(take);
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
        };
        let mut positions = Vec::with_capacity(3 * take);
        for plane in &self.mrope_planes {
            positions.extend_from_slice(&plane[start..end]);
        }
        let mut indices = Vec::<i32>::new();
        let mut embeds = Vec::<u16>::new();
        for (local, visual) in self.visual_indices[start..end].iter().enumerate() {
            let Some((image, row)) = *visual else {
                continue;
            };
            if self.host_embeds[image].is_none() {
                engine.ensure_vision_encoder()?;
                let rank = &engine.ranks[0];
                rank.device.make_current()?;
                let device = engine
                    .vision_encoder
                    .as_ref()
                    .ok_or_else(|| loader_err("vision encoder unavailable"))?
                    .encode_images(
                        std::slice::from_ref(&self.images[image]),
                        &rank.cublas,
                        &rank.stream,
                    )?;
                let count = self.visual_counts[image]
                    .checked_mul(self.hidden_size)
                    .ok_or_else(|| loader_err("visual embedding size overflow"))?;
                if count.checked_mul(2) != Some(device.len_bytes) {
                    return Err(loader_err("visual token embedding count mismatch"));
                }
                let mut host = vec![0u16; count];
                device.copy_to_host(&mut host, &rank.stream)?;
                rank.stream.synchronize()?;
                self.host_embeds[image] = Some(host);
            }
            let host = self.host_embeds[image].as_ref().unwrap();
            let offset = row * self.hidden_size;
            let embedding = host
                .get(offset..offset + self.hidden_size)
                .ok_or_else(|| loader_err("missing visual token embedding"))?;
            indices.push(local as i32);
            embeds.extend_from_slice(embedding);
        }
        // Keep uploads alive through the forward and stream completion, including
        // error paths. No device pointer is retained in request state.
        let mut buffers = Vec::new();
        let mut pointers = Vec::new();
        for rank in &engine.ranks {
            if indices.is_empty() {
                break;
            }
            rank.device.make_current()?;
            let emb = DeviceBuffer::alloc(rank.device.device_id, embeds.len() * 2)?;
            let idx = DeviceBuffer::alloc(rank.device.device_id, indices.len() * 4)?;
            emb.copy_from_host(&embeds, &rank.stream)?;
            idx.copy_from_host(&indices, &rank.stream)?;
            pointers.push((emb.addr(), idx.addr(), indices.len()));
            buffers.push((emb, idx));
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
        let result = engine.step_single_slice_with_mrope_and_vision(
            &engine_slice,
            &positions,
            take as i32,
            if pointers.is_empty() {
                None
            } else {
                Some(&pointers)
            },
        );
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
        drop(buffers);
        let token = result?;
        sync_result?;
        Ok((token, slice))
    }
}
