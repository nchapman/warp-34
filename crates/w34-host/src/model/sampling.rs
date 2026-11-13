//! GPU-resident sampling and speculative acceptance.

use super::*;

#[cfg(feature = "cuda")]
pub(crate) struct PreparedSpecSamplerRows {
    base_sampler: crate::parser::SamplerV1,
    after_draft_states: Vec<(crate::parser::SamplerV1, usize, usize)>,
    pub row_cfgs: Vec<ffi::W34SamplerRowCfg>,
    pub bitmask_words: Vec<u32>,
    pub sparse_entries: Vec<ffi::W34SamplerSparseEntry>,
    recovery_u: f32,
}

#[cfg(feature = "cuda")]
impl PreparedSpecSamplerRows {
    /// Validates `drafts` against `sampler`'s active constraint and builds per-draft
    /// `W34SamplerRowCfg`, bitmask words, and sparse entries in a single grammar walk
    /// (eliminating duplicate sampler clones and `31 KiB` bitmask refreshes).
    pub fn prepare_drafts(
        vocab_size: usize,
        eos_token_ids: &[i32],
        sampler: &crate::parser::SamplerV1,
        drafts: &mut Vec<i32>,
    ) -> Result<Self> {
        let mut hypothetical = sampler.clone();
        let pos = sampler.output_history.len();
        let recovery_u =
            crate::parser::SamplerV1::uniform_f32(sampler.params.seed ^ 0xe7037ed1a0b428db, pos);
        let mut row_cfgs = Vec::with_capacity(drafts.len() + 1);
        let mut bitmask_words = Vec::new();
        let mut sparse_entries = Vec::new();
        let mut after_draft_states = Vec::with_capacity(drafts.len());
        let mut valid = 0usize;

        for (row, &draft_id) in drafts.iter().enumerate() {
            if hypothetical.materialize_pending_constraint().is_err()
                || hypothetical
                    .constraint
                    .as_ref()
                    .is_some_and(|c| c.is_terminal_eos() || !c.is_token_allowed(draft_id))
            {
                break;
            }
            let mask_before = bitmask_words.len();
            let sparse_before = sparse_entries.len();
            let accept_u = crate::parser::SamplerV1::uniform_f32(
                sampler.params.seed ^ 0xa0761d6478bd642f,
                pos + row,
            );
            let cfg = hypothetical
                .append_gpu_row(
                    vocab_size,
                    eos_token_ids,
                    accept_u,
                    recovery_u,
                    draft_id,
                    false,
                    &mut bitmask_words,
                    &mut sparse_entries,
                )
                .map_err(|_| loader_err("invalid speculative GPU sampler row"))?;
            let mut next = hypothetical.clone();
            next.try_commit_token(draft_id)
                .map_err(|_| loader_err("draft grammar commit failed"))?;
            if next
                .constraint
                .as_ref()
                .is_some_and(|c| c.is_terminal_eos())
            {
                bitmask_words.truncate(mask_before);
                sparse_entries.truncate(sparse_before);
                break;
            }
            row_cfgs.push(cfg);
            after_draft_states.push((next.clone(), bitmask_words.len(), sparse_entries.len()));
            hypothetical = next;
            valid += 1;
        }
        drafts.truncate(valid);
        Ok(Self {
            base_sampler: sampler.clone(),
            after_draft_states,
            row_cfgs,
            bitmask_words,
            sparse_entries,
            recovery_u,
        })
    }

    /// Truncates the prepared draft prefix in `O(1)` when fewer GDN state slots (`k`) are available.
    pub fn truncate_drafts(&mut self, k: usize) {
        if k >= self.row_cfgs.len() {
            return;
        }
        self.row_cfgs.truncate(k);
        if k == 0 {
            self.bitmask_words.clear();
            self.sparse_entries.clear();
            self.after_draft_states.clear();
        } else {
            let (_, mask_end, sparse_end) = self.after_draft_states[k - 1];
            self.bitmask_words.truncate(mask_end);
            self.sparse_entries.truncate(sparse_end);
            self.after_draft_states.truncate(k);
        }
    }

    /// Appends the bonus verification row (`k`-th row, after target forward has been enqueued).
    pub fn append_bonus_into(
        mut self,
        vocab_size: usize,
        eos_token_ids: &[i32],
        out_cfgs: &mut Vec<ffi::W34SamplerRowCfg>,
        out_masks: &mut Vec<u32>,
        out_sparse: &mut Vec<ffi::W34SamplerSparseEntry>,
    ) -> Result<()> {
        let bonus_state = match self.after_draft_states.pop() {
            Some((st, _, _)) => st,
            None => self.base_sampler,
        };
        let word_base = out_masks.len() as i32;
        let sparse_base = out_sparse.len() as i32;
        for cfg in &mut self.row_cfgs {
            if cfg.has_bitmask != 0 && cfg.row_bitmask_word_offset >= 0 {
                cfg.row_bitmask_word_offset += word_base;
            }
            if cfg.sparse_count > 0 && cfg.sparse_offset >= 0 {
                cfg.sparse_offset += sparse_base;
            }
        }
        out_cfgs.extend_from_slice(&self.row_cfgs);
        out_masks.extend_from_slice(&self.bitmask_words);
        out_sparse.extend_from_slice(&self.sparse_entries);
        let bonus_cfg = bonus_state
            .append_gpu_row(
                vocab_size,
                eos_token_ids,
                0.0,
                self.recovery_u,
                -1,
                true,
                out_masks,
                out_sparse,
            )
            .map_err(|_| loader_err("invalid speculative bonus GPU sampler row"))?;
        out_cfgs.push(bonus_cfg);
        Ok(())
    }
}

#[cfg(feature = "cuda")]
fn row_sparse_valid(
    cfg: &ffi::W34SamplerRowCfg,
    sparse_entries: &[ffi::W34SamplerSparseEntry],
    allow_bans: bool,
) -> bool {
    if cfg.sparse_count < 0 || cfg.sparse_count > 32 {
        return false;
    }
    if cfg.sparse_count == 0 {
        return true;
    }
    if cfg.sparse_offset < 0 {
        return false;
    }
    let start = cfg.sparse_offset as usize;
    let end = start.saturating_add(cfg.sparse_count as usize);
    end <= sparse_entries.len()
        && (allow_bans || sparse_entries[start..end].iter().all(|e| e.banned == 0))
}

#[cfg(feature = "cuda")]
fn stage_sampler_inputs_on_rank(
    r: &mut Qwen3_5ModelRank,
    row_cfgs: &[ffi::W34SamplerRowCfg],
    bitmask_words: &[u32],
    sparse_entries: &[ffi::W34SamplerSparseEntry],
    out_bytes: usize,
) -> Result<(u64, u64, u64, usize, bool)> {
    r.device.make_current()?;
    r.batch_meta_arena.reset();
    let cfg_bytes = core::mem::size_of_val(row_cfgs);
    let sparse_bytes = core::mem::size_of_val(sparse_entries);
    let mask_bytes = core::mem::size_of_val(bitmask_words);
    let cfg_off = 0usize;
    let sparse_off = (cfg_bytes + 63) & !63;
    let mask_off = (sparse_off + sparse_bytes + 63) & !63;
    let out_off = (mask_off + mask_bytes + 63) & !63;
    let use_pinned = out_off + out_bytes <= BatchedDecodeGraphIo::PINNED_SAMPLER_BYTES;
    let (cfg_dev, mask_dev, sparse_dev) = if use_pinned {
        if r.batched_decode_io
            .pinned_sampler_in_flight
            .swap(false, std::sync::atomic::Ordering::AcqRel)
        {
            r.stream.synchronize()?;
        }
        // SAFETY: the IO allocation includes the slot-0 sampler region;
        // `use_pinned` bounds all of this call's input/output offsets.
        let pbase = unsafe {
            r.batched_decode_io
                .pinned_staging
                .host_ptr
                .cast::<u8>()
                .add(BatchedDecodeGraphIo::PINNED_SAMPLER_OFFSET)
        };
        // SAFETY: cfg/sparse/mask byte ranges are disjoint and fit the checked
        // pinned region. The input slices are separate host allocations; cfg
        // and sparse ABI structs have initialized scalar fields without padding.
        unsafe {
            core::ptr::copy_nonoverlapping(
                row_cfgs.as_ptr().cast::<u8>(),
                pbase.add(cfg_off),
                cfg_bytes,
            );
            if sparse_bytes > 0 {
                core::ptr::copy_nonoverlapping(
                    sparse_entries.as_ptr().cast::<u8>(),
                    pbase.add(sparse_off),
                    sparse_bytes,
                );
            }
            if mask_bytes > 0 {
                core::ptr::copy_nonoverlapping(
                    bitmask_words.as_ptr().cast::<u8>(),
                    pbase.add(mask_off),
                    mask_bytes,
                );
            }
        }
        let h2d_bytes = if mask_bytes > 0 {
            mask_off + mask_bytes
        } else if sparse_bytes > 0 {
            sparse_off + sparse_bytes
        } else {
            cfg_bytes
        };
        let dev_base = r.batch_meta_arena.alloc_aligned(h2d_bytes.max(64))?;
        r.batched_decode_io
            .pinned_sampler_in_flight
            .store(true, std::sync::atomic::Ordering::Release);
        check_rc(
            // SAFETY: dev_base reserves h2d_bytes and the pinned owner remains
            // in the rank until stream synchronization below.
            unsafe {
                ffi::w34_memcpy_h2d_async(
                    dev_base as *mut c_void,
                    pbase.cast(),
                    h2d_bytes,
                    r.stream.handle,
                )
            },
            "upload pinned GPU sampler inputs",
        )?;
        let mk_dev = if mask_bytes > 0 {
            dev_base + mask_off as u64
        } else {
            0
        };
        let sp_dev = if sparse_bytes > 0 {
            dev_base + sparse_off as u64
        } else {
            0
        };
        (dev_base, mk_dev, sp_dev)
    } else {
        let c_dev = r.batch_meta_arena.upload_slice(row_cfgs, &r.stream)?;
        let m_dev = if bitmask_words.is_empty() {
            0
        } else {
            r.batch_meta_arena.upload_slice(bitmask_words, &r.stream)?
        };
        let s_dev = if sparse_entries.is_empty() {
            0
        } else {
            r.batch_meta_arena.upload_slice(sparse_entries, &r.stream)?
        };
        (c_dev, m_dev, s_dev)
    };
    Ok((cfg_dev, mask_dev, sparse_dev, out_off, use_pinned))
}

#[cfg(feature = "cuda")]
impl Qwen3_5Engine {
    fn try_run_distributed_fast_greedy_rows(
        &mut self,
        row_cfgs: &[ffi::W34SamplerRowCfg],
        bitmask_words: &[u32],
        sparse_entries: &[ffi::W34SamplerSparseEntry],
    ) -> Result<Option<Vec<ffi::W34SamplerRowOut>>> {
        let num_rows = row_cfgs.len();
        let vocab_size = self.cfg.vocab_size;
        let words_per_row = vocab_size.div_ceil(32);
        let eligible = self.tp_size > 1
            && vocab_size >= 1024
            && self.ranks.iter().all(|r| r.local_vocab_size >= 512)
            && self
                .tp2_comm
                .as_ref()
                .is_some_and(|c| num_rows * 8 <= c.max_stage_bytes_per_epoch)
            && row_cfgs.iter().all(|c| {
                let is_greedy = c.temperature <= 1e-6 || c.top_k == 1;
                let unmasked_ok = (c.has_bitmask == 0 || c.row_bitmask_word_offset < 0)
                    && row_sparse_valid(c, sparse_entries, true)
                    && c.top_logprobs_k < 0;
                let masked_ok = !bitmask_words.is_empty()
                    && c.has_bitmask != 0
                    && c.row_bitmask_word_offset >= 0
                    && (c.row_bitmask_word_offset as usize).saturating_add(words_per_row)
                        <= bitmask_words.len()
                    && row_sparse_valid(c, sparse_entries, false)
                    && c.top_logprobs_k < 0;
                is_greedy && (unmasked_ok || masked_ok)
            });
        if !eligible {
            return Ok(None);
        }
        for r in &self.ranks {
            if r.last_logits_ptr == 0 || r.last_logits_rows < num_rows {
                return Err(loader_err(
                    "missing rank device logits for distributed GPU sampler",
                ));
            }
        }
        let comm = self.tp2_comm.as_ref().unwrap().clone();
        let ids_bytes = num_rows * core::mem::size_of::<i32>();
        let mut rank0_global_ids_dev = 0u64;
        let mut rank0_out_off = 0usize;
        let mut rank0_use_pinned = false;
        for (rank_idx, r) in self.ranks.iter_mut().enumerate() {
            let (ws_ptr, ws_cap) = r.sampler_workspace_around_last_logits(num_rows)?;
            let (cfg_dev, mask_dev, sparse_dev, out_off, use_pinned) =
                stage_sampler_inputs_on_rank(
                    r,
                    row_cfgs,
                    bitmask_words,
                    sparse_entries,
                    ids_bytes,
                )?;
            let local_vals = r.batch_meta_arena.alloc_aligned(num_rows * 4)?;
            let local_ids = r.batch_meta_arena.alloc_aligned(num_rows * 4)?;
            let global_vals = r.batch_meta_arena.alloc_aligned(num_rows * 4)?;
            let global_ids = r.batch_meta_arena.alloc_aligned(num_rows * 4)?;
            let ex_s = r.stream.exec(ws_ptr, ws_cap);
            let mask_ptr = if mask_dev == 0 {
                core::ptr::null()
            } else {
                mask_dev as *const u32
            };
            let sparse_ptr = if sparse_dev == 0 {
                core::ptr::null()
            } else {
                sparse_dev as *const ffi::W34SamplerSparseEntry
            };
            check_rc(
                // SAFETY: `r.last_logits_ptr` holds `num_rows * r.local_vocab_size` FP16 elements,
                // `cfg_dev`/`mask_dev`/`sparse_dev` were uploaded on `r.stream`, and scratch +
                // `local_vals`/`local_ids` are disjoint device allocations.
                unsafe {
                    ffi::w34_sampler_shard_fast_greedy_f16(
                        &ex_s,
                        r.last_logits_ptr as *const c_void,
                        cfg_dev as *const ffi::W34SamplerRowCfg,
                        mask_ptr,
                        sparse_ptr,
                        local_vals as *mut f32,
                        local_ids as *mut i32,
                        num_rows as i32,
                        r.local_vocab_size as i32,
                        r.vocab_offset as i32,
                    )
                },
                "w34_sampler_shard_fast_greedy_f16",
            )?;
            comm.all_gather_top1_f32_i32(
                &ex_s,
                local_vals,
                local_ids,
                global_vals,
                global_ids,
                rank_idx as i32,
                num_rows as i32,
            )?;
            if rank_idx == 0 {
                rank0_global_ids_dev = global_ids;
                rank0_out_off = out_off;
                rank0_use_pinned = use_pinned;
            }
        }
        let global_ids_host = self.read_distributed_greedy_ids(
            num_rows,
            rank0_global_ids_dev,
            rank0_out_off,
            rank0_use_pinned,
        )?;
        let outs = global_ids_host
            .into_iter()
            .zip(row_cfgs)
            .map(|(gid, cfg)| {
                let mut out = ffi::W34SamplerRowOut {
                    top_ids: [-1; ffi::W34_MAX_TOP_LOGPROBS],
                    top_logprobs: [f32::NEG_INFINITY; ffi::W34_MAX_TOP_LOGPROBS],
                    ..Default::default()
                };
                if gid == -2 {
                    out.sampled_id = -1;
                    out.status = 2;
                } else if gid < 0 {
                    out.sampled_id = -1;
                    out.status = if gid == -1 { 1 } else { 3 };
                } else {
                    out.sampled_id = gid;
                    out.status = 0;
                    if cfg.draft_token_id >= 0 {
                        let matched = gid == cfg.draft_token_id;
                        out.draft_accepted = i32::from(matched);
                        out.draft_prob = if matched { 1.0 } else { 0.0 };
                    }
                }
                out
            })
            .collect();
        Ok(Some(outs))
    }

    fn read_distributed_greedy_ids(
        &mut self,
        num_rows: usize,
        rank0_global_ids_dev: u64,
        rank0_out_off: usize,
        rank0_use_pinned: bool,
    ) -> Result<Vec<i32>> {
        let ids_bytes = num_rows * core::mem::size_of::<i32>();
        let mut global_ids_host = vec![0i32; num_rows];
        {
            let r0 = &mut self.ranks[0];
            r0.device.make_current()?;
            if rank0_use_pinned {
                // SAFETY: `rank0_use_pinned` verified `rank0_out_off + ids_bytes` fits inside
                // `PINNED_SAMPLER_BYTES` at `64`-byte aligned offset.
                let pout = unsafe {
                    r0.batched_decode_io
                        .pinned_staging
                        .host_ptr
                        .cast::<u8>()
                        .add(BatchedDecodeGraphIo::PINNED_SAMPLER_OFFSET + rank0_out_off)
                };
                check_rc(
                    // SAFETY: `rank0_global_ids_dev` and `pout` both span `ids_bytes` on `r0.stream`.
                    unsafe {
                        ffi::w34_memcpy_d2h_async(
                            pout.cast(),
                            rank0_global_ids_dev as *const c_void,
                            ids_bytes,
                            r0.stream.handle,
                        )
                    },
                    "copy pinned distributed greedy ids",
                )?;
                r0.stream.synchronize()?;
                r0.batched_decode_io
                    .pinned_in_flight
                    .store(false, std::sync::atomic::Ordering::Release);
                r0.batched_decode_io
                    .pinned_sampler_in_flight
                    .store(false, std::sync::atomic::Ordering::Release);
                // SAFETY: D2H synchronized above; `pout` is 64-byte aligned (`>= align_of::<i32>()`).
                unsafe {
                    core::ptr::copy_nonoverlapping(
                        pout.cast::<i32>(),
                        global_ids_host.as_mut_ptr(),
                        num_rows,
                    );
                }
            } else {
                check_rc(
                    // SAFETY: `global_ids_host` and `rank0_global_ids_dev` hold `num_rows` i32s.
                    unsafe {
                        ffi::w34_memcpy_d2h_async(
                            global_ids_host.as_mut_ptr().cast(),
                            rank0_global_ids_dev as *const c_void,
                            ids_bytes,
                            r0.stream.handle,
                        )
                    },
                    "copy distributed greedy ids",
                )?;
                r0.stream.synchronize()?;
                r0.batched_decode_io
                    .pinned_in_flight
                    .store(false, std::sync::atomic::Ordering::Release);
            }
        }
        for r in &self.ranks[1..] {
            r.device.make_current()?;
            r.stream.synchronize()?;
            r.batched_decode_io
                .pinned_in_flight
                .store(false, std::sync::atomic::Ordering::Release);
            r.batched_decode_io
                .pinned_sampler_in_flight
                .store(false, std::sync::atomic::Ordering::Release);
        }
        Ok(global_ids_host)
    }

    /// Runs GPU-resident `w34_sampler_step_f16` over the `num_rows` device logits produced by the
    /// most recent forward step (`TP=1` directly in VRAM; `TP=2` via mapped-sysmem GPU-to-GPU
    /// `w34_tp2_gather_vocab_f16` onto Rank 0's scratch workspace, or shard-local distributed
    /// fast greedy `w34_sampler_shard_fast_greedy_f16` + `8 B/row` `all_gather_top1_f32_i32`),
    /// copying only compact `W34SamplerRowOut` records (`184 B/row`, zero full-vocab logit D2H) back to host.
    pub fn run_gpu_sampler_rows(
        &mut self,
        row_cfgs: &[ffi::W34SamplerRowCfg],
        bitmask_words: &[u32],
        sparse_entries: &[ffi::W34SamplerSparseEntry],
    ) -> Result<Vec<ffi::W34SamplerRowOut>> {
        let num_rows = row_cfgs.len();
        if num_rows == 0 {
            return Ok(Vec::new());
        }
        let _sampler_range = crate::profiling::Range::new(c"w34.sampler");
        if let Some(outs) =
            self.try_run_distributed_fast_greedy_rows(row_cfgs, bitmask_words, sparse_entries)?
        {
            return Ok(outs);
        }
        let vocab_size = self.cfg.vocab_size;
        let words_per_row = vocab_size.div_ceil(32);

        let (ws_base, ws_cap) = self.ranks[0].sampler_workspace_around_last_logits(num_rows)?;

        let (full_logits_ptr, sampler_ws_ptr, sampler_ws_cap) = if self.tp_size == 1 {
            (self.ranks[0].last_logits_ptr, ws_base, ws_cap)
        } else {
            let comm = self.tp2_comm.as_ref().unwrap().clone();
            let full_bytes = (num_rows * vocab_size * 2 + 255) & !255;
            if ws_cap <= full_bytes {
                return Err(loader_err(
                    "insufficient scratch workspace for TP=2 GPU sampler",
                ));
            }
            for r in &self.ranks[1..] {
                if r.last_logits_ptr == 0 || r.last_logits_rows < num_rows {
                    return Err(loader_err("missing rank 1 device logits for GPU sampler"));
                }
            }
            let shard_bytes = num_rows
                .checked_mul(self.ranks[0].local_vocab_size)
                .and_then(|v| v.checked_mul(2))
                .ok_or_else(|| loader_err("tp2 vocab gather byte overflow"))?;
            if shard_bytes > comm.max_stage_bytes_per_epoch {
                return Err(loader_err(
                    "tp2 vocab gather exceeds max_stage_bytes_per_epoch",
                ));
            }
            let full_ptr = ws_base;
            for (rank_idx, r) in self.ranks.iter().enumerate() {
                let ex = r.stream.exec(0, 0);
                r.device.make_current()?;
                comm.gather_vocab_f16(
                    &ex,
                    r.last_logits_ptr,
                    if rank_idx == 0 { full_ptr } else { 0 },
                    rank_idx as i32,
                    num_rows as i32,
                    r.local_vocab_size as i32,
                )?;
            }
            (full_ptr, ws_base + full_bytes as u64, ws_cap - full_bytes)
        };

        let out_bytes = num_rows * core::mem::size_of::<ffi::W34SamplerRowOut>();
        let r0 = &mut self.ranks[0];
        let (cfg_dev, mask_dev, sparse_dev, out_off, use_pinned_sampler) =
            stage_sampler_inputs_on_rank(r0, row_cfgs, bitmask_words, sparse_entries, out_bytes)?;
        let out_dev = r0.batch_meta_arena.alloc_aligned(out_bytes)?;
        let ex_sampler = r0.stream.exec(sampler_ws_ptr, sampler_ws_cap);
        let use_fast_greedy = vocab_size >= 1024
            && row_cfgs.iter().all(|c| {
                let is_greedy = c.temperature <= 1e-6 || c.top_k == 1;
                let unmasked_ok = (c.has_bitmask == 0 || c.row_bitmask_word_offset < 0)
                    && row_sparse_valid(c, sparse_entries, true)
                    && c.top_logprobs_k <= 0;
                let masked_ok = mask_dev != 0
                    && c.has_bitmask != 0
                    && c.row_bitmask_word_offset >= 0
                    && (c.row_bitmask_word_offset as usize).saturating_add(words_per_row)
                        <= bitmask_words.len()
                    && row_sparse_valid(c, sparse_entries, false)
                    && c.top_logprobs_k < 0;
                is_greedy && (unmasked_ok || masked_ok)
            });
        if use_fast_greedy {
            check_rc(
                // SAFETY: logits rows and output extents are allocated above,
                // inputs were uploaded on this stream, and scratch excludes logits.
                // The fast-path predicate bounds mask rows. Sparse offsets remain
                // the row-config caller's responsibility (not fully checked here).
                unsafe {
                    ffi::w34_sampler_step_fast_greedy_masked_f16(
                        &ex_sampler,
                        full_logits_ptr as *const c_void,
                        cfg_dev as *const ffi::W34SamplerRowCfg,
                        if mask_dev == 0 {
                            core::ptr::null()
                        } else {
                            mask_dev as *const u32
                        },
                        if sparse_dev == 0 {
                            core::ptr::null()
                        } else {
                            sparse_dev as *const ffi::W34SamplerSparseEntry
                        },
                        out_dev as *mut ffi::W34SamplerRowOut,
                        num_rows as i32,
                        vocab_size as i32,
                    )
                },
                "w34_sampler_step_fast_greedy_masked_f16",
            )?;
        } else {
            check_rc(
                // SAFETY: cfg/output have num_rows records and logits have that
                // many full-vocab rows; scratch is disjoint from them. The caller
                // must supply cfg offsets consistent with mask/sparse arrays; their
                // device-side contents are not comprehensively validated here.
                unsafe {
                    ffi::w34_sampler_step_f16(
                        &ex_sampler,
                        full_logits_ptr as *const c_void,
                        cfg_dev as *const ffi::W34SamplerRowCfg,
                        if mask_dev == 0 {
                            core::ptr::null()
                        } else {
                            mask_dev as *const u32
                        },
                        if sparse_dev == 0 {
                            core::ptr::null()
                        } else {
                            sparse_dev as *const ffi::W34SamplerSparseEntry
                        },
                        out_dev as *mut ffi::W34SamplerRowOut,
                        num_rows as i32,
                        vocab_size as i32,
                        sparse_entries.len() as i32,
                    )
                },
                "w34_sampler_step_f16",
            )?;
        }

        let mut outs = vec![ffi::W34SamplerRowOut::default(); num_rows];
        if use_pinned_sampler {
            // SAFETY: use_pinned_sampler checked out_off + out_bytes against the
            // slot's sampler capacity. The allocation and offsets are CUDA-aligned.
            let pout = unsafe {
                r0.batched_decode_io
                    .pinned_staging
                    .host_ptr
                    .cast::<u8>()
                    .add(BatchedDecodeGraphIo::PINNED_SAMPLER_OFFSET + out_off)
            };
            check_rc(
                // SAFETY: both out_dev and the checked pinned output region hold
                // out_bytes. The rank retains the pinned owner until stream sync.
                unsafe {
                    ffi::w34_memcpy_d2h_async(
                        pout.cast(),
                        out_dev as *const c_void,
                        out_bytes,
                        r0.stream.handle,
                    )
                },
                "copy pinned GPU sampler outputs",
            )?;
            r0.stream.synchronize()?;
            r0.batched_decode_io
                .pinned_in_flight
                .store(false, std::sync::atomic::Ordering::Release);
            r0.batched_decode_io
                .pinned_sampler_in_flight
                .store(false, std::sync::atomic::Ordering::Release);
            // SAFETY: the D2H completed at synchronization above. pout is aligned
            // for RowOut, whose scalar fields accept all bit patterns; outs is a
            // separate allocation of exactly num_rows records.
            unsafe {
                core::ptr::copy_nonoverlapping(
                    pout.cast::<ffi::W34SamplerRowOut>(),
                    outs.as_mut_ptr(),
                    num_rows,
                );
            }
        } else {
            check_rc(
                // SAFETY: outs and out_dev both hold num_rows RowOut records;
                // outs stays live until the following stream synchronization.
                unsafe {
                    ffi::w34_memcpy_d2h_async(
                        outs.as_mut_ptr().cast(),
                        out_dev as *const c_void,
                        out_bytes,
                        r0.stream.handle,
                    )
                },
                "copy GPU sampler outputs",
            )?;
            r0.stream.synchronize()?;
            r0.batched_decode_io
                .pinned_in_flight
                .store(false, std::sync::atomic::Ordering::Release);
        }
        if self.tp_size > 1 {
            for r in &self.ranks[1..] {
                r.device.make_current()?;
                r.stream.synchronize()?;
                r.batched_decode_io
                    .pinned_in_flight
                    .store(false, std::sync::atomic::Ordering::Release);
            }
        }
        Ok(outs)
    }

    pub fn extract_step_logprobs(
        emitted_id: i32,
        out: &ffi::W34SamplerRowOut,
        top_k: usize,
    ) -> crate::parser::StepLogprobs {
        let k = top_k.min(ffi::W34_MAX_TOP_LOGPROBS);
        let mut top_logprobs = Vec::with_capacity(k);
        for j in 0..k {
            if out.top_ids[j] >= 0 {
                top_logprobs.push(crate::parser::TokenLogprob {
                    token_id: out.top_ids[j],
                    logprob: out.top_logprobs[j],
                });
            }
        }
        crate::parser::StepLogprobs {
            token_id: emitted_id,
            logprob: out.sampled_logprob,
            top_logprobs,
        }
    }

    /// Samples and commits 1 token from the last 1-row forward step on GPU.
    pub fn sample_single_gpu_step(
        &mut self,
        sampler: &mut crate::parser::SamplerV1,
    ) -> Result<(i32, Option<crate::parser::StepLogprobs>)> {
        sampler
            .materialize_pending_constraint()
            .map_err(|_| loader_err("GPU sampler grammar materialization failed"))?;
        let mut bitmask_words = Vec::new();
        let mut sparse_entries = Vec::new();
        let u = crate::parser::SamplerV1::uniform_f32(
            sampler.params.seed,
            sampler.output_history.len(),
        );
        let cfg = sampler
            .append_gpu_row(
                self.cfg.vocab_size,
                &self.cfg.eos_token_ids,
                u,
                0.0,
                -1,
                false,
                &mut bitmask_words,
                &mut sparse_entries,
            )
            .map_err(|_| loader_err("invalid GPU sampler configuration"))?;
        let outs = self.run_gpu_sampler_rows(&[cfg], &bitmask_words, &sparse_entries)?;
        let out = &outs[0];
        if out.status != 0 || out.sampled_id < 0 {
            return Err(loader_err(
                "GPU sampler rejected empty or nonfinite distribution",
            ));
        }
        sampler
            .try_commit_token(out.sampled_id)
            .map_err(|_| loader_err("GPU sampler grammar commit failed"))?;
        let lp = sampler
            .params
            .logprobs
            .then(|| Self::extract_step_logprobs(out.sampled_id, out, sampler.params.top_logprobs));
        Ok((out.sampled_id, lp))
    }

    pub(crate) fn verify_speculative_gpu_prepared(
        &mut self,
        prepared: PreparedSpecSamplerRows,
        sampler: &crate::parser::SamplerV1,
        drafts: &[i32],
    ) -> Result<(
        crate::speculative::Verification,
        Vec<crate::parser::StepLogprobs>,
    )> {
        let k = drafts.len();
        let mut row_cfgs = Vec::with_capacity(k + 1);
        let mut bitmask_words = Vec::new();
        let mut sparse_entries = Vec::new();
        prepared.append_bonus_into(
            self.cfg.vocab_size,
            &self.cfg.eos_token_ids,
            &mut row_cfgs,
            &mut bitmask_words,
            &mut sparse_entries,
        )?;

        let outs = self.run_gpu_sampler_rows(&row_cfgs, &bitmask_words, &sparse_entries)?;
        if outs.iter().any(|o| o.status != 0) {
            return Err(loader_err(
                "speculative GPU verification encountered invalid target distribution",
            ));
        }
        let accepted = outs[..k]
            .iter()
            .take_while(|o| o.draft_accepted != 0)
            .count();
        let bonus_or_recovered = outs[accepted].sampled_id;
        let decision = crate::speculative::RejectionSampler::from_gpu_verified(
            drafts,
            accepted,
            bonus_or_recovered,
        )
        .map_err(|_| loader_err("GPU speculative verification failed"))?;

        let mut step_lps = Vec::new();
        if sampler.params.logprobs {
            for (idx, &emitted_tok) in decision.emitted_tokens().iter().enumerate() {
                step_lps.push(Self::extract_step_logprobs(
                    emitted_tok,
                    &outs[idx],
                    sampler.params.top_logprobs,
                ));
            }
        }
        Ok((decision, step_lps))
    }

    /// Runs GPU-resident point-mass speculative rejection verification over `k + 1` verify rows.
    pub fn verify_speculative_gpu_step(
        &mut self,
        sampler: &crate::parser::SamplerV1,
        drafts: &[i32],
    ) -> Result<(
        crate::speculative::Verification,
        Vec<crate::parser::StepLogprobs>,
    )> {
        let k = drafts.len();
        let mut hypothetical = sampler.clone();
        let mut row_cfgs = Vec::with_capacity(k + 1);
        let mut bitmask_words = Vec::new();
        let mut sparse_entries = Vec::new();
        let pos = sampler.output_history.len();
        let recovery_u =
            crate::parser::SamplerV1::uniform_f32(sampler.params.seed ^ 0xe7037ed1a0b428db, pos);

        for (row, &draft_id) in drafts.iter().enumerate() {
            let accept_u = crate::parser::SamplerV1::uniform_f32(
                sampler.params.seed ^ 0xa0761d6478bd642f,
                pos + row,
            );
            let cfg = hypothetical
                .append_gpu_row(
                    self.cfg.vocab_size,
                    &self.cfg.eos_token_ids,
                    accept_u,
                    recovery_u,
                    draft_id,
                    false,
                    &mut bitmask_words,
                    &mut sparse_entries,
                )
                .map_err(|_| loader_err("invalid speculative GPU sampler row"))?;
            row_cfgs.push(cfg);
            hypothetical
                .try_commit_token(draft_id)
                .map_err(|_| loader_err("verification draft grammar commit failed"))?;
        }
        let bonus_cfg = hypothetical
            .append_gpu_row(
                self.cfg.vocab_size,
                &self.cfg.eos_token_ids,
                0.0,
                recovery_u,
                -1,
                true,
                &mut bitmask_words,
                &mut sparse_entries,
            )
            .map_err(|_| loader_err("invalid speculative bonus GPU sampler row"))?;
        row_cfgs.push(bonus_cfg);

        let outs = self.run_gpu_sampler_rows(&row_cfgs, &bitmask_words, &sparse_entries)?;
        if outs.iter().any(|o| o.status != 0) {
            return Err(loader_err(
                "speculative GPU verification encountered invalid target distribution",
            ));
        }
        let accepted = outs[..k]
            .iter()
            .take_while(|o| o.draft_accepted != 0)
            .count();
        let bonus_or_recovered = outs[accepted].sampled_id;
        let decision = crate::speculative::RejectionSampler::from_gpu_verified(
            drafts,
            accepted,
            bonus_or_recovered,
        )
        .map_err(|_| loader_err("GPU speculative verification failed"))?;

        let mut step_lps = Vec::new();
        if sampler.params.logprobs {
            for (idx, &emitted_tok) in decision.emitted_tokens().iter().enumerate() {
                step_lps.push(Self::extract_step_logprobs(
                    emitted_tok,
                    &outs[idx],
                    sampler.params.top_logprobs,
                ));
            }
        }
        Ok((decision, step_lps))
    }

    pub(crate) fn verify_batched_speculative_gpu_prepared(
        &mut self,
        items: Vec<(PreparedSpecSamplerRows, &[i32])>,
    ) -> Result<Vec<crate::speculative::Verification>> {
        let total_rows: usize = items.iter().map(|(_, d)| d.len() + 1).sum();
        let mut row_cfgs = Vec::with_capacity(total_rows);
        let mut bitmask_words = Vec::new();
        let mut sparse_entries = Vec::new();
        let drafts_slices: Vec<&[i32]> = items.iter().map(|(_, d)| *d).collect();

        for (prepared, _) in items {
            prepared.append_bonus_into(
                self.cfg.vocab_size,
                &self.cfg.eos_token_ids,
                &mut row_cfgs,
                &mut bitmask_words,
                &mut sparse_entries,
            )?;
        }

        let outs = self.run_gpu_sampler_rows(&row_cfgs, &bitmask_words, &sparse_entries)?;
        if outs.iter().any(|o| o.status != 0) {
            return Err(loader_err(
                "batched speculative GPU verification encountered invalid target distribution",
            ));
        }
        let mut decisions = Vec::with_capacity(drafts_slices.len());
        let mut cursor = 0usize;
        for drafts in drafts_slices {
            let k = drafts.len();
            let slice = &outs[cursor..cursor + k + 1];
            cursor += k + 1;
            let accepted = slice[..k]
                .iter()
                .take_while(|o| o.draft_accepted != 0)
                .count();
            let bonus_or_recovered = slice[accepted].sampled_id;
            let decision = crate::speculative::RejectionSampler::from_gpu_verified(
                drafts,
                accepted,
                bonus_or_recovered,
            )
            .map_err(|_| loader_err("batched GPU speculative verification failed"))?;
            decisions.push(decision);
        }
        Ok(decisions)
    }

    /// Runs GPU-resident speculative rejection verification across a batch of `B = 2..=4`
    /// greedy requests in a single `run_gpu_sampler_rows` launch (`sum(1 + k_b)` total rows).
    pub fn verify_batched_speculative_gpu_steps(
        &mut self,
        items: &[(&crate::parser::SamplerV1, &[i32])],
    ) -> Result<Vec<crate::speculative::Verification>> {
        let total_rows: usize = items.iter().map(|(_, d)| d.len() + 1).sum();
        let mut row_cfgs = Vec::with_capacity(total_rows);
        let mut bitmask_words = Vec::new();
        let mut sparse_entries = Vec::new();

        for &(sampler, drafts) in items {
            let mut hypothetical = sampler.clone();
            let pos = sampler.output_history.len();
            let recovery_u = crate::parser::SamplerV1::uniform_f32(
                sampler.params.seed ^ 0xe7037ed1a0b428db,
                pos,
            );
            for (row, &draft_id) in drafts.iter().enumerate() {
                let accept_u = crate::parser::SamplerV1::uniform_f32(
                    sampler.params.seed ^ 0xa0761d6478bd642f,
                    pos + row,
                );
                let cfg = hypothetical
                    .append_gpu_row(
                        self.cfg.vocab_size,
                        &self.cfg.eos_token_ids,
                        accept_u,
                        recovery_u,
                        draft_id,
                        false,
                        &mut bitmask_words,
                        &mut sparse_entries,
                    )
                    .map_err(|_| loader_err("invalid batched speculative GPU sampler row"))?;
                row_cfgs.push(cfg);
                hypothetical
                    .try_commit_token(draft_id)
                    .map_err(|_| loader_err("batched verification draft grammar commit failed"))?;
            }
            let bonus_cfg = hypothetical
                .append_gpu_row(
                    self.cfg.vocab_size,
                    &self.cfg.eos_token_ids,
                    0.0,
                    recovery_u,
                    -1,
                    true,
                    &mut bitmask_words,
                    &mut sparse_entries,
                )
                .map_err(|_| loader_err("invalid batched speculative bonus GPU sampler row"))?;
            row_cfgs.push(bonus_cfg);
        }

        let outs = self.run_gpu_sampler_rows(&row_cfgs, &bitmask_words, &sparse_entries)?;
        if outs.iter().any(|o| o.status != 0) {
            return Err(loader_err(
                "batched speculative GPU verification encountered invalid target distribution",
            ));
        }
        let mut decisions = Vec::with_capacity(items.len());
        let mut cursor = 0usize;
        for &(_, drafts) in items {
            let k = drafts.len();
            let slice = &outs[cursor..cursor + k + 1];
            cursor += k + 1;
            let accepted = slice[..k]
                .iter()
                .take_while(|o| o.draft_accepted != 0)
                .count();
            let bonus_or_recovered = slice[accepted].sampled_id;
            let decision = crate::speculative::RejectionSampler::from_gpu_verified(
                drafts,
                accepted,
                bonus_or_recovered,
            )
            .map_err(|_| loader_err("batched GPU speculative verification failed"))?;
            decisions.push(decision);
        }
        Ok(decisions)
    }
}
