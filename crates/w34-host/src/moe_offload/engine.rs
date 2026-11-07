use super::*;
use crate::loader::{HADAMARD_BLOCK_64, SafetensorsArchive};

/// Maximum batch token count for `MoeOffloadMode::Auto` to select `GpuLru` when misses fit in staging.
const TIERED_MOE_AUTO_GPU_LRU_MAX_TOKENS: usize = 32;
/// Alternate symmetric 4-bit GPTQ group size supported by the AVX2/FMA CPU bridge (`64` in addition to `128`).
const TIERED_MOE_CPU_BRIDGE_ALT_GROUP_SIZE: usize = 64;

/// Executes a tiered 512-expert MoE layer across `TP=1` or `TP=2 / EP=2` in `GpuLru`,
/// `CpuHybrid`, or double-buffered `StreamedPrefill` mode.
pub struct TieredMoeEngine;

impl TieredMoeEngine {
    /// Pure device-side CUDA-Graph-safe `GpuLru` execution for `num_tokens <= 16`.
    /// Uses `w34_moe_gpu_lru_update_and_pull` to update on-device LRU tables and pull any
    /// missed experts over zero-copy PCIe from `store.mapped_dev_ptr()` (when non-zero),
    /// followed by `w34_moe_w4a16_mlp_f16`.
    #[allow(clippy::too_many_arguments)]
    pub fn execute_layer_device_lru(
        store: &HostExpertStore,
        cache: &mut HotExpertCache,
        ws: &TieredMoeWorkspace,
        compute_stream: &crate::cuda::CudaStream,
        x_addr: u64,
        d_topk_ids_addr: u64,
        d_topk_weights_addr: u64,
        shared_expert_out_addr: u64,
        shared_expert_gate_addr: u64,
        out_addr: u64,
        num_tokens: usize,
        top_k: usize,
    ) -> crate::cuda::Result<()> {
        if num_tokens == 0 {
            return Ok(());
        }
        let k_hot = cache.slots.capacity;
        let ex = compute_stream.exec(ws.d_scratch.addr(), ws.d_scratch.len_bytes);
        if k_hot >= cache.expert_ids.len() {
            let total_pairs = num_tokens * top_k;
            crate::cuda::check_rc(
                // SAFETY: `d_topk_ids_addr`, `d_expert_to_slot`, and `d_remapped_topk_ids[0]` are
                // live device buffers sized for `total_pairs` and `cache.expert_ids.len()`.
                unsafe {
                    w34_kernels::ffi::w34_moe_remap_topk_ids_i32(
                        &ex,
                        d_topk_ids_addr as *const i32,
                        cache.d_expert_to_slot.addr() as *const i32,
                        ws.d_remapped_topk_ids[0].addr() as *mut i32,
                        total_pairs as i32,
                        cache.expert_ids.start as i32,
                        cache.expert_ids.len() as i32,
                    )
                },
                "w34_moe_remap_topk_ids_i32",
            )?;
            return cache.slots.launch_moe_w4a16(
                &ex,
                x_addr,
                ws.d_remapped_topk_ids[0].addr(),
                d_topk_weights_addr,
                shared_expert_out_addr,
                shared_expert_gate_addr,
                ws.d_interm_buf.addr(),
                out_addr,
                num_tokens,
                top_k,
            );
        }
        if let Some(layer_id) = cache.bridge_layer_id
            && num_tokens <= TIERED_MOE_CPU_BRIDGE_MAX_DECODE_TOKENS
            && top_k <= TIERED_MOE_CPU_BRIDGE_MAX_TOP_K
            && store.mapped_dev_ptr() != 0
        {
            crate::cuda::check_rc(
                // SAFETY: device pointers and capacities are owned and validated by `cache` and `ws`.
                unsafe {
                    w34_kernels::ffi::w34_moe_hybrid_submit_and_pull(
                        &ex,
                        layer_id as i32,
                        cache.bridge_mailbox.as_ref().unwrap().request_ptr(),
                        x_addr as *const core::ffi::c_void,
                        d_topk_ids_addr as *const i32,
                        d_topk_weights_addr as *const f32,
                        cache.d_expert_to_slot.addr() as *const i32,
                        ws.d_remapped_topk_ids[0].addr() as *mut i32,
                        ws.d_miss_local_experts.addr() as *mut i32,
                        ws.d_miss_victim_slots.addr() as *mut i32,
                        ws.d_num_misses.addr() as *mut i32,
                        store.mapped_dev_ptr() as *const core::ffi::c_void,
                        cache.slots.gate_up_packed.addr() as *mut u32,
                        cache.slots.gate_up_scales.addr() as *mut core::ffi::c_void,
                        cache.slots.down_packed.addr() as *mut u32,
                        cache.slots.down_scales.addr() as *mut core::ffi::c_void,
                        num_tokens as i32,
                        top_k as i32,
                        cache.expert_ids.start as i32,
                        cache.expert_ids.len() as i32,
                        cache.k_static as i32,
                        k_hot as i32,
                        store.layout().stride,
                        store.layout().intermediate as i32,
                        store.layout().hidden as i32,
                        store.layout().group_size as i32,
                        if cache.k_static < k_hot {
                            TIERED_MOE_CPU_BRIDGE_COLD_WEIGHT_THRESHOLD
                        } else {
                            0.0f32
                        },
                    )
                },
                "w34_moe_hybrid_submit_and_pull",
            )?;
            cache.slots.launch_moe_w4a16(
                &ex,
                x_addr,
                ws.d_remapped_topk_ids[0].addr(),
                d_topk_weights_addr,
                shared_expert_out_addr,
                shared_expert_gate_addr,
                ws.d_interm_buf.addr(),
                out_addr,
                num_tokens,
                top_k,
            )?;
            return crate::cuda::check_rc(
                // SAFETY: `out_addr` is a valid `[num_tokens, hidden]` FP16 buffer.
                unsafe {
                    w34_kernels::ffi::w34_moe_hybrid_wait_and_add(
                        &ex,
                        cache.bridge_mailbox.as_ref().unwrap().request_ptr(),
                        cache.bridge_mailbox.as_ref().unwrap().response_ptr(),
                        out_addr as *mut core::ffi::c_void,
                        ws.d_num_misses.addr() as *const i32,
                        num_tokens as i32,
                        store.layout().hidden as i32,
                    )
                },
                "w34_moe_hybrid_wait_and_add",
            );
        }
        cache.device_lru_dirty = true;
        let total_pairs = num_tokens * top_k;
        let max_misses = total_pairs
            .min(k_hot)
            .min(ws.d_miss_local_experts.len_bytes / 4);
        crate::cuda::check_rc(
            // SAFETY: device pointers in `cache` and `ws` are sized for `total_pairs`, `k_hot`, and `max_misses`.
            unsafe {
                w34_kernels::ffi::w34_moe_gpu_lru_update_and_pull(
                    &ex,
                    d_topk_ids_addr as *const i32,
                    cache.d_expert_to_slot.addr() as *mut i32,
                    cache.d_slot_to_expert.addr() as *mut i32,
                    cache.d_slot_last_used.addr() as *mut u64,
                    cache.d_step_counter.addr() as *mut u64,
                    ws.d_remapped_topk_ids[0].addr() as *mut i32,
                    ws.d_miss_local_experts.addr() as *mut i32,
                    ws.d_miss_victim_slots.addr() as *mut i32,
                    ws.d_num_misses.addr() as *mut i32,
                    store.mapped_dev_ptr() as *const core::ffi::c_void,
                    cache.slots.gate_up_packed.addr() as *mut u32,
                    cache.slots.gate_up_scales.addr() as *mut core::ffi::c_void,
                    cache.slots.down_packed.addr() as *mut u32,
                    cache.slots.down_scales.addr() as *mut core::ffi::c_void,
                    total_pairs as i32,
                    cache.expert_ids.start as i32,
                    cache.expert_ids.len() as i32,
                    k_hot as i32,
                    max_misses as i32,
                    store.layout().stride,
                    store.layout().intermediate as i32,
                    store.layout().hidden as i32,
                    store.layout().group_size as i32,
                )
            },
            "w34_moe_gpu_lru_update_and_pull",
        )?;
        cache.slots.launch_moe_w4a16(
            &ex,
            x_addr,
            ws.d_remapped_topk_ids[0].addr(),
            d_topk_weights_addr,
            shared_expert_out_addr,
            shared_expert_gate_addr,
            ws.d_interm_buf.addr(),
            out_addr,
            num_tokens,
            top_k,
        )
    }

    #[allow(clippy::too_many_arguments, clippy::cognitive_complexity)]
    pub fn execute_layer(
        mode: MoeOffloadMode,
        store: &HostExpertStore,
        cache: &mut HotExpertCache,
        ws: &mut TieredMoeWorkspace,
        compute_stream: &crate::cuda::CudaStream,
        x_addr: u64,
        d_topk_ids_addr: u64,
        d_topk_weights_addr: u64,
        h_topk_ids: &[i32],
        h_topk_weights: &[f32],
        shared_expert_out_addr: u64,
        shared_expert_gate_addr: u64,
        out_addr: u64,
        num_tokens: usize,
        top_k: usize,
    ) -> crate::cuda::Result<()> {
        if num_tokens == 0 {
            return Ok(());
        }
        cache.sync_from_device_if_dirty(compute_stream)?;
        assert_eq!(h_topk_ids.len(), num_tokens * top_k);
        if matches!(mode, MoeOffloadMode::CpuHybrid { .. }) {
            assert_eq!(h_topk_weights.len(), num_tokens * top_k);
        }

        // Identify unique active local experts and count cache hits/misses.
        let local_start = cache.expert_ids.start;
        let local_len = cache.expert_ids.len();
        let mut seen_local = vec![false; local_len];
        let mut active_local: Vec<usize> = Vec::new();
        let mut hits_in_step: Vec<usize> = Vec::new();
        let mut misses_in_step: Vec<usize> = Vec::new();

        for &raw_id in h_topk_ids {
            if raw_id < 0 {
                continue;
            }
            let ge = raw_id as usize;
            let le = ge.wrapping_sub(local_start);
            if le >= local_len {
                continue;
            }
            let is_hit = cache.expert_to_slot[le] >= 0;
            if is_hit {
                cache.hits += 1;
            } else {
                cache.misses += 1;
            }
            if !seen_local[le] {
                seen_local[le] = true;
                active_local.push(ge);
                if is_hit {
                    hits_in_step.push(ge);
                } else {
                    misses_in_step.push(ge);
                }
            }
        }

        let effective_mode = if cache.bridge_layer_id.is_some() {
            MoeOffloadMode::StreamedPrefill
        } else {
            match mode {
                MoeOffloadMode::Auto => {
                    if num_tokens <= TIERED_MOE_AUTO_GPU_LRU_MAX_TOKENS
                        && active_local.len() <= cache.slots.capacity
                        && misses_in_step.len() <= ws.stage_capacity
                    {
                        MoeOffloadMode::GpuLru
                    } else {
                        MoeOffloadMode::StreamedPrefill
                    }
                }
                MoeOffloadMode::GpuLru => {
                    if active_local.len() <= cache.slots.capacity
                        && misses_in_step.len() <= ws.stage_capacity
                    {
                        MoeOffloadMode::GpuLru
                    } else {
                        MoeOffloadMode::StreamedPrefill
                    }
                }
                other => other,
            }
        };

        let step = cache.step_clock;
        if !matches!(effective_mode, MoeOffloadMode::StreamedPrefill) {
            // Touch LRU timestamps on all hit slots first so they cannot be evicted this step.
            cache.step_clock += 1;
            for &ge in &hits_in_step {
                let slot = cache.slot_of(ge).unwrap() as usize;
                cache.slot_last_used[slot] = step;
            }
        }

        let ex = compute_stream.exec(ws.d_scratch.addr(), ws.d_scratch.len_bytes);

        match effective_mode {
            MoeOffloadMode::GpuLru => {
                if !misses_in_step.is_empty() {
                    let mut victim_slots = Vec::with_capacity(misses_in_step.len());
                    for &ge in &misses_in_step {
                        // Choose the slot with the smallest `slot_last_used` timestamp.
                        let (victim_slot, _) = cache
                            .slot_last_used
                            .iter()
                            .enumerate()
                            .min_by_key(|&(_, &ts)| ts)
                            .unwrap();
                        let old_ge = cache.slot_to_expert[victim_slot];
                        if old_ge < usize::MAX {
                            cache.expert_to_slot[old_ge - cache.expert_ids.start] = -1;
                            cache.evictions += 1;
                        }
                        cache.expert_to_slot[ge - cache.expert_ids.start] = victim_slot as i32;
                        cache.slot_to_expert[victim_slot] = ge;
                        cache.slot_last_used[victim_slot] = step;
                        victim_slots.push(victim_slot as i32);
                    }
                    cache.dma_experts_loaded += misses_in_step.len() as u64;
                    for (m_chunk, v_chunk) in misses_in_step
                        .chunks(ws.stage_capacity)
                        .zip(victim_slots.chunks(ws.stage_capacity))
                    {
                        ws.stage_experts_to_slots(
                            0,
                            store,
                            m_chunk,
                            v_chunk,
                            &cache.slots,
                            compute_stream,
                        )?;
                    }
                    ws.compute_done_events[0].record(compute_stream)?;
                    cache
                        .d_expert_to_slot
                        .copy_from_host(&cache.expert_to_slot, compute_stream)?;
                    for (dst, &e) in cache
                        .slot_to_expert_i32
                        .iter_mut()
                        .zip(&cache.slot_to_expert)
                    {
                        *dst = if e == usize::MAX { -1 } else { e as i32 };
                    }
                    cache
                        .d_slot_to_expert
                        .copy_from_host(&cache.slot_to_expert_i32, compute_stream)?;
                    cache
                        .d_slot_last_used
                        .copy_from_host(&cache.slot_last_used, compute_stream)?;
                    cache.step_clock_buf[0] = cache.step_clock;
                    cache
                        .d_step_counter
                        .copy_from_host(&cache.step_clock_buf, compute_stream)?;
                }
                // Remap global `d_topk_ids_addr` on GPU using `cache.d_expert_to_slot`.
                crate::cuda::check_rc(
                    // SAFETY: pointers and counts match allocated device buffers.
                    unsafe {
                        w34_kernels::ffi::w34_moe_remap_topk_ids_i32(
                            &ex,
                            d_topk_ids_addr as *const i32,
                            cache.d_expert_to_slot.addr() as *const i32,
                            ws.d_remapped_topk_ids[0].addr() as *mut i32,
                            (num_tokens * top_k) as i32,
                            cache.expert_ids.start as i32,
                            cache.expert_ids.len() as i32,
                        )
                    },
                    "w34_moe_remap_topk_ids_i32",
                )?;
                cache.slots.launch_moe_w4a16(
                    &ex,
                    x_addr,
                    ws.d_remapped_topk_ids[0].addr(),
                    d_topk_weights_addr,
                    shared_expert_out_addr,
                    shared_expert_gate_addr,
                    ws.d_interm_buf.addr(),
                    out_addr,
                    num_tokens,
                    top_k,
                )
            }

            MoeOffloadMode::CpuHybrid { gpu_pcie_budget } => {
                // Optionally promote up to `gpu_pcie_budget` cold misses into GPU LRU slots.
                let gpu_promote = gpu_pcie_budget
                    .min(misses_in_step.len())
                    .min(ws.stage_capacity)
                    .min(cache.slots.capacity.saturating_sub(hits_in_step.len()));
                if gpu_promote > 0 {
                    let promoted = &misses_in_step[..gpu_promote];
                    let mut victim_slots = Vec::with_capacity(gpu_promote);
                    for &ge in promoted {
                        let (victim_slot, _) = cache
                            .slot_last_used
                            .iter()
                            .enumerate()
                            .min_by_key(|&(_, &ts)| ts)
                            .unwrap();
                        let old_ge = cache.slot_to_expert[victim_slot];
                        if old_ge < usize::MAX {
                            cache.expert_to_slot[old_ge - cache.expert_ids.start] = -1;
                            cache.evictions += 1;
                        }
                        cache.expert_to_slot[ge - cache.expert_ids.start] = victim_slot as i32;
                        cache.slot_to_expert[victim_slot] = ge;
                        cache.slot_last_used[victim_slot] = step;
                        victim_slots.push(victim_slot as i32);
                    }
                    cache.dma_experts_loaded += gpu_promote as u64;
                    ws.stage_experts_to_slots(
                        0,
                        store,
                        promoted,
                        &victim_slots,
                        &cache.slots,
                        compute_stream,
                    )?;
                    ws.compute_done_events[0].record(compute_stream)?;
                    cache
                        .d_expert_to_slot
                        .copy_from_host(&cache.expert_to_slot, compute_stream)?;
                    for (dst, &e) in cache
                        .slot_to_expert_i32
                        .iter_mut()
                        .zip(&cache.slot_to_expert)
                    {
                        *dst = if e == usize::MAX { -1 } else { e as i32 };
                    }
                    cache
                        .d_slot_to_expert
                        .copy_from_host(&cache.slot_to_expert_i32, compute_stream)?;
                    cache
                        .d_slot_last_used
                        .copy_from_host(&cache.slot_last_used, compute_stream)?;
                    cache.step_clock_buf[0] = cache.step_clock;
                    cache
                        .d_step_counter
                        .copy_from_host(&cache.step_clock_buf, compute_stream)?;
                }

                // Partition pairs into GPU hot/promoted slots vs CPU cold tasks.
                let mut remapped_hot = vec![-1i32; num_tokens * top_k];
                let mut cold_tasks = Vec::new();
                for tok in 0..num_tokens {
                    for k in 0..top_k {
                        let idx = tok * top_k + k;
                        let raw_id = h_topk_ids[idx];
                        if raw_id < 0 {
                            continue;
                        }
                        let ge = raw_id as usize;
                        if !cache.expert_ids.contains(&ge) {
                            continue;
                        }
                        if let Some(slot) = cache.slot_of(ge) {
                            remapped_hot[idx] = slot;
                        } else {
                            cold_tasks.push((tok, ge, h_topk_weights[idx]));
                        }
                    }
                }

                ws.d_remapped_topk_ids[0].copy_from_host(&remapped_hot, compute_stream)?;

                // Download `x` to host if there are cold CPU tasks, then launch GPU hot MoE
                // concurrently while the CPU thread pool evaluates cold experts!
                let mut h_x = vec![
                    0u16;
                    if cold_tasks.is_empty() {
                        0
                    } else {
                        num_tokens * ws.layout.hidden
                    }
                ];
                if !cold_tasks.is_empty() {
                    crate::cuda::check_rc(
                        // SAFETY: `h_x` and `x_addr` cover `num_tokens * hidden` FP16 elements.
                        unsafe {
                            w34_kernels::ffi::w34_memcpy_d2h_async(
                                h_x.as_mut_ptr().cast::<core::ffi::c_void>(),
                                x_addr as *const core::ffi::c_void,
                                core::mem::size_of_val(h_x.as_slice()),
                                compute_stream.handle,
                            )
                        },
                        "w34_memcpy_d2h_async (CpuHybrid x)",
                    )?;
                    compute_stream.synchronize()?;
                }

                cache.slots.launch_moe_w4a16(
                    &ex,
                    x_addr,
                    ws.d_remapped_topk_ids[0].addr(),
                    d_topk_weights_addr,
                    shared_expert_out_addr,
                    shared_expert_gate_addr,
                    ws.d_interm_buf.addr(),
                    out_addr,
                    num_tokens,
                    top_k,
                )?;

                if !cold_tasks.is_empty() {
                    cache.cpu_cold_evals += cold_tasks.len() as u64;
                    let h_cold_out = store.eval_cold_tasks_cpu(&h_x, num_tokens, &cold_tasks);
                    ws.d_chunk_out.copy_from_host(&h_cold_out, compute_stream)?;
                    crate::cuda::check_rc(
                        // SAFETY: `out_addr` and `ws.d_chunk_out` are disjoint 4-byte-aligned FP16 buffers.
                        unsafe {
                            w34_kernels::ffi::w34_add_inplace_f16(
                                &ex,
                                out_addr as *mut core::ffi::c_void,
                                ws.d_chunk_out.addr() as *const core::ffi::c_void,
                                (num_tokens * ws.layout.hidden) as i64,
                            )
                        },
                        "w34_add_inplace_f16 (CpuHybrid)",
                    )?;
                }
                Ok(())
            }

            MoeOffloadMode::StreamedPrefill => {
                misses_in_step.sort_unstable();

                // When the C++ AVX2/FMA CPU bridge is active on this layer (`TP=1`) and
                // `h_topk_weights` is provided, partition `misses_in_step` so cold experts
                // with `<= 4` routed tokens (up to ~68% of cold experts, matching the
                // ~65 GB/s DDR5 vs ~25 GB/s PCIe bandwidth ratio) run on the 16-core CPU
                // worker pool in parallel with GPU Pass 0 + PCIe `StreamedPrefill`!
                let mut gpu_misses: Vec<usize> = Vec::new();
                let mut cpu_job_le: Vec<i32> = Vec::new();
                let mut cpu_job_n: Vec<i32> = Vec::new();
                let mut cpu_job_toks: Vec<i32> = Vec::new();
                let mut cpu_job_wts: Vec<f32> = Vec::new();

                let mut promote_victim_slots: Vec<i32> = Vec::new();
                let mut promote_new_ges: Vec<usize> = Vec::new();

                let local_start = cache.expert_ids.start;
                let local_len = cache.expert_ids.len();
                let adapt_end = if cache.bridge_layer_id.is_some() {
                    cache.k_static
                } else {
                    cache.slots.capacity
                };
                let use_cpu_bridge = cache.bridge_layer_id.is_some()
                    && num_tokens <= ws.max_tokens
                    && h_topk_weights.len() == num_tokens * top_k;
                let want_freq_stats = use_cpu_bridge || (num_tokens >= 1024 && adapt_end >= 64);

                if want_freq_stats && !misses_in_step.is_empty() {
                    let mut exp_toks: Vec<Vec<(i32, f32)>> = if use_cpu_bridge {
                        vec![Vec::new(); local_len]
                    } else {
                        Vec::new()
                    };
                    let mut miss_freq: Vec<usize> = vec![0; local_len];
                    let mut hit_freq: Vec<usize> = vec![0; local_len];
                    for (pair_idx, &raw_id) in h_topk_ids.iter().enumerate() {
                        if raw_id < 0 {
                            continue;
                        }
                        let ge = raw_id as usize;
                        let le = ge.wrapping_sub(local_start);
                        if le < local_len {
                            if cache.expert_to_slot[le] < 0 {
                                miss_freq[le] += 1;
                                if use_cpu_bridge {
                                    let tok = (pair_idx / top_k) as i32;
                                    let wt = h_topk_weights[pair_idx];
                                    let list = &mut exp_toks[le];
                                    if let Some(last) = list.last_mut()
                                        && last.0 == tok
                                    {
                                        last.1 += wt;
                                    } else {
                                        list.push((tok, wt));
                                    }
                                }
                            } else {
                                hit_freq[le] += 1;
                            }
                        }
                    }

                    if use_cpu_bridge {
                        // Rank cold experts with `1..=max_tok_per_job` routed tokens by token count ascending
                        // so the sparsest tail experts go to the CPU DDR5 pool (where row pairs are
                        // unpacked once and FMA-accumulated across all routed tokens) and the densest
                        // cold experts fit in at most 1 GPU chunk (`<= 384` tokens) or 2 GPU chunks (`> 384` tokens).
                        let (target_gpu_misses, cpu_share_pct, max_tok_per_job) =
                            if num_tokens <= 384 {
                                (ws.stage_capacity, 70usize, 8usize)
                            } else {
                                (2 * ws.stage_capacity, 58usize, 96usize)
                            };
                        let max_cpu_jobs = misses_in_step
                            .len()
                            .saturating_sub(target_gpu_misses)
                            .max((misses_in_step.len() * cpu_share_pct) / 100);
                        let mut light_candidates: Vec<(usize, usize)> = misses_in_step
                            .iter()
                            .copied()
                            .filter_map(|ge| {
                                let cnt = exp_toks[ge - local_start].len();
                                (1..=max_tok_per_job).contains(&cnt).then_some((cnt, ge))
                            })
                            .collect();
                        light_candidates.sort_unstable();
                        light_candidates.truncate(max_cpu_jobs);
                        // Schedule heaviest CPU jobs first (Longest-Processing-Time-First)
                        // so 1-token tail jobs fill in across all 16 worker threads at the end.
                        light_candidates.reverse();

                        let mut is_cpu_le = vec![false; local_len];
                        for &(_, ge) in &light_candidates {
                            let le = ge - local_start;
                            is_cpu_le[le] = true;
                            let list = &exp_toks[le];
                            cpu_job_le.push(le as i32);
                            cpu_job_n.push(list.len() as i32);
                            for s in 0..96 {
                                if s < list.len() {
                                    cpu_job_toks.push(list[s].0);
                                    cpu_job_wts.push(list[s].1);
                                } else {
                                    cpu_job_toks.push(0);
                                    cpu_job_wts.push(0.0);
                                }
                            }
                        }
                        for &ge in &misses_in_step {
                            if !is_cpu_le[ge - local_start] {
                                gpu_misses.push(ge);
                            }
                        }
                    } else {
                        gpu_misses.clone_from(&misses_in_step);
                    }

                    // On long prompts (`num_tokens >= 1024`, both `TP=1` and `TP=2 / EP=2`), adapt
                    // the bottom 20 slots of `0..adapt_end` by promoting up to 16 cold experts
                    // from `gpu_misses` that are >= 3x more frequent in this prompt than the
                    // lowest-frequency resident tail slots, unpacking them directly from
                    // `ws.d_stage_packs[0]` during Chunk 0 at zero extra PCIe DMA cost.
                    if num_tokens >= 1024 && adapt_end >= 64 && !gpu_misses.is_empty() {
                        let adapt_start = adapt_end - 20;
                        let mut victim_cands: Vec<(usize, usize)> = (adapt_start..adapt_end)
                            .map(|slot| {
                                let ge = cache.slot_to_expert[slot];
                                let freq = if ge < usize::MAX && cache.expert_ids.contains(&ge) {
                                    hit_freq[ge - local_start]
                                } else {
                                    0
                                };
                                (freq, slot)
                            })
                            .collect();
                        victim_cands.sort_unstable();

                        let mut heavy_cands: Vec<(usize, usize)> = gpu_misses
                            .iter()
                            .copied()
                            .map(|ge| (miss_freq[ge - local_start], ge))
                            .collect();
                        heavy_cands.sort_unstable_by(|a, b| b.cmp(a));

                        let min_freq = (num_tokens / 64).max(16);
                        let max_promote = victim_cands.len().min(heavy_cands.len()).min(16);
                        let mut pairs: Vec<(usize, i32)> = Vec::new();
                        for i in 0..max_promote {
                            let (new_freq, new_ge) = heavy_cands[i];
                            let (old_freq, slot) = victim_cands[i];
                            if new_freq >= min_freq && new_freq > old_freq * 3 {
                                pairs.push((new_ge, slot as i32));
                            } else {
                                break;
                            }
                        }
                        if !pairs.is_empty() {
                            pairs.sort_unstable_by_key(|&(ge, _)| ge);
                            let mut is_promoted = vec![false; local_len];
                            for &(ge, slot) in &pairs {
                                is_promoted[ge - local_start] = true;
                                promote_new_ges.push(ge);
                                promote_victim_slots.push(slot);
                            }
                            let mut reordered = Vec::with_capacity(gpu_misses.len());
                            reordered.extend_from_slice(&promote_new_ges);
                            for &ge in &gpu_misses {
                                if !is_promoted[ge - local_start] {
                                    reordered.push(ge);
                                }
                            }
                            gpu_misses = reordered;
                        }
                    }
                } else {
                    gpu_misses.clone_from(&misses_in_step);
                }

                // Enqueue async D2H of `x_addr` into pinned `ws.h_prefill_x` and record `ws.d2h_done_event`
                // WITHOUT blocking the host thread yet — we enqueue Pass 0 and all GPU cold chunks first!
                let x_bytes = num_tokens * ws.layout.hidden * core::mem::size_of::<u16>();
                if !cpu_job_le.is_empty() {
                    crate::cuda::check_rc(
                        // SAFETY: `ws.h_prefill_x` and `x_addr` cover `x_bytes` bytes.
                        unsafe {
                            w34_kernels::ffi::w34_memcpy_d2h_async(
                                ws.h_prefill_x.host_ptr,
                                x_addr as *const core::ffi::c_void,
                                x_bytes,
                                compute_stream.handle,
                            )
                        },
                        "w34_memcpy_d2h_async (StreamedPrefill hybrid x)",
                    )?;
                    ws.d2h_done_event.record(compute_stream)?;
                }

                // Pass 0: Execute all hot-cache hits on `compute_stream` (plus shared expert).
                crate::cuda::check_rc(
                    // SAFETY: pointers and counts match allocated device buffers.
                    unsafe {
                        w34_kernels::ffi::w34_moe_remap_topk_ids_i32(
                            &ex,
                            d_topk_ids_addr as *const i32,
                            cache.d_expert_to_slot.addr() as *const i32,
                            ws.d_remapped_topk_ids[0].addr() as *mut i32,
                            (num_tokens * top_k) as i32,
                            cache.expert_ids.start as i32,
                            cache.expert_ids.len() as i32,
                        )
                    },
                    "w34_moe_remap_topk_ids_i32 (Pass 0)",
                )?;

                // Before launching Pass 0 on `compute_stream`, kick off Chunk 0 H2D DMA on `ws.h2d_stream`
                // so hardware PCIe DMA overlaps 100% with Pass 0 GPU compute!
                let chunks: Vec<&[usize]> = gpu_misses.chunks(ws.stage_capacity).collect();
                if !chunks.is_empty() {
                    let c0 = chunks[0];
                    let dst_slots: Vec<i32> = (0..c0.len() as i32).collect();
                    ws.h2d_stream.wait_event(&ws.compute_done_events[0])?;
                    ws.stage_experts_h2d_dma(0, store, c0, &dst_slots, &ws.h2d_stream)?;
                    cache.dma_experts_loaded += c0.len() as u64;
                    ws.h2d_done_events[0].record(&ws.h2d_stream)?;
                }

                cache.slots.launch_moe_w4a16(
                    &ex,
                    x_addr,
                    ws.d_remapped_topk_ids[0].addr(),
                    d_topk_weights_addr,
                    shared_expert_out_addr,
                    shared_expert_gate_addr,
                    ws.d_interm_buf.addr(),
                    out_addr,
                    num_tokens,
                    top_k,
                )?;

                // Passes 1..=chunks.len(): Double-buffered hardware H2D DMA + Grouped-MMA streaming.
                for (c_idx, chunk) in chunks.iter().enumerate() {
                    let buf_idx = c_idx & 1;
                    // Enqueue H2D DMA for `c_idx + 1` on `ws.h2d_stream` into `1 - buf_idx`
                    // before launching compute for `c_idx` on `compute_stream`!
                    if c_idx + 1 < chunks.len() {
                        let next_buf = 1 - buf_idx;
                        let next_chunk = chunks[c_idx + 1];
                        let dst_slots: Vec<i32> = (0..next_chunk.len() as i32).collect();
                        ws.h2d_stream
                            .wait_event(&ws.compute_done_events[next_buf])?;
                        ws.stage_experts_h2d_dma(
                            next_buf,
                            store,
                            next_chunk,
                            &dst_slots,
                            &ws.h2d_stream,
                        )?;
                        cache.dma_experts_loaded += next_chunk.len() as u64;
                        ws.h2d_done_events[next_buf].record(&ws.h2d_stream)?;
                    }

                    // Build per-expert chunk slot table (`<= 512` i32s = 2 KB inline async H2D)
                    // and remap all `num_tokens * top_k` pairs on the GPU via `w34_moe_remap_topk_ids_i32`.
                    let local_start = cache.expert_ids.start;
                    let local_len = cache.expert_ids.len();
                    let mut local_to_chunk_slot = vec![-1i32; local_len];
                    for (pos, &ge) in chunk.iter().enumerate() {
                        local_to_chunk_slot[ge - local_start] = pos as i32;
                    }

                    compute_stream.wait_event(&ws.h2d_done_events[buf_idx])?;
                    ws.stage_tables[buf_idx].unpack_staged(
                        compute_stream,
                        ws.d_stage_packs[buf_idx].addr(),
                        ws.d_stage_slots[buf_idx].addr(),
                        chunk.len(),
                    )?;
                    if c_idx == 0 && !promote_victim_slots.is_empty() {
                        ws.d_miss_victim_slots
                            .copy_from_host(&promote_victim_slots, compute_stream)?;
                        cache.slots.unpack_staged(
                            compute_stream,
                            ws.d_stage_packs[0].addr(),
                            ws.d_miss_victim_slots.addr(),
                            promote_victim_slots.len(),
                        )?;
                        cache.step_clock += 8;
                        let promoted_ts = cache.step_clock;
                        for (&new_ge, &slot_i32) in
                            promote_new_ges.iter().zip(&promote_victim_slots)
                        {
                            let slot = slot_i32 as usize;
                            let old_ge = cache.slot_to_expert[slot];
                            if old_ge < usize::MAX && cache.expert_ids.contains(&old_ge) {
                                cache.expert_to_slot[old_ge - local_start] = -1;
                                cache.evictions += 1;
                            }
                            cache.expert_to_slot[new_ge - local_start] = slot_i32;
                            cache.slot_to_expert[slot] = new_ge;
                            cache.slot_to_expert_i32[slot] = new_ge as i32;
                            cache.slot_last_used[slot] = promoted_ts;
                        }
                        cache
                            .d_expert_to_slot
                            .copy_from_host(&cache.expert_to_slot, compute_stream)?;
                        cache
                            .d_slot_to_expert
                            .copy_from_host(&cache.slot_to_expert_i32, compute_stream)?;
                        cache
                            .d_slot_last_used
                            .copy_from_host(&cache.slot_last_used, compute_stream)?;
                        cache
                            .d_step_counter
                            .copy_from_host(&[cache.step_clock], compute_stream)?;
                    }
                    ws.compute_done_events[buf_idx].record(compute_stream)?;

                    ws.d_miss_local_experts
                        .copy_from_host(&local_to_chunk_slot, compute_stream)?;
                    crate::cuda::check_rc(
                        // SAFETY: pointers and counts match allocated device buffers.
                        unsafe {
                            w34_kernels::ffi::w34_moe_remap_topk_ids_i32(
                                &ex,
                                d_topk_ids_addr as *const i32,
                                ws.d_miss_local_experts.addr() as *const i32,
                                ws.d_remapped_topk_ids[buf_idx].addr() as *mut i32,
                                (num_tokens * top_k) as i32,
                                local_start as i32,
                                local_len as i32,
                            )
                        },
                        "w34_moe_remap_topk_ids_i32 (StreamedPrefill chunk)",
                    )?;
                    ws.stage_tables[buf_idx].launch_moe_w4a16(
                        &ex,
                        x_addr,
                        ws.d_remapped_topk_ids[buf_idx].addr(),
                        d_topk_weights_addr,
                        0,
                        0,
                        ws.d_interm_buf.addr(),
                        ws.d_chunk_out.addr(),
                        num_tokens,
                        top_k,
                    )?;
                    crate::cuda::check_rc(
                        // SAFETY: `out_addr` and `ws.d_chunk_out` are disjoint 4-byte-aligned FP16 buffers.
                        unsafe {
                            w34_kernels::ffi::w34_add_inplace_f16(
                                &ex,
                                out_addr as *mut core::ffi::c_void,
                                ws.d_chunk_out.addr() as *const core::ffi::c_void,
                                (num_tokens * ws.layout.hidden) as i64,
                            )
                        },
                        "w34_add_inplace_f16 (StreamedPrefill)",
                    )?;
                }

                // While the GPU is executing Pass 0 + `chunks` on `compute_stream`, wait only for
                // `ws.d2h_done_event` (which was recorded before Pass 0!) and evaluate `cpu_job_le`
                // across the 16-core AVX2/FMA worker pool in DDR5 RAM!
                if !cpu_job_le.is_empty()
                    && let Some(_layer_id) = cache.bridge_layer_id
                {
                    ws.d2h_done_event.synchronize()?;
                    cache.cpu_cold_evals += cpu_job_le.len() as u64;
                    // SAFETY: event completed D2H; these pinned buffers cover the batch and are exclusively used here.
                    unsafe {
                        cache.bridge_runtime.as_ref().unwrap().prefill(
                            store,
                            std::slice::from_raw_parts(
                                ws.h_prefill_x.host_ptr.cast::<u16>(),
                                num_tokens * ws.layout.hidden,
                            ),
                            num_tokens,
                            &cpu_job_le,
                            &cpu_job_n,
                            &cpu_job_toks,
                            &cpu_job_wts,
                            std::slice::from_raw_parts_mut(
                                ws.h_prefill_out.host_ptr.cast::<u16>(),
                                num_tokens * ws.layout.hidden,
                            ),
                        )?;
                    }
                    crate::cuda::check_rc(
                        // SAFETY: `ws.h_prefill_out` is pinned host memory and `ws.d_chunk_out` fits `x_bytes`.
                        unsafe {
                            w34_kernels::ffi::w34_memcpy_h2d_async(
                                ws.d_chunk_out.addr() as *mut core::ffi::c_void,
                                ws.h_prefill_out.host_ptr,
                                x_bytes,
                                compute_stream.handle,
                            )
                        },
                        "w34_memcpy_h2d_async (StreamedPrefill CPU hybrid out)",
                    )?;
                    crate::cuda::check_rc(
                        // SAFETY: `out_addr` and `ws.d_chunk_out` are disjoint 4-byte-aligned FP16 buffers.
                        unsafe {
                            w34_kernels::ffi::w34_add_inplace_f16(
                                &ex,
                                out_addr as *mut core::ffi::c_void,
                                ws.d_chunk_out.addr() as *const core::ffi::c_void,
                                (num_tokens * ws.layout.hidden) as i64,
                            )
                        },
                        "w34_add_inplace_f16 (StreamedPrefill CPU hybrid)",
                    )?;
                }
                Ok(())
            }

            MoeOffloadMode::Auto => unreachable!(),
        }
    }
}

/// Per-rank 48-layer tiered 512-expert MoE runtime (`stores`, `caches`, `ws`, and `mode`).
pub struct RankTieredMoe {
    pub stores: Vec<HostExpertStore>,
    pub caches: Vec<HotExpertCache>,
    pub ws: TieredMoeWorkspace,
    pub mode: MoeOffloadMode,
}

impl RankTieredMoe {
    #[allow(clippy::too_many_arguments)]
    pub fn load_for_rank(
        archive: &SafetensorsArchive,
        cfg: &crate::loader::ModelConfig,
        device_id: i32,
        ep_rank: usize,
        ep_size: usize,
        k_hot: usize,
        stage_capacity: usize,
        stream: &crate::cuda::CudaStream,
    ) -> crate::cuda::Result<Self> {
        let group_size = cfg.routed_expert_quant_scheme().group_size_or(128);
        let down_hadamard = cfg
            .quant
            .input_hadamard("model.layers.0.mlp.experts.0.down_proj")
            .ok()
            .flatten()
            .unwrap_or(0);
        let layout = ExpertPackLayout::new(cfg.moe_intermediate_size, cfg.hidden_size, group_size)
            .and_then(|l| l.with_down_hadamard(down_hadamard))
            .map_err(|e| {
                crate::cuda::CudaError::with_detail(-1, "ExpertPackLayout::new", e.to_string())
            })?;
        let mut ws = TieredMoeWorkspace::new(
            device_id,
            layout.clone(),
            stage_capacity,
            TIERED_MOE_MAX_BATCH_TOKENS,
            cfg.num_experts_per_tok.max(1),
        )?;
        let rankings = archive
            .base_dir()
            .and_then(HotExpertRankings::from_model_dir);
        let mut stores = Vec::with_capacity(cfg.num_hidden_layers);
        let mut caches = Vec::with_capacity(cfg.num_hidden_layers);
        let mut bridge_runtime = None;
        let local_experts = cfg.num_experts / ep_size.max(1);
        let fully_resident = k_hot >= local_experts;
        for layer_id in 0..cfg.num_hidden_layers {
            let prefix = format!("model.layers.{layer_id}.mlp.experts");
            let mut store = HostExpertStore::load_with_pin(
                archive,
                &prefix,
                &cfg.quant,
                layout.clone(),
                cfg.num_experts,
                ep_rank,
                ep_size,
                usize::MAX,
                !fully_resident,
            )
            .map_err(|e| {
                crate::cuda::CudaError::with_detail(-1, "HostExpertStore::load", e.to_string())
            })?;
            let mut cache =
                HotExpertCache::new(device_id, k_hot, store.expert_ids(), store.layout().clone())?;
            let initial = rankings
                .as_ref()
                .and_then(|r| r.select_for_layer(layer_id, store.expert_ids(), k_hot));
            cache.populate_initial(&store, initial.as_deref(), &mut ws, stream)?;
            if fully_resident {
                stream.synchronize()?;
                store.release_host_storage();
            } else if ep_size == 1
                && k_hot >= TIERED_MOE_CPU_BRIDGE_MIN_HOT_EXPERTS
                && cfg.num_hidden_layers <= TIERED_MOE_CPU_BRIDGE_MAX_LAYERS
                && matches!(
                    group_size,
                    TIERED_MOE_CPU_BRIDGE_ALT_GROUP_SIZE | TIERED_MOE_CPU_BRIDGE_GROUP_SIZE
                )
                && matches!(
                    store.layout().down_hadamard_block_size,
                    0 | HADAMARD_BLOCK_64
                )
                && HotExpertCache::auto_cpu_bridge_preferred()
            {
                let k_static = k_hot;
                if bridge_runtime.is_none() {
                    bridge_runtime = Some(crate::moe_cpu_bridge::Runtime::new(stream)?);
                }
                cache.bridge_runtime = bridge_runtime.clone();
                cache.register_cpu_bridge(layer_id, &store, k_static, stream)?;
            }
            stores.push(store);
            caches.push(cache);
        }
        Ok(Self {
            stores,
            caches,
            ws,
            mode: MoeOffloadMode::Auto,
        })
    }

    pub fn layer_has_mapped_store(&self, layer_idx: usize) -> bool {
        self.stores
            .get(layer_idx)
            .is_some_and(|s| s.mapped_dev_ptr() != 0)
    }

    pub fn layer_is_fully_resident(&self, layer_idx: usize) -> bool {
        self.caches
            .get(layer_idx)
            .is_some_and(|c| c.slots.capacity >= c.expert_ids.len())
    }

    pub fn layer_hot_k(&self, layer_idx: usize) -> usize {
        self.caches.get(layer_idx).map_or(0, |c| c.slots.capacity)
    }

    #[allow(clippy::too_many_arguments)]
    pub fn execute_layer_device_lru(
        &mut self,
        compute_stream: &crate::cuda::CudaStream,
        layer_idx: usize,
        x_addr: u64,
        d_topk_ids_addr: u64,
        d_topk_weights_addr: u64,
        shared_expert_out_addr: u64,
        shared_expert_gate_addr: u64,
        out_addr: u64,
        num_tokens: usize,
        top_k: usize,
    ) -> crate::cuda::Result<()> {
        TieredMoeEngine::execute_layer_device_lru(
            &self.stores[layer_idx],
            &mut self.caches[layer_idx],
            &self.ws,
            compute_stream,
            x_addr,
            d_topk_ids_addr,
            d_topk_weights_addr,
            shared_expert_out_addr,
            shared_expert_gate_addr,
            out_addr,
            num_tokens,
            top_k,
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub fn execute_layer(
        &mut self,
        compute_stream: &crate::cuda::CudaStream,
        layer_idx: usize,
        x_addr: u64,
        d_topk_ids_addr: u64,
        d_topk_weights_addr: u64,
        shared_expert_out_addr: u64,
        shared_expert_gate_addr: u64,
        out_addr: u64,
        num_tokens: usize,
        top_k: usize,
    ) -> crate::cuda::Result<()> {
        if num_tokens == 0 {
            return Ok(());
        }
        let pairs = num_tokens * top_k;
        let mut h_topk_ids = vec![0i32; pairs];
        crate::cuda::check_rc(
            // SAFETY: `d_topk_ids_addr` holds `pairs` i32 elements.
            unsafe {
                w34_kernels::ffi::w34_memcpy_d2h_async(
                    h_topk_ids.as_mut_ptr().cast::<core::ffi::c_void>(),
                    d_topk_ids_addr as *const core::ffi::c_void,
                    pairs * 4,
                    compute_stream.handle,
                )
            },
            "w34_memcpy_d2h_async (h_topk_ids)",
        )?;
        let mut h_topk_weights = Vec::new();
        if matches!(self.mode, MoeOffloadMode::CpuHybrid { .. })
            || self.caches[layer_idx].bridge_layer_id.is_some()
        {
            h_topk_weights.resize(pairs, 0.0f32);
            crate::cuda::check_rc(
                // SAFETY: `d_topk_weights_addr` holds `pairs` f32 elements.
                unsafe {
                    w34_kernels::ffi::w34_memcpy_d2h_async(
                        h_topk_weights.as_mut_ptr().cast::<core::ffi::c_void>(),
                        d_topk_weights_addr as *const core::ffi::c_void,
                        pairs * 4,
                        compute_stream.handle,
                    )
                },
                "w34_memcpy_d2h_async (h_topk_weights)",
            )?;
        }
        compute_stream.synchronize()?;
        TieredMoeEngine::execute_layer(
            self.mode,
            &self.stores[layer_idx],
            &mut self.caches[layer_idx],
            &mut self.ws,
            compute_stream,
            x_addr,
            d_topk_ids_addr,
            d_topk_weights_addr,
            &h_topk_ids,
            &h_topk_weights,
            shared_expert_out_addr,
            shared_expert_gate_addr,
            out_addr,
            num_tokens,
            top_k,
        )
    }
}
