use super::*;

#[test]
fn canonical_pack_has_exact_regions_and_page_aligned_stride() {
    let layout = ExpertPackLayout::new(640, 2560, 128).unwrap();
    assert_eq!(layout.gate_up_packed, 0..1_638_400);
    assert_eq!(layout.gate_up_scales.len(), 51_200);
    assert_eq!(layout.down_packed.len(), 819_200);
    assert_eq!(layout.down_scales.len(), 25_600);
    assert_eq!(layout.down_scales.end, 2_534_400);
    assert_eq!(layout.stride, 2_535_424);
}

#[test]
fn invalid_or_overflowing_geometry_is_rejected() {
    for (i, h, g) in [
        (0, 128, 128),
        (64, 128, 128),
        (640, 2561, 128),
        (640, 2560, 0),
        (usize::MAX, 2560, 128),
    ] {
        assert!(ExpertPackLayout::new(i, h, g).is_err());
    }
}

#[test]
fn hot_expert_rankings_filter_dedup_and_backfill_per_ep_rank() {
    let rankings = super::HotExpertRankings::from_json(&serde_json::json!({
        "0": [6, 2, 7, 2, 1, 99, 5],
        "1": [4]
    }))
    .unwrap();
    // TP=1 (0..8), k_hot=4 -> [6, 2, 7, 1]
    assert_eq!(
        rankings.select_for_layer(0, 0..8, 4),
        Some(vec![6, 2, 7, 1])
    );
    // EP=2 rank 0 (0..4), k_hot=3 -> [2, 1] from rankings + backfill [0]
    assert_eq!(rankings.select_for_layer(0, 0..4, 3), Some(vec![2, 1, 0]));
    // EP=2 rank 1 (4..8), k_hot=3 -> [6, 7, 5]
    assert_eq!(rankings.select_for_layer(0, 4..8, 3), Some(vec![6, 7, 5]));
    // Layer 1 on EP=2 rank 1 (4..8), k_hot=2 -> [4] + backfill [5]
    assert_eq!(rankings.select_for_layer(1, 4..8, 2), Some(vec![4, 5]));
    // Missing layer returns None (caller uses sequential 0..k_hot default)
    assert_eq!(rankings.select_for_layer(2, 0..8, 3), None);
}

#[cfg(feature = "cuda")]
mod cuda_tests {
    use super::super::*;
    use crate::cuda::{CudaDevice, CudaStream, DeviceBuffer};
    use w34_kernels::ffi;

    #[test]
    #[ignore = "requires the pinned Qwen3.8-Flash-Next checkpoint"]
    fn canonical_checkpoint_host_store_tp1_and_ep2_agree() {
        use super::super::{ExpertPackLayout, HostExpertStore};
        use crate::loader::{ModelConfig, SafetensorsArchive, resolve_hf_model_dir};
        let _device = CudaDevice::set_current(0).unwrap();
        let snapshot_buf =
            resolve_hf_model_dir("albucino/Qwen3.8-Flash-Next-W4A16-FP8PLE").unwrap();
        let snapshot = snapshot_buf.as_path();
        let archive = SafetensorsArchive::open(snapshot).unwrap();
        let cfg = ModelConfig::from_dir(snapshot).unwrap();
        let layout = ExpertPackLayout::new(640, 2560, 128).unwrap();
        let prefix = "model.language_model.layers.0.mlp.experts";
        // Two real experts keep this qualification small; EP uses the same partition rule at 512.
        let single = HostExpertStore::load(
            &archive,
            prefix,
            &cfg.quant,
            layout.clone(),
            2,
            0,
            1,
            2 * layout.stride,
        )
        .unwrap();
        for rank in 0..2 {
            let shard = HostExpertStore::load(
                &archive,
                prefix,
                &cfg.quant,
                layout.clone(),
                2,
                rank,
                2,
                layout.stride,
            )
            .unwrap();
            assert_eq!(shard.expert_ids(), rank..rank + 1);
            assert_eq!(shard.pinned_bytes(), layout.stride);
            assert_eq!(single.expert_pack(rank), shard.expert_pack(rank));
            assert!(shard.expert_pack(1 - rank).is_none());
            let pack = shard.expert_pack(rank).unwrap();
            for (projection, region, n, k) in [
                (
                    "gate_proj",
                    layout.gate_up_packed.start..layout.gate_up_packed.start + 819200,
                    640,
                    2560,
                ),
                ("up_proj", 819200..layout.gate_up_packed.end, 640, 2560),
                ("down_proj", layout.down_packed.clone(), 2560, 640),
            ] {
                let original = archive
                    .read_u32(&format!("{prefix}.{rank}.{projection}.qweight"))
                    .unwrap();
                for row in [0, n / 2, n - 1] {
                    for col in [0, k / 16, k / 8 - 1] {
                        let offset = region.start + (row * (k / 8) + col) * 4;
                        let word = u32::from_le_bytes(pack[offset..offset + 4].try_into().unwrap());
                        assert_eq!(word, original[col * n + row]);
                    }
                }
            }
            assert!(pack[layout.down_scales.end..].iter().all(|&v| v == 0));
        }
        assert!(
            HostExpertStore::load(
                &archive,
                prefix,
                &cfg.quant,
                layout.clone(),
                2,
                0,
                1,
                layout.stride
            )
            .is_err()
        );
    }

    fn upload<T: Copy>(values: &[T], stream: &CudaStream) -> DeviceBuffer {
        let buffer = DeviceBuffer::alloc(0, std::mem::size_of_val(values)).unwrap();
        buffer.copy_from_host(values, stream).unwrap();
        stream.synchronize().unwrap();
        buffer
    }

    #[test]
    fn staged_experts_scatter_all_regions_and_remap_ep_ids() {
        let _device = CudaDevice::set_current(0).unwrap();
        let stream = CudaStream::new(0).unwrap();
        let ex = stream.exec(0, 0);
        let layout = ExpertPackLayout::new(128, 128, 128).unwrap();
        let source: Vec<u8> = (0..layout.stride * 2)
            .map(|i| ((i * 17 + i / layout.stride) % 251) as u8)
            .collect();
        let staged = upload(&source, &stream);
        let slots = upload(&[2i32, 0], &stream);
        let regions = [
            &layout.gate_up_packed,
            &layout.gate_up_scales,
            &layout.down_packed,
            &layout.down_scales,
        ];
        let outputs: Vec<_> = regions
            .iter()
            .map(|r| upload(&vec![0u8; r.len() * 3], &stream))
            .collect();
        // SAFETY: all buffers cover the specified geometry and destination slots are unique and in bounds.
        assert_eq!(
            // SAFETY: allocated buffers cover the geometry; destination slots are unique and bounded.
            unsafe {
                ffi::w34_moe_unpack_staged_experts(
                    &ex,
                    staged.addr() as *const _,
                    slots.addr() as *const _,
                    outputs[0].addr() as *mut _,
                    outputs[1].addr() as *mut _,
                    outputs[2].addr() as *mut _,
                    outputs[3].addr() as *mut _,
                    2,
                    3,
                    layout.stride,
                    128,
                    128,
                    128,
                )
            },
            0
        );
        stream.synchronize().unwrap();
        for (region, buffer) in regions.iter().zip(&outputs) {
            let mut actual = vec![0u8; region.len() * 3];
            buffer.copy_to_host(&mut actual, &stream).unwrap();
            assert_eq!(
                &actual[..region.len()],
                &source[layout.stride + region.start..layout.stride + region.end]
            );
            assert!(
                actual[region.len()..2 * region.len()]
                    .iter()
                    .all(|&v| v == 0)
            );
            assert_eq!(&actual[2 * region.len()..], &source[(*region).clone()]);
        }
        let ids = upload(&[-1i32, 255, 256, 257, 258, i32::MIN, i32::MAX], &stream);
        let map = upload(&[2i32, -1], &stream);
        let result = upload(&[99i32; 7], &stream);
        // SAFETY: input/output arrays have seven entries and map covers two local experts.
        assert_eq!(
            // SAFETY: input/output have seven entries and map has two local experts.
            unsafe {
                ffi::w34_moe_remap_topk_ids_i32(
                    &ex,
                    ids.addr() as *const _,
                    map.addr() as *const _,
                    result.addr() as *mut _,
                    7,
                    256,
                    2,
                )
            },
            0
        );
        let mut actual = [0i32; 7];
        result.copy_to_host(&mut actual, &stream).unwrap();
        assert_eq!(actual, [-1, -1, 2, -1, -1, -1, -1]);
    }

    #[test]
    fn staged_experts_reject_invalid_geometry() {
        let _device = CudaDevice::set_current(0).unwrap();
        let stream = CudaStream::new(0).unwrap();
        let ex = stream.exec(0, 0);
        let input = DeviceBuffer::alloc(0, 65536).unwrap();
        let output = DeviceBuffer::alloc(0, 65536).unwrap();
        let slots = DeviceBuffer::alloc(0, 4).unwrap();
        slots.copy_from_host(&[-1i32], &stream).unwrap();
        stream.synchronize().unwrap();
        for (intermediate, hidden, group, stride) in [
            (64, 128, 128, 16384),  // incomplete down-projection group
            (128, 128, 128, 25089), // misaligned second staged pack
            (128, 128, 128, 16),    // undersized pack
        ] {
            // SAFETY: buffers are live, aligned allocations; negative slot suppresses writes.
            let rc = unsafe {
                ffi::w34_moe_unpack_staged_experts(
                    &ex,
                    input.addr() as *const _,
                    slots.addr() as *const i32,
                    output.addr() as *mut u32,
                    (output.addr() + 32768) as *mut _,
                    (output.addr() + 40960) as *mut u32,
                    (output.addr() + 57344) as *mut _,
                    1,
                    1,
                    stride,
                    intermediate,
                    hidden,
                    group,
                )
            };
            assert_ne!(
                rc, 0,
                "accepted geometry {intermediate}/{hidden}/{group}/{stride}"
            );
        }
        stream.synchronize().unwrap();
    }

    #[test]
    fn tiered_moe_gpu_lru_cpu_hybrid_and_streamed_prefill_parity_tp1_and_ep2() {
        use super::super::{
            DeviceExpertSlotTable, ExpertPackLayout, HostExpertStore, HotExpertCache,
            MoeOffloadMode, TieredMoeEngine, TieredMoeWorkspace,
        };
        use crate::cuda::{f16_to_f32, f32_to_f16};

        let _device = CudaDevice::set_current(0).unwrap();
        let stream = CudaStream::new(0).unwrap();

        let num_experts = 16usize;
        let n_interm = 640usize;
        let k_hidden = 128usize;
        let group_size = 128usize;
        let top_k = 4usize;
        let num_tokens = 6usize;
        let layout = ExpertPackLayout::new(n_interm, k_hidden, group_size).unwrap();

        // Build deterministic synthetic U4B8 weights and scales for 16 experts.
        let gu_words_per_exp = (2 * n_interm) * (k_hidden / 8);
        let gu_scales_per_exp = (2 * n_interm) * (k_hidden / group_size);
        let dn_words_per_exp = k_hidden * (n_interm / 8);
        let dn_scales_per_exp = k_hidden * (n_interm / group_size);

        let mut raw_packs = vec![0u8; num_experts * layout.stride];
        for e in 0..num_experts {
            let gu_p: Vec<u32> = (0..gu_words_per_exp)
                .map(|i| {
                    let mut w = 0u32;
                    for b in 0..8 {
                        let nib = (((e * 13 + i * 7 + b * 3) % 15) as u32) & 0xF;
                        w |= nib << (b * 4);
                    }
                    w
                })
                .collect();
            let gu_s: Vec<u16> = (0..gu_scales_per_exp)
                .map(|i| f32_to_f16(0.015 + 0.002 * (((e * 5 + i) % 9) as f32)))
                .collect();
            let dn_p: Vec<u32> = (0..dn_words_per_exp)
                .map(|i| {
                    let mut w = 0u32;
                    for b in 0..8 {
                        let nib = (((e * 11 + i * 5 + b * 7) % 15) as u32) & 0xF;
                        w |= nib << (b * 4);
                    }
                    w
                })
                .collect();
            let dn_s: Vec<u16> = (0..dn_scales_per_exp)
                .map(|i| f32_to_f16(0.012 + 0.0015 * (((e * 3 + i) % 11) as f32)))
                .collect();
            layout
                .write_pack(
                    &gu_p,
                    &gu_s,
                    &dn_p,
                    &dn_s,
                    &mut raw_packs[e * layout.stride..(e + 1) * layout.stride],
                )
                .unwrap();
        }

        let h_x: Vec<u16> = (0..num_tokens * k_hidden)
            .map(|i| f32_to_f16(((i as f32 * 0.13).sin()) * 0.4))
            .collect();
        let h_shared_out: Vec<u16> = (0..num_tokens * k_hidden)
            .map(|i| f32_to_f16(((i as f32 * 0.07).cos()) * 0.25))
            .collect();
        // Split `shared_out` into two additive partial sums to verify TP=2 shared expert fusion!
        let h_shared_out_r0: Vec<u16> = h_shared_out
            .iter()
            .enumerate()
            .map(|(i, &v)| f32_to_f16(f16_to_f32(v) * if i % 2 == 0 { 0.6 } else { 0.35 }))
            .collect();
        let h_shared_out_r1: Vec<u16> = h_shared_out
            .iter()
            .zip(&h_shared_out_r0)
            .map(|(&full, &r0)| f32_to_f16(f16_to_f32(full) - f16_to_f32(r0)))
            .collect();
        let h_shared_gate: Vec<u16> = (0..num_tokens)
            .map(|t| f32_to_f16(-0.3 + 0.25 * (t as f32)))
            .collect();

        // Route 6 tokens x top_k=4 across 10 distinct experts (`0, 1, 4, 5, 8, 9, 10, 11, 12, 14`),
        // leaving `2, 3, 6, 7, 13, 15` inactive in this batch so LRU eviction of inactive slots is exercised.
        let h_topk_ids: Vec<i32> = vec![
            0, 4, 8, 11, // tok 0
            1, 4, 9, 14, // tok 1
            0, 5, 10, 12, // tok 2
            1, 5, 8, 12, // tok 3
            0, 4, 11, 14, // tok 4
            4, 5, 9, 10, // tok 5
        ];
        let h_topk_weights: Vec<f32> = (0..num_tokens * top_k)
            .map(|i| 0.1 + 0.05 * ((i % 4) as f32))
            .collect();

        let d_x = upload(&h_x, &stream);
        let d_shared_out = upload(&h_shared_out, &stream);
        let d_shared_out_r0 = upload(&h_shared_out_r0, &stream);
        let d_shared_out_r1 = upload(&h_shared_out_r1, &stream);
        let d_shared_gate = upload(&h_shared_gate, &stream);
        let d_topk_ids = upload(&h_topk_ids, &stream);
        let d_topk_weights = upload(&h_topk_weights, &stream);

        // 1. Compute full-resident GPU oracle output (`K_hot = 16` all resident).
        let full_store = HostExpertStore::from_raw_packs(
            layout.clone(),
            0..num_experts,
            &raw_packs,
            num_experts * layout.stride,
        )
        .unwrap();
        let mut ws_oracle =
            TieredMoeWorkspace::new(0, layout.clone(), 8, num_tokens, top_k).unwrap();
        let mut full_cache =
            HotExpertCache::new(0, num_experts, 0..num_experts, layout.clone()).unwrap();
        full_cache
            .populate_initial(&full_store, None, &mut ws_oracle, &stream)
            .unwrap();
        let d_oracle_out =
            DeviceBuffer::alloc(0, num_tokens * k_hidden * std::mem::size_of::<u16>()).unwrap();
        TieredMoeEngine::execute_layer(
            MoeOffloadMode::GpuLru,
            &full_store,
            &mut full_cache,
            &mut ws_oracle,
            &stream,
            d_x.addr(),
            d_topk_ids.addr(),
            d_topk_weights.addr(),
            &h_topk_ids,
            &h_topk_weights,
            d_shared_out.addr(),
            d_shared_gate.addr(),
            d_oracle_out.addr(),
            num_tokens,
            top_k,
        )
        .unwrap();
        let mut h_oracle = vec![0u16; num_tokens * k_hidden];
        d_oracle_out.copy_to_host(&mut h_oracle, &stream).unwrap();

        let assert_close_f16 = |actual: &[u16], label: &str| {
            let max_diff = actual
                .iter()
                .zip(&h_oracle)
                .map(|(&a, &b)| (f16_to_f32(a) - f16_to_f32(b)).abs())
                .fold(0.0f32, f32::max);
            assert!(
                max_diff <= 1.6e-2,
                "{label} max_abs_diff={max_diff} exceeds 1.6e-2 (<= 4 FP16 ULPs)"
            );
        };

        // 2. Verify TP=1 `CpuHybrid { gpu_pcie_budget: 0 }` and `CpuHybrid { gpu_pcie_budget: 2 }`
        //    with only `k_hot = 4` experts initially cached in VRAM!
        for budget in [0usize, 2usize] {
            let mut cache = HotExpertCache::new(0, 4, 0..num_experts, layout.clone()).unwrap();
            let mut ws = TieredMoeWorkspace::new(0, layout.clone(), 3, num_tokens, top_k).unwrap();
            cache
                .populate_initial(&full_store, Some(&[0, 1, 2, 3]), &mut ws, &stream)
                .unwrap();
            let d_out =
                DeviceBuffer::alloc(0, num_tokens * k_hidden * std::mem::size_of::<u16>()).unwrap();
            TieredMoeEngine::execute_layer(
                MoeOffloadMode::CpuHybrid {
                    gpu_pcie_budget: budget,
                },
                &full_store,
                &mut cache,
                &mut ws,
                &stream,
                d_x.addr(),
                d_topk_ids.addr(),
                d_topk_weights.addr(),
                &h_topk_ids,
                &h_topk_weights,
                d_shared_out.addr(),
                d_shared_gate.addr(),
                d_out.addr(),
                num_tokens,
                top_k,
            )
            .unwrap();
            let mut h_out = vec![0u16; num_tokens * k_hidden];
            d_out.copy_to_host(&mut h_out, &stream).unwrap();
            assert!(cache.cpu_cold_evals > 0);
            assert_eq!(cache.dma_experts_loaded, budget as u64);
            assert_close_f16(&h_out, &format!("TP=1 CpuHybrid(budget={budget})"));
        }

        // 3. Verify TP=1 `StreamedPrefill` with `k_hot = 4` and `stage_capacity = 3` (forces 4 cold chunks
        //    through the double-buffered `stage_tables[0..2]` pipeline!).
        {
            let mut cache = HotExpertCache::new(0, 4, 0..num_experts, layout.clone()).unwrap();
            let mut ws = TieredMoeWorkspace::new(0, layout.clone(), 3, num_tokens, top_k).unwrap();
            cache
                .populate_initial(&full_store, Some(&[0, 1, 2, 3]), &mut ws, &stream)
                .unwrap();
            let d_out =
                DeviceBuffer::alloc(0, num_tokens * k_hidden * std::mem::size_of::<u16>()).unwrap();
            TieredMoeEngine::execute_layer(
                MoeOffloadMode::StreamedPrefill,
                &full_store,
                &mut cache,
                &mut ws,
                &stream,
                d_x.addr(),
                d_topk_ids.addr(),
                d_topk_weights.addr(),
                &h_topk_ids,
                &h_topk_weights,
                d_shared_out.addr(),
                d_shared_gate.addr(),
                d_out.addr(),
                num_tokens,
                top_k,
            )
            .unwrap();
            let mut h_out = vec![0u16; num_tokens * k_hidden];
            d_out.copy_to_host(&mut h_out, &stream).unwrap();
            assert_eq!(cache.dma_experts_loaded, 8);
            assert_close_f16(&h_out, "TP=1 StreamedPrefill(stage_cap=3)");
        }

        // 4. Verify TP=1 `GpuLru` with `k_hot = 12` and `stage_capacity = 12` (8 hits + 2 misses -> 2 LRU evictions + 2 DMA loads).
        {
            let mut cache = HotExpertCache::new(0, 12, 0..num_experts, layout.clone()).unwrap();
            let mut ws = TieredMoeWorkspace::new(0, layout.clone(), 12, num_tokens, top_k).unwrap();
            cache
                .populate_initial(&full_store, None, &mut ws, &stream)
                .unwrap();
            let d_out =
                DeviceBuffer::alloc(0, num_tokens * k_hidden * std::mem::size_of::<u16>()).unwrap();
            TieredMoeEngine::execute_layer(
                MoeOffloadMode::GpuLru,
                &full_store,
                &mut cache,
                &mut ws,
                &stream,
                d_x.addr(),
                d_topk_ids.addr(),
                d_topk_weights.addr(),
                &h_topk_ids,
                &h_topk_weights,
                d_shared_out.addr(),
                d_shared_gate.addr(),
                d_out.addr(),
                num_tokens,
                top_k,
            )
            .unwrap();
            let mut h_out = vec![0u16; num_tokens * k_hidden];
            d_out.copy_to_host(&mut h_out, &stream).unwrap();
            assert_eq!(cache.dma_experts_loaded, 2);
            assert_eq!(cache.evictions, 2);
            assert_close_f16(&h_out, "TP=1 GpuLru(k_hot=12)");
        }

        // 4b. Verify pure device-side CUDA-Graph-safe `execute_layer_device_lru` with `k_hot = 12`
        //     (pulling the 2 missed experts over zero-copy PCIe from `full_store.mapped_dev_ptr()`).
        {
            let mut cache = HotExpertCache::new(0, 12, 0..num_experts, layout.clone()).unwrap();
            let mut ws = TieredMoeWorkspace::new(0, layout.clone(), 12, num_tokens, top_k).unwrap();
            cache
                .populate_initial(&full_store, None, &mut ws, &stream)
                .unwrap();
            let d_out =
                DeviceBuffer::alloc(0, num_tokens * k_hidden * std::mem::size_of::<u16>()).unwrap();
            TieredMoeEngine::execute_layer_device_lru(
                &full_store,
                &mut cache,
                &ws,
                &stream,
                d_x.addr(),
                d_topk_ids.addr(),
                d_topk_weights.addr(),
                d_shared_out.addr(),
                d_shared_gate.addr(),
                d_out.addr(),
                num_tokens,
                top_k,
            )
            .unwrap();
            let mut h_out = vec![0u16; num_tokens * k_hidden];
            d_out.copy_to_host(&mut h_out, &stream).unwrap();
            cache.sync_from_device_if_dirty(&stream).unwrap();
            assert!(cache.slot_of(12).is_some());
            assert!(cache.slot_of(14).is_some());
            assert_close_f16(&h_out, "TP=1 DeviceGpuLru(k_hot=12)");
        }

        // 4c. Verify CUDA-Graph-safe CPU+GPU Hybrid Bridge (`register_cpu_bridge` with `k_hot = 6`,
        //     `k_static = 4`: 4 static hot slots in VRAM, 2 GPU PCIe cold staging slots, and remaining
        //     cold experts computed in parallel on the C++ AVX2/FMA worker pool) for `num_tokens = 4`.
        {
            let t_bridge = 4usize;
            let mut cache = HotExpertCache::new(0, 6, 0..num_experts, layout.clone()).unwrap();
            let mut ws = TieredMoeWorkspace::new(0, layout.clone(), 3, num_tokens, top_k).unwrap();
            cache
                .populate_initial(&full_store, Some(&[0, 1, 2, 3, 4, 5]), &mut ws, &stream)
                .unwrap();
            cache
                .register_cpu_bridge(0, &full_store, 4, &stream)
                .unwrap();
            let d_out =
                DeviceBuffer::alloc(0, num_tokens * k_hidden * std::mem::size_of::<u16>()).unwrap();
            TieredMoeEngine::execute_layer_device_lru(
                &full_store,
                &mut cache,
                &ws,
                &stream,
                d_x.addr(),
                d_topk_ids.addr(),
                d_topk_weights.addr(),
                d_shared_out.addr(),
                d_shared_gate.addr(),
                d_out.addr(),
                t_bridge,
                top_k,
            )
            .unwrap();
            let mut h_out = vec![0u16; t_bridge * k_hidden];
            d_out.copy_to_host(&mut h_out, &stream).unwrap();
            let max_diff = h_out
                .iter()
                .zip(&h_oracle[..t_bridge * k_hidden])
                .map(|(&a, &b)| (f16_to_f32(a) - f16_to_f32(b)).abs())
                .fold(0.0f32, f32::max);
            assert!(
                max_diff <= 1.6e-2,
                "TP=1 HybridCpuGpuBridge(k_static=4,k_hot=6) max_abs_diff={max_diff} exceeds 1.6e-2"
            );

            // Also verify CUDA Graph capture + multi-step replay (`replay 1..=3`) over the hybrid bridge!
            let graph = crate::cuda::CudaGraph::capture(&stream, || {
                TieredMoeEngine::execute_layer_device_lru(
                    &full_store,
                    &mut cache,
                    &ws,
                    &stream,
                    d_x.addr(),
                    d_topk_ids.addr(),
                    d_topk_weights.addr(),
                    d_shared_out.addr(),
                    d_shared_gate.addr(),
                    d_out.addr(),
                    t_bridge,
                    top_k,
                )
            })
            .unwrap();
            let zeros = vec![0u16; t_bridge * k_hidden];
            for replay in 1..=3 {
                d_out.copy_from_host(&zeros, &stream).unwrap();
                graph.launch(&stream).unwrap();
                let mut h_graph_out = vec![0u16; t_bridge * k_hidden];
                d_out.copy_to_host(&mut h_graph_out, &stream).unwrap();
                let replay_diff = h_graph_out
                    .iter()
                    .zip(&h_oracle[..t_bridge * k_hidden])
                    .map(|(&a, &b)| (f16_to_f32(a) - f16_to_f32(b)).abs())
                    .fold(0.0f32, f32::max);
                assert!(
                    replay_diff <= 1.6e-2,
                    "TP=1 HybridCpuGpuBridge graph replay {replay} max_abs_diff={replay_diff} exceeds 1.6e-2"
                );
            }

            // And verify `StreamedPrefill` with `bridge_layer_id` active (splitting cold experts
            // across the 16-core AVX2/FMA CPU pool and GPU hardware PCIe DMA) on `num_tokens = 6`!
            TieredMoeEngine::execute_layer(
                MoeOffloadMode::Auto,
                &full_store,
                &mut cache,
                &mut ws,
                &stream,
                d_x.addr(),
                d_topk_ids.addr(),
                d_topk_weights.addr(),
                &h_topk_ids,
                &h_topk_weights,
                d_shared_out.addr(),
                d_shared_gate.addr(),
                d_out.addr(),
                num_tokens,
                top_k,
            )
            .unwrap();
            let mut h_prefill_out = vec![0u16; num_tokens * k_hidden];
            d_out.copy_to_host(&mut h_prefill_out, &stream).unwrap();
            assert!(cache.cpu_cold_evals > 0);
            assert!(cache.dma_experts_loaded > 0);
            assert_close_f16(&h_prefill_out, "TP=1 HybridStreamedPrefill(k_static=4)");
        }

        // 5. Verify TP=2 / EP=2 (`Rank 0: 0..8`, `Rank 1: 8..16`) across `GpuLru`, `CpuHybrid`, and `StreamedPrefill`,
        //    including single-AllReduce fusion of EP=2 routed experts + TP=2 sharded shared_expert!
        let store_r0 = HostExpertStore::from_raw_packs(
            layout.clone(),
            0..8,
            &raw_packs[0..8 * layout.stride],
            8 * layout.stride,
        )
        .unwrap();
        let store_r1 = HostExpertStore::from_raw_packs(
            layout.clone(),
            8..16,
            &raw_packs[8 * layout.stride..16 * layout.stride],
            8 * layout.stride,
        )
        .unwrap();

        for mode in [
            MoeOffloadMode::GpuLru,
            MoeOffloadMode::CpuHybrid { gpu_pcie_budget: 1 },
            MoeOffloadMode::StreamedPrefill,
        ] {
            let k_hot_ep = if matches!(mode, MoeOffloadMode::GpuLru) {
                8
            } else {
                3
            };
            let mut cache_r0 = HotExpertCache::new(0, k_hot_ep, 0..8, layout.clone()).unwrap();
            let mut cache_r1 = HotExpertCache::new(0, k_hot_ep, 8..16, layout.clone()).unwrap();
            let mut ws_r0 =
                TieredMoeWorkspace::new(0, layout.clone(), 4, num_tokens, top_k).unwrap();
            let mut ws_r1 =
                TieredMoeWorkspace::new(0, layout.clone(), 4, num_tokens, top_k).unwrap();
            cache_r0
                .populate_initial(&store_r0, None, &mut ws_r0, &stream)
                .unwrap();
            cache_r1
                .populate_initial(&store_r1, None, &mut ws_r1, &stream)
                .unwrap();

            let d_out_r0 =
                DeviceBuffer::alloc(0, num_tokens * k_hidden * std::mem::size_of::<u16>()).unwrap();
            let d_out_r1 =
                DeviceBuffer::alloc(0, num_tokens * k_hidden * std::mem::size_of::<u16>()).unwrap();

            TieredMoeEngine::execute_layer(
                mode,
                &store_r0,
                &mut cache_r0,
                &mut ws_r0,
                &stream,
                d_x.addr(),
                d_topk_ids.addr(),
                d_topk_weights.addr(),
                &h_topk_ids,
                &h_topk_weights,
                d_shared_out_r0.addr(),
                d_shared_gate.addr(),
                d_out_r0.addr(),
                num_tokens,
                top_k,
            )
            .unwrap();
            TieredMoeEngine::execute_layer(
                mode,
                &store_r1,
                &mut cache_r1,
                &mut ws_r1,
                &stream,
                d_x.addr(),
                d_topk_ids.addr(),
                d_topk_weights.addr(),
                &h_topk_ids,
                &h_topk_weights,
                d_shared_out_r1.addr(),
                d_shared_gate.addr(),
                d_out_r1.addr(),
                num_tokens,
                top_k,
            )
            .unwrap();

            // Simulate the post-MoE TP=2 AllReduce (`out_r0 += out_r1`).
            let ex = stream.exec(0, 0);
            crate::cuda::check_rc(
                // SAFETY: `d_out_r0` and `d_out_r1` are disjoint 4-byte-aligned FP16 buffers.
                unsafe {
                    ffi::w34_add_inplace_f16(
                        &ex,
                        d_out_r0.addr() as *mut _,
                        d_out_r1.addr() as *const _,
                        (num_tokens * k_hidden) as i64,
                    )
                },
                "w34_add_inplace_f16 (EP=2 sum)",
            )
            .unwrap();

            let mut h_ep2 = vec![0u16; num_tokens * k_hidden];
            d_out_r0.copy_to_host(&mut h_ep2, &stream).unwrap();
            assert_close_f16(&h_ep2, &format!("TP=2/EP=2 {mode:?}"));
        }

        // 6. Verify `M = 4` (`num_tokens * top_k = 16 <= num_experts = 16`, `regime == 1` `moe_sparse_vec4_kernel`)
        //    with non-power-of-two `n_interm = 640` (`k_u4_per_row = 20`) against CPU reference.
        {
            let m4 = 4usize;
            let mut tasks = Vec::with_capacity(m4 * top_k);
            for tok in 0..m4 {
                for k in 0..top_k {
                    let idx = tok * top_k + k;
                    tasks.push((tok, h_topk_ids[idx] as usize, h_topk_weights[idx]));
                }
            }
            let h_cpu_m4 = full_store.eval_cold_tasks_cpu(&h_x[..m4 * k_hidden], m4, &tasks);
            let d_out_m4 =
                DeviceBuffer::alloc(0, m4 * k_hidden * std::mem::size_of::<u16>()).unwrap();
            TieredMoeEngine::execute_layer_device_lru(
                &full_store,
                &mut full_cache,
                &ws_oracle,
                &stream,
                d_x.addr(),
                d_topk_ids.addr(),
                d_topk_weights.addr(),
                0,
                0,
                d_out_m4.addr(),
                m4,
                top_k,
            )
            .unwrap();
            let mut h_gpu_m4 = vec![0u16; m4 * k_hidden];
            d_out_m4.copy_to_host(&mut h_gpu_m4, &stream).unwrap();
            let max_diff = h_gpu_m4
                .iter()
                .zip(&h_cpu_m4)
                .map(|(&a, &b)| (f16_to_f32(a) - f16_to_f32(b)).abs())
                .fold(0.0f32, f32::max);
            assert!(
                max_diff <= 1.6e-2,
                "M=4 sparse_vec4 (n_interm=640) max_abs_diff={max_diff} exceeds 1.6e-2"
            );
        }
        let _ = DeviceExpertSlotTable::alloc;
    }

    #[test]
    fn h64_g64_routed_moe_all_regimes_and_tiered_parity() {
        use super::super::{
            ExpertPackLayout, HostExpertStore, HotExpertCache, MoeOffloadMode, TieredMoeEngine,
            TieredMoeWorkspace,
        };
        use crate::cuda::{f16_to_f32, f32_to_f16};

        let _device = CudaDevice::set_current(0).unwrap();
        let stream = CudaStream::new(0).unwrap();
        let ex0 = stream.exec(0, 0);

        // 1. Verify `w34_block_hadamard_inplace_f16` matches `w34_block_hadamard_f16` 0-ULP for B=16 and B=64.
        let rows = 6usize;
        let cols = 128usize;
        let h_in: Vec<u16> = (0..rows * cols)
            .map(|i| f32_to_f16(((i as f32 * 0.19).sin()) * 0.7))
            .collect();
        for block_size in [16i32, 64] {
            let d_src = upload(&h_in, &stream);
            let d_dst = DeviceBuffer::alloc(0, rows * cols * 2).unwrap();
            let d_inp = upload(&h_in, &stream);
            // SAFETY: `d_src` and `d_dst` are disjoint `rows * cols` FP16 buffers; `d_inp` is in-place.
            unsafe {
                assert_eq!(
                    ffi::w34_block_hadamard_f16(
                        &ex0,
                        d_src.addr() as *const _,
                        d_dst.addr() as *mut _,
                        rows as i32,
                        cols as i32,
                        cols as i32,
                        block_size,
                    ),
                    0
                );
                assert_eq!(
                    ffi::w34_block_hadamard_inplace_f16(
                        &ex0,
                        d_inp.addr() as *mut _,
                        rows as i32,
                        cols as i32,
                        block_size,
                    ),
                    0
                );
            }
            let mut h_oop = vec![0u16; rows * cols];
            let mut h_inp = vec![0u16; rows * cols];
            d_dst.copy_to_host(&mut h_oop, &stream).unwrap();
            d_inp.copy_to_host(&mut h_inp, &stream).unwrap();
            assert_eq!(
                h_oop, h_inp,
                "inplace vs out-of-place mismatch for B={block_size}"
            );
        }

        // 2. Verify `group_size = 64` + `down_hadamard_block_size = 64` across all MoE regimes
        //    (`M = 1` GEMV, `M = 4` sparse_vec4, `M = 8` sparse_mma, `M = 20` chunked prefill)
        //    plus `CpuHybrid` and `StreamedPrefill`.
        let num_experts = 16usize;
        let n_interm = 640usize;
        let k_hidden = 128usize;
        let group_size = 64usize;
        let top_k = 2usize;
        let max_m = 20usize;
        let layout = ExpertPackLayout::new(n_interm, k_hidden, group_size)
            .unwrap()
            .with_down_hadamard(64)
            .unwrap();

        let gu_words_per_exp = (2 * n_interm) * (k_hidden / 8);
        let gu_scales_per_exp = (2 * n_interm) * (k_hidden / group_size);
        let dn_words_per_exp = k_hidden * (n_interm / 8);
        let dn_scales_per_exp = k_hidden * (n_interm / group_size);

        let mut raw_packs = vec![0u8; num_experts * layout.stride];
        for e in 0..num_experts {
            let gu_p: Vec<u32> = (0..gu_words_per_exp)
                .map(|i| {
                    let mut w = 0u32;
                    for b in 0..8 {
                        let nib = (((e * 17 + i * 11 + b * 5) % 15) as u32) & 0xF;
                        w |= nib << (b * 4);
                    }
                    w
                })
                .collect();
            let gu_s: Vec<u16> = (0..gu_scales_per_exp)
                .map(|i| f32_to_f16(0.014 + 0.0018 * (((e * 7 + i) % 9) as f32)))
                .collect();
            let dn_p: Vec<u32> = (0..dn_words_per_exp)
                .map(|i| {
                    let mut w = 0u32;
                    for b in 0..8 {
                        let nib = (((e * 19 + i * 3 + b * 7) % 15) as u32) & 0xF;
                        w |= nib << (b * 4);
                    }
                    w
                })
                .collect();
            let dn_s: Vec<u16> = (0..dn_scales_per_exp)
                .map(|i| f32_to_f16(0.011 + 0.0014 * (((e * 5 + i) % 11) as f32)))
                .collect();
            layout
                .write_pack(
                    &gu_p,
                    &gu_s,
                    &dn_p,
                    &dn_s,
                    &mut raw_packs[e * layout.stride..(e + 1) * layout.stride],
                )
                .unwrap();
        }

        let h_x: Vec<u16> = (0..max_m * k_hidden)
            .map(|i| f32_to_f16(((i as f32 * 0.11).cos()) * 0.45))
            .collect();
        let h_topk_ids: Vec<i32> = (0..max_m)
            .flat_map(|t| {
                [
                    (t * 3 % num_experts) as i32,
                    ((t * 5 + 1) % num_experts) as i32,
                ]
            })
            .collect();
        let h_topk_weights: Vec<f32> = (0..max_m * top_k)
            .map(|i| if i % 2 == 0 { 0.65 } else { 0.35 })
            .collect();

        let d_x = upload(&h_x, &stream);
        let d_topk_ids = upload(&h_topk_ids, &stream);
        let d_topk_weights = upload(&h_topk_weights, &stream);

        let full_store = HostExpertStore::from_raw_packs(
            layout.clone(),
            0..num_experts,
            &raw_packs,
            num_experts * layout.stride,
        )
        .unwrap();
        let mut ws = TieredMoeWorkspace::new(0, layout.clone(), 8, max_m, top_k).unwrap();
        let mut full_cache =
            HotExpertCache::new(0, num_experts, 0..num_experts, layout.clone()).unwrap();
        full_cache
            .populate_initial(&full_store, None, &mut ws, &stream)
            .unwrap();

        for m in [1usize, 4, 8, 20] {
            let mut tasks = Vec::with_capacity(m * top_k);
            for tok in 0..m {
                for k in 0..top_k {
                    let idx = tok * top_k + k;
                    tasks.push((tok, h_topk_ids[idx] as usize, h_topk_weights[idx]));
                }
            }
            let h_cpu = full_store.eval_cold_tasks_cpu(&h_x[..m * k_hidden], m, &tasks);
            let d_out = DeviceBuffer::alloc(0, m * k_hidden * 2).unwrap();
            TieredMoeEngine::execute_layer_device_lru(
                &full_store,
                &mut full_cache,
                &ws,
                &stream,
                d_x.addr(),
                d_topk_ids.addr(),
                d_topk_weights.addr(),
                0,
                0,
                d_out.addr(),
                m,
                top_k,
            )
            .unwrap();
            let mut h_gpu = vec![0u16; m * k_hidden];
            d_out.copy_to_host(&mut h_gpu, &stream).unwrap();
            let max_diff = h_gpu
                .iter()
                .zip(&h_cpu)
                .map(|(&a, &b)| (f16_to_f32(a) - f16_to_f32(b)).abs())
                .fold(0.0f32, f32::max);
            assert!(
                max_diff <= 1.6e-2,
                "H64+g64 MoE M={m} max_abs_diff={max_diff} exceeds 1.6e-2"
            );
        }

        // Also verify `CpuHybrid` and `StreamedPrefill` with `k_hot = 4` on `m = 6`.
        let m = 6usize;
        let mut tasks = Vec::with_capacity(m * top_k);
        for tok in 0..m {
            for k in 0..top_k {
                let idx = tok * top_k + k;
                tasks.push((tok, h_topk_ids[idx] as usize, h_topk_weights[idx]));
            }
        }
        let h_cpu = full_store.eval_cold_tasks_cpu(&h_x[..m * k_hidden], m, &tasks);
        for mode in [
            MoeOffloadMode::CpuHybrid { gpu_pcie_budget: 1 },
            MoeOffloadMode::StreamedPrefill,
        ] {
            let mut cache = HotExpertCache::new(0, 4, 0..num_experts, layout.clone()).unwrap();
            let mut ws_small = TieredMoeWorkspace::new(0, layout.clone(), 3, m, top_k).unwrap();
            cache
                .populate_initial(&full_store, Some(&[0, 1, 2, 3]), &mut ws_small, &stream)
                .unwrap();
            let d_out = DeviceBuffer::alloc(0, m * k_hidden * 2).unwrap();
            TieredMoeEngine::execute_layer(
                mode,
                &full_store,
                &mut cache,
                &mut ws_small,
                &stream,
                d_x.addr(),
                d_topk_ids.addr(),
                d_topk_weights.addr(),
                &h_topk_ids[..m * top_k],
                &h_topk_weights[..m * top_k],
                0,
                0,
                d_out.addr(),
                m,
                top_k,
            )
            .unwrap();
            let mut h_gpu = vec![0u16; m * k_hidden];
            d_out.copy_to_host(&mut h_gpu, &stream).unwrap();
            let max_diff = h_gpu
                .iter()
                .zip(&h_cpu)
                .map(|(&a, &b)| (f16_to_f32(a) - f16_to_f32(b)).abs())
                .fold(0.0f32, f32::max);
            assert!(
                max_diff <= 1.6e-2,
                "H64+g64 MoE {mode:?} max_abs_diff={max_diff} exceeds 1.6e-2"
            );
        }

        // Also verify the CUDA-Graph-safe CPU+GPU Hybrid Bridge (`register_cpu_bridge`) with
        // `group_size = 64` and `down_hadamard = 64` across `m = 4` (mailbox) and `m = 6` (prefill).
        assert!(HotExpertCache::cpu_bridge_supported());
        for (m_bridge, use_prefill) in [(4usize, false), (6usize, true)] {
            let mut tasks_b = Vec::with_capacity(m_bridge * top_k);
            for tok in 0..m_bridge {
                for k in 0..top_k {
                    let idx = tok * top_k + k;
                    tasks_b.push((tok, h_topk_ids[idx] as usize, h_topk_weights[idx]));
                }
            }
            let h_cpu_b =
                full_store.eval_cold_tasks_cpu(&h_x[..m_bridge * k_hidden], m_bridge, &tasks_b);
            let mut cache = HotExpertCache::new(0, 6, 0..num_experts, layout.clone()).unwrap();
            let mut ws_b = TieredMoeWorkspace::new(0, layout.clone(), 3, m_bridge, top_k).unwrap();
            cache
                .populate_initial(&full_store, Some(&[0, 1, 2, 3, 4, 5]), &mut ws_b, &stream)
                .unwrap();
            cache
                .register_cpu_bridge(0, &full_store, 4, &stream)
                .unwrap();
            let d_out = DeviceBuffer::alloc(0, m_bridge * k_hidden * 2).unwrap();
            if use_prefill {
                TieredMoeEngine::execute_layer(
                    MoeOffloadMode::StreamedPrefill,
                    &full_store,
                    &mut cache,
                    &mut ws_b,
                    &stream,
                    d_x.addr(),
                    d_topk_ids.addr(),
                    d_topk_weights.addr(),
                    &h_topk_ids[..m_bridge * top_k],
                    &h_topk_weights[..m_bridge * top_k],
                    0,
                    0,
                    d_out.addr(),
                    m_bridge,
                    top_k,
                )
                .unwrap();
            } else {
                TieredMoeEngine::execute_layer_device_lru(
                    &full_store,
                    &mut cache,
                    &ws_b,
                    &stream,
                    d_x.addr(),
                    d_topk_ids.addr(),
                    d_topk_weights.addr(),
                    0,
                    0,
                    d_out.addr(),
                    m_bridge,
                    top_k,
                )
                .unwrap();
            }
            let mut h_gpu = vec![0u16; m_bridge * k_hidden];
            d_out.copy_to_host(&mut h_gpu, &stream).unwrap();
            let max_diff = h_gpu
                .iter()
                .zip(&h_cpu_b)
                .map(|(&a, &b)| (f16_to_f32(a) - f16_to_f32(b)).abs())
                .fold(0.0f32, f32::max);
            assert!(
                max_diff <= 1.6e-2,
                "H64+g64 MoE bridge m={m_bridge} max_abs_diff={max_diff} exceeds 1.6e-2"
            );
        }
    }
}
