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
    #[allow(clippy::cognitive_complexity)]
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

        // 2. Verify TP=1 `StreamedPrefill` with `k_hot = 4` and `stage_capacity = 3` (forces 4 cold chunks
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
            cache.register_cpu_bridge(&full_store, 4, &stream).unwrap();
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

        // 4d. Verify multi-cold-expert hybrid staging (`k_static = 1, k_hot = 9` -> 8 evictable slots)
        //     with 7 distinct cold experts (`5, 8, 9, 10, 11, 12, 14`) and 2 evictable current-step
        //     cache hits (`1` in slot 1, `4` in slot 8): verifies `n_gpu_cold >= 3` pulled on GPU,
        //     `n_cpu_cold >= 3` evaluated on CPU, preservation of current-step cache hits, and
        //     FP16 oracle parity across repeated CUDA Graph replays with mutating routing/inputs.
        {
            let t_bridge = 4usize;
            let mut cache = HotExpertCache::new(0, 9, 0..num_experts, layout.clone()).unwrap();
            let mut ws = TieredMoeWorkspace::new(0, layout.clone(), 8, num_tokens, top_k).unwrap();
            cache
                .populate_initial(
                    &full_store,
                    Some(&[0, 1, 2, 3, 6, 7, 13, 15, 4]),
                    &mut ws,
                    &stream,
                )
                .unwrap();
            cache.register_cpu_bridge(&full_store, 1, &stream).unwrap();
            let d_out =
                DeviceBuffer::alloc(0, t_bridge * k_hidden * std::mem::size_of::<u16>()).unwrap();
            let d_x_mut = upload(&h_x[..t_bridge * k_hidden], &stream);
            let d_ids_mut = upload(&h_topk_ids[..t_bridge * top_k], &stream);

            let graph = crate::cuda::CudaGraph::capture(&stream, || {
                TieredMoeEngine::execute_layer_device_lru(
                    &full_store,
                    &mut cache,
                    &ws,
                    &stream,
                    d_x_mut.addr(),
                    d_ids_mut.addr(),
                    d_topk_weights.addr(),
                    d_shared_out.addr(),
                    d_shared_gate.addr(),
                    d_out.addr(),
                    t_bridge,
                    top_k,
                )
            })
            .unwrap();

            for replay in 1..=3usize {
                // Reset initial 9-slot cache before each replay and mutate routing/input scale
                // so every replay exercises >= 6 distinct cold misses (`n_gpu_cold >= 3`) while
                // keeping evictable hits (`1` in slot 1, `4` in slot 8) active in the step.
                let init_slots = if replay % 2 == 1 {
                    [0usize, 1, 2, 3, 6, 7, 13, 15, 4]
                } else {
                    [0usize, 1, 5, 8, 9, 10, 11, 12, 4]
                };
                cache
                    .populate_initial(&full_store, Some(&init_slots), &mut ws, &stream)
                    .unwrap();
                let step_ids: Vec<i32> = if replay % 2 == 1 {
                    // Hits: 0 (slot 0), 1 (slot 1), 4 (slot 8); 7 cold misses: 5, 8, 9, 10, 11, 12, 14
                    vec![
                        0, 4, 8, 11, // tok 0
                        1, 4, 9, 14, // tok 1
                        0, 5, 10, 12, // tok 2
                        1, 5, 8, 12, // tok 3
                    ]
                } else {
                    // Hits: 0 (slot 0), 1 (slot 1), 4 (slot 8); 7 cold misses: 2, 3, 6, 7, 13, 14, 15
                    vec![
                        0, 4, 2, 6, // tok 0
                        1, 4, 3, 7, // tok 1
                        0, 13, 14, 15, // tok 2
                        1, 2, 13, 15, // tok 3
                    ]
                };
                let scale = 1.0f32 + 0.125f32 * ((replay - 1) as f32);
                let step_x: Vec<u16> = h_x[..t_bridge * k_hidden]
                    .iter()
                    .map(|&v| f32_to_f16(f16_to_f32(v) * scale))
                    .collect();
                d_x_mut.copy_from_host(&step_x, &stream).unwrap();
                d_ids_mut.copy_from_host(&step_ids, &stream).unwrap();

                let d_step_oracle =
                    DeviceBuffer::alloc(0, t_bridge * k_hidden * std::mem::size_of::<u16>())
                        .unwrap();
                TieredMoeEngine::execute_layer(
                    MoeOffloadMode::GpuLru,
                    &full_store,
                    &mut full_cache,
                    &mut ws_oracle,
                    &stream,
                    d_x_mut.addr(),
                    d_ids_mut.addr(),
                    d_topk_weights.addr(),
                    &step_ids,
                    &h_topk_weights[..t_bridge * top_k],
                    d_shared_out.addr(),
                    d_shared_gate.addr(),
                    d_step_oracle.addr(),
                    t_bridge,
                    top_k,
                )
                .unwrap();
                let mut h_step_oracle = vec![0u16; t_bridge * k_hidden];
                d_step_oracle
                    .copy_to_host(&mut h_step_oracle, &stream)
                    .unwrap();

                graph.launch(&stream).unwrap();
                let mut h_graph_out = vec![0u16; t_bridge * k_hidden];
                d_out.copy_to_host(&mut h_graph_out, &stream).unwrap();
                let mut misses = [0i32; 2];
                ws.d_num_misses.copy_to_host(&mut misses, &stream).unwrap();
                cache.device_lru_dirty = true;
                cache.sync_from_device_if_dirty(&stream).unwrap();

                let (n_gpu_cold, n_cpu_cold) = (misses[0], misses[1]);
                assert!(
                    (3..=8).contains(&n_gpu_cold),
                    "replay {replay}: expected n_gpu_cold in 3..=8, got {n_gpu_cold}"
                );
                assert!(
                    n_cpu_cold >= 1,
                    "replay {replay}: expected n_cpu_cold >= 1, got {n_cpu_cold}"
                );
                assert_eq!(
                    n_gpu_cold + n_cpu_cold,
                    7,
                    "replay {replay}: expected 7 total cold experts partitioned across GPU+CPU"
                );
                // Current-step cache hits (static 0 in slot 0, evictable 1 in slot 1, evictable 4 in slot 8)
                // must never be evicted during the same step.
                assert_eq!(cache.slot_of(0), Some(0));
                assert_eq!(cache.slot_of(1), Some(1));
                assert_eq!(cache.slot_of(4), Some(8));

                let replay_diff = h_graph_out
                    .iter()
                    .zip(&h_step_oracle)
                    .map(|(&a, &b)| (f16_to_f32(a) - f16_to_f32(b)).abs())
                    .fold(0.0f32, f32::max);
                assert!(
                    replay_diff <= 1.6e-2,
                    "TP=1 HybridCpuGpuBridge(8 evictable slots) replay {replay} max_abs_diff={replay_diff} exceeds 1.6e-2"
                );
            }
        }

        // 5. Verify TP=2 / EP=2 (`Rank 0: 0..8`, `Rank 1: 8..16`) across `GpuLru` and `StreamedPrefill`,
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

        for mode in [MoeOffloadMode::GpuLru, MoeOffloadMode::StreamedPrefill] {
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
    #[allow(clippy::cognitive_complexity)]
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

            // Also verify 0-ULP bit-identity of epilogue-fused H_64 (`down_hadamard_block_size = 64`)
            // in `d_interm` vs unfused Stage-1 (`down_hadamard_block_size = 0`) + standalone
            // `w34_block_hadamard_inplace_f16(..., 64)` across all 4 regimes (`M = 1, 4, 8, 16`):
            let chunk_m = m.min(16);
            let d_interm_fused = DeviceBuffer::alloc(0, chunk_m * top_k * n_interm * 2).unwrap();
            let d_interm_unfused = DeviceBuffer::alloc(0, chunk_m * top_k * n_interm * 2).unwrap();
            let d_out_tmp = DeviceBuffer::alloc(0, chunk_m * k_hidden * 2).unwrap();
            let ex_ws = stream.exec(ws.d_scratch.addr(), ws.d_scratch.len_bytes);
            // SAFETY: All device pointers match `full_cache` and `ws` dimensions for `chunk_m`.
            unsafe {
                assert_eq!(
                    ffi::w34_moe_w4a16_mlp_ex_f16(
                        &ex_ws,
                        d_x.addr() as *const _,
                        full_cache.slots.gate_up_packed.addr() as *const _,
                        full_cache.slots.gate_up_scales.addr() as *const _,
                        full_cache.slots.down_packed.addr() as *const _,
                        full_cache.slots.down_scales.addr() as *const _,
                        d_topk_ids.addr() as *const _,
                        d_topk_weights.addr() as *const _,
                        core::ptr::null(),
                        core::ptr::null(),
                        d_interm_fused.addr() as *mut _,
                        d_out_tmp.addr() as *mut _,
                        chunk_m as i32,
                        top_k as i32,
                        num_experts as i32,
                        n_interm as i32,
                        k_hidden as i32,
                        group_size as i32,
                        64,
                    ),
                    0
                );
                assert_eq!(
                    ffi::w34_moe_w4a16_mlp_ex_f16(
                        &ex_ws,
                        d_x.addr() as *const _,
                        full_cache.slots.gate_up_packed.addr() as *const _,
                        full_cache.slots.gate_up_scales.addr() as *const _,
                        full_cache.slots.down_packed.addr() as *const _,
                        full_cache.slots.down_scales.addr() as *const _,
                        d_topk_ids.addr() as *const _,
                        d_topk_weights.addr() as *const _,
                        core::ptr::null(),
                        core::ptr::null(),
                        d_interm_unfused.addr() as *mut _,
                        d_out_tmp.addr() as *mut _,
                        chunk_m as i32,
                        top_k as i32,
                        num_experts as i32,
                        n_interm as i32,
                        k_hidden as i32,
                        group_size as i32,
                        0,
                    ),
                    0
                );
                assert_eq!(
                    ffi::w34_block_hadamard_inplace_f16(
                        &ex0,
                        d_interm_unfused.addr() as *mut _,
                        (chunk_m * top_k) as i32,
                        n_interm as i32,
                        64,
                    ),
                    0
                );
            }
            let mut h_interm_fused = vec![0u16; chunk_m * top_k * n_interm];
            let mut h_interm_unfused = vec![0u16; chunk_m * top_k * n_interm];
            d_interm_fused
                .copy_to_host(&mut h_interm_fused, &stream)
                .unwrap();
            d_interm_unfused
                .copy_to_host(&mut h_interm_unfused, &stream)
                .unwrap();
            assert_eq!(
                h_interm_fused, h_interm_unfused,
                "epilogue-fused H_64 interm_buf 0-ULP mismatch at chunk_m={chunk_m}"
            );
        }

        // Also verify `GpuLru` and `StreamedPrefill` with `k_hot = 4` on `m = 6`.
        let m = 6usize;
        let mut tasks = Vec::with_capacity(m * top_k);
        for tok in 0..m {
            for k in 0..top_k {
                let idx = tok * top_k + k;
                tasks.push((tok, h_topk_ids[idx] as usize, h_topk_weights[idx]));
            }
        }
        let h_cpu = full_store.eval_cold_tasks_cpu(&h_x[..m * k_hidden], m, &tasks);
        for mode in [MoeOffloadMode::GpuLru, MoeOffloadMode::StreamedPrefill] {
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
            cache.register_cpu_bridge(&full_store, 4, &stream).unwrap();
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

        // Verify the `n_interm = 640, top_k = 10` (`<false, 80, 10>`) down-accum specialization
        // across `M = 1..=3` both end-to-end (`execute_layer_device_lru`) and with `interm_in`
        // populated exclusively in tail words `64..80` (`cols 512..639`) so omitting or
        // misindexing the guarded third warp iteration (`iter = 2`) fails the test.
        let top_k_10 = 10usize;
        let max_m_10 = 3usize;
        let h_ids_10: Vec<i32> = (0..max_m_10)
            .flat_map(|t| (0..top_k_10).map(move |k| ((t * 3 + k) % num_experts) as i32))
            .collect();
        let h_wts_10: Vec<f32> = (0..max_m_10 * top_k_10)
            .map(|i| 0.05 + 0.01 * ((i % top_k_10) as f32))
            .collect();
        let d_ids_10 = upload(&h_ids_10, &stream);
        let d_wts_10 = upload(&h_wts_10, &stream);
        let ws_10 = TieredMoeWorkspace::new(0, layout.clone(), 8, max_m_10, top_k_10).unwrap();
        let ex_10 = stream.exec(ws_10.d_scratch.addr(), ws_10.d_scratch.len_bytes);

        for m_10 in 1..=max_m_10 {
            let mut tasks_10 = Vec::with_capacity(m_10 * top_k_10);
            for tok in 0..m_10 {
                for k in 0..top_k_10 {
                    let idx = tok * top_k_10 + k;
                    tasks_10.push((tok, h_ids_10[idx] as usize, h_wts_10[idx]));
                }
            }
            let h_cpu_10 = full_store.eval_cold_tasks_cpu(&h_x[..m_10 * k_hidden], m_10, &tasks_10);
            let d_out_10 = DeviceBuffer::alloc(0, m_10 * k_hidden * 2).unwrap();
            TieredMoeEngine::execute_layer_device_lru(
                &full_store,
                &mut full_cache,
                &ws_10,
                &stream,
                d_x.addr(),
                d_ids_10.addr(),
                d_wts_10.addr(),
                0,
                0,
                d_out_10.addr(),
                m_10,
                top_k_10,
            )
            .unwrap();
            let mut h_gpu_10 = vec![0u16; m_10 * k_hidden];
            d_out_10.copy_to_host(&mut h_gpu_10, &stream).unwrap();
            let max_diff_10 = h_gpu_10
                .iter()
                .zip(&h_cpu_10)
                .map(|(&a, &b)| (f16_to_f32(a) - f16_to_f32(b)).abs())
                .fold(0.0f32, f32::max);
            assert!(
                max_diff_10 <= 1.6e-2,
                "H64+g64 top_k=10 M={m_10} max_abs_diff={max_diff_10} exceeds 1.6e-2"
            );

            // Tail-only (`words 64..80` = `cols 512..639`) direct `w34_moe_w4a16_down_accum_f16` oracle:
            let mut h_interm_tail = vec![0u16; m_10 * top_k_10 * n_interm];
            for row in 0..(m_10 * top_k_10) {
                for col in 512..n_interm {
                    h_interm_tail[row * n_interm + col] =
                        f32_to_f16(0.25 + 0.03 * (((row * 7 + col) % 13) as f32));
                }
            }
            let d_interm_tail = upload(&h_interm_tail, &stream);
            let d_tail_out = DeviceBuffer::alloc(0, m_10 * k_hidden * 2).unwrap();
            assert_eq!(
                // SAFETY: `d_interm_tail`, `full_cache.slots`, `d_ids_10`, `d_wts_10`, and `d_tail_out`
                // are valid device buffers sized for `(m_10, top_k_10 = 10, n_interm = 640, k_hidden = 128)`.
                unsafe {
                    ffi::w34_moe_w4a16_down_accum_f16(
                        &ex_10,
                        d_interm_tail.addr() as *const _,
                        full_cache.slots.down_packed.addr() as *const _,
                        full_cache.slots.down_scales.addr() as *const _,
                        d_ids_10.addr() as *const _,
                        d_wts_10.addr() as *const _,
                        core::ptr::null(),
                        core::ptr::null(),
                        d_tail_out.addr() as *mut _,
                        m_10 as i32,
                        top_k_10 as i32,
                        num_experts as i32,
                        k_hidden as i32,
                        n_interm as i32,
                        group_size as i32,
                    )
                },
                0
            );
            let mut h_tail_gpu = vec![0u16; m_10 * k_hidden];
            d_tail_out.copy_to_host(&mut h_tail_gpu, &stream).unwrap();
            stream.synchronize().unwrap();

            let mut h_tail_ref = vec![0.0f32; m_10 * k_hidden];
            let words_per_interm = n_interm / 8;
            let groups_per_interm = n_interm / group_size;
            let words_per_group = group_size / 8;
            for tok in 0..m_10 {
                for k in 0..top_k_10 {
                    let pair = tok * top_k_10 + k;
                    let exp = h_ids_10[pair] as usize;
                    let tw_h = f16_to_f32(f32_to_f16(h_wts_10[pair]));
                    let pack = &raw_packs[exp * layout.stride..(exp + 1) * layout.stride];
                    let dn_p = &pack[layout.down_packed.clone()];
                    let dn_s = &pack[layout.down_scales.clone()];
                    let interm_row = &h_interm_tail[pair * n_interm..(pair + 1) * n_interm];
                    for h in 0..k_hidden {
                        let mut dot = 0.0f32;
                        let row_w = h * words_per_interm;
                        let row_s = h * groups_per_interm;
                        for g in 8..groups_per_interm {
                            let ds_off = (row_s + g) * 2;
                            let ds =
                                f16_to_f32(u16::from_le_bytes([dn_s[ds_off], dn_s[ds_off + 1]]));
                            for w in 0..words_per_group {
                                let w_idx = g * words_per_group + w;
                                let dw_off = (row_w + w_idx) * 4;
                                let dw = u32::from_le_bytes([
                                    dn_p[dw_off],
                                    dn_p[dw_off + 1],
                                    dn_p[dw_off + 2],
                                    dn_p[dw_off + 3],
                                ]);
                                let k_base = w_idx * 8;
                                for b in 0..8 {
                                    let x_val = f16_to_f32(interm_row[k_base + b]);
                                    let dnib = ((dw >> (b * 4)) & 0xF) as i32 - 8;
                                    let dw_h = f16_to_f32(f32_to_f16((dnib as f32) * ds));
                                    dot += dw_h * x_val;
                                }
                            }
                        }
                        h_tail_ref[tok * k_hidden + h] += dot * tw_h;
                    }
                }
            }
            let tail_norm: f32 = h_tail_ref.iter().map(|v| v.abs()).sum();
            assert!(tail_norm > 1.0, "expected non-trivial tail contribution");
            let tail_diff = h_tail_gpu
                .iter()
                .zip(&h_tail_ref)
                .map(|(&g, &r)| (f16_to_f32(g) - f16_to_f32(f32_to_f16(r))).abs())
                .fold(0.0f32, f32::max);
            assert!(
                tail_diff <= 1e-2,
                "n_interm=640 top_k=10 tail words 64..80 M={m_10} max_abs_diff={tail_diff} exceeds 1e-2"
            );
        }
    }

    #[test]
    #[allow(clippy::cognitive_complexity)]
    fn speculative_lru_bip_and_streamed_prefill_promotion() {
        use super::super::{
            ExpertPackLayout, HostExpertStore, HotExpertCache, MoeOffloadMode, TieredMoeEngine,
            TieredMoeWorkspace,
        };
        use crate::cuda::f32_to_f16;

        let _device = CudaDevice::set_current(0).unwrap();
        let stream = CudaStream::new(0).unwrap();

        let num_experts = 80usize;
        let k_hot = 64usize;
        let n_interm = 128usize;
        let k_hidden = 128usize;
        let group_size = 32usize;
        let top_k = 10usize;
        let m_verify = 8usize;
        let layout = ExpertPackLayout::new(n_interm, k_hidden, group_size).unwrap();

        let raw_packs = vec![0u8; num_experts * layout.stride];
        let full_store = HostExpertStore::from_raw_packs(
            layout.clone(),
            0..num_experts,
            &raw_packs,
            num_experts * layout.stride,
        )
        .unwrap();
        let mut ws = TieredMoeWorkspace::new(0, layout.clone(), 32, 1024, top_k).unwrap();
        let mut cache = HotExpertCache::new(0, k_hot, 0..num_experts, layout.clone()).unwrap();
        cache
            .populate_initial(&full_store, None, &mut ws, &stream)
            .unwrap();

        // Step 1: M=4 verify batch (`total_pairs = 40`, `k_protected = k_hot - 40 = 24`)
        // followed by M=8 verify batch (`total_pairs = 80`).
        let mut h_topk_m4: Vec<i32> = (0..4 * top_k).map(|i| (i % 10) as i32).collect();
        h_topk_m4[0] = 64;
        h_topk_m4[20] = 65;
        h_topk_m4[21] = 68;
        h_topk_m4[22] = 69;
        h_topk_m4[30] = 65;
        let h_wts = vec![0.1f32; 1024 * top_k];
        let h_x = vec![f32_to_f16(0.1); 1024 * k_hidden];
        let d_x = upload(&h_x, &stream);
        let d_topk_m4 = upload(&h_topk_m4, &stream);
        let d_wts = upload(&h_wts, &stream);
        let d_out = DeviceBuffer::alloc(0, 1024 * k_hidden * 2).unwrap();

        TieredMoeEngine::execute_layer_device_lru(
            &full_store,
            &mut cache,
            &ws,
            &stream,
            d_x.addr(),
            d_topk_m4.addr(),
            d_wts.addr(),
            0,
            0,
            d_out.addr(),
            4,
            top_k,
        )
        .unwrap();
        cache.sync_from_device_if_dirty(&stream).unwrap();

        // Protected prefix `0..24` (`k_hot - 40`) must remain untouched (`slot_to_expert[s] == s`).
        for s in 0..24 {
            assert_eq!(
                cache.slot_to_expert[s], s as i32,
                "protected slot {s} was evicted"
            );
        }
        let slot_64 = cache.slot_of(64).expect("anchor cold expert 64 loaded") as usize;
        let slot_65 = cache
            .slot_of(65)
            .expect("multi-token cold expert 65 loaded") as usize;
        let slot_68 = cache.slot_of(68).expect("tail cold expert 68 loaded") as usize;
        let slot_69 = cache.slot_of(69).expect("tail cold expert 69 loaded") as usize;
        assert_ne!(slot_68, slot_69);
        for s in [slot_64, slot_65, slot_68, slot_69] {
            assert!(
                cache.slot_last_used[s] > k_hot as u64,
                "loaded miss must get updated step timestamp"
            );
        }

        // Step 2: M=8 verify batch (`total_pairs = 80`) with cold misses `66` and `67`.
        // Even with `total_pairs = 80 > k_hot = 64`, `tail_budget = min(40, k_hot / 2) = 32`
        // protects slots `0..32` (`k_protected = 32`) from speculative tree misses!
        let mut h_topk_s2: Vec<i32> = (0..m_verify * top_k).map(|i| (i % 10) as i32).collect();
        h_topk_s2[60] = 66;
        h_topk_s2[70] = 67;
        let d_topk_s2 = upload(&h_topk_s2, &stream);
        TieredMoeEngine::execute_layer_device_lru(
            &full_store,
            &mut cache,
            &ws,
            &stream,
            d_x.addr(),
            d_topk_s2.addr(),
            d_wts.addr(),
            0,
            0,
            d_out.addr(),
            m_verify,
            top_k,
        )
        .unwrap();
        cache.sync_from_device_if_dirty(&stream).unwrap();
        for s in 0..32 {
            assert_eq!(
                cache.slot_to_expert[s], s as i32,
                "protected slot {s} was evicted during M=8 verify when k_hot=64 < total_pairs=80"
            );
        }
        assert!(cache.slot_of(66).is_some());
        assert!(cache.slot_of(67).is_some());
        assert_ne!(cache.slot_of(66), cache.slot_of(67));
        assert!(
            cache.slot_of(64).is_some(),
            "anchor expert 64 must remain resident"
        );
        assert!(
            cache.slot_of(65).is_some(),
            "multi-token expert 65 must remain resident"
        );
        let mut d_prefill_s2 = vec![0i32; num_experts];
        cache
            .d_prefill_expert_to_slot
            .copy_to_host(&mut d_prefill_s2, &stream)
            .unwrap();
        stream.synchronize().unwrap();
        for (le, &s) in cache.expert_to_slot.iter().enumerate() {
            let expected = if s >= 0 && (s as usize) < cache.prefill_hot_limit() {
                s
            } else {
                -1
            };
            assert_eq!(
                d_prefill_s2[le], expected,
                "d_prefill_expert_to_slot[{le}] stale after M=8 device-side LRU sync"
            );
        }

        // Step 3: Verify `StreamedPrefill` (`num_tokens = 1024`, `!cache.has_cpu_bridge()`)
        // promotes a cold expert (`70`) routed 32 times in the prompt into `cache.slots`.
        let m_prefill = 1024usize;
        let mut h_topk_pf: Vec<i32> = (0..m_prefill * top_k).map(|i| (i % 10) as i32).collect();
        for tok in 0..32 {
            h_topk_pf[tok * top_k] = 70;
        }
        let d_topk_pf = upload(&h_topk_pf, &stream);
        TieredMoeEngine::execute_layer(
            MoeOffloadMode::StreamedPrefill,
            &full_store,
            &mut cache,
            &mut ws,
            &stream,
            d_x.addr(),
            d_topk_pf.addr(),
            d_wts.addr(),
            &h_topk_pf,
            &h_wts,
            0,
            0,
            d_out.addr(),
            m_prefill,
            top_k,
        )
        .unwrap();
        let slot_70 = cache
            .slot_of(70)
            .expect("prompt-frequent cold expert 70 must be promoted during StreamedPrefill")
            as usize;
        assert_eq!(
            cache.slot_last_used[slot_70], cache.step_clock,
            "promoted prompt expert must have updated slot_last_used"
        );
        for s in 0..k_hot {
            let ge = cache.slot_to_expert[s] as usize;
            assert_eq!(
                cache.expert_to_slot[ge], s as i32,
                "expert_to_slot[{ge}] != {s} after StreamedPrefill promotion"
            );
        }
        assert!(
            cache.adapted_slots_dirty,
            "StreamedPrefill promotion must mark adapted_slots_dirty"
        );

        // Step 4: A subsequent short prompt (`num_tokens = 128 < 1024`) must restore
        // `0..k_static` back to `initial_slot_to_expert` so long-prompt / needle promotions
        // do not permanently displace general-purpose ranked experts.
        let m_short = 128usize;
        TieredMoeEngine::execute_layer(
            MoeOffloadMode::StreamedPrefill,
            &full_store,
            &mut cache,
            &mut ws,
            &stream,
            d_x.addr(),
            d_topk_pf.addr(),
            d_wts.addr(),
            &h_topk_pf[..m_short * top_k],
            &h_wts[..m_short * top_k],
            0,
            0,
            d_out.addr(),
            m_short,
            top_k,
        )
        .unwrap();
        assert!(!cache.adapted_slots_dirty);
        assert_eq!(cache.slot_to_expert, (0..k_hot as i32).collect::<Vec<_>>());
        let mut d_prefill_s4 = vec![0i32; num_experts];
        cache
            .d_prefill_expert_to_slot
            .copy_to_host(&mut d_prefill_s4, &stream)
            .unwrap();
        stream.synchronize().unwrap();
        for (le, &s) in cache.expert_to_slot.iter().enumerate() {
            let expected = if s >= 0 && (s as usize) < cache.prefill_hot_limit() {
                s
            } else {
                -1
            };
            assert_eq!(
                d_prefill_s4[le], expected,
                "d_prefill_expert_to_slot[{le}] stale after short-prompt restoration"
            );
        }
    }
}
