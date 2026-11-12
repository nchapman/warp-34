#![cfg(feature = "cuda")]
mod common;
use w34_host::{
    cache::KvFormatPolicy,
    dflash::{DFlash2Config, DFlash2Rank, DraftCandidates},
    model::Qwen3_5Engine,
};
use w34_kernels::ffi::{W34_KV_FORMAT_BF16, W34_KV_NAMESPACE_DFLASH_DRAFT_CTX};
#[test]
fn cached_dflash_configs_and_candidate_merge() {
    for (repo, block, taps) in [
        ("nchapman/Qwen3.5-4B-DFlash2", 16, vec![1, 9, 16, 23, 28]),
        ("incoai/Qwen3.8-27B-DFlash2", 8, vec![5, 19, 33, 47, 61]),
    ] {
        let Some(dir) =
            common::require_or_skip_cached_model(repo, "cached_dflash_configs_and_candidate_merge")
        else {
            continue;
        };
        let cfg = DFlash2Config::load(&dir).unwrap();
        assert_eq!(cfg.block_size, block);
        assert_eq!(cfg.target_layers, taps);
    }
    if let Ok(dir) =
        w34_host::loader::resolve_hf_model_dir("nchapman/Qwen3.8-Flash-Next-DFlash2-W4A16-HGPTQ")
    {
        let cfg = DFlash2Config::load(&dir).unwrap();
        assert_eq!(cfg.block_size, block_size_expected());
        assert_eq!(cfg.target_layers, vec![5, 15, 25, 35, 45]);
        assert_eq!(cfg.model.hidden_size, 2560);
        assert_eq!(cfg.model.head_dim, 128);
    }
    let a = DraftCandidates {
        ids: vec![0, 1, 2, 3],
        scores: vec![8., 3., 9., 5.],
        rows: 2,
        top_k: 2,
    };
    let b = DraftCandidates {
        ids: vec![4, 5, 6, 7],
        scores: vec![9., 7., 9., 6.],
        rows: 2,
        top_k: 2,
    };
    let global = DraftCandidates::merge(&[a, b]).unwrap();
    assert_eq!(global.ids, [4, 0, 2, 6]);
    assert_eq!(global.scores, [9., 8., 9., 9.]);
}
#[inline]
fn block_size_expected() -> usize {
    8
}

#[test]
fn rejects_selector_top_k_beyond_native_topk_limit() {
    let Some(source) = common::require_or_skip_cached_model(
        "nchapman/Qwen3.5-4B-DFlash2",
        "rejects_selector_top_k_beyond_native_topk_limit",
    ) else {
        return;
    };
    let mut config: serde_json::Value =
        serde_json::from_slice(&std::fs::read(source.join("config.json")).unwrap()).unwrap();
    let temp = std::env::temp_dir().join(format!("w34-dflash-topk-{}", std::process::id()));
    std::fs::create_dir(&temp).unwrap();
    for (top_k, supported) in [(16, true), (17, false), (32, false)] {
        if config.get("dflash_config").is_some() {
            config["dflash_config"]["selector_top_k"] = top_k.into();
        } else {
            config["selector_top_k"] = top_k.into();
        }
        std::fs::write(temp.join("config.json"), config.to_string()).unwrap();
        let result = DFlash2Config::load(&temp);
        // Clean before asserting so a failed regression leaves no stale directory.
        if result.is_ok() != supported {
            std::fs::remove_dir_all(&temp).unwrap();
            panic!("selector_top_k={top_k}: expected supported={supported}, got {result:?}");
        }
    }
    std::fs::remove_dir_all(temp).unwrap();
}
#[test]
fn cached_4b_native_bf16_dflash_block() {
    run_4b(false);
}
#[test]
fn actual_sglang_full_model_hidden_parity() {
    run_4b(true);
}
static GPU_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

fn assert_4b_tap_parity(target_dir: &std::path::Path, rel: f64, energy: f64) {
    if target_dir.ends_with("851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a") {
        assert!(
            rel.is_finite() && energy > 0.0 && rel < 0.20,
            "target tap drift exceeds unquantized 4B oracle budget: {rel}"
        );
    } else {
        let strict = ["W34_STRICT_TESTS", "CI"].into_iter().any(|var| {
            std::env::var(var).is_ok_and(|val| {
                let t = val.trim();
                t == "1" || t.eq_ignore_ascii_case("true")
            })
        });
        assert!(
            !strict,
            "strict mode requires pinned 4B provenance snapshot 851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a, got {}",
            target_dir.display()
        );
        assert!(
            rel.is_finite() && energy > 0.0 && rel < 0.20,
            "target taps drift {rel} or energy {energy} out of bounds on {}",
            target_dir.display()
        );
    }
}

fn resolve_4b_target_dir(strict_parity: bool) -> Option<std::path::PathBuf> {
    let resolved = common::require_or_skip_cached_model("Qwen/Qwen3.5-4B", "run_4b")?;
    if !strict_parity {
        return Some(resolved);
    }
    let provenance_snap = resolved
        .parent()
        .map(|p| p.join("851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a"))
        .expect("HF snapshot parent dir must exist");
    assert!(
        provenance_snap
            .join("model.safetensors.index.json")
            .exists(),
        "missing 4B provenance snapshot 851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a at {}; run `mise run test-prep`",
        provenance_snap.display()
    );
    Some(provenance_snap)
}

fn run_4b(strict_parity: bool) {
    let _gpu_guard = GPU_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    let Some(dir) = common::require_or_skip_cached_model("nchapman/Qwen3.5-4B-DFlash2", "run_4b")
    else {
        return;
    };
    let cfg = DFlash2Config::load(&dir).unwrap();
    let Some(target_dir) = resolve_4b_target_dir(strict_parity) else {
        return;
    };
    let mut engine = Qwen3_5Engine::load(
        target_dir.to_str().unwrap(),
        1,
        KvFormatPolicy::Fp16,
        Some(8),
        false,
    )
    .unwrap();
    engine.ranks[0]
        .configure_speculative_taps(&cfg.target_layers, false, 64)
        .unwrap();
    let tokens = engine.encode("The capital of France is").unwrap();
    let page = engine.alloc_kv_page().unwrap();
    let state = engine.alloc_state_slot().unwrap();
    let (anchor, _, _) = engine
        .step_tokens(&tokens, 0, &[page], state, state, false, None)
        .unwrap();
    let target = &engine.ranks[0];
    let mut draft = DFlash2Rank::load(target, &dir, 64, 64).unwrap();
    for l in draft.kv_layouts() {
        assert_eq!(l.namespace_id, W34_KV_NAMESPACE_DFLASH_DRAFT_CTX);
        assert_eq!(l.k_format, W34_KV_FORMAT_BF16);
    }
    let taps = target.tap_aux_bf16.as_ref().unwrap().addr();
    if strict_parity {
        let root =
            std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../tools/oracle/goldens");
        let golden = w34_host::loader::SafetensorsArchive::open(
            &root.join("sglang_dflash_model_4b.safetensors"),
        )
        .unwrap();
        let expected_taps = golden.read_bytes("input.taps_bf16").unwrap();
        let mut actual_taps = vec![0u8; tokens.len() * 5 * engine.cfg.hidden_size * 2];
        let tap_buffer = target.tap_aux_bf16.as_ref().unwrap();
        tap_buffer
            .copy_to_host(&mut actual_taps, &target.stream)
            .unwrap();
        assert_eq!(actual_taps.len(), expected_taps.len());
        let mut error = 0.0f64;
        let mut energy = 0.0f64;
        let mut changed = 0usize;
        for (actual, expected) in actual_taps
            .as_chunks::<2>()
            .0
            .iter()
            .zip(expected_taps.as_chunks::<2>().0.iter())
        {
            let a = f32::from_bits((u16::from_le_bytes([actual[0], actual[1]]) as u32) << 16);
            let b = f32::from_bits((u16::from_le_bytes([expected[0], expected[1]]) as u32) << 16);
            error += f64::from(a - b).powi(2);
            energy += f64::from(b).powi(2);
            changed += usize::from(actual != expected);
        }
        let rel = (error / energy).sqrt();
        eprintln!("target taps versus recorded input: relL2={rel}, changed={changed}");
        assert_4b_tap_parity(&target_dir, rel, energy);
        // The BF16 drafter's 0-ULP gate must use the donor's exact inputs,
        // independently of target FP16 attention's reduction order.
        tap_buffer
            .copy_from_host(&expected_taps, &target.stream)
            .unwrap();
    }
    draft.ingest(target, taps, tokens.len()).unwrap();
    assert_eq!(draft.committed_len(), tokens.len());
    let candidates = draft.forward(target, anchor, None).unwrap();
    if strict_parity {
        let root =
            std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../tools/oracle/goldens");
        let provenance: serde_json::Value = serde_json::from_slice(
            &std::fs::read(root.join("sglang_dflash_model_4b.json")).unwrap(),
        )
        .unwrap();
        assert_eq!(provenance["input"]["tokens"], serde_json::json!(tokens));
        assert_eq!(provenance["input"]["anchor"], serde_json::json!(anchor));
        let golden = w34_host::loader::SafetensorsArchive::open(
            &root.join("sglang_dflash_model_4b.safetensors"),
        )
        .unwrap();
        let expected = golden.read_f32("final_hidden").unwrap();
        let mut actual = vec![0u16; expected.len()];
        draft
            .hidden
            .copy_to_host(&mut actual, &target.stream)
            .unwrap();
        let (mut error, mut energy, mut max_abs) = (0f64, 0f64, 0f32);
        for (&x, &y) in actual.iter().zip(&expected) {
            let x = f32::from_bits((x as u32) << 16);
            assert!(x.is_finite());
            error += ((x - y) as f64).powi(2);
            energy += (y as f64).powi(2);
            max_abs = max_abs.max((x - y).abs());
        }
        let rel = (error / energy).sqrt();
        let active_gpu = target.device.props().unwrap().name;
        let provenance_gpu = provenance["gpu"].as_str().unwrap_or("");
        eprintln!(
            "actual SGLang final hidden on {active_gpu} (provenance {provenance_gpu}): relL2={rel}, maxabs={max_abs}"
        );
        if active_gpu == provenance_gpu {
            assert_eq!(
                error, 0.0,
                "BF16 drafter must retain 0-ULP donor parity on provenance GPU {provenance_gpu}: relL2={rel}, maxabs={max_abs}"
            );
        } else {
            // Layer 0 mlp.down_proj (K=9216, magnitude ~2.07e8 with ~30x cancellation) differs
            // by relL2=0.00275 across 128-SM vs 84-SM cuBLAS Split-K reductions, which compounds
            // across 5 transformer blocks + RMSNorms to relL2 ~ 0.070 on RTX 4090 while all
            // custom kernels (convs, norms, RoPE, SiLU) remain 0-ULP identical.
            assert!(
                rel < 0.08,
                "BF16 drafter cross-GPU cuBLAS parity exceeded relL2 < 0.08 on {active_gpu} vs {provenance_gpu}: relL2={rel}, maxabs={max_abs}"
            );
        }
    }

    let merged = DraftCandidates::merge(&[candidates]).unwrap();
    let proposal = draft.select_greedy(target, anchor, &merged).unwrap();
    eprintln!("DFlash anchor={anchor}, proposal={proposal:?}");
    assert_eq!(proposal.len(), 15);
    assert_eq!(draft.committed_len(), tokens.len());
    assert!(
        proposal
            .iter()
            .all(|&v| v >= 0 && (v as usize) < engine.cfg.vocab_size)
    );
    let replay = draft.forward(target, anchor, None).unwrap();
    assert_eq!(
        proposal,
        draft.select_greedy(target, anchor, &replay).unwrap()
    );
    assert_eq!(
        proposal,
        draft.forward_select_greedy(target, anchor).unwrap()
    );
    draft.truncate(0).unwrap();
    draft.ingest(target, taps, tokens.len()).unwrap();
    assert_eq!(
        proposal,
        draft.forward_select_greedy(target, anchor).unwrap()
    );
    let replay = draft.forward(target, anchor, None).unwrap();
    assert_eq!(
        proposal,
        draft.select_greedy(target, anchor, &replay).unwrap()
    );
}

/// Real checkpoint operands and the actual FP32 CUDA rotary cache, not the
/// synthetic BF16-table fixture. Isolates arithmetic from preceding GEMMs.
#[test]
fn real_shape_qk_rope_matches_actual_sglang() {
    use core::ffi::c_void;
    use w34_host::{
        cuda::{CudaDevice, CudaStream, DeviceBuffer},
        loader::SafetensorsArchive,
    };
    let _device = CudaDevice::set_current(0).unwrap();
    let stream = CudaStream::new(0).unwrap();
    let a = SafetensorsArchive::open(
        &std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../tools/oracle/goldens/sglang_dflash_model_4b.safetensors"),
    )
    .unwrap();
    for layer in 0..5 {
        let prefix = format!("ops.rope{layer}");
        let upload = |suffix: &str| {
            let bytes = a.read_bytes(&format!("{prefix}.{suffix}")).unwrap();
            let b = DeviceBuffer::alloc(0, bytes.len()).unwrap();
            b.copy_from_host(&bytes, &stream).unwrap();
            b
        };
        let qkv = upload("input");
        let positions: Vec<i32> = a
            .read_bytes(&format!("{prefix}.positions"))
            .unwrap()
            .as_chunks::<8>()
            .0
            .iter()
            .map(|b| i64::from_le_bytes(*b) as i32)
            .collect();
        let pos = DeviceBuffer::alloc(0, positions.len() * 4).unwrap();
        pos.copy_from_host(&positions, &stream).unwrap();
        let qw = upload("qw");
        let kw = upload("kw");
        let table_values = a.read_f32(&format!("{prefix}.table")).unwrap();
        let table = upload("table");
        // SAFETY: buffers are sized from the captured operator geometry and remain alive through readback.
        let rc = unsafe {
            w34_kernels::ffi::w34_dflash_qk_norm_rope_bf16_f32_table(
                &stream.exec(0, 0),
                qkv.addr() as *mut c_void,
                pos.addr() as *const i32,
                qw.addr() as *const c_void,
                kw.addr() as *const c_void,
                table.addr() as *const c_void,
                16,
                16,
                4,
                256,
                (table_values.len() / 256) as i32,
                1e-6,
            )
        };
        assert_eq!(rc, 0);
        let expected = a.read_f32(&format!("{prefix}.output")).unwrap();
        let mut actual = vec![0u16; expected.len()];
        qkv.copy_to_host(&mut actual, &stream).unwrap();
        let mut error = 0f64;
        let mut energy = 0f64;
        for (index, (&x, &y)) in actual.iter().zip(&expected).enumerate() {
            if index % 6144 >= 5120 {
                assert_eq!(f32::from_bits((x as u32) << 16), y);
                continue;
            }
            let x = f32::from_bits((x as u32) << 16);
            assert!(x.is_finite());
            error += ((x - y) as f64).powi(2);
            energy += (y as f64).powi(2);
        }
        let rel = (error / energy).sqrt();
        eprintln!("{prefix} relL2={rel}");
        assert!(rel < 0.0001, "{prefix} real-shape RoPE relL2={rel}");

        // Also verify the fused QK-norm/RoPE + contiguous Q-out + BF16 paged KV write kernel
        // (`w34_dflash_qk_norm_rope_kv_write_bf16_f32_table`) against `w34_kv_cache_write_strided_bf16`,
        // including reversed page-table mapping and negative (`slot < 0`) skip.
        let kv_pool = w34_host::cache::KvPool::new(
            0,
            w34_kernels::ffi::W34_KV_NAMESPACE_DFLASH_DRAFT_CTX,
            &[0],
            4,
            256,
            16,
            3,
            w34_host::cache::KvFormatPolicy::Bf16,
        )
        .unwrap();
        let ref_pool = w34_host::cache::KvPool::new(
            0,
            w34_kernels::ffi::W34_KV_NAMESPACE_DFLASH_DRAFT_CTX,
            &[0],
            4,
            256,
            16,
            3,
            w34_host::cache::KvFormatPolicy::Bf16,
        )
        .unwrap();
        let sentinel = vec![0x7E7Eu16; 3 * 4 * 16 * 256];
        kv_pool.layers[0]
            .k_buf
            .copy_from_host(&sentinel, &stream)
            .unwrap();
        kv_pool.layers[0]
            .v_buf
            .copy_from_host(&sentinel, &stream)
            .unwrap();
        ref_pool.layers[0]
            .k_buf
            .copy_from_host(&sentinel, &stream)
            .unwrap();
        ref_pool.layers[0]
            .v_buf
            .copy_from_host(&sentinel, &stream)
            .unwrap();

        let mut slot_map: Vec<i32> = (0..16).map(|r| 31 - r).collect();
        slot_map[5] = -1;
        let slots_dev = DeviceBuffer::alloc(0, slot_map.len() * 4).unwrap();
        slots_dev.copy_from_host(&slot_map, &stream).unwrap();
        let q_loc_dev = DeviceBuffer::alloc(0, 2 * 4).unwrap();
        q_loc_dev.copy_from_host(&[0i32, 16i32], &stream).unwrap();

        let qkv_fused = upload("input");
        let q_out_fused = DeviceBuffer::alloc(0, 16 * 16 * 256 * 2).unwrap();
        let q_out_ref = DeviceBuffer::alloc(0, 16 * 16 * 256 * 2).unwrap();
        let qkv_for_ref = upload("input");
        // SAFETY: device buffers match the operator geometry and outlive the stream readback.
        let rc = unsafe {
            w34_kernels::ffi::w34_dflash_qk_norm_rope_bf16_f32_table_qout(
                &stream.exec(0, 0),
                qkv_for_ref.addr() as *mut c_void,
                q_out_ref.addr() as *mut c_void,
                pos.addr() as *const i32,
                qw.addr() as *const c_void,
                kw.addr() as *const c_void,
                table.addr() as *const c_void,
                16,
                16,
                4,
                256,
                (table_values.len() / 256) as i32,
                1e-6,
            )
        };
        assert_eq!(rc, 0);
        let write_batch = w34_kernels::ffi::W34AttentionBatch {
            query_start_loc: q_loc_dev.addr() as *const i32,
            context_lens_before: q_loc_dev.addr() as *const i32,
            seq_lens_after: (q_loc_dev.addr() + 4) as *const i32,
            block_tables: q_loc_dev.addr() as *const i32,
            write_slot_mapping: slots_dev.addr() as *const i32,
            valid_token_mask: core::ptr::null(),
            valid_req_mask: core::ptr::null(),
            num_reqs: 1,
            total_tokens: 16,
            max_blocks_per_req: 1,
            max_q_len: 16,
            max_seq_len: 16,
            sliding_window: 0,
            mask_mode: w34_kernels::ffi::W34_ATTN_MASK_NONCAUSAL_WINDOW,
            sm_scale: 1.0,
            reserved: 0,
        };
        // SAFETY: ref_pool and qkv_for_ref are sized for 16 rows and 3 pages of 16 slots.
        let rc = unsafe {
            w34_kernels::ffi::w34_kv_cache_write_strided_bf16(
                &stream.exec(0, 0),
                &ref_pool.layers[0].layout,
                &write_batch,
                (qkv_for_ref.addr() + (16 * 256 * 2) as u64) as *const c_void,
                (qkv_for_ref.addr() + ((16 + 4) * 256 * 2) as u64) as *const c_void,
                (16 + 8) * 256,
            )
        };
        assert_eq!(rc, 0);
        // SAFETY: kv_pool, qkv_fused, and q_out_fused match the 16-row operator geometry.
        let rc = unsafe {
            w34_kernels::ffi::w34_dflash_qk_norm_rope_kv_write_bf16_f32_table(
                &stream.exec(0, 0),
                qkv_fused.addr() as *mut c_void,
                q_out_fused.addr() as *mut c_void,
                pos.addr() as *const i32,
                qw.addr() as *const c_void,
                kw.addr() as *const c_void,
                table.addr() as *const c_void,
                &kv_pool.layers[0].layout,
                slots_dev.addr() as *const i32,
                16,
                16,
                4,
                256,
                (table_values.len() / 256) as i32,
                1e-6,
            )
        };
        assert_eq!(rc, 0);

        let mut q_actual = vec![0u16; 16 * 16 * 256];
        let mut q_expected = vec![0u16; 16 * 16 * 256];
        q_out_fused.copy_to_host(&mut q_actual, &stream).unwrap();
        q_out_ref.copy_to_host(&mut q_expected, &stream).unwrap();
        assert_eq!(q_actual, q_expected, "{prefix} fused q_out mismatch");

        let mut k_actual = vec![0u16; sentinel.len()];
        let mut k_expected = vec![0u16; sentinel.len()];
        let mut v_actual = vec![0u16; sentinel.len()];
        let mut v_expected = vec![0u16; sentinel.len()];
        kv_pool.layers[0]
            .k_buf
            .copy_to_host(&mut k_actual, &stream)
            .unwrap();
        ref_pool.layers[0]
            .k_buf
            .copy_to_host(&mut k_expected, &stream)
            .unwrap();
        kv_pool.layers[0]
            .v_buf
            .copy_to_host(&mut v_actual, &stream)
            .unwrap();
        ref_pool.layers[0]
            .v_buf
            .copy_to_host(&mut v_expected, &stream)
            .unwrap();
        assert_eq!(k_actual, k_expected, "{prefix} fused K page write mismatch");
        assert_eq!(v_actual, v_expected, "{prefix} fused V page write mismatch");
    }
}

#[test]
fn dflash_select_greedy_device_matches_ties_and_rejects_bad_anchor() {
    use core::ffi::c_void;
    use w34_host::cuda::{CudaDevice, CudaStream, DeviceBuffer};
    let _device = CudaDevice::set_current(0).unwrap();
    let stream = CudaStream::new(0).unwrap();
    let rows = 3i32;
    let hidden = 32i32;
    let top_k = 4i32;
    let vocab = 16i32;
    let f32_to_bf16 = |x: f32| -> u16 { (x.to_bits() >> 16) as u16 };

    // Construct hidden_bf16 [3, 32], weight_bf16 [16, 32] (all ones so dot products equal
    // the sum of hidden_bf16), and topk_ids [3, 4] with a deliberate score tie on row 1
    // (candidates 1 and 2 have identical topk_scores so the strict `>` tie-break must pick col 1).
    let hidden_host = vec![f32_to_bf16(0.25); (rows * hidden) as usize];
    let weight_host = vec![f32_to_bf16(1.0); (vocab * hidden) as usize];
    let topk_ids_host: Vec<i32> = vec![
        3, 5, 7, 9, // row 0
        2, 4, 6, 8, // row 1 (col 1 and col 2 tie below)
        1, 10, 11, 12, // row 2
    ];
    let topk_scores_host: Vec<f32> = vec![
        1.0, 2.0, 0.5, 0.1, // row 0
        0.5, 4.0, 4.0,
        1.0, // row 1: col 1 (token 4) and col 2 (token 6) tie at 4.0 -> must pick col 1 (token 4)
        0.2, 0.9, 3.5,
        3.5, // row 2: col 2 (token 11) and col 3 (token 12) tie at 3.5 -> must pick col 2 (token 11)
    ];

    let d_hidden = DeviceBuffer::alloc(0, hidden_host.len() * 2).unwrap();
    let d_weight = DeviceBuffer::alloc(0, weight_host.len() * 2).unwrap();
    let d_ids = DeviceBuffer::alloc(0, topk_ids_host.len() * 4).unwrap();
    let d_scores = DeviceBuffer::alloc(0, topk_scores_host.len() * 4).unwrap();
    let d_path = DeviceBuffer::alloc(0, (rows - 1) as usize * 4).unwrap();
    d_hidden.copy_from_host(&hidden_host, &stream).unwrap();
    d_weight.copy_from_host(&weight_host, &stream).unwrap();
    d_ids.copy_from_host(&topk_ids_host, &stream).unwrap();
    d_scores.copy_from_host(&topk_scores_host, &stream).unwrap();

    // SAFETY: all device buffers match the 3x4x32 test geometry and remain live until readback.
    let rc = unsafe {
        w34_kernels::ffi::w34_dflash_select_greedy_bf16(
            &stream.exec(0, 0),
            d_ids.addr() as *const i32,
            d_scores.addr() as *const f32,
            d_hidden.addr() as *const c_void,
            d_weight.addr() as *const c_void,
            d_weight.addr() as *const c_void,
            d_path.addr() as *mut i32,
            2,
            rows,
            top_k,
            hidden,
            vocab,
        )
    };
    assert_eq!(rc, 0);
    let mut path = vec![0i32; (rows - 1) as usize];
    d_path.copy_to_host(&mut path, &stream).unwrap();
    assert_eq!(path, vec![5, 4]);

    // Out-of-vocab anchor is rejected immediately on the host boundary, and out-of-vocab
    // candidate IDs in `topk_ids` produce the `-1` sentinel in `out_path[0]`.
    // SAFETY: all device buffers remain live and out_path has capacity for at least 1 i32.
    let rc_bad_anchor = unsafe {
        w34_kernels::ffi::w34_dflash_select_greedy_bf16(
            &stream.exec(0, 0),
            d_ids.addr() as *const i32,
            d_scores.addr() as *const f32,
            d_hidden.addr() as *const c_void,
            d_weight.addr() as *const c_void,
            d_weight.addr() as *const c_void,
            d_path.addr() as *mut i32,
            vocab,
            rows,
            top_k,
            hidden,
            vocab,
        )
    };
    assert_ne!(rc_bad_anchor, 0);

    let bad_ids_host: Vec<i32> = vec![-1; (rows * top_k) as usize];
    d_ids.copy_from_host(&bad_ids_host, &stream).unwrap();
    // SAFETY: all device buffers remain live and out_path has capacity for at least 1 i32.
    let rc_bad_ids = unsafe {
        w34_kernels::ffi::w34_dflash_select_greedy_bf16(
            &stream.exec(0, 0),
            d_ids.addr() as *const i32,
            d_scores.addr() as *const f32,
            d_hidden.addr() as *const c_void,
            d_weight.addr() as *const c_void,
            d_weight.addr() as *const c_void,
            d_path.addr() as *mut i32,
            2,
            rows,
            top_k,
            hidden,
            vocab,
        )
    };
    assert_eq!(rc_bad_ids, 0);
    d_path.copy_to_host(&mut path, &stream).unwrap();
    assert_eq!(path[0], -1);
}

#[test]
fn real_shape_grouped_convs_match_actual_sglang() {
    use core::ffi::c_void;
    use w34_host::{
        cuda::{CudaDevice, CudaStream, DeviceBuffer},
        loader::SafetensorsArchive,
    };
    let _device = CudaDevice::set_current(0).unwrap();
    let stream = CudaStream::new(0).unwrap();
    let a = SafetensorsArchive::open(
        &std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../tools/oracle/goldens/sglang_dflash_model_4b.safetensors"),
    )
    .unwrap();
    let upload = |bytes: &[u8]| {
        let b = DeviceBuffer::alloc(0, bytes.len()).unwrap();
        b.copy_from_host(bytes, &stream).unwrap();
        b
    };
    for index in 0..20 {
        let prefix = format!("ops.conv{index}");
        let x = upload(&a.read_bytes(&format!("{prefix}.input")).unwrap());
        let base_bytes = a.read_bytes(&format!("{prefix}.base")).unwrap();
        let base = upload(&[base_bytes.clone(), base_bytes].concat());
        let delta_bytes = a.read_bytes(&format!("{prefix}.delta")).unwrap();
        let mut delta_packed = vec![0u8; delta_bytes.len() * 2];
        for row in 0..16 {
            delta_packed[row * 1280..row * 1280 + 640]
                .copy_from_slice(&delta_bytes[row * 640..(row + 1) * 640]);
        }
        let delta = upload(&delta_packed);
        let out = DeviceBuffer::alloc(0, 16 * 2560 * 2).unwrap();
        // SAFETY: buffers are sized from the captured operator geometry and remain alive through readback.
        let rc = unsafe {
            w34_kernels::ffi::w34_dflash_conv_bf16(
                &stream.exec(0, 0),
                x.addr() as *const c_void,
                delta.addr() as *const c_void,
                base.addr() as *const c_void,
                out.addr() as *mut c_void,
                16,
                2560,
                16,
                16,
                2,
                0,
            )
        };
        assert_eq!(rc, 0);
        let expected = a.read_f32(&format!("{prefix}.output")).unwrap();
        let mut actual = vec![0u16; expected.len()];
        out.copy_to_host(&mut actual, &stream).unwrap();
        let (mut error, mut energy) = (0f64, 0f64);
        for (&x, &y) in actual.iter().zip(&expected) {
            let x = f32::from_bits((x as u32) << 16);
            assert!(x.is_finite());
            error += ((x - y) as f64).powi(2);
            energy += (y as f64).powi(2);
        }
        let rel = (error / energy).sqrt();
        eprintln!("{prefix} relL2={rel}");
        assert!(rel < 0.0001, "{prefix} real-shape convolution relL2={rel}");
    }
}

#[test]
fn real_shape_silu_matches_actual_sglang() {
    use core::ffi::c_void;
    use w34_host::{
        cuda::{CudaDevice, CudaStream, DeviceBuffer},
        loader::SafetensorsArchive,
    };
    let _device = CudaDevice::set_current(0).unwrap();
    let stream = CudaStream::new(0).unwrap();
    let a = SafetensorsArchive::open(
        &std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../tools/oracle/goldens/sglang_dflash_model_4b.safetensors"),
    )
    .unwrap();
    for layer in 0..5 {
        let prefix = format!("ops.layers.{layer}.mlp.act_fn");
        let bytes = a.read_bytes(&format!("{prefix}.input0")).unwrap();
        let x = DeviceBuffer::alloc(0, bytes.len()).unwrap();
        x.copy_from_host(&bytes, &stream).unwrap();
        let out = DeviceBuffer::alloc(0, 16 * 9216 * 2).unwrap();
        // SAFETY: buffers are sized from the captured operator geometry and remain alive through readback.
        let rc = unsafe {
            w34_kernels::ffi::w34_silu_and_mul_bf16(
                &stream.exec(0, 0),
                x.addr() as *const c_void,
                out.addr() as *mut c_void,
                16,
                9216,
            )
        };
        assert_eq!(rc, 0);
        let expected = a.read_f32(&format!("{prefix}.output0")).unwrap();
        let mut actual = vec![0u16; expected.len()];
        out.copy_to_host(&mut actual, &stream).unwrap();
        let (mut error, mut energy) = (0f64, 0f64);
        for (&x, &y) in actual.iter().zip(&expected) {
            let x = f32::from_bits((x as u32) << 16);
            assert!(x.is_finite());
            error += ((x - y) as f64).powi(2);
            energy += (y as f64).powi(2);
        }
        let rel = (error / energy).sqrt();
        eprintln!("{prefix} relL2={rel}");
        assert!(rel < 0.0001, "{prefix} real-shape SiLU relL2={rel}");
    }
}

/// Genuine donor operands captured before in-place RMSNorm mutation.
#[test]
fn real_shape_norms_match_actual_sglang() {
    use core::ffi::c_void;
    use w34_host::{
        cuda::{CudaDevice, CudaStream, DeviceBuffer},
        loader::SafetensorsArchive,
    };
    let _device = CudaDevice::set_current(0).unwrap();
    let stream = CudaStream::new(0).unwrap();
    let a = SafetensorsArchive::open(
        &std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../tools/oracle/goldens/sglang_dflash_model_4b.safetensors"),
    )
    .unwrap();
    let mut failures = Vec::new();
    let mut count = 0;
    for name in a
        .tensor_names()
        .filter(|n| n.starts_with("norms.") && n.ends_with(".weight"))
    {
        let prefix = name.strip_suffix(".weight").unwrap();
        let upload = |suffix: &str| {
            let bytes = a.read_bytes(&format!("{prefix}.{suffix}")).unwrap();
            let b = DeviceBuffer::alloc(0, bytes.len()).unwrap();
            b.copy_from_host(&bytes, &stream).unwrap();
            b
        };
        let x = upload("input0");
        let w = upload("weight");
        let residual = a
            .contains(&format!("{prefix}.input1"))
            .then(|| upload("input1"));
        let expected = a.read_f32(&format!("{prefix}.output0")).unwrap();
        let hidden = a.read_f32(name).unwrap().len();
        let y = DeviceBuffer::alloc(0, expected.len() * 2).unwrap();
        let eps = a.read_f32(&format!("{prefix}.eps")).unwrap()[0];
        // SAFETY: all operands use captured BF16 geometry and live through readback.
        let rc = unsafe {
            w34_kernels::ffi::w34_rms_norm_bf16(
                &stream.exec(0, 0),
                x.addr() as *const c_void,
                residual
                    .as_ref()
                    .map_or(std::ptr::null_mut(), |r| r.addr() as *mut c_void),
                w.addr() as *const c_void,
                y.addr() as *mut c_void,
                (expected.len() / hidden) as i32,
                hidden as i32,
                eps,
            )
        };
        assert_eq!(rc, 0);
        for (suffix, buffer) in [("output0", Some(&y)), ("output1", residual.as_ref())] {
            let Some(buffer) = buffer else { continue };
            let expected = a.read_f32(&format!("{prefix}.{suffix}")).unwrap();
            let mut actual = vec![0u16; expected.len()];
            buffer.copy_to_host(&mut actual, &stream).unwrap();
            let (mut err, mut energy, mut maxabs) = (0f64, 0f64, 0f32);
            for (&x, &y) in actual.iter().zip(&expected) {
                let x = f32::from_bits((x as u32) << 16);
                assert!(x.is_finite());
                err += ((x - y) as f64).powi(2);
                energy += (y as f64).powi(2);
                maxabs = maxabs.max((x - y).abs());
            }
            let rel = (err / energy.max(1e-300)).sqrt();
            eprintln!(
                "{prefix}.{suffix} fused={} relL2={rel} maxabs={maxabs}",
                residual.is_some()
            );
            if rel >= 1e-4 {
                failures.push(format!("{prefix}.{suffix}: {rel}"));
            }
        }
        count += 1;
    }
    assert!(count >= 11, "missing actual model norm captures: {count}");
    assert!(failures.is_empty(), "real norm failures: {failures:?}");
}

#[test]
fn real_shape_layer0_gemms_match_actual_sglang() {
    use w34_host::{
        cuda::{CublasHandle, CudaDevice, CudaStream, DeviceBuffer},
        loader::SafetensorsArchive,
    };
    let dev = CudaDevice::set_current(0).unwrap();
    let active_gpu = dev.props().unwrap().name;
    let stream = CudaStream::new(0).unwrap();
    let blas = CublasHandle::new(&stream).unwrap();
    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../tools/oracle/goldens");
    let provenance: serde_json::Value =
        serde_json::from_slice(&std::fs::read(root.join("sglang_dflash_model_4b.json")).unwrap())
            .unwrap();
    let provenance_gpu = provenance["gpu"].as_str().unwrap_or("");
    let ops = SafetensorsArchive::open(&root.join("sglang_dflash_model_4b.safetensors")).unwrap();
    let Some(weights_dir) = common::require_or_skip_cached_model(
        "nchapman/Qwen3.5-4B-DFlash2",
        "real_shape_layer0_gemms_match_actual_sglang",
    ) else {
        return;
    };
    let weights = SafetensorsArchive::open(&weights_dir).unwrap();
    let upload = |bytes: &[u8]| {
        let b = DeviceBuffer::alloc(0, bytes.len()).unwrap();
        b.copy_from_host(bytes, &stream).unwrap();
        b
    };
    let mut failures = Vec::new();
    let mut count = 0;
    for (proj, w_names, n, k) in [
        (
            "self_attn.qkv_proj",
            &[
                "layers.0.self_attn.q_proj.weight",
                "layers.0.self_attn.k_proj.weight",
                "layers.0.self_attn.v_proj.weight",
            ][..],
            6144usize,
            2560usize,
        ),
        (
            "mlp.gate_up_proj",
            &[
                "layers.0.mlp.gate_proj.weight",
                "layers.0.mlp.up_proj.weight",
            ][..],
            18432,
            2560,
        ),
        (
            "mlp.down_proj",
            &["layers.0.mlp.down_proj.weight"][..],
            2560,
            9216,
        ),
    ] {
        let op_prefix = format!("ops.layers.0.{proj}");
        let x = upload(&ops.read_bytes(&format!("{op_prefix}.input0")).unwrap());
        let mut w_bytes = Vec::with_capacity(n * k * 2);
        for name in w_names {
            w_bytes.extend_from_slice(&weights.read_bytes(name).unwrap());
        }
        let w = upload(&w_bytes);
        let expected = ops.read_f32(&format!("{op_prefix}.output0")).unwrap();
        let y = DeviceBuffer::alloc(0, expected.len() * 2).unwrap();
        blas.gemm_bf16(
            x.addr(),
            w.addr(),
            y.addr(),
            (expected.len() / n) as i32,
            n as i32,
            k as i32,
        )
        .unwrap();
        let mut actual = vec![0u16; expected.len()];
        y.copy_to_host(&mut actual, &stream).unwrap();
        let (mut err, mut energy, mut maxabs, mut unequal) = (0f64, 0f64, 0f32, 0);
        for (&x, &y) in actual.iter().zip(&expected) {
            let x = f32::from_bits((x as u32) << 16);
            assert!(x.is_finite());
            err += ((x - y) as f64).powi(2);
            energy += (y as f64).powi(2);
            maxabs = maxabs.max((x - y).abs());
            unequal += usize::from(x != y);
        }
        let rel = (err / energy).sqrt();
        eprintln!(
            "{op_prefix} on {active_gpu} (provenance {provenance_gpu}) relL2={rel} maxabs={maxabs} unequal={unequal}"
        );
        // On the provenance GPU, cuBLAS chooses the identical kernel as SGLang (0-ULP).
        // Across different GPU architectures/SM counts (e.g. 128-SM RTX 4090 vs 84-SM 3090 Ti),
        // cuBLAS Split-K accumulation order differs; for K=9216 down_proj (where 3090 Ti vs FP64
        // already differs by relL2=0.0024 due to cancellation and 1 BF16 ULP is 2^-7 = 0.0078),
        // bound relL2 < 3e-3, and < 1e-4 for K=2560 projections.
        let tol = if active_gpu == provenance_gpu {
            0.0
        } else if k > 4096 {
            3e-3
        } else {
            1e-4
        };
        if if active_gpu == provenance_gpu {
            unequal > 0
        } else {
            rel >= tol
        } {
            failures.push(format!("{op_prefix}: {rel} (tol {tol})"));
        }
        count += 1;
    }
    assert_eq!(count, 3);
    assert!(failures.is_empty(), "real GEMM failures: {failures:?}");
}
