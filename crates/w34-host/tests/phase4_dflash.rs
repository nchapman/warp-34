#![cfg(feature = "cuda")]
use w34_host::{
    cache::KvFormatPolicy,
    dflash::{DFlash2Config, DFlash2Rank, DraftCandidates},
    loader::resolve_hf_model_dir,
    model::Qwen3_5Engine,
};
use w34_kernels::ffi::{W34_KV_FORMAT_BF16, W34_KV_NAMESPACE_DFLASH_DRAFT_CTX};
#[test]
fn cached_dflash_configs_and_candidate_merge() {
    for (repo, block, taps) in [
        ("nchapman/Qwen3.5-4B-DFlash2", 16, vec![1, 9, 16, 23, 28]),
        ("incoai/Qwen3.8-27B-DFlash2", 8, vec![5, 19, 33, 47, 61]),
    ] {
        let dir = resolve_hf_model_dir(repo).unwrap();
        let cfg = DFlash2Config::load(&dir).unwrap();
        assert_eq!(cfg.block_size, block);
        assert_eq!(cfg.target_layers, taps);
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
#[test]
fn rejects_selector_top_k_beyond_native_topk_limit() {
    let source = resolve_hf_model_dir("nchapman/Qwen3.5-4B-DFlash2").unwrap();
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
    run_4b(false, false, false);
}
#[test]
#[ignore = "explicit generation of inputs for the independent SGLang model oracle"]
fn export_dflash_oracle_inputs() {
    run_4b(true, false, false);
}
#[test]
fn actual_sglang_full_model_hidden_parity() {
    run_4b(false, true, false);
}
#[test]
#[ignore = "counterfactual donor attention injection; NOT native parity"]
fn diagnostic_donor_attention_full_model() {
    run_4b(false, true, true);
}
static GPU_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

fn run_4b(export: bool, strict_parity: bool, inject_attention: bool) {
    let _gpu_guard = GPU_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    let dir = resolve_hf_model_dir("nchapman/Qwen3.5-4B-DFlash2").unwrap();
    let cfg = DFlash2Config::load(&dir).unwrap();
    let mut engine = Qwen3_5Engine::load(
        "nchapman/Qwen3.5-4B-W4A16-GPTQ",
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
    if inject_attention {
        draft.trace_dir = Some(
            std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
                .join("../../.pi/dflash-model-inputs/trace-donor-attention"),
        );
        draft.diagnostic_attention_oracle = Some(
            std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
                .join("../../tools/oracle/goldens/sglang_dflash_model_4b.safetensors"),
        );
    }
    if export {
        draft.trace_dir = Some(
            std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
                .join("../../.pi/dflash-model-inputs/trace"),
        );
    }
    for l in draft.kv_layouts() {
        assert_eq!(l.namespace_id, W34_KV_NAMESPACE_DFLASH_DRAFT_CTX);
        assert_eq!(l.k_format, W34_KV_FORMAT_BF16);
    }
    let taps = target.tap_aux_f32.as_ref().unwrap().addr();
    draft.ingest(target, taps, tokens.len()).unwrap();
    assert_eq!(draft.committed_len(), tokens.len());
    if export {
        use w34_host::cuda::{f16_to_f32, f32_to_bf16};
        let h = engine.cfg.hidden_size;
        let out =
            std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../.pi/dflash-model-inputs");
        std::fs::create_dir_all(&out).unwrap();
        let mut values = vec![0f32; tokens.len() * 5 * h];
        target
            .tap_aux_f32
            .as_ref()
            .unwrap()
            .copy_to_host(&mut values, &target.stream)
            .unwrap();
        std::fs::write(
            out.join("taps.f32"),
            values
                .iter()
                .flat_map(|v| v.to_le_bytes())
                .collect::<Vec<_>>(),
        )
        .unwrap();
        let mut embeddings = Vec::with_capacity(16 * h);
        for id in std::iter::once(anchor).chain(std::iter::repeat_n(cfg.mask_token, 15)) {
            let mut row = vec![0u16; h];
            // SAFETY: id indexes a resident target embedding row; host row is H elements and is synchronized below.
            // SAFETY: buffers are sized from the captured operator geometry and remain alive through readback.
            let rc = unsafe {
                w34_kernels::ffi::w34_memcpy_d2h_async(
                    row.as_mut_ptr().cast(),
                    (target.embed_tokens_ptr + (id as usize * h * 2) as u64) as *const _,
                    h * 2,
                    target.stream.handle,
                )
            };
            assert_eq!(rc, 0);
            target.stream.synchronize().unwrap();
            embeddings.extend(row.into_iter().map(|v| f32_to_bf16(f16_to_f32(v))));
        }
        std::fs::write(
            out.join("embeddings.bf16"),
            embeddings
                .iter()
                .flat_map(|v| v.to_le_bytes())
                .collect::<Vec<_>>(),
        )
        .unwrap();
        std::fs::write(out.join("input.json"),serde_json::json!({"rows":tokens.len(),"block_size":16,"hidden_size":h,"anchor":anchor,"tokens":tokens}).to_string()).unwrap();
    }
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
        let mut taps_f32 = vec![0f32; tokens.len() * 5 * engine.cfg.hidden_size];
        target
            .tap_aux_f32
            .as_ref()
            .unwrap()
            .copy_to_host(&mut taps_f32, &target.stream)
            .unwrap();
        let expected_taps = golden.read_bytes("input.taps_bf16").unwrap();
        let actual_taps: Vec<u8> = taps_f32
            .iter()
            .flat_map(|&x| w34_host::cuda::f32_to_bf16(x).to_le_bytes())
            .collect();
        assert_eq!(actual_taps, expected_taps, "oracle input taps changed");
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
        eprintln!("actual SGLang final hidden relL2={rel}, maxabs={max_abs}");
        // Sub-percent normalized tensor agreement; do not turn the observed
        // multi-percent divergence into an accepted tolerance.
        assert!(
            rel < 0.005,
            "full-model hidden parity: relL2={rel}, maxabs={max_abs}"
        );
    }

    if export {
        let mut values = vec![0u16; 16 * engine.cfg.hidden_size];
        draft
            .hidden
            .copy_to_host(&mut values, &target.stream)
            .unwrap();
        let out = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../.pi/dflash-model-inputs/native_hidden.bf16");
        std::fs::write(
            out,
            values
                .iter()
                .flat_map(|v| v.to_le_bytes())
                .collect::<Vec<_>>(),
        )
        .unwrap();
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
    draft.truncate(0).unwrap();
    draft.ingest(target, taps, tokens.len()).unwrap();
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
            .join("../../tools/oracle/goldens/sglang_dflash_model_4b_operators.safetensors"),
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
    }
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
            .join("../../tools/oracle/goldens/sglang_dflash_model_4b_operators.safetensors"),
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
            .join("../../tools/oracle/goldens/sglang_dflash_model_4b_operators.safetensors"),
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
            .join("../../tools/oracle/goldens/sglang_dflash_model_4b_norms.safetensors"),
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
    let _device = CudaDevice::set_current(0).unwrap();
    let stream = CudaStream::new(0).unwrap();
    let blas = CublasHandle::new(&stream).unwrap();
    let a = SafetensorsArchive::open(
        &std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../tools/oracle/goldens/sglang_dflash_model_4b_gemms.safetensors"),
    )
    .unwrap();
    let mut failures = Vec::new();
    let mut count = 0;
    for name in a
        .tensor_names()
        .filter(|n| n.starts_with("gemms.") && n.ends_with(".weight"))
    {
        let prefix = name.strip_suffix(".weight").unwrap();
        let upload = |suffix: &str| {
            let bytes = a.read_bytes(&format!("{prefix}.{suffix}")).unwrap();
            let b = DeviceBuffer::alloc(0, bytes.len()).unwrap();
            b.copy_from_host(&bytes, &stream).unwrap();
            b
        };
        let x = upload("input");
        let w = upload("weight");
        let shape = &a.tensor_info(name).unwrap().shape;
        let (n, k) = (shape[0], shape[1]);
        let expected = a.read_f32(&format!("{prefix}.output")).unwrap();
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
        eprintln!("{prefix} relL2={rel} maxabs={maxabs} unequal={unequal}");
        if rel >= 1e-4 {
            failures.push(format!("{prefix}: {rel}"));
        }
        count += 1;
    }
    assert_eq!(count, 4);
    assert!(failures.is_empty(), "real GEMM failures: {failures:?}");
}
