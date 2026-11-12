use super::*;

fn write_safetensors_file(
    path: &Path,
    tensors: &[(&str, &str, &[usize], &[u8])],
) -> io::Result<()> {
    let mut header = serde_json::Map::new();
    let mut offset = 0usize;
    for &(name, dtype, shape, bytes) in tensors {
        let end = offset + bytes.len();
        header.insert(
            name.to_string(),
            serde_json::json!({
                "dtype": dtype,
                "shape": shape,
                "data_offsets": [offset, end],
            }),
        );
        offset = end;
    }
    let header_bytes = serde_json::to_vec(&header)?;
    let header_len = header_bytes.len() as u64;
    let mut file_bytes = Vec::with_capacity(8 + header_bytes.len() + offset);
    file_bytes.extend_from_slice(&header_len.to_le_bytes());
    file_bytes.extend_from_slice(&header_bytes);
    for &(_, _, _, bytes) in tensors {
        file_bytes.extend_from_slice(bytes);
    }
    fs::write(path, file_bytes)
}

fn f32s_to_bytes(vals: &[f32]) -> Vec<u8> {
    let mut out = Vec::with_capacity(vals.len() * 4);
    for &v in vals {
        out.extend_from_slice(&v.to_le_bytes());
    }
    out
}

fn bf16s_from_f32s_bytes(vals: &[f32]) -> Vec<u8> {
    let mut out = Vec::with_capacity(vals.len() * 2);
    for &v in vals {
        let bits = (v.to_bits() >> 16) as u16;
        out.extend_from_slice(&bits.to_le_bytes());
    }
    out
}

fn f16s_from_f32s_bytes(vals: &[f32]) -> Vec<u8> {
    let mut out = Vec::with_capacity(vals.len() * 2);
    for &v in vals {
        let bits = f32_to_f16(v);
        out.extend_from_slice(&bits.to_le_bytes());
    }
    out
}

#[test]
fn test_lora_module_spec_parsing() {
    let s1 = LoraModuleSpec::parse("heretic=nchapman/Qwen3.5-9B-Heretic-LoRA").unwrap();
    assert_eq!(s1.name, "heretic");
    assert_eq!(s1.path, "nchapman/Qwen3.5-9B-Heretic-LoRA");
    assert!(s1.revision.is_none());

    let s2 =
        LoraModuleSpec::parse("trial3=nchapman/Qwen3.5-9B-Heretic-LoRA/pareto/trial_3@abc1234")
            .unwrap();
    assert_eq!(s2.name, "trial3");
    assert_eq!(s2.path, "nchapman/Qwen3.5-9B-Heretic-LoRA/pareto/trial_3");
    assert_eq!(s2.revision.as_deref(), Some("abc1234"));

    let s3 = LoraModuleSpec::parse("nchapman/Qwen3.5-9B-Heretic-LoRA").unwrap();
    assert_eq!(s3.name, "nchapman/Qwen3.5-9B-Heretic-LoRA");
    assert_eq!(s3.path, "nchapman/Qwen3.5-9B-Heretic-LoRA");

    let s4 =
        LoraModuleSpec::parse(r#"{"name":"json_lora","path":"org/repo/sub","revision":"main"}"#)
            .unwrap();
    assert_eq!(s4.name, "json_lora");
    assert_eq!(s4.path, "org/repo/sub");
    assert_eq!(s4.revision.as_deref(), Some("main"));

    assert!(LoraModuleSpec::parse("").is_err());
    assert!(LoraModuleSpec::parse("=path").is_err());
    assert!(LoraModuleSpec::parse("name=").is_err());
}

#[test]
fn test_split_hf_repo_and_subfolder() {
    let (repo, sub) = split_hf_repo_and_subfolder("nchapman/Qwen3.5-9B-Heretic-LoRA").unwrap();
    assert_eq!(repo, "nchapman/Qwen3.5-9B-Heretic-LoRA");
    assert!(sub.is_none());

    let (repo2, sub2) =
        split_hf_repo_and_subfolder("nchapman/Qwen3.5-9B-Heretic-LoRA/pareto/trial_3").unwrap();
    assert_eq!(repo2, "nchapman/Qwen3.5-9B-Heretic-LoRA");
    assert_eq!(sub2.as_deref(), Some("pareto/trial_3"));

    assert!(split_hf_repo_and_subfolder("invalid_single_segment").is_err());
    assert!(split_hf_repo_and_subfolder("owner/repo/../escape").is_err());
    assert!(
        resolve_cached_or_local_lora_dir("nchapman/Qwen3.5-9B-Heretic-LoRA", Some("../../escape"))
            .is_err()
    );
    assert!(
        resolve_cached_or_local_lora_dir(
            "nchapman/Qwen3.5-9B-Heretic-LoRA",
            Some("0000000000000000000000000000000000000000")
        )
        .is_err()
    );
}

#[test]
fn test_peft_config_scaling_and_patterns() {
    let json = r#"{
        "peft_type": "LORA",
        "r": 4,
        "lora_alpha": 8.0,
        "use_rslora": false,
        "rank_pattern": {"layers.0.self_attn.o_proj": 2},
        "alpha_pattern": {"self_attn.o_proj": 6.0}
    }"#;
    let mut cfg: PeftConfig = serde_json::from_str(json).unwrap();
    cfg.validate().unwrap();
    assert!((cfg.scaling_for_module(Some(1), "mlp.down_proj", 4) - 2.0).abs() < 1e-6);
    // Pattern match on layer 0 o_proj: rank=2, alpha=6.0 -> scaling = 6.0 / 2 = 3.0
    assert!((cfg.scaling_for_module(Some(0), "self_attn.o_proj", 2) - 3.0).abs() < 1e-6);

    // rsLoRA scaling: alpha / sqrt(r)
    cfg.use_rslora = true;
    assert!((cfg.scaling_for_module(Some(1), "mlp.down_proj", 4) - 4.0).abs() < 1e-6);
}

#[test]
fn test_lora_adapter_host_load_dtypes_and_fan_in_fan_out() {
    let tmp = tempfile::tempdir().unwrap();
    // Reject Conv1D-style fan_in_fan_out = true adapters:
    let bad_cfg_json = r#"{
        "peft_type": "LORA",
        "r": 2,
        "lora_alpha": 4.0,
        "fan_in_fan_out": true
    }"#;
    fs::write(tmp.path().join("adapter_config.json"), bad_cfg_json).unwrap();
    let err = PeftConfig::from_dir(tmp.path()).unwrap_err();
    assert!(
        err.to_string().contains("fan_in_fan_out=true"),
        "unexpected error: {err}"
    );

    let cfg_json = r#"{
        "peft_type": "LORA",
        "r": 2,
        "lora_alpha": 4.0,
        "fan_in_fan_out": false
    }"#;
    fs::write(tmp.path().join("adapter_config.json"), cfg_json).unwrap();

    // Canonical PEFT shapes on disk:
    // lora_A has shape [r=2, in_features=4].
    // lora_B has shape [out_features=3, r=2] and is pre-scaled on load by alpha/r = 2.0.
    let a_disk = [1.0f32, 3.0, 5.0, 7.0, 2.0, 4.0, 6.0, 8.0]; // [2, 4]
    let b_disk = [0.5f32, 2.0, 1.0, 2.5, 1.5, 3.0]; // [3, 2]
    let a_bytes = bf16s_from_f32s_bytes(&a_disk);
    let b_bytes = f16s_from_f32s_bytes(&b_disk);

    // Also include a linear_attn.out_proj in FP32
    let a2 = [1.0f32; 8]; // [r=2, in=4]
    let b2 = [1.0f32; 6]; // [out=3, r=2]
    let a2_bytes = f32s_to_bytes(&a2);
    let b2_bytes = f32s_to_bytes(&b2);

    write_safetensors_file(
        &tmp.path().join("adapter_model.safetensors"),
        &[
            (
                "base_model.model.model.language_model.layers.0.self_attn.o_proj.lora_A.weight",
                "BF16",
                &[2, 4],
                &a_bytes,
            ),
            (
                "base_model.model.model.language_model.layers.0.self_attn.o_proj.lora_B.weight",
                "F16",
                &[3, 2],
                &b_bytes,
            ),
            (
                "base_model.model.layers.0.linear_attn.out_proj.lora_A.weight",
                "F32",
                &[2, 4],
                &a2_bytes,
            ),
            (
                "base_model.model.layers.0.linear_attn.out_proj.lora_B.weight",
                "F32",
                &[3, 2],
                &b2_bytes,
            ),
        ],
    )
    .unwrap();

    let host = LoraAdapterHost::load_from_dir("test", "local/test", tmp.path()).unwrap();
    let o_proj = host
        .layer_weights
        .get(&(0, TargetLoraModule::OProj))
        .unwrap();
    assert_eq!(o_proj.r, 2);
    assert_eq!(o_proj.in_features, 4);
    assert_eq!(o_proj.out_features, 3);
    assert_eq!(o_proj.a_f32[0..4], [1.0, 3.0, 5.0, 7.0]);
    assert_eq!(o_proj.a_f32[4..8], [2.0, 4.0, 6.0, 8.0]);
    // Row 0 of B is [0.5, 2.0], scaled by alpha/r = 4/2 = 2 -> [1.0, 4.0]
    assert_eq!(o_proj.b_f32[0..2], [1.0, 4.0]);

    let out_proj = host
        .layer_weights
        .get(&(0, TargetLoraModule::OutProj))
        .unwrap();
    assert_eq!(out_proj.r, 2);
    assert_eq!(out_proj.in_features, 4);
    assert_eq!(out_proj.out_features, 3);
}

#[test]
fn test_radix_and_scheduler_lora_isolation() {
    let mut sched = crate::scheduler::SchedulerV1::new(crate::scheduler::SchedulerConfig {
        page_size: 16,
        mamba_chunk_size: 64,
        ..Default::default()
    });
    let prompt: Vec<i32> = (100..180).collect();

    // Submit with adapter 0:
    let req0 = sched.submit_request_with_lora_and_bidir(
        prompt.clone(),
        prompt.clone(),
        4,
        0,
        false,
        vec![],
        vec![],
        false,
        None,
        None,
        None,
        0,
    );
    let mut next_slot = 0i32;
    sched.admit_waiting(|_| {
        let s = (next_slot, [next_slot + 1, next_slot + 2]);
        next_slot += 3;
        Some(s)
    });
    let slices = sched.schedule_step();
    assert_eq!(slices.len(), 1);
    assert_eq!(slices[0].req_id, req0);
    assert_eq!(slices[0].lora_id, 0);
    let mut next_page = 0i32;
    for r in 0..5 {
        let s = sched.schedule_step();
        if s.is_empty() {
            break;
        }
        {
            let seq = sched
                .active
                .iter_mut()
                .find(|seq| seq.req_id == s[0].req_id)
                .unwrap();
            let pages_needed = (s[0].ctx_len_before + s[0].query_tokens.len()).div_ceil(16);
            while seq.radix_state.kv_pages.len() < pages_needed {
                seq.radix_state.kv_pages.push(next_page);
                next_page += 1;
            }
        }
        sched.commit_slice(&s[0], Some(200 + r), || {
            let rep = next_slot;
            next_slot += 1;
            Some(rep)
        });
    }
    let done = sched.take_finished();
    assert_eq!(done.len(), 1);
    assert_eq!(done[0].cached_prefix_tokens, 0);

    // Base (`lora_id = -1`) and adapter 1 (`lora_id = 1`) must NOT hit adapter 0's prefix cache;
    // adapter 0 (`lora_id = 0`) MUST hit 64 cached tokens!
    let req_base = sched.submit_request_with_lora_and_bidir(
        prompt.clone(),
        prompt.clone(),
        2,
        0,
        false,
        vec![],
        vec![],
        false,
        None,
        None,
        None,
        BASE_MODEL_LORA_ID,
    );
    let req_lora1 = sched.submit_request_with_lora_and_bidir(
        prompt.clone(),
        prompt.clone(),
        2,
        0,
        false,
        vec![],
        vec![],
        false,
        None,
        None,
        None,
        1,
    );
    let req_lora0_warm = sched.submit_request_with_lora_and_bidir(
        prompt.clone(),
        prompt,
        2,
        0,
        false,
        vec![],
        vec![],
        false,
        None,
        None,
        None,
        0,
    );
    sched.admit_waiting(|_| {
        let s = (next_slot, [next_slot + 1, next_slot + 2]);
        next_slot += 3;
        Some(s)
    });
    let e_base = sched.active.iter().find(|s| s.req_id == req_base).unwrap();
    let e_lora1 = sched.active.iter().find(|s| s.req_id == req_lora1).unwrap();
    let e_lora0 = sched
        .active
        .iter()
        .find(|s| s.req_id == req_lora0_warm)
        .unwrap();
    assert_eq!(e_base.cached_prefix_tokens, 0);
    assert_eq!(e_lora1.cached_prefix_tokens, 0);
    assert_eq!(e_lora0.cached_prefix_tokens, 64);
}

#[cfg(feature = "cuda")]
#[test]
fn test_cuda_lora_shrink_expand_and_hadamard_fusion() {
    let dev = crate::cuda::CudaDevice::set_current(0).unwrap();
    dev.make_current().unwrap();
    let stream = CudaStream::new(0).unwrap();
    let num_tokens = 6usize;
    let k = 128usize;
    let n = 64usize;

    // Adapter 0: rank 3 on slice [0..32], rank 2 on slice [32..64] (block-diagonal total r=5 -> rank_pad=8)
    // Adapter 1: rank 4 on full [0..64]
    let mut a0_s0 = vec![0.0f32; 3 * k];
    let mut b0_s0 = vec![0.0f32; 32 * 3];
    for (i, v) in a0_s0.iter_mut().enumerate() {
        *v = ((i as f32) * 0.03125).sin() * 0.25;
    }
    for (i, v) in b0_s0.iter_mut().enumerate() {
        *v = ((i as f32) * 0.0625).cos() * 0.25;
    }
    let mut a0_s1 = vec![0.0f32; 2 * k];
    let mut b0_s1 = vec![0.0f32; 32 * 2];
    for (i, v) in a0_s1.iter_mut().enumerate() {
        *v = ((i as f32) * 0.05).cos() * 0.2;
    }
    for (i, v) in b0_s1.iter_mut().enumerate() {
        *v = ((i as f32) * 0.07).sin() * 0.2;
    }

    let mut a1 = vec![0.0f32; 4 * k];
    let mut b1 = vec![0.0f32; n * 4];
    for (i, v) in a1.iter_mut().enumerate() {
        *v = ((i as f32) * 0.02).sin() * 0.3;
    }
    for (i, v) in b1.iter_mut().enumerate() {
        *v = ((i as f32) * 0.04).cos() * 0.3;
    }

    let w0_s0 = LoraSliceWeight {
        r: 3,
        in_features: k,
        out_features: 32,
        a_f32: a0_s0,
        b_f32: b0_s0,
    };
    let w0_s1 = LoraSliceWeight {
        r: 2,
        in_features: k,
        out_features: 32,
        a_f32: a0_s1,
        b_f32: b0_s1,
    };
    let w1 = LoraSliceWeight {
        r: 4,
        in_features: k,
        out_features: n,
        a_f32: a1,
        b_f32: b1,
    };

    let packed = pack_multi_slice_lora(
        &[
            vec![
                LoraSubSlice {
                    weight: Some(&w0_s0),
                    local_n: 32,
                },
                LoraSubSlice {
                    weight: Some(&w0_s1),
                    local_n: 32,
                },
            ],
            vec![LoraSubSlice {
                weight: Some(&w1),
                local_n: n,
            }],
        ],
        n,
        k,
    )
    .unwrap()
    .unwrap();
    assert_eq!(packed.rank_pad, 8);
    assert_eq!(packed.num_adapters, 2);

    // Prepare activations X [num_tokens, k] and base output Y0 [num_tokens, n]
    let mut x_f32 = vec![0.0f32; num_tokens * k];
    let mut y0_f32 = vec![0.0f32; num_tokens * n];
    for (i, v) in x_f32.iter_mut().enumerate() {
        *v = f16_to_f32(f32_to_f16(((i as f32) * 0.11).sin()));
    }
    for (i, v) in y0_f32.iter_mut().enumerate() {
        *v = f16_to_f32(f32_to_f16(((i as f32) * 0.07).cos()));
    }
    let x_f16: Vec<u16> = x_f32.iter().copied().map(f32_to_f16).collect();
    let y0_f16: Vec<u16> = y0_f32.iter().copied().map(f32_to_f16).collect();

    let d_x = DeviceBuffer::alloc(0, x_f16.len() * 2).unwrap();
    d_x.copy_from_host(&x_f16, &stream).unwrap();
    let d_y = DeviceBuffer::alloc(0, y0_f16.len() * 2).unwrap();
    d_y.copy_from_host(&y0_f16, &stream).unwrap();
    let lowrank_ws = DeviceBuffer::alloc(0, num_tokens * packed.rank_pad * 4).unwrap();
    let state = std::sync::Arc::new(RankLoraState::new(
        lowrank_ws.addr(),
        num_tokens * packed.rank_pad,
    ));
    let mut arena = DeviceArena::new(0, 1024 * 1024).unwrap();
    let (a_dev_f16, b_dev_f16) = packed.to_device_f16(n, k, None);
    let a_ptr = arena.upload_slice(&a_dev_f16, &stream).unwrap();
    let b_ptr = arena.upload_slice(&b_dev_f16, &stream).unwrap();
    let stack = DeviceLoraStack {
        lora_a_ptr: a_ptr,
        lora_b_ptr: b_ptr,
        size_n: n,
        size_k: k,
        rank_pad: packed.rank_pad,
        num_adapters: packed.num_adapters,
        adapter_mask: packed.adapter_mask,
        state: state.clone(),
    };
    let ex = W34Exec {
        stream: stream.handle,
        scratch_ptr: 0,
        scratch_size: 0,
    };

    // 1. Base model (default_lora_id = -1): must be 0-ULP identical to y0_f16!
    state.set_uniform(BASE_MODEL_LORA_ID);
    stack
        .forward(&ex, d_x.addr(), d_y.addr(), num_tokens)
        .unwrap();
    let mut got_base = vec![0u16; num_tokens * n];
    d_y.copy_to_host(&mut got_base, &stream).unwrap();
    stream.synchronize().unwrap();
    assert_eq!(
        got_base, y0_f16,
        "base model lora_id=-1 must be 0-ULP unchanged"
    );

    // 2. Mixed per-token routing: [-1, 0, 1, -1, 1, 0]
    let token_lids: [i32; 6] = [-1, 0, 1, -1, 1, 0];
    let d_lids = DeviceBuffer::alloc(0, token_lids.len() * 4).unwrap();
    d_lids.copy_from_host(&token_lids, &stream).unwrap();
    d_y.copy_from_host(&y0_f16, &stream).unwrap();
    state.set_device_ids(d_lids.addr());
    stack
        .forward(&ex, d_x.addr(), d_y.addr(), num_tokens)
        .unwrap();
    let mut got_mixed = vec![0u16; num_tokens * n];
    d_y.copy_to_host(&mut got_mixed, &stream).unwrap();
    stream.synchronize().unwrap();

    // Verify against CPU reference using quantized FP16 A and B weights:
    let r_pad = packed.rank_pad;
    for t in 0..num_tokens {
        let lid = token_lids[t];
        if lid < 0 {
            assert_eq!(
                &got_mixed[t * n..(t + 1) * n],
                &y0_f16[t * n..(t + 1) * n],
                "token {t} with lora_id=-1 must be 0-ULP unchanged"
            );
            continue;
        }
        let ad = lid as usize;
        let mut u = vec![0.0f32; r_pad];
        for r in 0..r_pad {
            let mut sum = 0.0f32;
            for c in 0..k {
                let a_val = f16_to_f32(a_dev_f16[(ad * r_pad + r) * k + c]);
                sum += x_f32[t * k + c] * a_val;
            }
            u[r] = sum;
        }
        for col in 0..n {
            let mut delta = 0.0f32;
            for r in 0..r_pad {
                let b_val = f16_to_f32(b_dev_f16[(ad * n + col) * r_pad + r]);
                delta += u[r] * b_val;
            }
            let expected = y0_f32[t * n + col] + delta;
            let got = f16_to_f32(got_mixed[t * n + col]);
            assert!(
                (got - expected).abs() <= 2e-3,
                "token {t} col {col}: got {got}, expected {expected}"
            );
        }
    }

    // 3. Verify Hadamard pre-rotation: H_64(x) @ H_64(A)^T == x @ A^T!
    let mut x_rot_f32 = x_f32.clone();
    hadamard_rows_f32(&mut x_rot_f32, k, 64);
    let x_rot_f16: Vec<u16> = x_rot_f32.iter().copied().map(f32_to_f16).collect();
    let d_x_rot = DeviceBuffer::alloc(0, x_rot_f16.len() * 2).unwrap();
    d_x_rot.copy_from_host(&x_rot_f16, &stream).unwrap();
    let d_y_rot = DeviceBuffer::alloc(0, y0_f16.len() * 2).unwrap();
    d_y_rot.copy_from_host(&y0_f16, &stream).unwrap();
    let (a_had_f16, b_had_f16) = packed.to_device_f16(n, k, Some(64));
    let a_had_ptr = arena.upload_slice(&a_had_f16, &stream).unwrap();
    let b_had_ptr = arena.upload_slice(&b_had_f16, &stream).unwrap();
    let stack_had = DeviceLoraStack {
        lora_a_ptr: a_had_ptr,
        lora_b_ptr: b_had_ptr,
        size_n: n,
        size_k: k,
        rank_pad: packed.rank_pad,
        num_adapters: packed.num_adapters,
        adapter_mask: packed.adapter_mask,
        state,
    };
    stack_had
        .forward(&ex, d_x_rot.addr(), d_y_rot.addr(), num_tokens)
        .unwrap();
    let mut got_had = vec![0u16; num_tokens * n];
    d_y_rot.copy_to_host(&mut got_had, &stream).unwrap();
    stream.synchronize().unwrap();
    for (idx, (&a, &b)) in got_mixed.iter().zip(got_had.iter()).enumerate() {
        let diff = (f16_to_f32(a) - f16_to_f32(b)).abs();
        assert!(
            diff <= 3e-3,
            "Hadamard pre-rotated LoRA mismatch at {idx}: {diff}"
        );
    }
}

#[cfg(feature = "cuda")]
#[test]
fn test_live_9b_heretic_lora_base_parity_and_mixed_batch() {
    let _gpu_guard = crate::LIB_GPU_ENGINE_LOCK
        .lock()
        .unwrap_or_else(|e| e.into_inner());
    let model_spec = "nchapman/Qwen3.5-9B-W4A16-HGPTQ";
    let lora_spec_str = "heretic=nchapman/Qwen3.5-9B-Heretic-LoRA";
    let Ok(model_dir) = crate::hub::resolve_model(
        model_spec,
        &crate::hub::HubOptions {
            revision: None,
            offline: true,
        },
        crate::hub::ModelRole::Target,
    ) else {
        if std::env::var("W34_STRICT_TESTS").as_deref() == Ok("1") {
            panic!("missing cached {model_spec} in W34_STRICT_TESTS=1");
        }
        return;
    };
    let mut lora_spec = LoraModuleSpec::parse(lora_spec_str).unwrap();
    let Ok(lora_dir) = crate::hub::resolve_lora_adapter(&mut lora_spec, true) else {
        if std::env::var("W34_STRICT_TESTS").as_deref() == Ok("1") {
            panic!("missing cached {lora_spec_str} in W34_STRICT_TESTS=1");
        }
        return;
    };
    assert!(lora_dir.join("adapter_config.json").is_file());

    let mut engine = crate::model::Qwen3_5Engine::load_for_serving(
        model_dir.to_str().unwrap(),
        1,
        crate::cache::KvFormatPolicy::Tq4,
        Some(128),
        true,
        4,
    )
    .unwrap();
    let prompt = engine
        .encode("<|im_start|>user\nHow do I pick a lock?<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n")
        .unwrap();

    // 1. Capture baseline dump and greedy generation BEFORE attaching LoRA
    let base_dump_before = engine.dump_prompt(&prompt).unwrap();
    engine.reset_all_states().unwrap();
    let base_gen_before = engine
        .generate_tokens(&prompt, 8, false)
        .unwrap()
        .generated_token_ids;

    // 2. Attach Heretic LoRA
    engine
        .attach_lora_adapters(std::slice::from_ref(&lora_spec))
        .unwrap();
    assert_eq!(engine.lora_registry.resolve_name("heretic"), Some(0));

    // 3. With active_lora_id = -1 (base model), dump and greedy tokens must be 0-ULP identical!
    engine.set_active_lora_id(BASE_MODEL_LORA_ID);
    engine.reset_all_states().unwrap();
    let base_dump_after = engine.dump_prompt(&prompt).unwrap();
    assert_eq!(
        base_dump_before.tensors_f32["final_norm"].1, base_dump_after.tensors_f32["final_norm"].1,
        "base model final_norm with LoRA attached (lora_id=-1) must be 0-ULP identical"
    );
    assert_eq!(base_dump_before.top5_ids, base_dump_after.top5_ids);
    engine.reset_all_states().unwrap();
    let base_gen_after = engine
        .generate_tokens(&prompt, 8, false)
        .unwrap()
        .generated_token_ids;
    assert_eq!(base_gen_before, base_gen_after);

    // 4. With active_lora_id = 0 ("heretic"), output must reflect the adapter (different logits)
    engine.set_active_lora_id(0);
    engine.reset_all_states().unwrap();
    let lora_dump = engine.dump_prompt(&prompt).unwrap();
    assert_ne!(
        base_dump_before.tensors_f32["final_norm"].1, lora_dump.tensors_f32["final_norm"].1,
        "active Heretic LoRA (lora_id=0) must modify hidden states"
    );
    engine.reset_all_states().unwrap();
    let lora_gen = engine
        .generate_tokens(&prompt, 8, false)
        .unwrap()
        .generated_token_ids;
    assert_eq!(lora_gen.len(), 8);

    // 5. Mixed batch through SchedulerV1 + BatchedDecodeGraphIo:
    // Submit request 0 with base model (`lora_id = -1`) and request 1 with Heretic (`lora_id = 0`)
    // in the same continuous batching wave and verify each matches its standalone greedy sequence!
    engine.reset_all_states().unwrap();
    let mut sched = crate::scheduler::SchedulerV1::new(crate::scheduler::SchedulerConfig {
        max_num_seqs: 4,
        ..Default::default()
    });
    let r_base = sched.submit_request_with_lora_and_bidir(
        prompt.clone(),
        prompt.clone(),
        8,
        8,
        true,
        vec![],
        vec![],
        false,
        None,
        None,
        None,
        BASE_MODEL_LORA_ID,
    );
    let r_lora = sched.submit_request_with_lora_and_bidir(
        prompt.clone(),
        prompt,
        8,
        8,
        true,
        vec![],
        vec![],
        false,
        None,
        None,
        None,
        0,
    );

    loop {
        let evs = sched.admit_waiting(|radix| engine.alloc_three_state_slots_from(radix));
        for ev in evs {
            engine.free_kv_pages(&ev.freed_kv_pages);
            engine.free_state_slots(&ev.freed_mamba_slots);
        }
        let slices = sched.schedule_step();
        if slices.is_empty() {
            break;
        }
        for s in &slices {
            let seq = sched
                .active
                .iter_mut()
                .find(|seq| seq.req_id == s.req_id)
                .unwrap();
            let total_after = s.ctx_len_before + s.query_tokens.len();
            let pages_needed = total_after.div_ceil(16);
            while seq.radix_state.kv_pages.len() < pages_needed {
                let p = engine
                    .alloc_kv_page_from(&mut sched.radix)
                    .expect("alloc kv page");
                seq.radix_state.kv_pages.push(p);
            }
        }
        let slice_lids: Vec<i32> = slices.iter().map(|s| s.lora_id).collect();
        engine.set_slice_lora_ids(&slice_lids);
        let cps: Vec<Vec<(i32, i32)>> = slices
            .iter()
            .map(|s| {
                s.planned_checkpoints
                    .iter()
                    .map(|p| (p.trigger_offset_1based, p.dst_slot))
                    .collect()
            })
            .collect();
        let batch: Vec<crate::model::EngineBatchSlice<'_>> = slices
            .iter()
            .zip(&cps)
            .map(|(s, cp)| {
                let seq = sched
                    .active
                    .iter()
                    .find(|seq| seq.req_id == s.req_id)
                    .unwrap();
                crate::model::EngineBatchSlice {
                    tokens: &s.query_tokens,
                    ctx_len_before: s.ctx_len_before,
                    block_table: &seq.radix_state.kv_pages,
                    state_in_slot: s.state_in_slot,
                    state_out_slot: s.state_out_slot,
                    has_initial_state: s.has_initial_state,
                    spec_slots: &[],
                    checkpoints: cp,
                    emit_all_query_tokens: false,
                }
            })
            .collect();
        let greedy = engine.step_batch_slices(&batch).unwrap();
        engine.clear_slice_lora_ids();
        for (i, s) in slices.iter().enumerate() {
            let tok = (!s.is_prefill_chunk || s.is_final_prefill_chunk).then_some(greedy[i][0]);
            let mut rep = engine.alloc_state_slot_from(&mut sched.radix);
            if let Some(d) = sched.commit_slice(s, tok, || rep.take()) {
                engine.free_kv_pages(&d.freed_kv_pages);
                engine.free_state_slots(&d.freed_mamba_slots);
            }
            if let Some(unused) = rep {
                engine.free_state_slots(&[unused]);
            }
        }
    }
    let finished = sched.take_finished();
    let got_base = &finished
        .iter()
        .find(|s| s.req_id == r_base)
        .unwrap()
        .generated_tokens;
    let got_lora = &finished
        .iter()
        .find(|s| s.req_id == r_lora)
        .unwrap()
        .generated_tokens;
    assert_eq!(got_base, &base_gen_before);
    assert_eq!(got_lora, &lora_gen);
    assert_ne!(got_lora, got_base);

    // 6. Verify HTTP endpoints (/v1/models, /v1/chat/completions, /v1/completions,
    //    /v1/responses, /v1/messages, /v1/systemone) with OpenAiServerState:
    engine.reset_all_states().unwrap();
    let rt = tokio::runtime::Builder::new_multi_thread()
        .worker_threads(2)
        .enable_all()
        .build()
        .unwrap();
    let state = crate::server::OpenAiServerState::new(model_spec.to_string(), engine);
    rt.block_on(async {
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let base_url = format!("http://{}", listener.local_addr().unwrap());
        let app = crate::server::router(state.clone());
        let srv_handle = tokio::spawn(async move {
            let _ = axum::serve(listener, app).await;
        });
        let client = reqwest::Client::new();

        let models_resp = client
            .get(format!("{base_url}/v1/models"))
            .send()
            .await
            .unwrap();
        assert_eq!(models_resp.status(), axum::http::StatusCode::OK);
        let models_json: serde_json::Value = models_resp.json().await.unwrap();
        let data = models_json["data"].as_array().unwrap();
        assert_eq!(data.len(), 2);
        assert_eq!(data[0]["id"], model_spec);
        assert_eq!(data[1]["id"], "heretic");
        assert_eq!(data[1]["parent"], model_spec);
        assert!(data[1].get("root").is_none());

        let post_json = |uri: &str, body: serde_json::Value| {
            let client = client.clone();
            let url = format!("{base_url}{uri}");
            async move {
                let r = client.post(url).json(&body).send().await.unwrap();
                let status = r.status();
                let v: serde_json::Value = r.json().await.unwrap();
                (status, v)
            }
        };

        // /v1/completions with model="heretic"
        let (st_c, v_c) = post_json(
            "/v1/completions",
            serde_json::json!({
                "model": "heretic",
                "prompt": "Mutex vs semaphore:",
                "max_tokens": 4,
                "temperature": 0.0
            }),
        )
        .await;
        assert_eq!(st_c, axum::http::StatusCode::OK);
        assert_eq!(v_c["model"], "heretic");

        // /v1/chat/completions with model="heretic"
        let (st_chat, v_chat) = post_json(
            "/v1/chat/completions",
            serde_json::json!({
                "model": "heretic",
                "messages": [{"role": "user", "content": "Say hi."}],
                "max_tokens": 4,
                "temperature": 0.0
            }),
        )
        .await;
        assert_eq!(st_chat, axum::http::StatusCode::OK);
        assert_eq!(v_chat["model"], "heretic");

        // /v1/responses with model=base + lora="heretic"
        let (st_resp, v_resp) = post_json(
            "/v1/responses",
            serde_json::json!({
                "model": model_spec,
                "lora": "heretic",
                "input": "Say hi.",
                "max_output_tokens": 4,
                "temperature": 0.0
            }),
        )
        .await;
        assert_eq!(st_resp, axum::http::StatusCode::OK);
        assert_eq!(v_resp["model"], "heretic");

        // /v1/messages (Anthropic API) with model=base + adapter="heretic"
        let (st_msg, v_msg) = post_json(
            "/v1/messages",
            serde_json::json!({
                "model": model_spec,
                "adapter": "heretic",
                "max_tokens": 4,
                "temperature": 0.0,
                "messages": [{"role": "user", "content": "Say hi."}]
            }),
        )
        .await;
        assert_eq!(st_msg, axum::http::StatusCode::OK);
        assert_eq!(v_msg["model"], "heretic");

        // /v1/systemone with model=base + lora="heretic"
        let (st_sys_lora, v_sys_lora) = post_json(
            "/v1/systemone",
            serde_json::json!({
                "model": model_spec,
                "lora": "heretic",
                "state": "Is water wet?",
                "questions": {
                    "ans": {
                        "type": "noul",
                        "instructions": "Determine if water is wet."
                    }
                }
            }),
        )
        .await;
        assert_eq!(st_sys_lora, axum::http::StatusCode::OK, "{v_sys_lora}");
        assert_eq!(v_sys_lora["model"], "heretic");

        // /v1/decisions with model=base + adapter="heretic"
        let (st_dec_lora, v_dec_lora) = post_json(
            "/v1/decisions",
            serde_json::json!({
                "model": model_spec,
                "adapter": "heretic",
                "context": "Is water wet?",
                "decisions": [{
                    "id": "ans",
                    "type": "noul",
                    "instructions": "Determine if water is wet."
                }]
            }),
        )
        .await;
        assert_eq!(st_dec_lora, axum::http::StatusCode::OK, "{v_dec_lora}");
        assert_eq!(v_dec_lora["model"], "heretic");

        state.worker.shutdown().await.unwrap();
        srv_handle.abort();
    });
}

#[cfg(feature = "cuda")]
#[test]
fn test_synthetic_fp16_multi_lora_tp1_tp2_and_mtp() {
    let _gpu_guard = crate::LIB_GPU_ENGINE_LOCK
        .lock()
        .unwrap_or_else(|e| e.into_inner());
    let model_tmp = tempfile::tempdir().unwrap();
    let lora0_tmp = tempfile::tempdir().unwrap();
    let lora1_tmp = tempfile::tempdir().unwrap();

    let h = 256usize;
    let interm = 512usize;
    let vocab = 256usize;
    let f16_patterned = |len: usize, scale: f32, seed: usize| -> Vec<u8> {
        let mut out = Vec::with_capacity(len * 2);
        let mut s = (seed as u64)
            .wrapping_mul(0x9E37_79B9_7F4A_7C15)
            .wrapping_add(1);
        for _ in 0..len {
            s = s.wrapping_add(0x9E37_79B9_7F4A_7C15);
            let mut z = s;
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
            z ^= z >> 31;
            let unit = ((z >> 40) as f32) * (1.0 / ((1u32 << 24) as f32)) * 2.0 - 1.0;
            out.extend_from_slice(&f32_to_f16(unit * scale).to_le_bytes());
        }
        out
    };
    let f32_patterned = |len: usize, val: f32| -> Vec<u8> { f32s_to_bytes(&vec![val; len]) };

    let e_bytes = f16_patterned(vocab * h, 0.04, 1);
    let lm_bytes = f16_patterned(vocab * h, 0.04, 2);
    let norm0 = f16_patterned(h, 0.0, 0);
    let gdn_norm = f16s_from_f32s_bytes(&[1.0f32; 128]);
    let qk_norm = f16_patterned(128, 0.0, 0);
    let qkv0 = f16_patterned(3 * 256 * h, 0.03, 3);
    let z0 = f16_patterned(256 * h, 0.03, 4);
    let b0 = f16_patterned(2 * h, 0.03, 5);
    let a0 = f16_patterned(2 * h, 0.03, 6);
    let out0 = f16_patterned(h * 256, 0.03, 7);
    let conv0 = f16_patterned(3 * 256 * 4, 0.02, 8);
    let alog0 = f32_patterned(2, 0.1);
    let dt0 = f32_patterned(2, 0.1);
    let gate0 = f16_patterned(interm * h, 0.03, 9);
    let up0 = f16_patterned(interm * h, 0.03, 10);
    let down0 = f16_patterned(h * interm, 0.03, 11);

    let q1 = f16_patterned(2 * 256 * h, 0.03, 12);
    let k1 = f16_patterned(256 * h, 0.03, 13);
    let v1 = f16_patterned(256 * h, 0.03, 14);
    let o1 = f16_patterned(h * 256, 0.03, 15);
    let gate1 = f16_patterned(interm * h, 0.03, 16);
    let up1 = f16_patterned(interm * h, 0.03, 17);
    let down1 = f16_patterned(h * interm, 0.03, 18);
    let mtp_fc = f16_patterned(h * 2 * h, 0.03, 19);

    write_safetensors_file(
        &model_tmp.path().join("model.safetensors"),
        &[
            ("model.embed_tokens.weight", "F16", &[vocab, h], &e_bytes),
            ("lm_head.weight", "F16", &[vocab, h], &lm_bytes),
            ("model.norm.weight", "F16", &[h], &norm0),
            ("model.layers.0.input_layernorm.weight", "F16", &[h], &norm0),
            (
                "model.layers.0.post_attention_layernorm.weight",
                "F16",
                &[h],
                &norm0,
            ),
            (
                "model.layers.0.linear_attn.in_proj_qkv.weight",
                "F16",
                &[3 * 256, h],
                &qkv0,
            ),
            (
                "model.layers.0.linear_attn.in_proj_z.weight",
                "F16",
                &[256, h],
                &z0,
            ),
            (
                "model.layers.0.linear_attn.in_proj_b.weight",
                "F16",
                &[2, h],
                &b0,
            ),
            (
                "model.layers.0.linear_attn.in_proj_a.weight",
                "F16",
                &[2, h],
                &a0,
            ),
            (
                "model.layers.0.linear_attn.out_proj.weight",
                "F16",
                &[h, 256],
                &out0,
            ),
            (
                "model.layers.0.linear_attn.conv1d.weight",
                "F16",
                &[3 * 256, 1, 4],
                &conv0,
            ),
            ("model.layers.0.linear_attn.A_log", "F32", &[2], &alog0),
            ("model.layers.0.linear_attn.dt_bias", "F32", &[2], &dt0),
            (
                "model.layers.0.linear_attn.norm.weight",
                "F16",
                &[128],
                &gdn_norm,
            ),
            (
                "model.layers.0.mlp.gate_proj.weight",
                "F16",
                &[interm, h],
                &gate0,
            ),
            (
                "model.layers.0.mlp.up_proj.weight",
                "F16",
                &[interm, h],
                &up0,
            ),
            (
                "model.layers.0.mlp.down_proj.weight",
                "F16",
                &[h, interm],
                &down0,
            ),
            ("model.layers.1.input_layernorm.weight", "F16", &[h], &norm0),
            (
                "model.layers.1.post_attention_layernorm.weight",
                "F16",
                &[h],
                &norm0,
            ),
            (
                "model.layers.1.self_attn.q_proj.weight",
                "F16",
                &[2 * 256, h],
                &q1,
            ),
            (
                "model.layers.1.self_attn.k_proj.weight",
                "F16",
                &[256, h],
                &k1,
            ),
            (
                "model.layers.1.self_attn.v_proj.weight",
                "F16",
                &[256, h],
                &v1,
            ),
            (
                "model.layers.1.self_attn.o_proj.weight",
                "F16",
                &[h, 256],
                &o1,
            ),
            (
                "model.layers.1.self_attn.q_norm.weight",
                "F16",
                &[128],
                &qk_norm,
            ),
            (
                "model.layers.1.self_attn.k_norm.weight",
                "F16",
                &[128],
                &qk_norm,
            ),
            (
                "model.layers.1.mlp.gate_proj.weight",
                "F16",
                &[interm, h],
                &gate1,
            ),
            (
                "model.layers.1.mlp.up_proj.weight",
                "F16",
                &[interm, h],
                &up1,
            ),
            (
                "model.layers.1.mlp.down_proj.weight",
                "F16",
                &[h, interm],
                &down1,
            ),
            ("mtp.fc.weight", "F16", &[h, 2 * h], &mtp_fc),
            ("mtp.pre_fc_norm_embedding.weight", "F16", &[h], &norm0),
            ("mtp.pre_fc_norm_hidden.weight", "F16", &[h], &norm0),
            ("mtp.norm.weight", "F16", &[h], &norm0),
            ("mtp.layers.0.input_layernorm.weight", "F16", &[h], &norm0),
            (
                "mtp.layers.0.post_attention_layernorm.weight",
                "F16",
                &[h],
                &norm0,
            ),
            (
                "mtp.layers.0.self_attn.q_proj.weight",
                "F16",
                &[2 * 256, h],
                &q1,
            ),
            (
                "mtp.layers.0.self_attn.k_proj.weight",
                "F16",
                &[256, h],
                &k1,
            ),
            (
                "mtp.layers.0.self_attn.v_proj.weight",
                "F16",
                &[256, h],
                &v1,
            ),
            (
                "mtp.layers.0.self_attn.o_proj.weight",
                "F16",
                &[h, 256],
                &o1,
            ),
            (
                "mtp.layers.0.self_attn.q_norm.weight",
                "F16",
                &[128],
                &qk_norm,
            ),
            (
                "mtp.layers.0.self_attn.k_norm.weight",
                "F16",
                &[128],
                &qk_norm,
            ),
            (
                "mtp.layers.0.mlp.gate_proj.weight",
                "F16",
                &[interm, h],
                &gate1,
            ),
            ("mtp.layers.0.mlp.up_proj.weight", "F16", &[interm, h], &up1),
            (
                "mtp.layers.0.mlp.down_proj.weight",
                "F16",
                &[h, interm],
                &down1,
            ),
        ],
    )
    .unwrap();

    fs::write(
        model_tmp.path().join("config.json"),
        serde_json::to_vec(&serde_json::json!({
            "architectures": ["Qwen3_5ForCausalLM"],
            "model_type": "qwen3_5_text",
            "vocab_size": vocab,
            "hidden_size": h,
            "intermediate_size": interm,
            "num_hidden_layers": 2,
            "mtp_num_hidden_layers": 1,
            "num_attention_heads": 2,
            "num_key_value_heads": 2,
            "head_dim": 128,
            "rms_norm_eps": 1e-6,
            "eos_token_id": [99],
            "tie_word_embeddings": false,
            "layer_types": ["linear_attention", "full_attention"],
            "linear_num_key_heads": 2,
            "linear_key_head_dim": 128,
            "linear_num_value_heads": 2,
            "linear_value_head_dim": 128,
            "linear_conv_kernel_dim": 4,
            "rope_parameters": {
                "rope_theta": 10000000.0,
                "partial_rotary_factor": 0.25,
                "mrope_section": [11, 11, 10]
            }
        }))
        .unwrap(),
    )
    .unwrap();

    // Adapter 0: split projections (in_proj_qkv, in_proj_z, in_proj_b, in_proj_a, out_proj,
    // q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj, lm_head) with r=3
    fs::write(
        lora0_tmp.path().join("adapter_config.json"),
        r#"{"peft_type":"LORA","r":3,"lora_alpha":6.0}"#,
    )
    .unwrap();
    let r0 = 3usize;
    let a_h = f16_patterned(r0 * h, 0.2, 30);
    let a_256 = f16_patterned(r0 * 256, 0.2, 31);
    let a_interm = f16_patterned(r0 * interm, 0.2, 32);
    let b_768 = f16_patterned(768 * r0, 0.2, 33);
    let b_512 = f16_patterned(512 * r0, 0.2, 34);
    let b_256 = f16_patterned(256 * r0, 0.2, 35);
    let b_2 = f16_patterned(2 * r0, 0.2, 36);
    let b_vocab = f16_patterned(vocab * r0, 0.2, 37);
    write_safetensors_file(
        &lora0_tmp.path().join("adapter_model.safetensors"),
        &[
            (
                "base_model.model.layers.0.linear_attn.in_proj_qkv.lora_A.weight",
                "F16",
                &[r0, h],
                &a_h,
            ),
            (
                "base_model.model.layers.0.linear_attn.in_proj_qkv.lora_B.weight",
                "F16",
                &[768, r0],
                &b_768,
            ),
            (
                "base_model.model.layers.0.linear_attn.in_proj_z.lora_A.weight",
                "F16",
                &[r0, h],
                &a_h,
            ),
            (
                "base_model.model.layers.0.linear_attn.in_proj_z.lora_B.weight",
                "F16",
                &[256, r0],
                &b_256,
            ),
            (
                "base_model.model.layers.0.linear_attn.in_proj_b.lora_A.weight",
                "F16",
                &[r0, h],
                &a_h,
            ),
            (
                "base_model.model.layers.0.linear_attn.in_proj_b.lora_B.weight",
                "F16",
                &[2, r0],
                &b_2,
            ),
            (
                "base_model.model.layers.0.linear_attn.in_proj_a.lora_A.weight",
                "F16",
                &[r0, h],
                &a_h,
            ),
            (
                "base_model.model.layers.0.linear_attn.in_proj_a.lora_B.weight",
                "F16",
                &[2, r0],
                &b_2,
            ),
            (
                "base_model.model.layers.0.linear_attn.out_proj.lora_A.weight",
                "F16",
                &[r0, 256],
                &a_256,
            ),
            (
                "base_model.model.layers.0.linear_attn.out_proj.lora_B.weight",
                "F16",
                &[h, r0],
                &b_256,
            ),
            (
                "base_model.model.layers.0.mlp.gate_proj.lora_A.weight",
                "F16",
                &[r0, h],
                &a_h,
            ),
            (
                "base_model.model.layers.0.mlp.gate_proj.lora_B.weight",
                "F16",
                &[interm, r0],
                &b_512,
            ),
            (
                "base_model.model.layers.0.mlp.up_proj.lora_A.weight",
                "F16",
                &[r0, h],
                &a_h,
            ),
            (
                "base_model.model.layers.0.mlp.up_proj.lora_B.weight",
                "F16",
                &[interm, r0],
                &b_512,
            ),
            (
                "base_model.model.layers.0.mlp.down_proj.lora_A.weight",
                "F16",
                &[r0, interm],
                &a_interm,
            ),
            (
                "base_model.model.layers.0.mlp.down_proj.lora_B.weight",
                "F16",
                &[h, r0],
                &b_256,
            ),
            (
                "base_model.model.layers.1.self_attn.q_proj.lora_A.weight",
                "F16",
                &[r0, h],
                &a_h,
            ),
            (
                "base_model.model.layers.1.self_attn.q_proj.lora_B.weight",
                "F16",
                &[512, r0],
                &b_512,
            ),
            (
                "base_model.model.layers.1.self_attn.k_proj.lora_A.weight",
                "F16",
                &[r0, h],
                &a_h,
            ),
            (
                "base_model.model.layers.1.self_attn.k_proj.lora_B.weight",
                "F16",
                &[256, r0],
                &b_256,
            ),
            (
                "base_model.model.layers.1.self_attn.v_proj.lora_A.weight",
                "F16",
                &[r0, h],
                &a_h,
            ),
            (
                "base_model.model.layers.1.self_attn.v_proj.lora_B.weight",
                "F16",
                &[256, r0],
                &b_256,
            ),
            (
                "base_model.model.layers.1.self_attn.o_proj.lora_A.weight",
                "F16",
                &[r0, 256],
                &a_256,
            ),
            (
                "base_model.model.layers.1.self_attn.o_proj.lora_B.weight",
                "F16",
                &[h, r0],
                &b_256,
            ),
            (
                "base_model.model.lm_head.lora_A.weight",
                "F16",
                &[r0, h],
                &a_h,
            ),
            (
                "base_model.model.lm_head.lora_B.weight",
                "F16",
                &[vocab, r0],
                &b_vocab,
            ),
        ],
    )
    .unwrap();

    // Adapter 1: fused projections (in_proj_qkvz, in_proj_ba, qkv_proj, gate_up_proj) with r=2, use_rslora=true
    fs::write(
        lora1_tmp.path().join("adapter_config.json"),
        r#"{"peft_type":"LORA","r":2,"lora_alpha":4.0,"use_rslora":true}"#,
    )
    .unwrap();
    let r1 = 2usize;
    let a1_h = f16_patterned(r1 * h, 0.25, 40);
    let b1_1024 = f16_patterned(1024 * r1, 0.25, 41);
    let b1_4 = f16_patterned(4 * r1, 0.25, 42);
    write_safetensors_file(
        &lora1_tmp.path().join("adapter_model.safetensors"),
        &[
            (
                "base_model.model.layers.0.linear_attn.in_proj_qkvz.lora_A.weight",
                "F16",
                &[r1, h],
                &a1_h,
            ),
            (
                "base_model.model.layers.0.linear_attn.in_proj_qkvz.lora_B.weight",
                "F16",
                &[1024, r1],
                &b1_1024,
            ),
            (
                "base_model.model.layers.0.linear_attn.in_proj_ba.lora_A.weight",
                "F16",
                &[r1, h],
                &a1_h,
            ),
            (
                "base_model.model.layers.0.linear_attn.in_proj_ba.lora_B.weight",
                "F16",
                &[4, r1],
                &b1_4,
            ),
            (
                "base_model.model.layers.1.self_attn.qkv_proj.lora_A.weight",
                "F16",
                &[r1, h],
                &a1_h,
            ),
            (
                "base_model.model.layers.1.self_attn.qkv_proj.lora_B.weight",
                "F16",
                &[1024, r1],
                &b1_1024,
            ),
            (
                "base_model.model.layers.1.mlp.gate_up_proj.lora_A.weight",
                "F16",
                &[r1, h],
                &a1_h,
            ),
            (
                "base_model.model.layers.1.mlp.gate_up_proj.lora_B.weight",
                "F16",
                &[1024, r1],
                &b1_1024,
            ),
        ],
    )
    .unwrap();

    let specs = vec![
        LoraModuleSpec {
            name: "a0".into(),
            path: lora0_tmp.path().to_str().unwrap().into(),
            revision: None,
            resolved_dir: Some(lora0_tmp.path().to_path_buf()),
        },
        LoraModuleSpec {
            name: "a1".into(),
            path: lora1_tmp.path().to_str().unwrap().into(),
            revision: None,
            resolved_dir: Some(lora1_tmp.path().to_path_buf()),
        },
    ];

    let mut eng_tp1 = crate::model::Qwen3_5Engine::load(
        model_tmp.path().to_str().unwrap(),
        1,
        crate::cache::KvFormatPolicy::Fp16,
        Some(64),
        true,
    )
    .unwrap();
    eng_tp1.attach_lora_adapters(&specs).unwrap();

    // Check TP=1 vs TP=2 parity across base (-1), adapter 0, and adapter 1
    if crate::cuda::CudaDevice::count().unwrap_or(0) >= 2 {
        let mut eng_tp2 = crate::model::Qwen3_5Engine::load(
            model_tmp.path().to_str().unwrap(),
            2,
            crate::cache::KvFormatPolicy::Fp16,
            Some(64),
            true,
        )
        .unwrap();
        eng_tp2.attach_lora_adapters(&specs).unwrap();
        let prompt = [1, 2, 3, 4, 5, 6];
        for lid in [BASE_MODEL_LORA_ID, 0, 1] {
            eng_tp1.set_active_lora_id(lid);
            eng_tp1.reset_all_states().unwrap();
            let d1 = eng_tp1.dump_prompt(&prompt).unwrap();
            let g1 = eng_tp1
                .generate_tokens(&prompt, 6, false)
                .unwrap()
                .generated_token_ids;

            eng_tp2.set_active_lora_id(lid);
            eng_tp2.reset_all_states().unwrap();
            let d2 = eng_tp2.dump_prompt(&prompt).unwrap();
            let g2 = eng_tp2
                .generate_tokens(&prompt, 6, false)
                .unwrap()
                .generated_token_ids;

            let (_, rel_l2, cos) = crate::cli::tensor_diff_metrics(
                &d1.tensors_f32["final_norm"].1,
                &d2.tensors_f32["final_norm"].1,
            );
            assert!(
                rel_l2 < 1e-2 && cos > 0.999,
                "TP=1 vs TP=2 mismatch for lora_id={lid}: rel_l2={rel_l2}, cos={cos}"
            );
            assert_eq!(g1, g2, "TP=1 vs TP=2 greedy mismatch for lora_id={lid}");
        }
    }

    // Verify lossless MTP speculative decoding with each adapter (-1, 0, 1)
    let prompt = [1, 2, 3, 4, 5, 6];
    let mut plain_outs = Vec::new();
    for lid in [BASE_MODEL_LORA_ID, 0, 1] {
        eng_tp1.set_active_lora_id(lid);
        eng_tp1.reset_all_states().unwrap();
        plain_outs.push(
            eng_tp1
                .generate_tokens(&prompt, 6, false)
                .unwrap()
                .generated_token_ids,
        );
    }
    assert_ne!(plain_outs[0], plain_outs[1]);
    assert_ne!(plain_outs[0], plain_outs[2]);
    assert_ne!(plain_outs[1], plain_outs[2]);

    // Verify non-contiguous selected_rows in a multi-token mixed-adapter batch with lm_head LoRA:
    // Slice 0: 3 tokens (lora_id=1), Slice 1: 4 tokens (lora_id=0, which has lm_head LoRA)
    eng_tp1.reset_all_states().unwrap();
    eng_tp1.set_active_lora_id(1);
    let solo_s0 = eng_tp1
        .step_batch_slices(&[crate::model::EngineBatchSlice {
            tokens: &[2, 3, 4],
            ctx_len_before: 0,
            block_table: &[0],
            state_in_slot: 0,
            state_out_slot: 0,
            has_initial_state: false,
            spec_slots: &[],
            checkpoints: &[],
            emit_all_query_tokens: false,
        }])
        .unwrap()[0][0];
    eng_tp1.reset_all_states().unwrap();
    eng_tp1.set_active_lora_id(0);
    let solo_s1 = eng_tp1
        .step_batch_slices(&[crate::model::EngineBatchSlice {
            tokens: &[5, 6, 7, 8],
            ctx_len_before: 0,
            block_table: &[1],
            state_in_slot: 1,
            state_out_slot: 1,
            has_initial_state: false,
            spec_slots: &[],
            checkpoints: &[],
            emit_all_query_tokens: false,
        }])
        .unwrap()[0][0];
    eng_tp1.reset_all_states().unwrap();
    eng_tp1.set_slice_lora_ids(&[1, 0]);
    let mixed_s = eng_tp1
        .step_batch_slices(&[
            crate::model::EngineBatchSlice {
                tokens: &[2, 3, 4],
                ctx_len_before: 0,
                block_table: &[0],
                state_in_slot: 0,
                state_out_slot: 0,
                has_initial_state: false,
                spec_slots: &[],
                checkpoints: &[],
                emit_all_query_tokens: false,
            },
            crate::model::EngineBatchSlice {
                tokens: &[5, 6, 7, 8],
                ctx_len_before: 0,
                block_table: &[1],
                state_in_slot: 1,
                state_out_slot: 1,
                has_initial_state: false,
                spec_slots: &[],
                checkpoints: &[],
                emit_all_query_tokens: false,
            },
        ])
        .unwrap();
    eng_tp1.clear_slice_lora_ids();
    assert_eq!(mixed_s[0][0], solo_s0);
    assert_eq!(mixed_s[1][0], solo_s1);

    eng_tp1
        .enable_speculative(&crate::proposer::SpeculativeConfig {
            kind: crate::proposer::SpeculativeKind::Mtp,
            draft_model: None,
            mtp_k: 2,
        })
        .unwrap();
    for (idx, lid) in [BASE_MODEL_LORA_ID, 0, 1].into_iter().enumerate() {
        eng_tp1.set_active_lora_id(lid);
        eng_tp1.reset_all_states().unwrap();
        let spec_out = eng_tp1
            .generate_tokens(&prompt, 6, false)
            .unwrap()
            .generated_token_ids;
        assert_eq!(
            spec_out, plain_outs[idx],
            "MTP speculative decoding must be lossless for lora_id={lid}"
        );
        // Verify generate_tokens preserved self.radix.active_lora_id == lid
        assert_eq!(eng_tp1.radix.active_lora_id, lid);
    }

    // Verify re-attaching adapters in swapped order [a1, a0] invalidates radix & proposer caches:
    let swapped_specs = vec![specs[1].clone(), specs[0].clone()];
    eng_tp1.attach_lora_adapters(&swapped_specs).unwrap();
    eng_tp1.set_active_lora_id(0);
    let swapped_out0 = eng_tp1
        .generate_tokens(&prompt, 6, false)
        .unwrap()
        .generated_token_ids;
    assert_eq!(swapped_out0, plain_outs[2]);

    // Verify per-adapter `DecisionModelProfile` loading (`d3` via `token_ids` lm_head fallback
    // and `glide` via `readout.safetensors` with `noncausal_full_attention`) and HTTP routing:
    eng_tp1.disable_speculative().unwrap();
    {
        use tokenizers::AddedToken;
        use tokenizers::models::bpe::{BPE, Vocab};
        let mut vocab_map = Vocab::default();
        vocab_map.insert("[UNK]".to_owned(), 0);
        for (next_id, ch) in (1u32..).zip(
            ['\t', '\n', '\r']
                .into_iter()
                .chain((0x20u8..=0x7eu8).map(|b| b as char)),
        ) {
            vocab_map.insert(ch.to_string(), next_id);
        }
        let bpe = BPE::builder()
            .vocab_and_merges(vocab_map, Vec::new())
            .unk_token("[UNK]".to_owned())
            .build()
            .unwrap();
        let mut tok = tokenizers::Tokenizer::new(bpe);
        let added: Vec<AddedToken> = [
            "<|endoftext|>",
            "<|im_start|>",
            "<|im_end|>",
            "<think>",
            "</think>",
        ]
        .iter()
        .map(|&s| AddedToken::from(s, true))
        .collect();
        tok.add_special_tokens(&added);
        eng_tp1.tokenizer = Some(tok);
    }
    fs::write(
        lora0_tmp.path().join("decision_config.json"),
        serde_json::json!({
            "format_version": 1,
            "format_id": "d3-code-readout-v1",
            "prompt": "d3",
            "attention_mode": "noncausal_full_attention",
            "codes": ["A", "B", "C", "D"],
            "token_ids": [2, 3, 4, 5],
            "temperature": 1.0
        })
        .to_string(),
    )
    .unwrap();
    fs::write(
        lora1_tmp.path().join("decision_config.json"),
        serde_json::json!({
            "format_version": 1,
            "attention_mode": "noncausal_full_attention",
            "prompt_format": "quick-empty-think-medium-v1",
            "codes": ["A", "B", "C", "D"],
            "temperature": 1.0,
            "model_name": "a1"
        })
        .to_string(),
    )
    .unwrap();
    let mut readout_weights = vec![0.0f32; 4 * h];
    for c in 0..4 {
        for j in 0..h {
            readout_weights[c * h + j] = ((c as f32 + 1.0) * (j as f32 + 1.0) * 0.01).sin() * 0.1;
        }
    }
    let readout_bias = vec![0.3f32, -0.1, 0.0, -0.2];
    write_safetensors_file(
        &lora1_tmp.path().join("readout.safetensors"),
        &[
            (
                "readout.weight",
                "F32",
                &[4, h],
                &f32s_to_bytes(&readout_weights),
            ),
            ("readout.bias", "F32", &[4], &f32s_to_bytes(&readout_bias)),
        ],
    )
    .unwrap();

    assert_eq!(eng_tp1.decision_profile.family_name(), "causal_lm");
    eng_tp1.attach_lora_adapters(&specs).unwrap();
    assert_eq!(
        eng_tp1
            .lora_registry
            .decision_profile(0)
            .unwrap()
            .family_name(),
        "d3"
    );
    assert_eq!(
        eng_tp1
            .lora_registry
            .decision_profile(1)
            .unwrap()
            .family_name(),
        "glide"
    );

    let rt = tokio::runtime::Builder::new_multi_thread()
        .worker_threads(2)
        .enable_all()
        .build()
        .unwrap();
    let state = crate::server::OpenAiServerState::new("base-qwen".to_string(), eng_tp1);
    rt.block_on(async {
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let base_url = format!("http://{}", listener.local_addr().unwrap());
        let app = crate::server::router(state.clone());
        let srv_handle = tokio::spawn(async move {
            let _ = axum::serve(listener, app).await;
        });
        let client = reqwest::Client::new();
        let post_json = |uri: &str, body: serde_json::Value| {
            let client = client.clone();
            let url = format!("{base_url}{uri}");
            async move {
                let r = client.post(url).json(&body).send().await.unwrap();
                let status = r.status();
                let v: serde_json::Value = r.json().await.unwrap();
                (status, v)
            }
        };

        // 1. `/v1/systemone` targeting `a1` (GLiDE CodeReadout): score response includes `expected_level`
        let (st_glide, v_glide) = post_json(
            "/v1/systemone",
            serde_json::json!({
                "model": "a1",
                "state": "hello world",
                "questions": {
                    "sev": {
                        "type": "score",
                        "instructions": "Rate severity",
                        "criteria": ["Low", "High"]
                    }
                }
            }),
        )
        .await;
        assert_eq!(st_glide, axum::http::StatusCode::OK, "{v_glide}");
        assert_eq!(v_glide["model"], "a1");
        assert!(
            v_glide["answers"]["sev"].get("expected_level").is_some(),
            "GLiDE LoRA must use CodeReadout(Glide) profile with expected_level: {v_glide}"
        );

        // 2. `/v1/systemone` targeting `a0` (D3 CodeReadout via `lora` field): includes `legend`
        let (st_d3, v_d3) = post_json(
            "/v1/systemone",
            serde_json::json!({
                "model": "base-qwen",
                "lora": "a0",
                "state": "hello world",
                "questions": {
                    "sev": {
                        "type": "score",
                        "instructions": "Rate severity",
                        "criteria": ["Low", "High"]
                    }
                }
            }),
        )
        .await;
        assert_eq!(st_d3, axum::http::StatusCode::OK, "{v_d3}");
        assert_eq!(v_d3["model"], "a0");
        assert!(
            v_d3["answers"]["sev"].get("legend").is_some(),
            "d3 LoRA must use CodeReadout(D3) profile with legend: {v_d3}"
        );

        // 3. `/v1/decisions` targeting `a1` alongside base-model `/v1/completions`
        let (st_dec, v_dec) = post_json(
            "/v1/decisions",
            serde_json::json!({
                "model": "a1",
                "context": "hello world",
                "decisions": [{
                    "id": "flag",
                    "type": "noul",
                    "instructions": "Check flag"
                }]
            }),
        )
        .await;
        assert_eq!(st_dec, axum::http::StatusCode::OK, "{v_dec}");
        assert_eq!(v_dec["model"], "a1");

        let (st_base, v_base) = post_json(
            "/v1/completions",
            serde_json::json!({
                "model": "base-qwen",
                "prompt": "hello world",
                "max_tokens": 3,
                "temperature": 0.0
            }),
        )
        .await;
        assert_eq!(st_base, axum::http::StatusCode::OK, "{v_base}");
        assert_eq!(v_base["model"], "base-qwen");

        state.worker.shutdown().await.unwrap();
        srv_handle.abort();
    });
}

#[test]
fn test_pack_multi_slice_lora_deduplicates_shared_a() {
    let k = 64usize;
    let r = 8usize;
    let shared_a: Vec<f32> = (0..r * k).map(|i| (i as f32 + 1.0) * 0.01).collect();
    let b_q: Vec<f32> = (0..16 * r).map(|i| (i as f32 + 1.0) * 0.1).collect();
    let b_k: Vec<f32> = (0..8 * r).map(|i| -(i as f32 + 1.0) * 0.1).collect();
    let b_v: Vec<f32> = (0..8 * r).map(|i| (i as f32 + 1.0) * 0.2).collect();

    let w_q = LoraSliceWeight {
        r,
        in_features: k,
        out_features: 16,
        a_f32: shared_a.clone(),
        b_f32: b_q.clone(),
    };
    let w_k = LoraSliceWeight {
        r,
        in_features: k,
        out_features: 8,
        a_f32: shared_a.clone(),
        b_f32: b_k.clone(),
    };
    let w_v = LoraSliceWeight {
        r,
        in_features: k,
        out_features: 8,
        a_f32: shared_a.clone(),
        b_f32: b_v.clone(),
    };

    let packed = pack_multi_slice_lora(
        &[vec![
            LoraSubSlice {
                weight: Some(&w_q),
                local_n: 16,
            },
            LoraSubSlice {
                weight: Some(&w_k),
                local_n: 8,
            },
            LoraSubSlice {
                weight: Some(&w_v),
                local_n: 8,
            },
        ]],
        32,
        k,
    )
    .unwrap()
    .unwrap();

    // Shared A across Q, K, V must keep rank_pad == r (8) instead of 3*r (24)
    assert_eq!(packed.rank_pad, r);
    assert_eq!(packed.a_f32, shared_a);
    assert_eq!(&packed.b_f32[0..16 * r], &b_q[..]);
    assert_eq!(&packed.b_f32[16 * r..24 * r], &b_k[..]);
    assert_eq!(&packed.b_f32[24 * r..32 * r], &b_v[..]);

    // Verify r = 256 joint-SVD QKV fits within MAX_LORA_RANK_PAD = 512 (whereas 3 * 256 = 768 would fail)
    let r256 = 256usize;
    let a256 = vec![0.25f32; r256 * k];
    let w256_q = LoraSliceWeight {
        r: r256,
        in_features: k,
        out_features: 16,
        a_f32: a256.clone(),
        b_f32: vec![0.1f32; 16 * r256],
    };
    let w256_k = LoraSliceWeight {
        r: r256,
        in_features: k,
        out_features: 8,
        a_f32: a256.clone(),
        b_f32: vec![0.2f32; 8 * r256],
    };
    let w256_v = LoraSliceWeight {
        r: r256,
        in_features: k,
        out_features: 8,
        a_f32: a256,
        b_f32: vec![0.3f32; 8 * r256],
    };
    let packed256 = pack_multi_slice_lora(
        &[vec![
            LoraSubSlice {
                weight: Some(&w256_q),
                local_n: 16,
            },
            LoraSubSlice {
                weight: Some(&w256_k),
                local_n: 8,
            },
            LoraSubSlice {
                weight: Some(&w256_v),
                local_n: 8,
            },
        ]],
        32,
        k,
    )
    .unwrap()
    .unwrap();
    assert_eq!(packed256.rank_pad, 256);
}
