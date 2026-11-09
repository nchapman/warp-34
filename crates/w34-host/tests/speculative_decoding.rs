//! End-to-end speculative decoding integration tests using real checkpoint proposals.
#![cfg(feature = "cuda")]
use w34_host::{
    cache::KvFormatPolicy,
    model::Qwen3_5Engine,
    proposer::{SpeculativeConfig, SpeculativeKind},
};
static GPU_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

fn lossless_cold_and_warm(kind: SpeculativeKind, tp: usize, large: bool) {
    let model = if large {
        "nchapman/Qwen3.8-27B-W4A16-GPTQ"
    } else {
        "nchapman/Qwen3.5-4B-W4A16-GPTQ"
    };
    if common::require_or_skip_cached_model(model, "lossless_cold_and_warm").is_none() {
        return;
    }
    if kind == SpeculativeKind::DFlash2 {
        let draft = if large {
            "incoai/Qwen3.8-27B-DFlash2"
        } else {
            "nchapman/Qwen3.5-4B-DFlash2"
        };
        if common::require_or_skip_cached_model(draft, "lossless_cold_and_warm").is_none() {
            return;
        }
    }
    let _gpu_guard = GPU_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    let mut engine = Qwen3_5Engine::load(model, tp, KvFormatPolicy::Fp16, Some(16), false).unwrap();
    let seed = engine.encode("The capital of France is").unwrap();
    let mut prompt = Vec::new();
    while prompt.len() < 130 {
        prompt.extend_from_slice(&seed);
    }
    prompt.truncate(130);
    let baseline = engine
        .generate_with_radix_cache(&prompt, 12, 64, false)
        .unwrap()
        .generated_token_ids;
    engine
        .enable_speculative(&SpeculativeConfig {
            kind,
            mtp_k: 3,
            draft_model: if kind == SpeculativeKind::DFlash2 {
                Some(
                    if large {
                        "incoai/Qwen3.8-27B-DFlash2"
                    } else {
                        "nchapman/Qwen3.5-4B-DFlash2"
                    }
                    .into(),
                )
            } else {
                None
            },
        })
        .unwrap();
    // Existing target-only cache entries cannot be mistaken for a complete
    // proposer hit: the first native request must replay the target cold.
    let cold = engine
        .generate_with_radix_cache(&prompt, 12, 64, false)
        .unwrap();
    assert_eq!(cold.cached_prefix_tokens, 0);
    assert_eq!(cold.generated_token_ids, baseline, "cold {kind:?}");
    assert!(
        engine
            .speculative
            .as_ref()
            .unwrap()
            .last_accepted
            .iter()
            .any(|&a| a >= 1),
        "cold {kind:?}: expected at least one verify round accepting >= 1 draft token, got {:?}",
        engine.speculative.as_ref().unwrap().last_accepted
    );
    let warm = engine
        .generate_with_radix_cache(&prompt, 12, 64, false)
        .unwrap();
    eprintln!(
        "{kind:?}: cached={} accepted={:?}",
        warm.cached_prefix_tokens,
        engine.speculative.as_ref().unwrap().last_accepted
    );
    assert!(warm.cached_prefix_tokens >= 64);
    assert_eq!(
        engine.speculative.as_ref().unwrap().last_restored_tokens,
        warm.cached_prefix_tokens
    );
    assert_eq!(warm.generated_token_ids, baseline, "warm {kind:?}");
    if kind == SpeculativeKind::Mtp {
        verify_mtp_branching_and_multi_memo(&mut engine, &prompt, &seed);
        for k in [1usize, 2, 3] {
            engine.speculative.as_mut().unwrap().set_mtp_k(k).unwrap();
            let run_k = engine.generate_tokens(&prompt, 8, false).unwrap();
            assert_eq!(
                run_k.generated_token_ids,
                &baseline[..8],
                "MTP k={k} greedy tokens must match baseline"
            );
            let accepted_k = &engine.speculative.as_ref().unwrap().last_accepted;
            assert!(
                !accepted_k.is_empty() && accepted_k.iter().all(|&a| (0..=k).contains(&a)),
                "MTP k={k} accepted draft counts out of range 0..={k}: {accepted_k:?}"
            );
        }
    }
    let short = engine.generate_tokens(&prompt, 2, false).unwrap();
    assert_eq!(
        short.generated_token_ids,
        &baseline[..2],
        "zero-draft final round"
    );
    assert!(engine.generate_tokens(&[-1], 2, false).is_err());
    assert!(
        engine.speculative.is_some(),
        "an error must not drop proposer configuration"
    );
    assert_eq!(
        engine
            .generate_tokens(&prompt, 2, false)
            .unwrap()
            .generated_token_ids,
        &baseline[..2]
    );
    exercise_http(engine);
}

fn verify_mtp_branching_and_multi_memo(engine: &mut Qwen3_5Engine, prompt: &[i32], seed: &[i32]) {
    let mut branch_a = prompt.to_vec();
    branch_a.resize(140, seed[0]);
    branch_a[70..140].fill(seed[1 % seed.len()]);
    let mut branch_b = branch_a.clone();
    branch_b[70..140].fill(seed[2 % seed.len()]);
    let mut branch_c = branch_a.clone();
    branch_c[70..140].fill(seed[3 % seed.len()]);

    let run_a1 = engine
        .generate_with_radix_cache(&branch_a, 8, 256, false)
        .unwrap();
    let run_b1 = engine
        .generate_with_radix_cache(&branch_b, 8, 256, false)
        .unwrap();
    let run_c1 = engine
        .generate_with_radix_cache(&branch_c, 8, 256, false)
        .unwrap();
    assert_eq!(
        run_c1.cached_prefix_tokens, 64,
        "branching single-chunk prefill must split and donate 64-token shared prefix"
    );
    assert_eq!(
        engine.speculative.as_ref().unwrap().last_restored_tokens,
        64
    );
    let run_c2 = engine
        .generate_with_radix_cache(&branch_c, 8, 256, false)
        .unwrap();
    assert_eq!(run_c2.cached_prefix_tokens, 128);
    assert_eq!(
        engine.speculative.as_ref().unwrap().last_restored_tokens,
        128
    );
    assert_eq!(run_c2.generated_token_ids, run_c1.generated_token_ids);
    let run_a2 = engine
        .generate_with_radix_cache(&branch_a, 8, 256, false)
        .unwrap();
    assert!(run_a2.cached_prefix_tokens >= 64);
    assert_eq!(
        engine.speculative.as_ref().unwrap().last_restored_tokens,
        run_a2.cached_prefix_tokens
    );
    assert_eq!(run_a2.generated_token_ids, run_a1.generated_token_ids);
    let run_b2 = engine
        .generate_with_radix_cache(&branch_b, 8, 256, false)
        .unwrap();
    assert!(run_b2.cached_prefix_tokens >= 64);
    assert_eq!(
        engine.speculative.as_ref().unwrap().last_restored_tokens,
        run_b2.cached_prefix_tokens
    );
    assert_eq!(run_b2.generated_token_ids, run_b1.generated_token_ids);
}
mod common;
fn exercise_http(engine: Qwen3_5Engine) {
    use std::sync::atomic::Ordering;
    use w34_host::server::OpenAiServerState;
    let state = OpenAiServerState::new("native-spec-test".into(), engine);
    for stream in [false, true] {
        let rounds = state.worker.stats.verify_rounds.load(Ordering::Relaxed);
        let accepted = state.worker.stats.accepted_drafts.load(Ordering::Relaxed);
        let body=serde_json::json!({"model":"native-spec-test","messages":[{"role":"user","content":"Say hello."}],
            "max_tokens":8,"temperature":if stream {0.8}else{0.0},"seed":72,"top_k":20,"stream":stream,"chat_template_kwargs":{"enable_thinking":false}}).to_string();
        let response = common::raw_request(&state, "POST", "/v1/chat/completions", Some(&body));
        assert!(response.contains("200 OK"), "{response}");
        if stream {
            assert!(
                response.contains("text/event-stream") && response.contains("data: [DONE]"),
                "{response}"
            );
            let (_, body_text) = response.split_once("\r\n\r\n").unwrap();
            let mut delta_chars = 0usize;
            for line in body_text.lines() {
                if let Some(json_str) = line.strip_prefix("data: ")
                    && json_str.trim() != "[DONE]"
                {
                    let v: serde_json::Value = serde_json::from_str(json_str).unwrap();
                    if let Some(c) = v["choices"][0]["delta"]["content"].as_str() {
                        delta_chars += c.len();
                    }
                }
            }
            assert!(
                delta_chars > 0,
                "expected non-empty streamed delta content: {response}"
            );
        } else {
            assert!(response.contains("chat.completion"), "{response}");
            let (_, body_text) = response.split_once("\r\n\r\n").unwrap();
            let v: serde_json::Value = serde_json::from_str(body_text).unwrap();
            assert_eq!(v["usage"]["completion_tokens"].as_u64(), Some(8));
            assert!(
                !v["choices"][0]["message"]["content"]
                    .as_str()
                    .unwrap_or("")
                    .is_empty(),
                "expected non-empty completion content: {response}"
            );
        }
        assert!(
            state.worker.stats.verify_rounds.load(Ordering::Relaxed) > rounds,
            "server bypassed native proposer"
        );
        let new_accepted = state.worker.stats.accepted_drafts.load(Ordering::Relaxed);
        if !stream {
            assert!(
                new_accepted > accepted,
                "expected greedy HTTP request to accept at least 1 draft token (before={accepted}, after={new_accepted})"
            );
        } else {
            assert!(new_accepted >= accepted);
        }
    }
    common::shutdown(&state);
}

#[test]
fn native_sampled_terminal_grammar() {
    let model = "nchapman/Qwen3.5-4B-W4A16-GPTQ";
    if common::require_or_skip_cached_model(model, "native_sampled_terminal_grammar").is_none() {
        return;
    }
    let _gpu_guard = GPU_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    use w34_host::parser::{SamplingParams, TokenBitmaskConstraint};
    let mut engine = Qwen3_5Engine::load(model, 1, KvFormatPolicy::Fp16, Some(16), false).unwrap();
    let prompt = engine.encode("The capital of France is").unwrap();
    engine
        .enable_speculative(&SpeculativeConfig {
            kind: SpeculativeKind::Mtp,
            draft_model: None,
            mtp_k: 3,
        })
        .unwrap();
    let terminal = (0..engine.cfg.vocab_size as i32)
        .find(|id| !engine.cfg.eos_token_ids.contains(id))
        .unwrap();
    // Call the public proposer API directly: parameters and grammar are independent.
    let mut proposer = engine.speculative.take().unwrap();
    for min_tokens in [0, 3] {
        let params = SamplingParams {
            max_tokens: 4,
            min_tokens,
            ..SamplingParams::default()
        };
        assert!(!params.is_stop_or_eos(terminal, &engine.cfg.eos_token_ids));
        let constraint =
            TokenBitmaskConstraint::from_allowed_tokens(engine.cfg.vocab_size, vec![terminal])
                .with_terminal_eos_ids(vec![terminal]);
        let result = proposer
            .generate_sampled(&mut engine, &prompt, params, Some(constraint), 64)
            .unwrap();
        assert_eq!(result.generated_token_ids, vec![terminal]);
        assert_eq!(result.gen_len, 1);
    }
    engine.speculative = Some(proposer);
}

#[test]
fn native_processed_sampling_and_grammar() {
    if common::require_or_skip_cached_model(
        "nchapman/Qwen3.5-4B-W4A16-GPTQ",
        "native_processed_sampling_and_grammar",
    )
    .is_none()
    {
        return;
    }
    let _gpu_guard = GPU_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    use w34_host::parser::{SamplingParams, TokenBitmaskConstraint};
    let mut engine = Qwen3_5Engine::load(
        "nchapman/Qwen3.5-4B-W4A16-GPTQ",
        1,
        KvFormatPolicy::Fp16,
        Some(16),
        false,
    )
    .unwrap();
    let prompt = engine.encode("The capital of France is").unwrap();
    engine
        .enable_speculative(&SpeculativeConfig {
            kind: SpeculativeKind::Mtp,
            draft_model: None,
            mtp_k: 3,
        })
        .unwrap();
    let plain = engine
        .generate_tokens(&prompt, 12, false)
        .unwrap()
        .generated_token_ids;
    let greedy = SamplingParams {
        max_tokens: 12,
        ..SamplingParams::default()
    };
    assert_eq!(
        engine
            .generate_sampled_with_radix_cache(&prompt, greedy, None, 64)
            .unwrap()
            .generated_token_ids,
        plain
    );
    let params = SamplingParams {
        temperature: 0.8,
        top_k: 20,
        top_p: 0.9,
        repetition_penalty: 1.1,
        frequency_penalty: 0.1,
        presence_penalty: 0.2,
        seed: 72,
        max_tokens: 12,
        ..SamplingParams::default()
    };
    let first = engine
        .generate_sampled_with_radix_cache(&prompt, params.clone(), None, 64)
        .unwrap();
    assert!(
        !engine
            .speculative
            .as_ref()
            .unwrap()
            .last_accepted
            .is_empty()
    );
    let second = engine
        .generate_sampled_with_radix_cache(&prompt, params.clone(), None, 64)
        .unwrap();
    assert_eq!(
        first.generated_token_ids, second.generated_token_ids,
        "seeded rejection replay"
    );
    let allowed = vec![70123, 70124, 70125];
    let constraint =
        TokenBitmaskConstraint::from_allowed_tokens(engine.cfg.vocab_size, allowed.clone());
    let constrained = engine
        .generate_sampled_with_radix_cache(&prompt, params.clone(), Some(constraint), 64)
        .unwrap();
    assert!(
        constrained
            .generated_token_ids
            .iter()
            .all(|id| allowed.contains(id))
    );
    assert_eq!(constrained.gen_len, 12);
    let invalid = SamplingParams {
        temperature: -1.,
        ..params
    };
    assert!(
        engine
            .generate_sampled_with_radix_cache(&prompt, invalid, None, 64)
            .is_err()
    );
    assert!(engine.speculative.is_some());
    assert_eq!(
        engine
            .generate_tokens(&prompt, 2, false)
            .unwrap()
            .generated_token_ids,
        &plain[..2]
    );
}
#[test]
fn native_mtp_lossless_cold_warm_and_budget() {
    lossless_cold_and_warm(SpeculativeKind::Mtp, 1, false);
}
#[test]
fn native_dflash_lossless_cold_warm_and_budget() {
    lossless_cold_and_warm(SpeculativeKind::DFlash2, 1, false);
}
#[test]
fn native_mtp_tp2_lossless() {
    lossless_cold_and_warm(SpeculativeKind::Mtp, 2, false);
}
#[test]
fn native_dflash_tp2_lossless() {
    lossless_cold_and_warm(SpeculativeKind::DFlash2, 2, false);
}
#[test]
fn native_27b_dflash_m8_tp2_lossless() {
    lossless_cold_and_warm(SpeculativeKind::DFlash2, 2, true);
}

#[test]
fn sglang_dflash2_fixtures_operator_dump_diff_parity() {
    use core::ffi::c_void;
    use w34_host::{
        cache::KvPool,
        cuda::{CudaDevice, CudaStream, DeviceBuffer, check_rc},
        loader::SafetensorsArchive,
    };
    use w34_kernels::ffi;

    let _gpu_guard = GPU_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    let _device = CudaDevice::set_current(0).unwrap();
    let stream = CudaStream::new(0).unwrap();
    let ex = stream.exec(0, 0);
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../tools/oracle/goldens/sglang_dflash2_fixtures.safetensors");
    let a = SafetensorsArchive::open(&path).unwrap();
    let upload = |name: &str| {
        let bytes = a.read_bytes(name).unwrap();
        let b = DeviceBuffer::alloc(0, bytes.len()).unwrap();
        b.copy_from_host(&bytes, &stream).unwrap();
        b
    };
    let check_bf16 = |name: &str, out: &DeviceBuffer| {
        let expected = a.read_f32(name).unwrap();
        let mut bits = vec![0u16; expected.len()];
        out.copy_to_host(&mut bits, &stream).unwrap();
        for (&b, &y) in bits.iter().zip(&expected) {
            let x = f32::from_bits((b as u32) << 16);
            assert!(x.is_finite(), "{name} non-finite");
            assert!(
                (x - y).abs() <= 0.008 * y.abs() + 0.0002,
                "{name} {x} != {y}"
            );
        }
    };
    for hidden in [128, 256, 3072, 5120] {
        let x = upload(&format!("norm{hidden}.x"));
        let res = upload(&format!("norm{hidden}.residual"));
        let w = upload(&format!("norm{hidden}.weight"));
        let out = DeviceBuffer::alloc(0, 16 * hidden * 2).unwrap();
        // SAFETY: fixture tensors hold 16 BF16 rows of width `hidden`.
        unsafe {
            check_rc(
                ffi::w34_rms_norm_bf16(
                    &ex,
                    x.addr() as *const c_void,
                    res.addr() as *mut c_void,
                    w.addr() as *const c_void,
                    out.addr() as *mut c_void,
                    16,
                    hidden as i32,
                    1e-6,
                ),
                "fixture fused norm",
            )
            .unwrap();
        }
        check_bf16(&format!("norm{hidden}.fused"), &out);
        check_bf16(&format!("norm{hidden}.res_out"), &res);
    }
    let gu = upload("silu.input");
    let silu_out = DeviceBuffer::alloc(0, 32 * 128 * 2).unwrap();
    // SAFETY: silu.input is [32, 256] BF16 and silu_out is [32, 128] BF16.
    unsafe {
        check_rc(
            ffi::w34_silu_and_mul_bf16(
                &ex,
                gu.addr() as *const c_void,
                silu_out.addr() as *mut c_void,
                32,
                128,
            ),
            "fixture silu",
        )
        .unwrap();
    }
    check_bf16("silu.output", &silu_out);
    for (name, rows, seq_len, window) in [
        ("short", 8i32, 24i32, 16i32),
        ("full", 16, 24, 0),
        ("long", 8, 2064, 2048),
        ("narrow", 16, 24, 4),
    ] {
        let pages = (seq_len as usize).div_ceil(16);
        let mut pool = KvPool::new(
            0,
            ffi::W34_KV_NAMESPACE_DFLASH_DRAFT_CTX,
            &[0],
            2,
            128,
            16,
            pages,
            KvFormatPolicy::Bf16,
        )
        .unwrap();
        let table: Vec<i32> = (0..pages).map(|_| pool.alloc_page().unwrap()).collect();
        let slots: Vec<i32> = (0..seq_len).collect();
        let q = upload(&format!("{name}.q"));
        let k = upload(&format!("{name}.k"));
        let v = upload(&format!("{name}.v"));
        let out = DeviceBuffer::alloc(0, (rows as usize) * 8 * 128 * 2).unwrap();
        let upload_i32 = |vals: &[i32]| {
            let b = DeviceBuffer::alloc(0, vals.len() * 4).unwrap();
            b.copy_from_host(vals, &stream).unwrap();
            b
        };
        let q_start = upload_i32(&[0, seq_len]);
        let ctx_zero = upload_i32(&[0]);
        let seq_after = upload_i32(&[seq_len]);
        let d_table = upload_i32(&table);
        let d_slots = upload_i32(&slots);
        let write_batch = ffi::W34AttentionBatch {
            query_start_loc: q_start.addr() as *const i32,
            context_lens_before: ctx_zero.addr() as *const i32,
            seq_lens_after: seq_after.addr() as *const i32,
            block_tables: d_table.addr() as *const i32,
            write_slot_mapping: d_slots.addr() as *const i32,
            valid_token_mask: core::ptr::null(),
            valid_req_mask: core::ptr::null(),
            num_reqs: 1,
            total_tokens: seq_len,
            max_blocks_per_req: pages as i32,
            max_q_len: seq_len,
            max_seq_len: seq_len,
            sliding_window: window,
            mask_mode: ffi::W34_ATTN_MASK_NONCAUSAL_WINDOW,
            sm_scale: 1.0 / 128.0f32.sqrt(),
            reserved: 0,
        };
        // SAFETY: write_batch and KV buffers match the fixture [seq_len, 2, 128] BF16 geometry.
        unsafe {
            check_rc(
                ffi::w34_kv_cache_write_bf16(
                    &ex,
                    &pool.layers[0].layout,
                    &write_batch,
                    k.addr() as *const c_void,
                    v.addr() as *const c_void,
                ),
                "fixture kv write",
            )
            .unwrap();
        }
        let q_start_r = upload_i32(&[0, rows]);
        let ctx_before = upload_i32(&[seq_len - rows]);
        let attn_batch = ffi::W34AttentionBatch {
            query_start_loc: q_start_r.addr() as *const i32,
            context_lens_before: ctx_before.addr() as *const i32,
            max_q_len: rows,
            total_tokens: rows,
            ..write_batch
        };
        // SAFETY: attn_batch and q/out buffers match the [rows, 8, 128] BF16 geometry.
        unsafe {
            check_rc(
                ffi::w34_paged_attention_bf16(
                    &ex,
                    &pool.layers[0].layout,
                    &attn_batch,
                    q.addr() as *const c_void,
                    out.addr() as *mut c_void,
                    8,
                ),
                "fixture paged attention",
            )
            .unwrap();
        }
        // Native attention is unsplit. FlashInfer's auto planner splits the
        // long case and rounds BF16 partial outputs before merging; compare
        // to its explicitly unsplit capture, not that different arithmetic.
        check_bf16(&format!("{name}.unsplit_out"), &out);
    }
}

#[test]
fn speculative_verify_cuda_graph_matches_eager_and_sampled() {
    if common::require_or_skip_cached_model(
        "nchapman/Qwen3.5-4B-W4A16-GPTQ",
        "speculative_verify_cuda_graph_matches_eager_and_sampled",
    )
    .is_none()
    {
        return;
    }
    use w34_host::parser::SamplingParams;
    let _gpu_guard = GPU_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    let spec = SpeculativeConfig {
        kind: SpeculativeKind::Mtp,
        draft_model: None,
        mtp_k: 3,
    };
    let mut eager = Qwen3_5Engine::load(
        "nchapman/Qwen3.5-4B-W4A16-GPTQ",
        1,
        KvFormatPolicy::Tq4,
        Some(16),
        false,
    )
    .unwrap();
    eager.enable_speculative(&spec).unwrap();
    let prompt = eager
        .encode("Explain the difference between a mutex and a semaphore in two sentences.")
        .unwrap();
    let eager_greedy = eager
        .generate_with_radix_cache(&prompt, 16, 64, false)
        .unwrap()
        .generated_token_ids;
    let params = SamplingParams {
        temperature: 0.8,
        top_k: 20,
        top_p: 0.9,
        seed: 72,
        max_tokens: 16,
        ..SamplingParams::default()
    };
    let eager_sampled = eager
        .generate_sampled_with_radix_cache(&prompt, params.clone(), None, 64)
        .unwrap()
        .generated_token_ids;
    drop(eager);

    let mut graphed = Qwen3_5Engine::load(
        "nchapman/Qwen3.5-4B-W4A16-GPTQ",
        1,
        KvFormatPolicy::Tq4,
        Some(16),
        true,
    )
    .unwrap();
    graphed.enable_speculative(&spec).unwrap();
    assert!(graphed.use_cuda_graph);
    assert!(graphed.ranks[0].spec_verify_graph.is_none());
    let graphed_greedy = graphed
        .generate_with_radix_cache(&prompt, 16, 64, false)
        .unwrap()
        .generated_token_ids;
    assert_eq!(
        graphed.ranks[0]
            .spec_verify_graph
            .as_ref()
            .map(|((m, _), _)| *m),
        Some(4)
    );
    assert_eq!(graphed_greedy, eager_greedy);

    let graphed_sampled = graphed
        .generate_sampled_with_radix_cache(&prompt, params, None, 64)
        .unwrap()
        .generated_token_ids;
    assert_eq!(graphed_sampled, eager_sampled);
}

#[test]
fn dflash_and_tp2_proposer_cuda_graphs_match_eager() {
    if common::require_or_skip_cached_model(
        "nchapman/Qwen3.5-4B-W4A16-GPTQ",
        "dflash_and_tp2_proposer_cuda_graphs_match_eager",
    )
    .is_none()
        || common::require_or_skip_cached_model(
            "nchapman/Qwen3.5-4B-DFlash2",
            "dflash_and_tp2_proposer_cuda_graphs_match_eager",
        )
        .is_none()
    {
        return;
    }
    let _gpu_guard = GPU_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    let dflash_spec = SpeculativeConfig {
        kind: SpeculativeKind::DFlash2,
        draft_model: Some("nchapman/Qwen3.5-4B-DFlash2".to_string()),
        mtp_k: 3,
    };
    let mut eager_df = Qwen3_5Engine::load(
        "nchapman/Qwen3.5-4B-W4A16-GPTQ",
        1,
        KvFormatPolicy::Tq4,
        Some(16),
        false,
    )
    .unwrap();
    eager_df.enable_speculative(&dflash_spec).unwrap();
    let prompt = eager_df
        .encode("Write a compact Rust function computing Fibonacci numbers iteratively.")
        .unwrap();
    let expected_df = eager_df
        .generate_with_radix_cache(&prompt, 32, 64, false)
        .unwrap()
        .generated_token_ids;
    drop(eager_df);

    for tp in [1usize, 2usize] {
        let mut graphed_df = Qwen3_5Engine::load(
            "nchapman/Qwen3.5-4B-W4A16-GPTQ",
            tp,
            KvFormatPolicy::Tq4,
            Some(16),
            true,
        )
        .unwrap();
        graphed_df.enable_speculative(&dflash_spec).unwrap();
        let got_df = graphed_df
            .generate_with_radix_cache(&prompt, 32, 64, false)
            .unwrap()
            .generated_token_ids;
        assert_eq!(
            got_df, expected_df,
            "DFlash2 CUDA Graph parity failed at TP={tp}"
        );
    }

    let mtp_spec = SpeculativeConfig {
        kind: SpeculativeKind::Mtp,
        draft_model: None,
        mtp_k: 3,
    };
    let mut eager_mtp2 = Qwen3_5Engine::load(
        "nchapman/Qwen3.5-4B-W4A16-GPTQ",
        2,
        KvFormatPolicy::Tq4,
        Some(16),
        false,
    )
    .unwrap();
    eager_mtp2.enable_speculative(&mtp_spec).unwrap();
    let expected_mtp2 = eager_mtp2
        .generate_with_radix_cache(&prompt, 20, 64, false)
        .unwrap()
        .generated_token_ids;
    drop(eager_mtp2);

    let mut graphed_mtp2 = Qwen3_5Engine::load(
        "nchapman/Qwen3.5-4B-W4A16-GPTQ",
        2,
        KvFormatPolicy::Tq4,
        Some(16),
        true,
    )
    .unwrap();
    graphed_mtp2.enable_speculative(&mtp_spec).unwrap();
    let got_mtp2 = graphed_mtp2
        .generate_with_radix_cache(&prompt, 20, 64, false)
        .unwrap()
        .generated_token_ids;
    assert_eq!(
        got_mtp2, expected_mtp2,
        "TP=2 MTP composite CUDA Graph parity failed"
    );
}
