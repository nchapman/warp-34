//! Phase 6.6a GPU-resident stochastic `SamplerV1` + device grammar bitmasking gates.
#![cfg(feature = "cuda")]

mod common;

use std::collections::BTreeMap;
use std::ffi::c_void;
use std::sync::Mutex;

use serde_json::json;
use w34_host::cache::KvFormatPolicy;
use w34_host::constraints::ConstraintFactory;
use w34_host::cuda::{CudaDevice, CudaStream, DeviceArena, f32_to_f16, prepare_execution_plan};
use w34_host::model::Qwen3_5Engine;
use w34_host::parser::{SamplerV1, SamplingParams, TokenBitmaskConstraint};
use w34_host::proposer::{SpeculativeConfig, SpeculativeKind};
use w34_host::speculative::RejectionSampler;
use w34_kernels::ffi::{self, W34SamplerRowCfg, W34SamplerRowOut, W34SamplerSparseEntry};

static GPU_LOCK: Mutex<()> = Mutex::new(());

fn run_single_gpu_row(
    vocab_size: usize,
    logits_f16: &[u16],
    cfg: W34SamplerRowCfg,
    bitmask_words: &[u32],
    sparse_entries: &[W34SamplerSparseEntry],
) -> W34SamplerRowOut {
    prepare_execution_plan(0).unwrap();
    let _dev = CudaDevice::set_current(0).unwrap();
    let stream = CudaStream::new(0).unwrap();
    let mut arena = DeviceArena::new(0, 32 * 1024 * 1024).unwrap();
    let logits_dev = arena.upload_slice(logits_f16, &stream).unwrap();
    let cfg_dev = arena.upload_slice(&[cfg], &stream).unwrap();
    let mask_dev = if bitmask_words.is_empty() {
        0
    } else {
        arena.upload_slice(bitmask_words, &stream).unwrap()
    };
    let sparse_dev = if sparse_entries.is_empty() {
        0
    } else {
        arena.upload_slice(sparse_entries, &stream).unwrap()
    };
    let out_dev = arena
        .alloc_aligned(core::mem::size_of::<W34SamplerRowOut>())
        .unwrap();
    let (ws_ptr, ws_cap) = arena.remaining_workspace();
    let ex = stream.exec(ws_ptr, ws_cap);
    // SAFETY: all device pointers were allocated on device 0 inside `arena` for this stream.
    let rc = unsafe {
        ffi::w34_sampler_step_f16(
            &ex,
            logits_dev as *const c_void,
            cfg_dev as *const W34SamplerRowCfg,
            if mask_dev == 0 {
                core::ptr::null()
            } else {
                mask_dev as *const u32
            },
            if sparse_dev == 0 {
                core::ptr::null()
            } else {
                sparse_dev as *const W34SamplerSparseEntry
            },
            out_dev as *mut W34SamplerRowOut,
            1,
            vocab_size as i32,
            sparse_entries.len() as i32,
        )
    };
    assert_eq!(rc, 0, "w34_sampler_step_f16 rc={rc}");
    let mut out = W34SamplerRowOut::default();
    // SAFETY: `out` is live stack storage and `out_dev` holds `size_of::<W34SamplerRowOut>()` bytes.
    let copy_rc = unsafe {
        ffi::w34_memcpy_d2h_async(
            (&raw mut out).cast(),
            out_dev as *const c_void,
            core::mem::size_of::<W34SamplerRowOut>(),
            stream.handle,
        )
    };
    assert_eq!(copy_rc, 0);
    stream.synchronize().unwrap();
    out
}

#[test]
fn gpu_sampler_matches_cpu_reference_across_full_248k_vocab_and_rejection() {
    let _guard = GPU_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    let vocab_size = 248_320usize;
    let mut logits_f16 = Vec::with_capacity(vocab_size);
    let mut logits_f32 = Vec::with_capacity(vocab_size);
    for tid in 0..vocab_size {
        // Deterministic pseudo-random FP16 logits with occasional exact ties (including +-0.0).
        let raw = if tid % 997 == 0 {
            if tid % 1994 == 0 { 0.0f32 } else { -0.0f32 }
        } else {
            let h = (tid as u64)
                .wrapping_mul(0x9E37_79B9_7F4A_7C15)
                .rotate_left(19);
            (((h & 0xffff) as f32) / 65535.0) * 14.0 - 9.0
        };
        let bits = f32_to_f16(raw);
        logits_f16.push(bits);
        logits_f32.push(w34_host::cuda::f16_to_f32(bits));
    }

    let allowed: Vec<i32> = (0..vocab_size as i32)
        .filter(|&t| t % 7 == 3 || t == 151645)
        .collect();
    let constraint = TokenBitmaskConstraint::from_allowed_tokens(vocab_size, allowed);

    let mut logit_bias = BTreeMap::new();
    logit_bias.insert(7003, 2.5);
    logit_bias.insert(14003, -3.0);

    let prompt = vec![7003, 21003, 28003];
    for seed in 1u64..=16 {
        let params = SamplingParams {
            temperature: 0.77,
            top_k: 64,
            top_p: 0.88,
            min_p: 0.02,
            repetition_penalty: 1.15,
            frequency_penalty: 0.25,
            presence_penalty: 0.35,
            seed,
            logit_bias: logit_bias.clone(),
            min_tokens: 2,
            ignore_eos: false,
            logprobs: true,
            top_logprobs: 5,
            stop_token_ids: vec![151645],
            eos_token_ids: vec![151645],
            ..Default::default()
        };
        let mut cpu_sampler = SamplerV1::new(params, &prompt, Some(constraint.clone()));
        cpu_sampler.commit_token(35003);

        let mut mask_words = Vec::new();
        let mut sparse = Vec::new();
        let u = SamplerV1::uniform_f32(seed, cpu_sampler.output_history.len());
        let cfg = cpu_sampler
            .append_gpu_row(
                vocab_size,
                &[151645],
                u,
                0.0,
                -1,
                false,
                &mut mask_words,
                &mut sparse,
            )
            .unwrap();
        let gpu_out = run_single_gpu_row(vocab_size, &logits_f16, cfg, &mask_words, &sparse);
        assert_eq!(gpu_out.status, 0);
        let cpu_pick = cpu_sampler.try_sample_from_logits(&logits_f32).unwrap();
        assert_eq!(
            gpu_out.sampled_id, cpu_pick,
            "seed={seed}: GPU sampled_id != CPU SamplerV1"
        );
        assert_ne!(gpu_out.sampled_id, 151645, "min_tokens must suppress EOS");

        // Verify top-5 logprobs and sampled_logprob against exact f64 log_softmax(raw + bias).
        let mut biased_max = f64::NEG_INFINITY;
        for (t, &l) in logits_f32.iter().enumerate() {
            let v = f64::from(l + logit_bias.get(&(t as i32)).copied().unwrap_or(0.0));
            if v > biased_max {
                biased_max = v;
            }
        }
        let sum_exp: f64 = logits_f32
            .iter()
            .enumerate()
            .map(|(t, &l)| {
                let v = f64::from(l + logit_bias.get(&(t as i32)).copied().unwrap_or(0.0));
                (v - biased_max).exp()
            })
            .sum();
        let log_z = biased_max + sum_exp.ln();
        let expected_sampled_lp = (f64::from(
            logits_f32[cpu_pick as usize] + logit_bias.get(&cpu_pick).copied().unwrap_or(0.0),
        ) - log_z) as f32;
        assert!(
            (gpu_out.sampled_logprob - expected_sampled_lp).abs() < 1e-4,
            "sampled_logprob {} vs {}",
            gpu_out.sampled_logprob,
            expected_sampled_lp
        );
        assert!(gpu_out.top_logprobs[0] >= gpu_out.top_logprobs[1]);

        // Also verify point-mass speculative rejection (both wrong draft & high-prob draft).
        let base_sampler = SamplerV1::new(
            cpu_sampler.params.clone(),
            &prompt,
            Some(constraint.clone()),
        );
        let dist = base_sampler.processed_distribution(&logits_f32).unwrap();
        let best_allowed = dist
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .unwrap()
            .0 as i32;
        for draft_id in [best_allowed, 3] {
            let mut q = vec![0.0f32; vocab_size];
            q[draft_id as usize] = 1.0;
            let u_acc = SamplerV1::uniform_f32(seed ^ 0xa0761d6478bd642f, 0);
            let u_rec = SamplerV1::uniform_f32(seed ^ 0xe7037ed1a0b428db, 0);
            let cpu_ver = RejectionSampler::stochastic(
                &[draft_id],
                &[dist.clone(), dist.clone()],
                &[q],
                &[f64::from(u_acc)],
                f64::from(u_rec),
            )
            .unwrap();

            let mut m2 = Vec::new();
            let mut s2 = Vec::new();
            let row0_cfg = base_sampler
                .append_gpu_row(
                    vocab_size,
                    &[151645],
                    u_acc,
                    u_rec,
                    draft_id,
                    false,
                    &mut m2,
                    &mut s2,
                )
                .unwrap();
            let gpu_ver0 = run_single_gpu_row(vocab_size, &logits_f16, row0_cfg, &m2, &s2);
            assert_eq!(gpu_ver0.status, 0);
            assert_eq!(
                gpu_ver0.draft_accepted == 1,
                cpu_ver.accepted_drafts() == 1,
                "seed={seed} draft={draft_id} acceptance mismatch"
            );
            if cpu_ver.accepted_drafts() == 0 {
                assert_eq!(
                    gpu_ver0.sampled_id,
                    cpu_ver.emitted_tokens()[0],
                    "seed={seed} draft={draft_id} residual recovery mismatch"
                );
            } else {
                let mut m3 = Vec::new();
                let mut s3 = Vec::new();
                let bonus_cfg = base_sampler
                    .append_gpu_row(
                        vocab_size,
                        &[151645],
                        0.0,
                        u_rec,
                        -1,
                        true,
                        &mut m3,
                        &mut s3,
                    )
                    .unwrap();
                let gpu_bonus = run_single_gpu_row(vocab_size, &logits_f16, bonus_cfg, &m3, &s3);
                assert_eq!(gpu_bonus.sampled_id, cpu_ver.emitted_tokens()[1]);
            }
        }
    }

    // Error paths: empty grammar support (status=1) and nonfinite allowed logit (status=2).
    let empty_c = TokenBitmaskConstraint::from_allowed_tokens(vocab_size, vec![]);
    let empty_s = SamplerV1::new(SamplingParams::default(), &[], Some(empty_c));
    let mut mw = Vec::new();
    let mut sp = Vec::new();
    let cfg_empty = empty_s
        .append_gpu_row(vocab_size, &[], 0.5, 0.5, -1, false, &mut mw, &mut sp)
        .unwrap();
    assert_eq!(
        run_single_gpu_row(vocab_size, &logits_f16, cfg_empty, &mw, &sp).status,
        1
    );

    let mut nan_logits = logits_f16.clone();
    nan_logits[10] = 0x7e00u16; // FP16 NaN on an allowed token
    let plain_s = SamplerV1::new(SamplingParams::default(), &[], None);
    let mut mw2 = Vec::new();
    let mut sp2 = Vec::new();
    let cfg_nan = plain_s
        .append_gpu_row(vocab_size, &[], 0.5, 0.5, -1, false, &mut mw2, &mut sp2)
        .unwrap();
    assert_eq!(
        run_single_gpu_row(vocab_size, &nan_logits, cfg_nan, &mw2, &sp2).status,
        2
    );
}

#[test]
fn tp1_and_tp2_gpu_sampler_and_bitmask_parity_plain_and_speculative() {
    let _guard = GPU_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    if !common::require_or_skip_gpus(
        2,
        "tp1_and_tp2_gpu_sampler_and_bitmask_parity_plain_and_speculative",
    ) {
        return;
    }
    let model = "nchapman/Qwen3.5-4B-W4A16-GPTQ";
    let schema = json!({
        "type": "object",
        "properties": {
            "answer": { "type": "string", "enum": ["alpha", "beta"] },
            "count": { "type": "integer", "minimum": 1, "maximum": 9 }
        },
        "required": ["answer", "count"],
        "additionalProperties": false
    });

    let run_tp = |tp_size: usize,
                  with_mtp: bool,
                  temperature: f32|
     -> (Vec<i32>, serde_json::Value) {
        let mut engine =
            Qwen3_5Engine::load(model, tp_size, KvFormatPolicy::Tq4, Some(32), true).unwrap();
        if with_mtp {
            engine
                .enable_speculative(&SpeculativeConfig {
                    kind: SpeculativeKind::Mtp,
                    draft_model: None,
                    mtp_k: 3,
                })
                .unwrap();
        }
        let eos_u32: Vec<u32> = engine
            .cfg
            .eos_token_ids
            .iter()
            .filter_map(|&id| u32::try_from(id).ok())
            .collect();
        let factory = ConstraintFactory::new(
            engine.tokenizer.as_ref().unwrap().clone(),
            engine.cfg.vocab_size,
            &eos_u32,
        )
        .unwrap();
        let constraint = factory.json_schema(schema.clone(), false).unwrap();
        let prompt = engine
            .encode("<|im_start|>user\nReturn JSON.<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n")
            .unwrap();
        let mut logit_bias = BTreeMap::new();
        logit_bias.insert(220, 0.25);
        let params = SamplingParams {
            temperature,
            top_k: if temperature == 0.0 { 1 } else { 32 },
            top_p: if temperature == 0.0 { 1.0 } else { 0.9 },
            min_p: if temperature == 0.0 { 0.0 } else { 0.02 },
            repetition_penalty: 1.05,
            seed: 12345,
            max_tokens: 24,
            logit_bias,
            min_tokens: 4,
            logprobs: true,
            top_logprobs: 3,
            eos_token_ids: engine.cfg.eos_token_ids.clone(),
            stop_token_ids: engine.cfg.eos_token_ids.clone(),
            ..Default::default()
        };
        let res = engine
            .generate_sampled_with_radix_cache(&prompt, params, Some(constraint), 64)
            .unwrap();
        let text = engine.decode(&res.generated_token_ids).unwrap();
        let parsed: serde_json::Value = serde_json::from_str(&text).unwrap();
        assert!(parsed.get("answer").is_some() && parsed.get("count").is_some());
        assert!(res.generated_token_ids.len() >= 4);
        assert_eq!(res.token_logprobs.len(), res.generated_token_ids.len());
        for step_lp in &res.token_logprobs {
            assert!(step_lp.logprob.is_finite());
            assert_eq!(step_lp.top_logprobs.len(), 3);
            assert!(step_lp.top_logprobs[0].logprob >= step_lp.top_logprobs[1].logprob);
            assert!(step_lp.top_logprobs[1].logprob >= step_lp.top_logprobs[2].logprob);
        }
        (res.generated_token_ids, parsed)
    };

    // Stochastic (`temperature = 0.7`) GPU sampling + JSON-schema bitmasking across TP=1/TP=2
    // plain and MTP: under point-mass speculative rejection, ~1e-4 FP16 TP=2 row-parallel
    // reduction differences can accept a BPE merge (`"al"` + `"pha"`) on one TP degree while
    // residual-recovering its split BPE prefix (`"a"` + `"l"` + `"pha"`) on the other, so
    // stochastic assertions check schema validity and per-step logprob invariants rather than
    // identical BPE token counts.
    let (tp1_plain, _) = run_tp(1, false, 0.7);
    let (tp2_plain, _) = run_tp(2, false, 0.7);
    assert!(!tp1_plain.is_empty() && !tp2_plain.is_empty());

    let (tp1_mtp, _) = run_tp(1, true, 0.7);
    let (tp2_mtp, _) = run_tp(2, true, 0.7);
    assert!(!tp1_mtp.is_empty() && !tp2_mtp.is_empty());

    // Deterministic (`temperature = 0.0`, still routed through full `w34_sampler_step_f16` via
    // JSON-schema bitmask + `logit_bias` + `repetition_penalty` + `top_logprobs = 3`) must
    // converge to the exact same JSON value across TP=1/TP=2 and plain/MTP (cuBLAS `lm_head`
    // `M=1` GEMV vs `M=4` MMA can break a 1-ULP `"\\n"` vs `"\\n "` whitespace tie).
    let (g_tp1_plain, g_json1) = run_tp(1, false, 0.0);
    let (g_tp2_plain, g_json2) = run_tp(2, false, 0.0);
    let (g_tp1_mtp, g_json1_mtp) = run_tp(1, true, 0.0);
    let (g_tp2_mtp, g_json2_mtp) = run_tp(2, true, 0.0);
    assert!(!g_tp1_plain.is_empty() && !g_tp2_plain.is_empty());
    assert!(!g_tp1_mtp.is_empty() && !g_tp2_mtp.is_empty());
    assert_eq!(g_json1, g_json1_mtp);
    assert_eq!(g_json2, g_json2_mtp);
    assert_eq!(g_json1, g_json2);
}

fn assert_sample_matches(
    actual: &W34SamplerRowOut,
    expected: &W34SamplerRowOut,
    row: usize,
    path: &str,
) {
    assert_eq!(
        actual.sampled_id, expected.sampled_id,
        "row {row} {path} id"
    );
    assert!(
        (actual.sampled_logprob - expected.sampled_logprob).abs() < 1e-4,
        "row {row} {path} logprob={} vs ref={}",
        actual.sampled_logprob,
        expected.sampled_logprob,
    );
}

fn copy_sampler_rows_d2h(
    stream: &CudaStream,
    dev_addr: u64,
    num_rows: usize,
) -> Vec<W34SamplerRowOut> {
    let out_bytes = num_rows * core::mem::size_of::<W34SamplerRowOut>();
    let mut host = vec![W34SamplerRowOut::default(); num_rows];
    // SAFETY: `host` and `dev_addr` are sized for `out_bytes` and synchronized before returning.
    unsafe {
        assert_eq!(
            ffi::w34_memcpy_d2h_async(
                host.as_mut_ptr() as *mut c_void,
                dev_addr as *const c_void,
                out_bytes,
                stream.handle,
            ),
            0
        );
    }
    stream.synchronize().unwrap();
    host
}

fn build_fast_greedy_test_fixture(
    num_rows: usize,
    vocab_size: usize,
) -> (Vec<u16>, Vec<W34SamplerRowCfg>, Vec<W34SamplerSparseEntry>) {
    let mut logits_f16 = vec![0u16; num_rows * vocab_size];
    for r in 0..num_rows {
        for v in 0..vocab_size {
            let val = (((r * 131 + v * 17) % 1024) as f32 - 512.0) * 0.015625;
            logits_f16[r * vocab_size + v] = f32_to_f16(val);
        }
        // Plant distinct peaks (including one penalized peak on even rows so sparse penalties flip argmax).
        let primary = 10_000 + r * 7777;
        let secondary = 50_000 + r * 3333;
        logits_f16[r * vocab_size + primary] = f32_to_f16(11.5);
        logits_f16[r * vocab_size + secondary] = f32_to_f16(11.25);
    }

    let mut row_cfgs = Vec::with_capacity(num_rows);
    let mut sparse = Vec::new();
    for r in 0..num_rows {
        let primary = (10_000 + r * 7777) as i32;
        let tertiary = (90_000 + r * 1111) as i32;
        let off = sparse.len() as i32;
        let cnt = if r % 2 == 0 {
            sparse.push(W34SamplerSparseEntry {
                token_id: primary,
                output_count: 3,
                logit_bias: -1.5,
                in_prompt: 1,
                banned: 0,
            });
            sparse.push(W34SamplerSparseEntry {
                token_id: tertiary,
                output_count: 0,
                logit_bias: 0.5,
                in_prompt: 0,
                banned: if r == 4 { 1 } else { 0 },
            });
            2
        } else {
            0
        };
        row_cfgs.push(W34SamplerRowCfg {
            temperature: 0.0,
            top_k: if r % 3 == 0 { -1 } else { 1 },
            top_p: 1.0,
            min_p: 0.0,
            repetition_penalty: if r % 2 == 0 { 1.2 } else { 1.0 },
            frequency_penalty: if r % 2 == 0 { 0.25 } else { 0.0 },
            presence_penalty: if r % 2 == 0 { 0.35 } else { 0.0 },
            uniform_sample: 0.5,
            recovery_uniform: 0.5,
            draft_token_id: -1,
            has_bitmask: 0,
            row_bitmask_word_offset: 0,
            sparse_offset: off,
            sparse_count: cnt,
            top_logprobs_k: 0,
            use_token_id_cdf: 0,
        });
    }
    (logits_f16, row_cfgs, sparse)
}

#[test]
fn fast_greedy_sampler_matches_full_sampler_and_rejects_ineligible_rows() {
    let _guard = GPU_LOCK.lock().unwrap();
    let dev = CudaDevice::set_current(0).unwrap();
    let stream = CudaStream::new(dev.device_id).unwrap();

    // Vocab size matches Qwen3.5 (`248320`, >= 1024 fast-greedy threshold).
    let vocab_size = 248_320usize;
    let num_rows = 8usize;
    let (logits_f16, row_cfgs, sparse) = build_fast_greedy_test_fixture(num_rows, vocab_size);

    let mut arena = DeviceArena::new(dev.device_id, 256 * 1024 * 1024).unwrap();
    let d_logits = arena.upload_slice(&logits_f16, &stream).unwrap();
    let d_cfgs = arena.upload_slice(&row_cfgs, &stream).unwrap();
    let d_sparse = arena.upload_slice(&sparse, &stream).unwrap();
    let out_bytes = num_rows * core::mem::size_of::<W34SamplerRowOut>();
    let d_out_fast = arena.alloc_aligned(out_bytes).unwrap();
    let d_out_auto = arena.alloc_aligned(out_bytes).unwrap();
    let (ws_ptr, ws_cap) = arena.remaining_workspace();
    // Reserve room for follow-up allocations (`d_nosparse_cfgs`, `d_full_cfgs`, etc.) by allocating an explicit workspace slice.
    let _ = (ws_ptr, ws_cap);
    let ws_full = (num_rows * vocab_size * 4 * 4 + 4096) & !255;
    let d_ws = arena.alloc_aligned(ws_full).unwrap();
    let ex = stream.exec(d_ws, ws_full);

    // Direct fast-greedy entrypoint (supports sparse penalties/biases/bans).
    // SAFETY: all device pointers belong to `arena` on device 0 and remain live past `stream.synchronize()`.
    let rc_fast = unsafe {
        ffi::w34_sampler_step_fast_greedy_f16(
            &ex,
            d_logits as *const c_void,
            d_cfgs as *const W34SamplerRowCfg,
            d_sparse as *const W34SamplerSparseEntry,
            d_out_fast as *mut W34SamplerRowOut,
            num_rows as i32,
            vocab_size as i32,
        )
    };
    assert_eq!(rc_fast, 0);
    // Automatic `w34_sampler_step_f16` fast-path dispatch when `total_sparse_entries == 0`.
    let mut nosparse_cfgs = row_cfgs.clone();
    for c in &mut nosparse_cfgs {
        c.sparse_offset = 0;
        c.sparse_count = 0;
        c.repetition_penalty = 1.0;
        c.frequency_penalty = 0.0;
        c.presence_penalty = 0.0;
    }
    let d_nosparse_cfgs = arena.upload_slice(&nosparse_cfgs, &stream).unwrap();
    // SAFETY: device buffers in `arena` hold `num_rows * vocab_size` halves and `num_rows` row configs/outputs.
    let rc_auto = unsafe {
        ffi::w34_sampler_step_f16(
            &ex,
            d_logits as *const c_void,
            d_nosparse_cfgs as *const W34SamplerRowCfg,
            core::ptr::null(),
            core::ptr::null(),
            d_out_auto as *mut W34SamplerRowOut,
            num_rows as i32,
            vocab_size as i32,
            0,
        )
    };
    assert_eq!(rc_auto, 0);

    // Reference full 9-step path: set `top_logprobs_k = 1` so `w34_sampler_step_f16` forces the full path
    // without changing greedy `sampled_id` or `sampled_logprob`.
    let mut full_cfgs = row_cfgs.clone();
    for c in &mut full_cfgs {
        c.top_logprobs_k = 1;
    }
    let d_full_cfgs = arena.upload_slice(&full_cfgs, &stream).unwrap();
    let d_out_ref = arena.alloc_aligned(out_bytes).unwrap();
    // SAFETY: `d_full_cfgs`, `d_sparse`, and `d_out_ref` are valid device allocations on device 0.
    let rc_ref = unsafe {
        ffi::w34_sampler_step_f16(
            &ex,
            d_logits as *const c_void,
            d_full_cfgs as *const W34SamplerRowCfg,
            core::ptr::null(),
            d_sparse as *const W34SamplerSparseEntry,
            d_out_ref as *mut W34SamplerRowOut,
            num_rows as i32,
            vocab_size as i32,
            sparse.len() as i32,
        )
    };
    assert_eq!(rc_ref, 0);
    let mut full_nosparse_cfgs = nosparse_cfgs.clone();
    for c in &mut full_nosparse_cfgs {
        c.top_logprobs_k = 1;
    }
    let d_full_nosparse_cfgs = arena.upload_slice(&full_nosparse_cfgs, &stream).unwrap();
    let d_out_nosparse_ref = arena.alloc_aligned(out_bytes).unwrap();
    // SAFETY: `d_full_nosparse_cfgs` and `d_out_nosparse_ref` are valid device allocations on device 0.
    let rc_nosparse_ref = unsafe {
        ffi::w34_sampler_step_f16(
            &ex,
            d_logits as *const c_void,
            d_full_nosparse_cfgs as *const W34SamplerRowCfg,
            core::ptr::null(),
            core::ptr::null(),
            d_out_nosparse_ref as *mut W34SamplerRowOut,
            num_rows as i32,
            vocab_size as i32,
            0,
        )
    };
    assert_eq!(rc_nosparse_ref, 0);

    let got_fast = copy_sampler_rows_d2h(&stream, d_out_fast, num_rows);
    let got_auto = copy_sampler_rows_d2h(&stream, d_out_auto, num_rows);
    let got_ref = copy_sampler_rows_d2h(&stream, d_out_ref, num_rows);
    let got_nosparse_ref = copy_sampler_rows_d2h(&stream, d_out_nosparse_ref, num_rows);

    for r in 0..num_rows {
        assert_sample_matches(&got_fast[r], &got_ref[r], r, "fast");
        assert_sample_matches(&got_auto[r], &got_nosparse_ref[r], r, "auto");
    }

    // Direct `w34_sampler_step_fast_greedy_f16` rejects `vocab_size < 1024` (`cudaErrorInvalidValue == 1`),
    // while `w34_sampler_step_f16` seamlessly falls back per-row when a batch mixes greedy and stochastic rows.
    // SAFETY: `vocab_size < 1024` returns immediately before launching any kernel.
    let rc_small = unsafe {
        ffi::w34_sampler_step_fast_greedy_f16(
            &ex,
            d_logits as *const c_void,
            d_cfgs as *const W34SamplerRowCfg,
            d_sparse as *const W34SamplerSparseEntry,
            d_out_fast as *mut W34SamplerRowOut,
            num_rows as i32,
            512,
        )
    };
    assert_eq!(rc_small, 1);
    let mut mixed_cfgs = nosparse_cfgs;
    // Row 0 is stochastic (`work_prob[0..vocab_size]` written by `sampler_step_f16_kernel`),
    // while rows 1..num_rows take the fast-greedy split path (`fast_parts` must not alias `work_prob`).
    mixed_cfgs[0].temperature = 0.7;
    let d_mixed = arena.upload_slice(&mixed_cfgs, &stream).unwrap();
    let mut got_mixed = vec![W34SamplerRowOut::default(); num_rows];
    // SAFETY: `d_mixed` and `d_out_auto` are valid device allocations on device 0.
    let rc_mixed = unsafe {
        ffi::w34_sampler_step_f16(
            &ex,
            d_logits as *const c_void,
            d_mixed as *const W34SamplerRowCfg,
            core::ptr::null(),
            core::ptr::null(),
            d_out_auto as *mut W34SamplerRowOut,
            num_rows as i32,
            vocab_size as i32,
            0,
        )
    };
    assert_eq!(rc_mixed, 0);
    // SAFETY: `got_mixed` and `d_out_auto` have length `out_bytes` on `stream`.
    unsafe {
        assert_eq!(
            ffi::w34_memcpy_d2h_async(
                got_mixed.as_mut_ptr() as *mut c_void,
                d_out_auto as *const c_void,
                out_bytes,
                stream.handle,
            ),
            0
        );
    }
    stream.synchronize().unwrap();
    assert_eq!(got_mixed[0].status, 0);
    assert!((0..vocab_size as i32).contains(&got_mixed[0].sampled_id));
    for r in 1..num_rows {
        assert_eq!(got_mixed[r].status, 0, "mixed row {r} status");
        assert_eq!(
            got_mixed[r].sampled_id, got_nosparse_ref[r].sampled_id,
            "mixed row {r} greedy id corrupted by stochastic row 0"
        );
        assert!(
            (got_mixed[r].sampled_logprob - got_nosparse_ref[r].sampled_logprob).abs() < 1e-4,
            "mixed row {r} logprob={} vs ref={}",
            got_mixed[r].sampled_logprob,
            got_nosparse_ref[r].sampled_logprob
        );
    }

    // Tight `2 * slice_bytes` scratch without room for `fast_parts` still succeeds via full-kernel fallback.
    let slice_bytes = ((num_rows * vocab_size * 4) + 255) & !255;
    let ex_tight = stream.exec(d_ws, slice_bytes * 2);
    let mut got_tight = vec![W34SamplerRowOut::default(); num_rows];
    // SAFETY: `ex_tight` provides exact `2 * slice_bytes` workspace required by `w34_sampler_step_f16`.
    unsafe {
        assert_eq!(
            ffi::w34_sampler_step_f16(
                &ex_tight,
                d_logits as *const c_void,
                d_mixed as *const W34SamplerRowCfg,
                core::ptr::null(),
                core::ptr::null(),
                d_out_auto as *mut W34SamplerRowOut,
                num_rows as i32,
                vocab_size as i32,
                0,
            ),
            0
        );
        assert_eq!(
            ffi::w34_memcpy_d2h_async(
                got_tight.as_mut_ptr() as *mut c_void,
                d_out_auto as *const c_void,
                out_bytes,
                stream.handle,
            ),
            0
        );
    }
    stream.synchronize().unwrap();
    for r in 1..num_rows {
        assert_eq!(got_tight[r].sampled_id, got_nosparse_ref[r].sampled_id);
    }
}

#[test]
fn test_fast_greedy_masked_matches_full_sampler_and_edges() {
    if !common::require_or_skip_gpus(1, "test_fast_greedy_masked_matches_full_sampler_and_edges") {
        return;
    }
    let _gpu_guard = GPU_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    let dev = CudaDevice::set_current(0).expect("CudaDevice(0)");
    let stream = CudaStream::new(dev.device_id).unwrap();
    // Use an odd non-multiple-of-8 vocab size above 248,320 so both vectorized `uint4` and tail
    // bitmask paths are exercised across all 16 splits.
    let vocab_size = 248_323usize;
    let words_per_row = vocab_size.div_ceil(32);
    let num_rows = 6usize;

    let mut logits_f16 = vec![0u16; num_rows * vocab_size];
    for r in 0..num_rows {
        for v in 0..vocab_size {
            let x = (((v * 1103515245 + r * 12345 + 7) & 0x7ff) as f32 - 1024.0) * 0.003;
            logits_f16[r * vocab_size + v] = f32_to_f16(x);
        }
    }
    let mut bitmasks = vec![u32::MAX; num_rows * words_per_row];
    let clear_bit = |masks: &mut [u32], row: usize, tok: usize| {
        masks[row * words_per_row + (tok >> 5)] &= !(1u32 << (tok & 31));
    };

    // Row 0: mask out global peaks + tail peak + a masked-out NaN; winner must be `allowed_win`.
    let banned_peaks_r0 = [17usize, 62_080, 199_999, vocab_size - 1];
    for &t in &banned_peaks_r0 {
        logits_f16[t] = f32_to_f16(14.0);
        clear_bit(&mut bitmasks, 0, t);
    }
    // Masked-out NaN must be ignored (matching full reference Pass 0).
    logits_f16[12_345] = 0x7e00;
    clear_bit(&mut bitmasks, 0, 12_345);
    let allowed_win_r0 = vocab_size - 2;
    logits_f16[allowed_win_r0] = f32_to_f16(11.75);

    // Row 1: unmasked row with sparse penalty mixed in same `fast_greedy_masked` launch.
    logits_f16[vocab_size + 500] = f32_to_f16(12.0);
    logits_f16[vocab_size + 501] = f32_to_f16(11.5);
    let sparse = [W34SamplerSparseEntry {
        token_id: 500,
        output_count: 1,
        logit_bias: -2.0,
        in_prompt: 0,
        banned: 0,
    }];

    // Row 2: masked row with matching draft token.
    logits_f16[2 * vocab_size + 88_888] = f32_to_f16(13.0);
    clear_bit(&mut bitmasks, 2, 88_888);
    logits_f16[2 * vocab_size + 88_889] = f32_to_f16(10.5);

    // Row 3: masked row with mismatching draft token.
    logits_f16[3 * vocab_size + 140_000] = f32_to_f16(13.0);
    clear_bit(&mut bitmasks, 3, 140_000);
    logits_f16[3 * vocab_size + 140_001] = f32_to_f16(10.25);

    // Row 4: all-zero bitmask -> status 1.
    for w in 0..words_per_row {
        bitmasks[4 * words_per_row + w] = 0;
    }

    // Row 5: allowed +Inf logit -> status 2.
    logits_f16[5 * vocab_size + 77_777] = 0x7c00;

    let base_cfg = W34SamplerRowCfg {
        temperature: 0.0,
        top_k: 1,
        top_p: 1.0,
        min_p: 0.0,
        repetition_penalty: 1.0,
        frequency_penalty: 0.0,
        presence_penalty: 0.0,
        uniform_sample: 0.5,
        recovery_uniform: 0.5,
        draft_token_id: -1,
        has_bitmask: 1,
        row_bitmask_word_offset: 0,
        sparse_offset: 0,
        sparse_count: 0,
        top_logprobs_k: -1,
        use_token_id_cdf: 0,
    };
    let cfgs = vec![
        W34SamplerRowCfg {
            row_bitmask_word_offset: 0,
            ..base_cfg
        },
        W34SamplerRowCfg {
            has_bitmask: 0,
            row_bitmask_word_offset: -1,
            sparse_offset: 0,
            sparse_count: 1,
            top_logprobs_k: 0,
            ..base_cfg
        },
        W34SamplerRowCfg {
            row_bitmask_word_offset: (2 * words_per_row) as i32,
            draft_token_id: 88_889,
            ..base_cfg
        },
        W34SamplerRowCfg {
            row_bitmask_word_offset: (3 * words_per_row) as i32,
            draft_token_id: 140_000,
            ..base_cfg
        },
        W34SamplerRowCfg {
            row_bitmask_word_offset: (4 * words_per_row) as i32,
            ..base_cfg
        },
        W34SamplerRowCfg {
            row_bitmask_word_offset: (5 * words_per_row) as i32,
            ..base_cfg
        },
    ];

    let mut arena = DeviceArena::new(dev.device_id, 128 * 1024 * 1024).unwrap();
    let d_logits = arena.upload_slice(&logits_f16, &stream).unwrap();
    let d_masks = arena.upload_slice(&bitmasks, &stream).unwrap();
    let d_sparse = arena.upload_slice(&sparse, &stream).unwrap();
    let d_cfgs = arena.upload_slice(&cfgs, &stream).unwrap();
    let out_bytes = num_rows * core::mem::size_of::<W34SamplerRowOut>();
    let d_out_fast = arena.alloc_aligned(out_bytes).unwrap();
    let d_out_ref = arena.alloc_aligned(out_bytes).unwrap();
    let ws_full = (num_rows * vocab_size * 4 * 4 + 4096) & !255;
    let d_ws = arena.alloc_aligned(ws_full).unwrap();
    let ex = stream.exec(d_ws, ws_full);

    // SAFETY: all pointers belong to `arena` on device 0 and remain valid past `synchronize()`.
    let rc_fast = unsafe {
        ffi::w34_sampler_step_fast_greedy_masked_f16(
            &ex,
            d_logits as *const c_void,
            d_cfgs as *const W34SamplerRowCfg,
            d_masks as *const u32,
            d_sparse as *const W34SamplerSparseEntry,
            d_out_fast as *mut W34SamplerRowOut,
            num_rows as i32,
            vocab_size as i32,
        )
    };
    assert_eq!(rc_fast, 0);

    // Force full 1-CTA reference path via `total_sparse_entries == 1` (`fast_parts == null`).
    let slice_bytes = ((num_rows * vocab_size * 4) + 255) & !255;
    let ex_ref_sparse = stream.exec(d_ws, slice_bytes * 4);
    // SAFETY: `ex_ref_sparse` provides `4 * slice_bytes` scratch and all device pointers belong to `arena`.
    let rc_ref = unsafe {
        ffi::w34_sampler_step_f16(
            &ex_ref_sparse,
            d_logits as *const c_void,
            d_cfgs as *const W34SamplerRowCfg,
            d_masks as *const u32,
            d_sparse as *const W34SamplerSparseEntry,
            d_out_ref as *mut W34SamplerRowOut,
            num_rows as i32,
            vocab_size as i32,
            sparse.len() as i32,
        )
    };
    assert_eq!(rc_ref, 0);

    let got_fast = copy_sampler_rows_d2h(&stream, d_out_fast, num_rows);
    let got_ref = copy_sampler_rows_d2h(&stream, d_out_ref, num_rows);

    for (r, expected_status, expected_id) in [
        (0usize, 0, allowed_win_r0 as i32),
        (1, 0, 501),
        (2, 0, 88_889),
        (3, 0, 140_001),
        (4, 1, -1),
        (5, 2, -1),
    ] {
        assert_eq!(
            (got_fast[r].status, got_fast[r].sampled_id),
            (expected_status, expected_id),
            "row {r}"
        );
    }
    assert_eq!(
        (
            got_fast[2].draft_accepted,
            got_fast[2].draft_prob,
            got_fast[3].draft_accepted,
            got_fast[3].draft_prob,
        ),
        (1, 1.0, 0, 0.0)
    );

    for r in 0..num_rows {
        assert_eq!(
            (
                got_fast[r].status,
                got_fast[r].sampled_id,
                got_fast[r].draft_accepted,
                got_fast[r].draft_prob.to_bits(),
            ),
            (
                got_ref[r].status,
                got_ref[r].sampled_id,
                got_ref[r].draft_accepted,
                got_ref[r].draft_prob.to_bits(),
            ),
            "row {r}"
        );
    }

    // Also verify `w34_sampler_step_f16` mixed batch `[stochastic row 1 + masked greedy rows]`
    // with `total_sparse_entries == 0` exercises the automatic `fast_parts` masked split path.
    let mut mixed_cfgs = cfgs.clone();
    mixed_cfgs[1].temperature = 0.8;
    mixed_cfgs[1].top_k = 32;
    mixed_cfgs[1].sparse_count = 0;
    mixed_cfgs[1].top_logprobs_k = -1;
    let d_mixed_cfgs = arena.upload_slice(&mixed_cfgs, &stream).unwrap();
    let mut got_mixed = vec![W34SamplerRowOut::default(); num_rows];
    // SAFETY: `ex` has room for `2 * slice_bytes + parts_bytes`; all pointers live on `arena`.
    unsafe {
        assert_eq!(
            ffi::w34_sampler_step_f16(
                &ex,
                d_logits as *const c_void,
                d_mixed_cfgs as *const W34SamplerRowCfg,
                d_masks as *const u32,
                core::ptr::null(),
                d_out_fast as *mut W34SamplerRowOut,
                num_rows as i32,
                vocab_size as i32,
                0,
            ),
            0
        );
        assert_eq!(
            ffi::w34_memcpy_d2h_async(
                got_mixed.as_mut_ptr() as *mut c_void,
                d_out_fast as *const c_void,
                out_bytes,
                stream.handle,
            ),
            0
        );
    }
    stream.synchronize().unwrap();
    assert_eq!(got_mixed[1].status, 0);
    for r in [0usize, 2, 3, 4, 5] {
        assert_eq!(
            got_mixed[r].status, got_ref[r].status,
            "mixed row {r} status"
        );
        assert_eq!(
            got_mixed[r].sampled_id, got_ref[r].sampled_id,
            "mixed row {r} sampled_id"
        );
        assert_eq!(
            got_mixed[r].draft_accepted, got_ref[r].draft_accepted,
            "mixed row {r} draft_accepted"
        );
    }
}
