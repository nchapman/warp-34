//! Phase 6.6a GPU-resident stochastic `SamplerV1` + device grammar bitmasking gates.
#![cfg(feature = "cuda")]

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
    if CudaDevice::count().unwrap_or(0) < 2 {
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

    let run_tp = |tp_size: usize, with_mtp: bool| -> (Vec<i32>, usize) {
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
            temperature: 0.7,
            top_k: 32,
            top_p: 0.9,
            min_p: 0.02,
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
        assert_eq!(res.token_logprobs.len(), res.generated_token_ids.len());
        assert_eq!(res.token_logprobs[0].top_logprobs.len(), 3);
        (res.generated_token_ids, res.token_logprobs.len())
    };

    let (tp1_plain, lp1) = run_tp(1, false);
    let (tp2_plain, lp2) = run_tp(2, false);
    assert_eq!(lp1, lp2);
    let text1 = {
        let e = Qwen3_5Engine::load(model, 1, KvFormatPolicy::Tq4, Some(16), true).unwrap();
        (e.decode(&tp1_plain).unwrap(), e.decode(&tp2_plain).unwrap())
    };
    assert!(
        serde_json::from_str::<serde_json::Value>(&text1.0).is_ok()
            && serde_json::from_str::<serde_json::Value>(&text1.1).is_ok(),
        "both TP=1 and TP=2 non-speculative GPU sampler outputs must satisfy strict JSON schema"
    );

    let (tp1_mtp, mlp1) = run_tp(1, true);
    let (tp2_mtp, mlp2) = run_tp(2, true);
    assert_eq!(mlp1, mlp2);
    assert!(!tp1_mtp.is_empty() && !tp2_mtp.is_empty());
}
