#![cfg(feature = "cuda")]
mod common;
use w34_host::{
    cache::KvFormatPolicy,
    model::Qwen3_5Engine,
    mtp::{Qwen3_5MtpRank, validate_geometry},
};
use w34_kernels::ffi::{W34_KV_NAMESPACE_MTP_FULL_ATTN, W34_KV_NAMESPACE_TARGET_FULL_ATTN};

/// Real cached checkpoint, physical GPU chosen by CUDA_VISIBLE_DEVICES.
#[test]
fn cached_4b_mtp_k123() {
    if common::require_or_skip_cached_model("nchapman/Qwen3.5-4B-W4A16-GPTQ", "cached_4b_mtp_k123")
        .is_none()
    {
        return;
    }
    let mut engine = Qwen3_5Engine::load(
        "nchapman/Qwen3.5-4B-W4A16-GPTQ",
        1,
        KvFormatPolicy::Fp16,
        Some(8),
        false,
    )
    .unwrap();
    let tokens = engine.encode("The capital of France is").unwrap();
    engine.ranks[0]
        .configure_speculative_taps(&[], true, 64)
        .unwrap();
    let page = engine.alloc_kv_page().unwrap();
    let state = engine.alloc_state_slot().unwrap();
    let (anchor, _, _) = engine
        .step_tokens(&tokens, 0, &[page], state, state, false, None)
        .unwrap();
    let target = &engine.ranks[0];
    let hidden = target.last_final_hidden_f16.as_ref().unwrap().addr();
    let snapshot = || {
        let mut bytes = Vec::new();
        for buf in [
            &target.state_pool.layers[0].ssm_state,
            &target.state_pool.layers[0].conv_state,
            &target.kv_pool.layers[0].k_buf,
            target.last_final_hidden_f16.as_ref().unwrap(),
        ] {
            let mut data = vec![0u8; buf.len_bytes];
            buf.copy_to_host(&mut data, &target.stream).unwrap();
            bytes.extend(data);
        }
        bytes
    };
    let before = snapshot();
    let mut mtp = Qwen3_5MtpRank::load(target, &engine.model_dir, 64).unwrap();
    check_mtp_input_validation(&engine, &mut mtp, anchor, hidden);
    assert!(mtp.is_empty());
    mtp.ingest(target, &[], 0, None).unwrap();
    mtp.ingest(target, &tokens[1..], hidden, None).unwrap();
    let p = tokens.len() - 1;
    assert_eq!(mtp.committed_len(), p);
    let last = hidden + (p * engine.cfg.hidden_size * 2) as u64;
    assert!(mtp.propose(target, anchor, last, 0, None).is_err());
    assert!(mtp.propose(target, anchor, last, 4, None).is_err());
    assert_eq!(mtp.len(), p);
    let mut previous = vec![];
    for k in 1..=3 {
        let drafts = mtp.propose(target, anchor, last, k, None).unwrap();
        assert!(
            mtp.has_propose_graph(k),
            "MTP rank with quantized draft_lm_head must capture propose CUDA graph for k={k}"
        );
        eprintln!("k={k} anchor={anchor} MTP={drafts:?}");
        assert_eq!(drafts.len(), k);
        assert!(
            drafts
                .iter()
                .all(|&id| id >= 0 && (id as usize) < engine.cfg.vocab_size)
        );
        assert_eq!(&drafts[..previous.len()], previous.as_slice());
        assert_eq!(mtp.len(), p + k);
        assert_eq!(mtp.committed_len(), p);
        assert!(mtp.ingest(target, &[anchor], last, None).is_err());
        assert!(mtp.truncate(p + k + 1).is_err());
        mtp.truncate(p).unwrap();
        let replay = mtp.propose(target, anchor, last, k, None).unwrap();
        assert_eq!(drafts, replay, "stale provisional KV must be unreachable");
        mtp.truncate(p).unwrap();
        previous = drafts;
    }
    assert_eq!(
        before,
        snapshot(),
        "MTP must not mutate target KV, GDN, or hidden"
    );
    mtp.truncate(0).unwrap();
    assert!(mtp.is_empty());
    mtp.ingest(target, &tokens[1..], hidden, None).unwrap();
    assert_eq!(
        mtp.propose(target, anchor, last, 3, None).unwrap(),
        previous
    );
    mtp.truncate(p).unwrap();
    check_draft_lm_head_quantized_vs_fp16_fallback(
        &mut engine,
        &mut mtp,
        &tokens,
        p,
        anchor,
        last,
        &previous,
    );
    check_lossless_rounds(&mut engine, &mut mtp, &tokens, 2);
    check_lossless_rounds(&mut engine, &mut mtp, &tokens, 12);
    check_vllm_hidden_replay(&engine, &mut mtp);
}

fn check_draft_lm_head_quantized_vs_fp16_fallback(
    engine: &mut Qwen3_5Engine,
    mtp: &mut Qwen3_5MtpRank,
    tokens: &[i32],
    p: usize,
    anchor: i32,
    last: u64,
    quant_graph_drafts: &[i32],
) {
    let target = &engine.ranks[0];
    // 1. Eager quantized draft_lm_head must match CUDA Graph quantized draft_lm_head bit-for-bit.
    mtp.use_cuda_graph = false;
    let quant_eager = mtp.propose(target, anchor, last, 3, None).unwrap();
    assert_eq!(quant_eager, quant_graph_drafts);
    mtp.truncate(p).unwrap();

    // 2. Detach quantized draft_lm_head to exercise DenseF16 target.lm_head fallback (both eager & CUDA graph).
    let saved_head = mtp.take_quantized_draft_lm_head();
    assert!(saved_head.is_some() && !mtp.has_quantized_draft_lm_head());
    assert!(!mtp.has_propose_graph(3));
    let fp16_eager = mtp.propose(target, anchor, last, 3, None).unwrap();
    mtp.truncate(p).unwrap();
    mtp.use_cuda_graph = true;
    let fp16_graph = mtp.propose(target, anchor, last, 3, None).unwrap();
    assert!(mtp.has_propose_graph(3));
    assert_eq!(fp16_eager, fp16_graph);
    assert!(
        fp16_graph
            .iter()
            .all(|&id| id >= 0 && (id as usize) < engine.cfg.vocab_size)
    );
    mtp.truncate(p).unwrap();
    // Lossless target verification with DenseF16 target.lm_head fallback must match greedy baseline.
    check_lossless_rounds(engine, mtp, tokens, 6);

    // 3. Restore quantized draft_lm_head.
    mtp.restore_quantized_draft_lm_head(saved_head);
    assert!(mtp.has_quantized_draft_lm_head());
    assert!(!mtp.has_propose_graph(3));
}

fn check_mtp_input_validation(
    engine: &Qwen3_5Engine,
    mtp: &mut Qwen3_5MtpRank,
    anchor: i32,
    hidden: u64,
) {
    let target = &engine.ranks[0];
    assert!(
        mtp.has_quantized_draft_lm_head(),
        "4B DenseF16 target.lm_head with ample free VRAM must enable quantized MarlinW4A16 draft_lm_head"
    );
    assert_eq!(mtp.kv_layout().namespace_id, W34_KV_NAMESPACE_MTP_FULL_ATTN);
    assert_eq!(
        target.kv_pool.layers[0].layout.namespace_id,
        W34_KV_NAMESPACE_TARGET_FULL_ATTN
    );
    assert_ne!(
        mtp.kv_layout().k_base,
        target.kv_pool.layers[0].layout.k_base
    );
    assert!(validate_geometry(&engine.cfg, 0, 3).is_err());
    let mut bad = engine.cfg.clone();
    bad.num_key_value_heads = 1;
    assert!(validate_geometry(&bad, 0, 2).is_err());
    for (bad_tokens, bad_hidden) in [
        (&[anchor; 65][..], hidden),
        (&[-1][..], hidden),
        (&[anchor][..], 0),
        (&[anchor][..], u64::MAX - 1),
    ] {
        assert!(mtp.ingest(target, bad_tokens, bad_hidden, None).is_err());
    }
}

fn check_vllm_hidden_replay(engine: &Qwen3_5Engine, mtp: &mut Qwen3_5MtpRank) {
    use w34_host::{
        cuda::{DeviceBuffer, f16_to_f32},
        loader::SafetensorsArchive,
    };
    let target = &engine.ranks[0];
    let props = target.device.props().unwrap();
    let active_gpu = &props.name;
    let max_abs_tol = if props.arch_family().unwrap() == w34_host::cuda::CudaArchFamily::AmpereSm86
        && props.multi_processor_count == 84
    {
        0.05
    } else {
        0.06
    };
    let h = engine.cfg.hidden_size;
    mtp.truncate(0).unwrap();
    // Actual eager vLLM native MTP prefill and two recurrent draft forwards.
    // Later captures may include masked padding; these three are fully live.
    let mut mismatches = Vec::new();
    for call in 0..3 {
        let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join(format!(
            "../../tools/oracle/goldens/vllm_mtp_4b/call_{call}.safetensors"
        ));
        let archive = SafetensorsArchive::open(&path).expect("run pinned vllm_mtp_dump.py");
        let read_ids = |name: &str| -> Vec<i32> {
            match archive.tensor_info(name).unwrap().dtype.as_str() {
                "I32" => archive.read_i32(name).unwrap(),
                "I64" => archive
                    .read_bytes(name)
                    .unwrap()
                    .as_chunks::<8>()
                    .0
                    .iter()
                    .map(|b| i32::try_from(i64::from_le_bytes(*b)).unwrap())
                    .collect(),
                dtype => panic!("unexpected integer dtype {dtype}"),
            }
        };
        let ids = read_ids("input_ids");
        let positions = read_ids("positions");
        let hidden = archive.read_f16_bits("hidden_states").unwrap();
        let expected = archive.read_f32("output").unwrap();
        assert_eq!(hidden.len(), ids.len() * h);
        assert_eq!(positions.len(), ids.len());
        let input = DeviceBuffer::alloc(0, hidden.len() * 2).unwrap();
        input.copy_from_host(&hidden, &target.stream).unwrap();
        assert_eq!(ids.len(), if call == 0 { 5 } else { 1 });
        for (row, &position) in positions.iter().enumerate() {
            assert_eq!(position as usize, mtp.len() + row);
        }
        mtp.ingest(target, &ids, input.addr(), None).unwrap();
        assert_eq!(mtp.batch_hidden_rows, ids.len());
        let mut bits = vec![0u16; ids.len() * h];
        mtp.batch_hidden
            .copy_to_host(&mut bits, &target.stream)
            .unwrap();
        let mut last = vec![0u16; h];
        mtp.hidden.copy_to_host(&mut last, &target.stream).unwrap();
        assert_eq!(last, bits[(ids.len() - 1) * h..]);
        for row in 0..ids.len() {
            let actual: Vec<f32> = bits[row * h..(row + 1) * h]
                .iter()
                .map(|&b| f16_to_f32(b))
                .collect();
            let reference = &expected[row * h..(row + 1) * h];
            let max_abs = actual
                .iter()
                .zip(reference)
                .map(|(a, b)| (a - b).abs())
                .fold(0.0f32, f32::max);
            let mse: f64 = actual
                .iter()
                .zip(reference)
                .map(|(a, b)| f64::from(a - b).powi(2))
                .sum();
            let energy: f64 = reference.iter().map(|&v| f64::from(v).powi(2)).sum();
            let rel_l2 = (mse / energy.max(1e-12)).sqrt();
            eprintln!(
                "vLLM MTP on {active_gpu} call={call} row={row} max_abs={max_abs} rel_l2={rel_l2}"
            );
            if !(actual.iter().all(|v| v.is_finite()) && max_abs < max_abs_tol && rel_l2 < 0.005) {
                mismatches.push(format!(
                    "call={call} row={row} max_abs={max_abs} rel_l2={rel_l2}"
                ));
            }
        }
    }
    assert!(
        mismatches.is_empty(),
        "MTP donor mismatches: {mismatches:?}"
    );
}

/// Exercises real target verify rows and StateTx selection with actual MTP
/// proposals. No forced target-token substitution or synthetic proposer.
fn check_lossless_rounds(
    engine: &mut Qwen3_5Engine,
    mtp: &mut Qwen3_5MtpRank,
    prompt: &[i32],
    output_tokens: usize,
) {
    use w34_host::{
        cache::StateTx,
        model::EngineBatchSlice,
        parser::{SamplerV1, SamplingParams},
        speculative::RejectionSampler,
    };
    let expected = engine
        .generate_tokens(prompt, output_tokens, false)
        .unwrap()
        .generated_token_ids;
    engine.reset_all_states().unwrap();
    mtp.truncate(0).unwrap();
    let table: Vec<i32> = (0..2).map(|_| engine.alloc_kv_page().unwrap()).collect();
    let state = engine.alloc_state_slot().unwrap();
    let (mut anchor, _, _) = engine
        .step_tokens(prompt, 0, &table, state, state, false, None)
        .unwrap();
    let h = engine.cfg.hidden_size;
    let target_hidden = engine.ranks[0]
        .last_final_hidden_f16
        .as_ref()
        .unwrap()
        .addr();
    mtp.ingest(&engine.ranks[0], &prompt[1..], target_hidden, None)
        .unwrap();
    let mut last = target_hidden + ((prompt.len() - 1) * h * 2) as u64;
    let mut tx = StateTx::new(state);
    tx.commit_prefill(prompt.len());
    let mut sampler = SamplerV1::new(SamplingParams::default(), prompt, None);
    sampler.commit_token(anchor);
    let mut output = vec![anchor];
    let mut ctx = prompt.len();
    while output.len() < expected.len() {
        let k = 3.min(expected.len() - output.len() - 1);
        if k == 0 {
            // No speculative StateTx exists for an empty draft block.
            let (next, _, _) = engine
                .step_tokens(
                    &[anchor],
                    ctx,
                    &table,
                    tx.committed_slot,
                    tx.committed_slot,
                    true,
                    None,
                )
                .unwrap();
            tx.commit_prefill(ctx + 1);
            sampler.commit_token(next);
            output.push(next);
            break;
        }
        let start = mtp.committed_len();
        let drafts = if k > 0 {
            mtp.propose(&engine.ranks[0], anchor, last, k, None)
                .unwrap()
        } else {
            vec![]
        };
        // Recurrent proposal rows are not canonical context. Replay using the
        // target's hidden before that buffer is overwritten by verification.
        mtp.truncate(start).unwrap();
        mtp.ingest(&engine.ranks[0], &[anchor], last, None).unwrap();
        let extra: Vec<i32> = (0..k).map(|_| engine.alloc_state_slot().unwrap()).collect();
        tx.begin_spec_verify(extra);
        let schedule = tx.verify_slot_schedule();
        let verify: Vec<i32> = std::iter::once(anchor)
            .chain(drafts.iter().copied())
            .collect();
        let rows = engine
            .step_batch_slices(&[EngineBatchSlice {
                tokens: &verify,
                ctx_len_before: ctx,
                block_table: &table,
                state_in_slot: tx.committed_slot,
                state_out_slot: *schedule.last().unwrap(),
                has_initial_state: true,
                spec_slots: &schedule,
                checkpoints: &[],
                emit_all_query_tokens: true,
            }])
            .unwrap();
        let result = RejectionSampler::greedy(&drafts, &rows[0]).unwrap();
        let freed = result.commit(&mut tx, &mut sampler, 0).unwrap();
        engine.free_state_slots(&freed);
        let a = result.accepted_drafts();
        mtp.ingest(&engine.ranks[0], &drafts[..a], target_hidden, None)
            .unwrap();
        last = target_hidden + (a * h * 2) as u64;
        ctx += result.processed_rows();
        output.extend_from_slice(result.emitted_tokens());
        anchor = *output.last().unwrap();
        assert_eq!(mtp.committed_len(), ctx - 1);
        assert_eq!(sampler.output_history, output);
    }
    assert_eq!(
        output, expected,
        "native MTP verification must preserve target greedy output"
    );
    engine.free_state_slots(&[tx.committed_slot]);
    engine.free_kv_pages(&table);
}

#[test]
fn explicit_tp_geometry() {
    use w34_host::loader::ModelConfig;
    let cfg = ModelConfig::from_json(&serde_json::json!({
        "hidden_size":2560,"intermediate_size":9728,"num_attention_heads":16,
        "num_key_value_heads":4,"head_dim":256,"vocab_size":248320,
        "mtp_num_hidden_layers":1,"attn_output_gate":true
    }))
    .unwrap();
    validate_geometry(&cfg, 0, 1).unwrap();
    validate_geometry(&cfg, 1, 2).unwrap();
    assert!(validate_geometry(&cfg, 0, 0).is_err());
    assert!(validate_geometry(&cfg, 2, 2).is_err());
    let mut bad = cfg.clone();
    bad.num_experts = 8;
    assert!(validate_geometry(&bad, 0, 1).is_err());
    let moe_cfg = ModelConfig::from_json(&serde_json::json!({
        "hidden_size":2048,"intermediate_size":0,"num_attention_heads":16,
        "num_key_value_heads":2,"head_dim":256,"vocab_size":248320,
        "mtp_num_hidden_layers":1,"attn_output_gate":true,
        "num_experts":256,"num_experts_per_tok":8,"moe_intermediate_size":512,
        "shared_expert_intermediate_size":512
    }))
    .unwrap();
    validate_geometry(&moe_cfg, 0, 1).unwrap();
    validate_geometry(&moe_cfg, 1, 2).unwrap();
    let mut bad_moe = moe_cfg.clone();
    bad_moe.moe_intermediate_size = 513;
    assert!(validate_geometry(&bad_moe, 0, 2).is_err());
    bad = cfg.clone();
    bad.intermediate_size += 1;
    assert!(validate_geometry(&bad, 0, 2).is_err());
    bad = cfg.clone();
    bad.num_key_value_heads = 1;
    assert!(validate_geometry(&bad, 0, 2).is_err());
    bad = cfg;
    bad.mtp_num_hidden_layers = 2;
    assert!(validate_geometry(&bad, 0, 1).is_err());
}
