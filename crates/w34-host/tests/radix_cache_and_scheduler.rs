use w34_host::parser::{Qwen3Parser, SamplerV1, SamplingParams, TokenBitmaskConstraint};
use w34_host::radix::{MambaRadixCache, RadixNodeId};

#[cfg(feature = "cuda")]
mod common;
#[cfg(feature = "cuda")]
use w34_host::cache::KvFormatPolicy;
#[cfg(feature = "cuda")]
use w34_host::model::Qwen3_5Engine;
#[cfg(feature = "cuda")]
use w34_host::scheduler::SchedulerConfig;
#[cfg(feature = "cuda")]
use w34_host::server::OpenAiServerState;

#[test]
fn test_mamba_radix_cache_dual_lock_dual_lru_tombstone_and_split() {
    let mut cache = MambaRadixCache::new(16, 64);

    // Insert 128-token sequence A (8 pages, mamba slot 10)
    let tok_a: Vec<i32> = (0..128).map(|i| 1000 + i).collect();
    let pages_a: Vec<i32> = (0..8).collect();
    let ins_a = cache.insert(&tok_a, &pages_a, &[], 10, 0);
    assert!(!ins_a.mamba_already_existed);
    assert_eq!(cache.full_evictable_pages(), 8);
    assert_eq!(cache.mamba_evictable_slots(), 1);
    cache.sanity_check().expect("sanity check after insert A");

    // Insert 128-token sequence B sharing first 64 tokens (4 pages) with A, diverging at 64..128
    let mut tok_b = tok_a[..64].to_vec();
    tok_b.extend((0..64).map(|i| 2000 + i));
    let pages_b: Vec<i32> = vec![0, 1, 2, 3, 8, 9, 10, 11];
    let ins_b = cache.insert(&tok_b, &pages_b, &[], 11, 4);
    assert_eq!(ins_b.existing_prefix_tokens, 64);
    assert!(!ins_b.mamba_already_existed);
    // Split created an internal 64-token node (mamba_value = None) + two 64-token leaves (mamba 10, 11)
    assert_eq!(cache.full_evictable_pages(), 12);
    assert_eq!(cache.mamba_evictable_slots(), 2);
    cache.sanity_check().expect("sanity check after split B");

    // Revive the 64-token internal split node's Mamba tombstone with slot 12
    let ins_mid = cache.insert(&tok_a[..64], &pages_a[..4], &[], 12, 4);
    assert!(!ins_mid.mamba_already_existed);
    assert_eq!(cache.mamba_evictable_slots(), 3);
    cache
        .sanity_check()
        .expect("sanity check after reviving internal tombstone");

    // Lock leaf B (`ins_b.terminal_node`): locks mamba on leaf B (slot 11) and full KV on [leaf B, internal_64]
    cache.inc_lock_ref(ins_b.terminal_node);
    assert_eq!(cache.full_protected_pages(), 8);
    assert_eq!(cache.full_evictable_pages(), 4); // Only leaf A's 4 pages are unlocked
    assert_eq!(cache.mamba_protected_slots(), 1); // Only leaf B's mamba slot is locked; internal_64's mamba slot (12) is unlocked!
    cache
        .sanity_check()
        .expect("sanity check with leaf B locked");

    // Touch leaf A so internal_64 becomes the LRU unlocked mamba state
    let m_a = cache.match_prefix(&tok_a);
    assert_eq!(m_a.matched_tokens, 128);
    assert_eq!(m_a.mamba_slot, Some(10));

    // Evict 1 Mamba state: should tombstone `internal_64` (slot 12) while keeping its 4 KV pages
    // alive because `full_lock_ref > 0` from leaf B!
    let ev1 = cache.evict_mamba(1);
    assert_eq!(ev1.freed_mamba_slots, vec![12]);
    assert!(ev1.freed_kv_pages.is_empty());
    cache
        .sanity_check()
        .expect("sanity check after tombstoning internal_64");

    // Unlock leaf B and evict all remaining leaves: should delete leaf B and leaf A and iteratively
    // clean up the exposed tombstone parent `internal_64`!
    cache.dec_lock_ref(ins_b.terminal_node);
    assert!(cache.last_node_check(RadixNodeId::ROOT));
    let ev_all = cache.evict_full_pages(12);
    assert_eq!(ev_all.freed_kv_pages.len(), 12);
    assert_eq!(ev_all.freed_mamba_slots.len(), 2);
    assert_eq!(cache.full_evictable_pages(), 0);
    assert_eq!(cache.mamba_evictable_slots(), 0);
    cache
        .sanity_check()
        .expect("sanity check after full eviction");
}

trait RadixTestExt {
    fn last_node_check(&self, id: RadixNodeId) -> bool;
}
impl RadixTestExt for MambaRadixCache {
    fn last_node_check(&self, id: RadixNodeId) -> bool {
        self.node(id).alive
    }
}

#[test]
fn test_qwen3_parser_and_sampler_v1_speculative_rollback() {
    // 1. Test streaming Qwen3Parser with <think>...</think> and Qwen3-XML <tool_call>
    let mut parser = Qwen3Parser::new(true);
    let chunks = [
        "<think>Let me check the weather ",
        "in Tokyo first.</think>\nI will call the weather tool.\n<tool_",
        "call>\n<function=get_weather>\n<parameter=city>\nTokyo\n</parameter>\n<parameter=unit>celsius</parameter>\n</function>\n</tool_call>",
    ];
    let mut streamed_r = String::new();
    let mut streamed_c = String::new();
    let mut streamed_tools = Vec::new();
    for ch in chunks {
        let d = parser.push_str(ch);
        if let Some(r) = d.reasoning_delta {
            streamed_r.push_str(&r);
        }
        if let Some(c) = d.content_delta {
            streamed_c.push_str(&c);
        }
        streamed_tools.extend(d.completed_tool_calls);
    }
    let (tail_d, final_turn) = parser.finish();
    if let Some(r) = tail_d.reasoning_delta {
        streamed_r.push_str(&r);
    }
    if let Some(c) = tail_d.content_delta {
        streamed_c.push_str(&c);
    }
    streamed_tools.extend(tail_d.completed_tool_calls);

    assert_eq!(streamed_r, "Let me check the weather in Tokyo first.");
    assert_eq!(streamed_c.trim(), "I will call the weather tool.");
    assert_eq!(streamed_tools.len(), 1);
    assert_eq!(streamed_tools[0].function.name, "get_weather");
    assert_eq!(
        streamed_tools[0].function.arguments,
        r#"{"city":"Tokyo","unit":"celsius"}"#
    );
    assert_eq!(final_turn.tool_calls, streamed_tools);

    // 2. Test Hermes JSON <tool_call> and implicit </think> when <tool_call> starts inside reasoning
    let hermes_raw = r#"Reasoning directly into tool<tool_call>{"name":"search_docs","arguments":{"query":"MambaRadixCache"}}</tool_call>"#;
    let parsed_hermes = Qwen3Parser::parse_complete(hermes_raw, true);
    assert_eq!(
        parsed_hermes.reasoning_content.as_deref(),
        Some("Reasoning directly into tool")
    );
    assert_eq!(parsed_hermes.content, None);
    assert_eq!(parsed_hermes.tool_calls.len(), 1);
    assert_eq!(parsed_hermes.tool_calls[0].function.name, "search_docs");
    assert_eq!(
        parsed_hermes.tool_calls[0].function.arguments,
        r#"{"query":"MambaRadixCache"}"#
    );

    let typed_xml_raw = "<tool_call><function=query_db><parameter=count>5</parameter><parameter=active>true</parameter></function></tool_call>";
    let parsed_typed = Qwen3Parser::parse_complete(typed_xml_raw, false);
    assert_eq!(parsed_typed.tool_calls.len(), 1);
    assert_eq!(
        parsed_typed.tool_calls[0].function.arguments,
        r#"{"count":5,"active":true}"#
    );

    // 3. Test SamplerV1 + TokenBitmaskConstraint transactional rollback under speculative rejection
    let params = SamplingParams {
        temperature: 0.7,
        top_k: 4,
        top_p: 0.95,
        min_p: 0.05,
        repetition_penalty: 1.2,
        frequency_penalty: 0.4,
        presence_penalty: 0.2,
        seed: 12345,
        ..Default::default()
    };
    let constraint = TokenBitmaskConstraint::from_allowed_tokens(16, vec![3, 4, 5, 7]);
    let mut sampler_clean = SamplerV1::new(params.clone(), &[1, 2], Some(constraint.clone()));
    let mut sampler_spec = SamplerV1::new(params, &[1, 2], Some(constraint));

    let mut logits_step0 = vec![10.0f32; 16];
    logits_step0[3] = 100.0;
    let t0_clean = sampler_clean.sample_from_logits(&logits_step0);
    let t0_spec = sampler_spec.sample_from_logits(&logits_step0);
    assert_eq!(t0_clean, 3);
    assert_eq!(t0_spec, 3);

    // Speculative sampler advances 3 bogus draft tokens and then rolls them back
    sampler_spec.commit_token(5);
    sampler_spec.commit_token(4);
    sampler_spec.commit_token(7);
    sampler_spec.rollback_tokens(3);

    let logits_step1: Vec<f32> = (0..16).map(|i| (i as f32) * 0.3).collect();
    let t1_clean = sampler_clean.sample_from_logits(&logits_step1);
    let t1_spec = sampler_spec.sample_from_logits(&logits_step1);
    assert_eq!(t1_clean, t1_spec);
    assert_eq!(sampler_clean.output_history, sampler_spec.output_history);
}

#[cfg(feature = "cuda")]
static GPU_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

#[cfg(feature = "cuda")]
#[test]
fn test_warm_prefix_hit_speculation_and_cross_boundary_rejection() {
    let _gpu_guard = GPU_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    let mut engine = Qwen3_5Engine::load(
        "nchapman/Qwen3.5-4B-W4A16-GPTQ",
        1,
        KvFormatPolicy::Fp16,
        Some(128),
        false,
    )
    .expect("load 4B TP=1 for warm prefix speculation test");

    // Build a 122-token prompt (`64` prefix + `58` suffix -> 6 tokens short of the `C = 128` checkpoint boundary!)
    let prompt_122: Vec<i32> = (0..122).map(|i| 1500 + ((i * 7 + 3) % 400)).collect();
    let gen_len = 14usize; // Total sequence reaches 122 + 14 = 136 tokens (crossing the C = 128 boundary!)

    // 1. Cold non-speculative reference run (chunk size 8 so Marlin uses identical M<=8 tile config)
    engine.reset_all_states().expect("reset before cold ref");
    let cold_ref = engine
        .verify_speculative_with_radix(&prompt_122, gen_len, 8, true, |_round, _anchor| Vec::new())
        .expect("cold non-speculative reference run");
    assert_eq!(cold_ref.cached_prefix_tokens, 0);
    assert_eq!(cold_ref.generated_token_ids.len(), gen_len);

    // Capture the cold reference's final `conv_state` (FP16) and `ssm_state` (FP32) from `cold_ref.active_state_slot`
    let ref_slot = cold_ref.active_state_slot as usize;
    let r0 = &engine.ranks[0];
    r0.device.make_current().unwrap();
    let conv_elems_per_slot = r0.state_pool.conv_dim * r0.state_pool.conv_tap_count;
    let ssm_elems_per_slot =
        r0.state_pool.local_v_heads * r0.state_pool.head_v_dim * r0.state_pool.head_k_dim;
    let mut cold_conv = vec![0u16; conv_elems_per_slot];
    let mut cold_ssm = vec![0.0f32; ssm_elems_per_slot];
    // SAFETY: `cold_conv`/`cold_ssm` are sized to one slot and synchronized on `r0.stream` below.
    unsafe {
        w34_kernels::ffi::w34_memcpy_d2h_async(
            cold_conv.as_mut_ptr() as *mut core::ffi::c_void,
            (r0.state_pool.layers[0].conv_state.addr()
                + (ref_slot * conv_elems_per_slot * 2) as u64)
                as *const core::ffi::c_void,
            conv_elems_per_slot * 2,
            r0.stream.handle,
        );
        w34_kernels::ffi::w34_memcpy_d2h_async(
            cold_ssm.as_mut_ptr() as *mut core::ffi::c_void,
            (r0.state_pool.layers[0].ssm_state.addr() + (ref_slot * ssm_elems_per_slot * 4) as u64)
                as *const core::ffi::c_void,
            ssm_elems_per_slot * 4,
            r0.stream.handle,
        );
    }
    r0.stream.synchronize().unwrap();

    // 2. Reset and run Request A on the 72-token prefix (`prompt_122[..72]`) to populate `MambaRadixCache`
    // at `C = 64` (`4` KV pages + GDN `StateBundle` checkpoint at 64).
    engine.reset_all_states().expect("reset before Request A");
    let req_a = engine
        .verify_speculative_with_radix(&prompt_122[..72], 2, 8, false, |_r, _a| Vec::new())
        .expect("Request A prefix population");
    assert_eq!(req_a.donated_seq_len, 64);
    engine
        .radix
        .sanity_check()
        .expect("radix sanity after Req A");

    // 3. Request B hits the warm 64-token prefix in `MambaRadixCache`, runs suffix prefill on
    // `64..122` (`58` tokens), and then runs speculative verify (`1 + k = 8` tokens per round)
    // across the `C = 128` checkpoint boundary with a FORCED mid-block rejection before 128
    // (at round 0, draft index 2 -> token position 125 < 128) and another forced mid-block
    // rejection at round 1 (crossing 128)!
    let ref_tokens = cold_ref.generated_token_ids.clone();
    let mut emitted_so_far = 1usize; // anchor_tok is ref_tokens[0]
    let req_b = engine
        .verify_speculative_with_radix(&prompt_122, gen_len, 8, true, |round_idx, anchor| {
            // Find current position in `ref_tokens`
            assert_eq!(anchor, ref_tokens[emitted_so_far - 1]);
            let mut drafts = Vec::new();
            for j in 0..7 {
                let target_idx = emitted_so_far + j;
                let true_tok = if target_idx < ref_tokens.len() {
                    ref_tokens[target_idx]
                } else {
                    100
                };
                if round_idx == 0 && j == 2 {
                    // Force mid-block rejection at step 2 (seq_len = 122 + 1 + 2 = 125 < 128),
                    // even though the 8-token verify batch [122..130] crosses the C = 128 boundary!
                    drafts.push(true_tok ^ 0x1FF);
                    emitted_so_far += 3; // 2 accepted drafts + 1 recovered token
                } else if round_idx == 1 && j == 4 {
                    // Force a second mid-block rejection after crossing 128!
                    drafts.push(true_tok ^ 0x1FF);
                    emitted_so_far += 5; // 4 accepted drafts + 1 recovered token
                } else {
                    drafts.push(true_tok);
                }
            }
            if round_idx >= 2 {
                emitted_so_far += 8;
            }
            drafts
        })
        .expect("Request B warm prefix-hit speculative verify with cross-boundary rejection");

    assert_eq!(
        req_b.cached_prefix_tokens, 64,
        "Request B must hit the 64-token prefix in MambaRadixCache"
    );
    assert_eq!(
        req_b.accepted_counts_per_round[0], 2,
        "Round 0 must accept exactly 2 drafts before mid-block rejection at pos 125 < 128"
    );
    assert_eq!(
        req_b.accepted_counts_per_round[1], 4,
        "Round 1 must accept 4 drafts across the C = 128 boundary"
    );
    assert_eq!(
        req_b.donated_seq_len, 128,
        "Request B must donate the true committed C = 128 checkpoint to MambaRadixCache"
    );
    assert_eq!(
        req_b.generated_token_ids, cold_ref.generated_token_ids,
        "Request B generated tokens must match cold non-speculative reference token-for-token"
    );

    // Compare Request B's final `conv_state` (FP16) and `ssm_state` (FP32) against cold reference: must be 0.0 ULP!
    let b_slot = req_b.active_state_slot as usize;
    let r0 = &engine.ranks[0];
    let mut b_conv = vec![0u16; conv_elems_per_slot];
    let mut b_ssm = vec![0.0f32; ssm_elems_per_slot];
    // SAFETY: `b_conv`/`b_ssm` are sized to one slot and synchronized on `r0.stream` below.
    unsafe {
        w34_kernels::ffi::w34_memcpy_d2h_async(
            b_conv.as_mut_ptr() as *mut core::ffi::c_void,
            (r0.state_pool.layers[0].conv_state.addr() + (b_slot * conv_elems_per_slot * 2) as u64)
                as *const core::ffi::c_void,
            conv_elems_per_slot * 2,
            r0.stream.handle,
        );
        w34_kernels::ffi::w34_memcpy_d2h_async(
            b_ssm.as_mut_ptr() as *mut core::ffi::c_void,
            (r0.state_pool.layers[0].ssm_state.addr() + (b_slot * ssm_elems_per_slot * 4) as u64)
                as *const core::ffi::c_void,
            ssm_elems_per_slot * 4,
            r0.stream.handle,
        );
    }
    r0.stream.synchronize().unwrap();
    assert_eq!(
        b_conv, cold_conv,
        "conv_state must be bit-identical (0 ULP) to cold non-speculative reference"
    );
    let max_ssm_diff = b_ssm
        .iter()
        .zip(cold_ssm.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    assert_eq!(
        max_ssm_diff, 0.0,
        "ssm_state must be bit-identical (0.0 max abs diff) to cold non-speculative reference"
    );

    // 4. Request C hits Request B's newly donated 128-token checkpoint in `MambaRadixCache`
    // ( verifying the rejected speculative checkpoint at round 0 did NOT pollute the radix tree!).
    let mut full_seq_130 = prompt_122.clone();
    full_seq_130.extend_from_slice(&cold_ref.generated_token_ids[..8]); // 122 + 8 = 130 tokens
    let req_c = engine
        .verify_speculative_with_radix(&full_seq_130, 6, 8, false, |_r, _a| Vec::new())
        .expect("Request C hitting 128-token checkpoint donated by Request B");
    assert_eq!(
        req_c.cached_prefix_tokens, 128,
        "Request C must hit the 128-token checkpoint donated by Request B"
    );
    assert_eq!(
        req_c.generated_token_ids,
        cold_ref.generated_token_ids[8..14],
        "Request C continuation from 128-token checkpoint must match cold reference token-for-token"
    );
    engine
        .radix
        .sanity_check()
        .expect("radix sanity after cross-boundary rejection");
}

#[cfg(feature = "cuda")]
#[test]
fn test_warm_prefix_ttft_drop_scheduler_batch_and_openai_http_sse_server() {
    let _gpu_guard = GPU_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    let mut engine = Qwen3_5Engine::load(
        "nchapman/Qwen3.5-4B-W4A16-GPTQ",
        1,
        KvFormatPolicy::Fp16,
        Some(256),
        true,
    )
    .expect("load 4B TP=1 for warm-prefix benchmark & server test");
    engine.reset_all_states().unwrap();

    // 1. Measure cold vs warm TTFT on a 516-token prompt (`512` prefix tokens = 8 x 64 checkpoints + `4` suffix tokens)
    let prompt_516: Vec<i32> = (0..516).map(|i| 1200 + ((i * 13 + 5) % 500)).collect();
    let cold_m = engine
        .generate_with_radix_cache(&prompt_516, 8, 64, false)
        .expect("cold 516-token generate_with_radix_cache");
    assert_eq!(cold_m.cached_prefix_tokens, 0);
    assert_eq!(cold_m.donated_seq_len, 512);

    let warm_m = engine
        .generate_with_radix_cache(&prompt_516, 8, 64, false)
        .expect("warm 516-token generate_with_radix_cache");
    assert_eq!(warm_m.cached_prefix_tokens, 512);
    assert_eq!(
        warm_m.generated_token_ids, cold_m.generated_token_ids,
        "warm prefix-cached generation must match cold generation token-for-token"
    );
    let ttft_drop_ratio = 1.0 - (warm_m.ttft_ms / cold_m.ttft_ms);
    println!(
        "[radix-cache] 516-token prompt cold_ttft={:.2}ms warm_ttft={:.2}ms reduction={:.1}%",
        cold_m.ttft_ms,
        warm_m.ttft_ms,
        ttft_drop_ratio * 100.0
    );
    assert!(
        ttft_drop_ratio > 0.80,
        "expected warm prefix-cache TTFT drop > 80%, got {:.1}% (cold={:.2}ms, warm={:.2}ms)",
        ttft_drop_ratio * 100.0,
        cold_m.ttft_ms,
        warm_m.ttft_ms
    );

    // 2. Test SchedulerV1 continuous batching across 4 concurrent requests sharing the 512-token prefix
    let mut batch_prompts = Vec::new();
    for k in 0..4 {
        let mut p = prompt_516[..512].to_vec();
        p.extend_from_slice(&[1800 + k, 1810 + k, 1820 + k, 1830 + k]);
        batch_prompts.push(p);
    }
    let batch_out = engine
        .generate_batch_scheduled(&batch_prompts, 6, SchedulerConfig::default())
        .expect("generate_batch_scheduled on 4 concurrent requests");
    assert_eq!(batch_out.len(), 4);
    for (gen_ids, cached_len) in &batch_out {
        assert_eq!(gen_ids.len(), 6);
        assert_eq!(*cached_len, 512);
    }

    // 3. Test OpenAI HTTP/1.1 + SSE Server (`/health`, `/v1/models`, `/tokenize`, `/v1/chat/completions` JSON & SSE, `/metrics`)
    let state = OpenAiServerState::new("nchapman/Qwen3.5-4B-W4A16-GPTQ".to_string(), engine);
    let send_req = |method: &str, path: &str, body: Option<&str>| {
        common::raw_request(&state, method, path, body)
    };

    let health_resp = send_req("GET", "/health", None);
    assert!(health_resp.contains("200 OK") && health_resp.contains(r#""status":"ok""#));

    let models_resp = send_req("GET", "/v1/models", None);
    assert!(
        models_resp.contains("200 OK") && models_resp.contains("nchapman/Qwen3.5-4B-W4A16-GPTQ")
    );

    let tok_resp = send_req("POST", "/tokenize", Some(r#"{"prompt":"Hello warp-34!"}"#));
    assert!(tok_resp.contains("200 OK") && tok_resp.contains(r#""tokens":"#));

    let chat_req = r#"{"model":"nchapman/Qwen3.5-4B-W4A16-GPTQ","messages":[{"role":"user","content":"Say hello in one word."}],"max_tokens":12,"chat_template_kwargs":{"enable_thinking":false}}"#;
    let chat_resp = send_req("POST", "/v1/chat/completions", Some(chat_req));
    assert!(
        chat_resp.contains("200 OK") && chat_resp.contains(r#""object":"chat.completion""#),
        "unexpected chat_resp: {chat_resp}"
    );

    let chat_sse_req = r#"{"model":"nchapman/Qwen3.5-4B-W4A16-GPTQ","messages":[{"role":"user","content":"Say hello in one word."}],"max_tokens":8,"stream":true,"chat_template_kwargs":{"enable_thinking":false}}"#;
    let chat_sse_resp = send_req("POST", "/v1/chat/completions", Some(chat_sse_req));
    assert!(
        chat_sse_resp.contains("text/event-stream") && chat_sse_resp.contains("data: [DONE]"),
        "unexpected chat_sse_resp: {chat_sse_resp}"
    );

    verify_shared_live_telemetry(&state);
    common::shutdown(&state);
}

#[cfg(feature = "cuda")]
fn collect_engine_tokens(
    runtime: tokio::runtime::Runtime,
    mut generation: w34_host::engine_worker::GenerationStream,
) -> usize {
    use w34_host::engine_worker::GenerationEvent;
    runtime.block_on(async {
        let mut tokens = 0;
        while let Some(event) = generation.events.recv().await {
            match event {
                GenerationEvent::Token { .. } => tokens += 1,
                GenerationEvent::Error(e) => panic!("generation failed: {e}"),
                GenerationEvent::Finished { .. } => {}
            }
        }
        tokens
    })
}

#[cfg(feature = "cuda")]
fn verify_shared_live_telemetry(state: &std::sync::Arc<OpenAiServerState>) {
    use serde_json::{Value, json};
    let before = state.metrics.snapshot();
    assert_eq!(
        before.requests.submitted, 2,
        "both JSON and SSE enter the common worker lifecycle"
    );
    assert_eq!(before.requests.completed + before.requests.cancelled, 2);
    assert!(before.tokens.generated > 0);
    assert_eq!(before.latency.ttft.count, 2);
    assert_eq!(before.http_responses.success, 2);

    // Validation failures are HTTP responses, not fictitious engine requests.
    let (head, _) = common::request(state, "POST", "/v1/chat/completions", json!({}));
    assert!(head.contains("400 Bad Request"));
    assert_eq!(state.metrics.snapshot().requests.submitted, 2);
    assert_eq!(state.metrics.snapshot().http_responses.client_error, 1);

    let (head, snapshot) = common::request(state, "GET", "/api/metrics", Value::Null);
    assert!(
        head.to_ascii_lowercase()
            .contains("cache-control: no-store")
    );
    assert_eq!(snapshot["schema_version"], 1);
    for name in ["kv", "gdn"] {
        let pool = &snapshot["pools"][name];
        let used = pool["active"].as_u64().expect("known active occupancy");
        let free = pool["free"].as_u64().expect("known free occupancy");
        assert_eq!(
            used + free + pool["cached"].as_u64().unwrap(),
            pool["capacity"].as_u64().unwrap()
        );
    }
    assert_eq!(snapshot["tokens"]["generated"], before.tokens.generated);
    let scrape = common::raw_request(state, "GET", "/metrics", None);
    assert!(scrape.contains(&format!(
        "w34_tokens_generated_total {}\n",
        before.tokens.generated
    )));
    assert!(scrape.contains("# TYPE w34_ttft_seconds histogram"));
    assert_eq!(
        state.metrics.snapshot().http_responses.success,
        2,
        "scrapes are not inference responses"
    );

    verify_metrics_while_generating(state, before, snapshot);
}

#[cfg(feature = "cuda")]
fn verify_metrics_while_generating(
    state: &std::sync::Arc<OpenAiServerState>,
    before: w34_host::telemetry::Snapshot,
    snapshot: serde_json::Value,
) {
    use serde_json::Value;
    use std::time::{Duration, Instant};
    use w34_host::parser::SamplingParams;
    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .unwrap();
    // Decision readout is prefill-only: contributes queue/e2e/prompt, never TTFT/output.
    let (decision_stream, decision) = state
        .worker
        .submit_decision_candidate_logits(vec![1200; 65], vec![1201], None)
        .unwrap();
    runtime.block_on(async {
        decision.await.unwrap().unwrap();
    });
    drop(decision_stream);

    let generation = state
        .worker
        .submit(
            vec![1300; 128],
            SamplingParams {
                max_tokens: 256,
                min_tokens: 256,
                ignore_eos: true,
                temperature: 0.,
                top_k: 1,
                ..SamplingParams::default()
            },
            None,
        )
        .unwrap();
    let consumer = std::thread::spawn(move || collect_engine_tokens(runtime, generation));
    let mut saw_inflight = false;
    for _ in 0..3 {
        let start = Instant::now();
        let (head, live) = common::request(state, "GET", "/api/metrics", Value::Null);
        assert!(head.contains("200 OK"));
        assert!(
            start.elapsed() < Duration::from_millis(500),
            "scrape waited on engine work"
        );
        assert_eq!(live["instance_id"], snapshot["instance_id"]);
        let requests = &live["requests"];
        saw_inflight |=
            requests["running"].as_u64().unwrap() + requests["waiting"].as_u64().unwrap() > 0;
        assert_eq!(
            requests["submitted"].as_u64().unwrap(),
            ["waiting", "running", "completed", "failed", "cancelled"]
                .iter()
                .map(|k| requests[k].as_u64().unwrap())
                .sum::<u64>()
        );
    }
    assert!(
        saw_inflight,
        "must scrape while engine work is actually in flight"
    );
    assert_eq!(consumer.join().unwrap(), 256);
    let after = state.metrics.snapshot();
    assert_eq!(after.tokens.generated - before.tokens.generated, 256);
    assert_eq!(after.latency.ttft.count - before.latency.ttft.count, 1);
    assert_eq!(after.requests.waiting + after.requests.running, 0);
    assert_eq!(after.requests.failed, 0);
    let kv = &after.pools.kv;
    assert_eq!(
        kv.capacity,
        kv.active.unwrap() + kv.cached + kv.free.unwrap()
    );
}
