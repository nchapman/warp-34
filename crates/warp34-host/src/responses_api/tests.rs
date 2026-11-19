use super::*;
use crate::parser::{ParsedFunctionCall, Qwen3Parser};
use crate::request_constraints::RequestConstraint;

#[test]
fn translates_flat_tools_tool_choice_none_and_previous_response_chain() {
    let store = ResponseStore::with_capacity(8);
    let req1 = json!({
        "model": "qwen-test",
        "instructions": "Be concise.",
        "input": "Read Cargo.toml",
        "tools": [{
            "type": "function",
            "name": "read_file",
            "description": "Read a file",
            "parameters": {
                "type": "object",
                "properties": {"path": {"type": "string"}},
                "required": ["path"]
            }
        }],
        "tool_choice": "none",
        "max_output_tokens": 128
    });

    let (chat1, meta1) =
        translate_responses_to_chat(&req1, "qwen-test", &store, Some("tenant_a"), 1).unwrap();
    assert!(!meta1.store);
    assert!(meta1.response_id.starts_with("resp_") && meta1.response_id.len() == 37);
    assert!(
        translate_responses_to_chat(
            &json!({"input": "hi", "store": true}),
            "qwen-test",
            &store,
            None,
            1
        )
        .is_err()
    );
    assert_eq!(
        (
            chat1["messages"][0]["role"].as_str(),
            chat1["messages"][0]["content"].as_str(),
            chat1["messages"][1]["role"].as_str(),
            chat1["messages"][1]["content"].as_str(),
            chat1["tool_choice"].as_str(),
        ),
        (
            Some("system"),
            Some("Be concise."),
            Some("user"),
            Some("Read Cargo.toml"),
            Some("none"),
        )
    );

    // Verify stateful tool_choice="none" bitmask constraint parses identically to Chat API
    assert!(RequestConstraint::parse(&chat1).unwrap().is_none());

    let turn1 = ParsedAssistantTurn {
        reasoning_content: Some("Need to call read_file".into()),
        content: None,
        tool_calls: vec![ParsedToolCall {
            id: "call_abc".into(),
            call_type: "function".into(),
            function: ParsedFunctionCall {
                name: "read_file".into(),
                arguments: "{\"path\":\"Cargo.toml\"}".into(),
            },
        }],
    };
    let (resp1, hist1) = build_response_object(&meta1, &turn1, "tool_calls", 50, 0, 20, 8);
    assert_eq!(resp1["status"], "completed");
    assert_eq!(resp1["output"].as_array().unwrap().len(), 2); // reasoning + function_call
    assert_eq!(resp1["output"][1]["type"], "function_call");
    assert_eq!(resp1["output"][1]["call_id"], "call_abc");

    store.insert(StoredResponse {
        id: meta1.response_id.clone(),
        owner_scope: "tenant_a".into(),
        response: resp1,
        history_messages: hist1,
    });
    assert!(store.get_response(&meta1.response_id, "tenant_b").is_none());
    // Flooding tenant_b up to and beyond store capacity must not evict tenant_a's sole entry.
    for idx in 0..16 {
        store.insert(StoredResponse {
            id: format!("resp_flood_{idx}"),
            owner_scope: "tenant_b".into(),
            response: json!({"id": idx}),
            history_messages: Vec::new(),
        });
    }
    assert!(store.get_response(&meta1.response_id, "tenant_a").is_some());

    // Turn 2 chains via previous_response_id with function_call_output
    let req2 = json!({
        "previous_response_id": meta1.response_id,
        "input": [{
            "type": "function_call_output",
            "call_id": "call_abc",
            "output": "[package]\nname = \"warp34\""
        }],
        "reasoning": {"effort": "none"},
        "text": {
            "format": {
                "type": "json_schema",
                "name": "pkg",
                "schema": {
                    "type": "object",
                    "properties": {"name": {"type": "string"}},
                    "required": ["name"],
                    "additionalProperties": false
                }
            }
        }
    });
    let (chat2, _meta2) =
        translate_responses_to_chat(&req2, "qwen-test", &store, Some("tenant_a"), 2).unwrap();
    assert!(translate_responses_to_chat(&req2, "qwen-test", &store, Some("tenant_b"), 2).is_err());
    assert!(translate_responses_to_chat(&req2, "qwen-test", &store, None, 2).is_err());
    let msgs2 = chat2["messages"].as_array().unwrap();
    assert_eq!(msgs2.len(), 4); // system, user, assistant(tool_calls), tool
    assert_eq!(msgs2[2]["role"], "assistant");
    assert_eq!(msgs2[3]["role"], "tool");
    assert_eq!(chat2["chat_template_kwargs"]["enable_thinking"], false);
    assert_eq!(chat2["response_format"]["type"], "json_schema");

    // Unknown previous_response_id returns 404 NotFound
    let bad_prev = json!({
        "previous_response_id": "resp_missing",
        "input": "hello"
    });
    let err = translate_responses_to_chat(&bad_prev, "qwen-test", &store, Some("tenant_a"), 3)
        .unwrap_err();
    assert_eq!(err.kind, ResponsesErrorKind::NotFound);
}

#[test]
fn forwards_vllm_sampling_and_structured_outputs_params() {
    let store = ResponseStore::default();
    let req = json!({
        "input": "test",
        "reasoning": {"effort": "xhigh"},
        "presence_penalty": 0.5,
        "frequency_penalty": -0.25,
        "repetition_penalty": 1.1,
        "min_p": 0.05,
        "seed": 123,
        "stop_token_ids": [151643],
        "min_tokens": 4,
        "ignore_eos": true,
        "logit_bias": {"42": 1.5},
        "top_logprobs": 2,
        "structured_outputs": {"regex": "[0-9]+"},
        "chat_template_kwargs": {"preserve_thinking": true}
    });
    let (chat, _) = translate_responses_to_chat(&req, "q", &store, None, 1).unwrap();
    assert_eq!(chat["presence_penalty"], 0.5);
    assert_eq!(chat["frequency_penalty"], -0.25);
    assert_eq!(chat["repetition_penalty"], 1.1);
    assert_eq!(chat["min_p"], 0.05);
    assert_eq!(chat["seed"], 123);
    assert_eq!(chat["min_tokens"], 4);
    assert_eq!(chat["ignore_eos"], true);
    assert_eq!(chat["logprobs"], true);
    assert_eq!(chat["top_logprobs"], 2);
    assert_eq!(chat["structured_outputs"]["regex"], "[0-9]+");
    assert_eq!(chat["chat_template_kwargs"]["enable_thinking"], true);
    assert_eq!(chat["chat_template_kwargs"]["preserve_thinking"], true);

    // Conflict between structured_outputs and non-text text.format must fail
    let conflict = json!({
        "input": "test",
        "text": {"format": {"type": "json_object"}},
        "structured_outputs": {"regex": "[0-9]+"}
    });
    assert!(translate_responses_to_chat(&conflict, "q", &store, None, 1).is_err());
}

#[test]
fn sse_formatter_emits_monotonic_zero_indexed_sequence_numbers() {
    let store = ResponseStore::default();
    let req = json!({"input": "Solve 2+2 and call report"});
    let (_, meta) = translate_responses_to_chat(&req, "qwen-test", &store, None, 7).unwrap();

    let mut parser = Qwen3Parser::new(true);
    let mut sse = ResponsesSseFormatter::new(meta);
    let mut frames = sse.start();
    for chunk in [
        "2+2 is 4.</think>",
        "Answer: 4.\n<tool_call>\n<function=report>\n<parameter=ans>\n4\n</parameter>\n</function>\n</tool_call>",
        "\n",
    ] {
        let d = parser.push_str(chunk);
        frames.extend(sse.push_delta(d));
    }
    let (tail, turn) = parser.finish();
    frames.extend(sse.push_delta(tail));
    let (end_frames, final_resp, _) = sse.finish(&turn, "tool_calls", 32, 16, 24, 6);
    frames.extend(end_frames);
    // Trailing "\n" after </tool_call> must not spawn a duplicate/empty message item.
    assert_eq!(final_resp["output"].as_array().unwrap().len(), 3);
    assert_eq!(
        store.scope_for_credential(Some(b"Bearer secret-token")),
        store.scope_for_credential(Some(b"secret-token"))
    );

    for (expected_seq, (ev_name, data)) in frames.iter().enumerate() {
        assert_eq!(data["type"], ev_name.as_str());
        assert_eq!(data["sequence_number"], expected_seq as u64);
    }
    assert_eq!(frames.first().unwrap().0, "response.created");
    assert_eq!(frames[1].0, "response.in_progress");
    assert_eq!(frames.last().unwrap().0, "response.completed");
    assert_eq!(final_resp["output"].as_array().unwrap().len(), 3);
    assert_eq!(
        final_resp["usage"]["input_tokens_details"]["cached_tokens"],
        16
    );
    assert_eq!(
        final_resp["usage"]["output_tokens_details"]["reasoning_tokens"],
        6
    );
}

#[test]
fn responses_sse_streams_incremental_function_call_deltas_without_duplication() {
    let store = ResponseStore::default();
    let req = json!({"input": "Write src/scene.js"});
    let (_, meta) = translate_responses_to_chat(&req, "qwen-test", &store, None, 8).unwrap();
    let tools = vec![json!({"function":{"name":"write","parameters":{
        "type":"object",
        "properties":{
            "path":{"type":"string"},
            "content":{"type":"string"}
        }
    }}})];
    let mut parser = Qwen3Parser::with_tools(false, &tools).unwrap();
    let mut sse = ResponsesSseFormatter::new(meta);
    let mut frames = sse.start();

    let f1 =
        sse.push_delta(parser.push_str("<tool_call>\n<function=write>\n<parameter=path>\nsrc/"));
    assert!(
        f1.iter().any(|(n, v)| n == "response.output_item.added"
            && v["item"]["type"] == "function_call"
            && v["item"]["name"] == "write"),
        "expected function_call output_item.added before tool_call closes: {f1:?}"
    );
    assert!(
        f1.iter()
            .any(|(n, _)| n == "response.function_call_arguments.delta"),
        "expected function_call_arguments.delta on first chunk: {f1:?}"
    );
    assert!(
        !f1.iter()
            .any(|(n, _)| n == "response.function_call_arguments.done"),
        "function_call must remain open while streaming arguments"
    );
    frames.extend(f1);

    for chunk in [
        "scene.js\n</parameter>\n<parameter=content>\nconst a = 1;",
        "\nconst b = 2;\n</parameter>\n</function>\n</tool_call>",
    ] {
        frames.extend(sse.push_delta(parser.push_str(chunk)));
    }
    let (tail, turn) = parser.finish();
    frames.extend(sse.push_delta(tail));
    let (end_frames, final_resp, _) = sse.finish(&turn, "tool_calls", 20, 10, 15, 0);
    assert!(
        !end_frames
            .iter()
            .any(|(n, _)| n == "response.output_item.added"),
        "finish() emitted duplicate output_item.added: {end_frames:?}"
    );
    frames.extend(end_frames);

    let fc_added = frames
        .iter()
        .filter(|(n, v)| n == "response.output_item.added" && v["item"]["type"] == "function_call")
        .count();
    assert_eq!(fc_added, 1);

    let joined_deltas: String = frames
        .iter()
        .filter(|(n, _)| n == "response.function_call_arguments.delta")
        .filter_map(|(_, v)| v["delta"].as_str())
        .collect();
    let expected = r#"{"path":"src/scene.js","content":"const a = 1;\nconst b = 2;"}"#;
    assert_eq!(joined_deltas, expected);
    assert_eq!(final_resp["output"].as_array().unwrap().len(), 1);
    assert_eq!(final_resp["output"][0]["arguments"], expected);

    // Multi-tool chunk handoff (tool 0 finishes + tool 1 starts in one chunk, then tool 1
    // finishes + trailing text in the next chunk) + truncated tool call at finish().
    let (_, meta2) = translate_responses_to_chat(&req, "qwen-test", &store, None, 9).unwrap();
    let mut parser2 = Qwen3Parser::with_tools(false, &tools).unwrap();
    let mut sse2 = ResponsesSseFormatter::new(meta2);
    let mut frames2 = sse2.start();
    frames2.extend(sse2.push_delta(parser2.push_str(concat!(
        "<tool_call>\n<function=write>\n<parameter=path>a.js</parameter>\n</function>\n</tool_call>\n",
        "<tool_call>\n<function=write>\n<parameter=path>b."
    ))));
    frames2.extend(sse2.push_delta(
        parser2.push_str("js</parameter>\n</function>\n</tool_call>\nAll files written."),
    ));
    let (tail2, turn2) = parser2.finish();
    frames2.extend(sse2.push_delta(tail2));
    let (end2, resp2, _) = sse2.finish(&turn2, "stop", 20, 10, 15, 0);
    frames2.extend(end2);

    let out_items2 = resp2["output"].as_array().unwrap();
    assert_eq!(out_items2.len(), 3);
    assert_eq!(out_items2[0]["type"], "function_call");
    assert_eq!(out_items2[0]["arguments"], r#"{"path":"a.js"}"#);
    assert_eq!(out_items2[1]["type"], "function_call");
    assert_eq!(out_items2[1]["arguments"], r#"{"path":"b.js"}"#);
    assert_eq!(out_items2[2]["type"], "message");
    assert_eq!(out_items2[2]["content"][0]["text"], "All files written.");

    // Truncated tool call at max_tokens marks the function_call item as incomplete.
    let (_, meta3) = translate_responses_to_chat(&req, "qwen-test", &store, None, 10).unwrap();
    let mut parser3 = Qwen3Parser::with_tools(false, &tools).unwrap();
    let mut sse3 = ResponsesSseFormatter::new(meta3);
    let mut frames3 = sse3.start();
    frames3.extend(
        sse3.push_delta(parser3.push_str("<tool_call>\n<function=write>\n<parameter=path>trunc")),
    );
    let (tail3, turn3) = parser3.finish();
    frames3.extend(sse3.push_delta(tail3));
    let (end3, resp3, _) = sse3.finish(&turn3, "length", 20, 10, 15, 0);
    frames3.extend(end3);
    assert!(
        !frames3
            .iter()
            .any(|(n, _)| n == "response.function_call_arguments.done"),
        "truncated tool call must not emit function_call_arguments.done"
    );
    assert_eq!(resp3["status"], "incomplete");
    assert_eq!(resp3["output"][0]["type"], "function_call");
    assert_eq!(resp3["output"][0]["status"], "incomplete");
}

#[test]
fn response_store_cancellation_deletion_and_lru_limits() {
    let store = ResponseStore::with_capacity(3);
    let flag = Arc::new(AtomicBool::new(false));
    store.register_active("resp_active_1".into(), "tenant_x".into(), flag.clone());

    // Cancel with wrong scope returns None, flag unchanged
    assert!(store.cancel("resp_active_1", "tenant_y").is_none());
    assert!(!flag.load(Ordering::Acquire));

    // Cancel with matching scope sets flag and returns cancelled response
    let canc = store.cancel("resp_active_1", "tenant_x").unwrap();
    assert_eq!(canc["status"], "cancelled");
    assert!(flag.load(Ordering::Acquire));

    // Delete active/nonexistent
    assert!(!store.delete("resp_active_1", "tenant_x"));

    // Store responses with in_progress status and test cancel
    store.insert(StoredResponse {
        id: "resp_prog".into(),
        owner_scope: "tenant_x".into(),
        response: json!({"id": "resp_prog", "status": "in_progress"}),
        history_messages: vec![json!({"role": "user", "content": "hi"})],
    });
    let canc_stored = store.cancel("resp_prog", "tenant_x").unwrap();
    assert_eq!(canc_stored["status"], "cancelled");

    // Delete stored response
    assert!(store.delete("resp_prog", "tenant_x"));
    assert!(!store.delete("resp_prog", "tenant_x"));

    // LRU eviction when hitting capacity: insert 4 items
    for i in 1..=4 {
        store.insert(StoredResponse {
            id: format!("resp_lru_{i}"),
            owner_scope: "tenant_x".into(),
            response: json!({"id": format!("resp_lru_{i}"), "status": "completed"}),
            history_messages: Vec::new(),
        });
    }
    // First item (resp_lru_1) should have been evicted
    assert!(store.get_response("resp_lru_1", "tenant_x").is_none());
    assert!(store.get_response("resp_lru_4", "tenant_x").is_some());
}

#[test]
fn completed_response_cannot_be_cancelled_while_terminal_frames_are_pending() {
    let store = ResponseStore::default();
    let flag = Arc::new(AtomicBool::new(false));
    store.register_active("resp_done".into(), "owner".into(), flag.clone());
    let completed = json!({"id": "resp_done", "status": "completed", "output": []});
    store.insert(StoredResponse {
        id: "resp_done".into(),
        owner_scope: "owner".into(),
        response: completed.clone(),
        history_messages: Vec::new(),
    });
    // The HTTP stream still owns its guard while terminal frames drain.
    assert_eq!(store.cancel("resp_done", "owner"), Some(completed.clone()));
    assert_eq!(store.get_response("resp_done", "owner"), Some(completed));
    assert!(!flag.load(Ordering::Acquire));
    assert!(store.cancel("resp_done", "other").is_none());
}

#[test]
fn accepted_cancellation_wins_over_late_completion_publication() {
    let store = ResponseStore::default();
    let flag = Arc::new(AtomicBool::new(false));
    store.register_active("resp_cancel".into(), "owner".into(), flag.clone());
    let cancelled = store.cancel("resp_cancel", "owner").unwrap();
    store.insert(StoredResponse {
        id: "resp_cancel".into(),
        owner_scope: "owner".into(),
        response: json!({"id": "resp_cancel", "status": "completed"}),
        history_messages: vec![json!({"role": "assistant", "content": "partial"})],
    });
    assert_eq!(
        store.get_response("resp_cancel", "owner"),
        Some(cancelled.clone())
    );
    assert_eq!(store.cancel("resp_cancel", "owner"), Some(cancelled));
    assert!(flag.load(Ordering::Acquire));
    assert_eq!(store.get_history("resp_cancel", "owner"), Some(Vec::new()));
}

#[test]
fn cancellation_and_publication_race_has_one_terminal_outcome() {
    for _ in 0..64 {
        let store = ResponseStore::default();
        store.register_active(
            "resp_race".into(),
            "owner".into(),
            Arc::new(AtomicBool::new(false)),
        );
        let barrier = std::sync::Barrier::new(2);
        let (cancelled, published) = std::thread::scope(|threads| {
            let cancel = threads.spawn(|| {
                barrier.wait();
                store.cancel("resp_race", "owner").unwrap()
            });
            barrier.wait();
            let published = store.finalize(
                StoredResponse {
                    id: "resp_race".into(),
                    owner_scope: "owner".into(),
                    response: json!({"id": "resp_race", "status": "completed"}),
                    history_messages: Vec::new(),
                },
                true,
            );
            (cancel.join().unwrap(), published)
        });
        assert_eq!(cancelled, published);
        assert_eq!(store.get_response("resp_race", "owner"), Some(published));
    }
}

#[test]
fn transport_cancellation_is_not_explicit_response_cancellation() {
    let store = ResponseStore::default();
    let transport_flag = Arc::new(AtomicBool::new(true));
    store.register_active("resp_failed".into(), "owner".into(), transport_flag);
    let failed = store.finalize(
        StoredResponse {
            id: "resp_failed".into(),
            owner_scope: "owner".into(),
            response: json!({"status": "failed"}),
            history_messages: Vec::new(),
        },
        true,
    );
    assert_eq!(failed["status"], "failed");
    assert!(store.cancel("resp_failed", "owner").is_none());
    assert!(store.get_response("resp_failed", "owner").is_none());
}

#[test]
fn unstored_cancellation_is_terminal_without_retaining_history() {
    let store = ResponseStore::default();
    store.register_active(
        "resp_private".into(),
        "owner".into(),
        Arc::new(AtomicBool::new(false)),
    );
    let cancelled = store.cancel("resp_private", "owner").unwrap();
    assert_eq!(
        store.cancel("resp_private", "owner"),
        Some(cancelled.clone())
    );
    let published = store.finalize(
        StoredResponse {
            id: "resp_private".into(),
            owner_scope: "owner".into(),
            response: json!({"status": "failed"}),
            history_messages: Vec::new(),
        },
        false,
    );
    assert_eq!(published, cancelled);
    assert!(store.get_response("resp_private", "owner").is_none());
    assert!(store.cancel("resp_private", "owner").is_none());
}

#[test]
fn cancelled_stream_uses_winning_response_and_contiguous_sequence_numbers() {
    let store = ResponseStore::default();
    let (_, meta) = translate_responses_to_chat(
        &json!({"input": "hi", "store": true}),
        "q",
        &store,
        Some("owner"),
        1,
    )
    .unwrap();
    store.register_active(
        meta.response_id.clone(),
        "owner".into(),
        Arc::new(AtomicBool::new(false)),
    );
    let mut formatter = ResponsesSseFormatter::new(meta.clone());
    let mut frames = formatter.start();
    let cancelled = store.cancel(&meta.response_id, "owner").unwrap();
    let final_response = store.finalize(
        StoredResponse {
            id: meta.response_id.clone(),
            owner_scope: "owner".into(),
            response: formatter.failure_response("channel closed"),
            history_messages: Vec::new(),
        },
        true,
    );
    frames.push(formatter.terminal_response(final_response));
    for (sequence, (event, payload)) in frames.iter().enumerate() {
        assert_eq!(payload["sequence_number"], sequence);
        assert_eq!(payload["type"], event.as_str());
    }
    assert_eq!(frames.last().unwrap().0, "response.cancelled");
    assert_eq!(frames.last().unwrap().1["response"], cancelled);
    assert_eq!(
        store.get_response(&meta.response_id, "owner"),
        Some(cancelled)
    );
}

#[test]
fn responses_api_error_helpers_and_id_uniqueness() {
    let e1 = ResponsesApiError::bad_request("invalid field");
    let j1 = e1.to_openai_json();
    assert_eq!(j1["error"]["type"], "invalid_request_error");
    assert_eq!(j1["error"]["message"], "invalid field");

    let e2 = ResponsesApiError::not_found("not found id");
    let j2 = e2.to_openai_json();
    assert_eq!(j2["error"]["type"], "not_found_error");

    let mut ids = std::collections::HashSet::new();
    for i in 0..20 {
        let id = generate_unguessable_response_id(i);
        assert!(id.starts_with("resp_"));
        assert_eq!(id.len(), 37);
        assert!(ids.insert(id));
    }
}

#[test]
fn validates_all_responses_bad_request_paths() {
    let store = ResponseStore::default();
    let bad_cases = [
        json!([]),
        json!({"model": "q"}),
        json!({"input": []}),
        json!({"input": [123]}),
        json!({"input": [{"type": "message", "content": "hi"}]}),
        json!({"input": [{"type": "message", "role": "user"}]}),
        json!({"input": [{"type": "message", "role": "admin", "content": "hi"}]}),
        json!({"input": [{"type": "message", "role": "user", "content": [123]}]}),
        json!({"input": [{"type": "message", "role": "user", "content": [{"type": "text"}]}]}),
        json!({"input": [{"type": "message", "role": "user", "content": [{"type": "input_image"}]}]}),
        json!({"input": [{"type": "message", "role": "system", "content": [{"type": "input_image", "image_url": "https://a.png"}]}]}),
        json!({"input": [{"type": "message", "role": "user", "content": [{"type": "video"}]}]}),
        json!({"input": [{"type": "function_call", "arguments": "{}"}]}),
        json!({"input": [{"type": "function_call", "name": "foo"}]}),
        json!({"input": [{"type": "function_call", "name": "foo", "arguments": "not json"}]}),
        json!({"input": [{"type": "function_call_output"}]}),
        json!({"input": [{"type": "unsupported"}]}),
        json!({"input": "hi", "tools": "tools"}),
        json!({"input": "hi", "tools": [1]}),
        json!({"input": "hi", "tools": [{"type": "code_interpreter"}]}),
        json!({"input": "hi", "tool_choice": "maybe"}),
        json!({"input": "hi", "tool_choice": {"type": "function"}}),
        json!({"input": "hi", "reasoning": {"effort": "extreme"}}),
        json!({"input": "hi", "text": {"format": "yaml"}}),
        json!({"input": "hi", "temperature": "high"}),
        json!({"input": "hi", "top_p": "high"}),
        json!({"input": "hi", "max_output_tokens": 0}),
        json!({"input": "hi", "max_output_tokens": crate::server::MAX_OUTPUT_TOKENS + 1}),
    ];
    for case in &bad_cases {
        assert!(
            translate_responses_to_chat(case, "q", &store, None, 1).is_err(),
            "expected bad-request error for payload: {case}"
        );
    }
    let (large_chat, large_meta) = translate_responses_to_chat(
        &json!({"input": "hi", "max_output_tokens": 16_384}),
        "q",
        &store,
        None,
        1,
    )
    .unwrap();
    assert_eq!(large_chat["max_tokens"], 16_384);
    assert_eq!(large_meta.max_output_tokens, 16_384);

    // Omitted max_output_tokens leaves chat_req["max_tokens"] unset so prepare() resolves
    // the remaining context budget after prompt tokenization.
    let (omitted_chat, _) =
        translate_responses_to_chat(&json!({"input": "hi"}), "q", &store, None, 2).unwrap();
    assert!(omitted_chat.get("max_tokens").is_none());
}

#[test]
fn responses_sse_fail_and_incomplete_lifecycle() {
    let store = ResponseStore::default();
    let req = json!({"input": "stream test"});
    let (_, meta) = translate_responses_to_chat(&req, "qwen", &store, None, 1).unwrap();

    // Failure event
    let mut sse1 = ResponsesSseFormatter::new(meta.clone());
    let _ = sse1.start();
    let fail_events = sse1.fail("pipeline broken");
    assert_eq!(fail_events.len(), 1);
    assert_eq!(fail_events[0].0, "response.failed");
    assert_eq!(fail_events[0].1["response"]["status"], "failed");
    assert_eq!(
        fail_events[0].1["response"]["error"]["message"],
        "pipeline broken"
    );

    // Incomplete event due to length limit
    let mut sse2 = ResponsesSseFormatter::new(meta);
    let _ = sse2.start();
    let turn = ParsedAssistantTurn {
        reasoning_content: None,
        content: Some("cut off midway".into()),
        tool_calls: Vec::new(),
    };
    let (events, final_resp, _) = sse2.finish(&turn, "length", 10, 0, 5, 0);
    assert_eq!(events.last().unwrap().0, "response.incomplete");
    assert_eq!(final_resp["status"], "incomplete");
    assert_eq!(
        final_resp["incomplete_details"]["reason"],
        "max_output_tokens"
    );
}

#[test]
fn parallel_tool_calls_survives_responses_translation() {
    for parallel in [false, true] {
        let request = json!({"input":"ping", "parallel_tool_calls":parallel,
            "tools":[{"type":"function","name":"ping","parameters":{"type":"object"}}]});
        let (chat, meta) = translate_responses_to_chat(
            &request,
            "test",
            &ResponseStore::with_capacity(8),
            None,
            1,
        )
        .unwrap();
        assert_eq!(meta.parallel_tool_calls, parallel);
        assert_eq!(chat["parallel_tool_calls"], json!(parallel));
        assert!(
            matches!(RequestConstraint::parse(&chat).unwrap(), Some(RequestConstraint::Tools { parallel: p, .. }) if p == parallel)
        );
    }
}

#[test]
fn namespace_and_web_search_tools_roundtrip_for_codex() {
    let request = json!({
        "input": [
            {"role": "user", "content": "Spawn an agent"},
            {
                "type": "function_call",
                "call_id": "call_prev",
                "namespace": "multi_agent_v1",
                "name": "spawn_agent",
                "arguments": "{\"message\":\"hi\"}"
            },
            {
                "type": "function_call_output",
                "call_id": "call_prev",
                "output": "ok"
            }
        ],
        "tools": [
            {"type": "function", "name": "exec_command", "parameters": {"type": "object"}},
            {
                "type": "namespace",
                "name": "multi_agent_v1",
                "description": "Sub-agent tools.",
                "tools": [
                    {
                        "type": "function",
                        "name": "spawn_agent",
                        "description": "Spawn a sub-agent",
                        "parameters": {"type": "object", "properties": {"message": {"type": "string"}}}
                    }
                ]
            },
            {"type": "web_search"}
        ]
    });
    let (chat, meta) =
        translate_responses_to_chat(&request, "test", &ResponseStore::with_capacity(8), None, 1)
            .expect("Codex namespace + web_search tools must translate cleanly");

    let chat_tools = chat["tools"].as_array().unwrap();
    assert_eq!(
        chat_tools.len(),
        2,
        "web_search is skipped, namespace flattened"
    );
    assert_eq!(chat_tools[0]["function"]["name"], "exec_command");
    assert_eq!(
        chat_tools[1]["function"]["name"],
        "multi_agent_v1__spawn_agent"
    );
    // Input history function_call with namespace maps to the same qualified model name.
    assert_eq!(
        chat["messages"][1]["tool_calls"][0]["function"]["name"],
        "multi_agent_v1__spawn_agent"
    );

    // Non-streaming output restores wire `name` + `namespace`.
    let turn = ParsedAssistantTurn {
        reasoning_content: None,
        content: None,
        tool_calls: vec![ParsedToolCall {
            id: "call_1".into(),
            call_type: "function".into(),
            function: crate::parser::ParsedFunctionCall {
                name: "multi_agent_v1__spawn_agent".into(),
                arguments: "{\"message\":\"go\"}".into(),
            },
        }],
    };
    let (resp, _) = build_response_object(&meta, &turn, "tool_calls", 10, 0, 5, 0);
    assert_eq!(resp["output"][0]["type"], "function_call");
    assert_eq!(resp["output"][0]["name"], "spawn_agent");
    assert_eq!(resp["output"][0]["namespace"], "multi_agent_v1");

    // Streaming SSE output also restores wire `name` + `namespace`.
    let mut sse = ResponsesSseFormatter::new(meta);
    let _ = sse.start();
    let evs = sse.push_delta(crate::parser::StreamParseDelta {
        completed_tool_calls: turn.tool_calls.clone(),
        ..Default::default()
    });
    let added = evs
        .iter()
        .find(|(ev, _)| ev == "response.output_item.added")
        .unwrap();
    assert_eq!(added.1["item"]["name"], "spawn_agent");
    assert_eq!(added.1["item"]["namespace"], "multi_agent_v1");
    let done = evs
        .iter()
        .find(|(ev, _)| ev == "response.output_item.done")
        .unwrap();
    assert_eq!(done.1["item"]["name"], "spawn_agent");
    assert_eq!(done.1["item"]["namespace"], "multi_agent_v1");
}

#[test]
fn streaming_reasoning_uses_content_part_events_matching_vllm_034() {
    let body = json!({
        "model": "qwen3.8-27b",
        "input": "hello",
        "stream": true
    });
    let store = ResponseStore::with_capacity(16);
    let (_, meta) = translate_responses_to_chat(&body, "qwen3.8-27b", &store, None, 1).unwrap();
    let mut sse = ResponsesSseFormatter::new(meta);
    let _ = sse.start();
    let mut all = sse.push_delta(crate::parser::StreamParseDelta {
        reasoning_delta: Some("thinking".into()),
        ..Default::default()
    });
    all.extend(sse.push_delta(crate::parser::StreamParseDelta {
        content_delta: Some("done".into()),
        ..Default::default()
    }));
    let names: Vec<&str> = all.iter().map(|(ev, _)| ev.as_str()).collect();
    assert!(!names.contains(&"response.reasoning_part.added"));
    assert!(!names.contains(&"response.reasoning_part.done"));
    let added = all
        .iter()
        .find(|(ev, data)| {
            ev == "response.content_part.added" && data["part"]["type"] == "reasoning_text"
        })
        .expect("reasoning content_part.added");
    assert_eq!(added.1["part"]["text"], "");
    let done = all
        .iter()
        .find(|(ev, data)| {
            ev == "response.content_part.done" && data["part"]["type"] == "reasoning_text"
        })
        .expect("reasoning content_part.done");
    assert_eq!(done.1["part"]["text"], "thinking");
}
