#![cfg(feature = "cuda")]
use serde_json::{Value, json};
use std::sync::{Arc, Mutex, atomic::Ordering};
mod common;
use w34_host::{
    cache::KvFormatPolicy,
    model::Qwen3_5Engine,
    proposer::{SpeculativeConfig, SpeculativeKind},
    server::OpenAiServerState,
};

static GPU_LOCK: Mutex<()> = Mutex::new(());

fn request_endpoint(
    state: &Arc<OpenAiServerState>,
    endpoint: &str,
    body: Value,
) -> (String, Value) {
    common::request(state, "POST", endpoint, body)
}

fn request(state: &Arc<OpenAiServerState>, body: Value) -> (String, Value) {
    request_endpoint(state, "/v1/chat/completions", body)
}

fn exercise_tool_choices(state: &Arc<OpenAiServerState>, kind: SpeculativeKind, tools: &Value) {
    for choice in [
        json!("required"),
        json!({"type":"function","function":{"name":"report"}}),
        json!("auto"),
        json!("none"),
    ] {
        let (head, body) = request(
            state,
            json!({"messages":[{"role":"user","content":"Call report for Oslo."}],
            "tools":tools, "tool_choice":choice, "max_tokens":192, "chat_template_kwargs":{"enable_thinking":false}}),
        );
        assert!(head.contains("200 OK"), "{choice}: {head} {body}");
        let calls = body["choices"][0]["message"]["tool_calls"]
            .as_array()
            .unwrap();
        if choice == "none" {
            assert!(calls.is_empty(), "{body}");
        }
        if choice == "required" || choice.is_object() {
            assert!(!calls.is_empty(), "{body}");
        }
        for call in calls {
            let name = call["function"]["name"].as_str().unwrap();
            let args: Value =
                serde_json::from_str(call["function"]["arguments"].as_str().unwrap()).unwrap();
            match name {
                "report" => assert_eq!(args, json!({"city":"Oslo"})),
                "noop" if !choice.is_object() => assert_eq!(args, json!({})),
                _ => panic!("unexpected tool {call}"),
            }
        }
        println!(
            "{kind:?}: tool_choice={choice}, schema-valid calls={}",
            calls.len()
        );
    }
}

fn verify_speculative_telemetry(state: &OpenAiServerState, kind: SpeculativeKind, accepted: u64) {
    let telemetry = state.metrics.snapshot();
    assert_eq!(telemetry.speculation.accepted_tokens, accepted);
    let drafts = telemetry
        .speculation
        .draft_tokens
        .expect("actual verified drafts");
    assert!(drafts >= accepted);
    assert!(
        drafts > 0,
        "speculation must publish the actual proposal denominator"
    );
    assert_eq!(
        telemetry.speculation.rounds,
        state.worker.stats.verify_rounds.load(Ordering::Relaxed)
    );
    assert_eq!(telemetry.requests.submitted, 1);
    assert_eq!(telemetry.latency.ttft.count, 1);
    assert_eq!(
        telemetry.latency.time_per_output_token.count, 1,
        "one request-level observation, not speculative token drains"
    );
    assert!(telemetry.tokens.generated > 0);
    assert_eq!(
        telemetry.config.speculation,
        if kind == SpeculativeKind::Mtp {
            "mtp"
        } else {
            "dflash2"
        }
    );
}

fn exercise_completions_and_hermes_tools(state: &Arc<OpenAiServerState>) {
    let (head, body) = request_endpoint(
        state,
        "/v1/completions",
        json!({"prompt":"Result: ","max_tokens":16,"structured_outputs":{"grammar":"start: \"PASS\" | \"OK\""}}),
    );
    assert!(head.contains("200 OK"), "{head} {body}");
    let gtext = body["choices"][0]["text"].as_str().unwrap();
    assert!(
        matches!(gtext, "PASS" | "OK"),
        "unexpected grammar completion: {gtext:?}"
    );

    let (head, body) = request_endpoint(
        state,
        "/v1/completions",
        json!({"prompt":"Answer code: ","max_tokens":16,"structured_outputs":{"regex":"OK-[0-9]{2}"}}),
    );
    assert!(head.contains("200 OK"), "{head} {body}");
    let text = body["choices"][0]["text"].as_str().unwrap();
    assert!(
        text.starts_with("OK-") && text.len() == 5 && text[3..].chars().all(|c| c.is_ascii_digit()),
        "unexpected regex completion: {text}"
    );

    let (head, body) = request_endpoint(
        state,
        "/v1/completions",
        json!({"prompt":"Verdict: ","max_tokens":16,"structured_outputs":{"choice":["APPROVED","REJECTED"]}}),
    );
    assert!(head.contains("200 OK"), "{head} {body}");
    let ctext = body["choices"][0]["text"].as_str().unwrap();
    assert!(
        matches!(ctext, "APPROVED" | "REJECTED"),
        "unexpected choice completion: {ctext:?}"
    );

    let hermes_tools = json!([{
        "type":"function",
        "function":{
            "name":"dispatch",
            "parameters":{
                "type":"object",
                "anyOf":[
                    {"properties":{"mode":{"const":"fast"}},"required":["mode"],"additionalProperties":false}
                ]
            }
        }
    }]);
    let (head, body) = request(
        state,
        json!({
            "messages":[{"role":"user","content":"Dispatch in fast mode."}],
            "tools":hermes_tools,
            "tool_choice":"required",
            "parallel_tool_calls":false,
            "max_tokens":96,
            "chat_template_kwargs":{"enable_thinking":false,"tool_call_format":"json"}
        }),
    );
    assert!(head.contains("200 OK"), "{head} {body}");
    let hcalls = body["choices"][0]["message"]["tool_calls"]
        .as_array()
        .unwrap();
    assert_eq!(hcalls.len(), 1, "{body}");
    assert_eq!(hcalls[0]["function"]["name"], "dispatch");
    assert_eq!(
        serde_json::from_str::<Value>(hcalls[0]["function"]["arguments"].as_str().unwrap())
            .unwrap(),
        json!({"mode":"fast"})
    );
}

fn exercise(kind: SpeculativeKind) {
    if common::require_or_skip_cached_model("nchapman/Qwen3.5-4B-W4A16-GPTQ", "exercise").is_none()
        || (kind == SpeculativeKind::DFlash2
            && common::require_or_skip_cached_model("nchapman/Qwen3.5-4B-DFlash2", "exercise")
                .is_none())
    {
        return;
    }
    let mut engine = Qwen3_5Engine::load(
        "nchapman/Qwen3.5-4B-W4A16-GPTQ",
        1,
        KvFormatPolicy::Tq4,
        Some(64),
        true,
    )
    .unwrap();
    engine
        .enable_speculative(&SpeculativeConfig {
            kind,
            draft_model: if kind == SpeculativeKind::DFlash2 {
                Some("nchapman/Qwen3.5-4B-DFlash2".into())
            } else {
                None
            },
            mtp_k: 3,
        })
        .unwrap();
    let state = OpenAiServerState::new("constraint-test".into(), engine);
    let (head, body) = request(
        &state,
        json!({
            "messages":[{"role":"user","content":"Say hello."}],
            "chat_template_kwargs":{"enable_thinking":false}, "max_tokens":64,
            "response_format":{"type":"json_schema","json_schema":{"name":"result","strict":true,
                "schema":{"type":"object","properties":{"status":{"const":"schema-enforced"}},"required":["status"],"additionalProperties":false}}}
        }),
    );
    assert!(head.contains("200 OK"), "{head} {body}");
    let content = body["choices"][0]["message"]["content"].as_str().unwrap();
    assert_eq!(
        serde_json::from_str::<Value>(content).unwrap(),
        json!({"status":"schema-enforced"})
    );
    assert!(
        state.worker.stats.verify_rounds.load(Ordering::Relaxed) > 0,
        "speculative verifier did not run"
    );
    let accepted = state.worker.stats.accepted_drafts.load(Ordering::Relaxed);
    verify_speculative_telemetry(&state, kind, accepted);
    println!("{kind:?}: strict JSON schema enforced (accepted drafts={accepted})");

    let tools = json!([
        {"type":"function","function":{"name":"report","parameters":{"type":"object","properties":{"city":{"const":"Oslo"}},"required":["city"],"additionalProperties":false}}},
        {"type":"function","function":{"name":"noop","parameters":{"type":"object","properties":{},"additionalProperties":false}}}
    ]);
    exercise_tool_choices(&state, kind, &tools);
    let (head, body) = request(
        &state,
        json!({
            "messages":[
                {"role":"system","content":"Reasoning effort is set to low. Keep your thinking brief and focused, moving directly to the conclusion without unnecessary elaboration."},
                {"role":"user","content":"What is 1+1?"}
            ],
            "max_tokens":512,
            "chat_template_kwargs":{"enable_thinking":true},
            "response_format":{"type":"json_schema","json_schema":{"name":"sum","strict":true,
                "schema":{"type":"object","properties":{"sum":{"const":2}},"required":["sum"],"additionalProperties":false}}}
        }),
    );
    assert!(head.contains("200 OK"), "{head} {body}");
    let reasoning = body["choices"][0]["message"]["reasoning_content"]
        .as_str()
        .unwrap_or("");
    assert!(
        !reasoning.trim().is_empty(),
        "missing reasoning before </think>: {body}"
    );
    let content = body["choices"][0]["message"]["content"].as_str().unwrap();
    assert_eq!(
        serde_json::from_str::<Value>(content).unwrap(),
        json!({"sum": 2}),
        "{body}"
    );
    println!(
        "{kind:?}: think-aware json_schema enforced (reasoning_chars={}, content={content})",
        reasoning.len()
    );

    // Multi-turn system + tools + assistant.tool_calls + role:tool <tool_response> -> constrained answer
    let (head, body) = request(
        &state,
        json!({
            "messages":[
                {"role":"system","content":"Use tool results accurately."},
                {"role":"user","content":"Report weather for Oslo."},
                {"role":"assistant","content":null,"tool_calls":[
                    {"id":"call_1","type":"function","function":{"name":"report","arguments":"{\"city\":\"Oslo\"}"}}
                ]},
                {"role":"tool","tool_call_id":"call_1","content":"{\"temp_c\":7}"}
            ],
            "tools":tools,
            "tool_choice":"none",
            "max_tokens":64,
            "chat_template_kwargs":{"enable_thinking":false}
        }),
    );
    assert!(head.contains("200 OK"), "{head} {body}");
    assert!(
        body["choices"][0]["message"]["tool_calls"]
            .as_array()
            .unwrap()
            .is_empty(),
        "{body}"
    );

    // Structural tag + /v1/completions json_object, grammar, & regex
    let (head, body) = request(
        &state,
        json!({
            "messages":[{"role":"user","content":"Reply verbatim with <ans>{\"ok\":false}</ans> and nothing else."}],
            "max_tokens":64,
            "chat_template_kwargs":{"enable_thinking":false},
            "response_format":{
                "type":"structural_tag",
                "structures":[{"begin":"<ans>","end":"</ans>","schema":{"type":"object","properties":{"ok":{"const":true}},"required":["ok"],"additionalProperties":false}}],
                "triggers":["<ans>"]
            }
        }),
    );
    assert!(head.contains("200 OK"), "{head} {body}");
    let tagged = body["choices"][0]["message"]["content"].as_str().unwrap();
    let (_, rest) = tagged
        .split_once("<ans>")
        .unwrap_or_else(|| panic!("expected triggered <ans> tag in {tagged:?}"));
    let (payload, _) = rest.split_once("</ans>").expect("closed </ans> tag");
    assert_eq!(
        serde_json::from_str::<Value>(payload).unwrap(),
        json!({"ok":true})
    );

    let (head, body) = request(
        &state,
        json!({
            "messages":[{"role":"user","content":"Say hello in plain English."}],
            "max_tokens":64,
            "chat_template_kwargs":{"enable_thinking":false},
            "response_format":{"type":"json_object"}
        }),
    );
    assert!(head.contains("200 OK"), "{head} {body}");
    let obj_str = body["choices"][0]["message"]["content"].as_str().unwrap();
    assert!(
        serde_json::from_str::<Value>(obj_str).unwrap().is_object(),
        "expected JSON object, got {obj_str:?}"
    );

    exercise_completions_and_hermes_tools(&state);

    for bad in [
        json!({"messages":[{"role":"user","content":"hello"}], "response_format":{"type":"unknown"}}),
        json!({"messages":[{"role":"user","content":"hello"}], "response_format":{"type":"json_schema","json_schema":{"strict":false,"schema":{"type":"object"}}}}),
        json!({"messages":[{"role":"user","content":"hello"}], "response_format":{"type":"json_object"}, "structured_outputs":{"regex":"a"}}),
        json!({"messages":[{"role":"user","content":"hello"}], "structured_outputs":{"regex":"a".repeat(70_000)}}),
    ] {
        let (head, _) = request(&state, bad);
        assert!(head.contains("400 Bad Request"), "{head}");
    }

    let (head, _) = request(
        &state,
        json!({
            "messages":[{"role":"user","content":"Say hi."}],
            "max_tokens":8,
            "chat_template_kwargs":{"enable_thinking":false},
            "tools":null,
            "tool_choice":null,
            "response_format":null,
            "structured_outputs":null
        }),
    );
    assert!(head.contains("200 OK"), "{head}");

    let ref_tools = json!([{"type":"function","function":{"name":"ext","parameters":{"$ref":"https://example.invalid/schema"}}}]);
    let (head, _) = request(
        &state,
        json!({
            "messages":[{"role":"user","content":"Say hi."}],
            "max_tokens":8,
            "chat_template_kwargs":{"enable_thinking":false},
            "tools":ref_tools,
            "tool_choice":"auto"
        }),
    );
    assert!(head.contains("200 OK"), "{head}");
    let (head, _) = request(
        &state,
        json!({
            "messages":[{"role":"user","content":"Say hi."}],
            "max_tokens":8,
            "chat_template_kwargs":{"enable_thinking":false},
            "tools":ref_tools,
            "tool_choice":"required"
        }),
    );
    assert!(head.contains("400 Bad Request"), "{head}");
    common::shutdown(&state);
}

#[test]
fn non_speculative_and_multimodal_mandatory_constraints_are_enforced() {
    if common::require_or_skip_cached_model(
        "nchapman/Qwen3.5-4B-W4A16-GPTQ",
        "non_speculative_and_multimodal_mandatory_constraints_are_enforced",
    )
    .is_none()
    {
        return;
    }
    let _guard = GPU_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    let engine = Qwen3_5Engine::load(
        "nchapman/Qwen3.5-4B-W4A16-GPTQ",
        1,
        KvFormatPolicy::Tq4,
        Some(64),
        true,
    )
    .unwrap();
    let state = OpenAiServerState::new("plain-serve-test".into(), engine);
    let tools = json!([{"type":"function","function":{"name":"ping","parameters":{"type":"object","properties":{},"additionalProperties":false}}}]);

    for msgs in [
        json!([{"role":"user","content":"Say ok."}]),
        json!([{"role":"user","content":[{"type":"image_url","image_url":{"url":"rgb://sample"}},{"type":"text","text":"Say ok."}]}]),
    ] {
        let (head, body) = request(
            &state,
            json!({
                "messages":msgs,
                "tools":tools,
                "tool_choice":"none",
                "max_tokens":8,
                "chat_template_kwargs":{"enable_thinking":false}
            }),
        );
        assert!(head.contains("200 OK"), "{head} {body}");
    }
    let (head, body) = request(
        &state,
        json!({
            "messages":[{"role":"user","content":"Say ok."}],
            "tools":[],
            "tool_choice":"auto",
            "max_tokens":8,
            "chat_template_kwargs":{"enable_thinking":false}
        }),
    );
    assert!(head.contains("200 OK"), "{head} {body}");

    for messages in [
        json!([{"role":"user","content":"Say hello."}]),
        json!([{"role":"user","content":[{"type":"image_url","image_url":{"url":"rgb://sample"}},{"type":"text","text":"Say hello."}]}]),
    ] {
        let (head, body) = request(
            &state,
            json!({
                "messages":messages, "max_tokens":64, "chat_template_kwargs":{"enable_thinking":false},
                "response_format":{"type":"json_schema","json_schema":{"name":"result","strict":true,
                    "schema":{"type":"object","properties":{"ok":{"const":true}},"required":["ok"],"additionalProperties":false}}}
            }),
        );
        assert!(head.contains("200 OK"), "{head} {body}");
        assert_eq!(
            serde_json::from_str::<Value>(
                body["choices"][0]["message"]["content"].as_str().unwrap()
            )
            .unwrap(),
            json!({"ok":true})
        );
        let (head, body) = request(
            &state,
            json!({
                "messages":messages, "tools":tools, "tool_choice":"required", "max_tokens":128,
                "chat_template_kwargs":{"enable_thinking":false}
            }),
        );
        assert!(head.contains("200 OK"), "{head} {body}");
        let calls = body["choices"][0]["message"]["tool_calls"]
            .as_array()
            .unwrap();
        assert!(!calls.is_empty(), "{body}");
        for call in calls {
            assert_eq!(call["function"]["name"], "ping");
            assert_eq!(
                serde_json::from_str::<Value>(call["function"]["arguments"].as_str().unwrap())
                    .unwrap(),
                json!({})
            );
        }
    }
    common::shutdown(&state);
}

#[test]
fn mtp_http_strict_schema_is_enforced() {
    let _guard = GPU_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    exercise(SpeculativeKind::Mtp);
}

#[test]
fn dflash_http_strict_schema_is_enforced() {
    let _guard = GPU_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    exercise(SpeculativeKind::DFlash2);
}

#[test]
fn qwen3_6_moe_mtp_http_tool_and_schema_adherence() {
    let _guard = GPU_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    let model = "llmfan46/Qwen3.6-35B-A3B-uncensored-heretic-Native-MTP-Preserved-GPTQ-Int4";
    if !common::require_or_skip_gpus(2, "qwen3_6_moe_mtp_http_tool_and_schema_adherence")
        || common::require_or_skip_cached_model(
            model,
            "qwen3_6_moe_mtp_http_tool_and_schema_adherence",
        )
        .is_none()
    {
        return;
    }
    let mut engine = Qwen3_5Engine::load(model, 2, KvFormatPolicy::Tq4, Some(32), true).unwrap();
    engine
        .enable_speculative(&SpeculativeConfig {
            kind: SpeculativeKind::Mtp,
            draft_model: None,
            mtp_k: 3,
        })
        .unwrap();
    let state = OpenAiServerState::new("qwen3.6-moe-test".into(), engine);
    let tools = json!([{
        "type":"function",
        "function":{
            "name":"lookup_metric",
            "parameters":{
                "type":"object",
                "properties":{
                    "host":{"enum":["gpu0","gpu1"]},
                    "window_s":{"type":"integer","minimum":10,"maximum":60}
                },
                "required":["host","window_s"],
                "additionalProperties":false
            }
        }
    }]);
    for choice in [
        json!("required"),
        json!({"type":"function","function":{"name":"lookup_metric"}}),
    ] {
        let (head, body) = request(
            &state,
            json!({
                "messages":[{"role":"user","content":"Check gpu0 metrics over a 30 second window."}],
                "tools":tools,
                "tool_choice":choice,
                "max_tokens":128,
                "chat_template_kwargs":{"enable_thinking":false}
            }),
        );
        assert!(head.contains("200 OK"), "{choice}: {head} {body}");
        let calls = body["choices"][0]["message"]["tool_calls"]
            .as_array()
            .unwrap();
        assert!(!calls.is_empty(), "{body}");
        for call in calls {
            assert_eq!(call["function"]["name"], "lookup_metric");
            let args: Value =
                serde_json::from_str(call["function"]["arguments"].as_str().unwrap()).unwrap();
            let host = args["host"].as_str().unwrap();
            let win = args["window_s"].as_i64().unwrap();
            assert!(matches!(host, "gpu0" | "gpu1"), "{args}");
            assert!((10..=60).contains(&win), "{args}");
            assert_eq!(args.as_object().unwrap().len(), 2, "{args}");
        }
    }
    common::shutdown(&state);
}
