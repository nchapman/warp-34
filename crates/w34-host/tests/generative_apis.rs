#![cfg(feature = "cuda")]

mod common;
use serde_json::{Value, json};
use std::io::Cursor;
use w34_host::{
    cache::KvFormatPolicy,
    model::Qwen3_5Engine,
    server::{OpenAiServerState, serve},
};

static GPU_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

fn encode_base64(bytes: &[u8]) -> String {
    const TABLE: &[u8; 64] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    let mut out = String::with_capacity(bytes.len().div_ceil(3) * 4);
    for chunk in bytes.chunks(3) {
        let b0 = chunk[0] as u32;
        let b1 = if chunk.len() > 1 { chunk[1] as u32 } else { 0 };
        let b2 = if chunk.len() > 2 { chunk[2] as u32 } else { 0 };
        let triple = (b0 << 16) | (b1 << 8) | b2;
        out.push(TABLE[((triple >> 18) & 0x3F) as usize] as char);
        out.push(TABLE[((triple >> 12) & 0x3F) as usize] as char);
        out.push(if chunk.len() > 1 {
            TABLE[((triple >> 6) & 0x3F) as usize] as char
        } else {
            '='
        });
        out.push(if chunk.len() > 2 {
            TABLE[(triple & 0x3F) as usize] as char
        } else {
            '='
        });
    }
    out
}

fn tiny_red_png_base64() -> String {
    let img = image::RgbImage::from_pixel(32, 32, image::Rgb([230, 30, 30]));
    let mut buf = Cursor::new(Vec::new());
    img.write_to(&mut buf, image::ImageFormat::Png).unwrap();
    encode_base64(buf.get_ref())
}

fn parse_sse_frames(raw: &str) -> Vec<(String, Value)> {
    let mut frames = Vec::new();
    for block in raw.split("\n\n") {
        let mut ev = String::new();
        let mut data_lines = Vec::new();
        for line in block.lines() {
            if let Some(rest) = line.strip_prefix("event:") {
                ev = rest.trim().to_owned();
            } else if let Some(rest) = line.strip_prefix("data:") {
                data_lines.push(rest.trim());
            }
        }
        if !data_lines.is_empty() {
            let joined = data_lines.join("\n");
            if let Ok(val) = serde_json::from_str::<Value>(&joined) {
                frames.push((ev, val));
            }
        }
    }
    frames
}

#[test]
fn live_responses_anthropic_messages_and_token_count_endpoints() {
    if common::require_or_skip_cached_model(
        "nchapman/Qwen3.5-4B-W4A16-GPTQ",
        "live_responses_anthropic_messages_and_token_count_endpoints",
    )
    .is_none()
    {
        return;
    }
    let _gpu = GPU_LOCK.lock().unwrap();
    let state = OpenAiServerState::from_loader("Qwen3.5-4B-W4A16-GPTQ".into(), false, || {
        Qwen3_5Engine::load(
            "nchapman/Qwen3.5-4B-W4A16-GPTQ",
            1,
            KvFormatPolicy::Tq4,
            Some(256),
            true,
        )
        .map_err(|e| e.to_string())
    })
    .unwrap();

    tokio::runtime::Builder::new_multi_thread()
        .worker_threads(2)
        .enable_all()
        .build()
        .unwrap()
        .block_on(async {
            let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
            let addr = listener.local_addr().unwrap();
            let (stop_tx, stop_rx) = tokio::sync::oneshot::channel::<()>();
            let server_state = state.clone();
            let server_task = tokio::spawn(async move {
                serve(listener, server_state, async {
                    let _ = stop_rx.await;
                })
                .await
            });

            let base = format!("http://{addr}");
            let client = reqwest::Client::new();

            // 1. Token-counting agreement across Chat, Responses, and Anthropic Messages
            let chat_count: Value = client
                .post(format!("{base}/v1/chat/completions/input_tokens"))
                .json(&json!({
                    "messages": [
                        {"role": "system", "content": "Reply concisely."},
                        {"role": "user", "content": "What is 6 * 7?"}
                    ]
                }))
                .send()
                .await
                .unwrap()
                .json()
                .await
                .unwrap();
            assert_eq!(chat_count["object"], "response.input_tokens");
            let expected_tokens = chat_count["input_tokens"].as_u64().unwrap();
            assert!(expected_tokens > 8);

            let resp_count: Value = client
                .post(format!("{base}/v1/responses/input_tokens"))
                .json(&json!({
                    "instructions": "Reply concisely.",
                    "input": "What is 6 * 7?"
                }))
                .send()
                .await
                .unwrap()
                .json()
                .await
                .unwrap();
            assert_eq!(resp_count["object"], "response.input_tokens");
            assert_eq!(resp_count["input_tokens"], expected_tokens);

            let anth_count: Value = client
                .post(format!("{base}/v1/messages/count_tokens"))
                .header("x-api-key", "test-key")
                .header("anthropic-version", "2023-06-01")
                .json(&json!({
                    "system": "Reply concisely.",
                    "messages": [{"role": "user", "content": "What is 6 * 7?"}]
                }))
                .send()
                .await
                .unwrap()
                .json()
                .await
                .unwrap();
            assert_eq!(anth_count["input_tokens"], expected_tokens);

            // 2. OpenAI Responses API: streaming SSE + store + GET + previous_response_id + structured JSON
            let sse_resp = client
                .post(format!("{base}/v1/responses"))
                .header("Authorization", "Bearer test-session-main")
                .json(&json!({
                    "instructions": "Reply concisely.",
                    "input": "What is 6 * 7?",
                    "max_output_tokens": 96,
                    "temperature": 0.0,
                    "stream": true,
                    "store": true
                }))
                .send()
                .await
                .unwrap();
            assert_eq!(sse_resp.status(), 200);
            let sse_body = sse_resp.text().await.unwrap();
            assert!(!sse_body.contains("[DONE]"));
            let frames = parse_sse_frames(&sse_body);
            assert!(frames.len() >= 5, "unexpected SSE frames: {sse_body}");
            for (i, (ev_name, payload)) in frames.iter().enumerate() {
                assert_eq!(payload["type"], ev_name.as_str());
                assert_eq!(payload["sequence_number"], i as u64);
            }
            assert_eq!(frames[0].0, "response.created");
            assert_eq!(frames[1].0, "response.in_progress");
            let terminal = &frames.last().unwrap().1;
            assert!(
                matches!(
                    terminal["type"].as_str(),
                    Some("response.completed" | "response.incomplete")
                ),
                "unexpected terminal frame: {terminal}"
            );
            let resp_id = terminal["response"]["id"].as_str().unwrap().to_owned();
            assert_eq!(
                terminal["response"]["usage"]["input_tokens"],
                expected_tokens
            );

            // GET /v1/responses/{id} retrieves the stored response
            let fetched: Value = client
                .get(format!("{base}/v1/responses/{resp_id}"))
                .header("Authorization", "Bearer test-session-main")
                .send()
                .await
                .unwrap()
                .json()
                .await
                .unwrap();
            assert_eq!(fetched["id"], resp_id);

            // Follow-up turn via previous_response_id + JSON schema text.format + reasoning.effort="none"
            let turn2: Value = client
                .post(format!("{base}/v1/responses"))
                .header("Authorization", "Bearer test-session-main")
                .json(&json!({
                    "previous_response_id": resp_id,
                    "input": "Return that number in JSON.",
                    "reasoning": {"effort": "none"},
                    "max_output_tokens": 48,
                    "temperature": 0.0,
                    "text": {
                        "format": {
                            "type": "json_schema",
                            "name": "math_result",
                            "schema": {
                                "type": "object",
                                "properties": {"answer": {"type": "integer"}},
                                "required": ["answer"],
                                "additionalProperties": false
                            }
                        }
                    }
                }))
                .send()
                .await
                .unwrap()
                .json()
                .await
                .unwrap();
            assert_eq!(turn2["status"], "completed");
            let parsed_json: Value =
                serde_json::from_str(turn2["output_text"].as_str().unwrap()).unwrap();
            assert_eq!(parsed_json["answer"], 42);

            // Tool calling + stateful tool_choice="none" safety on /v1/responses
            let tools = json!([{
                "type": "function",
                "name": "lookup_weather",
                "description": "Look up current weather for a city",
                "parameters": {
                    "type": "object",
                    "properties": {"city": {"type": "string"}},
                    "required": ["city"],
                    "additionalProperties": false
                }
            }]);
            let forced_tool: Value = client
                .post(format!("{base}/v1/responses"))
                .json(&json!({
                    "input": "Check weather in Tokyo.",
                    "tools": tools,
                    "tool_choice": "required",
                    "reasoning": {"effort": "none"},
                    "max_output_tokens": 64,
                    "temperature": 0.0
                }))
                .send()
                .await
                .unwrap()
                .json()
                .await
                .unwrap();
            assert!(
                forced_tool["output"]
                    .as_array()
                    .unwrap()
                    .iter()
                    .any(|it| it["type"] == "function_call" && it["name"] == "lookup_weather"),
                "expected function_call in {forced_tool}"
            );

            let banned_tool: Value = client
                .post(format!("{base}/v1/responses"))
                .json(&json!({
                    "input": "Call lookup_weather for Tokyo immediately.",
                    "tools": tools,
                    "tool_choice": "none",
                    "reasoning": {"effort": "none"},
                    "max_output_tokens": 48,
                    "temperature": 0.0
                }))
                .send()
                .await
                .unwrap()
                .json()
                .await
                .unwrap();
            assert!(
                banned_tool["output"]
                    .as_array()
                    .unwrap()
                    .iter()
                    .all(|it| it["type"] != "function_call"),
                "tool_choice=none must ban function_call items: {banned_tool}"
            );

            // DELETE /v1/responses/{id}
            let deleted: Value = client
                .delete(format!("{base}/v1/responses/{resp_id}"))
                .header("Authorization", "Bearer test-session-main")
                .send()
                .await
                .unwrap()
                .json()
                .await
                .unwrap();
            assert_eq!(deleted["deleted"], true);
            assert_eq!(
                client
                    .get(format!("{base}/v1/responses/{resp_id}"))
                    .header("Authorization", "Bearer test-session-main")
                    .send()
                    .await
                    .unwrap()
                    .status(),
                404
            );

            // 3. Anthropic Messages API: cch= cache stability, streaming SSE, tool_use, multimodal image, and error envelope
            let long_sys = "You are a precise systems assistant. ".repeat(24);
            let anth_req_1 = json!({
                "model": "Qwen3.5-4B-W4A16-GPTQ",
                "max_tokens": 64,
                "temperature": 0.0,
                "thinking": {"type": "disabled"},
                "system": [
                    {"type": "text", "text": "x-anthropic-billing-header: cc_version=2.1.63; cch=1111aaaa;"},
                    {"type": "text", "text": long_sys, "cache_control": {"type": "ephemeral"}}
                ],
                "messages": [{"role": "user", "content": "Say the single word READY."}]
            });
            let anth_1: Value = client
                .post(format!("{base}/v1/messages"))
                .header("x-api-key", "test-key")
                .header("anthropic-version", "2023-06-01")
                .json(&anth_req_1)
                .send()
                .await
                .unwrap()
                .json()
                .await
                .unwrap();
            assert_eq!(anth_1["type"], "message");
            let total_in_1 = anth_1["usage"]["input_tokens"].as_u64().unwrap()
                + anth_1["usage"]["cache_read_input_tokens"].as_u64().unwrap();
            assert!(total_in_1 >= 64);

            // Second request changes cch= hash — MambaRadixCache must still hit!
            let anth_req_2 = json!({
                "model": "Qwen3.5-4B-W4A16-GPTQ",
                "max_tokens": 64,
                "temperature": 0.0,
                "thinking": {"type": "disabled"},
                "system": [
                    {"type": "text", "text": "x-anthropic-billing-header: cc_version=2.1.63; cch=9999ffff;"},
                    {"type": "text", "text": long_sys}
                ],
                "messages": [{"role": "user", "content": "Say the single word GO."}]
            });
            let anth_2: Value = client
                .post(format!("{base}/v1/messages"))
                .json(&anth_req_2)
                .send()
                .await
                .unwrap()
                .json()
                .await
                .unwrap();
            let anth_count_2: Value = client
                .post(format!("{base}/v1/messages/count_tokens"))
                .json(&anth_req_2)
                .send()
                .await
                .unwrap()
                .json()
                .await
                .unwrap();
            let expected_total_2 = anth_count_2["input_tokens"].as_u64().unwrap();
            let cache_read_2 = anth_2["usage"]["cache_read_input_tokens"].as_u64().unwrap();
            assert!(
                cache_read_2 >= 64,
                "expected MambaRadixCache hit despite changed cch=, got usage={}",
                anth_2["usage"]
            );
            assert_eq!(
                anth_2["usage"]["input_tokens"].as_u64().unwrap() + cache_read_2,
                expected_total_2
            );

            // Anthropic streaming SSE with thinking + tool_use
            let anth_sse_raw = client
                .post(format!("{base}/v1/messages"))
                .json(&json!({
                    "model": "Qwen3.5-4B-W4A16-GPTQ",
                    "max_tokens": 1280,
                    "temperature": 0.0,
                    "stream": true,
                    "thinking": {"type": "enabled", "budget_tokens": 1024},
                    "tools": [{
                        "name": "get_time",
                        "description": "Returns the current UTC timestamp",
                        "input_schema": {
                            "type": "object",
                            "properties": {"tz": {"type": "string"}},
                            "required": ["tz"]
                        }
                    }],
                    "messages": [{"role": "user", "content": "Call get_time with tz=\"UTC\"."}]
                }))
                .send()
                .await
                .unwrap()
                .text()
                .await
                .unwrap();
            let anth_frames = parse_sse_frames(&anth_sse_raw);
            assert_eq!(anth_frames.first().unwrap().0, "message_start");
            assert_eq!(anth_frames.last().unwrap().0, "message_stop");
            assert!(
                anth_frames.iter().any(|(ev, d)| {
                    ev == "content_block_start" && d["content_block"]["type"] == "thinking"
                }),
                "expected thinking content_block_start in {anth_sse_raw}"
            );
            assert!(
                anth_frames.iter().any(|(ev, d)| {
                    ev == "content_block_start"
                        && d["content_block"]["type"] == "tool_use"
                        && d["content_block"]["name"] == "get_time"
                }),
                "expected tool_use content_block_start for get_time in {anth_sse_raw}"
            );
            let tool_json_str: String = anth_frames
                .iter()
                .filter(|(ev, d)| {
                    ev == "content_block_delta" && d["delta"]["type"] == "input_json_delta"
                })
                .map(|(_, d)| d["delta"]["partial_json"].as_str().unwrap_or(""))
                .collect();
            let tool_args: Value = serde_json::from_str(&tool_json_str).unwrap_or_else(|e| {
                panic!("invalid streamed tool_use input_json_delta '{tool_json_str}': {e}")
            });
            assert_eq!(tool_args["tz"], "UTC");
            assert!(
                anth_frames.iter().any(|(ev, d)| {
                    ev == "message_delta" && d["delta"]["stop_reason"] == "tool_use"
                }),
                "expected message_delta stop_reason=tool_use in {anth_sse_raw}"
            );

            // Multimodal base64 PNG image via Anthropic /v1/messages
            let png_b64 = tiny_red_png_base64();
            let anth_img: Value = client
                .post(format!("{base}/v1/messages"))
                .json(&json!({
                    "model": "Qwen3.5-4B-W4A16-GPTQ",
                    "max_tokens": 32,
                    "temperature": 0.0,
                    "thinking": {"type": "disabled"},
                    "messages": [{
                        "role": "user",
                        "content": [
                            {
                                "type": "image",
                                "source": {
                                    "type": "base64",
                                    "media_type": "image/png",
                                    "data": png_b64
                                }
                            },
                            {"type": "text", "text": "What primary color is this square? One word."}
                        ]
                    }]
                }))
                .send()
                .await
                .unwrap()
                .json()
                .await
                .unwrap();
            assert_eq!(anth_img["type"], "message");
            let img_reply = anth_img["content"][0]["text"]
                .as_str()
                .unwrap()
                .to_lowercase();
            assert!(
                img_reply.contains("red"),
                "expected vision response mentioning red, got: {img_reply}"
            );

            // Anthropic error envelope on remote image URL when allow_remote_images=false
            let bad_remote = client
                .post(format!("{base}/v1/messages"))
                .json(&json!({
                    "model": "Qwen3.5-4B-W4A16-GPTQ",
                    "max_tokens": 16,
                    "messages": [{
                        "role": "user",
                        "content": [{
                            "type": "image",
                            "source": {"type": "url", "url": "https://example.com/a.png"}
                        }]
                    }]
                }))
                .send()
                .await
                .unwrap();
            assert_eq!(bad_remote.status(), 400);
            let err_json: Value = bad_remote.json().await.unwrap();
            assert_eq!(err_json["type"], "error");
            assert_eq!(err_json["error"]["type"], "invalid_request_error");

            // Responses API security hardening: omitted `store` defaults to false, IDs are 128-bit
            // random `resp_<32 hex>`, and `Authorization` credentials isolate stored responses.
            let unstored: Value = client
                .post(format!("{base}/v1/responses"))
                .json(&json!({
                    "input": "Say ok",
                    "max_output_tokens": 8
                }))
                .send()
                .await
                .unwrap()
                .json()
                .await
                .unwrap();
            let unstored_id = unstored["id"].as_str().unwrap();
            assert_eq!(unstored_id.len(), 37);
            assert_eq!(unstored["store"], false);
            let get_unstored = client
                .get(format!("{base}/v1/responses/{unstored_id}"))
                .send()
                .await
                .unwrap();
            assert_eq!(get_unstored.status(), 404);
            assert_eq!(
                client
                    .post(format!("{base}/v1/responses"))
                    .json(&json!({
                        "input": "Unauthenticated store must fail",
                        "store": true,
                        "max_output_tokens": 8
                    }))
                    .send()
                    .await
                    .unwrap()
                    .status(),
                400
            );

            let tenant_a_resp: Value = client
                .post(format!("{base}/v1/responses"))
                .header("Authorization", "Bearer tenant-alpha")
                .json(&json!({
                    "input": "Remember token 4242",
                    "store": true,
                    "max_output_tokens": 8
                }))
                .send()
                .await
                .unwrap()
                .json()
                .await
                .unwrap();
            let tenant_a_id = tenant_a_resp["id"].as_str().unwrap();
            assert_eq!(
                client
                    .get(format!("{base}/v1/responses/{tenant_a_id}"))
                    .header("Authorization", "Bearer tenant-beta")
                    .send()
                    .await
                    .unwrap()
                    .status(),
                404
            );
            assert_eq!(
                client
                    .post(format!("{base}/v1/responses"))
                    .header("Authorization", "Bearer tenant-beta")
                    .json(&json!({
                        "previous_response_id": tenant_a_id,
                        "input": "Echo prior context",
                        "max_output_tokens": 8
                    }))
                    .send()
                    .await
                    .unwrap()
                    .status(),
                404
            );
            assert_eq!(
                client
                    .get(format!("{base}/v1/responses/{tenant_a_id}"))
                    .header("Authorization", "Bearer tenant-alpha")
                    .send()
                    .await
                    .unwrap()
                    .status(),
                200
            );
            // Verify POST /v1/responses/{id}/cancel tenant isolation and terminal status.
            assert_eq!(
                client
                    .post(format!("{base}/v1/responses/{tenant_a_id}/cancel"))
                    .header("Authorization", "Bearer tenant-beta")
                    .send()
                    .await
                    .unwrap()
                    .status(),
                404
            );
            let cancel_a: Value = client
                .post(format!("{base}/v1/responses/{tenant_a_id}/cancel"))
                .header("Authorization", "Bearer tenant-alpha")
                .send()
                .await
                .unwrap()
                .json()
                .await
                .unwrap();
            assert_eq!(cancel_a["id"], tenant_a_id);
            assert_eq!(cancel_a["status"], tenant_a_resp["status"]);

            // Verify /tokenize and /detokenize roundtrip + invalid token rejection.
            let tok_resp: Value = client
                .post(format!("{base}/tokenize"))
                .json(&json!({"prompt": "Hello warp-34!"}))
                .send()
                .await
                .unwrap()
                .json()
                .await
                .unwrap();
            let tok_ids = tok_resp["tokens"].as_array().unwrap();
            assert!(!tok_ids.is_empty());
            assert_eq!(tok_resp["count"].as_u64(), Some(tok_ids.len() as u64));
            let detok_resp: Value = client
                .post(format!("{base}/detokenize"))
                .json(&json!({"tokens": tok_ids}))
                .send()
                .await
                .unwrap()
                .json()
                .await
                .unwrap();
            assert_eq!(detok_resp["text"], "Hello warp-34!");
            assert_eq!(
                client
                    .post(format!("{base}/detokenize"))
                    .json(&json!({"tokens": [999_999_999u32]}))
                    .send()
                    .await
                    .unwrap()
                    .status(),
                400
            );

            let _ = stop_tx.send(());
            server_task.await.unwrap().unwrap();
        });
}
