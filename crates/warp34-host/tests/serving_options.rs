#![cfg(feature = "cuda")]
mod common;

use serde_json::{Value, json};
use std::{sync::atomic::Ordering, time::Duration};
use warp34_host::{
    api_auth::ApiKey,
    cache::KvFormatPolicy,
    model::Qwen3_5Engine,
    server::{OpenAiServerState, ServeOptions, serve},
};

#[test]
fn invalid_options_do_not_load_model() {
    for options in [
        ServeOptions {
            max_num_seqs: 0,
            ..Default::default()
        },
        ServeOptions {
            max_num_seqs: 17,
            ..Default::default()
        },
        ServeOptions {
            max_model_len: Some(0),
            ..Default::default()
        },
    ] {
        assert!(
            OpenAiServerState::from_loader_with_options("alias".into(), false, options, || panic!(
                "invalid serving options must not load weights"
            ))
            .is_err()
        );
    }
}

#[test]
fn authenticated_serving_alias_context_and_concurrency() {
    if common::require_or_skip_cached_model(
        "Qwen/Qwen3.5-4B",
        "authenticated_serving_alias_context_and_concurrency",
    )
    .is_none()
    {
        return;
    }
    let options = ServeOptions {
        api_key: Some(ApiKey::new("deployment-test-key".into()).unwrap()),
        max_num_seqs: 2,
        max_model_len: Some(256),
        ..Default::default()
    };
    let state =
        OpenAiServerState::from_loader_with_options("coding-model".into(), false, options, || {
            let engine = Qwen3_5Engine::load_for_serving(
                "Qwen/Qwen3.5-4B",
                1,
                KvFormatPolicy::Tq4,
                Some(256),
                true,
                2,
            )
            .map_err(|e| e.to_string())?;
            assert_eq!(engine.ranks[0].state_pool.total_slots, 8);
            Ok(engine)
        })
        .unwrap();
    assert_eq!(state.worker.max_model_len(), 256);
    assert_eq!(state.worker.kv_token_capacity, 4096);
    let error = state
        .worker
        .submit_decision_candidate_logits(vec![100; 257], vec![100], None)
        .err()
        .expect("decision prefills obey the same context ceiling");
    assert!(error.to_string().contains("context limit"));
    tokio::runtime::Builder::new_multi_thread()
        .worker_threads(2)
        .enable_all()
        .build()
        .unwrap()
        .block_on(async {
            let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
            let base = format!("http://{}", listener.local_addr().unwrap());
            let (stop_tx, stop_rx) = tokio::sync::oneshot::channel();
            let server_state = state.clone();
            let server = tokio::spawn(async move {
                serve(listener, server_state, async {
                    let _ = stop_rx.await;
                })
                .await
            });
            let client = reqwest::Client::builder()
                .timeout(Duration::from_secs(60))
                .build()
                .unwrap();
            check_route_auth(&client, &base, true).await;
            let models: Value = client
                .get(format!("{base}/v1/models"))
                .bearer_auth("deployment-test-key")
                .send()
                .await
                .unwrap()
                .json()
                .await
                .unwrap();
            assert_eq!(models["data"][0]["id"], "coding-model");
            assert_eq!(models["data"][0]["max_model_len"], 256);
            assert_eq!(state.metrics.snapshot().model, "coding-model");
            check_context_limit(&client, &base).await;
            assert_eq!(state.metrics.snapshot().requests.submitted, 0);

            // Offer four concurrent requests; the configured scheduler admits at most two.
            let mut tasks = Vec::new();
            for i in 0..4 {
                let client = client.clone();
                let base = base.clone();
                tasks.push(tokio::spawn(async move {
                    let response = client
                        .post(format!("{base}/v1/completions"))
                        .bearer_auth("deployment-test-key")
                        .json(
                            &json!({"model":"coding-model", "prompt":format!("Count {i}:"),
                            "max_tokens":32,"temperature":0,"ignore_eos":true}),
                        )
                        .send()
                        .await
                        .unwrap();
                    assert_eq!(response.status(), 200);
                    let body: Value = response.json().await.unwrap();
                    assert_eq!(body["model"], "coding-model");
                    assert_eq!(body["usage"]["completion_tokens"], 32);
                }));
            }
            for task in tasks {
                task.await.unwrap();
            }
            assert_eq!(state.worker.stats.peak_active.load(Ordering::Relaxed), 2);
            stop_tx.send(()).unwrap();
            server.await.unwrap().unwrap();
        });
}

#[test]
fn disabled_webui_preserves_public_metrics_and_protected_apis() {
    if common::require_or_skip_cached_model(
        "Qwen/Qwen3.5-4B",
        "disabled_webui_preserves_public_metrics_and_protected_apis",
    )
    .is_none()
    {
        return;
    }
    let state = OpenAiServerState::from_loader_with_options(
        "private-engine".into(),
        false,
        ServeOptions {
            api_key: Some(ApiKey::new("deployment-test-key".into()).unwrap()),
            webui: false,
            max_num_seqs: 1,
            ..Default::default()
        },
        || {
            Qwen3_5Engine::load_for_serving(
                "Qwen/Qwen3.5-4B",
                1,
                KvFormatPolicy::Tq4,
                Some(512),
                true,
                1,
            )
            .map_err(|e| e.to_string())
        },
    )
    .unwrap();
    assert_eq!(state.worker.max_model_len(), 8192);
    tokio::runtime::Builder::new_multi_thread()
        .worker_threads(2)
        .enable_all()
        .build()
        .unwrap()
        .block_on(async {
            let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
            let base = format!("http://{}", listener.local_addr().unwrap());
            let (stop_tx, stop_rx) = tokio::sync::oneshot::channel();
            let server = tokio::spawn(async move {
                serve(listener, state, async {
                    let _ = stop_rx.await;
                })
                .await
            });
            let client = reqwest::Client::builder()
                .timeout(Duration::from_secs(60))
                .build()
                .unwrap();
            check_route_auth(&client, &base, false).await;
            let metrics: Value = client
                .get(format!("{base}/api/metrics"))
                .send()
                .await
                .unwrap()
                .json()
                .await
                .unwrap();
            assert_eq!(metrics["model"], "private-engine");
            check_large_and_omitted_output_budgets(&client, &base).await;
            stop_tx.send(()).unwrap();
            server.await.unwrap().unwrap();
        });
}

async fn check_route_auth(client: &reqwest::Client, base: &str, webui: bool) {
    // Bodies are deliberately malformed: auth must run before parsing or GPU submission.
    for path in [
        "/v1/chat/completions",
        "/v1/completions",
        "/v1/responses",
        "/v1/messages",
        "/v1/chat/completions/input_tokens",
        "/chat/completions/input_tokens",
        "/v1/responses/input_tokens",
        "/responses/input_tokens",
        "/v1/messages/count_tokens",
        "/v1/systemone",
        "/systemone",
        "/v1/decisions",
        "/decisions",
        "/tokenize",
        "/detokenize",
        "/v1/responses/resp_unknown/cancel",
    ] {
        let response = client
            .post(format!("{base}{path}"))
            .body("invalid json")
            .send()
            .await
            .unwrap();
        assert_eq!(response.status(), 401, "{path}");
    }
    for path in ["/v1/models", "/v1/responses/resp_unknown"] {
        assert_eq!(
            client
                .get(format!("{base}{path}"))
                .send()
                .await
                .unwrap()
                .status(),
            401
        );
    }
    assert_eq!(
        client
            .delete(format!("{base}/v1/responses/resp_unknown"))
            .send()
            .await
            .unwrap()
            .status(),
        401
    );
    for path in ["/health", "/metrics", "/api/metrics"] {
        assert_eq!(
            client
                .get(format!("{base}{path}"))
                .send()
                .await
                .unwrap()
                .status(),
            200,
            "{path}"
        );
    }
    // Exercise real hashed assets, not a nonexistent filename that always returns 404.
    let index = include_str!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../dashboard/dist/index.html"
    ));
    let assets: Vec<_> = index
        .split('"')
        .filter(|part| part.starts_with("/assets/"))
        .collect();
    assert!(!assets.is_empty());
    for path in [
        "/",
        "/connect",
        "/playground",
        "/playground/request",
        "/favicon.svg",
    ]
    .into_iter()
    .chain(assets)
    {
        for key in [None, Some("deployment-test-key")] {
            let request = client.get(format!("{base}{path}"));
            let request = if let Some(key) = key {
                request.bearer_auth(key)
            } else {
                request
            };
            assert_eq!(
                request.send().await.unwrap().status(),
                if webui { 200 } else { 404 },
                "{path}"
            );
        }
    }
    assert_eq!(
        client
            .get(format!("{base}/v1/models"))
            .header("x-api-key", "deployment-test-key")
            .send()
            .await
            .unwrap()
            .status(),
        200
    );
}

async fn check_context_limit(client: &reqwest::Client, base: &str) {
    let over_prompt = "hello ".repeat(300);
    // Token-count utilities must NOT fail on oversized prompts so coding agents can count
    // full conversation histories before compacting them.
    for (path, body) in [
        (
            "/v1/chat/completions/input_tokens",
            json!({"model":"coding-model", "messages":[{"role":"user","content":&over_prompt}]}),
        ),
        (
            "/v1/responses/input_tokens",
            json!({"model":"coding-model", "input":&over_prompt}),
        ),
        (
            "/v1/messages/count_tokens",
            json!({"model":"coding-model", "messages":[{"role":"user","content":&over_prompt}]}),
        ),
    ] {
        let resp = client
            .post(format!("{base}{path}"))
            .bearer_auth("deployment-test-key")
            .json(&body)
            .send()
            .await
            .unwrap();
        assert_eq!(resp.status(), 200, "{path}");
        let payload: Value = resp.json().await.unwrap();
        assert!(
            payload["input_tokens"].as_u64().unwrap_or(0) >= 256,
            "{path}: {payload}"
        );
    }

    // Generative endpoints must reject prompt_tokens >= max_model_len (256) with HTTP 400
    // and structured coding-agent context-overflow codes/messages.
    for stream in [false, true] {
        for (path, body, expected_param) in [
            (
                "/v1/completions",
                json!({"prompt":&over_prompt, "max_tokens":16, "stream":stream}),
                Some("prompt"),
            ),
            (
                "/v1/chat/completions",
                json!({"messages":[{"role":"user","content":&over_prompt}],
                "max_tokens":16, "stream":stream}),
                Some("messages"),
            ),
            (
                "/v1/responses",
                json!({"input":&over_prompt, "max_output_tokens":16, "stream":stream}),
                Some("input"),
            ),
            (
                "/v1/messages",
                json!({"model":"coding-model", "messages":[{"role":"user","content":&over_prompt}],
                "max_tokens":16, "stream":stream}),
                None,
            ),
        ] {
            let response = client
                .post(format!("{base}{path}"))
                .bearer_auth("deployment-test-key")
                .json(&body)
                .send()
                .await
                .unwrap();
            assert_eq!(response.status(), 400, "{path} stream={stream}");
            let err_json: Value = response.json().await.unwrap();
            if let Some(param) = expected_param {
                assert_eq!(
                    err_json["error"]["code"], "context_length_exceeded",
                    "{path}: {err_json}"
                );
                assert_eq!(err_json["error"]["param"], param, "{path}: {err_json}");
                assert_eq!(
                    err_json["error"]["type"], "invalid_request_error",
                    "{path}: {err_json}"
                );
            } else {
                assert_eq!(err_json["type"], "error", "{path}: {err_json}");
                assert_eq!(
                    err_json["error"]["type"], "invalid_request_error",
                    "{path}: {err_json}"
                );
                let msg = err_json["error"]["message"].as_str().unwrap_or("");
                assert!(
                    msg.starts_with("prompt is too long:") && msg.contains("> 255 maximum"),
                    "{path}: {msg}"
                );
            }
        }
    }
}

async fn check_large_and_omitted_output_budgets(client: &reqwest::Client, base: &str) {
    let models: Value = client
        .get(format!("{base}/v1/models"))
        .bearer_auth("deployment-test-key")
        .send()
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    assert_eq!(models["data"][0]["max_model_len"], 8192);

    // 1. Explicit max_tokens > 4096 within the 8192-token context limit succeeds on /v1/chat/completions.
    let chat_resp = client
        .post(format!("{base}/v1/chat/completions"))
        .bearer_auth("deployment-test-key")
        .json(&json!({
            "model": "private-engine",
            "messages": [{"role": "user", "content": "Reply with OK."}],
            "max_tokens": 5000,
            "temperature": 0,
            "stop": ["\n", "."],
            "chat_template_kwargs": {"enable_thinking": false}
        }))
        .send()
        .await
        .unwrap();
    assert_eq!(chat_resp.status(), 200);

    // 2. Omitted max_output_tokens on /v1/responses resolves to remaining context (8192 - prompt_tokens).
    let resp_omitted = client
        .post(format!("{base}/v1/responses"))
        .bearer_auth("deployment-test-key")
        .json(&json!({
            "model": "private-engine",
            "input": "Reply with OK.",
            "temperature": 0,
            "stop": ["\n", "."],
            "reasoning": {"effort": "none"}
        }))
        .send()
        .await
        .unwrap();
    assert_eq!(resp_omitted.status(), 200);
    let resp_body: Value = resp_omitted.json().await.unwrap();
    let input_toks = resp_body["usage"]["input_tokens"].as_u64().unwrap();
    assert_eq!(
        resp_body["max_output_tokens"].as_u64(),
        Some(8192 - input_toks)
    );

    // 3. Anthropic /v1/messages accepts max_tokens > 4096 and thinking.budget_tokens >= 4096.
    let anth_resp = client
        .post(format!("{base}/v1/messages"))
        .bearer_auth("deployment-test-key")
        .json(&json!({
            "model": "private-engine",
            "max_tokens": 6000,
            "thinking": {"type": "enabled", "budget_tokens": 5000},
            "stop_sequences": ["\n", "."],
            "messages": [{"role": "user", "content": "2+2="}]
        }))
        .send()
        .await
        .unwrap();
    assert_eq!(anth_resp.status(), 200);

    // 4. Static coding-agent max_tokens > remaining window (e.g. Claude Code 32000) clamps
    //    cleanly to (8192 - prompt_tokens) instead of failing with HTTP 400 when prompt < 8192.
    let clamped_resp = client
        .post(format!("{base}/v1/responses"))
        .bearer_auth("deployment-test-key")
        .json(&json!({
            "model": "private-engine",
            "input": "Reply with OK.",
            "max_output_tokens": 32000,
            "temperature": 0,
            "stop": ["\n", "."],
            "reasoning": {"effort": "none"}
        }))
        .send()
        .await
        .unwrap();
    assert_eq!(clamped_resp.status(), 200);
    let clamped_body: Value = clamped_resp.json().await.unwrap();
    let clamped_in = clamped_body["usage"]["input_tokens"].as_u64().unwrap();
    assert_eq!(
        clamped_body["max_output_tokens"].as_u64(),
        Some(8192 - clamped_in)
    );

    let clamped_anth = client
        .post(format!("{base}/v1/messages"))
        .bearer_auth("deployment-test-key")
        .json(&json!({
            "model": "private-engine",
            "max_tokens": 32000,
            "temperature": 0,
            "stop_sequences": ["\n", "."],
            "thinking": {"type": "disabled"},
            "messages": [{"role": "user", "content": "Reply with OK."}]
        }))
        .send()
        .await
        .unwrap();
    assert_eq!(clamped_anth.status(), 200);

    // 5. min_tokens exceeding the clamped remaining context budget still fails with HTTP 400.
    let min_over_resp = client
        .post(format!("{base}/v1/chat/completions"))
        .bearer_auth("deployment-test-key")
        .json(&json!({
            "model": "private-engine",
            "messages": [{"role": "user", "content": "Reply with OK."}],
            "min_tokens": 8190,
            "max_tokens": 32000
        }))
        .send()
        .await
        .unwrap();
    assert_eq!(min_over_resp.status(), 400);
    assert!(
        min_over_resp
            .text()
            .await
            .unwrap()
            .contains("min_tokens exceeds max_tokens")
    );
}
