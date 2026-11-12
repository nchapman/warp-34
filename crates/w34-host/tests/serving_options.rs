#![cfg(feature = "cuda")]
mod common;

use serde_json::{Value, json};
use std::{sync::atomic::Ordering, time::Duration};
use w34_host::{
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
            check_route_auth(&client, &base).await;
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

async fn check_route_auth(client: &reqwest::Client, base: &str) {
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
    for path in ["/", "/health", "/metrics", "/api/metrics"] {
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
    for stream in [false, true] {
        for (path, body) in [
            (
                "/v1/completions",
                json!({"prompt":"Hi", "max_tokens":257, "stream":stream}),
            ),
            (
                "/v1/chat/completions",
                json!({"messages":[{"role":"user","content":"Hi"}],
                "max_tokens":257, "stream":stream}),
            ),
            (
                "/v1/responses",
                json!({"input":"Hi", "max_output_tokens":257, "stream":stream}),
            ),
            (
                "/v1/messages",
                json!({"model":"coding-model", "messages":[{"role":"user","content":"Hi"}],
                "max_tokens":257, "stream":stream}),
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
            assert!(response.text().await.unwrap().contains("context limit"));
        }
    }
}
