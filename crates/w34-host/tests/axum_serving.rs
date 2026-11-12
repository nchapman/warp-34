#![cfg(feature = "cuda")]
mod common;

use serde_json::{Value, json};
use std::{
    sync::atomic::Ordering,
    time::{Duration, Instant},
};
use w34_host::{
    cache::KvFormatPolicy,
    engine_worker::GenerationEvent,
    model::Qwen3_5Engine,
    parser::SamplingParams,
    server::{OpenAiServerState, serve},
};

static GPU_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

#[test]
fn invalid_vision_request_does_not_abort_text_peer() {
    if common::require_or_skip_cached_model(
        "Qwen/Qwen3.5-4B",
        "invalid_vision_request_does_not_abort_text_peer",
    )
    .is_none()
    {
        return;
    }
    let _gpu = GPU_LOCK.lock().unwrap();
    let state = OpenAiServerState::from_loader("vision-isolation".into(), false, || {
        Qwen3_5Engine::load("Qwen/Qwen3.5-4B", 1, KvFormatPolicy::Tq4, Some(64), true)
            .map_err(|e| e.to_string())
    })
    .unwrap();
    tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .unwrap()
        .block_on(async {
            let mut text = state
                .worker
                .submit(
                    vec![100, 101, 102],
                    SamplingParams {
                        max_tokens: 128,
                        ignore_eos: true,
                        ..SamplingParams::default()
                    },
                    None,
                )
                .unwrap();
            assert!(matches!(
                text.events.recv().await,
                Some(GenerationEvent::Token { .. })
            ));
            let image = w34_host::vision::PreprocessedImage {
                patches_f32: vec![0.0; 6144].into(),
                grid_thw: [1, 2, 2],
                content_hash: 123,
            };
            let mut invalid = state
                .worker
                .submit_with_images(
                    vec![100, 101],
                    vec![image],
                    SamplingParams {
                        max_tokens: 8,
                        ..SamplingParams::default()
                    },
                    None,
                )
                .unwrap();
            let outcome = tokio::time::timeout(Duration::from_secs(10), async {
                let rejected = invalid.events.recv().await;
                let mask = w34_host::parser::TokenBitmaskConstraint::from_allowed_tokens(
                    state.worker.vocab_size,
                    vec![],
                );
                let mut unsatisfiable = state
                    .worker
                    .submit(
                        vec![100, 101],
                        SamplingParams {
                            max_tokens: 8,
                            ..SamplingParams::default()
                        },
                        Some(mask),
                    )
                    .unwrap();
                assert!(matches!(
                    unsatisfiable.events.recv().await,
                    Some(GenerationEvent::Error(_))
                ));
                let mut count = 1;
                while let Some(event) = text.events.recv().await {
                    match event {
                        GenerationEvent::Token { .. } => count += 1,
                        GenerationEvent::Finished { .. } => return (rejected, Ok(count)),
                        GenerationEvent::Error(e) => return (rejected, Err(e.to_string())),
                    }
                }
                (rejected, Err("unexpected EOF".to_string()))
            })
            .await;
            state.worker.shutdown().await.unwrap();
            let (rejected, text) = outcome.unwrap();
            assert!(matches!(rejected, Some(GenerationEvent::Error(_))));
            assert_eq!(text.unwrap(), 128);
        });
}

#[test]
fn kv_pressure_queues_newcomers_without_failing_running_requests() {
    if common::require_or_skip_cached_model(
        "Qwen/Qwen3.5-4B",
        "kv_pressure_queues_newcomers_without_failing_running_requests",
    )
    .is_none()
    {
        return;
    }
    let _gpu = GPU_LOCK.lock().unwrap();
    let state = OpenAiServerState::from_loader("isolation".into(), false, || {
        Qwen3_5Engine::load("Qwen/Qwen3.5-4B", 1, KvFormatPolicy::Tq4, Some(64), true)
            .map_err(|e| e.to_string())
    })
    .unwrap();
    tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .unwrap()
        .block_on(async {
            let params = SamplingParams {
                max_tokens: 512,
                ignore_eos: true,
                ..SamplingParams::default()
            };
            let mut first = state
                .worker
                .submit(vec![100, 101, 102], params.clone(), None)
                .unwrap();
            assert!(matches!(
                first.events.recv().await,
                Some(GenerationEvent::Token { .. })
            ));
            let second = state
                .worker
                .submit(vec![100, 101, 102], params, None)
                .unwrap();
            async fn drain(
                mut generation: w34_host::engine_worker::GenerationStream,
            ) -> Result<usize, String> {
                let mut count = 0;
                while let Some(event) = generation.events.recv().await {
                    match event {
                        GenerationEvent::Token { .. } => count += 1,
                        GenerationEvent::Finished { .. } => return Ok(count),
                        GenerationEvent::Error(e) => return Err(e.to_string()),
                    }
                }
                Err("unexpected worker EOF".into())
            }
            let result = tokio::time::timeout(Duration::from_secs(30), async {
                tokio::join!(drain(first), drain(second))
            })
            .await;
            state.worker.shutdown().await.unwrap();
            let (first, second) = result.unwrap();
            assert_eq!(first.unwrap(), 511);
            assert_eq!(second.unwrap(), 512);
        });
}

#[test]
fn axum_live_sse_concurrency_disconnect_backpressure_and_shutdown() {
    if common::require_or_skip_cached_model(
        "Qwen/Qwen3.5-4B",
        "axum_live_sse_concurrency_disconnect_backpressure_and_shutdown",
    )
    .is_none()
    {
        return;
    }
    let _gpu = GPU_LOCK.lock().unwrap();
    let state = OpenAiServerState::from_loader("axum-test".into(), false, || {
        assert_eq!(std::thread::current().name(), Some("w34-cuda"));
        Qwen3_5Engine::load("Qwen/Qwen3.5-4B", 1, KvFormatPolicy::Tq4, Some(256), true)
            .map_err(|e| e.to_string())
    })
    .unwrap();
    tokio::runtime::Builder::new_multi_thread().worker_threads(2).enable_all().build().unwrap().block_on(async {
        tokio::time::timeout(Duration::from_secs(90), async {
            let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
            let url = format!("http://{}",listener.local_addr().unwrap());
            let (shutdown_tx,shutdown_rx) = tokio::sync::oneshot::channel();
            let server = tokio::spawn(serve(listener,state.clone(), async { let _ = shutdown_rx.await; }));
            let client = reqwest::Client::builder().timeout(Duration::from_secs(30)).build().unwrap();
            let request = json!({"messages":[{"role":"user","content":"List the positive integers, starting with 1."}],"chat_template_kwargs":{"enable_thinking":false},"temperature":0,"ignore_eos":true,"max_tokens":256,"stream":true,"stream_options":{"include_usage":true}});
            let before = state.worker.stats.completed.load(Ordering::Relaxed);
            let mut response = client.post(format!("{url}/v1/chat/completions")).json(&request).send().await.unwrap();
            assert!(response.status().is_success());
            assert_eq!(response.headers()["content-type"],"text/event-stream");
            let mut chunks = String::new();
            loop {
                chunks.push_str(&String::from_utf8_lossy(&response.chunk().await.unwrap().expect("early EOF")));
                if chunks.contains("\"content\":") { break; }
            }
            assert_eq!(state.worker.stats.completed.load(Ordering::Relaxed),before,"SSE must arrive before generation completes");
            assert!(!chunks.contains("[DONE]"));
            let cancelled_before = state.worker.stats.cancelled.load(Ordering::Relaxed);
            drop(response);

            // HTTP health never queues behind the engine or an inference mutex.
            // A client stalled midway through its request body must not occupy an
            // executor thread or prevent unrelated HTTP requests from progressing.
            let mut stalled = std::net::TcpStream::connect(url.trim_start_matches("http://")).unwrap();
            std::io::Write::write_all(&mut stalled,b"POST /v1/completions HTTP/1.1\r\nHost: localhost\r\nContent-Length: 100\r\n\r\n{").unwrap();
            let health_start = Instant::now();
            assert!(client.get(format!("{url}/health")).send().await.unwrap().status().is_success());
            assert!(health_start.elapsed() < Duration::from_secs(1));

            for n in [8usize,16] {
                let start = Instant::now();
                let barrier = std::sync::Arc::new(tokio::sync::Barrier::new(n));
                let mut jobs = tokio::task::JoinSet::new();
                for _ in 0..n {
                    let client = client.clone(); let url = url.clone(); let barrier = barrier.clone();
                    jobs.spawn(async move {
                        barrier.wait().await;
                        let response = client.post(format!("{url}/v1/completions")).json(&json!({"prompt":"The positive integers are 1, 2, 3,","max_tokens":32,"temperature":0,"ignore_eos":true})).send().await.unwrap();
                        let status = response.status(); let body: Value = response.json().await.unwrap();
                        assert!(status.is_success(),"{status}: {body}");
                        assert_eq!(body["usage"]["completion_tokens"],32,"{body}");
                        assert_eq!(body["choices"][0]["finish_reason"],"length");
                        body["choices"][0]["text"].as_str().unwrap().to_owned()
                    });
                }
                let mut outputs = Vec::new();
                while let Some(job) = jobs.join_next().await { outputs.push(job.unwrap()); }
                assert!(outputs.iter().all(|s| s == &outputs[0]),"identical concurrent greedy requests must match");
                println!("Axum {n}-agent: {:.2} generated tok/s (includes prefill/HTTP)",n as f64*32.0/start.elapsed().as_secs_f64());
            }
            assert!(state.worker.stats.cancelled.load(Ordering::Relaxed) > cancelled_before,"disconnect must cancel the live request");
            assert!(state.worker.stats.peak_active.load(Ordering::Relaxed)>1,"requests were serialized");
            assert!(state.worker.stats.decode_bucket_mask.load(Ordering::Relaxed)>1,"multi-request graph never captured");

            // Deterministic backpressure: never drain one receiver, but drain its longer peer.
            let prompt = vec![100,101,102];
            let mut slow = state.worker.submit(prompt.clone(),SamplingParams { max_tokens:256,ignore_eos:true,..SamplingParams::default() },None).unwrap();
            let mut peer = state.worker.submit(prompt,SamplingParams { max_tokens:128,ignore_eos:true,..SamplingParams::default() },None).unwrap();
            let mut peer_tokens = 0;
            let mut finished = false;
            while let Some(event) = peer.events.recv().await {
                match event {
                    GenerationEvent::Token {..} => peer_tokens+=1,
                    GenerationEvent::Finished {..} => { finished=true; break; },
                    GenerationEvent::Error(error) => panic!("{error}"),
                }
            }
            assert!(finished); assert_eq!(peer_tokens,128);
            let mut slow_tokens = 0;
            while let Some(event) = slow.events.recv().await { if matches!(event,GenerationEvent::Token{..}) { slow_tokens+=1; } }
            assert!(slow_tokens<=64,"unbounded slow consumer buffer: {slow_tokens}");

            let mut usage_request = request.clone(); usage_request["max_tokens"]=json!(8);
            let response = client.post(format!("{url}/v1/chat/completions")).json(&usage_request).send().await.unwrap();
            let sse = response.text().await.unwrap();
            assert!(sse.contains("\"choices\":[],\"usage\":"),"{sse}");
            assert_eq!(sse.matches("[DONE]").count(),1);
            assert!(!sse.contains("\"error\""),"{sse}");

            // Legacy completions use integer logprobs and parallel arrays, in both transports.
            let mut legacy = json!({"prompt":"The positive integers are 1, 2, 3,","max_tokens":8,"temperature":0,"ignore_eos":true,"logprobs":2});
            let response = client.post(format!("{url}/v1/completions")).json(&legacy).send().await.unwrap();
            assert!(response.status().is_success());
            let full: Value = response.json().await.unwrap();
            let full_logs = &full["choices"][0]["logprobs"];
            for field in ["tokens","token_logprobs","top_logprobs","text_offset"] {
                assert_eq!(full_logs[field].as_array().unwrap().len(),8,"{full}");
            }
            assert!(full_logs["token_logprobs"].as_array().unwrap().iter().all(|p| p.as_f64().is_some_and(f64::is_finite)));
            legacy["stream"] = json!(true);
            let response = client.post(format!("{url}/v1/completions")).json(&legacy).send().await.unwrap();
            assert!(response.status().is_success());
            let sse = response.text().await.unwrap();
            assert_eq!(sse.matches("[DONE]").count(),1);
            let records: Vec<Value> = sse.lines().filter_map(|line| line.strip_prefix("data: ")).filter(|data| *data != "[DONE]").map(|data| serde_json::from_str(data).unwrap()).collect();
            for field in ["tokens","text_offset"] {
                let flattened: Vec<Value> = records.iter().filter_map(|r| r["choices"][0]["logprobs"][field].as_array()).flatten().cloned().collect();
                assert_eq!(flattened,*full_logs[field].as_array().unwrap(),"{sse}");
            }

            let mut warm_request = json!({"prompt":"hello ".repeat(100),"max_tokens":8,"temperature":0});
            let first: Value = client.post(format!("{url}/v1/completions")).json(&warm_request).send().await.unwrap().json().await.unwrap();
            let stop = first["choices"][0]["text"].as_str().unwrap().chars().next().unwrap().to_string();
            warm_request["stop"] = json!(stop);
            let warm: Value = client.post(format!("{url}/v1/completions")).json(&warm_request).send().await.unwrap().json().await.unwrap();
            assert_eq!(warm["choices"][0]["text"],"");
            assert!(warm["usage"]["prompt_tokens_details"]["cached_tokens"].as_u64().unwrap() >= 64,"{warm}");

            // Shutdown drains already-admitted generations instead of truncating SSE.
            let mut active = client.post(format!("{url}/v1/chat/completions")).json(&request).send().await.unwrap();
            assert!(active.chunk().await.unwrap().is_some());
            drop(stalled);
            shutdown_tx.send(()).unwrap();
            let tail = active.text().await.unwrap();
            assert!(tail.contains("[DONE]"),"{tail}");
            assert!(!tail.contains("\"error\""),"{tail}");
            tokio::time::timeout(Duration::from_secs(5),server).await.unwrap().unwrap().unwrap();
            assert!(state.worker.submit(vec![100],SamplingParams::default(),None).is_err());
        }).await.expect("serving test deadline");
    });
}

#[test]
fn legacy_terminal_token_logprobs_live() {
    if common::require_or_skip_cached_model(
        "Qwen/Qwen3.5-4B",
        "legacy_terminal_token_logprobs_live",
    )
    .is_none()
    {
        return;
    }
    let _gpu = GPU_LOCK.lock().unwrap();
    let state = OpenAiServerState::from_loader("terminal-test".into(), false, || {
        Qwen3_5Engine::load("Qwen/Qwen3.5-4B", 1, KvFormatPolicy::Tq4, Some(64), true)
            .map_err(|e| e.to_string())
    })
    .unwrap();
    tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .unwrap()
        .block_on(async {
            let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
            let url = format!("http://{}/v1/completions", listener.local_addr().unwrap());
            let (tx, rx) = tokio::sync::oneshot::channel();
            let server = tokio::spawn(serve(listener, state.clone(), async {
                let _ = rx.await;
            }));
            let client = reqwest::Client::builder()
                .timeout(Duration::from_secs(30))
                .build()
                .unwrap();
            // Force an ordinary hidden stop token and a real EOS, without ignore_eos.
            for (id, explicit_stop) in [(100, true), (state.worker.eos_token_ids[0], false)] {
                let mut request = json!({"prompt":"Hello", "temperature":0, "max_tokens":8,
                "logprobs":2, "logit_bias":{id.to_string():100},
                "stream_options":{"include_usage":true}});
                if explicit_stop {
                    request["stop_token_ids"] = json!([id]);
                }
                let full: Value = client
                    .post(&url)
                    .json(&request)
                    .send()
                    .await
                    .unwrap()
                    .error_for_status()
                    .unwrap()
                    .json()
                    .await
                    .unwrap();
                request["stream"] = json!(true);
                let sse = client
                    .post(&url)
                    .json(&request)
                    .send()
                    .await
                    .unwrap()
                    .error_for_status()
                    .unwrap()
                    .text()
                    .await
                    .unwrap();
                assert_eq!(full["usage"]["completion_tokens"], 1, "{full}");
                assert_eq!(full["choices"][0]["text"], "");
                assert_eq!(full["choices"][0]["finish_reason"], "stop");
                let lp = &full["choices"][0]["logprobs"];
                for key in ["tokens", "token_logprobs", "top_logprobs", "text_offset"] {
                    assert_eq!(lp[key].as_array().unwrap().len(), 1, "{key}: {full}");
                }
                assert_eq!(lp["text_offset"], json!([0]));
                assert_eq!(
                    lp["tokens"][0],
                    state
                        .worker
                        .tokenizer
                        .as_ref()
                        .unwrap()
                        .decode(&[id as u32], false)
                        .unwrap()
                );
                assert_eq!(sse.matches("data: [DONE]").count(), 1, "{sse}");
                let events: Vec<Value> = sse
                    .lines()
                    .filter_map(|line| line.strip_prefix("data: "))
                    .filter(|data| *data != "[DONE]")
                    .map(|data| serde_json::from_str(data).unwrap())
                    .collect();
                let mut records = Vec::new();
                let mut finishes = 0;
                let mut usages = 0;
                for event in events {
                    if !event["usage"].is_null() {
                        assert_eq!(event["usage"], full["usage"]);
                        usages += 1;
                    }
                    for choice in event["choices"].as_array().unwrap() {
                        assert_eq!(choice["text"], "");
                        if !choice["finish_reason"].is_null() {
                            assert_eq!(choice["finish_reason"], "stop");
                            finishes += 1;
                        }
                        if !choice["logprobs"].is_null() {
                            records.push(choice["logprobs"].clone());
                        }
                    }
                }
                assert_eq!(records, vec![lp.clone()], "{sse}");
                assert_eq!(finishes, 1);
                assert_eq!(usages, 1);
            }
            tx.send(()).unwrap();
            server.await.unwrap().unwrap();
        });
}
