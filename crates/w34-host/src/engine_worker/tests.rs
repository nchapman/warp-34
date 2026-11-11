use std::{
    collections::HashMap,
    sync::{
        Arc, Mutex,
        atomic::{AtomicBool, AtomicU64, Ordering},
    },
};
use tokio::sync::mpsc;

use super::{
    DecisionReadoutTap, DecisionTapOutput, EngineWorker, GenerationEvent, GenerationStream,
    Request, SamplerV1, SamplingParams, Telemetry, WorkerError, WorkerStats, generation_channel,
    notify_request_error,
    runner::{emit, require_healthy_host_compute, run_supervised_loop},
    validate_request_length,
};

fn terminal_test_request() -> (
    Request,
    GenerationStream,
    tokio::sync::oneshot::Receiver<Result<DecisionTapOutput, WorkerError>>,
) {
    let (output, events) = generation_channel(2);
    let (tx, rx) = tokio::sync::oneshot::channel();
    let cached_tokens = Arc::new(AtomicU64::new(0));
    let cancelled = Arc::new(AtomicBool::new(false));
    let request = Request {
        lifecycle: Some(Arc::new(Telemetry::default()).request(1)),
        prompt: vec![1],
        images: Vec::new(),
        vision: None,
        reserved_pages: 0,
        cached_tokens: cached_tokens.clone(),
        sampler: SamplerV1::new(SamplingParams::default(), &[1], None),
        output,
        terminal: events.terminal.clone(),
        cancelled: cancelled.clone(),
        max_prefix_lookup_len: None,
        mamba_branching_seqlen: None,
        prefix_bidir_len: None,
        decision_tap: Some(DecisionReadoutTap::CandidateLogits {
            token_ids: vec![1],
            tx,
        }),
    };
    (
        request,
        GenerationStream {
            events,
            cached_tokens,
            cancelled,
        },
        rx,
    )
}

#[test]
fn telemetry_counts_engine_output_even_when_delivery_is_full() {
    let (mut request, _stream, _tap) = terminal_test_request();
    let telemetry = Arc::new(Telemetry::default());
    request.lifecycle = Some(telemetry.request(1));
    request.lifecycle.as_ref().unwrap().scheduled(0);
    for id in 0..3 {
        emit(&request, id, None);
    }
    assert_eq!(telemetry.snapshot().tokens.generated, 3);
    assert!(request.cancelled.load(Ordering::Acquire));
    drop(request);
    let s = telemetry.snapshot();
    assert_eq!(s.requests.cancelled, 1);
    assert_eq!(s.requests.failed, 0);
    assert_eq!(s.requests.running, 0);
    assert_eq!(s.latency.ttft.count, 1);
}

#[tokio::test]
async fn terminal_finished_survives_full_queue_and_is_delivered_once() {
    let (request, mut stream, _tap) = terminal_test_request();
    for id in 0..2 {
        emit(&request, id, None);
    }
    request.set_terminal(GenerationEvent::Finished {
        cached_tokens: 3,
        generated_tokens: 2,
    });
    // A later cleanup must not overwrite the already committed terminal result.
    notify_request_error(request, WorkerError::worker_stopped("late stop"));
    for id in 0..2 {
        assert!(matches!(stream.events.recv().await,
            Some(GenerationEvent::Token { id: actual, .. }) if actual == id));
    }
    assert!(matches!(
        stream.events.recv().await,
        Some(GenerationEvent::Finished {
            cached_tokens: 3,
            generated_tokens: 2,
        })
    ));
    assert!(stream.events.recv().await.is_none());
    assert!(stream.events.recv().await.is_none());
}

#[tokio::test]
async fn terminal_delivery_wakes_a_waiting_receiver() {
    let (request, mut stream, _tap) = terminal_test_request();
    let receive = stream.events.recv();
    tokio::pin!(receive);
    // Poll recv while the channel is empty but still open, then terminate it.
    assert!(
        std::future::poll_fn(|cx| {
            assert!(receive.as_mut().poll(cx).is_pending());
            std::task::Poll::Ready(true)
        })
        .await
    );
    notify_request_error(request, WorkerError::worker_stopped("stop while waiting"));
    assert!(matches!(
        tokio::time::timeout(std::time::Duration::from_secs(1), receive).await,
        Ok(Some(GenerationEvent::Error(WorkerError::WorkerStopped(_))))
    ));
}

#[test]
fn supervisor_publishes_poison_before_borrowed_state_teardown() {
    struct TeardownProbe<'a> {
        panicked: &'a AtomicBool,
        reason: &'a Mutex<Option<String>>,
        touched: bool,
    }
    impl Drop for TeardownProbe<'_> {
        fn drop(&mut self) {
            assert!(self.touched);
            assert!(!std::thread::panicking());
            assert!(self.panicked.load(Ordering::Acquire));
            assert_eq!(
                self.reason.lock().unwrap().as_deref(),
                Some("borrowed state panic")
            );
        }
    }
    let panicked = AtomicBool::new(false);
    let reason = Mutex::new(None);
    let mut engine = TeardownProbe {
        panicked: &panicked,
        reason: &reason,
        touched: false,
    };
    let (_sender, mut receiver) = mpsc::channel(2);
    let mut requests = HashMap::new();
    // Mirror run(): borrow state across the caught closure, never move it in.
    run_supervised_loop(
        &mut receiver,
        &mut requests,
        &WorkerStats::default(),
        &panicked,
        &reason,
        |_, _| {
            let borrowed = &mut engine;
            borrowed.touched = true;
            panic!("borrowed state panic");
        },
    );
    drop(engine);
}

#[tokio::test]
async fn terminal_panic_survives_full_token_queue() {
    let (request, mut stream, tap) = terminal_test_request();
    for id in 0..2 {
        request
            .output
            .try_send(GenerationEvent::Token { id, logprobs: None })
            .unwrap();
    }
    let (_sender, mut receiver) = mpsc::channel(2);
    let mut requests = HashMap::from([(1, request)]);
    run_supervised_loop(
        &mut receiver,
        &mut requests,
        &WorkerStats::default(),
        &AtomicBool::new(false),
        &Mutex::new(None),
        |_, _| panic!("full queue panic"),
    );
    for id in 0..2 {
        assert!(
            matches!(stream.events.recv().await, Some(GenerationEvent::Token { id: actual, .. }) if actual == id)
        );
    }
    assert!(matches!(stream.events.recv().await,
        Some(GenerationEvent::Error(WorkerError::WorkerPanicked(msg))) if msg == "full queue panic"));
    assert!(
        matches!(tap.await, Ok(Err(WorkerError::WorkerPanicked(msg))) if msg == "full queue panic")
    );
    assert!(stream.events.recv().await.is_none());
}

#[tokio::test]
async fn terminal_normal_stop_drains_active_and_queued_decisions() {
    let (active, mut active_stream, active_tap) = terminal_test_request();
    for id in 0..2 {
        emit(&active, id, None);
    }
    let (queued, mut queued_stream, queued_tap) = terminal_test_request();
    let (sender, mut receiver) = mpsc::channel(2);
    assert!(sender.try_send(queued).is_ok());
    let mut requests = HashMap::from([(1, active)]);
    let stats = WorkerStats::default();
    stats.active.store(1, Ordering::Relaxed);
    let panicked = AtomicBool::new(false);
    let reason = Mutex::new(None);
    run_supervised_loop(
        &mut receiver,
        &mut requests,
        &stats,
        &panicked,
        &reason,
        |_, _| {},
    );
    assert!(sender.is_closed(), "normal stop must close ingress");
    assert!(requests.is_empty());
    assert_eq!(stats.active.load(Ordering::Relaxed), 0);
    assert!(!panicked.load(Ordering::Acquire));
    assert!(reason.lock().unwrap().is_none());
    for id in 0..2 {
        assert!(matches!(active_stream.events.recv().await,
            Some(GenerationEvent::Token { id: actual, .. }) if actual == id));
    }
    for stream in [&mut active_stream, &mut queued_stream] {
        match stream.events.recv().await {
            Some(GenerationEvent::Error(err @ WorkerError::WorkerStopped(_))) => assert_eq!(
                err.http_status(),
                axum::http::StatusCode::SERVICE_UNAVAILABLE
            ),
            other => panic!("expected WorkerStopped, got {other:?}"),
        }
        assert!(stream.events.recv().await.is_none());
    }
    for tap in [active_tap, queued_tap] {
        assert!(matches!(tap.await, Ok(Err(WorkerError::WorkerStopped(_)))));
    }
}

async fn check_locally_owned_panic(pulled_batch: bool) {
    let (mut request, mut stream, tap) = terminal_test_request();
    let telemetry = Arc::new(Telemetry::default());
    request.lifecycle = Some(telemetry.request(1));
    let (sender, mut receiver) = mpsc::channel(2);
    let mut requests = HashMap::new();
    if pulled_batch {
        requests.insert(1, request);
    } else {
        assert!(sender.try_send(request).is_ok());
    }
    run_supervised_loop(
        &mut receiver,
        &mut requests,
        &WorkerStats::default(),
        &AtomicBool::new(false),
        &Mutex::new(None),
        |rx, reqs| {
            if pulled_batch {
                // Exercise the heap-owned batch shape used by MTP verification.
                #[allow(clippy::useless_vec)]
                let _pulled = vec![(0, 1, Vec::<i32>::new(), reqs.remove(&1).unwrap())];
                panic!("panic while MTP owns pulled requests");
            } else {
                let _admitting = rx.try_recv().unwrap();
                panic!("panic during admission before registry insert");
            }
        },
    );
    assert!(matches!(stream.events.recv().await,
        Some(GenerationEvent::Error(WorkerError::WorkerPanicked(msg))) if !msg.is_empty()));
    assert!(matches!(tap.await, Ok(Err(WorkerError::WorkerPanicked(msg))) if !msg.is_empty()));
    assert!(stream.events.recv().await.is_none());
    let s = telemetry.snapshot();
    assert_eq!(s.requests.failed, 1);
    assert_eq!(s.requests.waiting + s.requests.running, 0);
    assert_eq!(s.latency.e2e.count, 1);
}

#[tokio::test]
async fn terminal_panic_during_admission() {
    check_locally_owned_panic(false).await;
}

#[tokio::test]
async fn terminal_panic_while_mtp_requests_are_pulled() {
    check_locally_owned_panic(true).await;
}

#[test]
fn request_length_limit_includes_output_and_handles_overflow() {
    assert!(validate_request_length(8, 2, 10).is_ok());
    assert!(validate_request_length(10, 0, 10).is_ok()); // Decision-only prefill.
    for (prompt, output) in [(9, 2), (11, 0), (usize::MAX, 1)] {
        let err = validate_request_length(prompt, output, 10).unwrap_err();
        assert_eq!(err.http_status(), axum::http::StatusCode::BAD_REQUEST);
        assert!(err.to_string().contains("context limit"));
    }
}

#[test]
fn terminal_host_compute_error_closes_ingress_before_model_teardown() {
    let failures = crate::cuda::HostFailures::default();
    let slot = failures.register();
    require_healthy_host_compute([&failures]);
    slot.set(crate::cuda::CudaError::with_detail(
        -1,
        "CPU MoE bridge",
        "injected failure",
    ))
    .unwrap();
    let (_sender, mut receiver) = mpsc::channel(1);
    let mut requests = HashMap::new();
    let stats = WorkerStats::default();
    let panicked = AtomicBool::new(false);
    let reason = Mutex::new(None);
    run_supervised_loop(
        &mut receiver,
        &mut requests,
        &stats,
        &panicked,
        &reason,
        |_, _| {
            require_healthy_host_compute([&failures]);
        },
    );
    assert!(panicked.load(Ordering::Acquire));
    assert!(receiver.is_closed());
    let reason = reason.lock().unwrap();
    assert!(reason.as_ref().unwrap().contains("restart engine"));
    assert!(reason.as_ref().unwrap().contains("injected failure"));
    assert!(
        failures.check().is_err(),
        "must not silently reset invalid model state"
    );
}

#[test]
fn worker_panic_containment_drains_in_flight_and_queued_and_poisons_future_submits() {
    let (sender, mut receiver) = mpsc::channel(32);
    let shutdown = Arc::new(AtomicBool::new(false));
    let panicked = Arc::new(AtomicBool::new(false));
    let panic_reason = Arc::new(Mutex::new(None));
    let stats = Arc::new(WorkerStats::default());
    stats.active.store(2, Ordering::Relaxed);

    let worker_panicked = panicked.clone();
    let worker_panic_reason = panic_reason.clone();
    let worker_stats = stats.clone();
    let (ingested_tx, ingested_rx) = std::sync::mpsc::sync_channel(1);
    let (unblock_tx, unblock_rx) = std::sync::mpsc::sync_channel(1);

    let thread = std::thread::Builder::new()
        .name("w34-cuda-test".into())
        .spawn(move || {
            let mut requests = HashMap::new();
            run_supervised_loop(
                &mut receiver,
                &mut requests,
                &worker_stats,
                &worker_panicked,
                &worker_panic_reason,
                |rx, reqs| {
                    let first = rx.blocking_recv().expect("first request");
                    reqs.insert(1, first);
                    ingested_tx.send(()).unwrap();
                    unblock_rx.recv().unwrap();
                    panic!("simulated scheduler invariant breach");
                },
            );
        })
        .unwrap();

    let worker = EngineWorker {
        sender: Mutex::new(Some(sender)),
        thread: Mutex::new(Some(thread)),
        shutdown,
        panicked,
        panic_reason,
        stats,
        tokenizer: None,
        eos_token_ids: vec![2],
        vocab_size: 1024,
        default_preserve_thinking: false,
        decision_profile: crate::decision::DecisionModelProfile::CausalLm { temperature: 1.0 },
        vision_config: None,
        has_vision_weights: false,
        hidden_size: 1024,
        kv_token_capacity: 4096,
        max_model_len: None,
    };

    let mut active_stream = worker
        .submit(vec![10, 20], SamplingParams::default(), None)
        .expect("submit active request");
    ingested_rx.recv().unwrap();

    // Enqueue a second request with a decision tap sitting in `receiver` (not yet dequeued).
    let (mut queued_stream, queued_tap_rx) = worker
        .submit_decision_candidate_logits(vec![30, 40], vec![1, 2], None)
        .expect("submit queued decision request");

    unblock_tx.send(()).unwrap();

    let rt = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .unwrap();
    rt.block_on(async {
        match active_stream.events.recv().await {
            Some(GenerationEvent::Error(WorkerError::WorkerPanicked(msg))) => {
                assert_eq!(msg, "simulated scheduler invariant breach");
            }
            other => panic!("expected WorkerPanicked on active stream, got {other:?}"),
        }

        match queued_stream.events.recv().await {
            Some(GenerationEvent::Error(WorkerError::WorkerPanicked(msg))) => {
                assert_eq!(msg, "simulated scheduler invariant breach");
            }
            other => panic!("expected WorkerPanicked on queued stream, got {other:?}"),
        }

        match queued_tap_rx.await {
            Ok(Err(WorkerError::WorkerPanicked(msg))) => {
                assert_eq!(msg, "simulated scheduler invariant breach");
            }
            other => panic!("expected WorkerPanicked on decision tap, got {other:?}"),
        }

        assert!(worker.is_panicked());
        assert_eq!(worker.stats.active.load(Ordering::Relaxed), 0);
        let metrics = worker.stats.telemetry.snapshot();
        assert_eq!(metrics.requests.submitted, 2);
        assert_eq!(metrics.requests.failed, 2);
        assert_eq!(metrics.requests.waiting + metrics.requests.running, 0);
        assert_eq!(metrics.latency.e2e.count, 2);
        assert_eq!(metrics.latency.ttft.count, 0);
        assert_eq!(metrics.status, "unhealthy");

        // Future submits must report WorkerPanicked (HTTP 500), never Overload/503.
        let post_panic = worker.submit(vec![1, 2], SamplingParams::default(), None);
        match post_panic {
            Err(err @ WorkerError::WorkerPanicked(_)) => {
                assert_eq!(
                    err.http_status(),
                    axum::http::StatusCode::INTERNAL_SERVER_ERROR
                );
            }
            Ok(_) => panic!("expected submit after panic to fail"),
            Err(other) => panic!("expected WorkerPanicked on post-panic submit, got {other:?}"),
        }

        let shutdown_err = worker
            .shutdown()
            .await
            .expect_err("shutdown must surface worker panic");
        assert!(
            shutdown_err.contains("simulated scheduler invariant breach"),
            "{shutdown_err}"
        );
    });
}
