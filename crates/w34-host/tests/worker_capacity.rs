#![cfg(feature = "cuda")]
mod common;

use std::{sync::atomic::Ordering, time::Duration};
use w34_host::{
    cache::KvFormatPolicy,
    engine_worker::{EngineWorker, GenerationEvent},
    model::Qwen3_5Engine,
    parser::SamplingParams,
    scheduler::SchedulerConfig,
};

// Run under flock /tmp/w34-gpu1.lock with CUDA_VISIBLE_DEVICES=1.
#[test]
fn serving_sixteen_requests_reach_b16() {
    if common::require_or_skip_cached_model(
        "nchapman/Qwen3.5-4B-W4A16-GPTQ",
        "serving_sixteen_requests_reach_b16",
    )
    .is_none()
    {
        return;
    }
    let worker = EngineWorker::from_loader(|| {
        let engine = Qwen3_5Engine::load_for_serving("nchapman/Qwen3.5-4B-W4A16-GPTQ", 1, KvFormatPolicy::Tq4, Some(256), true, 16)
            .map_err(|e| e.to_string())?;
        let rank = &engine.ranks[0];
        let state_bytes: usize = rank.state_pool.layers.iter()
            .map(|l| l.conv_state.len_bytes + l.ssm_state.len_bytes).sum();
        let kv_bytes: usize = rank.kv_pool.layers.iter()
            .map(|l| l.k_buf.len_bytes + l.v_buf.len_bytes).sum();
        eprintln!("capacity resources: state_slots={} state_bytes={state_bytes} kv_pages={} kv_bytes={kv_bytes} weight_used={} weight_capacity={} scratch_capacity={} min_headroom={}",
            rank.state_pool.total_slots, rank.kv_pool.num_pages,
            rank.weight_arena.used(), rank.weight_arena.capacity(), rank.scratch_arena.capacity(),
            w34_host::cuda::MIN_HEADROOM_BYTES);
        Ok(engine)
    }, SchedulerConfig::default()).unwrap();
    tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .unwrap()
        .block_on(async {
            let mut streams = Vec::new();
            for i in 0..16 {
                streams.push(
                    worker
                        .submit(
                            vec![100 + i, 200, 300],
                            SamplingParams {
                                max_tokens: 8,
                                ignore_eos: true,
                                temperature: 0.0,
                                ..SamplingParams::default()
                            },
                            None,
                        )
                        .unwrap(),
                );
            }
            let result = tokio::time::timeout(Duration::from_secs(180), async {
                for stream in &mut streams {
                    let mut tokens = 0;
                    let mut finished = false;
                    while let Some(event) = stream.events.recv().await {
                        match event {
                            GenerationEvent::Token { .. } => tokens += 1,
                            GenerationEvent::Finished { .. } => {
                                finished = true;
                                break;
                            }
                            GenerationEvent::Error(e) => return Err(e.to_string()),
                        }
                    }
                    if !finished || tokens != 8 {
                        return Err(format!("finished={finished}, tokens={tokens}"));
                    }
                }
                Ok(())
            })
            .await;
            worker.shutdown().await.unwrap();
            result.unwrap().unwrap();
            let peak = worker.stats.peak_active.load(Ordering::Relaxed);
            let mask = worker.stats.decode_bucket_mask.load(Ordering::Relaxed);
            eprintln!("serving capacity: peak_active={peak}, decode_bucket_mask={mask:#x}");
            assert_eq!(peak, 16);
            assert_ne!(mask & (1 << 4), 0, "B16 must be captured");
        });
}

#[test]
fn invalid_serving_capacity_fails_before_model_resolution() {
    for max_num_seqs in [0, usize::MAX / 4 + 1, usize::MAX] {
        let error = Qwen3_5Engine::load_for_serving(
            "/nonexistent/invalid-capacity-must-not-resolve",
            1,
            KvFormatPolicy::Tq4,
            Some(256),
            true,
            max_num_seqs,
        )
        .err()
        .expect("invalid capacity must fail")
        .to_string();
        assert!(error.contains("serving max_num_seqs"), "{error}");
    }
}
