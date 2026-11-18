//! One content-free JSON line per generation, including abandoned SSE bodies.
use crate::scheduler::CacheLookup;
use serde_json::{Value, json};
use std::{
    io::{self, Write},
    sync::{
        Arc, OnceLock,
        atomic::{AtomicU64, Ordering},
    },
    time::{Instant, SystemTime, UNIX_EPOCH},
};

fn emit(record: &Value) {
    // A single locked line prevents concurrent requests interleaving; a closed log
    // collector must not panic a serving task (especially during Drop).
    let _ = writeln!(io::stderr().lock(), "{record}");
}

pub(super) fn http_error(endpoint: &str, status: u16) {
    emit(
        &json!({"event":"inference.http_error", "level":if status >= 500 {"error"} else {"warn"},
        "timestamp_unix_ms":timestamp_ms(), "endpoint":endpoint, "http_status":status}),
    );
}

fn timestamp_ms() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis() as u64
}

pub(super) struct TurnLog {
    id: String,
    api: &'static str,
    model: String,
    stream: bool,
    prompt: usize,
    requested: Option<usize>,
    effective: usize,
    context: usize,
    start: Instant,
    first: Option<Instant>,
    last: Option<Instant>,
    tokens: u64,
    cached: Option<Arc<AtomicU64>>,
    cache_lookup: Option<Arc<OnceLock<CacheLookup>>>,
    finished: bool,
}

impl TurnLog {
    #[allow(clippy::too_many_arguments)]
    pub(super) fn new(
        id: String,
        api: &'static str,
        model: String,
        stream: bool,
        prompt: usize,
        requested: Option<usize>,
        effective: usize,
        context: usize,
    ) -> Self {
        Self {
            id,
            api,
            model,
            stream,
            prompt,
            requested,
            effective,
            context,
            start: Instant::now(),
            first: None,
            last: None,
            tokens: 0,
            cached: None,
            cache_lookup: None,
            finished: false,
        }
    }
    pub(super) fn track_cache(
        &mut self,
        cached: Arc<AtomicU64>,
        lookup: Arc<OnceLock<CacheLookup>>,
    ) {
        self.cache_lookup = Some(lookup);
        self.cached = Some(cached);
    }
    pub(super) fn token(&mut self) {
        self.token_at(Instant::now());
    }
    fn token_at(&mut self, now: Instant) {
        self.first.get_or_insert(now);
        self.last = Some(now);
        self.tokens += 1;
    }
    fn record(&self, reason: &'static str, hit_limit: bool, now: Instant) -> Value {
        let limit = matches!(reason, "length" | "max_tokens").then(|| {
            if hit_limit {
                "kv_capacity"
            } else if self.effective == self.context.saturating_sub(self.prompt) {
                "context_window"
            } else if self.requested.is_some() {
                "requested_output"
            } else {
                "default_output"
            }
        });
        let rate = self.first.zip(self.last).and_then(|(a, b)| {
            let seconds = (b - a).as_secs_f64();
            (seconds > 0.0).then(|| (self.tokens - 1) as f64 / seconds)
        });
        json!({"event":"inference.turn_end", "timestamp_unix_ms":timestamp_ms(), "level":if reason == "error" {"error"} else {"info"},
            "request_id":self.id, "api":self.api, "model":self.model, "stream":self.stream,
            "finish_reason":reason, "length_limit":limit, "context_window":self.context,
            "requested_max_output_tokens":self.requested, "effective_max_output_tokens":self.effective,
            "prompt_tokens":self.prompt, "cached_prompt_tokens":self.cached.as_ref().map(|c| c.load(Ordering::Acquire)),
            "cache_lookup":self.cache_lookup.as_ref().and_then(|lookup| lookup.get()),
            "output_tokens":self.tokens, "duration_ms":(now-self.start).as_secs_f64()*1000.,
            "ttft_ms":self.first.map(|t| (t-self.start).as_secs_f64()*1000.), "output_tokens_per_s":rate})
    }
    pub(super) fn finish(&mut self, reason: &'static str, hit_limit: bool) {
        if !self.finished {
            self.finished = true;
            emit(&self.record(reason, hit_limit, Instant::now()));
        }
    }
}
impl Drop for TurnLog {
    fn drop(&mut self) {
        self.finish(
            if std::thread::panicking() {
                "error"
            } else {
                "cancelled"
            },
            false,
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;

    #[test]
    fn distinguishes_output_context_and_runtime_capacity_limits() {
        for (requested, effective, capacity, expected) in [
            (Some(4), 4, false, "requested_output"),
            (Some(128), 32, false, "context_window"),
            (None, 32, false, "context_window"),
            (None, 4, false, "default_output"),
            (Some(128), 32, true, "kv_capacity"),
        ] {
            let log = TurnLog::new(
                "id".into(),
                "chat",
                "model".into(),
                true,
                32,
                requested,
                effective,
                64,
            );
            assert_eq!(
                log.record("length", capacity, log.start)["length_limit"],
                expected
            );
            assert!(log.record("stop", false, log.start)["length_limit"].is_null());
        }
    }

    #[test]
    fn reports_partial_usage_and_rate_excluding_first_token_latency() {
        let mut log = TurnLog::new(
            "id".into(),
            "chat",
            "model".into(),
            true,
            100,
            None,
            16,
            116,
        );
        let lookup = Arc::new(OnceLock::new());
        log.track_cache(Arc::new(AtomicU64::new(80)), lookup.clone());
        assert_eq!(
            log.record("cancelled", false, log.start)
                .get("cache_lookup"),
            Some(&Value::Null)
        );
        let snapshot = CacheLookup {
            restored_tokens: 80,
            ..Default::default()
        };
        lookup.set(snapshot).unwrap();
        assert!(log.record("cancelled", false, log.start)["ttft_ms"].is_null());
        let first = log.start + Duration::from_millis(200);
        log.token_at(first);
        assert!(log.record("stop", false, first)["output_tokens_per_s"].is_null());
        log.token_at(first + Duration::from_millis(10));
        let row = log.record("cancelled", false, first + Duration::from_secs(1));
        assert_eq!(row["ttft_ms"], 200.);
        assert_eq!(row["output_tokens_per_s"], 100.);
        assert_eq!(row["cache_lookup"], json!(snapshot));
        assert_eq!(row["cached_prompt_tokens"], 80);
        assert_eq!(row["output_tokens"], 2);
    }
}
