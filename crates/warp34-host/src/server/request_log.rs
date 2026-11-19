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
    messages: Option<usize>,
    tools: Option<usize>,
    enable_thinking: Option<bool>,
    reasoning_effort: Option<String>,
    reasoning_budget_tokens: Option<usize>,
    thinking_budget_exhausted: Option<bool>,
    sampling: Option<Value>,
    start: Instant,
    first: Option<Instant>,
    last: Option<Instant>,
    tokens: u64,
    reasoning_tokens: Option<u64>,
    tool_calls: Option<usize>,
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
            messages: None,
            tools: None,
            enable_thinking: None,
            reasoning_effort: None,
            reasoning_budget_tokens: None,
            thinking_budget_exhausted: None,
            sampling: None,
            start: Instant::now(),
            first: None,
            last: None,
            tokens: 0,
            reasoning_tokens: None,
            tool_calls: None,
            cached: None,
            cache_lookup: None,
            finished: false,
        }
    }
    pub(super) fn with_turn_metadata(
        mut self,
        messages: Option<usize>,
        tools: Option<usize>,
        enable_thinking: Option<bool>,
        reasoning_effort: Option<String>,
        reasoning_budget_tokens: Option<usize>,
        params: &crate::parser::SamplingParams,
    ) -> Self {
        self.messages = messages;
        self.tools = tools;
        self.enable_thinking = enable_thinking;
        self.reasoning_effort = reasoning_effort;
        self.reasoning_budget_tokens = reasoning_budget_tokens;
        self.thinking_budget_exhausted = reasoning_budget_tokens.map(|_| false);
        if enable_thinking.is_some() {
            self.reasoning_tokens = Some(0);
            self.tool_calls = Some(0);
        }
        fn f32_json(x: f32) -> f64 {
            x.to_string().parse::<f64>().unwrap_or(f64::from(x))
        }
        self.sampling = Some(json!({
            "temperature": f32_json(params.temperature),
            "top_p": f32_json(params.top_p),
            "top_k": params.top_k,
            "min_p": f32_json(params.min_p),
            "seed": params.seed,
        }));
        self
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
    pub(super) fn reasoning_token(&mut self) {
        *self.reasoning_tokens.get_or_insert(0) += 1;
    }
    pub(super) fn set_tool_calls(&mut self, count: usize) {
        self.tool_calls = Some(count);
    }
    pub(super) fn mark_thinking_budget_exhausted(&mut self) {
        if self.thinking_budget_exhausted == Some(true) {
            return;
        }
        self.thinking_budget_exhausted = Some(true);
        emit(&json!({
            "event": "inference.thinking_budget_exhausted",
            "level": "info",
            "timestamp_unix_ms": timestamp_ms(),
            "request_id": self.id,
            "api": self.api,
            "model": self.model,
            "reasoning_effort": self.reasoning_effort,
            "reasoning_budget_tokens": self.reasoning_budget_tokens,
            "reasoning_tokens": self.reasoning_tokens.map(|r| r.min(self.tokens)),
        }));
    }
    pub(super) fn set_turn_outcome(&mut self, reasoning_tokens: Option<u64>, tool_calls: usize) {
        if let Some(r) = reasoning_tokens {
            self.reasoning_tokens = Some(r.min(self.tokens));
        }
        self.tool_calls = Some(tool_calls);
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
            "messages":self.messages, "tools":self.tools, "enable_thinking":self.enable_thinking,
            "reasoning_effort":self.reasoning_effort,
            "reasoning_budget_tokens":self.reasoning_budget_tokens,
            "thinking_budget_exhausted":self.thinking_budget_exhausted,
            "sampling":self.sampling,
            "prompt_tokens":self.prompt, "cached_prompt_tokens":self.cached.as_ref().map(|c| c.load(Ordering::Acquire)),
            "cache_lookup":self.cache_lookup.as_ref().and_then(|lookup| lookup.get()),
            "output_tokens":self.tokens, "reasoning_tokens":self.reasoning_tokens.map(|r| r.min(self.tokens)),
            "tool_calls":self.tool_calls,
            "duration_ms":(now-self.start).as_secs_f64()*1000.,
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

    #[test]
    fn records_request_shape_sampling_and_reasoning_tool_counts() {
        let params = crate::parser::SamplingParams {
            temperature: 1.0,
            top_p: 0.95,
            top_k: 20,
            min_p: 0.05,
            seed: 42,
            ..Default::default()
        };
        let mut log = TurnLog::new(
            "chatcmpl-0123456789abcdef0123456789abcdef".into(),
            "chat",
            "qwen3.8-27b-warp34".into(),
            true,
            13269,
            Some(65536),
            65536,
            262144,
        )
        .with_turn_metadata(
            Some(5),
            Some(14),
            Some(true),
            Some("medium".into()),
            Some(16384),
            &params,
        );

        let t0 = log.start + Duration::from_millis(100);
        log.token_at(t0);
        log.reasoning_token();
        log.token_at(t0 + Duration::from_millis(10));
        log.reasoning_token();
        log.token_at(t0 + Duration::from_millis(20));
        log.set_tool_calls(1);

        let row = log.record("tool_calls", false, t0 + Duration::from_millis(30));
        assert_eq!(row["reasoning_effort"], "medium");
        assert_eq!(row["reasoning_budget_tokens"], 16384);
        assert_eq!(row["thinking_budget_exhausted"], false);
        log.mark_thinking_budget_exhausted();
        let exhausted_row = log.record("tool_calls", false, t0 + Duration::from_millis(30));
        assert_eq!(exhausted_row["thinking_budget_exhausted"], true);
        assert_eq!(row["messages"], 5);
        assert_eq!(row["tools"], 14);
        assert_eq!(row["enable_thinking"], true);
        assert_eq!(row["sampling"]["temperature"], 1.0);
        assert_eq!(row["sampling"]["top_p"], 0.95);
        assert_eq!(row["sampling"]["top_k"], 20);
        assert_eq!(row["sampling"]["seed"], 42);
        assert_eq!(row["output_tokens"], 3);
        assert_eq!(row["reasoning_tokens"], 2);
        assert_eq!(row["tool_calls"], 1);
    }
}
