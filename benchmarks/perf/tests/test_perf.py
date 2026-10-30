import copy
import http.server
import json
from pathlib import Path
import tempfile
import threading
import time
import unittest
from unittest.mock import patch

from w34perf.config import digest, load
from w34perf.metrics import distribution, summarize, tokens, tpot
from w34perf.report import render
from w34perf.runner import execute, Journal, run_trial
from w34perf.transport import stream
from w34perf.workloads import body, messages

ROOT = Path(__file__).resolve().parents[1]


def result():
    return {"status": "ok", "error": None, "usage": {"prompt_tokens": 50, "completion_tokens": 10},
            "first_output_ms": 10, "first_text_ms": 12, "last_output_ms": 100, "e2e_ms": 110,
            "content": "actual response", "reasoning": "", "tool_calls": {},
            "finish_reason": "stop", "events": [], "schedule_lag_ms": 0}


def frame(data):
    return b"data: " + json.dumps(data).encode() + b"\n\n"


def completion(text="hello", finish=None):
    return {"choices": [{"index": 0, "delta": {"content": text}, "finish_reason": finish}]}


class FakeServer:
    def __init__(self, mode="normal"):
        self.requests = []
        outer = self

        class Handler(http.server.BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def do_POST(self):
                payload = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                outer.requests.append((self.path, payload))
                if mode == "http_error":
                    self.send_error(429)
                    return
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.end_headers()
                try:
                    if mode == "trickle":
                        for _ in range(200):
                            self.wfile.write(b": ping\n\n")
                            self.wfile.flush()
                            time.sleep(0.005)
                        return
                    if mode in ("nan", "infinity", "overflow", "surrogate"):
                        raw = {"nan": b'{"bad":NaN}', "infinity": b'{"bad":Infinity}',
                               "overflow": b'{"bad":1e999}', "surrogate": b'{"bad":"\\ud800"}'}[mode]
                        self.wfile.write(b"data: " + raw + b"\n\n")
                        self.wfile.flush()
                        return
                    # Multi-line data frame, comments, empty role-only event.
                    self.wfile.write(b": keepalive\n\n")
                    self.wfile.write(frame({"choices": [{"delta": {"role": "assistant"}}]}))
                    self.wfile.write(b'data: {"choices":\ndata: [{"delta":{"reasoning_content":"hmm"}}]}\n\n')
                    self.wfile.write(frame(completion("hello world")))
                    if mode == "sse_error":
                        self.wfile.write(frame({"error": {"message": "secret upstream"}}))
                        return
                    if mode != "truncated":
                        self.wfile.write(frame(completion("", "length")))
                        if mode != "no_usage":
                            self.wfile.write(frame({"choices": [], "usage": {"prompt_tokens": 23,
                                                                                  "completion_tokens": 100}}))
                        self.wfile.write(b"data: [DONE]\n\n")
                    self.wfile.flush()
                except (BrokenPipeError, ConnectionResetError):
                    pass

        self.server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.engine = {"id": "fake", "model": "fake", "base_url": f"http://127.0.0.1:{self.server.server_port}/v1"}

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join()


class ConfigTests(unittest.TestCase):
    def test_suite_has_all_five_engines_and_dual_card_profiles(self):
        cfg = load(ROOT / "suite.toml")
        self.assertEqual({e["engine"] for e in cfg["engines"]},
                         {"warp-34", "vllm", "sglang", "llama.cpp", "ninfer-3090"})
        for name in {e["engine"] for e in cfg["engines"]}:
            self.assertEqual({e["gpu_count"] for e in cfg["engines"] if e["engine"] == name}, {1, 2})

    def test_reject_invalid_boundaries_before_running(self):
        original = (ROOT / "suite.toml").read_text()
        for old, new in (("repetitions = 3", "repetitions = 0"),
                         ("concurrency = 1", "concurrency = -1"),
                         ("rate_rps = 4", "rate_rps = nan"),
                         ("gpu_count = 1", "gpu_count = true"),
                         ("seed = 34", 'seed = "34"'),
                         ('http://127.0.0.1:8000/v1', 'http://127.0.0.1:bad/v1'),
                         ("context_repetitions = 64", "context_repetitions = 0"),
                         ('cache = "primed"', 'cache = "cold"'),
                         ('http://127.0.0.1:8000/v1', 'http://user:secret@localhost/v1')):
            with self.subTest(new=new), tempfile.TemporaryDirectory(dir=ROOT) as tmp:
                path = Path(tmp) / "bad.toml"
                path.write_text(original.replace(old, new).replace('"fixtures/', '"../fixtures/'))
                with self.assertRaises(ValueError):
                    load(path)

    def test_unique_nonce_precedes_system_prefix_and_is_engine_independent(self):
        cfg = load(ROOT / "suite.toml")
        case = cfg["cases"][0]
        a, b = messages(case, 0, "measured"), messages(case, 1, "measured")
        self.assertNotEqual(a[0], b[0])
        self.assertTrue(a[0]["content"].startswith("Benchmark session:"))
        self.assertNotEqual(a[0], messages(case, 0, "warmup")[0])
        self.assertEqual(body(cfg["engines"][0], case, a)["messages"], a)

    def test_agent_replay_explicitly_disables_new_live_tool_calls(self):
        cfg = load(ROOT / "suite.toml")
        case = next(c for c in cfg["cases"] if c["id"] == "agent-1")
        payload = body(cfg["engines"][0], case, messages(case, 0, "test"))
        self.assertEqual(payload.get("tool_choice"), "none")

    def test_hash_canonicalizes_mapping_order(self):
        self.assertEqual(digest({"b": 2, "a": 1}), digest({"a": 1, "b": 2}))


class TransportTests(unittest.TestCase):
    def test_stream_counts_usage_not_chunks_and_preserves_wire_evidence(self):
        with FakeServer() as server:
            r = stream(server.engine, {"messages": []}, 2)
            self.assertEqual(r["status"], "ok", r)
            self.assertEqual(tokens(r), [23, 100])
            self.assertEqual(r["content"], "hello world")
            self.assertEqual(r["reasoning"], "hmm")
            self.assertLessEqual(r["first_output_ms"], r["first_text_ms"])
            self.assertEqual(server.requests[0][0], "/v1/chat/completions")
            self.assertLess(len(r["events"]), 100)

    def test_http_sse_and_truncated_failures(self):
        for mode, expected in (("http_error", "HTTP 429"), ("sse_error", "SSE error"),
                               ("truncated", "missing [DONE]")):
            with self.subTest(mode=mode), FakeServer(mode) as server:
                r = stream(server.engine, {}, 2)
                self.assertEqual(r["status"], "error")
                self.assertIn(expected, r["error"])
                self.assertNotIn("secret upstream", r["error"])

    def test_malformed_json_values_fail_one_request_without_poisoning_journal(self):
        for mode in ("nan", "infinity", "overflow", "surrogate"):
            with self.subTest(mode=mode), FakeServer(mode) as server:
                r = stream(server.engine, {}, 2)
                self.assertEqual(r["status"], "error")
                # Same serialization rules as the journal; malformed event is not retained.
                json.dumps(r, ensure_ascii=False, allow_nan=False).encode("utf-8")

    def test_malformed_credentials_cannot_leak_in_exception_text(self):
        for secret in ("sensitive\nkey", "sensitive\rkey", "sensitive-é"):
            with self.subTest(secret=secret), FakeServer() as server, \
                    patch.dict("os.environ", {"PERF_TEST_KEY": secret}):
                r = stream({**server.engine, "api_key_env": "PERF_TEST_KEY"}, {}, 2)
                self.assertEqual(r["status"], "error")
                self.assertNotIn("sensitive", json.dumps(r))

    def test_usage_missing_does_not_invent_tokens(self):
        with FakeServer("no_usage") as server:
            r = stream(server.engine, {}, 2)
            self.assertEqual(r["status"], "ok")
            self.assertIsNone(tokens(r))

    def test_total_deadline_even_when_server_keeps_sending(self):
        with FakeServer("trickle") as server:
            r = stream(server.engine, {}, 0.05)
            self.assertEqual(r["status"], "timeout")
            self.assertLess(r["e2e_ms"], 500)

    def test_credentials_are_not_returned(self):
        with FakeServer() as server, patch.dict("os.environ", {"PERF_TEST_KEY": "sensitive"}):
            r = stream({**server.engine, "api_key_env": "PERF_TEST_KEY"}, {}, 2)
            self.assertNotIn("sensitive", json.dumps(r))


class MetricsTests(unittest.TestCase):
    def test_usage_validation_and_single_burst_no_decode_estimate(self):
        r = result()
        self.assertEqual(tpot(r), 10)
        r["last_output_ms"] = r["first_output_ms"]
        self.assertIsNone(tpot(r))
        for usage in ({}, {"prompt_tokens": 1, "completion_tokens": True},
                      {"prompt_tokens": 1, "completion_tokens": -1},
                      {"prompt_tokens": 1, "completion_tokens": 1.5}):
            self.assertIsNone(tokens({"usage": usage}))

    def test_failed_requests_remain_in_denominator_and_slo_goodput(self):
        r = result()
        failed = {"status": "timeout", "request_id": "bad", "error": "timeout"}
        summary = summarize([r, failed], [{"status": "ok", "e2e_ms": 110},
                                        {"status": "error"}], 2, {"ttft_slo_ms": 5})
        self.assertEqual(summary["successful_requests_s"], 0.5)
        self.assertEqual(summary["completion_tokens_s"], 5)
        self.assertEqual(summary["slo_goodput_requests_s"], 0)
        self.assertEqual(summary["failed_requests"], 1)

    def test_missing_usage_invalidates_trial_throughput_not_latency(self):
        a, b = result(), result()
        b["usage"] = None
        summary = summarize([a, b], [], 1, {})
        self.assertIsNone(summary["completion_tokens_s"])
        self.assertEqual(summary["first_output_ms"]["n"], 2)

    def test_empty_distribution_and_linear_percentiles(self):
        self.assertIsNone(distribution([])["p99"])
        self.assertEqual(distribution([0, 10])["p95"], 9.5)


class RunnerTests(unittest.TestCase):
    def config(self):
        cfg = load(ROOT / "suite.toml")
        cfg["engines"] = cfg["engines"][:1]
        cfg["cases"] = cfg["cases"][:1]
        cfg["run"].update(repetitions=1, warmup_requests=0)
        return cfg

    def test_end_to_end_artifacts_and_report_against_real_http(self):
        cfg = self.config()
        with FakeServer() as server, tempfile.TemporaryDirectory() as tmp:
            cfg["engines"][0].update(server.engine)
            out = Path(tmp) / "results"
            with patch("w34perf.runner.provenance", return_value={"test": True}):
                rows = execute(cfg, out)
            self.assertEqual(rows[0]["successful_requests"], 2)
            self.assertEqual(json.loads((out / "manifest.json").read_text())["status"], "complete")
            raw = [json.loads(line) for line in (out / "requests.jsonl").read_text().splitlines()]
            self.assertEqual(sum(r["type"] == "request" for r in raw), 2)
            self.assertTrue((out / "SHA256SUMS.json").exists())
            md, csv = render(out)
            self.assertIn("4-bit deployment equivalents", md)
            self.assertIn("completion_tokens_s", csv)
            with self.assertRaises(FileExistsError):
                execute(cfg, out)

    def test_agent_replay_grows_history_with_actual_responses_and_recorded_tools(self):
        cfg = self.config()
        cfg["cases"] = [next(c for c in load(ROOT / "suite.toml")["cases"] if c["id"] == "agent-1")]
        cfg["cases"][0].update(requests=1, think_time_s=0)
        payloads = []

        def fake(engine, payload, timeout):
            payloads.append(copy.deepcopy(payload))
            return result()

        with tempfile.TemporaryDirectory() as tmp, patch("w34perf.runner.stream", side_effect=fake):
            journal = Journal(Path(tmp) / "raw")
            try:
                stats = run_trial(cfg["engines"][0], cfg["cases"][0], cfg["run"],
                                  "measured", 0, journal, namespace="test")
            finally:
                journal.close()
        self.assertEqual(len(payloads), 4)
        self.assertEqual(stats["successful_sessions"], 1)
        self.assertGreater(len(payloads[-1]["messages"]), len(payloads[0]["messages"]))
        self.assertTrue(any(m.get("content") == "actual response" for m in payloads[-1]["messages"]))
        self.assertTrue(any(m["role"] == "tool" for m in payloads[0]["messages"]))

    def test_open_loop_rejects_instead_of_silently_queuing(self):
        cfg = self.config()
        case = {**cfg["cases"][0], "requests": 20, "arrival": "fixed", "rate_rps": 10000}
        cfg["run"]["max_inflight"] = 1

        overload_seen = threading.Event()

        def slow(*args):
            self.assertTrue(overload_seen.wait(2), "scheduler never recorded overload")
            return result()

        with tempfile.TemporaryDirectory() as tmp, patch("w34perf.runner.stream", side_effect=slow):
            path = Path(tmp) / "raw"
            journal = Journal(path)
            original_put = journal.put

            def observe(entry):
                original_put(entry)
                if entry["status"] == "client_overload":
                    overload_seen.set()

            journal.put = observe
            try:
                stats = run_trial(cfg["engines"][0], case, cfg["run"], "measured", 0,
                                  journal, namespace="test")
            finally:
                journal.close()
            entries = [json.loads(line) for line in path.read_text().splitlines()]
            self.assertEqual(sum(e["type"] == "session" for e in entries), 20)
        self.assertEqual(stats["requests"], 20)
        self.assertGreater(stats["failed_requests"], 0)
        self.assertTrue(all(e["status"] == "client_overload" for e in stats["errors"]))

    def test_priming_and_warmup_are_separate_from_measured_requests(self):
        cfg = self.config()
        cfg["cases"][0]["cache"] = "primed"
        cfg["run"]["warmup_requests"] = 1
        with tempfile.TemporaryDirectory() as tmp, patch("w34perf.runner.stream", return_value=result()), \
                patch("w34perf.runner.provenance", return_value={}):
            out = Path(tmp) / "run"
            rows = execute(cfg, out)
            requests = [json.loads(line) for line in (out / "requests.jsonl").read_text().splitlines()
                        if json.loads(line)["type"] == "request"]
        phases = {phase: [r for r in requests if r["phase"] == phase]
                  for phase in ("warmup", "prime", "measured")}
        self.assertEqual([len(phases[p]) for p in phases], [1, 2, 2])
        self.assertEqual(rows[0]["requests"], 2)
        self.assertEqual(phases["prime"][0]["request_sha256"], phases["measured"][0]["request_sha256"])
        self.assertNotEqual(phases["warmup"][0]["request_sha256"], phases["measured"][0]["request_sha256"])

    def test_failed_warmup_leaves_failed_manifest_and_raw_evidence(self):
        cfg = self.config()
        cfg["run"]["warmup_requests"] = 1
        bad = {**result(), "status": "timeout", "request_id": "x", "error": "timeout"}
        with tempfile.TemporaryDirectory() as tmp, patch("w34perf.runner.stream", return_value=bad), \
                patch("w34perf.runner.provenance", return_value={}):
            out = Path(tmp) / "run"
            with self.assertRaises(RuntimeError):
                execute(cfg, out)
            self.assertEqual(json.loads((out / "manifest.json").read_text())["status"], "failed")
            self.assertTrue((out / "requests.jsonl").stat().st_size)
            markdown, _ = render(out)
            self.assertIn("**failed**", markdown)
            self.assertIn("No measured trial completed", markdown)

    def test_malformed_upstream_does_not_abort_the_campaign(self):
        cfg = self.config()
        with FakeServer("surrogate") as server, tempfile.TemporaryDirectory() as tmp, \
                patch("w34perf.runner.provenance", return_value={}):
            cfg["engines"][0].update(server.engine)
            rows = execute(cfg, Path(tmp) / "run")
            self.assertEqual(rows[0]["failed_requests"], 2)
            self.assertEqual(rows[0]["requests"], 2)


if __name__ == "__main__":
    unittest.main()
