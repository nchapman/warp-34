"""Focused CPU-only soak tests, run from benchmarks/perf with unittest discovery."""

from concurrent.futures import Future
from contextlib import ExitStack
import copy
import hashlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import tempfile
import threading
import time
import unittest
from unittest.mock import patch

from w34perf import soak


def config():
    return {"engines": [{"id": "local", "model": "fake", "base_url": "http://127.0.0.1:1/v1"}],
            "run": {"max_inflight": 4, "timeout_s": 1, "warmup_requests": 0},
            "cases": [{"id": "text", "concurrency": 1, "max_tokens": 8,
                       "fixture_data": {"prompts": ["Hello"]}}]}


def success():
    return {"status": "ok", "content": "answer", "reasoning": "", "tool_calls": {},
            "usage": {"prompt_tokens": 3, "completion_tokens": 2}, "finish_reason": "stop",
            "first_output_ms": 1, "last_output_ms": 3, "e2e_ms": 4, "events": []}


class Clock:
    def __init__(self):
        self.now = 10.0

    def monotonic(self):
        return self.now

    def advance(self, seconds):
        self.now += seconds


class InlineExecutor:
    """Deterministic worker scheduling; actual concurrency has a separate test."""
    def __init__(self, **kwargs):
        pass

    def submit(self, function, *args):
        future = Future()
        try:
            future.set_result(function(*args))
        except BaseException as exc:
            future.set_exception(exc)
        return future

    def shutdown(self, **kwargs):
        pass


class Sampler:
    samples = None

    def __init__(self, path, **kwargs):
        self.path = path
        self.index = 0

    def __enter__(self):
        self.path.write_text('{"type":"mock_telemetry"}\n')
        return self

    def sample(self):
        rows = self.samples or [{"timestamp_monotonic_s": 1, "pid_rss_kib": {},
                                 "gpus": [], "errors": []}]
        row = rows[min(self.index, len(rows) - 1)]
        self.index += 1
        return copy.deepcopy(row)

    def stop(self):
        pass


class SoakTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.out = Path(self.tmp.name) / "result"
        self.cfg = config()
        self.clock = Clock()
        self.calls = []

    def run_soak(self, responder=None, *, controlled=True, **options):
        defaults = dict(duration_s=.5, min_completed=2, max_duration_s=4,
                        sustained_s=.5, burst_s=.5, progress_timeout_s=2, drain_s=0)
        defaults.update(options)

        def transport(engine, payload, timeout_s, **kwargs):
            self.calls.append((copy.deepcopy(payload), timeout_s, kwargs))
            if controlled:
                self.clock.advance(min(.25, timeout_s))
            else:
                time.sleep(.003)
            return responder(payload, kwargs) if responder else success()

        with ExitStack() as stack:
            stack.enter_context(patch.object(soak, "provenance", return_value={"source": "test"}))
            stack.enter_context(patch.object(soak, "TelemetrySampler", Sampler))
            stack.enter_context(patch.object(soak, "stream", side_effect=transport))
            if controlled:
                stack.enter_context(patch.object(soak, "ThreadPoolExecutor", InlineExecutor))
                stack.enter_context(patch.object(soak.time, "monotonic", self.clock.monotonic))
                stack.enter_context(patch.object(soak.time, "sleep", self.clock.advance))
            return soak.execute_soak(self.cfg, self.out, "local", **defaults)

    def rows(self, kind=None):
        rows = [json.loads(line) for line in (self.out / "requests.jsonl").read_text().splitlines()]
        return [r for r in rows if kind is None or r["type"] == kind]

    def assert_inventory(self):
        inventory = json.loads((self.out / "SHA256SUMS.json").read_text())
        self.assertEqual(set(inventory), {"manifest.json", "requests.jsonl", "telemetry.jsonl", "soak_summary.json"})
        for name, sha in inventory.items():
            self.assertEqual(sha, hashlib.sha256((self.out / name).read_bytes()).hexdigest())

    def test_both_targets_and_separate_probe(self):
        receipt = self.run_soak(min_completed=5, duration_s=.25)
        self.assertEqual(receipt["status"], "passed")
        self.assertEqual(receipt["counts"]["completed_requests"], 5)
        self.assertGreaterEqual(receipt["workload_duration_s"], 1.25)
        self.assertEqual(receipt["counts"]["probe_requests"], 1)
        self.assert_inventory()

    def test_duration_is_not_replaced_by_count(self):
        receipt = self.run_soak(duration_s=2, min_completed=1)
        self.assertEqual(receipt["status"], "passed")
        self.assertEqual(receipt["workload_duration_s"], 2)
        self.assertEqual(receipt["counts"]["completed_requests"], 8)
        phases = self.rows("phase")
        self.assertEqual([p["phase"] for p in phases], ["sustained", "burst", "sustained", "burst"])
        self.assertEqual([p["cycle"] for p in phases], [0, 0, 1, 1])

    def test_duration_can_equal_hard_deadline(self):
        receipt = self.run_soak(duration_s=.5, max_duration_s=.5)
        self.assertEqual(receipt["status"], "passed")

    def test_absolute_deadline_and_timeout_cap(self):
        receipt = self.run_soak(duration_s=.1, max_duration_s=.6, min_completed=100)
        self.assertIn("max_duration_exceeded", receipt["failure_reasons"])
        # Third measured request has only the remaining hard-deadline allowance.
        self.assertAlmostEqual(self.calls[2][1], .1)
        self.assertLessEqual(receipt["workload_duration_s"], .600001)
        self.assert_inventory()

    def test_watchdog_uses_completions_not_expected_cancel(self):
        def cancelled(payload, kwargs):
            if kwargs["cancel_after_output_events"]:
                return {"status": "cancelled", "cancellation_requested": True, "cancellation_observed": True}
            return success()
        receipt = self.run_soak(cancelled, cancel_every=1, progress_timeout_s=.5)
        self.assertIn("completion_progress_timeout", receipt["failure_reasons"])
        self.assertEqual(receipt["counts"]["completed_requests"], 0)
        self.assertEqual(receipt["counts"]["expected_cancellations"], 2)
        self.assertEqual(receipt["counts"]["unexpected_errors"], 0)
        self.assertEqual(receipt["probe"]["status"], "ok")

    def test_cancellation_and_completed_before_threshold(self):
        def cancelled(payload, kwargs):
            if kwargs["cancel_after_output_events"]:
                return {"status": "cancelled", "cancellation_requested": True, "cancellation_observed": True}
            return success()
        receipt = self.run_soak(cancelled, cancel_every=2, min_completed=3)
        self.assertEqual(receipt["status"], "passed")
        self.assertEqual(receipt["counts"]["completed_requests"], 3)
        self.assertEqual(receipt["counts"]["expected_cancellations"], 2)
        self.assertEqual(receipt["counts"]["injection_requested"], 2)
        self.assertEqual(receipt["counts"]["completion_tokens"], 6)
        self.out = Path(self.tmp.name) / "early"
        receipt = self.run_soak(cancel_every=1)
        self.assertEqual(receipt["status"], "passed")
        self.assertEqual(receipt["counts"]["completed_before_injection"], 2)
        self.assertEqual(receipt["counts"]["expected_cancellations"], 0)

    def test_invalid_usage_is_error_not_progress(self):
        def incomplete(payload, kwargs):
            row = success()
            row["usage"] = {"prompt_tokens": 1}  # Completion usage is mandatory.
            return row
        receipt = self.run_soak(incomplete, progress_timeout_s=.5)
        self.assertEqual(receipt["counts"]["completed_requests"], 0)
        self.assertEqual(receipt["counts"]["invalid_usage"], 2)
        self.assertIn("max_error_rate_exceeded", receipt["failure_reasons"])
        self.assertIn("drain_probe_failed", receipt["failure_reasons"])
        self.assert_inventory()

    def test_error_allowance_and_slo_gate(self):
        def once(payload, kwargs):
            return {"status": "error", "error": "SECRET credential"} if len(self.calls) == 1 else success()
        receipt = self.run_soak(once, max_error_rate=.5)
        self.assertEqual(receipt["status"], "passed")
        self.assertAlmostEqual(receipt["unexpected_error_rate"], 1 / 3)
        self.assertNotIn("SECRET", (self.out / "requests.jsonl").read_text())
        self.out = Path(self.tmp.name) / "slo"
        self.cfg["cases"][0]["ttft_slo_ms"] = .5
        receipt = self.run_soak()
        self.assertIn("case_slo_failed", receipt["failure_reasons"])
        self.assertEqual(receipt["case_slo_breaches"]["text"]["ttft_slo_ms"], 2)

    def test_warmups_once_per_case_rotation_unique_namespaces(self):
        other = copy.deepcopy(self.cfg["cases"][0])
        other["id"] = "other"
        self.cfg["cases"].append(other)
        self.cfg["run"]["warmup_requests"] = 2
        receipt = self.run_soak(min_completed=4)
        self.assertEqual(receipt["counts"]["warmup_requests"], 4)
        rows = self.rows("request")
        self.assertEqual([r["case_id"] for r in rows if r["phase"] in ("sustained", "burst")],
                         ["text", "other", "text", "other"])
        self.assertEqual(len({r["request_id"] for r in rows}), len(rows))
        self.assertEqual(len({r["request"]["messages"][0]["content"] for r in rows}), len(rows))
        self.assertEqual(receipt["workload_duration_s"], 1)
        old_tags = {r["request"]["messages"][0]["content"] for r in rows}
        self.out = Path(self.tmp.name) / "second"
        self.run_soak()
        self.assertTrue(old_tags.isdisjoint({r["request"]["messages"][0]["content"] for r in self.rows("request")}))

    def test_warmup_failure_preserves_receipt(self):
        self.cfg["run"]["warmup_requests"] = 2
        receipt = self.run_soak(lambda p, k: {"status": "timeout"})
        self.assertIn("warmup_failed", receipt["failure_reasons"])
        self.assertEqual(receipt["counts"]["requests"], 0)
        self.assert_inventory()

    def test_drain_then_non_cancelled_probe_failure(self):
        def respond(payload, kwargs):
            if len(self.calls) == 3:
                self.assertEqual(self.clock.now, 11.75)  # two requests, drain=1, probe=.25
                self.assertIsNone(kwargs["cancel_after_output_events"])
                return {"status": "timeout"}
            return success()
        receipt = self.run_soak(respond, drain_s=1, cancel_every=1)
        self.assertIn("drain_probe_failed", receipt["failure_reasons"])
        self.assertTrue(receipt["targets_met"])
        self.assertEqual(receipt["counts"]["completed_requests"], 2)

    def test_resource_missing_fails_closed_and_uuid_filtered(self):
        row = {"pid_rss_kib": {"123": 1024}, "gpus": [{"uuid": "unrelated", "memory_used_mib": 5}]}
        with patch.object(Sampler, "samples", [row, row]):
            receipt = self.run_soak(server_pids=(123,), gpu_uuids=("selected",),
                                    max_rss_growth_mib=0, max_gpu_growth_mib=0)
        self.assertIn("gpu_growth_missing_or_exceeded", receipt["failure_reasons"])
        self.assertEqual(receipt["baseline"]["gpus"], [])
        self.assertEqual(receipt["resource_growth"][0]["status"], "passed")
        self.assertIsNone(receipt["resource_growth"][1]["growth_mib"])
        for value in (None, float("nan"), float("inf"), True):
            rows, failures = soak._growth({"pid_rss_kib": {"1": value}},
                                         {"pid_rss_kib": {"1": 1024}}, (1,), (), 0, None)
            self.assertTrue(failures)
            self.assertIsNone(rows[0]["growth_mib"])

    def test_nonfinite_resource_sample_retains_failure_receipt(self):
        row = {"pid_rss_kib": {"123": float("nan")},
               "gpus": [{"uuid": "selected", "memory_used_mib": float("inf")}]}
        with patch.object(Sampler, "samples", [row]):
            receipt = self.run_soak(server_pids=(123,), gpu_uuids=("selected",),
                                    max_rss_growth_mib=0, max_gpu_growth_mib=0)
        self.assertEqual(receipt["status"], "failed")
        self.assertIsNone(receipt["baseline"]["pid_rss_kib"]["123"])
        self.assertIsNone(receipt["baseline"]["gpus"][0]["memory_used_mib"])
        self.assert_inventory()

    def test_telemetry_writer_failure_is_a_gate(self):
        with patch.object(Sampler, "stop", side_effect=RuntimeError("SECRET")):
            receipt = self.run_soak()
        self.assertIn("telemetry_failed", receipt["failure_reasons"])
        self.assertNotIn("SECRET", (self.out / "soak_summary.json").read_text())
        self.assert_inventory()

    def test_resource_growth_is_per_identity(self):
        baseline = {"pid_rss_kib": {"1": 2048, "2": 4096},
                    "gpus": [{"uuid": "a", "memory_used_mib": 10}]}
        final = {"pid_rss_kib": {"1": 4096, "2": 1024},
                 "gpus": [{"uuid": "a", "memory_used_mib": 11}]}
        rows, failures = soak._growth(baseline, final, (1, 2), ("a",), 1, 1)
        self.assertEqual([r["status"] for r in rows], ["failed", "passed", "passed"])
        self.assertEqual(failures, ["rss_growth_missing_or_exceeded"])

    def agent(self):
        self.cfg["cases"][0].update(kind="agent_replay", fixture_data={"system": "Replay",
                                    "turns": [{"user": "first"}, {"user": "next"}]})

    def test_replay_history_and_probe_one_turn(self):
        self.agent()
        receipt = self.run_soak(min_completed=2)
        self.assertEqual(receipt["status"], "passed")
        self.assertEqual(len(self.calls), 3)
        history = self.calls[1][0]["messages"]
        self.assertEqual(history[-2], {"role": "assistant", "content": "answer"})
        self.assertEqual(history[-1], {"role": "user", "content": "next"})

    def test_admitted_replay_finishes_after_targets(self):
        self.agent()
        self.cfg["cases"][0]["think_time_s"] = .025
        receipt = self.run_soak(controlled=False, duration_s=.001, min_completed=1,
                                max_duration_s=1)
        self.assertEqual(receipt["status"], "passed")
        measured = [r for r in self.rows("request") if r["phase"] != "probe"]
        self.assertEqual([r["step"] for r in measured], [0, 1])
        self.assertEqual(receipt["counts"]["stopped_sessions"], 0)

    def test_replay_error_stops_followup(self):
        self.agent()
        def live_tool(payload, kwargs):
            row = success()
            row["tool_calls"] = {"0": {"function": {"name": "not_replayed"}}}
            return row
        receipt = self.run_soak(live_tool, progress_timeout_s=.5)
        self.assertEqual(receipt["status"], "failed")
        self.assertTrue(all(r["step"] == 0 for r in self.rows("request")))
        self.assertTrue(all(r["status"] == "agent_replay_error" for r in self.rows("request")))

    def test_watchdog_wakes_agent_think_and_bounds_workers(self):
        self.agent()
        self.cfg["cases"][0]["think_time_s"] = 10
        before = time.monotonic()
        receipt = self.run_soak(controlled=False, duration_s=.02, min_completed=100,
                                max_duration_s=.5, progress_timeout_s=.05,
                                sustained_s=.01, burst_s=.01)
        self.assertLess(time.monotonic() - before, 2)
        self.assertEqual(receipt["status"], "failed")
        self.assertLessEqual(receipt["peak_inflight_sessions"], 2)
        self.assertGreater(receipt["counts"]["stopped_sessions"], 0)
        self.assertTrue(all(r["step"] == 0 for r in self.rows("request")))

    def test_actual_burst_concurrency_is_bounded(self):
        active, peak = 0, 0
        guard = threading.Lock()
        def slow(payload, kwargs):
            nonlocal active, peak
            with guard:
                active += 1
                peak = max(peak, active)
            time.sleep(.015)
            with guard:
                active -= 1
            return success()
        receipt = self.run_soak(slow, controlled=False, duration_s=.15, min_completed=4,
                                max_duration_s=2, sustained_s=.02, burst_s=.04)
        self.assertEqual(receipt["status"], "passed")
        self.assertEqual(peak, 2)
        self.assertEqual(receipt["peak_inflight_sessions"], 2)

    def test_interrupt_persists_receipt_and_reraises(self):
        def interrupt(payload, kwargs):
            raise KeyboardInterrupt()
        with self.assertRaises(KeyboardInterrupt):
            self.run_soak(interrupt)
        receipt = json.loads((self.out / "soak_summary.json").read_text())
        self.assertEqual(receipt["status"], "failed")
        self.assertTrue(receipt["interrupted"])
        self.assert_inventory()

    def test_unexpected_setup_failure_has_safe_receipt(self):
        with patch.object(soak, "provenance", side_effect=RuntimeError("SECRET")):
            receipt = soak.execute_soak(self.cfg, self.out, "local", duration_s=1,
                                        min_completed=1, max_duration_s=2)
        self.assertEqual(receipt["error_type"], "RuntimeError")
        self.assertNotIn("SECRET", (self.out / "soak_summary.json").read_text())
        self.assert_inventory()

    def test_validation_precedes_output_and_network(self):
        invalid = [{"duration_s": float("inf")}, {"duration_s": True}, {"min_completed": 1.5},
                   {"min_completed": 0}, {"burst_multiplier": 5}, {"max_duration_s": .1},
                   {"sustained_s": 0}, {"burst_s": float("nan")}, {"cancel_every": False},
                   {"cancel_after_output_events": 0}, {"progress_timeout_s": -1}, {"drain_s": -1},
                   {"max_error_rate": 2}, {"telemetry_interval": None}, {"server_pids": (True,)},
                   {"gpu_uuids": ("",)}, {"gpu_uuids": "GPU-one"}, {"server_pids": None},
                   {"max_gpu_growth_mib": 0}, {"max_rss_growth_mib": 0},
                   {"max_rss_growth_mib": float("nan")}, {"duration_s": "1"}]
        for bad in invalid:
            with self.subTest(bad=bad), patch.object(soak, "stream") as transport:
                options = dict(duration_s=1, min_completed=1, max_duration_s=2)
                options.update(bad)
                with self.assertRaises(ValueError):
                    soak.execute_soak(self.cfg, self.out, "local", **options)
                self.assertFalse(self.out.exists())
                transport.assert_not_called()
        for mutation in ("primed", "disabled"):
            with self.subTest(mutation=mutation):
                cfg = config()
                if mutation == "primed":
                    cfg["cases"][0]["cache"] = "primed"
                else:
                    cfg["engines"][0]["enabled"] = False
                with self.assertRaises(ValueError):
                    soak.execute_soak(cfg, self.out, "local", duration_s=1, min_completed=1, max_duration_s=2)
                self.assertFalse(self.out.exists())

    def test_local_sse_transport_smoke(self):
        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def do_POST(self):
                self.rfile.read(int(self.headers["Content-Length"]))
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.end_headers()
                events = [{"choices": [{"delta": {"content": "hi"}, "finish_reason": None}]},
                          {"choices": [{"delta": {}, "finish_reason": "stop"}],
                           "usage": {"prompt_tokens": 3, "completion_tokens": 1}}]
                for event in events:
                    self.wfile.write(("data: " + json.dumps(event) + "\n\n").encode())
                self.wfile.write(b"data: [DONE]\n\n")
                self.wfile.flush()
        server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            self.cfg["engines"][0]["base_url"] = f"http://127.0.0.1:{server.server_port}/v1"
            with patch.object(soak, "provenance", return_value={}), patch.object(soak, "TelemetrySampler", Sampler):
                receipt = soak.execute_soak(self.cfg, self.out, "local", duration_s=.03,
                                            min_completed=2, max_duration_s=3, progress_timeout_s=1,
                                            sustained_s=.01, burst_s=.01, drain_s=0)
            self.assertEqual(receipt["status"], "passed")
            self.assertGreaterEqual(receipt["counts"]["completed_requests"], 2)
            self.assert_inventory()
        finally:
            server.shutdown()
            server.server_close()
            thread.join()


if __name__ == "__main__":
    unittest.main()
