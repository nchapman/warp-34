"""Soak must honor workload/cache contracts, not silently change the traffic."""
import json
from pathlib import Path
import tempfile
import threading
import unittest
from unittest.mock import patch

from w34perf.soak import execute_soak


def configuration():
    return {
        "engines": [{"id": "fake", "model": "fake", "base_url": "http://127.0.0.1:1/v1"}],
        "run": {"max_inflight": 2, "timeout_s": 1, "warmup_requests": 0},
        "cases": [{"id": name, "concurrency": 1, "max_tokens": 8,
                   "cache": cache, "fixture_data": {"prompts": ["first", "second"]}}
                  for name, cache in (("mixed", "unique"), ("warm", "natural"), ("cold", "unique"))],
    }


class WorkloadContracts(unittest.TestCase):
    def test_open_loop_policy_is_rejected_before_artifacts(self):
        for arrival in ("fixed", "poisson"):
            cfg = configuration()
            cfg["cases"][0].update(arrival=arrival, rate_rps=4)
            with self.subTest(arrival=arrival), tempfile.TemporaryDirectory() as tmp:
                out = Path(tmp) / "run"
                with patch("w34perf.soak.authorization", side_effect=AssertionError("validation must precede setup")):
                    with self.assertRaisesRegex(ValueError, "closed"):
                        execute_soak(cfg, out, "fake", duration_s=1, min_completed=1, max_duration_s=2)
                self.assertFalse(out.exists())

    def test_round_robin_rotates_shapes_and_prompts_and_preserves_cache_modes(self):
        cfg = configuration()
        cfg["cases"][0]["mix"] = [{"max_tokens": n} for n in (8, 16, 24)]
        now = [10.0]
        lock = threading.Lock()

        def stream(*args, **kwargs):
            with lock:
                now[0] += .1
            return {"status": "ok", "content": "answer", "tool_calls": {},
                    "usage": {"prompt_tokens": 2, "completion_tokens": 2},
                    "first_output_ms": 1, "last_output_ms": 2, "e2e_ms": 3}

        with tempfile.TemporaryDirectory() as tmp, \
                patch("w34perf.soak.stream", side_effect=stream), \
                patch("w34perf.soak.time.monotonic", side_effect=lambda: now[0]), \
                patch("w34perf.soak.provenance", return_value={}), \
                patch("w34perf.soak.TelemetrySampler") as sampler:
            sampler.return_value.sample.return_value = {"gpus": [], "pid_rss_kib": {}, "errors": []}
            out = Path(tmp) / "run"
            result = execute_soak(cfg, out, "fake", duration_s=1.5, min_completed=15,
                                  max_duration_s=5, sustained_s=10, drain_s=0)
            self.assertEqual(result["status"], "passed", result)
            rows = [json.loads(line) for line in (out / "requests.jsonl").read_text().splitlines()]
            measured = [r for r in rows if r["type"] == "request" and r["phase"] in ("sustained", "burst")]
            mixed = [r for r in measured if r["case_id"] == "mixed"]
            with self.subTest(contract="shape rotation"):
                self.assertEqual({r["request"]["max_tokens"] for r in mixed}, {8, 16, 24})
            for case in cfg["cases"]:
                payloads = [r["request"] for r in measured if r["case_id"] == case["id"]]
                prefixes = {p["messages"][0]["content"] for p in payloads}
                with self.subTest(contract="cache mode", case=case["id"]):
                    self.assertEqual(len(prefixes), 1 if case["cache"] == "natural" else len(payloads))
                with self.subTest(contract="prompt rotation", case=case["id"]):
                    self.assertEqual({p["messages"][1]["content"] for p in payloads}, {"first", "second"})


if __name__ == "__main__":
    unittest.main()
