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

    def test_multi_trajectory_agent_replay_rotates_instances_and_validates_schema(self):
        from w34perf.config import load as load_cfg

        multi_fixture = {
            "source": "nvidia/Open-SWE-Traces@test",
            "trajectories": [
                {
                    "id": "repo_a_pr1",
                    "system": "Harness system prompt",
                    "turns": [
                        {"user": "Fix bug A", "tools": []},
                        {"user": None, "tools": [{"name": "read", "arguments": {"path": "a.py"}, "result": "x = 1\n"}]},
                    ],
                },
                {
                    "id": "repo_b_pr2",
                    "system": "Harness system prompt",
                    "turns": [
                        {"user": "Fix bug B", "tools": []},
                        {"user": None, "tools": [{"name": "bash", "arguments": {"cmd": "pytest"}, "result": "1 passed\n"}]},
                        {"user": None, "tools": [{"name": "git_diff", "arguments": {}, "result": "+y = 2\n"}]},
                    ],
                },
            ],
        }
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            (tmp_path / "swe.json").write_text(json.dumps(multi_fixture))
            toml_text = """
schema_version = 1
[run]
max_inflight = 2
timeout_s = 1
warmup_requests = 0
[[engines]]
id = "fake"
engine = "warp-34"
version = "test"
model = "fake"
model_revision = "test"
weight_format = "int4"
kv_format = "tq4"
parallelism = "single"
speculation = "mtp-3"
cache_config = "test"
launch_command = "w34 serve"
gpu_count = 1
base_url = "http://127.0.0.1:1/v1"
[[cases]]
id = "swe-shared"
kind = "agent_replay"
fixture = "swe.json"
concurrency = 1
max_tokens = 16
cache = "natural"
"""
            cfg_file = tmp_path / "campaign.toml"
            cfg_file.write_text(toml_text)
            cfg = load_cfg(cfg_file)

            now = [10.0]
            lock = threading.Lock()

            def stream(*args, **kwargs):
                with lock:
                    now[0] += 0.1
                return {"status": "ok", "content": "patch step", "tool_calls": {},
                        "usage": {"prompt_tokens": 10, "completion_tokens": 4},
                        "first_output_ms": 1, "last_output_ms": 2, "e2e_ms": 3}

            out = tmp_path / "run"
            with patch("w34perf.soak.stream", side_effect=stream), \
                    patch("w34perf.soak.time.monotonic", side_effect=lambda: now[0]), \
                    patch("w34perf.soak.provenance", return_value={}), \
                    patch("w34perf.soak.TelemetrySampler") as sampler:
                sampler.return_value.sample.return_value = {"gpus": [], "pid_rss_kib": {}, "errors": []}
                result = execute_soak(cfg, out, "fake", duration_s=1.0, min_completed=5,
                                      max_duration_s=5, sustained_s=10, drain_s=0)
            self.assertEqual(result["status"], "passed", result)
            rows = [json.loads(line) for line in (out / "requests.jsonl").read_text().splitlines()]
            measured = [r for r in rows if r["type"] == "request" and r["phase"] in ("sustained", "burst")]
            self.assertEqual({r.get("trajectory_id") for r in measured}, {"repo_a_pr1", "repo_b_pr2"})
            # Under natural cache mode, both trajectories share the identical system message prefix.
            self.assertEqual(len({r["request"]["messages"][0]["content"] for r in measured}), 1)
            step0_users = {r["request"]["messages"][1]["content"] for r in measured if r["step"] == 0}
            self.assertEqual(step0_users, {"Fix bug A", "Fix bug B"})

            # Reject ambiguous fixtures specifying both top-level turns and trajectories, or duplicate IDs.
            for bad in (
                {**multi_fixture, "turns": [{"user": "ambiguous", "tools": []}]},
                {"trajectories": [multi_fixture["trajectories"][0], multi_fixture["trajectories"][0]]},
                {"trajectories": []},
            ):
                (tmp_path / "swe.json").write_text(json.dumps(bad))
                with self.assertRaises(ValueError):
                    load_cfg(cfg_file)


if __name__ == "__main__":
    unittest.main()
