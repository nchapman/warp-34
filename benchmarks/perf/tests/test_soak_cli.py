"""Public soak CLI wiring and checked-in workload contract."""
import contextlib
import io
from pathlib import Path
import unittest
from unittest.mock import patch

from w34perf.__main__ import main
from w34perf.config import load

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "campaigns/soak-coding.toml"


class SoakCliTests(unittest.TestCase):
    def test_example_is_valid_and_replays_sanitized_fixtures(self):
        cfg = load(CONFIG)
        self.assertEqual(len(cfg["engines"]), 1)
        self.assertEqual({c.get("kind", "text") for c in cfg["cases"]}, {"text", "agent_replay"})
        self.assertTrue(any(c.get("mix") for c in cfg["cases"]))
        self.assertLessEqual(2 * max(c["concurrency"] for c in cfg["cases"]), cfg["run"]["max_inflight"])
        self.assertEqual(cfg["engines"][0]["support"], "unverified")

    def test_soak_exit_status_and_policy_forwarding(self):
        argv = ["soak", str(CONFIG), "--engine", "warp34-soak", "--out", "/unused",
                "--duration-s", "28800", "--min-completed", "10000", "--max-duration-s", "36000",
                "--cancel-every", "10", "--server-pid", "123", "--gpu-uuid", "GPU-example",
                "--max-rss-growth-mib", "512", "--max-gpu-growth-mib", "256"]
        for status, code in (("passed", 0), ("failed", 1)):
            with self.subTest(status=status), contextlib.redirect_stdout(io.StringIO()), \
                    patch("w34perf.soak.execute_soak", return_value={"status": status}) as execute:
                self.assertEqual(main(argv), code)
                args, kwargs = execute.call_args
                self.assertEqual(args[2], "warp34-soak")
                self.assertEqual(kwargs["duration_s"], 28800)
                self.assertEqual(kwargs["min_completed"], 10000)
                self.assertEqual(kwargs["max_duration_s"], 36000)
                self.assertEqual(kwargs["cancel_every"], 10)
                self.assertEqual(kwargs["server_pids"], [123])
                self.assertEqual(kwargs["gpu_uuids"], ["GPU-example"])
                self.assertEqual(kwargs["max_rss_growth_mib"], 512)
                self.assertEqual(kwargs["max_gpu_growth_mib"], 256)

    def test_soak_invalid_policy_and_interrupt_exit_status(self):
        argv = ["soak", str(CONFIG), "--engine", "warp34-soak", "--out", "/unused",
                "--duration-s", "1", "--min-completed", "1", "--max-duration-s", "2"]
        for error, code in ((ValueError("invalid policy"), 2), (KeyboardInterrupt(), 130)):
            with self.subTest(code=code), contextlib.redirect_stderr(io.StringIO()), \
                    patch("w34perf.soak.execute_soak", side_effect=error):
                self.assertEqual(main(argv), code)


if __name__ == "__main__":
    unittest.main()
