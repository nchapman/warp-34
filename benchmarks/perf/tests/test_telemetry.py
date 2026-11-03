"""External boundaries are mocked: these tests never query a real GPU."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import threading
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "benchmarks/perf"))
from w34perf.telemetry import TelemetrySampler


GPU = "GPU-one, 10, 100, 25, 5, 1000, 900, 2000, 45, 120.5\n"


def proc_read(path):
    if str(path) == "/proc/meminfo":
        return "MemTotal: 1024 kB\nMemAvailable: 512 kB\nHugePages_Total: 0\n"
    if str(path) == "/proc/42/status":
        return "Name:\tworker\nVmRSS:\t123 kB\n"
    raise FileNotFoundError(str(path))


class TelemetryTests(unittest.TestCase):
    def test_validation_and_no_io_on_construction(self):
        with patch("w34perf.telemetry.subprocess.run") as run:
            sampler = TelemetrySampler("unused")
            self.assertEqual(sampler.interval_s, 1)
            run.assert_not_called()
        for key in ("interval_s", "timeout_s"):
            for value in (0, -1, float("nan"), float("inf"), True):
                with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                    TelemetrySampler("unused", **{key: value})
        for pid in (0, -1, True, "42", 1.5):
            with self.assertRaises(ValueError):
                TelemetrySampler("unused", pids=[pid])

    @patch("pathlib.Path.read_text", autospec=True, side_effect=proc_read)
    @patch("w34perf.telemetry.subprocess.run")
    def test_schema_and_explicit_pids(self, run, read):
        run.return_value = subprocess.CompletedProcess([], 0, GPU + GPU.replace("one", "two"), "")
        row = TelemetrySampler("unused", pids=[42, 99, 42], timeout_s=.3).sample()
        self.assertEqual(len(row["gpus"]), 2)
        self.assertEqual(row["gpus"][0]["power_w"], 120.5)
        self.assertEqual(row["gpus"][1]["uuid"], "GPU-two")
        self.assertEqual(row["host_meminfo_kib"], {"MemTotal": 1024, "MemAvailable": 512})
        self.assertEqual(row["pid_rss_kib"], {"42": 123, "99": None})
        self.assertEqual(row["errors"][0]["source"], "pid:99")
        self.assertGreater(row["timestamp_unix_s"], 0)
        self.assertGreater(row["timestamp_monotonic_s"], 0)
        self.assertEqual(read.call_count, 3)
        args, kwargs = run.call_args
        self.assertEqual(args[0][0], "nvidia-smi")
        self.assertIn("uuid,memory.used", args[0][1])
        self.assertEqual(kwargs["timeout"], .3)
        self.assertTrue(kwargs["check"])
        self.assertNotIn("shell", kwargs)

    @patch("pathlib.Path.read_text", side_effect=PermissionError("denied"))
    def test_sampling_failures_retained(self, read):
        failures = [FileNotFoundError("missing nvidia-smi"),
                    subprocess.TimeoutExpired("nvidia-smi", .1),
                    subprocess.CalledProcessError(1, "nvidia-smi", stderr="driver unavailable")]
        for failure in failures:
            with self.subTest(failure=failure), patch("w34perf.telemetry.subprocess.run", side_effect=failure):
                row = TelemetrySampler("unused").sample()
                self.assertIsNone(row["gpus"])
                self.assertIsNone(row["host_meminfo_kib"])
                self.assertEqual(row["errors"][0]["type"], type(failure).__name__)
                self.assertEqual(row["pid_rss_kib"], {})
                json.dumps(row, allow_nan=False)

    @patch("pathlib.Path.read_text", return_value="VmRSS: broken kB\n")
    @patch("w34perf.telemetry.subprocess.run")
    def test_partial_unavailable_and_malformed_values(self, run, read):
        run.return_value.stdout = GPU.replace("120.5", "[Not Supported]") + "bad,row\n" + GPU.replace("120.5", "nan")
        row = TelemetrySampler("unused", pids=[42]).sample()
        self.assertIsNone(row["gpus"][0]["power_w"])
        self.assertEqual(row["gpus"][0]["memory_used_mib"], 10)
        self.assertIsNone(row["gpus"][1]["power_w"])
        self.assertIsNone(row["pid_rss_kib"]["42"])
        self.assertTrue(any(e.get("raw") == "[Not Supported]" for e in row["errors"]))
        self.assertTrue(any(e["type"] == "ParseError" for e in row["errors"]))

    def test_thread_wait_join_append_and_context_exception(self):
        sampled = threading.Event()
        def sample():
            sampled.set()
            return {"test": 1}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "telemetry.jsonl"
            path.write_text('{"earlier": true}\n')
            sampler = TelemetrySampler(path, interval_s=3600)
            with patch.object(sampler, "sample", side_effect=sample) as collect:
                with self.assertRaisesRegex(ValueError, "body failure"):
                    with sampler:
                        self.assertTrue(sampled.wait(2))
                        raise ValueError("body failure")
                self.assertFalse(sampler._thread.is_alive())
                self.assertTrue(sampler._file.closed)
                self.assertEqual(collect.call_count, 1)
                sampler.stop()
            self.assertEqual([json.loads(line) for line in path.read_text().splitlines()],
                             [{"earlier": True}, {"test": 1}])
            with self.assertRaises(RuntimeError):
                sampler.__enter__()

    def test_worker_failure_is_reported_and_closed(self):
        failed = threading.Event()
        def sample():
            failed.set()
            raise OSError("write/sample failure")
        with tempfile.TemporaryDirectory() as directory:
            sampler = TelemetrySampler(Path(directory) / "t.jsonl")
            with patch.object(sampler, "sample", side_effect=sample):
                with self.assertRaisesRegex(RuntimeError, "worker failed"):
                    with sampler:
                        self.assertTrue(failed.wait(2))
            self.assertFalse(sampler._thread.is_alive())
            self.assertTrue(sampler._file.closed)


if __name__ == "__main__":
    unittest.main()
