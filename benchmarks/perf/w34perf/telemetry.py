"""Opt-in, stdlib-only resource sampling; no service lifecycle management.

Integration::

    with TelemetrySampler(output_dir / "telemetry.jsonl", pids=[server_pid]):
        run_benchmark()

The file is appended to (and flushed after each sample). Each row has Unix and
monotonic timestamps, GPU readings in explicit units, /proc/meminfo values in
KiB, optional PID RSS in KiB, and source-specific errors. Missing readings are
null, never synthetic zeros. Sampling starts immediately, then waits interval_s
between completed samples (not a fixed-rate schedule). Exit waits for any active
sample, including at most one bounded nvidia-smi invocation. Instances are
single-use; stop() is idempotent. File/worker failures are raised by stop/exit,
but never replace an exception from the context body.
"""

import csv
import json
import math
from pathlib import Path
import subprocess
import threading
import time


_GPU_FIELDS = (
    ("memory.used", "memory_used_mib"),
    ("memory.total", "memory_total_mib"),
    ("utilization.gpu", "utilization_gpu_percent"),
    ("utilization.memory", "utilization_memory_percent"),
    ("clocks.current.graphics", "graphics_clock_mhz"),
    ("clocks.current.sm", "sm_clock_mhz"),
    ("clocks.current.memory", "memory_clock_mhz"),
    ("temperature.gpu", "temperature_c"),
    ("power.draw", "power_w"),
)


def _positive_finite(value, name):
    if isinstance(value, bool):
        raise ValueError(f"{name} must be positive and finite")
    value = float(value)
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be positive and finite")
    return value


def _error(errors, source, exc):
    errors.append({"source": source, "type": type(exc).__name__, "message": str(exc)})


def _gpu_readings(timeout, errors):
    command = ["nvidia-smi", "--query-gpu=uuid," + ",".join(f[0] for f in _GPU_FIELDS),
               "--format=csv,noheader,nounits"]
    try:
        result = subprocess.run(command, capture_output=True, text=True,
                                timeout=timeout, check=True)
    except (OSError, subprocess.SubprocessError) as exc:
        _error(errors, "gpu", exc)
        if isinstance(exc, subprocess.CalledProcessError):
            errors[-1]["stderr"] = exc.stderr
        return None
    readings = []
    for row in csv.reader(result.stdout.splitlines()):
        if not row:
            continue
        if len(row) != len(_GPU_FIELDS) + 1 or not row[0].strip():
            errors.append({"source": "gpu", "type": "ParseError", "raw": row})
            continue
        gpu = {"uuid": row[0].strip()}
        for (_, key), raw in zip(_GPU_FIELDS, row[1:]):
            try:
                value = float(raw.strip())
                if not math.isfinite(value) or value < 0:
                    raise ValueError("non-finite or negative GPU value")
                gpu[key] = value
            except ValueError:
                gpu[key] = None
                errors.append({"source": "gpu", "uuid": gpu["uuid"],
                               "field": key, "type": "Unavailable", "raw": raw.strip()})
        readings.append(gpu)
    return readings


def _kib_fields(path, errors, source, wanted=None):
    try:
        text = path.read_text()
    except OSError as exc:
        _error(errors, source, exc)
        return None
    values = {}
    for line in text.splitlines():
        key, sep, rest = line.partition(":")
        if not sep or (wanted is not None and key not in wanted):
            continue
        parts = rest.split()
        # meminfo also contains unitless counts, which are not memory sizes.
        if wanted is None and (not parts or parts[-1] != "kB"):
            continue
        try:
            if len(parts) != 2 or parts[1] != "kB" or int(parts[0]) < 0:
                raise ValueError("expected nonnegative KiB value")
            values[key] = int(parts[0])
        except ValueError as exc:
            values[key] = None
            _error(errors, source + "." + key, exc)
    if not values:
        _error(errors, source, ValueError("no requested memory fields found"))
        return None
    return values


class TelemetrySampler:
    """Append JSONL samples while a context is active.

    Only explicitly supplied positive integer PIDs are inspected; there is no
    process discovery. PID reuse is not tracked. timeout_s bounds each GPU
    subprocess (default 2 seconds); interval_s defaults to 1 second. Parent
    directories must already exist. Construction alone performs no I/O.
    """

    def __init__(self, path, *, interval_s=1.0, timeout_s=2.0, pids=()):
        self.path = Path(path)
        self.interval_s = _positive_finite(interval_s, "interval_s")
        self.timeout_s = _positive_finite(timeout_s, "timeout_s")
        self.pids = tuple(dict.fromkeys(pids))
        if any(isinstance(pid, bool) or not isinstance(pid, int) or pid <= 0
               for pid in self.pids):
            raise ValueError("pids must contain positive integers")
        self._stop = threading.Event()
        self._thread = None
        self._file = None
        self._failure = None
        self._started = False

    def sample(self):
        """Collect one row without writing it; source failures remain in errors."""
        row = {"timestamp_unix_s": time.time(), "timestamp_monotonic_s": time.monotonic(),
               "errors": []}
        errors = row["errors"]
        row["gpus"] = _gpu_readings(self.timeout_s, errors)
        row["host_meminfo_kib"] = _kib_fields(Path("/proc/meminfo"), errors, "host")
        row["pid_rss_kib"] = {}
        for pid in self.pids:
            values = _kib_fields(Path(f"/proc/{pid}/status"), errors,
                                 f"pid:{pid}", {"VmRSS"})
            row["pid_rss_kib"][str(pid)] = values.get("VmRSS") if values else None
        return row

    def _run(self):
        try:
            while not self._stop.is_set():
                self._file.write(json.dumps(self.sample(), allow_nan=False) + "\n")
                self._file.flush()
                if self._stop.wait(self.interval_s):
                    break
        except Exception as exc:
            self._failure = exc
            self._stop.set()

    def __enter__(self):
        if self._started:
            raise RuntimeError("TelemetrySampler instances are single-use")
        self._started = True
        self._file = self.path.open("a", encoding="utf-8")
        self._thread = threading.Thread(target=self._run, name="resource-telemetry")
        try:
            self._thread.start()
        except BaseException:
            self._thread = None
            self._file.close()
            raise
        return self

    def stop(self):
        """Wake, join, and close the writer; propagate worker/write failures."""
        self._stop.set()
        if self._thread is not None:
            self._thread.join()
        if self._file is not None and not self._file.closed:
            self._file.close()
        if self._failure is not None:
            raise RuntimeError("telemetry worker failed") from self._failure

    def __exit__(self, exc_type, exc, traceback):
        try:
            self.stop()
        except Exception:
            if exc_type is None:
                raise
        return False
