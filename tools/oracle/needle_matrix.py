#!/usr/bin/env python3
"""Run the 1-GPU 27B TQ4 needle matrix; retain failures and exit nonzero on any failure.

Each window includes 32 generation tokens and four engine-reserved tokens. The
report preserves exact prompt IDs so another implementation can replay them.
This is a retrieval gate, not a claim of parity with a synthetic attention test.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
WINDOWS = (32768, 65536, 131072, 262144)
POLICIES = ("tq4", "tq4_all")


def cases(depths, policies=POLICIES):
    if not depths or any(not 0 <= d <= 100 for d in depths):
        raise ValueError("depths must be nonempty percentages in 0..100")
    if not policies or any(p not in POLICIES for p in policies):
        raise ValueError(f"policies must be a nonempty subset of {POLICIES}")
    return [(policy, window, depth) for policy in policies for window in WINDOWS for depth in depths]


def passes_report(status, measured, policy, window, depth):
    if not isinstance(measured, dict) or not isinstance(measured.get("answer"), str):
        return False
    integers = ("context_window", "depth_percent", "tp", "prompt_tokens",
                "max_new_tokens", "reserved_tokens", "cached_prefix_tokens")
    if any(type(measured.get(key)) is not int for key in integers):
        return False
    return (status == 0 and measured.get("passed") is True
            and measured.get("context_window") == window
            and measured.get("kv_policy") == policy
            and measured.get("depth_percent") == depth
            and measured.get("tp") == 1
            and measured.get("prompt_tokens", 0) + measured.get("max_new_tokens", 0) + measured.get("reserved_tokens", 0) == window
            and measured.get("cached_prefix_tokens") == 0
            and measured.get("answer", "").strip() == measured.get("expected") == "73918462")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default="nchapman/Qwen3.8-27B-W4A16-HGPTQ")
    ap.add_argument("--out-dir", type=Path, default=ROOT / "outputs/needle")
    ap.add_argument("--depths", type=int, nargs="+", default=[50])
    ap.add_argument("--policies", nargs="+", choices=POLICIES, default=list(POLICIES))
    ap.add_argument("--skip-build", action="store_true", help="use an already-built target/release/examples/needle_gate")
    ap.add_argument("--timeout", type=float, default=3600, help="hard seconds per model process")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    matrix = cases(args.depths, tuple(args.policies))
    if args.timeout <= 0:
        ap.error("timeout must be positive")
    devices = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if not args.dry_run and (not devices or "," in devices or devices == "-1"):
        ap.error("set CUDA_VISIBLE_DEVICES to exactly one available GPU")
    args.out_dir = args.out_dir.resolve()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    binary = ROOT / "target/release/examples/needle_gate"
    results = []
    summary = {"model": args.model, "cuda_visible_devices": devices,
               "expected_cases": len(matrix), "results": results}
    if not args.dry_run:
        (args.out_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
        if not args.skip_build:
            subprocess.run(["cargo", "build", "--release", "--features", "cuda", "--example", "needle_gate"], cwd=ROOT, check=True)
    for policy, window, depth in matrix:
        name = f"{policy}-{window}-depth{depth}"
        report = args.out_dir / f"{name}.json"
        command = [str(binary), args.model, policy, str(window), str(depth), str(report)]
        if args.dry_run:
            print(json.dumps(command))
            continue
        # Never mistake a previous run's output for a fresh measurement.
        report.unlink(missing_ok=True)
        start = time.monotonic()
        with (args.out_dir / f"{name}.log").open("w") as log:
            try:
                status = subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, timeout=args.timeout, check=False).returncode
                failure = None
            except subprocess.TimeoutExpired:
                status, failure = -1, "timeout"
            except OSError as exc:
                status, failure = -1, f"launch error: {exc}"
        result = {"policy": policy, "window": window, "depth_percent": depth, "status": status,
                  "elapsed_seconds": time.monotonic() - start, "command": command, "failure": failure, "passed": False}
        if report.exists():
            try:
                measured = json.loads(report.read_text())
                result["measurement"] = measured
                if not isinstance(measured, dict):
                    raise ValueError("report must be an object")
                result.update({key: measured[key] for key in ("answer", "ttft_ms", "decode_tok_per_s", "model_snapshot") if key in measured})
                result["passed"] = passes_report(status, measured, policy, window, depth)
            except (OSError, ValueError, KeyError, TypeError) as exc:
                result["failure"] = f"unreadable report: {exc}"
                result["passed"] = False
        results.append(result)
        (args.out_dir / "summary.json").write_text(json.dumps({"model": args.model, "cuda_visible_devices": devices, "expected_cases": len(matrix), "results": results}, indent=2) + "\n")
        print(json.dumps(result), flush=True)
    if args.dry_run:
        return 0
    return 0 if len(results) == len(matrix) and all(r["passed"] for r in results) else 1


if __name__ == "__main__":
    sys.exit(main())
