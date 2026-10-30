#!/usr/bin/env python3
"""Capture and replay aligned real PNG/JPEG oracles on the three VL model families.

Use a fresh --out-dir. CUDA_VISIBLE_DEVICES must select one free GPU. Every
case records subprocess logs, donor provenance, tensor metrics and exact rollout
IDs. A partial capture, failed subprocess or failed comparison fails the matrix.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
MODELS = ["nchapman/Qwen3.5-4B-W4A16-GPTQ", "nchapman/Qwen3.8-27B-W4A16-GPTQ",
          "llmfan46/Qwen3.6-35B-A3B-uncensored-heretic-Native-MTP-Preserved-GPTQ-Int4"]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out-dir", required=True, type=Path)
    ap.add_argument("--models", nargs="+", default=MODELS)
    ap.add_argument("--timeout", type=float, default=600)
    args = ap.parse_args()
    if args.timeout <= 0:
        ap.error("timeout must be positive")
    devices = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if not devices or "," in devices or devices == "-1":
        ap.error("select exactly one GPU with CUDA_VISIBLE_DEVICES")
    out = args.out_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)
    if any(out.iterdir()):
        ap.error("out-dir must be empty to exclude stale results")
    subprocess.run(["cargo", "build", "--release", "--features", "cuda", "--example", "vision_gate"], cwd=ROOT, check=True)
    print("Warning: insecure serialization is enabled for capture only; use local trusted workers only.", file=sys.stderr)
    results = []
    for index, model in enumerate(args.models):
        for fmt in ("png", "jpg"):
            name = f"model{index}-{fmt}"
            oracle = out / name
            result_file = out / f"{name}.json"
            capture = [sys.executable, str(ROOT / "tools/oracle/vllm_vl_dump.py"), "--model", model,
                       "--image", str(ROOT / f"tools/oracle/fixtures/coco_sample.{fmt}"),
                       "--prompt", "<|vision_start|><|image_pad|><|vision_end|>Describe this image.",
                       "--out-dir", str(oracle), "--tp", "1", "--tokens", "8", "--max-len", "512"]
            replay = [str(ROOT / "target/release/examples/vision_gate"), str(oracle), str(result_file)]
            result = {"model": model, "format": fmt, "capture_command": capture, "replay_command": replay, "passed": False}
            replay_env = dict(os.environ)
            replay_env.pop("VLLM_ALLOW_INSECURE_SERIALIZATION", None)
            capture_env = {**replay_env, "VLLM_ALLOW_INSECURE_SERIALIZATION": "1"}
            with (out / f"{name}.log").open("w") as log:
                try:
                    for label, command in (("capture", capture), ("replay", replay)):
                        status = subprocess.run(command, env=capture_env if label == "capture" else replay_env, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, timeout=args.timeout, check=False).returncode
                        result[label + "_status"] = status
                        if status:
                            break
                    if result_file.exists():
                        measured = json.loads(result_file.read_text())
                        result["measurement"] = measured
                        if not isinstance(measured, dict):
                            raise ValueError("report must be an object")
                        result["passed"] = (result.get("capture_status") == 0
                                            and result.get("replay_status") == 0
                                            and measured.get("passed") is True)
                except subprocess.TimeoutExpired:
                    result["failure"] = "timeout"
                except (OSError, ValueError, KeyError, TypeError) as exc:
                    result["failure"] = f"unreadable report or subprocess error: {exc}"
                    result["passed"] = False
            results.append(result)
            (out / "summary.json").write_text(json.dumps({"expected_cases":len(args.models)*2,"results":results}, indent=2) + "\n")
            print(json.dumps(result), flush=True)
    return 0 if all(r["passed"] for r in results) else 1


if __name__ == "__main__":
    sys.exit(main())
