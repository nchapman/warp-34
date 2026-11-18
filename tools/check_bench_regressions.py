#!/usr/bin/env python3
"""Automated benchmark regression gate & Markdown comparison table generator.

Evaluates `warp34 bench` JSON summaries against hardware-keyed baselines in
`tools/oracle/goldens/bench_baselines.json` (`sm_80`, `sm_86`, `sm_89`) and
renders an apples-to-apples Markdown comparison table that explicitly separates:
  - Execution Mode: `Plain` vs Speculative (`MTP-3` / `DFlash2`)
  - Prompt Regime: `synthetic` (periodic seed for profiler/competitor alignment)
    vs `realistic` (non-repeating ChatML coding prompt measuring natural draft
    acceptance)

Checks both:
  1. Hard correctness invariants (`cuda_graph == True`, `radix.lossless_parity == True`,
     `vision.lossless_parity == True`, `vision.content_hash_isolated == True`, and
     expected `prompt_source` / `speculative` mode).
  2. Quantitative throughput, TTFT speedup, and speculative acceptance floors
     within a configurable tolerance (`--max-regression-pct`, default `12%`).

Can either audit pre-collected JSON files (`--input case_id=path.json`) or run
live `warp34 bench` invocations (`--live --suite quick|full` or `--case <id>`).
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import pathlib
import subprocess
import sys
import tempfile
from typing import Any

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
DEFAULT_BASELINES_PATH = REPO_ROOT / "tools" / "oracle" / "goldens" / "bench_baselines.json"
DEFAULT_WARP34_BIN = REPO_ROOT / "target" / "release" / "warp34"

QUICK_SUITE_CASES: tuple[str, ...] = (
    "4b_tp1_plain",
    "4b_tp1_mtp_synth",
    "4b_tp1_mtp_realistic",
    "4b_tp1_dflash2_synth",
    "4b_tp1_dflash2_realistic",
)

FULL_SUITE_CASES: tuple[str, ...] = (
    "4b_tp1_plain",
    "4b_tp1_mtp_synth",
    "4b_tp1_mtp_realistic",
    "4b_tp1_dflash2_synth",
    "4b_tp1_dflash2_realistic",
    "4b_tp2_plain",
    "4b_tp2_mtp_synth",
    "4b_tp2_dflash2_synth",
    "9b_tp1_lora_heretic",
    "27b_tp2_plain",
    "27b_tp2_mtp_synth",
    "27b_tp2_dflash2_synth",
    "35b_tp2_plain",
    "35b_tp2_mtp_synth",
    "flash_next_tp2_mtp_synth",
)


@dataclasses.dataclass(frozen=True)
class MetricComparison:
    metric: str
    measured: float
    baseline: float
    floor: float
    delta_pct: float
    passed: bool


@dataclasses.dataclass(frozen=True)
class CaseEvaluation:
    case_id: str
    label: str
    arch: str
    mode_label: str
    prompt_source: str
    tp: int
    decode_tok_per_s: float | None
    prefill_tok_per_s: float | None
    mean_accepted_drafts: float | None
    radix_ttft_speedup_x: float | None
    scheduler_decode_tok_per_s: float | None
    vision_ttft_speedup_x: float | None
    baseline_decode_tok_per_s: float | None
    decode_delta_pct: float | None
    competitor_summary: str
    metrics: tuple[MetricComparison, ...]
    invariant_failures: tuple[str, ...]

    @property
    def passed(self) -> bool:
        return not self.invariant_failures and all(m.passed for m in self.metrics)


def normalize_arch(raw: str) -> str:
    """Normalize compute capability strings (`8.6`, `86`, `sm_86`) to `sm_XX`."""
    cleaned = raw.strip().lower()
    if cleaned.startswith("sm_"):
        cleaned = cleaned[3:]
    cleaned = cleaned.replace(".", "")
    if cleaned in ("80", "86", "89"):
        return f"sm_{cleaned}"
    raise ValueError(f"unsupported or unrecognized CUDA architecture {raw!r} (expected sm_80, sm_86, or sm_89)")


def detect_host_arch(env_override: str | None = None) -> tuple[str, int]:
    """Detect `(sm_arch, gpu_count)` from `nvidia-smi` or an explicit override."""
    try:
        proc = subprocess.run(
            ["nvidia-smi", "--query-gpu=compute_cap", "--format=csv,noheader"],
            check=True,
            capture_output=True,
            text=True,
        )
        caps = [line.strip() for line in proc.stdout.splitlines() if line.strip()]
    except (FileNotFoundError, subprocess.SubprocessError):
        caps = []

    gpu_count = len(caps)
    if env_override:
        return normalize_arch(env_override), max(gpu_count, 1)
    if not caps:
        raise RuntimeError(
            "unable to query GPU compute capability via nvidia-smi; pass --arch sm_86|sm_89|sm_80"
        )
    return normalize_arch(caps[0]), gpu_count


def load_baselines(path: pathlib.Path = DEFAULT_BASELINES_PATH) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict) or "cases" not in data or not isinstance(data["cases"], dict):
        raise ValueError(f"invalid benchmark baselines schema in {path}")
    return data


def extract_decode_section(payload: dict[str, Any]) -> dict[str, Any]:
    """Return the decode summary dict from either `--stage all` or `--stage decode` JSON."""
    if isinstance(payload.get("decode"), dict):
        return payload["decode"]
    return payload


def format_mode_label(speculative: str | None, case_spec: dict[str, Any]) -> str:
    if speculative == "mtp":
        k = case_spec.get("mtp_k", 3)
        return f"MTP-{k}"
    if speculative == "dflash2":
        return "DFlash2"
    extra = case_spec.get("extra_args") or []
    if "--active-lora" in extra:
        idx = extra.index("--active-lora")
        if idx + 1 < len(extra):
            return f"Plain + LoRA ({extra[idx + 1]})"
    return "Plain"


def format_competitor_summary(
    decode_tok_per_s: float | None,
    competitors: dict[str, Any],
) -> str:
    if not competitors:
        return "—"
    parts: list[str] = []
    for comp_key, comp in competitors.items():
        comp_tps = comp.get("decode_tok_per_s")
        comp_label = comp.get("label", comp_key)
        if isinstance(comp_tps, (int, float)) and comp_tps > 0:
            if decode_tok_per_s is not None and decode_tok_per_s > 0:
                speedup = decode_tok_per_s / float(comp_tps)
                parts.append(f"**{speedup:.2f}x** vs {comp_label} ({comp_tps:.1f})")
            else:
                parts.append(f"{comp_label}: {comp_tps:.1f}")
    return "; ".join(parts) if parts else "—"


def evaluate_case(
    case_id: str,
    payload: dict[str, Any],
    baselines_doc: dict[str, Any],
    arch: str,
    max_regression_pct: float = 12.0,
    acceptance_tolerance: float = 0.25,
) -> CaseEvaluation:
    cases = baselines_doc.get("cases", {})
    if case_id not in cases:
        raise KeyError(f"unknown benchmark case {case_id!r} (known: {', '.join(sorted(cases))})")
    case_spec = cases[case_id]
    hw_map = case_spec.get("hardware", {})
    norm_arch = normalize_arch(arch)
    hw_base = hw_map.get(norm_arch, {})

    decode = extract_decode_section(payload)
    radix = payload.get("radix") if isinstance(payload.get("radix"), dict) else None
    scheduler = payload.get("scheduler") if isinstance(payload.get("scheduler"), dict) else None
    vision = payload.get("vision") if isinstance(payload.get("vision"), dict) else None

    invariant_failures: list[str] = []

    if not hw_base:
        invariant_failures.append(
            f"{case_id}: no hardware baseline entry defined for architecture {norm_arch}"
        )

    expected_spec = case_spec.get("speculative")
    actual_spec = decode.get("speculative", payload.get("speculative"))
    if actual_spec != expected_spec:
        invariant_failures.append(
            f"{case_id}: expected speculative={expected_spec!r}, got {actual_spec!r}"
        )

    expected_prompt_source = case_spec.get("prompt_source")
    actual_prompt_source = decode.get("prompt_source", "synthetic")
    if expected_prompt_source and actual_prompt_source != expected_prompt_source:
        invariant_failures.append(
            f"{case_id}: expected prompt_source={expected_prompt_source!r}, got {actual_prompt_source!r}"
        )

    expected_tp = int(case_spec.get("tp", 1))
    actual_tp = int(decode.get("tp", payload.get("tp", 1)))
    if actual_tp != expected_tp:
        invariant_failures.append(f"{case_id}: expected tp={expected_tp}, got {actual_tp}")

    if decode.get("cuda_graph") is not True:
        invariant_failures.append(
            f"{case_id}: expected decode.cuda_graph=True, got {decode.get('cuda_graph')!r}"
        )

    if radix is not None and radix.get("lossless_parity") is not True:
        invariant_failures.append(
            f"{case_id}: radix.lossless_parity={radix.get('lossless_parity')!r} (expected True)"
        )

    if vision is not None and not vision.get("skipped", False):
        if vision.get("lossless_parity") is not True:
            invariant_failures.append(
                f"{case_id}: vision.lossless_parity={vision.get('lossless_parity')!r} (expected True)"
            )
        if vision.get("content_hash_isolated") is not True:
            invariant_failures.append(
                f"{case_id}: vision.content_hash_isolated={vision.get('content_hash_isolated')!r} (expected True)"
            )

    decode_tps = _as_opt_float(decode.get("decode_tok_per_s"))
    prefill_tps = _as_opt_float(decode.get("prefill_tok_per_s"))
    mean_acc = _as_opt_float(decode.get("mean_accepted_drafts"))
    radix_speedup = _as_opt_float(radix.get("ttft_speedup_x")) if radix else None
    sched_decode_tps = (
        _as_opt_float(scheduler.get("aggregate_decode_tok_per_s")) if scheduler else None
    )
    sched_total_tps = _as_opt_float(scheduler.get("total_tok_per_s")) if scheduler else None
    vision_speedup = (
        _as_opt_float(vision.get("ttft_speedup_x"))
        if (vision and not vision.get("skipped", False))
        else None
    )

    measured_lookup: dict[str, float | None] = {
        "decode_tok_per_s": decode_tps,
        "prefill_tok_per_s": prefill_tps,
        "mean_accepted_drafts": mean_acc,
        "radix_ttft_speedup_x": radix_speedup,
        "scheduler_aggregate_decode_tok_per_s": sched_decode_tps,
        "scheduler_total_tok_per_s": sched_total_tps,
        "vision_ttft_speedup_x": vision_speedup,
    }

    comparisons: list[MetricComparison] = []
    scale = max(0.0, 1.0 - max_regression_pct / 100.0)

    for metric_name, measured_val in measured_lookup.items():
        if metric_name not in hw_base:
            continue
        base_val = float(hw_base[metric_name])
        if measured_val is None:
            invariant_failures.append(
                f"{case_id}: missing measured metric {metric_name!r} (baseline {base_val:.2f})"
            )
            continue
        if metric_name == "mean_accepted_drafts":
            floor_val = max(0.1, min(base_val * scale, base_val - acceptance_tolerance))
        elif metric_name.endswith("_speedup_x"):
            # TTFT warm/cold ratios have slightly higher run-to-run jitter on sub-6ms warm prefills
            floor_val = max(1.5, base_val * max(0.0, 1.0 - (max_regression_pct + 8.0) / 100.0))
        else:
            floor_val = base_val * scale
        delta_pct = ((measured_val - base_val) / base_val * 100.0) if base_val > 0 else 0.0
        passed = measured_val >= floor_val
        comparisons.append(
            MetricComparison(
                metric=metric_name,
                measured=measured_val,
                baseline=base_val,
                floor=floor_val,
                delta_pct=delta_pct,
                passed=passed,
            )
        )

    base_decode_tps = _as_opt_float(hw_base.get("decode_tok_per_s"))
    decode_delta_pct = (
        ((decode_tps - base_decode_tps) / base_decode_tps * 100.0)
        if (decode_tps is not None and base_decode_tps is not None and base_decode_tps > 0)
        else None
    )

    return CaseEvaluation(
        case_id=case_id,
        label=case_spec.get("label", case_id),
        arch=norm_arch,
        mode_label=format_mode_label(expected_spec, case_spec),
        prompt_source=actual_prompt_source,
        tp=actual_tp,
        decode_tok_per_s=decode_tps,
        prefill_tok_per_s=prefill_tps,
        mean_accepted_drafts=mean_acc,
        radix_ttft_speedup_x=radix_speedup,
        scheduler_decode_tok_per_s=sched_decode_tps,
        vision_ttft_speedup_x=vision_speedup,
        baseline_decode_tok_per_s=base_decode_tps,
        decode_delta_pct=decode_delta_pct,
        competitor_summary=format_competitor_summary(
            decode_tps, case_spec.get("competitors", {})
        ),
        metrics=tuple(comparisons),
        invariant_failures=tuple(invariant_failures),
    )


def _as_opt_float(val: Any) -> float | None:
    if isinstance(val, (int, float)) and not isinstance(val, bool):
        return float(val)
    return None


def render_markdown_report(
    evaluations: list[CaseEvaluation],
    arch: str,
    max_regression_pct: float,
) -> str:
    lines: list[str] = []
    passed_count = sum(1 for e in evaluations if e.passed)
    total_count = len(evaluations)
    status_badge = "PASS" if passed_count == total_count else "FAIL"

    lines.append(
        f"### `warp34 bench` Regression & Comparison Report (`{arch}`, `{status_badge}` {passed_count}/{total_count}, max regression `{max_regression_pct:.1f}%`)"
    )
    lines.append("")
    lines.append(
        "| Case | Mode | Prompt | TP | Decode (`tok/s`) | Baseline (`tok/s`) | Delta | Draft Accept | B=8 Sched (`tok/s`) | Radix TTFT | Competitor Reference | Status |"
    )
    lines.append(
        "| :--- | :--- | :--- | :-: | ---: | ---: | ---: | ---: | ---: | ---: | :--- | :-: |"
    )

    for ev in evaluations:
        dec_str = f"**{ev.decode_tok_per_s:.1f}**" if ev.decode_tok_per_s is not None else "—"
        base_str = (
            f"{ev.baseline_decode_tok_per_s:.1f}"
            if ev.baseline_decode_tok_per_s is not None
            else "—"
        )
        delta_str = (
            f"{ev.decode_delta_pct:+.1f}%" if ev.decode_delta_pct is not None else "—"
        )
        acc_str = (
            f"{ev.mean_accepted_drafts:.2f}" if ev.mean_accepted_drafts is not None else "—"
        )
        sched_str = (
            f"{ev.scheduler_decode_tok_per_s:.1f}"
            if ev.scheduler_decode_tok_per_s is not None
            else "—"
        )
        radix_str = (
            f"{ev.radix_ttft_speedup_x:.2f}x"
            if ev.radix_ttft_speedup_x is not None
            else "—"
        )
        status_str = "PASS" if ev.passed else "FAIL"
        lines.append(
            f"| `{ev.case_id}` | {ev.mode_label} | `{ev.prompt_source}` | {ev.tp} | {dec_str} | {base_str} | {delta_str} | {acc_str} | {sched_str} | {radix_str} | {ev.competitor_summary} | **{status_str}** |"
        )

    failures: list[str] = []
    for ev in evaluations:
        for inv in ev.invariant_failures:
            failures.append(inv)
        for m in ev.metrics:
            if not m.passed:
                failures.append(
                    f"{ev.case_id}.{m.metric}: measured {m.measured:.2f} < floor {m.floor:.2f} "
                    f"(baseline {m.baseline:.2f}, {m.delta_pct:+.1f}%)"
                )

    if failures:
        lines.append("")
        lines.append("#### Regression / Invariant Failures")
        for f in failures:
            lines.append(f"- {f}")

    return "\n".join(lines)


def build_bench_command(
    warp34_bin: pathlib.Path,
    case_spec: dict[str, Any],
    out_json: pathlib.Path,
) -> list[str]:
    cmd = [
        str(warp34_bin),
        "bench",
        "--offline",
        "--model",
        str(case_spec["model"]),
        "--tp",
        str(case_spec.get("tp", 1)),
        "--stage",
        str(case_spec.get("stage", "all")),
        "--kv-format",
        str(case_spec.get("kv_format", "tq4")),
        "--prompt-len",
        str(case_spec.get("prompt_len", 128)),
        "--gen-len",
        str(case_spec.get("gen_len", 128)),
        "--batch-size",
        str(case_spec.get("batch_size", 8)),
    ]
    spec = case_spec.get("speculative")
    if spec == "mtp":
        cmd.extend(["--speculative", "mtp", "--mtp-k", str(case_spec.get("mtp_k", 3))])
    elif spec == "dflash2":
        cmd.extend(
            [
                "--speculative",
                "dflash2",
                "--draft-model",
                str(case_spec["draft_model"]),
            ]
        )
    for arg in case_spec.get("extra_args", []):
        cmd.append(str(arg))
    cmd.extend(["--out", str(out_json)])
    return cmd


def parse_input_spec(spec: str) -> tuple[str, pathlib.Path]:
    if "=" not in spec:
        raise ValueError(
            f"invalid --input {spec!r}; expected format <case_id>=<path/to/output.json>"
        )
    case_id, raw_path = spec.split("=", 1)
    return case_id.strip(), pathlib.Path(raw_path.strip())


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--baselines",
        type=pathlib.Path,
        default=DEFAULT_BASELINES_PATH,
        help="Path to bench_baselines.json",
    )
    parser.add_argument(
        "--arch",
        type=str,
        default=None,
        help="Target GPU architecture (sm_80, sm_86, sm_89); auto-detected via nvidia-smi if omitted",
    )
    parser.add_argument(
        "--max-regression-pct",
        type=float,
        default=12.0,
        help="Maximum allowed throughput regression percentage below baseline (default: 12.0)",
    )
    parser.add_argument(
        "--input",
        action="append",
        default=[],
        metavar="CASE_ID=JSON_PATH",
        help="Evaluate an existing warp34 bench JSON file for CASE_ID (repeatable)",
    )
    parser.add_argument(
        "--live",
        action="store_true",
        help="Run warp34 bench live for selected cases",
    )
    parser.add_argument(
        "--suite",
        choices=("quick", "full"),
        default="quick",
        help="Preset suite when running with --live and no explicit --case (default: quick)",
    )
    parser.add_argument(
        "--case",
        action="append",
        default=[],
        dest="cases",
        help="Specific case ID(s) from bench_baselines.json to run in --live mode",
    )
    parser.add_argument(
        "--warp34-bin",
        type=pathlib.Path,
        default=DEFAULT_WARP34_BIN,
        help="Path to compiled warp34 release binary",
    )
    parser.add_argument(
        "--out-dir",
        type=pathlib.Path,
        default=None,
        help="Optional directory to persist per-case JSONs and summary.md",
    )
    parser.add_argument(
        "--markdown-out",
        type=pathlib.Path,
        default=None,
        help="Optional path to write the rendered Markdown report",
    )
    args = parser.parse_args(argv)

    baselines_doc = load_baselines(args.baselines)
    all_cases = baselines_doc["cases"]

    if not args.input and not args.live:
        parser.error("pass at least one --input CASE_ID=PATH or --live")

    arch, gpu_count = detect_host_arch(args.arch)
    evaluations: list[CaseEvaluation] = []

    for item in args.input:
        case_id, json_path = parse_input_spec(item)
        with json_path.open("r", encoding="utf-8") as f:
            payload = json.load(f)
        evaluations.append(
            evaluate_case(
                case_id=case_id,
                payload=payload,
                baselines_doc=baselines_doc,
                arch=arch,
                max_regression_pct=args.max_regression_pct,
            )
        )

    if args.live:
        if not args.warp34_bin.exists():
            print(
                f"error: warp34 binary not found at {args.warp34_bin}; run `mise run build` first",
                file=sys.stderr,
            )
            return 2
        selected = (
            tuple(args.cases)
            if args.cases
            else (QUICK_SUITE_CASES if args.suite == "quick" else FULL_SUITE_CASES)
        )
        out_root_ctx = (
            tempfile.TemporaryDirectory(prefix="warp34_bench_gate_")
            if args.out_dir is None
            else None
        )
        try:
            out_dir = (
                 pathlib.Path(out_root_ctx.name)
                if out_root_ctx is not None
                else args.out_dir
            )
            assert out_dir is not None
            out_dir.mkdir(parents=True, exist_ok=True)
            for case_id in selected:
                if case_id not in all_cases:
                    print(f"error: unknown case {case_id!r}", file=sys.stderr)
                    return 2
                case_spec = all_cases[case_id]
                req_tp = int(case_spec.get("tp", 1))
                if req_tp > gpu_count:
                    print(
                        f"[bench-gate] skipping {case_id} (requires TP={req_tp}, found {gpu_count} visible GPU(s))",
                        file=sys.stderr,
                    )
                    continue
                if arch not in case_spec.get("hardware", {}):
                    print(
                        f"[bench-gate] skipping {case_id} (no baseline for {arch})",
                        file=sys.stderr,
                    )
                    continue
                out_json = out_dir / f"{case_id}.json"
                cmd = build_bench_command(args.warp34_bin, case_spec, out_json)
                print(f"[bench-gate] running {case_id}: {' '.join(cmd)}", file=sys.stderr)
                proc = subprocess.run(cmd, check=False, capture_output=True, text=True)
                if proc.returncode != 0:
                    print(
                        f"[bench-gate] {case_id} command failed (exit {proc.returncode}):\n{proc.stderr}",
                        file=sys.stderr,
                    )
                    return 1
                with out_json.open("r", encoding="utf-8") as f:
                    payload = json.load(f)
                evaluations.append(
                    evaluate_case(
                        case_id=case_id,
                        payload=payload,
                        baselines_doc=baselines_doc,
                        arch=arch,
                        max_regression_pct=args.max_regression_pct,
                    )
                )
        finally:
            if out_root_ctx is not None:
                out_root_ctx.cleanup()

    report = render_markdown_report(
        evaluations=evaluations,
        arch=arch,
        max_regression_pct=args.max_regression_pct,
    )
    print(report)

    if args.markdown_out is not None:
        args.markdown_out.parent.mkdir(parents=True, exist_ok=True)
        args.markdown_out.write_text(report + "\n", encoding="utf-8")
    elif args.out_dir is not None:
        args.out_dir.mkdir(parents=True, exist_ok=True)
        (args.out_dir / "summary.md").write_text(report + "\n", encoding="utf-8")

    return 0 if all(ev.passed for ev in evaluations) else 1


if __name__ == "__main__":
    raise SystemExit(main())
