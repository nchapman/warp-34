"""Run with: python3 -m w34perf (from benchmarks/perf)."""

import argparse
import json
from pathlib import Path
import sys

from .audit import audit
from .compare import compare
from .config import load
from .artifacts import save_inventory, verify_inventory
from .selection import select
from .preflight import preflight
from .quality import qualify
from .presets import apply_preset
from .report import render
from .runner import execute, MeasurementError


def main(argv=None):
    parser = argparse.ArgumentParser(description="Independent black-box inference performance suite")
    subs = parser.add_subparsers(dest="command", required=True)
    for name in ("validate", "plan", "run"):
        sub = subs.add_parser(name)
        sub.add_argument("config", type=Path)
        if name in ("plan", "run"):
            sub.add_argument("--engine", action="append", required=name == "run",
                             help="explicit endpoint selection; repeat to select multiple")
            sub.add_argument("--case", action="append")
            sub.add_argument("--preset", choices=("quick", "full", "release"))
            sub.add_argument("--gpus", type=int, choices=(1, 2), help="restrict to one- or two-card profiles")
            sub.add_argument("--exclude-engine", action="append", default=[], help="omit a deployment, retaining coverage")
        if name == "run":
            sub.add_argument("--out", type=Path, required=True, help="new artifact directory")
            sub.add_argument("--evidence", action="append", type=Path, default=[],
                             help="attach sanitized inventory/preflight/quality/startup evidence; repeatable")
            sub.add_argument("--telemetry-interval", type=float, help="sample GPU/host resources every N seconds")
            sub.add_argument("--server-pid", action="append", type=int, default=[], help="explicit host PID to sample RSS")
    soak = subs.add_parser("soak", help="bounded sustained/burst replay with an explicit pass/fail receipt")
    soak.add_argument("config", type=Path)
    soak.add_argument("--engine", required=True, help="one externally managed deployment")
    soak.add_argument("--out", type=Path, required=True, help="new artifact directory")
    soak.add_argument("--duration-s", type=float, required=True, help="minimum measured workload duration")
    soak.add_argument("--min-completed", type=int, required=True, help="minimum successful measured responses, excluding injected disconnects")
    soak.add_argument("--max-duration-s", type=float, required=True, help="hard workload admission/request deadline; excludes setup and drain")
    soak.add_argument("--sustained-s", type=float, default=300)
    soak.add_argument("--burst-s", type=float, default=30)
    soak.add_argument("--burst-multiplier", type=int, default=2)
    soak.add_argument("--cancel-every", type=int, default=0, help="inject a local disconnect every N measured requests (0 disables)")
    soak.add_argument("--cancel-after-output-events", type=int, default=2)
    soak.add_argument("--progress-timeout-s", type=float, default=120, help="fail without successful completed responses for this interval")
    soak.add_argument("--drain-s", type=float, default=5, help="quiet interval before final real inference probe")
    soak.add_argument("--max-error-rate", type=float, default=0, help="allowed unexpected request error fraction, in [0,1]")
    soak.add_argument("--telemetry-interval", type=float, default=1)
    soak.add_argument("--server-pid", action="append", type=int, default=[])
    soak.add_argument("--gpu-uuid", action="append", default=[], help="explicit local GPU identity for memory-growth gates")
    soak.add_argument("--max-rss-growth-mib", type=float, help="per-PID post-warmup to post-drain RSS growth budget")
    soak.add_argument("--max-gpu-growth-mib", type=float, help="per-GPU post-warmup to post-drain memory growth budget")
    quality = subs.add_parser("quality", help="independent functional/retrieval HTTP qualification")
    quality.add_argument("config", type=Path)
    quality.add_argument("--engine", required=True)
    quality.add_argument("--fixture", type=Path, default=Path("fixtures/quality.json"))
    quality.add_argument("--out", type=Path, required=True)
    audit_parser = subs.add_parser("audit", help="inventory historical bundles without certifying publication")
    audit_parser.add_argument("directory", type=Path)
    comparison = subs.add_parser("compare", help="guarded cross-run evidence report; no automatic ranking")
    comparison.add_argument("directories", nargs="+", type=Path)
    report = subs.add_parser("report")
    report.add_argument("directory", type=Path)
    report.add_argument("--format", choices=("markdown", "csv"), default="markdown")
    inv = subs.add_parser("inventory", help="hash local model/tokenizer files without downloading")
    inv.add_argument("root", type=Path)
    inv.add_argument("--include", action="append", help="relative-name glob; repeat to select file sets")
    inv.add_argument("--out", type=Path, required=True)
    verify = subs.add_parser("verify", help="verify an immutable offline artifact inventory")
    verify.add_argument("inventory", type=Path)
    verify.add_argument("--root", type=Path, help="relocated artifact path")
    pre = subs.add_parser("preflight", help="check local inventory, indexed shards and advertised model ID")
    pre.add_argument("config", type=Path)
    pre.add_argument("--engine", required=True)
    pre.add_argument("--inventory", type=Path, required=True)
    pre.add_argument("--root", type=Path, help="local artifact root, if relocated")
    pre.add_argument("--out", type=Path, required=True, help="new evidence JSON file")
    args = parser.parse_args(argv)
    try:
        if args.command == "compare":
            print(json.dumps(compare(args.directories), indent=2))
            return 0
        if args.command == "audit":
            result = audit(args.directory)
            print(json.dumps(result, indent=2))
            return 1 if any(b["errors"] for b in result["bundles"]) else 0
        if args.command == "report":
            markdown, csv = render(args.directory)
            # Keep original raw artifacts immutable; report is derived on stdout.
            sys.stdout.write(csv if args.format == "csv" else markdown)
            return 0
        if args.command in ("inventory", "verify"):
            if args.command == "inventory":
                result = save_inventory(args.root, args.out, args.include)
            else:
                result = verify_inventory(args.inventory, args.root)
            print(json.dumps(result, indent=2))
            return 1 if result.get("status") == "mismatch" else 0
        cfg = load(args.config)
        if args.command in ("plan", "run"):
            cfg, preset_cases = apply_preset(cfg, args.preset)
            if preset_cases is not None:
                if args.case is not None:
                    raise ValueError("--preset and --case are mutually exclusive")
                args.case = preset_cases
        if args.command == "soak":
            from .soak import execute_soak
            result = execute_soak(
                cfg, args.out, args.engine,
                duration_s=args.duration_s, min_completed=args.min_completed,
                max_duration_s=args.max_duration_s, sustained_s=args.sustained_s,
                burst_s=args.burst_s, burst_multiplier=args.burst_multiplier,
                cancel_every=args.cancel_every,
                cancel_after_output_events=args.cancel_after_output_events,
                progress_timeout_s=args.progress_timeout_s, drain_s=args.drain_s,
                max_error_rate=args.max_error_rate,
                max_rss_growth_mib=args.max_rss_growth_mib,
                max_gpu_growth_mib=args.max_gpu_growth_mib,
                telemetry_interval=args.telemetry_interval,
                server_pids=args.server_pid, gpu_uuids=args.gpu_uuid)
            print(json.dumps(result, indent=2))
            return 0 if result["status"] == "passed" else 1
        if args.command == "quality":
            engines, _, _ = select(cfg, [args.engine])
            if len(engines) != 1:
                raise ValueError("quality requires one runnable deployment")
            result = qualify(engines[0], args.fixture, args.out, cfg["run"]["timeout_s"])
            print(json.dumps({"status": result["status"], "checks": len(result["checks"]),
                              "evidence": str(args.out)}, indent=2))
            return 0 if result["status"] == "passed" else 1
        if args.command == "preflight":
            result = preflight(cfg, args.engine, args.inventory, args.out, args.root)
            print(json.dumps(result, indent=2))
            return 1 if result["status"] == "failed" else 0
        if args.command == "validate":
            print(f"valid: {len(cfg['engines'])} deployments, {len(cfg['cases'])} cases")
        elif args.command == "plan":
            engines, cases, coverage = select(cfg, args.engine, args.case,
                                              gpu_count=args.gpus, exclude_ids=args.exclude_engine)
            print(json.dumps({"repetitions": cfg["run"]["repetitions"],
                              "selected_engines": [e["id"] for e in engines],
                              "coverage": coverage,
                              "deployments": [{k: v for k, v in e.items() if k != "request_overrides"}
                                              for e in cfg["engines"]],
                              "cases": [{k: v for k, v in c.items() if k != "fixture_data"}
                                        for c in cases]}, indent=2))
        else:
            rows = execute(cfg, args.out, args.engine, args.case,
                           gpu_count=args.gpus, exclude_ids=args.exclude_engine,
                           evidence_paths=args.evidence, telemetry_interval=args.telemetry_interval,
                           server_pids=args.server_pid)
            # Missing token usage is a failed measurement, not a silent token estimate.
            if any(r["failed_requests"] or not r["token_accounting_complete"]
                   or r["successful_sessions"] != r["sessions"] for r in rows):
                return 1
        return 0
    except MeasurementError as exc:
        print(f"w34perf: {exc}", file=sys.stderr)
        return 1
    except (ValueError, OSError, RuntimeError, KeyError) as exc:
        print(f"w34perf: {exc}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        print("w34perf: interrupted; partial evidence retained", file=sys.stderr)
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
