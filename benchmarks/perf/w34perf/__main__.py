"""Run with: python3 -m w34perf (from benchmarks/perf)."""

import argparse
import json
from pathlib import Path
import sys

from .config import load
from .artifacts import save_inventory, verify_inventory
from .selection import select
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
            sub.add_argument("--gpus", type=int, choices=(1, 2), help="restrict to one- or two-card profiles")
            sub.add_argument("--exclude-engine", action="append", default=[], help="omit a deployment, retaining coverage")
        if name == "run":
            sub.add_argument("--out", type=Path, required=True, help="new artifact directory")
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
    args = parser.parse_args(argv)
    try:
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
                           gpu_count=args.gpus, exclude_ids=args.exclude_engine)
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
