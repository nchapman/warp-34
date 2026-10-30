"""Run with: python3 -m w34perf (from benchmarks/perf)."""

import argparse
import json
from pathlib import Path
import sys

from .config import load
from .report import render
from .runner import execute, MeasurementError


def main(argv=None):
    parser = argparse.ArgumentParser(description="Independent black-box inference performance suite")
    subs = parser.add_subparsers(dest="command", required=True)
    for name in ("validate", "plan", "run"):
        sub = subs.add_parser(name)
        sub.add_argument("config", type=Path)
        if name == "run":
            sub.add_argument("--engine", action="append", required=True,
                             help="explicit endpoint selection; repeat to select multiple")
            sub.add_argument("--case", action="append")
            sub.add_argument("--out", type=Path, required=True, help="new artifact directory")
    report = subs.add_parser("report")
    report.add_argument("directory", type=Path)
    report.add_argument("--format", choices=("markdown", "csv"), default="markdown")
    args = parser.parse_args(argv)
    try:
        if args.command == "report":
            markdown, csv = render(args.directory)
            # Keep original raw artifacts immutable; report is derived on stdout.
            sys.stdout.write(csv if args.format == "csv" else markdown)
            return 0
        cfg = load(args.config)
        if args.command == "validate":
            print(f"valid: {len(cfg['engines'])} deployments, {len(cfg['cases'])} cases")
        elif args.command == "plan":
            print(json.dumps({"repetitions": cfg["run"]["repetitions"],
                              "deployments": [{k: v for k, v in e.items() if k != "request_overrides"}
                                              for e in cfg["engines"]],
                              "cases": [{k: v for k, v in c.items() if k != "fixture_data"}
                                        for c in cfg["cases"]]}, indent=2))
        else:
            rows = execute(cfg, args.out, args.engine, args.case)
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
