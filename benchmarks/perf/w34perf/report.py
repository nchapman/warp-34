"""Human/shareable tables without merging unlike quantization configurations."""

import csv
import io
import json
from pathlib import Path
import statistics


def render(directory):
    directory = Path(directory)
    manifest = json.loads((directory / "manifest.json").read_text())
    summary_path = directory / "summary.json"
    rows = json.loads(summary_path.read_text()) if summary_path.exists() else []
    coverage_path = directory / "coverage.json"
    coverage = json.loads(coverage_path.read_text()) if coverage_path.exists() else []
    groups = {}
    for row in rows:
        groups.setdefault((row["engine_id"], row["case_id"]), []).append(row)
    lines = ["# Inference serving performance", "",
             f"Run status: **{manifest['status']}**. Finite trials include fill/drain; not steady-state capacity.",
             "4-bit deployment equivalents, not bit-identical weights. No automatic winner ranking.",
             "First output includes visible reasoning/tool deltas. TPOT is a usage/span estimate, not token ITL.",
             "", "Values below are medians of per-trial statistics; JSON retains every trial and distribution.", "",
             "| Deployment | Case | GPUs | Weights / KV | Reps | Req OK/total | tok/s median [min, max] | First output p50/p95 ms | E2E p95 ms |",
             "|---|---|---:|---|---:|---:|---|---|---:|"]
    flat = []

    def value(v):
        return "n/a" if v is None else f"{v:.2f}"

    def escape(v):
        return str(v).replace("|", "\\|").replace("\n", " ")

    for (engine, case), trials in groups.items():
        def median(key, sub=None):
            vals = [t[key][sub] if sub else t[key] for t in trials]
            # Do not hide a trial's invalid accounting by averaging only valid trials.
            return statistics.median(vals) if all(v is not None for v in vals) else None
        speeds = [t["completion_tokens_s"] for t in trials]
        speed = (f"{value(statistics.median(speeds))} [{value(min(speeds))}, {value(max(speeds))}]"
                 if all(v is not None for v in speeds) else "n/a")
        t = trials[0]
        ok, total = sum(r["successful_requests"] for r in trials), sum(r["requests"] for r in trials)
        lines.append(f"| {escape(engine)} | {escape(case)} | {t['gpu_count']} | "
                     f"{escape(t['weight_format'])} / {escape(t['kv_format'])} | {len(trials)} | {ok}/{total} | "
                     f"{speed} | {value(median('first_output_ms', 'p50'))}/{value(median('first_output_ms', 'p95'))} | "
                     f"{value(median('e2e_ms', 'p95'))} |")
        for trial in trials:
            failed = (trial["failed_requests"] or not trial["token_accounting_complete"]
                      or trial["successful_sessions"] != trial["sessions"])
            flat.append({"row_type": "trial", "status": "measurement_failed" if failed else "measured", "reason": "",
                         **{k: trial[k] for k in ("engine_id", "case_id", "repetition", "gpu_count",
                                              "weight_format", "kv_format", "speculation", "cache",
                                              "duration_s", "requests", "successful_requests", "failed_requests",
                                              "completion_tokens_s", "successful_requests_s", "slo_goodput_requests_s")}})
    if not rows:
        message = ("No runnable deployments selected; no requests were sent."
                   if manifest["status"] == "skipped" else
                   "No measured trial completed. Inspect requests.jsonl for warmup/partial-request failures.")
        lines.extend(["", message])
    has_longctx = any(
        (t.get("peak_prompt_tokens") or 0) >= 4096 or t.get("cached_prompt_tokens") is not None
        for t in rows
    )
    if has_longctx:
        lines.extend([
            "", "## Long-context & prefix-cache breakdown", "",
            "Cold bursts have >= 4,096 uncached prompt tokens; warm turns have cached_tokens > 0 and < 4,096 uncached tokens.",
            "",
            "| Deployment | Case | Peak prompt tok | Cache hit % | Warm-turn TTFT p50 ms | Cold-burst TTFT p50 ms | Cold prefill tok/s p50 | TPOT p50 ms |",
            "|---|---|---:|---:|---:|---:|---:|---:|",
        ])
        for (engine, case), trials in groups.items():
            def med_opt(key, sub=None):
                vals = [t.get(key, {}).get(sub) if sub else t.get(key) for t in trials]
                valid = [v for v in vals if v is not None]
                return statistics.median(valid) if len(valid) == len(trials) and valid else None
            peak_tok = med_opt("peak_prompt_tokens")
            hit_ratio = med_opt("prefix_cache_hit_ratio")
            lines.append(
                f"| {escape(engine)} | {escape(case)} | "
                f"{'n/a' if peak_tok is None else int(round(peak_tok))} | "
                f"{'n/a' if hit_ratio is None else f'{hit_ratio * 100:.1f}%'} | "
                f"{value(med_opt('warm_turn_ttft_ms', 'p50'))} | "
                f"{value(med_opt('cold_burst_ttft_ms', 'p50'))} | "
                f"{value(med_opt('cold_prefill_tokens_s', 'p50'))} | "
                f"{value(med_opt('usage_span_tpot_estimate_ms', 'p50'))} |"
            )
    if any(t.get("steps") for t in rows):
        lines.extend(["", "## Agent replay per-step latency", "",
                      "Step numbers are zero-based. Token counts and TTFT are medians of per-trial p50s; missing cache usage stays unknown.", "",
                      "| Deployment | Case | Step | Req OK/total | Prompt tok | Cached tok | Uncached tok | TTFT ms |",
                      "|---|---|---:|---:|---:|---:|---:|---:|"])
        for (engine, case), trials in groups.items():
            steps = sorted({s["step"] for t in trials for s in t.get("steps", [])})
            for step in steps:
                entries = [s for t in trials for s in t.get("steps", []) if s["step"] == step]
                def step_median(key):
                    vals = [(s.get(key) or {}).get("p50") for s in entries]
                    return (statistics.median(vals) if len(vals) == len(trials)
                            and all(v is not None for v in vals) else None)
                ok = sum(s["successful_requests"] for s in entries)
                total = sum(s["requests"] for s in entries)
                lines.append(f"| {escape(engine)} | {escape(case)} | {step} | {ok}/{total} | "
                             f"{value(step_median('prompt_tokens'))} | {value(step_median('cached_tokens'))} | "
                             f"{value(step_median('uncached_tokens'))} | {value(step_median('first_output_ms'))} |")
    if coverage:
        lines.extend(["", "## Campaign coverage", "",
                      "Unsupported/disabled/filtered entries are not failures or zero-throughput measurements.", "",
                      "| Deployment | Case | GPUs | Status | Repetitions completed | Reason |",
                      "|---|---|---:|---|---:|---|"])
        for cell in coverage:
            lines.append(f"| {escape(cell['engine_id'])} | {escape(cell['case_id'])} | {cell['gpu_count']} | "
                         f"{escape(cell['status'])} | {cell['completed_repetitions']}/{cell['required_repetitions']} | "
                         f"{escape(cell['reason'] or '')} |")
            flat.append({"row_type": "coverage", **{k: cell[k] for k in
                         ("engine_id", "case_id", "gpu_count", "status", "reason", "support",
                          "completed_repetitions", "required_repetitions")}})
    lines.extend(["", "## Publication checklist / limitations", ""])
    for (engine, case), trials in groups.items():
        reasons = sorted({r for t in trials for r in t["publication_blockers"]})
        lines.append(f"- **{escape(engine)} / {escape(case)}:** " + "; ".join(reasons))
        if any(t["tail_warning"] for t in trials):
            lines.append("  - Fewer than 100 successes/trial: p99 is exploratory, not a stable tail claim.")
    output = io.StringIO()
    fields = ("row_type", "status", "reason", "engine_id", "case_id", "repetition", "gpu_count",
              "weight_format", "kv_format", "speculation", "cache", "duration_s", "requests",
              "successful_requests", "failed_requests", "completion_tokens_s", "successful_requests_s",
              "slo_goodput_requests_s", "support", "completed_repetitions", "required_repetitions")
    writer = csv.DictWriter(output, fieldnames=fields)
    writer.writeheader()
    writer.writerows(flat)
    return "\n".join(lines) + "\n", output.getvalue()
