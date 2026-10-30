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
            flat.append({k: trial[k] for k in ("engine_id", "case_id", "repetition", "gpu_count",
                                              "weight_format", "kv_format", "speculation", "cache",
                                              "duration_s", "requests", "successful_requests", "failed_requests",
                                              "completion_tokens_s", "successful_requests_s", "slo_goodput_requests_s")})
    if not rows:
        lines.extend(["", "No measured trial completed. Inspect requests.jsonl for warmup/partial-request failures."])
    lines.extend(["", "## Publication checklist / limitations", ""])
    for (engine, case), trials in groups.items():
        reasons = sorted({r for t in trials for r in t["publication_blockers"]})
        lines.append(f"- **{escape(engine)} / {escape(case)}:** " + "; ".join(reasons))
        if any(t["tail_warning"] for t in trials):
            lines.append("  - Fewer than 100 successes/trial: p99 is exploratory, not a stable tail claim.")
    output = io.StringIO()
    writer = csv.DictWriter(output, fieldnames=list(flat[0]) if flat else [])
    writer.writeheader()
    writer.writerows(flat)
    return "\n".join(lines) + "\n", output.getvalue()
