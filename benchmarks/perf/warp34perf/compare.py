"""Conservative, evidence-preserving cross-run comparison (never a ranking)."""

import hashlib
import itertools
import json
import math
from pathlib import Path

from .resources import summarize_resources

from .audit import _manifest_parts, _objects


def _number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _unknown(value):
    return value is None or value == "unknown" or value == "" or value == {}


def compare(directories):
    """Read bundles without modifying them; keep deployments and trials separate.

    Compatibility means recorded experimental controls match, not verified weights
    or publication readiness. Hardware matching is deliberately exact (including
    UUIDs when recorded). No latency statistic is relabeled as token ITL.
    """
    result = {"schema_version": 1, "runs": [], "deployments": [], "comparisons": [],
              "publication_ready": False, "rankings": [],
              "limitations": ["No automatic readiness certification or winner ranking.",
                              "Finite-trial throughput includes fill/drain.",
                              "Usage/span TPOT and chunk timing are not token ITL.",
                              "Matching metadata does not prove bit-identical artifacts."]}
    for index, directory in enumerate(directories):
        directory = Path(directory)
        run = {"run_id": index, "directory": str(directory), "evidence_sha256": {}, "blockers": []}
        data = {}
        for name, kind in (("manifest", dict), ("summary", list), ("coverage", list)):
            path = directory / (name + ".json")
            try:
                raw = path.read_bytes()
                run["evidence_sha256"][path.name] = hashlib.sha256(raw).hexdigest()
                value = json.loads(raw, parse_constant=lambda s: (_ for _ in ()).throw(ValueError(s)))
                if not isinstance(value, kind):
                    raise ValueError("unexpected JSON shape")
                data[name] = value
            except (OSError, ValueError) as exc:
                run["blockers"].append(f"{path.name}: {type(exc).__name__}: {exc}")
                data[name] = kind()
        manifest, rows = data["manifest"], data["summary"]
        run.update(manifest=manifest, coverage=data["coverage"],
                   resources=summarize_resources(directory / "telemetry.jsonl"))
        telemetry = directory / "telemetry.jsonl"
        if telemetry.is_file():
            run["evidence_sha256"][telemetry.name] = hashlib.sha256(telemetry.read_bytes()).hexdigest()
        if not rows:
            run["blockers"].append("No measured trials")
        if manifest.get("status") != "complete":
            run["blockers"].append("Run is not complete")
        run["blockers"].extend(manifest.get("publication_blockers", []))
        for cell in _objects(data["coverage"], "coverage.json", run["blockers"]):
            if cell.get("status") != "measured":
                run["blockers"].append({"coverage": cell})
        result["runs"].append(run)
        cfg, provenance = _manifest_parts(manifest, run["blockers"])
        groups = {}
        for row in rows:
            if not isinstance(row, dict):
                run["blockers"].append("Invalid summary row")
                continue
            engine = next((e for e in cfg.get("engines", []) if e.get("id") == row.get("engine_id")), {})
            case = next((c for c in cfg.get("cases", []) if c.get("id") == row.get("case_id")), {})
            controls = {"source": provenance.get("suite_source_sha256"),
                        "case_id": row.get("case_id"), "fixture": row.get("fixture_sha256"),
                        "request_shape": case or None, "run_settings": cfg.get("run"),
                        "gpu": provenance.get("gpu"), "topology": provenance.get("topology"),
                        "visible_devices": provenance.get("cuda_visible_devices")}
            for key in ("gpu_count", "cohort", "model_revision", "parallelism", "cache_config",
                        "artifact_sha256", "tokenizer_revision", "chat_template_sha256",
                        "activation_dtype", "source_model_revision"):
                controls[key] = row.get(key, engine.get(key))
            controls["request_overrides"] = engine.get("request_overrides", {})
            sweeps = {key: row.get(key, engine.get(key)) for key in ("weight_format", "kv_format", "speculation")}
            identity = {key: row.get(key, engine.get(key))
                        for key in ("engine", "version", "container")}
            # Splitting on all controls avoids silently pooling changed trial metadata.
            key = json.dumps([row.get("engine_id"), identity, controls, sweeps], sort_keys=True)
            if key not in groups:
                groups[key] = {"run_id": index, "engine_id": row.get("engine_id"),
                               "case_id": row.get("case_id"), "controls": controls,
                               "identity": identity, "sweeps": sweeps, "trials": []}
            groups[key]["trials"].append(row)
        for group in groups.values():
            trials = group["trials"]
            blockers = list(run["blockers"])
            for key, value in group["controls"].items():
                if key not in ("visible_devices", "request_overrides") and _unknown(value):
                    blockers.append(f"Unknown control: {key}")
                if isinstance(value, dict) and ("error" in value or value.get("returncode", 0) != 0):
                    blockers.append(f"Unavailable control: {key}")
            for key, value in group["identity"].items():
                if _unknown(value):
                    blockers.append(f"Unknown identity: {key}")
            speeds = []
            for trial in trials:
                blockers.extend(trial.get("publication_blockers", []))
                if trial.get("failed_requests", 0) or trial.get("successful_sessions") != trial.get("sessions"):
                    blockers.append("Failed requests or sessions retained")
                duration, tokens = trial.get("duration_s"), trial.get("completion_tokens")
                valid = (trial.get("token_accounting_complete") is True and _number(duration)
                         and duration > 0 and _number(tokens) and tokens >= 0)
                speeds.append(tokens / duration if valid else None)
                if not valid:
                    blockers.append("Missing usage or invalid duration; throughput unknown")
            complete = all(s is not None for s in speeds)
            group.update(id=len(result["deployments"]), blockers=blockers,
                         completion_tokens_s=(sum(t["completion_tokens"] for t in trials) /
                                              sum(t["duration_s"] for t in trials) if complete else None),
                         trial_completion_tokens_s=speeds,
                         trial_min=min(speeds) if complete else None,
                         trial_max=max(speeds) if complete else None,
                         publication_ready=False)
            result["deployments"].append(group)
    for left, right in itertools.combinations(result["deployments"], 2):
        differences = {k: {"left": v, "right": right["controls"][k]}
                       for k, v in left["controls"].items() if v != right["controls"][k]}
        sweep = {k: {"left": v, "right": right["sweeps"][k]}
                 for k, v in left["sweeps"].items() if v != right["sweeps"][k]}
        unknown = [b for g in (left, right) for b in g["blockers"]
                   if isinstance(b, str) and b.startswith(("Unknown control:", "Unavailable control:", "Unknown identity:"))]
        result["comparisons"].append({"left": left["id"], "right": right["id"],
                                      "status": "incompatible" if differences else "unverified" if unknown else
                                      "labeled_sweep" if sweep else "matching_controls",
                                      "differences": differences, "sweep_differences": sweep,
                                      "blockers": left["blockers"] + right["blockers"],
                                      "same_artifact_equivalent": False})
    return result
