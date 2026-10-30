"""Finite closed-loop and bounded open-loop trials, with append-only wire evidence."""

from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import random
import subprocess
import sys
import threading
import time

from . import __version__
from .config import digest, public_config
from .metrics import summarize
from .transport import stream
from .workloads import append_replay, body, messages


def command(args):
    try:
        result = subprocess.run(args, capture_output=True, text=True, timeout=15)
        return {"argv": args, "returncode": result.returncode, "stdout": result.stdout,
                "stderr": result.stderr}
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {"argv": args, "error": type(exc).__name__}


def provenance():
    root = Path(__file__).resolve().parents[1]
    source = {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
              for p in sorted(Path(__file__).parent.glob("*.py"))}
    return {"suite_version": __version__, "suite_source_sha256": digest(source), "source_files": source,
            "python": sys.version, "platform": platform.platform(), "cpu_count": os.cpu_count(),
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "git": command(["git", "-C", str(root), "rev-parse", "HEAD"]),
            "git_status": command(["git", "-C", str(root), "status", "--porcelain"]),
            "gpu": command(["nvidia-smi", "--query-gpu=index,uuid,name,driver_version,memory.total,power.limit,pcie.link.gen.current,pcie.link.width.current", "--format=csv"]),
            "topology": command(["nvidia-smi", "topo", "-m"]),
            "gpu_processes": command(["nvidia-smi", "--query-compute-apps=pid,used_memory", "--format=csv"])}


class MeasurementError(RuntimeError):
    """A warmup or priming request failed, rather than a configuration error."""


class Journal:
    def __init__(self, path):
        self.file = path.open("x")
        self.lock = threading.Lock()

    def put(self, value):
        with self.lock:
            self.file.write(json.dumps(value, ensure_ascii=True, allow_nan=False) + "\n")
            self.file.flush()

    def close(self):
        self.file.close()


def run_trial(engine, case, run_cfg, phase, repetition, journal, *, namespace):
    records, sessions = [], []
    lock = threading.Lock()
    count = run_cfg["warmup_requests"] if phase == "warmup" else case["requests"]
    started = time.perf_counter()

    def session(index, due):
        session_start = time.perf_counter()
        shape = case["mix"][index % len(case["mix"])] if case.get("mix") else {}
        request_case = {**case, **shape}
        history = messages(request_case, index, namespace)
        is_agent = case.get("kind", "text") == "agent_replay"
        turns = case["fixture_data"]["turns"] if is_agent else [None]
        successful = True
        for step, turn in enumerate(turns):
            if step and case.get("think_time_s", 0):
                time.sleep(case["think_time_s"])
            if turn is not None:
                append_replay(history, turn, index, step)
            payload = body(engine, request_case, history)
            submitted = time.perf_counter()
            result = stream(engine, payload, run_cfg["timeout_s"])
            if is_agent and result["status"] == "ok" and result["tool_calls"]:
                result["status"] = "agent_replay_error"
                result["error"] = "unexpected live tool call in replay"
            result.update({"engine_id": engine["id"], "case_id": case["id"], "phase": phase,
                           "repetition": repetition, "session_id": index, "step": step,
                           "request_id": f"{engine['id']}:{case['id']}:{phase}:{repetition}:{index}:{step}",
                           "scheduled_at_ms": (due - started) * 1000 if step == 0 else None,
                           "sent_at_ms": (submitted - started) * 1000,
                           "schedule_lag_ms": max(0, (submitted - due) * 1000) if step == 0 else None,
                           "request": payload,
                           "request_sha256": digest(payload),
                           "input_messages_sha256": digest(history)})
            journal.put({"type": "request", **result})
            with lock:
                records.append(result)
            if result["status"] != "ok":
                successful = False
                break
            if is_agent:
                assistant = {"role": "assistant", "content": result["content"]}
                if result["reasoning"]:
                    assistant["reasoning_content"] = result["reasoning"]
                history.append(assistant)
        summary = {"engine_id": engine["id"], "case_id": case["id"], "phase": phase,
                   "repetition": repetition, "session_id": index,
                   "status": "ok" if successful else "error",
                   "e2e_ms": (time.perf_counter() - session_start) * 1000}
        journal.put({"type": "session", **summary})
        with lock:
            sessions.append(summary)

    arrival = case.get("arrival", "closed") if phase == "measured" else "closed"
    workers = min(case["concurrency"], count) if arrival == "closed" else run_cfg["max_inflight"]
    with ThreadPoolExecutor(max_workers=max(1, workers)) as executor:
        pending = set()
        rng = random.Random(run_cfg.get("seed", 34) + repetition)
        due = started
        for index in range(count):
            if arrival == "closed":
                if len(pending) >= workers:
                    finished, pending = wait(pending, return_when=FIRST_COMPLETED)
                    for future in finished:
                        future.result()
                due = time.perf_counter()
            else:
                if index:
                    rate = case["rate_rps"]
                    due += 1 / rate if arrival == "fixed" else rng.expovariate(rate)
                delay = due - time.perf_counter()
                if delay > 0:
                    time.sleep(delay)
                finished = {future for future in pending if future.done()}
                for future in finished:
                    future.result()
                pending -= finished
                if len(pending) >= workers:
                    # Do not queue behind the executor or slow down arrivals:
                    # preserve overload as a failed offered session/request.
                    failed = {"engine_id": engine["id"], "case_id": case["id"], "phase": phase,
                              "repetition": repetition, "session_id": index, "step": 0,
                              "request_id": f"{engine['id']}:{case['id']}:{phase}:{repetition}:{index}:0",
                              "status": "client_overload", "error": "max_inflight exhausted",
                              "scheduled_at_ms": (due - started) * 1000,
                              "schedule_lag_ms": None}
                    dropped_session = {k: failed[k] for k in ("engine_id", "case_id", "phase",
                                                               "repetition", "session_id", "status")}
                    dropped_session["e2e_ms"] = 0
                    with lock:
                        records.append(failed)
                        sessions.append(dropped_session)
                    journal.put({"type": "request", **failed})
                    journal.put({"type": "session", **dropped_session})
                    continue
            pending.add(executor.submit(session, index, due))
        for future in pending:
            future.result()
    duration = max(time.perf_counter() - started, 1e-9)
    return summarize(records, sessions, duration, case)


def blockers(engine, cfg, summary):
    reasons = []
    for key in ("version", "model_revision", "weight_format", "kv_format", "parallelism",
                "speculation", "cache_config", "launch_command", "artifact_sha256",
                "tokenizer_revision", "chat_template_sha256", "source_model_revision",
                "quantization_details", "gpu_ids"):
        if not engine.get(key) or "unknown" in str(engine[key]).lower():
            reasons.append(f"unverified {key}")
    if not engine.get("quality_evidence"):
        reasons.append("no independent quality evidence for this 4-bit/KV deployment")
    if cfg["run"]["repetitions"] < 3:
        reasons.append("fewer than three repetitions")
    if summary["failed_requests"] or summary["successful_sessions"] != summary["sessions"]:
        reasons.append("failed requests or incomplete agent sessions; publish failures alongside metrics")
    if not summary["token_accounting_complete"]:
        reasons.append("missing/invalid server usage; no token-throughput claim")
    reasons.append("manual release audit required: hardware isolation, artifact/quality evidence, cache and template settings")
    return reasons


def execute(cfg, out, engine_ids=None, case_ids=None):
    engines = [e for e in cfg["engines"] if engine_ids is None or e["id"] in engine_ids]
    cases = [c for c in cfg["cases"] if case_ids is None or c["id"] in case_ids]
    if not engines or not cases:
        raise ValueError("selection matched no engines or cases")
    for selection, available, label in ((engine_ids, cfg["engines"], "engine"),
                                        (case_ids, cfg["cases"], "case")):
        if selection and set(selection) - {item["id"] for item in available}:
            raise ValueError(f"unknown {label} selection")
    if len({e["base_url"].rstrip("/") for e in engines}) != len(engines):
        raise ValueError("selected deployments share an endpoint; launch and measure one at a time")
    for engine in engines:
        if engine.get("api_key_env") and not os.environ.get(engine["api_key_env"]):
            raise ValueError(f"missing credential environment variable {engine['api_key_env']}")
    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)
    manifest = {"schema_version": 1, "created_utc": datetime.now(timezone.utc).isoformat(),
                "config": public_config(cfg), "config_sha256": digest(cfg),
                "provenance": provenance(), "status": "running",
                "selected_engines": [e["id"] for e in engines],
                "selected_cases": [c["id"] for c in cases]}
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    journal = Journal(out / "requests.jsonl")
    summaries = []
    try:
        for engine in engines:
            for case in cases:
                print(f"{engine['id']} / {case['id']}", flush=True)
                for repetition in range(cfg["run"]["repetitions"]):
                    ns = f"seed{cfg['run'].get('seed', 34)}:{case['id']}:rep{repetition}"
                    if cfg["run"]["warmup_requests"]:
                        warm = run_trial(engine, case, cfg["run"], "warmup", repetition,
                                         journal, namespace=ns + ":warmup")
                        if warm["failed_requests"]:
                            raise MeasurementError(f"warmup failed for {engine['id']}; see requests.jsonl")
                    if case.get("cache", "unique") == "primed":
                        prime_cfg = {**cfg["run"], "warmup_requests": case["requests"]}
                        prime = run_trial(engine, case, prime_cfg, "prime", repetition,
                                          journal, namespace=ns)
                        if prime["failed_requests"]:
                            raise MeasurementError("cache priming failed; see requests.jsonl")
                    summary = run_trial(engine, case, cfg["run"], "measured", repetition,
                                        journal, namespace=ns)
                    row = {"engine_id": engine["id"], "case_id": case["id"],
                           "repetition": repetition, "gpu_count": engine["gpu_count"],
                           "cohort": engine.get("cohort", "4bit-deployment-equivalent"),
                           "weight_format": engine["weight_format"], "kv_format": engine["kv_format"],
                           "speculation": engine["speculation"], "cache": case.get("cache", "unique"),
                           "fixture_sha256": case["fixture_sha256"],
                           "publication_blockers": blockers(engine, cfg, summary), **summary}
                    summaries.append(row)
                    (out / "summary.json").write_text(json.dumps(summaries, indent=2) + "\n")
                    print(f"  rep {repetition}: {summary['successful_requests']}/{summary['requests']} "
                          f"requests, {summary['completion_tokens_s']} tok/s", flush=True)
        manifest["status"] = "complete"
    except BaseException as exc:
        manifest["status"] = "interrupted" if isinstance(exc, KeyboardInterrupt) else "failed"
        manifest["error_type"] = type(exc).__name__
        raise
    finally:
        journal.close()
        manifest["finished_utc"] = datetime.now(timezone.utc).isoformat()
        (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
        inventory = {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                     for p in sorted(out.iterdir()) if p.is_file() and p.name != "SHA256SUMS.json"}
        (out / "SHA256SUMS.json").write_text(json.dumps(inventory, indent=2) + "\n")
    return summaries
