"""Bounded client-side soak campaigns (not the standard campaign/audit schema).

Sessions replay fixtures; they do not execute autonomous tools. A disconnect is
only local evidence, and a successful drain probe is not pool conservation.
"""

from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import threading
import time
import uuid

from .config import digest, public_config
from .metrics import tokens, tpot
from .runner import Journal, provenance
from .telemetry import TelemetrySampler
from .transport import authorization, stream
from .workloads import append_replay, body, messages


def _number(value, name, *, minimum=0, positive=False, integer=False):
    if (isinstance(value, bool) or not isinstance(value, (int, float))
            or (integer and not isinstance(value, int))):
        raise ValueError(f"{name} has an invalid numeric type")
    try:
        finite = math.isfinite(value)
    except OverflowError:
        finite = False
    if not finite or value < minimum or (positive and value <= 0):
        raise ValueError(f"{name} is outside its finite permitted range")
    return value


def _resource_snapshot(sampler, pids, uuids):
    row = sampler.sample()

    def finite(value):
        if (isinstance(value, bool) or not isinstance(value, (int, float))
                or not math.isfinite(value) or value < 0):
            return None
        return value

    # Explicit UUID selection: unrelated cards must never satisfy a budget.
    # Keep unavailable/non-finite readings as null so failure evidence is JSON-safe.
    return {"timestamp_monotonic_s": finite(row.get("timestamp_monotonic_s")),
            "pid_rss_kib": {str(p): finite((row.get("pid_rss_kib") or {}).get(str(p))) for p in pids},
            "gpus": [{"uuid": g["uuid"], "memory_used_mib": finite(g.get("memory_used_mib"))}
                     for g in (row.get("gpus") or []) if g.get("uuid") in uuids],
            "errors": [{"source": e.get("source"), "type": e.get("type")}
                       for e in row.get("errors", [])]}


def _growth(baseline, final, pids, uuids, rss_budget, gpu_budget):
    rows, failures = [], []
    for kind, identities, budget in (("rss", pids, rss_budget), ("gpu", uuids, gpu_budget)):
        if budget is None:
            continue
        for identity in identities:
            def reading(sample):
                if not sample:
                    return None
                if kind == "rss":
                    value = sample.get("pid_rss_kib", {}).get(str(identity))
                else:
                    matches = [g.get("memory_used_mib") for g in sample.get("gpus", [])
                               if g.get("uuid") == identity]
                    value = matches[0] if len(matches) == 1 else None
                if (isinstance(value, bool) or not isinstance(value, (int, float))
                        or not math.isfinite(value) or value < 0):
                    return None
                return value / 1024 if kind == "rss" else value
            before, after = reading(baseline), reading(final)
            delta = after - before if before is not None and after is not None else None
            passed = delta is not None and delta <= budget
            rows.append({"resource": kind, "identity": str(identity), "baseline_mib": before,
                         "final_mib": after, "growth_mib": delta, "budget_mib": budget,
                         "status": "passed" if passed else "failed"})
            if not passed:
                failures.append(f"{kind}_growth_missing_or_exceeded")
    return rows, list(dict.fromkeys(failures))


def _slo_breaches(result, case):
    breaches = []
    for setting, field in (("ttft_slo_ms", "first_output_ms"), ("e2e_slo_ms", "e2e_ms"),
                           ("tpot_slo_ms", None)):
        limit = case.get(setting)
        if limit is None:
            continue
        value = result.get(field) if field else tpot(result)
        if (isinstance(value, bool) or not isinstance(value, (int, float))
                or not math.isfinite(value) or value < 0 or value > limit):
            breaches.append(setting)
    return breaches


def execute_soak(cfg, out, engine_id, *, duration_s, min_completed, max_duration_s,
                 sustained_s=300, burst_s=30, burst_multiplier=2, cancel_every=0,
                 cancel_after_output_events=2, progress_timeout_s=120, drain_s=5,
                 max_error_rate=0, max_rss_growth_mib=None, max_gpu_growth_mib=None,
                 telemetry_interval=1, server_pids=(), gpu_uuids=()):
    """Return a soak-specific passed/failed receipt; persist it before re-raising Ctrl-C.

    The hard workload deadline starts after warmup/baseline. Warmup has its own
    max_duration_s bound. Drain and a single noncancelled probe are additional,
    bounded recovery checks. Transport must honor its total timeout; Python
    cannot forcibly terminate a misbehaving worker thread.
    """
    policy = dict(duration_s=duration_s, min_completed=min_completed, max_duration_s=max_duration_s,
                  sustained_s=sustained_s, burst_s=burst_s, burst_multiplier=burst_multiplier,
                  cancel_every=cancel_every, cancel_after_output_events=cancel_after_output_events,
                  progress_timeout_s=progress_timeout_s, drain_s=drain_s, max_error_rate=max_error_rate,
                  max_rss_growth_mib=max_rss_growth_mib, max_gpu_growth_mib=max_gpu_growth_mib,
                  telemetry_interval=telemetry_interval)
    for name in ("duration_s", "max_duration_s", "sustained_s", "burst_s",
                 "progress_timeout_s", "telemetry_interval"):
        _number(policy[name], name, positive=True)
    for name in ("min_completed", "burst_multiplier", "cancel_after_output_events"):
        _number(policy[name], name, positive=True, integer=True)
    _number(cancel_every, "cancel_every", integer=True)
    _number(drain_s, "drain_s")
    _number(max_error_rate, "max_error_rate")
    if max_error_rate > 1 or max_duration_s < duration_s:
        raise ValueError("invalid error rate or workload deadline")
    _number(sustained_s + burst_s, "phase period", positive=True)
    for name in ("max_rss_growth_mib", "max_gpu_growth_mib"):
        if policy[name] is not None:
            _number(policy[name], name)
    if isinstance(server_pids, (str, bytes)) or isinstance(gpu_uuids, (str, bytes)):
        raise ValueError("resource identities must be collections, not strings")
    try:
        server_pids, gpu_uuids = tuple(server_pids), tuple(gpu_uuids)
    except TypeError:
        raise ValueError("resource identities must be iterable collections") from None
    for pid in server_pids:
        _number(pid, "server PID", positive=True, integer=True)
    if any(not isinstance(u, str) or not u.strip() for u in gpu_uuids):
        raise ValueError("GPU UUIDs must be nonempty strings")
    if len(set(server_pids)) != len(server_pids) or len(set(gpu_uuids)) != len(gpu_uuids):
        raise ValueError("resource identities must be unique")
    if max_rss_growth_mib is not None and not server_pids:
        raise ValueError("RSS budget requires explicit server_pids")
    if max_gpu_growth_mib is not None and not gpu_uuids:
        raise ValueError("GPU budget requires explicit gpu_uuids")
    policy.update(server_pids=list(server_pids), gpu_uuids=list(gpu_uuids))
    engines = [e for e in cfg["engines"] if e["id"] == engine_id]
    if len(engines) != 1 or not engines[0].get("enabled", True) or engines[0].get("support") == "unsupported":
        raise ValueError("select exactly one runnable engine")
    engine, cases, run = engines[0], cfg["cases"], cfg["run"]
    if not cases or len({c["id"] for c in cases}) != len(cases):
        raise ValueError("soak requires distinct selected cases")
    for case in cases:
        if case.get("arrival", "closed") != "closed":
            raise ValueError("soak supports closed-loop arrivals only")
        _number(case["concurrency"], "case concurrency", positive=True, integer=True)
        for shape in [case] + [{**case, **m} for m in case.get("mix", [])]:
            if shape.get("cache") == "primed":
                raise ValueError("primed cache is not supported by soak")
            _number(shape.get("think_time_s", 0), "think time")
            for name in ("ttft_slo_ms", "e2e_slo_ms", "tpot_slo_ms"):
                if shape.get(name) is not None:
                    _number(shape[name], name, positive=True)
        if case.get("kind") == "agent_replay" and not case["fixture_data"]["turns"]:
            raise ValueError("agent replay requires turns")
    cap = _number(run["max_inflight"], "max_inflight", positive=True, integer=True)
    timeout = _number(run["timeout_s"], "timeout_s", positive=True)
    warmups = _number(run.get("warmup_requests", 0), "warmup_requests", integer=True)
    base = min(max(c["concurrency"] for c in cases), cap)
    burst = base * burst_multiplier
    if burst > cap:
        raise ValueError("promised burst concurrency exceeds run.max_inflight")
    authorization(engine)  # Credential validation, no network or output directory yet.
    policy.update(base_concurrency=base, burst_concurrency=burst, request_timeout_s=timeout,
                  warmup_requests_per_case=warmups)

    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)
    run_id = uuid.uuid4().hex
    receipt = {"schema_version": 1, "kind": "soak", "run_id": run_id, "status": "failed",
               "interrupted": False, "engine_id": engine_id, "selected_cases": [c["id"] for c in cases],
               "policy": policy, "failure_reasons": [], "targets_met": False,
               "baseline": None, "final": None, "resource_growth": [], "probe": None,
               "workload_duration_s": 0, "peak_inflight_sessions": 0,
               "caveats": ["Fixture replay, not autonomous tool execution.",
                           "Local cancellation only; no server cancellation acknowledgment.",
                           "Drain probe success does not establish pool conservation.",
                           "No supervisor, restart, or fault qualification.",
                           "Resource budgets compare endpoints, not slopes; PID reuse is not tracked.",
                           "Hard workload limit excludes bounded warmup, telemetry, drain and probe."]}
    counts = {k: 0 for k in ("requests", "completed_requests", "unexpected_errors", "invalid_usage",
                             "injection_requested", "expected_cancellations", "completed_before_injection",
                             "prompt_tokens", "completion_tokens", "sessions", "successful_sessions",
                             "stopped_sessions", "warmup_requests", "probe_requests")}
    receipt["counts"] = counts
    slo = {c["id"]: {"breached_requests": 0, "ttft_slo_ms": 0, "e2e_slo_ms": 0,
                       "tpot_slo_ms": 0} for c in cases}
    receipt["case_slo_breaches"] = slo
    manifest = {"schema_version": 1, "kind": "soak", "run_id": run_id, "status": "running",
                "created_utc": datetime.now(timezone.utc).isoformat(), "config": public_config(cfg),
                "config_sha256": digest(cfg), "policy": policy, "policy_sha256": digest(policy),
                "provenance": {"not_collected": "initialization incomplete"},
                "selected_engines": [engine_id], "selected_cases": receipt["selected_cases"]}
    journal = Journal(out / "requests.jsonl")
    (out / "telemetry.jsonl").touch()
    lock, stop = threading.Lock(), threading.Event()
    sampler = executor = None
    started = None
    deadline = float("inf")
    last_progress = time.monotonic()
    sequence = 0
    interrupted = None
    case_indices = {}

    def fail(reason):
        if reason not in receipt["failure_reasons"]:
            receipt["failure_reasons"].append(reason)

    def session(case, phase, sid, end):
        nonlocal last_progress
        begin = time.monotonic()
        traffic = "measured" if phase in ("sustained", "burst") else phase
        # Global round-robin session IDs alias shape counts (e.g. three cases
        # and three shapes). Rotate independently within each case instead.
        with lock:
            key = (traffic, case["id"])
            case_index = case_indices.get(key, 0)
            case_indices[key] = case_index + 1
        request_case = {**case, **(case["mix"][case_index % len(case["mix"])] if case.get("mix") else {})}
        # messages() owns the unique/shared suffix. A session ID in this prefix
        # would accidentally defeat natural cache sharing, including bursts.
        history = messages(request_case, case_index, f"soak:{run_id}:{traffic}:{case['id']}")
        agent = case.get("kind") == "agent_replay"
        turns = case["fixture_data"]["turns"] if agent else [None]
        if phase == "probe":
            turns = turns[:1]  # One real inference, not another full replay session.
        outcome = "ok"
        try:
            for step, turn in enumerate(turns):
                if stop.is_set() or time.monotonic() >= end:
                    outcome = "stopped"
                    break
                if step:
                    delay = min(request_case.get("think_time_s", 0), max(0, end - time.monotonic()))
                    if stop.wait(delay) or time.monotonic() >= end:
                        outcome = "stopped"
                        break
                if turn is not None:
                    append_replay(history, turn, sid, step)
                payload = body(engine, request_case, history)
                remaining = end - time.monotonic()
                if stop.is_set() or remaining <= 0:
                    outcome = "stopped"
                    break
                with lock:
                    if phase in ("sustained", "burst"):
                        counts["requests"] += 1
                        inject = bool(cancel_every and counts["requests"] % cancel_every == 0)
                        counts["injection_requested"] += int(inject)
                    else:
                        counts[phase + "_requests"] += 1
                        inject = False
                try:
                    result = stream(engine, payload, min(timeout, remaining),
                                    cancel_after_output_events=cancel_after_output_events if inject else None)
                except Exception as exc:
                    result = {"status": "error", "error_type": type(exc).__name__}
                # Arbitrary exception strings may include credentials or request text.
                result.pop("error", None)
                if agent and result.get("status") == "ok" and result.get("tool_calls"):
                    result["status"] = "agent_replay_error"
                usage = tokens(result) if result.get("status") == "ok" else None
                invalid = result.get("status") == "ok" and usage is None
                if invalid:
                    result["status"] = "invalid_usage"
                expected = (inject and result.get("status") == "cancelled"
                            and result.get("cancellation_requested") is True
                            and result.get("cancellation_observed") is True)
                good = result.get("status") == "ok"
                breaches = _slo_breaches(result, request_case) if good else []
                journal.put({**result, "type": "request", "engine_id": engine_id, "case_id": case["id"],
                             "phase": phase, "session_id": sid, "step": step,
                             "request_id": f"{run_id}:{sid}:{step}", "request": payload,
                             "request_sha256": digest(payload), "input_messages_sha256": digest(history),
                             "injection_scheduled": inject, "expected_cancellation": expected,
                             "slo_breaches": breaches})
                with lock:
                    if phase in ("sustained", "burst"):
                        counts["expected_cancellations"] += int(expected)
                        counts["unexpected_errors"] += int(not good and not expected)
                        counts["invalid_usage"] += int(invalid)
                        if good:
                            counts["completed_requests"] += 1
                            counts["completed_before_injection"] += int(inject)
                            counts["prompt_tokens"] += usage[0]
                            counts["completion_tokens"] += usage[1]
                            last_progress = time.monotonic()
                        if breaches:
                            slo[case["id"]]["breached_requests"] += 1
                            for breach in breaches:
                                slo[case["id"]][breach] += 1
                if not good:
                    outcome = "cancelled" if expected else "error"
                    break
                if agent:
                    assistant = {"role": "assistant", "content": result.get("content", "")}
                    if result.get("reasoning"):
                        assistant["reasoning_content"] = result["reasoning"]
                    history.append(assistant)
        except BaseException:
            outcome = "error"
            raise
        finally:
            journal.put({"type": "session", "engine_id": engine_id, "case_id": case["id"],
                         "session_id": sid, "phase": phase, "status": outcome,
                         "e2e_ms": (time.monotonic() - begin) * 1000})
            if phase in ("sustained", "burst"):
                with lock:
                    counts["sessions"] += 1
                    counts["successful_sessions"] += int(outcome == "ok")
                    counts["stopped_sessions"] += int(outcome == "stopped")
        return outcome

    try:
        manifest["provenance"] = provenance()
        (out / "manifest.json").write_text(json.dumps(manifest, indent=2, allow_nan=False) + "\n")
        sampler = TelemetrySampler(out / "telemetry.jsonl", interval_s=telemetry_interval, pids=server_pids)
        sampler.__enter__()
        warm_end = time.monotonic() + max_duration_s
        warm_ok = True
        for case in cases:
            for _ in range(warmups):
                outcome = session(case, "warmup", sequence, warm_end)
                sequence += 1
                if outcome != "ok":
                    fail("warmup_failed")
                    warm_ok = False
                    break
            if not warm_ok:
                break
        receipt["baseline"] = _resource_snapshot(sampler, server_pids, gpu_uuids)
        if warm_ok:
            started = last_progress = time.monotonic()
            deadline = started + max_duration_s
            executor = ThreadPoolExecutor(max_workers=burst, thread_name_prefix="soak")
            pending = set()
            previous_phase = None
            admitting = True
            submission = 0
            while admitting or pending:
                now = time.monotonic()
                with lock:
                    complete, progress = counts["completed_requests"], last_progress
                # Reap before checking deadlines, including the duration == limit case.
                finished = {f for f in pending if f.done()}
                for future in finished:
                    future.result()
                pending -= finished
                if admitting and now - started >= duration_s and complete >= min_completed:
                    receipt["targets_met"] = True
                    admitting = False
                    journal.put({"type": "admission_stopped", "reason": "targets_met",
                                 "elapsed_s": now - started})
                if not admitting and not pending:
                    break
                if now >= deadline:
                    fail("max_duration_exceeded")
                    stop.set()
                    break
                if now - progress >= progress_timeout_s:
                    fail("completion_progress_timeout")
                    stop.set()
                    break
                cycle = int((now - started) // (sustained_s + burst_s))
                phase = "sustained" if (now - started) % (sustained_s + burst_s) < sustained_s else "burst"
                phase_key = (cycle, phase)
                target = base if phase == "sustained" else burst
                if phase_key != previous_phase:
                    journal.put({"type": "phase", "cycle": cycle, "phase": phase,
                                 "elapsed_s": now - started, "concurrency": target})
                    previous_phase = phase_key
                # Never build an executor backlog.
                if admitting and len(pending) < target:
                    # One admission per iteration so targets/progress are rechecked.
                    case = cases[submission % len(cases)]
                    pending.add(executor.submit(session, case, phase, sequence, deadline))
                    sequence += 1
                    submission += 1
                    receipt["peak_inflight_sessions"] = max(receipt["peak_inflight_sessions"], len(pending))
                    continue
                if pending:
                    wait(pending, timeout=min(.05, max(0, deadline - now),
                                              max(0, progress_timeout_s - (now - progress))),
                         return_when=FIRST_COMPLETED)
            executor.shutdown(wait=True)
            executor = None
            receipt["workload_duration_s"] = time.monotonic() - started
            if counts["stopped_sessions"]:
                fail("incomplete_sessions")
        # Recovery is independent of the workload stop event and target counters.
        journal.put({"type": "drain", "duration_s": drain_s})
        time.sleep(drain_s)
        receipt["final"] = _resource_snapshot(sampler, server_pids, gpu_uuids)
        stop.clear()
        probe_end = time.monotonic() + min(timeout, progress_timeout_s, max_duration_s)
        receipt["probe"] = {"status": session(cases[0], "probe", sequence, probe_end),
                            "session_id": sequence, "noncancelled": True}
        if receipt["probe"]["status"] != "ok":
            fail("drain_probe_failed")
    except BaseException as exc:
        stop.set()
        if isinstance(exc, KeyboardInterrupt):
            receipt["interrupted"] = True
            interrupted = exc
            fail("interrupted")
        else:
            fail("execution_failed")
        receipt["error_type"] = type(exc).__name__
    finally:
        stop.set()
        if executor is not None:
            executor.shutdown(wait=True)
        if started is not None and not receipt["workload_duration_s"]:
            receipt["workload_duration_s"] = time.monotonic() - started
        if sampler is not None:
            try:
                sampler.stop()
            except Exception as exc:
                fail("telemetry_failed")
                receipt["telemetry_error_type"] = type(exc).__name__
        receipt["resource_growth"], resource_failures = _growth(
            receipt["baseline"], receipt["final"], server_pids, gpu_uuids,
            max_rss_growth_mib, max_gpu_growth_mib)
        for reason in resource_failures:
            fail(reason)
        denominator = counts["requests"] - counts["expected_cancellations"]
        receipt["unexpected_error_rate"] = counts["unexpected_errors"] / denominator if denominator else 0
        receipt["error_rate_denominator"] = "measured requests excluding expected local cancellations"
        if receipt["unexpected_error_rate"] > max_error_rate:
            fail("max_error_rate_exceeded")
        if any(row["breached_requests"] for row in slo.values()):
            fail("case_slo_failed")
        if not receipt["targets_met"]:
            fail("targets_not_met")
        receipt["status"] = "failed" if receipt["failure_reasons"] else "passed"
        manifest.update(status="interrupted" if receipt["interrupted"] else receipt["status"],
                        finished_utc=datetime.now(timezone.utc).isoformat())
        journal.close()
        for name, value in (("manifest.json", manifest), ("soak_summary.json", receipt)):
            (out / name).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
        inventory = {}
        for path in sorted(out.iterdir()):
            if path.is_file() and path.name != "SHA256SUMS.json":
                sha = hashlib.sha256()
                with path.open("rb") as file:
                    for chunk in iter(lambda: file.read(1024 * 1024), b""):
                        sha.update(chunk)
                inventory[path.name] = sha.hexdigest()
        (out / "SHA256SUMS.json").write_text(json.dumps(inventory, indent=2) + "\n")
    if interrupted is not None:
        raise interrupted
    return receipt
