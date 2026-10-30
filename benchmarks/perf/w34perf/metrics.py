"""Client-observed metrics. Streaming chunks are deliberately NOT token counts."""

import math
import statistics


def percentile(values, p):
    if not values:
        return None
    values = sorted(values)
    position = (len(values) - 1) * p / 100
    low = math.floor(position)
    high = math.ceil(position)
    return values[low] + (values[high] - values[low]) * (position - low)


def distribution(values):
    return {"n": len(values), "mean": statistics.mean(values) if values else None,
            **{f"p{p}": percentile(values, p) for p in (50, 90, 95, 99)},
            "max": max(values) if values else None}


def tokens(result):
    usage = result.get("usage")
    if not isinstance(usage, dict):
        return None
    counts = [usage.get(k) for k in ("prompt_tokens", "completion_tokens")]
    if any(isinstance(n, bool) or not isinstance(n, int) or n < 0 for n in counts):
        return None
    if counts[1] == 0:
        return None
    return counts


def tpot(result):
    counts = tokens(result)
    first, last = result.get("first_output_ms"), result.get("last_output_ms")
    # This is an endpoint-span estimate using server usage, NOT true per-token ITL.
    # A single burst cannot establish a decode rate.
    if counts and counts[1] > 1 and first is not None and last is not None and last > first:
        return (last - first) / (counts[1] - 1)
    return None


def summarize(records, sessions, duration_s, case):
    good = [r for r in records if r["status"] == "ok"]
    accounted = [r for r in good if tokens(r) is not None]
    all_accounted = len(accounted) == len(good) and bool(good)
    gaps = []
    for record in good:
        stamps = []
        for event in record["events"]:
            choices = event["data"].get("choices", [])
            delta = choices[0].get("delta", {}) if choices else {}
            if any(delta.get(k) for k in ("content", "reasoning_content", "reasoning", "tool_calls")):
                stamps.append(event["at_ms"])
        gaps.extend(b - a for a, b in zip(stamps, stamps[1:]))
    slo = {"first_output_ms": case.get("ttft_slo_ms"), "e2e_ms": case.get("e2e_slo_ms")}
    eligible = []
    for r in good:
        if any(limit is not None and (r.get(key) is None or r[key] > limit)
               for key, limit in slo.items()):
            continue
        if "tpot_slo_ms" in case and (tpot(r) is None or tpot(r) > case["tpot_slo_ms"]):
            continue
        eligible.append(r)
    has_slo = any(v is not None for v in slo.values()) or "tpot_slo_ms" in case
    return {
        "duration_s": duration_s, "requests": len(records), "successful_requests": len(good),
        "failed_requests": len(records) - len(good),
        "usage_accounted_requests": len(accounted), "token_accounting_complete": all_accounted,
        "successful_requests_s": len(good) / duration_s,
        "completion_tokens": sum(tokens(r)[1] for r in accounted) if all_accounted else None,
        "completion_tokens_s": sum(tokens(r)[1] for r in accounted) / duration_s if all_accounted else None,
        "reported_prompt_tokens": sum(tokens(r)[0] for r in accounted) if all_accounted else None,
        "first_output_ms": distribution([r["first_output_ms"] for r in good]),
        "first_text_ms": distribution([r["first_text_ms"] for r in good if r["first_text_ms"] is not None]),
        "e2e_ms": distribution([r["e2e_ms"] for r in good]),
        "usage_span_tpot_estimate_ms": distribution([tpot(r) for r in good if tpot(r) is not None]),
        "output_chunk_gap_ms": distribution(gaps),
        "completion_tokens_per_request": distribution([tokens(r)[1] for r in accounted]),
        "schedule_lag_ms": distribution([r["schedule_lag_ms"] for r in records if r.get("schedule_lag_ms") is not None]),
        "slo_goodput_requests_s": len(eligible) / duration_s if has_slo else None,
        "sessions": len(sessions), "successful_sessions": sum(s["status"] == "ok" for s in sessions),
        "session_e2e_ms": distribution([s["e2e_ms"] for s in sessions if s["status"] == "ok"]),
        "session_throughput_s": sum(s["status"] == "ok" for s in sessions) / duration_s,
        "tail_warning": "p99 is exploratory with fewer than 100 successful requests" if len(good) < 100 else None,
        "errors": [{"request_id": r["request_id"], "status": r["status"], "error": r.get("error")}
                   for r in records if r["status"] != "ok"],
    }
