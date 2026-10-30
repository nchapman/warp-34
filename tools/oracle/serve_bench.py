#!/usr/bin/env python3
"""Comparable raw-prompt SSE load benchmark (stdlib only).

Manifest: JSON {"prompts": ["full prompt for client 0", ...], ...}.
Other manifest fields (e.g. input_token_ids/tokenizer_hash) are preserved, not verified.
Timeout is urllib's socket-operation timeout, not an overall request deadline.
ITL uses all adjacent selected-token arrival gaps, including zero gaps in bursts;
TTFT is arrival of the first selected token, not necessarily visible text.
"""
import argparse
import concurrent.futures
import hashlib
import json
import math
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def digest(value):
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def percentile(values, q):
    ordered = sorted(values)
    return ordered[max(0, math.ceil(q * len(ordered)) - 1)] if ordered else None


def records(response):
    """SSE framing: urllib buffers fragmented HTTP reads into complete lines."""
    data = []
    for raw in response:
        line = raw.decode("utf-8").rstrip("\r\n")
        if not line:
            if data:
                yield "\n".join(data)
                data = []
        elif line.startswith("data:"):
            data.append(line[5:].removeprefix(" "))
    if data:
        raise ValueError("unterminated SSE record")


def request_one(base_url, payload, timeout, barrier):
    result = {"request": payload, "prompt": payload["prompt"],
              "prompt_sha256": digest(payload["prompt"]), "passed": False}
    arrivals, usages, errors = [], [], []
    first_event = first_visible = finish_time = None
    finishes = dones = 0
    req = urllib.request.Request(base_url.rstrip("/") + "/v1/completions",
                                 canonical(payload).encode(),
                                 {"Content-Type": "application/json", "Accept": "text/event-stream"})
    barrier.wait()
    start = time.perf_counter()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as response:
            result["http_status"] = response.status
            if response.status != 200:
                raise ValueError("HTTP status is not 200")
            if "text/event-stream" not in response.headers.get("Content-Type", "").lower():
                raise ValueError("response is not text/event-stream")
            for record in records(response):
                now = time.perf_counter() - start
                if record == "[DONE]":
                    dones += 1
                    continue
                if dones:
                    errors.append("JSON event after DONE")
                event = json.loads(record)
                if first_event is None:
                    first_event = now
                if "error" in event:
                    raise ValueError("server error: " + str(event["error"]))
                if usages:
                    errors.append("JSON event after final usage")
                if event.get("usage") is not None:
                    if not finishes and not any(c.get("finish_reason") is not None for c in event.get("choices", [])):
                        errors.append("usage before finish")
                    usages.append(event["usage"])
                for choice in event.get("choices", []):
                    if choice.get("index", 0) != 0:
                        errors.append("unexpected choice index")
                    text = choice.get("text", "")
                    if text and first_visible is None:
                        first_visible = now
                    logprobs = choice.get("logprobs")
                    tokens = logprobs.get("tokens") if isinstance(logprobs, dict) else None
                    if tokens is not None and (not isinstance(tokens, list) or
                                               not all(isinstance(t, str) for t in tokens)):
                        raise ValueError("invalid logprobs.tokens")
                    if text and not tokens:
                        errors.append("visible delta without selected-token logprobs; unsupported protocol")
                    if tokens:
                        if finishes:
                            errors.append("tokens after finish")
                        arrivals.extend([now] * len(tokens))
                    if choice.get("finish_reason") is not None:
                        finishes += 1
                        finish_time = now
    except Exception as exc:
        errors.append(type(exc).__name__ + ": " + str(exc))
    wall = time.perf_counter() - start
    if dones != 1:
        errors.append(f"expected one DONE, got {dones}")
    if finishes != 1:
        errors.append(f"expected one finish, got {finishes}")
    usage_count = usages[-1].get("completion_tokens") if usages and isinstance(usages[-1], dict) else None
    if len(usages) != 1 or type(usage_count) is not int:
        errors.append("expected exactly one final usage with integer completion_tokens")
    if len(arrivals) != payload["max_tokens"] or usage_count != len(arrivals):
        errors.append(f"token count mismatch: selected={len(arrivals)}, usage={usage_count}, expected={payload['max_tokens']}")
    gaps = [b - a for a, b in zip(arrivals, arrivals[1:])]
    result.update(passed=not errors, errors=errors, selected_tokens=len(arrivals),
                  server_usage=usages, token_arrival_s=arrivals, ttft_s=arrivals[0] if arrivals else None,
                  first_visible_text_ttft_s=first_visible, mean_itl_s=sum(gaps) / len(gaps) if gaps else None,
                  p50_itl_s=percentile(gaps, .50), p95_itl_s=percentile(gaps, .95),
                  request_wall_s=wall, first_event_s=first_event, finish_s=finish_time,
                  first_event_before_finish=first_event is not None and finish_time is not None and first_event < finish_time,
                  finish_count=finishes, done_count=dones)
    return result


def run_batch(base_url, model, n, tokens, timeout, prompts):
    payloads = [{"model": model, "prompt": prompts[i], "max_tokens": tokens,
                 "temperature": 0, "ignore_eos": True, "stream": True,
                 "stream_options": {"include_usage": True}, "logprobs": 0} for i in range(n)]
    protocol = {"method": "POST", "path": "/v1/completions", "requests": payloads}
    # Barrier action timestamps release, excluding thread-pool startup.
    release = []
    barrier = threading.Barrier(n, action=lambda: release.append(time.perf_counter()))
    with concurrent.futures.ThreadPoolExecutor(max_workers=n) as pool:
        futures = [pool.submit(request_one, base_url, p, timeout, barrier) for p in payloads]
        results = [f.result() for f in futures]
        wall = time.perf_counter() - release[0]
    passed = all(r["passed"] for r in results)
    total = sum(r["selected_tokens"] for r in results)
    return {"concurrency": n, "workload_sha256": digest(canonical(protocol)),
            "protocol": protocol, "passed": passed, "batch_wall_s": wall,
            "observed_selected_tokens": total,
            "generated_tokens_per_s": total / wall if passed else None, "requests": results}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--concurrency", nargs="+", type=int, default=[8, 16])
    parser.add_argument("--tokens", type=int, default=128)
    parser.add_argument("--warmup-repetitions", type=int, default=0,
                        help="complete unmeasured repetitions of all concurrency shapes; retained in report")
    parser.add_argument("--timeout", type=float, default=600)
    parser.add_argument("--prompt-file", help="JSON manifest with prompts list; extra alignment fields preserved")
    args = parser.parse_args(argv)
    if min(args.concurrency) < 1 or args.tokens < 1 or not math.isfinite(args.timeout) or args.timeout <= 0:
        parser.error("concurrency, tokens and timeout must be positive and finite")
    if args.warmup_repetitions < 0:
        parser.error("warmup-repetitions must be nonnegative")
    manifest = json.loads(Path(args.prompt_file).read_text()) if args.prompt_file else None
    prompts = manifest["prompts"] if manifest else [
        f"Continue this deterministic sequence with a detailed explanation: red blue green. Client {i}."
        for i in range(max(args.concurrency))]
    if not isinstance(prompts, list) or len(prompts) < max(args.concurrency) or not all(isinstance(p, str) for p in prompts):
        parser.error("manifest requires at least max(concurrency) string prompts")
    # Warm every requested shape before measuring any shape. Preserve warmup
    # results rather than silently hiding startup costs or failed requests.
    warmups = [[run_batch(args.base_url, args.model, n, args.tokens, args.timeout, prompts)
                for n in args.concurrency] for _ in range(args.warmup_repetitions)]
    report = {"schema_version": 1, "base_url": args.base_url, "manifest": manifest,
              "warmup_repetitions": args.warmup_repetitions, "warmups": warmups,
              "alignment": "Input token IDs/hash unavailable unless supplied in manifest; supplied alignment is unverified. No donor alignment or winner inferred.",
              "definitions": {"ttft": "client request start to first selected-token SSE arrival",
                              "itl": "adjacent selected-token arrival gaps, including zero gaps within a chunk; nearest-rank percentiles",
                              "batch_wall": "barrier release through completion of all HTTP requests",
                              "timeout": "urllib socket-operation timeout; not a total deadline"},
              "batches": [run_batch(args.base_url, args.model, n, args.tokens, args.timeout, prompts)
                          for n in args.concurrency]}
    report["passed"] = all(b["passed"] for b in report["batches"]) and all(
        b["passed"] for repetition in warmups for b in repetition)
    Path(args.out).write_text(json.dumps(report, indent=2) + "\n")
    for batch in report["batches"]:
        print(f"N={batch['concurrency']} passed={batch['passed']} generated_tokens/s={batch['generated_tokens_per_s']}")
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
