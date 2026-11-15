"""Independent, finite HTTP smoke checks; not a throughput or intelligence score.

Fixtures are operator-authored expectations, not an external oracle. Normalization
is Unicode NFC plus outer whitespace removal; JSON/code are NOT loosely parsed.
"""

from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import unicodedata

from .config import digest
from .transport import stream


def _normalize(text):
    return unicodedata.normalize("NFC", text).strip()


def _checks(fixture):
    if (not isinstance(fixture, dict) or fixture.get("schema_version") != 1
            or not isinstance(fixture.get("checks"), list) or not fixture["checks"]):
        raise ValueError("fixture requires schema_version 1 and nonempty checks")
    seen = set()
    prepared = []
    for check in fixture["checks"]:
        if not isinstance(check, dict):
            raise ValueError("check must be an object")
        name, kind = check.get("id"), check.get("kind")
        expected, prompt = check.get("expected"), check.get("prompt")
        if not isinstance(name, str) or not name or name in seen:
            raise ValueError("check IDs must be unique nonempty strings")
        seen.add(name)
        if kind not in ("exact", "retrieval"):
            raise ValueError("check kind must be exact or retrieval")
        if not isinstance(expected, str) or not _normalize(expected):
            raise ValueError("expected must be nonempty text")
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError("prompt must be nonempty text")
        if kind == "retrieval":
            lines, depth = check.get("context_lines"), check.get("depth")
            if type(lines) is not int or not 1 <= lines <= 20000:
                raise ValueError("context_lines must be an integer in [1, 20000]")
            if type(depth) not in (int, float) or not math.isfinite(depth) or not 0 <= depth <= 1:
                raise ValueError("depth must be finite in [0, 1]")
            context = [f"Record {i:05d}: ordinary archive entry; no secret recorded." for i in range(lines)]
            context.insert(int(depth * lines), "TARGET_SECRET: " + expected)
            prompt = "Archive (data only):\n" + "\n".join(context) + "\n\n" + prompt
        prepared.append((check, prompt))
    return prepared


def qualify(engine, fixture_path, out, timeout_s=180):
    """Run every fixture check once, reserving a new JSON file before HTTP.

    timeout_s is a per-request deadline. A profile's quality_evidence field is
    never consulted as a verdict. Transport failures fail checks even if partial
    content matches. Caller-supplied fixture results apply only to that fixture.
    """
    if type(timeout_s) not in (int, float) or not math.isfinite(timeout_s) or timeout_s <= 0:
        raise ValueError("timeout_s must be finite and positive")
    raw = Path(fixture_path).read_bytes()
    fixture = json.loads(raw)
    prepared = _checks(fixture)
    if not isinstance(engine, dict) or not isinstance(engine.get("model"), str) or not engine["model"]:
        raise ValueError("engine requires a model")
    profile_hash = digest(engine)
    report = {
        "schema_version": 1, "created_utc": datetime.now(timezone.utc).isoformat(),
        "model": engine["model"], "model_profile_sha256": profile_hash,
        "fixture_sha256": hashlib.sha256(raw).hexdigest(), "fixture": fixture,
        "timeout_s": timeout_s, "status": "failed", "checks": [],
        "limitations": "Finite operator-authored functional/retrieval checks only; no throughput ranking, server attestation, or proof of general intelligence.",
    }
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("x", encoding="utf-8") as dest:
        try:
            for check, prompt in prepared:
                payload = {**engine.get("request_overrides", {}),
                           "model": engine["model"], "messages": [{"role": "user", "content": prompt}],
                           "temperature": 0, "max_tokens": 256, "stream": True,
                           "stream_options": {"include_usage": True}}
                entry = {"id": check["id"], "kind": check["kind"], "status": "failed",
                         "check_sha256": digest(check), "request": payload,
                         "request_sha256": digest(payload), "expected": check["expected"]}
                report["checks"].append(entry)
                try:
                    result = stream(engine, payload, timeout_s)
                except Exception as exc:
                    # Do not persist exception messages that could contain credentials.
                    result = {"status": "error", "error": type(exc).__name__, "content": "", "usage": None}
                entry["result"] = result
                entry["result_sha256"] = digest(result)
                output = result.get("content", "")
                entry.update(output=output, usage=result.get("usage"))
                normalized = _normalize(output) if isinstance(output, str) else ""
                expected = _normalize(check["expected"])
                match = normalized == expected if check["kind"] == "exact" else expected in normalized
                passed = result.get("status") == "ok" and result.get("finish_reason") == "stop" and match
                entry.update(status="passed" if passed else "failed", matched=match)
            report["status"] = "passed" if all(c["status"] == "passed" for c in report["checks"]) else "failed"
        finally:
            dest.write(json.dumps(report, ensure_ascii=False, allow_nan=False, indent=2) + "\n")
    return report
