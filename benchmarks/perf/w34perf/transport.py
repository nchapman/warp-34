"""OpenAI chat/SSE boundary. No engine imports, SDK, proxies, or implicit retries."""

import http.client
import json
import os
import socket
import time
import threading
from urllib.parse import urlsplit

MAX_LINE = 4 * 1024 * 1024
MAX_RESPONSE = 64 * 1024 * 1024


def authorization(engine):
    if not engine.get("api_key_env"):
        return {}
    key = os.environ.get(engine["api_key_env"])
    if not key:
        raise ValueError(f"missing credential environment variable {engine['api_key_env']}")
    if any(ord(c) < 33 or ord(c) > 126 for c in key):
        raise ValueError("invalid credential format")
    return {"Authorization": "Bearer " + key}


def models(engine, timeout_s):
    """Bounded, single-attempt GET /models, preserving successful JSON evidence."""
    start = time.perf_counter()
    result = {"status": "error", "error": None, "http_status": None, "data": None}
    url = urlsplit(engine["base_url"])
    cls = http.client.HTTPSConnection if url.scheme == "https" else http.client.HTTPConnection
    connection = cls(url.hostname, url.port, timeout=timeout_s)
    timer = None
    expired = threading.Event()

    def abort(sock):
        expired.set()
        try:
            sock.shutdown(socket.SHUT_RDWR)
        except OSError:
            pass

    try:
        headers = {"Accept": "application/json", **authorization(engine)}
        connection.connect()
        remaining = timeout_s - (time.perf_counter() - start)
        if remaining <= 0:
            raise TimeoutError("model discovery deadline exceeded")
        sock = connection.sock
        sock.settimeout(remaining)
        timer = threading.Timer(remaining, abort, args=(sock,))
        timer.daemon = True
        timer.start()
        connection.request("GET", url.path.rstrip("/") + "/models", headers=headers)
        response = connection.getresponse()
        result["http_status"] = response.status
        if response.status != 200:
            raise ValueError(f"HTTP {response.status}")
        limit = 1024 * 1024
        raw = response.read(limit + 1)
        if len(raw) > limit:
            raise ValueError("model discovery response exceeded byte limit")
        try:
            data = json.loads(raw)
            json.dumps(data, ensure_ascii=False, allow_nan=False).encode("utf-8")
        except (ValueError, UnicodeError, RecursionError):
            raise ValueError("invalid model discovery JSON") from None
        result.update(status="ok", data=data)
    except (OSError, ValueError, TypeError, http.client.HTTPException) as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"[:500]
    finally:
        if timer:
            timer.cancel()
        if expired.is_set():
            result.update(status="error", error="model discovery timeout", data=None)
        result["e2e_ms"] = (time.perf_counter() - start) * 1000
        connection.close()
    return result


def stream(engine, payload, timeout_s, *, cancel_after_output_events=None):
    """Stream a completion, optionally injecting a bounded local disconnect.

    Cancellation fields describe only the requested/observed client-side close;
    they do not confirm server cancellation or release of server resources.
    """
    if cancel_after_output_events is not None and (
        isinstance(cancel_after_output_events, bool)
        or not isinstance(cancel_after_output_events, int)
        or cancel_after_output_events <= 0
    ):
        raise ValueError("cancel_after_output_events must be None or a positive integer")
    start = time.perf_counter()
    result = {"status": "error", "error": None, "http_status": None, "usage": None,
              "finish_reason": None, "events": [], "content": "", "reasoning": "",
              "tool_calls": {}, "first_output_ms": None, "first_text_ms": None,
              "last_output_ms": None, "e2e_ms": None}
    url = urlsplit(engine["base_url"])
    cls = http.client.HTTPSConnection if url.scheme == "https" else http.client.HTTPConnection
    connection = cls(url.hostname, url.port, timeout=timeout_s)
    deadline = start + timeout_s
    response = None
    sock = None
    output_events = 0
    if cancel_after_output_events is not None:
        result.update(cancellation_requested=True, cancellation_observed=False)

    def remaining():
        value = deadline - time.perf_counter()
        if value <= 0:
            raise TimeoutError("total request deadline exceeded")
        return value

    def elapsed():
        return (time.perf_counter() - start) * 1000

    timer = None
    expired = threading.Event()

    def abort(sock):
        expired.set()
        try:
            sock.shutdown(socket.SHUT_RDWR)
        except OSError:
            pass

    try:
        headers = {"Content-Type": "application/json", "Accept": "text/event-stream", **authorization(engine)}
        # Establish separately so the socket remains available even on Connection: close.
        connection.connect()
        sock = connection.sock
        timer = threading.Timer(remaining(), abort, args=(sock,))
        timer.daemon = True
        timer.start()
        sock.settimeout(remaining())
        connection.request("POST", url.path.rstrip("/") + "/chat/completions",
                           body=json.dumps(payload).encode(), headers=headers)
        sock.settimeout(remaining())
        response = connection.getresponse()
        result["http_status"] = response.status
        if response.status != 200:
            # Do not persist arbitrary upstream error bodies (may contain credentials).
            raise ValueError(f"HTTP {response.status}")
        if "text/event-stream" not in response.getheader("Content-Type", "").lower():
            raise ValueError("expected text/event-stream")
        total = 0
        data = []
        done = False
        while True:
            sock.settimeout(remaining())
            line = response.readline(MAX_LINE + 1)
            total += len(line)
            if len(line) > MAX_LINE or total > MAX_RESPONSE:
                raise ValueError("SSE response exceeded byte limit")
            if not line:
                break
            line = line.decode("utf-8").rstrip("\r\n")
            if line:
                if line.startswith("data:"):
                    data.append(line[5:].lstrip(" "))
                continue
            if not data:
                continue
            raw = "\n".join(data)
            data = []
            if raw == "[DONE]":
                done = True
                break
            event = json.loads(raw)
            # Python accepts nonstandard NaN/Infinity and escaped lone surrogates.
            # Reject those BEFORE retaining any event so one malformed response
            # cannot poison the append-only JSON evidence for the whole campaign.
            try:
                json.dumps(event, ensure_ascii=False, allow_nan=False).encode("utf-8")
            except (ValueError, UnicodeError):
                raise ValueError("invalid non-finite or Unicode SSE JSON") from None
            if not isinstance(event, dict):
                raise ValueError("SSE event must be an object")
            if "error" in event:
                raise ValueError("server emitted SSE error")
            stamp = elapsed()
            result["events"].append({"at_ms": stamp, "data": event})
            if event.get("usage") is not None:
                result["usage"] = event["usage"]
            choices = event.get("choices", [])
            if len(choices) > 1:
                raise ValueError("expected one completion choice")
            if not choices:
                continue
            choice = choices[0]
            if choice.get("index", 0) != 0:
                raise ValueError("unexpected completion choice index")
            delta = choice.get("delta", {})
            text = delta.get("content") or ""
            reasoning = delta.get("reasoning_content") or delta.get("reasoning") or ""
            calls = delta.get("tool_calls") or []
            if text or reasoning or calls:
                output_events += 1
                if result["first_output_ms"] is None:
                    result["first_output_ms"] = stamp
                result["last_output_ms"] = stamp
            if text and result["first_text_ms"] is None:
                result["first_text_ms"] = stamp
            result["content"] += text
            result["reasoning"] += reasoning
            for call in calls:
                index = str(call.get("index", 0))
                entry = result["tool_calls"].setdefault(index, {"id": "", "type": "function",
                                                               "function": {"name": "", "arguments": ""}})
                if call.get("id"):
                    entry["id"] = call["id"]
                for key in ("name", "arguments"):
                    entry["function"][key] += call.get("function", {}).get(key) or ""
            if choice.get("finish_reason") is not None:
                result["finish_reason"] = choice["finish_reason"]
            if (cancel_after_output_events is not None
                    and output_events >= cancel_after_output_events
                    and result["finish_reason"] is None):
                # Check the deadline before injecting, including time spent parsing.
                remaining()
                if expired.is_set():
                    raise TimeoutError("total request deadline exceeded")
                # HTTPConnection may no longer own this socket (Connection: close).
                # Shutdown first so closing a buffered response cannot leave it alive.
                try:
                    sock.shutdown(socket.SHUT_RDWR)
                except OSError:
                    # The peer may have closed/reset after buffering output.
                    # We still close our local owners; no server ack is implied.
                    pass
                response.close()
                sock.close()
                result.update(status="cancelled", error=None, cancellation_observed=True)
                return result
        if not done:
            raise ValueError("truncated SSE: missing [DONE]")
        if result["finish_reason"] is None:
            raise ValueError("missing finish_reason")
        if result["first_output_ms"] is None:
            raise ValueError("completion produced no observable output")
        result["status"] = "ok"
    except (TimeoutError, socket.timeout):
        result["error"] = "request timeout"
        result["status"] = "timeout"
    except (OSError, ValueError, TypeError, KeyError, AttributeError, http.client.HTTPException) as exc:
        # Never include payloads, headers, or remote JSON in errors.
        result["error"] = f"{type(exc).__name__}: {exc}"[:500]
    finally:
        result["e2e_ms"] = elapsed()
        if timer:
            timer.cancel()
        if timer:
            timer.join()
        if expired.is_set() or time.perf_counter() >= deadline:
            result["status"] = "timeout"
            result["error"] = "request timeout"
        # Close both owners explicitly, including failures and Connection: close.
        if response is not None:
            response.close()
        connection.close()
        if sock is not None:
            sock.close()
    return result


def _score_decision_answers(questions, answers, expected):
    if not isinstance(expected, dict) or not expected:
        return None, None
    correct = 0
    total = 0
    for qk, exp_val in expected.items():
        if qk not in questions or qk not in answers:
            continue
        ans = answers[qk]
        if not isinstance(ans, dict):
            continue
        qtype = ans.get("type")
        total += 1
        if qtype == "noul":
            prob = ans.get("noul")
            if isinstance(prob, (int, float)) and not isinstance(prob, bool):
                pred = "yes" if prob >= 0.5 else "no"
                if (exp_val in ("yes", True) and pred == "yes") or (exp_val in ("no", False) and pred == "no"):
                    correct += 1
        elif qtype == "choice":
            if ans.get("choice") == exp_val:
                correct += 1
        elif qtype == "score":
            probs = ans.get("probabilities")
            score_val = ans.get("score")
            if isinstance(probs, dict) and probs and all(
                isinstance(v, (int, float)) and not isinstance(v, bool) for v in probs.values()
            ):
                best_k = max(probs, key=lambda k: probs[k])
                if str(best_k) == str(exp_val):
                    correct += 1
            elif isinstance(score_val, (int, float)) and not isinstance(score_val, bool):
                if str(int(round(score_val))) == str(exp_val):
                    correct += 1
    return correct, total


def decision(engine, payload, timeout_s, *, expected=None):
    """Execute a bounded POST /systemone typed decision request."""
    start = time.perf_counter()
    result = {"status": "error", "error": None, "http_status": None, "usage": None,
              "finish_reason": "stop", "events": [], "content": "", "reasoning": "",
              "tool_calls": {}, "answers": None, "decision_correct": None,
              "decision_total": None, "first_output_ms": None, "first_text_ms": None,
              "last_output_ms": None, "e2e_ms": None}
    url = urlsplit(engine["base_url"])
    cls = http.client.HTTPSConnection if url.scheme == "https" else http.client.HTTPConnection
    connection = cls(url.hostname, url.port, timeout=timeout_s)
    deadline = start + timeout_s
    response = None
    sock = None
    timer = None
    expired = threading.Event()

    def remaining():
        value = deadline - time.perf_counter()
        if value <= 0:
            raise TimeoutError("total request deadline exceeded")
        return value

    def elapsed():
        return (time.perf_counter() - start) * 1000

    def abort(s):
        expired.set()
        try:
            s.shutdown(socket.SHUT_RDWR)
        except OSError:
            pass

    try:
        headers = {"Content-Type": "application/json", "Accept": "application/json", **authorization(engine)}
        connection.connect()
        sock = connection.sock
        timer = threading.Timer(remaining(), abort, args=(sock,))
        timer.daemon = True
        timer.start()
        sock.settimeout(remaining())
        connection.request("POST", url.path.rstrip("/") + "/systemone",
                           body=json.dumps(payload).encode("utf-8"), headers=headers)
        sock.settimeout(remaining())
        response = connection.getresponse()
        result["http_status"] = response.status
        if response.status != 200:
            raise ValueError(f"HTTP {response.status}")
        raw = response.read(MAX_RESPONSE + 1)
        if len(raw) > MAX_RESPONSE:
            raise ValueError("decision response exceeded byte limit")
        try:
            data = json.loads(raw)
            json.dumps(data, ensure_ascii=False, allow_nan=False).encode("utf-8")
        except (ValueError, UnicodeError, RecursionError):
            raise ValueError("invalid non-finite or Unicode decision JSON") from None
        if not isinstance(data, dict):
            raise ValueError("decision response must be an object")
        answers = data.get("answers")
        questions = payload.get("questions", {})
        if not isinstance(answers, dict) or set(answers) != set(questions):
            raise ValueError("decision response answers keys do not match questions")
        for qk, q in questions.items():
            ans = answers.get(qk)
            if not isinstance(ans, dict) or ans.get("type") != q.get("type"):
                raise ValueError(f"decision answer {qk!r} missing or wrong type")
        stamp = elapsed()
        raw_usage = data.get("usage")
        if isinstance(raw_usage, dict):
            in_tok = raw_usage.get("input_tokens", raw_usage.get("prompt_tokens"))
            cached_tok = raw_usage.get("cached_tokens", 0)
            if (isinstance(in_tok, int) and not isinstance(in_tok, bool) and in_tok > 0
                    and isinstance(cached_tok, int) and not isinstance(cached_tok, bool)
                    and 0 <= cached_tok <= in_tok):
                result["usage"] = {
                    "prompt_tokens": in_tok,
                    "completion_tokens": len(questions),
                    "prompt_tokens_details": {"cached_tokens": cached_tok},
                    "decision_questions": len(questions),
                }
        correct, total = _score_decision_answers(questions, answers, expected)
        result.update(status="ok", answers=answers, decision_correct=correct,
                      decision_total=total, content=json.dumps(answers, sort_keys=True),
                      first_output_ms=stamp, first_text_ms=stamp, last_output_ms=stamp)
    except (TimeoutError, socket.timeout):
        result["error"] = "request timeout"
        result["status"] = "timeout"
    except (OSError, ValueError, TypeError, KeyError, AttributeError, http.client.HTTPException) as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"[:500]
    finally:
        result["e2e_ms"] = elapsed()
        if timer:
            timer.cancel()
        if timer:
            timer.join()
        if expired.is_set() or time.perf_counter() >= deadline:
            result["status"] = "timeout"
            result["error"] = "request timeout"
        if response is not None:
            response.close()
        connection.close()
        if sock is not None:
            sock.close()
    return result

