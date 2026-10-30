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


def stream(engine, payload, timeout_s):
    start = time.perf_counter()
    result = {"status": "error", "error": None, "http_status": None, "usage": None,
              "finish_reason": None, "events": [], "content": "", "reasoning": "",
              "tool_calls": {}, "first_output_ms": None, "first_text_ms": None,
              "last_output_ms": None, "e2e_ms": None}
    url = urlsplit(engine["base_url"])
    cls = http.client.HTTPSConnection if url.scheme == "https" else http.client.HTTPConnection
    connection = cls(url.hostname, url.port, timeout=timeout_s)
    deadline = start + timeout_s

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
        headers = {"Content-Type": "application/json", "Accept": "text/event-stream"}
        if engine.get("api_key_env"):
            key = os.environ.get(engine["api_key_env"])
            if not key:
                raise ValueError(f"missing credential environment variable {engine['api_key_env']}")
            if any(ord(c) < 33 or ord(c) > 126 for c in key):
                raise ValueError("invalid credential format")
            headers["Authorization"] = "Bearer " + key
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
        if expired.is_set():
            result["status"] = "timeout"
            result["error"] = "request timeout"
        connection.close()
    return result
