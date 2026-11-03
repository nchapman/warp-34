"""Validated, engine-agnostic experiment descriptions. Paths are config-relative."""

import copy
import hashlib
import json
import math
import re
from pathlib import Path
import tomllib
from urllib.parse import urlsplit


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                     separators=(",", ":")).encode()).hexdigest()


def positive(value, name, *, zero=False, integer=False):
    if (isinstance(value, bool) or not isinstance(value, (int, float))
            or not math.isfinite(value) or value < 0 or (not zero and value == 0)
            or (integer and not isinstance(value, int))):
        raise ValueError(f"{name} must be a {'nonnegative' if zero else 'positive'} "
                         f"{'integer' if integer else 'number'}")


def load(path):
    path = Path(path).resolve()
    with path.open("rb") as f:
        cfg = tomllib.load(f)
    if cfg.get("schema_version") != 1:
        raise ValueError("schema_version must be 1")
    defaults = cfg.get("engine_defaults", {})
    raw_engines, cases = cfg.get("engines", []), cfg.get("cases", [])
    if not isinstance(defaults, dict):
        raise ValueError("engine_defaults must be a table")
    for collection, name in ((raw_engines, "engines"), (cases, "cases")):
        if not isinstance(collection, list) or any(not isinstance(item, dict) for item in collection):
            raise ValueError(f"{name} must be an array of tables")
    engines = [{**defaults, **engine} for engine in raw_engines]
    cfg["engines"] = engines
    if not engines or not cases:
        raise ValueError("engines and cases must not be empty")
    for collection, label in ((engines, "engine"), (cases, "case")):
        ids = [item.get("id") for item in collection]
        if any(not isinstance(i, str) or not i or not all(c.isalnum() or c in "_-" for c in i)
               for i in ids) or len(set(ids)) != len(ids):
            raise ValueError(f"{label} ids must be unique nonempty alphanumeric/_/- strings")
    for engine in engines:
        if not isinstance(engine.get("engine"), str) or not engine["engine"].strip():
            raise ValueError(f"{engine['id']}: engine name required")
        if isinstance(engine.get("gpu_count"), bool) or engine.get("gpu_count") not in (1, 2):
            raise ValueError(f"{engine['id']}: gpu_count must be 1 or 2")
        engine.setdefault("enabled", True)
        engine.setdefault("support", "unverified")
        if not isinstance(engine["enabled"], bool):
            raise ValueError("enabled must be a boolean")
        if engine["support"] not in ("supported", "unverified", "unsupported"):
            raise ValueError("support must be supported, unverified, or unsupported")
        if not engine["enabled"] or engine["support"] == "unsupported":
            if not isinstance(engine.get("skip_reason"), str) or not engine["skip_reason"].strip():
                raise ValueError("disabled/unsupported deployments require a nonempty skip_reason")
        if "base_url" in engine:
            if not isinstance(engine["base_url"], str):
                raise ValueError("base_url must be a string")
            url = urlsplit(engine["base_url"])
            if (url.scheme not in ("http", "https") or not url.hostname or url.username
                    or url.password or url.query or url.fragment):
                raise ValueError(f"{engine['id']}: base_url needs http(s), no credentials/query/fragment")
            if url.port is not None and url.port == 0:
                raise ValueError("base_url port must be in 1..65535")
        if "api_key_env" in engine and (not isinstance(engine["api_key_env"], str)
                or not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", engine["api_key_env"])):
            raise ValueError("api_key_env must be an environment variable name")
        if not engine["enabled"] or engine["support"] == "unsupported":
            continue
        if "base_url" not in engine:
            raise ValueError("runnable deployments require a base_url string")
        if not isinstance(engine.get("model"), str) or not engine["model"]:
            raise ValueError(f"{engine['id']}: model required")
        for key in ("engine", "version", "model_revision", "weight_format", "kv_format",
                    "parallelism", "speculation", "cache_config", "launch_command"):
            if not isinstance(engine.get(key), str) or not engine[key]:
                raise ValueError(f"{engine['id']}: {key} must be recorded (use 'unknown' if needed)")
        if not isinstance(engine.get("request_overrides", {}), dict):
            raise ValueError("request_overrides must be a table")
        reserved = {"model", "messages", "stream", "stream_options", "max_tokens", "n"}
        if reserved.intersection(engine.get("request_overrides", {})):
            raise ValueError("request_overrides cannot replace workload/stream/model fields")
    run = cfg.setdefault("run", {})
    if not isinstance(run, dict):
        raise ValueError("run must be a table")
    for key, default, zero, integer in (("repetitions", 3, False, True),
                                      ("warmup_requests", 2, True, True),
                                      ("timeout_s", 180, False, False),
                                      ("max_inflight", 128, False, True),
                                      ("seed", 34, True, True)):
        run.setdefault(key, default)
        positive(run[key], key, zero=zero, integer=integer)
    for case in cases:
        if case.get("kind", "text") not in ("text", "agent_replay"):
            raise ValueError("kind must be text or agent_replay")
        for key, default in (("requests", 32), ("concurrency", 1), ("max_tokens", 128)):
            case.setdefault(key, default)
            positive(case[key], key, integer=True)
        if case.get("arrival", "closed") not in ("closed", "poisson", "fixed"):
            raise ValueError("arrival must be closed, poisson, or fixed")
        if case.get("arrival", "closed") != "closed":
            positive(case.get("rate_rps"), "rate_rps")
        if case.get("cache", "unique") not in ("unique", "primed", "natural"):
            raise ValueError("cache must be unique, primed, or natural")
        positive(case.get("think_time_s", 0), "think_time_s", zero=True)
        positive(case.get("context_repetitions", 1), "context_repetitions", integer=True)
        if "mix" in case:
            if case.get("kind", "text") != "text" or not isinstance(case["mix"], list) or not case["mix"]:
                raise ValueError("mix requires a nonempty list of text request shapes")
            for shape in case["mix"]:
                if not isinstance(shape, dict):
                    raise ValueError("mix shapes must be tables")
                if set(shape) - {"context_repetitions", "max_tokens"}:
                    raise ValueError("mix shapes accept context_repetitions and max_tokens only")
                for key, value in shape.items():
                    positive(value, key, integer=True)
        for key in ("ttft_slo_ms", "e2e_slo_ms", "tpot_slo_ms"):
            if key in case:
                positive(case[key], key)
        if "fixture" not in case:
            raise ValueError(f"{case['id']}: fixture required")
        fixture = path.parent / case["fixture"]
        data = json.loads(fixture.read_text())
        if not isinstance(data, dict):
            raise ValueError("fixture must be a JSON object")
        if case.get("kind", "text") == "agent_replay":
            if (not isinstance(data.get("system"), str) or not data.get("turns")
                    or not isinstance(data["turns"], list)):
                raise ValueError("agent fixture requires system and turns")
            for turn_idx, turn in enumerate(data["turns"]):
                user_text = turn.get("user")
                tools = turn.get("tools", [])
                if not isinstance(tools, list):
                    raise ValueError("agent turn tools must be a list")
                if turn_idx == 0:
                    if not isinstance(user_text, str) or not user_text:
                        raise ValueError("first agent turn needs nonempty user text")
                elif user_text is not None and not isinstance(user_text, str):
                    raise ValueError("agent turn user must be a string or null")
                elif user_text is None and not tools:
                    raise ValueError("tool-only follow-up agent turn needs at least one tool")
                for tool in tools:
                    if not all(isinstance(tool.get(k), str) for k in ("name", "result")):
                        raise ValueError("replayed tool needs name, result, and arguments")
                    if not isinstance(tool.get("arguments"), dict):
                        raise ValueError("replayed tool arguments must be an object")
        elif (not isinstance(data.get("prompts"), list) or not data["prompts"]
              or any(not isinstance(p, str) or not p for p in data["prompts"])):
            raise ValueError("text fixture requires nonempty prompts")
        case["fixture_data"] = data
        case["fixture_sha256"] = digest(data)
    presets = cfg.get("presets", {})
    if not isinstance(presets, dict):
        raise ValueError("presets must be a table")
    from .presets import apply_preset
    for name in presets:
        if name not in ("quick", "full", "release"):
            raise ValueError(f"unknown preset {name!r}")
        apply_preset(cfg, name)
    return cfg


def public_config(cfg):
    """Config contains environment variable *names*, never resolved credentials."""
    return copy.deepcopy(cfg)
