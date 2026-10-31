"""Explicit local-byte and advertised-model checks, not server attestation."""

from datetime import datetime, timezone
import json
from pathlib import Path

from .artifacts import verify_inventory
from .config import digest
from .selection import availability
from .transport import models


def indexed_weights(root, inventory):
    """Require indexed shards to be present and covered by the trusted inventory."""
    root = Path(root).expanduser().absolute()
    files = inventory["files"]
    weights = sorted(name for name in files if name.endswith((".safetensors", ".gguf", ".ninfer")))
    if not weights:
        raise ValueError("inventory must include model weights")
    if root.is_file():
        return weights
    indices = sorted(root.glob("*.safetensors.index.json"))
    for index in indices:
        if index.name not in files:
            raise ValueError(f"inventory does not cover shard index: {index.name}")
        data = json.loads(index.read_text())
        mapping = data.get("weight_map") if isinstance(data, dict) else None
        if not isinstance(mapping, dict) or not mapping:
            raise ValueError(f"invalid weight_map in {index.name}")
        for name in mapping.values():
            if not isinstance(name, str) or not name:
                raise ValueError("indexed shard names must be nonempty relative paths")
            path = Path(name)
            if path.is_absolute() or ".." in path.parts:
                raise ValueError("indexed shard names must be relative without parent traversal")
            if not (root / path).is_file():
                raise ValueError(f"missing indexed shard: {name}")
            if path.as_posix() not in files:
                raise ValueError(f"inventory does not cover indexed shard: {name}")
    return weights


def preflight(cfg, engine_id, inventory_path, output, root=None):
    matches = [e for e in cfg["engines"] if e["id"] == engine_id]
    if not matches:
        raise ValueError("unknown engine selection")
    engine = matches[0]
    # Reserve evidence before doing any I/O to an endpoint. Repeated preflights
    # need a new filename; do not overwrite a failure with a later success.
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x") as dest:
        report = {"schema_version": 1, "created_utc": datetime.now(timezone.utc).isoformat(),
                  "engine_id": engine_id, "config_sha256": digest(cfg),
                  "status": "failed", "checks": {},
                  "limitations": "Local bytes and advertised IDs only; does not attest what the server loaded or validate quality."}
        state, reason = availability(engine)
        if state != "ready":
            report.update(status="skipped", reason=reason)
        else:
            try:
                inventory = json.loads(Path(inventory_path).read_text())
                verified = verify_inventory(inventory_path, root)
                report["checks"]["inventory"] = verified
                if verified["status"] != "verified":
                    raise ValueError("local artifact bytes differ from trusted inventory")
                artifact_root = root or inventory["root"]
                report["checks"]["indexed_weights"] = indexed_weights(artifact_root, inventory)
                expected_hash = engine.get("artifact_sha256")
                if expected_hash not in (None, "unknown", verified["inventory_sha256"]):
                    raise ValueError("profile artifact_sha256 differs from verified inventory")
                advertised = models(engine, cfg["run"]["timeout_s"])
                report["checks"]["models"] = advertised
                if advertised["status"] != "ok":
                    raise ValueError("model discovery failed; inspect checks.models")
                data = advertised["data"]
                if (not isinstance(data, dict) or not isinstance(data.get("data"), list)
                        or any(not isinstance(m, dict) or not isinstance(m.get("id"), str) for m in data["data"])):
                    raise ValueError("/models must return a data array of model ID objects")
                ids = [m["id"] for m in data["data"]]
                report["checks"]["advertised_model_ids"] = ids
                if engine["model"] not in ids:
                    raise ValueError("configured model ID is not advertised by endpoint")
                report["status"] = "passed"
            except KeyboardInterrupt:
                report.update(status="interrupted", error="preflight interrupted")
                dest.write(json.dumps(report, indent=2) + "\n")
                dest.flush()
                raise
            except (ValueError, OSError, KeyError) as exc:
                report["error"] = f"{type(exc).__name__}: {exc}"
        dest.write(json.dumps(report, indent=2) + "\n")
    return report
