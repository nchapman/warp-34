"""Streaming, offline weight/tokenizer inventories; never downloads model data."""

from datetime import datetime, timezone
import fnmatch
import hashlib
import json
from pathlib import Path

from .config import digest


def files_at(root, patterns, *, allow_empty=False):
    if not root.exists():
        raise ValueError(f"artifact path does not exist: {root}")
    candidates = [root] if root.is_file() else sorted(root.rglob("*"))
    files = {}
    for path in candidates:
        name = path.name if root.is_file() else path.relative_to(root).as_posix()
        if any(part.startswith(".") for part in Path(name).parts):
            continue
        if path.is_file() and (not patterns or any(fnmatch.fnmatchcase(name, p) for p in patterns)):
            files[name] = path
    if not files and not allow_empty:
        raise ValueError("artifact inventory matched no files")
    return files


def hash_file(path):
    # HF snapshots normally symlink to the shared blob store; hash the contents,
    # not the symlink text, while keeping logical snapshot-relative names.
    before = path.stat()
    sha = hashlib.sha256()
    with path.open("rb") as source:
        while block := source.read(8 * 1024 * 1024):
            sha.update(block)
    after = path.stat()
    if ((before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
            != (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns)):
        raise ValueError(f"artifact changed while hashing: {path.name}")
    return {"bytes": after.st_size, "sha256": sha.hexdigest()}


def inventory(root, patterns=None, *, allow_empty=False):
    root = Path(root).expanduser().absolute()
    patterns = patterns or []
    entries = {name: hash_file(path) for name, path in files_at(root, patterns, allow_empty=allow_empty).items()}
    return {"schema_version": 1, "created_utc": datetime.now(timezone.utc).isoformat(),
            "root": str(root), "include": patterns, "files": entries,
            "total_bytes": sum(e["bytes"] for e in entries.values()),
            "inventory_sha256": digest(entries)}


def save_inventory(root, output, patterns=None):
    root, output = Path(root).expanduser().absolute(), Path(output).expanduser().absolute()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite inventory: {output}")
    if output.resolve() == root.resolve() or (root.is_dir() and output.resolve().is_relative_to(root.resolve())):
        raise ValueError("write the inventory outside the read-only artifact directory")
    result = inventory(root, patterns)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x") as dest:
        dest.write(json.dumps(result, indent=2) + "\n")
    return result


def verify_inventory(path, root=None):
    expected = json.loads(Path(path).read_text())
    if (not isinstance(expected, dict) or expected.get("schema_version") != 1
            or not isinstance(expected.get("files"), dict) or not expected["files"]):
        raise ValueError("invalid artifact inventory")
    if expected.get("inventory_sha256") != digest(expected["files"]):
        raise ValueError("artifact inventory content hash is invalid")
    # Inventories and their root/include settings are trusted operator inputs.
    # Discover names locally rather than opening relative paths from the file map.
    actual = inventory(root or expected["root"], expected.get("include", []), allow_empty=True)
    differences = []
    for name in sorted(expected["files"].keys() | actual["files"].keys()):
        if expected["files"].get(name) != actual["files"].get(name):
            differences.append(name)
    return {"status": "verified" if not differences else "mismatch",
            "inventory_sha256": actual["inventory_sha256"],
            "total_bytes": actual["total_bytes"], "differences": differences}
