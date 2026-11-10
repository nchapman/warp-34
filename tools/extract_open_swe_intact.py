#!/usr/bin/env python3
"""Deterministically extract intact single-task trajectories from nvidia/Open-SWE-Traces.

Each selected row is preserved as a single-repository chronological turn sequence
(no cross-repository concatenation, line slicing, or repeated padding). A short
replay note is appended to the upstream harness system prompt so models running
with `tool_choice: "none"` analyze the replayed observations rather than stalling
for missing live tool definitions.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import time
import urllib.error
import urllib.parse
import urllib.request

DATASET = "nvidia/Open-SWE-Traces"
PINNED_DATASET_SHA = "4fa4c2ae9afc6b397bc50391d43450c79993c138"
PERMISSIVE_LICENSES = {"MIT", "Apache-2.0", "BSD-2-Clause", "BSD-3-Clause", "ISC"}

REPLAY_SYSTEM_NOTE = (
    "\n\n[Replay note: Tool invocations and observations in this session are "
    "replayed deterministically; do not invoke new tools—analyze the repository "
    "observations concisely and state the next engineering step or minimal patch.]"
)

# Balanced 6-trajectory 16 Ki pool: 2 Claude Code, 2 OpenCode, 2 OpenHands;
# all resolved == 1 and permissive-licensed, 7–15 turns each.
DEFAULT_SPECS = (
    ("v2.0", "claudecode", 38, "h5netcdf_h5netcdf_pr218"),
    ("v2.0", "claudecode", 94, "oauthlib_oauthlib_pr336"),
    ("v2.0", "opencode", 7, "app-sre_qontract-reconcile_pr702"),
    ("v2.0", "opencode", 78, "more-itertools_more-itertools_pr742"),
    ("v2.0", "openhands", 2, "python-tap_tappy_pr69"),
    ("v2.0", "openhands", 84, "fonttools_fonttools_pr2353"),
)

SECRET_RES = (
    re.compile(r"AKIA[0-9A-Z]{16}"),
    re.compile(r"gh[pousr]_[A-Za-z0-9_]{36,}"),
    re.compile(r"sk-[A-Za-z0-9]{32,}"),
    re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"),
)


def fetch_json(url: str, *, retries: int = 5) -> dict:
    delay = 1.0
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "warp-34-fixture-extractor/1.0"})
            with urllib.request.urlopen(req, timeout=30) as resp:
                return json.load(resp)
        except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError):
            if attempt + 1 == retries:
                raise
            time.sleep(delay)
            delay *= 2.0
    raise RuntimeError("unreachable")


def verify_dataset_revision(expected_sha: str) -> str:
    meta = fetch_json(f"https://huggingface.co/api/datasets/{DATASET}")
    actual = meta.get("sha")
    if actual != expected_sha:
        raise ValueError(
            f"{DATASET} HEAD moved ({actual} != {expected_sha}); "
            "verify row offsets before updating PINNED_DATASET_SHA"
        )
    return actual


def parse_intact_turns(messages: list[dict]) -> tuple[str, list[dict]]:
    if not messages or messages[0].get("role") != "system":
        raise ValueError("trajectory must begin with a system message")
    system_text = (messages[0].get("content") or "") + REPLAY_SYSTEM_NOTE
    turns: list[dict] = []
    cur_user: str | None = None
    cur_tools: list[dict] = []
    i = 1
    while i < len(messages):
        msg = messages[i]
        role = msg.get("role")
        if role == "user":
            if cur_user is not None or cur_tools:
                turns.append({"user": cur_user, "tools": cur_tools})
                cur_user, cur_tools = None, []
            cur_user = msg.get("content") or ""
            i += 1
        elif role == "assistant":
            tc_list = msg.get("tool_calls") or []
            if cur_user is not None or cur_tools:
                turns.append({"user": cur_user, "tools": cur_tools})
                cur_user, cur_tools = None, []
            i += 1
            if tc_list:
                by_id: dict[str, str] = {}
                ordered: list[str] = []
                while i < len(messages) and messages[i].get("role") == "tool":
                    tmsg = messages[i]
                    content = tmsg.get("content") or ""
                    ordered.append(content)
                    if tmsg.get("tool_call_id"):
                        by_id[tmsg["tool_call_id"]] = content
                    i += 1
                batch: list[dict] = []
                for ti, tc in enumerate(tc_list):
                    fn = tc.get("function") or {}
                    name = fn.get("name") or "tool"
                    raw_args = fn.get("arguments")
                    if isinstance(raw_args, str):
                        try:
                            args = json.loads(raw_args)
                            if not isinstance(args, dict):
                                args = {"value": args}
                        except json.JSONDecodeError:
                            args = {"raw": raw_args}
                    elif isinstance(raw_args, dict):
                        args = raw_args
                    else:
                        args = {}
                    result = by_id.get(tc.get("id"), ordered[ti] if ti < len(ordered) else "")
                    batch.append({"name": name, "arguments": args, "result": result})
                cur_tools = batch
        else:
            raise ValueError(f"unexpected message role {role!r} at index {i}")
    if cur_user is not None or cur_tools:
        turns.append({"user": cur_user, "tools": cur_tools})
    if not turns or not isinstance(turns[0].get("user"), str) or not turns[0]["user"]:
        raise ValueError("intact trajectory requires a nonempty initial user prompt")
    return system_text, turns


def assert_no_secrets(payload_text: str) -> None:
    for rx in SECRET_RES:
        match = rx.search(payload_text)
        if match:
            raise ValueError(f"secret-like token matched pattern {rx.pattern!r}")


def extract_fixture() -> dict:
    dataset_sha = verify_dataset_revision(PINNED_DATASET_SHA)
    trajectories = []
    for config, split, offset, expected_instance in DEFAULT_SPECS:
        qs = urllib.parse.urlencode(
            {"dataset": DATASET, "config": config, "split": split, "offset": offset, "length": 1}
        )
        rows = fetch_json(f"https://datasets-server.huggingface.co/rows?{qs}").get("rows") or []
        if len(rows) != 1 or rows[0].get("row_idx") != offset:
            raise ValueError(f"missing row for {config}/{split}/{offset}")
        row = rows[0]["row"]
        instance_id = row.get("instance_id")
        if instance_id != expected_instance:
            raise ValueError(
                f"row drift at {config}/{split}/{offset}: expected {expected_instance!r}, got {instance_id!r}"
            )
        lic = row.get("license")
        if lic not in PERMISSIVE_LICENSES:
            raise ValueError(f"non-permissive license {lic!r} on {instance_id}")
        if row.get("resolved") != 1:
            raise ValueError(f"expected resolved==1 on {instance_id}")
        system_text, turns = parse_intact_turns(row["messages"])
        trajectories.append(
            {
                "id": f"{split}:{instance_id}",
                "instance_id": instance_id,
                "harness": split,
                "dataset_config": config,
                "dataset_offset": offset,
                "trajectory_uuid": row.get("trajectory_id"),
                "repo": row.get("repo"),
                "license": lic,
                "language": row.get("language"),
                "resolved": row.get("resolved"),
                "system": system_text,
                "turns": turns,
            }
        )
    fixture = {
        "source": f"Intact single-instance trajectories from {DATASET}@{dataset_sha} (CC-BY-4.0)",
        "dataset": DATASET,
        "dataset_revision": dataset_sha,
        "trajectories": trajectories,
    }
    serialized = json.dumps(fixture, indent=2, ensure_ascii=False) + "\n"
    assert_no_secrets(serialized)
    return fixture


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("benchmarks/perf/fixtures/open_swe_intact_16k.json"),
        help="Output path for frozen intact Open-SWE fixture JSON",
    )
    args = parser.parse_args()
    fixture = extract_fixture()
    raw = (json.dumps(fixture, indent=2, ensure_ascii=False) + "\n").encode("utf-8")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_bytes(raw)
    sha = hashlib.sha256(raw).hexdigest()
    print(f"wrote {args.out} ({len(raw)} bytes, sha256={sha})")
    for traj in fixture["trajectories"]:
        print(
            f"  {traj['id']:48s} repo={traj['repo']:34s} "
            f"license={traj['license']:12s} turns={len(traj['turns']):2d}"
        )


if __name__ == "__main__":
    main()
