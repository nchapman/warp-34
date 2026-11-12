#!/usr/bin/env python3
"""Prepare and verify Hugging Face checkpoints and repo fixtures required by the warp-34 test suite.

Usage:
    python3 tools/prep_test_models.py            # ensure core 4B/drafter/tokenizer test fixtures are cached
    python3 tools/prep_test_models.py --all      # also ensure full 2B/9B/27B/35B-A3B/Flash-Next checkpoints are cached
    python3 tools/prep_test_models.py --check    # verify without downloading (exits 1 if any required item is missing)
"""

from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ModelSpec:
    repo_id: str
    revision: str
    required_files: tuple[str, ...]
    allow_patterns: tuple[str, ...] | None = None
    pin_ref_main: bool = True


CORE_TEST_MODELS: tuple[ModelSpec, ...] = (
    # Core 4B target checkpoint pinned to the unquantized snapshot (`851bf6e8...`)
    # required by `dflash_proposer::actual_sglang_full_model_hidden_parity` as well as
    # `mtp_proposer`, `tq4_vision_and_moe`, `axum_serving`, `decode_graphs`, etc.
    ModelSpec(
        repo_id="Qwen/Qwen3.5-4B",
        revision="851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a",
        required_files=("config.json", "model.safetensors.index.json", "tokenizer.json"),
    ),
    # Core 4B DFlash2 drafter checkpoint.
    ModelSpec(
        repo_id="nchapman/Qwen3.5-4B-DFlash2",
        revision="8d8bb62873a4fe6165e644163e9ff76e06ab78ba",
        required_files=("config.json", "model.safetensors"),
    ),
    # 27B DFlash2 config metadata checked by `dflash_proposer::cached_dflash_configs_and_candidate_merge`.
    ModelSpec(
        repo_id="incoai/Qwen3.8-27B-DFlash2",
        revision="015e795645c74b1a0eeef3b570031fb62e769bc5",
        required_files=("config.json",),
        allow_patterns=("config.json",),
    ),
)

FULL_SUITE_MODELS: tuple[ModelSpec, ...] = (
    ModelSpec(
        repo_id="nchapman/Qwen3.5-2B-W4A16-HGPTQ",
        revision="0fbb9a38f631ea63678f4b51a491e7ecb26c681c",
        required_files=("config.json", "model.safetensors", "tokenizer.json"),
    ),
    ModelSpec(
        repo_id="nchapman/Qwen3.5-4B-W4A16-HGPTQ",
        revision="493bf19cf3b9947d55320c93cb0f8d4c187a9083",
        required_files=("config.json", "model.safetensors", "tokenizer.json"),
    ),
    ModelSpec(
        repo_id="nchapman/Qwen3.5-9B-W4A16-HGPTQ",
        revision="7b2e48b7e7e8ee38925f5337c66feed22d3b031b",
        required_files=("config.json", "model.safetensors", "model_mtp.safetensors", "tokenizer.json"),
    ),
    ModelSpec(
        repo_id="nchapman/Qwen3.8-27B-W4A16-HGPTQ",
        revision="ae51987e3339a2a70070702548571e10fe8bcbc2",
        required_files=("config.json", "model.safetensors.index.json", "model_mtp.safetensors", "tokenizer.json"),
    ),
    ModelSpec(
        repo_id="nchapman/Qwen3.8-27B-DFlash2-W4A16-HGPTQ",
        revision="46b87ea125d24924affa8da9c3e981a937aab216",
        required_files=("config.json", "model.safetensors"),
    ),
    ModelSpec(
        repo_id="incoai/Qwen3.8-27B-DFlash2",
        revision="015e795645c74b1a0eeef3b570031fb62e769bc5",
        required_files=("config.json", "model.safetensors"),
    ),
    ModelSpec(
        repo_id="nchapman/Qwen3.6-35B-A3B-W4A16-HGPTQ",
        revision="4f5de21c886d79e17af5b8a4bfc625e226703fb6",
        required_files=("config.json", "model.safetensors.index.json", "tokenizer.json"),
    ),
    ModelSpec(
        repo_id="nchapman/Qwen3.8-Flash-Next-W4A16-HGPTQ",
        revision="6ee06d2936b111f19eb5eb331044cb8b8b28562b",
        required_files=("config.json", "model.safetensors.index.json", "tokenizer.json"),
    ),
    ModelSpec(
        repo_id="nchapman/Qwen3.8-Flash-Next-DFlash2-W4A16-HGPTQ",
        revision="24a62babba3b54387b3ea73f5599e5de83029dd8",
        required_files=("config.json", "model.safetensors"),
    ),
)

REPO_ORACLE_FIXTURES: tuple[str, ...] = (
    "tools/oracle/fixtures/coco_sample.png",
    "tools/oracle/fixtures/coco_sample.jpg",
    "tools/oracle/goldens/op_fixtures.safetensors",
    "tools/oracle/goldens/sglang_dflash2_fixtures.safetensors",
    "tools/oracle/goldens/sglang_dflash_attention.safetensors",
    "tools/oracle/goldens/sglang_dflash_attention_unsplit.safetensors",
    "tools/oracle/goldens/sglang_dflash_bf16.safetensors",
    "tools/oracle/goldens/sglang_dflash_model_4b.json",
    "tools/oracle/goldens/sglang_dflash_model_4b.safetensors",
    "tools/oracle/goldens/vllm_moe_layer0_m8.safetensors",
    "tools/oracle/goldens/vllm_mtp_4b/call_0.safetensors",
    "tools/oracle/goldens/vllm_mtp_4b/call_1.safetensors",
    "tools/oracle/goldens/vllm_mtp_4b/call_2.safetensors",
    "tools/oracle/goldens/vllm_mtp_4b/call_3.safetensors",
    "tools/oracle/goldens/4B/dflash2_oracle.ids",
)


def resolve_hub_cache_dir() -> Path:
    if env := os.environ.get("HF_HUB_CACHE"):
        return Path(env).expanduser()
    if env := os.environ.get("HF_HOME"):
        return Path(env).expanduser() / "hub"
    return Path.home() / ".cache" / "huggingface" / "hub"


def check_snapshot(hub_dir: Path, spec: ModelSpec) -> tuple[bool, Path]:
    repo_dir = hub_dir / f"models--{spec.repo_id.replace('/', '--')}"
    snap_dir = repo_dir / "snapshots" / spec.revision
    if not snap_dir.is_dir():
        return False, snap_dir
    for req in spec.required_files:
        p = snap_dir / req
        if not p.exists():
            return False, snap_dir
    return True, snap_dir


def pin_ref_main(hub_dir: Path, spec: ModelSpec) -> None:
    if not spec.pin_ref_main:
        return
    repo_dir = hub_dir / f"models--{spec.repo_id.replace('/', '--')}"
    refs_dir = repo_dir / "refs"
    refs_dir.mkdir(parents=True, exist_ok=True)
    ref_main = refs_dir / "main"
    current = ref_main.read_text().strip() if ref_main.exists() else ""
    if current != spec.revision:
        ref_main.write_text(spec.revision)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--all",
        action="store_true",
        help="Also verify/download full 2B/9B/27B/35B-A3B/Flash-Next checkpoints.",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Verify required fixtures without downloading; exit 1 if anything is missing.",
    )
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parent.parent
    missing_repo = [rel for rel in REPO_ORACLE_FIXTURES if not (repo_root / rel).exists()]
    if missing_repo:
        for rel in missing_repo:
            print(f"[MISSING REPO FIXTURE] {rel}", file=sys.stderr)
        return 1
    print(f"[OK] {len(REPO_ORACLE_FIXTURES)} repo-tracked oracle fixtures verified.")

    hub_dir = resolve_hub_cache_dir()
    specs = list(CORE_TEST_MODELS)
    if args.all:
        # Replace the config-only 27B DFlash2 entry when --all requests full weights.
        specs = [s for s in specs if s.repo_id != "incoai/Qwen3.8-27B-DFlash2"] + list(FULL_SUITE_MODELS)

    missing_specs: list[ModelSpec] = []
    for spec in specs:
        ok, snap_dir = check_snapshot(hub_dir, spec)
        if ok:
            if not args.check:
                pin_ref_main(hub_dir, spec)
            print(f"[OK] {spec.repo_id}@{spec.revision[:10]} -> {snap_dir}")
        else:
            missing_specs.append(spec)
            status = "MISSING" if args.check else "FETCH"
            print(f"[{status}] {spec.repo_id}@{spec.revision[:10]} ({snap_dir})")

    if not missing_specs:
        return 0
    if args.check:
        print(
            "\nRun `mise run test-prep` (or `python3 tools/prep_test_models.py [--all]`) to download missing checkpoints.",
            file=sys.stderr,
        )
        return 1

    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        print(
            f"huggingface_hub is required to download missing checkpoints ({exc}). "
            "Install via `pip install huggingface_hub`.",
            file=sys.stderr,
        )
        return 1

    for spec in missing_specs:
        print(f"Downloading {spec.repo_id}@{spec.revision} into {hub_dir}...")
        snapshot_download(
            repo_id=spec.repo_id,
            revision=spec.revision,
            cache_dir=str(hub_dir),
            allow_patterns=list(spec.allow_patterns) if spec.allow_patterns else None,
        )
        pin_ref_main(hub_dir, spec)
        ok, snap_dir = check_snapshot(hub_dir, spec)
        if not ok:
            print(
                f"Failed to verify {spec.repo_id}@{spec.revision} at {snap_dir} after download",
                file=sys.stderr,
            )
            return 1
        print(f"[OK] {spec.repo_id}@{spec.revision[:10]} -> {snap_dir}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
