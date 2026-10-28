#!/usr/bin/env python3
"""Bundle actual pinned SGLang DFlash2 operator captures for Phase 4 gates.

This does not execute SGLang or synthesize reference outputs. Generate the
source captures with sglang_dflash_bf16_dump.py,
sglang_dflash_attention_dump.py, and sglang_dflash_unsplit_dump.py in
lmsysorg/sglang:v0.5.19 first. Missing captures,
missing provenance, mismatched donors, and duplicate tensor names fail closed.
The obsolete Phase 0 block-end-mask reference is deliberately not included.
"""
import argparse
import hashlib
import json
from pathlib import Path

from safetensors import safe_open
from safetensors.torch import save_file


SOURCES = ("sglang_dflash_bf16.safetensors", "sglang_dflash_attention.safetensors",
           "sglang_dflash_attention_unsplit.safetensors")


def bundle(source_dir: Path, out_path: Path) -> None:
    tensors, sources = {}, {}
    for name in SOURCES:
        path = source_dir / name
        if path.resolve() == out_path.resolve():
            raise ValueError("Output must not overwrite a source capture")
        with safe_open(str(path), framework="pt", device="cpu") as capture:
            provenance = json.loads((capture.metadata() or {})["provenance"])
            if provenance["sglang"] != "0.5.19":
                raise ValueError(f"Unexpected SGLang donor in {path}")
            keys = capture.keys()
            if not keys:
                raise ValueError(f"Empty capture: {path}")
            for key in keys:
                if key in tensors:
                    raise ValueError(f"Duplicate captured tensor: {key}")
                tensors[key] = capture.get_tensor(key)
        sources[name] = {
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "provenance": provenance,
            "tensor_count": len(keys),
        }
    metadata = {
        "scope": "actual component and attention captures; not end-to-end serving",
        "sglang": "0.5.19",
        "sources": sources,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    save_file(tensors, str(out_path), metadata={"provenance": json.dumps(metadata, sort_keys=True)})
    out_path.chmod(0o644)
    out_path.with_suffix(".json").write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
    print(f"Bundled {len(tensors)} actual donor tensors into {out_path}")


def main() -> None:
    goldens = Path(__file__).resolve().parent / "goldens"
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--source-dir", type=Path, default=goldens)
    ap.add_argument("--out", type=Path, default=goldens / "sglang_dflash2_fixtures.safetensors")
    args = ap.parse_args()
    bundle(args.source_dir, args.out)


if __name__ == "__main__":
    main()
