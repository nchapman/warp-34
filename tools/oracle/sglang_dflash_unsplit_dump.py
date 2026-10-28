#!/usr/bin/env python3
"""Capture FlashInfer FA2 with the native operator's unsplit-KV geometry.

Run in lmsysorg/sglang:v0.5.19 with GPU access. Inputs come unchanged from
sglang_dflash_attention.safetensors. The original auto-split capture remains
separate: BF16 rounding of partial outputs makes it a different numerical
execution policy, even within FlashInfer. This is an operator oracle only.
"""
import argparse
import hashlib
import json
from pathlib import Path

import flashinfer
import sglang
import torch
from safetensors import safe_open
from safetensors.torch import save_file


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--inputs", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    if args.out.exists():
        raise FileExistsError(f"Refusing to overwrite capture: {args.out}")
    assert sglang.__version__ == "0.5.19"
    i32 = lambda x: torch.tensor(x, device="cuda", dtype=torch.int32)
    tensors = {}
    with safe_open(str(args.inputs), framework="pt", device="cpu") as a:
        source = json.loads(a.metadata()["provenance"])
        assert source["sglang"] == "0.5.19"
        for case in source["cases"]:
            name = case["name"]
            q, k, v = [a.get_tensor(f"{name}.{x}").cuda() for x in ("q", "k", "v")]
            rows, heads, dim = q.shape
            seq, kv_heads, _ = k.shape
            pages = (seq + 15) // 16
            kv = torch.zeros(pages, 2, 16, kv_heads, dim, device="cuda", dtype=q.dtype)
            for page in range(pages):
                n = min(16, seq - 16 * page)
                kv[page, 0, :n] = k[16 * page:16 * page + n]
                kv[page, 1, :n] = v[16 * page:16 * page + n]
            wrapper = flashinfer.BatchPrefillWithPagedKVCacheWrapper(
                torch.empty(128 * 1024 * 1024, dtype=torch.uint8, device="cuda"),
                kv_layout="NHD", backend="fa2",
            )
            wl = case["window_left"]
            wrapper.plan(i32([0, rows]), i32([0, pages]), i32(list(range(pages))),
                         i32([(seq - 1) % 16 + 1]), heads, kv_heads, dim, 16,
                         causal=False, window_left=wl, q_data_type=q.dtype,
                         kv_data_type=q.dtype, disable_split_kv=True)
            out = wrapper.forward(q, kv, causal=False, window_left=wl, sm_scale=dim**-0.5)
            torch.cuda.synchronize()
            tensors[f"{name}.unsplit_out"] = out.cpu().contiguous()
            reference = a.get_tensor(f"{name}.out").float()
            error = out.cpu().float() - reference
            print(name, "unsplit vs auto-split", "maxabs", error.abs().max().item(),
                  "relL2", (error.norm() / reference.norm()).item(), flush=True)
    metadata = dict(sglang=sglang.__version__, flashinfer=flashinfer.__version__,
                    torch=torch.__version__, gpu=torch.cuda.get_device_name(),
                    image="lmsysorg/sglang:v0.5.19", disable_split_kv=True,
                    scope="actual unsplit FA2 operator; not default split-KV or serving parity",
                    input_sha256=hashlib.sha256(args.inputs.read_bytes()).hexdigest(),
                    input_provenance=source,
                    script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    args.out.parent.mkdir(parents=True, exist_ok=True)
    save_file(tensors, str(args.out), metadata={"provenance": json.dumps(metadata)})
    args.out.chmod(0o644)
    args.out.with_suffix(".json").write_text(json.dumps(metadata, indent=2) + "\n")


if __name__ == "__main__":
    main()
