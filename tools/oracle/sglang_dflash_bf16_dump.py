#!/usr/bin/env python3
"""Actual SGLang v0.5.19 BF16 operator captures, not hand-written expectations.

Run with GPU access in lmsysorg/sglang:v0.5.19; --out names the safetensors file.
Covers standalone/fused RMSNorm, SiluAndMul, DFlashGroupedConv prepare/finish,
and the BF16 table_qk_norm_rope_ fast path used by DFlashAttention.
"""
import argparse
import hashlib
import inspect
import json
from pathlib import Path

import sglang
import torch
import torch.nn.functional as F
from safetensors.torch import save_file
from sglang.srt.layers.activation import SiluAndMul
from sglang.srt.layers.layernorm import RMSNorm
from sglang.srt.models.dflash import DFlashGroupedConv
from sglang.srt.speculative.dflash_utils import table_qk_norm_rope_


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    assert sglang.__version__ == "0.5.19"
    torch.manual_seed(3403)
    tensors = {}
    rand = lambda *shape: torch.randn(*shape, device="cuda", dtype=torch.bfloat16)
    for h in [128, 256, 3072, 5120]:
        x, residual, weight = rand(16, h), rand(16, h), rand(h)
        x[0, 0] = 131072.0
        norm = RMSNorm(h, eps=1e-6).to(device="cuda", dtype=torch.bfloat16)
        norm.weight.data.copy_(weight)
        standalone = norm.forward_cuda(x.clone())
        fused, res_out = norm.forward_cuda(x.clone(), residual.clone())
        for suffix, value in [("x", x), ("residual", residual), ("weight", weight),
                              ("standalone", standalone), ("fused", fused), ("res_out", res_out)]:
            tensors[f"norm{h}.{suffix}"] = value
    gu = rand(32, 256)
    tensors["silu.input"] = gu
    tensors["silu.output"] = SiluAndMul().forward_cuda(gu)
    for block in [8, 16]:
        conv = DFlashGroupedConv(128, block, 3, 64).to(device="cuda", dtype=torch.bfloat16)
        conv.base_kernel.data.copy_(rand(2, 3, 128) * 0.1)
        conv.kernel_projection.weight.data.copy_(rand(12, 128) * 0.01)
        x = rand(32, 128)
        x[0, 0] = 131072.0
        # Coefficients are captured from the real BF16 linear projection and
        # passed unchanged to both the donor convolution and the CUDA replay.
        delta = F.linear(x, conv.kernel_projection.weight).reshape(32, 2, 3, 2)
        prepared, coefficients = conv.prepare(x)
        finished = conv.finish(prepared, coefficients)
        for suffix, value in [("x", x), ("base", conv.base_kernel), ("delta", delta),
                              ("prepared", prepared), ("finished", finished)]:
            tensors[f"conv{block}.{suffix}"] = value
    rows, nq, nk, dim = 16, 8, 2, 128
    qkv = rand(rows, (nq + 2 * nk) * dim)
    qw, kw = rand(dim), rand(dim)
    positions = torch.tensor([0, 1, 7, 15, 127, 255, 511, 1023]*2,
                             dtype=torch.int32, device="cuda")
    inv = 1.0 / (10000.0 ** (torch.arange(0, dim, 2, device="cuda").float() / dim))
    angles = torch.arange(1024, device="cuda").float()[:, None] * inv[None, :]
    table = torch.cat([angles.cos(), angles.sin()], dim=1).bfloat16()
    rotated = qkv.clone()
    table_qk_norm_rope_(rotated, positions, qw, kw, table, nq, nk, dim, 1e-6)
    for suffix, value in [("input", qkv), ("positions", positions), ("qw", qw), ("kw", kw),
                          ("table", table), ("output", rotated)]:
        tensors[f"rope.{suffix}"] = value
    torch.cuda.synchronize()
    sources = {}
    for obj in [RMSNorm, SiluAndMul, DFlashGroupedConv, table_qk_norm_rope_]:
        path = Path(inspect.getfile(obj))
        sources[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
    metadata = dict(sglang=sglang.__version__, torch=torch.__version__,
                    gpu=torch.cuda.get_device_name(), sources=sources,
                    dtype="bfloat16", scope="actual component outputs, not full-model parity")
    path = Path(args.out)
    path.parent.mkdir(parents=True, exist_ok=True)
    save_file({k: v.detach().cpu().contiguous() for k, v in tensors.items()}, path,
              metadata={"provenance": json.dumps(metadata)})
    path.chmod(0o644)
    path.with_suffix(".json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(f"Wrote {len(tensors)} tensors to {path}", flush=True)


if __name__ == "__main__":
    main()
