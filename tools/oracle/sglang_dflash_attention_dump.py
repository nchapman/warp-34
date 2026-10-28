#!/usr/bin/env python3
"""Dump the actual SGLang v0.5.19 FlashInfer paged draft-attention operator.

Unlike sglang_dump.py's legacy hand-written block-end mask, outputs here come
from the backend used by sglang/srt/layers/attention/flashinfer_backend.py
forward_extend (ENCODER_ONLY => causal=False, window_left=HF window-1).
This captures the attention operator, NOT a complete DFlashWorkerV2 forward.

Run in the pinned image with GPU access and this repository mounted at /work:
  python /work/tools/oracle/sglang_dflash_attention_dump.py --out /work/tools/oracle/goldens/sglang_dflash_attention.safetensors
"""
import argparse
import hashlib
import inspect
import json
from pathlib import Path

import flashinfer
import sglang
import torch
from safetensors.torch import save_file
from sglang.srt.layers.attention import flashinfer_backend


def reference(q, k, v, window, mode):
    q_len, heads, dim = q.shape
    seq_len, kv_heads, _ = k.shape
    pos = seq_len - q_len + torch.arange(q_len, device=q.device)
    keys = torch.arange(seq_len, device=q.device)
    mask = torch.ones((q_len, seq_len), dtype=torch.bool, device=q.device)
    if window:
        if mode == "block_end":
            mask &= keys[None, :] >= seq_len - window
        else:
            mask &= keys[None, :] >= pos[:, None] - window + 1
            if mode == "symmetric":
                mask &= keys[None, :] <= pos[:, None] + window - 1
    k = k.float().repeat_interleave(heads // kv_heads, dim=1).transpose(0, 1)
    v = v.float().repeat_interleave(heads // kv_heads, dim=1).transpose(0, 1)
    scores = q.float().transpose(0, 1) @ k.transpose(-1, -2) / dim**0.5
    probs = torch.softmax(scores.masked_fill(~mask[None, :, :], -torch.inf), dim=-1)
    return (probs @ v).transpose(0, 1).contiguous().to(q.dtype)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    if sglang.__version__ != "0.5.19":
        raise RuntimeError(f"expected SGLang 0.5.19, got {sglang.__version__}")
    torch.manual_seed(3402)
    tensors, cases = {}, []
    # Last case makes the upper-window bound observable: real checkpoint blocks
    # are smaller than their 2048 window and cannot distinguish that bound.
    for name, q_len, seq_len, window in [
        ("short", 8, 24, 16), ("full", 16, 24, 0),
        ("long", 8, 2064, 2048), ("narrow", 16, 24, 4),
    ]:
        heads, kv_heads, dim, page_size = 8, 2, 128, 16
        q = torch.randn(q_len, heads, dim, device="cuda", dtype=torch.bfloat16)
        k = torch.randn(seq_len, kv_heads, dim, device="cuda", dtype=torch.bfloat16)
        v = torch.randn_like(k)
        pages = (seq_len + page_size - 1) // page_size
        # Physical page reversal also exercises paged, rather than contiguous, reads.
        kv = torch.zeros(pages, 2, page_size, kv_heads, dim, device="cuda", dtype=q.dtype)
        for p in range(pages):
            n = min(page_size, seq_len - p * page_size)
            kv[pages - p - 1, 0, :n] = k[p*page_size:p*page_size+n]
            kv[pages - p - 1, 1, :n] = v[p*page_size:p*page_size+n]
        i32 = lambda x: torch.tensor(x, device="cuda", dtype=torch.int32)
        wrapper = flashinfer.BatchPrefillWithPagedKVCacheWrapper(
            torch.empty(128 * 1024 * 1024, dtype=torch.uint8, device="cuda"),
            kv_layout="NHD", backend="fa2",
        )
        window_left = window - 1 if window else -1
        wrapper.plan(i32([0, q_len]), i32([0, pages]), i32(list(reversed(range(pages)))),
                     i32([(seq_len - 1) % page_size + 1]), heads, kv_heads, dim,
                     page_size, causal=False, window_left=window_left,
                     q_data_type=q.dtype, kv_data_type=k.dtype)
        output = wrapper.forward(q, kv, causal=False, window_left=window_left,
                                 sm_scale=dim**-0.5)
        torch.cuda.synchronize()
        errors = {mode: float((output.float() - reference(q, k, v, window, mode).float()).abs().max())
                  for mode in ["block_end", "query_left", "symmetric"]}
        print(name, errors, flush=True)
        for suffix, value in [("q", q), ("k", k), ("v", v), ("out", output)]:
            tensors[f"{name}.{suffix}"] = value.cpu().contiguous()
        cases.append(dict(name=name, q_len=q_len, seq_len=seq_len, q_pos_base=seq_len-q_len,
                          window=window, window_left=window_left, causal=False,
                          reference_max_abs_errors=errors))
    source = Path(inspect.getfile(flashinfer_backend))
    metadata = dict(sglang=sglang.__version__, flashinfer=flashinfer.__version__,
                    torch=torch.__version__, backend="FlashInfer fa2 paged prefill",
                    gpu=torch.cuda.get_device_name(), dtype="bfloat16", cases=cases,
                    source=str(source), source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                    scope="operator only; not a DFlashWorkerV2 end-to-end capture")
    path = Path(args.out)
    path.parent.mkdir(parents=True, exist_ok=True)
    save_file(tensors, path, metadata={"provenance": json.dumps(metadata)})
    # Containers may run as root; the non-secret fixture must remain readable
    # by the host test runner (safetensors creates files with restrictive mode).
    path.chmod(0o644)
    path.with_suffix(".json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(f"Wrote {path}", flush=True)


if __name__ == "__main__":
    main()
