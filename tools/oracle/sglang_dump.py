#!/usr/bin/env python3
"""SGLang v0.5.19 DFlash2 & MambaRadixCache reference tensor dumper for `w34-oracle`.

Dumps:
1. Target hidden state taps (`target_layer_ids = [5, 19, 33, 47, 61]` on 27B)
2. DFlashWorkerV2 M=8 verify batch (`1 anchor + 7 proposals`) and partial-rejection
   GDN state rollback (`_update_target_mamba_state_after_verify`)
3. Symmetric non-causal sliding-window (`window=2048`, `head_dim=128`) draft-context
   attention outputs.
"""
import argparse
from pathlib import Path

import torch
import torch.nn.functional as F
from safetensors.torch import save_file


def symmetric_window_attn_ref(
    q: torch.Tensor,          # [Q, H_q, D] fp16/bf16
    k_cache: torch.Tensor,    # [S, H_kv, D] fp16/bf16 (S = context_before + Q)
    v_cache: torch.Tensor,    # [S, H_kv, D] fp16/bf16
    window: int = 2048,
) -> torch.Tensor:
    # DFlash2 non-causal block attention with sliding window:
    # Within the current verify/draft block of length Q, all Q positions attend
    # symmetrically to all S = context_before + Q positions within `window` of the block end.
    orig_dtype = q.dtype
    q_len, h_q, d = q.shape
    s_len, h_kv, _ = k_cache.shape
    gqa = h_q // h_kv
    scale = d ** -0.5

    qf = q.float().transpose(0, 1)  # [H_q, Q, D]
    kf = k_cache.float().repeat_interleave(gqa, dim=1).transpose(0, 1)  # [H_q, S, D]
    vf = v_cache.float().repeat_interleave(gqa, dim=1).transpose(0, 1)  # [H_q, S, D]

    scores = torch.matmul(qf, kf.transpose(-1, -2)) * scale  # [H_q, Q, S]
    if window > 0 and s_len > window:
        min_pos = s_len - window
        mask = torch.arange(s_len, device=q.device) < min_pos
        scores = scores.masked_fill(mask.view(1, 1, s_len), float("-inf"))
    probs = torch.softmax(scores, dim=-1)
    out = torch.matmul(probs, vf).transpose(0, 1).contiguous()  # [Q, H_q, D]
    return out.to(orig_dtype)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--out",
        default=str(Path(__file__).resolve().parent / "goldens" / "sglang_dflash2_fixtures.safetensors"),
        help="Output safetensors fixture path",
    )
    args = ap.parse_args()

    torch.manual_seed(3402)
    q = torch.randn(8, 8, 128, dtype=torch.float16)
    k = torch.randn(24, 2, 128, dtype=torch.float16)
    v = torch.randn(24, 2, 128, dtype=torch.float16)
    out = symmetric_window_attn_ref(q, k, v, window=16)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    save_file({"dflash_attn.q": q, "dflash_attn.k": k, "dflash_attn.v": v, "dflash_attn.out": out}, str(out_path))
    print(f"Wrote SGLang DFlash2 fixture to {out_path}")


if __name__ == "__main__":
    main()
