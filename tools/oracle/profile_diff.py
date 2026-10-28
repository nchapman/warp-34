#!/usr/bin/env python3
"""Automated side-by-side nsys SQLite & kernel category scoreboard (`w34-oracle profile-diff`).

Categorizes GPU kernels from `nsys profile` SQLite exports (or JSON profile summaries)
into the 7 canonical warp-34 categories (`docs/ROADMAP.md` §3.3):
  1. marlin_gemm    : Marlin W4A16/W8A16/W4A8 + cuBLAS GEMMs (trunk + heads)
  2. full_attn      : FlashAttention prefill + paged decode/verify attention
  3. gdn_ssm        : causal_conv1d + chunk_gated_delta_rule + fused_recurrent GDN
  4. norms_rope_act : GemmaRMSNorm, fused_qk_rmsnorm_rope_gate, silu_and_mul, RMSNormGated
  5. tp_comm        : TP=2 pinned-sysmem all_reduce / NCCL / exchange kernels
  6. spec_decode    : MTP / DFlash2 drafter + rejection_sampler
  7. host_gap       : Inter-kernel / round-boundary GPU idle gap
"""
import argparse
import json
import sqlite3
from pathlib import Path

CATEGORIES = [
    "marlin_gemm",
    "full_attn",
    "gdn_ssm",
    "norms_rope_act",
    "tp_comm",
    "spec_decode",
]


def classify_kernel(name: str) -> str:
    s = name.lower()
    if any(k in s for k in ("all_reduce", "allreduce", "all_gather", "nccl", "exchange", "tp2_", "cross_device")):
        return "tp_comm"
    if any(k in s for k in ("dflash", "mtp", "rejection", "accept_walk", "draft")):
        return "spec_decode"
    if any(k in s for k in ("gdn", "delta_rule", "conv1d", "ssm", "mamba", "recurrent")):
        return "gdn_ssm"
    if any(k in s for k in ("rms_norm", "rmsnorm", "layernorm", "rope", "silu", "act_and_mul", "output_gate", "split_cols", "sampler", "elementwise", "embed")):
        return "norms_rope_act"
    if any(k in s for k in ("flash_attn", "fattn", "paged_attn", "paged_attention", "kv_cache_write", "fmha", "attn")):
        return "full_attn"
    if any(k in s for k in ("marlin", "gemm", "gemv", "cutlass", "ampere_fp16", "cublas", "matmul")):
        return "marlin_gemm"
    return "norms_rope_act"


def parse_nsys_sqlite(sqlite_path: Path, rounds: int = 128, device_id: int = 0) -> dict:
    conn = sqlite3.connect(str(sqlite_path))
    cur = conn.cursor()
    rows = cur.execute(
        """
        SELECT s.value, k.start, k.end
        FROM CUPTI_ACTIVITY_KIND_KERNEL k
        JOIN StringIds s ON k.shortName = s.id
        WHERE k.deviceId = ?
        ORDER BY k.start ASC
        """,
        (device_id,),
    ).fetchall()
    conn.close()

    cat_ns = {c: 0 for c in CATEGORIES}
    cat_launches = {c: 0 for c in CATEGORIES}
    if not rows:
        return {"rounds": rounds, "categories": {c: {"ms_per_round": 0.0, "launches_per_round": 0.0} for c in CATEGORIES + ["host_gap"]}, "total_ms_per_round": 0.0}

    # If trace contains w34 embed_lookup_f16_kernel launches, slice exactly the last
    # `rounds` decode steps (skipping model load repack, warmup, and prefill).
    embed_indices = [i for i, (kname, _, _) in enumerate(rows) if "embed_lookup" in kname.lower()]
    if len(embed_indices) >= 4:
        take_rounds = min(rounds, len(embed_indices) - 2) if len(embed_indices) > 2 else len(embed_indices)
        start_idx = embed_indices[-take_rounds]
        window = rows[start_idx:]
        effective_rounds = float(take_rounds)
    else:
        start_idx = len(rows) // 10 if len(rows) > 200 else 0
        window = rows[start_idx:]
        effective_rounds = max(1.0, rounds * (len(window) / max(1, len(rows))))

    prev_end = window[0][1]
    gap_ns = 0
    for kname, start, end in window:
        dur = max(0, end - start)
        if start > prev_end:
            gap_ns += start - prev_end
        prev_end = max(prev_end, end)
        cat = classify_kernel(kname)
        cat_ns[cat] += dur
        cat_launches[cat] += 1

    out_cats = {}
    total_ms = 0.0
    for c in CATEGORIES:
        ms_r = (cat_ns[c] / 1e6) / effective_rounds
        l_r = cat_launches[c] / effective_rounds
        out_cats[c] = {"ms_per_round": round(ms_r, 4), "launches_per_round": round(l_r, 2)}
        total_ms += ms_r
    gap_ms_r = (gap_ns / 1e6) / effective_rounds
    out_cats["host_gap"] = {"ms_per_round": round(gap_ms_r, 4), "launches_per_round": 0.0}
    total_ms += gap_ms_r
    return {
        "source": str(sqlite_path),
        "rounds": round(effective_rounds, 1),
        "categories": out_cats,
        "total_ms_per_round": round(total_ms, 4),
        "tok_per_s_m1": round(1000.0 / total_ms, 2) if total_ms > 0 else 0.0,
    }


def load_profile(path: Path, rounds: int, device_id: int, preset: str | None = None) -> dict:
    if path.suffix == ".json":
        data = json.loads(path.read_text())
        if preset and preset in data:
            return data[preset]
        if "categories" not in data and len(data) > 0:
            first_key = next(iter(data.keys()))
            return data[first_key]
        return data
    return parse_nsys_sqlite(path, rounds=rounds, device_id=device_id)


def render_scoreboard(oracle_label: str, oracle: dict, w34_label: str, w34: dict) -> str:
    lines = []
    lines.append(f"=== warp-34 vs {oracle_label} Kernel Scoreboard (ms/round) ===")
    header = f"{'Category':<18} | {oracle_label + ' (ms/r)':>16} | {w34_label + ' (ms/r)':>16} | {'Delta (ms/r)':>13} | {'Speedup':>9}"
    lines.append(header)
    lines.append("-" * len(header))
    for c in CATEGORIES + ["host_gap"]:
        o_ms = oracle["categories"].get(c, {}).get("ms_per_round", 0.0)
        w_ms = w34["categories"].get(c, {}).get("ms_per_round", 0.0)
        delta = w_ms - o_ms
        sp = f"{o_ms / w_ms:.2f}x" if w_ms > 1e-6 and o_ms > 1e-6 else "n/a"
        lines.append(f"{c:<18} | {o_ms:>16.4f} | {w_ms:>16.4f} | {delta:>+13.4f} | {sp:>9}")
    lines.append("-" * len(header))
    o_tot = oracle.get("total_ms_per_round", 0.0)
    w_tot = w34.get("total_ms_per_round", 0.0)
    tot_sp = f"{o_tot / w_tot:.2f}x" if w_tot > 1e-6 and o_tot > 1e-6 else "n/a"
    lines.append(f"{'TOTAL':<18} | {o_tot:>16.4f} | {w_tot:>16.4f} | {w_tot - o_tot:>+13.4f} | {tot_sp:>9}")
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--oracle", required=True, help="Oracle nsys .sqlite or .json profile")
    ap.add_argument("--preset", default=None, help="Preset key inside oracle_baselines.json")
    ap.add_argument("--oracle-label", default="vLLM 0.31.0", help="Oracle column header")
    ap.add_argument("--w34", default=None, help="warp-34 nsys .sqlite or .json profile")
    ap.add_argument("--w34-label", default="warp-34", help="warp-34 column header")
    ap.add_argument("--rounds", type=int, default=128, help="Nominal rounds in trace")
    ap.add_argument("--device", type=int, default=0, help="CUDA deviceId in nsys trace")
    ap.add_argument("--json-out", default=None, help="Optional JSON report path")
    args = ap.parse_args()

    oracle_prof = load_profile(Path(args.oracle), args.rounds, args.device, args.preset)
    if args.w34:
        w34_prof = load_profile(Path(args.w34), args.rounds, args.device, args.preset)
    else:
        w34_prof = oracle_prof

    table = render_scoreboard(args.oracle_label, oracle_prof, args.w34_label, w34_prof)
    print(table)
    if args.json_out:
        Path(args.json_out).write_text(json.dumps({"oracle": oracle_prof, "w34": w34_prof}, indent=2))


if __name__ == "__main__":
    main()
