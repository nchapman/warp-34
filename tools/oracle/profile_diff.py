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
import shlex
import sqlite3
import subprocess
import tempfile
from contextlib import nullcontext
from pathlib import Path

CATEGORIES = [
    "marlin_gemm",
    "full_attn",
    "gdn_ssm",
    "norms_rope_act",
    "tp_comm",
    "spec_decode",
]


def classify_kernel(name: str, in_spec_span: bool = False) -> str:
    s = name.lower()
    if any(k in s for k in ("all_reduce", "allreduce", "all_gather", "nccl", "exchange", "tp2_", "cross_device")):
        return "tp_comm"
    if in_spec_span or any(
        k in s
        for k in (
            "dflash",
            "mtp",
            "rejection",
            "accept_walk",
            "draft",
            "tap_add",
            "16bf",
            "bfloat16",
            "bf16",
        )
    ):
        return "spec_decode"
    if any(k in s for k in ("gdn", "delta_rule", "conv1d", "ssm", "mamba", "recurrent")):
        return "gdn_ssm"
    if "vision_qkv_rope_attn" in s:
        return "full_attn"
    if any(k in s for k in ("moe_w4a16", "moe_f16", "moe_wna16", "fused_moe")):
        return "marlin_gemm"
    if any(
        k in s
        for k in (
            "rms_norm",
            "rmsnorm",
            "layernorm",
            "rope",
            "silu",
            "act_and_mul",
            "output_gate",
            "split_cols",
            "sampler",
            "topk_rows",
            "moe_topk_router",
            "vision_bias",
            "embedding_splice",
            "elementwise",
            "embed",
        )
    ):
        return "norms_rope_act"
    if any(k in s for k in ("flash_attn", "fattn", "paged_attn", "paged_attention", "kv_cache_write", "fmha", "attn", "batchprefillwithpagedkvcache", "batchdecodewithpagedkvcache")):
        return "full_attn"
    if any(
        k in s
        for k in (
            "marlin",
            "gemm",
            "gemv",
            "cutlass",
            "kernel2",
            "splitkreduce",
            "ampere_fp16",
            "cublas",
            "matmul",
        )
    ):
        return "marlin_gemm"
    return "norms_rope_act"


def _find_target_passes(rows: list[tuple[str, int, int]]) -> list[tuple[int, int]]:
    """Returns `(start_idx, target_end_idx)` for each target forward pass in `rows`.

    Target passes start at `embed_lookup` and contain GDN (`conv1d`/`gdn`) layers before
    their final `topk_rows` / `sampler` (whereas MTP draft passes have 0 GDN layers and
    DFlash2 uses `embed_lookup_bf16`).
    """
    embed_f16 = [
        i
        for i, (kname, _, _) in enumerate(rows)
        if "embed_lookup" in kname.lower() and "bf16" not in kname.lower()
    ]
    if not embed_f16:
        return []

    has_any_gdn = any("conv1d" in k.lower() or "gdn" in k.lower() for k, _, _ in rows)
    target_starts: list[int] = []
    for pos, idx in enumerate(embed_f16):
        nxt = embed_f16[pos + 1] if pos + 1 < len(embed_f16) else len(rows)
        if not has_any_gdn:
            target_starts.append(idx)
            continue
        gdn_count = sum(
            1
            for k, _, _ in rows[idx:nxt]
            if "conv1d" in k.lower() or "gdn" in k.lower()
        )
        if gdn_count >= 4:
            target_starts.append(idx)

    passes: list[tuple[int, int]] = []
    for pos, s_idx in enumerate(target_starts):
        nxt_s = target_starts[pos + 1] if pos + 1 < len(target_starts) else len(rows)
        end_idx = nxt_s - 1
        for j in range(s_idx, nxt_s):
            kl = rows[j][0].lower()
            if "topk_rows" in kl or "sampler_greedy" in kl:
                end_idx = j
                if j + 1 < nxt_s and "all_gather_top1" in rows[j + 1][0].lower():
                    end_idx = j + 1
                break
        passes.append((s_idx, end_idx))
    return passes


def nvtx_candidates(path: Path) -> dict:
    """CPU intervals only: never use these directly as GPU-time boundaries."""
    with sqlite3.connect(str(path)) as conn:
        rows = conn.execute("""
            SELECT COALESCE(n.text, s.value), n.start, n.end, n.globalTid
            FROM NVTX_EVENTS n LEFT JOIN StringIds s ON n.textId = s.id
            WHERE n.end > n.start ORDER BY n.start
        """).fetchall()
    measured = [r for r in rows if r[0] == "w34.bench.measured"]
    if not measured:
        raise ValueError("No completed w34.bench.measured range")
    last = measured[-1]
    decode = [r for r in rows if r[0] == "w34.decode" and
              r[3] == last[3] and last[1] <= r[1] < r[2] <= last[2]]
    if not decode:
        raise ValueError("Last measured range has no nested decode range")
    return {"measured_cpu_interval": last, "decode_cpu_candidates": decode,
            "warning": "CPU times only; CUDA runtime/graph correlation not implemented"}


def parse_nvtx_correlated(path: Path, metadata: dict) -> dict:
    """Attribute launches, not GPU timestamp overlap; all threads/devices in marker PID."""
    if not isinstance(metadata, dict):
        raise ValueError("bench telemetry must be an object")
    rounds = metadata.get("verify_rounds")
    if rounds is None:
        rounds = metadata.get("measured_rounds")
    if type(rounds) is not int or rounds <= 0:
        raise ValueError("bench telemetry requires positive explicit verify_rounds/measured_rounds")
    mask = ~((1 << 24) - 1)
    with sqlite3.connect(str(path)) as conn:
        spans = conn.execute("""
            SELECT COALESCE(n.text,s.value),n.start,n.end,n.globalTid,n.endGlobalTid
            FROM NVTX_EVENTS n LEFT JOIN StringIds s ON n.textId=s.id
        """).fetchall()
        measured = [r for r in spans if r[0] == 'w34.bench.measured']
        if not measured:
            raise ValueError("Missing measured NVTX marker")
        latest = max(r[1] for r in measured)
        last = [r for r in measured if r[1] == latest]
        if len(last) != 1:
            raise ValueError("Ambiguous measured marker")
        _, a, b, tid, etid = last[0]
        if b is None or b <= a or tid is None or etid not in (None, 0, tid):
            raise ValueError("Invalid measured marker")
        dec = [r for r in spans if r[0] == 'w34.decode' and r[3] == tid
               and r[1] >= a and r[1] < b]
        if len(dec) != 1:
            raise ValueError("Missing/ambiguous nested decode marker")
        _, start, end, _, etid = dec[0]
        if end is None or not start < end <= b or etid not in (None, 0, tid):
            raise ValueError("Invalid decode marker")
        pid = tid & mask
        runtime = conn.execute("""
            SELECT r.start,r.end,r.globalTid,r.correlationId,s.value,r.returnValue
            FROM CUPTI_ACTIVITY_KIND_RUNTIME r LEFT JOIN StringIds s ON r.nameId=s.id
        """).fetchall()
        by_key = {}
        for r in runtime:
            if r[2] is not None and r[2] & mask == pid:
                by_key.setdefault(r[3], []).append(r)
        kernels = conn.execute("""
            SELECT k.start,k.end,k.deviceId,k.correlationId,s.value
            FROM CUPTI_ACTIVITY_KIND_KERNEL k LEFT JOIN StringIds s ON k.demangledName=s.id
            WHERE k.globalPid=? ORDER BY k.start
        """, (pid,)).fetchall()
    ns = dict.fromkeys(CATEGORIES, 0)
    counts = dict.fromkeys(CATEGORIES, 0)
    selected = []
    devices = {}
    proposer = [r for r in spans if r[0] == 'w34.proposer' and r[3] is not None
                and r[3] & mask == pid and r[2] is not None and start <= r[1] < r[2] <= end]
    for ka, kb, device, corr, name in kernels:
        matches = by_key.get(corr, [])
        inside = [r for r in matches if start <= r[0] < end]
        if not inside:
            continue
        if len(matches) != 1:
            raise ValueError("Ambiguous runtime correlation")
        ra, rb, rt, _, api, status = inside[0]
        if not ra <= rb <= end or status != 0 or not api or 'launch' not in api.lower():
            raise ValueError("Invalid correlated launch")
        if kb <= ka or not name:
            raise ValueError("Invalid kernel metadata")
        spec = any(t == rt and pa <= ra <= rb <= pb for _, pa, pb, t, _ in proposer)
        cat = classify_kernel(name, spec)
        ns[cat] += kb - ka
        counts[cat] += 1
        d = devices.setdefault(str(device), {
            'ns': dict.fromkeys(CATEGORIES, 0),
            'counts': dict.fromkeys(CATEGORIES, 0),
            'proposer_graph_launches': set(), 'proposer_graph_launch_threads': set(),
        })
        d['ns'][cat] += kb - ka
        d['counts'][cat] += 1
        if spec and api.startswith('cudaGraphLaunch'):
            d['proposer_graph_launches'].add(corr)
            d['proposer_graph_launch_threads'].add(rt)
        selected.append((ka, kb, device))
    if not selected:
        raise ValueError("No correlated kernels")
    cats = {c: {"ms_per_round": ns[c] / 1e6 / rounds,
                "launches_per_round": counts[c] / rounds} for c in CATEGORIES}
    per_device = {
        device: {
            'kernel_count': sum(d['counts'].values()),
            'summed_gpu_kernel_ms': sum(d['ns'].values()) / 1e6,
            'categories': {c: {'ms_per_round': d['ns'][c] / 1e6 / rounds,
                               'launches_per_round': d['counts'][c] / rounds}
                           for c in CATEGORIES},
            'proposer_graph_launch_count': len(d['proposer_graph_launches']),
            'proposer_graph_launch_threads': sorted(d['proposer_graph_launch_threads']),
        }
        for device, d in sorted(devices.items(), key=lambda item: int(item[0]))
    }
    return {"source": str(path), "measurement_mode": "nvtx_correlated",
            "devices": per_device,
            "boundary_verified": True, "gate_eligible": False, "rounds": rounds,
            "metadata": metadata, "decode_cpu_interval_ns": [start, end],
            "kernel_count": len(selected), "device_ids": sorted({r[2] for r in selected}),
            "categories": cats, "summed_gpu_kernel_ms": sum(ns.values()) / 1e6,
            "total_ms_per_round": (end-start) / 1e6 / rounds,
            "gpu_extent_ns": [min(r[0] for r in selected), max(r[1] for r in selected)],
            "cost_semantics": "categories are summed GPU kernel time across devices, NOT wall time; total is CPU decode wall time; host gap unavailable",
            "limitations": "Workload alignment unverified; donor comparison not valid. Categories heuristic except thread-local proposer launch spans."}


def parse_oracle_nvtx_correlated(path: Path, metadata: dict) -> dict:
    """TP1 RID ranges; runtime correlation includes every node of graph launches.

    Nsight globalPid and telemetry PID live in different namespaces. Match the
    unique measured worker by RID and count, never by numerical PID equality.
    """
    if not isinstance(metadata, dict):
        raise ValueError('oracle telemetry must be an object')
    rounds = metadata.get('verify_rounds')
    if type(rounds) is not int or rounds <= 0:
        raise ValueError('positive worker verify_rounds required')
    events = [e for e in metadata.get('worker_events', [])
              if e.get('kind') == 'decode' and e.get('rids') == ['oracle-measured']]
    if len(events) != rounds or len({e.get('pid') for e in events}) != 1:
        raise ValueError('require one TP1 telemetry worker with matching round count')
    mask = ~((1 << 24) - 1)
    with sqlite3.connect(str(path)) as conn:
        spans = conn.execute('''SELECT COALESCE(n.text,s.value),n.start,n.end,
            n.globalTid,n.endGlobalTid FROM NVTX_EVENTS n
            LEFT JOIN StringIds s ON n.textId=s.id''').fetchall()
        dec = sorted([r for r in spans if r[0] == 'oracle.decode.round:oracle-measured'], key=lambda r:r[1])
        if not dec:
            raise ValueError('Missing measured RID marker')
        if any(b is None or t is None or b <= a or et not in (None, 0, t)
               for _, a, b, t, et in dec):
            raise ValueError('Incomplete/invalid measured RID range')
        pids = {r[3] & mask for r in dec}
        if len(pids) != 1 or len(dec) != rounds:
            raise ValueError('Ambiguous worker PID or telemetry/range count mismatch')
        if any(x[2] > y[1] for x,y in zip(dec,dec[1:])):
            raise ValueError('Duplicate/overlapping worker ranges')
        pid = next(iter(pids))
        runtime = conn.execute('''SELECT r.start,r.end,r.globalTid,r.correlationId,
            s.value,r.returnValue FROM CUPTI_ACTIVITY_KIND_RUNTIME r
            LEFT JOIN StringIds s ON r.nameId=s.id''').fetchall()
        kernels = conn.execute('''SELECT k.start,k.end,k.deviceId,k.correlationId,s.value
            FROM CUPTI_ACTIVITY_KIND_KERNEL k LEFT JOIN StringIds s ON k.demangledName=s.id
            WHERE k.globalPid=? ORDER BY k.start''', (pid,)).fetchall()
    by_key = {}
    for r in runtime:
        if r[2] is not None and r[2] & mask == pid:
            by_key.setdefault(r[3], []).append(r)
    proposer = [r for r in spans if r[0] == 'w34.proposer' and r[3] is not None
                and r[3] & mask == pid and r[2] is not None]
    ns, counts = dict.fromkeys(CATEGORIES,0), dict.fromkeys(CATEGORIES,0)
    selected, seen_rounds = [], set()
    for ka,kb,device,corr,name in kernels:
        matches = by_key.get(corr, [])
        inside = [(r,i) for r in matches for i,(_,a,b,_,_) in enumerate(dec) if a <= r[0] < b]
        if not inside:
            continue
        if len(matches) != 1 or len(inside) != 1:
            raise ValueError('Ambiguous runtime correlation')
        (ra,rb,rt,_,api,status),i = inside[0]
        if not ra <= rb <= dec[i][2] or status != 0 or not api or 'launch' not in api.lower():
            raise ValueError('Invalid correlated launch')
        if kb <= ka or not name:
            raise ValueError('Invalid kernel metadata')
        spec = any(t == rt and dec[i][1] <= a <= ra <= rb <= b <= dec[i][2]
                   for _,a,b,t,_ in proposer)
        cat = classify_kernel(name,spec)
        ns[cat] += kb-ka
        counts[cat] += 1
        selected.append((ka,kb,device))
        seen_rounds.add(i)
    if len(seen_rounds) != rounds:
        raise ValueError('Missing correlated kernels in worker rounds (unflushed trace?)')
    cpu_ms = sum(r[2]-r[1] for r in dec)/1e6
    return dict(source=str(path), measurement_mode='oracle_nvtx_correlated',
        boundary_verified=True, gate_eligible=False, rounds=rounds,
        scheduler_verify_rounds=metadata.get('scheduler_verify_rounds'),
        trace_global_pid=pid, telemetry_container_pid=events[0]['pid'],
        pid_matching='Unique TP1 measured RID worker and exact telemetry count; no numeric cross-namespace PID comparison',
        categories={c:dict(ms_per_round=ns[c]/1e6/rounds, launches_per_round=counts[c]/rounds) for c in CATEGORIES},
        kernel_count=len(selected), device_ids=sorted({r[2] for r in selected}),
        summed_gpu_kernel_ms=sum(ns.values())/1e6,
        summed_gpu_kernel_ms_per_round=sum(ns.values())/1e6/rounds,
        total_ms_per_round=cpu_ms/rounds, worker_cpu_range_ms=cpu_ms,
        streamed_decode_ms=metadata.get('decode_ms'),
        gpu_extent_ns=[min(r[0] for r in selected),max(r[1] for r in selected)],
        cost_semantics='Categories sum GPU kernel durations, not elapsed wall time. Total is mean worker CPU range duration including synchronous telemetry; streamed decode is first-to-last delivered token. No host-gap estimate.',
        limitations='Input/output alignment not checked; no winner/gate claim. Proposer ingest+propose classified by thread-local NVTX launch scopes; remaining kernels use name heuristics. Overlap-scheduled extra work is retained, not normalized to scheduler rounds.')


def parse_explicit_window(path: Path, start: int, end: int, metadata: Path,
                          device_id: int = 0) -> dict:
    """Strict external GPU window. Reject partial kernels rather than clipping."""
    meta = json.loads(metadata.read_text())
    if not isinstance(meta, dict):
        raise ValueError("metadata must be an object")
    rounds = meta.get("rounds")
    if type(rounds) is not int or rounds <= 0:
        raise ValueError("metadata.rounds must be an exact positive integer")
    if start is None or end is None or start >= end:
        raise ValueError("Strict mode requires ordered explicit GPU-time boundaries")
    with sqlite3.connect(str(path)) as conn:
        rows = conn.execute("""
            SELECT s.value, k.start, k.end FROM CUPTI_ACTIVITY_KIND_KERNEL k
            LEFT JOIN StringIds s ON k.shortName = s.id
            WHERE k.deviceId = ? AND k.end > ? AND k.start < ? ORDER BY k.start
        """, (device_id, start, end)).fetchall()
    if not rows:
        raise ValueError("No kernels in explicit window")
    if any(not name for name, _, _ in rows):
        raise ValueError("Invalid kernel metadata: unnamed kernel")
    if any(a < start or b > end or b <= a for _, a, b in rows):
        raise ValueError("Kernel crosses window boundary or has invalid duration")
    ns = dict.fromkeys(CATEGORIES, 0)
    launches = dict.fromkeys(CATEGORIES, 0)
    gap, previous = 0, start
    for name, a, b in rows:
        cat = classify_kernel(name)
        ns[cat] += b - a
        launches[cat] += 1
        gap += max(0, a - previous)
        previous = max(previous, b)
    gap += end - previous
    cats = {c: {"ms_per_round": ns[c] / 1e6 / rounds,
                "launches_per_round": launches[c] / rounds} for c in CATEGORIES}
    cats["host_gap"] = {"ms_per_round": gap / 1e6 / rounds, "launches_per_round": 0}
    return {"source": str(path), "measurement_mode": "explicit_gpu_window",
            "gate_eligible": False, "rounds": rounds, "metadata": meta,
            "window_start_ns": start, "window_end_ns": end, "categories": cats,
            "total_ms_per_round": (end - start) / 1e6 / rounds,
            "limitations": "Externally supplied boundaries; alignment unverified; categories heuristic"}


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
        raise ValueError("No kernel data in trace")

    target_passes = _find_target_passes(rows)
    spec_indices: set[int] = set()
    for pos, (s_idx, e_idx) in enumerate(target_passes):
        nxt_s = target_passes[pos + 1][0] if pos + 1 < len(target_passes) else len(rows)
        for j in range(e_idx + 1, nxt_s):
            spec_indices.add(j)

    if len(target_passes) >= 4:
        # Exclude warmup, reset_all_states(), prefill, and CUDA-Graph-capture passes:
        # find the last pass that contains a prefill kernel, has duration >2.2x the
        # median pass duration (multi-token prefill), or is preceded by a >15ms
        # transition gap, and take only the steady-state decode/verify passes after it.
        pass_durs = sorted(max(1, rows[e][2] - rows[s][1]) for s, e in target_passes)
        median_dur_ns = pass_durs[len(pass_durs) // 2]
        last_transition_pos = 1
        for pos, (s_idx, e_idx) in enumerate(target_passes):
            has_prefill = any(
                "prefill" in rows[j][0].lower() or "chunk_gated" in rows[j][0].lower()
                for j in range(s_idx, e_idx + 1)
            )
            dur_ns = rows[e_idx][2] - rows[s_idx][1]
            prev_e = target_passes[pos - 1][1] if pos > 0 else s_idx
            gap_before_ns = rows[s_idx][1] - rows[prev_e][2] if pos > 0 else 0
            if has_prefill or dur_ns > 2.2 * median_dur_ns or gap_before_ns > 15_000_000:
                last_transition_pos = pos
        min_steady = min(4, len(target_passes) - 2)
        steady_passes = (
            target_passes[last_transition_pos + 1 :]
            if last_transition_pos + min_steady < len(target_passes)
            else target_passes[2:]
        )
        take_rounds = min(rounds, len(steady_passes))
        start_idx = steady_passes[-take_rounds][0]
        effective_rounds = float(take_rounds)
        window_indices = range(start_idx, len(rows))
    else:
        start_idx = len(rows) // 10 if len(rows) > 200 else 0
        window_indices = range(start_idx, len(rows))
        effective_rounds = max(1.0, rounds * (len(window_indices) / max(1, len(rows))))

    first_i = window_indices[0]
    prev_end = rows[first_i][1]
    gap_ns = 0
    for idx in window_indices:
        kname, start, end = rows[idx]
        dur = max(0, end - start)
        if start > prev_end:
            gap_ns += start - prev_end
        prev_end = max(prev_end, end)
        cat = classify_kernel(kname, in_spec_span=(idx in spec_indices))
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
        "measurement_mode": "legacy_heuristic",
        "gate_eligible": False,
        "source": str(sqlite_path),
        "rounds": round(effective_rounds, 1),
        "categories": out_cats,
        "total_ms_per_round": round(total_ms, 4),
        "tok_per_s_m1": round(1000.0 / total_ms, 2) if total_ms > 0 else 0.0,
    }


def run_live_nsys(live_cmd: str, rounds: int, device_id: int, retain_trace: str | None = None, nvtx_correlated: bool = False) -> dict:
    if retain_trace:
        Path(retain_trace).mkdir(parents=True, exist_ok=True)
    with (nullcontext(retain_trace) if retain_trace else tempfile.TemporaryDirectory(prefix="w34_nsys_")) as tmpdir:
        rep_prefix = Path(tmpdir) / "trace"
        rep_file = Path(tmpdir) / "trace.nsys-rep"
        sqlite_file = Path(tmpdir) / "trace.sqlite"
        cmd_str = live_cmd.strip()
        if cmd_str.startswith("["):
            cmd_parts = [str(x) for x in json.loads(cmd_str)]
        else:
            cmd_parts = shlex.split(live_cmd)
        nsys_cmd = [
            "nsys",
            "profile",
            "--trace=cuda,nvtx",
            "--cuda-graph-trace=node",
            "--sample=none",
            "--cpuctxsw=none",
            "--force-overwrite=true",
            "-o",
            str(rep_prefix),
            *cmd_parts,
        ]
        proc = subprocess.run(nsys_cmd, capture_output=True, text=True)
        if proc.returncode != 0:
            raise RuntimeError(
                f"nsys profile failed (rc={proc.returncode}):\n{proc.stderr}\n{proc.stdout}"
            )
        export_proc = subprocess.run(
            [
                "nsys",
                "export",
                "--type=sqlite",
                "--force-overwrite=true",
                "-o",
                str(sqlite_file),
                str(rep_file),
            ],
            capture_output=True,
            text=True,
        )
        if export_proc.returncode != 0:
            raise RuntimeError(
                f"nsys export failed (rc={export_proc.returncode}):\n{export_proc.stderr}\n{export_proc.stdout}"
            )
        bench_json = None
        stdout = proc.stdout
        brace_start = stdout.find("{")
        brace_end = stdout.rfind("}")
        if brace_start != -1 and brace_end > brace_start:
            try:
                bench_json = json.loads(stdout[brace_start : brace_end + 1])
            except json.JSONDecodeError:
                bench_json = None

        if retain_trace:
            (Path(tmpdir) / "stdout.txt").write_text(stdout)
            (Path(tmpdir) / "stderr.txt").write_text(proc.stderr)
            (Path(tmpdir) / "stdout.json").write_text(json.dumps(bench_json, indent=2))

        actual_rounds = rounds
        if isinstance(bench_json, dict):
            if bench_json.get("verify_rounds") is not None:
                actual_rounds = int(bench_json["verify_rounds"])
            elif bench_json.get("gen_len") is not None:
                actual_rounds = max(1, int(bench_json["gen_len"]) - 1)

        prof = (parse_nvtx_correlated(sqlite_file, bench_json or {}) if nvtx_correlated
                else parse_nsys_sqlite(sqlite_file, rounds=actual_rounds, device_id=device_id))
        prof["source"] = f"live nsys ({live_cmd})"
        if isinstance(bench_json, dict):
            if "decode_tok_per_s" in bench_json:
                prof["measured_decode_tok_per_s"] = round(
                    float(bench_json["decode_tok_per_s"]), 2
                )
            if bench_json.get("mean_tokens_per_round") is not None:
                tau = float(bench_json["mean_tokens_per_round"])
                prof["mean_tokens_per_round"] = round(tau, 3)
                if prof["total_ms_per_round"] > 0:
                    prof["tok_per_s_m1"] = round(
                        1000.0 * tau / prof["total_ms_per_round"], 2
                    )
        return prof


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
    lines = ["Diagnostic only — not a performance gate. "
             f"Modes: oracle={oracle.get('measurement_mode', 'unverified summary')}, "
             f"w34={w34.get('measurement_mode', 'unverified summary')}"]
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
    o_tps = oracle.get("tok_per_s_m1", 0.0)
    w_tps = w34.get("measured_decode_tok_per_s", w34.get("tok_per_s_m1", 0.0))
    tps_sp = f"{w_tps / o_tps:.2f}x" if o_tps > 1e-6 and w_tps > 1e-6 else "n/a"
    lines.append(f"{'TOK/S (effective)':<18} | {o_tps:>16.2f} | {w_tps:>16.2f} | {w_tps - o_tps:>+13.2f} | {tps_sp:>9}")
    if "mean_tokens_per_round" in w34:
        lines.append(
            f"rounds={w34.get('rounds')}  mean_tokens_per_round(tau)={w34['mean_tokens_per_round']:.2f}x"
        )
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--oracle", required=False, help="Oracle nsys .sqlite or .json profile")
    ap.add_argument("--preset", default=None, help="Preset key inside oracle_baselines.json")
    ap.add_argument("--oracle-label", default=None, help="Oracle column header")
    ap.add_argument("--w34", default=None, help="warp-34 nsys .sqlite or .json profile")
    ap.add_argument("--live-cmd", default=None, help="Run command under nsys profile and parse SQLite")
    ap.add_argument("--w34-label", default="warp-34", help="warp-34 column header")
    ap.add_argument("--rounds", type=int, default=128, help="Nominal rounds in trace")
    ap.add_argument("--device", type=int, default=0, help="CUDA deviceId in nsys trace")
    ap.add_argument("--json-out", default=None, help="Optional JSON report path")
    ap.add_argument("--mode", choices=["legacy-heuristic", "strict"], default="legacy-heuristic")
    ap.add_argument("--window-start-ns", type=int)
    ap.add_argument("--window-end-ns", type=int)
    ap.add_argument("--metadata", type=Path, help='JSON with exact {"rounds": N}')
    ap.add_argument("--oracle-nvtx-correlated", action="store_true")
    ap.add_argument("--nvtx-correlated", action="store_true", help="Exact launch attribution, all devices; requires bench telemetry")
    ap.add_argument("--nvtx-candidates", type=Path, help="Report CPU intervals only and exit")
    ap.add_argument("--retain-trace", nargs="?", const="w34-profile-trace")
    args = ap.parse_args()
    if args.oracle_nvtx_correlated:
        if not args.oracle or not args.metadata or args.w34 or args.live_cmd:
            ap.error('--oracle-nvtx-correlated requires --oracle SQLite and --metadata runtime JSON')
        prof = parse_oracle_nvtx_correlated(Path(args.oracle), json.loads(args.metadata.read_text()))
        text = json.dumps(prof, indent=2)
        print(text)
        if args.json_out:
            Path(args.json_out).write_text(text)
        return
    if args.nvtx_candidates:
        print(json.dumps(nvtx_candidates(args.nvtx_candidates), indent=2))
        return
    if args.nvtx_correlated:
        if args.mode != "legacy-heuristic" or args.oracle or bool(args.w34) == bool(args.live_cmd):
            ap.error("--nvtx-correlated requires exactly one of --w34/--live-cmd; no donor comparison")
        if args.live_cmd:
            prof = run_live_nsys(args.live_cmd, args.rounds, args.device, args.retain_trace, True)
        else:
            if not args.metadata:
                ap.error("--nvtx-correlated --w34 requires --metadata bench JSON")
            prof = parse_nvtx_correlated(Path(args.w34), json.loads(args.metadata.read_text()))
        text = json.dumps(prof, indent=2)
        print(text)
        if args.json_out:
            Path(args.json_out).write_text(text)
        return
    if args.mode == "strict":
        if not args.w34 or not args.metadata or args.live_cmd or args.oracle:
            ap.error("strict mode requires --w34 SQLite and --metadata; no live/comparison mode")
        prof = parse_explicit_window(Path(args.w34), args.window_start_ns,
                                     args.window_end_ns, args.metadata, args.device)
        text = json.dumps(prof, indent=2)
        print(text)
        if args.json_out:
            Path(args.json_out).write_text(text)
        return
    if not args.oracle:
        ap.error("legacy diagnostic mode requires --oracle")

    oracle_prof = load_profile(Path(args.oracle), args.rounds, args.device, args.preset)
    if args.w34:
        w34_prof = load_profile(Path(args.w34), args.rounds, args.device, args.preset)
    elif args.live_cmd:
        w34_prof = run_live_nsys(args.live_cmd, args.rounds, args.device, args.retain_trace)
    else:
        w34_prof = oracle_prof

    oracle_label = args.oracle_label
    if not oracle_label:
        oracle_label = "SGLang v0.5.19" if (args.preset and "sglang" in args.preset) else "vLLM 0.31.0"

    table = render_scoreboard(oracle_label, oracle_prof, args.w34_label, w34_prof)
    print(table)
    if args.json_out:
        Path(args.json_out).write_text(json.dumps({"oracle": oracle_prof, "w34": w34_prof}, indent=2))


if __name__ == "__main__":
    main()
