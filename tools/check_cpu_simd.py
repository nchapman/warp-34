#!/usr/bin/env python3
"""Static assembly & llvm-mca audit for w34-host CPU SIMD kernels.

Compiles `crates/w34-host/src/cuda/half.rs` and `crates/w34-host/src/moe_cpu_bridge/eval.rs`
with `rustc -C opt-level=3 --emit=asm` (~300 ms) and verifies:
  1. Zero unexpected `callq` instructions inside `#[target_feature]` kernels
     (catches intrinsics whose feature gate was omitted, e.g. `avx512dq`, or
     helpers that failed to inline across target-feature boundaries).
  2. Zero vector stack spills (`vmov(ups|aps|dqu|dqa).*(%rsp)`) in `hadamard64_f32_avx2`,
     `prepare_x_perm_*`, `half_simd::*`, and all `AVX-512` `row_tile` specializations,
     plus zero `(%rsp)` FMA operands inside `row_tile_avx2` `R*T <= 4` inner nibble loops.
  3. Zero scalar fallback instructions (`vaddss`/`vsubss`) in `hadamard64_f32_avx2`
     and zero calls to `*_portable` / `row_scalar` from `expert_avx2` or `expert_avx512`.
  4. When `llvm-mca` is available on PATH, simulates `znver2`, `znver3`, `znver4`,
     `alderlake`, and `sapphirerapids` pipelines and enforces `Block RThroughput` ceilings.
"""

from __future__ import annotations

import os
import pathlib
import platform
import re
import shutil
import subprocess
import sys
import tempfile
import time

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
HALF_RS = REPO_ROOT / "crates/w34-host/src/cuda/half.rs"
EVAL_RS = REPO_ROOT / "crates/w34-host/src/moe_cpu_bridge/eval.rs"

PROBE_SOURCE = """
#![allow(dead_code, private_interfaces)]
#[path = "{half_rs}"]
pub mod cuda;
#[path = "{eval_rs}"]
pub mod eval;

#[target_feature(enable = "avx2,fma")]
#[unsafe(no_mangle)]
pub unsafe fn probe_hadamard64_avx2(s: &mut [f32]) {{
    unsafe {{ eval::simd::hadamard64_f32_avx2(s) }}
}}

#[target_feature(enable = "avx2,fma,f16c")]
#[unsafe(no_mangle)]
pub unsafe fn probe_prepare_x_perm_avx2(x: &[f32], xp: &mut [f32], xg: &mut [f32]) {{
    unsafe {{ eval::simd::prepare_x_perm_avx2(x, xp, xg) }}
}}

#[target_feature(enable = "avx2,fma,f16c")]
#[unsafe(no_mangle)]
pub unsafe fn probe_prepare_x_perm_avx2_g64(x: &[f32], xp: &mut [f32], xg: &mut [f32]) {{
    unsafe {{ eval::simd::prepare_x_perm_avx2_g64(x, xp, xg) }}
}}

#[target_feature(enable = "avx2,fma,f16c")]
#[unsafe(no_mangle)]
pub unsafe fn probe_prepare_x_perm_f16c(x: &[u16], xp: &mut [f32], xg: &mut [f32]) {{
    unsafe {{ eval::simd::prepare_x_perm_f16c(x, xp, xg) }}
}}

#[target_feature(enable = "avx2,fma,f16c")]
#[unsafe(no_mangle)]
pub unsafe fn probe_prepare_x_perm_f16c_g64(x: &[u16], xp: &mut [f32], xg: &mut [f32]) {{
    unsafe {{ eval::simd::prepare_x_perm_f16c_g64(x, xp, xg) }}
}}

#[target_feature(enable = "avx2,fma,f16c")]
#[unsafe(no_mangle)]
pub unsafe fn probe_row_tile_avx2_r2_t1_g64(
    w: [&[u32]; 2], s: [&[u16]; 2], xp: &[f32], xg: &[f32], groups: usize,
) -> [[f32; 2]; 1] {{
    unsafe {{ eval::simd::row_tile_avx2::<2, 1, true, 4>(w, s, xp, xg, groups) }}
}}

#[target_feature(enable = "avx2,fma,f16c")]
#[unsafe(no_mangle)]
pub unsafe fn probe_row_tile_avx2_r2_t2_g64(
    w: [&[u32]; 2], s: [&[u16]; 2], xp: &[f32], xg: &[f32], groups: usize,
) -> [[f32; 2]; 2] {{
    unsafe {{ eval::simd::row_tile_avx2::<2, 2, true, 4>(w, s, xp, xg, groups) }}
}}

#[target_feature(enable = "avx2,fma,f16c")]
#[unsafe(no_mangle)]
pub unsafe fn probe_row_tile_avx2_r2_t4_g64(
    w: [&[u32]; 2], s: [&[u16]; 2], xp: &[f32], xg: &[f32], groups: usize,
) -> [[f32; 2]; 4] {{
    unsafe {{ eval::simd::row_tile_avx2::<2, 4, true, 4>(w, s, xp, xg, groups) }}
}}

#[target_feature(enable = "avx2,fma,f16c")]
#[unsafe(no_mangle)]
pub unsafe fn probe_row_tile_avx2_r2_t1_g64_u2(
    w: [&[u32]; 2], s: [&[u16]; 2], xp: &[f32], xg: &[f32], groups: usize,
) -> [[f32; 2]; 1] {{
    unsafe {{ eval::simd::row_tile_avx2::<2, 1, true, 2>(w, s, xp, xg, groups) }}
}}

#[target_feature(enable = "avx2,fma,f16c,avx512f,avx512vl")]
#[unsafe(no_mangle)]
pub unsafe fn probe_row_tile_avx512_r4_t1_g64(
    w: [&[u32]; 4], s: [&[u16]; 4], xp: &[f32], xg: &[f32], groups: usize,
) -> [[f32; 4]; 1] {{
    unsafe {{ eval::simd::row_tile::<4, 1, true, 4>(w, s, xp, xg, groups) }}
}}

#[target_feature(enable = "avx2,fma,f16c,avx512f,avx512vl")]
#[unsafe(no_mangle)]
pub unsafe fn probe_row_tile_avx512_r4_t2_g64(
    w: [&[u32]; 4], s: [&[u16]; 4], xp: &[f32], xg: &[f32], groups: usize,
) -> [[f32; 4]; 2] {{
    unsafe {{ eval::simd::row_tile::<4, 2, true, 4>(w, s, xp, xg, groups) }}
}}

#[target_feature(enable = "avx2,fma,f16c,avx512f,avx512vl")]
#[unsafe(no_mangle)]
pub unsafe fn probe_row_tile_avx512_r2_t4_g64(
    w: [&[u32]; 2], s: [&[u16]; 2], xp: &[f32], xg: &[f32], groups: usize,
) -> [[f32; 2]; 4] {{
    unsafe {{ eval::simd::row_tile::<2, 4, true, 4>(w, s, xp, xg, groups) }}
}}

#[target_feature(enable = "avx2,fma,f16c,avx512f,avx512vl")]
#[unsafe(no_mangle)]
pub unsafe fn probe_row_tile_avx512_r4_t1_g64_u2(
    w: [&[u32]; 4], s: [&[u16]; 4], xp: &[f32], xg: &[f32], groups: usize,
) -> [[f32; 4]; 1] {{
    unsafe {{ eval::simd::row_tile::<4, 1, true, 2>(w, s, xp, xg, groups) }}
}}

#[target_feature(enable = "avx2,f16c")]
#[unsafe(no_mangle)]
pub unsafe fn probe_f16_to_f32_f16c(src: &[u16], dst: &mut [f32]) {{
    unsafe {{ cuda::half_simd::f16_to_f32_f16c(src, dst) }}
}}

#[target_feature(enable = "avx2,f16c")]
#[unsafe(no_mangle)]
pub unsafe fn probe_f32_to_f16_f16c(src: &[f32], dst: &mut [u16]) {{
    unsafe {{ cuda::half_simd::f32_to_f16_f16c(src, dst) }}
}}

#[target_feature(enable = "avx2,f16c")]
#[unsafe(no_mangle)]
pub unsafe fn probe_bf16_to_f16_f16c(words: &mut [u16]) {{
    unsafe {{ cuda::half_simd::bf16_to_f16_f16c_inplace(words) }}
}}

#[target_feature(enable = "avx2,fma,f16c")]
#[unsafe(no_mangle)]
pub unsafe fn probe_expert_avx2(
    d: eval::ExpertLayout<'_>,
    range: std::ops::Range<usize>,
    xp: &[f32],
    xg: &[f32],
    wt: &[f32],
    out: &mut [f32],
    scratch: &mut [f32],
) {{
    unsafe {{ eval::simd::expert_avx2(d, range, [xp, xg], wt, out, scratch) }}
}}

#[target_feature(enable = "avx2,fma,f16c,avx512f,avx512vl")]
#[unsafe(no_mangle)]
pub unsafe fn probe_expert_avx512(
    d: eval::ExpertLayout<'_>,
    range: std::ops::Range<usize>,
    xp: &[f32],
    xg: &[f32],
    wt: &[f32],
    out: &mut [f32],
    scratch: &mut [f32],
) {{
    unsafe {{ eval::simd::expert_avx512(d, range, [xp, xg], wt, out, scratch, true) }}
}}
"""


def parse_insns(fn_body: str) -> list[str]:
    return [
        line.strip()
        for line in fn_body.splitlines()
        if line.startswith("\t") and not line.strip().startswith(".")
    ]


def extract_fn(asm_text: str, symbol: str) -> str:
    m = re.search(rf"^{re.escape(symbol)}:\n(.*?)\.Lfunc_end", asm_text, re.M | re.S)
    if not m:
        raise RuntimeError(f"missing symbol `{symbol}` in emitted assembly")
    body = m.group(1)
    insns = parse_insns(body)
    if len(insns) <= 20:
        targets = [
            i.split()[1]
            for i in insns
            if (i.startswith("jmp\t") or i.startswith("callq\t"))
            and i.split()[1].startswith("_R")
        ]
        if len(targets) == 1:
            return extract_fn(asm_text, targets[0])
    return body


def parse_mca_metric(output: str, label: str) -> float | None:
    for line in output.splitlines()[:20]:
        if line.startswith(label):
            parts = line.split(":", 1)
            if len(parts) == 2:
                try:
                    return float(parts[1].strip())
                except ValueError:
                    return None
    return None


def main() -> int:
    if platform.machine() not in ("x86_64", "AMD64"):
        print(f"check_cpu_simd: skipping on non-x86_64 host ({platform.machine()})")
        return 0

    env = {**os.environ, "LANG": "C", "LC_ALL": "C"}
    t0 = time.perf_counter()
    with tempfile.TemporaryDirectory() as tmp:
        src = pathlib.Path(tmp) / "simd_probe.rs"
        asm = pathlib.Path(tmp) / "simd_probe.s"
        src.write_text(PROBE_SOURCE.format(half_rs=HALF_RS, eval_rs=EVAL_RS))
        try:
            subprocess.run(
                [
                    "rustc",
                    "--edition=2024",
                    "--crate-type=lib",
                    "-C",
                    "opt-level=3",
                    "--emit=asm",
                    str(src),
                    "-o",
                    str(asm),
                ],
                env=env,
                timeout=60,
                check=True,
                capture_output=True,
                text=True,
            )
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
            print(f"check_cpu_simd: rustc failed: {exc}", file=sys.stderr)
            if getattr(exc, "stderr", None):
                print(exc.stderr, file=sys.stderr)
            return 1
        asm_text = asm.read_text()

    failures: list[str] = []

    zero_spill_kernels = {
        "probe_hadamard64_avx2": {
            "max_insns": 140,
            "required": ["vfmadd", "vaddps", "vsubps", "vshufps", "vpermpd"],
            "forbidden": ["vaddss", "vsubss", "addss", "subss"],
        },
        "probe_prepare_x_perm_avx2": {
            "max_insns": 140,
            "required": ["vunpcklps", "vunpckhps", "vunpcklpd", "vperm2f128"],
            "forbidden": [],
        },
        "probe_prepare_x_perm_avx2_g64": {
            "max_insns": 240,
            "required": ["vunpcklps", "vunpckhps", "vunpcklpd", "vperm2f128"],
            "forbidden": [],
        },
        "probe_prepare_x_perm_f16c": {
            "max_insns": 260,
            "required": ["vcvtph2ps", "vunpcklps", "vunpckhps", "vunpcklpd", "vperm2f128"],
            "forbidden": [],
        },
        "probe_prepare_x_perm_f16c_g64": {
            "max_insns": 260,
            "required": ["vcvtph2ps", "vunpcklps", "vunpckhps", "vunpcklpd", "vperm2f128"],
            "forbidden": [],
        },
        "probe_f16_to_f32_f16c": {
            "max_insns": 130,
            "required": ["vcvtph2ps"],
            "forbidden": [],
        },
        "probe_f32_to_f16_f16c": {
            "max_insns": 90,
            "required": ["vcvtps2ph"],
            "forbidden": [],
        },
        "probe_bf16_to_f16_f16c": {
            "max_insns": 90,
            "required": ["vpmovzxwd", "vpslld", "vcvtps2ph"],
            "forbidden": [],
        },
        "probe_row_tile_avx512_r4_t1_g64": {
            "max_insns": 320,
            "required": ["vfmadd", "vcvtdq2ps", "vinsertf64x4"],
            "forbidden": [],
        },
        "probe_row_tile_avx512_r4_t2_g64": {
            "max_insns": 440,
            "required": ["vfmadd", "vcvtdq2ps", "vinsertf64x4"],
            "forbidden": [],
        },
        "probe_row_tile_avx512_r2_t4_g64": {
            "max_insns": 380,
            "required": ["vfmadd", "vcvtdq2ps", "vinsertf64x4"],
            "forbidden": [],
        },
        "probe_row_tile_avx512_r4_t1_g64_u2": {
            "max_insns": 320,
            "required": ["vpmovzxwd", "vfmadd", "vcvtdq2ps", "vinsertf64x4"],
            "forbidden": [],
        },
    }

    for sym, spec in zero_spill_kernels.items():
        body = extract_fn(asm_text, sym)
        insns = parse_insns(body)
        calls = [i for i in insns if i.startswith("call")]
        spills = [i for i in insns if re.search(r"vmov(ups|aps|dqu|dqa).*\(%rsp\)", i)]
        if calls:
            failures.append(f"{sym}: unexpected call(s): {calls}")
        if spills:
            failures.append(f"{sym}: {len(spills)} unexpected vector stack spill(s)")
        if len(insns) > spec["max_insns"]:
            failures.append(
                f"{sym}: instruction count {len(insns)} exceeds ceiling {spec['max_insns']}"
            )
        for req in spec["required"]:
            if not any(req in i for i in insns):
                failures.append(f"{sym}: missing expected instruction `{req}`")
        for bad in spec["forbidden"]:
            if any(bad in i for i in insns):
                failures.append(f"{sym}: found forbidden scalar instruction `{bad}`")

    for sym, max_rsp_fmas in [
        ("probe_row_tile_avx2_r2_t1_g64", 0),
        ("probe_row_tile_avx2_r2_t1_g64_u2", 0),
        ("probe_row_tile_avx2_r2_t2_g64", 0),
        ("probe_row_tile_avx2_r2_t4_g64", 12),
    ]:
        body = extract_fn(asm_text, sym)
        insns = parse_insns(body)
        calls = [i for i in insns if i.startswith("call")]
        rsp_fmas = [i for i in insns if "vfmadd" in i and "(%rsp)" in i]
        if calls:
            failures.append(f"{sym}: unexpected call(s): {calls}")
        if len(rsp_fmas) > max_rsp_fmas:
            failures.append(
                f"{sym}: {len(rsp_fmas)} memory-(%rsp) vfmadd instructions > ceiling {max_rsp_fmas}"
            )

    for sym in [
        name
        for name in re.findall(r"^([A-Za-z0-9_$\.]+):", asm_text, re.M)
        if ("expert_avx2" in name or "expert_avx512" in name)
        and not name.startswith("probe_")
    ]:
        body = extract_fn(asm_text, sym)
        insns = parse_insns(body)
        portable_calls = [
            i
            for i in insns
            if i.startswith("call") and ("portable" in i or "row_scalar" in i)
        ]
        if portable_calls:
            failures.append(f"{sym}: calls portable fallback: {portable_calls}")

    mca_bin = shutil.which("llvm-mca")
    mca_results: list[str] = []
    if mca_bin:
        mca_specs = [
            (
                "probe_hadamard64_avx2",
                {"znver2": 32.0, "znver3": 24.0, "znver4": 22.0, "alderlake": 24.0},
            ),
            (
                "probe_prepare_x_perm_f16c_g64",
                {"znver2": 66.0, "znver3": 62.0, "znver4": 62.0, "alderlake": 62.0},
            ),
            (
                "probe_row_tile_avx2_r2_t2_g64",
                {"znver2": 95.0, "znver3": 64.0, "znver4": 64.0, "alderlake": 78.0},
            ),
            (
                "probe_row_tile_avx512_r4_t2_g64",
                {"znver4": 105.0, "sapphirerapids": 150.0},
            ),
        ]
        for sym, cpu_ceilings in mca_specs:
            raw_body = extract_fn(asm_text, sym)
            clean_body = (
                "\n".join(
                    line
                    for line in raw_body.splitlines()
                    if not line.strip().startswith(".cfi")
                )
                + "\n"
            )
            for cpu, max_rthroughput in cpu_ceilings.items():
                try:
                    res = subprocess.run(
                        [mca_bin, f"-mcpu={cpu}", "-iterations=10"],
                        input=clean_body,
                        env=env,
                        timeout=30,
                        text=True,
                        capture_output=True,
                        check=True,
                    )
                except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
                    failures.append(f"{sym}[{cpu}]: llvm-mca failed: {exc}")
                    continue
                rthroughput = parse_mca_metric(res.stdout, "Block RThroughput:")
                ipc = parse_mca_metric(res.stdout, "IPC:")
                if rthroughput is None or ipc is None:
                    failures.append(
                        f"{sym}[{cpu}]: failed to parse llvm-mca metrics from output"
                    )
                    continue
                mca_results.append(f"{sym}[{cpu}]: IPC={ipc:.2f} RThroughput={rthroughput:.1f}")
                if rthroughput > max_rthroughput:
                    failures.append(
                        f"{sym}[{cpu}]: llvm-mca Block RThroughput {rthroughput:.1f} > ceiling {max_rthroughput:.1f}"
                    )

    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    if failures:
        print(f"check_cpu_simd FAILED ({elapsed_ms:.0f} ms):", file=sys.stderr)
        for f in failures:
            print(f"  - {f}", file=sys.stderr)
        return 1

    print(
        f"check_cpu_simd: PASS in {elapsed_ms:.0f} ms "
        f"({len(zero_spill_kernels) + 3} kernels audited, "
        f"{len(mca_results)} llvm-mca targets verified)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
