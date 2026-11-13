#!/usr/bin/env python3
"""Static CUDA fatbinary resource & stack-spill gate for `w34-kernels`.

Inspects `libw34_kernels.a` via `cuobjdump --dump-resource-usage` across all
compiled `sm_86` and `sm_89` kernel specializations (~1,070 entries) and fails
fast if any kernel:
  1. Fails to compile for both `sm_86` and `sm_89` (unless `W34_ARCH` overrides).
  2. Allocates `.local` memory (`LOCAL > 0`).
  3. Exceeds the 48 KiB opt-in-free static shared-memory ceiling (`SHARED > 49152`).
  4. Allocates non-zero `STACK` outside the explicit call-frame / vendor allowlist
     (preventing accidental `.local` stack-spill regressions across collectives,
     MoE routers/GEMVs/CPU-bridge, norms, LoRA, GDN, KV/TQ4, Paged FA2, DFlash2,
     PLE, HC, and rejection kernels).
  5. Exceeds per-kernel `REG` occupancy ceilings on launch-bounded hot kernels
     (e.g. 1024-thread `sampler_step_f16_kernel` <= 64 regs, 512-thread fused
     collectives & LoRA <= 128 regs, warp routers <= 64 regs).
"""

from __future__ import annotations

import argparse
import dataclasses
import os
import pathlib
import re
import shutil
import subprocess
import sys
import time

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
MAX_STATIC_SHARED_BYTES = 49152
MIN_KERNELS_PER_ARCH = 400

# Explicit STACK ceilings (bytes) ONLY for kernels with unavoidable libdevice /
# `__noinline__` call frames or upstream vendor templates. Every other kernel in
# `w34-kernels` must compile with `STACK:0`.
STACK_ALLOWLIST: tuple[tuple[str, int, str], ...] = (
    ("_ZN6marlin6Marlin", 64, "Marlin template specialization"),
    ("_ZN5flash16flash_fwd_kernel", 192, "FlashAttention-2 vision fwd template"),
    (
        "fused_qk_rmsnorm_rope_gate_f16_kernel",
        32,
        "libdevice __nv_powf/__nv_sinf/__nv_cosf call frame",
    ),
    (
        "qsa_indexer_update_comp_k_f16_kernel",
        32,
        "libdevice __nv_powf/__nv_sinf/__nv_cosf call frame",
    ),
    (
        "qsa_indexer_norm_rope_f16_kernel",
        32,
        "libdevice __nv_powf/__nv_sinf/__nv_cosf call frame",
    ),
    (
        "sampler_step_f16_kernel",
        32,
        "__noinline__ radix helper call frame (0 register spill)",
    ),
    (
        "paged_attention_kernelI13__nv_bfloat16E",
        128,
        "cold scalar BF16 paged_attention fallback",
    ),
)

# Per-kernel register ceilings protecting launch bounds & occupancy on hot paths.
REG_CEILINGS: tuple[tuple[str, int, str], ...] = (
    ("sampler_step_f16_kernel", 64, "1024-thread launch bound (64 * 1024 = 65536 regs)"),
    ("tp2_all_reduce_hc_combine_tap_f16_kernel", 128, "512-thread launch bound"),
    ("tp4_all_reduce_hc_combine_tap_f16_kernel", 128, "512-thread launch bound"),
    (
        "tp2_all_reduce_residual_gemma_rms_norm_f16_kernelILi512",
        128,
        "512-thread fused TP=2 norm launch bound",
    ),
    (
        "tp4_all_reduce_residual_gemma_rms_norm_f16_kernelILi512",
        128,
        "512-thread fused TP=4 norm launch bound",
    ),
    ("lora_fused_shrink_expand_512t_f16_kernel", 128, "512-thread fused LoRA launch bound (128 * 512 = 65536 regs)"),
    ("moe_topk_router_256_k8_f16_kernel", 64, "warp-specialized 256/k8 router"),
    ("moe_topk_router_512_k10_f16_kernel", 64, "warp-specialized 512/k10 router"),
    ("moe_sparse_vec4_kernel", 128, "sparse MoE vec4 occupancy ceiling"),
    ("gdn_wy_chunk64_prefill_f16_kernel", 224, "256-thread WY64 chunked prefill"),
)


@dataclasses.dataclass(frozen=True)
class KernelResource:
    arch: str
    symbol: str
    reg: int
    stack: int
    shared: int
    local: int


_ARCH_RE = re.compile(r"arch\s*=\s*(sm_\d+)")
_FN_RE = re.compile(r"^\s*Function\s+(\S+):\s*$")
_RES_RE = re.compile(r"REG:(\d+)\s+STACK:(\d+)\s+SHARED:(\d+)\s+LOCAL:(\d+)")


def find_cuobjdump() -> str:
    env_bin = os.environ.get("CUOBJDUMP")
    if env_bin and pathlib.Path(env_bin).is_file():
        return env_bin
    found = shutil.which("cuobjdump")
    if found:
        return found
    opt_cuda = pathlib.Path("/opt/cuda/bin/cuobjdump")
    if opt_cuda.is_file():
        return str(opt_cuda)
    raise FileNotFoundError("cuobjdump not found on PATH or /opt/cuda/bin/cuobjdump")


def find_or_build_archive(repo_root: pathlib.Path, profile: str = "release") -> pathlib.Path:
    build_root = repo_root / "target" / profile / "build"
    env = os.environ.copy()
    if "/opt/cuda/bin" not in env.get("PATH", ""):
        env["PATH"] = f"/opt/cuda/bin:{env.get('PATH', '')}"
    cmd = ["cargo", "build", "-p", "w34-kernels", "--features", "cuda", "--quiet"]
    if profile == "release":
        cmd.append("--release")
    subprocess.run(cmd, cwd=repo_root, env=env, check=True)
    candidates = sorted(
        build_root.glob("w34-kernels-*/out/libw34_kernels.a"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if not candidates:
        raise FileNotFoundError(f"libw34_kernels.a not found under {build_root}")
    return candidates[0]


def parse_cuobjdump_resources(text: str) -> list[KernelResource]:
    arch = "unknown"
    cur_fn: str | None = None
    out: list[KernelResource] = []
    for line in text.splitlines():
        m_arch = _ARCH_RE.search(line)
        if m_arch:
            arch = m_arch.group(1)
            cur_fn = None
            continue
        m_fn = _FN_RE.match(line)
        if m_fn:
            cur_fn = m_fn.group(1)
            continue
        m_res = _RES_RE.search(line)
        if m_res and cur_fn is not None:
            reg, stack, shared, local = map(int, m_res.groups())
            out.append(
                KernelResource(
                    arch=arch,
                    symbol=cur_fn,
                    reg=reg,
                    stack=stack,
                    shared=shared,
                    local=local,
                )
            )
            cur_fn = None
    return out


def expected_arches_from_env(w34_arch_env: str | None) -> tuple[str, ...]:
    if not w34_arch_env or not w34_arch_env.strip():
        return ("sm_86", "sm_89")
    arches: list[str] = []
    for tok in re.split(r"[,\s;]+", w34_arch_env.strip()):
        if not tok:
            continue
        norm = tok if tok.startswith("sm_") else f"sm_{tok.removeprefix('compute_')}"
        if norm not in arches:
            arches.append(norm)
    return tuple(arches)


def audit_kernel_resources(
    entries: list[KernelResource],
    expected_arches: tuple[str, ...] = ("sm_86", "sm_89"),
    min_kernels_per_arch: int = MIN_KERNELS_PER_ARCH,
) -> list[str]:
    failures: list[str] = []
    if not entries:
        return ["no CUDA kernel resource entries parsed from cuobjdump output"]

    by_arch: dict[str, int] = {}
    for entry in entries:
        by_arch[entry.arch] = by_arch.get(entry.arch, 0) + 1

    for arch in expected_arches:
        cnt = by_arch.get(arch, 0)
        if cnt < min_kernels_per_arch:
            failures.append(
                f"expected >= {min_kernels_per_arch} kernels for {arch}, found {cnt} (present={sorted(by_arch)})"
            )

    if len(expected_arches) > 1:
        counts = {arch: by_arch.get(arch, 0) for arch in expected_arches}
        if len(set(counts.values())) > 1:
            failures.append(
                f"kernel count mismatch across expected architectures: {counts} (stale object or single-arch TU?)"
            )

    for entry in entries:
        if entry.local > 0:
            failures.append(
                f"{entry.arch} {entry.symbol}: LOCAL={entry.local} > 0 (.local memory allocation)"
            )
        if entry.shared > MAX_STATIC_SHARED_BYTES:
            failures.append(
                f"{entry.arch} {entry.symbol}: static SHARED={entry.shared} > {MAX_STATIC_SHARED_BYTES}"
            )

        max_allowed_stack = 0
        for substr, cap, _reason in STACK_ALLOWLIST:
            if substr in entry.symbol:
                max_allowed_stack = cap
                break
        if entry.stack > max_allowed_stack:
            failures.append(
                f"{entry.arch} {entry.symbol}: STACK={entry.stack} exceeds ceiling {max_allowed_stack} B"
            )

        for substr, reg_cap, reason in REG_CEILINGS:
            if substr in entry.symbol and entry.reg > reg_cap:
                failures.append(
                    f"{entry.arch} {entry.symbol}: REG={entry.reg} exceeds ceiling {reg_cap} ({reason})"
                )

    return failures


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--archive",
        type=pathlib.Path,
        default=None,
        help="Path to libw34_kernels.a (auto-discovered under target/release/build if omitted)",
    )
    args = parser.parse_args(argv)

    t0 = time.monotonic()
    cuobjdump = find_cuobjdump()
    archive = args.archive or find_or_build_archive(REPO_ROOT)
    proc = subprocess.run(
        [cuobjdump, "--dump-resource-usage", str(archive)],
        capture_output=True,
        text=True,
        check=True,
    )
    entries = parse_cuobjdump_resources(proc.stdout)
    expected_arches = expected_arches_from_env(os.environ.get("W34_ARCH"))
    failures = audit_kernel_resources(entries, expected_arches=expected_arches)
    elapsed_ms = (time.monotonic() - t0) * 1000.0

    by_arch: dict[str, int] = {}
    zero_stack = 0
    for e in entries:
        by_arch[e.arch] = by_arch.get(e.arch, 0) + 1
        if e.stack == 0:
            zero_stack += 1

    if failures:
        print(
            f"[cuda-bin-check] FAILED ({len(failures)} violation(s) across {len(entries)} kernel entries in {archive}):",
            file=sys.stderr,
        )
        for msg in failures[:30]:
            print(f"  - {msg}", file=sys.stderr)
        if len(failures) > 30:
            print(f"  ... and {len(failures) - 30} more", file=sys.stderr)
        return 1

    arch_summary = ", ".join(f"{k}={v}" for k, v in sorted(by_arch.items()))
    print(
        f"[cuda-bin-check] ok: {len(entries)} kernel entries ({arch_summary}), "
        f"zero_stack={zero_stack}/{len(entries)}, local=0, static_shared<=48KiB ({elapsed_ms:.0f} ms)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
