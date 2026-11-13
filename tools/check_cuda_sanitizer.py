#!/usr/bin/env python3
"""Targeted CUDA `compute-sanitizer` regression gate for `w34-kernels`.

Runs `/opt/cuda/bin/compute-sanitizer --error-exitcode=1` (`memcheck`,
`racecheck`, and `synccheck`) across a fast curated matrix covering all six
`w34-kernels` CUDA subsystems:
  1. Quantized Linear (`Marlin` W4A16/W2A16/BF16, `small_n_gemm`, fused `LoRA`)
  2. KV Cache, `TQ4`/`Q8_0`, Paged Attention, Paged FA2, & QSA
  3. Gated DeltaNet (`GDN` decode, `WY` / `WY64` chunked prefill, `conv1d`)
  4. Mixture-of-Experts (grouped MMA, sparse GEMV, `down_accum`, 256/512 routers,
     GPU LRU & CPU bridge)
  5. Norms, RoPE, Activations, FWHT (`hadamard`), HyperConnections (`HC`), & `PLE`
  6. Collectives (`TP=2` / `TP=4`), `DFlash2`, Rejection, GPU Sampler, & Vision

Single-GPU cases execute concurrently across available GPUs (each pinned via
`CUDA_VISIBLE_DEVICES` to a single card so CUDA context init stays minimal),
followed by serialized multi-GPU (`TP=2`) collective cases.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import dataclasses
import json
import os
import pathlib
import queue
import shutil
import signal
import subprocess
import sys
import time

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
ALL_TOOLS: tuple[str, ...] = ("memcheck", "racecheck", "synccheck")


@dataclasses.dataclass(frozen=True)
class SanitizerCase:
    subsystem: str
    target: str  # "lib" for `w34-host --lib`, otherwise `--test <target>`
    test_name: str
    min_gpus: int = 1
    tools: tuple[str, ...] = ALL_TOOLS


SANITIZER_MATRIX: tuple[SanitizerCase, ...] = (
    # 1. Quantized Linear (Marlin W4A16/W2A16/BF16, small_n_gemm, fused LoRA)
    SanitizerCase(
        subsystem="linear",
        target="lib",
        test_name="model::weights::hgptq::tests::marlin_w2a16_and_hadamard_w2a16_match_dilated_w4a16_zero_ulp",
    ),
    SanitizerCase(
        subsystem="linear",
        target="lib",
        test_name="dflash::tests::bf16_marlin_u4b8_and_hadamard_match_reference",
    ),
    SanitizerCase(
        subsystem="linear",
        target="lib",
        test_name="lora::tests::test_cuda_lora_shrink_expand_and_hadamard_fusion",
    ),
    SanitizerCase(
        subsystem="linear",
        target="moe_sparse",
        test_name="dual_small_n_gemm_and_warp256_router_match_references",
    ),
    # 2. KV Cache, TQ4/Q8_0, Paged Attention, Paged FA2, & QSA
    SanitizerCase(
        subsystem="kv_attn",
        target="paged_fa2",
        test_name="tight_two_page_partial_tile_does_not_read_past_block_table",
    ),
    SanitizerCase(
        subsystem="kv_attn",
        target="paged_fa2",
        test_name="kv_write_and_qsa_respect_valid_req_mask_and_reject_bad_gqa",
    ),
    SanitizerCase(
        subsystem="kv_attn",
        target="paged_fa2",
        test_name="multi_request_mixed_decode_and_prefill_fa2",
    ),
    SanitizerCase(
        subsystem="kv_attn",
        target="backbone_and_collectives",
        test_name="test_fragmented_paged_attention_and_tq4_strides",
    ),
    # 3. Gated DeltaNet (GDN decode, WY / WY64 prefill, conv1d)
    SanitizerCase(
        subsystem="gdn",
        target="collectives_and_gdn_wy",
        test_name="test_gdn_wy_extreme_decay_gates_produce_finite_outputs_and_states",
    ),
    SanitizerCase(
        subsystem="gdn",
        target="collectives_and_gdn_wy",
        test_name="test_gdn_wy_mixed_decode_prefill_batch_parity",
    ),
    SanitizerCase(
        subsystem="gdn",
        target="backbone_and_collectives",
        test_name="test_gdn_wy_split_and_fallback_prefill_parity",
    ),
    SanitizerCase(
        subsystem="gdn",
        target="backbone_and_collectives",
        test_name="test_gdn_conv_paired_prefill_matches_chunked_scalar",
    ),
    # 4. Mixture-of-Experts (grouped MMA, down_accum, 512/k10 router, GPU LRU & CPU bridge)
    SanitizerCase(
        subsystem="moe",
        target="moe_sparse",
        test_name="grouped_mma_moe_group_size_96_and_32_matches_cpu_reference",
    ),
    SanitizerCase(
        subsystem="moe",
        target="moe_sparse",
        test_name="moe_down_accum_subwarp_packing_matches_unaligned_fallback",
    ),
    SanitizerCase(
        subsystem="moe",
        target="moe_sparse",
        test_name="moe_router_512_k10_warp_matches_generic_and_unaligned_fallback",
    ),
    SanitizerCase(
        subsystem="moe",
        target="moe_sparse",
        test_name="moe_router_lru_and_bridge_validate_before_mutating_state",
    ),
    SanitizerCase(
        subsystem="moe",
        target="lib",
        test_name="moe_offload::tests::cuda_tests::tiered_moe_gpu_lru_cpu_hybrid_and_streamed_prefill_parity_tp1_and_ep2",
    ),
    # 5. Norms, RoPE, Activations, FWHT (Hadamard), HyperConnections (HC), & PLE
    SanitizerCase(
        subsystem="norms_hc_ple",
        target="gemma_residual_rounding",
        test_name="gemma_rms_norm_above_12288_and_silu_mul_grid_y_above_65535_and_alignment_guards",
    ),
    SanitizerCase(
        subsystem="norms_hc_ple",
        target="gemma_residual_rounding",
        test_name="vec8_rms_norm_and_silu_mul_match_vec2_fallback_zero_ulp",
    ),
    SanitizerCase(
        subsystem="norms_hc_ple",
        target="hgptq_hadamard",
        test_name="producer_fused_hadamard64_matches_standalone_0_ulp",
    ),
    SanitizerCase(
        subsystem="norms_hc_ple",
        target="target_rope_cache_rounding",
        test_name="target_rope_vectorized_fast_path_matches_unaligned_fallback",
    ),
    SanitizerCase(
        subsystem="norms_hc_ple",
        target="lib",
        test_name="ple::tests::gpu_tests::test_gpu_hc_ple_qsa_kernels_parity",
    ),
    SanitizerCase(
        subsystem="norms_hc_ple",
        target="lib",
        test_name="ple::tests::gpu_tests::test_gpu_hc_combine_tap_f16_parity",
    ),
    # 6. Collectives (TP=2 / TP=4), DFlash2, Rejection, Sampler, & Vision
    SanitizerCase(
        subsystem="sampler_dflash_vision",
        target="gpu_sampler",
        test_name="gpu_sampler_back_to_back_reductions_high_warp_only_valid_and_nan",
    ),
    SanitizerCase(
        subsystem="sampler_dflash_vision",
        target="gpu_sampler",
        test_name="test_fast_greedy_masked_matches_full_sampler_and_edges",
    ),
    SanitizerCase(
        subsystem="sampler_dflash_vision",
        target="gpu_sampler",
        test_name="split_k_greedy_top1_matches_single_cta",
    ),
    SanitizerCase(
        subsystem="sampler_dflash_vision",
        target="gpu_sampler",
        test_name="shard_fast_greedy_matches_full_sampler_across_edges",
    ),
    SanitizerCase(
        subsystem="sampler_dflash_vision",
        target="gpu_rejection_sampler",
        test_name="greedy_gpu_matches_host_at_each_rejection_and_masks_padding",
    ),
    SanitizerCase(
        subsystem="sampler_dflash_vision",
        target="gpu_rejection_sampler",
        test_name="stochastic_gpu_matches_host_on_residuals_bonus_and_invalid_requests",
    ),
    SanitizerCase(
        subsystem="sampler_dflash_vision",
        target="dflash_bf16_ops",
        test_name="bf16_norm_and_fused_residual_match_sglang",
    ),
    SanitizerCase(
        subsystem="sampler_dflash_vision",
        target="dflash_bf16_ops",
        test_name="bf16_qk_norm_rope_and_packed_copy_match_sglang",
    ),
    SanitizerCase(
        subsystem="sampler_dflash_vision",
        target="dflash_bf16_ops",
        test_name="bf16_silu_and_grouped_convolution_match_sglang",
    ),
    SanitizerCase(
        subsystem="sampler_dflash_vision",
        target="dflash_draft_kernels",
        test_name="draft_kv_preserves_bf16_range_through_attention",
    ),
    SanitizerCase(
        subsystem="sampler_dflash_vision",
        target="vision_rounding",
        test_name="vision_bias_and_rope_preserve_native_half_rounding",
    ),
    SanitizerCase(
        subsystem="sampler_dflash_vision",
        target="vision_rounding",
        test_name="vision_tanh_gelu_matches_native_activation",
    ),
    # Multi-GPU TP=2 / TP=4 cross-device pinned-sysmem / P2P collective kernels.
    # Note: `compute-sanitizer` serializes kernels sharing a physical GPU, so N-rank
    # cooperative spin-barriers require N distinct physical GPUs (`min_gpus=N`).
    SanitizerCase(
        subsystem="collectives_tp2",
        target="collectives_and_gdn_wy",
        test_name="test_tp2_alternating_cta_counts_preserve_staging_banks",
        min_gpus=2,
    ),
    SanitizerCase(
        subsystem="collectives_tp2",
        target="hgptq_hadamard",
        test_name="tp2_all_reduce_residual_gemma_rms_norm_hadamard64_matches_standalone_0_ulp",
        min_gpus=2,
    ),
    SanitizerCase(
        subsystem="collectives_tp2",
        target="collectives_and_gdn_wy",
        test_name="test_tp2_real_devices_all_gather_topk_and_hc_combine_2560",
        min_gpus=2,
    ),
)


def find_compute_sanitizer() -> str:
    env_bin = os.environ.get("COMPUTE_SANITIZER")
    if env_bin and pathlib.Path(env_bin).is_file():
        return env_bin
    found = shutil.which("compute-sanitizer")
    if found:
        return found
    opt_cuda = pathlib.Path("/opt/cuda/bin/compute-sanitizer")
    if opt_cuda.is_file():
        return str(opt_cuda)
    raise FileNotFoundError(
        "compute-sanitizer not found on PATH or /opt/cuda/bin/compute-sanitizer"
    )


def discover_visible_gpus(env_cvd: str | None = None) -> list[str]:
    cvd = env_cvd if env_cvd is not None else os.environ.get("CUDA_VISIBLE_DEVICES")
    if cvd is not None and cvd.strip():
        return [tok.strip() for tok in cvd.split(",") if tok.strip()]
    proc = subprocess.run(
        ["nvidia-smi", "--query-gpu=index", "--format=csv,noheader"],
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode != 0:
        return []
    return [line.strip() for line in proc.stdout.splitlines() if line.strip()]


def select_cases(
    cases: tuple[SanitizerCase, ...],
    selected_tools: tuple[str, ...],
    substr_filter: str | None,
) -> list[tuple[SanitizerCase, str]]:
    out: list[tuple[SanitizerCase, str]] = []
    needle = substr_filter.lower() if substr_filter else None
    for case in cases:
        haystack = f"{case.subsystem}:{case.target}:{case.test_name}".lower()
        if needle and needle not in haystack:
            continue
        for tool in selected_tools:
            if tool in case.tools:
                out.append((case, tool))
    return out


def build_test_executables(
    repo_root: pathlib.Path, targets: set[str]
) -> dict[str, pathlib.Path]:
    env = os.environ.copy()
    if "/opt/cuda/bin" not in env.get("PATH", ""):
        env["PATH"] = f"/opt/cuda/bin:{env.get('PATH', '')}"

    cmd = [
        "cargo",
        "test",
        "--release",
        "-p",
        "w34-host",
        "--features",
        "cuda",
        "--no-run",
        "--message-format=json",
    ]
    if "lib" in targets:
        cmd.append("--lib")
    for target in sorted(targets - {"lib"}):
        cmd.extend(["--test", target])

    proc = subprocess.run(
        cmd, cwd=repo_root, env=env, capture_output=True, text=True, check=False
    )
    if proc.returncode != 0:
        sys.stderr.write(proc.stderr)
        raise RuntimeError(f"cargo test --no-run failed (exit {proc.returncode})")

    executables: dict[str, pathlib.Path] = {}
    for line in proc.stdout.splitlines():
        line = line.strip()
        if not line.startswith("{"):
            continue
        msg = json.loads(line)
        if msg.get("reason") != "compiler-artifact":
            continue
        exe = msg.get("executable")
        if not exe:
            continue
        target_info = msg.get("target") or {}
        kinds = target_info.get("kind") or []
        name = target_info.get("name")
        if "lib" in kinds and name == "w34_host":
            executables["lib"] = pathlib.Path(exe)
        elif "test" in kinds and name in targets:
            executables[name] = pathlib.Path(exe)

    missing = sorted(targets - set(executables))
    if missing:
        raise FileNotFoundError(f"failed to locate built test executable(s) for: {missing}")
    return executables


def run_one_sanitizer_job(
    sanitizer_bin: str,
    exe: pathlib.Path,
    case: SanitizerCase,
    tool: str,
    cuda_visible_devices: str,
    timeout_secs: int,
) -> tuple[SanitizerCase, str, bool, float, str]:
    t0 = time.monotonic()
    env = os.environ.copy()
    if "/opt/cuda/bin" not in env.get("PATH", ""):
        env["PATH"] = f"/opt/cuda/bin:{env.get('PATH', '')}"
    env["CUDA_VISIBLE_DEVICES"] = cuda_visible_devices

    cmd = [
        sanitizer_bin,
        "--error-exitcode=1",
        "--tool",
        tool,
        str(exe),
        "--exact",
        case.test_name,
        "--test-threads=1",
        "--nocapture",
    ]
    # `compute-sanitizer` spawns the Rust test executable as a child process.
    # Start a dedicated session (`start_new_session=True`) so a timeout kills the
    # entire process group (`compute-sanitizer` + grandchild GPU test binary) and
    # never leaves an orphaned spin-barrier kernel squatting on the GPU.
    with subprocess.Popen(
        cmd,
        cwd=REPO_ROOT,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        start_new_session=True,
    ) as proc:
        try:
            stdout, stderr = proc.communicate(timeout=timeout_secs)
            dt = time.monotonic() - t0
            combined = f"{stdout}\n{stderr}".strip()
            # Guard against typos where `--exact` matches 0 tests and exits 0!
            ran_one = "1 passed;" in combined
            ok = proc.returncode == 0 and ran_one
            if proc.returncode == 0 and not ran_one:
                combined += "\n[cuda-sanitize] ERROR: test filter matched 0 tests!"
            return (case, tool, ok, dt, combined)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(proc.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            stdout, stderr = proc.communicate()
            dt = time.monotonic() - t0
            out = f"TIMEOUT after {timeout_secs}s\n{stdout or ''}\n{stderr or ''}"
            return (case, tool, False, dt, out.strip())


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--tool",
        choices=("all", *ALL_TOOLS),
        default="all",
        help="Sanitizer tool to run (default: all = memcheck + racecheck + synccheck)",
    )
    parser.add_argument(
        "--filter",
        default=None,
        help="Case-insensitive substring filter over subsystem:target:test_name",
    )
    parser.add_argument(
        "--jobs",
        type=int,
        default=0,
        help="Max concurrent single-GPU sanitizer jobs (default: 1 per visible GPU)",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=180,
        help="Per-invocation timeout in seconds (default: 180)",
    )
    parser.add_argument(
        "--require-multi-gpu",
        action="store_true",
        help="Fail instead of skipping min_gpus=2 cases when fewer than 2 GPUs are visible",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List selected (case, tool) pairs and exit",
    )
    args = parser.parse_args(argv)

    selected_tools = ALL_TOOLS if args.tool == "all" else (args.tool,)
    plan = select_cases(SANITIZER_MATRIX, selected_tools, args.filter)
    if not plan:
        print("[cuda-sanitize] no cases matched filter", file=sys.stderr)
        return 1

    if args.list:
        for case, tool in plan:
            print(f"{tool:9s}  gpus>={case.min_gpus}  {case.subsystem:20s}  {case.target}::{case.test_name}")
        print(f"Total: {len(plan)} sanitizer invocations")
        return 0

    t_start = time.monotonic()
    sanitizer_bin = find_compute_sanitizer()
    gpus = discover_visible_gpus()
    if not gpus:
        print("[cuda-sanitize] no visible NVIDIA GPUs discovered", file=sys.stderr)
        return 1

    single_gpu_jobs = [(c, t) for c, t in plan if c.min_gpus == 1]
    multi_gpu_jobs: list[tuple[SanitizerCase, str]] = []
    skipped_multi = 0
    for case, tool in plan:
        if case.min_gpus <= 1:
            continue
        if len(gpus) < case.min_gpus:
            if args.require_multi_gpu:
                print(
                    f"[cuda-sanitize] {case.target}::{case.test_name} requires >={case.min_gpus} GPUs, found {len(gpus)}",
                    file=sys.stderr,
                )
                return 1
            skipped_multi += 1
        else:
            multi_gpu_jobs.append((case, tool))

    active_jobs = single_gpu_jobs + multi_gpu_jobs
    needed_targets = {case.target for case, _ in active_jobs}
    print(
        f"[cuda-sanitize] building {len(needed_targets)} test target(s) for "
        f"{len(active_jobs)} sanitizer run(s) across GPU(s) {','.join(gpus)}...",
        flush=True,
    )
    executables = build_test_executables(REPO_ROOT, needed_targets)

    failures: list[tuple[SanitizerCase, str, float, str]] = []
    completed = 0
    total = len(active_jobs)

    # Phase 1: run single-GPU jobs concurrently across disjoint GPUs (1 process per GPU).
    if single_gpu_jobs:
        workers = max(1, min(len(gpus), args.jobs if args.jobs > 0 else len(gpus)))
        gpu_pool: queue.Queue[str] = queue.Queue()
        for gpu in gpus[:workers]:
            gpu_pool.put(gpu)

        def _worker(item: tuple[SanitizerCase, str]):
            case, tool = item
            gpu_id = gpu_pool.get()
            try:
                return run_one_sanitizer_job(
                    sanitizer_bin,
                    executables[case.target],
                    case,
                    tool,
                    gpu_id,
                    args.timeout,
                )
            finally:
                gpu_pool.put(gpu_id)

        with ThreadPoolExecutor(max_workers=workers) as pool:
            futs = [pool.submit(_worker, item) for item in single_gpu_jobs]
            for fut in as_completed(futs):
                case, tool, ok, dt, output = fut.result()
                completed += 1
                status = "ok" if ok else "FAIL"
                print(
                    f"  [{completed:3d}/{total:3d}] {status:4s}  {tool:9s}  {dt:5.1f}s  "
                    f"{case.target}::{case.test_name}",
                    flush=True,
                )
                if not ok:
                    failures.append((case, tool, dt, output))

    # Phase 2: run multi-GPU (TP=2 / TP=4) jobs serially across `case.min_gpus` visible GPUs.
    if multi_gpu_jobs:
        for case, tool in multi_gpu_jobs:
            pair_cvd = ",".join(gpus[: case.min_gpus])
            case, tool, ok, dt, output = run_one_sanitizer_job(
                sanitizer_bin,
                executables[case.target],
                case,
                tool,
                pair_cvd,
                args.timeout,
            )
            completed += 1
            status = "ok" if ok else "FAIL"
            print(
                f"  [{completed:3d}/{total:3d}] {status:4s}  {tool:9s}  {dt:5.1f}s  "
                f"{case.target}::{case.test_name} (gpus={pair_cvd})",
                flush=True,
            )
            if not ok:
                failures.append((case, tool, dt, output))

    wall_s = time.monotonic() - t_start
    if failures:
        print(
            f"\n[cuda-sanitize] FAILED: {len(failures)}/{total} sanitizer run(s) failed ({wall_s:.1f}s):",
            file=sys.stderr,
        )
        for case, tool, dt, output in failures:
            print(
                f"\n=== FAIL [{tool}] {case.target}::{case.test_name} ({dt:.1f}s) ===",
                file=sys.stderr,
            )
            tail = "\n".join(output.splitlines()[-60:])
            print(tail, file=sys.stderr)
        return 1

    skip_note = f", skipped_multi_gpu={skipped_multi}" if skipped_multi else ""
    print(
        f"[cuda-sanitize] ok: {total}/{total} sanitizer runs passed "
        f"({','.join(selected_tools)}{skip_note}) in {wall_s:.1f}s"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
