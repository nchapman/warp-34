"""Host-only unit tests for `tools/check_cuda_binaries.py` and `tools/check_cuda_sanitizer.py`."""

from __future__ import annotations

import pathlib
import sys
import unittest

TOOLS_DIR = pathlib.Path(__file__).resolve().parent.parent
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

import check_cuda_binaries as cuda_bin  # noqa: E402
import check_cuda_sanitizer as cuda_san  # noqa: E402


SAMPLE_CUOBJDUMP = """
Fatbin elf code:
================
arch = sm_86
code version = [1,8]
host = linux
compile_size = 64bit

Resource usage:
 Common:
  GLOBAL:0
 Function _ZN11w34_sampler23sampler_step_f16_kernelEPK6__half:
  REG:64 STACK:32 SHARED:28492 LOCAL:0 CONSTANT[0]:440 TEXTURE:0 SURFACE:0 SAMPLER:0
 Function _ZN3w3441tp2_all_reduce_hc_combine_tap_f16_kernelEPK6__half:
  REG:48 STACK:0 SHARED:0 LOCAL:0 CONSTANT[0]:456 TEXTURE:0 SURFACE:0 SAMPLER:0

Fatbin elf code:
================
arch = sm_89
code version = [1,8]
host = linux
compile_size = 64bit

Resource usage:
 Common:
  GLOBAL:0
 Function _ZN11w34_sampler23sampler_step_f16_kernelEPK6__half:
  REG:64 STACK:32 SHARED:28492 LOCAL:0 CONSTANT[0]:440 TEXTURE:0 SURFACE:0 SAMPLER:0
 Function _ZN3w3441tp2_all_reduce_hc_combine_tap_f16_kernelEPK6__half:
  REG:48 STACK:0 SHARED:0 LOCAL:0 CONSTANT[0]:456 TEXTURE:0 SURFACE:0 SAMPLER:0
"""


class CudaBinaryGateTests(unittest.TestCase):
    def test_parses_and_accepts_clean_multi_arch_dump(self) -> None:
        entries = cuda_bin.parse_cuobjdump_resources(SAMPLE_CUOBJDUMP)
        self.assertEqual(len(entries), 4)
        failures = cuda_bin.audit_kernel_resources(
            entries,
            expected_arches=("sm_86", "sm_89"),
            min_kernels_per_arch=2,
        )
        self.assertEqual(failures, [])

    def test_rejects_unexpected_stack_local_shared_reg_and_arch_skew(self) -> None:
        entries = [
            # Unexpected non-zero STACK on a non-allowlisted kernel
            cuda_bin.KernelResource("sm_86", "tp2_all_reduce_hc_combine_tap_f16_kernel", 48, 8, 0, 0),
            # Allowlisted kernel exceeding its STACK and REG ceilings
            cuda_bin.KernelResource("sm_86", "sampler_step_f16_kernel", 72, 64, 28492, 0),
            # Static SHARED > 48 KiB and LOCAL > 0
            cuda_bin.KernelResource("sm_89", "custom_kernel", 32, 0, 50000, 16),
        ]
        failures = cuda_bin.audit_kernel_resources(
            entries,
            expected_arches=("sm_86", "sm_89"),
            min_kernels_per_arch=1,
        )
        joined = "\n".join(failures)
        self.assertIn("kernel count mismatch", joined)
        self.assertIn("tp2_all_reduce_hc_combine_tap_f16_kernel: STACK=8 exceeds ceiling 0", joined)
        self.assertIn("sampler_step_f16_kernel: STACK=64 exceeds ceiling 32", joined)
        self.assertIn("sampler_step_f16_kernel: REG=72 exceeds ceiling 64", joined)
        self.assertIn("custom_kernel: LOCAL=16 > 0", joined)
        self.assertIn("custom_kernel: static SHARED=50000 > 49152", joined)

    def test_expected_arches_from_env(self) -> None:
        self.assertEqual(cuda_bin.expected_arches_from_env(None), ("sm_86", "sm_89"))
        self.assertEqual(cuda_bin.expected_arches_from_env(""), ("sm_86", "sm_89"))
        self.assertEqual(cuda_bin.expected_arches_from_env("sm_86"), ("sm_86",))
        self.assertEqual(cuda_bin.expected_arches_from_env("86, sm_89"), ("sm_86", "sm_89"))


class CudaSanitizerGateTests(unittest.TestCase):
    def test_matrix_covers_all_seven_subsystem_groups(self) -> None:
        subsystems = {c.subsystem for c in cuda_san.SANITIZER_MATRIX}
        expected = {
            "linear",
            "kv_attn",
            "gdn",
            "moe",
            "norms_hc_ple",
            "sampler_dflash_vision",
            "collectives_tp2",
        }
        self.assertTrue(expected.issubset(subsystems), f"missing: {expected - subsystems}")

    def test_select_cases_filters_by_tool_and_substring(self) -> None:
        only_race = cuda_san.select_cases(
            cuda_san.SANITIZER_MATRIX, ("racecheck",), "qsa"
        )
        self.assertGreaterEqual(len(only_race), 1)
        for case, tool in only_race:
            self.assertEqual(tool, "racecheck")
            self.assertIn(
                "qsa",
                f"{case.subsystem}:{case.target}:{case.test_name}".lower(),
            )

    def test_discover_visible_gpus_parses_explicit_env(self) -> None:
        self.assertEqual(cuda_san.discover_visible_gpus("0, 1"), ["0", "1"])
        self.assertEqual(cuda_san.discover_visible_gpus("GPU-abc"), ["GPU-abc"])


if __name__ == "__main__":
    unittest.main()
