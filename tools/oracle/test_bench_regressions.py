"""Host-only unit tests for `tools/check_bench_regressions.py` and `bench_baselines.json`."""

from __future__ import annotations

import pathlib
import sys
import tempfile
import unittest

TOOLS_DIR = pathlib.Path(__file__).resolve().parent.parent
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

import check_bench_regressions as bench_reg  # noqa: E402


class BenchRegressionsTests(unittest.TestCase):
    def setUp(self) -> None:
        self.baselines = bench_reg.load_baselines()

    def test_baselines_schema_covers_quick_and_full_suites(self) -> None:
        cases = self.baselines["cases"]
        for cid in bench_reg.QUICK_SUITE_CASES:
            self.assertIn(cid, cases)
        for cid in bench_reg.FULL_SUITE_CASES:
            self.assertIn(cid, cases)
        # Verify synthetic vs realistic and Plain vs MTP vs DFlash2 separation
        self.assertEqual(cases["4b_tp1_plain"]["speculative"], None)
        self.assertEqual(cases["4b_tp1_plain"]["prompt_source"], "synthetic")
        self.assertEqual(cases["4b_tp1_mtp_synth"]["speculative"], "mtp")
        self.assertEqual(cases["4b_tp1_mtp_synth"]["prompt_source"], "synthetic")
        self.assertEqual(cases["4b_tp1_mtp_realistic"]["speculative"], "mtp")
        self.assertEqual(cases["4b_tp1_mtp_realistic"]["prompt_source"], "realistic")
        self.assertIn("--realistic-prompt", cases["4b_tp1_mtp_realistic"]["extra_args"])
        self.assertEqual(cases["4b_tp1_dflash2_realistic"]["speculative"], "dflash2")
        self.assertEqual(cases["4b_tp1_dflash2_realistic"]["prompt_source"], "realistic")

    def test_normalize_arch_accepts_common_formats(self) -> None:
        self.assertEqual(bench_reg.normalize_arch("8.6"), "sm_86")
        self.assertEqual(bench_reg.normalize_arch("sm_86"), "sm_86")
        self.assertEqual(bench_reg.normalize_arch("89"), "sm_89")
        self.assertEqual(bench_reg.normalize_arch("8.0"), "sm_80")
        with self.assertRaises(ValueError):
            bench_reg.normalize_arch("sm_75")

    def test_evaluate_case_passes_healthy_stage_all_payload(self) -> None:
        payload = {
            "model": "nchapman/Qwen3.5-4B-W4A16-HGPTQ",
            "tp": 1,
            "kv_format": "tq4",
            "speculative": "mtp",
            "decode": {
                "tp": 1,
                "cuda_graph": True,
                "speculative": "mtp",
                "prompt_source": "synthetic",
                "mean_accepted_drafts": 2.97,
                "prefill_tok_per_s": 6610.0,
                "decode_tok_per_s": 589.4,
            },
            "radix": {
                "ttft_speedup_x": 4.81,
                "lossless_parity": True,
            },
            "scheduler": {
                "aggregate_decode_tok_per_s": 1350.4,
                "total_tok_per_s": 5739.2,
            },
            "vision": {
                "ttft_speedup_x": 2.40,
                "lossless_parity": True,
                "content_hash_isolated": True,
            },
        }
        ev = bench_reg.evaluate_case(
            case_id="4b_tp1_mtp_synth",
            payload=payload,
            baselines_doc=self.baselines,
            arch="sm_86",
            max_regression_pct=12.0,
        )
        self.assertTrue(ev.passed)
        self.assertEqual(ev.mode_label, "MTP-3")
        self.assertEqual(ev.prompt_source, "synthetic")
        self.assertIn("vLLM 0.31.0", ev.competitor_summary)

    def test_evaluate_case_catches_throughput_and_invariant_regressions(self) -> None:
        payload = {
            "tp": 1,
            "speculative": "mtp",
            "decode": {
                "tp": 1,
                "cuda_graph": False,
                "speculative": "mtp",
                "prompt_source": "synthetic",
                "mean_accepted_drafts": 0.90,
                "prefill_tok_per_s": 5757.0,
                "decode_tok_per_s": 290.0,
            },
        }
        ev = bench_reg.evaluate_case(
            case_id="4b_tp1_mtp_realistic",
            payload=payload,
            baselines_doc=self.baselines,
            arch="sm_86",
            max_regression_pct=10.0,
        )
        self.assertFalse(ev.passed)
        joined_inv = "\n".join(ev.invariant_failures)
        self.assertIn("expected prompt_source='realistic', got 'synthetic'", joined_inv)
        self.assertIn("expected decode.cuda_graph=True", joined_inv)
        failed_metrics = {m.metric for m in ev.metrics if not m.passed}
        self.assertIn("decode_tok_per_s", failed_metrics)
        self.assertIn("mean_accepted_drafts", failed_metrics)

    def test_render_markdown_report_and_cli_input_mode(self) -> None:
        payload = {
            "tp": 1,
            "cuda_graph": True,
            "speculative": "mtp",
            "prompt_source": "realistic",
            "mean_accepted_drafts": 1.65,
            "prefill_tok_per_s": 5757.3,
            "decode_tok_per_s": 394.0,
        }
        with tempfile.TemporaryDirectory() as tmp:
            json_path = pathlib.Path(tmp) / "mtp_real.json"
            md_path = pathlib.Path(tmp) / "report.md"
            import json

            json_path.write_text(json.dumps(payload), encoding="utf-8")
            rc = bench_reg.main(
                [
                    "--arch",
                    "sm_86",
                    "--input",
                    f"4b_tp1_mtp_realistic={json_path}",
                    "--markdown-out",
                    str(md_path),
                ]
            )
            self.assertEqual(rc, 0)
            md = md_path.read_text(encoding="utf-8")
            self.assertIn("`4b_tp1_mtp_realistic`", md)
            self.assertIn("MTP-3", md)
            self.assertIn("`realistic`", md)
            self.assertIn("**PASS**", md)

    def test_build_bench_command_includes_realistic_and_lora_flags(self) -> None:
        cases = self.baselines["cases"]
        cmd_real = bench_reg.build_bench_command(
            pathlib.Path("/bin/warp34"),
            cases["4b_tp1_mtp_realistic"],
            pathlib.Path("/tmp/out.json"),
        )
        self.assertIn("--realistic-prompt", cmd_real)
        self.assertIn("--speculative", cmd_real)
        self.assertIn("--out", cmd_real)
        cmd_lora = bench_reg.build_bench_command(
            pathlib.Path("/bin/warp34"),
            cases["4b_tp1_lora_cai"],
            pathlib.Path("/tmp/out_lora.json"),
        )
        self.assertIn("--active-lora", cmd_lora)
        self.assertIn("cai", cmd_lora)

    def test_multi_arch_baselines_cover_sm80_sm86_sm89(self) -> None:
        cases = self.baselines["cases"]
        multi_arch_cases = (
            "4b_tp1_plain",
            "4b_tp1_mtp_synth",
            "4b_tp1_dflash2_synth",
            "4b_tp2_plain",
            "4b_tp2_mtp_synth",
            "4b_tp2_dflash2_synth",
            "27b_tp2_plain",
            "27b_tp2_mtp_synth",
            "27b_tp2_dflash2_synth",
            "35b_tp2_plain",
            "35b_tp2_mtp_synth",
            "flash_next_tp2_mtp_synth",
        )
        for cid in multi_arch_cases:
            hw = cases[cid]["hardware"]
            for arch in ("sm_80", "sm_86", "sm_89"):
                self.assertIn(arch, hw, f"{cid} missing {arch} baseline")
                self.assertGreater(hw[arch]["decode_tok_per_s"], 0.0)


if __name__ == "__main__":
    unittest.main()
