import copy
import csv
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from w34perf.__main__ import main
from w34perf.config import load
from w34perf.report import render
from w34perf.runner import execute
from w34perf.selection import select
from test_perf import ROOT, result


def config():
    cfg = load(ROOT / "suite.toml")
    cfg["engines"] = copy.deepcopy(cfg["engines"][:2])
    cfg["cases"] = cfg["cases"][:1]
    cfg["run"].update(repetitions=2, warmup_requests=0)
    return cfg


class SelectionTests(unittest.TestCase):
    def test_single_card_filter_retains_dual_card_coverage(self):
        cfg = config()
        engines, cases, coverage = select(cfg, gpu_count=1)
        self.assertEqual([e["id"] for e in engines], [cfg["engines"][0]["id"]])
        self.assertEqual([c["status"] for c in coverage], ["not_run", "filtered"])
        self.assertIn("1 GPU", coverage[1]["reason"])
        self.assertEqual(cases, cfg["cases"])

    def test_disabled_and_unsupported_need_no_endpoint_or_model(self):
        for settings, expected in (("enabled = false", "disabled"),
                                   ('support = "unsupported"', "unsupported")):
            with self.subTest(settings=settings), tempfile.TemporaryDirectory() as tmp:
                path = Path(tmp) / "suite.toml"
                text = ('schema_version = 1\n[[engines]]\nid = "skip"\nengine = "example"\n'
                        'gpu_count = 2\n' + settings + '\nskip_reason = "no dual-card backend"\n'
                        '[[cases]]\nid = "smoke"\nfixture = ' +
                        json.dumps(str(ROOT / "fixtures/text.json")) + '\n')
                path.write_text(text)
                cfg = load(path)
                engines, _, coverage = select(cfg, ["skip"], gpu_count=1)
                self.assertEqual(engines, [])
                self.assertEqual(coverage[0]["status"], expected)
                self.assertEqual(coverage[0]["reason"], "no dual-card backend")
                path.write_text(text.replace('skip_reason = "no dual-card backend"', 'skip_reason = " "'))
                with self.assertRaisesRegex(ValueError, "skip_reason"):
                    load(path)

    def test_optional_endpoint_cannot_hide_credentials_in_disabled_metadata(self):
        original = (ROOT / "suite.toml").read_text()
        original = original.replace('enabled = true', 'enabled = false\nskip_reason = "disabled"')
        original = original.replace('http://127.0.0.1:8000/v1', 'http://user:secret@localhost/v1')
        with tempfile.TemporaryDirectory(dir=ROOT) as tmp:
            path = Path(tmp) / "bad.toml"
            path.write_text(original.replace('"fixtures/', '"../fixtures/'))
            with self.assertRaisesRegex(ValueError, "no credentials"):
                load(path)

    def test_exclusions_and_unknown_ids_are_validated_consistently(self):
        cfg = config()
        engines, _, coverage = select(cfg, exclude_ids=[cfg["engines"][0]["id"]])
        self.assertEqual(len(engines), 1)
        self.assertEqual(coverage[0]["status"], "excluded")
        for kwargs in ({"engine_ids": ["typo"]}, {"exclude_ids": ["typo"]},
                       {"case_ids": ["typo"]}, {"engine_ids": []}, {"gpu_count": True}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                select(cfg, **kwargs)

    def test_all_skipped_run_has_no_network_credentials_or_gpu_probes(self):
        cfg = config()
        cfg["engines"][0].update(enabled=False, skip_reason="temporarily unavailable", api_key_env="ABSENT_TEST_KEY")
        cfg["engines"][1].update(support="unsupported", skip_reason="no two-card backend")
        with tempfile.TemporaryDirectory() as tmp, patch("w34perf.runner.stream") as stream, \
                patch("w34perf.runner.provenance") as provenance:
            out = Path(tmp) / "run"
            self.assertEqual(execute(cfg, out, [e["id"] for e in cfg["engines"]]), [])
            stream.assert_not_called()
            provenance.assert_not_called()
            self.assertEqual(json.loads((out / "manifest.json").read_text())["status"], "skipped")
            self.assertEqual((out / "requests.jsonl").read_text(), "")
            self.assertIn("coverage.json", json.loads((out / "SHA256SUMS.json").read_text()))
            md, csv_text = render(out)
            self.assertIn("no requests were sent", md)
            rows = list(csv.DictReader(io.StringIO(csv_text)))
            self.assertEqual([r["status"] for r in rows], ["disabled", "unsupported"])
            self.assertTrue(all(r["completion_tokens_s"] == "" for r in rows))
            self.assertTrue(all(r["row_type"] == "coverage" for r in rows))

    def test_filtered_endpoints_do_not_trigger_duplicate_url_guard(self):
        cfg = config()
        with tempfile.TemporaryDirectory() as tmp, patch("w34perf.runner.stream", return_value=result()), \
                patch("w34perf.runner.provenance", return_value={}):
            out = Path(tmp) / "run"
            rows = execute(cfg, out, gpu_count=1)
            coverage = json.loads((out / "coverage.json").read_text())
            self.assertEqual(len(rows), 2)
            self.assertEqual(coverage[0]["status"], "measured")
            self.assertEqual(coverage[0]["completed_repetitions"], 2)
            self.assertEqual(coverage[1]["status"], "filtered")

    def test_failed_trial_not_run_and_failed_warmup_remain_distinct(self):
        cfg = config()
        cfg["engines"][1]["base_url"] = "http://127.0.0.1:19001/v1"
        cfg["run"].update(repetitions=1, warmup_requests=1)
        with tempfile.TemporaryDirectory() as tmp, patch("w34perf.runner.provenance", return_value={}), \
                patch("w34perf.runner.stream", return_value={**result(), "status": "timeout"}):
            out = Path(tmp) / "failed"
            with self.assertRaises(RuntimeError):
                execute(cfg, out)
            cells = json.loads((out / "coverage.json").read_text())
            self.assertEqual([c["status"] for c in cells], ["failed", "not_run"])
            self.assertEqual(cells[0]["completed_repetitions"], 0)
        cfg["run"]["warmup_requests"] = 0
        with tempfile.TemporaryDirectory() as tmp, patch("w34perf.runner.provenance", return_value={}), \
                patch("w34perf.runner.stream", return_value={**result(), "usage": {}}):
            out = Path(tmp) / "usage"
            execute(cfg, out, [cfg["engines"][0]["id"]])
            cells = json.loads((out / "coverage.json").read_text())
            self.assertEqual(cells[0]["status"], "measurement_failed")
            self.assertEqual(cells[1]["status"], "not_selected")

    def test_plan_and_run_cli_share_filters_and_all_skipped_exit_zero(self):
        args = [str(ROOT / "suite.toml"), "--engine", "warp34-2-tq4", "--gpus", "1", "--case", "smoke"]
        output = io.StringIO()
        with patch("sys.stdout", output):
            self.assertEqual(main(["plan", *args]), 0)
        plan = json.loads(output.getvalue())
        self.assertEqual(plan["selected_engines"], [])
        with tempfile.TemporaryDirectory() as tmp, patch("sys.stdout", io.StringIO()):
            out = Path(tmp) / "run"
            self.assertEqual(main(["run", *args, "--out", str(out)]), 0)
            self.assertEqual(json.loads((out / "coverage.json").read_text()), plan["coverage"])
