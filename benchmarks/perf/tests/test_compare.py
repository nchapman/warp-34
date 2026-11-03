import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from w34perf.compare import compare


class CompareTests(unittest.TestCase):
    def bundle(self, root, name, change=None, rows=None):
        path = root / name
        path.mkdir()
        manifest = {"status": "complete", "config": {
            "run": {"seed": 34}, "cases": [{"id": "c", "max_tokens": 100}],
            "engines": [{"id": "e", "model_revision": "rev", "parallelism": "tp1",
                         "cache_config": "off", "request_overrides": {},
                         "engine": "vllm", "version": "1", "container": "image",
                         "artifact_sha256": "artifact", "tokenizer_revision": "tok",
                         "chat_template_sha256": "template", "activation_dtype": "fp16",
                         "source_model_revision": "source-rev"}]},
            "provenance": {"suite_source_sha256": "source", "gpu": {"stdout": "GPU A"},
                           "topology": {"stdout": "topology"}}}
        trial = {"engine_id": "e", "case_id": "c", "fixture_sha256": "fixture",
                 "cohort": "cohort", "gpu_count": 1, "weight_format": "int4",
                 "kv_format": "fp16", "speculation": "off", "duration_s": 2,
                 "completion_tokens": 100, "token_accounting_complete": True,
                 "failed_requests": 0, "publication_blockers": ["quality unverified"]}
        if change:
            change(manifest, trial)
        for filename, value in (("manifest", manifest), ("summary", rows if rows is not None else [trial]),
                                ("coverage", [{"status": "unsupported", "reason": "not available"}])):
            (path / (filename + ".json")).write_text(json.dumps(value))
        return path

    def test_controls_and_sweeps(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            a = self.bundle(root, "a")
            for key in ("cohort", "fixture_sha256", "gpu_count"):
                b = self.bundle(root, key, lambda m, t: t.update({key: "different"}))
                pair = compare([a, b])["comparisons"][0]
                self.assertEqual(pair["status"], "incompatible")
                self.assertTrue(pair["differences"])
            b = self.bundle(root, "hardware", lambda m, t: m["provenance"].update(gpu={"stdout": "GPU B"}))
            self.assertEqual(compare([a, b])["comparisons"][0]["status"], "incompatible")
            b = self.bundle(root, "sweep", lambda m, t: t.update(weight_format="fp4"))
            result = compare([a, b])
            self.assertEqual(result["comparisons"][0]["status"], "labeled_sweep")
            self.assertFalse(result["publication_ready"])
            self.assertTrue(result["runs"][0]["evidence_sha256"])
            self.assertEqual(result["runs"][0]["coverage"][0]["status"], "unsupported")
            json.dumps(result, allow_nan=False)

    def test_weighted_throughput_failures_and_missing_usage(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = self.bundle(Path(tmp), "a")
            first = json.loads((path / "summary.json").read_text())[0]
            second = copy.deepcopy(first)
            second.update(duration_s=8, completion_tokens=80, failed_requests=1)
            (path / "summary.json").write_text(json.dumps([first, second]))
            group = compare([path])["deployments"][0]
            self.assertEqual(group["completion_tokens_s"], 18)
            self.assertEqual((group["trial_min"], group["trial_max"]), (10, 50))
            self.assertEqual(group["trials"][1]["failed_requests"], 1)
            second.update(completion_tokens=None, token_accounting_complete=False)
            (path / "summary.json").write_text(json.dumps([first, second]))
            group = compare([path])["deployments"][0]
            self.assertIsNone(group["completion_tokens_s"])
            self.assertEqual(group["trial_completion_tokens_s"], [50, None])

    def test_identity_controls(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            a = self.bundle(root, "a")
            for field in ("artifact_sha256", "tokenizer_revision", "chat_template_sha256",
                          "activation_dtype", "source_model_revision"):
                for value in (None, "different"):
                    b = self.bundle(root, field + str(value),
                                    lambda m, t: m["config"]["engines"][0].update({field: value}))
                    pair = compare([a, b])["comparisons"][0]
                    self.assertIn(field, pair["differences"])
                    self.assertFalse(pair["same_artifact_equivalent"])
                    if value is None:
                        self.assertEqual(compare([b, b])["comparisons"][0]["status"], "unverified")
            b = self.bundle(root, "engine", lambda m, t: m["config"]["engines"][0].update(
                engine="sglang", version="2", container="other"))
            result = compare([a, b])
            self.assertEqual(result["comparisons"][0]["status"], "matching_controls")
            self.assertEqual(result["deployments"][1]["identity"],
                             {"engine": "sglang", "version": "2", "container": "other"})
            b = self.bundle(root, "missing-identity", lambda m, t: m["config"]["engines"][0].pop("version"))
            self.assertEqual(compare([a, b])["comparisons"][0]["status"], "unverified")

    def test_malformed_nested_evidence(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            mutations = [(field, value) for field in ("config", "provenance")
                         for value in (None, [], "bad")]
            mutations += [(field, value) for field in ("engines", "cases")
                          for value in (None, {}, [None, "bad"])]
            for i, (field, value) in enumerate(mutations):
                def change(m, t):
                    (m if field in ("config", "provenance") else m["config"])[field] = value
                path = self.bundle(root, str(i), change)
                result = compare([path])
                self.assertTrue(any(field in str(b) for b in result["runs"][0]["blockers"]))
            (path / "coverage.json").write_text('[null, "bad"]')
            self.assertTrue(any("coverage" in str(b) for b in compare([path])["runs"][0]["blockers"]))

    def test_empty_and_missing(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = self.bundle(Path(tmp), "empty", rows=[])
            result = compare([path, Path(tmp) / "missing"])
            self.assertEqual(result["deployments"], [])
            self.assertTrue(all(run["blockers"] for run in result["runs"]))
            self.assertEqual(compare([])["rankings"], [])


if __name__ == "__main__":
    unittest.main()
