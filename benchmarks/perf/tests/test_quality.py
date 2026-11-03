import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from w34perf.quality import qualify


FIXTURE = Path(__file__).resolve().parents[1] / "fixtures/quality.json"
ENGINE = {"model": "test-model", "base_url": "http://localhost:9999/v1",
          "quality_evidence": "already certified"}


class QualityTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.out = Path(self.tmp.name) / "evidence.json"
        self.fixture = json.loads(FIXTURE.read_text())

    def answers(self):
        return [{"status": "ok", "finish_reason": "stop",
                 "content": " \n" + c["expected"] + "\n", "usage": {"completion_tokens": 3}}
                for c in self.fixture["checks"]]

    def test_pass_evidence_and_no_overwrite(self):
        with patch("w34perf.quality.stream", side_effect=self.answers()) as http:
            report = qualify(ENGINE, FIXTURE, self.out, 12)
            self.assertEqual(report["status"], "passed")
            self.assertEqual(json.loads(self.out.read_text()), report)
            self.assertEqual(report["fixture_sha256"], hashlib.sha256(FIXTURE.read_bytes()).hexdigest())
            self.assertEqual(http.call_count, 6)
            self.assertTrue(all(c["status"] == "passed" for c in report["checks"]))
            for call in http.call_args_list:
                self.assertEqual(call.args[0], ENGINE)
                self.assertEqual(call.args[2], 12)
                self.assertTrue(call.args[1]["stream"])
            with self.assertRaises(FileExistsError):
                qualify(ENGINE, FIXTURE, self.out)
            self.assertEqual(http.call_count, 6)

    def test_failures_retained_and_not_certified_by_profile(self):
        answers = self.answers()
        answers[0]["content"] = "The answer is 93"
        answers[1]["status"] = "timeout"
        answers[2]["finish_reason"] = "length"
        answers[3]["content"] = "Found: cobalt-otter-7183!"
        answers[4] = OSError("private credential")
        with patch("w34perf.quality.stream", side_effect=answers):
            report = qualify(ENGINE, FIXTURE, self.out)
        self.assertEqual(report["status"], "failed")
        self.assertEqual([c["status"] for c in report["checks"]],
                         ["failed", "failed", "failed", "passed", "failed", "passed"])
        self.assertEqual(report["checks"][1]["usage"], {"completion_tokens": 3})
        self.assertNotIn("private credential", self.out.read_text())

    def test_validation_before_http(self):
        with patch("w34perf.quality.stream") as http:
            for timeout in (True, 0, -1, float("nan"), float("inf"), "180"):
                with self.assertRaises(ValueError):
                    qualify(ENGINE, FIXTURE, self.out, timeout)
            for bad in ({"schema_version": 1, "checks": []},
                        {"schema_version": 1, "checks": [{"id": "x", "kind": "exact", "prompt": "x", "expected": " "}]},
                        {"schema_version": 1, "checks": [{"id": "x", "kind": "retrieval", "prompt": "x", "expected": "secret", "context_lines": 3, "depth": 2}]}):
                path = Path(self.tmp.name) / "invalid.json"
                path.write_text(json.dumps(bad))
                with self.assertRaises(ValueError):
                    qualify(ENGINE, path, self.out)
            http.assert_not_called()
        self.assertFalse(self.out.exists())

    def test_generated_context_is_repeatable_and_depth_diverse(self):
        with patch("w34perf.quality.stream", side_effect=self.answers() * 2):
            first = qualify(ENGINE, FIXTURE, self.out)
            second = qualify(ENGINE, FIXTURE, self.out.with_name("second.json"))
        self.assertEqual([c["request_sha256"] for c in first["checks"]],
                         [c["request_sha256"] for c in second["checks"]])
        for check, evidence in zip(self.fixture["checks"][3:], first["checks"][3:]):
            lines = evidence["request"]["messages"][0]["content"].splitlines()
            self.assertEqual(lines.index("TARGET_SECRET: " + check["expected"]),
                             1 + int(check["context_lines"] * check["depth"]))


if __name__ == "__main__":
    unittest.main()
