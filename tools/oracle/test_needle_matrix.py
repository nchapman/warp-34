import unittest
from needle_matrix import cases, passes_report


class NeedleMatrixTests(unittest.TestCase):
    def test_default_matrix_covers_both_policies_at_all_four_windows(self):
        matrix = cases([50])
        self.assertEqual(len(matrix), 8)
        self.assertEqual({window for _, window, _ in matrix}, {32768, 65536, 131072, 262144})
        self.assertEqual({policy for policy, _, _ in matrix}, {"tq4", "tq4_all"})

    def test_multiple_depths_are_measured_independently(self):
        self.assertEqual(len(cases([0, 25, 50, 75, 100])), 40)
        self.assertEqual(len(cases([50], ("tq4",))), 4)
        self.assertEqual({policy for policy, _, _ in cases([50], ("tq4_all",))}, {"tq4_all"})
        for depths in ([], [-1], [101]):
            with self.assertRaises(ValueError):
                cases(depths)
        for policies in ([], ("fp16",)):
            with self.assertRaises(ValueError):
                cases([50], policies)

    def test_gate_requires_successful_process_exact_retrieval_and_matching_geometry(self):
        report = {"passed": True, "context_window": 32768, "kv_policy": "tq4", "depth_percent": 50,
                  "tp": 1, "prompt_tokens": 32732, "max_new_tokens": 32, "reserved_tokens": 4,
                  "cached_prefix_tokens": 0, "answer": "73918462", "expected": "73918462"}
        self.assertTrue(passes_report(0, report, "tq4", 32768, 50))
        self.assertFalse(passes_report(1, report, "tq4", 32768, 50))
        for key, value in [("passed", False), ("context_window", 4096), ("kv_policy", "tq4_all"),
                           ("depth_percent", 25), ("tp", 2), ("prompt_tokens", 20),
                           ("cached_prefix_tokens", 64), ("answer", None), ("answer", []),
                           ("prompt_tokens", "32732"), ("tp", True), ("answer", "code is 73918462"), ("expected", "wrong")]:
            with self.subTest(key=key):
                self.assertFalse(passes_report(0, {**report, key: value}, "tq4", 32768, 50))

class OrchestrationTests(unittest.TestCase):
    def run_matrix(self, out, run):
        from unittest.mock import patch
        import needle_matrix
        with patch('sys.argv', ['needle_matrix', '--out-dir', str(out)]), patch.dict('os.environ', {'CUDA_VISIBLE_DEVICES': '0'}), patch.object(needle_matrix.subprocess, 'run', side_effect=run):
            return needle_matrix.main()

    def test_build_failure_replaces_stale_summary(self):
        import tempfile, json, subprocess
        from pathlib import Path
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            (out / 'summary.json').write_text('{"results": [{"passed": true}]}')
            with self.assertRaises(subprocess.CalledProcessError):
                self.run_matrix(out, subprocess.CalledProcessError(1, 'cargo'))
            summary = json.loads((out / 'summary.json').read_text())
            self.assertEqual(summary['results'], [])
            self.assertEqual(summary['expected_cases'], 8)

    def test_launch_error_and_malformed_reports_continue(self):
        import tempfile, json
        from pathlib import Path
        from types import SimpleNamespace
        calls = []
        def run(command, **kwargs):
            if command[0] == 'cargo':
                return SimpleNamespace(returncode=0)
            calls.append(command)
            if len(calls) == 1:
                raise OSError('launch failed')
            Path(command[-1]).write_text(json.dumps({'answer': None if len(calls) % 2 else [], 'ttft_ms': 17, 'decode_tok_per_s': 2, 'model_snapshot': 'test', 'passed': True}))
            return SimpleNamespace(returncode=0)
        with tempfile.TemporaryDirectory() as tmp:
            self.assertEqual(self.run_matrix(Path(tmp), run), 1)
            results = json.loads((Path(tmp) / 'summary.json').read_text())['results']
            self.assertEqual(len(results), 8)
            self.assertTrue(results[0]['failure'])
            self.assertTrue(all(not r['passed'] for r in results))
            self.assertTrue(all(r['ttft_ms'] == 17 for r in results[1:]))

    def test_unexpected_report_types_fail_closed(self):
        for report in (None, [], True, 'bad', {'answer': None}, {'answer': []}):
            with self.subTest(report=report):
                self.assertFalse(passes_report(0, report, 'tq4', 32768, 50))


if __name__ == "__main__":
    unittest.main()
