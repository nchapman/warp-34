import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

SCRIPT = Path(__file__).resolve().parents[3] / 'tools/perf_completion_report.py'


class CompletionReportTests(unittest.TestCase):
    def test_cli_retains_startup_failure_and_missing_usage(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / 'results'
            root.mkdir()
            (root / 'qualification.json').write_text(json.dumps({
                'engine': {'id': 'failed-start'}, 'status': 'failed', 'error': 'OOM'}))
            run = root / 'run'
            run.mkdir()
            (run / 'manifest.json').write_text(json.dumps({'status': 'complete',
                'selected_engines': ['example'], 'config': {'engines': [{'id': 'example'}]}}))
            (run / 'summary.json').write_text(json.dumps([{
                'engine_id': 'example', 'case_id': 'decode', 'requests': 2,
                'successful_requests': 1, 'failed_requests': 1,
                'token_accounting_complete': False, 'duration_s': 2,
                'publication_blockers': ['missing usage']}]))
            out = Path(tmp) / 'report'
            command = [sys.executable, str(SCRIPT), str(root), '--out', str(out)]
            first = subprocess.run(command, capture_output=True, text=True)
            self.assertEqual(first.returncode, 0, first.stderr)
            report = json.loads(out.with_suffix('.json').read_text())
            self.assertFalse(report['publication_ready'])
            self.assertEqual(report['startup_outcomes'][0]['error'], 'OOM')
            self.assertIsNone(report['runs'][0]['cases'][0]['output_tokens_s'])
            self.assertEqual(report['runs'][0]['quality'], {'example': 'missing'})
            self.assertIn('unknown', out.with_suffix('.md').read_text())
            self.assertNotEqual(subprocess.run(command, capture_output=True).returncode, 0)

    def test_missing_root_is_not_a_successful_empty_campaign(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / 'report'
            subprocess.run([sys.executable, str(SCRIPT), str(Path(tmp) / 'missing'),
                            '--out', str(out)], check=True, capture_output=True)
            report = json.loads(out.with_suffix('.json').read_text())
            self.assertEqual(report['startup_outcomes'][0]['status'], 'not_run')
            self.assertEqual(report['runs'], [])
