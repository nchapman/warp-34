import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import vision_matrix


class VisionMatrixTests(unittest.TestCase):
    def test_failed_gate_retains_metrics_and_capture_only_environment(self):
        self.check_reports({'passed': True, 'tensor_metrics': {'error': 0.25}}, 1)

    def test_unexpected_report_types_continue(self):
        for report in (None, [], True, 'bad', {'passed': 1}):
            with self.subTest(report=report):
                self.check_reports(report, 0)

    def check_reports(self, report, status):
        calls = []
        def run(command, **kwargs):
            if command[0] == 'cargo':
                return SimpleNamespace(returncode=0)
            calls.append((command, kwargs))
            if 'vision_gate' in command[0]:
                Path(command[-1]).write_text(json.dumps(report))
                return SimpleNamespace(returncode=status)
            return SimpleNamespace(returncode=0)
        with tempfile.TemporaryDirectory() as tmp:
            with patch('sys.argv', ['vision_matrix', '--out-dir', tmp, '--models', 'test']), patch.dict('os.environ', {'CUDA_VISIBLE_DEVICES': '0'}), patch.object(vision_matrix.subprocess, 'run', side_effect=run):
                self.assertEqual(vision_matrix.main(), 1)
            results = json.loads((Path(tmp) / 'summary.json').read_text())['results']
            self.assertEqual(len(results), 2)
            self.assertTrue(all(not r['passed'] for r in results))
            self.assertTrue(all(r['measurement'] == report for r in results))
            for command, kwargs in calls:
                if 'vision_gate' in command[0]:
                    self.assertNotIn('VLLM_ALLOW_INSECURE_SERIALIZATION', kwargs.get('env') or {})
                else:
                    self.assertEqual(kwargs['env']['VLLM_ALLOW_INSECURE_SERIALIZATION'], '1')

    def test_nonempty_directory_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            (Path(tmp) / 'model0-png.json').write_text('{"passed": true}')
            with patch('sys.argv', ['vision_matrix', '--out-dir', tmp]), patch.dict('os.environ', {'CUDA_VISIBLE_DEVICES': '0'}), patch.object(vision_matrix.subprocess, 'run') as run:
                with self.assertRaises(SystemExit):
                    vision_matrix.main()
                run.assert_not_called()
