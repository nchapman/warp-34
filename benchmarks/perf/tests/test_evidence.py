import hashlib
from pathlib import Path
import tempfile
import unittest

from w34perf.evidence import attach


class EvidenceTests(unittest.TestCase):
    def test_copies_bytes_and_hashes_with_colliding_basenames(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            out = root / 'out'
            out.mkdir()
            sources = []
            for name, content in [('a', b'first'), ('b', b'second')]:
                folder = root / name
                folder.mkdir()
                source = folder / 'startup.log'
                source.write_bytes(content)
                sources.append(source)
            receipts = attach(sources, out)
            self.assertEqual(len(receipts), 2)
            for source, receipt in zip(sources, receipts):
                self.assertEqual((out / receipt['path']).read_bytes(), source.read_bytes())
                self.assertEqual(receipt['sha256'], hashlib.sha256(source.read_bytes()).hexdigest())
                self.assertNotIn(str(root), receipt['path'])
            sources[0].write_text('changed')
            self.assertEqual((out / receipts[0]['path']).read_bytes(), b'first')

    def test_missing_input_fails_before_creating_attachment_directory(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with self.assertRaises(ValueError):
                attach([root / 'missing'], root)
            self.assertFalse((root / 'evidence').exists())

    def test_run_rejects_missing_evidence_or_inactive_pid_sampling_before_output(self):
        from w34perf.config import load
        from w34perf.runner import execute
        cfg = load(Path(__file__).resolve().parents[1] / 'suite.toml')
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / 'run'
            for kwargs in ({'evidence_paths': [Path(tmp) / 'missing']},
                           {'server_pids': [123]}):
                with self.assertRaises(ValueError):
                    execute(cfg, out, ['warp34-1-fp16'], ['smoke'], **kwargs)
                self.assertFalse(out.exists())

    def test_quality_results_are_profile_bound_and_failures_are_not_hidden(self):
        import json
        from w34perf.config import digest
        from w34perf.evidence import quality_status
        engine = {'id': 'test', 'version': 'one'}
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            records = []
            for index, status in enumerate(('passed', 'failed')):
                path = root / f'{index}.json'
                path.write_text(json.dumps({'schema_version': 1,
                    'model_profile_sha256': digest(engine), 'status': status,
                    'checks': [{'status': status}]}))
                records.append({'path': path.name})
            self.assertEqual(quality_status(records[:1], root, engine), 'passed')
            self.assertEqual(quality_status(records, root, engine), 'failed')
            self.assertEqual(quality_status(records, root, {**engine, 'version': 'two'}), 'missing')
            self.assertEqual(quality_status([], root, engine), 'missing')

    def test_empty_input_does_not_create_directory(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.assertEqual(attach([], Path(tmp)), [])
            self.assertEqual(list(Path(tmp).iterdir()), [])
