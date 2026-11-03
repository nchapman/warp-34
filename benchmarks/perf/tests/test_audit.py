"""Historical evidence reconciliation does not promote runs to publication."""
import json
from pathlib import Path
import tempfile
import unittest

from w34perf.audit import audit


class AuditTests(unittest.TestCase):
    def test_counts_only_measured_rows_not_unselected_coverage(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / 'manifest.json').write_text(json.dumps({'status': 'completed'}))
            rows = [{'engine_id': 'mtp', 'case_id': 'decode', 'repetition': i,
                     'requests': 4, 'successful_requests': 4, 'failed_requests': 0,
                     'publication_blockers': ['quality missing']} for i in range(2)]
            (root / 'summary.json').write_text(json.dumps(rows))
            (root / 'coverage.json').write_text(json.dumps([
                {'engine_id': 'other', 'status': 'not_selected'}]))
            result = audit(root)
            self.assertEqual(result['totals']['successful_requests'], 8)
            cell = result['bundles'][0]['cells'][0]
            self.assertEqual(cell['repetitions'], [0, 1])
            self.assertEqual(cell['publication_blockers'], ['quality missing'])
            self.assertEqual(len(result['bundles'][0]['cells']), 1)
            self.assertEqual(result['bundles'][0]['coverage'][0]['status'], 'not_selected')
            self.assertFalse(result['publication_certified'])

    def test_incomplete_and_invalid_bundles_are_retained(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for name in ('missing', 'invalid'):
                p = root / name
                p.mkdir()
                (p / 'manifest.json').write_text('{}')
            (root / 'invalid' / 'summary.json').write_text('broken')
            result = audit(root)
            self.assertEqual(len(result['bundles']), 2)
            self.assertTrue(all(b['errors'] for b in result['bundles']))
            self.assertEqual(result['totals']['requests'], 0)

    def test_hashes_identify_inputs_and_failed_requests_remain(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / 'manifest.json').write_text('{}')
            (root / 'summary.json').write_text(json.dumps([
                {'engine_id': 'e', 'case_id': 'c', 'repetition': 0,
                 'requests': 2, 'successful_requests': 1, 'failed_requests': 1}]))
            first = audit(root)
            self.assertEqual(first['totals']['failed_requests'], 1)
            (root / 'manifest.json').write_text('{"status":"failed"}')
            second = audit(root)
            self.assertNotEqual(first['bundles'][0]['sha256']['manifest.json'],
                                second['bundles'][0]['sha256']['manifest.json'])

    def test_malformed_nested_evidence_retained(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            variants = [{field: value} for field in ('config', 'provenance')
                        for value in (None, [], 'bad')]
            variants += [{'config': {field: value}} for field in ('engines', 'cases')
                         for value in (None, {}, [None, 'bad'])]
            for i, manifest in enumerate(variants):
                path = root / str(i)
                path.mkdir()
                (path / 'manifest.json').write_text(json.dumps(manifest))
                (path / 'summary.json').write_text('[]')
                (path / 'coverage.json').write_text('[]')
            result = audit(root)
            self.assertEqual(len(result['bundles']), len(variants))
            self.assertTrue(all(b['errors'] for b in result['bundles']))
            for coverage in ({}, [None, 'bad']):
                (path / 'manifest.json').write_text('{}')
                (path / 'coverage.json').write_text(json.dumps(coverage))
                bundle = next(b for b in audit(root)['bundles'] if b['path'] == path.name)
                self.assertTrue(any('coverage' in e for e in bundle['errors']))
