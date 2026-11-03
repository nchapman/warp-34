import unittest
from w34perf.presets import apply_preset


class PresetTests(unittest.TestCase):
    def setUp(self):
        self.cfg = {'run': {'repetitions': 3}, 'cases': [{'id': 'decode'}],
                    'presets': {'quick': {'cases': ['decode'], 'repetitions': 1},
                                'release': {'cases': ['decode'], 'repetitions': 5}}}

    def test_explicit_selection_does_not_mutate_original(self):
        result, ids = apply_preset(self.cfg, 'quick')
        self.assertEqual(result['run']['repetitions'], 1)
        self.assertEqual(ids, ['decode'])
        self.assertEqual(self.cfg['run']['repetitions'], 3)
        self.assertEqual(apply_preset(self.cfg, None), (self.cfg, None))

    def test_config_load_rejects_invalid_preset(self):
        from pathlib import Path
        import tempfile
        from w34perf.config import load
        suite = Path(__file__).resolve().parents[1] / 'suite.toml'
        # Keep fixture resolution relative to the original suite directory.
        text = suite.read_text().replace('fixture = "fixtures/',
                                         f'fixture = "{suite.parent}/fixtures/')
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'suite.toml'
            path.write_text(text + '\n[presets.quick]\ncases = ["absent"]\nrepetitions = 1\n')
            with self.assertRaisesRegex(ValueError, 'preset cases'):
                load(path)

    def test_missing_or_invalid_presets_fail(self):
        with self.assertRaises(ValueError):
            apply_preset(self.cfg, 'full')
        for preset in ({'cases': ['absent'], 'repetitions': 3},
                       {'cases': ['decode'], 'repetitions': True},
                       {'cases': ['decode', 'decode'], 'repetitions': 3},
                       {'cases': ['decode'], 'repetitions': 2}):
            self.cfg['presets']['release'] = preset
            with self.assertRaises(ValueError):
                apply_preset(self.cfg, 'release')
