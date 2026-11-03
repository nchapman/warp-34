"""Frozen campaign configuration and fixture evidence consistency."""
import hashlib
import json
from pathlib import Path
import unittest

from w34perf.config import load
from w34perf.presets import apply_preset
from w34perf.quality import _checks
from w34perf.selection import select

ROOT = Path(__file__).resolve().parents[1]


class CampaignTests(unittest.TestCase):
    def test_completion_campaigns_and_all_presets_plan_without_network(self):
        for name in ('27b-completion', '27b-kv-spec-completion', '27b-capacity-completion',
                     'flash-next-serving', 'flash-next-longctx'):
            cfg = load(ROOT / 'campaigns' / (name + '.toml'))
            for tier in ('quick', 'full', 'release'):
                with self.subTest(campaign=name, tier=tier):
                    preset, cases = apply_preset(cfg, tier)
                    engines, selected, coverage = select(preset, case_ids=cases)
                    self.assertTrue(engines)
                    self.assertEqual({c['id'] for c in selected}, set(cases))
                    self.assertEqual(len(coverage), len(cfg['engines']) * len(cases))

    def test_capacity_calibration_identifies_current_fixture_and_increasing_lengths(self):
        fixture = ROOT / 'fixtures/quality-capacity.json'
        calibration = json.loads((ROOT / 'qualification/27b-capacity-calibration.json').read_text())
        self.assertEqual(calibration['fixture_sha256'], hashlib.sha256(fixture.read_bytes()).hexdigest())
        prepared = _checks(json.loads(fixture.read_text()))
        self.assertEqual([c['id'] for c, _ in prepared], [c['check'] for c in calibration['checks']])
        counts = [c['reference_template_prompt_tokens'] for c in calibration['checks']]
        self.assertTrue(all(32000 <= n < 33000 for n in counts[:2]))
        self.assertTrue(all(127000 <= n < 128000 for n in counts[2:4]))
        self.assertTrue(all(251000 <= n < 252000 for n in counts[4:]))
        self.assertIn('not server attestation', calibration['method'])

    def test_flash_capacity_calibration_uses_same_fixture_but_its_own_tokenizer(self):
        report = json.loads((ROOT / 'qualification/flash-capacity-calibration.json').read_text())
        fixture = ROOT / 'fixtures/quality-capacity.json'
        self.assertEqual(report['fixture_sha256'], hashlib.sha256(fixture.read_bytes()).hexdigest())
        self.assertEqual(report['model_revision'], 'ef554143369a706525336f6b42a09094835dc077')
        self.assertEqual(len(report['checks']), 6)
        self.assertTrue(all(32000 < c['reference_template_prompt_tokens'] < 262144
                            for c in report['checks']))

    def test_long_quality_fixture_remains_separate_from_baseline(self):
        short = json.loads((ROOT / 'fixtures/quality.json').read_text())
        long = json.loads((ROOT / 'fixtures/quality-long.json').read_text())
        self.assertEqual(max(c.get('context_lines', 0) for c in short['checks']), 256)
        self.assertEqual(max(c.get('context_lines', 0) for c in long['checks']), 1200)
