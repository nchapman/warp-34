"""Frozen campaign configuration and fixture evidence consistency."""
import hashlib
import json
import shlex
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
                     '27b-swe-multiclient', '35b-swe-multiclient', 'flash-next-swe-multiclient',
                     'flash-next-serving', 'flash-next-hgptq', 'flash-next-longctx',
                     'decision-models'):
            cfg = load(ROOT / 'campaigns' / (name + '.toml'))
            for tier in ('quick', 'full', 'release'):
                with self.subTest(campaign=name, tier=tier):
                    preset, cases = apply_preset(cfg, tier)
                    engines, selected, coverage = select(preset, case_ids=cases)
                    self.assertTrue(engines)
                    self.assertEqual({c['id'] for c in selected}, set(cases))
                    self.assertEqual(len(coverage), len(cfg['engines']) * len(cases))

    def test_primary_suite_and_swe_multiclient_campaigns_cover_all_models_and_modes(self):
        suite = load(ROOT / 'suite.toml')
        w34_engines = [e for e in suite['engines'] if e['engine'] == 'warp-34']
        w34_models = {e['model'] for e in w34_engines}
        for family in ('Qwen3.5-4B', 'Qwen3.8-27B', 'Qwen3.6-35B-A3B', 'Qwen3.8-Flash-Next'):
            self.assertTrue(any(family in m for m in w34_models), f'missing {family} in suite.toml')
        self.assertEqual({e['gpu_count'] for e in w34_engines}, {1, 2})
        self.assertTrue({'none', 'mtp-3', 'dflash2'}.issubset({e['speculation'] for e in w34_engines}))
        suite_cases = {c['id'] for c in suite['cases']}
        self.assertTrue({'swe-c4-shared', 'swe-c4-unique', 'swe-c8-burst'}.issubset(suite_cases))

        expected_modes = {
            '27b-swe-multiclient': {'none', 'mtp-3', 'dflash2-m8'},
            '35b-swe-multiclient': {'none', 'mtp-3'},
            'flash-next-swe-multiclient': {'none', 'mtp-3', 'dflash2'},
        }
        for name, modes in expected_modes.items():
            with self.subTest(campaign=name):
                cfg = load(ROOT / 'campaigns' / (name + '.toml'))
                w34 = [e for e in cfg['engines'] if e['engine'] == 'warp-34']
                for tp in (1, 2):
                    tp_modes = {e['speculation'] for e in w34 if e['gpu_count'] == tp}
                    self.assertEqual(tp_modes, modes, f'{name} TP={tp} modes mismatch')
                cases = {c['id']: c for c in cfg['cases']}
                self.assertEqual(set(cases), {'smoke-warmup', 'swe-c4-shared', 'swe-c4-unique', 'swe-c8-burst'})
                self.assertEqual(cases['swe-c4-shared']['concurrency'], 4)
                self.assertEqual(cases['swe-c4-unique']['concurrency'], 4)
                self.assertEqual(cases['swe-c8-burst']['concurrency'], 8)

    def test_decision_models_campaign_covers_glide_d3_causallm_and_jevbench(self):
        cfg = load(ROOT / 'campaigns' / 'decision-models.toml')
        models = {e['model'] for e in cfg['engines']}
        for expected in (
            'fastino/GLiDE-0.8B',
            'fastino/GLiDE-2B',
            'fastino/GLiDE-9B',
            'fastino/GLiDE-27B',
            'vllm-sr/d3-flash',
            'vllm-sr/d3-turbo',
            'vllm-sr/d3-ultra',
            'nchapman/Qwen3.5-4B-W4A16-GPTQ',
            'nchapman/Qwen3.8-27B-W4A16-GPTQ',
        ):
            self.assertIn(expected, models)
        cases = {c['id']: c for c in cfg['cases']}
        self.assertEqual(len(cases['jevbench-full-128-c1']['fixture_data']['requests']), 128)
        self.assertEqual(len(cases['jevbench-multiq-c1']['fixture_data']['requests']), 16)
        self.assertEqual(cases['jevbench-single-c1-p2']['order_passes'], 2)

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

    def test_flash_hgptq_pins_and_launch_matrix(self):
        cfg = load(ROOT / 'campaigns/flash-next-hgptq.toml')
        target_rev = '6ee06d2936b111f19eb5eb331044cb8b8b28562b'
        draft_rev = '24a62babba3b54387b3ea73f5599e5de83029dd8'
        root = '/workspace/.hf_home/hub/models--nchapman--'
        target = root + 'Qwen3.8-Flash-Next-W4A16-HGPTQ/snapshots/' + target_rev
        draft = root + 'Qwen3.8-Flash-Next-DFlash2-W4A16-HGPTQ/snapshots/' + draft_rev
        self.assertEqual(len(cfg['engines']), 6)
        self.assertEqual({(e['gpu_count'], e['speculation']) for e in cfg['engines']},
                         {(tp, mode) for tp in (1, 2) for mode in ('none', 'mtp-k3', 'dflash2')})
        for engine in cfg['engines']:
            with self.subTest(engine=engine['id']):
                self.assertEqual(engine['engine'], 'warp-34')
                self.assertEqual(engine['version'], 'b7f5fdd')
                self.assertEqual(engine['support'], 'supported')
                self.assertEqual(engine['model'], target)
                self.assertEqual(engine['model_revision'], target_rev)
                self.assertEqual(engine['tokenizer_revision'], target_rev)
                for key in ('artifact_sha256', 'chat_template_sha256'):
                    self.assertRegex(engine[key], r'^[0-9a-f]{64}$')
                self.assertRegex(engine['binary_sha256'], r'^[0-9a-f]{64}$')
                self.assertEqual(engine['activation_dtype'], 'float16')
                self.assertEqual(engine['base_url'], 'http://127.0.0.1:18034/v1')
                self.assertEqual(engine['kv_format'], 'tq4')
                args = shlex.split(engine['launch_command'])
                self.assertIn('/workspace/warp-34/target/release/w34', args)
                self.assertIn('--offline', args)
                expected = {'--model': target, '--tp': str(engine['gpu_count']),
                            '--kv-format': 'tq4', '--kv-pages': '512',
                            '--max-num-seqs': '4', '--host': '127.0.0.1', '--port': '18034'}
                mode = engine['speculation']
                if mode == 'dflash2':
                    expected.update({'--speculative': 'dflash2', '--draft-model': draft})
                    self.assertEqual(engine['draft_model'], draft)
                    self.assertEqual(engine['draft_model_revision'], draft_rev)
                    self.assertRegex(engine['draft_artifact_sha256'], r'^[0-9a-f]{64}$')
                    self.assertEqual(engine['draft_block_size'], 8)
                    self.assertEqual(engine['draft_activation_dtype'], 'bfloat16')
                else:
                    self.assertNotIn('--draft-model', args)
                    self.assertNotIn('draft_model', engine)
                    self.assertNotIn('draft_model_revision', engine)
                if mode == 'mtp-k3':
                    expected.update({'--speculative': 'mtp', '--mtp-k': '3'})
                else:
                    self.assertNotIn('--mtp-k', args)
                if mode == 'none':
                    self.assertNotIn('--speculative', args)
                for flag, value in expected.items():
                    self.assertEqual(args.count(flag), 1)
                    self.assertEqual(args[args.index(flag) + 1], value)

    def test_flash_hgptq_reuses_historical_workloads(self):
        cfg = load(ROOT / 'campaigns/flash-next-hgptq.toml')
        historical = load(ROOT / 'campaigns/flash-next-serving.toml')
        # Includes loaded fixture contents and hashes, not just workload names.
        self.assertEqual(cfg['cases'], historical['cases'])
        self.assertEqual(len(cfg['cases']), 5)
        self.assertEqual(cfg['run'], historical['run'])
        for tier, count, repetitions in (('quick', 2, 1), ('full', 5, 3), ('release', 5, 5)):
            self.assertEqual(cfg['presets'][tier], historical['presets'][tier])
            preset, cases = apply_preset(cfg, tier)
            self.assertEqual(len(cases), count)
            self.assertEqual(preset['run']['repetitions'], repetitions)
