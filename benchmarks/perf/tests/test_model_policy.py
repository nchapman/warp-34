"""Keep runnable benchmark profiles on approved checkpoints, not retired baselines."""
from pathlib import Path
import re
import shlex
import unittest

from w34perf.config import load

ROOT = Path(__file__).resolve().parents[1]
PINS = {
    'Qwen3.8-27B': ('Qwen3.8-27B-AWQ-INT4', 'e3a700906e6a73ba4290b1873dc2ba8198527ab5'),
    'Qwen3.6-35B-A3B': ('Qwen3.6-35B-A3B-AWQ-4bit', 'f1e8415c4f29508fe281da4e256df26573213bfd'),
}


def profiles():
    paths = [ROOT / 'suite.toml', *sorted((ROOT / 'campaigns').glob('*.toml')),
             *sorted((ROOT / 'qualification').glob('*.toml'))]
    for path in paths:
        for engine in load(path)['engines']:
            if engine.get('enabled', True):
                yield path.name, engine


class ModelPolicyTests(unittest.TestCase):
    def test_active_profiles_do_not_select_retired_models(self):
        for campaign, engine in profiles():
            with self.subTest(campaign=campaign, engine=engine['id']):
                selection = ' '.join(str(engine.get(k, '')) for k in
                                     ('model', 'draft_model', 'launch_command'))
                for retired in ('llmfan46', 'SergiioB', 'rcs1000', 'uncensored'):
                    self.assertNotIn(retired.lower(), selection.lower())
                self.assertNotRegex(selection, r'nchapman(?:/|--)[^\s"\']*-W\d+(?:\.\d+)?A\d+-GPTQ')
                if engine['engine'] == 'warp-34':
                    args = shlex.split(engine.get('launch_command', ''))
                    for flag in ('--model', '--draft-model'):
                        if flag not in args:
                            continue
                        model = args[args.index(flag) + 1]
                        if re.search(r'-W\d+(?:\.\d+)?A\d+-', model):
                            self.assertRegex(model, r'(?:models--nchapman--|^nchapman/)')
                            self.assertIn('-HGPTQ', model)

    def test_competitor_commands_and_metadata_agree_on_cyankiwi_pins(self):
        checked = set()
        for campaign, engine in profiles():
            command = engine.get('launch_command', '')
            if engine['engine'] not in ('vllm', 'sglang') or not command:
                continue
            family = next((f for f in PINS if f in command), None)
            if family is None:
                continue
            with self.subTest(campaign=campaign, engine=engine['id']):
                repo, revision = PINS[family]
                snapshot = f'models--cyankiwi--{repo}/snapshots/{revision}'
                self.assertIn(snapshot, command)
                self.assertEqual(engine['model_revision'], revision)
                self.assertEqual(engine['tokenizer_revision'], revision)
                self.assertIn('AWQ-int4', engine['weight_format'])
                self.assertIn('cyankiwi', engine['quantization_details'])
                self.assertRegex(engine['quantization_details'], r'(?:g|group[- ]?)32')
                # Never inherit the target HGPTQ checkpoint's hashes.
                self.assertEqual(engine['artifact_sha256'], 'unknown')
                self.assertEqual(engine['chat_template_sha256'], 'unknown')
                args = shlex.split(command)
                if '--served-model-name' in args:
                    model = args[args.index('--served-model-name') + 1]
                else:
                    model = args[args.index('serve') + 1]
                self.assertEqual(engine['model'], model)
                checked.add(family)
        self.assertEqual(checked, set(PINS))
