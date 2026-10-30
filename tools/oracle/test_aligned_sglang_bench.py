"""CPU-only protocol checks; no torch, transformers, SGLang or GPU imports."""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock
import os
import sys
import types
import aligned_worker_hook as hook

spec = importlib.util.spec_from_file_location(
    'aligned', Path(__file__).with_name('aligned_sglang_bench.py'))
a = importlib.util.module_from_spec(spec)
spec.loader.exec_module(a)


class Tokenizer:
    def encode(self, text):
        assert text == a.PHRASE
        return [101, 5, 8]


class AlignedProtocolTests(unittest.TestCase):
    def draft(self, root, name):
        source = root / name
        source.mkdir()
        raw = dict(architectures=['DFlash2DraftModel'], transformer_layer_config={},
                   dtype='bfloat16', block_size=16, aux_hidden_state_layer_ids=[1])
        raw.update({k: 1 for k in ('conv_kernel_size', 'conv_group_size',
                    'selector_rank', 'selector_top_k', 'mask_token_id', 'sample_from_anchor')})
        (source / 'config.json').write_text(json.dumps(raw))
        (source / 'model.safetensors').write_bytes(name.encode())
        return source

    def test_normalize_refuses_reuse(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            first, second = self.draft(root, 'first'), self.draft(root, 'second')
            out = root / 'out'
            hook.normalize_draft(first, out)
            before = (out / 'config.json').read_bytes()
            for source in (second, first):
                with self.assertRaises(ValueError):
                    hook.normalize_draft(source, out)
                self.assertEqual((out / 'model.safetensors').read_bytes(), b'first')
                self.assertEqual((out / 'config.json').read_bytes(), before)

    def test_normalize_architecture_explicit_error(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            source = self.draft(root, 'source')
            (source / 'config.json').write_text('{"architectures": ["wrong"]}')
            with self.assertRaises(ValueError):
                hook.normalize_draft(source, root / 'out')

    def patch_tree(self, root):
        originals = {}
        for name in ('srt/models/dflash.py', 'srt/configs/model_config.py',
                     'srt/speculative/dflash_worker_v2.py'):
            path = root / name
            path.parent.mkdir(parents=True, exist_ok=True)
            originals[path] = (b'# fake source\r\n        # --- 1) Draft a fixed block with the draft model.\n'
                              b'        pass\n        # --- 2) Target verify.\n')
            import hashlib
            patcher = mock.patch.object(hook, 'DFLASH_SOURCE_SHA256', hashlib.sha256(originals[path]).hexdigest())
            patcher.start()
            self.addCleanup(patcher.stop)
            path.write_bytes(originals[path])
        return originals

    def test_inject_restores_normal_and_exception(self):
        for fail in (False, True):
            with self.subTest(fail=fail), tempfile.TemporaryDirectory() as d:
                originals = self.patch_tree(Path(d))
                try:
                    with hook.inject(d):
                        for path, original in originals.items():
                            self.assertNotEqual(path.read_bytes(), original)
                        if fail:
                            raise RuntimeError('body failed')
                except RuntimeError:
                    self.assertTrue(fail)
                for path, original in originals.items():
                    self.assertEqual(path.read_bytes(), original)

    def test_inject_restores_partial_failure(self):
        with tempfile.TemporaryDirectory() as d:
            originals = self.patch_tree(Path(d))
            missing = list(originals)[1]
            missing.unlink()
            with self.assertRaises(FileNotFoundError):
                with hook.inject(d):
                    self.fail('must not enter')
            for path, original in originals.items():
                if path != missing:
                    self.assertEqual(path.read_bytes(), original)

    def test_execute_restores_constructor_and_generation_failure(self):
        for constructor_failure in (True, False):
            with self.subTest(constructor_failure=constructor_failure), tempfile.TemporaryDirectory() as d:
                root = Path(d)
                originals = self.patch_tree(root)
                shutdown = []
                test = self
                class Engine:
                    def __init__(self, **settings):
                        test.assertIsNone(settings['quantization'])
                        for path, original in originals.items():
                            test.assertNotEqual(path.read_bytes(), original)
                        if constructor_failure:
                            raise RuntimeError('constructor failed')
                    def generate(self, **kwargs):
                        raise RuntimeError('generation failed')
                    def shutdown(self):
                        for path, original in originals.items():
                            test.assertNotEqual(path.read_bytes(), original)
                        shutdown.append(True)
                args = types.SimpleNamespace(out=root / 'runtime.json', target=root,
                                             draft=self.draft(root, 'draft'))
                data = dict(protocol=a.protocol(Tokenizer()))
                with mock.patch.dict(sys.modules, {'sglang': types.SimpleNamespace(Engine=Engine)}), mock.patch.dict(
                        os.environ, {'ORACLE_EVENTS': 'original-events'}, clear=False):
                    before = {k: os.environ.get(k) for k in ('ORACLE_EVENTS', 'SGLANG_MAMBA_CONV_DTYPE')}
                    with self.assertRaises(RuntimeError):
                        a.execute(args, data, root)
                    self.assertEqual(before, {k: os.environ.get(k) for k in before})
                self.assertEqual(shutdown, [] if constructor_failure else [True])
                for path, original in originals.items():
                    self.assertEqual(path.read_bytes(), original)
                self.assertFalse(data['phase6_passed'])

    def test_prompt_repeats_ids_not_text(self):
        ids = a.prompt_ids(Tokenizer(), 128)
        self.assertEqual(ids, ([101, 5, 8] * 43)[:128])
        self.assertEqual(a.prompt_ids(Tokenizer(), 16), ([101, 5, 8] * 6)[:16])
        self.assertEqual(ids, a.prompt_ids(Tokenizer(), 128))

    def test_protocol_has_no_invented_runtime(self):
        p = a.protocol(Tokenizer())
        self.assertEqual(p['output_tokens'], 128)
        self.assertEqual(p['decode_commits'], 127)
        self.assertEqual(p['warm_output_tokens'], 32)
        self.assertTrue(p['sampling']['ignore_eos'])
        self.assertEqual(p['sampling']['temperature'], 0)
        self.assertFalse(p['prefix_cache'])
        self.assertTrue(p['reset_after_warm'])
        self.assertEqual((p['tp'], p['M']), (1, 16))
        self.assertIsNone(p['verify_rounds'])
        self.assertIsNone(p['output_ids'])
        self.assertNotEqual(p['kv_policy']['oracle'], p['kv_policy']['w34'])

    def test_snapshot_fail_closed(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            with self.assertRaises(ValueError):
                a.snapshot(root, a.TARGET)
            p = root / ('models--' + a.TARGET.replace('/', '--')) / 'snapshots' / ('a' * 40)
            p.mkdir(parents=True)
            (p / 'config.json').write_text('{}')
            with self.assertRaises(ValueError):
                a.snapshot(p, a.TARGET)
            (p / 'model.safetensors').write_bytes(b'test fixture, not real weights')
            info = a.snapshot(p, a.TARGET)
            self.assertEqual(info['snapshot_sha'], 'a' * 40)
            self.assertEqual(info['metadata_sha256']['config.json'], a.sha(p / 'config.json'))
            (p / 'model.safetensors.index.json').write_text(json.dumps(
                {'weight_map': {'x': 'missing.safetensors'}}))
            with self.assertRaises(ValueError):
                a.snapshot(p, a.TARGET)

    def test_version_and_source_fail_closed(self):
        with self.assertRaisesRegex(ValueError, 'require sglang'):
            a.evidence(Path('/nonexistent'), '0.5.18')
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            for name, needles in a.SOURCES.items():
                p = root / name
                p.parent.mkdir(parents=True, exist_ok=True)
                p.write_text('\n'.join(needles))
            self.assertEqual(set(a.evidence(root, '0.5.19')), set(a.SOURCES))
            (root / next(iter(a.SOURCES))).write_text('changed contract')
            with self.assertRaisesRegex(ValueError, 'unsupported source contract'):
                a.evidence(root, '0.5.19')

    def test_profile_refused_before_importing_runtime(self):
        with self.assertRaises(SystemExit) as exc:
            a.main(['--target', '/missing', '--draft', '/missing',
                    '--out', '/unused', '--profile'])
        self.assertEqual(exc.exception.code, 2)


if __name__ == '__main__':
    unittest.main()
