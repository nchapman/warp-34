"""Check integration settings only; benchmarks/scoring remain entirely upstream."""
from argparse import Namespace
import json
from pathlib import Path
import unittest

from lm_eval.config.evaluate_config import EvaluatorConfig
from lm_eval.models.openai_completions import LocalChatCompletion

ROOT = Path(__file__).resolve().parents[1]


class UpstreamConfigTests(unittest.TestCase):
    def setUp(self):
        self.config = EvaluatorConfig.from_cli(Namespace(
            config=str(ROOT / 'lm-eval.yaml'), output_path='/tmp/quality-config-test'))

    def test_full_upstream_tasks_without_evaluation_overrides(self):
        cfg = self.config
        self.assertEqual(cfg.tasks, ['ifeval', 'gsm8k'])
        for key in ('limit', 'samples', 'num_fewshot', 'system_instruction',
                    'include_path', 'use_cache'):
            self.assertIsNone(getattr(cfg, key), key)
        self.assertFalse(cfg.cache_requests)
        self.assertFalse(cfg.predict_only)
        self.assertTrue(cfg.log_samples)
        self.assertEqual(cfg.gen_kwargs, {'chat_template_kwargs': {'enable_thinking': False}})

    def test_native_smoke_limit_leaves_full_run_default_and_tasks_unchanged(self):
        for limit in (10, 50):
            with self.subTest(limit=limit):
                smoke = EvaluatorConfig.from_cli(Namespace(
                    config=str(ROOT / 'lm-eval.yaml'), output_path='/tmp/quality-smoke-test',
                    limit=limit))
                self.assertEqual(smoke.limit, limit)
                self.assertIsNone(self.config.limit)
                self.assertEqual(smoke.tasks, self.config.tasks)
                self.assertEqual(smoke.gen_kwargs, self.config.gen_kwargs)
                self.assertEqual(smoke.num_fewshot, self.config.num_fewshot)
                self.assertTrue(smoke.log_samples)

    def test_chat_backend_sends_messages_without_client_side_template_rendering(self):
        self.assertEqual(self.config.model, 'local-chat-completions')
        self.assertTrue(self.config.apply_chat_template)
        self.assertIsNone(self.config.model_args['tokenizer_backend'])
        self.assertEqual(self.config.model_args['max_length'], 4096)
        model = LocalChatCompletion(**self.config.model_args)
        messages = [{'role': 'user', 'content': 'An unchanged task prompt.'}]
        self.assertEqual(json.loads(model.apply_chat_template(messages).prompt), messages)

    def test_upstream_payload_preserves_task_generation_settings(self):
        model = LocalChatCompletion(**self.config.model_args)
        messages = [{'role': 'user', 'content': 'An unchanged task prompt.'}]
        # Exercise upstream adapter behavior, not our own payload implementation.
        payload = model._create_payload(messages, generate=True, gen_kwargs={
            'max_gen_toks': 1280, 'temperature': 0.0, 'do_sample': False,
            'until': [], **self.config.gen_kwargs})
        self.assertEqual(payload['messages'], messages)
        self.assertEqual(payload['max_tokens'], 1280)
        self.assertEqual(payload['temperature'], 0.0)
        self.assertEqual(payload['stop'], [])
        self.assertEqual(payload['chat_template_kwargs'], {'enable_thinking': False})
        # gsm8k has no max_gen_toks in its task YAML: retain the API default.
        self.assertEqual(model._create_payload(messages, generate=True,
                                              gen_kwargs={})['max_tokens'], 256)


if __name__ == '__main__':
    unittest.main()
