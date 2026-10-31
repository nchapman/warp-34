import unittest

from w34perf.metrics import cached_tokens, summarize
from test_perf import result


class LongContextMetricsTests(unittest.TestCase):
    def test_cached_tokens_rejects_invalid_counts(self):
        for count in (-1, 51, True, 1.5, "12", None):
            with self.subTest(count=count):
                record = result()
                record["usage"]["prompt_tokens_details"] = {"cached_tokens": count}
                self.assertIsNone(cached_tokens(record))
        for count in (0, 50):
            record = result()
            record["usage"]["prompt_tokens_details"] = {"cached_tokens": count}
            self.assertEqual(cached_tokens(record), count)

    def test_step_breakdown_keeps_missing_cache_usage_unknown(self):
        first, second = result(), result()
        first["step"] = second["step"] = 1
        first["first_output_ms"], second["first_output_ms"] = 10, 30
        first["usage"]["prompt_tokens_details"] = {"cached_tokens": 40}
        summary = summarize([first, second], [], 1, {"kind": "agent_replay"})
        step = summary["steps"][0]
        self.assertEqual(step["step"], 1)
        self.assertEqual(step["requests"], 2)
        self.assertEqual(step["successful_requests"], 2)
        self.assertEqual(step["prompt_tokens"]["p50"], 50)
        self.assertEqual(step["first_output_ms"]["p50"], 20)
        self.assertIsNone(step["cached_tokens"])
        self.assertIsNone(step["uncached_tokens"])

    def test_step_breakdown_retains_failure_denominator(self):
        good, failed = result(), result()
        good["step"] = failed["step"] = 2
        failed.update(status="timeout", request_id="failed")
        good["usage"]["prompt_tokens_details"] = {"cached_tokens": 40}
        step = summarize([good, failed], [], 1, {"kind": "agent_replay"})["steps"][0]
        self.assertEqual(step["requests"], 2)
        self.assertEqual(step["successful_requests"], 1)
        self.assertEqual(step["cached_tokens"]["p50"], 40)
        self.assertEqual(step["uncached_tokens"]["p50"], 10)
