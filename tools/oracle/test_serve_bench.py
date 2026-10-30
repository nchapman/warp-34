"""CPU-only HTTP boundary tests: no listener, server process, or GPU."""
import http.client
import io
import json
import threading
import tempfile
from pathlib import Path
import unittest
from unittest.mock import patch

from tools.oracle import serve_bench as bench


class Fragmented(io.RawIOBase):
    def __init__(self, data):
        self.data = io.BytesIO(data)

    def readable(self):
        return True

    def readinto(self, buffer):
        data = self.data.read(min(len(buffer), 3))
        buffer[:len(data)] = data
        return len(data)


class Socket:
    def __init__(self, data):
        self.data = data

    def makefile(self, *args):
        return io.BufferedReader(Fragmented(self.data))


def response(body):
    raw = (b'HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nContent-Length: '
           + str(len(body)).encode() + b'\r\n\r\n' + body)
    result = http.client.HTTPResponse(Socket(raw))
    result.begin()
    return result


def event(tokens, finish=None):
    return {"choices": [{"index": 0, "text": "".join(tokens),
                         "logprobs": {"tokens": tokens}, "finish_reason": finish}]}


def wire(events, done=True):
    text = ': keepalive\r\n\r\n' + ''.join('data: ' + json.dumps(e) + '\r\n\r\n' for e in events)
    return (text + ('data: [DONE]\r\n\r\n' if done else '')).encode()


class BenchmarkTests(unittest.TestCase):
    def run_stream(self, body):
        payload = {"prompt": "test", "max_tokens": 2}
        with patch.object(bench.urllib.request, 'urlopen', return_value=response(body)) as opened:
            result = bench.request_one('http://localhost:1', payload, 600, threading.Barrier(1))
        req = opened.call_args.args[0]
        self.assertEqual(req.full_url, 'http://localhost:1/v1/completions')
        self.assertEqual(json.loads(req.data), payload)
        return result

    def events(self, burst=False, count=2):
        chunks = [event(['a', 'b'])] if burst else [event(['a']), event(['b'])]
        return chunks + [event([], 'length'), {"choices": [], "usage": {"completion_tokens": count}}]

    def test_valid_fragmented_two_tokens(self):
        result = self.run_stream(wire(self.events()))
        self.assertTrue(result['passed'], result['errors'])
        self.assertEqual(result['selected_tokens'], 2)
        self.assertTrue(result['first_event_before_finish'])
        self.assertIsNotNone(result['ttft_s'])

    def test_burst(self):
        result = self.run_stream(wire(self.events(burst=True)))
        self.assertTrue(result['passed'], result['errors'])
        self.assertEqual(result['mean_itl_s'], 0)
        self.assertEqual(result['p95_itl_s'], 0)

    def test_rejected_streams(self):
        cases = [b'data: {bad}\n\n', wire(self.events(), done=False),
                 wire(self.events(count=3)), wire(self.events()[:-1]),
                 wire(self.events()) + b'data: [DONE]\n\n',
                 wire([event(['a'])] + self.events()[2:]),
                 wire(self.events()[:2] + [event([], 'length'), event([], 'length')] + self.events()[-1:])]
        unsupported = self.events()
        unsupported[0]['choices'][0]['logprobs'] = None
        cases.append(wire(unsupported))
        for body in cases:
            with self.subTest(body=body):
                self.assertFalse(self.run_stream(body)['passed'])

    def test_warmups_preserve_protocol_and_are_separate_from_measured_batches(self):
        opened_payloads = []
        def fake(req, **kwargs):
            opened_payloads.append(json.loads(req.data))
            return response(wire(self.events()))
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory) / 'report.json'
            with patch.object(bench.urllib.request, 'urlopen', side_effect=fake):
                code = bench.main(['--base-url', 'http://localhost:1', '--model', 'model',
                                   '--out', str(out), '--concurrency', '1', '2',
                                   '--tokens', '2', '--warmup-repetitions', '1'])
            report = json.loads(out.read_text())
        self.assertEqual(code, 0)
        self.assertEqual(len(opened_payloads), 6)
        self.assertEqual(len(report['warmups']), 1)
        self.assertEqual([b['concurrency'] for b in report['batches']], [1, 2])
        self.assertEqual([b['workload_sha256'] for b in report['warmups'][0]],
                         [b['workload_sha256'] for b in report['batches']])

    def test_failed_warmup_is_not_hidden_by_passing_measurement(self):
        bodies = iter([wire(self.events(count=3)), wire(self.events())])
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory) / 'report.json'
            with patch.object(bench.urllib.request, 'urlopen',
                              side_effect=lambda *a, **k: response(next(bodies))):
                code = bench.main(['--base-url', 'http://localhost:1', '--model', 'model',
                                   '--out', str(out), '--concurrency', '1',
                                   '--tokens', '2', '--warmup-repetitions', '1'])
            report = json.loads(out.read_text())
        self.assertEqual(code, 1)
        self.assertTrue(report['batches'][0]['passed'])
        self.assertFalse(report['passed'])
        self.assertFalse(report['warmups'][0][0]['passed'])

    def test_batch_rejects_aggregate_and_hash_is_stable(self):
        def fake(*args, **kwargs):
            return response(wire(self.events(count=3)))
        with patch.object(bench.urllib.request, 'urlopen', side_effect=fake):
            a = bench.run_batch('http://localhost:1', 'model', 2, 2, 600, ['a', 'b'])
            b = bench.run_batch('http://localhost:2', 'model', 2, 2, 600, ['a', 'b'])
        self.assertFalse(a['passed'])
        self.assertIsNone(a['generated_tokens_per_s'])
        self.assertEqual(a['workload_sha256'], b['workload_sha256'])


if __name__ == '__main__':
    unittest.main()
