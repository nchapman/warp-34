"""CPU-only lifecycle and native-artifact contract tests."""
import importlib.util
import json
import os
from pathlib import Path
import socket
import signal
import io
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch, Mock

SPEC = importlib.util.spec_from_file_location('runner', Path(__file__).resolve().parents[1] / 'run_suite.py')
r = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(r)

FIXTURE = '''import sys,json
from pathlib import Path
from http.server import BaseHTTPRequestHandler,HTTPServer
if sys.argv[1]=='server':
 class Handler(BaseHTTPRequestHandler):
  def do_GET(self):
   self.send_response(200); self.end_headers()
   self.wfile.write(b'{"data":[{"id":"warp34-quality"}]}')
 HTTPServer(('127.0.0.1',int(sys.argv[2])),Handler).serve_forever()
else:
 def arg(k): return sys.argv[sys.argv.index(k)+1]
 task=arg('--tasks'); p=Path(arg('--output_path')); p.mkdir(parents=True)
 n=int(arg('--limit')) if '--limit' in sys.argv else {'ifeval':541,'gsm8k':1319}[task]
 (p/'results.json').write_text(json.dumps({'results':{task:{'accuracy':0.5}},'n-samples':{task:{'effective':n}}}))
 (p/('samples_'+task+'.jsonl')).write_text(''.join(json.dumps({'doc_id':i,'filter':f,'resps':[['answer']]})+'\\n' for f in (['strict-match','flexible-extract'] if task=='gsm8k' else ['none']) for i in range(n)))
'''


class RunnerTests(unittest.TestCase):
    def test_optional_evidence_read_failures_are_recorded(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            refs = root / 'hub/datasets--example/refs'
            refs.mkdir(parents=True)
            (refs / 'main').write_text('revision')
            resource = root / 'tokenizers/punkt_tab.zip'
            resource.parent.mkdir()
            resource.write_bytes(b'fixture')
            original = Path.read_text
            def read(path, *args, **kwargs):
                if path.name == 'main':
                    raise PermissionError('unreadable ref')
                return original(path, *args, **kwargs)
            with patch.object(Path, 'read_text', read), patch.object(r, 'digest', side_effect=PermissionError('unreadable resource')), patch.object(r.subprocess, 'check_output', return_value='gpu snapshot'):
                r.collect_evidence(root, '/missing/server', '/missing/client', {'HF_HOME': tmp, 'NLTK_DATA': tmp})
            evidence = json.loads((root / 'environment.json').read_text())
            self.assertIn('dataset_refs_error', evidence)
            self.assertIn('nltk_error', evidence)

    def test_config_flags(self):
        c = r.tomllib.loads((r.HERE / 'suite.toml').read_text())
        self.assertIn('fp16', c['engines']['warp']['command'])
        self.assertIn('half', c['engines']['vllm']['command'])
        self.assertIn('float16', c['engines']['vllm']['command'])
        self.assertNotIn('--speculative-algorithm', c['engines']['sglang']['command'])
        self.assertEqual(c['engines']['llama']['command'].count('f16'), 4)

    def test_unresolved_and_limits(self):
        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaises(ValueError):
                r.expand('${MISSING}/bin')
        self.assertEqual(r.parser().parse_args(['--output-dir','x']).limit, 10)
        self.assertTrue(r.parser().parse_args(['--output-dir','x','--full']).full)
        with self.assertRaises(ValueError):
            r.run(r.parser().parse_args(['--output-dir','x','--limit','0']))

    def test_occupied(self):
        with socket.socket() as s:
            s.bind(('127.0.0.1',0)); s.listen()
            with self.assertRaises(OSError):
                r.port_free(s.getsockname()[1])

    def test_dead_server_cleanup(self):
        p = subprocess.Popen([sys.executable,'-c','pass'], start_new_session=True)
        p.wait()
        with self.assertRaises(RuntimeError):
            r.ready(p, 1, .1)
        self.assertTrue(r.cleanup(p))
        p = subprocess.Popen([sys.executable,'-c','import time; time.sleep(60)'], start_new_session=True)
        self.assertTrue(r.cleanup(p))
        self.assertIsNotNone(p.poll())

    def make_suite(self, base, evaluator=None, engines=('fake',)):
        fixture = base / 'fixture.py'
        fixture.write_text(FIXTURE)
        client = base / 'lm-eval'
        client.write_text('#!' + sys.executable + '\n' + (evaluator or FIXTURE))
        client.chmod(0o755)
        with socket.socket() as sock:
            sock.bind(('127.0.0.1', 0))
            port = sock.getsockname()[1]
        config = base / 'suite.toml'
        config.write_text('evaluator=' + json.dumps(str(client)) +
                          '\nreadiness_timeout=5\nclient_timeout=0.3\n' +
                          ''.join('[engines.' + name + ']\ncommand=' +
                                  json.dumps([sys.executable, str(fixture), 'server', '{port}']) +
                                  '\n' for name in engines))
        return r.parser().parse_args(['--config', str(config), '--output-dir',
                                     str(base / 'out'), '--port', str(port),
                                     '--limit', '2', '--engines', *engines])

    def test_gpu_guards(self):
        cases = [(['GPU-test', 'GPU-test, 123'], RuntimeError, 'occupied'),
                 (['invalid'], RuntimeError, 'identify'),
                 (['GPU-test', 'bad row'], RuntimeError, 'interpret'),
                 ([FileNotFoundError('nvidia-smi')], FileNotFoundError, 'nvidia-smi'),
                 (['GPU-test', subprocess.CalledProcessError(1, 'nvidia-smi')],
                  subprocess.CalledProcessError, 'non-zero'),
                 ([subprocess.TimeoutExpired('nvidia-smi', 15)],
                  subprocess.TimeoutExpired, 'timed out')]
        for outputs, error, message in cases:
            with self.subTest(outputs=outputs), patch.object(
                    r.subprocess, 'check_output', side_effect=outputs) as query:
                with self.assertRaisesRegex(error, message):
                    r.gpu_idle('2')
                self.assertEqual(query.call_args_list[0].args[0],
                                 ['nvidia-smi', '-i', '2', '--query-gpu=uuid',
                                  '--format=csv,noheader,nounits'])
                self.assertEqual(query.call_args.kwargs, {'text': True, 'timeout': 15})
        with patch.object(r.subprocess, 'check_output',
                          side_effect=['GPU-test', 'GPU-other, 123']):
            self.assertEqual(r.gpu_idle('2'), 'GPU-test')

    def test_dry_run_and_unresolved_environment(self):
        with tempfile.TemporaryDirectory() as tmp:
            args = self.make_suite(Path(tmp))
            args.dry_run = True
            with patch.object(r.subprocess, 'Popen') as popen, patch.object(
                    r.subprocess, 'check_output') as query, patch.object(r, 'port_free') as port:
                self.assertEqual(r.run(args), 0)
                popen.assert_not_called()
                query.assert_not_called()
                port.assert_not_called()
                manifest = json.loads((args.output_dir / 'manifest.json').read_text())
                self.assertEqual(manifest['limit'], 2)
                self.assertEqual(len(manifest['commands']['fake']), 2)
                self.assertFalse((args.output_dir / 'summary.json').exists())
                args.output_dir = Path(tmp) / 'unresolved'
                args.config.write_text(args.config.read_text().replace(str(Path(tmp) / 'lm-eval'),
                                                           '${QUALITY_TEST_MISSING}/lm-eval'))
                with patch.dict(os.environ, {}, clear=True):
                    with self.assertRaisesRegex(ValueError, 'Unresolved'):
                        r.run(args)
                self.assertFalse(args.output_dir.exists())
                popen.assert_not_called()
                query.assert_not_called()

    def test_readiness_alias_and_timeout(self):
        proc = Mock()
        proc.poll.return_value = None
        opener = Mock()
        opener.open.side_effect = [io.BytesIO(b'ok'),
                                  io.BytesIO(b'{"data":[{"id":"wrong"}]}')]
        with patch.object(r.urllib.request, 'build_opener', return_value=opener):
            with self.assertRaisesRegex(ValueError, 'alias mismatch'):
                r.ready(proc, 1234, 1)
        opener.open.side_effect = OSError('not listening')
        with patch.object(r.urllib.request, 'build_opener', return_value=opener), \
                patch.object(r.time, 'monotonic', side_effect=[0, 0, 2]), \
                patch.object(r.time, 'sleep'):
            with self.assertRaisesRegex(TimeoutError, 'readiness timed out'):
                r.ready(proc, 1234, 1)

    def test_client_failure_receipt_cleanup_and_next_engine(self):
        for failure, expected in [('raise SystemExit(7)', 'ifeval exited 7'),
                                  ('import time; time.sleep(60)', 'TimeoutExpired')]:
            with self.subTest(failure=failure), tempfile.TemporaryDirectory() as tmp:
                # Only the first engine fails; the second produces real native artifacts.
                client = "import sys\nif '/first/' in sys.argv[sys.argv.index('--output_path')+1]:\n " + failure + '\n' + FIXTURE
                args = self.make_suite(Path(tmp), client, ('first', 'second'))
                servers = []
                original_ready = r.ready
                def ready(proc, port, timeout):
                    if servers:
                        self.assertIsNotNone(servers[-1].poll())
                        self.assertFalse(r.group_running(servers[-1].pid))
                    servers.append(proc)
                    return original_ready(proc, port, timeout)
                try:
                    with patch.object(r, 'gpu_idle', return_value='GPU-test'), \
                            patch.object(r, 'collect_evidence'), patch.object(r, 'ready', side_effect=ready):
                        self.assertEqual(r.run(args), 1)
                    meta = json.loads((args.output_dir / 'first/metadata.json').read_text())
                    self.assertEqual(meta['status'], 'failed')
                    self.assertIn(expected, meta['error'])
                    self.assertTrue(meta['cleanup_established'])
                    self.assertIsNotNone(meta['server_exit'])
                    self.assertIn('ifeval.log', meta['sha256'])
                    summary = json.loads((args.output_dir / 'summary.json').read_text())
                    self.assertEqual(summary['first']['error'], meta['error'])
                    self.assertEqual(summary['second']['status'], 'complete')
                    self.assertEqual(len(servers), 2)
                    for proc in servers:
                        self.assertFalse(r.group_running(proc.pid))
                finally:
                    for proc in servers:
                        r.cleanup(proc)

    @unittest.skipUnless(sys.platform == 'linux', 'requires Linux /proc and process groups')
    def test_cleanup_descendant_after_leader_exit(self):
        with tempfile.TemporaryDirectory() as tmp:
            pidfile = Path(tmp) / 'pid'
            code = ('import subprocess,sys; from pathlib import Path; '
                    'p=subprocess.Popen([sys.executable,"-c","import time; time.sleep(60)"]); '
                    'Path(sys.argv[1]).write_text(str(p.pid))')
            proc = subprocess.Popen([sys.executable, '-c', code, str(pidfile)],
                                    start_new_session=True)
            try:
                proc.wait(timeout=5)
                self.assertTrue(pidfile.exists())
                self.assertTrue(r.group_running(proc.pid))
                self.assertTrue(r.cleanup(proc))
                self.assertFalse(r.group_running(proc.pid))
                stat = Path('/proc') / pidfile.read_text() / 'stat'
                if stat.exists():
                    self.assertIn(stat.read_text().rsplit(')', 1)[1].split()[0], ('Z', 'X'))
            finally:
                r.cleanup(proc)

    def test_sigterm_preserves_receipt_and_stops_suite(self):
        with tempfile.TemporaryDirectory() as tmp:
            args = self.make_suite(Path(tmp), engines=('first', 'second'))
            servers = []
            original_ready = r.ready
            def ready(proc, port, timeout):
                servers.append(proc)
                return original_ready(proc, port, timeout)
            def terminate(*unused):
                # Deliver only after the real CPU HTTP server is ready. main installs
                # the production handler; no timing-dependent external signal race.
                os.kill(os.getpid(), signal.SIGTERM)
            old_handler = signal.getsignal(signal.SIGTERM)
            try:
                with patch.object(r, 'gpu_idle', return_value='GPU-test'), \
                        patch.object(r, 'ready', side_effect=ready), \
                        patch.object(r, 'collect_evidence', side_effect=terminate):
                    self.assertEqual(r.main(['--config', str(args.config), '--output-dir',
                                             str(args.output_dir), '--port', str(args.port),
                                             '--engines', 'first', 'second']), 1)
                self.assertEqual(signal.getsignal(signal.SIGTERM), old_handler)
                meta = json.loads((args.output_dir / 'first/metadata.json').read_text())
                self.assertEqual(meta['status'], 'failed')
                self.assertIn('KeyboardInterrupt: Signal', meta['error'])
                self.assertTrue(meta['cleanup_established'])
                summary = json.loads((args.output_dir / 'summary.json').read_text())
                self.assertEqual(summary['second']['status'], 'not_run')
                self.assertFalse((args.output_dir / 'second').exists())
                self.assertEqual(len(servers), 1)
                self.assertFalse(r.group_running(servers[0].pid))
            finally:
                for proc in servers:
                    r.cleanup(proc)

    def test_native_filter_document_counts(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            result = {'results': {'gsm8k': {'accuracy': .5}},
                      'n-samples': {'gsm8k': {'effective': 2}}}
            rows = [{'doc_id': i, 'filter': f, 'resps': [['answer']]}
                    for f in ('strict-match', 'flexible-extract') for i in range(2)]
            results = directory / 'results.json'
            samples = directory / 'samples_gsm8k.jsonl'
            results.write_text(json.dumps(result))
            def write_rows(value):
                samples.write_text(''.join(json.dumps(row) + '\n' for row in value))
            write_rows(rows)
            self.assertEqual(r.validate_results(directory, 'gsm8k', 2), {'accuracy': .5})
            cases = [(rows[:2], 'Missing native task filters'),
                     (rows + [rows[0]], 'Wrong native sample count'),
                     ([rows[0], rows[0], *rows[2:]], 'Wrong native sample count'),
                     (rows[:-1], 'Wrong native sample count'),
                     ([*rows[:3], dict(rows[3], doc_id=9)], 'document sets differ')]
            for invalid, error in cases:
                with self.subTest(error=error, rows=invalid):
                    write_rows(invalid)
                    with self.assertRaisesRegex(ValueError, error):
                        r.validate_results(directory, 'gsm8k', 2)
            write_rows(rows)
            result['n-samples']['gsm8k']['effective'] = 1
            results.write_text(json.dumps(result))
            with self.assertRaisesRegex(ValueError, 'effective sample count'):
                r.validate_results(directory, 'gsm8k', 2)

    def test_integration(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            fixture = base/'fixture.py'; fixture.write_text(FIXTURE)
            evaluator = base/'lm-eval'
            evaluator.write_text('#!'+sys.executable+'\n'+FIXTURE)
            evaluator.chmod(0o755)
            with socket.socket() as s:
                s.bind(('127.0.0.1',0)); port=s.getsockname()[1]
            config=base/'suite.toml'
            config.write_text('evaluator = '+json.dumps(str(evaluator))+'\nreadiness_timeout=5\nclient_timeout=5\n[engines.fake]\ncommand='+json.dumps([sys.executable,str(fixture),'server','{port}']))
            for full in (False, True):
                out=base/str(full)
                args=r.parser().parse_args(['--config',str(config),'--output-dir',str(out),'--engines','fake','--port',str(port)]+(['--full'] if full else ['--limit','2']))
                with patch.object(r,'gpu_idle',return_value='GPU-test'):
                    self.assertEqual(r.run(args),0)
                meta=json.loads((out/'fake/metadata.json').read_text())
                self.assertEqual(meta['status'],'complete')
                self.assertTrue(meta['cleanup_established'])
                sample=next((out/'fake/ifeval').glob('samples*'))
                sample.write_text('{"doc_id":0,"resps":[[""]]}\n')
                with self.assertRaises(ValueError):
                    r.validate_results(out/'fake/ifeval','ifeval',1)
                sample.unlink()
                with self.assertRaises(ValueError):
                    r.validate_results(out/'fake/ifeval','ifeval',1)
                with self.assertRaises(FileExistsError):
                    r.run(args)


if __name__ == '__main__':
    unittest.main()
