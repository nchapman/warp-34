#!/usr/bin/env python3
"""Sequential process supervision only; all evaluation is upstream lm-eval."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import signal
import socket
import subprocess
import time
import tomllib
import urllib.request

HERE = Path(__file__).resolve().parent
SIZES = {"ifeval": 541, "gsm8k": 1319}
WHITELIST = ("CUDA_VISIBLE_DEVICES", "HF_HOME", "NLTK_DATA", "SGLANG_MAMBA_CONV_DTYPE")


def expand(value):
    value = os.path.expandvars(value)
    if "$" in value:
        raise ValueError(f"Unresolved environment variable: {value}")
    return value


def port_free(port):
    with socket.socket() as sock:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.bind(("127.0.0.1", port))


def gpu_idle(gpu):
    def query(args):
        return subprocess.check_output(["nvidia-smi", *args, "--format=csv,noheader,nounits"], text=True, timeout=15).strip()
    uuid = query(["-i", gpu, "--query-gpu=uuid"])
    if not re.fullmatch(r"GPU-[\w-]+", uuid):
        raise RuntimeError("Cannot identify selected GPU")
    processes = query(["--query-compute-apps=gpu_uuid,pid"])
    for row in processes.splitlines():
        fields = row.split(",")
        if len(fields) != 2:
            raise RuntimeError("Cannot interpret GPU process query")
        if fields[0].strip() == uuid:
            raise RuntimeError(f"GPU {uuid} is occupied")
    return uuid


def group_running(group):
    """Linux containers may retain orphan zombies after their GPU resources close."""
    try:
        os.killpg(group, 0)
    except ProcessLookupError:
        return False
    for path in Path('/proc').glob('[0-9]*/stat'):
        try:
            fields = path.read_text().rsplit(')', 1)[1].split()
            if int(fields[2]) == group and fields[0] not in ('Z', 'X'):
                return True
        except FileNotFoundError:
            continue  # Process exited during inspection.
        except (PermissionError, ValueError, IndexError):
            return True  # Cannot establish cleanup safely.
    return not Path('/proc/self/stat').exists()


def cleanup(proc):
    """Signal the group even when its leader has already exited."""
    if proc is None:
        return True
    for sig, seconds in ((signal.SIGTERM, 15), (signal.SIGKILL, 5)):
        try:
            os.killpg(proc.pid, sig)
        except ProcessLookupError:
            proc.wait()
            return True
        deadline = time.monotonic() + seconds
        while time.monotonic() < deadline:
            proc.poll()
            if not group_running(proc.pid):
                proc.wait(timeout=1)
                return True
            time.sleep(.05)
    return False  # Never start another server with uncertain cleanup.


def ready(proc, port, timeout):
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            raise RuntimeError("Server exited before readiness")
        try:
            with opener.open(f"http://127.0.0.1:{port}/health", timeout=2) as response:
                response.read()
            with opener.open(f"http://127.0.0.1:{port}/v1/models", timeout=2) as response:
                models = json.load(response)
            if "warp34-quality" not in [m["id"] for m in models["data"]]:
                raise ValueError("Server alias mismatch")
            return models
        except (OSError, urllib.error.URLError):
            time.sleep(.2)
    raise TimeoutError("Server readiness timed out")


def validate_results(directory, task, limit):
    results = list(directory.rglob("results*.json"))
    samples = list(directory.rglob(f"samples_{task}*.jsonl"))
    if len(results) != 1 or len(samples) != 1:
        raise ValueError("Missing or ambiguous native results/sample files")
    result = json.loads(results[0].read_text())
    rows = [json.loads(line) for line in samples[0].read_text().splitlines()]
    expected = min(limit, SIZES[task]) if limit else SIZES[task]
    filters = {'strict-match', 'flexible-extract'} if task == 'gsm8k' else {'none'}
    if result['n-samples'][task]['effective'] != expected:
        raise ValueError('Wrong native effective sample count')
    if {r.get('filter') for r in rows} != filters:
        raise ValueError('Missing native task filters')
    document_sets = []
    for name in filters:
        documents = [r['doc_id'] for r in rows if r['filter'] == name]
        if len(documents) != expected or len(set(documents)) != expected:
            raise ValueError('Wrong native sample count')
        document_sets.append(set(documents))
    if any(ids != document_sets[0] for ids in document_sets):
        raise ValueError('Native filter document sets differ')
    def nonempty(value):
        return bool(value.strip()) if isinstance(value, str) else isinstance(value, list) and bool(value) and all(nonempty(v) for v in value)
    if any(not nonempty(r.get("resps")) for r in rows):
        raise ValueError("Empty native response")
    metrics = result["results"][task]
    if not metrics:
        raise ValueError("Missing native metrics")
    return metrics


def dump(path, data):
    path.write_text(json.dumps(data, indent=2) + "\n")


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def collect_evidence(directory, executable, evaluator, env):
    """Optional reads are best-effort; inability to persist evidence is fatal."""
    diagnostics = {}
    for label, program in [('server', executable), ('client', evaluator)]:
        path = Path(shutil.which(program) or program)
        python = path if path.name.startswith('python') else path.parent / 'python'
        if python.is_file():
            code = 'import importlib.metadata as m,json; print(json.dumps(sorted((d.metadata["Name"],d.version) for d in m.distributions())))'
            try:
                diagnostics[label + '_packages'] = json.loads(subprocess.check_output(
                    [str(python), '-c', code], env=env, text=True, timeout=30))
            except (OSError, subprocess.SubprocessError, ValueError) as exc:
                diagnostics[label + '_packages_error'] = str(exc)
    try:
        diagnostics['gpu_snapshot_csv'] = subprocess.check_output(
            ['nvidia-smi', '--query-gpu=index,uuid,name,memory.total,memory.used,driver_version', '--format=csv'],
            text=True, timeout=15)
    except (OSError, subprocess.SubprocessError) as exc:
        diagnostics['gpu_snapshot_error'] = str(exc)
    hf_home = Path(env.get('HF_HOME', str(Path.home() / '.cache/huggingface')))
    try:
        diagnostics['dataset_refs'] = {str(p): p.read_text().strip() for p in (hf_home / 'hub').glob('datasets--*/refs/*') if p.is_file()}
    except (OSError, ValueError) as exc:
        diagnostics['dataset_refs_error'] = str(exc)
    diagnostics['nltk_sha256'] = {}
    try:
        for folder in env.get('NLTK_DATA', '').split(os.pathsep):
            resource = Path(folder) / 'tokenizers/punkt_tab.zip'
            if folder and resource.is_file():
                diagnostics['nltk_sha256'][str(resource)] = digest(resource)
    except (OSError, ValueError) as exc:
        diagnostics['nltk_error'] = str(exc)
    dump(directory / 'environment.json', diagnostics)


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path, default=HERE / "suite.toml")
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--engines", nargs="+", default=["warp", "vllm", "llama", "sglang"])
    p.add_argument("--gpu", default="0")
    p.add_argument("--port", type=int, default=18034)
    group = p.add_mutually_exclusive_group()
    group.add_argument("--full", action="store_true")
    group.add_argument("--limit", type=int, default=10)
    p.add_argument("--dry-run", action="store_true")
    return p


def run(args):
    if args.limit <= 0 or not 0 < args.port < 65536:
        raise ValueError("Limit must be positive and port valid")
    if not re.fullmatch(r"(?:[0-9]+|GPU-[\w-]+)", args.gpu):
        raise ValueError("Specify exactly one GPU index or UUID")
    config = tomllib.loads(args.config.read_text())
    names = args.engines
    if len(set(names)) != len(names) or any(n not in config['engines'] for n in names):
        raise ValueError("Unknown or duplicate engines")
    evaluator = expand(config['evaluator'])
    engines = {n: {**config['engines'][n], 'command': [expand(x).replace('{port}', str(args.port)) for x in config['engines'][n]['command']]} for n in names}
    root = args.output_dir.resolve()
    root.mkdir(parents=True, exist_ok=False)
    env = {k: v for k, v in os.environ.items() if not k.startswith('LLAMA_ARG_')}
    env['CUDA_VISIBLE_DEVICES'] = args.gpu
    env['SGLANG_MAMBA_CONV_DTYPE'] = 'float16'
    for name in ('requirements.lock', 'requirements.in', 'lm-eval.yaml', 'run_suite.py'):
        shutil.copyfile(HERE / name, root / name)
    native = root / 'lm-eval.yaml'
    native.write_text(native.read_text().replace(':18034/', f':{args.port}/'))
    shutil.copyfile(args.config, root / 'suite.toml')
    limit = None if args.full else args.limit
    manifest = {'engines': engines, 'environment': {k: env[k] for k in WHITELIST if k in env}, 'limit': limit, 'commands': {}, 'sha256': {p.name: digest(p) for p in root.iterdir() if p.is_file()}}
    for n in names:
        manifest['commands'][n] = [[evaluator, 'run', '--config', str(native), '--tasks', task, '--output_path', str(root / n / task)] + ([] if limit is None else ['--limit', str(limit)]) for task in SIZES]
    dump(root / 'manifest.json', manifest)
    if args.dry_run:
        return 0
    failed = False
    summary = {name: {'status': 'not_run'} for name in names}
    dump(root / 'summary.json', summary)
    for name, engine in engines.items():
        print(f'START {name}: {engine.get("mode", "configured")} / FP16 KV', flush=True)
        directory = root / name
        directory.mkdir()
        meta = {'status': 'running', 'mode': engine.get('mode'), 'metrics': {}, 'artifact_sha256': {}}
        dump(directory / 'metadata.json', meta)
        server = client = None
        clean = True
        interrupted = False
        try:
            port_free(args.port)
            meta['gpu_uuid'] = gpu_idle(args.gpu)
            # Hash executables and small model configuration files, not multi-GB weights.
            for token in [engine['command'][0], evaluator, *engine['command'][1:]]:
                path = Path(shutil.which(token) or token)
                if path.is_dir():
                    path = path / 'config.json'
                if path.is_file() and path.stat().st_size < 256 * 1024 * 1024:
                    meta['artifact_sha256'][str(path)] = digest(path)
            with (directory / 'server.log').open('w') as log:
                server = subprocess.Popen(engine['command'], env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
                dump(directory / 'models.json', ready(server, args.port, config['readiness_timeout']))
                collect_evidence(directory, engine['command'][0], evaluator, env)
                for task, command in zip(SIZES, manifest['commands'][name]):
                    with (directory / f'{task}.log').open('w') as client_log:
                        client = subprocess.Popen(command, env=env, stdout=client_log, stderr=subprocess.STDOUT, start_new_session=True)
                        code = client.wait(timeout=config['client_timeout'])
                    clean = cleanup(client)
                    if not clean:
                        raise RuntimeError('Client group cleanup not established')
                    client = None
                    if code:
                        raise RuntimeError(f'{task} exited {code}')
                    if server.poll() is not None:
                        raise RuntimeError('Server died during evaluation')
                    meta['metrics'][task] = validate_results(directory / task, task, limit)
            meta['status'] = 'complete'
        except BaseException as exc:
            meta.update(status='failed', error=f'{type(exc).__name__}: {exc}')
            failed = True
            interrupted = isinstance(exc, KeyboardInterrupt)
        finally:
            # Ignore repeated termination while establishing cleanup.
            old = signal.signal(signal.SIGTERM, signal.SIG_IGN)
            old_int = signal.signal(signal.SIGINT, signal.SIG_IGN)
            try:
                clean = cleanup(client) and clean
                clean = cleanup(server) and clean
            finally:
                signal.signal(signal.SIGTERM, old)
                signal.signal(signal.SIGINT, old_int)
            meta['cleanup_established'] = clean
            meta['server_exit'] = server.returncode if server else None
            if not clean:
                meta.update(status='failed', error='Process group cleanup not established')
                failed = True
            meta['sha256'] = {str(p.relative_to(directory)): digest(p) for p in directory.rglob('*') if p.is_file() and p.name != 'metadata.json'}
            dump(directory / 'metadata.json', meta)
            summary[name] = {k: meta[k] for k in ('status', 'mode', 'metrics', 'cleanup_established')}
            if 'error' in meta:
                summary[name]['error'] = meta['error']
            dump(root / 'summary.json', summary)
        print(f'{name}: {meta["status"]}' + (f' — {meta["error"]}' if 'error' in meta else ''), flush=True)
        if not clean or interrupted:
            return 1
    return int(failed)


def main(argv=None):
    def terminate(signum, frame):
        raise KeyboardInterrupt(f'Signal {signum}')
    old = signal.signal(signal.SIGTERM, terminate)
    try:
        return run(parser().parse_args(argv))
    except (Exception, KeyboardInterrupt) as exc:
        print(f'quality suite: {exc}', file=__import__('sys').stderr)
        return 1
    finally:
        signal.signal(signal.SIGTERM, old)


if __name__ == '__main__':
    raise SystemExit(main())
