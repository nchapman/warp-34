#!/usr/bin/env python3
"""Pinned SGLang donor preflight and executable streaming benchmark.

Use --execute inside the pinned writable-container image, GPU1 only; mount this
repository read-only and an output directory writable. The worker helper injects
hooks into installed sources and restores their bytes on exit. Run exactly one
execution per standalone process: loaded Python hooks survive file restoration;
concurrent users of the same package tree are not supported.
--profile is an external capture responsibility: run nsys inside the container.
Use profile_diff --oracle-nvtx-correlated with runtime JSON after export.
Runtime timings include synchronizing acceptance telemetry.
"""
import argparse
import hashlib
import importlib.metadata
import importlib.util
import json
from pathlib import Path
import re
import sys

IMAGE = 'sha256:d6e7288627be8b02be88e4bba38e73f6d50e2826869f753c13a4c4385ab3eda9'
PHRASE = 'The quick brown fox jumps over the lazy dog. '
TARGET = 'Qwen/Qwen3.5-4B'
DRAFT = 'nchapman/Qwen3.5-4B-DFlash2'


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def prompt_ids(tokenizer, count):
    # Match encode's default special-token policy; never encode repeated text.
    ids = tokenizer.encode(PHRASE)
    if not ids or any(type(x) is not int or x < 0 for x in ids):
        raise ValueError('tokenizer.encode returned invalid token IDs')
    if count <= 0:
        raise ValueError('prompt length must be positive')
    return (ids * ((count + len(ids) - 1) // len(ids)))[:count]


def snapshot(path, repo):
    path = Path(path).absolute()
    if (path.parent.name != 'snapshots' or
            not re.fullmatch(r'[0-9a-f]{40}', path.name) or
            path.parent.parent.name != 'models--' + repo.replace('/', '--')):
        raise ValueError(f'{repo}: require explicit HF snapshots/<40-hex-commit> path')
    if not path.is_dir() or not (path / 'config.json').is_file():
        raise ValueError(f'missing snapshot/config: {path}')
    weights = sorted(path.glob('*.safetensors'))
    if not weights or any(not p.is_file() or p.stat().st_size == 0 for p in weights):
        raise ValueError(f'missing/empty weight blobs: {path}')
    index = path / 'model.safetensors.index.json'
    if index.exists():
        names = set(json.loads(index.read_text())['weight_map'].values())
        if not names or any(not (path / n).is_file() for n in names):
            raise ValueError(f'missing indexed weight shard: {path}')
    metadata = {p.name: sha(p) for p in sorted(path.iterdir())
                if p.is_file() and p.suffix in ('.json', '.model', '.txt', '.jinja')}
    return dict(repo=repo, path=str(path), snapshot_sha=path.name,
                metadata_sha256=metadata,
                weights=[dict(name=p.name, resolved=str(p.resolve()),
                              bytes=p.stat().st_size) for p in weights],
                identity_caveat='snapshot commit + metadata hashes; weight bytes not hashed')


def protocol(tokenizer):
    return dict(prompt_text=PHRASE, prompt_ids=prompt_ids(tokenizer, 128),
                warm_prompt_ids=prompt_ids(tokenizer, 16),
                warm_output_tokens=32, output_tokens=128,
                decode_commits=127, first_prefill_token_included=True,
                sampling=dict(temperature=0, max_new_tokens=128, ignore_eos=True),
                tp=1, M=16, target_dtype='float16', draft_dtype='bfloat16',
                prefix_cache=False, reset_after_warm=True,
                kv_policy=dict(oracle='float16', w34='tq4',
                               comparison='different KV policies, NOT identical KV'),
                timing_contract='wall TTFT to first streamed token; decode to final token',
                verify_rounds=None, output_ids=None,
                runtime_telemetry='absent: this is preflight, not execution')


SOURCES = {
    'srt/entrypoints/engine.py': ['def generate(', 'input_ids:', 'def flush_cache('],
    'srt/speculative/dflash_worker_v2.py': [
        'def forward_batch_generation(', 'batch.forward_mode.is_extend()',
        'batch.forward_mode.is_idle()', 'accept_lens=commit_lens',
        'speculative_num_draft_tokens=int(self.block_size)'],
    'srt/arg_groups/speculative_hook.py': [
        'def _handle_dflash(', 'speculative_dflash_block_size'],
}


def evidence(root, version):
    if version != '0.5.19':
        raise ValueError(f'require sglang==0.5.19, got {version}')
    result = {}
    for name, needles in SOURCES.items():
        p = root / name
        text = p.read_text()
        for needle in needles:
            if needle not in text:
                raise ValueError(f'unsupported source contract: {name}: {needle}')
        result[name] = dict(sha256=sha(p), lines=[
            dict(line=i, text=line.strip())
            for i, line in enumerate(text.splitlines(), 1)
            if any(n in line for n in needles)])
    return result


PLAN = [
    'Use offline Engine.generate(input_ids=..., stream=True, rid=...) with distinct '
    'warm/measured IDs. API verified in installed image; no text re-tokenization.',
    'Resolve BF16 drafter initialization and actual target/draft KV dtypes in worker '
    'before accepting results; target dtype alone is not evidence of draft dtype.',
    'Warm 16/32, drain request, flush cache and prove draft/target request state reset. '
    'Engine.flush_cache exists, but its complete DFlash state-reset semantics are not verified.',
    'Inject hooks in spawned GPU worker, not client. forward_batch_generation has '
    'extend/prefill and idle branches that must be excluded. Bound each real decode '
    'call with oracle.decode.round, recording PID, request ID and measured/warm phase.',
    'Capture returned accept_lens as commit_lens, not accepted-draft counts. '
    '_accept_block distinguishes accept_len from commit_lens; scheduler truncation at '
    'max_new_tokens means sum of raw worker commits need not equal delivered tokens. '
    'Record actual round events and delivered IDs; never infer rounds from M or 127.',
    'Use profile_diff.parse_oracle_nvtx_correlated for RID worker ranges; validate '
    'actual worker telemetry count, not scheduler count. Set ORACLE_PROFILE_FLUSH=1 '
    'during external nsys capture to retain traces before worker termination.',
    'Optional CUDA profiler start/stop must execute in GPU worker around measured '
    'work after warm reset. Otherwise externally wrap nsys for full process-tree capture; '
    'retain prefill in trace but exclude its correlated launches from decode analysis.',
    'Save streamed output IDs, actual generated count=128, TTFT/decode wall ms, worker '
    'round events and resolved settings. Missing IDs/telemetry/dtype evidence fails closed. '
    'No parity claim until fresh donor and w34 artifacts are compared.',
]


def execute(args, data, root):
    import os
    import copy
    import time
    from aligned_worker_hook import inject, normalize_draft
    events = args.out.with_suffix('.workers.jsonl')
    events.write_text('')
    normalized = args.out.parent / 'draft-native-config'
    data['draft_config_translation'] = normalize_draft(args.draft, normalized)
    settings = dict(model_path=str(args.target), dtype='float16',
                    quantization=None, tp_size=1,
                    speculative_algorithm='DFLASH',
                    speculative_draft_model_path=str(normalized),
                    speculative_dflash_block_size=16,
                    attention_backend='flashinfer',
                    speculative_draft_attention_backend='flashinfer',
                    context_length=1024, max_running_requests=1,
                    mem_fraction_static=0.7, disable_radix_cache=True,
                    stream_interval=1, cuda_graph_backend_prefill='disabled',
                    cuda_graph_max_bs_decode=1)
    data.update(status='runtime_starting', engine_args=settings,
                runtime_environment={'SGLANG_MAMBA_CONV_DTYPE': 'float16'},
                helper_sha256=sha(Path(__file__).with_name('aligned_worker_hook.py')),
                quantization_deviation='Pinned target declares compressed-tensors; explicit gptq_marlin rejected. Auto resolves checkpoint quantization.',
                profiling_limitations='External nsys required; ORACLE_PROFILE_FLUSH=1 flushes worker traces before shutdown. No matched w34 correctness/profile gate.',
                instrumentation='worker NVTX per decode round; synchronous commit telemetry; draft ModelConfig dtype override',
                comparison_complete=False)
    data['protocol']['runtime_telemetry'] = 'Execution starting; runtime evidence pending.'
    args.out.write_text(json.dumps(data, indent=2)+'\n')
    environment = {'ORACLE_EVENTS': str(events), 'SGLANG_MAMBA_CONV_DTYPE': 'float16'}
    previous = {key: os.environ.get(key) for key in environment}
    try:
        os.environ.update(environment)
        with inject(root):
            engine = None
            try:
                from sglang import Engine
                engine = Engine(**settings)
                def run(rid, ids, count):
                    start = time.monotonic_ns()
                    chunks = []
                    for chunk in engine.generate(input_ids=ids, sampling_params=dict(
                            temperature=0, max_new_tokens=count, ignore_eos=True),
                            stream=True, rid=rid):
                        chunks.append(dict(elapsed_ms=(time.monotonic_ns()-start)/1e6, response=copy.deepcopy(chunk)))
                    return dict(rid=rid, input_ids=ids, chunks=chunks)
                data['warm'] = run('oracle-warm', data['protocol']['warm_prompt_ids'], 32)
                flush = engine.flush_cache()
                data['flush_cache_result'] = dict(success=flush.success, message=flush.message)
                data['measured'] = run('oracle-measured', data['protocol']['prompt_ids'], 128)
                data['worker_events'] = [json.loads(l) for l in events.read_text().splitlines()]
                chunks = data['measured']['chunks']
                data['output_ids'] = chunks[-1]['response']['output_ids']
                data['output_count'] = len(data['output_ids'])
                nonempty = [c for c in chunks if c['response'].get('output_ids')]
                data['ttft_ms'] = nonempty[0]['elapsed_ms']
                data['decode_ms'] = nonempty[-1]['elapsed_ms']-nonempty[0]['elapsed_ms']
                rounds = [e for e in data['worker_events'] if e['kind']=='decode' and e['rids']==['oracle-measured']]
                data['verify_rounds'] = len(rounds)
                data['commit_lens'] = [e['commit_lens'] for e in rounds]
                data['fresh_prefill_verified'] = any(
                    e['kind']=='prefill' and e['rids']==['oracle-measured']
                    and e['prefix_lens']==[0] and e['extend_lens']==[128]
                    for e in data['worker_events'])
                data['scheduler_verify_rounds'] = chunks[-1]['response']['meta_info']['spec_verify_ct']
                data['round_count_note'] = 'Worker calls include overlap-scheduled work beyond final delivered token; scheduler count is separate.'
                data['status'] = 'runtime_completed'
                assert data['flush_cache_result']['success'] and data['fresh_prefill_verified']
                assert rounds
                data['protocol']['runtime_telemetry'] = 'See top-level worker_events and measured chunks; preflight protocol retained for identity.'
                assert data['output_count'] == 128, data['output_count']
            finally:
                try:
                    args.out.write_text(json.dumps(data, indent=2)+'\n')
                finally:
                    if engine is not None:
                        engine.shutdown()
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--target', required=True, type=Path)
    ap.add_argument('--draft', required=True, type=Path)
    ap.add_argument('--out', required=True, type=Path)
    ap.add_argument('--profile', action='store_true')
    ap.add_argument('--execute', action='store_true')
    args = ap.parse_args(argv)
    if args.profile:
        ap.error('use external nsys inside Docker with --execute and ORACLE_PROFILE_FLUSH=1; parse with --oracle-nvtx-correlated')
    try:
        version = importlib.metadata.version('sglang')
        # The pinned Docker image uses an editable install: distribution metadata
        # lives in site-packages, while the importable sources live elsewhere.
        spec = importlib.util.find_spec('sglang')
        if spec is None or not spec.submodule_search_locations:
            raise ValueError('cannot locate installed SGLang package sources')
        root = Path(next(iter(spec.submodule_search_locations)))
        src = evidence(root, version)
        target = snapshot(args.target, TARGET)
        draft = snapshot(args.draft, DRAFT)
        if 'tokenizer.json' not in target['metadata_sha256']:
            raise ValueError('target snapshot missing tokenizer.json')
        # Preflight does not import heavy serving modules/torch.
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(str(args.target),
                                                  local_files_only=True,
                                                  trust_remote_code=False)
        data = dict(status='preflight_only_capture_blocked', comparison_complete=False,
                    image_id=IMAGE, sglang_version=version, target=target, draft=draft,
                    tokenizer_hash=target['metadata_sha256']['tokenizer.json'],
                    protocol=protocol(tokenizer), source_evidence=src,
                    capture_plan=PLAN)
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(data, indent=2) + '\n')
        if args.execute:
            execute(args, data, root)
        else:
            print(f'Preflight only: {args.out}; no serving/profile capture performed')
    except (ValueError, OSError, KeyError, ImportError) as exc:
        print(f'Preflight failed: {exc}', file=sys.stderr)
        return 2
    return 0


if __name__ == '__main__':
    sys.exit(main())
