#!/usr/bin/env python3
"""Actual pinned SGLang serving integration oracle, not a component adapter.
Host invocation runs four sequential, offline, GPU0-only disposable containers.
No model conversion, Hub resolution, or fallback dtype is permitted.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import traceback

IMAGE = 'lmsysorg/sglang:v0.5.19'
SNAP = 'models--Qwen--Qwen3.5-4B/snapshots/851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a'
PREFIX = 'sglang_mtp_serving_4b'
PROMPT = 'The capital of France is'


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def save(p, data):
    p.write_text(json.dumps(data, indent=2, default=str) + '\n')


def worker(steps, out):
    import sglang
    root = Path(sglang.__file__).parent
    model = Path('/hf/hub') / SNAP
    assert model.is_dir(), model
    files = ['srt/server_args.py', 'srt/entrypoints/engine.py',
             'srt/models/qwen3_5_mtp.py', 'srt/models/qwen3_5.py',
             'srt/models/qwen3_5_text.py', 'srt/configs/model_config.py',
             'srt/arg_groups/speculative_hook.py', 'srt/speculative/spec_info.py',
             'srt/speculative/eagle_worker_v2.py', 'srt/managers/tokenizer_manager.py']
    args = dict(model_path=str(model), tp_size=1, dtype='float16',
                mem_fraction_static=0.6, context_length=512,
                disable_cuda_graph=True, random_seed=0)
    if steps:
        args.update(speculative_algorithm='NEXTN', speculative_num_steps=steps,
                    speculative_eagle_topk=1, speculative_num_draft_tokens=steps + 1,
                    speculative_draft_model_path=str(model))
    data = dict(oracle_kind='actual_sglang_engine_serving_not_component_adapter',
                sglang_version=sglang.__version__, steps=steps, requested_engine_args=args,
                prompt=PROMPT, sampling_params=dict(temperature=0, max_new_tokens=12),
                source_sha256={f: sha(root / f) for f in files},
                script_sha256=sha(Path(__file__)),
                checkpoint_config=json.loads((model / 'config.json').read_text()),
                checkpoint_metadata_sha256={p.name: sha(p) for p in model.iterdir()
                    if p.suffix == '.json'},
                weight_blob_inventory={p.name: dict(resolved_path=str(p.resolve()),
                    bytes=p.stat().st_size) for p in model.glob('*.safetensors')},
                source_contract={f: [{'line': i+1, 'text': line}
                    for i, line in enumerate((root / f).read_text().splitlines())
                    if any(k in line for k in ['NEXTN', 'EAGLE', 'set_embed_and_head',
                                              'set_lm_head_from_target', 'float16', 'bfloat16'])]
                    for f in files}, status='starting')
    save(out, data)
    engine = None
    try:
        engine = sglang.Engine(**args)
        data['resolved_server_args'] = vars(engine.server_args)
        save(out, data)
        response = engine.generate(PROMPT, sampling_params=data['sampling_params'],
                                   return_logprob=True, logprob_start_len=0)
        data['response'] = response
        meta = response.get('meta_info', {})
        ids = response.get('output_ids')
        if ids is None:
            ids = [entry[1] for entry in meta['output_token_logprobs']]
            data['output_ids_source'] = 'meta_info.output_token_logprobs[*][1]'
        else:
            data['output_ids_source'] = 'output_ids'
        data['output_ids'] = ids
        data['accept_stats'] = {k: v for k, v in meta.items() if 'spec' in k}
        assert len(ids) == meta['completion_tokens'], 'Incomplete token ID capture'
        data['status'] = 'success'
    except BaseException:
        data['status'] = 'error'
        data['traceback'] = traceback.format_exc()
        traceback.print_exc()
    finally:
        save(out, data)
        if engine is not None:
            engine.shutdown()


def host():
    repo = Path(__file__).resolve().parents[2]
    cache = Path.home() / '.cache/huggingface'
    assert (cache / 'hub' / SNAP).is_dir(), 'Pinned local snapshot missing'
    outdir = repo / 'tools/oracle/goldens'
    outdir.mkdir(exist_ok=True)
    image_id = subprocess.check_output(['docker', 'image', 'inspect', IMAGE,
                                       '--format', '{{.Id}}'], text=True).strip()
    runs = []
    for steps in range(4):
        label = 'baseline' if steps == 0 else f'nextn{steps}'
        result = outdir / f'{PREFIX}_{label}.json'
        log = outdir / f'{PREFIX}_{label}.log'
        name = f'w34-sglang-mtp-oracle-{os.getpid()}-{steps}'
        cmd = ['docker', 'run', '--rm', '--name', name, '--network', 'none',
               '--gpus', 'device=0', '--shm-size', '8g',
               '-e', 'CUDA_VISIBLE_DEVICES=0', '-e', 'HF_HUB_OFFLINE=1',
               '-e', 'TRANSFORMERS_OFFLINE=1', '-e', 'HF_HOME=/hf',
               '-v', f'{cache}:/hf:ro', '-v', f'{repo}:/repo:rw',
               '--entrypoint', 'python', image_id,
               '/repo/tools/oracle/sglang_mtp_serving_dump.py', '--worker',
               '--steps', str(steps), '--output', f'/repo/tools/oracle/goldens/{result.name}']
        if result.exists():
            result.unlink()
        try:
            with log.open('w') as stream:
                rc = subprocess.run(cmd, stdout=stream, stderr=subprocess.STDOUT,
                                    timeout=900).returncode
        finally:
            # Also remove a container whose Engine child crashed or host timed out.
            subprocess.run(['docker', 'rm', '-f', name], capture_output=True)
        data = json.loads(result.read_text()) if result.exists() else {'status': 'no_result'}
        runs.append(dict(label=label, command=cmd, returncode=rc,
                         log=log.name, result=result.name, status=data['status'],
                         output_ids=data.get('output_ids')))
        print(label, rc, data['status'], flush=True)
    baseline = runs[0]['output_ids']
    comparisons = {r['label']: (r['output_ids'] == baseline
                   if baseline is not None and r['output_ids'] is not None else None)
                   for r in runs[1:]}
    save(outdir / f'{PREFIX}_summary.json', dict(image=IMAGE, image_id=image_id,
         checkpoint=SNAP, runs=runs, exact_output_id_equality=comparisons,
         scope='Same pinned checkpoint, FP16, TP1, prompt and greedy sampling; only speculative settings differ. Equality is a scoped within-SGLang check, not full losslessness proof. Null means unavailable. vLLM component oracle retained unchanged.'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--worker', action='store_true')
    parser.add_argument('--steps', type=int, choices=range(4), default=0)
    parser.add_argument('--output', type=Path)
    opts = parser.parse_args()
    if opts.worker:
        worker(opts.steps, opts.output)
    else:
        host()
