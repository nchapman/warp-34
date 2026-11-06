#!/usr/bin/env python3
"""Pinned HGPTQ donor, not a native parity assertion.

CUDA_VISIBLE_DEVICES=1 PATH=/opt/cuda/bin:$PATH python tools/oracle/hgptq_oracle.py --capture-layers
Outputs require a new directory; no chat template, sampling, or arithmetic override.
"""
import argparse
import hashlib
import inspect
import json
import os
from pathlib import Path
import subprocess
import sys
import textwrap
import traceback
import time
import statistics

REVISION = "0fbb9a38f631ea63678f4b51a491e7ecb26c681c"
HF_HOME = Path(os.environ.get('HF_HOME', Path(os.environ.get('XDG_CACHE_HOME', Path.home() / '.cache')) / 'huggingface'))
MODEL = Path(os.environ.get('HF_HUB_CACHE', HF_HOME / 'hub')) / 'models--nchapman--Qwen3.5-2B-W4A16-HGPTQ' / 'snapshots' / REVISION
PROMPTS = ["The capital of France is", "Calculate 7 * 8. The answer is", "Write a Python function that adds two numbers.\n", "The quick brown fox"]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def patch_transforms():
    import vllm
    if vllm.__version__ != '0.31.0':
        raise RuntimeError(f'Requires vLLM 0.31.0, got {vllm.__version__}')
    # Initialize the public quantization module first (linear imports it back).
    from vllm.model_executor.layers.quantization.compressed_tensors import compressed_tensors  # noqa: F401
    from vllm.model_executor.layers.quantization.compressed_tensors.transform import linear, module as tfm_module
    from vllm.model_executor.kernels.linear.mixed_precision import marlin as mp_marlin
    cls = linear.CompressedTensorsLinearTransformMethod
    records = []
    for name, old, new in [
        ('apply', 'assert bias is None', 'assert bias is None or self.output_transform is None'),
        ('_validate_tfm_schemes', 'for part_index in range(num_partitions):', 'for part_index in self.input_tfms:'),
    ]:
        source = textwrap.dedent(inspect.getsource(getattr(cls, name)))
        if source.count(old) != 1 or new in source:
            raise RuntimeError(f'Unexpected donor source for {name}; refusing patch')
        replacement = source.replace(old, new)
        namespace = {}
        exec(compile(replacement, f'<hgptq-oracle:{name}>', 'exec'), vars(linear), namespace)
        setattr(cls, name, namespace[name])
        records.append(dict(method=name, old=old, new=new, original_sha256=hashlib.sha256(source.encode()).hexdigest(), patched_sha256=hashlib.sha256(replacement.encode()).hexdigest()))
    # Patch HadamardTransform.forward so non-multiple-of-32 block counts (e.g.
    # 1200 * (4304 // 16) = 322800 in 9B/27B vision fc2) do not hit hadacore's
    # tail-tile corruption on the final 16 rows.
    orig_had_fwd = tfm_module.HadamardTransform.forward
    orig_had_src = textwrap.dedent(inspect.getsource(orig_had_fwd))
    def patched_had_fwd(self, value, part_id=0):
        if part_id in self.weight.partitions and self.transforms[part_id].scheme.type == 'hadamard':
            ws = self.transforms[part_id].scheme.head_dim or value.shape[-1]
            orig_shape = value.shape
            flat = value.reshape(-1, ws)
            rem = flat.shape[0] % 32
            if rem != 0:
                flat = tfm_module.torch.nn.functional.pad(flat, (0, 0, 0, 32 - rem))
            out = tfm_module.ops.hadacore_transform(flat, inplace=False)
            if rem != 0:
                out = out[:flat.shape[0] - (32 - rem)]
            return out.reshape(orig_shape)
        return orig_had_fwd(self, value, part_id)
    tfm_module.HadamardTransform.forward = patched_had_fwd
    records.append(dict(
        method='HadamardTransform.forward',
        module=tfm_module.__file__,
        module_sha256=sha(tfm_module.__file__),
        original_sha256=hashlib.sha256(orig_had_src.encode()).hexdigest(),
        patched_sha256=hashlib.sha256(textwrap.dedent(inspect.getsource(patched_had_fwd)).encode()).hexdigest(),
    ))
    # Patch MarlinLinearKernel.apply_weights for group_size == -1 (channelwise)
    # with an exact 2^-3 / 2^+3 exponent shift so vLLM's unscaled FP16 cast
    # inside marlin_template.h does not overflow FP16 before multiplying w_s.
    orig_marlin_apply = mp_marlin.MarlinLinearKernel.apply_weights
    orig_marlin_src = textwrap.dedent(inspect.getsource(orig_marlin_apply))
    def patched_marlin_apply(self, layer, x, bias=None):
        if self.config.group_size == -1:
            c = self.config
            w_q, w_s, w_zp = self._get_weight_params(layer)
            return mp_marlin.apply_gptq_marlin_linear(
                input=x * 0.125,
                weight=w_q,
                weight_scale=w_s * 8.0,
                weight_zp=w_zp,
                workspace=None,
                wtype=c.weight_type,
                input_size_per_partition=c.partition_weight_shape[0],
                output_size_per_partition=c.partition_weight_shape[1],
                input_global_scale=getattr(layer, 'input_global_scale', None),
                bias=bias,
                input_dtype=c.act_type,
            )
        return orig_marlin_apply(self, layer, x, bias=bias)
    mp_marlin.MarlinLinearKernel.apply_weights = patched_marlin_apply
    records.append(dict(
        method='MarlinLinearKernel.apply_weights',
        module=mp_marlin.__file__,
        module_sha256=sha(mp_marlin.__file__),
        original_sha256=hashlib.sha256(orig_marlin_src.encode()).hexdigest(),
        patched_sha256=hashlib.sha256(textwrap.dedent(inspect.getsource(patched_marlin_apply)).encode()).hexdigest(),
    ))
    return dict(path=linear.__file__, file_sha256=sha(linear.__file__), patches=records)


def install_text_hooks(worker, out_dir):
    """Capture first real prefill, all rows, raw hidden and residual separately."""
    import torch
    from safetensors.torch import save_file
    model = worker.model_runner.model.language_model.model
    state = {'done': False, 'active': False, 'tensors': {}}
    def hook(index):
        def capture(module, args, output):
            if state['active']:
                hidden, residual = output
                summed = hidden if residual is None else hidden + residual
                state['tensors'][f'layer_{index}_hidden_after'] = summed.detach().half().cpu().contiguous().clone()
                state['tensors'][f'layer_{index}.hidden'] = hidden.detach().cpu().contiguous().clone()
                if residual is not None:
                    state['tensors'][f'layer_{index}.residual'] = residual.detach().cpu().contiguous().clone()
        return capture
    for i, layer in enumerate(model.layers):
        layer.register_forward_hook(hook(i))
    original = model.forward
    def forward(*args, **kwargs):
        if state['done']:
            return original(*args, **kwargs)
        state['active'] = True
        for key in ('input_ids', 'positions', 'inputs_embeds'):
            value = kwargs.get(key)
            if isinstance(value, torch.Tensor):
                state['tensors'][key] = value.detach().cpu().contiguous().clone()
        result = original(*args, **kwargs)
        state['tensors']['final_norm'] = result.detach().cpu().contiguous().clone()
        state['active'] = False
        state['done'] = True
        if not any(k.startswith('layer_') for k in state['tensors']):
            raise RuntimeError('No decoder hooks fired')
        save_file(state['tensors'], str(Path(out_dir) / 'text_prefill.safetensors'))
        save_file({k: v for k, v in state['tensors'].items() if k.endswith('_hidden_after') or k == 'final_norm'}, str(Path(out_dir) / 'golden.safetensors'))
        return result
    model.forward = forward
    return {'layers': len(model.layers), 'scope': 'first request prefill; raw hidden and residual separately'}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--out-dir', default='outputs/hgptq-oracle/baseline')
    ap.add_argument('--tokens', type=int, default=32)
    ap.add_argument('--capture-layers', action='store_true')
    ap.add_argument('--model', type=Path, default=MODEL)
    ap.add_argument('--image', help='PNG/JPEG; use audited vllm_vl_dump preprocessing and hooks')
    ap.add_argument('--prompt', default='<|vision_start|><|image_pad|><|vision_end|>Describe this image.')
    ap.add_argument('--capture-vision-layers', action='store_true')
    ap.add_argument('--bench-prompt-ids', type=Path)
    ap.add_argument('--graph-mode', action='store_true', help='Enable vLLM graphs; benchmark only, never capture hooks')
    ap.add_argument('--warm-runs', type=int, default=2)
    ap.add_argument('--bench-runs', type=int, default=5)
    ap.add_argument('--max-len', type=int, default=4096)
    ap.add_argument('--gpu-memory-utilization', type=float, default=0.88)
    args = ap.parse_args()
    if args.bench_prompt_ids and (args.image or args.capture_layers or args.capture_vision_layers):
        ap.error('benchmark must run separately without capture hooks')
    if args.graph_mode and not args.bench_prompt_ids:
        ap.error('--graph-mode requires --bench-prompt-ids')
    if args.capture_vision_layers and not args.image:
        ap.error('--capture-vision-layers requires --image')
    if args.warm_runs < 1 or args.bench_runs < 1:
        ap.error('warm-runs and bench-runs must be positive')
    model = args.model.expanduser().resolve()

    if not os.environ.get('CUDA_VISIBLE_DEVICES') or ',' in os.environ['CUDA_VISIBLE_DEVICES']:
        ap.error('select one GPU with CUDA_VISIBLE_DEVICES')
    if args.tokens < 1 or args.tokens > 128:
        ap.error('--tokens must be 1..128')
    os.environ['PATH'] = '/opt/cuda/bin:' + os.environ['PATH']
    os.environ['VLLM_ENABLE_V1_MULTIPROCESSING'] = '0'
    os.environ['VLLM_ALLOW_INSECURE_SERIALIZATION'] = '1'
    os.environ['HF_HUB_OFFLINE'] = '1'
    out = Path(args.out_dir).resolve()
    if not out.is_relative_to(Path('outputs/hgptq-oracle').resolve()):
        ap.error('output must be under outputs/hgptq-oracle')
    out.mkdir(parents=True, exist_ok=True)
    if any(out.iterdir()):
        ap.error('output directory must be empty')
    manifest = dict(status='starting', argv=sys.argv, model=str(model), revision=REVISION,
                    options={k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}, prompts=PROMPTS,
                    arithmetic='vLLM Hadacore (32-row tail-padded for non-32 block counts) + exact 2^-3/2^+3 channelwise Marlin exponent shift',
                    env={k: os.environ.get(k) for k in ('CUDA_VISIBLE_DEVICES', 'PATH', 'VLLM_ENABLE_V1_MULTIPROCESSING', 'HF_HUB_OFFLINE')},
                    python=sys.version, executable=sys.executable,
                    script_sha256=sha(__file__))
    def record():
        (out / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    record()
    try:
        import importlib.metadata
        import torch
        from vllm import LLM, SamplingParams
        manifest['packages'] = {k: importlib.metadata.version(k) for k in ('vllm', 'torch', 'transformers', 'compressed-tensors', 'safetensors')}
        manifest['cuda'] = torch.version.cuda
        manifest['gpu'] = subprocess.check_output(['nvidia-smi', '-i', os.environ['CUDA_VISIBLE_DEVICES'], '--query-gpu=name,uuid,driver_version,memory.used', '--format=csv,noheader'], text=True).strip()
        manifest['checkpoint_files'] = {p.name: sha(p) for p in model.iterdir() if p.suffix == '.json'}
        manifest['transform_patch'] = patch_transforms()
        if args.image:
            # Reuse the audited driver verbatim, including processor validation,
            # actual transformer boundary hooks, trace names and gate metadata.
            import vllm_vl_dump as vl
            capture_dir = out / 'vision'
            argv = ['vllm_vl_dump', '--model', str(model), '--image', args.image,
                    '--prompt', args.prompt, '--out-dir', str(capture_dir),
                    '--tokens', str(args.tokens), '--max-len', str(args.max_len),
                    '--gpu-memory-utilization', str(args.gpu_memory_utilization)]
            if args.capture_layers:
                argv.append('--capture-layers')
            if args.capture_vision_layers:
                argv.append('--capture-vision-layers')
            saved_argv = sys.argv
            try:
                sys.argv = argv
                vl.main()
            finally:
                sys.argv = saved_argv
            from safetensors.torch import load_file, save_file
            tensors = load_file(str(capture_dir / 'rank_0.safetensors'))
            save_file({k: v for k, v in tensors.items() if k == 'visual_embeddings' or k == 'final_norm' or k.endswith('_hidden_after') or k.startswith('vision_trace.')}, str(out / 'golden.safetensors'))
            request = json.loads((capture_dir / 'request.json').read_text())
            generation = json.loads((capture_dir / 'generation.json').read_text())
            (out / 'greedy.json').write_text(json.dumps({'prompts': [dict(prompt_token_ids=request['prompt_token_ids'], generated_token_ids=generation['token_ids'])]}, indent=2) + '\n')
            manifest.update(status='complete', vision_request=request)
            return
        options = dict(model=str(model), tokenizer=str(model), dtype='float16', tensor_parallel_size=1,
                       distributed_executor_backend='uni', enforce_eager=not args.graph_mode, seed=0,
                       max_model_len=args.max_len, max_num_seqs=1, gpu_memory_utilization=args.gpu_memory_utilization,
                       enable_prefix_caching=False, enable_chunked_prefill=True)
        manifest['llm_options'] = options
        record()
        llm = LLM(**options)
        if args.bench_prompt_ids:
            payload = json.loads(args.bench_prompt_ids.read_text())
            entry = payload['prompts'][0] if 'prompts' in payload else payload
            ids = entry['prompt_token_ids']
            if not ids or any(type(i) is not int or i < 0 for i in ids):
                raise ValueError('expected nonempty nonnegative integer prompt_token_ids')
            settings = dict(temperature=0, max_tokens=32, ignore_eos=True, seed=0)
            sampling = SamplingParams(**settings)
            runs = []
            for i in range(args.warm_runs + args.bench_runs):
                torch.cuda.synchronize()
                start = time.perf_counter()
                result = llm.generate([{'prompt_token_ids': ids}], sampling, use_tqdm=False)[0]
                torch.cuda.synchronize()
                elapsed = time.perf_counter() - start
                tokens = list(result.outputs[0].token_ids)
                if len(tokens) != 32:
                    raise RuntimeError('benchmark did not produce 32 tokens')
                runs.append(dict(warmup=i < args.warm_runs, seconds=elapsed, generated_token_ids=tokens))
            expected = entry.get('generated_token_ids')
            report = dict(prompt_token_ids=ids, sampling=settings, runs=runs,
                          input_sha256=sha(args.bench_prompt_ids),
                          all_runs_match=all(r['generated_token_ids'] == runs[0]['generated_token_ids'] for r in runs),
                          matches_reference=None if expected is None else all(r['generated_token_ids'] == expected for r in runs),
                          median_seconds=statistics.median(r['seconds'] for r in runs if not r['warmup']),
                          timing_scope='synchronized end-to-end generate; no hooks; prefix caching disabled')
            (out / 'benchmark.json').write_text(json.dumps(report, indent=2) + '\n')
            (out / 'greedy.json').write_text(json.dumps({'prompts': [dict(prompt_token_ids=ids, generated_token_ids=runs[-1]['generated_token_ids'])]}, indent=2) + '\n')
            manifest.update(status='complete', benchmark=report)
            return
        if args.capture_layers:
            manifest['capture'] = llm.collective_rpc(install_text_hooks, args=(str(out),))
        sampling = SamplingParams(temperature=0, max_tokens=args.tokens, ignore_eos=True, logprobs=5, seed=0)
        results = []
        for prompt in PROMPTS:
            result = llm.generate([prompt], sampling, use_tqdm=False)[0]
            output = result.outputs[0]
            results.append(dict(prompt=prompt, prompt_token_ids=result.prompt_token_ids,
                                text=output.text, token_ids=list(output.token_ids), finish_reason=output.finish_reason,
                                logprobs=[{str(k): dict(logprob=v.logprob, rank=v.rank) for k, v in step.items()} for step in output.logprobs]))
            (out / 'generations.json').write_text(json.dumps(results, indent=2) + '\n')
        (out / 'greedy.json').write_text(json.dumps({'prompts': [dict(prompt_token_ids=r['prompt_token_ids'], generated_token_ids=r['token_ids']) for r in results]}, indent=2) + '\n')
        manifest.update(status='complete', sampling=dict(temperature=0, max_tokens=args.tokens, ignore_eos=True, logprobs=5, seed=0), batch_shape='one prompt per generate, sequential', chat_template=False)
        if args.capture_layers and not (out / 'text_prefill.safetensors').exists():
            raise RuntimeError('Missing requested prefill capture')
    except Exception:
        manifest.update(status='error', error=traceback.format_exc())
        raise
    finally:
        record()


if __name__ == '__main__':
    main()
