#!/usr/bin/env python3
"""CPU-only metrics for immutable real-operator captures and native BF16 traces."""
import argparse
import json
from pathlib import Path
import torch
from safetensors.torch import load_file


def metrics(x, y):
    x, y = x.double().flatten(), y.double().flatten()
    assert x.shape == y.shape
    return dict(relL2=((x-y).square().sum()/y.square().sum().clamp_min(1e-300)).sqrt().item(),
                maxabs=(x-y).abs().max().item(), unequal=(x != y).sum().item())


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--fixture', type=Path, required=True)
    p.add_argument('--trace', type=Path, required=True)
    p.add_argument('--original', type=Path, required=True)
    args = p.parse_args()
    a = load_file(str(args.fixture))
    original = load_file(str(args.original))
    result = {'capture_vs_original': {}, 'norm_counterfactual': {}, 'native_trace': {}}
    for name in original.keys() & a.keys():
        result['capture_vs_original'][name] = metrics(a[name], original[name])
    for name in sorted(a):
        if not name.startswith('norms.') or not name.endswith('.input1'):
            continue
        prefix = name[:-7]
        x, r, w = (a[prefix+s].float() for s in ('.input0', '.input1', '.weight'))
        eps = a[prefix+'.eps'].item()
        summed = x+r
        variants = {}
        for label, v in [('fp32_sum', summed), ('rounded_bf16_sum', summed.bfloat16().float())]:
            out = (v * torch.rsqrt(v.square().mean(-1, keepdim=True)+eps) * w).bfloat16()
            variants[label] = metrics(out, a[prefix+'.output0'])
        variants['max_operand_abs'] = max(x.abs().max().item(), r.abs().max().item())
        result['norm_counterfactual'][prefix] = variants
    for path in sorted(args.trace.glob('*.bf16')):
        name = path.name[:-5]
        if name not in a:
            continue
        x = torch.frombuffer(bytearray(path.read_bytes()), dtype=torch.bfloat16)
        result['native_trace'][name] = metrics(x, a[name])
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
