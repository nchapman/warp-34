"""Compare opt-in W34_MTP_TRACE_DIR dumps with traced actual donor calls."""
import argparse
import numpy as np
import torch
from safetensors.torch import load_file
p = argparse.ArgumentParser()
p.add_argument('donor'); p.add_argument('w34'); a = p.parse_args()
x = load_file(a.donor)
mapping = {'concat':'fc.in.0','fc':'fc.out','input_norm':'layers.0.input_layernorm.out','qkv':'layers.0.self_attn.qkv_proj.out.0','o_proj':'layers.0.self_attn.o_proj.out.0','post_norm':'layers.0.post_attention_layernorm.out.0','residual':'layers.0.post_attention_layernorm.out.1','gate_up':'layers.0.mlp.gate_up_proj.out.0','act':'layers.0.mlp.act_fn.out','down':'layers.0.mlp.down_proj.out.0','output':'norm.out.0'}
for name, key in mapping.items():
    u = torch.from_numpy(np.fromfile(f'{a.w34}/row_0_{name}.f16', dtype=np.float16)).float()
    captured = x['trace.'+key].float()
    # Legacy row-only dumps and native full-batch dumps are both supported.
    v = captured.flatten() if u.numel() == captured.numel() else captured[0].flatten()
    assert u.numel() == v.numel(), (name, u.shape, captured.shape)
    print(f'{name:12} maxabs={(u-v).abs().max():.9g} relL2={((u-v).norm()/v.norm()):.9g} refmax={v.abs().max():.9g} unequal={(u!=v).sum().item()}/{u.numel()}')
