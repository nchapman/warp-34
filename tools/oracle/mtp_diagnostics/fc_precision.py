"""FC counterfactual using captured native inputs and pinned checkpoint weight."""
import argparse
import numpy as np
import torch
from safetensors.torch import load_file
p=argparse.ArgumentParser();p.add_argument('donor');p.add_argument('w34');p.add_argument('weights');a=p.parse_args()
x=load_file(a.donor); w=load_file(a.weights)['mtp.fc.weight'].cuda().half()
z=x['trace.fc.in.0'].cuda(); ref=x['trace.fc.out'][0].float()
actual=torch.from_numpy(np.fromfile(a.w34+'/row_0_fc.f16',dtype=np.float16)).float()
for reduced in (True,False):
 torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction=reduced
 for rows in (1,z.shape[0]):
  v=(z[:rows]@w.T)[0].cpu().float()
  print('reduced',reduced,'rows',rows,'vs donor', (v-ref).abs().max().item(),'vs W34',(v-actual).abs().max().item(),'unequal W34',(v!=actual).sum().item())
v=(z[:1].float()@w.float().T).half()[0].cpu().float()
print('FP32 accumulation vs donor', (v-ref).abs().max().item(),'vs W34',(v-actual).abs().max().item(),'unequal W34',(v!=actual).sum().item())
# Cancellation at residual addition amplifies the tiny FC/projection errors.
f=x['trace.fc.out'][0].float(); o=x['trace.layers.0.self_attn.o_proj.out.0'][0].float()
print('FC norm',f.norm().item(),'O norm',o.norm().item(),'sum norm',(f+o).norm().item())
