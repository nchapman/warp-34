#!/usr/bin/env python3
"""Run inside pinned SGLang image, exposing physical GPU1 ONLY.
Example (from repository root):
 docker run --rm --gpus '"device=1"' --network=none -v "$PWD:/work:ro" \
   --entrypoint python lmsysorg/sglang:v0.5.19 /work/tools/oracle/trace_bf16_attention.py
Print actual FA2 launch names and independent torch contraction counterfactuals.
Does not rewrite fixtures. CPU double reductions distinguish rounding policies.

Measured on physical GPU1 (RTX 3090 Ti), pinned FlashInfer 0.6.18:
KernelTraits<MaskMode=0, CTA_TILE_Q=64, NUM_MMA_Q=1, NUM_MMA_KV=1,
 NUM_MMA_D_QK=16, NUM_MMA_D_VO=16, NUM_WARPS_Q=4, NUM_WARPS_KV=1,
 PosEncodingMode=0, bf16,bf16,bf16,float,int,DefaultAttention<false,...>>.
KV tile is NUM_MMA_KV * NUM_WARPS_KV * 16 = 16, not CTA_TILE_Q!
Fresh donor replay is bit-identical to all five operator fixtures. The older
model archive has byte-identical Q/K/V/attention operands to this archive.

Native relL2, layers 0..4 (unchanged bound 1e-4):
  tile64 scalar: 5.1513e-5, 8.04152e-4, 3.56591e-4, 5.91687e-4, 7.00413e-4
  tile64 QK MMA: 4.3603e-5, 8.04152e-4, 3.54565e-4, 5.91687e-4, 7.00413e-4
  + PV MMA:     4.3603e-5, 8.04149e-4, 3.54533e-4, 5.91661e-4, 7.00413e-4
Explicit rcp.approx, explicit score-scale FMA and separately rounded scale
products did not change the printed +PV MMA metrics. Those experiments were
reverted; the retained arithmetic change is ONLY tile64 -> tile16.
  tile16 scalar: 5.1513e-5, 4.242e-6, 3.8260e-5, 5.020e-6, 3.8e-8
  maxabs:        2,         1,        64,        2,        .0078125
Independent CPU double online16 contraction reproduces the improvement;
layer4 is even bit-identical. Full-tile rounded probabilities do not.
Tests: phase4_bf16_attention 2/2; phase4_draft_kernels 2/2 (includes unchanged
short/full/narrow and 2064-key fixtures). No full-model or full verify run.
Residual scalar/MMA differences and dynamic/split-KV parity remain; no claim
of bit parity, full-model strict parity, or performance improvement.
"""
import torch
import flashinfer
from safetensors.torch import load_file
from pathlib import Path
p = Path(__file__).resolve().parent / 'goldens/sglang_dflash_model_4b_operators.safetensors'
a = load_file(str(p), device='cuda')
ws = torch.empty(128*1024*1024, dtype=torch.uint8, device='cuda')
i32 = lambda x: torch.tensor(x, dtype=torch.int32, device='cuda')
w = flashinfer.BatchPrefillWithPagedKVCacheWrapper(ws, kv_layout='NHD', backend='fa2')
w.plan(i32([0,16]), i32([0,21]), i32(list(range(21))), i32([1]),16,4,256,1,causal=False,window_left=-1,q_data_type=torch.bfloat16,kv_data_type=torch.bfloat16)
def metrics(x,y):
    x,y=x.double(),y.double()
    return {'relL2': ((x-y).norm()/y.norm()).item(), 'maxabs': (x-y).abs().max().item(), 'different': (x!=y).sum().item()}
for l in range(5):
    q,k,v,y=[a[f'layers.{l}.{n}'] for n in ['q','k_all','v_all','attention']]
    kv=torch.stack((k,v),dim=1).unsqueeze(2).contiguous()
    out=w.forward(q,kv,causal=False,window_left=-1,sm_scale=1/16)
    torch.cuda.synchronize()
    print(l,'donor replay',metrics(out,y),flush=True)
    if l==0:
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA]) as prof:
            w.forward(q,kv,causal=False,window_left=-1,sm_scale=1/16)
            torch.cuda.synchronize()
        for e in prof.events():
            if e.device_type == torch.autograd.DeviceType.CUDA: print('KERNEL',e.name,flush=True)
    qd=q.cpu().double().transpose(0,1)
    kd=k.cpu().double().repeat_interleave(4,1).transpose(0,1)
    vd=v.cpu().double().repeat_interleave(4,1).transpose(0,1)
    s=qd@kd.transpose(1,2)/16
    p=(s-s.amax(-1,keepdim=True)).exp()
    for rounded in [False,True]:
        pp=p.bfloat16().double() if rounded else p
        z=((pp@vd)/pp.sum(-1,keepdim=True)).transpose(0,1).bfloat16()
        print(l,'double contraction rounded_p='+str(rounded),metrics(z,y.cpu()),flush=True)
    # Same real operands, but round probabilities at each 16-key online update.
    m=torch.full_like(s[:,:,:1],-torch.inf)
    d=torch.zeros_like(m)
    o=torch.zeros_like(qd)
    for start in range(0,21,16):
        st=s[:,:,start:start+16]
        mn=torch.maximum(m,st.amax(-1,keepdim=True))
        alpha=(m-mn).exp()
        pp=(st-mn).exp().bfloat16().double()
        o=o*alpha+pp@vd[:,start:start+16]
        d=d*alpha+pp.sum(-1,keepdim=True)
        m=mn
    print(l,'double contraction online16',metrics((o/d).transpose(0,1).bfloat16(),y.cpu()),flush=True)
