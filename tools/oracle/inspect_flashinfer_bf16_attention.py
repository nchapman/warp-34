#!/usr/bin/env python3
"""Print pinned FA2 arithmetic evidence without exposing any GPU to Docker.

Investigation results (physical GPU1, CUDA_VISIBLE_DEVICES=1 on host):
Actual model donor operands, native paged BF16 replay, relL2 / maxabs:
layer   original                 BF16 tile probability policy
0       .001536437 / 8            .000051513 / 2
1       .001457836 / 16           .000804152 / 32
2       .000855953 / 256          .000356591 / 128
3       .001197535 / 32           .000591687 / 32
4       .001523777 / 16           .000700413 / 16

First-layer regression passes its pre-fix 1e-4 relative bound. All-five-layer
regression intentionally still fails that same bound (layers 1..4). No golden
or tolerance changes. This is a partial arithmetic fix, NOT full-model parity.

Important: installed FlashInfer 0.6.18 compute_sfm_v rounds exp2 probabilities
BEFORE BOTH PV MMA and denominator rowsum; denominator accumulates in FP32
from BF16 inputs, not original FP32 exponentials. update_mdo_states rescales
previous numerator and denominator with the updated block max. Native now
implements this with fixed 64-key online tiles. Remaining differences include
scalar QK/PV reduction order vs MMA, dynamic donor tile/warp partitioning and
split-KV merging. The actual launched sm86 tile specialization was NOT traced;
21-key donor tests cannot distinguish 64/128 tiles. Long-context bit parity
is unresolved. The existing 2064-key donor fixture still passes unchanged
maxabs < .02, as do short/full/narrow variants and request-mask checks.

Validation: phase4_bf16_ops (3), phase4_draft_kernels (2), phase4_dflash
real_shape_ (3) pass. New phase4_bf16_attention: first layer passes, all-five
fails as above. No full-model verification performed. No ABI, workspace or
warmup changes required. Rebuild kernels; native branch uses shared scores
and no allocations. Performance has not been benchmarked.
"""
import subprocess

IMAGE = "lmsysorg/sglang:v0.5.19"
HEADER = "/opt/sglang/lib/python3.12/site-packages/flashinfer/data/include/flashinfer/attention/prefill.cuh"
if __name__ == "__main__":
    subprocess.run([
        "docker", "run", "--rm", "--network=none", "--entrypoint", "bash", IMAGE,
        "-lc", f"sed -n '70,96p;1593,1646p;1686,1726p;1774,1788p;4333,4455p' {HEADER}",
    ], check=True)
