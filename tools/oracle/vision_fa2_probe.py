"""Exact-bit FA2 production adapter probe (GPU0 only, no model engine).

Usage: CUDA_VISIBLE_DEVICES=0 python tools/oracle/vision_fa2_probe.py \
    --library /tmp/w34-vision-fa2.so --capture /tmp/w34-attn-investigation/native-attn.safetensors
The shared library must contain op_vision_fa2.cu and op_vision.cu.
"""
import argparse
import ctypes as C
import math
import itertools
import os

assert os.environ.get("CUDA_VISIBLE_DEVICES") == "0", "GPU0 only"
import torch
from safetensors.torch import load_file
from vllm.vllm_flash_attn import flash_attn_varlen_func
from vllm.config import VllmConfig, set_current_vllm_config
from vllm.model_executor.layers.rotary_embedding.base import RotaryEmbedding
from vllm.model_executor.layers.rotary_embedding.common import ApplyRotaryEmb


class Exec(C.Structure):
    _fields_ = [("stream", C.c_uint64), ("scratch", C.c_uint64), ("size", C.c_size_t)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--library", required=True)
    ap.add_argument("--capture", required=True)
    args = ap.parse_args()
    lib = C.CDLL(args.library)
    fn = lib.w34_vision_fa2
    fn.argtypes = [C.POINTER(Exec), C.c_void_p, C.c_void_p, C.c_void_p] + [C.c_int] * 4
    fn.restype = C.c_int
    assert lib.w34_vision_fa2_prepare() == 0

    def check(q, k, v, cu, gold, label):
        q, k, v, gold = [x.reshape(-1, x.shape[-2], x.shape[-1]) for x in (q, k, v, gold)]
        n, h, d = q.shape
        packed = torch.stack((q, k, v), dim=1).contiguous()
        out = torch.empty_like(q)
        scratch = torch.empty(n * h, device="cuda", dtype=torch.float32)
        ex = Exec(torch.cuda.current_stream().cuda_stream, scratch.data_ptr(), scratch.numel() * 4)

        def run():
            assert fn(C.byref(ex), packed.data_ptr(), cu.data_ptr(), out.data_ptr(), n, cu.numel()-1, h, d) == 0

        run()
        torch.cuda.synchronize()
        diff = (out.view(torch.int16) != gold.contiguous().view(torch.int16)).sum().item()
        print(f"{label}: {diff}/{out.numel()} differing half bits", flush=True)
        assert diff == 0
        assert scratch.isfinite().all()
        # Exercise the real fused entrypoint with identity RoPE and no bias.
        # Inputs here are already post-RoPE; zero positions leave them unchanged.
        positions = torch.zeros((n, 2), device="cuda", dtype=torch.int32)
        fused = lib.w34_vision_qkv_rope_attn_f16
        fused.argtypes = [C.POINTER(Exec)] + [C.c_void_p] * 5 + [C.c_int] * 4 + [C.c_float]
        assert fused(C.byref(ex), packed.data_ptr(), None, positions.data_ptr(), cu.data_ptr(), out.data_ptr(), n, cu.numel()-1, h, d, 10000.0) == 0
        torch.cuda.synchronize()
        assert torch.equal(out.view(torch.int16), gold.contiguous().view(torch.int16))
        # Exercise actual graph capture/replay, with caller-owned persistent scratch.
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        ex.stream = stream.cuda_stream
        with torch.cuda.stream(stream):
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, stream=stream):
                run()
            out.fill_(float("nan"))
            graph.replay()
        stream.synchronize()
        assert torch.equal(out.view(torch.int16), gold.contiguous().view(torch.int16))
        ex.size = 0
        assert fn(C.byref(ex), packed.data_ptr(), cu.data_ptr(), out.data_ptr(), n, cu.numel()-1, h, d) != 0

    t = load_file(args.capture, device="cuda")
    check(t["query"], t["key"], t["value"], t["cu_seqlens"], t["output"], "D72 full native capture")
    torch.manual_seed(20260902)
    for d in (64, 72):
        for lengths in ([1], [128], [129, 67, 256], [1200]):
            n, h = sum(lengths), 16
            q, k, v = [torch.randn(n, h, d, dtype=torch.float16, device="cuda") for _ in range(3)]
            cu = torch.tensor([0] + list(itertools.accumulate(lengths)), device="cuda", dtype=torch.int32)
            gold = flash_attn_varlen_func(q, k, v, cu_seqlens_q=cu, cu_seqlens_k=cu, max_seqlen_q=max(lengths), max_seqlen_k=max(lengths), softmax_scale=1.0 / math.sqrt(d), causal=False)
            check(q, k, v, cu, gold, f"D{d} native lengths={lengths}")

        # Nonzero axial positions exercise the complete production path, not just
        # FA2 on already-rotated inputs. Native small-angle sin must retain its
        # FP16 cache precision even though our CUDA file uses -use_fast_math.
        n, h = 1200, 16
        positions = torch.stack((torch.arange(n) // 40, torch.arange(n) % 40), dim=1).to(device="cuda", dtype=torch.int32)
        packed = torch.randn(n, 3, h, d, device="cuda", dtype=torch.float16)
        with set_current_vllm_config(VllmConfig()), torch.device("cuda"):
            rope = RotaryEmbedding(d, d // 2, 8192, 10000.0, True, torch.float16)
            rotary = ApplyRotaryEmb(enforce_enable=True)
        cos, sin = rope.cos_sin_cache.chunk(2, dim=-1)
        cos, sin = [torch.cat((x[positions[:, 0].long()], x[positions[:, 1].long()]), dim=-1).contiguous() for x in (cos, sin)]
        rotated = packed.clone()
        rotated[:, :2] = rotary.forward_cuda(packed[:, :2].reshape(n, 2*h, d).contiguous(), cos, sin).reshape(n, 2, h, d)
        cu = torch.tensor([0, n], device="cuda", dtype=torch.int32)
        gold = flash_attn_varlen_func(*[rotated[:, i].contiguous() for i in range(3)], cu_seqlens_q=cu, cu_seqlens_k=cu, max_seqlen_q=n, max_seqlen_k=n, softmax_scale=1.0 / math.sqrt(d), causal=False)
        out = torch.empty(n, h, d, device="cuda", dtype=torch.float16)
        scratch = torch.empty(n*h, device="cuda", dtype=torch.float32)
        ex = Exec(torch.cuda.current_stream().cuda_stream, scratch.data_ptr(), scratch.numel()*4)
        fused = lib.w34_vision_qkv_rope_attn_f16
        assert fused(C.byref(ex), packed.data_ptr(), None, positions.data_ptr(), cu.data_ptr(), out.data_ptr(), n, 1, h, d, 10000.0) == 0
        torch.cuda.synchronize()
        assert torch.equal(packed.view(torch.int16), rotated.view(torch.int16)), f"D{d} nonzero axial RoPE"
        assert torch.equal(out.view(torch.int16), gold.contiguous().view(torch.int16)), f"D{d} nonzero axial attention"
        print(f"D{d} nonzero axial RoPE + FA2: exact", flush=True)


if __name__ == "__main__":
    main()
