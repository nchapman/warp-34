#!/usr/bin/env python3
"""Compare the native HGPTQ input transform with vLLM's Hadacore on one GPU.

Build the standalone probe library (no engine weights required):
  /opt/cuda/bin/nvcc -std=c++17 -O3 --shared -Xcompiler -fPIC \
    -gencode arch=compute_86,code=sm_86 -gencode arch=compute_89,code=sm_89 \
    crates/w34-kernels/csrc/ops/op_hadamard.cu -o /tmp/w34_hadamard.so
Run on an idle card:
  CUDA_VISIBLE_DEVICES=0 python3 tools/oracle/hgptq_hadamard_probe.py \
    --library /tmp/w34_hadamard.so

Timings use CUDA graphs for both implementations and exclude Python launch
latency. This is an op-level comparison, not an end-to-end HGPTQ benchmark.
"""

import argparse
import ctypes
import json


class Exec(ctypes.Structure):
    _fields_ = [
        ("stream", ctypes.c_size_t),
        ("scratch_ptr", ctypes.c_uint64),
        ("scratch_size", ctypes.c_size_t),
    ]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--library", required=True)
    args = parser.parse_args()

    import torch
    from vllm import _custom_ops as ops

    lib = ctypes.CDLL(args.library)
    transform = lib.w34_block_hadamard_f16
    transform.argtypes = [ctypes.POINTER(Exec), ctypes.c_void_p, ctypes.c_void_p] + [ctypes.c_int] * 4
    transform.restype = ctypes.c_int
    torch.manual_seed(42)
    stream = torch.cuda.Stream()

    def benchmark(fn):
        for _ in range(5):
            fn()
        stream.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            for _ in range(100):
                fn()
        graph.replay()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record(stream)
        for _ in range(10):
            graph.replay()
        end.record(stream)
        end.synchronize()
        return start.elapsed_time(end)  # ms / 1000 launches * 1000 = us/launch

    with torch.cuda.stream(stream):
        ex = Exec(stream.cuda_stream, 0, 0)
        for block, k, padded_k in [(64, 2048, 2048), (64, 6144, 6144), (16, 4304, 4352)]:
            for rows in [1, 8, 16, 512, 4096]:
                x = torch.randn(rows, k, device="cuda", dtype=torch.float16)
                original = x.clone()
                out = torch.empty(rows, padded_k, device="cuda", dtype=torch.float16)

                def native():
                    rc = transform(ctypes.byref(ex), x.data_ptr(), out.data_ptr(), rows, k, padded_k, block)
                    if rc:
                        raise RuntimeError(f"w34_block_hadamard_f16 returned {rc}")
                    return out

                def donor():
                    return ops.hadacore_transform(x.unflatten(-1, (-1, block)), inplace=False).flatten(-2)

                native()
                expected = donor()
                stream.synchronize()
                torch.testing.assert_close(x, original, atol=0, rtol=0)
                torch.testing.assert_close(out[:, k:], torch.zeros_like(out[:, k:]), atol=0, rtol=0)
                # Hadacore tensor-core accumulation may round differently from FP32
                # butterflies. Report actual error as well as enforcing a bound.
                delta = out[:, :k].float() - expected.float()
                rel_l2 = (delta.norm() / expected.float().norm()).item()
                max_abs = delta.abs().max().item()
                if not torch.isfinite(delta).all() or rel_l2 > 0.002:
                    raise AssertionError(f"Hadacore mismatch: rel_l2={rel_l2}, max_abs={max_abs}")
                native_us = benchmark(native)
                donor_us = benchmark(donor)
                print(json.dumps({
                    "gpu": torch.cuda.get_device_name(), "block": block, "rows": rows,
                    "k": k, "padded_k": padded_k, "max_abs": max_abs, "rel_l2": rel_l2,
                    "native_us": native_us, "hadacore_us": donor_us,
                }), flush=True)


if __name__ == "__main__":
    main()
