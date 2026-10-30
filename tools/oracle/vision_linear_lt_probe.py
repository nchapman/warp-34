"""Production ABI regression and timing probe (no adapter).

Build op_init.cu as a shared library with nvcc -shared -Xcompiler -fPIC
-lcublas -lcublasLt; load lazily (unused FA2 preparation symbols).
Run under flock /tmp/w34-gpu0.lock CUDA_VISIBLE_DEVICES=0.
Pass library and native capture.pt; --expect-old demonstrates the regression.
Captured tensors are external fixtures from the pinned native oracle.
"""
import argparse
import ctypes as C
import os
import time

import torch

p = argparse.ArgumentParser(__doc__)
p.add_argument("library")
p.add_argument("capture")
p.add_argument("--expect-old", action="store_true")
a = p.parse_args()
lib = C.CDLL(a.library, mode=os.RTLD_LAZY | os.RTLD_LOCAL)
P, I, H = C.c_void_p, C.c_int, C.c_size_t
lib.w34_cublas_create.argtypes = [C.POINTER(H), H]
lib.w34_cublas_destroy.argtypes = [H]
lib.w34_cublas_gemm_f16_bias.argtypes = [H] + [P] * 4 + [I] * 3
stream = torch.cuda.Stream()
torch.cuda.set_stream(stream)
h = H()
assert lib.w34_cublas_create(C.byref(h), stream.cuda_stream) == 0
failures = []
fixtures = torch.load(a.capture, weights_only=True)
torch.manual_seed(34)
for m, n, k in [(128, 256, 512), (32, 1152, 4304)]:
    x = torch.randn(m, k, device="cuda", dtype=torch.float16)
    w = torch.randn(n, k, device="cuda", dtype=torch.float16)
    b = torch.randn(n, device="cuda", dtype=torch.float16)
    fixtures[f"synthetic_{m}_{n}_{k}"] = dict(input=x, weight=w, bias=b, output=torch.nn.functional.linear(x, w, b))
for name, d in fixtures.items():
    x, w, b, gold = [d[key].cuda() for key in ("input", "weight", "bias", "output")]
    y = torch.empty_like(gold)
    k, n = x.shape[-1], w.shape[0]
    m = x.numel() // k
    def run():
        assert lib.w34_cublas_gemm_f16_bias(h, x.data_ptr(), w.data_ptr(), b.data_ptr(), y.data_ptr(), m, n, k) == 0
    run()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        run()
    graph.replay()
    stream.synchronize()
    different = int((y.view(torch.int16) != gold.view(torch.int16)).sum())
    failures.append(different)
    start, end = [torch.cuda.Event(enable_timing=True) for _ in range(2)]
    start.record()
    for _ in range(100):
        graph.replay()
    end.record()
    end.synchronize()
    gpu_us = start.elapsed_time(end) * 10
    t = time.perf_counter()
    for _ in range(100):
        run()
    host_us = (time.perf_counter() - t) * 1e4
    stream.synchronize()
    print(name, dict(different=different, graph_us=gpu_us, host_us=host_us), flush=True)
    del graph
assert lib.w34_cublas_destroy(h) == 0
print(sorted({line.split()[-1] for line in open('/proc/self/maps') if 'libcublas' in line}))
assert any(failures) if a.expect_old else not any(failures), failures
