"""Full-row FP16 parity with installed native Torch LayerNorm.
Run on an exclusively reserved GPU, for example:
flock /tmp/w34-gpu1.lock env CUDA_VISIBLE_DEVICES=1 python tools/oracle/vision_norm_probe.py
This GPU/native-version probe is intentionally separate from CPU unittest discovery.
Compiles the production translation unit with its production fast-math setting.
Donor source: pytorch/pytorch cf30153c4c131c8164ee7798e5022d810682e2cb,
aten/src/ATen/native/cuda/layer_norm_kernel.cu and thread_constants.h.
"""
import ctypes
import os
from pathlib import Path
import subprocess
import tempfile
import unittest


class VisionNorm(unittest.TestCase):
    def test_native_full_rows(self):
        import torch
        self.assertEqual(torch.version.git_version,
                         "cf30153c4c131c8164ee7798e5022d810682e2cb")
        root = Path(__file__).resolve().parents[2]
        class Exec(ctypes.Structure):
            _fields_ = [("stream", ctypes.c_size_t), ("scratch", ctypes.c_uint64),
                        ("size", ctypes.c_size_t)]
        with tempfile.TemporaryDirectory() as tmp:
            libpath = str(Path(tmp) / "vision.so")
            subprocess.run(["/opt/cuda/bin/nvcc", "-shared", "-Xcompiler", "-fPIC",
                            "-O3", "-use_fast_math", "-std=c++17",
                            "-gencode=arch=compute_86,code=sm_86",
                            "-gencode=arch=compute_89,code=sm_89",
                            "-gencode=arch=compute_89,code=compute_89",
                            str(root / "crates/w34-kernels/csrc/ops/op_vision.cu"),
                            "-o", libpath], check=True)
            lib = ctypes.CDLL(libpath, mode=os.RTLD_LAZY)
            norm = lib.w34_vision_layernorm_f16
            norm.argtypes = [ctypes.POINTER(Exec)] + [ctypes.c_void_p]*4 + [ctypes.c_int]*2 + [ctypes.c_float]
            norm.restype = ctypes.c_int
            torch.manual_seed(7429)
            for width in (768, 1024, 1152):
                # Full rows, ordinary and offset/low-variance inputs; no sampled comparison.
                for scale, offset in ((1., 0.), (3., -0.5), (0.01, 1.)):
                    x = (torch.randn(37, width, device="cuda")*scale+offset).half()
                    w = torch.randn(width, device="cuda").half()
                    b = torch.randn(width, device="cuda").half()
                    for affine in (True, False):
                        weight, bias = (w, b) if affine else (None, None)
                        expected = torch.nn.functional.layer_norm(x, (width,), weight, bias, 1e-6)
                        actual = torch.empty_like(x)
                        ex = Exec(torch.cuda.current_stream().cuda_stream, 0, 0)
                        self.assertEqual(norm(ctypes.byref(ex), x.data_ptr(),
                                              w.data_ptr() if affine else None,
                                              b.data_ptr() if affine else None,
                                              actual.data_ptr(), len(x), width, 1e-6), 0)
                        torch.cuda.synchronize()
                        mismatches = (actual.view(torch.int16) != expected.view(torch.int16)).sum().item()
                        self.assertEqual(mismatches, 0, (width, scale, offset, affine,
                                         (actual.float()-expected.float()).abs().max().item()))
                        print("exact", width, scale, offset, affine, x.numel(), flush=True)


if __name__ == "__main__":
    unittest.main()
