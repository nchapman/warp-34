"""Scoped fixture rewrite tests: CPU only, no vLLM import or CUDA runtime."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

try:
    import torch
    from safetensors import safe_open
    from safetensors.torch import save_file
except ModuleNotFoundError:
    torch = None
    safe_open = None
    save_file = None

import vllm_dump


class FusedGemmaRefreshTests(unittest.TestCase):
    def test_help_without_site_packages(self):
        result = subprocess.run(
            [sys.executable, "-S", str(Path(vllm_dump.__file__)), "--help"],
            capture_output=True, text=True, check=True,
        )
        self.assertIn("--refresh-fused-gemma", result.stdout)

    @unittest.skipIf(torch is None or safe_open is None, "torch/safetensors not installed")
    def test_refresh_preserves_other_bits_and_metadata(self):
        tensors = {
            "gemma_rms_norm.x": torch.tensor([[1., 2.]], dtype=torch.float16),
            "gemma_rms_norm.w": torch.tensor([.25, -.25], dtype=torch.float16),
            "fused_add_gemma_rms_norm.res_in": torch.tensor([[.1, .2]], dtype=torch.float16),
            "fused_add_gemma_rms_norm.out": torch.zeros(1, 2, dtype=torch.float16),
            "fused_add_gemma_rms_norm.res_out": torch.zeros(1, 2, dtype=torch.float16),
            "unrelated": torch.tensor([-0., float("nan")]),
        }
        scope = ["fused_add_gemma_rms_norm.out", "fused_add_gemma_rms_norm.res_out"]
        outputs = (torch.ones(1, 2, dtype=torch.float16),
                   torch.full((1, 2), 3., dtype=torch.float16))
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "fixture.safetensors"
            save_file(tensors, str(path), metadata={"existing": "untouched"})
            with patch.object(vllm_dump, "capture_fused_gemma", return_value=(outputs, {"scope": scope})) as capture:
                vllm_dump.refresh_fused_gemma(path)
            for actual, key in zip(capture.call_args.args, ["gemma_rms_norm.x", "fused_add_gemma_rms_norm.res_in", "gemma_rms_norm.w"]):
                self.assertTrue(torch.equal(actual, tensors[key]))
            with safe_open(str(path), framework="pt") as f:
                self.assertEqual(set(f.keys()), set(tensors))
                self.assertEqual(f.metadata()["existing"], "untouched")
                self.assertIn("source_file_sha256", json.loads(f.metadata()["fused_add_gemma_native_capture"]))
                for key, old in tensors.items():
                    expected = outputs[scope.index(key)] if key in scope else old
                    self.assertTrue(torch.equal(f.get_tensor(key).view(torch.uint8), expected.view(torch.uint8)), key)


if __name__ == "__main__":
    unittest.main()
