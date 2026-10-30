"""Host-only tests; no torch/vLLM/transformers imports or model loading."""
import importlib.util
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace as NS
import unittest

spec = importlib.util.spec_from_file_location("vl_dump", Path(__file__).with_name("vllm_vl_dump.py"))
vl = importlib.util.module_from_spec(spec)
spec.loader.exec_module(vl)


class OracleTests(unittest.TestCase):
    def test_import_and_help_without_gpu_stack(self):
        code = f"""
import runpy, sys
class Block:
    def find_spec(self, fullname, *args):
        if fullname.split('.')[0] in ('torch', 'vllm', 'transformers', 'PIL'):
            raise RuntimeError('GPU/runtime dependency imported: ' + fullname)
sys.meta_path.insert(0, Block())
runpy.run_path({str(Path(vl.__file__).resolve())!r}, run_name='host_test')
"""
        subprocess.run([sys.executable, "-c", code], check=True)
        subprocess.run([sys.executable, vl.__file__, "--help"], check=True, capture_output=True)

    def test_decorated_vision_boundary_bypasses_module_hooks(self):
        # vLLM support_torch_compile installs this direct-forward __call__,
        # including when enforce_eager disables compilation.
        class Decorated:
            def __call__(self, *args, **kwargs):
                return self.forward(*args, **kwargs)

            def forward(self, x, *, metadata=None):
                self.metadata = metadata
                return x + 1

        module = Decorated()
        captured = []
        vl.trace_vision_boundary(module, lambda name, value: captured.append((name, value)),
                                 "blocks.0.output", "blocks.0.input")
        self.assertEqual(module(x=7, metadata="unchanged"), 8)
        self.assertEqual(module.metadata, "unchanged")
        self.assertEqual(captured, [("blocks.0.input", 7), ("blocks.0.output", 8)])
        patch = Decorated()
        vl.trace_vision_boundary(patch, lambda name, value: captured.append((name, value)),
                                 "patch_embed")
        self.assertEqual(patch(3), 4)
        self.assertEqual(captured[-1], ("patch_embed", 4))

    def test_prompt(self):
        vl.validate_prompt("user: " + vl.MARKER + "Describe this. assistant:")
        for bad in ("", "<|image_pad|>", vl.MARKER * 2,
                    vl.MARKER + "<|image_pad|>", vl.MARKER + "<|video_pad|>"):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                vl.validate_prompt(bad)

    def test_geometry(self):
        vl.validate_shape([1, 2560], [[1, 2, 2]], 2560)
        vl.validate_shape([1024, 2560], [[1, 64, 64]], 2560)
        for shape, grid in (([4, 2560], [[1, 2, 2]]), ([1, 5120], [[1, 2, 2]]),
                            ([1, 2560], []), ([1, 2560], [[2, 2, 2]]),
                            ([1, 2560], [[1, 3, 2]])):
            with self.subTest(shape=shape, grid=grid), self.assertRaises(ValueError):
                vl.validate_shape(shape, grid, 2560)

    def test_top5_excludes_extra_sampled_token(self):
        entries = {i: NS(logprob=-float(i), rank=i + 1) for i in reversed(range(6))}
        out = vl.generation_json(NS(token_ids=[5], logprobs=[entries]), 1)
        self.assertEqual(out["token_ids"], [5])
        self.assertEqual([x["token_id"] for x in out["steps"][0]["top5"]], list(range(5)))
        self.assertEqual(out["steps"][0]["logprob"], -5)

    def test_incomplete_generation_fails(self):
        for out in (NS(token_ids=[], logprobs=[]), NS(token_ids=[0], logprobs=None),
                    NS(token_ids=[0], logprobs=[{}])):
            with self.assertRaises(RuntimeError):
                vl.generation_json(out, 1)

    def test_empty_hook_fails(self):
        with self.assertRaises(RuntimeError):
            vl.capture_status(NS(rank=0, _w34_vl_capture={"count": 0}))
        with self.assertRaises(RuntimeError):
            vl.capture_status(NS(rank=0, _w34_vl_capture={"count": 1, "text_count": 0}))
        state = {"count": 1, "text_count": 1}
        self.assertIs(vl.capture_status(NS(_w34_vl_capture=state)), state)

    def test_processor_bounds(self):
        self.assertEqual(vl.PROCESSOR, dict(min_pixels=1024, max_pixels=1048576,
                                          patch_size=16, merge_size=2, temporal_patch_size=2))


if __name__ == "__main__":
    unittest.main()
