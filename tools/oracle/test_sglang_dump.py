"""Fail-closed/provenance tests for the packer; no GPU or donor installation."""
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from safetensors import safe_open
from safetensors.torch import save_file
import torch

from sglang_dump import SOURCES, bundle


class BundleTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.out = self.root / "bundle.safetensors"
        for i, name in enumerate(SOURCES):
            save_file({f"source{i}": torch.tensor([i], dtype=torch.bfloat16)},
                      str(self.root / name),
                      metadata={"provenance": json.dumps({"sglang": "0.5.19", "source": i})})

    def test_preserves_values_and_per_capture_provenance(self):
        bundle(self.root, self.out)
        with safe_open(str(self.out), framework="pt") as f:
            metadata = json.loads(f.metadata()["provenance"])
            self.assertEqual(len(f.keys()), len(SOURCES))
            for i, name in enumerate(SOURCES):
                self.assertEqual(f.get_tensor(f"source{i}").item(), i)
                self.assertEqual(metadata["sources"][name]["provenance"]["source"], i)
                self.assertEqual(metadata["sources"][name]["sha256"],
                                 hashlib.sha256((self.root / name).read_bytes()).hexdigest())

    def test_missing_source_leaves_existing_output_untouched(self):
        self.out.write_bytes(b"existing")
        (self.root / SOURCES[-1]).unlink()
        with self.assertRaises(FileNotFoundError):
            bundle(self.root, self.out)
        self.assertEqual(self.out.read_bytes(), b"existing")

    def test_rejects_missing_provenance(self):
        save_file({"other": torch.zeros(1)}, str(self.root / SOURCES[-1]))
        with self.assertRaises(KeyError):
            bundle(self.root, self.out)
        self.assertFalse(self.out.exists())

    def test_rejects_wrong_donor(self):
        save_file({"other": torch.zeros(1)}, str(self.root / SOURCES[-1]),
                  metadata={"provenance": json.dumps({"sglang": "other"})})
        with self.assertRaisesRegex(ValueError, "Unexpected SGLang donor"):
            bundle(self.root, self.out)

    def test_rejects_duplicate_tensor_names(self):
        save_file({"source0": torch.zeros(1)}, str(self.root / SOURCES[-1]),
                  metadata={"provenance": json.dumps({"sglang": "0.5.19"})})
        with self.assertRaisesRegex(ValueError, "Duplicate captured tensor"):
            bundle(self.root, self.out)

    def test_rejects_source_overwrite(self):
        with self.assertRaisesRegex(ValueError, "overwrite a source"):
            bundle(self.root, self.root / SOURCES[0])


if __name__ == "__main__":
    unittest.main()
