import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import io

from w34perf.__main__ import main
from w34perf.artifacts import inventory, save_inventory, verify_inventory


class ArtifactTests(unittest.TestCase):
    def test_snapshot_inventory_hashes_symlink_contents_and_all_visible_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            blob = tmp / "blob"
            blob.write_bytes(b"quantized weight bytes")
            root = tmp / "snapshot"
            root.mkdir()
            (root / "model.safetensors").symlink_to(blob)
            (root / "config.json").write_text('{}')
            (root / ".cache").mkdir()
            (root / ".cache/state").write_text("not model data")
            result = save_inventory(root, tmp / "inventory.json")
            self.assertEqual(set(result["files"]), {"model.safetensors", "config.json"})
            self.assertEqual(result["files"]["model.safetensors"]["sha256"], hashlib.sha256(blob.read_bytes()).hexdigest())
            self.assertEqual(result["total_bytes"], blob.stat().st_size + 2)
            self.assertEqual(verify_inventory(tmp / "inventory.json")["status"], "verified")
            with self.assertRaises(FileExistsError):
                save_inventory(root, tmp / "inventory.json")
            with self.assertRaises(ValueError):
                save_inventory(root, root / "self.json")

    def test_changed_added_and_removed_files_fail_verification(self):
        for change in ("changed", "added", "removed"):
            with self.subTest(change=change), tempfile.TemporaryDirectory() as tmp:
                tmp = Path(tmp)
                root = tmp / "model"
                root.mkdir()
                (root / "config.json").write_text('{}')
                model = root / "model.gguf"
                model.write_bytes(b"weights")
                save_inventory(root, tmp / "inventory.json")
                if change == "changed":
                    model.write_bytes(b"different")
                elif change == "added":
                    (root / "extra.gguf").write_bytes(b"extra")
                else:
                    model.unlink()
                result = verify_inventory(tmp / "inventory.json")
                self.assertEqual(result["status"], "mismatch")
                self.assertEqual(result["differences"], ["extra.gguf" if change == "added" else "model.gguf"])

    def test_verification_reports_mismatch_when_every_included_file_is_deleted(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            root = tmp / "model"
            root.mkdir()
            model = root / "only.gguf"
            model.write_bytes(b"weights")
            save_inventory(root, tmp / "inventory.json")
            model.unlink()
            self.assertEqual(verify_inventory(tmp / "inventory.json")["differences"], ["only.gguf"])
            with patch("sys.stdout", io.StringIO()):
                self.assertEqual(main(["verify", str(tmp / "inventory.json")]), 1)

    def test_glob_scope_is_retained_and_inventory_can_be_relocated(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            root = tmp / "original"
            root.mkdir()
            (root / "a.ninfer").write_bytes(b"a")
            (root / "b.json").write_bytes(b"b")
            result = save_inventory(root, tmp / "inventory.json", ["*.ninfer"])
            self.assertEqual(list(result["files"]), ["a.ninfer"])
            root.rename(tmp / "relocated")
            self.assertEqual(verify_inventory(tmp / "inventory.json", tmp / "relocated")["status"], "verified")
            data = json.loads((tmp / "inventory.json").read_text())
            data["files"]["a.ninfer"]["bytes"] += 1
            (tmp / "inventory.json").write_text(json.dumps(data))
            with self.assertRaisesRegex(ValueError, "content hash"):
                verify_inventory(tmp / "inventory.json", tmp / "relocated")

    def test_single_file_symlink_keeps_logical_filename(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            blob = tmp / "opaque_blob_hash"
            blob.write_bytes(b"weights")
            model = tmp / "model.gguf"
            model.symlink_to(blob)
            self.assertEqual(list(inventory(model)["files"]), ["model.gguf"])

    def test_cli_reports_verification_failure_without_overwriting_evidence(self):
        with tempfile.TemporaryDirectory() as tmp, patch("sys.stdout", io.StringIO()):
            tmp = Path(tmp)
            model = tmp / "model.ninfer"
            model.write_bytes(b"weights")
            out = tmp / "inventory.json"
            self.assertEqual(main(["inventory", str(model), "--out", str(out)]), 0)
            self.assertEqual(main(["verify", str(out)]), 0)
            original = out.read_bytes()
            model.write_bytes(b"changed")
            self.assertEqual(main(["verify", str(out)]), 1)
            self.assertEqual(out.read_bytes(), original)
            with self.assertRaises(ValueError):
                inventory(tmp / "missing")
            with self.assertRaises(ValueError):
                inventory(tmp, ["*.missing"])
