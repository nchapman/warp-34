import http.server
import io
import json
from pathlib import Path
import tempfile
import threading
import time
import unittest
from unittest.mock import patch

from w34perf.__main__ import main
from w34perf.artifacts import save_inventory
from w34perf.config import digest, load
from w34perf.preflight import preflight
from w34perf.transport import models
from test_perf import ROOT


class ModelServer:
    def __init__(self, mode="normal", model_id="served"):
        self.requests = []
        outer = self

        class Handler(http.server.BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def do_GET(self):
                outer.requests.append((self.path, self.headers.get("Authorization")))
                if mode == "error":
                    self.send_error(401, "private upstream error")
                    return
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                try:
                    if mode == "trickle":
                        for _ in range(200):
                            self.wfile.write(b" ")
                            self.wfile.flush()
                            time.sleep(0.005)
                        return
                    data = {"data": [{"id": model_id}]}
                    raw = {"nan": b'{"data":NaN}', "bad": b'invalid-json',
                           "oversized": b' ' * (1024 * 1024 + 1),
                           "shape": b'{"data":"not an array"}'}.get(mode, json.dumps(data).encode())
                    self.wfile.write(raw)
                    self.wfile.flush()
                except (BrokenPipeError, ConnectionResetError):
                    pass

        self.server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.engine = {"id": "local", "model": "served", "gpu_count": 1,
                       "base_url": f"http://127.0.0.1:{self.server.server_port}/v1"}

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join()


def snapshot(tmp):
    root = tmp / "snapshot"
    root.mkdir()
    (root / "model.safetensors").write_bytes(b"weights")
    (root / "model.safetensors.index.json").write_text(json.dumps({"weight_map": {"a": "model.safetensors"}}))
    path = tmp / "inventory.json"
    save_inventory(root, path)
    return root, path


def config(engine):
    return {"engines": [engine], "run": {"timeout_s": 2}}


class PreflightTests(unittest.TestCase):
    def test_discovery_uses_openai_path_and_auth_without_persisting_headers(self):
        with ModelServer() as server, patch.dict("os.environ", {"PERF_TEST_KEY": "sensitive"}):
            response = models({**server.engine, "api_key_env": "PERF_TEST_KEY"}, 2)
            self.assertEqual(response["status"], "ok")
            self.assertEqual(response["data"]["data"][0]["id"], "served")
            self.assertEqual(server.requests, [("/v1/models", "Bearer sensitive")])
            self.assertNotIn("sensitive", json.dumps(response))

    def test_discovery_rejects_error_malformed_and_oversized_responses(self):
        for mode in ("error", "nan", "bad", "oversized"):
            with self.subTest(mode=mode), ModelServer(mode) as server:
                response = models(server.engine, 2)
                self.assertEqual(response["status"], "error")
                self.assertIsNone(response["data"])
                self.assertNotIn("private upstream", json.dumps(response))

    def test_discovery_total_deadline_and_invalid_credentials(self):
        with ModelServer("trickle") as server:
            response = models(server.engine, 0.05)
            self.assertEqual(response["status"], "error")
            self.assertLess(response["e2e_ms"], 500)
        with ModelServer() as server, patch.dict("os.environ", {"PERF_TEST_KEY": "sensitive\nkey"}):
            response = models({**server.engine, "api_key_env": "PERF_TEST_KEY"}, 2)
            self.assertEqual(response["status"], "error")
            self.assertEqual(server.requests, [])
            self.assertNotIn("sensitive", json.dumps(response))

    def test_passed_preflight_retains_advertised_ids_and_inventory_hash(self):
        with tempfile.TemporaryDirectory() as tmp, ModelServer() as server:
            tmp = Path(tmp)
            _, inv = snapshot(tmp)
            server.engine["artifact_sha256"] = json.loads(inv.read_text())["inventory_sha256"]
            report = preflight(config(server.engine), "local", inv, tmp / "passed.json")
            self.assertEqual(report["status"], "passed")
            self.assertEqual(report["checks"]["indexed_weights"], ["model.safetensors"])
            self.assertEqual(report["checks"]["advertised_model_ids"], ["served"])
            self.assertEqual(report["checks"]["inventory"]["status"], "verified")
            self.assertEqual(json.loads((tmp / "passed.json").read_text()), report)
            with self.assertRaises(FileExistsError):
                preflight(config(server.engine), "local", inv, tmp / "passed.json")
            self.assertEqual(len(server.requests), 1)

    def test_model_id_or_response_shape_mismatch_is_retained(self):
        for mode in ("normal", "shape"):
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as tmp, ModelServer(mode) as server:
                tmp = Path(tmp)
                _, inv = snapshot(tmp)
                server.engine["model"] = "wrong"
                report = preflight(config(server.engine), "local", inv, tmp / "failed.json")
                self.assertEqual(report["status"], "failed")
                self.assertIn("error", report)
                self.assertEqual(json.loads((tmp / "failed.json").read_text())["status"], "failed")

    def test_drift_and_profile_hash_mismatch_fail_before_network(self):
        for mode in ("drift", "hash"):
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as tmp, ModelServer() as server:
                tmp = Path(tmp)
                root, inv = snapshot(tmp)
                if mode == "drift":
                    (root / "model.safetensors").write_bytes(b"changed")
                else:
                    server.engine["artifact_sha256"] = "0" * 64
                report = preflight(config(server.engine), "local", inv, tmp / "failed.json")
                self.assertEqual(report["status"], "failed")
                self.assertEqual(server.requests, [])

    def test_missing_unhashed_and_traversal_shards_fail_before_network(self):
        for mode in ("missing", "unhashed", "traversal"):
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as tmp, ModelServer() as server:
                tmp = Path(tmp)
                root, _ = snapshot(tmp)
                name = "../outside.safetensors" if mode == "traversal" else "missing.safetensors"
                (root / "model.safetensors.index.json").write_text(json.dumps({"weight_map": {"a": name}}))
                patterns = ["*.safetensors"] if mode == "unhashed" else None
                inv = tmp / "updated.json"
                save_inventory(root, inv, patterns)
                report = preflight(config(server.engine), "local", inv, tmp / "failed.json")
                self.assertEqual(report["status"], "failed")
                self.assertEqual(server.requests, [])

    def test_interrupted_inventory_read_retains_interrupted_evidence(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "interrupted.json"
            with patch.object(Path, "read_text", side_effect=KeyboardInterrupt), self.assertRaises(KeyboardInterrupt):
                preflight(config({"id": "probe"}), "probe", Path(tmp) / "inventory.json", output)
            self.assertEqual(json.loads(output.read_text())["status"], "interrupted")

    def test_disabled_profile_skips_without_accessing_inventory_or_network(self):
        with tempfile.TemporaryDirectory() as tmp, patch("w34perf.preflight.models") as discover:
            report = preflight(config({"id": "skip", "enabled": False, "skip_reason": "unsupported release"}),
                               "skip", Path(tmp) / "absent", Path(tmp) / "skipped.json")
            self.assertEqual(report["status"], "skipped")
            discover.assert_not_called()

    def test_qualification_preset_pins_inventory_and_labels_precision_and_failures(self):
        cfg = load(ROOT / "qualification" / "27b-one-card.toml")
        inventory = json.loads((ROOT / "qualification" / "27b-gptq-0df.inventory.json").read_text())
        self.assertEqual(digest(inventory["files"]), inventory["inventory_sha256"])
        runnable = [e for e in cfg["engines"] if e["enabled"]]
        self.assertEqual(len(runnable), 3)
        self.assertEqual({e["artifact_sha256"] for e in runnable}, {inventory["inventory_sha256"]})
        self.assertEqual({e["model_revision"] for e in runnable}, {"0df81eadcfd59601cb82ccece2f16ea7d39244bf"})
        self.assertEqual({e["kv_format"] for e in runnable}, {"bf16", "fp16", "fp16-auto"})
        failed = [e for e in cfg["engines"] if not e["enabled"]]
        self.assertEqual(len(failed), 2)
        self.assertTrue(all(e["skip_reason"].startswith("Observed") for e in failed))
        old_failure = next(e for e in failed if "c11-failed" in e["id"])
        self.assertNotEqual(old_failure["artifact_sha256"], inventory["inventory_sha256"])
        self.assertEqual(old_failure["model_revision"], "c11a884fe4b659eccdce66761f26d3775cf7696c")

    def test_cli_preflight_creates_evidence_and_returns_failure_for_wrong_id(self):
        with tempfile.TemporaryDirectory() as tmp, ModelServer() as server, patch("sys.stdout", io.StringIO()):
            tmp = Path(tmp)
            _, inv = snapshot(tmp)
            text = (ROOT / "suite.toml").read_text()
            text = text.replace('http://127.0.0.1:8000/v1', server.engine["base_url"])
            text = text.replace('"fixtures/', '"' + str(ROOT / "fixtures") + '/')
            path = tmp / "suite.toml"
            path.write_text(text)
            args = ["preflight", str(path), "--engine", "warp34-1-fp16", "--inventory", str(inv),
                    "--out", str(tmp / "failed.json")]
            self.assertEqual(main(args), 1)
            self.assertEqual(json.loads((tmp / "failed.json").read_text())["status"], "failed")
            args[args.index("--inventory") + 1] = str(tmp / "absent.json")
            args[args.index("--out") + 1] = str(tmp / "missing-inventory.json")
            self.assertEqual(main(args), 1)
            self.assertEqual(json.loads((tmp / "missing-inventory.json").read_text())["status"], "failed")
            self.assertEqual(len(server.requests), 1)
