import json
from pathlib import Path
import tempfile
import unittest

from corridorkey.benchmark import load_manifest, sha256


class ManifestTests(unittest.TestCase):
    def test_correction_benchmark_identity_verifies_actual_assets(self):
        from test_corrections import document_data, raster_entry
        from corridorkey.benchmark import correction_identity
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            source, mask = root / "source.mp4", root / "mask.png"
            source.write_bytes(b"source hash fixture")
            mask.write_bytes(b"asset hash fixture")
            data = document_data()
            data["source"]["sha256"] = sha256(source)
            data["entries"] = [raster_entry()]
            data["entries"][0]["asset"]["sha256"] = sha256(mask)
            path = root / "corrections.json"
            path.write_text(json.dumps(data))
            result = correction_identity(path, source)
            self.assertEqual(result["assets"][0]["sha256"], sha256(mask))
            self.assertEqual(result["manifest_sha256"], sha256(path))
            mask.write_bytes(b"stale")
            with self.assertRaisesRegex(ValueError, "hash"):
                correction_identity(path, source)
    def test_rejects_wrong_version_and_insufficient_repetitions(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder, "manifest.json")
            for values in ({"schema_version": 2, "kind": "cpu_correctness"},
                           {"schema_version": 1, "kind": "inference"},
                           {"schema_version": 1, "kind": "cpu_correctness", "repetitions": 1}):
                path.write_text(json.dumps(values))
                with self.assertRaises(ValueError):
                    load_manifest(path)

    def test_asset_identity_changes_with_contents(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder, "asset")
            path.write_bytes(b"abc")
            self.assertEqual(sha256(path), "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad")
            path.write_bytes(b"abd")
            self.assertNotEqual(sha256(path), "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad")

    def test_inference_manifest_requires_explicit_model_state_and_known_fields(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder, "benchmark.json")
            value = {"schema_version": 1, "kind": "inference", "repetitions": 3,
                     "batch_manifest": "batch.json", "model_state": "warm_session"}
            path.write_text(json.dumps(value))
            self.assertEqual(load_manifest(path)["model_state"], "warm_session")
            for update in ({"model_state": "cold_model"}, {"invented_setting": True}):
                path.write_text(json.dumps(dict(value, **update)))
                with self.assertRaises(ValueError):
                    load_manifest(path)
