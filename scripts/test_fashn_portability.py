"""Transfer-integrity tests and optional real native noise-input rejection tests."""
import json
import os
from pathlib import Path
import struct
import subprocess
import tempfile
import unittest
from unittest.mock import patch

import package_fashn_portability as package
from run_fashn_portability import pe_machine


class BundleTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        model = self.root / "weights" / "model-bf16.gguf"
        model.parent.mkdir()
        model.write_bytes(b"unit-test-not-a-model")
        self.model = model
        self.manifest = {
            "schema": "fashn-portability-v1", "base_commit": package.BASE_COMMIT,
            "ggml_commit": package.GGML_COMMIT, "total_bytes": model.stat().st_size,
            "files": [{"path": "weights/model-bf16.gguf", "bytes": model.stat().st_size,
                       "sha256": package.digest(model)}],
        }
        self.addCleanup(patch.stopall)
        patch.object(package, "MODEL_SHA256", package.digest(model)).start()
        patch.object(package, "required_paths", return_value={"weights/model-bf16.gguf"}).start()
        self.write_manifest()

    def write_manifest(self):
        (self.root / "bundle.json").write_text(json.dumps(self.manifest), encoding="utf-8")

    def test_verify_and_detect_corruption(self):
        package.verify(self.root)
        self.model.write_bytes(b"corrupted")
        with self.assertRaisesRegex(ValueError, "integrity"):
            package.verify(self.root)

    def test_missing_entry_and_duplicate(self):
        with patch.object(package, "required_paths", return_value={"missing"}):
            with self.assertRaisesRegex(ValueError, "missing required"):
                package.verify(self.root)
        self.manifest["files"] *= 2
        self.write_manifest()
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            package.verify(self.root)

    def test_reject_escaping_paths(self):
        for value in ("../secret", "/absolute", "C:/secret", "folder\\file", "a/../b", "a//b"):
            with self.subTest(value=value), self.assertRaises(ValueError):
                package.contained(self.root, value)

    def test_wrong_base_and_byte_count(self):
        self.manifest["total_bytes"] += 1
        self.write_manifest()
        with self.assertRaisesRegex(ValueError, "byte count"):
            package.verify(self.root)
        self.manifest["base_commit"] = "other"
        self.write_manifest()
        with self.assertRaisesRegex(ValueError, "incompatible"):
            package.verify(self.root)

    def test_pe_architecture(self):
        exe = self.root / "test.exe"
        header = bytearray(134)
        header[:2] = b"MZ"
        struct.pack_into("<I", header, 0x3C, 128)
        header[128:132] = b"PE\0\0"
        struct.pack_into("<H", header, 132, 0xAA64)
        exe.write_bytes(header)
        self.assertEqual(pe_machine(exe), 0xAA64)
        exe.write_bytes(b"not executable")
        with self.assertRaises(ValueError):
            pe_machine(exe)


@unittest.skipUnless(os.environ.get("FASHN_TRAJECTORY_EXE"), "Set FASHN_TRAJECTORY_EXE for native input tests")
class NativeNoiseTests(unittest.TestCase):
    def test_empty_noise_path_is_not_a_silent_rng_fallback(self):
        with tempfile.TemporaryDirectory() as directory:
            command = [os.environ["FASHN_TRAJECTORY_EXE"], "missing-model", "missing-conditions",
                       str(Path(directory) / "output"), "2", "1.5", "1.5", "1", "42", "1", "1",
                       "--initial-noise", ""]
            result = subprocess.run(command, capture_output=True, text=True, timeout=30)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("Initial noise path must not be empty", result.stdout + result.stderr)

    def test_valid_shapes_reach_model_loading(self):
        import numpy as np
        from safetensors.numpy import save_file

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for index, shape in enumerate(((1, 3, 864, 576), (3, 864, 576))):
                path = root / f"valid-{index}.safetensors"
                save_file({"noise": np.zeros(shape, dtype=np.float32)}, str(path))
                output = root / f"output-{index}"
                command = [os.environ["FASHN_TRAJECTORY_EXE"], "missing-model", "missing-conditions",
                           str(output), "2", "1.5", "1.5", "1", "42", "1", "1",
                           "--initial-noise", str(path)]
                result = subprocess.run(command, capture_output=True, text=True, timeout=30)
                self.assertNotEqual(result.returncode, 0)
                self.assertTrue(output.is_dir(), result.stdout + result.stderr)
                self.assertNotIn("Initial noise must", result.stdout + result.stderr)

    def test_invalid_noise_rejected_before_model_loading(self):
        import numpy as np
        from safetensors.numpy import save_file

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            valid = np.zeros((1, 3, 864, 576), dtype=np.float32)
            tests = [
                ({"other": valid}, "must contain only"),
                ({"noise": valid.astype(np.float16)}, "must be F32"),
                ({"noise": np.zeros((1, 3, 2, 2), dtype=np.float32)}, "must be F32"),
                ({"noise": np.full_like(valid, np.nan)}, "finite F32"),
                ({"noise": np.full_like(valid, np.inf)}, "finite F32"),
            ]
            for index, (tensors, error) in enumerate(tests):
                path = root / f"noise-{index}.safetensors"
                save_file(tensors, str(path))
                output = root / f"output-{index}"
                command = [os.environ["FASHN_TRAJECTORY_EXE"], "missing-model", "missing-conditions",
                           str(output), "2", "1.5", "1.5", "1", "42", "1", "1",
                           "--initial-noise", str(path)]
                result = subprocess.run(command, capture_output=True, text=True, timeout=30)
                self.assertNotEqual(result.returncode, 0)
                self.assertIn(error, result.stdout + result.stderr)
                self.assertFalse(output.exists())


@unittest.skipUnless(os.environ.get("FASHN_BUNDLE"), "Set FASHN_BUNDLE for prepared input lineage")
class PreparedBundleTests(unittest.TestCase):
    def test_four_prepared_cases_match_float_conditions(self):
        import numpy as np
        from PIL import Image
        from safetensors.numpy import load_file

        root = Path(os.environ["FASHN_BUNDLE"])
        for case in ("black-shirt", "cardigan", "bottoms", "dress"):
            prepared_root = root / "cases" / case / "prepared"
            manifest = json.loads((prepared_root / "manifest.json").read_text(encoding="utf-8"))
            tensors = load_file(str(prepared_root.parent / "conditions.safetensors"))
            for key, tensor_name in zip(package.IMAGE_KEYS,
                                        ("ca_images", "garment_images", "person_poses", "garment_poses")):
                with Image.open(prepared_root / manifest[key]) as image:
                    self.assertEqual(image.size, (576, 864))
                    self.assertEqual(image.mode, "RGB" if key in package.IMAGE_KEYS[:2] else "L")
                    array = np.asarray(image).astype(np.float32)
                normalized = array / np.float32(127.5) - np.float32(1)
                normalized = (normalized.transpose(2, 0, 1)[None] if array.ndim == 3
                              else normalized[None, None])
                self.assertTrue(np.array_equal(normalized, tensors[tensor_name]), case + ":" + key)


if __name__ == "__main__":
    unittest.main()
