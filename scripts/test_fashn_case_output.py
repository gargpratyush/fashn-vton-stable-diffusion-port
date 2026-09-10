import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image
from safetensors.numpy import save_file

import compare_fashn_case_output as case


def normalized(image, _):
    array = image.transpose(2, 0, 1) if image.ndim == 3 else image[None]
    array = array[None].astype(np.float32) / np.float32(127.5) - np.float32(1)
    return SimpleNamespace(numpy=lambda: array)


class CaseOutputTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        root = Path(self.temp.name)
        self.args = SimpleNamespace(source=root / "source", reference=root / "reference",
                                    reference_inputs=root / "reference-inputs", native_run=root / "native",
                                    prepared=root / "prepared" / "manifest.json", output=root / "comparison")
        for directory in (self.args.reference, self.args.reference_inputs, self.args.native_run, self.args.prepared.parent):
            directory.mkdir()
        self.crop = {"x": 1, "y": 1, "width": 3, "height": 2}
        prepared = {"category": "bottoms", "crop": self.crop}
        conditions, hashes = {}, {}
        names = {"ca_image": "ca_images", "garment_image": "garment_images",
                 "person_pose": "person_poses", "garment_pose": "garment_poses"}
        for key, plural in names.items():
            shape = (4, 5) if "pose" in key else (4, 5, 3)
            image = np.arange(np.prod(shape), dtype=np.uint8).reshape(shape)
            prepared[key] = key + ".png"
            for directory in (self.args.prepared.parent, self.args.reference_inputs):
                Image.fromarray(image).save(directory / prepared[key])
            conditions[plural] = normalized(image, None).numpy()
            hashes[key] = case.sha256(self.args.prepared.parent / prepared[key])
        save_file(conditions, str(self.args.reference_inputs / "conditions.safetensors"))
        case.write_report(self.args.prepared, prepared)
        final = np.linspace(-1, 1, 60, dtype=np.float32).reshape(1, 3, 4, 5)
        save_file({"image": final}, str(self.args.reference / "step-1.safetensors"))
        Image.fromarray(case.pixels(final)).save(self.args.reference / "expected.png")
        self.expected = case.crop_pixels(case.pixels(final), self.crop)
        self.set_native(self.expected)
        self.reference = {
            "source_revision": case.SOURCE_REVISION, "input_kind": "prepared", "dtype": "F32",
            "checkpoint_sha256": "matching-checkpoint", "category": "bottoms",
            "inputs_sha256": case.sha256(self.args.reference_inputs / "conditions.safetensors"),
            "steps": 2, "cfg": 1.5, "skip_cfg_last_n_steps": 1, "seed": 42, "schedule": [0, .5, 1],
            "artifacts": {name: case.sha256(self.args.reference / name) for name in ("expected.png", "step-1.safetensors")},
        }
        self.benchmark = {
            "passed": True, "checkpoint_sha256": "matching-checkpoint",
            "manifest_sha256": case.sha256(self.args.prepared), "inputs_sha256": hashes,
            "settings": {"steps": 2, "cfg": 1.5, "shift": 1.5, "skip_cfg_last_n_steps": 1, "seed": 42},
            "runs": [{"exit_code": 0, "output_sha256": case.sha256(self.args.native_run / "sample-0.png")}],
        }
        self.save_manifests()
        self.addCleanup(patch.stopall)
        patch.object(case, "normalized_condition", normalized).start()
        patch.object(case, "load_reference", return_value=(
            None, SimpleNamespace(get_rf_schedule=lambda steps, mu: [0, mu / 3, 1]), None)).start()

    def set_native(self, pixels):
        Image.fromarray(pixels).save(self.args.native_run / "sample-0.png")

    def save_manifests(self):
        case.write_report(self.args.reference / "manifest.json", self.reference)
        case.write_report(self.args.native_run / "benchmark.json", self.benchmark)

    def test_exact_crop_and_persisted_report(self):
        report = case.compare_case(self.args)
        self.assertTrue(report["passed"])
        self.assertEqual(report["changed_channels"], 0)
        self.assertIsNone(report["psnr_db"])
        self.assertEqual(json.loads((self.args.output / "comparison.json").read_text())["crop"], self.crop)

    def test_one_byte_gate_accepts_one_rejects_two(self):
        for error in (1, 2):
            with self.subTest(error=error):
                changed = self.expected.copy()
                changed[0, 0, 0] += error
                self.set_native(changed)
                self.benchmark["runs"][0]["output_sha256"] = case.sha256(self.args.native_run / "sample-0.png")
                self.save_manifests()
                self.args.output = self.args.output.parent / ("comparison-" + str(error))
                report = case.compare_case(self.args)
                self.assertEqual(report["passed"], error == 1)
                self.assertEqual(report["changed_channels"], 1)

    def test_changed_native_output_rejected(self):
        self.set_native(self.expected + 1)
        with self.assertRaisesRegex(ValueError, "Native output changed"):
            case.compare_case(self.args)

    def test_changed_sampling_rejected(self):
        self.benchmark["settings"]["shift"] = 2
        self.save_manifests()
        with self.assertRaisesRegex(ValueError, "schedule/shift"):
            case.compare_case(self.args)

    def test_condition_to_png_lineage_rejected(self):
        conditions = case.load_file(str(self.args.reference_inputs / "conditions.safetensors"))
        conditions["ca_images"][0, 0, 0, 0] += .1
        save_file(conditions, str(self.args.reference_inputs / "conditions.safetensors"))
        self.reference["inputs_sha256"] = case.sha256(self.args.reference_inputs / "conditions.safetensors")
        self.save_manifests()
        with self.assertRaisesRegex(ValueError, "conditions do not match"):
            case.compare_case(self.args)

    def test_prepared_reference_pixels_must_match(self):
        Image.fromarray(np.zeros((4, 5, 3), dtype=np.uint8)).save(self.args.reference_inputs / "ca_image.png")
        with self.assertRaisesRegex(ValueError, "Prepared native/reference pixels differ"):
            case.compare_case(self.args)

    def test_nonfinite_or_reduced_precision_final_tensor_rejected(self):
        for dtype, value in ((np.float32, np.nan), (np.float32, np.inf), (np.float16, 0)):
            with self.subTest(dtype=dtype, value=value):
                save_file({"image": np.full((1, 3, 4, 5), value, dtype=dtype)},
                          str(self.args.reference / "step-1.safetensors"))
                self.reference["artifacts"]["step-1.safetensors"] = case.sha256(self.args.reference / "step-1.safetensors")
                self.save_manifests()
                with self.assertRaisesRegex(ValueError, "finite F32"):
                    case.compare_case(self.args)

    def test_invalid_crop_rejected(self):
        for crop in (dict(self.crop, y=-1), dict(self.crop, width=20),
                     dict(self.crop, x=True), dict(self.crop, height=0)):
            with self.subTest(crop=crop), self.assertRaises(ValueError):
                case.crop_pixels(np.zeros((4, 5, 3)), crop)


if __name__ == "__main__":
    unittest.main()
