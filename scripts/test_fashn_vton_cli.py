#!/usr/bin/env python3
"""Prepared-input CLI regressions; --inference adds one expensive CPU forward."""

import argparse
import json
import subprocess
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image


ARGS = None
METRICS = {}


class PreparedCliTests(unittest.TestCase):
    def command(self, manifest, *extra):
        return [str(ARGS.cli), "--mode", "try_on", "--diffusion-model", str(ARGS.checkpoint),
                "--try-on-inputs", str(manifest), "--steps", "1", "--cfg-scale", "1",
                "--seed", "42", "--rng", "cpu", "-t", "8", *extra]

    def manifest(self):
        document = json.loads(ARGS.prepared.read_text(encoding="utf-8"))
        for key in ("ca_image", "garment_image", "person_pose", "garment_pose"):
            document[key] = str((ARGS.prepared.parent / document[key]).resolve())
        return document

    def failure(self, document, expected, *extra):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "manifest.json"
            path.write_text(json.dumps(document), encoding="utf-8")
            result = subprocess.run(self.command(path, *extra), capture_output=True, text=True, timeout=30)
            self.assertEqual(result.returncode, 1, result.stdout + result.stderr)
            self.assertIn(expected, result.stdout + result.stderr)

    def test_invalid_manifests(self):
        cases = [
            ("schema", "unknown", "schema"),
            ("category", "dresses", "category"),
            ("unknown", 1, "Unknown"),
            ("ca_image", "", "Missing"),
            ("person_pose", self.manifest()["ca_image"], "exactly 1 channels"),
            ("crop", {"x": 575, "y": 0, "width": 2, "height": 864}, "outside"),
            ("crop", {"x": 0, "y": 0, "width": 0, "height": 864}, "outside"),
            ("crop", {"x": 0.5, "y": 0, "width": 100, "height": 100}, "integer"),
        ]
        for key, value, expected in cases:
            with self.subTest(key=key, value=value):
                document = self.manifest()
                document[key] = value
                self.failure(document, expected)

    def test_wrong_dimensions_and_bit_depth(self):
        with tempfile.TemporaryDirectory() as directory:
            for name, pixels in (("small", np.zeros((64, 64), dtype=np.uint8)),
                                 ("sixteen", np.zeros((864, 576), dtype=np.uint16))):
                with self.subTest(name=name):
                    path = Path(directory) / (name + ".png")
                    Image.fromarray(pixels).save(path)
                    document = self.manifest()
                    document["person_pose"] = str(path)
                    self.failure(document, "must be 576x864")

    def test_unsupported_or_invalid_options(self):
        for options, expected in [
            (("--sampling-method", "heun"), "own ascending-time"),
            (("--scheduler", "karras"), "own ascending-time"),
            (("--prompt", "not a text model"), "prompts"),
            (("--type", "q8_0"), "requires F32"),
            (("--steps", "0"), "Invalid FASHN"),
            (("--batch-count", "0"), "Invalid FASHN"),
            (("--skip-cfg-last-n-steps", "2"), "Invalid FASHN"),
        ]:
            with self.subTest(options=options):
                self.failure(self.manifest(), expected, *options)

    def compare_pixels(self, name, expected, actual):
        self.assertEqual(actual.shape, expected.shape)
        error = np.abs(actual.astype(np.int16) - expected.astype(np.int16))
        metrics = {"shape": list(actual.shape), "max_byte_error": int(error.max()),
                   "mean_byte_error": float(error.mean()), "different_channels": int(np.count_nonzero(error))}
        METRICS[name] = metrics
        self.assertLessEqual(metrics["max_byte_error"], 1, metrics)
        self.assertLessEqual(metrics["mean_byte_error"], 0.01, metrics)

    def test_recorded_two_step_trajectory(self):
        if ARGS.trajectory is None or ARGS.native_trajectory is None:
            self.skipTest("Recorded native/reference trajectory paths not supplied")
        with Image.open(ARGS.trajectory / "expected.png") as expected, Image.open(ARGS.native_trajectory) as actual:
            self.assertEqual(actual.mode, "RGB")
            self.compare_pixels("two_step", np.array(expected), np.array(actual))

    def test_one_step_cropped_inference(self):
        if not ARGS.inference:
            self.skipTest("--inference not supplied")
        import torch
        from safetensors.torch import load_file

        document = self.manifest()
        document["crop"] = {"x": 17, "y": 29, "width": 101, "height": 153}
        path = ARGS.output / "cropped-manifest.json"
        path.write_text(json.dumps(document, indent=2) + "\n", encoding="utf-8")
        destination = ARGS.output / "native-one-step-cropped.png"
        result = subprocess.run(self.command(path, "-o", str(destination)),
                                capture_output=True, text=True, timeout=900)
        (ARGS.output / "inference.log").write_text(result.stdout + result.stderr, encoding="utf-8")
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        inputs = load_file(str(ARGS.oracle / "inputs.safetensors"))
        self.assertEqual(inputs["times"].item(), 0)
        velocity = load_file(str(ARGS.oracle / "conditional.safetensors"))["velocity"]
        expected = ((inputs["noise"] + velocity + 1) * 0.5).clamp(0, 1).mul(255).to(torch.uint8)
        expected = expected[0].permute(1, 2, 0).numpy()[29:182, 17:118]
        Image.fromarray(expected).save(ARGS.output / "expected-one-step-cropped.png")
        with Image.open(destination) as actual:
            self.assertEqual(actual.mode, "RGB")
            self.compare_pixels("one_step_crop", expected, np.array(actual))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("cli", "checkpoint", "prepared", "oracle", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--trajectory", type=Path)
    parser.add_argument("--native-trajectory", type=Path)
    parser.add_argument("--inference", action="store_true")
    ARGS, remaining = parser.parse_known_args()
    ARGS.output.mkdir(parents=True, exist_ok=False)
    result = unittest.main(argv=[__file__, *remaining], exit=False).result
    report = {"passed": result.wasSuccessful(), "pixel_comparisons": METRICS,
              "tests_run": result.testsRun, "skipped": len(result.skipped)}
    (ARGS.output / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    raise SystemExit(0 if result.wasSuccessful() else 1)
