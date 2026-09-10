"""Native-only preparation versus pinned Python transforms, masks, DWPose and parser."""
import argparse
import copy
import json
from pathlib import Path
import subprocess
import unittest

import cv2
import numpy as np
from PIL import Image
from prepare_fashn_vton import prepare_images, reference_modules
from fashn_parser_eval import load_parser, require_evaluation_consent
from export_fashn_vton_reference import sha256

ARGS = None
REPORT = {"comparisons": {}}


class Cached:
    def __init__(self, function):
        self.function = function
        self.values = {}

    def __call__(self, image):
        key = image.tobytes()
        if key not in self.values:
            self.values[key] = self.function(image)
        return copy.deepcopy(self.values[key])

    def predict(self, image):
        return self(image)


class NativePreparationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        require_evaluation_consent(ARGS.accept_parser_research_license)
        import torch
        torch.set_num_threads(8)
        cls.transforms, cls.pose_module, cls.utils = reference_modules(ARGS.source)
        parser, cls.agnostic = load_parser(ARGS.source, ARGS.parser_dir, True)
        cls.parser = Cached(parser.predict)
        cls.detector = Cached(cls.pose_module.DWposeDetector(str(ARGS.dwpose_dir), device="cpu"))

    def compare(self, name, expected, actual_path):
        with Image.open(actual_path) as image:
            actual = np.array(image)
        np.testing.assert_array_equal(actual, expected, err_msg=name)
        REPORT["comparisons"][name] = {"exact": True, "shape": list(actual.shape)}

    def test_transform_mask_primitives(self):
        for index, (width, height) in enumerate(((47, 63), (865, 577), (931, 1249))):
            root = ARGS.output / f"primitives-{index}"
            root.mkdir()
            y, x = np.indices((height, width))
            image = np.stack(((x * 7 + y * 13) % 256, (x * 17) % 256, (y * 19) % 256), axis=-1).astype(np.uint8)
            labels = ((x // max(1, width // 6) + 6 * (y // max(1, height // 3))) % 18).astype(np.uint8)
            Image.fromarray(image).save(root / "image.png")
            Image.fromarray(labels).save(root / "labels.png")
            result = subprocess.run([str(ARGS.primitives), str(root), str(root / "native")],
                                    capture_output=True, text=True, timeout=120)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            self.assertIn(sha256(root / "image.png"), result.stdout)
            resized = np.array(self.transforms.AspectPreserveResize((864,864), backend="pil")(
                Image.fromarray(image), allow_upsampling=False))
            self.compare(f"resize-{index}", resized, root / "native" / "preresize.png")
            self.compare(f"pad-{index}", self.transforms.ResizePad((576,864))(resized), root / "native" / "padded.png")
            self.compare(f"pose-pad-{index}", self.transforms.ResizePad((576,864))(labels, interpolation=cv2.INTER_NEAREST_EXACT),
                         root / "native" / "pose-padded.png")
            for category, coverage in (("tops","upper"), ("bottoms","lower"), ("one-pieces","full")):
                selected = [self.agnostic.FASHN_LABELS_TO_IDS[name] for name in self.agnostic.BODY_COVERAGE_TO_FASHN_LABELS[coverage]]
                person = self.agnostic.create_clothing_agnostic_image(image.copy(), labels.copy(), selected.copy(), coverage)
                garment = self.agnostic.create_garment_image(image.copy(), labels.copy(), selected.copy())
                self.compare(f"{index}-{category}-person", person, root / "native" / f"{category}-person.png")
                self.compare(f"{index}-{category}-garment", garment, root / "native" / f"{category}-garment.png")

    def command(self, output, photo="model", masked=False):
        command = [str(ARGS.native), "--dwpose-dir", str(ARGS.dwpose_dir),
                   "--person-image", str(ARGS.source / "examples" / "data" / "model.webp"),
                   "--garment-image", str(ARGS.source / "examples" / "data" / "garment.webp"),
                   "--category", "tops", "--garment-photo-type", photo, "--output", str(output)]
        if photo == "model" or masked:
            command += ["--parser-dir", str(ARGS.parser_native_dir), "--accept-parser-research-license"]
        if masked:
            command.append("--no-segmentation-free")
        if ARGS.ort_no_arena:
            command.append("--ort-no-arena")
        return command

    def test_all_twelve_preparation_modes(self):
        with Image.open(ARGS.source / "examples" / "data" / "model.webp") as p, \
                Image.open(ARGS.source / "examples" / "data" / "garment.webp") as g:
            person, garment = p.convert("RGB"), g.convert("RGB")
        # The flat-lay cases exercise declared-mode semantics; the source garment
        # photograph is worn, not relabelled as a flat-lay quality fixture.
        for category in ("tops", "bottoms", "one-pieces"):
            for photo in ("flat-lay", "model"):
                for masked in (False, True):
                    name = f"{category}-{photo}-{'masked' if masked else 'free'}"
                    destination = ARGS.output / name
                    command = self.command(destination, photo, masked)
                    command[command.index("--category") + 1] = category
                    result = subprocess.run(command, capture_output=True, text=True, timeout=180)
                    (ARGS.output / (name + ".log")).write_text(result.stdout + result.stderr, encoding="utf-8")
                    self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                    expected, crop, _ = prepare_images(person, garment, self.detector, self.transforms, self.pose_module,
                        self.utils, category=category, garment_photo_type=photo, segmentation_free=not masked,
                        parser=self.parser, agnostic=self.agnostic)
                    manifest = json.loads((destination / "manifest.json").read_text())
                    self.assertEqual(manifest["crop"], crop)
                    for key, image in expected.items():
                        self.compare(name + "/" + key, image, destination / manifest[key])
                    provenance = json.loads((destination / "provenance.json").read_text())
                    self.assertEqual(provenance["parser"] is None, photo == "flat-lay" and not masked)
                    for filename, expected_hash in provenance["artifacts"].items():
                        self.assertEqual(sha256(destination / filename), expected_hash)

    def test_failure_contract(self):
        for index, change in enumerate(("consent", "category", "missing-weight", "existing-output")):
            output = ARGS.output / f"invalid-{index}"
            command = self.command(output)
            if change == "consent":
                command.remove("--accept-parser-research-license")
            elif change == "category":
                command[command.index("--category") + 1] = "unknown"
            elif change == "missing-weight":
                command[command.index("--dwpose-dir") + 1] = str(output / "absent")
            else:
                output.mkdir()
            result = subprocess.run(command, capture_output=True, text=True, timeout=30)
            self.assertEqual(result.returncode, 1, result.stdout + result.stderr)
            self.assertIn("failed:", result.stderr)
            self.assertFalse((output / "manifest.json").exists())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("native", "primitives", "source", "dwpose-dir", "parser-dir", "parser-native-dir", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--accept-parser-research-license", action="store_true")
    parser.add_argument("--ort-no-arena", action="store_true")
    ARGS, remaining = parser.parse_known_args()
    ARGS.output.mkdir(parents=True, exist_ok=False)
    result = unittest.main(argv=[__file__, *remaining], exit=False).result
    REPORT.update(passed=result.wasSuccessful(), tests_run=result.testsRun)
    (ARGS.output / "report.json").write_text(json.dumps(REPORT, indent=2) + "\n", encoding="utf-8")
    raise SystemExit(0 if result.wasSuccessful() else 1)
