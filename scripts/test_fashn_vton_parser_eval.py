#!/usr/bin/env python3
"""Authorized, local-only parser and masking parity with the pinned upstream pipeline."""

import argparse
import json
import os
import socket
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import torch
from PIL import Image

import fashn_parser_eval as evaluation
import prepare_fashn_vton as prepare
import test_fashn_vton_preprocess as prep_tests


ARGS = None
METRICS = {}


class ParserEvaluationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.transforms, cls.pose_module, cls.utils = prepare.reference_modules(ARGS.source)
        prepare.verify_pose_weights(ARGS.dwpose_dir)
        cls.parser, cls.agnostic = evaluation.load_parser(ARGS.source, ARGS.parser_dir, True)
        cls.detector = cls.pose_module.DWposeDetector(str(ARGS.dwpose_dir), device="cpu")
        cls.original = staticmethod(prep_tests.upstream_call(ARGS.source, cls.transforms, cls.pose_module, cls.utils, cls.agnostic))

    def test_local_parser_matches_original_constructor(self):
        from fashn_human_parser import FashnHumanParser
        original = FashnHumanParser(model_id=str(ARGS.parser_dir.resolve()), device="cpu")
        for name in ("model.webp", "garment.webp"):
            with self.subTest(image=name), Image.open(ARGS.source / "examples" / "data" / name) as image:
                image = self.transforms.AspectPreserveResize((864, 864), backend="pil")(
                    image.convert("RGB"), allow_upsampling=False)
                image = np.array(image)
                actual = self.parser.predict(image, return_logits=True)
                expected = original.predict(image, return_logits=True)
                torch.testing.assert_close(actual, expected, atol=0, rtol=0)
                METRICS[name] = {"parser_logits_max_abs": 0,
                                 "classes": [int(v) for v in np.unique(actual.argmax(1).numpy())]}

    def test_all_categories_and_masking_modes(self):
        fake_detector = lambda image: prep_tests.PreparationTests.fake_detector(self, image)

        def labels(image):
            h, w = image.shape[:2]
            return (np.arange(h)[:, None] * 3 + np.arange(w)[None, :] // 7) % 18

        fake_parser = types.SimpleNamespace(predict=labels)
        for category in ("tops", "bottoms", "one-pieces"):
            for photo_type in ("flat-lay", "model"):
                for segmentation_free in (False, True):
                    with self.subTest(category=category, photo=photo_type, segmentation_free=segmentation_free):
                        prep_tests.PreparationTests.compare_upstream(
                            self, Image.new("RGB", (301, 501), (71, 83, 199)),
                            Image.new("RGB", (437, 253), (19, 201, 11)), fake_detector,
                            category=category, garment_photo_type=photo_type, segmentation_free=segmentation_free,
                            parser=fake_parser, agnostic=self.agnostic)
        METRICS["synthetic_mode_combinations_exact"] = 12

    def test_invalid_parser_outputs(self):
        for mode in ("shape", "dtype", "range", "empty"):
            def bad(image):
                shape = image.shape[:2] if mode != "shape" else (1, 1)
                return np.full(shape, 18 if mode == "range" else 0,
                               dtype=np.float32 if mode == "dtype" else np.int64)
            with self.subTest(mode=mode), self.assertRaises(ValueError):
                prepare.prepare_images(
                    Image.new("RGB", (64, 64)), Image.new("RGB", (64, 64)),
                    lambda image: prep_tests.PreparationTests.fake_detector(self, image),
                    self.transforms, self.pose_module, self.utils, garment_photo_type="model",
                    parser=types.SimpleNamespace(predict=bad), agnostic=self.agnostic)
        calls = 0

        def missing_garment_pose(image):
            nonlocal calls
            calls += 1
            return (prep_tests.PreparationTests.fake_detector(self, image) if calls == 1
                    else self.utils.get_dummy_dw_keypoints())

        with self.assertRaisesRegex(ValueError, "worn-garment"):
            prepare.prepare_images(
                Image.new("RGB", (64, 64)), Image.new("RGB", (64, 64)), missing_garment_pose,
                self.transforms, self.pose_module, self.utils, garment_photo_type="model",
                parser=self.parser, agnostic=self.agnostic)

    def test_hash_and_consent_guards(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            (path / "config.json").write_bytes(b"wrong")
            with self.assertRaisesRegex(ValueError, "hash"):
                evaluation.verify_parser_weights(path)
            with self.assertRaisesRegex(ValueError, "research/evaluation"):
                evaluation.load_parser(ARGS.source, path, False)

    def test_real_photographic_preparation(self):
        person_path = ARGS.source / "examples" / "data" / "model.webp"
        garment_path = ARGS.source / "examples" / "data" / "garment.webp"
        with Image.open(person_path) as person, Image.open(garment_path) as garment:
            for segmentation_free in (True, False):
                with self.subTest(segmentation_free=segmentation_free):
                    images, _ = prep_tests.PreparationTests.compare_upstream(
                        self, person.convert("RGB"), garment.convert("RGB"), self.detector,
                        garment_photo_type="model", segmentation_free=segmentation_free,
                        parser=self.parser, agnostic=self.agnostic)
                    self.assertGreater(np.count_nonzero(images["garment_pose"]), 0)
                    self.assertGreater(np.count_nonzero(np.all(images["garment_image"] == 127, axis=-1)), 0)
        prepare.export_prepared(types.SimpleNamespace(
            source=ARGS.source, dwpose_dir=ARGS.dwpose_dir, parser_dir=ARGS.parser_dir,
            accept_parser_research_license=True, person_image=person_path, garment_image=garment_path,
            garment_photo_type="model", segmentation_free=True, category="tops", output=ARGS.output / "prepared"))
        METRICS["real_preparation_exact"] = ["segmentation-free", "masked-person"]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("source", "dwpose-dir", "parser-dir", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--accept-parser-research-license", action="store_true")
    ARGS, remaining = parser.parse_known_args()
    evaluation.require_evaluation_consent(ARGS.accept_parser_research_license)
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    torch.set_num_threads(8)
    ARGS.output.mkdir(parents=True, exist_ok=False)
    with mock.patch.object(socket.socket, "connect", side_effect=AssertionError("Unexpected network access")):
        result = unittest.main(argv=[__file__, *remaining], exit=False).result
    report = {"passed": result.wasSuccessful(), "research_evaluation_only": True,
              "tests_run": result.testsRun, "metrics": METRICS}
    (ARGS.output / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    raise SystemExit(0 if result.wasSuccessful() else 1)
