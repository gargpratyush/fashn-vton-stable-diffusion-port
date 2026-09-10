#!/usr/bin/env python3
"""Compare preparation with the unchanged upstream pipeline call, without parser weights."""

import argparse
import ast
import logging
import json
import tempfile
import types
import unittest
from pathlib import Path
from typing import List, Literal, Optional

import cv2
import numpy as np
import torch
from PIL import Image, ImageDraw

import prepare_fashn_vton as prepare


ARGS = None


def upstream_call(source, transforms, pose_module, utils, agnostic=None):
    # Execute the pinned method and masking functions unchanged. Only unused
    # parser predictions and diffusion generation are replaced by test doubles.
    environment = {
        "np": np, "torch": torch, "cv2": cv2, "Image": Image, "logging": logging,
        "List": List, "Literal": Literal, "Optional": Optional,
        "PipelineOutput": types.SimpleNamespace, "draw_pose": pose_module.draw_pose,
        "CATEGORY_TO_BODY_COVERAGE": {"tops": "upper", "bottoms": "lower", "one-pieces": "full"},
        "BODY_COVERAGE_TO_FASHN_LABELS": {"upper": [], "lower": [], "full": []},
        "FASHN_LABELS_TO_IDS": {},
    }
    environment.update({key: getattr(utils, key) for key in (
        "setup_logger", "get_dummy_dw_keypoints", "numpy_to_torch", "normalize_uint8_to_neg1_1")})
    root = source / "src" / "fashn_vton"
    if agnostic is None:
        tree = ast.parse((root / "preprocessing" / "agnostic.py").read_text(encoding="utf-8"))
        functions = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in
                     ("_default", "create_clothing_agnostic_image", "create_garment_image")]
    else:
        from fashn_human_parser import CATEGORY_TO_BODY_COVERAGE
        environment["CATEGORY_TO_BODY_COVERAGE"] = CATEGORY_TO_BODY_COVERAGE
        environment.update({key: getattr(agnostic, key) for key in (
            "BODY_COVERAGE_TO_FASHN_LABELS", "FASHN_LABELS_TO_IDS",
            "create_clothing_agnostic_image", "create_garment_image")})
        functions = []
    pipeline = ast.parse((root / "pipeline.py").read_text(encoding="utf-8"))
    cls = next(n for n in pipeline.body if isinstance(n, ast.ClassDef) and n.name == "TryOnPipeline")
    functions.append(next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "__call__"))
    exec(compile(ast.Module(body=functions, type_ignores=[]), str(root / "pipeline.py"), "exec"), environment)
    return environment["__call__"]


class PreparationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.transforms, cls.pose_module, cls.utils = prepare.reference_modules(ARGS.source)
        cls.original = staticmethod(upstream_call(ARGS.source, cls.transforms, cls.pose_module, cls.utils))

    def compare_upstream(self, person, garment, detector, *, category="tops",
                         garment_photo_type="flat-lay", segmentation_free=True, parser=None, agnostic=None):
        observed = []

        def record_bgr(image):
            observed.append(image.copy())
            return detector(image)

        images, crop, _ = prepare.prepare_images(person, garment, record_bgr,
                                                  self.transforms, self.pose_module, self.utils,
                                                  category=category, garment_photo_type=garment_photo_type,
                                                  segmentation_free=segmentation_free, parser=parser, agnostic=agnostic)
        captures = {}

        def capture(**kwargs):
            captures.update(kwargs)
            return [Image.new("RGB", (576, 864))]

        pipeline = types.SimpleNamespace(
            device=torch.device("cpu"), inference_dtype=torch.float32,
            pre_resize=self.transforms.AspectPreserveResize((864, 864), mode="fit", backend="pil"),
            resize_pad_fn=self.transforms.ResizePad((576, 864), backend="opencv"),
            pose_model=record_bgr, hp_model=parser if parser is not None else types.SimpleNamespace(
                predict=lambda image: np.zeros(image.shape[:2], dtype=np.uint8)),
            logger=logging.getLogger("preprocess-test"), _sample=capture,
            CATEGORY_TO_LABEL={"tops": 1, "bottoms": 2, "one-pieces": 3})
        result = self.original(pipeline, person, garment, category, garment_photo_type=garment_photo_type,
                               num_samples=1, num_timesteps=1, segmentation_free=segmentation_free)
        self.assertEqual(len(observed), 4 if garment_photo_type == "model" else 2)
        for actual_bgr, reference_bgr in zip(observed[:len(observed) // 2], observed[len(observed) // 2:]):
            np.testing.assert_array_equal(actual_bgr, reference_bgr)
        for key, tensor_key in (("ca_image", "ca_images"), ("garment_image", "garment_images"),
                                ("person_pose", "person_poses"), ("garment_pose", "garment_poses")):
            actual = prepare.normalized_condition(images[key], self.utils)
            torch.testing.assert_close(actual, captures[tensor_key], rtol=0, atol=0)
        self.assertEqual(result.images[0].size, (crop["width"], crop["height"]))
        left, top, right, bottom = pipeline.resize_pad_fn.pad_fn.padding_mem
        self.assertEqual(crop, {"x": left, "y": top, "width": 576 - left - right, "height": 864 - top - bottom})
        if garment_photo_type == "flat-lay":
            self.assertEqual(np.count_nonzero(images["garment_pose"]), 0)
        return images, crop

    def fake_detector(self, image):
        pose = self.utils.get_dummy_dw_keypoints()
        pose["bodies"] = {
            "candidate": np.column_stack((np.linspace(.2, .8, 18), np.linspace(.1, .9, 18))),
            "subset": np.arange(18, dtype=np.float32)[None],
        }
        return pose

    def test_upstream_pixel_parity(self):
        for width, height in ((576, 864), (301, 501), (1301, 1777), (1201, 431), (233, 864)):
            with self.subTest(size=(width, height)):
                pixels = np.arange(width * height * 3, dtype=np.uint8).reshape(height, width, 3)
                self.compare_upstream(Image.fromarray(pixels), Image.new("RGB", (811, 307), (17, 121, 203)),
                                      self.fake_detector)

    def test_blank_pose_rejected(self):
        with self.assertRaisesRegex(ValueError, "visible body keypoints"):
            prepare.prepare_images(Image.new("RGB", (64, 64)), Image.new("RGB", (64, 64)),
                                   lambda image: self.utils.get_dummy_dw_keypoints(),
                                   self.transforms, self.pose_module, self.utils)

    def test_non_rgb_rejected(self):
        with self.assertRaisesRegex(ValueError, "RGB"):
            prepare.prepare_images(Image.new("L", (64, 64)), Image.new("RGB", (64, 64)),
                                   self.fake_detector, self.transforms, self.pose_module, self.utils)

    def test_bad_weight_hash_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            for name in prepare.POSE_HASHES:
                (Path(directory) / name).write_bytes(b"invalid")
            with self.assertRaisesRegex(ValueError, "hash"):
                prepare.verify_pose_weights(Path(directory))
            with self.assertRaisesRegex(ValueError, "overwrite"):
                prepare.download_pose(Path(directory))

    def test_parser_requires_explicit_consent(self):
        from fashn_parser_eval import download_parser
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "not-created"
            with self.assertRaisesRegex(ValueError, "research/evaluation"):
                download_parser(output, False)
            with self.assertRaisesRegex(ValueError, "research/evaluation"):
                prepare.export_prepared(types.SimpleNamespace(
                    output=output, garment_photo_type="model", accept_parser_research_license=False))
            self.assertFalse(output.exists())

    def test_real_person_pose_and_export(self):
        if ARGS.dwpose_dir is None or ARGS.person_image is None or ARGS.output is None:
            self.skipTest("Real-person test paths not supplied")
        prepare.verify_pose_weights(ARGS.dwpose_dir)
        detector = self.pose_module.DWposeDetector(str(ARGS.dwpose_dir), device="cpu")
        # A clearly labelled procedural product fixture, not a real garment photo.
        garment = Image.new("RGB", (400, 500), "white")
        ImageDraw.Draw(garment).polygon(
            [(120, 60), (170, 40), (230, 40), (280, 60), (365, 150), (295, 220),
             (265, 190), (265, 435), (135, 435), (135, 190), (105, 220), (35, 150)], fill=(30, 90, 170))
        with Image.open(ARGS.person_image) as image:
            self.compare_upstream(image.convert("RGB"), garment, detector)
        ARGS.output.mkdir(parents=True, exist_ok=False)
        garment_path = ARGS.output / "synthetic-flat-lay.png"
        garment.save(garment_path)
        prepare.export_prepared(types.SimpleNamespace(
            source=ARGS.source, dwpose_dir=ARGS.dwpose_dir, person_image=ARGS.person_image,
            garment_image=garment_path, category="tops", output=ARGS.output / "prepared"))
        from safetensors.torch import load_file
        from export_fashn_vton_reference import validate_conditions
        validate_conditions(load_file(str(ARGS.output / "prepared" / "conditions.safetensors")))
        (ARGS.output / "fixture-provenance.json").write_text(json.dumps({
            "person": "pinned upstream example image",
            "garment": "procedural polygon, not a real garment photograph",
            "exact_upstream_tensor_and_crop_parity": True,
            "human_parser_used": False,
        }, indent=2) + "\n", encoding="utf-8")
        self.assertNotIn("fashn_human_parser", __import__("sys").modules)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    for name in ("dwpose-dir", "person-image", "output"):
        parser.add_argument("--" + name, type=Path)
    ARGS, remaining = parser.parse_known_args()
    torch.set_num_threads(8)
    unittest.main(argv=[__file__, *remaining])
