#!/usr/bin/env python3
"""Focused oracle tests using the standard library test runner."""

import argparse
import copy
import json
import math
import struct
import tempfile
import unittest
from unittest import mock
from pathlib import Path

import torch
from torch.nn.attention import SDPBackend, sdpa_kernel

import export_fashn_vton_reference as oracle


SOURCE = None
CHECKPOINT = None


def synthetic_header(prefix=""):
    header, offset = {}, 0
    for name, shape in oracle.expected_shapes().items():
        size = math.prod(shape) * 2
        header[prefix + name] = {"dtype": "BF16", "shape": shape, "data_offsets": [offset, offset + size]}
        offset += size
    return header, offset


class CheckpointContractTests(unittest.TestCase):
    def test_exact_released_counts(self):
        shapes = oracle.expected_shapes()
        self.assertEqual(len(shapes), 366)
        self.assertEqual(sum(math.prod(v) for v in shapes.values()), 971814240)

    def test_raw_and_prefixed(self):
        raw, payload = synthetic_header()
        prefixed, _ = synthetic_header(oracle.PREFIX)
        self.assertEqual(oracle.validate_header(raw, payload), oracle.validate_header(prefixed, payload))

    def test_unused_buffer_optional(self):
        header, payload = synthetic_header()
        del header["patch_mixer_token"]
        self.assertEqual(len(oracle.validate_header(header, payload - 864)), 365)

    def test_missing_interior_block(self):
        header, _ = synthetic_header()
        del header["double_blocks.3.img_attn.qkv.weight"]
        with self.assertRaisesRegex(ValueError, "Missing"):
            oracle.validate_header(header)

    def test_wrong_shape(self):
        header, _ = synthetic_header()
        header["garment_embedder.proj.weight"]["shape"][1] = 3
        with self.assertRaisesRegex(ValueError, "Shape mismatch"):
            oracle.validate_header(header)

    def test_extra_block(self):
        header, _ = synthetic_header()
        header["single_blocks.16.linear1.weight"] = copy.deepcopy(header["single_blocks.0.linear1.weight"])
        with self.assertRaisesRegex(ValueError, "Unexpected"):
            oracle.validate_header(header)

    def test_prefix_collision(self):
        header, _ = synthetic_header()
        header[oracle.PREFIX + "y_embedder.weight"] = header["y_embedder.weight"]
        with self.assertRaisesRegex(ValueError, "collision"):
            oracle.validate_header(header)

    def test_bad_dtype(self):
        for dtype in ("I64", [], None):
            header, _ = synthetic_header()
            header["y_embedder.weight"]["dtype"] = dtype
            with self.subTest(dtype=dtype), self.assertRaisesRegex(ValueError, "dtype"):
                oracle.validate_header(header)

    def test_bad_offsets(self):
        for offsets in ([0, 2], [-1, 3], [False, 4]):
            header, _ = synthetic_header()
            header["y_embedder.weight"]["data_offsets"] = offsets
            with self.assertRaisesRegex(ValueError, "offsets"):
                oracle.validate_header(header)

    def test_overlap(self):
        header, _ = synthetic_header()
        size = math.prod(header["y_embedder.weight"]["shape"]) * 2
        header["y_embedder.weight"]["data_offsets"] = [0, size]
        with self.assertRaisesRegex(ValueError, "Overlapping"):
            oracle.validate_header(header)

    def test_truncated_payload(self):
        header, payload = synthetic_header()
        with self.assertRaisesRegex(ValueError, "Payload length"):
            oracle.validate_header(header, payload - 1)

    def test_invalid_header_length_and_duplicate_keys(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "invalid.safetensors"
            for data in (b"", b"\x01", struct.pack("<Q", 2**63), struct.pack("<Q", 2) + b"[]"):
                path.write_bytes(data)
                with self.assertRaises(ValueError):
                    oracle.read_header(path)
            data = b'{"a":1,"a":2}'
            path.write_bytes(struct.pack("<Q", len(data)) + data)
            with self.assertRaisesRegex(ValueError, "Duplicate"):
                oracle.read_header(path)

    def test_official_header(self):
        if CHECKPOINT is None:
            self.skipTest("--checkpoint not supplied")
        header, payload = oracle.read_header(CHECKPOINT)
        self.assertEqual(len(oracle.validate_header(header, payload)), 366)
        self.assertEqual(payload, 1943628480)


class TensorContractTests(unittest.TestCase):
    def test_trajectory_uses_prepared_conditions_and_category(self):
        from safetensors.torch import load_file

        values = oracle.synthetic_conditions(oracle.HEIGHT, oracle.WIDTH)
        calls = []

        def model(x, times, **kwargs):
            calls.append(kwargs["garment_categories"].item())
            for name, expected in values.items():
                torch.testing.assert_close(kwargs[name], expected, rtol=0, atol=0)
            return {"x": torch.zeros_like(x)}

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            inputs = root / "conditions.safetensors"
            oracle.save_tensors(inputs, values)
            args = argparse.Namespace(
                source=root, checkpoint=root / "unused", output=root / "trajectory", inputs=inputs,
                threads=1, steps=1, cfg=1., shift=1.5, skip_cfg_last_n_steps=1, seed=42, category="bottoms")
            sampling = argparse.Namespace(get_rf_schedule=lambda steps, mu: [0., 1.])
            with mock.patch.object(oracle, "load_reference", return_value=(None, sampling, None)), \
                    mock.patch.object(oracle, "load_pinned_model", return_value=(model, {}, "test")), \
                    mock.patch.object(torch, "set_num_interop_threads"), \
                    mock.patch.object(torch, "set_num_threads"), \
                    mock.patch.object(torch, "use_deterministic_algorithms"):
                oracle.export_trajectory(args)
            self.assertEqual(calls, [2])
            manifest = json.loads((args.output / "manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["category"], "bottoms")
            self.assertEqual(manifest["input_kind"], "prepared")
            self.assertEqual(manifest["inputs_sha256"], oracle.sha256(inputs))
            initial = load_file(str(args.output / "initial.safetensors"))["noise"]
            output = load_file(str(args.output / "step-0.safetensors"))["image"]
            torch.testing.assert_close(initial, output, rtol=0, atol=0)

    def test_prepared_png_roundtrip(self):
        import numpy as np
        from PIL import Image

        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "source"
            source.mkdir()
            (source / "manifest.json").write_text('{"category":"tops"}', encoding="utf-8")
            values = oracle.synthetic_conditions(oracle.HEIGHT, oracle.WIDTH)
            oracle.save_tensors(source / "inputs.safetensors", values)
            output = Path(directory) / "prepared"
            oracle.export_prepared_fixtures(source, output)
            document = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(document["schema"], "fashn-vton-prepared-v1")
            for tensor_key, image_key in (("ca_images", "ca_image"), ("garment_images", "garment_image"),
                                          ("person_poses", "person_pose"), ("garment_poses", "garment_pose")):
                with Image.open(output / document[image_key]) as image:
                    self.assertEqual(image.mode, "RGB" if values[tensor_key].shape[1] == 3 else "L")
                    pixels = torch.from_numpy(np.array(image))
                decoded = pixels.permute(2, 0, 1) if pixels.dim() == 3 else pixels.unsqueeze(0)
                torch.testing.assert_close(decoded.float() / 127.5 - 1, values[tensor_key][0], atol=1e-7, rtol=0)
            with self.assertRaises(FileExistsError):
                oracle.export_prepared_fixtures(source, output)
            values["ca_images"][0, 0, 0, 0] = 0.123456
            oracle.save_tensors(source / "inputs.safetensors", values)
            with self.assertRaisesRegex(ValueError, "losslessly"):
                oracle.export_prepared_fixtures(source, Path(directory) / "invalid")

    def test_native_fixture_transport(self):
        from safetensors.torch import load_file

        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "source"
            output = Path(directory) / "native"
            source.mkdir()
            (source / "manifest.json").write_text("{}", encoding="utf-8")
            original = {
                "pe": torch.arange(48, dtype=torch.float32).reshape(1, 1, 2, 6, 2, 2),
                "category": torch.tensor([3]),
                "image": torch.zeros(1, 3, 24, 36),
            }
            oracle.save_tensors(source / "inputs.safetensors", original)
            oracle.export_native_fixtures(source, output)
            loaded = load_file(str(output / "inputs.safetensors"))
            self.assertEqual(loaded["pe"].shape, (2, 6, 2, 2))
            self.assertTrue(torch.equal(loaded["pe"].flatten(), original["pe"].flatten()))
            self.assertEqual(loaded["image"].shape, original["image"].shape)
            self.assertEqual(loaded["category"].item(), 3)
            manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["artifacts"]["inputs.safetensors"], oracle.sha256(output / "inputs.safetensors"))

    def test_conditions(self):
        values = oracle.synthetic_conditions(24, 36)
        oracle.validate_conditions(values, 24, 36)
        self.assertTrue(torch.all(values["garment_poses"] == -1))
        self.assertFalse(torch.equal(values["garment_poses"], torch.zeros_like(values["garment_poses"])))

    def test_invalid_conditions(self):
        for mode in ("missing", "shape", "dtype", "nan", "range"):
            values = oracle.synthetic_conditions(24, 36)
            if mode == "missing":
                del values["person_poses"]
            elif mode == "shape":
                values["person_poses"] = values["person_poses"].squeeze(0)
            elif mode == "dtype":
                values["person_poses"] = values["person_poses"].half()
            else:
                values["person_poses"][0, 0, 0, 0] = float("nan") if mode == "nan" else 2
            with self.subTest(mode=mode), self.assertRaises(ValueError):
                oracle.validate_conditions(values, 24, 36)

    def test_single_logical_noise_draw(self):
        generator = torch.Generator().manual_seed(42)
        expected = torch.randn((4, 3, 24, 36), generator=generator)
        self.assertTrue(torch.equal(oracle.make_noise(4, 24, 36, 42), expected))
        self.assertTrue(torch.equal(oracle.make_noise(4, 24, 36, 42), expected))

    def test_euler_and_cfg(self):
        x, vc, vu = torch.ones(3), torch.full((3,), 3.0), torch.full((3,), -1.0)
        for cfg in (0, 1, 1.5):
            for skip in (False, True):
                guided, updated = oracle.euler_update(x, vc, vu, cfg, 0.25, skip)
                expected = vc if skip else vu + cfg * (vc - vu)
                torch.testing.assert_close(guided, expected, rtol=0, atol=0)
                torch.testing.assert_close(updated, x + 0.25 * expected, rtol=0, atol=0)

    def test_invalid_sampling(self):
        for args in ((0, 0, 1.5, 1.5, 0), (1, 1, 1.5, 1.5, 0), (1, 0, float("nan"), 1.5, 0),
                     (1, 0, 1.5, float("inf"), 0), (1, 0, 1.5, 1.5, 2), (1, 0, -1, 1.5, 0)):
            with self.subTest(args=args), self.assertRaises(ValueError):
                oracle.validate_sampling(*args)

    def test_comparison_detects_real_errors(self):
        expected = {"x": torch.ones(10)}
        self.assertTrue(oracle.compare_tensors(expected, expected)["x"]["passed"])
        self.assertFalse(oracle.compare_tensors(expected, {"x": torch.zeros(10)})["x"]["passed"])
        with self.assertRaisesRegex(ValueError, "Non-finite"):
            oracle.compare_tensors(expected, {"x": torch.full((10,), float("nan"))})
        with self.assertRaisesRegex(ValueError, "key mismatch"):
            oracle.compare_tensors(expected, {})
        with self.assertRaisesRegex(ValueError, "Shape/dtype"):
            oracle.compare_tensors(expected, {"x": torch.ones(10, dtype=torch.float64)})
        with self.assertRaisesRegex(ValueError, "tolerances"):
            oracle.compare_tensors(expected, expected, atol=float("nan"))

    def test_integer_comparison_is_exact(self):
        expected = {"category": torch.tensor([1])}
        self.assertFalse(oracle.compare_tensors(expected, {"category": torch.tensor([2])},
                                               atol=100, rtol=100)["category"]["passed"])

    def test_save_rejects_nonfinite(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, "Non-finite"):
                oracle.save_tensors(Path(directory) / "bad.safetensors", {"x": torch.tensor([float("inf")])})


class OriginalSourceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if SOURCE is None:
            raise unittest.SkipTest("--source not supplied")
        cls.module, cls.sampling, cls.tensor = oracle.load_reference(SOURCE)
        torch.set_num_threads(2)

    def test_metadata_matches_original_model(self):
        with torch.device("meta"):
            model = self.module.TryOnModel()
        self.assertEqual({name: list(value.shape) for name, value in model.state_dict().items()},
                         oracle.expected_shapes())

    def test_schedule_against_independent_formula(self):
        for steps in (1, 4, 30):
            for shift in (-1.5, 0, 1.5):
                actual = self.sampling.get_rf_schedule(steps, mu=shift)
                a = math.exp(-shift)
                expected = [a * (i / steps) / (1 - i / steps + a * (i / steps)) for i in range(steps + 1)]
                # The original F32 reciprocal-based expression differs from the
                # analytic F64 formula by 2.98e-7 at step 29/30, mu=1.5.
                torch.testing.assert_close(torch.tensor(actual), torch.tensor(expected), atol=4e-7, rtol=0)
                self.assertEqual(actual[0], 0)
                self.assertEqual(actual[-1], 1)

    def test_layout_and_normalization(self):
        from safetensors.torch import load_file

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "layout.safetensors"
            oracle.export_layout(self.module, self.tensor, path)
            values = load_file(str(path))
            torch.testing.assert_close(values["image"], values["reconstructed"], atol=0, rtol=0)
            self.assertEqual(values["patches"].shape, (1, 6, 432))
            self.assertEqual(values["patches"][0, 0, 12].item(), 36)
            self.assertEqual(values["patches"][0, 0, 144].item(), 24 * 36)
            self.assertEqual(values["normalized"][0].item(), -1)
            self.assertEqual(values["normalized"][255].item(), 1)
            self.assertAlmostEqual(values["normalized"][127].item(), -1 / 255, places=7)

    def test_tiny_model_null_branch_capture_and_replay(self):
        torch.manual_seed(123)
        model = self.module.TryOnModel(input_shape=(24, 36), hidden_size=16, n_heads=2,
                                     double_blocks_depth=2, single_blocks_depth=2,
                                     axes_dim=(2, 2, 4), patch_mixer_depth=2).eval()
        x = oracle.make_noise(1, 24, 36, 42)
        times = torch.tensor([0.25])
        conditions = oracle.synthetic_conditions(24, 36)
        category = torch.tensor([2])
        with torch.inference_mode(), sdpa_kernel(SDPBackend.MATH):
            conditional = oracle.capture_forward(model, x, times, conditions, category)
            null = oracle.capture_forward(model, x, times, conditions, category, unconditional=True)
            replay = oracle.capture_forward(model, x, times, conditions, category)
            batched = model.forward_for_cfg(x, times, **conditions, garment_categories=category)
            zeros = {name: torch.zeros_like(value) for name, value in conditions.items()}
            explicit_null = model(x, times, **zeros, garment_categories=torch.zeros_like(category))["x"]
        for name in conditional:
            torch.testing.assert_close(conditional[name], replay[name], atol=0, rtol=0)
        torch.testing.assert_close(null["velocity"], explicit_null, atol=0, rtol=0)
        torch.testing.assert_close(conditional["velocity"], batched["v_c"], atol=1e-5, rtol=1e-5)
        torch.testing.assert_close(null["velocity"], batched["v_u"], atol=1e-5, rtol=1e-5)
        self.assertFalse(torch.equal(conditional["velocity"], null["velocity"]))
        self.assertIn("double_blocks.0.call0.output.1", conditional)
        self.assertIn("single_blocks.1.call0.output", conditional)
        self.assertIn("pe_embedder.call1.output", conditional)
        self.assertTrue(all(not module._forward_hooks for module in model.modules()))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path)
    parser.add_argument("--checkpoint", type=Path)
    args, remaining = parser.parse_known_args()
    SOURCE, CHECKPOINT = args.source, args.checkpoint
    unittest.main(argv=[__file__] + remaining)
