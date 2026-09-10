import unittest
from pathlib import Path
import tempfile

import numpy as np
from PIL import Image
from safetensors.numpy import save_file

from build_fashn_gallery import build, display_pixels, pixel_metrics, write_report


class GalleryTests(unittest.TestCase):
    def test_exact_and_one_byte(self):
        image = np.zeros((2, 3, 3), dtype=np.uint8)
        self.assertIsNone(pixel_metrics(image, image)["psnr_db"])
        changed = image.copy()
        changed[0, 0, 0] = 1
        result = pixel_metrics(image, changed)
        self.assertEqual(result["changed_channels"], 1)
        self.assertEqual(result["changed_pixels"], 1)
        self.assertAlmostEqual(result["mae"], 1 / 18)
        self.assertEqual(result["max_abs"], 1)

    def test_no_resize_or_channel_conversion(self):
        with self.assertRaises(ValueError):
            pixel_metrics(np.zeros((2, 3, 3)), np.zeros((3, 2, 3)))
        with self.assertRaises(ValueError):
            pixel_metrics(np.zeros((2, 3)), np.zeros((2, 3)))

    def test_float_state_rendering_and_finiteness(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "step-19.safetensors"
            save_file({"image": np.zeros((1, 3, 4, 6), dtype=np.float32)}, str(path))
            self.assertTrue(np.array_equal(display_pixels(path, True), np.full((4, 6, 3), 127, dtype=np.uint8)))
            with self.assertRaises(ValueError):
                display_pixels(path, False)
            for dtype, value in ((np.float16, 0), (np.float32, np.nan)):
                save_file({"image": np.full((1, 3, 4, 6), value, dtype=dtype)}, str(path))
                with self.assertRaisesRegex(ValueError, "finite F32"):
                    display_pixels(path, True)
            for shape in ((2, 3, 4, 6), (4, 6, 3)):
                save_file({"image": np.zeros(shape, dtype=np.float32)}, str(path))
                with self.assertRaisesRegex(ValueError, "CHW RGB"):
                    display_pixels(path, True)

    def test_trajectory_gallery_crop_and_provenance(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            tensor = np.zeros((1, 3, 4, 6), dtype=np.float32)
            tensor[:, :, 0, :] = -2
            tensor[:, :, -1, :] = 2
            save_file({"image": tensor}, str(root / "step-19.safetensors"))
            rendered = display_pixels(root / "step-19.safetensors", True)
            self.assertTrue(np.all(rendered[0] == 0))
            self.assertTrue(np.all(rendered[-1] == 255))
            Image.fromarray(np.full((2, 3, 3), 127, dtype=np.uint8)).save(root / "reference.png")
            entry = {"id": "tensor", "title": "Final state", "scope": "Rendering only",
                     "native": str(root / "step-19.safetensors"), "reference": str(root / "reference.png"),
                     "person": str(root / "reference.png"), "garment": str(root / "reference.png"),
                     "native_label": "native", "reference_label": "reference",
                     "native_crop": {"x": 1, "y": 1, "width": 3, "height": 2}}
            write_report(root / "spec.json", [entry])
            result = build(root / "spec.json", root / "gallery")["cases"]["tensor"]
            self.assertEqual(result["metrics"]["changed_channels"], 0)
            self.assertEqual(result["sources"]["native"]["path"], str(root / "step-19.safetensors"))
            self.assertEqual(result["sources"]["native"]["crop"], entry["native_crop"])
            self.assertEqual(len(result["sources"]["native"]["sha256"]), 64)

    def test_gallery_explicit_crop_and_escaped_labels(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            full = np.arange(72, dtype=np.uint8).reshape(4, 6, 3)
            Image.fromarray(full).save(root / "full.png")
            Image.fromarray(full[1:3, 1:4]).save(root / "cropped.png")
            entry = {"id": "case", "title": "<script>not markup</script>", "scope": "test",
                     "native": str(root / "full.png"), "reference": str(root / "cropped.png"),
                     "person": str(root / "full.png"), "garment": str(root / "full.png"),
                     "native_label": "native", "reference_label": "reference",
                     "native_crop": {"x": 1, "y": 1, "width": 3, "height": 2}}
            write_report(root / "spec.json", [entry])
            report = build(root / "spec.json", root / "gallery")
            self.assertEqual(report["cases"]["case"]["metrics"]["changed_channels"], 0)
            self.assertNotIn("<script>", (root / "gallery" / "index.html").read_text())


if __name__ == "__main__":
    unittest.main()
