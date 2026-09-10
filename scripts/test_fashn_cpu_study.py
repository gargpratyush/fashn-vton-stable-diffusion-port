from pathlib import Path
import tempfile
import unittest
from PIL import Image
from run_fashn_cpu_study import pixel_difference


class CPUStudyTests(unittest.TestCase):
    def test_pixel_gate_and_shape(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            Image.new("RGB", (4, 4), (0, 0, 0)).save(root / "a.png")
            Image.new("RGB", (4, 4), (1, 0, 0)).save(root / "b.png")
            self.assertTrue(pixel_difference(root / "a.png", root / "b.png")["passed"])
            Image.new("RGB", (4, 4), (2, 0, 0)).save(root / "b.png")
            self.assertFalse(pixel_difference(root / "a.png", root / "b.png")["passed"])
            Image.new("RGB", (3, 4)).save(root / "b.png")
            with self.assertRaises(ValueError):
                pixel_difference(root / "a.png", root / "b.png")


if __name__ == "__main__":
    unittest.main()
