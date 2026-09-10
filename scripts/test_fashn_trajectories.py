import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
from PIL import Image
from safetensors.numpy import save_file

from compare_fashn_trajectories import compare, metrics, pixels
from export_fashn_vton_reference import sha256


class TrajectoryTests(unittest.TestCase):
    def test_metrics_and_nonfinite(self):
        image = np.ones((1, 3, 4, 4), dtype=np.float32)
        self.assertTrue(metrics(image, image[0])["passed"])
        self.assertFalse(metrics(image, image * 1.01)["passed"])
        with self.assertRaises(ValueError):
            metrics(image, image * float("nan"))

    def test_complete_and_corrupted_export(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            reference, native = root / "reference", root / "native"
            reference.mkdir()
            native.mkdir()
            image = np.ones((1, 3, 4, 4), dtype=np.float32) * .125
            for directory in (reference, native):
                save_file({"noise": image}, str(directory / "initial.safetensors"))
                save_file({"image": image, "guided_velocity": image}, str(directory / "step-0.safetensors"))
            Image.fromarray(pixels(image)).save(reference / "expected.png")
            manifest = {"category": "tops", "steps": 1, "cfg": 1., "skip_cfg_last_n_steps": 1, "seed": 42, "schedule": [0., 1.]}
            manifest["artifacts"] = {p.name: sha256(p) for p in reference.iterdir()}
            (reference / "manifest.json").write_text(json.dumps(manifest))
            manifest["category"] = 1
            (native / "manifest.json").write_text(json.dumps(manifest))
            self.assertTrue(compare(reference, native, root / "pass")["passed"])
            (reference / "expected.png").write_bytes(b"corrupted")
            with self.assertRaisesRegex(ValueError, "hash mismatch"):
                compare(reference, native, root / "fail")


if __name__ == "__main__":
    unittest.main()
