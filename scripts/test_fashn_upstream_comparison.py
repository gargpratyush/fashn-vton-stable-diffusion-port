from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image

from compare_fashn_upstream import SOURCE_REVISION, compare, sha256, write_report


class UpstreamComparisonTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.upstream, self.native = self.root / "upstream", self.root / "native"
        self.upstream.mkdir()
        self.native.mkdir()
        for directory in (self.upstream, self.native):
            Image.fromarray(np.zeros((2, 3, 3), dtype=np.uint8)).save(directory / "sample-0.png")
        self.settings = {"steps": 20, "cfg": 1.5, "shift": 1.5, "skip_cfg_last_n_steps": 1, "seed": 42, "threads": 16}
        self.reference = {
            "source_revision": SOURCE_REVISION, "dtype": "F32", "checkpoint_sha256": "same",
            "prepared_manifest_sha256": "manifest", "prepared_images_sha256": {"image": "input"},
            "settings": self.settings, "output_sha256": sha256(self.upstream / "sample-0.png"),
            "crop": {"width": 3, "height": 2}, "category": "tops", "sampling_and_pil_seconds": 10,
        }
        self.baseline = {
            "passed": True, "checkpoint_sha256": "same", "manifest_sha256": "manifest",
            "inputs_sha256": {"image": "input"}, "settings": dict(self.settings),
            "runs": [{"exit_code": 0, "output_sha256": sha256(self.native / "sample-0.png"), "wall_seconds": 20}],
        }
        self.save()

    def save(self):
        write_report(self.upstream / "result.json", self.reference)
        write_report(self.native / "benchmark.json", self.baseline)

    def test_matching_output(self):
        with patch("builtins.print"):
            result = compare(self.upstream, self.native, self.root / "result")
        self.assertTrue(result["within_one_byte"])
        self.assertEqual(result["pixels"]["changed_channels"], 0)
        self.assertEqual(result["historical_native_cli_wall_seconds"], 20)

    def test_changed_threads_rejected(self):
        self.baseline["settings"]["threads"] = 8
        self.save()
        with self.assertRaisesRegex(ValueError, "thread mismatch"):
            compare(self.upstream, self.native, self.root / "result")

    def test_changed_input_rejected(self):
        self.baseline["inputs_sha256"]["image"] = "changed"
        self.save()
        with self.assertRaisesRegex(ValueError, "lineage mismatch"):
            compare(self.upstream, self.native, self.root / "result")

    def test_changed_output_rejected(self):
        Image.fromarray(np.ones((2, 3, 3), dtype=np.uint8)).save(self.native / "sample-0.png")
        with self.assertRaisesRegex(ValueError, "Native output changed"):
            compare(self.upstream, self.native, self.root / "result")

    def test_unrelated_measurement_rejected(self):
        measurement = self.root / "measurement.json"
        write_report(measurement, {"execution_passed": True,
                                  "run": {"exit_code": 0, "command": ["python", "--output", str(self.native)]}})
        with self.assertRaisesRegex(ValueError, "different upstream output"):
            compare(self.upstream, self.native, self.root / "result", measurement)

    def test_unverified_launcher_memory_excluded(self):
        measurement = self.root / "measurement.json"
        write_report(measurement, {"execution_passed": True, "run": {
            "exit_code": 0, "command": ["python", "--output", str(self.upstream)], "wall_seconds": 10,
            "peak_working_set_bytes": 100,
        }})
        with patch("builtins.print"):
            result = compare(self.upstream, self.native, self.root / "result", measurement)
        self.assertNotIn("peak_working_set_bytes", result["upstream_process"])
        self.assertIn("memory_warning", result)


if __name__ == "__main__":
    unittest.main()
