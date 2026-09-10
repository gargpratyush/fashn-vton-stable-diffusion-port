from argparse import Namespace
import json
from pathlib import Path
import tempfile
import unittest

import torch
from safetensors.torch import save_file

from export_fashn_vton_reference import CHECKPOINT_SHA256, SOURCE_REVISION, make_noise, oracle_input_state, sha256


class TrajectoryProbeTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.inputs = self.root / "conditions.safetensors"
        save_file({"sentinel": torch.zeros(1)}, str(self.inputs))
        self.image = torch.full((1, 3, 864, 576), .25)
        save_file({"image": self.image}, str(self.root / "step-0.safetensors"))
        self.args = Namespace(trajectory=self.root, inputs=self.inputs, steps=2, step_index=1,
                              cfg=1.5, skip_cfg_last_n_steps=1, seed=42, category="bottoms")
        self.schedule = [0., .25, 1.]
        self.manifest = {
            "source_revision": SOURCE_REVISION, "checkpoint_sha256": CHECKPOINT_SHA256,
            "dtype": "F32", "input_kind": "prepared", "inputs_sha256": sha256(self.inputs),
            "steps": 2, "cfg": 1.5, "skip_cfg_last_n_steps": 1, "seed": 42, "category": "bottoms",
            "schedule": self.schedule,
            "artifacts": {"step-0.safetensors": sha256(self.root / "step-0.safetensors")},
        }
        self.save_manifest()

    def save_manifest(self):
        (self.root / "manifest.json").write_text(json.dumps(self.manifest))

    def test_uses_previous_updated_image_not_new_noise(self):
        image, provenance = oracle_input_state(self.args, self.schedule)
        self.assertTrue(torch.equal(image, self.image))
        self.assertEqual(provenance["state_file"], "step-0.safetensors")

    def test_initial_state_uses_recorded_noise(self):
        self.args.step_index = 0
        save_file({"noise": self.image}, str(self.root / "initial.safetensors"))
        self.manifest["artifacts"]["initial.safetensors"] = sha256(self.root / "initial.safetensors")
        self.save_manifest()
        image, provenance = oracle_input_state(self.args, self.schedule)
        self.assertTrue(torch.equal(image, self.image))
        self.assertEqual(provenance["state_file"], "initial.safetensors")

    def test_default_retains_seeded_noise(self):
        self.args.trajectory = None
        image, provenance = oracle_input_state(self.args, self.schedule)
        self.assertTrue(torch.equal(image, make_noise(1, 864, 576, 42)))
        self.assertEqual(provenance, {"kind": "seeded_noise"})

    def test_rejects_mismatched_settings_or_schedule(self):
        self.args.seed = 43
        with self.assertRaisesRegex(ValueError, "parameter mismatch"):
            oracle_input_state(self.args, self.schedule)
        self.args.seed = 42
        with self.assertRaisesRegex(ValueError, "schedule mismatch"):
            oracle_input_state(self.args, [0., .5, 1.])

    def test_rejects_changed_state(self):
        save_file({"image": self.image + 1}, str(self.root / "step-0.safetensors"))
        with self.assertRaisesRegex(ValueError, "artifact hash mismatch"):
            oracle_input_state(self.args, self.schedule)

    def test_rejects_nonfinite_state_even_with_updated_hash(self):
        save_file({"image": torch.full_like(self.image, float("nan"))}, str(self.root / "step-0.safetensors"))
        self.manifest["artifacts"]["step-0.safetensors"] = sha256(self.root / "step-0.safetensors")
        self.save_manifest()
        with self.assertRaisesRegex(ValueError, "finite F32"):
            oracle_input_state(self.args, self.schedule)


if __name__ == "__main__":
    unittest.main()
