"""Exercise the actual upstream sampler with a tiny fake model; no weights or parser instances."""
import argparse
import importlib
import os
from pathlib import Path
import unittest

import numpy as np
import torch

from export_fashn_vton_reference import load_reference_package

SOURCE = None


class UpstreamSamplerTest(unittest.TestCase):
    def test_original_sampler_seed_and_batched_cfg_call_count(self):
        if SOURCE is None:
            self.skipTest("Supply --source for the pinned upstream sampler smoke test")
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        pipeline_type = importlib.import_module(load_reference_package(SOURCE) + ".pipeline").TryOnPipeline

        class Model:
            channels_in = 3
            input_shape = (8, 12)
            calls = 0

            def forward_for_cfg(self, image, times, **kwargs):
                self.calls += 1
                return {"v_c": torch.zeros_like(image), "v_u": torch.zeros_like(image)}

        pipeline = pipeline_type.__new__(pipeline_type)
        pipeline.tryon_model = Model()
        rgb, pose = torch.zeros((1, 3, 8, 12)), torch.zeros((1, 1, 8, 12))
        torch.manual_seed(42)
        expected = ((torch.randn((1, 3, 8, 12))[0] + 1) / 2).clamp(0, 1).mul(255).to(torch.uint8).permute(1, 2, 0).numpy()
        torch.manual_seed(42)
        images = pipeline._sample(ca_images=rgb, garment_images=rgb, person_poses=pose, garment_poses=pose,
                                  garment_categories=torch.tensor([1]), num_timesteps=2, guidance_scale=1,
                                  skip_cfg_last_n_steps=1, use_tqdm=False)
        self.assertEqual(pipeline.tryon_model.calls, 2)
        self.assertTrue(np.array_equal(np.array(images[0]), expected))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    args, remaining = parser.parse_known_args()
    SOURCE = args.source
    unittest.main(argv=[__file__, *remaining])
