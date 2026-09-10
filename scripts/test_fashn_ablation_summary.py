import unittest

from summarize_fashn_ablation import positive_candidates, restoration_bytes


class AblationSummaryTests(unittest.TestCase):
    def test_storage_cost(self):
        self.assertEqual(restoration_bytes("x_patch_mixer.0.linear1.weight"), 33689600)
        self.assertEqual(restoration_bytes("double_blocks.0.img_mlp.2.weight"), 19251200)

    def test_unknown_or_nonmatrix_rejected(self):
        for name in ("unknown.weight", "x_patch_mixer.0.linear1.bias"):
            with self.subTest(name=name), self.assertRaises(ValueError):
                restoration_bytes(name)

    def test_never_fill_policy_with_negative_interventions(self):
        rows = [{"matrix": "best", "relative_score_reduction": .1},
                {"matrix": "worse", "relative_score_reduction": -.1}]
        self.assertEqual(positive_candidates(rows, 1), ["best"])
        self.assertEqual(positive_candidates(rows, 2), [])


if __name__ == "__main__":
    unittest.main()
