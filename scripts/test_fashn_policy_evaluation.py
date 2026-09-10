import argparse
from pathlib import Path
import tempfile
import unittest

from evaluate_fashn_precision_policies import labeled_path, policy_names, write_report


class PolicyEvaluationTests(unittest.TestCase):
    def test_labeled_paths(self):
        self.assertEqual(labeled_path("step15=fixtures")[0], "step15")
        for value in ("../escape=fixtures", "no-label", "empty="):
            with self.subTest(value=value), self.assertRaises(argparse.ArgumentTypeError):
                labeled_path(value)

    def test_policy_membership_and_duplicates(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "policy.json"
            write_report(path, {"f32_matrices": ["matrix.weight"]})
            self.assertEqual(policy_names(path, {"matrix.weight"}), ["matrix.weight"])
            for names in (["matrix.weight"] * 2, ["other.weight"], [3]):
                write_report(path, {"f32_matrices": names})
                with self.subTest(names=names), self.assertRaises(ValueError):
                    policy_names(path, {"matrix.weight"})


if __name__ == "__main__":
    unittest.main()
