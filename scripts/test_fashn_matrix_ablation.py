import unittest

from run_fashn_matrix_ablation import ranking, summarize


class MatrixAblationTests(unittest.TestCase):
    def result(self):
        return {"matrix_type": "q8_0", "flash_attention_nodes": 28, "matrix_types": {"a.weight": "q8_0"},
                "passed": False, **{branch: {name: {"relative_l2": .01, "passed": False}
                                           for name in ["velocity", *map(str, range(16))]}
                                  for branch in ("conditional", "unconditional")}}

    def test_numerical_rejection_is_a_valid_measurement(self):
        row = summarize(self.result(), ["a.weight"], [])
        self.assertEqual(row["failed_captures"], 34)
        self.assertEqual(row["score"], .01)
        self.assertFalse(row["numerical_gate_passed"])

    def test_intervention_must_actually_apply(self):
        with self.assertRaisesRegex(ValueError, "precision"):
            summarize(self.result(), ["a.weight"], ["a.weight"])

    def test_incomplete_or_nonfinite_rejected(self):
        result = self.result()
        del result["conditional"]["velocity"]
        with self.assertRaisesRegex(ValueError, "Incomplete"):
            summarize(result, ["a.weight"], [])
        result = self.result()
        result["conditional"]["velocity"]["relative_l2"] = float("nan")
        with self.assertRaisesRegex(ValueError, "Invalid capture"):
            summarize(result, ["a.weight"], [])

    def test_ranking_preserves_negative_interventions(self):
        rows = ranking(.01, {"better": {"summary": {"score": .005}},
                             "worse": {"summary": {"score": .02}}})
        self.assertEqual([row["matrix"] for row in rows], ["better", "worse"])
        self.assertAlmostEqual(rows[0]["relative_score_reduction"], .5)
        self.assertAlmostEqual(rows[1]["relative_score_reduction"], -1)


if __name__ == "__main__":
    unittest.main()
