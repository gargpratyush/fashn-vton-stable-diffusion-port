import json
from pathlib import Path
import re
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from export_fashn_runtime_weights import conversion_rules, export


class RuntimeExportTests(unittest.TestCase):
    def fixture(self):
        return {"matrix_types": {f"block.{i}.weight": "q8_0" for i in range(104)}}

    def test_bf16_defaults_protected_to_f32(self):
        default, rule = conversion_rules(self.fixture(), "bf16")
        pattern, dtype = rule.rsplit("=", 1)
        self.assertEqual((default, dtype), ("f32", "bf16"))
        self.assertTrue(re.search(pattern, "model.diffusion_model.block.103.weight"))
        self.assertIsNone(re.search(pattern, "block.0.bias"))
        self.assertIsNone(re.search(pattern, "notblock.0.weight"))

    def test_q8_reuses_native_protection_and_exact_restoration(self):
        self.assertEqual(conversion_rules(self.fixture(), "q8_0"), ("q8_0", ""))
        default, rule = conversion_rules(self.fixture(), "q8_0", {"f32_matrices": ["block.12.weight"]})
        self.assertEqual(default, "q8_0")
        pattern, dtype = rule.rsplit("=", 1)
        self.assertEqual(dtype, "f32")
        self.assertTrue(re.search(pattern, "model.diffusion_model.block.12.weight"))
        self.assertIsNone(re.search(pattern, "block.112.weight"))

    def test_lower_bit_formats_preserve_restoration_rules(self):
        for kind in ("q4_0", "q5_0", "q4_K", "q5_K"):
            with self.subTest(kind=kind):
                self.assertEqual(conversion_rules(self.fixture(), kind), (kind, ""))
                default, rule = conversion_rules(self.fixture(), kind, {"f32_matrices": ["block.12.weight"]})
                pattern, dtype = rule.rsplit("=", 1)
                self.assertEqual((default, dtype), (kind, "f32"))
                self.assertTrue(re.search(pattern, "model.diffusion_model.block.12.weight"))
                self.assertIsNone(re.search(pattern, "block.112.weight"))
        with self.assertRaises(ValueError):
            conversion_rules(self.fixture(), "q2_K")

    def test_rejects_wrong_counts_and_unsafe_names(self):
        data = self.fixture()
        del data["matrix_types"]["block.0.weight"]
        with self.assertRaises(ValueError):
            conversion_rules(data, "bf16")
        data["matrix_types"]["unsafe|all.weight"] = "q8_0"
        with self.assertRaises(ValueError):
            conversion_rules(data, "bf16")

    def test_rejects_invalid_restorations(self):
        for policy in ({"f32_matrices": ["unknown.weight"]},
                       {"f32_matrices": ["block.1.weight", "block.1.weight"]},
                       {"f32_matrices": [4]}, {"f32_matrices": [], "extra": True}):
            with self.subTest(policy=policy), self.assertRaises(ValueError):
                conversion_rules(self.fixture(), "q8_0", policy)

    def test_failed_exports_never_claim_completion(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            stub = root / "stub"
            stub.write_bytes(b"test")
            metadata = root / "matrix-map.json"
            metadata.write_text(json.dumps(self.fixture()), encoding="utf-8")
            for reuse in (False, True):
                args = SimpleNamespace(threads=2, cli=stub, verifier=stub, checkpoint=stub,
                                       matrix_map=metadata, f32_matrices=None, matrix_type="bf16",
                                       reuse_file=stub if reuse else None, output=root / str(reuse))
                with self.subTest(reuse=reuse), patch("export_fashn_runtime_weights.measure_command",
                                                     return_value={"exit_code": 1}) as measured:
                    with self.assertRaises(RuntimeError):
                        export(args)
                    self.assertEqual(measured.call_count, 1)
                    result = json.loads((args.output / "export.json").read_text())
                    self.assertFalse(result["complete"])
                    self.assertNotIn("output_sha256", result)


if __name__ == "__main__":
    unittest.main()
