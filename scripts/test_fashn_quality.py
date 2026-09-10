from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import json

from PIL import Image
from run_fashn_quality import cases, contact_sheet, run, validate_assets, write_report
from export_fashn_vton_reference import sha256


class QualityTests(unittest.TestCase):
    def test_case_coverage(self):
        matrix = cases(Path("assets"), Path("source"))
        self.assertEqual(len(matrix), 6)
        self.assertEqual({row[3] for row in matrix}, {"tops", "bottoms", "one-pieces"})
        self.assertEqual({row[4] for row in matrix}, {"flat-lay", "model"})
        self.assertEqual({row[5] for row in matrix}, {False, True})
        self.assertEqual(matrix[0][1:6], matrix[-1][1:6])
        self.assertNotEqual(matrix[0][-1], matrix[-1][-1])

    def test_consent_before_io(self):
        with self.assertRaisesRegex(ValueError, "consent"):
            run(SimpleNamespace(accept_noncommercial_demo_license=False, accept_parser_research_license=True))

    def test_missing_assets_and_review_sheet(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            with self.assertRaisesRegex(ValueError, "Missing"):
                validate_assets(directory, False)
            source = directory / "source.png"
            Image.new("RGB", (40, 60), "red").save(source)
            contact_sheet(source, source, source, source, directory / "review.jpg", "test")
            with Image.open(directory / "review.jpg") as image:
                self.assertEqual(image.size, (1200, 520))
            write_report(directory / "report.json", {"passed": False})
            self.assertFalse((directory / "report.json.tmp").exists())

    def test_limited_batches_resume_without_claiming_full_completion(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source"
            (source / "examples" / "data").mkdir(parents=True)
            (source / "examples" / "data" / "model.webp").write_bytes(b"fixture identity")
            image = root / "input.png"
            Image.new("RGB", (4, 6), "red").save(image)
            binary = root / "binary"
            binary.write_bytes(b"mock executable/checkpoint")
            parser_dir, pose_dir = root / "parser", root / "pose"
            parser_dir.mkdir()
            pose_dir.mkdir()
            (parser_dir / "manifest.json").write_text("{}")
            args = SimpleNamespace(accept_noncommercial_demo_license=True, accept_parser_research_license=True,
                assets=root, source=source, cli=binary, preparer=binary, checkpoint=binary,
                dwpose_dir=pose_dir, parser_dir=parser_dir, output=root / "suite",
                steps=20, threads=8, max_new_cases=1, resume=False, prepare_only=False, download=False)
            matrix = [(name, image, image, "tops", "flat-lay", False, 42) for name in ("a", "b")]

            def prepare(command, **kwargs):
                destination = Path(command[command.index("--output") + 1])
                destination.mkdir()
                Image.new("RGB", (4, 6), "red").save(destination / "ca_image.png")
                write_report(destination / "manifest.json", {"category": "tops"})
                write_report(destination / "provenance.json", {
                    "inputs_sha256": {"person": sha256(image), "garment": sha256(image)},
                    "artifacts": {"ca_image.png": sha256(destination / "ca_image.png")}})

            def measure(options):
                options.output.mkdir()
                Image.new("RGB", (4, 6), "red").save(options.output / "sample-0.png")
                write_report(options.output / "benchmark.json", {
                    "executable_sha256": sha256(binary), "checkpoint_sha256": sha256(binary),
                    "runs": [{"wall_seconds": 1., "peak_working_set_bytes": 100}]})

            with patch("run_fashn_quality.validate_assets"), patch("run_fashn_quality.cases", return_value=matrix), \
                    patch("run_fashn_quality.subprocess.run", side_effect=prepare), \
                    patch("run_fashn_quality.benchmark", side_effect=measure) as measured:
                run(args)
                report = json.loads((args.output / "suite.json").read_text())
                self.assertFalse(report["execution_passed"])
                self.assertEqual(report["expected_cases"], ["a", "b"])
                args.resume = True
                run(args)
                self.assertTrue(json.loads((args.output / "suite.json").read_text())["execution_passed"])
                self.assertEqual(measured.call_count, 2)


if __name__ == "__main__":
    unittest.main()
