import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from safetensors.numpy import save_file

from run_fashn_q45_study import (
    CASES, LOWER, Study, distribution, pixel_metrics, read_json, reference_view,
    select_candidate, validate_execution, worst_region, write_report,
)
from compare_fashn_trajectories import compare


class LowbitStudyTests(unittest.TestCase):
    def test_variance_definitions(self):
        result = distribution([1, 2, 3])
        self.assertEqual(result["mean"], 2)
        self.assertEqual(result["sample_stddev"], 1)
        self.assertEqual(result["coefficient_of_variation"], .5)
        self.assertIsNone(distribution([1])["sample_stddev"])
        for values in ([], [float("nan")], [float("inf")]):
            with self.assertRaises(ValueError):
                distribution(values)
        expected = np.full((1, 2, 3), 10, dtype=np.uint8)
        actual = expected.copy()
        actual[0, 0] += 2
        actual[0, 1] -= 2
        error = pixel_metrics(expected, actual)
        self.assertEqual(error["rmse"], 2)
        self.assertEqual(error["error_population_variance"], 4)
        self.assertEqual(error["signed_error_mean"], 0)
        self.assertTrue(pixel_metrics(expected, expected)["exact"])

    def test_selection_uses_both_cases_then_time_then_name(self):
        rows = {f"full-{case}-{policy}-s42": {
            "status": "done", "cropped_vs_floating": {"rmse": 3}, "sampling_seconds": 10}
                for case in CASES for policy in LOWER}
        self.assertEqual(select_candidate(rows), "q4_0")
        rows["full-cardigan-q5_K-s42"]["cropped_vs_floating"]["rmse"] = 1
        rows["full-bottoms-q5_K-s42"]["cropped_vs_floating"]["rmse"] = 4
        self.assertEqual(select_candidate(rows), "q5_K")
        rows["full-bottoms-q5_K-s42"]["cropped_vs_floating"]["rmse"] = 5
        rows["full-cardigan-q5_K-s42"]["sampling_seconds"] = 8
        self.assertEqual(select_candidate(rows), "q5_K")
        rows["full-cardigan-q4_0-s42"]["status"] = "running"
        with self.assertRaises(ValueError):
            select_candidate(rows)

    def test_error_region_includes_right_bottom_edges(self):
        expected = np.zeros((75, 83, 3), dtype=np.uint8)
        actual = expected.copy()
        actual[-1, -1] = 255
        region = worst_region(expected, actual)
        self.assertEqual((region["x"], region["y"]), (19, 11))

    def fixture_manifest(self):
        return {"steps": 20, "cfg": 1.5, "shift": 1.5, "skip_cfg_last_n_steps": 1,
                "seed": 42, "threads": 16, "load_threads": 2, "category": 1,
                "backend": "CPU", "attention": "F32 flash", "upcast_matrices": False,
                "mmap": False, "precompute_modulations": True, "fused_gelu": True,
                "noise_source": "cpu_rng", "recorded_trajectory": False, "probe_forwards": 2,
                "matrix_types": {f"block.{i}.weight": "q4_K" for i in range(104)},
                "matrix_backends": {"CPU": 107}}

    def test_rejects_precision_or_execution_drift(self):
        manifest = self.fixture_manifest()
        events = [{"phase": "process_setup", "mapped_model_files": []},
                  {"phase": "sampling_complete", "conversion": {"converted_tensors": 0}, "mapped_model_files": []}]
        validate_execution(manifest, events, "q4_K", set(), "probe", "cardigan", 42)
        for key, value in (("upcast_matrices", True), ("mmap", True), ("backend", "BLAS"),
                           ("seed", 43), ("fused_gelu", False), ("matrix_backends", {"BLAS": 107})):
            broken = copy.deepcopy(manifest)
            broken[key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                validate_execution(broken, events, "q4_K", set(), "probe", "cardigan", 42)
        broken = copy.deepcopy(manifest)
        broken["matrix_types"]["block.0.weight"] = "f32"
        with self.assertRaises(ValueError):
            validate_execution(broken, events, "q4_K", set(), "probe", "cardigan", 42)
        validate_execution(broken, events, "q4_K", {"block.0.weight"}, "probe", "cardigan", 42)
        events[-1]["conversion"]["converted_tensors"] = 1
        with self.assertRaises(ValueError):
            validate_execution(manifest, events, "q4_K", set(), "probe", "cardigan", 42)

    def test_reference_view_preserves_gates_and_detects_changes(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source"
            source.mkdir()
            manifest = {"steps": 1, "cfg": 1.5, "skip_cfg_last_n_steps": 1,
                        "seed": 42, "category": 1, "schedule": [0, 1]}
            write_report(source / "manifest.json", manifest)
            tensor = np.zeros((1, 3, 4, 4), dtype=np.float32)
            save_file({"noise": tensor}, source / "initial.safetensors")
            save_file({"image": tensor, "guided_velocity": tensor}, source / "step-0.safetensors")
            view = reference_view(source, root / "reference", "synthetic_unit_fixture")
            self.assertTrue(compare(view, source, root / "same")["passed"])
            tensor += 1
            save_file({"image": tensor, "guided_velocity": tensor}, source / "step-0.safetensors")
            self.assertFalse(compare(view, source, root / "changed")["passed"])
            self.assertTrue((root / "changed/native.png").exists())
            with self.assertRaises(ValueError):
                reference_view(source, view, "synthetic_unit_fixture")

    def test_partial_gallery_does_not_claim_completion(self):
        with tempfile.TemporaryDirectory() as temporary:
            study = Study.__new__(Study)
            study.root = Path(temporary)
            study.state_path = study.root / "results.json"
            study.weights = {"bf16": {"output_bytes": 1000}}
            study.state = {"complete": False, "phase": "QP3", "active_job": "probe-bf16-0",
                           "jobs": {"probe-bf16-0": {"status": "running"}}}
            study.publish()
            self.assertFalse(read_json(study.state_path)["complete"])
            self.assertIn("probe-bf16-0", (study.root / "index.html").read_text())
            self.assertIn("Human review (not yet performed)", (study.root / "report.md").read_text())

    def test_probe_job_records_native_measurements_and_is_not_relaunched(self):
        from export_fashn_vton_reference import sha256
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            executable = root / "stub.exe"
            executable.write_bytes(b"not a real executable")
            study = Study.__new__(Study)
            study.root = root
            study.binary = executable
            study.conditions = lambda case: executable
            study.state_path = root / "results.json"
            study.weights = {"bf16": {"output": str(executable), "output_sha256": sha256(executable),
                                      "output_bytes": executable.stat().st_size, "matrix_type": "bf16", "policy": None}}
            study.state = {"complete": False, "phase": "QP3", "active_job": None, "jobs": {},
                           "provenance": {"files": {str(executable): sha256(executable)}}}

            def native_fixture(command, log, timeout, timeline):
                destination = Path(command[3])
                destination.mkdir()
                manifest = self.fixture_manifest()
                manifest["matrix_types"] = {name: "bf16" for name in manifest["matrix_types"]}
                manifest.update(upcast_matrices=True, sampling_seconds=3, probe_forward_seconds=[2, 1])
                write_report(destination / "manifest.json", manifest)
                event = {"phase": "sampling_complete", "mapped_model_files": [],
                         "conversion": {"converted_tensors": 0, "scratch_peak_bytes": 0},
                         "weights": {"assigned_payload_bytes": 100}, "runtime_buffer_bytes": 50,
                         "planned_cpu_work_bytes": 10, "modulation_cache_bytes": 5}
                (destination / "memory-profile.jsonl").write_text(json.dumps(event) + "\n")
                save_file({"velocity": np.zeros((1, 3, 864, 576), dtype=np.float32)},
                          destination / "probe.safetensors")
                log.write_text("synthetic native-process fixture")
                return {"exit_code": 0, "wall_seconds": 4, "peak_working_set_bytes": 1000,
                        "sampled_peak_private_bytes": 1100}

            with patch("run_fashn_q45_study.measure_command", side_effect=native_fixture) as measured:
                row = study.run_job("probe-bf16-0", "cardigan", "bf16", kind="probe")
                self.assertEqual(row["status"], "done")
                self.assertEqual(row["forward_seconds"], [2, 1])
                self.assertEqual(row["velocity_vs_floating"]["relative_l2"], 0)
                self.assertIn("runs/probe-bf16-0/trajectory/probe.safetensors", row["artifacts"])
                study.run_job("probe-bf16-0", "cardigan", "bf16", kind="probe")
                self.assertEqual(measured.call_count, 1)


if __name__ == "__main__":
    unittest.main()
