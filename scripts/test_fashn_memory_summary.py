import copy
import os
from pathlib import Path
import subprocess
import tempfile
import unittest

from summarize_fashn_memory import summarize_phases, validate_controls


class MemorySummaryTests(unittest.TestCase):
    def fixture(self, probes=0):
        manifest = {"memory_profile": True, "load_only": probes == 0, "probe_forwards": probes,
                    "mmap": False, "load_threads": 2, "matrix_type": "q8_0", "upcast_matrices": False}
        names = ["process_setup", "metadata_ready", "load_only_ready"]
        for _ in range(probes):
            names.extend(("graph_execute_begin", "graph_execute_end"))
        names.extend(("before_model_release", "model_released"))
        events = [{"phase": name, "elapsed_seconds": index,
                   "weights": {"registered_payload_bytes": 100, "assigned_payload_bytes": 100},
                   "conversion": {"scratch_live_bytes": 0}, "mapped_model_files": [],
                   "runtime_buffer_bytes": 0, "planned_cpu_work_bytes": 0,
                   "diagnostic_seconds": .01, "process": {"working_set_bytes": 100}}
                  for index, name in enumerate(names)]
        return manifest, events

    def test_load_only_is_not_inference(self):
        row = summarize_phases(*self.fixture())
        self.assertEqual(row["scope"], "load_only")
        self.assertEqual(row["graph_execution_seconds_instrumented"], [])

    def test_repeated_forward_count(self):
        manifest, events = self.fixture(2)
        self.assertEqual(summarize_phases(manifest, events)["graph_execution_seconds_instrumented"], [1, 1])
        manifest["probe_forwards"] = 3
        with self.assertRaisesRegex(ValueError, "Incomplete graph"):
            summarize_phases(manifest, events)

    def test_rejects_live_scratch_mapping_or_unassigned_weights(self):
        manifest, events = self.fixture()
        for mutation in ("scratch", "mapping", "weights"):
            modified = copy.deepcopy(events)
            ready = modified[2]
            if mutation == "scratch":
                ready["conversion"]["scratch_live_bytes"] = 10
            elif mutation == "mapping":
                ready["mapped_model_files"] = [{"virtual_bytes": 10}]
            else:
                ready["weights"]["assigned_payload_bytes"] = 90
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                summarize_phases(manifest, modified)

    def test_rejects_partial_and_nonmonotonic_profile(self):
        manifest, events = self.fixture()
        with self.assertRaises(ValueError):
            summarize_phases(manifest, events[:-1])
        events[2]["elapsed_seconds"] = -1
        with self.assertRaisesRegex(ValueError, "backwards"):
            summarize_phases(manifest, events)

    def test_sampling_count_respects_cfg_skip(self):
        manifest, events = self.fixture(3)
        manifest.update(load_only=False, probe_forwards=0, steps=2, cfg=1.5, skip_cfg_last_n_steps=1)
        self.assertEqual(summarize_phases(manifest, events)["scope"], "sampling")
        manifest["cfg"] = 1
        with self.assertRaisesRegex(ValueError, "Incomplete graph"):
            summarize_phases(manifest, events)

    def test_rejects_control_manifest_mismatch(self):
        manifest, _ = self.fixture(2)
        manifest.update(page_classification=True, recorded_trajectory=False)
        command = ["trajectory", "checkpoint", "conditions", "output",
                   "20", "1.5", "1.5", "1", "42", "2", "16",
                   "--matrix-type", "q8_0", "--mmap", "off", "--load-threads", "2",
                   "--probe-forwards", "2", "--memory-profile", "--classify-pages", "--no-record"]
        validate_controls(manifest, command)
        command.append("--fused-gelu")
        with self.assertRaisesRegex(ValueError, "mismatch: fused_gelu"):
            validate_controls(manifest, command)
        manifest["fused_gelu"] = True
        validate_controls(manifest, command)
        manifest["mmap"] = True
        with self.assertRaisesRegex(ValueError, "mismatch: mmap"):
            validate_controls(manifest, command)

    def test_inactive_weights_require_explicit_precomputation(self):
        manifest, events = self.fixture(2)
        ready = next(event for event in events if event["phase"] == "before_model_release")
        ready["weights"]["assigned_payload_bytes"] = 60
        ready["inactive_modulation_parameter_bytes"] = 40
        ready["modulation_cache_bytes"] = 3
        with self.assertRaisesRegex(ValueError, "without modulation"):
            summarize_phases(manifest, events)
        manifest["precompute_modulations"] = True
        result = summarize_phases(manifest, events)
        self.assertEqual(result["inactive_modulation_parameter_bytes"], 40)
        self.assertEqual(result["modulation_cache_bytes"], 3)
        ready["weights"]["assigned_payload_bytes"] = 61
        with self.assertRaisesRegex(ValueError, "Not all parameters"):
            summarize_phases(manifest, events)


@unittest.skipUnless(os.environ.get("FASHN_TRAJECTORY_EXE"), "Set FASHN_TRAJECTORY_EXE for argument regressions")
class DiagnosticArgumentTests(unittest.TestCase):
    def test_rejects_invalid_options_before_model_loading(self):
        cases = [
            (["--mmap", "auto"], "mmap must be on or off"),
            (["--load-threads", "0"], "Diagnostic counts"),
            (["--probe-forwards", "1025"], "Diagnostic counts"),
            (["--load-threads", "2junk"], "Diagnostic counts"),
            (["--load-threads"], "Unknown or incomplete"),
            (["--load-only", "--probe-forwards", "1"], "exclusive"),
            (["--classify-pages"], "requires profiling"),
            (["--precompute-modulations"], "requires --mmap off"),
            (["--precompute-modulations", "--mmap", "off", "--load-only"], "inference mode"),
            (["--modulation-cache-mib", "1"], "must be enabled"),
            (["--precompute-modulations", "--mmap", "off", "--modulation-cache-mib", "129"], "1..128"),
            (["--unknown-option"], "Unknown or incomplete"),
        ]
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "output"
            command = [os.environ["FASHN_TRAJECTORY_EXE"], str(Path(temporary) / "missing.safetensors"),
                       str(Path(temporary) / "conditions.safetensors"), str(output),
                       "20", "1.5", "1.5", "1", "42", "2", "16"]
            for options, message in cases:
                with self.subTest(options=options):
                    result = subprocess.run(command + options, capture_output=True, text=True, timeout=10)
                    self.assertNotEqual(result.returncode, 0)
                    self.assertIn(message, result.stderr)
                    self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main()
