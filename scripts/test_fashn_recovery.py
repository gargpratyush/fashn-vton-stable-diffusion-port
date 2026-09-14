from pathlib import Path
import os
import sys
import tempfile
import unittest
from unittest.mock import patch

from fashn_artifacts import atomic_text, contained, exclusive_lock, read_json, sha256, write_report
from recover_fashn_study import recover
from run_fashn_job import run


class RecoveryTests(unittest.TestCase):
    @unittest.skipUnless(os.name == "nt", "Windows process-identity integration test")
    def test_actual_python_worker_is_recorded_and_exits(self):
        from benchmark_fashn_vton import measure_command
        from fashn_process_state import identity_active
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            script = root / "worker.py"
            script.write_text("import time\ntime.sleep(.3)\n")
            result = measure_command([sys.executable, str(script)], root / "run.log", 10,
                                     root / "memory.jsonl", root / "process.json")
            process = read_json(root / "process.json")
            self.assertEqual(result["exit_code"], 0)
            self.assertEqual(process["phase"], "exited")
            self.assertIsNotNone(process["worker"])
            self.assertFalse(identity_active(process["worker"]))

    def test_nonfinite_job_timeout_is_rejected_before_output(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            binary = root / "fake.exe"
            binary.write_bytes(b"synthetic")
            for timeout in (float("nan"), float("inf"), 0):
                with self.assertRaises(ValueError):
                    run(root / "output", [str(binary)], timeout)
            self.assertFalse((root / "output").exists())

    def fixture(self, root):
        source = root / "runs/job"
        source.mkdir(parents=True)
        (source / "partial.log").write_text("preserve this interrupted output")
        identity = {"pid": 1, "created": 2, "host": "test"}
        write_report(source / "process.json", {"schema": "fashn-process-v1", "phase": "running",
                                               "owner": identity, "child": identity})
        done = root / "done.txt"
        done.write_text("verified completed output")
        state = {"schema": "fashn-q45-study-v2", "complete": False, "active_job": "job",
                 "provenance": {"files": {str(done): sha256(done)}}, "weights": {},
                 "jobs": {"job": {"status": "interrupted"},
                          "done": {"status": "done", "artifacts": {"done.txt": sha256(done)}}}}
        write_report(root / "results.json", state)
        return state

    def test_recovery_preserves_completed_work_and_partial_attempt(self):
        with tempfile.TemporaryDirectory() as directory, patch("recover_fashn_study.identity_active", return_value=False):
            root = Path(directory)
            before = self.fixture(root)
            archive = recover(root, "job")
            self.assertTrue((archive / "partial.log").is_file())
            after = read_json(root / "results.json")
            self.assertEqual(after["jobs"]["done"], before["jobs"]["done"])
            self.assertNotIn("job", after["jobs"])
            self.assertIsNone(after["active_job"])

    def test_live_unknown_legacy_and_changed_work_refused(self):
        for problem in ("live", "launching", "legacy", "changed", "escape"):
            with self.subTest(problem=problem), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                state = self.fixture(root)
                if problem == "launching":
                    process = read_json(root / "runs/job/process.json")
                    process["phase"] = "launching"
                    write_report(root / "runs/job/process.json", process)
                if problem == "legacy":
                    del state["schema"]
                if problem == "changed":
                    (root / "done.txt").write_text("changed")
                if problem == "escape":
                    state["jobs"]["done"]["artifacts"] = {"../outside": "bad"}
                write_report(root / "results.json", state)
                with patch("recover_fashn_study.identity_active", return_value=problem == "live"):
                    with self.assertRaises(ValueError):
                        recover(root, "job")
                self.assertTrue((root / "runs/job/partial.log").exists())

    def test_recovery_resumes_after_state_write_failure(self):
        with tempfile.TemporaryDirectory() as directory, patch("recover_fashn_study.identity_active", return_value=False):
            root = Path(directory)
            self.fixture(root)
            original = write_report

            def fail_state(path, value):
                if path.name == "results.json":
                    raise OSError("injected disk failure")
                original(path, value)

            with patch("recover_fashn_study.write_report", side_effect=fail_state), self.assertRaises(OSError):
                recover(root, "job")
            self.assertFalse((root / "runs/job").exists())
            self.assertTrue((recover(root, "job") / "partial.log").exists())

    def test_atomic_failure_preserves_original_and_lock_excludes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            target = root / "state.json"
            atomic_text(target, "original")
            with patch("fashn_artifacts.os.replace", side_effect=OSError("injected")), self.assertRaises(OSError):
                atomic_text(target, "replacement")
            self.assertEqual(target.read_text(), "original")
            self.assertEqual(list(root.glob("*.tmp")), [])
            with exclusive_lock(root / "lock"), self.assertRaises(OSError):
                with exclusive_lock(root / "lock"):
                    self.fail("second writer acquired lock")
            with self.assertRaises(ValueError):
                contained(root, "../outside")

    def test_job_finalizes_timeout_and_interrupt(self):
        for error, status in ((TimeoutError("timeout"), "failed"), (KeyboardInterrupt(), "interrupted")):
            with tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                binary = root / "fake.exe"
                binary.write_bytes(b"synthetic")
                with patch("run_fashn_job.measure_command", side_effect=error), self.assertRaises(type(error)):
                    run(root / "output", [str(binary)], 5)
                self.assertEqual(read_json(root / "output/run.json")["status"], status)


if __name__ == "__main__":
    unittest.main()
