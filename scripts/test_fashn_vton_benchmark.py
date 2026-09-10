"""Focused benchmark command and validation regressions; no model needed."""

from argparse import Namespace
from pathlib import Path
import tempfile
import sys
import json
import unittest
from unittest.mock import patch

import benchmark_fashn_vton as benchmark


class BenchmarkTests(unittest.TestCase):
    def args(self):
        return Namespace(cli=Path("sd-cli.exe"), checkpoint=Path("model.safetensors"),
                         prepared=Path("manifest.json"), output=Path("benchmark"),
                         steps=1, cfg=1., shift=1.5, skip_cfg_last_n_steps=1,
                         seed=42, threads=8, repeats=1, timeout=900., flash_attention=False, weight_type="f32")

    def test_manual_command(self):
        args = self.args()
        command = benchmark.command_for(args, Path("result.png"))
        self.assertEqual(command[0], str(args.cli.resolve()))
        self.assertEqual(command[command.index("--try-on-inputs") + 1], str(args.prepared.resolve()))
        self.assertEqual(command[command.index("--rng") + 1], "cpu")
        self.assertNotIn("--diffusion-fa", command)

    def test_flash_quality_command(self):
        args = self.args()
        args.flash_attention, args.steps, args.cfg = True, 20, 1.5
        command = benchmark.command_for(args, Path("result.png"))
        for flag, expected in (("--steps", "20"), ("--cfg-scale", "1.5"),
                               ("--skip-cfg-last-n-steps", "1"), ("-t", "8")):
            self.assertEqual(command[command.index(flag) + 1], expected)
        self.assertEqual(command.count("--diffusion-fa"), 1)
        args.weight_type = "bf16"
        command = benchmark.command_for(args, Path("result.png"))
        self.assertEqual(command[command.index("--type") + 1], "bf16")

    def test_invalid_settings_do_not_launch(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(benchmark.os, "name", "nt"), \
                patch.object(benchmark.subprocess, "Popen") as launch:
            for key in ("steps", "threads", "repeats", "timeout"):
                args = self.args()
                args.output = Path(directory) / "must-not-exist"
                setattr(args, key, 0)
                with self.subTest(key=key), self.assertRaises(ValueError):
                    benchmark.benchmark(args)
                self.assertFalse(args.output.exists())
            launch.assert_not_called()

    def test_rejects_unsupported_platform(self):
        args = self.args()
        with patch.object(benchmark.os, "name", "posix"), self.assertRaisesRegex(ValueError, "Windows"):
            benchmark.benchmark(args)

    @unittest.skipUnless(benchmark.os.name == "nt", "Windows process measurement")
    def test_actual_python_worker_memory(self):
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            script = directory / "allocate.py"
            script.write_text("import time\nbuffer = bytearray(32 * 1024 * 1024)\ntime.sleep(.8)\n")
            timeline = directory / "timeline.jsonl"
            row = benchmark.measure_command([sys.executable, str(script)], directory / "run.log", 20, timeline)
            identity = json.loads((directory / "run.worker.json").read_text())
            self.assertEqual(row["exit_code"], 0)
            self.assertEqual(row["monitored_pid"], identity["pid"])
            self.assertEqual(row["memory_scope"], "active_python_interpreter")
            self.assertGreater(row["peak_working_set_bytes"], 32 * 1024**2)
            samples = [json.loads(line) for line in timeline.read_text().splitlines()]
            self.assertEqual(len(samples), row["memory_samples"])
            self.assertEqual(row["memory_timeline"], str(timeline))
            self.assertTrue(all(sample["pid"] == identity["pid"] for sample in samples))
            self.assertTrue(all(sample["peak_working_set_bytes"] >= sample["working_set_bytes"] for sample in samples))
            self.assertTrue(all(a["elapsed_seconds"] < b["elapsed_seconds"] for a, b in zip(samples, samples[1:])))

    @unittest.skipUnless(benchmark.os.name == "nt", "Windows process measurement")
    def test_native_process_memory_path(self):
        with tempfile.TemporaryDirectory() as directory:
            shell = Path(benchmark.os.environ["SystemRoot"]) / "System32" / "WindowsPowerShell" / "v1.0" / "powershell.exe"
            row = benchmark.measure_command([str(shell), "-NoProfile", "-NonInteractive", "-Command",
                                             "Start-Sleep -Milliseconds 600"], Path(directory) / "run.log", 30)
            self.assertEqual(row["exit_code"], 0)
            self.assertEqual(row["monitored_pid"], row["launcher_pid"])
            self.assertEqual(row["memory_scope"], "launched_process")

    @unittest.skipUnless(benchmark.os.name == "nt", "Windows worker registration")
    def test_worker_waits_for_owner_acknowledgement(self):
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            marker, worker = directory / "executed.txt", directory / "worker.json"
            script = directory / "payload.py"
            script.write_text(f"from pathlib import Path\nPath({str(marker)!r}).write_text('executed')\n")
            environment = dict(benchmark.os.environ, FASHN_MEASUREMENT_PID_FILE=str(worker))
            process = benchmark.subprocess.Popen(
                [sys.executable, str(Path(benchmark.__file__).with_name("fashn_measurement_worker.py")), str(script)],
                env=environment, stdout=benchmark.subprocess.DEVNULL, stderr=benchmark.subprocess.DEVNULL)
            try:
                deadline = benchmark.time.monotonic() + 20
                while not worker.exists() and benchmark.time.monotonic() < deadline:
                    benchmark.time.sleep(.01)
                self.assertTrue(worker.exists())
                self.assertFalse(marker.exists())
                identity = json.loads(worker.read_text())
                temporary = directory / "ack.tmp"
                temporary.write_text(json.dumps({"pid": identity["pid"]}))
                temporary.replace(worker.with_suffix(".ack.json"))
                self.assertEqual(process.wait(timeout=10), 0)
                self.assertTrue(marker.exists())
            finally:
                if process.poll() is None:
                    benchmark.terminate_owned_tree(process.pid)
                    process.wait(timeout=30)

    @unittest.skipUnless(benchmark.os.name == "nt", "Windows process measurement")
    def test_python_timeout_terminates_worker_tree(self):
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            script = directory / "sleep.py"
            script.write_text("import time\ntime.sleep(60)\n")
            with self.assertRaises(TimeoutError):
                benchmark.measure_command([sys.executable, str(script)], directory / "run.log", 10)
            pid = json.loads((directory / "run.worker.json").read_text())["pid"]
            kernel = benchmark.ctypes.WinDLL("kernel32", use_last_error=True)
            kernel.OpenProcess.argtypes = [benchmark.wintypes.DWORD, benchmark.wintypes.BOOL, benchmark.wintypes.DWORD]
            kernel.OpenProcess.restype = benchmark.wintypes.HANDLE
            kernel.GetExitCodeProcess.argtypes = [benchmark.wintypes.HANDLE, benchmark.ctypes.POINTER(benchmark.wintypes.DWORD)]
            kernel.CloseHandle.argtypes = [benchmark.wintypes.HANDLE]
            handle = kernel.OpenProcess(0x0400, False, pid)
            if handle:
                try:
                    code = benchmark.wintypes.DWORD()
                    self.assertTrue(kernel.GetExitCodeProcess(handle, benchmark.ctypes.byref(code)))
                    self.assertNotEqual(code.value, 259)
                finally:
                    kernel.CloseHandle(handle)


if __name__ == "__main__":
    unittest.main()
