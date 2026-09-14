#!/usr/bin/env python3
"""Windows CPU try-on benchmark: isolated process wall time and kernel peak working set."""

import argparse
from contextlib import nullcontext
import ctypes
from ctypes import wintypes
import json
import math
import os
from pathlib import Path
import subprocess
import time

from export_fashn_vton_reference import sha256, validate_sampling
from fashn_process_state import process_identity, record_process, identity_active


class MemoryCounters(ctypes.Structure):
    _fields_ = [("cb", wintypes.DWORD), ("page_faults", wintypes.DWORD)] + [
        (name, ctypes.c_size_t) for name in (
            "peak_working_set", "working_set", "peak_paged_pool", "paged_pool",
            "peak_nonpaged_pool", "nonpaged_pool", "pagefile", "peak_pagefile", "private_usage")]


class ProcessMemory:
    def __init__(self, pid):
        self.kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        self.psapi = ctypes.WinDLL("psapi", use_last_error=True)
        self.kernel.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
        self.kernel.OpenProcess.restype = wintypes.HANDLE
        self.kernel.CloseHandle.argtypes = [wintypes.HANDLE]
        self.psapi.GetProcessMemoryInfo.argtypes = [wintypes.HANDLE, ctypes.POINTER(MemoryCounters), wintypes.DWORD]
        self.psapi.GetProcessMemoryInfo.restype = wintypes.BOOL
        self.handle = self.kernel.OpenProcess(0x0400 | 0x0010, False, pid)
        if not self.handle:
            raise ctypes.WinError(ctypes.get_last_error())

    def read(self):
        counters = MemoryCounters()
        counters.cb = ctypes.sizeof(counters)
        if not self.psapi.GetProcessMemoryInfo(self.handle, ctypes.byref(counters), counters.cb):
            raise ctypes.WinError(ctypes.get_last_error())
        return counters

    def close(self):
        self.kernel.CloseHandle(self.handle)


def terminate_owned_tree(root_pid):
    class ProcessEntry(ctypes.Structure):
        _fields_ = [("size", wintypes.DWORD), ("usage", wintypes.DWORD), ("pid", wintypes.DWORD),
                    ("heap", ctypes.c_size_t), ("module", wintypes.DWORD), ("threads", wintypes.DWORD),
                    ("parent", wintypes.DWORD), ("priority", wintypes.LONG), ("flags", wintypes.DWORD),
                    ("filename", wintypes.WCHAR * 260)]

    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel.CreateToolhelp32Snapshot.argtypes = [wintypes.DWORD, wintypes.DWORD]
    kernel.CreateToolhelp32Snapshot.restype = wintypes.HANDLE
    kernel.Process32FirstW.argtypes = kernel.Process32NextW.argtypes = [wintypes.HANDLE, ctypes.POINTER(ProcessEntry)]
    kernel.CloseHandle.argtypes = [wintypes.HANDLE]
    kernel.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
    kernel.OpenProcess.restype = wintypes.HANDLE
    kernel.GetExitCodeProcess.argtypes = [wintypes.HANDLE, ctypes.POINTER(wintypes.DWORD)]
    kernel.GetProcessTimes.argtypes = [wintypes.HANDLE, *([ctypes.POINTER(wintypes.FILETIME)] * 4)]
    kernel.TerminateProcess.argtypes = [wintypes.HANDLE, wintypes.UINT]
    kernel.WaitForSingleObject.argtypes = [wintypes.HANDLE, wintypes.DWORD]
    kernel.WaitForSingleObject.restype = wintypes.DWORD
    snapshot = kernel.CreateToolhelp32Snapshot(2, 0)
    if snapshot == ctypes.c_void_p(-1).value:
        raise ctypes.WinError(ctypes.get_last_error())
    children = {}
    try:
        entry = ProcessEntry()
        entry.size = ctypes.sizeof(entry)
        more = kernel.Process32FirstW(snapshot, ctypes.byref(entry))
        while more:
            children.setdefault(entry.parent, []).append(entry.pid)
            more = kernel.Process32NextW(snapshot, ctypes.byref(entry))
        if ctypes.get_last_error() != 18:
            raise ctypes.WinError(ctypes.get_last_error())
    finally:
        kernel.CloseHandle(snapshot)
    owned = []

    def retain(pid, parent_created=0):
        handle = kernel.OpenProcess(0x100001 | 0x0400, False, pid)
        if not handle:
            if ctypes.get_last_error() == 87:
                return  # The owned process already exited after the snapshot.
            raise ctypes.WinError(ctypes.get_last_error())
        times = [wintypes.FILETIME() for _ in range(4)]
        if not kernel.GetProcessTimes(handle, *[ctypes.byref(value) for value in times]):
            error = ctypes.get_last_error()
            kernel.CloseHandle(handle)
            raise ctypes.WinError(error)
        created = times[0].dwLowDateTime | (times[0].dwHighDateTime << 32)
        if created < parent_created:
            kernel.CloseHandle(handle)
            return  # Exclude old orphaned processes with a reused parent PID.
        owned.append((pid, handle, created))

    try:
        retain(root_pid)
        for pid, _, created in owned:
            for child in children.get(pid, []):
                if not any(child == row[0] for row in owned):
                    retain(child, created)
        for pid, handle, _ in reversed(owned):
            code = wintypes.DWORD()
            if not kernel.GetExitCodeProcess(handle, ctypes.byref(code)):
                raise ctypes.WinError(ctypes.get_last_error())
            if code.value == 259 and not kernel.TerminateProcess(handle, 1):
                error = ctypes.get_last_error()
                if not kernel.GetExitCodeProcess(handle, ctypes.byref(code)) or code.value == 259:
                    raise ctypes.WinError(error)
            if kernel.WaitForSingleObject(handle, 30000) != 0:
                raise RuntimeError("Owned process did not terminate: " + str(pid))
    finally:
        for _, handle, _ in owned:
            kernel.CloseHandle(handle)


def command_for(args, output):
    command = [
        str(args.cli.resolve()), "--mode", "try_on", "--diffusion-model", str(args.checkpoint.resolve()),
        "--try-on-inputs", str(args.prepared.resolve()), "--steps", str(args.steps),
        "--cfg-scale", str(args.cfg), "--flow-shift", str(args.shift),
        "--skip-cfg-last-n-steps", str(args.skip_cfg_last_n_steps), "--seed", str(args.seed),
        "--rng", "cpu", "--type", args.weight_type, "-t", str(args.threads), "-o", str(output)]
    if args.flash_attention:
        command.append("--diffusion-fa")
    return command


def measure_command(command, log_path, timeout=14400, timeline_path=None, process_path=None):
    if os.name != "nt" or not math.isfinite(timeout) or timeout < 1:
        raise ValueError("Command measurement requires Windows and a positive timeout")
    peak = private = samples = 0
    last_sample_time = 0.
    python = Path(command[0]).name.lower() in ("python.exe", "pythonw.exe")
    worker_file = log_path.with_suffix(".worker.json") if python else None
    environment = os.environ.copy()
    launched_command = command
    if python:
        if (len(command) < 2 or not Path(command[1]).is_file() or worker_file.exists()
                or worker_file.with_suffix(".ack.json").exists()):
            raise ValueError("Python measurement needs a script file and a fresh worker registration path")
        environment["FASHN_MEASUREMENT_PID_FILE"] = str(worker_file.resolve())
        launched_command = [command[0], str(Path(__file__).with_name("fashn_measurement_worker.py")), *command[1:]]
    start = time.perf_counter()
    monitored_pid = None
    owner = process_identity(os.getpid()) if process_path is not None else None
    child = None
    worker_identity = None
    if process_path is not None:
        record_process(process_path, "pending", owner)
    with log_path.open("w", encoding="utf-8") as log, \
            (timeline_path.open("x", encoding="utf-8") if timeline_path is not None else nullcontext()) as timeline:
        if process_path is not None:
            record_process(process_path, "launching", owner)
        try:
            process = subprocess.Popen(launched_command, stdout=log, stderr=subprocess.STDOUT, env=environment)
        except OSError:
            if process_path is not None:
                record_process(process_path, "exited", owner)
            raise
        memory = None
        try:
            if process_path is not None:
                child = process_identity(process.pid)
                record_process(process_path, "running", owner, child)
            if not python:
                monitored_pid = process.pid
                memory = ProcessMemory(monitored_pid)
            while process.poll() is None:
                if python and memory is None and worker_file.exists():
                    identity = json.loads(worker_file.read_text(encoding="utf-8"))
                    monitored_pid = identity["pid"]
                    if (type(monitored_pid) is not int or monitored_pid <= 0 or
                            (monitored_pid != process.pid and identity["parent_pid"] != process.pid)):
                        raise ValueError("Worker is not the launched interpreter or its direct child")
                    memory = ProcessMemory(monitored_pid)
                    if process_path is not None:
                        worker_identity = process_identity(monitored_pid)
                        record_process(process_path, "running", owner, child, worker_identity)
                    acknowledgement = worker_file.with_suffix(".ack.json")
                    temporary = acknowledgement.with_suffix(".tmp")
                    temporary.write_text(json.dumps({"pid": monitored_pid}), encoding="utf-8")
                    temporary.replace(acknowledgement)
                if memory is not None:
                    counters = memory.read()
                    peak = max(peak, counters.peak_working_set)
                    private = max(private, counters.private_usage)
                    samples += 1
                    last_sample_time = time.perf_counter() - start
                    if timeline is not None:
                        timeline.write(json.dumps({
                            "unix_ns": time.time_ns(), "elapsed_seconds": last_sample_time,
                            "pid": monitored_pid, "working_set_bytes": counters.working_set,
                            "peak_working_set_bytes": counters.peak_working_set,
                            "private_commit_bytes": counters.private_usage,
                            "page_fault_count": counters.page_faults}, allow_nan=False) + "\n")
                        timeline.flush()
                if time.perf_counter() - start > timeout:
                    raise TimeoutError("Benchmark inference timeout")
                time.sleep(.2)
        finally:
            try:
                if python and worker_identity and identity_active(worker_identity) and process.poll() is not None:
                    terminate_owned_tree(process.pid)
                if process.poll() is None:
                    if python:
                        terminate_owned_tree(process.pid)
                    else:
                        process.terminate()
                    process.wait(timeout=30)
            finally:
                if memory is not None:
                    memory.close()
            if process_path is not None:
                record_process(process_path, "exited", owner, child, worker_identity)
    if python and memory is None:
        raise RuntimeError("Python worker did not register; memory measurement is unavailable")
    return {"command": command, "exit_code": process.returncode, "wall_seconds": time.perf_counter() - start,
            "peak_working_set_bytes": peak, "sampled_peak_private_bytes": private,
            "memory_samples": samples, "last_memory_sample_seconds": last_sample_time,
            "launcher_pid": process.pid, "monitored_pid": monitored_pid, "launched_command": launched_command,
            "memory_timeline": str(timeline_path) if timeline_path is not None else None,
            "memory_scope": "active_python_interpreter" if python else "launched_process"}


def benchmark(args):
    if os.name != "nt":
        raise ValueError("This benchmark currently measures Windows process memory only")
    from PIL import Image
    validate_sampling(args.steps, 0, args.cfg, args.shift, args.skip_cfg_last_n_steps)
    if args.threads < 1 or args.repeats < 1 or args.timeout < 1:
        raise ValueError("Threads, repeats, and timeout must be positive")
    manifest = json.loads(args.prepared.read_text(encoding="utf-8"))
    inputs = {}
    for key in ("ca_image", "garment_image", "person_pose", "garment_pose"):
        inputs[key] = sha256(args.prepared.parent / manifest[key])
    provenance = {
        "executable_sha256": sha256(args.cli), "checkpoint_sha256": sha256(args.checkpoint),
        "manifest_sha256": sha256(args.prepared), "inputs_sha256": inputs,
        "attention": "flash" if args.flash_attention else "manual",
        "settings": {key: getattr(args, key) for key in
                     ("steps", "cfg", "shift", "skip_cfg_last_n_steps", "seed", "threads", "repeats", "weight_type")},
        "measurement": "Separate cold process per repeat; wall time includes model loading and PNG encoding. "
                       "Peak working set is the Windows kernel high-water value through the last successful sample; "
                       "private usage is sampled, not a kernel peak. Do not compare concurrently loaded runs.",
        "runs": [], "passed": False,
    }
    args.output.mkdir(parents=True, exist_ok=False)
    report = args.output / "benchmark.json"
    report.write_text(json.dumps(provenance, indent=2) + "\n", encoding="utf-8")
    for index in range(args.repeats):
        output = args.output.resolve() / f"sample-{index}.png"
        command = command_for(args, output)
        run = measure_command(command, args.output / f"run-{index}.log", args.timeout)
        provenance["runs"].append(run)
        report.write_text(json.dumps(provenance, indent=2) + "\n", encoding="utf-8")
        if run["exit_code"] != 0:
            raise RuntimeError(f"Native inference failed; see {args.output / f'run-{index}.log'}")
        crop = manifest.get("crop", {"width": 576, "height": 864})
        with Image.open(output) as image:
            if image.mode != "RGB" or image.size != (crop["width"], crop["height"]):
                raise ValueError("Benchmark output shape/mode mismatch")
        run["output_sha256"] = sha256(output)
        print(f"Run {index}: {run['wall_seconds']:.2f}s; peak working set {run['peak_working_set_bytes'] / 1024**3:.3f} GiB", flush=True)
    provenance["passed"] = True
    report.write_text(json.dumps(provenance, indent=2) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("cli", "checkpoint", "prepared", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--steps", type=int, default=1)
    parser.add_argument("--cfg", type=float, default=1.)
    parser.add_argument("--shift", type=float, default=1.5)
    parser.add_argument("--skip-cfg-last-n-steps", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--timeout", type=float, default=14400)
    parser.add_argument("--flash-attention", action="store_true")
    parser.add_argument("--weight-type", choices=("f32", "f16", "bf16"), default="f32")
    benchmark(parser.parse_args())


if __name__ == "__main__":
    main()
