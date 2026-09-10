"""Identify the actual interpreter behind a Windows venv launcher before importing model dependencies."""
import json
import os
from pathlib import Path
import runpy
import sys
import ctypes
from ctypes import wintypes
import time


if __name__ == "__main__":
    destination = Path(os.environ["FASHN_MEASUREMENT_PID_FILE"])
    temporary = destination.with_suffix(".tmp")
    temporary.write_text(json.dumps({"pid": os.getpid(), "parent_pid": os.getppid(),
                                     "executable": sys.executable, "prefix": sys.prefix}), encoding="utf-8")
    temporary.replace(destination)
    acknowledgement = destination.with_suffix(".ack.json")
    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
    kernel.OpenProcess.restype = wintypes.HANDLE
    kernel.WaitForSingleObject.argtypes = [wintypes.HANDLE, wintypes.DWORD]
    kernel.WaitForSingleObject.restype = wintypes.DWORD
    kernel.CloseHandle.argtypes = [wintypes.HANDLE]
    parent = kernel.OpenProcess(0x100000, False, os.getppid())
    if not parent:
        raise ctypes.WinError(ctypes.get_last_error())
    try:
        deadline = time.monotonic() + 30
        # Never start model work if its measuring owner timed out during interpreter startup.
        while not acknowledgement.exists():
            if kernel.WaitForSingleObject(parent, 0) != 258 or time.monotonic() > deadline:
                raise RuntimeError("Measurement owner exited or did not acknowledge the worker")
            time.sleep(.01)
        if json.loads(acknowledgement.read_text(encoding="utf-8"))["pid"] != os.getpid():
            raise RuntimeError("Measurement acknowledged a different worker")
    finally:
        kernel.CloseHandle(parent)
    script = Path(sys.argv[1]).resolve()
    sys.argv = [str(script), *sys.argv[2:]]
    sys.path.insert(0, str(script.parent))
    runpy.run_path(str(script), run_name="__main__")
