"""Persistent Windows process identity; no PID-only probing or process-name termination."""
import ctypes
from ctypes import wintypes
import os
import platform
from pathlib import Path

from fashn_artifacts import write_report


def process_identity(pid):
    if os.name != "nt":
        raise RuntimeError("Windows process identity required by this measurement harness")
    if type(pid) is not int or pid <= 0:
        raise ValueError("Invalid process PID")
    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
    kernel.OpenProcess.restype = wintypes.HANDLE
    kernel.CloseHandle.argtypes = [wintypes.HANDLE]
    kernel.GetProcessTimes.argtypes = [wintypes.HANDLE, *([ctypes.POINTER(wintypes.FILETIME)] * 4)]
    kernel.WaitForSingleObject.argtypes = [wintypes.HANDLE, wintypes.DWORD]
    kernel.WaitForSingleObject.restype = wintypes.DWORD
    handle = kernel.OpenProcess(0x100000 | 0x1000, False, pid)
    if not handle:
        error = ctypes.get_last_error()
        if error == 87:
            return None
        raise ctypes.WinError(error)
    try:
        status = kernel.WaitForSingleObject(handle, 0)
        if status == 0:
            return None
        if status != 258:
            raise ctypes.WinError(ctypes.get_last_error())
        times = [wintypes.FILETIME() for _ in range(4)]
        if not kernel.GetProcessTimes(handle, *[ctypes.byref(value) for value in times]):
            raise ctypes.WinError(ctypes.get_last_error())
        created = times[0].dwLowDateTime | (times[0].dwHighDateTime << 32)
        return {"pid": pid, "created": created, "host": platform.node()}
    finally:
        kernel.CloseHandle(handle)


def identity_active(identity):
    if identity["host"] != platform.node():
        raise ValueError("Process liveness must be checked on the original measurement host")
    return process_identity(identity["pid"]) == identity


def record_process(path, phase, owner, child=None, worker=None):
    write_report(Path(path), {"schema": "fashn-process-v1", "phase": phase,
                              "owner": owner, "child": child, "worker": worker})
