"""Small standard-library artifact primitives; importing this module never starts a job."""
from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import tempfile


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def atomic_text(path, text):
    path = Path(path)
    descriptor, temporary = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as stream:
            stream.write(text)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def write_report(path, value):
    atomic_text(path, json.dumps(value, indent=2, allow_nan=False) + "\n")


def contained(root, name):
    root = Path(root).resolve()
    path = root / name
    if Path(name).is_absolute() or not path.resolve().is_relative_to(root) or path.resolve() == root:
        raise ValueError("Artifact path escapes its root: " + name)
    return path


@contextmanager
def exclusive_lock(path):
    with Path(path).open("a+b") as lock:
        if lock.tell() == 0:
            lock.write(b"\0")
            lock.flush()
        lock.seek(0)
        if os.name == "nt":
            import msvcrt
            msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            yield
        finally:
            lock.seek(0)
            if os.name == "nt":
                msvcrt.locking(lock.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(lock, fcntl.LOCK_UN)


def portable(value, roots):
    if isinstance(value, dict):
        return {portable(key, roots): portable(item, roots) for key, item in value.items()}
    if isinstance(value, list):
        return [portable(item, roots) for item in value]
    if isinstance(value, str):
        for path, label in roots:
            value = value.replace(str(path), label).replace(path.as_posix(), label)
    return value
