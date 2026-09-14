"""Versioned Windows measured-job wrapper; frozen transfer harnesses remain unchanged."""
import argparse
import math
from pathlib import Path
import time

from benchmark_fashn_vton import measure_command
from fashn_artifacts import sha256, write_report


def run(output, command, timeout):
    if not command or not Path(command[0]).is_file() or not math.isfinite(timeout) or timeout < 1:
        raise ValueError("An explicit executable path and positive timeout are required")
    output.mkdir(parents=True, exist_ok=False)
    record = {"schema": "fashn-measured-job-v2", "status": "running", "command": command,
              "binary_sha256": sha256(command[0]), "started": time.time(), "timeout": timeout}
    write_report(output / "run.json", record)
    try:
        record["measurement"] = measure_command(command, output / "run.log", timeout,
                                                output / "memory.jsonl", output / "process.json")
        record["status"] = "completed" if record["measurement"]["exit_code"] == 0 else "failed"
    except (Exception, KeyboardInterrupt) as error:
        record.update(status="interrupted" if isinstance(error, KeyboardInterrupt) else "failed", error=str(error))
        raise
    finally:
        record["finished"] = time.time()
        write_report(output / "run.json", record)
    return record["measurement"]["exit_code"]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--timeout", type=float, default=14400)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    raise SystemExit(run(args.output, command, args.timeout))
