"""Measure one explicit diagnostic command, without implying numerical or visual acceptance."""
import argparse
import json
from pathlib import Path

from benchmark_fashn_vton import measure_command
from export_fashn_vton_reference import sha256
from run_fashn_quality import write_report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    spec = json.loads(args.spec.read_text(encoding="utf-8"))
    command = spec["command"]
    if not isinstance(command, list) or not command or any(not isinstance(x, str) or not x for x in command):
        raise ValueError("Command must be a nonempty argument array")
    hashes = {path: sha256(Path(path)) for path in spec["provenance_files"]}
    if Path(command[0]).name.lower() in ("python.exe", "pythonw.exe"):
        bootstrap = Path(__file__).with_name("fashn_measurement_worker.py")
        hashes[str(bootstrap)] = sha256(bootstrap)
    args.output.mkdir(parents=True, exist_ok=False)
    report = {"execution_passed": False, "label": spec["label"], "provenance_sha256": hashes,
              "spec_sha256": sha256(args.spec), "measurement": "Fresh-process wall time; Windows kernel peak working set through last sample; no concurrent model experiments."}
    write_report(args.output / "measurement.json", report)
    timeline = args.output / "memory-timeline.jsonl" if spec.get("memory_timeline", False) else None
    report["run"] = measure_command(command, args.output / "run.log", spec.get("timeout", 14400), timeline)
    report["execution_passed"] = report["run"]["exit_code"] == 0
    write_report(args.output / "measurement.json", report)
    print(json.dumps(report, indent=2), flush=True)
    return 0 if report["execution_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
