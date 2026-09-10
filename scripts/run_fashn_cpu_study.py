"""Serial, isolated CPU thread/kernel comparisons with one-step pixel regression checks."""
import argparse
import json
from pathlib import Path
import statistics
from types import SimpleNamespace

import numpy as np
from PIL import Image

from benchmark_fashn_vton import benchmark
from run_fashn_quality import write_report


def pixel_difference(expected, actual):
    with Image.open(expected) as a, Image.open(actual) as b:
        if a.mode != "RGB" or b.mode != "RGB" or a.size != b.size:
            raise ValueError("Benchmark PNG shape/mode mismatch")
        delta = np.abs(np.array(a).astype(np.int16) - np.array(b).astype(np.int16))
    return {"changed_channels": int(np.count_nonzero(delta)), "max_abs": int(delta.max()),
            "mae": float(delta.mean()), "passed": bool(delta.max() <= 1)}


def run(args):
    if args.repeats < 2:
        raise ValueError("Use at least two repeats per configuration")
    args.output.mkdir(parents=True, exist_ok=False)
    report = {"passed": False, "configurations": {},
              "measurement": "Serial fresh processes, not cold filesystem caches. Exploratory repeats, not confidence intervals. No concurrent model experiments allowed.",
              "pixel_gate": "At most one byte level from the supplied matching one-step CFG-1/seed-42 PNG; full model gates remain separate."}
    write_report(args.output / "study.json", report)
    configurations = [("baseline-t8", args.baseline_cli, 8), ("candidate-t8", args.candidate_cli, 8),
                      ("baseline-t4", args.baseline_cli, 4), ("baseline-t16", args.baseline_cli, 16)]
    for name, executable, threads in configurations:
        output = args.output / name
        benchmark(SimpleNamespace(cli=executable, checkpoint=args.checkpoint, prepared=args.prepared,
                                  output=output, steps=1, cfg=1., shift=1.5, skip_cfg_last_n_steps=1,
                                  seed=42, threads=threads, repeats=args.repeats, timeout=1800,
                                  flash_attention=True, weight_type="bf16"))
        measured = json.loads((output / "benchmark.json").read_text(encoding="utf-8"))
        row = {"median_seconds": statistics.median(value["wall_seconds"] for value in measured["runs"]),
               "wall_seconds": [value["wall_seconds"] for value in measured["runs"]],
               "peak_working_set_bytes": [value["peak_working_set_bytes"] for value in measured["runs"]],
               "executable_sha256": measured["executable_sha256"],
               "pixels": [pixel_difference(args.expected, output / f"sample-{i}.png") for i in range(args.repeats)]}
        report["configurations"][name] = row
        write_report(args.output / "study.json", report)
        if not all(image["passed"] for image in row["pixels"]):
            raise RuntimeError("Candidate failed the predeclared one-step pixel check: " + name)
        print(name + ": " + str(row["median_seconds"]) + " seconds median", flush=True)
    report["passed"] = True
    write_report(args.output / "study.json", report)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("baseline-cli", "candidate-cli", "checkpoint", "prepared", "expected", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=2)
    args = parser.parse_args()
    for name, value in vars(args).items():
        if isinstance(value, Path):
            setattr(args, name, value.resolve())
    run(args)
