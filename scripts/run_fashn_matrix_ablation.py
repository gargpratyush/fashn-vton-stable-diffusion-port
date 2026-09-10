"""Serial, resumable one-matrix-at-a-time F32 restoration study against a fixed photographic oracle."""
import argparse
import json
import math
from pathlib import Path
import time

from benchmark_fashn_vton import measure_command
from export_fashn_vton_reference import sha256
from run_fashn_quality import write_report


def summarize(result, names, restored):
    if result.get("error") or result.get("matrix_type") != "q8_0" or result.get("flash_attention_nodes") != 28:
        raise ValueError("Incomplete or unexpected graph diagnostic")
    expected = {name: "f32" if name in restored else "q8_0" for name in names}
    if result.get("matrix_types") != expected:
        raise ValueError("Actual matrix precision does not match the requested intervention")
    rows = []
    velocity = {}
    for branch in ("conditional", "unconditional"):
        captures = result.get(branch, {})
        if len(captures) != 17 or "velocity" not in captures:
            raise ValueError("Incomplete branch captures")
        for entry in captures.values():
            if not math.isfinite(entry["relative_l2"]) or entry["relative_l2"] < 0:
                raise ValueError("Invalid capture metric")
        rows.extend(captures.values())
        velocity[branch] = captures["velocity"]["relative_l2"]
    return {"score": max(velocity.values()), "velocity_relative_l2": velocity,
            "max_capture_relative_l2": max(row["relative_l2"] for row in rows),
            "failed_captures": sum(not row["passed"] for row in rows),
            "numerical_gate_passed": result["passed"]}


def ranking(baseline, candidates):
    return [{"matrix": name, "score": row["summary"]["score"],
             "relative_score_reduction": (baseline - row["summary"]["score"]) / max(baseline, 1e-30)}
            for name, row in sorted(candidates.items(), key=lambda item: item[1]["summary"]["score"])]


def run_probe(args, name, restored, names, upcast=False):
    directory = args.output / name
    if directory.exists():
        directory.rename(directory.with_name(name + "-interrupted-" + str(time.time_ns())))
    directory.mkdir()
    policy = directory / "policy.json"
    write_report(policy, {"f32_matrices": restored})
    command = [str(args.graph), str(args.checkpoint), str(args.fixtures), str(directory),
               "--flash-attention", "--matrix-type", "q8_0", "--threads", str(args.threads),
               "--no-capture-files", "--f32-matrices", str(policy)]
    if upcast:
        command.append("--upcast-matrices")
    measured = measure_command(command, directory / "run.log", args.timeout)
    write_report(directory / "measurement.json", measured)
    # Exit 1 can mean a completed numerical rejection, not a crashed/incomplete inference.
    if measured["exit_code"] not in (0, 1) or not (directory / "comparison.json").is_file():
        raise RuntimeError("Graph diagnostic failed: " + str(directory))
    result = json.loads((directory / "comparison.json").read_text(encoding="utf-8"))
    if result.get("upcast_matrices") != upcast:
        raise ValueError("Unexpected matrix computation mode")
    summary = summarize(result, names, restored)
    if (measured["exit_code"] == 0) != summary["numerical_gate_passed"]:
        raise ValueError("Graph exit status and numerical result disagree")
    return {"summary": summary, "measurement": measured, "directory": str(directory), "upcast_matrices": upcast,
            "artifacts_sha256": {name: sha256(directory / name) for name in
                                ("comparison.json", "policy.json", "measurement.json")}}


def study(args):
    if args.threads < 1 or args.max_new_matrices < 0 or args.timeout < 1:
        raise ValueError("Invalid study limits")
    fixture_manifest = json.loads((args.fixtures / "manifest.json").read_text(encoding="utf-8"))
    if fixture_manifest.get("input_kind") != "prepared":
        raise ValueError("Calibrate on photographic prepared conditions, not the old synthetic stress fixture")
    metadata = json.loads(args.matrix_map.read_text(encoding="utf-8"))
    types = metadata["matrix_types"]
    if len(types) != 104 or set(types.values()) != {"q8_0"}:
        raise ValueError("Expected a completed all-104-Q8 diagnostic matrix map")
    names = sorted(types, key=lambda name: (not name.startswith("x_patch_mixer."), name))
    if any(not name.endswith(".weight") or "/" in name or "\\" in name or ".." in name for name in names):
        raise ValueError("Unsafe matrix names")
    fingerprint = {
        "graph": sha256(args.graph), "checkpoint": sha256(args.checkpoint), "matrix_map": sha256(args.matrix_map),
        "fixtures": {name: sha256(args.fixtures / name) for name in
                     ("manifest.json", "inputs.safetensors", "conditional.safetensors", "unconditional.safetensors")},
        "threads": args.threads, "matrices": names,
        "ranking_objective": "Minimum maximum conditional/null final-velocity relative L2 at this fixed probe",
    }
    path = args.output / "study.json"
    if args.resume:
        report = json.loads(path.read_text(encoding="utf-8"))
        if report["fingerprint"] != fingerprint:
            raise ValueError("Changed study executable, model, inputs, settings or matrix map")
        for row in [report["baseline"], *report["candidates"].values()]:
            if row:
                for name, digest in row["artifacts_sha256"].items():
                    if sha256(Path(row["directory"]) / name) != digest:
                        raise ValueError("Completed diagnostic was changed")
    else:
        args.output.mkdir(parents=True, exist_ok=False)
        report = {"fingerprint": fingerprint, "baseline": None, "candidates": {}, "complete": False,
                  "note": "Conditional single-matrix restoration effects, not independent causal importance. Interactions and held-out trajectories require separate evaluation."}
        write_report(path, report)
    if report["baseline"] is None:
        report["baseline"] = run_probe(args, "baseline", [], names)
        write_report(path, report)
    count = 0
    for index, name in enumerate(names):
        if name in report["candidates"]:
            continue
        print("Restoring only " + name, flush=True)
        row = run_probe(args, f"matrix-{index:03d}", [name], names)
        report["candidates"][name] = row
        report["ranking"] = ranking(report["baseline"]["summary"]["score"], report["candidates"])
        report["complete"] = len(report["candidates"]) == len(names)
        write_report(path, report)
        print(json.dumps({"matrix": name, **row["summary"]}), flush=True)
        count += 1
        if args.max_new_matrices and count >= args.max_new_matrices:
            break
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("graph", "checkpoint", "fixtures", "matrix-map", "output"):
        parser.add_argument("--" + name, type=lambda text: Path(text).resolve(), required=True)
    parser.add_argument("--threads", type=int, default=16)
    parser.add_argument("--max-new-matrices", type=int, default=0)
    parser.add_argument("--timeout", type=float, default=1800)
    parser.add_argument("--resume", action="store_true")
    study(parser.parse_args())
