"""Evaluate small mixed-precision policies at fixed photographic states; no automatic public promotion."""
import argparse
from argparse import Namespace
import json
from pathlib import Path
import re

from export_fashn_vton_reference import sha256
from run_fashn_matrix_ablation import run_probe
from run_fashn_quality import write_report


def labeled_path(text):
    label, separator, value = text.partition("=")
    if not separator or not re.fullmatch(r"[a-zA-Z0-9_-]+", label) or not value:
        raise argparse.ArgumentTypeError("Expected safe-label=path")
    return label, Path(value).resolve()


def policy_names(path, eligible):
    policy = json.loads(path.read_text(encoding="utf-8"))
    if set(policy) != {"f32_matrices"} or not isinstance(policy["f32_matrices"], list):
        raise ValueError("Expected only f32_matrices array")
    names = policy["f32_matrices"]
    if any(not isinstance(name, str) or name not in eligible for name in names) or len(names) != len(set(names)):
        raise ValueError("Invalid, ineligible or duplicate matrix names")
    return names


def evaluate(args):
    if args.threads < 1 or args.timeout < 1:
        raise ValueError("Invalid execution limits")
    metadata = json.loads(args.matrix_map.read_text(encoding="utf-8"))
    types = metadata["matrix_types"]
    if len(types) != 104 or set(types.values()) != {"q8_0"}:
        raise ValueError("Use the completed all-104-Q8 matrix map")
    fixtures, policies = dict(args.fixture), dict(args.policy)
    if len(fixtures) != len(args.fixture) or len(policies) != len(args.policy):
        raise ValueError("Duplicate fixture or policy labels")
    if "all-q8" in policies or "weight-only-q8" in policies:
        raise ValueError("Reserved control policy label")
    names = {label: policy_names(path, types) for label, path in policies.items()}
    for path in fixtures.values():
        manifest = json.loads((path / "manifest.json").read_text(encoding="utf-8"))
        if manifest.get("input_kind") != "prepared":
            raise ValueError("Use photographic fixtures")
    args.output.mkdir(parents=True, exist_ok=False)
    report = {
        "complete": False, "graph_sha256": sha256(args.graph), "checkpoint_sha256": sha256(args.checkpoint),
        "matrix_map_sha256": sha256(args.matrix_map), "threads": args.threads,
        "policies": {label: {"sha256": sha256(policies[label]), "matrices": value} for label, value in names.items()},
        "fixtures": {label: {name: sha256(path / name) for name in
                             ("manifest.json", "inputs.safetensors", "conditional.safetensors", "unconditional.safetensors")}
                     for label, path in fixtures.items()},
        "results": {}, "scope": "Fixed-state probes; combined policies and weight-only control. Full free-running images still required."
    }
    write_report(args.output / "evaluation.json", report)
    for label, fixture in fixtures.items():
        directory = args.output / label
        directory.mkdir()
        probe = Namespace(graph=args.graph, checkpoint=args.checkpoint, fixtures=fixture,
                          output=directory, threads=args.threads, timeout=args.timeout)
        report["results"][label] = {}
        for policy, restored in {"all-q8": [], **names}.items():
            print(f"Evaluating {label}: {policy}", flush=True)
            row = run_probe(probe, policy, restored, types)
            report["results"][label][policy] = row
            write_report(args.output / "evaluation.json", report)
            print(json.dumps({"fixture": label, "policy": policy, **row["summary"]}), flush=True)
        if args.weight_only_control:
            row = run_probe(probe, "weight-only-q8", [], types, upcast=True)
            report["results"][label]["weight-only-q8"] = row
            write_report(args.output / "evaluation.json", report)
            print(json.dumps({"fixture": label, "policy": "weight-only-q8", **row["summary"]}), flush=True)
    report["complete"] = True
    write_report(args.output / "evaluation.json", report)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("graph", "checkpoint", "matrix-map", "output"):
        parser.add_argument("--" + name, type=lambda value: Path(value).resolve(), required=True)
    parser.add_argument("--fixture", type=labeled_path, action="append", required=True)
    parser.add_argument("--policy", type=labeled_path, action="append", required=True)
    parser.add_argument("--threads", type=int, default=16)
    parser.add_argument("--timeout", type=float, default=1800)
    parser.add_argument("--weight-only-control", action="store_true")
    evaluate(parser.parse_args())
