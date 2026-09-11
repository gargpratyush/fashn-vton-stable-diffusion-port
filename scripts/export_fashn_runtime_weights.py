"""Export and verify runtime-policy weights using the existing native converter."""
import argparse
import json
from pathlib import Path
import re

from benchmark_fashn_vton import measure_command
from export_fashn_vton_reference import sha256
from run_fashn_quality import write_report

MATRIX_TYPES = ("bf16", "q8_0", "q4_0", "q5_0", "q4_K", "q5_K")


def conversion_rules(matrix_map, matrix_type, policy=None):
    types = matrix_map["matrix_types"]
    if len(types) != 104 or any(not re.fullmatch(r"[A-Za-z0-9_.]+[.]weight", name) or ".." in name
                                for name in types):
        raise ValueError("Expected exactly 104 safe native matrix names")
    if matrix_type not in MATRIX_TYPES:
        raise ValueError("Unsupported runtime export matrix type")
    restored = []
    if policy is not None:
        if set(policy) != {"f32_matrices"} or not isinstance(policy["f32_matrices"], list):
            raise ValueError("Expected an f32_matrices-only policy")
        restored = policy["f32_matrices"]
        if any(not isinstance(name, str) for name in restored) or len(set(restored)) != len(restored):
            raise ValueError("Invalid or duplicate restored matrices")
        if not set(restored).issubset(types):
            raise ValueError("Unknown or protected restored matrix")
    selected = sorted(set(types) - set(restored)) if matrix_type == "bf16" else sorted(restored)
    # Floating export otherwise affects protected tensors; quantized export preserves them.
    default_type, rule_type = ("f32", "bf16") if matrix_type == "bf16" else (matrix_type, "f32")
    rule = "(?:^|[.])(?:" + "|".join(re.escape(name) for name in selected) + ")$=" + rule_type
    return default_type, rule if selected else ""


def export(args):
    if args.threads < 1:
        raise ValueError("Loading threads must be positive")
    metadata = json.loads(args.matrix_map.read_text(encoding="utf-8"))
    policy = json.loads(args.f32_matrices.read_text(encoding="utf-8")) if args.f32_matrices else None
    default_type, rules = conversion_rules(metadata, args.matrix_type, policy)
    args.output.mkdir(parents=True, exist_ok=False)
    files = [args.cli, args.verifier, args.checkpoint, args.matrix_map, Path(__file__)]
    if args.f32_matrices:
        files.append(args.f32_matrices)
    report = {"complete": False, "diagnostic_only": True, "matrix_type": args.matrix_type,
              "policy": policy, "provenance_sha256": {str(path): sha256(path) for path in files}}
    report_path = args.output / "export.json"
    write_report(report_path, report)
    destination = args.reuse_file or (args.output / "model.gguf")
    if args.reuse_file:
        if not destination.is_file():
            raise ValueError("Reused GGUF does not exist")
        report["reused_file"] = str(destination)
    else:
        command = [str(args.cli), "--mode", "convert", "--diffusion-model", str(args.checkpoint),
                   "--type", default_type, "-t", str(args.threads), "-o", str(destination)]
        if rules:
            command += ["--tensor-type-rules", rules]
        report["conversion"] = measure_command(command, args.output / "conversion.log")
        write_report(report_path, report)
        if report["conversion"]["exit_code"] != 0:
            raise RuntimeError("Conversion failed; export remains incomplete")
    command = [str(args.verifier), str(args.checkpoint), str(destination), args.matrix_type, "--runtime-policy"]
    if args.f32_matrices:
        command += ["--f32-matrices", str(args.f32_matrices)]
    report["verification"] = measure_command(command, args.output / "verification.log")
    write_report(report_path, report)
    if report["verification"]["exit_code"] != 0:
        raise RuntimeError("Runtime policy/value verification failed; export remains incomplete")
    report["output"] = str(destination)
    report["output_sha256"] = sha256(destination)
    report["output_bytes"] = destination.stat().st_size
    report["complete"] = True
    write_report(report_path, report)
    print(json.dumps(report, indent=2), flush=True)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("cli", "verifier", "checkpoint", "matrix-map", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--matrix-type", choices=MATRIX_TYPES, required=True)
    parser.add_argument("--f32-matrices", type=Path)
    parser.add_argument("--reuse-file", type=Path)
    parser.add_argument("--threads", type=int, default=2)
    export(parser.parse_args())
