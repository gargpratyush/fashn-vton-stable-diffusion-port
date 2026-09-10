"""Export an immutable sensitivity-study snapshot, storage costs and unvalidated small-policy candidates."""
import argparse
import csv
import hashlib
import io
import json
import math
from pathlib import Path

from export_fashn_vton_reference import expected_shapes, sha256
from run_fashn_matrix_ablation import ranking
from run_fashn_quality import write_report


def restoration_bytes(name):
    shape = expected_shapes().get(name)
    if shape is None or len(shape) != 2 or shape[-1] % 32:
        raise ValueError("Expected an aligned model matrix")
    elements = math.prod(shape)
    return elements * 4 - (elements // 32) * 34


def positive_candidates(rows, count):
    names = [row["matrix"] for row in rows if row["relative_score_reduction"] > 0]
    return names[:count] if len(names) >= count else []


def summarize(study, output):
    payload = study.read_bytes()
    report = json.loads(payload)
    if not report["baseline"] or not report["candidates"]:
        raise ValueError("Study has no completed interventions")
    for result in [report["baseline"], *report["candidates"].values()]:
        for name, digest in result["artifacts_sha256"].items():
            if sha256(Path(result["directory"]) / name) != digest:
                raise ValueError("Completed study artifact changed")
    rows = ranking(report["baseline"]["summary"]["score"], report["candidates"])
    output.mkdir(parents=True, exist_ok=False)
    (output / "study-snapshot.json").write_bytes(payload)
    table = []
    for index, item in enumerate(rows, 1):
        result = report["candidates"][item["matrix"]]
        summary = result["summary"]
        additional = restoration_bytes(item["matrix"])
        improvement = item["relative_score_reduction"] * 100
        table.append({
            "rank": index, "matrix": item["matrix"], "score": item["score"],
            "error_reduction_percent": improvement,
            "conditional_velocity_l2": summary["velocity_relative_l2"]["conditional"],
            "null_velocity_l2": summary["velocity_relative_l2"]["unconditional"],
            "failed_captures": summary["failed_captures"], "added_weight_bytes": additional,
            "added_weight_mib": additional / 1024**2,
            "reduction_percent_per_added_mib": improvement / (additional / 1024**2),
            "probe_wall_seconds": result["measurement"]["wall_seconds"],
        })
    text = io.StringIO(newline="")
    writer = csv.DictWriter(text, fieldnames=list(table[0]))
    writer.writeheader()
    writer.writerows(table)
    (output / "ranking.csv").write_text(text.getvalue(), encoding="utf-8", newline="")
    complete = report["complete"] and len(table) == len(report["fingerprint"]["matrices"]) == 104
    proposals = {}
    if complete:
        for count in (1, 2, 4):
            names = positive_candidates(rows, count)
            if names:
                name = f"candidate-top{count}.json"
                write_report(output / name, {"f32_matrices": names})
                proposals[name] = {"evaluated_as_combination": False, "matrices": names,
                                   "added_weight_bytes": sum(restoration_bytes(matrix) for matrix in names)}
    metadata = {
        "study_snapshot_sha256": hashlib.sha256(payload).hexdigest(), "complete": complete,
        "completed_matrices": len(table), "total_matrices": len(report["fingerprint"]["matrices"]),
        "baseline_score": report["baseline"]["summary"]["score"], "candidate_policies": proposals,
        "storage_formula": "F32 minus Q8_0 payload: 4*N - 34*(N/32) bytes. Excludes alignment, metadata, graphs, activations and mapped files.",
        "scope": "Single-matrix restoration effects on one fixed probe. Candidate combinations are unvalidated; held-out and full-image results required.",
    }
    write_report(output / "summary.json", metadata)
    lines = ["# Matrix restoration sensitivity", "",
             f"Completed {len(table)}/{metadata['total_matrices']} matrices; baseline score {metadata['baseline_score']:.10g}.",
             "", metadata["scope"], "", metadata["storage_formula"], "",
             "| Rank | Restored matrix | Worst-branch velocity L2 | Error reduction | Added weight MiB |",
             "|---|---|---:|---:|---:|"]
    for row in table:
        lines.append(f"| {row['rank']} | `{row['matrix']}` | {row['score']:.10g} | "
                     f"{row['error_reduction_percent']:+.5f}% | {row['added_weight_mib']:.3f} |")
    (output / "ranking.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(metadata, indent=2))
    return metadata


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    summarize(args.study, args.output)
