"""Compare original upstream sampler and native CLI outputs with matching input/settings provenance."""
import argparse
import json
from pathlib import Path
import shutil

import numpy as np
from PIL import Image

from build_fashn_gallery import pixel_metrics
from export_fashn_vton_reference import SOURCE_REVISION, sha256
from run_fashn_quality import write_report


def compare(upstream, native, output, measurement=None):
    reference = json.loads((upstream / "result.json").read_text(encoding="utf-8"))
    baseline = json.loads((native / "benchmark.json").read_text(encoding="utf-8"))
    if reference["source_revision"] != SOURCE_REVISION or reference["dtype"] != "F32":
        raise ValueError("Unpinned or unexpected upstream arithmetic")
    if baseline["passed"] is not True or len(baseline["runs"]) != 1 or baseline["runs"][0]["exit_code"] != 0:
        raise ValueError("Incomplete native benchmark")
    for ref_key, native_key in (("checkpoint_sha256", "checkpoint_sha256"),
                                ("prepared_manifest_sha256", "manifest_sha256"),
                                ("prepared_images_sha256", "inputs_sha256")):
        if reference[ref_key] != baseline[native_key]:
            raise ValueError("Input/checkpoint lineage mismatch: " + ref_key)
    for key in ("steps", "cfg", "shift", "skip_cfg_last_n_steps", "seed", "threads"):
        if reference["settings"][key] != baseline["settings"][key]:
            raise ValueError("Sampling/thread mismatch: " + key)
    if sha256(upstream / "sample-0.png") != reference["output_sha256"]:
        raise ValueError("Upstream output changed")
    if sha256(native / "sample-0.png") != baseline["runs"][0]["output_sha256"]:
        raise ValueError("Native output changed")
    with Image.open(upstream / "sample-0.png") as a, Image.open(native / "sample-0.png") as b:
        expected_size = (reference["crop"]["width"], reference["crop"]["height"])
        if a.mode != "RGB" or b.mode != "RGB" or a.size != expected_size or b.size != expected_size:
            raise ValueError("Output mode/crop mismatch")
        metrics = pixel_metrics(np.array(a), np.array(b))
    report = {
        "input_settings_lineage_matched": True, "category": reference["category"], "settings": reference["settings"],
        "pixels": metrics, "within_one_byte": metrics["max_abs"] <= 1,
        "reference_result_sha256": sha256(upstream / "result.json"),
        "native_benchmark_sha256": sha256(native / "benchmark.json"),
        "upstream_sampling_and_pil_seconds": reference["sampling_and_pil_seconds"],
        "historical_native_cli_wall_seconds": baseline["runs"][0]["wall_seconds"],
        "scope": "Original upstream prepared sampler, default attention/batched CFG, versus native CLI. Pixel metrics are not garment-quality scores or intermediate-float parity. Historical native timing is not a fresh interleaved control.",
    }
    if measurement is not None:
        measured = json.loads(measurement.read_text(encoding="utf-8"))
        if not measured["execution_passed"] or measured["run"]["exit_code"] != 0:
            raise ValueError("Incomplete upstream process measurement")
        command = measured["run"]["command"]
        if (command.count("--output") != 1 or command.index("--output") + 1 >= len(command)
                or Path(command[command.index("--output") + 1]).resolve() != upstream.resolve()):
            raise ValueError("Measurement belongs to a different upstream output")
        if measured["run"].get("memory_scope") == "active_python_interpreter":
            report["upstream_process"] = measured["run"]
        else:
            report["upstream_process"] = {key: measured["run"][key] for key in ("command", "exit_code", "wall_seconds")}
            report["memory_warning"] = "Original measurement did not identify the active Python interpreter; launcher memory is not model memory and is excluded."
        report["measurement_sha256"] = sha256(measurement)
    output.mkdir(parents=True, exist_ok=False)
    shutil.copyfile(upstream / "sample-0.png", output / "reference.png")
    shutil.copyfile(native / "sample-0.png", output / "native.png")
    write_report(output / "comparison.json", report)
    print(json.dumps(report, indent=2))
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("upstream", "native", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--measurement", type=Path)
    args = parser.parse_args()
    compare(args.upstream, args.native, args.output, args.measurement)
