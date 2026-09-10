"""Compare complete native/reference FASHN trajectories without relaxing layer gates."""
import argparse
import json
import math
from pathlib import Path

import numpy as np
from PIL import Image
from safetensors.numpy import load_file

from export_fashn_vton_reference import sha256


def canonical(array):
    while array.ndim > 3 and array.shape[0] == 1:
        array = array[0]
    return array


def metrics(expected, actual):
    expected, actual = canonical(expected), canonical(actual)
    if expected.shape != actual.shape or expected.dtype != np.float32 or actual.dtype != np.float32:
        raise ValueError("Trajectory tensor shape/dtype mismatch")
    if not np.isfinite(expected).all() or not np.isfinite(actual).all():
        raise ValueError("Non-finite trajectory tensor")
    delta = actual.astype(np.float64) - expected
    relative = float(np.linalg.norm(delta.ravel()) / max(np.linalg.norm(expected.astype(np.float64).ravel()), 1e-12))
    return {"relative_l2": relative, "max_abs": float(np.max(np.abs(delta))),
            "rmse": float(np.sqrt(np.mean(delta * delta))),
            "outside_pointwise_1e4": int(np.count_nonzero(np.abs(delta) > 1e-4 + 1e-4 * np.abs(expected))),
            "passed": relative <= 1e-3}


def pixels(array):
    array = canonical(array)
    if array.ndim != 3 or array.shape[0] != 3:
        raise ValueError("Expected CHW RGB image")
    return (np.clip((array + np.float32(1)) * np.float32(.5), 0, 1) * np.float32(255)).astype(np.uint8).transpose(1, 2, 0)


def compare(reference, native, output):
    reference, native, output = Path(reference), Path(native), Path(output)
    rm = json.loads((reference / "manifest.json").read_text(encoding="utf-8"))
    nm = json.loads((native / "manifest.json").read_text(encoding="utf-8"))
    if type(rm.get("steps")) is not int or not 1 <= rm["steps"] <= 1000:
        raise ValueError("Invalid reference step count")
    categories = {"tops": 1, "bottoms": 2, "one-pieces": 3}
    category = categories.get(rm["category"], rm["category"])
    if category not in (1, 2, 3) or category != nm["category"]:
        raise ValueError("Category mismatch")
    for key in ("steps", "cfg", "skip_cfg_last_n_steps", "seed"):
        if rm[key] != nm[key]:
            raise ValueError("Sampling mismatch: " + key)
    if len(rm["schedule"]) != rm["steps"] + 1 or len(nm["schedule"]) != nm["steps"] + 1:
        raise ValueError("Incomplete schedule")
    for manifest in (rm, nm):
        schedule = np.asarray(manifest["schedule"], dtype=np.float64)
        if not np.isfinite(schedule).all() or schedule[0] != 0 or schedule[-1] != 1 or np.any(np.diff(schedule) <= 0):
            raise ValueError("Invalid schedule")
    schedule_error = float(np.max(np.abs(np.array(rm["schedule"]) - nm["schedule"])))
    if not math.isfinite(schedule_error) or schedule_error > 1e-7:
        raise ValueError("Schedule mismatch")
    required = {"initial.safetensors", "expected.png"} | {f"step-{i}.safetensors" for i in range(rm["steps"])}
    if not required.issubset(rm.get("artifacts", {})):
        raise ValueError("Reference manifest is incomplete")
    for name in sorted(required):
        if sha256(reference / name) != rm["artifacts"][name]:
            raise ValueError("Reference artifact hash mismatch: " + name)
    output.mkdir(parents=True, exist_ok=False)
    report = {"passed": False, "schedule_max_abs": schedule_error, "steps": [],
              "reference_manifest_sha256": sha256(reference / "manifest.json"),
              "native_manifest_sha256": sha256(native / "manifest.json"),
              "gate": "relative L2 <= 0.001 at every updated image and guided velocity; initial max absolute <= 1e-6",
              "note": "Pointwise errors and image PSNR are diagnostics, not perceptual try-on quality scores."}
    initial = metrics(load_file(str(reference / "initial.safetensors"))["noise"],
                      load_file(str(native / "initial.safetensors"))["noise"])
    initial["passed"] = initial["max_abs"] <= 1e-6
    report["initial_noise"] = initial
    passed = initial["passed"]
    for index in range(rm["steps"]):
        name = f"step-{index}.safetensors"
        ref = load_file(str(reference / name))
        actual = load_file(str(native / name))
        if set(ref) != {"image", "guided_velocity"} or set(actual) != set(ref):
            raise ValueError("Incomplete trajectory state")
        step = {"index": index, "native_sha256": sha256(native / name)}
        for key in ("image", "guided_velocity"):
            step[key] = metrics(ref[key], actual[key])
            passed = passed and step[key]["passed"]
        report["steps"].append(step)
        (output / "comparison.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    native_pixels = pixels(actual["image"])
    expected_pixels = pixels(ref["image"])
    with Image.open(reference / "expected.png") as image:
        if image.mode != "RGB" or not np.array_equal(np.array(image), expected_pixels):
            raise ValueError("Reference final PNG does not match saved trajectory")
    Image.fromarray(native_pixels).save(output / "native.png")
    Image.fromarray(expected_pixels).save(output / "reference.png")
    delta = native_pixels.astype(np.float64) - expected_pixels
    mse = float(np.mean(delta * delta))
    report["pixels"] = {"changed_channels": int(np.count_nonzero(delta)), "max_abs": float(np.max(np.abs(delta))),
                        "mae": float(np.mean(np.abs(delta))), "psnr_db": 10 * math.log10(255**2 / mse) if mse else None,
                        "exact": mse == 0}
    Image.fromarray(np.clip(np.abs(delta) * 32, 0, 255).astype(np.uint8)).save(output / "difference-x32.png")
    report["passed"] = bool(passed)
    (output / "comparison.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("reference", "native", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    result = compare(args.reference, args.native, args.output)
    print(json.dumps({"passed": result["passed"], "pixels": result["pixels"]}, indent=2))
    raise SystemExit(0 if result["passed"] else 1)
