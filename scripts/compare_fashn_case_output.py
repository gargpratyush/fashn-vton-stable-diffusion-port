"""Compare a native quality-case PNG with matched reference inference, including crop and input lineage."""
import argparse
import json
import math
from pathlib import Path

import numpy as np
from PIL import Image
from safetensors.numpy import load_file

from compare_fashn_trajectories import pixels
from export_fashn_vton_reference import SOURCE_REVISION, load_reference, sha256
from prepare_fashn_vton import normalized_condition
from run_fashn_quality import write_report


def crop_pixels(image, crop):
    names = ("x", "y", "width", "height")
    if set(crop) != set(names) or any(type(crop[key]) is not int for key in names):
        raise ValueError("Invalid crop fields")
    x, y, width, height = (crop[key] for key in names)
    if min(x, y) < 0 or min(width, height) < 1 or x + width > image.shape[1] or y + height > image.shape[0]:
        raise ValueError("Crop outside reference canvas")
    return image[y:y + height, x:x + width]


def compare_case(args):
    reference = json.loads((args.reference / "manifest.json").read_text(encoding="utf-8"))
    benchmark = json.loads((args.native_run / "benchmark.json").read_text(encoding="utf-8"))
    prepared = json.loads(args.prepared.read_text(encoding="utf-8"))
    if (reference["source_revision"] != SOURCE_REVISION or reference["input_kind"] != "prepared"
            or reference["dtype"] != "F32" or benchmark["passed"] is not True or len(benchmark["runs"]) != 1):
        raise ValueError("Incomplete or unpinned reference/native run")
    if reference["checkpoint_sha256"] != benchmark["checkpoint_sha256"]:
        raise ValueError("Checkpoint mismatch")
    if reference["inputs_sha256"] != sha256(args.reference_inputs / "conditions.safetensors"):
        raise ValueError("Reference conditions changed")
    if benchmark["manifest_sha256"] != sha256(args.prepared) or reference["category"] != prepared["category"]:
        raise ValueError("Prepared manifest/category mismatch")
    for name in ("steps", "cfg", "skip_cfg_last_n_steps", "seed"):
        if reference[name] != benchmark["settings"][name]:
            raise ValueError("Sampling mismatch: " + name)
    _, sampling, tensor_utils = load_reference(args.source)
    expected_schedule = sampling.get_rf_schedule(reference["steps"], mu=benchmark["settings"]["shift"])
    if len(expected_schedule) != len(reference["schedule"]) or not np.allclose(
            expected_schedule, reference["schedule"], atol=1e-7, rtol=0):
        raise ValueError("Reference schedule/shift mismatch")
    conditions = load_file(str(args.reference_inputs / "conditions.safetensors"))
    names = {"ca_image": "ca_images", "garment_image": "garment_images",
             "person_pose": "person_poses", "garment_pose": "garment_poses"}
    if set(conditions) != set(names.values()):
        raise ValueError("Incomplete reference conditions")
    input_hashes = {}
    for key, plural in names.items():
        native_path = args.prepared.parent / prepared[key]
        digest = sha256(native_path)
        if benchmark["inputs_sha256"][key] != digest:
            raise ValueError("Native prepared input changed: " + key)
        with Image.open(native_path) as a, Image.open(args.reference_inputs / (key + ".png")) as b:
            if a.mode != b.mode or a.size != b.size or a.tobytes() != b.tobytes():
                raise ValueError("Prepared native/reference pixels differ: " + key)
            restored = normalized_condition(np.array(a), tensor_utils).numpy()
        if (conditions[plural].dtype != np.float32 or restored.shape != conditions[plural].shape
                or not np.array_equal(restored, conditions[plural])):
            raise ValueError("Reference conditions do not match the prepared PNG: " + key)
        input_hashes[key] = digest
    final_name = f"step-{reference['steps'] - 1}.safetensors"
    for name in ("expected.png", final_name):
        if sha256(args.reference / name) != reference["artifacts"][name]:
            raise ValueError("Reference output hash mismatch")
    final_tensor = load_file(str(args.reference / final_name))["image"]
    if final_tensor.dtype != np.float32 or not np.isfinite(final_tensor).all():
        raise ValueError("Reference final tensor must be finite F32")
    full_reference = pixels(final_tensor)
    with Image.open(args.reference / "expected.png") as image:
        if image.mode != "RGB" or not np.array_equal(np.array(image), full_reference):
            raise ValueError("Reference PNG does not match its final tensor")
    native_path = args.native_run / "sample-0.png"
    if benchmark["runs"][0]["exit_code"] != 0 or sha256(native_path) != benchmark["runs"][0]["output_sha256"]:
        raise ValueError("Native output changed or failed")
    crop = prepared.get("crop", {"x": 0, "y": 0, "width": full_reference.shape[1], "height": full_reference.shape[0]})
    expected = crop_pixels(full_reference, crop)
    with Image.open(native_path) as image:
        actual = np.array(image)
        if image.mode != "RGB" or actual.shape != expected.shape:
            raise ValueError("Native output shape/mode does not match reference crop")
    delta = actual.astype(np.float64) - expected
    mse = float(np.mean(delta * delta))
    report = {
        "passed": bool(np.max(np.abs(delta)) <= 1),
        "gate": "Final cropped PNG max absolute difference <= one byte level; not a per-step float comparison for this case.",
        "category": prepared["category"], "crop": crop, "settings": benchmark["settings"],
        "checkpoint_sha256": reference["checkpoint_sha256"], "prepared_inputs_sha256": input_hashes,
        "reference_manifest_sha256": sha256(args.reference / "manifest.json"),
        "benchmark_sha256": sha256(args.native_run / "benchmark.json"),
        "prepared_manifest_sha256": sha256(args.prepared),
        "native_output_sha256": sha256(native_path), "reference_output_sha256": sha256(args.reference / "expected.png"),
        "changed_channels": int(np.count_nonzero(delta)), "max_abs": float(np.max(np.abs(delta))),
        "mae": float(np.mean(np.abs(delta))), "psnr_db": 10 * math.log10(255**2 / mse) if mse else None,
        "note": "Image agreement can establish that an observed artifact is reproduced by the reference pipeline; it is not a garment-quality score."
    }
    args.output.mkdir(parents=True, exist_ok=False)
    Image.fromarray(expected).save(args.output / "reference-cropped.png")
    Image.fromarray(actual).save(args.output / "native.png")
    Image.fromarray(np.clip(np.abs(delta) * 32, 0, 255).astype(np.uint8)).save(args.output / "difference-x32.png")
    write_report(args.output / "comparison.json", report)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("source", "reference", "reference-inputs", "native-run", "prepared", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    result = compare_case(args)
    print(json.dumps(result, indent=2))
    raise SystemExit(0 if result["passed"] else 1)
