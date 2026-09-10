"""Independent reference preparation checks on the selected quality photographs."""
import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image
import torch

from export_fashn_vton_reference import save_tensors
from fashn_parser_eval import load_parser
from prepare_fashn_vton import normalized_condition, prepare_images, reference_modules
from run_fashn_quality import cases, validate_assets, write_report
from test_fashn_native_preprocess import Cached


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("source", "assets", "dwpose-dir", "parser-dir", "suite", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--accept-parser-research-license", action="store_true")
    args = parser.parse_args()
    if not args.accept_parser_research_license:
        parser.error("Explicit parser research/evaluation consent required")
    validate_assets(args.assets, False)
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(8)
    torch.set_num_interop_threads(1)
    transforms, pose_module, utils = reference_modules(args.source)
    human_parser, agnostic = load_parser(args.source, args.parser_dir, True)
    human_parser = Cached(human_parser.predict)
    detector = Cached(pose_module.DWposeDetector(str(args.dwpose_dir), device="cpu"))
    report = {"passed": False, "cases": {}}
    for name, person, garment, category, photo, masked, _ in cases(args.assets, args.source):
        if name == "flat-top-seed43":
            continue
        with Image.open(person) as p, Image.open(garment) as g:
            expected, crop, _ = prepare_images(p.convert("RGB"), g.convert("RGB"), detector, transforms,
                pose_module, utils, category=category, garment_photo_type=photo, segmentation_free=not masked,
                parser=human_parser, agnostic=agnostic)
        native = args.suite / ("prepared-" + name)
        manifest = json.loads((native / "manifest.json").read_text(encoding="utf-8"))
        entry = {"crop_exact": crop == manifest["crop"], "images": {}}
        destination = args.output / name
        destination.mkdir()
        conditions = {}
        for key, value in expected.items():
            with Image.open(native / manifest[key]) as image:
                actual = np.array(image)
            if actual.shape != value.shape:
                raise ValueError("Preparation shape mismatch: " + name + "/" + key)
            delta = actual.astype(np.int16) - value.astype(np.int16)
            entry["images"][key] = {"changed_channels": int(np.count_nonzero(delta)), "max_abs": int(np.max(np.abs(delta)))}
            Image.fromarray(value).save(destination / (key + ".png"))
            plural = {"ca_image": "ca_images", "garment_image": "garment_images", "person_pose": "person_poses", "garment_pose": "garment_poses"}[key]
            conditions[plural] = normalized_condition(value, utils)
        save_tensors(destination / "conditions.safetensors", conditions)
        entry["passed"] = entry["crop_exact"] and all(v["changed_channels"] == 0 for v in entry["images"].values())
        report["cases"][name] = entry
        write_report(args.output / "comparison.json", report)
        print(name + ": " + ("exact" if entry["passed"] else "DIFFERENCES"), flush=True)
    report["passed"] = all(row["passed"] for row in report["cases"].values())
    write_report(args.output / "comparison.json", report)
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
