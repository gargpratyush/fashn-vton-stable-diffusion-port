#!/usr/bin/env python3
"""Pinned CPU FASHN preparation, with an explicitly optional evaluation-only human parser."""

import argparse
import importlib
import importlib.metadata
import importlib.util
import json
import sys
import types
from pathlib import Path

from export_fashn_vton_reference import SOURCE_REVISION, download_verified_files, load_reference_package, save_tensors, sha256


POSE_REVISION = "548b5df25b84d9f4aac0611dfa1c2a7a12f15571"
POSE_HASHES = {
    "yolox_l.onnx": "7860ae79de6c89a3c1eb72ae9a2756c0ccfbe04b7791bb5880afabd97855a411",
    "dw-ll_ucoco_384.onnx": "724f4ff2439ed61afb86fb8a1951ec39c6220682803b4a8bd4f598cd913b1843",
}


def verify_pose_weights(directory):
    for name, expected in POSE_HASHES.items():
        if sha256(directory / name) != expected:
            raise ValueError(f"Unexpected DWPose checkpoint hash: {name}")


def download_pose(directory):
    download_verified_files(directory, "fashn-ai/DWPose", POSE_REVISION, POSE_HASHES)
    verify_pose_weights(directory)


def reference_modules(source):
    package = load_reference_package(source)
    name = package + ".preprocessing"
    if name not in sys.modules:
        module = types.ModuleType(name)
        module.__path__ = [str(Path(source).resolve() / "src" / "fashn_vton" / "preprocessing")]
        module.__package__ = name
        module.__spec__ = importlib.util.spec_from_loader(name, loader=None, is_package=True)
        sys.modules[name] = module
    return (importlib.import_module(name + ".transforms"),
            importlib.import_module(package + ".dwpose"),
            importlib.import_module(package + ".utils"))


def prepare_images(person, garment, detector, transforms, pose_module, utils, *,
                   category="tops", garment_photo_type="flat-lay", segmentation_free=True,
                   parser=None, agnostic=None, diagnostics=None):
    import cv2
    import numpy as np

    if person.mode != "RGB" or garment.mode != "RGB":
        raise ValueError("Preparation requires RGB images")
    if category not in ("tops", "bottoms", "one-pieces") or garment_photo_type not in ("flat-lay", "model"):
        raise ValueError("Invalid garment category or photo type")
    if not isinstance(segmentation_free, bool):
        raise ValueError("segmentation_free must be a boolean")
    needs_parser = garment_photo_type == "model" or not segmentation_free
    if needs_parser and (parser is None or agnostic is None):
        raise ValueError("This mode requires the explicitly authorized evaluation-only human parser")
    pre_resize = transforms.AspectPreserveResize((864, 864), mode="fit", backend="pil")
    person = np.array(pre_resize(person, allow_upsampling=False))
    garment = np.array(pre_resize(garment, allow_upsampling=False))
    pose = detector(person[..., ::-1])
    if np.count_nonzero(pose["bodies"]["subset"] >= 0) < 3:
        raise ValueError("DWPose did not find enough visible body keypoints; no blank-pose fallback is used")
    person_pose = pose_module.draw_pose(pose, *person.shape[:2], grayscale=True)
    garment_keypoints = (detector(garment[..., ::-1]) if garment_photo_type == "model"
                         else utils.get_dummy_dw_keypoints())
    if garment_photo_type == "model" and np.count_nonzero(garment_keypoints["bodies"]["subset"] >= 0) < 3:
        raise ValueError("DWPose did not find enough visible body keypoints in the worn-garment photo")
    garment_pose = pose_module.draw_pose(garment_keypoints, *garment.shape[:2], grayscale=True)
    person_seg = garment_seg = None
    if needs_parser:
        from fashn_human_parser import CATEGORY_TO_BODY_COVERAGE
        coverage = CATEGORY_TO_BODY_COVERAGE[category]
        labels = [agnostic.FASHN_LABELS_TO_IDS[label] for label in agnostic.BODY_COVERAGE_TO_FASHN_LABELS[coverage]]

        def segment(image):
            prediction = parser.predict(image)
            if (not isinstance(prediction, np.ndarray) or prediction.shape != image.shape[:2]
                    or not np.issubdtype(prediction.dtype, np.integer)
                    or np.any(prediction < 0) or np.any(prediction > 17)):
                raise ValueError("Human parser returned invalid label-map dimensions, dtype or labels")
            return prediction

        if not segmentation_free:
            person_seg = segment(person)
            person = agnostic.create_clothing_agnostic_image(
                person.copy(), person_seg.copy(), labels.copy(), coverage, disable_masking=False)
        if garment_photo_type == "model":
            garment_seg = segment(garment)
            if not np.any(np.isin(garment_seg, labels)):
                raise ValueError("Human parser found no garment pixels for the selected category")
            garment = agnostic.create_garment_image(garment.copy(), garment_seg.copy(), labels.copy())
    resize = transforms.ResizePad((576, 864), backend="opencv")
    images = {
        "ca_image": resize(person, mem_padding=True),
        "garment_image": resize(garment),
        "person_pose": resize(person_pose, interpolation=cv2.INTER_NEAREST_EXACT),
        "garment_pose": resize(garment_pose, interpolation=cv2.INTER_NEAREST_EXACT),
    }
    left, top, right, bottom = resize.pad_fn.padding_mem
    crop = {"x": left, "y": top, "width": 576 - left - right, "height": 864 - top - bottom}
    if diagnostics is not None:
        diagnostics.update(person_segmentation=person_seg, garment_segmentation=garment_seg,
                           garment_keypoints=garment_keypoints)
    return images, crop, pose


def normalized_condition(image, utils):
    tensor = utils.normalize_uint8_to_neg1_1(utils.numpy_to_torch(image))
    if tensor.ndim == 2:
        tensor = tensor.unsqueeze(0)
    return tensor.unsqueeze(0)


def opencv_distribution():
    found = []
    for package in ("opencv-python-headless", "opencv-python"):
        try:
            importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            continue
        found.append(package)
    if len(found) != 1:
        raise ValueError("Install exactly one OpenCV distribution in the preparation environment")
    return found[0]


def export_prepared(args):
    import numpy as np
    from PIL import Image

    if args.output.exists():
        raise ValueError("Choose a new output directory to avoid stale prepared inputs")
    opencv_package = opencv_distribution()
    photo_type = getattr(args, "garment_photo_type", "flat-lay")
    segmentation_free = getattr(args, "segmentation_free", True)
    needs_parser = photo_type == "model" or not segmentation_free
    parser = agnostic = parser_info = None
    if needs_parser:
        from fashn_parser_eval import load_parser, parser_provenance, require_evaluation_consent
        accepted = getattr(args, "accept_parser_research_license", False)
        require_evaluation_consent(accepted)
        if getattr(args, "parser_dir", None) is None:
            raise ValueError("--parser-dir is required for parser-dependent preparation")
    verify_pose_weights(args.dwpose_dir)
    transforms, pose_module, utils = reference_modules(args.source)
    if needs_parser:
        parser, agnostic = load_parser(args.source, args.parser_dir, accepted)
        parser_info = parser_provenance()
    detector = pose_module.DWposeDetector(str(args.dwpose_dir), device="cpu")
    diagnostics = {}
    with Image.open(args.person_image) as person, Image.open(args.garment_image) as garment:
        images, crop, pose = prepare_images(person.convert("RGB"), garment.convert("RGB"),
                                            detector, transforms, pose_module, utils, category=args.category,
                                            garment_photo_type=photo_type, segmentation_free=segmentation_free,
                                            parser=parser, agnostic=agnostic, diagnostics=diagnostics)
    args.output.mkdir(parents=True, exist_ok=False)
    document = {"schema": "fashn-vton-prepared-v1", "category": args.category, "crop": crop}
    tensors = {}
    for key, value in images.items():
        filename = key + ".png"
        Image.fromarray(value).save(args.output / filename)
        document[key] = filename
        tensor_key = {"ca_image": "ca_images", "garment_image": "garment_images",
                      "person_pose": "person_poses", "garment_pose": "garment_poses"}[key]
        tensors[tensor_key] = normalized_condition(value, utils)
    save_tensors(args.output / "conditions.safetensors", tensors)
    np.savez(args.output / "person-pose.npz", candidate=pose["bodies"]["candidate"],
             subset=pose["bodies"]["subset"], hands=pose["hands"], faces=pose["faces"])
    if photo_type == "model":
        garment_pose = diagnostics["garment_keypoints"]
        np.savez(args.output / "garment-pose.npz", candidate=garment_pose["bodies"]["candidate"],
                 subset=garment_pose["bodies"]["subset"], hands=garment_pose["hands"], faces=garment_pose["faces"])
    for key in ("person_segmentation", "garment_segmentation"):
        if diagnostics[key] is not None:
            Image.fromarray(diagnostics[key].astype(np.uint8)).save(args.output / (key + ".png"))
    provenance = {
        "source_revision": SOURCE_REVISION, "dwpose_revision": POSE_REVISION, "dwpose_sha256": POSE_HASHES,
        "input_kind": "raw-person-and-" + photo_type + "-garment",
        "segmentation_free": segmentation_free, "human_parser": parser_info,
        "inputs_sha256": {"person": sha256(args.person_image), "garment": sha256(args.garment_image)},
        "packages": {p: importlib.metadata.version(p) for p in (
            "torch", "numpy", "pillow", opencv_package, "onnxruntime", "matplotlib")},
        "crop": crop, "artifacts": {p.name: sha256(p) for p in args.output.iterdir() if p.is_file()},
    }
    (args.output / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n", encoding="utf-8")
    (args.output / "manifest.json").write_text(json.dumps(document, indent=2) + "\n", encoding="utf-8")
    print(f"Prepared inputs: {args.output / 'manifest.json'}")
    print(f"Output crop: {crop}; " + ("RESEARCH/EVALUATION ONLY: human parser used" if needs_parser else "no human parser loaded"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    download = commands.add_parser("download-pose", help="Explicitly download hash-pinned Apache-2.0 pose weights")
    download.add_argument("--dwpose-dir", type=Path, required=True)
    parser_download = commands.add_parser("download-parser", help="Optional restricted research/evaluation weights")
    parser_download.add_argument("--parser-dir", type=Path, required=True)
    parser_download.add_argument("--accept-parser-research-license", action="store_true")
    prepare = commands.add_parser("prepare", help="Create inputs; parser-dependent modes require explicit evaluation consent")
    for name in ("source", "dwpose-dir", "person-image", "garment-image", "output"):
        prepare.add_argument("--" + name, type=Path, required=True)
    prepare.add_argument("--category", choices=("tops", "bottoms", "one-pieces"), required=True)
    prepare.add_argument("--garment-photo-type", choices=("flat-lay", "model"), required=True)
    prepare.add_argument("--no-segmentation-free", dest="segmentation_free", action="store_false", default=True)
    prepare.add_argument("--parser-dir", type=Path)
    prepare.add_argument("--accept-parser-research-license", action="store_true")
    args = parser.parse_args()
    if args.command == "download-pose":
        download_pose(args.dwpose_dir)
    elif args.command == "download-parser":
        from fashn_parser_eval import download_parser
        download_parser(args.parser_dir, args.accept_parser_research_license)
    else:
        export_prepared(args)


if __name__ == "__main__":
    main()
