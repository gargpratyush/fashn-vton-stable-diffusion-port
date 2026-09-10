"""Run the pinned, unmodified upstream prepared-input sampler with its default CPU attention dispatch."""
import argparse
import importlib
import json
import os
from pathlib import Path
import time

import numpy as np
from PIL import Image
import torch
from safetensors.torch import load_file

from compare_fashn_case_output import crop_pixels
from export_fashn_vton_reference import (
    SOURCE_REVISION, load_pinned_model, load_reference, load_reference_package,
    sha256, validate_conditions, validate_sampling,
)
from prepare_fashn_vton import normalized_condition
from run_fashn_quality import write_report


def run(args):
    validate_sampling(args.steps, 0, args.cfg, args.shift, args.skip_cfg_last_n_steps)
    if args.threads < 1 or args.output.exists():
        raise ValueError("Use positive threads and a new output directory")
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    module, sampling, utils = load_reference(args.source)
    pipeline_type = importlib.import_module(load_reference_package(args.source) + ".pipeline").TryOnPipeline
    prepared = json.loads(args.prepared.read_text(encoding="utf-8"))
    conditions = load_file(str(args.inputs))
    validate_conditions(conditions)
    hashes = {}
    for key, plural in (("ca_image", "ca_images"), ("garment_image", "garment_images"),
                        ("person_pose", "person_poses"), ("garment_pose", "garment_poses")):
        path = args.prepared.parent / prepared[key]
        with Image.open(path) as image:
            restored = normalized_condition(np.array(image), utils)
        if not torch.equal(restored, conditions[plural]):
            raise ValueError("Prepared input lineage mismatch: " + key)
        hashes[key] = sha256(path)
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    load_start = time.perf_counter()
    model, _, checkpoint_hash = load_pinned_model(module, args.checkpoint)
    load_seconds = time.perf_counter() - load_start
    # Only the raw-image preparation/model-setup constructor is bypassed; _sample is original source.
    pipeline = pipeline_type.__new__(pipeline_type)
    pipeline.tryon_model = model
    torch.manual_seed(args.seed)
    start = time.perf_counter()
    images = pipeline._sample(
        **conditions, garment_categories=torch.tensor([pipeline.CATEGORY_TO_LABEL[prepared["category"]]]),
        num_timesteps=args.steps, time_shift_mu=args.shift, guidance_scale=args.cfg,
        skip_cfg_last_n_steps=args.skip_cfg_last_n_steps, use_tqdm=True,
    )
    seconds = time.perf_counter() - start
    if len(images) != 1 or images[0].mode != "RGB" or images[0].size != (576, 864):
        raise ValueError("Unexpected upstream output")
    args.output.mkdir(parents=True, exist_ok=False)
    images[0].save(args.output / "full.png")
    crop = prepared.get("crop", {"x": 0, "y": 0, "width": 576, "height": 864})
    Image.fromarray(crop_pixels(np.array(images[0]), crop)).save(args.output / "sample-0.png")
    report = {
        "source_revision": SOURCE_REVISION, "checkpoint_sha256": checkpoint_hash,
        "inputs_sha256": sha256(args.inputs), "prepared_manifest_sha256": sha256(args.prepared),
        "prepared_images_sha256": hashes, "category": prepared["category"], "crop": crop,
        "settings": {key: getattr(args, key) for key in ("steps", "cfg", "shift", "skip_cfg_last_n_steps", "seed", "threads")},
        "schedule": sampling.get_rf_schedule(args.steps, mu=args.shift),
        "dtype": "F32", "torch": torch.__version__, "attention": "upstream default SDPA dispatch; not forced math",
        "sampler": "Unmodified TryOnPipeline._sample; batched conditional/null, including skipped-CFG final step",
        "scope": "Prepared-input sampling, not raw __call__ or pipeline initialization. Meta-device checkpoint loading avoids redundant random weight initialization. No parser/detector instance loaded.",
        "model_load_and_hash_seconds": load_seconds, "sampling_and_pil_seconds": seconds,
        "output_sha256": sha256(args.output / "sample-0.png"),
    }
    write_report(args.output / "result.json", report)
    print(json.dumps(report, indent=2), flush=True)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("source", "checkpoint", "inputs", "prepared", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--cfg", type=float, default=1.5)
    parser.add_argument("--shift", type=float, default=1.5)
    parser.add_argument("--skip-cfg-last-n-steps", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--threads", type=int, default=16)
    run(parser.parse_args())
