#!/usr/bin/env python3
"""Development-only FASHN 1.5 CPU oracle; does not implement native inference."""

import argparse
import hashlib
import importlib
import importlib.metadata
import importlib.util
import json
import math
import struct
import subprocess
import sys
import time
import types
from pathlib import Path


SOURCE_REVISION = "7c0f10af3f91ad4048fe9729c470a13ef905d25a"
CHECKPOINT_REVISION = "7720683168567eb5a2a4c67f15116c6e29c83ded"
CHECKPOINT_SHA256 = "d6cd38286885bc29fa487ea9383f80ffeb95862e7747c630d42c5d3c05bdd35a"
PREFIX = "model.diffusion_model."
HEIGHT, WIDTH = 864, 576
CONDITION_CHANNELS = {
    "ca_images": 3,
    "garment_images": 3,
    "person_poses": 1,
    "garment_poses": 1,
}
FLOAT_SIZES = {"BF16": 2, "F16": 2, "F32": 4}


def expected_shapes():
    """Released preset in PyTorch dimension order, including the unused buffer."""
    shapes = {}

    def linear(name, output, input_size):
        shapes[name + ".weight"] = [output, input_size]
        shapes[name + ".bias"] = [output]

    def norm(name):
        shapes[name + ".query_norm.scale"] = [128]
        shapes[name + ".key_norm.scale"] = [128]

    for name, channels in (("x_embedder", 7), ("garment_embedder", 4)):
        shapes[name + ".proj.weight"] = [1280, channels, 12, 12]
        shapes[name + ".proj.bias"] = [1280]
    linear("t_embedder.mlp.in_layer", 1280, 256)
    linear("t_embedder.mlp.out_layer", 1280, 1280)
    shapes["y_embedder.weight"] = [4, 1280]
    for family, depth in (("x_patch_mixer", 4), ("single_blocks", 16)):
        for index in range(depth):
            root = f"{family}.{index}."
            linear(root + "linear1", 8960, 1280)
            linear(root + "linear2", 1280, 6400)
            linear(root + "modulation.lin", 3840, 1280)
            norm(root + "norm")
    for index in range(8):
        for stream in ("img", "txt"):
            root = f"double_blocks.{index}.{stream}_"
            linear(root + "attn.qkv", 3840, 1280)
            linear(root + "attn.proj", 1280, 1280)
            norm(root + "attn.norm")
            linear(root + "mod.lin", 7680, 1280)
            linear(root + "mlp.0", 5120, 1280)
            linear(root + "mlp.2", 1280, 5120)
    linear("final_layer.linear", 432, 1280)
    linear("final_layer.adaLN_modulation.1", 2560, 1280)
    shapes["patch_mixer_token"] = [1, 1, 432]
    return shapes


def unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate JSON key: {key}")
        result[key] = value
    return result


def read_header(path):
    path = Path(path)
    size = path.stat().st_size
    with path.open("rb") as handle:
        length_bytes = handle.read(8)
        if len(length_bytes) != 8:
            raise ValueError("Truncated safetensors length prefix")
        length = struct.unpack("<Q", length_bytes)[0]
        if not 2 <= length <= min(16 * 1024 * 1024, size - 8):
            raise ValueError("Invalid or oversized safetensors header length")
        header = json.loads(handle.read(length), object_pairs_hook=unique_object)
    if not isinstance(header, dict):
        raise ValueError("Safetensors header must be an object")
    return header, size - 8 - length


def validate_header(header, payload_bytes=None):
    expected = expected_shapes()
    found, intervals = {}, []
    for original, info in header.items():
        if original == "__metadata__":
            continue
        name = original.removeprefix(PREFIX)
        if name in found:
            raise ValueError(f"Checkpoint prefix collision: {name}")
        if name not in expected:
            raise ValueError(f"Unexpected FASHN tensor: {original}")
        if not isinstance(info, dict):
            raise ValueError(f"Invalid metadata for {original}")
        shape = info.get("shape")
        if not isinstance(shape, list) or any(type(d) is not int for d in shape) or shape != expected[name]:
            raise ValueError(f"Shape mismatch for {original}: {shape}, expected {expected[name]}")
        dtype = info.get("dtype")
        if not isinstance(dtype, str) or dtype not in FLOAT_SIZES:
            raise ValueError(f"Unsupported reference dtype for {original}: {dtype}")
        offsets = info.get("data_offsets")
        if (not isinstance(offsets, list) or len(offsets) != 2
                or any(type(v) is not int for v in offsets) or offsets[0] < 0
                or offsets[1] - offsets[0] != math.prod(shape) * FLOAT_SIZES[dtype]):
            raise ValueError(f"Invalid data offsets for {original}")
        found[name] = info
        intervals.append((offsets[0], offsets[1], name))
    missing = set(expected) - set(found) - {"patch_mixer_token"}
    if missing:
        raise ValueError("Missing FASHN tensors: " + ", ".join(sorted(missing)))
    end = 0
    for start, stop, name in sorted(intervals):
        if start != end:
            raise ValueError(f"Overlapping or non-contiguous tensor payload at {name}")
        end = stop
    if payload_bytes is not None and end != payload_bytes:
        raise ValueError(f"Payload length mismatch: header describes {end}, file has {payload_bytes}")
    return found


def sha256(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def download_verified_files(directory, repository, revision, hashes):
    import urllib.request

    directory.mkdir(parents=True, exist_ok=True)
    for name, expected in hashes.items():
        target = directory / name
        if target.exists():
            if sha256(target) != expected:
                raise ValueError(f"Refusing to overwrite unrecognized checkpoint: {target}")
            continue
        partial = directory / (name + ".part")
        if partial.exists():
            raise ValueError(f"Remove the interrupted download explicitly before retrying: {partial}")
        print(f"Downloading pinned {repository}/{name}", flush=True)
        urllib.request.urlretrieve(f"https://huggingface.co/{repository}/resolve/{revision}/{name}", partial)
        if sha256(partial) != expected:
            raise ValueError(f"Downloaded checkpoint checksum mismatch: {partial}")
        partial.rename(target)


def load_reference_package(source):
    """Register a clean pinned source package without its eager pipeline initializer."""
    source = Path(source).resolve()
    revision = subprocess.check_output(
        ["git", "-C", str(source), "rev-parse", "HEAD"], text=True).strip()
    if revision != SOURCE_REVISION:
        raise ValueError(f"Reference source must be {SOURCE_REVISION}, found {revision}")
    changes = subprocess.check_output(
        ["git", "-C", str(source), "status", "--porcelain", "--untracked-files=all", "--", "src/fashn_vton"],
        text=True,
    ).strip()
    if changes:
        raise ValueError("Reference source has local modifications; use a clean pinned checkout")
    package_path = source / "src" / "fashn_vton"
    package_name = "_sdcpp_fashn_" + hashlib.sha256(str(source).encode()).hexdigest()[:16]
    if package_name not in sys.modules:
        package = types.ModuleType(package_name)
        package.__path__ = [str(package_path)]
        package.__package__ = package_name
        package.__spec__ = importlib.util.spec_from_loader(package_name, loader=None, is_package=True)
        sys.modules[package_name] = package
    return package_name


def load_reference(source):
    """Import original model/utils without importing the eager detector/parser pipeline."""
    package_name = load_reference_package(source)
    model_module = importlib.import_module(package_name + ".tryon_mmdit")
    sampling_module = importlib.import_module(package_name + ".utils.sampling")
    tensor_module = importlib.import_module(package_name + ".utils.tensor")
    return model_module, sampling_module, tensor_module


def make_noise(batch_size, height, width, seed):
    import torch

    generator = torch.Generator(device="cpu").manual_seed(seed)
    return torch.randn((batch_size, 3, height, width), generator=generator, dtype=torch.float32)


def synthetic_conditions(height, width):
    import torch

    rows = torch.arange(height, dtype=torch.int64)[:, None]
    cols = torch.arange(width, dtype=torch.int64)[None, :]
    rgb = torch.stack([(rows * (c + 1) + cols * (c + 3)) % 256 for c in range(3)])
    pose = ((rows * 7 + cols * 11) % 256).unsqueeze(0)
    return {
        "ca_images": rgb.unsqueeze(0).float() / 127.5 - 1,
        "garment_images": rgb.flip(-1).unsqueeze(0).float() / 127.5 - 1,
        "person_poses": pose.unsqueeze(0).float() / 127.5 - 1,
        "garment_poses": torch.full((1, 1, height, width), -1.0),
    }


def validate_conditions(conditions, height=HEIGHT, width=WIDTH):
    import torch

    if set(conditions) != set(CONDITION_CHANNELS):
        raise ValueError("Prepared tensors must contain exactly: " + ", ".join(CONDITION_CHANNELS))
    for name, channels in CONDITION_CHANNELS.items():
        value = conditions[name]
        if value.dtype != torch.float32 or tuple(value.shape) != (1, channels, height, width):
            raise ValueError(f"{name} must be F32 [1,{channels},{height},{width}]")
        if not torch.isfinite(value).all() or (value < -1).any() or (value > 1).any():
            raise ValueError(f"{name} must contain finite normalized values in [-1,1]")


def validate_sampling(steps, step_index, cfg, shift, skip):
    if steps < 1 or not 0 <= step_index < steps:
        raise ValueError("Steps must be positive and step-index must be in [0,steps)")
    if not math.isfinite(cfg) or cfg < 0:
        raise ValueError("CFG must be finite and non-negative")
    if not math.isfinite(shift) or abs(shift) > 20:
        raise ValueError("Reference time shift must be finite and within [-20,20]")
    if not 0 <= skip <= steps:
        raise ValueError("skip-cfg-last-n-steps must be in [0,steps]")


def euler_update(noisy, conditional, unconditional, cfg, delta, skip_cfg):
    if noisy.shape != conditional.shape or noisy.shape != unconditional.shape:
        raise ValueError("Noise and both velocities must have identical shapes")
    if not math.isfinite(cfg) or cfg < 0 or not math.isfinite(delta) or delta <= 0:
        raise ValueError("Invalid CFG or non-positive Euler time increment")
    velocity = conditional if skip_cfg else unconditional + cfg * (conditional - unconditional)
    return velocity, noisy + delta * velocity


def save_tensors(path, tensors):
    import torch
    from safetensors.torch import save_file

    values = {}
    for name, value in tensors.items():
        if value.is_floating_point() and not torch.isfinite(value).all():
            raise ValueError(f"Non-finite reference tensor: {name}")
        values[name] = value.detach().cpu().contiguous().clone()
    save_file(values, str(path))


def capture_forward(model, noisy, times, conditions, category, unconditional=False):
    import torch

    captured, calls, handles = {}, {}, []
    names = {"x_embedder", "garment_embedder", "t_embedder", "y_embedder", "pe_embedder", "final_layer"}
    for family in ("x_patch_mixer", "double_blocks", "single_blocks"):
        blocks = getattr(model, family)
        names.update(f"{family}.{index}" for index in (0, len(blocks) - 1))

    def record(name, value):
        if isinstance(value, torch.Tensor):
            captured[name] = value.detach().cpu().contiguous().clone()
        elif isinstance(value, (tuple, list)):
            for index, child in enumerate(value):
                record(f"{name}.{index}", child)
        elif isinstance(value, dict):
            for key, child in value.items():
                record(f"{name}.{key}", child)

    def make_hook(name):
        def hook(module, args, kwargs, output):
            index = calls.get(name, 0)
            calls[name] = index + 1
            root = f"{name}.call{index}"
            if name.endswith(".0") or name in ("t_embedder", "y_embedder", "final_layer"):
                record(root + ".args", args)
                record(root + ".kwargs", kwargs)
            record(root + ".output", output)
        return hook

    for name, module in model.named_modules():
        if name in names:
            handles.append(module.register_forward_hook(make_hook(name), with_kwargs=True))
    try:
        # Use the original dropout path rather than reimplementing the null branch.
        mask = torch.full((noisy.shape[0],), not unconditional, dtype=torch.bool)
        with torch.inference_mode():
            result = model(noisy, times, **conditions, garment_categories=category, mask=mask)["x"]
        record("velocity", result)
    finally:
        for handle in handles:
            handle.remove()
    return captured


def export_layout(model_module, tensor_module, path):
    import torch

    image = torch.arange(3 * 24 * 36, dtype=torch.float32).reshape(1, 3, 24, 36)
    patches, ids = model_module.prepare(image, patch_size=12)
    packed = patches.transpose(1, 2).reshape(1, 432, 2, 3)
    reconstructed = tensor_module.unpack_images(packed, 12)
    torch.testing.assert_close(image, reconstructed, rtol=0, atol=0)
    pixels = torch.arange(256, dtype=torch.uint8)
    save_tensors(path, {
        "image": image, "patches": patches, "ids": ids, "reconstructed": reconstructed,
        "pixels": pixels, "normalized": tensor_module.normalize_uint8_to_neg1_1(pixels),
    })


def load_pinned_model(model_module, checkpoint):
    import torch
    from safetensors.torch import load_file

    header, payload = read_header(checkpoint)
    metadata = validate_header(header, payload)
    checkpoint_hash = sha256(checkpoint)
    if checkpoint_hash != CHECKPOINT_SHA256:
        raise ValueError("Oracle requires the exact pinned official BF16 checkpoint (SHA256 mismatch)")
    print("Loading pinned checkpoint as CPU F32...", flush=True)
    with torch.device("meta"):
        model = model_module.TryOnModel()
    model.load_state_dict(load_file(str(checkpoint), device="cpu"), strict=True, assign=True)
    return model.float().eval(), metadata, checkpoint_hash


def oracle_input_state(args, schedule):
    import torch
    from safetensors.torch import load_file

    trajectory = getattr(args, "trajectory", None)
    if trajectory is None:
        return make_noise(1, HEIGHT, WIDTH, args.seed), {"kind": "seeded_noise"}
    if args.inputs is None:
        raise ValueError("Trajectory probes require matching prepared inputs")
    manifest_path = trajectory / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if (manifest["source_revision"] != SOURCE_REVISION or manifest["checkpoint_sha256"] != CHECKPOINT_SHA256
            or manifest["dtype"] != "F32" or manifest["input_kind"] != "prepared"):
        raise ValueError("Expected a pinned F32 photographic reference trajectory")
    if manifest["inputs_sha256"] != sha256(args.inputs):
        raise ValueError("Trajectory/prepared input hash mismatch")
    for key in ("steps", "cfg", "skip_cfg_last_n_steps", "seed", "category"):
        if manifest[key] != getattr(args, key):
            raise ValueError("Trajectory probe parameter mismatch: " + key)
    if manifest["schedule"] != schedule:
        raise ValueError("Trajectory probe schedule mismatch")
    name = "initial.safetensors" if args.step_index == 0 else f"step-{args.step_index - 1}.safetensors"
    key = "noise" if args.step_index == 0 else "image"
    state_path = trajectory / name
    digest = sha256(state_path)
    if manifest["artifacts"].get(name) != digest:
        raise ValueError("Trajectory input-state artifact hash mismatch")
    image = load_file(str(state_path))[key]
    if image.dtype != torch.float32 or tuple(image.shape) != (1, 3, HEIGHT, WIDTH) or not torch.isfinite(image).all():
        raise ValueError("Trajectory input state must be finite F32 [1,3,864,576]")
    return image, {"kind": "reference_trajectory", "trajectory_manifest_sha256": sha256(manifest_path),
                   "state_file": name, "state_sha256": digest}


def export_oracle(args):
    import torch
    from safetensors.torch import load_file
    from torch.nn.attention import SDPBackend, sdpa_kernel

    validate_sampling(args.steps, args.step_index, args.cfg, args.shift, args.skip_cfg_last_n_steps)
    if args.threads < 1:
        raise ValueError("Thread count must be positive")
    if args.output.exists():
        raise ValueError("Output directory already exists; choose a new directory to avoid stale fixtures")
    model_module, sampling_module, tensor_module = load_reference(args.source)
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    conditions = load_file(str(args.inputs)) if args.inputs else synthetic_conditions(HEIGHT, WIDTH)
    validate_conditions(conditions)
    schedule = sampling_module.get_rf_schedule(args.steps, mu=args.shift)
    if (not all(math.isfinite(t) for t in schedule) or schedule[0] != 0 or schedule[-1] != 1
            or any(b <= a for a, b in zip(schedule, schedule[1:]))):
        raise ValueError("Reference produced a non-finite or non-increasing schedule")
    noisy, state_source = oracle_input_state(args, schedule)
    times = torch.tensor([schedule[args.step_index]], dtype=torch.float32)
    category = torch.tensor([{"tops": 1, "bottoms": 2, "one-pieces": 3}[args.category]], dtype=torch.int64)

    model, metadata, checkpoint_hash = load_pinned_model(model_module, args.checkpoint)
    args.output.mkdir(parents=True, exist_ok=False)
    export_layout(model_module, tensor_module, args.output / "layout.safetensors")
    save_tensors(args.output / "inputs.safetensors", {
        **conditions, "noise": noisy, "times": times, "category": category,
        "schedule": torch.tensor(schedule, dtype=torch.float32),
        "time_features": model_module.timestep_embedding(times, 256),
    })
    durations, velocities = {}, {}
    for branch in ("conditional", "unconditional"):
        print(f"Running {branch} full-canvas F32 forward with SDPA math...", flush=True)
        start = time.perf_counter()
        with sdpa_kernel(SDPBackend.MATH):
            captured = capture_forward(model, noisy, times, conditions, category, branch == "unconditional")
        durations[branch] = time.perf_counter() - start
        velocity = captured["velocity"]
        if tuple(velocity.shape) != (1, 3, HEIGHT, WIDTH):
            raise ValueError(f"Wrong reference output shape: {tuple(velocity.shape)}")
        save_tensors(args.output / f"{branch}.safetensors", captured)
        velocities[branch] = velocity
        print(f"{branch}: {durations[branch]:.3f}s; {len(captured)} captured tensors", flush=True)
        del captured
    skip = args.step_index >= args.steps - args.skip_cfg_last_n_steps
    guided, updated = euler_update(
        noisy, velocities["conditional"], velocities["unconditional"], args.cfg,
        schedule[args.step_index + 1] - schedule[args.step_index], skip,
    )
    save_tensors(args.output / "euler.safetensors", {"guided_velocity": guided, "updated_image": updated})
    packages = {item.metadata["Name"]: item.version for item in importlib.metadata.distributions()}
    source_root = args.source / "src" / "fashn_vton"
    source_hashes = {p.relative_to(source_root).as_posix(): sha256(p) for p in sorted(source_root.rglob("*.py"))}
    manifest = {
        "format": "sdcpp-fashn-reference-v1",
        "source_revision": SOURCE_REVISION,
        "source_sha256": source_hashes,
        "checkpoint_revision": CHECKPOINT_REVISION,
        "checkpoint_sha256": checkpoint_hash,
        "checkpoint_tensor_count": len(metadata),
        "checkpoint_scalar_count": sum(math.prod(v["shape"]) for v in metadata.values()),
        "inference_tensor_count": len(metadata) - int("patch_mixer_token" in metadata),
        "input_kind": "prepared" if args.inputs else "synthetic-diagnostic-not-a-real-try-on",
        "inputs_sha256": sha256(args.inputs) if args.inputs else None,
        "input_state": state_source,
        "layout": "PyTorch contiguous; reverse shape dimensions for GGML, do not transpose payload",
        "height": HEIGHT, "width": WIDTH, "batch_size": 1, "dtype": "F32",
        "attention": "torch SDPA MATH", "deterministic_algorithms": True,
        "threads": args.threads, "seed": args.seed, "category": args.category,
        "steps": args.steps, "step_index": args.step_index, "cfg": args.cfg,
        "shift": args.shift, "skip_cfg_last_n_steps": args.skip_cfg_last_n_steps,
        "schedule": schedule, "cfg_skipped_for_exported_step": skip,
        "forward_seconds": durations, "python": sys.version, "packages": dict(sorted(packages.items())),
        "artifacts": {p.name: sha256(p) for p in sorted(args.output.glob("*.safetensors"))},
    }
    # Written last: absence of this manifest means an interrupted/incomplete export.
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"Complete reference fixture: {args.output}", flush=True)


def compare_tensors(reference, actual, atol=1e-4, rtol=1e-4, relative_l2_limit=1e-3):
    import torch

    if any(not math.isfinite(v) or v < 0 for v in (atol, rtol, relative_l2_limit)):
        raise ValueError("Comparison tolerances must be finite and non-negative")
    if set(reference) != set(actual):
        raise ValueError(f"Tensor key mismatch: missing={sorted(set(reference) - set(actual))}, "
                         f"extra={sorted(set(actual) - set(reference))}")
    report = {}
    for name, expected in reference.items():
        value = actual[name]
        if expected.shape != value.shape or expected.dtype != value.dtype:
            raise ValueError(f"Shape/dtype mismatch for {name}")
        if not expected.is_floating_point():
            report[name] = {"passed": torch.equal(expected, value), "comparison": "exact"}
            continue
        if not torch.isfinite(expected).all() or not torch.isfinite(value).all():
            raise ValueError(f"Non-finite comparison tensor: {name}")
        delta = value.double() - expected.double()
        relative_l2 = (torch.linalg.vector_norm(delta)
                       / torch.linalg.vector_norm(expected.double()).clamp_min(1e-12)).item()
        report[name] = {
            "passed": bool(torch.allclose(value, expected, atol=atol, rtol=rtol))
                      and relative_l2 <= relative_l2_limit,
            "max_abs": delta.abs().max().item() if delta.numel() else 0.0,
            "rmse": delta.square().mean().sqrt().item() if delta.numel() else 0.0,
            "relative_l2": relative_l2,
        }
    return report


def export_native_fixtures(source, output):
    import torch
    from safetensors.torch import load_file

    if not (source / "manifest.json").is_file():
        raise ValueError("Reference export is incomplete (missing manifest)")
    output.mkdir(parents=True, exist_ok=False)
    for path in sorted(source.glob("*.safetensors")):
        values = load_file(str(path))
        for name, value in values.items():
            while value.ndim > 4 and value.shape[0] == 1:
                value = value.squeeze(0)
            if value.ndim > 4:
                raise ValueError(f"Cannot represent {name} with GGML's four dimensions")
            if not value.is_floating_point():
                converted = value.float()
                if not torch.equal(converted.to(value.dtype), value):
                    raise ValueError(f"Integer fixture cannot be represented exactly as F32: {name}")
                value = converted
            values[name] = value
        save_tensors(output / path.name, values)
    manifest = json.loads((source / "manifest.json").read_text(encoding="utf-8"))
    manifest["transport"] = "GGML rank <=4; only leading singleton axes removed; small integers losslessly F32"
    manifest["artifacts"] = {p.name: sha256(p) for p in sorted(output.glob("*.safetensors"))}
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


def export_prepared_fixtures(source, output):
    import torch
    from PIL import Image
    from safetensors.torch import load_file

    manifest = json.loads((source / "manifest.json").read_text(encoding="utf-8"))
    inputs = load_file(str(source / "inputs.safetensors"))
    conditions = {key: inputs[key] for key in CONDITION_CHANNELS}
    validate_conditions(conditions)
    output.mkdir(parents=True, exist_ok=False)
    document = {"schema": "fashn-vton-prepared-v1", "category": manifest["category"]}
    keys = {"ca_images": "ca_image", "garment_images": "garment_image",
            "person_poses": "person_pose", "garment_poses": "garment_pose"}
    for name, value in conditions.items():
        pixels = ((value[0] + 1) * 127.5).round().clamp(0, 255).to(torch.uint8)
        restored = pixels.float() / 127.5 - 1
        if not torch.allclose(restored, value[0], atol=1e-7, rtol=0):
            raise ValueError(f"{name} cannot be losslessly represented by prepared uint8 pixels")
        pixels = pixels.permute(1, 2, 0).numpy() if pixels.shape[0] == 3 else pixels[0].numpy()
        filename = keys[name] + ".png"
        Image.fromarray(pixels).save(output / filename)
        document[keys[name]] = filename
    (output / "manifest.json").write_text(json.dumps(document, indent=2) + "\n", encoding="utf-8")
    (output / "provenance.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


def export_trajectory(args):
    import torch
    from PIL import Image
    from safetensors.torch import load_file
    from torch.nn.attention import SDPBackend, sdpa_kernel

    validate_sampling(args.steps, 0, args.cfg, args.shift, args.skip_cfg_last_n_steps)
    if args.output.exists() or args.threads < 1:
        raise ValueError("Use a new output directory and a positive thread count")
    module, sampling, _ = load_reference(args.source)
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    conditions = load_file(str(args.inputs)) if args.inputs else synthetic_conditions(HEIGHT, WIDTH)
    validate_conditions(conditions)
    model, _, checkpoint_hash = load_pinned_model(module, args.checkpoint)
    x = make_noise(1, HEIGHT, WIDTH, args.seed)
    category = torch.tensor([{"tops": 1, "bottoms": 2, "one-pieces": 3}[args.category]], dtype=torch.int64)
    schedule = sampling.get_rf_schedule(args.steps, mu=args.shift)
    args.output.mkdir(parents=True, exist_ok=False)
    save_tensors(args.output / "initial.safetensors", {"noise": x})
    with torch.inference_mode(), sdpa_kernel(SDPBackend.MATH):
        for i in range(args.steps):
            times = torch.tensor([schedule[i]], dtype=torch.float32)
            vc = model(x, times, **conditions, garment_categories=category)["x"]
            skip = i >= args.steps - args.skip_cfg_last_n_steps
            vu = model(x, times, **conditions, garment_categories=category,
                       mask=torch.zeros(1, dtype=torch.bool))["x"] if not skip and args.cfg != 1 else vc
            guided, x = euler_update(x, vc, vu, args.cfg, schedule[i + 1] - schedule[i], skip)
            save_tensors(args.output / f"step-{i}.safetensors", {"guided_velocity": guided, "image": x})
            print(f"Reference trajectory step {i + 1}/{args.steps}", flush=True)
    pixels = ((x[0] + 1) * 0.5).clamp(0, 1).mul(255).to(torch.uint8).permute(1, 2, 0).numpy()
    Image.fromarray(pixels).save(args.output / "expected.png")
    manifest = {
        "source_revision": SOURCE_REVISION, "checkpoint_sha256": checkpoint_hash,
        "input_kind": "prepared" if args.inputs else "synthetic-diagnostic-not-a-real-try-on",
        "inputs_sha256": sha256(args.inputs) if args.inputs else None, "category": args.category,
        "steps": args.steps, "schedule": schedule, "cfg": args.cfg,
        "skip_cfg_last_n_steps": args.skip_cfg_last_n_steps, "seed": args.seed,
        "attention": "SDPA MATH", "dtype": "F32", "torch": torch.__version__,
        "artifacts": {p.name: sha256(p) for p in args.output.iterdir() if p.is_file()},
    }
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    contract = commands.add_parser("contract", help="Validate complete metadata without loading tensor payloads")
    contract.add_argument("checkpoint", type=Path)
    export = commands.add_parser("export", help="Export one real-weight full-canvas forward per CFG branch")
    export.add_argument("--source", type=Path, required=True)
    export.add_argument("--checkpoint", type=Path, required=True)
    export.add_argument("--output", type=Path, required=True)
    export.add_argument("--inputs", type=Path, help="Four normalized prepared F32 tensors; omitted = synthetic")
    export.add_argument("--trajectory", type=Path, help="Use the actual input state at step-index from a matching complete reference trajectory")
    export.add_argument("--threads", type=int, default=8)
    export.add_argument("--seed", type=int, default=42)
    export.add_argument("--category", choices=["tops", "bottoms", "one-pieces"], default="tops")
    export.add_argument("--steps", type=int, default=30)
    export.add_argument("--step-index", type=int, default=0)
    export.add_argument("--cfg", type=float, default=1.5)
    export.add_argument("--shift", type=float, default=1.5)
    export.add_argument("--skip-cfg-last-n-steps", type=int, default=1)
    compare = commands.add_parser("compare", help="Compare safetensors fixtures; exit nonzero on any mismatch")
    compare.add_argument("reference", type=Path)
    compare.add_argument("actual", type=Path)
    compare.add_argument("--atol", type=float, default=1e-4)
    compare.add_argument("--rtol", type=float, default=1e-4)
    compare.add_argument("--relative-l2-limit", type=float, default=1e-3)
    native = commands.add_parser("native-fixtures", help="Adapt reference rank/integers for native fixture I/O")
    native.add_argument("source", type=Path)
    native.add_argument("output", type=Path)
    prepared = commands.add_parser("prepared-fixtures", help="Export exact uint8 oracle conditions as CLI PNG inputs")
    prepared.add_argument("source", type=Path)
    prepared.add_argument("output", type=Path)
    trajectory = commands.add_parser("trajectory", help="Synthetic or prepared trajectory for end-to-end native regression")
    trajectory.add_argument("--source", type=Path, required=True)
    trajectory.add_argument("--checkpoint", type=Path, required=True)
    trajectory.add_argument("--output", type=Path, required=True)
    trajectory.add_argument("--inputs", type=Path, help="Four normalized F32 conditions; omitted = synthetic")
    trajectory.add_argument("--category", choices=["tops", "bottoms", "one-pieces"], default="tops")
    trajectory.add_argument("--steps", type=int, default=2)
    trajectory.add_argument("--cfg", type=float, default=1.5)
    trajectory.add_argument("--shift", type=float, default=1.5)
    trajectory.add_argument("--skip-cfg-last-n-steps", type=int, default=1)
    trajectory.add_argument("--seed", type=int, default=42)
    trajectory.add_argument("--threads", type=int, default=8)
    args = parser.parse_args()
    if args.command == "contract":
        header, payload = read_header(args.checkpoint)
        tensors = validate_header(header, payload)
        print(json.dumps({"tensor_count": len(tensors), "payload_bytes": payload,
                          "scalar_count": sum(math.prod(v["shape"]) for v in tensors.values())}, indent=2))
    elif args.command == "export":
        export_oracle(args)
    elif args.command == "native-fixtures":
        export_native_fixtures(args.source, args.output)
    elif args.command == "prepared-fixtures":
        export_prepared_fixtures(args.source, args.output)
    elif args.command == "trajectory":
        export_trajectory(args)
    else:
        from safetensors.torch import load_file

        report = compare_tensors(load_file(str(args.reference)), load_file(str(args.actual)),
                                 args.atol, args.rtol, args.relative_l2_limit)
        print(json.dumps(report, indent=2))
        return 0 if all(item["passed"] for item in report.values()) else 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
