"""Build a local-only, full-resolution comparison gallery; pixel metrics are not perceptual quality scores."""
import argparse
import html
import json
import math
from pathlib import Path
import re

import numpy as np
from PIL import Image, ImageDraw, ImageOps
from safetensors.numpy import load_file

from compare_fashn_case_output import crop_pixels
from compare_fashn_trajectories import pixels as tensor_pixels
from export_fashn_vton_reference import sha256
from run_fashn_quality import write_report


def pixel_metrics(reference, actual):
    if reference.shape != actual.shape or reference.ndim != 3 or reference.shape[2] != 3:
        raise ValueError("Comparison requires matching RGB shapes; no implicit resizing")
    delta = actual.astype(np.float64) - reference
    mse = float(np.mean(delta * delta))
    return {"changed_channels": int(np.count_nonzero(delta)), "total_channels": int(delta.size),
            "changed_pixels": int(np.count_nonzero(np.any(delta != 0, axis=2))),
            "max_abs": float(np.max(np.abs(delta))), "mae": float(np.mean(np.abs(delta))),
            "rmse": math.sqrt(mse), "psnr_db": 10 * math.log10(255**2 / mse) if mse else None}


def display_pixels(path, output_image):
    if path.suffix.lower() == ".safetensors":
        if not output_image:
            raise ValueError("Only generated output panels accept trajectory tensors")
        image = load_file(str(path))["image"]
        if image.dtype != np.float32 or not np.isfinite(image).all():
            raise ValueError("Trajectory image must be finite F32")
        return tensor_pixels(image)
    with Image.open(path) as image:
        if output_image and image.mode != "RGB":
            raise ValueError("Comparison outputs must be RGB")
        return np.array(image.convert("RGB"))


def build(spec, output, extra_specs=()):
    entries = json.loads(spec.read_text(encoding="utf-8"))
    for extra in extra_specs:
        entries.extend(json.loads(extra.read_text(encoding="utf-8")))
    identifiers = [entry["id"] for entry in entries]
    if len(set(identifiers)) != len(identifiers) or any(not re.fullmatch(r"[a-zA-Z0-9-]+", name) for name in identifiers):
        raise ValueError("Unique safe gallery IDs required")
    output.mkdir(parents=True, exist_ok=False)
    report = {"scope": "Numerical image comparisons, not garment quality scores; original images linked at full resolution.",
              "spec_sha256": sha256(spec), "additional_specs_sha256": {str(path): sha256(path) for path in extra_specs},
              "cases": {}}
    sections = []
    for entry in entries:
        directory = output / entry["id"]
        directory.mkdir()
        sources = {}
        for key in ("person", "garment", "native", "reference"):
            path = Path(entry[key])
            sources[key] = {"path": str(path), "sha256": sha256(path)}
            pixels = display_pixels(path, key in ("native", "reference"))
            if key + "_crop" in entry:
                pixels = crop_pixels(pixels, entry[key + "_crop"])
                sources[key]["crop"] = entry[key + "_crop"]
            Image.fromarray(pixels).save(directory / (key + ".png"))
        with Image.open(directory / "native.png") as image:
            native = np.array(image)
        with Image.open(directory / "reference.png") as image:
            reference = np.array(image)
        metrics = pixel_metrics(reference, native)
        Image.fromarray(np.clip(np.abs(native.astype(np.int16) - reference) * 32, 0, 255).astype(np.uint8)).save(
            directory / "difference-x32.png")
        labels = [("person", "Person input"), ("garment", "Garment input"),
                  ("reference", entry["reference_label"]), ("native", entry["native_label"])]
        sheet = Image.new("RGB", (1200, 490), "white")
        draw = ImageDraw.Draw(sheet)
        for index, (key, label) in enumerate(labels):
            draw.text((index * 300 + 4, 5), label, fill="black")
            with Image.open(directory / (key + ".png")) as image:
                thumbnail = ImageOps.contain(image, (288, 432))
                sheet.paste(thumbnail, (index * 300 + (300 - thumbnail.width) // 2, 34))
        draw.text((4, 473), "Research/evaluation only. Input attribution and comparison scope: see gallery.", fill="black")
        sheet.save(directory / "side-by-side.jpg", quality=95)
        write_report(directory / "metrics.json", {"metrics": metrics, "sources": sources, "scope": entry["scope"]})
        report["cases"][entry["id"]] = {"metrics": metrics, "sources": sources, "scope": entry["scope"]}
        cards = "".join(
            f'<figure><figcaption>{html.escape(label)}</figcaption><a href="{entry["id"]}/{key}.png">'
            f'<img loading="lazy" src="{entry["id"]}/{key}.png" alt="{html.escape(label)}"></a></figure>'
            for key, label in labels)
        sections.append(
            f'<section><h2>{html.escape(entry["title"])}</h2><p>{html.escape(entry["scope"])}</p>'
            f'<div class="grid">{cards}</div><p>{metrics["changed_channels"]:,} / {metrics["total_channels"]:,} '
            f'channels differ; max {metrics["max_abs"]:g}; MAE {metrics["mae"]:.8g}; '
            f'PSNR {"exact" if metrics["psnr_db"] is None else format(metrics["psnr_db"], ".4f") + " dB"}.</p>'
            f'<a href="{entry["id"]}/side-by-side.jpg">Contact sheet</a> | '
            f'<a href="{entry["id"]}/difference-x32.png">Absolute difference x32</a> | '
            f'<a href="{entry["id"]}/metrics.json">Metrics/provenance</a></section>')
    write_report(output / "manifest.json", report)
    page = (
        '<!doctype html><html lang="en"><meta charset="utf-8"><title>FASHN comparisons</title>'
        '<style>body{font:16px system-ui;margin:24px;background:#f5f5f5;color:#222}'
        '.grid{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:12px}'
        'figure{margin:0}img{width:100%;height:432px;object-fit:contain;background:white}'
        'section{background:white;padding:20px;margin:24px 0}figcaption{min-height:44px}'
        '@media(max-width:800px){.grid{grid-template-columns:repeat(2,minmax(0,1fr))}}</style>'
        '<h1>FASHN native / Python / precision comparisons</h1>'
        '<p>Click each image for full resolution. No external resources are loaded. '
        'Pixel metrics quantify agreement, not product fidelity. Read each row: a numerical oracle is not '
        'the original default sampler, and Q8 diagnostics are not validated public inference.</p>'
        '<p>CatVTON demo inputs: Zheng-Chong/CatVTON, revision 999bdbe81e6008a3f5749af7c1e0b0fa3d21b48e, '
        'repository-stated CC BY-NC-SA 4.0; original FASHN example images where labeled. '
        'Research/evaluation only; preserve applicable attribution and license restrictions.</p>'
        + "".join(sections) + "</html>")
    (output / "index.html").write_text(page, encoding="utf-8")
    print(output / "index.html")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--extra-spec", type=Path, action="append", default=[])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    build(args.spec, args.output, args.extra_spec)
