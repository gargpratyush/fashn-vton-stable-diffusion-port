"""Reproducible small research-only FASHN quality matrix; no quality score is inferred from execution."""
import argparse
import html
import json
from pathlib import Path
import subprocess
import time
from types import SimpleNamespace
import urllib.request

from PIL import Image, ImageDraw, ImageOps

from benchmark_fashn_vton import benchmark
from export_fashn_vton_reference import sha256, validate_sampling

REVISION = "999bdbe81e6008a3f5749af7c1e0b0fa3d21b48e"
REPOSITORY = "https://github.com/Zheng-Chong/CatVTON"
LICENSE = "CC-BY-NC-SA-4.0"
ASSETS = {
    "LICENSE": ("LICENSE", "095791f7f830b929a5c4c5bd8fb9e62a1694336fc6b915f771e25a44015c403c"),
    "README.md": ("README.md", "795aa6a5aeb83deb82c793d7a18430b6b48c09393f4939941fd1b0688233630a"),
    "model_5.png": ("resource/demo/example/person/men/model_5.png", "5faaea84635da215bd0819cf5ce65512ca1b742c39ee8fd67176b19e084ed872"),
    "model_8.png": ("resource/demo/example/person/women/model_8.png", "bf4f2064cc6efe6aa4b6bb503d6d2aadba4feff13b5784457ed7ddd63c1fc20f"),
    "21514384_52353349_1000.jpg": ("resource/demo/example/condition/upper/21514384_52353349_1000.jpg", "fd74b6db0fea913db3a18e4a154d4035c0149c9250f7e1335dbd6077c0318064"),
    "21744571_51588794_1000.jpg": ("resource/demo/example/condition/overall/21744571_51588794_1000.jpg", "735397cb34a0c9c941739633951f10a6dde23a6005b2a4928f7fe50a884feeea"),
    "baumu30483223c3_1719437121402_2-0._QL90_UX564_V12524t6_.jpg": ("resource/demo/example/condition/person/baumu30483223c3_1719437121402_2-0._QL90_UX564_V12524t6_.jpg", "9373d531e44e4cf05cba87e6a80e74bbb0aba992e1432486bb1485be13d26b63"),
}


def cases(assets, source):
    male, female = assets / "model_5.png", assets / "model_8.png"
    top, dress = assets / "21514384_52353349_1000.jpg", assets / "21744571_51588794_1000.jpg"
    worn = assets / "baumu30483223c3_1719437121402_2-0._QL90_UX564_V12524t6_.jpg"
    return [
        ("flat-top-free", male, top, "tops", "flat-lay", False, 42),
        ("flat-top-masked", male, top, "tops", "flat-lay", True, 42),
        ("flat-dress-free", female, dress, "one-pieces", "flat-lay", False, 42),
        ("worn-bottoms-free", female, source / "examples" / "data" / "model.webp", "bottoms", "model", False, 42),
        ("worn-top-free", female, worn, "tops", "model", False, 42),
        ("flat-top-seed43", male, top, "tops", "flat-lay", False, 43),
    ]


def write_report(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def validate_assets(directory, download):
    directory.mkdir(parents=True, exist_ok=True)
    for name, (relative, expected) in ASSETS.items():
        path = directory / name
        if not path.exists() and download:
            url = f"https://raw.githubusercontent.com/Zheng-Chong/CatVTON/{REVISION}/{relative}"
            with urllib.request.urlopen(url, timeout=60) as response:
                data = response.read(8 * 1024 * 1024 + 1)
            import hashlib
            if len(data) > 8 * 1024 * 1024 or hashlib.sha256(data).hexdigest() != expected:
                raise ValueError("Downloaded asset hash/size mismatch: " + name)
            path.write_bytes(data)
        if not path.is_file() or sha256(path) != expected:
            raise ValueError("Missing or changed licensed fixture: " + str(path))


def contact_sheet(person, garment, prepared, result, destination, title):
    sheet = Image.new("RGB", (1200, 520), "white")
    draw = ImageDraw.Draw(sheet)
    for index, (label, path) in enumerate((("Person", person), ("Garment", garment),
                                          ("Prepared person", prepared), ("Native output", result))):
        draw.text((index * 300 + 5, 25), label, fill="black")
        with Image.open(path) as image:
            sheet.paste(ImageOps.contain(image.convert("RGB"), (290, 450)), (index * 300, 50))
    draw.text((5, 5), title, fill="black")
    draw.text((5, 503), "CatVTON demo inputs: CC BY-NC-SA 4.0; local research evaluation. FASHN example where recorded.", fill="black")
    sheet.save(destination)


def run(args):
    if not args.accept_noncommercial_demo_license or not args.accept_parser_research_license:
        raise ValueError("Explicit demo CC-BY-NC-SA and parser research/evaluation consent are required")
    validate_sampling(args.steps, 0, 1.5, 1.5, 1)
    if args.threads < 1:
        raise ValueError("Threads must be positive")
    if args.max_new_cases < 0:
        raise ValueError("Case limit must be nonnegative")
    validate_assets(args.assets, args.download)
    selected = cases(args.assets, args.source)
    fingerprint = {
        "cli_sha256": sha256(args.cli), "preparer_sha256": sha256(args.preparer),
        "checkpoint_sha256": sha256(args.checkpoint), "steps": args.steps, "threads": args.threads,
        "source_person_sha256": sha256(args.source / "examples" / "data" / "model.webp"),
        "pose_weights": {p.name: sha256(p) for p in sorted(args.dwpose_dir.glob("*.onnx"))},
        "parser_manifest_sha256": sha256(args.parser_dir / "manifest.json"),
    }
    report_path = args.output / "suite.json"
    if args.resume:
        report = json.loads(report_path.read_text(encoding="utf-8"))
        if report["fingerprint"] != fingerprint:
            raise ValueError("Cannot resume with changed executables, model, inputs or settings")
    else:
        args.output.mkdir(parents=True, exist_ok=False)
        report = {"fingerprint": fingerprint, "cases": {}, "execution_passed": False,
                  "scope": "Small unpaired exploratory regression set; not a statistical benchmark or paired ground truth.",
                  "visual_review": "pending", "source_repository": REPOSITORY, "source_revision": REVISION,
                  "demo_license": LICENSE, "assets": ASSETS}
    report["expected_cases"] = [row[0] for row in selected]
    write_report(report_path, report)
    generated = 0
    for name, person, garment, category, photo, masked, seed in selected:
        definition = {
            "category": category, "photo_type": photo, "masked_person": masked, "seed": seed,
            "person_sha256": sha256(person), "garment_sha256": sha256(garment)}
        entry = report["cases"].setdefault(name, {**definition, "status": "pending"})
        if any(entry.get(key) != value for key, value in definition.items()):
            raise ValueError("Cannot resume changed case definition: " + name)
        prepared = args.output / ("prepared-flat-top-free" if name == "flat-top-seed43" else "prepared-" + name)
        if not (prepared / "manifest.json").is_file():
            command = [str(args.preparer), "--dwpose-dir", str(args.dwpose_dir),
                       "--person-image", str(person), "--garment-image", str(garment), "--category", category,
                       "--garment-photo-type", photo, "--output", str(prepared)]
            if masked or photo == "model":
                command += ["--parser-dir", str(args.parser_dir), "--accept-parser-research-license"]
            if masked:
                command.append("--no-segmentation-free")
            with (args.output / (name + "-prepare.log")).open("w", encoding="utf-8") as log:
                subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True, timeout=180)
        provenance = json.loads((prepared / "provenance.json").read_text(encoding="utf-8"))
        if provenance["inputs_sha256"] != {"person": sha256(person), "garment": sha256(garment)}:
            raise ValueError("Prepared source images changed: " + name)
        for filename, digest in provenance["artifacts"].items():
            if Path(filename).name != filename or sha256(prepared / filename) != digest:
                raise ValueError("Prepared artifact changed: " + filename)
        digest = sha256(prepared / "manifest.json")
        if entry.get("prepared_manifest_sha256", digest) != digest:
            raise ValueError("Prepared manifest changed: " + name)
        entry["prepared_manifest_sha256"] = digest
        if args.prepare_only:
            if entry["status"] != "completed":
                entry["status"] = "prepared"
            write_report(report_path, report)
            continue
        destination = args.output / name
        if entry["status"] == "completed":
            if sha256(destination / "sample-0.png") != entry["output_sha256"]:
                raise ValueError("Completed output was changed: " + name)
            continue
        entry["status"] = "running"
        write_report(report_path, report)
        print("Starting quality case: " + name, flush=True)
        if destination.exists():
            archive = destination.with_name(name + "-interrupted-" + str(time.time_ns()))
            destination.rename(archive)
            entry.setdefault("previous_attempts", []).append(archive.name)
            write_report(report_path, report)
        benchmark(SimpleNamespace(cli=args.cli, checkpoint=args.checkpoint, prepared=prepared / "manifest.json",
                                  output=destination, steps=args.steps, cfg=1.5, shift=1.5, skip_cfg_last_n_steps=1,
                                  seed=seed, threads=args.threads, repeats=1, timeout=14400,
                                  flash_attention=True, weight_type="bf16"))
        measured = json.loads((destination / "benchmark.json").read_text(encoding="utf-8"))
        if measured["executable_sha256"] != fingerprint["cli_sha256"] or measured["checkpoint_sha256"] != fingerprint["checkpoint_sha256"]:
            raise ValueError("Executable or checkpoint changed during suite execution")
        entry.update(status="completed", output_sha256=sha256(destination / "sample-0.png"),
                     wall_seconds=measured["runs"][0]["wall_seconds"],
                     peak_working_set_bytes=measured["runs"][0]["peak_working_set_bytes"])
        contact_sheet(person, garment, prepared / "ca_image.png", destination / "sample-0.png",
                      args.output / (name + "-review.jpg"), name)
        write_report(report_path, report)
        print("Completed quality case: " + name, flush=True)
        generated += 1
        if args.max_new_cases and generated >= args.max_new_cases:
            break
    report["execution_passed"] = all(report["cases"].get(row[0], {}).get("status") == "completed" for row in selected)
    write_report(report_path, report)
    cards = "".join(f"<h2>{html.escape(name)}</h2><img width='1200' src='{name}-review.jpg'>"
                    for name in report["expected_cases"] if report["cases"].get(name, {}).get("status") == "completed")
    (args.output / "review.html").write_text(
        "<!doctype html><meta charset='utf-8'><title>FASHN research quality review</title>"
        "<h1>Exploratory, unpaired quality review</h1><p>Execution success is not a quality score. "
        "Review garment color/pattern, fit, face/pose, hands, background and boundary artifacts. "
        "CatVTON demo inputs are CC BY-NC-SA 4.0; preserve attribution and research-use restrictions.</p>" + cards,
        encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("cli", "preparer", "checkpoint", "source", "assets", "dwpose-dir", "parser-dir", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--max-new-cases", type=int, default=0, help="Generate at most N unfinished cases in this invocation; zero runs all")
    for flag in ("prepare-only", "resume", "download", "accept-noncommercial-demo-license", "accept-parser-research-license"):
        parser.add_argument("--" + flag, action="store_true")
    args = parser.parse_args()
    for name, value in vars(args).items():
        if isinstance(value, Path):
            setattr(args, name, value.resolve())
    run(args)
