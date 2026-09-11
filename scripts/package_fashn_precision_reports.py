"""Publish the completed precision study and README contact sheets without model payloads."""
import argparse
import io
import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont, ImageOps

from build_fashn_comparison_report import portable
from run_fashn_q45_study import read_json, sha256, write_report


def contact_sheet(paths, labels, title):
    if len(paths) != 10 or len(labels) != 10:
        raise ValueError("Expected two rows of five labeled images")
    width, height, margin, header = 250, 365, 14, 65
    canvas = Image.new("RGB", (5 * width + 6 * margin, 2 * height + 3 * margin + header), "#edf3f7")
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default(size=17)
    draw.text((margin, 10), title, fill="#142839", font=ImageFont.load_default(size=22))
    draw.text((margin, 39), "Actual generated outputs (not error maps). Seed 42. See report notices for attribution.", fill="#405366", font=font)
    for index, (path, label) in enumerate(zip(paths, labels)):
        x = margin + (index % 5) * (width + margin)
        y = header + margin + (index // 5) * (height + margin)
        draw.rectangle((x, y, x + width, y + height), fill="white")
        draw.text((x + 8, y + 6), label, fill="#142839", font=font)
        with Image.open(path) as image:
            scaled = ImageOps.contain(image.convert("RGB"), (width, height - 34), Image.Resampling.LANCZOS)
        canvas.paste(scaled, (x + (width - scaled.width) // 2, y + 34))
    stream = io.BytesIO()
    canvas.save(stream, format="PNG")
    return stream.getvalue()


def package(repo, experiment):
    repo, experiment = repo.resolve(), experiment.resolve()
    study = experiment / "q45-study"
    report_dir = repo / "reports"
    destination = report_dir / "comparison-gallery-precision"
    destination.mkdir(parents=True, exist_ok=True)
    state = read_json(study / "results.json")
    if not state["complete"] or len(state["jobs"]) != 36 or any(row["status"] != "done" for row in state["jobs"].values()):
        raise ValueError("Only the complete study can be published")
    entries = {}

    def publish(source, target, transform=None):
        data = source.read_bytes()
        if source.suffix.lower() not in (".json", ".png", ".html"):
            raise ValueError("Unexpected publication payload")
        published = transform(data) if transform else data
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(published)
        entries[target.relative_to(repo).as_posix()] = {
            "origin": "local-experiment/" + source.relative_to(experiment).as_posix(),
            "original_sha256": sha256(source), "published_sha256": sha256(target),
            "bytes": len(published), "unchanged": data == published}

    roots = [(experiment, "local-experiment"), (repo, "repository")]

    def normalized(data):
        return (json.dumps(portable(json.loads(data), roots), indent=2, allow_nan=False) + "\n").encode("utf-8")

    publish(study / "fashn-all-comparisons.html", report_dir / "fashn-all-comparisons.html")
    for name in ("results.json", "q8-vs-lowbit-pixel-differences.json"):
        publish(study / name, report_dir / "evidence" / "q45-study" / name, normalized)
    policies = ("bf16", "q8_0", "q4_0", "q4_K", "q5_0", "q5_K", "selected-mixed")
    for case in ("cardigan", "bottoms"):
        images = {}
        for policy in policies:
            label = f"full-{case}-{policy}-s42"
            source = study / "runs" / label / "sample.png"
            expected = state["jobs"][label]["artifacts"][source.relative_to(study).as_posix()]
            if sha256(source) != expected:
                raise ValueError("Generated image hash mismatch: " + label)
            target = destination / "images" / f"{case}-{policy}.png"
            publish(source, target)
            images[policy] = target
        for kind in ("ca_image", "garment_image", "python"):
            source = study / "assets" / f"{case}-{kind}.png"
            target = destination / "images" / source.name
            publish(source, target)
            images[kind] = target
        order = ("ca_image", "garment_image", "python", "bf16", "q8_0",
                 "q4_0", "q4_K", "q5_0", "q5_K", "selected-mixed")
        labels = ("Prepared person", "Prepared garment", "Original Python", "BF16 / F32", "Q8_0",
                  "Q4_0", "Q4_K", "Q5_0", "Q5_K", "Q5_K + 4 F32")
        target = destination / f"{case}-overview.png"
        target.write_bytes(contact_sheet([images[key] for key in order], labels,
                                         "FASHN precision comparison: " + case))
        entries[target.relative_to(repo).as_posix()] = {
            "origin": "Derived labeled contact sheet; thumbnails only, metrics use unresized originals",
            "published_sha256": sha256(target), "bytes": target.stat().st_size,
            "source_images": [path.relative_to(repo).as_posix() for path in [images[key] for key in order]]}
    manifest = {"schema": "fashn-precision-publication-v1", "files": entries,
                "scope": "Current study HTML, exact output PNGs, resized labeled contact sheets and path-normalized numerical evidence. No weights, executables or trajectory tensor payloads.",
                "notice": "reports/NOTICE.md; imagery/adaptations have separate noncommercial attribution/share-alike requirements.",
                "study_results_sha256": sha256(study / "results.json")}
    write_report(report_dir / "precision-publication-manifest.json", manifest)
    print(json.dumps({"published_files": len(entries), "bytes": sum(row["bytes"] for row in entries.values())}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--experiment", type=Path, required=True)
    args = parser.parse_args()
    package(args.repo, args.experiment)
