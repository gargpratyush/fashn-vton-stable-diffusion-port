"""Publish selected local FASHN reports without weights, binaries or large trajectories.

Numeric results and image bytes are preserved. Local links and metadata paths
are normalized; the manifest records both original and publication hashes.
"""
import argparse
import hashlib
import html
import json
import os
from pathlib import Path
import re
from urllib.parse import quote, unquote, urlsplit


def digest(data):
    return hashlib.sha256(data).hexdigest()


def package(args):
    repo = args.repository.resolve()
    reference = args.reference_root.resolve()
    plan = args.integration_plan.resolve()
    output = repo / "reports"
    if not (output / "README.md").is_file() or not (output / "NOTICE.md").is_file():
        raise ValueError("The publication README and licensing notice must exist first")
    selected = {}

    def select(source, relative):
        source = source.resolve()
        if not source.is_file():
            raise FileNotFoundError(source)
        destination = output / relative
        if destination in selected.values():
            raise ValueError("Duplicate publication destination: " + str(destination))
        selected[source] = destination

    names = [
        "implementation-and-experiment-history.md", "quantization-and-upstream-comparison.md",
        "memory-and-latency-optimization-plan.md", "memory-and-latency-optimization-results.md",
        "fashn-vton-project-history.html", "full-image-precision-summary.json",
        "optimization-gallery-spec.json",
        "checkpoint31-memory-attribution.md", "checkpoint32-runtime-ready-weights.md",
        "checkpoint33-modulation-progress.md", "checkpoint34-gelu-progress.md",
        "checkpoint35-blas-progress.md", "checkpoint36-cfg-and-reuse.md",
        "checkpoint37-preparation-progress.md", "checkpoint38-integrated-progress.md",
    ]
    for name in names:
        select(reference / "reports" / name, name)
    select(plan, "original-integration-plan.md")
    select(reference / "source" / "LICENSE", "licenses/FASHN-Apache-2.0.txt")
    for name in (
        "checkpoint35_controls.py", "checkpoint35_full_acceptance.py",
        "checkpoint38_integrated.py", "finalize_fashn_optimizations.py",
        "probe_openblas_memory.py", "build_fashn_project_history.py",
        "verify_fashn_final_artifacts.py", "verify_fashn_project_history.py",
    ):
        select(plan.parent / name, Path("reproduction") / (name + ".txt"))
    gallery = reference / "reports" / "comparison-gallery-optimized"
    for path in sorted(gallery.rglob("*")):
        if path.is_file():
            if path.suffix not in (".png", ".jpg", ".json", ".html"):
                raise ValueError("Unexpected gallery payload: " + str(path))
            select(path, Path("comparison-gallery-optimized") / path.relative_to(gallery))
    ranking = reference / "reports" / "matrix-sensitivity-complete"
    for path in sorted(ranking.iterdir()):
        if path.is_file() and path.suffix in (".json", ".csv", ".md"):
            select(path, Path("matrix-sensitivity-complete") / path.name)
    for relative in (
        "checkpoint38-integrated/results.json",
        "checkpoint38-integrated/final-summary.json",
        "checkpoint38-integrated/openblas-dll-memory.json",
        "checkpoint35-full-acceptance/results.json",
        "checkpoint37-server-budget-soak/report.json",
        "checkpoint37-ort-comparison.json",
        "checkpoint29-policy-evaluation/evaluation.json",
    ):
        select(reference / relative, Path("evidence") / relative)

    destinations = set(selected.values()) | {output / "README.md", output / "NOTICE.md"}
    directories = {parent for path in destinations for parent in path.parents if parent == output or output in parent.parents}
    original_roots = {
        "c:/source/stable-diffusion.cpp": repo,
        "c:/source/fashn-vton-reference": reference,
        str(repo).replace("\\", "/").lower(): repo,
        str(reference).replace("\\", "/").lower(): reference,
    }

    def resolve_local(value, base):
        if value.startswith("file:"):
            value = unquote(urlsplit(value).path)
            if re.match(r"^/[A-Za-z]:/", value):
                value = value[1:]
        value = value.replace("\\", "/")
        lowered = value.lower()
        for root, actual in original_roots.items():
            if lowered == root or lowered.startswith(root + "/"):
                return (actual / value[len(root):].lstrip("/")).resolve()
        if value.endswith("/fashn-vton-1.5-integration-plan.md"):
            return plan
        if re.match(r"^[A-Za-z]:/", value):
            return None
        return (base / value).resolve()

    def publication_target(source):
        if source in selected:
            return selected[source]
        if source is not None and source.is_relative_to(reference / "reports"):
            target = output / source.relative_to(reference / "reports")
            if target in destinations or target in directories:
                return target
        if source is not None and source.is_relative_to(repo):
            relative = source.relative_to(repo)
            if relative.parts and (relative.parts[0].startswith(("build", ".")) or
                                   relative.parts[0] in ("models", "outputs")):
                return None
            if source.exists():
                return source
        return None

    def relative_url(target, destination):
        return quote(Path(os.path.relpath(target, destination.parent)).as_posix(), safe="/.-_")

    def clean_home(value):
        return re.sub(r"(?:[A-Za-z]:)?[\\/]+Users[\\/]+[^\\/\s\"<]+", "<USER_HOME>", value, flags=re.I)

    def metadata(value):
        if isinstance(value, dict):
            return {metadata(key): metadata(item) for key, item in value.items()}
        if isinstance(value, list):
            return [metadata(item) for item in value]
        if isinstance(value, str):
            for spelling, replacement in (
                (r"C:\source\stable-diffusion.cpp", "repository"),
                (r"C:\source\fashn-vton-reference", "local-experiment"),
            ):
                value = re.sub(re.escape(spelling.replace("\\", "/")), replacement, value, flags=re.I)
                value = re.sub(re.escape(spelling), replacement, value, flags=re.I)
            return clean_home(value)
        return value

    omitted = set()

    def html_copy(value, source, destination):
        def anchor(match):
            before, raw_href, after, body = match.groups()
            href = html.unescape(raw_href)
            if href.startswith(("#", "http://", "https://", "data:", "mailto:")):
                return match.group(0)
            local = resolve_local(href, source.parent)
            target = publication_target(local)
            if target is None:
                omitted.add(metadata(href))
                return ('<span class="local-only-artifact" title="Historical local artifact; not bundled">'
                        + body + " [local-only artifact]</span>")
            return "<a" + before + 'href="' + html.escape(relative_url(target, destination), quote=True) + '"' + after + ">" + body + "</a>"
        value = re.sub(r'<a\b([^>]*?)href="([^"]*)"([^>]*)>(.*?)</a>', anchor, value, flags=re.S)
        value = re.sub(r"(?:[A-Za-z]:)?[\\/]+Users[\\/]+[^\\/\s\"<]+", "&lt;USER_HOME&gt;", value, flags=re.I)
        notice = ('<aside class="callout" style="padding:16px;border-left:4px solid #087f8c;background:#edf7f8">'
                  '<strong>Repository publication copy.</strong> Included source/report links are relative; omitted local artifacts are labeled. '
                  'Historical home paths are normalized. Images and numeric results are unchanged. '
                  '<a href="' + relative_url(output / "README.md", destination) + '">Publication scope</a> | '
                  '<a href="' + relative_url(output / "NOTICE.md", destination) + '">Image/license notices</a>.</aside>')
        if '<main id="main">' in value:
            value = value.replace('<main id="main">', '<main id="main">' + notice, 1)
        else:
            value = value.replace("<h1>", notice + "<h1>", 1)
        return value

    def markdown_copy(value, source, destination):
        def replace(match):
            label, href = match.groups()
            if href.startswith(("http://", "https://", "#", "mailto:")):
                return match.group(0)
            target = publication_target(resolve_local(href, source.parent))
            if target is None:
                omitted.add(metadata(href))
                return label + " (historical local-only artifact)"
            return "[" + label + "](" + relative_url(target, destination) + ")"
        return clean_home(re.sub(r"\[([^\]]+)\]\(([^)]+)\)", replace, value))

    manifest = {"schema": "fashn-report-publication-v1", "files": {},
                "scope": "Selected reports, rendered images and numeric summaries; no weights, binaries or full trajectory tensors.",
                "path_labels": {"repository": "This source checkout", "local-experiment": "Original experiment tree, not bundled",
                                "<USER_HOME>": "Redacted historical home-directory prefix"}}
    snapshots = {}
    for source, destination in selected.items():
        data = source.read_bytes()
        snapshots[source] = digest(data)
        suffix = source.suffix
        if suffix == ".html":
            published = html_copy(data.decode("utf-8-sig"), source, destination).encode("utf-8")
        elif suffix == ".md":
            published = markdown_copy(data.decode("utf-8-sig"), source, destination).encode("utf-8")
        elif suffix == ".json":
            published = (json.dumps(metadata(json.loads(data.decode("utf-8-sig"))), indent=2, allow_nan=False) + "\n").encode()
        elif suffix == ".py":
            published = clean_home(data.decode("utf-8-sig")).encode("utf-8")
        else:
            published = data
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(published)
        origin = ("local-experiment/" + source.relative_to(reference).as_posix()
                  if source.is_relative_to(reference) else "session-artifacts/" + source.name)
        manifest["files"][destination.relative_to(repo).as_posix()] = {
            "origin": origin, "original_sha256": digest(data), "published_sha256": digest(published),
            "bytes": len(published), "image_bytes_unchanged": data == published if suffix in (".png", ".jpg") else None}
    for source, expected in snapshots.items():
        if digest(source.read_bytes()) != expected:
            raise RuntimeError("Source changed during publication: " + str(source))
    manifest["omitted_local_links"] = sorted(omitted)
    (output / "publication-manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"published_files": len(selected), "bytes": sum(v["bytes"] for v in manifest["files"].values()),
                      "omitted_local_links_labeled": len(omitted)}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--reference-root", type=Path, required=True)
    parser.add_argument("--integration-plan", type=Path, required=True)
    package(parser.parse_args())
