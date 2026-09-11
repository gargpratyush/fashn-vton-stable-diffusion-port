"""Assemble or verify the private, hash-checked FASHN device-transfer bundle."""
import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import shutil
import subprocess


BASE_COMMIT = "55db93d6f2536f12428e170acbafd0cca3182c59"
GGML_COMMIT = "e20c3a14aa70ee84ca58499814206dd08d8026bc"
MODEL_SHA256 = "e1f2d13d441f0e4631cf4cf8e1837cd8998e4916b78dfc0ec62c5fe879d087b1"
OVERLAY = (
    "docs/fashn_android_plan.md", "docs/fashn_arm_transfer.md", "docs/fashn_quickstart.md",
    "tests/fashn_noise_fixture.h", "tests/test_fashn_vton_trajectory.cpp",
    "scripts/package_fashn_portability.py", "scripts/run_fashn_portability.py",
    "scripts/test_fashn_portability.py",
    "reports/android-host-preparation.md",
)
IMAGE_KEYS = ("ca_image", "garment_image", "person_pose", "garment_pose")
ANDROID_BINARIES = (
    "sd-cli", "test-fashn-vton", "test-fashn-vton-c-api", "test-fashn-vton-graph",
    "test-fashn-vton-sampling", "test-fashn-vton-precision", "test-fashn-gelu",
    "test-fashn-math", "test-fashn-modulation", "test-fashn-vton-trajectory",
)


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def contained(root, relative):
    part = PurePosixPath(relative)
    if (not relative or part.is_absolute() or ".." in part.parts or "\\" in relative
            or ":" in relative or part.as_posix() != relative):
        raise ValueError("Unsafe bundle path: " + relative)
    root = Path(root).resolve()
    path = root.joinpath(*part.parts).resolve()
    if not path.is_relative_to(root):
        raise ValueError("Bundle path escapes its root: " + relative)
    return path


def git(repo, *args):
    return subprocess.check_output(["git", "-C", str(repo), *args], text=True).strip()


def required_paths():
    paths = {"weights/model-bf16.gguf", "NOTICE.md"} | {"source-overlay/" + name for name in OVERLAY}
    for case in ("black-shirt", "cardigan", "bottoms", "dress"):
        paths |= {f"cases/{case}/conditions.safetensors", f"cases/{case}/prepared/manifest.json"}
        paths |= {f"cases/{case}/prepared/{name}.png" for name in IMAGE_KEYS}
    for steps in (2, 20):
        prefix = f"references/black-shirt-{steps}step/"
        paths |= {prefix + name for name in ("manifest.json", "expected.png", "initial.safetensors")}
        paths |= {prefix + f"step-{index}.safetensors" for index in range(steps)}
    return paths


def verify(root):
    root = Path(root)
    manifest = json.loads((root / "bundle.json").read_text(encoding="utf-8"))
    if (manifest.get("schema") != "fashn-portability-v1"
            or manifest.get("base_commit") != BASE_COMMIT
            or manifest.get("ggml_commit") != GGML_COMMIT
            or not manifest.get("files")):
        raise ValueError("Invalid or incompatible bundle manifest")
    seen = set()
    for row in manifest["files"]:
        name = row["path"]
        if name in seen:
            raise ValueError("Duplicate bundle path: " + name)
        seen.add(name)
        path = contained(root, name)
        if not path.is_file() or path.stat().st_size != row["bytes"] or digest(path) != row["sha256"]:
            raise ValueError("Bundle integrity failure: " + name)
    if not required_paths().issubset(seen):
        raise ValueError("Bundle is missing required files")
    if manifest.get("total_bytes") != sum(row["bytes"] for row in manifest["files"]):
        raise ValueError("Bundle byte count mismatch")
    model = next((r for r in manifest["files"] if r["path"] == "weights/model-bf16.gguf"), None)
    if not model or model["sha256"] != MODEL_SHA256:
        raise ValueError("Unexpected runtime-ready model identity")
    return manifest


def assemble(repo, experiment, output, android_bin=None, evidence=None):
    repo, experiment, output = Path(repo), Path(experiment), Path(output)
    if git(repo, "rev-parse", "HEAD") != BASE_COMMIT:
        raise ValueError("Build this bundle from the documented base commit")
    if git(repo / "ggml", "rev-parse", "HEAD") != GGML_COMMIT or git(repo / "ggml", "status", "--porcelain"):
        raise ValueError("Unexpected or modified GGML checkout")
    output.mkdir(parents=True, exist_ok=False)
    files = []

    def copy(source, relative, role):
        target = contained(output, relative)
        target.parent.mkdir(parents=True, exist_ok=True)
        before = digest(source)
        shutil.copyfile(source, target)
        if digest(target) != before:
            raise ValueError("Copy verification failed: " + relative)
        files.append({"path": relative, "bytes": target.stat().st_size, "sha256": before, "role": role})

    model = experiment / "checkpoint32-runtime-weights" / "bf16" / "model.gguf"
    if digest(model) != MODEL_SHA256:
        raise ValueError("Runtime-ready floating model changed")
    copy(model, "weights/model-bf16.gguf", "runtime-weight")
    cases = {
        "black-shirt": ("parser-checkpoint9/prepared", "parser-checkpoint9/prepared"),
        "cardigan": ("checkpoint22-quality-t16/prepared-flat-top-free",
                     "checkpoint22-quality-reference-preparation/flat-top-free"),
        "bottoms": ("checkpoint22-quality-t16/prepared-worn-bottoms-free",
                    "checkpoint22-quality-reference-preparation/worn-bottoms-free"),
        "dress": ("checkpoint22-quality-t16/prepared-flat-dress-free",
                  "checkpoint22-quality-reference-preparation/flat-dress-free"),
    }
    for name, (prepared_dir, conditions_dir) in cases.items():
        prepared_root = contained(experiment, prepared_dir)
        prepared = json.loads((prepared_root / "manifest.json").read_text(encoding="utf-8"))
        copy(prepared_root / "manifest.json", f"cases/{name}/prepared/manifest.json", "prepared-input")
        for key in IMAGE_KEYS:
            source = contained(prepared_root, prepared[key])
            copy(source, f"cases/{name}/prepared/{prepared[key]}", "prepared-input")
        copy(contained(experiment, conditions_dir) / "conditions.safetensors",
             f"cases/{name}/conditions.safetensors", "normalized-conditions")
    for steps, source_dir in ((2, "parser-checkpoint9/oracle-two-step"),
                              (20, "checkpoint21-reference-20-step")):
        source = contained(experiment, source_dir)
        manifest = json.loads((source / "manifest.json").read_text(encoding="utf-8"))
        if (manifest["steps"] != steps or manifest["input_kind"] != "prepared"
                or manifest["inputs_sha256"] != digest(output / "cases/black-shirt/conditions.safetensors")):
            raise ValueError("Reference input lineage mismatch")
        required = {"initial.safetensors", "expected.png"} | {f"step-{i}.safetensors" for i in range(steps)}
        if set(manifest["artifacts"]) != required:
            raise ValueError("Incomplete reference trajectory")
        copy(source / "manifest.json", f"references/black-shirt-{steps}step/manifest.json", "oracle-manifest")
        for name in sorted(required):
            if digest(source / name) != manifest["artifacts"][name]:
                raise ValueError("Reference artifact changed: " + name)
            copy(source / name, f"references/black-shirt-{steps}step/{name}", "oracle-output")
    for case in ("cardigan", "bottoms"):
        for name in ("full.png", "sample-0.png", "result.json"):
            copy(experiment / f"checkpoint27-upstream-{case}" / name,
                 f"references/{case}-upstream/{name}", "historical-upstream-image-comparison")
    for name in OVERLAY:
        copy(contained(repo, name), "source-overlay/" + name, "source-overlay")
    copy(repo / "reports" / "NOTICE.md", "NOTICE.md", "research-and-image-notice")
    if android_bin:
        for name in ANDROID_BINARIES:
            source = Path(android_bin) / name
            with source.open("rb") as stream:
                header = stream.read(20)
            if header[:6] != b"\x7fELF\x02\x01" or int.from_bytes(header[18:20], "little") != 183:
                raise ValueError("Expected ELF64 little-endian AArch64: " + name)
            copy(source, "android/bin/" + name, "android-cross-built-not-device-validated")
    if evidence:
        for source in sorted(Path(evidence).rglob("*")):
            if source.is_file() and source.suffix in (".json", ".jsonl", ".log", ".png"):
                name = source.relative_to(evidence).as_posix()
                copy(source, "host-evidence/" + name, "private-host-evidence")
    manifest = {
        "schema": "fashn-portability-v1", "base_commit": BASE_COMMIT, "ggml_commit": GGML_COMMIT,
        "model_policy": "BF16 core/F32 protected; compute F32; CPU only",
        "scope": "Private research transfer; prepared inputs; no parser/pose weights or Python required for native inference",
        "files": files, "total_bytes": sum(row["bytes"] for row in files),
        "source_state": "Recursive clone at base_commit plus exact source-overlay; overlay changes are not committed",
        "android_execution": "not established; physical-device gates remain open",
    }
    (output / "bundle.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    verify(output)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    create = commands.add_parser("create")
    for name in ("repo", "experiment", "output"):
        create.add_argument("--" + name, type=Path, required=True)
    create.add_argument("--android-bin", type=Path)
    create.add_argument("--evidence", type=Path)
    check = commands.add_parser("verify")
    check.add_argument("bundle", type=Path)
    args = parser.parse_args()
    result = (assemble(args.repo, args.experiment, args.output, args.android_bin, args.evidence)
              if args.command == "create" else verify(args.bundle))
    print(json.dumps({"files": len(result["files"]), "bytes": result["total_bytes"], "verified": True}))


if __name__ == "__main__":
    main()
