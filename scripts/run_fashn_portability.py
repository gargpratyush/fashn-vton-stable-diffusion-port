"""Run an architecture-checked Windows FASHN portability job; Python standard library only."""
import argparse
import json
from pathlib import Path
import struct
import subprocess
import time

from package_fashn_portability import BASE_COMMIT, GGML_COMMIT, digest, git, verify


def pe_machine(path):
    with Path(path).open("rb") as stream:
        if stream.read(2) != b"MZ":
            raise ValueError("Expected a Windows PE executable")
        stream.seek(0x3C)
        offset = struct.unpack("<I", stream.read(4))[0]
        if offset > Path(path).stat().st_size - 6:
            raise ValueError("Invalid PE offset")
        stream.seek(offset)
        if stream.read(4) != b"PE\0\0":
            raise ValueError("Invalid PE signature")
        return struct.unpack("<H", stream.read(2))[0]


def run(args):
    bundle = args.bundle.resolve()
    manifest = verify(bundle)
    repo = Path(__file__).resolve().parent.parent
    if git(repo, "rev-parse", "HEAD") != BASE_COMMIT or git(repo / "ggml", "rev-parse", "HEAD") != GGML_COMMIT:
        raise ValueError("Unexpected source baseline; use the documented checkout plus overlay")
    for row in manifest["files"]:
        prefix = "source-overlay/"
        if row["path"].startswith(prefix):
            source = repo.joinpath(*row["path"][len(prefix):].split("/"))
            if not source.is_file() or digest(source) != row["sha256"]:
                raise ValueError("Source overlay not applied exactly: " + row["path"])
    binary = args.bin_dir / ("sd-cli.exe" if args.public_cli else "test-fashn-vton-trajectory.exe")
    machine = pe_machine(binary)
    expected = {"arm64": 0xAA64, "x64": 0x8664}[args.machine]
    if machine != expected:
        raise ValueError(f"Expected {args.machine} PE, found 0x{machine:04X}")
    if args.threads < 1 or args.threads > 128:
        raise ValueError("Threads must be in [1,128]")
    if args.public_cli and args.noise != "cpu":
        raise ValueError("Public CLI uses CPU RNG; imported noise is diagnostic only")
    model = bundle / "weights" / "model-bf16.gguf"
    prepared = bundle / "cases" / args.case / "prepared" / "manifest.json"
    metadata = json.loads(prepared.read_text(encoding="utf-8"))
    category = {"tops": 1, "bottoms": 2, "one-pieces": 3}[metadata["category"]]
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    if args.public_cli:
        command = [str(binary), "--mode", "try_on", "--diffusion-model", str(model),
                   "--try-on-inputs", str(prepared), "--type", "bf16", "--diffusion-fa",
                   "--rng", "cpu", "--steps", str(args.steps), "--cfg-scale", "1.5",
                   "--flow-shift", "1.5", "--skip-cfg-last-n-steps", "1",
                   "--seed", "42", "-t", str(args.threads), "-o", str(output / "sample.png"),
                   "--model-args", '{"fashn_modulation_cache":true,"fashn_modulation_cache_mib":128,"fashn_fused_gelu":false}']
    else:
        command = [str(binary), str(model), str(bundle / "cases" / args.case / "conditions.safetensors"),
                   str(output / "trajectory"), str(args.steps), "1.5", "1.5", "1", "42",
                   str(category), str(args.threads), "--matrix-type", "bf16", "--upcast-matrices",
                   "--mmap", "off", "--load-threads", "2", "--precompute-modulations",
                   "--modulation-cache-mib", "128", "--memory-profile"]
        if args.noise == "reference":
            command += ["--initial-noise", str(bundle / "references" / f"black-shirt-{args.steps}step" / "initial.safetensors")]
    record = {"status": "running", "machine": f"0x{machine:04X}", "case": args.case,
              "binary_sha256": digest(binary), "bundle_manifest_sha256": digest(bundle / "bundle.json"),
              "command": command, "steps": args.steps, "threads": args.threads,
              "noise_source": args.noise, "public_cli": args.public_cli}
    report = output / "run.json"
    report.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    start = time.perf_counter()
    with (output / "run.log").open("w", encoding="utf-8") as log:
        completed = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=False)
    record.update(status="completed" if completed.returncode == 0 else "failed",
                  exit_code=completed.returncode, wall_seconds=time.perf_counter() - start)
    report.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(record, indent=2))
    return completed.returncode


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--bin-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--machine", choices=("arm64", "x64"), default="arm64")
    parser.add_argument("--case", choices=("black-shirt", "cardigan", "bottoms", "dress"), default="black-shirt")
    parser.add_argument("--steps", type=int, choices=(2, 20), default=2)
    parser.add_argument("--threads", type=int, default=6)
    parser.add_argument("--noise", choices=("cpu", "reference"), default="cpu")
    parser.add_argument("--public-cli", action="store_true")
    raise SystemExit(run(parser.parse_args()))


if __name__ == "__main__":
    main()
