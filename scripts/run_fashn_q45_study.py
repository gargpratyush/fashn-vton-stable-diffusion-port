"""Sequential, resumable Windows Q4/Q5 study with progressive numerical reports and images."""
import argparse
from contextlib import contextmanager
import html
import json
import os
from pathlib import Path
import shutil
import statistics
import time
import uuid
from types import SimpleNamespace

import numpy as np
from PIL import Image
from safetensors.numpy import load_file

from benchmark_fashn_vton import measure_command
from compare_fashn_case_output import crop_pixels
from compare_fashn_trajectories import canonical, compare, metrics, pixels
from export_fashn_runtime_weights import export
from fashn_artifacts import sha256, read_json, write_report, atomic_text, exclusive_lock
from fashn_metrics import distribution, pixel_metrics, worst_region


POLICIES = ("bf16", "q8_0", "q4_0", "q5_0", "q4_K", "q5_K")
LOWER = POLICIES[2:]
CASES = {"cardigan": ("flat-top-free", 1), "bottoms": ("worn-bottoms-free", 2)}
RESTORED = ["single_blocks.12.linear1.weight", "single_blocks.11.linear1.weight",
            "single_blocks.8.linear1.weight", "single_blocks.14.linear1.weight"]


def validate_execution(manifest, events, matrix_type, restored, kind, case, seed):
    expected_types = {name: ("f32" if name in restored else matrix_type) for name in manifest["matrix_types"]}
    expected = {"steps": 20, "cfg": 1.5, "shift": 1.5, "skip_cfg_last_n_steps": 1, "seed": seed,
                "threads": 16, "load_threads": 2, "category": CASES[case][1], "backend": "CPU",
                "attention": "F32 flash", "upcast_matrices": matrix_type == "bf16", "mmap": False,
                "precompute_modulations": True, "fused_gelu": True, "noise_source": "cpu_rng",
                "recorded_trajectory": kind == "full", "probe_forwards": 2 if kind == "probe" else 0}
    if (any(manifest[key] != value for key, value in expected.items()) or len(expected_types) != 104
            or manifest["matrix_types"] != expected_types or not restored.issubset(expected_types)
            or set(manifest["matrix_backends"]) != {"CPU"} or manifest["matrix_backends"]["CPU"] < 104):
        raise ValueError("Unexpected native sampling/precision/backend policy")
    if not events or not any(event["phase"] == "sampling_complete" for event in events):
        raise ValueError("Incomplete memory profile")
    if (max(event["conversion"]["converted_tensors"] for event in events if event["phase"] != "process_setup") != 0
            or any(event["mapped_model_files"] for event in events)):
        raise ValueError("Ready-weight run unexpectedly converted or mapped source weights")


def select_candidate(rows):
    scores = []
    for policy in LOWER:
        selected = [rows[f"full-{case}-{policy}-s42"] for case in CASES]
        if any(row["status"] != "done" for row in selected):
            raise ValueError("Candidate selection requires both full cases")
        scores.append((statistics.mean(row["cropped_vs_floating"]["rmse"] for row in selected),
                       statistics.mean(row["sampling_seconds"] for row in selected), policy))
    return min(scores)[2]


@contextmanager
def exclusive_run(root):
    if os.name != "nt":
        raise RuntimeError("This measurement study requires Windows process accounting")
    with exclusive_lock(root / "study.lock"):
        yield


def reference_view(source, destination, kind):
    manifest = read_json(source / "manifest.json")
    files = ["initial.safetensors"] + [f"step-{index}.safetensors" for index in range(manifest["steps"])]
    source_hashes = {name: sha256(source / name) for name in files}
    if destination.exists():
        existing = read_json(destination / "manifest.json")
        if existing.get("source_manifest_sha256") != sha256(source / "manifest.json"):
            raise ValueError("Reference view provenance changed")
        for name, digest in existing["artifacts"].items():
            if sha256(destination / name) != digest:
                raise ValueError("Reference view artifact changed")
        if any(existing["artifacts"][name] != digest for name, digest in source_hashes.items()):
            raise ValueError("Reference source tensor changed")
        return destination
    destination.mkdir(parents=True)
    for name in files:
        shutil.copyfile(source / name, destination / name)
    image = load_file(str(source / f"step-{manifest['steps'] - 1}.safetensors"))["image"]
    Image.fromarray(pixels(image)).save(destination / "expected.png")
    manifest["artifacts"] = {**source_hashes, "expected.png": sha256(destination / "expected.png")}
    manifest["reference_kind"] = kind
    manifest["source_manifest_sha256"] = sha256(source / "manifest.json")
    write_report(destination / "manifest.json", manifest)
    return destination


class Study:
    def __init__(self, args):
        self.args = args
        self.root = args.output.resolve()
        self.repo = args.repo.resolve()
        self.experiment = args.experiment.resolve()
        self.binary = self.repo / "build/bin/Release/test-fashn-vton-trajectory.exe"
        self.cli = self.binary.with_name("sd-cli.exe")
        self.verifier = self.binary.with_name("test-fashn-vton-conversion.exe")
        self.state_path = self.root / "results.json"
        self.weights = {}
        for policy in POLICIES:
            record = read_json(self.root / "ready" / policy / "export.json")
            if (not record["complete"] or record["matrix_type"] != policy
                    or sha256(Path(record["output"])) != record["output_sha256"]):
                raise ValueError("Missing or changed verified export: " + policy)
            if record["provenance_sha256"].get(str(self.verifier)) != sha256(self.verifier):
                raise ValueError("Export was not verified by the current verifier")
            self.weights[policy] = record
        inputs = [self.binary, self.cli, self.verifier, Path(__file__),
                  self.repo / "scripts/compare_fashn_trajectories.py",
                  self.repo / "scripts/benchmark_fashn_vton.py",
                  self.repo / "scripts/export_fashn_runtime_weights.py",
                  self.repo / "scripts/compare_fashn_case_output.py",
                  self.repo / "scripts/export_fashn_vton_reference.py",
                  self.repo / "scripts/run_fashn_quality.py",
                  self.repo / "scripts/fashn_artifacts.py",
                  self.repo / "scripts/fashn_metrics.py",
                  self.repo / "scripts/fashn_process_state.py",
                  self.repo / "docs/fashn_q4_q5_plan.md"]
        inputs += [self.repo / path for path in (
            "src/model/diffusion/fashn_vton.h", "src/convert.cpp", "src/model_loader.cpp",
            "src/stable-diffusion.cpp", "tests/test_fashn_vton_trajectory.cpp",
            "tests/fashn_test_runner.h", "tests/fashn_memory_profile.h",
            "tests/test_fashn_vton_conversion.cpp")]
        for case, (fixture, _) in CASES.items():
            prepared = self.prepared(case)
            inputs += [prepared, self.conditions(case)]
            metadata = read_json(prepared)
            for key in ("ca_image", "garment_image", "person_pose", "garment_pose"):
                inputs.append(prepared.parent / metadata[key])
            inputs += [self.experiment / f"checkpoint27-upstream-{case}" / name
                       for name in ("result.json", "sample-0.png")]
            historic = self.historical(case)
            inputs += [historic / "manifest.json", historic / "initial.safetensors"]
            inputs += [historic / f"step-{index}.safetensors" for index in range(20)]
        provenance = {"files": {str(path): sha256(path) for path in inputs},
                      "weights": {key: value["output_sha256"] for key, value in self.weights.items()},
                      "settings": {"threads": 16, "load_threads": 2, "steps": 20, "cfg": 1.5,
                                   "shift": 1.5, "skip_cfg": 1, "fused_gelu": True, "mmap": False}}
        if self.state_path.exists():
            if not args.resume:
                raise ValueError("Study exists; use --resume")
            self.state = read_json(self.state_path)
            if self.state.get("schema") != "fashn-q45-study-v2":
                raise ValueError("Frozen study: use its original checkout; new runs need a fresh v2 output directory")
            if self.state["provenance"] != provenance:
                raise ValueError("Study provenance changed; do not reuse prior measurements")
            for label, row in self.state["jobs"].items():
                if row["status"] != "done":
                    raise ValueError("Incomplete job requires explicit recovery after confirming its process ended: " + label)
                for relative, digest in row["artifacts"].items():
                    if sha256(self.root / relative) != digest:
                        raise ValueError("Completed job artifact changed: " + relative)
        else:
            self.state = {"schema": "fashn-q45-study-v2", "complete": False, "phase": "QP3", "active_job": None, "provenance": provenance,
                          "weights": self.weights, "jobs": {}, "scientific_scope": "Diagnostic only; no human approval",
                          "expected_probe_jobs": 18, "expected_full_jobs": 18}
        self.prepare_gallery_inputs()
        self.publish()

    def prepared(self, case):
        return self.experiment / "checkpoint22-quality-t16" / f"prepared-{CASES[case][0]}" / "manifest.json"

    def conditions(self, case):
        return self.experiment / "checkpoint22-quality-reference-preparation" / CASES[case][0] / "conditions.safetensors"

    def historical(self, case):
        return self.experiment / ("checkpoint38-integrated/cardigan-floating-cpu" if case == "cardigan"
                                  else "checkpoint35-full-acceptance/bottoms")

    def prepare_gallery_inputs(self):
        assets = self.root / "assets"
        assets.mkdir(exist_ok=True)
        for case in CASES:
            prepared = self.prepared(case)
            manifest = read_json(prepared)
            tensors = load_file(str(self.conditions(case)))
            for key, name in (("ca_image", "ca_images"), ("garment_image", "garment_images"),
                              ("person_pose", "person_poses"), ("garment_pose", "garment_poses")):
                with Image.open(prepared.parent / manifest[key]) as image:
                    array = np.asarray(image).astype(np.float32)
                normalized = array / np.float32(127.5) - np.float32(1)
                normalized = (normalized.transpose(2, 0, 1)[None] if array.ndim == 3 else normalized[None, None])
                if not np.array_equal(normalized, tensors[name]):
                    raise ValueError("Prepared pixel/float lineage mismatch: " + case)
            for key in ("ca_image", "garment_image"):
                shutil.copyfile(prepared.parent / manifest[key], assets / f"{case}-{key}.png")
            upstream = self.experiment / f"checkpoint27-upstream-{case}"
            record = read_json(upstream / "result.json")
            if (record["inputs_sha256"] != sha256(self.conditions(case))
                    or record["settings"] != {"steps": 20, "cfg": 1.5, "shift": 1.5,
                                               "skip_cfg_last_n_steps": 1, "seed": 42, "threads": 16}
                    or record["category"] != manifest["category"]
                    or record["prepared_manifest_sha256"] != sha256(prepared)
                    or any(record["prepared_images_sha256"][key] != sha256(prepared.parent / manifest[key])
                           for key in ("ca_image", "garment_image", "person_pose", "garment_pose"))
                    or record["crop"] != manifest["crop"]
                    or record["output_sha256"] != sha256(upstream / "sample-0.png")):
                raise ValueError("Historical upstream lineage mismatch: " + case)
            shutil.copyfile(upstream / "sample-0.png", assets / f"{case}-python.png")

    def publish(self):
        self.state["updated_unix_seconds"] = time.time()
        lines = ["# FASHN Q4/Q5 live results", "",
                 f"Complete: **{self.state['complete']}**. Phase: **{self.state['phase']}**. "
                 f"Active: **{self.state['active_job']}**.", "",
                 f"Completed native jobs: {sum(row['status'] == 'done' for row in self.state['jobs'].values())}/36.", "",
                 "Windows x64 CPU. Quantized numerical failures remain visible; no human approval is implied.",
                 "Recorded full trajectories are compared with recorded full trajectories. Private commit and working set overlap.",
                 "Historical Python PNGs are visual controls; their timings are not fresh measurements.", "",
                 "BF16 seed-42 trajectory gates use historical native floating controls (CPU cardigan, BLAS bottoms).",
                 "Other policy gates use fresh native BF16. BF16 seed 43 is explicitly a self-control, not an independent oracle.", "",
                 "## Weight storage", "",
                 "| Policy | Ready file bytes | Reduction vs BF16 |",
                 "|---|---:|---:|"]
        for policy, record in self.weights.items():
            lines.append(f"| {policy} | {record['output_bytes']} | "
                         f"{100 * (1-record['output_bytes']/self.weights['bf16']['output_bytes']):.2f}% |")
        lines += ["", "## Full generations", "",
                 "| Job | Sampling s | Process wall s | Peak WS GiB | Private GiB | Crop RMSE vs BF16 | Float gates |",
                 "|---|---:|---:|---:|---:|---:|---|"]
        panels = []
        for label, row in self.state["jobs"].items():
            if row["status"] != "done" or row["kind"] != "full":
                continue
            measure = row["measurement"]
            error = row.get("cropped_vs_floating", {}).get("rmse", 0)
            passed = row["comparison_passed"]
            lines.append(f"| {label} | {row['sampling_seconds']:.3f} | {measure['wall_seconds']:.3f} | "
                         f"{measure['peak_working_set_bytes']/2**30:.4f} | "
                         f"{measure['sampled_peak_private_bytes']/2**30:.4f} | {error:.4f} | {passed} |")
            prefix = "runs/" + label
            panels.append(f"<section><h3>{html.escape(label)}</h3><p>Numerical gates: {passed}; "
                          f"crop RMSE versus fresh BF16: {error:.4f}. Human review pending.</p>"
                          "<p>Fresh BF16 reference, candidate, 16x difference, and worst-region pair:</p>"
                          f'<a href="{prefix}/floating-reference.png"><img src="{prefix}/floating-reference.png" alt="Fresh BF16 reference"></a>'
                          f'<a href="{prefix}/sample.png"><img src="{prefix}/sample.png" alt="Generated crop"></a>'
                          f'<a href="{prefix}/difference-x16.png"><img src="{prefix}/difference-x16.png" alt="Absolute difference times 16"></a>'
                          f'<a href="{prefix}/worst-region.png"><img src="{prefix}/worst-region.png" alt="Worst error region: reference then candidate"></a>'
                          f'<p><a href="{prefix}/full.png">Full canvas</a> | '
                          f'<a href="{prefix}/comparison/comparison.json">Per-step numerical report</a></p></section>')
        lines += ["", "## Logical allocation counters (not process memory)", "",
                  "Assigned weights are measured at sampling completion, after modulation retirement.",
                  "Arena/cache/scratch are their observed high-water values; these peaks need not coincide.", "",
                  "| Job | Active assigned weights MiB | Arena MiB | Cache MiB | Conversion scratch bytes |",
                  "|---|---:|---:|---:|---:|"]
        for label, row in self.state["jobs"].items():
            if row["status"] == "done":
                logical = row["logical_memory_high_water"]
                assigned = row["logical_memory_at_sampling_complete"]["weights"]["assigned_payload_bytes"]
                lines.append(f"| {label} | {assigned/2**20:.3f} | {logical['runtime_buffer_bytes']/2**20:.3f} | "
                             f"{logical['modulation_cache_bytes']/2**20:.3f} | {logical['conversion_scratch_bytes']} |")
        lines += ["", "## Repeated fixed-state forwards", "",
                  "Three separate processes per policy, one first and one warm forward each. Not full-image latency variance.", "",
                  "| Policy | Warm n | Warm mean s | Sample stddev s | First mean s | Mean peak WS GiB | Mean private GiB | Worst velocity rel L2 |",
                  "|---|---:|---:|---:|---:|---:|---:|---:|"]
        self.state["probe_statistics"] = {}
        for policy in POLICIES:
            rows = [row for row in self.state["jobs"].values()
                    if row["status"] == "done" and row["kind"] == "probe" and row["policy"] == policy]
            if rows:
                warm = distribution([row["forward_seconds"][1] for row in rows])
                first = distribution([row["forward_seconds"][0] for row in rows])
                self.state["probe_statistics"][policy] = {
                    "warm_forward_seconds": warm, "first_forward_seconds": first,
                    "peak_working_set_bytes": distribution([row["measurement"]["peak_working_set_bytes"] for row in rows]),
                    "sampled_peak_private_bytes": distribution([row["measurement"]["sampled_peak_private_bytes"] for row in rows]),
                    "velocity_relative_l2": [row["velocity_vs_floating"]["relative_l2"] for row in rows]}
                deviation = f"{warm['sample_stddev']:.5f}" if warm["sample_stddev"] is not None else "not-estimable"
                stats = self.state["probe_statistics"][policy]
                lines.append(f"| {policy} | {warm['n']} | {warm['mean']:.4f} | {deviation} | {first['mean']:.4f} | "
                             f"{stats['peak_working_set_bytes']['mean']/2**30:.4f} | "
                             f"{stats['sampled_peak_private_bytes']['mean']/2**30:.4f} | "
                             f"{max(stats['velocity_relative_l2']):.7f} |")
        if "selection" in self.state:
            lines += ["", "Selected lower-bit policy: `" + self.state["selection"] + "` (numerical RMSE screening, not human preference)."]
        if "repeatability" in self.state:
            lines += ["", "## Full-run repeatability", "", "```json",
                      json.dumps(self.state["repeatability"], indent=2), "```"]
        if "seed_sensitivity" in self.state:
            lines += ["", "## Seed sensitivity (descriptive, two seeds)", "", "```json",
                      json.dumps(self.state["seed_sensitivity"], indent=2), "```"]
        if "mixed_precision" in self.state:
            lines += ["", "## Four-matrix restoration", "",
                      "Local comparison uses the same coordinates: the selected unmixed policy's worst 64x64 scan region.",
                      "This measures image error, not a segmented garment-quality metric.", "", "```json",
                      json.dumps(self.state["mixed_precision"], indent=2), "```"]
        incomplete = [f"- `{name}`: {row['status']}" for name, row in self.state["jobs"].items() if row["status"] != "done"]
        lines += ["", "## Pending or failed execution", ""] + (incomplete or ["No started job is incomplete."])
        if "operational_error" in self.state:
            lines += ["", "**Operational error:** " + self.state["operational_error"]]
        lines += ["", "## Human review (not yet performed)", "",
                  "Inspect garment identity, sleeve/hem placement, boundaries, texture/logo preservation,",
                  "person/background preservation, and whether quantization amplifies errors present in BF16."]
        write_report(self.state_path, self.state)
        atomic_text(self.root / "report.md", "\n".join(lines) + "\n")
        inputs = "".join(f'<section><h2>{case}: prepared person, garment, historical Python output</h2>'
                         f'<img src="assets/{case}-ca_image.png" alt="Prepared person">'
                         f'<img src="assets/{case}-garment_image.png" alt="Prepared garment">'
                         f'<img src="assets/{case}-python.png" alt="Historical original Python output"></section>'
                         for case in CASES)
        document = ('<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width">'
                    '<title>FASHN Q4/Q5 study</title><style>body{font:16px system-ui;margin:24px;background:#111;color:#eee}'
                    'a{color:#8cf}section{border-top:1px solid #555;padding:16px 0}img{width:min(30%,320px);'
                    'max-height:480px;object-fit:contain;vertical-align:top;margin:6px}pre{white-space:pre-wrap}</style>'
                    '<h1>FASHN Q4/Q5 diagnostic images</h1>'
                    f"<p>Complete: {self.state['complete']}; phase: {html.escape(self.state['phase'])}; "
                    f"active: {html.escape(str(self.state['active_job']))}. Refresh to see newly completed jobs.</p>"
                    '<p>Numerical agreement is not human perceptual approval. Click images for full resolution. '
                    'Difference images are amplified 16x; worst-region panels show reference then candidate.</p>'
                    '<p><a href="report.md">Measurements</a> | <a href="results.json">Structured results/provenance</a></p>'
                    + inputs + "".join(panels) + "</html>")
        atomic_text(self.root / "index.html", document)

    def run_job(self, label, case, policy, seed=42, kind="full"):
        if label in self.state["jobs"]:
            if self.state["jobs"][label]["status"] != "done":
                raise ValueError("Recover the incomplete job before resuming: " + label)
            return self.state["jobs"][label]
        folder = self.root / "runs" / label
        folder.mkdir(parents=True, exist_ok=False)
        record = self.weights[policy]
        for path in (self.binary, self.conditions(case)):
            if sha256(path) != self.state["provenance"]["files"][str(path)]:
                raise ValueError("Executable/input changed before job: " + label)
        if sha256(Path(record["output"])) != record["output_sha256"]:
            raise ValueError("Weights changed before job: " + label)
        matrix_type = record["matrix_type"]
        command = [str(self.binary), record["output"], str(self.conditions(case)), str(folder / "trajectory"),
                   "20", "1.5", "1.5", "1", str(seed), str(CASES[case][1]), "16",
                   "--matrix-type", matrix_type, "--mmap", "off", "--load-threads", "2",
                   "--precompute-modulations", "--modulation-cache-mib", "128", "--fused-gelu", "--memory-profile"]
        if matrix_type == "bf16":
            command += ["--upcast-matrices"]
        if record["policy"]:
            command += ["--f32-matrices", str(self.root / "restored-four.json")]
        if kind == "probe":
            command += ["--probe-forwards", "2", "--no-record"]
        row = {"attempt_id": uuid.uuid4().hex, "status": "running", "kind": kind, "case": case, "policy": policy, "seed": seed,
               "command": command, "started_unix_seconds": time.time()}
        self.state["jobs"][label] = row
        self.state["active_job"] = label
        self.publish()
        print(f"START {self.state['phase']} {label}", flush=True)
        measurement = measure_command(command, folder / "run.log", 14400, folder / "memory-timeline.jsonl",
                                      process_path=folder / "process.json")
        row["measurement"] = measurement
        write_report(folder / "measurement.json", measurement)
        if measurement["exit_code"] != 0:
            row["status"] = "failed"
            self.publish()
            raise RuntimeError("Native job failed: " + label)
        native = folder / "trajectory"
        manifest = read_json(native / "manifest.json")
        expected_restored = set(record["policy"]["f32_matrices"]) if record["policy"] else set()
        events = [json.loads(line) for line in (native / "memory-profile.jsonl").read_text().splitlines()]
        validate_execution(manifest, events, matrix_type, expected_restored, kind, case, seed)
        active = next(event for event in events if event["phase"] == "sampling_complete")
        row["logical_memory_at_sampling_complete"] = {
            key: active.get(key) for key in ("weights", "modulation_cache_bytes", "runtime_buffer_bytes",
                                            "workspace_bytes_by_backend", "inactive_modulation_parameter_bytes")}
        row["logical_memory_high_water"] = {
            "runtime_buffer_bytes": max(event["runtime_buffer_bytes"] for event in events),
            "planned_cpu_work_bytes": max(event["planned_cpu_work_bytes"] for event in events),
            "modulation_cache_bytes": max(event["modulation_cache_bytes"] for event in events),
            "conversion_scratch_bytes": max(event["conversion"]["scratch_peak_bytes"]
                                            for event in events if event["phase"] != "process_setup")}
        row["sampling_seconds"] = manifest["sampling_seconds"]
        if kind == "probe":
            row["forward_seconds"] = manifest["probe_forward_seconds"]
            if len(row["forward_seconds"]) != 2:
                raise ValueError("Incomplete forward probe")
            velocity = load_file(str(native / "probe.safetensors"))["velocity"]
            if canonical(velocity).shape != (3, 864, 576):
                raise ValueError("Wrong probe output shape")
            expected = (velocity if label == "probe-bf16-0" else load_file(str(
                self.root / "runs/probe-bf16-0/trajectory/probe.safetensors"))["velocity"])
            row["velocity_vs_floating"] = metrics(expected, velocity)
        else:
            baseline_label = f"full-{case}-bf16-s{seed}"
            if policy == "bf16" and seed == 42:
                reference = reference_view(self.historical(case), self.root / "references" / f"historical-{case}",
                                           "historical_native_floating_CPU_cardigan_BLAS_bottoms")
            elif policy == "bf16":
                reference = reference_view(native, self.root / "references" / f"fresh-{case}-s{seed}",
                                           "fresh_native_bf16_self_control_not_independent_oracle")
            else:
                source = self.root / "runs" / baseline_label / "trajectory"
                reference = reference_view(source, self.root / "references" / f"fresh-{case}-s{seed}",
                                           "fresh_native_bf16")
            comparison = compare(reference, native, folder / "comparison")
            row["comparison_passed"] = comparison["passed"]
            row["reference_kind"] = read_json(reference / "manifest.json")["reference_kind"]
            if policy == "bf16" and seed == 42 and not comparison["passed"]:
                raise ValueError("Fresh floating baseline failed historical floating gates")
            image = pixels(load_file(str(native / "step-19.safetensors"))["image"])
            if image.shape != (864, 576, 3):
                raise ValueError("Wrong generated image dimensions")
            Image.fromarray(image).save(folder / "full.png")
            crop = read_json(self.prepared(case))["crop"]
            cropped = crop_pixels(image, crop)
            Image.fromarray(cropped).save(folder / "sample.png")
            if policy == "bf16":
                expected_crop = cropped
            else:
                with Image.open(self.root / "runs" / baseline_label / "sample.png") as previous:
                    expected_crop = np.array(previous)
            Image.fromarray(expected_crop).save(folder / "floating-reference.png")
            row["cropped_vs_floating"] = pixel_metrics(expected_crop, cropped)
            if policy == "bf16":
                expected_full = image
            else:
                with Image.open(reference / "expected.png") as previous:
                    expected_full = np.array(previous)
            row["full_vs_floating"] = pixel_metrics(expected_full, image)
            if seed == 42:
                with Image.open(self.root / "assets" / f"{case}-python.png") as original:
                    row["cropped_vs_historical_python"] = pixel_metrics(np.array(original), cropped)
            delta = np.abs(cropped.astype(np.int16) - expected_crop).astype(np.int32)
            Image.fromarray(np.clip(delta * 16, 0, 255).astype(np.uint8)).save(folder / "difference-x16.png")
            region = worst_region(expected_crop, cropped)
            row["worst_region"] = region
            x, y, width, height = (region[key] for key in ("x", "y", "width", "height"))
            panel = np.concatenate((expected_crop[y:y + height, x:x + width],
                                    cropped[y:y + height, x:x + width]), axis=1)
            Image.fromarray(panel).resize((width * 6, height * 3), Image.Resampling.NEAREST).save(folder / "worst-region.png")
        row["status"] = "done"
        row["artifacts"] = {path.relative_to(self.root).as_posix(): sha256(path)
                            for path in sorted(folder.rglob("*")) if path.is_file()}
        self.state["active_job"] = None
        self.publish()
        print(f"DONE {label}: {measurement['wall_seconds']:.3f}s; "
              f"peak WS {measurement['peak_working_set_bytes']/2**30:.4f} GiB", flush=True)
        return row

    def execute(self):
        self.state["phase"] = "QP3"
        for policy in POLICIES:
            for repeat in range(3):
                self.run_job(f"probe-{policy}-{repeat}", "cardigan", policy, kind="probe")
        self.state["probe_checkpoint_complete"] = True
        self.publish()
        if self.args.through == "probes":
            return
        self.state["phase"] = "QP4"
        for case in CASES:
            for policy in POLICIES:
                self.run_job(f"full-{case}-{policy}-s42", case, policy)
        self.state["full_matrix_complete"] = True
        self.publish()
        if self.args.through == "images":
            return
        self.state["phase"] = "QP5"
        selected = select_candidate(self.state["jobs"])
        self.state["selection"] = selected
        policy_path = self.root / "restored-four.json"
        if policy_path.exists() and read_json(policy_path) != {"f32_matrices": RESTORED}:
            raise ValueError("Restoration policy changed")
        write_report(policy_path, {"f32_matrices": RESTORED})
        destination = self.root / "ready" / "selected-mixed"
        if destination.exists():
            mixed = read_json(destination / "export.json")
            if (not mixed["complete"] or mixed["matrix_type"] != selected
                    or mixed["policy"] != {"f32_matrices": RESTORED}
                    or sha256(Path(mixed["output"])) != mixed["output_sha256"]
                    or mixed["provenance_sha256"].get(str(self.verifier)) != sha256(self.verifier)):
                raise ValueError("Mixed export changed or incomplete")
        else:
            mixed = export(SimpleNamespace(
                threads=2, cli=self.cli, verifier=self.verifier,
                checkpoint=self.experiment / "weights/model.safetensors",
                matrix_map=self.experiment / "checkpoint31-memory-load/q8_0-mmap-on-load-16/manifest.json",
                matrix_type=selected, f32_matrices=policy_path, reuse_file=None, output=destination))
        self.weights["selected-mixed"] = mixed
        self.state["weights"]["selected-mixed"] = mixed
        for case in CASES:
            self.run_job(f"full-{case}-selected-mixed-s42", case, "selected-mixed")
        self.state["mixed_precision"] = {}
        for case in CASES:
            names = [f"full-{case}-{policy}-s42" for policy in ("bf16", selected, "selected-mixed")]
            images = []
            for name in names:
                with Image.open(self.root / "runs" / name / "sample.png") as image:
                    images.append(np.array(image))
            region = self.state["jobs"][names[1]]["worst_region"]
            x, y, width, height = (region[key] for key in ("x", "y", "width", "height"))
            patches = [image[y:y + height, x:x + width] for image in images]
            self.state["mixed_precision"][case] = {
                "unmixed_global": pixel_metrics(images[0], images[1]),
                "mixed_global": pixel_metrics(images[0], images[2]), "matched_region": region,
                "unmixed_local": pixel_metrics(patches[0], patches[1]),
                "mixed_local": pixel_metrics(patches[0], patches[2])}
        for index in (1, 2):
            self.run_job(f"repeat-cardigan-{selected}-{index}", "cardigan", selected)
        names = [f"full-cardigan-{selected}-s42"] + [f"repeat-cardigan-{selected}-{i}" for i in (1, 2)]
        repeat_rows = [self.state["jobs"][name] for name in names]
        self.state["repeatability"] = {
            "sampling_seconds": distribution([row["sampling_seconds"] for row in repeat_rows]),
            "wall_seconds": distribution([row["measurement"]["wall_seconds"] for row in repeat_rows]),
            "same_seed_comparisons": []}
        first = load_file(str(self.root / "runs" / names[0] / "trajectory/step-19.safetensors"))["image"]
        for name in names[1:]:
            image = load_file(str(self.root / "runs" / name / "trajectory/step-19.safetensors"))["image"]
            self.state["repeatability"]["same_seed_comparisons"].append({
                "run": name, "float": metrics(first, image), "pixels": pixel_metrics(pixels(first), pixels(image))})
        self.run_job("full-cardigan-bf16-s43", "cardigan", "bf16", seed=43)
        self.run_job(f"full-cardigan-{selected}-s43", "cardigan", selected, seed=43)
        self.state["seed_sensitivity"] = {}
        for policy in ("bf16", selected):
            images = []
            for seed in (42, 43):
                with Image.open(self.root / "runs" / f"full-cardigan-{policy}-s{seed}" / "sample.png") as image:
                    images.append(np.array(image))
            self.state["seed_sensitivity"][policy] = pixel_metrics(*images)
        self.state["phase"] = "QP6"
        if sum(row["kind"] == "full" for row in self.state["jobs"].values()) != 18:
            raise ValueError("Full-generation matrix incomplete")
        self.state["complete"] = True
        self.publish()
        print("COMPLETE: QP3-QP6, all 18 full images and 18 probe processes recorded.", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("repo", "experiment", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--through", choices=("probes", "images", "all"), default="all")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    with exclusive_run(args.output):
        study = Study(args)
        try:
            study.execute()
        except (Exception, KeyboardInterrupt) as error:
            study.state["complete"] = False
            study.state["operational_error"] = str(error)
            active = study.state["active_job"]
            if active is not None:
                row = study.state["jobs"][active]
                row["status"] = ("interrupted" if isinstance(error, KeyboardInterrupt) else
                                 "analysis_failed" if row.get("measurement", {}).get("exit_code") == 0 else "failed")
            # Persist failure independently of HTML generation, which may itself have failed.
            write_report(study.state_path, study.state)
            raise


if __name__ == "__main__":
    main()
