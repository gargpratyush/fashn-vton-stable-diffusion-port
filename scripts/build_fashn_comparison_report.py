"""Build a standalone offline report from completed FASHN precision experiments."""
import argparse
import base64
import hashlib
import html
import io
import json
from pathlib import Path

import numpy as np
from PIL import Image
from safetensors.numpy import load_file

from compare_fashn_case_output import crop_pixels
from compare_fashn_trajectories import pixels
from run_fashn_q45_study import pixel_metrics, worst_region, read_json, sha256, atomic_text


LABELS = {"bf16": "BF16 storage / F32 compute", "q8_0": "Q8_0", "q4_0": "Q4_0",
          "q5_0": "Q5_0", "q4_K": "Q4_K", "q5_K": "Q5_K",
          "selected-mixed": "Q5_K + 4 original F32 matrices"}


def compare_images(reference, candidate):
    result = pixel_metrics(reference, candidate)
    delta = np.abs(candidate.astype(np.int16) - reference.astype(np.int16))
    result["pixels_above_10_percent"] = float(100 * np.mean(delta.max(axis=2) > 10))
    result["pixels_above_32_percent"] = float(100 * np.mean(delta.max(axis=2) > 32))
    region = worst_region(reference, candidate)
    x, y, w, h = (region[key] for key in ("x", "y", "width", "height"))
    result["region"] = region
    result["local"] = pixel_metrics(reference[y:y+h, x:x+w], candidate[y:y+h, x:x+w])
    return result


def script_json(value):
    return json.dumps(value, ensure_ascii=True, allow_nan=False).replace("<", "\\u003c").replace(">", "\\u003e").replace("&", "\\u0026")


def portable(value, roots):
    if isinstance(value, dict):
        return {portable(key, roots): portable(item, roots) for key, item in value.items()}
    if isinstance(value, list):
        return [portable(item, roots) for item in value]
    if isinstance(value, str):
        for path, label in roots:
            value = value.replace(str(path), label).replace(path.as_posix(), label)
    return value


class Report:
    def __init__(self, experiment, repo):
        self.experiment, self.repo = experiment.resolve(), repo.resolve()
        self.root = self.experiment / "q45-study"
        self.state = read_json(self.root / "results.json")
        if not self.state["complete"] or len(self.state["jobs"]) != 36:
            raise ValueError("Report requires the complete 36-job study")
        self.images, self.arrays, self.rows, self.evidence = {}, {}, [], {}

    def evidence_file(self, path, expected=None):
        digest = sha256(path)
        if expected is not None and digest != expected:
            raise ValueError("Evidence hash mismatch: " + str(path))
        try:
            relative = "experiment/" + path.relative_to(self.experiment).as_posix()
        except ValueError:
            relative = "repository/" + path.relative_to(self.repo).as_posix()
        self.evidence[relative] = digest
        return digest

    def image(self, array):
        if array.dtype != np.uint8 or array.ndim != 3 or array.shape[2] != 3:
            raise ValueError("Report images must be uint8 RGB")
        stream = io.BytesIO()
        Image.fromarray(array).save(stream, format="PNG")
        payload = stream.getvalue()
        key = hashlib.sha256(payload).hexdigest()
        self.images[key] = "data:image/png;base64," + base64.b64encode(payload).decode("ascii")
        self.arrays[key] = array
        return key

    def load_image(self, path, expected=None):
        self.evidence_file(path, expected)
        with Image.open(path) as image:
            if image.mode != "RGB":
                raise ValueError("Expected RGB output: " + str(path))
            return self.image(np.array(image))

    def add_current(self):
        self.evidence_file(self.root / "results.json")
        for name, row in self.state["jobs"].items():
            if row["status"] != "done" or row["measurement"]["exit_code"] != 0:
                raise ValueError("Unsuccessful study job: " + name)
            if row["kind"] != "full":
                continue
            folder = self.root / "runs" / name
            for relative, digest in row["artifacts"].items():
                self.evidence_file(self.root / relative, digest)
            comparison = read_json(folder / "comparison" / "comparison.json")
            self.rows.append({
                "id": name, "label": LABELS[row["policy"]] + (" (repeat)" if name.startswith("repeat-") else ""),
                "policy": row["policy"], "case": row["case"], "seed": row["seed"],
                "group": "current" if name.startswith("full-") and row["seed"] == 42 else "repeat-seed",
                "sample": self.load_image(folder / "sample.png"), "full": self.load_image(folder / "full.png"),
                "sampling": row["sampling_seconds"], "wall": row["measurement"]["wall_seconds"],
                "ws": row["measurement"]["peak_working_set_bytes"], "private": row["measurement"]["sampled_peak_private_bytes"],
                "file_bytes": self.state["weights"][row["policy"]]["output_bytes"],
                "logical": row["logical_memory_at_sampling_complete"], "high_water": row["logical_memory_high_water"],
                "gate": comparison["passed"], "gate_reference": row["reference_kind"],
                "trajectory": comparison, "sampling_scope": "Recorded native sampling including modulation preparation",
                "memory_note": "Native process", "source": "Completed Q4/Q5 study",
            })

    def add_history(self):
        folder = self.experiment / "checkpoint38-integrated"
        records = read_json(folder / "results.json")
        summary = read_json(folder / "final-summary.json")
        self.evidence_file(folder / "results.json")
        self.evidence_file(folder / "final-summary.json")
        if not records["passed"]:
            raise ValueError("Historical optimization acceptance incomplete")
        for name in ("cardigan-floating-cpu", "cardigan-floating-blas", "cardigan-q8",
                     "bottoms-q8", "cardigan-mixed", "bottoms-mixed"):
            row, extra = records["runs"][name], summary["runs"][name]
            if not row["completed"] or row["measurement"]["exit_code"] != 0:
                raise ValueError("Incomplete historical run")
            self.evidence_file(folder / name / "manifest.json", row["manifest_sha256"])
            self.evidence_file(folder / name / "step-19.safetensors")
            full = pixels(load_file(str(folder / name / "step-19.safetensors"))["image"])
            case = name.split("-")[0]
            current = next(item for item in self.rows if item["case"] == case and item["policy"] == "bf16" and item["seed"] == 42)
            if full.shape != self.arrays[current["full"]].shape:
                raise ValueError("Historical canvas dimensions differ")
            cropped = crop_pixels(full, {"x": 0, "y": 48, "width": 576, "height": 768})
            stored = self.load_image(folder / (name + ".png"))
            if not np.array_equal(cropped, self.arrays[stored]):
                raise ValueError("Historical PNG does not match trajectory: " + name)
            label = ("Q8_0 + 4 original F32 matrices" if name.endswith("mixed") else
                     "BF16 storage / BLAS F32" if name.endswith("blas") else
                     "BF16 storage / CPU F32" if name.endswith("cpu") else "Q8_0")
            self.rows.append({
                "id": "history-" + name, "case": case, "seed": 42, "label": label,
                "group": "historical", "policy": name.split("-", 1)[1], "sample": stored, "full": self.image(full),
                "sampling": extra["sampling_seconds"], "wall": extra["process_wall_seconds"],
                "ws": extra["peak_working_set_bytes"], "private": extra["sampled_peak_private_bytes"],
                "file_bytes": extra["file_size_bytes"], "gate": None,
                "sampling_scope": "Historical recorded native sampling; separate experiment, not a fresh repeat",
                "memory_note": "BLAS library allocations included" if name.endswith("blas") else "Native process",
                "source": "Historical checkpoint 38",
            })
        for case in ("cardigan", "bottoms"):
            folder = self.experiment / ("checkpoint27-upstream-" + case)
            result = read_json(folder / "result.json")
            measured_path = self.experiment / ("checkpoint27-upstream-" + case + "-measurement") / "measurement.json"
            measured = read_json(measured_path)
            self.evidence_file(folder / "result.json")
            self.evidence_file(measured_path)
            measurement = measured["run"]
            if not measured["execution_passed"] or measurement["exit_code"] != 0:
                raise ValueError("Original Python inference failed")
            valid_memory = measurement.get("memory_scope") == "active_python_interpreter"
            self.rows.append({
                "id": "python-" + case, "case": case, "seed": 42, "label": "Original PyTorch F32",
                "group": "historical", "policy": "python", "sample": self.load_image(folder / "sample-0.png", result["output_sha256"]),
                "full": self.load_image(folder / "full.png"), "sampling": result["sampling_and_pil_seconds"],
                "wall": measurement["wall_seconds"], "ws": measurement["peak_working_set_bytes"] if valid_memory else None,
                "private": measurement["sampled_peak_private_bytes"] if valid_memory else None,
                "file_bytes": None, "gate": None, "source": "Historical original Python / checkpoint 27",
                "sampling_scope": "Unmodified batched-CFG sampler + PIL, no trajectory recording; not the same timing interval as native",
                "memory_note": "Actual Python interpreter" if valid_memory else "Unavailable: old measurement tracked the launcher, not the interpreter",
            })

    def build(self, output):
        self.add_current()
        self.add_history()
        by_id = {row["id"]: row for row in self.rows}
        cache = {}
        for row in self.rows:
            references = {"bf16": f"full-{row['case']}-bf16-s{row['seed']}"}
            if row["seed"] == 42:
                references.update(q8_0=f"full-{row['case']}-q8_0-s42", python="python-" + row["case"])
            row["comparisons"] = {}
            for label, reference_id in references.items():
                reference = by_id[reference_id]
                key = (reference["sample"], row["sample"])
                if key not in cache:
                    cache[key] = compare_images(self.arrays[key[0]], self.arrays[key[1]])
                row["comparisons"][label] = cache[key]
        inputs = {}
        for case in ("cardigan", "bottoms"):
            inputs[case] = {key: self.load_image(self.root / "assets" / f"{case}-{key}.png")
                            for key in ("ca_image", "garment_image")}
        appendices = {}
        for name in ("memory-and-latency-optimization-results.md", "quantization-and-upstream-comparison.md", "NOTICE.md"):
            path = self.repo / "reports" / name
            self.evidence_file(path)
            appendices[name] = path.read_text(encoding="utf-8")
        legacy_path = self.repo / "reports" / "full-image-precision-summary.json"
        self.evidence_file(legacy_path)
        legacy = read_json(legacy_path)
        payload = {
            "rows": self.rows, "images": self.images, "inputs": inputs,
            "probes": self.state["probe_statistics"], "repeatability": self.state["repeatability"],
            "mixed": self.state["mixed_precision"], "seed_sensitivity": self.state["seed_sensitivity"],
            "weights": {key: {k: value[k] for k in ("matrix_type", "policy", "output_sha256", "output_bytes")}
                        for key, value in self.state["weights"].items()},
            "evidence": self.evidence, "legacy": legacy,
        }
        payload = portable(payload, [(self.experiment, "local-experiment"), (self.repo, "repository")])
        appendix = "".join("<details><summary>" + html.escape(name) + " (historical source text)</summary><pre>" +
                           html.escape(portable(text, [(self.experiment, "local-experiment"), (self.repo, "repository")])) +
                           "</pre></details>" for name, text in appendices.items())
        template = Path(__file__).with_name("fashn_comparison_template.html").read_text(encoding="utf-8")
        self.evidence_file(Path(__file__))
        self.evidence_file(Path(__file__).with_name("fashn_comparison_template.html"))
        payload["evidence"] = self.evidence
        document = template.replace("@@DATA@@", script_json(payload)).replace("@@APPENDICES@@", appendix)
        output.parent.mkdir(parents=True, exist_ok=True)
        atomic_text(output, document)
        print(json.dumps({"output": str(output), "bytes": output.stat().st_size, "sha256": sha256(output),
                          "image_blobs": len(self.images), "full_runs": len(self.rows),
                          "evidence_files": len(self.evidence)}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("experiment", "repo", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    Report(args.experiment, args.repo).build(args.output)
