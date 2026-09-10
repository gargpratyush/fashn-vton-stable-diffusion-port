"""Summarize phase-tagged diagnostic memory; load-only runs are not image-generation evidence."""
import argparse
import json
from pathlib import Path

from export_fashn_vton_reference import sha256
from run_fashn_quality import write_report


def validate_controls(manifest, command):
    controls = {"mmap": True, "load_threads": int(command[10]), "load_only": False,
                "probe_forwards": 0, "memory_profile": False, "page_classification": False,
                "matrix_type": "bf16", "precompute_modulations": False, "modulation_cache_mib": 128, "fused_gelu": False}
    values = {"--mmap": ("mmap", lambda value: value == "on"),
              "--load-threads": ("load_threads", int), "--probe-forwards": ("probe_forwards", int),
              "--matrix-type": ("matrix_type", str), "--modulation-cache-mib": ("modulation_cache_mib", int)}
    switches = {"--load-only": "load_only", "--memory-profile": "memory_profile",
                "--classify-pages": "page_classification", "--precompute-modulations": "precompute_modulations",
                "--fused-gelu": "fused_gelu"}
    explicit_upcast, recording = False, True
    options = iter(command[11:])
    for option in options:
        if option in values:
            name, convert = values[option]
            controls[name] = convert(next(options))
        elif option in switches:
            controls[switches[option]] = True
        elif option == "--upcast-matrices":
            explicit_upcast = True
        elif option == "--no-record":
            recording = False
        elif option == "--f32-matrices":
            next(options)
        else:
            raise ValueError("Unrecognized diagnostic control: " + option)
    controls["upcast_matrices"] = explicit_upcast or controls["matrix_type"] != "q8_0"
    controls["recorded_trajectory"] = recording and not controls["load_only"] and not controls["probe_forwards"]
    for name, value in controls.items():
        default = {"precompute_modulations": False, "modulation_cache_mib": 128, "fused_gelu": False}.get(name)
        if manifest.get(name, default) != value:
            raise ValueError("Manifest/command control mismatch: " + name)


def summarize_phases(manifest, events):
    required = ("process_setup", "metadata_ready", "before_model_release", "model_released")
    names = [event["phase"] for event in events]
    if any(name not in names for name in required):
        raise ValueError("Incomplete phase profile")
    if any(a["elapsed_seconds"] > b["elapsed_seconds"] for a, b in zip(events, events[1:])):
        raise ValueError("Phase time moved backwards")
    if not manifest.get("memory_profile") or names[-1] != "model_released":
        raise ValueError("Expected a completed profiled run")
    ready_name = "load_only_ready" if manifest["load_only"] else "before_model_release"
    ready = next(event for event in events if event["phase"] == ready_name)
    inactive = ready.get("inactive_modulation_parameter_bytes", 0)
    if inactive and not manifest.get("precompute_modulations"):
        raise ValueError("Unexpected inactive parameters without modulation precomputation")
    cache_peak = max(event.get("modulation_cache_bytes", 0) for event in events)
    if cache_peak > manifest.get("modulation_cache_mib", 128) * 2**20:
        raise ValueError("Modulation cache payload exceeds its configured budget")
    if ready["weights"]["registered_payload_bytes"] - inactive != ready["weights"]["assigned_payload_bytes"]:
        raise ValueError("Not all parameters have storage")
    if ready["conversion"]["scratch_live_bytes"] != 0:
        raise ValueError("Loading scratch remains live after loading")
    if not manifest["mmap"] and ready["mapped_model_files"]:
        raise ValueError("Unexpected source mapping in non-mmap control")
    if events[-1]["mapped_model_files"]:
        raise ValueError("Model mappings remain after destruction")
    durations = []
    begin = None
    for event in events:
        if event["phase"] == "graph_execute_begin":
            if begin is not None:
                raise ValueError("Nested graph execution")
            begin = event["elapsed_seconds"]
        elif event["phase"] == "graph_execute_end":
            if begin is None:
                raise ValueError("Graph execution ended without a start")
            durations.append(event["elapsed_seconds"] - begin)
            begin = None
    expected = manifest["probe_forwards"]
    if manifest["load_only"]:
        expected = 0
    elif not expected:
        expected = manifest["steps"]
        if manifest["cfg"] != 1:
            expected += max(0, manifest["steps"] - manifest["skip_cfg_last_n_steps"])
    if begin is not None or len(durations) != expected:
        raise ValueError("Incomplete graph executions")
    graph_ends = [event for event in events if event["phase"] == "graph_execute_end"]
    classified = [event for event in events if "resident_bytes_by_region_type" in event]
    return {
        "scope": "load_only" if manifest["load_only"] else
                 ("repeated_conditional_t0" if manifest["probe_forwards"] else "sampling"),
        "mmap": manifest["mmap"], "load_threads": manifest["load_threads"],
        "matrix_type": manifest["matrix_type"], "upcast_matrices": manifest["upcast_matrices"],
        "matrix_types": manifest.get("matrix_types", {}),
        "weights": ready["weights"], "conversion": ready["conversion"],
        "inactive_modulation_parameter_bytes": inactive,
        "modulation_cache_bytes": ready.get("modulation_cache_bytes", 0),
        "modulation_cache_peak_bytes": cache_peak,
        "modulation_refills": manifest.get("modulation_refills"),
        "ready_process": ready["process"], "mapped_files_at_ready": ready["mapped_model_files"],
        "graph_execution_seconds_instrumented": durations,
        "graph_end_process_samples": [event["process"] for event in graph_ends],
        "max_runtime_buffer_bytes": max(event["runtime_buffer_bytes"] for event in events),
        "max_planned_cpu_work_bytes": max(event["planned_cpu_work_bytes"] for event in events),
        "classification_phases": [event["phase"] for event in classified],
        "diagnostic_seconds": sum(event["diagnostic_seconds"] for event in events),
        "after_model_release_process": events[-1]["process"],
        "note": "Instrumented durations include observer overhead. Buffer capacity is not resident RAM; private commit is not private working set. Individual scratch peaks need not occur together."
    }


def summarize(directory, output):
    specs = sorted(directory.glob("*-spec.json"))
    if not specs:
        raise ValueError("No diagnostic specifications")
    report = {"complete": False, "runs": {}}
    for path in specs:
        spec = json.loads(path.read_text(encoding="utf-8"))
        name = path.stem.removesuffix("-spec")
        measurement_path = directory / (name + "-measurement") / "measurement.json"
        measurement = json.loads(measurement_path.read_text(encoding="utf-8"))
        if not measurement["execution_passed"] or measurement["run"]["exit_code"] != 0:
            raise ValueError("Unsuccessful diagnostic: " + name)
        if measurement["spec_sha256"] != sha256(path) or measurement["run"]["command"] != spec["command"]:
            raise ValueError("Specification/command changed: " + name)
        native = Path(spec["command"][3])
        manifest_path = native / "manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        validate_controls(manifest, spec["command"])
        profile = native / "memory-profile.jsonl"
        events = [json.loads(line) for line in profile.read_text(encoding="utf-8").splitlines()]
        if any(event["process"] is None for event in events):
            raise ValueError("This report requires Windows process memory accounting")
        if any(event["process"]["pid"] != measurement["run"]["monitored_pid"] for event in events):
            raise ValueError("Profile/measurement process mismatch: " + name)
        row = summarize_phases(manifest, events)
        row["measurement"] = measurement["run"]
        row["provenance_sha256"] = measurement["provenance_sha256"]
        row["artifacts_sha256"] = {str(p): sha256(p) for p in (path, measurement_path, manifest_path, profile)}
        timeline = measurement["run"]["memory_timeline"]
        if timeline is not None:
            row["artifacts_sha256"][timeline] = sha256(Path(timeline))
        report["runs"][name] = row
    report["complete"] = True
    output.mkdir(parents=True, exist_ok=False)
    write_report(output / "summary.json", report)
    lines = ["# Phase-resolved memory", "",
             "Diagnostic observations, not clean performance benchmarks. Load-only memory does not fault in every zero-copy weight.", "",
             "| Run | Process peak GiB | Ready working set GiB | Combined scratch peak MiB | Runtime arena peak MiB |",
             "|---|---:|---:|---:|---:|"]
    for name, row in report["runs"].items():
        lines.append(f'| {name} | {row["measurement"]["peak_working_set_bytes"]/2**30:.4f} | '
                     f'{row["ready_process"]["working_set_bytes"]/2**30:.4f} | '
                     f'{row["conversion"]["scratch_peak_bytes"]/2**20:.2f} | '
                     f'{row["max_runtime_buffer_bytes"]/2**20:.2f} |')
    (output / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(output / "summary.md")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    summarize(args.directory, args.output)
