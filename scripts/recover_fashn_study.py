"""Archive one interrupted v2 study attempt after verifying ownership, provenance and completed work."""
import argparse
from pathlib import Path
import re
import uuid

from fashn_artifacts import contained, exclusive_lock, read_json, sha256, write_report
from fashn_process_state import identity_active


def recover(root, label):
    root = Path(root).resolve()
    if not re.fullmatch(r"[A-Za-z0-9_-]+", label):
        raise ValueError("Expected one job label, not a path")
    with exclusive_lock(root / "study.lock"):
        state_path = root / "results.json"
        state = read_json(state_path)
        if state.get("schema") != "fashn-q45-study-v2":
            raise ValueError("Frozen/legacy studies require their original recovery procedure; no implicit migration")
        journal_path = root / ("recovery-" + label + ".json")
        if label not in state["jobs"] and journal_path.exists():
            journal = read_json(journal_path)
            if any(attempt["label"] == label and attempt["archive"] == journal["archive"]
                   for attempt in state.get("recovered_attempts", [])):
                archive = contained(root, journal["archive"])
                if not archive.is_dir():
                    raise ValueError("Recorded recovery archive is missing")
                journal_path.unlink()
                return archive
        row = state["jobs"][label]
        if row["status"] == "done":
            raise ValueError("Completed jobs must never be recovered or rerun")
        for path, digest in state["provenance"]["files"].items():
            if sha256(path) != digest:
                raise ValueError("Study provenance changed: " + path)
        for weight in state["weights"].values():
            if sha256(weight["output"]) != weight["output_sha256"]:
                raise ValueError("Study weights changed")
        for other in state["jobs"].values():
            if other["status"] == "done":
                for name, digest in other["artifacts"].items():
                    if sha256(contained(root, name)) != digest:
                        raise ValueError("Completed artifact changed: " + name)
        source = root / "runs" / label
        if journal_path.exists():
            journal = read_json(journal_path)
            if journal["state_sha256"] != sha256(state_path):
                raise ValueError("State changed during interrupted recovery")
            archive = contained(root, journal["archive"])
        else:
            process = read_json(source / "process.json")
            if process["schema"] != "fashn-process-v1" or process["phase"] not in ("pending", "running", "exited"):
                raise ValueError("Launch identity is incomplete; automatic recovery cannot establish child ownership")
            if any(identity_active(identity) for identity in
                   (process["owner"], process["child"], process.get("worker")) if identity):
                raise ValueError("An owned process is still active; recovery never terminates processes")
            archive = root / "interrupted-attempts" / (label + "-" + uuid.uuid4().hex)
            journal = {"state_sha256": sha256(state_path), "archive": archive.relative_to(root).as_posix()}
            write_report(journal_path, journal)
        if source.exists() == archive.exists():
            raise ValueError("Ambiguous recovery archive state")
        archive.parent.mkdir(exist_ok=True)
        if source.exists():
            source.rename(archive)
        state.setdefault("recovered_attempts", []).append({"label": label, "archive": journal["archive"], "job": row})
        del state["jobs"][label]
        if state["active_job"] == label:
            state["active_job"] = None
        state["complete"] = False
        write_report(state_path, state)
        journal_path.unlink()
        return archive


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", type=Path, required=True)
    parser.add_argument("--job", required=True)
    args = parser.parse_args()
    print(recover(args.study, args.job))
