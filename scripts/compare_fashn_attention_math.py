"""Compare native F32 flash with upstream default CPU SDPA at FASHN attention shapes."""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from safetensors.numpy import load_file

from compare_fashn_trajectories import metrics
from export_fashn_vton_reference import load_reference, sha256


def values(tokens, seed):
    indices = np.arange(tokens * 1280, dtype=np.uint32)
    hashed = indices * np.uint32(2654435761) + np.uint32(seed)
    hashed ^= hashed >> np.uint32(16)
    result = ((hashed & np.uint32(65535)).astype(np.float32) - np.float32(32768)) / np.float32(32768)
    return torch.from_numpy(result.reshape(1, tokens, 10, 128)).permute(0, 2, 1, 3)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--native", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(16)
    torch.set_num_interop_threads(1)
    module, _, _ = load_reference(args.source)
    native = json.loads((args.native / "native.json").read_text(encoding="utf-8"))
    args.output.mkdir(exist_ok=False)
    report = {"passed": False, "torch": torch.__version__, "threads": torch.get_num_threads(),
              "interop_threads": torch.get_num_interop_threads(), "torch_config": torch.__config__.show(),
              "native_manifest_sha256": sha256(args.native / "native.json"), "cases": []}
    with torch.inference_mode():
        for row in native:
            tokens = row["tokens"]
            if (tokens not in (3456, 6912) or row["heads"] != 10 or row["head_dim"] != 128 or row["threads"] != 16):
                raise ValueError("Unexpected native attention configuration")
            expected = load_file(str(args.native / f"attention-{tokens}.safetensors"))["attention"].reshape(tokens, 1280)
            source_values = [values(tokens, seed) for seed in (101, 203, 307)]
            # Obtain actual single-block strides from upstream normalization/RoPE, not assumed layouts.
            storage = torch.zeros((1, tokens, 8960), dtype=torch.float32)
            q, k, v = [storage[..., start:start + 1280].reshape(1, tokens, 10, 128).permute(0, 2, 1, 3)
                       for start in (0, 1280, 2560)]
            q, k = module.QKNorm(128).eval()(q, k, v)
            pe = torch.eye(2).expand(1, 1, tokens, 64, 2, 2)
            q, k = module.apply_rope(q, k, pe)
            for destination, source in zip((q, k, v), source_values):
                destination.copy_(source)
            for mode in ("upstream_single_layout", "packed_heads", "packed_heads_with_io"):
                inputs = (q, k, v) if mode == "upstream_single_layout" else tuple(t.contiguous() for t in source_values)
                seconds = []
                for _ in range(3):
                    start = time.perf_counter()
                    if mode == "packed_heads_with_io":
                        inputs = tuple(t.clone().contiguous() for t in source_values)
                    result = module._attn_processor(*inputs).transpose(1, 2).contiguous().reshape(tokens, 1280).numpy().copy()
                    seconds.append(time.perf_counter() - start)
                comparison = metrics(expected, result)
                comparison["passed"] = comparison["passed"] and comparison["max_abs"] <= 2e-5
                report["cases"].append({
                    "tokens": tokens, "mode": mode, "seconds": seconds, "comparison": comparison,
                    "input_strides_elements": [list(t.stride()) for t in inputs], "native": row,
                    "note": "Synthetic equal values; native includes input/layout/graph/output costs. Packed IO mode includes explicit clone/layout/output costs; no model speedup is inferred."
                })
                (args.output / "comparison.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
                if not comparison["passed"]:
                    raise RuntimeError("F32 attention numerical gate failed")
    report["passed"] = True
    (args.output / "comparison.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps([{"tokens": row["tokens"], "mode": row["mode"], "seconds": row["seconds"],
                       "relative_l2": row["comparison"]["relative_l2"]} for row in report["cases"]], indent=2))


if __name__ == "__main__":
    main()
