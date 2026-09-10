"""Explicit research/evaluation-only parser export for the optional native preparer."""
import argparse
import json
from pathlib import Path
import shutil

from fashn_parser_eval import load_parser, require_evaluation_consent, PARSER_HASHES
from export_fashn_vton_reference import sha256


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("source", "parser-dir", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--accept-parser-research-license", action="store_true")
    args = parser.parse_args()
    require_evaluation_consent(args.accept_parser_research_license)
    if args.output.exists():
        raise ValueError("Choose a new output directory")
    import numpy as np
    import onnx
    import onnxruntime as ort
    import torch
    from PIL import Image
    from prepare_fashn_vton import reference_modules

    torch.set_num_threads(8)
    original, _ = load_parser(args.source, args.parser_dir, True)

    class ExportModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.model = original.model

        def forward(self, pixel_values):
            return self.model(pixel_values=pixel_values).logits

    model = ExportModel().eval()
    args.output.mkdir(parents=True)
    destination = args.output / "parser.onnx"
    with torch.inference_mode():
        torch.onnx.export(model, torch.zeros(1, 3, 576, 384), str(destination),
                          input_names=["pixel_values"], output_names=["logits"],
                          opset_version=17, dynamo=False)
    onnx.checker.check_model(str(destination))
    options = ort.SessionOptions()
    options.intra_op_num_threads = 8
    options.inter_op_num_threads = 1
    session = ort.InferenceSession(str(destination), sess_options=options, providers=["CPUExecutionProvider"])
    transforms, _, _ = reference_modules(args.source)
    resize = transforms.AspectPreserveResize((864, 864), mode="fit", backend="pil")
    comparisons = {}
    for name in ("model.webp", "garment.webp"):
        with Image.open(args.source / "examples" / "data" / name) as image:
            pixels = np.array(resize(image.convert("RGB"), allow_upsampling=False))
        values = original._preprocess_single(pixels).unsqueeze(0)
        with torch.inference_mode():
            expected = model(values).numpy()
        actual = session.run(None, {"pixel_values": values.numpy()})[0]
        error = actual.astype(np.float64) - expected
        relative = float(np.linalg.norm(error) / np.linalg.norm(expected.astype(np.float64)))
        if not np.isfinite(actual).all() or relative > 1e-4:
            raise ValueError(f"Parser ONNX parity failed on {name}: {relative}")
        comparisons[name] = {"relative_l2": relative, "max_abs": float(np.abs(error).max())}
    shutil.copyfile(args.parser_dir / "LICENSE", args.output / "LICENSE")
    manifest = {"schema": "fashn-parser-onnx-eval-v1",
                "license": "NVIDIA SegFormer: non-commercial research/evaluation only",
                "checkpoint_sha256": PARSER_HASHES["model.safetensors"],
                "onnx_sha256": sha256(destination), "license_sha256": sha256(args.output / "LICENSE"),
                "torch": torch.__version__, "onnx": onnx.__version__, "onnxruntime": ort.__version__,
                "opset": 17, "shape": [1, 3, 576, 384], "comparisons": comparisons}
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
