"""Optional NVIDIA SegFormer research/evaluation-only adapter; never imported by the native runtime."""

import importlib
import importlib.metadata
import shutil

from export_fashn_vton_reference import download_verified_files, load_reference_package, sha256


PARSER_REVISION = "1f80c34dbab321c5730dda5c3fea279fd3e97498"
PARSER_VERSION = "0.1.1"
PARSER_HASHES = {
    "config.json": "87bd5b66419dbfa7c02cfbeb94e292454863b1606c5df29526b19a6c1de70d9f",
    "preprocessor_config.json": "54007eb4cedae02565c4f035750aeebe3ddc53d36ab575e8534615623e6b6225",
    "model.safetensors": "e43c8c8a9b04f28798f0a4630cf18caa2cdb27a0d454fae43a5716e6f7078244",
}
LICENSE_URL = "https://github.com/NVlabs/SegFormer/blob/master/LICENSE"


def require_evaluation_consent(accepted):
    if accepted is not True:
        raise ValueError("The human parser is restricted to non-commercial research/evaluation; "
                         "review its license and pass --accept-parser-research-license explicitly")


def parser_distribution():
    distribution = importlib.metadata.distribution("fashn-human-parser")
    if distribution.version != PARSER_VERSION:
        raise ValueError(f"Expected fashn-human-parser=={PARSER_VERSION}")
    return distribution


def verify_parser_weights(directory):
    for name, expected in PARSER_HASHES.items():
        if sha256(directory / name) != expected:
            raise ValueError(f"Unexpected human-parser checkpoint hash: {name}")


def download_parser(directory, accepted):
    require_evaluation_consent(accepted)
    distribution = parser_distribution()
    licenses = [distribution.locate_file(p) for p in distribution.files if p.name == "LICENSE"]
    if len(licenses) != 1:
        raise ValueError("Cannot locate the installed parser's unambiguous license file")
    directory.mkdir(parents=True, exist_ok=True)
    destination = directory / "LICENSE"
    if destination.exists() and sha256(destination) != sha256(licenses[0]):
        raise ValueError("Refusing to overwrite a different parser license")
    if not destination.exists():
        shutil.copyfile(licenses[0], destination)
    download_verified_files(directory, "fashn-ai/fashn-human-parser", PARSER_REVISION, PARSER_HASHES)


def load_parser(source, directory, accepted):
    require_evaluation_consent(accepted)
    parser_distribution()
    verify_parser_weights(directory)
    import torch
    from fashn_human_parser import FashnHumanParser
    from transformers import SegformerForSemanticSegmentation

    class LocalEvaluationParser(FashnHumanParser):
        def __init__(self):
            # Keep upstream predict/preprocessing, but forbid implicit Hub downloads.
            self.device = "cpu"
            self.model = SegformerForSemanticSegmentation.from_pretrained(
                str(directory.resolve()), local_files_only=True, use_safetensors=True,
                torch_dtype=torch.float32).to("cpu").eval()

    agnostic = importlib.import_module(load_reference_package(source) + ".preprocessing.agnostic")
    return LocalEvaluationParser(), agnostic


def parser_provenance():
    distribution = parser_distribution()
    return {
        "license": "NVIDIA SegFormer: non-commercial research/evaluation only",
        "license_url": LICENSE_URL, "revision": PARSER_REVISION, "sha256": PARSER_HASHES,
        "package_version": distribution.version,
        "predict_source_sha256": sha256(distribution.locate_file("fashn_human_parser/parser.py")),
        "transformers": importlib.metadata.version("transformers"),
        "huggingface_hub": importlib.metadata.version("huggingface-hub"),
        "local_files_only": True, "device": "cpu", "dtype": "F32",
    }
