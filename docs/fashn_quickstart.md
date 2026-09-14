# FASHN VTON 1.5: build and run

This is a native port built on stable-diffusion.cpp, not a Python inference
server behind a C++ wrapper. The deployed prepared-input path needs no
Python, VAE or text encoder. Optional raw preparation uses OpenCV and ONNX
Runtime with separately downloaded pose/parser models.

The measured/validated platform is Windows x64 CPU with AVX2/FMA and
Visual Studio 2022. WSL is suitable for Git operations; that does not imply
the published Windows latency/memory results were measured under WSL.
Other platforms and GPUs need their own validation.

For the proposed ARM64 Android path, platform gaps and sequential acceptance
checkpoints, see [the Android feasibility plan](fashn_android_plan.md).
That plan does not establish Android runtime support.
Subsequently, [one diagnostic S23 Q4_K generation completed](../reports/android-s23-q4k/README.md),
but failed strict cross-platform parity. It does not enable the public quantized
API or replace this quickstart's validated floating path.
For the prepared-input source/data bundle and exact Windows ARM64/Android
handoff steps, see [the device transfer runbook](fashn_arm_transfer.md).

## Clone the complete source

```powershell
git clone --recurse-submodules https://github.com/gargpratyush/fashn-vton-stable-diffusion-port.git
Set-Location fashn-vton-stable-diffusion-port
```

For an existing clone:

```powershell
git submodule update --init --recursive
```

GGML, the upstream frontend, libwebp and libwebm are pinned submodules.
They are not replaced by locally modified dependency sources.

## Build the prepared-input CPU runtime

Run from a Visual Studio developer shell with CMake on PATH:

```powershell
cmake -S . -B build -G "Visual Studio 17 2022" -A x64 `
  -DSD_BUILD_TESTS=ON -DSD_BUILD_EXAMPLES=ON `
  -DSD_FASHN_PREPROCESS=OFF -DSD_SERVER_BUILD_FRONTEND=OFF `
  -DGGML_BLAS=OFF -DGGML_LLAMAFILE=OFF
cmake --build build --config Release --parallel
ctest --test-dir build -C Release --output-on-failure
```

This produces `build\bin\Release\sd-cli.exe` and `sd-server.exe`.
The standalone FASHN `/try-on` page is embedded independently; disabling
the general upstream frontend avoids an unrelated pnpm install/build.
Default CTests use primitives and synthetic contracts. Full real-weight
tests additionally require the optional checkpoint/fixture CMake settings
documented in [the developer guide](fashn_vton.md). They are not silently
replaced by the lightweight tests.

## Obtain the model and prepared inputs

Download `model.safetensors` from
[fashn-ai/fashn-vton-1.5](https://huggingface.co/fashn-ai/fashn-vton-1.5/tree/7720683168567eb5a2a4c67f15116c6e29c83ded).
The pinned file SHA256 is:

```text
d6cd38286885bc29fa487ea9383f80ffeb95862e7747c630d42c5d3c05bdd35a
```

Store weights outside Git, for example in `models`. No model or pose/parser
weights are bundled with this repository.

A prepared manifest references two 576x864 RGB PNGs and two 576x864
grayscale PNGs, plus category and optional output crop. The runtime does
not implicitly resize malformed prepared inputs. Create them with either
the optional native preparer below or the pinned Python preparation tools
in [the developer guide](fashn_vton.md). Do not supply raw photographs as
prepared conditions.

Run a prepared request:

```powershell
.\build\bin\Release\sd-cli.exe --mode try_on `
  --diffusion-model .\models\model.safetensors `
  --try-on-inputs .\prepared\manifest.json `
  --steps 20 --cfg-scale 1.5 --flow-shift 1.5 `
  --skip-cfg-last-n-steps 1 --seed 42 --rng cpu `
  --type bf16 --diffusion-fa -t 16 -o result.png
```

`--type bf16` selects lower weight storage with the validated F32 matrix
computation path; it does not enable the rejected direct BF16 arithmetic
experiment. The fixed released canvas is 576x864. Generation is followed
by an explicit crop, not output resizing.

For the lower-memory modulation-cache/fused-GELU options, use existing
model arguments and leave mmap/eager loading disabled:

```powershell
$modelArgs = '{"fashn_modulation_cache":true,"fashn_modulation_cache_mib":128,"fashn_fused_gelu":true}'
# Append --model-args $modelArgs to the command above in PowerShell 7.
```

Both optimizations are opt-in. The cache must release modulation weights,
not retain them alongside the cached vectors. For zero runtime conversion,
export a matching runtime-policy GGUF using `export_fashn_runtime_weights.py`;
the exact recipe and verification rules are in the developer guide.

## Optional Python-free raw preparation

Obtain the official OpenCV 4.12 C/C++ package and ONNX Runtime 1.22.1 C/C++
package. A Python `opencv-python` wheel is not the C++ SDK.

```powershell
cmake -S . -B build-native -G "Visual Studio 17 2022" -A x64 `
  -DSD_BUILD_TESTS=ON -DSD_BUILD_EXAMPLES=ON -DSD_FASHN_PREPROCESS=ON `
  -DSD_SERVER_BUILD_FRONTEND=OFF `
  -DOpenCV_DIR="<opencv-package>\build" `
  -DONNXRUNTIME_ROOT="<onnxruntime-win-x64-1.22.1>"
cmake --build build-native --config Release --parallel
```

Download the pinned DWPose weights separately using
`scripts\prepare_fashn_vton.py download-pose` and the documented development
environment, or provision the exact hash-verified files yourself.

Parser-free flat-lay preparation:

```powershell
.\build-native\bin\Release\sd-fashn-prepare.exe `
  --dwpose-dir .\models\dwpose `
  --person-image person.png --garment-image garment.png `
  --category tops --garment-photo-type flat-lay --output prepared
```

Worn-garment photos or person masking additionally require the separately
exported local human parser and explicit
`--accept-parser-research-license`. The parser inherits a **noncommercial
research/evaluation-only** license; it is not covered by the core code
license and is not bundled. Consult the developer guide before preparing it.

`--ort-no-arena` lowers retained preparer memory without changing the
default threading policy. Actual ONNX parser session creation is lazy;
startup checks consent/hashes, and first parser use rechecks artifacts.

## Local server and browser

```powershell
.\build\bin\Release\sd-server.exe `
  --diffusion-model .\models\model.safetensors `
  --type bf16 --diffusion-fa --rng cpu -t 16 `
  --listen-ip 127.0.0.1 --listen-port 1234
```

Open `http://127.0.0.1:1234/try-on` for prepared-input jobs.
An SDK-enabled server can additionally configure `--try-on-dwpose-dir`,
`--try-on-parser-dir`, explicit parser consent and
`--try-on-ort-no-arena`. Raw uploads remain unavailable without native
preparation support/configuration.

This is a local experimental service, not authenticated production hosting.
Cancellation is cooperative and may wait for a complete CPU forward.
The 512 MiB image-storage budget is separate from the job count/TTLs and
does not cap model, ORT, graph or total process memory.

## What the published results establish

For the same 20-step cardigan workload on the evaluation VM:

| Configuration | Sampling interval | Peak working set | Sampled peak private commit |
|---|---:|---:|---:|
| Original Python F32, historical cardigan | 1241.33 s, sampler + PIL | N/A | N/A |
| Floating CPU, optimized | 2371.13 s | 2.01 GiB | 2.14 GiB |
| Floating OpenBLAS, diagnostic | 1598.14 s | 2.03 GiB | 4.28 GiB |
| Q8, diagnostic | 1863.07 s | 1.40 GiB | 1.53 GiB |
| Mixed, diagnostic | 1898.82 s | 1.53 GiB | 1.65 GiB |

Private commit and working set overlap; do not add them.
The [original Python default sampler](../reports/original-python-baseline.md)
took 1258.04 s process wall for cardigan. It uses batched CFG/default CPU SDPA,
no trajectory recording, and bypasses raw preparation/constructor startup.
Cardigan memory is invalid; the separate bottoms case measured 6.573 GiB
working set and 9.603 GiB private commit. Python was faster in these historical
observations, not a controlled same-interval speed comparison.
Quantized outputs preserve their historical same-policy
pixels, but still fail the floating numerical gates.

The public path therefore stays floating/CPU. Q8/mixed and BLAS are
diagnostic experiments; GPU execution is unvalidated. See
[the reports index](../reports/README.md) for the full chronology,
comparison gallery, exact numerical errors, provenance and rejected ideas.

The inherited upstream CI retains its original branch triggers; it does
not automatically establish FASHN validation on this repository's `main`.
Use the explicit build/CTest commands above.
