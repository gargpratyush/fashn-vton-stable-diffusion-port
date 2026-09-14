# FASHN VTON 1.5 native C++ port

Run local virtual try-on with a native implementation of
[FASHN VTON 1.5](https://github.com/fashn-AI/fashn-vton-1.5), built on
[stable-diffusion.cpp](https://github.com/leejet/stable-diffusion.cpp) and GGML.
This is not a C++ wrapper around Python inference: the transformer, conditioning,
sampling and prepared-input inference execute natively.

**[Build and run](docs/fashn_quickstart.md)** |
**[API and developer guide](docs/fashn_vton.md)** |
**[Comparison results](reports/q4-q5-results.md)** |
**[Original Python baseline](reports/original-python-baseline.md)** |
**[Project history](reports/README.md)** |
**[Tests and maintenance](docs/fashn_quality.md)** |
**[ARM64 / Android plan](docs/fashn_android_plan.md)**

## What this project implements

- Native FASHN transformer and pixel-space sampler, including CFG, RNG,
  category conditioning and the released 576x864 canvas. No VAE or text encoder.
- Prepared-input C API and CLI, optional Python-free pose/parser preprocessing,
  and a local asynchronous HTTP server with a standalone `/try-on` interface.
- Lower-memory floating inference: BF16/F16 matrix storage with F32 computation,
  runtime-ready GGUF, bounded modulation caching with weight retirement,
  F32 flash attention and opt-in strict fused GELU.
- Diagnostic Q8/Q4/Q5 and selective mixed precision, with exact conversion
  verification, complete image trajectories, memory profiling and reproducible
  numerical/visual comparisons.
- ARM64 transfer/integrity tooling and native Android CPU execution demonstrated
  by a complete Samsung S23 Q4_K cardigan run; numerical acceptance remains open.
- Reliability safeguards: scoped C API ownership and exception handling,
  worker failure/shutdown handling, a separately testable try-on runtime,
  versioned experiment recovery, and focused native/Python/UI CI.

**Validated deployment path: Windows x64 floating CPU inference.**
Quantized inference and OpenBLAS remain diagnostic-only; public generation
rejects quantized checkpoints. One Android Q4_K generation completed, but failed
strict same-policy numerical parity; it is not production/device acceptance.
A separate Windows ARM64 SDXS smoke test does not establish FASHN support on
Windows ARM64. GPU execution is unvalidated here.

## Finding your way around

| You want to... | Start here |
|---|---|
| Run a prepared try-on or enable preprocessing/server mode | [Quickstart](docs/fashn_quickstart.md) |
| Integrate the C API | [`include/stable-diffusion.h`](include/stable-diffusion.h), [API guide](docs/fashn_vton.md) |
| Understand model detection, weights and the transformer | [`src/model/diffusion/fashn_vton.h`](src/model/diffusion/fashn_vton.h), [`fashn_vton_model.h`](src/model/diffusion/fashn_vton_model.h) |
| Follow request execution and sampling | [`src/runtime/fashn_try_on.cpp`](src/runtime/fashn_try_on.cpp), [`fashn_vton_sampling.h`](src/runtime/fashn_vton_sampling.h) |
| Change CLI, HTTP jobs or the try-on UI | [`examples/cli`](examples/cli), [`examples/server`](examples/server); the owned UI is `try_on.html` / `try_on.js` |
| Work on optional native pose/parser preparation | [`examples/fashn-preprocess`](examples/fashn-preprocess) |
| Reproduce precision results or recover a new experiment | [`scripts`](scripts), [maintenance commands](docs/fashn_quality.md), [study plan](docs/fashn_q4_q5_plan.md) |
| Inspect actual images and numerical evidence | [Results](reports/q4-q5-results.md), [offline HTML](reports/fashn-all-comparisons.html), [history index](reports/README.md) |
| Build/test the port or prepare ARM64/Android work | [`tests`](tests), [test tiers](docs/fashn_quality.md), [transfer runbook](docs/fashn_arm_transfer.md) |

The prepared-input path is **C API / CLI / HTTP -> FASHN request runtime ->
sampler -> transformer -> GGML**. Raw-image preprocessing is an optional stage
before that path. `src/core` and `src/model_io` provide shared execution and
loading; `ggml`, `thirdparty`, and the separate server `frontend` are dependencies,
not the primary places to edit this port.

## Actual try-on comparisons

The contact sheets show **actual outputs, not amplified error maps**.
Top row: prepared person, prepared garment, original Python, native BF16/F32,
Q8. Bottom row: Q4_0, Q4_K, Q5_0, Q5_K, and Q5_K with four original-F32 matrices.
Thumbnails are resized only for display; metrics use the original unresized crops.

### Cardigan

![Cardigan: prepared inputs, original Python and native precision outputs](reports/comparison-gallery-precision/cardigan-overview.png)

### Bottoms

![Bottoms: prepared inputs, original Python and native precision outputs](reports/comparison-gallery-precision/bottoms-overview.png)

**[Download/open the all-in-one interactive HTML report](reports/fashn-all-comparisons.html).**
Use GitHub's download button, then open the file locally; GitHub displays HTML
as source. The approximately 25 MB file embeds all images/data and works offline.
It includes side-by-side/overlay comparisons, selectable Q8/BF16/Python references,
local error crops, PSNR/RMSE, memory, latency, per-step drift and historical controls.
Error maps are hidden by default and explicitly labeled.

Image fixtures and adaptations have separate attribution and
**noncommercial/share-alike** requirements: read [report notices](reports/NOTICE.md).
Numerical agreement is not proof of garment fidelity or a commercial-use license.

## Measured latency, memory and image differences

Current study: Windows x64 CPU, AMD EPYC 7763 exposure, 16 inference threads,
20 steps, CFG 1.5, seed 42, 576x864 canvas / 576x768 displayed crop.
Same prepared inputs, ready weights, F32 flash attention, modulation cache,
strict fused GELU and recorded trajectories for every current policy.

| Implementation / policy | Mean sampling interval | Mean peak working set | Crop RMSE vs BF16: cardigan / bottoms |
|---|---:|---:|---:|
| Original Python F32, historical default sampler | 20.9 min, sampler + PIL | N/A; bottoms alone: 6.573 GiB | 0.0072 / 0.0058 |
| BF16 storage / F32 compute | 38.7 min | 2.01 GiB | Reference |
| Q8_0 | 30.5 min | 1.40 GiB | 1.59 / 0.42 |
| Q4_0 | 34.1 min | 1.07 GiB | 3.49 / 1.79 |
| Q4_K | 30.0 min | 1.07 GiB | 3.00 / 1.24 |
| Q5_0 | 37.1 min | 1.15 GiB | 2.42 / 0.95 |
| Q5_K | 36.5 min | 1.16 GiB | 2.28 / 1.02 |
| Q5_K + 4 original F32 matrices | 35.9 min | 1.30 GiB | 2.23 / 0.87 |

The [original Python baseline](reports/original-python-baseline.md) uses the
unmodified upstream sampler/default CPU SDPA and batched CFG, with the same
prepared cases and 16 threads. It bypasses raw preparation/pipeline construction
and records no intermediate trajectory. Its sampler/PIL interval differs from
native recording; these are historical observations, not controlled speedups.
Python process wall was 20.97 / 22.09 min for cardigan / bottoms.
Only bottoms has valid interpreter memory: 6.573 GiB working set and 9.603 GiB
private commit. Cardigan memory and the two-case Python memory mean are unavailable.

Native times and working sets are arithmetic means across two cases, not confidence
intervals. Sampling includes recording and modulation preparation; raw-image
preprocessing and HTTP overhead are excluded. RMSE uses byte-channel levels
0..255, lower is closer. Working set and private commit overlap and must not
be added; both are reported separately in the detailed results.

Q4_K reduced mean peak working set by about **47%** and mean sampling time by
about **22%** versus the current floating control. Q5_K had the lowest average
cropped error among the four lower-bit formats. Three same-seed Q5_K runs
produced identical final floats/pixels, with **2.58% sampling-time CV**.

These are diagnostic results, **not quantized quality acceptance**. All
quantized policies fail the unchanged strict floating trajectory gates.
Four-matrix restoration improves aggregate error but slightly worsens the
cardigan's matched worst-error region. Original Python was faster in the
historical CPU observations; those are labeled separately, not pooled into
the current study or presented as controlled speedup claims.

See [full measured results and caveats](reports/q4-q5-results.md) and
[the sequential experiment plan](docs/fashn_q4_q5_plan.md).

## Build and run

Clone with the pinned submodules:

```powershell
git clone --recurse-submodules https://github.com/gargpratyush/fashn-vton-stable-diffusion-port.git
Set-Location fashn-vton-stable-diffusion-port
```

From a Visual Studio 2022 developer shell with CMake:

```powershell
cmake -S . -B build -G "Visual Studio 17 2022" -A x64 `
  -DSD_BUILD_TESTS=ON -DSD_BUILD_EXAMPLES=ON `
  -DSD_FASHN_PREPROCESS=OFF -DSD_SERVER_BUILD_FRONTEND=OFF `
  -DGGML_BLAS=OFF -DGGML_LLAMAFILE=OFF
cmake --build build --config Release --parallel
```

Obtain the pinned [FASHN weights](https://huggingface.co/fashn-ai/fashn-vton-1.5/tree/7720683168567eb5a2a4c67f15116c6e29c83ded)
and create a prepared-input manifest as described in
[the quickstart](docs/fashn_quickstart.md). A prepared manifest contains
normalized-input PNG assets and pose maps; raw photos are not interchangeable.

```powershell
.\build\bin\Release\sd-cli.exe --mode try_on `
  --diffusion-model .\models\model.safetensors `
  --try-on-inputs .\prepared\manifest.json `
  --steps 20 --cfg-scale 1.5 --flow-shift 1.5 `
  --skip-cfg-last-n-steps 1 --seed 42 --rng cpu `
  --type bf16 --diffusion-fa -t 16 -o result.png
```

The quickstart covers optional native preprocessing, the local web server,
memory optimizations and development dependencies. Python is used for
reference experiments and tooling, not the deployed prepared-input runtime.

## ARM64 and Android

**A Samsung S23 (8 GB RAM) completed a native Q4_K cardigan generation:**
20 steps / 39 forwards in **6 h 41 min 39 s**, using one inference thread,
approximately **1.03 GiB peak sampled RSS**, and **640 MiB peak sampled swap**.
Maximum battery temperature was 41.3 C and thermal status reached MODERATE;
the powered run needed no watchdog abort.

The crop was close to x64 Q4_K in pixel space (RMSE 0.799, PSNR 50.08 dB),
but final float relative L2 **0.00824 failed the 0.001 gate**. This demonstrates
execution, not fast or numerically accepted deployment. It tested source
`e86c564`, not the later maintenance changes.
See [the Android run details and actual image](reports/android-s23-q4k/README.md).
For context, original Python on the 16-thread Windows CPU took 20 min 58 s
process wall for cardigan; no original-Python baseline was measured on the S23.
This is not a same-device speed comparison.

Use [the transfer runbook](docs/fashn_arm_transfer.md) and
[host preparation evidence](reports/android-host-preparation.md).
A Git checkout supplies code, **not weights or prepared reference fixtures**.
The frozen private ARM transfer bundle targets a specific source baseline;
do not overlay it blindly onto a newer checkout.

## Reproducibility and licensing

- [Complete implementation and experiment history](reports/fashn-vton-project-history.html).
- [Historical memory/latency optimization results](reports/memory-and-latency-optimization-results.md).
- [Original Python and Q8/mixed study](reports/quantization-and-upstream-comparison.md).
- [Current study evidence](reports/evidence/q45-study/results.json) and
  [publication provenance](reports/precision-publication-manifest.json).
- [Contribution guidelines](CONTRIBUTING.md), [core code license](LICENSE),
  [image/model notices](reports/NOTICE.md) and
  [native preprocessing notices](examples/fashn-preprocess/NOTICE.txt).

Weights, executables, SDKs, virtual environments and full trajectory tensor
payloads are not distributed in this repository. The optional human parser
has a separate **noncommercial research/evaluation-only** license and requires
explicit consent. Upstream stable-diffusion.cpp/GGML code and attribution
remain intact; this fork's landing page describes only the FASHN project.
