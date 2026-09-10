# FASHN VTON 1.5 native implementation: complete history and evidence

## Scope and reading guide

This report records the work completed through checkpoint 25 and the subsequent
comparison/quantization experiments through checkpoint 30, including the final
independent cardigan images and matched floating recorder control.
Historical measurements are preserved rather than replaced by newer, more
favorable numbers. "Passed" always refers to a stated contract or numerical
criterion, not universal visual quality or production readiness.

Repository: `C:\source\stable-diffusion.cpp`.
Experimental outputs: `C:\source\fashn-vton-reference`.
The implementation is uncommitted. No PR, push, or model upload has been made.
Model files, generated images and machine-specific reports stay outside the
source tree. The general-purpose frontend, GGML submodule and vendored
third-party sources have not been modified.

The requested experiment batch is complete. The final comparison gallery is
`comparison-gallery-v6\index.html`; the detailed numerical/performance report
is `quantization-and-upstream-comparison.md`, and the hash-linked full-image
snapshot is `full-image-precision-summary.json`. No model experiment remains
running. Public quantized inference and GPU execution remain gated.

The native workflow is real end-to-end local inference, not a Python server
hidden behind a C++ wrapper. However, it is not a rewrite of every dependency:
we reuse GGML, stable-diffusion.cpp's loader/runner/server infrastructure,
OpenCV and ONNX Runtime, and the published detector/parser model weights.

### An important distinction about the Python comparison

There are three different execution paths:

| Path | Model arithmetic | CFG execution | Attention selection |
|---|---|---|---|
| Native validated path | F32; optionally BF16/F16 matrix storage with F32 upcasts | Separate conditional/null forwards; redundant null forwards omitted | Manual by default; optional F32 flash |
| Numerical Python oracle used through checkpoint 25 | Original pinned model in F32 | Separate conditional/null forwards; redundant null forwards omitted | Explicit PyTorch SDPA MATH |
| Original Python `TryOnPipeline._sample` | F32 on CPU; upstream optionally BF16 on supported CUDA | Batched conditional/null every step, including the final skipped-CFG step | Default PyTorch SDPA dispatch |

The previous oracle is an independent implementation of the reference
sampling mathematics around the original model, designed for numerical
diagnosis. It is **not an unchanged invocation of the upstream sampler**.
It would be misleading to treat its execution time as the fastest original
Python pipeline. The new experiments explicitly run the unmodified upstream
`_sample` on identical prepared inputs and distinguish that result.

Prepared-input timing excludes raw pose/parsing preparation. The upstream
benchmark harness bypasses the constructor and uses meta-device weight
loading to avoid initializing unused random weights; it does not benchmark
upstream raw `__call__` startup. These scope differences are recorded.

## 1. Machine assessment and initial stable-diffusion.cpp setup

- Windows 11 x64 VM, AMD EPYC 7763 CPU exposure, 8 cores / 16 logical processors,
  64 GB RAM, AVX2/FMA. The CPU marketing name reports more physical cores than
  this VM exposes; thread recommendations use the VM's actual processors.
- No exposed compute GPU. No CUDA/Vulkan/ROCm speed or correctness claims.
- Installed/cloned stable-diffusion.cpp under `C:\source`, configured Visual
  Studio 2022/CMake CPU builds, and produced working CLI/server binaries.
- Generated an SD 1.5 Q4_0 256x256 four-step smoke image before FASHN work.
- Rechecked the original SD 1.5 output during integration; its pixels remained
  unchanged. FASHN-specific numerical fixes did not change other model defaults.

Pinned sources:

| Component | Revision |
|---|---|
| stable-diffusion.cpp base | `d04e8950c1ec8d30248cbe996682b3182fb1adf6` |
| GGML | `e20c3a14aa70ee84ca58499814206dd08d8026bc` |
| FASHN source | `7c0f10af3f91ad4048fe9729c470a13ef905d25a` |
| Released checkpoint revision | `7720683168567eb5a2a4c67f15116c6e29c83ded` |
| Checkpoint SHA256 | `d6cd38286885bc29fa487ea9383f80ffeb95862e7747c630d42c5d3c05bdd35a` |

The checkpoint is 1,943,668,048 bytes: 366 BF16 tensors, 971,814,240 scalars.
There are 365 inference tensors; `patch_mixer_token` is an optional unused
buffer. BF16 checkpoint storage does not imply BF16 inference arithmetic.

## 2. Architecture investigation and integration design

Read the pinned FASHN architecture, sampling, preprocessing, tensor utilities
and checkpoint metadata. Mapped them against stable-diffusion.cpp's model
families, loader, GGML operations, runner, C API, CLI, server and conversion
paths. Produced the requested detailed staged design before implementation.

The architecture we implemented is the released fixed preset:

| Property | Value |
|---|---|
| Representation | Pixel-space RGB; no VAE, text encoder or text prompt |
| Canvas | 576x864 |
| Target input | Noisy RGB + person RGB + grayscale pose = 7 channels |
| Garment input | Garment RGB + grayscale pose = 4 channels |
| Patches | 12x12, 48x72 grid, 3456 tokens per stream |
| Joint attention tokens | 6912 |
| Hidden / MLP width | 1280 / 5120 |
| Attention | 10 heads, head dimension 128 |
| Blocks | 4 patch mixers, 8 double-stream, 16 single-stream |
| RoPE | Axes 16/56/56, theta 10000, coordinates `[0,row,column]` |
| Categories | Null=0, tops=1, bottoms=2, one-pieces=3 |
| Native graph batch | One sample per graph; 1-4 output samples sequentially |

Existing FLUX-like blocks were useful building blocks, but the patch
embedders, conditioning, patch-mixer arrangement, masks, output layout and
pixel-space sampler needed FASHN-specific wiring.

## 3. Reference oracle and reproducibility foundation

Implemented `scripts\export_fashn_vton_reference.py` and reference tests:

- Exact source revision and clean-source checks.
- Checkpoint header, shape, dtype and whole-file hash validation.
- Local original model loading, expanded to CPU F32.
- Deterministic noise, fixed categories, schedule and CFG branches.
- Conditional and normalized-null forwards at time zero and a nonzero time.
- Named captures at embeddings, positional embeddings, patch mixers,
  double/single blocks and final velocity.
- Native-layout fixture adaptation, including integer/category conversion
  and singleton dimensions.
- Prepared PNG export only when normalized conditions round-trip correctly.
- Synthetic geometry/primitive fixtures clearly distinguished from real
  photographic try-on.
- Euler and short/full trajectory exports.
- Pinned environments and local artifacts; no runtime model downloads.

The initial reference suite passed 25 tests. A repeated full-resolution
reference export reproduced all 77 captured tensors exactly.

Environments are isolated:

- `.venv`: parser-free model/oracle and basic comparison work.
- `.venv-parser-eval`: authorized human-parser evaluation and original
  pipeline imports.
- `.venv-browser`: Playwright/Chromium checks.

Core versions include Python 3.12.10, CPU torch 2.8.0, torchvision 0.23.0,
NumPy 2.2.6, Pillow 11.3.0, safetensors 0.6.2, ONNX Runtime 1.22.1 and
OpenCV 4.12.0.88. Headless and regular OpenCV are not installed together
in one environment.

## 4. Native checkpoint recognition and validation

Implemented a FASHN version/config, detection from characteristic tensors,
released-preset shape validation, raw/canonical tensor names, and collision
rejection before loading or conversion. A malformed or unsupported preset
fails explicitly rather than being treated as another model family.

Integrated model version/name handling and loader registration. The early
implementation rejected inference before backend execution until its graph
was available. Library, CLI, server and actual invalid-context cases were
exercised. Optional unused weights are not mistaken for required execution
parameters.

Important implementation locations:
`src\model\diffusion\fashn_vton.h`, `fashn_vton_model.h`, `model.h`,
`model_loader.cpp`, `name_conversion.cpp`, `convert.cpp`, and model dispatch.

## 5. Complete native transformer and the GELU correction

Ported the complete target/garment embedding path, time/category conditioning,
position/RoPE arrangement, four patch mixers, joint double-stream blocks,
single-stream blocks and final RGB reconstruction. Reused existing runner,
tensor operations and compatible FLUX components without changing their
default semantics for other model families.

Added full-graph diagnostics and named activation captures. Checked every
operation for backend support. Model parameters are resident and execution is
monolithic; this is not an offloaded/chunked graph implementation.

**Failure and fix:** GGML's ordinary F32 GELU used an FP16 lookup-table route.
F32 tensor labels therefore did not guarantee the desired arithmetic. Added
FASHN-only full-F32 tanh-GELU behavior. Its primitive sweep covers 24,001
values over [-12,12], with a tight error bound. Full-model comparisons then
passed without relaxing the model's numerical gates.

Tensor layout required care: native tensors list the most contiguous dimension
first, unlike PyTorch. GGML can omit a trailing singleton batch dimension.
Fixture adaptation/recording restores the logical shape rather than assuming
that a missing singleton means a different batch.

## 6. Native sampling, noise and pixel semantics

Implemented the ascending 0-to-1 shifted rectified-flow schedule and:

`x += dt * (vu + cfg * (vc - vu))`

The final skipped-CFG steps use conditional velocity alone. CFG=1 omits null
forwards entirely. Null person/garment/pose tensors contain exact normalized
zero with category 0; they are not black PNGs, which normalize to -1.

Seed handling matches a single logical-batch random draw. Multiple samples
are slices of that draw, not independent calls with `seed+i`.
Pixel conversion clamps `(x+1)/2`, multiplies by 255 and truncates to uint8.
Cropping follows generation and never resizes the output.

Implemented parameter validation, sequential sample handling, progress,
cancellation and cleanup. Added an optional internal trajectory observer;
observer failure propagates as generation failure. Ordinary generation does
not install the observer.

## 7. Prepared-input C API and CLI

Added an independent try-on API without altering existing public struct layouts:

- Initialized, size-versioned `sd_try_on_params_t`.
- Model capability query and try-on generation entry points.
- Caller-owned input buffers, explicit output ownership/freeing.
- Failure clears returned output pointers/count.
- Request-local progress/cancellation callbacks.
- Generic image/video generation rejects FASHN contexts.
- Do not concurrently generate on one model context.

CLI `--mode try_on` accepts a strict prepared manifest and four lossless
576x864 images: two RGB conditions and two grayscale poses. Manifest-relative
paths work independently of the caller's working directory.

Rejects unknown fields, incompatible options, wrong channel counts/sizes,
HDR/16-bit inputs and invalid crop/category/sampling values. No implicit
resize makes a malformed prepared request appear successful.

Defaults: 30 steps, CFG1.5, shift1.5, skip-last1, CPU RNG, fixed canvas.
Supports 1-4 sequential samples. Two-step runs are integration diagnostics,
not quality recommendations.

## 8. Python raw-image preparation and parser authorization

First established a parser-free flat-lay path using pinned CPU DWPose and
original transforms. A flat lay gets the original blank garment pose.
Verified synthetic geometry, real-person pose, crop and normalized inputs.

Worn garments and masked-person preparation require the human parser.
The NVIDIA-derived parser's research/evaluation restriction was explained,
and the user explicitly authorized local research/evaluation use.

Implemented separate environment/dependency pins, local parser files,
explicit consent, hash checks, original logits and category/masking behavior.
All 12 category/photo-type/person-masking combinations matched the reference,
with photographic follow-up. A native photographic two-step comparison
differed in only 47 channels, by at most one byte level.

This authorization and dependency isolation are not an unrestricted commercial
license. The parser is not silently loaded in parser-free mode.

## 9. Asynchronous prepared-input HTTP service

Added FASHN capabilities and strict prepared-input requests to the existing
server queue. Implemented job publication, queued/running/terminal states,
progress, cancellation and PNG result delivery.

Covered malformed requests, uint64 seeds, failed jobs, queue capacity,
concurrent publication, queued cancellation and exact HTTP/CLI output.
One regression round covered 31 invalid requests and 63 queued cancellations.
Listener/request/result limits were kept explicit rather than allowing
unbounded raw decode or work queues.

This is a trusted local experimental service, not authenticated production
hosting or durable job storage.

## 10. Baseline measurement and F32 flash attention

Added a Windows process benchmark harness with:

- Fresh process per repeat.
- Exact command and hashes for executable, checkpoint, manifest, inputs/output.
- Wall time including model load and PNG encoding.
- Kernel peak working set through the last successful sample.
- Sampled private-memory peak, sample count and last sample time.
- Explicit timeout/failure handling and PNG mode/crop validation.

Historical one-step, CFG1, eight-thread photographic results:

| Configuration | Wall seconds | Peak working set |
|---|---:|---:|
| Manual F32 attention / F32 residency | 113.03 | 5.896 GiB |
| F32 flash / F32 residency | 81.87 | 4.217 GiB |
| F32 flash / BF16 core-matrix residency, F32 compute | 81.83 | 2.950 GiB |
| F32 flash / F16 core-matrix residency, F32 compute | 82.54 | 2.950 GiB |

**Failed flash attempt:** the shared helper cast keys/values to F16. Merely
requesting F32 accumulation did not undo that rounding. Three late conditional
captures failed; maximum relative L2 was 0.0011681 versus the 0.001 gate.

**Successful flash path:** preserved F32 keys/values for FASHN only. All 28
attention nodes must actually be flash nodes. Unsupported fallback is not
silently advertised as flash. Other models retain their prior F16-K/V behavior.

Time-zero conditional/null velocity relative L2 was 6.93e-7 / 5.03e-7;
the nonzero probe was 7.51e-6 / 2.13e-6. All capture gates passed.
Manual/flash one-step outputs differed in 64 byte channels, each by at most
one level. A separate two-step flash run took 235.25s / 4.222GiB and differed
from the oracle in 81 channels, each by one level.

These are small-sample historical observations, not universal speedups,
confidence intervals or perceptual similarity scores.

## 11. First recommended-fast photographic image

The original seated person / worn black-shirt pair completed at 20 steps,
CFG1.5, shift1.5, skip-last1, seed42, eight threads, F32 flash and F32 residency:
**2985.95s (49m46s), 4.222GiB**.

The result had much sharper shirt lettering, hands, trousers and boots than
the short smoke trajectory. Broad pose/background remained recognizable,
but fit, folds and nearby details were regenerated. Later checkpoint-21
work supplied the initially missing full 20-step numerical comparison.

Evidence: `checkpoint12-photo-20-step`.

## 12. Active cancellation and standalone browser interface

Added request-local C callback APIs, atomic progress/cancel state and server
integration. Accepted cancellation wins the completion/encoding race.
Cancelling one request does not contaminate the next request.

Cancellation is cooperative, between native stages/forwards. It cannot
interrupt a CPU attention kernel immediately. An early active cancellation
after the first step took about 78.42s in that experiment.

Added an embedded `/try-on` page, separate from the frontend submodule:
manifest/prepared image upload, exact uint64 seed strings, progress, cancel,
download, errors and transport retry without duplicate submission.

## 13. Native raw-image preparation

Implemented optional OpenCV/ONNX Runtime preprocessing and
`sd-fashn-prepare`. The core prepared-input build requires neither SDK.

Native parity includes:

- Pillow-compatible Lanczos3 initial fit, no upsampling, dimension truncation,
  22-bit coefficients and intermediate uint8 clipping.
- OpenCV area downsampling/Lanczos4 upsampling for the final fit.
- Nearest-exact pose resizing.
- Pinned YOLOX/RTMPose detector execution and original neck/visibility mapping.
- Double-coordinate affine/postprocessing where required.
- Original grayscale pose rendering, normalization, masks and crop.
- Authorized fixed-shape parser ONNX export with provenance/weight/license hashes.
- Manifest/provenance output and strict failure reporting.

**Failed interpolation attempt:** OpenCV resizing of parser logits flipped a
boundary label. Replaced it with explicit Torch-compatible F32 half-pixel
bilinear interpolation; did not loosen the exact-pixel preparation criterion.

All 12 original mode combinations, primitive transforms and later five
distinct quality preparations matched all four PNGs and crops exactly.

Native runtime does not invoke Python, subprocesses, downloads or temporary
files for HTTP preparation.

## 14. Precision residency and conversion experiments

Implemented F32/F16/BF16 GGUF conversion and selective Q8_0 diagnostic
conversion, with metadata/payload checks across all 366 tensors.

| GGUF file | Bytes |
|---|---:|
| F32 | 3,887,290,272 |
| BF16 | 1,943,661,792 |
| F16 | 1,943,661,792 |
| Selective Q8_0 | 1,808,160,672 |

104 core matrices are eligible: patch-mixer/single-block linear weights and
double-block attention/MLP weights. Embeddings, patch kernels, modulation,
norms, biases and final layers stay protected.

Do not confuse converted-file size with native resident memory: floating
GGUF files can store protected parameters in reduced precision, while the
native validated graph expands protected parameters to F32.

Raw arithmetic experiments, 34 captures over two branches, on the original
synthetic full-canvas stress fixture (not a photographic quality case):

| Matrix arithmetic | Failed captures | Largest relative L2 |
|---|---:|---:|
| F16 | 6/34 | 0.002438 |
| BF16 | 9/34 | 0.013573 |
| Q8_0 | 19/34 | 0.064830 |

Q8 conditional final-velocity relative L2 was about 0.008893.
Its first captured patch mixer was already at 0.003694; the last captured
patch mixer was at 0.005658. The large late stream error is not proof that
one particular late matrix caused it: earlier errors propagate downstream.

GGML's ordinary CPU quantized matrix products also round/quantize activation
operands. F32 accumulation cannot recover information removed before the
dot product. Quantized checkpoint conversion working is therefore not enough
to establish acceptable inference.

**Successful compromise:** retain eligible matrices as BF16/F16 but cast
each to F32 immediately before multiplication. Protected weights remain F32;
all 146 matrix multiplications request F32 precision. Both storage variants
passed both conditional/null probe sets. BF16 one-step pixels matched F32
exactly; F16 differed in 29 channels, each by one level.

Public Q8 inference stayed intentionally rejected; Q4/Q5 were not enabled.
The new work now permits complete Q8 generation only in the diagnostic runner,
so visual evidence can be gathered without disguising a failed public gate.

## 15. Shared native preparation, raw HTTP and real-browser coverage

Extracted reusable native preparation for both CLI and server. Raw HTTP mode
is opt-in and advertises its availability. Added person/garment uploads,
category, garment-photo type, masking, preparation progress/cancellation,
and UI capability gates.

Bounds include 4,194,304 decoded pixels per HTTP image, dimensions at most
4096, and a 3MiB encoded-string limit. Queued raw inputs retain encoded
strings, not decoded full images. Standalone raw preparation has a 20MP limit.
EXIF orientation is intentionally not applied, matching the reference path.

Raw/native-prepared/Python-prepared outputs were pixel-equivalent under the
tested settings. Actual Chromium upload, inference, download and cancellation
passed, with no page errors or external browser requests.

The browser checks use an isolated Playwright environment; they are not
merely JavaScript mocks. Focused Node tests additionally cover state races.

## 16. Backend diagnostics, packaging and dependency-free operation

Added explicit device listing/selection, operation support checks, precision
reporting and actionable rejection of unsupported backends. CPU primitives
and both full-graph oracle probes passed. No available GPU means no GPU
execution promotion; the public one-CPU-backend guard remains.

Built SDK-enabled and core-only variants. Installed the native preparer with
required OpenCV/ORT DLLs and licenses, but no weights. Tested installed
preparation with only its bin directory and Windows System32 on PATH.
Tested the dependency-free server with only System32 on PATH.

Prepared input works without preprocessing SDKs. Raw preparation is exposed
only where compiled/configured. Public/library/CLI/server/browser documentation
records these distinctions rather than implying universal feature availability.

## 17. Full 20-step numerical trajectories: checkpoint 21

Added a recorder around the actual native sampler, shared test-runner/export
helpers and a strict Python trajectory comparator. Saves initial noise, every
updated image and guided velocity. Recording failures abort rather than leave
a success-shaped incomplete result.

The one-step recorder matched public CLI pixels exactly. Independent original
model/F32-math-oracle and native 20-step runs were compared at both 8 and
16 native threads:

| Quantity | Result |
|---|---:|
| Initial noise maximum absolute error | 4.76837158203125e-7 |
| Schedule maximum error | 0 |
| Maximum updated-image relative L2 across all steps | 1.37657243835065e-6 |
| Maximum guided-velocity relative L2 across all steps | 1.44126307809698e-6 |
| Updated-image / velocity acceptance gate | 0.001 at every step |
| Final differing byte channels | 101 / 1,492,992 |
| Final maximum byte difference | 1 |
| Mean byte difference | 0.0000676494 |
| Numerical PNG PSNR | 89.8282dB |

The noise gate remains 1e-6 and schedule gate 1e-7. Captured primitives also
use elementwise 1e-4 criteria where applicable. No gate was relaxed to promote
an optimization. PNG PSNR here measures numerical agreement, not garment quality.

Evidence: `checkpoint21-reference-20-step`, `checkpoint21-native-20-step`,
`checkpoint21-native-20-step-t16`, and the corresponding comparison directories.

## 18. CPU profiling and controlled thread/kernel experiments: checkpoint 23

Added per-node operation timing using existing backend evaluation callbacks,
with primitive checks and explicit thread controls.

The profiling is intrusive: it synchronizes each node, prevents cross-node
fusion and includes dispatch costs. It must not be added up and presented as
ordinary end-to-end latency.

Eight-thread instrumented conditional/null costs:

| Operation | Conditional | Null |
|---|---:|---:|
| 146 matrix multiplies | 46.22s | 45.50s |
| 28 flash attention nodes | 19.93s | 19.29s |
| 36 tanh operations | 13.37s | 10.62s |

Layer gates still passed during instrumentation.

Built a separate `GGML_LLAMAFILE=ON` candidate without touching submodules
or installing a new runtime dependency. It passed both oracle probes.
The main build remains on default kernels.

Fresh-process, two-repeat, one-step CFG1 controls:

| Configuration | Runs | Median |
|---|---|---:|
| Default, 4 threads | 111.41 / 112.02s | 111.71s |
| Default, 8 threads | 76.28 / 74.19s | 75.24s |
| Default, 16 threads | 62.72 / 63.61s | 63.17s |
| LLAMAFILE, 8 threads | 68.95 / 69.58s | 69.26s |
| LLAMAFILE, 16 threads | 63.14 / 63.15s | 63.14s |

Peak working set remained about 2.950GiB. All predeclared one-byte pixel
checks passed. Earlier separate candidate runs were 79.09/75.45s; they are
not substituted for these controls.

Selected default kernels at 16 threads: the extra kernel change adds no
meaningful benefit there. Full 20-step numerical parity at 16 threads was
verified separately; a fast one-step image alone was not treated as enough.
Two repeats do not justify confidence intervals or universal speed claims.

## 19. Reliability, restart and graceful shutdown: checkpoint 24

Implemented FASHN-specific cooperative SIGINT/SIGTERM/Windows SIGBREAK handling.
The handler sets only a lock-free atomic flag. A normal thread stops
admission, cancels queued jobs, requests active cancellation, stops the
listener and joins the worker. Non-FASHN signal behavior is unchanged.

Fixed a listener race: httplib `stop()` is ineffective before the listener
is actually running, even if its socket is bound. The monitor now waits for
the appropriate running state or main-thread completion. Failed listener
startup exits nonzero.

Shutdown requests after the publication lock closes receive HTTP503.
Stopping jobs is idempotent. IDs include a random 128-bit process namespace,
so restart does not normally reuse a timestamp/counter job ID. IDs are not
authentication credentials. Old jobs are not restored after restart.

UI recovery:

- 404/410 releases unavailable/expired/restarted-server jobs.
- Transport errors retain the active job so retry does not duplicate it.
- A cancellation response remembers its original polling path and cannot
  discard a newer job.
- Cleared jobs stop polling.

Measured six-request/seven-image repeated-use run:

- Each first sample matched the CLI.
- Final two-sample request returned the correct count/progress; its first
  sample matched the single-image result and the second was distinct.
- Eight clients submitted/cancelled 16 queued jobs successfully.
- Post-warmup adjusted private-memory span: 562,008 bytes, or 0.54MiB,
  versus a predeclared 128MiB budget.
- Raw private memory increased roughly 2394.0 to 2404.4MiB, largely reflecting
  retained encoded results. Accounting subtracts known base64 storage and
  respects advertised completed-result TTL; expired storage is not subtracted
  indefinitely.
- Busy shutdown cancelled two queued jobs and stopped an active job, exiting
  in 73.156s. This is approximately a forward boundary, not immediate interrupt.

Actual restart/old-job404, failed bind and idle shutdown passed in SDK and
core-only builds. Latest core-only lifecycle checks also covered TTL metadata
with System32-only PATH. The later Chromium raw flow retained exact CLI pixels
and no external requests/errors; seven focused UI tests passed.

This is a bounded soak, not a proof of no memory leaks during indefinite use.

## 20. Six-case visual quality baseline: checkpoint 22

Used pinned public CatVTON demo assets at
`999bdbe81e6008a3f5749af7c1e0b0fa3d21b48e`, under the repository's stated
CC BY-NC-SA 4.0 terms, alongside existing FASHN examples. Preserved attribution
and explicit research-use flags. Did not obtain or claim access to DressCode.

Added a resumable quality driver with hash/policy/settings checks, interrupted
attempt archiving, one-case batches, benchmark JSON, PNGs, contact sheets and
review HTML. Completion of one invocation is not completion of the suite;
`execution_passed` requires all expected cases. Human/assistant visual review
is recorded separately from execution.

All five distinct preparations matched independent pinned Python results
exactly. Six generation cases use default CPU kernels, 16 threads, BF16
matrix residency, F32 flash, 20 steps, CFG1.5, shift1.5 and skip-last1:

| Case | Seed | Wall seconds | Visual findings |
|---|---:|---:|---|
| Gray/green cardigan, unmasked person | 42 | 2549.62 | Recognizable knit/trim; distorted chest graphic; hem/front opening reinterpreted |
| Same cardigan, masked person | 42 | 2410.98 | Better lower coverage, but pushed-up/bunched sleeves expose more forearm |
| Floral dress | 42 | 2382.17 | Recognizable flowers/colors; altered neckline/straps and motif geometry |
| Worn beige trousers | 42 | 2491.28 | Color transfers; tailored construction becomes simplified/tight; top shortens |
| Patterned worn crochet top | 42 | 2541.91 | Bands/motifs recognizable; softened stitches and altered front closure |
| Same unmasked cardigan | 43 | 2487.24 | Deeper neckline/different closure; undershirt visible at seed42 disappears |

Median: **2489.26s = 41m29s**. Range: **39m42s to 42m30s**.
Peak working set: approximately **2.957GiB** each.
Combined generation time: **14863.21s**, approximately 4h08m.
These are 39-forward quality runs, not one-step timings or repeated samples
of a single latency distribution.

Common limitations: regenerated non-target jeans/folds/shading, graphics and
construction not copied exactly, changed layering/closure, altered shoe or
background details. Broad facial/pose appearance being recognizable is not
identity verification. Generated exposed legs are not proof of preservation
of anatomy hidden by the original clothing.

Masking was not universally better. A seed change affects styling materially,
not merely fine texture. This is stochastic variation across different seeds,
not observed nondeterminism at a fixed seed.

This tiny unpaired set has no ground-truth worn target images. It supports
neither FID/SSIM quality claims nor production/product-fidelity certification.

## 21. Trouser failure attribution: checkpoint 25

Ran matched 20-step original-model/F32-math reference inference on the exact
bottoms conditions. Verified checkpoint, settings/schedule, prepared pixels,
normalized conditions, artifact hashes and final tensor-to-PNG conversion.
Applied the native manifest crop: x0, y48, width576, height768.

Native/reference outputs differ in only **43 of 1,327,104 byte channels**,
each by one level; mean byte error **0.0000324014**. Visual inspection shows
the same simplified trousers, lost construction and shortened white top.

Therefore these particular visible failures are reproduced by the reference
pipeline, not a native-specific output discrepancy. This does **not** separate
model behavior from preprocessing effects, and is final-image parity rather
than a second all-intermediate-state proof.

Evidence: `checkpoint25-bottoms-reference` and
`checkpoint25-bottoms-comparison`. The comparison script persists both images,
an amplified difference image and numerical/provenance JSON.

## 22. What has and has not been rewritten

| Area | Work performed | Reused |
|---|---|---|
| Model support | FASHN config, recognition, graph and conditioning | GGML tensor/backend infrastructure; compatible FLUX blocks |
| Sampling | FASHN Euler/CFG/noise/category/pixel semantics | Existing RNG/tensor infrastructure |
| Public workflow | New try-on API, CLI mode and strict validators | Existing library/context/model management |
| HTTP | Try-on jobs, raw preparation, progress/cancel, lifecycle fixes | Existing HTTP server and queue foundations |
| Browser | Standalone FASHN page and state handling | Browser APIs; no frontend-submodule rewrite |
| Raw preparation | Native transforms, pose/parser wiring and masks | OpenCV, ONNX Runtime, original learned detectors/parser |
| Precision | Eligibility policy, conversion diagnostics, F32-safe residency | Existing GGML quantizers/converters |
| Evidence | Oracle, graph/trajectory tests, benchmarks, quality/reliability tools | Original model code; standard Python/Node/CTest tooling |

Native advantages demonstrated here are a Python-free deployed inference
path, optional Python-free raw preparation, validated lower resident memory,
an explicit C API, bounded asynchronous service/UI, and better local lifecycle
control. A speed advantage over the actual default upstream Python sampler
has **not been established**. In the first new matched-input cardigan case,
the original sampler's process took1258.04s versus the historical native
2549.62s, while outputs differed in only69 one-level byte channels.
Python was faster in that observation; this is not an interleaved statistical
benchmark. The second, bottoms case took1325.48s in Python versus the
historical native2491.28s, with only44 one-level channel differences.
Verified Python peak working set was6.573GiB versus native2.957GiB:
Python was faster, native used less memory in these observations.
Language choice alone does not establish performance.

## 23. New full-image quantization and surgical mixed-precision work

Started after the historical baseline above:

- Diagnostic trajectory runner accepts Q8_0 for all 104 eligible matrices,
  retaining protected parameters in F32.
- Exact-name JSON exclusions restore only specified eligible matrices to F32.
- Unknown/protected names, duplicates and attempting to restore original
  precision from an already-quantized source are rejected.
- Per-run metadata records the actual type of every eligible matrix.
- Optional Q8-to-F32 upcasting is available to isolate weight quantization
  from the usual quantized-activation dot-product path.
- Public inference gates remain unchanged.
- Shared Windows process measurement records complete diagnostic commands
  without confusing execution success with numerical/visual acceptance.
- First full 20-step Q8 bottoms trajectory completed in 1947.35s. The cropped
  image differs from validated native floating output by 0.130162 byte levels
  on average (maximum41, numerical PSNR55.584dB) and looks very similar at
  normal scale. The strict trajectory gate still fails. See the separate
  quantization/upstream comparison report for details. At this initial stage,
  no improved mixed-precision configuration had been measured; the subsequent
  completed study and full-image results appear below.

The subsequent experimental sequence was: inspect completed Q8 output; rank interventions
that restore individual matrices from the original floating checkpoint;
measure actual downstream improvement rather than equating large layer
error with causal matrix sensitivity; test a small restored subset across
timesteps and held-out images. Interactions and calibration overfitting must
be reported. A single successful image will not silently promote public Q8.

## 24. Remaining boundaries

- GPU execution is hardware-blocked and guarded.
- Public quantized arithmetic remains gated; full-image diagnostics completed
  on two pairs with numerical improvements but failed strict agreement.
- BF16/F16 storage is validated; native reduced-precision arithmetic was not.
- Broad populations, difficult poses/occlusions, many garments/seeds, longer
  30/50-step sweeps and paired/statistical quality assessment remain.
- Production authentication, durable queues/storage, multi-process deployment,
  broad cross-platform certification and general frontend integration are
  outside the implemented local research workflow.
- Parser and demo-image licenses remain restrictions even when the model code
  or core native runtime has different licensing.
- Improvements to actual garment fidelity may require changes beyond a
  faithful port. Such changes must be explicit and separately evaluated.

## 25. Evidence map

| Evidence family | Local directories |
|---|---|
| Initial reference/probes | `oracle-step0`, `oracle-step0-replay`, `oracle-step15`, `native-fixtures-step0`, `native-fixtures-step15` |
| Initial graph/GELU | `native-graph-step0`, `native-graph-step15`, `native-graph-f32-gelu` |
| Early preprocessing/service | `preprocess-checkpoint7*`, `server-checkpoint8*`, `parser-checkpoint9*` |
| Baseline/flash | `checkpoint10-manual-baseline`, `checkpoint11-flash-*` |
| First quality image | `checkpoint12-photo-20-step` |
| Cancellation/UI/native preparation | `checkpoint13-active-jobs`, `checkpoint14-ui-routes`, `checkpoint15-native-*` |
| Precision/conversion | `checkpoint16-*` |
| Shared/raw/browser/backend/package | `checkpoint17-*`, `checkpoint18-browser`, `checkpoint19-*`, `checkpoint20-*` |
| Full trajectory | `checkpoint21-*` |
| Visual suite | `checkpoint22-quality-t16`, `checkpoint22-quality-reference-preparation` |
| CPU study | `checkpoint23-*` |
| Reliability | `checkpoint24-*` |
| Bottoms attribution | `checkpoint25-bottoms-*`, `checkpoint25-core-server-contract` |
| Complete Q8 images and floating recorder control | `checkpoint26-q8-bottoms*`, `checkpoint26-q8-cardigan*`, `checkpoint26-bf16-bottoms-control*` |
| Actual original Python sampler | `checkpoint27-upstream-*`, `checkpoint27-cardigan-comparison-v2`, `checkpoint27-bottoms-comparison` |
| Full104-matrix restoration scan | `checkpoint28-matrix-ablation`, `reports\matrix-sensitivity-complete` |
| Real-state/combined precision probes | `checkpoint28-photo-oracle*`, `checkpoint28-photo-native-fixtures*`, `checkpoint29-policy-evaluation` |
| Four-matrix full images | `checkpoint30-mixed-top4-bottoms*`, `checkpoint30-mixed-top4-cardigan*` |
| Final gallery/evidence snapshot | `reports\comparison-gallery-v6`, `reports\full-image-precision-summary.json` |

Main implementation guide: `C:\source\stable-diffusion.cpp\docs\fashn_vton.md`.
API contract: `examples\server\api.md`. Graph/metadata/C API/sampler tests
are in `tests`; Python/Node reference, benchmark, UI and integration utilities
are in `scripts`. The completed standalone comparison gallery,
`comparison-gallery-v6\index.html`, contains11 rows: two historical oracle
comparisons, two actual original-Python comparisons, two all-Q8/floating,
two mixed/floating, two mixed/all-Q8 and one floating recorder/CLI control.
These overlap three input pairs; the actual-Python and quantization studies
use two pairs, not11 independent cases.
See the detailed comparison report for the corrected Python memory-monitoring
issue; the first run's launcher memory is not valid model-memory evidence.
Photographic calibration and the first eight individual patch-mixer matrix
restorations completed. The best reduced worst-branch velocity relative L2
by only0.044%; six interventions slightly worsened it. No blanket patch-mixer
restoration is justified by those individual results. Coverage subsequently
reached40/104; the best single restoration so far
(`double_blocks.0.img_mlp.2.weight`) reduces the objective by only0.17025%.
All104 interventions subsequently completed. The strongest restoration,
`single_blocks.12.linear1.weight`, reduces worst-branch velocity error by
40.6711%, leaving103 matrices Q8 and adding32.129MiB weight payload. It still
misses the strict gate alone. Full results/costs are in
`matrix-sensitivity-complete`; the earlier partial snapshot remains in
`matrix-sensitivity-snapshot-1`. Top1/top2/top4 and a branch-aware pair were
then evaluated jointly at initial and real step15 states, together with a
fresh all-Q8 baseline and a weight-only Q8/F32-compute control at each state.
All12 probes completed; none passes the unchanged complete capture gate.

The four-matrix subset (single-block linear1 weights12,11,8,14) is the best
small direct-Q8 combination tested at both states. Worst-branch velocity L2
falls from0.0026250719 to0.0012749801 initially (51.4307% reduction) and from
0.0034749969 to0.0012152331 at step15 (65.0292% reduction). It leaves100
matrices Q8 with128.516MiB additional weight payload. Top1, top2 and the
branch-aware pair all improve over all-Q8 but less than top4 at both states.
The step15 input is the actual recorded step14 updated image from the
floating reference. It is teacher-forced on the same photographic case,
not a separate image or free-running mixed-precision trajectory.

Weight-only Q8 followed by F32 computation does better on these objectives
(0.0010505147 initially,0.0010230602 later), but still fails16/34 captures
at each state. Rounded Q8 weights cannot be restored by dequantization.
Top4 still fails17/34 and20/34 captures, respectively. The full table and
branch/internal errors are recorded in `quantization-and-upstream-comparison.md`
and `..\checkpoint29-policy-evaluation\evaluation.json`.

Top4 is an experimental full-image candidate, not a public policy or a
proven minimal subset. Complete mixed bottoms generation finished in
2007.3931s (33m27s), with4.277GiB peak working set. Cropped MAE against
floating native output fell from0.13016237 to0.05832625 (55.1896% lower);
PSNR rose from55.58397dB to58.89603dB and maximum byte error fell41 to30.
Final trajectory-image L2 fell35.2206% to0.0026928717 and maximum velocity
L2 fell62.8437% to0.0031627102. Four image states (indices16-19) and all20
guided velocities still fail the0.001 gate. The all-Q8 run failed9 image
states and20 velocities. With an identical recorder executable, measured
mixed wall time was3.0835% higher than all-Q8, in single non-interleaved runs.

Normal-scale visual inspection shows the same simplified/tight trousers,
shortened white top, pose, face and background. The largest amplified
residual is near the trouser crotch/seams. No new major visible failure was
apparent, but numerical improvement is not product-quality certification.
`comparison-gallery-mixed-bottoms\index.html` contains floating/mixed and
all-Q8/mixed full-resolution comparisons, contact sheets and differences.

Direct-Q8 cardigan, mixed cardigan and the matched BF16 bottoms recorder
control also completed. The gallery builder now renders finite F32 saved
image tensors using the same clamp/scale/truncate helper as the trajectory
comparator; this permits final-image comparison without fabricating a
reference trajectory for cardigan.

## 26. Actual original Python sampler and measurement corrections

Ran the pinned, unmodified `TryOnPipeline._sample`, preserving default CPU
SDPA dispatch and batched conditional/null computation. Conditions, category,
seed42,20steps, CFG1.5, shift1.5, skip1 and16threads match native inputs.
The harness bypasses the constructor and uses meta-device weight loading:
this is prepared-input inference, not raw `__call__` startup, and neither
parser nor pose detector is loaded. It is distinct from the F32-MATH,
separate-branch numerical oracle.

| Pair | Original Python wall time | Original sampling/PIL time | Earlier native CLI wall time | Differing RGB byte channels |
|---|---:|---:|---:|---:|
| Cardigan | 1258.0365s | 1241.3262s | 2549.6227s | 69 /1,327,104, each by1 |
| Trousers | 1325.4767s | 1263.7368s | 2491.2815s | 44 /1,327,104, each by1 |

Cardigan MAE0.0000519929/PSNR90.97136dB; trousers
MAE0.0000331549/PSNR92.92533dB. Normal-scale outputs look the same, including
their garment-fidelity limitations. Python was faster in these observations.
Historical native times, different startup paths and no repeated interleaved
trials prevent an intrinsic language-level or universal2x claim.

The first Python memory report was wrong: it monitored the Windows venv
launcher rather than the interpreter. Wall time still covered complete
execution, but approximately11.7MB launcher working set is not model-memory
evidence. Raw data is retained with a correction; the corrected comparison
excludes those memory fields.

Implemented `fashn_measurement_worker.py`: register the active interpreter
PID before model imports, atomically acknowledge measurement ownership, then
execute the original script/arguments with `runpy`. The parent validates the
PID relationship and samples the actual interpreter. A real32MiB allocation
regression distinguishes this from launcher-only measurement.

An intermediate acknowledgement-placement bug was caught and corrected.
Timeout testing exposed unreliable `taskkill` exit behavior under load;
owned-process cleanup now uses Windows process snapshots and retained handles
with creation-time checks, avoiding stale/reused-PID relationships. The worker
does not start model work before ownership acknowledgement, and timeout
cleanup was exercised with actual child processes. A too-short startup
allowance under CPU load was corrected in the timeout regression. A transient
temporary-file replacement access denial did not recur; no speculative broad
retry or silent success fallback was added.

The second Python run has valid interpreter attribution: peak working set
7,057,371,136 bytes (6.573GiB), versus3,175,022,592 (2.957GiB) for the
historical native BF16/F32 CLI. Python sampled private bytes were
10,310,897,664 versus3,334,754,304. This supports lower native memory in that
comparison, not the invalid first-cardigan measurement. Commands, process
IDs, source hashes and scope remain in the measurement artifacts.

## 27. Final independent image, matched recorder and evidence closure

The bottoms-selected four-matrix policy was fixed before the cardigan run.
No matrix ranking was tuned to cardigan. Its full-image results are:

| Cardigan result | All104 Q8 | Mixed100 Q8 +4 F32 |
|---|---:|---:|
| Wall time | 1956.6835s (32m37s) | 1975.8638s (32m56s) |
| Peak working set | 4.164GiB | 4.304GiB |
| Cropped mean byte error versus native floating | 0.25865494 | 0.19067760 |
| Cropped RMSE | 1.59386299 | 1.51533736 |
| Numerical PNG PSNR | 44.08178dB | 44.52062dB |
| Maximum byte error | 83 | 82 |
| Differing RGB channels | 252,503 | 170,921 |

Mean error improves26.2811%, but the worst boundary error largely persists.
The maximum is at cropped x184,y524, red channel, near the image-left
sleeve/torso gap. The amplified image also highlights hem/waist and neckline.
Pixels with any channel error above16 number336 for Q8 and328 for mixed;
above32 they number285 and278. The99th-percentile channel error is2 in both.
These concentrated residuals should not be hidden by an improved average.

Normal-scale and full-resolution inspection shows essentially the same
overall cardigan, graphic, pose, face and background. The original model's
graphic and garment-construction inaccuracies remain. No obvious new major
visual failure was apparent, but this is not perceptual/product certification.
Cardigan has final-image comparisons only, not a complete floating-reference
trajectory. The weight-only Q8/F32-compute experiment likewise remains a
two-local-state control, not a full-image run.

The matched BF16/F32 bottoms recorder completed in2640.8716s (44m01s), with
3,716,132,864 bytes peak working set (3.461GiB) and1,908,715,520 sampled peak
private bytes. Exact executable/checkpoint/condition hashes, settings and
16threads match the Q8/mixed recorder. Compared with this control, all-Q8
used26.2612% less wall time and mixed23.9875% less. However, their peaks were
higher:4.151GiB and4.277GiB. Smaller weight payload did not translate to lower
measured process memory in this diagnostic loading/recording path.

All floating-control reference states pass: maximum image relative L2
0.00000201699 and velocity L2 0.00000245062; initial-noise max error
4.76837e-7 and schedule error0. The cropped PNG exactly reproduces the
earlier native CLI pixels. That closes the recorder-versus-CLI baseline
without conflating its performance with the historical CLI timings.
Original Python remains faster than the native Q8/mixed observations.

The final gallery renders all11 rows with exact crops, no resizing for
metrics, full-resolution PNGs, contact sheets and difference images. The
saved-image renderer has regressions for clipping/truncation, finite F32,
RGB shape, explicit crops and preserved source provenance. The final evidence
snapshot confirms all five native runs, exact104-entry precision maps,
100 finite F32 image/velocity states, upstream input/settings lineage,
unchanged executable hashes, reproduced gallery pixels/metrics and the
floating control. Source states and rendered comparison PNGs are hash-linked.

All requested image experiments are complete; no model process remains.
The results establish working diagnostic quantization and a useful small
restoration policy, not public numerical acceptance, a globally minimal
policy, broader seed/image quality, lower-bit support or GPU correctness.
