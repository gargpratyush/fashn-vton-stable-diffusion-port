# FASHN VTON 1.5 development

**Experimental prepared-input FASHN generation is available in the native
library, CLI, and asynchronous server API.** The released transformer runs in F32 on one CPU backend,
without Python, text encoders, or a VAE at inference time. This is not yet a
general-purpose production pipeline. Raw-image preparation is available through
an optional native `sd-fashn-prepare` executable and opt-in raw HTTP/UI mode
(OpenCV/ONNX Runtime), or the pinned Python reference helper. The default
prepared-input build requires neither dependency.
Worn-garment preparation and person masking require the separately authorized
research/evaluation-only human parser. A standalone browser UI is embedded at
`/try-on`, independently of the frontend submodule.

## Native prepared-input workflow

Build `sd-cli` normally and use the original released `model.safetensors`.
BF16 checkpoint storage is expanded to F32 for computation. Quantized inference,
native BF16/F16 matrix arithmetic, GPU execution, disk parameter offload, custom
model variants, and generic generation extensions are not supported by this
initial path. Weights remain resident and graph execution is monolithic.
Manual attention remains the default. CPU flash attention with F32 keys and
values is available through `--diffusion-fa` (or the global `--fa`).
These flags do not enable lower-precision FASHN weights.

To reduce resident memory, add `--type bf16` or `--type f16`. Only the 104
eligible core attention/MLP matrices use that resident storage type; embeddings,
patch kernels, modulation, biases, norms and final layers remain F32. Matrices
are cast to F32 just before multiplication, avoiding low-precision activation
rounding. All graph matrix operands are checked to be F32 and matrix
multiplications explicitly request F32 precision. This is reduced
**storage**, not BF16/F16 computation. The default remains F32 residency.

Create a manifest alongside four prepared 8-bit PNG images:

```json
{
  "schema": "fashn-vton-prepared-v1",
  "ca_image": "person.png",
  "garment_image": "garment.png",
  "person_pose": "person-pose.png",
  "garment_pose": "garment-pose.png",
  "category": "tops",
  "crop": {"x": 0, "y": 0, "width": 576, "height": 864}
}
```

All inputs must be exactly 576 x 864. Person and garment images must decode as
three-channel RGB; poses must be single-channel grayscale, using the original
FASHN pose representation, **not an RGB skeleton converted to grayscale**.
The person condition is the reference pipeline's prepared `ca_image`, not an
arbitrary raw photograph. A flat-lay garment uses the reference blank pose.
Paths resolve relative to the manifest. Unknown fields, invalid categories,
wrong dimensions/channels, HDR and 16-bit inputs are rejected without automatic
resizing. Use PNG for lossless prepared pixels; WebP is not supported by this
metadata-validation path.

Category is `tops`, `bottoms`, or `one-pieces`. The crop is optional and defaults
to the entire canvas; when supplied it is a positive integer rectangle inside
the canvas. Cropping occurs after sampling and does not resize the output.

```powershell
.\build\bin\Release\sd-cli.exe --mode try_on --diffusion-model ..\fashn-vton-reference\weights\model.safetensors --try-on-inputs prepared\manifest.json --steps 30 --cfg-scale 1.5 --flow-shift 1.5 --skip-cfg-last-n-steps 1 --seed 42 --rng cpu -t 8 -o try-on.png
```

Mode defaults are 30 steps, CFG 1.5, shift 1.5, one skipped final CFG step,
576 x 864, and CPU RNG. `--steps 2` is a much cheaper integration diagnostic,
not a quality setting. Two steps at CFG 1.5 with one skipped CFG step require
three full transformer forwards. CPU inference is expensive.

### CPU flash attention and profiling

FASHN opts into F32 K/V storage in the shared attention helper. Other models
retain their existing F16 K/V flash behavior. Merely selecting F32 accumulation
was insufficient: the original helper cast K/V to F16 before attention.
That experiment failed three late conditional activation comparisons, with a
maximum relative L2 of 0.0011681 against the unchanged 0.001 gate.
It is not the FASHN path exposed here.

The F32-preserving path passed all 17 captured tensors per branch at time zero
and a nonzero schedule time, with both conditional and null inputs. Conditional/
null velocity relative L2 was 6.93e-7 / 5.03e-7 at time zero and
7.51e-6 / 2.13e-6 at the nonzero time. The graph diagnostic checks that all
28 attention operations are actual flash nodes. The production runner likewise
fails graph construction if requested flash operations fall back on an
unsupported backend, rather than silently claiming the optimization.
GPU execution remains rejected.

Append `--flash-attention` to `test-fashn-vton-graph` to reproduce these
comparisons. With optional checkpoint/fixture CMake settings, a separate
`fashn-vton-graph-flash` CTest exercises this path. The fast primitive diagnostic
also compares F32 flash with manual attention and checks that the existing
F16 K/V path retains its storage types.

Measured on the Windows AMD EPYC 7763 development VM, eight CPU threads,
576 x 864 photographic prepared inputs, seed 42, one step, CFG 1:

| Attention | Wall time | Peak process working set |
|---|---:|---:|
| Manual F32 | 113.03 s | 5.896 GiB |
| Flash, F32 K/V | 81.87 s | 4.217 GiB |

These are separate, non-overlapping, single-run measurements with a fresh
process, not confidence intervals or cold filesystem-cache measurements.
Wall time includes model loading and PNG writing: flash reduced it by 27.6%;
peak working set fell by 28.5%. The one-step PNG differs from manual attention
in 64 of 1,492,992 byte channels, each by at most one level. A separate
two-step, CFG-1.5 photographic run took 235.25 s and peaked at 4.222 GiB;
only 81 byte channels differ from the original PyTorch trajectory, each by
one level (mean absolute byte error 0.0000542535). This is numerical evidence,
not a general image-quality guarantee. Original SD 1.5 smoke-test pixels
remain unchanged.

The Windows benchmark helper uses the existing Python environment and the
Windows process-memory API; it installs no profiler. Each repeat runs a new
CLI process and creates a PNG, log, and JSON report with executable, checkpoint,
manifest, input and output hashes. It records the kernel peak working set
through the last successful sample, sampled peak private usage (not a kernel
high-water value), sample count, last sample time, wall time, command, and exit
status. An incomplete or failed run is not marked passed. Output PNG mode and
crop dimensions are checked before completion.

For the upstream recommended-fast 20-step setting (39 transformer forwards):

```powershell
& ..\fashn-vton-reference\.venv\Scripts\python.exe scripts\benchmark_fashn_vton.py --cli build\bin\Release\sd-cli.exe --checkpoint ..\fashn-vton-reference\weights\model.safetensors --prepared prepared\manifest.json --output ..\fashn-vton-reference\photo-20-step --steps 20 --cfg 1.5 --shift 1.5 --skip-cfg-last-n-steps 1 --seed 42 --threads 8 --flash-attention
```

Use a new output directory for every benchmark. Omit `--flash-attention` for
the manual baseline. This is a Windows process-memory harness, not a portable
profiler or a raw-image preparation command.

The same isolated one-step benchmark with on-demand F32 matrix casts measured:

| Resident core matrices | Compute / attention | Wall time | Peak working set |
|---|---|---:|---:|
| BF16 | F32 / F32 flash | 81.83 s | 2.950 GiB |
| F16 | F32 / F32 flash | 82.54 s | 2.950 GiB |

Peak memory is about 30% below F32 flash and 50% below the original manual
baseline; the one-run timings do not establish a speed difference. BF16
residency matched the F32 CLI pixels exactly. F16 residency differed in 29 byte
channels, each by one level. Both passed all captured-layer gates in both
branches at time zero and the nonzero probe. Add `--weight-type bf16` or
`--weight-type f16` to the benchmark helper to reproduce these modes.

The representative photographic 20-step run completed in **2,985.95 seconds
(49 minutes 46 seconds)** with **4.222 GiB** peak working set. It used the
upstream person and worn-garment photographs, parser-evaluation preparation,
tops, CFG 1.5, shift 1.5, one skipped final CFG step, seed 42, eight CPU threads,
and F32 K/V flash attention. Its persisted RGB output is 576 x 864.

Visual inspection shows a clear black-shirt transfer and substantially sharper
lettering, hands, trousers and boots than the two-step diagnostic. The seated
pose, hat and overall background composition are broadly retained. This does
not imply pixel-locked preservation: clothing fit, folds and nearby details
are regenerated. This is one pair/seed/category, not a quantitative identity,
typography, garment-fit, or production-quality assessment. The later
[baseline experiments](#quality-trajectory-and-reliability-baseline-experiments)
include an independent 20-step PyTorch-model trajectory comparison.
That oracle uses F32 math attention and separate CFG branches; it is not a
latency measurement of the unmodified upstream batched-CFG sampler.
No 30/50-step quality sweep has been performed.

The sampler uses the original ascending-time shifted schedule and
`x += dt * (vu + cfg * (vc - vu))`. The skipped final steps use `vc` directly;
CFG 1 omits null forwards entirely. Null image and pose tensors contain exact
normalized zero and category 0, distinct from black pixels normalized to -1.
The CLI rejects generic samplers/schedulers and incompatible generation inputs.
`--batch-count` supports 1..4 sequential samples, using slices of one logical
noise draw rather than reseeding with `seed + i`.

### C API

Existing public structures retain their layout. The additive
`sd_try_on_params_t` is initialized with `sd_try_on_params_init`, which sets its
`struct_size` and defaults. A minimal call, with caller-owned prepared images:

```c
sd_ctx_params_t context;
sd_ctx_params_init(&context);
context.diffusion_model_path = "model.safetensors";
context.rng_type = CPU_RNG;
context.n_threads = 8;
sd_ctx_t *ctx = new_sd_ctx(&context);
if (ctx == NULL || !sd_ctx_supports_try_on(ctx)) {
    free_sd_ctx(ctx);
    return 1;
}
sd_try_on_params_t request;
sd_try_on_params_init(&request);
request.ca_image = prepared_person;
request.garment_image = prepared_garment;
request.person_pose = prepared_person_pose;
request.garment_pose = prepared_garment_pose;
request.category = SD_TRY_ON_TOPS;
sd_image_t *images = NULL;
int count = 0;
bool ok = generate_try_on(ctx, &request, &images, &count);
if (ok) {
    /* Consume count RGB images before releasing them. */
    free_sd_images(images, count);
}
free_sd_ctx(ctx);
return ok ? 0 : 1;
```

Input buffers must remain valid throughout the synchronous call. Outputs are
interleaved RGB bytes using reference-compatible clipping and truncation.
Failures clear output pointers/count. Progress is reported per completed step
across the requested samples. `SD_CANCEL_ALL` discards the request, including
cancellation from the final progress callback; `SD_CANCEL_NEW_LATENTS` finishes
the current sample and returns completed samples. Cancellation is checked
between forwards, not inside a running attention graph. Do not concurrently
generate on one context. Generic image/video APIs reject a FASHN context.

For request-local progress/cancellation, use the additive
`generate_try_on_with_callbacks` entry point with:

```c
sd_try_on_callbacks_t callbacks = {0};
callbacks.struct_size = sizeof(callbacks);
callbacks.progress = my_progress_callback;
callbacks.cancelled = my_cancellation_callback; /* bool callback(void *data) */
callbacks.data = my_request_state;
bool ok = generate_try_on_with_callbacks(ctx, &request, &callbacks, &images, &count);
```

The callback object and data must live until return. Callbacks run on the
generation thread and must not re-enter generation. A non-null callback object
replaces the global progress callback for this request; the original entry
point retains its behavior. The cancellation predicate supplements context
cancellation and is not cleared by initialization of a new generation. This
closes the server's queued-to-generating cancellation race without installing
process-global per-job callbacks. Progress counts completed sampling steps
across all samples; cancellation can wait for one full model forward.

## Native raw-image preparation

Enable the optional component with local OpenCV 4.12 and ONNX Runtime 1.22.1
C/C++ distributions. The normal build defaults to `SD_FASHN_PREPROCESS=OFF`
and requires neither dependency, Python, nor model downloads.

```powershell
cmake -S . -B build -DSD_FASHN_PREPROCESS=ON -DOpenCV_DIR=..\fashn-native-deps\opencv\build -DONNXRUNTIME_ROOT=..\fashn-native-deps\onnxruntime-win-x64-1.22.1
cmake --build build --config Release --target sd-fashn-prepare
.\build\bin\Release\sd-fashn-prepare.exe --dwpose-dir ..\fashn-vton-reference\weights\dwpose --person-image person.png --garment-image flat-lay.png --category tops --garment-photo-type flat-lay --output prepared-native
.\build\bin\Release\sd-cli.exe --mode try_on --diffusion-model ..\fashn-vton-reference\weights\model.safetensors --try-on-inputs prepared-native\manifest.json --type bf16 --diffusion-fa --steps 20 --cfg-scale 1.5 --seed 42 --rng cpu -t 8 -o try-on.png
```

The two executables form a Python-free runtime workflow. The same native
preparation library also links into `sd-server` when this build option is on.
The core library and `sd-cli` do not acquire these dependencies. Raw HTTP
preparation requires an additional operator startup opt-in; it never loads
client-supplied file paths. The UI supports both prepared and enabled raw modes.

For worn-garment or masked-person modes, export the already authorized parser
once in the isolated evaluation environment:

```powershell
& ..\fashn-vton-reference\.venv-parser-eval\Scripts\python.exe -m pip install -r scripts\requirements-fashn-parser-onnx.txt
& ..\fashn-vton-reference\.venv-parser-eval\Scripts\python.exe scripts\export_fashn_parser_onnx.py --source ..\fashn-vton-reference\source --parser-dir ..\fashn-vton-reference\weights\human-parser-eval --output ..\fashn-vton-reference\weights\human-parser-onnx-eval --accept-parser-research-license
.\build\bin\Release\sd-fashn-prepare.exe --dwpose-dir ..\fashn-vton-reference\weights\dwpose --person-image person.png --garment-image worn-garment.png --category tops --garment-photo-type model --parser-dir ..\fashn-vton-reference\weights\human-parser-onnx-eval --accept-parser-research-license --output prepared-native-worn
```

Add `--no-segmentation-free` for person masking. Native parser-dependent modes
always require the explicit consent flag. The export contains `parser.onnx`,
its original LICENSE and a versioned manifest; export verifies local original
weights and ONNX logits before writing the manifest. Native loading checks
artifact/license hashes and model shapes. It does not download anything.
Parser-free operation does not initialize or require the parser.

Native preparation follows Pillow-compatible antialiased Lanczos3 pre-resizing,
the pinned YOLOX/RTMPose decoding and single-person selection, grayscale
body/hand/face rasterization, category masks and resize/padding. The parser uses
reference-compatible F32 bilinear coordinates rather than OpenCV's differing
interpolation arithmetic, which otherwise flipped a boundary label.
RGB decoding ignores EXIF orientation, matching the reference helper's PIL
image path; orient images explicitly before preparation. Decoded images above
20 megapixels and aspect ratios yielding zero resized dimensions are rejected.

The output directory must be new. `manifest.json` is written last; without it,
the directory is incomplete. The directory also contains diagnostic pre-resize
PNGs, poses, used segmentation maps, and provenance with input/model/artifact
hashes and runtime versions. Only the manifest and its four named PNGs are
generation inputs.

All 12 category/photo-type/person-masking combinations matched reference crop
metadata and four prepared images exactly on the photographic fixtures.
Separate odd/small/downsampled resize and masking primitive cases also matched
exactly. Flat-lay-mode checks using the worn source photo are declared-mode
regressions, not flat-lay quality evidence. Native preparation followed by
native inference matched the reference-prepared CLI output exactly.

```powershell
& ..\fashn-vton-reference\.venv-parser-eval\Scripts\python.exe scripts\test_fashn_native_preprocess.py --native build\bin\Release\sd-fashn-prepare.exe --primitives build\bin\Release\test-fashn-vton-preprocess.exe --source ..\fashn-vton-reference\source --dwpose-dir ..\fashn-vton-reference\weights\dwpose --parser-dir ..\fashn-vton-reference\weights\human-parser-eval --parser-native-dir ..\fashn-vton-reference\weights\human-parser-onnx-eval --output ..\fashn-vton-reference\native-preprocess-regression --accept-parser-research-license -v
```

Windows builds copy the runtime DLLs beside the executable. The standalone
install component includes the executable, DLLs and dependency licenses:

```powershell
cmake --install build --config Release --component fashn-preprocess --prefix ..\fashn-native-install
```

Model weights are never included in this install. Retain dependency licenses
and notices when redistributing; set `FASHN_OPENCV_LICENSE` if it was not found
in your SDK. Windows requires the corresponding MSVC runtime. The native
component has been exercised on Windows CPU; other-platform builds remain
unverified. Non-Windows SHA256 uses the optional component's OpenSSL dependency.
The Windows install has also performed actual preparation with only its own
`bin` and Windows System32 on PATH; its four PNGs and crop matched the reference.

### Opt-in native raw HTTP and browser uploads

Build `sd-server` with `SD_FASHN_PREPROCESS=ON`, then enable local DWPose sessions:

```powershell
cmake --build build --config Release --target sd-server
.\build\bin\Release\sd-server.exe --diffusion-model model.safetensors --rng cpu --type bf16 --diffusion-fa -t 8 --listen-ip 127.0.0.1 --try-on-dwpose-dir weights\dwpose
```

This enables segmentation-free person / flat-lay garment preparation only.
To enable worn-garment and masked-person modes, additionally supply
`--try-on-parser-dir weights\human-parser-onnx-eval --accept-parser-research-license`.
These are operator options, never client request fields. Invalid hashes or
missing consent fail startup. No Python, child processes, downloads, or
per-request temporary files are used. DWPose sessions load at startup. The
parser configuration, artifact hashes and license are validated at startup,
but its ONNX session loads only on the first parser-dependent request.
The parser artifact and license hashes are rechecked before that delayed
load. ONNX session-construction errors are consequently reported on first
use, not startup. Capabilities indicate that the parser is configured, not
that its session is already resident. Loaded sessions are reused by the
single asynchronous worker; flat-lay segmentation-free requests do not
instantiate an unused parser.

For lower retained preparation memory, add `--try-on-ort-no-arena` to the
server (requires native DWPose configuration), or `--ort-no-arena` to
`sd-fashn-prepare`. This disables ORT CPU arenas for all preparation sessions;
threading and math remain unchanged. It is opt-in: repeated preparation can
cost more allocations. A measured warmed preparer retained about 661 MiB
instead of 1596 MiB, with identical prepared pixels. This is preparation
memory, not a promise about the entire diffusion process.

The asynchronous try-on queue additionally reserves from a **512 MiB image
storage budget**, including queued decoded/base64 inputs, the raw-to-prepared
transition, and capacity for requested results. Requests exceeding either
this budget or the existing pending-count limit receive HTTP 429.
Cancellation/failure releases input reservations; completion retains actual
result string capacities until the existing TTL expires. Capabilities expose
`limits.max_try_on_image_bytes` and `limits.reserved_try_on_image_bytes`.
These are conservative image-storage reservations, not current RSS:
model/ORT/graph workspaces, transient HTTP copies and job metadata are outside
this budget. Existing result TTLs and the 64-pending-job limit are unchanged.

Send exactly one of prepared `inputs` or this alternative `raw_inputs` object:

```json
{
  "raw_inputs": {
    "person_image": "<base64 8-bit RGB PNG>",
    "garment_image": "<base64 8-bit RGB PNG>",
    "category": "tops",
    "garment_photo_type": "flat-lay",
    "segmentation_free": true
  },
  "steps": 20,
  "cfg": 1.5,
  "shift": 1.5,
  "skip_cfg_last_n_steps": 1,
  "seed": 42,
  "sample_count": 1
}
```

All five raw fields are required. `garment_photo_type` is `flat-lay` or `model`.
Raw HTTP inputs are intentionally more bounded than standalone preparation:
3 MiB per encoded string (including any PNG data-URI prefix), dimensions at
most 4096 each, and at most 4,194,304 pixels per image. They must decode as
8-bit RGB PNGs; RGBA/grayscale/16-bit, URLs, and filesystem paths are rejected.
Crop is computed by the preparer, not supplied by the client. The existing
12 MiB + 4 KiB request limit and 64-pending-job limit are unchanged.
Validation decodes one bounded image at a time; queued raw requests retain only
encoded strings (at most 6 MiB each), then decode again on the worker.
Raw strings and prepared buffers are released after use/cancellation.

Capabilities expose `features.raw_image_preprocessing` and
`features.raw_parser_modes`, also under `features_by_mode.try_on`. They are
false unless the operator configured the corresponding sessions. `/try-on`
uses these flags to enable the two-photo form and restricted choices.
Preparation cancellation is checked between stages; it does not interrupt an
ONNX Runtime call or individual pose candidates within a detection stage.

The shared pipeline retained exact four-image/crop parity across all 12
preparation combinations. Raw HTTP one-step inference matched prepared CLI
pixels exactly; raw preparation cancellation and 14 additional invalid-input
cases passed. These are infrastructure/numerical checks, not expanded quality
coverage.

### Real-browser regression

Browser tooling is optional and separate from both inference environments:

```powershell
python -m venv ..\fashn-vton-reference\.venv-browser
& ..\fashn-vton-reference\.venv-browser\Scripts\python.exe -m pip install -r scripts\requirements-fashn-browser.txt
$env:PLAYWRIGHT_BROWSERS_PATH = "$PWD\..\fashn-browser-deps"
& ..\fashn-vton-reference\.venv-browser\Scripts\python.exe -m playwright install chromium
& ..\fashn-vton-reference\.venv-browser\Scripts\python.exe scripts\test_fashn_vton_browser.py --server build\bin\Release\sd-server.exe --checkpoint model.safetensors --dwpose-dir weights\dwpose --parser-dir weights\human-parser-onnx-eval --accept-parser-research-license --person person.png --garment worn-garment.png --prepared prepared\manifest.json --expected native-one-step.png --output browser-results
```

Use a matching one-step, CFG-1, seed-42, BF16-resident/F32-flash reference PNG.
The test starts/stops its own loopback server, blocks non-loopback page requests,
uploads photos in Chromium, downloads and compares output pixels, cancels a
second job, and checks prepared-mode validation. It saves a screenshot and
JSON report. The exercised run completed generation in 86.36 seconds, matched
the prepared CLI pixels exactly, and had no page errors or external requests.

## Optional raw-image preparation

The parser-free preparation mode is **segmentation-free person + flat-lay
garment**. In this upstream mode both masking functions return their input
images unchanged, so the original pipeline's two parser predictions have no
effect and can be omitted. Worn-garment photos and person masking require the
separately licensed human parser described below. Specify
`--garment-photo-type flat-lay` explicitly for this path; do not pass worn-garment
photos as flat lays.

Use the existing isolated reference environment and clean pinned source checkout:

```powershell
& ..\fashn-vton-reference\.venv\Scripts\python.exe -m pip install --extra-index-url https://download.pytorch.org/whl/cpu -r scripts\requirements-fashn-preprocess.txt
& ..\fashn-vton-reference\.venv\Scripts\python.exe scripts\prepare_fashn_vton.py download-pose --dwpose-dir ..\fashn-vton-reference\weights\dwpose
& ..\fashn-vton-reference\.venv\Scripts\python.exe scripts\prepare_fashn_vton.py prepare --source ..\fashn-vton-reference\source --dwpose-dir ..\fashn-vton-reference\weights\dwpose --person-image person.jpg --garment-image flat-lay.jpg --garment-photo-type flat-lay --category tops --output prepared
```

Only explicit download commands download weights. Preparation
authenticates both ONNX files before creating CPU sessions, never downloads a
parser, and refuses existing output directories. DWPose revision
`548b5df25b84d9f4aac0611dfa1c2a7a12f15571` is pinned with SHA256:

| File | SHA256 |
|---|---|
| `yolox_l.onnx` | `7860ae79de6c89a3c1eb72ae9a2756c0ccfbe04b7791bb5880afabd97855a411` |
| `dw-ll_ucoco_384.onnx` | `724f4ff2439ed61afb86fb8a1951ec39c6220682803b4a8bd4f598cd913b1843` |

The helper preserves the upstream PIL pre-resize (long side at most 864, no
upsampling), RGB-to-BGR detector input, grayscale pose renderer, OpenCV RGB
resize/padding, and `INTER_NEAREST_EXACT` pose resize. It records the person's
padding separately from the garment's padding and rejects insufficient visible
body keypoints rather than supplying a blank person pose.

Outputs include four PNGs, the CLI-compatible `manifest.json`, normalized
`conditions.safetensors` for the oracle, raw person keypoints, and provenance
with input/model hashes and package versions. The crop restores the unpadded
model-canvas dimensions, **not the original photograph's resolution**.
Python/OpenCV/ONNX Runtime are preparation dependencies only, not native
generation dependencies.

```powershell
& ..\fashn-vton-reference\.venv\Scripts\python.exe scripts\test_fashn_vton_preprocess.py --source ..\fashn-vton-reference\source --dwpose-dir ..\fashn-vton-reference\weights\dwpose --person-image ..\fashn-vton-reference\source\examples\data\model.webp --output ..\fashn-vton-reference\preprocess-results -v
```

This compares the unchanged upstream pipeline call and masking functions against
the helper, substituting only unused parser predictions and diffusion output.
All four normalized tensors and crop geometry matched exactly across five
aspect/size cases and actual CPU DWPose inference. The real-person integration
fixture uses a **procedural flat-lay garment**, not a real garment photograph;
its native smoke test is not production try-on quality validation.

### Optional parser-dependent preparation: research/evaluation only

**The official human parser inherits the NVIDIA SegFormer license limiting use
to non-commercial research or evaluation.** Review
[the upstream license](https://github.com/NVlabs/SegFormer/blob/master/LICENSE)
before installing or using it. These restrictions are separate from the
Apache-2.0 VTON model. This adapter does not grant permission for commercial
use and is never loaded by the native library, CLI, or server.

Use a separate environment: the parser package requires `opencv-python`, while
the parser-free environment uses `opencv-python-headless`. Installing both
distributions in the same environment is unsupported. The helper detects and
rejects that conflict.

```powershell
python -m venv ..\fashn-vton-reference\.venv-parser-eval
& ..\fashn-vton-reference\.venv-parser-eval\Scripts\python.exe -m pip install --extra-index-url https://download.pytorch.org/whl/cpu -r scripts\requirements-fashn-parser-eval.txt
& ..\fashn-vton-reference\.venv-parser-eval\Scripts\python.exe scripts\prepare_fashn_vton.py download-parser --parser-dir ..\fashn-vton-reference\weights\human-parser-eval --accept-parser-research-license
& ..\fashn-vton-reference\.venv-parser-eval\Scripts\python.exe scripts\prepare_fashn_vton.py prepare --source ..\fashn-vton-reference\source --dwpose-dir ..\fashn-vton-reference\weights\dwpose --parser-dir ..\fashn-vton-reference\weights\human-parser-eval --accept-parser-research-license --person-image person.jpg --garment-image worn-garment.jpg --garment-photo-type model --category tops --output prepared-eval
```

The license flag is required separately for downloading and for preparation
that needs the parser. Omitting it fails before model initialization/download.
Use `--no-segmentation-free` to enable person masking with either garment photo
type. The default still preserves the original person image. Flat-lay plus
segmentation-free mode does not import the human parser even when the optional
package is installed.

The adapter pins `fashn-human-parser==0.1.1`, Transformers 4.55.4, and model
revision `1f80c34dbab321c5730dda5c3fea279fd3e97498`. The model SHA256 is
`e43c8c8a9b04f28798f0a4630cf18caa2cdb27a0d454fae43a5716e6f7078244`.
Both configuration files are also hash-pinned. Downloads preserve the installed
package's license alongside the weights. Initialization requires authenticated
local safetensors, `local_files_only=True`, CPU, and F32.

The package's original parser preprocessing/prediction is preserved: OpenCV
INTER_AREA resize to 384 x 576, ImageNet normalization, SegFormer logits,
bilinear upsampling, and argmax. No generic Hugging Face image processor is
substituted. Original FASHN category mappings and masking functions are imported
from the clean pinned source. Worn garments get their own detected/rendered
pose and category mask, using gray value 127 outside the garment. Person
masking retains the original limb/identity preservation rules.

Parser predictions whose result is unused are skipped. Invalid label maps,
missing garment-category pixels, and insufficient worn-garment keypoints fail
explicitly rather than producing a blank success-shaped condition. Exports add
segmentation label PNGs for masks actually used, worn-garment keypoints, parser
package/source hashes, and the research/evaluation restriction in provenance.
The four generation inputs retain the existing manifest/API schema.

The evaluation suite blocks socket connections, compares parser logits against
the original class initialized from the same local weights, and compares
prepared tensors/crops against the unchanged original pipeline call:

```powershell
& ..\fashn-vton-reference\.venv-parser-eval\Scripts\python.exe scripts\test_fashn_vton_parser_eval.py --source ..\fashn-vton-reference\source --dwpose-dir ..\fashn-vton-reference\weights\dwpose --parser-dir ..\fashn-vton-reference\weights\human-parser-eval --accept-parser-research-license --output ..\fashn-vton-reference\parser-eval-results -v
```

Parser logits matched exactly on the two upstream example photographs. All four
prepared tensors and crop geometry matched exactly in 12 synthetic
category/photo/masking combinations, plus both segmentation-free and masked
person modes using those photographs. The suite exports the photographic
segmentation-free/worn-garment fixture for native generation.

The original-model trajectory exporter also accepts the resulting
`--inputs prepared-eval\conditions.safetensors --category tops`. Omitting
`--inputs` retains its synthetic default. Its `expected.png` is the full canvas;
apply the manifest crop when comparing a cropped native result. A two-step run
is only a numerical integration diagnostic, not the recommended 20-50-step
quality evaluation.

Using the upstream seated-person and black-shirt photographs, the two-step
native output differed from the original PyTorch trajectory in just 47 of
1,492,992 byte channels, by at most one level (mean absolute byte error
0.00003148). This run used segmentation-free preparation, tops category,
CFG 1.5, shift 1.5, skip-last 1, seed 42, CPU/F32. The image transfers the black
shirt but remains soft and loses fine lettering at two steps; recommended-step
quality assessment remains separate.

For reproduction after exporting `prepared-eval`:

```powershell
& ..\fashn-vton-reference\.venv\Scripts\python.exe scripts\export_fashn_vton_reference.py trajectory --source ..\fashn-vton-reference\source --checkpoint ..\fashn-vton-reference\weights\model.safetensors --inputs prepared-eval\conditions.safetensors --category tops --steps 2 --cfg 1.5 --seed 42 --threads 8 --output oracle-real-two-step
.\build\bin\Release\sd-cli.exe --mode try_on --diffusion-model ..\fashn-vton-reference\weights\model.safetensors --try-on-inputs prepared-eval\manifest.json --steps 2 --cfg-scale 1.5 --seed 42 --rng cpu -t 8 -o native-real-two-step.png
```

## Asynchronous server

```powershell
.\build\bin\Release\sd-server.exe --diffusion-model ..\fashn-vton-reference\weights\model.safetensors --rng cpu -t 8 --listen-ip 127.0.0.1 --listen-port 1234
```

Use `POST /sdcpp/v1/try_on`, then the existing job polling/cancellation routes.
The request embeds the prepared manifest under `inputs`, replacing its four
filenames with padded base64 PNG strings or `data:image/png;base64,...` URIs.
There is no HTTP file-path/URL loader and no server-side Python preprocessing.
See [the server API contract](../examples/server/api.md#fashn-prepared-input-try-on).

Sampling defaults are the dedicated FASHN defaults, independent of generic
image/video defaults. The server requires `--rng cpu` and rejects generic
generation CLI options for a FASHN context; send steps/CFG/seed in request JSON.
Capabilities advertise only `try_on`, fixed canvas limits, supported categories,
PNG output, progress and active cancellation. Queued cancellation returns 200.
Generating try-on cancellation returns 202 with `cancellation_requested: true`;
poll until `cancelled`. It is cooperative at preparation-stage/sampling-forward boundaries, not an immediate
kernel interruption. Accepted cancellation wins a race with output encoding/
completion, and cancelled jobs never publish partial results. Inputs are
released when queued jobs are cancelled or generation
finishes. The existing queue limit, terminal job TTL, and context mutex apply.
Image/video active-cancellation behavior remains unchanged. Poll responses for
try-on include `progress: {completed_steps, total_steps, unit: "sampling_steps", phase}`.
Phases are `queued`, `preparing`, `sampling`, `encoding`, and `done`; preparation
does not increment sampling-step progress.

Open `/try-on` for the standalone embedded interface. Select `manifest.json`
and its four named PNGs together, then submit sampling settings. It displays
queue/progress, supports cancellation and PNG download, preserves uint64 seeds,
and retains an accepted job for retry/cancellation after polling errors. It
loads no external scripts or services. The frontend submodule and compatibility
API schemas are unchanged; no pnpm build is required for this page. When native
raw preparation is configured, choose raw mode and select two RGB PNGs instead.

Keep this example server on a trusted loopback interface; this change does not
add authentication or change its existing CORS behavior.

The loopback regression suite starts and stops its own server, disables HTTP
proxies, checks malformed/oversized inputs, runtime failure, and the async
lifecycle. With `--inference`, it fills the queue, cancels 63 queued requests,
and compares the completed output to a matching one-step CFG-1/seed-42 CLI PNG:

```powershell
& ..\fashn-vton-reference\.venv\Scripts\python.exe scripts\test_fashn_vton_server.py --server build\bin\Release\sd-server.exe --checkpoint ..\fashn-vton-reference\weights\model.safetensors --prepared prepared\manifest.json --expected native-one-step.png --output ..\fashn-vton-reference\server-results --inference -v
```

The tested server output matched the CLI pixels exactly. The test server is
terminated at the end, not left running.
To exercise F32 flash attention, add `--flash-attention` to the regression
command and supply an `--expected` PNG from a matching flash-enabled CLI run.
The server itself accepts `--diffusion-fa` at startup; attention is a context
setting, not a per-request sampling option.
The suite covers 31 invalid request cases, cancellation of 63 queued jobs,
exact CLI/output equality, per-step progress, active cancellation after a
completed step, and an unaffected following job. Active cancellation took
about one forward (78 seconds) in the exercised case. This is expected CPU
latency, not a promise of immediate interruption. UI logic has separate Node
tests (`node --test scripts\test_fashn_vton_ui.cjs`); the HTTP suite also checks
the embedded page, script and response headers.
To exercise raw HTTP cases, additionally pass `--dwpose-dir`, `--parser-dir`,
`--accept-parser-research-license`, `--raw-person`, and `--raw-garment`. Raw
positive tests use the worn-garment, segmentation-free, tops mode and compare
against the same matching prepared one-step output.

The final combined server suite passed all 10 tests with none skipped:
31 prepared/sampling invalid cases, 14 raw invalid cases, 63 queued
cancellations, active sampling and preparation cancellation, exact prepared
and raw output pixels, startup/license failures, and unaffected subsequent
jobs. Separate configurations confirmed parser-free mode restrictions and
that the native-dependency-free server builds and runs with only Windows
System32 on PATH. Both SDK-enabled and SDK-disabled CLI/server builds remain
supported; raw preprocessing is unavailable in the latter.

### Backend diagnostic readiness

The graph harness can list compiled/available devices and select an exact name:

```powershell
.\build\bin\Release\test-fashn-vton-graph.exe --list-backends
.\build\bin\Release\test-fashn-vton-graph.exe --backend CPU
.\build\bin\Release\test-fashn-vton-graph.exe model.safetensors native-fixtures results --backend CPU --flash-attention --matrix-type bf16 --upcast-matrices
```

The second command runs primitives without loading model weights. The full
graph records the actual backend/device, supported-node count, F32-precision
matrix count, and captured-layer comparisons. Unknown devices fail rather
than select a fallback. Graph construction rejects unsupported operations and
missing requested flash nodes. Full-precision execution also sets
`GGML_PREC_F32`, not merely F32 operand types; this is important for future
backends whose default arithmetic may be lower precision.

CPU validation passed both oracle timesteps with 3,070 supported graph nodes,
146 explicit-F32 matrix multiplies and 28 flash nodes. Every captured tensor
passed the unchanged gates. These diagnostics do not enable GPU production
inference: no compute GPU is present, and actual GPU primitive, layer and
trajectory parity remains mandatory before removing the public CPU guard.

## Pinned transformer oracle

Use Python 3.12 and an isolated virtual environment. From the repository root:

```powershell
python -m venv ..\fashn-vton-reference\.venv
& ..\fashn-vton-reference\.venv\Scripts\python.exe -m pip install --extra-index-url https://download.pytorch.org/whl/cpu -r scripts\requirements-fashn-reference.txt
git clone https://github.com/fashn-AI/fashn-vton-1.5 ..\fashn-vton-reference\source
git -C ..\fashn-vton-reference\source checkout 7c0f10af3f91ad4048fe9729c470a13ef905d25a
New-Item -ItemType Directory -Force ..\fashn-vton-reference\weights
curl.exe --fail --location --output ..\fashn-vton-reference\weights\model.safetensors https://huggingface.co/fashn-ai/fashn-vton-1.5/resolve/7720683168567eb5a2a4c67f15116c6e29c83ded/model.safetensors
```

The official checkpoint is 1,943,668,048 bytes. The exporter requires SHA256
`d6cd38286885bc29fa487ea9383f80ffeb95862e7747c630d42c5d3c05bdd35a`, a clean
pinned source checkout, and strict state-dictionary loading. It expands the
original BF16 values into CPU F32 and explicitly selects PyTorch SDPA's math
backend. It does not download anything automatically.

Only the original transformer and utilities are imported. The package's eager
pipeline initializer is bypassed without rewriting the model's math. No parser,
pose model, CUDA package, ONNX Runtime, or preprocessing weight is required.
The native CMake build acquires no new Python dependency.

Run focused tests, including the actual source architecture and local checkpoint
metadata:

```powershell
& ..\fashn-vton-reference\.venv\Scripts\python.exe scripts\test_fashn_vton_reference.py --source ..\fashn-vton-reference\source --checkpoint ..\fashn-vton-reference\weights\model.safetensors -v
```

Omitting the source or checkpoint explicitly skips their associated tests.
Small-model tests exercise the original blocks, exact null conditioning,
sequential versus batched CFG, hook cleanup, and deterministic replay.
Those diagnostic dimensions are **not** supported production resolutions.

Export full-canvas, real-weight conditional and unconditional forwards:

```powershell
& ..\fashn-vton-reference\.venv\Scripts\python.exe scripts\export_fashn_vton_reference.py export --source ..\fashn-vton-reference\source --checkpoint ..\fashn-vton-reference\weights\model.safetensors --output ..\fashn-vton-reference\oracle-step0 --threads 8
```

This is deliberately expensive on CPU: two forwards at 576 x 864 with 6,912
joint tokens. It is not a full 30-step generation. By default the conditioning
contains deterministic synthetic patterns, not a real person/garment, and the
manifest labels this explicitly. The initial noise is one CPU-generator draw.
`--steps`, `--step-index`, `--shift`, `--cfg`, `--skip-cfg-last-n-steps`,
`--category`, and `--seed` select the exported single-step case.
Changing `--step-index` changes the evaluation time; it does not advance through
the preceding trajectory. The noisy input remains the recorded Gaussian probe.

For real prepared inputs, supply `--inputs prepared.safetensors`, containing
exactly these F32, normalized [-1,1] tensors:

| Key | PyTorch shape |
|---|---|
| `ca_images` | `[1,3,864,576]` |
| `garment_images` | `[1,3,864,576]` |
| `person_poses` | `[1,1,864,576]` |
| `garment_poses` | `[1,1,864,576]` |

A blank flat-lay garment pose is -1 after normalization. The unconditional
branch uses the original model's dropout mask to produce exact zero conditions
and null category 0; it does not convert a black image into a null image.

## Artifact contract

An export creates a new directory and refuses to overwrite an existing one:

| File | Contents |
|---|---|
| `layout.safetensors` | Coordinate-coded patch round trip, positions, byte normalization |
| `inputs.safetensors` | Conditions, shared noise, category, time features, full schedule |
| `conditional.safetensors` | Selected layer inputs/outputs and conditional velocity |
| `unconditional.safetensors` | The same capture points with conditioning dropped |
| `euler.safetensors` | Guided velocity and one ascending-time Euler update |
| `manifest.json` | Revisions, source/checkpoint/artifact hashes, resolved packages, precision, timing |

The manifest is written last. A directory without it is incomplete, not a valid
fixture. Use a new directory after an interrupted export. Keep generated
fixtures and model weights outside the repository.

Captures include both patch projections, time/category embeddings, both RoPE
calls, the first/last patch mixer, double and single blocks, and the final layer.
Double-block tuple outputs `.output.0` and `.output.1` mean target and garment.
All payloads are contiguous PyTorch tensors: reverse the dimension metadata
for GGML, **do not transpose the payload a second time**.

Compare a native-exported tensor set against the matching reference tensor set:

```powershell
& ..\fashn-vton-reference\.venv\Scripts\python.exe scripts\export_fashn_vton_reference.py compare reference.safetensors native.safetensors
```

The command reports maximum absolute error, RMSE, and relative L2 per float
tensor and exits nonzero on failure. Keys, shapes, and dtypes must match;
integer tensors must match exactly. Defaults require both `atol=1e-4,
rtol=1e-4` and relative L2 <= 1e-3. These are initial engineering gates, not
universal backend guarantees. Diagnose the earliest divergence before changing
tolerances.

The exported schedule preserves the original implementation's F32 operations.
For 30 steps and shift 1.5, its step-29 value is `0.8661447763442993`;
the analytically equivalent F64 formula rounded to F32 is
`0.8661450743675232`. The analytic-formula unit test allows `4e-7` for this
measured reciprocal/linspace rounding difference. Native sampling should be
compared against the exported original schedule, not silently replace it with
the analytical formula.

The `contract` subcommand validates all 365 inference tensor names and shapes,
the optional known unused buffer, floating dtypes, and payload intervals without
loading tensor values. It accepts raw or canonically prefixed names, rejects
prefix collisions, and rejects incomplete/extra model variants.

## Native numerical diagnostics

The native runner captures 17 tensors per branch. Its opt-in F32 tanh-GELU
avoids the GGML CPU FP16 lookup used by default, preserving existing behavior
for other models. Full-canvas official-weight comparisons at time zero and a
nonzero schedule time pass the captured-activation relative-L2 gate of 1e-3.
Time-zero conditional/null velocity relative L2 was approximately
5.79e-7 / 3.58e-7 on the development CPU.

Adapt rank-six reference RoPE metadata for the native safetensors reader, then
run the native graph and sampler diagnostics:

```powershell
& ..\fashn-vton-reference\.venv\Scripts\python.exe scripts\export_fashn_vton_reference.py native-fixtures ..\fashn-vton-reference\oracle-step0 ..\fashn-vton-reference\native-fixtures-step0
.\build\bin\Release\test-fashn-vton-graph.exe ..\fashn-vton-reference\weights\model.safetensors ..\fashn-vton-reference\native-fixtures-step0 ..\fashn-vton-reference\native-results
.\build\bin\Release\test-fashn-vton-sampling.exe ..\fashn-vton-reference\native-fixtures-step0
```

The adapter removes leading singleton axes, never transposes payloads, and
losslessly transports small integers as F32. Graph reports include maximum
absolute error, relative L2, RMSE, and pointwise outlier counts. Primitive
captures additionally require atol/rtol 1e-4. The sampler diagnostic compares
the entire original schedule, CPU noise, and real-weight Euler update. On the
development CPU, the 30-step schedule and Euler tensors matched exactly;
MT19937 normal samples differed by at most 4.77e-7.

For an end-to-end synthetic regression, export lossless PNG conditions and an
actual multistep original-model trajectory:

```powershell
& ..\fashn-vton-reference\.venv\Scripts\python.exe scripts\export_fashn_vton_reference.py prepared-fixtures ..\fashn-vton-reference\oracle-step0 ..\fashn-vton-reference\prepared-step0
& ..\fashn-vton-reference\.venv\Scripts\python.exe scripts\export_fashn_vton_reference.py trajectory --source ..\fashn-vton-reference\source --checkpoint ..\fashn-vton-reference\weights\model.safetensors --output ..\fashn-vton-reference\trajectory --steps 2 --threads 8
.\build\bin\Release\sd-cli.exe --mode try_on --diffusion-model ..\fashn-vton-reference\weights\model.safetensors --try-on-inputs ..\fashn-vton-reference\prepared-step0\manifest.json --steps 2 --seed 42 --rng cpu -t 8 -o native-two-step.png
```

The trajectory command deliberately uses synthetic tops conditions, saves the
initial noise, each updated float image, `expected.png`, and a provenance
manifest. PNG export refuses normalized values that cannot be represented
losslessly as uint8. These fixtures demonstrate numerical integration, **not
real-person try-on quality**.

The full 576 x 864 two-step CLI result differed from the original trajectory in
88 of 1,492,992 byte channels, by one byte level at most (mean absolute byte
error 0.00005894). A separate one-step 101 x 153 crop differed in six channels,
also by at most one level. These results include CPU RNG, input normalization,
sampling, output truncation, and crop integration; they are not byte-exact
cross-runtime guarantees.

The existing Python environment can also exercise strict manifest/CLI failures
and reproduce the one-step cropped inference:

```powershell
& ..\fashn-vton-reference\.venv\Scripts\python.exe scripts\test_fashn_vton_cli.py --cli build\bin\Release\sd-cli.exe --checkpoint ..\fashn-vton-reference\weights\model.safetensors --prepared ..\fashn-vton-reference\prepared-step0\manifest.json --oracle ..\fashn-vton-reference\oracle-step0 --output ..\fashn-vton-reference\cli-regression --trajectory ..\fashn-vton-reference\trajectory --native-trajectory native-two-step.png --inference -v
```

Use a new output directory for every run. Omit `--inference` to skip the
expensive forward, and omit both trajectory arguments to skip the saved
two-step comparison. Reports explicitly record skipped tests.

## Conversion and precision gates

The standard converter now has an explicit FASHN policy. F32/F16/BF16 exports
convert every tensor, including both patch embeddings and protected layers;
generic model conversion behavior is unchanged. Selective diagnostic
Q8_0/Q4_0/Q5_0/Q4_K/Q5_K exports quantize
only the 104 eligible attention/MLP matrices and keep all other parameters F32.
Modulation matrices, embeddings, patch kernels, biases, normalization scales
and the final layers are never accidentally quantized by that policy.
Other quantized formats are rejected. These exports do not enable public
quantized inference. K-block formats additionally require rows divisible by 256.
See [the Q4/Q5 study plan](fashn_q4_q5_plan.md) for full-image, memory and
repeatability experiments. Exact encoded-block verification uses the same
uniform importance vector as conversion; this is not activation calibration.

After the complete study, build a single-file offline comparison report:

```powershell
python scripts\build_fashn_comparison_report.py --repo . --experiment ..\fashn-vton-reference --output ..\fashn-vton-reference\q45-study\fashn-all-comparisons.html
```

The report embeds generated images, switchable Q8/BF16/original-Python pixel
comparisons, current and historical timing/memory tables, repeatability,
trajectory gates and evidence hashes. Amplified error maps are hidden and
explicitly labeled as diagnostics. Historical checkpoint-27/38 artifacts
and the completed Q4/Q5 study must be present; missing evidence is an error.
No inference, model download or external web service is used when building
or viewing the report.

```powershell
.\build\bin\Release\sd-cli.exe --mode convert --diffusion-model model.safetensors --type bf16 -t 8 -o model-bf16.gguf
.\build\bin\Release\sd-cli.exe --mode convert --diffusion-model model.safetensors --type q8_0 -t 8 -o model-q8_0.gguf
.\build\bin\Release\test-fashn-vton-conversion.exe model.safetensors model-bf16.gguf bf16
.\build\bin\Release\test-fashn-vton-conversion.exe model.safetensors model-q8_0.gguf q8_0
```

All 366 source tensors round-tripped with their names, shapes and intended
storage types. F32/BF16 round trips preserve the released values exactly;
F16 matches its explicit cast (worst weight relative L2 6.73e-7). The Q8 test
checks nearest-quantization error plus the actual FP16 scale-storage rounding
bound, including subnormal scales. Its worst weight relative L2 was 0.00846.
Those are conversion checks, **not Q8 inference or quality approval**.
`SD_FASHN_TEST_CONVERSION_DIR` can enable the four artifact CTests.

| GGUF type | File bytes | Public execution |
|---|---:|---|
| F32 | 3,887,290,272 | F32 compute |
| BF16 | 1,943,661,792 | F32 compute; optional BF16 matrix residency |
| F16 | 1,943,661,792 | F32 compute; optional F16 matrix residency |
| Selective Q8_0 | 1,808,160,672 | Conversion/diagnostics only; rejected by generation |

Native low-precision matrix arithmetic was tested but is not exposed:

| Diagnostic matrix arithmetic | Failed captures (of 34) | Worst relative L2 |
|---|---:|---:|
| F16 | 6 | 0.002438 |
| BF16 | 9 | 0.013573 |
| Q8_0 | 19 | 0.064830 |

These measurements use the original synthetic full-canvas stress fixture,
not photographic garment-quality inputs. They exceed the unchanged 0.001
activation gate but cannot predict the visual severity of a complete
photographic generation. The accepted low-memory
implementation instead casts resident F16/BF16 matrices to F32 before use.
The diagnostic accepts `--matrix-type f32|f16|bf16|q8_0|q4_0|q5_0|q4_K|q5_K`; add `--upcast-matrices`
to test F32 matrix operands separately from raw low-precision arithmetic.
This preserves failed experiments as reproducible evidence, not supported modes.
Converting a Q8 candidate back to float does not restore the lost precision.

## Native checkpoint recognition

The native loader recognizes the released FASHN preset before generic FLUX
detection. It preserves the original suffixes under `model.diffusion_model.`,
validates all required logical shapes and floating dtypes, and rejects missing
or extra tensors and prefix collisions. The unused `patch_mixer_token` buffer
is optional. Eligible Q8_0/Q4_0/Q5_0/Q4_K/Q5_K matrix metadata is accepted for conversion diagnostics,
but the public generation context explicitly rejects quantized checkpoints.

Name-conversion errors propagate to model initialization and conversion callers
instead of continuing with an invalid checkpoint. Existing non-FASHN name
conversion behavior is preserved. FASHN initializes its dedicated runner before
the generic conditioner/VAE setup and advertises only the try-on capability.

Enable the focused native tests with the existing CMake/CTest toolchain:

```powershell
cmake -S . -B build -DSD_BUILD_TESTS=ON -DSD_FASHN_TEST_CHECKPOINT=..\fashn-vton-reference\weights\model.safetensors
cmake --build build --config Release --target test-fashn-vton test-fashn-vton-c-api test-fashn-vton-graph test-fashn-vton-sampling sd-cli sd-server
ctest --test-dir build -C Release -R fashn-vton --output-on-failure
```

Use an absolute checkpoint path if configuring from another working directory.
`SD_BUILD_TESTS` defaults to OFF and introduces no external test framework.
Without `SD_FASHN_TEST_CHECKPOINT`, the metadata-only test still runs; supplying
it additionally exercises the real safetensors reader and C API capability/
negative-input behavior. `SD_FASHN_TEST_FIXTURES` enables full graph and sampler
oracle CTests. Synthetic FLUX and Hunyuan detection cases protect the shared
classification paths, but do not replace those models' future graph comparisons.

## Quality, trajectory, and reliability baseline experiments

The initial recommended-step numerical, visual, and repeated-use baselines
are established. Keep numerical equivalence, visual try-on quality, and
performance claims separate. The independent pinned PyTorch 20-step
photographic run and native intermediate-trajectory comparisons have completed.
The final PNG differs in 101 of 1,492,992 byte channels, each by one level
(mean byte error 0.0000676494). This is numerical agreement for that case,
not a general image-quality result.

`test-fashn-vton-trajectory` runs the actual native sampler and saves initial
noise plus every updated image and guided velocity. Its optional internal
observer propagates recording failures; ordinary generation does not install
an observer. The real-model one-step recording produced exactly the same
pixels as the public CLI. The full default-build, eight-thread 20-step comparison
also passed: initial-noise max error was 4.76837e-7, schedule error was zero,
maximum updated-image relative L2 was 1.37657e-6, and maximum guided-velocity
relative L2 was 1.44126e-6. All 20 steps passed the unchanged 0.001 gate.
Final PNG differences were the same 101 one-level byte channels described above.
It consumes the reference helper's normalized `conditions.safetensors`, not
the standalone native preparer's PNG manifest.

```powershell
.\build\bin\Release\test-fashn-vton-trajectory.exe model.safetensors prepared\conditions.safetensors native-trajectory 20 1.5 1.5 1 42 1 8
python scripts\compare_fashn_trajectories.py --reference reference-trajectory --native native-trajectory --output trajectory-comparison
```

The comparator checks matching sampling parameters/schedules, reference
artifact hashes, finite matching tensors, initial-noise absolute error at most
1e-6, and relative L2 at most 0.001 for every updated image and guided velocity.
It reports pointwise errors, final byte differences, and numerical PNG PSNR
separately. None of those are perceptual garment-fidelity scores.

### Small research quality matrix

`run_fashn_quality.py` defines six 20-step cases: flat-lay tops with and without
person masking, a flat-lay dress, worn trousers, a patterned worn top, and a
second seed for the same flat-lay-top input. This is deliberately varied but
small and unpaired; it is not a statistical benchmark and has no paired
ground-truth try-on outputs.

Additional inputs are pinned public CatVTON demo assets at revision
`999bdbe81e6008a3f5749af7c1e0b0fa3d21b48e`, distributed under that repository's
stated CC BY-NC-SA 4.0 terms, alongside an existing FASHN example. The driver
verifies hashes and preserves source/license attribution. Images and
derivatives remain outside the source repository. This does not grant access
to DressCode or any other separately restricted dataset.

```powershell
python scripts\run_fashn_quality.py --cli build\bin\Release\sd-cli.exe --preparer build\bin\Release\sd-fashn-prepare.exe --checkpoint model.safetensors --source fashn-source --assets quality-assets --dwpose-dir weights\dwpose --parser-dir weights\human-parser-onnx-eval --output quality-results --steps 20 --threads 16 --download --accept-noncommercial-demo-license --accept-parser-research-license --prepare-only
```

Repeat without `--prepare-only` and with `--resume` to generate. Completed
case checkpoints are retained; interrupted attempts are archived rather than
deleted when retried. Resume refuses changed executables/model/settings.
`--max-new-cases 1 --resume` runs one unfinished case at a time for review.
The suite's `execution_passed` flag stays false until every expected case is
complete; successful completion of one invocation is not full-suite completion.
The report records execution separately from pending visual review and creates
per-case contact sheets. Review garment color/pattern, fit, face/pose,
hands/anatomy, background, and boundary artifacts; do not equate successful
generation with acceptable quality.

All five distinct preparation combinations in this set matched the independent
pinned Python preparation exactly, including crops and all four images.
`test_fashn_quality_preparation.py` also saves reference conditions for
case-specific numerical follow-up.

All six cases completed and were visually reviewed at 20 steps, CFG 1.5,
shift 1.5, skip-last 1, BF16 matrix residency, F32 flash attention, and
16 threads with default CPU kernels. These are 39-forward runs, not the
one-step/CFG1 measurements below. Cold-process wall time includes loading
and PNG encoding but not the separately completed raw-image preparation.

| Case | Seed | Wall time | Main visual limitation |
|---|---:|---:|---|
| Flat-lay cardigan, unmasked person | 42 | 2549.62 s | Distorted graphic and reinterpreted hem/opening |
| Same cardigan, masked person | 42 | 2410.98 s | Improved lower coverage but pushed-up sleeves |
| Floral dress | 42 | 2382.17 s | Neckline/straps and flower motifs reinterpreted |
| Worn trousers | 42 | 2491.28 s | Tight/simplified shape, missing construction details, shortened top |
| Patterned worn top | 42 | 2541.91 s | Changed front closure and softened/deformed stitch details |
| Same unmasked cardigan, second seed | 43 | 2487.24 s | Deeper neckline and disappearance of the white undershirt |

Median latency was **2489.26 seconds (41m29s)**, with a range of
39m42s to 42m30s and peak working set about **2.957 GiB** per case.
These six serial runs took approximately 4h08m in total. They are not
repeated measurements of one workload and do not establish confidence intervals.

The outputs generally transfer recognizable garment colors and broad patterns,
but graphics, closures, neckline, sleeves and construction are not reliably
product-faithful. Non-target clothing and background/shoe shading also change.
Masking is not uniformly better: the controlled cardigan pair trades improved
lower coverage for different sleeve presentation. Changing only the seed
changes styling and layering materially; this is expected stochastic variation,
not evidence of same-seed nondeterminism. The trouser result is insufficient
for product-faithful try-on. Matched pinned PyTorch bottoms inference reproduced
the same simplified trousers, missing construction details and shortened top.
After applying the manifest's crop (x=0, y=48, width=576, height=768), the native
and reference outputs differ in only **43 of 1,327,104 byte channels**, each by
one level (mean byte error 0.0000324014). Checkpoint, sampling schedule/settings,
prepared pixels, normalized conditions and output artifact lineage matched.
This establishes that these visible failures are reproduced by the reference
pipeline, not a C++-specific output discrepancy in this case. It does not
isolate model behavior from preparation effects, nor establish intermediate
trajectory parity for this second case. Do not silently change
masking/compositing semantics to conceal these limitations.

Local evidence is in `C:\source\fashn-vton-reference\checkpoint22-quality-t16`:
`suite.json` records execution and hashes, `visual-review.json` records all six
observations, and `review.html` links contact sheets. Execution completion is
not a "quality passed" flag. These unpaired observations support neither
FID/SSIM claims nor identity verification or production-quality certification.
The targeted bottoms evidence is in
`C:\source\fashn-vton-reference\checkpoint25-bottoms-comparison`, including
`comparison.json`, `reference-cropped.png`, `native.png`, and `difference-x32.png`.

`compare_fashn_case_output.py` supports a targeted reference follow-up without
rerunning native inference. It verifies checkpoint/settings/schedule, normalized
condition-to-PNG lineage, identical native/reference prepared pixels, output
hashes and final reference tensor-to-PNG conversion, then applies the native
manifest's exact crop. Its predeclared gate is at most one byte level per
output channel. It saves both cropped outputs, an amplified difference image,
and a report. This is **final-image parity**, not a second full intermediate
trajectory comparison or a garment-quality metric.

```powershell
python scripts\compare_fashn_case_output.py --source fashn-source --reference reference-trajectory --reference-inputs reference-prepared --native-run quality-results\worn-bottoms-free --prepared quality-results\prepared-worn-bottoms-free\manifest.json --output bottoms-comparison
```

### CPU operation profiling and alternative kernels

The graph diagnostic accepts `--profile-ops` and `--threads N`.
`--profile-primitives` checks the instrumentation without model weights.
Profiling synchronizes and executes each node separately, preventing cross-node
fusion and adding dispatch overhead: these timings identify likely costs,
not ordinary end-to-end latency.

On the default eight-thread F32-flash/BF16-resident build, the instrumented
conditional forward spent approximately 46.22 seconds in matrix multiplies,
19.93 seconds in flash attention, and 13.37 seconds in tanh. The corresponding
null-branch figures were 45.50, 19.29, and 10.62 seconds. All captured-layer
gates still passed under instrumentation.

A separate `GGML_LLAMAFILE=ON` CPU build required no submodule changes or new
runtime dependency. Both oracle timesteps passed. Two isolated one-step CLI
runs took 79.09 and 75.45 seconds at about 2.950 GiB, with 45 byte channels
differing from the historical default-build one-step output, by at most one.
Fresh controls and thread scaling subsequently measured:

| CPU configuration | Individual one-step runs | Median |
|---|---|---:|
| Default kernels, 4 threads | 111.41 s, 112.02 s | 111.71 s |
| Default kernels, 8 threads | 76.28 s, 74.19 s | 75.24 s |
| Default kernels, 16 threads | 62.72 s, 63.61 s | 63.17 s |
| LLAMAFILE kernels, 8 threads | 68.95 s, 69.58 s | 69.26 s |
| LLAMAFILE kernels, 16 threads | 63.14 s, 63.15 s | 63.14 s |

These were serial fresh-process runs on this VM, using BF16 matrix residency,
F32 flash attention, one step, CFG 1 and seed 42. Peak working set remained
about 2.950 GiB and all one-step pixel checks passed. Two repeats do not
establish confidence intervals. The combined kernel/thread change provides no
meaningful additional benefit over default kernels at 16 threads, so default
kernels are preferred for the quality sweep. The full sixteen-thread trajectory
comparison passed all 20 steps, with the same maximum image/velocity errors
and 101 one-level PNG differences as the eight-thread comparison. On this VM,
`--type bf16 --diffusion-fa -t 16` is therefore the selected setting for the
quality matrix; the best thread count on another machine must be measured.
`run_fashn_cpu_study.py` schedules two fresh-process repeats each for the default
build at 8/4/16 threads and the alternative build at eight threads. It checks
output dimensions and a predeclared one-byte maximum difference from a matching
one-step PNG, retains individual measurements, and does not claim confidence
intervals. Run it without any other model experiment running on the machine.

### Repeated-use and graceful shutdown

FASHN server contexts now handle SIGINT/SIGTERM and Windows Ctrl+Break
cooperatively. The signal handler only sets a lock-free flag; a normal thread
stops accepting requests, cancels queued jobs, requests active cancellation,
and waits for the worker. Cancellation still waits for a native stage or
sampling forward. Requests reaching the try-on publication lock after shutdown
starts receive 503. Failed FASHN listener startup returns a nonzero exit code.
Non-FASHN signal behavior is unchanged.

This is clean termination, not a persistent job database: in-memory jobs are
not restored after restart. Job IDs now include a random server-instance
namespace so a restarted server cannot normally reuse an old timestamp/counter
ID. IDs remain opaque identifiers, not authorization credentials. The browser
releases unavailable jobs on 404/410 instead of retaining them indefinitely.
Actual restart/old-job lookup, failed-listen, and idle Windows signal tests
passed. The six-request sustained-use run also passed, producing seven images
including its final two-sample request. The adjusted post-warmup private-memory
span was 562,008 bytes (0.54 MiB), below the 128 MiB budget. All first samples
matched the single-sample CLI, the second batch sample was distinct, and 16
queued requests from eight concurrent clients were cancelled correctly.
Busy shutdown cancelled two queued jobs and stopped the active job, exiting
cleanly in 73.16 seconds: still approximately one CPU forward, not an immediate
kernel interruption.

The updated real Chromium raw-upload/download/cancellation flow passed with
exact CLI pixels and no page errors or external requests. The UI additionally
has focused tests for 404/410 recovery and late cancellation responses.

The HTTP test runner adds opt-in `--soak-rounds 6 --graceful-shutdown`.
The Windows soak compares each request's first result to the CLI, samples private memory,
excludes two warmup requests, approximately accounts for retained base64
results, and applies an explicit 128 MiB post-warmup span budget (configurable).
Retention accounting uses the server-advertised completed-job TTL so extending
the soak does not subtract already-expired results indefinitely.
Its last request asks for two samples, verifies the first matches single-sample
generation and the second is distinct, and checks aggregate progress.
It also exercises eight concurrent clients submitting/cancelling 16 queued
jobs, and active/queued shutdown. A short bounded-memory test is not proof
that indefinite operation cannot leak.

## Deferred work and validation boundaries

### Full-image quantization diagnostics

The test-only trajectory runner can generate complete Q8 images even when
the public numerical acceptance gate fails. Public CLI/C API/server Q8
generation remains rejected. Use the original floating checkpoint:

```powershell
.\build\bin\Release\test-fashn-vton-trajectory.exe model.safetensors conditions.safetensors q8-trajectory 20 1.5 1.5 1 42 2 16 --matrix-type q8_0
python scripts\compare_fashn_trajectories.py --reference reference-trajectory --native q8-trajectory --output q8-comparison
```

The first complete photographic bottoms run used all 104 eligible Q8
matrices at 20 steps/CFG1.5/shift1.5/skip1/seed42/16 threads. It completed in
1947.35 seconds with 4.151 GiB peak working set in the trajectory recorder.
That is not yet a matched public-CLI memory or speed comparison: conversion,
mapping and recording affect this diagnostic process.
Against the cropped validated native floating output, mean byte error was
0.130162, maximum 41, and numerical PNG PSNR 55.584 dB. Normal-scale visual
inspection found very similar appearance and no obvious new garment/pose
failure in this single case. The existing simplified trousers/shortened top
remain present in both images.

The complete F32-math-reference trajectory comparison still failed: 9/20
updated-image states and 20/20 guided velocities exceeded 0.001 relative L2.
Maximum image/velocity relative L2 was 0.00446338/0.00851191. This distinction
is deliberate: a strict numerical failure is not automatically a visibly
unusable image, and a similar-looking image does not establish broad quality.

Here category `2` means bottoms. The comparator saves final PNGs and per-step
metrics even for numerical gate failures; a nonzero comparison exit code must
not be relabeled a numerical pass merely because the image looks plausible.
It requires finite valid trajectories. Apply the prepared manifest's crop
before comparing with a cropped CLI output.

Both graph and trajectory diagnostics accept `--f32-matrices policy.json`.
The policy lists exact eligible names to restore from floating source weights:

```json
{"f32_matrices": ["x_patch_mixer.0.linear1.weight"]}
```

This is a syntax example, not a measured recommended precision policy.
Unknown/protected names, duplicates and attempting to recover original F32
precision from already-quantized source tensors are rejected. Each run records
the actual types of all 104 eligible matrices. Protected tensors remain F32.
Adding `--upcast-matrices` to a Q8 diagnostic isolates weight rounding from
the usual quantized-activation dot-product path by dequantizing before F32
matrix multiplication. Such modes are diagnostic-only, not validated public
support. The default trajectory mode remains BF16 storage with F32 computation.

`run_fashn_matrix_ablation.py` performs resumable, serial restoration of one
eligible matrix at a time against a fixed photographic graph oracle, with
an all-Q8 baseline. `--max-new-matrices N` bounds a checkpoint batch; `--resume`
refuses changed executables, weights, fixtures or settings. It records the
actual precision map, both branches' velocity errors, capture failures and
per-intervention timings. Its objective is the maximum of conditional/null
final-velocity relative L2. Rankings describe restoration effects on that
probe, not independent causal importance or generalization guarantees.
The graph's `--no-capture-files` retains metrics without saving large activation
payloads for every candidate. Combined policies and held-out inputs/states
must be evaluated separately before adopting a precision configuration.

The first eight photographic interventions (the two linear matrices in each
of the four patch mixers) did not materially reduce the ranking objective.
The all-Q8 worst-branch velocity relative L2 was 0.00262507; the best of those
eight, restoring `x_patch_mixer.3.linear1.weight`, reached 0.00262392, only
0.044% lower. Six interventions slightly increased the objective.
These are 8/104 single-matrix results, not a full ranking or a combined-policy
test, and do not justify blanket restoration of the patch mixers.
At 40/104 coverage, the best individual restoration was
`double_blocks.0.img_mlp.2.weight`, reducing the same objective by 0.17025%
to 0.00262060, still above the gate. That partial scan did not yet include
the later single-stream blocks.
The subsequent complete 104-matrix scan identified a stronger intervention:
restoring `single_blocks.12.linear1.weight` reduced the worst-branch velocity
relative L2 to 0.00155743 (40.6711% lower), with 103 eligible matrices still Q8
and 32.129 MiB additional weight payload. It did not pass the gate alone.
Small combined subsets and a weight-only Q8/F32-compute control were evaluated
at initial and actual later reference states. Restoring single-block linear1
weights 12, 11, 8 and 14 was the best small direct-Q8 combination tested at
both states, leaving 100 matrices Q8 and adding 128.516 MiB of weight payload.
Worst-branch velocity relative L2 fell from 0.00262507 to 0.00127498 initially
and from 0.00347500 to 0.00121523 at step 15 (51.4% and 65.0% reductions).
The weight-only Q8/F32-compute control improved further to 0.00105051 and
0.00102306, but all 12 probes failed the complete capture gate. These are
teacher-forced states of one photographic case, not independent image tests
or complete quantized trajectories. The four-matrix subset is an experimental
full-image candidate; no mixed policy has been promoted to public inference.
The complete four-matrix bottoms image subsequently reduced cropped mean
byte error versus floating native output from 0.130162 to 0.058326 (55.2%).
Final trajectory-image relative L2 improved from 0.004157 to 0.002693, but
four of twenty updated images and all twenty guided velocities still failed
the unchanged gate. Normal-scale appearance remained very similar, including
the same garment-fidelity limitations. The mixed recorder took 2007.39 seconds
versus 1947.35 seconds for all-Q8 with the same executable and inputs; these
are single observations, not repeated benchmark confidence intervals.

The same fixed four-matrix policy was then applied to an independent cardigan
image. Cropped mean byte error fell from 0.258655 to 0.190678 (26.3%), but
maximum error remained 82 levels versus 83 for all-Q8, concentrated at a
sleeve/torso boundary. Numerical PNG PSNR was 44.082 versus 44.521 dB.
Normal-scale appearance remained very similar; this is not evidence that
garment-construction inaccuracies or localized boundary differences are solved.
Cardigan comparisons cover final images, not an intermediate reference
trajectory. All-Q8 took 1956.68 seconds and mixed took 1975.86 seconds.

A matched BF16-storage/F32-compute bottoms recorder subsequently took 2640.87
seconds, with the same executable, checkpoint, conditions and settings.
Q8 and mixed used 26.3% and 24.0% less wall time in these single observations,
but **more** measured peak working set: 4.151/4.277 GiB versus the floating
recorder's 3.461 GiB. These diagnostic loading/conversion/recording results
are not a benchmark of a public prequantized-GGUF deployment. The floating
recorder passed every reference image/velocity state (maximum relative L2
0.000002017/0.000002451), and its cropped PNG exactly matched the earlier
native CLI. Neither quantized policy has passed the unchanged numerical gate.

### Phase-resolved memory diagnostics

The trajectory executable also accepts `--mmap on|off`, `--load-threads N`,
`--load-only`, `--probe-forwards N`, `--no-record`, `--memory-profile`, and
`--classify-pages`. Its historical mmap default is **on**, unlike the public
CLI default; compare explicit matching settings rather than assuming they
have identical loading memory. Loading threads default to inference threads.
`--load-only` eagerly loads weights without using the conditions file.
`--probe-forwards N` instead repeats conditional time-zero inference on the
same seeded noise; it is not a trajectory or a quality/image-generation test.
These two modes are mutually exclusive. Probes save `probe.safetensors`;
unrecorded sampling saves only `final.safetensors`. The ordinary default still
records initial noise and every sampler step.

```powershell
.\build\bin\Release\test-fashn-vton-trajectory.exe model.safetensors conditions.safetensors memory-probe 20 1.5 1.5 1 42 2 16 --matrix-type q8_0 --mmap off --load-threads 2 --probe-forwards 2 --no-record --memory-profile --classify-pages
```

`memory-profile.jsonl` tags loader, conversion, graph-build/allocation/execution,
observer, workspace-release, and model-destruction phases. It distinguishes
logical parameter payload, allocated buffer capacity, directly mapped weights,
retained runtime buffers, conversion read/F32-buffer live/peak capacities, and
planned CPU work space. Unavailable backend-internal allocation totals are
explicitly null. Scratch counters do not include every backend/allocator
allocation or transient vector-reallocation overlap.

Windows profiles additionally record current/kernel-peak working set, private
commit, page faults, and thread count. Opt-in page classification uses
`VirtualQueryEx`/`QueryWorkingSetEx` at selected boundaries and identifies
resident source-model pages. Scans are non-atomic; region type is not the same
as shared/private page ownership. Private commit is not private working set.
These instrumented runs must not be presented as clean latency benchmarks.
A low load-only working set for mmap weights can simply mean their pages have
not yet been touched; include warm forwards.

The generic command-measurement specification supports `"memory_timeline": true`
to save timestamped process samples. `summarize_fashn_memory.py --directory
study --output summary` validates completed `*-spec.json` runs, command/spec
identity, graph counts, scratch release, and mapping destruction, and writes
hash-linked JSON/Markdown observations. Each corresponding measurement must
be in `<spec-name-without--spec>-measurement`. Numerical same-policy comparisons
remain a separate requirement; successful profiling does not enable quantized
public inference or relax its reference gates.

### Runtime-policy-matched weight files

`export_fashn_runtime_weights.py` wraps the existing native converter and
conversion verifier. It exports BF16 core/F32 protected tensors, selective Q8,
or a selective-Q8 file with explicit original-F32 restorations. Supply the
`matrix_types` manifest from a completed native diagnostic so the BF16 export
rule names exactly the eligible matrices; an ordinary all-BF16 conversion
still requires protected-weight conversion at runtime.

```powershell
python scripts\export_fashn_runtime_weights.py --cli build\bin\Release\sd-cli.exe --verifier build\bin\Release\test-fashn-vton-conversion.exe --checkpoint model.safetensors --matrix-map diagnostic\manifest.json --matrix-type bf16 --output runtime-bf16
python scripts\export_fashn_runtime_weights.py --cli build\bin\Release\sd-cli.exe --verifier build\bin\Release\test-fashn-vton-conversion.exe --checkpoint model.safetensors --matrix-map diagnostic\manifest.json --matrix-type q8_0 --f32-matrices policy.json --output runtime-mixed
```

`--reuse-file existing.gguf` verifies an existing file without reconverting it.
Loading/conversion threads default to two for this helper. Output directories
must be new. Its hash-linked `export.json` remains incomplete on conversion or
verification failure. The verifier's `--runtime-policy` mode checks every
shape/type, original protected/restored values, and exact Q8 blocks against
quantization of the original floating weights, not just a dequantization
error bound. Original-F32 restoration must start from floating source weights.
The model's 365 inference tensors are checked alongside the optional unused
token when present.

Use the same matrix type and restoration policy when loading the result.
For a ready BF16 file, the public CLI still needs the corresponding `--type
bf16`; adding `--mmap` permits direct mapped storage of matching tensors.
Q8/mixed files remain test-only for generation. Do not remove public numerical
guards merely because loading is cheaper. Ready files should be checked for
zero runtime conversions and measured through warm forwards; file size or
untouched mmap pages alone do not prove a memory improvement.

### Experimental modulation precomputation

The numerical graph and trajectory diagnostics accept
`--precompute-modulations`. This is an opt-in infrastructure experiment,
not a new default. The trajectory diagnostic also
requires `--mmap off` and rejects load-only mode; the graph diagnostic selects
unmapped loading for this experiment. This restriction prevents a supposedly
released modulation weight from remaining resident in a retained source view.

The runner first computes the time/category/combined vectors, then processes
one of the 37 modulation projections at a time using the existing F32
operators and per-vector ordering. Between layers the diagnostic's weight
scope finishes the runner, releases parameter storage and re-registers the
unchanged metadata. Denoising consumes the cached outputs through the existing
single/double-stream supplied-modulation routes and a supplied final-layer
route; the corresponding original weight tensors are no longer graph inputs.
Caching without weight release is not considered a memory optimization.

The trajectory mode precomputes the complete requested schedule/category set,
even when `--probe-forwards` only executes repeated t0 forwards. A 20-step,
CFG1.5, skip1 schedule requires 39 pairs: 30.088 MiB of modulation outputs plus
0.571 MiB of time/category/combined vectors needed to preserve debug captures.
Cache keys retain exact F32 time bits and category; reuse also requires the
same thread count and matrix-arithmetic setting. They are scoped to the same
runner/model instance, with weight adapters disallowed. Missing keys outside
the configured request or a changed arithmetic setting fail explicitly rather
than silently using stale vectors or falling back.

The current experimental cache has a 128 MiB default payload budget.
`--modulation-cache-mib N` selects 1..128 MiB in the trajectory diagnostic.
Large requests use bounded windows rather than allocating every pair at once;
the low-level single-window API rejects oversized allocations before touching
weights. Refills can occur between CFG branches without discarding the
already copied conditional result. Cancellation is polled during
precomputation, and failed partial caches are cleared. Profiles report cache
payload separately from runtime buffers and identify inactive registered
parameter bytes; those tensors must not be counted as resident weights.

The library/CLI/server context has an explicit experimental opt-in through
its existing model-arguments field, without changing the C ABI:

```powershell
--model-args '{"fashn_modulation_cache":true,"fashn_modulation_cache_mib":128}'
```

Omit `--mmap` and `--eager-load` with this option; incompatible combinations
are rejected. Unknown keys, non-boolean enable flags and budgets outside
1..128 MiB are rejected. The per-request schedule/CFG plan is shared with the
diagnostics. Cache refills honor the request cancellation callback, progress
retains ordinary sampling-step semantics, and request-scoped callbacks are
cleared on every exit while image-independent cached vectors may be reused.
The public floating-only and CPU-only guards remain unchanged. Full-trajectory
and public multi-sample/windowed validation are required for acceptance; the
option is not enabled by default. The completed 20-step photographic bottoms
trajectory passed every original-reference gate, taking 2577.73 seconds with
2.009 GiB peak working set on the CPU VM. A two-sample public request with a
deliberately tiny 1 MiB cache produced pixel-identical outputs to the ordinary
path but was about 4.4% slower because of refills. Modulation tensors also
matched exactly at three times and all four category IDs. This establishes a
memory tradeoff, not a general throughput improvement.

### Experimental strict-F32 GELU fusion

The graph and trajectory diagnostics accept `--fused-gelu`. It replaces the
decomposed strict GELU operations with a threaded, stride-aware custom CPU
operation using the same tanh formula. It does not enable FP16 lookup or
approximate tanh. Default FASHN and other-model behavior remains unchanged.
`test-fashn-gelu` covers analytic/stride/thread behavior; `--benchmark` runs
real MLP shapes with synthetic inputs. Primitive improvements alone do not
establish full-image speed or numerical acceptance. Initial measurements
showed no graph-arena reduction, because the original graph already reused
its elementwise buffers. Full-model comparisons are a separate gate.
The explicit public model-argument option is `"fashn_fused_gelu":true`,
alongside the required `fashn_modulation_cache` boolean (which may be false).
Both remain off by default. Short A-B-B-A model controls were bit-identical
and showed an approximately 1.2% warm-median improvement with overlapping
timing ranges, not a large or statistically established speedup.

### Diagnostic CPU BLAS comparison

An isolated build with `GGML_BLAS=ON` and a supported BLAS vendor can run
`test-fashn-vton-trajectory ... --backend BLAS` or the graph diagnostic's
existing `--backend BLAS` option. This is **not public BLAS/GPU support**.
The diagnostic explicitly configures the BLAS thread count and uses the
existing scheduler's CPU fallback for unsupported operations, including
strict F32 flash attention and small modulation projections. Cached
modulations remain available. No GGML source modification is required.

F32 matrix operands are mandatory. Use F32 weights, or `--matrix-type bf16
--upcast-matrices`; direct Q8 execution is rejected rather than silently
changing its arithmetic. The trajectory tool enables floating upcasts by
default. Its manifest records actual scheduled matrix-backend counts.
Per-backend workspace capacities exclude backend-private allocations;
shared host buffers are reported only once by GGML's allocator. Compare
fresh-process peak memory as well as warm forwards before selecting a backend.

`test-fashn-math --benchmark` compares supported real-size F32 GEMMs with CPU
and BLAS. `--check-blas` exercises CPU-fallback flash parity and precision/
thread guards. `--attention-benchmark NEW_DIRECTORY` records native F32 flash
at 3456/6912 tokens and ten 128-dimensional heads. Then run
`scripts\compare_fashn_attention_math.py --native NEW_DIRECTORY --source
PINNED_FASHN_SOURCE --output NEW_COMPARISON_DIRECTORY` with the existing
reference Python dependencies. It compares the unchanged upstream default
CPU SDPA processor, including original single-block strides and explicit
packed-layout/I/O controls. Synthetic kernel timings are not full-model
throughput or image-quality claims. Keep dependency DLL lookup process-local
and use a separate build so the ordinary CPU binary remains available.

### Integrated CPU optimization results

For an explicit upstream comparison, the
[original Python baseline](../reports/original-python-baseline.md) records the
unmodified default CPU sampler at 1241.326 / 1263.737 s sampler-plus-PIL and
1258.037 / 1325.477 s process wall for cardigan / bottoms. Valid bottoms memory
is 6.573 GiB working set and 9.603 GiB private commit; cardigan memory is
unavailable. This is historical prepared-input inference with batched CFG, not
raw pipeline startup, the forced-math oracle or a matched native recording interval.

On the 16-logical-CPU Windows evaluation VM, matching runtime-ready weights,
mmap off, bounded modulation precomputation with parameter release, and
strict fused GELU reduced full 20-step Q8 working-set peaks from approximately
4.16 to 1.40 GiB. Mixed precision fell from approximately 4.30 to 1.53 GiB.
All four full photographic Q8/mixed runs preserved their 21 historical
same-policy recordings bit-for-bit. This establishes loading/lifetime
equivalence, **not floating-equivalent quantized quality**; public quantized
guards remain unchanged.

For the independent cardigan case, floating CPU/BLAS sampling took
2371.13 / 1598.14 seconds at the same settings. Alternating short controls
showed approximately 33% lower warm-forward time with BLAS. Floating
trajectory gates passed, but the memory tradeoff is important: working-set
peaks were 2.01 / 2.03 GiB, while sampled private-commit peaks were
2.14 / 4.28 GiB. These metrics overlap and must not be added together.
Isolated DLL-only probes attributed approximately 1.88 GiB of private commit
to 16-thread OpenBLAS startup before any model or matrix multiplication.
Selecting CPU in a binary that loads this DLL does not eliminate that cost;
retain the separate no-BLAS binary for lower-commit floating execution.
BLAS remains diagnostic-only.

The lower-memory floating options are explicitly enabled through existing
model arguments, with mmap and eager loading disabled:

```json
{"fashn_modulation_cache":true,"fashn_modulation_cache_mib":128,"fashn_fused_gelu":true}
```

These measurements use prepared inputs, 576x864 generation, 20 steps,
CFG1.5, shift1.5, skip1, seed42 and 16 threads, not raw HTTP end-to-end
latency. Existing original-Python observations remain faster than native
BLAS; no universal speed claim is made. See the published
[optimization evidence](../reports/memory-and-latency-optimization-results.md)
and [full-resolution gallery](../reports/comparison-gallery-optimized/index.html).

### CFG pairing diagnostics

`test-fashn-math --cfg-benchmark` compares two sequential F32 BLAS matrix
operations with batch-dimension and flattened-token alternatives at core
FASHN shapes, under a 1 GiB per-matrix workspace ceiling.
`--cfg-attention-benchmark` compares independent sequential and paired
branches in the F32 attention primitive. Both verify branch outputs.
Neither option enables full-model B=2 support or changes the public sampler.
Initial measurements showed substantially larger workspaces without a
consistent throughput benefit, so sequential CFG remains the default.
Graph-construction profiling likewise did not justify additional
image-specific caches or weakening per-forward validation.

### Additional quantization and upstream evidence

`summarize_fashn_ablation.py` exports a hash-verified CSV/Markdown snapshot
with per-matrix F32-versus-Q8 weight-payload costs. It proposes small positive-
effect subsets only after all 104 interventions complete; those combinations
remain unvalidated until evaluated jointly and on held-out states/images.

For a real nonzero-time validation probe, the reference `export` subcommand
accepts `--trajectory reference-trajectory`. It loads the previous updated
image at `--step-index` (or recorded initial noise for index zero), verifies
the state hash, source/checkpoint/input provenance, sampling settings and
schedule, and records that lineage. Without this option, the diagnostic keeps
using seeded noise even at a nonzero time; that older probe is not an actual
intermediate sampling state.
The reference-state probe is teacher-forced and does not replace a complete
free-running trajectory comparison for a selected precision policy.
The gallery builder accepts saved `.safetensors` output images in addition to
RGB PNGs. It requires finite F32 tensors, uses the trajectory comparator's
clamp/scale/truncate conversion, and applies only an explicitly supplied crop.
Rendering a final state does not establish intermediate numerical agreement.

`run_fashn_upstream.py` separately calls the unmodified pinned
`TryOnPipeline._sample` with default CPU attention dispatch and batched CFG.
It consumes verified prepared conditions, bypasses raw preparation/constructor
setup, and uses meta-device checkpoint loading. Its sampling time must not be
described as raw-pipeline startup/end-to-end latency. Historical F32-math
oracle timings are not measurements of this upstream default sampler.

The first actual-upstream cardigan run at 20 steps and 16 threads took
1258.04 seconds wall time (1241.33 seconds sampling/PIL), versus the earlier
native floating CLI's 2549.62 seconds on identical prepared inputs/settings.
Only 69 of 1,327,104 output byte channels differ, each by one level.
Python is faster in this observation; it is not an interleaved repeated
benchmark or proof of a universal speed ratio.

The second actual-upstream case (bottoms) took 1325.48 seconds wall time
and 1263.74 seconds sampling/PIL, compared with the historical native CLI's
2491.28 seconds. Only 44 output channels differ, each by one byte level.
Correct interpreter-PID monitoring recorded 6.573 GiB peak working set for
Python versus 2.957 GiB for the native baseline. These observations show
faster Python execution and lower native memory use, not a universal
language-level advantage. Both exclude raw preparation and are not
interleaved repeated measurements.

The first Python memory observation is invalid for model comparison: Windows'
venv launcher was monitored instead of its interpreter. The measurement
harness now registers and validates the active interpreter PID before model
imports, samples that process, and terminates the launched worker tree on
timeout. Reports without verified interpreter scope omit Python memory
comparisons. Native executable memory measurements are unaffected.

### Remaining support gates

Native preparation, opt-in raw HTTP/browser uploads, active-job cancellation/progress, a standalone UI,
floating GGUF conversion, selective Q8 conversion diagnostics and validated
reduced-memory F16/BF16 residency are implemented. The remaining gates are:

- GPU execution/backend validation: unavailable on the CPU-only development
  machine. The one-CPU-backend guard remains; no GPU support is claimed.
- Native low-precision arithmetic and quantized inference: failed the current
  numerical gates. Q8 remains conversion/diagnostic-only; Q4/Q5 are not enabled.
- Broad statistical quality coverage, larger batches and 30/50-step sweeps
  remain beyond the completed initial six-case/20-step baseline.
- Further throughput tuning beyond the measured flash/residency and CPU-thread
  improvements.

Parser-dependent preparation remains separately restricted to authorized
non-commercial research/evaluation. It is not an unrestricted commercial
preprocessor. Authenticated production hosting and integration into the
separate general-purpose frontend remain outside this experimental local
workflow. The browser and raw HTTP path do not change those trust boundaries.

## Licensing

FASHN's model/source are Apache-2.0 and attribute FLUX-derived components.
The optional reference human parser inherits NVIDIA SegFormer's
non-commercial research/evaluation license. It is not automatically installed
or downloaded automatically. Native ONNX/OpenCV preparation preserves the
same parser restrictions. Additional execution backends remain unvalidated.
