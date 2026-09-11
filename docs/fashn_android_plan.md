# FASHN VTON 1.5 on Android: feasibility and checkpoint plan

**Status: original analysis and proposed work, not an implemented Android app.**
Subsequent host preparation has produced Android cross-builds and a
diagnostic imported-noise path; see [the device transfer runbook](fashn_arm_transfer.md)
for executable handoff instructions and the current host report. Native
Windows ARM64/Android FASHN execution, APK, GPU and NPU acceptance remain
separate gates. Interface sketches below remain proposals.

Source baseline: `55db93d6f2536f12428e170acbafd0cca3182c59`.
GGML baseline: `e20c3a14aa70ee84ca58499814206dd08d8026bc`.
The submodule uses the `leejet/ggml` fork, as recorded in `.gitmodules`.
The existing [developer guide](fashn_vton.md), [quickstart](fashn_quickstart.md),
and published reports remain authoritative for completed Windows work.

## 1. Verdict

**An ARM64 Android CPU port is technically credible. A fast, reliable,
general-purpose phone application is not yet established.**

The neural network already executes through portable C++ and GGML rather
than an embedded Python interpreter. GGML has ARM CPU kernels and explicit
Android ARM backend variants. Android's NDK provides the required native
toolchain. Prepared inference does not require PyTorch, a text encoder,
a VAE, OpenCV, or ONNX Runtime.

The difficult part is not translating the model into Java. It is preserving
its numerical behavior while fitting a large, long-running workload into a
phone's memory, thermal envelope, and application lifecycle. Raw photograph
preparation and accelerated GPU/NPU execution introduce additional work.

Recommended first target: an explicitly supported, high-memory ARM64 Android
device, **prepared inputs, CPU, BF16 weight storage with F32 computation**.
Keep the existing floating baseline and experimental quantization boundaries.
Prove that path before investing in a full camera/gallery application.

### What the available evidence does and does not prove

| Evidence | Established | Not established |
|---|---|---|
| Existing Windows x64 FASHN results | Full model, upstream parity, resource attribution, complete images | Android or Windows ARM64 FASHN behavior |
| User-reported Windows ARM64 smoke | Checkout `d04e895`; Clang 19.1.5; native ARM64 PE machine `0xAA64`; SDXS 512x512 cat, one step, six CPU threads, 33.4 seconds | FASHN kernels, FASHN accuracy, Android ABI, phone performance, GPU/NPU |
| GGML source | ARM kernels, Android-specific backend variant definitions, CPU abort API | Every relevant operation passing on a particular phone |
| Android/ORT documentation | Supported native build/deployment mechanisms | This checkout building or running successfully without adaptation |

The Windows executable is a **PE binary using the Windows environment**.
Android needs an **AArch64 ELF binary built against the Android NDK/Bionic**.
Matching instruction architecture does not make these binaries interchangeable.

The ARM machine should now run the FASHN revision above, not only the
upstream `d04e895` checkout used for the SDXS smoke. It is an excellent
intermediate correctness platform. It cannot replace a real Android device
for allocator, driver, lifecycle, thermal, or sustained-performance work.

## 2. The workload we are actually porting

| Property | Current FASHN implementation |
|---|---|
| Model domain | Pixel RGB; no latent VAE or text encoder |
| Prepared canvas | Width 576, height 864 |
| Patch size | 12x12 |
| Tokens | 3,456 per stream; 6,912 joint tokens |
| Hidden dimension | 1,280 |
| Attention | 10 heads of width 128; 28 attention operations per forward |
| Blocks | 4 target patch mixers, 8 double blocks, 16 single blocks |
| Target conditioning | Noisy RGB + person RGB + grayscale pose |
| Garment conditioning | Garment RGB + grayscale pose |
| Baseline sampling | 20 steps, CFG 1.5, shift 1.5, skip final CFG step |
| Forward count for that baseline | 39 |
| Runtime sample handling | Sequential B=1 graphs; public sample count 1..4 |

A single unstreamed joint-attention score tensor has:

```text
6,912 * 6,912 * 10 * 4 = 1,911,029,760 bytes
                         approximately 1.78 GiB
```

That is one tensor, not the whole process. Losing memory-efficient attention
is therefore a serious mobile regression. The baseline must keep all
**28 F32-key/value flash operations**, rather than silently falling back to
large materialized attention matrices.

The SDXS one-step cat result cannot be multiplied by 20 or 39 to predict
FASHN latency. These models have different architectures, sequence lengths,
operation counts, and conditioning paths.

Changing the canvas, reducing CFG/steps, using distilled weights, or changing
the sampler can be useful product research. None is a transparent Android
port optimization. Such changes require a separately labeled quality study.
Explicitly set 20 steps in experiments: the C API initializer currently
defaults to 30.

## 3. Existing support and concrete gaps

### 3.1 CPU execution: substantial reusable infrastructure

The following source is already present:

| Source | Relevance |
|---|---|
| `ggml\src\ggml-cpu\CMakeLists.txt`, lines 102-236 | ARM source selection, native detection, explicit ARM architecture flags |
| `ggml\src\CMakeLists.txt`, lines 403-424 | Separate Linux and Android ARM backend variant lists |
| `ggml\src\ggml-cpu\arch\arm\cpu-feats.cpp`, lines 3-117 | AArch64 feature detection using Linux HWCAP; optional dot-product, FP16, I8MM, SVE, SME |
| `src\model\diffusion\fashn_vton_model.h`, lines 518-563 | F32 matrix operands/precision, F32 K/V, backend operation validation, flash-node guard |
| `include\stable-diffusion.h` | Existing prepared-input C API and output ownership |
| `tests\CMakeLists.txt` | Existing contract, primitive, sampling, GELU, modulation and math targets |

The initial distribution should use an ARMv8-A baseline, not
`-mcpu=native` inferred from the development computer. AArch64 provides
NEON; I8MM, BF16 arithmetic, SVE, and SME are not universal requirements
that can be assumed across phones.

GGML's all-variants mechanism requires dynamically loaded backends. Do not
enable it in a static baseline build and assume dispatch is working.
Device-tuned builds or properly packaged dynamic variants belong in a later
optimization checkpoint.

**BF16 storage does not require native BF16 arithmetic here.** The protected
parameters and matrix operands remain F32; the core weights are expanded for
computation. Likewise, the presence of an INT8 instruction extension does
not prove that an actual FASHN operation used it.

### 3.2 Build and packaging are not Android-ready application infrastructure

The root project supports shared/static libraries, but has no FASHN Android
application, JNI bridge, Android presets, device harness, or demonstrated
Android dependency package.

`SD_BUILD_EXAMPLES=ON` adds both CLI and server directories. Building just
the `sd-cli` target need not build the server, but still configures its
directory. `SD_SERVER_BUILD_FRONTEND=OFF` avoids the unrelated frontend
build. A library/JNI build should disable examples entirely.

The root source glob includes other model families and tokenizer sources.
A CLI-only target is not a FASHN-only engine. Measure resulting binary size
before considering selective compilation; do not initially rewrite the
model dispatcher or inspect/copy large vocabulary payloads.

Keep exceptions and RTTI: the current implementation uses both. Do not
apply common game-engine `-fno-exceptions`/`-fno-rtti` presets blindly.

For the CLI proof, static GGML/core and `c++_static` simplify deployment.
For the eventual multi-library JNI + ORT + OpenCV package, use one compatible
`libc++_shared.so` and a coherent NDK/STL dependency set. A Windows DLL or
desktop Linux/glibc `.so` cannot substitute for an Android library.

### 3.3 Numerical portability needs its own acceptance ladder

There are at least four independent sources of drift:

1. ARM reduction ordering, vector kernels, and multiply-add behavior.
2. Compiler treatment of explicit rounding and mathematical expressions.
3. Bionic/libc++ versus Windows transcendental implementations.
4. Alternative storage, attention, BLAS, or device backend arithmetic.

`src\core\strict_gelu.h` uses a scoped MSVC precision path when
`_MSC_VER >= 1930`; other compilers use explicit volatile F32 intermediates.
Android Clang follows the latter branch. Windows Clang's branch depends on
its target and compatibility macros. Both paths still use `std::tanh`.

The default unfused strict GELU in `src\model\diffusion\flux.hpp`, lines
22-34, builds elementary F32 operations rather than the legacy CPU GELU
lookup-table operation. The fused alternative uses `ggml_map_custom1`.
Test both; initially leave fusion off on Android.

`src\core\rng_mt19937.hpp` has a fixed integer engine but uses
`log`, `sqrt`, `sin`, `cos`, and a small-array `log1p` path to obtain normal
samples. Equal seeds do not by themselves prove equal floating noise across
platforms. Do not substitute `std::normal_distribution`.

Separate two experiments:

```text
Identical imported noise + identical conditions
    -> isolate transformer/sampler/backend differences

Public CPU RNG + same seed + same logical sample count
    -> establish actual end-to-end reproducibility
```

The graph fixture harness accepts exported model inputs. The subsequent
host-preparation change adds `--initial-noise` to the diagnostic trajectory
harness, with strict dtype/shape/finite-value validation. It is not a public
CLI feature. The transfer runbook separates imported-noise and seeded runs.

Preserve the original acceptance rules:

| Gate | Requirement |
|---|---|
| Initial noise | Maximum absolute difference <= 1e-6 |
| Schedule | Maximum absolute difference <= 1e-7 |
| Each recorded floating image/guided velocity | Relative L2 <= 0.001 |
| Existing primitive captures | Retain their existing pointwise and relative thresholds |
| Output contract | Correct RGB/grayscale layouts, normalization, crop, clamping and truncation |
| Graph execution | Finite outputs, intended backend assignment, 28 F32 K/V flash nodes |

Do not loosen thresholds just because the destination is ARM. If a gate
fails, locate the first divergent primitive/block and report the failed
platform configuration separately from passing ones.

PyTorch does not need to be installed on the ARM Windows machine or phone:
export reference fixtures on the existing x64 oracle system and compare
native outputs there.

### 3.4 Process memory measurement is currently Windows-specific

`tests\fashn_memory_profile.h`, lines 128-168, uses PSAPI and Windows page
accounting. On other systems it explicitly writes `process: null` and
`process_memory_unavailable`. It does not measure Android RSS/PSS.

Retain the portable logical counters for registered weights, actual buffers,
conversion scratch, graph workspaces, and modulation cache. Add an Android
process provider rather than interpreting null as zero.

Use the process's own readable proc entries for RSS, high-water mark,
thread count and mapping statistics; obtain PSS/private-dirty/swap details
from `smaps_rollup` when available. Use `dumpsys meminfo` as a complementary
device-side observation. Report inaccessible fields as unavailable. Do not
require root or assume arbitrary cross-process proc access is permitted.

Android native allocations still contribute to system memory pressure.
The Java heap limit is not the native engine's complete memory budget, and
`largeHeap` does not solve LMKD termination. Even foreground processes can
be killed. Do not add PSS to RSS or combine independently observed peaks.

### 3.5 Cancellation can be improved without first changing GGML

Current public sampling cancellation can wait for a complete forward;
historical Windows cancellation/shutdown observations were approximately
73-79 seconds. Android could differ substantially.

The pinned GGML already exports:

```cpp
ggml_backend_cpu_set_abort_callback(backend, callback, callback_data);
```

`ggml\src\ggml-cpu\ggml-cpu.c`, lines 3257-3283, checks the callback after a
node or fused node group and propagates `GGML_STATUS_ABORTED`. This is a
concrete first integration point, not a proposal to kill a computation
thread.

Proposed request-scoped use:

```cpp
struct RequestCancellation {
    std::atomic<bool> requested{false};
};

static bool should_abort(void* data) {
    return static_cast<RequestCancellation*>(data)
        ->requested.load(std::memory_order_relaxed);
}

// Within an exclusive, lifetime-managed CPU request scope:
ggml_backend_cpu_set_abort_callback(cpu_backend, should_abort, &cancellation);
// Execute; distinguish ABORTED from failure; synchronize before cleanup.
// Restore the previous callback/state before cancellation goes out of scope.
```

This is an integration sketch, not a complete RAII implementation. Handle
precomputation and denoising, callback restoration, errors, cache cleanup,
and a subsequent request. The largest individual node/fused group remains
non-preemptible by this hook; do not promise instantaneous cancellation.
Evaluate both GGML threading configurations before promoting the change.

### 3.6 Raw photograph preparation has additional dependencies

Prepared tensors are sufficient to prove the transformer. They are not
enough for a consumer application accepting arbitrary photos.

The existing optional native preparer already implements the required
resize/crop, DWPose, mask/parser, garment mode and normalization semantics.
Reuse that implementation instead of independently approximating it in
Kotlin.

Concrete adaptations:

| Existing surface | Android work |
|---|---|
| `examples\fashn-preprocess\CMakeLists.txt` | Import Android ORT/OpenCV artifacts rather than desktop include/lib layout |
| OpenCV 4.12 core/imgproc/imgcodecs | Pin compatible Android SDK/build and image codecs; validate pixel behavior |
| ORT 1.22.1 baseline | Provision compatible Android C/C++ artifacts; evaluate upgrades separately |
| Non-Windows OpenSSL dependency | Explicit Android crypto dependency or injected trusted hash-verification interface |
| `preprocess.cpp`, lines 428-462 | Preserve streaming SHA256 and read/error handling |
| `pipeline.cpp` artifact/consent checks | Preserve model identity and parser-license enforcement |
| Lazy parser / optional no-arena | Carry over, then measure Android session residency and teardown |

The NDK does not automatically supply a public desktop OpenSSL installation.
Do not link against private platform crypto libraries or simply remove the
hash checks to make the build pass.

A reasonable Android design is a small verifier interface injected into the
preparation pipeline: Windows keeps BCrypt, ordinary desktop builds keep
their current crypto implementation, and Android uses an explicitly
provisioned implementation or a Java `MessageDigest` service. If verification
moves to Java, native code must receive a trusted verification result tied
to the exact immutable file being opened; it must not accept an unchecked
caller-provided "verified" boolean. Preserve first-use parser revalidation.
Choose this abstraction versus a packaged crypto library based on measured
dependency cost and maintainability during implementation.

ORT supports Android and can accelerate the separate ONNX pose/parser
models. That does not automatically accelerate the GGML transformer.
Start with ORT CPU and compare prepared tensors before changing execution
providers or enabling FP16 relaxation.

**Licensing remains a separate gate.** The existing parser integration was
authorized for local research/evaluation only. Moving it into an APK does
not grant redistribution or commercial rights. Keep it out of public
packages until appropriately licensed or replaced. Parser-free flat-lay
mode is a useful first raw-input milestone, not a replacement for all worn
garment and masking behavior. Audit pose/model/dependency licenses too.

## 4. Resource expectations: use measurements, not phone guesses

These are the final **Windows x64 prepared-input** observations, not Android
requirements or forecasts:

| Policy | Sampling seconds | Peak working set GiB | Sampled private commit GiB |
|---|---:|---:|---:|
| Floating CPU, cardigan | 2371.130 | 2.0109 | 2.1402 |
| Floating OpenBLAS, cardigan, diagnostic | 1598.136 | 2.0281 | 4.2843 |
| Q8, cardigan, diagnostic | 1863.072 | 1.4002 | 1.5281 |
| Mixed, cardigan, diagnostic | 1898.820 | 1.5266 | 1.6532 |

The measured floating CPU run takes about 39.5 minutes. This is evidence of
a substantial workload, not a prediction that a phone will take exactly
that long. There is no defensible Android seconds/image figure yet.
These optimized historical measurements include the fused-GELU option;
the proposed initial Android baseline leaves it off until independently
qualified.

### Weight and working-buffer accounting

| Policy | Runtime-ready file bytes | Active denoising parameter bytes | Observed graph arena bytes |
|---|---:|---:|---:|
| BF16 core / F32 protected | 2,471,712,672 | 1,427,504,832 | 600,355,904 |
| Q8 / F32 protected | 1,808,160,672 | 763,952,832 | 564,180,032 |
| Mixed: 100 Q8 / 4 F32 | 1,942,919,072 | 898,711,232 | 564,180,032 |

The existing bounded precomputation retires 1,044,172,800 bytes of
conditioning/modulation parameters. The 39-pair cache is 32,148,480 bytes.
For the floating path, active parameters + that arena + that cache total
2,060,009,216 bytes, about 1.92 GiB. This is a useful accounting subtotal,
**not a complete Android process peak**.

The runtime-ready floating file is larger than the original roughly
1.94 GB checkpoint because protected weights are stored in their required
F32 runtime form. Ready files eliminate runtime conversion and duplicate
source residency; they do not mean every registered byte remains active
during sampling.

Use ready weights, no source mmap, lazy loading, and modulation precomputation
for the low-memory baseline. The public cache option rejects mmap/eager
loading. The POSIX mapping code in `src\core\util.cpp`, lines 239-288, maps
whole files at offset zero; this inspected path does not assume a 4 KiB
offset. That is not a completed audit of every dependency's page handling.

Avoid on-phone conversion. The old path could retain approximately 1.94 GB
of source mapping alongside destination storage and conversion scratch.
Small model files alone are not proof of lower runtime memory.

### Preparation and application overhead

Previous Windows preparer experiments measured approximately 661 MiB warm
with ORT CPU arenas disabled versus approximately 1,596 MiB otherwise;
the no-arena lifecycle peak was approximately 1,025.75 MiB. These are
separate process/lifecycle observations and must not simply be added to
the transformer peak.

For the first Android raw pipeline, prefer phase separation:

```text
Decode bounded inputs
  -> create preparation sessions
  -> produce four prepared images + crop/category
  -> destroy sessions and release raw/decode intermediates
  -> load/run FASHN
  -> save result
  -> retain or unload the engine according to measured memory pressure
```

Measure whether allocators actually release memory; object destruction is
not proof that RSS immediately falls. An optional separate inference
process can later isolate native crashes and guarantee reclamation when
it exits, but is not the default first implementation.

Start device qualification on a **12-16 GB RAM class phone/tablet** if
available. An 8 GB device may be viable; it needs measurement, not a blanket
promise. Do not initially claim support for 4-6 GB devices or 32-bit Android.
These are research priorities, not experimentally established minimum RAM.
OS pressure, other apps, GPU allocations and thermal behavior matter.

Provision several GiB of storage beyond the weight itself for preparation
models, temporary downloads, fixtures and updates. Compute required free
space from the actual selected manifest, keeping an existing verified model
while staging its replacement. Do not embed a 2.47 GB model in the base APK,
read it into one Java byte array, or allocate a model-sized direct buffer.

## 5. Proposed Android architecture

```text
Kotlin UI / lifecycle owner
        |
        +-- streaming model provisioning + immutable artifact manifest
        |
        +-- one worker: blocking JNI entry, no UI-thread inference
                  |
                  +-- optional native preparation, CPU first
                  |      OpenCV + ORT + permitted pose/parser artifacts
                  |
                  +-- existing FASHN C API / GGML CPU
                  |      ready BF16 weights + F32 compute
                  |      flash attention + bounded modulation cache
                  |
                  +-- bounded output + metadata, saved atomically
```

Do not transplant the Windows HTTP server, browser, base64 uploads and
512 MiB server image queue into the initial app. Direct JNI removes those
unnecessary copies and network/lifecycle surfaces.

### Native ownership contract

The initial app should permit one engine request at a time. The bridge
owns the engine and its request buffers until computation has ended.
Use an opaque handle registry with lifetime-managed entries, rather than
exposing a raw pointer as an unchecked `jlong`.

The existing C API already supplies `generate_try_on_with_callbacks` and
`free_sd_images`. Always free results with the matching native API; a Java
direct buffer must not outlive the native allocation backing it.

Proposed Kotlin surface:

```kotlin
// Proposed interface only. Implementations do not exist yet.
external fun createEngine(modelPath: String, threads: Int): Long
external fun runPrepared(
    engine: Long,
    requestId: Long,
    personRgb: ByteBuffer,
    garmentRgb: ByteBuffer,
    personPose: ByteBuffer,
    garmentPose: ByteBuffer,
    category: Int,
    crop: IntArray,
    seed: Long,
    outputPath: String
): String // structured result metadata; execution blocks the worker
external fun cancel(engine: Long, requestId: Long)
external fun progress(engine: Long, requestId: Long): IntArray
external fun closeEngine(engine: Long)
```

The implementation must validate direct-buffer capacity, shape/channel
counts, crop bounds, handle identity and request identity. Copy the small
prepared inputs into request-owned storage or otherwise hold their lifetime
explicitly. Do not pin a Java primitive array with a critical JNI access
for a minutes-long inference.

Prefer native callbacks updating atomic progress/cancellation state, with
Kotlin collecting it and updating the UI. Never share `JNIEnv*` across
threads or retain local JNI references after their scope ends. If callbacks
do enter Java from native-created threads, attach/detach correctly and own
global references explicitly. Translate native exceptions at the JNI
boundary; propagate failures instead of fabricating empty successes.

For bitmaps, validate format, stride, alpha/premultiplication and color
space. Do not reinterpret Android RGBA memory as tightly packed RGB or
grayscale. Begin with canonical lossless prepared PNG fixtures and compare
bytes before introducing camera JPEG, EXIF and color-conversion policies.

### Reusing the public C API

The relevant configuration can be expressed without a new model engine:

```cpp
// Proposed setup inside an owning Android bridge.
sd_ctx_params_t options;
sd_ctx_params_init(&options);
options.diffusion_model_path = verified_model_path.c_str();
options.backend = "CPU";
options.params_backend = "CPU";
options.n_threads = selected_threads;
options.wtype = SD_TYPE_BF16;
options.rng_type = CPU_RNG;
options.enable_mmap = false;
options.eager_load = false;
options.diffusion_flash_attn = true;
options.model_args =
    R"({"fashn_modulation_cache":true,"fashn_modulation_cache_mib":128,"fashn_fused_gelu":false})";

sd_ctx_t* context = new_sd_ctx(&options);
// Check context, establish RAII ownership, then initialize sd_try_on_params_t.
// Set the four prepared images, category/crop, 20 steps, CFG/shift and seed.
// Call generate_try_on_with_callbacks and pair output ownership with free_sd_images.
```

The owning bridge must keep required strings and buffers valid. Public
floating/backend restrictions remain in force. Do not pass a Q8 ready
file to the floating policy and mistake re-expansion for quantized inference.

### Thermal and lifecycle policy

First prototype: user-visible work, no promise of continued generation after
the app is backgrounded or the process dies. Persist request identity and
status; on restart mark interrupted work explicitly. The current runtime
does not implement resumable denoising checkpoints.

Use `PowerManager` thermal status listeners and, where available, thermal
headroom. Headroom is a dimensionless indicator, not a number of seconds;
the forecast argument supplies the look-ahead interval. Rate-limit headroom
queries, handle NaN/unavailable readings, and unregister listeners.

**Important existing invariant:** cached modulation values are tied to the
thread/arithmetic configuration (`fashn_vton_model.h`, lines 493-509).
Do not change the thread count halfway through a cached request in response
to thermal status. Initially stop/defer work or choose a lower thread count
for the next request, rebuilding cache state as necessary.

Do not disable OS thermal controls, pin all cores indiscriminately, or
assume a charging phone is thermally equivalent to an unplugged one.

For a later background-capable app, determine the legitimate foreground
service category, notification/permission requirements, start restrictions,
timeouts and store policy. A `shortService` is roughly a three-minute
mechanism, not a fit for the measured desktop workload. Neither WorkManager
nor a foreground service guarantees unlimited computation or immunity from
process death. Image-processing classification must be checked against the
actual application, not asserted merely because the model produces a PNG.

## 6. Sequential checkpoints

Use Android-specific IDs **A0-A12**, rather than changing the numbering of
the 38 completed historical checkpoints. Every checkpoint produces a
short report recording source/artifact hashes, commands, outcomes, failures,
and the next decision. A blocked checkpoint must remain blocked; a build
alone does not count as inference success.

### A0. Freeze inputs, inventory devices and define acceptance

**Work:** Record the pushed revision and submodule pins, actual ARM Windows
SoC/RAM/compiler, and target Android model/SoC/ABI/API/RAM/page size/storage.
Record visible CPU capabilities, Vulkan driver/extensions when available,
power state and thermal APIs. Do not infer a Qualcomm NPU from "ARM64."
Select existing photographic cases covering tops, bottoms and one-pieces;
use already prepared inputs to avoid initially requiring the parser.

Package a small hashed fixture set: primitive inputs/expected outputs,
prepared image bytes/crop/category, reference noise, schedules and selected
trajectory states. Large golden data stays outside Git and is copied only
when required.

**Gate:** Reproduce the fixture comparisons on the existing x64 baseline.
Record predeclared per-device memory, sustained-latency and cancellation
acceptance budgets before evaluating optimizations. Initially these can be
research budgets, but do not label the result "interactive" without an
explicit seconds-per-image target.

**Stop condition:** No actual Android device is available. Host preparation
can continue, but phone runtime feasibility remains unproven.

### A1. Native Windows ARM64 FASHN baseline

**Work:** Rebuild revision `55db93d6...` using the working native ARM64 Clang
setup. Archive architecture, toolchain and compilation flags. Run existing
contract, C API, math/GELU, sampling, modulation and attention primitives.
Use exported oracle inputs for conditional/null forward comparisons.

Run real-weight BF16-storage/F32-compute inference with flash and the
modulation cache. Establish a short run first, then one full 20-step
photographic trajectory. Compare imported-noise and public-RNG paths
separately; record all original thresholds.

Benchmark a small thread shortlist such as 2/4/6, plus a device-appropriate
choice, without assuming six is optimal because the SDXS smoke used six.
Record model/precompute/forward/sample/process timing and memory.

**Gate:** Native ARM64 executable, all required numerical gates, correct
full output, no implicit x64 execution. A visually plausible PNG alone is
not a parity result.

**If failing:** Bisect GELU, RNG, RoPE/norms, attention and matrix arithmetic.
Do not port an unexplained ARM discrepancy into the Android baseline.

### A2. Minimal Android NDK build and device primitives

**Work:** Establish an NDK/Ninja build and a small ADB deployment harness.
Recommended initial research minimum is API 29 for a simple modern device
floor, not because the model itself requires Android 10. Pin an NDK r28+
version; select current policy-compliant compile/target SDKs for the later
app independently of minSdk.

Proposed PowerShell configuration, from the repository root:

```powershell
$ndk = $env:ANDROID_NDK_HOME
if (-not $ndk) { throw "Set ANDROID_NDK_HOME to the selected pinned NDK" }

cmake -S . -B build-android-arm64-cpu -G Ninja `
  "-DCMAKE_TOOLCHAIN_FILE=$ndk\build\cmake\android.toolchain.cmake" `
  -DANDROID_ABI=arm64-v8a -DANDROID_PLATFORM=android-29 `
  -DANDROID_STL=c++_static -DCMAKE_BUILD_TYPE=Release `
  -DCMAKE_EXPORT_COMPILE_COMMANDS=ON `
  -DSD_BUILD_SHARED_LIBS=OFF -DSD_BUILD_SHARED_GGML_LIB=OFF `
  -DSD_BUILD_TESTS=ON -DSD_BUILD_EXAMPLES=ON `
  -DSD_SERVER_BUILD_FRONTEND=OFF -DSD_FASHN_PREPROCESS=OFF `
  -DSD_WEBP=OFF -DSD_WEBM=OFF -DSD_VULKAN=OFF `
  -DGGML_NATIVE=OFF -DGGML_CPU_ARM_ARCH=armv8-a `
  -DGGML_BACKEND_DL=OFF -DGGML_CPU_ALL_VARIANTS=OFF `
  -DGGML_OPENMP=OFF -DGGML_BLAS=OFF -DGGML_LLAMAFILE=OFF

cmake --build build-android-arm64-cpu --parallel 4 --target `
  sd-cli test-fashn-vton test-fashn-vton-c-api `
  test-fashn-vton-graph test-fashn-vton-sampling `
  test-fashn-gelu test-fashn-math test-fashn-modulation `
  test-fashn-vton-trajectory
```

The options above exist in the inspected tree, but the configuration has
not been executed. Resolve actual NDK errors surgically. Do not install
desktop OpenCV/ORT to fix a prepared-only build.

Use the NDK toolchain's default implementation; do not force
`ANDROID_USE_LEGACY_TOOLCHAIN_FILE=OFF`. Keep OpenMP disabled initially to
reduce runtime packaging variables; benchmark it later rather than assuming
it is inherently unsuitable.

Verify ELF64/AArch64, Android dependencies and segment alignment with the
NDK tools. Execute tests **on the device** through ADB. Host CTest cannot
run Android binaries directly without an explicit device/emulator harness.
Run the no-weight graph, sampling, GELU, math and contract tests first.

**Gate:** Native Android execution, no illegal instructions/missing symbols,
correct small fixture outputs, and no unreported fallback. Archive binary
hashes and compiler flags. A cross-compile without execution is partial.

### A3. Real-weight Android correctness and native memory telemetry

**Work:** Provision the floating ready file and small prepared fixture set.
Add Android process telemetry to the existing logical allocation events.
Add the diagnostic imported-noise trajectory input discussed above.

Run load-only first, then precomputation, a conditional forward, a null
forward, and a short trajectory. Record actual F32 operands, 28 flash nodes,
zero ready-file conversion, source mappings, cache bytes and retired weights.
Compare first/selected intermediate outputs against x64 and Windows ARM64.

Do not keep every full graph capture live on a phone: that changes the
memory problem. Add selected/streamed captures or run one failing layer at
a time if needed. Keep full trajectory recordings separate from
no-record latency/memory runs.

**Gate:** Correct small real-weight run within the A0 device budget, with
valid Android RSS/PSS observations rather than Windows-only null fields.
RNG and imported-noise outcomes must be reported separately.

**If failing:** Attribute memory to parameters, casts, graph, CPU workspace,
captures and allocator retention before proposing smaller precision. If
one efficient flash path is unsupported, do not quietly accept the large
manual-attention allocation as the mobile baseline.

### A4. Full prepared-input generation and practical feasibility decision

**Work:** Complete the reference 20-step workload on a selected device.
Start with one image, then tops/bottoms/one-pieces using the same prepared
bytes/noise and crop as the reference. Produce original/native comparison
images, difference maps and numerical reports on the host.

Separate cold load, modulation precompute, first forward, warm forward,
sampling, output encoding and process wall time. Record per-step timing,
RSS/PSS/high-water marks, thermal status, battery/charging state and thread
configuration. Shortlist threads with shorter runs before repeating full
trajectories.

Run at least three consecutive full jobs on the selected configuration to
expose heating and retention, plus a separately labeled cold run. Report
individual values and medians; do not market a meaningful p95 from three
measurements.

**Gate:** Original floating gates pass for every selected case; no OOM/LMKD
or native crash; complete outputs and a measured sustained resource budget.
Decide whether CPU-only is suitable for research, a slow offline product,
or neither. Passing numerical gates is not the same as acceptable latency.

**If too slow:** Keep the proven CPU path as the oracle and prioritize the
later acceleration experiments. Do not first build an elaborate UI around
an unacceptable latency assumption.

### A5. Minimal JNI prepared-input application and cancellation

**Work:** Add a small Android example module with a shared JNI wrapper over
the existing static core, PIC enabled, and no CLI/server/frontend build.
Initially select prepared fixtures; show progress and save a cropped result.
Use one worker and the ownership rules in section 5.

Integrate the existing CPU graph abort callback with request-scoped
lifetime and synchronization. Preserve cancellation semantics in both
precompute and forward execution. Treat abort as cancellation, not a
successful partially initialized result.

**Gate:** APK output matches the standalone native path. Exercise cancel
before work, during cache generation and during denoising; cancel followed
by a fresh job; close during work; duplicate/stale request IDs; rotation,
backgrounding and activity recreation. UI callbacks must remain responsive.
Report measured cancel latency and its longest-node lower bound.

Do not destroy an engine or free a callback/input buffer while a kernel is
still using it. Do not perform a blocking worker join on the UI thread.

### A6. Model provisioning and Android packaging

**Work:** Implement bounded streaming downloads/imports with byte count,
SHA256, temporary files, atomic publication and explicit errors. Preserve
a known-good model during updates; reject truncated/incompatible artifacts.
Resolve Storage Access Framework URIs through the platform, rather than
treating `content:` URIs as native filenames.

Keep large weights in app-controlled non-backed-up storage as appropriate.
Keep user photos/results private, provide explicit deletion/export, and
avoid placing biometric/person imagery or local paths in routine logs.
Verify offline inference after provisioning.

Use a compatible AGP version (8.5.1+ supports the documented uncompressed
16 KiB native-library packaging). NDK r28+ provides 16 KiB ELF alignment by
default. Older NDKs can be configured with the documented linker options;
r28 is a recommendation, not an absolute technical requirement.

Check every packaged `.so`, including libc++, ORT/OpenCV when later added,
for suitable LOAD alignment and RELRO. Check final APK alignment using
`zipalign -c -P 16 -v 4`. Run on 4 KiB and 16 KiB environments; query actual
page size instead of assuming 4096. A 16 KiB emulator is useful for
compatibility but not phone latency.

**Gate:** Clean install/update, interrupted download, low disk, corrupted
model, process death and restart behave explicitly; no model-sized Java
allocation; all shipped dependencies load on the supported page-size/ABI
matrix. Recheck store policy at release time.

### A7. Android native raw preparation

**Work:** Add Android ORT/OpenCV imported targets and resolve the hash
verification dependency without removing artifact checks. First implement
parser-free flat-lay CPU preparation, with sessions released before
transformer execution. Integrate native preparation directly, not by
spawning a Python interpreter.

Compare detector/pose outputs, resize/crop and the four prepared images
against the existing reference. Extend to the 12 preparation mode
combinations only where the parser license and artifacts are permitted.
Test codec/EXIF/color-space behavior explicitly before accepting arbitrary
camera images.

Add bounded image decode and pixel-count checks. Do not silently replace
the reference resize with an Android thumbnail/downsample path. Any
intentional preprocessing change needs matched reference fixtures.

**Gate:** Required prepared tensors/masks pass their existing gates;
raw-to-result output passes the generation baseline; joint application
peak is measured; no unwanted pose/parser session remains live during
denoising in the phase-separated policy. Repeat on a warmed process.

**If blocked by licensing:** Deliver a clearly labeled prepared-only or
parser-free app, not an allegedly complete commercial try-on pipeline.

### A8. ARM CPU optimization, one variable at a time

**Work:** Profile real FASHN shapes before tuning. Compare baseline NEON
against device-supported ISA variants; inspect which kernels actually ran.
Evaluate GGML repacking where applicable, fused strict GELU, thread count,
OpenMP versus the baseline, and an isolated ARM BLAS integration.

The historical Windows OpenBLAS startup allocation is not an Android fact.
Measure library-load-only cost, scratch, oversubscription and real
whole-pipeline performance for the Android library actually selected.
Do not equate a fast GEMM microbenchmark with a faster request.

Retain cache release, ready-file loading and flash invariants. Do not
reintroduce unconditional CFG B=2 batching or large condition caches:
those were not validated wins and can grow mobile workspaces.

**Gate:** Each accepted change passes the original gates and the A0 resource
budgets, with a repeatable sustained improvement. Publish rejected
configurations as well as winners. Never change cache thread settings
mid-request.

### A9. ARM quantization and mixed precision, diagnostic first

**Work:** Run the exact existing ready Q8 and mixed policies on ARM Android.
The mixed policy restores original F32 values for
`single_blocks.{12,11,8,14}.linear1.weight`; 100 matrices remain Q8.
Do not restore a matrix by merely dequantizing already damaged values.

Measure load, active weights, workspace, process memory, selected CPU
kernels, thermal behavior and complete image generation. Distinguish
weight quantization from activation conversion and matrix execution.
I8MM/dot-product availability is an opportunity, not proof of acceleration.

Compare same-policy ARM versus same-policy x64 to isolate port drift, and
each quantized policy versus floating to measure the original quantization
error. Extend the 104-matrix sensitivity investigation only if on-device
errors motivate it; reuse existing data rather than starting blindly.

**Gate:** Quantized file savings survive as actual measured process savings;
full comparison images and localized garment-boundary errors are reported.
Public promotion still requires the applicable original correctness/quality
policy. Existing Q8/mixed outputs failed floating gates and remain
experimental unless that changes; successful Android execution is not a
waiver.

### A10. Vulkan acceleration feasibility, then a guarded prototype

**Work:** Inventory the exact mobile driver and Vulkan features, allocation
limits and memory budget. Build backend primitives before the full model.
Audit actual kernel arithmetic and backend assignment for every relevant
operation, especially attention, GELU, normalization, RoPE and F32 GEMMs.

The public CPU guard must remain until a separate diagnostic path passes.
Evaluate the unfused strict GELU first, rather than attempting to send a
CPU `ggml_map_custom1` callback to a GPU.

Resolve modulation precompute's CPU-only restriction deliberately: for
example, CPU preparation with explicit small-vector transfer and verified
weight retirement. Measure host/device duplicate weights and transfer cost,
even on unified-memory phones. Do not retain the entire CPU model merely
to produce a few modulation vectors.

**Gate:** First primitive, then one real forward, then a full trajectory
passes the original gates. Count actual GPU/CPU assignments and transfer
bytes; an all-CPU fallback is not GPU success. Require measured sustained
latency and total-memory improvement before promotion.

**If backend changes are necessary:** Propose the smallest GGML change or
validated dependency update for explicit approval. This plan does not
authorize editing the pinned submodule.

### A11. NPU/ONNX/QNN feasibility as a separate track

**Work:** First evaluate ONNX pose/parser acceleration, since those models
already exist in ONNX. Compare ORT CPU against an appropriate supported EP;
inspect partitioning, fallback, latency, memory and preparation accuracy.

For the transformer, choose explicitly between a new GGML backend and a
separate ONNX/QNN forward implementation. Neither exists in this port.
An ONNX route needs model export, operator/shape coverage, external weight
data for large graphs where needed, fixed-shape handling, numerically
equivalent conditioning and attention, and host-controlled sampler/RNG.

Use original model parameters, not a GGUF file passed to QNN. GGUF Q8 block
formats and QNN quantization contracts are not interchangeable. Determine
the actual SDK/device's FP16 or quantized support, calibration requirements,
HTP tensor limits and fallback partitions through compilation and profiling.

**Gate:** Small operators, real forward and full trajectory must each have
measured coverage and numerical evidence. If large attention or mixed
precision breaks feasibility, record that stop point instead of advertising
NPU support based on a provider initialization log.

This checkpoint can be deferred indefinitely without invalidating a
working CPU Android port.

### A12. Qualification, reproducible reports and support declaration

**Work:** Publish a supported-device/build matrix with actual ABI, SoC,
Android/API/page size, model policy, preparation mode and backend.
Repeat complete quality cases and sustained runs under defined thermal
conditions. Record cold/warm latency, peak memory, cancellation, interrupted
jobs, battery observations and known garment limitations.

Add Android cross-build/host tests to CI where available, explicit `main`
coverage, and a separate physical-device execution job or reproducible
manual harness. Cross-build CI alone cannot certify quality or performance.
Keep model weights and user photographs out of normal source/CI artifacts.

Ship the minimal example and exact provisioning instructions; add symbols,
dependency/license manifests and reproducible report scripts. Audit every
prebuilt library's ABI/STL/page-size compatibility.

**Gate:** Support claims match completed measurements. Clearly distinguish
"prepared CPU supported," "raw preparation supported for these modes,"
"quantization experimental," and "GPU/NPU unvalidated" as appropriate.
No ARM64/Android blanket claim from one SDXS image.

## 7. Why Vulkan is not just a switch

The public FASHN initialization explicitly requires one CPU runtime backend
and CPU parameters (`src\stable-diffusion.cpp`, lines 1008-1016). Cached
modulation preparation also has a CPU/known-BLAS restriction. Removing those
checks is not implementation of GPU support.

There is nevertheless more existing Vulkan capability than a simple
"F32 attention is unsupported" statement would suggest:

| Source evidence | Consequence |
|---|---|
| `ggml-vulkan.cpp`, lines 18270-18313, accepts F32 K/V | It is incorrect to claim the backend universally rejects F32 attention inputs |
| Same block requires suitable head sizes and subgroup features unless cooperative-matrix-2 is used | Query the actual mobile device, not just Vulkan presence |
| `flash_attn_cm2.comp`, lines 40-86, returns FP16/scalar/vector values for F32 K/V decoding | F32 storage can still undergo lower-precision internal arithmetic |
| Scalar flash shader uses configurable `FLOAT_TYPE` and F32 decode support | Investigate a genuinely F32 path rather than assuming every shader rounds identically |
| Fused strict GELU is a CPU custom callback | Use elementary operations first or implement a separately validated device equivalent |

The explicit cooperative-matrix casts are particularly relevant because
the earlier CPU F16 K/V experiment failed parity. They do not prove that
every mobile Vulkan path fails; they prove that dtype acceptance is not
sufficient evidence of preserved precision.

Trace shader selection, compile-time types and internal accumulation for
the exact graph/device. Device numerical gates take precedence over nominal
F32 labels, FP16 TOPS, or a backend's `supports_op` response.

## 8. NPU expectations and alternatives

Android NNAPI is deprecated from Android 15. It should not be the default
long-term foundation merely because ORT can still use its execution
provider. ORT also warns that NNAPI FP16 relaxation can reduce accuracy
and that fallback placement affects performance.

ORT QNN supports appropriate Qualcomm Android/Windows hardware, but requires
the corresponding SDK/backend packaging and supported operations. Current
QNN documentation includes FP16-related options; "all NPUs require INT8"
is not an accurate universal statement. Conversely, an option appearing
in documentation does not guarantee this device/model supports it.

Our already rejected direct reduced-precision experiments make NPU quality
work consequential. It might require carefully partitioned precision,
attention tiling, or eventually model distillation/retraining. Those are
new engineering/research projects, not ABI changes.

If the requirement becomes a few seconds per image on ordinary phones, the
project should explicitly compare an accelerated exact-model path against
a smaller/distilled model or an opt-in remote service. A remote service
would change the offline/privacy requirement and must never become an
unannounced fallback.

## 9. Experiment artifact contract

Each device run should emit a machine-readable manifest resembling:

```json
{
  "checkpoint": "A4",
  "source_commit": "<full hash>",
  "ggml_commit": "<full hash>",
  "binary_sha256": "<hash>",
  "model_sha256": "<hash>",
  "fixture_manifest_sha256": "<hash>",
  "device": {
    "model": "<actual device>",
    "soc": "<actual SoC or unavailable>",
    "abi": "arm64-v8a",
    "api": null,
    "page_size_bytes": null
  },
  "policy": {
    "matrix_storage": "bf16",
    "compute": "f32",
    "backend": "CPU",
    "threads": null,
    "steps": 20,
    "cfg": 1.5,
    "shift": 1.5,
    "skip_cfg_last_n_steps": 1,
    "noise_source": "imported-reference",
    "mmap": false,
    "modulation_cache": true,
    "fused_gelu": false
  },
  "phases": [],
  "memory_samples": [],
  "thermal_samples": [],
  "comparisons": [],
  "outcome": "not-run"
}
```

Use null with an explanation for unavailable metrics, not invented zeroes.
Record seed as a string or another lossless representation if crossing JSON
implementations that cannot exactly represent all uint64 values. Record
logical sample count, crop/category, toolchain/build flags and error details
in the actual expanded schema.

A checkpoint report should link input/output hashes, individual timing
observations, numerical gates and visual comparisons. Disclose capture and
profiling overhead; compare performance in equivalent no-record modes.
Do not reinterpret historical Windows working-set/private-commit numbers
as Android RSS/PSS.

## 10. Recommended execution decision

**Authorize A0-A4 first:** device inventory, native Windows ARM64 FASHN,
Android primitives, real-weight correctness, and complete prepared
generation. Those checkpoints answer whether the exact model is practical
enough to justify the application work.

If they pass, A5-A7 produce a usable prepared-input app and then the
permitted raw-photo pipeline. A8 improves measured CPU behavior. A9-A11
are separately gated acceleration/precision work, not prerequisites for
calling the CPU port functional. A12 closes the supported-device release.

The existing inference engine can be reused. The missing work is Android
build/deployment, platform parity, measured resource management, JNI and
lifecycle ownership, optional preparation dependencies, and genuinely
validated acceleration. There is no need to start by rewriting FASHN in
Java or replacing the working native sampler.

## 11. Primary references

The conclusions above combine inspected pinned source with these official
platform documents. Platform/store policies and provider capabilities may
change; pin dependencies and recheck requirements during implementation.

| Topic | Reference |
|---|---|
| NDK CMake toolchain, ABI and platform | https://developer.android.com/ndk/guides/cmake |
| C++ runtime, exceptions and RTTI | https://developer.android.com/ndk/guides/cpp-support |
| JNI thread/reference/lifetime guidance | https://developer.android.com/ndk/guides/jni-tips |
| 16 KiB ELF/APK alignment and page-size compatibility | https://developer.android.com/guide/practices/page-sizes |
| Android memory overview | https://developer.android.com/topic/performance/memory-overview |
| Low-memory termination, including foreground processes | https://developer.android.com/topic/performance/vitals/lmk |
| Thermal status/headroom | https://developer.android.com/games/optimize/adpf/thermal |
| Foreground service types and restrictions | https://developer.android.com/develop/background-work/services/fgs/service-types |
| ORT Android build | https://onnxruntime.ai/docs/build/android.html |
| ORT NNAPI options and precision tradeoffs | https://onnxruntime.ai/docs/execution-providers/NNAPI-ExecutionProvider.html |
| NNAPI deprecation | https://developer.android.com/ndk/guides/neuralnetworks |
| ORT QNN capabilities and packaging | https://onnxruntime.ai/docs/execution-providers/QNN-ExecutionProvider.html |

Representative immutable source references:

- [FASHN public CPU/precision restrictions](https://github.com/gargpratyush/fashn-vton-stable-diffusion-port/blob/55db93d6f2536f12428e170acbafd0cca3182c59/src/stable-diffusion.cpp#L994-L1037)
- [FASHN graph precision/operation checks](https://github.com/gargpratyush/fashn-vton-stable-diffusion-port/blob/55db93d6f2536f12428e170acbafd0cca3182c59/src/model/diffusion/fashn_vton_model.h#L493-L563)
- [Portable fused GELU](https://github.com/gargpratyush/fashn-vton-stable-diffusion-port/blob/55db93d6f2536f12428e170acbafd0cca3182c59/src/core/strict_gelu.h)
- [Native preparation build dependencies](https://github.com/gargpratyush/fashn-vton-stable-diffusion-port/blob/55db93d6f2536f12428e170acbafd0cca3182c59/examples/fashn-preprocess/CMakeLists.txt)
- [Current platform memory accounting](https://github.com/gargpratyush/fashn-vton-stable-diffusion-port/blob/55db93d6f2536f12428e170acbafd0cca3182c59/tests/fashn_memory_profile.h#L128-L168)
- [GGML Android CPU variants](https://github.com/leejet/ggml/blob/e20c3a14aa70ee84ca58499814206dd08d8026bc/src/CMakeLists.txt#L403-L424)
- [GGML CPU abort boundary](https://github.com/leejet/ggml/blob/e20c3a14aa70ee84ca58499814206dd08d8026bc/src/ggml-cpu/ggml-cpu.c#L3257-L3283)
- [Vulkan F32 attention input support](https://github.com/leejet/ggml/blob/e20c3a14aa70ee84ca58499814206dd08d8026bc/src/ggml-vulkan/ggml-vulkan.cpp#L18270-L18313)
- [Cooperative-matrix attention decode precision](https://github.com/leejet/ggml/blob/e20c3a14aa70ee84ca58499814206dd08d8026bc/src/ggml-vulkan/vulkan-shaders/flash_attn_cm2.comp#L40-L86)
