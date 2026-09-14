# Moving FASHN to Windows ARM64, then Android

This is the operational handoff for the host preparation following
[the Android plan](fashn_android_plan.md). The first destination is the
Windows ARM64 machine that already ran the native SDXS smoke.

**Later result:** the separate Q4_K bundle/source `e86c564` completed a
[Samsung S23 cardigan run](../reports/android-s23-q4k/README.md).
Strict numerical parity failed. The original BF16 bundle pins and handoff
instructions below remain unchanged; this is not evidence that every original
acceptance checkpoint or a newer checkout passed.

**The Git repository alone is not enough to run an image.** It contains
source and reports, not model weights or the complete numerical reference
fixtures. Use a recursive source checkout **plus the private transfer
bundle** described here. Do not copy the entire historical experiment
directory or the x64 build/environment.

## 1. What is ready and what still needs the other machine

| Checkpoint | Work possible on this x64 host | Destination work |
|---|---|---|
| A0 | Prepare hashed ready weights, four input cases, two/full-step oracle trajectories and transfer tools | Record ARM/phone hardware and resource budgets |
| A1 | Run x64 controls and produce comparison material | Build/run native Windows ARM64 FASHN, not just SDXS |
| A2 | Cross-build Android ARM64 CLI and diagnostic executables with NDK r28c | Execute them on Android; no physical device was connected here |
| A3 | Add strict diagnostic imported-noise input and validate on x64 | Android real-weight parity and Android process-memory telemetry |
| A4 | Supply an existing complete 20-step photographic oracle | Complete ARM/Android trajectories and sustained resource observations |
| A5 onward | Design is in the Android plan | App/JNI/lifecycle, raw preparation and acceleration remain separate work |

The host's measured outcomes are recorded in
[the host preparation report](../reports/android-host-preparation.md).
Android cross-compilation is not Android runtime acceptance. An x64 Android
emulator would not establish native ARM64 FASHN performance.

## 2. Transfer these two things

### A. Source checkout

Repository:

```text
https://github.com/gargpratyush/fashn-vton-stable-diffusion-port
```

This handoff is pinned to:

```text
main repository: 55db93d6f2536f12428e170acbafd0cca3182c59
ggml submodule:  e20c3a14aa70ee84ca58499814206dd08d8026bc
```

At bundle creation, additional handoff changes were **uncommitted/unpushed**,
including the diagnostic noise option and these scripts/docs. The bundle carries an
exact `source-overlay` so you do not accidentally use the old executable
with the new instructions. The pinned baseline commit does not contain those
additions. Later repository revisions include this tooling and newer Q4/Q5
diagnostics, but are not automatically equivalent to the frozen bundle overlay.
For the original transfer acceptance, use the pinned baseline plus the
bundle's overlay exactly; do not apply it over a newer working tree or weaken
the wrapper's identity checks. The already-transferred bundle is unchanged.

### B. `fashn-arm-transfer-v1` folder

| Bundle path | Purpose | Needed on Windows ARM64? |
|---|---|---|
| `bundle.json` | Exact byte counts, SHA256 values, source pins and roles | Yes |
| `weights\model-bf16.gguf` | Runtime-ready BF16-core/F32-protected model | Yes |
| `cases\black-shirt\prepared` | Four prepared PNGs and a relative-path manifest | Yes |
| `cases\black-shirt\conditions.safetensors` | The same inputs as normalized float tensors for diagnostics | Yes for trajectory tests |
| `cases\cardigan`, `cases\bottoms`, `cases\dress` | Additional prepared photographic cases | Later coverage |
| `references\black-shirt-2step` | Complete short PyTorch F32/MATH oracle | First numerical comparison |
| `references\black-shirt-20step` | Complete full PyTorch F32/MATH trajectory | Full numerical comparison |
| `references\cardigan-upstream`, `references\bottoms-upstream` | Historical original-upstream 20-step PNGs and metadata | Supplemental visual comparison, not complete float trajectories |
| `source-overlay` | Exact changed source/docs/scripts to apply to the pinned clone | Yes |
| `android\bin` | Android ELF executables already cross-built here, when included | Not Windows programs; save for Android |
| `NOTICE.md` | Image/research restrictions | Yes |
| `host-evidence`, when included | Host logs, comparison images and measurements | Useful for interpreting the handoff |

The model alone is **2,471,712,672 bytes**, about 2.30 GiB. The full bundle's
exact size is recorded in `bundle.json`; numerical fixtures and Android
executables add to that. Leave additional space for build outputs and
results. The recorded 20-step diagnostic trajectory writes about 245 MB
before logs/comparison images.

Model SHA256:

```text
e1f2d13d441f0e4631cf4cf8e1837cd8998e4916b78dfc0ec62c5fe879d087b1
```

This is **not** the original Hugging Face `model.safetensors` hash.
It is the verified runtime-policy export, preserving BF16 core weights and
F32 protected parameters. Using it avoids on-device conversion and the old
duplicate-source-memory problem.

### Do not transfer these for the first milestone

No x64 `.exe`/DLL files, Visual Studio installation, Python virtualenv,
OpenBLAS DLL, OpenCV SDK, ORT SDK, pose/parser models, browser dependencies,
server frontend, raw-photo parser setup, or full sensitivity-study directory
is required for prepared inference.

The source submodules still need to be present for a reproducible build.
Use the recursive clone; a GitHub ZIP download normally omits submodule
contents. For an offline destination, prepare a recursive source checkout
in advance rather than assuming the data bundle includes all source.

The prepared research images can be used without executing the parser.
That does not remove image/derived-data licensing restrictions or authorize
shipping parser weights in a commercial APK. Keep this bundle private.

## 3. Copying and integrity

Prefer copying the folder directly using your normal file-transfer tool.
For a mounted destination:

```powershell
# Run from the workspace containing the bundle; replace E: with your destination.
robocopy .\fashn-arm-transfer-v1 E:\fashn-arm-transfer-v1 /E /Z /R:2 /W:2
if ($LASTEXITCODE -ge 8) { throw "Transfer failed" }
```

Use NTFS/exFAT or another filesystem supporting large files. Do not use
PowerShell `Compress-Archive` for this model: its documented per-file size
limitation is unsuitable for the >2 GiB weight. If one archive is needed,
use a large-file-capable archiver, for example Windows `tar.exe`:

```powershell
tar -cf fashn-arm-transfer-v1.tar fashn-arm-transfer-v1
# On the destination, from the intended parent folder:
tar -xf fashn-arm-transfer-v1.tar
```

Do not put weights or the transfer archive into ordinary Git.

The verifier needs only Python 3.11+ and its standard library. It does not
need PyTorch, NumPy, Pillow or safetensors:

```powershell
# Run from the folder containing the bundle.
python .\fashn-arm-transfer-v1\source-overlay\scripts\package_fashn_portability.py `
  verify .\fashn-arm-transfer-v1
if ($LASTEXITCODE -ne 0) { throw "Bundle verification failed" }
```

For a machine without Python, basic file integrity can also be checked with
PowerShell, using the trusted manifest from this transfer:

```powershell
$bundle = (Resolve-Path .\fashn-arm-transfer-v1).Path
$manifest = Get-Content "$bundle\bundle.json" -Raw | ConvertFrom-Json
foreach ($entry in $manifest.files) {
    if ($entry.path -match '(^/|\\|:|(^|/)\.\.(/|$))') { throw "Unsafe manifest path" }
    $file = Join-Path $bundle ($entry.path.Replace('/', '\'))
    if (!(Test-Path -LiteralPath $file -PathType Leaf)) { throw "Missing: $file" }
    if ((Get-Item -LiteralPath $file).Length -ne $entry.bytes) { throw "Wrong size: $file" }
    if ((Get-FileHash -LiteralPath $file -Algorithm SHA256).Hash -ne $entry.sha256) {
        throw "Wrong hash: $file"
    }
}
```

The Python verifier additionally checks the pinned model identity, required
bundle entries, duplicate paths and aggregate byte count.

## 4. Set up the Windows ARM64 source tree

Recommended: a fresh clone, leaving the successful SDXS checkout/build alone.
From the same workspace that contains the transferred bundle:

```powershell
git clone --recurse-submodules `
  https://github.com/gargpratyush/fashn-vton-stable-diffusion-port.git `
  fashn-vton-stable-diffusion-port
if ($LASTEXITCODE -ne 0) { throw "Clone failed" }
Set-Location .\fashn-vton-stable-diffusion-port
git checkout --detach 55db93d6f2536f12428e170acbafd0cca3182c59
if ($LASTEXITCODE -ne 0) { throw "Checkout failed" }
git submodule update --init --recursive
if ($LASTEXITCODE -ne 0) { throw "Submodule setup failed" }

$bundle = (Resolve-Path ..\fashn-arm-transfer-v1).Path
if (git status --porcelain) { throw "Use a clean source checkout before applying the overlay" }
robocopy "$bundle\source-overlay" . /E /R:2 /W:2
if ($LASTEXITCODE -ge 8) { throw "Source overlay copy failed" }
git status --short
```

The resulting worktree is intentionally dirty with the listed handoff
files. Do not run `git reset --hard` to "clean it up": that would remove
part of this handoff. Do not merge the overlay over unrelated local edits.

The wrapper intentionally checks this exact base + overlay. If these
changes are later committed under another hash, either reproduce the
pinned recipe above or regenerate the bundle/source pin for that revision.
Do not silently change the revision and assume recorded hashes still apply.

## 5. Build natively on Windows ARM64

Use the **same native ARM64 developer shell and Clang/SDK installation**
that successfully built the SDXS smoke. No Android NDK is needed for this
Windows build.

If that working setup uses Ninja and the Clang driver, this is the proposed
fresh build configuration:

```powershell
clang --version
cmake -S . -B build-arm64-fashn -G Ninja `
  -DCMAKE_C_COMPILER=clang -DCMAKE_CXX_COMPILER=clang++ `
  -DCMAKE_C_COMPILER_TARGET=aarch64-pc-windows-msvc `
  -DCMAKE_CXX_COMPILER_TARGET=aarch64-pc-windows-msvc `
  -DCMAKE_BUILD_TYPE=Release `
  -DSD_BUILD_TESTS=ON -DSD_BUILD_EXAMPLES=ON `
  -DSD_FASHN_PREPROCESS=OFF -DSD_SERVER_BUILD_FRONTEND=OFF `
  -DSD_WEBP=OFF -DSD_WEBM=OFF `
  -DSD_BUILD_SHARED_LIBS=OFF -DSD_BUILD_SHARED_GGML_LIB=OFF `
  -DGGML_NATIVE=OFF -DGGML_CPU_ARM_ARCH=armv8-a `
  -DGGML_OPENMP=OFF -DGGML_BLAS=OFF -DGGML_LLAMAFILE=OFF
if ($LASTEXITCODE -ne 0) { throw "ARM64 configuration failed" }

cmake --build build-arm64-fashn --parallel 4 --target `
  sd-cli test-fashn-vton test-fashn-vton-c-api `
  test-fashn-vton-graph test-fashn-vton-sampling test-fashn-vton-precision `
  test-fashn-gelu test-fashn-math test-fashn-modulation test-fashn-vton-trajectory
if ($LASTEXITCODE -ne 0) { throw "ARM64 build failed" }
```

If your working smoke used `clang-cl` or a Visual Studio generator, preserve
that known-working ARM64 toolchain setup and apply the feature switches
above. Do not mix generators in an existing build directory. These
Windows ARM64 build commands cannot be certified by this x64-only host.

Missing `kernel32.lib`, ARM64 CRT libraries or linker errors indicate SDK/
developer-shell setup, not evidence that FASHN requires Python. Do not
solve them by copying x64 runtime libraries.

### Confirm PE architecture before running tests

For a Ninja build, executables should be under `build-arm64-fashn\bin`.
A multi-configuration generator may add `Release`.

```powershell
$exe = (Resolve-Path .\build-arm64-fashn\bin\sd-cli.exe).Path
$reader = [IO.BinaryReader]::new([IO.File]::OpenRead($exe))
try {
    $reader.BaseStream.Position = 0x3c
    $offset = $reader.ReadUInt32()
    $reader.BaseStream.Position = $offset
    if ($reader.ReadUInt32() -ne 0x00004550) { throw "Not PE" }
    $machine = $reader.ReadUInt16()
    if ($machine -ne 0xaa64) { throw ("Not native ARM64: 0x{0:X4}" -f $machine) }
    "Native ARM64 PE: 0xAA64"
} finally {
    $reader.Dispose()
}
```

The Python run wrapper independently enforces the executable architecture.
Python itself may be native ARM64 or x64-emulated: it is only orchestration.
The inference executable must be ARM64.

### Run the small native tests first

```powershell
ctest --test-dir build-arm64-fashn -C Release --output-on-failure `
  -R '^(fashn-vton-contract|fashn-vton-c-api|fashn-vton-primitives|fashn-vton-sampling|fashn-vton-precision|fashn-modulation|fashn-gelu|fashn-math)$'
if ($LASTEXITCODE -ne 0) { throw "Stop and investigate primitive failure" }
```

Do not call an unfiltered CTest after building only the listed targets:
other registered tests may refer to executables not built yet.

## 6. First actual image: two-step public prepared inference

Start with `black-shirt`, sample count one, six threads, BF16 storage/F32
compute, flash attention, cache enabled, fused GELU disabled.

Using the standard-library wrapper:

```powershell
python scripts\run_fashn_portability.py `
  --bundle ..\fashn-arm-transfer-v1 `
  --bin-dir .\build-arm64-fashn\bin `
  --machine arm64 --steps 2 --threads 6 --public-cli `
  --output .\results-arm64\public-2step
if ($LASTEXITCODE -ne 0) { throw "Public prepared inference failed" }
```

The wrapper checks all bundle hashes and the source overlay before it runs.
Its wall time excludes this hash preflight but includes the native process.
It saves `run.json`, `run.log`, and `sample.png`.

No Python is needed if you invoke the native CLI directly instead:

```powershell
$bundle = (Resolve-Path ..\fashn-arm-transfer-v1).Path
$modelArgs = '{"fashn_modulation_cache":true,"fashn_modulation_cache_mib":128,"fashn_fused_gelu":false}'
# PowerShell 7 is recommended for predictable native argument quoting.
.\build-arm64-fashn\bin\sd-cli.exe --mode try_on `
  --diffusion-model "$bundle\weights\model-bf16.gguf" `
  --try-on-inputs "$bundle\cases\black-shirt\prepared\manifest.json" `
  --type bf16 --diffusion-fa --rng cpu -t 6 `
  --steps 2 --cfg-scale 1.5 --flow-shift 1.5 --skip-cfg-last-n-steps 1 `
  --seed 42 --model-args $modelArgs -o .\arm64-smoke.png
if ($LASTEXITCODE -ne 0) { throw "Inference failed" }
```

Do not add `--mmap` or `--eager-load`; they conflict with the memory-saving
modulation cache. Do not use a Q8 file with `--type bf16` and call it a Q8
run. Do not replace the prepared images with raw photographs.

The two-step result is a correctness/smoke image, **not a quality showcase**.
Compare with `references\black-shirt-2step\expected.png`, not the 20-step
output.

## 7. Numerical ARM64 acceptance: two separate runs

### Same-seed CPU RNG path

```powershell
python scripts\run_fashn_portability.py `
  --bundle ..\fashn-arm-transfer-v1 --bin-dir .\build-arm64-fashn\bin `
  --machine arm64 --steps 2 --threads 6 --noise cpu `
  --output .\results-arm64\trajectory-2step-cpu
if ($LASTEXITCODE -ne 0) { throw "Trajectory failed" }
```

### Identical imported-noise path

```powershell
python scripts\run_fashn_portability.py `
  --bundle ..\fashn-arm-transfer-v1 --bin-dir .\build-arm64-fashn\bin `
  --machine arm64 --steps 2 --threads 6 --noise reference `
  --output .\results-arm64\trajectory-2step-imported
if ($LASTEXITCODE -ne 0) { throw "Imported-noise trajectory failed" }
```

Both produce `trajectory\initial.safetensors`, per-step states, a manifest
and `memory-profile.jsonl`. The diagnostic `--initial-noise` option requires
exactly one finite F32 `noise` tensor of NCHW shape `[1,3,864,576]` or CHW
shape `[3,864,576]`. It is not a new public CLI option.

The reference noise is common to the same seed/batch/shape, not specific to
the shirt. However, the bundled complete numerical trajectories are for
the black-shirt case only. Do not compare another garment's trajectory to
the black-shirt oracle.

### Compare on the existing x64 host

Copy `results-arm64` back to the x64 machine. It already has the comparison
environment; there is no reason to install ARM PyTorch just to compare:

```powershell
# Run from the source repository, using its existing reference Python environment.
# $python points to the original x64 reference environment's python.exe.
& $python scripts\compare_fashn_trajectories.py `
  --reference ..\fashn-arm-transfer-v1\references\black-shirt-2step `
  --native .\results-arm64\trajectory-2step-cpu\trajectory `
  --output .\results-arm64\comparison-2step-cpu
if ($LASTEXITCODE -ne 0) { throw "Same-seed numerical gate failed" }

& $python scripts\compare_fashn_trajectories.py `
  --reference ..\fashn-arm-transfer-v1\references\black-shirt-2step `
  --native .\results-arm64\trajectory-2step-imported\trajectory `
  --output .\results-arm64\comparison-2step-imported
if ($LASTEXITCODE -ne 0) { throw "Imported-noise numerical gate failed" }
```

The comparator needs NumPy, Pillow and safetensors but does not need a model
weight or a running PyTorch inference. Its imported reference helper uses
standard-library functionality at module load. If running comparisons in a
new environment, install these packages there; native generation itself
does not need them.

Acceptance stays unchanged: initial noise max absolute error <= 1e-6,
schedule max absolute error <= 1e-7, and relative L2 <= 0.001 for **every**
updated image and guided velocity. Generated comparison directories contain
native/reference images and an amplified difference image.

Interpret failures separately: if imported-noise passes but same-seed fails,
investigate RNG/libm first. If both fail, isolate transformer/sampler
arithmetic. Never relax the limits just to mark ARM supported.

## 8. Full run, additional photographs and measurements

Only after the two-step gates pass:

```powershell
python scripts\run_fashn_portability.py `
  --bundle ..\fashn-arm-transfer-v1 --bin-dir .\build-arm64-fashn\bin `
  --machine arm64 --steps 20 --threads 6 --noise cpu `
  --output .\results-arm64\trajectory-20step-cpu
if ($LASTEXITCODE -ne 0) { throw "Full trajectory failed" }
```

Compare that `trajectory` folder to `references\black-shirt-20step`.
Repeat with `--noise reference` if needed to isolate RNG drift. Do not
assume the 33.4-second SDXS result predicts this run's duration.

For public generated PNGs:

```powershell
python scripts\run_fashn_portability.py `
  --bundle ..\fashn-arm-transfer-v1 --bin-dir .\build-arm64-fashn\bin `
  --machine arm64 --steps 20 --threads 6 --public-cli --case cardigan `
  --output .\results-arm64\cardigan-20step
```

Repeat with `--case bottoms` and `--case dress`. The cardigan/bottoms
upstream reference PNGs are matched 20-step, seed-42 visual controls.
The dress has prepared inputs but no bundled independent full float oracle;
label that gap rather than asserting numerical coverage.

Record cold and warmed runs separately. Try 2/4/6 threads as distinct new
requests, then select a configuration for consecutive full runs. Do not
change threads midway through a cached request. Keep profiling/trajectory
recording overhead separate from public CLI latency.

`run.json` reporting `completed` means the native process exited zero; it
does not mean numerical or garment-quality acceptance passed.

## 9. What to return from Windows ARM64

Return the complete `results-arm64` directory, including:

- `run.json`, logs, generated PNGs, and trajectory manifests/states.
- Memory-profile JSONL files and numerical comparison reports, if run there.
- `ctest` output, CMake cache, compiler version and the PE architecture result.
- Windows version, SoC/CPU description, RAM, power mode and thread settings.

Do not return the weight file: it is already hash-identified and available
on the x64 host. Do not return credentials or a complete user-profile dump.
Keep each run in a new directory; the wrapper intentionally rejects an
existing output directory.

If interrupted, retain the partial files and mark the run interrupted.
The runtime does not resume from a saved denoising step automatically.

## 10. Android: use the same data, different executables

The optional `android\bin` folder contains **Android**, not Windows,
executables. This host produced them with:

```text
NDK: 28.2.13676358 (r28c)
Compiler: NDK Clang 19.0.1
ABI: arm64-v8a / ELF64 AArch64
Minimum native API: 29
ISA: armv8-a
Runtime: static C++/GGML/core; Android libc/libm/libdl dependencies
Backends: CPU only; OpenMP/BLAS/LLAMAFILE disabled
ELF LOAD alignment: 0x4000 (16 KiB); GNU_RELRO present
```

The transfer binaries may have debug sections removed with the NDK's
`llvm-strip`; unstripped build outputs stay on the development host.
This is not an APK/JNI package, and ELF alignment is not final APK
alignment. No Android inference result is claimed.

### Device inventory before pushing the model

On the computer with ADB and a physically connected, authorized device:

```powershell
adb devices -l
adb shell getprop ro.product.model
adb shell getprop ro.product.cpu.abilist
adb shell getprop ro.build.version.sdk
adb shell getconf PAGE_SIZE
adb shell cat /proc/meminfo
adb shell df -h /data
```

Select the intended serial with `adb -s SERIAL` if more than one device is
present. The remote paths below are Android POSIX paths; local paths remain
Windows paths.

### Small tests first

```powershell
$bundle = (Resolve-Path ..\fashn-arm-transfer-v1).Path
adb shell "mkdir -p /data/local/tmp/fashn/bin"
adb push "$bundle\android\bin\." /data/local/tmp/fashn/bin/
adb shell "chmod 700 /data/local/tmp/fashn/bin/*"
adb shell "/data/local/tmp/fashn/bin/test-fashn-vton"
adb shell "/data/local/tmp/fashn/bin/test-fashn-vton-c-api"
adb shell "/data/local/tmp/fashn/bin/test-fashn-vton-precision"
adb shell "/data/local/tmp/fashn/bin/test-fashn-gelu"
adb shell "/data/local/tmp/fashn/bin/test-fashn-math"
adb shell "/data/local/tmp/fashn/bin/test-fashn-modulation"
adb shell "/data/local/tmp/fashn/bin/test-fashn-vton-sampling"
adb shell "/data/local/tmp/fashn/bin/test-fashn-vton-graph"
```

Check every exit status and save logs; stop at failure rather than continuing
to the model run. These native tests exercise different surfaces from the
Windows SDXS image.

### Provision just the first case

```powershell
adb shell "mkdir -p /data/local/tmp/fashn/weights /data/local/tmp/fashn/case"
adb push "$bundle\weights\model-bf16.gguf" /data/local/tmp/fashn/weights/model.gguf
adb push "$bundle\cases\black-shirt\." /data/local/tmp/fashn/case/
adb push "$bundle\references\black-shirt-2step\initial.safetensors" /data/local/tmp/fashn/noise.safetensors
adb shell "sha256sum /data/local/tmp/fashn/weights/model.gguf"
```

Match the device hash against the pinned model hash above. The phone does
not need every golden trajectory: retain references on the host and pull
native results back for comparison. Public inference only needs the model
and five prepared-input files.

### First diagnostic Android trajectory

Run only after native primitives pass and the device has an appropriate
resource budget. Start with two threads and a new output path:

```powershell
adb shell "/data/local/tmp/fashn/bin/test-fashn-vton-trajectory /data/local/tmp/fashn/weights/model.gguf /data/local/tmp/fashn/case/conditions.safetensors /data/local/tmp/fashn/result-2step-imported 2 1.5 1.5 1 42 1 2 --matrix-type bf16 --upcast-matrices --mmap off --load-threads 2 --precompute-modulations --initial-noise /data/local/tmp/fashn/noise.safetensors"
```

Pull the output directory:

```powershell
adb pull /data/local/tmp/fashn/result-2step-imported .\results-android\
```

Use the same host comparison script against `black-shirt-2step`. Repeat
without `--initial-noise` into a fresh directory for the CPU RNG path.
For full acceptance use 20 steps and the 20-step reference, not two-step
quality impressions.

The current native memory profiler has **Windows process accounting only**;
its Android process fields are unavailable. Logical weight/workspace
counters are not total Android RSS/PSS. Complete A3's Android telemetry
work before claiming the mobile memory gate passed. Device `/proc` and
`dumpsys` observations can help diagnose runs but are not a substitute for
the missing integrated measurement.

Do not install ORT/OpenCV or transfer parser weights to fix transformer
failures. Those dependencies belong to the later raw-preparation checkpoint.
Do not enable Vulkan or remove the CPU guard for this baseline.

## 11. Rebuilding Android binaries on an x64 Windows development host

From the overlaid source checkout, with CMake/Ninja on PATH and the pinned
NDK already installed:

```powershell
$ndk = Join-Path $env:LOCALAPPDATA 'Android\Sdk\ndk\28.2.13676358'
cmake -S . -B build-android-arm64-migration -G Ninja `
  "-DCMAKE_TOOLCHAIN_FILE=$ndk\build\cmake\android.toolchain.cmake" `
  -DANDROID_ABI=arm64-v8a -DANDROID_PLATFORM=android-29 `
  -DANDROID_STL=c++_static -DCMAKE_BUILD_TYPE=Release `
  -DCMAKE_EXPORT_COMPILE_COMMANDS=ON `
  -DSD_BUILD_TESTS=ON -DSD_BUILD_EXAMPLES=ON `
  -DSD_BUILD_SHARED_LIBS=OFF -DSD_BUILD_SHARED_GGML_LIB=OFF `
  -DSD_FASHN_PREPROCESS=OFF -DSD_SERVER_BUILD_FRONTEND=OFF `
  -DSD_WEBP=OFF -DSD_WEBM=OFF `
  -DGGML_NATIVE=OFF -DGGML_CPU_ARM_ARCH=armv8-a `
  -DGGML_OPENMP=OFF -DGGML_BLAS=OFF -DGGML_LLAMAFILE=OFF
if ($LASTEXITCODE -ne 0) { throw "Android configuration failed" }
cmake --build build-android-arm64-migration --parallel 4 --target `
  sd-cli test-fashn-vton test-fashn-vton-c-api test-fashn-vton-graph `
  test-fashn-vton-sampling test-fashn-vton-precision test-fashn-gelu `
  test-fashn-math test-fashn-modulation test-fashn-vton-trajectory
if ($LASTEXITCODE -ne 0) { throw "Android build failed" }
```

NDK host-tool availability on Windows ARM64 is a separate concern: use the
already cross-built ELF files or the x64 development host first. Do not
assume the Windows ARM64 Clang installation can replace the Android NDK.

## 12. Regenerating the private bundle on the original development host

The packager consumes the existing experiment directory; it is not a
downloader or a model converter:

```powershell
python scripts\package_fashn_portability.py create `
  --repo . --experiment ..\fashn-vton-reference `
  --output ..\fashn-arm-transfer-v1 `
  --android-bin .\build-android-arm64-migration\bin
```

Use a new output directory for each version. For a Windows-only bundle,
omit `--android-bin`. The model export recipe is separately documented in
[the developer guide](fashn_vton.md); do not attempt to regenerate it on the
phone.

The source overlay and bundle manifest bind this transfer together.
If a source-overlay file changes, regenerate the bundle rather than editing
the manifest by hand. Native outputs should always retain the binary hash
and bundle-manifest hash that were used for their run.
