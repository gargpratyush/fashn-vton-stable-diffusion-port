# FASHN quality gates and maintenance

This is the operational companion to the
[audit and sequential plan](fashn_code_quality_plan.md).
The original audit is a baseline, not a claim that all upstream code was reviewed.

## Implemented safeguards

The fork has a dedicated `FASHN quality` workflow for `main` pushes and all pull
requests, without path filters that miss Python/UI changes. Its CPU jobs build
the CLI/server and focused tests without weights, preparation SDKs, or the
independent frontend. Python and Node contracts run separately. Linux additionally
runs focused clang-tidy and encoder ASan/UBSan checks.

Public FASHN generation now uses a small adapter in `src/stable-diffusion.cpp`
and the isolated `src/runtime/fashn_try_on.cpp`. Prepared normalization, sampler
order, RNG draw order, crop conversion, callbacks, and bounded modulation caching
are preserved. Context construction and FASHN C entry points translate exceptions
to logged failure; result buffers have scoped ownership through the last callback.

An unexpected generation exception retires that context. Free and recreate it;
`sd_ctx_supports_try_on` becomes false. Expected validation errors and cancellation
do not retire a healthy context. Callback lifetimes remain request-local, callbacks
must not throw or reenter generation, and cancellation remains between forwards.
`free_sd_ctx(NULL)` is explicitly supported.

The server contains worker exceptions, marks pending work terminal, releases image
charges, and stops admission/listening instead of reusing an uncertain shared model.
Ordinary boolean generation failures remain per-job errors. Worker and monitor
threads have scoped stop/join ownership, including setup failures. Test-only fault
seams are compiled only into the server-job test, not the deployed server.
The shared encoder now uses defined unsigned bit operations.

## Test tiers

| Tier | Requirements | Entry point |
|---|---|---|
| Native fast | C++17 CPU build, no model | `ctest --test-dir build-quality -C Release -L fast --output-on-failure --no-tests=error` |
| Python fast | Python 3.12 and `scripts/requirements-fashn-tests.txt` | `python scripts/run_fashn_fast_tests.py` |
| Oracle components | CPU torch/reference requirements; no full model | `python scripts/run_fashn_fast_tests.py --oracle` |
| UI contract | Node 20 | `node --test scripts/test_fashn_vton_ui.cjs` |
| Model | Locally verified checkpoint, conversion/trajectory fixtures | Optional CMake inputs below; `ctest ... -L model` |
| Preprocess | Optional OpenCV/ONNX Runtime and consented research fixtures | Native-preparation scripts described in the [quickstart](fashn_quickstart.md) |
| Live server/browser | Local server/model and browser prerequisites | `scripts/test_fashn_vton_server.py` and `scripts/test_fashn_vton_browser.py`; use their `--help` |
| Publication | Completed study and historical artifact set | `scripts/build_fashn_comparison_report.py`, then `scripts/package_fashn_precision_reports.py` |
| Device | Native ARM64 machine or Android device, weights and prepared fixtures | [Android checkpoints](fashn_android_plan.md), [transfer runbook](fashn_arm_transfer.md) |

From a compiler developer shell, a no-SDK configuration is:

```powershell
cmake -S . -B build-quality -DSD_BUILD_TESTS=ON -DSD_BUILD_EXAMPLES=ON `
  -DSD_FASHN_PREPROCESS=OFF -DSD_SERVER_BUILD_FRONTEND=OFF `
  -DSD_WEBP=OFF -DSD_WEBM=OFF -DGGML_BLAS=OFF -DGGML_LLAMAFILE=OFF
cmake --build build-quality --config Release --parallel 4
ctest --test-dir build-quality -C Release -L fast --output-on-failure --no-tests=error
python -m pip install -r scripts\requirements-fashn-tests.txt
python scripts\run_fashn_fast_tests.py
node --test scripts\test_fashn_vton_ui.cjs
```

The Python runner deliberately does not use unrestricted discovery: some integration
scripts require command-line arguments and real fixtures. Four optional portability
tests are explicitly skipped unless `FASHN_TRAJECTORY_EXE` and `FASHN_BUNDLE` are
set. The latter refers to the original verified BF16 bundle, not an arbitrary
newer bundle. Skips are not device or model passes.
The six torch-based trajectory-state component tests are in the explicit `--oracle`
tier, not silently imported into the minimal CI environment.

Optional CMake inputs are `SD_FASHN_TEST_CHECKPOINT`,
`SD_FASHN_TEST_CONVERSION_DIR`, and `SD_FASHN_TEST_FIXTURES`. Requesting conversion
tests requires all eight `model-<type>.gguf` files: `f32`, `bf16`, `f16`, `q8_0`,
`q4_0`, `q5_0`, `q4_K`, `q5_K`. Missing requested conversions are configure errors.
Synthetic tests cover protected tensors, mixed restoration and unsupported
types without loading those large files. Quantization remains diagnostic-only;
these checks do not remove the public quantization guard.

## Formatting and static analysis

Use clang-format **19.x** with the cross-platform wrapper:

```powershell
python scripts\format_code.py --base HEAD --check
python scripts\format_code.py --base HEAD --write
```

Before a PR, replace `HEAD` with its base commit to check the complete change.
Without `--base`, the tool checks all tracked/untracked owned C/C++ candidates;
this exposes inherited formatting debt. `--formatter` accepts an explicit tool
path. Both shell wrappers forward the same arguments and nonzero failures.

Candidate selection excludes vocabulary, dependencies, frontend submodules and
generated/local data before reading content. The changed-line gate is intentional:
this task does not rewrite every pre-existing style difference in shared code.
New files are checked in full. Check mode never writes source; `--write` applies
only selected replacements.

For focused analysis, generate `compile_commands.json` with a Ninja/Makefile
build and `CMAKE_EXPORT_COMPILE_COMMANDS=ON`, then run:

```text
clang-tidy-19 -p build-quality src/core/api_boundary.cpp src/runtime/fashn_try_on.cpp --warnings-as-errors=*
```

On Windows, use the **x64** clang-tidy executable and a Visual Studio developer
shell. The bundled x86 tool crashed in this environment. MSVC compilation
databases contain `/MP`, which is irrelevant to analysis; use
`--extra-arg=-Wno-unused-command-line-argument` for that driver-only warning.
Header warnings outside the selected translation units remain inherited debt,
not a blanket claim of a warning-free dependency tree.

## Versioned study recovery

New Q4/Q5 studies write `fashn-q45-study-v2` state. Their measurement records include
the harness/child PID, Windows creation time, original host, and process phase.
The driver and recovery command share an exclusive lock.

```powershell
python scripts\recover_fashn_study.py --study <new-study-directory> --job <job-label>
python scripts\run_fashn_q45_study.py --repo . --experiment <experiment-root> `
  --output <new-study-directory> --resume
```

Recovery verifies source/input/weight hashes and all completed artifacts before
archiving the one incomplete attempt. It refuses live owned processes, checks
creation identity rather than PID alone, never terminates a process, and leaves
completed measurements intact. An interrupted archive/state update can resume
through its recovery journal.

An abrupt crash in the launch-to-identity-recording window is intentionally
**not** automatically recoverable: the harness cannot prove child ownership.
Perform process inspection on the original host and retain the attempt; do not
edit provenance or treat uncertain process state as permission to rerun.
Foreign-host liveness checks, legacy schema migration, and changed provenance
are refused. Frozen historical studies and transfer bundles retain their original
source/harness requirements.

For a new standalone measured Windows run, rather than modifying the original
portability wrapper:

```powershell
python scripts\run_fashn_job.py --output <fresh-output-directory> --timeout 14400 `
  -- <absolute-executable-path> <arguments>
```

This records process identity, binary hash, command, timing/memory and a terminal
failure/interruption state. It is a process harness, not a replacement for bundle,
architecture, thermal or model/fixture validation. Follow the device runbook too.

## Reports and publication

Shared hashes/atomic writes/path handling live in `fashn_artifacts.py`; numerical
metrics live in `fashn_metrics.py`. Report generation no longer imports the study
orchestrator for these utilities. `fashn_report_policy.py` validates the recorded
selected format, mixed matrices, and repeat/seed row identities.

Labels and row selection now follow metadata, including selections other than
Q5_K. Generated files are staged and hashed before installation. The publication
manifest is installed last and removed before a multi-file replacement begins.
An interrupted install has **no success ledger**; rerun publication from verified
inputs. This is not an atomic directory swap. The lock serializes publishers.

The historical HTML, contact sheets, evidence, original transfer bundles and
their manifests were not regenerated by this maintenance change.

## Acceptance boundaries

Later [Samsung S23 Q4_K evidence](../reports/android-s23-q4k/README.md) concerns
the older `e86c564` diagnostic build, not this maintenance implementation.
It demonstrates native execution but failed strict numerical parity; the
maintenance-build device limitations below still apply.

Host verification for this implementation:

| Check | Observed result |
|---|---|
| Windows CPU build | Full Release build and a clean Ninja build without preparation SDKs succeeded |
| Native fast tier | 17/17 passed, including runtime/API exceptions, worker execution/finalization failures and thread cleanup |
| Python with local fixtures and oracle components | 49/49 passed, no skips |
| Minimal fresh Python environment | 43 collected; 39 passed, four explicitly optional fixture tests skipped; no torch installed |
| Node UI contracts | 7/7 passed |
| Live HTTP contracts | 16 collected; 10 passed, six inference/raw/soak tests skipped; 31 malformed-request cases, restart identity and graceful shutdown checked |
| Report browser check | Current template exercised in Edge with recorded Q5_K and synthetic Q4_K selection metadata; no page errors or external requests |
| Real prepared-input CLI control | Two steps, CFG 1.5, seed 42, eight CPU threads, F32 flash attention: exact pixels versus the existing two-step reference (RMSE 0, max error 0) |
| Formatting / analysis | Changed-line clang-format check passed; native x64 clang-tidy passed for the two new boundary/runtime translation units |
| Encoder | All-byte output matched Python Base64; WSL GCC ASan/UBSan run passed |
| Android | ARM64 CLI, server, trajectory diagnostic and new runtime/worker tests cross-built; not executed on a device |

The two-step CLI control took 213.867 seconds including loading/output. It is a
correctness control, not a controlled performance comparison with the optimized
20-step precision study. No fresh full 20-step matrix, full-model trajectory
sweep, raw-preparation browser run, or sustained device thermal test was performed.

No public ABI signature or numerical tolerance was relaxed. Existing initial-noise,
schedule and trajectory gates remain in force, and known quantization drift is
not relabeled as floating parity.

GitHub workflow execution and required branch-protection checks are only proven
after the owner publishes the change and observes the runs. Full Linux-library
and device execution are separate from a Windows host build, a Linux encoder
sanitizer run, or Android cross-compilation. Native Windows ARM64, Android/S23
thermal behavior, GPU execution and every unrelated upstream model remain outside
this host's acceptance evidence.
