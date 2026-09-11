# FASHN Q4/Q5 experiments: execution plan

## Goal and boundaries

Generate complete images with stronger quantization, not just weight-error
or one-forward estimates. Measure latency, actual process memory and
numerical/image differences, and produce an offline comparison gallery.
Proceed through the checkpoints below without another approval prompt.

This is Windows x64 CPU research. It does not establish ARM64, Android,
GPU or public quantized inference support. Do not modify GGML or the
already-created ARM transfer bundle. Preserve the existing uncommitted
ARM handoff changes. Do not commit/push model weights, binaries or results.

Public quantized FASHN initialization remains rejected. Extend only the
explicit conversion/diagnostic surfaces needed to run this study.
Numerical gate failure is a reportable research result, not a reason to
silently suppress an otherwise finite image.

## Policies and controlled settings

| Policy | Role |
|---|---|
| BF16 core / F32 protected / F32 compute | Fresh floating baseline |
| Q8_0 core / F32 protected | Fresh quantized baseline |
| Q4_0 core / F32 protected | Classic four-bit experiment |
| Q5_0 core / F32 protected | Classic five-bit experiment |
| Q4_K core / F32 protected | Four-bit K-block alternative |
| Q5_K core / F32 protected | Five-bit K-block alternative |
| Selected Q4/Q5 + four original F32 matrices | Surgical mixed-precision follow-up |

Q4_K/Q5_K refer to actual GGML tensor formats, not model-wide
`Q4_K_M`/`Q5_K_M` presets. Do not mislabel them.

Keep the existing 104-matrix eligibility rule. Preserve embeddings,
modulation/conditioning weights, norms, biases, patch kernels and final
layers as F32. K-block row divisibility must be checked explicitly.
Always quantize from the original floating checkpoint, never from Q8.
Match the converter's uniform all-ones importance vector during block
verification. This is not an activation-calibrated importance matrix;
passing null selects a different quantizer path for some lower-bit types.

The four prior restoration candidates are:

```text
single_blocks.12.linear1.weight
single_blocks.11.linear1.weight
single_blocks.8.linear1.weight
single_blocks.14.linear1.weight
```

These are a hypothesis for lower-bit restoration, not proven Q4/Q5 optima.
Restore original floating values, not dequantized approximations.

Controlled generation: prepared 576x864 inputs, sample count one, 20 steps,
CFG 1.5, shift 1.5, skip final CFG step, seed 42, 16 CPU threads, two loading
threads, CPU-only backend, F32 K/V flash attention, bounded modulation cache,
no mmap/eager loading and fused strict GELU enabled consistently. No BLAS.
Quantized matrix execution is direct unless explicitly marked weight-only.

Use cardigan and bottoms, whose matching prepared inputs and original
upstream PNGs already exist. Both expose known garment-boundary weaknesses.
Do not regenerate parser outputs or add preprocessing timing to a
prepared-inference comparison.

## Define "variance" explicitly

1. **Numerical/image drift:** every recorded image and guided velocity
   versus the matched fresh floating run; relative L2, RMSE, max error.
   Final full/cropped images: changed channels, MAE/RMSE, PSNR, signed
   difference mean/variance, and worst local error regions.
2. **Runtime variability:** three separate fixed-state probe processes per
   policy, each with a first and warmed forward; report individual samples,
   mean, sample standard deviation, min/max and coefficient of variation.
   These are forward benchmarks, not full-generation latency estimates.
3. **Repeatability:** two additional complete same-seed cardigan runs for
   the selected lower-bit policy, giving three same-case full observations.
   Compare their final floats/pixels for nondeterminism.
4. **Seed sensitivity:** an additional seed-43 cardigan floating/selected
   pair. Compare policies within the same seed. Two seeds provide a
   descriptive sensitivity check, not a population-quality estimate.

Report Windows working set and private commit separately; they overlap.
Report file bytes, active parameter bytes, graph workspace, cache,
conversion scratch and phase memory. Never infer memory savings from
file size alone.

## QP0: Freeze protocol and provenance

Write this plan before implementation. Inventory original/ready weights,
matrix map, prepared PNGs/normalized tensors, historical original-upstream
outputs, toolchain/source state and available storage.

Pin hashes of all inputs and relevant executables/scripts before the
automated run. Verify prepared PNG-to-float lineage. Use a new experiment
directory, with an atomic progress manifest and per-job logs. Preserve
historical baselines without overwriting them.

Acceptance: explicit settings, two matched cases, required local artifacts,
and no concurrent model benchmarks on this machine.

## QP1: Extend diagnostic conversion and execution safely

The inspected converter currently restricts FASHN export to floating/Q8.
Extend its explicit quantization allowlist to Q4_0/Q5_0/Q4_K/Q5_K, keeping
protected weights F32 and public inference guarded.

Extend the runtime-policy exporter, conversion verifier, graph diagnostic
and trajectory diagnostic coherently. Use a shared format parser rather
than incompatible independently maintained lists.

Critically, replace the trajectory's Q8-specific direct-arithmetic test
with quantized-type detection. Otherwise newly accepted Q4/Q5 might be
silently expanded to F32 computation and the memory/latency comparison
would answer the wrong question.

For conversion verification:
- Confirm every tensor name/shape/type and protected floating value.
- Requantize original weights with the same GGML quantizer and compare
  encoded blocks byte-for-byte.
- Preserve the existing Q8 rounding-bound check.
- Do not apply a Q8 numerical error bound to unrelated lower-bit formats.
- Reject unsupported formats, incompatible block dimensions and attempts
  to restore original precision from an already quantized source.

Acceptance: existing targeted tests, new format/export tests and public
guard checks pass. No GGML changes and no default floating behavior change.

## QP2: Export and verify ready files

Create four ready lower-bit GGUF files from the original checkpoint.
Reuse verified BF16/Q8 files for the controls. Export the selected mixed
policy only after selecting its lower-bit base.

Verify complete runtime policy and exact quantized block payloads.
Log conversion/verifier measurements separately from inference.
Save export manifests, hashes, exact sizes and weight-error diagnostics.

Acceptance: all intended 104 matrices quantized, other parameters protected;
no runtime conversion or resident source mmap required.

## QP3: Fixed-state support, latency and memory probes

For each of the six base policies, run three fresh processes, two
conditional t=0 forwards per process, with the same seed/cardigan inputs.
Record first/warm timings, process high-water working set, sampled private
commit and logical allocation counters. Keep source mappings off.

Require finite correct-shape outputs, expected matrix-type inventory,
28 flash operations through the existing runner guard, and no runtime
conversion. Compare velocity against the floating policy.

A failed floating numerical gate for a quantized policy does not block
full-image diagnostics. An operational crash, invalid tensor, unexpected
precision/upcast or wrong execution path must be investigated first.

Acceptance: actual direct lower-bit execution established; lower process
memory demonstrated or any contrary result attributed explicitly.

## QP4: Complete images and fresh baseline comparisons

Run full 20-step recorded trajectories sequentially:

```text
cardigan: BF16, Q8, Q4_0, Q5_0, Q4_K, Q5_K
bottoms:  BF16, Q8, Q4_0, Q5_0, Q4_K, Q5_K
```

Use the same recording/profile policy for all compared jobs. Capture
fresh-process wall time, sampling time and peak memory. Do not compare a
recorded diagnostic duration directly to an unrecorded CLI measurement.

Render final full-resolution PNGs directly from the saved floating image,
using the existing clamp/scale/truncate convention; apply the identical
prepared crop for human-facing comparisons. Publish difference maps and
local high-error crops. Compare against fresh BF16 numerically and the
historical original Python images separately, explicitly labeling their
provenance and timing as historical.

Acceptance: complete images for every operational lower-bit policy in both
cases; per-step and final metrics report PASS or FAIL without changing the
original relative-L2 <= 0.001, noise <= 1e-6 and schedule <= 1e-7 gates.

## QP5: Select a candidate, mixed precision, repeatability and seed 43

Select among the four lower-bit policies by lowest average cropped pixel
RMSE versus fresh BF16 across both cases; break ties by measured sampling
time, then policy name. This is a numerical screening rule, not a claim
of human perceptual preference. Keep all candidate images visible.

Export that policy with the four prior sensitive matrices restored to
original F32, then generate both full cases. Quantify whether localized
garment errors actually improve, not merely whether aggregate error falls.

Repeat the selected unmixed cardigan run twice, preserving seed/settings,
and report three-run full latency variation and same-seed determinism.
Generate a fresh seed-43 floating/selected cardigan pair.

Acceptance: mixed restoration evaluated visually/numerically, full-run
variance separated from forward variance, and same-seed versus cross-seed
effects clearly distinguished.

## QP6: Offline gallery and final report

Create an HTML gallery with original prepared person/garment views,
original-Python reference, fresh BF16, Q8, all Q4/Q5 candidates and mixed
outputs. Include full-resolution links, amplified difference maps,
worst-region crops, settings and execution/numerical status.

Regenerate the partial gallery after each completed full image, so early
results remain usable while later jobs run. Write results.json and a
Markdown report with measured latency/memory, numerical drift, variability,
and explicit open human-review questions.

Questions for human review: garment identity, sleeve/hem placement,
boundaries, texture/logo preservation, person/background preservation,
and whether quantization amplifies failures already present in floating.
Do not invent a human approval or a perceptual-quality score.

Acceptance: links/images/metrics correspond to completed artifacts; failed
or pending runs are visibly labeled, all planned jobs have an explicit
outcome, and no quantized public promotion is implied.

## Execution and recovery

This is a multi-hour study: twelve initial full generations, two mixed
generations, two repeats and a seed-43 pair, plus probes and conversion.
Use one model process at a time; do not run simultaneous benchmarks or
compete with another local inference workload.

The orchestration must write progress before and after each job, preserve
per-job exit codes/timings, fail explicitly on operational errors, and
support resuming completed jobs only after checking their provenance and
artifact hashes. A numerical FAIL is not an operational exception.

Long-running work remains attached to this CLI session. Keep the session
open; do not claim that execution survives closing it. The final report is
not complete merely because jobs have been queued.

### Sequential driver

After all six verified exports exist under the study's `ready` directory:

```powershell
& ..\fashn-vton-reference\.venv\Scripts\python.exe scripts\run_fashn_q45_study.py --repo . --experiment ..\fashn-vton-reference --output ..\fashn-vton-reference\q45-study
```

The default runs QP3 through QP6 without another prompt. `--through probes`
or `--through images` can intentionally stop at a checkpoint. `--resume`
continues only after verifying frozen executable/source/input identities
and all completed job artifacts. A Windows file lock prevents simultaneous
drivers. Failed/incomplete jobs require explicit investigation and recovery;
they are never automatically relaunched while an old process may remain.

Open `..\fashn-vton-reference\q45-study\index.html` for the progressive
gallery, `report.md` for the live tables and `results.json` for raw values,
hashes, commands and status. The driver does not modify the ARM bundle.

### Conversion observations before inference

All four lower-bit files passed exact encoded-block verification for
104 matrices and protected-value verification across the checkpoint:

| Format | Ready file bytes |
|---|---:|
| Q4_0 | 1,454,266,272 |
| Q5_0 | 1,542,739,872 |
| Q4_K | 1,454,266,272 |
| Q5_K | 1,542,739,872 |
| BF16/F32 control | 2,471,712,672 |
| Q8_0/F32 control | 1,808,160,672 |

These are file sizes, not process memory or quality results.

Two early verification attempts failed and were preserved: the first
exposed the remaining Q8-only metadata allowlist; the second exposed a
null-versus-uniform importance-vector mismatch in the test's requantizer.
Both were corrected without changing the quantizer implementation or
requantizing a quantized checkpoint. The successful verifier matches the
converter's all-ones importance input and compares exact encoded blocks.
