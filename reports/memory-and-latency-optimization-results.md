# FASHN memory and latency optimization: completed results

The eight-checkpoint optimization/evaluation pass is complete. Quantization
now reduces measured process memory, not just file size. The fastest tested
floating native configuration uses diagnostic OpenBLAS with CPU fallback,
at a significant private-commit cost. This is not a promotion of public
quantized, BLAS or GPU support, nor a claim that every proposed candidate
was worth implementing.

The original [plan](memory-and-latency-optimization-plan.md) is preserved as
a historical proposal. Earlier model development, Python parity and
quantization experiments are covered by
[implementation history](implementation-and-experiment-history.md) and
[quantization/upstream comparisons](quantization-and-upstream-comparison.md).

## 1. Final configurations and measurements

Windows CPU-only VM, AMD EPYC 7763 exposure, 8 physical/16 logical CPUs,
64 GB RAM, AVX2/FMA, VS2022. Model experiments ran serially; global caches
were not flushed. A fresh process is not a cold filesystem-cache experiment.

All rows below generate 576x864 at 20 steps, CFG1.5, shift1.5, skip1, seed42,
16 inference threads and sequential conditional/null execution (39 forwards).
They use matching runtime-ready weights, mmap off, layerwise modulation
precomputation with weight release, and strict fused F32 GELU. Sampling
includes modulation preparation; process wall includes recording/startup.
These are prepared-input runs, not raw-image HTTP end-to-end timings.
GiB means 2^30 bytes.

| Case / policy | Sampling seconds | Process wall seconds | Peak working set GiB | Sampled peak private commit GiB |
|---|---:|---:|---:|---:|
| Cardigan, BF16 storage / CPU F32 | 2371.130 | 2371.550 | 2.0109 | 2.1402 |
| Cardigan, BF16 storage / BLAS F32 | 1598.136 | 1598.867 | 2.0281 | 4.2843 |
| Bottoms, Q8 | 1858.787 | 1859.345 | 1.4016 | 1.5277 |
| Cardigan, Q8 | 1863.072 | 1863.484 | 1.4002 | 1.5281 |
| Bottoms, mixed | 1902.516 | 1902.918 | 1.5261 | 1.6547 |
| Cardigan, mixed | 1898.820 | 1899.201 | 1.5266 | 1.6532 |

These six full runs and four fresh-process, three-forward cardigan controls
are recorded in `..\checkpoint38-integrated\results.json`, with root
`passed=true`. Four quantized runs preserve all 21 legacy tensor files each:
**84/84 same-policy file comparisons are bitwise equal**, and all four
cropped PNGs are unchanged. Quantized floating-quality acceptance remains false.
The declared working-set ceilings were 3 GiB floating and 2 GiB quantized;
these were not private-commit caps.

Warm cardigan CPU/BLAS/BLAS/CPU medians, excluding first forwards, were
60.9395 / 40.7543 seconds: **33.12% less time**. Full cardigan sampling was
32.60% faster with BLAS. Checkpoint 35's separate, matched bottoms controls
were 62.5503 / 42.1245 seconds, a 32.65% reduction.

Earlier accepted full BLAS runs remain valid but are not relabeled as new
checkpoint-38 repeats: original example tops 1623.9465 seconds / 2.0289 GiB
working set; bottoms 1615.2627 seconds / 2.0288 GiB. The original example tops
is the black-shirt/worn-garment example, not the independent flat-lay cardigan.

## 2. Quantization memory: cause, fix and resulting ownership

Checkpoint 31 isolated the original-checkpoint mmap policy. Q8 retained
1,943,633,920 resident source bytes alongside 1,808,125,632 destination
weight bytes. Accounted conversion scratch reached 881,049,600 bytes with
16 mapped loading workers. This was source/destination duplication and
conversion pressure, not intrinsic Q8 memory overhead.

Runtime-ready GGUF avoids runtime conversion and the duplicate original
source. Layerwise modulation preparation additionally retires
1,044,172,800 bytes of conditioning/modulation parameters before denoising.
The 37 large projections themselves account for 1,035,468,800 bytes.
The implementation caches 206,080 F32 values per pair; 39 pairs consume
32,148,480 bytes. These are the measured implementation counts, superseding
the original plan's approximate 202,240-value / 30.088-MiB estimate.

| Quantity, bytes | BF16-core/F32-protected | Q8-core/F32-protected | Mixed: 100 Q8 + 4 original F32 |
|---|---:|---:|---:|
| Runtime-ready file | 2,471,712,672 | 1,808,160,672 | 1,942,919,072 |
| Registered parameter payload | 2,471,677,632 | 1,808,125,632 | 1,942,884,032 |
| Active allocated denoising parameters | 1,427,504,832 | 763,952,832 | 898,711,232 |
| Graph arena | 600,355,904 | 564,180,032 | 564,180,032 |
| Cached modulation vectors | 32,148,480 | 32,148,480 | 32,148,480 |
| Original source mappings | 0 | 0 | 0 |
| Runtime converted tensors / conversion scratch | 0 / 0 | 0 / 0 | 0 / 0 |

Registered bytes are a logical inventory, not simultaneous live allocation.
Graph arenas exclude library-private allocation. Shared BLAS/CPU host
workspace appears as `BLAS=600355904, CPU=0` because GGML counts the shared
capacity once, not because the CPU fallback consumes no space.

At the last cardigan execution boundary, current working sets were
2,120,548,352 bytes floating CPU, 2,137,829,376 floating BLAS,
1,464,598,528 Q8 and 1,599,520,768 mixed. Corresponding private commit was
2,284,990,464 / 4,516,913,152 / 1,627,975,680 / 1,763,143,680 bytes.
These are specific warm boundary samples, not idle-service retention or
process high-water marks. First denoising execution began at approximately
8.60 / 9.39 / 8.36 / 8.72 seconds respectively, including preceding setup,
modulation preparation and graph preparation; this is not isolated disk I/O.

### Historical same-policy comparisons

| Case / policy | Old peak working set GiB | New peak GiB | Reduction | Old/new process wall seconds |
|---|---:|---:|---:|---:|
| Bottoms Q8 | 4.1514 | 1.4016 | 66.24% | 1947.348 / 1859.345 |
| Cardigan Q8 | 4.1635 | 1.4002 | 66.37% | 1956.684 / 1863.484 |
| Bottoms mixed | 4.2769 | 1.5261 | 64.32% | 2007.393 / 1902.918 |
| Cardigan mixed | 4.3042 | 1.5266 | 64.53% | 1975.864 / 1899.201 |

Historical measurements are the checkpoint-26/30 `*-measurement\measurement.json`
files, under their `run` objects. The roughly 3.9-5.2% wall-time differences
are historical single-run observations, not new interleaved speed estimates.

The approximately two-thirds historical RAM reduction combines loader and
modulation-lifetime improvements. Quantization alone does not explain it:
Q8 saves 663,552,000 raw parameter bytes (0.618 GiB) versus the protected
BF16 policy. Optimized cardigan Q8 uses about 30.4% less peak working set
than optimized floating CPU. F32 activations/protected weights still prevent
a blanket 2x or 4x total-process claim.

## 3. BLAS private commit: isolated startup cost

Do not interpret the similar 2.01/2.03-GiB working sets as total-memory parity.
The ordinary CPU binary peaks at 2.14 GiB private commit; the BLAS binary
peaks at 4.28 GiB in the final cardigan comparison.

An additional bounded diagnostic loaded only the official OpenBLAS 0.3.34
LP64 x64 DLL in fresh standard-library-only Python processes. There was no
NumPy, GGML, checkpoint, inference or matrix multiplication. The requested
thread count was also queried from OpenBLAS.

| `OPENBLAS_NUM_THREADS` | Actual threads | Private-commit increase on DLL load, bytes |
|---|---:|---:|
| 1 | 1 | 262,144 |
| 4 | 4 | 403,812,352 |
| 8 | 8 | 941,871,104 |
| 16 | 16 | 2,018,000,896 |

Thus about **1.88 GiB is already attributable to 16-thread OpenBLAS startup**,
scaling approximately 128 MiB per additional worker. Resident working set
rose by less than 0.6 MiB in each probe. This isolates library startup as the
dominant additional commit source, independently of FASHN or weight packing.
It does not assign every remaining backend-private byte to a named allocation.
The tagged OpenBLAS [Windows thread driver](https://github.com/OpenMathLib/OpenBLAS/blob/v0.3.34/driver/others/blas_server_win32.c)
also allocates a per-worker `blas_memory_alloc(2)` buffer before processing work.

Evidence: `..\checkpoint38-integrated\openblas-dll-memory.json`;
reproducer: session artifact `probe_openblas_memory.py`. The official archive's
verified SHA256 is
`e9cb6134541f36c27346d5fc5995652f060fba227cebbbabcbda5a5a44d7c76b`.
No third-party source was changed.

Checkpoint 35's CPU/BLAS controls used the same BLAS-capable binary and
already had approximately 4.00 / 4.20 GiB private peaks. Checkpoint 38 uses
the ordinary no-BLAS CPU binary for CPU controls, exposing this previously
shared startup cost. These comparison designs must not be conflated.
Selecting the CPU backend in a binary that loads this DLL does not itself
remove its startup cost.

Lower startup thread counts are not a demonstrated free memory saving at
16-thread inference throughput: the backend explicitly sets its thread count.
Do not set one initial thread and assume a later increase to 16 is free.
The selected BLAS configuration is a throughput/private-commit tradeoff;
the separate ordinary CPU binary remains the lower-commit floating choice.

## 4. Floating correctness and visual/numerical comparisons

The new floating CPU cardigan crop exactly reproduces the historical native
PNG. The BLAS recording passes every native floating trajectory gate, with
maximum relative L2 8.83848e-7. This is a native/native intermediate comparison:
there is still no complete Python intermediate cardigan oracle.

The [ten-row local gallery](comparison-gallery-optimized/index.html) includes
inputs, full-resolution outputs, explicit 576x768 crops, side-by-side sheets,
32x difference images, numerical metrics, SHA256 lineage and source attribution.
It compares floating CPU/BLAS, actual original upstream/BLAS, Q8/floating,
mixed/floating and mixed/Q8 for both photographic cases.

| Final cropped comparison | Changed byte channels / 1,327,104 | Maximum byte difference |
|---|---:|---:|
| Cardigan BLAS versus floating CPU | 23 | 1 |
| Cardigan BLAS versus actual original upstream | 82 | 1 |
| Bottoms BLAS versus floating CPU | 18 | 1 |
| Bottoms BLAS versus actual original upstream | 44 | 1 |

Bottoms uses the accepted checkpoint-34 floating and checkpoint-35 BLAS
recordings, not newly generated checkpoint-38 floating outputs.
The actual upstream comparisons use unchanged default `_sample`, not the
slower explicit F32/MATH numerical oracle.

Actual upstream historical process walls remain 1258.04 seconds cardigan
and 1325.48 seconds bottoms. Native BLAS substantially narrows the gap but
does not beat those observations. They are not interleaved reruns.
Only the correctly monitored bottoms Python peak of 6.573 GiB is a valid
historical Python memory observation; discard the launcher-attributed
cardigan Python memory result. Do not compare sampling to process wall or
label either prepared-only path raw-pipeline latency.

### Quantized quality is unchanged, not fixed

The four original-value F32 restorations remain
`single_blocks.{12,11,8,14}.linear1.weight`. No new tuning or blanket
dequantization was performed. Against native floating CPU:

| Case / policy | Cropped MAE, byte levels | Max byte error | PSNR dB | Failed image / guided-velocity steps |
|---|---:|---:|---:|---:|
| Cardigan Q8 | 0.25865494 | 83 | 44.0818 | 9/20 and 20/20 |
| Cardigan mixed | 0.19067760 | 82 | 44.5206 | 7/20 and 20/20 |
| Bottoms Q8 | 0.13016237 | 41 | 55.5840 | 9/20 and 20/20 |
| Bottoms mixed | 0.05832625 | 30 | 58.8960 | 4/20 and 20/20 |

The unchanged relative-L2 gate is 0.001. Detailed per-step values, error
percentiles, counts above 16/32 byte levels and peak-error coordinates are
in `..\checkpoint38-integrated\final-summary.json`.
The cardigan intermediate reference here is explicitly native F32.
The initial noise and all legacy same-policy state bytes remain unchanged.

Mixed reduces mean cropped error by 26.28% cardigan and approximately 55.19%
bottoms, but the cardigan worst error remains near cropped x184/y524 at the
sleeve/torso gap; both versions retain large local differences there.
The bottoms peak is at x296/y393. The contact sheet and amplified cardigan
difference image were inspected: the important residuals follow the sleeve/
torso gap, hem and neckline, rather than disappearing with the lower average.
Pixel agreement is not an independent garment-fidelity score.
The historical independent-cardigan evidence remains independent of the
frozen bottoms-based restoration selection; no broader held-out claim is made.

## 5. Sequential checkpoint outcomes and implementation surfaces

| Checkpoint | Outcome and evidence |
|---|---|
| 1 / 31 | Phase/process/mapping/conversion instrumentation; 16 load-only controls and mmap/forward controls isolated ownership. [Report](checkpoint31-memory-attribution.md). |
| 2 / 32 | Reused/exported exact runtime-ready BF16, Q8 and mixed GGUF; all 366 file tensors verified, including exact Q8 blocks and original restored values. Zero-conversion direct loading succeeded, so the generic chunked-converter/source-lifetime fallback was not implemented. [Report](checkpoint32-runtime-ready-weights.md). |
| 3 / 33 | Layerwise 37-projection precomputation, exact keys, bounded windows, cancellation, multi-sample reuse and actual parameter release. Category/time captures exact; full floating trajectory accepted at 2577.729 seconds / 2.0094 GiB. [Report](checkpoint33-modulation-progress.md). |
| 4 / 34 | Stride-aware, threaded strict-F32 GELU fusion. Initial scalar version was slower; scoped precise floating behavior improved it. Exact full trajectory; 2507.495 seconds / 2.0121 GiB. Warm improvement about 1.2% with overlapping ranges; **no arena saving**, so no strong memory/performance claim. [Report](checkpoint34-gelu-progress.md). |
| 5 / 35 | Isolated supported OpenBLAS build, explicit F32 operands/thread ownership, CPU attention fallback, actual 107 matrix-backend assignments and workspace diagnostics. Initial fallback bug was caught by the 28-flash-node guard and fixed. Primitive, capture and full-trajectory results accepted. [Report](checkpoint35-blas-progress.md). |
| 6 / 36 | Sequential/B=2/flattened GEMM and paired-attention evaluation. Outputs exact but workspaces approximately 1.77-2x, without consistent throughput gain. Graph construction only 20-27 ms of approximately 42-second forwards. **Full-model B=2 and extra image caches were not implemented.** [Report](checkpoint36-cfg-and-reuse.md). |
| 7 / 37 | Lazy parser with startup consent/hash/license validation and pre-load revalidation; ORT controls, opt-in arena disabling, aggregate image-storage budgeting, cancellation/readmission and service soak. [Report](checkpoint37-preparation-progress.md). |
| 8 / 38 | Four alternating warm controls; six complete recordings; exact same-policy preservation; floating gates; phase/commit accounting; final numerical/visual gallery. [Report](checkpoint38-integrated-progress.md). |

Principal source surfaces are loader/model-manager/GGML-runner diagnostics,
`fashn_modulation.h`, `fashn_vton_model.h`, supplied FLUX modulation routes,
`strict_gelu.h`, attention fallback plumbing, math/trajectory diagnostics,
the native preparer pipeline and asynchronous server job management.
`docs\fashn_vton.md` documents supported switches and experimental boundaries.
No changes were made to GGML, thirdparty, or the server frontend submodules.

One checkpoint-34 analysis invocation passed a native recording to a
Python-manifest comparator. Analysis was corrected without repeating
generation; all 21 native files were then confirmed byte-identical.
Final consolidation likewise corrected a historical measurement lookup
from the root object to its `run` object; no successful inference was rerun.

## 6. Preparation and service results

Lazy parser creation saved approximately 444 MiB at startup and 447 MiB after
parser-free preparation. Configured capability remains available before the
session is resident. Startup still checks consent, license and artifact
hashes; delayed loading rechecks integrity. ONNX session-construction
failures now surface at first use, explicitly rather than being swallowed.

All 16 eager/lazy lifecycle images and 12 original-Python preparation modes
were pixel-identical. ORT thread/spinning experiments did not justify a
default change. Opt-in arena disabling, with original eight threads and
spinning, reduced warmed preparer memory from approximately 1596 to 661 MiB.
Flags are `--ort-no-arena` for the preparer and `--try-on-ort-no-arena` for
the server with native DWPose configured.

The server now reserves a 512-MiB aggregate budget for try-on image storage,
separate from the existing 64-job count limit and TTLs. Charges cover raw/
prepared inputs, transition allowance and up to 4 MiB output-string capacity
per sample; active charges are stable while workers mutate storage. Terminal
charges reflect retained capacities. Raw strings are explicitly released.
Admission returns HTTP 429 when the byte budget would be exceeded.
This is **not a cap on models, ORT, graph workspaces, metadata or transient HTTP
copies**, and therefore not a total-process memory limit.

The final combined run passed seven selected service checks with no skips:
raw pixels, cancellation, four-round soak, two samples, 16 concurrent
cancellations, budget exhaustion/recovery, and graceful shutdown.
The budget rejected further work at 24 queued four-sample jobs, before the
count limit: 516,912,790 / 536,870,912 bytes reserved. Cancellation released
capacity and readmission succeeded. Final retained reservations were
10,532,502 bytes, not a claim that retained results consume zero storage.

End-of-soak private commit fell from 4,707,848,192 to 3,179,429,888 bytes.
The adjusted post-warmup span was 884,088 bytes. Shutdown took 74.67 seconds:
cancellation still waits for the current whole CPU forward, not kernel-level
interruption. Evidence is `..\checkpoint37-server-budget-soak\report.json`.
These service/preparer observations are separate from prepared-only full
generation memory and must not be arithmetically combined as a measured
raw-pipeline peak.

## 7. Selection, limits and deliberately unimplemented candidates

For lower-memory supported floating execution, retain the ordinary CPU build,
matching BF16-ready file, mmap/eager loading off, and opt into the bounded
modulation cache. Strict fused GELU is also opt-in; default behavior was not
changed based on its small timing difference. Public model arguments:

```json
{"fashn_modulation_cache":true,"fashn_modulation_cache_mib":128,"fashn_fused_gelu":true}
```

The highest-throughput measured floating option is the separate diagnostic
BLAS build, with an observed sampled private-commit peak of 4.28 GiB and
approximately 2.03-GiB peak working set. These overlapping memory metrics
must not be added together or both described as resident RAM.
The smallest measured diagnostic option is direct Q8 with the same loading/
cache configuration; mixed trades about 128.5 MiB of parameter payload for
better average agreement. Neither quantized policy clears public quality gates.

The successful ready-file path made a generic loader conversion rewrite
unnecessary. No new approximate/vectorized tanh, normalization/residual
fusion, full B=2 transformer, image-specific result cache, shared ORT global
pool, or new retained-workspace/idle-reclamation policy was promoted.
Existing workspace reuse and validation were retained rather than weakened.
Fine-grained timings for every raw substage (decode through response
transport) and a full 20-step raw HTTP latency matrix are not supplied by
these prepared-only runs and whole-preparation/service experiments.

GPU validation remains hardware-blocked. Public Q8/mixed and BLAS selection
remain unsupported. Broader seed/garment quality coverage, further quantized
arithmetic changes and a complete original-Python cardigan intermediate
oracle remain separate work, not implied by this optimization acceptance.
There was no resolution, step-count, CFG or numerical-threshold shortcut.

## 8. Persistent evidence and reproduction

All paths below are relative to `C:\source\fashn-vton-reference`:

- `checkpoint38-integrated\results.json`: commands, manifests, actual types/
  backends, source/executable/model/condition provenance and acceptance.
- `checkpoint38-integrated\final-summary.json`: consolidated phase memory,
  file sizes, timings, per-step quantized error and gallery provenance.
- `checkpoint38-integrated\openblas-dll-memory.json`: isolated startup controls.
- Per-run `memory-profile.jsonl`, external `*-timeline.jsonl`, logs, initial
  state and 20 step files: process peaks versus warm boundaries and source
  lifetimes remain inspectable rather than inferred from a final sample.
- `reports\optimization-gallery-spec.json` and
  `reports\comparison-gallery-optimized\manifest.json`: ten full-resolution
  comparisons with hashes, exact crops, labels, metrics and source attribution.
- `checkpoint35-full-acceptance\results.json`, checkpoint-31 through -37
  reports, and the referenced service/preparation reports: earlier controls,
  failures, rejected candidates and accepted results.

The source/model pins remain FASHN
`7c0f10af3f91ad4048fe9729c470a13ef905d25a` and HF
`7720683168567eb5a2a4c67f15116c6e29c83ded`; original checkpoint SHA256
`d6cd38286885bc29fa487ea9383f80ffeb95862e7747c630d42c5d3c05bdd35a`.
Local research/evaluation consent and image/model license restrictions
continue to apply. No commits, pushes or external image uploads were made.
