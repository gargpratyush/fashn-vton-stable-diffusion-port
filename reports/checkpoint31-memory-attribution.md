# Optimization checkpoint 1: phase-resolved memory attribution

Checkpoint 1 is complete. This is diagnostic memory/loader evidence, not a
new full-image quality study or clean performance benchmark. All experiments
ran serially on the CPU VM; system file caches were not flushed.

## What changed

Added opt-in loader read-buffer/F32-conversion live and peak counters;
actual parameter-buffer capacities and directly mapped parameter payload;
source-file mapping inspection; graph construction/allocation/execution,
output, workspace-release and model-destruction events; Windows process
working set, kernel high-water working set, private commit, page faults and
thread count; and selected-boundary virtual-region/resident-source page scans.
Backend-internal allocation totals that cannot be queried remain explicitly
unknown. Planned CPU work bytes are not mislabeled as complete backend memory.

The diagnostic trajectory executable now supports explicit mmap and loading
thread controls, eager-load-only, repeated conditional-t0 probes, unrecorded
sampling, and JSONL profiling. Default numerical behavior and recorded
trajectory behavior remain unchanged. The external measurement harness can
save a timestamped process timeline. The new summary utility checks completion,
command/specification identity, graph counts, scratch release and mapping
destruction, and records artifact hashes.

## The quantization memory reversal is explained

For original BF16 checkpoint -> Q8/F32, mmap on, 16 loading workers:

| Quantity at load-ready, unless marked peak | Bytes |
|---|---:|
| Actual allocated destination parameter buffers | 1,808,125,632 |
| Directly mapped runtime parameter payload | 0 |
| Retained original-file virtual mapping | 1,943,668,048 |
| **Resident original-file pages** | **1,943,633,920** |
| Current process working set | 3,763,859,456 |
| Kernel peak working set | 4,370,685,952 |
| Private commit | 1,837,506,560 |
| Resident MEM_PRIVATE regions | 1,813,762,048 |
| Resident MEM_MAPPED regions | 1,943,990,272 |
| Peak accounted read-buffer capacity | 438,681,600 |
| Peak accounted F32 conversion capacity | 452,198,400 |
| Peak simultaneous accounted conversion scratch | 881,049,600 |

The original weights were not merely mapped virtually: almost the entire
source was resident alongside separately allocated converted parameters.
Scratch live counts returned to zero after loading; the source mapping did
not disappear until model destruction. Private commit did not include the
same mapped-file residency, explaining why different memory metrics can
appear contradictory.

With mmap off and the same 16 workers, peak working set was 2,371,211,264
bytes (2.2084 GiB), versus 4.0705 GiB with mmap on. Two loading workers reduced
the non-mapped load-only peak further to 1.7882 GiB, with 134.77 MiB accounted
scratch. The corresponding 16-worker mapped run reached 840.23 MiB scratch.
Concurrency, allocation capacity growth and scheduling affect these peaks.
Individual read/F32 peaks need not occur simultaneously; do not add them
and call the sum an observed combined peak.

## Warm-forward controls

Same original checkpoint, photographic bottoms prepared conditions, seed 42,
16 inference threads, two loading workers, F32 flash attention. Each process
performed two conditional t0 forwards on the same noise, then released the
workspace and model. BF16 storage uses F32 matrix computation; Q8 uses the
existing direct quantized arithmetic path.

| Policy / mmap | Process peak GiB | Second graph-end working set GiB | Retained runtime arena MiB | Planned CPU work MiB |
|---|---:|---:|---:|---:|
| BF16/F32, off | 2.9337 | 2.9262 | 590.477 | 2.501 |
| BF16/F32, on | 3.4256 | 3.4181 | 590.477 | 2.501 |
| Q8/F32, off | 2.3058 | 2.2981 | 538.102 | 44.825 |
| Q8/F32, on | 4.1161 | 4.1084 | 538.102 | 44.825 |

**Q8 does use less memory when the duplicate source mapping is absent.**
The matched non-mapped warm controls differ by approximately 0.628 GiB.
The logical weight saving is 0.618 GiB; graph arena and CPU scratch also
differ, so not every process byte is attributable to weight storage.

BF16 mmap load-only results were misleadingly low because untouched mapped
core weights had not yet been faulted into the process working set. Warm
controls explicitly eliminate that interpretation.

Instrumented second-graph intervals were 66.27/66.13 seconds for BF16 off/on
and 49.43/49.68 seconds for Q8 off/on. These include observer overhead and
are single observations, not clean latency estimates or confidence intervals.

## Correctness and lifecycle

- BF16 mmap-on/off velocity outputs are bit-identical, finite, and contain
  1,492,992 values. Q8 mmap-on/off velocities are also bit-identical.
- All 34 conditional/null floating graph captures pass the unchanged
  reference gates; maximum capture relative L2 is 0.0000099220.
- A real one-step, CFG1.5, skip0, unrecorded sampler run exercised both
  conditional and null paths and produced only the intended final tensor,
  manifest and phase profile. Against original-model t0 velocities/noise,
  its Euler output relative L2 is 0.00000133136; max abs is 0.00000619888.
- All 16 load-only profiles and four two-forward profiles close cleanly,
  with no source mappings retained after model destruction.
- Native focused regressions passed (five tests). Python measurement,
  policy and ablation regressions passed (14 tests); the subsequent summary,
  argument and measurement batch also passed (14 tests, including eight
  invalid-option subcases). An analysis command initially requested `x`
  instead of the reference tensor's actual name `noise`; it failed explicitly
  before comparison and was corrected. No model experiment was rerun for it.

These checks preserve existing same-policy Q8 behavior; they do not resolve
the previously demonstrated Q8-versus-F32 full-trajectory disagreement.
Public quantized/GPU guards and all numerical gates remain unchanged.

## Persistent evidence

- `checkpoint31-memory-load`: 16 specifications, native profiles/manifests,
  external measurements/timelines and logs.
- `checkpoint31-memory-probes`: four two-forward controls and velocities.
- `checkpoint31-floating-graph\comparison.json`: 34 reference captures.
- `checkpoint31-sampler-smoke`: actual conditional/null sampler output.
- `reports\checkpoint31-load-summary`, `checkpoint31-probe-summary`,
  `checkpoint31-sampler-summary`: structured summaries and parity records.

Executable SHA256 for the trajectory experiments:
`3ddd69a44a333ad38c9e1d91b791ac579e7a7482b6ed825b1c2ab161980a2b52`.
Model/input hashes and full commands are recorded in each measurement.

## Checkpoint 2 decision

Proceed with runtime-policy-matched BF16, Q8 and four-matrix mixed GGUF.
Loading a matching file should avoid runtime conversion entirely and make
mmap useful rather than retaining a second precision representation.
The existing original-checkpoint non-mapped, two-loader path is already a
working lower-memory diagnostic control. Do not introduce general loader
lifetime or conversion-queue changes unless the ready-file path leaves a
measured need: preserving live zero-copy mappings and non-FASHN behavior
is more important than speculative refactoring.
