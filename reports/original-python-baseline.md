# Original FASHN Python inference baseline

The upstream reference is [FASHN AI's original `fashn-vton-1.5`
implementation](https://github.com/fashn-AI/fashn-vton-1.5/tree/7c0f10af3f91ad4048fe9729c470a13ef905d25a).
The native BF16/F32, Q8, Q4 and Q5 rows describe implementations/precision
policies of this same model, not different model architectures.

**The original Python sampler was faster in the recorded Windows CPU
observations; native inference used less measured memory for the case with
valid Python memory telemetry.** These are historical observations, not a
controlled language-level speedup claim or an Android forecast.

## What "original/default Python" means here

The [reference harness](../scripts/run_fashn_upstream.py) calls the unmodified
`TryOnPipeline._sample` with upstream default CPU SDPA dispatch and batched
conditional/null execution. It does not substitute the slower forced-F32-math
oracle used by some numerical diagnostics. It also retains upstream's redundant
final null computation when final-step CFG is skipped.

The measurements use Windows x64 CPU, AMD EPYC 7763 exposure, 16 inference
threads, PyTorch `2.8.0+cpu`, F32, 20 steps, CFG 1.5, shift 1.5, seed 42,
and `skip_cfg_last_n_steps=1`. Canvas is 576x864; the displayed crop is
576x768. Inputs are the same prepared cardigan/bottoms fixtures used in
the native comparisons.

**This is not the full default raw-photo pipeline or an upstream GPU
benchmark.** The harness bypasses the pipeline constructor and raw `__call__`,
loads the original weights through meta-device initialization, and creates no
parser/detector. "Default" refers to the unchanged sampler and CPU attention
dispatch, not every upstream device/dtype/startup default.

## Measured baseline

| Original Python case | Sampler + PIL s | Measured process wall s | Peak working set GiB | Sampled peak private commit GiB |
|---|---:|---:|---:|---:|
| Cardigan | 1241.326 | 1258.037 | N/A: invalid launcher measurement | N/A |
| Bottoms | 1263.737 | 1325.477 | 6.573 | 9.603 |
| Two-case arithmetic mean | 1252.532 (20.876 min) | 1291.757 (21.529 min) | N/A: one case unavailable | N/A |

Bottoms memory was measured on the actual Python interpreter:
`7,057,371,136` bytes peak working set and `10,310,897,664` bytes sampled peak
private commit. These overlap; do not add them. The cardigan run's approximately
11.7 MB launcher working set is invalid for model comparison and must never
be treated as Python's inference memory or included in a mean.

There is one full run per case, with no repeated-run confidence interval.
Sampling includes PIL conversion but no intermediate trajectory recording.
Process wall covers the harness process interval, not raw-photo preparation or
the original full pipeline constructor.

## Reading the comparisons

The [main README](../README.md) and [Q4/Q5 results](q4-q5-results.md) include
Python alongside the current native policies. The
[offline HTML](fashn-all-comparisons.html) includes it in the main per-case
latency/memory/error table and two-case timing summary, as well as the historical
controls and selectable image references.

Native recorder sampling includes modulation preparation and saved intermediate
states; Python uses different attention kernels and batched CFG without those
recordings. Do not interpret the table as a controlled speed ratio or attribute
the difference solely to Python versus C++. Python's two-case mean memory is
unavailable, not zero and not the single valid bottoms peak.

The [Android S23 comparison](android-s23-q4k/README.md) includes this Windows
Python baseline for context. No original-Python timing or memory baseline was
measured on the S23. Its one-thread CPU run, final-only recording, charging and
thermal conditions cannot be pooled with the 16-thread Windows results.

Python-specific GGML arenas, modulation-retirement counters and ready-GGUF sizes
are not available/applicable. A model file's size is not process memory. There
are no corresponding full-model Python repeatability, seed-43, or Android
measurements in this evidence set; those comparisons must say unavailable
rather than substitute unrelated oracle or microbenchmark timings.

## Actual images and provenance

Original Python output crops:
[cardigan](comparison-gallery-precision/images/cardigan-python.png) and
[bottoms](comparison-gallery-precision/images/bottoms-python.png).
These are generated outputs, not error maps. [Image notices](NOTICE.md) apply.

| Item | Revision / SHA256 |
|---|---|
| Upstream source | `7c0f10af3f91ad4048fe9729c470a13ef905d25a` |
| Original checkpoint | `d6cd38286885bc29fa487ea9383f80ffeb95862e7747c630d42c5d3c05bdd35a` |
| Cardigan measurement JSON | `7d85aaca0f1272fba3f4165f37a0e4ee3d5f656aec8e34f399162c166b9e951a` |
| Bottoms measurement JSON | `58493538abb6bf0f510911306c876f33cb328eba6d3b4078cdfaf85891f4b889` |
| Cardigan output PNG | `574ab54cb2d92bdef468cc3c9ba308346898f4660bf29dece21646f36be78646` |
| Bottoms output PNG | `0acfe49e5fac5904645c7b98473c6af99d0988bb7e7082e8ed118d43be14e1b4` |

The HTML's downloadable JSON contains `python-cardigan` / `python-bottoms`
rows with memory-scope notes and source-evidence hashes. Original private files
are `checkpoint27-upstream-{case}\result.json` and
`checkpoint27-upstream-{case}-measurement\measurement.json`; full private
experiment payloads are not bundled here.

See the [original experiment and instrumentation correction](quantization-and-upstream-comparison.md#actual-original-python-sampler-two-photographic-cases-complete).
Historical report snapshots retain their original chronology; this page and
the current comparison summaries make the corrected baseline explicit.
