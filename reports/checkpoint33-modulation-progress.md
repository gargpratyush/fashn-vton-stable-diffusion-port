# Optimization checkpoint 3: modulation residency (complete)

Checkpoint 3 is complete. Fixed-schedule, windowed public generation and
complete-trajectory acceptance have passed. The server has been relinked.
The optimization remains explicitly opt-in and does not enable public Q8.

## Final acceptance

- The two-sample public 1 MiB windowed request produced exactly the same PNG
  pixels as the uncached request for both distinct samples. Its forced
  mid-CFG and next-sample refills preserve the retained conditional result
  and the single logical RNG batch.
- Public uncached/cached peaks were 2.967876/1.978584 GiB; wall times were
  269.478/281.269 seconds. The deliberately tiny cache saves memory but is
  approximately 4.4% slower here because of repeated precomputation.
- Complete 20-step cached floating inference took 2577.729 seconds
  (42m57.7s), including precomputation and trajectory recording, and peaked
  at **2.009415 GiB**.
- Every reference image and guided-velocity gate passed. Maximum relative
  L2 was 0.000002016990 for images and 0.000002450617 for velocities.
  Schedule error was zero; initial-noise max abs was 0.000000476837.
- Full-canvas PNG agreement with the original reference differs in only
  43 channels, each by one byte level (93.537 dB numerical PSNR).
- Additional direct-versus-cached tensor checks were bit-exact for all
  37 modulation projections and conditioning captures at times 0, 0.4,
  0.8 and categories 0, 1, 2, 3. This specifically covers null, tops,
  bottoms and one-pieces, not only the photographic bottoms trajectory.
- Real cancellation/retry/reuse and public option-validation tests passed.
  The relinked server uses the same validated C API path; a new server soak
  is reserved for the service checkpoint.

This is not evidence of a statistically significant full-generation speedup;
the main gain is avoiding approximately 0.972 GiB of persistent parameters.

## Completed evidence

Same rebuilt executable, runtime-ready weights, mmap off, two loading workers,
16 inference threads, identical photographic bottoms conditions and seed.
Each case executes two conditional-t0 forwards. Cached cases additionally
precompute all 39 time/category pairs of the actual 20-step/CFG1.5/skip1 plan.

| Policy | Ordinary peak WS GiB | Cached peak WS GiB | Cached second-forward WS GiB |
|---|---:|---:|---:|
| BF16 core / F32 protected | 2.933815 | 1.975945 | 1.968006 |
| Q8 core / F32 protected | 2.305897 | 1.363804 | 1.357250 |

Cached outputs are bit-identical to their same-binary uncached controls for
both policies. All 34 conditional/null floating reference captures pass;
maximum capture relative L2 is 0.0000099220123.

The cache contains 30.659 MiB: projected modulation vectors plus the original
time/category/combined-vector debug captures. The corresponding inactive
registered parameters occupy 0.972462 GiB. Those weights are released between
precompute layers and are not allocated for denoising. This is an actual
residency reduction, not a cache added alongside all original weights.

Warm instrumented intervals were 67.656/67.129 seconds without/with cache for
BF16 and 51.461/51.120 seconds for Q8. These are single observations; no
significant denoising-speed improvement is established. This checkpoint
primarily saves memory.

## Implemented since the fixed-cache experiment

- Bounded request windows, default 128 MiB; diagnostics can force 1..128 MiB.
  Larger schedules retain only a window of projected values. The low-level
  single-window API rejects excessive allocations before loading weights.
- Exact time-bit/category keys; instance/thread/arithmetic matching; adapters
  remain unsupported. Missing keys outside the configured request fail.
- Cancellation polling inside precomputation, partial-cache cleanup, cache
  reuse and explicit cleanup of request-scoped callbacks.
- Shared construction of the requested CFG/schedule pairs for API and tests.
- Explicit public opt-in through existing model arguments:
  `{"fashn_modulation_cache":true,"fashn_modulation_cache_mib":128}`.
  Unknown/malformed options, mmap and eager-load conflicts are rejected.
  The default remains unchanged; no C ABI fields were added.
- Public CPU-only/floating-only restrictions are preserved.

Native layout/key/budget/schedule checks pass. The real-weight regression
cancels after conditioning computation, confirms zero assigned weight
storage and an empty partial cache, then retries successfully and confirms
same-key reuse. It also exercises public configuration acceptance and
rejections. Schedule construction covers 20, 30, 50 and 1000 steps; this is
not a claim that a complete 1000-step inference was performed.

## Completed acceptance batch

`checkpoint33-public-and-trajectory`:

1. Ordinary public CLI: one Euler step, CFG1.5/skip0, two samples.
2. Public CLI with a 1 MiB cache: the same request, forcing refills between
   CFG branches and again for the next sample.
3. Assert both cached PNGs exactly match their ordinary counterparts and
   that the two logical samples differ.
4. Record a complete cached 20-step native trajectory and compare every
   image/velocity state to `checkpoint25-bottoms-reference`.

The batch completed successfully. Measurements, commands, timelines
and hashes persist beneath that directory. No other model experiment runs
concurrently. Shell `fashn-opt-cp3-full-validation` has finished.

Its parity reports and full-trajectory measurements have been inspected.
The server was rebuilt against the updated static library.

Completed short-run artifacts: `checkpoint33-modulation-controls`,
`checkpoint33-modulation-graph`, `reports\checkpoint33-modulation-summary`.
Cancellation/configuration evidence: `checkpoint33-cancellation-reuse.log`.
All-category/time tensor evidence: `checkpoint33-categories-parity.log`.
