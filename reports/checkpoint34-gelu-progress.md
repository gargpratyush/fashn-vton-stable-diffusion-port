# Optimization checkpoint 4: strict-F32 GELU fusion (complete)

Checkpoint 4 is complete. The fused operation remains an explicit opt-in:
the observed whole-model speed improvement is small, and memory is unchanged.

## Final acceptance

- Q8 same-policy probe velocities were bit-identical to the earlier unfused
  cached output. Public Q8 inference remains blocked.
- Both samples of the public fused/1 MiB-window request were pixel-identical
  to the prior ordinary public request. It took 279.408 seconds and peaked
  at 1.978611 GiB.
- Full 20-step fused floating inference completed in 2507.495 seconds
  (41m47.5s), peaking at 2.012066 GiB.
- Every original reference image/velocity gate passed. Maximum relative L2
  remained 0.000002016990 / 0.000002450617, respectively.
- All 21 native recording files (initial noise and twenty image/velocity
  states) were **byte-identical** to checkpoint 3's unfused cached recording.
  Full-canvas reference PNG differences remain 43 channels, all by one level.
- The full fused run was about 2.7% faster than the earlier unfused cached
  observation. Those full runs were not interleaved repeats; use the modest
  1.22% A-B-B-A warm-forward observation, with overlapping ranges, rather
  than claiming a universal 2.7% improvement.

The batch wrapper exited with code 1 only after successful generation and
original-reference comparison: it incorrectly passed a native manifest to
the Python-reference-only comparison helper. That helper correctly rejected
the missing reference artifact manifest. A separate native-recording
comparison checked settings, prior accepted reference hashes and all 21
files directly. It passed exactly; **no image generation was repeated**.
The original comparator and all numerical thresholds were left unchanged.
Correction evidence: `checkpoint34-full-acceptance\native-recording-comparison.json`.

## Completed model controls

The A-B-B-A controls completed successfully. All four final velocities were
bit-identical. Both conditional/null original-reference branches passed.
The graph changed from 2948 to 2696 nodes, consistent with replacing 36
eight-operation GELUs by single custom operations.

Warm forward median was 66.4127 seconds ordinarily and 65.6049 seconds fused,
an observed 1.22% difference. Ordinary samples ranged 65.7173-67.3212 seconds;
fused samples ranged 64.9375-66.9343 seconds. The ranges overlap and only two
processes per configuration were measured. This is a modest candidate
improvement, not a statistically strong or universal speed claim.

Every run retained the same 600,355,904-byte runtime workspace. Fusion did
not reduce the measured whole-model arena either.

The strengthened mixed-sign stride and analytic tests have now been rebuilt
and passed. Additional microbenchmark output is in
`checkpoint34-mixed-sign-micro.log`.

The explicit public opt-in now accepts `"fashn_fused_gelu":true` alongside the
existing `fashn_modulation_cache` boolean in model arguments. Defaults are
unchanged; malformed flags are rejected. CLI/server have been relinked, and
the public option guards/projection tests passed.

## Completed acceptance batch

Shell `fashn-opt-cp4-full-acceptance` runs serially:

1. A cached Q8 fused two-forward control, requiring bit-identical output to
   the previous same-policy unfused control.
2. A two-sample public request combining fusion and a 1 MiB modulation window,
   requiring pixel-identical results to the earlier ordinary public request.
3. A full 20-step fused floating trajectory, compared both to the original
   reference and to checkpoint 3's unfused cached native trajectory.

Artifacts are in `checkpoint34-full-acceptance`; results and measurements
have been inspected, including the corrected native-recording comparison.

## Implementation

`src\core\strict_gelu.h` uses the existing public GGML custom-operation API.
It preserves the tanh GELU formula and rounded F32 intermediate boundaries;
it does not use the rejected FP16 lookup or a new tanh approximation.
Rows are partitioned across the requested CPU threads. Addressing respects
all four tensor strides, including a strided innermost dimension and the
5120-wide MLP slice of an 8960-wide QKV/MLP projection.

The switch requires both the existing strict-F32 GELU context and the new
explicit fused flag. Other model defaults remain unchanged. Native graph
and trajectory diagnostics expose `--fused-gelu`; public generation requires
the explicit experimental model-argument opt-in described above.

The first implementation used volatile F32 intermediates everywhere. It
passed the 2e-6 analytic/primitive tolerance but was slower on the dense
microbenchmark. The current VS2022 implementation uses a scoped
`float_control(precise,on,push)` / `pop`; VS2022 also disables contraction
inside that scope and restores the previous setting afterward. Other
compilers retain explicit volatile rounding barriers.

Authoritative compiler behavior:
https://learn.microsoft.com/en-us/cpp/preprocessor/float-control?view=msvc-170
https://learn.microsoft.com/en-us/cpp/preprocessor/fp-contract?view=msvc-170

## Microbenchmarks completed

16 threads, one cold plus three warm repetitions, real matrix dimensions.
These measure the runner including graph construction and output copying,
not an isolated arithmetic kernel. Uniform inputs are synthetic, not an
assertion about actual model activation distributions.

| Shape | Input radius | Ordinary warm median s | Scoped-precise fused warm median s |
|---|---:|---:|---:|
| Dense 5120 x 3456 | 1 | 0.0841 | 0.0595 |
| Dense 5120 x 3456 | 12 | 0.0832 | 0.0635 |
| 5120 x 6912 slice in 8960-wide QKV/MLP storage | 1 | 0.1879 | 0.1422 |
| Same strided slice | 12 | 0.2131 | 0.1371 |

**No arena saving was observed in these microbenchmarks.** Both variants
use 141,557,824 bytes for the dense case and 389,283,904 bytes for the
strided case. The original graph already reuses elementwise intermediates;
counting each source-level operation as another simultaneous full activation
would overstate its memory cost.

GGML currently schedules its ordinary TANH unary operation with one task;
the custom operation uses the available thread count. This, fewer graph
operations, and fewer memory passes are candidate explanations, not a
validated decomposition of the complete model's latency.

Raw evidence: `checkpoint34-gelu-micro.log` (initial volatile version),
`checkpoint34-gelu-precise-micro.log` (scoped-precise version).
The primitive covers analytic accuracy, row gaps, transposes, 4D strides
and 1/16 thread settings. Its small-tensor data generator was subsequently
strengthened to mix signs throughout every small view, rather than allowing
some tiny views to contain only saturated negative outputs. That strengthened
revision was rebuilt and passed after the short model batch; its separate
microbenchmark log is retained rather than relabeling the earlier binary.

## Completed short model acceptance

`checkpoint34-forward-controls` runs A-B-B-A in fresh processes:
ordinary, fused, fused, ordinary. Each precomputes the same 39 modulation
pairs, then performs three conditional-t0 forwards on identical inputs.
Native page/profile scanning is disabled. The first forward is startup;
the remaining two are warm. The diagnostic manifest records those intervals
and retained workspace bytes. External process memory is still sampled.

Then the fused cached model ran both branches against the original
photographic reference captures. The batch checks finite same-policy
velocity comparisons before proceeding. Shell:
`fashn-opt-cp4-forward-controls`.

`parity-and-timing.json`, per-run measurements and
`checkpoint34-fused-graph\comparison.json` have been inspected. The strengthened
primitive and full-trajectory acceptance have passed. Further normalization/
residual fusion was not introduced speculatively: measured GELU arena reuse
already eliminates the assumed large temporary-memory win. The retained
change reduces memory passes and parallelizes accurate math, without a new
tanh approximation or a claim of additional arena savings.
