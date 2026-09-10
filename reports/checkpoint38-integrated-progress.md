# Optimization checkpoint 8: integrated acceptance (complete)

Checkpoints 1-7 are complete, including the deliberate decision not to
promote CFG batching or image caches without a consistent measured benefit.
The final batch completed successfully: four short alternating controls and
six full 20-step recordings. No generation remains running or pending.
Root acceptance is `checkpoint38-integrated\results.json`, `passed=true`.
The [consolidated report](memory-and-latency-optimization-results.md) and
[ten-row comparison gallery](comparison-gallery-optimized/index.html)
contain final timings, source/phase memory, numerical errors and limitations.

## Final outcome

Warm cardigan CPU/BLAS medians were 60.9395 / 40.7543 seconds, a 33.12%
reduction. Full floating CPU/BLAS sampling was 2371.130 / 1598.136 seconds.
Their peak working sets were 2.0109 / 2.0281 GiB, but sampled private commit
was 2.1402 / 4.2843 GiB. Isolated DLL-only probes attributed approximately
1.88 GiB to 16-thread OpenBLAS startup before any model or GEMM.

Q8 full runs peaked at 1.4002-1.4016 GiB working set and mixed at
1.5261-1.5266 GiB, versus historical same-policy peaks of 4.15-4.30 GiB.
All four quantized runs preserve all 21 historical tensor files bit-for-bit
(84 comparisons) and exact final cropped pixels. No runtime weight
conversions or source mappings remain in these selected runs.

Floating CPU cardigan reproduces its historical native crop exactly.
BLAS passes every native floating trajectory gate (maximum relative L2
8.83848e-7); only 23 crop channels differ, each by one.
All predeclared working-set ceilings passed. These are not commit caps.
Quantized floating-quality gates still fail; public restrictions are unchanged.

## Scope clarification

Checkpoint 5 validated full BLAS trajectories on original example tops
(the black-shirt/worn-garment example) and the photographic bottoms case.
The original tops example is **not** the independent flat-lay cardigan
case. This final batch adds the actual cardigan case used in the prior
upstream and quantization comparisons.

There was no complete floating cardigan trajectory previously recorded.
The new CPU recording reproduces the historical native cropped PNG
exactly. That historical PNG already agreed with the original upstream
sampler within one byte, but this does not provide a Python intermediate
trajectory. The new BLAS recording is compared against the complete new
native floating trajectory with the unchanged numerical thresholds.

## Serial experiment sequence

1. Cardigan CPU/BLAS/BLAS/CPU controls: three forwards per fresh process,
   same BF16-ready weights, 39 modulation pairs, strict fused GELU,
   seed42 and 16 threads. Warm medians exclude first forwards.
2. Full 20-step floating CPU cardigan recording and exact comparison with
   historical cropped native pixels.
3. Full floating BLAS cardigan recording and all-step native-F32 comparison.
4. Runtime-ready Q8 bottoms, mixed cardigan, mixed bottoms and Q8 cardigan:
   full 20-step recordings with modulation precomputation and fused GELU.

The quantized runs preserve every recorded image and guided velocity
bit-for-bit against the original same-policy checkpoint-26/30 recordings.
They are not required or claimed to become floating-equivalent. The
four restored matrices remain frozen; no new calibration is performed.

All runs retain 20 steps, CFG1.5, shift1.5, skip1, seed42, 576x864 generation,
and the exact x0/y48/576x768 photographic crops. Declared per-process
working-set ceilings are 3 GiB for floating and 2 GiB for Q8/mixed, based
on the earlier accepted short controls; these are not private-commit caps.

## Evidence and failure handling

The orchestration script is `checkpoint38_integrated.py` in the session
artifacts. Results go to `checkpoint38-integrated`, with executable/model/
source/condition hashes, complete phase profiles, external process timelines,
measurements, native comparisons and contact sheets.

The script refuses automatic regeneration of a partial native run.
`--resume` reuses completed measured runs only if provenance and commands
match, allowing analysis to be repaired without repeating finished
generation. The batch stops on failed execution, unexpected runtime weight
conversion, memory-ceiling failure or output disagreement.

Earlier accepted floating bottoms/original-tops trajectories, loader
attribution, runtime-ready export checks, modulation/GELU controls and
service arena/budget results are linked in the final consolidated report
rather than falsely relabeled as new repeated observations.
