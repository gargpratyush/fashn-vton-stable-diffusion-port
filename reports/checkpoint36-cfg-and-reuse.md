# Optimization checkpoint 6: CFG and constant-reuse evaluation

Status: evaluation complete; no new runtime optimization promoted.
Sequential B=1 CFG remains the supported low-memory path. Whole-model B=2
inference was deliberately not implemented or claimed as validated.

## Why batching was not adopted

The supported BLAS backend loops over tensor batch dimensions and calls
SGEMM separately for each batch member. Merely widening B=1 to B=2 does not
turn those calls into a larger fused GEMM. Flattening batch into the token
axis at linear operations is a separate possibility; it must be reversed
before attention and conditioning operations.

Added reproducible `test-fashn-math --cfg-benchmark` controls with independent
branch values. They compare two sequential multiplies, a true batch
dimension, and batch-flattened token matrices, followed by another sequential
control. Each configuration has one cold plus three warm runs, 16 BLAS
threads, and an explicit 1 GiB **individual matrix workspace** ceiling.
That ceiling is not a whole-model or process-memory budget.

| Matrix (K,M,N per branch) | Sequential warm medians, before/after s | B=2 median s | Flattened median s | Sequential / paired workspace bytes |
|---|---|---:|---:|---|
| 1280,5120,3456 | 0.2134 / 0.2331 | 0.2446 | 0.2155 | 114688064 / 203161664 |
| 1280,8960,6912 | 0.7722 / 0.7086 | 0.7704 | 0.7666 | 328990784 / 612106304 |
| 6400,1280,6912 | 0.4608 / 0.4806 | 0.4766 | 0.4210 | 245104704 / 457441344 |

All outputs were bit-identical between modes. Only the flattened final
projection showed a clear local gain; it was not a consistent win across
the dominant matrices. Workspaces increased approximately 1.77-1.87x.
These measurements include graph/input/output overhead and use synthetic
values; they do not establish full-CFG wall time.

## Attention pairing

Added `--cfg-attention-benchmark`: two distinct synthetic branches at both
actual token counts, sequential/paired/sequential controls and four repeats.
Every branch output was bit-identical, demonstrating isolation in the
attention primitive rather than claiming complete model batch support.

At 6912 tokens, sequential warm medians were 1.1904 / 1.1750 seconds;
paired was 1.2247 seconds. Workspace doubled from 141557824 to 283115584
bytes. There was no attention throughput justification for doubling
activation residency. Raw 3456-token and all individual measurements remain
in `checkpoint36-cfg-attention.log`.

## Measured graph overhead and cache tradeoffs

The phase instrumentation from checkpoint 5 measures approximately
19.9-26.5 ms graph construction against 42-second warm BLAS forwards:
roughly 0.05%. Eliminating graph construction entirely would not explain a
material speedup. Native workspace reservations already persist.

The garment patch projection is only one of 107 denoising matrix multiplies.
Its smaller 576-input-width projection should not be assigned the timing
of a core 1280x8960 operation. Even the larger target patch benchmark took
only 27-31 ms with BLAS; this is scale evidence, not a measured garment
timing. Caching two 1280x3456 F32 garment embeddings would add 35,389,440
bytes of image-specific state plus invalidation/ownership complexity.
No extra image cache or skipped validation was introduced for this
unproven sub-percent opportunity.

Static RoPE values are already generated once by the runner; deeper
garment K/V and hidden states remain image/time dependent and must not be
reused between steps. Conditions remain validated as before. Request
cleanup and sensitive-image retention semantics therefore remain unchanged.

## Explicit remaining requirements if B=2 is revisited

The current model still has B=1 validation, category reshaping, cached
modulation views, patch-capture reshaping and sampler/output assumptions.
A future full implementation must fix all those together, audit position
broadcasting and per-example attention, supply an actual whole-graph memory
budget, and run complete branch/cancellation/sample-count/reference gates.
The primitive evidence here does not waive any of those requirements.

The existing sampler regression passed unchanged, including CFG/skip
branch counts, normalized-zero null conditions, observer failure,
cancellation and runner cleanup. The default path and RNG semantics were
not changed. No new complete images were generated for rejected candidates.
