# Optimization checkpoint 2: runtime-ready weights

Checkpoint 2 is complete. Matching runtime-policy files remove conversion
scratch and duplicate source residency without changing the inference
arithmetic. No new quantization kernel or public quantized-inference support
was introduced.

## Implementation

Added `scripts\export_fashn_runtime_weights.py`, a reproducible wrapper around
the existing native converter and verifier, not a second model converter.
The wrapper builds exact suffix-anchored rules from a native 104-matrix map,
supports explicit F32 restorations and reuse of an existing file, defaults to
two conversion workers, and writes hash-linked completion/provenance records.
Failures leave an incomplete export rather than a success-shaped result.

Extended `test-fashn-vton-conversion` with `--runtime-policy` and the existing
`--f32-matrices` policy parser. It checks every shape and type, all protected
and restored values, and exact Q8 blocks against current native quantization
of the original floating weights. Existing all-F32/F16/BF16/selective-Q8
verification remains supported. BF16 expectations now explicitly apply BF16
rounding, also making the verifier correct for original F32 sources.

The phase-summary utility additionally checks that manifest controls match
the actual measured command and that native/external process IDs agree.
Documentation covers ready-file generation and the required matching load
policy. Public quantization/GPU guards remain unchanged.

## Files and exact-weight checks

All exports use the original official BF16 checkpoint. The previously
converted selective-Q8 file was reused rather than unnecessarily regenerated.

| Policy | File bytes | Inference payload bytes | Native verification |
|---|---:|---:|---|
| 104 BF16, protected F32 | 2,471,712,672 | 2,471,677,632 | 366 tensors, exact floating values |
| 104 Q8, protected F32 | 1,808,160,672 | 1,808,125,632 | 366 tensors, all 104 Q8 matrices byte-exact |
| 100 Q8, 4 restored F32, protected F32 | 1,942,919,072 | 1,942,884,032 | 366 tensors, all 100 Q8 matrices byte-exact |

The unused patch token is included in export verification but not among the
365 registered inference tensors. File overhead and that token account for
the difference between file size and registered payload.

BF16 export took 11.970 seconds with 85.625 MiB process peak working set.
Mixed export took 9.437 seconds with 141.059 MiB peak. These are one-time
conversion observations with two workers, not model inference latencies.

Negative controls rejected an all-BF16 file as a BF16-core/F32-protected
runtime file, and rejected restoring original F32 matrices from an already
quantized source. The legacy all-BF16 verification still passed all 366
tensors. Python rule/completion/summary tests passed; the environment-gated
native invalid-argument test was skipped in that final Python invocation,
having been exercised explicitly during checkpoint 1.

## Actual warm memory

Same trajectory executable as checkpoint 1, same photographic bottoms inputs,
seed 42, 16 inference threads and two loading workers. Each mapped ready-file
run performs two identical conditional-t0 forwards. A fresh original-weight
mixed/non-mapped run supplies its same-policy control. Three additional
non-mapped ready-file runs cover load-only behavior.

| Warm two-forward run | Process peak WS GiB | Second graph-end WS GiB | Second graph-end private commit GiB |
|---|---:|---:|---:|
| Ready BF16/F32, mmap on | 2.9338 | 2.9262 | 0.7798 |
| Ready Q8/F32, mmap on | 2.3059 | 2.2982 | 0.7680 |
| Ready mixed, mmap on | 2.4313 | 2.4237 | 0.7683 |
| Original mixed, mmap off | 2.4313 | 2.4236 | 2.5777 |

For comparison, checkpoint 1's original-file mapped Q8 control peaked at
4.1161 GiB; the ready Q8 file peaks at 2.3059 GiB, approximately **44% less**.
The original-file mapped BF16 control peaked at 3.4256 GiB; ready BF16 peaks
at 2.9338 GiB. These comparisons use the same short-forward workload and
binary, not the older full-trajectory recorder numbers.

The ready Q8 policy is approximately **0.628 GiB below ready BF16** in both
process peak and second-forward working set. Mixed costs about 0.125 GiB
more than all-Q8, consistent with restoring four original F32 matrices.

All six ready-file runs report **zero converted tensors and zero accounted
read/F32 conversion scratch**. With mmap on, allocated parameter-buffer bytes
are zero and directly mapped parameter payload equals the entire runtime
payload. There is only the matching mapped representation, not original BF16
pages plus converted weights. The source pages are genuinely resident after
forwards; this is not an untouched-mapping/load-only artifact.

Non-mapped ready-file load-only peaks are 2.3137 GiB BF16, 1.6958 GiB Q8 and
1.8213 GiB mixed. These are explicitly not inference peaks.

Private commit is not private working set. In particular, the near-zero
private-commit samples after workspace release are not the warm inference
numbers shown above. File-backed mappings do still consume resident RAM;
they are not "free memory." Process-wide backend/allocator effects beyond
the instrumented buffers remain separately unaccounted rather than hidden
in a fictitious total.

## Numerical behavior and timing

All three ready-file velocity outputs are finite and **bit-identical** to
their original-checkpoint same-policy controls across 1,492,992 values.
The mapping/source-format change does not alter floating, all-Q8 or mixed
arithmetic.

Instrumented warm graph intervals: BF16 62.870 seconds; Q8 50.587 seconds;
mixed 50.158 seconds. The matched original mixed warm interval was 50.847
seconds. These are single, instrumented observations and do not establish
a statistically significant speed improvement from file conversion.
The win here is memory ownership/loading, not a new denoising kernel.

No complete 20-step image was rerun in this checkpoint. Existing Q8/mixed
versus F32 trajectory failures remain failures; same-policy bit identity
does not waive those gates. Integrated full-image measurements remain
checkpoint 8.

## Persistent evidence and next decision

- `checkpoint32-runtime-weights\bf16\model.gguf`
- `checkpoint16-conversion\model-q8_0.gguf` (reused)
- `checkpoint32-runtime-weights\mixed\model.gguf`
- Each export directory: `export.json`, conversion/verification logs.
- `checkpoint32-memory-controls`: seven commands, native profiles/outputs,
  external measurements/timelines and logs.
- `reports\checkpoint32-memory-summary`: hash-linked summary and bitwise parity.

The preferred ready-file path fully addresses the measured duplication and
conversion scratch. General loader ownership/chunked-conversion changes are
therefore not justified for this checkpoint. The existing two-worker,
non-mapped original-checkpoint path also already has a lower-memory profile.
Avoiding unnecessary cross-model loader changes preserves ownership safety.

Checkpoint 3 will target modulation weight residency. It must actually avoid
retaining/faulting the large modulation matrices, not merely add a cache on
top of existing allocations.
