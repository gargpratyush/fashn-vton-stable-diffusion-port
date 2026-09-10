# Optimization checkpoint 7: preparation and service memory (complete)

Checkpoint 7 is complete. Lazy parsing, ORT configuration experiments,
opt-in arena disabling and aggregate image-storage budgets passed the final
combined service acceptance.

## Final combined service acceptance passed

All seven selected checks passed without skips. Raw/prepared output pixels,
preparation cancellation, two-sample behavior, sixteen concurrent-client
cancellations and graceful shutdown remained correct.

The four-round arena-off service ended at 3,179,429,888 bytes of private
commit versus 4,707,848,192 in the earlier arena-on baseline: approximately
1.42 GiB less at that boundary. These are current private-commit snapshots,
not resident-memory or peak-working-set measurements.
The post-warmup adjusted span was 884,088 bytes (0.84 MiB), within the
unchanged 128 MiB short-soak budget.

The byte-budget test admitted 24 queued four-sample jobs alongside an active
job, then rejected further admission before the 64-pending count limit.
Reserved storage was 516,912,790 bytes under the 536,870,912-byte ceiling.
Queued cancellation released capacity, a replacement request was admitted,
and final reservations fell to 10,532,502 bytes for the remaining stored
results/empty-string capacities. Existing results were not evicted early.

Graceful shutdown took 74.67 seconds, again reflecting whole-forward CPU
cancellation granularity rather than immediate kernel interruption.
Evidence: `checkpoint37-server-budget-soak\report.json`.

## Lazy-parser service baseline passed

All six selected service checks passed, with no skips: capabilities/startup,
14 malformed raw cases, exact raw/prepared output agreement, preparation
cancellation, four soak rounds, two-sample behavior, 16 unique cancelled
jobs across eight clients and graceful shutdown.
After excluding two warmup requests and approximately subtracting retained
result lengths, the private-memory span was 1,224,056 bytes (1.17 MiB),
within the unchanged 128 MiB budget. This is a short stability check,
not proof of absence of leaks. Graceful shutdown took 74.77 seconds because
an active CPU forward finishes before cancellation is observed.

Evidence: `checkpoint37-server-lazy-soak\report.json`.

## ORT experiments completed

Each configuration used the same four-request lifecycle: flat-lay/free,
worn/free, flat-lay/masked, then repeated worn/free. All sixteen PNGs in
every variant were identical to the eight-thread control.

| Configuration | Final current working set MiB | Repeated worn preparation s |
|---|---:|---:|
| 8 threads, default arenas/spinning | 1597.73 | 2.316 |
| 8 threads, no spinning | 1590.82 | 2.460 |
| 4 threads, no spinning | 1587.00 | 2.765 |
| 1 thread, no spinning | 1579.74 | 7.167 |
| 16 threads, no spinning | 1596.38 | 2.293 |
| 8 threads, no arena/no spinning | 662.18 | 2.564 |
| Repeated default control | 1595.81 | 2.439 |
| Isolated no-arena, original spinning/8 threads | 661.14 | 2.564 |

The isolated no-arena variant peaked at 1025.75 MiB over its lifecycle.
Its approximately 935 MiB lower warmed residency is distinct from peak
process memory. Small timing/working-set differences among the thread
variants are not sufficient grounds to change defaults. One thread was
clearly slower. No shared global ORT thread pool was introduced.

Selected option: disable CPU arenas explicitly, preserving the original
eight-thread/inter-op-one/spinning settings. Native CLI exposes
`--ort-no-arena`; the server exposes `--try-on-ort-no-arena`.
Defaults remain unchanged. The selected variant subsequently passed all
twelve original-Python preparation modes, transform/mask primitives and
failure contracts (3 test methods, 119.428 seconds).

Evidence: `checkpoint37-ort-comparison.json`,
`checkpoint37-ort-no-arena-spinning8`,
`checkpoint37-no-arena-reference`.

## Aggregate service image budget implemented

The server now reserves from 512 MiB of image storage before admitting a
try-on job, separately from the existing pending-count limit. It charges
decoded input payloads and base64 capacities, a raw-to-prepared transition
allowance, and up to 4 MiB of string capacity per requested output.
The encoder enforces that output-capacity bound.

Active reservations remain stable while the worker changes input storage,
avoiding concurrent inspection of mutable request buffers. Completion
replaces the reservation with retained result capacities; cancellation and
failure release input/output storage. Expiry naturally removes the job's
charge. Raw strings are explicitly swapped empty to release their capacity
portably, rather than relying on assignment of an empty string.

Admission uses subtraction-based capacity checks and returns HTTP 429 on
budget exhaustion. Capabilities report the fixed ceiling and current
reservations under the manager lock. The existing 64-pending limit, TTLs,
cancellation ordering and public model restrictions remain.

Scope is explicit: this is an image-storage budget, not a total-process
cap. Model/workspaces, transient HTTP copies and metadata are outside it.

## Combined final service acceptance execution

Completed shell `fashn37-budget-soak` combined no-arena preparation and the new budget.
In addition to the earlier raw/cancellation/pixel/four-round/concurrency/
shutdown checks, it fills the image reservation budget with queued
four-sample requests, checks rejection before the count limit, cancels them,
and requires successful readmission and released reservations.
Artifacts: `checkpoint37-server-budget-soak`.

## Implemented lazy parser

- Configured parser capabilities remain available immediately.
- Startup consent, manifest, ONNX and license checks are preserved.
- The parser ONNX session is not created for segmentation-free flat-lay
  requests. Detector/pose sessions retain existing startup behavior.
- The configured directory is resolved at startup. Before first parser
  session creation, ONNX and license hashes are checked again against the
  already validated manifest, preventing delayed use of changed artifacts.
- The first parser-dependent request loads the session once; later requests
  reuse it. No image-specific cache is added.
- Cancellation is checked before preparation and after delayed loading.
- Actual ONNX session-construction failures are intentionally deferred to
  first use; valid artifact hashes do not prove a runnable ONNX graph.
- An explicit C++ eager-construction control is retained for diagnostics.
  Public native/service defaults use lazy construction.

## Measured lifecycle

Same pinned inputs and native sessions; separate eager and lazy processes.
These are single observations, not interleaved repeated timing statistics.

| Boundary | Eager working set MiB | Lazy working set MiB |
|---|---:|---:|
| Startup ready | 837.17 | 393.64 |
| After parser-free flat-lay preparation | 943.06 | 496.50 |
| After first worn-garment preparation | 1563.58 | 1550.88 |
| After masked-person preparation | 1607.00 | 1592.33 |
| After repeated worn-garment preparation | 1605.36 | 1592.87 |

Lazy initialization saved approximately **444 MiB at startup** and
**447 MiB after a parser-free request**. Startup took 2.35 seconds rather
than 4.50. This defers work, not removes required work: first worn-garment
preparation took 4.05 seconds lazily versus 2.63 with an already loaded parser.
Once parsing is required, the large memory saving disappears as expected.
Small later working-set differences must not be advertised as additional
proven savings.

## Correctness and integrity

All sixteen prepared PNGs from four eager/lazy requests were pixel-identical;
repeated worn-garment requests also matched exactly. Configured-versus-loaded
states and cancellation before loading behaved as expected.

A separate test creates an isolated license copy and a read-only-used hard
link to the model, validates startup, then changes only the copied license.
First parser use fails before loading the session. The original model/license
are untouched, and the temporary fixture links/files are removed afterward.

The existing original-Python preparation comparison then passed all twelve
category/photo/masking modes, transform/mask primitives and failure contracts:
3 test methods, 135.606 seconds. No numerical or pixel tolerance was relaxed.

Evidence:
`checkpoint37-parser-comparison.json`,
`checkpoint37-parser-eager`, `checkpoint37-parser-lazy`,
`checkpoint37-parser-tamper-cleanup`,
`checkpoint37-reference-preparation\report.json`.

## Earlier lazy-parser service validation

The rebuilt lazy-parser server completed raw-input capability, startup
consent, malformed-input, preparation cancellation, exact output and
four-round retained-memory/concurrent-client/graceful-shutdown checks.
The service ran serially with no competing model experiment.
Shell: `fashn37-server-soak`.
Artifacts: `checkpoint37-server-lazy-soak` and its external log.

This was the lazy-parser service baseline before adding aggregate byte
admission budgets. Existing pending-count/TTL/input-release behavior remains
unchanged at that point; the later budget integration is described above.
