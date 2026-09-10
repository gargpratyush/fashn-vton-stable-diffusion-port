# FASHN memory and latency: investigation and implementation plan

Execution update: the eight-checkpoint optimization/evaluation pass is now
complete. See [measured results and implementation decisions](memory-and-latency-optimization-results.md)
and the [final comparison gallery](comparison-gallery-optimized/index.html).
The original proposal below is preserved; its estimates and future-tense
options are not claims that every candidate was subsequently implemented.

Status: plan only. No inference implementation, loader changes, builds,
package installations or new model experiments were performed for this plan.
Investigation used the current source, completed measurement artifacts,
the recorded matrix-size inventory and the installed PyTorch build configuration.

## 1. Executive finding

Quantization does reduce our weight payload. The current diagnostic loading
path does not fully realize that saving in process memory:

1. It always enables mmap of the original BF16 checkpoint.
2. BF16 core weights can point directly into that mapping.
3. Q8 weights cannot: the loader allocates new Q8 destination buffers while
   retaining the original BF16 mapping.
4. On-the-fly BF16-to-Q8 conversion also allocates full-tensor BF16 read
   buffers and F32 intermediate buffers, potentially across16 loading workers.
5. Activations, attention and most intermediate arithmetic remain F32.

This is a memory-ownership/loading issue, not evidence that Q8 intrinsically
requires more memory than BF16. The source-backed mechanisms are confirmed;
their exact contributions to the historical high-water marks are not yet
measured separately. Phase-resolved profiling is the first checkpoint.

There is also an important broader optimization: approximately0.964GiB of
protected F32 modulation weights produce vectors depending only on timestep
and category. For the existing20-step/CFG configuration, their outputs need
only about30.1MiB. Precomputing those outputs and excluding their weights from
denoising residency could save substantially more than small pipeline copies.

## 2. What the weight arithmetic actually predicts

Derived from the completed104-row matrix-size inventory and the released
architecture. The unused432-element `patch_mixer_token` is excluded.
Numbers below are raw registered tensor payload, not GGUF file sizes or
process-memory forecasts; alignment, headers, packing and workspaces are extra.

| Quantity | Value |
|---|---:|
| Registered parameter elements | 971,813,808 |
| Eligible104-matrix elements | 707,788,800 (72.83%) |
| Protected parameter elements | 264,025,008 |
| Protected F32 payload | 1,056,100,032 bytes (0.984GiB) |
| Eligible BF16 payload | 1,415,577,600 bytes |
| Eligible Q8_0 payload | 752,025,600 bytes (717.188MiB) |
| BF16-core/F32-protected total | 2,471,677,632 bytes (2.302GiB) |
| Q8-core/F32-protected total | 1,808,125,632 bytes (1.684GiB) |
| Four-matrix-restored total | 1,942,884,032 bytes (1.809GiB) |

Q8_0 uses34 bytes per32 weights, including its block scale:1.0625 bytes per
weight, not exactly1. Compared with the current BF16/F32 policy, all-Q8
saves663,552,000 bytes, or0.618GiB/26.85% of parameter payload. It cannot
promise a2x or4x reduction in total process memory with this protected policy
and F32 activations.

The observed memory differences are consistent with adding converted weights
without dropping their mapped source:

| Difference | Observed peak-working-set increase | Corresponding destination payload |
|---|---:|---:|
| All-Q8 versus floating recorder | 707.027MiB | Q8 core weights:717.188MiB |
| Four-restoration mixed versus all-Q8 | 128.488MiB | Restoration increment:128.516MiB |

The close correspondence is strong supporting evidence, not an allocation
trace. Do not add independent phase peaks or assume every mapped page is
resident simultaneously.

## 3. Source-backed findings and confidence

### 3.1 Original source and converted destinations coexist

Relevant source in `C:\source\stable-diffusion.cpp`:

- `tests\fashn_test_runner.h`: `FashnGraphTestContext::init` explicitly calls
  `manager->set_enable_mmap(true)` and forwards inference threads to loading.
- `src\model_loader.cpp:793-852`: `process_model_files` creates a mapping
  and retains shared mapping/backend-buffer ownership in `file_data`.
- `src\model_loader.cpp:899-920`: `mmap_tensors` requires source and destination
  types to match before assigning a direct pointer into the mapping.
- `src\model_manager.cpp:359-404`: mapping is attempted first, then destination
  buffers are allocated for weights that cannot be mapped directly.
- `src\core\util.cpp:119-159,297-303`: Windows maps the whole file;
  mapped conversion input is read through `memcpy`.
- `src\model_loader.h`: the loader retains file-data ownership; there is no
  post-conversion source-only mapping-release method.

For BF16 core residency, the checkpoint is the weight storage. For Q8
residency, it is still a retained conversion source and there are additional
Q8 destination allocations. Mapped address space is not synonymous with
resident RAM, but touched source pages can contribute to working set.

The public CLI defaults to mmap **off**
(`examples\common\common.h:172`, `src\stable-diffusion.cpp:3657`).
The historical trousers CLI command did not pass `--mmap`.
Consequently its2.957GiB and the floating recorder's3.461GiB are not merely
the same loader with extra trajectory files: their mapping policies differ.
The three matched recorder experiments do share the same mmap policy.

### 3.2 Conversion has avoidable large temporary allocations

`src\model_loader.cpp:1020-1050,1146-1203` creates per-worker read buffers,
up to the configured loading-thread count. `convert_tensor` at96-150 uses
a complete F32 intermediate for BF16-to-Q8 conversion.

One1280x8960 single-block linear1 matrix needs approximately:

-21.875MiB BF16 read buffer;
-43.750MiB temporary F32 buffer;
-its final Q8 allocation, separately.

Sixteen simultaneous conversions of this shape would require1,050MiB of
read/F32 scratch alone. This illustrates the scaling of the existing code;
it is not a claim that this exact concurrency occurred at the measured peak.
Worker buffers are released when workers finish, so this can be a startup
high-water effect rather than a permanent live allocation. Heap retention
must also be distinguished from live C++ buffers.

### 3.3 The current measurements do not identify the peak phase

`scripts\benchmark_fashn_vton.py:133-195` retains the maximum kernel working
set and sampled private commit, but not a phase-tagged memory timeline.
Sampling occurs roughly every200ms. Current working-set samples are not
persisted, and the time at which the actual kernel high-water mark occurred
is not known.

Private commit is not resident private RAM. Subtracting sampled peak private
bytes from peak working set does not reveal mapped memory. Similarly,
system file cache and this process's mapped resident pages must not be
conflated. The corrected native PID attribution is not the old Python
launcher-measurement problem.

### 3.4 F32 activations remain large

The direct-Q8 runs have `upcast_matrices=false`; they do not intentionally
expand every Q8 weight back to F32 before each multiply.
The BF16/F32 path deliberately uses F32 weight casts:
`src\model\common\ggml_block.hpp:210-212`.

Nevertheless, Q8 matrix outputs, residuals, normalization and attention
inputs/intermediates remain F32. At6912 joint tokens:

- one5120-wide F32 MLP activation is135MiB;
- one8960-wide fused QKV/MLP projection is236.25MiB.

GGML Q8 dot products also quantize F32 activations into backend scratch:
`ggml\src\ggml-cpu\ggml-cpu.c:1480-1514,3012-3023`.
A6912x6400 activation would require about44.82MiB of Q8_0 scratch. Backend
workspace reuse means this must not be summed across all104 matrices.

### 3.5 Recording is not retaining20 full activation graphs

`tests\test_fashn_vton_trajectory.cpp` writes each image/velocity pair and
releases temporary copies. `sample_fashn_vton` does not accumulate all20
steps in RAM. Recording introduces copies and I/O, but is not a supported
explanation for a multi-GiB retained trajectory history.

### 3.6 Existing optimizations must not be counted as new work

- F32 flash attention is already implemented and exercised.
- `ComputeWorkspace` already reuses its allocation reservation when graph
  layout matches (`src\core\compute_workspace.cpp:113-127`).
- Graph metadata is rebuilt/freed each forward; the workspace is not simply
  reallocated from zero every forward.
- CPU ParamBackend weights remain resident across requests; `runner_end`
  does not indiscriminately unload them.
- Native already skips unnecessary null forwards:39 forwards for the
  current20-step/CFG1.5/skip1 configuration, rather than40.
- The native raw preparer already reuses detector/pose/parser sessions.
- LLAMAFILE at16 threads was already tested without a meaningful improvement.

## 4. Checkpoint 1: phase-resolved memory and latency accounting

Priority: P0. Establish where bytes/time go before changing behavior.

Extend the existing diagnostic and measurement infrastructure, not the public
generation contract, with explicit controls for mmap, loading threads,
eager-load-only, recording on/off, and short warm-forward probes.

Record markers for:

1. Process/context setup and metadata parsing.
2. Source mapping established.
3. Weight allocation/conversion begin/end.
4. First graph construction/reservation.
5. First conditional and null execution.
6. Later warm forwards and observer/serialization.
7. End of sampling, workspace release and model destruction.

Persist timestamped current working set, kernel peak working set, private
commit and thread count. Add internal counters for logical weights, mapped
weights versus source-only mappings, actual backend allocations, conversion
scratch, graph arena, persistent caches and planned CPU work size.
Distinguish estimates from actual allocator counters.

Use Windows `VirtualQueryEx`/mapped-file identification and
`QueryWorkingSetEx` at selected diagnostic boundaries to classify resident
pages. Do not scan all pages continuously during clean latency benchmarks.
If backend-internal allocations cannot be queried through supported APIs,
record that gap instead of labeling graph-arena bytes total memory.

Initial bounded matrix:

| Input file | Runtime policy | mmap | Loading workers |
|---|---|---|---|
| Original BF16 safetensors | BF16 core/F32 protected | on/off | 1,2,4,16 |
| Original BF16 safetensors | Q8 core/F32 protected | on/off | 1,2,4,16 |
| Runtime-ready GGUF | Matching BF16 or Q8 policy | on/off | bounded |
| Runtime-ready mixed GGUF | Fixed100Q8/4F32 policy | on/off | bounded |

Start with load-only and short forwards, not32 expensive complete runs.
After attribution, run complete images only for selected controls/candidates.
One model experiment at a time; do not flush global system caches on this
shared machine. Label fresh-process startup separately from cold filesystem
cache and warm resident-request performance.

Acceptance: reconcile the observed excess to named live allocations/resident
mapping categories and phase timing. Retain both startup peak and warm
steady-state memory; a lower final sample does not establish a lower peak.

## 5. Checkpoint 2: runtime-ready quantized loading

Priority: P0. First attempt should avoid new inference kernels.

### Preferred path: quantize once, load matching weights directly

The current converter already supports selective Q8 and tensor-type rules
(`src\convert.cpp:33-61`). The diagnostic runner accepts eligible Q8
checkpoint tensors. Therefore the first all-Q8 comparison can reuse the
existing converted GGUF rather than quantizing original BF16 at startup.

For mixed precision, export from the original checkpoint with exact rules
restoring the four selected matrices. The existing converter's type-rule
mechanism should be reused; verify the emitted types before inference.
Do not reconstruct original values by dequantizing an existing Q8 file.

Illustrative future conversion, not executed during this investigation:

```powershell
.\build\bin\Release\sd-cli.exe --mode convert --diffusion-model model.safetensors --type q8_0 -t 2 --tensor-type-rules 'single_blocks. (historical local-only artifact)[.]linear1[.]weight$=f32' -o model-mixed.gguf
```

Generate a matching runtime-ready BF16 control too:104 eligible BF16
matrices and protected F32 tensors, using explicit whitelist-derived rules.
An all-BF16 file is not that exact runtime policy: protected weights would
still be converted during loading.

Check365 registered tensors, exact shapes/types, protected tensors, original
values for the restored matrices, and Q8 blocks matching the existing
on-the-fly quantizer. Loading the mixed file with the same explicit four-name
policy should leave all selected matrices F32 and the other100 Q8.
Public CLI/C API/server quantized guards stay unchanged.

### Fallback/robustness path: bounded runtime conversion

- Separate loading concurrency from inference thread count.
- Use a byte-budgeted conversion queue, not merely a worker-count cap.
- Convert block-aligned row chunks with bounded F32 scratch, rather than
  materializing a whole F32 weight per worker.
- Avoid an extra complete read copy when a read-only source span can be
  safely consumed; otherwise use bounded reads.
- Give source-only mappings an explicit releasable lifetime after successful
  conversion. Retain mappings referenced by zero-copy live weights.
- Do not reset the entire loader to achieve unmapping: names, metadata,
  future loading and any shared ownership must remain correct.

Acceptance: identical converted parameter bytes and output behavior versus
the current diagnostic mode; lower actual memory, not just lower file size.
The goal is Q8 below the matching BF16 control in warm working set and
ideally process peak as well. Report a startup-only or steady-only gain
honestly if both do not improve.

## 6. Checkpoint 3: remove modulation weights from denoising residency

Priority: P1, high potential memory benefit; moderate implementation scope.

Confirmed dependencies:

```text
vec = time_embedding(t) + category_embedding(category)
modulation[layer] = Linearlayer (historical local-only artifact))
```

These operations do not depend on person, garment, pose or noisy image.
Relevant code:

- `src\model\diffusion\fashn_vton_model.h:68-83`;
- `src\model\diffusion\flux.hpp:396-426`;
- single/double blocks already have routes for supplied modulation outputs;
- `LastLayer::forward` at748-770 still computes its own modulation.

From the architecture:

| Modulation component | Weight elements |
|---|---:|
| 20 patch/single-block3840x1280 matrices | 98,304,000 |
| 16 double-stream7680x1280 matrices | 157,286,400 |
| Final2560x1280 matrix | 3,276,800 |
| Total | 258,867,200 |

Those weights alone occupy1,035,468,800 bytes, or0.964GiB in F32.
They account for about98% of protected parameter elements.
All modulation outputs require202,240 floats per time/category pair.
For39 forwards that is31,549,440 bytes, or30.088MiB, before bookkeeping.

Proposed execution:

1. Build only the required time/category pairs from the actual schedule/CFG.
2. Compute and retain their F32 modulation vectors.
3. Feed those vectors through existing block modulation arguments; add a
   FASHN-specific supplied-modulation route for the final layer.
4. Exclude modulation weight tensors from denoising's active parameter set.
5. Load/compute them layerwise and release their source ownership before
   denoising allocation, so startup peak can improve too.

Simply caching vectors while keeping every original weight allocated would
not provide the proposed memory saving.

Initially preserve the existing per-vector F32 computation ordering.
Batched schedule precomputation is a subsequent kernel experiment because
GEMV-to-GEMM changes can alter rounding. Cache keys include checkpoint,
precision policy, exact schedule/category, arithmetic implementation and
adapter epoch if adapters ever become supported. Keep caches bounded;
20/30/50/1000 steps must not create an unbounded precompute allocation.
Reuse within a multi-sample request without changing RNG semantics.

A complementary/lower-scope option is storing large protected weights as
BF16 and upcasting for F32 computation. Original checkpoint values are BF16,
so this can preserve their values; it must not downcast arbitrary F32
checkpoints silently. Modulation precomputation is preferable if it can
avoid their continued residency altogether.

Acceptance: matching modulation tensors, original floating graph/trajectory
gates, category0/null and all user categories, multiple schedules and
sample counts, cancellation during precompute, cache invalidation and
measured lower memory including load/precompute peaks.

## 7. Checkpoint 4: fused strict-F32 elementwise operations

Priority: P1, latency and activation-memory opportunity.

The FASHN F32 GELU workaround in `src\model\diffusion\flux.hpp:21-32`
constructs square, multiply, scale, add, tanh and gate operations separately.
This was necessary to avoid GGML's FP16-lookup behavior, but creates several
large intermediate tensors and memory passes.

The earlier intrusive8-thread profile reported:

| Operation family | Conditional | Null |
|---|---:|---:|
| Matrix multiplies | 46.22s | 45.50s |
| Flash attention | 19.93s | 19.29s |
| Tanh | 13.37s | 10.62s |

These node-synchronized times disable some optimizations and must not be
summed into a clean latency prediction or treated as current16-thread shares.
They do identify worthwhile kernel investigations.

Plan a FASHN-only fused F32 GELU using existing custom-operation facilities.
Preserve the mathematical approximation and controlled floating operation
order; do not switch back to the rejected FP16 lookup or silently enable
lower-accuracy fast math. Handle strided QKV/MLP views correctly.
First fuse memory passes with accurate math; then evaluate AVX2 vectorized
tanh independently. A scalar fused loop can still be slow.

Afterwards investigate fused normalization/modulation/residual operations
and unnecessary contiguous copies, guided by actual graph liveness.
Do not assume all135MiB GELU temporaries are simultaneously live.

Acceptance: existing primitive tolerances and complete floating trajectory
gates, actual tensor-stride coverage, measured graph-arena reduction,
clean16-thread latency improvement and no changes to other model defaults.

## 8. Checkpoint 5: CPU GEMM and attention backend comparison

Priority: P1, largest plausible throughput opportunity, benefit unmeasured.

Current native build:

- `GGML_BLAS=OFF`;
- `GGML_LLAMAFILE=OFF`;
- OpenMP/native CPU enabled.

Installed PyTorch2.8.0+cpu reports MKL2025.2, oneDNN3.7.1 and AVX2.
This is a concrete difference in available backend infrastructure, not proof
that a particular library alone explains the observed Python advantage.
The native profile and the existing Python/native gap justify examining it.

Plan:

1. Capture representative matrix shapes, strides, precision and call counts
   for patch, double-stream, single-stream and modulation operations.
2. Microbenchmark those real shapes with warm inputs, fixed thread budgets
   and identical arithmetic requirements.
3. Evaluate a supported BLAS-enabled native build and appropriate F32
   matrix paths; keep the original build as control.
4. Compare default Torch SDPA and GGML F32 flash at3456/6912 tokens,
   ten128-dimensional heads, and the actual QKV layouts.
5. Measure packing/transposition costs and backend workspaces, not merely
   kernel duration after hiding all preparation.
6. Audit OpenMP/BLAS thread ownership to avoid nested oversubscription.

Do not link opportunistically against arbitrary PyTorch DLLs. Use supported,
reproducible dependencies if the experiment later requires installation.
Do not make a Q8 "optimization" that silently expands the entire model into
F32 and loses the memory goal. No GGML or vendored source modifications are
part of this plan; any necessity for such changes requires separate scope.
Repeat LLAMAFILE only if a material shape/build change gives a new reason.

Acceptance: existing numerical gates for the floating path, explicit
memory-versus-throughput tradeoffs, short clean forwards first, then both
complete image cases for winning configurations. No promised2x speedup.

## 9. Checkpoint 6: CFG execution and constant-input reuse

Priority: P2, after weights/activation accounting.

Original Python batches conditional/null branches. Native currently uses
separate B=1 forwards and already skips the redundant last null forward.
Batched CFG could improve GEMM utilization, but can approximately double
activation demand; it is not automatically a memory optimization.

Evaluate sequential versus paired CFG under explicit memory budgets.
Changing this requires fixing all B=1 assumptions: input validation,
category reshaping, position broadcasting, attention batch layout, output
splitting and capture semantics. Batch dimension must not accidentally
become token dimension or permit cross-example attention. Keep sequential
CFG as the low-memory option if pairing costs too much.

Safe reuse candidates:

- garment patch embedding for fixed conditional and null inputs;
- static RoPE tensors and constant input bindings;
- graph metadata/execution plans for unchanged shapes and precision.

Existing workspace reservations are already reused. Graph construction and
full input validation run every forward; measure their actual fraction
before investing heavily in caching them. Validate immutable conditions
once per request while retaining noisy-state/time checks.

Do not cache deeper garment K/V or hidden states across steps: joint
attention makes them image/time dependent. Splitting the target projection
into noisy and static pieces changes F32 accumulation order, so treat that
as a separate numerical experiment, not an automatically exact cache.

Acceptance: branch parity, unchanged CFG skip/noise/seed rules, all supported
sample counts, no cross-request stale tensors, cancellation at least as
responsive as the chosen execution granularity allows.

## 10. Checkpoint 7: raw preparation and service memory

Priority: P2. Relevant to the complete application, not the already measured
prepared-only generation gap.

The preparer already retains ONNX sessions. However:

- every `Model` owns an environment/session with8 intra-op and1 inter-op
  threads (`examples\fashn-preprocess\preprocess.cpp:23-30`);
- a configured parser is eagerly loaded even if a request is segmentation-
  free flat-lay and does not invoke it (`pipeline.cpp:25-40,71-79`);
- graph workspace is released after a sample/request even while model
  parameters remain resident;
- pending jobs are count-limited, but count alone is not a precise total-byte
  budget when inputs/results differ in size.

Plan stage timings for decode, resize, detector, pose, parser, normalization,
inference, output conversion/PNG, queue wait and response transport.
Benchmark ORT thread/arena settings and idle-thread behavior; do not simply
run all models at16 threads or overlap preparation with16-thread generation.

Evaluate lazy parser session creation while preserving startup license and
artifact validation; shared/bounded preparation caches keyed by image hashes,
model revisions, category, photo type, segmentation mode and resize policy.
Keep sensitive-image caches memory-bounded and nonpersistent by default.
Add aggregate queued/result byte budgeting and consider a bounded warm
workspace policy with idle reclamation rather than unconditional retention.

Existing prepared-image reuse, input release, TTLs, session reuse and stable
soak results must not be misrepresented as missing or leaking.

Acceptance: exact native/reference prepared pixels across existing cases,
license behavior, startup failures, queue/cancellation/TTL contracts, no
cross-request stale results and repeated-use memory stability.
Report raw end-to-end latency separately from sampling and queue wait.

## 11. Checkpoint 8: controlled integration and acceptance

Run improvements independently before combining them. Keep three separate
optimization targets:

1. Lowest-memory validated floating inference.
2. Lowest-memory diagnostic Q8/mixed inference.
3. Highest-throughput configuration within an explicit memory ceiling.

For selected candidates, use serial fresh-process and warm-request repeats,
alternating control/candidate order, on both existing photographic pairs.
Start with short probes; reserve full20-step repeats for configurations that
show a meaningful advantage. Preserve seed42/CFG1.5/shift1.5/skip1 and all
existing crops; do not report fewer steps/resolution/CFG as equivalent
pipeline optimization.

Publish:

- logical payload, file size, source-mapped bytes and actual resident pages;
- startup peak, warm current working set, private commit and idle retention;
- loading, first-forward, steady-forward, sampling and raw end-to-end times;
- actual tensor types, executable/source/checkpoint hashes and thread settings;
- numerical trajectories, final-image metrics and boundary-region differences.

Correctness rules:

- Loader/lifetime-only changes must preserve weight bytes and existing
  same-policy outputs, including quantized outputs.
- Floating compute changes must pass the unchanged reference gates.
- Existing Q8 disagreement is not erased by a memory improvement. Do not
  require it to become F32-equivalent merely to prove a loader optimization,
  and do not promote it to public support as a side effect.
- New arithmetic/quantization changes require separate numerical and image
  evaluation. Preserve the existing independent-image findings; avoid
  presenting retuned calibration cases as held-out results.

Success means lower measured memory and/or latency under a declared scope,
not lower model-file size, a lucky single run, or improved averages hiding
the remaining cardigan boundary error.

## 12. Recommended order and non-goals

Execute checkpoints1 and2 first: attribute the peak, then remove duplicate
source/conversion residency using runtime-ready files and bounded loading.
Next prioritize modulation precomputation and fused F32 GELU; use real-shape
GEMM/attention microbenchmarks to choose the subsequent throughput work.
CFG batching, graph/input reuse and service caching follow only with
explicit memory budgets and measured justification.

No GPU assumptions, lower-quality resolution/step shortcuts, blanket
unquantization, global file-cache flushing, hidden F32 model copies or
numerical-gate relaxation are proposed.

The first memory objective is to realize the existing0.618GiB Q8-versus-BF16
weight saving. The larger architectural opportunity is removing roughly
0.964GiB of modulation matrices from denoising residency in favor of about
30MiB of cached outputs for the current schedule. Neither is a guaranteed
total-process reduction until peak and steady-state traces establish it.
