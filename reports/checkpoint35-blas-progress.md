# Optimization checkpoint 5: CPU math backends (complete)

Checkpoint 5 is complete. Both full 20-step photographic trajectories passed
every unchanged reference gate. BLAS remains a diagnostic throughput option,
not a relaxation of public backend or quantization restrictions.

## Final complete-image acceptance

| Case | Sampling including modulation preparation | Peak working set | Maximum image relative L2 | Maximum velocity relative L2 |
|---|---:|---:|---:|---:|
| Tops | 1623.947 s (27m03.9s) | 2.028912 GiB | 1.35935e-6 | 1.54838e-6 |
| Bottoms | 1615.263 s (26m55.3s) | 2.028805 GiB | 1.95498e-6 | 2.41928e-6 |

All 20 updated images and guided velocities in each case passed the 0.001
relative-L2 gate; initial noise and schedules also passed their original
checks. All 107 denoising matrices remained on BLAS.
Final full-canvas PNGs differ from the F32 PyTorch oracle in only 99 / 39
byte channels respectively, each by exactly one intensity level.
This measures numerical agreement, not broad perceptual quality.

The bottoms run is 35.58% faster than checkpoint 4's 2507.495-second
complete run, with peak working set about 17 MiB higher. These full runs
were separate single observations: use the matched 32.65% short-control
result for the more controlled speed comparison.

Persistent comparison images:
`checkpoint35-full-acceptance\tops-comparison.png` and
`checkpoint35-full-acceptance\bottoms-comparison.png`.
Each contains the reference, native result and amplified absolute difference.
All measurements, commands, hashes, trajectories and reference comparisons
are retained in `checkpoint35-full-acceptance\results.json` and its siblings.

## Clean controls completed

Same binary, precision, inputs, 39-entry cache, fused GELU and 16-thread
settings; A-B-B-A fresh processes, three forwards each:

| Configuration | First forwards s | Four warm samples s | Warm median s | Peak working set |
|---|---|---|---:|---:|
| CPU | 65.0375, 62.8882 | 63.0546, 62.2332, 62.8675, 61.5211 | 62.5503 | about 1.98 GiB |
| BLAS + CPU fallback | 44.7443, 44.0959 | 42.6205, 42.2336, 42.0154, 41.3747 | 42.1245 | about 1.99 GiB |

Observed warm median improvement: **32.65%**. This is a matched short-control
result, not yet a complete-image speed claim. The ranges do not overlap,
but there are only two processes per configuration on a shared VM.

Both paths retain the same **600,355,904-byte** no-capture graph workspace.
BLAS introduces a small resident-memory increase in these controls and a
larger private-commit tradeoff: sampled peak private bytes were about
4.20 GiB versus 4.00 GiB. Private commit is not resident RAM. Do not describe
the BLAS path as using no additional memory merely because arenas match.

Both CPU repetitions were bit-identical, including comparison with the
previously accepted checkpoint-34 fused output. Both BLAS repetitions also
matched one another exactly. BLAS-versus-CPU velocity relative L2 was
3.53951e-7, max absolute 3.33786e-6; every value passed the diagnostic
1e-4 absolute/relative pointwise comparison. All 107 matrices actually ran
on the selected primary backend.

## Attention comparison completed

Native strict-F32 flash and the unchanged pinned upstream default SDPA
processor ran at 3456/6912 tokens, ten heads, dimension 128 and 16 threads.
The helper obtains original single-block strides from upstream QK
normalization and RoPE. Additional packed-head and explicit input/output
copy/layout controls make overhead differences visible.

| Tokens | Native two warm times s | Upstream single-layout warm times s | Torch packed + explicit I/O warm times s |
|---|---|---|---|
| 3456 | 0.1770, 0.1682 | 0.1650, 0.1364 | 0.1469, 0.1524 |
| 6912 | 0.5837, 0.5977 | 0.5225, 0.5252 | 0.5096, 0.5324 |

All six layout/size comparisons passed relative L2 <= 0.001 and max
absolute <= 2e-5. Relative L2 was 8.68e-7 / 1.23e-6 by token count.
Native arenas were 70,778,944 / 141,557,824 bytes. Torch private workspace
is not measured by those native arena values. No Torch DLL or attention
kernel is embedded in the native application. This modest synthetic
attention gap does not by itself explain historical whole-model latency.

Evidence: `checkpoint35-attention-native`,
`checkpoint35-attention-comparison`.
The ordinary CPU CLI/server were rebuilt; seven targeted baseline checks
and four final BLAS-path checks passed.

## Full-image acceptance

The two existing 20-step tops/bottoms reference cases completed serially.
Their exact prepared-input SHA256 values were resolved before launch.
`checkpoint35_full_acceptance.py` records generation, process timelines,
all-step comparisons and three-panel reference/native/difference images
under `checkpoint35-full-acceptance`. It stops on an execution or unchanged
numerical-gate failure. These references are the original F32 model/MATH
trajectory oracle, not new latency runs of the unmodified Python sampler.

## Implemented

- Added a separate supported GGML BLAS build under `build\cpu-blas`.
  Baseline build and public defaults remain unchanged.
- Installed official OpenBLAS 0.3.34 LP64 x64, not PyTorch-private DLLs.
  Archive SHA256:
  `e9cb6134541f36c27346d5fc5995652f060fba227cebbbabcbda5a5a44d7c76b`.
  Runtime configuration: `DYNAMIC_ARCH NO_AFFINITY Zen MAX_THREADS=64`,
  pthreads. DLL lookup is process-local, not a global PATH modification.
- Added `test-fashn-math`: double-accumulated small F32 GEMM reference,
  actual-size GEMM measurements, strict CPU-fallback attention checks,
  precision/thread guard checks, and real-size attention diagnostics.
- Added diagnostic `--backend BLAS` trajectory selection. Explicit F32
  matrix computation is required; direct Q8 does not silently change into
  weight-only F32 SGEMM.
- Used the existing scheduler's CPU fallback, rather than adding a kernel
  or editing GGML. BLAS handles supported large matrix operations; CPU
  handles flash attention, normalization, GELU and small modulation GEMVs.
- Configured BLAS through its registered thread-setting API. The existing
  runner separately configures the automatic CPU fallback.
- Added actual scheduled matrix-backend counts and per-backend workspace
  accounting. GGML's allocator reports shared host buffers only once;
  a zero CPU entry does not mean CPU operations need no memory.
- Kept the public CPU-only/floating-only guards. Public BLAS selection is
  not promoted by these diagnostic changes.

## First integration failure, retained

The first full-graph attempt stopped before executing a denoising forward.
The shared attention builder checked only the primary BLAS backend,
expanded attention instead of constructing the F32 flash node, and FASHN's
28-flash-node guard rejected it. Modulation precomputation had succeeded.

The fix passes an explicit, supported fallback device to the attention
builder. Only the diagnostic FASHN BLAS route supplies it; defaults remain
null. Small CPU-versus-BLAS-fallback attention is bit-identical.
The original failed directory/log remain `checkpoint35-blas-graph`.

## Real-shape F32 GEMM results

16 threads per active math backend, same F32 values/precision, no simultaneous
model experiments. One cold plus two warm measurements per shape/backend.
These include graph construction, workspace/input copies and output copying,
not just a hidden warmed kernel. Synthetic values are not model activations.

| Shape (K, output width, tokens) | CPU warm seconds | BLAS warm seconds |
|---|---:|---:|
| Target patch (1008,1280,3456) | 0.0419-0.0434 | 0.0270-0.0305 |
| Double MLP (1280,5120,3456) | 0.1899-0.1942 | 0.0948-0.1118 |
| Single QKV/MLP (1280,8960,6912) | 0.6845-0.6855 | 0.3120-0.3702 |
| Single projection (6400,1280,6912) | 0.7162-0.7438 | 0.2251-0.2484 |

All relative L2 differences were below 1.75e-7. Large GEMMs were supported
by the primary BLAS backend. The (1280,7680,1) modulation GEMV correctly
used CPU fallback and was bit-identical. Matrix benchmark workspace
capacities were unchanged; backend-internal allocations are not included.

## Full-graph correctness and dispatch

`checkpoint35-blas-graph-v2\comparison.json` passed all 34 existing
conditional/null reference captures without changing any tolerances.
All 28 F32 K/V flash attention nodes were retained.
The scheduler assigned all 107 remaining denoising matrix multiplies to BLAS.

Captured conditional/null forwards took 44.3167 / 41.7770 seconds.
These are not clean repeated timing controls. Captures retain extra outputs;
their 1,050,234,944-byte shared workspace must not be compared directly to
the ordinary no-capture arena.

The current serial A-B-B-A controls use the same BLAS-capable binary,
BF16-ready weights, 39-pair modulation precomputation and strict fused GELU.
Three t0 forwards per fresh process separate first use from two warm samples.
They record both external process memory and phase markers without intrusive
page scans. Results are under `checkpoint35-forward-controls`.

## Thread and memory scope

Both BLAS and CPU fallback are configured for 16 threads. That is a
per-backend setting, not a claim that only 16 OS threads exist. The scheduler
executes the backend fragments sequentially, not by invoking BLAS inside a
CPU worker's matrix kernel. Resident and briefly spinning worker pools can
still affect switching costs; clean model timings include those effects.

No complete model-wide F32 parameter copy is introduced. BF16 matrix casts
remain graph intermediates, as in the accepted floating baseline.
Scheduler liveness and backend-internal packing can still alter peak memory;
the controls must establish that tradeoff rather than infer it from file size.
