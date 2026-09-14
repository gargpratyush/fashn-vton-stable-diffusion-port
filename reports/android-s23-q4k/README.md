# Samsung S23: completed native Q4_K try-on

**Execution demonstrated; strict numerical agreement failed.** The device agent's
first attempt in the authorized cardigan retry queue completed 20 steps / 39
forwards and exited 0. No retry was needed in that queue and no inference remained
active. This does not erase the earlier interrupted 13/20-step run or earlier
failed probes.

This page summarizes the returned `FINAL-CARDIGAN-REPORT.md`.
The full run was not repeated on the x64 host. The returned PNG's
SHA256 was checked against the device report before copying it here unchanged.

## Actual Android output

![Actual Samsung S23 Q4_K cardigan output, not an error map](sample.png)

The image is the generated 576x768 crop of the 576x864 canvas, not a source
photograph or an amplified difference map. Visual plausibility is not proof of
garment fidelity. The fixture/adaptation attribution and noncommercial/share-alike
restrictions in [the report notices](../NOTICE.md) apply.

## Configuration

Samsung S23 with 8 GB RAM; native Android ARM64 CPU execution, one inference
thread and one loading thread. No GPU or NPU acceleration was used.

- Original-F32-derived ready Q4_K GGUF; 104 direct Q4_K matrices with protected
  parameters retained in F32.
- F32 flash attention, fused GELU, modulation precompute/cache with a 128 MiB
  budget; no source mmap or runtime weight conversion.
- Imported bundled cardigan noise, seed 42, category 1, 20 steps, CFG 1.5,
  shift 1.5, and no unconditional forward on the final step.
- Final-only recording, not a saved tensor trajectory for every step.

## Time, memory and thermal observations

| Measurement | Reported result |
|---|---:|
| Full wall time, including supervisor exit observation | 24,099.465 s = 6 h 41 min 39 s |
| Native sampling | 24,096.030 s |
| Modulation precompute | 4.864 s |
| Peak sampled RSS | 1,109,581,824 bytes, approximately 1.03 GiB |
| Peak sampled PSS | 1,106,865,152 bytes, approximately 1.03 GiB |
| Peak sampled swap | 670,822,400 bytes, approximately 640 MiB |
| Minimum system MemAvailable | 1,463,033,856 bytes, approximately 1.36 GiB |
| Maximum battery temperature | 41.3 C |
| Maximum Android thermal status | 2, MODERATE |
| External telemetry | 4,119 samples; maximum observed gap 6.621 s |
| Start | 37.4 C; thermal 0; 43% charge; powered/charging |
| End | 33.2 C; thermal 0; 100% charge; powered/full |
| Watchdog / native exit | No guard abort; exit 0 |

RSS, PSS, swap and available memory are separate observations; do not add
independent peaks. Swap occupancy alone does not establish how much time was
spent paging. Battery temperature is not SoC temperature, and CPU-frequency
observations alone do not establish throttling.

The powered run stayed below its stop limits. It does not establish sustained
unpowered operation, battery endurance, general thermal safety, or controlled
memory savings versus floating inference on this same phone.

### Original Python and Windows context, not a same-device benchmark

| Implementation / case | Hardware / threads | Sampling interval s | Process wall s | Process memory observation |
|---|---|---:|---:|---|
| Original Python F32, cardigan | Windows x64 CPU / 16 | 1241.326, sampler + PIL | 1258.037 | N/A: invalid launcher measurement |
| Original Python F32, bottoms | Windows x64 CPU / 16 | 1263.737, sampler + PIL | 1325.477 | 6.573 GiB peak WS; 9.603 GiB private commit |
| Native Q4_K, cardigan, current x64 study | Windows x64 CPU / 16 | 1857.082, recorded native sampling | 1857.616 | 1.0745 GiB peak WS; 1.2016 GiB private commit |
| Native Q4_K, cardigan, this device run | S23 Android CPU / 1 | 24096.030, final-only sampling | 24099.465 | Approximately 1.03 GiB peak RSS; separate 640 MiB peak swap |

The [original Python baseline](../original-python-baseline.md) runs the unchanged
upstream default CPU sampler with batched CFG and prepared inputs, not the
full raw-photo pipeline. The [x64 Q4/Q5 study](../q4-q5-results.md) records
intermediate states; Android saves only the final state. Hardware, threads,
recording, memory accounting and thermal conditions differ. Windows working set,
private commit and Android RSS/PSS/swap are not interchangeable measurements.
**No original-Python latency or memory baseline was measured on the S23.**
Do not present these rows as an Android quantization speedup.

## Numerical comparisons

The device report checked all 39 forwards, the final finite F32 image shape,
direct quantization policy, CPU backend and F32 flash policy. Schedule maximum
difference was `4.47e-8`, passing the unchanged `1e-7` gate.

Same-policy Android/x64 Q4_K final float relative L2 was **0.008239485**, failing
the unchanged **0.001** gate. Float RMSE was `0.005354120`, MAE `0.001805577`,
and maximum absolute error `0.519490629`. The root cause remains unresolved;
the final-only run cannot establish intermediate-step agreement.

Pixel metrics use original, unresized RGB uint8 channels in the range 0..255.
The local metric is the worst scanned 64x64 region, not a semantic garment mask.

| Reference / scope | RMSE | MAE | PSNR dB | Max error | Worst-local RMSE |
|---|---:|---:|---:|---:|---:|
| x64 Q4_K, full canvas | 0.75445 | 0.22036 | 50.578 | 66 | 6.29258 |
| x64 Q4_K, crop | 0.79949 | 0.24673 | 50.075 | 66 | 6.28190 |
| Historical BF16/F32, crop | 2.91208 | 1.19201 | 38.847 | 95 | 11.54871 |
| Historical Q8_0, crop | 3.23812 | 1.23382 | 37.925 | 98 | 18.66660 |
| Original Python, crop | 2.91209 | 1.19201 | 38.847 | 95 | 11.54820 |

The historical-reference rows combine platform and quantization differences.
They do not isolate Android quantization error. Q4_K already failed the strict
floating-reference gates on x64; that separate failure has not been waived.

## What quantization does here

FASHN does not have an LLM's prompt-prefill / one-token-at-a-time decode phases.
Each diffusion forward processes thousands of image-patch tokens: 3,456 per
stream and 6,912 in the joint stream. Smaller weights reduce storage and weight
traffic, but do not reduce token count, forward count, or the number of model
operations.

The direct CPU Q4_K path is not "expand every matrix to FP16, then use a float
matmul." The diagnostic runner leaves quantized matrices packed when
`--upcast-matrices` is absent. At the pinned GGML revision, the ordinary CPU
Q4_K matmul uses a Q4_K x Q8_K dot-product kernel: F32 input activations are
temporarily packed into Q8_K, and results are F32. ARM-specific and repacked
kernels provide implementation variants. Block unpacking and scale corrections
still cost work; quantization is not free.

This is also not a fully low-bit network: F32 flash attention and other
activation operations/protected parameters remain. The report does not record
the precise hot kernel variant or attribute the seven-hour runtime to particular
operations. Source support alone does not prove the optimal SIMD/matrix path ran.

Source references:

- [Diagnostic upcast selection](../../tests/test_fashn_vton_trajectory.cpp) and
  [runner arithmetic configuration](../../tests/fashn_test_runner.h).
- [Linear weight casting](../../src/model/common/ggml_block.hpp) and
  [FASHN graph policy](../../src/model/diffusion/fashn_vton_model.h).
- Pinned GGML [Q4_K dot-product traits and CPU matmul](https://github.com/leejet/ggml/blob/e20c3a14aa70ee84ca58499814206dd08d8026bc/src/ggml-cpu/ggml-cpu.c)
  and [ARM quantized kernels](https://github.com/leejet/ggml/blob/e20c3a14aa70ee84ca58499814206dd08d8026bc/src/ggml-cpu/arch/arm/quants.c).

Use seconds per forward, sampling time and end-to-end seconds per image, not
LLM tokens/sec. Keep shape, steps, CFG, threads, noise, recording policy and
thermal/start conditions controlled. No same-phone floating-versus-Q4_K speedup
has been established by this run.

## Provenance and reproduction boundary

| Artifact | Identity / SHA256 |
|---|---|
| Tested source | `e86c564fd42d876b30befcaf4a782e437f0fe134` |
| GGML | `e20c3a14aa70ee84ca58499814206dd08d8026bc` |
| Q4_K model | `a70b93d63af49d953e3bb535887af39b9bfd33ba9910b43515b66f1f9f21b34d` |
| Android trajectory binary | `6a96df895351b8f493fd5152bba9b184773e4a9464e585d757936e29716e80ee` |
| Final tensor | `cff7bf938217c5607e78246eb2d59600a51cfa9bacfa5900a9ce76f4d904fd61` |
| Published crop, unchanged | `56c7361ed2efe08406a3e53d478e8250693f94511fd3f6ee18e42cbf1a4f2bd7` |
| Returned source report | `c4725fe7eecd1ac01c5095fec5de96546ba93de5c0ea4e62d593c5b947804fcd` |

The report identifies private run directory
`device-v1\full-cardigan-queue-v1-attempt-1`, including `run.json`, native
`manifest.json`, `comparison.json`, telemetry, logs, final tensors and exit
records. Those files are not bundled in this publication; the returned summary
and PNG do not independently verify the entire raw run.

Reproduction requires the original verified bundle and supervised
`retry-cardigan.py` launcher. Its 15-hour wall ceiling, 42 C / thermal-SEVERE
stops, charge/memory budgets and 30-second lost-host-heartbeat stop remained
enabled. Phone-side normal-exit and heartbeat-loss termination tests passed
before inference. Do not run an unsupervised raw native command.

This is one prepared-input cardigan case on the older diagnostic source pin,
not acceptance of subsequent maintenance revisions, public quantized generation,
bottoms, native raw-photo preprocessing, an APK, standalone phone launching,
GPU/NPU acceleration or strict cross-platform numerical parity.
