# Completed FASHN Q4/Q5 and mixed-precision study

**18 full generations and 18 probe processes completed.**
The report includes all current policies, original PyTorch outputs and
historical floating CPU/BLAS and Q8 mixed-precision controls.

- [Standalone interactive HTML](fashn-all-comparisons.html): download and
  open locally; all images/data are embedded, with no external requests.
- [GitHub-viewable comparison images](comparison-gallery-precision/README.md).
- [Detailed measurements and hashes](evidence/q45-study/results.json).
- [Direct Q8-versus-lower-bit metrics](evidence/q45-study/q8-vs-lowbit-pixel-differences.json).
- [Publication manifest](precision-publication-manifest.json).
- [Plan and reproduction protocol](../docs/fashn_q4_q5_plan.md).

The generated crops are `sample.png` in the original study. An amplified
`difference-x16.png` is an **error visualization, not an inference output**.
The interactive report hides error maps by default.

## Controlled full-image measurements

Windows x64 CPU, AMD EPYC 7763 exposure, 16 inference threads, two load
threads, 20 steps, CFG 1.5, shift 1.5, skip last CFG step, seed 42.
Canvas 576x864; matched displayed crop 576x768. Ready GGUF, no mmap,
F32 flash attention, bounded modulation cache and strict fused GELU.
Sampling includes modulation preparation and intermediate recording.
Preprocessing, HTTP and downloads are excluded.

| Policy | Cardigan sampling s | Bottoms sampling s | Cardigan peak WS GiB | Bottoms peak WS GiB | Cardigan private GiB | Bottoms private GiB |
|---|---:|---:|---:|---:|---:|---:|
| BF16/F32 | 2326.329 | 2317.110 | 2.0107 | 2.0118 | 2.1393 | 2.1394 |
| Q8_0 | 1815.613 | 1846.951 | 1.4013 | 1.4003 | 1.5285 | 1.5285 |
| Q4_0 | 2031.929 | 2058.341 | 1.0718 | 1.0716 | 1.1975 | 1.1974 |
| Q4_K | 1857.082 | 1745.683 | 1.0745 | 1.0739 | 1.2016 | 1.2008 |
| Q5_0 | 2201.090 | 2250.332 | 1.1546 | 1.1539 | 1.2810 | 1.2805 |
| Q5_K | 2234.039 | 2151.171 | 1.1578 | 1.1555 | 1.2831 | 1.2827 |
| Q5_K + four F32 | 2150.181 | 2158.221 | 1.2990 | 1.2979 | 1.4255 | 1.4255 |

GiB means 2^30 bytes. Working set and private commit overlap: **do not add
them**. These are single observations per case/policy, not statistical
confidence intervals. Filesystem caches were not flushed.

## Final cropped numerical differences

RMSE uses RGB byte-channel levels 0..255. PSNR = 20 log10(255/RMSE).
Lower RMSE/higher PSNR means closer pixels, not better garment fidelity.
The interactive report can switch the reference between BF16, Q8 and
historical original Python.

| Policy | Cardigan RMSE vs BF16 | Bottoms RMSE vs BF16 | Cardigan PSNR vs Q8 dB | Bottoms PSNR vs Q8 dB |
|---|---:|---:|---:|---:|
| Q8_0 | 1.5939 | 0.4240 | Exact | Exact |
| Q4_0 | 3.4936 | 1.7913 | 37.236 | 43.194 |
| Q4_K | 3.0025 | 1.2374 | 37.747 | 45.799 |
| Q5_0 | 2.4248 | 0.9453 | 43.836 | 49.453 |
| Q5_K | 2.2792 | 1.0215 | 44.769 | 48.498 |
| Q5_K + four F32 | 2.2348 | 0.8683 | 45.422 | 49.905 |

Global averages conceal local deviations. Cardigan Q4_K versus Q8 has a
worst scanned 64x64-region RMSE of **20.054**, PSNR **22.087 dB** and a
maximum channel difference of **99/255**. The 64x64 search uses 32-pixel
strides plus the image boundaries; it is not garment segmentation.

Every quantized policy fails the unchanged strict floating trajectory
gates: relative L2 <= 0.001 for every image and guided velocity state.
This does not suppress finite images, and does not promote public
quantized CLI/C API/server support. Human visual acceptance remains open.

## Storage and logical allocations

| Policy | Ready file bytes | Active denoising weights MiB | Arena peak MiB |
|---|---:|---:|---:|
| BF16/F32 | 2,471,712,672 | 1361.375 | 572.544 |
| Q8_0 | 1,808,160,672 | 728.562 | 538.044 |
| Q4_0 / Q4_K | 1,454,266,272 | 391.062 | 538.044 |
| Q5_0 / Q5_K | 1,542,739,872 | 475.437 | 538.044 |
| Q5_K + four F32 | 1,694,701,472 | 620.359 | 538.044 |

The cache is 30.659 MiB for every policy. There were zero runtime weight
conversions, zero conversion scratch and no source mmap. Modulation
precomputation retires 1,044,172,800 bytes of parameters before denoising.
Logical allocations and process peaks are different measurements;
activations and protected F32 parameters limit total memory savings.

Exact-block conversion verification covered all 104 eligible matrices.
Protected embeddings/modulations/norms/patch kernels/biases/final layers
remain F32. Q4_K/Q5_K are actual GGML tensor formats, not model-wide
Q4_K_M/Q5_K_M presets. Uniform quantizer importance is not activation calibration.

## Repeated runs, second seed and mixed precision

Q5_K was selected by the lowest mean cropped RMSE versus fresh BF16 across
the two cases, not by human preference.

Three same-seed cardigan runs sampled in 2234.039, 2153.032 and 2126.305
seconds. Mean **2171.125 s**, sample SD **56.100 s**, CV **2.584%**.
Final floating tensors and PNG pixels were exactly identical.
One interrupted attempt stopped before completion, was preserved, and
was rerun from the original seed; it is not counted as a completed observation.

The seed-43 pair sampled in **2244.711 s** (BF16) and **2085.721 s** (Q5_K).
Q5_K versus its matching seed-43 BF16 crop has RMSE **1.4147** and PSNR
**45.118 dB**. Two seeds are descriptive evidence, not a population estimate.

Restoring original F32 `single_blocks.{12,11,8,14}.linear1.weight` improved
overall RMSE in both cases, at an increased working set of about 1.30 GiB.
At matched worst-error coordinates, cardigan RMSE worsened from **17.641**
to **18.135**, while bottoms improved from **5.548** to **5.415**.
The old Q8-sensitive subset is not proven to be the optimum Q5_K policy.

## Historical controls and limitations

Original Python and CPU/BLAS/Q8-mixed results are embedded in the HTML but
kept separate from current averages. Their execution scopes and recording
overheads differ. Original Python was faster in those historical CPU
observations. Its old cardigan memory sample monitored the launcher, not
the interpreter: the new report marks that value unavailable.

No new ARM64, Android or GPU inference was performed for this study.
Photographic fixtures, generated adaptations and contact sheets retain
the separate licensing/attribution requirements in [NOTICE.md](NOTICE.md).

## Rebuild the publication

With the documented Python experiment environment and complete local
checkpoint-27/38 and Q4/Q5 evidence:

```powershell
python scripts\build_fashn_comparison_report.py --repo . --experiment ..\fashn-vton-reference --output ..\fashn-vton-reference\q45-study\fashn-all-comparisons.html
python scripts\package_fashn_precision_reports.py --repo . --experiment ..\fashn-vton-reference
```

The repository copies preserve numerical values and original output PNG
bytes. JSON paths are normalized to `local-experiment`/`repository`.
Only contact-sheet thumbnails are resized. The source/publication hashes
are recorded separately; full tensors, weights and executable payloads
remain deliberately unbundled.
