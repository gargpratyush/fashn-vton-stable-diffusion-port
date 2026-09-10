# Full-image quantization and actual upstream comparisons

This experiment batch is complete: original-Python comparisons on two image
pairs, all104 individual matrix restorations,12 combined/control probes,
all-Q8 and four-matrix mixed full images on both pairs, and a matched floating
recorder control. Historical intermediate findings are retained below.
No public quantized inference gate has been relaxed.

Final gallery: `comparison-gallery-v6\index.html` (11 comparison rows across
three source pairs; only two pairs were used for actual-upstream and Q8/mixed
comparisons). Machine-readable evidence: `full-image-precision-summary.json`.
Complete implementation history: `implementation-and-experiment-history.md`.

**Main conclusion:** restoring only four matrices reduced cropped mean byte
error55.2% for trousers and26.3% for cardigan while retaining100 Q8 matrices.
Both look close to floating output, but strict quantized numerical agreement
is still not established. Q8 and mixed ran faster than the matched native
floating recorder, but used more measured peak memory. Original Python was
still faster than our native paths in these CPU observations.

## Completed: first full Q8 image

Inputs: standing CatVTON person, original FASHN seated garment source wearing
beige trousers, category bottoms. Same prepared inputs and seed as the
validated floating quality case. Settings:20steps, CFG1.5, shift1.5, skip1,
seed42, CPU16threads, F32 flash attention.

All104 eligible matrices were Q8_0. Embeddings, modulation, norms, patch
kernels, biases and final layers remained F32. This was direct GGML Q8
matrix arithmetic, including its quantized-activation dot-product path,
not weight-only Q8 followed by F32 upcasting.

| Measurement | Result |
|---|---:|
| Complete diagnostic process | 1947.3476s = 32m27s |
| Recorder sampling interval | 1945.9799s |
| Peak working set | 4,457,504,768 bytes, approximately4.151GiB |
| Sampled peak private bytes | 2,804,740,096 |
| Exit status | 0; complete trajectory produced |

The recorder includes lazy execution setup and per-step tensor export.
The working-set peak includes this process's conversion/mapping behavior.
Do not call this a measured memory saving or controlled speedup over the
earlier public BF16 CLI run. The subsequently completed matched BF16 recorder
control is reported separately below.

### Cropped Q8 versus validated native BF16-storage/F32-compute output

Exact manifest crop: x0, y48, width576, height768. No resizing for metrics.

| Metric | Result |
|---|---:|
| Different color channels | 167,013 / 1,327,104 |
| Different pixels | 136,079 / 442,368 |
| Maximum byte difference | 41 |
| Mean absolute byte difference | 0.13016237 |
| RMSE in byte levels | 0.42397650 |
| Numerical PNG PSNR | 55.58397dB |

At normal viewing size, the outputs look very similar. No obvious new
garment, pose or major anatomical failure was apparent in this inspection.
The same simplified trouser construction and shortened white top occur in
both. Small pixel-level differences remain; one visually quiet case is not
a product-quality guarantee, and numerical PSNR is not a perceptual score.

### Full Q8 versus original-model F32-math oracle trajectory

| Criterion | Result |
|---|---:|
| Initial noise max absolute error | 4.76837e-7; passed |
| Failed updated-image states | 9/20, beginning at step index11 |
| Failed guided-velocity states | 20/20 |
| Maximum image relative L2 | 0.0044633785 |
| Final image relative L2 | 0.0041569855 |
| Maximum guided-velocity relative L2 | 0.0085119091 |
| Unchanged per-state gate | 0.001 |

Full-canvas PNG metrics include padding and compare against Python rather
than the native floating PNG: MAE0.1156925, max41, PSNR56.09697dB.
Use the cropped native/native figures above for the displayed Q8 comparison;
the two sets are not interchangeable.

### What this changes

The earlier "19/34 failed captures, maximum relative L2 approximately0.06483"
came from a **synthetic stress fixture**. It correctly exposed numerical
drift but did not establish how severe a photographic image would look.
The first full image is much more visually similar than a binary failed-
capture count alone suggests. It would be premature to restore all104
matrices merely because that count was high.

This motivated a photographic calibration study, restoring one matrix at
a time and recording whether it actually reduces downstream error.
The ranking objective is the worst conditional/null
final-velocity relative L2 for the same fixed probe. Negative interventions
are retained: removing one quantization error can worsen cancellation of
others. Small combined policies require separate full-image/held-out checks.

## Actual original Python sampler: two photographic cases complete

The unmodified pinned `TryOnPipeline._sample` completed for the cardigan
and bottoms cases. CPUF32, default SDPA dispatch and original
batched conditional/null execution are retained, including redundant final
null computation. The harness does not replace the sampler.

This is prepared-input inference, not original raw-image `__call__` startup.
The constructor is bypassed; local original weights are loaded using a
meta-device initialization, and no parser/detector instance is created.
Condition/PNG lineage is verified before sampling. Reports separate
sampling/PIL time from the measured child-process wall time and memory.

Cardigan results with the exact same prepared inputs, settings and16threads:

| Quantity | Original Python | Native floating implementation |
|---|---:|---:|
| Process wall time | 1258.0365s (20m58s) | 2549.6227s (42m30s), historical |
| Sampler plus PIL conversion | 1241.3262s | Not separately comparable |
| Final differing byte channels | 69 / 1,327,104 across the pair | Each differs by1 |
| Mean byte difference | 0.0000519929 across the pair | |
| Numerical PNG PSNR | 90.97136dB across the pair | |

The outputs look the same at normal scale, including the graphic/hem
limitations. **Python is faster in this observation, not C++.** These are
not interleaved repeated controls and have different startup paths, so this
is not a universal or statistically controlled2x speed claim.
Torch kernels, default attention and batched CFG are investigation areas;
their individual contributions have not been isolated.

The second case confirms the same direction, now with valid interpreter
memory tracking:

| Bottoms measurement | Original Python | Native floating implementation |
|---|---:|---:|
| Process wall time | 1325.4767s (22m05s) | 2491.2815s (41m31s), historical |
| Sampler plus PIL conversion | 1263.7368s (21m04s) | Not separately comparable |
| Peak working set | 7,057,371,136 bytes (6.573GiB) | 3,175,022,592 bytes (2.957GiB) |
| Sampled peak private bytes | 10,310,897,664 | 3,334,754,304 |
| Different color channels | 44 / 1,327,104 across the pair | Each differs by1 |
| Mean byte difference | 0.0000331549 across the pair | |
| Numerical PNG PSNR | 92.92533dB across the pair | |

The measured Python interpreter was PID27516, distinct from launcher29320.
Normal-scale appearance is the same, including the simplified trousers and
shortened top. On these observations, Python runs faster while native BF16
residency uses less process memory. The scope and historical-control caveats
above still apply; there are no repeated-trial confidence intervals.

### Memory instrumentation failure and correction

The first Python run reported only11.7MB peak working set. The monitor had
attached to Windows' virtual-environment launcher, not the actual interpreter.
Those memory values are invalid for model comparison and are excluded from
the corrected comparison report. Wall time still covered the child's full
execution, and the output comparison is unaffected.

The harness now bootstraps the script before importing model dependencies,
atomically registers the active interpreter PID, validates its relationship
to the launched process, and samples that interpreter's memory. Native
executable measurement remains unchanged. A real32MiB allocation regression
and a timeout/process-tree cleanup regression passed. Python measurement
reports now explicitly identify the launcher PID, monitored PID, actual
command and memory scope. The second image uses this corrected path.

A repeated timeout test subsequently exposed an unreliable `taskkill` exit
under load. Cleanup now uses native Windows process handles for the owned
PID tree, checks creation times to exclude stale/reused-parent-PID matches,
and waits for termination. A startup acknowledgement prevents an orphaned
interpreter from beginning model work before its measuring owner is ready.
Tests also caught and corrected an intermediate acknowledgement-placement
bug. The final native/Python measurement, handshake, timeout, comparison,
gallery and ablation-helper run passed21 tests. The completed second image
has valid interpreter-PID attribution; its ordinary completion did not
exercise the timeout path.

Original raw evidence is retained at
`..\checkpoint27-upstream-cardigan-measurement\measurement.json`, but its
memory counters must not be used. The corrected derived comparison is
`..\checkpoint27-cardigan-comparison-v2\comparison.json`.

## Photographic sensitivity and mixed-precision study

### Complete104-matrix sensitivity scan

All104 individual restorations have completed.55 lowered the chosen
worst-branch velocity error and49 worsened it; none individually passed the
unchanged numerical gate. The strongest result is clearly separated from
the others:

| Individual F32 restoration | Worst-branch velocity L2 | Relative error reduction |
|---|---:|---:|
| `single_blocks.12.linear1.weight` | 0.0015574260 | **40.6711%** |
| `single_blocks.11.linear1.weight` | 0.0025476291 | 2.9501% |
| `single_blocks.8.linear1.weight` | 0.0025489503 | 2.8998% |
| `single_blocks.14.linear1.weight` | 0.0025698115 | 2.1051% |
| `single_blocks.15.linear2.weight` | 0.0025781198 | 1.7886% |

Restoring block12/linear1 changes conditional velocity relative L2 from
0.0026250719 to0.0013878256 and null velocity from0.0016766500 to0.0015574260.
The null branch becomes the limiting branch.103 eligible matrices remain
Q8, and the additional weight payload is33,689,600 bytes (32.129MiB).
The maximum captured-layer error and19 failed-capture count remain unchanged,
despite the substantial output-error reduction. This is another reason not
to select a policy by failure count alone.

This is a restoration effect on one photographic initial-state probe,
including the local F32 activation/multiplication path. It is not a claim
that this matrix universally causes40.7% of quantization error.

`matrix-sensitivity-complete` contains all104 rows in `ranking.csv` and
`ranking.md`, an immutable hash-verified study snapshot, and small candidate
policies. They are proposals, not adopted inference settings:

| Candidate | Restored linear1 matrices in single blocks | Q8 matrices left | Added weight payload |
|---|---|---:|---:|
| Top1 | 12 | 103 | 32.129MiB |
| Top2 | 12,11 | 102 | 64.258MiB |
| Top4 | 12,11,8,14 | 100 | 128.516MiB |
| Branch-aware pair | 12,14 | 102 | 64.258MiB |

The branch-aware pair includes block14/linear1 because it provides the
strongest isolated null-branch improvement, rather than simply taking the
second-best overall initial ranking.

### Completed: combined policies and weight-only control

All12 probes completed: six policies at each of the initial state and step15
of the actual reference trajectory. Step15 consumes the recorded updated
image from step14, not new noise at a later timestep. Both states use the
same photographic bottoms pair: this is a later-state check, not an
independent held-out image.

| Policy | Initial worst-branch velocity L2 | Step15 worst-branch velocity L2 |
|---|---:|---:|
| All104 Q8, direct computation | 0.0026250719 | 0.0034749969 |
| Restore block12/linear1 | 0.0015574260 | 0.0018426699 |
| Restore blocks12,11/linear1 | 0.0015441206 | 0.0015640206 |
| Restore blocks12,11,8,14/linear1 | **0.0012749801** | **0.0012152331** |
| Restore blocks12,14/linear1 | 0.0013461362 | 0.0017551888 |
| All104 Q8 weights, F32 multiplication | 0.0010505147 | 0.0010230602 |

The four-matrix subset is the strongest small direct-Q8 combination tested
at both states: worst-branch error falls51.4307% and65.0292%, respectively.
It leaves100 matrices Q8 and adds128.516MiB of weight payload versus all-Q8.
This is not proof of a globally optimal or minimal subset.

The all-Q8/F32-compute control retains rounded Q8 weights but dequantizes
them before multiplication, removing the usual quantized-activation path.
It improves both objectives further, but dequantization cannot recover
the original weight values. Restoring an original matrix also changes
that operation's arithmetic path; the individual scan does not isolate
weight rounding from activation arithmetic.

**None of the12 probes passes the unchanged complete capture gate.**
Top4 still fails17/34 captures initially and20/34 at step15; maximum capture
relative L2 is0.03136824 and0.07225986. Its conditional/null velocity errors
are0.0009584650/0.0012749801 initially and0.0011902909/0.0012152331 later.
Weight-only Q8 fails16/34 captures at each state, with maximum capture errors
0.01234718 and0.03260408; its conditional/null errors are
0.0008045028/0.0010505147 initially and0.0010230602/0.0007557981 later.
Near-threshold output errors do not erase larger internal differences.

Evidence: `..\checkpoint29-policy-evaluation\evaluation.json`, with
per-policy precision maps, comparison metrics, logs and measurements.
These are teacher-forced local comparisons, not free-running trajectories.
Top4 was selected only for experimental full-image evaluation. Complete
bottoms and cardigan images and the matched floating control have now
finished; their results are reported below.
Public generation guards remain unchanged.

### Completed: full20-step four-matrix mixed bottoms image

The selected four matrices use original checkpoint values represented in
F32;100 eligible matrices retain direct Q8 computation. The same original
checkpoint, condition tensor, recorder executable, sampling settings and
16threads were used as the earlier all-Q8 run. Executable hash matches
exactly. These are single observations at different times, not repeated
interleaved benchmarks.

| Result | All104 Q8 | Mixed100 Q8 +4 F32 |
|---|---:|---:|
| Process wall time | 1947.3476s (32m27s) | 2007.3931s (33m27s) |
| Peak working set | 4,457,504,768 bytes | 4,592,234,496 bytes |
| Sampled peak private bytes | 2,804,740,096 | 3,001,339,904 |
| Cropped MAE versus floating native PNG | 0.13016237 | **0.05832625** |
| Cropped RMSE | 0.42397650 | 0.28955997 |
| Cropped numerical PNG PSNR | 55.58397dB | **58.89603dB** |
| Maximum cropped byte difference | 41 | 30 |
| Differing cropped byte channels | 167,013 /1,327,104 | 74,044 /1,327,104 |
| Failed updated-image states versus F32 oracle | 9/20 | 4/20 |
| Failed guided-velocity states | 20/20 | 20/20 |
| Final image relative L2 | 0.0041569855 | 0.0026928717 |
| Maximum image relative L2 | 0.0044633785 | 0.0026928717 |
| Maximum guided-velocity relative L2 | 0.0085119091 | 0.0031627102 |

The restoration reduced cropped mean byte error55.1896%, final float-image
relative L2 by35.2206%, and maximum guided-velocity error62.8437%.
Measured wall time increased3.0835%; the mixed peak working set was4.277GiB.
The completed matched BF16 recorder control below supplies the missing
floating/Q8/mixed recorder performance comparison.

The mixed trajectory still fails the unchanged gate: updated images16-19
exceed0.001, and every guided velocity does. Initial noise and schedule
remain within their gates. Full-canvas comparison against the Python oracle
has MAE0.05184087 and PSNR59.40963dB; do not substitute those values for the
cropped native/native comparisons above.

Visual inspection of the floating/mixed and all-Q8/mixed contact sheets
shows the same overall garment silhouette, pose, face and background at
normal scale. The amplified difference localizes the strongest residual
around the trouser crotch/seams, with smaller scattered differences.
The simplified, tight trousers and shortened white top remain. There is no
obvious new major visual failure, but the numerical improvement is not an
obvious garment-quality improvement or certification of product fidelity.

The full-resolution gallery is
`comparison-gallery-mixed-bottoms\index.html`. It includes floating/mixed
and all-Q8/mixed rows, source images, contact sheets, amplified differences,
exact crops and hashes. Direct all-Q8 versus mixed output has MAE0.11341236,
max11 and PSNR57.42822dB; that measures the intervention's effect, not error
against ground truth.

Evidence: `..\checkpoint30-mixed-top4-bottoms`,
`..\checkpoint30-mixed-top4-bottoms-measurement\measurement.json`,
`..\checkpoint30-mixed-top4-bottoms-comparison\comparison.json`, and
`comparison-gallery-mixed-bottoms\visual-review.json`.

### Completed: independent cardigan full images

The four-matrix policy was fixed from the bottoms study before generating
the cardigan image; no cardigan-specific matrix selection was performed.
Same seed42,20steps, CFG1.5, shift1.5, skip1 and16threads. Both outputs use
the exact crop x0/y48/width576/height768, and are compared against the
previously established native floating output.

| Result | All104 Q8 | Mixed100 Q8 +4 F32 |
|---|---:|---:|
| Process wall time | 1956.6835s (32m37s) | 1975.8638s (32m56s) |
| Peak working set | 4,470,542,336 bytes (4.164GiB) | 4,621,623,296 bytes (4.304GiB) |
| Sampled peak private bytes | 2,868,330,496 | 3,120,369,664 |
| Cropped mean byte error | 0.25865494 | **0.19067760** |
| Cropped RMSE | 1.59386299 | 1.51533736 |
| Numerical PNG PSNR | 44.08178dB | 44.52062dB |
| Maximum byte difference | 83 | 82 |
| Different color channels | 252,503 /1,327,104 | 170,921 /1,327,104 |
| Different pixels | 175,282 /442,368 | 118,827 /442,368 |

Mean byte error falls26.2811%, but the worst localized residual is largely
unchanged. The maximum remains at cropped coordinate x184,y524, red channel,
near the gap between the image-left sleeve and torso. Amplified differences
also highlight the hem/waist and neckline. There are336 versus328 pixels
whose largest channel error exceeds16, and285 versus278 exceeding32.
The99th-percentile channel error is2 in both modes; the99.9th percentile is
11 for all-Q8 and10 for mixed. The lower average error therefore must not be
described as eliminating these concentrated boundary differences.

Normal-scale contact sheets and full-resolution floating/mixed outputs
retain the same overall pose, face, cardigan color/shape and graphic.
No obvious new major visual failure is apparent. The original model's
graphic, hem and construction inaccuracies remain. This is numerical
improvement on one independent image, not broad perceptual or identity
validation. It is weaker than the improvement on the calibration case.

Direct Q8 versus mixed cardigan has MAE0.13324125, max15 and PSNR56.43386dB;
neither is ground truth. No complete floating-reference cardigan trajectory
was recorded: its final images are compared against native floating output,
whose original-Python agreement was established separately. All saved native image/velocity
states were finite F32 with the expected shape; that is not intermediate
reference agreement.

Evidence: `..\checkpoint26-q8-cardigan` and
`..\checkpoint30-mixed-top4-cardigan`, with their matching `-measurement`
directories; `comparison-gallery-v6` contains all-Q8/floating, mixed/floating
and all-Q8/mixed rows, full-resolution PNGs and amplified differences.

### Completed: matched floating recorder control and performance

The fresh BF16-residency/F32-compute bottoms recorder uses the exact same
executable, checkpoint, conditions, settings and16threads as the Q8/mixed
recorders. All three executable SHA256 values are
`06e04b13b021c969b02276848c84eb5e8423d3643a7f95740191a74d8f07b0c3`.
These are serial single observations, not interleaved repeated benchmarks.

| Bottoms recorder | Floating BF16 storage/F32 compute | All104 Q8 | Mixed100 Q8 +4 F32 |
|---|---:|---:|---:|
| Process wall time | 2640.8716s (44m01s) | 1947.3476s (32m27s) | 2007.3931s (33m27s) |
| Peak working set | 3,716,132,864 bytes (3.461GiB) | 4,457,504,768 (4.151GiB) | 4,592,234,496 (4.277GiB) |
| Sampled peak private bytes | 1,908,715,520 | 2,804,740,096 | 3,001,339,904 |
| Wall time reduction versus floating recorder | Reference | 26.2612% | 23.9875% |

The Q8 and mixed representations did **not** produce lower process-memory
peaks in this diagnostic path. Smaller weight payload must not be equated
with lower end-to-end working set. Loading/conversion, mappings, graph
buffers and recording are included; this experiment does not isolate which
allocation causes the difference. It is not a benchmark of a public
prequantized-GGUF deployment, which remains gated.

The floating control passes every reference image/velocity state: maximum
image relative L2 is0.00000201699, maximum velocity L2 is0.00000245062,
initial-noise max absolute error is4.76837e-7 and schedule error is0.
Its final cropped PNG is **byte-identical to the historical native CLI
output**, confirming the displayed floating baseline. Against the F32-math
Python reference, only43 byte channels differ by one level.

This recorder comparison does not reverse the actual-upstream finding:
original Python took1325.48s for bottoms and1258.04s for cardigan, faster
than the native Q8/mixed observations. Different startup and recording
paths, plus the lack of repeated interleaved trials, prohibit a universal
language-level speed claim.

Evidence: `..\checkpoint26-bf16-bottoms-control`,
`..\checkpoint26-bf16-bottoms-control-measurement\measurement.json` and
`..\checkpoint26-bf16-bottoms-control-comparison\comparison.json`.

### Earlier checkpoint: 40/104

The first40 interventions now cover all eight patch-mixer matrices and
32 matrices from double-stream blocks0-3. Eleven reduced the chosen error
objective slightly;29 worsened it. None passed the unchanged0.001 gate.

| Best individual restoration among these40 | Worst-branch velocity L2 | Relative error reduction |
|---|---:|---:|
| `double_blocks.0.img_mlp.2.weight` | 0.0026206027 | 0.17025% |
| `double_blocks.0.txt_mlp.0.weight` | 0.0026224456 | 0.10005% |
| `double_blocks.3.txt_mlp.0.weight` | 0.0026224535 | 0.09974% |
| `double_blocks.1.img_mlp.2.weight` | 0.0026231842 | 0.07191% |
| `double_blocks.1.txt_mlp.2.weight` | 0.0026236044 | 0.05590% |

The best observed reduction is still small. Restoring that matrix adds
18.359MiB of weight payload versus Q8, before alignment/runtime effects.
This is not a selected deployment policy or proof that later matrices cannot
have much larger effects. The completed scan above subsequently identified
the stronger late-single-block intervention.

`matrix-sensitivity-snapshot-1` contains a hash-verified snapshot, complete
CSV/Markdown ranking and added weight-payload costs. It captured41 completed
interventions, including the first result of the next batch. No combined
policy files are generated while the104-matrix study is incomplete.
The completed export now proposes top1/top2/top4 positive-effect subsets,
explicitly marked unvalidated until joint/held-out/image experiments.

### Earlier checkpoint: first8/104 interventions

Calibration uses photographic bottoms conditions at the initial timestep,
seed42, the same pinned model and16threads. Each intervention restores one
matrix from the floating checkpoint; all other eligible matrices remain Q8.
The multiplication for the restored matrix also returns to the F32 operand
path, so this measures a deployable restoration effect, not weight rounding
in isolation.

All-Q8 baseline: conditional velocity relative L2 **0.0026250719**, null
velocity **0.0016766500**, maximum captured-layer relative L2 **0.03136824**,
19/34 failed captures. This is the new photographic baseline, not the older
synthetic fixture with a larger maximum error.

The objective below is the larger of the two branch velocity errors.
Positive reduction is better; negative means the intervention made it worse.

| Restored matrix | Worst-branch velocity relative L2 | Reduction versus baseline |
|---|---:|---:|
| `x_patch_mixer.0.linear1.weight` | 0.0026262294 | -0.0441% |
| `x_patch_mixer.0.linear2.weight` | 0.0026329933 | -0.3018% |
| `x_patch_mixer.1.linear1.weight` | 0.0026326414 | -0.2884% |
| `x_patch_mixer.1.linear2.weight` | 0.0026321304 | -0.2689% |
| `x_patch_mixer.2.linear1.weight` | 0.0026306769 | -0.2135% |
| `x_patch_mixer.2.linear2.weight` | 0.0026294148 | -0.1654% |
| `x_patch_mixer.3.linear1.weight` | 0.0026239173 | +0.0440% |
| `x_patch_mixer.3.linear2.weight` | 0.0026244310 | +0.0244% |

None materially improves the selected output metric, and all still exceed
0.001. Restoring mixer0/linear2 reduced the failed-capture count from19 to18
while making the velocity objective slightly worse: failed-capture counts
alone are not a good optimization objective.

The tiny positive changes are single-probe point estimates, not robust
quality wins. This does not test restoring all eight jointly; interactions
remain possible. It does not justify blanket restoration of the patch mixers
or identify the worst matrices among the other96. Subsequent coverage is
reported above.

The reference exporter now supports real nonzero-time input states from a
complete reference trajectory. The ready step15 probe uses the updated image
from step14, not freshly generated noise at a later time. It validates
source/checkpoint/input hashes, settings, schedule and state artifacts.
This is a teacher-forced local check; it does not replace free-running
full-image validation of a combined precision policy.

Evidence: `..\checkpoint28-matrix-ablation\study.json` and its per-matrix
policy, comparison, measurement and log files. The real-data reference is
`..\checkpoint28-photo-oracle`, with native-layout fixtures beside it.

### Completion and remaining limits

The requested experiment batch is complete, including all five diagnostic
full-image runs and both actual-upstream comparisons. No model experiment
remains running. Generation success, numerical agreement and visual review
are separate outcomes.

Q8 and mixed can generate complete diagnostic images; they are not prevented
by an unsupported CPU kernel. Public acceptance remains blocked by the
unchanged numerical gates. The four-matrix subset improves both image pairs
but is not proven minimal or globally optimal, and does not eliminate the
cardigan boundary residual. Weight-only Q8/F32 computation was tested on two
local states, not as another full-image trajectory. Lower-bit Q4/Q5, GPU
execution, broader quantized image/seed coverage and30/50-step sweeps have
not been established by this batch.

Any further policy tuning would need a broader calibration set and separate
evaluation cases, including localized boundary metrics rather than only
mean error. Reusing the cardigan result to tune its policy would make it
calibration data, not an independent held-out result. No existing gate was
loosened to turn the present observations into a numerical pass.

The graph diagnostic now has `--no-capture-files` so the sensitivity sweep
does not fill disk with hundreds of large activation dumps. Metrics, logs,
actual precision maps and provenance remain persistent.

## Artifacts

- Final gallery: `comparison-gallery-v6\index.html`.
- Q8 contact sheet: `comparison-gallery-v2\bottoms-q8\side-by-side.jpg`.
- Full-resolution Q8/floating PNGs and amplified difference are beside it.
- Quantized trajectory: `..\checkpoint26-q8-bottoms`.
- Strict trajectory comparison: `..\checkpoint26-q8-bottoms-comparison`.
- Timing/provenance: `..\checkpoint26-q8-bottoms-measurement\measurement.json`.
- Complete historical implementation report: `implementation-and-experiment-history.md`.
- Full matrix ranking and candidate policy files: `matrix-sensitivity-complete`.
- Combined state probes: `..\checkpoint29-policy-evaluation` (all12 complete).
- Complete measured image/provenance snapshot: `full-image-precision-summary.json`.
- Final visual review: `comparison-gallery-v6\visual-review.json`.
- Actual original-sampler comparisons:
  `..\checkpoint27-cardigan-comparison-v2\comparison.json` and
  `..\checkpoint27-bottoms-comparison\comparison.json`.

CatVTON inputs remain under the repository-stated CC BY-NC-SA4.0 terms;
original FASHN examples are identified in the gallery. Research/evaluation
use and applicable attribution restrictions remain in force.
