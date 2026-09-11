# FASHN VTON project history and evidence

These reports document the implementation and experiments, including
unsuccessful approaches and corrections. They are research evidence, not a
claim of production readiness or public quantized/GPU support.

## Start here

- [Completed Q4/Q5/mixed results](q4-q5-results.md): 18 full generations,
  18 probe processes, actual memory, latency, direct Q8 differences and repeats.
- [All-in-one precision HTML](fashn-all-comparisons.html): download/open
  locally; images/data are embedded, with error maps clearly separated.
- [GitHub-viewable precision images](comparison-gallery-precision/README.md).
- [Android host preparation evidence](android-host-preparation.md) and
  [the ARM64 device handoff](../docs/fashn_arm_transfer.md).
- [Complete chronological HTML history](fashn-vton-project-history.html):
  initial setup, 38 checkpoints, 42 code excerpts, 22 embedded illustrations,
  all 104 matrix interventions and the original reports. Download/open the
  HTML locally; GitHub normally displays HTML as source rather than a website.
- [Final memory/latency results](memory-and-latency-optimization-results.md).
- [Full-resolution comparison gallery](comparison-gallery-optimized/index.html).
- [Earlier implementation and experiment history](implementation-and-experiment-history.md).
- [Original Python and quantization study](quantization-and-upstream-comparison.md).
- [Original integration plan](original-integration-plan.md).
- [Original memory/latency plan](memory-and-latency-optimization-plan.md).
- [All 104 restoration results](matrix-sensitivity-complete/ranking.csv).
- [Historical orchestration/analysis source snapshots](reproduction/README.md).

Checkpoint 31-38 reports preserve individual optimization decisions.
`evidence` contains selected structured summaries, not the large binary
trajectories or weights needed to independently rerun every comparison.
The runtime, exporters, reference tooling, comparators and regression
sources are included in the repository.

## Publication scope and provenance

These are publication copies of local research artifacts. Links to included
code/reports/images are made relative. Links to deliberately omitted local
binaries, full experiment directories and tensor files are labeled
local-only rather than presented as working downloads.

Historical command strings and machine-specific timings describe the
original Windows environment; adapt paths using
[the quickstart](../docs/fashn_quickstart.md). Original input/model/source
hashes remain historical evidence. `publication-manifest.json` records
original and publication-copy hashes so path normalization is not confused
with a new inference experiment. No images were regenerated for publication.

The newer Q4/Q5 publication has its own
[`precision-publication-manifest.json`](precision-publication-manifest.json).
It preserves output PNG bytes and numerical values; the explicitly labeled
README contact sheets resize thumbnails for display only. The old manifest
remains the historical publication ledger, not an inventory of subsequent
source/report additions.

The history's code excerpts are final working-tree snapshots, not invented
per-checkpoint commits. Old future-tense statements and failed measurements
remain in historical reports with their corrections explained by the later
chronology.
Statements such as "uncommitted" describe the recording date, not the later
publication status of this repository.

No checkpoint weights, parser/pose model weights, executables, SDKs, virtual
environments, authentication material or large safetensors trajectories
are included. This publication does not change the numerical acceptance
gates or imply that all raw data is bundled.

## Image and dependency notices

Read [NOTICE.md](NOTICE.md) before redistributing report images or treating
the optional parser as commercially licensed. Image metrics measure
numerical agreement, not garment fidelity, identity or paired benchmark
quality. The six-case photographic suite and repeated comparison rows are
not a large independent evaluation set.
