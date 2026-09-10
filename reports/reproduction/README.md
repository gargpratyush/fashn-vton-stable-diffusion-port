# Historical orchestration and analysis source

The supported argument-driven runtime/reference/comparison tools are in
[`scripts`](../../scripts). This folder additionally preserves the one-off
batch and analysis source used for the final experiments and HTML history.

They are **documentary `.py.txt` snapshots**, not installed executables or
automatically run tests. They assume the original Windows experiment tree,
completed earlier fixtures and models that are intentionally not bundled.
Historical home-directory prefixes are replaced with `<USER_HOME>`;
review/adapt paths and choose new output directories before running a
working copy. Some batches perform hours of CPU inference and must not
be run merely to open or regenerate a report.

| Snapshot | Purpose |
|---|---|
| [checkpoint35_controls.py.txt](checkpoint35_controls.py.txt) | Alternating CPU/BLAS full-graph controls |
| [checkpoint35_full_acceptance.py.txt](checkpoint35_full_acceptance.py.txt) | Full original tops/bottoms BLAS trajectories |
| [checkpoint38_integrated.py.txt](checkpoint38_integrated.py.txt) | Final controls, six full recordings and same-policy comparisons |
| [finalize_fashn_optimizations.py.txt](finalize_fashn_optimizations.py.txt) | Consolidated memory/error accounting and gallery |
| [probe_openblas_memory.py.txt](probe_openblas_memory.py.txt) | Isolated DLL-only private-commit measurements |
| [build_fashn_project_history.py.txt](build_fashn_project_history.py.txt) | Original comprehensive HTML assembly |
| [verify_fashn_final_artifacts.py.txt](verify_fashn_final_artifacts.py.txt) | Original gallery/source/crop/provenance checks |
| [verify_fashn_project_history.py.txt](verify_fashn_project_history.py.txt) | Original HTML checks using the existing Chromium environment |

Original and publication-copy hashes are recorded in
[`publication-manifest.json`](../publication-manifest.json). Path edits
are a publication transformation, not a new measured model run.
