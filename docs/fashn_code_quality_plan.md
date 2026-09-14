# FASHN fork: coding-quality audit and remediation plan

**Status: host-side implementation completed after this baseline audit.**
See [quality gates and maintenance](fashn_quality.md) for the implemented behavior,
commands and explicit acceptance limits. The findings and original checkpoint
criteria below are retained as the design record, not rewritten as historical
claims of tests that were never run. Device execution and observed GitHub workflow
runs remain external acceptance steps.

Audited source: `e86c564fd42d876b30befcaf4a782e437f0fe134` (`q4,q5`).
Upstream comparison base: `d04e8950c1ec8d30248cbe996682b3182fb1adf6`.
GGML pin: `e20c3a14aa70ee84ca58499814206dd08d8026bc`.
File locations below refer to the audited revision, before this document.

## 1. Verdict and limits

The fork has substantial correctness and reproducibility work, but it does **not**
uniformly satisfy its documented engineering practices. The most important gaps
are automated enforcement and failure handling, not whether every function is
short or every implementation uses a particular design pattern.

The audit inventoried **269 owned code files / 106,581 lines**, including
**117 files changed by the fork** relative to the upstream base. It combined
that census with targeted inspection of FASHN runtime and API boundaries,
shared infrastructure, server execution, preprocessing, UI, Python tooling,
formatting scripts, CMake tests, and CI. It did **not** manually review every
line or prove every supported model correct.

Excluded: GGML and other dependencies, the independent server frontend
submodule, tokenizer vocabulary data, local models, generated build contents
except existing test execution, and large generated report/image payloads.
This is a maintainability and reliability review, not a security certification.
Other model families received limited structural coverage, not new inference
or numerical validation.

Use [AGENTS.md](../AGENTS.md) for agent boundaries and
[CONTRIBUTING.md](../CONTRIBUTING.md) for contribution/style policy. Do not
duplicate or silently replace those policies with a new style guide.

## 2. What is already working well

- The public FASHN path validates prepared shapes, channels, normalized values,
  crop bounds, categories, sampling parameters, and versioned structure sizes.
  Quantized public generation remains explicitly rejected.
- Sampling checks schedule monotonicity and finite values, and uses scoped
  runner/request cleanup. Request-local cancellation/progress avoids replacing
  global callbacks for every request.
- The server serializes model access, accounts for retained image capacity,
  releases prepared/raw data, and arbitrates cancellation versus completion
  under its job-manager mutex.
- The UI preserves uint64 seeds and retains accepted job identity through
  transport failures. It guards against a late cancellation response affecting
  a newer job.
- Conversion verification checks actual quantized blocks with the converter's
  importance-vector policy. Reports distinguish storage, arithmetic, process
  working set, private commit, and numerical drift.
- Experiment provenance, exact artifact hashes, architecture checks, and
  explicit research-license consent are valuable safeguards.
- Existing helpers already provide atomic text replacement, owned-process
  cleanup, scoped memory accounting, parser ownership, and graph cleanup.
  Reuse these rather than introducing a general-purpose framework.

The plan must preserve these properties.

## 3. Findings

Priority: **P1** = address before expanding deployment or relying on automation;
**P2** = bounded maintainability/reproducibility improvement.
These priorities are not vulnerability severity ratings.

| ID | Priority | Evidence and location | Consequence / disposition |
|---|---|---|---|
| CQ-01 | P1 | `.github/workflows/build.yml:10-46` lists `master` and `ci`, not this fork's `main`, for push events. Path filters omit Python and the owned JS/HTML surfaces. The inspected build workflow does not invoke CTest, Python/UI tests, formatting, or static analysis. | A successful manual run is not an enforced regression gate. Fix triggers and add a focused fork workflow; do not replace the entire upstream build matrix. |
| CQ-02 | P1 | Both `scripts/format-code.sh` and `scripts/format-code.ps1` explicitly include `src/tokenizers/vocab/*`, but their exclusion tests match only paths beginning with `vocab`. They omit `include`, `tests`, native preprocessing, and server headers. PowerShell does not check the external formatter's exit status. | The normal formatter can enter excluded data, miss owned code, and report apparent success after tool failure. Fix file selection and failure propagation before running it. |
| CQ-03 | P1 | `src/stable-diffusion.cpp:3987-4005` allocates a C wrapper with `malloc`, then uses throwing `new` and initialization without an exception boundary. `generate_try_on_with_callbacks` around `4109-4254` also performs throwing C++ operations without translating exceptions at the exported boundary. | Construction/initialization exceptions can bypass cleanup; an exception can escape an API intended for C callers. Static control-flow finding, not a reproduced out-of-memory incident. Add ownership and explicit boundary failure translation. |
| CQ-04 | P1 | `examples/server/main.cpp:166` directly starts `async_job_worker`; `examples/server/async_jobs.cpp:368-471` dispatches work and finalizes jobs without exception containment. | A thrown allocation, encoding, or execution exception can leave the thread via its entry function and terminate the process. Current boolean-error handling is not sufficient for exceptions. Add bounded recovery and worker-lifetime ownership. |
| CQ-05 | P1 | `examples/server/runtime.cpp:34-48` accumulates Base64 input in signed `int` using `val << 8`, and shifts it again for the final sextet. This code is inherited from the comparison base. | In C++17, left-shifting a negative signed operand is undefined behavior. For ordinary 32-bit `int`, the PNG signature makes the accumulator negative before the fifth iteration. Replace signed bit accumulation with defined unsigned operations and exact-output tests. No observed output corruption is attributed to this finding. |
| CQ-06 | P1 | `tests/CMakeLists.txt:14-20` registers optional conversions only for `f32`, `bf16`, `f16`, and `q8_0`. Python, UI, full-model, and device checks have separate invocation/setup conventions. | Q4/Q5 verification was performed during the study, but its complete supported-policy matrix is not registered as a repeatable build gate. Add layered registration without making ordinary PRs download large weights. |
| CQ-07 | P2 | `scripts/run_fashn_q45_study.py:201-213` intentionally refuses any incomplete job. Recovery of the actual interrupted experiment required a session-only helper. `scripts/run_fashn_portability.py:67-78` directly writes state around an unbounded `subprocess.run`, without interrupted/failure finalization. | Fail-closed provenance is correct, but safe operator recovery is not a first-class committed workflow. Introduce versioned state/recovery tooling; do not loosen the frozen bundle checks. |
| CQ-08 | P2 | `build_fashn_comparison_report.py:17` imports utilities from the study driver; publication code imports through report/study modules. `Study` handles execution, measurement, selection, state, and publishing. | Small reusable functions depend on command orchestration and its dependencies. Extract only neutral helpers with stable tests; leave experiment-specific policy in the driver. |
| CQ-09 | P2 | `build_fashn_comparison_report.py:20-22` hardcodes the selected mixed policy as Q5_K. `fashn_comparison_template.html:266-270` hardcodes Q5_K repeat/seed rows, while the study selects a policy dynamically. | Correct for the completed published study, but unsafe to reuse for a different selected policy. Either explicitly reject unsupported study variants or derive all labels and row selection from recorded metadata. Preserve the historical publication. |
| CQ-10 | P2 | Read-only clang-format 19.1.5 checks failed for **39 of 62 fork-touched C/C++ files**. `.clang-tidy` provides a narrow modernize configuration, without demonstrated workflow enforcement. | Style is not uniformly enforced. These are whole-file formatting findings, including potentially inherited lines, not 39 functional defects. Use a touched-code baseline and gradual analysis rather than mass cleanup. |
| CQ-11 | P2 | `src/stable-diffusion.cpp` has 7,581 lines; FASHN dispatch, prepared-image conversion, generation orchestration, and result allocation live in the shared entry implementation. The model runner also exposes diagnostic/capture/cache responsibilities. | These are change-isolation and testability concerns, not proof that the architecture is broken. Extract the FASHN-specific orchestration only after failure-path tests exist. Avoid rewriting unrelated model families. |

### Important distinctions

`free_sd_ctx` currently dereferences its argument. The inspected header does not
establish a null-safe contract, so this audit does not label a null caller a
demonstrated library bug. Decide and document that contract before changing it.

The core runner's existing `optional<Tensor>` methods are inherited. In
`src/core/ggml_runner.cpp:845`, success without a returned tensor can be an
engaged empty tensor, distinct from failure. Do **not** mechanically replace
those interfaces. New ordinary internal tensor-returning APIs should follow
the empty-tensor convention; an additional state needs a demonstrated purpose.

Cancellation between forwards is documented at
`include/stable-diffusion.h:538-540`. It is not currently a promise of immediate
mid-graph interruption. Repeated condition scans in the model runner may be
worth profiling, but this audit has not established them as a material latency
bottleneck. Neither observation justifies weakening validation.

## 4. Verification performed for this audit

| Check | Result | What it establishes |
|---|---|---|
| Owned-code census and fork diff | 269 files; 106,581 lines; 117 fork-changed files | Review scope, not complete manual coverage |
| clang-format 19.1.5, `--dry-run --Werror` | 62 touched C/C++ files checked; 23 passed, 39 failed | Reproducible formatting debt; no source edits |
| Existing Release CTest build | 13/13 passed | Available contract, primitive, sampling, precision, memory and math regressions |
| Six focused Python unittest modules | 36/36 passed; no skips | Report/study/weight/trajectory/probe/portability checks, including enabled native rejection and prepared-lineage tests |
| `node --test scripts\test_fashn_vton_ui.cjs` | 7/7 passed | Mock-DOM/UI contract tests, not a fresh real-browser run |
| Git status before writing this plan | Clean | Audit checks did not modify implementation |

Python modules run from `scripts`:

```text
test_fashn_comparison_report
test_fashn_q45_study
test_fashn_runtime_weights
test_fashn_trajectories
test_fashn_trajectory_probe
test_fashn_portability
```

`FASHN_TRAJECTORY_EXE` and `FASHN_BUNDLE` were set to the existing diagnostic
binary and original verified BF16 transfer bundle for the optional tests.
CTest used the existing Release build, not a clean rebuild during this audit.

**Not run in this audit:** new full-model generations, fault-injection tests,
sanitizers, clang-tidy, a fresh compiler matrix, live HTTP/browser integration,
native preparation/oracle integration, ARM64 execution, or Android execution.
Earlier successful experiments remain historical evidence, not new coverage.

## 5. Sequential implementation checkpoints

Execute in order after implementation is authorized. Each checkpoint should be
a small reviewable change or a clearly separated pair of changes. Passing a
checkpoint means its acceptance checks are recorded, not merely that it compiles.
Do not commit or push automatically.

### QL0 - Freeze the baseline and define test tiers

**Files:** `tests/CMakeLists.txt`, test documentation, existing script entry points.
**Dependency:** none. **Covers:** CQ-06 foundation.

1. Record source/compiler/GGML identities, enabled options, and test inventories.
2. Classify checks as `fast`, `model`, `preprocess`, `server`, `publication`,
   and `device`. Give each an explicit command, fixtures, and prerequisites.
3. Keep small synthetic fixtures separate from licensed/restricted assets and
   checkpoint-scale data. Never silently mark unavailable model tests as passed.
4. Record existing numerical and output references without regenerating them.

**Acceptance:** a clean checkout can configure and run the fast tier without
weights, optional preparation SDKs, a GPU, or the private transfer folder.
Model/device absence is explicit in the reported test inventory.

### QL1 - Add reliable fork CI

**Files:** a focused workflow under `.github/workflows`, build/test documentation.
**Dependency:** QL0. **Covers:** CQ-01.

1. Cover pushes to `main` and pull requests. Include owned C/C++, Python, JS,
   HTML templates, CMake, test manifests, and relevant policy/configuration paths.
2. Build with `SD_BUILD_TESTS=ON`; execute fast CTest, selected Python unit
   modules, and the Node UI suite. Use explicit module selection: some existing
   integration scripts require command-line fixtures and are not generic
   unittest-discovery targets.
3. Separate minimal Python test dependencies from the heavyweight CPU oracle.
   Declare dependencies before installation; do not make torch/ONNX a default
   native build requirement.
4. Start with one portable CPU job and one Windows build/test job. Keep existing
   upstream jobs intact. Add opt-in model lanes with verified local artifacts.
5. Capture failure logs and effective build settings. Document the required
   status-check names; repository branch-protection changes require owner action.

**Acceptance:** representative Python-only and UI-only changes select the
workflow. An intentionally failing native/Python/UI test fails its job. The
ordinary lane has no model download and no dependency on developer absolute paths.

### QL2 - Make formatting and analysis safe

**Files:** both formatting wrappers, shared file-selection helper if needed,
tests for selection, `.clang-tidy` only as justified, focused workflow.
**Dependency:** QL1. **Covers:** CQ-02, CQ-10.

1. Derive candidates from tracked owned files rather than enumerating excluded
   directories. Explicitly exclude vocabulary, dependencies, the frontend
   submodule, and generated/local data before opening files.
2. Include public headers, tests, native preprocessing, and server headers.
   Provide read-only check mode and explicit write mode.
3. Pin or validate the formatter version; propagate missing-tool, configuration,
   and formatting failures in both shells. Test paths containing spaces and
   Windows separators with synthetic fixtures, not real vocabulary files.
4. Establish a checked-in baseline or changed-line gate. Format new FASHN-only
   files in a separate mechanical change; in shared files, avoid unrelated churn.
5. Run existing clang-tidy configuration against actual compile commands;
   inventory warnings before selectively adding high-signal diagnostics.
   Scope gates to owned code, not third-party headers.

**Acceptance:** dry-run opens no excluded file and changes no bytes; omitted
owned directories are included; a nonzero formatter status fails the wrapper.
New/touched code has no unaccounted formatting or analysis regression. Existing
debt is explicit rather than suppressed by a success-shaped fallback.

### QL3 - Harden C API ownership and failure boundaries

**Files:** `src/stable-diffusion.cpp`, `include/stable-diffusion.h`,
focused C/C++ API tests.
**Dependency:** QL2. **Covers:** CQ-03.

1. Hold the outer C context and internal C++ object in scoped ownership until
   construction and initialization succeed. Release ownership only on success.
   Remove ineffective null checks following ordinary throwing `new`.
2. Add explicit exception-to-error translation at exported C boundaries reached
   by FASHN construction/generation. Preserve `nullptr`/`false`, cleared outputs,
   existing allocation/free conventions, and diagnostic logging.
3. Keep cleanup nonthrowing. Low-memory reporting must not require constructing
   another large error object. Do not add broad catches inside tensor/math
   helpers that turn a real failure into an apparently valid empty result.
4. Document callback lifetime, no-throw and reentrancy constraints. Check callback
   paths through sampling, modulation, output allocation, and cleanup.
5. Add narrow fault-injection seams for construction, initialization, tensor/
   result allocation, and callbacks. Do not add process-wide allocator overrides
   to normal builds.

**Acceptance:** controlled exceptions do not cross the C ABI; each owned resource
is released exactly once; output pointers/counts remain failure-shaped; error
logs identify the failure; subsequent use is permitted only when the context
remains valid. C compilation and ABI signatures are unchanged.

### QL4 - Contain worker failures and own thread shutdown

**Files:** `examples/server/async_jobs.cpp`, `async_jobs.h`, `main.cpp`,
server lifecycle tests.
**Dependency:** QL3. **Covers:** CQ-04.

1. Contain exceptions around job execution and finalization, not just the native
   generation call. Preprocessing and encoding can also throw.
2. Centralize the terminal transition while preserving cancellation precedence,
   result ownership, image-budget release, and retention policy.
3. Define recoverable versus context-invalidating failures. A damaged native
   context must not be reused simply because an exception was caught.
4. Add scoped stop/notify/join ownership so an exception after thread creation
   cannot destroy a joinable thread during server setup/shutdown.
5. Preserve explicit diagnostics. If even terminal-state reporting cannot
   allocate, use a bounded fatal-worker path and controlled shutdown rather than
   trying to continue with an unrecorded generating job.

**Acceptance:** injected preprocessing, generation, encoding, and finalization
failures do not unexpectedly terminate the service; jobs reach a documented
terminal state or controlled shutdown; memory charges are released correctly;
a healthy context accepts the next request; cancellation/shutdown races pass.

### QL5 - Remove inherited undefined bit operations

**Files:** `examples/server/runtime.cpp`, small server utility tests.
**Dependency:** QL4. **Covers:** CQ-05.

Use a defined unsigned accumulator or an existing compatible owned helper;
preserve the exact Base64 alphabet and padding. There is no reason to introduce
a new encoding dependency for this change.

**Acceptance:** standard Base64 vectors, empty input, all byte values,
high-bit-heavy inputs, PNG signatures, and lengths across padding boundaries
match a trusted reference. Run focused UBSan on a supported host and preserve
server PNG round-trip behavior. Keep this inherited correction separate from
FASHN numerical changes.

### QL6 - Complete precision and failure-path test registration

**Files:** `tests/CMakeLists.txt`, conversion/runtime tests, focused workflow.
**Dependency:** QL5. **Covers:** CQ-06 and regression protection for QL3-QL5.

1. Cover Q4_0, Q5_0, Q4_K, and Q5_K as well as existing formats, respecting the
   exact type spelling and row-block eligibility.
2. Use small synthetic tests for dispatch, protected F32 tensors, importance
   policy, quantized matmul selection, and invalid policy/metadata cases.
3. Register optional actual conversion/trajectory checks with explicit fixture
   requirements. Represent mixed policies from their manifest instead of
   pretending every matrix has the same type.
4. Include owned API/server failure tests in an appropriate non-model tier.
   Add optional preparation SDK and CPU compiler/architecture lanes separately.

**Acceptance:** the test inventory includes every supported diagnostic format;
each fast quantization path is exercised without full model files; requesting
a missing optional fixture fails clearly. Quantized public initialization is
still rejected. Direct Q4/Q5 execution cannot silently upcast all matrices.

### QL7 - Make experiment recovery a supported workflow

**Files:** study driver, neutral state/process utilities, portability tooling
or a versioned successor, focused Python tests.
**Dependency:** QL6. **Covers:** CQ-07.

1. Define versioned `pending/running/completed/failed/interrupted` state and
   explicit attempt identities. Reuse atomic-write and owned-process helpers.
2. Expose explicit recovery: verify provenance and completed hashes, establish
   that the recorded process is no longer the owned active process, archive
   only that attempt, then allow a new attempt with unchanged inputs.
3. Record process creation identity as well as PID to avoid PID-reuse mistakes.
   Preserve interrupted logs; never kill by executable name or guess ownership.
4. Add timeout/cancellation controls and guaranteed failure-state finalization
   where feasible. An abrupt power loss still requires recovery on restart.
5. Keep frozen BF16/Q4_K bundles and historical study records immutable. If the
   harness contract changes, use a new schema/run location and explicit migration.

**Acceptance:** interruption before launch, during inference, and during report
publication is recoverable; concurrent recovery is rejected; changed artifacts
and provenance remain rejected; completed measurements are never rerun silently.
Mock long-running processes test the workflow without hours of inference.

### QL8 - Separate reusable analysis from orchestration

**Files:** study/report/publication modules, template, related unit tests.
**Dependency:** QL7. **Covers:** CQ-08, CQ-09.

1. Extract only shared hashing, atomic state, pixel metrics, and region selection
   helpers into neutral modules. Avoid cycles and a new generic experiment engine.
2. Keep fixture selection, historical paths, and experiment policy in commands.
   Make platform-specific measurement adapters explicit.
3. Derive selected-policy labels, repeats, seed pairs, and mixed-matrix counts
   from validated metadata, or explicitly reject unsupported report variants.
4. Validate inputs before publication and stage generated files before replacing
   a published set. Write the verified publication manifest last; document that
   a multi-file replacement is not automatically an atomic transaction.

**Acceptance:** helper tests import without execution side effects; synthetic
studies selecting Q4_K and Q5_K render truthful labels or clear errors; failed
publication cannot claim a complete manifest. Numerical metric definitions,
image conversion/crop conventions, and historical publication bytes remain
unchanged unless intentionally versioned.

### QL9 - Isolate FASHN orchestration without redesigning the library

**Files:** shared public entry implementation, a small internal FASHN runtime
module, existing model/configuration and test files as needed.
**Dependency:** QL8. **Covers:** CQ-11.

1. Keep exported C wrappers thin; move prepared-input conversion, request
   orchestration, and result construction behind one internal FASHN boundary.
   Do not expose the generic `StableDiffusionGGML` implementation in a new
   public header merely to make extraction convenient.
2. Reuse `FashnVTONConfig` for released architecture facts rather than adding
   another competing constant registry. Keep public validation independent of
   loading weights and preserve the fixed released canvas.
3. Keep production execution independent of diagnostic report/file writing.
   Preserve useful observers, captures, and accounting without adding hot-path
   allocations when diagnostics are disabled.
4. Do not combine this extraction with numerical optimization, cache redesign,
   tensor-layout changes, optional-return rewrites, or `.hpp` renaming.

**Acceptance:** public ABI and ordinary model dispatch remain unchanged;
FASHN deterministic outputs and captured trajectories preserve existing gates;
repeated success/failure requests leave no new retained memory. Review the diff
as code movement plus small explicit adapters, not a new runtime architecture.

### QL10 - Portability validation and closeout

**Files:** relevant build/test documentation and portability manifests; device
harness changes only in coordination with the ARM-machine work.
**Dependency:** QL9.

1. Clean-build CPU configurations with supported Windows and Linux compilers,
   then cross-build Android ARM64. Preparation SDKs remain optional.
2. Run the native Windows ARM64 checks on that machine and Android checks on
   the device; label cross-compilation separately from successful execution.
3. Preserve thread/load-thread controls and model memory measurements. Measure
   quality refactors separately from performance experiments.
4. Keep Android thermal protections enabled. Cancellation timing and stop
   policies must respect the documented between-forward boundary; an emergency
   harness stop targets only its verified owned process.
5. Update the coverage table, close findings with evidence, and list unresolved
   inherited debt. Do not declare every upstream model verified.

**Acceptance:** clean fast suites and supported builds pass; representative
model/device results are recorded or explicitly blocked; same-policy ARM/Android
drift is distinguished from known quantization-versus-floating drift. No claim
that Q4_K alone eliminates S23 thermal throttling.

## 6. Regression invariants and completion criteria

| Area | Required invariant |
|---|---|
| Public interface | Existing signatures, structure-size handling, output ownership and quantized guard preserved |
| Tensor/math | Low-dimension-first layout and implicit trailing singleton semantics preserved; no silent invalid/failure tensor |
| Numerical baseline | Existing schedule/noise/per-step gates remain unchanged; quantized policies are not reclassified as floating-parity passes |
| Reproducibility | Same inputs, arithmetic policy, seed and build settings remain identifiable; imported-noise references retain their lineage |
| Memory | No leaked context/job/result ownership; distinguish transient peak from retained memory and working set from private commit |
| Measurement | No fixed latency guarantee across different machines; use identical settings and repeated runs before attributing a regression |
| Reports | Actual outputs remain distinct from amplified error maps; local/global differences and failed quality gates remain visible |
| Source boundaries | No dependency/frontend/vocabulary edits, generated binaries, model weights, private absolute paths, or secrets |

Use existing trajectory tolerances for same-reference regression checks:
initial-noise max absolute error `<= 1e-6`, schedule max absolute error
`<= 1e-7`, and per-step image/velocity relative L2 `<= 0.001` where the
existing verifier applies them. These gates do not imply Q4/Q5 passes against
the floating baseline; the published study says otherwise.

Formatting findings are closed when the scoped gate is clean or a reviewed
inherited baseline accounts for them. Reliability findings require negative
tests, not just successful inference. CI findings require demonstrated trigger
and failure behavior, not just valid-looking YAML.

## 7. Deliberate non-goals

No wholesale rewrite of the 106k-line owned tree; no mandatory class hierarchy,
generic framework, blanket smart-pointer conversion, mass header rename, or
unrelated formatting sweep. No changes to historical comparison results or
frozen transfer bundles. No automatic commits, pushes, branch-protection edits,
or coordination changes on the other machine.

The recommended first implementation tranche is **QL0-QL6**: make coverage
repeatable, tools safe, and failures contained. **QL7-QL10** improve long-term
recovery, reuse, change isolation, and portability after those protections exist.
