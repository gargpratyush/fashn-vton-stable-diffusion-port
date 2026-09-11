# Android/Windows ARM64 handoff: host preparation

## Scope

This checkpoint executes the parts of the Android plan that are appropriate
on the existing Windows x64 machine. It does not claim native Windows
ARM64 FASHN execution or Android device inference.

Baseline source is `55db93d6f2536f12428e170acbafd0cca3182c59`, with the
uncommitted handoff overlay documented in
[the transfer runbook](../docs/fashn_arm_transfer.md).
GGML remains at `e20c3a14aa70ee84ca58499814206dd08d8026bc`, unmodified.

## Completed host work

1. Ran nine existing native FASHN contract/primitive/sampling/precision/
   modulation/GELU/math/memory-profile tests: all passed.
2. Added diagnostic `--initial-noise` support to the trajectory executable.
   It accepts one finite F32 CHW/NCHW noise tensor and rejects wrong keys,
   dtype, dimensions, NaN and infinity before model loading.
3. Built the updated x64 diagnostic executable and ran both same-seed CPU
   RNG and identical imported-noise photographic trajectories.
4. Compared every two-step updated image/guided velocity against the
   already pinned F32/MATH PyTorch reference.
5. Installed the missing NDK r28c through the existing Android SDK manager,
   then cross-built the Android ARM64 CPU CLI and nine diagnostics.
6. Inspected all ten ELF executables: ELF64/AArch64, Android linker64,
   16 KiB LOAD alignment, GNU_RELRO, and only libc/libm/libdl dependencies.
7. Added private-bundle assembly/integrity tooling and an architecture-
   checked Windows execution wrapper, plus a detailed target runbook.

The data bundle includes prepared inputs and references, not parser/pose
weights, a Python runtime, raw preprocessing SDKs, or a commercial parser
license. Its exact inventory is generated as `bundle.json`.

## Fresh x64 two-step photographic results

Configuration: black-shirt prepared case, seed 42, two steps, CFG 1.5,
shift 1.5, skip final CFG step, 16 CPU threads, BF16 core storage with F32
computation, F32 flash attention, lazy unmapped loading, modulation cache,
two load threads, unfused strict GELU. Three model forwards per run.
Trajectories and Windows phase memory accounting were enabled.

| Measurement | CPU RNG | Imported PyTorch noise |
|---|---:|---:|
| Native process wall seconds | 195.8395213 | 196.7768579 |
| Peak working set bytes | 2,127,699,968 | 2,127,749,120 |
| Sampled peak private commit bytes | 2,265,948,160 | 2,266,099,712 |
| Initial-noise maximum absolute difference | 4.76837158203125e-7 | 0 |
| Schedule maximum absolute difference | 0 | 0 |
| Worst updated-image relative L2 | 1.2464825406774358e-6 | 1.2429414614085434e-6 |
| Worst guided-velocity relative L2 | 8.689710006182104e-7 | 8.671259059312627e-7 |
| Changed full-canvas byte channels | 81 | 77 |
| Maximum pixel-channel difference | 1 | 1 |
| Original numerical gates | PASS | PASS |

Peak working set is about 1.98 GiB. Working set and private commit overlap
and must not be added. These are short Windows x64 diagnostic runs, not
phone predictions or a new full-20-step latency claim. Lightweight host
tooling activity was not controlled as a dedicated performance study.

The images are two-step smoke/correctness outputs. Numerical agreement
does not imply adequate garment fidelity at two steps.
The control binary/source hashes are retained in
`host-evidence/control-build-provenance.json`. After those controls, an
explicit rejection of an empty noise filename was added and rebuilt; it
does not change valid-input arithmetic. Final transfer binary hashes are
recorded separately in `bundle.json`.

Reference lineage:

```text
Original checkpoint SHA256:
d6cd38286885bc29fa487ea9383f80ffeb95862e7747c630d42c5d3c05bdd35a

Runtime-ready BF16/F32 policy SHA256:
e1f2d13d441f0e4631cf4cf8e1837cd8998e4916b78dfc0ec62c5fe879d087b1

Black-shirt normalized conditions SHA256:
40f01aca7508272f67f1b7db212653706042d79011754b73cfbfd7532eb4fb76
```

The reference is the existing prepared-image F32/MATH oracle, not a newly
modified upstream model. Complete 20-step reference states are transferred
for target-machine acceptance; that long inference was not unnecessarily
rerun on the same x64 host in this checkpoint.

## Android cross-build result

Initial configuration failed because the NDK toolchain file was absent.
The existing SDK manager installed `ndk;28.2.13676358`; configuration and
build then succeeded. No core portability patch or GGML modification was
needed for the selected prepared-input build.

| Property | Actual configuration |
|---|---|
| NDK | 28.2.13676358, r28c |
| Compiler | NDK Clang 19.0.1 |
| Generator | Ninja |
| Target | arm64-v8a, Android API 29 |
| ISA | armv8-a, host-native inference disabled |
| C++/core/GGML | Static linkage |
| Optional components | ORT/OpenCV/WebP/WebM/frontend disabled |
| Compute | CPU; OpenMP/BLAS/LLAMAFILE disabled |
| ELF | 64-bit little-endian AArch64; Android linker64 |
| LOAD alignment | 0x4000 on every inspected executable |
| RELRO | Present |
| NEEDED libraries | libc.so, libm.so, libdl.so |

The original unstripped outputs stay in the local build. Transfer copies
have unnecessary symbol/debug sections removed using the same NDK's
`llvm-strip`.

| Transfer executable | Bytes |
|---|---:|
| sd-cli | 56,339,288 |
| test-fashn-vton | 55,394,760 |
| test-fashn-vton-c-api | 1,818,608 |
| test-fashn-vton-graph | 2,860,224 |
| test-fashn-vton-sampling | 2,253,120 |
| test-fashn-vton-precision | 606,520 |
| test-fashn-gelu | 1,629,976 |
| test-fashn-math | 1,864,920 |
| test-fashn-modulation | 55,025,840 |
| test-fashn-vton-trajectory | 2,893,520 |

These sizes also show that the linked core is not yet a FASHN-only build;
stripping symbols does not remove all other model-family code/data.

**No Android device was listed by ADB.** The executables were inspected,
not executed on Android. Existing compiler warnings about unrelated
missing override annotations were left unchanged.

## Remaining destination gates

Windows ARM64 must establish native PE architecture, primitive behavior,
same-seed/imported-noise comparisons, full-trajectory correctness and its
own latency/memory results. The reported SDXS smoke cannot substitute for
those gates.

Android must establish all runtime gates on actual hardware. The current
memory-profile helper still reports Windows process statistics only;
Android RSS/PSS integration, thermal observations and sustained-run
qualification remain open. APK/JNI, lifecycle/cancellation integration,
raw preparation, quantization promotion, Vulkan and NPU support were not
implemented in this handoff.

Use the runbook's sequence: source + verified bundle, native ARM64 build,
small tests, public two-step PNG, separate RNG/imported-noise trajectories,
host numerical comparison, then full runs.
