# Platform readiness and evidence

Readiness describes a prerequisite or capability; validation records an actual
build or execution. Use the [Windows](platform-windows.md) or
[Linux](platform-linux.md) guide for setup, [backend reference](backend-notes.md)
for selectable paths, and [GPU qualification](gpu-qualification.md) for the run plan.

## Evidence classes

| Class | Establishes | Owner |
|---|---|---|
| Discovered | A tool, header, library or target was found | `tools/check_dependencies.py` and CMake find modules |
| Implemented | A real path exists with explicit contracts and rejection behavior | [Public headers](../include/rns8/rns8.h), source and [backend reference](backend-notes.md) |
| Compiled | A recorded configuration produced the target objects/executables | Build log and immutable manifest |
| Device validated | The applicable cases executed with independent exact references on that device | Dated correctness record and raw logs |
| Measured | Matched workloads executed at a stated timing/output boundary | Dated performance report |
| Promoted | Reviewed evidence authorizes a specific route/cache key | [Performance gates](performance.md) and explicit cache review |

These classes are separate. A source implementation may be available on targets
without device evidence. A measured result may still lack promotion gates.

## Recorded platform boundary

The [October 23 expanded record](local-all-features-20261023.md) documents
`c7f7e05`, Windows, VS2026/HIP 7.1 and RX 7900 XTX/gfx1100 with the locally
available optional backends enabled. It identifies the target-inapplicable
sparse skips and distinguishes earlier sanitizer/rocWMMA configurations.
The [larger-workload measurements](7900xtx-large-throughput-2026-10-23.md) use that
recorded stack and keep public/native/residue boundaries separate. A later source
revision requires its own evidence for changed paths.

Linux CPU CI evidence is recorded independently. Linux Radeon/Instinct GPU,
RDNA2/RDNA4 and CDNA2/CDNA3/CDNA4 requests remain per-device qualification work.
The [README hardware matrix](../README.md#hardware-tested-and-hardware-wanted)
links AMD's release-specific compatibility sources; source coverage target lists
cannot establish installable or validated stacks.

## Dependency reports and optional backends

The dependency checker collects host commands, SDK/device metadata, package
presence, optional components and preset coverage. Its JSON records use
`evidence_class=candidate_accelerator_evidence_only` and
`candidate_evidence_is_correctness_validation=false` for probes. Those fields
describe the probe, not the absence of an implemented runtime backend.

`readiness.correctness_backend_validation.validated_by_this_report` remains
empty, with count zero, because discovery runs no backend differential campaign.
`exact_wide_platform_validation` keeps Windows evidence separate from Linux and
Instinct. `hard_cut_self_checks` tests report consistency rather than hardware.
Some discovery fields retain conservative historical enablement wording; consult
compiled capabilities and the dated records for actual runtime support.

hipBLASLt, CK, rocWMMA and AMDGPU builtins are implemented opt-in public paths.
They require `RNS8_ENABLE_HIP=ON`, their explicit enable switches, compatible
components and the selected target. Probe-only presets disable those switches.
Invalid combinations fail configuration; enabling AMDGPU builtins with HIP is
not universally rejected. Its source includes target-specific WMMA/MFMA and
sparse instructions, with runtime restrictions described in the backend guide.

`rns8_get_backend_capability_info` reports compiled/source declarations.
`rns8_get_plan_backend_info` adds the selected kernel, accumulation and workspace.
Neither runs validation. Use the recorded test configuration to establish what
was actually exercised.

## Exact-wide ABI owner

The authoritative [exact-wide contract](correctness.md#exact-wide-output) covers
bound-none descriptors, 1–32 little-endian limbs, element strides, signed
centering, sufficient represented range and export destination preservation.
The [export/error rules](correctness.md#export-and-error-handling) cover status
and currentness. Setup/readiness prose does not redefine arithmetic or device
storage semantics.

## Remaining setup acceptance

[#57](https://github.com/doublemover/RNS8/issues/57) owns automatic toolchain and
dependency-root discovery, [#58](https://github.com/doublemover/RNS8/issues/58)
owns reproducible bounded Linux GPU sessions, and
[#60](https://github.com/doublemover/RNS8/issues/60) owns compatibility-gated
upgrades and locks. Current setup uses explicit overrides and installed/pinned
components. Documentation checks verify names, paths and links; clean-install
configure/build/example execution still needs each matching supported OS/GPU.
