# Documentation

Choose the task you want to complete. Public API behavior is defined in the
[headers](../include/rns8/rns8.h) and [correctness contract](correctness.md).
Dated records identify the builds and devices actually checked.

- [October 25 focused measurements](focused-closeout-measurements-20261025.md): genuine wide values, scalar-first proof controls and failed held-out models

## Build

- [Windows HIP SDK](platform-windows.md): prerequisites, explicit roots, configure and build
- [Native Linux ROCm](platform-linux.md): Radeon and Instinct selection; WSL boundary
- [Dependencies and pins](../third_party/README.md): shared requirements and installation sources
- [Dependency versions and upgrade gates](dependency-versions.md): selected candidates, installed artifacts and qualification
- [Shared discovery](dependency-discovery.md): root precedence, early configure handoff, evidence and remaining probe gates
- [Readiness terminology](platform-readiness.md): discovery, implementation, compilation and hardware evidence

## Use

- [Checked examples](../examples/README.md) and [installed CMake consumer](../examples/downstream-cmake/README.md)
- [Arithmetic, bounds and exports](correctness.md), with a [short glossary](glossary.md)
- [Architecture and ownership](design.md), [immutable plans](plan-validation.md)
- [Experimental ordered runtime bases](experimental-bases.md): explicit ownership, packing, products, CRT and integer conversion
- [Prepared execution and completion](prepared-execution.md)
- [Resident DAGs](resident-graphs.md) and [exact algebraic epilogues](exact-epilogues.md)
- [Experimental public exact complex composition](complex-residue-composition.md)
- [Experimental public narrow native products](narrow-native-owner.md), [encoded B](encoded-b-native.md) and [block materializations](block-sparse.md#public-generation-bound-owner)
- [Backend capabilities and selection](backend-notes.md)

## Understand and extend

The [research index](research/README.md) groups representation/reconstruction,
kernels/scheduling, planning/generation and application studies. Each topic
keeps its arithmetic contract and public/private boundary beside its source.
The [technical specification](RNS8_RESEARCH_SPEC.md) owns normative architecture
and research acceptance rules; the [roadmap](roadmap.md) owns current gaps and
issue links.

## Benchmark and validate

- [Performance methodology](performance.md), [timing admission](timing-admission.md)
- [Hardware qualification](gpu-qualification.md): test/capture commands and acceptance gates
- [Calibration and planner rates](calibration.md), [native conversion/queue controls](native-calibration.md)
- [Public finite verifier calibration](finite-verifier-calibration.md): replay and full modular-reference cost boundaries; commands and receipt contracts, with measured qualification recorded in the dated reports
- [Public block-sparse calibration](block-sparse-calibration.md): dense, warm/cold owners and observed reuse controls; commands and receipt contracts, with measured qualification recorded in the dated reports
- [Canonical word-modular calibration](word-modular-calibration.md): full-u64 baseline/packed-dot4 and admitted scalar binary64 digit/slab controls; commands and receipt contracts, with measured qualification recorded in the dated reports
- [INT8 instruction-chain probes](instruction-chain-calibration.md): dependent/independent WMMA chains, kernel-event versus probe-inclusive host scopes; commands and receipt contracts, with measured qualification recorded in the dated reports
- [Public split-K calibration](split-k-calibration.md): unsplit/full/window owners, scratch and changed-generation recreation; commands and receipt contracts, with measured qualification recorded in the dated reports
- [Public ragged calibration](ragged-calibration.md): uniform/ragged host boundaries, reconstruction-only reuse and retained capacities; commands and receipt contracts, with measured qualification recorded in the dated reports
- [Retained Gram cohorts](gram-prepared-reuse.md): prepared dense, ordinary/prepared single-source controls and each/once reconstruction; commands and receipt contracts, with measured qualification recorded in the dated reports
- [Exact import and CRT calibration](conversion-calibration.md): scalar/dot4, every-prefix and resident/cold controls; commands and receipt contracts, with measured qualification recorded in the dated reports
- [Canonical wrap64 calibration](wrap64-calibration.md): signed radix/native U8, direct-HIP v4 and corrected rocWMMA controls; commands and receipt contracts, with measured qualification recorded in the dated reports
- [RX 7900 XTX 512³/1024³ results](7900xtx-large-throughput-2026-10-23.md)
- [Native WMMA/queue measurements and `436d235` validation](native-queue-performance-2026-10-23.md)
- [October 25 consolidated local correctness and issue closeout](local-validation-20261025.md)
- [October 23 local correctness record](local-all-features-20261023.md)

- [Genuine exact-wide magnitude controls](exact-wide-magnitude-controls.md): public two/three-limb profiles, exact admission and the completed #73 fixture checks

## Contribute

- [Contributor workflow](../CONTRIBUTING.md): CPU oracle, tests, sanitizers, packaging and documentation checks
- [Offer hardware validation](https://github.com/doublemover/RNS8/issues/new?template=hardware_validation_offer.yml)
- [Documentation ownership and migration inventory](documentation-maintenance.md)
- [Security reporting](../SECURITY.md), [changelog](../CHANGELOG.md)

## History

[Historical records](archive/README.md) preserve earlier audit findings, local
checkpoints, the resident-output proposal and superseded chronological status.
Their dates and source revisions remain intact. Current setup and API guidance
live in the guides above. Raw captures and disassembly stay in ignored `temp/`,
`build/` or `out/` paths; durable reports summarize reviewed evidence.
