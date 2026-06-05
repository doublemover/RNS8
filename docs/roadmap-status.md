# RNS8 Roadmap Status

Status date: 2026-06-05

This file summarizes live implementation status against
[RNS8_RESEARCH_SPEC.md](RNS8_RESEARCH_SPEC.md). The research spec remains the
architecture and roadmap source of truth when details disagree.

## Implemented And Verified

- Phase 0 host foundation: C ABI, C++ wrapper, CMake targets, CPU reference,
  tests, dependency checker, benchmark schema, and result comparison tooling.
- Windows direct-HIP bring-up on Radeon RX 7900 XTX / `gfx1100`: explicit HIP
  SDK compiler integration, device inspection, residue conversion,
  one-modulus ring GEMM, K-block splitting, and CPU differential tests.
- Persistent RNS matrices: direct-HIP matrices own device residue, upload,
  export, and status buffers; resident GEMM/export consume device-current
  residues rather than silently uploading stale host state.
- Bounded i64/u64: CPU and direct-HIP fixed-prefix and per-tile bounded paths
  support exact CRT export with range-error preservation.
- Native vector ALU: the Windows HIP bounded i64/u64 backend supports explicit
  bounded contracts and remains separate from exact-wide, finite, and wrap64
  semantics.
- Exact-wide signed/unsigned: persistent RNS output and fixed-width
  little-endian limb export are implemented for CPU and direct HIP.
- Strict wrap64: CPU byte-limb reference and direct-HIP byte-limb correctness
  paths implement explicit `mod 2^64` semantics without odd-modulus CRT.
- Finite u8: CPU and direct-HIP finite-ring and finite-field paths use explicit
  modulus contracts, prefix-zero storage, and canonical byte export.
- Benchmark schema v4 and review tooling: captures record command line, git
  commit, compiler/HIP/device metadata, backend metadata, phase timings, event
  timing availability, and comparison-baseline status.
- Optional accelerators: hipBLASLt, CK, and rocWMMA have opt-in Windows
  `gfx1100` correctness backends with exact differential and ISA/schema
  coverage where applicable.
- AUTO selection: reviewed release cache hits can select compiled HIP-resident
  accelerator candidates for supported contracts; missing or rejected hits stay
  on the configured correctness path.

## Backend Promotion Status

| Backend | Correctness status | Performance status |
|---|---|---|
| CPU reference | Required reference backend | Baseline only |
| Direct HIP | Required Windows GPU correctness path | Production correctness baseline; not a matrix-engine speed claim |
| Native vector ALU | Bounded i64/u64 correctness backend | Useful explicit backend and long-K `n == 1` microkernel evidence, but the current skinny-GEMV release review keeps reviewed N=1 scenario routing on Direct HIP; not a current AUTO cache winner |
| hipBLASLt | Opt-in correctness baseline | Reviewed Windows `gfx1100` cache wins exist for selected bounded-i64, finite-u8, and exact-wide shapes, including eligible 4096 bounded, finite hot-modulus, and exact-wide entries |
| CK | Opt-in correctness backend | Reviewed Windows `gfx1100` cache wins exist for selected bounded-i64, finite-u8, and exact-wide shapes, including bounded-i64 2048, generic finite-field 2048, and finite ring-255 4096 |
| rocWMMA | Opt-in correctness backend | Reviewed Windows `gfx1100` cache wins exist for selected bounded-u64, finite-u8, and exact-wide shapes, including bounded-u64 2048, generic finite-ring 2048 entries, finite-field 512 entries, and smaller hot finite-u8 shapes; post-fix hot finite-u8 2048 winners are hipBLASLt |
| AMDGPU builtins | Not implemented | Fail-fast until real exact kernels exist |
| Wrap64 matrix-engine candidate | Internal rocWMMA harness only | Not public, not AUTO-selected, and not faster than direct HIP in current reviewed shapes |

Detailed benchmark policy, current wins, and reviewed release summaries live in
[performance-model.md](performance-model.md),
[performance-wins.md](performance-wins.md), and
[reviewed-local-evidence.md](reviewed-local-evidence.md).

## Cleanup Consolidation Status

The current cleanup program is an internal consolidation lane, not a public API
or routing change. The branch now has a checked-in metadata registry under
`metadata/`, generated Python/C++ constants, stale-label registry tests, a repo
hygiene reporter, a compact golden regression runner, and durable documentation
claim validation for target-readiness and speedup wording.

Cleanup adoption has progressed beyond the guardrail slice: benchmark scenario
families now live in data under `benchmarks/scenarios/` with explicit
review-mode and promotion-eligibility contracts backed by the metadata
registry. Benchmark schema validation has a compatibility wrapper over a
package entrypoint, with GPU event, execution-mode, and contract-metadata
validators plus helper/output-policy and backend metadata split into focused
modules and CTest-backed self-tests. Shared benchmark support and semantic-mode
helpers have been split out of
`benchmarks/rns8_bench.cpp`; core output setup, benchmark exact-wide pack
materialization, Direct-HIP output stamping, and native-to-RNS bridge
currentness transitions are helper-routed; HIP event/stream/pinned-staging/
temporary-device-buffer ownership uses internal RAII wrappers; Direct-HIP
timing support is split into `src/backend_hip_direct/hip_timing.cpp`; and
portable non-Windows CPU ASan/UBSan presets are available while Windows MSVC
ASan stays as `CMakeUserPresets.json` guidance for hosts with the optional
runtime installed. The hygiene report now filters intentional helper/RAII
implementation sites so remaining findings point at scattered cleanup debt.

Remaining cleanup work is intentionally incremental: deeper benchmark semantic
lane splitting, residual schema package decomposition for semantic contract
validators, workspace identity/schedule/resource decomposition, broader
Direct-HIP source splitting, narrower currentness helpers for failure-path and
test-owned mutations, grouped descriptor enforcement, and export/reconstruction
planning. These changes must preserve public ABI, reviewed cache behavior,
existing benchmark CLI compatibility, and Windows `gfx1100` validation
boundaries.

## Not Yet Implemented

- Linux ROCm direct-HIP parity on a real supported Linux host.
- Instinct CDNA validation, profiling, power runs, and cluster reproducibility.
- Multi-GPU modulus-split experiments.
- Public optimized matrix-engine strict wrap64 backend.
- AMDGPU builtin hot kernels.
- Durable packed-layout/prepack-cache production paths.
- Optimized finite-field algorithms beyond current explicit-modulus CPU,
  direct-HIP correctness, and narrow reviewed accelerator-cache paths.
- Broader production performance gates beyond reviewed Windows `gfx1100`
  shape-scoped evidence.
- Linux/Instinct promotion gates remain unvalidated. Exact-wide 2048 and
  eligible exact-wide 4096 rows now have Windows `gfx1100`
  CPU/direct/accelerator release review and installed local cache entries;
  strict wrap64 2048/4096 has CPU/direct-HIP release review but no public
  optimized matrix-engine backend.

## Validation Boundary

Windows `gfx1100` proof does not validate Linux ROCm, Linux Radeon, Instinct
CDNA, profiling, power, or cluster production gates. Dependency discovery does
not validate correctness backends. Accelerator probes remain candidate evidence
until a dedicated preset builds real kernels and exact differentials pass.

Raw captures, historical smoke files, and review scratch output stay under
ignored `temp/` paths. Durable docs summarize only reviewed facts.
