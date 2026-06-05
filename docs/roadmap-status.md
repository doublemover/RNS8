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
| hipBLASLt | Opt-in correctness baseline | Reviewed Windows `gfx1100` cache wins exist for selected bounded-i64, finite-u8, and exact-wide shapes, including finite-u8 hot-modulus 2048 and exact-wide 2048 entries |
| CK | Opt-in correctness backend | Reviewed Windows `gfx1100` cache wins exist for selected bounded-i64, finite-u8, and exact-wide shapes, including bounded-i64 2048 and generic finite-field 2048 |
| rocWMMA | Opt-in correctness backend | Reviewed Windows `gfx1100` cache wins exist for selected bounded-u64, finite-u8, and exact-wide shapes, including bounded-u64 2048, finite-u8 2048 hot-modulus entries, generic finite-ring 2048 entries, and finite-field 512 entries |
| AMDGPU builtins | Not implemented | Fail-fast until real exact kernels exist |
| Wrap64 matrix-engine candidate | Internal rocWMMA harness only | Not public, not AUTO-selected, and not faster than direct HIP in current reviewed shapes |

Detailed benchmark policy, current wins, and reviewed release summaries live in
[performance-model.md](performance-model.md),
[performance-wins.md](performance-wins.md), and
[reviewed-local-evidence.md](reviewed-local-evidence.md).

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
- 4096 large-shape matrices and Linux/Instinct promotion gates remain
  unvalidated. Exact-wide 2048 now has Windows `gfx1100` CPU/direct/accelerator
  release review and installed local cache entries; strict wrap64 2048 has
  CPU/direct-HIP release review but no public optimized matrix-engine backend.

## Validation Boundary

Windows `gfx1100` proof does not validate Linux ROCm, Linux Radeon, Instinct
CDNA, profiling, power, or cluster production gates. Dependency discovery does
not validate correctness backends. Accelerator probes remain candidate evidence
until a dedicated preset builds real kernels and exact differentials pass.

Raw captures, historical smoke files, and review scratch output stay under
ignored `temp/` paths. Durable docs summarize only reviewed facts.
