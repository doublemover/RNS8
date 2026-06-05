# RNS8

RNS8 explores exact integer matrix multiplication on AMD GPU matrix engines.

Unlike traditional BLAS libraries that operate on floating-point values,
RNS8 computes exact integer results using residue number systems (RNS),
Chinese Remainder Theorem reconstruction (CRT), and GPU matrix engines.

RNS8 treats GPU matrix engines as exact integer compute devices rather than
floating-point accelerators.

Status: Pre-1.0 research project. Active development.
Windows HIP SDK / gfx1100 is the primary validated platform.

## Use of AI

RNS8 is an open-source HPC codebase built entirely through AI-directed
development workflows. The architecture, constraints, performance goals,
and correctness requirements are human defined. Codex does the typing.

## What Works Now

- CPU reference backend with Boost.Multiprecision CRT/Garner reconstruction.
- Public C ABI, limited C++ RAII wrapper, CMake install/export targets, and
  downstream examples.
- Explicit semantic modes for bounded i64/u64, exact-wide signed/unsigned,
  strict `mod 2^64`, finite rings, and prime fields.
- Persistent plan, matrix, workspace, and prepack-cache handles with explicit
  ownership and compatibility checks.
- Windows HIP direct backend on Radeon RX 7900 XTX / `gfx1100`, including
  device-resident pack, GEMM, and export paths.
- Native vector-ALU runtime backend for explicit bounded i64/u64 contracts.
- Opt-in hipBLASLt, CK, and rocWMMA accelerator backends with reviewed local
  `gfx1100` winners for selected bounded, finite-u8, and exact-wide shapes.
- AUTO backend selection from reviewed autotune-cache entries only; unsupported
  or unreviewed backends fall back to correctness paths.
- Benchmark schema v4, release-sweep review tooling, result comparison, GPU
  event reports, ISA reports, and cache-install validation.
- Benchmark-only workload evidence for reusable operands, many-small grouped
  dispatch, RNS-chain final-output reuse, vector N=1 kernels, and wrap64
  byte-limb tuning.

## Quick Start

CPU-only build:

```powershell
cmake --preset cpu-debug
cmake --build --preset cpu-debug
ctest --preset cpu-debug --output-on-failure
```

Windows HIP bring-up:

```powershell
python tools\check_dependencies.py
python tools\windows_dev.py cmake --preset windows-msvc-hip-debug
python tools\windows_dev.py cmake --build --preset windows-debug
python tools\windows_dev.py ctest --preset windows-debug --output-on-failure
build\windows-msvc-hip-debug\rns8-verify.exe --hip-smoke
```

Package smoke:

```powershell
cmake --install build/cpu-debug --prefix temp/install-rns8/Debug
cmake -S examples/downstream-cmake -B temp/downstream-rns8/Debug -G Ninja -DCMAKE_PREFIX_PATH=%CD%/temp/install-rns8/Debug
cmake --build temp/downstream-rns8/Debug
```

Exported package targets are `rns8::rns8` and `rns8::rns8_static`.

## Hardware Scope

Minimum useful evaluation is CPU-only with the `cpu-debug` preset. The local GPU
proof path is Windows HIP SDK on Radeon RX 7900 XTX / `gfx1100`. Linux ROCm,
Instinct CDNA, RDNA4, multi-GPU, profiling, and power evidence remain separate
validation targets and are not implied by Windows proof.

## Threading And Lifetimes

RNS8 handles are explicit ownership objects. Contexts own backend selection,
plans own validated semantic and schedule metadata, matrices own current host
and device storage state, workspaces belong to compatible plans, and prepack
caches belong to the source matrix/plan/device contract that created them.

The library does not promise internal synchronization for sharing mutable
handles across threads. Use separate handles per thread or externally
synchronize access to contexts, matrices, workspaces, and caches.

## Exactness And Performance

Correctness is established through CPU reference comparisons before GPU paths
count. Unsupported backends report unsupported status instead of silently
downgrading exactness.

Performance claims require reviewed same-contract captures with fixed seeds,
recorded toolchain and GPU metadata, CPU/reference and GPU baselines, and
release-mode repeat counts. Raw benchmark captures are evidence, not promotion;
see [docs/performance-model.md](docs/performance-model.md),
[docs/performance-wins.md](docs/performance-wins.md), and
[docs/reviewed-local-evidence.md](docs/reviewed-local-evidence.md).

Exactness rules are explicit:

- Bounded `i64`/`u64` uses range-proven CRT/RNS, not type inference.
- Exact-wide signed/unsigned outputs use fixed-width limb export contracts.
- Finite-u8 rows are keyed by explicit ring or prime-field modulus.
- Strict `mod 2^64` uses byte limbs, not odd-modulus CRT.

Current local Windows `gfx1100` release-reviewed snapshot:

| Family | Best local evidence | Boundary |
|---|---:|---|
| Bounded exact | i64 4096 hipBLASLt: 35.3 ms, 3.65x vs Direct HIP.<br/>u64 4096 hipBLASLt: 37.5 ms, 3.16x. | Cache entries installed for selected 1024-4096 keys; i64 512 stays Direct HIP. |
| Finite u8 | 4096 field-251 hipBLASLt: 6.4 ms, 5.25x.<br/>1024 ring-256 hipBLASLt: 1.8 ms, 7.05x. | Cache entries installed for selected 128-4096 modulus/shape keys, including generic 127/253 coverage. |
| Exact-wide | 4096 signed hipBLASLt: 176.9 ms, 3.61x.<br/>4096 unsigned hipBLASLt: 162.4 ms, 3.78x. | Cache entries installed for selected semantic/shape/limb keys. |
| Strict wrap64 | Direct HIP 2048: 58.3 ms, 230.1x vs CPU byte-limb.<br/>Direct HIP 4096: 295.7 ms, 348.1x. | Correctness path only; no AUTO cache entry or matrix-engine promotion. |
| Reusable operands | 2048 bounded-u64 hipBLASLt A+B: 9.2 ms per repeat, 2.35x vs same-run fastest non-reuse. | Explicit workload evidence; not AUTO-promoted. |
| Many-small grouped | Exact-wide signed group32: 66.5 us/task, 58.4x.<br/>Unsigned group32: 79.1 us/task, 18.7x. | Benchmark-owned grouped-dispatch evidence; no public grouped API yet. |
| RNS chains | Exact-wide signed 128 chain3: 1.77 ms, 9.80x.<br/>Unsigned 256 chain3: 2.84 ms, 10.80x. | Benchmark-only lazy-output evidence; no AUTO cache entry. |
| Shape-specialized paths | Vector N=1 i64: 7.41x vs old vector path.<br/>Direct-HIP one-shot i64 512: 3.07x vs prior Direct-HIP kernel. | Active explicit routes; not cross-backend cache claims. |
| Planner/prepass | Adaptive bounded-u64 tile scan: 557.6 ms to 414.4 ms, 1.35x. | Setup-path win; full bound-discovery routing stayed out of promotion. |

The installed reviewed cache currently contains 39 validated exact-key entries.
The table above is intentionally compact; long kernel identities, per-row
baselines, checksums, event status, caveats, and reproduction commands live in
[docs/performance-wins.md](docs/performance-wins.md) and
[docs/reviewed-local-evidence.md](docs/reviewed-local-evidence.md). Linux ROCm,
Instinct, RDNA4, and profiler-backed production claims remain separate
validation work.

## Known Limitations

- Pre-1.0 public names and structs may change through deliberate hard cuts.
- The C++ wrapper is limited RAII support.
- AUTO selection only promotes reviewed cache entries for supported contracts.
- Small-shape finite-u8 accelerator wins must beat CPU as well as Direct HIP;
  backend-relative wins that lose to CPU are not promoted.
- hipBLASLt, CK, and rocWMMA are opt-in accelerators, not required for
  correctness.
- Strict wrap64 matrix-engine acceleration is an internal candidate, not a
  public optimized backend.
- Linux ROCm and Instinct readiness require live validation on those platforms.

## What This Is Not

RNS8 is not a general BLAS replacement or a claim that every AMD GPU target
is production-ready. It is a hardware-realistic exact integer GEMM project with
explicit semantics and evidence gates.

## Documentation

- [docs/README.md](docs/README.md): documentation map.
- [docs/RNS8_RESEARCH_SPEC.md](docs/RNS8_RESEARCH_SPEC.md): architecture, roadmap, and semantic contracts.
- [docs/public-roadmap.md](docs/public-roadmap.md): compact public roadmap.
- [docs/release-checklist.md](docs/release-checklist.md): release gate.
- [docs/glossary.md](docs/glossary.md): terminology.
- [docs/prior-art.md](docs/prior-art.md): related systems and scope boundary.
- [docs/platform-windows.md](docs/platform-windows.md): Windows HIP setup.
- [third_party/README.md](third_party/README.md): third-party and submodule policy.
