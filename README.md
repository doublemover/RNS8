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

Current local Windows `gfx1100` release-reviewed cache snapshot:

| Family | Case | Winner | Median | vs Direct HIP | vs CPU/ref | Disposition |
|---|---|---|---:|---:|---:|---|
| Bounded exact | i64 4096 | hipBLASLt | 35.3 ms | 3.65x | 601.8x | Installed cache key |
| Bounded exact | u64 4096 | hipBLASLt | 37.5 ms | 3.16x | 441.9x | Installed cache key |
| Finite u8 | field-251 4096 | hipBLASLt | 6.4 ms | 5.25x | 747.8x | Installed cache key |
| Finite u8 | ring-256 4096 | hipBLASLt | 6.9 ms | 4.73x | 680.9x | Installed cache key |
| Exact-wide | signed 4096 | hipBLASLt | 176.9 ms | 3.61x | 639.1x | Installed cache key |
| Exact-wide | unsigned 4096 | hipBLASLt | 162.4 ms | 3.78x | 649.5x | Installed cache key |
| Strict wrap64 | 2048 | Direct HIP | 58.3 ms | n/a | 230.1x | Correctness path |
| Strict wrap64 | 4096 | Direct HIP | 295.7 ms | n/a | 348.1x | Correctness path |

Explicit workload and implementation wins:

| Area | Case | Path | Median | Same-path gain | Control gain | Boundary |
|---|---|---|---:|---:|---:|---|
| Reusable operands | bounded-u64 2048 A+B | hipBLASLt | 9.2 ms/repeat | 3.99x vs non-reuse | 2.35x vs fastest | Workload contract only |
| Many-small grouped | exact-wide signed 64/128 group32 | Direct HIP grouped | 68.1-185.0 us/task | 7.6-16.8x vs hostbatch | 7.9-18.5x vs independent | Benchmark evidence |
| Many-small grouped | exact-wide unsigned 64/128 group32 | Direct HIP grouped | 75.5-155.9 us/task | 8.9-14.0x vs hostbatch | 11.7-16.2x vs independent | Benchmark evidence |
| Many-small grouped | bounded/finite grouped matrix | Direct HIP grouped | 33.5-100.8 us/task | 14.6-26.4x vs hostbatch | 2.3-26.6x vs best independent | Benchmark evidence |
| RNS chains | exact-wide signed 128 chain3 | Direct HIP final-output | 1.77 ms | 9.80x vs independent | n/a | Benchmark evidence |
| RNS chains | exact-wide unsigned 256 chain3 | Direct HIP final-output | 2.84 ms | 10.80x vs independent | n/a | Benchmark evidence |
| Shape-specialized | vector N=1 i64 | Vector ALU | n/a | 7.41x vs old path | 35.9x kernel | Active explicit route |
| Shape-specialized | one-shot i64 512 | Direct HIP colpair | n/a | 3.07x vs old path | 3.02x API event | Active explicit route |
| Planner/prepass | bounded-u64 adaptive scan | Direct HIP setup | 414.4 ms | 1.35x vs old scan | n/a | Not promoted |

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
