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

## Semantic Modes

| Mode | Output contract | Current backends |
|---|---|---|
| `RNS8_BOUNDED_I64` | Exact `int64_t` within explicit signed bound | CPU, direct HIP, vector ALU, opt-in accelerators |
| `RNS8_BOUNDED_U64` | Exact `uint64_t` within explicit unsigned bound | CPU, direct HIP, vector ALU, opt-in accelerators |
| `RNS8_EXACT_WIDE_SIGNED` | Signed fixed-width little-endian limbs | CPU, direct HIP, opt-in accelerators |
| `RNS8_EXACT_WIDE_UNSIGNED` | Unsigned fixed-width little-endian limbs | CPU, direct HIP, opt-in accelerators |
| `RNS8_WRAP_U64_MOD_2_64` | Strict low-64-bit wraparound | CPU and direct-HIP byte-limb paths |
| `RNS8_FINITE_RING_U8` | Canonical byte output modulo explicit `2..256` | CPU, direct HIP, opt-in accelerators |
| `RNS8_FINITE_FIELD_U8` | Canonical byte output modulo prime `2..251` | CPU, direct HIP, opt-in accelerators |

Public backend strings are `cpu-reference`, `hip-direct`,
`hip-vector-alu-int64`, `wrap64-byte-limb`, `hipblaslt`, `ck`, and `rocwmma`.

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

Current local Windows `gfx1100` release-reviewed snapshot:

| Contract | Shape | Current reviewed path | Median end-to-end | Comparison | Cache |
|---|---:|---|---:|---:|---|
| bounded i64 | 512 | Direct HIP `direct_hip_tiled_active_prefix_rns_gemm_v2` | 1851 us | no accelerator win | none |
| bounded i64 | 1024 | hipBLASLt `hipblaslt_int8_i32_scratch_reduce_specialized_251_255_256_v2` | 4174 us | 1.09x vs Direct HIP | installed |
| finite ring u8 mod 251 | 128 | rocWMMA `rocwmma_i8_i32_signed_finite_u8_mod251_hot_residue_v2` | 1136 us | 1.11x vs Direct HIP | installed |
| finite ring u8 mod 251 | 1024 | rocWMMA `rocwmma_i8_i32_signed_finite_u8_mod251_hot_residue_v2` | 1709 us | 2.74x vs Direct HIP | installed |
| finite ring u8 mod 255 | 1024 | CK `ck_wmma_cshuffle_finite_u8_mod255_centered_epilogue_v2` | 1938 us | 3.00x vs Direct HIP | installed |
| finite ring u8 mod 256 | 128 | rocWMMA `rocwmma_i8_i32_signed_finite_u8_mod256_hot_residue_v2` | 1132 us | 1.02x vs Direct HIP | installed |
| finite ring u8 mod 256 | 512 | rocWMMA `rocwmma_i8_i32_signed_finite_u8_mod256_hot_residue_v2` | 1365 us | 4.08x vs Direct HIP | installed |
| finite ring u8 mod 256 | 1024 | hipBLASLt `hipblaslt_int8_i32_scratch_reduce_specialized_251_255_256_v2` | 1792 us | 7.05x vs Direct HIP | installed |
| finite field u8 mod 251 | 1024 | CK `ck_wmma_cshuffle_finite_u8_mod251_centered_epilogue_v2` | 1860 us | 5.68x vs Direct HIP | installed |
| exact-wide signed | 512 | rocWMMA `rocwmma_i8_i32_signed_mod251_255_256_hot_residue_v2` | 7162 us | 1.02x vs Direct HIP | installed |
| exact-wide signed | 1024 | hipBLASLt `hipblaslt_int8_i32_scratch_reduce_specialized_251_255_256_v2` | 17092 us | 1.32x vs Direct HIP | installed |
| exact-wide unsigned | 1024 | CK `ck_wmma_cshuffle_i8_i32_mod251_255_256_centered_epilogue_v2` | 20481 us | 1.22x vs Direct HIP | installed |

The installed reviewed cache currently covers 11 exact plan keys: one
bounded-i64 key, seven finite-u8 keys, and three exact-wide keys. Some rows are
deliberately narrow local wins; Linux ROCm, Instinct, RDNA4, and profiler-backed
production claims remain separate validation work.

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
