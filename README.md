# RNS8

RNS8 is an exact integer GEMM library for AMD GPUs. It stores matrices in a
residue number system, evaluates small-modulus `int8 x int8 -> int32` GEMMs,
and reconstructs integer outputs only under an explicit semantic contract.

The project is pre-1.0 and intentionally direct: signedness, wraparound,
finite-ring behavior, and exact-width export are never inferred from a C++ type
alone. The public C ABI is the primary supported API; the C++ wrapper is a
small RAII handle layer, not a full C++ API surface.

## What Works Now

- CPU reference backend with Boost.Multiprecision CRT/Garner reconstruction.
- Public C ABI, limited C++ RAII wrapper, install/export targets, and examples.
- Windows HIP direct backend on Radeon RX 7900 XTX / `gfx1100`.
- Persistent RNS matrices with device-resident direct-HIP pack, GEMM, and
  export paths.
- Bounded signed/unsigned 64-bit GEMM with fixed and per-tile modulus counts.
- Native vector-ALU backend for explicit bounded i64/u64 contracts.
- Exact-wide signed/unsigned RNS output with fixed-width limb export.
- Strict `mod 2^64` CPU and direct-HIP byte-limb paths.
- Explicit finite-ring and finite-field `uint8_t` GEMM for moduli up to 256.
- Opt-in Windows `gfx1100` hipBLASLt, CK, and rocWMMA correctness backends.
- Benchmark schema v4, result comparison, GPU event reporting, ISA reporting,
  and reviewed autotune-cache validation.

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

## Status Codes

| Code | Meaning |
|---|---|
| `RNS8_SUCCESS` | Operation completed. |
| `RNS8_INVALID_ARGUMENT` | Public ABI contract, descriptor, handle, layout, or semantic input is invalid. |
| `RNS8_UNSUPPORTED_OS` | Requested path is not supported on this OS. |
| `RNS8_UNSUPPORTED_ARCH` | Requested path is not supported on this architecture or target. |
| `RNS8_UNSUPPORTED_BACKEND` | Backend is known but unavailable, disabled, or unsupported for the contract. |
| `RNS8_RANGE_ERROR` | Exact export cannot fit the requested bounded output. |
| `RNS8_ACCUMULATION_OVERFLOW_RISK` | Contract risks overflowing the backend accumulator. |
| `RNS8_WORKSPACE_TOO_SMALL` | Supplied workspace is smaller than the plan requires. |
| `RNS8_BACKEND_FAILURE` | Backend runtime or device operation failed. |
| `RNS8_VERIFICATION_FAILED` | Explicit verification failed. |
| `RNS8_INTERNAL_ERROR` | Internal invariant failure. |

## Environment Variables

| Variable | Used by | Purpose |
|---|---|---|
| `RNS8_AUTOTUNE_CACHE_PATH` | Runtime and benchmark tools | Overrides the reviewed autotune-cache location. Missing or rejected hits fall back to correctness paths. |
| `VCPKG_ROOT` | CMake presets and CI | Locates the vcpkg toolchain file for CPU presets. |
| `HIP_PATH` / `ROCM_PATH` | Dependency checks and HIP discovery | Helps locate HIP SDK or ROCm when preset roots are not enough. |
| `LOCALAPPDATA`, `USERPROFILE`, `XDG_CACHE_HOME`, `HOME` | Autotune tooling | Default cache-root discovery when `RNS8_AUTOTUNE_CACHE_PATH` is unset. |

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
release-mode repeat counts. Raw benchmark captures are evidence, not promotion.
See [docs/performance-model.md](docs/performance-model.md) and
[docs/performance-wins.md](docs/performance-wins.md).

## Known Limitations

- Pre-1.0 public names and structs may change through deliberate hard cuts.
- The C++ wrapper is limited RAII support.
- AUTO selection only promotes reviewed cache entries for supported contracts.
- hipBLASLt, CK, and rocWMMA are opt-in accelerators, not required for
  correctness.
- Strict wrap64 matrix-engine acceleration is an internal candidate, not a
  public optimized backend.
- Linux ROCm and Instinct readiness require live validation on those platforms.

## What This Is Not

RNS8 is not a general BLAS replacement, a symbolic algebra system, an FHE
library, a CPU arbitrary-precision package, or a claim that every AMD GPU target
is production-ready. It is a hardware-realistic exact integer GEMM project with
explicit semantics and evidence gates.

## Documentation

- [docs/README.md](docs/README.md): documentation map.
- [docs/RNS8_RESEARCH_SPEC.md](docs/RNS8_RESEARCH_SPEC.md): architecture,
  roadmap, and semantic contracts.
- [docs/public-roadmap.md](docs/public-roadmap.md): compact public roadmap.
- [docs/release-checklist.md](docs/release-checklist.md): release gate.
- [docs/glossary.md](docs/glossary.md): terminology.
- [docs/prior-art.md](docs/prior-art.md): related systems and scope boundary.
- [docs/platform-windows.md](docs/platform-windows.md): Windows HIP setup.
- [third_party/README.md](third_party/README.md): third-party and submodule
  policy.

Temporary captures, probes, installers, and scratch binaries belong under
ignored `temp/`, `build/`, or `out/` paths.
