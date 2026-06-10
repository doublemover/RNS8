# RNS8

RNS8 is an exact integer matrix multiplication library for AMD GPUs. It
computes `C = A * B` where A, B, and C are integer matrices, and the result
is mathematically exact -- no floating-point rounding, no approximation.

## How It Works

Traditional BLAS libraries (cuBLAS, rocBLAS, hipBLAS) operate on
floating-point values. You cannot ask them for an exact 64-bit integer
matrix product. RNS8 solves this by decomposing the problem:

1. **Residue Number System (RNS).** Each input matrix is converted into
   multiple "residue planes" by taking each element modulo a set of
   carefully chosen integers (the modulus ladder: 256, 255, 253, ...).
   A 64-bit integer becomes a set of 9 small 8-bit residues.

2. **INT8 Matrix Engine GEMM.** Each residue plane is an INT8 x INT8
   matrix multiply. This is exactly what AMD GPU matrix engines (WMMA,
   MFMA) are designed to compute. RNS8 runs one small GEMM per modulus
   plane, accumulating results in INT32.

3. **Chinese Remainder Theorem (CRT) reconstruction.** The per-modulus
   results are combined using CRT to recover the exact integer output.
   Because the modulus ladder is chosen so that the product of all moduli
   exceeds the maximum possible output value, the reconstruction is exact.

The key insight: GPU matrix engines are fundamentally integer compute
devices packaged as floating-point accelerators. RNS8 unwraps that
packaging and uses them for what they actually are.

## What RNS8 Does

RNS8 provides a C ABI (with a thin C++ RAII wrapper) that accepts integer
matrices, packs them into persistent RNS storage on the GPU, dispatches
per-modulus INT8 GEMM across one or more backends, and exports exact
integer results.

**Semantic contracts** -- you declare what you want, RNS8 proves it can
deliver it:

| Contract | What it computes | How |
|---|---|---|
| `BOUNDED_I64` / `BOUNDED_U64` | Exact signed/unsigned 64-bit GEMM with a known output bound | 9 CRT moduli, range-checked export |
| `EXACT_WIDE_SIGNED` / `_UNSIGNED` | Exact GEMM with arbitrary output width | Up to 20+ CRT moduli, multi-limb export |
| `WRAP_U64_MOD_2_64` | Strict wraparound multiplication mod 2^64 | Byte-limb Comba accumulation, not CRT |
| `FINITE_RING_U8` / `FINITE_FIELD_U8` | GEMM modulo an explicit small integer | Single-modulus centered residue GEMM |

**Backends** -- multiple GPU execution paths, selected explicitly or via
reviewed autotune cache:

| Backend | What it is | Status |
|---|---|---|
| `hip-direct` | Hand-written HIP INT8 GEMM kernels with fused CRT export | Production baseline, wins most shapes |
| `hipblaslt` | AMD hipBLASLt library INT8 GEMM | Wins 4096 shapes (2.5-5.2x vs Direct HIP) |
| `ck` | AMD Composable Kernel library | Competitive on finite-u8 512 |
| `rocwmma` | AMD rocWMMA matrix-core library | Wins bounded u64 512/1024 (1.17-1.49x) |
| `amdgpu-builtins` | Hand-written WMMA/MFMA kernels | Wins skinny GEMV (N=1,4,8) and exact-wide 512 |
| `hip-vector-alu-int64` | Native 64-bit integer HIP kernels | Reference comparator for bounded i64/u64 |
| `cpu-reference` | CPU scalar reference with Boost.Multiprecision | Correctness anchor, wins tiny shapes (<128) |

## Performance (June 10, 2026)

All numbers from full release sweeps on Radeon RX 7900 XTX / gfx1100,
Windows HIP SDK 7.1. 3 warmups, 9 measured repeats, fixed seed, exact
CPU reference differentials. See [docs/performance-wins.md](docs/performance-wins.md)
for complete tables.

### Direct HIP baseline (our code, no external libraries)

| Semantics | Shape | End-to-end | vs CPU |
|---|---|---|---:|
| Bounded i64 | 1024x1024x1024 | 6,301 us | 34.9x |
| Bounded i64 | 512x512x512 | 2,456 us | 49.5x |
| Bounded u64 | 1024x1024x1024 | 6,856 us | 117.4x |
| Bounded u64 | 512x512x512 | 3,275 us | 39.6x |
| Exact-wide signed | 512x512x512 | 6,390 us | 37.5x |
| Finite field u8 | 512x512x512 | 1,736 us | 16.1x |
| Strict wrap64 | 2048x2048x2048 | 41,538 us | n/a |

### Backends beating Direct HIP

| Shape | Winner | Speedup vs Direct HIP |
|---|---|---|
| Bounded u64 256x256x256 | AMDGPU builtin | 1.52x |
| Exact-wide signed 128x128x128 | AMDGPU builtin | 1.41x |
| Bounded u64 512x512x512 | rocWMMA | 1.17x |
| Bounded u64 1024x1024x1024 | rocWMMA | 1.17x |
| Finite field u8 512x512x512 | rocWMMA | 1.49x |

### hipBLASLt at 4096x4096

| Semantics | hipBLASLt | Direct HIP | Speedup |
|---|---|---|---:|
| Bounded i64 | 46,825 us | 129,734 us | 2.77x |
| Bounded u64 | 51,239 us | 128,598 us | 2.51x |
| Exact-wide signed | 125,862 us | 577,811 us | 4.59x |
| Exact-wide unsigned | 120,893 us | 632,243 us | 5.23x |
| Finite field u8 | 9,369 us | 35,225 us | 3.76x |

### Optimization campaign gains (June 2026)

| Shape | Before | After | Gain |
|---|---|---|---:|
| bounded u64 256 | 3,235 us | 2,008 us | +37.9% |
| bounded i64 512 | 3,781 us | 2,456 us | +35.1% |
| bounded i64 512x4 | 2,381 us | 1,727 us | +27.4% |
| bounded i64 512x8 | 2,400 us | 1,769 us | +26.3% |
| finite field u8 512 | 2,269 us | 1,736 us | +23.5% |

## Quick Start

### Prerequisites

- Windows 11 with AMD Radeon RX 7900 XTX (gfx1100)
- [AMD HIP SDK 7.1](https://www.amd.com/en/developer/rocm.html)
- Visual Studio 2022 with C++ workload
- CMake 3.22+, Ninja, Python 3.11+
- [vcpkg](https://github.com/microsoft/vcpkg) at `C:\vcpkg`

### CPU-only build (no GPU required)

```powershell
cmake --preset cpu-debug
cmake --build --preset cpu-debug
ctest --preset cpu-debug --output-on-failure
```

### HIP debug build (gfx1100)

```powershell
python tools\check_dependencies.py
python tools\windows_dev.py cmake --preset windows-msvc-hip-debug
python tools\windows_dev.py cmake --build --preset windows-debug
python tools\windows_dev.py ctest --preset windows-debug --output-on-failure
```

### Accelerator backends (CK, rocWMMA, hipBLASLt, AMDGPU builtins)

```powershell
# Build all four accelerator backends (debug)
python tools\build_accelerators.py --debug

# Build specific backend
python tools\build_accelerators.py --release --backend ck

# List available presets
python tools\build_accelerators.py --list
```

### Benchmark sweep

```powershell
# Full release-candidate sweep with all backends
python tools\build_accelerators.py --release
python tools\benchmark_sweep.py --scenario release-candidates `
    --bench build\windows-msvc-hip-release\rns8-bench.exe `
    --bench-for ck=build\windows-msvc-ck-release\rns8-bench.exe `
    --bench-for rocwmma=build\windows-msvc-rocwmma-release\rns8-bench.exe `
    --bench-for hipblaslt=build\windows-msvc-hipblaslt-release\rns8-bench.exe `
    --bench-for amdgpu-builtins=build\windows-msvc-amdgpu-builtins-release\rns8-bench.exe `
    --out-root temp\rdna3-sweep --progress

# Generate performance dashboard
python tools\generate_performance_dashboard.py --capture-root temp\rdna3-sweep
```

### Package smoke

```powershell
cmake --install build/cpu-debug --prefix temp/install-rns8/Debug
cmake -S examples/downstream-cmake -B temp/downstream-rns8/Debug -G Ninja -DCMAKE_PREFIX_PATH=%CD%/temp/install-rns8/Debug
cmake --build temp/downstream-rns8/Debug
```

Exported package targets: `rns8::rns8` and `rns8::rns8_static`.

## Architecture

### Ownership model

RNS8 uses explicit handle types with defined lifetimes:

- `rns8_context` -- owns backend selection and device binding
- `rns8_plan` -- owns validated semantic contract, modulus schedule, and tile layout
- `rns8_matrix` -- owns host and device storage for residue planes
- `rns8_workspace` -- owns transient device buffers for a specific plan
- `rns8_prepack_cache` -- owns reusable accelerator-specific packed operand storage

Matrices track their currentness: a matrix knows whether its device residues,
host residues, native values, or byte limbs are up to date. Pack operations
stamp a source version. The library elides redundant pack uploads when the
source version matches and the device state is current.

### Data flow

```
int64/uint64 source matrices
  -> validated bounds metadata (plan)
  -> centered INT8 RNS packing (one kernel launch per pack call)
  -> persistent modulus-major residue planes on device
  -> per-modulus INT8 x INT8 -> INT32 GEMM (one kernel launch per modulus)
  -> fused INT32-to-centered-residue reduction
  -> CRT reconstruction (Garner algorithm)
  -> range-checked i64/u64 host output
```

For strict wrap64 (mod 2^64), the path uses byte-limb Comba accumulation
instead of CRT:

```
uint64 source matrices
  -> base-256 byte limbs
  -> 36 low-64-relevant byte-product pairs across 8 Comba diagonals
  -> Comba diagonal accumulation with signed-INT8 byte correction
  -> delayed carry propagation to low 64-bit output
```

### Modulus ladder

The default CRT ladder has 28 pairwise-coprime values, all <= 256:

```
256, 255, 253, 251, 247, 239, 233, 229,
227, 223, 217, 211, 199, 197, 193, 191,
181, 179, 173, 167, 163, 157, 151, 149,
139, 137, 131, 127
```

Prefix selection uses the strict condition `product(moduli[0:s]) > range`.
For bounded i64, the range is 2 * bound. For bounded u64, the range is bound.
Prefix 9 is the first prefix covering the full signed and unsigned 64-bit
ranges. The full 28-modulus ladder covers exact-wide outputs up to ~220 bits.

## Correctness

Every GPU path has CPU reference coverage. The CPU reference uses
Boost.Multiprecision for arbitrary-precision integer arithmetic and
deterministic Garner CRT reconstruction. GPU outputs are compared
bit-exact against CPU reference before any performance claim is made.

Unsupported backends report `RNS8_UNSUPPORTED_BACKEND`. They never silently
downgrade exactness, fall through to a weaker semantic, or substitute an
approximate result.

Exactness rules are explicit and enforced by the API:

- `BOUNDED_I64` / `BOUNDED_U64`: uses range-proven CRT with the minimum
  prefix that covers the declared bound.
- `EXACT_WIDE_SIGNED` / `_UNSIGNED`: fixed-width little-endian limb export;
  rejects bounded metadata and wrap shortcuts.
- `WRAP_U64_MOD_2_64`: byte-limb Comba path; rejects CRT metadata and odd-modulus
  routing unless a valid exact bound is supplied by the caller.
- `FINITE_RING_U8` / `FINITE_FIELD_U8`: explicit modulus contract; no CRT
  ladder involvement.

## Known Limitations

- Pre-1.0. Public names, struct layouts, and ABI versions may change through
  deliberate hard cutovers. No compatibility shims are preserved.
- The C++ wrapper provides RAII lifetime management only. Full C++ API surface
  is not yet implemented.
- AUTO backend selection only promotes reviewed autotune cache entries for
  supported contracts. Unreviewed or unsupported contracts fall back to the
  configured correctness backend (typically Direct HIP).
- Accelerator backends (CK, rocWMMA, hipBLASLt, AMDGPU builtins) are opt-in.
  They must be explicitly compiled via their respective CMake presets and are
  not required for correctness.
- rocWMMA captures lack HIP event timings in the release preset, blocking
  the repeated-B prepack cache benchmark.
- INT4/IU4, FP8/Ozaki, Strassen, Freivalds verification, and adversarial
  input detection are research-only APIs with `RNS8_UNSUPPORTED_BACKEND` stubs.
- Linux ROCm, Instinct CDNA, RDNA4, multi-GPU, profiling, and power evidence
  are separate validation targets. Windows gfx1100 evidence does not imply
  readiness on those platforms.
- Persistent small GEMM dispatch is enabled for m*n <= 64 only. Coalesced
  and persistent pack dispatch are compiled but not yet wired (pending
  differential test debugging).

## Hardware Scope

| Tier | Target | Status |
|---|---|---|
| W0 | Radeon RX 7900 XTX / gfx1100 | Primary validated platform |
| W1 | RDNA4 gfx1200/gfx1201 | Compile-only readiness |
| I0 | Instinct CDNA3 gfx942 | Compile-only readiness |
| I1 | Instinct CDNA4 gfx950 | Compile-only readiness |
| I2 | Instinct CDNA2 gfx90a | Compile-only readiness |

Minimum useful evaluation is CPU-only. GPU proof requires HIP SDK on a
supported AMD GPU. Linux ROCm parity, Instinct validation, multi-GPU, and
profiler-backed production claims require live hardware on those platforms.

## What This Is Not

RNS8 is not a general BLAS replacement. It does not intercept BLAS calls
or provide a drop-in BLAS API. It is not a claim that every AMD GPU target
is production-ready. It is a hardware-realistic exact integer GEMM project
with explicit semantics, correctness gates, and evidence standards.

## Documentation

- [docs/README.md](docs/README.md) -- documentation map and index
- [docs/RNS8_RESEARCH_SPEC.md](docs/RNS8_RESEARCH_SPEC.md) -- architecture, roadmap, semantic contracts, ship rules
- [docs/design.md](docs/design.md) -- current implementation design notes
- [docs/glossary.md](docs/glossary.md) -- terminology
- [docs/performance-wins.md](docs/performance-wins.md) -- every measured speedup vs Direct HIP
- [docs/reviewed-local-evidence.md](docs/reviewed-local-evidence.md) -- sweep evidence registry
- [docs/performance-model.md](docs/performance-model.md) -- performance modeling and methodology
- [docs/performance-gain-work-queue.md](docs/performance-gain-work-queue.md) -- active optimization ranks
- [docs/performance-gain-completed-work.md](docs/performance-gain-completed-work.md) -- closed rank archive
- [docs/platform-windows.md](docs/platform-windows.md) -- Windows HIP SDK setup guide
- [docs/platform-linux.md](docs/platform-linux.md) -- Linux ROCm setup guide
- [docs/public-roadmap.md](docs/public-roadmap.md) -- compact public roadmap
- [docs/prior-art.md](docs/prior-art.md) -- related systems and scope boundary
- [docs/correctness.md](docs/correctness.md) -- correctness standards and methodology
- [docs/dashboard.html](docs/dashboard.html) -- interactive performance dashboard
- [third_party/README.md](third_party/README.md) -- third-party and submodule policy
