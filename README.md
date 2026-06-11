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

## Performance (June 11, 2026)

243 captures, 0 failures, 5 backends, 281 tests. Windows gfx1100, HIP SDK 7.1,
release builds, fixed seeds, schema-valid, CPU reference checked.

### Quick reference

| What | Who | Speed |
|---|---|---|
| Square bounded >= 256 | Direct HIP | Production baseline |
| Small bounded (32-64) | rocWMMA | up to 76x faster |
| Skinny GEMV (N <= 8) | AMDGPU builtins / rocWMMA | up to 76x |
| Exact-wide | rocWMMA / CK | up to 7x |
| Finite-u8 | CK / rocWMMA | up to 1.6x |
| Wrap64 | Direct HIP | 230x vs CPU |
| 4096 shapes | hipBLASLt | up to 5.2x |

### Active optimizations

| Layer | What is live |
|---|---|
| Pack | Persistent (<=4096 cells), Coalesced 4-wide (>=256, contiguous), Standard. Non-temporal loads. |
| GEMM | DP4A (`v_dot4_i32_iu8 neg_lo:[1,1,0]` -- works around ROCm 7.1 bug). Persistent small (m*n <= 64). HIP graph replay. Plane parallelism via grid.z. |
| Export | VOPD DPP (prefix 1-8, status needed). Combined final-output (prefix 1-8, status elided). u192 CRT (prefix 9+). Precomputed Garner weights. Status elision all paths. |
| AMDGPU builtins | WMMA skinny dispatch (N=1->64t, N<=4->128t, N<=8->256t). |
| Wrap64 | Tiled u64acc (>=1024). |
| Infrastructure | Zero-skip detection. Adaptive prefix. Verification amortization. Scenario lint (0 errors). |
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

- **Not multi-GPU.** Current API assumes single-GPU operation.
- **No async API.** All public calls are synchronous host-explicit.
- **Windows HIP SDK 7.1** is the only RDNA3-supported runtime path.
  Linux ROCm is the full production and CDNA validation path (deferred).
- **Streaming overlap** is architecturally optimal as-is: the grouped-prefix
  GEMM kernel uses grid.z for plane-level parallelism across 96 CUs.
  Per-plane HIP stream launches are a regression, not an optimization.
- **64-bit multi-precision GEMM** (`v_dot2_i32_i16`) is CDNA3-only.
  RDNA3 only has `v_dot4_i32_i8` (DP4A).
## Hardware Scope

| Target | Status |
|---|---|
| RDNA3 (gfx1100 / RX 7900 XTX) | Primary bring-up. 281 tests, 243-capture sweep. |
| CDNA2 (gfx90a / MI200) | Planned, not yet validated. |
| CDNA3 (gfx942 / MI300) | MFMA/SMFMAC kernels compiled, no Linux sweep. |
| RDNA4 (gfx1200) | WMMA kernels compiled, gated pending hardware. |
| CDNA4 (gfx950) | Target metadata registered. |
| CPU (x86-64, OpenMP) | Reference backend, all semantics validated. |
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
