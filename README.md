# RNS8

Exact integer matrix multiplication for AMD GPUs, with a C ABI, a thin C++
wrapper, and a portable CPU reference. RNS8 computes `C = A × B` using explicit
integer contracts: bounded signed/unsigned results, exact-wide results, small
finite rings/fields, or strict modulo-2⁶⁴ wraparound.

**Status: pre-1.0, hardware requalification required.** The October 2026 source
audit found correctness defects in recent fast paths and removed or repaired
them. CPU validation is useful evidence, but does not validate HIP compilation
or GPU execution. The old June `gfx1100` sweep summaries are preserved as
[historical records](docs/archive/README.md); they are not performance guarantees
for this revision. Start with the [audit](docs/audit-2026-10-06.md) and
[hardware qualification guide](docs/gpu-qualification.md) before evaluating speed.

## What the library computes

| Contract | Result and requirements |
| --- | --- |
| `RNS8_BOUNDED_I64` | Exact signed 64-bit result under a caller-supplied absolute output bound |
| `RNS8_BOUNDED_U64` | Exact unsigned 64-bit result under a caller-supplied output bound |
| `RNS8_EXACT_WIDE_SIGNED` / `_UNSIGNED` | Full-width native 64-bit inputs, range-proven RNS product, fixed-width limb export |
| `RNS8_FINITE_RING_U8` | Arithmetic modulo an explicitly supplied integer from 2 through 256 |
| `RNS8_FINITE_FIELD_U8` | Arithmetic modulo an explicitly supplied prime at most 251 |
| `RNS8_WRAP_U64_MOD_2_64` | Low 64 bits of the integer product; unsigned byte-limb storage, independent of CRT bounds |

These contracts are separate APIs. A C++ type does not select the semantics.
`EXACT_WIDE` is not unlimited-precision storage: the current implementation
supports at most 20 of the 28 default moduli, about 154.84 bits of CRT range.
Export accepts 1–32 little-endian 64-bit limbs, which does not increase the
information represented by the selected prefix. Signed export is two's complement.

A bounded plan checks that its CRT range covers the declared bound. It does not
prove the bound true for arbitrary input data. Incorrect caller bounds can alias
modulo the CRT product; range checking cannot detect every false bound. See the
[exactness contract](docs/correctness.md) for the precise obligations.

## How it works

1. Pack native integers into persistent modulus-major planes of centered signed
   8-bit residues.
2. Multiply each plane with signed `int8 × int8 → int32` arithmetic, reducing
   between safe accumulation blocks.
3. Reconstruct with CRT/Garner at the requested output boundary, or keep the
   residues resident for another operation.

The default ladder begins `256, 255, 253, 251, 247, 239, 233, 229, 227`.
It is pairwise coprime; its members need not all be prime. For CRT product `P`,
bounded signed plans require `P > 2 × bound`, and unsigned plans require
`P > bound`. Prefix 9 covers the full bounded 64-bit output ranges. Smaller
proven bounds can use fewer planes; per-tile bounds can select different prefixes.

Strict wrap64 uses its separate byte-limb representation and low-product
arithmetic. Small finite-ring/field operations use one explicit modulus, not a
CRT ladder. [Design and ownership](docs/design.md) explains resident storage,
source versions, workspaces, grouping, and export boundaries.

## Build without a GPU

Requirements: a C++17 compiler, CMake 3.22+, Ninja, Python 3.10+, Boost headers,
nlohmann-json, and Catch2 **3.x** for tests. OpenMP is optional. GMP and FLINT
are optional reference dependencies. The build does not download dependencies.

### Linux

Install native development packages for those dependencies, then:

```sh
cmake --preset linux-cpu-debug
cmake --build --preset linux-cpu-debug --parallel 4
ctest --preset linux-cpu-debug --output-on-failure
./build/linux-cpu-debug/rns8-verify
```

If your distribution provides only Catch2 2.x, install Catch2 3.x separately and
point `CMAKE_PREFIX_PATH` at that native installation. Do not use a Windows
vcpkg directory from Linux or WSL; the project intentionally rejects it.

For a library-only build without Catch2:

```sh
cmake --preset linux-cpu-release -DRNS8_BUILD_TESTS=OFF
cmake --build --preset linux-cpu-release --parallel 4
```

### Windows (PowerShell)

Install Visual Studio 2022's C++ workload, CMake, Ninja, Python, and vcpkg. Set
`VCPKG_ROOT` to your actual installation; the CPU presets use it explicitly.

```powershell
$env:VCPKG_ROOT = 'C:\vcpkg'
python tools\windows_dev.py cmake --preset cpu-debug
python tools\windows_dev.py cmake --build --preset cpu-debug --parallel 4
python tools\windows_dev.py ctest --preset cpu-debug --output-on-failure
.\build\cpu-debug\rns8-verify.exe
```

## Build for AMD hardware

RNS8 uses explicit `hipcc` integration rather than CMake's HIP language on
Windows. The default local Radeon target is RX 7900 XTX (`gfx1100`); the MI300X
qualification target is `gfx942`. Support depends on the installed AMD software
and target-specific backend. A preset is not evidence of hardware qualification.

Windows presets currently name `C:/Program Files/AMD/ROCm/7.1` and `C:/vcpkg`.
Override both paths if needed:

```powershell
python tools\check_dependencies.py
python tools\windows_dev.py cmake --preset windows-msvc-hip-debug `
  -DRNS8_HIP_ROOT='C:/Program Files/AMD/ROCm/7.1' `
  -DCMAKE_TOOLCHAIN_FILE="$env:VCPKG_ROOT/scripts/buildsystems/vcpkg.cmake"
python tools\windows_dev.py cmake --build --preset windows-debug --parallel 4
python tools\windows_dev.py ctest --preset windows-debug --output-on-failure
.\build\windows-msvc-hip-debug\rns8-verify.exe --hip-smoke
```

For a native Linux ROCm installation at `/opt/rocm`:

```sh
cmake --preset linux-cdna-debug -DRNS8_AMDGPU_TARGETS=gfx942
cmake --build --preset linux-cdna-debug --parallel 4
ctest --preset linux-cdna-debug --output-on-failure
./build/linux-cdna-debug/rns8-verify --hip-smoke
```

See [Windows setup](docs/platform-windows.md), [Linux setup](docs/platform-linux.md),
and [backend requirements](docs/backend-notes.md). Optional accelerator builds
are independent of the CPU/direct-HIP correctness paths.

| Backend | Implementation and scope |
| --- | --- |
| `cpu-reference` | Portable RNS arithmetic and Boost.Multiprecision oracle |
| `wrap64-byte-limb` | CPU strict-wrap reference |
| `hip-direct` | Direct HIP RNS, finite, and wrap kernels; checked wide CRT export |
| `hip-vector-alu-int64` | Native multiprecision integer comparator for bounded contracts |
| `hipblaslt` | Optional INT8 GEMM with INT32 scratch and residue reduction |
| `ck` | Optional Composable Kernel matrix-engine path and fused reduction |
| `rocwmma` | Optional rocWMMA matrix-engine path; internal wrap candidate is separate |
| `amdgpu-builtins` | Target-specific WMMA/MFMA and explicit sparse-A candidates |

`AUTO` starts from the available correctness context and only changes the plan
backend for an eligible, exact-match reviewed cache entry. It does not benchmark
at runtime or automatically select the fastest backend for nearby shapes.
This revision changes the implementation key, so old entries do not silently
qualify the repaired code. [Performance policy](docs/performance.md) describes
review and promotion requirements.

## Use the API

The simplest complete programs are the checked-in examples:

- [Bounded signed GEMM](examples/bounded_i64_oneshot.cpp)
- [Finite-ring GEMM](examples/finite_ring_u8_oneshot.cpp)
- [Strict wrap64 GEMM](examples/wrap64_u64_oneshot.cpp)
- [Exact-wide limb export](examples/exact_wide_limb_export.cpp)

Initialize descriptor `struct_size` and `abi_version`, choose a semantic contract
and backend, check every returned status, and destroy handles after use. For
repeated products, use `context → plan → matrix/workspace → pack → GEMM → export`.
Keep source versions accurate: reusing a version asserts that the input has not
changed. Public operations are synchronous and row-major; column-major execution
is unsupported. The [public header](include/rns8/rns8.h) documents individual calls.

Install and consume with CMake:

```sh
cmake --install build/linux-cpu-debug --prefix "$PWD/temp/install-rns8"
cmake -S examples/downstream-cmake -B temp/downstream-rns8 -G Ninja \
  -DCMAKE_PREFIX_PATH="$PWD/temp/install-rns8"
cmake --build temp/downstream-rns8
```

In PowerShell, use an absolute prefix such as
`"$PWD/temp/install-rns8"` and the `build/cpu-debug` build directory.
The exported targets are `rns8::rns8` (when the shared library is built) and
`rns8::rns8_static`.

## Test and qualify

```sh
python tools/golden_regression_suite.py
python tools/gpu_qualification.py --target mi300x
python tools/gpu_qualification.py --target 7900xtx
```

The last two commands only write manifests. They enumerate source kernels,
exact build/test/capture commands, and remaining hardware gates. Nothing uses a
GPU unless `--execute` is supplied on the matching host. See the
[qualification guide](docs/gpu-qualification.md) for execution and pass criteria.

A small benchmark, after hardware correctness succeeds:

```sh
./build/linux-cdna-debug/rns8-bench --backend hip-direct --semantics bounded-i64 \
  --m 32 --n 32 --k 64 --warmups 1 --repeats 3 --seed 20261006 --progress \
  > temp/bounded-smoke.json
python tools/benchmark_schema.py temp/bounded-smoke.json
```

This is smoke evidence, not a release performance claim. Preserve raw captures,
compiler/runtime versions, target identity, exact reference comparisons, phase
measurements, and ISA/counter evidence before promoting any result.

## Limits and next steps

- No general multi-GPU GEMM or asynchronous public execution contract
- No implemented Ozaki/FP8, INT4/IU4, Strassen, or Freivalds backend; reserved
  entrypoints return `RNS8_UNSUPPORTED_BACKEND`
- Optional backends have target, shape, modulus, and accumulation limits
- Sparse-A is explicit and target-specific; dense calls do not infer sparsity
- HIP compilation, MI300X execution, RX 7900 XTX regression coverage, and fresh
  performance campaigns remain required for this revision

See the [documentation map](docs/README.md), [current roadmap](docs/roadmap.md),
[research specification](docs/RNS8_RESEARCH_SPEC.md), and
[contributing guide](CONTRIBUTING.md). License: [MIT](LICENSE).
