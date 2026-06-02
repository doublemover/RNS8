# RNS8

RNS8 is a greenfield exact-integer GEMM project for AMD GPUs. The core idea is
to store matrices persistently in a residue number system (RNS), run many
`int8 x int8 -> int32` matrix multiplies over small pairwise-coprime moduli,
reduce each result back to residues, then reconstruct bounded `int64_t` or
`uint64_t` outputs with CRT when requested.

The first development target is Windows on Radeon through the AMD HIP SDK. The
full production target remains Linux ROCm for Radeon and Instinct systems, but
Windows `gfx1100` evidence does not validate Linux ROCm or Instinct CDNA. See
[docs/RNS8_RESEARCH_SPEC.md](docs/RNS8_RESEARCH_SPEC.md) for the full research
and implementation plan. See [docs/roadmap-status.md](docs/roadmap-status.md)
for the current implementation status and verified gaps.

## Current Implementation Status

Implemented:

- CMake scaffold for shared/static `rns8`, `rns8-inspect`, `rns8-verify`,
  `rns8-bench`, and Catch2 tests.
- Explicit C ABI headers and a C++ RAII wrapper skeleton.
- CPU reference path for bounded exact signed and unsigned 64-bit GEMM using
  persistent RNS matrices, centered residues, scalar per-modulus ring GEMM,
  Boost.Multiprecision CRT/Garner reconstruction, and range-error checks.
- Exact-wide signed and unsigned persistent RNS output with Boost-backed
  residue oracles and CPU little-endian limb export. Signed exact-wide export
  uses fixed-width two's-complement limbs over the centered CRT representative
  with the `x >= ceil(P / 2)` negative threshold; unsigned exact-wide export
  uses fixed-width magnitude limbs. Both use element-stride `ld`,
  `limb_count` in `[1, 32]`, and destination-preserving range errors instead
  of truncation. Exact-wide descriptors require `RNS8_BOUND_NONE` and are not
  routed through bounded i64/u64 or strict wrap64 export surfaces.
- Strict `mod 2^64` wraparound CPU GEMM through the explicit byte-limb backend,
  including one-shot and persistent byte-limb matrix APIs. This path returns
  low-64-bit `uint64_t` output and does not use odd-modulus CRT.
- Explicit finite-ring and finite-field `uint8_t` GEMM APIs for CPU reference
  and direct HIP, including one-shot and persistent resident matrix paths.
  Finite-ring calls accept moduli in `[2, 256]`; finite-field calls require
  prime moduli `<= 251`. These paths use explicit modulus arguments and
  prefix-zero finite storage, not the bounded CRT prefix ladder, exact-wide limb
  export, or strict wrap64 byte-limb backend.
- Default modulus ladder validation, prefix range-bit checks, composite and
  prime modulus tests, full 64-bit boundary tests, alternating-sign
  cancellation, and K-block splitting around 65536.
- Windows direct HIP bring-up through explicit hipcc object compilation for
  `gfx1100`, HIP device inspection, device-resident RNS matrix storage,
  signed/unsigned GPU residue conversion, fused INT32-to-centered-residue
  K-block reduction, bounded i64/u64 GPU CRT export, public strict wrap64
  byte-limb correctness path with device-current GEMM/export, and real
  one-modulus plus bounded i64/u64 and wrap64 smoke tests compared against the
  CPU reference.
- Benchmark schema v4 with host wall-clock phase timings, live git commit
  capture, raw timing arrays, summaries, direct-HIP GPU event timing arrays
  when complete, and explicit per-tile adaptive bounded capture metadata.

Not implemented yet:

- Optimized matrix-engine HIP kernels, hipBLASLt, CK, rocWMMA, AMDGPU builtin
  hot kernels, and optimized GPU strict `mod 2^64` byte-GEMM kernels.
- Optimized finite-field algorithms beyond the explicit-modulus
  correctness-grade CPU/direct-HIP finite path.
- Reviewed production performance claims; current benchmark captures are raw
  evidence only until baselines and gates are established.

## Windows Development Requirements

Required:

- Windows 11 x86-64.
- AMD Radeon GPU supported by the Windows HIP SDK. Local bring-up currently
  targets Radeon RX 7900 XTX / `gfx1100`.
- AMD HIP SDK for Windows.
- Visual Studio 2022 Community or Build Tools with MSVC C++ and Windows SDK.
- CMake.
- Ninja.
- Git.
- Python 3.11 or newer.
- vcpkg at `C:\vcpkg`.
- vcpkg packages declared in [vcpkg.json](vcpkg.json). The dependency checker
  marks only current host-required packages as readiness blockers; optional
  reference/planned packages are reported without enabling backends or making
  correctness-validation claims:
  - `benchmark`
  - `boost-multiprecision`
  - `catch2`
  - `cli11`
  - `flint`
  - `fmt`
  - `gmp`
  - `nlohmann-json`
  - `spdlog`
- Optional Radeon Developer Tool Suite for profiling and ISA inspection.

Python packages for benchmark orchestration and result analysis:

- `numpy`
- `pandas`
- `matplotlib`
- `pytest`
- `scipy`

Optional comparison libraries that are not required for Windows bring-up:

- GMP
- FLINT
- NTL
- FFLAS-FFPACK
- LinBox

## Windows Install Commands

Install standard tools with winget:

```powershell
winget install --exact --id Ninja-build.Ninja --source winget --accept-source-agreements --accept-package-agreements
winget install --exact --id Kitware.CMake --source winget --accept-source-agreements --accept-package-agreements
winget install --exact --id Git.Git --source winget --accept-source-agreements --accept-package-agreements
winget install --exact --id Python.Python.3.13 --source winget --accept-source-agreements --accept-package-agreements
```

Install Visual Studio tooling if MSVC is not already installed:

```powershell
winget install --exact --id Microsoft.VisualStudio.2022.BuildTools --source winget --accept-source-agreements --accept-package-agreements --override "--quiet --wait --norestart --add Microsoft.VisualStudio.Workload.VCTools --includeRecommended"
```

Install the AMD HIP SDK from AMD:

https://www.amd.com/en/developer/resources/rocm-hub/hip-sdk.html

Install vcpkg and C++ dependencies:

```powershell
git clone https://github.com/microsoft/vcpkg.git C:\vcpkg
C:\vcpkg\bootstrap-vcpkg.bat -disableMetrics
C:\vcpkg\vcpkg.exe integrate install
C:\vcpkg\vcpkg.exe install --classic --triplet x64-windows benchmark boost-multiprecision catch2 cli11 flint fmt gmp nlohmann-json spdlog
```

Add HIP SDK and vcpkg to the user PATH:

```powershell
$hipBin = 'C:\Program Files\AMD\ROCm\7.1\bin'
$vcpkgRoot = 'C:\vcpkg'
$vcpkgBin = 'C:\vcpkg\installed\x64-windows\bin'

$userPath = [Environment]::GetEnvironmentVariable('Path', 'User')
$parts = @($userPath -split ';' | Where-Object { $_ })

foreach ($entry in @($hipBin, $vcpkgRoot, $vcpkgBin)) {
  if ($parts -notcontains $entry) {
    $parts = @($entry) + $parts
  }
}

[Environment]::SetEnvironmentVariable('Path', ($parts -join ';'), 'User')
[Environment]::SetEnvironmentVariable('VCPKG_ROOT', $vcpkgRoot, 'User')
```

Restart the shell after changing PATH.

Install the Radeon Developer Tool Suite by extracting the AMD zip into a stable
local tools directory. The current local setup uses:

```text
~\Tools\RadeonDeveloperToolSuite-2026-02-02-1757
```

Add that directory to PATH if you want `rga.exe` and the Radeon tools to be
discoverable from a shell:

```powershell
$rdts = '~\Tools\RadeonDeveloperToolSuite-2026-02-02-1757'
$userPath = [Environment]::GetEnvironmentVariable('Path', 'User')
$parts = @($userPath -split ';' | Where-Object { $_ })
if ($parts -notcontains $rdts) {
  [Environment]::SetEnvironmentVariable('Path', (($parts + $rdts) -join ';'), 'User')
}
```

Do not commit the downloaded zip or extracted suite into this repository. Keep
downloaded installers and raw captures under `temp/`.

## Windows Validation

Verify the development environment:

```powershell
python tools/check_dependencies.py
```

Verify MSVC from a plain PowerShell shell. The wrapper locates Visual Studio
and loads `VsDevCmd.bat` automatically when the current shell is not already a
developer environment:

```powershell
python tools\windows_dev.py where cl
```

Initialize and probe repo-local CK and rocWMMA dependencies only when working
on those accelerator paths:

```powershell
python tools\bootstrap_rocm_accelerators.py --init --probe --target gfx1100
```

This command uses the pinned submodules under `third_party\rocm\`, verifies the
Windows HIP/MSVC toolchain through the automatic developer-environment wrapper,
compiles tiny CK and rocWMMA HIP probes for `gfx1100`, and writes its JSON
record under `temp\accelerator-deps\`. Do not clone CK or rocWMMA source trees
under `C:\`; generated dependency output belongs under ignored `temp\`, `out\`,
or `build\` paths in this repository.

When configuring CMake with vcpkg, pass the toolchain file explicitly:

```powershell
cmake -S . -B build -G Ninja -DCMAKE_TOOLCHAIN_FILE=C:\vcpkg\scripts\buildsystems\vcpkg.cmake
```

Once the CMake project scaffold exists, use the checked-in presets:

```powershell
cmake --list-presets
python tools\windows_dev.py cmake --preset windows-msvc-hip-debug
python tools\windows_dev.py cmake --build --preset windows-debug
python tools\windows_dev.py ctest --preset windows-debug --output-on-failure
```

Do not rely on CMake's HIP language support for the Windows path. The Windows
build should compile HIP sources through the HIP SDK compiler integration used
by the project CMake files.

Run the current tools:

```powershell
build\windows-msvc-hip-debug\rns8-inspect.exe --backend hip-direct --json
build\windows-msvc-hip-debug\rns8-verify.exe --hip-smoke
build\windows-msvc-hip-debug\rns8-bench.exe --backend cpu --semantics bounded-i64 --m 64 --n 64 --k 64 --warmups 1 --repeats 5 --seed 1
build\windows-msvc-hip-debug\rns8-bench.exe --backend hip-direct --semantics bounded-u64 --m 16 --n 16 --k 16 --warmups 1 --repeats 3 --seed 1
python tools\result_compare.py temp\baseline.json temp\candidate.json
```

`rns8-inspect --backend` accepts only explicit backend names. Unknown backend
strings are rejected instead of being routed to `auto`. In the default HIP
preset, `hipblaslt`, `ck`, and `rocwmma` print `unsupported backend` plus an
evidence-only accelerator note. In the opt-in hipBLASLt preset,
`hipblaslt` reports the compiled baseline backend while `ck` and `rocwmma`
remain unsupported. Inspect output includes the public backend capability
metadata so accelerator fail-fast state is visible without requiring benchmark
execution.

For CPU-only scaffold validation, configure without HIP:

```powershell
cmake -S . -B build\cpu-debug -G Ninja -DCMAKE_TOOLCHAIN_FILE=C:\vcpkg\scripts\buildsystems\vcpkg.cmake -DRNS8_ENABLE_HIP=OFF
cmake --build build\cpu-debug
ctest --test-dir build\cpu-debug --output-on-failure
```

## Repository Setup Files

- [AGENTS.md](AGENTS.md) contains durable instructions for future Codex agents.
- [CMakePresets.json](CMakePresets.json) defines Windows HIP and Linux ROCm
  configure/build/test presets plus evidence-only accelerator probe presets.
- [vcpkg.json](vcpkg.json) declares the C++ dependency set.
- [tools/bootstrap_rocm_accelerators.py](tools/bootstrap_rocm_accelerators.py)
  initializes the pinned repo-local CK and rocWMMA submodules and compile-probes
  them with `hipcc` for the local target. Its output is dependency readiness
  evidence only and does not enable CK or rocWMMA backends.
- [tools/check_dependencies.py](tools/check_dependencies.py) reports the local
  toolchain, HIP device, Python packages, `vcpkg.json` manifest packages,
  `CMakePresets.json` Windows/Linux HIP target representation, MSVC install,
  repo-local CK/rocWMMA submodule state, optional accelerator/reference
  components, project tools, and optional Radeon Developer Tool Suite utilities.
  `--accelerator-probes` runs opt-in tiny compile/run probes under
  `temp\accelerator-deps\` for discovered accelerator components; these probes
  are evidence only and do not enable correctness backends. The JSON
  report separates implemented correctness backend families from candidate
  accelerator evidence through `readiness.correctness_backend_validation`,
  records exact-wide Windows/Linux/Instinct validation boundaries, keeps
  accelerator records marked with
  `candidate_evidence_is_correctness_validation=false`, and uses
  `hard_cut_self_checks` only for internal report consistency.
- [tools/result_compare.py](tools/result_compare.py) compares two `rns8-bench`
  JSON captures without treating timing deltas as correctness or performance
  claims.
- [include/rns8/rns8.h](include/rns8/rns8.h) is the public C ABI. Packing is
  explicitly matrix-descriptor based; the ABI does not infer operand role or
  semantics from C++ types. Exact-wide limb export is separate from bounded
  i64/u64 and strict wrap64 export. Backend capability and plan backend
  metadata APIs expose selected kernels, accelerator readiness, workspace mode,
  ISA evidence, and autotune keys without enabling evidence-only accelerators.
- `temp/` is intentionally ignored and is the place for scratch files, raw
  benchmark captures, downloaded installers, and anything else that should not
  be tracked by git.
