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
- Opt-in Windows `gfx1100` accelerator correctness backends for hipBLASLt, CK,
  and rocWMMA. hipBLASLt is a baseline INT8-to-INT32 path with separate residue
  reduction. CK and rocWMMA use matrix-engine INT8 GEMM with fused centered
  residue output and exact CPU/direct-HIP differential coverage. All three
  still report `performance_validated=false` until reviewed captures prove
  target-shape wins.

Not implemented yet:

- Validated-fastest accelerator promotion, AMDGPU builtin hot kernels, and
  optimized GPU strict `mod 2^64` byte-GEMM kernels.
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
compiles header probes and object-only int8 matrix-engine primitive probes for
CK and rocWMMA on `gfx1100`, and writes its JSON record under
`temp\accelerator-deps\`. These probes are dependency readiness evidence only:
they do not run a backend, compare CPU/direct-HIP differentials, or enable CK or
rocWMMA. Do not clone CK or rocWMMA source trees under `C:\`; generated
dependency output belongs under ignored `temp\`, `out\`, or `build\` paths in
this repository.

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
build\windows-msvc-hip-debug\rns8-bench.exe --backend hip-vector-alu-int64 --semantics bounded-i64 --m 64 --n 64 --k 64 --warmups 1 --repeats 5 --seed 1
build\windows-msvc-hip-debug\rns8-bench.exe --backend auto --semantics bounded-i64 --m 8 --n 8 --k 8 --warmups 1 --repeats 1 --seed 23
build\windows-msvc-hip-debug\rns8-bench.exe --backend hip-direct --semantics exact-wide-signed --m 16 --n 16 --k 16 --warmups 1 --repeats 3 --seed 1
build\windows-msvc-hip-debug\rns8-bench.exe --backend hip-direct --semantics exact-wide-unsigned --m 16 --n 16 --k 16 --warmups 1 --repeats 3 --seed 1
build\windows-msvc-hip-debug\rns8-bench.exe --backend hip-direct --semantics finite-u8-ring --modulus 255 --m 64 --n 64 --k 64 --warmups 1 --repeats 3 --seed 1
build\windows-msvc-hip-debug\rns8-bench.exe --backend hip-direct --semantics finite-u8-field --modulus 251 --m 64 --n 64 --k 64 --warmups 1 --repeats 3 --seed 1
python tools\result_compare.py temp\baseline.json temp\candidate.json
```

Benchmark JSON includes a structured `comparison_baseline` object. Current raw
captures deliberately report `status=required_not_recorded` and
`speedup_claimed=false`; schema validation requires same-contract CPU/reference
and GPU baseline prerequisites before a capture can ever be promoted to a
speedup claim.

`hip-vector-alu-int64` is a benchmark-only backend name, not a public
`rns8_backend_kind`. It runs bounded i64/u64 inputs through benchmark-owned HIP
buffers and exact 192-bit-limb vector-ALU kernels, then emits
`backend_selected=hip-vector-alu-int64` with `performance_validated=false`.
Use it as the same-contract GPU baseline for accelerator reviews, not as a
production backend.

Run small Windows `gfx1100` benchmark sweeps and review reports under ignored
`temp\benchmark-sweeps\`:

```powershell
python tools\benchmark_sweep.py --bench build\windows-msvc-hip-debug\rns8-bench.exe --out-root temp\benchmark-sweeps\windows-gfx1100 --shape 64 --backend cpu --backend hip-direct --backend hip-vector-alu-int64 --warmups 1 --repeats 3 --seed 1
python tools\benchmark_sweep.py --bench build\windows-msvc-hip-debug\rns8-bench.exe --out-root temp\benchmark-sweeps\windows-gfx1100-exact-wide --semantics exact-wide-signed --semantics exact-wide-unsigned --case small:64,64,64 --backend cpu --backend hip-direct --warmups 1 --repeats 3 --seed 1
python tools\benchmark_sweep.py --bench build\windows-msvc-hip-debug\rns8-bench.exe --out-root temp\benchmark-sweeps\windows-gfx1100-finite --semantics finite-u8-ring --modulus 251 --modulus 255 --case small:64,64,64 --backend cpu --backend hip-direct --warmups 1 --repeats 3 --seed 1
python tools\benchmark_sweep.py --review-only --out-root temp\benchmark-sweeps\windows-gfx1100-reviewed --capture temp\benchmark-sweeps\windows-gfx1100\bounded-i64-shape-64x64x64-64x64x64-cpu.json --capture temp\benchmark-sweeps\windows-gfx1100\bounded-i64-shape-64x64x64-64x64x64-hip-direct.json --capture temp\benchmark-sweeps\windows-gfx1100\bounded-i64-shape-64x64x64-64x64x64-hip-vector-alu-int64.json
```

Add `--reuse-packed-inputs` to `rns8-bench` or `tools\benchmark_sweep.py`
when the contract is repeated use of the same packed A/B inputs. Use
`--reuse-packed-a` or `--reuse-packed-b` for repeated-A or repeated-B
amortization sweeps that keep one operand resident and repack the other per
repeat. These modes report explicit `pack_mode` and `prepack_reuse_operands`
metadata, record `prepack_setup_us`, report `prepack_reuse_strategy`, and keep
`end_to_end` scoped to the measured repeated workload. Eligible rocWMMA
`--reuse-packed-b` RNS captures materialize the real reusable B prepack cache
and stamp `prepack_reuse_strategy="rocwmma_reusable_b_cache"`; other reuse
captures currently report `persistent_matrix_residency`. They are benchmark
evidence for pack amortization, not production prepack caches; review tooling
marks these captures ineligible for normal AUTO autotune-cache promotion.

Created plans expose their current packing contract through
`rns8_get_plan_packing_info` and `rns8::Plan::packing_info()`. The query reports
the selected backend, persistent input/output layout versions, transient A/B
pack workspace bytes, accumulator or library workspace bytes, and reusable or
production prepack-cache availability. hipBLASLt and CK report transient
per-dispatch pack workspaces only. rocWMMA reports transient A workspaces plus a
reusable B prepack cache for eligible non-tiled RNS plans with `K <= 65536`.
Every backend still reports `production_prepack_cache_available=0` until a
broader source-versioned production cache policy exists. Matrix handles expose
the matching resident storage state through
`rns8_get_matrix_storage_info` and `rns8::Matrix::storage_info()`: source
version, finite modulus, host/device currentness, byte counts, and persistent
layout version. Prepack caches key or reject against both the plan packing
contract and this matrix storage state. `rns8_get_prepack_cache_key_info` and
`rns8::prepack_cache_key_info()` validate a plan plus A/B operand matrix before
emitting deterministic key material; they reject role, shape, layout, backend,
device id, currentness, source-version, and finite-modulus mismatches.
Materialized caches expose the same key material plus device and allocation
bytes through `rns8_get_prepack_cache_info` and `rns8::PrepackCache::info()`.
They still report no production cache availability.

The review report groups captures by semantic input contract, reports CPU,
direct-HIP, and vector-ALU baseline coverage for bounded i64/u64. Exact-wide
signed/unsigned and finite-u8 reviews require CPU and direct-HIP baselines;
vector-ALU is not applicable. finite-u8 plan/autotune keys include the explicit
finite modulus so reviewed entries cannot alias different rings or fields.
Accelerator entries are promotable only when they beat the required
same-contract GPU baselines, and only the fastest promotable accelerator in a
contract group is written to the autotune cache. Raw
`rns8-bench --write-autotune-cache` writes are always refused; use
`tools\benchmark_sweep.py --bench-for ck=build\windows-msvc-ck-release\rns8-bench.exe`
style overrides when a release sweep combines captures from opt-in accelerator
build directories, and `--write-autotune-cache --autotune-cache temp\reviewed-autotune.json`
until the generated report has been reviewed. Production promotion also
requires `--review-mode release`; the default smoke review mode never writes a
`performance_validated=true` entry.
Reviewed temp cache files can be merged into an installable cache only through
the validating installer; it rejects non-reviewed entries and stale identity
fields before writing:

```powershell
python tools\install_autotune_cache.py --source temp\reviewed-autotune-bounded-i64-full.json --source temp\reviewed-autotune-adaptive-bounded-full.json --source temp\reviewed-autotune-finite-full-plan-keyed.json --source temp\reviewed-autotune-exact-wide-full.json --destination temp\reviewed-autotune-production-candidate.json
```

The default local cache can then be populated from that reviewed candidate:

```powershell
python tools\install_autotune_cache.py --source temp\reviewed-autotune-production-candidate.json --replace-existing
```

`--replace-existing` is intentionally explicit. A normal merge refuses to carry
forward stale or non-reviewed destination entries; replacement validates the
reviewed sources and writes only those entries to the default cache path
(`%LOCALAPPDATA%\rns8-gemm\autotune.json` on Windows).

Release performance promotion uses release opt-in presets, not debug captures:

```powershell
python tools\windows_dev.py cmake --preset windows-msvc-hipblaslt-release
python tools\windows_dev.py cmake --build --preset windows-hipblaslt-release
python tools\windows_dev.py ctest --preset windows-hipblaslt-release --output-on-failure
python tools\windows_dev.py cmake --preset windows-msvc-ck-release
python tools\windows_dev.py cmake --build --preset windows-ck-release
python tools\windows_dev.py ctest --preset windows-ck-release --output-on-failure
python tools\windows_dev.py cmake --preset windows-msvc-rocwmma-release
python tools\windows_dev.py cmake --build --preset windows-rocwmma-release
python tools\windows_dev.py ctest --preset windows-rocwmma-release --output-on-failure
python tools\benchmark_sweep.py --bench build\windows-msvc-hip-release\rns8-bench.exe --bench-for hipblaslt=build\windows-msvc-hipblaslt-release\rns8-bench.exe --bench-for ck=build\windows-msvc-ck-release\rns8-bench.exe --bench-for rocwmma=build\windows-msvc-rocwmma-release\rns8-bench.exe --out-root temp\benchmark-sweeps\windows-gfx1100-release-reviewed --review-mode release --release-matrix --include-default-adaptive --include-exact-wide --semantics bounded-i64 --semantics bounded-u64 --warmups 3 --repeats 9 --seed 1
```

Large `2048`, `4096`, and `8192` shapes can be added with
`--include-exploratory-large`; they remain exploratory unless the same-contract
CPU/reference baselines complete and the review report marks the group
promotable. AMDGPU builtin and wrap64 matrix-engine paths stay disabled unless
reviewed captures prove a concrete candidate beats the current CK/rocWMMA or
`direct_hip_wrap64_byte_gemm36_tiled_2d_v3` path with exact differentials and
ISA evidence.

The reviewed release cache entries have been merged into
`temp\reviewed-autotune-production-candidate.json` and installed into this
Windows `gfx1100` workstation's default local cache with 19 entries. The cache
reader, schema checks, review reports, and `rns8-inspect --autotune-key`
rationale exist, including runtime target/version rejection for exact cache
hits. AUTO HIP contexts can select reviewed release cache hits for compiled,
runtime-probed bounded-i64, adaptive bounded-i64, finite-u8, and exact-wide
HIP-resident accelerator candidates and otherwise fall back to the configured
direct-HIP GPU correctness path, or CPU when GPU support is unavailable. New
production entries still require `--review-mode release` and at least three
warmups plus nine measured repeats for the complete same-contract group;
uninstalled release-smoke cache files remain evidence only.
Opt-in accelerator CTest presets include hermetic fake-default-cache AUTO
smokes so selection does not rely on this workstation's real cache contents.
Bounded, exact-wide signed, and finite-u8 fake-default-cache smokes have been
run on Windows `gfx1100` for the relevant hipBLASLt, CK, and rocWMMA presets;
they are synthetic cache coverage separate from the release-reviewed entries
below.

A Windows `gfx1100` release-mode bounded-i64 matrix exists under
`temp\benchmark-sweeps\windows-gfx1100-release-bounded-i64-full`. It covered
square shapes 64, 128, 512, and 1024 with CPU, direct-HIP, vector-ALU,
hipBLASLt, CK, and rocWMMA at three warmups, nine repeats, and seed
`20260602`. All four same-contract groups had the required baselines. The
64 and 128 groups were not promoted because `hip-vector-alu-int64` remained
faster than every accelerator. The 512 group wrote a temp reviewed WMMA entry:
rocWMMA `rocwmma_i8_i32_signed_hot_residue_v1` measured 2399 us median
end-to-end, ahead of CK at 2408 us, vector-ALU at 3217 us, direct HIP at
4263 us, hipBLASLt at 6270 us, and CPU reference at 1542970 us. The 1024 group
wrote a temp reviewed hipBLASLt entry:
`hipblaslt_int8_i32_scratch_reduce_baseline_v1` measured 8326 us, ahead of
direct HIP at 11195 us, vector-ALU at 11327 us, rocWMMA at 11565 us, CK at
18109 us, and CPU reference at 15657400 us. The cache is
`temp\reviewed-autotune-bounded-i64-full.json`, with exact `rns8-inspect` hits
for runtime target `gfx1100` and runtime versions
`repo-local release/rocm-rel-7.1` for WMMA and `hipBLASLt 100100` for
hipBLASLt. With `RNS8_AUTOTUNE_CACHE_PATH` pointed at that temp cache,
schema-valid AUTO smokes select `backend_selected=wmma` for 512 and
`backend_selected=hipblaslt` for 1024, with
`backend_metadata.performance_validated=true` and
`comparison_baseline.status=reviewed_release_same_contract_baseline`.

A matching Windows `gfx1100` release-mode bounded-u64 matrix exists under
`temp\benchmark-sweeps\windows-gfx1100-release-bounded-u64-full`. It covered
the same four square shapes and backend set at three warmups, nine repeats,
and seed `20260602`. All four groups had complete baselines, but none wrote a
cache entry because `hip-vector-alu-int64` stayed fastest at every shape:
361 us at 64, 452 us at 128, 1653 us at 512, and 5649 us at 1024 median
end-to-end. The cache write status is `no_promotable_entries`, and AUTO
therefore has no bounded-u64 reviewed accelerator promotion from this matrix.

A Windows `gfx1100` release-mode adaptive bounded matrix exists under
`temp\benchmark-sweeps\windows-gfx1100-release-adaptive-bounded-full`. It
covered the default 65x65x64 and 1024x1024x1024 per-tile schedules for bounded
i64 and bounded u64 with CPU, direct-HIP, vector-ALU, CK, and rocWMMA at three
warmups, nine repeats, and seed `20260602`. It wrote one temp reviewed cache
entry to `temp\reviewed-autotune-adaptive-bounded-full.json`: bounded i64
1024 selected rocWMMA `rocwmma_i8_i32_signed_tiled_hot_residue_v1` at 5095 us
median end-to-end, ahead of direct HIP at 6469 us, CK at 6854 us, vector-ALU at
13310 us, and CPU reference at 3774230 us. The bounded i64 tiny case and both
bounded u64 adaptive cases stayed blocked by vector-ALU baselines. A matching
AUTO smoke selects `backend_selected=wmma`, reports
`backend_metadata.performance_validated=true`, and validates as schema v4.

A Windows `gfx1100` release-mode finite-u8 matrix exists under
`temp\benchmark-sweeps\windows-gfx1100-release-finite-full-plan-keyed`. It
covered ring moduli 251 and 255 plus field modulus 251 for shapes 64, 128,
512, and 1024 with CPU, direct-HIP, hipBLASLt, CK, and rocWMMA captures at
three warmups, nine repeats, and seed `20260602`. All 12 review groups had the
required baselines and release counts, and the report wrote 12 temp reviewed
entries to `temp\reviewed-autotune-finite-full-plan-keyed.json`. rocWMMA won
the 64, 128, and 512 groups for all three finite contracts; CK won the 1024
ring groups at 1428 us for modulus 251 and 1354 us for modulus 255; hipBLASLt
won the 1024 field-251 group at 2327 us. `rns8-inspect` reports exact
runtime-target/version hits for representative hipBLASLt, CK, and rocWMMA
entries, and schema-valid AUTO smokes select those three backends with
`backend_metadata.performance_validated=true`.

A Windows `gfx1100` release-mode exact-wide matrix exists under
`temp\benchmark-sweeps\windows-gfx1100-release-exact-wide-full`. It covered
exact-wide signed and unsigned for shapes 64, 128, 512, and 1024 with CPU,
direct-HIP, hipBLASLt, CK, and rocWMMA captures at three warmups, nine repeats,
and seed `20260602`. All eight same-contract groups had required CPU and
direct-HIP baselines. The report wrote four temp reviewed CK entries to
`temp\reviewed-autotune-exact-wide-full.json`: exact-wide signed 1024 at
19686 us, exact-wide unsigned 128 at 2995 us, exact-wide unsigned 512 at
6753 us, and exact-wide unsigned 1024 at 15393 us median end-to-end. Exact-wide
signed 64/128/512 and exact-wide unsigned 64 stayed on direct-HIP because no
accelerator beat the same-contract direct-HIP baseline. `rns8-inspect` reports
exact default-cache hits for the four CK entries, and schema-valid AUTO captures
under `temp\default-cache-auto-exact-wide-reviewed` select
`backend_selected=ck`, report `backend_metadata.performance_validated=true`,
and include exact-wide export GPU event phases.

A Windows `gfx1100` release-mode strict wrap64 baseline review exists under
`temp\benchmark-sweeps\windows-gfx1100-release-wrap64-baseline-full`. It
covered 64x64x64, 128x128x128, 512x512x512, and 1024x1024x1024 with CPU
byte-limb reference and direct HIP at three warmups, nine repeats, and seed
`20260602`. Direct HIP `direct_hip_wrap64_byte_gemm36_tiled_2d_v3` remains the
measured production GPU path: 1828 us at 64, 2090 us at 128, 7757 us at 512,
and 39359 us at 1024 median end-to-end. The CPU
`cpu_wrap64_byte_limb_reference_v1` path measured 710 us, 5845 us, 576082 us,
and 4729230 us at those shapes while still consuming persistent byte-limb
storage and using exact unsigned `uint64_t` wraparound arithmetic for the
low-64 product. An internal rocWMMA wrap64 byte-GEMM36 candidate now consumes
the same compact byte-limb device buffers, decomposes each unsigned byte
product into signed WMMA plus high-bit correction WMMA terms, and matches
direct HIP plus the CPU byte-pair oracle on Windows `gfx1100` across
single-cell K tails, exact 16x16x16 WMMA tiles, padded carry-heavy tile tails,
ragged two-tile output, release-shaped 64x64x64 and 128x128x128 full-output
differentials against the CPU oracle, full 512x512x512 and 1024x1024x1024
candidate-vs-direct-HIP output checks with sampled CPU-oracle cells, and the
current `k=32768` accepted / `k=32769` rejected boundary. The benchmark smoke
also runs same-seed 64x64x64 CPU byte-limb, direct-HIP, and rocWMMA-candidate
captures and requires matching `checksum_u64` values. It is not a public
backend, not selected by AUTO, and not release performance evidence. Raw timing
capture is available only through
`rns8-bench --backend rocwmma-wrap64-candidate --semantics wrap-u64`, which
reports `backend_selected: "wmma"`, a benchmark-owned static 16x16 schedule,
candidate-specific HIP event label
`wrap64_wmma_candidate_gemm36_kernel_group`, and
`performance_validated: false`.
`tools\benchmark_sweep.py --include-wrap64 --release-matrix` now generates the
same 64, 128, 512, and 1024 square-shape wrap64 CPU/direct-HIP baseline matrix
used by other release reviews, plus optional exploratory large shapes when
`--include-exploratory-large` is set. Add
`--include-wrap64-wmma-candidate` to include raw internal candidate captures in
the review report; the reviewer marks them with
`internal_candidate_not_public_backend`, so they cannot produce autotune cache
entries. The matrix-engine path still needs public backend integration,
reviewed release captures proving it beats direct-HIP v3, and promotion review
of any required exhaustive 512/1024 CPU-oracle dumps before it can displace the
current production GPU path.

A candidate-inclusive Windows `gfx1100` release review under
`temp\benchmark-sweeps\windows-gfx1100-release-wrap64-wmma-candidate-current`
used three warmups, nine repeats, and seed `20260603` for 64, 128, 512, and
1024 square wrap64 shapes. All CPU byte-limb, direct-HIP, and rocWMMA-candidate
captures matched `checksum_u64` within each shape, but the candidate did not win
any shape: candidate medians were 4825 us, 5202 us, 37481 us, and 264657 us
versus direct-HIP medians of 3653 us, 1852 us, 9430 us, and 41237 us. The review
reported zero promotable entries with `internal_candidate_not_public_backend`
and `not_faster_than_direct_hip` blockers for the candidate.

The packed low-bit matrix-engine pipeline is also roadmap work, not a completed
runtime backend. Planned layout families include `rns_i8_modulus_major_v2`,
`rns_i8_tile_swizzled_b_v1`, `finite_u8_centered_plane_v2`,
`wrap64_byte_limb_gemm36_v2`, and research-only `rns_i4_packed_v0`. Those
layouts must prove source-version invalidation, layout mismatch rejection,
exact CPU/direct-HIP differentials, ISA evidence, and pack amortization for
one-shot, repeated-A, repeated-B, and repeated-A/B workloads before they can
displace current layouts. The current benchmark can generate those repeated
workload evidence modes with `--reuse-packed-a`, `--reuse-packed-b`, and
`--reuse-packed-inputs`, but broad durable packed-layout/prepack-cache
production work still remains roadmap work. A narrow reusable rocWMMA B-operand
cache now exists for non-tiled RNS plans with `K <= 65536`:
`rns8_create_prepack_cache` packs B into the rocWMMA column-major layout once,
and `rns8_gemm_rns_prepacked_b` reuses that cache while A remains a transient
per-dispatch pack. Unsupported roles, backends, finite/wrap64 semantics, tiled
schedules, and oversize K shapes return unsupported or invalid status instead
of falling back silently. `rns8_get_plan_packing_info` exposes the current
plan-specific transient pack workspace layout, B-cache availability, and byte
contract so cache tooling can reject layout/cache mismatches instead of
inferring them from backend names; `rns8_get_matrix_storage_info` exposes the
matrix source version and resident currentness needed for source-version
invalidation, `rns8_get_prepack_cache_key_info` validates concrete plan/operand
key material before a cache can be reused, and `rns8_get_prepack_cache_info`
reports the created cache's matching key, device id, and allocation byte
contract. The production cache flag remains
`production_prepack_cache_available=0` until broader validated production cache
policy exists.

`rns8-inspect --backend` accepts only explicit backend names. Unknown backend
strings are rejected instead of being routed to `auto`. In the default HIP
preset, `hipblaslt`, `ck`, and `rocwmma` print `unsupported backend` plus an
evidence-only accelerator note. In the opt-in hipBLASLt preset, `hipblaslt`
reports the compiled baseline backend while `ck` and `rocwmma` remain
unsupported. In the opt-in CK or rocWMMA presets, the selected backend reports
compiled correctness support, selected kernel, workspace mode, exact
differential validation, and ISA evidence while unselected accelerators remain
unsupported. Inspect output includes the public backend capability metadata so
accelerator fail-fast state is visible without requiring benchmark execution.

CK is an opt-in Windows `gfx1100` accelerator build:

```powershell
python tools\windows_dev.py cmake --preset windows-msvc-ck-debug
python tools\windows_dev.py cmake --build --preset windows-ck-debug
python tools\windows_dev.py ctest --preset windows-ck-debug --output-on-failure
build\windows-msvc-ck-debug\rns8-bench.exe --backend ck --semantics bounded-i64 --m 64 --n 128 --k 64 --warmups 1 --repeats 2 --seed 13
build\windows-msvc-ck-debug\rns8-bench.exe --backend ck --semantics bounded-u64 --bound-mode per-tile --m 65 --n 65 --k 64 --tile-m 64 --tile-n 64 --warmups 1 --repeats 2 --seed 7 --require-adaptive-execution
build\windows-msvc-ck-debug\rns8-bench.exe --backend ck --semantics finite-u8-ring --modulus 255 --m 64 --n 128 --k 64 --warmups 1 --repeats 2 --seed 13
```

The CK backend uses repo-local Composable Kernel headers and RNS8-owned HIP
packing/output kernels to run fused centered-residue `int8 x int8 -> int32`
GEMM for fixed-prefix bounded plans, adaptive per-tile bounded plans,
exact-wide RNS output, and finite u8. It remains opt-in and
`performance_validated=false` until reviewed captures prove it is the fastest
accepted backend for a target shape. CK benchmark captures currently report
host wall-clock phase timings plus HIP event operation-group timings when event
capture is complete. The CK preset generates RNS8's repo-local WMMA no-divide
block-map include overlay from the pinned CK header during configure, puts that
overlay before CK's include directory for the CK HIP compile, records the
generated patched header as a dependency of the compiled CK HIP object, and
fails configure if CK's `MakeDefaultBlock2CTileMap` block no longer matches
the expected upstream or patched form. The CK ISA gate requires the expected `v_wmma`
instruction and rejects scalar divide/remainder/reciprocal mnemonics plus
unintended INT32 global stores in matched CK WMMA symbols. The CK event hook
records a single `rns_gemm_kernel_group` label for the backend device call;
finer CK per-kernel phase breakdowns and reviewed fastest-backend validation
are still separate readiness items.

rocWMMA is an opt-in Windows `gfx1100` accelerator build:

```powershell
python tools\windows_dev.py cmake --preset windows-msvc-rocwmma-debug
python tools\windows_dev.py cmake --build --preset windows-rocwmma-debug
python tools\windows_dev.py ctest --preset windows-rocwmma-debug --output-on-failure
build\windows-msvc-rocwmma-debug\rns8-bench.exe --backend rocwmma --semantics bounded-i64 --m 64 --n 128 --k 64 --warmups 1 --repeats 2 --seed 23
build\windows-msvc-rocwmma-debug\rns8-bench.exe --backend rocwmma --semantics bounded-u64 --bound-mode per-tile --m 65 --n 65 --k 64 --tile-m 64 --tile-n 64 --warmups 1 --repeats 2 --seed 29 --require-adaptive-execution
build\windows-msvc-rocwmma-debug\rns8-bench.exe --backend rocwmma --semantics finite-u8-field --modulus 251 --m 64 --n 128 --k 64 --warmups 1 --repeats 2 --seed 23
```

The rocWMMA backend uses repo-local rocWMMA headers and RNS8-owned HIP kernels
to pack signed centered residues into 16-aligned panels, execute
`int8 x int8 -> int32` WMMA on `gfx1100`, and fuse INT32 reduction back to
centered `int8_t` residues without global INT32 scratch output. It supports
fixed-prefix bounded plans, adaptive per-tile bounded plans, exact-wide RNS
output, and finite u8. The ISA gate requires the expected `v_wmma` instruction
and rejects scalar divide/remainder/reciprocal mnemonics plus unintended INT32
global stores. rocWMMA benchmark captures currently report host wall-clock
phase timings plus HIP event operation-group timings when event capture is
complete. The rocWMMA event hook records a single `rns_gemm_kernel_group` label
for the backend device call; finer per-kernel/per-tile phase breakdowns and
reviewed fastest-backend validation remain separate readiness items.
`RNS8_ENABLE_AMDGPU_BUILTINS`
still fails fast until target-specific builtin kernels exist.

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
  initializes the pinned repo-local CK and rocWMMA submodules, compile-probes
  their headers, and object-compiles RNS8-owned int8 matrix-engine primitive
  probes with `hipcc` for the local target. Its output is dependency readiness
  evidence only and does not enable CK or rocWMMA backends by itself.
- [tools/check_dependencies.py](tools/check_dependencies.py) reports the local
  toolchain, HIP device, Python packages, `vcpkg.json` manifest packages,
  `CMakePresets.json` Windows/Linux HIP target representation, MSVC install,
  repo-local CK/rocWMMA submodule state, optional accelerator/reference
  components, project tools, and optional Radeon Developer Tool Suite utilities.
  `--accelerator-probes` runs opt-in compile/run probes plus CK/rocWMMA int8
  primitive object probes under `temp\accelerator-deps\` for discovered
  accelerator components; these probes are evidence only and do not enable
  correctness backends by themselves. The JSON
  report separates implemented correctness backend families from candidate
  accelerator evidence through `readiness.correctness_backend_validation`,
  records exact-wide Windows/Linux/Instinct validation boundaries, keeps
  accelerator records marked with
  `candidate_evidence_is_correctness_validation=false`, and uses
  `hard_cut_self_checks` only for internal report consistency.
- [tools/result_compare.py](tools/result_compare.py) compares two `rns8-bench`
  JSON captures without treating timing deltas as correctness or performance
  claims. Backend and selected-kernel differences are reported separately from
  the same semantic contract; GPU target/toolchain compatibility is enforced
  only for GPU-vs-GPU comparisons so CPU/reference baselines remain comparable.
- [tools/benchmark_sweep.py](tools/benchmark_sweep.py) runs fixed or explicit
  command matrices for bounded, adaptive bounded, exact-wide, finite-u8, and wrap64
  captures, writes JSON/Markdown review reports under ignored
  `temp\benchmark-sweeps\`, and writes autotune cache entries only for fastest
  reviewed same-contract accelerator winners. Review reports use schema v3 and
  include source target/toolchain/library metadata, per-phase speedups,
  promotion blockers, winner rationale, workspace bytes, event status, and
  explicit cache-write state.
- [include/rns8/rns8.h](include/rns8/rns8.h) is the public C ABI. Packing is
  explicitly matrix-descriptor based; the ABI does not infer operand role or
  semantics from C++ types. Exact-wide limb export is separate from bounded
  i64/u64 and strict wrap64 export. Backend capability and plan backend
  metadata APIs expose selected kernels, accelerator readiness, workspace mode,
  ISA evidence, and autotune keys without enabling evidence-only accelerators.
- `temp/` is intentionally ignored and is the place for scratch files, raw
  benchmark captures, downloaded installers, and anything else that should not
  be tracked by git.
