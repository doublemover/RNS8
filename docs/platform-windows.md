# Windows HIP SDK setup

Use this guide for native Windows Radeon builds. The [dependency guide](../third_party/README.md)
owns common requirements and pins. The [readiness reference](platform-readiness.md)
separates installed tools from compiled and device-validated paths.

## Prerequisites

Use the selected Python 3.15.1 and exact non-GPU tool/package candidates from the
[version policy](dependency-versions.md). Their upgrade qualification remains
pending across the full platform matrix. The user authorized the ROCm Core SDK
10.1 migration on 2026-10-26. The fresh local Release validation below uses
`C:/TheRock/build` and automatically selects its
`lib/llvm/bin/amdclang++.exe` compiler.
Use [SDK initialization](dependency-discovery.md#sdk-initialization) to download
the correct family archive or reuse an existing extraction.

Install a GPU/driver combination supported by the selected
[Windows HIP SDK release](https://rocm.docs.amd.com/projects/install-on-windows/en/docs-7.1.1/reference/system-requirements.html)
and the [HIP SDK development components](https://rocm.docs.amd.com/projects/install-on-windows/en/docs-7.1.1/how-to/install.html).
Install [Visual Studio's C++ desktop workload and Windows SDK](https://learn.microsoft.com/en-us/cpp/build/vscpp-step-0-installation),
CMake, Ninja, Git, Python and vcpkg dependencies as described in the dependency
guide. Historical measurements use VS2026 and HIP 7.1, including the build-local
LLVM math-header fix described below; they do not qualify the new SDK.

From the repository root, inspect existing tools:

```powershell
python tools/check_dependencies.py
python tools/windows_dev.py where cl
```

The checker reports optional profiling and accelerator components too. Its
missing optional tools do not all block a direct-HIP build. It runs no installer;
`--accelerator-probes` separately requests compile/run probes. Driver installation
and system changes are explicit administrative tasks.

## Configure and build

Run `python tools/init.py` to select the exact repository Python, detect GPU
targets, reuse or provision the matching SDK, resolve compiler/LLVM/bitcode,
VS, CMake/Ninja and prepared vcpkg package paths, and build Release. Use
`--dry-run --json` to inspect the plan or `--sdk-test` to run registered tests
after building. Selection is scoped to child processes; no manual `HIP_PATH`,
`LLVM_PATH` or PATH edits are needed. See
[SDK initialization](dependency-discovery.md#sdk-initialization) for overrides
and optional-backend flags.

The shared pre-project [selection contract](dependency-discovery.md) resolves an
unambiguous existing SDK/vcpkg stack. Explicit roots take precedence and invalid
choices fail. These examples intentionally use nondefault roots with spaces:

```powershell
$env:VCPKG_ROOT = 'D:/Development Tools/vcpkg'
$rns8HipRoot = 'D:/AMD SDK/ROCm/7.1'
python tools/check_dependencies.py --resolve-only --json --set "RNS8_HIP_ROOT=$rns8HipRoot" --set "VCPKG_ROOT=$env:VCPKG_ROOT" --set RNS8_ENABLE_HIP=ON
python tools/windows_dev.py cmake --preset windows-msvc-hip-debug "-DRNS8_HIP_ROOT=$rns8HipRoot" "-DCMAKE_TOOLCHAIN_FILE=$env:VCPKG_ROOT/scripts/buildsystems/vcpkg.cmake" -DRNS8_AMDGPU_TARGETS=gfx1100
python tools/windows_dev.py cmake --build --preset windows-debug --parallel 4
.\build\windows-msvc-hip-debug\rns8-bench.exe --backend hip-direct --semantics bounded-i64 --m 32 --n 32 --k 64 --warmups 1 --repeats 3 --seed 3135
```

`gfx1100` selects the RX 7900 XTX target. For a different GPU, first check AMD's
release matrix and set the matching target. The benchmark verifies its seeded
integer result and emits JSON. This small checked workload is a first execution;
[qualification](gpu-qualification.md) covers a broader campaign.

The preset no longer requires workstation SDK/vcpkg paths. Environment roots,
PATH evidence and bounded installation roots are considered by the same checker
used by CMake; equally eligible SDKs require an explicit choice. Selection is
filesystem evidence, with broader API/ABI/probe acceptance still tracked by
[#57](https://github.com/doublemover/RNS8/issues/57). The presets default
`VCPKG_MANIFEST_INSTALL=OFF`; install manifest packages explicitly as described in
the dependency guide before configuring. Existing caches may need that flag set
explicitly to disable their vcpkg toolchain's previous install setting.
`windows-msvc-hip-release` configures the corresponding Release tree; use the
`windows-release` build preset. Optional roots include `RNS8_CK_ROOT`,
`RNS8_ROCWMMA_ROOT`, `RNS8_ROCM_DEPS_BUILD_ROOT` and
`RNS8_ROCM_DEPS_INSTALL_ROOT` when enabling those components.

The wrapper loads the VS developer environment. Existing developer environments
and `VSINSTALLDIR` overrides are honored. A fresh build directory is required
when changing MSVC generations; cached compiler paths and newer STL headers
must match. Windows uses explicit `hipcc` integration without CMake's HIP
language. With HIP 7.1/MSVC 14.51, CMake applies
[LLVM's declaration-order fix](https://github.com/llvm/llvm-project/pull/201563)
to a build-local wrapper when needed, leaving the installed SDK intact.

## Optional accelerators

[Backend capabilities](backend-notes.md) describe the public opt-in paths.
Each requires HIP plus discoverable compatible libraries/headers:

| Component | Configure preset | Build/test preset | Explicit switch |
|---|---|---|---|
| hipBLASLt | `windows-msvc-hipblaslt-debug` | `windows-hipblaslt-debug` | `RNS8_ENABLE_HIPBLASLT=ON` |
| CK | `windows-msvc-ck-debug` | `windows-ck-debug` | `RNS8_ENABLE_CK=ON` |
| rocWMMA | `windows-msvc-rocwmma-debug` | `windows-rocwmma-debug` | `RNS8_ENABLE_ROCWMMA=ON` |
| AMDGPU builtins | `windows-msvc-amdgpu-builtins-debug` | `windows-amdgpu-builtins-debug` | `RNS8_ENABLE_AMDGPU_BUILTINS=ON` |

Pass the same HIP/toolchain overrides to these configure presets. The pinned
CK/rocWMMA submodules can be initialized and probed explicitly:

```powershell
python tools/bootstrap_rocm_accelerators.py --dry-run --init --probe --target gfx1100 --json
python tools/bootstrap_rocm_accelerators.py --init --probe --target gfx1100
```

The first command prints a plan. The second can fetch pinned submodules and
compile probes; it does not install drivers or qualify backend arithmetic.
[Third-party policy](../third_party/README.md) records the pins and overlay rules.

## Tests, packaging and evidence

Follow [Contributing](../CONTRIBUTING.md) for CPU-only development, sanitizer,
CTest and install/consumer commands. Use [GPU qualification](gpu-qualification.md)
for target testing and [performance methodology](performance.md) for measurements.
The [October 23 all-feature record](local-all-features-20261023.md) and
[larger-workload report](7900xtx-large-throughput-2026-10-23.md) identify the
specific compiled stack and observed subset. This guide's nondefault-root
commands have been checked against current interfaces; a clean installation at
those illustrative paths has not been exercised.

The Windows clang-cl sanitizer option instruments all project C++ targets and
links the matching LLVM runtime thunk. Use the `windows-clang-asan-debug` preset
for libFuzzer: its static CRT and `x64-windows-static` packages must agree with
LLVM's shipped fuzzer library. A dynamic-CRT fuzzer configuration is rejected
before building. Installed workload checks copy the selected ASan DLL beside
consumer executables and preserve instrumentation and CRT selection.

## October 26 Core SDK 10.1 local Release validation

The [source-bound record](../metadata/hip101-validation-20261026.json) reports
the complete fresh Windows x64 Release run at `059fa45`: **1,080 passed,
30 skipped, zero failures across 1,110 registered tests**, in 963.43 seconds.
The skipped tests require unavailable sparse/CDNA/FP8/TDM/AMX features. The
build enables direct HIP, AMDGPU builtins, CK, rocWMMA, hipBLASLt, OpenMP,
GMP/FLINT, static/shared libraries, tools, benchmarks and examples. Installed
public consumers are included in the suite; the installed native-device
example also passes synchronous and prepared-graph execution with both static
and shared RNS8 packages, **four additional checks**.

The selected stack is Core SDK10.1.0, HIP7.16.26385, AMD Clang24.0.0git,
CK1.3.0, rocWMMA2.2.1, hipBLASLt1.4.1, VS2026 18.11.2/MSVC19.51,
Python3.15.1, CMake4.4.4 and Ninja1.13.2. Manifest packages use
Boost1.92.0, Catch2 3.16.1, GMP6.3.0, FLINT3.6.0 and MPFR4.2.2.
The RX7900XTX reports `gfx1100` and 25,753,026,560 device bytes; its Windows
display driver is 32.0.31041.5005. HIP's API driver version is a separate value.

Windows System32 DLL search precedence initially loaded HIP71626384 despite
the selected SDK being on PATH. That partial run was stopped. Build/install
runtime staging now places the selected HIP DLL and `rocm_kpack.dll` beside
executables; a loaded-module regression requires their exact SDK hashes.
The final complete rerun and sampled hipBLASLt/compiler modules use the selected
SDK. Earlier failures, their repairs and the stopped run remain in the record.
The 3 GiB-budget address regression physically allocates and checks a
2,147,549,184-byte device input, crossing signed 32-bit addressing; this check
is separate from the historical over-VRAM capacity study.

The automatic no-location initializer was also exercised from PATH Python3.13.3:
it selected Python3.15.1, the nested SDK compiler and the matching managed
vcpkg installation, then completed a fresh static Release build. Optional
GMP/FLINT bindings follow that selected prefix rather than mixing headers and
libraries from another installation. CK's detected signed-input contract
selects centered packing on CK1.3, with exact CPU/HIP differential coverage
including K=65,536 and 65,537.

The same SDK compiles the gfx942 FP8 path and emits
`v_mfma_f32_16x16x32_fp8_fp8`; this is compile/ISA evidence only. Its compiler
lacks `__builtin_amdgcn_s_wait_dscnt`, so the guarded gfx1250 TDM path remains
unavailable. Linux, other GPU families, multi-GPU and matched prior/new-stack
performance retain their own qualification gates. The local configuration
follow-up below uses this selected SDK; no stopped performance campaign was resumed.

## October 26 configuration follow-up

The [compact results](../metadata/sdk101-followup-20261026.json) and
[verified full archive](../metadata/sdk101-followup-archive-20261026.json)
bind the following builds and checks to clean source `72c55c2`:

| Configuration | Selected tests | Passed | Skipped | Failed |
|---|---:|---:|---:|---:|
| HIP Debug, all optional providers | 1,014 | 983 | 31 | 0 |
| HIP Release, OpenMP OFF, all optional providers | 1,014 | 984 | 30 | 0 |
| CPU clang-cl ASan, RelWithDebInfo, OpenMP OFF | 540 | 505 | 35 | 0 |
| Fresh primary Release artifact integration | 8 | 8 | 0 | 0 |

The three matrix lanes cover compiled tests, build-specific CLI/ISA checks and
installed static/shared consumers. Each excludes 98 unchanged source-only
tooling tests. Target/feature skips remain explicit; Debug additionally skips
the allocation-fault harness because it excludes Debug CRT. The primary Release
artifact's eight checks cover selected-runtime loading, reference/verification,
ISA and installed consumers; they do not repeat its entire earlier arithmetic
suite. Production benchmark outputs receive exact reference audits. The legacy
public instrumented processes cross-match contracts and output checksums to the
validated production observations; the new cache-equipped collector additionally
compares every instrumented native output element against its full reference.

CPU ASan uses LLVM23.1.4, `/MD`, optimization level O1 and debug symbols, with
HIP and fuzzers OFF and GMP/FLINT ON. LLVM rejects the first attempted Debug-CRT
`/MDd` configuration; that failed build receipt is preserved, and the supported
RelWithDebInfo configuration passes. This does not qualify Debug-CRT ASan or
the separate static-CRT libFuzzer configuration.

The follow-up totals **2,480 passes, 96 skips and zero failures in the supported
final configurations**. Loaded HIP/core-loader hashes and sampled hipBLASLt and
compiler DLL paths match the selected SDK. The new performance observations are
documented in [performance evidence](performance.md#october-26-sdk101-bounded-measurements).
