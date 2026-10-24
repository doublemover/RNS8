# Windows HIP SDK setup

Use this guide for native Windows Radeon builds. The [dependency guide](../third_party/README.md)
owns common requirements and pins. The [readiness reference](platform-readiness.md)
separates installed tools from compiled and device-validated paths.

## Prerequisites

Install a GPU/driver combination supported by the selected
[Windows HIP SDK release](https://rocm.docs.amd.com/projects/install-on-windows/en/docs-7.1.1/reference/system-requirements.html)
and the [HIP SDK development components](https://rocm.docs.amd.com/projects/install-on-windows/en/docs-7.1.1/how-to/install.html).
Install [Visual Studio's C++ desktop workload and Windows SDK](https://learn.microsoft.com/en-us/cpp/build/vscpp-step-0-installation),
CMake, Ninja, Git, Python and vcpkg dependencies as described in the dependency
guide. The measured local stack uses VS2026 and HIP 7.1; it includes the
build-local LLVM math-header fix described below.

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

Point the current preset at the real SDK and vcpkg installations. These examples
intentionally use nondefault roots with spaces:

```powershell
$env:VCPKG_ROOT = 'D:/Development Tools/vcpkg'
$rns8HipRoot = 'D:/AMD SDK/ROCm/7.1'
python tools/windows_dev.py cmake --preset windows-msvc-hip-debug "-DRNS8_HIP_ROOT=$rns8HipRoot" "-DCMAKE_TOOLCHAIN_FILE=$env:VCPKG_ROOT/scripts/buildsystems/vcpkg.cmake" -DRNS8_AMDGPU_TARGETS=gfx1100
python tools/windows_dev.py cmake --build --preset windows-debug --parallel 4
.\build\windows-msvc-hip-debug\rns8-bench.exe --backend hip-direct --semantics bounded-i64 --m 32 --n 32 --k 64 --warmups 1 --repeats 3 --seed 3135
```

`gfx1100` selects the RX 7900 XTX target. For a different GPU, first check AMD's
release matrix and set the matching target. The benchmark verifies its seeded
integer result and emits JSON. This small checked workload is a first execution;
[qualification](gpu-qualification.md) covers a broader campaign.

The preset currently contains `C:/vcpkg` and
`C:/Program Files/AMD/ROCm/7.1` defaults. Explicit overrides above take precedence;
[#57](https://github.com/doublemover/RNS8/issues/57) tracks broader autodetection.
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
