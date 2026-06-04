# Windows HIP SDK Platform Notes

Windows on Radeon RX 7900 XTX / `gfx1100` is the local bring-up path. Linux
ROCm remains the production, profiling, multi-GPU, and Instinct validation
path; Windows evidence does not validate those targets.

## Requirements

- Windows 11 x86-64.
- AMD Radeon GPU supported by the Windows HIP SDK.
- AMD HIP SDK for Windows.
- Visual Studio 2022 Build Tools or Community with MSVC C++ and Windows SDK.
- CMake, Ninja, Git, Python 3.11 or newer.
- vcpkg at `C:\vcpkg` or `VCPKG_ROOT` with packages from `vcpkg.json`.
- Optional Radeon Developer Tool Suite CLI utilities for ISA/profiling work.

Use `tools\check_dependencies.py` for the authoritative local readiness report:

```powershell
python tools\check_dependencies.py
```

The checker reports host tools, HIP SDK/device metadata, vcpkg packages, Python
packages, preset coverage, optional accelerator discovery, and validation
boundaries. Discovery and probes do not enable correctness backends by
themselves.

## Build And Test

Use the Visual Studio wrapper from a plain PowerShell shell:

```powershell
python tools\windows_dev.py cmake --preset windows-msvc-hip-debug
python tools\windows_dev.py cmake --build --preset windows-debug
python tools\windows_dev.py ctest --preset windows-debug --output-on-failure
build\windows-msvc-hip-debug\rns8-inspect.exe --backend hip-direct --json
build\windows-msvc-hip-debug\rns8-verify.exe --hip-smoke
```

The Windows build uses explicit HIP SDK compiler integration and does not rely
on CMake `enable_language(HIP)`. The default local offload target is `gfx1100`.

CPU-only scaffold validation remains available:

```powershell
cmake --preset cpu-debug
cmake --build --preset cpu-debug
ctest --preset cpu-debug --output-on-failure
```

## Optional Accelerators

hipBLASLt, CK, and rocWMMA are opt-in Windows `gfx1100` correctness backends.
They are not required for CPU or direct-HIP correctness. Dedicated presets
enable and test each one:

```powershell
python tools\windows_dev.py cmake --preset windows-msvc-hipblaslt-debug
python tools\windows_dev.py cmake --build --preset windows-hipblaslt-debug
python tools\windows_dev.py ctest --preset windows-hipblaslt-debug --output-on-failure

python tools\windows_dev.py cmake --preset windows-msvc-ck-debug
python tools\windows_dev.py cmake --build --preset windows-ck-debug
python tools\windows_dev.py ctest --preset windows-ck-debug --output-on-failure

python tools\windows_dev.py cmake --preset windows-msvc-rocwmma-debug
python tools\windows_dev.py cmake --build --preset windows-rocwmma-debug
python tools\windows_dev.py ctest --preset windows-rocwmma-debug --output-on-failure
```

Dependency-only probes are separate from backend validation:

```powershell
python tools\bootstrap_rocm_accelerators.py --init --probe --target gfx1100
python tools\check_dependencies.py --accelerator-probes --json
python tools\windows_dev.py cmake --preset windows-msvc-hip-accelerator-probe
```

Probe artifacts belong under ignored `temp/`, `build/`, or `out/` paths. Do
not clone CK or rocWMMA under `C:\`; use the pinned repo-local submodules.
