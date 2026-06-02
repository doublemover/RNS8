# Windows HIP SDK Platform Notes

The local bring-up target is Windows on Radeon RX 7900 XTX / `gfx1100`.

Validated in this slice:

- `tools/check_dependencies.py` detects hipcc, hipInfo, hipconfig, MSVC, vcpkg,
  Python packages, Radeon CLI tools, and optional accelerator headers/libraries.
- `cmake --preset windows-msvc-hip-debug` configures with vcpkg and
  `RNS8_ENABLE_HIP=ON`.
- HIP sources are compiled by explicit hipcc custom commands. The build does
  not call `enable_language(HIP)`.
- The explicit HIP compile passes `--offload-arch=gfx1100` by default.
- Host HIP runtime code defines `__HIP_PLATFORM_AMD__` and links against
  `amdhip64`.
- Debug builds pass the MSVC debug runtime settings through hipcc to avoid CRT
  and iterator-debug-level mismatches.
- `rns8-verify --hip-smoke` exercises direct HIP residue conversion, ring GEMM,
  K-block splitting, and public bounded signed/unsigned API paths against the
  CPU reference.

Current proof command:

```powershell
cmake --preset windows-msvc-hip-debug
cmake --build --preset windows-debug
ctest --preset windows-debug --output-on-failure
build\windows-msvc-hip-debug\rns8-inspect.exe --backend hip-direct --json
build\windows-msvc-hip-debug\rns8-verify.exe --hip-smoke
```

The current HIP kernel is a correctness bring-up kernel, not an optimized
matrix-engine implementation.
