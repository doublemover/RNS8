# Windows HIP SDK Platform Notes

The local bring-up target is Windows on Radeon RX 7900 XTX / `gfx1100`.

Recorded Windows validation coverage:

- `tools/check_dependencies.py` detects hipcc, hipInfo, hipconfig, MSVC, vcpkg,
  Python packages, Radeon CLI tools, and optional accelerator headers/libraries.
- The dependency checker now reports spec-aligned readiness gates:
  - `E001` host compiler/reference readiness.
  - `E002` Windows HIP SDK detection.
  - `E004` GPU architecture detection.
  - `E070` Windows RDNA3 direct HIP readiness for the local `gfx1100` bring-up
    target.
- Its process exit status follows host-required readiness gates. Optional
  reference/planned manifest packages and accelerator probes are reported as
  evidence, not as correctness or Windows bring-up blockers.
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
  K-block splitting, public bounded signed/unsigned API paths, exact-wide
  signed/unsigned RNS output and limb export, and the public strict wrap64
  byte-limb path against CPU references.

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

Windows `gfx1100` evidence does not validate Linux ROCm, Instinct CDNA, or
cluster production readiness. Those gates remain represented in presets and
readiness reports, but they require a supported Linux ROCm host and actual
target hardware.

hipBLASLt, CK, rocWMMA, and AMDGPU builtin paths remain feature-detected
accelerators on Windows. `tools/check_dependencies.py` may report discovered
headers or libraries as candidate evidence, but it does not enable or validate
those backends. They require compiled capability probes and exact CPU
differential tests before any backend can be treated as ready. The CMake
accelerator enable flags intentionally fail fast until real correctness
backends exist; discovery evidence is not a bypass around the
direct HIP correctness path.

Opt-in accelerator evidence probes are available without changing backend
selection:

```powershell
python tools\check_dependencies.py --accelerator-probes --json
cmake --preset windows-msvc-hip-accelerator-probe
```

Both paths are evidence-only. The Python probe writes tiny sources and binaries
under `temp\accelerator-probes\`, records compile/link/runtime status, and keeps
`backend_enablement=disabled`. The CMake probe preset sets
`RNS8_PROBE_ACCELERATORS=ON` while keeping `RNS8_ENABLE_HIPBLASLT`,
`RNS8_ENABLE_CK`, and `RNS8_ENABLE_ROCWMMA` off. On the current Windows HIP SDK
install, hipBLASLt is candidate evidence through headers and a
`libhipblaslt.dll.a` archive, but the opt-in hipcc/lld-link probe does not find
a linkable `hipblaslt.lib`; this is not a correctness blocker.
