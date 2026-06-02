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
- Its process exit status follows host-required readiness gates plus internal
  hard-cut self-check consistency. Optional reference/planned manifest packages
  and accelerator probes are reported as evidence, not as correctness or
  Windows bring-up blockers.
- The JSON report's `readiness.correctness_backend_validation` section records
  that the dependency checker validates no correctness backend by itself. It
  distinguishes implemented CPU/direct-HIP/wrap64 backend families from
  candidate accelerator evidence, whose records keep
  `candidate_evidence_is_correctness_validation=false`.
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
python tools\windows_dev.py cmake --preset windows-msvc-hip-debug
python tools\windows_dev.py cmake --build --preset windows-debug
python tools\windows_dev.py ctest --preset windows-debug --output-on-failure
build\windows-msvc-hip-debug\rns8-inspect.exe --backend hip-direct --json
build\windows-msvc-hip-debug\rns8-verify.exe --hip-smoke
```

`rns8-inspect --backend` is explicit: unknown backend strings are invalid.
With the default Windows HIP preset, accelerator names such as `hipblaslt`,
`ck`, and `rocwmma` report unsupported status and are not routed to `auto`.
With the opt-in `windows-msvc-hipblaslt-debug` preset, `hipblaslt` reports the
implemented baseline backend on the local device while `ck` and `rocwmma`
remain unsupported.

The current HIP kernel is a correctness bring-up kernel, not an optimized
matrix-engine implementation.

Windows `gfx1100` evidence does not validate Linux ROCm, Instinct CDNA, or
cluster production readiness. Those gates remain represented in presets and
readiness reports, but they require a supported Linux ROCm host and actual
target hardware. Exact-wide signed/unsigned limb ABI coverage on Windows is
host and `gfx1100` evidence only. Direct-HIP exact-wide export on Windows uses
device-current resident RNS output and rejects host-current stale device
residues; that local contract still does not stand in for Linux Radeon or
Instinct exact-wide validation.
The report also emits `hard_cut_self_checks` to keep those boundaries
machine-readable: accelerator evidence must not enable backends, and Windows
host evidence must not be promoted to Linux ROCm or Instinct validation.

hipBLASLt, CK, rocWMMA, and AMDGPU builtin paths remain feature-detected
accelerators on Windows. `tools/check_dependencies.py` may report discovered
headers or libraries as candidate evidence for component-backed accelerators,
and reports AMDGPU builtin readiness as not ready until target-specific exact
kernels exist. Discovery does not enable or validate a backend by itself.
hipBLASLt, CK, and rocWMMA are current opt-in correctness backends:
`RNS8_ENABLE_HIPBLASLT=ON` builds a real INT8-to-INT32 scratch-and-reduce
backend when HIP and hipBLASLt are present, `RNS8_ENABLE_CK=ON` builds the CK
fused backend, and `RNS8_ENABLE_ROCWMMA=ON` builds the rocWMMA fused backend.
Their dedicated Windows presets run exact CPU/direct-HIP differentials.
AMDGPU builtin enablement intentionally fails fast until real target-specific
correctness kernels exist.

Opt-in accelerator evidence probes are available without changing backend
selection:

```powershell
python tools\bootstrap_rocm_accelerators.py --init --probe --target gfx1100
python tools\check_dependencies.py --accelerator-probes --json
python tools\windows_dev.py cmake --preset windows-msvc-hip-accelerator-probe
```

These paths are evidence-only. `bootstrap_rocm_accelerators.py` initializes the
pinned repo-local CK and rocWMMA submodules under `third_party\rocm\`, compiles
their tiny `gfx1100` HIP dependency probes, and writes its record under
`temp\accelerator-deps\`. `check_dependencies.py --accelerator-probes` writes
tiny sources and binaries under `temp\accelerator-deps\`, loads the Visual
Studio developer environment automatically for Windows MSVC link probes,
records compile/link/runtime status, and keeps
`backend_enablement=disabled`; for AMDGPU builtins it records
`NOT_RUN_NO_CORRECTNESS_KERNEL` until a real target-specific exact kernel
exists. The CMake probe preset sets
`RNS8_PROBE_ACCELERATORS=ON` while keeping `RNS8_ENABLE_HIPBLASLT`,
`RNS8_ENABLE_CK`, `RNS8_ENABLE_ROCWMMA`, and
`RNS8_ENABLE_AMDGPU_BUILTINS` off. For the real hipBLASLt baseline backend use:

```powershell
python tools\windows_dev.py cmake --preset windows-msvc-hipblaslt-debug
python tools\windows_dev.py cmake --build --preset windows-hipblaslt-debug
python tools\windows_dev.py ctest --preset windows-hipblaslt-debug --output-on-failure
```

On the current Windows HIP SDK install,
hipBLASLt is candidate evidence through AMD's `roc::hipblaslt` CMake target,
headers, `libhipblaslt.dll.a` import archive, and `libhipblaslt.dll` runtime.
No separate MSVC `hipblaslt.lib` is required. The opt-in CMake MSVC link probe
passes when configure is run through `tools\windows_dev.py` or from an already
initialized Visual Studio developer environment; from a direct plain
PowerShell `cmake` invocation it reports `not_run_missing_msvc_environment`
instead of treating missing MSVC standard-library include paths as a hipBLASLt
failure.
