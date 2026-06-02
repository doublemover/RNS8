# Linux ROCm Platform Notes

Linux ROCm remains the production, profiling, multi-GPU, and Instinct
validation path. The scaffold keeps Linux ROCm presets and toolchain variables
represented, but Windows validation does not validate Linux ROCm, Instinct
CDNA, profiling, power, or cluster production gates.

`tools/check_dependencies.py` reports Linux readiness separately from Windows
readiness. On non-Linux hosts, `E003` Linux ROCm detection and the Linux platform
matrix gates are reported as not applicable rather than as Windows blockers.
On Linux, `hipcc`, `hipconfig`, `rocminfo`, and either `rocm-smi` or `amd-smi`
are treated as the ROCm capability-inspection command set.

Expected Linux configure path:

```bash
cmake --preset linux-rocm-debug
cmake --build --preset linux-debug
ctest --preset linux-debug --output-on-failure
```

The Linux preset keeps two target lists separate:

- `RNS8_AMDGPU_TARGETS` is the active offload list used by explicit HIP
  compilation. It should be set to targets supported by the active ROCm release
  and the validation machine.
- `RNS8_ROCM_COVERAGE_TARGETS` is source-level readiness metadata. It records
  the documented RDNA2/RDNA3/RDNA4 and CDNA2/CDNA3/CDNA4 coverage families:
  `gfx1030`, `gfx1100`, `gfx1200`, `gfx1201`, `gfx90a`, `gfx942`, and
  `gfx950`. It does not add compiler offload architectures by itself.

Linux-specific accelerator paths are intentionally not required for core
correctness:

- hipBLASLt INT8 GEMM remains a later feature-detected backend.
- CK grouped/fused kernels remain a later feature-detected backend.
- rocWMMA and AMDGPU builtins remain target-specific hot paths.

The checker and CMake find modules report hipBLASLt, CK, and rocWMMA discovery
as candidate evidence only. The hipBLASLt shallow probe looks for headers,
libraries, and the optional `hipblaslt-bench` utility; CK and rocWMMA shallow
probes look for their headers. Optional compiled evidence can be requested with:

```bash
python tools/check_dependencies.py --accelerator-probes --json
cmake --preset linux-rocm-accelerator-probe
```

These probes write scratch evidence under `temp/` or configure an evidence-only
CMake preset. They keep `RNS8_ENABLE_HIPBLASLT`, `RNS8_ENABLE_CK`, and
`RNS8_ENABLE_ROCWMMA` disabled. Linux production readiness still requires
target-supported components, compiled capability probes, exact CPU differential
coverage, and measured performance before enabling advanced backend stages.
AMDGPU builtin hot kernels follow the same rule: compiler or architecture
availability is only candidate evidence until a target-specific kernel has exact
CPU differential coverage.

Before claiming Linux production readiness, run direct HIP parity tests on the
target ROCm release and actual supported Radeon or Instinct hardware. Until
that happens, the Linux and Instinct entries are represented roadmap targets,
not validated substitutes for the Windows `gfx1100` direct HIP evidence.
