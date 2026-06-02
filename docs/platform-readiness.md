# Platform Readiness

This file describes readiness policy, not a fresh validation record.

## Host Gates

- Windows Radeon `gfx1100` remains the local direct-HIP bring-up target.
- Windows readiness requires host/reference tooling, Windows HIP SDK detection,
  GPU architecture detection, and the Windows RDNA3 direct-HIP gate.
- Windows evidence does not validate Linux ROCm, Instinct CDNA, profiling,
  power, or cluster gates.
- Linux ROCm and Instinct gates are represented in presets and dependency
  reports, but they are unvalidated on Windows. They require a real Linux ROCm
  host and actual supported Radeon or Instinct hardware.

## Exact-Wide ABI Readiness

- `RNS8_EXACT_WIDE_SIGNED` and `RNS8_EXACT_WIDE_UNSIGNED` use
  `RNS8_BOUND_NONE`, `bound = 0`, persistent RNS storage, and explicit limb
  export.
- `ld` is a leading dimension in matrix elements, not limbs.
- `limb_count` is the fixed output width and must be in `[1, 32]`.
- Signed export emits fixed-width little-endian two's-complement limbs for the
  centered exact integer.
- Unsigned export emits fixed-width little-endian magnitude limbs for the
  canonical nonnegative integer.
- Too-small widths return `RNS8_RANGE_ERROR`; invalid width, stride, or null
  pointers return `RNS8_INVALID_ARGUMENT`.
- Exact-wide export is not bounded `i64/u64` export and is not strict
  `mod 2^64` wraparound.

## Accelerator Gates

- `RNS8_ENABLE_HIPBLASLT`, `RNS8_ENABLE_CK`, `RNS8_ENABLE_ROCWMMA`, and
  `RNS8_ENABLE_AMDGPU_BUILTINS` intentionally fail fast until real correctness
  backends exist.
- `RNS8_PROBE_ACCELERATORS=ON` and
  `tools/check_dependencies.py --accelerator-probes` collect evidence only.
  They never enable backends and never satisfy correctness.
- hipBLASLt, CK, and rocWMMA probes may report discovered files or tiny
  compile/run probe status.
- AMDGPU builtins have no discovery-only readiness path; they remain not ready
  until target-specific exact kernels, CPU differentials, and ISA evidence
  exist.
