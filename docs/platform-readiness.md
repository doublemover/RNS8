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
  centered exact integer. The centered threshold is `x >= ceil(P / 2)` for the
  selected modulus product `P`, so an even-product half residue class is
  represented as negative.
- Unsigned export emits fixed-width little-endian magnitude limbs for the
  canonical nonnegative integer.
- Too-small widths return `RNS8_RANGE_ERROR`; invalid width, stride, or null
  pointers return `RNS8_INVALID_ARGUMENT`.
- CPU export stages all cells before writing the caller's host layout; direct
  HIP export checks the device status before copying compact output back. In
  both cases, range errors preserve every destination limb.
- Direct HIP exact-wide export requires device-current, device-resident RNS
  output. Host-current stale device residues are malformed for this export
  surface and are rejected instead of being copied implicitly.
- Exact-wide export is not bounded `i64/u64` export and is not strict
  `mod 2^64` wraparound.

## Requirement Audit

| Requirement | Current contract |
|---|---|
| Explicit bound-none exact-wide descriptors | Exact-wide plan descriptors require `RNS8_BOUND_NONE`, `bound = 0`, and no `tile_bounds` pointer or count. Exact-wide matrix descriptors also require `RNS8_BOUND_NONE`. Stale bound metadata is malformed input and returns `RNS8_INVALID_ARGUMENT`, not an accelerator or alternate route. |
| Fixed-width signed limbs | Signed export reconstructs the centered exact integer with the `x >= ceil(P / 2)` negative threshold and emits exactly `limb_count` little-endian two's-complement limbs. Too-small widths return `RNS8_RANGE_ERROR`; no low-limb truncation or wrap interpretation is accepted. |
| Fixed-width unsigned limbs | Unsigned export reconstructs the canonical nonnegative integer and emits exactly `limb_count` little-endian magnitude limbs. Too-small widths return `RNS8_RANGE_ERROR`; no low-limb truncation or wrap interpretation is accepted. |
| Public export ABI | `ld` is an element stride, `limb_count` must be in `[1, 32]`, null `ctx`, `plan`, `matrix`, or `dst` arguments are invalid API calls, and range-error exports preserve the caller's destination. |
| Direct HIP currentness | Exact-wide direct-HIP export requires device-current resident RNS output. Host-current stale device residues are rejected; export does not perform an implicit hot-path upload. |
| Accelerator enablement | hipBLASLt, CK, rocWMMA, and AMDGPU builtin enable flags intentionally fail fast until real correctness backends exist. Probes collect candidate evidence only. |
| Linux ROCm and Instinct | Linux ROCm, Radeon Linux, and Instinct CDNA gates are represented by presets, target metadata, and dependency reports. Windows evidence does not validate them; they require a real Linux ROCm host with supported hardware. |

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
