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
| Public storage rejection | Exact-wide exports reject stale-prefix RNS matrices, bounded RNS matrices, wrap64 byte-limb matrices, and signed/unsigned cross-export calls. None of those handles are alternate routes into exact-wide limbs. |
| Direct HIP currentness | Exact-wide direct-HIP export requires device-current resident RNS output. Host-current stale device residues are rejected; export does not perform an implicit hot-path upload. |
| Accelerator enablement | hipBLASLt, CK, rocWMMA, and AMDGPU builtin enable flags intentionally fail fast until real correctness backends exist. Probes collect candidate evidence only. CTest negative configure cases pin the fail-fast message for each enable flag. |
| Linux ROCm and Instinct | Linux ROCm, Radeon Linux, and Instinct CDNA gates are represented by presets, target metadata, and dependency reports. Windows evidence does not validate them; they require a real Linux ROCm host with supported hardware. |

## Accelerator Gates

- `RNS8_ENABLE_HIPBLASLT`, `RNS8_ENABLE_CK`, `RNS8_ENABLE_ROCWMMA`, and
  `RNS8_ENABLE_AMDGPU_BUILTINS` intentionally fail fast until real correctness
  backends exist.
- The CTest suite registers one configure-negative test per enable flag. Each
  scratch configure must fail before any placeholder backend is generated and
  must direct users to evidence-only probes instead.
- `RNS8_PROBE_ACCELERATORS=ON` and
  `tools/check_dependencies.py --accelerator-probes` collect evidence only.
  They never enable backends and never satisfy correctness.
- The dependency checker's JSON readiness object includes
  `accelerator_enablement`, whose per-flag records keep
  `backend_enablement=disabled`, `correctness_backend=not_implemented`,
  `validated_correctness_backend=false`, `can_enable_correctness_backend=false`,
  and `enable_flags_fail_fast=true` until a real exact correctness backend
  exists.
- The same JSON readiness object includes
  `exact_wide_platform_validation`. On this Windows bring-up host it records
  Windows `gfx1100` exact-wide evidence scope only, sets
  `windows_evidence_validates_linux_rocm=false` and
  `windows_evidence_validates_instinct=false`, and keeps Linux ROCm and
  Instinct validation false until a real supported Linux ROCm host runs exact
  CPU differentials.
- hipBLASLt, CK, and rocWMMA probes may report discovered files or tiny
  compile/run probe status.
- AMDGPU builtins have no discovery-only readiness path; they remain not ready
  until target-specific exact kernels, CPU differentials, and ISA evidence
  exist.
