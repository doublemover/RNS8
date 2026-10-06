# Changelog

All notable first-party RNS8 changes are tracked here. RNS8 is pre-1.0, so
public APIs may change between minor releases when the change improves semantic
clarity or correctness.

## 0.1.0 - Unreleased

### October 2026 correctness and qualification audit

- Repairs bounded CRT/export, persistent K-block accumulation, packing plane
  tails, and AMDGPU skinny/finite dispatch; removes unsafe unused prototypes.
- Fixes the 32-bit C++ enum ABI representation so malformed C/FFI values can be
  rejected without undefined behavior. Invalid-value sanitizer tests remain on.
- Adds CPU/host-arithmetic and GPU regressions, bounded MI300X/gfx942 and
  RX 7900 XTX/gfx1100 qualification manifests, and honest progress/evidence tools.
- Separates historical performance claims from current documentation and changes
  runtime implementation keys so old caches do not qualify repaired source.
- GPU compilation, hardware execution and fresh performance qualification remain
  pending. See [the audit](docs/audit-2026-10-06.md).

### Initial implementation

- Establishes the public C ABI, limited C++ RAII wrapper, CPU reference backend,
  and CPU-only CMake package/export path.
- Adds explicit semantic modes for bounded i64/u64, exact-wide limb export,
  strict wrap64, finite u8 rings, and finite u8 fields.
- Adds Windows HIP direct, native vector-ALU, hipBLASLt, CK, and rocWMMA
  bring-up paths with opt-in accelerator validation boundaries.
- Adds public plan introspection for grouped-dispatch descriptor and lifetime
  contracts plus narrow Direct-HIP resident grouped GEMM entry points; grouped
  pack/export remains explicit benchmark or caller work.
- Publishes benchmark schema v4, result comparison tooling, autotune-cache
  validation, and reviewed-evidence policy.
- Normalizes first-party metadata to MIT.
- Hard-cuts public backend spelling from `wmma` to `rocwmma`.
