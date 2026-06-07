# Changelog

All notable first-party RNS8 changes are tracked here. RNS8 is pre-1.0, so
public APIs may change between minor releases when the change improves semantic
clarity or correctness.

## 0.1.0 - Unreleased

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
