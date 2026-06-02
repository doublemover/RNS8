# Strict Wraparound Backend

Reserved for strict `mod 2^64` byte-limb GEMM.

This path must use unsigned byte-limb semantics. It must not reuse odd-modulus
CRT as a wraparound substitute unless an explicit exact bound is supplied by a
bounded semantic contract.

Current status:

- `RNS8_WRAP_U64_MOD_2_64` is unsupported in plan and matrix creation.
- The `RNS8_BACKEND_WRAP64_BYTE_LIMB` backend kind is reserved and unsupported
  until public pack/export plumbing, real byte-limb GPU kernels, and GPU
  differential tests exist.
- `wrap64_reference.cpp` contains an internal CPU byte-limb Comba reference for
  low-64-bit product and GEMM-cell behavior. It exists to keep strict
  wraparound arithmetic separate from the odd-modulus CRT path; it is not a
  public backend yet.
- Bounded `RNS8_BOUNDED_U64` calls are exact-result calls, not wraparound
  calls. They may use odd-modulus CRT only when the exact mathematical output is
  recoverable inside the caller-supplied bound.

Acceptance bar for enabling this backend:

- Pack inputs as unsigned base-256 limbs.
- Compute the 36 low-product byte GEMMs needed for the low 64 output bits.
- Accumulate Comba diagonals with deterministic carry propagation.
- Test unsigned byte signedness handling explicitly when a selected accelerator
  exposes only signed INT8 GEMM.
- Compare minimal GPU smoke/correctness tests against a CPU reference before
  any performance claim.
