# Strict Wraparound Backend

Reserved for strict `mod 2^64` byte-limb GEMM.

This path must use unsigned byte-limb semantics. It must not reuse odd-modulus
CRT as a wraparound substitute unless an explicit exact bound is supplied by a
bounded semantic contract.

Current status:

- `RNS8_BACKEND_WRAP64_BYTE_LIMB` is implemented as a CPU reference context for
  `rns8_gemm_wrap_u64_oneshot`.
- `RNS8_WRAP_U64_MOD_2_64` remains unsupported in plan and matrix creation
  until persistent byte-limb storage exists.
- `wrap64_reference.cpp` contains the CPU byte-limb Comba reference for
  low-64-bit product and GEMM-cell behavior. It keeps strict wraparound
  arithmetic separate from the odd-modulus CRT path and backs the public
  one-shot CPU backend.
- Public pack/export plumbing, real byte-limb GPU kernels, and GPU differential
  tests are not implemented yet.
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
