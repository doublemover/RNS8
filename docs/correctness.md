# Correctness Notes

Implemented correctness coverage:

- Default ladder stability and pairwise-coprime validation.
- Prefix range-bit checks against the research spec table.
- Centered residue conversion for `m = 256`, composite odd moduli, prime
  moduli, negative inputs, and full signed input boundaries.
- Scalar ring GEMM over composite and prime moduli.
- K-block splitting above 65536 to avoid signed INT32 accumulation overflow.
- Boost.Multiprecision CRT/Garner reconstruction for bounded signed and
  unsigned outputs.
- Range errors when selected modulus prefixes cannot satisfy supplied bounds.
- Bounded signed and unsigned one-shot GEMM boundary tests, including
  `INT64_MAX`, `-INT64_MAX`, and `UINT64_MAX` outputs.
- Public bounded signed and unsigned CPU one-shot GEMM sweeps over all
  dimensions 1 through 8, with Boost.Multiprecision exact oracles.
- Fixed-seed random bounded signed and unsigned CPU checks with padded leading
  dimensions and Boost.Multiprecision exact oracles.
- Worst-case positive, negative, and unsigned accumulation checks at and just
  above the 65536 K-block split point.
- Semantic guard tests that bounded APIs reject `RNS8_BOUND_NONE`, exact-wide
  rejects bounded-looking metadata, and finite-ring, finite-field, strict
  wraparound, and future accelerator backend requests report unsupported
  instead of falling through to bounded CRT behavior.
- Negative semantic tests that exact-wide signed/unsigned and strict
  `mod 2^64` wraparound reject bounded-looking metadata, including explicit
  global bounds and input-range bounds. A bounded prefix alone is not a license
  to reinterpret these contracts as current odd-modulus CRT.
- Exact-wide signed and unsigned RNS-output tests for CPU and direct HIP,
  including full-width 64-bit inputs that are compared against
  Boost.Multiprecision residue oracles.
- Exact-wide signed and unsigned CPU limb export tests. Signed export is
  fixed-width little-endian two's-complement, unsigned export is fixed-width
  little-endian magnitude, and both report range errors when too few limbs are
  supplied.
- Internal strict `mod 2^64` byte-limb product and GEMM-cell reference tests
  compared against Boost.Multiprecision low-64-bit results. The public
  wraparound backend remains unsupported.
- Direct HIP signed and unsigned residue packing compared against CPU reference
  residue storage, including full-width boundary values and padded leading
  dimensions.
- A direct HIP one-modulus ring-GEMM smoke test compared against CPU reference
  on `gfx1100` when HIP is enabled and a device is visible.
- Direct HIP device-resident RNS matrices, K-block splitting above 65536, fused
  INT32-to-centered-residue reduction without INT32 global output, and bounded
  signed/unsigned GPU CRT export smoke tests against the CPU reference.

Not yet implemented:

- Per-tile adaptive bounds.
- Exact-wide GPU reconstruction.
- Bounded GPU export prefixes wider than the current direct HIP 128-bit Garner
  path.
- Public strict `mod 2^64` byte-limb GEMM backend and GPU kernels.
- Backend signedness corrections for unsigned byte-limb wraparound.

Semantic guardrail:

- `RNS8_BOUNDED_I64` and `RNS8_BOUNDED_U64` are exact-result contracts. The
  caller-supplied bound is part of that contract, and the current CPU and direct
  HIP paths use odd-modulus CRT reconstruction only for results recoverable
  inside the stated range.
- `RNS8_EXACT_WIDE_SIGNED` and `RNS8_EXACT_WIDE_UNSIGNED` are not aliases for
  bounded 64-bit export with a larger prefix. They support persistent RNS output
  with `RNS8_BOUND_NONE` and explicit CPU little-endian limb export. GPU
  exact-wide export remains a separate unsupported milestone.
- `RNS8_WRAP_U64_MOD_2_64` is not implemented by the odd-modulus CRT ladder.
  Strict low-64-bit wraparound requires the byte-limb backend so unsigned byte
  semantics, Comba accumulation, carry handling, and low-limb export are tested
  directly. A bounded API call is only valid for wrap-like inputs when the exact
  mathematical result is also within the supplied bounded contract.

Do not treat the current direct HIP kernel as performance evidence. It is a
minimal correctness proof for the Windows HIP compile/run path.
