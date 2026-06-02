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
- Direct HIP signed and unsigned residue packing compared against CPU reference
  residue storage, including full-width boundary values and padded leading
  dimensions.
- A direct HIP one-modulus ring-GEMM smoke test compared against CPU reference
  on `gfx1100` when HIP is enabled and a device is visible.
- Direct HIP K-block splitting above 65536, including public bounded signed and
  unsigned API smoke tests against the CPU reference.

Not yet implemented:

- Per-tile adaptive bounds.
- GPU CRT reconstruction.
- Exact-wide output.
- Strict `mod 2^64` byte-limb GEMM.
- Backend signedness corrections for unsigned byte-limb wraparound.

Do not treat the current direct HIP kernel as performance evidence. It is a
minimal correctness proof for the Windows HIP compile/run path.
