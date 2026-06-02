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
- Plan schedule inspection for output tile grid, exact required prefix,
  selected prefix, and prefix-group metadata. Global bounded plans use a fixed
  selected prefix for every tile. CPU reference and direct HIP per-tile bounded
  plans copy the caller's tile bounds at plan creation, select the minimum
  exact prefix per tile, report adaptive prefix/skip metadata, execute only the
  selected per-tile prefixes, and export with the tile-local bound.
- Bounded signed and unsigned one-shot GEMM boundary tests, including
  `INT64_MAX`, `INT64_MIN`, and `UINT64_MAX` outputs.
- Public bounded signed and unsigned CPU one-shot GEMM sweeps over all
  dimensions 1 through 8, with Boost.Multiprecision exact oracles.
- Fixed-seed random bounded signed and unsigned CPU checks with padded leading
  dimensions and Boost.Multiprecision exact oracles.
- Worst-case positive, negative, and unsigned accumulation checks at and just
  above the 65536 K-block split point.
- Semantic guard tests that bounded APIs reject `RNS8_BOUND_NONE`, exact-wide
  rejects bounded-looking metadata, finite-ring/finite-field/future accelerator
  requests report unsupported, and strict wraparound never falls through to
  bounded CRT behavior.
- Negative semantic tests that exact-wide signed/unsigned and strict
  `mod 2^64` wraparound reject bounded-looking metadata, including explicit
  global bounds and input-range bounds. A bounded prefix alone is not a license
  to reinterpret these contracts as current odd-modulus CRT.
- Exact-wide signed and unsigned RNS-output tests for CPU and direct HIP,
  including full-width 64-bit inputs that are compared against
  Boost.Multiprecision residue oracles.
- Exact-wide signed and unsigned CPU and direct HIP limb export tests. Signed
  export is fixed-width little-endian two's-complement, unsigned export is
  fixed-width little-endian magnitude, `ld` is an element stride rather than a
  limb stride, both report range errors when too few limbs are supplied, and
  direct HIP export leaves device-resident residues on device instead of
  synchronizing host residue storage.
- Strict `mod 2^64` byte-limb product, GEMM-cell, public CPU one-shot, and
  persistent byte-limb matrix tests compared against Boost.Multiprecision
  low-64-bit results. The public wrap path requires explicit wrap64 semantics
  and byte-limb storage, uses separate pack/GEMM/export APIs for persistent
  matrices, and rejects CRT bounds/prefixes.
- Unsigned byte-limb signedness correction tests cover every byte pair and
  verify that the signed-INT8 correction algebra composes through Comba
  diagonals. This is readiness coverage for future signed-INT8 accelerator use,
  not an enabled accelerator backend.
- A separate CPU 36-byte-GEMM oracle sums the low-product byte diagonals with
  the signed-INT8 correction helper, performs Comba carry propagation, and is
  compared against both Boost.Multiprecision low-64-bit results and the existing
  byte-limb Comba GEMM-cell reference.
- Public direct HIP strict `mod 2^64` byte-limb one-shot and persistent API
  tests compared against the CPU byte-limb backend. HIP wrap matrices own
  device-resident byte-limb buffers, do not allocate RNS residues, preserve
  device pointer stability through pack/GEMM/export, and support padded host
  leading dimensions on export. The HIP GEMM correctness kernel sums the 36
  low-product byte diagonals with device-side signed-INT8 correction algebra and
  then performs deterministic carry propagation into the low 64 bits.
- Direct HIP signed and unsigned residue packing compared against CPU reference
  residue storage, including full-width boundary values and padded leading
  dimensions.
- Direct HIP one-modulus ring-GEMM smoke tests compared against CPU reference
  on `gfx1100` when HIP is enabled and a device is visible, including a
  centered-correction boundary case for negative, positive-threshold, and
  near-zero residues.
- Direct HIP device-resident RNS matrices, K-block splitting above 65536, fused
  INT32-to-centered-residue reduction without INT32 global output, and bounded
  signed/unsigned GPU CRT export smoke tests through prefix 20 against the CPU
  reference.
- Direct HIP per-tile bounded signed/unsigned GEMM tests compare output against
  the CPU reference, cover tile-local range errors, padded host export layouts,
  schedule parity, and verify skipped residue planes above each tile's selected
  prefix remain untouched on device.
- Benchmark schema v4 captures direct-HIP adaptive per-tile bounded runs with
  exact seeded-input tile-bound prepass metadata, selected tiled kernel name,
  adaptive execution flags, and aggregate HIP event timing scope. This is
  benchmark evidence metadata for the correctness path, not an optimized GPU
  performance claim.
- Private direct HIP strict `mod 2^64` byte-limb smoke also remains as
  low-level coverage. The public and private HIP wrap64 tests are correctness
  coverage for the tiled byte-limb kernel, not optimized matrix-engine
  byte-GEMM performance evidence.

Not yet implemented:

- Optimized strict `mod 2^64` GPU byte-GEMM kernels.
- Accelerator integration of the signed-INT8 correction algebra for unsigned
  byte-limb wraparound.

Semantic guardrail:

- `RNS8_BOUNDED_I64` and `RNS8_BOUNDED_U64` are exact-result contracts. The
  caller-supplied bound is part of that contract, and the current CPU and direct
  HIP paths use odd-modulus CRT reconstruction only for results recoverable
  inside the stated range.
- `RNS8_EXACT_WIDE_SIGNED` and `RNS8_EXACT_WIDE_UNSIGNED` are not aliases for
  bounded 64-bit export with a larger prefix. They support persistent RNS output
  with `RNS8_BOUND_NONE` and explicit little-endian limb export. CPU Boost
  reconstruction remains the reference; direct HIP export reconstructs fixed
  limbs on device for correctness validation and copies only the requested limb
  layout to host. Signed export interprets the CRT result as a centered exact
  integer and emits exactly `limb_count` two's-complement limbs; unsigned export
  interprets the canonical nonnegative result and emits exactly `limb_count`
  magnitude limbs. The APIs report `RNS8_RANGE_ERROR` rather than truncating
  when the requested fixed width is too small.
- `RNS8_WRAP_U64_MOD_2_64` is not implemented by the odd-modulus CRT ladder.
  Strict low-64-bit wraparound requires the byte-limb backend so unsigned byte
  semantics, Comba accumulation, carry handling, and low-limb export are tested
  directly. The current public surface includes the CPU byte-limb backend and a
  direct HIP correctness path with device-resident byte-limb matrices. RNS/CRT
  GEMM and bounded exports still reject wrap descriptors. A bounded API call is
  only valid for wrap-like inputs when the exact mathematical result is also
  within the supplied bounded contract.

Do not treat the current direct HIP kernel as performance evidence. It is a
minimal correctness proof for the Windows HIP compile/run path. Its
centered-range corrections are source-level branchless, but reciprocal
reduction and instruction-level validation remain future optimization work.
