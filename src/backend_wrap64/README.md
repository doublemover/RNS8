# Strict Wraparound Backend

Reserved for strict `mod 2^64` byte-limb GEMM.

This path must use unsigned byte-limb semantics. It must not reuse odd-modulus
CRT as a wraparound substitute unless an explicit exact bound is supplied by a
bounded semantic contract.

Current status:

- `RNS8_BACKEND_WRAP64_BYTE_LIMB` is implemented as a CPU reference context for
  `rns8_gemm_wrap_u64_oneshot` and persistent byte-limb matrix storage.
- `RNS8_WRAP_U64_MOD_2_64` is accepted only with `RNS8_BOUND_NONE`, no CRT
  prefix, and byte-limb storage. CPU reference RNS contexts reject it.
  `RNS8_BACKEND_HIP_DIRECT` accepts it through device-resident byte-limb
  matrices, not through RNS residue storage.
- Public wrap descriptors and wrap matrix descriptors are validated before
  backend availability. Malformed wrap metadata returns `RNS8_INVALID_ARGUMENT`;
  a valid wrap descriptor on a backend that does not implement strict byte-limb
  wraparound returns `RNS8_UNSUPPORTED_BACKEND`.
- `wrap64_reference.cpp` contains the CPU byte-limb reference for persistent
  matrix packing, persistent GEMM, and low-64-bit export. Persistent CPU GEMM
  consumes compact resident byte-limb matrices directly after pack, not padded
  host storage, and uses native unsigned `uint64_t` multiply/accumulate because
  unsigned overflow is defined modulo `2^64`. The file keeps strict wraparound
  arithmetic separate from the odd-modulus CRT path and retains the full
  byte-diagonal decomposition oracle used by wrap64 correctness tests. It also
  contains the signed-INT8 correction algebra future accelerator paths must use
  when unsigned byte products are computed through signed INT8 hardware
  instructions: signed byte product plus explicit high-bit correction terms,
  not a shortcut through odd-modulus CRT or a separate unsigned product.
- `wrap64_hip_kernels.hip` contains direct-HIP pack, tiled byte-limb
  correctness, and export kernels. The GEMM kernel stages 16x16 output tiles
  through K tiles, sums only the low eight Comba product diagonals, and uses the
  explicit signed-INT8 correction algebra for the 36 byte-product pairs that can
  affect the low 64 bits. Higher product diagonals are multiples of `2^64` and
  are intentionally not materialized. Device-side carry propagation writes the
  final low-64-bit byte limbs. The public HIP_DIRECT one-shot and persistent
  wrap64 APIs use matrix-owned device byte-limb storage and are tested against
  the CPU reference. Newly created HIP wrap matrices are not current until
  `rns8_pack_u64` populates their device byte limbs. Persistent HIP GEMM/export
  require device-current, host-not-current byte limbs from wrap64 pack or GEMM;
  they do not upload host-current byte limbs as a hidden fallback during
  GEMM/export. Private HIP pack/export helper tests lock
  padded host rows, compact device byte-limb layout, tile-tail dimensions, and
  helper-buffer reuse.
- Public wrap64 pack, GEMM, and export reject residue-backed matrices, bounded
  metadata, nonzero CRT prefixes, and RNS export/GEMM APIs. A wrap descriptor
  must remain byte-limb-only from matrix creation through export.
- The current optimized direct-HIP byte-GEMM36 path is
  `direct_hip_wrap64_byte_gemm36_u32acc_tiled_2d_v4` for `K <= 4096` and the
  v4 u64-accumulator fallback above that cap. It uses 2D launch geometry for
  pack, GEMM tile selection, and export so shape-dependent row/column decoding
  does not introduce variable reciprocal/divide instructions. The scalar
  direct-HIP kernel uses direct unsigned byte products, accumulates the low
  Comba diagonals in uint32 where safe, widens during carry propagation, and
  keeps scalar pack/export kernels for 64-like shapes where vectorized compact
  pack/export lost end-to-end. An experimental two-output-cell colpair kernel,
  `direct_hip_wrap64_byte_gemm36_u32acc_colpair_2d_v5`, is compiled and can be
  selected for large `K <= 4096` shapes by setting
  `RNS8_WRAP64_HIP_COLPAIR_EXPERIMENT=1`; it is not the default because current
  Windows `gfx1100` captures did not show a net end-to-end win over v4. The
  signedness correction algebra remains implemented and tested on CPU for any
  future backend that exposes only signed INT8 products. Matrix-engine
  byte-GEMM36 remains intentionally disabled until a compiled unsigned-byte or
  correctly corrected signed-INT8 matrix instruction path has ISA evidence and
  exact differentials.
- Bounded `RNS8_BOUNDED_U64` calls are exact-result calls, not wraparound
  calls. They may use odd-modulus CRT only when the exact mathematical output is
  recoverable inside the caller-supplied bound.

Acceptance bar for enabling this backend:

- Pack inputs as unsigned base-256 limbs.
- Compute the 36 low-64-relevant byte-product pairs across the low eight
  Comba diagonals.
- Accumulate Comba diagonals with deterministic carry propagation.
- Test unsigned byte signedness handling explicitly when a selected accelerator
  exposes only signed INT8 GEMM.
- Compare minimal GPU smoke/correctness tests against a CPU reference before
  any performance claim.
