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
- `wrap64_reference.cpp` contains the CPU byte-limb Comba reference for
  low-64-bit product, GEMM-cell behavior, persistent matrix packing, persistent
  GEMM, and low-64-bit export. It keeps strict wraparound arithmetic separate
  from the odd-modulus CRT path. It also contains the signed-INT8 correction
  algebra future accelerator paths must use when unsigned byte products are
  computed through signed INT8 hardware instructions, plus a full byte-diagonal
  decomposition oracle used by wrap64 correctness tests.
- `wrap64_hip_kernels.hip` contains direct-HIP pack, tiled byte-limb
  correctness, and export kernels. The GEMM kernel stages 16x16 output tiles
  through K tiles, sums the 36 low-product byte diagonals with device-side
  signed-INT8 correction algebra, then performs deterministic carry propagation
  into the low 64 bits. The public HIP_DIRECT one-shot and persistent wrap64
  APIs use matrix-owned device byte-limb storage and are tested against the CPU
  reference.
- Public wrap64 pack, GEMM, and export reject residue-backed matrices, bounded
  metadata, nonzero CRT prefixes, and RNS export/GEMM APIs. A wrap descriptor
  must remain byte-limb-only from matrix creation through export.
- Optimized matrix-engine byte-GEMM kernels and production GPU performance
  evidence are not implemented yet. The signedness correction algebra is
  implemented and tested on CPU and consumed by the direct-HIP correctness
  kernel, but no signed-INT8 accelerator backend is enabled by it.
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
