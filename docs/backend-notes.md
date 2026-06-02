# Backend Notes

Backend status:

- CPU reference: implemented and tested.
- Direct HIP: implemented for device inspection, signed/unsigned residue
  conversion, persistent device-resident RNS matrix buffers, one-modulus
  correctness smoke, fused INT32-to-centered-residue reduction with
  source-level branchless centered correction, and bounded i64/u64 GPU export
  through the supported prefix-20 bound. Exact-wide signed/unsigned limb export
  also reconstructs fixed-width limbs from device-resident RNS output.
  Public bounded GEMM can execute the direct HIP pack, RNS GEMM, and export
  path, with K split into blocks no larger than 65536 before centered residue
  reduction. Per-tile bounded plans use grouped tile launches over only each
  tile's selected prefix and tile-local device CRT export. Internal allocation
  counters and differential tests verify that repeated same-shape persistent
  pack/GEMM/export calls reuse warmed matrix-owned buffers without additional
  direct-HIP allocation or free calls.
- hipBLASLt: not implemented.
- CK: not implemented.
- rocWMMA/AMDGPU builtins: not implemented.
- Wraparound byte-limb backend: CPU reference implemented for one-shot and
  persistent byte-limb matrix APIs. Direct HIP supports a public tiled
  byte-limb correctness path for `RNS8_WRAP_U64_MOD_2_64` under
  `RNS8_BACKEND_HIP_DIRECT` with device-resident byte-limb buffers. Optimized
  matrix-engine byte GEMMs are not implemented.

Unsupported backends must return unsupported status. They must not expose stub
paths that appear to validate GPU behavior.

The future backend directories under `src/` are scaffold markers only. They
exist to keep ownership boundaries visible while preserving the rule that no
accelerator path counts until it has compiled kernels and exact CPU
differential validation.

Optional accelerator discovery is platform evidence, not backend enablement.
`tools/check_dependencies.py` and the `FindRNS8HIPBLASLT.cmake`,
`FindRNS8CK.cmake`, and `FindRNS8ROCWMMA.cmake` modules can report candidate
hipBLASLt, CK, and rocWMMA component files. These probes are shallow
header/library/tool discovery only. They do not compile kernels, link an
accelerator backend, run device capability checks, or satisfy correctness
requirements.

The direct HIP pack kernels copy logical host `int64_t` and `uint64_t` inputs
to a matrix-owned device upload buffer and write centered residues into
matrix-owned device residue storage. The direct HIP RNS GEMM path consumes those
device residues directly, launches inspectable 16x16 output tiles per modulus,
and reduces each INT32 K-block sum to a centered residue in the kernel without
materializing INT32 output matrices. For K above 65536, it launches multiple
block kernels and accumulates the centered residue on device. The current
centered-range correction code uses mask arithmetic instead of source-level
`if` branches, but the kernel still uses ordinary modulo operations and has not
been promoted to a reciprocal-reduction or ISA-verified performance kernel.

Persistent same-shape direct-HIP calls are allocation-observed in tests. The
first pack/export may grow matrix-owned upload/export/status buffers. A repeated
pack/GEMM/export cycle over the same persistent matrices must leave the direct
HIP allocation counters, device residue pointers, upload buffers, export buffer,
and status buffer unchanged.

Bounded direct HIP export reconstructs i64/u64 outputs on device with a fixed
three-limb Garner kernel for prefixes up to `RNS8_MAX_SUPPORTED_PREFIX`, writes
a device status for range errors, and copies the compact output to the caller's
host layout. Per-tile bounded export uses the same device reconstruction
helpers with full-matrix residue strides, each tile's selected prefix, and each
tile's copied bound. Signed export supports the full `int64_t` range, including
`INT64_MIN` when the bounded contract supplies magnitude `2^63`. CPU
Boost.Multiprecision CRT/Garner remains the reference and debug path. The direct
HIP kernels are intentionally inspectable and unoptimized; they are correctness
bring-up kernels, not performance evidence.

Exact-wide signed and unsigned semantics are supported as persistent RNS output
with `RNS8_BOUND_NONE`. They are not exported through the bounded i64/u64 APIs,
and they are not strict low-64-bit wraparound. The limb export ABI treats `ld`
as a leading dimension in output elements, not limbs. Each element owns exactly
`limb_count` contiguous little-endian `uint64_t` limbs at
`dst[((row * ld) + col) * limb_count + limb]`.

CPU export uses explicit fixed-width limbs: signed output reconstructs the
centered integer and emits two's-complement in exactly `limb_count` limbs,
while unsigned output reconstructs the canonical nonnegative integer and emits
magnitude limbs in exactly `limb_count` limbs. Both return `RNS8_RANGE_ERROR`
when the requested width cannot represent the reconstructed value. Direct HIP
exports exact-wide limbs from device-resident RNS matrices with the same
fixed-width ABI, range-error behavior for too few limbs, and strided host
layout. CPU Boost.Multiprecision reconstruction remains the reference and debug
path.

Strict wraparound `RNS8_WRAP_U64_MOD_2_64` is exposed through byte-limb storage,
not odd-modulus CRT. `RNS8_BACKEND_WRAP64_BYTE_LIMB` is the CPU reference
backend; `RNS8_BACKEND_HIP_DIRECT` owns device byte-limb buffers for the same
semantics. Both support `rns8_gemm_wrap_u64_oneshot` and persistent byte-limb
matrices via `rns8_pack_u64`, `rns8_gemm_wrap_u64`, and
`rns8_export_wrap_u64`. The paths return low-64-bit `uint64_t` output, do not
allocate RNS residue matrices for wrap descriptors, do not use CRT
reconstruction, and reject bounds or prefixes in the descriptor.

The direct HIP wrap64 path is a tiled byte-limb correctness kernel. It stages
16x16 output tiles through K tiles while each output still sums the 36
low-product byte diagonals with the same signed-INT8 correction algebra as the
CPU oracle, performs one deterministic carry pass into the low 64 bits, keeps
A/B/C byte-limb storage device-resident across pack/GEMM/export, and is tested
against the CPU byte-limb reference. It is not an optimized matrix-engine
byte-GEMM accelerator path, and it is not performance evidence.

Unsigned byte semantics are explicit. The CPU reference includes a tested
signed-INT8 correction helper that reconstructs each unsigned byte product from
the product a signed INT8 accelerator would expose plus a deterministic
correction term. It also includes a separate 36-byte-GEMM decomposition oracle
that sums byte-product diagonals and then performs Comba carry propagation. The
direct HIP correctness kernel consumes the same correction algebra at device
source level; no signed-INT8 accelerator backend is enabled by this.

hipBLASLt, CK, rocWMMA, and AMDGPU builtin paths remain accelerator candidates
only. Shallow discovery, compile/link probes, or builtin availability notes do
not promote a backend to correctness-ready status. A future accelerator backend
must have compiled kernels, explicit semantic support, and exact CPU
differential coverage before enable flags stop failing fast.

Wrap64 benchmark captures support both the CPU byte-limb reference and the
direct-HIP tiled byte-limb correctness path. HIP wrap64 event captures use
wrap64-specific tiled byte-GEMM/export labels, report
`selected_kernel=direct_hip_wrap64_tiled_byte_limb_gemm_v1`, and keep
schema-compatible aggregate aliases; they are raw timing evidence for the
correctness path only, not optimized byte-GEMM performance evidence.
