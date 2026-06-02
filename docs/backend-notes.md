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
- Finite ring/field u8: CPU reference and direct HIP are implemented through
  explicit modulus APIs for both one-shot calls and persistent resident finite
  matrices.

Unsupported backends must return unsupported status. They must not expose stub
paths that appear to validate GPU behavior.
`RNS8_BACKEND_AUTO` selects only the current context's default backend at
context/plan creation time; it does not route valid descriptors across semantic
backend families such as CPU reference to wrap64 byte-limb.
Status precedence is part of the public C ABI hard cut: malformed descriptors
and ABI misuse return `RNS8_INVALID_ARGUMENT` before backend routing, including
unknown semantics, bound-kind, or layout enum values. Valid descriptors naming
known but unavailable backend families, future backend enum values, known
unimplemented column-major layout, or known unimplemented bounded input-range
contracts report `RNS8_UNSUPPORTED_BACKEND` only after descriptor validation.
`rns8_status_string` covers every public status code plus out-of-range
`unknown status`. `rns8-inspect` adds requested-backend context to unsupported
backend errors and tells users when an accelerator request is evidence-only; it
does not reinterpret unsupported accelerator requests as working correctness
backends. CTest runs `tools/test_inspect_cli.py` against the built
`rns8-inspect` executable to pin invalid backend-string rejection and
evidence-only accelerator diagnostics.

The future backend directories under `src/` are scaffold markers only. They
exist to keep ownership boundaries visible while preserving the rule that no
accelerator path counts until it has compiled kernels and exact CPU
differential validation.

Optional accelerator discovery is platform evidence, not backend enablement.
`tools/check_dependencies.py` and the `FindRNS8HIPBLASLT.cmake`,
`FindRNS8CK.cmake`, and `FindRNS8ROCWMMA.cmake` modules can report candidate
hipBLASLt, CK, and rocWMMA component files. AMDGPU builtins have no
discovery-only readiness path because they require target-specific kernels.
These probes are shallow header/library/tool discovery only. They do not
compile kernels, link an accelerator backend, run device capability checks, or
satisfy correctness requirements.
The dependency checker's machine-readable readiness object also carries a
separate `accelerator_enablement` section. Every accelerator enable flag remains
`fail_fast_until_real_exact_correctness_backend`, every correctness backend is
`not_implemented`, `validated_correctness_backend` is false, and
`backend_enablement` stays `disabled` regardless of component discovery or
optional compile/run probe evidence.
The same report carries `readiness.correctness_backend_validation`, which is
the hard boundary between implemented correctness backend families and
candidate accelerator evidence. The dependency checker does not validate CPU,
direct-HIP, wrap64, or accelerator correctness; its
`validated_by_this_report` list remains empty. Accelerator component/probe
records carry `evidence_class=candidate_accelerator_evidence_only` and
`candidate_evidence_is_correctness_validation=false`.

The direct HIP pack kernels copy logical host `int64_t` and `uint64_t` inputs
to a matrix-owned device upload buffer and write centered residues into
matrix-owned device residue storage. The direct HIP RNS GEMM path consumes those
device residues directly, launches inspectable 16x16 output tiles per modulus,
stages A/B residue tiles in shared memory, and reduces each INT32 K-block sum
to a centered residue in the kernel without materializing INT32 output
matrices. For K above 65536, it launches multiple block kernels and
accumulates the centered residue on device. The resident RNS
GEMM host path routes every per-modulus launch through one metadata contract
that carries the modulus value, modulus index, selected prefix for that full
matrix or tile, and the safe K-block cap; the host helper and HIP launch
entrypoint reject inconsistent metadata before queueing kernels, including
modulus values that do not match the default ladder entry for the supplied
index, and reciprocal values that do not match the accepted small modulus.
Tiled GEMM/export wrappers also require the copied tile schedule to form a
complete row-major output grid with no duplicate or missing tile coordinates,
consistent row/column extents, valid prefix metadata, and group indices that
match the sorted selected-prefix groups. Zero-output tiles may carry zero range
bits. Malformed schedules are rejected before GEMM launch or export/status
buffer growth. Tiled GEMM dispatches by those selected-prefix groups while still
launching only the selected prefix for each tile. GEMM centered reduction uses
validated reciprocal metadata for the accepted small modulus and mask arithmetic
for centered-range correction, but the kernel has not been promoted to an
ISA-verified performance kernel.
Bounded direct-HIP GEMM requires A and B to have current device residues. A
host-current bounded matrix with stale device residues is rejected by persistent
GEMM instead of being uploaded implicitly at dispatch time.

Persistent same-shape direct-HIP calls are allocation-observed in tests. The
first pack/export may grow matrix-owned upload/export/status buffers. A repeated
pack/GEMM/export cycle over the same persistent matrices must leave the direct
HIP allocation counters, device residue pointers, upload buffers, export buffer,
and status buffer unchanged.
Bounded direct-HIP exports require current device residues. A host-current
matrix with stale device residues is rejected by bounded i64/u64 export instead
of being uploaded implicitly during CRT reconstruction.

Workspace ownership is also part of the semantic contract. Workspaces are
created from a plan and remain tagged with that plan's backend, shape, prefix,
semantics, bound kind, bound value, tile geometry, selected-prefix schedule
metadata, and an internal schedule fingerprint over copied per-tile bounds and
tile entries. A same-shape workspace from exact-wide, bounded global, bounded
per-tile, wrap64 semantics, or a different per-tile bounded schedule is rejected
instead of being silently reused across contracts. Per-tile bounded matrices
must also carry the plan's tile geometry; stale matrix tile metadata is rejected
before GEMM/export dispatch. Successful bounded RNS GEMM stamps the output
matrix source version from the packed A/B source versions; rejected dispatch
leaves the existing output version unchanged.

Bounded direct HIP export reconstructs i64/u64 outputs on device with a fixed
three-limb Garner kernel for prefixes up to `RNS8_MAX_SUPPORTED_PREFIX`, writes
a device status for range errors, and copies the compact output to the caller's
host layout only after the device status reports success. Range-error exports
leave the caller's host output untouched and reuse the same matrix-owned
export/status buffers on repeated same-shape calls; no upload buffer is grown by
bounded export, and C does not gain an upload buffer through GEMM/export.
Per-tile bounded export uses the same device reconstruction
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
`limb_count` contiguous little-endian `uint64_t` limbs, with `limb_count` in
`[1, 32]`, at
`dst[((row * ld) + col) * limb_count + limb]`.

Exact-wide descriptors must carry no bound value and no tile-bound storage.
Global bounded descriptors likewise reject stray tile-bound pointers/counts.
These checks return `RNS8_INVALID_ARGUMENT` for malformed descriptors, while
valid semantics on unavailable backends still return `RNS8_UNSUPPORTED_BACKEND`.
That split also applies to one-shot helpers and keeps stale CRT metadata from
becoming an implicit alternate route: a malformed descriptor naming a
future/evidence-only backend is invalid before backend availability is
considered.
Exact-wide limb export applies the same split to public ABI arguments: null
handles, null destinations, invalid `limb_count`, invalid `ld`, and overflowing
output layout calculations are malformed calls, not unsupported backend cases.

CPU export uses explicit fixed-width limbs: signed output reconstructs the
centered integer and emits two's-complement in exactly `limb_count` limbs,
while unsigned output reconstructs the canonical nonnegative integer and emits
magnitude limbs in exactly `limb_count` limbs. Both return `RNS8_RANGE_ERROR`
when the requested width cannot represent the reconstructed value and stage the
whole export before writing the caller's padded host layout, so range errors
leave every destination limb untouched. No low-limb truncation, saturation,
bounded i64/u64 export, or strict wrap64 interpretation is accepted. Direct HIP
exports exact-wide limbs only from device-current, device-resident RNS matrices
with the same fixed-width ABI, range-error preservation for too few limbs, and
strided host layout. A host-current direct-HIP matrix with stale device residues
is rejected by exact-wide export instead of being uploaded implicitly during
reconstruction. CPU Boost.Multiprecision reconstruction remains the reference
and debug path.
Dependency readiness reports expose `exact_wide_platform_validation` separately
from host dependency gates: Windows `gfx1100` exact-wide evidence is not Linux
ROCm or Instinct validation, and those targets remain unvalidated until run on a
real supported Linux ROCm host with exact CPU differentials. The report-level
`hard_cut_self_checks` section is an internal consistency check that keeps
Windows-to-Linux/Instinct promotion false and keeps accelerator evidence from
turning into backend enablement.
The CPU signed CRT representative uses the centered threshold
`x >= ceil(P / 2)`, so the exact half-product residue class maps negative for
even modulus products. Unit coverage pins one-limb signed min/max boundaries,
negative two's-complement sign extension through 32 limbs, unsigned one-limb
overflow rejection, two-limb unsigned success, padded element-stride export,
descriptor rejection, and wrong export-function rejection.
Direct-HIP differential coverage additionally compares CPU and resident-device
export for max-width 32-limb padded layouts, negative centered signed values,
unsigned high-bit magnitudes, destination-preserving range errors, wrong
signed/unsigned export functions, and hard rejection of bounded or wrap64
matrix handles under exact-wide plans.

Strict wraparound `RNS8_WRAP_U64_MOD_2_64` is exposed through byte-limb storage,
not odd-modulus CRT. `RNS8_BACKEND_WRAP64_BYTE_LIMB` is the CPU reference
backend; `RNS8_BACKEND_HIP_DIRECT` owns device byte-limb buffers for the same
semantics. Both support `rns8_gemm_wrap_u64_oneshot` and persistent byte-limb
matrices via `rns8_pack_u64`, `rns8_gemm_wrap_u64`, and
`rns8_export_wrap_u64`. The paths return low-64-bit `uint64_t` output, do not
allocate RNS residue matrices for wrap descriptors, do not use CRT
reconstruction, and reject bounds or prefixes in the descriptor as invalid
arguments.

The direct HIP wrap64 path is a tiled byte-limb correctness kernel. It stages
16x16 output tiles through K tiles while each output sums the low eight Comba
product diagonals with the same signed-INT8 correction algebra as the CPU
oracle for the 36 byte-product pairs that can affect the low 64 bits, performs
one deterministic carry pass into the low 64 bits, keeps A/B/C byte-limb
storage device-resident across pack/GEMM/export, and is tested against the CPU
byte-limb reference. It is not an optimized matrix-engine byte-GEMM accelerator
path, and it is not performance evidence.

Wrap64 host leading dimensions are boundary-only metadata. CPU and direct-HIP
pack/export accept padded host layouts, but persistent byte-limb matrices and
device buffers are compact row-major `rows * cols * 8` storage. The direct-HIP
wrapper validates that compact byte-limb layout separately from padded host
pitch so wrap64 cannot route through RNS residue storage or treat host padding
as device limbs. CPU wrap64 GEMM consumes the same compact resident byte-limb
layout and accumulates only the low eight Comba diagonals, covering the 36
byte-product pairs that can affect the low 64 bits, with the signed-INT8
correction helper before carry export; it does not reread padded host matrices
after pack. CPU wrap64 GEMM also rejects matrices carrying stale residue
currentness, device byte-limb currentness, or bounded per-tile schedule metadata.
The public CPU and direct-HIP wrap64 pack/GEMM/export boundaries reject stale
RNS residue currentness, CRT prefixes, bounded schedule fields, stale tile
geometry, and residue storage before backend dispatch; those fields are not
alternate wrap64 routes.
Public one-shot descriptors and wrap matrix descriptors also validate stale
bound, prefix, tile-bound, flag, and layout metadata before backend availability:
malformed wrap metadata is `RNS8_INVALID_ARGUMENT`, while a valid wrap contract
on an unavailable backend remains `RNS8_UNSUPPORTED_BACKEND`.
On HIP, newly created persistent matrices allocate resident storage but are
non-current until pack or GEMM produces them. Bounded/exact-wide RNS GEMM and
export require the appropriate current residue state for the selected backend,
and wrap64 GEMM/export require device-current byte limbs. A host-current wrap
matrix is invalid at GEMM/export, even when device byte limbs are also marked
current, instead of being uploaded implicitly or treated as a second current
copy; `rns8_pack_u64` is the public host-to-device ingress for strict wrap64
inputs, and GEMM is the device-current producer for outputs.

Unsigned byte semantics are explicit. The CPU reference includes a tested
signed-INT8 correction helper that reconstructs each unsigned byte product from
the product a signed INT8 accelerator would expose plus a deterministic
correction term. It also includes a separate 36-byte-pair decomposition oracle
that sums the low eight byte-product diagonals and then performs Comba carry
propagation. The direct HIP correctness kernel consumes the same correction
algebra at device source level; no signed-INT8 accelerator backend is enabled
by this.

hipBLASLt, CK, rocWMMA, and AMDGPU builtin paths remain accelerator candidates
only. Shallow discovery, compile/link probes, or builtin availability notes do
not promote a backend to correctness-ready status. A future accelerator backend
must have compiled kernels, explicit semantic support, and exact CPU
differential coverage before enable flags stop failing fast. The fail-fast
flags are `RNS8_ENABLE_HIPBLASLT`, `RNS8_ENABLE_CK`,
`RNS8_ENABLE_ROCWMMA`, and `RNS8_ENABLE_AMDGPU_BUILTINS`. CTest registers a
negative configure case for each flag so accidental placeholder backend
enablement fails at the integration gate.

Wrap64 benchmark captures support both the CPU byte-limb reference and the
direct-HIP tiled byte-limb correctness path. HIP wrap64 event captures use
wrap64-specific tiled byte-GEMM/export labels, report
`selected_kernel=direct_hip_wrap64_tiled_byte_limb_gemm_v1`, and keep
current aggregate phase labels; they are raw timing evidence for the
correctness path only, not optimized byte-GEMM performance evidence.

Finite `uint8_t` GEMM uses explicit modulus arguments rather than the CRT
prefix ladder. `RNS8_FINITE_RING_U8` accepts moduli in `[2, 256]`;
`RNS8_FINITE_FIELD_U8` requires a prime modulus `<= 251`. Finite descriptors
require `RNS8_BOUND_NONE`, `bound = 0`, `max_prefix = 0`, no tile-bound
metadata, and matching finite semantics.

CPU finite one-shot and persistent paths pack canonical bytes to centered
residues for the requested modulus, run the same K-split ring GEMM reference,
and export canonical byte residues. Persistent finite matrices own one
prefix-zero residue plane and are stamped with the modulus used by
`rns8_pack_finite_u8`; resident finite GEMM/export reject mismatched or stale
matrix modulus state. Direct HIP persistent finite matrices own device-resident
one-plane residues plus matrix-owned upload/export buffers, and the resident
GEMM calls the inspectable tiled INT8xINT8->INT32 ring kernel with fused
centered reciprocal reduction for the explicit modulus; HIP launch wrappers
reject stale reciprocal metadata before queueing work. The one-shot direct HIP
finite path is still available as a convenience surface but does not define a
separate backend contract. Finite is not an odd-modulus CRT route, not
exact-wide export, and not strict mod 2^64 wraparound.
