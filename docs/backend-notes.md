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
- hipBLASLt: implemented as an opt-in baseline when configured with
  `RNS8_ENABLE_HIPBLASLT=ON` and `RNS8_ENABLE_HIP=ON`. The backend uses
  resident HIP RNS matrices, padded 16-aligned transposed INT8 pack buffers,
  hipBLASLt `int8 x int8 -> int32`, padded INT32 scratch, and a separate HIP
  centered-residue reduction kernel. It supports fixed-prefix bounded i64/u64,
  exact-wide RNS output, and finite u8 ring/field GEMM; adaptive per-tile
  bounded plans and wrap64 remain unsupported. This is a correctness baseline,
  not a performance-validated production accelerator.
- CK: implemented as an opt-in Windows `gfx1100` correctness backend under
  `RNS8_ENABLE_CK=ON`. It uses repo-local CK headers plus RNS8-owned HIP
  pack/output kernels for fused centered-residue `int8 x int8 -> int32` GEMM
  over fixed-prefix bounded plans, adaptive per-tile bounded plans, exact-wide
  RNS output, and finite u8. It is not performance-validated.
- rocWMMA: implemented as an opt-in Windows `gfx1100` correctness backend
  under `RNS8_ENABLE_ROCWMMA=ON`. It uses repo-local rocWMMA headers and
  RNS8-owned HIP kernels for signed INT8 WMMA with fused centered-residue
  output over fixed-prefix bounded plans, adaptive per-tile bounded plans,
  exact-wide RNS output, and finite u8. It is not performance-validated.
- AMDGPU builtins: not implemented.
- Wraparound byte-limb backend: CPU reference implemented for one-shot and
  persistent byte-limb matrix APIs. Direct HIP supports a public tiled
  byte-limb correctness path for `RNS8_WRAP_U64_MOD_2_64` under
  `RNS8_BACKEND_HIP_DIRECT` with device-resident byte-limb buffers. Optimized
  matrix-engine byte GEMMs are not implemented.
- Finite ring/field u8: CPU reference and direct HIP are implemented through
  explicit modulus APIs. Public one-shot calls now construct the same resident
  finite matrices and workspace used by the persistent API, then pack, GEMM,
  and export through that path. There is no separate direct-HIP finite one-shot
  backend route.

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
also reports the public backend capability status, selected kernel, enable flag,
epilogue mode, workspace mode, and ISA-evidence state. It does not reinterpret
unsupported accelerator requests as working correctness backends. CTest runs
`tools/test_inspect_cli.py` against the built
`rns8-inspect` executable to pin invalid backend-string rejection and
evidence-only accelerator diagnostics.

The public C ABI exposes accelerator readiness through
`rns8_get_backend_capability_info` and plan-selected backend metadata through
`rns8_get_plan_backend_info`. Current implemented correctness backends report
compiled kernels and exact differential validation, but no performance
validation. In default builds, non-enabled accelerator backend kinds report
evidence-only or fail-fast status with no compiled correctness kernel. In
explicit accelerator presets, hipBLASLt, CK, and rocWMMA report their selected
kernel and exact validation state. AMDGPU builtins still report
`enable_flag_fail_fast`, no compiled kernel, no exact differential validation,
and no performance validation. Plan backend metadata includes selected kernel,
accelerator library/version, capability status, epilogue mode, workspace mode,
workspace byte requirement, ISA evidence, and an autotune key. Workspaces copy
those fields from the plan, and workspace validation treats them as part of the
same deterministic contract as schedule metadata.

Autotune cache support is implemented as an explicit reviewed-evidence layer,
not as an automatic accelerator promotion path. Raw
`rns8-bench --write-autotune-cache` writes are refused unless the selected plan
is already performance validated. `tools/benchmark_sweep.py
--write-autotune-cache` is the promotion path: it validates captures, groups
them by same-contract semantics/shape/layout/target/toolchain/input seed,
requires the matching CPU/direct-HIP/vector-ALU baselines where applicable, and
writes only the fastest promotable accelerator entry. Cache entries record the
plan autotune key, selected backend/kernel, target id, library or HIP SDK
version, semantic contract, shape, layout, K-block, tile size, epilogue,
workspace bytes, reviewed median timings, and validation status in
`%LOCALAPPDATA%\rns8-gemm\autotune.json` on Windows, or the equivalent
`$XDG_CACHE_HOME/rns8-gemm/autotune.json` path on Unix-like hosts.
`RNS8_AUTOTUNE_CACHE_PATH` can override the path for tests and isolated smoke
runs. `rns8-inspect --autotune-key ...` reports exact-hit versus missing-cache
rationale; unreviewed, wrong-schema, or non-reviewed cache hits are reported as
rejected and are not eligible for validated selection.

Benchmark captures make the performance gate explicit with a structured
`comparison_baseline` object. Unreviewed captures keep
`status=required_not_recorded` and `speedup_claimed=false`; schema validation
rejects bounded captures that do not name the same-contract CPU reference and
direct-HIP vector-ALU baseline prerequisites, rejects wrap64 captures that do
not name CPU byte-limb and direct-HIP byte-GEMM36 prerequisites, and rejects any
future `performance_validated=true` capture unless a reviewed same-contract
baseline is attached.

The CK and rocWMMA backend directories under `src/` now contain opt-in
correctness backend implementations. The AMDGPU builtin backend path remains a
reserved ownership boundary. No accelerator path counts from discovery alone:
it must have compiled kernels and exact CPU/direct-HIP differential validation.
hipBLASLt, CK, and rocWMMA are real compiled correctness backends under their
opt-in presets, but still carry `perf_validated=0`.

The Windows `gfx1100` release review workflow has been exercised through
`tools/benchmark_sweep.py` with temp-only cache outputs. Reviewed release
entries currently promote CK/rocWMMA for selected bounded i64 and adaptive
bounded shapes, CK/rocWMMA/hipBLASLt for selected finite-u8 shapes, and no
bounded u64 global shapes. These reviewed entries live in temp cache files and
do not make dependency discovery an enablement signal. hipBLASLt remains a
baseline accelerator for bounded i64/u64, with one finite ring modulus-255
release winner in the reviewed temp cache. AMDGPU builtins remain fail-fast,
and strict wrap64 remains on the direct-HIP
`direct_hip_wrap64_byte_gemm36_tiled_2d_v3` path until a matrix-engine
candidate beats it with exact differentials and ISA evidence.

Optional accelerator discovery is platform evidence, not backend enablement.
`tools/check_dependencies.py` and the `FindRNS8HIPBLASLT.cmake`,
`FindRNS8CK.cmake`, and `FindRNS8ROCWMMA.cmake` modules can report candidate
hipBLASLt, CK, and rocWMMA component files. AMDGPU builtins have no
discovery-only readiness path because they require target-specific kernels.
These probes are shallow header/library/tool discovery only; hipBLASLt
discovery also records AMD's `roc::hipblaslt` CMake target when the installed
HIP SDK exposes it. On Windows, the hipBLASLt backend preset loads the Visual
Studio developer environment automatically and links the installed
`libhipblaslt.dll.a` import archive. Probe-only paths still do not compile
production kernels, link CK/rocWMMA/builtin accelerator backends, run device
capability checks for those families, or satisfy correctness requirements by
themselves.
The dependency checker's machine-readable readiness object also carries a
separate `accelerator_enablement` section. CK and rocWMMA are enabled only by
their explicit backend build flags and validated runtime tests, not discovery
evidence alone. AMDGPU builtin enablement remains
`fail_fast_until_real_exact_correctness_backend`; its correctness backend is
`not_implemented`, `validated_correctness_backend` is false, and
`backend_enablement` stays `disabled` regardless of component discovery or
optional compile/run probe evidence. hipBLASLt is enabled only by the explicit
backend build flag and validated runtime tests, not discovery evidence alone.
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
`tools/check_hip_kernel_isa.py` is registered as a HIP-only CTest gate on the
compiled direct-HIP object. It extracts the `.hip_fatbin`, unbundles the active
AMDGPU target code object, disassembles the direct RNS GEMM kernels, rejects
divide/remainder/rcp mnemonics, and requires `v_mul_hi_u32` as reciprocal
multiply-high evidence. This is an instruction-shape guard for the correctness
kernel, not a throughput claim.
Bounded direct-HIP GEMM requires A and B to have current device residues. A
host-current bounded matrix with stale device residues is rejected by persistent
GEMM instead of being uploaded implicitly at dispatch time.
Public bounded, wrap64, and finite one-shot GEMM APIs share the same internal
resident one-shot owner: they create a plan, resident A/B/C matrices, and a
matching workspace, then pack, dispatch the corresponding persistent GEMM API,
and export. There are no semantic-specific one-shot lifetime paths.

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
oracle for the 36 byte-product pairs that can affect the low 64 bits. That
algebra is signed byte product plus explicit high-bit correction terms, not a
separate unsigned-product shortcut. The kernel performs one deterministic carry
pass into the low 64 bits, keeps A/B/C byte-limb storage device-resident across
pack/GEMM/export, and is tested against the CPU byte-limb reference. It is not
an optimized matrix-engine byte-GEMM accelerator path, and it is not performance
evidence.

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

CK and rocWMMA are opt-in accelerator correctness backends on Windows
`gfx1100`; AMDGPU builtin paths remain accelerator candidates only. Shallow
discovery, compile/link probes, or builtin availability notes do not promote
those backends to correctness-ready status. A future accelerator backend must
have compiled kernels, explicit semantic support, and exact CPU/direct-HIP
differential coverage before its enable flag stops failing fast. The remaining
fail-fast flag is `RNS8_ENABLE_AMDGPU_BUILTINS`; CTest registers negative
configure coverage for that flag. In presets where hipBLASLt, CK, or rocWMMA
are off, discovery-only evidence still does not enable them.

Wrap64 benchmark captures support both the CPU byte-limb reference and the
direct-HIP tiled byte-limb correctness path. HIP wrap64 event captures use
wrap64-specific byte-GEMM36/export labels, report
`selected_kernel=direct_hip_wrap64_byte_gemm36_tiled_2d_v3`, and keep
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
