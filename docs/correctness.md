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
- Phase 2 fixed 9-modulus CPU reference milestone checks lock the default
  prefix-9 schedule, signed cancellation and unsigned accumulation at K values
  65535, 65536, and 65537, full-width `UINT64_MAX` output with padded leading
  dimensions, and signed/unsigned per-tile selected-prefix schedule parity.
- Worst-case positive, negative, and unsigned accumulation checks at and just
  above the 65536 K-block split point.
- Semantic guard tests that bounded APIs reject `RNS8_BOUND_NONE`, exact-wide
  rejects bounded-looking metadata, finite-ring/finite-field/future accelerator
  requests report unsupported, and strict wraparound never falls through to
  bounded CRT behavior.
- Public API hard-cut tests cover exact-wide invalid limb layout, null context,
  plan, matrix, and output pointers, stale bound/tile-bound descriptor
  metadata, stale-prefix matrix handles, bounded matrix handles, wrap byte-limb
  matrix handles, bounded export shortcuts, wrap export shortcuts,
  signed/unsigned exact-wide cross-export attempts, and unsupported accelerator
  context kinds. One-shot helpers preserve the same status precedence as plan
  creation: malformed descriptors return `RNS8_INVALID_ARGUMENT` even when they
  name future/evidence-only backends, while valid descriptors for unavailable
  backends return `RNS8_UNSUPPORTED_BACKEND`. Unknown public enum values for
  semantics, bound kinds, and layouts are malformed ABI input; known but
  unimplemented contracts such as column-major layout, finite-ring/field
  semantics, input-range bounded contracts, or future backend enums remain
  unsupported backend requests after descriptor validation succeeds.
- User-visible diagnostic tests pin `rns8_status_string` for every public
  status code and the out-of-range `unknown status` case. API guard tests also
  pin `RNS8_BACKEND_AUTO` as a context-default selector only: CPU AUTO accepts
  CPU-backed exact-wide descriptors, CPU AUTO rejects wrap64, and wrap64 AUTO
  rejects bounded and exact-wide descriptors instead of routing across semantic
  backend families.
- Descriptor hard-cut tests reject unbounded exact-wide plans carrying stale
  nonzero bounds, global plans carrying tile-bound storage, and matrix
  descriptors whose owned RNS or byte-limb storage would overflow the host
  allocation size.
- Persistent workspace guard tests reject same-shape workspaces from different
  semantic contracts, bound kinds, bound values, tile geometry, or per-tile
  selected-prefix schedules, so workspace reuse cannot silently route a
  bounded, exact-wide, per-tile, or wrap64 plan through another contract.
  Per-tile bounded matrix tile geometry is also part of the persistent contract
  and is rejected when stale before GEMM/export dispatch.
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
  limb stride, and `limb_count` must be in [1, 32]. Both report range errors
  when too few limbs are supplied, and neither truncates nor wraps on
  insufficient width. Range-error exports preserve the caller's destination.
  Descriptor and export tests reject cross-semantic bounded, signed/unsigned
  exact-wide, and wrap64 interpretations. Direct HIP export requires
  device-current resident RNS output, rejects host-current stale device
  residues, and leaves device-resident residues on device instead of
  synchronizing host residue storage. Direct HIP differential tests also compare
  CPU and HIP fixed-width export at `limb_count == 32` with padded element
  strides, negative centered signed values, unsigned high-bit magnitudes, and
  exact-wide plans paired with bounded or wrap64 matrix handles.
- CPU exact-wide fixed-width export tests also pin the signed centered
  half-product representative at small and maximum supported prefixes,
  one-limb signed min/max boundaries, negative two's-complement sign extension
  through 32 limbs, public signed high-bit negative export, unsigned one-limb
  overflow rejection, two-limb unsigned success including public high-bit
  magnitude cases, padded element-stride export, descriptor rejection, stale
  matrix-state rejection, null-handle rejection, and wrong export-function
  rejection.
- Strict `mod 2^64` byte-limb product, GEMM-cell, public CPU one-shot, and
  persistent byte-limb matrix tests compared against Boost.Multiprecision
  low-64-bit results. The public wrap path requires explicit wrap64 semantics
  and byte-limb storage, uses separate pack/GEMM/export APIs for persistent
  matrices, and rejects CRT bounds/prefixes. CPU persistent GEMM consumes
  compact resident byte limbs after pack, not padded host input matrices. CPU
  and direct-HIP persistent tests also mutate host inputs after pack plus stale
  residue currentness, stale byte-limb currentness, and bounded schedule
  metadata to prove those fields are rejected at pack/GEMM/export boundaries
  instead of ignored.
- Unsigned byte-limb signedness correction tests cover every byte pair and
  verify that the signed-INT8 correction algebra composes through Comba
  diagonals. This is readiness coverage for future signed-INT8 accelerator use,
  not an enabled accelerator backend.
- A separate CPU 36-byte-GEMM oracle sums the low-product byte diagonals with
  the signed-INT8 correction helper, performs Comba carry propagation, and is
  compared against both Boost.Multiprecision low-64-bit results and the existing
  byte-limb Comba GEMM-cell reference over boundary and fixed-seed full-width
  random inputs.
- Public direct HIP strict `mod 2^64` byte-limb one-shot and persistent API
  tests compared against the CPU byte-limb backend. HIP wrap matrices own
  device-resident byte-limb buffers, do not allocate RNS residues, preserve
  device pointer stability through pack/GEMM/export, and support padded host
  leading dimensions on pack and export while keeping compact row-major
  `rows * cols * 8` device byte-limb storage. GEMM/export require
  device-current byte limbs instead of silently uploading host-current wrap
  matrices. Same-shape wrap64 HIP resident tests also check allocation counters
  across repeat pack/GEMM/export cycles, including larger multi-tile padded
  shapes with host inputs mutated after pack. The HIP GEMM correctness kernel
  sums the 36 low-product byte diagonals with device-side signed-INT8
  correction algebra and then performs deterministic carry propagation into the
  low 64 bits.
- Direct HIP signed and unsigned residue packing compared against CPU reference
  residue storage, including full-width boundary values and padded leading
  dimensions.
- Direct HIP one-modulus ring-GEMM smoke tests compared against CPU reference
  on `gfx1100` when HIP is enabled and a device is visible, including a
  centered-correction boundary case for negative, positive-threshold, and
  near-zero residues. Private direct-HIP launch metadata tests reject a modulus
  value that does not match the default ladder entry for the supplied modulus
  index before queueing work.
- Direct HIP device-resident RNS matrices, K-block splitting above 65536, fused
  INT32-to-centered-residue reduction without INT32 global output, and bounded
  signed/unsigned GPU CRT export smoke tests through prefix 20 against the CPU
  reference. Bounded direct-HIP GEMM and export also cover host-current/
  device-stale residue matrices and reject them instead of uploading during
  dispatch. Signed and unsigned range-error export cases compare CPU and HIP
  status, preserve caller output sentinels, and check repeated device-current
  exports reuse matrix-owned export/status buffers without growing upload
  buffers.
- Direct HIP bounded persistent tests cover fixed prefix-9 unsigned GEMM at the
  exact 65536 K-block boundary with padded host input/output layouts, CPU
  reference comparison, and repeated same-shape allocation reuse. The private
  ring GEMM differential also covers non-multiple 16x16 output tile tails,
  partial K tiles, and padded leading dimensions against the CPU ring reference.
- Direct HIP per-tile bounded signed/unsigned GEMM tests compare output against
  the CPU reference, cover tile-local range errors, padded host export layouts,
  schedule parity, signed K-split cancellation under selected-prefix execution,
  and verify skipped residue planes above each tile's selected prefix remain
  untouched on device. Private tiled wrapper tests reject corrupted tile
  metadata where `required_prefix > selected_prefix`, selected-prefix
  `group_index` is stale, tile coordinates are duplicated or missing, tile
  extents do not cover the output grid, or prefix metadata is invalid before
  GEMM launch or export/status buffer allocation. Zero-output tiles may have
  zero range bits when their prefix metadata is otherwise valid. Adaptive
  per-tile K-split reuse coverage compares against CPU with padded output and
  mixed selected-prefix groups while checking same-shape resident buffer
  allocation and workspace schedule-metadata stability after warmup. Direct HIP
  also rejects same-shape stale per-tile workspace schedules and stale per-tile
  matrix tile metadata without changing warmed resident allocation counters.
- Benchmark schema v4 captures direct-HIP adaptive per-tile bounded runs with
  exact seeded-input tile-bound prepass metadata, selected tiled kernel name,
  adaptive execution flags, and aggregate HIP event timing scope. This is
  benchmark evidence metadata for the correctness path, not an optimized GPU
  performance claim.
- Private direct HIP strict `mod 2^64` byte-limb smoke also remains as
  low-level coverage. Private wrap64 HIP tests also cover padded-host pack and
  export into compact device byte-limb storage with reusable helper buffers. The
  public and private HIP wrap64 tests are correctness coverage for the tiled
  byte-limb kernel, not optimized matrix-engine byte-GEMM performance evidence.
- CTest configure-negative coverage asserts that
  `RNS8_ENABLE_HIPBLASLT`, `RNS8_ENABLE_CK`, `RNS8_ENABLE_ROCWMMA`, and
  `RNS8_ENABLE_AMDGPU_BUILTINS` fail fast with the evidence-only policy message
  while no correctness backend exists.

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
  reconstruction remains the reference and CPU public export stages all cells
  before writing the caller's padded host layout. Direct HIP export reconstructs
  fixed limbs on device from device-current resident RNS storage and copies only
  the requested limb layout to host after device status reports success. Signed
  export interprets the CRT result as a centered exact
  integer and emits exactly `limb_count` two's-complement limbs. The centered
  representative uses `x >= ceil(P / 2)` as the negative threshold for selected
  modulus product `P`, matching centered residue packing. Unsigned export
  interprets the canonical nonnegative result and emits exactly `limb_count`
  magnitude limbs. The APIs report `RNS8_RANGE_ERROR` rather than truncating
  when the requested fixed width is too small, and they reject attempts to use
  bounded i64/u64 or strict wrap64 export as an exact-wide shortcut.
  `limb_count == 0`, `limb_count > 32`, null handles, null destinations, and
  output leading dimensions smaller than the matrix width are invalid ABI
  calls.
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
