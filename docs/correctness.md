# Exactness contract and tests

## CRT range and trusted bounds

Let `P` be the product of the selected pairwise-coprime default moduli.

- Bounded signed output requires `P > 2 × bound`, then uses the centered
  representative and checks both the declared bound and int64 range.
- Bounded unsigned output requires `P > bound`, then checks the canonical
  representative against the declared bound and uint64 range.
- These are strict inequalities. Equality does not give unique reconstruction.
- Global/per-tile output bounds and zero-output/zero-row/zero-column proofs are
  caller contracts. The library does not recompute every such proof from inputs.
  A false bound can produce a different in-range CRT representative. A successful
  export alone cannot prove the original bound was correct.
- `RNS8_BOUND_INPUT_RANGE_AND_K` derives the output bound from the declared input
  magnitudes and K using multiprecision arithmetic. The caller must still supply
  true input ranges. It is implemented, not a future enum-only feature.

The default ladder has 28 values. `RNS8_MAX_SUPPORTED_PREFIX` is currently 20.
The first nine cover bounded 64-bit output; the first twenty give approximately
154.84 bits of CRT range. Moduli include composites (256, 255, 253, 247, 217),
so pairwise coprimality, rather than individual primality, is the CRT requirement.

## Exact-wide output

Exact-wide plan validation requires enough range for arbitrary native 64-bit
inputs of the declared K: `P > K × 2^127` for signed and `P > K × 2^128` for
unsigned. It rejects a prefix that cannot establish that contract. The CPU and
GPU store residues; GEMM does not grow the ladder dynamically.

Exports use 1–32 little-endian uint64 limbs per element. Signed output is a
fixed-width two's-complement value; unsigned output is a magnitude. `ld` counts
elements, not limbs. Too few limbs produce `RNS8_RANGE_ERROR`, not truncation.
Providing more limbs does not enlarge the represented CRT range.

Public exact-wide native packing records the maximum logical input magnitude
(padding is ignored and `INT64_MIN` is handled without signed overflow). Each
resident GEMM requires `P > 2 x bound` for signed or `P > bound` for unsigned
output. Dense, grouped and incremental admission use the minimum of
`K x row_max x col_max`, `row_abs_sum x col_max`, and
`row_max x col_abs_sum`, with multiprecision throughout. Native-packed axes
add exact sums and midpoint deviations: for centers a and b, the dot product is
`a x sum(B) + b x sum(A) - K x a x b + dot(A-a, B-b)`. Its remainder is bounded
by the same absolute-sum/max inequalities. This can admit constant-axis balanced
signed products that the old global bound rejected, including later zero chains.
No exact host GEMM or probabilistic test is used for admission.

Output row/column magnitude maxima and absolute-sum bounds are staged before
execution and committed on success; they never masquerade as exact signed sums.
Missing summaries use the scalar `K x max_abs(A) x max_abs(B)` proof. Prepacked
and sparse routes keep that conservative scalar admission and clear older axis
summaries when producing output. Failure returns `RNS8_RANGE_ERROR` before
changing the destination. Some cancellation-safe products remain unprovable.
Native summary storage is O(rows + columns); two logical input scans build it,
and pairwise admission/output-summary propagation is O(M x N). These are
correctness bounds, with no performance qualification. Repacking replaces the old proof. Stable nonzero
source versions retain the packed value and its proof under the existing caller
currentness contract.

Proofs also record the prefix actually written, separately from allocated
storage capacity. A later operation/export cannot treat unwritten higher planes
as current. `rns8_lift_exact_wide_cpu` explicitly initializes missing planes
within a CPU matrix's already allocated capacity. It requires current host
residues and a known source proof satisfying the strict range inequality at the
initialized prefix. A larger target product cannot repair earlier CRT aliasing.
For each cell, it checks centered source bytes, reconstructs using only those
initialized planes, centers signed integers, checks the retained magnitude proof,
then computes the suffix residues. It never derives a proof from an unknown
representative or weakens the existing row/column admission proof.

The caller supplies a maximum suffix staging payload in bytes. The required
payload is `rows x cols x (target_prefix - initialized_prefix)`; a smaller limit
returns `RNS8_WORKSPACE_TOO_SMALL`. Size arithmetic is checked before allocation.
Allocator bookkeeping and bounded per-cell CRT scratch are excluded; no array of
full reconstructed integers is retained. Every missing plane is staged before
the nonthrowing resident copy and final prefix update. Any failure preserves
all resident bytes, metadata, proofs, and source version. A target already
initialized is a no-op after structural/currentness/proof validation, without
reconstructing cells. Other backends return `RNS8_UNSUPPORTED_BACKEND`; no device
transfer or automatic backend fallback occurs. Unknown proofs, nonunique ranges,
or a reconstructed cell exceeding its proof return `RNS8_RANGE_ERROR`.

Lifting preserves the integer, matrix identity, source version, range summaries,
storage allocation, and currentness. It changes only the suffix bytes and
initialized-prefix proof. Plans, workspaces, storage ceilings, and existing
GEMM/export admission remain unchanged. Choose an adequate fixed-prefix plan
before producing values outside the current output range; explicitly lift inputs
before using higher planes.

`rns8_gemm_exact_wide_cpu_auto` is an explicit opt-in CPU dense GEMM route. It
validates the original plan/workspace binding, then selects the smallest output
prefix at or above the plan's selected prefix and within the plan/A/B/C ceilings.
Fixed-prefix plans retain their fixed prefix. Selection uses the same scalar,
per-axis and centered cancellation bounds as ordinary admission. Each input must
already have a unique proof at its own initialized prefix; even a zero output
bound does not authorize an unknown input. No probabilistic or reference GEMM
result is used to admit missing range.

Both required input suffixes and the entire selected output are staged under one
explicit residue-byte payload limit. Shared `A == B` input storage counts once;
output/input aliases are invalid. The payload is the sum of missing input bytes
and `M x N x selected_prefix` output bytes. Size and combined-budget arithmetic
are checked before staging. Range-summary metadata, allocator bookkeeping,
bounded per-cell CRT scratch and the serial blocked kernel's `4 x N` byte row
accumulator are excluded from this payload budget. The route forces serial
blocked ring execution so allocation failures stay inside the public guard.
Ordinary CPU GEMM retains its existing parallel policy.

Only after both input lifts and all output planes finish does the operation
commit their bytes, updated initialized prefixes, and output proof/currentness.
Any rejection or caught allocation/arithmetic exception leaves A, B, C, the
plan/workspace and the caller's selected-prefix output unchanged. Existing input
proofs, logical values, identities, source versions and storage allocations are
preserved. Output uses the existing GEMM source-version/currentness conventions.
Successful selection reports its prefix without modifying plan/workspace state.
When a result exceeds the original plan's CRT range, create a matching
`RNS8_PLAN_FORCE_FIXED_PREFIX` plan for existing export APIs. Export through the
inadequate original prefix correctly remains a range error. Repeated opt-in
calls can continue within the existing twenty-plane ceiling; insufficient proof,
capacity, or byte budget fails conservatively.

Exact-wide output/input handle aliases
are rejected. Grouped exact-wide tasks also reject cross-task input/output or
output/output aliases. Prepacked B retains its immutable input proof; incremental
output reuse keeps the existing exact identity and dirty-output-region contract.
Sparse raw-plane packing interprets the supplied residues as canonical unsigned
or centered signed integers and records those magnitudes; it cannot recover an
integer value that was already aliased before the caller supplied the residues.

The admission/proof bookkeeping is shared host code. CPU acceptance does not
qualify the HIP grouped, incremental, prepacked, sparse or ordinary kernels.
Private backend-only raw-residue helpers used by benchmark/CRT fixtures are not
public chain admission routes and must not be used to manufacture public proofs.

## INT8 accumulation

Centered residues are signed int8. At modulus 256 the interval is `[-128, 127]`;
for odd modulus m it is `[-floor(m/2), floor(m/2)]`. For any int8 inputs,
`65536 × 128 × 128 = 2^30`, so a block accumulator plus the centered carry fits
int32. Each block must be reduced before accumulating another block. Merely
writing a loop in chunks does not reset its accumulator.

The persistent small HIP path uses `backend_common/residue_dot.hpp`, which is
also exercised against independent CPU oracles. Matrix-engine backends may use
smaller block caps or reject K above their supported cap. They must not silently
execute an unsafe larger accumulation.

## Wrap64 and finite rings

Strict wrap64 is modulo `2^64` arithmetic on separate unsigned byte-limb storage.
Its 36 low-product byte pairs and carry behavior are tested against
Boost.Multiprecision low-64-bit results. Signed INT8 matrix primitives need
explicit unsigned-byte correction; reinterpreting bytes is insufficient. The
CPU fast path uses defined unsigned uint64 wraparound, with a separate byte-pair
oracle retained for differential testing.

Finite-ring U8 accepts modulus 2–256, including composites. Finite-field U8
requires a prime no greater than 251. Canonical outputs lie in `[0, m-1]`.
Finite operations carry no CRT prefix/bound metadata. A modulus must never be
passed to a kernel parameter meaning "number of RNS planes."

## Export and error handling

Checked bounded host export stages device output, reads the error status, and
only then copies to the caller's destination. Every prefix, including 9 and 20,
requires range status. A large CRT range is not a proof that an arbitrary stored
representative meets a smaller bound. An error must preserve every caller output
cell and its padding.

Exact-wide structural status elision is a separate width proof: it applies only
when all possible representatives fit the requested limb width. All-zero output
proofs are also distinct. Neither justifies blanket bounded status elision.

Public C++ enums have fixed uint32 backing so malformed C/FFI values can be
validated without an invalid-enum load. C clients must use the normal 32-bit
enum ABI, not `-fshort-enums`.

Unsupported contracts/targets return explicit errors. Ozaki/FP8, INT4/IU4,
Strassen, and Freivalds execution are unimplemented; reserved research APIs
return `RNS8_UNSUPPORTED_BACKEND`. Deterministic exact APIs do not use
probabilistic verification as a substitute for reconstruction correctness.

## Test map

| Area | Tests |
| --- | --- |
| Ladder, strict prefix boundaries | `tests/unit/test_moduli.cpp`, `test_crt.cpp` |
| Explicit CPU lifting and opt-in selection transactions | `test_exact_wide_lift_cases.inc`, `test_exact_wide_auto_cases.inc` |
| Pack, finite reduction, blocked dot | `test_residues.cpp`, `test_ring_gemm.cpp`, `test_residue_dot.cpp` |
| Full native boundaries and padded layouts | `test_bounded_gemm.cpp`, `test_bounded_reference_sweeps.cpp` |
| Wide limb width/sign/currentness | `test_exact_wide.cpp` and included cases |
| Wrap carry and all 65,536 byte pairs | `test_wrap64.cpp` and included cases |
| ABI, plans, workspace, reuse and cache | `test_api.cpp`, `test_semantics.cpp`, `test_autotune_cache.cpp` |
| Host execution of actual scalar HIP arithmetic | `test_hip_arithmetic_host.cpp` (non-MSVC; not hardware evidence) |
| New pack/CRT/K-block GPU regressions | `tests/differential/test_hip_direct_qualification_cases.inc` |
| Optional accelerator differentials | `tests/differential/test_{hipblaslt,ck,rocwmma,amdgpu_builtins}.cpp` |
| Schema, reports, and promotion guards | `tools/golden_regression_suite.py` |

The qualification regressions include prefix-8 `x=2`, errors in every position
of a 65-cell output, prefix-9/20 range errors, packing tails at 257/258/259 cells,
and K values through 262145. Existing tests retain full-width i64/u64 boundaries,
finite composite moduli, stale storage, grouping, sparse contracts, and wrap
carry checks.

Run the [qualification guide](gpu-qualification.md) on each actual target.
CPU success, schema-valid fixtures, compiled kernel metadata, and skipped tests
are not GPU correctness evidence.
