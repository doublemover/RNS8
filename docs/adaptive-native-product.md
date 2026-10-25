# Adaptive native product with complete residue output

The private synchronous `hip_direct_gemm_adaptive_native_device` composes fresh
device bound discovery, native grouped RDNA3 WMMA, and exact suffix completion.
Each 16-by-16 output tile selects its strict minimal default-basis prefix using
tile maxima, axis L1 bounds, or K-panel L1 bounds. Proven-zero tiles require no
matrix-engine work. Other tiles compute only their selected planes, with safe
65536-element accumulation slabs and explicit signed or unsigned native inputs.

Before publication, mixed-radix reconstruction fills every missing plane from
the uniquely determined integer. Negative values subtract the complete source
basis product when evaluating suffix residues. A subsequent ordinary residue
GEMM can therefore contract across tile boundaries without treating missing
planes as zero. Reconstruction checks the discovered magnitude bound.

All computation uses private staged storage. Range rejection, insufficient
scratch, unsupported device capability, and failures before final commit leave
the caller's output unchanged. `commit_started` tells the caller when a failed
commit requires invalidation. Input storage must remain immutable through the
synchronous call; bounds are rediscovered on every call, including after
same-address mutation. This is not a public generation-bound proof cache.

Scratch admission includes compact output, bounds, bucket indices and offsets,
status, and peak bound-discovery workspace. The existing qualified i32 address
domains apply. Group widths 1, 2 and 4 are supported. Device discovery, WMMA,
suffix completion and commit have separate timing labels. The original grid path retains slots above a tile's required prefix as a comparison control. The optional compact path below schedules only admitted groups.

Differential fixtures cover signed and unsigned full-width inputs, zero and
nonzero neighboring tiles, all bound methods and group widths, same-address
mutation, output padding, rejection without publication, and a second GEMM
whose contraction crosses precision boundaries. Registered applicable execution
is covered by the [dated validation](local-all-features-20261023.md). Broader target
qualification and performance remain separate work. The public ownership bridge
below is newly authored and is not covered by that earlier execution.


## Compact bounded workers

A nonzero `compact_workers` (1..65535) selects a fixed worker grid over the
existing device prefix buckets. Each worker traverses work ordinals at its grid
stride. A small prefix decoder maps each ordinal to a compact tile index and
channel-group start; proven-zero tiles and absent suffix groups have no work.
The count uses 64-bit arithmetic, including the UINT32_MAX-tile descriptor limit.

This path adds no task array, atomic queue, counter initialization or host count
readback. It uses the same WMMA tile function, exact suffix completion, scratch
budget and staged commit as the original grid. Zero workers preserves that grid
for matched comparison. Static strides can still have load imbalance; worker
count is explicit and no default-route or speedup claim is made.

CPU enumeration fixtures cover exact work coverage, empty buckets, grouping
and large-count boundaries. The signed/unsigned Boost GPU differential and
cross-precision chain fixtures now cover both launch policies. They are authored
but await the consolidated validation pass with the new compiler/dependencies.

## Explicit matrix-owned native inputs

`RNS8_MATRIX_RETAIN_NATIVE_INPUT` opts an exact-wide signed or unsigned direct-HIP
matrix into retaining a compact 64-bit native device copy alongside its residue
storage when packed with `rns8_pack_i64` or `rns8_pack_u64`. Other backends reject
the flag; bounded/wrap/finite semantics and unknown bits reject it. Storage is
allocated lazily on pack and costs an additional `8 * rows * cols` bytes per
retained input. The pack's source leading dimension may include padding; the
retained copy contains only logical cells. Existing matrix memory reporting
includes these bytes. No caller device pointer is borrowed.

`rns8_gemm_adaptive_exact` explicitly invokes device discovery and adaptive
compute using those current native copies. It validates the normal plan,
workspace, matrix backend/device, exact-range proofs and async exclusions before
execution. All bound methods, groups 1/2/4, grid/compact workers and the transient
scratch budget are explicit. Existing matrix storage is outside that transient
budget. Full common-basis residue output and matrix-owned exact range metadata
are published together; downstream ordinary GEMM/export uses the usual API.

Any other value-producing write retires the native copy, including an ordinary
GEMM into a flagged matrix. A later adaptive use rejects until native repacking
refreshes it. Repacking updates both materializations, including same-address
host mutations. The retained buffer's allocated bytes may survive invalidation,
but are never accepted as a current value merely because a pointer exists.
A successful adaptive output also retires its own previous native copy.
Preflight and discovery failures preserve prior output; failed commit invalidates
it. Calls require the existing external host-thread serialization contract.

This is a synchronous experimental API, with no implicit default selection,
summary reuse, graph replay or performance claim. Packing still builds complete
residue inputs as well as retaining native values; that setup cost belongs in
cold/end-to-end measurements. New CPU rejection and signed/unsigned Boost GPU
fixtures cover descriptor admission, padded source packing, native mutation,
staged rejection, full-width tails, export and retirement/repack after ordinary
resident GEMM. They await the consolidated validation batch.

## Shared static preflight

`query_adaptive_native_support` is the internal version-1 owner of argument,
full padded address-span and peak scratch admission. The raw device dispatcher
uses its result before allocation or output writes; the calibrator records the
same resource requirement. The typed result names the failing constraint and
separates an unknown requirement from a known requirement that exceeds budget.
Grid and compact workers share the same peak allocation calculation.

A successful static query is deliberately incomplete runtime admission: its
`requires_device_capability_check` and `requires_data_bound_check` fields stay
true. Actual compiled-target capability and exact input bounds are established
by the existing device path before commit. Pointer non-overlap and public
matrix generation/proof ownership are checked at their respective dispatch
boundaries. The query launches nothing and does not mutate routing or evidence.
CPU fixtures cover exact budget boundaries, malformed parameters, panel-count
limits, padded address overflow and invariant scratch across launch policies;
they await the final batch.

### Single owned native upload

Retained signed and unsigned inputs upload once to the matrix-owned compact
native device allocation. Residue conversion reads that same allocation; it
neither creates a second native upload buffer nor transfers the source again.
Padded host inputs use a temporary compact host vector before this one upload.
The ordinary unretained packing route is unchanged. Successful replacement by
GEMM still retires native currentness, and failed packing cannot publish a
partially converted value. The existing native-to-residue kernel performs the
conversion; matched conversion and end-to-end timings remain pending.

## Versioned distribution and complete-output controls

The native calibrator and collector accept paired `dataset_profile` and
`dataset_seed` controls for capacity-20 `wmma`, `pack_gemm`, `adaptive_grid`,
`adaptive_compact` and `global_minimum` cases, one complete product per call.
The original full-width
MT19937-64/forced-extrema stream remains `full-width-v1`. New deterministic
SplitMix-derived profiles are `uniform-small-v2`, `mixed-tiles-v2` (alternating
16-row/column small and 48-bit magnitudes), `zero-tiles-v2`, and
`localized-outlier-v2` (one 60-bit A/B pair). Signed signs are independently
seeded; a small cancellation result is never used as a range certificate.

Before probing or allocating on the GPU, the executable computes exact axis-L1
summaries from the actual inputs, independently compares multiprecision tile
bounds and strict uniqueness prefixes against the production helpers, and
rejects degenerate advertised bucket/zero/savings distributions. It records all
21 required and executed histogram bins, zero tiles, required/executed channels,
`sum ceil(prefix/group)` work, the fixed-capacity control, a deterministic input
hash and the globally sufficient prefix. Executed adaptive prefixes currently
equal required prefixes; a partially filled group is not rounded up into a
claim of executed channels. Counts model source work, not ISA or speedup.

The collector independently regenerates inputs and exact histograms, validates
all fields, includes its model source in provenance, and can emit a `--plan-only`
preflight without opening the executable. The tracked
80-case `adaptive-distributions.json` corpus pairs all five complete-output routes on
small, tail and 512-square shapes, retaining full-width no-benefit controls and
an unsigned heterogeneous cross-check. Every adaptive call still pays discovery,
compaction, scratch ownership, exact suffix completion, staged commit and sync;
there is no hidden count readback or amortized-away discovery. Existing complete
Boost residue/padding checks and hook-disabled sustained bursts remain intact.

Comparison reports keep input/seed, semantic, shape, group and capacity cohorts
separate, with one sustained complete-call observation per independent process
and diagnostic events retained alongside it. `global_minimum` now provides the
minimum uniform compute prefix admitted by the same fresh device axis-L1 tile
certificates. A device reduction takes their maximum magnitude; the host reads
one 32-byte certificate, rechecks strict prefix selection, and launches the
existing grouped native WMMA producer at that prefix. The existing exact suffix
primitive then completes all requested planes before staged publication. Zero
bounds use an explicit one-plane compute floor. This is a minimum for the
certificate, not an oracle of the smallest range of the actual result.

Every call charges fresh discovery/compaction, reduction and proof readback,
native conversion/product, exact suffix completion, padded commit, all scratch
allocation/release and synchronization. Samples retain these host phase times
for both diagnostic and hook-disabled calls. The proof readback payload is 36
bytes including discovery's rejection word; ordinary capability/status traffic
is additional. The resource query records simultaneous requested scratch
high-water including worst-case retained suffix scratch; it is not an observed
allocator peak. The control borrows immutable native allocations for each
synchronous call and never reuses a certificate after mutation. No persistent
summary owner/cache is introduced.

The independent Boost oracle checks all completed residue planes and canaries,
and independently computed axis-L1 maximum limbs/prefixes validate the device
receipt. Authored CPU query and GPU differential fixtures cover signed/unsigned
extrema, tails, padding, groups 1/2/4, changed inputs, all-zero values, insufficient
capacity/budget and alias rejection. Collector regressions reject missing proof,
incorrect range/prefix, omitted phases and hidden summary reuse. Source and
fixtures await integrated execution; no new result or gain is established.
The harness compares axis-L1 only, so it does not advertise a K-panel-skew
ablation. No new measurements, regression execution or promotion accompanies
these authored fixtures.
