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

This fresh-scan API is synchronous and experimental, with no implicit default
selection, summary reuse, graph replay or performance claim. Packing still builds complete
residue inputs as well as retaining native values; that setup cost belongs in
cold/end-to-end measurements. New CPU rejection and signed/unsigned Boost GPU
fixtures cover descriptor admission, padded source packing, native mutation,
staged rejection, full-width tails, export and retirement/repack after ordinary
resident GEMM. They await the consolidated validation batch.

## Workspace-owned summary reuse

`rns8_gemm_adaptive_exact_cached` explicitly retains one A/B summary pair in the
existing workspace. It supports tile maxima, axis L1 and panel L1 using the same
scan and bound arithmetic as the fresh control. Each side binds the matrix's
nonrecycled instance, internal content generation, current owned native pointer,
explicit signedness, device, shape/stride, axis role and normalized discovery
policy. A source-version assertion or pointer alone never admits reuse. Repacking
one operand refreshes only that side; changing method or panel depth invalidates
the affected certificates before overwriting or replacing their storage.

Every call validates live native materializations and ordinary exact proofs,
then rebuilds tile bounds, strict prefix selection and buckets. An ordinary
GEMM that retires native currentness rejects cached use until native repacking.
The workspace retains only summary payload and identity stamps; it never
dereferences an old matrix or native pointer. A replacement matrix's different
instance forces a fresh scan. Staging, suffix completion and publication keep
their existing contract. A synchronous operation lease pins operands/workspace
and quarantines them if a failed drain cannot establish completion.

`max_scratch_bytes` includes the retained pair and all transient adaptive payload,
excluding pre-existing ordinary matrix/workspace storage and allocator overhead.
The retained pair replaces the fresh discovery summary allocation. Policy
replacement frees old storage before allocating its replacement. The old payload
must itself fit the requested budget. A fresh control on a workspace containing
a cache charges both payloads. Workspace destruction uses retryable device owners.

`rns8_get_adaptive_summary_info` reports retained requested bytes, last successfully
scanned instance/generation stamps, and per-side completed scan/reuse counters.
Counters saturate at UINT64_MAX. Observation is unavailable while in flight;
recorded stamps do not certify a matrix mutated after the observed call.

The public narrow-route collector accepts `adaptive-fresh` and `adaptive-cached`
for a retained-input ablation. Both use scalar-first host admission, axis L1,
group 2 and 32 compact workers, with complete RNS output and the same padded
three-limb host adapter. Setup, first call, steady latency and sustained bursts
stay separate; the oracle stays outside timing. Counters must prove one initial
scan per side and reuse on every later completed cached call. The affine-small
fixture isolates summary reuse and does not replace heterogeneous distributions.
The focused shortlist expands to `temp/benchmark-cases/current/narrow-public/adaptive-summaries/`.

The slice passes the complete local batch at compiled source **4e21e9c**:
**556 sanitizer passes / 23 expected skips**, **942 GPU passes / 16 expected
skips**, zero failures, **12 installed-consumer checks** and two new installed
C17 static/shared ABI probes. New GPU differentials cover every method and
signedness, selective A/B mutation, policy changes, replacement matrix instances,
budget rejection, complete exports and native retirement/repack.

### Matched summary reuse observations

All **40 independent-process captures** pass with the same clean `4e21e9c`
executable; collector/profile packaging is `4f04ea3`. Five processes per policy
and workload use three warmups, nine latency samples and three bursts reaching
at least 16 calls and 250 ms. Sustained process means weight completed duration
by calls; the table reports their medians and observed process ranges. Both
routes include the complete public product and padded three-limb host export,
with required proof/status/synchronization inside and independent oracles outside.

| Shape / signedness, P20 | Fresh scan, ms | Cached summaries, ms | Qualification |
|---|---:|---:|---|
| 17×19×31 signed | 2.318 (2.242–2.505) | 2.132 (2.079–2.181) | Baseline process spread 11.32% exceeds fixed 10% gate |
| 512³ signed | 9.179 (8.616–9.325) | 9.043 (8.522–9.404) | Ranges overlap |
| 17×19×65537 signed | 23.727 (22.709–24.269) | 21.935 (21.347–23.328) | Ranges overlap |
| 128×3×65537 unsigned | 26.040 (22.284–29.600) | 24.053 (20.058–28.123) | Ranges overlap; spread gate fails |

Median improvements are 1.5–8.7%, but **none passes both predeclared range
separation and process-spread gates**. No gain or automatic route is promoted.
The 512³ useful integer rates are 0.02924 versus 0.02968 TOPS for this complete
exact-wide host-output boundary. First-call fresh/cached medians are respectively
3.677/3.285, 12.679/16.420, 28.187/29.688 and 26.559/26.710 ms in table order;
they remain separate from steady observations and setup.

Every cached process records exactly one scan per operand, with reuse on every
later call: 61–375 completed calls per process. Retained summary payload is
96 bytes for both 17-row cases, 1536 bytes at 512³ and 216 bytes for the skinny
unsigned case. These observations establish actual scan elimination, independently
of the inconclusive elapsed benefit. The
[qualification receipt](../metadata/adaptive-summary-qualification-20261025.json)
preserves all samples, counters, identities, capture hashes and full validation.

### Separate resident-output boundary

The authored resident shortlist in
`temp/benchmark-cases/current/narrow-public/adaptive-summaries-resident/` measures completed public
products with full P20 RNS output retained on device. `adaptive-resident-fresh`
and `adaptive-resident-cached` share the same native/RNS input owners and scalar-first
proof policy. `rns-resident-public`, `wmma-resident-fused` and the signed
`scalar-resident-fused` control provide alternative public producers at the same
complete-output boundary. The 19 cases cover the same four workloads; unsigned
scalar-native lowering remains unsupported and is not fabricated as a control.

Required input proof, status, suffix completion, commit and synchronization stay
inside every measured call. Per-call host export is absent from this boundary;
the benchmark performs independent padded limb export/oracle checks after the
timed interval and checks all resident residue planes. Timed export must be
exactly zero in the receipt, and explicit boundary/proof-policy fields prevent
pooling these samples with the completed host-output study. Setup/first-call and
steady/burst observations stay separate.

All **95/95 captures** pass at clean compiled source **3532e57**, with five fresh
processes per route/workload. Both GPU and sanitizer targets compile, and all six
collector regressions pass. Production library/kernel sources are unchanged from
the complete `4e21e9c` qualification. Every sustained burst reaches its declared
250 ms minimum. The table reports medians of duration-weighted process means;
all columns return the same complete P20 RNS output.

| Shape / semantics | Fixed RNS, ms | Narrow scalar fused, ms | Narrow WMMA fused, ms | Adaptive fresh, ms | Adaptive cached, ms |
|---|---:|---:|---:|---:|---:|
| 17×19×31 signed | **0.081** | 0.331 | 0.346 | 1.053 | 0.946 |
| 512³ signed | 1.817 | 0.738 | **0.573** | 5.784 | 5.704 |
| 17×19×65537 signed | **3.087** | 12.018 | 22.667 | 20.569 | 18.913 |
| 128×3×65537 unsigned | **3.346** | Unsupported | 19.669 | 22.223 | 20.844 |

At 512³, fused narrow WMMA beats the next-fastest measured control, fused narrow
scalar, by **1.288×**: 0.737890 versus 0.572776 ms, with disjoint process ranges
0.728450–0.745404 and 0.568429–0.573541 ms. Both satisfy the fixed 10% spread gate.
Useful integer throughput is **0.36379 versus 0.46866 TOPS**. Against fixed RNS,
the same winner is 3.172× faster; the stronger scalar comparison is the headline.
These low-magnitude, resident-input/output observations do not select AUTO.

Context/plan/matrix/packing setup and first call remain separate. The observed
512³ WMMA setup is about 541 ms, versus 519 ms for scalar; setup is material.
Per-process diagnostic projections from recorded setup + first call + remaining
steady calls give scalar/WMMA median ratios **0.960× at one call, 0.970× at 32,
0.998× at 128 and 1.128× at 1024**. The 1024-call totals project to 1.273 versus
1.128 s. These are projections, not separately timed fixed-length batches.

Summary reuse itself passes the fixed elapsed gates within the adaptive route
at 17×19×31 (**1.113×**) and signed 17×19×65537 (**1.088×**). Fixed RNS remains
much faster on both workloads, so these do not establish adaptive routing wins.
The 512³ reuse ranges overlap; the unsigned long-K reuse spread fails. All
controls and losses remain in the receipt rather than being retired by intuition.

The exact measured executable contains the inspected original device code object.
Its selected signed/unsigned fused native-byte kernels contain actual integer
`v_wmma_i32_16x16x16_iu8`, use 49 VGPR / 54 SGPR, wave32 and zero LDS/scratch/spills.
The fused scalar kernel has no matrix instruction, uses 17 VGPR / 18 SGPR and
zero LDS/scratch/spills. These are binary static resources, not achieved dynamic
occupancy or profiler instruction/traffic totals. The
[resident qualification](../metadata/adaptive-resident-qualification-20261025.json)
preserves all 95 capture hashes/samples, producer identities, retained-summary
counters, setup projections and the exact artifact-bound instruction evidence.

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
high-water as the maximum discovery/production phase. Discovery records retire
before production; suffix scratch is allocated only when required, at the
selected prefix, after the native producer's capability word retires. The query
reserves a conservative worst-case suffix ceiling; it is not an observed
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
