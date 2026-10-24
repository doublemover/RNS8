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
qualification, performance and public dispatch remain separate work.


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
