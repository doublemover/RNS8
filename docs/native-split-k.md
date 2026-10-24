# Native matrix-engine split-K

The private `hip_direct_gemm_native_wmma_split_k_device` combines the persistent
native WMMA queue with the existing exact modular merge tree. It partitions K
into 1-4096 nonempty fragments, assigns each fragment a disjoint residue-byte
output in scratch, and queues grouped matrix-engine tiles. Every producer still
uses the established 65536-element internal accumulation cap, so a fragment can
itself span multiple safe slabs.

Native WMMA outputs are centered signed bytes. A separate, timed conversion
canonicalizes every fragment before the unsigned merge tree; reinterpreting
negative bytes would be wrong for odd moduli. Each merge has fan-in at most 128
and sums at most 32640 before reduction. Canonical byte outputs feed the next
level, keeping the bound independent of tree depth. The final store centers
residues and preserves destination pitch padding.

Admission checks full native/output address domains and aliases. The explicit
scratch budget covers producer and merge buffers plus the queue descriptors.
Output remains untouched through producer, canonicalization and merge failures;
`commit_started` becomes true only before the final store. A failed store requires
caller invalidation. Scratch is invocation-owned and synchronous; public prepared
or asynchronous integration remains open.

The queue's input collision spans now end at the last accessed element instead
of an unused final pitch. This matters for a K fragment whose pointer starts
inside a native row. A CPU fixture covers valid adjacent scratch and a real
last-element overlap. Authored Boost GPU fixtures compare splits 1/3/129, groups
1/2/4, signed/unsigned full-width inputs, prefixes 3/20, tile/pitch tails and
K=257/65537. Split 129 exercises more than one merge level. Registered
applicable tests are covered by the [dated validation](local-all-features-20261023.md).
The [dense control measurements](7900xtx-large-throughput-2026-10-23.md#conversion-and-scheduling-controls)
record specific configurations; measured selection and public integration remain open.


## Public reusable residue split-K owner

`rns8_query_split_k` is a pure shape/resource query. It reports canonical-byte
partials, merge storage, total scratch and merge levels; failed queries preserve
the caller's report. The explicit split count is 1-4096, cannot exceed K, and must
keep each balanced fragment at or below 65536 products. Every producer reduces
before global storage. Fixed fan-in 128 bounds merge sums by 32640 at every level.

`rns8_create_split_k` retains current direct-HIP exact-wide signed or unsigned
inputs and their context, derives a conservative complete-product range proof,
and allocates the reported scratch once. Repeated `rns8_execute_split_k` calls
reuse it with compatible distinct outputs. The owner fixes source generations;
repacking requires recreation. Duplicate source handles retain one lifetime hold.
There is no native input conversion, CRT export or implicit AUTO selection in
this owner. Existing private native-WMMA split-K remains a separate candidate.

The output is untouched during partial production and merge. Only the final store
begins publication. Any failed final store invalidates the output. Reader and
writer reservations persist if device completion cannot be established, with a
poisoned context; `rns8_destroy_split_k` retries drain/free and preserves the owner
on failure. Ordinary external host exclusion applies to owning API calls.

Authored fixtures cover budgets, hierarchy boundaries, odd host strides, signed
and unsigned full-width values, long-K cancellation, repeated execution and stale
generations with full independent Boost residue comparisons. These additions
have not been compiled or run yet. Adaptive per-tile fragment counts and measured
planner promotion remain separate work.


### Bounded-window execution

The explicit `rns8_query_split_k_windowed` / `rns8_create_split_k_windowed` pair
selects a maximum number of simultaneously stored fragments. Global fragment
boundaries remain defined by total K and total fragment count, including a short
last window. Each window produces locally canonical partials, completes the same
merge hierarchy, and adds its result into one canonical running plane. The carry
is included in the accumulation bound: two canonical bytes sum to at most 510.
There are no global atomics, spin barriers or uninitialized first-window reads.

Scratch consists of `cells * prefix * window` partial bytes, rounded-up groups of
128 merge slots, and one running plane when the window is smaller than the total
fragment count. Resource reports disclose the window count and maximum merge
depth per window. Each boundary drains before scratch reuse. The public output is
stored only after all windows finish, so a missing or failed window cannot publish
a complete value. Equal window and fragment counts use the full-scratch route.

This trades additional launch/synchronization boundaries for lower peak explicit
scratch. It is an explicit user choice, not a measured automatic policy. CPU
resource fixtures and full independent Boost HIP comparisons exercise windows
1, 2, 128, short final windows and multi-level full reductions. Qualification and
matched full/window/unsplit timings remain in the final campaign.
