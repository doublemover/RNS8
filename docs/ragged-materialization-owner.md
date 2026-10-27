# Generation-bound ragged materializations

`rns8_prepare_ragged_materialization` creates an explicit compact representation
of a current signed/unsigned exact-wide matrix on CPU or direct HIP. The owner
retains the source and context, binds the source instance and content generation,
and records the complete immutable ordered basis with its ragged directory. Source mutation makes
materialization fail until the caller prepares a replacement owner. Source and
context destruction return BUSY while the owner exists.

Preparation is a cold conversion. CPU scans source residue planes; HIP discovers
per-tile bounds and compacts payload on device, downloading only the small
bound directory. Both reconstruct unique integers under the actual ordered basis,
reject malformed centered residues and values exceeding the source proof, and
choose the smallest sufficient prefix for each tile. Include preparation cost
when comparing compact routes. Tile dimensions are explicit, independently selected
from a later consumer's GEMM tiles, and limited to 1..128 in each direction.

The directory contains per-tile payload offset, selected prefix and magnitude
bound. Each payload includes full physical tile padding, initialized to zero;
padding never contributes to bounds or logical values. Zero tiles use a proven
one-plane representation. Missing suffix planes are absent information, not
stored numerical zeros. Resource queries disclose payload and directory bytes,
minimum/maximum prefix, tile count and source identity. Logical identity stays
bound to the source through format conversion.

`rns8_materialize_ragged_exact` reconstructs/extends into a distinct ordinary
uniform matrix with the same complete ordered basis using the transactional
CPU/HIP adapter. Default and experimental basis identities cannot be mixed.
Packed materialization recipes can bind these ragged owners on CPU/direct HIP. The destination
may have a different sufficient prefix. All requested planes become valid, so a
subsequent ordinary GEMM can use a different tile geometry without consuming
uncomputed channels. Exact bound metadata is published with the new destination
generation. It is an ordinary value copy into another matrix handle, not identity
reuse between handles. Native input caches on the destination are retired.

Preparation budgets cover the host directory, HIP device directory and proof
status, and compact backend payload. CPU needs directory
plus compact payload. Fixed proof temporaries and allocator bookkeeping are not
included. The retained GPU owner has a host directory and device payload; its
host payload is released after upload. Old live owners remain separate allocations.
Materialization budgets follow the existing adapter: CPU full staged destination;
HIP staged destination, uploaded directory and status word. Inputs and output
storage are caller-owned and excluded from this transient budget.

The output is preserved for invalid/stale/busy operands, insufficient range,
malformed compact storage, resource rejection and precommit backend errors. A
failed HIP final copy invalidates output currentness. Calls are synchronous and
the caller excludes concurrent mutation or destruction of these handles. No
implicit AUTO selection, asynchronous capture or layout-cost promotion is added.

Authored CPU/HIP fixtures cover signed/unsigned inputs, zero and differently
sized per-tile ranges, 3x5 tiles over a 17x19 source, partial edges, repeated
materialization, budgets, BUSY/lifetime holds and stale-source rejection. They
compare every restored residue and then run a 17x19 by 19x7 ordinary GEMM against
an independent Boost integer result. Existing private adapter tests retain
malformed directory, padding, prefix and transactional checks. All execution and
performance qualification remain pending the consolidated final batch.
