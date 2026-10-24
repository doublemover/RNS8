# Exact ragged residue materialization

The private `RaggedResidueLayout` directory stores only the prefix present in
each tile. Tiles use a common default basis and fixed spatial shape, with
per-tile plane counts and checked contiguous payload offsets. Each tile carries
an exact magnitude bound; the directory names a logical value instance and
generation. A caller must supply the matching identity and own the proof that
the directory belongs to the source bytes. This is not yet a public matrix
storage or cache contract.

Every source tile must uniquely represent its values: P > 2*bound for signed
values and P > bound for unsigned values. Every requested destination prefix
must satisfy the same condition. Missing planes cannot be interpreted as zero.
The materializer reconstructs canonical [0,P), applies signed centering only
under the explicit signed contract, checks magnitude, and computes any missing
residues from that integer value. Existing planes may be copied after checking
their centered alphabet. Destination layout can differ from the source tiling.

CPU materialization stages the complete destination and preserves all padding.
The private HIP route borrows the directory, status and staged destination;
the public owner retains these allocations for reuse. The supplied budget covers
all three, including a larger staging capacity retained by an earlier call. Source/destination byte spans cannot
overlap. Metadata, range and kernel failures before commit leave destination
bytes unchanged. `commit_started` becomes true immediately before the final
device copy; a failure then requires the caller to invalidate destination
currentness. This follows the existing fallible-write publication rule.

Private CPU/GPU fixtures cover mixed-prefix neighbors, 140-bit values, stale
identities, insufficient target range and a later GEMM whose K panels cross
source tile boundaries. Registered applicable fixture execution is recorded in the
[dated all-feature validation](local-all-features-20261023.md).
The initial implementation reconstructs every active cell and stages full output;
no conversion cost or speed advantage is asserted. Public proof ownership,
prepared reuse and resident-DAG integration remain
required before exposing this representation as a reusable public matrix.


## Public device-side preparation (authored, execution pending)

The public generation-bound owner now discovers each tile's exact magnitude on
HIP. A 128-thread block reconstructs cells with the current full prefix, checks
the centered alphabet and source proof, then reduces the maximum in shared memory.
Only the directory is copied to the host for checked prefix selection and a
checked offset scan. A second kernel compacts the original device residues and
zeros spatial padding directly into the retained payload. No full residue
snapshot or compact payload upload is needed. The CPU path retains its independent
Boost reconstruction. This is a hybrid directory builder: prefix selection and
layout offsets still run on the host.

The preparation budget covers the compact payload and host directory; HIP adds
one device directory and a four-byte proof status. These device scratch allocations
remain owned until destruction, and are included in that budget. Source readers
remain pinned during discovery and compaction. Failed drains retain unpublished
owners on the poisoned context for cleanup retry, including the source reference;
failed releases preserve each remaining allocation and the owner then accepts
cleanup retry only. Successful preparation ends
the reader reservation but retains the source handle and generation binding.

Authored shared CPU/HIP fixtures cover exact preparation-budget boundaries,
signed/unsigned 140-bit values, zero cells, partial tiles, a forged smaller source
bound, every reconstructed residue and subsequent Boost-checked GEMM. No speedup
or new execution qualification is claimed before the final campaign.


Materialization now uses owner-retained staging, directory and status buffers.
Repeated same-capacity exports allocate no new device buffers through the RNS8
allocator. Resource queries report device metadata and staging capacity separately.
A failed final drain keeps the output writer/reference and all scratch reserved,
poisons further context arithmetic, and requires owner destruction to retry the
drain before releasing anything. Failures after publication starts invalidate the
output; earlier staged failures preserve its previous value. The private borrowed
interface documents the same caller drain obligation. These are authored lifecycle
and reuse changes, with counter assertions queued for the final validation pass.
