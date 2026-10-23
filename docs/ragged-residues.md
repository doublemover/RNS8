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
The HIP route separately allocates the directory, status and staged destination;
the supplied budget covers all three. Source/destination byte spans cannot
overlap. Metadata, range and kernel failures before commit leave destination
bytes unchanged. `commit_started` becomes true immediately before the final
device copy; a failure then requires the caller to invalidate destination
currentness. This follows the existing fallible-write publication rule.

Private CPU/GPU fixtures cover mixed-prefix neighbors, 140-bit values, stale
identities, insufficient target range and a later GEMM whose K panels cross
source tile boundaries. These fixtures are authored but not executed yet.
The initial implementation reconstructs every active cell and stages full output;
no conversion cost or speed advantage is asserted. Public proof ownership,
device-produced directories, prepared reuse and resident-DAG integration remain
required before exposing this representation as a reusable public matrix.
