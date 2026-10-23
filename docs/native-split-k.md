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
K=257/65537. Split 129 exercises more than one merge level. Runtime execution
and performance qualification remain pending; no scheduler default changes.
