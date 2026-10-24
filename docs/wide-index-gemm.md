# Generic 64-bit-index residue control

The private `hip_direct_gemm_rns_wide_device` route uses unsigned 64-bit logical
coordinates, pitches, plane products, K positions and device addresses throughout.
It proves each accessed byte span against host object/difference limits before
launch and rejects output/input overlap. A fixed grid of at most 1024 blocks uses
a 64-bit grid-stride loop, avoiding shape-derived launch narrowing. Accumulation
still reduces every 65536 terms, preserving the full signed-byte INT32 bound.

This is a scalar correctness control with no global scratch. Existing optimized
and public routes retain their own conservative admission limits; this private
candidate does not establish packing/export or public-plan support for large shapes.
Callers own exact range proofs and output invalidation after execution failure.

Authored CPU fixtures prove spans above 32-bit index capacity and reject overflow.
GPU fixtures use all prefixes, padded tails, K-slab boundaries and a valid unused
64-bit pitch. They compare Boost products and preserve output padding/suffix planes.
The unused-pitch case does not exercise actual device memory beyond 2GiB. Large
physical allocations, target performance and public integration remain separate
qualification work. All new execution is pending the consolidated campaign.
