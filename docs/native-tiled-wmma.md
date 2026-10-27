# Native multiwave WMMA candidates

The private `hip_direct_gemm_native_tiled_device` family consumes signed or unsigned
native 64-bit matrices and produces default-basis centered residues. Four waves
own a 32 by 32 output tile. Groups of one, two or four moduli share native loads.
This is a residue producer; callers must separately establish exact range admission.

The register control uses depth 16. Staged variants use depth 16 or 32 and one or
two padded raw-native LDS buffers. A has depth-plus-one row stride; B has 33-column
stride. Encoded fragments remain in registers. The two-buffer implementation loads
the next tile into disjoint storage before current-tile arithmetic, then synchronizes
before reuse. This is software staging, without a claim of asynchronous DMA overlap.

`query_native_tiled_resources` reports exact static LDS bytes and rejects an
insufficient caller budget before device work. The wrapper checks full address
spans, output aliases and compiled RDNA3 capability before setting `commit_started`.
After that boundary a failed call requires the caller to invalidate the output.
There is no public dispatch or default promotion.

Authored differential coverage compares Boost full-width products for every prefix,
both signednesses, all storage/group variants, padded leading dimensions, partial
32 by 32 tiles, extra-plane canaries and K values around 65536. Resource fixtures
cover exact budget boundaries. These additions await the consolidated validation
campaign; no speedup or occupancy result is claimed. Compare each staged mode with
the identical four-wave register control and existing single-wave producer, recording
LDS, registers, spills, full call timing and kernel timing separately.
