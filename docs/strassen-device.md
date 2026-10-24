# One-level modular Strassen baseline

The private direct-HIP route uses the seven-product division-free Strassen
program. A compile-time exact integer coefficient expansion checks every output
quadrant/input-quadrant monomial. Mutation fixtures alter every coefficient
position and require rejection. The identity therefore applies to the default
basis's composite and even moduli as well as primes.

Each operand is padded conceptually to an even row/column count. Transform
kernels load only valid cells, combine at most two centered bytes, and reduce
back to a centered byte before calling the existing direct-HIP GEMM. Thus the
original GEMM accumulation contract still applies, including K slabs. A separate
single-owner recombination kernel reduces after each product. Both transform
and recombination intermediates are bounded by two centered bytes. Output
padding is preserved by a final logical-cell commit kernel.

Only one transformed A, transformed B and product buffer are live at once;
their storage is reused across seven sequential subproducts. A compact complete
output stage accumulates the recombination, so pre-commit failure preserves C.
The exact scratch budget is prefix*(ceil(M/2)*ceil(K/2) +
ceil(K/2)*ceil(N/2) + ceil(M/2)*ceil(N/2) + M*N) bytes. The private caller owns
integer range proofs and must invalidate C after any failed final commit. The
`commit_started` result exposes that boundary. Aliases and insufficient scratch
are rejected before execution.

Transform-pair, underlying GEMM, recombination and commit event labels remain
separate. This baseline has repeated launches and synchronizations; fewer
algebraic products are not a measured speedup. No recursion or automatic routing
is enabled. The explicit [rank-23 candidate](rank23-device.md) now shares this
bounded transform/recombination scheduler. The existing 32-bit device address admission
still limits supported shapes and prefixes.

Authored GPU fixtures use a Boost classical reference over prefixes 1/3/9/20,
signed extrema, odd output/K tails, K=131073 crossing a transformed GEMM slab,
padding canaries and budget rejection. The [dated all-feature validation](local-all-features-20261023.md) records
registered applicable compilation/execution evidence for these candidates.
Grouped/persistent subproducts, matrix-engine comparisons, final integer export
qualification and the specification's >=1.15x at N>=16384 with <=2.2x memory
gate remain open under #48. Failure to meet those gates retains classical routing.
