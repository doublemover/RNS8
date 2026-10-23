# One-level rank-23 modular prototype

The private `hip_direct_gemm_rank23_device` evaluates a division-free 3x3 block
bilinear program. Its original-coordinate coefficients come from Stapleton and
Perminov's [public certificate](https://github.com/Joshua-Stapleton/51-addition-alt-basis-3x3-rank23/blob/473661e52ea9d6db7988e038212b4de9f34aa6e1/3x3x3_m23_cr51_alt_basis_certificate.json)
for [arXiv:2609.32388](https://arxiv.org/abs/2609.32388). The source revision is
`473661e52ea9d6db7988e038212b4de9f34aa6e1`; its certificate Git blob is
`7e5d2249242ddeff8dd88ff7da80b4d4e0689d06`.

Import independently expanded all 729 integer tensor coefficients and checked
the supplied basis inverses and original/changed matrix equations. The derived
coefficient header includes its own constexpr tensor proof. Only numerical
coefficient data is imported; the authors' verifier/straight-line code is not
used. This prototype evaluates original-coordinate transforms directly and
does **not** implement or claim the paper's 51-addition alternative-basis kernel.

Each dimension is padded logically to three blocks. Transform kernels mask
missing rows/columns and reduce each sum to a centered byte before calling the
existing direct-HIP GEMM. Every coefficient is -1, 0 or 1: a transform is bounded
by nine byte magnitudes, and recombination reduces after each subproduct. No
inverse or exact-division assumption is needed for composite/even moduli.

Strassen and rank-23 share templated transform/recombination kernels and the
bounded host scheduler. One transformed A, transformed B and product buffer
are reused for all 23 terms. A staged C accumulates the result; no caller output
write begins until final commit. Scratch admission accounts for all four
buffers and padding. The candidate is one level only and uses ordinary
sequential subproduct launches. It does not yet implement grouped/persistent
subproducts, transform-SLP liveness reuse or automatic planner selection.

The shared `query_bilinear_resources` routine returns an explicit status plus
region sizes/offsets and source/destination address spans. A failed query is not
a zero-byte workspace. It qualifies full residue-stack offsets using the
transform/commit kernels' actual 64-bit arithmetic, while retaining 32-bit
qualification for their per-plane dimensions and each transformed sub-GEMM.
This admits the N=16384/prefix-20 address envelope without pretending the
classical full-stack 32-bit GEMM can address it. Resource-query tests check the
large descriptors without allocating them. Actual available device memory and
execution remain separate requirements; no large-shape run is claimed.

The independent CPU scalar prototype covers every modulus 2-256 against Boost.
Authored coefficient-mutation tests alter each A/B/C coefficient individually.
HIP fixtures compare both bilinear candidates to classical Boost products with
odd shapes, padded strides, full byte extrema, prefixes 1/3/9/20 and subproblem
K above 65536. Zero-scratch admission preserves output canaries. Runtime
execution, ISA/resources and matched end-to-end performance remain pending.

The research specification's >=1.15x speedup at N>=16384 and <=2.2x memory gate
still applies to production consideration. Fewer products and an exact tensor
identity are not performance evidence; classical routing remains unchanged.
