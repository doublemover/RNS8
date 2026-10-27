# One-level rank-23 modular prototype

The private `hip_direct_gemm_rank23_device` evaluates a division-free 3x3 block
bilinear program. Its original-coordinate coefficients come from Stapleton and
Perminov's [public certificate](https://github.com/Joshua-Stapleton/51-addition-alt-basis-3x3-rank23/blob/473661e52ea9d6db7988e038212b4de9f34aa6e1/3x3x3_m23_cr51_alt_basis_certificate.json)
for [arXiv:2609.32388](https://arxiv.org/abs/2609.32388). The source revision is
`473661e52ea9d6db7988e038212b4de9f34aa6e1`; its certificate Git blob is
`7e5d2249242ddeff8dd88ff7da80b4d4e0689d06`.

The original coefficient import independently expanded all 729 integer tensor
coefficients and checked the supplied basis inverses and original/changed matrix
equations. Its header includes a constexpr tensor proof. That existing control
evaluates original-coordinate transforms directly. The distinct SLP candidate
below imports the actual alternative-basis program; neither route substitutes
an addition-count estimate for a measured result.

Each dimension is padded logically to three blocks. Transform kernels mask
missing rows/columns and reduce each sum to a centered byte before calling the
existing direct-HIP GEMM. Every coefficient is -1, 0 or 1: a transform is bounded
by nine byte magnitudes, and recombination reduces after each subproduct. No
inverse or exact-division assumption is needed for composite/even moduli.
Recombination skips output blocks whose coefficient for the current product is
zero. Those blocks retain their staged bytes without product loads or stage
read/reduce/write traffic. Timing impact is unmeasured.

Strassen and rank-23 share templated transform/recombination kernels and the
bounded host scheduler. One transformed A, transformed B and product buffer
are reused for all 23 terms. A staged C accumulates the result; no caller output
write begins until final commit. Scratch admission accounts for all four
buffers and padding. The candidate is one level only and uses ordinary
sequential subproduct launches. It does not yet implement grouped/persistent
subproducts or automatic planner selection. This paragraph describes the
coefficient control; its low scratch use differs from the SLP schedule below.

The shared `query_bilinear_resources` routine returns an explicit status plus
region sizes/offsets and source/destination address spans. A failed query is not
a zero-byte workspace. It qualifies full residue-stack offsets using the
transform/commit kernels' actual 64-bit arithmetic, while retaining 32-bit
qualification for their per-plane dimensions and each transformed sub-GEMM.
This admits the N=16384/prefix-20 address envelope without pretending the
classical full-stack 32-bit GEMM can address it. Resource-query tests check the
large descriptors without allocating them. Actual available device memory and
execution remain separate requirements; no large-shape run is claimed.

The independent coefficient CPU scalar prototype covers every modulus 2-256 against Boost.
Authored coefficient-mutation tests alter each A/B/C coefficient individually.
HIP fixtures compare both bilinear candidates to classical Boost products with
odd shapes, padded strides, full byte extrema, prefixes 1/3/9/20 and subproblem
K above 65536. Zero-scratch admission preserves output canaries. Registered
applicable execution is covered by the [dated validation](local-all-features-20261023.md).
Target ISA/resources and matched end-to-end performance remain separate gates.

The research specification's >=1.15x speedup at N>=16384 and <=2.2x memory gate
still applies to production consideration. Fewer products and an exact tensor
identity are not performance evidence; classical routing remains unchanged.

## Pinned alternative-basis SLP

`tools/data/rank23-certificate.json` retains the complete mathematical
certificate from the commit and Git blob above. Git line-ending normalization
does not change its identity: the importer reconstructs the upstream CRLF bytes
without a final newline and checks the Git blob hash. The authors' Python
verifier is not executed or imported. The certificate's ordered assignments,
including both writes to `u22` and `v12`, are executable data rather than an
invented program derived from the tensor coefficients.

`tools/rank23_slp.py` has a strict assignment grammar and arbitrary-precision
integer expansion. It checks both inverse orders for each basis, the three
original/changed matrix equations, every SLP output coefficient and all 729
integer tensor equations. It counts the source stages as 13 + 12 + 26 = 51
binary additions/subtractions. The full computation also executes one input-A,
three input-B and one output basis operation, for 56 total. Composite and even
byte rings require no division or field assumption.

```powershell
python tools/rank23_slp.py
```

The resulting versioned report binds the pinned source, verifier source digest
and deterministic plans with SHA-256. It labels device execution as `not_run`,
performance as null and
automatic dispatch eligibility as false. It is an algebra/schedule receipt,
not a target qualification record.

Both the Python planner and private C++ `rank23_slp.hpp` convert mutable names to
SSA values. Copies become aliases. Identical binary expressions share one
value; commutative addition uses a stable operand order, while subtraction keeps
its direction. Last uses include every final output, so output values remain
live until the stage ends. The lowest available register slot is reused only
after its operands have been read. The C++ constexpr check expands the allocated
plans against the independently tensor-verified original coefficients. This
checks the actual allocated execution schedule, including the five basis steps.

## Private SLP device schedule

`hip_direct_gemm_rank23_slp_device` is separate from
`hip_direct_gemm_rank23_device`. One fused transform kernel per operand loads
nine logical input blocks, evaluates the complete basis/SLP for each block cell,
reduces every binary result to a centered byte, and writes the 23 transformed
operands. Existing direct-HIP modular GEMM computes 23 sequential subproducts.
One recombination kernel evaluates the shared W program and output basis, then
writes a complete staged C. A final commit publishes the destination bytes;
earlier admission/allocation/transform/subproduct failures preserve C. As with
the existing raw private route, the caller owns lifetime, proof and publication
and must invalidate C after a failure with `commit_started=true`.

The register plan is consumed by these kernels; register allocation, spills,
occupancy and timing have not been measured. Register-slot reuse is local to
each thread. This initial schedule retains all 23 A transforms, 23 B transforms
and 23 products globally. It does not claim global buffer liveness reuse,
grouped/persistent launches, fusion across subproducts or DMA overlap.

`query_rank23_slp_resources` first applies the shared bilinear address admission,
then checks every retained region against `size_t`, `ptrdiff_t` and the supplied
budget. For padded block dimensions `(hm,hn,hk)` and prefix `p`, scratch is
`p * (23*(hm*hk + hk*hn + hm*hn) + m*n)` bytes. Input/destination matrices are
additional live storage. This explicit materialization can exceed the <=2.2x
research memory gate; no planner promotion is inferred from an accepted budget.

Authored Python fixtures check the pinned blob, source/header lowering,
destructive assignment order, malformed statements, individual matrix/basis
and SLP sign mutations, CSE, output liveness and arbitrary-precision products
in every modulus 2-256. Authored C++ fixtures cover allocated-plan expansion,
Boost differentials, full byte extrema, aliased scalar output and exact scratch
boundaries. HIP fixtures exercise all three controls with tails, canary strides,
prefixes 1/3/9/20 and K-block splits above 65536. These new fixtures and the SLP
kernels are authored only, pending the consolidated validation batch. Earlier
dated coefficient-route evidence does not qualify this new route. Large-shape
execution, ISA/resources and matched end-to-end speedup remain open gates.

## October division-free 55-addition DAG

`RNS8_BILINEAR_RANK23_OCTOBER55` executes the pinned
[gsidebottom/logic artifact](https://github.com/gsidebottom/logic/blob/938ea3feddc3ebf44eace7a55222d0292a96c1dd/matmul/external/i19-55adds-slp.txt),
Git blob `f3b618ea06ec620180317e6054b3e901b24a0781`. The unchanged artifact and
Apache-2.0 license/attribution are checked in. It is **not** the September
51-addition alternative-basis program: this DAG has 13 A, 14 B and 28 output
binary additions, 23 ordered products and no basis change or division.

A strict data-only importer emits typed nodes and hashes the artifact, DAG and
allocated schedule. The C++ compiler independently expands the allocated schedule
and statically checks all 729 ordered tensor coefficients against classical 3×3
multiplication, plus the exact 8/6/11 intermediate coefficient L1 bounds. The
separate Python verifier uses immutable nodes and checks the allocated schedule
against them; its execution is deferred. Ordered A×B coefficients preserve the
noncommutative block-ring identity, including even/composite moduli.

The shared liveness compiler supports explicit widened unary negation. Each
binary node reads centered operands into `int`, applies signs/addition, then uses
one conditional modular correction. Negating -128 never happens in `int8_t`.
The existing September DAG also uses this one-correction path; arbitrary input
loads and product reductions retain their general reducers.

Retained October mode evaluates each input DAG once per local cell and writes
23 transformed matrices. `RNS8_BILINEAR_RANK23_OCTOBER55_RECOMPUTE` explicitly
reevaluates the input DAG for each product, retaining only one transformed A/B
pair. Both execute the actual 28-add output DAG over 23 product buffers. The
recompute query removes exactly `22*(A_block_bytes+B_block_bytes)`; it does not
pretend that local register reuse eliminates the remaining global products.
These are distinct arithmetic/launch/memory tradeoffs, not an asserted speedup.

All public bilinear methods now retain scratch in the ordinary workspace, with
optional eager `rns8_prepare_bilinear_workspace`. Repeats with enough capacity
allocate no bilinear scratch. A/B reader, C writer and workspace leases cover the
whole synchronous operation. Precommit failure preserves C; possible-commit
failure invalidates C; an undrainable failure retains the scratch and all leases
until explicit draining succeeds. Checked workspace destruction retains failed
free identities. Successful publication installs residues and the original
complete-expression range proof together, including exceptions during publishing.

Host C++ syntax/constexpr checks passed; generated data and metadata were refreshed.
Regressions for all byte rings, signs, altered coefficients, recomputation budgets,
and public signed/unsigned routes are authored but unexecuted. HIP compilation,
real-device failure paths and matched performance campaigns remain deferred. The
one-level N≥16384 / ≥1.15× / ≤2.2× total-memory promotion gates are unchanged.

### Matched matrix-engine and public measurement controls

Methods 5–9 repeat the same five algorithms with explicitly requested AMDGPU
builtin matrix-engine subproducts. They require a compiled supported target and
block K≤65536; there is no scalar fallback. Ordinary methods 0–4 keep the direct
RNS subproduct baseline. Both bounded signed/unsigned and exact-wide contracts
now use the same original plan/range admission, with per-tile/adaptive plans
rejected before writes. Explicit finite/wrap reinterpretation is not admitted.

Retained September/October matrix variants launch their 23 independent products
as one `[product][plane][tile]` grid. The existing RDNA3/RDNA4 WMMA and CDNA3 MFMA
kernels have a distinct batched instantiation; batch=1 keeps the original
nonbatched specialization. Full per-product and aggregate byte domains are
checked before dispatch, every output belongs to one product/plane/tile, and one
completion replaces 23 serial discovery/launch/completion boundaries. Recompute
mode remains serial because its single A/B pair is overwritten between products.
This changes source-level scheduling; no measured benefit is asserted.

`rns8-bilinear-calibrate` and its independent-process collector compare classical,
Strassen, coefficient rank23, September SLP, October retained and October
recomputed routes through public packing, preparation, product and padded
three-limb export. Cases bind the subproduct backend, explicit/automatic matrix
family, prefix, signedness, residency, budgets and instrumentation. Separate
16×16/32×32/automatic CDNA-family controls enable tuning the classical comparator
without silently moving a candidate into another family cohort. All transforms,
reductions, commits, synchronization and cold cleanup stay in the measured route.

An independent period-15 exact integer oracle checks every output in O(15MN),
including N=16384 cases; it does not infer large correctness from tiny samples.
A post-timing all-plane audit checks residues and every output/padding limb.
Fixtures include odd rectangles, prefix tails, full-byte-ring source regressions,
transformed K=65535/65536/65537 boundaries, negative smaller controls, cold calls,
resident repeats and repacking. Matrix-cap/address/budget failures stay rejected,
not extrapolated into supported large cohorts.

Production processes disable timing hooks; separately labeled diagnostic
processes retain transform/subproduct/recombine/commit events. The collector
binds actual executable/build/process/target/runtime/environment identities,
never mixes diagnostic latency into production medians, and reports paired
complete-route costs and known live payload ratios. It reports allocator/driver
and opaque runtime overhead as excluded—not a proven total-memory measurement.
The ≥1.15×/≤2.2×/N≥16384 gates still require real reviewed evidence and cannot be
satisfied by this source implementation or a logical scratch ratio alone.
