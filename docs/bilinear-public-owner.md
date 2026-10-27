# Public bilinear exact products

`rns8_gemm_bilinear_exact` exposes explicit one-level direct-HIP methods:
Strassen, rank23 coefficient and retained/recomputed straight-line programs,
and the mixed-channel rank48 dyadic coefficient program. The existing Strassen research entrypoint delegates to
the same owner and preserves its one-level contract. These methods support
signed/unsigned exact-wide plans with complete current RNS inputs.

Shared public admission checks context, workspace, operand shapes/semantics,
reservations, output aliases, generation capacity and the exact original-input
range proof before invoking a private implementation. Transformed operands use
modular arithmetic; native intermediate overflow or cancellation assumptions
are not used to justify the original integer result. Padding supplies exact
zeros for odd dimensions.

The caller gives an explicit scratch budget. Strassen and coefficient rank23
reuse one transformed operand pair and product buffer. The SLP method retains
all23 transformed pairs and product buffers, plus staged output. Its resource
query counts those allocations; register-level liveness in the pinned program
does not eliminate global buffers. No 2.2x memory or speedup gate is claimed.
The returned ratio is additional transient bytes divided by resident A+B+C
residue bytes and is written only after successful publication.

Private implementations stage the complete result. Early rejection and errors
before the final copy preserve the old public output; a failed final publication
invalidates its currentness. Successful publication retires stale native caches
and commits the exact proof without further allocation. Paired transform launches
are drained even if the second launch fails before temporary buffers are released.

This is explicit opt-in execution with no AUTO change. Authored public fixtures
exercise signed/unsigned full-width inputs, odd dimensions, changed input epochs,
workspace/BUSY/invalid-method preservation and every exported cell against Boost.
Private coefficient/SLP algebra and resource fixtures remain independent. The
[October27 implementation validation](../metadata/implementation-closeout-20261027.json) passes applicable public and private CPU/HIP
regressions. Matched timing and broader-target qualification remain open.
## One-level rank48 dyadic extension

Methods 10/11 add a four-block rank48 coefficient program with direct or explicit
matrix-engine subproducts. The [pinned mathematical matrices](../third_party/rank48/provenance.json)
and their retained license come from the [published construction](https://arxiv.org/abs/2609.12027v2).
`tools/rank48_certificate.py` independently expands all 4,096 exact rational tensor
coefficients. Stored coefficients are twice their rational values. Each odd
modulus uses its exact inverse of two for both input transforms and output
recombination; transformed values are recentered before INT8 GEMM. This includes
odd composite rings. Even channels, including256, execute their original
classical product into the same private stage. The full original basis and
integer uniqueness proof are retained.

One A transform, B transform and product buffer are reused across 48 products;
the complete C stage publishes only after both algebraic and classical channel
work succeeds. Existing retained-workspace, budget, address, lifetime and failed
commit rules apply. Matrix subproducts keep their own K cap and target probe.
The collector accepts `rank48-dyadic` and charges four-block padding, all 48
transform evaluations, even-channel work, reconstruction and final output.
The coefficient implementation remains a separate control. CPU tensor/Boost, altered-coefficient, ring, resource and public HIP
full-width/tail fixtures pass the [October27 implementation validation](../metadata/implementation-closeout-20261027.json), including
actual direct and matrix subproducts on gfx1100. Broader-target execution and
measured routing remain deferred; no default route changes.

## Canonical rank48 straight-line execution

Methods12/13 execute the pinned canonical PLinOpt programs with direct/matrix
subproducts; methods14/15 recompute their input transforms per product to reduce
global operand storage. The strict importer accepts only named arithmetic,
addition/subtraction, negation and division by two. It symbolically expands all
three maps with rational arithmetic and compares them to the independently
tensor-checked coefficients. The source program has200 binary additions or
subtractions and four halvings. Generated live-slot schedules and their hashes
bind the actual evaluation order; all local values are reduced to centered bytes
at every node. Halving is multiplication by the actual odd modulus's inverse
of two. Even planes execute their original classical product into the common
private output stage before publication.

Retained mode holds48 transformed A/B pairs and products. Recompute mode holds
one pair and48 products; its repeated transforms are counted in captures.
Both reserve their full output stage, original integer range proof and explicit
payload budget. The204-operation count describes the canonical logical program;
recomputation and modular reduction add actual work. New CPU and HIP fixtures and
collector controls are implemented; their consolidated execution is pending.
