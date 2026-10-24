# Public bilinear exact products

`rns8_gemm_bilinear_exact` exposes three explicit one-level direct-HIP methods:
Strassen, the rank23 coefficient control, and the pinned alternative-basis rank23
straight-line program. The existing Strassen research entrypoint delegates to
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
Private coefficient/SLP algebra and resource fixtures remain independent. All
new public routes await the one consolidated qualification and timing campaign.
