# Public narrow native products

`rns8_gemm_narrow_exact` is an experimental synchronous direct-HIP operation
with an explicit method and ordinary exact-wide signed or unsigned semantics.
It consumes current matrix-owned native inputs retained by
`RNS8_MATRIX_RETAIN_NATIVE_INPUT`, and publishes a complete persistent RNS
output with the same proof/generation rules as ordinary GEMM.

`RNS8_NARROW_SCALAR_I64` accepts signed exact-wide plans. An actual device scan
finds both input magnitude maxima; a checked `K*max_abs(A)*max_abs(B)` envelope
must fit `INT64_MAX`, bounding every accumulation partial. This deliberately
rejects cancellation-only safety. `RNS8_NARROW_BYTE_WMMA` accepts signed or
unsigned exact-wide plans, scans for actual `[-128,127]` or `[0,255]` membership,
and bounds every partial by `INT32_MAX`. WMMA capability is detected on the
selected compiled device. Unsupported targets report unsupported; there is no
silent scalar replacement, AUTO selection, or cached admission proof.

The independent matrix-owned exact-range proof must also fit the selected CRT
basis. The native result is staged, then encoded into every output residue
plane. Peak transient device payload is `8*M*N+16` bytes for scalar, or
`8*M*N+32` bytes for byte WMMA: the native output overlaps the input scan words.
The explicit scratch budget covers both; resident inputs/output and the caller's
ordinary plan workspace are separate. The implementation does not repack inputs
or export results through the host. Bounds are read back for admission.

Bad shape, prefix, budget, stale native inputs, domain or accumulation bounds
preserve the prior output. Only the encoding launch starts the public write
transaction; a failure there invalidates output currentness. Success retires any
old native copy of C, publishes the current RNS value and exact proof, and allows
ordinary resident chaining/export. Input/output aliases and outstanding write or
output-reader reservations follow the shared public validators.

The authored fixtures cover padded host sources, signed/unsigned byte boundaries,
tails, independent Boost exports, changed domains, stale native copies, and exact
scratch accounting. This source slice has not yet been executed. Existing
private native-output measurements do not qualify this complete public call.
Performance comparisons must include scan, native product, RNS encoding and
allocation, against the same persistent-output contract and input reuse count.

Grouped, asynchronous, fused-epilogue and automatic planner integration remain
separate work. The default resident GEMM route is unchanged.

## Public skinny route identity and width-exact control

Uniform direct-HIP metadata and dispatch now share `direct_rns_route`. Default
compact M*N<=64 still reports the actual persistent-small kernel; other default
uniform calls report grouped-prefix GEMM. They no longer advertise the private
prefix-nine GEMV shortcut. Explicit `RNS8_PLAN_DIRECT_RNS_PARALLEL` admits the
existing K/plane-parallel GEMV implementation at the **selected** prefix for
N=1..8; no forced prefix-nine inflation and no unmeasured default/AUTO promotion.
This flag is synchronous-only until its specific prepared owner is supported.

`RNS8_PLAN_VECTOR_SMALL_N_EXACT_WIDTH` explicitly selects compile-time N=2..8
UInt192 reduction families for K>=512. The original width-eight control remains
the default. Signed sign/magnitude arithmetic, unsigned carry and final native
range/status checks are unchanged. At N=2, declared shared arrays shrink from
51,200 to 12,800 signed bytes and 49,152 to 12,288 unsigned bytes; reduction
columns shrink from eight to N. These are source counts, not occupancy, emitted
ISA or speedup measurements. Both flags are explicit-backend research controls
and are part of plan/cache identity; existing reviewed records cannot silently
qualify a changed route. Host route/metadata and array-contract regressions are
authored, not executed.
