# Public narrow native products

`rns8_gemm_narrow_exact` is an experimental synchronous direct-HIP operation
with an explicit method and ordinary exact-wide signed or unsigned semantics.
It consumes current matrix-owned native inputs retained by
`RNS8_MATRIX_RETAIN_NATIVE_INPUT`, and publishes a complete persistent RNS
output with the same proof/generation rules as ordinary GEMM.

`RNS8_NARROW_SCALAR_I64` accepts signed exact-wide plans. Certified input extrema
or an actual device scan supply maxima; a checked `K*max_abs(A)*max_abs(B)` envelope
must fit `INT64_MAX`, bounding every accumulation partial. This deliberately
rejects cancellation-only safety. `RNS8_NARROW_BYTE_WMMA` accepts signed or
unsigned exact-wide plans, proves actual `[-128,127]` or `[0,255]` membership,
and bounds every partial by `INT32_MAX`. WMMA capability is detected on the
selected compiled device. Unsupported targets report unsupported; there is no
silent scalar replacement or AUTO selection.

The current source frontload retains typed native min/max metadata from the
existing exact host-pack scan, without another input traversal. It binds that
metadata after successful publication to the same complete instance/generation,
device, owned native pointer, shape/stride and signedness identity used by
adaptive summaries. Publication and native invalidation retire it. Signed
`-128` and `+128` remain distinct, and `INT64_MIN`/`UINT64_MAX` extrema use
unsigned-safe magnitude arithmetic. Public execution consumes matching proofs
under the existing input/output/workspace lease, avoiding data scans/readback.
It recomputes the partial-sum admission for the current K and accumulator width;
the independent CRT bound cannot replace this check. Portable extrema, cap±1,
padding, identity and real-HIP warm counters/exact-output controls are authored;
runtime validation remains deferred.

Public narrow microcase version2 records the host-pack certificate policy and
fused versus retained staged-product storage explicitly. Its conservative
scratch bound is not an observed allocation counter. This keeps source-current
owner calls distinct from earlier always-scanned captures and private native
output controls; no prior timing result qualifies this changed execution path.

The actual compiled native WMMA probe and its context capability owner are now
shared with block-sparse preparation. The cache binds the target/device and
loaded provider, publishes only successful observations, and retains an explicit
unsupported result. Poisoned contexts and mismatched providers/targets cannot
reuse it. Certified public byte execution observes it once per context; the
private checked byte helper still probes each raw-pointer call. Device imports
without a host-pack domain certificate retain the checked device-scan route.
Generation-bound device summary caching for those imports is still in progress.

The independent matrix-owned exact-range proof must also fit the selected CRT
basis. The native result is staged, then encoded into every output residue
plane. Peak transient device payload is `8*M*N+16` bytes for scalar, or
`8*M*N+32` bytes for byte WMMA: the native output overlaps the input scan words.
The explicit scratch budget covers both; resident inputs/output and the caller's
ordinary plan workspace are separate. A certified staged call instead retains
`8*M*N` native product bytes in its workspace and reuses them; a certified fused
call needs no product allocation. Switching to fused or untrusted-input execution
releases an inactive cached product before issuing work, avoiding hidden overlap.
The conservative public scratch budget is unchanged. The implementation does not
repack inputs or export results through the host. Uncertified admission reads
bounds back; certified admission uses the validated extrema.

Bad shape, prefix, budget, stale native inputs, domain or accumulation bounds
preserve the prior output. A staged call starts the public write at encoding;
a fused call starts it at the product launch. Failure after that point invalidates
output currentness. Success retires any
old native copy of C, publishes the current RNS value and exact proof, and allows
ordinary resident chaining/export. Input/output aliases and outstanding write or
output-reader reservations follow the shared public validators.

The authored fixtures cover padded host sources, signed/unsigned byte boundaries,
tails, independent Boost exports, changed domains, stale native copies, and exact
scratch accounting. This source slice has not yet been executed. Existing
private native-output measurements do not qualify this complete public call.
Performance comparisons must include scan, native product, RNS encoding and
allocation, against the same persistent-output contract and input reuse count.

Grouped, asynchronous and automatic planner integration remain separate work.
The explicit fused variants are described below. The default resident GEMM route
is unchanged.

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

## Explicit fused native-accumulator to RNS variants

`RNS8_NARROW_SCALAR_FUSED_RNS` and `RNS8_NARROW_BYTE_WMMA_FUSED_RNS`
reuse the same actual-device input scan and absolute-partial-sum admission as
the staged variants. Each already-exact accumulator is centered directly into
the selected plane-major output; native-only private controls are unchanged.
The complete-expression CRT proof remains independent. Scan/domain rejection
preserves output; launch failure invalidates it. Context-owned synchronous
leases retain all borrowed A/B/C/workspace storage after an undrainable error.
`rns8_drain_failed_work` releases those leases only after a successful drain and
never publishes C or restores a poisoned context's execution authority.

Fused scratch is the 16-byte scalar or 32-byte byte-WMMA scan state, rather than
that state plus an 8*M*N native product. The separate encoding launch and native
product write/read are absent in source; required RNS stores remain. This is not
a claim about HBM counters, register pressure, crossover or latency. Staged
variants remain selectable.

The public narrow calibrator compares staged/fused scalar and WMMA, actual
ordinary public RNS, explicit selected-prefix parallel RNS, and production
UInt192 width8/exact-N controls. All routes publish a padded three-limb integer
output. The vector control explicitly includes public native64 export and host
sign/zero extension; it is not relabelled as native 192-bit arithmetic output.
Minimal-prefix and fixed-prefix cases, cold ownership, resident/repacked inputs,
slab/tail/64-cell boundaries and N=1/9 controls are separate fixtures. Full exact
outputs and padding are checked outside observations; a post-timing public
product audits every RNS plane. Counterbalanced collectors bind independent
process/artifact/build/environment cohorts and cannot promote AUTO. Signed
full-width cancellation and unsigned overflow-status fixtures separately cover
the UInt192 control; those are not byte-domain narrow workloads. All execution,
device compilation, ISA/resources and performance qualification remain deferred.
