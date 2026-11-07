# Public narrow native products

Current execution results and remaining qualification are owned by the
[current roadmap](roadmap.md#current-validation-and-remaining-work).

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
scratch bound is not an observed allocation counter. First/final public summary
queries run outside the production intervals. Retained and repacked cases
validate the actual input identities, completed proof counts and requested
device bytes. Cold workspaces are destroyed inside each completed interval;
their snapshots are explicitly unobserved, rather than queried inside timing
or inferred from the scratch bound. This keeps source-current
owner calls distinct from earlier always-scanned captures and private native
output controls; no prior timing result qualifies this changed execution path.
One additional complete public call enables native diagnostics separately from
all production samples and sustained intervals, then verifies its complete
integer output. It records actual allocation/free deltas, completed synchronous
copy-wrapper counts/bytes, and per-label event counts/durations. Cold diagnostics
use a separate observer context while the measured owner creates and destroys
its own context normally. A separate post-timing product audits residue planes
when the cold owner has already been destroyed. Label durations can overlap and
are not summed into an end-to-end latency; asynchronous transfers outside those
wrappers are not relabelled as measured traffic.

The actual compiled native WMMA probe and its context capability owner are now
shared with block-sparse preparation. The cache binds the target/device and
loaded provider, publishes only successful observations, and retains an explicit
unsupported result. Poisoned contexts and mismatched providers/targets cannot
reuse it. Certified public byte execution observes it once per context; the
private checked byte helper still probes each raw-pointer call. Public device
imports without a host-pack certificate scan each changed input once into a
workspace-owned certificate. Unchanged A and B proofs are reused independently;
identical input handles can share the completed proof. Both changed keys are
retired before scanning starts, so a failed scan cannot revive an older value.
Each call revalidates the complete engine identity and recomputes its K-dependent
partial bound. Caller source versions and a prior output bound confer no reuse
authority. Exhausted generations cannot execute this route.

The scan computes exact signed/unsigned min and max with bounded grid-stride
work, a compact-input specialization, and one retained 16-byte device buffer
shared by A/B scans. Its single compound readback uses an owned destination;
pending scratch and all input/output/workspace leases survive an undrainable
error. Certificates publish only after completed scans. The authored differential
compares independent CPU extrema for narrow/full-width values, poisoned padding
and a forced extreme beyond the first grid span. Public import/host-pack controls
cover changed domains, separate A/B reuse and warm allocation/copy counters.
Runtime validation remains deferred.

`rns8_get_narrow_summary_info` exposes completed device scans, host-proof
derivations, reuses, retained device bytes and the last observed WMMA capability
snapshot without GPU work. Counts saturate; they describe logical proof work,
not kernel launches. Keys describe the cached certificates, which may be older
than a subsequently mutated input; execution always validates them again.
Capability support is meaningful only when the observation flag is set.
Malformed headers, nonzero reserved fields and busy workspaces preserve the
caller output. These observations do not grant execution authority.

The independent matrix-owned exact-range proof must also fit the selected CRT
basis. A staged native result is encoded into every output residue plane.
The conservative public device payload budget is `8*M*N+16` bytes for scalar, or
`8*M*N+32` bytes for byte WMMA.
The explicit scratch budget covers both; resident inputs/output and the caller's
ordinary plan workspace are separate. A certified staged call instead retains
`8*M*N` native product bytes in its workspace and reuses them; a certified fused
call needs no product allocation. A device-import cache retains another 16 bytes
for scans; a host-pack-only cache needs none. Switching to fused execution
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

Asynchronous specialization and automatic promotion remain separate work.
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
qualify a changed route. Host route/metadata and array-contract regressions are registered.

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

## Explicit signed hierarchical split-K

`RNS8_NARROW_SPLIT_K_I64_FUSED_RNS` is a separate synchronous signed exact-wide
candidate. One 256-thread block reduces each output's 4096-element K slab;
a second launch merges the signed native partials and encodes the result into
every selected RNS plane. It uses no cross-block atomic accumulation. The same
full-K `K*max_abs(A)*max_abs(B) <= INT64_MAX` certificate bounds each product,
lane sum, block-tree sum and final merge, including cancellation-heavy inputs.
Unsigned and byte-WMMA semantics are not inferred from its method number.

The conservative scratch query charges
`8*M*N*ceil(K/4096)+16` bytes and checks the full partial-plane address domain.
The workspace retains the native partial allocation across unchanged calls;
generation-bound A/B proof reuse remains independent. Failed partial production
preserves C; the output transaction begins at merge submission. Uncertain
partials remain leased until completion is observed. Switching methods releases
or replaces inactive storage before submitting new work.

Operation queries, explicit pipeline recipes and analytical stage kind 15 carry
this method's own resource and cost identity. Ordinary GEMM/WMMA rate models do
not price its complete-call cost. The public narrow calibrator accepts
`scalar-split-k-fused` and `scalar-split-k-resident-fused`, with retained partial
bytes and separate partial/merge native diagnostic labels. Both retain full
Boost/output/residue checks outside production intervals. Authored controls
cover slab tails, K=65535/65536/65537, stale proofs, poisoned padding, exact
budget boundaries and warm allocation reuse. Only source/frontend checks have
run; target execution, ISA/resource inspection and speedup remain unmeasured.

## Independent grouped narrow execution

`rns8_gemm_narrow_exact_grouped` batches 2..4096 same-plan tasks with explicit
fused scalar, byte-WMMA or signed split-K methods. It accepts the existing
`rns8_grouped_gemm_task` descriptors. Tasks may share either input; destinations
and workspaces must be distinct, and no destination may be any task's input.
This surface represents independent products, not an implicitly ordered graph.
Staged single-product methods report unsupported rather than executing a loop
of single calls under a grouped label.

Every task passes ordinary context/plan/workspace/currentness checks, its own
exact CRT range proof and the current native domain/partial-sum admission before
any output launch. The whole batch holds synchronous input/output/workspace
leases. The scalar and byte-WMMA paths launch their existing arithmetic bodies
with a device task directory; split-K batches partial production and merge in
two launches. Each task's original destination is preserved. Grouped WMMA uses
the same compiled capability and byte-signedness rules as the single route.

The complete device budget covers the sum of the conservative per-task scratch
and one four-pointer directory entry per task. The directory lives in the first
task's workspace; its unchanged pointer inventory avoids repeated uploads.
That inventory grants no numerical authority: input generation/domain proofs
are validated on every call. Owned host upload bytes and all device buffers
remain leased after uncertain failure. Inactive group directories and mismatched
partial storage are released before replacement scans can allocate.
`rns8_get_narrow_summary_info` reports retained group-directory bytes and cached
task count separately from product/scan bytes. A generic GROUP operation query
has no task-count field, so its resource result remains a per-task lower bound;
the actual grouped entrypoint enforces the full checked batch budget.

A failed admission preserves every C. Partial-only failure also preserves every
C; a failed grouped output/merge launch invalidates all destinations. A failed
drain keeps each workspace and destination reserved until observed completion,
without publishing partial results. Authored CPU resource/descriptor and public
Boost/HIP controls cover shared inputs, distinct outputs, cache reuse, invalid
last-task domains, duplicate output/workspace rejection and failed-drain custody.
They have not executed, and no grouped latency or speedup is claimed.
The [grouped calibration source](narrow-group-calibration.md) provides matched
single/group/RNS cohorts with bounded raw acquisition and complete task oracles.

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
the UInt192 control; those are not byte-domain narrow workloads.
Current local execution follows the roadmap; target-specific ISA/resource and
performance qualification remain separate gates.
