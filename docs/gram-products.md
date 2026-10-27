# Explicit single-source Gram products

`rns8_gram_residues(ctx, X, C, orientation, method, budget)` computes either
X-transpose times X or X times X-transpose from one resident matrix instance and
generation. It never infers symmetry from two equal pointers. X is packed once;
no separate transpose matrix or native upload is required. C is a distinct dense
matrix, with the same semantics and residue basis/prefix. The first public
subset admits bounded-i64, exact-wide signed and finite ring/field contracts.
Unsigned, complex, wraparound, alpha/beta, async and capture remain unsupported.

The CPU scalar and direct-HIP scalar/WMMA methods share a versioned internal
problem descriptor and checked shape/span/resource admission. Finite field
admission requires prime modulus; finite rings include even composites.
Exact-wide signed admission requires X's current unique range proof and proves
K*maxabs(X)^2 fits C's basis before writes. Off-diagonal results may be negative;
symmetry supplies no range reduction. Full source proof metadata is retained
independently of its payload. Output receives a conservative global bound;
axis-specific metadata is cleared rather than guessed. Signed native packing now
records a verified scalar maximum using an unsigned magnitude scan (including
INT64_MIN), without retaining a host mirror or allocating axis metadata. Bounded
Gram requires that unique source proof and also K*maxabs(X)^2<=INT64_MAX. A
bounded result without a retained proof is rejected; no caller small-bound claim
is promoted to a fact. The added bounded-pack scan is part of conversion cost.

Gram reads the complete source basis. Its source proof must therefore cover that
entire prefix, rather than only a narrower computed prefix inside a larger
allocation. Such a narrowed source rejects both ordinary and prepared Gram
admission until the suffix is explicitly materialized. Plan-specific ordinary
GEMM/export may still use the already-written prefix; Gram does not silently
lift it or treat spare capacity as current data.

For Q=ceil(N/16), checked integer arithmetic constructs Q*(Q+1)/2 jobs, one for
each upper triangular tile. An off-diagonal job stores its tile and the mirrored
transpose. A diagonal job stores each independent upper cell and mirrors it
once; diagonal elements have one writer. The output is published only after
all jobs finish. Scalar code skips dependent cells; WMMA may compute the full
diagonal tile and mask dependent stores. No floating triangular inversion or
cross-job mirror race occurs. The initial mirror writes share the producing
kernel. Explicit methods 2/3 instead launch a padded 16x17-byte LDS transpose
mirror with coalesced lower-triangle stores after the triangular producer.
The second launch is charged; C is not published between the two kernels.
No compact-triangular value is advertised as complete dense storage.

The HIP WMMA path uses signed centered bytes, wave32 replicated fragments and
16-term chunks. Reduction after each chunk bounds the INT32 sum independently
of full K. Tails load zero, and full dense output is written before generation
publication. Device capability is probed before mutation; gfx1100/1101/1102 and
the actual compiler builtin are required. Scalar is a separately selected
control, not a hidden WMMA fallback. Both orientations support padded internal
source/output views; public owned matrices use their compact resident storage.

Budget covers the host job list, its device copy on HIP, and a four-byte WMMA
capability word. Query and launch share admission. CPU uses only the host list.
Output alias, stale private source binding, unsupported mode, invalid range or
insufficient budget rejects before mutation. Issued execution failure invalidates
C; X remains current. The synchronous caller excludes concurrent host mutation
or destruction; existing in-flight readers/writers are respected.

For repeated work, `rns8_prepare_gram` retains X, C and the context, constructs the
triangular job list once and allocates its device storage once. The first
`rns8_execute_prepared_gram` uploads jobs and probes WMMA support; later calls
reuse both. Creation and first-use work belong in cold timings. This owner is
synchronous and does not imply capture or asynchronous completion. Output C is
fixed; input repacking makes execution reject until `rns8_refresh_prepared_gram`
accepts its new generation. Refresh requires the original finite modulus and an
exact bound no larger than the original envelope. A larger envelope requires
recreation. Outputs retain that conservative original proof after refresh.

Execution reserves the input reader and output writer until GPU completion.
Precommit rejection preserves C; issued writes followed by failure invalidate C.
An uncertain drain quarantines the owner, retains reservations and poisons the
context. Destruction retries drain/free and leaves the owner available for retry
on failure; it releases all permanent lifetime holds only after successful
cleanup. Callers must exclude concurrent host mutation and destruction.

Authored tests check triangular single-writer coverage, orders 1/15/16/17/31/32/33,
both orientations, padded host inputs, full signed extrema, composite/prime
finite moduli, K=65537, exact Boost residue outputs, dense symmetry, budget
rejection and reservations. Prepared-owner fixtures additionally cover repeated
execution without GPU allocation, stale generations, refresh, retained lifetimes
and rejection of an expanded proof envelope. Compilation and execution remain pending.
Hardware compilation, regression execution, ISA inspection and measured sustained
profitability remain deferred. No source result qualifies a target or speedup.
The prime-channel five-product experiment is implemented through
`rns8_prepare_gram_five_product`, followed by the ordinary prepared Gram execute,
refresh, info and destroy calls. It executes one level of
[Algorithm2](https://arxiv.org/pdf/2001.04109v4). Each eligible prime channel has
a checked block-diagonal matrix Y with Y*transpose(Y)=-I. Half-K panels are
zero-padded to an even dimension so the actual two-column Y blocks are valid
even for odd logical K. Transforms recenter every intermediate before INT8
products. Original-input range proofs still govern the complete integer output.

Five block products and their explicit recombination execute on the CPU or
direct HIP; composite channels use the original classical triangular product.
No prime-field identity is applied to those channels. Matrix subproducts require
the actual compiled AMDGPU provider, its target/K admission and either the
default RNS basis or finite modulus251. An unavailable matrix provider rejects
without a scalar substitution. The owner retains one transformed operand pair,
five product buffers and a whole-output stage under its hard payload budget.
The info record reports each region and algebraic/classical channel counts.
Replays allocate no algebraic payload; changed inputs require ordinary generation
refresh. Publication, undrainable failure and cleanup use the existing Gram
owner rules. Native/transpose input adapters, reconstruction-once export and
async capture are not implicitly attached to this new explicit recipe.

Independent CPU/Boost fixtures cover every admitted prime, skew certificates,
both orientations, padded inputs, odd dimensions and original full-width values.
HIP fixtures cover mixed prime/composite bases, real matrix subproducts, K-slab
tails, retained lifetimes, budgets and stale-generation rejection. Host and
gfx1100 HIP syntax checks pass; execution and performance gates remain deferred.

## Independent prepared input and output ablations

`rns8_prepare_gram_ex` retains the same symmetry certificate and offers three
explicit input policies: resident centered bytes; retained current signed i64;
and a generation-bound compact transposed residue adapter. Native admission
requires `RNS8_MATRIX_RETAIN_NATIVE_INPUT`; it never reinterprets finite bytes.
The native scalar and actual gfx11 IU8-WMMA kernels cooperatively stage one
16x16 raw panel on diagonal jobs and two on off-diagonal jobs, deriving the
correct distinct A/B fragments from them. Every 16-term slab is reduced modulo
the channel, including masked K tails and INT64_MIN-safe native reduction.
These kernels reload panels per residue plane: cross-plane reuse is not claimed.
Current public packing retains both the RNS proof-bearing materialization and
native storage; this ablation does not claim to eliminate initial RNS packing.

The transpose adapter is allocated once and built during the first execution
following preparation or refresh. Its rows/columns and orientation are inverted
together, preserving the original single-source instance/generation certificate.
It is not rebuilt for unchanged input. Refresh invalidates adapter readiness and
the old output certificate, even if repacked values happen to compare equal.

A nonzero `export_limbs` reserves complete signed limb staging and enables
`rns8_export_prepared_gram_limbs`. It reconstructs N*(N+1)/2 independent cells
once, mirrors each native result, and preserves the full padded destination.
CPU uses the independent wide CRT implementation; HIP uses the wide integer CRT
and checked signed-width guard before any mirror read. Complete private staging
and checked status transfer precede user-visible host stores. Output generation
must match the owner's last successful execution and X's bound generation:
ordinary writes to C or repacking X reject export rather than reusing stale
symmetry. C itself remains a complete dense residue matrix throughout.

`rns8_get_prepared_gram_info` exposes actual retained job, adapter, export and
total payload capacities, source/output generations and independent/dense cell
counts. The shared owner/provider resource planner checks overflow, uint64 limb
alignment after odd byte adapters, status-word alignment, all host/device payload
and CPU residue-gather scratch. Caller matrices, allocator/proof metadata and CPU
arbitrary-precision CRT temporaries are outside that owned-payload budget.
Replay performs no device or owned-payload allocation. Failed drains retain the
owner, export staging and reservations until checked cleanup; output publication
never precedes the complete mirror or export transaction.

The existing [Gram workload guide](gram-prepared-reuse.md) describes matched
native, transpose, separate-mirror and reconstruct-once controls. These are
independent ablations, not one bundled result. They retain the dense prepared
GEMM and full Boost comparators and unchanged-input/generation2 lifecycle.

`integer-cross-product --gram` includes the ordinary dense control and a
single-source scalar candidate, plus WMMA on RDNA3. Both bounded and wide modes
use the public Gram API. The candidate allocates and packs only X (the dataset's
B orientation) and dense C; the dense control still packs both A and B. Each
candidate exports the same full application output, follows the same reuse and
mutation lifecycle, and uses the same independent reference outside production
timing. Reports distinguish independent N*(N+1)/2*K products from dense-equivalent
N*N*K products. The conservative memory envelope still reserves dense workspace;
actual matrix allocation reporting distinguishes the omitted A allocation.
