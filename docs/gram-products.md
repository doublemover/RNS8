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

For Q=ceil(N/16), checked integer arithmetic constructs Q*(Q+1)/2 jobs, one for
each upper triangular tile. An off-diagonal job stores its tile and the mirrored
transpose. A diagonal job stores each independent upper cell and mirrors it
once; diagonal elements have one writer. The output is published only after
all jobs finish. Scalar code skips dependent cells; WMMA may compute the full
diagonal tile and mask dependent stores. No floating triangular inversion or
cross-job mirror race occurs. The initial mirror writes share the producing
kernel; coalesced separate-mirror and compact-output variants are not implied.

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
and rejection of an expanded proof envelope. Compilation and execution remain pending. Longer-K instruction accumulation,
reconstruct-once export, and matched sustained dense-GEMM speedup measurements
remain separate development/qualification work. No performance promotion is
claimed by reducing the number of independent dots.

`integer-cross-product --gram` includes the ordinary dense control and a
single-source scalar candidate, plus WMMA on RDNA3. Both bounded and wide modes
use the public Gram API. The candidate allocates and packs only X (the dataset's
B orientation) and dense C; the dense control still packs both A and B. Each
candidate exports the same full application output, follows the same reuse and
mutation lifecycle, and uses the same independent reference outside production
timing. Reports distinguish independent N*(N+1)/2*K products from dense-equivalent
N*N*K products. The conservative memory envelope still reserves dense workspace;
actual matrix allocation reporting distinguishes the omitted A allocation.
