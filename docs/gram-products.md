# Explicit single-source Gram products

`rns8_gram_residues(ctx, X, C, orientation, method, budget)` computes either
X-transpose times X or X times X-transpose from one resident matrix instance and
generation. It never infers symmetry from two equal pointers. X is packed once;
no separate transpose matrix or native upload is required. C is a distinct dense
matrix, with the same semantics and residue basis/prefix. The first public
subset admits exact-wide signed and finite ring/field contracts. Bounded signed,
unsigned, complex, wraparound, alpha/beta, async and capture remain unsupported.

The CPU scalar and direct-HIP scalar/WMMA methods share a versioned internal
problem descriptor and checked shape/span/resource admission. Finite field
admission requires prime modulus; finite rings include even composites.
Exact-wide signed admission requires X's current unique range proof and proves
K*maxabs(X)^2 fits C's basis before writes. Off-diagonal results may be negative;
symmetry supplies no range reduction. Full source proof metadata is retained
independently of its payload. Output receives a conservative global bound;
axis-specific metadata is cleared rather than guessed.

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
or destruction; existing in-flight readers/writers are respected. No new lifetime
owner, cache or completion mechanism is introduced.

Authored tests check triangular single-writer coverage, orders 1/15/16/17/31/32/33,
both orientations, padded host inputs, full signed extrema, composite/prime
finite moduli, K=65537, exact Boost residue outputs, dense symmetry, budget
rejection and reservations. Compilation and execution remain pending. Public
workload integration, bounded-i64 admission, longer-K instruction accumulation,
reconstruct-once export, and matched sustained dense-GEMM speedup measurements
remain separate development/qualification work. No performance promotion is
claimed by reducing the number of independent dots.
