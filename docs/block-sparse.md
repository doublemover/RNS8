# Exact block-sparse research baseline

The private block-sparse packer detects zeros in native signed or unsigned
64-bit values. A block whose stored residues are all zero remains present when
its native values are nonzero. This distinction is necessary before treating
an omitted block as an exact zero.

LHS storage uses CSR over 16-row tiles with 16-by-64 blocks; RHS storage uses
CSC over 16-column tiles with 64-by-16 blocks. Each entry contains contiguous
1024-byte planes for the selected default-basis prefix. Sorted unique K-block
indices, layout/basis versions, dimensions, prefix and logical value identity
are validated before consumption. Native packing stages both metadata and
payload before publishing them and enforces a metadata-plus-payload budget.

The HIP baseline intersects the two sorted K lists. One 256-thread block owns
each 16-by-16 output tile and modulus. It loads matching operand blocks into
LDS, accumulates at most 64 products before modular reduction, and writes each
output once. The accumulator bound is 64*128*128 plus a carried residue, so
correctness does not depend on cancellation. Missing intersections produce zero;
K and output tails are clipped. No atomics or completion counters are needed.

This is a scalar LDS kernel, not a sparse matrix-instruction or 2:4 route. The
private caller owns integer-range admission, device payload lifetime and output
publication after a runtime failure. Descriptor generation fields are checked
against caller-supplied identities; they are not a public proof owner.

Authored tests compare CPU materialization and actual HIP products with exact
Boost references, including full-width native values, all two-block presence
combinations, empty structures, padded leading dimensions, K tails, stale
generations and insufficient budgets. Registered applicable compilation/test evidence is recorded in the
[October 23 all-feature validation](local-all-features-20261023.md).
## Public generation-bound owner

`rns8_prepare_block_sparse_operand` creates a direct-HIP
exact-wide signed/unsigned handle from a current retained-native matrix. Explicit
LHS/RHS roles select the layouts above. Preparation downloads the native snapshot
for exact-zero discovery and uploads the packed payload and immutable CSR/CSC
directory; this cold cost is part
of any reuse/break-even comparison. The handle retains its source matrix and
context lifetimes. Repacking the source is legal, but makes the old handle stale;
`rns8_gemm_block_sparse_exact` rejects it before changing C. Prepare a replacement
handle to use the new generation. Destruction/use require external host exclusion.

Construction retains the source's read lease and the complete unpublished owner
through download, packing, upload and publication. Failed drains keep host source
vectors and device payloads on the context's existing retry cleanup path. Failed
destruction preserves the handle and its source/context references for retry;
cleanup-started handles cannot execute. These ownership controls are authored
and await the final validation batch.

The preparation budget covers the native snapshot, packed host/device bytes,
actual host directory capacities, retained device directories, owner/retry
objects and conservative probe scratch. The admitted upper bound is checked
before the retained device allocations. Allocator/driver rounding and shared
context/runtime/TLS storage are excluded. `rns8_get_block_sparse_storage_info`
reports these requests/capacities and the preparation bound without GPU work.
Preparation failure leaves the caller's output handle untouched. Existing handles
are separate retained allocations and must be added when budgeting replacement.
Warm execution needs zero transient directory scratch; its scratch cap may be
zero. Retained payloads/directories, dense RNS inputs/output and ordinary
workspace remain separate resources. The private unprepared baseline still
includes metadata allocation and upload on each call.

Execution uses the existing scalar intersection kernel after shared public
shape/semantic/reservation and exact-range proof checks. Successful output is a
complete ordinary persistent RNS matrix. Prepared directory admission checks
device, extents, role, basis, prefix and source generation without rescanning
immutable host indices. Preflight failures preserve C; failure after its kernel launch invalidates C. Source instance,
generation, role, prefix and context must all match. There is no AUTO promotion. The original entrypoint remains the scalar control;
the explicit method entrypoint below adds matrix-engine execution. Source matrices must remain current, even though
the handle has an independent packed payload.

New public fixtures cover full-width signed/unsigned values, tails, Boost export,
source lifetime, stale generation rejection, empty sparse payloads and output
zero-scratch warm calls and preservation on BUSY/stale errors. These additions are authored and unexecuted;
the older private baseline receipt above does not qualify this owner. The authored
WMMA/hybrid extension below also needs fresh qualification and end-to-end
break-even measurements under issue #51.


## Explicit WMMA and hybrid tile dispatch

`rns8_gemm_block_sparse_exact_with_method` adds two opt-in implementations beside
the unchanged scalar control. WMMA uses one full wave per 16x16 output tile and
four signed INT8 matrix operations per intersecting 64-wide K block. It reduces
every accumulator after each block, so every partial is bounded by
64*128*128 plus a carried residue. Packed zero tails are consumed as zeros and
stores mask logical output tails. The current target guard covers gfx1100/1101/
1102 with the required compiler builtin; other targets reject before output writes.

HYBRID counts the exact intersection of the prepared CSR/CSC block lists for
each output tile. It runs the scalar path at or below the explicit caller-chosen
intersection threshold and the WMMA path above it. The decision is uniform across
the workgroup before scalar barriers. This is a work-count dispatch experiment,
not a calibrated universal crossover or pruning of small nonzero values. Setting
threshold zero or UINT32_MAX supplies useful all-nonempty-WMMA/all-scalar controls.
Non-hybrid methods require threshold zero. Both accelerated modes positively
require a positive matrix-instruction observation before publishing output.
Preparation performs the real compiled probe once per context/provider and
caches both supported and unsupported outcomes; errors remain unobserved and
fail preparation. Device/target and loaded provider identity bind this
process-local observation. Warm scalar, WMMA and hybrid calls allocate/upload no
directories and repeat no capability probe. This is capability admission, not
ISA, numerical or performance qualification.

The public owners retain the same source-generation, range, alias, workspace and
transaction checks. Dense public GEMM and AUTO do not select these modes. Existing
packed blocks and directories are reused; preparation/metadata/probe costs belong in whole-call
measurements. The hybrid kernel reserves scalar shared-memory resources even
when a particular tile chooses WMMA, so it may lose to the explicit WMMA route.
No positive performance claim is made before the final campaign.

Authored differentials cover all block-intersection masks, zero payloads, odd
M/N/K tails, padded destinations, prefixes1/9/20, signed full-width products and
K=65535/65536/65537. Public signed/unsigned owners exercise both added modes with
independent Boost export checks and the existing failure/lifetime cases. These
fixtures have not run during frontloaded development.

The authored [public block-sparse calibration](block-sparse-calibration.md)
compares the dense ordinary control, warm prepared owners and measured cold
preparation/reuse/destruction batches. Its full Boost audits, case sets and
collector receipts also await the final consolidated validation campaign.
