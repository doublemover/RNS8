# Experimental ordered runtime bases

The opt-in C API in `rns8/experimental_basis.h` owns an explicit ordered basis
and executes exact-wide signed or unsigned products on the CPU reference or
direct HIP provider. It does not replace the default ladder or select a basis
implicitly. The explicit cost-bound selector below adds opt-in runtime choice.
The [October27 validation](../metadata/replay-extension-validation-20261027.json)
passes applicable CPU/direct-HIP and sanitizer fixtures. Measurements and broader
target qualification remain separate.

`rns8_create_basis` copies 1–20 moduli in the range 2–256. It rejects duplicate
or noncoprime channels and owns prefix products, mixed-radix inverses and a stable
fingerprint of the ordered channels and centered-INT8 codec. Matrix and plan
owners retain immutable basis data independently of the public basis handle.
Destroying that handle cannot invalidate a live matrix or plan. Equality checks
compare complete ordered content; fingerprints are diagnostic identities.

`rns8_create_basis_matrix` creates an ordinary matrix handle with an explicit
experimental basis identity. Storage is compact modulus-major centered signed
bytes, with allocated capacity separate from the currently proven prefix.
`rns8_get_matrix_basis` reports the actual ordered channels and prefix product.
Ordinary default-basis GEMM plans and accelerator prepack caches reject
these matrices before consuming their bytes. A reordered basis is a different
storage contract, even when its full product is unchanged. Passing a null basis
to the experimental constructors explicitly selects the unchanged default basis
for matched controls and conversion destinations.

`rns8_query_basis_plan` computes checked host/device payload requirements for a
fixed M/N/K problem. `rns8_create_basis_plan` retains its context and allocates
staging within separate hard budgets. `rns8_get_basis_plan_info` reports the
retained plan's ordered basis, selected prefix and required payloads. CPU plans
own residue staging and compact native/limb staging. HIP plans own one aligned
arena containing target/source basis packets, native upload and shared result/CRT
scratch, plus compact host output staging. Matrices and fixed descriptor/runtime
metadata are outside those scratch budgets. There is no nested pinned allocation
and no device allocation in an execution call.

The native packing functions accept explicit signedness, padded host inputs and
optional stable source versions. They verify the exact maximum magnitude before
staging a write. Signed packing handles `INT64_MIN` through unsigned magnitude
arithmetic. Packing, multiplication, CRT and conversion use the owned channel
order throughout. GEMM computes per-channel INT8 products with fused residue
reduction and safe INT32 K slabs of at most 65536, including an incomplete final
slab. Its complete expression bound must satisfy `P > 2B` for signed values or
`P > B` for unsigned values before any output write.

`rns8_export_basis_limbs` reconstructs against the actual proven prefix, verifies
the current magnitude and output width, and scatters only logical cells to the
borrowed host destination. It supports one to three least-significant-first
64-bit limbs; signed outputs use two's complement. A symmetric magnitude proof
must fit the requested width, so a signed minimum may require another limb.
CPU reconstruction uses Boost integers; HIP reconstruction uses a separately
implemented three-word mixed-radix kernel with the plan-owned products/inverses.
All admitted 20-channel byte bases have products below the three-word capacity.

`rns8_convert_basis_values` is the explicit bridge between different ordered
bases, including default-basis values. The source must have an engine-owned
unique-range proof in its actual written prefix. Source and destination must
have the same signed/unsigned exact-wide contract and logical shape, and the
destination basis must uniquely cover that proof. Conversion reconstructs the
integer and computes each destination residue; it never treats a permutation,
missing channel or larger prefix as a byte copy or zero fill. It preserves the
source generation and publishes one new destination value. This also supplies
an explicit wider-materialization operation for a smaller proven prefix.

Product recipes accept a common explicit ordered basis on all inputs and
outputs. Full-prefix and minimum-compute-prefix chains, diamonds and multiple
retained outputs use that basis's prefix products for every range proof and
channel arithmetic. Rebinding compares complete ordered content and frozen
magnitude bounds. `rns8_get_product_recipe_basis` reports the retained identity.
CPU lifting uses Boost reconstruction; HIP lifting uses one retained aligned
full-target staging area, checked CRT status and an immutable device packet,
then copies only the newly proven suffix. The arena budget includes all of
that storage. Private cost profiles bind the complete basis and provider and
price the same storage; default-ladder estimates cannot silently select custom
runtime-modulus kernels.

Internal uniform layout descriptors use ABI 2 with the complete ordered basis.
Plain/tile-major format conversion preserves it and rejects a reordered basis
or missing prefix. Changing a basis still requires explicit integer conversion.

`rns8_query_basis_operation` reports typed static compatibility for the concrete
packing, GEMM, CRT/export, integer-conversion and basis-plan owners. It returns
the complete ordered basis alongside the shared operation-support record and
uses the same payload calculation as plan creation. Its provider digest binds
all channels, prefix, signedness, shape, backend, target, build and budgets.
Static success still requires actual current matrix/range/generation binding.
Direct-HIP GEMM supports borrowed streams and owned graphs through
`rns8_prepare_basis_gemm_hip` and the ordinary prepared completion, cancellation
and refresh APIs. It retains the complete basis and matrix owners independently
of the basis-plan handle. `rns8_prepare_basis_conversion_hip` prepares exact
conversion/lift between complete ordered bases with checked device staging,
pinned status and guarded whole-output publication. Its explicit total budget
includes both payloads. `rns8_prepare_native_transfer_hip` accepts ordered-basis
matrices for lease-owned signed/unsigned native import and checked scalar/limb
export. It preserves the same readiness events, generation refresh, ownership,
stream/graph completion and dependent-token rules as ordinary native transfers.
Value packet kernel arguments freeze full basis content without borrowed packet
uploads. The original host-pointer packing/conversion/export calls remain
synchronous. New prepared transfer/conversion fixtures pass the applicable CPU-ASan
and gfx1100 HIP lanes in the [October27 implementation validation](../metadata/implementation-closeout-20261027.json).
Prepared stream/graph execution currently uses the direct-HIP compute owner;
library plans expose their explicit synchronous execution mode.
Ordinary inventory identifies the implemented source and directs
callers to this explicit query; it cannot infer an experimental basis from a
default-ladder descriptor. Neither query marks arithmetic or performance as
qualified.

Every operation stages its complete result before committing. Rejected proof,
basis, shape, capacity or width leaves the prior output intact. A failed commit
invalidates the destination. Existing context failure leases retain borrowed
matrix readers/writers and the plan workspace through an undrainable failure;
`rns8_drain_failed_work` releases those reservations after a successful drain
without unpoisoning the context. Plan destruction is busy while pending and
retryable after a failed buffer release. Shared handles require host-thread
serialization; one plan rejects overlapping use through its mutex.

This phase provides explicit CPU/direct-HIP exact-wide packing, GEMM, CRT export
and integer basis conversion, product recipes, packed/ragged materialization,
uniform layout conversion and prepared stream/graph GEMM. Ragged directories
retain actual basis products and inverses; packed recipe bindings compare the
complete ordered identity. Finite and wraparound arithmetic remain separate
contracts rather than inferred experimental-basis semantics.
CPU/HIP fixtures cover full native boundaries, independent Boost
products, reordered/default conversion, output padding, basis/width rejection,
ownership after handle destruction and K=65535/65536/65537. No speedup follows
from this source implementation.

## Explicit library computation and runtime basis selection

The basis plan's `backend` now selects direct HIP, CK, rocWMMA, hipBLASLt or
AMDGPU builtins over matrices owned by a direct-HIP context. Packing, exact
conversion and CRT keep the owned packet and actual ordered moduli. Library
GEMM processes each complete plane with its actual modulus, including small and
composite channels outside the default ladder. Every partial result remains in
the plan's full private output stage until successful complete publication.
Reusable provider scratch and INT32 staging are charged to the explicit device
budget. hipBLASLt retains its own handle; a failed handle release preserves a
retryable plan. If handle creation fails after acquisition, a nonnull failed-plan
output is cleanup-only and must be destroyed. Opaque vendor setup is additional
to counted payload. Uncompiled providers reject; builtins keep their65536 K cap.

`rns8_select_basis_plan` deterministically chooses among caller-owned candidates
for one explicit shape, signedness, dataset/output cohort and magnitude bound.
Each observed or modeled price must bind the complete ordered basis, prefix,
provider, current build/target and exact queried payload. Unknown prices,
overflowing lifetime costs, mismatched identities, insufficient CRT range and
infeasible budgets cannot win. The charge includes setup, both-input packing,
every GEMM and final export; flags charge packing/export on every replay instead.
Stable candidate order resolves equal projected cost. `rns8_create_selected_basis_plan`
creates the selected retained owner. Actual values still require ordinary range
and generation admission; caller-supplied cost data does not qualify performance.
New selection, full-width provider and dynamic-reducer fixtures are implemented
for the final consolidated batch. Existing measurement receipts retain their
original source; no global AUTO or measured routing promotion follows.
