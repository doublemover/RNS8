# Resident storage allocation

HIP residue, finite-ring/field, and wrap matrices derive device payload capacity
from their checked logical dimensions and representation. Creation allocates no
full-sized host residue or byte-limb mirror. Device allocations remain zeroed;
initialization finishes before creation returns, including for subsequent use on
nonblocking streams. Fresh device capacity is not a published current value. CPU matrices retain
their previous host allocation and initial-value behavior, including the exact
zero proof for fresh exact-wide CPU matrices and the unbound finite modulus.

Storage validators check device allocation capacity independently of host vector
size. An absent host mirror cannot be marked host-current. CPU admission still
requires the complete host payload. Exact bounds and axis certificates remain
small, separate host metadata.

## Host payload consumers

CPU pack, GEMM, sparse expansion, lift, epilogue, verifier, and CRT export operate
only after CPU/backend/currentness admission. Complex and product-recipe views
choose host or device addresses explicitly. Ragged preparation explicitly
stages a device download; ordinary GPU export reconstructs into its export
buffer and downloads the requested output, without populating a residue mirror.
Arbitrary-width limb import explicitly converts on the CPU into call-local
residue staging, uploads it, then releases it. This staging is temporary and can
be as large as the complete residue payload. No implicit CPU fallback or host
view is introduced.

Retained native pack uploads one compact matrix-owned device allocation, then
converts directly from it. Padded caller rows compact into call-local host
staging before that upload. The retained native allocation has one owner;
residue conversion borrows it and cannot free it. Existing write reservations
and generation invalidation remain in force.

`rns8_get_matrix_storage_info` reports resident representation payload bytes
separately for host and device, with independent currentness flags.
`rns8_get_matrix_allocation_info` additionally reports checked logical primary
bytes, host vector capacities, device payload capacities, current payload bytes,
reusable upload and export staging, status/schedule allocations, and total owned
device capacity. These are requested allocation capacities, not allocator
padding, process RSS, transient peak, or sampled device memory. Host proof
objects and temporary conversion vectors are excluded explicitly. No measured
memory peak or speedup is claimed.

`rns8_trim_matrix_staging` (C++ `Matrix::trim_staging`) explicitly releases only
reusable device upload/export/status/schedule allocations. It returns BUSY for
prepared references, active readers or a writer, before touching cache state.
It never releases residue, byte-limb or native value allocations, including a
promised retained native input, and never changes value generation or proof.
Subsequent pack/export calls rebuild their staging normally. There is no hidden
trim on a hot path. A failed free stops the operation: successful prior frees
remain released, unreleased buffers remain owned, and the export descriptor
fingerprint stays invalid so no partially released cache can be reused. The
caller must exclude concurrent host API calls as for other matrix mutations.

Authored coverage includes allocation-independent mock admission, malformed
capacity/currentness, overflow, all seven direct-HIP semantic families, and the
existing compact/padded signed/unsigned exact native ownership differential
fixtures. Qualification is pending the consolidated validation batch.
