# Resident storage allocation

HIP residue, finite-ring/field, and wrap matrices derive device payload capacity
from their checked logical dimensions and representation. Creation allocates no
full-sized host residue or byte-limb mirror. Device allocations remain zeroed,
but fresh device capacity is not a published current value. CPU matrices retain
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

`rns8_get_matrix_storage_info` reports resident representation bytes separately
for host and device domains, with independent currentness flags. These payload
fields do not claim allocator overhead, process RSS, transient peak, or upload
and export staging totals. Existing reusable upload/export staging remains
matrix-owned until destruction. A comprehensive allocation/peak diagnostic and
explicit staging trim are still separate pending work; no speedup or measured
memory peak is claimed by this allocation change.

Authored coverage includes allocation-independent mock admission, malformed
capacity/currentness, overflow, all seven direct-HIP semantic families, and the
existing compact/padded signed/unsigned exact native ownership differential
fixtures. Qualification is pending the consolidated validation batch.
