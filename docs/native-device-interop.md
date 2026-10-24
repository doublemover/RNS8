# Synchronous native HIP buffer leases

The experimental public C boundary provides `rns8_query_native_device`,
`rns8_create_native_device_lease`, `rns8_import_native_device`,
`rns8_import_native_device_verified`, `rns8_export_native_device`, and `rns8_destroy_native_device_lease`.
The initial execution mode is explicitly synchronous direct HIP. CPU builds
support descriptor/resource queries and reject lease creation. No host pointer
API accepts a device pointer implicitly.

A versioned descriptor names native i64/u64 dtype, read OR write access,
allocation base/capacity, byte offset, rows/columns/leading dimension, device,
owner identity, nonzero binding generation, explicit BORROW or COPY_OWNED policy, producer
completion and acquire/release callbacks. Signedness does not select arithmetic
semantics: import/export independently admit bounded or exact-wide matching
matrix semantics. Finite, wrap and native limb tensors are not admitted in V1.
A matrix created with `RNS8_MATRIX_RETAIN_NATIVE_INPUT` receives an owned compact
device copy on import. Conversion reads that copy; native and residue currentness
publish together after completion. Destroying the lease or recycling its source
cannot change the owned snapshot. Reimport refreshes the same allocation. There
is no borrowed external pointer stored as the matrix's native representation.

The pure query validates the complete accessible view, pointer arithmetic,
alignment, stride and qualified index domain, and returns compact output plus
four-byte status scratch requirements. It does not query HIP or call owners.
Creation first acquires an owner reference, then checks HIP pointer attributes,
same device, unmanaged device memory and actual allocation base/capacity using
`hipMemGetAddressRange`. Interior views use byte_offset; passing an interior
pointer as an allocation base rejects. Unsupported pointer-query/runtime
capabilities are explicit failures. A failed creation after acquisition releases
that reference; an unsuccessful acquire transfers no reference.

The caller must complete producer work before creation, truthfully set
producer_complete=1, and exclude all external writes/recycling while a read
lease exists. A write lease excludes other external readers/writers until each
synchronous export returns success. Binding generations and owner IDs describe
the immutable view; they cannot detect a caller violating this lifetime
contract. Destroy/recreate a lease to refresh a view. Recycled allocations need
new owner IDs. No cache hit is inferred from pointer equality or external
source generation, and external generations never enter host-pack's version
cache. Concurrent conflicting leases are a caller error in this synchronous
V1; prepared-token reservations and external stream dependencies remain work.

Import reuses existing strided device pack kernels into ordinary owned RNS
storage. It certifies the complete native type envelope (2^63 signed magnitude
or UINT64_MAX unsigned), checks unique CRT coverage before writes, and publishes
that conservative scalar proof. It never treats a caller's small bound as a
verified fact and never downloads input payload for a host scan. Every import
reads the leased view, including a newly leased recycled address. Retained-native
imports add a separately timed `interop_native_retain_d2d` logical-row copy and
retain rows*cols*8 bytes in the matrix; no host native mirror or duplicate upload
buffer is allocated. The normal matrix allocation report includes those bytes.
Input/output leases overlapping either the matrix's residue or native allocation
reject before writes, preserving both representations. A failed issued import
invalidates both, and uncertain completion retains the existing matrix/lease pins.

The opt-in verified import scans actual device magnitudes with the same block
reduction used by narrow native admission, extended to unsigned values. It
charges eight retained device scratch bytes and reads back only the eight-byte
maximum. The scan handles INT64_MIN without signed negation, excludes padding,
and completes before the matrix write transaction. Its measured magnitude then
passes the same exact uniqueness check and replaces the conservative type
bound. This supports tighter exact-GEMM/native-output admission without trusting
caller assertions. A failed scan or insufficient CRT range preserves the old
matrix value; failed drain still pins the matrix and owner. The input remains
immutable across scan and conversion under the lease contract.

Export requires current device residues and a current unique scalar proof
fitting the native type. Ordinary bounded results without that proof reject;
use a path that publishes such a proof, rather than inventing one at export.
The proof must cover every residue plane consumed by native export, currently
the matrix's complete basis. A narrowed exact-wide computation in a larger
allocation is rejected until those suffix planes are explicitly materialized;
capacity alone is not evidence. Ordinary plan-specific host export can still
read its actually written prefix.
Existing CRT kernels reconstruct into lease-owned compact scratch, then read
only a four-byte status. Only successful range/status verification permits the
D2D logical-row copy to the external destination. Padding is preserved. The
scratch budget includes all retained export payload/status allocation. Static
rejections and reconstruction range failure preserve destination bytes; an
issued commit failure makes the destination invalid. Consumers require a
successful return, not merely an observed device event.

Each operation pins its matrix and lease until real completion. An error is
followed by an explicit drain. Failed drain retains the matrix reservation,
external owner and scratch and poisons the context. Destruction retries the
drain/free and preserves the lease on failure; successful destruction invokes
release exactly once. Callbacks run on the caller's host thread, must not throw
or reenter this lease, and are never HIP runtime callbacks. Context destruction
is BUSY while leases exist. This uses existing matrix reservation/context poison
rules, not a second async completion type.

This is zero host **payload** round trip, not zero-copy conversion: import writes
RNS storage, export stages native data and performs a D2D copy, and compact
status travels to the host. Conversion/reconstruction timings remain separate
from status and commit traffic. Authored CPU descriptor and HIP full-width
Boost/residue/output/padding/ownership fixtures await the consolidated campaign.
Prepared async/graph binding, failed-drain
mock fixtures, limb outputs, DLPack and matched performance
receipts remain implementation/qualification work. No measured speedup or
broader target support is claimed.

The installed `examples/device-native` consumer supplies custom HIP producer and
consumer kernels around public exact GEMM. It independently checks all output
and padding and separately labels validation-only host payload readback. Its
standalone CMake recipe uses explicit hipcc object compilation and exported
static/shared targets; build/runtime qualification is pending the final batch.

## Copy-owned read leases

`RNS8_NATIVE_DEVICE_COPY_OWNED` is an explicit synchronous read-only alternative
to borrowing. Set `snapshot_budget_bytes` to at least the queried compact native
size. Creation acquires the external owner, validates its actual device allocation,
and copies only logical rows into a new compact D2D allocation. It releases the
external owner exactly once after successful copy completion. The caller may then
mutate or free the original source; repeated ordinary or verified imports read
the immutable snapshot. Creating a fresh lease is the refresh operation. Export
requires a borrowed WRITE lease; copied destinations are rejected.

Query reports `snapshot_bytes` separately from import/export scratch. BORROW
requires a zero snapshot budget and retains the original lifetime contract.
Snapshot traffic is exactly rows*cols*8 logical D2D bytes, not zero-copy. It adds
one retained compact allocation; operation scratch budgets remain separate.
An unpublished snapshot whose copy/drain/free fails remains owned by the poisoned
context, including the external owner reference when it may still be in use.
Context destruction retries cleanup and returns failure until safe release.
Public handles remain BUSY lifetime references to the context. Host callbacks
must not throw or reenter the lease.

CPU descriptor tests and GPU signed/unsigned padded-view differential cases are
authored for budget rejection, early external release, source overwrite/free,
repeated verified import and exactly-once callback release. Execution remains
pending the final consolidated campaign; no copy-mode speedup is claimed.
