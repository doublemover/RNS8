# Direct operation providers

`backend_common/direct_operation_registry.hpp` is an internal, read-only static
inventory for eight concrete synchronous direct-HIP operations: residue GEMM,
exact axpby, exact suffix lift, scalar/byte narrow product, direct/balanced canonical
CRT and finite verification. It calls the same pure shape/address/resource
predicates used by their execution paths. Product recipe preparation also uses
the residue GEMM predicate. This slice is authored, not yet qualified.

The query distinguishes a successful static predicate from compilation, runtime
observation, exact qualification and performance promotion. The latter three
remain unknown/false in this context-free inventory. Positive static results
still require current operands, correct semantic domain, bound proof where
applicable, pointer/alias validation and a compatible device. Querying never
allocates GPU storage, launches a kernel, probes a device or edits AUTO caches.

Each record identifies its concrete implementation, storage domains, address
index width, native value scan and matrix-instruction probe requirement. Typed
rejection constraints identify shape, prefix, address span, scratch budget,
modulus, execution mode and operation/version. Unknown operations/versions fail
closed. Borrowed streams, capture and distributed execution reject for these
specific wrappers; support in another wrapper is not inferred from its backend.

Resource bytes are meaningful only when `resources_known` is true. Zero scratch
is valid for successful GEMM/fused epilogue/canonical CRT queries. CRT payload
bytes describe the caller-owned three-limb output. Narrow scratch includes its
native output and overlapping scan allocation. Verification includes host/device
challenge, projection and status payload. Existing exact-lift admission accepts
a **suffix payload budget**, while its returned scratch also includes a status
word; the inventory deliberately preserves that existing contract explicitly.

This is a bounded internal provider expansion. It does not claim a complete
cross-backend registry, target/code-object identity service, public discovery ABI,
calibrated cost profile, sparse capability table, or graph-safe library inventory.
Those providers must carry their own predicates rather than inheriting these
synchronous direct-kernel records. There is no production dispatch change.

Authored fixtures exercise every inventory entry, unsupported execution modes,
unknown versions/operations, full padded address rejection, finite modulus
boundaries and exact resource budgets. Existing real operation fixtures cover
execution. All execution and query/launch qualification remain in the final batch.
