# Direct operation providers

`backend_common/direct_operation_registry.hpp` is an internal, read-only static
inventory for ten concrete synchronous direct-HIP operations: residue GEMM,
exact axpby, exact suffix lift, scalar/byte narrow product, direct/balanced canonical
CRT, local/split group CRT and finite verification. It calls the same pure shape/address/resource
predicates used by their execution paths. Product recipe preparation also uses
the residue GEMM predicate. The source-bound local group/CRT validation is recorded
in the [287737d receipt](../metadata/group-crt-qualification-20261026.json).

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

This bounded internal inventory complements the contextful public
`rns8_query_operation_support` contract; it does not replace that query or infer
its execution owners from synchronous kernel support. Calibration, observed
target/code-object identity and performance promotion retain separate gates.
Other providers carry their own predicates. There is no production dispatch change.

Authored fixtures exercise every inventory entry, unsupported execution modes,
unknown versions/operations, full padded address rejection, finite modulus
boundaries and exact resource budgets. Existing real operation fixtures cover
execution. Their named source checkpoints retain their own validation scope.

## Resident DAG owner parity

Dense GEMM support does not imply an owned resident DAG. Public GEMM/prepared
discovery and `rns8_create_resident_graph` now share one backend predicate:
CPU reference, direct HIP and wrap64 byte-limb owners admit resident graphs.
hipBLASLt, CK, rocWMMA, AMDGPU builtins, the int64 vector backend and CPU AMX do
not acquire a DAG owner merely by implementing synchronous GEMM. Unsupported
semantic/compile conditions remain distinct from unsupported execution modes.

A table-driven regression compares both operation families with actual owner
admission across all nine concrete backends, without GPU probes or device
allocations. Positive source-mode declarations still require runtime binding and
do not assert target correctness/performance qualification. The parity repair's
[36568f5 validation receipt](../metadata/dag-owner-parity-validation-20261026.json)
records 972 GPU and 562 CPU-ASan passes, 16/23 explicit skips, no failures, 12
installed-consumer passes and a fresh 1,006.60-second logical checker pass. Earlier
arithmetic measurements retain their original artifacts.
