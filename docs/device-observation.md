# Runtime device observations

rns8_query_runtime_device_identity queries an ordinal in the calling process's
current HIP visibility namespace. Ordinal zero inside an isolated process is
not automatically parent ordinal zero or a physical board number.

Initialize the output's struct_size and abi_version. A readable eight-byte
size prefix is mandatory; success requires exactly the current full structure
size. Invalid ordinal/header admission and unsupported/failed queries preserve
every output byte. The query requests no device switch, allocation, kernel
launch or synchronization.

The output explicitly separates availability from values:

| Field | Meaning |
| --- | --- |
| runtime_ordinal | The queried process-local HIP ordinal |
| valid_fields | PCI BDF and/or UUID availability bits |
| pci_bdf | Canonical PCI domain:bus:slot.function when available |
| uuid | Sixteen opaque HIP UUID bytes when available |
| native_pci_status, native_uuid_status | Raw native API results for diagnosing unavailable fields |

A successful inquiry can report zero available fields. That is an unknown
association, not a successful physical-device check. The UUID API is marked
beta by the selected SDK; nil/unavailable UUIDs are not fabricated. A PCI BDF
identifies a function on one node, not a globally unique board. Device
partitions can share a PCI function while retaining distinct UUIDs.

Optional query errors are checked and cleared before another kernel's launch
check can observe them. An already pending runtime error is reported without
being consumed; unexpected failure during the inquiry returns backend failure.

The inspector emits a versioned runtime_device_identity observation. The
benchmark emits a device_identity_witness with before/after queries outside
timed workload intervals. Stable identity requires matching availability,
ordinal and every available identifier; missing, changed or failed observations
remain explicitly unavailable/changed. Serialization preserves raw UUID bytes
as fixed hexadecimal, without changing the output stream's number format.

This source slice supplies native observations. Session visibility mapping and
the standalone multi-GPU reporter still need to consume and bind these records
to their exact capture/execution/artifact and rank/case receipts under #96.
Filesystem names and launcher intent must not overwrite contradictory runtime
observations. Hardware validation of these new query/witness paths is deferred
to the final batch.
