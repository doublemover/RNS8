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

The session runner records the launched process ID and a bounded mapping
environment for every command. An unmasked parent inspection selects the
requested parent ordinal; the isolated child and benchmark observe ordinal
zero. Their architecture, device name, runtime/driver, memory size and every
common available native identifier must agree. Different selected parent
ordinals require distinguishable execution-device identities. UUIDs permit
distinct partitions sharing one PCI function; BDF-only observations cannot
establish that distinction. Inadequate identifiers remain unknown, and cannot
complete multi-device qualification.

The standalone reporter consumes the existing session's immutable manifest,
results, raw stdout, binary identities and capture files. Creating its binding
receipts performs no GPU work:

    python tools/multigpu_observation.py --session-manifest temp/SESSION/manifest.json --session-results temp/SESSION/results.json --out-dir temp/SESSION/observations
    python tools/multigpu_shard_report.py --observation-dir temp/SESSION/observations --out-dir temp/SESSION/report

Use the actual manifest/results filenames produced by the session. The first
output directory must be fresh. Receipts are content-bound associations, not
signatures or proof of trust against deliberate replacement of every input.
Both emission and reporting reopen the raw inputs, verify artifact hashes and
compiled identities, and require their actual process IDs, arguments,
visibility, rank/world and runtime observations to agree. The reporter retains
launcher intent separately from observed fields. It never uses a filename,
status label or generic management-tool UUID to invent a HIP observation.

Report schema v2 marks legacy, missing or contradictory bindings unqualified
and returns a critical exit decision. Complete case IDs and shared exact-work
contracts—including seed, input profile, modulus and output semantics—partition
checksum comparisons. One case label cannot change work across ranks. Missing
cases/ranks, copied executions and within-case checksum differences remain
critical. Explicit dry runs remain plans. These smoke reports always retain
nonpromotion status; device association does not establish numerical correctness,
release qualification or a performance gain.

Native query/witness and authored loader/report regression execution are
deferred to the final validation batch. No new multi-GPU hardware result is
claimed.
