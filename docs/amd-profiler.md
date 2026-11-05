# AMD profiler evidence

Profiling explains an already admitted workload. It cannot replace exact output
checks or provide unprofiled production latency. Raw captures stay in ignored
temporary storage; derived report files describe their actual evidence scope.

## Maintained source contracts

| Path | Supported source contract | Current boundary |
| --- | --- | --- |
| Linux rocprofv3 | ROCprofiler-SDK 1.4.1 native JSON kernel dispatch | Process/agent/queue/dispatch, exact symbol and file-backed code-object association |
| Linux rocpd | Native database schema 3.0.4 | Same dispatch join, native graph/stream/event relationships, individual counter observations and vendor definitions |
| Windows RDP CLI | Candidate 3.5.1 read-only version/profiling help | Controller and service are discovered separately; capture/export integration remains pending |
| Raw RGP, unsupported databases and exports | Inventory only | No inferred dispatch identity or fabricated counters |

The rocpd reader follows AMD's pinned
[schema source](https://github.com/ROCm/rocm-systems/blob/7da43026e4f2ef7600433c38c820732f7a205d22/projects/rocprofiler-sdk/source/share/rocprofiler-sdk-rocpd/rocpd_tables.sql).
It reads a finalized standalone database in immutable read-only mode, rejects
live journal/WAL sidecars, unsupported/mixed versions and ambiguous process
identities, and bounds SQL execution and materialized rows. Database IDs,
profiler logical selectors and HIP ordinals are distinct namespaces.

AMD's [rocpd documentation](https://rocm.docs.amd.com/projects/rocprofiler-sdk/en/latest/how-to/using-rocpd-output-format.html)
recommends retaining the database before converting to other formats. An
export's display name, renamed kernel or temporal proximity cannot establish
the original mangled symbol or loaded artifact. The adapter joins native
process/node/GUID-scoped records and reuses the existing selected static ISA
and immutable artifact receipts.

## Plan and preflight

Plan-only emits supported-feature policy and read-only commands:

    python tools/amd_profiler_capabilities.py --system Linux --target gfx942 --agent 7
    python tools/amd_profiler_capabilities.py --system Windows --target gfx1100

Explicit preflight runs those commands through the existing bounded owned-process
runner, preserving stdout/stderr, command lines and executable hashes:

    python tools/amd_profiler_capabilities.py --system Linux --target gfx942 --agent 7 --counter SQ_WAVES --preflight --out-dir temp/profiler-preflight

The output directory must be fresh. No application workload, GUI or service is
launched, and no driver, clock or permission setting is modified. At most eight
unique counters may be requested for the explicit profiler agent. Preflight
records supported tuple, located executable, parsed version, advertised feature,
permission, capture, decode and evidence-binding states separately. A successful
help command does not establish capture completion or counter permission.
See AMD's [availability commands](https://rocm.docs.amd.com/projects/rocprofiler-sdk/en/latest/how-to/using-rocprofv3-avail.html).

The acquisition-side preflight consumer reopens the receipt, tool executables
and exact stdout/stderr files. It reconstructs the maintained probe inventory
and recomputes version, advertised-feature and agent decisions. Resealing an
edited readiness field cannot replace these raw inputs. The consumer reports
inventory facts; permissions and physical HIP-to-profiler mapping remain
separate capture gates.

The shared process runner retains at most 64 MiB of combined stdout/stderr by
default, draining both streams concurrently. Output overrun, decoding failure
and incomplete cleanup remain failures even if the child exits zero. Optional
owned artifact roots have a sampled byte budget, bounded inventory scans and
explicit rejection of links/junctions. Artifact monitoring can overshoot
between polls; it preserves partial files and reports their observed size.
It never deletes captures or kills an unrelated service to recover space.
Session, preflight, observation and qualification consumers share the same
process-success rule.

The maintained policy keeps discrete Radeon outside Compute Profiler's
Instinct/APU support envelope, gfx1100 ATT trace-only, and stochastic PC sampling
separate from CDNA2 host-trap sampling. Unknown/new tuples remain unqualified
until their documented and observed capabilities are reconciled.

## Native report attachments

The dispatch receipt schema remains the shared
[candidate qualification](target-qualification.md) input. Select the explicit
adapter matching the retained output:

- rocprofiler-sdk-json-kernel-dispatch-v1, with explicit JSON acquisition.
- rocprofiler-sdk-rocpd-3.0.4-kernel-dispatch-v1, with explicit rocpd acquisition.

Both require their own profiled process/capture, the exact production relation,
native output digest, selected static ISA, profiler executable/version receipt
and original acquisition argv. rocpd selected agent/queue/dispatch IDs refer to
its database namespace, not the JSON handle namespace.

For automated collection, rns8-bench accepts --json-output NEW_FILE. It
exclusively reserves the destination after argument validation and before
work, writes native JSON after all measurement intervals, and preserves
stdout/stderr for tool logs. Existing files are never overwritten. Failures
retain an empty/partial native artifact for diagnosis.

A profiled file-output relation additionally retains native_benchmark_output
as an immutable raw-file reference. Validation requires the separately bound
profiled capture to equal that raw output apart from the four declared
collector attachments, and requires its absolute --json-output argument to
name that exact artifact. File-output plumbing is an explicit observation
difference; arbitrary argument changes still fail the common work contract.

    python tools/gpu_counter_report.py temp/production.json --dispatch-receipt temp/dispatch-receipt.json --out-dir temp/counter-review

For multiple captures use per-capture dispatch_receipts entries in the existing
attachment manifest. Flat CSV/JSON attachments remain diagnostic inventory:
matching execution/kernel labels alone cannot qualify resource or counter data.

Native metrics preserve vendor name, description, unit, block, expression,
value type, dispatch interval and XCC/SE/instance dimensions. A missing unit
remains unknown; an observed zero stays zero. The report neither evaluates
vendor formulas nor sums different instances, metrics or replay passes.
Static register/LDS/private-segment requirements remain separate from dynamic
occupancy, residency and traffic. Generic column-name maxima no longer become
memory-pressure, wait or occupancy measurements.

The review reports selected service time, interval union, overlap and uncovered
span separately for each profiled process/pass. Unselected kernels or copies
can occupy gaps, so gaps alone do not prove a host bottleneck. Missing loss/full
coverage remains explicit; no complete-workload utilization or speedup is
inferred. Markdown limits displayed metric rows; JSON retains all observations.

## Optional ROCTx ranges

Native Linux builds can opt into RNS8_ENABLE_ROCTX=ON when the current
rocprofiler-sdk-roctx headers are discoverable. No marker API or profiler
library is required by correctness builds or installed public consumers.
The runtime loader remains disabled until RNS8_PROFILE_RANGES=1 is explicitly
set. Windows requests are rejected because its marker backend is not established.

The adapter uses the current SDK's process start/stop range API, loaded on
demand. Its phase vocabulary covers preparation, pack, proof, dispatch enqueue,
reduction, export, completion, publication, oracle and cleanup. No range is
placed inside a per-cell/inner-product loop. Fused reduction does not acquire a
fabricated separate duration. Regular benchmark warmup/measured loops carry
numeric iteration/operation tags. Untagged paths remain unclassified.

Nested host scopes restore their parent's tag on errors and exceptions.
Prepared HIP operations retain a separate cross-thread range until completion
and publication; a submit scope ending is not GPU completion. Prepared ID/epoch
and benchmark iteration IDs have separate namespaces. Host parent relationships
are explicit metadata, not inferred device dependencies.

The benchmark emits a bounded 4096-record numeric sidecar inside its JSON.
It records actual returned provider range IDs, phase, role, operation namespace,
host stop state, dropped and unpublished records, and the loaded provider path
where available. Provider availability does not establish that a profiler
actually captured markers. Loss/truncation and unmatched IDs cannot become
complete semantic-to-dispatch coverage. More detailed/long-lived correlation
must use retained native traces. No file I/O or JSON formatting occurs inside
the range path.

Active/requested tracing is excluded from production timing admission. The
profiler relation permits the explicitly declared range opt-in environment,
retains its own capture identity, and still requires a matching exact result.
The disabled build path uses empty inline scopes and has no dynamic loader,
counter, allocation, synchronization or profiler call.

## Remaining implementation and validation

Native acquisition, converter/export receipts, Windows RDP/RGP machine evidence,
remaining phase/async family coverage and native range-to-dispatch joins, and
additional ATT/sampling/counter-pass coverage remain under #104.
Unsupported formats and memory-backed code-object loads remain explicitly
unqualified. Hardware unavailable locally can be qualified later, after its
source implementation is ready.

The new preflight/database/report fixtures are authored protocol controls, not
collected GPU evidence. Their execution and all new hardware checks are deferred
until the development pass is complete.
