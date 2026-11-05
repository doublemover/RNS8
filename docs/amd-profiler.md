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

## Remaining implementation and validation

Native acquisition, converter/export receipts, Windows RDP/RGP machine evidence,
ROCTx ranges and additional ATT/sampling/counter-pass coverage remain under #104.
Unsupported formats and memory-backed code-object loads remain explicitly
unqualified. Hardware unavailable locally can be qualified later, after its
source implementation is ready.

The new preflight/database/report fixtures are authored protocol controls, not
collected GPU evidence. Their execution and all new hardware checks are deferred
until the development pass is complete.
