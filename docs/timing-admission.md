# Timing admission through review and promotion

`tools/timing_eligibility.py` centralizes host-phase admission for pairwise
comparison, variance reports, sweep review and final promotion ledgers. Version
two, an explicit supported result kind, measured phase state, finite nonempty
samples and every sample above the declared censor threshold are required.
Censored, fused, elided, unsupported and absent phases do not establish measured
speedups. Batch-amortized and single-operation results are not interchangeable.
Legacy descriptive statistics remain readable and are not silently promoted.

CPU arithmetic anchors and CPU timing baselines have separate environment
contracts. An arithmetic anchor verifies the complete mathematical result; it
does not establish a speedup. CPU/GPU timing comparisons retain the observed OS
instance, source/build, compiler, toolchain, configuration, clock policy and host
timer contract. Only GPU-specific device identity is omitted for this explicit
comparison. Same-device GPU comparisons retain it. Report partitions and baseline
selection apply this contract before selecting the best aggregated implementation.
The adaptive scheduler report also requires hook-disabled production cohorts and
`--diagnostic-binding` receipts for their separate GPU-event captures. Its CPU
rows verify arithmetic only; they do not supply CPU timing ratios. Local scheduler
decisions require a completed-output gain above both cohorts' observed noise.

Sweep reviews record the actual baseline path and canonical digests of both
captures for their end-to-end timing comparison. Private loader path annotations
are excluded from content digests. The ledger requires that binding even if a
variance report is absent, rereads and schema-validates the bound baseline,
checks both digests and independently recomputes eligibility. It does not trust
an `eligible` flag or a claimed speedup to establish timing resolution. A changed
or unavailable baseline blocks admission until evidence is supplied again.
This is an additional timing gate; it does not replace environment, correctness,
instruction, variance or review requirements. Historical unbound reviews need
new bound comparisons, not an automatic migration to qualified evidence.

The bounded review snapshot retains every raw production capture and both files
in each diagnostic binding, including a relocated candidate pathname read by a
summary. Final hash checks therefore include companions and aliases. A standalone
pending-validation decision closes its own snapshot before returning; a batch
shares one owner and closes it before publishing the complete summary. Changes
during a standalone decision clear its bound/qualified flags and produce a
concrete blocker rather than a local promotion disposition.

Before/after source changes use an explicit experiment rather than weakening
ordinary source/build matching. `tools/collect_paired_experiment.py` writes a
plan by default; it starts processes only with `--execute`. Its definition is a
JSON object with `allowed_delta` (a concrete description of the implementation
change) and `arms.baseline` / `arms.candidate`. Each arm supplies `bench`,
`manifest: {"path": "...", "sha256": "..."}`, `backend_selected`, and
`selected_kernel`. Manifest references bind the actual generated build-manifest
JSON, and paths resolve relative to the definition file. The tool binds the
complete manifest, source commit/digest, static executable hash and selected
backend/kernel before collection. Shared-library experiments are unsupported.

```powershell
python tools/collect_paired_experiment.py --definition temp/experiment.json --output temp/paired-plan --runs 2 -- --m 1024 --n 1024 --k 1024 --warmups 3 --repeats 9
python tools/collect_paired_experiment.py --collect-plan temp/paired-plan/plan.json --execute
python tools/paired_experiment_report.py --plan temp/paired-plan/plan.json --collection temp/paired-plan/collection.json --output temp/paired-plan/report.json
```

Use an even number of independent runs per arm. The collector counterbalances
both build order and production/diagnostic order and records actual sequential,
non-overlapping process intervals. Report replay verifies the sealed plan,
collection file hashes, every execution/request, declared arm membership and
every production process's matching diagnostic. Diagnostics add no production
reruns. The shared report gate still requires full numerical coverage, matching
mathematical work, native host/device identity, toolchain/runtime/clock and
environment, process repeatability and the observed noise margin. Only the two
declared source identities may differ. Filenames do not declare experiment arms.
The receipt records collector order; it is not hardware attestation. The report
grants no promotion/cache authority and cannot restart a stopped campaign.

Synthetic fixtures use an explicit shared provenance builder with matching
manifest/capture commits and complete artifact identities. Independent runs get
new UUIDs; copying a capture preserves its UUID. Authored integration checks cover
measured controls, schema-consistent censored data, unsupported phase states,
mixed result kinds and a baseline changed after review. Registered applicable execution evidence is recorded in the
[dated validation](local-all-features-20261023.md).
