# Timing admission through review and promotion

`tools/timing_eligibility.py` centralizes host-phase admission for pairwise
comparison, variance reports, sweep review and final promotion ledgers. Version
two, an explicit supported result kind, measured phase state, finite nonempty
samples and every sample above the declared censor threshold are required.
Censored, fused, elided, unsupported and absent phases do not establish measured
speedups. Batch-amortized and single-operation results are not interchangeable.
Legacy descriptive statistics remain readable and are not silently promoted.

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

Synthetic fixtures use an explicit shared provenance builder with matching
manifest/capture commits and complete artifact identities. Independent runs get
new UUIDs; copying a capture preserves its UUID. Authored integration checks cover
measured controls, schema-consistent censored data, unsupported phase states,
mixed result kinds and a baseline changed after review. Execution is pending.
