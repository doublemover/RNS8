# Bounded out-of-core panel planning

`StreamPanelProblem` is an internal versioned contract for one host-backed
bounded i64/u64 product, with explicit M/N/K panels and one uniform prefix.
`plan_stream_panels` checks the complete padded host spans before admitting
locally qualified GPU panel dimensions. It does not reinterpret finite, wrap,
or exact-wide contracts as bounded arithmetic.

The serial resource plan contains eight nonoverlapping 256-byte-aligned regions:
A/B residue panels, two ping-pong C accumulators, native A/B upload panels,
native output staging and a status word. A single device arena owns all regions.
Host payload allowance covers compact A/B staging only; caller input/output
spans are reported separately and never copied into full-sized library shadows.
There are no pinned buffers, event slots or retained B cache in this initial
serial schedule. Driver/allocator overhead and pageable runtime staging remain
outside requested payload capacity; no observed peak claim is made.

The query returns required and exact 1x1x1 minimum capacities even when the
caller budget is insufficient. It does not silently shrink an explicitly
requested panel or allocate past the budget. Panel extents are clipped to the
logical problem, so oversized user panels do not inflate resource admission.
Every arithmetic size and global/panel address has a checked bound.

`StreamPanelCoverage` enumerates disjoint output macrotiles and visits their
complete K ranges in serial order. Every step covers all admitted modulus
planes. Acknowledgment requires the exact step index, execution epoch and both
input instance/generation snapshots. Missing, duplicate, reordered or stale
acknowledgments cannot complete a product. Only a last-K panel makes a tile
eligible for export; only full coverage makes whole-operation success eligible.
Sources must remain immutable throughout execution. The planned streaming host
destination is uncommitted until whole-operation success and may contain
completed tiles after an issued failure; rollback would require separately
budgeted full staging.

CPU fixtures exhaust small panel combinations, count every output/plane/K cell
exactly once, verify slot nonoverlap, reject budget-minus-one, stale/duplicate
completion and overflowing global spans. This is authored planning code only;
the serial GPU executor, actual range proof, public lifecycle, quarantine,
benchmark/trace and measured capacity demonstration are still development work.
No tests or performance qualification have been run for this slice.
