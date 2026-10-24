# Bounded out-of-core panel planning

`StreamPanelProblem` is an internal versioned contract for one host-backed
bounded i64/u64 product, with explicit M/N/K panels and one uniform prefix.
`plan_stream_panels` checks the complete padded host spans before admitting
locally qualified GPU panel dimensions. It does not reinterpret finite, wrap,
or exact-wide contracts as bounded arithmetic.

The serial resource plan contains eight nonoverlapping 256-byte-aligned regions:
A/B residue panels, two ping-pong C accumulators, native A/B upload panels,
native output staging and a status word. A single device arena owns all regions.
Host payload allowance covers compact A/B and export-tile staging; caller input/output
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

The experimental public C interface comprises `rns8_query_stream_gemm`,
`rns8_create_stream_gemm`, `rns8_execute_stream_gemm`, and
`rns8_destroy_stream_gemm`. Query works without a GPU. Creation requires direct
HIP and allocates all staging plus the single device arena before execution.
The immutable plan retains its context; context destruction is BUSY until plan
release. Execution is synchronous and serialized by a try-lock.

The descriptor declares signed/unsigned semantics, global and panel dimensions,
strides, prefix, maximum input magnitudes and separate hard payload budgets.
Plan admission proves K*amax*bmax fits native output and the uniform CRT basis.
Execution independently scans both complete native inputs to verify those
magnitudes before any output mutation. It does not assume final cancellation
bounds intermediate native products. Each panel converts to residues, computes
A*B+the previous residue accumulator with fused modular reduction, then swaps
two disjoint accumulators. No partial K sum is reconstructed into native64.
Only after complete K coverage does the existing checked CRT export reconstruct
the tile into owned host staging and copy its logical cells to caller output.
Padded destination cells remain untouched.

Source instances/generations bind the execution epoch but never bypass the
input scan; no retained-panel cache exists. Host inputs are borrowed and must
remain immutable throughout the synchronous call. The result reports the epoch,
completed tiles/panels and whole-output validity. Static rejection preserves
caller output and result. After execution begins, failure publishes invalid
result status; earlier completed tiles may already be present. On any execution
error, an explicit drain precedes cleanup. A failed drain poisons the context
and quarantines the plan's arena and owned host staging; destroy preserves the
plan until a drain and free succeed. Caller output is never a DMA target, so
pending DMA cannot retain it after return. Poisoned contexts stay poisoned.

CPU fixtures exhaust small panel combinations, count every output/plane/K cell
exactly once, verify slot nonoverlap, reject budget-minus-one, stale/duplicate
completion and overflowing global spans. Public HIP fixtures force many M/N/K
panels under a budget smaller than complete resident operands, compare full
signed/unsigned output to Boost, preserve padding and reject violated bounds.
All execution remains pending. Async overlap, exact-wide/finite/wrap extensions,
calibrated panel selection and capacity/performance receipts
remain development/qualification work; no throughput claim follows from the
serial implementation.

An installed public-header consumer is available as
`examples/downstream-cmake/stream_bounded_i64.cpp`. Its default mode checks
budget queries without a device; explicit `--hip` performs two full products,
checks padding and generation/coverage receipts, and verifies BUSY ownership
and preflight rejection. Installed CTest GPU execution is opt-in via
`RNS8_DOWNSTREAM_GPU_TESTS`. This authored example is awaiting the final batch.
