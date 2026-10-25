# Calibration evidence and planner rates

`tools/calibration_manifest.py` defines immutable identity, run and quantity
records for offline calibration. Identity binds target/device, compiler/runtime,
artifact and ISA hashes, configuration, dataset, clocks/power policy, cache state,
instruction/symbol, final-output contract, timing source and profiling mode.
Serialize the summary with ordinary JSON: its `raw` field retains every input
sample and counter, including nulls. Synthetic test identities are not hardware
evidence.

Use null for censored elapsed times and unavailable counters. A counter explicitly
distinguishes measured observations from modeled quantities. Conversion and
traffic producers should name unique source elements, issued source loads,
actual bytes, residue/encoded-plane evaluations, temporary reads/writes and final
bytes separately. The existing `evidence_database_lib.work_model` estimates
remain modeled counts; elapsed time cannot turn them into hardware counters.

Raw validation and rate admission are separate. Rate export requires at least
five unique independent process runs with identical work counts, no censored
samples, exact-reference and ISA checks, and uninstrumented timing. Within-run
repetitions are averaged first; they cannot masquerade as independent runs.
The interval uses the df=4 two-sided 95% Student-t coefficient for all supported
run counts, plus the largest recorded timing resolution. This is conservative
for larger degrees of freedom under independent, approximately normal,
stationary run means. It does not establish those assumptions or clock stability.
Drift and correlated runs need a new collection campaign.

`planner_rates` requires exact expected identities and one common device,
binary, runtime/compiler and clocks/power policy. It derives the cohort from
the identities and run IDs and exports reciprocal latency intervals as the
existing analytical planner's rational `Rate` objects. Any opcode, layout,
compiler or artifact identity change requires remeasurement. These rates remain
offline inputs; they do not install or qualify runtime routes.

The held-out report rejects overlapping training/test run IDs and requires the
entire observed interval inside the explicitly tolerated prediction interval.
Repeated-B arithmetic reports a modeled strict break-even call count, or no
positive amortization case when per-call packing savings are absent. It is not
the smallest measured profitable count: that requires paired complete-pipeline
runs charging retained storage, eviction, repacking and mutation.

Kernel/application replay timings remain raw observations and are rejected as
uninstrumented planner calibration. Replay preservation, snapshot costs and
allocation/concurrency limits still require target-specific qualification.

This slice supplies evidence plumbing and synthetic schema/algebra fixtures.
Instruction-chain, LDS/cache/memory, conversion, launch/descriptor, reconstruction
and interaction microbenchmark collection, ISA/resource evidence, sustainable
clock verification and real held-out qualification remain open in #31. Tests
for this addition are covered as registered applicable cases in the
[dated validation](local-all-features-20261023.md).

## CRT collection executable

The existing `rns8-crt-calibrate` target now supplies explicit signed/unsigned
native import and canonical CRT controls, including fixed-dot4, scalar
imports, Garner/direct/balanced reconstruction and tile-major consumers.
[Conversion calibration](conversion-calibration.md) defines the complete CLI,
resident/cold boundaries, every-prefix cases, full Boost comparisons and
provenance rules. Existing six-argument calls retain resident scope; v2 captures
make the expanded contract explicit.

The driver pairs instrumented calls with production calls and includes bounded
sustained bursts. Its updated fixtures and modes await the final consolidated
validation; earlier dated evidence does not validate these changes. No ISA,
traffic, timing-resolution, bandwidth or dispatch promotion follows from a raw
capture.

## Public completed-output timing and sustained controls

`rns8-bench --host-timing production` disables diagnostic GPU event hooks without
removing production proof/status/completion work. The default `instrumented`
lane preserves historical diagnostic behavior. Captures declare the actual host
instrumentation state; old event-enabled captures normalize to diagnostic, not
production. State participates in comparison, cohort, rate and cache-promotion
admission. Diagnostic event observations are retained, but cannot install a
production timing claim.

`--sustained-batches` adds separately recorded bounded whole-call intervals to
the hook-disabled independent single-GEMM lane. Each interval records actual
completed calls/tasks, wall duration, throughput quotient and whether duration
plus minimum calls or the hard call cap terminated it. Source generations
advance across intervals; retained inputs, plans, final native output and
required synchronization remain identical to the latency lane. Full integer
postflight checks occur outside timing. These throughput observations are not
appended as invented latency samples. Graph/streaming/dependent-chain experiments
have distinct boundaries and are not silently routed through this helper.

`tools/collect_public_calibration.py` uses the existing artifact/run-ID collector
and alternates seeded AB/BA instrumented/production order across independent
process pairs. The small manifests in `benchmarks/public-calibration/` cover
launch-sized, larger, reused-B, genuine-wide and grouped output-boundary controls.
Its plan-only mode emits commands without executing them. Raw captures remain
separate; no causal instrumentation-overhead estimate or promotion is inferred
from order, a ratio, or a sustained interval alone.

Grouped output defaults to completed per-task host destinations, with every
required slab-to-destination scatter inside export/end-to-end timing. Explicit
`--grouped-output compact-slab` preserves compact throughput without a redundant
timed scatter; that boundary disallows requested per-task padding and is keyed
separately. The final scatter for the oracle is outside timing only in that
compact contract. Historical grouped-slab captures remain compact observations,
not retroactively completed padded destinations.

## Disjoint events and consistent work units

Roofline diagnostics select a declared non-overlapping cover of pack, compute
and export roots. Complete raw child arrays may replace a parent; any uncovered
parent residual remains explicitly unattributed. Missing/inconsistent detail
falls back to the parent. Prefix subtotals and backend totals are not counted
twice. Corresponding disjoint raw events are summed before category summaries
where possible; raw reported events remain intact. Service-time shares are not
wall-time fractions, particularly across overlapping streams.

Work estimates count all independent tasks in the measured batch, never the
number of repeated observations. Modeled pack payload respects retained A/B,
and native export payload respects the output domain. Dependent chains and
heterogeneous stages without complete shape descriptions have unavailable work
estimates rather than fabricated single-task GOP/s. These logical payload rates
are not measured HBM/PCIe traffic. Mislabelled public `prefix1` rows that actually
requested prefix3 were removed; catalog loading now checks explicit fixed-prefix
annotations against the request without weakening range admission.

The new public lanes, regression sources and collection recipes are authored;
only lightweight syntax/static checks were performed. Test execution, hardware
collection, performance outcomes and scientific qualification remain deferred.
