# Cost-aware runtime prefix planning

The internal `RuntimeProductRecipe::create` accepts an optional
`ProductCostProfile`. It selects the prefixes actually consumed by the existing
CPU/HIP arena executor. This is source implementation with synthetic fixtures;
there are no new measured performance results or AUTO promotions.

Each node supplies a table of complete GEMM costs at prefixes 1..capacity and a
table of exact lift costs from each source prefix to a larger target prefix.
Costs use integer nanoseconds; UINT64_MAX is unknown and never treated as free.
The profile includes version, graph fingerprint, backend, semantics, target and
provenance label. Runtime binding rejects a different graph, backend, semantic
contract or target. A provenance label identifies the caller's estimates; it does
not authenticate a measurement receipt or make it eligible for reviewed caches.

Exact range propagation independently establishes every node's minimum prefix
before the planner sees costs. Changing estimates cannot authorize insufficient
range, unsupported address spans or an over-budget arena. External inputs remain
full-prefix and generation-bound. The final output is fully materialized at the
same boundary as the ordinary full-prefix and minimum-prefix controls.

For each candidate, the planner follows the real topological execution order.
A shared producer is lifted only when its next reader requires more planes, and
subsequent readers reuse the materialized planes. If different readers demand
successively wider prefixes, each incremental lift is charged. The final lift is
charged too. Lifetimes and aligned slot reuse come from `plan_product_arena`.
The HIP peak adds the same conservative per-value lift scratch reservation as
the runtime owner; CPU extension writes the private arena directly. The planner
does not pretend all phase costs overlap or omit the next consumer's work.

Tiny search spaces are enumerated exhaustively under an explicit evaluation
limit. Larger spaces use deterministic minimum/full seeds followed by bounded
coordinate descent. Ties prefer lower peak bytes, then lexicographically smaller
prefix choices. The result reports evaluation count and whether enumeration was
exhaustive. No feasible priced candidate returns unsupported; a bounded search
failure does **not** prove no feasible plan exists. Search is limited to 64 nodes,
65536 evaluations and eight coordinate passes. No global optimality is claimed
for the larger-graph heuristic, and common final-publication/setup costs are not
part of its ranking estimate.

This adapter currently chooses only uniform default-basis compute prefixes.
It does not select physical layouts, native/ragged/encoded-B representations,
backend kernels, graph topology, reassociation or asynchronous schedules. The
public recipe APIs continue to expose their two explicit existing policies;
the cost-profile interface is internal pending measured cohort and broader
provider qualification. Model predictions are distinct from measured latency.

Authored tests exhaust a small diamond, demonstrate cheap versus expensive lift
choices, enforce the actual memory budget, charge shared incremental extension,
reject missing costs/stale graph keys, preserve minimum proof prefixes and
execute a cost-selected CPU diamond with a full independently checked result.
Final qualification must compare the selected route with exhaustive same-contract
measurements and report worst-case regret, planning cost and observed peak memory.
