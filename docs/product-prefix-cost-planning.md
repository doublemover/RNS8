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
limit. Strict chains additionally use an exact cost dynamic program over
producer/consumer prefix pairs, including the final export lift. A chain may
multiply the previous value by an external input on either side or by itself;
reusing any older intermediate routes to general DAG search. Larger spaces seed
search with the chain optimum when available, then minimum/full seeds as the
evaluation budget permits, followed by bounded coordinate descent. Ties prefer lower peak bytes, then lexicographically smaller
prefix choices. The result reports evaluation count and whether enumeration was
exhaustive. No feasible priced candidate returns unsupported; a bounded search
failure does **not** prove no feasible plan exists. Search is limited to 64 nodes,
65536 arena evaluations and eight coordinate passes. The chain DP examines at
most 64*20*20 transitions, independent of the exponential prefix combinations.
Its cost-only optimum is passed through the same real arena/scratch admission;
a memory rejection never authorizes that path or proves no feasible path exists.
The result separately records the chain cost lower bound and `cost_optimal`.
A feasible selection attaining that bound proves minimum estimated cost under
the supplied table, but does not prove minimum peak memory among all cost ties.
Exhaustive feasible search also proves cost optimality. No global optimality is
claimed for other larger-graph heuristic results, and common final-publication/setup costs are not
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

Additional synthetic fixtures compare chain-DP costs to exhaustive enumeration
with missing transitions, solve a 64-node chain within two arena evaluations,
and reject zero budgets and non-chain reuse. These are authored algorithm
checks, not measured regret or performance qualification.
