# Performance Gain Work Queue

This queue is ordered by expected Windows RX 7900 XTX / `gfx1100`
end-to-end performance value, not ease. Use it to drive implementation slices
from this point forward. Keep evidence claims local to the measured platform:
Windows `gfx1100` evidence does not imply Linux ROCm, Radeon Linux, or
Instinct CDNA readiness.

The central optimization question is no longer only "which backend computes a
single GEMM fastest?" RNS8 has to ask more structural questions:

- How many residue or slice GEMMs can be avoided?
- How many pack, launch, scratch, status, export, and D2H materializations can
  be removed?
- Can the output stay in the domain the next operation needs?
- Can reconstruction be partial, delayed, fused, or skipped?
- Can many small or irregular exact tasks become one persistent grouped
  workload?

## Ground Rules

- Every performance slice needs same-contract CPU/direct-HIP baseline, release
  build, fixed seed, at least 3 warmups, at least 9 repeats, schema validation,
  selected-kernel metadata, and exact CPU differential before promotion.
- Do not promote discovery captures, smoke captures, or Windows evidence into
  Linux or Instinct claims.
- Every new kernel or layout must update `selected_kernel`, `epilogue_mode`,
  `workspace_mode`, `isa_evidence`, autotune key fields, docs, benchmark schema
  fixtures, and stale-cache rejection.

## Active Performance Queue

Use this table as the working control panel. The detailed backlog below
preserves the reasoning, experiments, and historical context, but the next
implementation chunks should pull from this ranked list first.

Evidence sources for current promotion state are
[performance-wins.md](performance-wins.md),
[reviewed-local-evidence.md](reviewed-local-evidence.md),
[roadmap-status.md](roadmap-status.md), and the README's
[current local performance snapshot](../README.md#exactness-and-performance).

## Recent Execution Status

June 4, 2026 updates:

- PR #10 closes the helper/evidence infrastructure for ranks 23, 25, 26, 27,
  32, 34, 35, 37, 40, and 41. Those closures add benchmark/schema/tooling
  surfaces for generated reducer identities, residue-channel fusion metadata,
  multi-modulus pack layout metadata, fused one-shot comparison surfaces,
  next-op hints, output policy metadata, AUTO selector explanations,
  plan/allocation fingerprints, counter report ingestion, and target-keyed
  review grouping. They are not promoted speedup claims by themselves.
- Current-v2 bounded-u64 square and selected skinny reruns replaced the stale
  vector-leadership assumption. The promoted local AUTO keys are now the
  measured CK 512 square winner and the measured hipBLASLt 256x1x4096 skinny
  winner; smaller square cases still prefer CPU or Direct HIP.
- Finite-u8 generic prime/composite 512 was promoted for three local AUTO keys:
  ring 127 through rocWMMA, field 127 through CK, and ring 253 through
  hipBLASLt. The field-127 hipBLASLt capture remains validation debt because
  required hipBLASLt events were missing.
- Finite-u8 2048 hot-modulus work has exploratory GPU evidence but no CPU
  baseline-backed promotion yet. Keep it as bottleneck classification until
  release review can include the required CPU/reference coverage.
- Branch-local native-to-RNS and vector-to-RNS work closes the bridge exposure
  lane: native bounded device output can now be materialized into RNS device
  residues for Direct-HIP consumers, and benchmark/schema/sweep coverage exists
  for native-to-RNS, vector-to-RNS chain, and reusable consumer-B chain
  captures. This is not yet a selector promotion or public output-domain API.
- Branch-local reusable-B scenario work advances the repeated/chain workload
  corpus with reusable-B RNS-chain scenarios and larger bounded reusable-B
  coverage. The next gate is release-reviewed, same-contract promotion evidence
  for the workload shapes that these scenarios now expose.
- Evidence database work now closes the rank 1 analysis surface: validated
  captures get row-level roofline targets and optimization hints, and generated
  evidence summaries include corpus-level and GPU-only Roofline Priority tables
  ranked by measured bottleneck time. Broad temp-corpus ingestion can now use
  `--skip-invalid` to record stale captures without blocking valid current
  evidence. This is planning infrastructure only, not a speedup claim.
- Scenario corpus work now includes an explicit `layout-search` family covering
  RNS final-export, RNS-next-op, exact-wide prefix-20 limb export, exact-wide
  lazy RNS continuation, finite-u8 ring/field layouts, and strict wrap64 byte
  layout comparisons. This closes the scenario surface for rank 28 but still
  requires release A/B evidence before any layout promotion.
- Current branch-local work after PR #10 also closes the first practical queue
  control surfaces: evidence/roofline prioritization, robust temp-corpus
  ingestion, layout-search scenario generation, native-to-RNS bridge exposure,
  vector-to-RNS chain captures, reusable consumer-B chain captures, reusable-B
  RNS-chain scenarios, and large bounded reusable-B scenario coverage. Treat
  those as completed benchmark/tooling surfaces, not as automatic runtime
  routing or public API promotions.
- Large-shape validation now has a dedicated `large-release-validation`
  scenario family. It emits the missing CPU/direct/vector/accelerator
  comparator matrix for 2048 bounded i64/u64, setup-contract reuse-B 2048,
  exact-wide signed/unsigned 2048, finite-u8 hot-modulus 2048, and wrap64 2048.
  This closes the executable validation surface for the 2048 side of the large
  matrix; the expensive release captures still need to be run before promotion.
- Adaptive bounded current-v2 validation is no longer blocked at the capture
  plumbing layer. The branch fixes compact per-tile CPU storage acceptance,
  one-shot compact schedule allocation, CK/rocWMMA zero-row/column-product
  schedule flag handling, runtime vector comparator schema policy, and
  `adaptive-bands` CPU comparator generation. One-repeat smoke captures under
  `temp/perf-work-queue/adaptive-current-v2-smoke/` are schema-valid for CPU,
  Direct HIP, runtime vector ALU, CK, and rocWMMA at bounded-i64
  256x256x512, with required GPU events for the GPU records. This is capture
  validity evidence only; the 512/1024 release rerun is still required before
  replacing the old adaptive winner.

| Rank | Work Item | Why Now | Evidence Gate | Disposition Rule |
|---:|---|---|---|---|
| 1 | Closed analysis lane: evidence database and roofline priority summary | The repo needed a compact analysis layer that ranks where corpus time is going | `tools/evidence_database.py` validates schema-v4 captures, can skip/report stale temp captures, joins review/scenario/ISA inputs, writes ignored JSON/CSV/Markdown, and ranks global plus GPU-only roofline priority groups by measured bottleneck time | Closed as planning infrastructure; continue using it to choose release A/B work, not as a speedup claim |
| 2 | Closed scenario-surface lane: benchmark corpus for repeated, small, skinny, chain, finite, exact-wide, wrap64, CAS, and FHE-proxy workloads | Shape-only GEMM comparisons are no longer enough to choose real performance work | Scenario mode now covers repeated-B, small one-shot, many-small, skinny/GEMV, RNS-chain, finite distributions, finite generic moduli, exact-wide export, wrap64 carry/large probes, CAS/FHE proxies, native/vector-to-RNS, fused/pack/fusion, generated-prefix, adaptive, large-shape, and layout-search families | Closed as corpus infrastructure; keep expanding only for newly discovered workload classes with exact CPU checks and no unsupported product-scope implication |
| 3 | Partially completed 2048/4096 large-shape matrix | Current evidence is heavy on 512/1024, and larger shapes show whether RNS8 is launch/export-bound or throughput-bound | Large-shape and large reusable-B scenario coverage exists; `large-release-validation` now emits the CPU/direct/vector/accelerator 2048 comparator matrix missing from exploratory captures; current evidence-database GPU priority points at large bounded compute, pack, and export time | Run the 2048 validation matrix before promotion; keep 4096 as throughput classification unless a CPU/reference release baseline is intentionally budgeted |
| 4 | Completed current-v2 bounded-u64 rerun | Stale vector-leadership claims needed replacement after Direct-HIP and selector changes | Current-code release review covered square 64/128/512/1024 plus selected skinny cases | Closed for current claim refresh; follow-on tuning should target CK 512, hipBLASLt 256x1x4096, and Direct-HIP-favored 128/1024 cases |
| 5 | Partially unblocked current-v2 adaptive bounded rerun | The adaptive bounded-i64 winner uses an older rocWMMA tiled-v1 identity, but the branch now fixes the capture blockers for CPU compact storage, CK/rocWMMA schedule flags, runtime vector comparator schema, and CPU comparator generation | One-repeat bounded-i64 256x256x512 smoke captures are schema-valid for CPU, Direct HIP, runtime vector ALU, CK, and rocWMMA with required GPU events where applicable; release review with current selected-kernel identities is still required at 512/1024 | Promote only event-valid current-v2 release winners; mark old v1 evidence historical and keep the smoke results out of cache decisions |
| 6 | Bounded-i64 Direct-HIP 512 tuning | Current reviewed 512 winner is Direct HIP, so local gains come from the correctness baseline | Same-contract 512 release A/B against `direct_hip_tiled_active_prefix_rns_gemm_v2` | Route only if end-to-end median improves and events explain the win |
| 7 | Bounded-i64 hipBLASLt 1024 tuning | 1024 has the only current bounded-i64 accelerator cache win, but it is narrow versus Direct HIP | Release A/B against current hipBLASLt v2 and Direct-HIP baseline | Keep cache entry only if correctness, event timing, and setup-inclusive end-to-end win survive |
| 8 | Many-small persistent/grouped workload path | Batching many 64/128/skinny exact jobs into one grouped path is likely more valuable than more isolated single-GEMM tuning | Grouped scenario captures with CPU/direct-HIP baselines, independent-call comparison, per-task correctness, and setup/error aggregation | Promote only when grouping beats independent calls including queue/setup overhead |
| 9 | Partially completed RNS-chain internal path with residue-current outputs | `RNS GEMM -> RNS GEMM -> final export` can skip intermediate reconstruction and is one of the cleanest structural wins | Current branch exposes native-to-RNS conversion, vector-to-RNS consumers, reusable consumer-B chains, and reusable-B RNS-chain scenarios; release proof still needs same-contract timing and one final exact CPU comparison | Promote only when skipped export is semantically visible, setup/reuse policy is explicit, and CPU reference remains exact |
| 10 | Direct-HIP prefix-9/prefix-20 fusion | Doing fewer launches and materializations in the correctness baseline is higher leverage than chasing more accelerator variants | Prefix-9 bounded and prefix-20 exact-wide captures with event-visible launch/materialization reduction | Keep variants only when prefix-specific end-to-end wins beat current grouped/generic paths |
| 11 | Exact-wide export specialization | Fixed limb counts, compact D2H, status elision when impossible, and prefix-specialized CRT are likely practical wins | Same-contract export-heavy captures by limb count with GPU events and checksum/limb equality | Promote only setup-inclusive export path wins, not isolated copy improvements |
| 12 | Exact-wide lazy-export scenarios | Exact-wide chains may win by delaying reconstruction rather than accelerating a single GEMM | Scenario captures for chained, residue-current, and final-export workflows | Promote lazy/export changes only when output-domain metadata proves the same contract |
| 13 | Exact-wide 64/128/2048/4096 and limb-count release matrix | Current exact-wide claims cover only 512/1024 for selected signedness cases | `large-release-validation` now covers exact-wide signed/unsigned 2048 with 4-limb CPU/direct/accelerator comparators; small 64/128, 4096, and limb-count variants still need release review | Install cache entries only for exact shape/semantic/limb keys with required events |
| 14 | Finite-u8 2048/4096 hot-modulus release matrix | 2048 GPU-only evidence now exists, but CPU-backed release proof is still missing | `large-release-validation` now covers ring 251/255/256 and field 251 at 2048 with CPU/direct/accelerator comparators; tolerable 4096 remains exploratory | Promote hot-modulus cache keys only when they beat CPU and Direct HIP with required events |
| 15 | Partially completed finite-u8 generic prime/composite coverage | Generic 512 now has promoted local keys, but broader sizes and the field-127 hipBLASLt event gap remain | Minimal generic prime/composite correctness and timing evidence with selector explanations | Keep non-promoted generic paths experimental until they prove feature value or fill unsupported contracts |
| 16 | Closed helper lane: vector/native-to-RNS bridge | Native bounded output can now feed Direct-HIP RNS consumers instead of becoming a dead end | Device-to-device native-to-RNS kernels plus native-to-RNS and vector-to-RNS benchmark/schema/sweep coverage exist; release A/B and selector policy still need proof | Closed as bridge exposure; route only explicit conversion paths with stale-kernel schema rejection |
| 17 | Reframe Vector N=1 GEMV selector and cache policy | Current-v2 bounded-u64 skinny evidence did not preserve the old vector-leading assumption | Release matrix for N=1 families plus selector explanation output | Route only for gated N=1/K thresholds where vector beats CPU, Direct HIP, and accelerator alternatives end-to-end |
| 18 | Advanced reuse/prepack workload contract promotion | Repeated-A/B wins are real, and this branch now exposes reusable-B chain, reusable consumer-B, and large-shape reusable-B scenarios; the contract still needs repeat count, setup amortization, source identity, and lifetime semantics | Define review keys for setup cost, repeat count, operand identity, cache lifetime, stale-source rejection, and chain consumer identity | Keep reuse out of AUTO until workload contract and break-even policy are explicit |
| 19 | hipBLASLt A/B reuse conversion from benchmark win to explicit workload contract | hipBLASLt A/B reuse is the strongest event-valid reuse signal | Public or benchmark-contract design plus release evidence including setup amortization | Promote only when one-time setup and repeated-call semantics are visible and correct |
| 20 | Advanced Direct-HIP reuse-A/reuse-B expansion beyond uniform-small bounded cases | Direct-HIP reuse now has reusable-B chain, reusable consumer-B, and large bounded scenario coverage, but reuse-A and non-bounded profiles remain thin | Release evidence for adaptive, finite, exact-wide, non-uniform inputs, and RNS-chain consumers | Keep per-profile routing explicit; do not infer reuse from C++ type or backend alone |
| 21 | Bound-discovery proof-mask setup-inclusive release matrix | Proof masks reduced scan cost but need broader end-to-end proof | Release captures comparing static profile, input-scan, tile-bound, and proof-mask modes | Promote only when scan cost plus execution savings beat setup-inclusive baselines |
| 22 | Zero-tile and zero-row/column skip expansion beyond Direct-HIP | Direct-HIP has proof-mask execution skips; other backends are incomplete | CPU, CK, rocWMMA, and hipBLASLt correctness/event evidence for skipped work | Extend only where event traces show skipped backend work, not just metadata |
| 23 | Closed helper lane: generated prefix-specific reducers for bounded prefixes 1..9 | PR #10 adds fixed-prefix reducer identity, dispatch, and ISA-gate surfaces | Release A/B still needs prefix-specific end-to-end proof against generic reducers | Closed as infrastructure; reopen only for measured prefix-specific speedup work |
| 24 | Shared epilogue DSL for Direct-HIP, hipBLASLt, CK, and rocWMMA reducers | Reducer specialization is duplicated across accelerators | One DSL-generated family with schema fixtures and stale-cache rejection | Adopt only if generated names, metadata, and ISA reports remain inspectable |
| 25 | Closed helper lane: residue-channel fusion experiments | PR #10 adds benchmark-only residue-channel fusion metadata and stale-schema rejection | Microbenchmarks and release captures still need to show fewer launches/materializations | Closed as exposure; continue only if fusion wins end-to-end after pack/export effects |
| 26 | Closed helper lane: multi-modulus pack experiments | PR #10 adds pack-layout and residue-group metadata for comparison keys | Same-contract pack event comparisons still need required correctness checks | Closed as metadata surface; keep variants only if pack savings survive GEMM/export timing |
| 27 | Closed helper lane: fused pack+GEMM for small one-shot bounded and finite workloads | PR #10 adds comparison surfaces for one-shot/transient fused-path experiments | Release captures still need 64/128 bounded and finite one-shot proof | Closed as benchmark surface; promote only if fused path beats CPU, Direct HIP, and current accelerator winner |
| 28 | Closed scenario surface: end-to-end layout search across RNS, finite, exact-wide, and wrap64 | Layout decisions now affect pack, GEMM, reducer, export, and reuse together | `layout-search` emits layout-metadata captures for bounded RNS final export, RNS-next-op, exact-wide prefix-20, finite-u8 ring/field, and strict wrap64 byte-limb paths | Closed as benchmark surface; keep layout variants only after complete same-contract release evidence and event attribution |
| 29 | Persistent/grouped scheduler for adaptive prefix groups | Adaptive prefix groups need launch and scheduling amortization beyond simple tile skipping | Grouped scenario captures with CPU/direct-HIP baselines and per-task correctness | Promote only when grouping beats independent calls including queue/setup overhead |
| 30 | HIP Graph replay for repeated fixed-shape pack/GEMM/export | Repeated workflows can remove launch overhead without changing math | Internal graph replay benchmark with fixed-shape identity and handle lifetime checks | Keep internal until exact status/error behavior matches ordinary calls |
| 31 | Host API batching for many-small workloads | Some workloads may be too dynamic for graph capture | Benchmark-only begin/enqueue/end flow with explicit status aggregation | Promote only if batching wins while preserving deterministic per-operation errors |
| 32 | Closed helper lane: next-op and lazy-export metadata | PR #10 adds requested next-op and output-domain planning metadata to schema-v4 captures | Chain/lazy-export scenarios still need release proof with exact final CPU comparison | Closed as metadata; keep advisory until public API semantics are clear |
| 33 | Reconstruction backend variants for GPU CRT/export | Current wins often move with export/status timing | Release A/B for GPU CRT, compact export, status handling, and host scatter variants | Promote only setup-inclusive export path wins, not isolated copy improvements |
| 34 | Closed helper lane: explicit padded/contiguous output policy | PR #10 records output policy and status/export handling in schema-v4 captures | Release captures still need padded versus contiguous A/B with event-visible D2H/status phases | Closed as metadata; default-enable only per semantic/layout where repeated evidence wins |
| 35 | Closed helper lane: CPU/GPU hybrid AUTO selector explanations | PR #10 reports AUTO cache hits, fallbacks, and rejection reasons for benchmark/inspect tooling | Selector behavior still needs reviewed cache evidence before routing changes | Closed as explanation surface; promote selector behavior only without weakening evidence gates |
| 36 | AUTO cache shape-family recommendation layer | Exact-shape cache hits are too narrow for real workloads | Reviewed shape-family policy layered above exact-shape entries | Keep disabled until family recommendations cannot cross semantic/layout/target boundaries |
| 37 | Closed helper lane: device plan cache and workspace arena evidence | PR #10 adds plan/workspace fingerprints and post-warmup allocation evidence | Benchmarks still need to separate planning, allocation, pack, GEMM, and export for real cache work | Closed as evidence surface; promote only when cache identity includes plan, target, backend, and source versions |
| 38 | Verification-cost reduction for repeated exact workloads | Exact checks can dominate validation of repeated scenarios | Validation modes that reuse CPU/reference structure without reducing correctness | Keep tooling-only unless every promoted capture still has exact CPU differential coverage |
| 39 | Error-detecting exact fast path with explicit metadata | Probabilistic or checked fast paths are research-only but may unlock workloads | Research captures with verification metadata and false-negative policy | Never make default exact API probabilistic; keep explicitly research-marked |
| 40 | Closed helper lane: hardware-counter/RGA ingestion into evidence reports | PR #10 adds counter report tooling and ISA/capture cross-links | Optional counter/ISA summaries can now attach to reviewed captures | Closed as tooling; use as explanation evidence only, never as a replacement for timing and correctness gates |
| 41 | Closed helper lane: architecture-specific kernel namespaces and target-keyed variants | PR #10 adds target-id, namespace, configured target, runtime version, and review grouping metadata | Target-specific cache entries still need real host evidence | Closed as schema/tooling; promote non-`gfx1100` only after evidence on that target |
| 42 | Linux/Instinct/toolchain matrix gates for future non-Windows promotion | Production platform readiness is still unproven | Real Linux ROCm/Instinct builds, tests, profiling, and release captures | Keep Windows claims local until target-specific evidence exists |

## Validation Debt

| Debt | Why It Matters | Required Refresh |
|---|---|---|
| Adaptive bounded current-v2 still needs release-review evidence after capture unblock | The branch now makes CPU, Direct HIP, runtime vector ALU, CK, and rocWMMA adaptive-bands smoke captures schema-valid, but the proof is one-repeat plumbing evidence rather than release performance evidence | Current-v2 adaptive release review at 512/1024 with schema-valid CPU, Direct HIP, vector, and accelerator records |
| Adaptive bounded-i64 1024 winner uses older rocWMMA tiled-v1 identity | Current selected-kernel identities and reducer paths changed | Current-v2 adaptive release review before promotion or cache install |
| Native-to-RNS and vector-to-RNS chain captures are helper surfaces, not routing proof | The branch can expose and validate bridge/chain scenarios, but AUTO/public routing still needs same-contract release wins | Release review for bridge and chain scenarios with explicit conversion timing, reuse setup cost, and final exact CPU comparison |
| Large 2048/4096 captures are bottleneck-classification evidence, not promotion evidence | Existing large bounded, large reusable-B, exact-wide, finite-u8, and wrap64 captures are useful for ranking work, but several are exploratory or missing CPU/reference and complete baseline coverage | Run `large-release-validation` for the 2048 CPU-backed matrix, then keep 4096 claims exploratory unless a full CPU/reference release pass is explicitly budgeted |
| Exact-wide 64/128 evidence is historical | Current v2 exact-wide evidence covers 512/1024 only | Current-v2 release review for 64/128 and selected limb counts |
| Field-251 512 hipBLASLt near-win lacked required events | Timing-only near wins cannot enter durable cache | Rerun with complete hipBLASLt GPU events or keep Direct HIP |
| Field-127 generic hipBLASLt capture lacked required events | Generic finite-u8 promotion cannot rely on a timing-only hipBLASLt field path | Rerun with `hipblaslt_int8_i32_matmul` and `hipblaslt_i32_to_residue_reduce` events or keep CK for field 127 |
| Reuse/prepack wins use explicit reuse contracts | They are not same-contract AUTO replacements for one-shot calls | Workload-level promotion policy with setup-inclusive break-even and source identity |

## Do Not Chase Next

| Path | Current Disposition | Reason |
|---|---|---|
| Vector 1024 repeated-A and full A+B reuse | Deprioritize | Latest reuse comparison regressed setup-inclusive and steady-state timing |
| rocWMMA 1024 repeated-B reuse | Deprioritize | Latest reuse comparison lost after setup cost |
| Wrap64 rocWMMA matrix-engine candidate | Deprioritize | Correct but loses to Direct-HIP v4 at every reviewed 64/128/512/1024 shape |
| Wrap64 Direct-HIP colpair experiment as default route | Deprioritize | Narrow GEMM improvement did not beat v4 end-to-end |
| Wrap64 pinned export staging as default route | Deprioritize | Forced staging lost badly at 512 versus default policy |
| CK/rocWMMA v1 bounded/exact-wide cache promotion | Do not promote | Selected-kernel identities are stale under current v2 reducer paths |
| Raw smoke or discovery captures as durable claims | Do not promote | They lack the release-review and required-event gates for public claims |

## Detailed Backlog And Research Notes

### Imported Maximum-Performance Roadmap Triage (June 3-4, 2026)

The external maximum-performance checklist was deduped against the current
repo before being queued. Items already handled by prior cleanup and
instrumentation work are not re-queued here: rocWMMA public naming, backend
schema v4, deep CK/rocWMMA/vector-ALU event labels, GPU event and ISA report
tools, release evidence summaries, install/package smoke, finite-u8 backends,
vector-ALU runtime visibility, public setup docs, and the existing reusable-B
and hipBLASLt workspace-local prepack evidence.

Implemented in the June 4, 2026 prefix-policy metadata pass:

- Default global RNS plans now execute the minimum proven prefix rather than
  always executing the requested max prefix. The requested ceiling remains in
  `rns8_gemm_desc.max_prefix`; the selected execution prefix is `plan.prefix`.
  Fixed-prefix experiments opt in with `RNS8_PLAN_FORCE_FIXED_PREFIX`.
  Code references: `include/rns8/rns8.h`, `src/core/api_plan.cpp`,
  `src/core/api_matrix_workspace.cpp`, and `src/core/api_gemm.cpp`.
- Per-tile plans can also force the requested prefix for controlled A/B
  captures; otherwise each tile group selects its required prefix. This keeps
  adaptive evidence separate from fixed-requested evidence.
  Code references: `src/core/api_plan.cpp` and
  `src/core/plan_lowering.cpp`.
- Benchmarks now expose requested prefix, selected prefix, prefix policy, and
  residue-plane skip fraction. Persistent benchmark matrix allocation uses the
  schedule max selected prefix, so pack/GEMM/export work reflects the selected
  plan instead of just reporting it.
  Code references: `benchmarks/rns8_bench.cpp`,
  `tools/benchmark_schema.py`, `tools/benchmark_sweep.py`, and
  `tools/result_compare.py`.

Implemented in the June 4, 2026 bound-discovery pass:

- `rns8-bench` now has an explicit `--bound-source static-profile|input-scan`
  switch for bounded global i64/u64 benchmark captures. The `input-scan` path
  scans seeded A/B inputs before plan creation, computes row/column
  absolute-summary candidates, records zero row/column counts, selects the
  tight safe global bound, and feeds that bound into the existing prefix
  policy.
- Public bounded i64/u64 plans now accept `RNS8_BOUND_INPUT_RANGE_AND_K`.
  Callers provide trusted `lhs_bound` and `rhs_bound` input magnitude
  contracts; plan creation derives `k * lhs_bound * rhs_bound` as the effective
  global output bound, records it in schedule metadata, and keeps persistent
  matrices on ordinary global signed/unsigned bounded storage.
- Bound discovery is timed as a first-class `global_bound_scan` phase with
  schema-validated raw timings, top-level averages, phase notes, and phase
  availability metadata. Static profile captures remain the default and legacy
  captures without `bound_source` are treated as static-profile evidence by
  comparison tooling.
- Sweep and compare tooling now carry bound-source metadata through candidate
  evidence, cache schedule hashes, and same-contract comparisons so discovered
  bounds are not accidentally mixed with static-profile captures.

Implemented in the June 4, 2026 proven-zero tile skip pass:

- Per-tile bounded plans now have an explicit
  `RNS8_PLAN_ALLOW_PROVEN_ZERO_TILE_SKIPS` opt-in for zero tile bounds produced
  by trusted exact scans. Without the opt-in, zero bounds stay ordinary range
  contracts and still produce range errors when the actual output is nonzero.
- Schedule entries can mark `RNS8_TILE_SCHEDULE_ZERO_OUTPUT`; schedule info
  aggregates the flag, workspace fingerprints include it, and stale/mutated
  plans are rejected if flags do not match copied tile bounds.
- CPU reference, direct HIP scheduled GEMM/export, direct HIP host tiled
  fallback, CK, and rocWMMA tiled paths materialize zero output tiles without
  running per-tile GEMM work for selected residue planes. Residue planes above
  the selected prefix remain untouched.
- Direct-HIP active-prefix schedules now exclude zero-output tile entries from
  the compact GEMM workspace schedule. Nonzero adaptive schedules now upload
  only that compact active schedule, not the unused public row-major device
  schedule. Mixed zero/nonzero schedules keep the public device schedule for
  tile-local zero materialization, while uniform all-zero schedules skip both
  schedule buffers and use the `direct_hip_zero_output_tile_memset` event label
  for a contiguous selected-plane zero fill. Adaptive zero-skip plans advertise
  `direct_hip_tiled_active_prefix_zero_skip_rns_gemm_v3`, while nonzero
  adaptive plans keep `direct_hip_tiled_active_prefix_rns_gemm_v2`.
- Direct-HIP uniform all-zero scheduled GEMM no longer requires current A/B
  resident residues: the trusted tile-bound schedule proves no backend kernel
  reads those operands, so `rns8_gemm_rns` can materialize the zero output from
  allocated matrix storage alone. Matching `rns8-bench` captures skip
  per-repeat A/B packing for that contract and report zero-valued `pack_h2d`,
  `pack_kernel`, and `pack` event phases.
- Direct-HIP all-zero scheduled exports now skip CRT/status work entirely:
  the compact native export buffer is zero-filled, copied back, and the export
  path no longer uploads unused tile schedule/bounds metadata before taking the
  all-zero branch. Benchmark event captures report zero-valued
  `crt_export_status_memset` and `crt_export_status_d2h` phases for that
  all-zero schedule contract.
- `rns8-bench` enables the skip only for exact seeded per-tile bound prepasses
  and reports zero tile counts, zero tile fraction, selected residue-plane skip
  counts, and schedule flags in `schedule_metadata`. Schema and sweep cache
  keys validate and preserve those fields.
- Per-tile bounded plans now also have an explicit
  `RNS8_PLAN_ALLOW_PROVEN_ZERO_ROW_COL_SKIPS` opt-in for trusted zero A-row and
  B-column proof masks. Schedule entries mark
  `RNS8_TILE_SCHEDULE_ZERO_ROW_COL_PRODUCT` for tiles intersecting those masks,
  plan/workspace fingerprints include the copied masks, Direct-HIP workspaces
  upload them, and scheduled GEMM/export can write proven zero row/column
  products without doing per-cell dot products or CRT reconstruction.
- Direct-HIP adaptive plans advertise distinct row/column skip kernel families:
  `direct_hip_tiled_active_prefix_zero_row_col_skip_rns_gemm_v1` for row/column
  proofs only and
  `direct_hip_tiled_active_prefix_zero_tile_row_col_skip_rns_gemm_v1` when
  zero-output tile skips and row/column proof skips are both active. Benchmark
  schema v4, release-review grouping, result comparison, and autotune-key
  validation now preserve the proof counts so masked and unmasked evidence
  cannot be mixed.

Implemented in the June 4, 2026 accumulator-safety metadata pass:

- `rns8_get_plan_backend_info` now reports the selected backend's accumulator
  input domain, signedness, accumulator type, modulus policy, K-block size,
  K-block cap, max product, and safety status. CK declares its canonical
  32,768-term `int32` cap; CPU/direct HIP/hipBLASLt/rocWMMA RNS and finite
  paths declare the existing 65,536-term signed `int32` cap; vector-ALU and
  wrap64 byte-limb paths explicitly declare non-`int32` accumulator contracts.
- Autotune keys now include accumulator type, signedness, modulus policy,
  K-block size, and K-block cap before the selected kernel/epilogue identity,
  so reviewed cache entries cannot silently cross backend accumulator policies.
- Benchmark schema v4 now requires `backend_metadata.accumulator_safety`,
  rejects stale key fields, rejects unsafe `int8 x int8 -> int32` declarations
  before evidence can pass review, and keeps top-level `k_block_size` tied to
  the backend-specific contract.

Remaining high-value imported work goes at the front of the queue:

1. **Bound Discovery Pipeline**

   Benchmark-only global input scans now select tighter bounded i64/u64 global
   bounds from exact row/column absolute summaries before plan creation, and
   benchmark per-tile input scans now feed exact tile max-product discovery into
   the existing adaptive prefix and zero-tile skip schedule. Public plans accept
   trusted whole-input `lhs_bound`/`rhs_bound` contracts through
   `RNS8_BOUND_INPUT_RANGE_AND_K` and trusted tile-bound arrays through the
   per-tile bound contracts. The per-tile benchmark scanner now also builds
   nonzero A-row and B-column summaries before the exact cell scan, preserving
   exact tile bounds while skipping scan work for tile/row/column products that
   are already proven zero. A June 4, 2026 Windows `gfx1100` 512x512x512
   bounded-u64 adaptive-band capture reduced `tile_bound_scan` from 557635 us
   to 414379 us, a 1.35x prepass speedup, with the tile-bound hash, selected
   prefix, prefix groups, zero-output tile count, selected kernel, schema v4,
   and required GPU events unchanged. Row/column proof masks have since been
   promoted from benchmark scan metadata into an explicit trusted per-tile plan
   contract for Direct-HIP scheduled GEMM/export. The remaining work is to
   gather broader release `gfx1100` evidence, decide when these proof masks are
   setup-inclusive wins, extend the execution skip beyond Direct-HIP, and avoid
   inferring semantics from C++ types.

   Code references: current benchmark scans in
   `benchmarks/rns8_bench.cpp` (`compute_i64_tile_bounds`,
   `compute_u64_tile_bounds`), planner prefix selection in
   `src/core/api_plan.cpp`, matrix compatibility in
   `src/core/api_matrix_workspace.cpp`, and public descriptor definitions in
   `include/rns8/rns8.h`.

2. **Zero-Tile And Zero-Plane Execution Skips**

   Proven zero tile bounds now skip tiled GEMM/export work in CPU, direct HIP,
   CK, and rocWMMA paths. Direct-HIP scheduled GEMM now also keeps a compact
   active-prefix tile schedule in the workspace, so each modulus-plane launch
   visits only tiles whose selected prefix actually includes that plane instead
   of launching row-major blocks that immediately return for skipped planes.
   Direct-HIP now extends this from whole output tiles to explicit zero
   row/column products using copied proof masks, and schema/event smokes cover
   combined zero-tile plus row/column captures. The remaining work is to gather
   release evidence showing the skip counters translate into setup-inclusive
   end-to-end wins, extend the proof-mask execution path to other backends, and
   explore other provably unused selected-prefix ranges.

   Code references: tile schedule entries in `include/rns8/rns8.h`, schedule
   construction in `src/core/api_plan.cpp`, direct-HIP dispatch in
   `src/core/api_gemm.cpp`, direct-HIP kernels under `src/backend_hip_direct/`,
   CK kernels under `src/backend_ck/`, rocWMMA kernels under
   `src/backend_rocwmma/`, and export paths in `src/core/api_export.cpp`.

3. **HIP Graph And Async Executable Path**

   Repeated fixed-shape workloads should be capturable as executable graph
   shapes: pack, per-prefix GEMM, reducer/export, status, and D2H. Keep this
   internal first. Public async APIs should wait until graph replay has exact
   CPU/direct-HIP proof and clear handle lifetime rules. Graph identity must
   include backend, selected prefix, requested max prefix, K-block, tile shape,
   reuse mode, output mode, and device target.

   Code references: operation sequencing in `src/core/api_gemm.cpp`, workspace
   and matrix lifetime in `src/core/api_matrix_workspace.cpp`, benchmark phase
   timing in `benchmarks/rns8_bench.cpp`, and future public API surface in
   `include/rns8/rns8.h`.

4. **Generated Reducers And CRT Export Families**

   Generate modulus-specific and prefix-specific reducers instead of leaning on
   generic runtime division paths. The first production families should cover
   bounded selected prefixes 1..9, exact-wide common selected prefixes, finite
   moduli 251/255/256, and direct-HIP/accelerator epilogues that avoid INT32
   global stores where possible. Each generated variant needs selected-kernel
   naming, ISA gates, schema fixtures, and stale autotune rejection.

   Code references: modulus ladder and products in `src/core/moduli.cpp`,
   direct-HIP kernels under `src/backend_hip_direct/`, reconstruction in
   `src/reconstruct/crt.cpp`, selected-kernel metadata in
   `src/core/api_plan.cpp`, and ISA/report tooling in `tools/gpu_isa_report.py`.

   Helper-lane status: Direct-HIP fixed-prefix native pack dispatch now covers
   bounded prefixes 1..9 plus exact-wide prefix 20, and default-RNS Direct-HIP
   reduction sites route through fixed default-modulus reducers for the first
   20 moduli before falling back to the generic runtime reducer. Benchmark
   captures identify generated/fixed reducer evidence through
   `timing_metadata.generated_reducer_identity`, and schema fixtures reject
   stale generic identities for generated captures. Next optimization work
   should run ISA gates for the prefix-specific symbols, prove integer divide
   avoidance, and then compare end-to-end pack/GEMM/export captures for
   prefixes 1, 3, 5, 9, and 20 before promoting any selected-kernel name.
   Use `tools/check_generated_reducer_isa.py --object <hip_direct_object>
   --target gfx1100` for the Direct-HIP generated symbol/no-divide gate, and
   use `tools/gpu_isa_report.py --capture <capture.json>` for explanatory
   resource summaries.

5. **Architecture-Specific Kernel Namespaces**

   Split RDNA3 `gfx1100` tuning from future RDNA4 and CDNA work. Do not let a
   Windows RX 7900 XTX win become a Linux ROCm or Instinct claim. Kernel
   variants, launch bounds, wave size, LDS layout, and occupancy assumptions
   should be target-id keyed in both benchmark output and autotune cache keys.

   Current status: plan autotune keys, benchmark-owned synthetic keys, schema
   fixtures, and reviewed-cache installation now require explicit
   `target_id=...` key material. Remaining work is to add real target-specific
   kernel namespaces and promotion evidence for non-`gfx1100` families rather
   than inheriting local Windows timings.

   Helper-lane status: schema-v4 captures can now carry `target_variant` with a
   concrete target id, target namespace, configured target string, runtime
   versions, and review grouping key. New HIP helper-lane captures must include
   a concrete target id/namespace. This is namespace readiness only; it does not
   make `gfx11xx`, `gfx12xx`, or `gfx9xx/gfx94x` performance claims without
   host evidence on those targets.

   Code references: backend source roots `src/backend_hip_direct/`,
   `src/backend_ck/`, `src/backend_rocwmma/`, configured target metadata in
   `CMakeLists.txt`, capture validation in `tools/benchmark_schema.py`, and
   release review grouping in `tools/benchmark_sweep.py`.

6. **Hardware-Counter Promotion Gate**

   Add a non-promoting profiler ingestion lane for occupancy, VALU/MFMA/WMMA
   counts, LDS traffic, global load/store bytes, wave stalls, and achieved
   bandwidth. Use it to explain wins and blockers, not to replace exact output
   checks or host/GPU timing. Keep raw profiler dumps in `temp/`; durable docs
   should summarize reviewed counters only.

   Code references: event report tooling in `tools/gpu_event_report.py`, ISA
   reporting in `tools/gpu_isa_report.py`, schema promotion policy in
   `tools/benchmark_schema.py`, and release review in `tools/benchmark_sweep.py`.

   Helper-lane status: `tools/gpu_counter_report.py` validates schema-v4
   captures, optionally ingests JSON/CSV profiler counter exports and
   `tools/gpu_isa_report.py` summaries, and writes temp-only JSON/Markdown
   reports under `temp/gpu-counter-reports/`. `tools/gpu_isa_report.py` can
   cross-link a validated capture with `--capture`. Counter and ISA conclusions
   are explanation evidence only and cannot replace correctness, host timing,
   HIP event timing, or release baseline gates.

## Current Evidence Snapshot

- `hip-vector-alu-int64` is a real bounded i64/u64 runtime backend, but the
  current-v2 bounded-u64 refresh no longer treats it as the universal leader:
  the reviewed local winners split across CPU, Direct HIP, CK, and hipBLASLt by
  shape. It remains bounded-only and must not be generalized into exact-wide,
  finite, or wrap64 semantics.
- Bounded i64 has current Windows `gfx1100` v2 release-review evidence for 512
  and 1024. The June 4, 2026 seed `20260604` sweep kept 512 on Direct HIP
  `direct_hip_tiled_active_prefix_rns_gemm_v2` at 1851 us and selected hipBLASLt
  `hipblaslt_int8_i32_scratch_reduce_specialized_251_255_256_v2` at 1024 with a
  4174 us median, 1.09x faster than Direct HIP and 8.13x faster than vector ALU.
  The bounded-i64 local default runtime cache coverage contains only that
  reviewed 1024 hipBLASLt v2 entry; finite-u8 and exact-wide cache entries are
  tracked separately below.
- Current CK and rocWMMA RNS selected kernels have v2 common-modulus reducer
  identities after the shared epilogues gained explicit 256/255/251 reduction.
  The 512/1024 bounded-i64 v2 release captures are now complete and did not
  promote CK or rocWMMA. Older CK/rocWMMA v1 bounded/exact-wide timings remain
  historical for any shape or path that has not been rerun with the matching v2
  selected-kernel identity.
- The active hipBLASLt source path now uses
  `hipblaslt_int8_i32_scratch_reduce_specialized_251_255_256_v2`; previous
  `hipblaslt_int8_i32_scratch_reduce_baseline_v1` timings are historical and
  should not be mixed into new autotune cache evidence.
- Adaptive bounded i64 at 1024 has a reviewed historical rocWMMA winner:
  `rocwmma_i8_i32_signed_tiled_hot_residue_v1`. The current branch unblocks
  the current-v2 adaptive capture path at smoke-test scope: compact per-tile CPU
  storage, Direct HIP, runtime vector comparator records, CK, and rocWMMA all
  validate on a bounded-i64 256x256x512 adaptive-bands smoke. The reviewed
  adaptive winner still uses the older tiled v1 identity and must be rerun at
  release settings before promotion under the current selected-kernel identity.
- Finite-u8 has current Windows `gfx1100` v2 release-review winners for 64, 128,
  512, and 1024 across ring 251/255/256 and field 251. Seven event-valid entries
  are now installed in the local default cache: ring-251 128 and 1024 rocWMMA,
  ring-255 1024 CK, ring-256 128 and 512 rocWMMA, ring-256 1024 hipBLASLt, and
  field-251 1024 CK. The field-251 512 hipBLASLt near-tie is deliberately not
  promoted because its GPU event capture was incomplete; ring-255 64 is
  deliberately not promoted because CPU reference is faster.
- Exact-wide has current Windows `gfx1100` v2 release-review winners for 512 and
  1024. Three event-valid entries are installed in the local default cache:
  signed 512 rocWMMA, signed 1024 hipBLASLt, and unsigned 1024 CK. Unsigned 512
  stays on Direct HIP. Older 64/128 exact-wide evidence remains historical until
  rerun with current selected-kernel identities.
- Direct HIP `direct_hip_wrap64_byte_gemm36_u32acc_tiled_2d_v4` is the
  measured strict wrap64 GPU path for reviewed 64/128/512/1024 local
  `gfx1100` shapes. The internal rocWMMA wrap64 candidate matches
  checksums in candidate-inclusive release review but loses to direct HIP at
  every 64/128/512/1024 shape.
- rocWMMA has a narrow runtime reusable B cache for non-tiled RNS plans with
  `K <= 65536`. hipBLASLt now has workspace-local repeated-A and repeated-B
  prepack caches for fixed-prefix single-K-block RNS work. Neither is a broad
  production prepack cache. Current event-valid reuse wins and setup
  break-even points are tracked in [performance-wins.md](performance-wins.md).

## FHE/Lattice Alignment Notes

FHE and lattice-crypto systems are strong workload inspiration for this queue,
but they do not turn RNS8 into an FHE library. Modern CKKS/BFV/BGV
implementations are RNS-heavy and GPU FHE systems are dominated by NTT/INTT,
base conversion, key switching, rotations/automorphisms, rescale or modulus
switching, bootstrapping linear transforms, coefficientwise modular products,
memory residency, and key-material movement. Dense exact GEMM is an adjacent
and conditional opportunity, not the default FHE hot path.

RNS8-specific implications:

- Do not claim that the byte-sized RNS8 modulus ladder is an FHE coefficient
  modulus chain. FHE parameter sets require separate ring-dimension,
  NTT-prime, scale/error, security, and scheme metadata.
- Use FHE/lattice papers to sharpen RNS8 architecture: persistent residue
  domains, lazy export, base-conversion thinking, grouped scheduling,
  reducer/epilogue fusion, and scenario benchmarks.
- Track external-scenario metadata explicitly: `scheme_adapter`,
  `evidence_scope`, `semantic_contract`, `backend_family`, `wave_size`,
  `rocm_or_hip_sdk`, and `validated_on_real_target`.
- Treat CUDA FHE papers and libraries as design input until reproduced on AMD.
  Porting requires wave32/wave64, lane-mask, launch-bound, LDS/register,
  memory-coalescing, target-id, and HIP/ROCm-version audits.
- Add FHE-derived scenario proxies for NTT/INTT pressure, key-switch digit
  aggregation, rotation-heavy linear transforms, CKKS rescale/mod-drop chains,
  BFV/BGV explicit-modulus arithmetic, bootstrapping stages, and
  encrypted-inference linear layers.
- Keep cryptographic security, RLWE parameter selection, noise growth,
  side-channel behavior, and decryption safety out of RNS8 performance claims
  unless a future FHE-specific project adds those contracts explicitly.

See `docs/fhe-lattice-alignment.md` for the source-ranked research synthesis
behind these notes.

## Computational Algebra Alignment Notes

- RNS8 maps most directly to exact dense GEMM and to small explicit finite
  rings or prime fields. It is not a general computer algebra system, a
  polynomial/NTT library, or a full finite-field BLAS/LAPACK replacement.
- Dense modular GEMM is a real kernel beneath exact rank, determinant, solve,
  nullspace, characteristic/minimal polynomial, and rational-reconstruction
  workflows. Those workflows also need factorization, triangular solve,
  CRT/CRA, verification, and certificate phases that RNS8 does not yet expose.
- Exact-linear-algebra phase labels should be precise: PLUQ/CUP/PLE
  rank-profile work, echelon recovery, determinant, inverse, solve, nullspace,
  characteristic/minimal polynomial, Freivalds verification, and certificate
  workflows are adjacent phases around GEMM unless RNS8 implements them
  explicitly.
- Symbolic-computation labels need even stronger boundaries. Dense F4
  finite-field matrices and FGLM multiplication-matrix phases can be adjacent
  dense-LA scenarios; sparse F4, F5 signature control, resultants,
  subresultants, NTT polynomial multiplication, and CAS-wide workflows are not
  dense-GEMM evidence.
- Finite-u8 should distinguish `Z/qZ` rings from prime fields `GF(p)`.
  Extension fields such as `GF(2^e)` and word-size prime fields are not current
  RNS8 contracts.
- Polynomial workloads should be scenario and lowering vocabulary first:
  NTT/FFT, product trees, remainder trees, interpolation, modular composition,
  subresultants, Sylvester matrices, and polynomial matrices do not imply that
  the current dense GEMM core replaces polynomial kernels.
- External libraries such as FFLAS-FFPACK, Givaro, LinBox, FLINT, NTL, Sage,
  Nemo, Magma, M4RI, and M4RIE are oracle/reference/comparison sources, not
  required production backends.
- CUDA exact-algebra artifacts such as Linac and CUMODP are design inputs and
  port-risk studies until HIP-native kernels produce target-specific RNS8
  correctness and timing evidence.
- CAS domain and coercion models should be used as scenario metadata, not
  inherited semantics. Track parent/domain family, coefficient ring, finite
  modulus, prime/composite status, extension degree, exactness mode,
  coercion/export policy, and oracle role explicitly. AUTO backend selection is
  not algebraic coercion.
- Treat CAS systems as workload and oracle ecosystems. Sage, Magma, Maple,
  Wolfram, Singular, Macaulay2, GAP, Oscar/Nemo/Hecke, PARI/GP, and Normaliz
  can classify phases and provide external comparison outputs, but RNS8 should
  not imply CAS-wide correctness, implicit coercion, symbolic orchestration, or
  package-runtime behavior from GEMM evidence.
- Do not use phrases such as "CAS-correct", "certified exact LA", or "secure
  probabilistic verification" for raw GEMM evidence. Certificates,
  Freivalds-style checks, bad-prime handling, CRA early termination, and
  redundant residues are scenario or research metadata unless implemented as
  explicit APIs.

## Ordered Work Items

### 1. Adaptive Prefix Minimization

This is the largest RNS8-native win because it deletes whole residue GEMMs
instead of only making them faster.

Technical direction:

- Move beyond fixed-prefix and coarse per-tile schedules toward the minimum
  exact prefix per tile, tile group, row/column norm class, or repeated
  workload profile.
- Use row absolute sums, column absolute sums, tile max bounds, finite
  histograms, zero density, and caller-provided bounds as plan inputs when the
  scan cost can amortize.
- Add plan metadata that records why a tile group selected a prefix and lets
  benchmark review compare "planes computed" against the fixed-prefix
  baseline.
- Treat modulus order as a performance object. Cheap reducers, high
  contribution moduli, sign/range helper channels, and CRT locality can justify
  a different execution order than the default ladder order, as long as the
  public mathematical prefix remains explicit.

RNS8-specific notes:

- The existing direct-HIP adaptive correctness path already proves tile-local
  prefix scheduling. The next step is to make adaptive selection shape-aware,
  data-profile-aware, and accelerator-friendly.
- Bounded-u64 reviewed release matrices show vector-ALU remains difficult to
  beat. Adaptive prefix work should especially target bounded-i64 and
  exact-wide-ish distributions where plane deletion can beat native vector ALU.
- Exact-wide export should reuse prefix and bound analysis to avoid exporting or
  checking limbs the contract cannot reach.

Likely first slices:

- Add a benchmark workload family for adaptive real distributions, not only the
  default 65x65x64 and 1024x1024x1024 cases.
  Implemented as `rns8-bench --input-profile adaptive-bands` plus
  `tools/benchmark_sweep.py --include-adaptive-workloads`, which adds
  profile-driven 256, 512x1024 rectangular, and 1024 adaptive cases while
  preserving the old uniform default-adaptive cases.
- Add row/column/tile-bound scan timing as explicit benchmark phases.
  Implemented for current per-tile bounded captures as `tile_bound_scan` host
  timing, with schema-required raw timing, summary, phase-order, and
  phase-availability metadata.
- Use row/column summaries inside exact per-tile bound discovery. Implemented
  for bounded i64/u64 benchmark scans: the prepass marks nonzero A rows and B
  columns, skips whole tile scans when either side is all zero, and skips known
  zero rows/columns inside mixed tiles while keeping the existing exact
  output-cell maximum for every nonzero product tile. Release evidence under
  `temp/perf-work-queue/tile-bound-zero-shortcut/` shows a 1.35x
  `tile_bound_scan` speedup on the 512 bounded-u64 adaptive-band case with
  unchanged tile-bound hash and adaptive schedule metadata; the bounded-i64
  sibling capture is schema-valid and event-valid after the same scanner change.
- Collapse uniform per-tile schedules back to fixed-prefix dispatch without
  duplicating the fixed-prefix implementation.
  Implemented for no-op per-tile captures where every tile still requires the
  existing full bounded prefix; uniform reduced-prefix schedules remain
  materialized until their tiled dispatch path is independently validated as a
  net win.
- Compact direct-HIP active-prefix scheduled launches. Implemented for
  per-tile direct-HIP RNS GEMM as
  `direct_hip_tiled_active_prefix_rns_gemm_v2`: the workspace stores a compact
  per-modulus active-entry schedule and the GEMM dispatch uses it to avoid tile
  blocks for residue planes that a tile did not select. The public row-major
  device schedule is no longer uploaded for nonzero adaptive schedules because
  the backend validates against the host schedule and launches from the compact
  active schedule.
- Compact Direct-HIP zero-output schedules. Implemented as
  `direct_hip_tiled_active_prefix_zero_skip_rns_gemm_v3`: zero-output public
  schedule entries are omitted from the active GEMM schedule and handled by the
  event-visible `direct_hip_zero_output_tile_memset` operation before nonzero
  scheduled GEMM work. Mixed zero/nonzero schedules still upload the public
  device schedule for tile extents; uniform all-zero schedules use a contiguous
  selected-plane memset and report zero GEMM schedule workspace bytes. Smoke
  evidence lives under
  `temp/perf-work-queue/direct-hip-zero-active-schedule/` and validates schema
  v4 plus required GPU events for a 256x256x512 bounded-u64 adaptive-bands
  capture.
- Direct-HIP zero row/column product schedules. Implemented for explicit
  trusted per-tile proof masks as
  `direct_hip_tiled_active_prefix_zero_row_col_skip_rns_gemm_v1` and the
  combined
  `direct_hip_tiled_active_prefix_zero_tile_row_col_skip_rns_gemm_v1`. The
  public descriptor carries copied zero A-row and B-column masks, schedule info
  reports proof counts and covered output products, Direct-HIP workspaces and
  scheduled exports upload the masks, and stale/mismatched captures are rejected
  by schema, release-review keying, result comparison, and autotune-key checks.
  Release-build C++ tests cover the plan contract and a real `gfx1100`
  scheduled GEMM/export CPU comparison; smoke captures under
  `temp/perf-zero-rowcol-smoke/` validate schema v4 plus required GPU events for
  bounded i64/u64 65x65x64 adaptive-band combined zero-tile/row-column cases.

Relation to existing queue:

- Deepens "Adaptive Scheduling", "Small Shapes", "AUTO And Cache
  Architecture", and "Exact-Wide".

### 2. Reconstruction Backend And Lazy Export

Treat reconstruction as an algorithmic backend, not merely an export step.
Once GEMM accelerates, CRT/Garner/status/D2H traffic can dominate.

Technical direction:

- Compare direct CRT/Garner, mixed-radix reconstruction, prefix-specialized
  fixed code, partial sign/range reconstruction, and limb-count-specialized
  exact-wide export.
- Compare reconstruction controllers, not only kernels: prefix-specialized
  Garner/MRS, balanced or product-tree batched CRT, residue-current no-export,
  optional rational reconstruction, and diagnostic check-residue modes should
  have separate identities.
- For FHE-derived scenarios, model `ModUp`, `ModDown`, base extension,
  rescale, level drop, Q/P tower movement, and partial sign/range conversion
  as explicit conversion phases rather than generic export.
- Move CRT constants into constant memory, LDS, or compact device tables where
  access patterns justify it.
- Search output layouts for reconstruction, not only for GEMM: residue-major,
  tile-major, cell-major residue vectors, and CRT-friendly staging.
- Add lazy residue output as a first-class plan result when the next operation
  is another RNS operation.
- Batch reconstruction for many small GEMMs when launch overhead dominates.

RNS8-specific notes:

- Direct HIP already performs bounded i64/u64 GPU CRT export with device status
  reporting. This gives a correctness baseline for alternative reconstruction
  kernels.
- Exact-wide reviewed CK winners should be profiled export-first before CK GEMM
  variants are expanded further.
- Lazy export must not blur semantics: bounded native output, exact-wide limbs,
  finite canonical bytes, strict wrap64 low64 output, and residue-current
  output remain distinct plan outcomes.
- Rational reconstruction is useful for computational-algebra consumers, but it
  is an explicit future export surface, not a reinterpretation of bounded,
  exact-wide, finite-u8, or wrap64 outputs.
- Product-tree or balanced CRT should be benchmarked for exact-wide,
  many-output, and many-small workloads, but prefix-9 bounded export should not
  assume it beats prefix-specialized Garner until measured.

Likely first slices:

- Prefix-9 and prefix-20 bounded export specializations.
  Implemented for Direct-HIP fixed-prefix host exports as compile-time
  fixed-prefix CRT reconstruction kernels for prefix 9 and prefix 20, with the
  generic runtime-prefix export kernel retained as fallback for all other
  prefixes. Exact-wide Direct-HIP limb export also dispatches the prefix-20
  fixed-prefix reconstruction kernel. Direct-HIP scheduled per-tile bounded
  i64/u64 export now dispatches compile-time CRT reconstruction variants for
  selected prefixes 1 through 9 inside the adaptive export kernel, retaining the
  runtime-prefix reconstruction helper only as the fallback for unusual
  selected prefixes.
- Exact-wide 1/2/3/4/8/16/32 limb export variants with compact D2H staging.
  Implemented benchmark coverage for the limb-count variants via
  `rns8-bench --exact-wide-limbs` and
  `tools/benchmark_sweep.py --include-exact-wide-limb-variants`; this measures
  the exact-wide export path at requested limb widths. Direct-HIP export stages
  and copies `rows * cols * limb_count` limbs, and full-width device exports
  now elide range-status memset/D2H traffic when overflow is structurally
  impossible: signed limb counts 4..32 and unsigned limb counts 3..32.
  Prefix-20 Direct-HIP signed and unsigned export kernels also dispatch
  compile-time fixed limb-count variants for 1/2/3/4/8/16/32 limbs. The
  3-limb variant is especially important for unsigned exact-wide captures
  because it is the compact full-width 192-bit device reconstruction output and
  can avoid both status traffic and a fourth all-zero output limb. The runtime
  limb-count kernel is retained for other widths. Windows `gfx1100` release
  captures under `temp/perf-work-queue/exact-wide-3limb-export-current/` and
  `temp/perf-work-queue/exact-wide-3limb-export-rerun/` are schema-valid and
  event-valid, and direct-HIP GPU differential tests now compare limb 3 against
  CPU for signed and unsigned exports. Do not promote this as a stable speedup
  yet: the first unsigned 512 pass favored 3 limbs on host export median
  (1676 us versus 7220 us for 4 limbs), but the reverse-order rerun favored
  4 limbs on host export median (3030 us versus 3513 us for 3 limbs).
- A residue-current output mode for chained RNS GEMM benchmarks. Implemented as
  exact-wide benchmark/tooling coverage via `rns8-bench --residue-chain-length`
  and `tools/benchmark_sweep.py --residue-chain-length`: measured repeats keep
  intermediate outputs resident in RNS form, report zero per-repeat
  `crt_export`, and run one final untimed host limb export only for checksum
  evidence. The June 4, 2026 chain-event pass made supported chain captures
  GPU-event-visible for per-repeat pack and chained `rns_gemm` work while
  keeping export phases absent from `gpu_event_phase_order`; the validation
  helper also accepts `tools/gpu_event_report.py --require-events` as the strict
  event gate. This is not yet a public API output-domain mode.
- A batched CRT/reconstruction report that separates kernel time, status
  handling, compact copy time, constants placement, prefix grouping, limb count,
  and tree setup cost.

Relation to existing queue:

- Deepens "Export/Status Overhead", "Exact-Wide", "AUTO And Cache
  Architecture", and "RNS-Native Chains And Next-Op API".

### 3. Residue-Channel Fusion

The current mental model still tends toward one residue plane as one GEMM-like
unit. Breaking that abstraction can remove repeated source loads, pack work,
launches, LDS traffic, and stores.

Technical direction:

- Pack multiple residues from one native load.
- Interleave two or more residues in adjacent INT8 lanes or fragment groups
  where backend mapping allows it.
- Let one CTA own the same C tile for multiple moduli when VGPR/LDS pressure is
  tolerable.
- Pair moduli with compatible reducer families so one epilogue can emit
  multiple residue planes.
- Share A transforms across multiple B residue planes for repeated-B workloads.

RNS8-specific notes:

- This is not simply grouped GEMM. It changes the representation and epilogue
  surface so a single scheduled unit can carry multiple channels.
- Moduli 256, 255, and 251 are the obvious first family because reducers are
  specialized and finite-u8 uses the same constants heavily.
- Exact-wide and bounded prefix-9 both benefit if the first few high-value
  channels are fused.

Likely first slices:

- Multi-modulus pack kernels that load native A/B once and emit two or three
  centered residue planes. Implemented first for direct-HIP fixed-prefix RNS
  packs: bounded prefix 9 and exact-wide prefix 20 now use native-load-once
  pack kernels for signed and unsigned inputs, with the generic per-plane pack
  kernel retained for all other prefixes. This improves the existing pack phase
  without changing persistent RNS matrix semantics.
- Benchmark-only residue-channel fusion experiments are now exposed through
  `rns8-bench --residue-channel-fusion` for explicit Direct-HIP, global-bound,
  fixed-requested prefix-9 bounded captures. Schema validation requires
  `fusion_mode=residue_channel_width3_experimental_benchmark_only`,
  `pack_layout=native_i8_row_major_residue_channel_width3`,
  `residue_group_width=3`, and the generated reducer identity. AUTO and public
  one-shot calls never route to this experiment.
- A direct-HIP small-shape fused multi-plane baseline before CK/rocWMMA
  variants.
- Autotune key extension for residue group identity and layout.

Relation to existing queue:

- Deepens "Direct-HIP RNS Fallback", "CK Path", "rocWMMA Path", "Finite-u8",
  and "Reusable B Prepack".

### 4. Fused Pack+GEMM

For one-shot, small, and transient-A workloads, global packed intermediates can
cost more than they save.

Technical direction:

- Load native bounded inputs, canonical finite bytes, or resident RNS source
  tiles directly into wave/CTA-local transform code, then feed matrix fragments
  without writing a global packed buffer.
- Materialize only the reused operand for repeated-B or repeated-A workloads.
- Keep transient A in registers or LDS while B uses a reusable backend-native
  layout.
- Add finite-u8 paths that center canonical bytes in registers for moduli
  251/255/256. Direct-HIP finite-u8 pack, resident GEMM reduction, and
  export now dispatch fixed-modulus kernels for 251, 255, and 256.
  Direct-HIP public finite-u8 one-shot ring/field calls now also use a native
  transient-input GEMM path that copies canonical `uint8_t` A/B buffers,
  centers them inside the GEMM tile load, skips resident A/B pack kernels, and
  materializes C through the existing finite export path. This is not yet a
  reviewed speedup claim.
- CK and rocWMMA finite-u8 reducer specialization was tested for the same
  fixed-modulus family. The 251/255 pseudo-Mersenne reducer variants were not
  promoted because focused Windows `gfx1100` measurements made them worse or
  too noisy relative to the existing reviewed v1 accelerator identities.
  Modulus 256 now has explicit CK/rocWMMA v2 selected-kernel identities with a
  shared mask reducer, but the June 4, 2026 release/event sweep at 512x512x512
  still favored direct-HIP end-to-end, so no reviewed cache entry was written.
  Current CK/rocWMMA plans now report fixed-modulus 251/255/256 v2 identities
  and route those cases through shared common-modulus reducer helpers; the old
  generic accelerator identities are rejected for those explicit moduli by
  schema/cache tooling. Fresh release captures are still required before any
  historical 251/255 winner timing can be transferred to the new v2 identities.

RNS8-specific notes:

- hipBLASLt and rocWMMA repeated-B evidence says B reuse matters. Fused
  pack+GEMM should focus first on transient A with reusable B.
- Small 64/128 shapes often lose to setup. Removing pack buffers and launches
  is more useful than expanding heavy matrix-engine kernels there.

Likely first slices:

- Direct-HIP one-shot bounded prefix-9 fused load/compute path.
  Implemented for direct-HIP global bounded i64/u64 one-shot calls whose plan
  resolves to prefix 9: A and B are copied as native int64/uint64 device
  buffers, the grouped prefix-9 direct-HIP kernel centers native inputs inside
  the CTA tile load path, and C is materialized directly as resident RNS
  residues for the existing CRT export path. Per-tile/adaptive plans,
  wider-prefix stress cases, persistent matrix APIs, and non-direct-HIP
  backends keep the established resident pack/GEMM/export route. A follow-up
  large-shape bounded-u64 specialization now routes public one-shot
  `m/n/k >= 512` Direct-HIP calls to
  `direct_hip_prefix9_native_input_colpair_grouped_rns_gemm_v2`, where each
  worker computes two neighboring output columns and reuses the centered A tile
  value across both accumulators. Bounded i64 and smaller bounded-u64 one-shot
  shapes remain on `direct_hip_prefix9_native_input_grouped_rns_gemm_v1`
  because release evidence showed i64 regressions and noisy small-shape u64
  averages. Windows `gfx1100` release captures under
  `temp/oneshot-colpair-before/` and
  `temp/oneshot-colpair-release-gated/` show the routed bounded-u64 512 case
  improving average end-to-end time by 1.09x and median end-to-end time by
  1.21x against the prior v1 one-shot kernel, with schema-valid and
  event-valid final captures.
- rocWMMA transient-A fused pack against reusable B for non-tiled RNS.
- Benchmark split between one-shot and persistent reuse so wins are not hidden.
  Implemented as `rns8-bench --oneshot` for bounded i64/u64 CPU and
  direct-HIP global-bound captures plus `tools/benchmark_sweep.py
  --include-oneshot` / `--oneshot-only`. One-shot captures use the public
  one-shot API per repeat, report zero external `pack` and `crt_export`
  phases because those costs are folded into the measured API call, emit a
  distinct `benchmark_execution_mode`, and keep direct-HIP one-shot captures
  out of autotune promotion. This is a measurement surface only until release
  captures compare one-shot end-to-end time against persistent direct-HIP,
  vector-ALU, CPU, and accelerator baselines.
- Direct-HIP one-shot finite-u8 fused load/compute path.
  Implemented for public finite ring/field `uint8_t` one-shot calls. The
  benchmark/schema surface reports `rns8_finite_u8_public_oneshot`, exact
  native-input event phases, zero external pack/export timings, and a persistent
  direct-HIP finite baseline prerequisite before any speedup claim.
- Direct-HIP transient uniform-small A+B bounded prefix-9 path. Implemented for
  explicit fixed-prefix global-bound `bounded-i64` and `bounded-u64`
  `rns8-bench --backend hip-direct --prefix-policy fixed-requested
  --transient-uniform-small-inputs` captures.
  The benchmark copies A and B as row-major `int8_t` device buffers every
  repeat, dispatches
  `direct_hip_uniform_small_i8_ab_colpair_prefix9_transient_grouped_rns_gemm_v1`,
  writes resident prefix-9 RNS residues, and uses the normal CRT export path.
  This is a benchmark/runtime path only; it does not change public matrix
  semantics or default AUTO routing. The benchmark/schema surface reports
  `rns8_bench_uniform_small_i8_ab_transient_path`,
  `transient_uniform_small_i8_ab_inputs`,
  `uniform_small_i8_ab_transient_residue_then_crt_export`, and explicit
  `bounded_uniform_small_i8_a_h2d`,
  `bounded_uniform_small_i8_b_h2d`, and
  `bounded_uniform_small_i8_ab_transient_gemm_kernel_group` GPU event phases.
  Windows `gfx1100` release captures under
  `temp/perf-work-queue/uniform-small-transient-i8/` used release binaries,
  3 warmups, 9 measured repeats, fixed requested prefix 9, and seed 123. All
  transient, persistent, and one-shot captures are schema-valid and event-valid,
  and checksums match within each same-shape trio.

  | Semantics | N | Persistent median end-to-end us | Transient median end-to-end us | One-shot median end-to-end us | Transient vs persistent | Transient vs one-shot |
  |---|---:|---:|---:|---:|---:|---:|
  | bounded-i64 | 64 | 930 | 1026 | 1108 | 0.91x | 1.08x |
  | bounded-i64 | 128 | 1533 | 1060 | 3336 | 1.45x | 3.15x |
  | bounded-i64 | 512 | 3701 | 1844 | 5130 | 2.01x | 2.78x |
  | bounded-u64 | 64 | 1379 | 1270 | 901 | 1.09x | 0.71x |
  | bounded-u64 | 128 | 1733 | 1169 | 1153 | 1.48x | 0.99x |
  | bounded-u64 | 512 | 2617 | 2056 | 4120 | 1.27x | 2.00x |

  Decision: keep as an explicit measured small/fixed-prefix candidate and use it
  as the direct-HIP fused pack+GEMM comparison surface. It beats same-contract
  persistent direct-HIP in five of six measured rows, but it is not a blanket
  default-routing promotion because bounded-i64 64 regressed against persistent
  and bounded-u64 64/128 remains better or roughly tied with public one-shot.
- Direct-HIP transient-A plus reusable-B finite-u8 path.
  Implemented for `rns8-bench --backend hip-direct --semantics finite-u8-*`
  with `--reuse-packed-b`: B is packed once into persistent finite storage,
  A is copied as canonical native `uint8_t` per repeat, and the Direct-HIP GEMM
  centers A inside the tile load path while consuming resident centered B. The
  benchmark/schema surface reports `transient_native_a_resident_b_reuse`,
  `rns8_bench_native_a_reuse_b_path`, `finite_native_a_gemm_kernel`, and a
  zero-valued `finite_pack_kernel`. Windows `gfx1100` release captures under
  `temp/finite-native-a-reuse-b-release-r33/` show setup-amortized wins at 33
  repeats for ring-255 512/1024 and field-251 512/1024, while 128-size cases
  stay experimental because setup and Windows timing variance can erase the
  per-repeat pack savings.
- Direct-HIP transient-A plus reusable-B bounded prefix-9 path.
  Implemented for global-bound `bounded-i64` and `bounded-u64`
  `rns8-bench --backend hip-direct --reuse-packed-b` captures whose Direct-HIP
  plan resolves to fixed prefix 9. Adaptive-band captures generally use the
  native-A/resident-RNS-B route: B is packed once into resident RNS storage,
  A is copied as native `int64_t`/`uint64_t` per repeat, and the grouped
  prefix-9 GEMM centers A inside the tile load while consuming resident centered
  B. The benchmark/schema surface reports `transient_native_a_resident_b_reuse`,
  `rns8_bench_native_a_reuse_b_path`, a distinct
  `bounded_native_a_reuse_b_gemm_kernel_group`, and zero `pack_kernel`.
  Large bounded-u64 adaptive-band reuse-B captures with `m/n/k >= 512` now route
  to `direct_hip_native_a_u64_colpair_prefix9_reuse_b_grouped_rns_gemm_v2` and
  emit `bounded_native_a_colpair_reuse_b_gemm_kernel_group`, reusing each
  centered native A tile value across two neighboring output columns while B
  stays resident in RNS storage. A Windows `gfx1100` release smoke under
  `temp/perf-work-queue/direct-hip-u64-reuse-b-colpair/` is schema-valid and
  event-valid; at 512 with 33 repeats it measured 3218.94 us for same-build
  non-reuse direct HIP versus 2842.46 us setup-inclusive per repeat for the
  colpair reuse-B route, a 1.13x setup-amortized win. The 5-repeat smoke still
  loses setup-inclusively, so this remains a many-repeat explicit reuse path,
  not an AUTO/default-routing claim.
  The default uniform-small benchmark profile now takes the faster specialized
  route: A and B are represented as single row-major `int8_t` planes because
  all prefix-9 centered residues are identical for the generated `[-16,16]`
  signed and `[0,16]` unsigned values. B is copied once during reuse setup,
  A is copied per repeat, and v2
  `direct_hip_uniform_small_i8_ab_colpair_prefix9_reuse_b_grouped_rns_gemm_v2`
  lets each thread compute two neighboring output columns so it can reuse the
  same A tile value across both accumulators while fanning the same A/B planes
  across the prefix-9 RNS output planes in one grouped launch. That surface
  reports
  `transient_uniform_small_i8_a_resident_i8_b_reuse`,
  `rns8_bench_uniform_small_i8_ab_reuse_b_path`, the
  `uniform_small_i8_ab_resident_b_residue_then_crt_export` epilogue, the
  `bounded_uniform_small_i8_ab_colpair_reuse_b_gemm_kernel_group` event phase,
  and zero `pack_kernel`.
  Windows `gfx1100` release captures under
  `temp/uniform-small-i8-ab-colpair-release/` and the bounded-u64 1024 reruns
  under `temp/uniform-small-i8-ab-colpair-u64-1024-rerun/` used release
  binaries, 3 warmups, and 9 measured repeats. They are schema-valid and
  event-valid. Against the prior v1 single-column implementation in
  `temp/uniform-small-i8-ab-colpair-before/`, the v2 first-pass release matrix
  improved per-repeat end-to-end time by 2.10x for bounded i64 512, 2.54x for
  bounded i64 1024, and 1.43x for bounded u64 512; bounded u64 1024 stayed
  export-noise sensitive but v2 reduced the GEMM phase in that first pass by
  1.29x and beat the same-backend non-reuse baseline setup-inclusively in three
  focused reruns. Keep this as an explicit reuse-path implementation win, not an
  AUTO/default-routing claim, until workload-level reuse policy decides when
  setup and reuse metadata should drive backend selection.
- Direct-HIP reusable-A bounded prefix-9 uniform-small path. Implemented for
  global-bound `bounded-i64` and `bounded-u64`
  `rns8-bench --backend hip-direct --reuse-packed-a --prefix-policy
  fixed-requested --max-prefix 9` captures whose Direct-HIP plan resolves to
  fixed prefix 9. The path copies uniform-small A once during prepack setup,
  copies uniform-small B per repeat, and reuses the same colpair grouped kernel
  launch under the explicit
  `direct_hip_uniform_small_i8_ab_colpair_prefix9_reuse_a_grouped_rns_gemm_v1`
  selected-kernel id. The benchmark/schema surface reports
  `transient_uniform_small_i8_b_resident_i8_a_reuse`,
  `rns8_bench_uniform_small_i8_ab_reuse_a_path`, the
  `uniform_small_i8_ab_resident_a_residue_then_crt_export` epilogue, the
  `bounded_uniform_small_i8_ab_colpair_reuse_a_gemm_kernel_group` event phase,
  and zero `pack_kernel`. Windows `gfx1100` release captures under
  `temp/perf-work-queue/` used release binaries, 3 warmups, and 33 measured
  repeats. Against a clean `a75b0a2` same-contract repeated-A baseline,
  setup-inclusive speedups were 3.04x and 1.32x for bounded i64 512/1024, and
  1.33x and 1.30x for bounded u64 512/1024. Keep this as an explicit
  fixed-prefix reuse-path implementation win, not an AUTO/default-routing claim.
  Large bounded-u64 adaptive-band repeated-A captures with `m/n/k >= 512` now
  have a matching native-B/resident-RNS-A colpair route:
  `direct_hip_native_b_u64_colpair_prefix9_reuse_a_grouped_rns_gemm_v1`.
  A is packed once into resident RNS storage, B is copied as native `uint64_t`
  per repeat, and the grouped prefix-9 GEMM centers B inside the tile load while
  consuming resident centered A. The benchmark/schema surface reports
  `transient_native_b_resident_a_reuse`, `rns8_bench_native_b_reuse_a_path`, the
  `resident_a_native_b_centered_residue_then_crt_export` epilogue, the
  `bounded_native_b_colpair_reuse_a_gemm_kernel_group` event phase, and zero
  `pack_kernel`. Windows `gfx1100` release captures under
  `temp/perf-work-queue/direct-hip-u64-reuse-a-colpair/` used release binaries,
  3 warmups, and 33 measured repeats. They are schema-valid and event-valid. At
  512 the same-build non-reuse Direct-HIP baseline averaged 6475.15 us per
  repeat while the native-B reuse-A route averaged 4633.30 us setup-inclusive
  per repeat, a 1.40x setup-amortized win. At 1024 the corresponding numbers
  were 10613.20 us versus 9892.63 us, a 1.07x setup-amortized win. Event traces
  show the native-B colpair GEMM phase itself is slower than the normal grouped
  Direct-HIP GEMM, so keep this as an explicit repeated-A path whose current win
  comes from reduced pack/export cost, not as an AUTO/default-routing claim.

Relation to existing queue:

- Deepens "Small Shapes", "Reusable B Prepack", "hipBLASLt Path", and
  "Host/Transfer".

### 5. End-To-End Layout Search

Layout search must optimize the whole pipeline, not just GEMM.

Technical direction:

- Search residue-major, cell-major, tile-major, prefix-major, and interleaved
  residue layouts.
- For FHE/lattice proxies, include polynomial-tower layouts: coefficient-major,
  NTT-domain, modulus-major, digit-major, key-switch-key-major, Q/P-basis,
  chain-level, and automorphism-friendly permutation layouts.
- Search B cache layouts: backend-native, residue-swizzled, K-blocked,
  modulus-interleaved, and repeated-B column-major panels.
- Search output layouts for CRT/export: residue planes, cell-major residue
  vectors, compact D2H staging, and limb-major exact-wide export.
- Include compressed residue blocks where `int8_t` planes waste bandwidth due
  to padding or metadata rather than arithmetic.

RNS8-specific notes:

- `rns_i8_modulus_major_v2`, `rns_i8_tile_swizzled_b_v1`,
  `finite_u8_centered_plane_v2`, `wrap64_byte_limb_gemm36_v2`, and
  `rns_i4_packed_v0` should be treated as one layout family map, not isolated
  experiments.
- A layout that slows raw GEMM slightly can still win if it removes export or
  pack traffic.

Likely first slices:

- A layout matrix document section in the benchmark reports.
- Separate pack A, pack B, raw GEMM, residue store, CRT/export, D2H, and
  end-to-end rows for each layout.
- First compare current modulus-major, rocWMMA B cache, and a CRT-friendly
  output staging layout on 512/1024 bounded i64 and exact-wide.

Relation to existing queue:

- Deepens "Reusable B Prepack", "Finite-u8", "Exact-Wide", "Wrap64 Direct-HIP
  v3", and "INT4/IU4".

### 6. Persistent Grouped Scheduler

HIP Graphs reduce launch overhead, but RNS8 also needs a device-resident
grouped scheduler for irregular residue/tile/prefix work.

Technical direction:

- Build grouped plans for same shape/many moduli, same B/many A, many small
  independent GEMMs, tile groups with the same prefix, and mixed exact workloads.
- Include FHE-shaped task tables for many NTTs, many coefficient primes,
  base-conversion fragments, key-switch/relinearization fragments, rotation
  batches, bootstrapping stages, and repeated key-material reads.
- Use persistent blocks or grouped backend dispatch to pull pack/GEMM/reduce/
  export work from a device-side task list.
- Explore Stream-K and split-K ideas for small M/N with large K, prefix-heavy
  schedules, and wave quantization tails.
- Combine repeated-B caches with split-K so B stays hot while work is balanced.

RNS8-specific notes:

- The current adaptive scheduler groups selected prefixes. The next step is to
  make that grouping visible to matrix-engine backends and grouped batch APIs.
- Many exact tasks with different semantic outputs should be grouped only when
  the plan metadata stays explicit.

Likely first slices:

- `rns8_grouped_plan` benchmark-only API shape for several independent GEMMs.
- Device task table for prefix/tile groups.
- Persistent or grouped direct-HIP implementation before accelerator-specific
  versions.

Relation to existing queue:

- Deepens "HIP Graphs And Launch Batching", "Adaptive Scheduling",
  "Large-Shape Release Matrix", and "Small Shapes".

### 7. Shared Epilogue DSL

Reducers, status, CRT fragments, finite canonical export, and stores should not
be reinvented separately in direct HIP, CK, rocWMMA, and hipBLASLt-adjacent
paths.

Technical direction:

- Define internal epilogue nodes such as `CenteredMod256`, `CenteredMod255`,
  `CenteredMod251`, `BarrettModP`, `ResidueStore`, `RangeFlag`,
  `NativeStore`, `CRTPrefix9`, `ExactWideLimbStore`, and `CanonicalU8Store`.
- Add research vocabulary for FHE/lattice modular pipelines: `Butterfly`,
  `PointwiseMul`, `LazyReduce`, `Montgomery`, `BaseExtend`, `ModDrop`,
  `Rescale`, `KeySwitchDigit`, `ExternalProduct`, and `Automorphism`.
- Compose nodes into backend-specific epilogues without duplicating arithmetic
  policy.
- Keep backend limitations explicit: hipBLASLt may need external post kernels,
  while CK/rocWMMA/direct HIP can fuse more.

RNS8-specific notes:

- This is an internal architecture cleanup with performance implications. It
  should reduce drift between finite direct-HIP reducers and CK/rocWMMA
  reducers.
- The DSL should make status behavior, reduction family, and output layout part
  of the selected epilogue identity.

Likely first slices:

- Shared reducer metadata and device helper functions for 256/255/251.
- Refactor direct-HIP and rocWMMA finite reducers to use the same reducer node
  definitions.
- Add epilogue family strings that describe composed behavior.

Relation to existing queue:

- Deepens "CK Path", "rocWMMA Path", "Finite-u8", "Export/Status Overhead",
  and "Wrap64 Matrix Engine Redesign".

### 8. Generated Kernel Search

Hand-picked variants will bottleneck. RNS8 needs a generated kernel zoo with
boring variants included.

Technical direction:

- Generate combinations of M/N/K tile, waves per CTA, LDS staging depth,
  K-block cap, A/B vector width, pack layout, residue count per CTA, epilogue
  reducer, store vectorization, split-K mode, and persistent mode.
- Compile with constants baked in for modulus, prefix length, signedness,
  finite modulus, status mode, layout, and tile sizes.
- Prune build-time variant sets with modes such as minimal, `gfx1100_full`, and
  research.
- Use microprobes to rank candidates during exploration, then run normal
  release review for candidates that survive.

RNS8-specific notes:

- CK is already a template-heavy model. RNS8 should generate both CK aliases
  and direct HIP/rocWMMA variants so one backend does not define the search
  space.
- Generated variant names must flow into `selected_kernel`, autotune keys, and
  docs.

Likely first slices:

- Python generator for CK/rocWMMA parameter aliases under ignored generated
  output.
- Static selected-kernel registry for generated variants.
- Build-time kernel-set option to avoid making normal builds huge.

Relation to existing queue:

- Deepens "CK Path", "rocWMMA Path", "AMDGPU Builtins", "INT4/IU4", and
  "AUTO And Cache Architecture".

### 9. Roofline And Evidence Database

RNS8 needs to identify the current bottleneck before optimizing kernels.

Technical direction:

- Build a RNS8-specific roofline model for GEMM arithmetic intensity, pack
  bandwidth, B prepack amortization, residue store bandwidth, CRT/export
  bandwidth, host/device transfer, launch overhead, and API scheduling.
- Add FHE/lattice proxy bottleneck classes: twiddle/root traffic,
  NTT/INTT pass count, base-conversion traffic, key-material reads,
  automorphism permutation traffic, bootstrapping stage traffic, and
  evaluation-key residency.
- Store benchmark knowledge as a durable corpus: target, HIP/ROCm version,
  backend, semantic, shape, selected kernel, layout, timing summaries, ISA hash,
  VGPR/SGPR/LDS, workspace bytes, correctness hash, and thermal/power metadata
  when available.
- Classify shapes as compute-bound, pack-bound, launch-bound, export-bound,
  transfer-bound, or status-bound.

RNS8-specific notes:

- Current benchmark schema and review reports already capture much of the
  identity material. The first analysis layer now tells the next implementer
  which semantic/scenario/backend groups consume the most measured bottleneck
  time; remaining work is to feed it broader current capture corpora and let it
  steer release A/B work.
- Exact-wide and finite-u8 should be modeled separately from bounded i64/u64;
  their export and modulus costs differ.
- Computational-algebra scenarios need phase labels in the evidence database:
  dense GEMM, modular factorization, triangular solve, CRT/CRA combine,
  rational reconstruction, verification, certificate generation, polynomial
  transform, product/remainder tree, and black-box sparse matvec.
- Exact-LA and symbolic scenarios should also record controller and
  preprocessing time: rank-profile selection, symbolic preprocessing, sparse
  reduction, tree setup, p-adic lifting, certificate generation, and
  reconstruction/export can dominate even when the dense modular GEMM phase is
  fast.
- External CAS or exact-algebra libraries should be recorded by role: CPU
  oracle, CAS semantic oracle, finite-field reference, algorithm reference,
  benchmark comparison, workload source, CUDA translation study, or non-goal.
  Record licensing and dependency status separately so optional or proprietary
  oracles never become implied production dependencies.
- CAS-oriented evidence rows should include domain/coercion metadata:
  `domain_family`, `parent_domain_id`, `coercion_policy`,
  `coefficient_ring`, `exactness_mode`, `finite_modulus`,
  `prime_or_composite`, `extension_degree`, `phase_label`, `oracle_role`,
  `dense_kernel_extracted`, and `reconstruction_mode`.

Likely first slices:

- Extend review Markdown with bottleneck classification derived from existing
  timing phases. Implemented in `tools/benchmark_sweep.py`: review JSON now
  stores a per-candidate bottleneck classification from pack, GEMM, export, and
  unattributed end-to-end overhead medians, and `review_report.md` prints that
  classification beside promotion blockers and primary loss phase.
- Add an ignored evidence database builder that ingests schema v4 captures and
  review reports. Implemented as `tools/evidence_database.py`: it validates
  capture JSON with the existing schema, optionally joins
  `tools/benchmark_sweep.py` review reports and scenario manifests, estimates
  coarse per-capture work/traffic/arithmetic-intensity metrics, classifies the
  dominant bottleneck from host phases and GPU event categories, and writes
  ignored `evidence_database.json`, `evidence_rows.csv`, and
  `evidence_summary.md` outputs under `temp/` by default. The first database
  schema records target/device/toolchain identity, backend/kernel, semantic,
  shape, finite modulus, prefix/selected-prefix, exact-wide limb count,
  pack/reuse mode, output domain, timing medians, event bottleneck category,
  estimated GOP/s, pack/export bandwidth estimates, review promotion blockers,
  and scenario-family metadata when available.
- Add optional RGA/ISA resource summaries to that database. Implemented as
  `tools/evidence_database.py --isa-report <file-or-dir>`: the database ingests
  `tools/gpu_isa_report.py` `*-isa-summary.json` outputs, matches them to
  captures by normalized backend and GPU target, records report paths, symbol
  counts, WMMA/MFMA/global-store/LDS/wait/instruction totals, VGPR, SGPR,
  occupancy when available, RGA status, and emits compact ISA resource tables in
  `evidence_summary.md`. The reports remain temp-only evidence inputs.
- Add a corpus-level roofline priority surface. Implemented in
  `tools/evidence_database.py`: each row now carries a conservative
  `roofline_target` and `optimization_hint`, and `evidence_summary.md` includes
  both `GPU Roofline Priority` and global `Roofline Priority` tables grouped by
  target/scenario/semantic/target-id and ranked by total measured bottleneck
  time. The loader also has an opt-in `--skip-invalid` mode for broad ignored
  temp corpora, recording stale rejected captures in the output instead of
  blocking all valid captures. This makes the database an execution control
  surface for deciding the next A/B run without turning analysis output into an
  autotune or promotion claim.

Relation to existing queue:

- Deepens "Instrumentation", "Host/Transfer", "Large-Shape Release Matrix",
  and "AUTO And Cache Architecture".

### 10. Finite Data Specialization

Finite-u8 should specialize not only by modulus, but also by data distribution.

Technical direction:

- Add histogram-guided plans for binary, sparse-small, uniform field, ring 256,
  ring 255, and prime-field workloads.
- Evaluate bitset/popcount style paths for binary or nearly binary finite
  matrices.
- Push pseudo-Mersenne, fold, low-byte, and reciprocal reducers through CK and
  rocWMMA epilogues.
- Keep finite layouts separate from bounded CRT layouts; finite paths use one
  explicit modulus and prefix-zero storage.

RNS8-specific notes:

- Current finite release evidence shows different winners by modulus and shape.
  AUTO should eventually include finite data profile classes, not only modulus
  and shape.
- Ring 251, ring 255, ring 256, field 251, and representative generic
  composite/prime moduli need separate workload buckets.

Likely first slices:

- Add finite histogram capture for benchmark-generated inputs.
- Add CK/rocWMMA reducers for 251/255/256.
- Add finite workload suites for binary, sparse, and full-uniform inputs.

Relation to existing queue:

- Deepens "Finite-u8", "CK Path", "rocWMMA Path", "End-To-End Layout Search",
  and "AUTO And Cache Architecture".

### 11. RNS-Native Chains And Next-Op API

Exporting after every GEMM wastes work when the next operation can consume RNS.

Technical direction:

- Add plan hints or explicit output modes for residue-current output,
  native-current output, final export, repeated-B reuse, many same-shape calls,
  validation-only calls, and RNS-chain continuation.
- For FHE/lattice-derived research scenarios, model `ntt-current`,
  `coefficient-current`, `tower-current`, key-material-current, and
  modulus-chain-current states as planning vocabulary before exposing any
  public API state.
- Keep `rns8_matrix` capable of representing currentness honestly:
  residue-current, native-current, finite-current, wrap-byte-current, and stale
  host/native states must not be inferred from type alone.
- Allow `RNS GEMM -> RNS GEMM -> export` without reconstructing the
  intermediate result.
- Add fast device-to-device conversion kernels where AUTO needs to move between
  vector-native and RNS domains.

RNS8-specific notes:

- Vector-produced bounded C currently cannot feed RNS GEMM without conversion
  or rejection. That should become an explicit planning choice instead of an
  accidental dead end.
- Lazy residue output pairs naturally with reconstruction backend work and
  expression-level fusion.

Likely first slices:

- Benchmark-only `output_mode=residue` path for bounded RNS chains.
  Implemented for bounded i64/u64 `rns8-bench --residue-chain-length` with
  explicit RNS backends, global chain-safe bounds, square RNS chain shapes, and
  zero-valued measured `crt_export`; the benchmark exports once after measured
  repeats only for checksum generation. AUTO/vector-ALU chains stay rejected at
  the benchmark CLI until mixed-backend chain lowering selects explicit
  conversion points; per-tile bounded chains stay rejected until adaptive chain
  bounds exist.
- Device native-to-RNS bounded conversion kernels.
  Implemented as internal HIP device-to-device `native_i64_to_rns_kernel` and
  `native_u64_to_rns_kernel` wrappers that reuse the centered direct-HIP pack
  kernels with device-native sources. AUTO/direct-HIP bounded matrices can now
  lazily materialize stale RNS residues from current native device storage
  before an RNS-backend GEMM consumes them. This is an internal transition
  primitive, not a public output-domain API or vector autotune promotion policy.
- Plan metadata exposing next-op hints and chosen output domain.
  Implemented in `rns8_get_plan_packing_info` with explicit input/output
  domains, host/device output currentness, and next-operation flags for final
  export, RNS GEMM continuation, native GEMM continuation, native-to-RNS
  conversion eligibility, and reusable B prepack availability.

Relation to existing queue:

- Deepens "Native Vector-ALU Production Backend", "Reconstruction Backend",
  "AUTO And Cache Architecture", and "Expression-Level Fusion".

### 12. Plan-Level Algebraic Lowering

AUTO cannot stay a bag of local backend heuristics. RNS8 needs an internal
operation representation before lowering to a backend.

Technical direction:

- Represent operations such as `MatMul`, `Export`, `ResidueAdd`, `FiniteAdd`,
  `NativeToRns`, `RnsToNative`, `RankUpdate`, and `Batch` with semantic,
  bounds, reuse, layout, and desired-output metadata.
- Add research-only operation labels for FHE/lattice scenario modeling:
  `Ntt`, `Intt`, `Hadamard`, `BaseExtend`, `ModUp`, `ModDown`, `Rescale`,
  `GadgetDecompose`, `ExternalProduct`, `KeySwitch`, `Automorphism`,
  `Relinearize`, and `Bootstrap`. These labels should prevent dense-GEMM
  overclaiming; they are not public FHE APIs.
- Include computational-algebra vocabulary in the internal IR even before those
  operations become public APIs: `Rank`, `Determinant`, `Solve`, `Nullspace`,
  `RankProfile`, `PLUQ`, `CUP`, `PLE`, `TriangularSolve`, `Echelon`,
  `CharPoly`, `MinPoly`, `Certificate`, `CRABuild`, `DixonSolve`,
  `PadicLift`, `RationalReconstruct`, `FreivaldsCheck`, `Ntt`, `Intt`,
  `PointwiseMul`, `ProductTree`, `RemainderTree`, `Interpolate`,
  `ModularCompose`, `GroebnerF4Sparse`, `GroebnerF4DenseFiniteField`,
  `F5SignatureReduction`, `FGLM`, `SubresultantPRS`, `ResultantSylvesterDet`,
  `SylvesterMat`, `Popov`, `Hermite`, `Smith`, and `PolyMatMul`.
- Lower `Export(MatMul(A,B))` differently from `MatMul(A,B)` whose result feeds
  another RNS GEMM.
- Lower repeated-B workloads toward prepack and persistent scheduling.
- Lower finite and bounded small-value workloads toward specialized finite,
  direct INT32, vector, or RNS paths based on the explicit semantic contract.

RNS8-specific notes:

- Semantics remain explicit. The IR should not infer signedness, exact-wide,
  finite, or wraparound behavior from C++ types.
- The first version can be internal to plan creation and benchmark tooling
  without changing the public ABI.
- Polynomial and structured-matrix terms are classification and scenario
  vocabulary until a real backend exists. The IR should prevent overclaiming by
  distinguishing dense GEMM lowerings from NTT, product-tree, black-box sparse,
  or structured-matrix workloads.

Likely first slices:

- Internal plan-lowering description object populated from current descriptors.
  Implemented as the internal `PlanLoweringDescription` helper, derived from
  current plan backend, packing, and schedule metadata without adding another
  public ABI surface.
- Debug/inspect output that explains selected operation lowering.
  Implemented in `rns8-inspect` for autotune exact-hit plans as text and JSON
  `plan_lowering` output covering operation, semantic contract, backend family,
  input/output domain, desired output, schedule, packing, reuse, conversion, and
  lowering path.
- AUTO selection that considers desired output domain and reuse hints.
  Implemented for the current final/native output-domain case by allowing
  reviewed `hip-vector-alu-int64` bounded i64/u64 autotune entries to be
  selected by AUTO. Reuse-hint promotion remains intentionally constrained by
  the existing prepacked-reuse-not-autotune-promotable policy until setup-cost
  and end-to-end wins are promoted as production cache behavior.

Relation to existing queue:

- Deepens "AUTO And Cache Architecture", "Native Vector-ALU Production
  Backend", "RNS-Native Chains", and "CPU/GPU Hybrid AUTO".

### 13. Expression-Level Fusion

RNS8 should eventually fuse exact algebra around GEMM, not only GEMM itself.

Technical direction:

- Support internal expressions such as `C = A*B + D`, `C = A*B - A*E`,
  `C = alpha*A*B + beta*C`, finite residue add/sub, and validation/status
  updates in one output path.
- Fuse residue add/sub and finite reduction into GEMM epilogues where backend
  mapping allows it.
- Avoid reconstructing intermediate exact integers for algebra that is valid in
  the residue domain.

RNS8-specific notes:

- This belongs behind explicit semantics and should not change current GEMM
  public APIs until the internal lowering is stable.
- Finite-u8 and RNS chains are the best early targets because their algebra can
  stay residue-domain.

Likely first slices:

- Internal residue add/sub kernels for RNS-current matrices.
- Fused finite `A*B + D mod q` benchmark path.
- Epilogue DSL node for add/sub before store.

Relation to existing queue:

- Deepens "Shared Epilogue DSL", "Finite Data Specialization", and
  "RNS-Native Chains".

### 14. Shape-Specialized Paths

Not every workload should be forced through square GEMM architecture.

Technical direction:

- Add separate paths for GEMV, batched GEMV, tall-skinny GEMM, wide-skinny
  GEMM, tiny square GEMM, symmetric/Gram products, triangular products,
  banded/diagonal products, and submatrix views.
- Consider direct vector-ALU or wave reductions for true GEMV instead of
  matrix-engine setup.
- Batch many GEMVs with the same B into GEMM when that reformulation wins.
- Use direct INT32 as a middle lane when bounded inputs and output bounds fit.

RNS8-specific notes:

- The release matrix is currently square-heavy. Shape families can reorder
  backend winners and expose wins hidden by square-only review.
- Structured shortcuts are useful only when workload structure is explicit or
  obvious from API shape, not inferred unsafely.

Likely first slices:

- Add shape-family benchmark cases: GEMV, skinny, many small, symmetric/Gram,
  and submatrix view.
- Add direct-HIP/vector GEMV baseline.
  Partially implemented for the native vector-ALU backend as a conservative
  long-K N=1 specialization: `n == 1`, `k >= 4096` bounded
  i64/u64 plans now select `hip_vector_alu_i64_gemv_n1_exact_192b_v1` or
  `hip_vector_alu_u64_gemv_n1_exact_192b_v1`. The kernel parallel-reduces K
  across a 256-thread block while preserving the existing software 192-bit
  exact accumulator and native output/export contract. The active route now
  covers all long-K N=1 vector captures (`n == 1`, `k >= 4096`), not only the
  original 1x1 dot-product gate; multi-row correctness is covered by the HIP
  differential suite, and a 128x1x4096 bounded-i64 runtime smoke under
  `temp/vector-gemv-n1-tall-smoke/` is schema-valid and event-valid with the
  GEMV selected-kernel id. Windows `gfx1100` release captures under
  `temp/perf-work-queue/vector-gemv-n1/` compare against detached pre-change
  commit `96781eb`; at 1x1x65536, bounded-u64 improved average end-to-end by
  2.22x and median end-to-end by 3.39x, while bounded-i64 improved average
  end-to-end by 4.44x and median end-to-end by 7.41x. A broader 1024x1x1024
  smoke remains below the long-K threshold because pack/copy overhead dominated
  and the reduction kernel was not a setup-inclusive win there.
- Add view lowering choice: direct strided load vs copy-to-packed.

Relation to existing queue:

- Deepens "Small Shapes", "Large-Shape Release Matrix", "Direct-HIP RNS
  Fallback", and "FP8/Ozaki, Strassen, Sparsity".

### 15. CPU/GPU Hybrid AUTO

For tiny or export-bound cases, GPU arithmetic is not automatically the best
end-to-end path.

Technical direction:

- Let AUTO consider CPU direct bounded, CPU pack plus GPU GEMM, GPU residues
  plus CPU export, GPU GEMM plus CPU CRT, and fully GPU paths.
- Separate one-shot from persistent reuse: the hybrid winner can differ when
  A/B are already resident.
- Keep CPU/reference timing comparable to GPU captures without fabricating GPU
  target metadata.

RNS8-specific notes:

- Current-v2 bounded-u64 evidence splits small and medium choices across CPU,
  Direct HIP, CK, and hipBLASLt. Vector should be routed only where current
  same-contract evidence shows an end-to-end win.
- Wrap64 CPU is faster than direct HIP at 64 in the release baseline, while
  direct HIP wins larger shapes. AUTO needs that kind of shape split.

Likely first slices:

- Add CPU-direct tiny bounded one-shot cases to review reports.
- Add hybrid export experiments for exact-wide and bounded prefix-9.
- Add selector explanation for CPU/hybrid choices in inspect output.

Relation to existing queue:

- Deepens "Small Shapes", "Native Vector-ALU Production Backend",
  "Reconstruction Backend", and "Host/Transfer".

### 16. Device Plan Cache And Workspace Arena

Repeated workloads should not repeatedly marshal schedules, constants, and
temporary allocations.

Technical direction:

- Add device-resident plan tables containing pointers, strides, prefix
  schedules, reducer constants, layout descriptors, and output modes for
  grouped/persistent kernels.
- Replace per-path scratch management with workspace lifetime classes:
  transient tile, persistent prepack, export staging, status, autotune probe,
  and grouped scheduler state.
- Partition arenas by stream, device, semantic, shape family, and reuse profile
  where needed.

RNS8-specific notes:

- Current matrix/workspace APIs already expose storage and packing info. The
  arena should build on that rather than invent backend-private hidden state.
- hipBLASLt and rocWMMA repeated-B caches are early examples of workspace-local
  reuse that should become a consistent policy.

Likely first slices:

- Add workspace arena accounting to benchmark metadata.
- Move repeated-B cache allocation into named workspace lifetime classes.
- Add device schedule buffer reuse for matching adaptive schedule fingerprints.

Relation to existing queue:

- Deepens "Reusable B Prepack", "HIP Graphs And Launch Batching",
  "Host/Transfer", and "Persistent Grouped Scheduler".

### 17. Verification-Cost Reduction

Full exact CPU differentials are required for promotion, but exploratory search
can use cheaper oracles to iterate faster.

Technical direction:

- Use residue-domain comparison before CRT when testing layout or epilogue
  variants.
- Use checksum or sampled cell oracles during generated-kernel exploration.
- Use metamorphic identities such as `(A+B)C == AC+BC` in the residue domain.
- Compare candidates against the simple direct-HIP correctness path before
  running full CPU exact output over every cell.

RNS8-specific notes:

- This is development-loop acceleration, not relaxed correctness. It should
  keep the existing promotion standards untouched.
- Wrap64 candidate work already uses checksum and sampled CPU-oracle cells as
  exploratory evidence; that pattern can be generalized.

Likely first slices:

- Benchmark/test helper for residue-domain candidate-vs-direct-HIP comparison.
- Checksum metadata for generated-kernel sweeps.
- Metamorphic finite-u8 and RNS-current tests.

Relation to existing queue:

- Deepens "Generated Kernel Search", "Wrap64 Matrix Engine Redesign",
  "Instrumentation", and "Large-Shape Release Matrix".

### 18. Error-Detecting Exact Fast Path

Some exact workloads may usually need fewer residues than worst-case bounds.
RNS8 can exploit that only when fallback is deterministic and semantics stay
exact.

Technical direction:

- Compute likely-needed prefix first, verify with redundant residue or bound
  metadata, then compute additional residue planes only when needed.
- Use redundant channels for sign, range, overflow, or error detection when
  they are cheaper than full reconstruction.
- Schedule high-value or cheap residue planes first so verification can overlap
  later work.

RNS8-specific notes:

- This is distinct from probabilistic early termination. Default exact APIs
  remain deterministic.
- Adaptive prefix minimization and residue-plane priority scheduling are the
  natural entry points.

Likely first slices:

- Deterministic speculative prefix benchmark mode for bounded-i64 distributions.
- Redundant small-modulus status channel experiment.
- Device-side partial verification for already-computed prefix groups.

Relation to existing queue:

- Deepens "Adaptive Prefix Minimization", "Reconstruction Backend", and
  "Persistent Grouped Scheduler".

### 19. Result Cache And Incremental GEMM

Repeated or slowly changing workloads should not always pay full GEMM cost.

Technical direction:

- Add optional fingerprints for repeated operands, prepack metadata, and
  result reuse. Avoid expensive full-content hashes unless the workload
  explicitly amortizes them.
- Add delta paths for changed rows, changed columns, block updates, and
  low-rank updates.
- Add `rank-k` and batched outer-product update paths that choose direct
  HIP/vector for small k and group into GEMM when k grows.

RNS8-specific notes:

- Source version is necessary but not sufficient for memoization. Cheap
  checksums or user-managed stable IDs may be useful for repeated workloads.
- Incremental paths are workload-specific and should stay out of default GEMM
  until benchmark suites prove they matter.

Likely first slices:

- Repeated-operand fingerprint metadata in benchmark-only mode.
- Changed-row direct-HIP bounded update prototype.
- RNS-current rank-k update benchmark.

Relation to existing queue:

- Deepens "Reusable A And A/B Caches", "RNS-Native Chains", and
  "Shape-Specialized Paths".

### 20. Lane/LDS/Store/Prefetch Microarchitecture Audits

The matrix-engine path needs target-specific microarchitecture facts, not only
API-level backend calls.

Technical direction:

- Build fragment-map harnesses for rocWMMA/AMDGPU target ids that record lane
  id, fragment element id, matrix coordinate, and store order.
- Audit LDS bank conflicts for A tiles, B tiles, accumulator staging, epilogue
  staging, and packed residue stores.
- Search store paths: vectorized residue stores, interleaved C stores, compact
  no-padding stores, bit-packed status, and export-staging layouts.
- Explore AMD analogues for async copy/prefetch/ping-pong pipelines: rotating
  LDS buffers, global-to-LDS overlap, MMA-to-epilogue overlap, and K-pipeline
  depth.

RNS8-specific notes:

- rocWMMA residue emission depends on stable fragment mapping. Capture that as
  an artifact before writing fragile lane-owned store kernels.
- RGA and ISA capture should feed the evidence database rather than remain
  one-off temp files.

Likely first slices:

- `fragment_map_gfx1100.json` generated under ignored `temp/`.
- LDS pattern microbenchmarks for A/B loads and C stores.
- Store-path variants for residue and exact-wide export staging.

Relation to existing queue:

- Deepens "rocWMMA Path", "AMDGPU Builtins", "Instrumentation", "Store/Export
  Overhead", and "Generated Kernel Search".

### 21. Toolchain Matrix

Toolchain versions and compile flags are optimization variables.

Technical direction:

- Compare HIP SDK/ROCm versions, clang flags, CK versions, rocWMMA versions,
  Windows versus Linux, and target ids such as `gfx1100`, `gfx1101/gfx1102`,
  `gfx1200/gfx1201`, `gfx942`, and `gfx950` where real hosts exist.
- Keep generated kernel and autotune databases versioned by compiler,
  accelerator library, target id, runtime, and driver.
- Track whether a backend regression is algorithmic, compiler-generated, or
  library-version-specific.

RNS8-specific notes:

- Windows `gfx1100` is the local bring-up path. Linux ROCm and Instinct remain
  separate validation targets and cannot inherit Windows evidence.
- CK and rocWMMA move quickly; a tuned parameter set can become stale when the
  pinned dependency changes.
- Computational-algebra libraries should be tracked by role rather than
  dependency status: FLINT/NTL/Boost as CPU exact oracles, FFLAS-FFPACK/Givaro
  as finite-field references, LinBox/IML as workload and certificate sources,
  Sage/Magma/Maple/Wolfram/Singular/Macaulay2/GAP/Oscar/Nemo/Hecke/PARI/GP as
  CAS semantic oracles or phase classifiers where available, Normaliz and GAP
  packages as workload sources, and M4RI/M4RIE as small-characteristic
  non-goals unless a real extension-field backend appears.
- CUDA artifacts such as Linac, CUMODP, and GPU NTT systems are translation
  studies. Review their lane assumptions, CUDA library dependencies,
  warp-size assumptions, memory layouts, and NVIDIA-specific instructions
  before any HIP experiment is scoped.
- Proprietary or differently licensed CAS artifacts should stay external:
  compare against their outputs or documented phases when useful, but do not
  make them hidden runtime, build, or correctness dependencies.

Likely first slices:

- Toolchain comparison report format using existing benchmark metadata.
- Pin generated-kernel ISA hashes to compiler/library identity.
- Add Linux ROCm placeholders that require real host captures before any claim.
- Add a computational-algebra oracle matrix to the evidence database so future
  reports separate optional CPU/CAS comparison from RNS8 production evidence.

Relation to existing queue:

- Deepens "Large-Shape Release Matrix", "AUTO And Cache Architecture",
  "Linux ROCm Port", and "Instinct Production" roadmap work.

### 22. Scenario Benchmark Corpus

Square synthetic matrices are not enough. RNS8 needs workload classes that
exercise actual performance risks.

Technical direction:

- Add canonical scenarios: one-shot small, one-shot large, repeated-B,
  repeated-A/B, many small independent GEMMs, exact-wide export-heavy,
  finite fixed modulus, RNS-chain no export, adaptive bounded real
  distributions, GEMV/skinny, wrap64 carry-heavy, and large exploratory shapes.
- Add FHE/lattice-derived proxies: NTT/INTT pressure, key-switch digit
  aggregation, rotation/automorphism-heavy linear transforms, CKKS
  rescale/mod-drop chains, BFV/BGV explicit-modulus arithmetic, bootstrapping
  stages, and encrypted-inference linear-layer lowerings labeled by whether
  they are dense GEMM, diagonal/rotation, MVM/convolution, or coefficientwise
  arithmetic.
- Add parameter fixtures inspired by SEAL, OpenFHE, Lattigo, HElib, HEonGPU,
  PhantomFHE, cuHE/cuFHE, and FIDESlib: `N`/`LogN`, slot count,
  coefficient-modulus chain, Q/P towers, plaintext modulus, scale bits,
  decomposition digit count, ciphertext component count, and evaluation-key
  count.
- Add computational-algebra scenarios: dense finite-field BLAS, modular rank,
  determinant, solve, nullspace, rational reconstruction, Freivalds-verified
  product, PLUQ/CUP/PLE rank profile, echelon recovery, block LU/TRSM-like
  update, rectangular rank-k, characteristic/minimal polynomial, p-adic/Dixon
  solve, early-terminated CRA, polynomial matrix multiplication,
  modular-composition BSGS, F4 dense finite-field matrix phases, F4 sparse
  reduction, F5 signature control, FGLM multiplication-matrix conversion, NTT
  pressure, batched NTT, product/remainder tree, subresultant/PRS, Sylvester
  determinant, structured band/triangular/diagonal, Toeplitz/Hankel/Cauchy,
  low-displacement-rank, and black-box Wiedemann/Lanczos matvec.
- Add failure-mode scenarios: max bounds, negative centered residues,
  modulus-edge residues, overflow-near accumulators, stale cache layouts,
  padded dimensions, non-contiguous strides, K-block boundaries, and mismatched
  finite moduli.
- Report winners by scenario family, not only by shape.

RNS8-specific notes:

- The current 64/128/512/1024 release matrices are valuable but not sufficient
  to rank next work across chained RNS, repeated-B, exact-wide export-heavy, and
  finite distribution-sensitive workloads.
- Scenario labels should flow into review reports and the evidence database so
  future queue ordering is driven by real workload classes.
- Scenario metadata should include `algebra_family`, `structure_id`,
  `shape_signature`, `bound_profile`, `prefix_budget`, `density`, `reuse_mode`,
  `fast_mm_level`, `determinism_mode`, field/ring metadata, and reconstruction
  profile where applicable. Exact-LA/symbolic metadata should additionally
  include `phase_id`, `symbolic_precompute`, `controller_mode`,
  `certificate_mode`, `structure_declared`, and whether the reported timing is
  raw dense GEMM, dense-LA phase, reconstruction, or whole symbolic workflow.
  CAS-oriented metadata should further include `source_role`, `cas_system`,
  `parent_domain_id`, `coercion_policy`, `coefficient_ring`,
  `prime_or_composite`, `extension_degree`, `exactness_mode`, `oracle_role`,
  `dense_kernel_extracted`, and `artifact_lineage` for CUDA or external
  comparison artifacts.
- CAS scenario labels should include full-workflow non-goals as well as direct
  dense-kernel extraction: `SageMatrixBenchmark`, `PARIGPLewisWester`,
  `F4DenseFiniteFieldPhase`, `F4SparseReduction`, `F5SignatureControl`,
  `FGLMMultiplicationMatrix`, `CUMODPPolynomial`, `LinacFiniteFieldElim`,
  `GBLAF4DenseBlock`, `MagmaDenseF4`, `NormalizConeMatrix`,
  `GAPPackageMatrix`, and `CUDATranslationStudy`.

Likely first slices:

- Extend `tools/benchmark_sweep.py` scenario definitions without changing raw
  capture schema first. Implemented as `tools/benchmark_sweep.py --scenario`
  with named scenario families for `adaptive-bands`, `repeated-b`,
  `exact-wide-export`, `finite-distributions`, `rns-chain`, `small-oneshot`,
  `finite-generic-moduli`, `many-small`, `skinny-gemv`,
  `computational-algebra-proxies`, `fhe-lattice-proxies`, `wrap64-carry`,
  `large-exploratory`, `layout-search`, and `all`.
  Scenario mode reuses the normal schema v4 captures and release-review logic,
  but writes a separate `scenario_manifest.json` plus `scenario_manifest.md`
  under the sweep output root. The manifest records scenario family, item name,
  shape, backend, pack/reuse mode, output domain, evidence scope, rationale,
  command, capture path, and optional proxy metadata so workload labels do not
  mutate dense-GEMM capture schema or autotune contract keys.
- Expand large-shape scenario coverage beyond bounded 2048. Implemented in
  `large-exploratory`: bounded i64/u64, exact-wide signed/unsigned, finite-u8
  ring/field, and strict wrap64 probes now cover 2048 plus tolerable 4096
  shapes. These entries are explicitly exploratory and are meant to classify
  launch/export-bound versus throughput-bound behavior before backend-specific
  tuning.
- Add generic finite-u8 modulus coverage without overstating accelerator
  support. Implemented in `finite-generic-moduli`: non-hot prime 127 and
  composite 253 ring cases, field-prime 127, and 2048 exploratory ring probes
  are pinned to CPU/Direct-HIP evidence so arbitrary-modulus behavior cannot be
  confused with the specialized 251/255/256 accelerator paths.
- Expand many-small scenario coverage beyond the first tiny proxy. Implemented
  in `many-small`: bounded i64/u64 square jobs, skinny N=1 bounded-u64 jobs,
  exact-wide signed jobs, finite-u8 ring jobs, and public one-shot baselines now
  share pre-grouped baseline metadata. These are not grouped-dispatch speedup
  claims; they are the control surface for proving whether batching 64/128 and
  skinny exact jobs is worth implementing.
- Add scenario tables to review Markdown. Implemented as the separate scenario
  manifest Markdown table beside `review_report.md`; raw review groups stay
  contract-keyed so scenario labels cannot accidentally make incompatible
  captures comparable.
- Include repeated-B hipBLASLt/rocWMMA and RNS-chain lazy-export cases.
  Implemented in the scenario corpus: `repeated-b` emits bounded-i64 512/1024
  `--reuse-packed-b` captures across the current GPU backend set, while
  `rns-chain` emits signed/unsigned bounded and exact-wide residue-current
  chains at 128/256 with three- and four-GEMM depths. Final checksum export is
  kept outside the measured repeat loop so lazy-export timing is not mixed with
  host-output timing.
- Add explicit end-to-end layout-search coverage. Implemented in
  `layout-search`: bounded fixed-prefix RNS final-export and RNS-next-op
  captures, exact-wide prefix-20 limb-export and RNS-next-op captures,
  finite-u8 hot ring/field layout captures, and strict wrap64 byte-limb layout
  captures share `workflow_name=end_to_end_layout_search` metadata so layout
  comparisons do not get inferred from unrelated backend timings.
- Include FHE/lattice proxy metadata: ring dimension or polynomial degree,
  coefficient-modulus count, decomposition digit count, transform/current
  domain, key-material reuse profile, evidence scope, and output-domain
  requirement. Implemented in `tools/benchmark_sweep.py` as
  `fhe-lattice-proxies`; the metadata is carried into
  `tools/evidence_database.py` rows, CSV columns, and Markdown summary tables.
- Add a computational-algebra scenario table before promoting any dense-GEMM
  claim into rank/determinant/solve/polynomial wording. Implemented in
  `tools/benchmark_sweep.py` as `computational-algebra-proxies` with dense
  finite-field BLAS, rank-k update, F4 dense phase, FGLM multiplication-matrix,
  and CRT/rational-reconstruction export labels.

Relation to existing queue:

- Deepens every item in this queue because it gives each optimization a
  workload context.

### 23. Native Vector-ALU Production Backend

Status: runtime backend implemented in
[src/backend_vector_alu](../src/backend_vector_alu). The benchmark harness in
[benchmarks/hip_vector_alu_baseline_kernels.hip](../benchmarks/hip_vector_alu_baseline_kernels.hip)
still exists for same-contract release comparisons.

Technical direction:

- Keep the backend semantic-specific: bounded i64/u64 only; no exact-wide,
  finite, wrap64, or CRT fallback.
- Preserve compact native device storage, native currentness, byte counts, and
  source-version accounting.
- Make native-to-RNS conversion an explicit plan choice when a vector-produced C
  needs to feed an RNS operation.
- Keep bounded-u64 AUTO honest: the June 4, 2026 current-v2 refresh replaced
  the old vector-leadership assumption with shape-specific CPU, Direct-HIP, CK,
  and hipBLASLt winners. Vector routing needs current same-contract evidence
  for each promoted family.

Likely first slices:

- Add device-to-device native i64/u64 to RNS conversion kernels. Implemented in
  the current HIP direct/native currentness path as
  `hip_direct_native_i64_to_rns_device` and
  `hip_direct_native_u64_to_rns_device`, with
  `ensure_bounded_native_residues_current_for_rns_plan` materializing RNS
  device residues when a native-current bounded vector matrix feeds an RNS
  plan.
- Add selector explanations for vector vs RNS vs CPU choices.
- Add shape-family vector baselines for skinny/GEMV scenarios. Implemented in
  the scenario corpus as `skinny-gemv`, which emits N=1 bounded i64/u64
  scenarios and includes the canonical runtime `hip-vector-alu-int64` backend.
  `tools/test_benchmark_sweep.py` now pins that scenario to the runtime vector
  backend and N=1 shapes.

Relation to new architecture work:

- Feeds "CPU/GPU Hybrid AUTO", "RNS-Native Chains", and "Plan-Level Algebraic
  Lowering".

### 24. Reusable B Prepack And Tile-Swizzled Layout

Status: rocWMMA has a narrow non-tiled RNS B cache with
`rns_i8_tile_swizzled_b_v1` identity and `prepack-v2` keying. hipBLASLt has
workspace-local repeated-A and repeated-B prepack paths for fixed-prefix
single-K-block RNS work. Direct-HIP now has benchmark-only finite-u8 and
bounded prefix-9 transient-A/resident-B reuse paths. `production_prepack_cache_available`
remains `0`.

Details: the first hipBLASLt 1024 slice removes repeated heuristic selection
from identical hot dispatches by caching the selected matmul algorithm in
process-local memory keyed by device, padded shape, scratch leading dimension,
and workspace size. Windows `gfx1100` smoke captures improved the saved 1024
bounded-i64 sample from 26.6 ms end-to-end / 12.4 ms host RNS GEMM to 14.6 ms
then 12.8 ms end-to-end / 6.3 ms then 5.6 ms host RNS GEMM, but those are
five-repeat smoke captures with visible component-timing noise, not
promotion-grade evidence.
The next hipBLASLt repeated-B slice adds a workspace-local B prepack cache for
single-K-block fixed-prefix RNS GEMM. On Windows `gfx1100`, 1024 bounded-i64
release repeated-B captures reported host RNS GEMM at 5.1 ms and 4.7 ms with B
reuse, while the paired repeated-A run was 9.2 ms. The hipBLASLt transpose-pack
event dropped to 0.35-0.38 ms for repeated-B versus 2.1 ms for repeated-A.
This is still workspace-local repeated-workload evidence, not a durable
production cache or full 64/128/512/1024/2048 promotion sweep.
A matching A-cache slice now caches stable repeated-A operands in the same
workspace-local style. On Windows `gfx1100`, 1024 bounded-i64 repeated-A
captures reported host RNS GEMM at 6.6 ms and 7.5 ms versus the earlier 9.2 ms
paired repeated-A capture; a repeated-A/B capture reported 5.0 ms host RNS GEMM
with zero per-repeat pack phase. These are repeated-workload tuning captures,
not production-cache promotion evidence.

Technical direction:

- Expand B reuse from narrow runtime cache slices into a layout-family policy.
- Keep cache identity tied to backend, target id, selected kernel, semantic,
  prefix schedule hash, tile shape, K-block, operand role, source version,
  finite modulus, device id, matrix layout, and operand layout.
- Support repeated-B first, then A-only and A/B caches after B cache behavior is
  stable across finite, exact-wide, fixed-prefix, and adaptive workloads.
- Pair reusable B with fused transient-A pack and persistent/grouped scheduling.

Likely first slices:

- Benchmark repeated-B scenarios for hipBLASLt, rocWMMA, CK, and direct HIP
  with the same review vocabulary.
- Add finite and exact-wide repeated-B cache experiments.
- Add schedule-aware cache keys for adaptive/tiled plans.

Relation to new architecture work:

- Feeds "Fused Pack+GEMM", "End-To-End Layout Search", "Persistent Grouped
  Scheduler", and "Workspace Arena".

### 25. Bounded-i64 Winner Tuning

Status: current Windows `gfx1100` v2 release-review winners are Direct HIP at
512 and hipBLASLt at 1024. The June 4, 2026 bounded-i64 sweep wrote and locally
installed one reviewed 1024 hipBLASLt v2 cache entry; no 512 accelerator entry
was promotable. The first hipBLASLt slices removed repeated heuristic selection
and added workspace-local repeated-A and repeated-B prepack evidence. The
heuristic cache is intentionally non-durable and does not replace reviewed
autotune-cache identity, library-version rejection, stale-kernel rejection,
timing-split, split-K, finite-u8, or durable cache work. The fixed-prefix RNS
path caches A and B transposed hipBLASLt operands only when source version,
device, shape, prefix, and byte-size identity match; it is skipped for finite-u8,
adaptive/tiled plans, and split-K.

Technical direction:

- Tune winners, not losers. For 512, current evidence points back to Direct HIP;
  focus on pack/export cost, launch count, and one-shot active-prefix behavior
  before spending more time on CK/rocWMMA variants that already lost the v2
  release review.
- For 1024 hipBLASLt, focus on repeated-A/B prepack, scratch/reduce behavior,
  heuristic replay, and external reducer locality.
- Re-run 64/128 plus 2048 before assuming the current 512/1024 split persists
  across the surrounding shape range.
- Treat bounded i64 as the first production proving ground for residue-channel
  fusion, layout search, and generated variants.

Likely first slices:

- Direct-HIP 512 pack/export and launch-count tuning.
- hipBLASLt 1024 repeated-A/B release matrix.
- 2048 bounded-i64 exploratory release matrix.

Relation to new architecture work:

- Feeds "Residue-Channel Fusion", "Fused Pack+GEMM", "End-To-End Layout
  Search", and "Generated Kernel Search".

### 26. Large-Shape Release Matrix

Technical direction:

- Run 2048/4096/8192 exploratory release matrices within caps for bounded,
  finite, exact-wide, and wrap64 where feasible.
- Keep large-shape reports separate from promotion if required baselines cannot
  finish within run caps.
- Use large shapes to detect when launch overhead stops dominating and raw
  matrix-engine throughput, export, memory traffic, or K scheduling becomes the
  real limiter.

Likely first slices:

- 2048 bounded-i64/u64 and finite-u8 first.
- Exact-wide 2048 export-heavy profile.
- Wrap64 direct-HIP v4 2048 exploratory run if runtime is tolerable.

Relation to new architecture work:

- Feeds "Roofline And Evidence Database", "Toolchain Matrix", and "Scenario
  Benchmark Corpus".

### 27. hipBLASLt Path

Status: heuristic lookup is cached in process-local memory for matching
device/shape/workspace. Fixed-prefix RNS repeated-A and repeated-B can reuse
workspace-local transposed operands when identity matches. This is not a public
prepack cache. The active hipBLASLt kernel identity is
`hipblaslt_int8_i32_scratch_reduce_specialized_251_255_256_v2`: the separate
INT32 scratch reducer dispatches fixed-modulus kernels for 256, 255, and 251
and keeps the generic reducer for other ladder moduli.

Technical direction:

- Treat hipBLASLt as a black-box INT8 matmul primitive surrounded by RNS8-owned
  generated pack/reduce/export pipeline.
- Prewarm and replay selected algorithms at plan/workspace creation.
- Separate A pack, B pack, heuristic, matmul, scratch, reduce, export, and D2H
  phases wherever possible.
- Specialize external reduce kernels for 256/255/251 and prefix-9 bounded
  paths. Implemented for hipBLASLt scratch reduction in
  `src/backend_hipblaslt/hipblaslt_kernels.hip`; the source metadata,
  schema gates, stale-autotune checks, and fixtures now use
  `hipblaslt_int8_i32_scratch_reduce_specialized_251_255_256_v2`.
  Windows `gfx1100` smoke evidence under
  `temp/hipblaslt-reducer-v2-smoke/` validates schema and GPU events for
  bounded-i64 512/1024; the 1024 r9 capture reported
  `hipblaslt_i32_to_residue_reduce` median 80.38 us. This is local smoke
  evidence. The later June 4, 2026 current-v2 release review selected this
  kernel at 1024 with 4174 us median end-to-end and installed the reviewed local
  default cache entry for that exact bounded-i64 plan key.
- Use HIP Graphs or grouped host dispatch for repeated hipBLASLt workflows.

Likely first slices:

- Release repeated-A/B bounded-i64 1024 matrix with current workspace-local
  cache.
- A/B prepack support for finite-u8 and exact-wide where layout matches.
- External reducer specialization for 251/255/256 and prefix-9. Implemented
  for the current hipBLASLt scratch-reduce path as the v2 selected kernel above.

Relation to new architecture work:

- Feeds "Reusable B Prepack", "Shared Epilogue DSL", "Fused Pack+GEMM", and
  "Host API Batching".

### 28. CK Path

Technical direction:

- Generate a broader CK alias set: block size, M/N/K per block, CShuffle tile,
  vector widths, M01, K-block cap, reducer family, store vectorization, and
  split-K/Stream-K variants.
- Replace branch/while style centered modulo with shared reducer nodes.
- Avoid `temp_c` copy/add paths when padded or K-split cases can use a cleaner
  output epilogue.
- Dispatch by shape, semantic, finite modulus, prefix schedule, and scenario
  family.

Likely first slices:

- CK finite 1024 ring winner tuning.
- Exact-wide CK export-first profiling before GEMM variants.
- CK generated alias matrix for 512/1024 bounded i64.

Relation to new architecture work:

- Feeds "Generated Kernel Search", "Shared Epilogue DSL", "Finite Data
  Specialization", and "Toolchain Matrix".

### 29. rocWMMA Path

Technical direction:

- Reduce shared-memory round-trip after `store_matrix_sync`; use fragment-map
  evidence before lane-owned residue emission.
- Specialize 256/255/251 reductions in the epilogue with shared reducer nodes.
  Implemented in the shared CK/rocWMMA RNS epilogue reducers and reflected in
  v2 selected-kernel/schema/cache identities; release-reviewed speedups remain
  open.
- Expand B swizzle, transient-A pack, store-path, K-block, and split-K variants.
- For adaptive bounded, group tile entries by prefix, shape, and resource
  behavior to avoid per-entry overhead and tail effects.

Likely first slices:

- Fragment-map harness for `gfx1100`.
- rocWMMA 512 bounded-i64 store/reducer variants.
- Adaptive bounded-i64 1024 prefix-group scheduler variants.

Relation to new architecture work:

- Feeds "Lane/LDS/Store/Prefetch Audits", "Residue-Channel Fusion",
  "Generated Kernel Search", and "Persistent Grouped Scheduler".

### 30. Finite-u8

Status: direct HIP has fixed-modulus pack, GEMM reduction, and export kernels
for 251/255/256. CK and rocWMMA expose common-modulus 251/255/256 selected
kernel identities backed by shared reducer helpers. The June 4, 2026 current-v2
release reviews closed the 64/128/512/1024 promotion question for ring 251,
ring 255, ring 256, and field 251: seven event-valid accelerator entries beat
both CPU and Direct HIP where required and were installed in the local default
cache. Field-251 512 was not promoted because hipBLASLt event timing was
incomplete, and ring-255 64 was not promoted because CPU reference was faster
than the accelerator path.

Technical direction:

- Push finite reducer specialization into CK and rocWMMA epilogues.
- Add `finite_u8_centered_plane_v2` with layout selected by backend and
  distribution.
- Extend the reviewed matrix to 2048, generic prime, and generic composite
  cases before assuming the current explicit-modulus split generalizes.
- Include finite modulus and finite data profile in plan/autotune identity.
- Keep finite semantics explicit: `RNS8_FINITE_RING_U8` is `Z/qZ`, while
  `RNS8_FINITE_FIELD_U8` is a prime-field `GF(p)` contract for `p <= 251`.
  `GF(2^e)` extension fields and word-size prime fields require separate
  future semantics or lowering.
- Treat FFLAS-FFPACK/Givaro as algorithm references and optional CPU
  comparisons, not as evidence that current finite-u8 covers all computational
  finite-field BLAS workloads.

Likely first slices:

- Direct-HIP fixed-modulus 251/255/256 finite-u8 pack/GEMM/export.
  Implemented: canonical byte ingress, centered-residue GEMM reduction, and
  canonical byte export use compile-time modulus kernels for the hot finite
  moduli, while generic runtime-modulus kernels remain the fallback for other
  ring/field moduli.
- CK/rocWMMA 251/255/256 reducer variants.
  Implemented: API planning, schema validation, cache installation, and
  backend-info tests now require explicit v2 selected-kernel identities for
  those moduli. Release-reviewed speedups remain open.
- Finite histogram-guided workload suite.
- 1024 finite winner retuning for CK ring and hipBLASLt field.

Relation to new architecture work:

- Feeds "Finite Data Specialization", "Shared Epilogue DSL", and
  "End-To-End Layout Search".

### 31. Exact-Wide

Status: the June 4, 2026 current-v2 release review covered exact-wide signed
and unsigned 512/1024. Signed 512 now promotes rocWMMA, signed 1024 promotes
hipBLASLt, unsigned 1024 promotes CK, and unsigned 512 stays on Direct HIP. The
three promoted entries are event-valid and installed in the local default cache.
Older exact-wide 64/128 evidence remains historical until rerun with current
selected-kernel identities.

Technical direction:

- Optimize export first: limb loop coalescing, status traffic, compact D2H,
  fixed limb-count specialization, and output precision tiering.
- Keep signed and unsigned export functions and metadata separate.
- Benchmark whether GEMM or limb export dominates before expanding CK/rocWMMA
  matrix kernels.
- Use lazy RNS output for exact-wide chains where no immediate limb export is
  required.

Likely first slices:

- 1/2/4/8/16/32 limb export variants. Implemented for Direct-HIP prefix-20
  signed and unsigned exports with compile-time fixed limb-count kernels,
  plus existing benchmark sweep coverage for those widths.
- Prefix-specialized exact-wide export kernels. Implemented for Direct-HIP
  prefix-20 reconstruction; nonstandard prefixes still use the generic export
  kernel.
- Exact-wide RNS-chain scenario benchmark.

Relation to new architecture work:

- Feeds "Reconstruction Backend", "RNS-Native Chains", "Roofline", and
  "CK Path".

### 32. Direct-HIP RNS Fallback

Technical direction:

- Keep direct HIP as the inspectable correctness/performance fallback and a
  comparison target for accelerator variants.
- Specialize prefix-9 bounded kernels.
- Try one thread computing multiple neighboring columns to reuse A.
- Batch multiple moduli per launch for small shapes.
- Explore fused pack+GEMM for one-shot/small direct-HIP workloads.

Likely first slices:

- Prefix-9 bounded direct-HIP kernel. Implemented first as
  `direct_hip_prefix9_grouped_rns_gemm_v1`, which batches the nine default
  bounded RNS planes into one `grid.z` grouped launch per K block while keeping
  the existing tiled math and centered reducers.
- Multi-modulus launch batching for 64/128. Implemented first for fixed-prefix
  direct-HIP RNS plans: prefix 9 bounded and prefix 20 exact-wide now share the
  grouped launch path, with exact-wide plans reporting
  `direct_hip_prefix20_grouped_rns_gemm_v1`. This has build, correctness,
  schema, and event-smoke evidence, but still needs release-sweep performance
  review before it becomes a durable speedup claim.
- Direct-HIP fused pack+GEMM small-shape baseline. Implemented as the explicit
  `--transient-uniform-small-inputs` bounded prefix-9 benchmark/runtime path
  described in "Fused Pack+GEMM"; it provides schema-valid, event-valid
  same-shape evidence against persistent direct-HIP and public one-shot for
  64/128/512 on local Windows `gfx1100`.

Relation to new architecture work:

- Feeds "Fused Pack+GEMM", "Residue-Channel Fusion", and
  "Verification-Cost Reduction".

### 33. Export/Status Overhead

Technical direction:

- Avoid status memset/D2H when plan bounds or output mode make status traffic
  unnecessary.
- Use compact contiguous D2H staging and separate padded host copy only when
  needed.
- Specialize bounded export for prefix 9 and prefix 20.
- Make status mode semantic: impossible, deferred, sampled/debug, or exact
  depending on plan and benchmark mode.

Likely first slices:

- Prefix-9 bounded export specialization. Implemented for fixed-prefix
  Direct-HIP bounded export, and now also used by scheduled per-tile adaptive
  bounded export through selected-prefix 1..9 compile-time CRT dispatch.
- Exact-wide compact D2H staging. Implemented through the shared Direct-HIP
  compact export copier for exact-wide signed and unsigned limb exports. Large
  padded signed exact-wide exports now stay on the direct compact/pitched D2H
  path by default because repeated Windows `gfx1100` release captures under
  `temp/perf-work-queue/exact-wide-export-staging/` showed forced pinned staging
  losing for signed 512x512x512 limb-4 export (`crt_export` median 3084.39 us
  and 3497.97 us in two forced r33 runs versus 1009.40 us and 2282.29 us in the
  matching disabled r33 runs). `RNS8_HIP_PINNED_EXPORT_STAGING=1` can still force
  the path for experiments, and unsigned exact-wide limb-3 keeps the padded
  default because the same matrix showed a host-export median win
  (774.88 us forced versus 1084.08 us disabled).
- Status phase timing split across accelerator exports. Implemented in the
  benchmark/schema contract: bounded exports report `crt_export_status_memset`,
  `crt_export_kernel`, `crt_export_status_d2h`, and `crt_export_d2h`, while
  exact-wide exports report `exact_wide_export_status_memset`,
  `exact_wide_export_kernel`, `exact_wide_export_status_d2h`, and
  `exact_wide_export_d2h` across Direct-HIP, CK, rocWMMA, and hipBLASLt captures
  that use Direct-HIP export. Full-width exact-wide captures whose limb count
  covers the backend reconstruction width require zero-valued status phases and
  reject stale nonzero status timing.

Relation to new architecture work:

- Feeds "Reconstruction Backend", "Shared Epilogue DSL", "CPU/GPU Hybrid
  AUTO", and "Roofline".

### 34. Adaptive Scheduling

Technical direction:

- Compress schedule entries by selected-prefix group, tile extent, shape, and
  resource behavior.
- Collapse uniform tile bounds to fixed-prefix paths.
- Reuse device schedule buffers when fingerprints match.
- Tune tile size 64/128/256/512 by shape and backend.
- Add priority scheduling for cheap/high-value residue planes when speculative
  exact fallback is in play.

Likely first slices:

- Direct-HIP device schedule upload minimization. Implemented for scheduled
  GEMM workspaces: nonzero adaptive schedules upload only the compact active
  schedule, mixed zero/nonzero schedules retain the public device schedule for
  zero-tile extents, and uniform all-zero schedules report zero schedule
  workspace bytes.
- Broader device schedule buffer reuse across matching fingerprints.
- Prefix-grouped accelerator dispatch for adaptive bounded.
- Adaptive tile-size scenario matrix.

Relation to new architecture work:

- Feeds "Adaptive Prefix Minimization", "Persistent Grouped Scheduler", and
  "Error-Detecting Exact Fast Path".

### 35. Small Shapes

Technical direction:

- Prefer low-overhead direct/vector/CPU/hybrid paths where matrix-engine pack
  work cannot amortize.
- Use HIP Graph capture, host API batching, or persistent grouped scheduler for
  repeated fixed small shapes.
- Keep one-shot and persistent-reuse benchmarks separate.
- Consider fused pack+GEMM and multi-modulus batching before adding heavier
  accelerator variants.

Likely first slices:

- 64/128 one-shot vs persistent scenario matrix. Started with the fixed-prefix
  uniform-small bounded direct-HIP matrix under
  `temp/perf-work-queue/uniform-small-transient-i8/`, covering persistent,
  transient-native A+B, and public one-shot at 64/128/512 for bounded i64/u64.
  This is enough to keep the new transient path honest for small-shape work, but
  the broader CPU/vector/accelerator small-shape selector matrix remains open.
- CPU/direct/vector/accelerator selector explanation for tiny cases.
- HIP Graph repeated-small benchmark mode.

Relation to new architecture work:

- Feeds "CPU/GPU Hybrid AUTO", "Fused Pack+GEMM", "Shape-Specialized Paths",
  and "Persistent Grouped Scheduler".

### 36. Wrap64 Direct-HIP v4

Status: direct HIP v4 is the measured strict wrap64 GPU path for the local
Windows `gfx1100` 64/128/512/1024 validation matrix. Paired release captures
under `temp/perf-work-queue/wrap64-v4/` showed median end-to-end speedups over
v3 of 1.07x, 1.17x, 1.02x, and 5.60x for default 64/128/512/1024 captures, and
1.22x, 4.67x, 1.07x, and 6.74x for reuse-packed-input captures.

Technical direction:

- Keep optimizing the direct-HIP baseline before another matrix-engine
  candidate. v4 uses direct unsigned byte products in the scalar direct-HIP
  kernel, dispatches a safe uint32 low-diagonal accumulator for `K <= 4096`,
  widens at carry propagation, and keeps scalar pack/export kernels for 64-like
  small shapes where vectorized compact pack/export lost end-to-end.
- Use vectorized byte-limb load/store through `uint64_t` where layout and shape
  evidence permits.
- Increase tile K or compute multiple output cells per thread if register
  pressure allows.

Likely first slices:

- Multi-output-cell direct-HIP variant. Implemented as the opt-in
  `direct_hip_wrap64_byte_gemm36_u32acc_colpair_2d_v5` experiment behind
  `RNS8_WRAP64_HIP_COLPAIR_EXPERIMENT=1` for large `K <= 4096` wrap64 shapes.
  It computes two adjacent output cells per thread while reusing the staged A
  byte-limb cell, reports a distinct `wrap64_byte_gemm36_colpair_2d_kernel`
  GPU event phase, and has CPU differential coverage for a 256x257x5 odd-column
  tail. Windows `gfx1100` release captures under
  `temp/perf-work-queue/wrap64-colpair-experiment-current/` are schema/event
  valid, but do not promote the variant: at 512, default v4 beat colpair on
  both GEMM median (904.7 us vs 2501.2 us) and end-to-end median (2339 us vs
  7750 us); at 1024, colpair improved GEMM median only narrowly (4466.0 us vs
  4838.8 us) while losing end-to-end median (9775 us vs 9657 us). Keep default
  routing on v4 and treat colpair as an inspectable tuning candidate, not a
  performance win.
- 2048 v4 exploratory run and ISA/resource report.

Relation to new architecture work:

- Feeds "Lane/LDS/Store/Prefetch Audits", "Shape-Specialized Paths", and
  "Scenario Benchmark Corpus".

### 37. Wrap64 Matrix Engine Redesign

Status: the internal rocWMMA candidate has strong correctness evidence but
loses structurally to direct HIP v4 at every reviewed 64/128/512/1024 shape.

Technical direction:

- Do not lightly iterate on the current candidate.
- Reduce the number of WMMA passes for the 36 byte-pair products or radically
  reduce high-bit correction cost.
- Try grouped diagonals, reused fragments, nibble/byte hybrid layouts, and
  carry-friendly output staging.
- Consider CRT/Ozaki-like slice hybrids only as research paths until they show
  a real structural advantage.

Likely first slices:

- Diagonal grouping design note and microbenchmark.
- Nibble/byte high-bit correction experiment.
- Compare candidate variants against direct-HIP v4 in wrap64 scenario corpus.

Relation to new architecture work:

- Feeds "Generated Kernel Search", "End-To-End Layout Search", "INT4/IU4",
  and "Lane/LDS/Store/Prefetch Audits".

### 38. Reusable A And A/B Caches

Technical direction:

- Add A cache only after B cache behavior is stable.
- Full A/B cache identity must include both source versions and plan
  fingerprints.
- Benchmark setup amortization across repeat counts, not only one repeated
  GEMM.
- Use cache hints from next-op/reuse metadata rather than hidden backend
  inference.

Likely first slices:

- A-only repeated benchmark mode using current cache metadata vocabulary.
- Full A/B repeated bounded-i64 and finite-u8 scenarios.
- Cache-key mismatch diagnostics for both operands.

Relation to new architecture work:

- Feeds "Result Cache And Incremental GEMM", "RNS-Native Chains", and
  "Workspace Arena".

### 39. HIP Graphs And Launch Batching

Technical direction:

- Use graph capture for repeated fixed-shape workflows: pack, per-prefix GEMM,
  export.
- Add FHE/lattice proxy graphs for repeated key switching, rotation batches,
  bootstrapping stage sequences, and fixed-shape NTT/base-conversion pipelines.
- Use host API batching for many enqueued GEMMs/exports where graph capture is
  too rigid.
- Preserve status/error behavior during graph replay.
- Combine with persistent grouped scheduler for small shapes and adaptive
  prefix groups.

Likely first slices:

- Benchmark-only begin/enqueue/end batch API.
- HIP Graph replay for repeated 64/128 bounded and finite cases.
- Compare graph replay against device persistent queue for many small tasks.

Relation to new architecture work:

- Feeds "Persistent Grouped Scheduler", "Small Shapes", and "Device Plan
  Cache".

### 40. Instrumentation

Status: the June 2026 GFX1100 evidence-tooling pass added the first dedicated
event and ISA reporting lane.

- Benchmark captures now carry helper-lane objects for `plan_packing`,
  `plan_lowering`, `requested_next_op`, `output_policy`, `auto_selector`,
  `target_variant`, and `device_allocation`. These fields make pack layout,
  lowering path, next-operation intent, status/export policy, AUTO fallback
  reasoning, target namespace, and post-warmup allocation behavior visible to
  `tools/result_compare.py` and `tools/benchmark_sweep.py`.
- CK and rocWMMA event captures now have a deep scope,
  `accelerator_backend_default_stream_deep_kernel_events_with_direct_hip_pack_export`,
  with aggregate pack/matmul/copy/add labels plus zero-based per-prefix labels.
  This is attribution evidence only; it is not a speedup claim.
- Native vector-ALU captures use
  `vector_alu_default_stream_native_int64_operation_groups` and expose pack A,
  pack B, status memset, native kernel, status D2H, output D2H, and aggregate
  pack/GEMM/export labels.
- Use `tools/gpu_event_report.py` to validate a capture and rank event phases by
  median/share before deciding where the optimizer should spend time.
- Use `tools/gpu_isa_report.py --target gfx1100 --object <hip-object>` or
  `--build-tree <build-dir>` to write LLVM objdump ISA summaries under
  `temp/isa-reports/`. The report records symbols, WMMA/MFMA counts, global
  stores, LDS mentions, waits, and VGPR/SGPR/occupancy when available. RGA CLI
  reporting remains optional. Add `--capture <capture.json>` to link the ISA
  summary to a validated benchmark contract.
- Use `tools/gpu_counter_report.py <capture.json> --counter <csv-or-json>
  --isa-summary <isa-summary.json>` to assemble temp-only counter/ISA
  explanations under `temp/gpu-counter-reports/`.
- Keep all captures, dumps, and reports in ignored `temp/`; do not promote
  instrumentation output into autotune cache entries or performance claims.

Technical direction:

- Add per-kernel/per-prefix/per-tile event hooks for CK and rocWMMA.
- For FHE/lattice proxies, add per-transform, per-prime, per-key-switch,
  per-rotation, and per-bootstrapping-stage timing labels before comparing
  dense-GEMM-adjacent work against NTT/key-switch-dominated work.
- Feed RGA/disassembly output, VGPR/SGPR/LDS, waits, stores, and occupancy into
  the evidence database.
- Keep counters debug/probe-only and out of normal benchmark hot paths.
- Add thermal/power/run-order metadata where available for consumer GPU runs.

Likely first slices:

- CK/rocWMMA per-kernel timing split.
- RGA summary parser into benchmark review reports.
- Interleaved baseline ordering in sweeps to reduce thermal noise.

Relation to new architecture work:

- Feeds "Roofline And Evidence Database", "Generated Kernel Search", and
  "Toolchain Matrix".

### 41. Host/Transfer

Technical direction:

- Use pinned staging pools, mapped host memory, async D2H export, and compact
  export buffers only when benchmark metadata reflects them.
- Separate host API overhead from device work.
- Cache workspace allocations and avoid first-use allocation inside measured
  repeats.
- Overlap pack next, GEMM current, and export previous with streams when
  workspace partitioning is clear.

Likely first slices:

- Compact contiguous export D2H fast path. Implemented for Direct-HIP bounded,
  finite-u8, exact-wide, and wrap64 exports: contiguous host outputs now use a
  linear `hipMemcpy`, while padded leading dimensions keep the existing
  `hipMemcpy2D` path. Windows `gfx1100` release smokes under
  `temp/compact-export-release/` validated schema/events across bounded i64,
  finite-u8, exact-wide, and wrap64; finite-u8 512 median export D2H improved
  from the prior 66.2 us r33 capture to 49.2 us, but end-to-end averages remain
  too noisy for a headline claim.
- Pinned host export staging for large padded Direct-HIP outputs. Implemented
  as an internal reusable thread-local HIP pinned host buffer for bounded,
  finite-u8, and selected exact-wide Direct-HIP exports whose compact output
  copy is at least 64 KiB and whose caller destination has padding. The backend
  copies compact device export buffers into pinned staging, scatters into the
  caller's requested leading dimension on the host, preserves the existing
  required `crt_export_d2h` / `finite_export_d2h` /
  `exact_wide_export_d2h` GPU event labels, and emits an ignored diagnostic
  timing sample named `export_host_staging_copy` when backend timing is enabled.
  `rns8-bench` now records `direct_hip_export_staging_policy`,
  `direct_hip_pinned_export_staging_threshold_bytes`, and the benchmark output
  destination layout in `timing_metadata`, so capture metadata distinguishes
  padded-only default, disabled, and forced staging runs. The benchmark also
  exposes `--output-ld-padding`; captures now report `output_logical_ld`,
  `output_ld_padding`, `benchmark_output_logical_ld`, and
  `benchmark_output_ld_padding`, and schema/sweep review keys treat padded and
  contiguous destinations as different contracts.
  `RNS8_HIP_PINNED_EXPORT_STAGING=0` disables the staging path, while
  `RNS8_HIP_PINNED_EXPORT_STAGING=1` forces it for contiguous-output A/B
  measurements and for force-only semantics. A 512x512 contiguous-output
  Windows `gfx1100` smoke did not support default-on staging, so contiguous
  benchmark outputs keep the existing linear D2H path by default. Focused
  128x128 Windows `gfx1100` smokes under
  `temp/perf-work-queue/padded-output-ld/` validated default padded, disabled
  padded, and forced contiguous captures with matching logical checksums and
  required GPU events. Exact-wide signed padded staging is now force-only after
  local 512x512x512 limb-4 r33 captures showed it losing to disabled staging on
  host export medians, while exact-wide unsigned limb-3 keeps default padded
  staging because the same validation showed a host-export median win. This is
  an implemented transfer path with semantic-specific default policy, not a
  blanket headline speedup claim.
- Wrap64 padded export staging follow-up. Implemented the reusable compact
  export copier as a shared Direct-HIP internal utility so the wrap64 byte-limb
  export path can exercise the pinned staging pool and host scatter path without
  duplicating transfer code. Wrap64 does not enable padded pinned staging by
  default: `RNS8_HIP_PINNED_EXPORT_STAGING=1` can force the path for experiments,
  but default wrap64 export stays on the direct compact/pitched D2H copy until a
  real win exists. Windows `gfx1100` captures under
  `temp/perf-work-queue/wrap64-padded-export-staging/` validated schema/events
  for default, disabled, and forced staging at 128 and 512 with matching
  checksums inside each shape. The 512x512 forced capture lost badly versus the
  default policy (`wrap64_export_d2h` median 244.8 us versus 66.4 us, end-to-end
  average 7177.78 us versus 3029.56 us), so this is correctness/tooling
  coverage and an explicit deprioritization of default wrap64 pinned export
  staging, not a promoted speedup.
- Async exact-wide export/D2H overlap experiment.
- Multi-stream repeated-B pipeline scenario.

Relation to new architecture work:

- Feeds "CPU/GPU Hybrid AUTO", "Reconstruction Backend", "Workspace Arena",
  and "Roofline".

### 42. AUTO And Cache Architecture

Technical direction:

- Keep exact-shape keys mature, then add shape-family and scenario-family
  recommendations.
- Consider semantic, shape, prefix schedule, K-block, tile size, finite
  modulus, target id, HIP/accelerator version, data profile, reuse profile,
  output mode, and next-op hints.
- Reject stale kernel names, stale epilogue names, and mismatched toolchains.
- Keep raw benchmark cache writes separate from reviewed cache installation.

Likely first slices:

- Selector explanation output for reviewed cache hits and fallbacks.
- Data-profile and reuse-profile fields in planning metadata.
- Shape-family recommendation layer after exact-shape cache hits.

Relation to new architecture work:

- Feeds "Plan-Level Algebraic Lowering", "Toolchain Matrix", "Scenario
  Benchmark Corpus", and "CPU/GPU Hybrid AUTO".

### 43. AMDGPU Builtins

Technical direction:

- Keep builtins as a microkernel lab after CK/rocWMMA identify a concrete
  bottleneck.
- Start with one semantic, one shape family, one target id, and one instruction
  family.
- Use direct lane mapping and store-path control to answer questions rocWMMA
  cannot.

Likely first slices:

- `gfx1100` builtin residue-store microkernel if rocWMMA store path remains
  blocked.
- Builtin prototype for one finite-u8 or bounded-i64 shape.
- Compare against generated rocWMMA/CK variants before expanding.

Relation to new architecture work:

- Feeds "Lane/LDS/Store/Prefetch Audits" and "Generated Kernel Search".

### 44. INT4/IU4

Technical direction:

- Treat as packed layout research, not production work.
- Evaluate bit-sliced, nibble-sliced, byte-sliced, and mixed 4x8
  decompositions.
- Account for centered residue range, pack/unpack cost, K-block safety,
  reducer cost, and memory traffic.
- Retire per semantic/target when packed layout plus epilogue cannot beat tuned
  INT8.

Likely first slices:

- `rns_i4_packed_v0` finite or bounded narrow-distribution experiment.
- Nibble-unpack overhead microbenchmark.
- Compare against residue-channel fusion before expanding.

Relation to new architecture work:

- Feeds "End-To-End Layout Search", "Wrap64 Matrix Engine Redesign", and
  "Generated Kernel Search".

### 45. FP8/Ozaki, Strassen, Sparsity

Technical direction:

- Keep FP8/Ozaki and CRT/Ozaki hybrids research-only and out of default exact
  APIs.
- Use Ozaki/slice thinking for exact-wide and wrap64 only when it may reduce
  the number of low-precision GEMMs or high-bit correction passes.
- Treat Strassen and structured sparsity as workload-backed experiments, not
  default dense GEMM work.
- Account for Strassen/Winograd prefix inflation: added/subtracted
  intermediates can widen bounds, increase selected modulus count, add pack and
  temporary traffic, and erase the multiplication-count win.
- Add sparse/structured-zero exact paths only for explicit workload structure:
  zero rows/cols, block sparse, diagonal/banded, triangular, or Gram products.

Likely first slices:

- Hybrid slice/RNS design sketch for wrap64 and exact-wide.
- Structured-zero scenario benchmark.
- Symmetric/Gram direct-HIP prototype if workload appears.

Relation to new architecture work:

- Feeds "Shape-Specialized Paths", "Wrap64 Matrix Engine Redesign", and
  "Scenario Benchmark Corpus".

### 46. Multi-GPU

Technical direction:

- Linux-only later. Split by modulus groups first, not K, unless a real Linux
  profile says otherwise.
- Account for reconstruction, scheduling, peer transfer, and output-domain
  costs.
- Treat multi-GPU as a production-platform project, not a Windows `gfx1100`
  local speedup.

Likely first slices:

- Linux ROCm multi-GPU capability and topology inspection.
- Modulus-group scheduling design for bounded prefix-9 and exact-wide.
- Reconstruction placement comparison: per-GPU partial vs final gathered.

Relation to new architecture work:

- Feeds "Persistent Grouped Scheduler", "Reconstruction Backend", and
  "Toolchain Matrix".

## Best Next Batches

### Batch A: Measurement And Scenario Foundation

- Build the roofline/evidence database analysis layer from current schema v4
  captures and review reports.
- Add scenario benchmark families for repeated-B, exact-wide export-heavy,
  finite distributions, RNS chains, small one-shot, many-small, skinny/GEMV,
  wrap64 carry-heavy, and large exploratory shapes.
- Add computational-algebra scenarios for finite-field BLAS, modular
  rank/determinant/solve/nullspace, rational reconstruction, polynomial
  matrix/modular-composition lowerings, PLUQ/CUP/PLE rank profiles,
  p-adic/Dixon solve, early-terminated CRA, F4 dense finite-field matrices,
  FGLM multiplication-matrix conversion, NTT/product-tree pressure, rectangular
  rank-k, and structured matrix families.
- Add FHE/lattice-derived NTT/key-switch/rotation/bootstrap proxies to the
  same scenario benchmark family, with evidence scope and output-domain
  metadata recorded separately from dense-GEMM claims.
- Add per-kernel CK/rocWMMA event timing and RGA resource summaries.

### Batch B: Immediate Shape Wins

- Finish bounded-i64 winner tuning for Direct HIP 512 and hipBLASLt 1024; the
  current v2 release review installed the 1024 hipBLASLt cache entry and left
  512 on Direct HIP.
- Continue exact-wide export tuning before broadening exact-wide GEMM variants;
  the current 512/1024 v2 matrix is reviewed and installed, but 64/128, 2048,
  limb-count variants, and chain/lazy-export workloads remain open.
- Extend finite-u8 CK/rocWMMA reducer specialization beyond the now-reviewed
  64/128/512/1024 ring-251/ring-255/ring-256/field-251 matrix into 2048,
  generic prime, and generic composite cases.
- Continue direct-HIP wrap64 v4 follow-up tuning before another matrix-engine
  candidate.

### Batch C: Representation Wins

- Implement multi-modulus pack and residue-channel fusion experiments.
- Add fused pack+GEMM for one-shot/small bounded and finite workloads.
- Compare end-to-end layouts across RNS, finite, exact-wide, and wrap64.
- Add polynomial-tower and Q/P-basis layout sketches for FHE/lattice proxy
  scenarios without treating them as public RNS8 storage formats.

### Batch D: Scheduler And Reuse Wins

- Expand repeated-B cache work across rocWMMA, hipBLASLt, CK, finite, and
  exact-wide.
- Add persistent/grouped scheduler experiments for adaptive prefix groups and
  many small GEMMs.
- Add tower/key-material reuse scenario labels so FHE/lattice-inspired reuse
  does not get collapsed into ordinary A/B matrix reuse.
- Add HIP Graph and host batching modes for repeated fixed-shape workflows.

### Batch E: Research And Platform Work

- Keep AMDGPU builtins, INT4/IU4, Ozaki hybrids, Strassen, sparsity, and
  multi-GPU behind explicit research/platform work.
- Run Linux ROCm and Instinct work only on real supported hosts with separate
  evidence from Windows `gfx1100`.
