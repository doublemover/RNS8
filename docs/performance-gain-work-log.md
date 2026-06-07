# Performance Gain Work Log

This document preserves dated execution updates and branch-local status notes
that used to live in [performance-gain-work-queue.md](performance-gain-work-queue.md).
Use the work queue as the active control panel; use this log only for historical
context and evidence breadcrumbs.

## Recent Execution Status

June 6, 2026 rank-74 K-block/tile-K large-shape closeout:

- Rank 74 moved from the active queue to the completed-work archive after the
  single-GPU K-block policy surface gained full scenario, schema, report, and
  local Windows `gfx1100` evidence. `k_block_tile_variants.json` now covers
  bounded-i64, exact-wide-signed, finite-ring u8, and strict wrap64 at
  1024/2048/4096 with default anchors and explicit benchmark-only K-block
  candidates. Multi-GPU split-K and distributed GEMM remain out of scope.
- The closeout under `temp/rank74-k-block-tile-variants-20260606/` produced
  34 schema-valid release captures, required GPU events for GPU rows, a
  Direct-HIP ISA sidecar, and a tile-shape report with 12 K-block candidates.
  The report finds no local promotions: 11 candidates stay experimental because
  profiler occupancy/counter evidence is still missing, and finite-u8 1024 is
  deprioritized because the K-block policy is slower than the default
  Direct-HIP anchor. The 4096 rows stay non-promotional where CPU/reference
  anchors are intentionally omitted to avoid impractical local runtime.

June 6, 2026 rank-68 strict wrap64 Direct-HIP v4 tuning closeout:

- Rank 68 moved from the active queue to the completed-work archive after the
  strict wrap64 tuning lane gained a dedicated report gate and current
  Windows `gfx1100` release evidence. `wrap64-carry` now covers CPU byte-limb
  reference plus Direct-HIP v4 at 512/1024/2048, exploratory Direct-HIP 4096
  without the pathological CPU byte-limb release-repeat row, reuse-packed
  inputs at 512/1024/2048, full-path HIP Graph replay at 512/1024, K-block
  policy rows at 1024/2048/4096, and rocWMMA matrix-engine candidates at
  512/1024/2048.
- The closeout under `temp/rank68-wrap64-direct-hip-tuning-20260606/` produced
  18 schema-valid release captures. Required GPU events passed for every
  non-graph GPU capture, and ISA sidecars were generated for Direct-HIP and
  rocWMMA. `tools/wrap64_direct_hip_tuning_report.py` reports four local
  workload wins, six deprioritized candidates, zero missing Direct-HIP
  baselines, and one intentional missing-reference exploratory 4096 group.
  Reuse-packed inputs win at 512 and 1024; full-path HIP Graph replay wins at
  512 and 1024; reuse at 2048, K-block policy rows, and rocWMMA candidates are
  not promoted from this gate. The 4096 K-block row is event-valid but remains
  experimental because the byte-limb CPU reference is omitted from release
  sweeps due to impractical runtime.

June 6, 2026 rank-58 HIP Graph replay full-path closeout:

- Rank 58 moved from the active queue to the completed-work archive after the
  graph lane expanded beyond resident RNS chains into full pack/GEMM/export
  captures for bounded i64/u64, finite ring/field u8, and strict wrap64. The
  bounded, finite, and wrap64 graph bodies capture H2D inputs, pack kernels,
  Direct-HIP compute, export/status work, and D2H output on explicit streams
  with graph setup amortized into the reported per-repeat comparison. The
  resident-chain graph mode remains unchanged.
- `benchmarks/scenarios/hip_graph_replay.json`, schema validation, sweep
  metadata, generated metadata registry constants, and
  `tools/hip_graph_replay_report.py` now distinguish resident-chain graph,
  bounded full pack/GEMM/export graph, finite-u8 full pack/GEMM/export graph,
  and strict wrap64 full pack/GEMM/export graph execution modes. The finite
  path also fixed its stream correctness issue by launching fixed-modulus GEMM
  kernels on the captured stream.
- The Windows `gfx1100` release sweep under
  `temp/rank58-hip-graph-full-release-20260606/` produced 20 schema-valid
  captures and 10 checksum-matched graph/baseline comparisons with zero missing
  baselines. Bounded full-path graph rows are deprioritized at 0.60x to 0.86x
  versus ordinary Direct-HIP, and finite-u8 full-path graph rows are
  deprioritized at 0.48x to 0.61x. Strict wrap64 graph replay is a local
  benchmark-only workload win: 512 improves from 2739 us to 1626 us, 1.69x,
  and 1024 improves from 6546 us to 5340 us, 1.23x. This adds no README
  headline claim, installed cache entry, AUTO route, public async API, Linux
  readiness claim, or CDNA performance claim. Mixed accelerator graph capture
  remains deferred until hipBLASLt/CK/rocWMMA handle, stream, and scratch
  lifetimes are graph-safe.

June 6, 2026 rank-56 tile-shape autotuning gate closeout:

- Rank 56 moved from the active queue to the completed-work archive after the
  tile-shape lane gained release-mode CPU/default Direct-HIP anchor scenarios
  and an enforceable report gate. `tile_shape_sweeps.json` now covers
  bounded-i64 512/1024, bounded-u64 1024, exact-wide signed 1024, and
  finite-ring u8 2048 with CPU anchors, default 128x128 Direct-HIP anchors, and
  non-default Direct-HIP tile candidates. `tools/tile_shape_report.py` now
  requires schema-valid release captures, same-contract CPU and default
  Direct-HIP baselines, required GPU events, selected-kernel/resource identity,
  tile-aware autotune keys, and counter or ISA resource signals before a local
  tile-shape candidate can promote. This is scenario/report infrastructure and
  does not add a README claim, installed cache entry, AUTO route, Linux
  readiness claim, or CDNA performance claim.
- The Windows `gfx1100` release sweep under
  `temp/rank56-tile-shape-release-20260606/` produced 20 schema-valid captures
  with zero deferrals: 10 non-default candidates, five CPU anchors, and five
  default 128x128 Direct-HIP anchors. Required GPU events passed for every
  GPU capture, and `tools/gpu_isa_report.py` now extracts VGPR/SGPR/LDS/scratch
  resource fields from AMDGPU metadata via `llvm-readobj` instead of leaving
  them null when disassembly omits them. RGA binary-analysis parsing failed on
  the extracted code object, so occupancy/profiler evidence remains missing on
  Windows. The final tile report deprioritized seven slower candidates and kept
  three near/faster candidates experimental: exact-wide signed 1024 64x128
  at 1.011x, finite-ring u8 2048 64x64 at 1.069x, and finite-ring u8 2048
  256x128 at 1.083x versus their default Direct-HIP tile anchors. No tile-shape
  variant promoted locally because the occupancy/profiler gate remains intact.

June 6, 2026 rank-55 streaming overlap gate closeout:

- Rank 55 moved from the active queue to the completed-work archive after the
  streaming-overlap lane became an executed Direct-HIP bounded capture
  contract instead of metadata-only bookkeeping. `--streaming-overlap` now
  routes fixed-prefix bounded i64/u64 reuse-B captures through a benchmark-only
  pack-next/GEMM-current/export-previous pipeline with pinned host staging,
  double-buffered A/C matrices, three nonblocking streams, explicit HIP
  dependency events, per-repeat export status checks, schema-visible stream and
  buffer counts, and the
  `direct_hip_streaming_overlap_multistream_operation_groups` event scope. The
  `streaming-overlap` scenario family now includes CPU and serial Direct-HIP
  reuse-B baselines plus fixed-prefix streaming candidates, and
  `tools/streaming_overlap_report.py` requires those baselines, release
  warmups/repeats, correctness, and required GPU events before local promotion.
  The Windows `gfx1100` smoke under
  `temp/rank55-streaming-overlap-smoke-20260606/` produced schema-valid,
  event-complete bounded i64/u64 streaming captures. This does not add a README
  headline claim, installed cache entry, AUTO route, Linux readiness claim, or
  CDNA performance claim.

June 6, 2026 rank-54 adaptive grouped scheduler gate closeout:

- Rank 54 moved from the active queue to the completed-work archive after the
  adaptive scheduler surface became an executed Direct-HIP capture contract
  instead of metadata-only bookkeeping. `--adaptive-grouped-scheduler` now
  reports executed status when the capture uses
  `direct_hip_grouped_active_prefix_schedule_rns_gemm_v3`, records active
  prefix count, active entry count, independent-launch model, aggregate launch
  model, launch-reduction ratio, and aggregate `rns_gemm_kernel_group` event
  scope, and schema rejects requested Direct-HIP grouped-kernel captures that
  still claim unsupported status. `tools/adaptive_grouped_scheduler_report.py`
  now requires CPU and non-requested Direct-HIP same-contract baselines,
  release warmups/repeats, required GPU events, adaptive execution, and
  launch-reduction metadata before a candidate can promote locally. The
  `adaptive-grouped-scheduler` scenario family now includes bounded-i64 and
  bounded-u64 1024 uniform-small adaptive-prefix candidates plus CPU/direct
  baselines. This
  does not add a README headline claim, installed cache entry, AUTO route,
  Linux readiness claim, or CDNA performance claim.

June 6, 2026 rank-53 modulus-set and residue-count gate closeout:

- Rank 53 moved from the active queue to the completed-work archive after the
  modulus-set lane became an explicit non-promoting search and review surface.
  `tools/modulus_set_search.py` now emits schema-v2 offline reports for
  candidate RNS ladders, generated default/prime/NTT-front/coprime candidates,
  modulus ordering, prefix products, bounded/exact-wide required-prefix bits,
  reducer-cost summaries, Garner/CRT prefix constants, and NTT-friendly small
  prime hints. `tools/modulus_set_autotune_report.py` joins those search
  reports to benchmark captures, requires schema-valid metadata, a
  same-workload default anchor for every experimental row, and explicit
  non-promoting/cache-blocker policy. Benchmark JSON now records
  `runtime_selectable`, `search_report_required`, `default_change_gate`, and
  residue-count promotion eligibility. The schema rejects stale experimental
  modulus metadata, and the `modulus-set-autotune` scenario family now covers
  fewer-plane experiments, fixed-prefix anchors, exact-wide range-product
  anchors, and an NTT-prime-front candidate. This does not change the runtime
  default ladder, installed cache entries, README claims, or Linux/CDNA status.

June 6, 2026 rank-51 Direct-HIP resident redesign closeout:

- Rank 51 moved from the active queue to the completed-work archive after the
  rejected selected-prefix colpair route was replaced by
  `direct_hip_grouped_active_prefix_schedule_rns_gemm_v3`, a grouped
  active-schedule resident RNS launch path for arbitrary selected prefixes.
  The implementation keeps public plan schedule entries unchanged while using
  internal active-schedule entries to carry the modulus plane, adds benchmark
  `resident_redesign` metadata, schema validation, scenario plumbing, and
  `tools/direct_hip_resident_redesign_report.py`. The Windows `gfx1100`
  closeout compared the new route against the previous selected-prefix
  resident default at bounded-i64 512 with seed `20260605`, three warmups, nine
  repeats, schema-valid captures, required GPU events, matching checksum, and
  ISA/counter resource explanation. The report classifies the new route as a
  `route_candidate`: 2082 us median end-to-end versus 29116 us before
  (`13.98x`), 523 us versus 6460 us RNS GEMM (`12.35x`), and 743 us versus
  11450 us export (`15.41x`). This is local Windows `gfx1100` Direct-HIP route
  evidence only; profiler counters were not present, and no Linux/CDNA,
  installed-cache, or README headline claim changes from the closeout.

June 6, 2026 rank-50 bounded-i64 1024 hipBLASLt review-gate closeout:

- Rank 50 moved from the active queue to the completed-work archive after
  `tools/bounded_i64_1024_review.py` became a complete same-target disposition
  gate for the narrow bounded-i64 1024 hipBLASLt lane. The report now requires
  CPU, Direct HIP, runtime vector ALU, CK, and rocWMMA comparator coverage;
  requires hipBLASLt per-repeat, reuse-A, reuse-B, and reuse-A+B pack-mode
  coverage; compares the candidate against both Direct HIP and the fastest
  required comparator; and consumes target-validation, variance,
  counter/resource, and promotion-ledger sidecars before reporting keep,
  replace, experimental, or drop/deprioritize dispositions. The self-test
  covers full success, missing comparator, missing reuse mode, slow hipBLASLt,
  and unsupported-accelerator cases. This is cache-maintenance/review-control
  infrastructure only; it does not add a README headline claim, installed cache
  replacement, Linux readiness claim, or CDNA performance claim.

June 6, 2026 rank-39 error-detection policy closeout:

- Rank 39 moved from the active queue to the completed-work archive after
  `error_detection_policy` became a schema-visible benchmark/sweep metadata
  surface and `tools/error_detection_policy_report.py` added a dedicated
  research-only safety gate. Enabled captures must declare verification basis,
  false-negative policy, final exact comparison, research-only scope,
  non-promotable/cache-ineligible status, no runtime routing, and unchanged
  deterministic default exact API behavior. Probabilistic product-check rows
  must also record positive verification rounds and RNG seed status. The
  `error-detecting-fast-path` scenario family includes CPU, Direct-HIP, and CK
  research rows, and the closeout under
  `temp/rank39-error-detection-policy-closeout-20260606/` validates one
  Freivalds-style proxy row with zero blockers. This is not a probabilistic
  correctness mode, cache claim, AUTO route, README speedup, or platform
  performance claim.

June 6, 2026 rank-38 verification amortization closeout:

- Rank 38 moved from the active queue to the completed-work archive after
  `tools/verification_amortization_report.py` added a focused safety gate for
  repeated exact workloads that reuse CPU/reference structure. The report
  consumes schema-valid captures and release review sidecars, requires final
  exact comparison to remain required and explicit, blocks any amortized row
  that becomes promotable or cache-eligible, requires tooling/proxy-only
  scenario scope, and requires CPU/reference coverage in the review group. The
  FHE/lattice key-switch proxy scenario now schedules a CPU backend, and the
  benchmark-sweep self-test rejects future verification-amortized scenarios
  that omit CPU. The closeout under
  `temp/rank38-verification-amortization-closeout-20260606/` validates one
  schema-v4 amortized proxy row with CPU-reference, Direct-HIP, and hipBLASLt
  review candidates and zero blockers. This does not add probabilistic
  correctness, cache promotion, AUTO routing, or platform performance claims.

June 6, 2026 rank-36 AUTO shape-family gate closeout:

- Rank 36 moved from the active queue to the completed-work archive after
  `tools/auto_shape_family_gate.py` added a concrete exact-cache/AUTO safety
  gate around the existing shape-family shadow recommendations. The gate reads
  reviewed cache JSON plus `shape_family_shadow_report.py` output, verifies
  that `src/core/autotune_cache.cpp` still uses exact cache lookup with no
  runtime shape-family lookup, requires all family recommendations to remain
  non-routing and non-promotable, checks the required target, semantic,
  layout, and output-contract boundary fields, and verifies recommendation
  basis keys exist in the reviewed cache. The closeout under
  `temp/rank36-auto-shape-family-gate-20260606/` regenerated the shadow report
  and validates three query classes with zero blockers: an exact 512 cache hit,
  a same-boundary 768 advisory representative, and a rejected
  cross-output-contract 768 query. AUTO routing remains exact-cache only.

June 6, 2026 rank-33 reconstruction/export closeout:

- Rank 33 moved from the active queue to the completed-work archive after
  `tools/reconstruction_export_report.py` added a rank-level classifier for
  GPU CRT/export reconstruction variants. The closeout report under
  `temp/rank33-reconstruction-export-closeout-20260606/` consumes the 14
  schema-valid Windows `gfx1100` rank-48 selector A/B captures plus their
  release review reports, checks all four required variant classes, and finds
  no blockers. Seven hip-direct captures have required GPU events. The
  explicit selector comparisons classify compact D2H and status-elided
  hip-direct candidates as end-to-end losers (`0.991x` and `0.960x`), while
  prefix20 fixed export and tree/CRT reconstruction remain narrow local
  experimental wins (`1.020x` and `1.048x`). This is selector/reconstruction
  evidence only: no README headline claim, installed cache entry, default
  AUTO route, Linux, or CDNA claim changes.

June 6, 2026 rank-24 reducer/epilogue registry closeout:

- Rank 24 moved from the active queue to the completed-work archive after
  reducer and epilogue identity validation became registry-backed.
  `metadata/epilogues.yaml` now declares generated reducer identities,
  finite-modulus identity patterns, and reducer families that point at known
  selected kernels, generated helpers, and ISA evidence. The metadata generator
  emits Python/C++ constants for epilogue modes and generated reducer
  identities, and benchmark schema helper metadata validates
  `timing_metadata.generated_reducer_identity` through those generated
  constants instead of duplicated regexes. This is registry/schema
  infrastructure only: no runtime hot path parses the registry, no default
  route or cache entry changes, and no new speed claim is made.

June 6, 2026 rank-22 zero-skip expansion closeout:

- Rank 22 moved from the active queue to the completed-work archive after
  `tools/zero_skip_expansion_report.py` made the backend support boundary
  explicit. The closeout report under
  `temp/rank22-zero-skip-closeout-20260606/` reuses the 51 schema-valid
  `bound-discovery` captures. It finds 15 row/column proof-product captures,
  15 zero-output tile captures, three Direct-HIP row/column skip rows that are
  all scan-derived, six CK/rocWMMA rows that are correct full-tile fallbacks
  rather than row/column product skips, six CPU/vector rows unsupported for the
  scheduled proof-mask contract, and zero caller-provided or naturally sparse
  proof captures. This closes rank 22 as no-expansion evidence: Direct-HIP
  remains the only backend with row/column product mask execution, CK/rocWMMA
  keep whole zero-output tile handling only, hipBLASLt remains unsupported for
  scheduled/adaptive tiled RNS plans, and scan-derived proof masks stay out of
  default routing.

June 6, 2026 rank-10 Direct-HIP prefix fusion closeout:

- Rank 10 moved from the active queue to the completed-work archive after
  `tools/direct_hip_prefix_fusion_report.py` consolidated the prefix-specific
  implementation evidence. The closeout report under
  `temp/rank10-prefix-fusion-closeout-20260606/prefix-fusion-report/` compares
  current schema-valid/event-valid public one-shot colpair captures against the
  legacy v1 before-captures: bounded-i64 512 improves from 9368 us to 3048 us
  median end-to-end, `3.07x`, and bounded-u64 512 improves from 4353 us to
  2841 us, `1.53x`. The same report keeps the resident selected-prefix colpair
  attempt deprioritized at 4010 us versus 2434 us median and imports the
  prefix20 fixed-export selector rows as experimental. This closes the rank as
  classified implementation evidence only; no installed cache entry, README
  headline claim, default resident route, or Linux/CDNA claim changes.

June 6, 2026 rank-30 HIP Graph replay release-size closeout:

- Rank 30 moved from the active queue to the completed-work archive after a
  release-size Windows `gfx1100` same-contract graph-vs-ordinary chain
  validation under `temp/rank30-hip-graph-replay-release-20260606/`. The
  refreshed `hip-graph-replay` scenario family now covers bounded-i64,
  bounded-u64, exact-wide signed, and exact-wide unsigned residue-current
  chain3 workflows at 512 and 1024. The sweep produced 16 schema-v4 captures,
  eight release-satisfied generic review groups, zero duplicate backend records,
  zero missing required baselines, required GPU events for all ordinary
  non-graph baselines, and schema-valid wall-clock graph timing for all graph
  captures. `tools/hip_graph_replay_report.py` classified eight checksum-matched
  graph comparisons with prepack setup plus graph capture and instantiate cost
  amortized into the per-repeat decision: four candidate workload wins and four
  deprioritized rows. This closes release-size graph replay classification only;
  no public async API, AUTO/cache entry, README claim, default route, or
  Linux/CDNA claim changes.

June 6, 2026 rank-20 Direct-HIP reuse expansion closeout:

- Rank 20 moved from the active queue to the completed-work archive after a
  current Windows `gfx1100` release classification under
  `temp/rank20-direct-hip-reuse-expansion-release-20260606/`. The new
  `direct-hip-reuse-expansion` scenario family covers adaptive bounded-u64
  reuse-A/reuse-B, finite-u8 native-A/reuse-B, exact-wide residue-current chain
  reuse-B, and strict wrap64 reuse-packed-inputs profiles. The matrix produced
  70 schema-v4 captures, 26 release-satisfied review groups, zero duplicate
  backend records, and zero missing required baselines after explicit reuse
  evidence groups delegate setup-inclusive matching to
  `tools/direct_hip_reuse_expansion_report.py`. The rank-specific report
  classified 14 Direct-HIP reuse comparisons: strict wrap64 A+B reuse wins at
  512 and 1024, while adaptive bounded-u64, finite-u8, and exact-wide chain
  reuse rows are deprioritized at the 9-repeat gate. This closes the
  classification lane only; no AUTO/cache entry, README claim, default route,
  or Linux/CDNA claim changes.

June 6, 2026 rank-44 resident lifetime contract closeout:

- Rank 44 moved from the active queue to the completed-work archive after the
  general resident matrix contract became a public, read-only inspection
  surface. `rns8_get_resident_lifetime_info` now reports A/B/C role binding,
  source-version validity, current output domain, host/device currentness,
  device residency, plan fingerprint, workspace fingerprint, workspace
  identity/schedule/backend matches, device-id match, next-operation
  eligibility, and deterministic mismatch policy. The C++ wrapper exposes the
  same matrix-only and plan/workspace-bound checks. Benchmark resident-lifetime
  JSON now names the contract API, A/B/C role shape policy, workspace binding
  policy, and output-currentness policy. CPU unit tests cover matching resident
  A/workspace contracts, wrong-role rejection, mismatched-workspace rejection,
  zero-version C output reporting, invalid ABI rejection, and wrapper coverage.
  This is lifetime-contract infrastructure only; it does not add a speed claim,
  cache entry, default route, README row, or Linux/CDNA evidence.

June 6, 2026 rank-11 export-specialization cleanup:

- Rank 11 moved from the active queue to the completed-work archive as a
  covered duplicate. The active row still asked for exact-wide fixed-limb,
  compact-D2H, status-elision, prefix20, tree/CRT, and three-limb/four-limb
  Direct-HIP A/B coverage, but ranks 47 and 48 already closed that selector and
  export-bound implementation surface with schema-valid Windows `gfx1100`
  evidence. Future new export/reconstruction candidates now belong under rank
  33 with a fresh same-contract CPU/event/selector gate instead of a broad
  duplicate exact-wide export row.

June 6, 2026 rank-70 variance gate closeout:

- Rank 70 moved from the active queue to the completed-work archive after the
  variance gate became enforceable instead of advisory. `promotion_ledger.py`
  now has `--require-variance-gate`, which blocks otherwise promotable rows
  that do not have a matching `perf_variance_report.py` entry. Real
  `install_autotune_cache.py` writes and replacements now require a promotion
  ledger, and any supplied promotion ledger must include a ready variance gate;
  dry-runs remain available for cache JSON validation without writing. Tests
  cover missing variance rows, variance-blocked narrow wins, variance-ready
  installs, and the new no-ledger write rejection. This closes the cache/evidence
  promotion-control surface only; it does not add new performance claims.

June 6, 2026 rank-43 reuse contract closeout:

- Rank 43 moved from the active queue to the completed-work archive after a
  fresh Windows `gfx1100` release rerun under
  `temp/rank43-reuse-contract-release-20260606/`. The matrix produced 96
  schema-v4 captures across 16 release groups with seed `20260606`, three
  warmups, nine repeats, CPU, Direct HIP, runtime vector ALU, hipBLASLt, CK,
  and rocWMMA comparators where supported. `tools/reuse_contract_report.py`
  now consumes current `reuse_contract` metadata, normalizes reuse-only
  resident-lifetime and backend-specific export-selector fields for workload
  grouping, reports source-version stale rejection and explicit selector
  eligibility, and classifies 11/72 reuse rows as explicit-workload
  selector-ready with zero missing baselines. `benchmark_schema.py` passes over
  all 96 captures and `gpu_event_report.py --require-events` passes over every
  non-CPU capture. This closes explicit reuse workload policy only; no
  AUTO/cache entry, README headline claim, or Linux/CDNA claim changes.

June 6, 2026 queue-triage update:

- PR #12 is merged into `main`, so the active queue has been re-ranked against
  current evidence instead of older branch-local assumptions. Grouped dispatch,
  grouped many-small execution, export/reconstruction selection, RNS-chain
  lazy-output semantics, reuse lifetime policy, variance gates, exact-wide
  export variants, finite generic modulus mapping, shape-family shadow routing,
  counter/resource audits, and Linux/RDNA/CDNA target gates now lead the table.
  Completed exact-wide release-matrix, finite generic
  duplicate, and bounded reuse-contract classification rows moved to the
  completed-work archive; 8192 and multi-GPU work moved to the future/platform
  scout subsection.
- Queue promotion now requires the usual release/schema/correctness evidence
  plus golden-regression/perf-smoke observation for optimization lanes. The
  branch now has `tools/perf_variance_report.py`, which groups same-contract
  reruns by backend/kernel, reports within-capture and run-to-run timing
  spread, derives the speedup margin needed to clear observed repeatability
  noise, and can feed `tools/promotion_ledger.py --variance-report` so narrow
  wins block cache-promotion review. README, cache, or durable evidence-doc
  updates must pass claim validation before publication.
- Grouped-dispatch descriptor enforcement is now real internal runtime
  guardrail work instead of JSON-only metadata: grouped setup validates one
  shared Direct-HIP plan, unique workspace and resident A/B/C triplet ownership
  per task, same-shape descriptors, device-resident lifetimes, explicit stride
  policy, per-task source-version repack, device-current outputs, and bounded C
  source-version binding through `hip_direct_grouped_gemm_descriptor`. The
  current branch now also routes benchmark grouped pack, GEMM, and export
  phases through descriptor-backed Direct-HIP helpers that consume a backend
  same-shape grouped bucket plan, its single-bucket descriptor, and the
  internal grouped resource aggregate; A/B/C slabs, optional status storage,
  residue pointer tables, and A/B/C matrix vectors no longer get assembled
  independently inside each benchmark lane. Schema and report validation
  require the matching ownership, descriptor-reuse, stride,
  output-currentness, and lifetime policy fields. Debug smokes under
  `temp/perf-queue-grouped-contract-smoke/`,
  `temp/perf-queue-grouped-descriptor-smoke/`,
  `temp/grouped-resource-helper-smoke/`, and
  `temp/grouped-resource-helper-smoke-unsigned/`, plus the descriptor-backed
  execution smokes under `temp/grouped-execution-helper-smoke/` and
  `temp/grouped-execution-helper-smoke-unsigned/` and the bucket-plan smoke
  under `temp/grouped-bucket-plan-smoke/`, validate bounded,
  exact-wide signed/unsigned, and finite grouped captures as schema v4 with
  required Direct-HIP GPU events. The branch now exposes
  `rns8_get_grouped_dispatch_contract_info` plus a C++ wrapper so callers can
  inspect the same shape, stride, workspace, source-version, device-current
  output, status, checksum, lifetime, and resident execution boundaries for
  a created plan. The public C/C++ API now also exposes
  `rns8_gemm_rns_grouped` and `rns8_gemm_finite_u8_grouped` for narrow
  same-shape Direct-HIP resident grouped GEMM. Grouped host packing, final
  grouped export, AUTO routing, and broader generic dispatch remain active.
- Export/reconstruction selector work has its first reusable internal core
  surface: `src/core/export_plan.cpp` now owns the output layout, limb count,
  status policy, D2H policy, selected export kernel, tiled metadata, and
  all-zero tiled-output decisions that used to be private to
  `api_export.cpp`. Public export APIs still route through the same export
  behavior and mutable cache preparation point. Benchmark captures now record
  schema-validated selector metadata under `export_variant`: selector source,
  output layout, selector status policy, D2H policy, selected export kernel,
  tiled metadata requirement, and all-zero tiled-output state. Rank 48 remains
  active for compact/padded D2H variants, cache/stale-entry integration,
  final-output/chain selection, and release-size A/B evidence.
- The rank 47 Direct-HIP tree/CRT export follow-up narrowed the production
  route instead of promoting every selector candidate. The fixed-prefix export
  pass under `temp/rank47-exact-wide-prefix18-fixed-20260606/` showed local
  `gfx1100` 4-limb exact-wide Direct-HIP wins at 1024/2048 after the concrete
  prefix18 fixed-limb kernels and redundant export sync removal. The compact
  D2H selector under `temp/rank47-compact-d2h-20260606/` lost to the padded
  default. The tree/CRT captures under `temp/rank47-tree-crt-20260606/` and
  `temp/rank47-tree-crt-followup-20260606/` show signed prefix18 1024 and 2048
  wins, with roughly 2x export-event improvement versus fixed-prefix Garner
  export; 64 is favorable but too small to drive a default route, 128 and 512
  do not pass the end-to-end gate, unsigned tree/CRT is not worth routing, and
  prefix20 remains experimental. The resulting Direct-HIP default route is
  intentionally narrow: signed prefix18, four requested limbs, and large
  outputs only. This is Windows `gfx1100` evidence, not Linux/CDNA proof.
- Rank 45 is closed as an ABI/lifetime and descriptor-bucketing gate, not as a
  broad public grouped route. `rns8_get_grouped_dispatch_contract_info` reports
  caller-owned unique task triplets, call-return lifetime, explicit caller
  pack/export phases, generic public bucketing unavailable, and AUTO routing
  disabled through stable flag bits and policy strings. The internal
  Direct-HIP descriptor builder now also accepts same-contract shape buckets and
  rejects cross-bucket device mismatches, stale shape/semantic contracts,
  overlapping matrix/workspace ownership, non-contiguous task offsets, and
  task-count overflow. Registry, schema, and grouped reports understand
  `same_contract_shape_buckets`. Existing release artifacts under
  `temp/perf-queue-grouped-broader-release/` and
  `temp/perf-queue-grouped-exact128-release/` provide nine local Windows
  `gfx1100` grouped candidate wins with no missing baselines. Generic public
  pack/export, AUTO/cache routing, README claims, and Linux/CDNA validation
  remain outside this closed rank.
- Rank 8 is closed as the current many-small grouped workload matrix. A fresh
  combined `tools/many_small_grouped_report.py` run over
  `temp/perf-queue-grouped-broader-release/` and
  `temp/perf-queue-grouped-exact128-release/` writes
  `temp/rank8-many-small-closeout/many-small-grouped-report.json` and reports
  nine candidate wins, zero missing baselines, zero experimental rows, and zero
  deprioritized grouped rows. The closed matrix covers exact-wide
  signed/unsigned 64 and 128 group32, bounded-i64/u64 64 group32,
  bounded-i64 128 group64, bounded-u64 128x1x1024 group128, and finite-ring u8
  mod251 64 group32. Every grouped row has required GPU events, a valid grouped
  task descriptor, fastest-independent comparison, same-task-count Direct-HIP
  hostbatch comparison, matching hostbatch task count, and hostbatch checksum
  parity. This is benchmark-owned same-shape grouped workload evidence only,
  not a public generic grouped ABI, AUTO/cache entry, README headline claim, or
  Linux/CDNA claim.
- Rank 9 is closed as the current RNS-chain final-output matrix. A new
  release sweep under `temp/rank9-rns-chain-broader-release/` adds 24
  schema-v4 captures for bounded-i64/u64 and exact-wide signed/unsigned at 512
  and 1024, each with CPU final-output baseline, Direct-HIP resident chain, and
  same-backend Direct-HIP independent export/repack control. The regenerated
  `tools/rns_chain_report.py` output reports eight candidate final-output chain
  wins, zero missing baselines, zero experimental rows, zero deprioritized rows,
  and required GPU events for every Direct-HIP capture. The report now filters
  benchmark-sweep sidecars and excludes backend-specific export selector keys
  from the same-output contract key so CPU and GPU chain captures group
  correctly. This is local Windows `gfx1100` benchmark evidence only, not a
  public lazy-output API, AUTO/cache route, README headline claim, or
  Linux/CDNA claim.
- Rank 46 is closed as the current exact-wide residue-current/final-output
  chain matrix and API-draft lane. The release sweep under
  `temp/rank46-exact-wide-output-chain-release/` adds 16 schema-v4 captures for
  exact-wide signed/unsigned 512 and 1024 residue-current chain3/chain4 rows
  plus matching chain4 final-output CPU and Direct-HIP pairs. Combined with the
  rank-9 final-output chain3 captures,
  `tools/exact_wide_chain_report.py` writes
  `temp/rank46-exact-wide-output-chain-release/exact-wide-chain-report/` and
  reports eight paired residue-current/final-output rows, eight ready pairs,
  zero missing residue-current captures, zero missing final-output captures,
  zero missing CPU baselines, and required GPU events for every Direct-HIP
  capture. `docs/resident-output-api-draft.md` now records the benchmark-owned
  residue-current lifetime rules this evidence assumes. This is not a public
  resident-output API, AUTO/cache route, README headline claim, or Linux/CDNA
  claim.
- Pending local-validation and multi-GPU readiness tooling now exists without
  changing README/cache/evidence claims. `tools/pending_validation.py` now owns
  the reusable command-planning, capture discovery, review-report indexing,
  post-report execution, target matching, and summary classification core;
  `tools/gfx1100_pending_validation.py` is a thin Windows `gfx1100` wrapper
  over that shared driver. The June 6 closeout under
  `temp/gfx1100-pending-validation-20260606/` has 147 valid captures, four
  release review reports, zero duplicate backend groups, zero failed
  post-reports, and four local exact-wide fixed-limb export-selector wins:
  CK at 1024 signed/unsigned and hipBLASLt at 2048 signed/unsigned. These are
  evidence-only selector wins, not README/cache/default-route claims. The only
  remaining local validation debt in that root is three 4096 K-block reference
  baselines: exact-wide signed CPU, finite-u8 CPU, and wrap64 byte-limb.
  Linux multi-GPU readiness now has
  `tools/multigpu_shard_report.py`; `scripts/cdna_env_probe.sh` emits a
  `physical_devices` topology array, and `scripts/cdna_multigpu_smoke.sh`
  maps shard records by physical device id for lists such as `4,5,6,7`.
  This is infrastructure only: Windows `gfx1100` validation remains local
  RDNA3 evidence, and Linux/CDNA multi-GPU performance validation still waits
  for a real ROCm host.

June 4-5, 2026 updates:

- Cleanup consolidation has started as a guardrail/infrastructure lane, not a
  speedup claim. The branch now has a checked-in `metadata/` registry,
  generated Python/C++ constants, metadata-registry self-tests, a repo hygiene
  reporter, a compact golden regression runner, and docs claim validation for
  target-readiness and speedup wording. Scenario families now live under
  `benchmarks/scenarios/` with registry-backed review-mode and
  promotion-eligibility labels; schema validation has a package seam behind the
  compatibility CLI plus focused GPU-event, semantic-contract, reuse-timing,
  execution-mode, contract-metadata, helper/output-policy, and backend metadata
  modules/tests; benchmark argument parsing/backend selection, grouped
  descriptor contracts, support code, and large semantic lane bodies have moved
  behind helper translation units or include units; core output setup,
  benchmark exact-wide pack materialization, Direct-HIP output stamping,
  native-to-RNS bridge, and test-owned currentness transitions are
  helper-routed; workspace identity, schedule metadata, backend metadata,
  accelerator scratch, and prepack/resource teardown now flow through named
  helpers; export/reconstruction paths create an internal plan before touching
  the documented mutable export cache; HIP event/stream/pinned-staging/
  temporary-buffer ownership uses internal RAII wrappers including CK/rocWMMA
  event timing helpers; Direct-HIP host and kernel code are split by resource,
  pack, GEMM, common helper, and export concerns behind the same object labels;
  the hygiene report filters intentional helper/RAII implementation sites; and
  hardening now includes the portable non-Windows CPU ASan/UBSan preset, a
  Windows clang-cl CPU-only ASan/libFuzzer preset, deterministic fuzz harnesses,
  and a non-GUI `cdb.exe` triage helper. This is meant to reduce future
  metadata drift, benchmark growth, resource-cleanup risk, and durable
  documentation claim drift while preserving current public ABI, AUTO cache
  behavior, and reviewed evidence claims.
- Current branch reconciliation: the active queue below is updated through the
  branch-local helper/evidence work, PR #10 infrastructure merge, adaptive
  release rerun, bounded 2048 release review, finite-u8 2048 release review,
  generic finite-u8 2048 ring and field refreshes, exact-wide 64/128 and 2048
  refreshes, exact-wide chain/export specialization, Direct-HIP one-shot
  colpair routing, many-small baseline review, focused diagnostic event
  cleanup, benchmark-owned host API batching, bound-discovery no-promotion
  validation, bounded 4096 exploratory classification, non-bounded 4096
  exploratory classification, the full bounded A/B/A+B reuse-contract release
  matrix, bounded RNS-chain independent final-output/export-repack controls,
  and exact-wide RNS-chain independent final-output/export-repack controls.
  Completed and closed ranks have been moved to
  [performance-gain-completed-work.md](performance-gain-completed-work.md) so
  this active queue stays focused on execution. The archive currently holds 31
  ranks: infrastructure lanes 1, 2, 16, 23, 25, 26, 27, 28, 32, 34, 35, 37,
  40, and 41; current-claim validation lanes 4, 5, 17, 19, and 21; finite-u8
  hot-modulus 2048 rank 14; exact-wide baseline matrix rank 13; finite generic
  duplicate rank 15; bounded reuse-contract classification rank 18; host API
  batching rank 31; large-shape matrix rank 3; workspace arena allocation gate
  rank 57; and folded duplicate control-panel rows 7, 29, and 42. Rank 49 is
  now closed for the budgeted
  4096 release-reference gate, including
  exact-wide signed, exact-wide unsigned, and strict wrap64 reference coverage.
  Rank 3 is now closed for the large 2048/4096 non-reuse matrix after the
  promotion-ledger closeout installed eight eligible 4096 cache entries.
  Partially advanced lanes remain in the active table when they still own
  implementation, promotion, release-size validation, or public-contract work.
- HIP Graph replay work has moved from a generic queue item to a branch-local
  benchmark implementation lane. The first surface is intentionally narrow:
  Direct-HIP resident RNS GEMM chains with `--reuse-packed-inputs`,
  `--residue-chain-length > 1`, and `--next-op-hint rns-gemm`, where graph
  replay captures only the resident GEMM launches while A/B prepack and final
  checksum export remain outside the graph. The older tiny smoke under
  `temp/perf-work-queue/hip-graph-replay-smoke/` has since been superseded by
  the rank-30 release-size closeout above; graph replay remains benchmark-only
  evidence, not a public async API, cache entry, default route, README claim, or
  Linux/CDNA claim.
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
- The current skinny-GEMV release scenario also closes rank 17's selector-policy
  refresh for current claims. The 18 captures under
  `temp/perf-work-queue/skinny-gemv-current-release/` cover bounded-i64
  512x1x512, bounded-i64 256x1x4096, and bounded-u64 1024x1x1024 across CPU,
  Direct HIP, runtime vector ALU, hipBLASLt, CK, and rocWMMA. All three groups
  passed release review with required GPU events and no missing baselines, but
  Direct HIP won every reviewed shape, so no additional skinny-GEMV scenario
  cache entry is promoted. Keep the older vector GEMV microkernel win as
  explicit-backend evidence only, not AUTO selector policy.
- Finite-u8 generic prime/composite 512 was promoted for three local AUTO keys:
  ring 127 through rocWMMA, field 127 through CK, and ring 253 through
  hipBLASLt. A June 5 focused field-127 release rerun refreshed the CK field
  127 entry at 1289 us median end-to-end and confirmed hipBLASLt now has
  required GPU events for that contract, but loses to Direct HIP and CK.
- Finite-u8 generic ring 2048 now has CPU-backed release review instead of the
  previous GPU-only diagnostic captures. The June 5 current rerun installed two
  local rocWMMA cache entries: ring 127 at 3427 us median end-to-end, 1.59x
  faster than Direct HIP with CPU reference at 764044 us, and ring 253 at
  4856 us median end-to-end, 1.12x faster than Direct HIP with CPU reference at
  864589 us. The stale mod-127 hipBLASLt capture was event-incomplete and lost
  to Direct HIP; the focused timing-fallback rerun is event-complete, but it is
  still diagnostic until the full same-contract release group is intentionally
  rerun.
- Finite-u8 field refreshes also advanced rank 15 and cleared the stale
  field-251 512 event-debt note. Field 127 at 2048 now installs a CK cache entry
  at 3424 us median end-to-end, 1.57x faster than Direct HIP with CPU reference
  at 781139 us. Field 251 at 512 now installs a rocWMMA cache entry at 1241 us,
  1.05x faster than Direct HIP with CPU reference at 16019 us. The refreshed
  field-251 hipBLASLt capture has required GPU events but loses to Direct HIP
  and rocWMMA.
- Finite-u8 2048 hot-modulus work is now CPU-backed and release-reviewed for
  ring 251/255/256 plus field 251 after the finite hipBLASLt timing fallback.
  The post-fix rerun under
  `temp/perf-work-queue/finite-u8-2048-post-hipblaslt-event-release/` promotes
  hipBLASLt for all four hot 2048 contracts: ring 251 at 3244 us, ring 255 at
  2425 us, ring 256 at 3017 us, and field 251 at 3079 us. The reviewed temp
  cache wrote four entries, and `tools/install_autotune_cache.py` refreshed the
  default local cache by replacing two entries and adding two backend-keyed
  entries. A later 4096 promotion-ledger closeout added eight more eligible
  bounded, finite hot-modulus, and exact-wide entries; the cache now contains
  39 validated entries.
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
- The many-small release review now completes under
  `temp/perf-work-queue/many-small-current-release/` after the schema was
  tightened to distinguish prefix-9 native-input one-shot captures from smaller
  selected-prefix Direct-HIP resident fallbacks and after host-batch captures got
  distinct review backend identities. The same-commit review covers 61 captures:
  41 independent-call baselines plus 20 host API batch captures across seven
  many-small proxy groups. It has no missing required baselines, no duplicate
  backend records, compatible git/target metadata, and no cache entries
  promoted. The independent-call winners are CPU for bounded-i64 32,
  bounded-u64 64, and finite-u8 64; runtime vector ALU for bounded-i64 128;
  and Direct HIP for bounded-u64 128x1x1024 and exact-wide signed 64.
- The many-small diagnostic event holes are now closed at the focused-capture
  level. A Direct-HIP release capture under
  `temp/perf-work-queue/many-small-resident-oneshot-events/` validates the
  32x32x32 bounded-i64 selected-prefix one-shot fallback with schema v4 and
  required GPU events under the explicit
  `direct_hip_oneshot_resident_fallback_default_stream_operation_groups` scope.
  Branch-local grouped-dispatch evidence also moved from metadata-only to
  executable benchmark coverage: `rns8-bench --grouped-dispatch N` now runs
  same-shape persistent resident tasks through one shared plan with one
  matrix/workspace triplet per task, aggregate pack/GEMM/export timings,
  per-task checksums folded into the capture checksum, schema-v4
  `benchmark_grouped_dispatch_evidence` metadata, and downstream
  `many_small_grouped_report.py` classification. Tiny bounded-i64, finite-u8,
  exact-wide signed, and exact-wide unsigned Direct-HIP `gfx1100` smokes under
  `temp/grouped-dispatch-*.json` validate schema and required GPU events. A
  release-count exact-wide signed 64 group32 follow-up under
  `temp/perf-work-queue/many-small-grouped-dispatch-current/` first classified
  as a grouped-dispatch candidate win at 991.94 us per task. The current
  async exact-wide export-slab follow-up under
  `temp/perf-work-queue/many-small-grouped-dispatch-slab-current/` improves
  that to 792.66 us per task, 4.89x faster than the 3880 us independent
  Direct-HIP baseline and 2.40x faster than the same-backend hostbatch32 row,
  with checksum parity and required GPU events. The one-kernel grouped export
  follow-up under
  `temp/perf-work-queue/many-small-grouped-dispatch-kernel-current/` changes
  the contiguous exact-wide grouped export strategy to
  `device_grouped_exact_wide_export_kernel_batched_d2h`; the signed capture is
  schema/event-valid at 795.19 us per task, 4.88x faster than independent
  Direct HIP and 2.39x faster than hostbatch32, while export average drops
  from 5670 us to 1212 us. Its signed end-to-end median is effectively flat
  versus the 792.66 us slab median; its unsigned twin remains historical
  smoke evidence superseded by the focused unsigned closeout below. The
  grouped pack+export follow-up under
  `temp/perf-work-queue/many-small-grouped-pack-current/` then changes the
  strategy to
  `device_grouped_pack_and_exact_wide_export_kernels_batched_d2h`: compact A
  and B slabs are copied once per measured repeat, one grouped pack kernel is
  launched per operand, GEMM still loops through the resident tasks host-side,
  and the one-kernel grouped export path handles contiguous output. The signed
  capture is schema/event-valid at 228.06 us per task, 17.01x faster than
  independent Direct HIP, 8.34x faster than hostbatch32, and 3.49x faster than
  the previous grouped-export-only capture; aggregate host pack average drops
  from 16873 us to 778 us, and GPU pack-event average drops from 12815 us to
  461 us. Its unsigned twin remains historical smoke evidence superseded by
  the focused unsigned closeout below.
  The current grouped pack+GEMM+export follow-up under
  `temp/perf-work-queue/many-small-grouped-gemm-current/` changes the strategy
  to `device_grouped_pack_gemm_and_exact_wide_export_kernels_batched_d2h`: one
  grouped task-prefix Direct-HIP GEMM kernel group now covers all same-shape
  resident tasks. The signed capture is schema/event-valid at 66.47 us per
  task, 58.37x faster than independent Direct HIP, 28.63x faster than
  hostbatch32, and 3.43x faster than grouped pack+export; event median
  `rns_gemm` drops from 4701.94 us to 168.70 us. A focused unsigned closeout
  under `temp/perf-work-queue/many-small-grouped-unsigned-current/` adds the
  missing exact-wide unsigned 64 independent and hostbatch32 baselines for CPU
  and Direct HIP, tightens `many_small_grouped_report.py` so candidate wins
  require same-task-count hostbatch checksum parity, and classifies the
  Direct-HIP grouped pack+GEMM+export unsigned row as a candidate win at
  79.09 us per task, 18.70x faster than independent Direct HIP and 13.48x
  faster than Direct-HIP hostbatch32. This is still benchmark-owned evidence
  plumbing, not a public grouped API, generic descriptor queue, AUTO cache
  entry, or promotion claim.
  The grouped-dispatch descriptor-contract follow-up now emits a schema-v4
  `task_descriptor_contract` for grouped captures, validates task count,
  same-shape bucket, source-version, workspace, output-domain, checksum/status,
  and host-loop versus device-pointer/slab policy, and has a rebuilt
  `windows-release` exact-wide unsigned grouped smoke under
  `temp/perf-work-queue/grouped-descriptor-contract-smoke/` with required
  Direct-HIP GPU events. This closes the benchmark-side descriptor-contract
  evidence gap, but not the public/generic dispatcher.
  The bounded grouped pack+GEMM follow-up adds
  `device_grouped_pack_gemm_host_exports` for bounded-i64/u64 same-shape
  group32. Focused release controls under
  `temp/perf-work-queue/grouped-bounded-release-controls/` cover CPU and
  Direct-HIP independent baselines, Direct-HIP hostbatch32, and Direct-HIP
  grouped-dispatch rows for both bounded contracts.
  `many_small_grouped_report.py` classifies bounded-i64 64 group32 at
  544.56 us per task, 1.97x faster than the best independent CPU baseline and
  2.19x faster than Direct-HIP hostbatch32, and bounded-u64 64 group32 at
  532.47 us per task, 1.46x faster than the best independent CPU baseline and
  2.11x faster than Direct-HIP hostbatch32. Required Direct-HIP GPU events
  pass, but that first bounded follow-up still used per-task bounded CRT export
  and is superseded by the grouped export closeout below.
  The bounded grouped export closeout under
  `temp/perf-queue-grouped-bounded-export-release/` changes the bounded
  strategy to
  `device_grouped_pack_gemm_and_bounded_export_kernels_batched_d2h`: grouped
  bounded CRT export kernels now write one compact device output slab, followed
  by one compact D2H. The same release-control matrix classifies both bounded
  rows as stronger candidate wins: bounded-i64 64 group32 is 53.625 us per
  task, 19.97x faster than the best independent CPU baseline and 23.32x faster
  than Direct-HIP hostbatch32; bounded-u64 64 group32 is 53.25 us per task,
  10.95x faster than the best independent CPU baseline and 22.95x faster than
  Direct-HIP hostbatch32. Required Direct-HIP GPU events and same-task-count
  checksum parity pass. This closes the bounded 64 group32 per-task export
  bottleneck, but it remains benchmark-owned same-shape evidence, not
  AUTO/public routing or a Linux/Instinct claim.
  A hipBLASLt release capture under
  `temp/perf-work-queue/many-small-hipblaslt-finite-events/` validates the
  finite ring-251 64x64x64 diagnostic with required pack, matmul, reduce, and
  export events. This is event/tooling cleanup for non-promoted diagnostics,
  not a grouped execution implementation or speedup claim.
- Host API batching now has both benchmark/schema plumbing and release-reviewed
  same-commit evidence. `tools/host_api_batch_report.py` compares batch
  per-task medians against same-backend independent calls and the fastest
  independent-call baseline for the same contract. The report under
  `temp/perf-work-queue/many-small-current-release/host_api_batch_report.md`
  covers 20 host-batch comparisons with required GPU events for every GPU
  host-batch capture. Direct-HIP exact-wide signed 64 hostbatch32 is the only
  full workload win: 1903 us per task versus the 3880 us Direct-HIP independent
  baseline, 2.04x faster. The other 19 host-batch candidates are deprioritized.
  This closes rank 31 and advances rank 8, but it is still benchmark-only
  workload evidence rather than a public grouped API or AUTO cache promotion.
- Reuse/prepack promotion now has an explicit setup-amortized workload gate.
  `tools/reuse_contract_report.py` groups reuse captures against their
  non-reuse workload contracts, computes setup-inclusive per-repeat time,
  same-backend and fastest-non-reuse speedups, required break-even repeat
  counts, event availability, and source-identity metadata. The CPU-backed
  2048 large-release matrix report under
  `temp/perf-work-queue/large-release-validation-2048-current/` classifies 4
  of 12 repeated-B captures as setup-inclusive workload candidates:
  hipBLASLt bounded-i64, CK bounded-u64, hipBLASLt bounded-u64, and rocWMMA
  bounded-u64. The bounded 4096 exploratory report under
  `temp/perf-work-queue/large-4096-bounded-exploratory/` classifies 3 of 16
  repeated-B captures as workload candidates: CK bounded-i64 2048, hipBLASLt
  bounded-i64 2048, and hipBLASLt bounded-i64 4096. All remain explicit
  workload-contract candidates rather than AUTO cache entries.
- Bound-discovery/proof-mask setup-inclusive validation now has its own
  `bound-discovery` scenario family and comparison report. The June 5, 2026
  release matrix under
  `temp/perf-work-queue/bound-discovery-current-release/` captured 51
  schema-valid records across bounded-i64 256/1024 adaptive-band workloads and
  a bounded-u64 512x1024 adaptive-band workload, comparing static global bounds,
  global input-scan bounds, and per-tile proof-mask bounds across CPU, Direct
  HIP, runtime vector ALU, hipBLASLt where supported, CK, and rocWMMA. The
  generic release review has nine groups, no missing required baselines, no
  duplicate backends, compatible git/target metadata, and required GPU events
  for every non-CPU capture. `tools/bound_discovery_report.py` adds
  `global_bound_scan` or `tile_bound_scan` setup cost to median end-to-end
  timing and deprioritizes all 33 discovery/proof candidates. hipBLASLt global
  input-scan improves over its own 256 bounded-i64 static baseline and CK
  global input-scan improves over its own rectangular bounded-u64 static
  baseline, but both lose to the fastest static workload baseline after setup
  cost. Proof-mask per-tile candidates are event-visible and correct, but the
  exact tile-bound prepass dominates end-to-end timing. Rank 21 is closed as
  validated no-promotion evidence.
- June 5 queue expansion adds ranks 43-63 for the next architecture-level work:
  reuse contract policy, resident matrix lifetime, device grouped execution,
  exact-wide export-bound work, CRT/reconstruction fusion, 4096 validation
  budgeting, hipBLASLt bounded-i64 1024 A/B work, Direct-HIP resident redesign,
  modulus-set and residue-count search, adaptive grouped scheduling, streaming
  overlap, tile-shape autotuning, workspace arenas, HIP graphs, shape-family
  AUTO shadow mode, promotion ledger tooling, counter-driven occupancy audits,
  Linux/RDNA/CDNA validation gates, verification amortization, and a real
  FHE/lattice workload suite. These are queue additions only; none are promoted
  claims until they pass the release-review gates below.
- The follow-up queue audit folds duplicate active rows into their stronger
  owners: hipBLASLt bounded-i64 1024 tuning moves under rank 50, adaptive
  grouped scheduling moves under rank 54, and non-Windows platform validation
  moves under rank 62. It also adds ranks 68-77 for strict wrap64 v4 tuning,
  CPU small-shape fallbacks, variance/regression gates, 8192 scouting,
  vector/native-to-RNS fused producer-consumer work, finite input-distribution
  matrices, split-K/K-block large-shape variants, result-cache research,
  multi-GPU platform work, and real layout implementation search.
- Branch-local Starfoundry closeout now gives the full rank 43-63 queue block
  executable evidence surfaces: schema-v4 objects for
  reuse/resident-lifetime/output/export/reconstruction/modulus/tile/grouped/
  adaptive-scheduler/workspace-arena/streaming-overlap/graph/release-gate/
  verification/proxy metadata, `rns8-bench` passthrough flags, `rns8-inspect
  --selector-shadow`, disabled-by-default sweep families, and temp-only report
  tools. This is optimizer-enablement scaffolding plus benchmark-only
  prototype surface; it does not claim a speedup, install cache entries,
  change public ABI, or route AUTO on shape-family recommendations.
- Large-shape validation now has a dedicated `large-release-validation`
  scenario family. It emits the missing CPU/direct/vector/accelerator
  comparator matrix for 2048 bounded i64/u64, setup-contract reuse-B 2048,
  exact-wide signed/unsigned 2048, finite-u8 hot-modulus 2048, and wrap64 2048.
  This closes the executable validation surface for the 2048 side of the large
  matrix. Bounded, finite-u8 hot-modulus, strict wrap64, and exact-wide 2048
  release captures have now been run; repeated-B remains explicit
  workload-contract evidence. The later rank-3 closeout completed the eligible
  4096 non-reuse cache promotion separately.
- The bounded 4096 large-shape lane now has same-commit GPU-only exploratory
  classification under
  `temp/perf-work-queue/large-4096-bounded-exploratory/`. The 32 release-mode
  captures cover bounded i64/u64 at 2048 and 4096, non-reuse and repeated-B,
  across Direct HIP, hipBLASLt, CK, and rocWMMA, with required GPU events for
  every capture. The generic review intentionally reports missing required
  baselines because CPU and runtime vector captures were not included, so this
  is not AUTO/cache evidence. The 4096 best paths are hipBLASLt: bounded i64
  non-reuse at 52259 us versus Direct HIP at 140393 us, bounded i64 repeated-B
  at 40108 us versus Direct HIP at 132674 us, bounded u64 non-reuse at
  47467 us versus Direct HIP at 191947 us, and bounded u64 repeated-B at
  45843 us versus Direct HIP at 133831 us. Same-commit best-path 4096/2048
  scaling is 4.11x for bounded i64 non-reuse, 4.22x for bounded i64 repeated-B,
  3.76x for bounded u64 non-reuse, and 3.43x for bounded u64 repeated-B.
  Keep these as throughput-classification signals until CPU/vector baselines or
  an explicit 4096 promotion budget are added.
- The first rank-3/rank-49 follow-up now has actual budgeted 4096 evidence
  under `temp/perf-work-queue/large-4096-budgeted-release-current-v2/`.
  Bounded i64/u64 4096 completed CPU, Direct HIP, runtime vector ALU,
  hipBLASLt, CK, and rocWMMA release captures; hipBLASLt is fastest in both
  groups at 35303 us median end-to-end for bounded i64 and 37543 us for bounded
  u64. Finite hot-modulus 4096 also completed CPU, Direct HIP, hipBLASLt, CK,
  and rocWMMA groups for field-251 and ring 251/255/256; hipBLASLt wins
  field-251, ring-251, and ring-256, while CK wins ring-255. The exact-wide
  signed 4096 row completed a same-commit CPU/GPU closeout with required
  events, with hipBLASLt fastest at 176943 us versus Direct HIP at 639360 us.
  Strict wrap64 4096 has an event-valid Direct-HIP row at 295657 us. A
  follow-up exact-wide unsigned
  budgeted group under
  `temp/perf-work-queue/large-4096-unsigned-budgeted-release-current/`
  completed CPU, Direct HIP, hipBLASLt, CK, and rocWMMA rows with required GPU
  events; hipBLASLt is fastest at 162382 us versus Direct HIP at 614116 us and
  CPU at 105462000 us, with matching checksum `9643325300233475427`. The
  regenerated combined release-gate report under
  `temp/perf-work-queue/large-4096-combined-release-gate-current/` ingests
  both `.failed.json` timeout records alongside 44 completed captures across 9
  groups and 46 total inputs; historical timeout rows remain visible but no
  longer count as active group blockers once the required backend has a valid
  release capture. A follow-up one-pass reference run under
  `temp/perf-work-queue/large-4096-reference-onepass-current/` completed full
  exact-wide signed `cpu-reference` and strict wrap64 `wrap64-byte-limb`
  captures with `warmups=0` and `repeats=1`; a later release-reference run
  completed the same two required references with `warmups=3` and `repeats=9`.
  The exact-wide signed release reference recorded 113755000 us median
  end-to-end with checksum `5508849193854467465`; the wrap64 byte-limb release
  reference recorded 102905000 us with checksum `13518998852724169131`. The
  combined report has no unattempted required baselines, no missing valid
  required baselines, and no missing release-review required baselines for the
  included 4096 groups. A follow-up promotion-ledger closeout under
  `temp/perf-work-queue/large-4096-cache-closeout-current/` wrote and
  installed eight eligible 4096 non-reuse cache entries: bounded i64/u64,
  finite field-251, finite ring 251/255/256, and exact-wide signed/unsigned.
  The installed ledger reports 39 total cache entries and zero blockers for
  those eight captures. Strict wrap64 remains a Direct-HIP correctness path
  rather than an AUTO cache entry, and repeated-B remains an explicit workload
  contract.
- The non-bounded 4096 large-shape lane now also has same-commit GPU-only
  exploratory classification under
  `temp/perf-work-queue/large-4096-nonbounded-exploratory/`. The 50
  release-mode captures cover exact-wide signed/unsigned, finite-u8 ring
  251/255/256, finite-u8 field 251, and strict wrap64 at 2048 and 4096 across
  the available GPU backend set for each contract. The generic review
  intentionally reports missing required baselines because CPU and runtime
  vector captures were not included. The original required-event review was
  mixed: 46 of 50 captures were event-complete, all event-valid 4096 winners
  had required GPU events, and four non-promoted hipBLASLt finite captures
  missed the residue-reduce event label. The backend timing fallback added on
  this branch reran every stale finite hipBLASLt reduce-label miss under
  `temp/perf-work-queue/finite-hipblaslt-event-reruns-all/`; all 17 focused
  reruns are schema-valid and pass `gpu_event_report.py --require-events`.
  Event-valid exploratory 4096 winners are CK for exact-wide signed and
  unsigned,
  hipBLASLt for finite field-251 plus finite ring-251 and ring-256, CK for
  finite ring-255, and Direct HIP for strict wrap64. Keep this as throughput
  and scaling evidence only; it does not close CPU/reference release review,
  vector comparison, or public 4096 promotion. The later budgeted exact-wide
  unsigned group supersedes the exploratory unsigned row for reviewed local
  decision-making.
- The first bounded 2048 slice of that validation matrix is now captured and
  release-reviewed under
  `temp/perf-work-queue/large-release-validation-2048-current-bounded-review/`.
  The 24 reviewed captures cover bounded i64/u64 baseline and repeated-B
  2048x2048x2048 cases across CPU, Direct HIP, runtime vector ALU, hipBLASLt,
  CK, and rocWMMA with required GPU events. The non-reuse reviewed cache
  winners are CK for bounded-i64 2048 at 14220 us and rocWMMA for bounded-u64
  2048 at 15128 us; both are now installed in the local reviewed cache.
  Repeated-B remains workload-contract evidence rather than
  AUTO promotion: hipBLASLt wins its own repeated-B backend comparison at 9482
  us for bounded-i64 and 8727 us for bounded-u64, but `prepacked_reuse` stays
  non-autotune-promotable until setup identity, lifetime, and break-even policy
  become explicit.
- The strict wrap64 2048 slice of `large-release-validation` is complete after
  CPU reference-path cleanup. The two captures under
  `temp/perf-work-queue/large-release-validation-2048-wrap64-current/` cover
  same-contract byte-limb CPU and Direct HIP v4 at 2048x2048x2048 with release
  settings, schema-valid captures, matching checksums, compatible
  target/toolchain/commit metadata, and required Direct-HIP GPU events. Direct
  HIP v4 measured 58331 us median end-to-end versus 13423400 us for the CPU
  byte-limb reference. This is current Direct-HIP correctness-path evidence, not
  an AUTO cache entry.
- The finite-u8 2048 hot-modulus slice of `large-release-validation` is also
  complete. The original 20 captures under
  `temp/perf-work-queue/large-release-validation-2048-finite-current/` covered
  CPU, Direct HIP, hipBLASLt, CK, and rocWMMA for ring 251/255/256 and field
  251. After the hipBLASLt finite event-timing fallback, the post-fix rerun
  under `temp/perf-work-queue/finite-u8-2048-post-hipblaslt-event-release/`
  supersedes those hot 2048 cache decisions: hipBLASLt wins ring 251 at
  3244 us, ring 255 at 2425 us, ring 256 at 3017 us, and field 251 at 3079 us,
  with required GPU events, CPU reference baselines, and refreshed local cache
  entries.
- The exact-wide 64/128 current-v2 refresh is complete. The 20 captures under
  `temp/perf-work-queue/exact-wide-small-v2-release/` cover signed/unsigned
  exact-wide 64 and 128 across CPU, Direct HIP, hipBLASLt, CK, and rocWMMA.
  Only exact-wide unsigned 64 promoted: hipBLASLt at 4611 us, 1.67x faster than
  Direct HIP, with required GPU events and a local cache entry installed.
  Signed 64, signed 128, and unsigned 128 stay on Direct HIP.
- The exact-wide 2048 slice of `large-release-validation` is complete after
  CPU reference-path cleanup made chunked release execution practical. The 10
  captures under
  `temp/perf-work-queue/large-release-validation-2048-exact-wide-current/`
  cover signed and unsigned exact-wide 2048 across CPU, Direct HIP, hipBLASLt,
  CK, and rocWMMA with schema-valid captures, compatible target/toolchain/commit
  metadata, CPU/direct baselines, and required GPU events for GPU captures.
  hipBLASLt promoted and was installed for signed 2048 at 59074 us, 2.23x
  faster than Direct HIP and 322.3x faster than CPU, and unsigned 2048 at
  40985 us, 3.04x faster than Direct HIP and 384.1x faster than CPU. Both
  promoted captures are export-bound after GEMM acceleration, so fixed-width
  export specialization and lazy residue-current workflows stay near the front
  of the queue.
- Exact-wide signed three-limb export now uses the full-width status-elided
  Direct-HIP path. Prefix-20 reconstruction is 155 bits wide, so signed centered
  outputs fit in three 64-bit limbs; the runtime, benchmark metadata, and
  schema now treat signed limb counts 3..32 the same way unsigned 3..32 already
  behaved for range-status elision. The focused 2048 Direct-HIP A/B captures
  under `temp/exact-wide-signed-2048-limbs3-direct.json` and
  `temp/exact-wide-signed-2048-limbs4-direct.json` are schema/event-valid and
  show the three-limb output contract at 190940 us median end-to-end versus
  194115 us for four limbs in that run. This is useful export specialization
  evidence, not a reviewed cache entry or a replacement for four-limb output
  when callers request four limbs.
- Exact-wide signed RNS-chain lazy-output evidence is now captured for the
  Direct-HIP correctness backend. The release-mode chain captures under
  `temp/perf-work-queue/exact-wide-rns-chain-direct-current/` cover
  128x128x128 chain length 3 with residue-current output, no per-repeat CRT
  export, schema-v4 validation, and required Direct-HIP GPU events. The
  per-repeat repack chain measured 6102 us median end-to-end with 4888 us pack
  and 2647 us RNS GEMM timing, while the explicit reusable-B chain measured
  1201 us median end-to-end after an 11718 us setup cost. A rough workload
  comparison against three independent 128 host-export calls is favorable for
  the residue-current chain, especially reusable-B, but this remains
  workload-contract evidence: it is not a same-output AUTO/cache entry, and
  broader promotion still needs final-output contract coverage plus explicit
  reuse lifetime and break-even policy.
- Branch-local final-output RNS-chain benchmark coverage now exists for the
  same-output contract. `rns8-bench --residue-chain-length N
  --residue-chain-final-export` records `residue_chain_final_host_export`
  captures where each measured repeat performs the chained resident RNS GEMMs
  and then exports the final logical host/exact-wide output. Schema v4 now
  distinguishes this from `residue_current_rns_chain`, and
  `tools/benchmark_sweep.py --scenario rns-chain-final-output` emits bounded
  i64/u64 and exact-wide signed/unsigned chain candidates with
  `--next-op-hint final-export`. Tiny Windows `gfx1100` Direct-HIP smokes under
  `temp/residue-chain-final-output/` are schema-valid and have required GPU
  events for bounded-i64 CRT export and exact-wide signed limb export. This is
  contract/tooling proof, not a release-size performance promotion.
- The release-style final-output chain control report now validates both
  resident final-output chains and bounded independent export/repack controls.
  `tools/rns_chain_report.py` accepts `residue_chain_final_host_export` and
  `residue_chain_independent_final_host_export`, groups the same requested
  final-output contract, requires CPU baseline and GPU events, adds
  reusable-input setup cost back into per-repeat medians, and reports
  break-even/deprioritization decisions. Focused Windows `gfx1100` captures
  under `temp/perf-work-queue/rns-chain-final-output-current/` cover
  bounded-i64 128 and exact-wide signed 128 with CPU, Direct HIP, and
  Direct-HIP reusable B. Direct HIP wins the same final-output chain contract
  versus CPU at 1614 us versus 15745 us for bounded-i64 and 1358 us versus
  41656 us for exact-wide signed, with required GPU events. Reusable-B loses
  the setup-inclusive same-backend gate for both rows: 3002 us versus 1614 us
  for bounded-i64 and 3161 us versus 1358 us for exact-wide signed.
- Bounded same-output export/repack controls now exist for the RNS-chain path.
  The focused release-style captures under
  `temp/perf-work-queue/rns-chain-independent-final-output-current/` add
  schema/event-valid CPU and Direct-HIP resident and independent controls for
  bounded-i64 128 chain3 and bounded-u64 256 chain3. The same report classifies
  Direct-HIP resident final-output chains as candidate wins against their
  same-backend independent export/repack controls: bounded-i64 128 is 1805 us
  versus 3324 us, 1.84x faster, and bounded-u64 256 is 2240 us versus 4387 us,
  1.96x faster. Exact-wide independent export/repack remains open because it
  needs an explicit limb-output import/repack contract before the comparison is
  fair.
- Direct-HIP public bounded-i64 one-shot large shapes now use the existing
  prefix-9 colpair native-input kernel. The focused before/after captures under
  `temp/perf-work-queue/direct-hip-i64-oneshot-colpair/` compare the prior v1
  one-shot route against `direct_hip_prefix9_native_input_colpair_grouped_rns_gemm_v2`
  for 512x512x512 with fixed prefix 9, three warmups, nine repeats, and seed
  `20260605`. The new route keeps the checksum identical and improves median
  one-shot end-to-end time from 9368 us to 3048 us, a 3.07x win for that public
  one-shot contract. The same-shape persistent resident Direct-HIP capture is
  still faster at 2126 us median end-to-end, so this advances one-shot and
  prefix-9 fusion work only; it is not an AUTO/cache promotion or a reason to
  prefer transient one-shot over resident matrix reuse.
- Direct-HIP resident selected-prefix colpair was also attempted for large
  bounded resident matrices, then deliberately not routed. Captures under
  `temp/perf-work-queue/direct-hip-resident-colpair-current/` show that the
  experimental resident colpair kernel could improve the 512 bounded-i64
  selected-prefix GEMM median in one rerun, but it introduced unstable GEMM
  outliers and failed the setup-inclusive end-to-end gate: the rerun baseline
  `direct_hip_tiled_active_prefix_rns_gemm_v2` measured 2434 us median
  end-to-end, while the colpair attempt measured 4010 us. Fixed-prefix 9 and
  1024 captures were kept on their existing kernels. Treat resident colpair as
  deprioritized until a different design wins end-to-end, not as completed rank
  6 or rank 10 routing work.
- The bounded-i64 Direct-HIP 512 tuning pass is closed as no-new-route
  evidence. The selected-prefix grouped launcher experiment under
  `temp/perf-work-queue/direct-hip-512-rank6-selected-prefix-grouped/` was
  schema/event-valid but did not pass the end-to-end gate. A paired export-sync
  A/B under `temp/perf-work-queue/direct-hip-512-rank6-paired-export-sync-baseline/`
  and `temp/perf-work-queue/direct-hip-512-rank6-paired-export-nosync-candidate/`
  also rejected removing the bounded export kernel synchronization: the no-sync
  candidate lost 2321 us versus 1655 us median end-to-end and 467.55 us versus
  234.48 us `crt_export` event median. The older reviewed 1851 us 512
  front-page snapshot remains the durable claim. Future 512 work should move to
  active resident-layout, pack/export, and reconstruction ranks with paired A/B
  proof, not another isolated retry of these rejected routes.
- `tools/benchmark_sweep.py` now has chunk/resume controls for expensive
  matrices: `--skip-existing` reuses schema-valid existing captures and
  `--max-new-captures` caps how many new captures a pass may execute before
  reviewing the accumulated set.
- Adaptive bounded current-v2 validation is no longer blocked at the capture
  plumbing layer. The branch fixes compact per-tile CPU storage acceptance,
  one-shot compact schedule allocation, CK/rocWMMA zero-row/column-product
  schedule flag handling, runtime vector comparator schema policy, and
  `adaptive-bands` CPU comparator generation. One-repeat smoke captures under
  `temp/perf-work-queue/adaptive-current-v2-smoke/` are schema-valid for CPU,
  Direct HIP, runtime vector ALU, CK, and rocWMMA at bounded-i64
  256x256x512, with required GPU events for the GPU records.
- The current-v2 adaptive-bands release rerun is complete. The corrected review
  grouping report under
  `temp/perf-work-queue/adaptive-current-v2-release-reviewed/` has three
  release groups, no missing required baselines, no duplicate backends, complete
  target/toolchain metadata, schema-valid captures, and required GPU events.
  Direct HIP is fastest for bounded-i64 256x256x512 at 1848 us, bounded-i64
  1024x1024x1024 at 4937 us, and bounded-u64 512x1024x512 at 4224 us. No
  adaptive accelerator cache entry is promoted; the old rocWMMA tiled-v1
  adaptive cache identity is historical.
- The branch now has a full bounded reuse-contract release matrix under
  `temp/perf-work-queue/reuse-contract-release-current/`. The new
  `reuse-contract` scenario family covers non-reuse, stable-A, stable-B, and
  stable-A+B at 1024 and 2048 for bounded i64/u64 across CPU, Direct HIP,
  runtime vector ALU, hipBLASLt, CK, and rocWMMA. The accumulated review has 96
  schema-valid captures, 16 same-contract groups, no missing baselines, no
  duplicate backend records, and required GPU events for non-CPU records.
  `tools/reuse_contract_report.py` classifies 17/72 reuse comparisons as
  setup-inclusive workload candidates, 43 as deprioritized, 12 as experimental,
  and zero as missing a baseline. For hipBLASLt specifically, 2048 stable-A and
  stable-A+B are explicit workload candidates for bounded i64 and u64; 1024
  stable-A and stable-A+B are deprioritized against the fastest non-reuse
  workload baseline. The bounded-u64 2048 full A+B row is the strongest
  hipBLASLt reuse signal in this matrix at 9198 us setup-inclusive per repeat,
  3.99x faster than same-backend non-reuse and 2.35x faster than the same-run
  fastest non-reuse baseline. This is still a reuse-workload contract, not an
  AUTO cache entry.
- Rank 52 finite generic modulus family map is closed as local Windows
  `gfx1100` non-promoting evidence. The June 6 release sweep under
  `temp/rank52-finite-modulus-map-release-20260606/` produced 200 captures:
  ring moduli 127, 241, 243, 251, 253, 255, and 256 plus field moduli 127,
  241, and 251 at 128, 512, 1024, and 2048 across CPU, Direct HIP, hipBLASLt,
  CK, and rocWMMA. `tools/finite_modulus_map_report.py --require-complete-map`
  reports 40/40 ready groups, zero missing expected groups, zero missing
  backends, zero missing GPU events, and zero non-release-ready groups. The
  generic sweep review now carries scenario metadata on captures and reports
  zero autotune-promotable entries because the scenario scope is
  `non_promoting_modulus_map`. Winner distribution is CPU reference for all
  128 groups, mixed Direct HIP/CK/rocWMMA/hipBLASLt at 512 and 1024, and
  hipBLASLt for all 2048 groups. No README/cache/default-route/Linux/CDNA claim
  changes from this map.
