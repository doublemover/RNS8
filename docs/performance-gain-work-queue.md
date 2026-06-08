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

Use this table as the working control panel. The next implementation chunks
should pull from this ranked list first.

Evidence sources for current promotion state are
[performance-wins.md](performance-wins.md),
[reviewed-local-evidence.md](reviewed-local-evidence.md),
[roadmap-status.md](roadmap-status.md), and the README's
[current local performance snapshot](../README.md#exactness-and-performance).
Completed and closed queue ranks are archived in
[performance-gain-completed-work.md](performance-gain-completed-work.md).
The former detailed backlog/research-notes material lives in
[performance-gain-research-backlog.md](performance-gain-research-backlog.md);
dated execution updates and non-active disposition tables live in
[performance-gain-work-log.md](performance-gain-work-log.md) and
[performance-gain-queue-dispositions.md](performance-gain-queue-dispositions.md).
The active table below now contains 38 ranks. Rank IDs are historical/stable
references; row order is the current execution priority. Non-active material
lives outside this file so the control panel stays execution-focused.

These ranks were opened after the June 8, 2026 MI300X repeated-workload review
showed clean correctness but unresolved performance gaps: Direct HIP remains
the fastest production route for skinny GEMV and exact-wide 512 exports,
rocWMMA has one bounded-i64 1024 non-reuse win, prepacked-B reuse still loses
setup-inclusively, graph replay has no production win, and accelerator rows
need clearer loss-phase, setup, and ISA evidence. The latest summary surface is
`tools/benchmark_sweep_failure_summary.py`; do not replace it with ad hoc JSON
inspection commands.

### Rank 79 - CDNA Review Integrity And Loss-Phase Summaries

- Priority: P0.
- Target: all Linux ROCm/CDNA release reviews.
- Problem: review output now reports fastest production and accelerator
  routes, but optimizer triage still requires reading many candidate rows to
  see whether a workload is pack-bound, GEMM-bound, CRT/export-bound,
  launch/API-bound, setup-bound, or missing evidence.
- Implementation scope: extend `tools/benchmark_sweep_failure_summary.py` and
  review reports with scenario/backend/semantic/shape-family loss aggregation,
  phase-ratio summaries, setup-inclusive reuse/graph details, direct-HIP winner
  reporting, and a compact "next work" section.
- Local progress: `benchmark_sweep_failure_summary.py` now prints route counts,
  Direct-HIP production wins, loss-phase aggregation by backend/semantic/
  shape/scenario, setup-inclusive reuse/graph details, next-work rows, and
  compact per-phase ratio diagnostics for route and actionable-candidate lines.
  It also has dedicated matrix-core route rows for MFMA, SMFMAC, WMMA, and
  SWMMAC candidates so CDNA/RDNA matrix-core evidence is visible without
  digging through review JSON. Runtime prepack-cache summary rows include the
  source version beside production eligibility, scope, hash, and byte counts.
- Required evidence: Python tests for summary formatting, synthetic review
  fixtures for every blocker family, and one VM refresh proving zero failed
  captures, zero comparable checksum mismatches, and no hidden missing-baseline
  groups.
- Do not promote: evidence-only rows, graph-only rows, smoke captures,
  internal candidates, or raw per-repeat reuse timings.
- Supported VM shortcut: `benchmark_sweep.py --scenario release-candidates`
  runs only scenario rows whose promotion scope is `release_review_candidate`;
  use `--scenario all` only when deliberately collecting exhaustive evidence
  and accepting evidence-only blocker noise.

### Rank 80 - CDNA Skinny GEMV Direct-HIP Route

- Priority: P1.
- Target: `skinny-gemv` on MI300X first, then RDNA3.
- Problem: `N=1` shapes are direct-HIP production winners, while rocWMMA/MFMA
  tiles lose because matrix-core tile setup and packing do not match GEMV.
- Implementation scope: add row-parallel Direct-HIP GEMV kernels for bounded
  i64/u64, optimized for `N=1` and small-N; use vectorized/coalesced K loads,
  fixed-prefix generated reducers, deterministic status handling, and separate
  pack/GEMV/export event labels.
- Local implementation status: implemented for fixed-prefix resident-RNS
  bounded i64/u64 `N=1` captures as
  `direct_hip_prefix9_rns_gemv_n1_i64_v1` and
  `direct_hip_prefix9_rns_gemv_n1_u64_v1`; local Windows HIP smokes selected
  the new `direct_hip_skinny_gemv_n1_resident_rns` route and schema-validated.
  Public Direct-HIP plan metadata now reports the same GEMV selected kernel,
  epilogue, and workspace for bounded fixed-prefix `N=1` plans instead of the
  generic grouped GEMM identity.
- Required evidence: release captures for `512x1x512`, `256x1x4096`, and
  `1024x1x1024` against CPU anchor, current tiled Direct HIP, vector-ALU
  where applicable, and rocWMMA; schema must report a GEMV selected kernel and
  phase timings.
- Promotion rule: promote only if setup-inclusive end-to-end beats the current
  Direct-HIP production route for the same contract.

### Rank 81 - Vector-ALU GEMV And Native Exact Small-N Baseline

- Priority: P1.
- Target: bounded i64/u64 native-output GEMV and small-N workloads.
- Problem: vector-ALU has known skinny advantages on some local shapes, but
  its GEMV path needs to be explicit and comparable instead of incidental.
- Implementation scope: add `hip_vector_alu_*_gemv_n1` routing and benchmark
  metadata for native exact i64/u64 GEMV, with exact status checks and
  explicit output-domain metadata.
- Local implementation status: implemented for `N=1,K>=4096` native exact
  vector-ALU captures with
  `public_runtime_vector_alu_gemv_n1_native_buffers` /
  `benchmark_owned_vector_alu_gemv_n1_native_buffers`, selected
  `hip_vector_alu_*_gemv_n1_exact_192b_v1` kernels, GEMV-specific GPU event
  labels, native output-domain packing metadata, and local Windows HIP
  signed/unsigned schema-validated smokes.
- Required evidence: `skinny-gemv` release groups with CPU, Direct HIP,
  vector-ALU, and accelerator comparators; vector-ALU rows must not be compared
  against exact-wide, finite-u8, or wrap64 contracts.
- Promotion rule: vector-ALU may win production only for bounded native-output
  contracts with exact same-contract baselines.

### Rank 82 - Exact-Wide Fused Device CRT Export

- Priority: P1.
- Target: exact-wide signed/unsigned final host-output captures.
- Problem: accelerator GEMM often loses because final CRT/export dominates
  after GEMM acceleration. MI300X exact-wide 512 still picks Direct HIP as
  production winner.
- Implementation scope: add fused or immediately chained device CRT/export
  routes for exact-wide signed/unsigned, vectorized Garner/CRT constants,
  fixed-prefix specialization, per-cell limb export, deterministic first-error
  reporting, and selected export-kernel metadata.
- Local implementation status: exact-wide Direct HIP device export routes and
  fixed-prefix18/fixed-prefix20 selector kernels are implemented. Export kernel
  identities are now registry-backed through `metadata/kernels.yaml`, generated
  into Python/C++ metadata constants, and schema-checked across
  `export_variant.selected_kernel`, `exact_output_contract.kernel_identity`,
  and `reconstruction_variant.kernel_identity`. Local Windows HIP fixed-prefix18
  signed/unsigned smokes selected the matching device export kernels and passed
  schema validation.
- Required evidence: `exact-wide-export`, `export-bound-limb-variants`, and
  `reconstruction-zoo` release captures with exact CPU parity, Direct HIP,
  hipBLASLt, CK, rocWMMA, and AMDGPU builtin comparators where built.
- Constraints: keep signed prefix18 tree-CRT route where it wins; do not flip
  unsigned tree-CRT by default until end-to-end CDNA evidence beats fixed-prefix
  Garner, because prior evidence showed unsigned tree improves the export event
  but loses end-to-end.

### Rank 83 - Bounded i64/u64 Fused CRT Export

- Priority: P1.
- Target: bounded signed/unsigned final host-output captures.
- Problem: bounded accelerators can win raw RNS GEMM but lose once CRT export,
  status, and D2H are included.
- Implementation scope: add final-output kernels that combine residue
  reconstruction, range/status handling, and compact host-output staging for
  bounded i64/u64; preserve exact range checks and deterministic status.
- Local implementation status: Direct HIP bounded device export already runs
  final-output device CRT reconstruction, range-status handling, device output
  staging, status D2H, and output D2H. Export kernel identities are now
  registry-backed for bounded i64/u64, schema-checked across
  `export_variant.selected_kernel`, `exact_output_contract.kernel_identity`,
  and `reconstruction_variant.kernel_identity`, and grouped bounded captures now
  report the actual grouped export kernels plus `compact_contiguous` D2H policy.
  Local Windows HIP bounded i64/u64 normal and grouped smokes passed schema
  validation.
- Required evidence: repeated-B, large-release-validation, and export-bound
  scenarios with per-phase event labels for GEMM, CRT kernel, status memset,
  status D2H, output D2H, and end-to-end timing.
- Promotion rule: accelerator route must beat Direct HIP setup-inclusively for
  the final-output contract, not only residue-current GEMM.

### Rank 84 - Prepacked-B Setup Cost Reduction

- Priority: P1.
- Target: repeated-B production candidates.
- Problem: rocWMMA bounded-i64 1024 non-reuse has a real MI300X win, but
  prepacked-B reuse still loses setup-inclusively.
- Implementation scope: reduce B-side prepack setup cost, persist reusable
  B tiles, avoid repeated descriptor/setup work, key cache identity by backend,
  target, shape, prefix, modulus set, signedness, source version, layout
  version, and selected kernel.
- Local progress: benchmark-owned repeated-B cache setup no longer performs a
  separate public key-info query before creating the reusable B prepack cache;
  `rns8_create_prepack_cache` is the measured production path, unsupported
  plans stay on persistent matrix residency, and successful cache captures still
  emit `rns8_get_prepack_cache_info` metadata for review/cache-key visibility.
  The reuse-contract report now treats `runtime_prepack_cache` as first-class
  source identity for rocWMMA B-cache rows and requires production runtime
  cache metadata before marking an explicit repeated-B selector ready.
- Required evidence: `repeated-b`, `direct-hip-reuse-expansion`, and
  `reuse-contract` release groups comparing CPU anchor, Direct HIP,
  same-backend non-reuse, fastest non-reuse, and prepacked-B reuse.
- Promotion rule: setup-inclusive reuse must beat both same-backend non-reuse
  and fastest same-contract non-reuse at the declared reuse count.

### Rank 85 - Production Persistent Packed Operand Caches

- Priority: P1.
- Target: public B-side RNS cache first; A-side and A/B cache later.
- Problem: current reuse evidence mixes persistent residency, workspace-local
  reuse, and production cache semantics.
- Implementation scope: harden public cache eligibility, stale source-version
  rejection, backend/target mismatch rejection, cache-key serialization,
  packing-info metadata, and autotune/review visibility for production-capable
  caches.
- Local status: bounded i64/u64 rocWMMA B caches are the only production-cache
  eligible public surface; exact-wide rocWMMA B caches remain reusable
  evidence-only while preserving cached-GEMM correctness coverage. Runtime
  cache creation and cache metadata now require a nonzero operand
  `source_version`; zero-version operands remain key-inspectable only and
  cannot be treated as reusable cache evidence. Reuse reports now distinguish
  legacy persistent-residency source identity from public runtime-cache source
  identity, so production B-cache selector readiness cannot be inferred from
  allocation metadata alone. Benchmark schema validation now also requires
  runtime B-cache metadata to match the capture backend, semantics, B operand
  shape, prefix, finite modulus, cache source version, cache-key hash, and
  byte-size envelope before a capture can count as production-cache evidence.
- Required evidence: API/unit tests for identity rejection, stale version
  rejection, unsupported backend/shape rejection, and release captures proving
  setup-inclusive wins.
- Deferred: finite-u8, wrap64, sparse, and exact-wide cache promotion until
  each has its own same-contract setup-inclusive evidence.

### Rank 86 - HIP Graph Replay Break-Even Gate

- Priority: P1.
- Target: repeated workloads, not one-off GEMM.
- Problem: graph replay is benchmark-visible but not production-winning in the
  current CDNA review.
- Implementation scope: capture graph-safe pack/prepack/GEMM/export chains
  only when plan, workspace, buffers, source versions, device, prefix, and
  layout are stable; report graph capture cost, instantiate cost, replay cost,
  non-graph baseline cost, and break-even repeat count.
- Local status: graph reports, release review metadata, and compact failure
  summaries now emit graph total setup, baseline total setup, setup overhead,
  steady-state delta, declared repeats, break-even repeat count, and computed
  declared-repeat break-even satisfaction even when older review JSON omitted
  the precomputed boolean. Benchmark schema validation now binds graph replay
  captures to the selected plan autotune key and fixed m/n/k descriptor
  identity, so stale graph captures cannot be reused across shape or plan
  changes.
- Required evidence: `hip-graph-replay` release captures with checksum parity,
  graph availability metadata, setup-inclusive comparison, and no missing
  non-graph baseline.
- Deferred: public async/graph API. Keep graph replay internal to benchmark and
  autotune until a production contract is justified.

### Rank 87 - Full-Path Graph Replay For Pack/GEMM/Export

- Priority: P1.
- Target: bounded, finite-u8, exact-wide, and wrap64 full-path captures.
- Problem: earlier graph wins were workload-specific and not CDNA production
  evidence; accelerator graph safety for library handles/scratch is not solved.
- Implementation scope: make full-path graph capture cover H2D, pack kernels,
  GEMM, residue reduction/export, status, and D2H on explicit streams where
  safe; reject paths with unstable handles, scratch, or source versions.
- Required evidence: graph/non-graph same-contract release groups with event
  labels and setup-inclusive break-even analysis.
- Do not promote: graph-only rows or graph captures without same-contract
  non-graph release baselines.

### Rank 88 - Direct-HIP Pack Kernel Reduction

- Priority: P1.
- Target: pack-bound Direct-HIP and accelerator workflows.
- Problem: latest production winners still show pack as a primary loss phase on
  skinny and exact-wide cases.
- Implementation scope: coalesced/vectorized pack kernels, fused native-to-RNS
  pack plus GEMM, separate A/B pack event labels, pack elision when resident RNS
  inputs are current, and stricter source-currentness metadata.
- Required evidence: `fused-pack-gemm-small`, `native-to-rns-bridge`,
  `multi-modulus-pack`, `direct-hip-reuse-expansion`, and skinny scenarios.
- Promotion rule: pack optimization must improve setup-inclusive end-to-end, not
  only the isolated pack event.
- Local progress: top-level benchmark timing can now emit optional
  `raw_timings_us.pack_a` and `raw_timings_us.pack_b` phases, plus matching
  summaries and averages, for resident bounded, exact-wide, finite-u8, wrap64,
  vector-ALU, and vector-to-RNS chain captures. `raw_timings_us.pack` remains
  the aggregate measured phase, so existing setup-inclusive comparisons stay
  stable while pack-bound diagnostics identify the operand side that needs
  kernel or reuse work. Promoted autotune cache entries and `rns8-inspect`
  cache-hit output preserve `pack_a` and `pack_b` medians when present. Release
  review candidates and the supported failure-summary tool now also print the
  split pack medians, and the evidence database records per-side pack medians,
  bandwidth estimates, and dominant operand classification. The diagnosis now
  survives review refresh, cache installation, compact VM triage output, and
  durable JSON/CSV/Markdown evidence export. Host-batch per-task pack paths now
  return measured A/B pack totals for bounded i64/u64, exact-wide
  signed/unsigned, and finite-u8 captures. Direct-HIP host-batch GPU event
  captures now split the per-task A/B H2D and pack-kernel labels while also
  preserving aggregate pack labels for existing setup-inclusive comparisons.
  Direct-HIP, wrap64 Direct-HIP, finite-u8 Direct-HIP, and AMDGPU builtin
  resident-input plans now advertise source-versioned same-version pack elision
  through `rns8_get_plan_packing_info`; repeated nonzero imports with unchanged
  `source_version` and already-current device storage return before H2D or pack
  kernel launch.
  Grouped-dispatch slab pack captures now split the existing A-then-B slab
  uploads and pack kernels into matching A/B GPU event labels while preserving
  aggregate pack labels. Direct-HIP exact-wide prefix18 now uses fixed-prefix
  i64/u64 residue pack launchers for both single and grouped pack paths instead
  of falling through to the generic per-plane pack kernel; this matches the
  prefix18 export specialization and avoids reloading each source element once
  per modulus plane on prefix18 exact-wide captures. Direct-HIP pack launchers
  now also route contiguous `ld == cols` native i64/u64 and finite-u8 single and
  grouped packs through dedicated contiguous kernels. Those kernels keep the
  existing strided fallback unchanged but avoid per-thread row/column division
  and load source cells as `src[cell]` on the common row-major release path.
- Remaining work: implement wider vectorized pack kernels and fused native-pack
  plus GEMM routes selected by split evidence, then prove setup-inclusive wins
  against the contiguous-pack baseline.

### Rank 89 - Native-To-RNS And Vector-To-RNS Device Handoff

- Priority: P1.
- Target: chains where native vector output feeds RNS GEMM.
- Problem: host export/repack controls are expensive; device-current handoff can
  dominate chained workloads.
- Implementation scope: device repack bridge, native-output domain metadata,
  stale source tracking, and same-contract host-export/repack controls.
- Required evidence: `native-to-rns-bridge` and `vector-to-rns-chain` release
  groups with final checksum parity and explicit control-mode metadata.
- Constraint: do not add generic algebra API shims; keep this as a concrete
  domain-transition implementation.
- Local progress: the device bridge now rejects zero-version native producers
  before materializing Direct-HIP RNS input storage, leaves the target
  unmodified on stale-source rejection, and stamps successful handoff outputs
  with the producer source version. The public AUTO Direct-HIP materialization
  path now applies the same nonzero source-version guard before converting
  current native bounded storage into RNS residues, so cache/reuse evidence
  cannot be generated from anonymous native producers.

### Rank 90 - CDNA3 AMDGPU Builtin Dense MFMA Backend

- Priority: P1.
- Target: MI300X `gfx942`.
- Problem: CK/rocWMMA provide MFMA evidence, but the public AMDGPU builtin
  backend must own target-specific kernels where the library needs control over
  layout and epilogue.
- Implementation scope: real runtime kernels for
  `v_mfma_i32_16x16x32_i8` and `v_mfma_i32_32x32x16_i8`, centered-residue
  epilogues, finite-u8 epilogues, selected-kernel metadata, dispatch gating,
  and fallback-free unsupported status.
- Required evidence: targeted GPU differential tests, ISA histograms proving
  MFMA use, no forbidden divide/remainder/reciprocal in hot symbols, and release
  comparisons against Direct HIP, hipBLASLt, CK, and rocWMMA.
- Local progress: AMDGPU builtin captures now emit schema-checked
  `backend_metadata.matrix_instruction_family`,
  `matrix_instruction_shape`, `matrix_instruction_dtype`, and
  `matrix_instruction_sparsity` derived from the selected builtin kernel.
  Local `gfx1100` smoke captures report the RDNA3 dense WMMA route as
  `wmma/16x16x16/iu8/dense`; compiled ISA reports remain the proof of the
  actual instruction histogram. AMDGPU builtin GPU-event capture now records
  the selected target-specific matrix-core label instead of a generic backend
  label, including local bounded and finite-u8 schema smokes for the RDNA3
  WMMA route. Local differential coverage now also compares AMDGPU builtin
  exact-wide signed/unsigned RNS output against the CPU reference, and a tiny
  benchmark smoke schema-validates the exact-wide RDNA3 WMMA selected kernel,
  matrix metadata, and `amdgpu_builtin_fused_i32_to_centered_residue_rns_output`
  epilogue. Dense bounded and finite-u8 AMDGPU builtin differential coverage
  now includes non-multiple M/N/K shapes, so matrix-core tile tails,
  finite-output tails, and K-tail zero-padding are checked against CPU before
  deeper CDNA/RDNA tile tuning.
- Promotion rule: builtin wins only when exact CPU parity and setup-inclusive
  release review beat Direct HIP.

### Rank 91 - CDNA3 MFMA Tile, LDS, And K-Block Tuning

- Priority: P1.
- Target: dense MFMA kernels after rank 90 lands.
- Problem: a compiled MFMA kernel is not enough; tile shape, LDS layout,
  register pressure, and K-blocking decide end-to-end value.
- Implementation scope: tune 16x16x32 versus 32x32x16, K-block policies,
  swizzled B layouts, LDS staging, accumulator pressure, and occupancy.
- Current implementation status: explicit `amdgpu-cdna3-mfma-16x16x32` and
  `amdgpu-cdna3-mfma-32x32x16` tile-shape variants force the AMDGPU builtin
  dense CDNA3 runtime down each MFMA path and record the selected kernel in
  benchmark metadata; LDS, K-block, swizzled-layout, and occupancy tuning still
  need measured follow-through.
- Required evidence: `tile-shape-sweeps`, `k-block-tile-variants`, and
  `layout-search` release captures plus ISA/resource/counter reports.
- Promotion rule: no tile variant promotes without same-contract default
  baseline, ISA/resource evidence, and end-to-end improvement.

### Rank 92 - CK CDNA Pack/Epilogue Tuning

- Priority: P1.
- Target: CK/XDL on CDNA.
- Problem: CK often loses to Direct HIP on repeated/export-heavy CDNA cases.
- Implementation scope: reduce CK pack/copy/add centered overhead, tune XDL
  tile choices, avoid scratch paths when fused epilogues are possible, and
  expose selected-prefix deep event labels.
- Required evidence: CK deep event labels, CK ISA histograms, same-contract
  release review against Direct HIP and rocWMMA.
- Constraint: CK remains optional; do not make it required for correctness.

### Rank 93 - rocWMMA CDNA Pack, Reuse, And Shape Tuning

- Priority: P1.
- Target: rocWMMA bounded/exact/finite CDNA paths.
- Problem: rocWMMA has one bounded 1024 non-reuse win but loses skinny, 512,
  reuse, and exact-wide export-heavy rows.
- Implementation scope: improve prepacked-B path, reduce pack-A-prepacked-B
  overhead, specialize 512 versus 1024, reject unsuitable skinny shapes, and
  keep MFMA evidence attached to captures.
- Required evidence: repeated-B, skinny-GEMV, exact-wide-export, and finite
  release groups with deep rocWMMA event labels.
- Promotion rule: rocWMMA can promote only per contract; a 1024 win does not
  justify 512, skinny, exact-wide, or reuse promotion.

### Rank 94 - hipBLASLt Exact-Wide And Chain Tuning

- Priority: P1.
- Target: exact-wide and residue-chain workloads where hipBLASLt can still win.
- Problem: hipBLASLt can win larger exact-wide shapes, but scratch reduction,
  source-version identity, and final export can erase gains.
- Implementation scope: improve prepack reuse, grouped/batched scheduling,
  scratch-to-residue conversion, exact-wide chain source-version propagation,
  and final-output versus residue-current routing.
- Required evidence: exact-wide large-shape, RNS-chain, and final-output chain
  captures with CPU/direct baselines and checksum parity.
- Constraint: hipBLASLt remains an optional baseline accelerator, not a
  correctness dependency.

### Rank 95 - Sparse-A 4:2 Packing And CPU Reference

- Priority: P1.
- Target: explicit sparse-A v1 contract.
- Problem: sparse hardware paths cannot be meaningful until sparse storage,
  validation, and CPU reference are complete end to end.
- Implementation scope: dense-to-sparse validation/packing for caller-supplied
  A-side 4:2 structure, resident sparse-A handles, canonical index layout,
  signed/unsigned byte interpretation, K-divisibility checks, source-version
  identity, and CPU expand-to-dense reference.
- Required evidence: API/unit tests for valid/invalid groups, signedness,
  K divisibility, dense/sparse API rejection, cache-key mismatch, and sparse
  CPU-vs-dense parity.
- Local progress: `rns8-bench --sparse-a-4-to-2` now generates explicit
  finite-u8 sparse-A 4:2 inputs, records the structured sparse input
  distribution, routes CPU through sparse expand-to-dense reference, and
  validates CPU sparse captures through schema v4. Bounded i64/u64 benchmark
  lanes now also generate true 4:2 sparse native A inputs, pack them into signed
  centered RNS sparse planes, route CPU/AMDGPU builtin captures through
  `rns8_gemm_rns_sparse_a`, keep dense-baseline captures on the dense path for
  the same sparse-shaped input, and schema-validate tiny CPU sparse bounded
  captures. Exact-wide signed/unsigned benchmark lanes now use the same
  explicit sparse-A RNS storage path for CPU/AMDGPU builtin captures, with
  matching dense Direct-HIP and dense AMDGPU baseline rows for the expanded
  sparse-shaped input. CPU sparse-A contract tests now cover finite-u8, bounded
  i64/u64, and exact-wide signed/unsigned expand-to-dense parity against the
  dense CPU reference, including source version stamping on sparse RNS GEMM
  outputs. The AMD matrix-instruction calculator report now labels sparse
  SMFMAC/SWMMAC candidates as requiring this explicit sparse-A 4:2 contract
  instead of stale future-only wording, while still forbidding implicit dense
  GEMM routing to sparse instructions. Sparse public API tests now also reject
  mismatched RNS/finite sparse contracts and wrong finite moduli before
  mutating output source versions or residue contents.
- Constraint: no automatic pruning, no B-side sparsity, no unstructured sparse
  path, and no sampled correctness.

### Rank 96 - CDNA3 Sparse SMFMAC Runtime

- Priority: P1.
- Target: MI300X `gfx942` sparse-A runtime.
- Problem: metadata names sparse candidates, but production needs real SMFMAC
  kernels behind explicit sparse-A handles.
- Implementation scope: implement `v_smfmac_i32_16x16x64_i8` first, then
  `v_smfmac_i32_32x32x32_i8` only if real; add sparse pack/setup timing,
  selected-kernel metadata, ISA histograms, and exact CPU sparse parity.
- Required evidence: sparse release captures comparing sparse setup-inclusive
  path against dense Direct HIP, dense AMDGPU builtin, CK/rocWMMA where
  comparable, and CPU sparse reference.
- Local progress: the `sparse-a-4-to-2` sweep family now emits CPU sparse
  reference, Direct-HIP dense same-input baseline, AMDGPU sparse runtime, and
  AMDGPU dense sparse-input baseline rows. Review backend ids separate
  `amdgpu-builtins-sparse-a-runtime` from
  `amdgpu-builtins-dense-sparse-a-input` to avoid duplicate-backend blockers
  while keeping both rows in the same sparse contract group. AMDGPU builtin
  differential coverage now includes bounded i64/u64 and exact-wide
  signed/unsigned sparse-A RNS paths against the dense AMDGPU builtin path for
  the same sparse-shaped native input; it builds locally and skips on RDNA3,
  then executes on CDNA3 SMFMAC or RDNA4 SWMMAC targets. Benchmark schema
  validation now distinguishes finite-u8 sparse workspace/events from bounded
  and exact-wide RNS sparse workspace/events, including sparse-A value/index
  upload phases for RNS sparse captures. Sparse bounded, finite-u8, and
  exact-wide AMDGPU differential shapes now use non-multiple M/N with
  K-divisible-by-4 inputs, so CDNA3 SMFMAC and RDNA4 SWMMAC runtime gates cover
  sparse tile tails instead of only exact 16x16 output tiles.
- Promotion rule: sparse ships only if end-to-end sparse-A execution beats the
  dense path for the same expanded mathematical input.

### Rank 97 - RDNA4 Sparse SWMMAC Runtime Readiness

- Priority: P1.
- Target: `gfx1200` and `gfx1201`.
- Problem: RDNA4 sparse readiness must exist without making hardware
  performance claims before hardware proof.
- Implementation scope: compile/schema/runtime gates for
  `v_swmmac_i32_16x16x32_iu8` and explicit `iu4` research metadata, using the
  sparse-A contract from ranks 95 and 96.
- Required evidence: compile/ISA/schema tests on available toolchains; runtime
  performance claims are deferred until real RDNA4 hardware captures exist.
- Constraint: RDNA3 has no sparse runtime backend because it has WMMA, not
  SWMMAC.

### Rank 98 - RDNA3 Dense WMMA Backend

- Priority: P1.
- Target: Windows `gfx1100` and Linux RDNA3 where supported.
- Problem: RDNA3 needs a real dense builtin route, not only CK/rocWMMA
  wrappers.
- Implementation scope: implement `v_wmma_i32_16x16x16_iu8` dense kernels for
  bounded and finite paths, preserve wave32/wave64 layout metadata separately,
  encode signedness through `NEG[0]`, `NEG[1]`, and integer constraints, and
  keep `iu4` as explicit research only.
- Required evidence: CPU/direct differential tests, ISA histograms, matrix
  calculator layout artifacts, and release comparisons against Direct HIP, CK,
  rocWMMA, and hipBLASLt where available.
- Constraint: no sparse RDNA3 runtime claim.

### Rank 99 - RDNA3 VALU Optimization Lane

- Priority: P1.
- Target: non-matrix hot code on `gfx1100`.
- Problem: pack, export, reduction, byte-limb wrap64, zero-mask handling, and
  small/skinny work can be VALU-bound rather than matrix-core-bound.
- Implementation scope: VOPD-friendly instruction selection, DPP/cross-lane
  reductions, vectorized pack/export, wrap64 byte-limb improvements, and
  compiled ISA proof for useful dual-issue or lane operations.
- Required evidence: Direct-HIP/vector-ALU event reports and ISA reports
  proving the intended instruction families appear in hot objects.
- Promotion rule: end-to-end same-contract improvement only.

### Rank 100 - RDNA4 Dense WMMA Readiness

- Priority: P2.
- Target: `gfx1200` and `gfx1201`.
- Problem: RDNA4 should compile and report dense WMMA readiness before real
  hardware is available, without claiming performance.
- Implementation scope: dense WMMA selected-kernel metadata, target gating,
  schema/review support, and calculator layout artifacts for RDNA4 wave modes.
- Required evidence: compile/schema/ISA readiness where possible; hardware
  runtime evidence is a separate future gate.
- Constraint: no README or cache performance claim without real RDNA4 captures.

### Rank 101 - CDNA4 Readiness

- Priority: P2.
- Target: `gfx950`.
- Problem: CDNA4 target metadata exists but must remain explicit readiness,
  not inferred MI300X performance.
- Implementation scope: target gating, preset coverage, ISA classifier support,
  matrix calculator artifacts, and unsupported/runtime status when hardware is
  unavailable.
- Required evidence: compile/schema readiness first; real hardware release
  validation later.
- Constraint: no CDNA4 performance claim from CDNA3 data.

### Rank 102 - Resident Lifetime And Workspace Arena Reduction

- Priority: P2.
- Target: all repeated GPU workloads.
- Problem: allocation, descriptor, and workspace churn contaminate end-to-end
  timing and can erase kernel wins.
- Implementation scope: reusable workspace slabs for pack, GEMM scratch,
  export status, CRT constants, graph buffers, prepack descriptors, and
  resident output buffers; expose allocation counters before warmups and after
  repeats.
- Required evidence: `resident-lifetime-arena`, repeated-B, graph, and chain
  captures with setup-inclusive allocation metadata.
- Constraint: no hidden global singleton cache; identity and lifetime must be
  explicit.

### Rank 103 - Persistent Grouped Task Execution

- Priority: P2.
- Target: many small, repeated, and batched same-shape workloads.
- Problem: launch/API overhead dominates many-small work and graph replay is
  not always the right answer.
- Implementation scope: persistent grouped CDNA task descriptors, device task
  queues where justified, same-shape resident inputs, grouped finite and RNS
  paths, exact status/checksum policy, and descriptor reuse validation.
- Required evidence: `grouped-dispatch`, `many-small`, `small-oneshot`, and
  repeated workload release captures.
- Constraint: grouped APIs must not perform hidden host packing, hidden AUTO
  routing, or host final export unless explicitly contracted.

### Rank 104 - Streaming Overlap Pipeline

- Priority: P2.
- Target: repeated or batched pack/GEMM/export pipelines.
- Problem: serial pack, GEMM, export, and D2H leave overlap opportunities when
  dependencies are independent across tasks or repeats.
- Implementation scope: pack-next/GEMM-current/export-previous streams,
  dependency contracts, status synchronization, final checksum synchronization,
  and event attribution.
- Required evidence: `streaming-overlap` release captures against serial
  same-contract baselines.
- Constraint: do not overlap across dependencies that affect exactness,
  source-version visibility, or first-error ordering.

### Rank 105 - Residue-Current And Lazy Final Export

- Priority: P2.
- Target: RNS chains and final-output workflows.
- Problem: exporting to host after every GEMM wastes work when the next
  operation consumes RNS residues.
- Implementation scope: residue-current output routing, lazy final export,
  chain metadata, cache identity, and clear next-op contracts for RNS GEMM,
  native GEMM, final export, and reuse-B.
- Required evidence: `rns-chain`, `rns-chain-final-output`,
  `exact-wide-output-chain`, and vector/native-to-RNS chain captures.
- Promotion rule: compare against independent export/repack controls.

### Rank 106 - Verification Amortization Without Correctness Weakening

- Priority: P2.
- Target: expensive CPU validation and repeated sweeps.
- Problem: CPU correctness anchors are necessary, but timed CPU baselines should
  not be repeated wastefully when correctness-anchor mode is enough.
- Implementation scope: de-duplicate CPU reference captures by semantic
  contract, shape, seed, output policy, checksum policy, and source identity;
  keep timed CPU baselines only for CPU performance claims.
- Required evidence: benchmark-sweep tests proving identical checksums across
  de-duplicated anchors and release reviews that still reject missing required
  baselines.
- Constraint: no sampled or probabilistic CPU reference as a substitute for
  exact anchors.

### Rank 107 - Error-Detection And Status Fast Paths

- Priority: P2.
- Target: bounded and exact-wide exports.
- Problem: status memset/D2H and range checks can be pure overhead when the
  semantic contract structurally proves full coverage.
- Implementation scope: status elision where mathematically proven, vectorized
  status aggregation where required, deterministic first-error selection, and
  schema-visible status policy.
- Required evidence: range-bound edge tests, exact-wide limb coverage tests,
  schema fixtures, and event reports showing status overhead reduction.
- Constraint: never weaken exact range checks to make an accelerator pass.

### Rank 108 - Finite-u8 Modulus-Family Tuning

- Priority: P2.
- Target: finite ring/field u8 for hot moduli and generic moduli.
- Problem: finite-u8 performance depends on modulus family, distribution, and
  pack/export overhead.
- Implementation scope: direct-HIP finite pack/export tuning, CK/rocWMMA finite
  epilogues, hot modulus reducers for 251/255/256, generic modulus fallback,
  and distribution-sensitive scenario coverage.
- Required evidence: `finite-distributions`, `finite-modulus-map`,
  `finite-generic-moduli`, and large finite release captures.
- Constraint: modulus-specific wins must not be generalized to other moduli.

### Rank 109 - Strict Wrap64 Next-Generation Backend

- Priority: P2.
- Target: strict `mod 2^64` byte-limb workloads.
- Problem: Direct HIP v4 is strong; the current rocWMMA wrap64 candidate loses
  and remains internal.
- Implementation scope: only pursue a materially different byte-limb
  matrix-engine or VALU strategy, such as better byte-diagonal accumulation,
  carry handling, or low64 export; do not polish the losing candidate.
- Required evidence: CPU byte-limb oracle, Direct-HIP v4 baseline, full output
  differentials, ISA proof, and release review.
- Constraint: wrap64 must stay byte-limb based, not odd-modulus CRT, unless a
  valid exact bound is supplied.

### Rank 110 - Adaptive Prefix And Zero-Skip Expansion

- Priority: P2.
- Target: bounded adaptive per-tile workloads.
- Problem: adaptive prefix and zero-tile skip can remove residue planes and
  work, but only when tile bounds prove it deterministically.
- Implementation scope: improve per-tile bound computation cost, prefix grouping
  execution, zero-row/zero-column/output skip, and tile metadata reuse.
- Required evidence: `adaptive-bands`, `adaptive-grouped-scheduler`,
  `bound-discovery`, and zero-skip scenarios with exact proof metadata.
- Constraint: no probabilistic early termination in production.

### Rank 111 - Autotune Cache Production Route Semantics

- Priority: P2.
- Target: AUTO routing and cache entries.
- Problem: Direct HIP production wins must be represented clearly, while
  accelerator promotion remains strict.
- Implementation scope: separate fastest production route from fastest
  accelerator route in cache/review tooling, exact-key matching, shape-family
  advisory-only recommendations, target/toolchain/source identity gates, and
  cache stale rejection.
- Required evidence: review tests for Direct-HIP winners, accelerator winners,
  duplicate backend captures, missing baselines, source mismatch, and target
  mismatch.
- Local progress: the supported compact failure summary now emits
  `TOP_ACTIONABLE_ACCELERATOR_BLOCKERS`, sorted by setup-sensitive blockers,
  setup-inclusive reuse/graph gaps, direct/vector underperformance, and the
  worst phase ratio. Each row includes backend, shape, selected kernel, primary
  loss phase, bottleneck, pack split, route score, blocker set, capture path,
  and contract key, so one VM summary identifies the exact accelerator rows to
  optimize after Direct HIP is the current production winner.
- Constraint: no approximate family promotion without a real family contract.

### Rank 112 - Scenario Catalog Cleanup And Pre-VM Linting

- Priority: P2.
- Target: all benchmark scenario JSON files.
- Problem: VM time is wasted when scenarios are invalid, evidence-only rows are
  mislabeled as promotable, or required baselines are omitted.
- Implementation scope: stricter scenario linting for fixed-prefix range
  validity, promotion eligibility, required baselines, output domain,
  checksum policy, graph/reuse evidence scopes, and backend support.
- Required evidence: Python scenario-catalog tests and a dry planning command
  or equivalent manifest generation that fails before paid VM work.
- Local progress: `benchmark_sweep.py --lint-scenarios` now performs a
  no-execute catalog lint that defaults to `--scenario all`, generates the same
  scenario command entries and manifest as real sweeps using a placeholder
  `rns8-bench` path when none is supplied, prints a compact JSON summary, and
  exits before capture execution or review. The command-matrix test invokes the
  real CLI for `--scenario release-candidates` and verifies that only release
  candidates appear in the manifest and no review report is written. The CDNA
  first-pass wrapper now runs the lint mode before every rank-scenario sweep and
  records it in `command-plan.txt`, so scenario catalog failures stop before
  paid benchmark captures start.
- Constraint: do not make scripts permissive to hide invalid contracts.

### Rank 113 - GPU Event Coverage Completion

- Priority: P2.
- Target: all HIP-resident backends.
- Problem: promotion needs phase attribution, but some paths still have coarse
  or missing event labels.
- Implementation scope: complete event hooks for Direct HIP, hipBLASLt, CK,
  rocWMMA, AMDGPU builtins, graph replay, reuse/prepack, exact-wide export,
  finite-u8, wrap64, and sparse paths.
- Required evidence: schema tests for every required event set and VM captures
  showing required events available for promotable rows.
- Constraint: nullable events are allowed for unsupported paths, but missing
  required events must block promotion.
- Local progress: AMDGPU builtins are now event-capable in `rns8-bench`; dense
  bounded/exact and finite-u8 captures use deep accelerator source scope and
  selected MFMA/WMMA/SMFMAC/SWMMAC event labels derived from the public selected
  kernel.

### Rank 114 - Counter And Resource Evidence Gate

- Priority: P2.
- Target: CDNA/RDNA optimizer decisions.
- Problem: timing alone does not tell whether a route is memory-bound,
  occupancy-bound, launch-bound, register-bound, or instruction-bound.
- Implementation scope: integrate `gpu_counter_report.py`, ISA resource fields,
  VGPR/SGPR/LDS/scratch occupancy, wait-state signals, global memory traffic,
  and matrix-instruction histograms into review artifacts.
- Required evidence: counter/resource reports attached to tile/MFMA/WMMA/pack
  tuning decisions.
- Constraint: counters guide optimization; they do not replace exact CPU parity
  or same-contract release timing.

### Rank 115 - Large Release Validation Refresh

- Priority: P3.
- Target: final broad CDNA validation after targeted ranks land.
- Problem: all-backend all-scenario captures are expensive and should not be
  used as the first debugging step after every edit.
- Implementation scope: run targeted correctness/perf scenarios first, then
  broad `large-release-validation`, `all`, or repeated-workload captures only
  after focused gates pass.
- Required evidence: zero failed captures, zero comparable checksum mismatches,
  no invalid missing baselines, reviewed promotable entries only from valid
  release scopes, and summary artifact saved under `temp/`.
- Constraint: do not call broad all-runs "done" when targeted blockers remain.

### Rank 116 - Benchmark Ergonomics And Progress Discipline

- Priority: P3.
- Target: scripts and developer workflow.
- Problem: long VM runs are expensive and opaque when progress, active command,
  and remaining work are unclear.
- Implementation scope: progress output for sweep commands, per-command
  start/end/duration, supported review-refresh path, compact failure summary,
  stable VM command shapes, and no brittle shell-array one-liners in docs.
- Required evidence: CLI tests for progress/review-only/capture-root paths and
  VM usage notes that produce the same supported summary output.
- Local progress: `benchmark_sweep.py --review-only --capture-root ...` is
  covered by a subprocess command-matrix test that copies schema-valid captures,
  runs without `--bench`, and verifies review JSON/Markdown output. This is the
  supported review-refresh path for VM captures and replaces shell-expanded
  `--capture` arrays.
- Constraint: diagnostics must not change benchmark semantics or promotion
  strictness.
