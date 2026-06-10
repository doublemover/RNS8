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
The active table below previously contained 38 ranks.
Ranks 82, 83, 87, 99, 103, 104, 109, 111, 112, 113, 114 were closed on
June 10, 2026 after implementation completion, gfx1100 compilation, and
measured sweep evidence. See
[performance-gain-completed-work.md](performance-gain-completed-work.md).

The queue below retains 27 active ranks (79-81, 84-86, 88-98, 100-102,
105-108, 110, 115-116) plus 14 new ranks (117-130) opened June 10, 2026
for creative optimization work and deferred research.

The active table below now contains 41 ranks (27 active + 14 new). Rank IDs are historical/stable
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
  Skinny GEMV summaries now pair specialized Direct-HIP rows with explicit
  generic-tiled control captures and report the local disposition, speedup,
  selected kernels, and capture paths.
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
  The Direct-HIP backend now also has a resident-RNS `2 <= N <= 8` row-parallel
  route, reported as `direct_hip_prefix9_rns_gemv_small_n_i64_v1` /
  `direct_hip_prefix9_rns_gemv_small_n_u64_v1` with the
  `direct_hip_skinny_gemv_small_n_resident_rns` execution mode and a dedicated
  `rns_gemv_small_n_kernel_group` event phase.
  Public Direct-HIP plan metadata now reports the same GEMV selected kernel,
  epilogue, and workspace for bounded fixed-prefix `N=1` plans instead of the
  generic grouped GEMM identity.
  Release review now treats stale `N <= 8` Direct-HIP captures as route-invalid
  unless they report a GEMV selected kernel, the matching skinny GEMV execution
  mode, and GEMV-specific GPU event phase metadata, so generic tiled GEMM
  cannot be reported as the production skinny route by accident. The
  `skinny-gemv` scenario family includes bounded i64/u64 `N=4` and `N=8` rows to compare
  the specialized small-N route against current tiled Direct HIP and accelerator
  candidates. It also includes Direct-HIP-only `tile_shape_evidence_only`
  tiled-control rows using `direct-hip-skinny-tiled-control-128x128`, which
  intentionally disables skinny auto-routing through the non-default tile-shape
  guard so the supported failure-summary tool can print specialized-vs-tiled
  comparisons without changing release-review grouping. The promotable
  `skinny-gemv` rows are now release-mode evidence, while the tiled controls
  remain smoke-only comparison captures.
- Required evidence: release captures for `512x1x512`, `256x1x4096`, and
  `1024x1x1024`, plus small-N `512x4x512`, `1024x4x1024`,
  `512x8x512`, and `1024x8x1024`, against CPU
  anchor, current tiled Direct HIP, vector-ALU where applicable, and rocWMMA;
  schema must report a GEMV selected kernel and phase timings for `N <= 8`.
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
  signed/unsigned schema-validated smokes. The vector-ALU backend now also has
  an explicit `2 <= N <= 8,K>=512` native exact small-N route with
  `hip_vector_alu_*_gemv_small_n_exact_192b_v1`, the
  `public_runtime_vector_alu_gemv_small_n_native_buffers` /
  `benchmark_owned_vector_alu_gemv_small_n_native_buffers` execution modes,
  `vector_alu_*_gemv_small_n_kernel` event labels, public plan selected-kernel
  metadata, C++ autotune-cache rejection for stale generic small-N rows, and a
  local Windows HIP bounded-u64 `16x4x512` schema-valid smoke whose checksum
  matched CPU. Release review now blocks stale vector-ALU skinny captures for
  both `N=1` and small-N unless their execution mode, selected kernel, and GPU
  event phases identify the matching GEMV route explicitly.
- Required evidence: `skinny-gemv` release groups with CPU, Direct HIP,
  vector-ALU, and accelerator comparators; vector-ALU rows must not be compared
  against exact-wide, finite-u8, or wrap64 contracts.
- Promotion rule: vector-ALU may win production only for bounded native-output
  contracts with exact same-contract baselines.


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
  cache metadata before marking an explicit repeated-B selector ready. Current
  captures also split one-time `prepack_setup_us` into A pack, B pack, runtime
  cache materialization, and unclassified setup overhead so Rank 84 follow-up
  work can attack the actual setup loss phase instead of the aggregate number.
  The benchmark now skips reusable-B cache materialization probes unless the
  plan actually selected rocWMMA, so CPU/Direct-HIP/CK repeated-B controls no
  longer pay an unsupported rocWMMA-cache setup attempt. The standalone
  reuse-contract report now carries the same setup breakdown and primary setup
  phase counts, and the compact VM failure summary emits
  `PREPACK_SETUP_PRIMARY_PHASE_COUNTS`, so cache selector diagnostics can
  identify whether a reuse row is B-pack-bound, cache-bound, or dominated by
  unclassified setup overhead. The rocWMMA public runtime B-cache materializer
  now uses one prefix-aware 3D B-pack launch for all RNS planes instead of one
  launch per modulus plane, directly reducing setup launch count for bounded
  prefix-9 repeated-B rows while preserving the ordinary per-GEMM rocWMMA pack
  path. The next reduction slice adds explicit host-native bounded i64/u64
  B-cache constructors and routes rocWMMA `--reuse-packed-b` benchmark setup
  through them, so repeated-B captures can materialize the reusable B cache
  directly from row-major native input instead of first packing B into resident
  RNS storage and then swizzling that storage into the rocWMMA cache layout.
  The production `repeated-b` scenario family now includes bounded-u64 512 and
  1024 release baselines plus matching rocWMMA prepacked-B candidates, so the
  public B-cache gate exercises both bounded signed and unsigned contracts
  setup-inclusively. The broader `reuse-contract` matrix is now release-mode
  evidence too, covering non-reuse baselines plus stable-A, stable-B, and
  stable-A+B explicit reuse contracts at 1024 and 2048.
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
  The serialized `prepack-v2` cache key itself must carry the same backend,
  semantic, shape, prefix, source-version, device, layout-version,
  plan-fingerprint, and hash identity so stale side metadata cannot certify a
  mismatched runtime cache.
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
  changes. Executed graph captures must now report positive capture and
  instantiate timings, so zero setup placeholders cannot pass as break-even
  evidence.
- Required evidence: `hip-graph-replay` release captures with checksum parity,
  graph availability metadata, setup-inclusive comparison, and no missing
  non-graph baseline.
- Deferred: public async/graph API. Keep graph replay internal to benchmark and
  autotune until a production contract is justified.


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
  Release review candidates now carry `pack_diagnostics` with pack, pack-A,
  pack-B, pack share, dominant operand, pack layout, source-versioned-input,
  and same-version pack-elision metadata. The supported failure-summary tool
  prints a dedicated `PACK_PHASE_DIAGNOSTICS` section so pack-bound VM results
  identify whether A, B, or missing split timing is the next optimization target
  without digging through review JSON. Review `next_work` now promotes those
  diagnostics into concrete pack-action rows, including missing split timing,
  A-dominant, B-dominant, balanced pack, missing same-version pack elision, and
  native-input pack-fusion follow-up. Direct-HIP fixed-prefix contiguous i64/u64
  single and grouped pack launchers now use four-cell kernels that process four
  adjacent row-major source cells per thread while preserving the original
  strided fallback and output residue-plane layout. `fused-pack-gemm-small`
  now has fixed-prefix bounded i64/u64 release-review pairs for the existing
  transient uniform-small native i8 A/B input path: each pair includes CPU and
  ordinary Direct-HIP baselines plus a Direct-HIP
  `--transient-uniform-small-inputs` candidate with distinct capture naming and
  backend identity, so VM evidence can compare the fused native-input route
  against the contiguous and widened pack baselines without ad hoc commands.
  `fused-pack-gemm-small` is now entirely release-mode evidence, so the
  one-shot bounded/finite and persistent bounded/finite companion rows stay in
  the same release review as the transient fused native-input candidates.
  Direct-HIP fixed-modulus finite-u8 contiguous single and grouped pack
  launchers now also use four-cell kernels for moduli 251, 255, and 256. The
  generic dynamic-modulus and strided paths stay on the existing one-cell
  kernels, while the hot row-major finite release paths reduce launch work-items
  further and keep the same centered-residue output layout. Wrap64 Direct-HIP
  contiguous row-major pack and export now use four-cell byte-limb kernels as
  well, preserving strided pack fallbacks while reducing work-items and address
  arithmetic on small and medium strict `mod 2^64` captures. The wrap64 ISA
  checker names the widened pack/export kernels explicitly so the hot-symbol
  no-divide/no-matrix-engine gate still covers the executed code path.
  `multi-modulus-pack` is now release-mode evidence for fixed bounded prefixes
  3, 5, and 9 plus exact-wide prefix 20, keeping the fixed-prefix pack pressure
  lane in the same release review as the fused native-input rows.
  `generated-prefix-reducers` is also release-mode evidence for bounded
  prefixes 3/5/9 and exact-wide prefix 20, with prefix-1 host-export scenarios
  still rejected by catalog lint before VM time is spent.
- Remaining work: implement broader fused native-pack plus GEMM routes selected
  by split evidence, then prove setup-inclusive wins beyond the current
  uniform-small fixed-prefix Direct-HIP candidate lane.


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
  cannot be generated from anonymous native producers. Release review now
  blocks native-to-RNS bridge and vector-to-RNS chain captures that omit forced
  bridge metadata, producer/consumer backend identity, control mode, or the
  device handoff phase scope, making stale handoff evidence visible in compact
  summaries instead of only schema failures. Review candidates now also carry
  `native_to_rns_handoff_diagnostics`, and the supported failure-summary tool
  prints the conversion event label, conversion median, host-repack control
  median, vector-output D2H median, consumer GEMM median, and conversion share
  of consumer GEMM so the unfused materialization cost is visible before
  implementing fused native-pack plus GEMM kernels.
  The supported failure-summary tool now pairs vector-to-RNS fused device
  handoff rows against matching host export/repack controls even when review
  scenario identities differ. It reports local promote/drop/keep-experimental
  disposition, fused/control end-to-end medians, conversion, host-repack,
  vector-output D2H, consumer-GEMM costs, checksum blockers, and capture paths
  so VM output immediately shows whether the device-current handoff is worth
  deeper kernel work. `native-to-rns-bridge` rows are now release-mode evidence
  alongside `vector-to-rns-chain`, so the AUTO-forced Direct-HIP
  materialization path is reviewed with the same release discipline as the
  producer/consumer chain handoff controls.


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
  Captures now also report matrix operand signedness, A/B value contracts,
  sparse contract fields when applicable, and RDNA integer modifier policy,
  so `iu8` WMMA/SWMMAC rows cannot hide whether signed centered residues were
  actually requested through the builtin operands.
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
  deeper CDNA/RDNA tile tuning. Release review now blocks AMDGPU builtin
  matrix-core promotion with `missing_amdgpu_builtin_matrix_isa_histogram`
  unless a compiled ISA sidecar supplies an exact MFMA/WMMA/SMFMAC/SWMMAC
  histogram for the selected target/backend. Review also checks the selected
  builtin kernel against the expected matrix instruction mnemonic, so a
  `16x16x32` MFMA route cannot be certified by a different WMMA/SMFMAC/SWMMAC
  histogram or by the wrong dense/sparse instruction shape. Schema validation
  now also rejects AMDGPU builtin captures whose selected kernel does not match
  the capture semantics and runtime target family, so finite-u8 rows cannot
  claim the non-finite CDNA3 `32x32x16` route and RDNA3 rows cannot claim
  sparse SMFMAC/SWMMAC kernels. Release review applies the same compatibility
  check as a promotion blocker for older or programmatic review inputs that
  bypass schema validation. Runtime reviewed-cache selection now uses the same
  target-aware dense AMDGPU builtin kernel rules for CDNA3, RDNA3, and RDNA4
  entries, while still rejecting sparse SMFMAC/SWMMAC and INT4 research kernels
  from the dense `rns8_plan` autotune cache path.
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
  benchmark metadata. The tile-shape sweep catalog now covers bounded i64,
  bounded u64, exact-wide signed, exact-wide unsigned, and finite-u8 contracts
  for both dense MFMA variants, so CDNA3 tuning evidence is no longer inferred
  from signed bounded and finite proxy rows alone. LDS, K-block,
  swizzled-layout, and occupancy tuning still need measured follow-through.
  Release review now blocks AMDGPU builtin
  CDNA3 tile-shape captures when the `amdgpu-cdna3-mfma-*` variant name
  disagrees with the selected kernel or matrix-instruction family/shape/dtype,
  so a 32x32x16 tuning row cannot be certified by a 16x16x32 kernel.
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
  Release review now treats sparse-A route identity as promotion-critical:
  sparse captures must carry the explicit sparse scenario contract, K must be
  divisible by 4, input distribution must identify 4:2 sparse-A structure, and
  the autotune key must record A-side 4:2, group size 4, exactly 2 nonzeros,
  canonical 2-bit K-group indices, explicit sparse-A value signedness, and
  dense B. Schema validation now separately requires the public A value
  contract, dense-B contract, and canonical compression index layout in
  `backend_metadata`, so finite-u8 sparse public values and centered RNS sparse
  planes cannot be conflated.
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
  while keeping both rows in the same sparse contract group. Sparse synthetic
  backend IDs now normalize back to the `amdgpu-builtins` family for ISA
  sidecar/resource lookup, so sparse runtime and dense sparse-input rows can
  display MFMA/SMFMAC histograms from the shared compiled-object report instead
  of falling back to `matrix_isa=none`. AMDGPU builtin
  differential coverage now includes bounded i64/u64 and exact-wide
  signed/unsigned sparse-A RNS paths against the dense AMDGPU builtin path for
  the same sparse-shaped native input; it builds locally and skips on RDNA3,
  then executes on CDNA3 SMFMAC or RDNA4 SWMMAC targets. Benchmark schema
  validation now distinguishes finite-u8 sparse workspace/events from bounded
  and exact-wide RNS sparse workspace/events, including sparse-A value/index
  upload phases for RNS sparse captures. Sparse bounded, finite-u8, and
  exact-wide AMDGPU differential shapes now use non-multiple M/N with
  K-divisible-by-4 inputs, so CDNA3 SMFMAC and RDNA4 SWMMAC runtime gates cover
  sparse tile tails instead of only exact 16x16 output tiles. Review now blocks
  sparse runtime rows unless the selected kernel is a sparse-A kernel, AMDGPU
  builtin metadata reports structured 4:2 SMFMAC/SWMMAC family evidence, the
  metadata names the explicit sparse-A contract, dense-B operand, canonical
  low2/high2 compression index layout, and matching public value signedness,
  and dense sparse-input baselines remain separated from sparse runtime kernels.
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
- Local progress: benchmark schema validation now rejects `_research_`
  AMDGPU builtin selected kernels as executed runtime captures, so RDNA4 IU4
  SWMMAC metadata cannot accidentally count as implemented sparse runtime
  evidence.
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
- Local progress: RDNA3 IU4 WMMA remains registry/calculator research metadata
  only; schema validation rejects it as an executed AMDGPU builtin runtime
  capture until a real INT4 semantic contract and runtime gate exist. RDNA
  dense WMMA/SWMMAC metadata now records the integer `NEG[0]` and `NEG[1]`
  signedness policy plus the `NEG[2]`/`NEG_HI` zero constraint, matching the
  current signed-centered operand use in runtime builtins.
- Constraint: no sparse RDNA3 runtime claim.


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
- Local progress: `resident-lifetime-arena` now carries signed and unsigned
  bounded coverage at 512/1024/2048 plus signed and unsigned exact-wide
  residue-current chain coverage at 256/1024. These rows remain explicit
  evidence-only gates until setup-inclusive allocation counters are proven on
  MI300X.
- Required evidence: `resident-lifetime-arena`, repeated-B, graph, and chain
  captures with setup-inclusive allocation metadata.
- Constraint: no hidden global singleton cache; identity and lifetime must be
  explicit.


### Rank 105 - Residue-Current And Lazy Final Export

- Priority: P2.
- Target: RNS chains and final-output workflows.
- Problem: exporting to host after every GEMM wastes work when the next
  operation consumes RNS residues.
- Implementation scope: residue-current output routing, lazy final export,
  chain metadata, cache identity, and clear next-op contracts for RNS GEMM,
  native GEMM, final export, and reuse-B.
- Local progress: canonical `rns-chain` and `rns-chain-final-output` rows are
  release-mode evidence, while the broader 512/1024 exact-wide final-output
  matrix remains the larger-shape companion. This keeps lazy residue-current
  and same-final-output comparisons visible in ordinary release reviews.
  `residue-channel-fusion` is now release-mode Direct-HIP evidence for width-3
  bounded final-export fusion, keeping the fused-residue route comparable to
  ordinary final-output paths. The compact `exact-wide-output-chain` family is
  also release-mode evidence for chain3 residue-current and reusable-B lazy
  export contracts.
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
- Local progress: `computational-algebra-proxies` is release-mode evidence for
  dense finite-field BLAS, rank-k update, F4 dense matrix, FGLM
  multiplication-matrix, and exact CRT/Garner export proxy phases. The scenario
  metadata keeps these as computational-algebra proxy phases rather than
  CAS-wide correctness claims.
  `finite-generic-moduli` now contributes release-mode feature-boundary rows
  for generic modulus 127 prime field/ring and generic composite modulus 253
  at 512-square shape; the 2048 generic-modulus probes remain smoke-only
  exploratory evidence.
- Required evidence: `finite-distributions`, `finite-modulus-map`,
  `finite-generic-moduli`, and large finite release captures.
- Constraint: modulus-specific wins must not be generalized to other moduli.


### Rank 110 - Adaptive Prefix And Zero-Skip Expansion

- Priority: P2.
- Target: bounded adaptive per-tile workloads.
- Problem: adaptive prefix and zero-tile skip can remove residue planes and
  work, but only when tile bounds prove it deterministically.
- Implementation scope: improve per-tile bound computation cost, prefix grouping
  execution, zero-row/zero-column/output skip, and tile metadata reuse.
- Local progress: `adaptive-bands` is now release-mode evidence with
  input-scan bound source, per-tile bound mode, and required adaptive execution
  for signed and unsigned bounded shapes. `bound-discovery` is also release-mode
  evidence, preserving static-profile baselines, global input-scan candidates,
  and per-tile proof-mask candidates for the same adaptive-band input family.
- Required evidence: `adaptive-bands`, `adaptive-grouped-scheduler`,
  `bound-discovery`, and zero-skip scenarios with exact proof metadata.
- Constraint: no probabilistic early termination in production.


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
  `--capture` arrays. `benchmark_sweep.py --list-scenarios` now prints the
  supported scenario catalog as JSON, and `benchmark_sweep.py --dry-run` writes
  `command_plan.json`, `command_plan.txt`, and the scenario manifest without
  executing or reviewing captures. That gives VM work a supported preflight path
  for release-candidate/all-scenario command inspection instead of ad hoc shell
  expansion or guessed scenario names. Release-candidate dry-runs now include a
  `release_readiness` block in stdout and `command_plan.json`, plus visible
  warnings in `command_plan.txt`, whenever the planned candidate captures are
  still configured as smoke-mode, below the release warmup floor, or below the
  release repeat floor.
- Constraint: diagnostics must not change benchmark semantics or promotion
  strictness.


### Rank 117 - Fix Persistent Small GEMM And Pack Kernel Dispatch

- Priority: P0.
- Target: all small-shape Direct HIP workflows.
- Problem: persistent small GEMM and coalesced/persistent pack kernels are
  compiled and have extern wrappers, but wiring them into the dispatch path
  causes one-shot test regressions (all-zero output for persistent GEMM,
  wrong output for coalesced pack). The kernels themselves have correctness
  bugs that need debugging before dispatch can be enabled.
- Implementation scope: write targeted differential tests for each kernel at
  minimal shapes (16x16 for GEMM, 64x64 for pack). Compare cell-by-cell
  against CPU reference. Fix kernel bugs. Then wire dispatch with minimal
  threshold (m*n <= 64 for GEMM, m*n <= 256 for pack). Incrementally expand
  the threshold as correctness is proven.
- Required evidence: differential tests passing for all three kernel families.
  One-shot and persistent bounded tests must pass with dispatch enabled.
  Sweep must show 0 new failures.
- Unlocks: all other dispatch wiring (coalesced pack, persistent GEMM, fused
  GEMM+export). This is the single highest-priority item.

### Rank 118 - RDNA3 Async Multi-Stream GEMM Plane Parallelism

- Priority: P0.
- Target: prefix >= 6 Direct HIP GEMM.
- Problem: per-modulus GEMM planes are independent but execute serially on
  one HIP stream. RDNA3 has 2 Shader Engines capable of concurrent execution.
  Launching plane groups on separate streams can overlap GEMM computation,
  reducing GEMM time by up to 50% for prefix >= 6.
- Implementation scope: partition modulus planes into 2 groups. Launch each
  group on a separate HIP stream. Use HIP events for synchronization between
  groups and before export. Select between 2-way and serial based on
  hipGetDeviceProperties.multiProcessorCount >= 48 (RDNA3 has 48 WGP CU).
- Required evidence: differential test comparing multi-stream output vs
  single-stream output for prefix-6 and prefix-9 bounded shapes. Phase
  timing showing GEMM reduction. No correctness regression.
- Expected win: 30-50% GEMM reduction for prefix >= 6 shapes.

### Rank 119 - Zero-Copy Host I/O Via ReBAR Mapped Memory

- Priority: P0.
- Target: pack H2D and export D2H phases.
- Problem: pack H2D is 38-66% of end-to-end time on bounded shapes. Export
  D2H is 10-20%. With ReBAR (Resizable BAR), the GPU can directly access
  host memory via hipHostRegister + hipHostGetDevicePointer. This
  eliminates the explicit hipMemcpy for pack and export, replacing it with
  zero-copy kernel accesses.
- Implementation scope: probe ReBAR availability via hipDeviceGetAttribute
  with hipDeviceAttributeCanAccessPeer and hipDeviceAttributeDirectManagedMemAccessFromHost.
  If available, register host matrices with hipHostRegister and pass device
  pointers to pack/export kernels. Kernels read/write directly.
- Required evidence: differential test with zero-copy vs explicit copy paths.
  Timing comparison showing H2D and D2H reduction. Must not regress on
  systems without ReBAR.
- Expected win: eliminate 80-100% of explicit pack H2D and export D2H time.

### Rank 120 - Kernel Fusion HIP Graph Auto-Capture

- Priority: P0.
- Target: repeated workloads.
- Problem: every resident GEMM call incurs pack, GEMM, and export kernel
  launch overhead. HIP graphs can capture the entire pipeline on first call
  and replay with a single hipGraphLaunch on subsequent calls. Graph
  infrastructure exists (Ranks 86-87) but isn't auto-triggered.
- Implementation scope: on first GEMM call for a plan+workspace combination,
  wrap the entire pack/GEMM/export sequence in hipStreamBeginCapture /
  hipStreamEndCapture. Store the graph in the workspace. On subsequent
  calls, hipGraphLaunch instead of individual kernel launches. Invalidate
  the graph on source-version change.
- Required evidence: differential test with graph vs non-graph paths.
  Benchmark comparing per-call overhead. Must preserve exact correctness
  and source-version semantics.
- Expected win: 20-40% end-to-end reduction for repeated workloads on small
  shapes; 5-15% on large shapes (amortized over GEMM time).

### Rank 121 - Auto-Tuning Kernel Scheduler

- Priority: P1.
- Target: replacing manual dispatch heuristics with data-driven selection.
- Problem: current kernel dispatch uses hardcoded thresholds (if m*n <= X
  use kernel A else kernel B). An auto-tuning scheduler can sweep
  threadblock sizes, K-block policies, LDS configs, and register budgets
  per shape family, then store winners in the autotune cache for production
  selection.
- Implementation scope: extend 	ile-shape-sweeps scenario infrastructure
  to cover all kernel dispatch parameters. Run sweep on CI. Store winners
  in utotune.json. At plan creation, query autotune cache for the best
  kernel identity. Fall back to default on cache miss.
- Required evidence: auto-tuned kernel selection producing >= same performance
  as manual heuristics on all swept shapes. No correctness regression.
- Promotion rule: auto-tuned selection must never regress vs manual.

### Rank 122 - Occupancy-Driven Kernel Dispatch

- Priority: P1.
- Target: optimal kernel selection per GPU.
- Problem: kernel performance depends on occupancy (waves per CU), which
  depends on VGPR/SGPR/LDS usage. Different GPUs have different resource
  limits. At plan creation, the scheduler should select the kernel variant
  with the highest theoretical occupancy for the target GPU.
- Implementation scope: query gpu_isa_report.py output for VGPR/SGPR/LDS
  per kernel variant. Compute occupancy from GPU specs. Select kernel with
  max occupancy. Cache selection in autotune key.
- Required evidence: occupancy computation matches hipOccupancyMaxActiveBlocksPerMultiprocessor output.
  Selected kernels maintain correctness.
- Promotion rule: must not regress vs manual selection.

### Rank 123 - Zero-Copy Accelerator Paths

- Priority: P1.
- Target: rocWMMA and hipBLASLt backends.
- Problem: accelerator backends copy data to their internal layouts, adding
  conversion overhead. With ReBAR zero-copy, host data can be accessed
  directly by accelerator pack kernels, eliminating intermediate copies.
- Implementation scope: after Rank 119 proves zero-copy for Direct HIP,
  extend the pattern to rocWMMA and hipBLASLt pack paths. Native B-cache
  constructors (Rank 84) should write directly to ReBAR-mapped memory.
- Required evidence: differential test with zero-copy accelerator pack.
  Timing comparison vs explicit copy path.

### Rank 124 - rocWMMA HIP Event Labels

- Priority: P1.
- Target: rocWMMA backend benchmarking.
- Problem: rocWMMA captures lack GPU event timings, preventing phase
  attribution and blocking the repeated-b benchmark. Event scopes are
  registered in the schema; recording calls need to be added to the backend.
- Implementation scope: add 	imed_hip_operation calls with rocWMMA-specific
  event labels (
ocwmma_pack_kernel, 
ocwmma_gemm_kernel_group,
  
ocwmma_epilogue_kernel) in 
ocwmma_backend.cpp and
  
ocwmma_backend_rns_wrappers.inc. Ensure event labels survive through
  the xtern "C" kernel call boundary.
- Required evidence: GPU event report showing rocWMMA event labels available.
  Repeated-b benchmark passing schema validation with events.

### Rank 125 - rocWMMA Prepack-B Cache Benchmark

- Priority: P1.
- Target: bounded i64/u64 512 and 1024 shapes.
- Problem: rocWMMA has existing wins on bounded u64 512 (1.17x) and 1024
  (1.17x) without prepack caching. With cached B operands, pack-A overhead
  drops from ~50% to ~10% of e2e. The native B-cache constructors exist
  (Rank 84). This rank measures setup-inclusive break-even.
- Implementation scope: run enchmark_sweep.py --scenario repeated-b after
  Rank 124 completes. Measure rocWMMA with --reuse-packed-b against
  same-backend non-reuse and fastest non-reuse baseline. Compute break-even
  repeat count.
- Required evidence: schema-valid captures with GPU events. Setup-inclusive
  break-even analysis. Promotion only if beats both same-backend non-reuse
  and fastest contract non-reuse.
- Promotion rule: promoted cache entries must carry setup-inclusive evidence
  and explicit reuse contract metadata.

### Rank 126 - WMMA Tile-Sweep Variant Dispatch

- Priority: P1.
- Target: AMDGPU builtin skinny GEMV paths.
- Problem: three WMMA tile-sweep kernel variants (64t, 128t, 256t) are
  compiled but not selected by the scheduler. Each targets different occupancy
  profiles (N=1, N=4, N=8). Wiring them into the AMDGPU builtin dispatch
  with shape-based selection could improve skinny GEMV wins by 5-15%.
- Implementation scope: in mdgpu_builtins_backend.cpp, add shape-based
  kernel selection. When N==1 select 64t, N<=4 select 128t, N<=8 select 256t.
  Record selected variant in backend metadata. Add schema fixtures for new
  kernel identities.
- Required evidence: ISA report showing correct kernel selected per shape.
  Differential tests for each variant. Sweep showing improved skinny GEMV.

### Rank 127 - INT4/IU4 Matrix Engine Proof-Of-Concept

- Priority: P2 (Research).
- Target: RDNA3 WMMA iu4 operations.
- Problem: RDNA3 WMMA supports 4-bit integer matrix multiplication. At half
  the memory bandwidth of INT8, IU4 could win on memory-bound large shapes.
  Research infrastructure exists (semantics enum, pack kernel).
- Implementation scope: add 
ns8_amdgpu_builtin_wmma_iu4_gemm_research
  kernel using __builtin_amdgcn_wmma_i32_16x16x16_iu4. Schema-gate behind
  _research_ prefix in selected_kernel. Add ISA histogram proving IU4
  instructions. Do not route to default paths.
- Required evidence: ISA report showing v_wmma_iu4 instructions. Differential
  test comparing IU4 path against INT8 path. Benchmark on 2048/4096 shapes.
- Constraint: research only. No default routing, no cache promotion.

### Rank 128 - Python Kernel Generator

- Priority: P2 (Infrastructure).
- Target: eliminating hand-written per-modulus specialization.
- Problem: the default modulus ladder has 28 values. Each specialized reducer,
  pack variant, and CRT constant table is hand-written. A code generator can
  produce specialized kernels for each modulus family (hot 251/255/256,
  prime-only, composite-only) with known reductions at compile time.
- Implementation scope: 	ools/generate_modulus_kernels.py reads the modulus
  ladder from metadata, generates specialized .cuh files with
  __device__ constexpr modulus constants and precomputed Garner weights.
  Output goes to src/generated/. Integrate into CMake build.
- Required evidence: generated kernels produce identical output to generic
  kernels. ISA report shows eliminated runtime modulus lookups.
- Constraint: generated files must not be committed; they are build artifacts.

### Rank 129 - Interactive Performance Dashboard

- Priority: P2 (Infrastructure).
- Target: benchmark data exploration.
- Problem: sweep results are in JSON files under 	emp/. Comparing shapes,
  backends, and phases requires ad hoc Python scripts. An interactive HTML
  dashboard makes the data explorable.
- Implementation scope: 	ools/generate_performance_dashboard.py reads sweep
  captures, generates a single docs/dashboard.html with sortable tables,
  phase breakdown charts, backend comparison bar charts, and shape × backend
  heatmap. Host on GitHub Pages. Update on every sweep commit.
- Required evidence: regenerated dashboard matches sweep data. All links and
  charts render correctly.
- Constraint: static HTML only. No server required.

### Rank 130 - Continuous Benchmark Performance CI

- Priority: P2 (Infrastructure).
- Target: automated regression detection.
- Problem: performance regressions are discovered manually during sweeps.
  Automated CI that runs a lightweight benchmark matrix on each push can
  catch regressions before they reach the sweep.
- Implementation scope: add .github/workflows/perf-regression.yml that
  builds HIP debug, runs 10 key shapes × 3 backends, compares against
  stored baseline JSON, and fails if any shape regresses >5%. Baseline
  updates require manual approval.
- Required evidence: CI workflow completes successfully. Regression detection
  correctly identifies intentional slowdown in test fixture.
- Constraint: CI runs on self-hosted runner with gfx1100 GPU.

### Rank 131 - Adversarial Input Detection

- Priority: P3 (Research).
- Target: worst-case input patterns.
- Problem: some signed input distributions cause alternating large
  positive/negative accumulation that can overflow the INT32 accumulator
  within the standard K-block size. Detecting these patterns and adjusting
  prefix or K-block prevents silent overflow.
- Implementation scope: input scanner in plan creation that checks for
  alternating large-magnitude sign patterns. If detected, increase
  min_required_prefix by 1 or halve K-block size. Record in metadata.
- Required evidence: adversarial input fixture that triggers overflow with
  default prefix. Adjusted prefix prevents overflow. Exact CPU parity.
- Constraint: must not trigger false positives on normal inputs.

### Rank 132 - Power-Aware Scheduling

- Priority: P3 (Research).
- Target: sustained throughput under thermal limits.
- Problem: gfx1100 throttles GPU frequency under sustained load. A scheduler
  that monitors temperature/frequency and adjusts launch rate can maintain
  peak frequency for longer, improving sustained throughput.
- Implementation scope: query GPU frequency via 
smi_dev_gpu_clk_freq_get
  or 
ocm-smi. If frequency drops below sustained threshold, insert
  hipDeviceSynchronize + brief sleep between launches. Record power
  metadata in captures.
- Required evidence: sustained sweep showing higher average frequency and
  improved throughput vs unthrottled baseline.
- Constraint: power data collection requires Radeon Developer Tools or
  Linux ROCm profiling tools. Windows gfx1100 support may be limited.

### Rank 133 - Mixed-Precision Modulus Family Scheduler

- Priority: P3 (Research).
- Target: routing modulus families to their fastest backend.
- Problem: hot moduli (251, 255, 256) have specialized CK/rocWMMA kernels
  that are 2-3x faster than generic. The default ladder interleaves hot and
  cold moduli. A scheduler that groups tiles by modulus family and routes
  each group to the optimal backend can improve overall GEMM time.
- Implementation scope: at schedule creation, classify modulus planes into
  families. Route hot-modulus groups to CK/rocWMMA, cold-modulus groups to
  Direct HIP. Launch groups on separate streams if available.
- Required evidence: mixed-backend GEMM produces same output as single-backend.
  Phase timing shows GEMM improvement. No correctness regression from
  backend mixing.
- Constraint: must not mix backends across tiles that depend on each other's
  CRT reconstruction state.

