# Performance Gain Imported Research Notes

This document preserves imported maximum-performance roadmap triage notes that
used to live in the active work-queue archive. Keep active execution priority
in [performance-gain-work-queue.md](performance-gain-work-queue.md), closed
rank history in
[performance-gain-completed-work.md](performance-gain-completed-work.md), and
dated status updates in
[performance-gain-work-log.md](performance-gain-work-log.md).

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
