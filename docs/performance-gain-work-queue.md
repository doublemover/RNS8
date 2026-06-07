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
The active table below now contains 8 ranks. Rank IDs are historical/stable
references; row order is the current execution priority. Non-active material
lives outside this file so the control panel stays execution-focused.

| Rank | Work Item | Why Now | Evidence Gate | Disposition Rule |
|---:|---|---|---|---|
| 60 | Advanced promotion ledger adoption and cache-install gate | Installed reviewed cache entries need durable auditability | `tools/promotion_ledger.py` now records target-validation group, target/cache eligibility, stale invalidation reasons, variance state, and coverage summaries; `tools/install_autotune_cache.py` now records add/replace history, cache coverage, `--require-target-validation-gate`, and automatic CDNA target-gate enforcement | Do not install or replace cache entries without ledger consistency, target validation where required, variance gates for narrow lanes, and claim validation |
| 63 | Verification amortization and real FHE/lattice workload suite | Exact repeated validation and FHE/lattice-inspired workloads need realistic contracts | Add CKKS/BFV/BGV-like NTT, key-switch, relinearization, rotation, ModUp/ModDown/rescale, bootstrapping-stage, tower reuse proxies, and safe verification amortization | Keep as workload/proxy evidence, not cryptographic correctness or library support claims |
| 69 | CPU small-shape optimized fallback and selector thresholds | The many-small review shows CPU wins several tiny exact workloads | CPU microbench and selector A/B for bounded-i64 32, bounded-u64 64, finite-u8 64, cache locality, threading policy, vectorized host paths, and cutoff thresholds versus GPU paths | Route to CPU only when same-contract release evidence beats GPU paths and selector explanations stay explicit |
| 72 | Vector/native-output-to-RNS fused producer-consumer path | The bridge surface exists, but extra materialization can erase vector wins | Release A/B for vector/native producer output feeding Direct-HIP RNS consumers with fused conversion, reusable-B setup, final-output comparison, and selected-kernel/currentness metadata | Promote only when the chain beats native host export plus repack and preserves exact final-output checks |
| 73 | Finite-u8 data-distribution release matrix | Current finite wins are modulus-heavy; input distribution may change reducer, pack, and CPU/GPU cutoffs | Release matrix for binary, sparse, low-Hamming, small-centered, and full-uniform finite-ring/field inputs across hot and generic moduli at 128/512/1024/2048 | Keep distribution-specific routing explicit; never infer finite workload structure from modulus alone |
| 74 | Split-K and K-block large-shape variants | Large bounded/exact-wide throughput may be limited by K-block policy, accumulator caps, and schedule upload | `rns8-bench` and scenario sweeps now carry benchmark-only K-block policy metadata, tile-shape captures include split-K mode, accumulator-safety key, and resource-report requirements, `tools/tile_shape_report.py` blocks non-default policies pending counter evidence, and `k_block_tile_variants.json` covers bounded, exact-wide, finite-u8, and wrap64 at 1024/2048/4096. Multi-GPU split-K remains out of scope | Promote only per semantic/backend/target when selected-kernel and autotune keys encode the K-block contract |
| 75 | Result cache and incremental GEMM research lane | Repeated exact workloads may reuse intermediate products or partial results, but that is workload-specific and easy to overclaim | Research-only captures for source identity, dirty-region metadata, partial recompute, result lifetime, and exact final CPU comparison across repeated workloads | Keep out of default GEMM and AUTO until caller-visible mutation/version contracts make reuse exact and auditable |
| 77 | Layout implementation search after scenario surface | The layout-search scenario family is closed as a surface, but no layout implementation has won the end-to-end gate | Implement and benchmark residue-plane interleave, leading-dimension policy, packed-residue layout, output-current layout, finite layout, and wrap64 byte layout variants with pack/GEMM/export attribution | Keep layouts only when complete same-contract release evidence beats current layout including conversion/setup cost |
