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
The active table below now contains 0 ranks. Rank IDs are historical/stable
references; row order is the current execution priority. Non-active material
lives outside this file so the control panel stays execution-focused.

No active performance-queue ranks remain before the next real Linux ROCm/CDNA
validation pass. New performance work should start from fresh evidence produced
by `scripts/cdna_first_pass.sh`, target-validation reports, profiler/counter
reports, and same-target release captures, then add a new active row only when
there is a concrete implementation or validation gap to close.

Rank 78, the public incremental result-cache contract and promotion gate, was
opened and closed on June 7, 2026; its durable status lives in the completed
work archive, wins doc, reviewed local evidence, and work log.
