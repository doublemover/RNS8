# Performance Gain Queue Dispositions

This file holds queue-adjacent material that should stay out of
[performance-gain-work-queue.md](performance-gain-work-queue.md). The work
queue is the active execution control panel; this file tracks future platform
scouts, validation debt, and paths that should not be chased next.

## Future / Platform Scouts

These rows stay tracked but are no longer near-term active queue priorities.

| Rank | Scout | Why It Is Deferred | Return Gate |
|---:|---|---|---|
| 71 | 8192 GPU-only throughput scout | 4096 already proves most current launch/export/throughput behavior, and 8192 mostly adds memory-pressure, timeout, thermal, and classification risk | Reopen only for explicit memory-pressure/scale evidence or when a budgeted CPU/reference method exists |
| 76 | Multi-GPU sharding and device-concurrency platform lane | Single-GPU Windows `gfx1100` still has higher-value grouping/export/reuse work, and Linux/CDNA validation is not available locally | Reopen after real Linux/RDNA/CDNA target validation and a concrete topology/cross-device workload gate |

## Validation Debt

| Debt | Why It Matters | Required Refresh |
|---|---|---|
| Native-to-RNS, vector-to-RNS, and exact-wide residue-chain captures are helper/workload surfaces, not routing proof | The branch can expose and validate bridge/chain scenarios, and exact-wide Direct-HIP chain captures now have release-mode event timing plus a focused final-output CPU comparison report, but AUTO/public routing still needs same-output independent-call/export-repack wins | Release review for bridge and chain scenarios with explicit conversion timing, reuse setup cost, final export timing, exact CPU comparison for the requested output, and an independent-call baseline when the workload contract claims skipped intermediate export |
| Many-small grouped public routing remains benchmark-owned | The current matrix now includes release-reviewed host-batch proof plus exact-wide signed/unsigned 64 and 128 group32, bounded-i64/u64 64 group32, bounded-i64 128 group64, bounded-u64 128x1x1024 group128, and finite-ring u8 64 group32 candidate wins; these rows use grouped pack, same-shape Direct-HIP grouped GEMM, compact grouped export, one compact output D2H, schema-validated benchmark task descriptors, and internal same-contract bucket descriptor checks | Add public grouped pack/export lifetime contracts, broader output/currentness policy, durable workload-family variance proof, and real target-validation evidence before promoting grouped or batched GPU candidates beyond benchmark-owned same-shape workload evidence |
| HIP Graph replay is implemented as a narrow benchmark lane, not a promoted workload contract | The branch-local graph path is deliberately scoped to Direct-HIP resident RNS chains and records wall-clock graph launch timing instead of normal per-kernel GPU event timing; schema/sweep/build/tiny smoke evidence now exists | Run release-size captures against the same non-graph chain, include capture/instantiate setup cost, and keep the result experimental unless it beats the same-contract non-graph path end-to-end |
| Large 2048/4096 captures are now split between installed non-reuse wins and explicit follow-up contracts | Bounded i64/u64 2048/4096, finite-u8 hot-modulus 2048/4096, exact-wide signed/unsigned 2048/4096, and strict wrap64 2048/4096 now have CPU/reference-backed release evidence where required. Eligible bounded/finite/exact-wide non-reuse winners are installed where AUTO cache promotion is valid; repeated-B is still contract-limited and wrap64 is a Direct-HIP correctness path rather than cache promotion | Keep repeated-B under the reuse workload ranks until setup identity/lifetime policy is explicit; keep strict wrap64 tuning under rank 68; do not generalize Windows `gfx1100` rows to Linux or Instinct |
| Reuse/prepack wins use explicit reuse contracts | The branch now has a release-contract A/B/A+B matrix, but those captures intentionally change input lifetime and setup semantics versus one-shot calls | Convert only explicit reusable-input workloads with setup-inclusive break-even, source identity, stale-input rejection, and caller-visible lifetime metadata; do not install AUTO cache entries from reuse captures |

## Do Not Chase Next

| Path | Current Disposition | Reason |
|---|---|---|
| Vector 1024 repeated-A and full A+B reuse | Deprioritize | Latest reuse comparison regressed setup-inclusive and steady-state timing |
| hipBLASLt 1024 repeated-A and full A+B reuse | Deprioritize | The full reuse-contract matrix loses to the fastest non-reuse baseline after setup cost despite earlier same-backend mechanism wins |
| rocWMMA 1024 repeated-B reuse | Deprioritize | Latest reuse comparison lost after setup cost |
| Wrap64 rocWMMA matrix-engine candidate | Deprioritize | Correct but loses to Direct-HIP v4 at every reviewed 64/128/512/1024 shape |
| Wrap64 Direct-HIP colpair experiment as default route | Deprioritize | Narrow GEMM improvement did not beat v4 end-to-end |
| Direct-HIP resident selected-prefix colpair as default route | Deprioritize | The 512 bounded-i64 rerun showed unstable GEMM outliers and lost end-to-end despite a narrower GEMM-median signal |
| Wrap64 pinned export staging as default route | Deprioritize | Forced staging lost badly at 512 versus default policy |
| CK/rocWMMA v1 bounded/exact-wide cache promotion | Do not promote | Selected-kernel identities are stale under current v2 reducer paths |
| CK/rocWMMA adaptive current-v2 cache promotion | Do not promote | Current adaptive-bands release review is valid, but CK and rocWMMA lose badly to Direct HIP at every reviewed group |
| Raw smoke or discovery captures as durable claims | Do not promote | They lack the release-review and required-event gates for public claims |
| Single-call accelerator tuning for many-small 32/64 proxies | Deprioritize | The current release review is CPU- or Direct-HIP-favored for the small proxies, and the only vector-fast bounded-u64 64 result is not an accelerator/cache entry; grouped execution is the higher-value path |
| Current bound-discovery/proof-mask modes as default routing | Deprioritize | The setup-inclusive release matrix found zero candidate wins; per-tile proof masks are event-visible but exact tile-bound scans dominate end-to-end timing |
| AMDGPU builtins, INT4/IU4, FP8/Ozaki hybrids, Strassen, and generic sparsity | Research archive only | These need a concrete correctness kernel, explicit workload structure, or measured bottleneck that current CK/rocWMMA/Direct-HIP paths cannot answer before they deserve active execution slots |
