# Parked implementation handoff — 2026-10-25

Subsequent development resumed with validation deferred. The #68 serial owner now
has a bounded retained-B cache, conflict-aware reload accounting, and authored
regressions/calibration controls. Serial export now disables unbudgeted nested pinned staging. Overlap and general
cost selection remain unfinished. Counts below describe the original parked
snapshot; the per-issue ledger records this later partial implementation.

Development was parked at the user's request, **not because the backlog is
finished**. This branch starts at `5a6c757dc7ecb3a7711248ae8b1de0406a2a8c70` on
`codex/runtime-cache-transactions-20261011`. It changes no main/source-branch
history and does not publish qualification outcomes.

## Scope and honest completion boundary

The durable [per-issue ledger](../metadata/remaining-implementation-20261024.json)
tracks the 51-issue inventory, implemented source, regression paths, dependencies,
commits and remaining work. Its source assessment is:

- **39 source-complete, validation deferred.** This is not 39 verified fixes or
  permission to close evidence/outcome issues.
- **6 advanced but partial:** #30, #32, #40, #41, #42, #43.
- **5 remaining implementation scopes not advanced in this pass:** #45, #50,
  #52, #56, #68. Some have baseline planners/contracts/serial implementations;
  “not advanced” does not mean their repository contains no groundwork.
- **1 qualification-only outcome:** #60.

Thus **11 issues still require development**. All newly authored test execution,
benchmarks, GPU campaigns, complete HIP builds and scientific qualification are
still deferred. No speedup or target-wide correctness claim follows from source.

## Delivered source clusters

- Ownership/publication: retryable/deferred HIP cleanup, generation-bound cache
  and prepared ownership, status-gated exact lift/native transfer pipelines,
  failure leases and explicit pinned-host CPU/DMA access windows. Data-ready,
  proof-ready and current-value publication remain distinct.
- Planning/composition: exact-integer arithmetic certificates, shared typed
  operation admission and cost shortlists, dependency-proven scratch lifetimes,
  ragged materialization into shared multi-output recipes, and explicit
  current-value fixed-basis plans. Default full-native plan admission is retained;
  execution rechecks actual engine-owned proofs.
- Arithmetic/backend work: native persistent/grouped queues and locality controls,
  narrow fused products, real INT4 WMMA, rank-23 straight-line bilinear methods,
  native/transpose/mirror/CRT Gram variants, signed-radix CDNA MFMA, bounded-exact
  FP64 modular products, and an isolated opt-in Linux x86 AMX-INT8 backend.
- Measurement infrastructure: matched public/private controls, independent
  integer oracles outside timing, sustained/counterbalanced collection,
  source/artifact/instruction identities, strict receipt admission and explicit
  resource/copy boundaries. These are enabling implementations, not results.

The last bounded edit repaired a cross-feature admission inconsistency: short
fixed-basis calibration/planner cases now request an explicit value-bound plan
policy instead of being rejected by the full-native default. Descriptor and
schedule sealing share one flag mask, including the existing research controls.

## Exact remaining implementation

| Issue | Remaining source work |
| --- | --- |
| #30 | Finish provider/binding parity as unfinished representation, distributed and asynchronous owners land; do not infer implementation from a registry record. |
| #32 | Runtime experimental ordered bases, products/inverses and full matrix/layout/plan/cache/export identities; pack, multiply, reconstruction and explicit conversion. |
| #40 | General fragment algebra and GF(2) paired-column consumer; rocWMMA direct-A/panel reuse; hipBLASLt none/A/B/both retention and composed controls. |
| #41 | Nondefault-basis conversion integrated with #32; default-basis synchronous/prepared lift and selector are written. |
| #42 | General packed/materialization alternatives with #40; asynchronous NativeToRns/BaseExtend/Export compiled graph-node adapters with #43. |
| #43 | Those compiled graph-node adapters, additional semantic/provider async integration, and matched pinned-host transfer calibration. |
| #45 | Actual FP4/FP6/FP8 matrix-engine provider and public representation bridge; contracts alone are not a backend. |
| #50 | Real multi-device HIP owners/executor, full gather and whole-output transactional commit. |
| #52 | Verified compiler/ISA-specific CDNA5 TDM/cluster-barrier litmus, two-stage transfer and multicast kernels; do not invent target interfaces. |
| #56 | Complete the unfinished child implementation scopes; roadmap text is not implementation. |
| #68 | Overlapped streaming and calibrated panel selection; bounded retained-B reuse and serial export budget repair are authored, execution deferred. |

Resume dependency-first: #32→#41 and #40/#43→#42, then remaining independent
accelerator/distributed/streaming slices. Preserve a source-completion ledger
separate from hardware outcome evidence. Newly exposed source defects should be
reconciled rather than hidden by a previously recorded completion count.

## Checks and unverified boundaries

Lightweight host C/C++ syntax checks, Python byte-compilation and repository
metadata/documentation checks were used during authoring. The final value-bound
edit received host syntax checks for its core, unit-test translation unit and
six affected calibration programs. AMX additionally received compile-only
instruction emission inspection earlier; no AMX instructions were executed.
HIP-gated code was not compiled by these host-only checks. No test suite,
benchmark, GPU workload or scientific qualification was run in this pass.

No background campaign, deployment, merge or further implementation is left
running by this handoff. The draft PR is a parked review/backup boundary, not a
ready-to-merge or all-issues-complete claim.
