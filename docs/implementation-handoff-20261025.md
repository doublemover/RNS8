# Parked implementation handoff — 2026-10-25

This is the original parked source snapshot. Subsequent development and the
[consolidated local validation](local-validation-20261025.md) supersede its
execution-deferred status: local applicable checks now pass, six tooling issues
are closed, and the report records the remaining source/hardware/performance
gates. The per-issue ledger retains the historical counts below alongside its
new `validation_closeout` record. Later #68 work includes retained B, overlap
and calibrated panel shortlisting; later #40 work includes paired-column
layout algebra and independent operand reuse controls.

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
| #32 | Explicit bases now compose with uniform layouts, product recipes and typed provider inventory; builds/execution/calibration remain unqualified. |
| #40 | GF(2) algebra/paired-column consumer, rocWMMA direct-A/call-local reuse and hipBLASLt independent retention controls are authored; execution/measurement deferred. |
| #41 | Default and nondefault conversion/lift paths and product-recipe binding are authored; execution/measurement deferred. |
| #42 | NativeToRns/BaseExtend/Export graph adapters, ragged input adapters and prefix cost selection are authored. General packed representation selection remains source work. |
| #43 | Matched pinned-host sync/stream/graph calibration is authored. Remaining provider-specific async/capture support stays explicit rather than inferred. |
| #45 | Actual FP4/FP6/FP8 matrix-engine provider and public representation bridge; contracts alone are not a backend. |
| #50 | Real multi-device HIP owners/executor, full gather and whole-output transactional commit. |
| #52 | Verified compiler/ISA-specific CDNA5 TDM/cluster-barrier litmus, two-stage transfer and multicast kernels; do not invent target interfaces. |
| #56 | Complete the unfinished child implementation scopes; roadmap text is not implementation. |
| #68 | Serial/retained-B/three-stream execution and identity-bound panel shortlisting are authored; correctness, real overlap/capacity and held-out calibration await validation. |

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
