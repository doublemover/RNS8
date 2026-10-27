# Current roadmap and release gates

This page owns remaining execution gates. The [technical specification](RNS8_RESEARCH_SPEC.md)
owns normative architecture/acceptance rules; [backend capabilities](backend-notes.md)
and the [research index](research/README.md) distinguish public paths from private
prototypes. The [implementation chronology](archive/implementation-checkpoints-2026-10-23.md)
preserves earlier work without duplicating current status.

## Current implementation pass

The [October27 implementation closeout](../metadata/implementation-closeout-20261027.json)
records the requested source milestones for #32, #43, #45, #48, #50, #54, #63 and
#67 at compiled source `cf0ad32`. The receipt identifies the first batch's failures,
their consolidated repairs, the final CPU-ASan/HIP results, installed DLPack
consumers and compile-only instruction evidence. No new performance campaign ran.

| Issue | Implemented extension | Remaining qualification |
|---|---|---|
| #32 | Prepared ordered-basis native transfers and exact basis conversion/lift | Broader targets, measured complete-pipeline basis/form selection |
| #43 | Individually bound library/builtins streams and explicit caller graphs | Native Linux/CDNA, real fault recovery, physical overlap and payoff |
| #45 | gfx942 FNUZ/gfx1201 OCP four/three-product FP8 recipes | Actual target lane/accumulator behavior, exact execution and complete-route timing |
| #48 | Certified one-level rank48 coefficient execution with classical even channels | Large-shape strongest-control performance/memory gates |
| #50 | Modulus, M/N, 2-D output and exact K-fragment HIP partitioning | Actual multi-GPU transport/failure/scaling qualification |
| #54 | Optional isolated Linux AArch64 I8MM owner beside AMX | Actual supported CPU/OS execution and end-to-end measurements |
| #63 | Prime-channel five-product Gram with composite-channel preservation | Broader targets and measured preparation/reuse/export benefit |
| #67 | Optional installed DLPack1.2 adapter and standalone HIP consumer | Native Linux, real failed drains and matched interoperability timing |

The [replay extension receipt](../metadata/replay-extension-validation-20261027.json)
separately records lifetime cost models, external packed-B reuse, bound recipes,
experimental-basis composition and full integer reference caching. Its counts and
measurements retain their original source identity. Implementation does not close
an issue's unfulfilled hardware or measured-performance acceptance gates.

## Recorded baseline

The local stack is TheRock Core10.1 / HIP7.16.26385 / AMDClang24, VS2026,
Python3.15.1 and Radeon RX7900XTX (`gfx1100`). Current compiled validation is
recorded above. Earlier Release, Debug, OpenMP-OFF, CPU-ASan and installed-package
checks remain in the [SDK record](../metadata/hip101-validation-20261026.json),
[configuration follow-up](platform-windows.md#october-26-configuration-follow-up)
and [replay receipt](../metadata/replay-extension-validation-20261027.json).

The [SDK10.1 bounded measurements](performance.md#october-26-sdk101-bounded-measurements)
preserve original and post-restart cohorts at source `72c55c2`, strongest tested
controls and fixed gates. These within-SDK results are not a matched prior/new-SDK
regression study and are not transferred to the new implementation. The stopped
5,382/16,513-run campaign remains stopped. The [implementation chronology](archive/implementation-checkpoints-2026-10-23.md)
and [historical index](archive/README.md) retain older checkpoints.

## 1. Source and CPU gate

Preserve exact semantics, deterministic default proofs, explicit unsupported
behavior, output invalidation on fallible writes and transactional preflight
rejection. Every change needs applicable CPU/Boost/API regressions; public
layout/ownership and installed static/shared consumers remain release checks.
Run relevant tooling tests and available sanitizers, record the source identity,
and keep required skips/failures visible. CPU evidence does not establish GPU
execution. Debug-STL allocation-fault sweeps retain their documented boundary.

## 2. RX 7900 XTX / gfx1100 gate

For changed kernels/configurations, rebuild direct HIP and each enabled optional
backend with the recorded toolchain, then run exact differentials and inspect
all skips. Required cases include full-width signed/unsigned values, prefixes
1–20, packing/pitch/wave tails, skinny shapes, finite composite moduli and long-K
accumulation splits. Check selected matrix instructions, export arithmetic,
resource use and prepared/DAG completion/lifetime behavior.

Re-establish matched timing when implementation or artifact identity changes.
The linked results cover specific workloads; independent variance, counters,
clock policy and held-out planner regret remain distinct acceptance requirements.
[Qualification](gpu-qualification.md) and [performance methodology](performance.md)
own the commands and evidence gates.

## 3. MI300X / gfx942 gate

Run native Linux ROCm compilation and direct-HIP CPU differentials on the actual
supported device first. Then qualify MFMA, CK, rocWMMA and hipBLASLt as available.
Cover wave64, dense/sparse target instructions, padded/tail shapes, finite
composite moduli and accumulation-cap rejection. Preserve target/OS/compiler,
ISA, resources, counters and independent-process timing. Use one explicitly
selected physical GPU before expanding to per-device checks.

Independent per-GPU shard scripts do not implement distributed GEMM. The new
[private modulus-sharded HIP owner](distributed-hip.md) supplies real runtime
source and applicable single-device Windows/gfx1100 regression coverage. Native
Linux and actual multi-device execution retain their qualification gates. RDNA2,
RDNA4, CDNA2 and CDNA4 require their own per-device/OS/backend campaigns using
AMD-supported stacks; the [hardware matrix](../README.md#hardware-tested-and-hardware-wanted)
lists validation requests.

## 4. Release decision

A release requires reviewed evidence for every shipped target/backend/contract,
no unexplained failures/skipped required tests, package/ABI checks on Windows
and Linux, reproducible matched performance review and explicit routing/cache
admission. Publish a support matrix with actual shape/target limits. Completing
a source change or documentation cleanup does not satisfy these gates.

## Research and integration gaps

The architecture backlog is tracked by [#56](https://github.com/doublemover/RNS8/issues/56).
Focused topic guides retain arithmetic proofs and remaining gates:

| Work | Current boundary / next acceptance | Issue |
|---|---|---|
| Conversion-aware planning and roofs | Bounded gfx1100 instruction/conversion/CRT/system and stage measurements are recorded with fixed held-out gates; unseen workloads, physical clocks/counters/resources and selection-regret qualification remain | [#30](https://github.com/doublemover/RNS8/issues/30), [#31](https://github.com/doublemover/RNS8/issues/31) |
| Bound discovery and variable prefixes | Device discovery/adaptive/ragged candidates, compact device-bucket workers and explicit matrix-owned adaptive API are authored; four-wave register/one-buffer/two-buffer native WMMA candidates are authored; summary reuse and measured cost tradeoffs remain | [#34](https://github.com/doublemover/RNS8/issues/34), [#35](https://github.com/doublemover/RNS8/issues/35) |
| Scheduling | Private queue/split-K and logical locality models; measured configuration selection and broader target work remain | [#36](https://github.com/doublemover/RNS8/issues/36), [#37](https://github.com/doublemover/RNS8/issues/37), [#38](https://github.com/doublemover/RNS8/issues/38) |
| Reconstruction/materialization | Public explicit CPU/HIP lift; private balanced/mixed-radix alternatives include an authored tile-major CRT consumer; performance and broader representation integration remain | [#39](https://github.com/doublemover/RNS8/issues/39), [#40](https://github.com/doublemover/RNS8/issues/40), [#41](https://github.com/doublemover/RNS8/issues/41) |
| Resident execution | Native/uniform/ragged/packed planning and retained arenas, direct-HIP tokens/dependencies, transfers/lifts/exports and owned graphs are implemented; bounded same-graph measurements retain independent-process gates. Observed setup amortization, physical peak memory, broader graphs/providers and hardware qualification remain | [#42](https://github.com/doublemover/RNS8/issues/42), [#43](https://github.com/doublemover/RNS8/issues/43) |
| Algebraic composition | Exact epilogues, paired 3M/4M and one-level Strassen/rank-23 coefficient/straight-line and rank48 mixed-channel matrix subproducts are written; measured routing and conditional extensions remain gated | [#47](https://github.com/doublemover/RNS8/issues/47), [#48](https://github.com/doublemover/RNS8/issues/48) |
| Low-bit/floating encodings | Scalar certificates now separate proof verdicts; a host rational/binary32 rewrite oracle is authored. Explicit gfx942 FNUZ/gfx1201 OCP FP8 finite owners and MFMA/WMMA implementations are written; target proof/ISA/performance execution gates remain | [#44](https://github.com/doublemover/RNS8/issues/44), [#45](https://github.com/doublemover/RNS8/issues/45), [#46](https://github.com/doublemover/RNS8/issues/46) |
| Integrity/sparse/multi-GPU | Private integrity candidates, public Sparse-A/hybrid owners and real modulus/M/N/2-D/K-sharded M1/M2 source; actual multi-GPU and sparse target execution/traffic/failure qualification remain | [#49](https://github.com/doublemover/RNS8/issues/49), [#50](https://github.com/doublemover/RNS8/issues/50), [#51](https://github.com/doublemover/RNS8/issues/51) |
| Narrow/generated kernels | Explicit public proof-selected native paths and source-bound generated families are written; target execution and measured selection remain | [#53](https://github.com/doublemover/RNS8/issues/53), [#55](https://github.com/doublemover/RNS8/issues/55) |
| Additional hardware | Future CDNA5 TDM/cluster; Linux AMX-INT8 and AArch64 I8MM kernels, OS admission and packing/reuse owners are written but require actual supported CPU qualification | [#52](https://github.com/doublemover/RNS8/issues/52), [#54](https://github.com/doublemover/RNS8/issues/54) |

## Setup and examples dependencies

Shared write-free toolchain discovery is implemented under
[#57](https://github.com/doublemover/RNS8/issues/57); explicit roots take precedence.
[Dependency locks](dependency-versions.md) select the new non-GPU dependency stack
under [#60](https://github.com/doublemover/RNS8/issues/60). Installation and lock
selection are distinct from compilation and runtime qualification. Bounded Linux
GPU sessions [#58](https://github.com/doublemover/RNS8/issues/58) now author bounded
plans, isolated environments, checkpoints and resume through the existing runner;
Linux rehearsal remains deferred. Public application examples and
independent baselines are implemented under [#62](https://github.com/doublemover/RNS8/issues/62);
the paired-complex fifth workload passes applicable local registered checks,
while sustained production-boundary measurements remain unqualified. Documentation consolidation
[#61](https://github.com/doublemover/RNS8/issues/61) consumes those concrete boundaries.

### Current validation and remaining work

The current local source/test receipt is the [implementation closeout](../metadata/implementation-closeout-20261027.json).
It distinguishes real gfx1100 execution, target-inapplicable skips and compile-only
AMX/I8MM/FP8 source evidence. Native Linux, multi-GPU, Instinct/CDNA and unavailable
instruction families remain separate qualification lanes. The current Windows
suite cannot establish their behavior.

Remaining local research work is measured strongest-control comparison, profitable
whole-route selection, setup/reuse/export amortization, physical overlap/traffic
and fault-recovery evidence. Existing reviewed gains remain in [performance](performance.md).
No new campaign or route promotion follows from completing this implementation.

Broader compiler/ABI/package/runner checks and native Linux command/session
rehearsals remain under #57/#58/#60/#61. The selected Git candidate remains
uninstalled. Explicit gfx950 FP4/FP6/FP8 fixed/dynamic/mixed-format kernels are
implemented under a conditional instruction model, with live qualification
pending. M3 distributed transports remain a separately gated extension. There are 40 open issues;
child acceptance gates remain the roadmap's retirement criterion.

## Stop conditions for hardware work

Stop at the first arithmetic mismatch, out-of-bounds access, device fault,
wrong-target error or unexplained skip. Preserve the exact build/logs and
diagnose correctness before timing. An unavailable optional backend is recorded
as unavailable; a fallback cannot be relabeled as that accelerator.

## October 23 issue repair qualification

The original detailed entry is preserved in the
[dated implementation chronology](archive/implementation-checkpoints-2026-10-23.md#october-23-issue-repair-qualification).

### Pending issue-batch qualification (2026-10-23)

This former checkpoint now lives in the
[chronology](archive/implementation-checkpoints-2026-10-23.md#pending-issue-batch-qualification-2026-10-23).
Use the recorded baseline above for subsequent registered-test evidence.

### Subsequent implementation boundary

The chronological [implementation boundary](archive/implementation-checkpoints-2026-10-23.md#subsequent-implementation-boundary)
is preserved with its original context; current public/private boundaries are
owned by the linked topic guides.


### Current materialization and provider boundary

The [runtime product arena](runtime-product-arena.md) executes CPU/direct-HIP
chains and diamonds with exact suffix growth, retained observations, staged
publication and reusable scratch. Its [opt-in materialization planner](planning-recipes.md#cost-planned-product-materializations)
adds shared packed B alternatives, joint prefix/cost search and a physical
liveness arena. The latest CPU/HIP and installed-consumer fixtures passed.
Capacity-aware schema-2 prices and frozen measured-stage ingestion are implemented.
The [October 25 independent collection](focused-closeout-measurements-20261025.md#subsequent-stage-price-and-proof-policy-experiment)
passes exact checks but fails all four historical held-out prediction cohorts.
The [October 26 SDK10.1 study](performance.md#october-26-sdk101-bounded-measurements)
adds source-bound same-graph/process-held-out results and preserves each
original/post-restart cohort separately. Broader representation-choice,
unseen-workload prediction, measured setup amortization and qualified routing
remain under #42/#30/#31. The global-minimum and live/peak allocation controls pass local CPU/GPU
and installed-ABI checks. Explicit generation-bound summary ownership passes
the [full local batch and 40 matched captures](adaptive-native-product.md#matched-summary-reuse-observations).
Its host-output elapsed effects fail the fixed qualification gates. A separate
[95-capture resident-output comparison](adaptive-native-product.md#separate-resident-output-boundary)
qualifies two within-adaptive reuse improvements, while fixed RNS still wins
those workloads. Fused narrow WMMA wins the 512³ comparison against the strongest
tested scalar alternative; qualified measured routing remains open.

The [physical capacity qualification](out-of-core-panels.md#real-32-gib-input-on-the-24-gib-rx-7900-xtx)
now completes a real >32 GiB native/minimum-RNS input on the 24 GiB gfx1100 card:
four independent processes, 12 exact products, observed bounded device/pinned
payload peaks and complete retirement. Independent steady crossover,
achieved-overlap traces and held-out panel calibration remain under #68.
Address-domain issue #24 and the completed cold/steady hipBLASLt metadata
study #70 are closed; the latter accepts its negligible/negative elapsed result
without raw cross-call descriptors or routing promotion.

The [bound recipe layer](bound-pipeline-recipes.md) now composes existing producer,
representation, queue, stream/graph and checked-export owners with explicit costs.
The [FP8 finite owner](fp8-modular-research.md), [CDNA5 transfer/cluster kernels](cdna5-tdm-research.md)
and [distributed independent commit check](distributed-hip.md#optional-independent-integrity-before-commit)
are implemented in source. Broader provider parity and target-specific execution,
ISA, fault coverage and measured routing remain qualification gates under
#30/#45/#49/#50/#52. Unsupported providers do not inherit support from a registry entry
or a passing direct-HIP path. The [per-issue ledger](../metadata/remaining-implementation-20261024.json)
tracks current source, local evidence and remaining measurements/hardware.
