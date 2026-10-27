# Current roadmap and release gates

This page owns remaining execution gates. The [technical specification](RNS8_RESEARCH_SPEC.md)
owns normative architecture/acceptance rules; [backend capabilities](backend-notes.md)
and the [research index](research/README.md) distinguish public paths from private
prototypes. The [implementation chronology](archive/implementation-checkpoints-2026-10-23.md)
preserves earlier work without duplicating current status.

## Recorded baseline

The [October 26 Core SDK10.1 local Release validation](platform-windows.md#october-26-core-sdk-101-local-release-validation)
records a fresh automatically resolved Windows/gfx1100 stack at `059fa45`:
1,080 passes, 30 target/feature skips and zero failures in 1,110 registered
tests, plus four installed static/shared native-device checks. Selected HIP
runtime hashes are verified after loading. This supersedes pending local
SDK10.1 Release bring-up, while broader hardware/configuration and matched
performance gates remain. Historical measurements below retain their original
source and toolchain boundaries.

The [October 26 selected implementation audit](selected-implementation-20261026.md)
records source and fixtures for all 25 requested issue slices, including the new
real modulus-sharded HIP owner and standalone system/LDS/memory probes. Only
syntax checks were performed for that source; runtime and measurement gates
remain. #28/#29 were already closed; this pass closes no issue from source alone.

The [frontloaded combined retest](../metadata/frontloaded-combined-validation-20261026.json)
records successful full builds at `f406c29`, 1,038 GPU passes / 17 skips and
572 CPU-ASan passes / 32 skips, with one stale fixture assertion failing in each
lane. The folded per-tile HIP export repair and new CPU/HIP regressions pass;
18 installed consumers, two strict C17 probes and all 98 independently replayed
CRT outputs pass. The stale assertion correction and direct reader-exception
coverage subsequently pass the explicitly authorized
[four-check targeted follow-up](../metadata/frontloaded-targeted-validation-20261026.json)
at `84b4d9c`. Both historical full-suite failures remain in the receipt; the
full suites were not repeated again. No research campaign resumed.

The latest [materialization integration](local-validation-20261025.md#subsequent-materialization-integration)
passed at `6ce489b`; subsequent address/metadata qualification brings the total
to 11 evidence-backed closures and 40 open issues.
The per-issue ledger separates the completed local checks from unresolved
implementation, measurement and hardware requirements.

The [October 25 consolidated validation](local-validation-20261025.md) records
the expanded Windows/gfx1100 and CPU ASan correctness coverage, repaired
failures, installed static/shared graph consumers and six closed tooling issues.
Its separate artifact boundaries and pending dependency/external-link gates
are explicit. Research performance and broader hardware qualification remain
with their issue-specific gates below.

The [October 24 partial campaign](7900xtx-partial-campaign-2026-10-24.md) records
`ad71df5` production validation (747 passes, three expected skips), 5,382 admitted
benchmark captures and reviewed matched gains. Collection is stopped by user
request; unfinished cohorts remain unqualified. Its measurements supersede
older results only for the exact source/configuration and contracts reported.

The [expanded October 23 validation](local-all-features-20261023.md) records the
`51ae8d8` Windows/gfx1100 all-feature build, 489 passes, three target-inapplicable
sparse skips and zero failures. It supersedes older local pending-runtime notes
for the registered applicable tests, including repaired hipBLASLt operand-cache
identity. The [512³/1024³ report](7900xtx-large-throughput-2026-10-23.md) remains the
historical large-shape comparison; [256³](7900xtx-throughput-2026-10-23.md) remains the
initial diagnostic. Changed source/configurations need their own evidence.

The [October 6 audit](audit-2026-10-06.md) preserves defect/repair history.
Subsequent local CPU, allocation, package, lineage, range and cache checkpoints
are catalogued in [history](archive/README.md); they retain their exact scope.

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
| Conversion-aware planning and roofs | Typed models/calibration and instruction/conversion/CRT/system probes are written; target measurements, counters and qualified held-out prediction remain | [#30](https://github.com/doublemover/RNS8/issues/30), [#31](https://github.com/doublemover/RNS8/issues/31) |
| Bound discovery and variable prefixes | Device discovery/adaptive/ragged candidates, compact device-bucket workers and explicit matrix-owned adaptive API are authored; four-wave register/one-buffer/two-buffer native WMMA candidates are authored; summary reuse and measured cost tradeoffs remain | [#34](https://github.com/doublemover/RNS8/issues/34), [#35](https://github.com/doublemover/RNS8/issues/35) |
| Scheduling | Private queue/split-K and logical locality models; measured configuration selection and broader target work remain | [#36](https://github.com/doublemover/RNS8/issues/36), [#37](https://github.com/doublemover/RNS8/issues/37), [#38](https://github.com/doublemover/RNS8/issues/38) |
| Reconstruction/materialization | Public explicit CPU/HIP lift; private balanced/mixed-radix alternatives include an authored tile-major CRT consumer; performance and broader representation integration remain | [#39](https://github.com/doublemover/RNS8/issues/39), [#40](https://github.com/doublemover/RNS8/issues/40), [#41](https://github.com/doublemover/RNS8/issues/41) |
| Resident execution | Native/uniform/ragged/packed materialization planning and retained arena execution; direct-HIP tokens/dependencies, transfer/lift/export, supported semantics and owned graphs are written; provider-specific unsupported modes, measurements and hardware qualification stay explicit | [#42](https://github.com/doublemover/RNS8/issues/42), [#43](https://github.com/doublemover/RNS8/issues/43) |
| Algebraic composition | Exact epilogues, paired 3M/4M and one-level Strassen/rank-23 coefficient/straight-line matrix subproducts are written; measured routing and conditional extensions remain gated | [#47](https://github.com/doublemover/RNS8/issues/47), [#48](https://github.com/doublemover/RNS8/issues/48) |
| Low-bit/floating encodings | Scalar certificates now separate proof verdicts; a host rational/binary32 rewrite oracle is authored. An explicit gfx942 FNUZ FP8 finite owner and MFMA implementation are authored; target proof/ISA/performance execution gates remain | [#44](https://github.com/doublemover/RNS8/issues/44), [#45](https://github.com/doublemover/RNS8/issues/45), [#46](https://github.com/doublemover/RNS8/issues/46) |
| Integrity/sparse/multi-GPU | Private integrity candidates, public Sparse-A/hybrid owners and real modulus-sharded M1/M2 source; actual multi-GPU and sparse target execution/traffic/failure qualification remain | [#49](https://github.com/doublemover/RNS8/issues/49), [#50](https://github.com/doublemover/RNS8/issues/50), [#51](https://github.com/doublemover/RNS8/issues/51) |
| Narrow/generated kernels | Explicit public proof-selected native paths and source-bound generated families are written; target execution and measured selection remain | [#53](https://github.com/doublemover/RNS8/issues/53), [#55](https://github.com/doublemover/RNS8/issues/55) |
| Additional hardware | Future CDNA5 TDM/cluster; Linux AMX-INT8 kernel, OS permission, packing/reuse and collector are written but require actual supported CPU qualification | [#52](https://github.com/doublemover/RNS8/issues/52), [#54](https://github.com/doublemover/RNS8/issues/54) |

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

Core SDK10.1 Windows/gfx1100 Release validation is complete at `059fa45`:
1,080 registered passes, 30 target/feature skips and zero failures, plus four
installed static/shared native-device checks. The automatic initializer and
loaded SDK runtime identities are verified. Subsequent repository cleanup
preserves every benchmark input and measurement record; its final integration
batch passes 52 tests with 11 expected FP8/TDM hardware skips. See the
[SDK record](../metadata/hip101-validation-20261026.json) and
[cleanup record](../metadata/repository-cleanup-20261026.json).

The next local performance work is a fresh bounded SDK10.1 comparison against
the strongest same-contract controls, followed by qualified stage prices for
the planner and resident DAG. Keep exact-reference audits outside timed
production intervals and include setup/reuse/export costs at the claimed
boundary. The stopped campaign remains stopped; historical timing results do
not establish new-stack performance. Local Debug/ASan, OpenMP-OFF and broader
dependency/configuration qualification remain separate release work.

The [October 25 allocation/global-control retry](local-validation-20261025.md#allocation-and-global-control-integration-retry)
records clean `2fcb91c` artifacts: 547 CPU AddressSanitizer passes and 931 full-feature
gfx1100 passes, with no failures. Both arithmetic controls passed 673 cases at
`168a160`; 12 installed consumers and two strict C17 ABI probes also pass. The
seven first-pass capture-contract failures were repaired together before the
complete CPU/GPU retry. Unsupported target skips and the unchanged shared logical
proof remain explicitly accounted for. A separate brace-only GCC warning repair
passes actual Debug/Release compilation in WSL2; Linux CI completion is separate.

Generation-bound native/pinned-host transfers, event-dependent joins, exact
lift/export, paired-complex and block-sparse owners, custom bases and product
arenas have applicable local regression evidence. Their issue-specific timing,
profiler, capacity and broader-platform gates remain separate. A pushed source
implementation or passing local test is not blanket hardware qualification.

Unavailable Linux multi-GPU, Instinct/CDNA and instruction-family qualification
remain deferred. SDK10.1 is the locally validated Release stack; retained HIP7.1
source fallback pins and historical measurements keep their original scope.
The selected Git candidate remains uninstalled (local2.52.0.windows.1 versus
selected2.56.0.windows.3). Broader OS/compiler/package/runner qualification is
still required under #57/#58/#60. All 44 official dependency/documentation URLs
responded successfully in the recorded #61 check;
actual matching-platform command rehearsals remain separate.
The [focused measurements](focused-closeout-measurements-20261025.md) completed
560 captures on the explicitly authorized shared host, including 20 independent
stage-training and 140 new complete-route captures. Explicit scalar-first public
proofs beat the default eager-axis policy at 128/256; compiled recipes do not
establish a benefit over this stronger baseline. All four measured models fail
the predeclared held-out tolerance; no route is promoted. The original
5,382/16,513-run campaign remains stopped.

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
The [new independent collection](focused-closeout-measurements-20261025.md#subsequent-stage-price-and-proof-policy-experiment)
passes exact checks but fails all four held-out prediction cohorts. Broader
representation-choice integration and qualified measured routing remain under
#42/#30/#31. The fresh global-minimum and live/peak allocation controls now pass local CPU/GPU
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
