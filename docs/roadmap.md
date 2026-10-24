# Current roadmap and release gates

This page owns remaining execution gates. The [technical specification](RNS8_RESEARCH_SPEC.md)
owns normative architecture/acceptance rules; [backend capabilities](backend-notes.md)
and the [research index](research/README.md) distinguish public paths from private
prototypes. The [implementation chronology](archive/implementation-checkpoints-2026-10-23.md)
preserves earlier work without duplicating current status.

## Recorded baseline

The [expanded October 23 validation](local-all-features-20261023.md) records the
`51ae8d8` Windows/gfx1100 all-feature build, 489 passes, three target-inapplicable
sparse skips and zero failures. It supersedes older local pending-runtime notes
for the registered applicable tests, including repaired hipBLASLt operand-cache
identity. The [512³/1024³ report](7900xtx-large-throughput-2026-10-23.md) is the
current scaling comparison; [256³](7900xtx-throughput-2026-10-23.md) remains the
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

Independent per-GPU shard scripts do not implement distributed GEMM. RDNA2,
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
| Conversion-aware planning and roofs | Offline certified models/calibration; target measurements, memory/instruction roofs and held-out regret remain | [#30](https://github.com/doublemover/RNS8/issues/30), [#31](https://github.com/doublemover/RNS8/issues/31) |
| Bound discovery and variable prefixes | Device discovery/adaptive/ragged candidates, compact device-bucket workers and explicit matrix-owned adaptive API are authored; four-wave register/one-buffer/two-buffer native WMMA candidates are authored; summary reuse and measured cost tradeoffs remain | [#34](https://github.com/doublemover/RNS8/issues/34), [#35](https://github.com/doublemover/RNS8/issues/35) |
| Scheduling | Private queue/split-K and logical locality models; measured configuration selection and broader target work remain | [#36](https://github.com/doublemover/RNS8/issues/36), [#37](https://github.com/doublemover/RNS8/issues/37), [#38](https://github.com/doublemover/RNS8/issues/38) |
| Reconstruction/materialization | Public explicit CPU/HIP lift; private balanced/mixed-radix alternatives include an authored tile-major CRT consumer; performance and broader representation integration remain | [#39](https://github.com/doublemover/RNS8/issues/39), [#40](https://github.com/doublemover/RNS8/issues/40), [#41](https://github.com/doublemover/RNS8/issues/41) |
| Resident execution | Public synchronous DAGs and fixed-prefix HIP stream/capture subsets; cancellation, timed observation, failed-drain quarantine and synchronous paired-complex nodes are authored; broader async nodes, runtime allocation reuse and qualification remain | [#42](https://github.com/doublemover/RNS8/issues/42), [#43](https://github.com/doublemover/RNS8/issues/43) |
| Algebraic composition | Public exact-wide epilogue and experimental paired complex 3M/4M; opt-in public one-level Strassen ownership is authored; rank-23 straight-line execution and measured routing remain | [#47](https://github.com/doublemover/RNS8/issues/47), [#48](https://github.com/doublemover/RNS8/issues/48) |
| Low-bit/floating encodings | Scalar certificates now separate proof verdicts; a host rational/binary32 rewrite oracle is authored. Floating hardware/public execution remains reserved, with target proof/ISA/performance gates | [#44](https://github.com/doublemover/RNS8/issues/44), [#45](https://github.com/doublemover/RNS8/issues/45), [#46](https://github.com/doublemover/RNS8/issues/46) |
| Integrity/sparse/multi-GPU | Private check-residue/block-sparse candidates and explicit public Sparse-A; integration and true distributed execution remain | [#49](https://github.com/doublemover/RNS8/issues/49), [#50](https://github.com/doublemover/RNS8/issues/50), [#51](https://github.com/doublemover/RNS8/issues/51) |
| Narrow/generated kernels | Checked research candidates/generator; public dispatch, target expansion and measured selection remain | [#53](https://github.com/doublemover/RNS8/issues/53), [#55](https://github.com/doublemover/RNS8/issues/55) |
| Additional hardware | Future CDNA5 TDM/cluster and optional CPU matrix instruction work | [#52](https://github.com/doublemover/RNS8/issues/52), [#54](https://github.com/doublemover/RNS8/issues/54) |

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
the paired-complex fifth workload is authored, while sustained production-boundary
measurements remain in the final validation batch. Documentation consolidation
[#61](https://github.com/doublemover/RNS8/issues/61) consumes those concrete boundaries.

### Current frontloaded batch

The last completed local all-feature retry at `1bb5e33` passed 491 tests with
three expected target-inapplicable sparse skips and no failures; its raw receipt
is `temp/1bb5e33-tests-retry.log`. The separate 17-mode native smoke collection
also passed. These results predate the new dependency and source changes.

Subsequent commits author failed-drain quarantine, token-bound diagnostics,
cooperative cancellation, timed observation, typed async support queries, an
experimental public paired-complex API and resident graph node, compact adaptive
workers, an explicit retained-native adaptive API, offline dependency-proven arena
reuse, and direct tile-major CRT consumption with matched route calibration. Their
fixtures are written; no new project validation or benchmark pass has run.
The user requested all remaining implementation before one consolidated pass.
Do not interpret a pushed implementation or an installed compiler as qualification.

The user explicitly deferred ROCm 10.1 and hardware-dependent Linux multi-GPU,
Instinct/CDNA and unavailable instruction-family qualification. Retain the
HIP 7.1 accelerator pins for this local batch. VS 2026 18.11.1 and the isolated
non-GPU dependency stack (including GMP/MPFR/FLINT) are installed; their project
qualification remains pending. Git's installer update remains
pending after automatic approval review blocked its command. These gaps cannot
be represented as completed issue acceptance.

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


### Current source frontload

Source additions after the last executed checkpoint include multiwave native WMMA,
public one-level Strassen exact export ownership, floating rewrite host oracles and
shared epilogue/lift admission providers and a [public narrow native owner](narrow-native-owner.md). A [generation-bound block-sparse public owner](block-sparse.md#public-generation-bound-owner) is also authored. Their fixtures are authored, not executed.
The final upgraded-stack campaign remains deferred until the implementation frontload
ends. No new source commit inherits the old checkpoint's qualification.

A one-producer token binding with explicit nonblocking progress is now authored.
Concrete remaining development includes multi-producer/event-pipelined dependencies and async transfers/lift/export,
runtime materialization/liveness allocation, broader runtime recipe providers,
generation-owned encoded-B integration, block-sparse hybrid scheduling, broader narrow scheduling and rank-23 straight-line
execution. These are code gaps, distinct from unavailable hardware and measured
promotion gates. Completion of one validation batch cannot by itself close them.
