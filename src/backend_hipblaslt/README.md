# hipBLASLt Backend

Public availability and target evidence are summarized in the
[backend reference](../../docs/backend-notes.md); this page owns implementation
layout and admission details. The [dated local validation](../../docs/local-all-features-20261023.md)
records the specific Windows/gfx1100 configuration exercised.


Feature-detected hipBLASLt baseline accelerator path.

`RNS8_ENABLE_HIPBLASLT=ON` builds an opt-in HIP/ROCm baseline backend
when `RNS8_ENABLE_HIP=ON` and AMD's `roc::hipblaslt` target or import archive
is discoverable. The backend uses resident HIP RNS matrices, packs each selected
residue plane into 16-aligned transposed INT8 buffers, runs hipBLASLt
`int8 x int8 -> int32`, and reduces padded INT32 scratch back to centered
`int8_t` residues with a separate HIP kernel.

For fixed-prefix RNS GEMM with `k <= RNS8_SAFE_INT32_K_BLOCK`, the caller
workspace keeps non-durable A and B prepack caches keyed by HIP device, operand
stable instance and content generation, shape/leading-dimension metadata, prefix,
and cache byte size. Zero/exhausted generations cannot produce a cache hit.
Stable repeated-A, repeated-B, or repeated-A/B calls skip the matching
transpose-pack kernel after the first warmup/materialization. Finite-u8,
adaptive schedules, and split-K calls still use the transient pack path.

Private measurement controls select `none`, `a`, `b` or `both` retained operands
independently through `--hipblaslt-prepack-policy` in `rns8-bench`. The default is
`both`. A disabled operand always uses the existing transient transpose pack;
it cannot consume an older retained buffer. Switching policy does not free a
workspace's existing allocations. Use a fresh process/workspace for allocation
comparisons. This control composes with `--hipblaslt-metadata-policy per-call|per-block`.
It applies to bounded and exact-wide fixed-prefix RNS calls. Split-K remains
transient regardless of the requested retention policy; finite-u8 does not accept
this explicit CLI control.

The `hipblaslt_operand_prepack` receipt records successful retained builds,
generation/identity matches, and retained/transient plane-pack submissions for
both operands at the warmup and measured-repeat boundaries. These are source
work counts, not byte capacities or elapsed time. Reports keep different
retention policies in separate implementation cohorts. The authored
`benchmarks/public-calibration/hipblaslt-prepack.json` controls include both
metadata modes, ragged/large shapes and a split-K case. The [October 25 integration](../../docs/local-validation-20261025.md#allocation-and-global-control-integration-retry)
passes the applicable local correctness and receipt checks. Independent physical
metadata/retention timing remains a separate qualification gate.

Supported contracts:

- Fixed-prefix bounded `int64_t` and `uint64_t` RNS GEMM.
- Exact-wide signed/unsigned RNS output.
- Finite ring/field `uint8_t` GEMM through explicit centered residues.

Unsupported contracts:

- Adaptive/per-tile bounded schedules.
- Strict wrap64 byte-limb semantics.
- Fused hipBLASLt epilogues.
- Durable public hipBLASLt prepack-cache objects.

This optional baseline is independent of CPU/direct-HIP correctness. The
[October 23 report](../../docs/7900xtx-large-throughput-2026-10-23.md#public-bounded-i64-pipelines-and-reuse)
records matched public calls and setup-inclusive operand reuse. Metadata retains
`perf_validated=0` until the relevant reviewed routing gate is satisfied.
CK, rocWMMA and AMDGPU builtins are separate opt-in implemented paths; consult
the backend reference for their contract and target limits.


### Selected algorithm observations

The benchmark opts into a bounded inventory of actual library algorithm indices
per device and padded geometry. A compile/link probe feature-detects
[`hipblaslt_ext::getIndexFromAlgo`](https://rocm.docs.amd.com/projects/hipBLASLt/en/docs-7.0.2/reference/ext-reference.html).
The query runs once per cached algorithm when observation is enabled, using a
copy of the selected object. Normal library callers leave observation disabled.
The existing cache retains the queried integer with its actual algorithm; this
does not retain another descriptor set or alter algorithm selection.

`hipblaslt_algorithms` records warmup/repeat inventories, query counts and explicit
availability/completeness. Indices are scoped to the captured library/build,
device and complete padded geometry; the same integer from another library is
not an identity match. Missing APIs, negative/unknown indices, bounded-inventory
overflow and counter overflow never become a known algorithm zero. These cases
leave arithmetic available while withholding complete comparison evidence.
The SDK-compiled C bridge keeps the official extension declaration out of the
MSVC host translation unit; the feature probe separately compiles that exact
bridge and links it with the host compiler/library.

The observation increment passes the local full CPU sanitizer and gfx1100 GPU
suite at `668f9ea`, including warm-cache physical indices and query reuse.
The repaired same-algorithm collection completes 300/300 captures across 15
workloads, both policies/timing modes and five independent processes. All
physical algorithm/work identities match. Descriptor constructions fall by
9–20× in multi-plane controls, while small elapsed-time differences do not
qualify amid process variation. No additional throughput gain or routing
promotion follows. See the [reviewed observations](../../metadata/hipblaslt-metadata-observations-20261025.json).
Fresh first-invocation setup accounting also passes: 60/60 captures across
six representative contracts, both policies and five independent processes,
with matching physical algorithm/work/input-output identities. The
collector's `--first-call` mode owns zero warmups/one completed production call,
then reports disjoint plan/workspace, matrix allocation and completed-call costs
after context creation. It excludes input generation and the independent oracle;
these observations never enter steady-state promotion. The representative
`hipblaslt-first-call.json` corpus and the
[first-invocation review](../../metadata/hipblaslt-first-call-observations-20261025.json)
retain all setup/call samples separately. The completed cold/steady study accepts
the negligible/no-benefit timing result; cross-call raw descriptor retention is
not justified by this evidence.

The ordinary benchmark creates an exact-wide plan before binding input values.
Its initial envelope is therefore the full native-type range even for a small
magnitude profile. The public collector checks that envelope before executing
any case. The K=65535/65536/65537 metadata controls use prefix 19 rather than 18;
prefix 18 remains covered at smaller K. A profile change never relaxes the
library's range admission or substitutes caller data for an unbound proof.
