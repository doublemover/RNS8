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
metadata modes, ragged/large shapes and a split-K case. These additions have not
been built, executed or measured; validation remains deferred.

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
