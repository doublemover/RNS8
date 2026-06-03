# hipBLASLt Backend

Feature-detected hipBLASLt baseline accelerator path.

`RNS8_ENABLE_HIPBLASLT=ON` builds an opt-in Windows HIP SDK baseline backend
when `RNS8_ENABLE_HIP=ON` and AMD's `roc::hipblaslt` target or import archive
is discoverable. The backend uses resident HIP RNS matrices, packs each selected
residue plane into 16-aligned transposed INT8 buffers, runs hipBLASLt
`int8 x int8 -> int32`, and reduces padded INT32 scratch back to centered
`int8_t` residues with a separate HIP kernel.

For fixed-prefix RNS GEMM with `k <= RNS8_SAFE_INT32_K_BLOCK`, the caller
workspace keeps a non-durable B prepack cache keyed by HIP device, B source
version, `k/n/ldb`, prefix, and cache byte size. Stable repeated-B calls skip
the transient B transpose-pack kernel after the first warmup/materialization.
Finite-u8, adaptive schedules, and split-K calls still use the transient B pack
path.

Supported contracts:

- Fixed-prefix bounded `int64_t` and `uint64_t` RNS GEMM.
- Exact-wide signed/unsigned RNS output.
- Finite ring/field `uint8_t` GEMM through explicit centered residues.

Unsupported contracts:

- Adaptive/per-tile bounded schedules.
- Strict wrap64 byte-limb semantics.
- Fused hipBLASLt epilogues or performance claims.
- Durable public hipBLASLt prepack-cache objects.

This is a correctness baseline only. It remains optional, is not required for
CPU or direct-HIP correctness, and benchmark metadata reports
`perf_validated=0` until reviewed captures prove otherwise. CK, rocWMMA, and
AMDGPU builtin accelerator flags remain fail-fast until their real kernels and
exact differentials exist.
