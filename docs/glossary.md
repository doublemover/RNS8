# Glossary

| Term | Meaning |
|---|---|
| AUTO | Backend selection mode that may use reviewed autotune cache entries; missing or rejected entries stay on a correctness path. |
| bounded i64/u64 | Exact signed or unsigned 64-bit GEMM with an explicit range bound. |
| CRT | Chinese remainder reconstruction from residue planes. |
| exact-wide | RNS output exported as fixed-width little-endian limbs instead of narrowed i64/u64. |
| finite u8 | Explicit modulus arithmetic over byte-sized rings or fields. |
| prefix | Number of moduli selected from the default ladder for an RNS operation. Benchmark schema v4 keeps `prefix` as the requested max prefix and reports `selected_prefix` when execution uses fewer planes. |
| prepack cache | Backend-owned reusable packed representation, currently limited and explicitly reported. |
| RNS | Residue number system matrix storage. |
| rocWMMA | AMD rocWMMA-based opt-in matrix-engine backend. Public backend spelling is `rocwmma`. |
| AMDGPU builtins | Reserved public backend identity for target-specific MFMA, WMMA, SMFMAC, and SWMMAC kernels. Public backend spelling is `amdgpu-builtins`; runtime dispatch remains unsupported until compiled kernels pass exact evidence gates. |
| sparse-A v1 | Explicit A-side 4:2 structured sparsity contract with dense B, canonical K-group indices, explicit byte signedness, and CPU expand/reference correctness anchor. It is not automatic dense pruning or general sparse GEMM. |
| strict wrap64 | Multiplication modulo `2^64` using byte-limb semantics, not odd-modulus CRT. |
| vector ALU | Native integer HIP backend for bounded i64/u64, separate from matrix-engine paths. |
| WMMA | Matrix instruction family. Used in CK and rocWMMA ISA descriptions, not as a public RNS8 backend name. |
