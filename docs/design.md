# RNS8 Design Notes

`docs/RNS8_RESEARCH_SPEC.md` remains the architecture source of truth. This
file records the implemented scaffold state.

Current core design:

- Public ABI is C with explicit `struct_size` and `abi_version` fields.
- Semantics are explicit through `rns8_semantics`; bounded signed and unsigned
  64-bit, exact-wide signed/unsigned limb export, and strict wrap64 byte-limb
  contracts remain separate ABI surfaces.
- Persistent RNS matrices are created from `rns8_matrix_desc` and store
  modulus-major centered `int8_t` residues.
- Packing writes into an explicit matrix object. The ABI does not infer whether
  an input is A, B, or C from a plan call order.
- CPU reference GEMM runs one scalar ring GEMM per selected modulus and splits
  K into blocks no larger than 65536 before residue reduction.
- CRT reconstruction uses Boost.Multiprecision incremental Garner/CRT logic
  and checks signed/unsigned range contracts before export.
- Exact-wide export writes fixed-width little-endian limbs. Signed export uses
  the centered CRT representative with the same `x >= ceil(P / 2)` negative
  threshold as centered residue packing; unsigned export uses canonical
  magnitude limbs. Both use `ld` as an element stride, require `limb_count` in
  `[1, 32]`, and report destination-preserving range errors rather than
  truncating. Exact-wide remains separate from bounded i64/u64 export and
  strict wrap64 byte-limb export.

Current backend boundary:

- `RNS8_BACKEND_CPU_REFERENCE` is the deterministic correctness backend.
- `RNS8_BACKEND_HIP_DIRECT` is a real Windows HIP bring-up path with device
  inspection, GPU residue conversion, one-modulus ring-GEMM smoke coverage,
  K-block splitting, and bounded API smoke coverage.
- hipBLASLt, CK, rocWMMA, and AMDGPU builtin paths are not implemented and must
  remain feature-detected evidence-only accelerators, not correctness
  requirements. Their enable flags must fail fast until real correctness
  backends exist.
