# rocWMMA And AMDGPU Builtin Backend

Public availability and target evidence are summarized in the
[backend reference](../../docs/backend-notes.md); this page owns implementation
layout and admission details. The [dated local validation](../../docs/local-all-features-20261023.md)
records the specific Windows/gfx1100 configuration exercised.


Opt-in rocWMMA matrix-engine backend. AMDGPU builtins are a separate
implemented opt-in backend with target-specific runtime restrictions.

The rocWMMA backend is compiled only with `RNS8_ENABLE_ROCWMMA=ON`. It uses the
pinned repo-local rocWMMA headers and RNS8-owned HIP kernels to pack centered
signed residues into 16-aligned panels, execute `int8 x int8 -> int32`
matrix instructions, and fuse INT32 accumulators back to centered `int8_t`
residues without global INT32 scratch output. RDNA builds use the rocWMMA
wave32 path; CDNA builds select wave64 launch geometry from the active
`RNS8_AMDGPU_TARGETS` codegen target.

Eligible non-tiled bounded i64/u64 B operands with `K <= 65536` can be packed
once into the rocWMMA column-major B panel layout through
`rns8_create_prepack_cache` for resident RNS matrices or through the explicit
host-native `rns8_create_b_prepack_cache_i64/u64` constructors. The native
constructors avoid first materializing B as a persistent RNS matrix: they stage
host row-major B into device memory, center it per modulus, and write the
rocWMMA B panel cache directly. A remains packed through the normal transient
workspace per dispatch. `rns8_get_prepack_cache_info` exposes the created cache
key, source version, layout identities, device id, and allocation byte contract.
This is a narrow runtime cache path, not a broad production prepack-cache
policy: tiled schedules, finite/wrap64 semantics, exact-wide caches, A-operand
caches, oversize K, and other backends remain unsupported.

The implemented rocWMMA coverage includes fixed-prefix bounded RNS plans,
adaptive per-tile bounded schedules, exact-wide RNS output, and finite u8. The
ISA gate requires a target-appropriate matrix instruction (`v_wmma` on RDNA,
`v_mfma` on CDNA) and rejects scalar divide/remainder/reciprocal mnemonics plus
unintended INT32 global stores.
The same object also contains an internal strict wrap64 byte-GEMM36 candidate:
it consumes compact byte-limb device buffers, uses signed matrix instructions
plus high-bit correction terms to recover unsigned byte products, and writes
compact byte-limb output for comparison against direct HIP and the CPU byte-pair
oracle. That candidate is not a public backend and is not AUTO-selected.
The [larger-workload report](../../docs/7900xtx-large-throughput-2026-10-23.md)
contains fresh public and private residue-epilogue comparisons with explicit
host/event scopes. `performance_validated=false` metadata continues to distinguish
measurement from reviewed automatic routing.

Dependency discovery and primitive probes do not enable either runtime backend.
`RNS8_ENABLE_AMDGPU_BUILTINS=ON` with HIP compiles the separate target-specific
WMMA/MFMA/SMFMAC/SWMMAC path described in the backend reference. The source target,
compiled configuration and device validation remain distinct evidence.
