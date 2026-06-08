# rocWMMA And AMDGPU Builtin Backend

Opt-in rocWMMA accelerator backend plus reserved AMDGPU builtin hot-kernel
path.

The rocWMMA backend is compiled only with `RNS8_ENABLE_ROCWMMA=ON`. It uses the
pinned repo-local rocWMMA headers and RNS8-owned HIP kernels to pack centered
signed residues into 16-aligned panels, execute `int8 x int8 -> int32`
matrix instructions, and fuse INT32 accumulators back to centered `int8_t`
residues without global INT32 scratch output. RDNA builds use the rocWMMA
wave32 path; CDNA builds select wave64 launch geometry from the active
`RNS8_AMDGPU_TARGETS` codegen target.

Eligible non-tiled RNS B operands with `K <= 65536` can be packed once into the
rocWMMA column-major B panel layout through `rns8_create_prepack_cache` and
reused by `rns8_gemm_rns_prepacked_b`. A remains packed through the normal
transient workspace per dispatch. `rns8_get_prepack_cache_info` exposes the
created cache key, device id, and allocation byte contract. This is a narrow
runtime cache path, not a broad production prepack-cache policy: tiled
schedules, finite/wrap64 semantics, A-operand caches, oversize K, and other
backends remain unsupported.

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
Benchmark fixtures currently record host wall-clock evidence only, so the
backend keeps `performance_validated=false` until reviewed captures prove a
target-shape win.

Dependency discovery and primitive compile probes remain evidence only; they
do not enable rocWMMA by themselves. `RNS8_ENABLE_AMDGPU_BUILTINS` now exposes
the public AMDGPU builtin backend identity and disabled capability metadata,
but runtime contexts and GEMM dispatch remain unsupported until target-specific
MFMA/WMMA/SMFMAC/SWMMAC kernels have exact CPU parity, timing, and ISA
evidence.
