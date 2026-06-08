# Composable Kernel Backend

Opt-in Composable Kernel accelerator backend for the active HIP/ROCm target.

The backend is compiled only with `RNS8_ENABLE_CK=ON`. It uses the pinned
repo-local CK headers plus RNS8-owned HIP pack/output helper kernels to run
signed `int8 x int8 -> int32` matrix GEMM for fixed-prefix bounded plans,
adaptive per-tile bounded plans, exact-wide RNS output, and finite u8. CK
CShuffle keeps the INT32 accumulation internal to the matrix-engine kernel and
uses an RNS8 centered-residue epilogue to store `int8_t` output. RNS8 helper
kernels only copy or add centered byte tiles when padding or K-split accumulation
requires temporary output storage. The path is optional and is not required for
CPU or direct-HIP correctness.

Dependency discovery and compile probes remain evidence only. CK reports an
enabled correctness backend only in the explicit CK preset after the compiled
kernels, exact CPU/direct-HIP differential tests, benchmark schema fixtures,
and ISA gate are present. The current ISA gate is target-aware: RDNA targets
must produce CK WMMA matrix instructions, while CDNA targets must produce CK XDL
MFMA matrix instructions. It rejects scalar divide/remainder mnemonics and
reports INT32 global stores in matched CK GEMM symbols; the CDNA XDL gate allows
CK's internal block-map `v_rcp_iflag_f32` because it is launch geometry
indexing, not modular arithmetic.
Current CK captures keep
`performance_validated=false` until reviewed target-shape captures prove it is
the fastest accepted backend.

The CK preset generates RNS8's WMMA no-divide block-map include overlay from
the pinned repo-local CK header during configure. The patch is exact-match
guarded, no-ops when the source already has the patched form, fails fast if the
expected `MakeDefaultBlock2CTileMap` block has drifted, puts the overlay before
CK's include directory for the CK HIP compile, and registers the generated
patched header as a dependency of the compiled CK HIP object. CDNA builds select
CK XDL and do not use the WMMA overlay at runtime.
