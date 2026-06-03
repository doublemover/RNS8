# Composable Kernel Backend

Opt-in Windows `gfx1100` Composable Kernel accelerator backend.

The backend is compiled only with `RNS8_ENABLE_CK=ON`. It uses the pinned
repo-local CK headers plus RNS8-owned HIP pack/output kernels to provide fused
centered-residue `int8 x int8 -> int32` GEMM for fixed-prefix bounded plans,
adaptive per-tile bounded plans, exact-wide RNS output, and finite u8. The path
is optional and is not required for CPU or direct-HIP correctness.

Dependency discovery and compile probes remain evidence only. CK reports an
enabled correctness backend only in the explicit CK preset after the compiled
kernels, exact CPU/direct-HIP differential tests, benchmark schema fixtures,
and ISA gate are present. The current ISA gate requires CK WMMA instructions
and rejects scalar divide/remainder/reciprocal mnemonics plus unintended INT32
global stores in matched CK GEMM symbols. Current CK captures keep
`performance_validated=false` until reviewed target-shape captures prove it is
the fastest accepted backend.

The CK preset generates RNS8's WMMA no-divide block-map include overlay from
the pinned repo-local CK header during configure. The patch is exact-match
guarded, no-ops when the source already has the patched form, fails fast if the
expected `MakeDefaultBlock2CTileMap` block has drifted, puts the overlay before
CK's include directory for the CK HIP compile, and registers the generated
patched header as a dependency of the compiled CK HIP object.
