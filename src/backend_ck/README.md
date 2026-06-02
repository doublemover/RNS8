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
and ISA gate are present. Current CK captures are host wall-clock evidence and
keep `performance_validated=false` until reviewed target-shape captures prove
it is the fastest accepted backend.
