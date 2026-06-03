# Prior Art And Related Systems

RNS8 combines ideas from exact arithmetic, residue number systems, and GPU GEMM.
This page is a public orientation map, not a benchmark comparison.

## Compute Libraries

- AMD HIP and ROCm: GPU runtime/toolchain family for AMD accelerators.
- hipBLASLt: AMD library path for int8 GEMM experiments and baselines.
- Composable Kernel: template-based AMD GPU kernel library used by the CK
  accelerator lane.
- rocWMMA: AMD matrix-instruction library used by the rocWMMA backend lane.

## Exact Arithmetic

- Boost.Multiprecision: first-party CPU reference dependency for exact
  reconstruction and differential checks.
- GMP and FLINT: optional comparison/reference libraries behind the
  `optional-exact-libs` vcpkg feature.

## Scope Boundary

RNS8 is not trying to replace general BLAS, symbolic algebra systems, FHE
libraries, or arbitrary-precision CPU packages. It targets explicit exact
integer GEMM contracts with hardware-realistic AMD GPU evidence.
