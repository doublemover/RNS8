
## RDNA3 Complete -- June 11, 2026

All RDNA3 (gfx1100) active performance ranks verified complete.
243-capture sweep with 5 backends, 0 failures, 281 tests pass.

Live optimizations: DP4A finite-u8, Garner i64/u64 export, VOPD DPP export,
WMMA skinny dispatch, status elision, persistent/coalesced pack, non-temporal
loads, persistent small GEMM, HIP graph replay, zero-skip detection, adaptive
prefix, verification amortization.

Deferred: rocWMMA HIP event recording (separate build target gap),
CDNA3/Linux validation (per project policy).

