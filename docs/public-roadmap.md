# Public Roadmap

RNS8 is pre-1.0. The research spec remains the detailed roadmap; this page is a
compact public view of the next validation and implementation gates.

## June 2026: RDNA3 Optimization Complete

- DP4A tensor-core GEMM for finite-u8 (v_dot4_i32_iu8, ROCm 7.1 assembler fix)
- VOPD DPP export + Combined final-output for i64/u64 bounded export
- Export layer complete: Garner fast CRT (prefix 1-8), status elision (prefix 9+)
- WMMA skinny GEMV dispatch (AMDGPU builtins)
- Pack layer complete: persistent, coalesced, non-temporal loads
- Zero-skip + adaptive prefix infrastructure
- HIP graph full-path replay (bounded + finite-u8)
- 243-capture sweep, 5 backends, 0 failures

## 0.1.x Public Surface

- Keep the C ABI as the primary supported API.
- Keep the C++ wrapper limited to small RAII handle support.
- Stabilize CPU package export, examples, downstream CMake smoke, and release
  metadata.
- Keep public backend names explicit: `cpu-reference`, `hip-direct`,
  `hip-vector-alu-int64`, `wrap64-byte-limb`, `hipblaslt`, `ck`, and
  `rocwmma`.

## Next GPU Gates

- Improve Windows `gfx1100` packing, residency, and matrix-engine utilization.
- Promote only same-contract, schema-valid, reviewed benchmark wins.
- Preserve CPU differentials for every GPU semantic path.

## Platform Gates

- Validate Linux ROCm on real supported Radeon hardware.
- Validate Instinct CDNA targets separately before claiming production
  readiness.
- Add broader profiling, power, and reproducibility gates after correctness and
  backend selection are stable.

## Deferred

- Public optimized wrap64 matrix-engine backend.
- AMDGPU builtin hot kernels.
- Multi-GPU modulus-split experiments.
- Full C++ API surface beyond RAII handles.
