# RNS8 Roadmap Status

Status date: 2026-06-02

This document records live implementation status against the current roadmap.
It is not a substitute for `docs/RNS8_RESEARCH_SPEC.md`; when status and spec
disagree, the spec remains the target and this file identifies the gap.

## Implemented And Verified

- Phase 0 host foundation: C ABI, CMake targets, CPU reference, tests, tools,
  dependency checker, and benchmark/result comparison shell.
- Phase 1 Windows direct HIP bring-up on Radeon RX 7900 XTX / `gfx1100`:
  explicit hipcc integration, device inspection, residue conversion,
  one-modulus ring GEMM, K-block splitting, and CPU differential tests.
- Device-resident direct HIP RNS matrices: HIP matrices own device residue
  buffers, upload buffers, export buffers, and status buffers; `rns8_gemm_rns`
  consumes device residues directly instead of copying host residues in the hot
  GEMM path.
- Direct HIP fused INT32-to-centered-residue reduction: the correctness kernel
  reduces each K block to the centered residue in the kernel and does not write
  full INT32 output matrices to global memory.
- Bounded i64/u64 GPU export: direct HIP reconstructs bounded i64/u64 output on
  device with a compact 128-bit Garner path for prefixes up to 16, reports range
  errors through device status, and copies compact output into the caller's host
  layout.
- Fixed 9-modulus bounded i64/u64 GEMM: CPU and direct HIP public one-shot
  bounded APIs pass CPU differential tests, including full-width boundary and
  K-block cases.
- Persistent RNS behavior: public matrix/workspace APIs exercise persistent A/B/C
  storage and verify device pointer stability through pack, GEMM, and export.
- Exact-wide RNS output: exact-wide signed and unsigned semantics accept
  `RNS8_BOUND_NONE`, compute persistent RNS output, and reject bounded-looking
  CRT metadata. CPU and direct HIP RNS output are checked against
  Boost.Multiprecision residue oracles.
- Strict wraparound reference: internal CPU byte-limb Comba product and GEMM-cell
  reference paths match Boost.Multiprecision low-64-bit results. The public
  wraparound backend remains unsupported.
- Benchmark schema v2: benchmark captures include stable schema version, command
  line, git commit, compiler/HIP/device metadata, raw timings, summaries, null
  placeholders for unavailable fields, and comparison-tool support for v1/v2.
- Platform readiness reporting: dependency checker reports host readiness gates,
  Windows HIP/RDNA3 gates, Linux ROCm gates as not applicable on Windows, and
  optional accelerator components as candidate evidence only.

## Not Yet Implemented

- Optimized matrix-engine HIP kernels. The direct HIP kernels are correctness
  bring-up kernels, not performance evidence.
- HIP event timing around individual GPU phases. Current benchmark timings are
  host wall-clock timings.
- Per-tile adaptive bounds, per-tile prefix selection, grouped scheduling, and
  adaptive skip behavior.
- hipBLASLt, CK, rocWMMA, or AMDGPU builtin accelerator backends. They remain
  feature-detected future paths and are not correctness requirements.
- Exact-wide scalar or multi-limb export ABI, CPU export, or GPU export.
- Public strict `mod 2^64` byte-limb backend, unsigned-byte packing, GPU byte
  GEMMs, signed-INT8 bias correction, and GPU differential tests.
- Linux ROCm direct HIP parity, Linux hipBLASLt baseline, Linux CK validation,
  Instinct CDNA validation, profiling, power runs, and cluster reproducibility
  notes. These require a real Linux ROCm host with supported hardware.
- Architecture hot kernels, autotune selection, and production performance gate
  evaluation.
- Multi-GPU modulus split experiments.

## Latest Evidence

- `ctest --test-dir build/cpu-debug --output-on-failure`: 32/32 passed; HIP
  smoke tests skipped in CPU-only build.
- `ctest --preset windows-debug --output-on-failure`: 32/32 passed on
  `gfx1100`.
- `build\windows-msvc-hip-debug\rns8-inspect.exe --backend hip-direct --json`:
  detected AMD Radeon RX 7900 XTX / `gfx1100`.
- `build\windows-msvc-hip-debug\rns8-verify.exe --hip-smoke`: CPU reference
  verification and direct HIP smoke passed.
- Benchmark captures are kept under `temp/`:
  `rns8-cpu-bounded-i64.json`, `rns8-cpu-bounded-u64.json`,
  `rns8-hip-bounded-u64.json`, and `rns8-hip-bounded-u64-repeat.json`.
  They are raw evidence only and do not establish a performance claim.
