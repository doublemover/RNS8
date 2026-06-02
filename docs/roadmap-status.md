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
  device with a fixed three-limb Garner path for prefixes up to 20, reports
  range errors through device status, handles the full signed output range
  including `INT64_MIN`, and copies compact output into the caller's host
  layout.
- Fixed 9-modulus bounded i64/u64 GEMM: CPU and direct HIP public one-shot
  bounded APIs pass CPU differential tests, including full-width boundary and
  K-block cases.
- Persistent RNS behavior: public matrix/workspace APIs exercise persistent A/B/C
  storage and verify device pointer stability through pack, GEMM, and export.
- Exact-wide RNS output: exact-wide signed and unsigned semantics accept
  `RNS8_BOUND_NONE`, compute persistent RNS output, and reject bounded-looking
  CRT metadata. CPU and direct HIP RNS output are checked against
  Boost.Multiprecision residue oracles. CPU little-endian limb export is
  implemented for signed two's-complement and unsigned magnitude output.
- Strict wraparound byte-limb backend: CPU one-shot and persistent `mod 2^64`
  GEMM use byte-limb matrix storage and the Comba reference, match
  Boost.Multiprecision low-64-bit results, and keep RNS/CRT APIs fenced off
  from wrap descriptors.
- Private direct-HIP strict wrap64 byte-limb smoke: a compiled direct HIP
  one-thread-per-output Comba kernel matches the CPU byte-limb reference. This
  is not public HIP wrap64 backend support and is not an optimized byte-GEMM
  accelerator path.
- Benchmark schema v2: benchmark captures include stable schema version, command
  line, live git commit, compiler/HIP/device metadata, raw timings, summaries,
  null placeholders for unavailable fields, direct-HIP GPU event timing arrays
  when complete, explicit unavailable metadata when event timing is not
  applicable, strict wrap64 CPU byte-limb benchmark metadata, schema validation
  tooling, and comparison-tool support for v1/v2.
- Platform readiness reporting: dependency checker reports host readiness gates,
  Windows HIP/RDNA3 gates, Linux ROCm gates as not applicable on Windows, and
  optional accelerator components as candidate evidence only.

## Not Yet Implemented

- Optimized matrix-engine HIP kernels. The direct HIP kernels are correctness
  bring-up kernels, not performance evidence.
- Per-tile adaptive bounds, per-tile prefix selection, grouped scheduling, and
  adaptive skip behavior.
- hipBLASLt, CK, rocWMMA, or AMDGPU builtin accelerator backends. They remain
  feature-detected future paths and are not correctness requirements.
- Exact-wide GPU export. Current exact-wide HIP export synchronizes residues to
  host and uses the CPU Boost.Multiprecision limb exporter.
- Public/optimized strict `mod 2^64` GPU byte GEMMs, signed-INT8 bias
  correction, and production GPU differential tests.
- Linux ROCm direct HIP parity, Linux hipBLASLt baseline, Linux CK validation,
  Instinct CDNA validation, profiling, power runs, and cluster reproducibility
  notes. These require a real Linux ROCm host with supported hardware.
- Architecture hot kernels, autotune selection, and production performance gate
  evaluation.
- Multi-GPU modulus split experiments.

## Latest Evidence

- `ctest --test-dir build/cpu-debug --output-on-failure`: 39/39 passed; HIP
  smoke tests skipped in CPU-only build.
- `ctest --preset windows-debug --output-on-failure`: 39/39 passed on
  `gfx1100`.
- The Windows HIP test pass includes prefix-20 bounded signed and unsigned GPU
  export checks against the CPU reference, including `INT64_MIN` and
  `UINT64_MAX` boundary outputs.
- The Windows HIP test pass included
  `private HIP wrap64 byte-limb GEMM matches CPU reference`, which exercises a
  compiled direct-HIP byte-limb Comba smoke kernel against the CPU reference.
- The Windows HIP test pass also includes signed and unsigned exact-wide RNS
  differential checks against CPU residues and CPU limb export.
- `build\windows-msvc-hip-debug\rns8-inspect.exe --backend hip-direct --json`:
  detected AMD Radeon RX 7900 XTX / `gfx1100`.
- `build\windows-msvc-hip-debug\rns8-inspect.exe --backend wrap64-byte-limb
  --json`: reported the CPU wrap64 byte-limb reference backend.
- `build\windows-msvc-hip-debug\rns8-verify.exe --hip-smoke`: CPU reference
  verification and direct HIP smoke passed.
- `python tools\check_dependencies.py`: host readiness and Windows RDNA3 direct
  HIP gates passed; Linux ROCm/Instinct gates reported not applicable on this
  Windows host.
- Benchmark captures are kept under `temp/`:
  `rns8-cpu-bounded-i64.json`, `rns8-cpu-bounded-u64.json`,
  `rns8-hip-bounded-u64.json`, `rns8-hip-bounded-u64-repeat.json`, and
  `rns8-hip-bounded-u64-exactwide-limb-export.json`.
- `temp\rns8-hip-bounded-u64-event-smoke.json`: checked schema v2, `gfx1100`,
  live `git_commit`, `gpu_event_timing=true`, and nonnegative direct-HIP event
  arrays for `pack`, `rns_gemm`, and `crt_export`.
- `python tools\result_compare.py --json temp\rns8-hip-bounded-u64.json
  temp\rns8-hip-bounded-u64-event-smoke.json`: same-contract comparison passed;
  event-summary comparison is enabled only when both captures carry compatible
  GPU event timing metadata. Captures are raw evidence only and do not establish
  a performance claim.
- `temp\rns8-wrap-u64-bench.json` and
  `temp\rns8-wrap-u64-bench-repeat.json`: fixed-seed strict wrap64 CPU
  byte-limb captures with `prefix=0`, `bound_kind=none`,
  `packed_layout_version=byte_limb_v1`, nullable GPU event timing, and successful
  `tools\result_compare.py --json` contract comparison. Captures are raw
  evidence only and do not establish a performance claim.
- `python tools\test_benchmark_schema.py`: benchmark schema fixture self-test
  passed, including malformed raw timing length, GPU event summary, wrap64
  prefix, and event-nullability rejection checks.
- `python tools\benchmark_schema.py` validated representative v2 CPU, direct
  HIP, and wrap64 captures under `temp\`, plus synthetic v1/v2 fixtures under
  `tests\fixtures\benchmark_schema\`.
