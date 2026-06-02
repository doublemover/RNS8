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
  full INT32 output matrices to global memory. Centered-range correction uses
  source-level mask arithmetic instead of source-level `if` branches.
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
- Plan schedule inspection: bounded and wrap64 plans expose output tile grid,
  required prefix, selected prefix, and prefix-group metadata through public ABI
  queries. Global bounded plans still use one fixed selected prefix for every
  tile. CPU reference and direct HIP per-tile bounded plans copy tile bounds
  into the plan, select variable exact prefixes, report adaptive prefix/skip
  metadata, execute only selected per-tile prefixes, and export with
  tile-local bounds.
- Exact-wide RNS output: exact-wide signed and unsigned semantics accept
  `RNS8_BOUND_NONE`, compute persistent RNS output, and reject bounded-looking
  CRT metadata. CPU and direct HIP RNS output are checked against
  Boost.Multiprecision residue oracles. CPU little-endian limb export is
  implemented for signed two's-complement and unsigned magnitude output. Direct
  HIP exports signed and unsigned exact-wide limbs from device-resident RNS
  matrices without synchronizing host residue storage.
- Strict wraparound byte-limb backend: CPU one-shot and persistent `mod 2^64`
  GEMM use byte-limb matrix storage and the Comba reference, match
  Boost.Multiprecision low-64-bit results, and keep RNS/CRT APIs fenced off
  from wrap descriptors. The CPU reference also includes an exhaustively tested
  signed-INT8 correction helper for reconstructing unsigned byte products when
  future accelerator paths expose only signed INT8 products, plus a 36-byte-GEMM
  decomposition oracle that matches Boost low-64 results and the current Comba
  reference.
- Public direct-HIP strict wrap64 byte-limb correctness path: HIP_DIRECT wrap
  matrices own device byte-limb buffers, pack/GEMM/export consume those buffers
  without RNS residue allocation, public one-shot and persistent APIs match the
  CPU byte-limb reference, and padded host export layouts are tested. The GEMM
  kernel is now a one-thread-per-output byte-GEMM36 correctness path that sums
  low-product byte diagonals with device-side signed-INT8 correction and then
  carries into the low 64 bits; it is not an optimized matrix-engine byte-GEMM
  accelerator path.
- Direct-HIP per-tile bounded adaptive correctness path: HIP_DIRECT bounded
  plans with `RNS8_BOUND_PER_TILE_MAX_ABS` or
  `RNS8_BOUND_PER_TILE_MAX_UNSIGNED` use grouped direct HIP tile launches for
  selected prefixes and tile-local device CRT export. Tests compare signed and
  unsigned output against the CPU reference, cover tile-local range errors,
  prove skipped high-prefix residue planes remain untouched, and keep matrices
  device-resident through GEMM/export.
- Benchmark schema v4: benchmark captures include stable schema version, command
  line, live git commit, compiler/HIP/device metadata, raw timings, summaries,
  null placeholders for unavailable fields, direct-HIP GPU event timing arrays
  when complete, explicit unavailable metadata when event timing is not
  applicable, strict wrap64 CPU and direct-HIP byte-limb benchmark metadata,
  fixed-prefix schedule metadata, measured schedule-info query timing,
  explicit phase-availability metadata for fused or not-applicable reduction,
  direct-HIP per-tile adaptive bounded capture metadata, schema validation
  tooling, and comparison-tool support for v1/v2/v3/v4 plus capture-specific
  GPU event phase orders. Adaptive captures are evidence for the direct-HIP
  tiled correctness path only; they are not optimized matrix-engine performance
  claims.
- Platform readiness reporting: dependency checker reports host readiness gates,
  Windows HIP/RDNA3 gates, Linux ROCm gates as not applicable on Windows, and
  optional accelerator components as candidate evidence only. Linux presets keep
  active offload targets separate from RDNA/CDNA coverage metadata, and shallow
  hipBLASLt/CK/rocWMMA probes report headers, libraries, tools, and CMake module
  evidence without enabling accelerator backends. Opt-in Python and CMake
  accelerator probe modes record compile/link/runtime evidence under `temp/` or
  probe-only build directories while keeping all accelerator backend enablement
  disabled.

## Not Yet Implemented

- Optimized matrix-engine HIP kernels, reciprocal-reduction kernels, and
  instruction-level validation. The direct HIP kernels are correctness bring-up
  kernels, not performance evidence.
- hipBLASLt, CK, rocWMMA, or AMDGPU builtin accelerator backends. They remain
  feature-detected future paths and are not correctness requirements.
- Device capability and exact CPU differential probes for hipBLASLt, CK,
  rocWMMA, or AMDGPU builtin backends. Current opt-in probes are compile/link
  and tiny runtime evidence only.
- Optimized strict `mod 2^64` GPU byte GEMMs, accelerator integration of the
  signed-INT8 correction algebra, and production GPU differential tests.
- Linux ROCm direct HIP parity, Linux hipBLASLt baseline, Linux CK validation,
  Instinct CDNA validation, profiling, power runs, and cluster reproducibility
  notes. These require a real Linux ROCm host with supported hardware.
- Architecture hot kernels, autotune selection, and production performance gate
  evaluation.
- Multi-GPU modulus split experiments.

## Latest Evidence

- `ctest --test-dir build/cpu-debug --output-on-failure`: 53/53 passed; HIP
  smoke tests skipped in CPU-only build.
- `ctest --preset windows-debug --output-on-failure`: 53/53 passed on
  `gfx1100`.
- The CPU and Windows HIP test passes include plan schedule inspection coverage
  for fixed-prefix bounded tile groups, CPU per-tile adaptive bounded groups,
  copied per-tile bound lifetime, wrap64 prefix-zero byte-limb scheduling, and
  tile-size validation.
- The CPU test pass includes bounded signed and unsigned one-shot GEMMs over
  2x2 output tile grids whose tiles use selected prefixes 1, 2, 3, and 4 and
  export against tile-local bounds.
- The Windows HIP test pass includes prefix-20 bounded signed and unsigned GPU
  export checks against the CPU reference, including `INT64_MIN` and
  `UINT64_MAX` boundary outputs.
- The Windows HIP test pass includes direct-HIP per-tile bounded signed and
  unsigned output comparisons against the CPU reference, tile-local range-error
  checks, padded host export sentinels, schedule parity checks, and skipped
  high-prefix residue plane checks.
- The Windows HIP test pass includes a direct HIP one-modulus centered
  correction boundary case that compares negative, threshold, and near-zero
  residues against the CPU ring-GEMM reference.
- The Windows HIP test pass includes
  `private HIP wrap64 byte-limb GEMM matches CPU reference` and
  `direct HIP public wrap64 byte-limb path matches CPU reference`, covering both
  the low-level kernel smoke and the public HIP_DIRECT one-shot/persistent
  byte-limb APIs against the CPU reference.
- The Windows HIP test pass also includes signed and unsigned exact-wide RNS
  differential checks against CPU residues plus direct HIP exact-wide limb
  export checks for padded host layouts, range errors, and signed
  two's-complement sign extension.
- `build\windows-msvc-hip-debug\rns8-inspect.exe --backend hip-direct --json`:
  detected AMD Radeon RX 7900 XTX / `gfx1100`.
- `build\windows-msvc-hip-debug\rns8-inspect.exe --backend wrap64-byte-limb
  --json`: reported the CPU wrap64 byte-limb reference backend.
- `build\windows-msvc-hip-debug\rns8-verify.exe --hip-smoke`: CPU reference
  verification and direct HIP pack, ring, bounded GEMM, adaptive bounded GEMM,
  and wrap64 smoke passed.
- `python tools\check_dependencies.py`: host readiness and Windows RDNA3 direct
  HIP gates passed; Linux ROCm/Instinct gates reported not applicable on this
  Windows host. hipBLASLt was reported as candidate evidence only on this host;
  CK and rocWMMA remained not ready, and none were promoted to correctness
  requirements.
- `python tools\check_dependencies.py --accelerator-probes --json`: host
  readiness stayed true while accelerator gates stayed `ok=false`. CK and
  rocWMMA probes did not run because headers were not discovered. hipBLASLt was
  candidate evidence but the tiny Windows hipcc/lld-link probe failed because
  this HIP SDK install exposes `libhipblaslt.dll.a` rather than a linkable
  `hipblaslt.lib`; backend enablement remained disabled.
- `cmake --preset windows-msvc-hip-accelerator-probe`: configured successfully,
  reported hipBLASLt header evidence, CK/rocWMMA not discovered, and
  accelerator backend enablement disabled.
- `cmake --build --preset windows-accelerator-probe --target rns8-inspect`:
  built the direct-HIP inspection binary from the probe preset while keeping all
  accelerator backend enablement flags disabled.
- Benchmark captures are kept under `temp/`:
  `rns8-cpu-bounded-i64.json`, `rns8-cpu-bounded-u64.json`,
  `rns8-hip-bounded-u64.json`, `rns8-hip-bounded-u64-repeat.json`,
  `rns8-hip-bounded-u64-event-smoke.json`, and
  `rns8-hip-bounded-u64-schedule-smoke.json`. Schema v3 smoke captures are
  `rns8-v3-cpu-bounded-i64.json`, `rns8-v3-hip-bounded-u64*.json`,
  `rns8-v3-wrap-u64.json`, and `rns8-v3-hip-wrap-u64.json`.
- `temp\rns8-hip-bounded-u64-event-smoke.json`: checked schema v2, `gfx1100`,
  live `git_commit`, `gpu_event_timing=true`, and nonnegative direct-HIP event
  arrays for `pack`, `rns_gemm`, and `crt_export`.
- `temp\rns8-hip-bounded-u64-schedule-smoke.json`: checked schema v2 with
  `--tile-m 64 --tile-n 64`, fixed selected prefix metadata, required prefix
  metadata, one prefix group, and `adaptive_execution_applied=false`.
- `python tools\result_compare.py --json temp\rns8-hip-bounded-u64.json
  temp\rns8-hip-bounded-u64-repeat.json`: same-contract comparison passed,
  including matching fixed-prefix schedule metadata and compatible GPU event
  timing metadata. Captures are raw evidence only and do not establish a
  performance claim.
- `python tools\benchmark_schema.py temp\rns8-v3-cpu-bounded-i64.json
  temp\rns8-v3-hip-bounded-u64.json
  temp\rns8-v3-hip-bounded-u64-repeat.json
  temp\rns8-v3-hip-bounded-u64-repeat2.json temp\rns8-v3-wrap-u64.json
  temp\rns8-v3-hip-wrap-u64.json`: all runtime captures validated as schema
  v3, including measured `scheduling` timing and explicit reduction
  availability metadata.
- `python tools\result_compare.py --json
  temp\rns8-v3-hip-bounded-u64-repeat.json
  temp\rns8-v3-hip-bounded-u64-repeat2.json`: same-contract schema v3
  comparison passed with comparable direct-HIP GPU event phase order. Captures
  are raw evidence only and do not establish a performance claim.
- `temp\rns8-v4-hip-bounded-u64-adaptive.json` and
  `temp\rns8-v4-hip-bounded-u64-adaptive-repeat.json`: direct-HIP per-tile
  adaptive bounded captures validated as schema v4, with exact seeded-input
  tile-bound metadata, `selected_kernel=direct_hip_tiled_rns_gemm_v1`,
  `adaptive_execution_applied=true`, complete HIP event timing in
  `direct_hip_bounded_adaptive_default_stream_backend_operation_groups`, and
  same-contract `tools\result_compare.py --json` comparison including matching
  tile-bound hash. Captures are raw evidence only and do not establish a
  performance claim.
- `temp\rns8-wrap-u64-bench.json` and
  `temp\rns8-wrap-u64-bench-repeat.json`: fixed-seed strict wrap64 CPU
  byte-limb captures with `prefix=0`, `bound_kind=none`,
  `packed_layout_version=byte_limb_v1`, nullable GPU event timing, and successful
  `tools\result_compare.py --json` contract comparison. Captures are raw
  evidence only and do not establish a performance claim.
- `temp\rns8-hip-wrap-u64-event-smoke.json` and
  `temp\rns8-hip-wrap-u64-event-smoke-repeat.json`: fixed-seed strict wrap64
  direct-HIP byte-limb captures with `prefix=0`, `bound_kind=none`,
  `packed_layout_version=byte_limb_v1`, `gpu_event_timing=true`, and
  wrap64-specific event phases from the older Comba correctness kernel. Current
  schema v4 HIP wrap64 captures report
  `selected_kernel=direct_hip_wrap64_byte_gemm36_correctness_v1`,
  `wrap64_byte_gemm36_kernel`, `wrap64_export_kernel`, and
  `wrap64_export_d2h` event phases. Captures are raw evidence only and do not
  establish a performance claim.
- `temp\rns8-v4-hip-wrap-u64-byte-gemm36.json` and
  `temp\rns8-v4-hip-wrap-u64-byte-gemm36-repeat.json`: fixed-seed strict
  wrap64 direct-HIP byte-GEMM36 correctness captures validated as schema v4.
  `tools\result_compare.py --json` reported the same selected kernel, event
  source scope, GPU event phase order, shape, seed, and semantic contract.
  Captures are raw evidence only and do not establish a performance claim.
- `python tools\test_benchmark_schema.py`: benchmark schema fixture self-test
  passed, including malformed raw timing length, GPU event summary, invalid
  schedule metadata, wrap64 prefix, event-nullability, v3 scheduling, v3
  reduction-availability, and v4 per-tile adaptive contract rejection checks.
- `python tools\benchmark_schema.py` validated representative v2/v3 CPU,
  direct HIP, and wrap64 captures under `temp\`, plus synthetic v1/v2/v3/v4
  fixtures under `tests\fixtures\benchmark_schema\`.
