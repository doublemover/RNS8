# Performance Model Notes

The research spec defines the long-term performance model. The current scaffold
does not make optimized GPU performance claims.

The commands and JSON contracts below are capture mechanics, not performance
baselines. A timing capture becomes comparison evidence only after the current
schema validator accepts it, the semantic contract matches the comparison
target, and a reviewed baseline exists for the same backend family, target, and
shape.

Current benchmark shell:

```powershell
build\windows-msvc-hip-debug\rns8-bench.exe --backend cpu --semantics bounded-i64 --m 64 --n 64 --k 64 --warmups 1 --repeats 5 --seed 1
build\windows-msvc-hip-debug\rns8-bench.exe --backend hip-direct --semantics bounded-u64 --m 16 --n 16 --k 16 --warmups 1 --repeats 3 --seed 1
build\windows-msvc-hip-debug\rns8-bench.exe --backend wrap64-byte-limb --semantics wrap-u64 --m 16 --n 16 --k 16 --warmups 1 --repeats 5 --seed 7
build\windows-msvc-hip-debug\rns8-bench.exe --backend hip-direct --semantics wrap-u64 --m 4 --n 4 --k 8 --warmups 1 --repeats 2 --seed 11
build\windows-msvc-hip-debug\rns8-bench.exe --backend hip-direct --semantics bounded-u64 --m 16 --n 16 --k 16 --tile-m 64 --tile-n 64 --warmups 1 --repeats 3 --seed 1
build\windows-msvc-hip-debug\rns8-bench.exe --backend hip-direct --semantics bounded-u64 --bound-mode per-tile --require-adaptive-execution --m 65 --n 65 --k 64 --tile-m 64 --tile-n 64 --warmups 1 --repeats 3 --seed 7
build\windows-msvc-hip-debug\rns8-bench.exe --backend hip-direct --semantics finite-u8-ring --modulus 255 --m 64 --n 64 --k 64 --warmups 1 --repeats 3 --seed 1
```

The benchmark reports:

- stable `schema_version` metadata,
- requested and selected backend,
- selected kernel reported by the plan backend metadata API,
- `backend_metadata` from `rns8_get_plan_backend_info`, including selected
  kernel, accelerator/correctness/matrix-engine booleans, compiled/exact/perf
  validation booleans, accelerator library/version, capability status,
  epilogue mode, workspace mode, workspace byte requirement, ISA evidence, and
  autotune key,
- semantic contract,
- bound mode plus per-tile bound source/order/min/max/hash metadata when the
  capture uses `RNS8_BOUND_PER_TILE_*`,
- matrix shape,
- layout, K-block size, tile size, epilogue type, and packed layout version
  when exposed,
- schedule metadata from `rns8_get_plan_schedule_info`, including tile grid,
  required prefix, selected prefix, prefix group count, and adaptive
  prefix/skip flags,
- fixed seed,
- warmup and repeat counts,
- prefix count,
- command line,
- git commit resolved from the configured source checkout at benchmark runtime,
  with the CMake configure-time value used only when git is
  unavailable,
- compiler version,
- configured AMDGPU target list,
- configured HIP toolchain metadata, including HIP enablement, HIP SDK/ROCm
  root, hipcc path, hipcc version captured from `hipcc --version`, and parsed
  SDK/ROCm root version when available,
- HIP device identity and runtime metadata when using the direct HIP backend,
- clock/power settings when available; currently `null`,
- structured comparison-baseline status. Current unreviewed captures use
  `comparison_baseline.status: "required_not_recorded"` and
  `speedup_claimed: false`, with explicit prerequisite baseline names for the
  same semantic contract. Bounded captures require at least
  `same_contract_cpu_reference` and
  `same_contract_direct_hip_vector_alu_int64`; accelerator captures also name
  the same-contract direct-HIP correctness baseline. Strict wrap64 captures
  require the CPU byte-limb reference and direct-HIP byte-GEMM36 baseline.
  finite-u8 captures require CPU reference and direct-HIP finite-u8 baselines.
  `derived_tops_equivalent` remains `null` until a reviewed same-contract
  baseline is attached,
- timing source, timing caveat, and structured timing metadata,
- explicit GPU event timing availability metadata and direct-HIP event timing
  arrays when backend hooks collect a complete repeat,
- one-time planning and matrix allocation time,
- one-time schedule metadata query time,
- average packing time,
- average persistent RNS GEMM time,
- average per-modulus GEMM estimate for RNS captures,
- average CRT export time,
- average end-to-end time for the measured phases,
- raw per-repeat timing arrays plus average, median, and p95 summaries.

Raw benchmark captures do not write production autotune cache entries. The
review path is `tools/benchmark_sweep.py --write-autotune-cache`, which first
validates schema, groups captures by same-contract semantics/shape/layout/
target/toolchain/input seed, requires the matching CPU/GPU baselines, and
writes only fastest reviewed accelerator winners. Cache entries are keyed by
`backend_metadata.autotune_key` and store backend, target, HIP SDK or
accelerator library version, shape, semantic contract, layout, prefix schedule
hash, K-block, tile size, epilogue, selected kernel, workspace bytes, reviewed
median timings, and validation status. Unreviewed raw captures are not
performance validation claims.

## Windows `gfx1100` release review snapshot

The first release review run on Windows `gfx1100` used release opt-in
hipBLASLt, CK, and rocWMMA builds plus fixed seed `20260602`, one warmup, and
one measured repeat for the full release matrices. Raw captures and temp cache
outputs live under `temp/benchmark-sweeps/windows-gfx1100-release-*` and
`temp/accelerator-release-smoke/`; they are intentionally not tracked.

Reviewed bounded global captures covered CPU reference, direct HIP,
`hip-vector-alu-int64`, hipBLASLt, CK, and rocWMMA for bounded i64/u64 square
shapes 64, 128, 512, and 1024. The review accepted two promotable temp cache
entries: bounded i64 512 selected rocWMMA
`rocwmma_i8_i32_signed_hot_residue_v1` at 2513 us end-to-end, and bounded i64
1024 selected CK `ck_wmma_cshuffle_i8_i32_centered_epilogue_v1` at 7838 us
end-to-end. Bounded u64 produced no promotable accelerator entries because
direct-HIP or vector-ALU baselines were faster for the reviewed shapes.

Reviewed adaptive bounded captures covered the default 65x65x64 and
1024x1024x1024 per-tile schedules with CPU, direct HIP, vector-ALU, CK, and
rocWMMA. Only the 65x65x64 adaptive cases promoted temp entries, both selecting
rocWMMA `rocwmma_i8_i32_signed_tiled_hot_residue_v1`: 1152 us for bounded i64
and 1238 us for bounded u64. The 1024 adaptive cases remained blocked by
direct-HIP/vector baselines.

Reviewed finite-u8 release captures covered ring moduli 251 and 255 plus field
modulus 251 for square shapes 64, 128, 512, and 1024. Ring modulus 251 selected
CK for 64, 128, and 1024, and rocWMMA for 512. Ring modulus 255 selected
rocWMMA for 64, 128, and 512, and hipBLASLt for 1024. Field modulus 251
selected CK for 64 and 128, and rocWMMA for 512 and 1024.

Reviewed wrap64 baseline captures kept `direct_hip_wrap64_byte_gemm36_tiled_2d_v3`
as the measured production GPU path for strict `mod 2^64`: 2128 us end-to-end
versus 49256 us for the CPU byte-limb reference at 64x64x64. No wrap64
matrix-engine candidate exists yet, so no wrap64 accelerator promotion was
made. AMDGPU builtins remain fail-fast because the release reviews did not
identify a shape requiring a builtin kernel with exact differentials, ISA
evidence, and better timings than CK/rocWMMA.

Bounded i64/u64 captures use persistent RNS matrices, a nonzero CRT prefix, and
`epilogue_type: "crt_export"`. Strict wrap captures use byte-limb storage with
either the CPU byte-limb reference backend or the direct-HIP tiled byte-limb
correctness path: `semantics: "wrap_u64_mod_2_64"`, `bound_kind: "none"`, `bound: 0`,
`prefix: 0`, `packed_layout_version: "byte_limb_v1"`, and `epilogue_type:
"low64_wrap_export"`. Wrap captures use the current host timing keys
`rns_gemm` and `crt_export`; their phase notes identify these as
`rns8_gemm_wrap_u64` and `rns8_export_wrap_u64`.
`per_modulus_gemm_estimate_applicable` is `false` for wrap captures.
Exact-wide captures, when added, must use exact-wide limb semantics and cannot
be normalized into bounded i64/u64 or strict wrap64 timing contracts.
finite-u8 captures use prefix-zero finite storage with
`semantics: "finite_ring_u8"` or `"finite_field_u8"`, an explicit
`finite_modulus`, `bound_kind: "none"`, `bound: 0`, and
`epilogue_type: "canonical_u8_export"`.

Schema version 4 is the only accepted tracked capture schema. Current captures
must carry an explicit integer `"schema_version": 4`; missing version fields are
rejected instead of inferred. Schema v4 requires `backend_metadata` to mirror
the top-level `selected_kernel`, so accelerator readiness and selected-kernel
claims are tied to the public plan API instead of free-form benchmark text.
Schema v4 also includes a measured `scheduling` phase for the public
schedule-info query. The timing contract is:

```json
"raw_timings_us": {
  "planning": [123],
  "scheduling": [4],
  "matrix_alloc": [456],
  "pack": [10, 11],
  "rns_gemm": [20, 21],
  "crt_export": [30, 31],
  "end_to_end": [60, 63]
},
"timing_summary_us": {
  "planning": {"avg": 123, "median": 123, "p95": 123},
  "scheduling": {"avg": 4, "median": 4, "p95": 4},
  "pack": {"avg": 10.5, "median": 11, "p95": 11}
}
```

Schema v4 includes `timing_metadata.phase_availability`, per-tile adaptive
bounded capture metadata:
`bound_mode`, `tile_bounds_u64`, non-null `selected_kernel`, strict adaptive
schedule consistency, configured HIP toolchain metadata, and exact direct-HIP
event timing source/scope validation. The
current RNS bounded paths report `reduction.timed=false` with
`scope: "fused_into_rns_gemm"` because centered residue reduction happens inside
the `rns_gemm` phase. Strict wrap64 byte-limb captures report
`scope: "not_applicable_wrap64_byte_limb"`. Do not synthesize a reduction timing
from GEMM time.

Use `tools\benchmark_schema.py` to validate benchmark captures before using
them as comparison evidence. The validator enforces schema v4 required fields,
raw timing array lengths against `repeats`, average/median/p95 consistency,
phase-availability metadata, per-tile adaptive metadata, GPU event timing
nullability or completeness, `gpu_event_phase_order: null` when events are
unavailable, explicit event phase order for event-enabled captures, exact
matching of event timing keys to that phase order, and the strict wrap64
`prefix: 0` / `packed_layout_version: "byte_limb_v1"` metadata contract. It
also checks schedule metadata. The CTest suite runs the schema self-test, all
tracked current schema fixtures, and a same-contract `result_compare.py` check
so retired schemas and stale event labels are not only rejected manually.

Current benchmark inputs are inspectable planning contracts. Global bounded
captures remain fixed-prefix contracts. With `--bound-mode per-tile`, the
benchmark computes exact per-output-tile bounds from the seeded A/B inputs
before plan creation, passes those bounds through `rns8_gemm_desc.tile_bounds`,
requires actual prefix grouping or prefix skipping, and emits
`adaptive_execution_applied=true` only for the direct-HIP tiled bounded path.
Strict wrap64 captures report prefix zero and no RNS prefix groups.

Current direct-HIP benchmark timings use host `std::chrono::steady_clock`.
They include the current correctness backend's synchronization, first-use
matrix-owned upload/export buffer allocation when it occurs, host/device copies,
kernel launches, fused residue reduction, and GPU bounded export.

## Direct-HIP event timing status

The benchmark enables direct-HIP event timing through internal backend hooks for
measured repeats. Events are recorded inside the backend around operation groups
that the public benchmark phase cannot otherwise see.

When the selected backend is not direct HIP, or when a complete expected event
set is not available, event fields remain nullable:

```json
"timing_metadata": {
  "gpu_event_timing": false,
  "gpu_event_timing_reason": "backend_not_hip_direct"
},
"gpu_event_timings_us": null,
"gpu_event_timing_summary_us": null
```

For bounded direct-HIP captures with complete event data, `gpu_event_timing` is
`true`, `gpu_event_timings_us` contains raw per-repeat arrays, and
`gpu_event_timing_summary_us` contains average, median, and p95 summaries for:

- `pack_h2d`
- `pack_kernel`
- `pack`
- `rns_gemm_kernel_group`
- `rns_gemm`
- `crt_export_status_memset`
- `crt_export_kernel`
- `crt_export_status_d2h`
- `crt_export_d2h`
- `crt_export`

For strict wrap64 direct-HIP captures, event timing uses wrap64-specific labels
plus current aggregate phase labels:

- `pack_h2d`
- `pack_kernel`
- `pack`
- `wrap64_byte_gemm36_tiled_2d_kernel`
- `rns_gemm`
- `wrap64_export_kernel`
- `wrap64_export_d2h`
- `crt_export`

The wrap64 direct-HIP event source scope is
`direct_hip_wrap64_byte_gemm36_default_stream_backend_operation_groups`. It
describes the tiled byte-limb correctness path, not an optimized matrix-engine
byte-GEMM backend.

Host timings and HIP event timings answer different questions. Host
`std::chrono::steady_clock` timings include API dispatch, CPU scheduling,
allocations, and synchronous host-side overhead. HIP event timings record
default-stream backend operation groups only. Do not compare event timings to
host timings as replacements, and do not replace nullable event fields with
host wall-clock timings or estimates.

Future benchmark work must add deeper scheduler internals, reviewed raw sweeps,
comparison baselines, and performance gates before any speedup claims are made.

`tools/result_compare.py` validates both captures before comparing host timing
phases for schema v4 captures. Its contract check includes backend,
selected kernel, semantics, bound mode, bounds, tile-bound source/order/min/max
and hash, shape, prefix, seed, warmups/repeats, input distribution, timing
source, epilogue, packed layout, schedule metadata, compiler, configured
target, and HIP device/runtime fields when present. It also compares
`gpu_event_timing_summary_us` phases only when both captures set
`timing_metadata.gpu_event_timing=true` and report the same event timing source,
source scope, and GPU event phase order. Per-modulus timing rows are flagged as
not applicable when a capture says `per_modulus_gemm_estimate_applicable:
false`.
