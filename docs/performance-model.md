# Performance Model Notes

The research spec defines the long-term performance model. The current scaffold
does not make optimized GPU performance claims.

Current benchmark shell:

```powershell
build\windows-msvc-hip-debug\rns8-bench.exe --backend cpu --semantics bounded-i64 --m 64 --n 64 --k 64 --warmups 1 --repeats 5 --seed 1
build\windows-msvc-hip-debug\rns8-bench.exe --backend hip-direct --semantics bounded-u64 --m 16 --n 16 --k 16 --warmups 1 --repeats 3 --seed 1
build\windows-msvc-hip-debug\rns8-bench.exe --backend wrap64-byte-limb --semantics wrap-u64 --m 16 --n 16 --k 16 --warmups 1 --repeats 5 --seed 7
build\windows-msvc-hip-debug\rns8-bench.exe --backend hip-direct --semantics wrap-u64 --m 4 --n 4 --k 8 --warmups 1 --repeats 2 --seed 11
build\windows-msvc-hip-debug\rns8-bench.exe --backend hip-direct --semantics bounded-u64 --m 16 --n 16 --k 16 --tile-m 64 --tile-n 64 --warmups 1 --repeats 3 --seed 1
build\windows-msvc-hip-debug\rns8-bench.exe --backend hip-direct --semantics bounded-u64 --bound-mode per-tile --require-adaptive-execution --m 65 --n 65 --k 64 --tile-m 64 --tile-n 64 --warmups 1 --repeats 3 --seed 7
```

The benchmark reports:

- stable `schema_version` metadata,
- requested and selected backend,
- selected kernel when a backend can report it; direct-HIP adaptive per-tile
  bounded captures report `direct_hip_tiled_rns_gemm_v1`,
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
  with the CMake configure-time value used only as a fallback when git is
  unavailable,
- compiler version,
- configured AMDGPU target list,
- HIP device identity and runtime metadata when using the direct HIP backend,
- clock/power settings when available; currently `null`,
- comparison baseline and derived TOPS-equivalent when reviewed baselines exist;
  currently `null`,
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

Bounded i64/u64 captures use persistent RNS matrices, a nonzero CRT prefix, and
`epilogue_type: "crt_export"`. Strict wrap captures use byte-limb storage with
either the CPU byte-limb reference backend or the direct-HIP Comba correctness
path: `semantics: "wrap_u64_mod_2_64"`, `bound_kind: "none"`, `bound: 0`,
`prefix: 0`, `packed_layout_version: "byte_limb_v1"`, and `epilogue_type:
"low64_wrap_export"`. For schema compatibility, wrap captures keep the host
timing keys `rns_gemm` and `crt_export`; their phase notes identify these as
`rns8_gemm_wrap_u64` and `rns8_export_wrap_u64`.
`per_modulus_gemm_estimate_applicable` is `false` for wrap captures.

Schema version 3 keeps the legacy top-level average fields and adds a measured
`scheduling` phase for the public schedule-info query. Schema version 2 captures
remain valid without this phase. The preferred timing contract for new captures
is:

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

Schema v3 added `timing_metadata.phase_availability`. Schema v4 keeps v1/v2/v3
compatibility and adds per-tile adaptive bounded capture metadata:
`bound_mode`, `tile_bounds_u64`, non-null `selected_kernel`, strict adaptive
schedule consistency, and an adaptive direct-HIP event timing source scope. The
current RNS bounded paths report `reduction.timed=false` with
`scope: "fused_into_rns_gemm"` because centered residue reduction happens inside
the `rns_gemm` phase. Strict wrap64 byte-limb captures report
`scope: "not_applicable_wrap64_byte_limb"`. Do not synthesize a reduction timing
from GEMM time.

Use `tools\benchmark_schema.py` to validate benchmark captures before using
them as comparison evidence. The validator enforces schema v2/v3/v4 required
fields, raw timing array lengths against `repeats`, average/median/p95
consistency, v3+ phase-availability metadata, v4 per-tile adaptive metadata,
GPU event timing nullability or completeness, and the strict wrap64
`prefix: 0` / `packed_layout_version: "byte_limb_v1"` metadata contract. It
also checks schedule metadata and keeps a compatibility check for legacy v1
captures that only expose the older top-level timing fields.

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
plus schema-compatible aggregate aliases:

- `pack_h2d`
- `pack_kernel`
- `pack`
- `wrap64_comba_gemm_kernel`
- `rns_gemm`
- `wrap64_export_kernel`
- `wrap64_export_d2h`
- `crt_export`

The wrap64 direct-HIP event source scope is
`direct_hip_wrap64_comba_default_stream_backend_operation_groups`. It describes
the one-thread-per-output Comba correctness path, not an optimized byte-GEMM
backend.

Host timings and HIP event timings answer different questions. Host
`std::chrono::steady_clock` timings include API dispatch, CPU scheduling,
allocations, and synchronous host-side overhead. HIP event timings record
default-stream backend operation groups only. Do not compare event timings to
host timings as replacements, and do not replace nullable event fields with
host wall-clock timings or estimates.

Future benchmark work must add deeper scheduler internals, reviewed raw sweeps,
comparison baselines, and performance gates before any speedup claims are made.

`tools/result_compare.py` validates both captures before comparing host timing
phases for schema v1/v2/v3/v4 captures. Its contract check includes backend,
selected kernel, semantics, bound mode, bounds, tile-bound source/order/min/max
and hash, shape, prefix, seed, warmups/repeats, input distribution, timing
source, epilogue, packed layout, schedule metadata, compiler, configured
target, and HIP device/runtime fields when present. It also compares
`gpu_event_timing_summary_us` phases only when both captures set
`timing_metadata.gpu_event_timing=true` and report the same event timing source,
source scope, and GPU event phase order. Per-modulus timing rows are flagged as
not applicable when a capture says `per_modulus_gemm_estimate_applicable:
false`.
