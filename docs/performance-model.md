# Performance Model Notes

The research spec defines the long-term performance model. The current scaffold
does not make optimized GPU performance claims.

Current benchmark shell:

```powershell
build\windows-msvc-hip-debug\rns8-bench.exe --backend cpu --semantics bounded-i64 --m 64 --n 64 --k 64 --warmups 1 --repeats 5 --seed 1
build\windows-msvc-hip-debug\rns8-bench.exe --backend hip-direct --semantics bounded-u64 --m 16 --n 16 --k 16 --warmups 1 --repeats 3 --seed 1
build\windows-msvc-hip-debug\rns8-bench.exe --backend wrap64-byte-limb --semantics wrap-u64 --m 16 --n 16 --k 16 --warmups 1 --repeats 5 --seed 7
```

The benchmark reports:

- stable `schema_version` metadata,
- requested and selected backend,
- selected kernel when a backend can report it; currently `null`,
- semantic contract,
- matrix shape,
- layout, K-block size, tile size, epilogue type, and packed layout version
  when exposed,
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
- average packing time,
- average persistent RNS GEMM time,
- average per-modulus GEMM estimate for RNS captures,
- average CRT export time,
- average end-to-end time for the measured phases,
- raw per-repeat timing arrays plus average, median, and p95 summaries.

Bounded i64/u64 captures use persistent RNS matrices, a nonzero CRT prefix, and
`epilogue_type: "crt_export"`. Strict wrap captures use the explicit CPU
byte-limb backend only: `semantics: "wrap_u64_mod_2_64"`, `bound_kind:
"none"`, `bound: 0`, `prefix: 0`, `packed_layout_version: "byte_limb_v1"`,
and `epilogue_type: "low64_wrap_export"`. For schema compatibility, wrap
captures keep the host timing keys `rns_gemm` and `crt_export`; their phase
notes identify these as `rns8_gemm_wrap_u64` and `rns8_export_wrap_u64`.
`per_modulus_gemm_estimate_applicable` is `false` for wrap captures.

Schema version 2 keeps the legacy top-level average fields, but the preferred
timing contract is:

```json
"raw_timings_us": {
  "planning": [123],
  "matrix_alloc": [456],
  "pack": [10, 11],
  "rns_gemm": [20, 21],
  "crt_export": [30, 31],
  "end_to_end": [60, 63]
},
"timing_summary_us": {
  "planning": {"avg": 123, "median": 123, "p95": 123},
  "pack": {"avg": 10.5, "median": 11, "p95": 11}
}
```

Current direct-HIP benchmark timings use host `std::chrono::steady_clock`.
They include the current correctness backend's synchronization, first-use
matrix-owned upload/export buffer allocation when it occurs, host/device copies,
kernel launches, fused residue reduction, and GPU bounded export.

## Direct-HIP event timing status

The benchmark enables direct-HIP event timing through internal backend hooks for
measured repeats. Events are recorded inside the backend around operation groups
that the public benchmark phase cannot otherwise see: pack upload, pack kernel,
the per-modulus RNS GEMM kernel group, export status initialization, export
kernel, export status readback, and export device-to-host copy.

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

For direct-HIP captures with complete event data, `gpu_event_timing` is `true`,
`gpu_event_timings_us` contains raw per-repeat arrays, and
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

Host timings and HIP event timings answer different questions. Host
`std::chrono::steady_clock` timings include API dispatch, CPU scheduling,
allocations, and synchronous host-side overhead. HIP event timings record
default-stream backend operation groups only. Do not compare event timings to
host timings as replacements, and do not replace nullable event fields with
host wall-clock timings or estimates.

Future benchmark work must add finer scheduling overhead capture, reviewed raw
sweeps, comparison baselines, and performance gates before any speedup claims
are made.

`tools/result_compare.py` compares host timing phases for schema v1/v2 captures.
Its contract check includes backend, semantics, bounds, shape, prefix, seed,
warmups/repeats, input distribution, timing source, epilogue, packed layout,
compiler, configured target, and HIP device/runtime fields when present. It
also compares `gpu_event_timing_summary_us` phases only when both captures set
`timing_metadata.gpu_event_timing=true` and report the same event timing source
and source scope. Per-modulus timing rows are flagged as not applicable when a
capture says `per_modulus_gemm_estimate_applicable: false`.
