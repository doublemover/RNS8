# Performance Model Notes

The research spec defines the long-term performance model. The current scaffold
does not make optimized GPU performance claims.

Current benchmark shell:

```powershell
build\windows-msvc-hip-debug\rns8-bench.exe --backend cpu --semantics bounded-i64 --m 64 --n 64 --k 64 --warmups 1 --repeats 5 --seed 1
build\windows-msvc-hip-debug\rns8-bench.exe --backend hip-direct --semantics bounded-u64 --m 16 --n 16 --k 16 --warmups 1 --repeats 3 --seed 1
```

The benchmark reports:

- stable `schema_version` metadata,
- requested and selected backend,
- selected kernel when a backend can report it; currently `null`,
- semantic contract,
- matrix shape,
- layout, K-block size, tile size, epilogue type, and packed layout version
  when exposed; currently `null`,
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
- explicit GPU event timing availability metadata,
- one-time planning and matrix allocation time,
- average packing time,
- average persistent RNS GEMM time,
- average per-modulus GEMM estimate,
- average CRT export time,
- average end-to-end time for the measured phases,
- raw per-repeat timing arrays plus average, median, and p95 summaries.

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

HIP event timing is not currently available from the benchmark alone. The public
benchmark phases call `rns8_pack_i64`/`rns8_pack_u64`, `rns8_gemm_rns`, and
`rns8_export_i64`/`rns8_export_u64`. Those calls hide the direct-HIP backend's
copies, kernel launches, default-stream use, and internal synchronization. An
external event pair wrapped around the public call would not identify which
backend operations were timed, would miss or conflate synchronous host-side copy
costs depending on runtime behavior, and could not split GEMM launch groups from
export or scheduling work. The benchmark therefore reports:

```json
"timing_metadata": {
  "gpu_event_timing": false,
  "gpu_event_timing_reason": "requires_backend_or_public_timing_hooks"
},
"gpu_event_timings_us": null,
"gpu_event_timing_summary_us": null
```

Do not replace these `null` fields with host wall-clock timings or estimates.

The minimal focused implementation needed for real direct-HIP event capture is:

1. Add a backend timing capture object that can be optionally supplied to the
   direct-HIP helpers without changing exactness decisions or data movement.
2. Record HIP events inside the backend around each measured operation group:
   host-to-device upload, pack kernel, per-modulus or grouped RNS GEMM launches,
   export status initialization, export kernel, status readback, and output
   device-to-host copy where HIP event timing can validly observe it.
3. Preserve stable benchmark phase labels: `pack`, `rns_gemm`, `crt_export`,
   and optional subphase labels such as `pack_h2d`, `pack_kernel`,
   `rns_gemm_kernel_group`, `crt_export_kernel`, and `crt_export_d2h`.
4. Expose a benchmark-visible query or internal benchmark hook that drains the
   last-call timing capture. Unavailable phases must be reported as `null` with
   a reason, never synthesized from host timings.
5. Include timing source metadata for each capture: HIP runtime version, stream
   identity or policy, event timing unit, whether copies were asynchronous
   stream operations, warmup/repeat index, and selected backend.

Future benchmark work must add those hooks, reduction-specific timing,
scheduling overhead, raw captures, and comparison baselines before any speedup
claims are made.
