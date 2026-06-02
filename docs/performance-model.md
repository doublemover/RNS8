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
- git commit captured at CMake configure time,
- compiler version,
- configured AMDGPU target list,
- HIP device identity and runtime metadata when using the direct HIP backend,
- clock/power settings when available; currently `null`,
- comparison baseline and derived TOPS-equivalent when reviewed baselines exist;
  currently `null`,
- timing source, timing caveat, and structured timing metadata,
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
kernel launches, fused residue reduction, and GPU bounded export. Future
benchmark work must add GPU HIP event timing, reduction-specific timing,
scheduling overhead, raw captures, and comparison baselines before any speedup
claims are made.
