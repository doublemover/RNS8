# Performance Model Notes

The research spec defines the long-term performance model. The current scaffold
does not make optimized GPU performance claims.

Current benchmark shell:

```powershell
build\windows-msvc-hip-debug\rns8-bench.exe --backend cpu --semantics bounded-i64 --m 64 --n 64 --k 64 --warmups 1 --repeats 5 --seed 1
build\windows-msvc-hip-debug\rns8-bench.exe --backend hip-direct --semantics bounded-u64 --m 16 --n 16 --k 16 --warmups 1 --repeats 3 --seed 1
```

The benchmark reports:

- requested and selected backend,
- semantic contract,
- matrix shape,
- fixed seed,
- warmup and repeat counts,
- prefix count,
- command line,
- compiler version,
- configured AMDGPU target list,
- HIP device identity and runtime metadata when using the direct HIP backend,
- timing source and timing caveat,
- one-time plan and matrix allocation time,
- average packing time,
- average persistent RNS GEMM time,
- average per-modulus GEMM estimate,
- average CRT export time,
- average end-to-end time for the measured phases.

Current direct-HIP benchmark timings use host `std::chrono::steady_clock`.
They include the current correctness backend's synchronization, allocation,
copies, kernel launches, and host CRT export. Future benchmark work must add
GPU HIP event timing, reduction-specific timing, scheduling overhead, raw
captures, and comparison baselines before any speedup claims are made.
