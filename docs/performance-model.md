# Performance Model Notes

The research spec defines the long-term performance model. The current scaffold
does not make optimized GPU performance claims.

Current benchmark shell:

```powershell
build\windows-msvc-hip-debug\rns8-bench.exe --m 64 --n 64 --k 64 --repeats 5 --seed 1
```

The benchmark reports:

- backend,
- semantic contract,
- matrix shape,
- fixed seed,
- repeat count,
- prefix count,
- timing source,
- one-time plan and matrix allocation time,
- average packing time,
- average CPU per-modulus ring-GEMM time,
- average CRT export time,
- average end-to-end time for the measured phases.

Future benchmark work must add GPU HIP event timing, reduction-specific timing,
scheduling overhead, raw captures, and comparison baselines before any speedup
claims are made.
