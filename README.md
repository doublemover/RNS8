# RNS8

RNS8 is an exact integer GEMM library for AMD GPUs. It stores matrices in a
residue number system, evaluates small-modulus `int8 x int8 -> int32` GEMMs,
and reconstructs exact integer outputs only when the selected semantic contract
allows it.

The project is greenfield and intentionally explicit: bounded exact `int64_t`
and `uint64_t`, exact-wide RNS output, strict `mod 2^64` wraparound, and finite
`uint8_t` rings/fields are separate contracts. A C++ type alone never selects
signedness, wraparound, or exactness behavior.

## What Works Now

- Public C ABI and C++ RAII wrapper.
- CPU reference backend with Boost.Multiprecision CRT/Garner reconstruction.
- Windows HIP SDK direct backend on Radeon RX 7900 XTX / `gfx1100`.
- Persistent RNS matrices with device-resident direct-HIP pack, GEMM, and
  export paths.
- Bounded signed/unsigned 64-bit GEMM with fixed and per-tile modulus counts.
- Bounded signed/unsigned 64-bit native vector-ALU backend for explicit
  bounded contracts.
- Exact-wide signed/unsigned RNS output with fixed-width limb export.
- Strict `mod 2^64` CPU and direct-HIP byte-limb paths.
- Explicit finite-ring and finite-field `uint8_t` GEMM for moduli up to 256.
- Opt-in Windows `gfx1100` hipBLASLt, CK, and rocWMMA correctness backends.
- Benchmark schema, review tooling, result comparison, and autotune-cache
  validation for reviewed release evidence.

RNS8 does not claim general Linux ROCm or Instinct readiness from Windows
evidence. Linux ROCm, Instinct CDNA, multi-GPU, and production profiling remain
validation targets that require real supported Linux hardware.

## Quick Start

Windows HIP bring-up uses the checked-in presets and the Visual Studio wrapper:

```powershell
python tools\check_dependencies.py
python tools\windows_dev.py cmake --preset windows-msvc-hip-debug
python tools\windows_dev.py cmake --build --preset windows-debug
python tools\windows_dev.py ctest --preset windows-debug --output-on-failure
build\windows-msvc-hip-debug\rns8-verify.exe --hip-smoke
```

For a CPU-only build:

```powershell
cmake --preset cpu-debug
cmake --build --preset cpu-debug
ctest --preset cpu-debug --output-on-failure
```

The Windows setup details are in
[docs/platform-windows.md](docs/platform-windows.md). The architecture and
roadmap source of truth is
[docs/RNS8_RESEARCH_SPEC.md](docs/RNS8_RESEARCH_SPEC.md).

## Correctness And Performance

Correctness is established through CPU reference comparisons before GPU paths
count. GPU paths are accepted only for explicit semantics, and unsupported
backends report unsupported status rather than silently downgrading exactness.

Performance claims require reviewed same-contract captures with fixed seeds,
recorded toolchain and GPU metadata, CPU/reference and GPU baselines, and
release-mode repeat counts. Raw benchmark captures are evidence, not promotion.
See [docs/performance-model.md](docs/performance-model.md).

### Current Windows `gfx1100` Speedup Snapshot

The compact current wins table lives in
[docs/performance-wins.md](docs/performance-wins.md). Latest local evidence is
Windows RX 7900 XTX / `gfx1100` only; it does not imply Linux ROCm or Instinct
readiness.

| Case | Measured speedup including setup | Status |
|---|---:|---|
| Bounded i64 1024 one-shot CK vs direct HIP | 1.04x vs direct HIP, 2.58x vs vector ALU | Release-reviewed local snapshot; cache not installed |
| hipBLASLt repeated A+B, 512 | 5.84x over 9 repeats; 7.68x steady-state | Event-valid experimental reuse path; 8323 us setup |
| hipBLASLt repeated A+B, 1024 | 4.32x over 9 repeats; 4.81x steady-state | Event-valid experimental reuse path; 5992 us setup |
| hipBLASLt repeated B, 512 | 4.72x over 9 repeats; 5.05x steady-state | Event-valid experimental reuse path; 2811 us setup |
| Vector ALU repeated B, 1024 | 1.21x over 9 repeats; 1.23x steady-state | Event-valid experimental reuse path; 3187 us setup |

Reuse speedups compare against the same backend without reuse. The headline
reuse numbers above include one-time setup over the measured nine-repeat
validation capture; steady-state is the per-repeat limit after setup has already
been paid. They are not default AUTO promotion claims.

## Documentation

- [docs/README.md](docs/README.md): documentation map.
- [docs/RNS8_RESEARCH_SPEC.md](docs/RNS8_RESEARCH_SPEC.md): architecture,
  roadmap, and semantic contracts.
- [docs/roadmap-status.md](docs/roadmap-status.md): current implementation
  status and remaining gaps.
- [docs/performance-wins.md](docs/performance-wins.md): current Windows
  `gfx1100` performance wins and promotion boundaries.
- [docs/performance-gain-work-queue.md](docs/performance-gain-work-queue.md):
  ordered performance implementation queue.
- [docs/correctness.md](docs/correctness.md): correctness coverage and guardrails.
- [docs/backend-notes.md](docs/backend-notes.md): backend policy and status.
- [docs/platform-windows.md](docs/platform-windows.md): local Windows HIP setup.
- [docs/platform-linux.md](docs/platform-linux.md): Linux ROCm readiness scope.
- [docs/platform-readiness.md](docs/platform-readiness.md): readiness-report
  policy.
- [third_party/README.md](third_party/README.md): third-party and submodule
  policy.

Temporary captures, probes, installers, and scratch binaries belong under
ignored `temp/`, `build/`, or `out/` paths.
