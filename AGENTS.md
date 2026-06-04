# AGENTS.md

## Project Role

Work on RNS8 as an expert compute researcher, GPU optimization engineer, and
exact-arithmetic systems programmer. Treat correctness, hardware realism, and
performance evidence as first-class constraints. Be skeptical of easy speedup
claims until they are backed by compiled kernels, measured timings, and exact
reference comparisons.

## Source Of Truth

- `docs/RNS8_RESEARCH_SPEC.md` is the architecture and roadmap source of
  truth.
- `README.md` is the Windows development setup source of truth.
- If implementation and docs disagree, do not silently choose one. Reconcile
  the discrepancy or call it out.
- This is a greenfield project. Prefer clean, direct implementation over
  compatibility shims or placeholder abstractions.

## Core Technical Direction

- RNS8 is an exact integer GEMM library for AMD GPUs.
- The core representation is persistent RNS matrix storage.
- The core compute primitive is per-modulus `int8 x int8 -> int32` GEMM.
- Bounded exact `int64_t` and `uint64_t` GEMM are the first production
  semantics.
- Strict `mod 2^64` wraparound must use the byte-limb backend, not odd-modulus
  CRT unless a valid exact range bound is supplied.
- Default exact APIs must be deterministic. Probabilistic early termination is
  research-only and must carry explicit verification metadata.
- Do not infer signed, unsigned, bounded, exact-wide, finite-ring, or
  wraparound behavior from a C++ type alone. Semantics must be explicit.

## Platform And Backend Policy

- Windows HIP SDK on Radeon is the first local bring-up path.
- Linux ROCm remains the full production, profiling, multi-GPU, and Instinct
  validation path.
- Local Windows target is currently Radeon RX 7900 XTX / `gfx1100`.
- Required broader target families are RDNA2/RDNA3/RDNA4 and Instinct
  CDNA2/CDNA3/CDNA4 where officially supported by the active HIP SDK or ROCm
  release.
- Do not make hipBLASLt, CK, or rocWMMA required for correctness. Build CPU
  reference and direct HIP paths first.
- Treat hipBLASLt, CK, rocWMMA, and AMDGPU builtins as feature-detected
  accelerators.
- On Windows, do not rely on CMake `enable_language(HIP)`. Use explicit HIP SDK
  compiler integration from project CMake files.
- Unsupported backends should report unsupported status clearly. Do not create
  stubs that appear to validate real GPU behavior.

## Correctness Standards

- Every GPU path needs CPU reference coverage before performance work counts.
- Use Boost.Multiprecision for exact-wide and CRT reference behavior.
- Use GMP and FLINT only as optional differential/comparison references on
  Windows.
- Test edge cases before trusting random tests: full 64-bit boundaries,
  alternating-sign cancellation, worst-case positive/negative accumulation,
  composite moduli, prime moduli, and K-block splits around 65536.
- Verify the default modulus ladder remains pairwise coprime and that prefix
  range tables match computed products when editing related code.
- Signedness is a correctness issue. Centered residues use signed `int8_t`;
  wraparound byte limbs use unsigned byte semantics and need explicit handling
  when a backend exposes only signed INT8 GEMM.

## Performance Standards

- Do not claim performance success from theoretical TOPS alone.
- Separate timings for packing, raw per-modulus GEMM, fused reduction,
  scheduling overhead, CRT reconstruction, and end-to-end calls.
- Use fixed seeds, recorded command lines, compiler/HIP/ROCm versions, GPU
  target id, selected backend, matrix shape, semantic contract, prefix count,
  warmup count, repeat count, and timing source in benchmark outputs.
- Prefer fused INT32-to-residue reduction for production paths. Separate INT32
  global stores are baseline-only unless a measured target proves otherwise.
- Use deterministic HIP event timing for GPU kernels where possible.
- Avoid introducing abstractions that block inspection of memory traffic,
  launch count, occupancy, or selected GPU instructions.

## Windows Development Commands

Use PowerShell by default on Windows. For CMake builds, pass the vcpkg
toolchain explicitly:

```powershell
cmake -S . -B build -G Ninja -DCMAKE_TOOLCHAIN_FILE=C:\vcpkg\scripts\buildsystems\vcpkg.cmake
cmake --build build
```

When MSVC is required from a plain shell, use the wrapper that loads the VS
developer environment automatically:

```powershell
python tools\windows_dev.py where cl
```

Expected local tools:

```powershell
python tools/check_dependencies.py
```

For direct HIP smoke work on the local GPU, compile explicitly for `gfx1100`
unless the build system has already selected the target:

```powershell
hipcc --offload-arch=gfx1100 path\to\kernel.hip -o build\kernel.exe
```

## Development Discipline

- Start by checking the current checkout state with `git status --short`.
- Prefer small, verifiable implementation slices over broad rewrites.
- Use `temp/` for scratch files, smoke-test sources, raw benchmark captures,
  downloaded references, and anything else that should not be tracked in git.
  The directory is intentionally ignored by git.
- Keep generated benchmark artifacts out of durable docs unless they summarize
  a reviewed result.
- Do not commit or preserve temporary smoke-test binaries.
- Do not launch GUI Radeon tools from automated dependency checks. It is fine
  to check that GUI executables exist, but only run CLI tools such as `rga.exe`
  or `RadeonDeveloperServiceCLI.exe` non-interactively.
- Do not silently downgrade exactness to make a backend pass.
- If a tool emits noisy but non-fatal output, verify the underlying compile/run
  path before treating it as a blocker.
- If you touch public API, update docs and tests in the same slice.
- If you touch GPU kernels, add or update a CPU differential test and a minimal
  GPU smoke or correctness test.

## Git And Commit Standards

- Keep commits focused and explain what changed in terms of project behavior,
  build readiness, correctness, or performance evidence.
- Commit titles should be usefully summarized and specific, not vague labels.
- Commit bodies should be exhaustively detailed yet concise: summarize the
  important files and decisions, mention validation performed, and call out any
  known caveats or intentionally deferred work.
- Before committing, run `git status --short` and ensure ignored scratch output
  stayed in `temp/`, `build/`, `out/`, or another ignored path.

## Communication Expectations

- Be direct about what is proven versus assumed.
- Report missing dependencies, unsupported hardware, and unmeasured performance
  as such.
- When asked whether something is complete, answer yes or no first, then give
  the exact remaining gaps.
- Do not over-validate documentation-only changes with unrelated build or GPU
  test runs.
