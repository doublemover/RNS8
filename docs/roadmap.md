# Current roadmap and release gates

This is the current execution checklist. The [research specification](RNS8_RESEARCH_SPEC.md)
retains long-term design requirements; [historical queues](archive/README.md)
retain earlier decisions without asserting they are satisfied by current code.

## 1. Source and CPU gate

- Repair/remove the unsafe weighted CRT, unchecked bounded export, coalesced
  pack-tail, persistent K-block, and scalar "WMMA skinny" paths
- Keep exact semantic boundaries, error reporting, and caller-buffer preservation
- Pin malformed public enums to a safely validated 32-bit ABI representation
- Run CPU tests, host-only scalar HIP arithmetic regressions, schema/report
  checks, examples, install/downstream smoke, and available sanitizers
- Record exact validation failures, unavailable tools, and source revision

The [audit](audit-2026-10-06.md) records completion evidence for this preparation
slice. No source-only gate establishes hardware qualification.

## 2. RX 7900 XTX / gfx1100 gate

- Compile direct HIP and each optional backend with the actual Windows HIP SDK
- Run the new qualification tests and full differential suite; inspect all skips
- Cover prefix 8 and 9/20 errors, full-width signed/unsigned values, odd packing
  tails, padded strides, wave tails, small output shapes and long K
- Recheck finite moduli 2–256 where supported, particularly skinny N=1/2/4/5/8
- Check actual dot4 signedness, matrix instructions, export code and resource use
- Recheck graph setup/replay/cleanup and persistent storage lifetime
- Re-establish timing with the corrected baseline before any promotion

## 3. MI300X / gfx942 gate

- Compile with a native Linux ROCm toolchain and explicit `gfx942` target
- Run direct-HIP CPU differentials first, then MFMA, CK, rocWMMA, and hipBLASLt
- Cover wave64 behavior, dense MFMA tile variants, explicit sparse-A SMFMAC,
  padded/tail shapes, finite composite moduli, and accumulation cap rejection
- Capture exact source/toolchain/device provenance, profiler/counter data, ISA,
  memory/resource usage and independent-run timing variance
- Keep one physical GPU selected; independent per-device shard scripts are not
  distributed GEMM or collective correctness tests

The [qualification guide](gpu-qualification.md) supplies bounded commands for
both targets and makes unsupported optional dependencies visible.

## 4. Release decision

A release requires all applicable preceding gates plus:

- Reviewed raw evidence retained for every shipped target/backend/contract
- No unexplained failures, skipped required tests, stale kernel identities, or
  unverified claims in active documentation
- Package installation and downstream C/C++ ABI checks on Windows and Linux
- Reproducible same-contract performance review and explicit cache installation
- An explicit support matrix and limitations for unsupported shapes/targets

This source audit does not satisfy that release decision.

## Research that remains open

- A measured safe short-prefix CRT optimization, with overflow-proof arithmetic
- Matrix-engine small-N improvements that genuinely preserve lane/stride/finite
  semantics; a scalar shuffle kernel is not WMMA evidence
- Persistent graph/API lifetime design, asynchronous execution, and multi-GPU GEMM
- Automatic admission-driven exact-wide prefix selection and device lifting;
  explicit transactional CPU lifting within allocated capacity is implemented
- Stronger correlation-aware range proofs; scalar and per-axis conservative
  chain propagation are implemented
- Broader target qualification (RDNA2/4, CDNA2/4) under supported toolchains
- INT4/IU4, Ozaki/FP8, Strassen/Winograd, and probabilistic verification, each
  behind the specification's explicit correctness, measurement, and retirement
  criteria; none is implemented simply because a reserved enum/API exists

## Stop conditions for hardware work

Stop at the first arithmetic, out-of-bounds, device fault, unexplained skip, or
wrong-target error. Preserve logs and the exact build. Diagnose correctness
before timing. A missing optional dependency is a qualification blocker for that
backend, not permission to label a fallback as the requested accelerator.
