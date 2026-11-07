# Runtime cache continuation, 2026-10-11

This is a correctness checkpoint, not a performance promotion or production
qualification. It continues the [integer moment campaign](local-moment-range-20261011.md)
and the current roadmap's deterministic runtime/source gates.

## Publication and source boundary

The approved moment checkpoint was pushed and published as draft
[PR #14](https://github.com/doublemover/RNS8/pull/14) in `doublemover/RNS8`.
Verified remote head: `78a85d28c8f16369ff50ce4e2618ff07540c276a`;
base `main`: `29e5b64782437226b2a4a6fd802a93a8564caab5`.
The prior cancelled continuation had not created a remote branch or PR; those
were checked before publishing to avoid duplication. No merge or deployment.
This supersedes the earlier receipt's local-only publication status.

New work is local on `codex/runtime-cache-transactions-20261011`, based on that
approved head, using the existing project-contained worktree
`temp/worktrees/correlation-range-20261011`. Implementation checkpoints:

- `05a81b4`: strict autotune admission and transactional file replacement.
- `19f1d1f`: resident-cache fail-closed state and recovery checks.

The primary checkout remains at `29e5b64`. Its original untracked archive,
`RNS8-Range-Continuation-80b9ffe-2026-10-06.zip`, retains SHA256
`1dbce716c64356824cbdd6ae933e150d2c411363d16c8e3eda8966ec6965ae25`.
Other continuation branches were not imported or modified.

## Implementation inventory

`src/core/autotune_cache.cpp` now rejects fractional/out-of-range integer fields,
negative/non-finite timings, duplicate JSON members, duplicate entry keys,
malformed record containers, and ambiguous duplicate plan-key fields. Failed
reads clear all partially accumulated entries. Reviewed selection additionally
requires a positive end-to-end median; synthetic or unreviewed data does not
constitute timing evidence. The schema remains version 1 and the arithmetic
implementation key remains revision 4: valid existing identities remain valid.

Writes serialize before touching files, use an exclusively reserved sibling
staging directory, check stream write/flush/close, and atomically replace the
file with `MoveFileExW` on Windows or rename on POSIX. Failures preserve prior
bytes and clean staging. Unsupported or unreadable existing schemas are not
silently rewritten. This does not promise concurrent writer merge isolation,
power-loss durability, or behavior on every network filesystem.

`src/core/api_gemm.cpp` clears result-cache hit eligibility before snapshot
allocation/copy and release, and publishes a hit only after successful binding.
Restore clears every output representation's currentness for all supported
contracts before copying. Handle construction uses unique ownership through
fallible string initialization. Public ABI and GPU kernels are unchanged.

Tests cover malformed identity/timing records after an earlier valid record,
serialization failure, unsupported schemas, locked Windows destinations,
staging cleanup, successful replacement, real AUTO fallback computation,
resident preparation rejection, export blocking, and recovery. The GPU failure
fixtures use the backend's null-pointer precondition; they do not cause an
illegal device access or simulate completion of a failing hardware transfer.

The inspected exact-wide reconstruction/export paths did not yield a new
confirmed arithmetic defect in this slice. Existing signed/unsigned width,
overflow, padding and K-boundary tests were retained and run by the CPU gate.

## Validation and reproducibility

Reused the prior receipt's explicit CMake configurations and installed local
toolchain: MSVC 19.44.35228, VS 17.14.40, clang-cl 22.1.6 ASan, CMake 4.2.3,
Ninja 1.13.2, Windows HIP 7.1.51803-d3a86bd04, RX 7900 XTX / gfx1100.
CPU and ASan builds disable OpenMP; HIP optional accelerators remain disabled.
No benchmarks, paid compute, remote device access or profiler campaign ran.

From the worktree, final commands included:

```powershell
python tools/windows_dev.py cmake --build build/campaign-cpu -j 4
ctest --test-dir build/campaign-cpu --output-on-failure -j 4
python tools/windows_dev.py ctest --test-dir build/campaign-cpu -R '^install_downstream_cmake_smoke$' --output-on-failure
python tools/windows_dev.py cmake --build build/campaign-asan --target rns8_tests -j 4
build/campaign-asan/rns8_tests.exe '*autotune*,[result-cache],[exact-auto]' --rng-seed 20261011
python tools/windows_dev.py cmake --build build/campaign-hip --target rns8_tests -j 4
build/campaign-hip/rns8_tests.exe 'AUTO plan consumes reviewed vector ALU*,*autotune*' --rng-seed 20261011
build/campaign-hip/rns8_tests.exe '[result-cache],AUTO plan consumes reviewed vector ALU*,[exact-moment][hip]' --rng-seed 20261011
```

Results (overlapping suites are not additive):

- Autotune checkpoint CPU and ASan: 18 cases / 2,411 assertions each.
- Autotune checkpoint actual HIP: 8 cases / 6,533 assertions, including five
  malformed reviewed-cache fallbacks to direct HIP and exact output checks.
- Final ASan: 19 cases / 2,417 assertions, no sanitizer finding.
- Final actual HIP: 8 cases / 6,394 assertions, no skips. Bounded and finite
  resident-cache recovery still matches CPU reference output; exact-wide
  moment chain and native-vector AUTO selection/fallback also pass.
- Final full CPU CTest: 242 registered, 240 passed, one expected HIP-only skip,
  one package-link failure from missing Windows SDK `LIB` environment. The
  package-only retry through `windows_dev.py` passed, including installed static
  and DLL consumers. Combined gate: 241 passes and one expected HIP-only skip.
  No source change was needed for the retry.

Raw runtime logs and their SHA256 manifest stay ignored under `temp/`.
Compiler warnings remain for pre-existing unused serial-build variables/helpers
and unhandled research enum switch values. A restricted primary-checkout status
probe could not read submodule metadata; the unrestricted read succeeded and
confirmed preservation. No automatic approval review rejected any operation.

## Remaining gates and publication proposal

Linux/POSIX rename execution, native Linux ROCm/packages, other RDNA/CDNA GPUs,
Instinct, optional accelerators, full HIP CTest/package qualification, actual
in-flight device-copy/allocator failures, Debug-STL OOM, OpenMP, DLL allocation
fault sweeps, UBSan/LSan and performance promotion remain unqualified. Staging
write/flush/close error branches were implemented but disk-full/close failures
were not injected; replacement and serialization failures were exercised.

Propose pushing `codex/runtime-cache-transactions-20261011` to the same origin
and opening a separate stacked draft PR with base
`codex/correlation-range-20261011`, so review isolates this continuation from
PR #14. Suggested title: **Harden autotune persistence and resident-cache
failure recovery**. Approval is pending for this new branch; its commits have
not been pushed or added to PR #14. No merge or deployment is proposed.
