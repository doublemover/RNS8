# Integer moment range campaign — 2026-10-11

## Scope and lineage

Local implementation and correctness checkpoint, not a release or performance
qualification. The campaign branches from main `29e5b64` on
`codex/correlation-range-20261011`, in the project-local worktree
`temp/worktrees/correlation-range-20261011`. No existing branch was rewritten,
and no divergent async, grouped-workflow, preflight or Garner work was imported.
There was no `.agents` directory in the checkout. AGENTS.md, README, the research
specification, current roadmap, prior lineage receipt and relevant source/tests
guided the work. No subagents, installs, remote machines or paid compute were used.
The final owner-checkout check still reports main at `29e5b64` and only its
original untracked recovery ZIP. Its SHA256 remains
`1dbce716c64356824cbdd6ae933e150d2c411363d16c8e3eda8966ec6965ae25`.

Implementation checkpoints:

| Commit | Change |
| --- | --- |
| `880d5ec` | Integer second-moment and covariance admission, propagation, tests and contracts |
| `73a05d8` | Dependency human-report crash and portable dashboard escaping test |
| `cfae07c` | Checked prefix-9 HIP export regression at K=65536 |

## Arithmetic and behavior

Native exact-wide packing now records exact squared sums on each logical row
and column, ignoring padding. Dense admission takes the minimum of existing
scalar/L1/midpoint bounds and integer Cauchy-Schwarz. Exact signed sums additionally
permit a centered covariance bound. All arithmetic, including square roots, is
Boost.Multiprecision integer arithmetic. The formulas and justification are in
[correctness.md](correctness.md#exact-wide-output).

Successful output proofs retain sums of squared cell bounds. These are upper
bounds, never claimed as exact signed sums. Subsequent resident GEMMs can use
their norm bounds; covariance remains disabled without exact signed sums.
Missing metadata retains existing conservative fallbacks. Repacking, CPU
lifting, opt-in CPU selection and rollback retain their existing ownership and
initialized-prefix rules. Sparse/prepack scalar commits still clear axis data.

A concrete signed/unsigned chain regression produces a wide column by an outer
product, then multiplies it by an uneven native row. The old L1 bound rejects
prefix 17; the norm bound admits it and CPU and HIP exports match independently
constructed full integers. The CPU transaction also rejects a 16-byte output
budget without mutation and succeeds with exactly 17 bytes.

No public symbols, descriptor layouts, semantic enums, modulus ladder or kernel
arithmetic changed. The extra private metadata is O(rows + columns); native
scanning and pairwise admission still have the existing asymptotic bounds, but
gain multiprecision work and storage. Roots that cannot tighten the established
bound are skipped. There is no latency, throughput or peak-memory claim. The
existing staging budget excludes proof metadata, including these new fields.
Autotune implementation revision advances from 3 to 4 to invalidate old approvals.

## Adjacent defects repaired

- Default dependency reporting raised `NameError` for `vcpkg_package_role`.
  The human renderer now imports it explicitly. Its self-test renders the
  entire missing-tool report and verifies package roles and readiness status.
- The dashboard self-test created `<capture>.json`, an invalid Windows filename.
  A portable ampersand/apostrophe filename now tests escaping, while script
  markup in the device name tests angle-bracket escaping through real schema
  validation. Invalid-kernel rejection remains covered.
- A legacy GPU regression asserted that prefix-9 bounded export had no status
  buffer. That contradicted the checked-export repair already on main. The test
  now requires the buffer and verifies RANGE_ERROR, padded caller-buffer
  preservation and stable resident allocations under a stricter bound.

## Local validation

Windows x64, MSVC 19.44.35228 / VS 17.14.40, CMake 4.2.3, Ninja 1.13.2,
Boost.Multiprecision 1.91.0, Catch2 3.15.0 and nlohmann-json 3.12.0#2.
CPU builds use RelWithDebInfo, serial execution, static and DLL targets.
HIP uses the installed HIP SDK 7.1.51803-d3a86bd04 and explicit `gfx1100` on
AMD Radeon RX 7900 XTX (reported 23.98 GB). Optional accelerator backends and
benchmark targets were disabled. ASan uses installed clang-cl 22.1.6, static
MSVC runtime and existing x64-windows-static dependencies.

All paths below are relative to this worktree. Raw logs are ignored under
`temp/`; these counts are correctness evidence, not performance measurements.
`temp/campaign-evidence.json` records log sizes/hashes and pins executable
source to `cfae07c`; the final receipt changes documentation only.

| Check | Result / retained log |
| --- | --- |
| MSVC CPU build and verifier | Passed; `cpu-build-final.log`, `cpu-verify.log` |
| Initial complete CPU CTest | 236 passed, one expected HIP skip, one dashboard fixture failure; `cpu-ctest.log` |
| Repaired tooling CTests | Both passed under Python 3.13; `tooling-ctest-final.log`; closes the only CPU suite failure |
| Moment CPU tests | 4 cases / 46,964 assertions; `moment-cpu.log` |
| ASan exact moment/axis/chain/lift/auto selection | 30 cases / 64,076 assertions; `asan-exact.log` |
| MSVC and ASan allocation rollback | Each passes 936 injected failures and retries across six scenarios; ASan also reports six K=131073 scratch cases; `cpu-ctest.log`, `asan-faults.log` |
| Static and DLL installed consumers | Passed inside CPU CTest, including the public continuation example |
| Direct HIP compile | Passed; `hip-build.log`, `hip-test-rebuild-final.log` |
| HIP moment chain | 1 case / 55 assertions; `hip-moment.log` |
| HIP qualification plus CPU axis/chain selection | 20 cases / 15,001 assertions; `hip-qualification.log` |
| HIP-name differential selection after test repair | 71 cases / 108,046 assertions, no skips; `hip-name-differentials-final.log` |
| Vector-ALU and persistent-small selection | 4 cases / 2,521 assertions; `hip-remaining-differentials.log` (its malformed HIP subfilter matched no cases) |
| Checked prefix-9 K boundary | 1 case / 134 assertions; `hip-checked-export-final.log` |
| Documented HIP smoke | CPU, pack, ring, bounded, adaptive, finite and wrap64 pass; `hip-smoke.log` |
| ISA selection | Five passes, including direct-HIP, generated reducer and wrap64 object checks; `hip-isa.log` |
| Default dependency report | Completes, host readiness OK outside sandbox; `dependency-human-unrestricted.log` |

The named HIP selections overlap; their counts must not be added to claim a
larger unique suite. CPU CTest is qualified by the initial run plus the targeted
repair rerun, not described as an originally all-green run. GPU checks cover
this local direct-HIP/vector-ALU build, not all optional backend specializations.

Reproduction commands (configure logs contain full resolved toolchain output):

```powershell
python tools/windows_dev.py cmake -S . -B build/campaign-cpu -G Ninja -DCMAKE_BUILD_TYPE=RelWithDebInfo -DCMAKE_TOOLCHAIN_FILE=C:/vcpkg/scripts/buildsystems/vcpkg.cmake -DVCPKG_MANIFEST_INSTALL=OFF -DVCPKG_INSTALLED_DIR=C:/vcpkg/installed -DRNS8_ENABLE_HIP=OFF -DRNS8_ENABLE_CPU_PARALLEL=OFF -DRNS8_BUILD_BENCHMARKS=OFF
python tools/windows_dev.py cmake --build build/campaign-cpu --parallel 4
python tools/windows_dev.py ctest --test-dir build/campaign-cpu --output-on-failure --parallel 4 --timeout 120
ctest --test-dir build/campaign-cpu -R 'performance_dashboard_self_test|dependency_checker_self_test' --output-on-failure
build/campaign-cpu/rns8-verify.exe

python tools/windows_dev.py cmake -S . -B build/campaign-hip -G Ninja -DCMAKE_BUILD_TYPE=RelWithDebInfo -DCMAKE_TOOLCHAIN_FILE=C:/vcpkg/scripts/buildsystems/vcpkg.cmake -DVCPKG_MANIFEST_INSTALL=OFF -DVCPKG_INSTALLED_DIR=C:/vcpkg/installed -DRNS8_ENABLE_HIP=ON '-DRNS8_HIP_ROOT=C:/Program Files/AMD/ROCm/7.1' -DRNS8_AMDGPU_TARGETS=gfx1100 -DRNS8_ENABLE_CPU_PARALLEL=OFF -DRNS8_BUILD_BENCHMARKS=OFF -DRNS8_BUILD_SHARED=OFF
python tools/windows_dev.py cmake --build build/campaign-hip --target rns8_tests rns8-verify rns8-inspect --parallel 4
build/campaign-hip/rns8_tests.exe '[exact-moment][hip]' --rng-seed 20261011
build/campaign-hip/rns8_tests.exe '[qualification],[exact-axis],[exact-chain]' --rng-seed 20261011 --abort
build/campaign-hip/rns8_tests.exe '*HIP*' --rng-seed 20261011 --abort
build/campaign-hip/rns8_tests.exe 'vector ALU*,persistent small GEMM*' --rng-seed 20261011 --abort
build/campaign-hip/rns8-verify.exe --hip-smoke
ctest --test-dir build/campaign-hip -L isa --output-on-failure

python tools/windows_dev.py cmake -S . -B build/campaign-asan -G Ninja -DCMAKE_BUILD_TYPE=RelWithDebInfo '-DCMAKE_CXX_COMPILER=C:/Program Files/LLVM/bin/clang-cl.exe' -DCMAKE_TOOLCHAIN_FILE=C:/vcpkg/scripts/buildsystems/vcpkg.cmake -DVCPKG_MANIFEST_INSTALL=OFF -DVCPKG_INSTALLED_DIR=C:/Users/sneak/Development/RNS8/build/windows-clang-asan-debug/vcpkg_installed -DVCPKG_TARGET_TRIPLET=x64-windows-static -DCMAKE_MSVC_RUNTIME_LIBRARY=MultiThreaded '-DCMAKE_CXX_FLAGS_RELWITHDEBINFO=/fsanitize=address /Zi /O1 /MT' -DRNS8_ENABLE_WINDOWS_CLANG_ASAN=ON -DRNS8_BUILD_SHARED=OFF -DRNS8_ENABLE_HIP=OFF -DRNS8_ENABLE_CPU_PARALLEL=OFF -DRNS8_BUILD_BENCHMARKS=OFF -DRNS8_BUILD_EXAMPLES=OFF -DRNS8_BUILD_TOOLS=OFF
python tools/windows_dev.py cmake --build build/campaign-asan --target rns8_tests rns8_exact_cpu_allocation_faults --parallel 4
build/campaign-asan/rns8_tests.exe '[exact-moment]~[hip],[exact-axis],[exact-chain],[exact-lift],[exact-auto]' --rng-seed 20261011
build/campaign-asan/rns8_exact_cpu_allocation_faults.exe
```

## Failures and limits retained

The restricted compiler probe failed with MSVC C1902/PDB-manager access; the
same installed toolchain succeeded outside the sandbox. Restricted Python 3.13
CTest temporary-directory access also failed, and restricted vcpkg discovery
misreported installed required packages as missing; unrestricted reruns passed.
No automatic approval review rejected an operation.

The first moment-chain fixture expected element positions to survive a diagonal
multiply through axis summaries. The conservative implementation correctly did
not establish that proof. The fixture was changed to an outer product with
independently provable cell magnitudes. Exhaustive small-axis bounds passed even
in that initial run. The first added HIP rejection assertion failed compilation
because of an uncaptured local constant in a lambda; direct vector comparison
fixed it. A stale-binary rerun after that build failure is retained separately.

The VS command wrapper misparsed `|` in a CTest regex; direct CTest invocation
preserved it. A mixed Catch2 name/tag filter matched no HIP names (while running
its vector/persistent cases); the corrected plain name filter completed all 71
cases. An attempted continuation list did not match earlier registration order,
so it was discarded rather than treated as coverage. These setup/test failures
are not represented as arithmetic passes.

Remaining gates: native Linux ROCm and packages; MI300X/gfx942 and other required
RDNA/CDNA families; hipBLASLt, CK, rocWMMA and builtins; full hardware qualification
runner/capture matrix, graph lanes, profiler counters, timing variance and
reviewed performance promotion. No full HIP CTest/package lane or benchmark
campaign was run. Windows ASan is not UBSan/LSan, Debug-STL OOM, OpenMP or DLL
allocation-failure qualification. General cross-axis correlations and device
prefix lifting remain open. Nothing here claims production readiness.

## Proposed publication, pending approval

Propose pushing this complete branch to the configured project origin
`https://github.com/doublemover/RNS8.git`, as
`codex/correlation-range-20261011`, then opening a draft PR targeting `main`.
Suggested title: **Tighten exact-wide range admission with integer moments and
qualify local correctness**. Recheck destination refs and review the complete
base-to-head diff before publication; remote freshness was not queried in this
local-only campaign. No push, PR, merge or deployment has occurred.
