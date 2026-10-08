# CPU allocation and scratch checkpoint - 2026-10-08

## Lineage and scope

This bounded slice continues accepted commit
`8626a548b4663f6c870d592036559792765f0b18` in
`C:/Users/sneak/Development/RNS8/temp/worktrees/row-column-range-20261008`, branch
`codex/row-column-range-20261008`. The retained source is
`80b9ffe1f8439ad0254918152746a77feb8035de`; the prior complete qualification
campaign was not repeated. Owner main remains
`80791bd3d2900bd80b2ac6ba957d15329771043a`. Its untracked recovery ZIP retains
SHA256 `1dbce716c64356824cbdd6ae933e150d2c411363d16c8e3eda8966ec6965ae25`.
Origin identifies `https://github.com/doublemover/RNS8.git`. No publication, PR,
merge, installation, GPU execution or unrelated branch-tip change occurred.

## Implementation and measured allocation reduction

The opt-in CPU auto route now allocates its INT32 row accumulator once, then
reuses it through every selected modulus plane. The private serial primitive
performs no allocation and resets scratch at each row/K block. The shared row
arithmetic remains the same; ordinary CPU GEMM keeps its existing parallel
policy. Public API/ABI and residue-payload budgets are unchanged. The row buffer
remains `4 * N` bytes outside the residue-payload limit, and all allocation and
arithmetic still finish before resident inputs/output commit.

The dedicated fault executable owns replacement normal/array/aligned/nothrow
global C++ allocators, separately from shipping libraries and the normal Catch
test runner. Hooks are active only on the calling thread and inside explicit
API intervals. Fixture construction, restoration, snapshots and export/oracle
checks are outside those intervals. Six fixtures construct wide operands through
public packing/GEMM and cover signed/unsigned dual inputs, asymmetric initialized
prefixes, shared input identity and standalone lifting. Every observed successful
allocation point is failed individually. Each case requires INTERNAL_ERROR,
unchanged selected-prefix output, resident residues/storage, identities,
versions, native/currentness and scalar/axis proofs, unchanged plan/workspace
binding, and return to the same live C++ allocation count. Every failure is
followed by a successful retry and independent direct full-integer limb or
residue comparison. Nothrow/aligned/array pairing is self-checked.

MSVC Debug successful controls measured the following allocation counts and
requested byte totals; both pre-change and post-change profiles repeated
identically twice:

| Scenario | Selected prefix | Before calls / bytes | After calls / bytes | Removed calls / bytes |
| --- | ---: | ---: | ---: | ---: |
| signed dual, 2x13, K=1 | 19 | 299 / 19192 | 263 / 17968 | 36 / 1224 |
| unsigned dual, 2x13, K=1 | 20 | 283 / 18149 | 245 / 16857 | 38 / 1292 |
| signed asymmetric, 2x13, K=1 | 19 | 266 / 17213 | 230 / 15989 | 36 / 1224 |
| unsigned shared, 2x2, K=2 | 20 | 177 / 7971 | 139 / 7515 | 38 / 456 |
| signed standalone lift | 20 | 12 / 567 | 12 / 567 | 0 / 0 |
| unsigned standalone lift | 20 | 10 / 439 | 10 / 439 | 0 / 0 |

These Debug counts include a 16-byte STL iterator proxy per temporary vector:
removing 18/19 vector constructions removes 36/38 C++ allocation calls in this
configuration. Release/ASan counts below differ accordingly. This is measured
allocation reduction, not a latency, throughput or peak-memory speedup claim.
No benchmark campaign was launched.

## Focused correctness and build evidence

Installed MSVC 19.44.35228.0 built static/DLL/normal-test/fault-test Debug targets
in `build/row-column-cpu`, plus the isolated fault target in a separate
`build/row-column-release` RelWithDebInfo configuration. Installed clang-cl
22.1.6 built the existing `build/row-column-asan` static test targets with
AddressSanitizer. CPU OpenMP, HIP, accelerator probing, benchmarks, tools and
examples were disabled; VCPKG_MANIFEST_INSTALL=OFF reused installed dependencies.

Both Release and ASan sweeps passed all **641 injected allocation failures and
641 successful retries** per configuration:

| Scenario | Allocation failure points per build |
| --- | ---: |
| signed dual | 199 |
| unsigned dual | 181 |
| signed asymmetric | 167 |
| unsigned shared | 76 |
| signed standalone lift | 10 |
| unsigned standalone lift | 8 |

Both configurations also passed six independent serial-scratch cases with dirty
scratch, two rows, padded B/C strides, moduli 256/255/253 and K=131073 (two safe
65536-element blocks plus a tail). These assert zero internal C++ allocations,
stable scratch storage, exact INT64 dot/modular results and preserved padding.
MSVC Debug profile-only controls passed those same six scratch cases.

Normal focused MSVC Debug and ASan runs each passed **21 cases / 9301 assertions**
for `[exact-auto],[exact-axis],[exact-chain]`. Two directly affected ordinary
scalar-ring cases passed **13 assertions** in each configuration. These runs
qualify both the new serial primitive and its ordinary serial caller; OpenMP
compilation/execution remains unqualified. ASAN_OPTIONS=detect_leaks=0 was used;
no unexpected ASan diagnostic occurred. CTest supplies a 60-second timeout for
the isolated sweep. Local launches additionally captured combined stdout/stderr
with subprocess timeouts and CREATE_NO_WINDOW. Changed-line/new-file formatting
and staged `git diff --check` passed before committing this checkpoint.

Representative commands (capture paths are ignored under temp):

```powershell
python tools/windows_dev.py cmake --build build/row-column-cpu --target rns8_exact_cpu_allocation_faults rns8_tests rns8 --parallel 2
python tools/windows_dev.py cmake --build build/row-column-release --target rns8_exact_cpu_allocation_faults --parallel 2
ctest --test-dir build/row-column-release -R '^exact_cpu_allocation_faults$' --verbose
python tools/windows_dev.py cmake --build build/row-column-asan --target rns8_exact_cpu_allocation_faults rns8_tests --parallel 2
$env:ASAN_OPTIONS = 'detect_leaks=0'
ctest --test-dir build/row-column-asan -R '^exact_cpu_allocation_faults$' --verbose
build/row-column-cpu/rns8_tests.exe '[exact-auto],[exact-axis],[exact-chain]' --reporter compact --rng-seed 20261008
build/row-column-asan/rns8_tests.exe '[exact-auto],[exact-axis],[exact-chain]' --reporter compact --rng-seed 20261008
```

## Failures, root cause and noninteractive handling

The first unrestricted MSVC Debug fault sweep opened an intrusive runtime abort
dialog. This was a failed run, not an accepted injected-error result. Its stdout
was buffered and empty. The exact test process (PID 9884, confirmed executable
path in this worktree) was stopped; no owner action or debugger was needed.
Win32_Process CIM inspection was denied, but Get-Process returned the exact
path and permitted targeted termination. No broader process termination occurred.

One bounded diagnostic rerun used process-local noninteractive reporting and
returned failure code 86. Captured frames, symbolized from the local EXE/PDB,
proved fault point 35 entered std::allocator<std::_Container_proxy>::allocate
inside a noexcept Debug std::string move used by build_autotune_key at
api_plan_workspace_packing.inc:309. Termination occurs before guard_api can
catch std::bad_alloc. This is a Debug STL allocation qualification limit; it is
not evidence of an arithmetic/assertion defect or qualified API error handling.
The temporary diagnostic option was removed after diagnosis.

The regular fault executable now returns skip code 77 when
_ITERATOR_DEBUG_LEVEL>0; CTest explicitly reports the skip. Debug successful
profiles remain available with --profile-only. Full fault qualification runs
with _ITERATOR_DEBUG_LEVEL=0, as in the Release and ASan builds above. Nothing
disables or catches product assertions as passes. Both test entrypoints route
CRT errors/assertions to stderr and disable error-dialog/debugger/WER UI using
settings scoped to the test process. The fault termination handler emits case,
allocation point and stack addresses before a nonzero exit. No global Windows,
security or debugger setting changed. The Debug skip is not counted as a pass.

Three strict compile errors in new test code were corrected: an untyped modulus
initializer caused MSVC C4244 under /WX; including Windows headers introduced
min/max macro collisions (fixed with NOMINMAX); and Release CRT report macros
made the Debug report-loop variable unused under clang-cl /WX (fixed by limiting
that loop to the Debug CRT). Strict warnings remained enabled. Existing unrelated
unused-function/variable and reserved-enum warnings remain in library sources.

## Raw evidence

Raw logs remain ignored in this worktree's temp directory:

| Evidence | SHA256 |
| --- | --- |
| `fault-before-profile.log` | `c867e32f07c0b1f3707e67589204fe63f18f76305c57d16ef68e8119466d3c5e` |
| `fault-after-profile.log` | `a00829cb9c3a7c5a60413b244f6b587088198da619ce16418e46052eac02c021` |
| `fault-final-debug-profile.log` | `e72019f8e605bf879c79fe6c88cc94de9c5b2cda2751173d54e1cbd4421ba4cd` |
| `fault-debug-stl-diagnosis.log` | `322f42fe076d8a6ac6c0b441a637d114dda9791101c875b2744cb92d1d18ca0a` |
| `fault-debug-stl-symbols.log` | `d2ea9dbebd6a211a450ee222218c8baaba0f2e95570a47a6230d302d359fc8f1` |
| `fault-debug-gate.log` | `473e9d4aecf035194895209a18c1890a3376716dfc8cd0db20ccd3ad4da02050` |
| `fault-release-sweep.log` | `ea3a50f02da52bbd4cedbebc95e1f75e63d240059001c8c85b87e74f10cb5482` |
| `fault-asan-sweep.log` | `b7dc3a1a67558cfdeb2a01bfc37b426623bcf96d0564384da3fc5e56f2b66c9d` |
| `fault-focused-msvc.log` | `9b6ed740f0a38e17e53b674088cc31910769c382e3fca78c90ef0c126a773e54` |
| `fault-focused-asan.log` | `e2cfd6c842bff2439d031cd07808d115f6919090369dc9593be3aa328b82ccd3` |
| `fault-ring-msvc.log` | `c74c3d1789b8dadef17fe75ea72efd3ee15e7e3a8c6cf5e73a6f8b5078c50774` |
| `fault-ring-asan.log` | `17fd9e5474a2f9984cf8f9e6521fb42eec08b8c8db4aea60f0c174731ef13d65` |
| `fault-final-msvc-build.log` | `252849e8c9293bf14c74812351108cd9d40f928c024cce9d961c1a90d744b9ae` |
| `fault-release-configure.log` | `f6dd89882a3f65058e02803b5ab1721bb4850d3a297bdb1e57f17d8ab3c35735` |
| `fault-release-build.log` | `f0903b484d45e08c8a4da2d712d168c06f348c4e118671dc6322d37830bdc4d7` |
| `fault-asan-build-fixed.log` | `96597ba64168eee784200b27589178bd5c4372d484acc153848ad6a5fdf524c2` |
| `fault-noninteractive-build.log` | `f66433026c88a0cf3068eb83e7fc2e5663fd40e627fba1dc0e8c2d2224bbf992` |
| `fault-asan-build.log` | `d32821b51fc808142aa0741c0f29dd6afa20ad36b4a57ecbbbeb944760f05978` |

## Completion and next useful work

This slice closes a bounded CPU correctness/efficiency gap without enlarging the
public API. The qualification is C++ allocation failure in the six tested serial
fixtures, not real OS OOM, C malloc failure, cross-thread/DLL allocator behavior,
Debug STL OOM, OpenMP, UBSan/LeakSanitizer or complete Windows package/runtime
qualification. No GPU result, HIP installation status or GPU memory/workload
limit was established. GPU gates and publication remain untouched.

A useful next CPU chunk is a packaged example/install-consumer check for the
opt-in continuation, including combined budget failure and matching-prefix
export. Optional independent arithmetic-library oracles remain useful separate
work under installed-environment constraints. Stronger correlation proofs,
sparse/prepacked propagation and persistent cross-call scratch reuse remain
separate design/qualification work; this checkpoint makes no claim to them.
Integration still requires owner review of the complete lineage from main.
