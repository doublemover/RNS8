# Full continuation lineage review - 2026-10-08

## Recommendation and scope

Recommend integrating the complete continuation as a reviewed CPU/source
correctness checkpoint after owner approval of the public contracts and full
lineage. This is a bounded source/contract review by the continuing agent, with
focused local package/ABI checks; it is not an independent formal proof or a
production/hardware release qualification. No blocking CPU arithmetic or
transaction defect was identified within this review. Two public comment
discrepancies were corrected below. Hardware regression risk remains explicit.

The reviewed implementation head is `be958805c7d33436e24ada890ceee80944fad7a6`
(tree `fc99a49d85208b8d3e0f48440b4f8c3335272664`). Original main, current
local main and origin/main are `80791bd3d2900bd80b2ac6ba957d15329771043a`.
A read-only `git ls-remote origin refs/heads/main` also confirmed that exact
GitHub main tip during this review. It changed no refs. Main is the merge base;
the ten commits are a linear descendant chain with no intervening merge.

The verified project origin is `https://github.com/doublemover/RNS8.git`.
All work stays on `codex/row-column-range-20261008` in the project-local
`temp/worktrees/row-column-range-20261008` worktree. The owning main checkout,
unrelated branch tips and recovery ZIP are preserved. ZIP SHA256 remains
`1dbce716c64356824cbdd6ae933e150d2c411363d16c8e3eda8966ec6965ae25`.
No publication, PR, merge, GPU work, agent, installation or benchmark occurred.

## Complete reviewed lineage

| Full commit | Coherent change |
| --- | --- |
| `252e45b1ee0b7c94b91e5883327fbe01b6ec4bb5` | Repair unsafe exact-arithmetic paths and add qualification regressions |
| `3b01333a9a62eb8b71376974d84a5901be503485` | Consolidate documentation around audited contracts and hardware gates |
| `3855bb5b3e4dcb8d1111f0d4e478ebd19e2bb8b1` | Keep generated qualification commands portable to target hosts |
| `8dc5445ff13a93243d6d832715a3148c328e465d` | Remove unsound host-mirror zero inference from tiled HIP GEMM |
| `80b9ffe1f8439ad0254918152746a77feb8035de` | Prove exact-wide resident chain range before execution |
| `11618181e6074bf3e3dcbf9b7c80b85c874884e9` | Tighten exact-wide chain admission with row and column proofs |
| `6d9dcbd910fc94acda8d817ca28cd6f114b7a0c9` | Add transactional budgeted CPU exact-wide prefix lifting |
| `8626a548b4663f6c870d592036559792765f0b18` | Add opt-in CPU prefix selection with operand and output transactions |
| `d5049dffd5ebcda68ed19a78a18f4e7deb0d7e92` | Reuse CPU exact-wide row scratch and qualify allocation rollback |
| `be958805c7d33436e24ada890ceee80944fad7a6` | Ship CPU continuation example and verify clean package consumers |

The net reviewed range changes 114 files: 6,042 insertions / 2,976 deletions,
including documentation moves. The later receipt/comment commit descends from
the reviewed implementation head and adds no executable-code change.

The inherited audit removes unsafe weighted-CRT and unchecked-export dispatch,
repairs persistent K-block reduction and four-cell packing tails, restores real
RDNA3 finite/RNS WMMA dispatch, widens common device offsets, and removes dead
fused/native/graph/pack candidates. Live byte-limb wrap64, benchmark-owned graph
execution, resident/grouped APIs and reserved research entrypoints remain.
Kernel registry IDs for removed candidates are removed consistently. The
documentation/archive/tooling changes separate present contracts from historical
performance claims and make qualification fail closed. The immutable tiled HIP
schedule repair eliminates host-mirror zero inference.

The exact-wide chain repair adds initialized-prefix/magnitude evidence to opaque
handles, rejects CRT aliasing and unwritten-prefix use, and propagates proofs
through dense/grouped/incremental/sparse/prepacked routes. The continuation
adds multiprecision axis bounds, explicit CPU lifting and deterministic opt-in
prefix selection, transactional operand/output staging, reused serial row scratch,
allocation-fault tests and the installed public consumer. All ten commits belong
to this dependency chain; reviewing or cherry-picking only the latest example
commit would omit both correctness repairs and its required public implementation.

## API, ABI and serialization implications

- Two additive C symbols: `rns8_lift_exact_wide_cpu` and
  `rns8_gemm_exact_wide_cpu_auto`; matching C++ Matrix/free-function wrappers.
  Existing calls remain explicit and do not auto-lift, transfer or fall back.
- Public numeric enum values, descriptor fields, existing function signatures
  and `RNS8_ABI_VERSION=1` remain. C++ enum backing becomes `uint32_t` so malformed
  C/FFI bit patterns can be inspected safely. That changes C++ underlying-type
  traits/signedness; clients depending on an `int` underlying type need review.
  C requires the normal 32-bit enum ABI; short-enum clients are unsupported.
- MSVC x64 C17 and C++17 probes of original and continuation headers produced
  four identical outputs: all 13 public enums' size/alignment, all 20 structs'
  size/alignment, all 443 field offsets, and ABI version. The output SHA256 is
  `a05df03f1d6d1e5e407725260df554f2a37339e09baea5705a9816da5e3ef745`.
  This does not prove every calling convention/compiler/CRT or old-binary usage.
- The Release DLL exports all 69 currently declared public functions, including
  all 67 original public functions and exactly the two additions. Reserved
  research functions still return UNSUPPORTED_BACKEND. New callers using these
  two symbols require the newer library. Current package C++ consumers linked
  and executed successfully with static and DLL linkage.
- Extra proof fields are private to opaque matrix/sparse/prepack objects. There
  is no newly serialized matrix or handle format; copying private object bytes
  across DLL/build versions is outside the API. C++ wrappers add no data members
  or virtual dispatch. Callers continue to serialize access to resident matrices.
- Existing autotune JSON schema remains version 1, but exact cache keys include
  `implementation_revision=3` (audit revision 2, then zero-inference repair 3).
  Old reviewed timing keys cannot establish repaired-path performance. Removed
  kernel identities can invalidate old captures under current registry checks;
  archived records retain historical meaning. No cache was promoted. Qualification
  manifest schema 1 is new tooling, while golden reports gain timeout/progress
  evidence and the capture explorer replaces shape-only winner claims.
- Behavior deliberately tightens: exact-wide chains lacking sufficient proofs,
  unwritten planes and output/input aliases fail before execution. Cases formerly
  returning an aliased representative now return RANGE_ERROR. This is observable
  contract repair, despite preserving binary layouts. Exactness is not relaxed.

## Correctness and resource invariants reviewed

Range arithmetic and strict CRT products use Boost.Multiprecision: signed
`2*bound < product`, unsigned `bound < product`, with a twenty-plane ceiling.
Axis summaries exclude padding and retain exact native sums/extrema/L1
deviations. The centered identity is used only when both sums are known;
conservative GEMM output summaries never masquerade as exact sums. Missing
summaries fall back to the scalar bound; scalar-only sparse/prepack outputs
discard stale axis evidence. Tighter inequalities may admit cancellation-safe
cases, but correlation still causes conservative rejection.

Allocated capacity never establishes initialized-prefix evidence. Input lifting
requires a unique source proof before reconstructing; it cannot recover prior
CRT aliasing. Every staged source byte is checked for centered validity and each
reconstructed integer against its retained magnitude. Version/currentness and
immutable plan/workspace binding govern reuse; caller source-version and dirty
region contracts remain trusted. Ordinary execution invalidates proofs before
potential partial writes and commits a new proof only on success.

The opt-in route validates structural inputs, bounds, ceilings and budget before its
payload allocation; stages distinct input suffixes and the entire output; then
copies into existing allocations only after all fallible work. Identical A==B
counts once, C aliases are rejected, and size products/sums are checked. Inputs'
integer/proof/version/native state and identities stay intact. Failures preserve
A/B/C and selected-prefix output; plan/workspace are immutable. Wider results
need a matching fixed-prefix export plan; the original insufficient export
continues to reject and preserve its caller buffer.

The byte limit covers residue payload, not total process memory. It excludes
allocator metadata, O(rows+cols) multiprecision summaries, bounded CRT scratch
and the `4*N` byte row accumulator. Native proof scans add CPU/allocation cost;
axis product summaries add O(M*N) bound work. The serial transactional primitive
clears reused scratch for each row/K block and reduces between safe 65,536-term
blocks. It stays within the API exception guard; ordinary CPU OpenMP policy is
unchanged and its build/execution remains unqualified here. Only the measured
MSVC Debug profiles support the recorded allocation-count reduction; no latency,
throughput, peak-memory or performance promotion is inferred.

The strong all-failure rollback claim applies to the new CPU lift/auto routes.
It does not promise universal rollback after asynchronous device failure,
grouped resource cleanup failure, or arbitrary allocation failure in ordinary
packing/sparse/cache APIs. Sparse repacking stages malformed groups, but the
allocation-fault sweep does not qualify every sparse/cache failure path.

## Review corrections and current focused checks

The public ordinary-GEMM comment still required the global K*max(A)*max(B) bound
to fit after row/column proofs had been implemented. It now describes a
conservative bound tightened by those proofs. The standalone lift payload
formula now names missing planes and explicitly gives zero for an initialized
target, avoiding a negative suffix count for a no-op. Comment-stripped header
tokens are unchanged; no API or runtime implementation changed in this review.

The remaining Release DLL package qualification was completed with already
installed MSVC 19.44.35228.0 / VS 17.14.40, x64 RelWithDebInfo (`/O2 /DNDEBUG /MD`),
Ninja and existing dependencies. HIP/OpenMP/optional GPU backends/benchmarks/tools
were OFF, examples/tests ON and VCPKG_MANIFEST_INSTALL=OFF. Configure reused
`build/row-column-release` with RNS8_BUILD_SHARED=ON; DLL compilation used two
jobs. No tools were installed or accelerator probes invoked.

```powershell
python tools/windows_dev.py cmake -S . -B build/row-column-release -DRNS8_BUILD_SHARED=ON -DVCPKG_MANIFEST_INSTALL=OFF
python tools/windows_dev.py cmake --build build/row-column-release --target rns8 --parallel 2
python tools/windows_dev.py ctest --test-dir build/row-column-release -R '^install_downstream_cmake_smoke$' --verbose --timeout 120
python tools/windows_dev.py dumpbin /exports build/row-column-release/rns8.dll
```

The one outer install CTest passed, containing four clean child executable
tests (two static, two DLL). Both continuation executions verified signed
prefix 19 / 23-byte budget / 19-byte reuse and unsigned prefix 20 / 26-byte budget /
20-byte reuse, one-byte-short rollback, original-export RANGE_ERROR and independent
three-limb expected integers. Both bounded consumer executions returned 111.
The static checks accompany the newly enabled DLL install configuration; no
earlier full correctness/fault/sanitizer/tooling campaign was repeated.

All five shipped consumer files match source/install/copy byte-for-byte. Both
fresh consumer caches have no CMAKE_TOOLCHAIN_FILE, disable package registries,
use only the staged RNS8 package, and build the copied self-contained project
without library-source includes. Verbose commands link rns8_static.lib or the
staged rns8.lib. Built, installed and consumer-copied DLL bytes are identical
(SHA256 `fb2a739f612873d60273782d9876be0fd101d6cb91696950c489e1d448b96a1c`).
Consumer compile warnings are errors; library compilation retains existing
unrelated unused-work-variable warnings. No library, link, runtime, assertion,
dialog or unexpected package failure occurred. Process-local error handling
remained enabled. CTest elapsed times are not benchmarks; ordinary Blendslop work
was permitted to coexist and no other workload was stopped.

The first ABI scratch setup failed compilation because its original-header copy
omitted moduli.h, an included dependency. The failed log is retained; the fixture
was corrected to copy the tracked original include/rns8 headers. All four final
MSVC probes compiled under /W4 /WX and ran successfully. The initial sandboxed
read-only GitHub ref query failed to connect through its local proxy; the allowed
noninteractive escalated read succeeded for the exact RNS8 main target. Search
commands with PowerShell literal wildcards were corrected with rg glob options.
These setup/access/search failures are not represented as passing product tests.

The integrity helper initially stripped leading porcelain-status spaces and
misparsed its first filename. Retaining those spaces corrected the helper;
the failed integrity-helper log is retained separately from its final pass.

The shell formatter entrypoint could not create its MSYS signal pipe in the
sandbox. Native Python avoided that launcher, but its temporary
`RNS8/.git/worktrees/row-column-range-20261008/index.lock` was denied
by the filesystem sandbox. Authorized project write access allowed the
installed formatter to proceed. Its first successful diff check returned
1 for two comment reflow differences; the proposed reflow was applied,
and the final check passed. This installed formatter also returns 1 when
applying changes; that apply status was expected. No auto-review rejection or owner action
was needed, and main was unchanged. Current-documentation claim validation
passed. Comment-stripped header tokens remain equal to the reviewed head.

Temporary consumer build directories and ABI probe executables/objects were
removed after exact resolved-path containment checks; source/header snapshots,
manifests, local staged product packages and raw evidence remain. No consumer or
ABI probe process remains running. The cleanup record is retained separately.

## Retained evidence and remaining regression risks

The recovery checkpoint reported 216 CPU checks, 216 ASan/UBSan checks with LSan
disabled, and 49 tooling checks. Later focused receipts qualify axis/lift/auto
operations under MSVC and clang-cl ASan, including 26 cases / 16,156 assertions
at the auto checkpoint. The allocation checkpoint reports 641 injected C++
allocation failures plus 641 retries per static Release/ASan build, six serial
scratch cases and 23 normal cases / 9,314 assertions per Debug/ASan build.
These are retained scoped results, not fresh repetitions or whole-library proof.
See the [axis](local-range-continuation-20261008.md),
[lifting](local-cpu-lifting-20261008.md),
[auto selection](local-cpu-auto-continuation-20261008.md),
[allocation](local-cpu-allocation-20261008.md) and
[package](local-cpu-package-20261008.md) receipts. Initial failures and skips
remain disclosed.

GPU source changes have the highest unresolved risk: actual HIP compilation,
lane layouts, compiler specialization/addressing, optional backends, sparse/wave
tails, graph lifetimes, on-device differentials and performance were not run.
Removing unsafe shortcuts can change speed; there is no measured estimate.
Historical GPU captures on original main do not qualify this source.

Current MSVC x64 package normal-path and C/C++ layout evidence is narrow.
Pure C installed consumers, cross-CRT/DLL fault injection, other compilers/OSes,
OpenMP, device lifting, broader current-source UBSan and LeakSanitizer remain
separate qualifications. Debug-STL OOM remains explicitly skipped because a
noexcept iterator-proxy allocation can terminate before guard_api; the skip is
not a pass. C++ allocator counters do not prove C malloc/OS OOM or global leaks.
Malformed source/shape/range/budget tests are stronger than success-only use,
but no exhaustive handle corruption, concurrency or dimension stress claim is made.

## Exact proposed integration route and owner decision

The integration candidate is the complete linear history from
`80791bd3d2900bd80b2ac6ba957d15329771043a` through `be958805c7d33436e24ada890ceee80944fad7a6`
and the review/comment receipt commit containing this file. Owner must accept
the inherited audit/documentation dispositions, the two additive CPU APIs,
payload-only budget, conservative rejection and matching-prefix export workflow,
then explicitly authorize integration. No publication/PR/merge authority is
assumed by this recommendation; a source integration does not authorize a release
or any hardware/paid-compute campaign.

After authorization, re-read current main/remote and require clean intended
checkout state. If both still equal the pinned base, `git merge --ff-only
codex/row-column-range-20261008` on owner main is the proposed history-preserving
local route. This advances main through every reviewed commit without rewriting
history or changing other saved branch tips; preserve the recovery ZIP and its
hash. Record the exact new main hash and rerun only integration checks warranted
by actual changes.
Push, PR or remote merge requires its own explicit authorization and exact target.

If main has advanced, stop the fast-forward route and review the full merge-base
diff against the new target in an isolated worktree. Resolve/recheck actual
conflicts there under the newly authorized integration scope. Do not reset or
force-push main, rebase existing saved tips, cherry-pick only be958805, or silently
use an unrelated divergent branch. Approval of the complete candidate is the
concrete owner decision remaining; no CPU package toolchain blocker remains.

## Current raw evidence

Ignored worktree temp files (SHA256):

| Evidence | SHA256 |
| --- | --- |
| `lineage-origin-main.log` | `82d9eda484818d19ab6fd19a079dd069cf5d6bf0634dd20b03a33490e3fe06cd` |
| `lineage-release-dll-configure.log` | `ca57aa3bd5a00ba66b54955554227840c11356523d274384004a03848192a515` |
| `lineage-release-dll-build.log` | `33823da433934c68740d8400dcd7fd0f531af0000a923ce5062c237ee1bf7f3f` |
| `lineage-release-dll-consumer.log` | `246a7badce7984dda577f8f212c0492860be3fbd211b4ec09ecf0e3d1fe378bb` |
| `lineage-release-dll-exports.log` | `0fba975bab9787c8f3f6757bd02d689515f85fb1dd4aa838c827d15bccc29e0e` |
| `lineage-release-package-manifest.json` | `40349ca789957eedb9a73c5823b3b885741f73d22cd8bafccb052bedba0605ca` |
| `lineage-release-package-manifest.log` | `13fbb08bd39ce7de2fb4dacb9785343177efa904cfe1e22996416a3b7962886e` |
| `lineage-msvc-layout.log` | `9630759a04623d4c0febd55b562cdd1cf2b03b3c340094b0777fe901f65a8f7f` |
| `lineage-msvc-layout-final.log` | `6fbe8632e642545b20f4d1852e2ecfc556cdd9f16e2740895bf3731bb339c14a` |
| `lineage-abi/manifest.json` | `9014a8c8d5c7edcbb74f282f80ca32a6defd92edbe99ae393ccb7810008e344b` |
| `lineage-abi/public_layout.c` | `b0864fe24d577af48947127451e5345f3673c788e4184ec3a1cb6b2098b491e5` |
| `lineage-abi/main-c17-layout.txt` | `a05df03f1d6d1e5e407725260df554f2a37339e09baea5705a9816da5e3ef745` |
| `lineage-abi/main-cpp17-layout.txt` | `a05df03f1d6d1e5e407725260df554f2a37339e09baea5705a9816da5e3ef745` |
| `lineage-abi/continuation-c17-layout.txt` | `a05df03f1d6d1e5e407725260df554f2a37339e09baea5705a9816da5e3ef745` |
| `lineage-abi/continuation-cpp17-layout.txt` | `a05df03f1d6d1e5e407725260df554f2a37339e09baea5705a9816da5e3ef745` |
| `lineage-cleanup.json` | `7bd9d3290a3adb628f2360ed187bd2346adfa232585126b7caba10f335be7f65` |
| `lineage-claims.log` | `aba9b288bd639f3d5d002f2453b946ccf9cff28f54e9879f272bc53faeadd807` |
| `lineage-format-diff.log` | `68ffbb5f50b07cc4b9059cf2e52bc24af45d575ba7706153d0fb6301db967982` |
| `lineage-format-apply.log` | `4e83cda3cde0457f40fd4753f13fb136a4c02710cba45a5fbaa07cebc03dc6fa` |
| `lineage-format-final.log` | `bb9d0c1ee1554da19bb32cb48bcc6bb6c63c619a7a47865de996352ed6ddf2e0` |
| `lineage-review-integrity.log` | `a95d0519aaa21a0402221b5fb048d6c634c49fff63dca0aa95d62e61ace64516` |
| `lineage-review-integrity.json` | `04b31b197b5b6d80ff6126e7241e9e89f4a75292753650c6660768a05fab7def` |
| `lineage-review-integrity-final.log` | `975dbf4e06a267970b66f7901b05a8d569a350f4232eb7d84154332498fe2cb6` |
