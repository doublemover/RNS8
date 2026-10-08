# Local row/column exact-wide continuation — 2026-10-08

## Recovery and lineage

The owner-provided `RNS8-Range-Continuation-80b9ffe-2026-10-06.zip` at the
verified RNS8 root is unchanged. Its size is 4,612,937 bytes and SHA-256 is
`1dbce716c64356824cbdd6ae933e150d2c411363d16c8e3eda8966ec6965ae25`.
Both archive levels were checked for path escapes, duplicate case-insensitive
paths, and link/special-file entries, and their CRC checks passed. The six
artifact size/SHA-256 records in `checkpoint.json` passed. The complete Git
bundle verified, and all 615 tracked source files in the nested `source/` ZIP
matched their saved Git blob hashes.

Recovered head: `80b9ffe1f8439ad0254918152746a77feb8035de`.
Recovered tree: `73eb14d5e2c1091ee2f9bd1ebf682bbb495abd1b`.
Packet continuation base: `3855bb5b3e4dcb8d1111f0d4e478ebd19e2bb8b1`.
Owner main: `80791bd3d2900bd80b2ac6ba957d15329771043a`, unchanged.
Main is an ancestor of the recovered head. Five inherited unpublished commits
contain the earlier audit, documentation reconciliation, zero-inference repair,
and initialized-prefix/range repair. This batch descends directly from the
recovered head on `codex/row-column-range-20261008`, in the project-local
`temp/worktrees/row-column-range-20261008` worktree. Existing branch tips and
owner files were preserved. No Library retry, origin fetch, publication, PR,
merge, new agent, toolchain installation, or benchmark campaign occurred in
this local recovery/implementation run.

## Implemented batch

Native exact-wide packing stages private row/column maximum magnitudes,
absolute sums, exact sums, extrema and midpoint L1 deviations. All summary
arithmetic is Boost.Multiprecision; padding is excluded and signed minima and
unsigned sums beyond 64 bits remain exact. Stable device-source versions retain
the existing logical value/proof; rejected repacks preserve the prior proof.

Dense, grouped and incremental host admission tighten the global K*max(A)*max(B)
proof using pairwise row/column L1/max inequalities. When both packed axis sums
are exact, the centered-dot-product identity supplies another conservative
bound. A constant INT64_MAX axis against balanced signed columns now yields a
zero proof and can continue through a later GEMM without CRT aliasing.

Successful operations swap in staged output row/column magnitude maxima and
absolute-sum bounds. These output summaries do not claim exact signed sums.
Missing summaries retain scalar admission, and scalar-only sparse/prepacked
outputs discard stale axis summaries. Written-prefix checks, strict signed
factor-of-two CRT admission, aliases, immutable zero-skip contracts, and public
ABI layouts are preserved. The implementation performs two linear input scans
and O(M*N) pairwise admission with O(rows+columns) retained metadata. No speedup
is asserted, and some cancellation-safe products remain conservatively rejected.

Seven new focused cases cover wide signed cancellation and later rejection,
64-bit boundary sums and padding, unsigned heterogeneous chains, missing-proof
fallback and restoration by repack, transactional rejection/stale-summary
clearing, 128 deterministic small signed/unsigned differential products, and
rectangular/padded output. Existing chain tests assert sound bound inequalities
instead of requiring the earlier looser global bound.

## Validation on the installed Windows environment

- MSVC 19.44.35228.0 Debug static library, shared DLL and test target built.
- `rns8_tests.exe 'exact-wide*' --reporter compact --rng-seed 20261008`:
  31 cases, 8,026 assertions passed.
- `rns8_tests.exe 'default modulus*,prefix range bits*,reserved research APIs*'
  --reporter compact --rng-seed 20261008`: six cases, 496 assertions passed.
  This includes ladder coprimality/product boundaries and the C++ enum ABI
  compile assertions in the public-ABI translation unit.
- A downstream MSVC C17 caller compiled with `/W4 /WX`, linked the new import
  library, and executed the wide cancellation and zero-chain export against
  `rns8.dll`. This is one Windows C shared-library smoke, not full ABI qualification.
- Installed clang-cl 22.1.6 ASan static build completed with installed
  x64-windows-static dependencies. `ASAN_OPTIONS=detect_leaks=0` and
  `rns8_tests.exe '[exact-axis],[exact-chain]' --reporter compact
  --rng-seed 20261008`: 13 cases, 7,311 assertions passed; stderr was empty.
- Changed-line git-clang-format check, new/header clang-format dry-run with
  warnings treated as errors, and `git diff --check` passed. Formatting used
  Google style, column limit 110; dependent textual test includes disable sorting.

Both CMake configurations explicitly used `VCPKG_MANIFEST_INSTALL=OFF`; HIP,
accelerators, benchmarks, tools, examples and CPU OpenMP were disabled. MSVC
reused `C:/Users/sneak/Development/RNS8/build/cpu-release/vcpkg_installed`.
ASan reused `C:/Users/sneak/Development/RNS8/build/windows-clang-asan-debug/vcpkg_installed`.
The builds live at `build/row-column-cpu` and `build/row-column-asan` within the
isolated worktree. Focused logs are under its ignored `temp/`:
`exact-wide-msvc.log`, `moduli-abi-msvc.log`, `axis-asan-build.log`,
`exact-axis-asan.log`, and `exact-axis-asan-stderr.log`.
The C smoke source is `temp/axis-c-abi.c`, SHA-256
`de18fd5307ed3bb24dfd37d8743af0a2b81a337ab640bd365af29ca68257f098`.
Its temporary executable/object were removed after the successful run.

Validation corrections retained in the run transcript: the initial source-ZIP
comparison assumed no `source/` prefix; the corrected 615-file comparison passed.
Formatting initially sorted dependent test fragments before their fixture,
causing a compilation failure; explicit fragment order fixed it, and both final
builds passed. The first C command applied `/TC` to the import library as well
as the source; moving the library after `/link` fixed the command. A default
sandbox formatting check could not create the isolated Git scratch index lock
at `C:/Users/sneak/Development/RNS8/.git/worktrees/row-column-range-20261008/index.lock`;
the authorized rerun succeeded. These failed attempts are not counted as passes.
Existing unused-variable/function and reserved-enum switch warnings remain in
unrelated source; no arithmetic or sanitizer failure occurred.

## Remaining gates and next coherent chunk

Automatic ladder lifting remains unimplemented. The next bounded chunk is staged
CPU lifting from a proven initialized prefix: reconstruct only when the current
range proof identifies the integer uniquely, compute additional residue planes,
and atomically commit the widened prefix while preserving all aliases/currentness
and failure guarantees. Device variants require their own compilation/execution
and transfer/lifetime checks. Stronger correlation-aware bounds and immutable
axis summaries in sparse/prepacked input paths also remain open.

HIP compilation/execution and grouped/incremental/prepacked GPU behavior,
performance/profiler evidence, UBSan and LeakSanitizer for this new batch,
broader Windows ABI/runtime coverage, and Linux/Instinct qualification remain
unqualified. The packet's historical 216 CPU, 216 ASan/UBSan and 49 tooling
campaign is preserved evidence for that saved source and was not repeated.
For future integration, review the entire `main..codex/row-column-range-20261008`
lineage, including those inherited audit commits. No integration action has
been performed or authorized by this run.
