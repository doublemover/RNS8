# Opt-in CPU continuation checkpoint — 2026-10-08

## Lineage and scope

This slice continues accepted lifting commit
`6d9dcbd910fc94acda8d817ca28cd6f114b7a0c9`, after row/column admission commit
`11618181e6074bf3e3dcbf9b7c80b85c874884e9` and retained source
`80b9ffe1f8439ad0254918152746a77feb8035de`. It remains in
`C:/Users/sneak/Development/RNS8/temp/worktrees/row-column-range-20261008` on
`codex/row-column-range-20261008`.

The owner checkout remains at
`80791bd3d2900bd80b2ac6ba957d15329771043a`, with its untracked recovery ZIP
unchanged (SHA256
`1dbce716c64356824cbdd6ae933e150d2c411363d16c8e3eda8966ec6965ae25`). Origin
still identifies `https://github.com/doublemover/RNS8.git`. No publication, PR,
merge or existing unrelated branch-tip change occurred.

## Implemented behavior

The additive C API `rns8_gemm_exact_wide_cpu_auto` and C++
`gemm_exact_wide_cpu_auto` wrapper provide explicitly opt-in CPU dense GEMM
selection and transactional operand/output staging. Existing GEMM/export calls,
ABI version, enum/descriptor layouts and backend routing retain their behavior.

The operation validates the original CPU plan/workspace identity and schedule,
matching exact-wide shapes, CPU storage/currentness, and each input's known
unique integer proof at its own initialized prefix. It requires strict signed
`2 * bound < source_product` or unsigned `bound < source_product`; unknown or
nonunique inputs remain errors even when the output could be zero.

The original range-bound construction is now shared as a pure helper, retaining
scalar, axis/L1 and exact centered cancellation proofs. The route selects the
smallest sufficient output prefix from the original plan's selected prefix up
to the minimum of plan/A/B/C ceilings. FORCE_FIXED_PREFIX is honored, with no
lower-prefix optimization or plan/workspace mutation. If the conservative proof
cannot establish any allowed range, it rejects without computing a reference
GEMM to justify acceptance.

Both missing input suffixes and every selected output plane are staged under one
caller-provided residue-payload byte limit. The exact payload is:

```text
sum(distinct_input_cells * max(selected - initialized_prefix, 0))
+ M * N * selected
```

Identical `A == B` inputs are staged/count once when shapes permit; output/input
aliases are invalid. Capacity products and the combined sum are checked before
payload allocation. Allocator bookkeeping, O(M+N) output range summaries,
bounded per-cell CRT scratch and the serial blocked ring kernel's `4 * N` byte
row accumulator are explicitly excluded from this residue-payload limit.
Insufficient budget returns WORKSPACE_TOO_SMALL; insufficient range/proof or
size overflow returns RANGE_ERROR; invalid handles/contracts/storage return
INVALID_ARGUMENT, unsupported backends fail explicitly, and caught allocation
or other exceptions return INTERNAL_ERROR.

Shared lift helpers validate centered source bytes, reconstruct and center signed
integers using multiprecision Garner, enforce retained magnitudes, and stage only
needed suffixes. Computation reads a plane view combining current input storage
and staged suffixes, without copying complete input residue arrays. The existing
blocked ring primitive runs serially for this operation so accumulator allocation
failures stay inside the public guard even in an OpenMP-enabled library. Ordinary
CPU GEMM retains its existing parallel policy.

All preparation, payload allocation, reconstruction and output computation
finish before the first resident commit. Input suffixes and output bytes then
copy into their existing allocations; proof/currentness commits require no
allocating work. Failures leave A, B, C, plan/workspace and selected-prefix output
unchanged. Success preserves each input's logical value, source version, identity,
native state and scalar/axis proofs. C receives the existing GEMM source-version
and host-currentness conventions plus its selected-prefix output proof. Bytes
above that output prefix remain unchanged and are not admitted as current.

The selected prefix is returned explicitly. A result exceeding the original
plan's range needs a matching FORCE_FIXED_PREFIX plan for existing export APIs;
export through the inadequate original prefix still rejects and preserves the
caller buffer. Repeated opt-in calls can continue within allocated twenty-plane
ceilings without rewriting plan/workspace handles.

## Focused evidence

Installed MSVC 19.44.35228.0 built the static library, DLL and tests in
`build/row-column-cpu`. Installed clang-cl 22.1.6 built the existing
`build/row-column-asan` configuration. HIP, benchmarks, tools, examples and CPU
OpenMP were disabled; `VCPKG_MANIFEST_INSTALL=OFF` prevented dependency installs.

```powershell
python tools/windows_dev.py cmake --build build/row-column-cpu --target rns8_tests rns8 --parallel 2
build/row-column-cpu/rns8_tests.exe '[exact-auto],[exact-lift],[exact-axis],[exact-chain]' --reporter compact --rng-seed 20261008
python tools/windows_dev.py cmake --build build/row-column-asan --target rns8_tests --parallel 2
$env:ASAN_OPTIONS = 'detect_leaks=0'
build/row-column-asan/rns8_tests.exe '[exact-auto],[exact-lift],[exact-axis],[exact-chain]' --reporter compact --rng-seed 20261008
```

Both focused runs passed **26 cases / 16,156 assertions**, with empty ASan stderr.
The selection has eight new auto cases plus the directly affected lift, axis and
chain cases. It is not a repeat of the retained full qualification campaign.

New cases cover public native-to-wide signed/unsigned chains requiring larger
prefixes, asymmetric initialized input prefixes, strict signed/unsigned output
product boundaries, one-byte-short/exact combined budgets, rectangular staging,
late malformed/proof-inconsistent B cells after all of A staged, complete rollback
and retry, unchanged destinations and allocation identities, shared-input budget
deduplication, fixed-prefix behavior, each operand/output/plan ceiling, wrong
workspace binding, unknown/negative/equality proofs, zero-input conservatism,
wide cancellation with/without available axis proofs, inappropriate aliases,
stale state, shape mismatch and explicit backend rejection.

Downstream C17 and C++17 DLL compile/runs passed under `/W4 /WX` (C++ also
`/EHsc /MDd`). C exercised two public prefix-17 products, combined-budget rollback,
prefix-19 multiplication, inadequate-original-prefix export preservation,
matching-prefix limb export and a subsequent opt-in call with no new input planes.
C++ exercised the returning wrapper, signed output and budget-exception rollback.
Changed-line/new-fragment clang-format and staged `git diff --check` passed.
Temporary smoke executables/objects were removed after verification.

Raw evidence is ignored under worktree `temp/`:

| Evidence | SHA256 |
| --- | --- |
| `cpu-auto-msvc-build.log` | `578f329c2c6a0db407d3f87d6159711bd96ac6db1fef6e86788f9c874efe6650` |
| `cpu-auto-asan-build.log` | `cbd4910ff036c21ac2b93a18fa8277637eff20d3e04344e0207db052b042fe10` |
| Identical `cpu-auto-msvc.log` / `cpu-auto-asan.log` | `7f4617e554acf2d3e7d2a8d46803b697d90056b25ed7687f10139e81522c4328` |
| Empty `cpu-auto-asan-stderr.log` | `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855` |
| `cpu-auto-c-abi.c` | `092fb0182252a22f27c538393872e982cbd16dc1cd6ce732956cd83b5c47aa4c` |
| `cpu-auto-cpp-abi.cpp` | `a9ccee2d8c2b232c3a271a7c0996fea614e501a8e17daa07c95a4d3d111c48a7` |
| `cpu-auto-c-abi.log` | `29b4095bad6f2e17a653ab143994489961de9b4dac25fe85074a1125767b2662` |
| `cpu-auto-cpp-abi.log` | `78062e91baaeb1797a6e337ba4d25ef6205e28b57485252c5f93a03d4ac2fa92` |

No build, correctness, sanitizer or DLL smoke failure occurred. Existing unrelated
unused-variable/function and reserved-enum switch warnings remain; no new lift or
bound-helper warning was reported. Expected negative-test statuses are asserted
passes. No allocator fault injection or OS out-of-memory experiment was run.

## Completion and remaining acceptance

The planned bounded CPU continuation is implemented: row/column chain proofs,
unique-range CPU lifting, and opt-in selection with transactional staging of
both operands and output. No additional CPU feature is required to close this
implementation sequence within the documented ceiling and payload contract.

The owner acceptance point is review of the additive opt-in API, its payload-only
budget and matching-prefix export workflow, together with the complete
`main..codex/row-column-range-20261008` lineage including inherited audit commits.
Publication/integration remains unauthorized and was not attempted.

This checkpoint does not establish release qualification. HIP/driver installation
status remains unverified and untouched; device lifting/GPU execution, Linux/
Instinct runs, UBSan/LeakSanitizer, OpenMP build qualification, allocation-failure
injection, broader Windows/package ABI coverage and performance measurement
remain separate gates. Stronger correlation proofs and sparse/prepacked axis
propagation remain research outside this accepted CPU continuation scope.
