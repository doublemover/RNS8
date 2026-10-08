# Local CPU ladder-lifting checkpoint — 2026-10-08

## Lineage and ownership

This batch continues accepted row/column checkpoint
`11618181e6074bf3e3dcbf9b7c80b85c874884e9`, which continues recovered source
`80b9ffe1f8439ad0254918152746a77feb8035de`. Work remains on
`codex/row-column-range-20261008` in the project-local worktree
`C:/Users/sneak/Development/RNS8/temp/worktrees/row-column-range-20261008`.
The owning checkout remains at
`80791bd3d2900bd80b2ac6ba957d15329771043a`; its untracked recovery ZIP remains
unchanged, SHA256
`1dbce716c64356824cbdd6ae933e150d2c411363d16c8e3eda8966ec6965ae25`.
Origin still identifies `https://github.com/doublemover/RNS8.git`.

## Implemented contract

`rns8_lift_exact_wide_cpu(ctx, matrix, target_prefix, max_staged_residue_bytes)`
is an explicit additive C API with a forwarding C++ `Matrix` method. It preserves
existing GEMM/export behavior, all public enum/descriptor layouts and ABI version.
Plan selection and allocated storage ceilings remain unchanged. CPU matrices can
now initialize missing planes before a later larger-prefix operation; ordinary
GEMM still rejects unwritten planes until the caller explicitly lifts them.

The operation requires exact-wide semantics, CPU_REFERENCE ownership, structurally
valid current host storage, and a known initialized-prefix range proof. It checks
the strict signed `2 * bound < P` or unsigned `bound < P` inequality at the source
prefix. A larger target product cannot repair a prior alias or an absent proof.
Source bytes must lie in each modulus's centered INT8 interval. Each logical cell
is reconstructed using the existing multiprecision Garner reference, centered
for signed semantics, checked against the retained magnitude proof, then encoded
into the missing suffix planes.

The staging payload is exactly
`rows * cols * (target_prefix - initialized_prefix)` bytes when lifting. Checked
capacity arithmetic precedes allocation. The caller's byte limit is mandatory;
insufficient budget returns WORKSPACE_TOO_SMALL. Allocator bookkeeping and bounded
per-cell CRT scratch are excluded from that payload budget. At most 20 source
residue bytes and 155-bit range arithmetic are needed per cell; no full-matrix
integer reconstruction buffer is retained. Runtime scales with logical cells
and source/target prefix lengths, with no performance qualification claimed.

All suffix bytes are staged before any resident write. The final copy is into
already allocated INT8 storage; no potentially allocating work follows the first
write. Failures preserve resident bytes, initialized prefix, magnitude/axis
proofs, identity, source version, native state and currentness. Successful lifting
preserves those logical-value properties while updating only suffix bytes and
initialized prefix. Already initialized targets validate structure/currentness
and uniqueness before returning a no-op; they do not reconstruct cells.
Unsupported backend ownership is rejected without device transfers or fallback.

## Focused evidence

All execution used installed tools. Dependency installation remained disabled
with `VCPKG_MANIFEST_INSTALL=OFF`; HIP, benchmarks, tools, examples and CPU OpenMP
remained disabled. No toolchain, driver or HIP installation was attempted.

- MSVC 19.44.35228.0 Debug build: static library, shared DLL and `rns8_tests`
  passed using `python tools/windows_dev.py cmake --build build/row-column-cpu
  --target rns8_tests rns8 --parallel 2`.
- MSVC focused correctness:
  `build/row-column-cpu/rns8_tests.exe '[exact-lift],[exact-axis],[exact-chain]'
  --reporter compact --rng-seed 20261008` passed **18 cases / 14,166 assertions**.
- Installed clang-cl 22.1.6 AddressSanitizer build passed using the existing
  `build/row-column-asan` configuration and `--target rns8_tests --parallel 2`.
  The same focused selection passed **18 cases / 14,166 assertions** with
  `ASAN_OPTIONS=detect_leaks=0` and empty stderr. This is Windows ASan evidence,
  not UBSan or LeakSanitizer evidence.
- Downstream C17 `/W4 /WX` compilation and DLL execution passed for the new
  exported symbol, 18→19→20 stages, exact budget boundaries, signed continuation,
  failed higher-prefix export, and preserved padded input/output layouts.
- Downstream C++17 `/W4 /WX /EHsc /MDd` compilation and DLL execution passed for
  the forwarding `Matrix` method and rejected invalid-target exception.
- Changed-line clang-format and explicit new-source `.cpp`/`.inc` format checks,
  plus `git diff --check`, passed.

Five new lift cases cover every 1→2 through 19→20 source transition with zero,
positive/negative one and strict signed/unsigned CRT endpoints, including values
wider than 64/128 bits. They also cover public native-width products, signed
reconstruction, cancellation, overflow after lifting, source-proof equality
rejection, suffix-byte budgets, successful partial stages followed by rejection,
late malformed source bytes and proof contradictions, retry after correction,
unknown/negative proofs, stale state, foreign/unsupported ownership, insufficient
capacity, dimension multiplication overflow, truncated storage and idempotence.
The previous axis and chain cases protect the retained admission behavior.
No full qualification or large benchmark campaign was repeated.

Raw evidence is ignored under this worktree's `temp/`:

| Evidence | SHA256 |
| --- | --- |
| `ladder-msvc.log` and `ladder-asan.log` (identical summaries) | `ceee7c19bdb2bb46fd608bbe2bfdce66dcc843857282456493b39a5ae4d9462f` |
| Empty `ladder-asan-stderr.log` | `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855` |
| `ladder-c-abi.c` | `5f2fbf3dedefe1857e00403ccd7a295bc2d2e88e2600a525c71eaa21092fa18b` |
| `ladder-cpp-abi.cpp` | `602150b239d5617d357947de4d23b7bf038b74382bb619959b70a61d7b29a0b9` |
| `ladder-c-abi.log` | `f8a2664e7684110e8d36ad54fbe3c3ad517ec5df7fef9aea8f264a3085e2234e` |
| `ladder-cpp-abi.log` | `f4ebc2aa3f32f5c9b1b55e9b29a7e8f3fa6297274fae7133677ac789d11589ec` |

Temporary C/C++ smoke executables and object files were removed after verification.
The focused builds/tests/smokes had no failure. Existing unrelated unused-variable,
unused-function and reserved-enum switch warnings remain; the new implementation
had no reported warning. Allocation failure is caught transactionally by the
public guard, but no allocator fault injection or OS out-of-memory experiment was
performed.

## Remaining gates and next bounded chunk

Automatic admission-driven prefix selection, multi-operand staged lifting and
device-owned variants remain open. A coherent next CPU chunk is an explicit
opt-in admission route that chooses a sufficient prefix within plan/input/output
ceilings, stages both required input lifts with one resource budget, validates
workspace ownership and output range, and commits only after all preparation
passes. Existing calls must retain their current behavior.

HIP installation status is unverified. HIP compilation/execution, GPU transfers
and lifetime qualification, Linux/Instinct runs, UBSan/LeakSanitizer, allocator
fault injection, broader package/Windows ABI coverage and performance measurement
remain unqualified. Driver/HIP changes require a separately authorized step.
Stronger correlation-aware bounds and sparse/prepacked axis-summary propagation
also remain open.

Future integration must review the entire `main..codex/row-column-range-20261008`
lineage, including inherited audit/recovery commits. No publication, PR, merge,
branch-tip overwrite, or external integration action was performed.
