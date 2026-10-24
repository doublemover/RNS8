# Practical exact integer workloads

These four C++17 programs use installed public RNS8 headers and exported CMake
targets. Each starts with a task, proves its required integer range, compares
every output against an independent Boost.Multiprecision reference, and prints
an all-output checksum. GPU execution requests a named backend and device;
unsupported requests return status `unsupported`, exit77, and never fall back
to CPU. `--reference-only` explicitly selects CPU reference, device−1.

The programs and measurement paths are new. Application timings and Windows/
Linux installed-consumer qualification remain pending execution. Existing
[kernel measurements](../../docs/native-queue-performance-2026-10-23.md) describe
different scopes and do not establish an application speedup here.

## Tasks and tiny answers

| Program / target suffix | Task and arithmetic | Tiny row-major answer | All-output checksum |
|---|---|---|---|
| `exactness-walkthrough` | Checked exact integer arithmetic above64 and128bits, signed decimal and limbs | First witness `1180591620717411303409` (`2^70−15`); five fixed witnesses | `86e3a3eb9c2c71e6` |
| `integer-cross-product` | Transform integer observations by integer coefficients, exact bounded i64; `--wide` selects exact-wide | `[20,-27,4,-14]` | `93c5204bc9b095c2` |
| `integer-cross-product --gram` | Integer Gram matrix `XᵀX`, K observations and M=N features; squared feature units | `[14,-22,-22,161]` | `b18e5fce6ed0647a` |
| `gf251-transform` | Batched linear transform over prime field GF(251); canonical byte output | `[44,48,139,154]` | `af3cec33c3f7c268` |
| `resident-chain` | Two exact integer transforms `(AB)T`, same final signed192 output with resident or exported/reimported intermediate | `[67,-34,22,-24]` | `f4de4c5aad00cd96` |

Target names start with `rns8-workload-`. Checksums cover every output, including
all limbs, with the versioned serialization in [recipes.json](recipes.json).
They are reproducibility aids; correctness compares full outputs. The tiny
fixture is2×2×3, selected by `--size tiny` with no explicit seed/bits and the
default seeded pattern. Explicit seed/bits selects generated data instead.
`--gram` uses B as the observation matrix and derives A by transposition.

The exactness walkthrough evaluates four two-term products: `2^70−15`, signed
cancellation to zero, zero operands against full-width inputs, and
`INT64_MIN+INT64_MAX=−1`. Checked64 rejects overflowing multiplication or ordered
partial sums before executing them. Unsigned64 produces explicitly labeled
low bits; it is a wraparound negative control. Actual runtime binary64
conversion/multiplication/addition loses15 in the first witness under recorded
round-to-nearest. A genuine compiler `__int128` comparator runs when available;
MSVC reports it unavailable rather than substituting a Boost type. A separate
resident scalar chain yields `2^129`, beyond even unsigned128, then rejects
`2^156` outside the20-plane range while preserving the previous result.
Arbitrary-precision CPU arithmetic can compute all these finite expressions.

## Build and run

Use the explicit dependency/compiler configuration in the
[Windows](../../docs/platform-windows.md) or
[Linux](../../docs/platform-linux.md) setup guide. `RNS8_BUILD_EXAMPLES=ON`
builds these targets along with the existing tiny API examples. No new installer,
dependency autodetection, asynchronous API, or sparse API is required.

For an already configured build, the following PowerShell commands request
the direct HIP backend on device0. Replace `build/issues-all` with your own
build directory; on Linux omit `.exe`.

```powershell
cmake --build build/issues-all --target rns8-workload-exactness-walkthrough rns8-workload-integer-cross-product rns8-workload-gf251-transform rns8-workload-resident-chain
build/issues-all/examples/workloads/rns8-workload-exactness-walkthrough.exe --backend hip-direct --device 0 --size tiny --json
build/issues-all/examples/workloads/rns8-workload-integer-cross-product.exe --backend hip-direct --device 0 --size tiny --repeats 3 --warmups 1 --json
build/issues-all/examples/workloads/rns8-workload-integer-cross-product.exe --backend hip-direct --device 0 --gram --size modest --bits 12 --seed 3135 --json
build/issues-all/examples/workloads/rns8-workload-gf251-transform.exe --backend hip-direct --device 0 --size modest --seed 3135 --json
build/issues-all/examples/workloads/rns8-workload-resident-chain.exe --backend hip-direct --device 0 --size tiny --json
```

The walkthrough always makes one observation per fixed witness; its only
accepted explicit repetition controls are `--repeats 1 --warmups 0`. For other
programs, unchanged inputs reuse source version1; an actual A change uses
version2 and triggers a new independent complete reference comparison. Gram
updates the corresponding B entry too. A deterministic cyclic order rotates
comparators between warmups and repetitions. These are synchronous calls, not
overlapped GPU work. Other named backends can be requested with `--backend`;
their implemented semantic contracts determine support. No AUTO route exists
in this harness. The finite, resident and walkthrough programs reject controls
that do not apply to their contracts.

For explicit CPU correctness runs use, for example:

```powershell
build/issues-all/examples/workloads/rns8-workload-integer-cross-product.exe --reference-only --size tiny --json
ctest --test-dir build/issues-all -R "^workload_" --output-on-failure
```

The CTest workload set includes CPU cases, configured HIP cases, independent
baseline unit checks, CLI/report checks, and installed consumers. The installed
consumer tests deliberately install/configure/build in test-owned build
subdirectories and may take several minutes. GPU cases skip only when the
actual request reports unsupported; a CPU run is never counted as GPU coverage.

## Range, comparators and resource limits

The integer cross-product admits bounded64 only when
`K*max_abs(A)*max_abs(B) <= INT64_MAX`, including the changed input. This bounds
every ordered partial sum without relying on cancellation. `--wide` requires
`P20 > 2*bound` and exports three signed little-endian u64 limbs per cell. The
actual selected prefix and its information range are recorded.

The resident experiment proves `AB` fits signed64, then proves the full `(AB)T`
envelope is uniquely represented by the20-plane odd-modulus product. Both
variants retain their allocations, pack the same initial native inputs, perform
both GEMMs every time, and export the same final signed192 matrix. One variant
keeps `AB` resident; the other reconstructs/downloads one signed64 limb per cell,
copies its bit pattern into i64, and calls public `pack_i64` with version0 to
force reimport. This charges the actual intermediate reconstruction, transfer,
repacking and proof work. Public arbitrary-width limb import is unavailable;
the experiment therefore rejects a wider intermediate rather than using private
headers. It demonstrates storage reuse for a matched two-stage computation,
not a general arbitrary-width import/export loop or result-cache speedup.

Integer baselines include straightforward `cpp_int`, blocked `cpp_int`, blocked
native64 where the entire expression fits, and genuine native128 where supported
and sufficient. Their measured loops include conversion to the same final word
format as RNS8. They are serial C++ comparators, not vendor-tuned BLAS or a claim
of the fastest possible CPU implementation. No GMP/FLINT application adapter is
implemented; each report says `not_run` for these optional references. The
field compares reduction after every product with a blocked u32 implementation
reducing every256 K terms. Its partial maximum is
`250 + 256*250² = 16000250 < 2^32`. GF(251) is neither GF(2^8) nor the mod256 ring.

Defaults are M=N64,K96,bits8,seed3135,three repeats and one warmup. `--size tiny`
uses2×2×3; `--size large` explicitly opts into256³. `--m/--n/--k` override
individual dimensions within1..512. Work above4million scalar products requires
`--size large`; all programs retain a complete independent reference, so this
is also a real CPU cost. `--baseline-max-products` bounds whether the additional
naive comparator is timed; omitted timing stays `not_run`, without extrapolation.
All reports retain the strongest implemented correct blocked comparator.

Before allocating data, a shared budget reserves1024bytes per modeled host
input/intermediate/output cell for all simultaneously retained baselines and
old/new references. Cross-product counts `MK+KN+MN` cells; the chain counts
`MK+KN+NN+3MN`. At allowed inputs each exact value stays below192bits, requiring
at most three significant64-bit limbs; the allowance covers two retained
`cpp_int` comparator states (four intermediate/output arrays for a chain),
native64/128 arrays, their serialized outputs, inputs and full reference
temporaries. This is a conservative storage model, including room for normal
multiprecision capacity growth; it is not an allocator-independent peak bound.
Every retained RNS8 candidate then adds its own128bytes per matrix cell plus
publicly reported workspace bytes to the same aggregate budget before allocating
those matrices/workspaces. Both resident and roundtrip candidates are charged,
including the latter's extra imported matrix. The aggregate reservation must
fit `--memory-mib` (default256, range1..4096), and is emitted in the report.
For a large chain, explicitly raise the budget, for example `--size large
--memory-mib 1024`. Device envelopes also check total device memory. These are
estimated admission envelopes, not free-memory measurements; allocation
failures remain real errors. There
is no machine-independent time bound. The collector timeout bounds each process
and retains timeout/error output. Large examples are opt-in and may be slow in
debug/reference builds.

## Measurement and independent collection

Each JSON observation keeps the raw host monotonic samples, clock calibration,
nominal resolution, timer-pair median/p95, warmup count, order, setup, teardown,
phase scopes, complete reference cost, and separately measured verification.
`median_steady_ns` is the usual median (average of the two central values for
even counts); `calls_per_second` counts complete application calls, not INT8
TOPS or mathematical products. Min/max show within-process variation.

Cold application total is explicitly the sum of measured common input/layout
preparation, per-variant resource setup, first synchronous call and eventual
resource teardown. Teardown occurs after the reuse experiment; this sum is not
one contiguous wall-clock sample. Steady samples retain allocations and stable
input generations, but recompute and export every result. Verification, report
formatting and I/O are outside both measured calls. Public APIs combine pack/
upload and reconstruction/download, so isolated transfers, kernel-only timing
and device events remain null with a reason. No invented timing separation is
reported. Bounds/proof work inside the public calls stays charged.

Collect independent processes rather than treating inner repetitions as
independent experiments:

```powershell
python tools/collect_workload_examples.py --exe build/issues-all/examples/workloads/rns8-workload-integer-cross-product.exe --output temp/workloads/cross-modest.json --runs 5 --timeout 120 -- --backend hip-direct --device 0 --size modest --bits 12 --seed 3135 --repeats 9 --warmups 3
python tools/collect_workload_examples.py --exe build/issues-all/examples/workloads/rns8-workload-resident-chain.exe --output temp/workloads/chain-tiny.json --runs 5 -- --backend hip-direct --device 0 --size tiny --repeats 9 --warmups 3
```

The collector binds executable hashes before/after each process, unique run IDs,
commands, collector hash, compiler/library build manifest, example source and
recipe hashes, requested/effective device/backend, semantic/output contracts,
exact range, full checksums, and changed-input generations. It reconstructs the
recipe bytes and records their SHA-256. Cross-process metadata disagreement,
binary mutation, unsupported requests, malformed output and failed comparisons
are retained as failed collection records; no fallback or success substitution
occurs. The library manifest and independent example source identity describe
different source scopes. An installed consumer records its copied library
manifest and newly compiled example hashes; this alone is not runtime attestation
of an arbitrary substituted shared library.

The optional `--isa-file FILE --variant NAME --instruction LABEL --clocks-policy
POLICY --power-policy POLICY` adapter uses the existing
[`rns8_calibration_v1` tooling](../../docs/calibration.md) for one complete
steady application variant in units of calls. Supply a real ISA artifact; no
placeholder digest is generated. Below-threshold times are preserved raw and
censored to null in the adapter. A file hash alone does not verify correspondence
between instructions and selected kernel: `isa_verified=false`, rate export and
promotion remain ineligible. Direct observations are not benchmark-schema4
qualification. There is no speedup conclusion or winner selection in this
harness; matched cold/steady costs can show a slowdown or an inconclusive result.

## Installed public consumer

After the configured library and workload targets build:

```powershell
cmake --install build/issues-all --prefix temp/workload-package
cmake -S temp/workload-package/share/RNS8/examples/workloads -B temp/workload-consumer -G Ninja -DCMAKE_PREFIX_PATH="$PWD/temp/workload-package" -DCMAKE_TOOLCHAIN_FILE=C:/vcpkg/scripts/buildsystems/vcpkg.cmake -DRNS8_WORKLOAD_GPU_TESTS=OFF
cmake --build temp/workload-consumer
temp/workload-consumer/rns8-workload-integer-cross-product.exe --backend hip-direct --device 0 --size tiny --json
```

The consumer requires external Boost.Multiprecision headers and nlohmann-json;
use the existing [dependency guide](../../third_party/README.md) overrides if
needed. It links `rns8::rns8_static` by default; select
`-DRNS8_WORKLOAD_USE_SHARED=ON` when the installed package exports `rns8::rns8`.
No private source headers, repository data files, source-tree caches, or network
downloads are needed by the installed source. The separate
[small downstream consumer](../downstream-cmake/README.md) remains available.

These are arithmetic primitives for integer statistics, symbolic computations
and prime-field transforms, not complete application integrations; see the
[application research catalogue](../../docs/research/README.md). Sparse input
contracts and installed sparse examples remain owned by
[#51](https://github.com/doublemover/RNS8/issues/51). Multi-GPU and asynchronous
workloads require their own implemented lifecycle and evidence. Setup/session
and dependency acceptance gaps tracked by#57/#58/#60 remain separate from these
application sources; no untested platform or backend qualification is implied.
