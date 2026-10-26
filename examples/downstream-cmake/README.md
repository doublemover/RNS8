# Consume the RNS8 package

This directory is a self-contained C++17 consumer of the installed public API.
It builds a bounded scalar smoke program, a signed/unsigned CPU exact-wide
continuation example and two explicit Sparse-A programs. It needs the RNS8 package and a C++ compiler; it uses no
repository-private headers, Boost headers, package manager or GPU.

The development package installs these sources under
`share/RNS8/examples/downstream-cmake` by default. From a Visual Studio developer
PowerShell on Windows:

```powershell
$rns8Prefix = 'C:/path/to/staged/RNS8'
cmake -S "$rns8Prefix/share/RNS8/examples/downstream-cmake" -B consumer-build -G Ninja "-DCMAKE_PREFIX_PATH=$rns8Prefix" -DCMAKE_BUILD_TYPE=Debug
cmake --build consumer-build --parallel 2
ctest --test-dir consumer-build --output-on-failure
```

Match the consumer build configuration and MSVC runtime with the installed
package. In a repository checkout, `tools/windows_dev.py` can load the developer
environment for each CMake command. Other platforms use the same CMake project
with their native compiler.

The default link target is `rns8::rns8_static`. To consume an installed shared
package, configure with `-DRNS8_DOWNSTREAM_USE_SHARED=ON`; that selects
`rns8::rns8`. Windows builds copy the imported RNS8 DLL next to the programs.
The shared option fails clearly if the package has no shared target. All
consumer targets compile with strict warnings and have ten-second CTest limits.
The private runtime helper routes Windows errors/assertions to stderr using
process-local settings. It does not turn failures into passes.

# CPU continuation workflow

`exact_wide_cpu_continuation.cpp` uses RAII handles from `<rns8/rns8.hpp>` and
expected-error status checks from the C API. Every matrix has capacity for twenty
modulus planes. Ordinary public GEMM first produces operands wider than 64 bits
at the original plan's selected prefix, currently seventeen.

| Explicit semantics | Wide operand | Product | Selected prefix | Exact initial payload | Reuse payload |
| --- | --- | --- | ---: | ---: | ---: |
| signed | `INT64_MIN * 1024 = -2^73` | `2^146` | 19 | 23 bytes | 19 bytes |
| unsigned | `UINT64_MAX * 1024` | `(2^64 - 1)^2 * 2^20` | 20 | 26 bytes | 20 bytes |

The program checks these observable steps:

1. Ordinary GEMM rejects the wider product with `RNS8_RANGE_ERROR` and preserves
   the existing output.
2. Two consecutive one-byte-short opt-in calls return
   `RNS8_WORKSPACE_TOO_SMALL`, preserve the selected-prefix output argument, and
   retain input/output integers and public storage/version/currentness metadata.
3. The exact-budget call succeeds through `rns8::gemm_exact_wide_cpu_auto` and
   returns the selected prefix. Input integer values and source versions persist.
4. Export through the original seventeen-plane plan returns `RNS8_RANGE_ERROR`
   and preserves its destination. A new `RNS8_PLAN_FORCE_FIXED_PREFIX` plan using
   the returned prefix exports the complete value into three 64-bit limbs.
5. A repeated call reuses initialized input planes and needs only the selected
   output-plane payload. The original plan remains unchanged.

Limb arrays are least-significant limb first; signed export uses fixed-width
two's complement. The program compares against independently derived power-of-two
limb constants, including the negative signed operand, without a CRT oracle.

The general residue-payload formula is

```text
sum(distinct_input_cells * missing_planes) + output_cells * selected_prefix
```

For a caller that does not know the selected prefix or initialized input ranges,
`ceiling * (A_cells + B_cells + C_cells)` is a conservative payload budget for
distinct A/B, with checked size arithmetic. The returned prefix determines the
export plan. The 19/20 expectations above are fixtures for the current default
ladder, not a rule for arbitrary matrices. Shared input identity counts once.
Range summaries, allocator overhead, CRT scratch and the CPU row accumulator
are outside this residue-payload limit; it is not a total memory limit.

The examples use `RNS8_BACKEND_CPU_REFERENCE` explicitly and perform no backend
fallback or device work. They demonstrate successful and budget/range-rejected
calls; they do not qualify Debug STL out-of-memory handling, allocator faults,
OpenMP, GPU execution or all platform/package combinations.

## Explicit CPU Sparse-A workflows

`sparse_finite_ring.cpp` and `sparse_bounded_i64.cpp` build as
`rns8-downstream-sparse-finite-ring` and `rns8-downstream-sparse-bounded-i64`.
They also build from the repository as `rns8-example-sparse-finite-ring` and
`rns8-example-sparse-bounded-i64`. The normal standalone build/CTest commands
above include all four consumers, for either installed static or shared target.
These two new programs and package registrations are authored only; execution
and installed-platform validation await the consolidated batch.

Both initialize every ABI, semantic, layout and sparse descriptor field explicitly.
Six rows cover all six ascending two-position patterns in four-lane K groups.
K=8 is divisible by four; A/B inputs and C exports have padded host strides. They
query compression sizes, compare compressed values/indices against an independent
lane oracle, import those packed arrays through the public expansion helper,
then pack a separate resident sparse handle. Public APIs currently do not directly
import packed arrays into a resident handle; expansion plus resident packing is
the available explicit sequence. No private storage accessor is used.

The finite-ring example uses composite modulus250, canonical unsigned bytes and
one plane (`max_prefix=0`). Values include125 and2: their product is zero in that
ring even though both inputs are nonzero. The bounded signed example uses the
public default modulus accessor and `RNS8_DEFAULT_BOUNDED_PREFIX`, with explicit
signed centered bytes for every plane (`finite_modulus=0`). Its original integer
values have magnitude at most8, below every selected modulus, so the selected
two nonzero positions remain valid independently in all planes. The unsigned-byte
helper transports those signed byte bit patterns; it does not change semantics.

Exactly two **stored nonzero bytes** per four-element group are required by V1;
"at most two" is not accepted. In general, nonzero integer input can become zero
in one channel. The bounded example re-encodes one value as the first modulus,
then checks that the resulting one-nonzero first-plane group is rejected rather
than treating the original integer mask as a per-plane structural proof.
RNS and finite-ring zero identities must not be used as integer-zero proofs.

The programs reject malformed zero/one/three/four-count groups, duplicate/reversed
or reserved index encodings and non-four-aligned K. A failed later-group resident
replacement must preserve its values, instance, source version and currentness.
The finite example also checks that a mismatched modulus preserves the old C.

Each program creates a plan/workspace, runs sparse and dense public controls,
exports every result and compares both with an independent ordinary scalar
integer product. Finite sums are bounded by `8*249^2` before reduction; signed
partial sums by `8*8*3=192`, below the explicit output bound4096 and `INT64_MAX`.
The reference therefore needs no Boost/GMP dependency or CRT reconstruction.
Full output matrices, deterministic recipe identities and a checksum over every
logical result are printed; padding is checked separately.

The selected route is always `CPU_REFERENCE`, printed as CPU sparse expansion.
There is no implicit fallback or target probing. These examples do not qualify
SMFMAC/SWMMAC hardware, gfx1100 sparse support, hybrid scheduling or a speedup.
GPU examples and measured same-contract profiles remain separate #51 acceptance.

## Hard-budget streaming consumer

`rns8-downstream-stream-bounded-i64` queries a 17x19x33 product with 3x4x5
panels through the installed C API, reports required/minimum payload capacities,
and proves that its device budget is smaller than complete resident RNS storage.
The default query-only mode requires no GPU and does not claim execution.
Pass `--hip` to explicitly require direct HIP, run two source generations,
compare every native result/padding cell to an independently bounded scalar
oracle, and check context retention and rejected-bound output preservation.
No fallback is performed. Configure `RNS8_DOWNSTREAM_GPU_TESTS=ON` to register
this execution in the installed consumer CTest suite. Reported capacities are
requested payload bytes, not observed allocator/driver peaks or speedups.

## Experimental materialized product consumer

`rns8-downstream-materialized-product-recipe` uses the installed
`rns8/experimental_product.h` header. It queries a graph/build identity, supplies
explicit schema 2 illustrative cost tables, executes `2*3*3*3`, and verifies the
exact result 54 with one shared B pack. It prints the retained arena footprint and
checks every cleanup status. It sets `cost_schema_version=2` and
`node_costs_size=sizeof(rns8_product_materialization_node_costs)`, stores the
81,312-byte node tables on the heap, and prices every valid physical capacity.
Packed GEMM uses `capacity*21+compute_prefix`; packing uses
`(capacity*21+old_prefix)*21+new_prefix`. The physical capacity is shared across
B readers even while only a smaller prefix has been packed. Unpriced cells
remain `UINT64_MAX`. Its CPU estimates exercise selection and are not measured
performance data. The same standalone CMake consumer tests static and
shared linkage; the schema 2 consumer awaits integrated validation.


## Checked integer CRT policy exports

`checked_crt_export.cpp` uses only installed public headers and targets. It runs
Garner, direct, balanced, local-group and split-group policies with signed
one-limb output, padded destinations and stale-source rejection. Explicit
rebinding accepts a new source generation within the retained range envelope;
the owner retains its payload reservation. The installed static and shared
consumer projects include this example. `RNS8_DOWNSTREAM_GPU_TESTS=ON` adds its
`--hip` run alongside streaming; an uncompiled HIP backend returns a visible skip.

These new cases are authored and syntax-checked, with execution deferred. The
[CRT guide](https://github.com/doublemover/RNS8/blob/codex/runtime-cache-transactions-20261011/docs/balanced-crt.md#prepared-checked-integer-exports) describes
the contracts and payload accounting; installed users can read the same guide
in the source repository. Explicit policy selection does not change defaults.
