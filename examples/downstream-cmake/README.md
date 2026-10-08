# Consume the RNS8 package

This directory is a self-contained C++17 consumer of the installed public API.
It builds a bounded scalar smoke program and a signed/unsigned CPU exact-wide
continuation example. It needs the RNS8 package and a C++ compiler; it uses no
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
The shared option fails clearly if the package has no shared target. Both
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
