# RX 7900 XTX expanded validation — 2026-10-23

The complete local optional-backend build at `c7f7e05` registered 471 tests:
**468 passed, three target-inapplicable skips, zero failures**. The full repair
pass took 158.83 seconds with `OMP_NUM_THREADS=3` and serial CTest scheduling.

| Configuration | Evidence |
|---|---|
| VS2026/MSVC 19.51.36260, HIP 7.1.51803, gfx1100, RelWithDebInfo | Full all-feature build and CTest pass |
| Direct HIP, vector ALU, byte-limb wrap64 | Registered arithmetic, ownership, native, finite, exact-wide and export differentials |
| hipBLASLt, CK, rocWMMA, AMDGPU builtins | Explicitly enabled; backend differentials, cache tests and applicable ISA gates |
| Generated native WMMA and experimental Garner table | Both enabled in this build; generated-variant and reconstruction checks |
| OpenMP | Enabled, three threads during validation |
| GMP and FLINT | Enabled; optional exact-reference link/execution smoke passed |
| Public workflows | Prepared submissions, resident DAGs, packing/reuse, examples, benchmark/schema tests and installed package consumers |
| Sparse AMDGPU matrix instructions | Three explicit sparse-A bounded/exact-wide/finite tests skipped on this target; dense and other supported sparse paths still tested |

The separate rocWMMA/OpenMP configuration at `8c79b2c` passed 431/431 tests.
LLVM 22 CPU ASan at that revision registered 303 tests: 297 passed and six
HIP-only tests skipped, with zero failures. Those earlier runs precede the
hipBLASLt repair below; the all-feature repair pass is the latest GPU evidence.

GitHub CI at `c7f7e05` also passed all 18 push/PR check runs: the Ubuntu CPU job
and the GCC/Clang × OpenMP on/off × shared/static CPU matrix. These are Linux CPU
results; the Linux GPU qualification boundary is unchanged.

## Failure found and repaired

The first all-feature pass at `8c79b2c` found two hipBLASLt failures. Its operand
prepack cache compared an engine-owned generation without the matrix instance
ID. Distinct generated matrices both at generation 1 could reuse the wrong
packed operand and produce incorrect exact-wide chains. A second test still
expected caller source labels in metadata that now represents engine generations.

`fb9f94f` binds both operand caches to stable matrix identity and generation,
rejects cache hits for zero/exhausted generations, and updates the private
metadata names and assertions. Regressions cover signed/unsigned chains,
same-generation B-handle swaps, retained allocations and generation exhaustion.
`c7f7e05` ensures the exhaustion fixture performs a fresh import before testing
publication. The complete suite was then retried once, with the result above.

## Reproduction

The `build/issues-all` configuration enables `RNS8_ENABLE_HIP`,
`RNS8_ENABLE_CPU_PARALLEL`, `RNS8_ENABLE_HIPBLASLT`, `RNS8_ENABLE_CK`,
`RNS8_ENABLE_ROCWMMA`, `RNS8_ENABLE_AMDGPU_BUILTINS`, `RNS8_ENABLE_GMP`,
`RNS8_ENABLE_FLINT`, `RNS8_EXPERIMENTAL_GENERATED_NATIVE_WMMA`,
`RNS8_EXPERIMENTAL_GARNER_TABLE` and
`RNS8_ENABLE_ROCWMMA_WRAP64_CANDIDATE_TESTS`. The explicit target is `gfx1100`,
the toolchain is `C:/vcpkg/scripts/buildsystems/vcpkg.cmake`, and HIP, CK and
rocWMMA roots point to the installed SDK and existing pinned dependency trees.

```powershell
python tools/windows_dev.py cmake --build build/issues-all -j 8
$env:OMP_NUM_THREADS = '3'
python tools/windows_dev.py ctest --test-dir build/issues-all --output-on-failure -j 1
```

Raw configure/build/test logs and the CTest inventory are retained in ignored
`temp/all-features-*` files. Hardware-dependent Linux multi-GPU, Instinct/CDNA,
floating low-precision matrix and CPU matrix-extension qualification remains
deferred because no qualification host is available. This record supersedes
older local pending-runtime statements for the registered tests exercised here;
it does not complete the remaining architecture implementations or promotion gates.
