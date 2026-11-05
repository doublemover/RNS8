# Host dynamic checks and source coverage

Host instrumentation is a separate build identity. It is neither ordinary
production performance nor HIP device instrumentation. The current source
frontload adds build selection and sensitivity/concurrency controls under
[#102](https://github.com/doublemover/RNS8/issues/102) and coverage instrumentation
under [#103](https://github.com/doublemover/RNS8/issues/103). Lane drivers,
artifact/inventory reports and CI integration remain implementation work.

| Configure preset | Instrumentation | Initial scope |
|---|---|---|
| `cpu-asan-ubsan-debug` | Nonrecovering address + undefined checks | Native Linux Clang, OpenMP OFF |
| `cpu-asan-ubsan-release` | Same checks with RelWithDebInfo | Separate optimized build |
| `cpu-tsan-release` | ThreadSanitizer | Native Linux Clang, OpenMP OFF |
| `cpu-coverage-debug` | LLVM source profiles and coverage mapping | Separate host CPU objects |
| `windows-clang-asan-debug` | Existing clang-cl ASan/runtime setup | Windows host checks; no UBSan/TSan/device claim |

`RNS8_HOST_SANITIZER` selects `none`, `address-undefined` or `thread`.
`RNS8_ENABLE_HOST_COVERAGE` selects source coverage. The Linux lanes require
Clang and reject HIP or a mixed sanitizer/coverage build; TSan rejects the
ASan/libFuzzer object lane. Global compile/link options instrument first-party
targets, while installed library interfaces carry matching consumer flags.
Existing Windows clang-cl runtime/STL handling remains its own implementation.
The compiled manifest records selectors and instrumentation flags. Production
timing admission and profiler-baseline admission reject these builds, including
recognized sanitizer/profile flags in older configuration records.

`RNS8_BUILD_SANITIZER_CONTROLS` builds `rns8-host-sanitizer-control` only for a
selected detector lane. Its capability query reports actual compiler address
and thread features separately from the requested lane. Safe and deliberately
bad modes are not registered as ordinary passing CTest cases. The bad modes
are disposable host-only heap-overread, signed-overflow and race controls; an
undetected zero exit, unsupported return, timeout, crash without the intended
diagnostic or missing runtime must not become a detector PASS. No such program
has been executed in the current source pass, and no GPU fault was introduced.

The normal `[cpu][concurrency]` corpus uses independent public contexts, plans,
workspaces and outputs from four host threads, verifies complete Boost values,
new publication generations, unique matrix instances and cleanup, and checks
shared handles only under a real external mutex. The disposable race control
and these legal calls share a deterministic start gate; sleeps do not define
interleavings. OpenMP sensitivity and synchronization controls are authored
separately; their presence does not qualify an uninstrumented OpenMP runtime.

Clang documents that ASan/UBSan require appropriate final executable linking,
and that TSan uses PIE and does not support statically linked libc/libstdc++.
An RNS8 static library with the dynamic host runtime is a distinct contract.
[ASan usage](https://clang.llvm.org/docs/AddressSanitizer.html#usage),
[UBSan usage](https://clang.llvm.org/docs/UndefinedBehaviorSanitizer.html#usage)
and [TSan limitations](https://clang.llvm.org/docs/ThreadSanitizer.html#limitations)
own those compiler/runtime requirements. Matching llvm-profdata/llvm-cov,
instrumented-library/test inventories, raw artifact identities and native
detector observations still need the maintained reporting driver.

`tools/execution_inventory.py` reconciles retained native CTest JSON-v1 full
discovery and selected discovery with its JUnit report. It requires explicitly
named mandatory passing tests. Missing/extra cases, changed argv/properties,
empty execution, unapproved disabled/skipped cases and application failures
block acceptance. Optional skips require exact test names and reviewed reasons;
startup, fixture and resource failures cannot be approved as ordinary skips.
Disabled and skipped cases remain outside the executed denominator. The report
binds immutable input digests and writes only a fresh child of ignored `temp/`:

```sh
python tools/execution_inventory.py --discovered temp/lane/discovered.json \
  --selected temp/lane/selected.json --junit temp/lane/tests.xml \
  --required 'mandatory test name' --output temp/lane/inventory.json
```

This reader follows the selected CMake producer's
[JSON property encoding](https://github.com/Kitware/CMake/blob/v4.4.5/Source/CTest/cmCTestMultiProcessHandler.cxx)
and [JUnit statuses](https://github.com/Kitware/CMake/blob/v4.4.5/Source/CTest/cmCTestTestHandler.cxx).
The output qualifies inventory reconciliation only; it does not prove that
objects were instrumented, detectors worked or GPU memory was checked.

Windows gfx1100 device ASan remains unsupported/unqualified. Native Linux
ROCm device ASan needs its own release/target/HMM/XNACK/runtime and actual
device-object instrumentation proof. Host compiler flags and canary tests
cannot supply it. Existing stopped hardware fault campaigns remain stopped.
