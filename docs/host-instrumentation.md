# Host dynamic checks and source coverage

Host instrumentation is a separate build identity. It is neither ordinary
production performance nor HIP device instrumentation. The current source
frontload adds build selection and sensitivity/concurrency controls under
[#102](https://github.com/doublemover/RNS8/issues/102) and coverage instrumentation
under [#103](https://github.com/doublemover/RNS8/issues/103). The native host
session, native LLVM coverage and separate device-ASan preflight drivers are
authored. Native test-process and installed-consumer loader bookends are also
authored; additional launcher/application coverage, device correctness session
integration and continuous CI reporting remain implementation work. Python
branch/subprocess collection source is authored.

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
own those compiler/runtime requirements. The separate authored
[native LLVM coverage lane](host-source-coverage.md) binds its matching tools,
profiles and source denominator. The separate authored
[Python branch collector](python-source-coverage.md) observes selected production
modules and real test contexts. The authored
[Python production mutation lane](production-mutations.md) retains separate
baselines and intended native assertion/source-line observations. The authored
[native C++ mutation lane](native-production-mutations.md) builds separate CPU
production variants with ordinary assertion controls. All new lane execution,
compatible optional Mull and continuous CI remain pending.

`tools/host_instrumentation_session.py` prints a plan by default. Adding
`--execute` configures a fresh build, builds its complete first-party targets,
inspects configured argv and actual object/archive symbols, queries executable
and linked-library identities, runs sensitivity controls, runs the complete
registered test inventory serially, and installs/builds/tests all seven packaged
CPU consumers. It uses the shared bounded process capture and retains the first
failure without automatic retry. Configure/build/test commands, native outputs,
sources, objects, tools, matching compiler runtime files, binaries and installed
static library are digest-bound and checked again at completion.

```sh
python tools/host_instrumentation_session.py --preset cpu-tsan-release \
  --out temp/host-tsan-candidate --cmake /path/to/cmake --ctest /path/to/ctest \
  --compiler /path/to/clang++ --nm /path/to/llvm-nm \
  --symbolizer /path/to/llvm-symbolizer
```

Use `cpu-asan-ubsan-debug`, `cpu-asan-ubsan-release` or
`windows-clang-asan-debug` for the other lanes. Windows also requires an explicit
`--toolchain` pointing to the vcpkg toolchain and its VS developer environment.
An explicit `--compiler` overrides preset discovery. Compiler, llvm-nm and
llvm-symbolizer must report the same complete LLVM version. The Windows lane
retains static CRT/STL/runtime wiring and binds the copied ASan DLL; PE binary
hashes and sensitivity controls are separate from archive symbol inspection.
The default lanes disable HIP, OpenMP, shared libraries and the separately
owned libFuzzer corpus; they do not qualify those disabled features.

For a separate native Linux TSan/OpenMP candidate, add
`--openmp-archer /absolute/path/to/libarcher.so` to the `cpu-tsan-release`
command. This requires actual CMake OpenMP discovery and LLVM libomp, an
exported OMPT entrypoint in the selected Archer, and two observed OpenMP
workers. The driver fixes the OpenMP/Archer environment, binds the configured
libomp to the library supplying `omp_get_num_threads`, requires the selected
Archer in the native loader snapshot and checks its positive startup message.
It runs both the real OpenMP race and synchronized controls before application
tests. No broad runtime suppression or `ignore_noninstrumented_modules` option
is added. A runtime false positive is a blocker for review, not an automatic
suppression. Archer supplies synchronization annotations; the receipt does not
call this full instrumentation of libomp's own memory accesses. LLVM's
[selected Archer implementation and usage](https://github.com/llvm/llvm-project/tree/llvmorg-23.1.5/openmp/tools/archer)
describe the OMPT/TSan relationship. No OpenMP control has run in this pass.

`--allowed-skips` accepts a JSON object mapping exact selected test names to
reviewed reasons. There is no blanket HIP skip or historical-count allowance.
Both named legal-concurrency tests and every installed consumer must pass.
Expected bad controls require the intended diagnostic, configured nonzero exit,
exact marked source line in the top access frame (or the UBSan source
diagnostic) and native process identity where the runtime emits one; a
timeout, generic crash, unsupported control or missed defect cannot pass.
Object symbols establish observed instrumentation sites, not coverage of every
memory access. A separate control query now observes actual loaded modules
through native Windows Toolhelp or Linux `dl_iterate_phdr`, with lazy OpenMP
initialization completed before enumeration. The bounded inventory requires
the captured native PID, executable, exact Windows ASan DLL or Linux sanitizer
symbol owner, immutable module file identities and explicit virtual images.
This snapshot belongs to the detector query process. It is separate from the
available compiler-runtime inventory and does not prove every library loaded
later by every application/consumer process or instrumentation of external
libraries. A failed sensitivity control stops before application tests. No host
session, loader query, sensitivity control or test has executed during this
implementation frontload.

The host sanitizer session also requests two bounded witnesses from each real
`rns8_tests` invocation and every packaged CPU consumer. A fresh 256-bit stage
token is supplied only to CTest execution, never discovery. Native Toolhelp or
`getppid` parent identity binds each child to that captured CTest process;
executable path, exact arguments, address/thread compiler features and ordered
entry/exit module snapshots must agree with the discovered case. Missing,
truncated, duplicated or contradictory bookends block the lane. A skip can
retain observed bookends but receives no passing loader credit. Other CTest
commands retain their inventory dispositions and are explicitly outside this
native witness denominator. The launcher PID is never substituted for a child.

Witnesses compile only into the selected sanitizer test configuration.
Installed consumers opt in with
`RNS8_DOWNSTREAM_RUNTIME_WITNESS_ROOT`, pointing to the recorded checkout and
its private witness sources; the driver supplies the configured JSON dependency.
They still link the exact installed instrumented library. Ordinary installed
consumer builds require neither that source checkout nor the witness dependency.
Snapshots are emitted before test work and after its scope, outside injected
allocation-failure windows. Failed emission forces failure rather than a clean
exit without evidence.

The reader hashes each distinct observed module file once per stage and
rechecks those identities at final publication. This binds files available
after execution; it is not an in-memory image attestation or a history of
modules loaded and unloaded between bookends. An application's snapshot does
not force unrelated OpenMP work merely to initialize Archer. It records whether
the selected annotation adapter was actually present; the separate positive
OpenMP controls remain required and external runtime access instrumentation
remains unproven. None of these observations provide production timing credit.
The reader uses CTest's per-case `system-out`, whose producer is explicit in the
[selected CMake source](https://github.com/Kitware/CMake/blob/v4.4.5/Source/CTest/cmCTestTestHandler.cxx#L2558).
Windows parent identity follows the documented
[`PROCESSENTRY32W` field](https://learn.microsoft.com/en-us/windows/win32/api/tlhelp32/ns-tlhelp32-processentry32w).
These witness paths and protocol controls have not executed in this source pass.

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
device-object instrumentation proof. The separate authored
[native ROCm preflight](device-asan.md) can inspect prerequisites and build real
first-party HIP objects without launching workloads or faults. Host compiler flags and canary tests
cannot supply it. Existing stopped hardware fault campaigns remain stopped.
