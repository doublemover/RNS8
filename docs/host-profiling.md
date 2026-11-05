# Host CPU and allocation diagnostics

Host profiles explain a completed exact call; they do not replace its
uninstrumented end-to-end measurements. Acquisition, native decoding, operation
range selection, symbol/module association, loss assessment and production
comparison are distinct evidence steps. The implementation frontload is in
progress under [#105](https://github.com/doublemover/RNS8/issues/105).

## Native Linux CPU sample decoding

The plan-only acquisition adapter reuses the existing bounded session runner
and shared production-input admission. It requires a completed hook-disabled
production observation, stable static executable identity and matching native
ELF64 benchmark/perf/Python tools. It queries perf version, the selected
interpreter's standalone native-module capability and compiled benchmark
identity before process-scoped sampling. A permission/capability failure never
lowers system security or enables an automatic fallback.

```sh
python tools/host_profile_session.py --production temp/production.json --out temp/host-profile/session
```

Dependency discovery finds perf on PATH. An explicit `--perf` can select the
actual native executable when the distribution exposes a launcher script;
unsealed launcher scripts are rejected. Inspect the emitted commands, then
explicitly add `--execute` to start acquisition. The maintained recipe requests
software `cpu-clock:u` samples at 99 Hz, bounded DWARF callchains,
`CLOCK_MONOTONIC` and a 64 MiB record limit. Actual CPU exposure comes from native sample periods;
requested frequency is not proof of achieved sample rate. Record/runtime and
decoded-artifact limits remain independent.

The adapter disables persistent build-ID cache writes and debuginfod downloads,
isolates perf configuration, and retains partial data/logs on failure. The
benchmark's native argv/environment witness includes `PERF_` variables. Artifact
monitoring continues over the previously created capture directory during
decode; it does not create or overlay a second recording. Finalization binds
the benchmark's observed PID separately from perf's launcher PID, verifies
numerical output, artifact/module identities, physical GPU and matching
work/proof/output/reuse contract, and labels the bound capture externally
instrumented. Its clock is an explicit acquisition request, with header/cross-clock
and selected-operation joins still unbound. These source controls are authored
and unexecuted; no new acquisition or attribution result is claimed.

The explicit decoder consumes an existing `perf.data` through Linux perf's
[standalone Python interface](https://github.com/torvalds/linux/blob/master/tools/perf/Documentation/perf-script-python.txt),
using `perf.session` and native sample objects. Upstream has removed the older
embedded callback API; an incompatible or missing native module is an error.
The module must match the selected Python interpreter's ABI. It is an optional
profiling dependency, absent from ordinary RNS8 builds and consumers.

```sh
python tools/host_profile_native.py --input temp/host-profile/perf.data --pid 1234 --output temp/host-profile/native-samples.json
```

Use the benchmark's actual recorded process ID, not the profiler launcher's PID.
Decoding never launches a workload or changes profiler/security settings. It
records native PID/TID, timestamp, CPU-clock period, leaf DSO build ID,
symbol/source information and bounded callchains. CPU-clock periods describe
sampled CPU exposure, not wall latency. Unresolved symbols/source locations stay
missing; callchain paths alone do not establish module build identities.

Input, sample, frame and decoded-document limits apply. Changes to the input or
native module during decode reject the receipt. Existing outputs are not
overwritten. `recording_sha256` identifies the same raw recording across file
copies/renames; the decoder receipt itself does not authenticate acquisition or
create an independent execution.

The current native session interface does not expose loss counts. The decoder
therefore records unknown loss and process-total scope, with no claim of
complete operation coverage, production latency or performance eligibility.
Acquisition/clock/range binding and a separate native loss assessment remain
required for stronger attribution. Offline synthetic controls are authored and
registered but have not run; native Linux decoding remains unexecuted locally.

## Windows acquisition boundary

The maintained source workflow will use uniquely named WPR instances and a
reviewed WPA export profile. Microsoft requires `-instancename` last on all
commands affecting that instance; an unnamed stop/cancel must never affect
another recording. [WPR command reference](https://learn.microsoft.com/en-us/windows-hardware/test/wpt/wpr-command-line-options)
and [WPA exporter reference](https://learn.microsoft.com/en-us/windows-hardware/test/wpt/exporter)
own the native command contracts. Windows acquisition/export, allocator and wait
adapters, range/cross-clock joins and the source-linked critical-path report
remain implementation work. No WPR recording, elevation, persistent heap
configuration or system profiler has been started by the source frontload.
