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

Failed acquisition writes `host-profile.json` as a non-qualified failure record.
It binds bounded manifest/result snapshots and preserves the first failed stage,
return code and partial raw artifacts. Only a completed, nonzero recording with
the specific upstream event-access or page-mapping diagnostics is classified as
a reported permission block. These patterns come from upstream
[event-open diagnostics](https://code.googlesource.com/linux/torvalds/linux/+/473f6c8f437b049f8ec015d57cd59bb983b1d85c/tools/perf/util/evsel.c)
and [recording diagnostics](https://code.googlesource.com/linux/torvalds/linux/+/4edcdefd4083ae04b1a5656f4be6cd83ae919ef4/tools/perf/builtin-record.c).
Generic permission text and policy warnings remain unclassified; interruptions,
timeouts, capture/budget errors and failed startup have separate reasons.
This is stderr attribution, not an observed syscall errno, proof of a particular
sysctl setting, successful recording or profile qualification. No retry or system
configuration change occurs. New denial/checkpoint controls are authored and
remain unexecuted with the source batch.

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
instrumented. The file audit independently observes its event clock and optional
resolution/wall-clock anchor. Cross-clock and selected-operation joins remain
unbound; a single anchor does not establish calibration or an error bound. These source controls are authored
and unexecuted; no new acquisition or attribution result is claimed.

The explicit decoder consumes an existing `perf.data` through Linux perf's
[standalone Python interface](https://github.com/torvalds/linux/blob/master/tools/perf/Documentation/perf-script-python.txt),
using `perf.session` and native sample objects. Upstream has removed the older
embedded callback API; an incompatible or missing native module is an error.
The module must match the selected Python interpreter's ABI. It is an optional
profiling dependency, absent from ordinary RNS8 builds and consumers.
The capability stage checks the exported `sample_event` attribute frontier,
including native mapping and symbol/source accessors, before the workload stage.
A module with only callable `session`/`data` is insufficient. Missing accessors
report unsupported without constructing a native session or sampling workload.

```sh
python tools/host_profile_native.py --input temp/host-profile/perf.data --pid 1234 --output temp/host-profile/native-samples.json
```

Use the benchmark's actual recorded process ID, not the profiler launcher's PID.
Decoding never launches a workload or changes profiler/security settings. It
records native PID/TID, timestamp, CPU-clock period, leaf DSO build ID,
symbol/source information and bounded callchains. CPU-clock periods describe
sampled CPU exposure, not wall latency. Unresolved symbols/source locations stay
missing; callchain paths alone do not establish module build identities.

Leaf module custody is checked separately with a bounded ELF reader over the
actual current file, using the [Linux ELF ABI](https://github.com/torvalds/linux/blob/47f05f71ad988c3180bdba807151e6e86ed75aa3/include/uapi/linux/elf.h).
The reader accepts native little-endian x86-64/AArch64 executable/shared files,
walks program/section notes (including extended section counts), rejects
contradictory GNU build IDs and hashes complete file contents. Limits are128
module identities, 512MiB per file, 2GiB aggregate module-file bytes and bounded
header/note extents. Files that fail format inspection consume the byte budget
and are not repeatedly decoded. Earlier files and native path aliases are
rechecked before publishing the inventory; the acquisition finalizer repeats
the same binding and retains file references through its ordinary bookend.

The report distinguishes matched, mismatched and unresolved leaf DSO identities,
and counts symbolized/source-location samples within matched files separately
from raw native location counts. Empty/missing IDs, deleted files and special
DSOs stay unresolved. A [GNU build ID is not a content checksum](https://sourceware.org/binutils/docs/ld/Options.html#index-_002d_002dbuild_002did),
so it is retained alongside SHA256. Matching the native ID to a current file does
not establish transient loaded-image history or verify the contents of a source
file named by debug information. Callchain modules still have native-path-only
identity. These boundaries remain explicit even when every leaf ID matches.
The ELF/native integration controls are source-authored and unexecuted; no
native profile, source-attributed bottleneck or performance benefit is claimed.

Native sample schema v3 also retains map start/end/file offset. Resolved maps
must contain their sampled IP and fit the64-bit address ABI. ELF executable
load segments provide a separate count of samples whose native map translates
to an executable file extent. That count is distinct from build-ID coverage:
an ID can match while mapping attribution stays unresolved. The native map
observation is not an independent raw-MMAP witness or a source-content proof;
cross-clock joins remain open.

The existing optional Linux ROCTx producer now adds a versioned host-clock
extension to its range records: actual process/start-thread/stop-thread IDs and
`CLOCK_MONOTONIC` timestamps bracketing each native ROCTx start/stop call. It
records a missing timestamp explicitly if the clock query fails. Pending
cross-thread ranges expose no unobserved stop fields; completion publishes them
with the existing atomic outcome. These timestamps observe diagnostic host
scopes, including observer cost, and do not certify GPU completion. The runtime
disabled path takes no new timestamp/thread queries.

`host_profile_session.py --operation-ranges` explicitly activates these existing
scopes for the profiled child of the same ROCTx-enabled production binary. The
baseline remains runtime-disabled. The finalizer first repeats raw recording,
PID, sample and module custody, then joins samples to measured benchmark roots
using their observed `CLOCK_MONOTONIC` brackets. It rejects dropped/unpublished
records, unknown tags, invalid ancestry, child intervals outside their parent,
crossing scopes and conflicting operation identities. Deepest same-thread host
phases receive exclusive sample counts and CPU-clock period weights. Samples
inside an observer bracket stay ambiguous; samples on unbound workers stay
unbound. Cross-thread asynchronous completion ranges are not CPU-work intervals.

The report separates sampled CPU exposure from the union, sum and span of
observed host windows. Neither is exact CPU duration, GPU service or production
latency. No selected samples is inconclusive rather than evidence of zero work.
The default session retains process-total scope and an unbound operation field.
Active Linux compilation, scope/cross-thread controls, pure join controls and
the integrated native acquisition remain deferred to the final validation batch.

Input, sample, frame and decoded-document limits apply. Changes to the input or
native module during decode reject the receipt. Existing outputs are not
overwritten. `recording_sha256` identifies the same raw recording across file
copies/renames; the decoder receipt itself does not authenticate acquisition or
create an independent execution.

The native session interface supplies symbols and DWARF; a bounded
`PERFILE2` audit now supplies recorded event/clock, loss and sample custody.
The v2 decoder and finalizer share that audit. The narrow initial contract is
one user-only software CPU-clock sampling event, optionally accompanied by
explicit software-dummy metadata tracking attributes, in a little-endian regular
file. Attribute order does not select the sampling authority. Mixed attributes
require sample identifiers; loss/read records use their owning event's ID,
trailer layout and counter format. Shared IDs, foreign sampling events and
ambiguous trailers are rejected. Tracking clocks that are missing or differ from
the sampling clock remain an explicit partial-scope blocker.
Compressed/AUX/stat/directory streams, unknown record types and incompatible
event dialects are rejected explicitly. Every section/record extent and the
sample fixed prefix is bounded before native decoding. Per-thread sample
multisets bind PID/TID, timestamps, periods, IP and CPU directly to file bytes
while allowing the native decoder to time-sort buffers. Finalization reconstructs
these facts instead of trusting a resealed sample summary.

Ring-loss events, lost-sample events and cumulative read-counter maxima remain
separate; summing them would double-count overlapping observations. Reported
loss or throttling marks sampling partial. A complete supported record walk
with no known loss is not proof of a completely loss-free acquisition.
The collector explicitly requests CPU, timestamp, period and identifier fields. Event clock
IDs are read from `perf_event_attr`; the header's `CLOCKID` feature supplies
resolution, and optional versioned `CLOCK_DATA` supplies one clock/wall anchor.
The ABI owners are the pinned Linux sources recorded in
`tools/perf_recording_contract.py`. Process totals still do not establish
complete operation coverage, cross-clock attribution, production latency or
performance eligibility. Offline synthetic file/API controls are authored and
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
