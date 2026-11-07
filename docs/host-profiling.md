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
contradictory GNU build IDs and hashes complete file contents. Limits are 128
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
not establish transient loaded-image history or the source bytes used at
compilation. Separate recorded-mapping and current-source checks below extend
attribution without borrowing those claims from leaf build-ID coverage.
The ELF/native integration controls are source-authored and unexecuted; no
native profile, source-attributed bottleneck or performance benefit is claimed.

Native sample schema v4 retains map start/end/file offset. Resolved maps
must contain their sampled IP and fit the64-bit address ABI. ELF executable
load segments provide a separate count of samples whose native map translates
to an executable file extent. That count is distinct from build-ID coverage:
an ID can match while mapping attribution stays unresolved. The native map
observation is checked separately against raw-MMAP evidence where available;
cross-clock joins remain open.

The recording audit now preserves bounded native MMAP/MMAP2 and process
address-space boundaries. A separate indexed join resolves leaf and callchain
PID/IP/time observations only against unique recorded user mappings with an
earlier timestamp in the same clock domain, executable protection and an actual
recorded MMAP2 build ID. The acquisition recipe explicitly requests the upstream
[build-ID mapping mode](https://github.com/torvalds/linux/blob/473f6c8f437b049f8ec015d57cd59bb983b1d85c/tools/perf/Documentation/perf-record.txt).
Matching current ELF build ID and executable file extent are separate checks.
Native names must agree with the address-selected map; a basename alone never
selects an identity. Wrong processes, untimed or foreign-clock maps, inherited
maps without their own records, missing IDs and address reuse remain unresolved.
Any overlapping address reuse in the recording is conservatively ambiguous,
even if a more complete temporal map history could resolve it. Lookup uses a
sweep-built interval index and binary search; limits are8,192 mappings,4MiB of
mapping records and500,000 leaf/caller observations. Raw unwind stacks are not
independently replayed and complete transient loaded-image history is not claimed.

The native producer also resolves source tuples for caller addresses through
the upstream optional-address `srccode` interface. Bounded current first-party
files are hashed once, requested lines are compared with the native text, and
paths and bytes are rechecked before publication. Missing/relative/external
locations, line mismatches and unavailable text retain distinct states. Limits
are256 files,8MiB per file and64MiB total; files outside the repository are not
opened by this check. These are current source snapshots and native debug-line
observations, not compilation source checksums. Both mapping/ELF and source
inventories are independently reconstructed by acquisition finalization, so
editing and resealing their counts cannot supply missing evidence. Their binary,
source, substitution and budget controls are authored; runtime execution remains
deferred with the final batch.

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

`tools/pe_symbol_identity.py` supplies bounded local symbol custody for the
Windows adapter. It reads actual PE32+ RSDS and MSF7 PDB info records, requires
matching GUID and age, retains complete file hashes and repeats content checks.
Section/RVA/raw-file disagreement, contradictory CodeView records, unsupported
legacy formats and missing info streams are errors. MSF directory/stream blocks
must be bounded, disjoint, outside reserved pages and allocated in the active
free-page map. Only documented 512/1,024/2,048/4,096-byte MSF blocks and the
VC70 info layout are admitted; each file is limited to 512 MiB and directories
to 8 MiB. GUID/age agreement identifies the selected symbol file; it does not
establish usable function/source coverage or compilation-source checksums.
The authored byte-layout, mismatch, extent, alias, format and hash controls remain
unexecuted. Native Windows symbolization remains a separate qualification step.

`tools/host_profile_windows.py` now authors the bounded acquisition/export path:

```powershell
python tools/host_profile_windows.py --production temp/production.json --pdb build/Release/rns8-bench.pdb --out temp/host-profile/windows
```

Plan-only reads source/artifact/symbol identities without launching a tool.
Dependency discovery locates optional `wpr` and `wpaexporter` through PATH or
the standard Windows Performance Toolkit installation. Explicit `--wpr` and
`--wpaexporter` paths override discovery. The tools and benchmark require the
same native PE architecture. No tool is installed and no GUI is launched.

The Windows tools build also supplies `rns8-etl-inspect`. Build that target with
the selected Windows SDK and pass `--etl-inspector` when it is outside the
benchmark directory. The plan binds the native inspector's file and architecture;
its identity query must report the exact compiled source hash before recording.
CMake regenerates that definition when the reader changes. No current source
hash is assigned to a stale reader binary merely because their paths match.

Recording additionally requires both `--execute` and
`--authorize-system-recording`: the short ETW window includes system CPU
samples even though decoding selects the benchmark PID. No elevation is
launched. The maintained WPR profile requests sampled stacks, process/thread
and image records, with 128 KiB buffers, 64 buffers and a 128 MiB sequential trace
cap. It does not change the global sampling interval, heap settings, registry,
driver, service startup or security policy. Native help, profile listing and
the actual benchmark build-identity query precede recording.

Every run has a fresh random WPR instance name. The native absence HRESULT is
required before arming ownership; an access error or partial query cannot stand
in for absence. Microsoft requires `-instancename` last on all commands affecting
that instance. Start, status, stop and cancellation retain that suffix. Failed
or interrupted workload/start attempts still stop the owned instance; a failed
stop attempts only its named cancellation. Cleanup retains a separate time
reserve and can run after the artifact monitor's budget is exhausted. It never
uses an unnamed stop/cancel or alters another recording. Partial files and the
first failed native stage remain in the failure receipt; unconfirmed cleanup is
reported explicitly. Native command contracts follow the
[WPR reference](https://learn.microsoft.com/en-us/windows-hardware/test/wpt/wpr-command-line-options)
and [WPA exporter reference](https://learn.microsoft.com/en-us/windows-hardware/test/wpt/exporter).

The maintained WPA candidate uses CPU sampling table/column GUIDs inspected in
the installed Toolkit's `Catalog/HtmlResponsivenessAnalysis.wpaprofile`.
Local symbol search is explicit; no symbol server is configured. The decoder
inventories exports through the shared link-free helper with the actual session
root, confining its input to the owned export subtree. It
accepts one English CSV with explicit seconds/milliseconds column units,
individual sample counts, exact selected PID/image/thread and bounded rows,
fields and bytes. It preserves foreign/idle row counts and native stack/module/
function strings. Different headers, units, locale or aggregated table layouts
are unsupported until reviewed; no conversion is guessed. Shared Linux/Windows
admission preserves the full numerical, physical-device, work/proof/output/reuse,
native argv and relevant environment relation to production.

After owned stop, the inspector opens only the finalized ETL file through native
`OpenTrace`/`ProcessTrace` and uses TDH's actual property names/types for classic
process records. It never opens a live logger or controls a recording. The
native [log header](https://learn.microsoft.com/en-us/windows/win32/api/evntrace/ns-evntrace-trace_logfile_header)
supplies recorded clock/frequency and separate lost-event/lost-buffer counters;
these overlapping observations are not summed. Native process records retain
key, PID, parent, Start/End/rundown opcode and converted100ns FILETIME. A complete
observed key needs one actual Start/End pair, the recording owner's parent PID
and the benchmark's ASCII basename. Rundown alone cannot replace a lifetime.
Missing/undecodable records, PID reuse, unknown image encoding and reported loss
remain blockers. Zero header loss is not proof of loss-free acquisition.
The reader bounds file, event, selected-record and TDH schema storage; failed
schema lookups are cached. Exhausted event, selected-record or schema budgets
stop native decoding at the next buffer boundary and prevent completed metadata;
they cannot be downgraded to an unsupported-format count. The budget exception
does not allocate. Genuine schema-format failures retain their separate partial
classification. It emits no command lines or SIDs. Decoder/source/PID
and ETL content bookends are checked before retaining its metadata alongside
the WPA export. The late-bound inspector argument is the exact completed
benchmark PID, not the profiler launcher's or a selected display name.

The Windows result still remains partial: actually loaded PDB identity,
measured-operation selection and API/GPU clock alignment are not established by
the current tables. Current PE/PDB agreement cannot supply those observations.
Native ETL/WPR/WPA compatibility and symbol attribution plus all authored
ownership/cleanup/producer/unit/lifetime controls remain unexecuted. The native
reader passes a Windows C++17 frontend check, and WPR XML passes the installed
SDK's schema, as source checks only. Allocation, scheduler/wait, loaded-symbol
and range/cross-clock adapters and the critical-path report remain implementation
work. No ETL decoder, WPR recording, elevation or system profiler has run during
the source frontload.

## Separate Linux host allocation capture

`tools/host_heap_session.py` authors a bounded Linux recipe against an explicit
native `libheaptrack_preload.so`. Its plan reads the existing admissible
production observation, same optimized benchmark, ELF tuple and source/library
hashes without launching or installing anything:

```sh
python tools/host_heap_session.py --production temp/production.json \
  --preload /path/to/libheaptrack_preload.so --out temp/host-heap-candidate
```

`--execute` is reserved for the validation/diagnostic batch. The adapter launches
the benchmark directly, first checking its compiled identity without injection,
then supplying the explicit preload and owned raw output. This follows the
upstream [preload producer](https://github.com/KDE/heaptrack/blob/v1.5.0/src/track/heaptrack_preload.cpp)
without a shell wrapper, compressor, GUI, process attachment or persistent
settings. The upstream library removes both injection variables before main;
the actual native argv/environment, process PID, exact output and common
work/proof/device/reuse contract must still match. Inherited heaptrack controls
are cleared, and the production baseline cannot carry interception controls.
The native environment witness and Python request policy share those key names.
Missing output, failed process, changed library/source/capture or wrong native
PID cannot publish a bound allocation result. Failure receipts preserve owned
logs and partial files; elapsed time, output and artifacts are bounded.

`tools/heaptrack_native.py` reads only the reviewed1.5.0 raw-format3
[producer records](https://github.com/KDE/heaptrack/blob/v1.5.0/src/track/libheaptrack.cpp).
It retains allocation pointer generations, parent trace nodes and module-cache
generations at allocation as well as trace-node creation. A reused trace ID
after a module reset cannot inherit its original image mapping. Matched frees
recover requested size from their live allocation.
Duplicate live pointers, future trace parents, mixed processes, unknown record
formats, reversed clocks, truncated strings and exhausted budgets fail closed.
Unmatched frees retain unknown pre-capture lifetime instead of subtracting zero.
Unmatched frees and events after the last clock bracket are named partial
observations. The native format has no complete-finalization marker; a valid
record walk cannot prove complete allocator interception. Total retained module
segments have their own budget in addition to file/record/trace/module limits.
The summary includes count/size/trace distributions, observed live high water,
retained requested payload and adjacent allocation/free pairs. Native free/new
pairs do not identify a realloc call, so the realloc count stays unavailable.

Timestamp records bracket groups of events at floored milliseconds. Allocation
lifetime bounds include that uncertainty; no per-event nanosecond timestamp or
operation/thread clock join is invented. RSS pages use the recorded page size
and remain sampled RSS, separate from tracked payload and its observed peak.
Neither quantity is an exact whole-process transient peak or leak proof. Device
and pinned storage stay unobserved by this host-heap lane. Raw addresses and
module records alone do not establish loaded symbols or source ownership.
Operation/phase association, symbol attribution, complete interception and
separate Windows heap/scheduler integration remain source work. The native
format controls and direct-child producer/consumer controls are authored and
registered; no decoder, preload, fixture, benchmark or profiler has run in this
source pass.
