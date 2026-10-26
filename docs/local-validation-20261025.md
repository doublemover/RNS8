# Windows and gfx1100 validation — 2026-10-25

The [native moment and completion-boundary validation](#native-moment-and-completion-boundary-validation)
records the latest `763a948` complete rebuild and test batch. Earlier tables retain
their original artifact boundaries; correctness validation alone qualifies no timing gain.

This record covers the consolidated correctness and tooling pass after the
frontloaded implementation on PR #15. The host is Windows with an RX 7900 XTX
(`gfx1100`), VS2026 18.11.1 / MSVC 19.51, HIP 7.1, Python 3.15.1,
LLVM 23.1.4, CMake 4.4.4 and Ninja 1.13.2. CPU sanitizer binaries use actual
AddressSanitizer instrumentation and the static MSVC runtime. Optional GPU
backends are hipBLASLt, CK, rocWMMA and AMDGPU builtins; GMP, FLINT and OpenMP
are enabled in the full-feature GPU build.

The [machine-readable receipt](../metadata/local-validation-20261025.json)
records source boundaries, configure arguments, artifact hashes and individual
batch results. Raw logs and XML remain under ignored `temp/validation-20261025/`.
The stopped performance campaign was not resumed.

## Results and coverage

| Gate | Observed result |
| --- | --- |
| Clean CPU ASan and full-feature HIP builds at `8f7ba18` | Both passed |
| Repaired full-feature HIP build at `9b2e3c5` | Passed |
| CPU ASan CTest batch, shared logical proof excluded | 528 passed, 22 explicit skips, 0 failures |
| Full-feature GPU CTest batch, shared logical proof excluded | 891 passed, 17 skips, 0 failures; two stream-mode skips repaired below |
| Packed-dot/Garner-table GPU configuration | 652 test cases passed, 5 explicit skips, 0 failures |
| Scalar-control GPU configuration | 652 test cases passed, 5 explicit skips, 0 failures |
| Corrected signed/unsigned host-transfer modes at `ef2dabd` | 8/8 passed, 0 skips; pageable sync, pinned sync, pinned stream and pinned graph |
| Compiled host-transfer CLI/collector contracts | Passed |
| Standalone tooling entrypoints | 28/28 passed |
| LLVM libFuzzer smoke | 3/3 passed, 1,000 runs each, seed 3135, per-input timeout 10 seconds |
| Installed ordinary static/shared consumers and CPU/HIP workloads | Passed in the applicable CTest batches |
| Installed native-device consumer, static and shared | 4/4 passed: synchronous and prepared graph execution for each linkage |
| Registered CPU and HIP verifiers at `5456408` | Both passed; CPU verifier also passed under ASan |
| Real-object ISA CLI | Default object-only, explicit backend and build-tree discovery passed; input hashes unchanged |
| Complete shared native-WMMA logical-proof regression at `ef2dabd` | Passed in 1020.41 seconds, within the unchanged 1500-second limit |

A final reconciliation by test name covers every entry in the refreshed
inventories: **CPU 530 passed / 22 skipped** and **GPU 896 passed / 15 skipped**,
with zero failures or missing tests. These are unions of the named batch and
repair partitions above, including the new verifier registrations and one shared
logical-proof execution, rather than totals from a single invocation.

The long Python logical-proof regression is shared by CPU and GPU inventories.
It runs once against its exact source, rather than repeating the identical
host-only proof for each backend configuration. The full batch counts above
exclude that one case. Catch2 JUnit's `tests` attribute counts assertions;
the two control-configuration rows count actual `testcase` elements instead.

All original CPU skips are HIP/platform-dependent cases. After the two repaired
stream smokes, the remaining full-feature GPU skips are three unsupported
sparse-A builtin cases, three CDNA instruction cases and nine CDNA FP64 MFMA
smokes. Each remains unqualified on this RDNA3 device.

The GPU fixtures cover full native signed/unsigned boundaries, default prefixes
1–20, K-block splits around 65536, padding and shape tails, custom ordered bases,
CRT/base extension, persistent and grouped queues, prepared/event/graph
publication, native and pinned transfers, serial/overlapped panel streaming,
and the enabled optional backends. Benchmark smoke cases verify their declared
reference/output contracts; they do not constitute complete calibration grids.

## Failures repaired

| Repair | Failure and correction |
| --- | --- |
| `9b2e3c5` | The paired-column WMMA kernel tested a macro already undefined by an earlier header, so its body compiled inactive. Use the same explicit target/builtin guard as adjacent tiled kernels. Refresh seven stale derived documentation inventory entries. |
| `6ba0b6e` | LLVM objcopy's single-file dump-section invocation rewrote the inspected COFF object. Supply a distinct scratch output, reject aliases and retain immutable input hashes. |
| `6f1aae3` | Plain pinned-stream calibration omitted completion-resource preparation. Prepare all four owners during setup, and reserve unsupported skips for initial context availability. |
| `ef2dabd` | The logical checker timed out after 1500.02 seconds because each ancestor reinterpreted already checked expression subtrees. Reuse checked child values while retaining every intermediate overflow check, physical fragment and original configuration/budget. Compare the new interpreter with the legacy traversal, including masked overflow and division by zero. |
| `5456408` | The standalone HIP verifier copied into a deliberately absent matrix host mirror. Use independent, extent-checked readback buffers, assert that the mirror remains absent, and register both verifier modes with CTest. |

The first GPU arithmetic failure stopped that lane. Independent CPU checks
continued. Repairs were committed and pushed before their grouped retests.
Subsequent corrections affect tooling/calibration/verifier code; the passing
main GPU arithmetic binary remains the `9b2e3c5` artifact. CPU arithmetic
binaries remain at `8f7ba18`. The transfer driver and verifier have their own
later build identities, recorded separately rather than attributed to checkout
HEAD.

## Issue closeout and remaining gates

Closed with specific validation comments: **#27, #72, #74, #75, #76 and #77**.
These cover independent observation admission, retired placeholder accounting,
cache path parity, resumed capture identity, object-only ISA reporting and the
non-promoting workflow/CLI contract.

Other research and qualification issues remain open. In particular:

- #42 still needs general packed/materialization representation selection.
- #45, #50 and #52 retain the hardware-dependent work explicitly deferred by
  the user. Broader Linux/RDNA/CDNA/CPU matrix-instruction validation is separate.
- #60's lock consistency check passes, but the configured doctor still observes
  Git 2.52.0.windows.1 against the selected 2.56.0.windows.3 pin. The requested
  ROCm 10.1 transition remains deferred.
- #61's local documentation checks pass. The official external-link check reached
  16 URLs; 28 AMD documentation URLs returned HTTP 429 and remain unresolved.
- Matched independent performance observations, instruction/counter evidence,
  held-out planner calibration, large-address device qualification and real
  device-fault injection retain their individual acceptance gates.

Reviewed performance results remain in the [stopped October 24 campaign](7900xtx-partial-campaign-2026-10-24.md).

## Reproduction

Use the canonical [Windows setup](platform-windows.md) with the exact versions
and explicit root/feature overrides in the receipt's build plan. Configure and
build the selected source boundary before running its matching inventory:

```powershell
python tools/windows_dev.py ctest --test-dir build/validation-20261025-gpu --output-on-failure -j 1
python tools/windows_dev.py ctest --test-dir build/validation-20261025-asan --output-on-failure -j 1
python tools/windows_dev.py ctest --test-dir build/validation-20261025-gpu -R '^host_transfer_[iu]_.*_smoke$' --output-on-failure
python tools/windows_dev.py ctest --test-dir build/validation-20261025-gpu -R '^(cpu_reference_verifier_smoke|hip_reference_verifier_smoke)$' --output-on-failure
python tools/check_documentation.py
python tools/check_dependencies.py --check-lock --json
```

Set `OMP_NUM_THREADS=3`, as used in the batches. The two experimental GPU
configurations use the same direct-HIP toolchain with optional accelerators off:
one enables `RNS8_EXPERIMENTAL_RESIDUE_DOT4` and
`RNS8_EXPERIMENTAL_GARNER_TABLE`; the other enables
`RNS8_EXPERIMENTAL_RESIDUE_SCALAR_CONTROL`. Their `rns8_tests` executable runs
the complete combined Catch2 suite. Source fixtures and case JSON are tracked;
no ignored runner is required to reproduce them.


## Subsequent materialization integration

Commits `a4ef2b9`, `c9e53a2` and `6ce489b` add the opt-in joint
prefix/packed-B planner, incremental layout conversion and a real descriptor-aware
CPU/HIP consumer. Graph prices bind ordered basis, proof envelopes, observations,
target and build. Packing views and exact lift scratch share a bounded liveness
arena. Ragged queries use the owner's gathered bound before destination mutation;
a prefix20 producer with a conservative envelope can feed a certified prefix8
consumer through different producer/consumer tile shapes.

The [new receipt](../metadata/materialization-validation-20261025.json) identifies
all rebuilt artifacts at clean source `6ce489b`, with the same source-input digest
across the four configurations. This supersedes the earlier #42 source gap above.

| Fresh repaired execution | Passed | Expected skips | Failed |
| --- | ---: | ---: | ---: |
| CPU AddressSanitizer | 536 | 22 | 0 |
| Full-feature gfx1100 | 911 | 15 | 0 |
| Packed-dot/Garner-table control, actual Catch test cases | 661 | 5 | 0 |
| Scalar control, actual Catch test cases | 661 | 5 | 0 |
| Installed static/shared ordinary consumers | 12 | 0 | 0 |

The name audit covers all 559 CPU and 927 GPU inventory entries. Each fresh CTest
run omits only the unchanged source-bound logical checker; its prior 1,020.41 s
passing execution is shared. The checker, lowering, generator and ingestion
sources match the preceding checkpoint after line-ending normalization. This
reuse does not qualify the new layout consumer: its independent Boost/CPU and
real HIP differentials ran afresh, including every prefix, signed extrema,
padding poison and K=65535/65536/65537.

The initial run found a generation fixture that repacked with the same
source-version key, correctly hitting the pack cache. The repaired fixture
advances that key. Strict installed builds also found an unsafe-CRT warning;
bounded literal copies now pass `/W4 /WX`. Review additionally found that ragged
price projection must use the gathered owner's bound rather than the source's
possibly wider conservative envelope. The consolidated repair covers a narrower
destination and compares pre/post-gather identities. All repairs were collected
before the single complete retry. No arithmetic mismatch or device fault occurred.

Both installed linkage modes execute the new public-only example. The installed
experimental header also passes strict C syntax compilation; the shared DLL
exports all six new C entrypoints. CPU/HIP fixtures cover replay allocation
counters, shared packing growth, multiple outputs, default/experimental bases,
source mutation/rebind, and differently tiled ragged refresh.

Issues **#19 and #29** are closed after evidence review of compiled provenance/ISA
association and arithmetic certificates. Together with the preceding six,
**nine issues are closed and 42 remain open**, including subsequent #73 magnitude-control closure. Correctness is separate from
issue-specific performance and broader hardware acceptance. The per-issue
[ledger](../metadata/remaining-implementation-20261024.json) records those gaps.

The existing lift/recipe collector now has full/minimum forced-packed and joint
layout-model controls, required v2 materialization receipts and independent cost,
packing-growth, liveness and budget checks. Six new GPU correctness smokes pass.
The model prices are explicitly unqualified analytical estimates. The subsequent
[focused measurements](focused-closeout-measurements-20261025.md) completed 400
captures after the user explicitly authorized shared-host collection. #73's
wide-result fixture scope is complete; #42's calibrated-price/native-roundtrip
and broader evidence gates remain. The old campaign stays stopped.


### Subsequent static instruction review

Restricted inspection of the validated layout consumer/converter objects found
a reporting omission: gfx1100 disassembly uses `ds_load`/`ds_store`, while the
scanner counted only the `ds_read`/`ds_write` spellings. The repaired scanner and
its mixed-spelling/vector-width/atomic fixture pass, along with residue-dot,
measurement-association and tooling-contract suites. Reinspection preserves all
input object hashes. No C++/GPU implementation changed or required another
hardware correctness run.

| Layout consumer build | Static dot4 | Static LDS reads | Static LDS writes | VGPR / SGPR | LDS bytes | Scratch / spills |
| --- | ---: | ---: | ---: | --- | ---: | --- |
| Default / packed-dot | 8 | 40 | 2 | 49 / 57 | 1,024 | 0 / 0 |
| Scalar source | 8 | 34 | 2 | 49 / 57 | 1,024 | 0 / 0 |

The scalar-source loop also emits signed dot4 with the required signed-operand
flags. That build is a source control, not a physical scalar-instruction
baseline for this consumer. Occupancy and dynamic counts remain unmeasured.
Inspection uses the HIP SDK's AMDGPU-capable disassembler; the standalone
LLVM23 distribution lacks that target backend. Exact tool paths, symbol names,
resource metadata and object/code-object hashes are retained in the receipt.


## Subsequent measured-price integration

The complete collected retry at clean `31aa691` follows the consolidated repairs
to the provider output ABI fixture and documentation inventory. The
[receipt](../metadata/product-stage-validation-20261025.json) preserves each test
name/state, commands, timings and the artifact hashes saved before later rebuilds.
The immutable artifacts, rather than the current checkout, identify this run.

| Lane | Passed | Skipped | Failed |
| --- | ---: | ---: | ---: |
| CPU, Clang AddressSanitizer | 541 | 22 | 0 |
| Full-feature gfx1100 | 921 | 15 | 0 |
| Packed-dot/Garner-table control, Catch cases | 663 | 5 | 1 |
| Scalar-source control, Catch cases | 664 | 5 | 0 |
| Installed static/shared consumers | 12 | 0 | 0 |

The remaining packed-dot failure is a missing `direct_hip_zero_output_tile_memset`
event label; all its numerical/residue assertions pass. Investigation found that
HIP event acquisition/record/synchronize/elapsed failures could silently discard
labels. Source commit `3ca72d8` records these failures explicitly, censors invalid
samples, distinguishes synchronized host fallback durations, and retains the
arithmetic assertions. It is a subsequent repair and is not claimed to have run
in this `31aa691` receipt.

The unchanged logical checker remains the single reused CTest entry in each full
configuration, bound to its previously recorded inputs. Unsupported sparse/CDNA
paths remain skips on gfx1100. The 160 subsequent stage-price and complete-route
[measurements](focused-closeout-measurements-20261025.md#subsequent-stage-price-and-proof-policy-experiment)
all pass exact checks on these pinned binaries, while every held-out prediction
cohort fails its declared performance-model tolerance. Correctness and model
qualification are separate outcomes.

Later commits add the global-minimum completed-output control, live/peak owned
allocation observations and phase-lifetime scratch reuse. Their syntax checks
do not extend this runtime checkpoint. Their fresh integration checks are
recorded separately when complete.


## Allocation and global-control integration retry

The consolidated retry passes. The
[reviewed receipt](../metadata/allocation-global-control-validation-20261025.json)
retains both complete passes, per-test names/states, clean compiled manifests,
commands, artifact hashes and the first pass's censored-capture diagnostics.

| Lane | Compiled source | Passed | Skipped | Failed |
| --- | --- | ---: | ---: | ---: |
| CPU, Clang AddressSanitizer | `2fcb91c` | 547 | 23 | 0 |
| Full-feature gfx1100 | `2fcb91c` | 931 | 15 | 0 |
| Packed-dot/Garner-table control, Catch cases | `168a160` | 673 | 5 | 0 |
| Scalar-source control, Catch cases | `168a160` | 673 | 5 | 0 |
| Installed static/shared consumers | `168a160` | 12 | 0 | 0 |
| Additional installed strict C17 ABI probes | `168a160` | 2 | 0 | 0 |

The first pass passed all numerical differentials and both arithmetic controls.
Seven full-feature benchmark/schema checks failed because they required complete
GPU-event timing even when the reporter explicitly censored failed event capture.
The consolidated repair publishes bounded unique label/phase/status diagnostics,
records observer overflow, and reconciles null GPU timing fields with the failure
reasons. Schema/AUTO/prepacking correctness checks accept this explicit loss;
event-timing and promotion admission still reject it. Silent loss, malformed
records and fabricated timing success fail the negative fixtures. The complete
CPU/GPU retry ran after this repair; already passing arithmetic controls are
preserved with their actual source, because their numerical implementation did
not change.

The CPU inventory contains 571 entries and the GPU inventory 947. Each fresh run
omits only the unchanged source-bound logical checker, whose earlier 1,020.41 s
pass is reused. The five proof inputs still match its checkpoint after line-ending
normalization. CPU skips comprise unsupported HIP/target paths, including the new
physical allocation probe. GPU skips remain three sparse-A and twelve CDNA/MFMA
cases unsupported on gfx1100.

Fresh GPU tests verify device/pinned live and peak accounting, failed/pending-free
retention, peak resets and owner moves. They also verify the global-minimum
completed-output path against independent integer references for signed/unsigned
extrema, zeros, mutation, padded output, admission/alias failures and a unique
maximum beyond the first 256 certificates. The measured device peak stays within
the admitted phase maximum and all owned scratch retires to its baseline.
The command-line global-minimum smoke and collector/schema tests pass. This
qualifies the local control, not generation-bound summary caching or a speedup.

The new installed C API passes `/std:c17 /W4 /WX`, static/shared linking and actual
CPU/HIP query/reset execution. Unsupported CPU queries preserve the output canary;
successful HIP resets reconcile epoch and live/peak values. These are requested
RNS8 allocation bytes; driver/library internals and process RSS remain outside
that observation scope.

A separate Linux CI failure at `2fcb91c` rejected a compressed unary guard under
`-Werror=misleading-indentation`. Commit `709ff6f` adds braces and line breaks,
with unchanged expressions/control flow. Actual GCC 13.3 Debug and optimized
Release/OpenMP compilation of the failing translation unit passes in Ubuntu 24.04
on WSL2. This is a CPU compilation check; GitHub CI completion and native Linux
GPU qualification are separate. Windows runtime artifacts remain identified as
`2fcb91c` rather than being relabeled with the formatting commit.

The fresh official-domain documentation check passes **44/44 URLs**, with 134
inventoried documents and zero local errors. The earlier 28 AMD HTTP 429 responses
have cleared. Actual matching-platform command rehearsals remain distinct from
link reachability. No stopped benchmark campaign was resumed, and this integration
batch adds no performance measurement or issue closure.

## Address-domain and algorithm-identity integration

Compiled source **668f9ea** passes a new complete applicable batch: **550 CPU
sanitizer passes / 23 skips / zero failures** and **935 gfx1100 GPU passes / 16
skips / zero failures**. Both builds complete. The GPU skips comprise the
opt-in budgeted address fixture plus the existing 15 target-inapplicable cases.
Five unchanged logical-proof inputs were compared with their prior qualified
source; the 1020.41-second interpreter result is retained under that identity.
The earlier arithmetic controls and installed allocation probes retain their
own dated artifacts rather than inheriting this source label.

Fresh coverage verifies both finite address-width specializations across K
slab boundaries, original-base/window admission and output canaries. The
hipBLASLt geometry regression verifies real warm-cache algorithm indices,
complete work accounting and no repeat index queries across signed, unsigned,
exact-wide and finite contracts.

After those checks, **five independent physical >2 GiB address runs pass** on
RX 7900 XTX. Every output/padding byte and the independent exact checksum match;
all observed allocations retire. Both exact selected kernels have binary-bound
ISA/resource records and successful module occupancy-capacity queries. The
[address qualification](wide-index-gemm.md#native-windows-gfx1100-qualification)
records the matched timing cost and platform limits. Full receipts, hashes,
commands, test states and source manifests are preserved in
[the reviewed record](../metadata/address-domain-qualification-20261025.json).

The first fresh metadata-policy collection stopped before execution on a
long-K prefix-18 planning-range rejection. The smaller magnitude inputs fit
that basis, but the ordinary benchmark's initial plan admits the full native
type before values are bound. Commit **c7b1e45** repairs all six affected
K=65535/65536/65537 cases to prefix 19 and adds a preflight. All **266** declared
public cases pass that capacity audit; metadata, public-measurement and streaming
collector contracts pass. The original partial collection is retained. One
fresh repaired collection uses the original `668f9ea` binaries and separate
collector/profile identities; the completed review follows below.

## Matched hipBLASLt metadata-policy observations

The single repaired fresh collection completes **300/300 captures**: 15 matched
workloads × two metadata policies × two timing modes × five independent
processes. All captures validate, and every policy comparison has matching
library/build/device/geometry algorithm indices, input seeds, output witnesses
and submitted matmul work. Both one-geometry and full/tail two-geometry cases
are covered. The 28 complete records from the stopped first attempt remain
excluded from these comparisons. The earlier stopped large campaign remains
stopped.

Production calls disable timing hooks and retain the same caller-visible
boundary. Each process has three warmups, nine latency samples and three
sustained bursts, each reaching at least 16 completed calls and 250 ms. Values
below are medians of five independent process means, weighted by actual burst
duration/call counts. Required proof, conversion, status and synchronization
remain inside; independent integer audits remain outside timing.

| Contract | Per-block metadata | Per-call geometry reuse | Observed median ratio |
|---|---:|---:|---:|
| 1024³ bounded unsigned, P9 | 3.810 ms (3.721–7.485) | 3.658 ms (3.646–4.029) | 1.041× |
| 512³ exact-wide signed, P20 | 516.811 ms (470.629–565.119) | 531.052 ms (507.458–545.155) | 0.973× |
| 2×3×65535 bounded signed, P9 | 7.489 ms (7.470–15.447) | 7.400 ms (7.348–7.460) | 1.012× |

These small effects do **not** add a qualified elapsed-time gain: process ranges
overlap, or the process-spread gate fails. The 1024³ candidate completes
**0.587 useful integer TOPS**, versus 0.564 for its baseline, with overlapping
ranges. The gains-only PR table is unchanged. No tolerance is widened and no
route is promoted.

The source-work reduction is directly observed: per-call metadata constructs
one set per distinct full/tail geometry instead of one per modulus/block,
reducing descriptor constructions by **9–20×** in the multi-plane controls.
For the P9 long-K example, 12 warmup/measured calls construct 540 versus 60
individual descriptors, with the same 108 matmul blocks and algorithm index
1194. These are API attempts, rather than allocator-byte or elapsed-time gains.
All 15 workload comparisons, process ranges, counters, scoped indices, 300
capture hashes and the retained failed attempt are in the reviewed
[observation record](../metadata/hipblaslt-metadata-observations-20261025.json).

GitHub's two CPU/OpenMP workflow runs at **c7b1e45** complete **18/18 checks
successfully**. This qualifies that revision's Linux CPU matrix; native Linux
GPU execution and later source additions retain their separate gates.

## Physical beyond-VRAM streaming capacity

Compiled source **6a52665** passes the explicit >32 GiB capacity gate on the
24 GiB RX 7900 XTX. All four independent serial/three-stream processes pass
their cold, warm and changed-generation products: **12 complete GEMMs** with
every output/padding byte checked, **2,359,332 numerical output checks**, exact
coverage/DMA replay and complete tracked payload retirement.

The actual native A allocation is **34,362,359,840 bytes**. Its minimum proved
eight-plane RNS form requires **34,360,786,952 bytes**, exceeding the actual
HIP-reported device total of **25,753,026,560 bytes**. This qualifies real
beyond-device-memory capacity rather than just a manually reduced budget.
Observed requested device peaks are **18,054,912 bytes serial** and
**36,096,000 bytes three-stream**, within the declared 64 MiB working allowance.
The three-stream pinned-host peak is **16,986,624 bytes**; the serial path
allocates none. All owned device/pinned payload returns to its live baseline.

Before the physical run, the focused source batch passes **19 GPU checks** and
**13 sanitizer checks with three expected skips**; all five affected Python
contract scripts pass. Both target builds complete with clean source manifests.
The new driver also passes MSVC and strict GCC CPU syntax. No production library
implementation changes are introduced after `668f9ea`; its full qualification
is retained separately. The original benchmark executable stays unchanged for
the same-artifact first-invocation metadata experiment.

See the [capacity report](out-of-core-panels.md#real-32-gib-input-on-the-24-gib-rx-7900-xtx)
and [reviewed receipt](../metadata/stream-capacity-qualification-20261025.json).
These capacity runs do not stand in for an achieved-overlap trace, a five-process
steady crossover or held-out panel-price qualification. Those issue gates remain.

## First-invocation hipBLASLt metadata setup

The separate first-invocation collection passes **60/60 captures**, using the
same unchanged `668f9ea` executable as the 300-capture steady study. Six
representative contracts cover P1/9/18/20, signed/unsigned/finite semantics,
small/skinny/512³ shapes and full/tail geometries. Each policy has five fresh
processes, zero warmups and one completed hook-disabled call. All six matched
policy comparisons have equal physical algorithm/work identities and input/output
witnesses.

Totals add the disjoint measured planning/workspace/metadata-query setup,
matrix allocation and first completed call. Context creation, input generation
and the independent oracle remain outside this explicitly named boundary.
Per-block versus per-call median totals are **122.812/114.922 ms** for the
finite P1 control, **158.763/155.309 ms** for bounded P9,
**164.381/160.491 ms** for signed P18, **162.415/159.159 ms** for unsigned P20,
**455.097/462.519 ms** for 512³ signed P20 and **161.923/166.128 ms** for the
K=65537 P9 tail control. First-invocation samples are not relabeled as steady
observations or mixed into the sustained means.

This completes the within-call metadata/preference optimization study requested
by #70. The source/API reduction and exact parity pass; the negligible/negative
elapsed result is accepted without promoting AUTO or adding cross-call raw
descriptor ownership. The reviewed
[first-call record](../metadata/hipblaslt-first-call-observations-20261025.json)
contains all samples, counters, scoped algorithms, capture hashes and collector
identity. Together the two metadata cohorts contain **360 complete captures**.

## Generation-bound adaptive summaries

Compiled source **4e21e9c** passes the complete consolidated lanes: **556 CPU
AddressSanitizer passes / 23 expected skips**, **942 gfx1100 passes / 16 expected
skips**, and zero failures. Both full builds succeed. All **12 installed
static/shared consumer tests** and **two new C17 ABI probes** pass. The probes
verify the 80-byte summary observation struct, new symbols, CPU unsupported
status and preservation of an old output. The unchanged five-input logical
checker retains its source-bound 1020.41-second passing run.

The explicit cached API retains one workspace-owned summary pair, checks each
operand's live internal identity/generation and native currentness, and refreshes
changed sides independently. Tile bounds/buckets rebuild each call. The GPU
fixture passes every bound method and signedness, mutation, replacement instances,
policy changes, hard-budget rejection, complete exports and native retirement.
Synchronous leases retain failed-work ownership; in-flight workspace destruction
now rejects.

The first review found a profile packaging error before physical collection: the
shared collector accepts individual objects rather than an eight-case array.
Commit **4f04ea3** splits the cases and adds a regression through the actual
validator. This repair changes no C++ production implementation. Both build
manifests and benchmark hashes retain their clean **4e21e9c** identity.

All **40/40 matched captures** pass in a fresh collection, with five processes
per route/workload, complete host-output boundaries, separate first-call/setup
and sustained means, and exact reference checks outside timing. Cached summaries
are actually reused, but no comparison passes both predeclared variance and
range-separation gates. The negative qualification is retained without retiming
or widening thresholds. See the [observations](adaptive-native-product.md#matched-summary-reuse-observations)
and [source/artifact-bound receipt](../metadata/adaptive-summary-qualification-20261025.json).

## Separate resident-output route comparison

Benchmark source **3532e57** compiles in the GPU and CPU-ASan targets; all six
collector regressions pass. At that benchmark checkpoint, production library/kernel/public-header sources
were unchanged from **4e21e9c**, whose 556-pass sanitizer / 942-pass GPU lanes
qualified that implementation. The later native moment change has its own
complete validation below. The focused build leaves the earlier
qualified general benchmark executable unchanged.

All **95/95 new captures** pass, five fresh processes for each applicable route
and workload. Every timed call returns complete P20 RNS output; proof, status,
completion, commit and synchronization stay inside. Independent limb/padding and
all-plane checks run outside timing. All bursts reach the 250 ms minimum.

Fused narrow WMMA wins 512³ at **0.572776 ms / 0.46866 useful integer TOPS**,
**1.288×** over fused scalar at 0.737890 ms. The ranges separate and both process
spreads pass the fixed 10% gate. Fixed RNS wins the tiny/long-K workloads. Two
within-adaptive reuse improvements pass timing gates but lose to that stronger
control; neither becomes an adaptive routing claim. Setup/first-call costs and
all losses are retained.

Static inspection verifies the original code object occurs byte-for-byte in
the measured executable. The selected fused native signed/unsigned kernels
contain actual integer WMMA, with 49 VGPR / 54 SGPR and no LDS/scratch/spills;
the scalar control has no matrix instruction and 17 VGPR / 18 SGPR. This does
not measure achieved occupancy or hardware traffic. See the
[complete comparison](adaptive-native-product.md#separate-resident-output-boundary)
and [95-capture/ISA receipt](../metadata/adaptive-resident-qualification-20261025.json).

## Native moment and completion-boundary validation

Production source **763a948** passes the complete available CPU-ASan and
full-feature gfx1100 batches after one build repair and one full rebuild retry.
The first build found that arbitrary-width limb import shares the range-proof
builder with native packing. Fixed accumulators now apply only to integral
inputs through 64 bits; wider values retain unbounded arithmetic. The regression
compares every retained fact and known flag against an independent Boost oracle,
including 130-bit values, squared moments beyond 256 bits, excluded padding,
native extrema, negative odd midpoints, empty axes and blocked-column tails.

| Gate | Result |
|---|---|
| Complete CPU-ASan batch | **557 passed, 23 explicit skips, 0 failures** |
| Complete full-feature HIP batch | **959 passed, 16 explicit skips, 0 failures** |
| Installed static/shared consumers | **12/12 passed** |
| New short-prefix public export smokes | **16/16 passed**, P3/P9, signed/unsigned, resident/cold |
| Existing logical WMMA proof | Prior 1,020.41 s pass reused after comparing all five source inputs |

Every new arithmetic and timing-contract test executes in this batch. The single
unchanged logical checker is the only excluded entry. The HIP skip set retains
the separately qualified opt-in address fixture and 15 unavailable sparse/CDNA
cases. No missing hardware execution is counted as a pass. The receipt binds
both clean compiled manifests, executable hashes, all test names, the failed
first build and the successful retry.

[Complete source-bound validation](../metadata/native-moment-validation-20261025.json)
· [Forty paired public packing observations](../metadata/native-moment-qualification-20261025.json)
· [Completed conversion collection](../metadata/conversion-completed-qualification-20261025.json)
