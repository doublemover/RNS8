# Windows and gfx1100 validation — 2026-10-25

The [subsequent materialization integration](#subsequent-materialization-integration)
records the latest `6ce489b` rebuild and complete retry. The earlier tables retain
their original artifact boundaries; neither checkpoint qualifies unmeasured gains.

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
