# Calibration evidence and planner rates

`tools/calibration_manifest.py` defines immutable identity, run and quantity
records for offline calibration. Identity binds target/device, compiler/runtime,
artifact and ISA hashes, configuration, dataset, clocks/power policy, cache state,
instruction/symbol, final-output contract, timing source and profiling mode.
Serialize the summary with ordinary JSON: its `raw` field retains every input
sample and counter, including nulls. Synthetic test identities are not hardware
evidence.

Use null for censored elapsed times and unavailable counters. A counter explicitly
distinguishes measured observations from modeled quantities. Conversion and
traffic producers should name unique source elements, issued source loads,
actual bytes, residue/encoded-plane evaluations, temporary reads/writes and final
bytes separately. The existing `evidence_database_lib.work_model` estimates
remain modeled counts; elapsed time cannot turn them into hardware counters.

Raw validation and rate admission are separate. Rate export requires at least
five unique independent process runs with identical work counts, no censored
samples, exact-reference and ISA checks, and uninstrumented timing. Within-run
repetitions are averaged first; they cannot masquerade as independent runs.
The interval uses the df=4 two-sided 95% Student-t coefficient for all supported
run counts, plus the largest recorded timing resolution. This is conservative
for larger degrees of freedom under independent, approximately normal,
stationary run means. It does not establish those assumptions or clock stability.
Drift and correlated runs need a new collection campaign.

`planner_rates` requires exact expected identities and one common device,
binary, runtime/compiler and clocks/power policy. It derives the cohort from
the identities and run IDs and exports reciprocal latency intervals as the
existing analytical planner's rational `Rate` objects. Any opcode, layout,
compiler or artifact identity change requires remeasurement. These rates remain
offline inputs; they do not install or qualify runtime routes.

The held-out report rejects overlapping training/test run IDs and requires the
entire observed interval inside the explicitly tolerated prediction interval.
Repeated-B arithmetic reports a modeled strict break-even call count, or no
positive amortization case when per-call packing savings are absent. It is not
the smallest measured profitable count: that requires paired complete-pipeline
runs charging retained storage, eviction, repacking and mutation.

Kernel/application replay timings remain raw observations and are rejected as
uninstrumented planner calibration. Replay preservation, snapshot costs and
allocation/concurrency limits still require target-specific qualification.

This slice supplies evidence plumbing and synthetic schema/algebra fixtures.
Instruction-chain, LDS/cache/memory, conversion, launch/descriptor, reconstruction
and interaction microbenchmark collection, ISA/resource evidence, sustainable
clock verification and real held-out qualification remain open in #31. Tests
for this addition are covered as registered applicable cases in the
[dated validation](local-all-features-20261023.md).

## CRT collection executable

The benchmark target `rns8-crt-calibrate` compares `garner`, `direct` and
`balanced` reconstruction with one canonical three-limb device-output contract.
Positional arguments are candidate, cell count, prefix, repetition count, warmup
count and a unique run ID. Every warmup and measured repetition is checked
against Boost, including output tail canaries. All prefixes 1–20 are supported.
Host-call times include candidate-specific synchronization/status overhead;
backend HIP event labels are retained separately. Verification copies occur
outside the measured call. This repeated-buffer workload is not a cold-HBM test.

For a later qualification campaign, build the target and collect raw records:

```powershell
python tools/windows_dev.py cmake --build build/issues-hip --target rns8-crt-calibrate
python tools/collect_crt_calibration.py --exe build/issues-hip/rns8-crt-calibrate.exe --output temp/crt-calibration
```

The collector shuffles candidate/shape/prefix/run order with a recorded seed,
starts a new process for every record, generates unique UUIDs, preserves command
lines and failures, and checks the executable digest before and after each run.
The embedded source/configuration manifest describes the compiled code. Captures
retain unknown timing resolution and actual HBM traffic as null, and ISA
verification as false. No capture is automatically converted into an admitted
rate or promoted. ISA/resource inspection, sustainable clock checks and a timing
resolution campaign must supply the missing evidence. This executable and its
collector are authored; registered applicable tests are covered by the [dated validation](local-all-features-20261023.md).
