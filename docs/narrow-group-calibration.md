# Grouped narrow complete-call calibration

The source-only `rns8-narrow-group-calibrate` target compares actual public
grouped fused scalar/byte-WMMA/split-K dispatch with a loop of the corresponding
public single-product calls and ordinary public RNS GEMM. Every task retains
its own destination and workspace; datasets vary by task, with either shared B
or independent B inputs. The RNS control retains only its required representation.
No private native-output timing substitutes for a complete batch.

The compact `benchmarks/cases/narrow-group.json` catalog contains seven matched
cohorts and 21 cases: tiny, long-K skinny, occupied dense WMMA, A/B/both repacking
and cold ownership. Each cohort contains all three dispatch modes. Requested
minimal prefixes, output contracts and proof policy stay identical within a
cohort. The source includes signed and unsigned byte operands where supported;
it never infers unsigned scalar support.

```powershell
cmake --build build --target rns8-narrow-group-calibrate
python tools/benchmark_cases.py
$cases = (Get-ChildItem temp/benchmark-cases/current/narrow-group/*.json).FullName
python tools/collect_narrow_group_calibration.py --binary build/rns8-narrow-group-calibrate.exe --cases $cases --output temp/narrow-group/review --plan-only
```

Plan-only emits commands and creates no capture directory. Removing that flag
starts a new collection in an absent ignored output directory. This is an
explicit measurement recipe; the current development pass has not executed it,
and stopped benchmark campaigns remain stopped. On Linux select the corresponding
executable without the Windows suffix.

Each fresh process computes a complete Boost reference before setup/timing.
Cold calls include context/plan/matrix/workspace creation, packing, products,
padded three-limb exports and cleanup. Resident calls retain these owners; A/B
repacking remains inside the declared call. Resident-output cases omit timed
host exports and perform their complete oracle export afterwards. Padded limb
outputs and all tasks are checked after each sample and sustained interval;
a separate post-timing audit checks every residue plane. Input generation
changes do not recompute an unchanged numerical reference.

The program uses fixed-size phase observations and disables phase clocks inside
sustained callbacks. Three completed-call bursts request at least 250 ms each,
with 16..4096 calls; the record distinguishes the duration threshold from the
call cap. There is no JSON serialization or CPU oracle inside those intervals.
One separate diagnostic batch records actual allocations, completed synchronous
copy counters and native event labels, followed by complete output verification.
Those timings are not summed into production latency. First/final public proof
and retained-storage snapshots are outside timing; destroyed cold workspaces
remain explicitly unobserved.

The collector reuses the bounded owned-process runner, preserves raw process
status/stdout/stderr separately, binds the actual benchmark PID and native
argv/environment witness, and checks the case bytes before and after execution.
It admits only matching stable static executables, compiled manifests and live
physical-device witnesses. Profiler injection is rejected before launch.
Resealed metadata cannot replace the native output projection, task count,
full-output coverage, partial/directory bytes, independent/shared input identities
or actual single/group launch counts. The raw process receipt is hashed again
when a capture is reviewed. Plans, process records and captures are exclusive
writes; partial failures remain available for inspection.

Counterbalanced order and at least two independent processes per mode produce
separate setup and complete-call medians. Baseline/group ratios are explicitly
unqualified: no confidence, crossover, gain or AUTO promotion is inferred.
ISA/resources, power/clock conditions and strongest-control comparisons retain
their own qualification gates. Existing full-width vector controls remain in
the [single narrow calibration](narrow-native-owner.md); their different output
contracts and binaries cannot silently enter these resident-RNS cohorts.

CPU/HIP frontend and Python source checks are recorded in the implementation
ledger. Protocol controls, case expansion, real builds, native execution and all
measurements remain deferred to the combined validation batch.
