# Exact INT8 instruction-chain probes

The internal `hip_direct_instruction_roof` calibration path runs native RDNA3
signed-byte WMMA with one, two, four or eight independent accumulators per wave.
One chain exposes a serial accumulator dependency; additional chains provide
instruction-level independence within a wave. A separate explicit mode applies
centered modulus-251 reduction after each WMMA instruction. It measures the
combined matrix/reducer sequence, not a standalone reducer throughput claim.

Each input array supplies one signed byte per chain, replicated into the actual
WMMA fragments at runtime. Every wave stores all 256 output cells of every chain.
There is no discarded accumulator or validation in the timed loop. Inputs include
full signed-byte extrema; at most 4096 iterations give K=65536 and a raw absolute
accumulator bound 2^30. The reduced sequence has at most 16*128^2+125 before each
reduction. Neither bound depends on cancellation or a small final result.

The pure query reports two chain-byte input arrays, full int32 output size,
wave-instruction count and integer operations (8192 per WMMA). The caller provides
all device storage and a separate four-byte capability word. A runtime capability
probe rejects unsupported targets before output writes; the kernel has no scalar
fallback. Borrowed buffers must remain alive until completion, including draining
failed work. This does not change production selection or reviewed caches.

The event label isolates the issued chain kernel from the preceding capability
probe. Host-call time includes that probe and synchronization and must be reported
separately. Single-wave event latency includes loop, setup and output stores;
ns per wave instruction is not a direct hardware cycle latency. Sweeps intended
to saturate the device vary launched wave jobs and independent chains; actual
residency and achieved occupancy still require measurement. They must
inspect the emitted instruction chain and spills. Compiler unrolling, scheduling
or instruction elimination must be checked against ISA before interpreting any
roof. Output traffic is modeled, not an observed memory counter.

Authored CPU fixtures check counts, address/cap bounds and signed extrema. HIP
fixtures check every output cell against an independent integer recurrence,
iteration cap-1/cap, a partial four-wave block, all chain counts, both modes and
output guards. Benchmark integration and collector fixtures are now authored
below; compilation, execution and disassembly qualification remain pending.
These sources establish no measured roof or speedup.

## Explicit driver and exact checks

The standalone driver uses the same build-manifest binding as the native,
conversion and word-modular calibrators:

```text
rns8-instruction-roof-calibrate DEVICE raw|reduce251 CHAINS ITERATIONS WAVES REPEATS WARMUPS RUN_ID
```

Device is0..63, chains1/2/4/8, iterations1..4096, and launched wave jobs1..4096.
Repeats are1..10000 and warmups1..1000. The run ID is1..128 ASCII letters,
digits, dots, underscores or hyphens. Malformed controls reject before device
probing. Unsupported target requests return77; there is no CPU, scalar or
different-device substitution. Eight small HIP smoke cases and compiled CLI
rejection fixtures are registered for the final validation batch.

The fixed versioned recipe `fixed_signed_runtime_chain_extrema_v1`, seed0,
uses the first CHAINS entries of these arrays:

```text
A = [-128, 127, -128,    1,   0, -1,  63, -64]
B = [-128, 127,  127, -128, 127, -1, -64,  63]
```

They are uploaded as runtime signed bytes. Each byte fills its chain's complete
16×16 operand fragment, so a scalar reference adds16 signed products at every
iteration. `reduce251` centers after each iteration; `raw` retains the entire
INT32 accumulator. Every cell in all256 outputs per wave/chain is checked
against the independent int64 recurrence. There is no checksum-only admission.
The complete output allocation includes16 INT32 tail canaries `0x12345678`.
Input preservation, every output/canary comparison and checksums run outside
instrumented and production timing. FNV-1a64 covers little-endian bytes of each
entire signed-byte input and the entire INT32 output allocation, canaries included.

The device buffers are allocated once: two chain-byte inputs, the complete
INT32 output, and a separate four-byte capability word. The output is poisoned
before each warmup and paired sample, outside timing. Input upload/allocation
setup, oracle cost and checked release are separate host phases. Failed work
requires a successful drain before buffers can be freed; uncertain allocations
are retained on the failure path until process exit. No successful capture is
written for such a failure.

The256MiB logical live-payload cap models
`4*(4*waves*chains*256+64)+6*chains+8+16384` bytes: three host output arrays plus
one device output, host/device inputs and input readbacks, capability space and
a fixed allowance. The maximum requested output is32MiB. Allocator bookkeeping,
context/events, compiler spills and process RSS are excluded; actual peak memory
and HBM traffic remain unknown.

## Event, host and production scopes

Each instrumented call records its probe-inclusive host wall time and its one
chain-kernel event separately. `kernel_event_ns` can be measured, censored to
null (`below_resolution`), or unavailable; normalized time is null in both
unmeasured states. A separate hook-disabled call follows each instrumented call,
with a complete reference comparison after both. The fixed pair order supplies
diagnostic overhead observations, not a randomized overhead estimate.

`event_ns_per_modeled_wave_instruction` divides the measured kernel event by
`waves*chains*iterations`. The event includes fragment setup, loop control,
reduction when selected, and all output stores. Its normalization is neither a
hardware cycle count nor exact single-instruction latency, especially when many
waves execute concurrently. Integer-operation counts model two operations per
MAC,8192 per WMMA; they exclude reducer arithmetic, capability probing and stores.
`raw` and `reduce251` are distinct programs, with distinct mathematical outputs.

The production burst disables timing hooks and makes at least16 calls, targeting
250ms with a4096-call cap. It retains capability probing and required sync on
every call. It does not poison, read back, compare, checksum or format evidence
between calls; one complete final audit follows. Inner calls are not independent
process observations. Launched waves are not observed resident waves. Actual
instruction counts, occupancy, clocks, cycle latency and compiler private scratch
remain null, with ISA verification and promotion false.

## Independent process collection

The tracked sets separate [eight small controls](../benchmarks/instruction-roof/smoke.json),
[48 one-wave dependency/ILP cases](../benchmarks/instruction-roof/one-wave.json),
[120 launched-wave/chain cases](../benchmarks/instruction-roof/wave-chain-sweep.json),
and [160 partial-block/tail cases](../benchmarks/instruction-roof/wave-tails.json).
The one-wave set uses iterations1/16/256/1024/4095/4096 and all chain counts.
The wave sweep requests32/128/512/2048/4096 jobs at iterations256/1024/4096;
its saturation and occupancy are unmeasured. Both modes appear throughout.

These are pending commands for an already configured build, not recorded evidence:

```powershell
build/issues-all/rns8-instruction-roof-calibrate.exe 0 raw 1 4096 1 9 3 manual-one-wave
python tools/collect_instruction_roof.py --exe build/issues-all/rns8-instruction-roof-calibrate.exe --cases benchmarks/instruction-roof/one-wave.json --output temp/roof-one-wave --runs 5 --repeats 9 --warmups 3
python tools/collect_instruction_roof.py --exe build/issues-all/rns8-instruction-roof-calibrate.exe --cases benchmarks/instruction-roof/wave-chain-sweep.json --output temp/roof-wave-chain --runs 5 --repeats 9 --warmups 3
```

On Linux omit `.exe`. The collector independently reconstructs recipe bytes,
the full recurrence/output/canary checksum, launch/work/resource counts and event
normalization. It validates explicit controls and build/device/runtime identity,
shuffles fresh processes with a recorded seed, and binds immutable executable,
case manifest, collector/helper and readable Python binary hashes. Raw output,
errors and timeout evidence are retained as failed records. Dirty, unavailable
or experimental-Garner provenance stays diagnostic with its original eligibility;
no collector self-promotion is possible.
Summary kernel medians use only measured events and retain separate measured,
censored and unavailable counts; a missing or censored event is never zero time.

Optional `--isa-file FILE` binds an actual file's hash and checks immutability.
It remains explicitly unreviewed, with correspondence unverified. A hash alone
does not prove that the file describes this executable, selected kernel or
device. Actual disassembly, dependency-chain inspection, instruction/resource
counts, measured clock policy and target qualification remain final-campaign
gates. No instruction roof or application speedup is established by these sources.
