# System and memory calibration probes

`rns8-system-roof-calibrate CASE.json RUN_ID` implements bounded private probes
for [#31](https://github.com/doublemover/RNS8/issues/31). It complements the
[instruction chains](instruction-chain-calibration.md),
[conversion/export controls](conversion-calibration.md),
[native tile/fragment controls](native-calibration.md) and
[measured product-stage interface](planning-recipes.md#measured-resident-stage-prices).
The kernels and driver are authored and syntax checked; no execution or timing
is recorded for this source pass.

## Explicit operations

| Mode | Operation and exact check |
|---|---|
| `global-copy` | Grid-stride copy of every runtime-generated uint32 value; complete source/output comparison and output guards |
| `lds-read` | 128 logical lanes per block, independent volatile LDS reads at physical word stride 1/2/4/8/16/32; exact uint32 checksum per lane |
| `lds-read-write` | The same physical family, with two explicit barriers per dependent read/write iteration; independent logical-lane scalar recurrence |
| `launch` | One actual kernel with a consumed one-word result and checked source |
| `event` | Record a prepared completion marker and enqueue a same-stream wait, then observe completion |
| `descriptor` | Upload the complete owned portable-pinned payload into the retained device input; full readback comparison outside timing |
| `graph` | Capture/instantiate one kernel during preparation, then explicitly replay and observe its checked output |

Descriptor mode isolates transfer of a supplied descriptor-sized payload. It
does not include semantic descriptor construction or grouped GEMM dispatch;
the existing native/queue controls retain those costs. The LDS stride experiment
exposes physical word spacing, active lanes, iterations, barriers and requested
shared bytes. It does not label a scalar LDS access as a matrix fragment load.
Use the actual generated/tiled consumers for the instruction-specific fragment
and swizzle comparison. Named-target ISA and counters must establish emitted
transaction widths, broadcasts and bank behavior.

Probe arithmetic is explicitly uint32 modulo 2^32. It makes no exact-integer
GEMM admission or CRT claim. Every observed output/source and every displacement
cell is checked; the reported complete-value sums are reproducibility checks,
not substitutes for those comparisons. Guards preserve sixteen output words.
All payload arithmetic and iteration domains are bounded before allocation.

## Timing, memory and capture identity

Cases specify device, words, blocks, iterations, stride, displacement words,
warmups/repeats and a hard combined requested-payload budget. Two pinned views,
device inputs/outputs, guards, displacement storage and a metadata reserve are
charged. Runtime allocator/event/graph internals remain unknown overhead.

Optional displacement is a separate actual volatile read/write sweep on a
retained buffer. Its source-level byte count is reported independently. Host
timing includes it; same-stream event timing starts afterward. Neither a large
allocation nor a displacement sweep proves cold HBM traffic. Actual HBM bytes,
LDS transactions, physical cache eviction, clock/power state and bank mapping
remain unknown until measured on the named device/compiler.

Each latency sample pairs event-instrumented and hook-free completed host calls.
Each of three sustained intervals requires at least sixteen calls and targets
250 ms, with a 4096-call cap and explicit censoring. JSON, host oracles, readback
audits and graph preparation are outside the measured probe interval. No JSON
object is constructed inside a sustained call. Every call completes its stream.
Event values below resolution remain null; no zero-time rate is invented.

The owner retains pinned inputs, outputs, streams, events and graphs together
until successful drain/cleanup. On failure, an isolated capture process keeps
the complete owner alive until process exit instead of freeing uncertain DMA
endpoints. This is a benchmark-process boundary, not a public runtime cleanup API.

## Future collection

The checked-in [smoke case](../benchmarks/system-roof/global-copy-smoke.json),
[512 MiB source case](../benchmarks/system-roof/global-large.json) and
[separate displacement control](../benchmarks/system-roof/global-displaced.json)
are examples. The LDS family supplies each admitted physical stride separately.
The normal CMake benchmark build creates the executable; CTest registers seven
mode smokes and CPU resource/oracle and collector-contract fixtures.

`tools/collect_system_roof.py --binary EXE --cases CASE.json ... --output temp/roofs`
uses the existing immutable request/artifact wrapper and counterbalanced
independent-process planner. `--plan-only` writes a plan without running a GPU.
It verifies exact call counts, process/device/build/environment identity, live
payloads, source counts, full-value checksums, timing censoring and no overwrite.
Independent process ranges are descriptive. Raw observations cannot install
planner rates or promote a route while ISA, clocks, counter and independent
qualification are missing. The existing typed calibration/rate and held-out
discrepancy interfaces retain their stricter admission gates.
