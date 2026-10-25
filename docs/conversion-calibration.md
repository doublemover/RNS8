# Exact import and CRT conversion calibration

`rns8-crt-calibrate` exposes matched private conversion controls. It uses direct
HIP on explicitly recorded device0; there is no CPU or accelerator fallback.
The changes and fixtures below are authored and await the consolidated build,
correctness and measurement batch. They establish no timing or ISA result.

| Operation | Candidates | Exact output |
|---|---|---|
| Signed native import | `import_scalar_signed`, `import_dot4_signed`, `import_wmma_signed` | The same centered default-basis residue planes |
| Unsigned native import | `import_scalar_unsigned`, `import_dot4_unsigned`, `import_wmma_unsigned` | The same centered default-basis residue planes |
| Plain residue CRT | `garner`, `direct`, `balanced`, `fixed_dot4`, `fixed_wmma` | The same canonical unsigned three-limb values |
| Tile-major CRT | `tile_balanced`, `tile_convert_balanced`, `tile_convert_garner` | The same canonical values; the latter two include conversion |

Scalar imports use the existing resident native-to-RNS helper used by the native
packing controls. `fixed_dot4`, `fixed_wmma` and their import variants use the
[fixed packed-byte transform](fixed-conversion.md), including its four-byte
capability probe, admission, allocation and synchronization on every call.
Only gfx1100/1101/1102 with the actual builtin are admitted. Unsupported paths
return77; the collector retains that process as unsuccessful rather than
substituting another implementation. These raw-buffer experiments do not measure
public matrix ownership or change default dispatch.

Arguments are candidate, cells, prefix, repetitions, warmups, unique run ID and
an optional `resident` or `cold` scope. Existing six-argument calls default to
`resident`. Cells are1..1048576, prefixes1..20, repetitions1..10000 and
warmups1..1000. A run ID is1..128 ASCII letters, digits, dots, underscores or
hyphens. Unsupported candidates, scopes, malformed numbers and irrelevant extra
arguments reject before probing a device. The capture schema is explicitly
`rns8_crt_microcase_v3`; collectors reject old captures rather than relabel them.

## Timing boundaries

`resident` allocates and uploads the selected candidate's active input once, then
keeps their storage and the output allocation across calls. Timed calls include
the candidate's checks, capability/status work, conversion and synchronization.
Poisoning, complete output/canary readback and active-input preservation checks
occur outside each timed call. The setup duration is reported separately.

`cold` allocates only the candidate-specific input/output/scratch/status buffers,
uploads the active source, poisons the fresh output, converts, reads the complete output
allocation back and checks every free inside the timed call. Numerical
comparison is outside that interval. Context creation, deterministic dataset,
Boost oracle and host staging allocations remain setup. Cold captures end at a
host output; resident captures end at a device output. Compare candidates within
the same scope. Inactive controls are not allocated or uploaded; the explicit
capacity map records every live candidate-specific device buffer.
Cold input preservation is checked by one separate resident audit before
warmups; its inputs are read-only during subsequent calls, but a per-cold-call
input readback is not charged or claimed.

Each measured instrumented call is followed by a separate call with timing hooks
disabled. Both undergo full comparison. `uninstrumented_host_call_ns` supplies
production call observations; HIP events are diagnostic and may contain censored
null durations. Neither instrumented phases nor within-process repeats count as
independent processes. Instrumented calls always precede their paired production
calls; no randomized instrumentation-overhead claim is made.

Each process then runs at least16 calls, targeting250ms and capped at4096 calls,
with hooks disabled and no numerical comparison between calls. Resident bursts
also omit intermediate poisoning/readback. Cold bursts retain all transfers,
allocation, poison and release work defined by their boundary. One full output
and tail audit follows the burst. A small cell count is a launch/capability/control
diagnostic, not a bandwidth result; large cases exercise the same scopes.

## Data, verification and resources

`splitmix64_conversion_seed3903_extrema_v1` consumes three SplitMix64 words per
cell in both operation families. Native inputs use the first word, with positions
0..5 modulo257 replaced by0,1,UINT64_MAX,2^63,0x8080808080808080 and
0x7f8081ff007fff80. Signed import interprets the entire native word as two's
complement; unsigned import retains its nonnegative value. Independent Boost
`cpp_int` references check every plane and cell.

CRT inputs combine the three words into192 bits, reduce modulo the selected
basis product P, then replace cells0/1/2 by0, P-1 and floor(P/2). Boost constructs
every centered input plane and all three canonical output limbs. Tile inputs
use8x16 physical tiles, initialized to99 including padding. Logical shape is
rows=cells/256, cols=256 when divisible by256, otherwise one row with cols=cells.
Only the source-layout-matched candidates share the same input boundary.

Every candidate reserves its logical output payload plus64 canary bytes,
initialized to99: `prefix*cells+64` for imports or `24*cells+64` for CRT.
Checksums are full16-digit FNV-1a hex over the complete active source and
complete expected output allocation, including padding/canaries. The collector
independently reconstructs both byte sequences; checksum agreement does not
replace the executable's full byte comparison. Poisoning prevents an empty
candidate inheriting a correct output from a preceding verified call.

The768MiB checked aggregate payload envelope is:
`native_host_bytes + plain_host_bytes + 3*output_allocation_bytes + tile_host_bytes + source_audit_bytes + sum(device_buffer_bytes) + 16384`.
It covers actual host/native inputs and planes, gold/readback/poison vectors,
host tile staging, active-input audit readback, active device buffers and the
capability probe. Cold and audit buffers do not overlap in lifetime. Physical
padding is included. The model excludes allocator bookkeeping, context/runtime,
fixed metadata/event objects and process RSS. It is not observed HBM traffic.
Records keep unavailable measurements null.

## Reproducible collection

Authored manifests provide [small controls](../benchmarks/conversion-calibration/smoke.json),
[every prefix](../benchmarks/conversion-calibration/every-prefix.json), and
[65536/1048576-cell controls](../benchmarks/conversion-calibration/large.json).
Each includes signed/unsigned import, all CRT controls and both scopes.

```powershell
python tools/collect_crt_calibration.py --exe build/issues-all/rns8-crt-calibrate.exe --cases benchmarks/conversion-calibration/every-prefix.json --output temp/conversion-prefixes --runs 5 --repeats 9 --warmups 3
python tools/collect_crt_calibration.py --exe build/issues-all/rns8-crt-calibrate.exe --cases benchmarks/conversion-calibration/large.json --output temp/conversion-large --runs 5 --repeats 9 --warmups 3
```

The existing cartesian CLI remains usable, for example
`--candidates direct fixed_dot4 --cells 257 4096 --prefixes 9 20 --scopes resident cold`. Defaults retain the
three original CRT controls and resident scope. Exact duplicate cases reject;
repetition comes only from `--runs`, which launches independent child processes.

Collection randomizes case/process order with a recorded seed and binds the
executable, explicit cases, collector, validator dependencies and readable Python
binary by SHA256. Raw stdout/stderr, timeout/nonzero failures, command line and
independent process identity are retained. Strict validation checks all controls,
layouts, memory sizes, reference checksums, samples, bursts and compiled/device
cohort. Tool/executable/case changes during collection reject. Dirty/unavailable
or experimental-Garner provenance can remain diagnostic evidence; original
provenance eligibility remains false where incomplete. Every capture and summary
has `promotion_eligible=false`, including complete provenance.

The campaign summary retains each process's actual sustained mean and median
production sample. Compare operation, signedness, cells, prefix, scope and source
boundary explicitly; do not combine resident and cold times or import and CRT.
ISA/resource inspection, timing-resolution characterization, measured memory
traffic, clocks and broader target evidence remain acceptance work.

### Byte-identity and packed-dot controls

Ordinary finite-ring modulus-256 host pack/export now copy the logical rectangle
straight between canonical host bytes and centered resident bytes. The two
representations have identical object bytes; padding is neither uploaded nor
written. The public generation/write transaction is unchanged. Modulus255 and
other moduli still execute their conversion kernels. The authored differential
checks all256 byte values, compact/padded rows, repacks, traffic counters and
absence of conversion events. It has not been executed for this change.

The shared reciprocal reducer uses one conditional correction: with
`R=2^32`, `q=floor(x*floor(R/m)/R)` is at most one below `floor(x/m)` for
uint32 inputs and `2<=m<=256`. Thus the remainder is below `2m`. Both direct
HIP and hipBLASLt consume that same implementation.

`RNS8_EXPERIMENTAL_RESIDUE_DOT4=ON` opts every ordinary, finite, region,
grouped and scheduled residue tile loop into the shared signed-byte dot4
primitive. OFF is the unchanged scalar control except the three pre-existing
packed modulus cases. Launch dimensions, LDS loading, tail zeroes, reduction
boundaries and scheduling do not change. The helper explicitly encodes the
known gfx11 sign-extension workaround; CDNA uses signed dot4 builtins and newer
RDNA uses signed-selection dot4 where available. Unknown targets execute the
scalar fallback, not a falsely advertised dot instruction. The host implementation
is separately checked against scalar/Boost arithmetic across signs, tails,
strides, all byte moduli and the K-block boundary. The build manifest records
the opt-in, so paired controls cannot silently share an artifact identity.

Target instruction availability follows AMD's
[integer dot builtin reference](https://rocm-handbook.amd.com/projects/amd-rocm-optimization-guide/en/develop/compiler-builtins/cross-arch/arithmetic-ref/dot-integer-builtins.html).
Compilation, ISA inspection, differential execution and paired runtime
measurements on each target remain deferred. This is not a default promotion
or a measured throughput result.

### Public signed-pack selection and minimal cold ownership

The calibration executable additionally accepts `import_public_signed`,
`import_fixed_signed` and `import_generic_signed`, plus each corresponding
`_padded_signed` candidate. These share the public signed pack's actual selector
and enqueue implementation rather than approximating it. Fixed-prefix controls
now instantiate every supported prefix1..20 and reuse each native load across
its prefix. The public heuristic is not changed without measurements. All
internal signed wrappers enqueue only and the checked boundary waits once.

Padded candidates use seven extra native words per row. Receipts bind the
physical pitch, full input/padding checksum, selected kernel, output/canary
checksum and exact buffer capacities. `signed-pack-crossovers.json` covers both
sides of the256/4096 public cutoffs, wider sizes, prefix1/2/9/18/20 and both cold
and resident scopes. The existing complete-prefix manifests include all new
controls. Public-selector timing does **not** claim public matrix-handle
ownership/currentness overhead: `public_ownership_measured` remains false.

The v3 cold contract allocates only the selected candidate's live buffers:
imports have no inactive residue-input allocation/upload; CRT has no inactive
native allocation/upload; direct tile CRT has no uniform staging allocation;
only Garner owns a status word. Output allocation is its logical payload plus
64 canary bytes. The receipt includes the full candidate-specific capacity map
and aggregate host/device/probe envelope. These changed ownership semantics
must not be compared as v2 captures. Execution and crossover evidence are still
deferred; no production selector is promoted by this implementation.


### Source arithmetic controls and actual emission

The scalar-control option chooses scalar source loops. It does not disable
compiler instruction combining. Static inspection of the new descriptor-aware
layout consumer at `6ce489b` finds eight signed dot4 instructions in both the
packed and scalar-source builds. Compare selected emitted instructions before
attributing any timing difference to scalar versus packed hardware arithmetic.
The existing collectors' source-update counts remain source counts.

The ISA scanner now recognizes both `ds_read`/`ds_write` and
`ds_load`/`ds_store` spellings. These bins count static instructions; vector
width does not multiply the count, and LDS atomics remain outside plain
load/store bins. They do not establish dynamically executed traffic.
