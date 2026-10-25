# Matched canonical wrap64 calibration

`rns8-wrap64-calibrate` compares private routes on identical canonical
compact byte-limb inputs and output: signed carry-recoded WMMA, native unsigned
IU8 WMMA, ordinary direct-HIP v4, and corrected rocWMMA on RDNA, a separate signed-INT8 MFMA route on CDNA,
and a native-U64 vector-ALU control on both. Source, fixtures and
case sets are authored and unexecuted. This is not public matrix ownership,
AUTO routing, ISA verification or production promotion. The
[recoding guide](wrap64-signed-radix.md) owns the arithmetic and current kernel
schedule. Strict wrap64 equality is checked without odd-modulus CRT.

## Controls and boundaries

The device ordinal 0..63 is explicit. `signed_radix` and `native_u8` require the
current gfx1100/1101/1102 intrinsic capability. `direct_v4` uses the existing
U32 accumulator through K4096 and U64 variant above it. Any active
`RNS8_WRAP64_HIP_COLPAIR_EXPERIMENT` is rejected before work; the control cannot
silently become v5. `rocwmma_corrected` requires its compiled optional backend
and retains its existing K<=32768 limit. Unsupported methods/devices exit 77;
the collector keeps diagnostic raw output and rejects them as passing evidence.

Each measured batch performs an explicit R=1..256 identical GEMMs and overwrites
one common output. There is no persistent recoded-input cache. For both WMMA
candidates, tile-local staging/recode and the capability allocation/probe remain
part of every private call. All routes use the same byte-limb pack and export
wrappers at the same boundaries.

| Scope | Measured work |
|---|---|
| `resident` | R synchronous GEMMs on inputs packed once during setup |
| `repeated_b` | B packed once in setup; pack/upload A, R GEMMs, export C; buffers retained |
| `roundtrip` | Pack/upload A and B, R GEMMs, export low64 output; buffers retained |
| `cold` | Allocate A/B, pack/upload, R GEMMs, export, checked free of A/B and transport buffers |

Context, driver state, common output allocation and audit buffers are retained
in every scope. Cold means fresh input/transport allocations through the API,
not a GPU reset or proof that the driver allocator cache is empty. Output is
poisoned before timing; cold retaining the common output lets that poison and
a trailing 64-byte device canary be checked outside timing. No diagnostic poison,
reference calculation or readback is hidden inside production call intervals.
The roundtrip/cold output export is intrinsic measured work, not a reference
comparison. Setup and resident input preparation have separate durations.

Each repeat records an instrumented call with HIP-event samples and diagnostic
host phase clocks, followed by a separately poisoned, hook-disabled production
call. Diagnostic phases split input allocation, pack/upload, R GEMMs, export and
input/transport cleanup. Upload staging allocation belongs to pack, and export
staging allocation belongs to export. Event duration can be null when censored.
Event/phase instrumentation is absent from production calls. Medians and GEMM
calls/second use production samples only, averaging central values for even N.
These call rates are not TOPS or an occupancy/ISA statement.

A separate uninstrumented sustained interval executes at least 16 batches, then
stops at 250ms or 4096 batches after that minimum. It performs no independent
verification or poison between batches, and checks its final output afterward.
Both batch and R*batch GEMM counts/means are explicit. Only the last overwritten
output of each batch is audited; no per-overwrite export claim is made.

## Exact dataset and reference

`splitmix64_u64_extrema_seed3135_v1` consumes SplitMix64 words for row-major A,
then B, starting at 3135. `seeded` overrides indices 0..7 modulo 257 with the edge
sequence below. `carry` repeats that sequence for every cell, while `zero`
sets every consumed cell to zero. Each operand's index restarts at 0; generator
state continues across operands even for forced values.

```text
0, 1, UINT64_MAX, 0x8080808080808080, 0x7f807f807f807f80,
0xfefdfcfbfaf9f8f7, 0x0101010101010101, 0x8000000000000000
```

The collector reconstructs these canonical little-endian bytes and records their
SHA256. Full 16-digit FNV-1a checksums cover all native input words and every
expected low64 output. No result is inferred from its checksum. The complete
independent product uses Boost `int256_t`: `sum < K*2^128 < 2^145` for K<=131071.
A separate native unsigned64 accumulation checks every result in the intended
ring, and `cpp_int` audits both edge columns of every row. Fixed Boost reference
operands are released before GPU setup. This is not a native128 baseline.

After each warmup, instrumented call and production call, the benchmark checks
every exported low64 value, all three host padding columns, every canonical
output byte and all 64 device canary bytes. Those audits run outside intervals.
The dataset includes carry-heavy byte boundaries, all-zero products, tails and
K values around the existing accumulator and matrix-control limits. New radix
kernels fold each 16-term WMMA or generation-specific 16/32/64-term MFMA; their declared mathematical 131071/33025
pair bounds are separate metadata and do not claim a longer accumulation slab.

## Resource and evidence accounting

M,N=1..512 and M*N*K<=134217728 bound the CPU reference; K is 1..131071 except
for rocWMMA's explicit 32768 cap. A 256MiB logical host/device-global array payload cap may further
restrict shapes. With AB=MK+KN and P=MN, the reference-phase bound is
`8*AB + 8*P + sizeof(int256_t)*AB + 2048`. The execution-phase bound is
`24*AB + 64*P + 24*M + 4096`. The maximum is charged, rather
than pretending the released reference operands coexist with GPU allocations.
It includes host/native inputs and expected outputs, raw/padded audit buffers,
common canaries, canonical device payload and maximum upload/export staging, including overlapping old/new upload capacity during growth.
Allocator bookkeeping, driver/HIP-event allocations, context and fixed metadata
are excluded; this is not measured RSS, a device-memory peak or free-memory
admission. WMMA static LDS 4096, MFMA explicit LDS 0 and capability scratch 4 bytes describe
source contracts; legacy LDS and fold lengths remain null where not queried.

Captures bind the immutable source/configuration/external-header manifest.
Collector receipts bind executable, case file, dataset, interpreter, collector
and validator module SHA256 values, command, independent process index, device
visibility and the colpair environment. Case/artifact changes, metadata drift,
resource/timing mismatch, runtime/build changes and differing matched outputs
reject the observation while preserving raw stdout/stderr. Optional rocWMMA
captures require its dependency identity. Diagnostic dirty/unavailable/Garner
provenance is preserved without upgrading `provenance_complete`; all records
retain `promotion_eligible=false`. This adapter does not assert a production
benchmark-schema pass or verified instructions.

## Commands and matched cases

Enable the existing `RNS8_BUILD_BENCHMARKS` option. An explicit device 0 example:

```powershell
build/issues-all/rns8-wrap64-calibrate.exe 0 signed_radix resident 17 19 257 carry 8 9 3 signed-resident-1
build/issues-all/rns8-wrap64-calibrate.exe 0 native_u8 cold 17 19 257 carry 8 9 3 unsigned-cold-1
build/issues-all/rns8-wrap64-calibrate.exe 0 direct_v4 roundtrip 17 19 257 carry 8 9 3 direct-roundtrip-1
python tools/collect_wrap64.py --exe build/issues-all/rns8-wrap64-calibrate.exe --cases benchmarks/wrap64/tail-reuse.json --output temp/wrap64-tail --runs 5 --repeats 9 --warmups 3
```

Tracked [smoke](../benchmarks/wrap64/smoke.json),
[tail/reuse](../benchmarks/wrap64/tail-reuse.json),
[512-cubed/reuse](../benchmarks/wrap64/512-reuse.json) and
[matched K-boundary](../benchmarks/wrap64/matched-boundaries.json) cases contain
four-way controls within K<=32768. The separate
[long-K set](../benchmarks/wrap64/long-k.json) covers 33025,65535,65536,65537 and
131071 using the other three candidates. It does not invent blocked rocWMMA
accumulation or a fake timing beyond the implemented cap. Copy a case file into
ignored `temp/` and explicitly edit the selected device when needed.

Each case/run launches a fresh process. Only independent process sustained batch
means enter the summary; samples/GEMMs inside a process are not independent runs.
The summary requires at least two matched independent process indices per
candidate. Legacy RDNA four-way/long-K three-way controls remain explicit;
`rdna-native-u64.json` adds the fifth native-U64 comparator. CDNA small,
carry-slab, boundary and sustained case sets require MFMA, direct-v4 and native
U64 together. Cross-family substitution is rejected. Case order is rotated and
reversed between independent runs rather than running one method to completion. It reports observed per-process medians/ranges
and direct-v4 ratios at each tested scope and reuse count. Missing cohorts remain
incomplete. There is no interpolation, confidence claim, inferred break-even,
route qualification or published performance result before the final campaign.


Collector `*.capture.json` companions retain raw `*.stdout.json` and add the
unchanged executable identity. `gpu_isa_report.py` accepts these captures for
explicit symbol/target/code-object membership checks, and reports VGPR/SGPR,
LDS/scratch, spills and matrix-opcode inventories. A shared-library build is
labelled unresolved until the actual library's artifact identity is supplied;
an executable hash cannot certify a different library's code object. Static
membership is not a claim that a profiler observed dynamic execution. No ISA
or benchmark collector was run while authoring this implementation.
