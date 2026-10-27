# Generic 64-bit-index residue control

The private `hip_direct_gemm_rns_wide_device` route uses unsigned 64-bit logical
coordinates, pitches, plane products, K positions and device addresses throughout.
It proves each accessed byte span against host object/difference limits before
launch and rejects output/input overlap. A fixed grid of at most 1024 blocks uses
a 64-bit grid-stride loop, avoiding shape-derived launch narrowing. Accumulation
still reduces every 65536 terms, preserving the full signed-byte INT32 bound.

This is a scalar correctness control with no global scratch. Existing optimized
and public routes retain their own conservative admission limits; this private
candidate does not establish packing/export or public-plan support for large shapes.
Callers own exact range proofs and output invalidation after execution failure.

Authored CPU fixtures prove spans above 32-bit index capacity and reject overflow.
GPU fixtures use all prefixes, padded tails, K-slab boundaries and a valid unused
64-bit pitch. They compare Boost products and preserve output padding/suffix planes.
The unused-pitch case does not exercise actual device memory beyond 2GiB. Large
physical allocations were a separate gate, now covered by the budgeted fixture
below on gfx1100. Public large-shape integration remains separate from this
private control. The [October 25 integration](local-validation-20261025.md#allocation-and-global-control-integration-retry)
passes those existing CPU/GPU fixtures; it does not establish physical large-address
traffic or qualify the new controls below.

Direct host pack, native export and limb import/export also prove the accessed
byte extent against `SIZE_MAX` and `PTRDIFF_MAX`, using the actual cell size
(including every 64-bit limb). A large unused pitch in a one-row matrix is legal;
a second row whose reachable byte offset is unrepresentable is rejected before
caller-memory access. This arithmetic check does not prove that an arbitrary
caller pointer names an allocation of the required size.


## Matched finite address-width controls

The private `hip_direct_gemm_finite_indexed_window_device` controls use the same
scalar kernel body, grid, K-slab reduction and logical-coordinate calculations.
Only global row/stride/address arithmetic specializes to signed 32-bit or unsigned
64-bit. The narrow route requires the existing complete conservative address
proof; unqualified requests return `RNS8_UNSUPPORTED_BACKEND` before launch.
Both routes retain the original allocation bases: the requested first row is
formed inside the kernel. A small row window therefore still tests its actual
large global row-base multiplication.

The pure support query proves complete original A/B/C spans, window bounds and
output/input non-overlap separately from device availability. Ordinary optimized
and public routes retain their existing limits. In particular, M=32768 and
M=32769 with K=65536 exceed the conservative narrow full-padded-span admission;
M=32767 is admitted. The M=32768 last logical A offset fits INT32, but that does
not relax the full padded-product guard. No default dispatch is changed.

### Budgeted physical fixture

The test `budgeted finite controls read real device addresses beyond signed 32-bit range`
is opt-in. Set an explicit device budget before selecting that test:

```powershell
$env:RNS8_LARGE_ADDRESS_BUDGET_BYTES = '3221225472'
.\build\rns8_tests.exe 'budgeted finite controls read real device addresses beyond signed 32-bit range'
Remove-Item Env:RNS8_LARGE_ADDRESS_BUDGET_BYTES
```

The fixture requests exactly 2,147,549,184 A bytes, 65,536 B bytes and 98,307 C
bytes, and checks available device memory with an additional 512 MiB reserve.
It keeps only a 16 MiB input chunk on the host. Missing/insufficient budgets or
available memory report an unrun/skipped gate, never a simulated device pass.

It fills the complete device input with nonconstant periodic centered residues
modulo the composite 253, then executes all rows of M=32769, N=1, K=65536 using
bounded 128-row windows. The wide kernel actually computes/loads row bases beyond
INT32_MAX; host compute pointers are not rebased. A separately derived Boost exact
period-sum oracle checks every output and padding byte plus an exact weighted
checksum. The signed32 specialization checks all 32767 admitted rows; actual
ordinary and scalar narrow rejection calls preserve the whole poisoned output
at M=32768/32769. A separate high-address call exercises the existing RNS u64
control with explicit modulus 256, so it is not mislabeled as the finite-253 test.

The receipt records the actual target/runtime/driver, available and requested
memory, observed requested-allocation peak and successful retirement, uploads,
completion, oracle identity and independent checksum. It retains matched
same-body i32/u64 samples at the same near-boundary logical shape/window, with
two warmups, nine AB/BA-ordered hook-disabled completed calls and separate
HIP-event diagnostics. Full qualification passes with different row counts are
not a matched timing comparison. Timing failures remain explicit; no speedup or
routing promotion follows from one fixture execution. The completed qualification
below records five independent processes and selected ISA/resource inspection.

For independent process captures, the collector pins the executable hash and
requires a clean source-bound compiled manifest. Its output directory must be
new; it retains timeouts, nonzero exits and malformed results as failures. A
Catch skip without a physical receipt returns 77 rather than qualifying the gate:

```powershell
python tools/qualify_large_addresses.py --binary build/rns8_tests.exe --output temp/large-address-qualification --device-budget-bytes 3221225472 --runs 5
```

### Native Windows gfx1100 qualification

At compiled source `668f9ea`, all five independent processes pass on Radeon RX
7900 XTX with HIP SDK 7.1. Each process reads the real 2,147,549,184-byte A
allocation, verifies every finite output and padding byte, and obtains the
independent weighted checksum **2,853,906**. Observed peak requested payload is
**2,147,713,027 bytes**; all three owned allocations retire to their prior live
baseline. Both ordinary/scalar narrow rejections preserve the poisoned output.
The existing RNS u64 primitive also passes its separate actual high-address
modulus-256 call.

Matched hook-disabled near-boundary completed-call means, aggregated by the
median of five processes, are **4.991 ms signed32** versus **5.150 ms unsigned64**
(process ranges 4.955–5.040 and 5.108–5.173 ms). Wide addressing costs about
**3.2%** in this scalar control; it is a correctness/capacity addition. Separate
event diagnostics succeed for both variants in every process. Full-row passes
with unequal row counts are excluded from that timing comparison.

The exact inspected code object is verified inside the measured executable.
Signed32 uses 17 VGPRs/36 SGPRs and unsigned64 uses 20 VGPRs/41 SGPRs; both have
zero LDS, scratch and spills. A HIP module occupancy query on those exact code
bytes reports eight maximum active 256-thread blocks per multiprocessor for
each, with zero dynamic shared bytes. This is a resource-capacity query;
achieved dynamic occupancy is unmeasured. A separate compiler diagnostic object
did not match the captured code-object bytes and is not used for this binary
association.

The accompanying full batch passes **550 CPU sanitizer tests** with 23 skips and
**935 GPU tests** with 16 skips, including the small finite controls and the
previous sparse/native/limb address regressions. The default suite explicitly
skips the budgeted case; its five opt-in executions are recorded separately.
The unchanged logical interpreter proof retains its earlier source identity.
See the reviewed [qualification receipt](../metadata/address-domain-qualification-20261025.json).
Unavailable gfx942 execution remains deferred; no default dispatch or public
large-shape support is inferred from this private control.
