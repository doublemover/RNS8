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
physical allocations, target performance and public integration remain separate
qualification work. The [October 25 integration](local-validation-20261025.md#allocation-and-global-control-integration-retry)
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
routing promotion follows from one fixture execution. Hardware runs and selected
ISA/resource inspection for these additions remain pending.

For independent process captures, the collector pins the executable hash and
requires a clean source-bound compiled manifest. Its output directory must be
new; it retains timeouts, nonzero exits and malformed results as failures. A
Catch skip without a physical receipt returns 77 rather than qualifying the gate:

```powershell
python tools/qualify_large_addresses.py --binary build/rns8_tests.exe --output temp/large-address-qualification --device-budget-bytes 3221225472 --runs 5
```
