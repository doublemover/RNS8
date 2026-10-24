# Canonical word-modular calibration

`rns8-word-mod-calibrate` compares the private direct-HIP binary double/add
baseline with the opt-in packed-dot4 lowering from
[word-modular research](word-modular-research.md). Both compute the same complete
canonical uint64 product in an explicit ring or prime field. This driver does
not implement or measure the asymmetric floating decomposition, a WMMA schedule,
public finite-u8 storage, a default route or a decomposition cache. Source,
fixtures and manifests are authored; compilation, execution and measurements
await the consolidated batch.

## Explicit contract

The CLI is:

```text
rns8-word-mod-calibrate DEVICE METHOD SCOPE M N K DOMAIN MODULUS REPEATS WARMUPS RUN_ID
```

Device is0..63, method is `baseline` or `packed_dot4`, scope is `resident` or
`cold`, and domain is `ring` or `prime_field`. No device/backend fallback occurs.
M/N are1..512, K1..65537, M*N*K<=134217728, and the aggregate logical payload
must fit512MiB. Repeats are1..10000 and warmups1..1000. The run ID is1..128 ASCII
letters, digits, dots, underscores or hyphens. Modulus is an integer2..UINT64_MAX;
2^64 itself is outside this contract. A prime-field request independently checks
primality with deterministic full-u64 Miller-Rabin; prime powers/composites may
use ring semantics. Invalid controls reject before device probing.

`operand_bits` is bit_length(modulus-1); `modulus_bits` is bit_length(modulus).
For example ring modulus256 has8-bit canonical operands and a9-bit modulus.
Full64-bit cases use UINT64_MAX as a composite modulus and
18446744073709551557 as a prime. Modulus is serialized as an exact JSON integer;
consumers must preserve64-bit integers rather than round them through binary64.

Every checked compute call includes domain admission, canonical input scan,
four-byte scratch allocation, status transfer and synchronization. Packed dot4
also includes its real target/builtin capability check and15 radix weights
computed on the host for that modulus. Its method metadata states4096-term
slabs and the proven worst diagonal bound2130739200; these are static arithmetic
facts, not measured occupancy or ISA. The baseline's digit/slab fields are zero
or null as appropriate. Packed dot4 admits only gfx1100/1101/1102 with the actual
builtin. An unsupported route returns77 and is retained as an unsuccessful
collector record; it cannot silently produce a baseline observation.

## Numerical and memory controls

`splitmix64_word_mod_seed6651_canonical_extrema_padding_v1` visits logical A
row-major, then B row-major, continuing one SplitMix64 stream. It consumes one
word per logical cell and reduces it modulo the actual modulus. Positions0..3
modulo257 in each operand instead contain0,1,p-1 and floor(p/2). A uses ld=K+2,
B ld=N+3, and C ld=N+3. All input/output padding and an additional64-byte output
tail are UINT64_MAX. Input padding is deliberately noncanonical and must be
ignored by both admission and computation.

Boost `int256_t` computes every complete unreduced dot product before applying
the target modulus. K<=65537 and each native product<2^128 bound the full sum
below2^145. Independent unbounded `cpp_int` audits check both edge columns of
every row. Every warmup, instrumented call, production call and final burst is
checked against all logical gold values, output padding and tail canaries.
Output poisoning is outside resident timing and inside each fresh cold call;
a skipped kernel cannot inherit an earlier correct product. Full16-digit
FNV-1a checksums cover canonical little-endian bytes of complete padded A/B and
gold C allocations. The collector independently replays the full integer
reference. It caches only three short checksums per shape/modulus, never payload
arrays or timing evidence.

The512MiB cap covers modeled live payload, with A=m*(k+2), B=k*(n+3), and
C=m*(n+3)+8 native words:
`8*(3*(A+B)+4*C)+16384`.
This includes host inputs and preservation readbacks, gold/output/poison arrays,
all device operands/output, and conservative fixed scratch. Cold buffers and
the separate preservation-audit buffers do not overlap. Exact reference scalars
fit the stated bound and do not retain full multiprecision operand copies.
Allocator bookkeeping, context/runtime/event objects, compiler private memory
and process RSS are excluded. Explicit scratch is the requested four-byte flag;
compiler-generated register spills and observed allocation peak remain unknown.
Actual HBM traffic and timing resolution stay null, ISA verification false.

## Resident and cold timing

Resident setup allocates/uploads A/B/C once and is reported separately. Timed
calls perform the complete checked private operation, including canonical scan
and domain/primality admission. Input preservation and full output audits occur
outside each verified call and after the burst. These are repeated-buffer
observations rather than cold-HBM measurements.

Cold calls allocate all three device buffers, upload padded A/B and poisoned C,
execute the checked operation, read back the complete output, and check buffer
release inside the interval. Context probing, dataset/oracle, host staging and
one separate resident input-preservation audit occur before warmups. Numerical
comparison is outside the interval; per-cold-call input readback is not claimed.
Cold ends at a host output; resident ends at a device output. Compare methods
within the same scope and domain.

Each instrumented call is followed by a separate hook-disabled production call;
full output checks follow both. Use `uninstrumented_host_call_ns` for production
observations. Event labels retain instrumented phases and allow null censored
durations. Instrumented calls always precede their production pairs; no randomized
instrumentation-overhead conclusion is implied.

A sustained burst then makes at least16 calls, targets250ms and stops at4096.
Hooks and intermediate numerical audits are disabled. Resident bursts also omit
intermediate poison/readback; cold bursts retain their full allocation/transfer
boundary. One complete output audit follows. Calls/repeats within one process
are not independent process observations.

## Collection and remaining gates

Versioned manifests provide [small controls](../benchmarks/word-mod-calibration/smoke.json),
[13 widths with17x19 tails](../benchmarks/word-mod-calibration/widths-tail.json),
[K4095/4096/4097/8193](../benchmarks/word-mod-calibration/slab-boundaries.json), and
[32..512 output orders](../benchmarks/word-mod-calibration/large.json). They pair
both methods/scopes and prime/composite contracts. The widths are8,16,23,26,27,
31,35,39,42,48,50,52 and64; moduli are explicit in each case, with no latest-prime
lookup at execution time.

```powershell
python tools/collect_word_mod_calibration.py --exe build/issues-all/rns8-word-mod-calibrate.exe --cases benchmarks/word-mod-calibration/widths-tail.json --output temp/word-mod-widths --runs 5 --repeats 9 --warmups 3
python tools/collect_word_mod_calibration.py --exe build/issues-all/rns8-word-mod-calibrate.exe --cases benchmarks/word-mod-calibration/slab-boundaries.json --output temp/word-mod-slabs --runs 5 --repeats 9 --warmups 3
```

Collection shuffles independent case/process order with a recorded seed. It
binds executable, exact case manifest, collector/validator and readable Python
binary hashes, retains raw stdout/stderr and failures, and checks immutability
before and after every process. Strict validation rejects altered controls,
recipes, complete-output hashes, target/provenance identities, memory arithmetic
and timing/burst evidence. Build/device/runtime changes reject a mixed campaign.
Dirty/unavailable or experimental-Garner manifests remain diagnostics; their
original provenance eligibility is preserved. Promotion is always false.

The summary retains actual per-process sustained means and medians of production
samples. It does not infer a speedup, extrapolate unmeasured shapes or qualify a
route. Correctness execution, ISA/resource inspection, clocks, memory counters,
timing resolution and target-specific measurements remain gates. The asymmetric
floating schedule, derived concatenation layouts and INT8/multiword WMMA
candidates remain separate implementation work; CDNA qualification still needs
the deferred hardware.
