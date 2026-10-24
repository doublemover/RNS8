# Canonical word-modular calibration

`rns8-word-mod-calibrate` compares the private direct-HIP binary double/add
baseline, opt-in packed-dot4 and byte-WMMA lowerings, and explicit scalar binary64 asymmetric
schedule from [word-modular research](word-modular-research.md). All compute the
same complete canonical uint64 product in an explicit ring or prime field.
The binary64 candidate has explicit digit/slab admission; it is not a WMMA
schedule. This driver does not measure public finite-u8 storage, a default route
or a decomposition cache. Source,
fixtures and manifests are authored; compilation, execution and measurements
await the consolidated batch.

## Explicit contract

The CLI is:

```text
rns8-word-mod-calibrate DEVICE METHOD SCOPE M N K DOMAIN MODULUS REPEATS WARMUPS RUN_ID
rns8-word-mod-calibrate DEVICE asymmetric_binary64 SCOPE M N K DOMAIN MODULUS REPEATS WARMUPS LEFT_DIGITS RIGHT_DIGITS SLAB_K RUN_ID
```

The first form accepts method `baseline`, `packed_dot4` or `byte_wmma`. The second form
requires both digit counts and the slab; integer methods reject these extra
controls. Device is0..63, scope is `resident` or
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

## Unsigned byte-WMMA diagonals

`byte_wmma` uses one wave per 16x16 output tile and unsigned byte fragments with
zero padding in spatial and K tails. Its canonical operand width is
`bytes=ceil(bit_length(modulus-1)/8)`, so modulus 256 uses one byte and modulus
257 uses two. The method forms all byte pairs across `2*bytes-1` radix-256
diagonals. Each 4096-term slab bounds every nonnegative accumulator by
`4096*bytes*65025 <= 2130739200 < INT32_MAX`. Fixed integer radix weights and
overflow-safe modular multiply/add recombine diagonals and slabs for both rings
and prime fields. There are no floating coefficients or residue inverses.

The receipt records byte count, diagonal count, slab, exact accumulator bound and
a versioned tile/signedness schedule. The collector derives these independently;
integer methods reject caller-supplied digit/slab controls or unrelated floating
metadata. The wrapper checks the real gfx1100/1101/1102 builtin capability, scans
canonical inputs and derives weights inside the checked call. The event label is
`word_mod_byte_wmma`; missing event instrumentation remains unavailable and null
durations remain censored. A duplicate event or a different word-mod arithmetic
label rejects the receipt. Production host wall cost retains the complete call.

Native operands are reread for each diagonal. This is the initial exact unsigned
byte-WMMA candidate, with no slab packing cache or claimed traffic optimization.
It does not implement the asymmetric floating matrix-engine schedule. Tile and
wave dimensions are source facts, not occupancy, ISA qualification or measured
hardware-cycle latency. CPU/GPU reference fixtures and calibration wiring are
authored; no compilation, execution or speedup is claimed.

## Explicit scalar binary64 schedules

`asymmetric_binary64` takes left/right digit counts in1..4 and slab K in1..65536.
The host derives integer ceil roots `alpha=ceil(p^(1/left))` and
`beta=ceil(p^(1/right))`, then requires the sufficient certificate
`slab_k*alpha*beta+p-1 <= 2^53`. Roots and admission are exact integer operations,
with no trusted caller proof or floating root approximation. An inadmissible
request fails before context probing or oracle allocation. Full64-bit moduli
remain available to the integer controls and are rejected for binary64.

Each digit-pair coefficient uses scalar binary64 FMA over nonnegative integer
digits. The certificate bounds every partial sum within the exact binary64
integer range. Fixed radix weights and overflow-safe integer modular arithmetic
recombine all pairs between slabs; no ring inverse is used. The method remains
valid for admitted composite rings as well as primes. This is distinct from
the packed-dot4 path and from unimplemented matrix-engine concatenation layouts.

Every floating call recomputes roots, admission and weights inside the checked
private operation. Those costs, canonical scan, four-byte flag allocation,
status transfer and required synchronization stay in both timing scopes.
There is no decomposition cache. Coefficient arrays can spill; actual register
counts, compiler private memory and useful throughput remain unmeasured.

The receipt records left/right orientation and slab, both roots,53-bit limit,
exact decimal admission lhs and coefficient cap, and all16 integer weights in
a stride4 array (unused positions zero). The collector independently derives
these values using Python integer roots/powers. It rejects altered orientation,
slab, weights, certificates, types and irrelevant floating fields on integer
methods. `asymmetric_floating_schedule_implemented` identifies the selected
floating route in this observation; it never certifies compilation, ISA or
qualification by itself. The receipt explicitly says `matrix_engine=false`.

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

The driver preserves each device pointer until its checked release succeeds.
On exceptional exit it first attempts a device drain; an unsuccessful drain
retains the borrowed operands/output until process exit. Failed cleanup cannot
produce a successful capture. These are failure-lifetime safeguards, with no
allocation-fault or hardware-error campaign claimed.

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
all three integer methods/scopes and prime/composite contracts. The widths are8,16,23,26,27,
31,35,39,42,48,50,52 and64; moduli are explicit in each case, with no latest-prime
lookup at execution time.

The integer sets contain 12 small, 156 width/tail, 48 slab-boundary and 144 large
cases. [84 matched tile-tail cases](../benchmarks/word-mod-calibration/tile-tails.json)
add output extents 1,15,16,17,19,31,32,33 around the 16x16 boundaries and modulus
256/257 at the one-byte/two-byte transition. All integer cohorts use identical
inputs, oracle, scope and mathematical product across the three methods.

Additional manifests retain matched integer controls and paired swapped digit
orientations: [16 small cases](../benchmarks/word-mod-calibration/floating-smoke.json),
[392 digit/slab/width cases](../benchmarks/word-mod-calibration/floating-pairs.json),
[144 K-tail cases](../benchmarks/word-mod-calibration/floating-slabs.json), and
[192 larger cases](../benchmarks/word-mod-calibration/floating-large.json).
The pair set covers slabs1/17/257 at admitted widths through52bits, while64-bit
cohorts retain only their original baseline/packed-dot4 controls. These floating
manifests are unchanged by the byte-WMMA addition. Tail cases span K16/17/18,
256/257/258 and65535/65536/65537 with explicit slabs17/257/65536. Larger cases
use output orders32/128/256/512, K257,26/52-bit prime/composite contracts and
slabs17/257. The swapped choices recompute the same complete product with the
same input recipe and oracle. They do not reinterpret A/B storage or transpose
the mathematical operation. Authored negative fixtures cover the exact
certificate boundary, invalid/missing/irrelevant digits/slabs, full64 rejection,
preserved failure records and schedule metadata mismatches.

```powershell
python tools/collect_word_mod_calibration.py --exe build/issues-all/rns8-word-mod-calibrate.exe --cases benchmarks/word-mod-calibration/widths-tail.json --output temp/word-mod-widths --runs 5 --repeats 9 --warmups 3
python tools/collect_word_mod_calibration.py --exe build/issues-all/rns8-word-mod-calibrate.exe --cases benchmarks/word-mod-calibration/slab-boundaries.json --output temp/word-mod-slabs --runs 5 --repeats 9 --warmups 3
python tools/collect_word_mod_calibration.py --exe build/issues-all/rns8-word-mod-calibrate.exe --cases benchmarks/word-mod-calibration/tile-tails.json --output temp/word-mod-tiles --runs 5 --repeats 9 --warmups 3
python tools/collect_word_mod_calibration.py --exe build/issues-all/rns8-word-mod-calibrate.exe --cases benchmarks/word-mod-calibration/floating-pairs.json --output temp/word-mod-floating-pairs --runs 5 --repeats 9 --warmups 3
python tools/collect_word_mod_calibration.py --exe build/issues-all/rns8-word-mod-calibrate.exe --cases benchmarks/word-mod-calibration/floating-large.json --output temp/word-mod-floating-large --runs 5 --repeats 9 --warmups 3
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
timing resolution and target-specific measurements remain gates. The scalar
binary64 schedule is authored, with execution still pending. Derived
asymmetric concatenation layouts and floating matrix-engine schedules remain separate
implementation work; CDNA qualification still needs
the deferred hardware.
