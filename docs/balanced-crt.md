# Balanced CRT device baseline

The private canonical export candidate builds a pairwise tree for each default
basis prefix 1 through 20. Successive levels merge adjacent subtrees, carrying
an unmatched final subtree forward. Its dependency depth is ceil(log2(prefix)),
unlike a sequential recurrence over precomputed pairs. Each immutable plan
records original channel slots, subtree products and modular inverses; a merge
overwrites only the left child's slot. Host plans are initialized once with
Boost, using the existing Garner small-modulus inverse owner to reconstruct
each wide inverse. The plan is supplied by value as kernel arguments.

For canonical children a in [0,A), b in [0,B), the kernel computes
t = ((b - a) mod B) * inverse(A mod B) mod B and x = a + A*t.
All subtree products are below 2^160. Bitwise exact reduction doubles a value
below B before adding a bit, requiring less than 161 bits. Modular multiplication
reduces after each doubling and addition, also requiring less than 161 bits.
The final nonnegative multiply A*t and its ordered Horner intermediates are
below A*B; adding a yields x below A*B. Three 64-bit limbs therefore suffice
throughout, without forming an unchecked double-width inverse product.

This per-thread candidate uses 20 reusable wide-value slots and exact bit-serial
arithmetic. Those choices can create register pressure, spills and expensive
integer work. No throughput advantage is claimed. It is private, synchronous,
and returns three little-endian limbs representing [0,P); root signed centering
and destination-width admission are not inferred from the limb type. The caller
must supply valid default-basis residues and own publication on device failure.
The maintained public Garner route is unchanged.

CPU fixtures verify every merge with Boost, all prefix products and inverse
identities, tree depth, random values, zero, P-1 and centered thresholds. GPU
fixtures compare direct and balanced candidates with the same exact values and
tail canaries. Registered applicable compilation/test evidence is recorded in the
[October 23 all-feature validation](local-all-features-20261023.md). Quotient-assisted CRT,
subgroup and tile-batched implementations, group-arrival scheduling, full export
contracts and measured target/prefix selection remain open under issue #39.

## Tile-major consumer candidate

The private `hip_direct_export_canonical_balanced_crt_layout_device` accepts the
versioned uniform-prefix `ResidueLayout` directly. It reads tile-major 8-bit
residues or padded modulus-major rows and emits the same canonical three-limb
logical row-major cells. Descriptor validation, complete input/output range
non-overlap and launch/index limits precede device writes. Basis/version mismatch
rejects; the consumer never treats padding or missing channels as values.

A cell computes its physical base and plane stride once before reconstruction.
The existing tightly packed modulus-major route retains a separate compile-time
addressing specialization. The alternative route allocates no intermediate
modulus-major buffer and performs no conversion launch. This is a real CRT
consumer, but remains a private candidate with the existing canonical output
contract; it does not change public matrix layout or default Garner export.

`rns8-crt-calibrate` adds `tile_balanced`, `tile_convert_balanced` and
`tile_convert_garner`. In the default resident scope, all three start with identical tile-major inputs and end
with identical canonical device limbs; the latter two include conversion and
its synchronization inside each timed call. Input construction/upload and
verification are outside the measured interval and are identified as setup.
Records report logical shape, source physical bytes, conversion scratch and
whether conversion occurs in the call. Compare these three routes to each other;
plain-input candidates have a different starting boundary. The collector accepts
these modes explicitly and rejects mismatched layout/boundary metadata.

New Boost GPU fixtures cover every prefix, 17-by-19 tails, padded rows,
tile padding, zero/threshold/P-1 representatives, malformed descriptors,
alias rejection and input/output canaries. These fixtures and matched route
measurements await the final consolidated validation; no layout speedup or
public promotion is established yet.

The [conversion calibration guide](conversion-calibration.md) also defines an
explicit cold allocation/upload/readback/free scope and the v2 receipt. Cold
comparisons retain the same tile input and canonical value contract, with a host
output boundary. Do not combine them with resident device-output timings.

## Explicit quarter-range FMA / delayed-carry export

The opt-in `rns8_prepare_quarter_crt_export` owner admits only an engine-proven
current exact-wide source with `4*B <= M`. Ordinary exports remain unchanged.
Signed, unsigned and canonical output modes have separate width checks. The
immutable certificate binds the ordered prefix, full weights `W_i=(M/m_i)*y_i`,
canonical representative interval, bound, source instance/generation, target,
build, plan proof and fixed FP operation order. A source mutation makes the
owner stale; missing/insufficient proofs return a range error before publication.
Callers may retain the ordinary general exporter for those cases.

The table generator rounds `y_i/m_i` to binary32 with integer tie-to-even logic;
it never uses an approximate reciprocal. For all twenty prefixes, each
coefficient error is at most `2^-25`. With `r_i<=255`, fixed ordered RNE FMAs and
all partial sums below 8192, the total error is bounded by
`20*(255+8192)/2^25 = 168940/33554432 < 1/64`. Together with `B/M<=1/4`, rounding
the quotient is strictly separated from the wrong half-integer decision.
The host certificate recomputes products, inverses, full weights and each
coefficient bound using exact integers. The independent host evaluator models
binary32 FMA with fixed-point integer rounding; no host rounding mode is assumed.

The HIP source uses explicit RNE FMA and float-to-integer intrinsics in prefix
order. These intrinsics are documented in AMD's
[HIP device API table](https://rocm.docs.amd.com/projects/HIPIFY/en/amd-staging/reference/tables/CUDA_Device_API_supported_by_HIP.html);
the emitted AMD instruction sequence still requires inspection and execution.
No source-level spelling is treated as that evidence.

Reconstruction accumulates 16-bit weight columns in bounded 32-bit registers,
subtracts `q*M` and propagates an explicit Euclidean carry/borrow. The complete
ordered sum/subtraction/carry bound is below INT32_MAX. Truncation is justified
only by the final admitted signed-fit bound; it is not reused for unrestricted
CRT. Signed output is sign-extended; unsigned is zero-extended. Canonical output
uses a full-width centered result and adds M when negative. It never substitutes
a centered array for the older canonical three-limb contract.

Preparation retains bounded scratch. Export stages the complete result and
publishes only after successful device completion/readback. An uncertain drain
retains its source reader and buffers, poisons the context, and permits checked
cleanup retry; failed cleanup never drops ownership. Public padding is untouched.
Regression sources cover every prefix, +/-quarter extrema, quotient agreement
with Boost, corrupted constants, carries/borrows, narrowing, stale generations,
public ownership, tail padding and authored HIP differentials. They remain
**unexecuted**; only host syntax was checked. No default selection or speedup is
claimed, and target execution qualification remains not-run.

The existing `rns8-crt-calibrate` and v3 collector now include matched quarter
cohorts, rather than comparing them with out-of-contract random canonical tuples.
Wide signed/unsigned cohorts compare the new canonical adapter with maintained
Garner, direct and balanced controls. Native-width cohorts compare delayed-carry
signed/unsigned exports with matching one-limb Garner. Four public-owner controls
exercise real packing proofs, retained export preparation, complete staging,
readback and caller publication; cold cases also charge context/plan/pack/prepare
and checked destruction. Resident setup and private certificate setup are
reported separately, not silently amortized into a claimed kernel win.

The independent Python replay reproduces all dataset/output bytes and checks
quarter bounds, widths, signedness and canaries. Pairing keys include dataset and
output contract, so canonical, signed, unsigned and public/private boundaries
cannot accidentally share an output identity. The exact-rational CRT prototype
independently derives coefficient errors, rounds every FMA, performs Euclidean
16-bit carry and checks recovered residues. Tracked cohorts cover all twenty
prefixes, single cells and tail sizes, with both resident and fresh-allocation
scopes. These are **harness/fixture implementations, not collected results**.
VGPR/spill/ISA inspection, hardware counters and target-specific profitability
remain unverified. Public logical-payload budgets exclude runtime/plan metadata;
reported retained payload is not asserted to equal physical process memory.


## Prepared checked integer exports

The public `rns8_prepare_crt_export` owner gives five integer policies the same
signed, unsigned or canonical publication boundary: maintained Garner, direct
CRT, balanced CRT, local group CRT and split group CRT. CPU-reference and direct
HIP are the initial owners. Source arithmetic is exact-wide with an engine-owned
current uniqueness proof. This API is an explicit research choice; default
export and AUTO routing remain unchanged.

A descriptor supplies `algorithm`, explicit `mode` (signed 0, unsigned 1,
canonical 2), `limb_count` (1–32), split producer order and a retained payload
budget. Group roots stay canonical until the complete basis has merged. Signed
centering, magnitude-envelope validation and destination narrowing happen only
at that root. Signed native minimum is admitted by the actual negative-width
check; an equally large positive value rejects for signed output. All cells are
staged before any caller publication, including when a later cell fails width.

Preparation holds source/plan/context references and reserves exact-size host
arrays. HIP owners reserve one contiguous allocation containing a three-limb
canonical root, requested decoded output, optional group words and a status
word. Every policy uses the same checked root-decoding kernel and status drain.
The host payload is the requested limb output; CPU split mode additionally
retains group words. The query and owner share checked payload/address arithmetic.
Allocator bookkeeping, driver granularity and opaque metadata are additional
setup, not physical VRAM measurements.

For N cells, L output limbs and G=ceil(prefix/8), retained host/device payloads are:

| Owner | Host bytes | Device bytes |
|---|---:|---:|
| CPU local integer policy | 8NL | 0 |
| CPU split group policy | 8NL + 8NG | 0 |
| HIP local integer policy | 8NL | 24N + 8NL + 4 |
| HIP split group policy | 8NL | 24N + 8NL + 8NG + 4 |

`rns8_get_crt_export_info` reports these domains, the fixed range envelope,
group products/channel masks, source identity, certificate digest and successful
export count. Each export checks currentness and generation. After compatible
mutation or substitution, `rns8_rebind_crt_export` accepts a current proof no
larger than the prepared envelope, updates binding identity and retains storage.
A rejected rebind leaves the old binding intact. A failed GPU drain keeps its
read lease and allocation with the owner until a successful close; CPU exceptions
release read leases automatically. The same exception guard protects quarter CRT.
These synchronous handles require external host-thread exclusion.

Operation discovery uses `RNS8_OP_CHECKED_CRT`, K=1, tight resident source planes,
no B stride, an optional padded output stride and `output_limbs`. Encode the
explicit output mode as `query.algorithm = policy + 5*mode`. A current-range-proof
flag describes hypothetical admission; actual preparation rechecks engine facts.
Borrowed streams, capture, DAG execution and other backends reject explicitly.

The [installed consumer](../examples/downstream-cmake/checked_crt_export.cpp)
shows creation, padded publication and rebinding. CPU/Boost and real-HIP
fixtures are authored for every prefix, all five policies, signed/unsigned and
canonical roots, 1/3/4/32-limb output, canaries and allocation reuse. Dedicated
CPU allocation-failure sweeps cover staged publication and quarter-export reader
unwinding. Focused MSVC and HIP syntax checks pass; execution and matched public
export/whole-GEMM measurements remain deferred. Existing stage-only gains retain
their original artifacts and do not qualify this new owner.


### Matched public policy calibration

`public_owned_{garner,direct,balanced,group_local,group_split}_{signed,unsigned}`
controls in the existing `rns8-crt-calibrate` use the prepared checked integer
owner. They share the same quarter-bounded native dataset, one-limb output and
complete host publication boundary with `public_garner_*` and
`public_quarter_*`. `checked-public-exports.json` covers short and large payloads,
prefixes 1/3/9/20, both signedness contracts and cold/resident ownership.

Cold calls include real context/plan creation, packing, owner preparation,
export and checked destruction; resident calls include current-proof validation,
canonical root production, checked decoding, completion, status/output readback
and caller publication. Independent reference and source audits are outside all
production intervals, including the sustained burst. The collector independently
replays the full dataset and verifies algorithm, group channel masks/products,
retained host/device/group payload, source identity, certificate and completed
export counts. A separate untimed cold owner supplies cold metadata; it is not
reported as an owner retained across timed calls. Logical envelopes charge the
new root/output/group payload, and exclude opaque owner/driver metadata. These
controls are authored and unexecuted; there are no new performance results.


### Complete GEMM and export calls

The same public candidates accept `gemm_resident` or `gemm_cold` and a required
final K argument. `cells` is the square output area, and inputs are dense signed
±1 or unsigned 0/1 matrices. The engine proves the global K magnitude bound;
`4*K <= prefix_product` admits every compared policy, including quarter CRT.
The fixed prefix and scalar-first proof policy are shared across controls.
Row 0 and column 0 force the positive K boundary, and signed row 1 forces -K.
Independent C++ fixed-word and Python arbitrary-integer population counts check
every output exactly, including K word tails and all caller canaries.

A resident call performs one public GEMM into a fresh output generation, refreshes
the export certificate, reconstructs, completes and publishes. Prepared integer
owners use retained-payload rebind. Quarter owners are destroyed and prepared
again because the current API binds their certificates immutably to a generation;
that work remains in their interval. The maintained default Garner export is
also present. Resident setup reports one priming GEMM separately. A cold call
creates context/plan/workspace, packs both operands, computes exactly one GEMM,
prepares and exports, then checks every destruction. It performs no hidden second
product. Completion and generation counts are recorded and verified.

`checked-gemm-exports.json` contains 364 matched cases spanning 32² to 1024²
output cells, K=64/256/1024/4096, prefixes 1/3/9/20 where the shared bound admits
K, both semantic contracts and both ownership scopes. This covers compute sizes
through 2,147,483,648 useful integer operations per call. v6 captures include
explicit M/N/K, useful operation count, selected prefix, certificate-refresh
policy and completed products. Retained workspace is separately admitted against
the aggregate 768 MiB payload budget before workspace/matrix allocation. This
budget models buffers and retained payload; opaque metadata and physical process
memory are separate. The existing collector seals K into commands and pairing
keys, and refuses raw-stage candidates or non-square output shapes in these
scopes. Production bursts contain public calls only; reference/source audits and
metadata serialization stay outside their intervals. These cases are authored,
unexecuted, and carry no new gain or qualification claims.


Quarter export can consume ordinary uniform residues produced by CPU reference,
CPU AMX, direct HIP, hipBLASLt, CK, rocWMMA or the builtin backend. Discovery and
the concrete owner share that source predicate; compilation, exact semantic
support, current 4B<=M certificate and device binding remain independent gates.
No native-vector or byte-limb storage is treated as centered residues. Broader
producer correctness cases are authored but unexecuted in this development pass.
