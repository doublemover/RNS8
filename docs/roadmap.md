# Current roadmap and release gates

This is the current execution checklist. The [research specification](RNS8_RESEARCH_SPEC.md)
retains long-term design requirements; [historical queues](archive/README.md)
retain earlier decisions without asserting they are satisfied by current code.

## 1. Source and CPU gate

- Repair/remove the unsafe weighted CRT, unchecked bounded export, coalesced
  pack-tail, persistent K-block, and scalar "WMMA skinny" paths
- Keep exact semantic boundaries, error reporting, and caller-buffer preservation
- Pin malformed public enums to a safely validated 32-bit ABI representation
- Run CPU tests, host-only scalar HIP arithmetic regressions, schema/report
  checks, examples, install/downstream smoke, and available sanitizers
- Record exact validation failures, unavailable tools, and source revision

The [audit](audit-2026-10-06.md) records completion evidence for this preparation
slice. The [CPU allocation checkpoint](local-cpu-allocation-20261008.md) adds
bounded Release/ASan allocation-failure rollback qualification and reuses the
opt-in route's row scratch. Debug-STL fault sweeps remain explicitly skipped.
The [package continuation checkpoint](local-cpu-package-20261008.md) adds a
shipped public example and narrow MSVC static/DLL consumer evidence.
The [full-lineage review](local-lineage-review-20261008.md) adds MSVC x64
RelWithDebInfo DLL package consumers, public-layout comparison and the
proposed complete-lineage integration route.
No source-only gate establishes hardware qualification.

The [integer moment checkpoint](local-moment-range-20261011.md) adds stronger
exact-wide proofs, CPU/ASan rollback coverage and scoped RX 7900 XTX direct-HIP
differential/ISA evidence. Its repaired tooling tests close the observed Windows
CPU gate failure; it does not complete the full hardware/release gates below.

The [runtime cache continuation](local-runtime-cache-20261011.md) adds strict
AUTO cache admission, failure-safe persistence and resident-cache invalidation.
It records final CPU/package, ASan and scoped gfx1100 recovery evidence, plus
the approved moment PR and the separate unpublished continuation boundary.

## 2. RX 7900 XTX / gfx1100 gate

- Compile direct HIP and each optional backend with the actual Windows HIP SDK
- Run the new qualification tests and full differential suite; inspect all skips
- Cover prefix 8 and 9/20 errors, full-width signed/unsigned values, odd packing
  tails, padded strides, wave tails, small output shapes and long K
- Recheck finite moduli 2–256 where supported, particularly skinny N=1/2/4/5/8
- Check actual dot4 signedness, matrix instructions, export code and resource use
- Recheck graph setup/replay/cleanup and persistent storage lifetime
- Re-establish timing with the corrected baseline before any promotion

## 3. MI300X / gfx942 gate

- Compile with a native Linux ROCm toolchain and explicit `gfx942` target
- Run direct-HIP CPU differentials first, then MFMA, CK, rocWMMA, and hipBLASLt
- Cover wave64 behavior, dense MFMA tile variants, explicit sparse-A SMFMAC,
  padded/tail shapes, finite composite moduli, and accumulation cap rejection
- Capture exact source/toolchain/device provenance, profiler/counter data, ISA,
  memory/resource usage and independent-run timing variance
- Keep one physical GPU selected; independent per-device shard scripts are not
  distributed GEMM or collective correctness tests

The [qualification guide](gpu-qualification.md) supplies bounded commands for
both targets and makes unsupported optional dependencies visible.

## 4. Release decision

A release requires all applicable preceding gates plus:

- Reviewed raw evidence retained for every shipped target/backend/contract
- No unexplained failures, skipped required tests, stale kernel identities, or
  unverified claims in active documentation
- Package installation and downstream C/C++ ABI checks on Windows and Linux
- Reproducible same-contract performance review and explicit cache installation
- An explicit support matrix and limitations for unsupported shapes/targets

This source audit does not satisfy that release decision.

## Research that remains open

- A measured safe short-prefix CRT optimization, with overflow-proof arithmetic
- Matrix-engine small-N improvements that genuinely preserve lane/stride/finite
  semantics; a scalar shuffle kernel is not WMMA evidence
- Persistent graph/API lifetime design, asynchronous execution, and multi-GPU GEMM
- Qualification and integration of synchronous device exact-wide lifting; explicit CPU
  lifting and opt-in admission-driven CPU selection with transactional operand/
  output staging are implemented within allocated twenty-plane ceilings
- Stronger correlation-aware range proofs; scalar, per-axis and integer
  second-moment/Cauchy-Schwarz chain propagation are implemented. Native exact
  sums also enable a centered covariance bound; full cross-axis correlations
  remain unrepresented
- Broader target qualification (RDNA2/4, CDNA2/4) under supported toolchains
- INT4/IU4, Ozaki/FP8, Strassen/Winograd, and probabilistic verification, each
  behind the specification's explicit correctness, measurement, and retirement
  criteria; none is implemented simply because a reserved enum/API exists

## Stop conditions for hardware work

Stop at the first arithmetic, out-of-bounds, device fault, unexplained skip, or
wrong-target error. Preserve logs and the exact build. Diagnose correctness
before timing. A missing optional dependency is a qualification blocker for that
backend, not permission to label a fallback as the requested accelerator.

## October 23 issue repair qualification

PR #15 implements the resident ownership, per-matrix content generation,
failure invalidation, scratch preparation, checked limb import and single-pass
exact admission repairs for #16-#21. The optional scalar-first exact proof policy
is explicit; eager moment propagation remains the default. Matrix materialization
preserves identity for a self-conversion and publishes a new value for a distinct
destination. Cache regression tests execute a two-stage GPU chain after both an
unversioned repack and a replacement handle using the same caller version.

Windows builds now select VS 2026 (18.x). Local qualification used MSVC
19.51.36260 (14.51.36231 tool directory), HIP 7.1.51803/Clang 21, gfx1100,
RelWithDebInfo, and OpenMP disabled. CPU CTest: 253 registered tests, zero
failures, two HIP-only skips. Direct HIP CTest: 335 tests, zero failures. The
experimental table build also passed all 335. These are local correctness
results, not cross-platform release qualification. The build-local LLVM #201563
wrapper fix is hashed in the manifest. Windows objects carry a manifest identity
record so a stale-object mixture is rejected by the linker. Source manifests and
launch-time executable hashes replace runtime-checkout attribution; dirty or
incomplete captures cannot qualify a release.

For #22, only the constexpr specification was imported from the isolated
`codex/garner-constants-20261008` branch. The CPU immutable reconstruction plan
is active. A separately implemented HIP table candidate is gated by
`RNS8_EXPERIMENTAL_GARNER_TABLE=ON`, defaults OFF, and is excluded from release
promotion. Its products, per-modulus and pair inverses are checked against Boost.
The safe three-limb recurrence is retained.

Local diagnostic evidence (scratch captures in `temp/`, not release receipts):

- Actual old CPU helper from `8315ccbd` versus the new helper, MSVC `/O2 /MD`,
  65,536 prebuilt rows for `97*i+13`, one setup warmup, five alternating
  observations: prefix 20 took 63.4-64.8 ms versus 25.8-26.2 ms. Prefix 9 took
  23.2-25.0 ms versus 12.3-13.1 ms. Checksums matched. This isolates helper work;
  it is not a matrix-export or end-to-end speedup.
- Host exact-range helper, all-one M-by-1 times 1-by-M, signed prefix 20,
  20 calls per observation and five observations: scalar-first avoids quadratic
  moment propagation after a successful scalar proof. At M=256 eager work was
  28.6-38.3 ms/call versus approximately 0.05 microseconds for the scalar branch.
  Both proved bound 1; the scalar branch intentionally returned no axis summary.
  The tiny branch timing is timer-sensitive and does not represent total API cost.
- Actual gfx1100 A/B: M=N in {1,64,256}, K=1, seed 1, three warmups and 15 repeats;
  bounded-u64 prefix 9 and exact-wide-unsigned prefixes 18/20, fixed and pair
  exports. All 15 case checksums matched. At 64-square prefix 20, fixed export
  kernel median was 201.76 versus 143.80 microseconds and host export was 516
  versus 460 microseconds. Pair export was 176.54 versus 106.68 microseconds.
  End-to-end results were noisy, and one pair capture lacked event samples.
  No default change follows from these diagnostics.
- Compiled ISA reports were generated for both gfx1100 objects. Aggregate code
  increased (610,588 to 929,554 disassembly instruction lines); aggregate maximum
  VGPR count rose from 55 to 83 while maximum SGPR count fell from 104 to 42.
  These are whole-object maxima, not selected-export occupancy. They do not prove
  elimination of every inverse/division loop or a per-kernel resource win.

The follow-up sweep removed the backend's quadratic grouped ownership scans,
permitted shared read-only operands, and preserved exclusive ownership for
packing writes. All 335 HIP tests passed with authored grouped A-squared and
alias-rejection checks; the 253-test CPU gate also passed. Scalar-first CLI runs
matched eager checksums on CPU and gfx1100. Both clean binaries reported the
compiled commit and clean manifest state.

Fresh LLVM 22 ASan instrumentation under the VS2026 environment passed 197 CPU
cases / 287,098 assertions, with two HIP-only skips; its standalone allocation
fault sweep also passed. The OpenMP build passed bounded sweeps at all 36 serial
and 37 two-worker allocation points, plus 896 exact allocation failures and six
K=131073 scratch checks with zero internal allocations. VS2026 STL-specific
annotations are disabled consistently for LLVM's runtime (including optional);
ordinary compiler ASan instrumentation remains enabled. A copied Windows
benchmark launched without Git retained its old compiled SHA after the checkout
advanced; generated mixed-identity objects failed linking as intended.

Selected prefix-20/four-limb unsigned export metadata shows the GPU tradeoff:
fixed baseline SGPR/VGPR/scratch = 98/51/8 bytes versus table 18/71/0; pair
baseline = 102/49/8 versus table 18/82/0. Baseline reconstruction uses out-of-line
cell helpers, while the table is inlined, so caller-only instruction counts are
not comparable. Occupancy and remaining division expansion are not yet qualified.

Remaining qualification includes actual gfx942 runs, per-export division
analysis, controlled partial HIP-write failure injection,
full source-free and stale-object artifact attribution across Linux,
and repeated same-contract release-grade timing. These gates remain open;
source changes and local correctness do not close them automatically.


### Pending issue-batch qualification (2026-10-23)

The development branch now rejects GPU plans and matrix storage whose complete
padded, multi-plane address domain is not qualified for the existing signed
32-bit global index intermediates. The shared checked-address helper proves
last-element offsets and byte extents with checked arithmetic; admission also
reserves space for tile-tail arithmetic. This deliberately rejects some shapes
whose final offset alone fits (including a plane with exactly 2^31 elements,
whose signed plane-size product does not). CPU descriptor capacity is unchanged.
A widened-index GPU specialization remains future work; rejection is not evidence
that large-address GPU execution has been validated. Native/finite, resident,
region and optional backend wrapper checks also cover supplied leading dimensions.

Native CPU packing uses integer remainder without per-cell Boost conversion.
The multiprecision oracle remains in differential tests and exact-wide imports.
Performance and the new regression cases remain unqualified until the deferred
validation campaign completes.

Benchmark comparisons include the finite modulus, chain length and logical
output boundary in shared contract keys. Missing finite modulus is invalid
comparison evidence. Variance reports count execution IDs within an immutable
artifact/environment cohort; legacy captures without these identities remain
descriptive and cannot qualify independent runs. The sweep launcher assigns an
ID once per launched process and copied captures preserve it. Promotion ledger
consumption binds the report entry to the capture digest. Multi-GPU reports derive
expected ranks from requested launch devices and reconcile per-rank case coverage,
so missing output cannot reduce the expected inventory.


Host timing contract version 2 preserves fractional microseconds from clock ticks
through sample storage and aggregation, printing 17 significant digits. The
nominal steady-clock period is recorded separately from physical accuracy
(which remains unknown); 1001 empty timer pairs characterize median/p95 overhead
outside workload intervals. A conservative censor threshold is the larger of one
nominal tick and five times p95 pair overhead. Phase states distinguish measured,
below-resolution, fused, elided and unsupported work. The existing explicit
`--host-api-batch-size` mode is bounded to 4096 independent tasks and labels its
aggregate measurements as amortized batches with per-task outputs. It does not
claim single-call latency. Legacy integer-us captures retain their quantization;
comparison output marks their ratios unqualified, and suppresses ratios for
censored or mismatched measurement modes. HIP-event samples remain separate.


Research foundation work (not completed hardware experiments):

- `tools/arithmetic_certificate.py` separates representation tables, ordered
  weighted recombination, accumulator domains, exact-range basis proofs and
  instruction identity. It checks all canonical byte pairs with exact rational
  arithmetic and emits a SHA-256 identity. An evidence reference alone cannot
  mark a certificate instruction-measured or dispatch-eligible.
- `tools/encoding_catalogue.py` derives a deterministic scaled-E2M1 base-13
  schoolbook oracle and generates host reference tables. Numeric scale 2 is
  explicit; hardware scale-bit encoding is unknown. The AWE six-product (8,17,48)
  construction is independently enumerated with explicit signed-byte coverage,
  including -128, and pinned source-document identity. Real floating matrix
  instruction qualification remains pending.
- `tools/crt_research.py` implements exact host direct CRT and a complete balanced
  merge tree, with per-intermediate width certificates. These are algorithm
  prototypes, not new GPU export selectors or measured wins.
- CI now has GCC/Clang, OpenMP enabled/disabled and static/shared combinations.
  `RNS8_REQUIRE_OPENMP=ON` prevents an enabled qualification job from silently
  falling back to serial execution. These new CI combinations are not yet claimed
  as passing.

The new research regression cases are authored and registered; execution remains
pending the development-first validation campaign. Research issues #29–#48
remain open until their concrete runtime and target-specific acceptance gates
are met, including device, calibration, lifecycle and performance evidence.


The explicit synchronous `rns8_lift_exact_wide_hip` entrypoint now has a direct-HIP
implementation. It reconstructs with the current 192-bit Garner helper, applies
signed centering, checks the certified magnitude and stages only missing suffix
planes. Only status returns to the host. Successful materialization preserves
logical generation; a final device-copy failure invalidates currentness. The
API requires an existing uniqueness proof and allocated capacity, and does not
provide device-owned bound discovery or graph-safe/asynchronous execution.
Every source/destination prefix pair has a Boost differential fixture, including
signed boundaries and suffix canaries. Actual GPU validation is still pending.


Offline search/model additions remain non-promoting:
`modulus_scheme_optimizer.py` performs bounded exact subset/form search under an
explicit additive picosecond objective and same-cohort costs. It reports the
finite search domain, strict CRT range products, incumbent, and search exhaustion;
its lower bound becomes the incumbent only after complete search. Unknown costs
are rejected. The default offline ladder now matches the runtime ladder,
including modulus 193, with a source-alignment regression.
`analytical_planner.py` schedules concrete named stages with dependencies,
exclusive resource reservations, conversion/matrix/reconstruction rate terms,
wave quantization, uncertainty intervals and live workspace accounting. All
current tests use synthetic calibration; no measured target model, winner regret,
runtime shortlist or promotion is claimed. Hardware calibration and held-out
validation remain required by #30/#31.


Unavailable-target qualification is deferred at the user's explicit request on
2026-10-23: no additional Linux multi-GPU, Instinct/CDNA, low-precision floating
matrix or CPU matrix-extension host is currently available. This does not retire
the experiments or qualify their implementations. Local CPU/gfx1100 work continues.

`residue_integrity.py` provides the offline #49 phase-one checker and authored
corruption fixtures. It excludes check channels from reconstruction range, binds
operation/input-generation/epoch/basis identities and rejects a consumed check
channel after basis growth. Check residues originate from original integer dot
inputs. Its result explicitly limits detection to errors nonzero modulo the check
product under a correct independent check path, with common-mode and undetectable
error fixtures. It neither corrects values nor authorizes runtime publication.
GPU check-plane execution and measured overhead remain future work.


`block_gemm_research.py` adds the #48 phase-one division-free Strassen polynomial
certificate and an exact one-level CPU modular implementation. Pre-additions are
reduced before signed-byte products; odd rectangular dimensions are zero-padded
and each product is accumulated immediately without retaining seven product
matrices. Exhaustive polynomial expansion and all byte-modulus tail fixtures are
authored. No rank-23 program, GPU transform path, performance result or promotion
is claimed.


The private direct-HIP #37 baseline now accepts explicit split count and scratch
budget, with no automatic runtime selection. Fragment producers reduce signed
INT8 dots (at most 65536 terms, absolute bound 2^30) to canonical byte scratch.
Each merge level owns disjoint groups of at most 128 bytes (sum at most 32640),
reduces again, and alternates bounded scratch regions. A final centered store
preserves padded output columns. Producer, merge and store have separate event
labels; each stage is synchronous in this initial baseline. Differential cases
cover prefixes 1/9/20, K around 65536 and 131073, multi-level merges, cancellation,
worst signed-byte products, padding and budget rejection. Execution remains
pending. Matrix-engine producers, prepared scratch, planner selection and timing
evidence are still required before calling the research issue complete.


The existing modulus generator now emits valid host/HIP C++ reducer templates,
including explicit centered handling of modulus 256, instead of undeclared
function specializations. Generation is bounded to 28 distinct byte moduli and
K slabs at most 65536, checks arithmetic certificates before writing, and emits
deterministic UTF-8/LF sources plus source/generator/certificate hashes. Generated
reducers are compiled into the unit target with all-byte-ladder and INT32 boundary
fixtures. This repairs the #55 reducer baseline only: matrix-tile variant
generation, compile-resource extraction and actual GPU qualification are pending.


The arithmetic checker also accepts `rns8_arithmetic_composition_v1` proof DAGs.
It verifies the stated evaluation order for integer input/add/multiply/dot,
canonical/centered reduction, divisible unreduced integer division, physical
rescaling and layout permutation nodes. Each node has a physical width, value
interval, divisibility information and separate modular/scale domain. Authored
fixtures reject overflowing fusion of individually safe dots, carried overflow,
reduction before noninvertible division, mixed accumulator scales and final-bound
substitution for partial-sum safety. Input assumptions still require binding-time
validation, and this host checker does not qualify instructions or route kernels.
CRT-merge nodes and runtime certificate consumption remain open.


The #47 private direct-HIP scalar baseline fuses `alpha*AB + beta*D` into one
output store, retaining slab reductions in registers rather than materializing
an intermediate AB matrix. Integer coefficients are reduced before multiplication
(including INT64_MIN), and the final sum has an absolute bound of 65280. The
shared host/device epilogue has Boost fixtures for every byte modulus; the GPU
fixture covers full coefficient extremes, padded strides, prefix 20 and K=65537.
Complete pointer spans reject overlapping output writers in this and split-K
baselines. No public expression API, whole-expression integer range admission,
matrix-engine fusion, cache selector or speedup is claimed yet.


The #53 private native signed-i64 candidate scans the actual device operands for
absolute maxima (INT64_MIN uses unsigned magnitude arithmetic), then admits only
`K*amax*bmax <= INT64_MAX` before launching scalar native products. The proof
covers each product and every partial sum; zero products remain valid even when
an operand contains INT64_MIN. Rejection preserves output bytes and does not
return a proof. Scan, status transfer and GEMM costs remain visible. Authored Boost
fixtures cover full-width boundaries, unsafe cancellation, padding and K=65537.
This synchronous scan-per-call baseline is not a performance promotion, cached
admission, INT8 matrix route or public planner integration. Validation is pending.


Compilation checkpoint (not test execution): at `3a639ed`, VS2026 completed both
`cmake --build build/issues-cpu --parallel 4` and
`cmake --build build/issues-hip --parallel 4` through `tools/windows_dev.py`.
This includes the benchmark, libraries, generated reducer test, and gfx1100
base-extension/split-K/epilogue/narrow kernels and differential executable. An
initial MSVC lambda-capture error in the generated-reducer test was corrected.
No new CTest, GPU execution, sanitizer or benchmark qualification is asserted.

The #40 phase-one adapter now has a checked uniform-prefix tile-major byte layout
and host/direct-HIP converters to/from modulus-major storage. Descriptors own
shape, strides, tile extents, grid size, default-basis version, prefix and checked
payload bytes. Conversion touches only active cells, preserves padding, rejects
full-span aliases and cannot add missing planes or reinterpret a different basis.
Authored mapping-bijection and GPU round-trip fixtures cover prefixes 1-20, tails,
padding, malformed descriptors and overflow. This addition follows the preceding
compile checkpoint; its gfx1100 test executable compiled at `65846cb`, but has
not executed. Fused consumers, ragged
directories, exact extension adapters and route-level cost selection remain open.


The #39 direct-CRT candidate now has an explicit private GPU canonical export.
Compile-time weights are P/m with inverses modulo m; each coefficient is reduced
before multiplying its 192-bit weight. Every term is below P and every ordered
sum below 20*2^160, so the unreduced numerator fits below 165 bits; at most
prefix-1 product subtractions canonicalize it. The output contract is exactly
three little-endian limbs in [0,P), with no inferred signed or bounded semantics.
Boost constant/width fixtures and prefix-1-through-20 random/boundary GPU cases
are authored, and the gfx1100 test executable compiled at `65846cb`. Execution,
resource measurement, balanced-tree GPU
implementation and benchmark-selector comparison remain pending.

The #46 local candidate uses the RDNA3 signed INT4 WMMA primitive for explicit
finite moduli 2-16. Every signed-byte input is first reduced into the centered
modulus alphabet, so nibble packing is exact; default-ladder residues cannot use
this route. Accumulators reduce after each 65536 logical K terms, bounding every
partial by 65536*64+15. Tail operands are zero and output padding is preserved.
Device-compiled target/builtin detection reports unsupported before touching C
on other targets. The synchronous private route separately exposes capability
probe and GEMM events; per-call probe overhead is not hidden or claimed optimal.
CPU exhaustive encoding/product checks and Boost GPU differential cases across
WMMA tile tails and K=65535/65536/65537 are authored. Compilation, execution,
instruction inspection, performance evidence and public planner integration are
pending. See AMD's
[RDNA3 dense WMMA builtin reference](https://rocm-handbook.amd.com/projects/amd-rocm-optimization-guide/en/develop/compiler-builtins/rdna/rdna3-dense-wmma-builtins.html)
for the underlying instruction interface.

The #35 one-shot prototype now includes private RDNA3 WMMA kernels for native
signed/unsigned 64-bit inputs and channel groups 1, 2 and 4. Each wave owns one
16x16 C tile and a contiguous group from the fixed basis. Raw A/B loads occur
outside the channel loop; each channel independently converts, accumulates and
reduces. Partial groups execute no omitted planes, and K slabs include the prior
residue in their INT32 bound. The route has no packed-input or INT32 global
workspace and writes centered output bytes directly. Group size is explicit,
not automatically maximized. CPU/Boost differential fixtures cover full-width
signedness, composite/even moduli, partial groups, padding, tile tails and safe
K boundaries. Compilation and execution are pending; source-level load reuse
does not prove generated load counts, occupancy or speedup. ISA/resource
inspection, repeated-B/resident variants, descriptor/planner integration and
matched end-to-end experiments remain open.

The #34 discovery baseline now scans device-native signed/unsigned inputs for
16-row A and 16-column B absolute maxima. It forms exact 192-bit
K*max_abs(A)*max_abs(B) tile bounds, including INT64_MIN and full uint64 values,
then chooses the smallest available strict signed/unsigned prefix. Zero is
proved from actual all-zero operand regions, not zero residues. Insufficient
capacity returns RANGE_ERROR. Device counts, offsets and scatter compact all
tiles into prefix buckets; only the range-error status returns to the host.
Within-bucket order is unspecified, but each tile has one owner and appears
once. The private API exposes summary/selection/compaction timings and scratch
budget; raw output records are valid only while the input values remain
unchanged. Generation-bound public proof reuse, L1/K-panel refinements, a
device-count consumer and discovery-versus-conversion planner selection remain
open. Boost width/prefix and GPU bucket/tail/mutation fixtures are authored;
compilation, execution and net performance are pending.

At `f3127af`, the gfx1100 HIP test executable compiled with grouped native WMMA
and device bound discovery. No tests or performance measurements were run.

The first #43 public slice now provides synchronous prepared GEMM bindings,
resource retention/BUSY destruction, explicit generation refresh and completion
tokens with prepared-instance identity plus nonwrapping epochs. Existing GEMM
write guards own publication. Old tokens cannot certify a later output mutation.
See [prepared execution](prepared-execution.md) for the exact state/lifetime
contract and the explicitly unsupported async/graph behavior. CPU/HIP API
fixtures cover arithmetic families, two retained consumers, A==B, reverse
execution/destruction order, stale inputs and old completions. Compilation and
execution of this slice are pending; async submission and graph envelopes remain
open.

The #38 first public baseline can retain heterogeneous prepared executions,
preflight all generations and writer ownership, refresh bindings transactionally,
and execute through ordinary sequential calls with per-task completion tokens.
Mixed-shape/semantic, stale-later-task and cross-task hazard fixtures are authored.
Persistent matrix-engine queues, uploaded tile descriptors, bounded chunk
scheduling, fairness and performance qualification remain open. The preceding
single-execution API compiled in the CPU build at `55feaea`; this group addition
and all new runtime tests remain unvalidated.

The #42 first resident executor now validates and retains ordered exact-wide
MatMul DAGs, preflights complete bound propagation through the existing proof
owner and executes ordinary CPU/direct-HIP calls without intermediate export.
Input-generation rebinding, diamonds and deterministic rejection of forward
references/duplicate writers are explicit. CPU/HIP Boost fixtures cover
three-node shared-value graphs, mutation, zero graphs and downstream range
exhaustion before writes. See [resident graphs](resident-graphs.md) for supported
uniform-storage contracts. Allocation/liveness optimization, explicit extension
and conversion nodes, ragged consumers and cost-based planning remain open.
Compilation and runtime validation of this slice are pending.

Both CPU and gfx1100 HIP test executables compiled at `3bc864b`, including the
prepared APIs and resident DAG. These are compilation checkpoints only.

The #40/#41 adapter now includes a compact ragged directory carrying each tile's
prefix, byte offset, exact magnitude bound, basis/version and logical value
instance/generation. CPU and HIP materializers reconstruct the unique signed or
unsigned value before filling missing planes into a uniform modulus-major or
tile-major destination. A suffix is never treated as zero. Destination prefixes
must independently retain uniqueness; insufficient target range, stale identity,
malformed directories/residues and violated bounds reject publication.
The GPU route stages output plus preserved padding and reports whether the final
commit began, so its future public owner can invalidate after a failed copy.
Authored fixtures include neighboring prefixes 1/2/9/3/12/20, different destination
tiling, values through 140 bits and a following GEMM crossing source tile bounds.
Private callers still own proof/generation binding; public ragged matrix
ownership and DAG scheduling integration are pending. Compilation and execution
of this addition remain pending.

The ragged materialization HIP test executable compiled at `91f8795`; runtime
validation remains pending. The #51 private block-sparse baseline now packs
actual native zeros, validates CSR/CSC block directories, and intersects K-block
lists in a single-owner HIP output kernel. See [block-sparse](block-sparse.md)
for its budget, generation and arithmetic contracts. Boost CPU and HIP fixtures
are authored; this addition is not yet compiled or performance-qualified.

The block-sparse HIP test executable compiled at `e6b7dac`; tests remain pending.
The #39 scalar-device baseline now also includes a fully balanced CRT merge
tree with checked 192-bit arithmetic and immutable host-generated plans. See
[balanced CRT](balanced-crt.md) for the exact intermediate bounds and remaining
comparison gates. CPU and GPU differential fixtures are authored; compilation,
execution, resource inspection and performance measurement remain pending.

The balanced CRT HIP test executable compiled at `6682403`; runtime validation
remains pending. The #31/#30 calibration bridge now preserves typed identity,
raw run samples, missing/censored data and measured-versus-modeled counters.
It exports same-environment rational planner rates only after independent-run,
reference, ISA and uncertainty admission, and supplies held-out discrepancy
checks. See [calibration](calibration.md). No hardware calibration constants
have been produced or promoted; real microbenchmark collection remains open.

A CRT collection executable now exercises Garner/direct/balanced canonical
three-limb exports with Boost verification outside each timed call. Its collector
records independent process UUIDs, compiled provenance, executable hashes and
shuffled order. Missing ISA/resolution/counter evidence remains explicit and
blocks promotion. This collection path is not yet compiled or executed.

Both the CRT collection executable and HIP test executable compiled at
`edee062`; collection and test execution remain pending. The #47 exact-wide
public linear epilogue now admits the complete alpha*AB+beta*D bound before
writes, rejects output aliases and publishes through the existing write guard.
See [exact epilogues](exact-epilogues.md). CPU/HIP API fixtures are authored;
compilation, execution and performance qualification of this addition are pending.

The public exact epilogue HIP test executable compiled at `0717c19`; runtime
validation remains pending. The #48 private GPU baseline now performs one-level
Strassen transforms, existing direct-HIP subproducts and staged recombination,
with scratch reuse and compile-time coefficient verification. See
[device Strassen](strassen-device.md) for width/lifetime bounds and remaining
retirement gates. This addition is not yet compiled or executed.

The one-level Strassen HIP test executable compiled at `8c5d00f`; runtime
validation remains pending. The #49 private device sentinel now compares
independently recomputed native-input checks modulo 257/263 with a range-admitted
candidate reconstruction. Identity mismatch and failed integrity are explicit;
an authored undetectable-error fixture preserves the stated coverage boundary.
See [device integrity](device-integrity.md). Public commit/cache integration,
compilation, execution and overhead qualification remain pending.

CPU and HIP test executables compiled at `160e652`, including the integrity
sentinel. The #43 explicit stream subset now prepares an event, reserves readers/
writers/workspace, enqueues exact-wide dense GEMM without publication, and
publishes only after query/wait observes completion. Capture and other operation
subsets report unsupported. CPU lifetime and HIP two-stream regression fixtures
are authored; this addition is not yet compiled or executed.

The stream-submission HIP test executable compiled at `06027bd`. The #43
fixed-envelope graph addition now captures the enqueue-only kernel path without
publishing C, owns the graph and captured addresses, and replays through the same
range/reservation/completion owner. Stable-address rebinding and repeated replay
fixtures are authored. Compilation, runtime correctness, leak/overlap checks and
break-even qualification of graph replay remain pending.

The graph-enabled HIP test executable compiled at `9ebdc7f`; runtime validation
remains pending. The #36 offline mapping simulator now records exact tile/group/
K-slab ownership, explicit slab barriers, tail/cost buckets and native-versus-
packed block traffic. See [locality schedules](locality-schedules.md). Authored
bijection and shuffled-arithmetic fixtures await combined validation. Runtime
remapping and measured cache/XCD behavior remain unqualified and are not inferred
from this simulator.

The #34 unchanged-schedule hot-path slice now validates and seals plans at
construction, caches fingerprints and active counts, and admits equivalent-plan
workspace reuse with structural equality before retaining the new identity.
See [plan validation](plan-validation.md). Diagnostic mutation tests retain full
revalidation; high-tile-count traversal counters and collision fixtures are
authored. Compilation, execution and latency measurements remain pending.

Fresh issue-comment repairs are implemented for #24's sparse host dense-plane
stride domain, #29's exact integer certificate schema, #30's interval-liveness
workspace bound and #39's pre-reduction direct-CRT width certificate. These
changes have authored adversarial fixtures and await combined execution.
Sparse conversions prove accessed bytes against SIZE_MAX/PTRDIFF_MAX before
access; unused single-row/plane pitch remains legal. Malformed certificate
endpoints receive no proof hash, and CLI input rejects non-finite JSON constants.
Planner workspace admission now bounds all lifetimes between earliest launch
and latest last-consumer completion; it is explicitly conservative and includes
equal-time boundaries. CRT certificates distinguish inverse multiply, weighted
multiply, ordered sum and final reduction widths; an optional temporary-width
check rejects a uniformly narrow lowering of an otherwise adequate sum.

The #23/#27 follow-up now centralizes host timing eligibility through variance,
review and ledger admission, binds reviews to both capture digests and rereads the
baseline at final admission. Synthetic provenance now keeps manifest/capture
commits consistent. See [timing admission](timing-admission.md); these integration
fixtures await combined validation. The HIP test executable compiled at
`ee133a6`, including sparse stride-domain fixtures, without executing tests.

The #34 device producer now has an explicit exact axis-L1 method alongside its
max-only baseline. Two-limb row/column sums and three-limb products admit the
minimum of three independently sound bounds; no default cost decision is made.
See [device bound discovery](device-bound-discovery.md). CPU/GPU differential
fixtures are authored; compilation, runtime and net-benefit measurements remain
pending.

The #38 private native WMMA queue now executes heterogeneous signed/unsigned
shapes and channel groups through a bounded persistent device cursor. Host
admission checks the entire batch for write hazards and scratch limits. Mapping
and Boost differential fixtures cover tails, group prefixes and K=65537. The
VS2026/gfx1100 HIP test executable compiles with this addition; runtime tests,
ISA/resource evidence and scheduling measurements remain pending. Public
prepared groups remain sequential; see [queue contract](persistent-native-queue.md).

The #32 [joint scheme oracle](joint-scheme-search.md) now enumerates bounded
basis/form/order/group/slab combinations with separate dot and epilogue caps,
resource/format compatibility and exact final range admission. Its authored
independent brute-force oracle includes a staged-search lost-optimum witness.
The declared objective uses supplied synthetic/calibrated costs; no measured
hardware optimum or runtime basis change is claimed. Runtime tests remain pending.

The #44/#29 encoding generator now emits source-bound executable scalar native
64-bit dot models with independent product-plane accumulators and exact division
before reduction. AWE6, base-13 and denominator-two generated fixtures compile
in the VS2026/gfx1100 test executable. Full-width Boost, exhaustive residue-pair
and K-slab fixtures await execution. This does not qualify floating instructions;
see [generated encodings](generated-encodings.md).
