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
