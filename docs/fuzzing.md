# CPU fuzz discovery and arithmetic corpus

`RNS8_BUILD_FUZZERS=ON` creates `rns8_fuzz_library`, a separate static copy of
the configured first-party CPU sources. Its objects receive libFuzzer feedback
and ASan; native Unix Clang also receives UBSan with nonrecovering checks.
Fuzz entrypoints alone link the fuzzer runtime. Ordinary libraries and installed
packages do not acquire fuzz instrumentation. HIP must be OFF in this build;
GPU replay is separate from discovery. Windows uses the existing
`windows-clang-asan-debug` preset and static CRT. The configured source inventory
is `rns8-fuzz-instrumentation-<CONFIG>.json`; this is not execution evidence.

The instrumentation control authors a real branch-containing library object,
checks actual compile commands and inspects the archive with the matching
`llvm-nm`. Missing native tools produce an explicit unavailable control.
Third-party dependencies are outside the library instrumentation claim.
See LLVM's [separate fuzzer instrumentation](https://llvm.org/docs/LibFuzzer.html#fuzzer-usage).

`rns8_fuzz_plan_descriptor` constructs known-valid plans in all six domains,
checks explicit backend/schedule/workspace admission, then applies one invalid
descriptor mutation and requires its specific rejection/null-output contract.
The existing live plan must remain queryable. No arbitrary dangling handles
are supplied. `rns8_fuzz_json_dependency` is explicitly a third-party JSON
parser/access smoke; it does not claim to exercise RNS8 metadata validation or
link the instrumented RNS8 library. The former metadata harness name is retired.

`rns8_fuzz_export_contract` consumes arithmetic-case-v1 bytes. Its primary
oracle uses scalar Boost integers without production reduction, CRT, prefix
selection or cached results. Finite results use mathematical modulo q;
wrap64 uses modulo 2^64. Signed/unsigned bounded values must fit their declared
destination and prefix. Wide cases must fit a conservative input/accumulation
proof within their explicit prefix, including each operand's independent
signed/unsigned packing range even when the product is zero, and use the explicit exact-value-bound
plan contract; three exported limbs include required
sign extension. Out-of-envelope inputs are not counted as valid arithmetic
executions. Wrong successful values, unexpected statuses, modified padding,
invalid-LD writes and stale prepared execution cause a crash/reproducer.

The binary format is little-endian and contains no pointers or implicit RNG:

| Bytes | Contract |
|---|---|
| 0–3 | ASCII `R8F1` |
| 4 | 0 bounded i64; 1 bounded u64; 2 finite ring u8; 3 wrap64; 4 signed wide; 5 unsigned wide |
| 5 | Bits: 1 repeat GEMM; 2 invalid-LD/no-write control; 4 prepared/repack/stale rejection/refresh. Prepared bit is RNS-only. |
| 6–7 | Literal M and N, each 1–8 |
| 8–11 | Literal K, 1–65537 |
| 12–14 | A/B/output row padding, each 0–7; layout is explicitly row-major |
| 15 | RNS prefix 1–20; zero for finite/wrap |
| 16–17 | Finite modulus 2–256; zero for other semantics |
| 18–23 | Reserved zero bytes |
| 24 onward | M×K A values then K×N B values, each uint64 little-endian; signed domains use two's-complement bits; finite values must fit u8 |

The decoder checks the full byte count, 262144 input-value/operation budgets
and semantic fields before allocation. Maximum input is 2097176 bytes.
Source buffers and output use minimal final-row extents plus guards.

`python tools/fuzz_cases.py --out temp/fuzz-corpus/arithmetic` expands named
seeds for all six domains, every supported prefix, every finite byte modulus,
native boundaries, cancellation and padded rectangular storage. Add `--large-k`
for skinny K=65535/65536/65537 cases. Raw corpus files remain ignored; a sibling
manifest records names, sizes and hashes without becoming a fuzzer input.
The Python integer oracle and codec are independent cross-language controls.

`python tools/fuzz_metadata.py --out temp/metadata-properties` plans a separate
property session against the actual production benchmark capture validator.
Add `--execute` to run its bounded child, `--cases`/`--seconds` to select limits,
or repeat `--fixture PATH` to choose an explicit fixture inventory. Every
selected fixture receives a positive control and a mutation share. Values are
replaced throughout the JSON tree, including wrong types and oversized
integers. The driver checks both diagnostic and throwing APIs, deterministic
errors and absence of input mutation. Only documented parser/schema rejection
is expected; unexpected exceptions retain exact reproducer bytes and fail.
Limits on bytes, nesting, nodes, child time/output and owned artifacts bound
the session. Source/interpreter/fixture hashes and compact case verdicts remain
in ignored output. Reaching a case limit reports bounded completion, not full
mutation coverage. No GPU or performance credit follows from this result.

`tools/fuzz_shrink.py` provides deterministic semantic minimization to an
explicit replay adapter. It reproduces the original failure first, then removes
operation bits, matching logical matrix/K coordinates and value intervals before
reducing individual value alphabets. Every retained candidate is wire-valid,
independently range-admitted and reproduces the same bounded failure signature.
Changed failure categories, unsupported/rejected cases and adapter exceptions
cannot become reductions. Semantic domain, prefix, modulus and padding stay
fixed. The original and minimized bytes can be reproduced with the existing
codec; the summary retains their hashes and every actual callback verdict.
Replay count and wall deadline are bounded; the callback must separately bound
its own native process. Exhaustion retains the best reproducer and does not
claim minimality. A completed pass is a local fixed point, not a global minimum.
This component launches nothing and does not attest native execution; wiring
the bounded session adapter and qualifying native replay remain open.
Authored minimizer controls use synthetic callbacks only and remain unexecuted.

## Deterministic native replay

`RNS8_BUILD_TOOLS=ON` also builds `rns8-arithmetic-replay` against the ordinary
static library, independently of libFuzzer and discovery instrumentation. It
consumes one unchanged arithmetic-case-v1 file per process:

```sh
rns8-arithmetic-replay --case temp/fuzz-corpus/arithmetic/CASE.bin --backend hip-direct --device 0 --run-id replay-001
```

Backend and device are mandatory. Backend names are `cpu-reference`,
`hip-direct`, `hipblaslt`, `ck`, `rocwmma`, `wrap64-byte-limb`,
`hip-vector-alu-int64`, `amdgpu-builtins` and `cpu-amx`; AUTO is excluded.
CPU reference/AMX require device -1; GPU paths require a nonnegative ordinal.
The separate wrap backend selects CPU or GPU explicitly with that ordinal.
Selected plan backend/kernel metadata is observed before packing; it is not
an executed-symbol ISA receipt. `--build-identity` uses the existing no-GPU
compiled executable/library identity protocol.

The fuzzer and native tool share the actual replay implementation and independent
Boost oracle. Repeated GEMM, invalid-LD/no-write, repack/stale-prepared rejection
and refreshed execution retain the same case bits. Each successful export
compares every logical word/limb, sign extension, padding and guards. Reports
include input digest, run/process identity, both compiled manifests, plan/device
observations, actual GEMM/comparison counts and first status/value/storage failure.
They carry no production latency or promotion eligibility.

Malformed/out-of-envelope input exits65 without arithmetic; unavailable context
or plan returns a typed unsupported report and exit77. Unexpected execution,
export or cleanup status is a failure, including unsupported after work admission.
On a checked failure the one-case child emits its first structured result and
exits without unwinding public owners or issuing further library calls. This
preserves the first failure and avoids unsafe cleanup after uncertain device
work; it does not demonstrate device recovery. A parent must bound/retain each
child and stop on unexplained failures. No GPU fault campaign is enabled.

CPU shared-replay controls cover all six domains and lifecycle counts; seeded
wrong-result/status/no-write controls exercise the same assertion boundary.
These controls and native CPU/GPU replay remain unexecuted during the source pass.

`tools/fuzz_replay.py` supplies the bounded parent session. Explicit `--case`
inputs (1–64 distinct byte identities), backend, device, executable and output
directory are required. It plans by default, reading only bounded case files;
it does not open the executable, create output or launch a child in plan mode.
Add `--execute` to query compiled identity and run one owned child per case,
bounded by `--seconds` (default300, maximum3600), `--case-seconds` (default60),
8MiB child output and 256MiB owned artifacts. The existing session runner owns
child termination and interrupted capture. Output must be a fresh directory.

```sh
python tools/fuzz_replay.py --exe build/rns8-arithmetic-replay --case temp/case.bin --backend hip-direct --device 0 --out temp/replay-plan
```

The parent rechecks actual executable/adapter/case bytes, retains immutable case
copies and raw process/native reports, validates executable/library manifests
against the no-GPU preflight, and requires exact run/PID/work/backend/count
bindings. Every original and retained case is bookended through session end.
A native failure, timeout, interruption, input drift or malformed report stops
further cases and preserves an incomplete receipt. An all-unsupported selection
returns77 rather than success. Passed counts require complete output checks and
successful public cleanup; they remain diagnostic evidence, not hardware
qualification or performance promotion. Semantic failure signatures use the
existing shrinker's verdict vocabulary, excluding movable cell coordinates.
Automatic shrinking/retry after a failed GPU child remains disabled; the caller
must separately authorize and bound any subsequent reproducer campaign.

`tools/fuzz_replay_shrink.py` is that separate, explicit command. Supply the same
backend/device/executable, original `--case`, fresh `--out`, and four-token
`--signature KIND PHASE ACTUAL EXPECTED`. It plans by default. Execution requires
`--execute`; a nonnegative device also requires `--allow-gpu`. It first reproduces
the exact supplied signature through a fresh ordinary replay session before
trying any reduction. A supplied signature alone is never evidence.

The existing semantic minimizer controls operation/dimension/value reductions;
each candidate uses the same full native replay, manifest, case and process
checks. Defaults are32 replays and300seconds; ceilings are128 replays,3600seconds
and256MiB across all nested sessions. The parent shares the outer wall deadline
and artifact root, freezes the executable, and retains original/minimized bytes,
every native session receipt and the shrinker's exact attempts. Device/backend
failures, cancellation, timeouts, identity and cleanup defects stop the adapter;
only value/padding/no-write and the explicitly admitted wrong-status classes can
be minimized. A changed original signature stops before reductions. Other
semantic failure categories cannot be accepted as reductions. Result status
distinguishes a bounded local fixed point from budget exhaustion; global
minimality and hardware qualification are not established. All native replay and
minimization execution remains deferred; authored controls mock the child owner.

`tools/fuzz_discovery.py` owns bounded CPU discovery against an already built,
isolated fuzz configuration. It plans without inspecting binaries or creating
output; add `--execute` to inspect and run. Supply `--build`, matching `--nm`
and `--symbolizer`, and a fresh `--out temp/NAME`. The default aggregate deadline
is600seconds, with1000discovery runs or20seconds per target, 2048MiB RSS,
256MiB per allocation, 512MiB retained artifacts and10seconds per input.
The parent stops on the first failed child, deadline, interrupted capture,
missing feedback report or input/artifact drift. It never retries or minimizes
a crash automatically. [LLVM's file-operand replay](https://llvm.org/docs/LibFuzzer.html#options)
executes every named seed before directory-based discovery.

The object denominator comes from the real exported compilation database,
registered first-party targets and generated fuzz-library source inventory.
Every configured production fuzz TU needs exactly one object with the selected
compiler, static CRT on Windows and full feedback/sanitizer flags. Each actual
object is inspected with matching `llvm-nm`; missing feedback sites remain
listed explicitly, including TUs without instrumentable branches. At least one
production object must contain feedback sites. These observations establish
configured flags and object sites; aggregate libFuzzer edge/feature counts do
not attribute executed branches to individual library functions. External
dependencies and HIP code are outside this denominator. The JSON target remains
honestly classified as third-party parser reachability.

The manual [CPU fuzz candidate workflow](../.github/workflows/cpu-fuzz-candidate.yml)
verifies an already prepared managed LLVM archive against the shared lock,
builds an isolated HIP-OFF library and all three entrypoints, then retains the
object denominator, every seed replay, bounded discovery and first failure.
It has not run. Required PR activation follows qualification of sensitivity and
cost in the deferred batch; no new push-triggered discovery campaign is enabled.

The named corpus also includes row/column and common-K permutations, legal zero
K padding, physical-padding changes, zero/identity operands, independently
range-admitted K split partials and sufficient-prefix changes for all six
arithmetic domains. Every legal combination of repeat/no-write/prepared-refresh
bits is represented on a nontrivial rectangle; finite/wrap cases exclude the
RNS-only prepared bit. Each derived input is compared with its own scalar oracle.
Authored Python controls separately check the mapped mathematical relations and
ring-correct split recombination. This is an explicit bounded interaction set,
not exhaustive API-history exploration or support for other layouts.

The [native production mutation registry](native-production-mutations.md) binds
real signed CRT, export rejection-status and prepared generation mutations to
three shared-replay assertion cases. An unchanged baseline must pass before
mutants execute in disposable copies; only the intended public-route assertion
failure counts as detection. These new witnesses remain unexecuted.

Implementation remains in progress under #101: source-attributed library branch
witnesses and broader reference-state lifecycle histories remain. Native replay,
minimizer, object inspection and discovery qualification are deferred.
