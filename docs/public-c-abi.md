# Public C ABI review

The public C ABI uses fixed uint32 C++ enums and the platform's normal
32-bit C enum representation. Global short-enum and nondefault packing flags
are unsupported. New public symbols use the rns8_ namespace. C++ wrappers
are source interfaces, outside the C binary compatibility contract.

metadata/public-c-abi.json records compiler-discovered declarations and
per-target layout candidates. It covers every installed rns8/*.h header
recursively: enum declarations/constants, complete records and members, opaque
handles, callbacks, aliases, external functions, header functions and public
integer macros. Clang's real C17 and C++17 ASTs must agree; C++ enum declarations
must explicitly have fixed uint32_t backing. Unmodeled public data exports,
anonymous members, bitfields and function-like public macros fail review
until their policy and probes are implemented.

Representation/layout coverage does not establish semantic admission. The real
CPU dispatcher controls separately exercise raw all-bits, high-bit and adjacent
invalid values against live handles, with valid-mode and unsupported-backend
controls. The current source pass extends those checks to prepared CRT options,
product prefix/execution policies and native-group dispatch policy. New controls
are authored and await the combined runtime/sanitizer batch; the remaining
external enum/flag inventory is still under review.

Grouped narrow method controls now use two complete live CPU tasks with distinct
outputs/workspaces, preserving generations, residues and reservations on raw
all-bits/high-bit/adjacent rejection. Every defined method still reaches the CPU
unsupported-provider distinction. The status formatter has a separate contract:
unknown representable values return `unknown status`; it cannot return an API
error code. Its raw-byte controls retain that behavior. These controls are
source-written and have not run in the deferred batch.

The native owner-acquire return value is another external status boundary.
Shared explicit named-status admission now rejects unknown callback values as
`INVALID_ARGUMENT` without treating them as successful acquisition or releasing
an unacquired owner. Defined failures still propagate. Whole-route candidate
admission uses the same named-status predicate; the status formatter retains
its independent `unknown status` fallback. No public declaration/layout changed.

The registry's Windows x86_64 LLP64 entry is a compiler-front-end candidate,
not a native-consumer qualification result. Linux LP64 and other supported
architectures need their own deliberately reviewed native-target entries;
one platform's offsets are never used as another platform's baseline.

## Updating and checking source

Use a fresh ignored output directory for each invocation. On Windows, invoke
through tools/windows_dev.py to load the installed Visual Studio environment.
The compiler may be supplied explicitly when it is not on PATH.

    python tools/windows_dev.py python tools/check_public_abi.py author --clang C:/TheRock/build/lib/llvm/bin/clang.exe --output temp/abi-author
    python tools/windows_dev.py python tools/check_public_abi.py source --clang C:/TheRock/build/lib/llvm/bin/clang.exe --output temp/abi-source

Authoring writes the declaration registry and the generated compiled inventory.
Review both diffs. It does not accept an ABI change on behalf of a reviewer or
run native consumers. The source check rejects declaration/layout drift or a
stale generated inventory. Header hashes identify actual receipt contents;
a comment-only edit does not change the declaration digest.

Generated type checks and native import references use stable declaration-based
names, with unambiguous length-prefixed record/member components. Adding one
declaration no longer renumbers unrelated checks. Public metrics, declaration
digests and native type/signature assertions remain the compatibility boundary;
these private generator names carry no ABI authority.

The compiled inventory uses actual sizeof/alignof/offsetof expressions,
member/callback/function type assertions and constants. The allocation-free
rns8_get_public_abi_info API returns a borrowed immutable table plus its
declaration digest, compiler/runtime identity and the actual library's embedded
build_manifest_json (null when provenance was not compiled in). Its caller must provide a
readable eight-byte size prefix; success requires exactly the current full
struct size and ABI version. Rejection preserves output bytes. Table/string
pointers remain valid only until the library is unloaded.

The shared header admission helper reads the mandatory size prefix before the version or any
payload field, and rejects misaligned records before typed access. A caller
must supply readable storage for that prefix (the declared four- or eight-byte
`struct_size`) and for every byte its accepted size promises. Extensible
records retain their existing policy: sizes at least the current record are
accepted, only current fields are accessed, and extension bytes remain untouched.
Exact-size contracts, including native-device, pinned-host, locality, runtime
device identity, public ABI and narrow summary records, reject larger sizes.
No API can discover that a caller lied about the actual backing allocation.
Raw header-prefix and canary controls are authored for the deferred sanitizer
batch; frontend compilation does not qualify those runtime checks.

Context and GEMM plan admission share the explicit backend-value validator.
Unknown backend representations return `INVALID_ARGUMENT` before allocating a
context or constructing a plan; a known but unavailable backend remains a
separate unsupported-backend case. Invalid backend support queries preserve
both the result storage and the caller's required-count value. Raw-byte controls
cover all-bits, high-bit and adjacent invalid backend, semantics, bounds, layout
and flag values through live CPU owners, alongside valid controls. This extends
the behavioral corpus without claiming every external enum/flag is covered.

The analytical pipeline query shares operation tag admission with the public
provider and preserves count/candidate outputs on rejection. Explicit whole-route
selection/preparation reject unknown status/reason/resource tags and inconsistent
cost-origin/Boolean metadata, including malformed later candidates after an
eligible earlier entry. Known unsuccessful statuses remain ineligible rather
than malformed. Raw status and count-preservation controls are authored; native
runtime and sanitizer validation remains deferred.

Context, plan and matrix constructors preserve caller handle slots when options
or descriptor preflight rejects input. After successful preflight they initialize
the slot to null before checking context health or acquiring backend resources.
Malformed plan/matrix descriptors therefore remain `INVALID_ARGUMENT` even on a
poisoned context; valid descriptors still report `BACKEND_FAILURE` there. GEMM
semantic and bound tags are admitted before flag-specific backend routing, so an
unsupported accelerator flag cannot mask an unknown tag. Authored canary,
short-size, wrong-version and poisoned-context controls await the final batch.

Sparse-A descriptor admission also validates bound tags explicitly, including
bounded semantics where a nonzero unknown tag previously passed the sparse
layout query. Malformed sparse descriptors preserve count/packed/dense outputs
and constructor handle slots before context health checks. Result-cache
construction admits reserved bits and the complete explicit flag contract before
CPU-provider rejection; FP8 preparation preserves handle slots on malformed
header/method/flag/scale preflight. Their valid unsupported-provider outcomes
remain distinct. The new real-entrypoint controls have not run yet.

Resident graph construction admits every node header/kind/complex-method tag
before allocating graph owners or preparing native inputs; malformed tags in a
later node preserve the graph handle slot. Prepack construction similarly rejects
unknown operand roles before context health and handle publication. Authored live
CPU controls also exercise resident/operand role queries and narrow, adaptive,
bilinear and complex arithmetic method rejection, checking output generations,
residue bytes, ownership counters and device traffic. Execution remains deferred.

Additional authored controls exercise raw streaming schedules through the public
resource query and raw locality orders through a live CPU context/plan. Known
locality modes still reach the distinct unsupported CPU-backend outcome.
Malformed prepared-Gram method/orientation/policy tags reject before owner
allocation and preserve the owner-output canary and matrix generation. These
source additions remain unexecuted until the combined validation batch.

## Installed native checks

Run once for each separately installed static and shared package. The checker
finds tools from the active environment; explicit --cc, --cxx and --cmake paths
override discovery. PE export inspection needs --llvm-readobj; ELF export
inspection needs --llvm-nm. No source/build-tree library fallback is allowed.

    python tools/windows_dev.py python tools/check_public_abi.py installed --prefix C:/RNS8/install-shared --linkage shared --clang C:/TheRock/build/lib/llvm/bin/clang.exe --llvm-readobj C:/TheRock/build/lib/llvm/bin/llvm-readobj.exe --output temp/abi-shared
    python tools/windows_dev.py python tools/check_public_abi.py installed --prefix C:/RNS8/install-static --linkage static --clang C:/TheRock/build/lib/llvm/bin/clang.exe --output temp/abi-static

Each run compiles, links and executes C17 and C++17 consumers using absolute
installed-header paths and the installed imported target. Consumers compare
every layout/constant metric and the declaration digest with both their own
compiler's results and the actual linked library's table. Volatile references
retain every external C import even under dead stripping. Shared exports must
match the complete symbol inventory; static packages have independent link
checks. Receipts bind headers, library, compiler executables, actual compilation
commands, CMake compiler definitions, CRT identity and all probe outputs.

Disposable C/C++ negative consumers alter packing, change an enum value, change
a field to a different type of the same size and request a missing export.
They must fail with relevant assertion/linker diagnostics. They are never run.
Unsupported packing/enum flags are also rejected from actual compile commands
and inherited cache flags.

Installed consumer configuration preserves the C/C++ driver invocation names,
including Unix `clang++` symlinks. Resolving that name to `clang` before CMake
selection can change default C++ linking. The native compiler profile records
both the invocation path and resolved binary/hash; layout and linkage checks
still use the actual installed package. The driver-symlink regression is
source-authored and awaits the deferred validation batch.

The native smoke exercises the CPU context/matrix owner/deleter boundary and
native-device contract query without accessing GPU memory. Its nine raw enum
cases and descriptor size/version controls do **not** cover every external
enum/flag validator. The receipt explicitly keeps
all_external_enum_paths_covered false. Full API behavior and GPU execution
remain separate validation requirements.

Incremental GEMM validates the dirty-region array/count, size/version, reserved
flags and rectangles before lazy input conversion, exact range proofs or cache
diagnostic updates. Both RNS and finite-ring entrypoints share this admission;
execution reuses its full-output result instead of rescanning the array.
Malformed requests preserve matrices, reservations and cache diagnostics. CPU
host-preflight controls retain a genuine unsupported-cache distinction; the
separate direct-HIP fixture uses a real created cache and exact recomputation.
These added controls are source-authored and await the combined validation batch.

## Optional Linux ELF types

Native Linux shared checks can additionally supply --abidw, --abidiff,
--abigail-version and --elf-baseline together. Both executables must match the
explicit complete release pin. The baseline JSON uses schema
rns8_linux_elf_abi_baseline_v1 with tuple, compiler_model, release_pin,
declaration_sha256, binary_sha256, a relative xml path and xml_sha256. compiler_model records
compiler_id, runtime_model, packing_policy and enum_policy, matching the
candidate's library and native checker receipt. Retain the reviewed baseline
binary's content identity alongside this evidence.

Generate XML from the intended baseline with abidw --no-show-locs. Retain full
debug type information. The adapter requires complete public record members,
enum values and function type references in both XML corpora before invoking
abidiff. Symbol-only output, unresolved types, cross-target/configuration
baselines and any nonzero comparison result are rejected.
See the upstream [abidw](https://sourceware.org/libabigail/manual/abidw.html)
and [abidiff](https://sourceware.org/libabigail/manual/abidiff.html) manuals.
This adapter does not inspect Windows PDBs or qualify GPU kernel ABIs.

## Version policy

Compatible additions can extend the API without changing existing layouts.
Changing an existing record, constant, calling convention or function type
requires an explicit compatibility decision. For an intentional breaking
greenfield change, bump the public ABI version and rebuild consumers before
authoring/reviewing new baselines. A fresh baseline is evidence to review,
never permission to disguise an accidental ABI break.
