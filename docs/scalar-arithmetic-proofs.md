# Scalar arithmetic proof pilot

The optional `tools/scalar_proof_session.py` workflow checks a small fixed-width
implementation layer under the existing arithmetic-certificate owner (#29).
It does not replace the integer/rational certificate checker or qualify a GPU
instruction. Ordinary certificate results explicitly leave
`fixed_width_implementation` at `not_run`; a scalar receipt is not inherited by
an unrelated representation, kernel or target.

The source pass is authored and unexecuted. No CBMC capability, proof, negative
control, native replay, timing or peak-memory result is claimed.

## Source and tool boundary

The pilot pins [CBMC 6.8.0](https://github.com/diffblue/cbmc/releases/tag/cbmc-6.8.0)
at `cdee49cb1a32c6d6703cebf6ae67161977264ad4`. It includes the real
`checked_arithmetic.hpp` and `finite_u8_reducer.hpp` headers through
`tests/proofs/scalar_arithmetic.cpp`. It does not hand-translate their operations.

CBMC's documented frontend exposes `--cpp11`; RNS8's headers use C++17 constructs,
including nested namespaces and `if constexpr`. Acceptance of these unmodified
headers is a capability gate. The runner retains preprocessing and property
discovery failures and stops before negative controls if a baseline cannot be
proved. It never erases a cast, changes a width, rewrites a header or lowers the
language contract to make an incompatible frontend succeed. See the pinned
[property inventory](https://github.com/diffblue/cbmc/blob/cdee49cb1a32c6d6703cebf6ae67161977264ad4/src/goto-programs/show_properties.cpp),
[result writer](https://github.com/diffblue/cbmc/blob/cdee49cb1a32c6d6703cebf6ae67161977264ad4/src/goto-checker/report_util.cpp),
and [CBMC CLI reference](https://diffblue.github.io/cbmc/man/cbmc.html).

The proof input is the retained complete preprocessor output. Receipts bind that
input, original and snapshot header/harness bytes, mutation policy, actual tool
executables, compiler-affecting environment, commands, ABI model, property
inventory, raw processes and result files. Entry/exit bookends reject source or
artifact drift. Every run uses a fresh output directory; old proofs are never
reused after a source change. The preprocessor, frontend, bundled SAT solver,
explicit integer model, harness/specification and result reader remain part of
the trusted computing base. Hashing them establishes identity, not translator
correctness.

## Stated domains and obligations

| Family | Complete symbolic domain | Independent specification |
| --- | --- | --- |
| Checked add/multiply | Every pair of u64 inputs, initial output value, operation and separate/left/right output alias | Exact 128-bit sum/product: success iff representable; exact successful output; failure preserves the original destination |
| Byte reduction | Every u32 input and every modulus 2 through 256; reciprocal core, dynamic dispatch and separate fixed 251/255/256 template paths | Mathematical `% q`, canonical interval, constructed reciprocal and one-correction precondition |
| Native conversion | Every u64 bit pattern and modulus 2 through 256, including the signed `INT64_MIN` pattern | Exact radix/signed bias, unsigned modulo and signed/unsigned centered representatives in `[-floor(q/2), ceil(q/2)-1]` |

Inputs are total finite-domain mappings of nondeterministic returns, with no
`assume` statements. A total byte-to-modulus map covers every admissible modulus;
the 255 input maps to 256 rather than creating an out-of-domain modulus. Concrete
ordinary replay witnesses include zero multiplication, overflow, each alias,
signed boundaries, centering thresholds and the special moduli. These witnesses
complement symbolic quantification; they are not sampled substitutes for it.

LP64 and LLP64 are separate CBMC models. The harness asserts 8-bit bytes,
32/64-bit integer widths, 64-bit pointers and the selected long width. Signed
conversions are explicitly modeled as two's complement, not claimed as a theorem
about every C++ implementation. The formal specification uses CBMC's exact
128-bit integer domains; ordinary replay uses Boost.Multiprecision independently.

Pointer/bounds, division-by-zero, signed-overflow and invalid-shift checks remain
enabled. Unwinding assertions are required with bound 258. Unsigned mask wrapping
is intentional reducer algebra, so unsigned-overflow rejection is not enabled
globally. Unknown, unchecked, unreachable, missing, duplicate or mismatched
properties, incomplete unwind, timeouts, capture limits and tool errors cannot
become `proven`. A failure of machine safety or an unwind assertion cannot count
as detection of an arithmetic mutation.

## Reproduction and negative controls

Supply selected tool paths and a Boost-capable CMake toolchain. The default
prints a source-bound plan without launching a tool or creating output:

```sh
python tools/scalar_proof_session.py --cbmc /path/to/cbmc \
  --cmake /path/to/cmake --cxx /path/to/clang++ \
  --out temp/scalar-proof-candidate --timeout 120
```

On Windows use the configured developer environment and add the explicit vcpkg
toolchain, for example
`--toolchain C:/vcpkg/scripts/buildsystems/vcpkg.cmake`. Add `--execute` to run the
plan; `--abi LP64` or `--abi LLP64` restricts the selected model. Ordinary replay
requires the current native ABI among the selected models. A proof for the other
model does not establish native execution there.

All original proofs precede the six reviewed mutations in
`metadata/scalar-proof-pilot.json`: reversed overflow predicate, writing on
failure, deleted reciprocal correction, wrong signed bias, wrong centering
threshold and wrong native radix. Each changes a unique source-bound production
operation only inside its owned ignored snapshot. Source drift requires review
of the seed. Refutations require an intended arithmetic property and a concrete
in-domain trace from the pinned JSON protocol. The driver builds ordinary
baseline/mutant replay binaries, runs the same counterexample through both, and
requires the baseline's exact Boost oracle to pass and the mutant's intended
property to fail. A crash or generic nonzero exit is not a detected mutation.
No automatic retries occur.

Reconstruct a retained session without launching tools:

```sh
python tools/review_scalar_proof.py temp/scalar-proof-candidate/scalar-proof.json
```

The reader shares source inventories, commands, proof/property admission,
replay recipes and exact-oracle result checks with the producer. It reconstructs
every verdict from raw process receipts, checks the exact production or reviewed
mutation bytes and actual retained preprocessing input, and verifies the native
project/compiler/build/executable custody. Resealing a derived summary cannot
turn a missing property or failed replay into a passing session. A shared bounded
document owner retains the report, plan, inventories and raw processes through
the final publication check; source and binary bookends remain independent.
These receipts are content custody, not cryptographic execution attestation.

The existing bounded process runner limits each command's duration and combined
stdout/stderr to 32 MiB, and the owned artifact tree to 512 MiB. Full raw outcomes
and process elapsed times remain in the ignored output directory. Peak memory is
explicitly `not_observed`; neither solver statistics nor a sampled working set
is silently presented as an exact peak. Resource measurement remains a separate
validation gap. No solver enters ordinary plan creation or an output-element
path, and no result changes backend eligibility, safe K caps or AUTO promotion.

The pilot excludes wrap64 convolution, address admission, CRT, layout/wave
ownership, GPU synchronization and compiler-generated ISA proofs. Expanding its
scope requires a separately reviewed proof obligation and source binding.
