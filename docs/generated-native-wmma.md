# Bounded generated native WMMA family

`RNS8_EXPERIMENTAL_GENERATED_NATIVE_WMMA=ON` compiles six explicit research
variants: signed/unsigned native 64-bit inputs crossed with groups of one, two,
or four default-basis channels. The default is OFF and no public dispatcher
selects these symbols. Exactly one gfx1100/gfx1101/gfx1102 target is required.

The supported envelope is a 16x16x16 matrix instruction, one wave per CTA,
masked tails, native row-major inputs, planar centered-byte outputs, independent
channel accumulators, and K=65536 reduction slabs. Vector width, pipeline depth,
split-K, layout, output semantics and group specifications outside that envelope
are rejected before emitting files. The generator emits standalone kernel bodies from the same structured lowering
that the bounded logical interpreter consumes. Target/builtin mismatches
are compile errors; a scalar fallback cannot acquire a generated variant ID.

Stable IDs bind the exact specification, ordered basis, per-modulus arithmetic
certificates, generator and shared source/proof dependency hashes. Both generated
source and registry headers have separate hashes. Each variant has its own
external kernel and launch symbol; launch wrappers check full qualified address
ranges and output overlap before dispatch. A/B sharing remains legal. The
compiler's actual configuration is recorded by the existing build manifest,
including the opt-in flag. Generator metadata alone never asserts compilation.

Each ID also binds a per-modulus composed proof of native reduction, byte
fragment values, the INT32 slab plus centered carry, and the final centered byte
store. Carry/output closure establishes the same range for subsequent slabs.
The existing proof-DAG verifier checks physical widths and modular domains;
the existing planner resource query checks the empty global temporary arena.
The compatibility record names form, group, slab, layout and output boundary.
It does not certify register use, spills or instruction execution.

The shared `fragment_layout.hpp` now describes the existing RDNA3 input and
accumulator coordinates as executable binary maps. Input slots are lane/word/byte;
output slots are lane/register. Inputs deliberately have two agreeing replicas;
outputs have one owner. Both the host interpreter and the shared native WMMA
kernel use these maps. Structural admission binds wave32, instruction generation,
element widths, role and edge masking. Canonical source selection and conversion
classification distinguish register-only permutations from within-wave exchange;
this slice does not emit new shuffle or LDS conversion kernels.

The exact map-definition SHA256 is part of each generated identity and source
dependency. Changing a basis vector or its preconditions regenerates the symbols
and invalidates previous artifact identities. CPU fixtures exhaustively compare
physical slots against independently written coordinate formulas, verify replica
coverage, canonical selection, tails and altered-contract rejection. Existing
Boost GPU differentials plus a full-tile/masked-edge fixture cover the changed
indexing. These additions are authored but have not been compiled or run.
Outer pitches, non-power-of-two residue counts and global storage permutations
remain separate from this bounded local map. No CDNA mapping or measured bank
conflict model is inferred from RDNA3 coordinates.

The existing GPU differential fixture is reused for every generated variant,
including signed/unsigned full-width values, prefix/group tails, padding and
K=65535/65536/65537. Generator fixtures cover reproducibility, duplicate IDs,
unsupported specifications and source/header drift. Earlier gfx1100 variants compiled with VS2026/HIP; the changed executable-map
source requires fresh compilation and runtime qualification.
Paired fixtures additionally compare each generated route with separate native
packing and direct residue GEMM, then reverse the contraction axis of both
operands to move values across tiles and slab boundaries while preserving AB.
Both routes check the same Boost result, padded bytes and unused suffix plane.
These cross-feature fixtures are authored; runtime qualification is pending.

`check_generated_native_wmma.py` regenerates expected artifacts before inspecting
the exact manifested symbols with the existing ISA extraction and resource
parsers. Its report binds host/device object hashes, code size, disassembly
hashes, matrix opcodes, resources, spill metadata and remainder diagnostics.
Missing matrix instructions, missing/incompatible wave/LDS metadata and global
INT32 stores fail inspection. This inspection is authored but not yet executed;
it is not a substitute for the differential suite or independent timing runs.

This is a bounded first matrix-family implementation for #55. Broader generated
layouts, split-K combinations, measured Pareto pruning, held-out performance,
upstream regression qualification and production promotion remain open. No
register-use, occupancy, throughput or compile-time improvement is claimed.


## Source-bound bounded logical assurance

`native_wmma_lowering.py` now owns a small expression/loop/lifetime IR. The
emitted HIP body—not a wrapper around an independently modeled tile—takes its
operand addresses, predicates, plane selection, store ownership, K/slab loops,
zero initialization and centered modular carry from that IR. The manifest binds
the full IR, its hash and the lowering source. The existing source checker
regenerates and compares the complete manifest, source and header before either
logical or selected-symbol ISA assurance can attach.

`check_native_wmma_logic.py` compares every physical fragment's symbolic native
cell with an independently indexed scalar residue specification. It verifies
initialized logical reads (padding is poison), initialized masked zero slots,
exact K-term coverage, independent modulus domains and exactly one final writer
per output. Accumulator bounds are proved before integer-algebra simplification;
all interpreted address intermediates must fit nonnegative signed 64-bit values,
matching explicit casts in emitted expressions. Centered carry closes the
composed byte-range invariant. No numerical sampling establishes equality.

Coverage is strictly the listed configurations: interior 16³; padded 17×19×31;
ragged 1×3×17; scalar-output K=65535/65536/65537; and a 20-plane padded tail case.
Every signedness/group variant is checked independently. Reuse requires full
structural equality, identical preconditions and the explicitly recorded empty
variable renaming; equal tile sizes alone never reuse a result. Work/time/cache
counts are produced only when the checker is actually run. Unknown instructions,
out-of-scope expressions, oversized checking budgets and unsupported arithmetic
stay unresolved. No shape extrapolation is made.

Seeded mutations change both the IR and its emitted body: wrong stride, omitted
mask, uninitialized accumulator, discarded prior slab, wrong modulus, missing or
duplicated slab, and wrong/duplicate store ownership. Authored regressions require
refutation; the command-line mutation report accepts only detection or an explicit
unsupported verdict, never silent success. Existing device poison/canary,
full-width, failure-publication and paired unfused differentials remain required.

Logical and memory-model verdicts are distinct from physical-lane correctness
and target execution. Trusted assumptions name RDNA3's matrix lane algebra,
source-bound native reducer semantics, compiler behavior, initialized input
allocations and nonaliasing/no-concurrent-mutation launch ownership. Single-wave
program-order reasoning does not certify emitted barriers or AMD scheduling.
`check_generated_native_wmma.py` can attach the source-current report; the existing
promotion ledger accepts the same bound report as **diagnostic-only evidence**,
without satisfying any execution, performance or promotion gate. Ingestion
reopens/hashes the bytes and verifies source/checker identities; it explicitly
does not independently replay expensive logical checks.

Only Python syntax checks were performed for this implementation. The logical
checker, mutation suite, generated HIP compilation, AMD differentials, timing and
qualification have **not** been executed. No logical proof outcome or speedup is
claimed by checking in this machinery.
