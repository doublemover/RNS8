# Distributed exact-product plan and CPU mock

The [experimental HIP executor](distributed-hip.md) now consumes this shared
plan/input contract for real device work. This page describes the independent
CPU mock; its observations remain mock evidence.

The private [plan](../src/backend_common/distributed_plan.hpp) and
[mock owner](../src/backend_common/distributed_mock.hpp) implement the
hardware-independent M0 slice of issue #50. They compute one exact product
through explicitly owned modulus, output-region or K-fragment work and gather
before reconstruction.
They do not launch HIP, create device contexts, allocate pinned memory, use
RCCL, or qualify a multi-GPU backend. Independent device smoke jobs remain a
different workload.

The [specification](RNS8_RESEARCH_SPEC.md#175-multi-gpu) still chooses modulus
split for production consideration: Linux ROCm, at least 1.55x for 8192-square
bounded GEMM. The mock supplies no such evidence.

## Immutable plan

Plan::create validates a copied request before replacing a shared const plan.
Selected device ordinals are explicit and may be noncontiguous. Device identity,
physical package, compute partition and memory partition are separate fields;
unknown stays unknown. Rank count is not a physical-card count. Labels are
bounded ASCII. The mock does not discover or authenticate supplied identities.

Supported semantics are bounded signed/unsigned 64-bit and exact-wide
signed/unsigned under the ordered default basis. Finite rings, fields, wrap64,
remain unsupported. Explicit M/N, 2-D output and disjoint K axes share the same
coverage verifier. Tiles supply prefix and
magnitude proofs or inherit the uniform request. Signed tiles require P>2B,
unsigned tiles P>B. Prefix zero requires a zero bound. A local shard is never
exported as a short global prefix: byte ranges explicitly map to tile and
logical basis plane.

Every work descriptor owns its rank, tile, logical modulus, complete K interval
and local byte extent. Assignment compares accumulated work against rank weights
with exact rational arithmetic. Optional relative per-plane costs carry a named
cost-profile digest. These supplied costs are planning inputs, not measured rates
or qualification. Ties choose the lower rank. Idle ranks remain in the inventory.

An exact structural identity binds shape/strides, arithmetic, input generations,
certificate/layout identities, tile proofs, rank map and expected descriptors.
Admission compares full identity and membership, not counts or truncated hashes.
Certificate digests are references, not a replacement arithmetic checker. The
mock independently checks every final integer against immutable native inputs.

## Transport and publication

Transport is explicitly local, peer, or host staging. Every used nonlocal edge
must positively support the selected direction/path. Unknown or unsupported
rejects without fallback. These supplied mock facts are not HIP peer probes.

1. Produce computes canonical residue bytes from native 64-bit input bits. Each
   multiply/add reduces modulo its own modulus; cancellation never licenses an
   overflowing machine accumulator.
2. Local completion validates epoch, generations, complete identity, descriptor,
   encoding and extent. Duplicate completion rejects.
3. A host-staged path explicitly stages into one bounded slot. That slot remains
   occupied until the destination copy completes; another request returns BUSY.
4. Transfer completion copies into private gather storage. Destination visibility
   is a separate event before proof/status readiness.
5. Every rank acknowledges complete assigned work, including explicit idle ranks.
6. Publication requires exact expected membership and all acknowledgments. It
   reconstructs each tile with its complete prefix, checks magnitude, compares
   every result to a full integer dot product, and enforces bounded export limits.

No partial export exists. Missing suffix planes are not zero. A false bound
that would reconstruct a plausible modular alias fails the full integer check.
Signed INT64_MIN and large cancelling products avoid native signed overflow.

Explicit materialization converts a published globally unique result to a
sufficient uniform prefix. It stages before replacing caller output and preserves
instance/epoch identity. Its byte limit covers new staging; retained caller
outputs and extra materializations remain caller-owned memory. This demonstrates
suffix completion in an adaptive-prefix chain, never extension from an
insufficient isolated rank.

## Epoch and resource ownership

Calls require external serialization. Input snapshots own immutable bytes and
generations. Epochs increase without wrapping. Pending, draining or quarantined
work cannot be replaced; retained result readers also block replacement.
Retain/release calls model the last consumer. A borrowed result pointer requires
retention across a possible next begin.

Cancel, deadline and injected rank failure stop new work and enter draining.
Fragment, gather, staging and input storage remain retained until every rank
positively acknowledges cleanup. Failed drain permanently quarantines that
owner. Successful cleanup permits only a fresh whole-operation epoch and
preserves progress history.

Supplied monotonic nanosecond values are logical mock observations. They test
ordering/deadline decisions, not measured performance or a real watchdog around
synchronous CPU computation. Real stream abort, lost-device cleanup and
quarantine require the shared runtime owner from #43.

Before execution, the plan checks complete address spans, descriptor limits,
aligned rank reservations and rank/host/staging budgets. Reservations cover
native replication, tile-local packed operands, local outputs, scratch and
overhead. The owner additionally charges full gather, CRT and limb-export
buffers, including both local output and gather copies.

The host model charges immutable inputs, all fragment payloads, gather/staging,
state/structural metadata, and 512 bytes per output cell for bounded Boost values
and reconstruction temporaries. Proof widths stay below 160 bits. The native
oracle stays below 192 bits because the address domain bounds K by
PTRDIFF_MAX/8 and each unsigned product is below 2^128. Structural
metadata has a separate reservation. Conservative host allowance covers
transactional replacement temporaries. Allocator overhead remains explicitly
unknown with caller headroom. This is a checked buffer model, not measured
allocator, pinned-memory, device-library or communication overhead.

## Receipt and authored coverage

The [versioned JSON schema](../tools/schemas/distributed-mock-v1.schema.json)
describes exact expected work, basis, tile proofs, rank/package mapping, phase
counts, missing work/acknowledgments, undrained ranks, failure identity and
reservations. Hardware execution and production eligibility are always false;
performance timing is null. It is not benchmark-schema evidence and cannot
install a reviewed route. The schema covers structure; runtime membership and
independent integer comparison enforce semantics.

[CPU fixtures](../tests/unit/test_distributed_plan.cpp) cover 1/2/3/4 ranks,
full-width signed/unsigned inputs, uneven tails/costs, idle ranks, random delivery,
noncontiguous devices, directional/unknown transport, malformed encoding/identity,
stale epochs, duplicate/missing work, overflow/budget rejection, phased staging,
cancel/deadline/quarantine, retained readers, cancellation arithmetic and adaptive
materialization followed by a changed-identity chain. Their classical Boost
oracle is independent of the modular producer and CRT.

The M0 fixtures ran in the earlier consolidated local suites. The subsequent
shared resource/window changes and real HIP owner have syntax checks only;
earlier passes do not qualify those changed sources.

The [HIP owner](distributed-hip.md) now implements selected devices, immutable
generation-bound native snapshots, reusable packing windows, maintained local
kernels, peer/portable-pinned gather, exact CRT/export, materialization and
retained cleanup. Its collector uses the existing artifact/request and bounded
independent-process framework. `HostStorage::limbs_only` models that runtime;
the mock requires `exact_reference` storage for its independent integer oracle.
Source readiness remains separate from native Linux/Windows execution,
topology, failure, traffic and scaling evidence.

Stable public C-ABI integration, RCCL, M/N/K distributions and multi-process or
multi-node execution are separately scoped follow-ons to the private M1/M2
modulus owner. No hardware, installation or promotion follows from this mock.

## Explicit output and contraction axes

`Axis::rows` and `Axis::columns` assign complete tile rows/columns to weighted
rank intervals; every owned region retains all its active basis planes and K.
`Axis::hybrid` is the separately named block-cyclic 2-D output grid, requiring
`rank_grid_rows*rank_grid_cols == ranks.size()`. It does not imply SUMMA, panel
broadcasts or a distributed K reduction. The grid, axis and all K intervals
participate in immutable plan identity.

`Axis::contraction` creates1..64 disjoint nonempty K intervals for each active
tile/plane. `k_fragments=0` selects the explicit rank count; a positive value is
an independent fragmentation choice. Packing capacity follows each rank's largest
fragment, while retained local outputs and owner partial-gather bytes include
every fragment. Descriptor limits and hard budgets are checked before allocation.

CPU delivery combines canonical partial residues using a widened integer sum
and modular reduction. Each fragment's exact work ID, interval and identity must
match once; arrival order cannot change the result. A missing, repeated, changed
or stale fragment cannot contribute publication. The final output bound never
bounds an unreduced partial dot. Every local dot independently uses safe modular
accumulation, and CRT runs only after complete per-channel K coverage.
