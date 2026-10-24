# Persistent native WMMA queue

The private HIP research entry point `hip_direct_gemm_native_wmma_queue`
accepts up to 4096 independent native 64-bit matrix products. Each descriptor
selects signed or unsigned inputs, shape, pitches, the default RNS prefix,
and one, two, or four channels per matrix-engine group. It produces centered
residue planes; the caller remains responsible for the semantic range proof.
This entry point does not implement strict modulo-2^64 arithmetic.

The host builder proves the qualified 32-bit address domain and rejects every
output/output or output/input span overlap across the batch. Shared readers
are legal. It enumerates tile/channel groups round-robin across active tasks,
with linear enumeration cost after sorting the memory spans. Queue storage is
budgeted before allocation, and failed admission preserves the caller's queue.

Each 32-thread CTA atomically claims a descriptor and owns its complete K
reduction. The existing RDNA3 native grouped WMMA tile implementation is shared
with the ordinary grid launch. Each modulus has an independent accumulator;
65536-element K slabs prevent signed INT32 overflow. Output tails and partial
channel groups retain their original masks. A bounded terminal-claim count
prevents the 32-bit queue cursor from wrapping.

After admission, queues whose descriptors all share signedness and modulus group
select a kernel specialized for that pair. Shapes, pitches and prefixes may still
differ. Mixed queues retain the general dispatch kernel. This avoids reserving
registers for larger modulus groups when a uniform group-one queue executes;
the selection does not change arithmetic or the round-robin work order.

Native conversion uses a per-channel reciprocal and the remainder of `2^32`,
initialized before the K loop. Each 64-bit input is split into two unsigned
32-bit words. Reducing each word and then their radix-weighted sum needs only
32-bit reciprocal reductions: the recombination is at most 65280 for a byte
modulus. Signed inputs subtract `2^64 mod m` when negative, including `INT64_MIN`.
This avoids a large constant-division switch inside each encoded product.
The CPU reducer fixture checks every modulus from 2 through 256 against Boost;
GPU fixtures include word carries, full-width boundaries and safe K-slab splits.

The synchronous backend probes the compiled device capability before touching
outputs. Descriptor upload, capability probing, and persistent execution have
separate timing labels. `outputs_may_be_written` becomes true immediately before
the compute launch; callers must treat outputs as invalid after a subsequent
failure. The one-shot entry point allocates device descriptors per invocation
and frees them on return.

The private `NativeWmmaPreparedQueue` additionally retains admitted descriptors
and device storage across synchronous calls. Preparation performs admission,
allocation, capability probing and upload before replacing any prior prepared
state. Its preparation budget charges both old and replacement descriptor
allocations at the transactional peak. Execution resets the cursor and launches
against the retained descriptors.
Concurrent preparation/execution returns `RNS8_BUSY`; a device execution failure
poisons the object until successful re-preparation. Referenced allocations must
remain alive at the original addresses and layouts. Contents can change between
completed calls because this object caches no operand values or transformed
panels. Callers own range proofs and exclusive buffer access. This private owner
does not supply public matrix generations, reader/writer reservations or graph
publication semantics. Its differential fixture passed at `51ae8d8`, including
changed input values, transactional preparation failure and all six uniform
signedness/group combinations. All 14 native calibration modes passed their
smoke collection at that revision; sustained prepared-path measurements are
being collected separately.

The same owner now offers `execute_ordinary` as a matched scheduling control:
one ordinary grid per admitted task followed by one final synchronization,
without a queue cursor. It retains the identical preparation and allocation
footprint, including device descriptors unused by this control, plus one host
metadata descriptor per task. The scratch budget covers device allocations;
host task/work vectors are separately retained metadata. Preparation replaces
the host descriptors only after device admission/upload succeeds. Both launch
modes share exclusion and failure poisoning. This control isolates repeated
execution from per-call allocation/capability checks; it is not a claim that an
ordinary-only implementation needs queue metadata. Its added GPU coverage and
measurements await the next coordinated build.

CPU mapping, admission, collision, and budget fixtures and a Boost-based GPU
differential fixture cover heterogeneous shapes, full-width signed/unsigned
values, prefix/group tails, padding, K=65537, and varying worker counts. These
fixtures passed the local all-feature campaign at `c7f7e05`; the uniform-kernel
extension passed all six signedness/group combinations at `6d8ab25` (469 passes,
three target-inapplicable skips across the full configuration). The reciprocal
conversion passed the fresh `436d235` campaign: 472 passes and three expected
skips across 475 registered tests, with all 13 calibration modes also passing.
Public prepared
groups still execute sequentially. Public integration, asynchronous ownership,
matched prepared-path performance qualification remain open under issue #38.
The [worker sweep and resource inspection](native-queue-performance-2026-10-23.md)
record the completed scheduling experiment without claiming a win over the
best ordinary launch.
