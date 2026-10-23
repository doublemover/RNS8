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

The synchronous backend probes the compiled device capability before touching
outputs. Descriptor upload, capability probing, and persistent execution have
separate timing labels. `outputs_may_be_written` becomes true immediately before
the compute launch; callers must treat outputs as invalid after a subsequent
failure. Device descriptor storage is allocated once per invocation and freed
on return. This is not yet a reusable prepared queue.

CPU mapping, admission, collision, and budget fixtures and a Boost-based GPU
differential fixture cover heterogeneous shapes, full-width signed/unsigned
values, prefix/group tails, padding, K=65537, and varying worker counts. These
fixtures await the consolidated runtime validation campaign. Public prepared
groups still execute sequentially. Public integration, asynchronous ownership,
descriptor reuse, ISA/resource inspection, and measured scheduling benefits
remain open under issue #38; no throughput improvement is claimed.
